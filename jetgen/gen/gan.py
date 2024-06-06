# pylint: disable=not-callable
# NOTE: Mistaken lint:
# E1102: self.criterion_gan is not callable (not-callable)

import torch

from jetgen.torch.select            import select_optimizer
from jetgen.torch.queue             import FastQueue
from jetgen.torch.funcs             import prepare_model, update_average_model
from jetgen.torch.layers.batch_head import BatchHeadWrapper, get_batch_head
from jetgen.base.losses             import GANLoss
from jetgen.torch.gradient_penalty  import GradientPenalty
from jetgen.torch.data_norm         import select_data_normalization
from jetgen.models.discriminator    import construct_discriminator
from jetgen.models.generator        import construct_generator

from .funcs import set_two_domain_input
from .model_base import ModelBase
from .named_dict import NamedDict

def queued_forward(batch_head_model, input_image, queue, update_queue = True):
    output, pred_body = batch_head_model.forward(
        input_image, extra_bodies = queue.query(), return_body = True
    )

    if update_queue:
        queue.push(pred_body)

    return output

class GAN(ModelBase):
    # pylint: disable=too-many-instance-attributes

    def _setup_images(self, _config):
        images = [ 'real_a', 'real_b', 'fake_b', ]

        return NamedDict(*images)

    def _construct_batch_head_disc(self, model_config, input_shape):
        disc_body = construct_discriminator(
            model_config, input_shape, self.device
        )

        disc_head = get_batch_head(self.head_config)
        disc_head = prepare_model(disc_head, self.device)

        return BatchHeadWrapper(disc_body, disc_head)

    def _setup_models(self, config):
        models = {}

        shape_0 = config.data.datasets[0].shape
        shape_1 = config.data.datasets[1].shape

        models['gen_ab'] = construct_generator(
            config.generator, shape_0, shape_1, self.device
        )

        if self.avg_momentum is not None:
            models['avg_gen_ab'] = construct_generator(
                config.generator, shape_0, shape_1, self.device
            )
            models['avg_gen_ab'].load_state_dict(models['gen_ab'].state_dict())

        if self.is_train:
            models['disc_b'] = self._construct_batch_head_disc(
                config.discriminator, shape_1
            )

        return NamedDict(**models)

    def _setup_losses(self, config):
        losses = [ 'gen_ab', 'disc', ]

        if self.is_train and config.gradient_penalty is not None:
            losses += [ 'gp', ]

        return NamedDict(*losses)

    def _setup_optimizers(self, config):
        optimizers = NamedDict('gen_ab', 'disc')

        optimizers.gen_ab = select_optimizer(
            self.models.gen_ab.parameters(), config.generator.optimizer
        )

        optimizers.disc = select_optimizer(
            self.models.disc_b.parameters(), config.discriminator.optimizer
        )

        return optimizers

    def __init__(
        self, savedir, config, is_train, device,
        head_config     = None,
        head_queue_size = 3,
        avg_momentum    = None,
        data_norm       = None,
    ):
        # pylint: disable=too-many-arguments
        self.avg_momentum   = avg_momentum
        self.head_config    = head_config or {}
        self.data_norm      = select_data_normalization(data_norm)

        assert len(config.data.datasets) == 2, \
            "NocycleGAN expects a pair of datasets"

        super().__init__(savedir, config, is_train, device)

        self.criterion_gan    = GANLoss(config.loss).to(self.device)
        self.gradient_penalty = config.gradient_penalty

        if self.is_train:
            self.queues = NamedDict(**{
                name : FastQueue(head_queue_size, device = device)
                    for name in [ 'real_b', 'fake_b' ]
            })

            self.gp = None

            if config.gradient_penalty is not None:
                self.gp = GradientPenalty(**config.gradient_penalty)

    def _set_input(self, inputs, domain):
        set_two_domain_input(self.images, inputs, domain, self.device)

        if (self.data_norm is not None) and (self.images.real_b is not None):
            with torch.no_grad():
                self.images.real_b \
                    = self.data_norm.normalize(self.images.real_b)

    def forward_dispatch(self, call_type):
        if call_type == 'fwd':
            self.images.fake_b = self.models.gen_ab(self.images.real_a)

        elif call_type == 'avg-fwd':
            self.images.fake_b = self.models.avg_gen_ab(self.images.real_a)

        else:
            raise ValueError(f"Unknown forward direction: '{call_type}'")

    def forward(self):
        if self.images.real_a is not None:
            if self.avg_momentum is not None:
                self.forward_dispatch(call_type = 'avg-fwd')
            else:
                self.forward_dispatch(call_type = 'fwd')

    def backward_gen(self):
        # NOTE: Queue is updated in discriminator backprop
        disc_pred_fake_b = queued_forward(
            self.models.disc_b, self.images.fake_b, self.queues.fake_b,
            update_queue = False
        )

        self.losses.gen_ab = self.criterion_gan(disc_pred_fake_b, True)
        self.losses.gen_ab.backward()

    def backward_discriminator_base(
        self, model, real, fake, queue_real, queue_fake
    ):
        # pylint: disable=too-many-arguments
        loss_gp = None

        if self.gp is not None:
            loss_gp = self.gp(
                model, fake, real,
                model_kwargs_fake = { 'extra_bodies' : queue_fake.query() },
                model_kwargs_real = { 'extra_bodies' : queue_real.query() },
            )
            loss_gp.backward()

        pred_real = queued_forward(
            model, real, queue_real, update_queue = True
        )
        loss_real = self.criterion_gan(pred_real, True)

        pred_fake = queued_forward(
            model, fake, queue_fake, update_queue = True
        )
        loss_fake = self.criterion_gan(pred_fake, False)

        loss = (loss_real + loss_fake) * 0.5
        loss.backward()

        return (loss_gp, loss)

    def backward_discriminator(self):
        fake_b = self.images.fake_b.detach()

        loss_gp, self.losses.disc \
            = self.backward_discriminator_base(
                self.models.disc_b, self.images.real_b, fake_b,
                self.queues.real_b, self.queues.fake_b
            )

        if self.gp is not None:
            self.losses.gp = loss_gp

    def optimization_step_gen(self):
        self.set_requires_grad([self.models.disc_b], False)
        self.optimizers.gen_ab.zero_grad(set_to_none = True)

        self.forward_dispatch('fwd')
        self.backward_gen()

        self.optimizers.gen_ab.step()

    def optimization_step_disc(self):
        self.set_requires_grad([self.models.disc_b], True)
        self.optimizers.disc.zero_grad(set_to_none = True)

        self.backward_discriminator()

        self.optimizers.disc.step()

    def _accumulate_averages(self):
        update_average_model(
            self.models.avg_gen_ab, self.models.gen_ab, self.avg_momentum
        )

    def optimization_step(self):
        self.optimization_step_gen()
        self.optimization_step_disc()

        if self.avg_momentum is not None:
            self._accumulate_averages()

    @torch.no_grad()
    def get_images(self):
        if self.data_norm is None:
            return self.images

        result = {
            k : v.detach() for (k, v) in self.images.items() if v is not None
        }

        if 'real_b' in result:
            result['real_b'] = self.data_norm.denormalize(result['real_b'])

        if 'fake_b' in result:
            result['fake_b'] = self.data_norm.denormalize(result['fake_b'])

        return result

