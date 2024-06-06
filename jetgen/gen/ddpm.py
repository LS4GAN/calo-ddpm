# pylint: disable=not-callable
import torch

from jetgen.torch.select         import select_optimizer, select_loss
from jetgen.torch.data_norm      import select_data_normalization
from jetgen.models.diffusion_gen import construct_diffusion_generator

from jetgen.diffusion      import select_diffusion_process
from jetgen.diffusion.ddpm import generate_variance_schedule, DDPM
from jetgen.torch.funcs    import update_average_model

from .funcs import set_two_domain_input
from .model_base import ModelBase
from .named_dict import NamedDict

class DDPMModel(ModelBase):
    # pylint: disable=too-many-instance-attributes

    def _setup_images(self, _config):
        images = [ 'real_a', 'real_b', 'fake_b', ]

        return NamedDict(*images)

    def _setup_models(self, config):
        models = {}
        shape  = config.data.datasets[0].shape

        models['gen'] = construct_diffusion_generator(
            config.generator, shape, shape, len(self.vsched), self.device
        )

        if self.avg_momentum is not None:
            models['avg_gen'] = construct_diffusion_generator(
                config.generator, shape, shape, len(self.vsched), self.device
            )
            models['avg_gen'].load_state_dict(models['gen'].state_dict())

        return NamedDict(**models)

    def _setup_losses(self, config):
        return NamedDict('simple')

    def _setup_optimizers(self, config):
        return NamedDict(
            gen = select_optimizer(
                self.models.gen.parameters(), config.generator.optimizer
            )
        )

    def __init__(
        self, savedir, config, is_train, device, vsched,
        avg_momentum = None,
        data_norm    = None,
        seed         = 0,
    ):
        # pylint: disable=too-many-arguments
        assert len(config.data.datasets) == 2, \
            "DDPM expects a pair of datasets: [ noise, data ]"

        self.prg = torch.Generator(device)
        self.prg.manual_seed(seed)

        self.vsched = generate_variance_schedule(vsched)
        self.dp     = DDPM(self.vsched, device, seed)
        self.seed   = seed

        self.data_norm = select_data_normalization(data_norm)
        self.loss_fn   = select_loss(config.loss).to(device)

        self.avg_momentum = avg_momentum

        super().__init__(savedir, config, is_train, device)

    def _set_input(self, inputs, domain):
        set_two_domain_input(self.images, inputs, domain, self.device)

        if (self.data_norm is not None) and (self.images.real_b is not None):
            with torch.no_grad():
                self.images.real_b \
                    = self.data_norm.normalize(self.images.real_b)

    def reseed(self, seed = None):
        if seed is None:
            return

        self.dp.prg.manual_seed(seed)

    @torch.no_grad()
    def diffuse(self, dp = None, subsample = None):
        if self.images.real_a is None:
            return

        if dp is not None:
            dp = { 'name' : dp, 'prg' : self.dp.prg, 'vsched' : self.vsched }
            dp = select_diffusion_process(dp, self.device)
        else:
            dp = self.dp

        if subsample is not None:
            dp = dp.subsample(subsample)

        self.images.fake_b = dp.marginal_variance().sqrt() * self.images.real_a

        timesteps = list(reversed(range(1, len(dp)+1)))
        n         = len(self.images.fake_b)

        if self.avg_momentum is not None:
            gen = self.models.avg_gen
        else:
            gen = self.models.gen

        for time in timesteps:
            t = time * torch.ones(n, device = self.device, dtype = torch.long)
            eps = gen(self.images.fake_b, dp.map_time(t))

            self.images.fake_b = dp.backward_step_given_eps(
                t, self.images.fake_b, eps
            )

    def forward(self):
        self.diffuse()

    def optimization_step(self):
        self.optimizers.gen.zero_grad(set_to_none = True)

        n = len(self.images.real_b)
        t = torch.randint(
            low = 1, high = len(self.dp) + 1, size = (n, ),
            generator = self.prg,
            device    = self.device
        )

        xt, eps  = self.dp.forward_jump(t, self.images.real_b)
        pred_eps = self.models.gen(xt, t)

        self.losses.simple = self.loss_fn(pred_eps, eps)
        self.losses.simple.backward()

        self.optimizers.gen.step()

        if self.avg_momentum is not None:
            update_average_model(
                self.models.avg_gen, self.models.gen, self.avg_momentum
            )

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

