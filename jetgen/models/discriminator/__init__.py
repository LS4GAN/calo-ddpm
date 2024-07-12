from jetgen.base.networks import select_base_discriminator
from jetgen.models.funcs  import default_model_init

from .dcgan import DCGANDiscriminator

DISC_DICT = {
    'dcgan' : DCGANDiscriminator,
}

def select_discriminator(name, **kwargs):
    if name in DISC_DICT:
        return DISC_DICT[name](**kwargs)

    return select_base_discriminator(name, **kwargs)

def construct_discriminator(model_config, image_shape, device):
    model = select_discriminator(
        model_config.model, image_shape = image_shape,
        **model_config.model_args
    )

    return default_model_init(model, model_config, device)

