from jetgen.base.networks import select_base_generator
from jetgen.models.funcs  import default_model_init

from .iddpm import IDDPMUNet

def select_generator(name, **kwargs):
    # pylint: disable=too-many-return-statements

    if name == 'iddpm-unet':
        return IDDPMUNet(**kwargs)

    raise ValueError(f"Unknown diffusion generator: '{name}'")

def construct_diffusion_generator(
    model_config, input_shape, output_shape, n_steps, device
):
    model = select_generator(
        model_config.model,
        input_shape  = input_shape,
        output_shape = output_shape,
        n_steps      = n_steps,
        **model_config.model_args
    )

    return default_model_init(model, model_config, device)

