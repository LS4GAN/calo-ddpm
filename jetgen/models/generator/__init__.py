from jetgen.base.networks    import select_base_generator
from jetgen.models.funcs     import default_model_init


def select_generator(name, **kwargs):

    input_shape  = kwargs.pop('input_shape')
    output_shape = kwargs.pop('output_shape')

    assert input_shape == output_shape
    return select_base_generator(name, image_shape = input_shape, **kwargs)

def construct_generator(model_config, input_shape, output_shape, device):
    model = select_generator(
        model_config.model,
        input_shape  = input_shape,
        output_shape = output_shape,
        **model_config.model_args
    )

    return default_model_init(model, model_config, device)

