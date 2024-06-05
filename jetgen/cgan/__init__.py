
CGAN_MODELS = {
}

def select_model(name, **kwargs):
    if name not in CGAN_MODELS:
        raise ValueError("Unknown model: %s" % name)

    return CGAN_MODELS[name](**kwargs)

def construct_model(savedir, config, is_train, device):
    model = select_model(
        config.model, savedir = savedir, config = config, is_train = is_train,
        device = device, **config.model_args
    )

    return model

