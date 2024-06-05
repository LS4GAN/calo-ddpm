from jetgen.torch.select import extract_name_kwargs

from .ddpm import DDPM
from .ddim import DDIM

DP_DICT = {
    'ddpm' : DDPM,
    'ddim' : DDIM,
}

def select_diffusion_process(dp, device):
    name, kwargs = extract_name_kwargs(dp)

    if name in DP_DICT:
        return DP_DICT[name](**kwargs, device = device)

    raise ValueError(f"Unknown diffusion process: '{name}'")


