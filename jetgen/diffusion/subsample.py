import torch

from jetgen.torch.select import extract_name_kwargs

def gen_ddpm_linear_subspace(s, t):
    subspace = torch.linspace(1, t, steps = s)
    subspace = torch.round(subspace)

    pad_left = torch.zeros((1,))
    subspace = torch.cat((pad_left, subspace), dim = 0).long()

    return subspace

def gen_ddim_linear_subspace(s, t):
    step = t // s

    subspace = torch.arange(1, t+1, step = step)
    pad_left = torch.zeros((1,))

    subspace = torch.cat((pad_left, subspace), dim = 0).long()

    return subspace

def generate_subsampling(subsample, t):
    name, kwargs = extract_name_kwargs(subsample)

    if name == 'linear-ddpm':
        return gen_ddpm_linear_subspace(**kwargs, t = t)

    if name == 'linear-ddim':
        return gen_ddim_linear_subspace(**kwargs, t = t)

    raise ValueError(f"Unknown subsampling: {name}")

