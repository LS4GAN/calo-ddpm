import torch

from jetgen.torch.select import extract_name_kwargs
from .normal import CondNorm

def generate_linear_schedule(T = 1000, beta1 = 1e-4, betaT = 0.02):
    var_step = torch.linspace(beta1, betaT, T)
    pad      = torch.zeros((1,), dtype = var_step.dtype)

    var_step   = torch.cat((pad, var_step), dim = 0)
    scale_step = (1 - var_step).sqrt()
    bias_step  = torch.zeros_like(scale_step)

    return CondNorm(scale_step, bias_step, var_step)

def generate_cosine_schedule(T = 1000, s = 0.008, clip = 0.999):
    def fn(t):
        return torch.cos((t / T + s) / (1 + s) * torch.pi / 2).square()

    t = torch.arange(0, T+1)

    var_step = (1 - fn(t[1:]) / fn(t[:-1]))
    var_step = torch.clamp(var_step, 0, clip)
    pad      = torch.zeros((1,), dtype = var_step.dtype)

    var_step   = torch.cat((pad, var_step), dim = 0)
    scale_step = (1 - var_step).sqrt()
    bias_step  = torch.zeros_like(scale_step)

    return CondNorm(scale_step, bias_step, var_step)

def generate_hybrid_schedule(scale, var):
    scale_p = generate_variance_schedule(scale)
    var_p   = generate_variance_schedule(var)

    scale_p.var = var_p.var
    return scale_p

def generate_s_schedule(scale, var):
    result = generate_variance_schedule(var)

    n = len(result.scale) - 1
    s = pow(scale, 1/n)

    result.scale[1:] = s * result.scale[1:]

    return result

def generate_variance_schedule(vsched):
    if isinstance(vsched, CondNorm):
        return vsched

    name, kwargs = extract_name_kwargs(vsched)

    if name == 'linear':
        return generate_linear_schedule(**kwargs)

    if name == 'cosine':
        return generate_cosine_schedule(**kwargs)

    if name == 'hybrid':
        return generate_hybrid_schedule(**kwargs)

    if name == 's-sched':
        return generate_s_schedule(**kwargs)

    raise ValueError(f"Unknown vairance sched: {name}")

