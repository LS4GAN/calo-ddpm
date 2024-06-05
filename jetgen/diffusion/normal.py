"""
Gaussian diffusion processes are built around conditional transition
probabilities of the form:

    p_t(y | x) = N(y; scale_t * x + bias_t; var_t)          (1)

which are also sometimes expressed as:

    y = scale * x + bias + sqrt(var) * eps                  (2)

where "eps" is a standard normal noise.

This file defines a structure `CondNorm` that parametrizes the transition
probability (1).

Likewise, it contains functions for:

1. Convolution of two transition probabilities:
       p_{0->2}(z | x) := \\int dy p_{1->2}(z | y) * p_{0->1}(y | x)
   which is also of the form (1).

2. Inversion of the transition probabilities:
       p_{2->1}(y | z, x) = p_{1->2}(z | y) * p_{0->1}(y | x) / p_{0->2}(z | x)
   which can be expressed as (1) as well.
"""

import torch
from .funcs import match_shape

class CondNorm:

    __slots__ = [
        'scale', 'bias', 'var'
    ]

    def __init__(self, scale, bias, var):
        self.scale = scale
        self.bias  = bias
        self.var   = var

    def __getitem__(self, index):
        scale = self.scale[index]
        bias  = self.bias[index]
        var   = self.var[index]

        return CondNorm(scale, bias, var)

    def match_shape(self, target):
        if self.scale.ndim != 1:
            return self

        scale = match_shape(self.scale, target)
        bias  = match_shape(self.bias, target)
        var   = match_shape(self.var, target)

        return CondNorm(scale, bias, var)

    def to(self, device):
        scale = self.scale.to(device)
        bias  = self.bias.to(device)
        var   = self.var.to(device)

        return CondNorm(scale, bias, var)

    def __len__(self):
        return len(self.scale)

    def __repr__(self):
        result  = 'CondNorm({\n'
        result += f'  scale = {self.scale}\n'
        result += f'  bias  = {self.bias}\n'
        result += f'  var   = {self.var}\n'
        result += '}'

        return result

def cat_cond_norms(cond_norms):
    scale = torch.cat([x.scale for x in cond_norms])
    bias  = torch.cat([x.bias  for x in cond_norms])
    var   = torch.cat([x.var   for x in cond_norms])

    return CondNorm(scale, bias, var)

def convolve_cond_norm_pdfs(p_zy : CondNorm, p_yx : CondNorm) -> CondNorm:
    """Perform convolution of two adjacent transition probabilities.

    This function finds parameters of the "p_{0->2}(z | x)" which is a result
    of a convolution of two adjacent gaussian steps:
        p_{0->2}(z | x) := \\int dy p_{1->2}(z | y) * p_{0->1}(y | x)
    """

    scale = p_zy.scale * p_yx.scale
    bias  = p_zy.bias + p_zy.scale    * p_yx.bias
    var   = p_zy.var  + p_zy.scale**2 * p_yx.var

    return CondNorm(scale, bias, var)

def invert_cond_norm_pdfs(p_zy : CondNorm, p_yx : CondNorm, x) -> CondNorm:
    """Perform conditional inversion of a transition probability.

    This function finds parameters of the "p_{2->1}(y | z, x0)" which is a
    conditional inversion of the gaussian transition probability
    "p_{1->2}(z | y)".

    The inversion is calculated via Bayes rule:

    p_{2->1}(y | z, x) = p_{1->2}(z | y) * p_{0->1}(y | x) / p_{0->2}(z | x)

    """
    norm  = (p_zy.var + p_zy.scale**2 * p_yx.var)

    scale = p_zy.scale * p_yx.var / norm

    bias  = (
          p_yx.scale * x * p_zy.var
        + p_yx.bias * p_zy.var
        - p_zy.scale * p_zy.bias * p_yx.var
    ) / norm

    var   = p_zy.var * p_yx.var / norm

    return CondNorm(scale, bias, var)

def x_from_eps(p_yx : CondNorm, y, eps):
    """Calculate value before gaussian step knowing the result and added noise.

    Assuming:
        y = scale * x + bias + sqrt(var) * eps

    The `x` is given by:
        x = 1 / scale * (y - bias - sqrt(var) * eps)
    """

    return 1 / p_yx.scale * (y - p_yx.bias - p_yx.var.sqrt() * eps)

def convolve_norm_scale_arr(scale_list):
    """Calculate cumulative transition prob scales given a series of steps.

    Given a series of scale parameters of adjacent transition steps p_{i->i+1},
    this function calculates scales of the cumulative transitions p_{0->i}:

        scale_{0->i} = scale_{i-1->i} * scale_{0->i-1}
        scale_{0->0} = 1

    See Also
    --------
        convolve_cond_norm_pdfs
        convolve_norm_bias_arr
        convolve_norm_var_arr
    """
    return torch.cumprod(scale_list, dim = 0)

def convolve_norm_bias_arr(scale, bias):
    """Calculate cumulative transition prob biases given a series of steps.

    Given series of scale and bias parameters of adjacent transition steps
    p_{i->i+1}, this function calculates biases of the cumulative transitions
    p_{0->i}:

        bias_{0->i} = bias_{i-1->i} + scale_{i-1->i} * bias_{0->i-1}
        bias_{0->0} = 0

    See Also
    --------
        convolve_cond_norm_pdfs
        convolve_norm_scale_arr
        convolve_norm_var_arr
    """

    result    = torch.zeros_like(bias)
    result[0] = bias[0]

    for i in range(1, len(result)):
        result[i] = bias[i] + scale[i] * result[i-1]

    return result

def convolve_norm_var_arr(scale, var):
    """Calculate cumulative transition prob variances given a series of steps.

    Given series of scale and variance parameters of adjacent transition
    steps p_{i->i+1}, this function calculates variances of the cumulative
    transitions p_{0->i}:

        var_{0->i} = var_{i-1->i} + scale_{i-1->i}^2 * var_{0->i-1}
        var_{0->0} = 0

    See Also
    --------
        convolve_cond_norm_pdfs
        convolve_norm_scale_arr
        convolve_norm_bias_arr
    """

    result    = torch.zeros_like(var)
    result[0] = var[0]

    for i in range(1, len(result)):
        result[i] = var[i] + scale[i]**2 * result[i-1]

    return result

def convolve_cond_norm_pdf_arr(p_steps : CondNorm) -> CondNorm:
    """Calculate cumulative transition probabilities given a series of steps

    See Also
    --------
        convolve_cond_norm_pdfs
        convolve_norm_scale_arr
        convolve_norm_bias_arr
        convolve_norm_var_arr
    """
    result_scale = convolve_norm_scale_arr(p_steps.scale)
    result_bias  = convolve_norm_bias_arr(p_steps.scale, p_steps.bias)
    result_var   = convolve_norm_var_arr(p_steps.scale, p_steps.var)

    return CondNorm(result_scale, result_bias, result_var)

def subsample_cond_norm_pdf_arr(
    p_steps : CondNorm, subspace : torch.Tensor
) -> CondNorm:
    """Sub-sample a list of transition probabilities by times `subspace`"""

    t_prev = 0
    result = [ ]

    for t in subspace:
        t_curr = t.cpu().item()

        p_jumps = convolve_cond_norm_pdf_arr(p_steps[t_prev:t_curr+1])
        p_jump  = p_jumps[-1:]

        result.append(p_jump)

        t_prev = t_curr+1

    return cat_cond_norms(result)

