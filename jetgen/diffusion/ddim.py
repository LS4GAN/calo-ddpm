"""
This file defines the DDIM diffusion process.

Ref.: https://arxiv.org/pdf/2010.02502

DDIM process is a bit more involved than the DDPM process (c.f. ./normal.py)
First, DDIM defined forward jump probabilities pt_{0->t} to match the
corresponding DDPM probabilities.

Next, DDIM considers a general form of the inverse step probability:

    ip_t{t->t-1} = N(x_{t-1} | iscale_t * x_t + ibias_t; ivar)  (3)

where
    ibias_t = \\lambda_t * x0 + \\mu_t

Then, DDIM constraints ip_t{t->t-1} on pt_{0->t} via the following relation
(c.f. Lemma 1 of Appendix B of the DDIM paper):

    pt_{0->t-1} = \\int dx_t ip_t{t->t-1} pt_{0->t-1}

This relation gives rise to the following constraints:
    \\lambda_t  + iscale_t * scale_{0->t} = scale_{0->t-1}
    \\mu_t      + iscale_t * bias_{0->t}  = bias_{0->t-1}
    ivar_t      + iscale_t^2 * var_{0->t} = var_{0->t-1}

Finally, DDIM imposes the constraint (ivar_t == 0), giving
    iscale_t = sqrt(var_{0->t-1} / var_{0->t})
    ibias_t  = (
        (scale_{0->t-1} - iscale_t * scale_{0->t}) * x0
      + (bias_{0->t-1}  - iscale_t * bias_{o->t})
    ivar_t   = 0

"""

import torch

from .dp     import DiffusionProcess
from .normal import (
    CondNorm, convolve_cond_norm_pdf_arr, x_from_eps, subsample_cond_norm_pdf_arr
)

from .subsample import generate_subsampling
from .vsched    import generate_variance_schedule

class DDIM(DiffusionProcess):
    # pylint: disable=abstract-method

    def __init__(
        self, vsched, device, seed = None, time_map = None, prg = None
    ):
        # pylint: disable=too-many-arguments
        self._fwd_p_steps = generate_variance_schedule(vsched).to(device)
        self._fwd_p_jumps = convolve_cond_norm_pdf_arr(self._fwd_p_steps)

        super().__init__(
            n        = len(self._fwd_p_steps)-1,
            device   = device,
            seed     = seed,
            time_map = time_map,
            prg      = prg,
        )

    def _generate_noise(self, x):
        return torch.randn(
            size = x.shape, generator = self._prg, device = x.device
        )

    def get_fwd_jump_p(self, t):
        return self._fwd_p_jumps[t]

    def get_bkw_p_step(self, t, x0):
        #
        # C.f. derivation in the docstrings in the header of this file
        #
        # var   = p_prev.var - scale**2 * p_curr.var
        # bias  = (
        #     (p_prev.scale - scale * p_curr.scale) * x0
        #   + (p_prev.bias  - scale * p_curr.bias)
        # )
        #
        # To make sure that var == 0, set
        #   scale = (p_prev.var / p_curr.var).sqrt()
        #

        p_prev = self._fwd_p_jumps[t-1].match_shape(x0)
        p_curr = self._fwd_p_jumps[t].match_shape(x0)

        scale_sq = (p_prev.var / p_curr.var)

        scale = scale_sq.sqrt()
        var   = torch.zeros_like(scale)
        bias  = (
              (p_prev.scale - scale * p_curr.scale) * x0
            + (p_prev.bias  - scale * p_curr.bias)
        )

        return CondNorm(scale, bias, var)

    def forward_jump(self, t, x0, eps = None):
        # pylint: disable=arguments-differ
        p = self._fwd_p_jumps[t].match_shape(x0)

        if eps is None:
            eps = self._generate_noise(x0)

        result = p.scale * x0 + p.bias + p.var.sqrt() * eps

        return (result, eps)

    def backward_step_given_x0(self, t, x, x0):
        # pylint: disable=arguments-differ
        p = self.get_bkw_p_step(t, x0)

        result = p.scale * x + p.bias

        return result

    def backward_step_given_eps(self, t, x, eps, **kwargs):
        p_jump = self._fwd_p_jumps[t].match_shape(x)
        x0     = x_from_eps(p_jump, x, eps)

        return self.backward_step_given_x0(t, x, x0, **kwargs)

    def subsample(self, subsample):
        # pylint: disable=arguments-differ
        subspace = generate_subsampling(subsample, len(self))

        vsched   = subsample_cond_norm_pdf_arr(self._fwd_p_steps, subspace)
        time_map = self.map_time(subspace.to(self._device))

        return DDIM(vsched, self._device, time_map = time_map, prg = self._prg)

    def marginal_variance(self):
        return self._fwd_p_jumps.var[-1]

