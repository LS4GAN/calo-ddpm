import itertools
import unittest

import torch

from jetgen.diffusion.ddim   import DDIM
from jetgen.diffusion.vsched import generate_linear_schedule

# https://arxiv.org/pdf/2010.02502

def gen_beta_schedule(beta1 = 1e-4, betaT = 0.02, T = 1000):
    result  = torch.linspace(beta1, betaT, T)
    result0 = torch.zeros((1,))

    return torch.cat((result0, result))

def alpha_from_beta(beta):
    return (1 - beta)

def calc_alpha_bar(alpha):
    return torch.cumprod(alpha, dim = 0)

def calc_forward_scale(alpha_bar):
    return alpha_bar.sqrt()

def calc_forward_var(alpha_bar):
    return (1 - alpha_bar)

def calc_imu(alpha, alpha_bar, beta, t, x0):
    # Equation (7) of https://arxiv.org/pdf/2010.02502
    # in DDPM notations
    scale = ((1 - alpha_bar[t-1]) / (1 - alpha_bar[t])).sqrt()
    bias  = (alpha_bar[t-1].sqrt() - scale * alpha_bar[t].sqrt()) * x0

    return (scale, bias)

class TestDDIM(unittest.TestCase):

    def test_linear_schedule(self):
        beta1_list = [ 0, 1e-4, 1]
        betaT_list = [ 0.01, 0.02, 1 ]
        T_list     = [ 0, 1, 10, 1000]

        search_space = itertools.product(beta1_list, betaT_list, T_list)

        for (beta1, betaT, T) in search_space:
            sched_step_null = gen_beta_schedule(beta1, betaT, T)

            scale_step_null = (1 - sched_step_null).sqrt()
            var_step_null   = sched_step_null

            p_steps = generate_linear_schedule(T, beta1, betaT)

            self.assertTrue(torch.allclose(var_step_null,   p_steps.var))
            self.assertTrue(torch.allclose(scale_step_null, p_steps.scale))

    def test_ddim_fwd_p_jump(self):
        beta1 = 1e-4
        betaT = 0.1
        T     = 10

        beta      = gen_beta_schedule(beta1, betaT, T)
        alpha     = alpha_from_beta(beta)
        alpha_bar = calc_alpha_bar(alpha)

        scale_null = calc_forward_scale(alpha_bar)
        bias_null  = torch.zeros_like(scale_null)
        var_null   = calc_forward_var(alpha_bar)

        vsched = generate_linear_schedule(T, beta1, betaT)
        ddim   = DDIM(vsched, seed = 0, device = 'cpu')

        p_jumps = ddim._fwd_p_jumps

        self.assertTrue(torch.allclose(scale_null, p_jumps.scale))
        self.assertTrue(torch.allclose(bias_null,  p_jumps.bias))
        self.assertTrue(torch.allclose(var_null,   p_jumps.var, atol = 1e-6))

    def test_ddim_bkw_p_step(self):
        beta1 = 1e-4
        betaT = 0.1
        T     = 10

        beta      = gen_beta_schedule(beta1, betaT, T)
        alpha     = alpha_from_beta(beta)
        alpha_bar = calc_alpha_bar(alpha)

        vsched = generate_linear_schedule(T, beta1, betaT)
        ddim   = DDIM(vsched, seed = 0, device = 'cpu')

        for t in range(1, T+1):
            for x0 in [ 0, 0.5, 1 ]:
                scale_null, bias_null = calc_imu(alpha, alpha_bar, beta, t, x0)
                var_null = torch.zeros_like(scale_null)

                p_test = ddim.get_bkw_p_step(t, x0)

                self.assertTrue(
                    torch.allclose(scale_null, p_test.scale, atol = 1e-3),
                    (
                        f'x0 = {x0}, t = {t}, scale, '
                        f'{scale_null} != {p_test.scale}'
                    )
                )

                self.assertTrue(
                    torch.allclose(bias_null, p_test.bias, atol = 1e-3),
                    f'x0 = {x0}, t = {t}, bias, {bias_null} != {p_test.bias}'
                )

                self.assertTrue(
                    torch.allclose(var_null, p_test.var, atol = 1e-3),
                    f'x0 = {x0}, t = {t}, var, {var_null} != {p_test.var}'
                )

if __name__ == '__main__':
    unittest.main()

