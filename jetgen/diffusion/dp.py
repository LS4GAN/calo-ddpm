import torch

class DiffusionProcess:

    def __init__(self, n, device, seed = None, time_map = None, prg = None):
        # pylint: disable=too-many-arguments

        assert (seed is None) != (prg is None), \
            f"Either seed={seed} or prg={prg} should be set. But not both."

        self._n        = n
        self._seed     = seed
        self._time_map = time_map
        self._device   = device

        if prg is not None:
            self._prg = prg
        else:
            self._prg = torch.Generator(device)
            self._prg.manual_seed(seed)

    @property
    def prg(self):
        return self._prg

    def forward_step(self, t, x_prev, **kwargs):
        """Calculate x[t] from x[t-1] = x_prev"""
        raise NotImplementedError

    def forward_jump(self, t, x0, **kwargs):
        """Calculate x[t] from x[0]"""
        raise NotImplementedError

    def backward_jump_given_eps(self, t, x, eps, **kwargs):
        """Calculate x[0] from x[t] and \\eps"""
        raise NotImplementedError

    def backward_step_given_x0(self, t, x, x0, **kwargs):
        """Calculate x[t-1] from x[t] and x[0]"""
        raise NotImplementedError

    def backward_step_given_eps(self, t, x, eps, **kwargs):
        """Calculate x[t-1] from x[t] and \\eps"""
        raise NotImplementedError

    def subsample(self, subsample, **kwargs):
        """Sub-sample diffusion process for faster inference"""
        raise NotImplementedError

    def marginal_variance(self):
        """Variance of the target normal distribution at t = T

        This should be 1 for the standard variance schedules.
        """
        raise NotImplementedError

    def __len__(self):
        return self._n

    def map_time(self, t):
        """Convert possibly sub-sampled time into the actual time"""
        if self._time_map is None:
            return t

        return self._time_map[t]

