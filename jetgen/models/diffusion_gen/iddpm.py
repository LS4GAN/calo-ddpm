from torch import nn

class IDDPMUNet(nn.Module):

    def __init__(
        self,
        n_steps,
        input_shape,
        output_shape,
        **kwargs,
    ):
        # pylint: disable=unused-argument
        super().__init__()

        # pylint: disable=import-outside-toplevel
        from improved_diffusion.unet import UNetModel

        self._net = UNetModel(
            in_channels  = input_shape[0],
            out_channels = output_shape[0],
            **kwargs
        )

    def forward(self, x, t, y=None):
        return self._net(x, t, y)

