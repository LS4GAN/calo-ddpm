# import math
import logging

from torch import nn
from torchvision.transforms import CenterCrop

from jetgen.torch.select import get_activ_layer, get_norm_layer

LOGGER = logging.getLogger('models.generator.dcgan')

DEF_FEATURES = 1024
DEF_NORM     = 'batch'
DEF_ACTIV    = 'relu'

def math_prod(shape):
    result = 1

    for x in shape:
        result *= x

    return result

class DCGANGenerator(nn.Module):

    def __init__(
        self, input_shape, output_shape, features_list,
        activ          = DEF_ACTIV,
        norm           = DEF_NORM,
        activ_output   = None
    ):
        # pylint: disable=dangerous-default-value
        # pylint: disable=too-many-arguments
        super().__init__()

        self._input_shae     = input_shape
        self._output_shape   = output_shape

        # to reshape into (2, 2)
        dense_features = 4 * features_list[0]

        self._net_in = nn.Sequential(
            nn.Flatten(),
            nn.Linear(math_prod(input_shape), dense_features),
            get_activ_layer(activ),
        )

        layers = []

        curr_shape = (features_list[0], 2, 2)

        for features in features_list[1:]:
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(
                    curr_shape[0], features,
                    kernel_size = 5, stride = 2, padding = 2,
                    output_padding = 1,
                ),
                get_norm_layer(norm, features),
                get_activ_layer(activ),
            ))

            curr_shape = (features, 2 * curr_shape[1], 2 * curr_shape[2])

        layers.append(nn.Sequential(
            nn.ConvTranspose2d(
                curr_shape[0], output_shape[0],
                kernel_size = 5, stride = 2, padding = 2,
                output_padding = 1,
            ),
            get_activ_layer(activ_output),
        ))

        curr_shape = (output_shape[0], 2 * curr_shape[1], 2 * curr_shape[2])

        if curr_shape != tuple(output_shape):
            LOGGER.warning(
                "DCGAN output shape '%s' is not equal to the expected output"
                " shape '%s'. Adding center cropping.",
                curr_shape, tuple(output_shape)
            )

            layers.append(CenterCrop(tuple(output_shape[1:])))

        self._net_main = nn.Sequential(*layers)

    def forward(self, z):
        # z : (N, F)

        # x : (N, 4 * C)
        x = self._net_in(z)

        # x : (N, C, 2, 2)
        x = x.reshape((x.shape[0], -1, 2, 2))

        result = self._net_main(x)

        return result

