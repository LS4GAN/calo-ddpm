import math
import logging

from torch import nn
from torchvision.transforms import CenterCrop

from jetgen.torch.select import get_activ_layer, get_norm_layer

LOGGER = logging.getLogger('models.discriminator.dcgan')

DEF_NORM     = 'batch'
DEF_ACTIV    = {
    'name' : 'leakyrelu',
    'negative_slope' : 0.2,
}

def math_prod(shape):
    result = 1

    for x in shape:
        result *= x

    return result

def get_padding_layer(input_shape, downscale_factor):
    need_pad = False

    if (
           (input_shape[1] < downscale_factor)
        or (input_shape[2] < downscale_factor)
    ):
        need_pad = True

        LOGGER.warning(
            "DCGAN input shape '%s' is smaller than the downscale factor '%d'."
            " Adding padding.", tuple(input_shape), downscale_factor
        )

    if (
           (input_shape[1] % downscale_factor != 0)
        or (input_shape[2] % downscale_factor != 0)
    ):
        need_pad = True

        LOGGER.warning(
            "DCGAN input shape '%s' is not divisible by the downscale "
            " factor '%d'. Adding padding.",
            tuple(input_shape), downscale_factor
        )

    h = math.ceil(input_shape[1] / downscale_factor) * downscale_factor
    w = math.ceil(input_shape[2] / downscale_factor) * downscale_factor

    if not need_pad:
        return None, h, w

    return CenterCrop((h, w)), h ,w

class DCGANDiscriminator(nn.Module):

    def __init__(
        self, image_shape, features_list, activ = DEF_ACTIV, norm = DEF_NORM,
    ):
        # pylint: disable=dangerous-default-value
        # pylint: disable=too-many-arguments
        super().__init__()

        self._input_shape = image_shape

        curr_features    = image_shape[0]
        downscale_factor = 1
        layers = []

        layers.append(nn.Sequential(
            nn.Conv2d(
                curr_features, features_list[0], kernel_size = 5,
                stride = 2, padding = 2
            ),
            get_activ_layer(activ)
        ))

        downscale_factor *= 2
        curr_features     = features_list[0]

        for features in features_list[1:]:
            layers.append(nn.Sequential(
                nn.Conv2d(
                    curr_features, features, kernel_size = 5,
                    stride = 2, padding = 2
                ),
                get_norm_layer(norm, features),
                get_activ_layer(activ)
            ))

            downscale_factor *= 2
            curr_features     = features

        padding, h, w = get_padding_layer(image_shape, downscale_factor)
        curr_shape    = (
            curr_features, h // downscale_factor, w // downscale_factor
        )

        if padding is not None:
            layers = [ padding, ] + layers

        self._net_main = nn.Sequential(*layers)
        dense_features = math_prod(curr_shape)

        self._net_output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dense_features, 1),
        )

    def forward(self, x):
        # x : (N, C, H, W)

        # y : (N, Ci, Hi, Wi)
        y = self._net_main(x)

        return self._net_output(y)

