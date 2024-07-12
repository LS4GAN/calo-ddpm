import torch
import torchvision
from torchvision import transforms

from jetgen.torch.select import extract_name_kwargs

FromNumpy = lambda : torch.from_numpy

TRANSFORM_DICT = {
    'to-tensor'              : transforms.ToTensor,
    'from-numpy'             : FromNumpy,
    'center-crop'            : transforms.CenterCrop,
    'color-jitter'           : transforms.ColorJitter,
    'random-crop'            : transforms.RandomCrop,
    'random-flip-vertical'   : transforms.RandomVerticalFlip,
    'random-flip-horizontal' : transforms.RandomHorizontalFlip,
    'random-rotation'        : transforms.RandomRotation,
    'resize'                 : transforms.Resize,
}

def select_single_transform(transform):
    name, kwargs = extract_name_kwargs(transform)

    if name not in TRANSFORM_DICT:
        raise ValueError(f"Unknown transform: '{name}'")

    return TRANSFORM_DICT[name](**kwargs)

def select_transform(transform):
    if transform is None:
        return None

    result = []

    if not isinstance(transform, (list, tuple)):
        transform = [ transform, ]

    result = [ select_single_transform(x) for x in transform ]

    return torchvision.transforms.Compose(result)

