import torch
from torch.utils.data import Dataset

class NoiseDataset(Dataset):

    def __init__(
        self, shape, length, mu = 0, sigma = 1, seed = 0, transform = None,
        **kwargs
    ):
        # pylint: disable=too-many-arguments
        super().__init__(**kwargs)

        self._seed   = seed
        self._length = length
        self._shape  = shape

        self._mu    = mu
        self._sigma = sigma

        self._transform = transform
        self._gen       = torch.Generator()

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        self._gen.manual_seed(self._seed + index)

        result = torch.normal(
            self._mu, self._sigma, size = self._shape, generator = self._gen
        )

        if self._transform is not None:
            result = self._transform(result)

        return result

