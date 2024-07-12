import os

import numpy as np
from torch.utils.data import Dataset

from jetgen.consts import SPLIT_TRAIN

DSET_INDEX = 'index'
DSET_DATA  = 'data'

H5_EXT = [ '', '.h5', '.hdf5' ]


class H5ArrayDomainHierarchy(Dataset):

    def __init__(
        self, path, domain,
        split     = SPLIT_TRAIN,
        transform = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self._path = None
        path_base  = os.path.join(path, split, domain)

        for ext in H5_EXT:
            path = path_base + ext
            if os.path.exists(path):
                self._path = path
                break
        else:
            raise RuntimeError(f"Failed to find h5 dataset '{path_base}'")

        # pylint: disable=import-outside-toplevel
        import h5py

        self._f    = h5py.File(self._path, 'r')
        self._dset = self._f.get(DSET_DATA)

        self._transform = transform

    def __len__(self):
        return len(self._dset)

    def __getitem__(self, index):
        result = np.float32(self._dset[index])

        if self._transform is not None:
            result = self._transform(result)

        return result

