import h5py
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform


class CustomImageFolderH5(Dataset):
    """ Dataset for loading images from an HDF5 file. """
    def __init__(
            self,
            folder_path: str,
            transform: Transform,
    ) -> None:
        self.folder_path = folder_path
        self.transform = transform
        with h5py.File(self.folder_path, 'r') as h5file:
            self.length = h5file["images"].shape[0]

    def _open_hdf5(self):
        self._h5file = h5py.File(self.folder_path, 'r')
        self._dataset = self._h5file['images']

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        if not hasattr(self, "_h5file"):
            self._open_hdf5()
        return self.transform(self._dataset[idx])