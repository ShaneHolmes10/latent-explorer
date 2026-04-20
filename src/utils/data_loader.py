import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

"""
Data loader for reading preprocessed images from HDF5 files.
Returns a PyTorch DataLoader that yields (image_tensor, index) pairs
where index maps each image to its corresponding learnable latent vector.
"""


class HDF5Dataset(Dataset):
    """Dataset that reads images from an HDF5 file."""

    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.length = f["images"].shape[0]

        # Keep file handle open for fast repeated access
        self.file = h5py.File(h5_path, "r")
        self.images = self.file["images"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Read image, convert from uint8 [0,255] to float32 [0,1]
        img = self.images[idx].astype(np.float32) / 255.0

        # Convert from (H, W, C) to (C, H, W) for PyTorch
        img = np.transpose(img, (2, 0, 1))

        return torch.from_numpy(img), idx

    def close(self):
        self.file.close()


def get_data_loader(dataset_name, batch_size):
    """
    Build a DataLoader for the specified dataset.

    @param dataset_name Name of the dataset folder (e.g. "faces").
    @param batch_size Batch size for the DataLoader.
    @return Tuple of (DataLoader, number of samples).
    """
    h5_path = os.path.join("data", dataset_name, "processed", "processed.h5")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"Processed data not found at {h5_path}\n"
            f"Run the preprocessor first: python data/{dataset_name}/preprocessor.py"
        )

    dataset = HDF5Dataset(h5_path)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    print(f"Loaded {len(dataset)} images from {h5_path}")

    return loader, len(dataset)
