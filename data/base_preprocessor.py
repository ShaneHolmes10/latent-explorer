import os
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


class BasePreprocessor:
    """
    Base preprocessor that handles common image preprocessing operations.

    To create a dataset specific preprocessor:
        1. Create a preprocessor.py in data/<dataset>/
        2. Define a class called Preprocessor that inherits from
        BasePreprocessor
        3. Override process_image() to add custom steps
    """

    def __init__(self, dataset, image_size=128):
        self.dataset = dataset
        self.image_size = image_size
        self.raw_path = os.path.join("data", dataset, "raw", "raw.h5")
        self.processed_path = os.path.join(
            "data", dataset, "processed", "processed.h5"
        )

    def resize(self, img):
        """Resize image to target dimensions."""
        return img.resize((self.image_size, self.image_size), Image.LANCZOS)

    def center_crop_square(self, img):
        """Crop the largest centered square from the image."""
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        return img.crop((left, top, left + side, top + side))

    def process_image(self, img):
        """
        Process a single image. Override this in subclasses to add
        dataset specific steps. Base implementation just resizes.
        """
        img = self.resize(img)
        return img

    def run(self):
        """Read from raw HDF5, process each image, write to processed HDF5."""
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)

        with h5py.File(self.raw_path, "r") as raw_f:
            raw_images = raw_f["images"]
            num_images = raw_images.shape[0]
            print(f"Found {num_images} images in {self.raw_path}")
            print(f"Target size: {self.image_size}x{self.image_size}")

            with h5py.File(self.processed_path, "w") as proc_f:
                dataset = proc_f.create_dataset(
                    "images",
                    shape=(num_images, self.image_size, self.image_size, 3),
                    dtype=np.uint8,
                    chunks=(1, self.image_size, self.image_size, 3),
                )

                for i in tqdm(range(num_images), desc="Processing"):
                    img = Image.fromarray(raw_images[i])
                    img = self.process_image(img)
                    dataset[i] = np.array(img)

                if "filenames" in raw_f:
                    proc_f.create_dataset(
                        "filenames", data=raw_f["filenames"][:]
                    )

        print(f"Done. Saved {num_images} images to {self.processed_path}")
