import os
import argparse
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

# Example usage:
# python pack_hdf5.py --input data/faces/raw/img_align_celeba/img_align_celeba --output data/faces/raw/raw.h5


def convert_to_hdf5(image_dir, output_path):
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [
        f
        for f in sorted(os.listdir(image_dir))
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]

    print(f"Found {len(files)} images in {image_dir}")

    first_img = np.array(
        Image.open(os.path.join(image_dir, files[0])).convert("RGB")
    )
    h, w, c = first_img.shape
    print(f"Image dimensions: {w}x{h}x{c}")

    with h5py.File(output_path, "w") as f:
        dataset = f.create_dataset(
            "images",
            shape=(len(files), h, w, c),
            dtype=np.uint8,
            chunks=(1, h, w, c),
        )

        filenames = []
        for i, fname in enumerate(tqdm(files, desc="Packing")):
            img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
            dataset[i] = np.array(img)
            filenames.append(fname)

        f.create_dataset("filenames", data=np.array(filenames, dtype="S"))

    size_gb = os.path.getsize(output_path) / (1024**3)
    print(
        f"Done. Saved {len(files)} images to {output_path} ({size_gb:.2f} GB)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pack images into an HDF5 file"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing the images",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output HDF5 file path"
    )

    args = parser.parse_args()
    convert_to_hdf5(args.input, args.output)


if __name__ == "__main__":
    main()
