import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Example usage:
# python data/view_h5.py --file data/faces/raw/raw.h5 --top 4
# python data/view_h5.py --file data/faces/processed/processed.h5 --random 10


def view_images(args):
    with h5py.File(args.file, "r") as f:
        images = f["images"]
        total = images.shape[0]

        if args.random:
            n = args.random
            indices = np.random.choice(
                total, size=min(n, total), replace=False
            )
        else:
            n = args.top
            indices = list(range(min(n, total)))

        cols = min(n, 5)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = np.array(axes).flatten()

        for i, idx in enumerate(indices):
            axes[i].imshow(images[idx])
            axes[i].set_title(f"#{idx}")
            axes[i].axis("off")

        for i in range(len(indices), len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="View images from an HDF5 file"
    )
    parser.add_argument(
        "--file", type=str, required=True, help="Path to the HDF5 file"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--top", type=int, help="View the first N images")
    group.add_argument("--random", type=int, help="View N random images")

    args = parser.parse_args()
    view_images(args)


if __name__ == "__main__":
    main()
