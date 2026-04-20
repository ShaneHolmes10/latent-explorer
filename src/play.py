import argparse

from config import DEFAULTS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Explore the latent space interactively"
    )

    parser.add_argument(
        "--run",
        type=str,
        default=DEFAULTS["run"],
        help="Path to the run folder to load",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULTS["model"],
        help="Which model architecture (must match the trained model)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULTS["dataset"],
        help="Which dataset (for loading the correct PCA)",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=DEFAULTS["components"],
        help="Number of PCA sliders to display",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=DEFAULTS["latent_dim"],
        help="Size of the latent space (must match the trained model)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=DEFAULTS["image_size"],
        help="Image resolution (must match the trained model)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Run:        {args.run}")
    print(f"Model:      {args.model}")
    print(f"Dataset:    {args.dataset}")
    print(f"Components: {args.components}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Image size: {args.image_size}")


if __name__ == "__main__":
    main()
