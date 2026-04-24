import argparse
import os
import sys
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)

from config import DEFAULTS
from models.pca_decoder import PcaDecoder

"""
Builds a PCA decoder .pt file from a trained decoder's saved weights.
Loads the decoder weights and latent vectors, fits PCA on the latent
vectors, and saves a new .pt file that play.py can load directly
with --model pca_decoder.
"""

# Example usage:
# python src/utils/build_pca_decoder.py --input output/faces/runs/2026_04_21_0622/model.pt --output output/faces/runs/2026_04_21_0622/pca_model.pt
# python src/play.py --trained output/faces/runs/2026_04_21_0622/pca_model.pt --model pca_decoder


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Build a PCA decoder from a trained decoder"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a trained decoder .pt file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the PCA decoder .pt file",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=DEFAULTS["latent_dim"],
        help="Latent space size (must match the trained decoder)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=DEFAULTS["image_size"],
        help="Image resolution (must match the trained decoder)",
    )

    return parser.parse_args()


def build(args):
    """
    Load a trained decoder's weights and latent vectors, fit PCA
    on the latent vectors, and save a PCA decoder .pt file.

    @param args Parsed argparse namespace.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained decoder's saved weights and latent vectors
    print(f"Loading trained decoder from {args.input}")
    saved_data = torch.load(
        args.input, weights_only=False, map_location=device
    )

    # Create PCA decoder and load the decoder weights into its inner decoder
    pca_model = PcaDecoder(
        latent_dim=args.latent_dim, image_size=args.image_size
    )
    pca_model.decoder.load_state_dict(saved_data["model_state_dict"])
    pca_model.to(device)

    # Fit PCA on the trained latent vectors
    latent_vectors = saved_data["latent_vectors"].to(device)
    print(f"Fitting PCA on {latent_vectors.shape[0]} latent vectors")
    pca_latent_vectors = pca_model.fit_pca(latent_vectors)

    # Save in the same format play.py expects:
    # model_state_dict: full PcaDecoder state (decoder weights + PCA parameters)
    # latent_vectors: PCA transformed vectors (for slider range computation in play.py)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(
        {
            "model_state_dict": pca_model.state_dict(),
            "latent_vectors": pca_latent_vectors,
        },
        args.output,
    )

    print(f"PCA decoder saved to {args.output}")
    print()
    print("Usage:")
    print(f"  python src/play.py --trained {args.output} --model pca_decoder")


def main():
    """Entry point. Parses arguments and builds the PCA decoder."""

    args = parse_args()
    build(args)


if __name__ == "__main__":
    main()
