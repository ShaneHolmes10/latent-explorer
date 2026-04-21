import argparse
import os
import sys
import h5py
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULTS
from utils.model_utils import get_model_class

"""
Evaluation script for trained decoder models.
Displays original images alongside their reconstructions,
computes quantitative metrics (MSE, PSNR) for both a fixed
sample and the full dataset, and saves all results to a yaml
file for tracking across runs.

Usage:
    python src/evaluate.py --run output/faces/runs/2026_04_20_1430 --model decoder
    python src/evaluate.py --run output/faces/runs/2026_04_20_1430 --model decoder --random 4
    python src/evaluate.py --run output/faces/runs/2026_04_20_1430 --model decoder --indices 0 5000 100000
"""

# Example usage:
# python src/evaluate.py --run output/faces/runs/<timestamp> --model decoder


# Fixed seed for reproducible sample selection
EVAL_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run", type=str, help="Path to a completed run folder"
    )
    source.add_argument(
        "--checkpoint", type=str, help="Path to a training checkpoint"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULTS["model"],
        help="Which model architecture (must match trained model)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULTS["dataset"],
        help="Which dataset to load originals from",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=DEFAULTS["latent_dim"],
        help="Latent space size (must match trained model)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=DEFAULTS["image_size"],
        help="Image resolution (must match trained model)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of reconstruction comparisons to show",
    )
    parser.add_argument(
        "--random",
        type=int,
        default=0,
        help="Number of random latent vector generations to show",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific image indices to reconstruct (overrides --n)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for full dataset evaluation",
    )

    return parser.parse_args()


def get_plot_dir(args):
    """Determine the plots directory from the run or checkpoint path."""

    if args.run:
        timestamp = os.path.basename(args.run)
    else:
        timestamp = "checkpoint"

    plot_dir = os.path.join("plots", timestamp)
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def load_model_and_vectors(args):
    """Load model weights and latent vectors from a run or checkpoint."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ModelClass = get_model_class(args.model)
    model = ModelClass(latent_dim=args.latent_dim, image_size=args.image_size)

    if args.run:
        path = os.path.join(args.run, "model.pt")
    else:
        path = args.checkpoint

    checkpoint = torch.load(path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    latent_vectors = checkpoint["latent_vectors"].to(device)

    return model, latent_vectors, device


def load_originals(dataset_name):
    """Load the processed HDF5 dataset for side by side comparison."""

    h5_path = os.path.join("data", dataset_name, "processed", "processed.h5")
    f = h5py.File(h5_path, "r")
    return f["images"], f


def compute_metrics(original, reconstructed):
    """
    Compute MSE and PSNR between an original and reconstructed image.

    @param original numpy array in [0, 1] range.
    @param reconstructed numpy array in [0, 1] range.
    @return Tuple of (mse, psnr).
    """
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * np.log10(1.0 / mse)
    return mse, psnr


def get_fixed_indices(num_vectors, n):
    """
    Get a fixed set of indices that is the same every time.
    Uses a fixed seed so results are reproducible across evaluations.
    """
    rng = np.random.RandomState(EVAL_SEED)
    return rng.choice(num_vectors, size=min(n, num_vectors), replace=False)


def compute_full_dataset_metrics(
    model, latent_vectors, originals, device, batch_size
):
    """
    Compute average MSE and PSNR across the entire dataset.

    @return Tuple of (average_mse, average_psnr).
    """
    num_samples = latent_vectors.shape[0]
    total_mse = 0.0
    count = 0

    for start in tqdm(
        range(0, num_samples, batch_size), desc="Full dataset eval"
    ):
        end = min(start + batch_size, num_samples)
        z = latent_vectors[start:end]

        with torch.no_grad():
            reconstructed = model(z)

        # Convert to numpy
        reconstructed = reconstructed.permute(0, 2, 3, 1).cpu().numpy()
        reconstructed = np.clip(reconstructed, 0, 1)

        # Load originals for this batch
        batch_originals = originals[start:end].astype(np.float32) / 255.0

        # Compute MSE per image
        batch_mse = np.mean(
            (batch_originals - reconstructed) ** 2, axis=(1, 2, 3)
        )
        total_mse += np.sum(batch_mse)
        count += len(batch_mse)

    avg_mse = total_mse / count
    avg_psnr = 10 * np.log10(1.0 / avg_mse) if avg_mse > 0 else float("inf")

    return avg_mse, avg_psnr


def reconstruct_and_compare(
    model, latent_vectors, originals, indices, device, plot_dir
):
    """Display original images alongside their reconstructions with metrics."""

    n = len(indices)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))

    if n == 1:
        axes = axes.reshape(2, 1)

    sample_metrics = []

    for i, idx in enumerate(indices):
        # Original image from HDF5 (uint8 0 to 255)
        original = originals[idx].astype(np.float32) / 255.0

        # Reconstruction from learned latent vector
        z = latent_vectors[idx].unsqueeze(0)
        with torch.no_grad():
            reconstructed = model(z)
        reconstructed = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
        reconstructed = np.clip(reconstructed, 0, 1)

        # Compute metrics
        mse, psnr = compute_metrics(original, reconstructed)
        sample_metrics.append(
            {"index": int(idx), "mse": float(mse), "psnr": float(psnr)}
        )

        axes[0, i].imshow(original)
        axes[0, i].set_title(f"Original #{idx}")
        axes[0, i].axis("off")

        axes[1, i].imshow(reconstructed)
        axes[1, i].set_title(f"MSE: {mse:.4f}  PSNR: {psnr:.1f} dB")
        axes[1, i].axis("off")

        print(f"  Image #{idx}  MSE: {mse:.4f}  PSNR: {psnr:.1f} dB")

    avg_mse = np.mean([m["mse"] for m in sample_metrics])
    avg_psnr = np.mean([m["psnr"] for m in sample_metrics])
    print(f"  Sample avg   MSE: {avg_mse:.4f}  PSNR: {avg_psnr:.1f} dB")

    plt.suptitle("Original vs Reconstruction", fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(plot_dir, "reconstruction.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Saved to {save_path}")

    plt.show()
    plt.close()

    return sample_metrics, avg_mse, avg_psnr


def generate_random(model, latent_dim, n, device, plot_dir):
    """Generate faces from random latent vectors."""

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))

    if n == 1:
        axes = [axes]

    for i in range(n):
        z = torch.randn(1, latent_dim, device=device)
        with torch.no_grad():
            img = model(z)
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].set_title(f"Random #{i + 1}")
        axes[i].axis("off")

    plt.suptitle("Random Latent Vector Generations", fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(plot_dir, "random_generation.png")
    plt.savefig(save_path, dpi=150)
    print(f"  Saved to {save_path}")

    plt.show()
    plt.close()


def save_metrics(
    plot_dir,
    sample_metrics,
    sample_avg_mse,
    sample_avg_psnr,
    full_mse,
    full_psnr,
):
    """Save all evaluation metrics to a yaml file."""

    metrics = {
        "sample_images": sample_metrics,
        "sample_average": {
            "mse": float(sample_avg_mse),
            "psnr": float(sample_avg_psnr),
        },
        "full_dataset": {
            "mse": float(full_mse),
            "psnr": float(full_psnr),
        },
    }

    path = os.path.join(plot_dir, "metrics.yaml")
    with open(path, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False)

    print(f"  Metrics saved to {path}")


def main():
    args = parse_args()

    model, latent_vectors, device = load_model_and_vectors(args)
    num_vectors = latent_vectors.shape[0]
    print(f"Loaded model with {num_vectors} latent vectors on {device}")

    plot_dir = get_plot_dir(args)

    # Determine indices (fixed or user specified)
    if args.indices:
        indices = args.indices
    else:
        indices = get_fixed_indices(num_vectors, args.n)

    # Reconstruction comparisons
    originals, h5_file = load_originals(args.dataset)
    print(f"Comparing {len(indices)} reconstructions (fixed indices):")
    sample_metrics, sample_avg_mse, sample_avg_psnr = reconstruct_and_compare(
        model, latent_vectors, originals, indices, device, plot_dir
    )

    # Full dataset metrics
    print("Computing full dataset metrics:")
    full_mse, full_psnr = compute_full_dataset_metrics(
        model, latent_vectors, originals, device, args.batch_size
    )
    print(f"  Full dataset MSE: {full_mse:.4f}  PSNR: {full_psnr:.1f} dB")

    h5_file.close()

    # Save all metrics
    save_metrics(
        plot_dir,
        sample_metrics,
        sample_avg_mse,
        sample_avg_psnr,
        full_mse,
        full_psnr,
    )

    # Random generations
    if args.random > 0:
        print(f"Generating {args.random} random faces:")
        generate_random(model, args.latent_dim, args.random, device, plot_dir)


if __name__ == "__main__":
    main()
