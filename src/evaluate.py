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
Compares original images against their reconstructions from learned
latent vectors, computes quantitative metrics (MSE, PSNR), and
optionally evaluates the full dataset.
"""

# Example usage:
# python src/evaluate.py --trained output/faces/runs/2026_04_20_1856/model.pt --model decoder --random 5
# python src/evaluate.py --trained output/faces/runs/2026_04_20_1856/model.pt --model decoder --indices 0 5000 100000
# python src/evaluate.py --trained output/faces/runs/2026_04_20_1856/model.pt --model decoder --eval-all
# python src/evaluate.py --trained output/faces/runs/2026_04_20_1856/model.pt --model decoder --random 5 --eval-all --save plots/my_eval/


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")

    parser.add_argument(
        "--trained",
        type=str,
        required=True,
        help="Path to a .pt file containing model weights and latent vectors",
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

    # What to evaluate (mutually exclusive for sample selection)
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument(
        "--random",
        type=int,
        default=None,
        help="Evaluate N randomly selected images",
    )
    sample_group.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Evaluate specific images by index (e.g. --indices 0 5000 100000)",
    )

    parser.add_argument(
        "--eval-all",
        action="store_true",
        help="Compute MSE and PSNR across the entire dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for full dataset evaluation",
    )

    # Output
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Directory to save plots and metrics to",
    )

    return parser.parse_args()


def load_model_and_vectors(args):
    """Load model weights and latent vectors from a .pt file."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ModelClass = get_model_class(args.model)
    model = ModelClass(latent_dim=args.latent_dim, image_size=args.image_size)

    checkpoint = torch.load(
        args.trained, weights_only=False, map_location=device
    )
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


def evaluate_samples(
    model, latent_vectors, originals, indices, device, save_dir=None
):
    """Display original images alongside their reconstructions with metrics."""

    n = len(indices)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))

    if n == 1:
        axes = axes.reshape(2, 1)

    sample_metrics = []

    for i, idx in enumerate(indices):
        original = originals[idx].astype(np.float32) / 255.0

        z = latent_vectors[idx].unsqueeze(0)
        with torch.no_grad():
            reconstructed = model(z)
        reconstructed = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
        reconstructed = np.clip(reconstructed, 0, 1)

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

    if save_dir:
        save_path = os.path.join(save_dir, "reconstruction.png")
        plt.savefig(save_path, dpi=150)
        print(f"  Saved to {save_path}")

    plt.show()
    plt.close()

    return {
        "samples": sample_metrics,
        "sample_average": {
            "mse": float(avg_mse),
            "psnr": float(avg_psnr),
        },
    }


def evaluate_full_dataset(
    model, latent_vectors, originals, device, batch_size
):
    """Compute average MSE and PSNR across the entire dataset."""

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

        reconstructed = reconstructed.permute(0, 2, 3, 1).cpu().numpy()
        reconstructed = np.clip(reconstructed, 0, 1)

        batch_originals = originals[start:end].astype(np.float32) / 255.0

        batch_mse = np.mean(
            (batch_originals - reconstructed) ** 2, axis=(1, 2, 3)
        )
        total_mse += np.sum(batch_mse)
        count += len(batch_mse)

    avg_mse = total_mse / count
    avg_psnr = 10 * np.log10(1.0 / avg_mse) if avg_mse > 0 else float("inf")

    return {
        "full_dataset": {
            "mse": float(avg_mse),
            "psnr": float(avg_psnr),
        },
    }


def save_results(save_dir, results):
    """Save all evaluation metrics to a yaml file."""

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "metrics.yaml")
    with open(path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"Metrics saved to {path}")


def main():
    args = parse_args()

    model, latent_vectors, device = load_model_and_vectors(args)
    num_vectors = latent_vectors.shape[0]
    print(f"Loaded model with {num_vectors} latent vectors on {device}")

    originals, h5_file = load_originals(args.dataset)

    results = {}

    # Sample evaluation
    if args.random is not None:
        indices = np.random.choice(
            num_vectors, size=min(args.random, num_vectors), replace=False
        )
        print(f"Evaluating {len(indices)} random samples:")
        sample_results = evaluate_samples(
            model,
            latent_vectors,
            originals,
            indices,
            device,
            save_dir=args.save,
        )
        results.update(sample_results)

    elif args.indices is not None:
        indices = args.indices
        print(f"Evaluating {len(indices)} specified samples:")
        sample_results = evaluate_samples(
            model,
            latent_vectors,
            originals,
            indices,
            device,
            save_dir=args.save,
        )
        results.update(sample_results)

    # Full dataset evaluation
    if args.eval_all:
        print("Computing full dataset metrics:")
        full_results = evaluate_full_dataset(
            model, latent_vectors, originals, device, args.batch_size
        )
        print(
            f"  Full dataset MSE: {full_results['full_dataset']['mse']:.4f}  PSNR: {full_results['full_dataset']['psnr']:.1f} dB"
        )
        results.update(full_results)

    h5_file.close()

    # Save results if requested
    if args.save and results:
        save_results(args.save, results)

    # If nothing was requested, print help
    if args.random is None and args.indices is None and not args.eval_all:
        print(
            "No evaluation mode specified. Use --random, --indices, or --eval-all."
        )


if __name__ == "__main__":
    main()
