import argparse
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from config import DEFAULTS
from utils.model_utils import (
    get_model_class,
    create_run_folder,
    save_meta,
    save_checkpoint,
    load_checkpoint,
    save_run,
)
from utils.data_loader import get_data_loader
from utils.plotting import plot_training_curves


"""
Training entry point for latent explorer models.
Loads preprocessed image data from HDF5, dynamically loads the specified
model architecture, and trains using learnable per sample latent vectors
with MSE reconstruction loss.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Train a latent space model")

    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULTS["dataset"],
        help="Which dataset to use (e.g. faces, cars)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULTS["model"],
        help="Which model architecture to use (e.g. decoder, vae_decoder)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULTS["epochs"],
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULTS["lr"], help="Learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULTS["batch_size"],
        help="Batch size",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=DEFAULTS["latent_dim"],
        help="Size of the latent space",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=DEFAULTS["image_size"],
        help="Image resolution (square)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=DEFAULTS["resume"],
        help="Path to a checkpoint to resume training from",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=DEFAULTS["save_every"],
        help="Save a checkpoint every N epochs",
    )

    return parser.parse_args()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:     {device}")
    print(f"Dataset:    {args.dataset}")
    print(f"Model:      {args.model}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Image size: {args.image_size}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR:         {args.lr}")
    print()

    # Load preprocessed data
    loader, num_samples = get_data_loader(args.dataset, args.batch_size)

    # Build model dynamically from --model flag
    ModelClass = get_model_class(args.model)
    model = ModelClass(
        latent_dim=args.latent_dim, image_size=args.image_size
    ).to(device)

    # One learnable latent vector per training sample
    latent_vectors = torch.randn(
        num_samples, args.latent_dim, device=device, requires_grad=True
    )

    # Optimizer covers both model weights and latent vectors
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [latent_vectors],
        lr=args.lr,
    )

    criterion = nn.MSELoss()

    # Output paths
    checkpoint_dir = os.path.join("output", args.dataset, "checkpoints")
    run_base = os.path.join("output", args.dataset, "runs")

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = load_checkpoint(
            args.resume, model, latent_vectors, optimizer
        )
        print(f"Resumed from epoch {start_epoch}")
    elif args.resume:
        print(
            f"Warning: checkpoint not found at {args.resume}, starting fresh"
        )

    # Training loop
    losses = []
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        num_batches = 0

        progress = tqdm(
            loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False
        )

        for images, indices in progress:
            images = images.to(device)
            z = latent_vectors[indices]

            # Forward pass: decode latent vectors into images
            reconstructed = model(z)

            # Reconstruction loss
            loss = criterion(reconstructed, images)

            # Backward pass: update both decoder weights and latent vectors
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            progress.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{args.epochs}  Loss: {avg_loss:.6f}")

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, latent_vectors, optimizer, epoch + 1, checkpoint_dir
            )
            print(f"  Checkpoint saved")

    # Save final run
    elapsed = time.time() - start_time
    run_path = create_run_folder(run_base)
    save_run(model, latent_vectors, run_path)
    save_meta(
        run_path,
        args,
        extra={
            "num_samples": num_samples,
            "final_loss": losses[-1] if losses else None,
            "training_time_seconds": round(elapsed, 2),
        },
    )
    plot_training_curves(losses, run_path)

    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"Training complete in {minutes}m {seconds}s")
    print(f"Run saved to {run_path}")

    # Clean up the HDF5 file handle
    loader.dataset.close()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
