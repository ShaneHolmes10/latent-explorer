import importlib
import os
import yaml
import torch
from datetime import datetime

"""
Utilities for dynamic model loading, checkpoint management, and
training run organization. Models are loaded via importlib using
the same naming convention as the motor babbling project.
"""


def get_model_class(model_name):
    """
    Dynamically import and return a model class from the models package.

    Naming convention:
        "decoder"     -> models.decoder.Decoder
        "vae_decoder" -> models.vae_decoder.VaeDecoder

    @param model_name Lowercase string identifier for the model.
    @return The model class, ready to be instantiated.
    @raises ValueError if the module or class cannot be found.
    """
    try:
        module = importlib.import_module(f"models.{model_name}")
        class_name = model_name.title().replace("_", "")
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not load model '{model_name}': {e}")


def create_run_folder(base_path):
    """
    Create a timestamped run folder and return its path.

    @param base_path Base directory where runs are stored.
    @return Full path to the new run folder.
    """
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
    run_path = os.path.join(base_path, timestamp)
    os.makedirs(run_path, exist_ok=True)
    return run_path


def save_meta(run_path, args, extra=None):
    """
    Save a meta.yaml alongside the model weights.

    @param run_path Path to the run folder.
    @param args Parsed argparse namespace with training configuration.
    @param extra Optional dict of additional metadata to include.
    """
    meta = {
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "model": args.model,
        "latent_dim": args.latent_dim,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }

    if extra:
        meta.update(extra)

    meta_path = os.path.join(run_path, "meta.yaml")
    with open(meta_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False)


def save_checkpoint(model, latent_vectors, optimizer, epoch, checkpoint_dir):
    """
    Save a resumable training checkpoint.

    @param model The decoder model.
    @param latent_vectors The learnable latent vectors tensor.
    @param optimizer The optimizer.
    @param epoch Current epoch number.
    @param checkpoint_dir Directory to save the checkpoint in.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "latent_vectors": latent_vectors.data,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(path, model, latent_vectors=None, optimizer=None):
    """
    Load a training checkpoint.

    @param path Path to the checkpoint file.
    @param model The decoder model to load weights into.
    @param latent_vectors Optional latent vectors tensor to restore.
    @param optimizer Optional optimizer to restore state into.
    @return The epoch number the checkpoint was saved at.
    """
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if latent_vectors is not None and "latent_vectors" in checkpoint:
        latent_vectors.data = checkpoint["latent_vectors"]

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint.get("epoch", 0)


def save_run(model, latent_vectors, run_path):
    """
    Save the final model weights and latent vectors to a run folder.

    @param model The decoder model.
    @param latent_vectors The learned latent vectors tensor.
    @param run_path Path to the run folder.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "latent_vectors": latent_vectors.data,
        },
        os.path.join(run_path, "model.pt"),
    )
