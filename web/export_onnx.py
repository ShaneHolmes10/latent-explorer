import argparse
import json
import os
import sys

import torch

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, _src)

from config import DEFAULTS  # noqa: E402
from utils.model_utils import get_model_class  # noqa: E402

"""
Export a trained .pt model to ONNX format for web browser inference.
Writes two files to web/element/models/:
  - <run>_<stem>.onnx        model weights only (no latent vectors)
  - <run>_<stem>_stats.json  per-dim means and stds for slider ranges

Also updates web/element/models/models.json so the browser dropdown
picks up the new model automatically.

Usage:
  # Export a pca_decoder (recommended - meaningful slider dimensions)
  python web/export_onnx.py \\
    --input output/faces/runs/2026_04_21_0622/pca_model.pt \\
    --model pca_decoder \\
    --name "Run Apr 21"

  # Export a raw decoder
  python web/export_onnx.py \\
    --input output/faces/runs/2026_04_21_0622/model.pt \\
    --model decoder \\
    --name "Run Apr 21 (raw)"

  # If you only have model.pt and want pca_decoder, run this first:
  python src/utils/build_pca_decoder.py \\
    --input output/faces/runs/2026_04_21_0622/model.pt \\
    --output output/faces/runs/2026_04_21_0622/pca_model.pt
"""

# Example Usage:
# python web/export_onnx.py \
#  --input output/faces/runs/2026_04_21_0622/pca_model.pt \
#  --model pca_decoder \
#  --name "PCA Decoder" \
#  --init-image 10624


_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_MODELS_DIR = os.path.join(_REPO_ROOT, "web", "element", "models")
_MANIFEST = os.path.join(_MODELS_DIR, "models.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a trained .pt model to ONNX for browser inference"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to trained .pt file (model weights + latent vectors)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULTS["model"],
        help=(
            "Model architecture: decoder or pca_decoder"
            " (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Display name shown in the browser dropdown",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=DEFAULTS["latent_dim"],
        help=(
            "Latent space size — must match the trained model"
            " (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=DEFAULTS["image_size"],
        help=(
            "Output image resolution — must match the trained model"
            " (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=_MODELS_DIR,
        help="Directory to write .onnx and _stats.json",
    )
    parser.add_argument(
        "--init-image",
        type=int,
        default=None,
        help=(
            "Index of the training image whose latent vector becomes the"
            " model's default (used on load and Reset)"
        ),
    )
    return parser.parse_args()


def export(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Always export on CPU — the browser runs on CPU anyway
    device = torch.device("cpu")

    # Load model architecture and weights
    print(f"Loading {args.model} from {args.input} ...")
    ModelClass = get_model_class(args.model)
    model = ModelClass(
        latent_dim=args.latent_dim, image_size=args.image_size
    )

    saved = torch.load(args.input, weights_only=False, map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    # Derive output base name from input path:
    # .../runs/2026_04_21_0622/pca_model.pt -> 2026_04_21_0622_pca_model
    run_dir = os.path.basename(
        os.path.dirname(os.path.abspath(args.input))
    )
    stem = os.path.splitext(os.path.basename(args.input))[0]
    base = f"{run_dir}_{stem}"

    onnx_path = os.path.join(args.output_dir, f"{base}.onnx")
    stats_path = os.path.join(args.output_dir, f"{base}_stats.json")

    # Export ONNX — bakes all model weights and PCA params as constants
    print(f"Exporting ONNX -> {onnx_path}")
    dummy = torch.zeros(1, args.latent_dim)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["z"],
        output_names=["image"],
        dynamic_axes={"z": {0: "batch"}, "image": {0: "batch"}},
        opset_version=17,
    )

    # Compute per-dimension stats from the training latent vectors.
    # These drive the slider ranges and the Random button distribution.
    latent_vectors = saved["latent_vectors"].to(device)
    means = latent_vectors.mean(dim=0).tolist()
    stds = latent_vectors.std(dim=0).tolist()

    print(f"Exporting stats  -> {stats_path}")
    with open(stats_path, "w") as f:
        json.dump(
            {"latent_dim": args.latent_dim, "means": means, "stds": stds},
            f,
        )

    # Optionally export a default init vector for this model
    init_path = None
    if args.init_image is not None:
        init_path = os.path.join(args.output_dir, f"{base}_init.json")
        init_vec = latent_vectors[args.init_image].tolist()
        print(f"Exporting init   -> {init_path}  (image #{args.init_image})")
        with open(init_path, "w") as f:
            json.dump(init_vec, f)

    # Update manifest so the browser dropdown picks up this model
    manifest = []
    if os.path.exists(_MANIFEST):
        with open(_MANIFEST) as f:
            manifest = json.load(f)

    # Remove any existing entry with the same display name
    manifest = [m for m in manifest if m["name"] != args.name]

    # Paths are relative to web/element/ (where index.html lives)
    manifest_dir = os.path.dirname(_MODELS_DIR)
    entry = {
        "name": args.name,
        "onnx": os.path.relpath(onnx_path, manifest_dir).replace("\\", "/"),
        "stats": os.path.relpath(stats_path, manifest_dir).replace("\\", "/"),
    }
    if init_path is not None:
        entry["init"] = (
            os.path.relpath(init_path, manifest_dir).replace("\\", "/")
        )
    manifest.append(entry)

    with open(_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Updated manifest -> {_MANIFEST}")
    print()
    print("To test in the browser:")
    print("  python -m http.server 8000 --directory web/element")
    print("  Open http://localhost:8000")


def main():
    args = parse_args()
    export(args)


if __name__ == "__main__":
    main()
