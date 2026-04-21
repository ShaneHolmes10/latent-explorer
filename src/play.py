import argparse
import os
import sys
import tkinter as tk
from tkinter import ttk
import torch
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULTS
from utils.model_utils import get_model_class

"""
Interactive GUI for exploring a trained model's latent space.
Displays a generated image on the left and sliders for each input
dimension on the right. Moving sliders updates the image in real time.
Supports any model that takes a vector input and outputs an image.
"""

# Example usage:
# python src/play.py --trained output/faces/runs/2026_04_21_0622/model.pt --model decoder
# python src/play.py --trained output/faces/runs/2026_04_21_0622/model.pt --model pca_decoder --components 20


def parse_args():
    """Parse command line arguments for the play GUI."""

    parser = argparse.ArgumentParser(
        description="Explore a model's latent space interactively"
    )

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
        help="Which model architecture to use (e.g. decoder, pca_decoder)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULTS["dataset"],
        help="Which dataset (used for loading latent vector statistics)",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=DEFAULTS["latent_dim"],
        help="Size of the latent space (must match trained model)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=DEFAULTS["image_size"],
        help="Image resolution (must match trained model)",
    )
    parser.add_argument(
        "--display_size",
        type=int,
        default=384,
        help="Size of the displayed image in the GUI",
    )
    parser.add_argument(
        "--num_std",
        type=float,
        default=3.0,
        help="Number of standard deviations for slider range",
    )

    return parser.parse_args()


def load_model_and_stats(args):
    """
    Load the trained model and compute per dimension statistics
    from the latent vectors for slider ranges.

    @param args Parsed argparse namespace.
    @return Tuple of (model, means, stds, device).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ModelClass = get_model_class(args.model)
    model = ModelClass(latent_dim=args.latent_dim, image_size=args.image_size)

    checkpoint = torch.load(
        args.trained, weights_only=False, map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    # Compute per dimension mean and std from the trained latent vectors
    latent_vectors = checkpoint["latent_vectors"].to(device)
    means = latent_vectors.mean(dim=0).cpu().numpy()
    stds = latent_vectors.std(dim=0).cpu().numpy()

    return model, means, stds, device


def generate_image(model, values, device, display_size):
    """
    Run a vector through the model and return a PIL Image for display.

    @param model Trained model (decoder or pca_decoder).
    @param values Numpy array of slider values.
    @param device Torch device.
    @param display_size Target display resolution for the GUI.
    @return PIL Image ready for tkinter display.
    """

    z = torch.tensor(values, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        img = model(z)

    # Convert from (1, 3, H, W) to (H, W, 3) uint8
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    # Scale up for display
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((display_size, display_size), Image.NEAREST)

    return pil_img


class PlayGUI:
    """
    Main GUI window. Left side displays the generated image,
    right side has scrollable sliders for each input dimension
    plus random and reset buttons.
    """

    def __init__(self, model, means, stds, device, args):
        self.model = model
        self.means = means
        self.stds = stds
        self.device = device
        self.display_size = args.display_size
        self.num_std = args.num_std
        self.num_dims = args.latent_dim
        self.slider_resolution = 1000

        # Current slider values, initialized to means
        self.values = means.copy()

        # Build the window
        self.root = tk.Tk()
        self.root.title(
            f"latent explorer  |  {args.model}  |  {args.latent_dim}D"
        )
        self.root.configure(bg="#1e1e1e")

        self.build_layout()
        self.update_image()

    def build_layout(self):
        """Construct the GUI layout: image on left, sliders on right."""

        main_frame = tk.Frame(self.root, bg="#1e1e1e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left: image display
        self.image_frame = tk.Frame(main_frame, bg="#1e1e1e")
        self.image_frame.pack(side=tk.LEFT, padx=(0, 10))

        self.image_label = tk.Label(self.image_frame, bg="#1e1e1e")
        self.image_label.pack()

        # Buttons below image
        button_frame = tk.Frame(self.image_frame, bg="#1e1e1e")
        button_frame.pack(pady=(10, 0))

        random_btn = tk.Button(
            button_frame,
            text="Random",
            command=self.on_random,
            bg="#3a3a3a",
            fg="white",
            activebackground="#505050",
            activeforeground="white",
            width=10,
        )
        random_btn.pack(side=tk.LEFT, padx=5)

        reset_btn = tk.Button(
            button_frame,
            text="Reset",
            command=self.on_reset,
            bg="#3a3a3a",
            fg="white",
            activebackground="#505050",
            activeforeground="white",
            width=10,
        )
        reset_btn.pack(side=tk.LEFT, padx=5)

        # Right: scrollable slider panel
        slider_container = tk.Frame(main_frame, bg="#1e1e1e")
        slider_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Canvas with scrollbar for many sliders
        canvas = tk.Canvas(
            slider_container, bg="#1e1e1e", highlightthickness=0, width=300
        )
        scrollbar = ttk.Scrollbar(
            slider_container, orient=tk.VERTICAL, command=canvas.yview
        )
        self.slider_frame = tk.Frame(canvas, bg="#1e1e1e")

        self.slider_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=self.slider_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Bind mouse wheel for scrolling
        canvas.bind_all(
            "<MouseWheel>",
            lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"),
        )
        canvas.bind_all(
            "<Button-4>",
            lambda e: canvas.yview_scroll(-1, "units"),
        )
        canvas.bind_all(
            "<Button-5>",
            lambda e: canvas.yview_scroll(1, "units"),
        )

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create sliders
        self.sliders = []
        self.slider_vars = []

        for i in range(self.num_dims):
            mean = self.means[i]
            std = self.stds[i]
            low = mean - self.num_std * std
            high = mean + self.num_std * std

            frame = tk.Frame(self.slider_frame, bg="#1e1e1e")
            frame.pack(fill=tk.X, pady=1)

            label = tk.Label(
                frame,
                text=f"D{i:02d}",
                fg="#aaaaaa",
                bg="#1e1e1e",
                width=4,
                font=("Consolas", 9),
            )
            label.pack(side=tk.LEFT)

            var = tk.DoubleVar(value=mean)
            self.slider_vars.append(var)

            slider = tk.Scale(
                frame,
                from_=low,
                to=high,
                resolution=(high - low) / self.slider_resolution,
                orient=tk.HORIZONTAL,
                variable=var,
                command=lambda val, idx=i: self.on_slider_change(idx, val),
                bg="#1e1e1e",
                fg="#aaaaaa",
                troughcolor="#3a3a3a",
                highlightthickness=0,
                length=220,
                showvalue=False,
            )
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

            value_label = tk.Label(
                frame,
                textvariable=var,
                fg="#aaaaaa",
                bg="#1e1e1e",
                width=7,
                font=("Consolas", 8),
            )
            value_label.pack(side=tk.RIGHT)

            self.sliders.append(slider)

    def on_slider_change(self, idx, val):
        """Handle a slider being moved. Updates the stored value and regenerates the image."""

        self.values[idx] = float(val)
        self.update_image()

    def on_random(self):
        """Set all sliders to random values sampled from the training distribution."""

        for i in range(self.num_dims):
            val = np.random.normal(self.means[i], self.stds[i])
            self.values[i] = val
            self.slider_vars[i].set(val)
        self.update_image()

    def on_reset(self):
        """Reset all sliders to their mean values."""

        for i in range(self.num_dims):
            self.values[i] = self.means[i]
            self.slider_vars[i].set(self.means[i])
        self.update_image()

    def update_image(self):
        """Generate a new image from current slider values and display it."""

        pil_img = generate_image(
            self.model, self.values, self.device, self.display_size
        )
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.image_label.configure(image=self.tk_img)

    def run(self):
        """Start the GUI event loop."""

        self.root.mainloop()


def main():
    """Entry point. Loads model and launches the interactive GUI."""

    args = parse_args()
    model, means, stds, device = load_model_and_stats(args)
    print(f"Loaded model on {device}")
    print(f"Latent dimensions: {args.latent_dim}")
    print(f"Slider range: mean +/- {args.num_std} std")
    print("Launching GUI...")

    gui = PlayGUI(model, means, stds, device, args)
    gui.run()


if __name__ == "__main__":
    main()
