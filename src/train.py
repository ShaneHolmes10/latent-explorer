import argparse
from src.config import DEFAULTS

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
 
    parser.add_argument("--dataset", type=str, default=DEFAULTS["dataset"],
                        help="Which dataset to use (e.g. faces, cars)")
    parser.add_argument("--model", type=str, default=DEFAULTS["model"],
                        help="Which model architecture to use (e.g. decoder, vae_decoder)")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"],
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"],
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"],
                        help="Batch size")
    parser.add_argument("--latent_dim", type=int, default=DEFAULTS["latent_dim"],
                        help="Size of the latent space")
    parser.add_argument("--image_size", type=int, default=DEFAULTS["image_size"],
                        help="Image resolution (square)")
    parser.add_argument("--resume", type=str, default=DEFAULTS["resume"],
                        help="Path to a checkpoint to resume training from")
    parser.add_argument("--save_every", type=int, default=DEFAULTS["save_every"],
                        help="Save a checkpoint every N epochs")
 
    return parser.parse_args()
 
 
def main():
    args = parse_args()

  
 
if __name__ == "__main__":
    main()