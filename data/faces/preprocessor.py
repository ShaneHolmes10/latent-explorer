import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from base_preprocessor import BasePreprocessor

# Example usage
# python data/faces/preprocessor.py --image_size 128


class Preprocessor(BasePreprocessor):
    """
    Preprocessor for the CelebA faces dataset.
    Center crops to square, then resizes.
    """

    def process_image(self, img):
        img = self.center_crop_square(img)
        img = self.resize(img)
        return img


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess faces dataset")
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Target image resolution (square)",
    )

    args = parser.parse_args()

    preprocessor = Preprocessor("faces", image_size=args.image_size)
    preprocessor.run()
