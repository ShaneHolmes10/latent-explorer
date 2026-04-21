import math
import torch
import torch.nn as nn


"""
Decoder network that maps latent vectors to images.
Takes a latent vector of configurable dimension and produces
an RGB image of configurable resolution through a series of
transposed convolution blocks.

To add a new model, create a new file in models/ following this pattern:
    - filename: <model_name>.py
    - class name: <ModelName> (PascalCase of filename)
    - must accept latent_dim and image_size in __init__
    - must implement forward(z) -> image tensor
"""


class Decoder(nn.Module):
    """
    Maps an N dimensional latent vector to an RGB image using a fully
    connected projection followed by a series of transposed convolution
    blocks. The number of upsample layers is computed dynamically from
    the target image_size.
    """

    def __init__(self, latent_dim=80, image_size=128):
        """
        @param latent_dim Dimensionality of the input latent vector.
        @param image_size Target output resolution (square). Must be a power of 2.
        """

        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Number of upsample steps needed to go from 4x4 to target size
        # 4 -> 8 -> 16 -> 32 -> 64 -> 128 = 5 steps for 128
        self.num_upsample = int(math.log2(image_size)) - 2
        self.init_size = 4
        self.init_channels = 512

        # Project latent vector to a small spatial feature map
        # Output shape: (batch, 512, 4, 4) after reshape
        self.fc = nn.Sequential(
            nn.Linear(
                latent_dim,
                self.init_channels * self.init_size * self.init_size,
            ),
            nn.ReLU(inplace=True),
        )

        # Build upsample blocks: each doubles spatial resolution and halves channels
        # Channel progression for 128px: 512 -> 256 -> 128 -> 64 -> 32 -> 3
        layers = []
        in_channels = self.init_channels
        for i in range(self.num_upsample):
            out_channels = in_channels // 2 if i < self.num_upsample - 1 else 3
            if out_channels < 3:
                out_channels = 3

            if i < self.num_upsample - 1:
                # Intermediate block: upsample + normalize + activate
                layers.extend(
                    [
                        nn.ConvTranspose2d(
                            in_channels,
                            out_channels,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    ]
                )
            else:
                # Final block: upsample to 3 channels (RGB) + sigmoid to clamp output to [0, 1]
                layers.extend(
                    [
                        nn.ConvTranspose2d(
                            in_channels, 3, kernel_size=4, stride=2, padding=1
                        ),
                        nn.Sigmoid(),
                    ]
                )

            in_channels = out_channels

        self.upsample = nn.Sequential(*layers)

    def forward(self, z):
        """
        Decode a latent vector into an RGB image.

        @param z Tensor of shape (batch, latent_dim).
        @return Tensor of shape (batch, 3, image_size, image_size) with values in [0, 1].
        """

        # Project flat latent vector to spatial feature map
        x = self.fc(z)
        x = x.view(-1, self.init_channels, self.init_size, self.init_size)

        # Progressive upsampling to target resolution
        x = self.upsample(x)
        return x
