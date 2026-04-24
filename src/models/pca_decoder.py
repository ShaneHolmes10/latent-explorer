import math
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np

from models.decoder import Decoder

"""
PCA Decoder model. Wraps a pretrained Decoder with a PCA inverse
transform so that inputs are principal component coordinates rather
than raw latent vectors. Components are ordered by explained variance
(D00 = highest variance, D01 = next highest, etc).

This model is not trained directly. Instead, use build_pca_decoder.py
to create a .pt file from a trained decoder checkpoint.
"""


class PcaDecoder(nn.Module):
    """
    Takes PCA coordinates as input, inverse transforms them to the
    original latent space, and passes them through a frozen pretrained
    decoder to produce an image.
    """

    def __init__(self, latent_dim=80, image_size=128):
        """
        @param latent_dim Number of PCA components (input dimensions).
        @param image_size Target output resolution (must match the pretrained decoder).
        """

        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Inner decoder that does the actual image generation
        self.decoder = Decoder(latent_dim=latent_dim, image_size=image_size)

        # PCA parameters stored as buffers so they get saved/loaded with state_dict
        # pca_components: (latent_dim, latent_dim) rotation matrix
        # pca_mean: (latent_dim,) mean of the latent vectors before PCA
        self.register_buffer(
            "pca_components", torch.zeros(latent_dim, latent_dim)
        )
        self.register_buffer("pca_mean", torch.zeros(latent_dim))

    def fit_pca(self, latent_vectors):
        """
        Fit PCA on the trained latent vectors and store the transform
        parameters. Components are automatically sorted by explained
        variance (highest first).

        @param latent_vectors Tensor of shape (num_samples, latent_dim).
        @return PCA transformed latent vectors as a tensor.
        """

        data = latent_vectors.detach().cpu().numpy()

        pca = PCA(n_components=self.latent_dim)
        transformed = pca.fit_transform(data)

        # Store PCA parameters as buffers
        self.pca_components.copy_(
            torch.tensor(pca.components_, dtype=torch.float32)
        )
        self.pca_mean.copy_(torch.tensor(pca.mean_, dtype=torch.float32))

        variance_explained = np.sum(pca.explained_variance_ratio_) * 100
        print(
            f"PCA fitted: {self.latent_dim} components explain {variance_explained:.1f}% of variance"
        )

        # Print top 10 component variance contributions
        for i in range(min(10, self.latent_dim)):
            ratio = pca.explained_variance_ratio_[i] * 100
            print(f"  PC{i:02d}: {ratio:.2f}% variance")

        return torch.tensor(transformed, dtype=torch.float32)

    def inverse_pca(self, z_pca):
        """
        Convert PCA coordinates back to the original latent space.

        @param z_pca Tensor of shape (batch, latent_dim) in PCA space.
        @return Tensor of shape (batch, latent_dim) in original latent space.
        """

        # z_latent = z_pca @ components + mean
        return z_pca @ self.pca_components + self.pca_mean

    def forward(self, z_pca):
        """
        Generate an image from PCA coordinates.

        @param z_pca Tensor of shape (batch, latent_dim) in PCA space.
        @return Tensor of shape (batch, 3, image_size, image_size) with values in [0, 1].
        """

        # Transform from PCA space back to the decoder's latent space
        z_latent = self.inverse_pca(z_pca)

        # Generate image with the frozen decoder
        return self.decoder(z_latent)
