import functools
import math

import torch.nn.functional
from torch import nn
from torch.distributions import Normal, NegativeBinomial
import torch.distributions as dist

from torch.utils.data import DataLoader

from modules import DenseLayers


class AutoEncoder(nn.Module):
    """
    Autoencoder with Gaussian response function
    """

    def __init__(
        self,
        in_dim,
        latent_dim,
        hidden_dims_encoder,
        hidden_dims_decoder,
        activation: nn.Module = nn.ReLU(),
        adjacency_p=2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.activation = activation
        self.adjacency_p = adjacency_p
        self.encoder = DenseLayers(
            self.in_dim, self.latent_dim, hidden_dims_encoder, activation, batch_norm=False
        )

        self.decoder = DenseLayers(
            self.latent_dim, self.in_dim, hidden_dims_decoder, activation, batch_norm=False
        )

        # Initialize feature's specific parameter: scale, dispersion ...
        self._feature_specific = nn.Parameter(torch.randn(self.in_dim))
        self.epsilon = 1e-8

    def get_adjacency_matrices(self, p=None):
        if p is None:
            p = self.adjacency_p
        ws_xz = self.encoder.get_weight_matrices()
        ws_zx = self.decoder.get_weight_matrices()

        # absolute value of the weight matrices
        a_xz = [torch.abs(w) ** p for w in ws_xz]
        a_zx = [torch.abs(w) ** p for w in ws_zx]

        return a_xz, a_zx

    def get_adjacency_matrix(self, mode="reverse", p=None):
        if p is None:
            p = self.adjacency_p
        a_xz, a_zx = self.get_adjacency_matrices(p)
        a_xz = functools.reduce(torch.matmul, a_xz)
        a_zx = functools.reduce(torch.matmul, a_zx)

        if mode == "reverse":
            return torch.matmul(a_zx, a_xz)
        else:
            return torch.matmul(a_xz, a_zx)

    def dag_loss(self, p=2):
        """
        Compute the DAG loss for the encoder and decoder.
        """
        A = self.get_adjacency_matrix("reverse", p=p)
        d = A.shape[0]
        s = 2.0
        # set s to the largest absolute value of eigenvalue of A
        # s = torch.max(torch.abs(torch.linalg.eigvals(A))).detach() + 0.1

        h = -torch.slogdet(s * torch.eye(d) - A).logabsdet + d * math.log(s)

        return h

    def l1_reg(self):
        """
        Compute the L1 regularization term for the encoder and decoder.
        """
        a_xz, a_zx = self.get_adjacency_matrices()
        return sum([a.sum() for a in a_xz]) + sum([a.sum() for a in a_zx])

    def l2_reg(self):
        """
        Compute the L2 regularization term for the encoder and decoder.
        Note: it is not applied to the bias terms.
        """
        a_xz, a_zx = self.get_adjacency_matrices()
        return sum([a.pow(2).sum() for a in a_xz]) + sum([a.pow(2).sum() for a in a_zx])

    @property
    def feature_specific(self):
        return torch.nn.functional.softplus(self._feature_specific)

    def forward(self, x):
        """
        Compute the reconstruction for a batch of data
        """
        z = self.encoder(x)
        z = self.activation(z)
        x_hat = self.decoder(z)

        return x_hat

    def loss(self, x, alpha=1.0, beta=1.0, gamma=1.0, n_observations=None):
        """
        Compute the loss for a batch of data
        """
        reconstruction_loss = self.reconstruction_loss(x)
        dag_loss = self.dag_loss(p=self.adjacency_p)
        l1_reg = self.l1_reg()
        l2_reg = self.l2_reg()

        if n_observations is not None:
            reconstruction_loss = reconstruction_loss / x.shape[0]
            dag_loss = dag_loss
            l1_reg = l1_reg / n_observations
            l2_reg = l2_reg / n_observations
            # l1_reg = l1_reg
            # l2_reg = l2_reg

        return reconstruction_loss + gamma * dag_loss + alpha * l1_reg + beta * l2_reg
        # return dag_loss + 1 / gamma * (reconstruction_loss + alpha * l1_reg + beta * self.l2_reg())

    def reconstruction_loss(self, x):
        """
        Compute the reconstruction loss for a batch of data
        """
        x_hat = self.forward(x)
        reconstruction_loss = torch.sum((x - x_hat) ** 2)
        return reconstruction_loss

