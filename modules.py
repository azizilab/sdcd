"""
Contains helper modules used in the models.
"""
import torch
import torch.nn as nn


class DenseLayers(nn.Module):
    def __init__(
        self, in_dim, out_dim, hidden_dims, activation, batch_norm=False
    ):
        super().__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        dims = [in_dim] + hidden_dims + [out_dim]
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if self.batch_norm and i < len(dims) - 2:
                self.batch_norms.append(nn.BatchNorm1d(dims[i + 1]))

        self.reset_parameters()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, in_dim)
        Returns:
            torch.Tensor: output tensor of shape (batch_size, out_dim)
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.batch_norm:
                    x = self.batch_norms[i](x)
                x = self.activation(x)

        return x

    def get_weight_matrices(self):
        return [layer.weight.T for layer in self.layers]

    @torch.no_grad()
    def reset_parameters(self):
        for layer in self.layers:
            bound = 2.0 / layer.in_features ** 0.5 / layer.out_features ** 0.5
            nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.uniform_(layer.bias, -bound, bound)


