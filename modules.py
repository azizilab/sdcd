"""
Contains helper modules used in the models.
"""
import scipy.sparse
import torch
import torch.nn as nn

import numpy as np


class DenseLayers(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, activation, batch_norm=False):
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
            bound = 2.0 / layer.in_features**0.5 / layer.out_features**0.5
            nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.uniform_(layer.bias, -bound, bound)


class LinearParallel(nn.Module):
    def __init__(self, in_dim, out_dim, parallel_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.parallel_dim = parallel_dim

        self.weight = nn.Parameter(torch.zeros(parallel_dim, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(parallel_dim, out_dim))
        self.reset_parameters()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, parallel_dim, in_dim)
        Returns:
            torch.Tensor: output tensor of shape (batch_size, parallel_dim, out_dim)
        """
        x = torch.einsum("npi, pio -> npo", x, self.weight) + self.bias
        return x

    @torch.no_grad()
    def reset_parameters(self):
        bound = 1.0 / self.in_dim**0.5
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def __repr__(self):
        return f"LinearParallel(in_dim={self.in_dim}, out_dim={self.out_dim}, parallel_dim={self.parallel_dim})"


class DispatcherLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, adjacency_p=2.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.adjacency_p = adjacency_p

        self.weight = nn.Parameter(torch.zeros(in_dim, out_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim, hidden_dim))
        self.reset_parameters()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, in_dim)
        Returns:
            torch.Tensor: output tensor of shape (batch_size, out_dim, hidden_dim)
        """
        x = torch.einsum("ni, ioh -> noh", x, self.weight) + self.bias
        return x

    @torch.no_grad()
    def reset_parameters(self):
        bound = 1.0 / self.in_dim / self.hidden_dim ** (1.0 / self.adjacency_p)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def get_adjacency_matrix(self):
        return torch.linalg.vector_norm(self.weight, dim=2, ord=self.adjacency_p)

    def __repr__(self):
        return f"DispatcherLayer(in_dim={self.in_dim}, out_dim={self.out_dim}, hidden_dim={self.hidden_dim}, adjacency_p={self.adjacency_p})"


class AutoEncoderLayers(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        activation=nn.ReLU(),
        shared_layers: bool = True,
        adjacency_p: float = 2.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.shared_layers = shared_layers
        self.adjacency_p = adjacency_p

        self.layers = nn.ModuleList()
        self.layers.append(
            DispatcherLayer(self.in_dim, self.in_dim, hidden_dims[0], self.adjacency_p)
        )
        self.identity = torch.eye(self.in_dim)

        # self.power_grad = PowerIterationGradient(self)
        self.power_grad = SCCPowerIteration(self, 1000)

        # if layers are shared, use regular dense layers
        # else use parallel layers
        if shared_layers:
            dims = self.hidden_dims
            for i in range(len(dims) - 1):
                self.layers.append(nn.Linear(dims[i], dims[i + 1]))
        else:
            dims = self.hidden_dims
            for i in range(len(dims) - 1):
                self.layers.append(LinearParallel(dims[i], dims[i + 1], self.in_dim))

        self.reset_parameters()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, in_dim)
        Returns:
            torch.Tensor: output tensor of shape (batch_size, out_dim, hidden_dim[-1])
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)

        return x

    def get_adjacency_matrix(self):
        return self.layers[0].get_adjacency_matrix()

    @torch.no_grad()
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def reconstruction_loss(self, x):
        x_mean = self(x).squeeze(2)
        nll = ((x_mean - x) ** 2).sum()
        # we normalize by the number of samples (but ideally we shouldn't, as it mess up
        # with the L1 and L2 regularization scales)
        nll /= x.shape[0]
        return nll

    def l1_reg_dispatcher(self):
        # maybe change to abs of the collapsed weights (sum over hidden dim)
        return torch.sum(torch.abs(self.layers[0].weight))

    def l2_reg_all_weights(self):
        return sum([torch.sum(p**2) for p in self.parameters()])

    def dag_reg(self):
        A = self.get_adjacency_matrix() ** 2
        h = -torch.slogdet(self.identity - A)[1]
        return h

    def dag_reg_power_grad(self):
        grad, A = self.power_grad.compute_gradient()
        # with torch.no_grad():
        #     grad = grad - A * (grad * A).sum() / ((A**2).sum() + 1e-6) / 2
        # grad = grad + torch.eye(self.in_dim)
        h_val = (grad.detach() * A).sum()
        return h_val

    def loss(self, x, alpha=1.0, beta=1.0, gamma=1.0, n_observations=None):
        nll = self.reconstruction_loss(x)
        l1_reg = alpha * self.l1_reg_dispatcher()  # * n_obs_norm
        l2_reg = beta * self.l2_reg_all_weights()  # * n_obs_norm
        # mu = 1 / gamma
        # dag_reg = self.dag_reg()
        dag_reg = self.dag_reg_power_grad()
        obj = nll + l1_reg + l2_reg + gamma * dag_reg
        # if np.random.rand() < 0.01:
        #     print(
        #         f"nll: {nll.item():.3f}, l1: {l1_reg.item():.3f}, l2: {l2_reg.item():.3f}, dag: {dag_reg.item():.3f}"
        #     )

        return obj


def normalize(v):
    return v / torch.linalg.vector_norm(v)


class SCCPowerIteration:
    def __init__(self, model: "AutoEncoderLayers", update_scc_freq=100):
        self.d = model.in_dim
        self.update_scc_freq = update_scc_freq
        self.get_matrix = model.get_adjacency_matrix

        self.scc_list = None
        self.update_scc()

        self.v = None
        self.vt = None
        self.initialize_eigenvectors()

        self.n_updates = 0

    def initialize_eigenvectors(self):
        self.v, self.vt = torch.ones(size=(2, self.d))
        self.v = normalize(self.v)
        self.vt = normalize(self.vt)
        return self.power_iteration(5)

    def update_scc(self):
        n_components, labels = scipy.sparse.csgraph.connected_components(
            csgraph=scipy.sparse.coo_matrix(self.get_matrix().detach().numpy()),
            directed=True,
            return_labels=True,
            connection="strong",
        )
        self.scc_list = []
        for i in range(n_components):
            scc = np.where(labels == i)[0]
            self.scc_list.append(scc)
        print(len(self.scc_list))

    def power_iteration(self, n_iter=5):
        matrix = self.get_matrix() ** 2
        for scc in self.scc_list:
            if len(scc) == self.d:
                scc = slice(None)
            sub_matrix = matrix[scc][:, scc]
            v = self.v[scc]
            vt = self.vt[scc]
            for i in range(n_iter):
                v = normalize(sub_matrix.mv(v) + 1e-6 * v.sum())
                vt = normalize(sub_matrix.T.mv(vt) + 1e-6 * vt.sum())
            self.v[scc] = v
            self.vt[scc] = vt

        return matrix

    def compute_gradient(self):
        if self.n_updates % self.update_scc_freq == 0:
            self.update_scc()
            self.initialize_eigenvectors()

        # matrix = self.power_iteration(4)
        matrix = self.initialize_eigenvectors()

        gradient = torch.zeros(size=(self.d, self.d))
        for scc in self.scc_list:
            if len(scc) == self.d:
                scc = slice(None)
            v = self.v[scc]
            vt = self.vt[scc]
            gradient[scc][:, scc] = torch.outer(vt, v) / torch.inner(vt, v)

        gradient += torch.eye(self.d)
        # gradient += matrix.T

        self.n_updates += 1

        return gradient, matrix


class PowerIterationGradient:
    def __init__(self, model: "AutoEncoderLayers"):
        self.model = model
        self.d = model.in_dim

        self.u = None
        self.v = None

        self.init_eigenvect()

    def init_eigenvect(self):
        self.u, self.v = torch.ones(size=(2, self.d))
        self.u = self.u / torch.linalg.vector_norm(self.u)
        self.v = self.v / torch.linalg.vector_norm(self.v)
        self.iterate(5)

    def iterate(self, n=2, A=None):
        with torch.no_grad():
            A = self.model.get_adjacency_matrix() ** 2 if A is None else A
            A = A + 1e-6
            for _ in range(n):
                self.one_iteration(A)

    def one_iteration(self, A):
        """One iteration of power method"""
        self.u = normalize(A.T @ self.u)
        self.v = normalize(A @ self.v)

    def compute_gradient(self):
        """Gradient eigenvalue"""
        A = self.model.get_adjacency_matrix() ** 2
        # self.iterate(4, A)
        self.init_eigenvect()
        grad = self.u[:, None] @ self.v[None] / (self.u.dot(self.v) + 1e-6)
        grad += torch.eye(self.d)
        grad += A.T
        return grad, A
