"""
Contains helper modules used in the models.
"""
from typing import Literal

import scipy.sparse
import torch
import torch.nn as nn

import numpy as np


def get_activation(activation: Literal["relu", "sigmoid", "tanh", "linear"]):
    """
    Args:
        activation (str): activation function name
    Returns:
        torch.nn.Module: activation function
    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "linear":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation function: {activation}")


class DenseLayers(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, activation, batch_norm=False):
        super().__init__()
        self.activation = get_activation(activation) if type(activation) == str else activation
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
    def reset_parameters(self, scale=1.0):
        for layer in self.layers:
            if layer.in_features == 0 or layer.out_features == 0:
                continue
            bound = 2.0 / layer.in_features**0.5 / layer.out_features**0.5 * scale
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
    def __init__(self, in_dim, out_dim, hidden_dim, adjacency_p=2.0, mask=None, warmstart=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.adjacency_p = adjacency_p

        if mask is not None and not warmstart:
            self.register_buffer("mask", torch.tensor(mask).float())
        else:
            self.register_buffer("mask", torch.ones((in_dim, out_dim)))

        if mask is not None and warmstart:
            warmstart_tensor = 0.3 * torch.tensor(mask).unsqueeze(-1).repeat((1, 1, hidden_dim))
            self._weight = nn.Parameter(warmstart_tensor)
        else:
            self._weight = nn.Parameter(torch.zeros(in_dim, out_dim, hidden_dim))

        self.bias = nn.Parameter(torch.zeros(out_dim, hidden_dim))
        self.reset_parameters()

    @property
    def weight(self):
        if self.mask is not None:
            return self._weight * self.mask[:, :, None]
        return self._weight

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
        return (
            f"DispatcherLayer("
            f"in_dim={self.in_dim}, "
            f"out_dim={self.out_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"adjacency_p={self.adjacency_p}"
            f")"
        )


class AutoEncoderLayers(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        activation=nn.ReLU(),
        shared_layers: bool = True,
        adjacency_p: float = 2.0,
        mask=None,
        warmstart=False,
        dag_penalty_flavor: Literal["scc", "power_iteration", "logdet", "none"] = "scc",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.shared_layers = shared_layers
        self.adjacency_p = adjacency_p

        self.dag_penalty_flavor = dag_penalty_flavor
        if dag_penalty_flavor == "none":
            # Need to mask out identity to prevent learning self-loops
            if mask is not None:
                mask = (mask.astype(bool) & (1 - np.eye(self.in_dim)).astype(bool)).astype(int)
            else:
                mask = 1 - np.eye(self.in_dim)

        self.layers = nn.ModuleList()
        self.layers.append(
            DispatcherLayer(
                self.in_dim,
                self.in_dim,
                hidden_dims[0],
                adjacency_p=self.adjacency_p,
                mask=mask,
                warmstart=warmstart,
            )
        )

        if dag_penalty_flavor == "scc":
            self.power_grad = SCCPowerIteration(self.get_adjacency_matrix(), self.in_dim, 1000)
        elif dag_penalty_flavor == "power_iteration":
            self.power_grad = PowerIterationGradient(self.get_adjacency_matrix(), self.in_dim)

        self.identity = torch.eye(self.in_dim)

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

    @property
    def device(self):
        return next(self.parameters()).device

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

    def reconstruction_loss(self, x, interventions=None):
        x_mean = self(x).squeeze(2)
        if interventions is not None:
            interventions[torch.where(interventions == -1)] = x.shape[1]
            interventions_oh = nn.functional.one_hot(
                interventions.squeeze(), num_classes=x.shape[1] + 1
            )
            interventions_oh = interventions_oh[:, :-1]  # cutoff obs
            mask_interventions_oh = 1 - interventions_oh
            nll = (mask_interventions_oh * (x_mean - x) ** 2).sum()
        else:
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
        grad, A = self.power_grad.compute_gradient(self.get_adjacency_matrix())
        # with torch.no_grad():
        #     grad = grad - A * (grad * A).sum() / ((A**2).sum() + 1e-6) / 2
        # grad = grad + torch.eye(self.in_dim)
        h_val = (grad.detach() * A).sum()
        return h_val

    def loss(
        self,
        x,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        n_observations=None,
        interventions=None,
        return_detailed_losses=False,
    ):
        nll = self.reconstruction_loss(x, interventions=interventions)
        l1_reg = alpha * self.l1_reg_dispatcher()  # * n_obs_norm
        l2_reg = beta * self.l2_reg_all_weights()  # * n_obs_norm
        # mu = 1 / gamma
        if self.dag_penalty_flavor == "logdet":
            dag_reg = self.dag_reg()
        elif self.dag_penalty_flavor in ("scc", "power_iteration"):
            dag_reg = self.dag_reg_power_grad()
        elif self.dag_penalty_flavor == "none":
            dag_reg = torch.zeros(1)
        dag_reg = dag_reg.to(self.device)

        total_loss = nll + l1_reg + l2_reg + gamma * dag_reg

        if return_detailed_losses:
            return total_loss, {
                "nll": nll.detach(),
                "l1": l1_reg.detach(),
                "l2": l2_reg.detach(),
                "dag": dag_reg.detach(),
            }
        else:
            return total_loss


def normalize(v):
    return v / torch.linalg.vector_norm(v)


class SCCPowerIteration(nn.Module):
    def __init__(self, init_adj_mtx, d, update_scc_freq=1000):
        super().__init__()
        self.d = d
        self.update_scc_freq = update_scc_freq

        self._dummy_param = nn.Parameter(torch.empty(1), requires_grad=False) # Used to track device

        self.scc_list = None
        self.update_scc(init_adj_mtx)

        self.register_buffer("v", None)
        self.register_buffer("vt", None)
        self.initialize_eigenvectors(init_adj_mtx)

        self.n_updates = 0

    @property
    def device(self):
        return self._dummy_param.device

    def initialize_eigenvectors(self, adj_mtx):
        self.v, self.vt = torch.ones(size=(2, self.d), device=self.device)
        self.v = normalize(self.v)
        self.vt = normalize(self.vt)
        return self.power_iteration(adj_mtx, 5)

    def update_scc(self, adj_mtx):
        n_components, labels = scipy.sparse.csgraph.connected_components(
            csgraph=scipy.sparse.coo_matrix(adj_mtx.cpu().detach().numpy()),
            directed=True,
            return_labels=True,
            connection="strong",
        )
        self.scc_list = []
        for i in range(n_components):
            scc = np.where(labels == i)[0]
            self.scc_list.append(scc)
        # print(len(self.scc_list))

    def power_iteration(self, adj_mtx, n_iter=5):
        matrix = adj_mtx ** 2
        for scc in self.scc_list:
            if len(scc) == self.d:
                sub_matrix = matrix
                v = self.v
                vt = self.vt
                for i in range(n_iter):
                    v = normalize(sub_matrix.mv(v) + 1e-6 * v.sum())
                    vt = normalize(sub_matrix.T.mv(vt) + 1e-6 * vt.sum())
                self.v = v
                self.vt = vt

            else:
                sub_matrix = matrix[scc][:, scc]
                v = self.v[scc]
                vt = self.vt[scc]
                for i in range(n_iter):
                    v = normalize(sub_matrix.mv(v) + 1e-6 * v.sum())
                    vt = normalize(sub_matrix.T.mv(vt) + 1e-6 * vt.sum())
                self.v[scc] = v
                self.vt[scc] = vt

        return matrix

    def compute_gradient(self, adj_mtx):
        if self.n_updates % self.update_scc_freq == 0:
            self.update_scc(adj_mtx)
            self.initialize_eigenvectors(adj_mtx)

        # matrix = self.power_iteration(4)
        matrix = self.initialize_eigenvectors(adj_mtx)

        gradient = torch.zeros(size=(self.d, self.d), device=self.device)
        for scc in self.scc_list:
            if len(scc) == self.d:
                v = self.v
                vt = self.vt
                gradient = torch.outer(vt, v) / torch.inner(vt, v)
            else:
                v = self.v[scc]
                vt = self.vt[scc]
                gradient[scc][:, scc] = torch.outer(vt, v) / torch.inner(vt, v)

        gradient += 100 * torch.eye(self.d, device=self.device)
        # gradient += matrix.T

        self.n_updates += 1

        return gradient, matrix


class PowerIterationGradient(nn.Module):
    def __init__(self, init_adj_mtx, d):
        super().__init__()
        self.d = d

        self._dummy_param = nn.Parameter(torch.empty(1), requires_grad=False) # Used to track device

        self.register_buffer("u", None) 
        self.register_buffer("v", None)

        self.init_eigenvect(init_adj_mtx)

    @property
    def device(self):
        return self._dummy_param.device

    def init_eigenvect(self, adj_mtx):
        self.u, self.v = torch.ones(size=(2, self.d), device=self.device)
        self.u = self.u / torch.linalg.vector_norm(self.u)
        self.v = self.v / torch.linalg.vector_norm(self.v)
        self.iterate(adj_mtx, 5)

    def iterate(self, adj_mtx, n=2):
        with torch.no_grad():
            A = adj_mtx + 1e-6
            for _ in range(n):
                self.one_iteration(A)

    def one_iteration(self, A):
        """One iteration of power method"""
        self.u = normalize(A.T @ self.u)
        self.v = normalize(A @ self.v)

    def compute_gradient(self, adj_mtx):
        """Gradient eigenvalue"""
        A = adj_mtx ** 2
        # self.iterate(4, A)
        self.init_eigenvect(adj_mtx)
        grad = self.u[:, None] @ self.v[None] / (self.u.dot(self.v) + 1e-6)
        # grad += torch.eye(self.d)
        # grad += A.T
        return grad, A
