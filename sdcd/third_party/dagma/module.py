import math

import numpy as np
import torch
import torch.nn as nn


class LocallyConnected(nn.Module):
    """Local linear layer, i.e. Conv1dLocal() with filter size 1.

    Args:
        num_linear: num of local linear layers, i.e.
        in_features: m1
        out_features: m2
        bias: whether to include bias or not

    Shape:
        - Input: [n, d, m1]
        - Output: [n, d, m2]

    Attributes:
        weight: [d, m1, m2]
        bias: [d, m2]
    """

    def __init__(self, num_linear, input_features, output_features, bias=True):
        super(LocallyConnected, self).__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(
            torch.Tensor(num_linear, input_features, output_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter("bias", None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor):
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        out = torch.matmul(input.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(dim=2)
        if self.bias is not None:
            # [n, d, m2] += [d, m2]
            out += self.bias
        return out

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return "num_linear={}, in_features={}, out_features={}, bias={}".format(
            self.num_linear, self.in_features, self.out_features, self.bias is not None
        )


class DagmaMLP(nn.Module):
    def __init__(self, dims, bias=True, dtype=torch.double):
        torch.set_default_dtype(dtype)
        super(DagmaMLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        self.dims, self.d = dims, dims[0]
        self.I = torch.eye(self.d)
        self.fc1 = nn.Linear(self.d, self.d * dims[1], bias=bias)
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(self.d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1(x)
        x = x.view(-1, self.dims[0], self.dims[1])
        for fc in self.fc2:
            x = torch.sigmoid(x)
            x = fc(x)
        x = x.squeeze(dim=2)
        return x

    def h_func(self, s=1.0):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        fc1_weight = self.fc1.weight
        fc1_weight = fc1_weight.view(self.d, -1, self.d)
        A = torch.sum(fc1_weight**2, dim=1).t()  # [i, j]
        h = -torch.slogdet(s * self.I - A)[1] + self.d * np.log(s)
        return h

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        return torch.sum(torch.abs(self.fc1.weight))

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        fc1_weight = self.fc1.weight
        fc1_weight = fc1_weight.view(self.d, -1, self.d)
        A = torch.sum(fc1_weight**2, dim=1).t()
        W = torch.sqrt(A)
        W = W.cpu().detach().numpy()  # [i, j]
        return W
