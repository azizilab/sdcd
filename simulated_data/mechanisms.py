from typing import Union, Optional, Callable

import networkx as nx
import torch

from causal_model import CausalModel
from causal_model.mechanisms import ParametricConditionalDistribution
from modules import DenseLayers


class ParentsToChildMLP(torch.nn.Module):
    """A dense neural networks that takes as input the values of the parents of a given variable and outputs the
    parameters of the distribution of that variable.

    The parameters of the distribution are given by a dictionary of outputs with default values (for the root nodes with
    no parents).

    Args:
        parent_names: the names of the parents of the variable.
        hidden_dims: the dimensions of the hidden layers.
        outputs_with_defaults: a dictionary with the names of the outputs as keys and the default values as values.
        outputs_transform: a dictionary to optionally specify a transformation to apply to some outputs.
        activation: the activation function to use.
        extra_outputs: a dictionary with extra outputs to add to the output of the MLP.
        normalize_inputs: whether to normalize the input of each layer. This is useful with random layers since the
            non-trivial variations of random variables are around 0 (with high probability).
        kwargs: additional arguments to pass to the DenseLayers module. (e.g. bias)
    """

    def __init__(
        self,
        parent_names: list[str],
        hidden_dims: list[int],
        outputs_with_defaults: dict,
        outputs_transform: Optional[dict[str, torch.nn.Module]] = None,
        activation: Union[str, torch.nn.Module] = "relu",
        extra_outputs: Optional[dict] = None,
        normalize_inputs: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.parent_names = parent_names
        self.hidden_dims = hidden_dims
        self.outputs_with_defaults = outputs_with_defaults
        self.outputs_transform = outputs_transform
        self.activation = activation
        self.extra_outputs = extra_outputs
        self.normalize_inputs = normalize_inputs
        self.input_normalization_mean = None
        self.input_normalization_std = None

        n_parents = len(parent_names)
        self.n_parents = n_parents
        self.n_outputs = len(self.outputs_with_defaults)
        if self.n_parents == 0:
            self.mlp = None
        else:
            self.mlp = DenseLayers(
                self.n_parents, self.n_outputs, hidden_dims, activation=activation, **kwargs
            )
            self.mlp.reset_parameters_away_from_zero()

    def forward(self, parents_values):
        if self.normalize_inputs:
            if self.input_normalization_mean is None:
                # compute the mean and std of the inputs
                self.input_normalization_mean = {
                    name: value.mean() for name, value in parents_values.items()
                }
                self.input_normalization_std = {
                    name: value.std() for name, value in parents_values.items()
                }
            parents_values = {
                name: (value - self.input_normalization_mean[name])
                / self.input_normalization_std[name]
                for name, value in parents_values.items()
            }
        if len(parents_values) != self.n_parents:
            raise ValueError("Wrong number of parents")
        if self.n_parents == 0:
            return {**self.outputs_with_defaults, **self.extra_outputs}
        # we sort the keys to have a consistent order of parents
        x = torch.stack([parents_values[parent] for parent in sorted(parents_values.keys())], dim=1)
        output = self.mlp(x)
        # chunk the output into the different variables
        output = torch.split(output, 1, dim=1)
        output = {
            name: value.squeeze(1) for name, value in zip(self.outputs_with_defaults.keys(), output)
        }
        if self.outputs_transform is not None:
            for name, transform in self.outputs_transform.items():
                output[name] = transform(output[name])

        output = {**output, **self.extra_outputs}
        return output


def generate_gaussian_mlp_fixed_scale_mechanisms(
    causal_model: CausalModel,
    hidden_dims: list[int],
    default_mean: float = 0.0,
    scale: Union[float, Callable] = 1.0,
    activation: str = "relu",
    **kwargs,
):
    """Generate a dictionary of Gaussian conditional MLP mechanisms with fixed variance for each node in the graph.
    The variance can be a fixed value or a function of the depth of the node in the graph.

    Args:
        causal_model: a CausalModel object.
        hidden_dims: list of hidden dimensions for each MLP.
        default_mean: default mean of the Gaussian distribution (for root nodes).
        scale: scale of the Gaussian distribution. If it is a float, it is used as the fixed scale for all nodes. If
            it is a callable, it is used as a function of the depth of the node in the graph (the depth is defined as
            the length of the longest path from the node to a root node).
        activation: activation function for the MLP.

    Returns:
        a dictionary of mechanisms for each node in the graph.
    """
    if type(scale) not in [float, int]:
        # compute the depth of each node, start from the root nodes and go down
        node_depth = {}
        for node in nx.topological_sort(causal_model.graph):
            parents = list(causal_model.graph.predecessors(node))
            if len(parents) == 0:
                node_depth[node] = 0
            else:
                node_depth[node] = max([node_depth[parent] for parent in parents]) + 1

    mechanisms = {}
    for node in causal_model.nodes:
        parents = causal_model.get_parents(node)
        if type(scale) in [float, int]:
            scale_value = scale
        else:
            scale_value = scale(node_depth[node])
        conditional_parameter_func = ParentsToChildMLP(
            parents,
            hidden_dims,
            {"loc": default_mean},
            activation=activation,
            extra_outputs={"scale": scale_value},
            **kwargs,
        )
        mechanisms[node] = ParametricConditionalDistribution(
            conditional_parameter_func, torch.distributions.Normal, parents
        )
    return mechanisms


def generate_gaussian_mlp_mechanisms(
    causal_model: CausalModel,
    hidden_dims: list[int],
    default_mean: float = 0.0,
    default_scale: float = 1.0,
    activation: str = "relu",
):
    """Generate a dictionary of Gaussian conditional MLP mechanisms with fixed variance for each node in the graph.

    Args:
        causal_model: a CausalModel object.
        hidden_dims: list of hidden dimensions for each MLP.
        default_mean: default mean of the Gaussian distribution (for root nodes).
        default_scale: default scale of the Gaussian distribution. (for root nodes).
        activation: activation function for the MLP.

    Returns:
        a dictionary of mechanisms for each node in the graph.
    """
    mechanisms = {}
    for node in causal_model.nodes:
        parents = causal_model.get_parents(node)
        conditional_parameter_func = ParentsToChildMLP(
            parents,
            hidden_dims,
            outputs_with_defaults={"loc": default_mean, "scale": default_scale},
            outputs_transform={"scale": torch.nn.Softplus()},
            activation=activation,
        )
        mechanisms[node] = ParametricConditionalDistribution(
            conditional_parameter_func, torch.distributions.Normal, parents
        )
    return mechanisms
