import torch

from causal_model.mechanisms import ParametricConditionalDistribution
from modules import DenseLayers


class MLP(torch.nn.Module):
    def __init__(self, parent_names, hidden_dims, default_mean=0.0, scale=1.0, activation="relu"):
        super().__init__()
        self.parent_names = parent_names
        self.hidden_dims = hidden_dims
        self.default_mean = default_mean
        self.scale = scale
        self.activation = activation

        n_parents = len(parent_names)
        self.n_parents = n_parents
        if self.n_parents == 0:
            self.conditional_parameters_func = lambda: {"loc": default_mean, "scale": self.scale}
        else:
            self.mlp = DenseLayers(self.n_parents, 1, hidden_dims, activation=activation)
            self.mlp.reset_parameters(10)

    def forward(self, **parents_values):
        if len(parents_values) != self.n_parents:
            raise ValueError("Wrong number of parents")
        if self.n_parents == 0:
            return {"loc": self.default_mean, "scale": self.scale}
        # we sort the keys to have a consistent order of parents
        x = torch.stack([parents_values[parent] for parent in sorted(parents_values.keys())], dim=1)
        mean = self.mlp(x).squeeze(-1)
        return {"loc": mean, "scale": self.scale}


#
# def get_conditional_mlp_loc(n_parents, hidden_dims, default_mean=0.0, scale=1.0, activation="relu"):
#     if n_parents == 0:
#         conditional_parameters_func = lambda: {"loc": default_mean, "scale": scale}
#     else:
#         # DenseLayers is initialized randomly
#         mlp = DenseLayers(n_parents, 1, hidden_dims, activation=activation)
#         mlp.reset_parameters(10)
#
#         def conditional_parameters_func(**parents_values):
#             if len(parents_values) != n_parents:
#                 raise ValueError("Wrong number of parents")
#             # we sort the keys to have a consistent order of parents
#             x = torch.stack(
#                 [parents_values[parent] for parent in sorted(parents_values.keys())], dim=1
#             )
#             mean = mlp(x).squeeze(-1)
#             return {"loc": mean, "scale": scale}
#
#     return conditional_parameters_func
#
#
# def get_conditional_mlp_loc_scale(
#     n_parents, hidden_dims, default_mean=0.0, default_scale=1.0, activation="relu"
# ):
#     if n_parents == 0:
#         conditional_parameters_func = lambda: {"loc": default_mean, "scale": default_scale}
#     else:
#         # DenseLayers is initialized randomly
#         mlp = DenseLayers(n_parents, 2, hidden_dims, activation=activation)
#
#         def conditional_parameters_func(**parents_values):
#             if len(parents_values) != n_parents:
#                 raise ValueError("Wrong number of parents")
#             # we sort the keys to have a consistent order of parents
#             x = torch.cat(
#                 [parents_values[parent] for parent in sorted(parents_values.keys())], dim=1
#             )
#             mean, log_scale = mlp(x).chunk(2, dim=1)
#             scale = torch.exp(log_scale)
#             return {"loc": mean, "scale": scale}
#
#     return conditional_parameters_func


def _generate_parametric_mechanisms_for_graph(
    conditional_parameters_func_generator, response_distribution_constructor, causal_graph, **kwargs
):
    """Generate a dictionary of mechanisms for each node in the graph.

    Args:
        conditional_parameters_func_generator: a function that takes as input the number of parents of the node and returns a function
            that takes as input the parents of the node and returns a dictionary of parameters for the building response
            distribution.
        response_distribution_constructor: a function that takes as input the parameters of the response distribution and
            returns a Distribution object.
        causal_graph: a CausalGraph object.
        **kwargs: additional arguments to pass to the conditional_parameters_func_generator.

    Returns:
        a dictionary of mechanisms for each node in the graph.
    """
    mechanisms = {}
    for node in causal_graph.nodes:
        parents = causal_graph.get_parents(node)
        conditional_parameters_func = conditional_parameters_func_generator(parents, **kwargs)
        mechanisms[node] = ParametricConditionalDistribution(
            conditional_parameters_func,
            response_distribution_constructor,
            parents,
        )
    return mechanisms


def generate_gaussian_mlp_mean_mechanisms_for_graph(
    causal_graph,
    hidden_dims,
    default_mean=0.0,
    scale=1.0,
    activation="sigmoid",
):
    """Generate a dictionary of Gaussian MLP mechanisms for each node in the graph.

    Args:
        causal_graph: a CausalGraph object.
        hidden_dims: a list of hidden dimensions for the MLP.
        default_mean: the default mean for the Gaussian MLP.
        scale: the scale of the Gaussian MLP.
        activation: the activation function for the MLP.
    Returns:
        a dictionary of Gaussian MLP mechanisms for each node in the graph.
    """
    conditional_parameters_func_generator = lambda n_parents: MLP(
        n_parents, hidden_dims, default_mean=default_mean, scale=scale, activation=activation
    )
    response_distribution_constructor = torch.distributions.Normal
    return _generate_parametric_mechanisms_for_graph(
        conditional_parameters_func_generator,
        response_distribution_constructor,
        causal_graph,
    )
