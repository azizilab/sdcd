import abc
import copy
from typing import Union

from torch.distributions import AffineTransform, Distribution, TransformedDistribution

# For simplicity, we directly use torch distribution,
# but what we need is just a class with a sample method
# class Distribution(abc.ABC):
#     @abc.abstractmethod
#     def sample(self, n_samples):
#         pass


class ConditionalDistribution(abc.ABC):
    @abc.abstractmethod
    def compute_distribution_from_parents(self, parents_values) -> Distribution:
        pass

    @abc.abstractmethod
    def get_parents(self) -> list[str]:
        pass

    def sample(self, sample_shape, parents_values):
        return self.compute_distribution_from_parents(parents_values).sample(
            sample_shape
        )


Mechanism = Union[Distribution, ConditionalDistribution]


class MarginalDistribution:
    """A distribution that is not conditional on any other variable."""

    def __init__(self, distribution: Distribution):
        self.distribution = distribution

    def compute_distribution_from_parents(self, parents_values) -> Distribution:
        if len(parents_values) != 0:
            raise ValueError("Marginal distribution should not have any parents")
        return self.distribution

    def get_parents(self) -> list[str]:
        return []

    def sample(self, sample_shape, parents_values):
        return self.distribution.sample(sample_shape)


class ParametricConditionalDistribution(ConditionalDistribution):
    """
    A conditional distribution of the form p(y|x) = response_distribution(conditional_parameters_func(x))
    That is p(y|x) is always from the same response_distribution family, with  parameters of the distribution that
    depend on x.

    A (non-conditional) distribution is a special case where the parent set is empty.

    Args:
        conditional_parameters_func: a function that takes as input the values of the parents of the node and returns
            a dictionary of parameters for the building response distribution.
        response_distribution_constructor: a function that takes as input the parameters of the response distribution
            and returns a Distribution object.
        parent_names: a list of the names of the parents of the node.
    """

    def __init__(
        self,
        conditional_parameters_func,
        response_distribution_constructor: type[Distribution],
        parent_names: list[str],
    ):
        self.response_distribution_constructor = response_distribution_constructor
        self.conditional_parameters_func = conditional_parameters_func
        self.parent_names = parent_names

        if len(self.parent_names) == 0:
            self.root = True
        else:
            self.root = False

    def get_parents(self) -> list[str]:
        return self.parent_names

    def compute_distribution_from_parents(self, parent_values) -> Distribution:
        conditional_parameters = self.conditional_parameters_func(parent_values)
        response_distribution = self.response_distribution_constructor(
            **conditional_parameters
        )
        return response_distribution


# class ScaledDistribution(Distribution):
#     """Simple wrapper around a distribution to scale the output"""
#
#     def __init__(self, distribution: Distribution, scale: float):
#         super().__init__()
#         self.distribution = distribution
#         self.scale = scale
#
#     def sample(self, sample_shape):
#         return self.distribution.sample(sample_shape) * self.scale

# We will use the Distribution transform from torch since we are using their Distribution class


def scale_mechanism(mechanism: Mechanism, scale: float) -> Mechanism:
    """Return a copy of the mechanism with the distribution scaled by scale.
    For instance, we can knock down a mechanism by setting scale < 1.
    """
    transform = AffineTransform(0.0, scale)
    if isinstance(mechanism, Distribution):
        # new_mechanism = ScaledDistribution(copy.deepcopy(mechanism), scale)
        new_mechanism = TransformedDistribution(mechanism, [transform])
    elif isinstance(mechanism, ConditionalDistribution):
        new_mechanism = copy.deepcopy(mechanism)
        new_mechanism.response_distribution_constructor = (
            lambda **kwargs: TransformedDistribution(
                mechanism.response_distribution_constructor(**kwargs), [transform]
            )
        )
    else:
        raise ValueError(f"Unknown mechanism type {type(mechanism)}")
    return new_mechanism
