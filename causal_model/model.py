from collections import defaultdict
from typing import Optional

import networkx as nx
import pandas as pd

from utils import set_random_seed_all


class CausalModel:
    """A causal model is based on a causal graph, which is a directed acyclic graph (DAG) where each node represents a
    variable and each edge represents a direct causal effect between the two variables.

    In addition, each node can be associated with a causal mechanism that describes how each variable is "computed" from
    its parents. It can either be a structural equation or a conditional distribution.
    For now, we only support conditional distribution.

    Finally, the causal graph supports interventions that change the mechanism of a node.
    Each intervention has a unique name and is a set of nodes with their associated new mechanisms.
    """

    def __init__(
        self, graph: nx.DiGraph, causal_mechanisms: dict = None, interventions: dict = None
    ):
        self.graph = graph
        self.causal_mechanisms = causal_mechanisms if causal_mechanisms is not None else dict()
        self.interventions = interventions if interventions is not None else defaultdict(dict)

        self.variables = list(self.graph.nodes)
        self.adjacency = nx.to_numpy_array(self.graph)
        self._check_acyclic()
        self._check_causal_mechanisms_graph_consistency()

    @property
    def n_interventions(self):
        return len(self.interventions)

    @property
    def nodes(self) -> list:
        return list(self.graph.nodes)

    def get_parents(self, node):
        """Return the parents of a variable."""
        return sorted(self.graph.predecessors(node))

    def sample_from_model(self, n_samples, intervention_name=None):
        if intervention_name is not None and intervention_name not in self.interventions:
            raise ValueError(f"Intervention does not exist, {intervention_name}")
        interventions = {} if intervention_name is None else self.interventions[intervention_name]

        samples_per_node = dict()
        for node in nx.topological_sort(self.graph):
            parents = list(self.graph.predecessors(node))
            parents_values = {parent: samples_per_node[parent] for parent in parents}
            if node in interventions:
                distribution = interventions[node]
            else:
                distribution = self.causal_mechanisms[node]
            sample_shape = [n_samples] if len(parents) == 0 else []
            samples_per_node[node] = distribution.sample(
                sample_shape=sample_shape, parents_values=parents_values
            )

        return samples_per_node

    def sample_from_observational_distribution(self, n_samples):
        """Sample from the observational distribution of the causal model."""
        return self.sample_from_model(n_samples)

    def sample_from_interventional_distribution(self, n_samples, intervention_name):
        """Sample from the interventional distribution of the causal model."""
        return self.sample_from_model(n_samples, intervention_name)

    def generate_dataframe_from_all_distributions(
        self,
        n_samples_control: int = 1_000,
        n_samples_per_intervention: int = 100,
        subset_interventions: Optional[list] = None,
        seed: int = 0,
    ):
        """Generate a dataset from the observational distribution and the interventional distributions (all or the
        specified subset).
        Note: even when a subset of interventions is specified, all the interventions are sampled, and only then they
        are filtered. This ensures reproducibility of the dataset: the same dataset per intervention will be generated
        even if the subset of interventions is changed (for a given seed).


        Args:
            n_samples_control (int): number of samples from the observational distribution
            n_samples_per_intervention (int): number of samples from each interventional distribution
            subset_interventions (list): subset of interventions to consider. If None, all interventions are considered.
            seed (int): random seed
        Returns:
            pd.DataFrame: dataset with the samples (one column per variable) and a column "perturbation_label" that
                indicates the intervention applied to each sample (or "obs" if no intervention was applied).
        """
        set_random_seed_all(seed)
        samples = self.sample_from_observational_distribution(n_samples_control)
        samples["perturbation_label"] = "obs"
        data = [pd.DataFrame(samples)]

        for intervention_name in self.interventions.keys():
            samples = self.sample_from_interventional_distribution(
                n_samples_per_intervention, intervention_name
            )
            samples["perturbation_label"] = intervention_name
            samples = pd.DataFrame(samples)
            data.append(samples)

        data = pd.concat(data, ignore_index=True)
        # sort columns according to the node order of the graph, to be consistent with the adjacency matrix
        data = data[self.nodes + ["perturbation_label"]]
        if subset_interventions is not None:
            subset_interventions = set(subset_interventions) | {"obs"}
            data = data[data["perturbation_label"].isin(subset_interventions)]
        return data

    def set_causal_mechanisms(self, causal_mechanisms):
        self.causal_mechanisms = causal_mechanisms
        self._check_causal_mechanisms_graph_consistency()

    def set_intervention(self, intervention_name, new_causal_mechanisms):
        self.interventions[intervention_name] = new_causal_mechanisms
        self._check_causal_mechanisms_graph_consistency()

    def update_intervention(self, intervention_name, new_causal_mechanisms):
        """Add new_causal_mechanism to the existing intervention intervention_name."""
        self.interventions[intervention_name].update(new_causal_mechanisms)
        self._check_causal_mechanisms_graph_consistency()

    def _check_acyclic(self):
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Graph is not acyclic.")

    def _check_causal_mechanisms_graph_consistency(self):
        """Check that the parents of the causal mechanisms are consistent with the graph."""
        for node, mechanism in self.causal_mechanisms.items():
            if not self._is_mechanism_consistent_with_graph(node, mechanism):
                raise ValueError(
                    f"Parents of the causal mechanism of node {node} are not consistent with the graph."
                )

        for intervention_name, new_mechanisms in self.interventions.items():
            for node, mechanism in new_mechanisms.items():
                if not self._is_mechanism_consistent_with_graph(node, mechanism):
                    raise ValueError(
                        f"Parents of the causal mechanism of node {node} are not consistent with "
                        f"the graph, for intervention {intervention_name}."
                    )

    def _is_mechanism_consistent_with_graph(self, node, mechanism):
        """Check that the parents of the causal mechanism are consistent with the graph: the parents of the mechanism
        are a subset of the parents of the node in the graph.
        """
        graph_parents = set(self.graph.predecessors(node))
        mechanism_parents = set(mechanism.get_parents())
        return mechanism_parents.issubset(graph_parents)
