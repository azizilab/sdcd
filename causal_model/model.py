from collections import defaultdict
import networkx as nx


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
        # make sure each variable name is a string to be able to call the parents as kwargs
        for n in graph:
            if not isinstance(n, str):
                raise ValueError(f"The node names must be str, {n}")

        self.graph = graph
        self.causal_mechanisms = causal_mechanisms if causal_mechanisms is not None else dict()
        self.interventions = interventions if interventions is not None else defaultdict(dict)

        self.variables = list(self.graph.nodes)
        self.adjacency = nx.to_numpy_array(self.graph)
        self._check_acyclic()
        self._check_causal_mechanisms_graph_consistency()

    @property
    def nodes(self):
        return self.graph.nodes

    def get_parents(self, node):
        """Return the parents of a node."""
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
                sample_shape=sample_shape, **parents_values
            )

        return samples_per_node

    def sample_from_observational_distribution(self, n_samples):
        """Sample from the observational distribution of the causal graph."""
        return self.sample_from_model(n_samples)

    def sample_from_interventional_distribution(self, n_samples, intervention_name):
        """Sample from the interventional distribution of the causal graph."""
        return self.sample_from_model(n_samples, intervention_name)

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
