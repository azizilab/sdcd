import networkx as nx
import matplotlib.pyplot as plt

from causal_model.model import CausalModel


def draw_dag_topological_sort(dag):
    """Plot the dag with a topological sort."""
    if type(dag) == CausalModel:
        draw_dag_topological_sort(dag.graph)
        return
    for layer, nodes in enumerate(nx.topological_generations(dag)):
        for node in nodes:
            dag.nodes[node]["layer"] = layer
    pos = nx.multipartite_layout(dag, subset_key="layer")
    nx.draw_networkx(dag, pos=pos, with_labels=True)
