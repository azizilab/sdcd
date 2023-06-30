import networkx as nx
import numpy as np


def _relabel_node_to_str(graph):
    """Relabel the nodes of a graph to str."""
    return nx.relabel_nodes(graph, {node: str(node) for node in graph.nodes if type(node) != str})


def chain_graph(n_variables=20):
    """Return a chain graph."""
    graph = nx.DiGraph()
    nodes = [str(i) for i in range(n_variables)]
    graph.add_nodes_from(nodes)
    for a, b in zip(nodes[:-1], nodes[1:]):
        graph.add_edge(a, b)

    return graph


def random_dag_from_undirected_graph(graph):
    """Return a random DAG from an undirected graph. By randomly reordering the nodes and ensuring that the edges are
    directed from lower to higher node index
    """
    nodes = list(graph.nodes)
    np.random.shuffle(nodes)
    edges = []
    for edge in graph.edges:
        if nodes.index(edge[0]) < nodes.index(edge[1]):
            edges.append(edge)
        else:
            edges.append((edge[1], edge[0]))
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)
    dag.add_edges_from(edges)

    return dag


def random_dag(n_nodes=20, n_edges=20, distribution="uniform"):
    """Return a random DAG.

    Args:
        n_nodes (int): Number of nodes.
        n_edges (int): Number of edges (only used for uniform distribution).
        distribution (str): Distribution of the random graph, one of "uniform" (or "erdos_renyi") or "scale_free".
    """
    if distribution in ["uniform", "erdos_renyi"]:
        graph = nx.gnm_random_graph(n_nodes, n_edges, directed=False)
    elif distribution == "scale_free":
        graph = nx.scale_free_graph(n_nodes, alpha=0.41, beta=0.54, gamma=0.05)
    else:
        raise ValueError(f"Unknown distribution {distribution}.")

    graph = _relabel_node_to_str(graph)
    return random_dag_from_undirected_graph(graph)
