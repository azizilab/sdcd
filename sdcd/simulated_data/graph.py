import networkx as nx
import numpy as np


def chain_graph(n_variables=20):
    """Return a chain graph."""
    graph = nx.DiGraph()
    nodes = [i for i in range(n_variables)]
    graph.add_nodes_from(nodes)
    for a, b in zip(nodes[:-1], nodes[1:]):
        graph.add_edge(a, b)

    return graph


def random_dag_from_undirected_graph(graph, keep_original_nodes=False):
    """Return a random DAG from an undirected graph. By randomly reordering the nodes and ensuring that the edges are
    directed from lower to higher node index
    """
    nodes = list(graph.nodes)
    nodes_original = nodes.copy()
    np.random.shuffle(nodes)
    edges = []
    for edge in graph.edges:
        if nodes.index(edge[0]) < nodes.index(edge[1]):
            edges.append(edge)
        else:
            edges.append((edge[1], edge[0]))
    dag = nx.DiGraph()
    if keep_original_nodes:
        dag.add_nodes_from(nodes_original)
    else:
        dag.add_nodes_from(nodes)
    dag.add_edges_from(edges)

    return dag


def random_dag(n_nodes: int = 20, n_edges: int = 20, distribution: str = "uniform"):
    """Return a random DAG.

    Args:
        n_nodes: Number of nodes.
        n_edges: Number of edges (only used for uniform distribution).
        distribution: Distribution of the random graph, one of "uniform" (or "erdos_renyi") or "scale_free".
    """
    if distribution in ["uniform", "erdos_renyi"]:
        graph = nx.gnm_random_graph(n_nodes, n_edges, directed=False)
    elif distribution == "scale_free":
        graph = nx.scale_free_graph(n_nodes, alpha=0.41, beta=0.54, gamma=0.05)
    else:
        raise ValueError(f"Unknown distribution {distribution}.")

    return random_dag_from_undirected_graph(graph)


def random_diagonal_band_dag(
    n_nodes=20, n_edges=50, bandwidth=4, n_edges_per_node=None
):
    """Return a random DAG with edges only between nodes that are at most band_size apart.

    Parameters
    ----------
    n_nodes: int
        Number of nodes.
    n_edges: int
        Number of edges. Ignored and set to n_nodes * n_edges_per_node if n_edges_per_node is not None.
    bandwidth: int
        Maximum distance between nodes that can be connected by an edge.
    n_edges_per_node: int
        Average number of edges per node. If None, n_edges is used.
    """
    if n_edges_per_node is not None:
        n_edges = n_nodes * n_edges_per_node

    possible_edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if abs(i - j) <= bandwidth:
                possible_edges.append((i, j))
    edges = np.random.choice(len(possible_edges), n_edges, replace=False)
    edges = [possible_edges[i] for i in edges]
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(edges)

    return random_dag_from_undirected_graph(graph)
