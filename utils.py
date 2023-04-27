import random

import networkx as nx
import numpy as np
import torch


def set_random_seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)


def get_leading_left_and_right_eigenvectors(A):
    """Get the leading left and right eigenvectors of a matrix.
    This is not optimized; it's just for testing.
    Args:
        A (np.ndarray or torch.Tensor): A square matrix.
    Returns:
        (np.ndarray, np.ndarray): The leading left and right eigenvectors.
    """
    assert A.shape[0] == A.shape[1]
    if isinstance(A, torch.Tensor):
        A = A.detach().numpy()
    # right eigenvector
    w, v = np.linalg.eig(A)
    idx = np.argmax(np.abs(w))
    right_eigenvector = v[:, idx]
    # left eigenvector
    w, v = np.linalg.eig(A.T)
    idx = np.argmax(np.abs(w))
    left_eigenvector = v[:, idx]
    return left_eigenvector, right_eigenvector


def print_graph_from_weights(d, B_pred, B_true, thresholds):
    B_true_square = B_true @ B_true
    for i in range(d):
        parents_weights = B_pred[:, i]
        parents = sorted(range(d), key=lambda j: parents_weights[j], reverse=True)
        parents_str = []
        for t in thresholds:
            if parents_weights[parents[0]] < t:
                parents_str.append("|")
        for idx, j in enumerate(parents):
            # print the parent in orange if it is actually a child of the node
            if B_true[j, i]:
                parents_str.append(f"\033[92m{j}\033[0m")
            elif B_true[i, j]:
                parents_str.append(f"\033[93m{j}\033[0m")
            # print the node in blue if it is actually the parent of a parent of the node
            elif B_true_square[j, i]:
                parents_str.append(f"\033[94m{j}\033[0m")
            else:
                parents_str.append(f"\033[91m{j}\033[0m")
            # add | if the parent weight is greater than one of the thresholds
            # and the next parent weight is less than the threshold
            for t in thresholds:
                conditions = [
                    (
                            idx < d - 1
                            and parents_weights[parents[idx]] > t > parents_weights[parents[idx + 1]]
                    ),
                    (idx == d - 1 and parents_weights[parents[idx]] > t),
                ]
                if any(conditions):
                    parents_str.append("|")
        print(f"Node {i:2}: " + " ".join(parents_str))
    print("Thresholds t:")
    for t in thresholds:
        is_dag = nx.is_directed_acyclic_graph(nx.DiGraph(B_pred > t))
        sdh = (B_true != (B_pred > t)).sum()
        print(f"\tt >{t}: is_dag={is_dag}, sdh={sdh}")
    print()
    print()
