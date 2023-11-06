import random
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn as nn
from tqdm import tqdm

_THRESHOLDS = [0.5, 0.3, 0.1]


def set_random_seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)


def move_modules_to_device(module: nn.Module, device: Optional[torch.device]):
    """Moves modules to a given device."""
    if hasattr(module, "to"):
        module.to(device)

    for submodule in module.children():
        move_modules_to_device(submodule, device)


class TorchStandardScaler:
    """
    Standardizes data by subtracting the mean and dividing by the standard deviation.

    From https://discuss.pytorch.org/t/pytorch-tensor-scaling/38576/8.
    """

    def fit(self, data):
        self.mean = data.mean(0, keepdim=True)
        self.std = data.std(0, unbiased=False, keepdim=True)

    def transform(self, data):
        data -= self.mean
        data /= self.std + 1e-7
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


def compute_p_vals(X_df):
    """Computes p-values of a KS test for each candidate edge of a causal graph given interventional data.

    Requires a input dataframe containing the interventional data and acolumn called `perturbation_label`
    containing labels for which column is perturbed. `obs` is reserved for the observational subset.

    Returns a dataframe of edges with associated p-values and adjusted p-values (BH corrected).
    """
    edge_rows = []
    d = X_df.shape[1] - 1
    observational_X_df = X_df[X_df.perturbation_label == "obs"]
    for target_gene_idx in tqdm(range(d)):
        for candidate_parent_idx in range(d):
            if target_gene_idx == candidate_parent_idx:
                continue
            cand_int_subset_df = X_df[X_df.perturbation_label == candidate_parent_idx]
            _, pval = scipy.stats.kstest(
                observational_X_df.loc[:, target_gene_idx].to_numpy(),
                cand_int_subset_df.loc[:, target_gene_idx].to_numpy(),
            )
            edge_rows.append((candidate_parent_idx, target_gene_idx, pval))
    edges_df = pd.DataFrame(
        edge_rows, columns=["candidate_parent_idx", "target_gene_idx", "pval"]
    )

    # Compute BH corrected pvals
    edges_df = edges_df.sort_values("pval")
    n_pvals = edges_df.shape[0]
    edges_df["pval_rank"] = np.arange(1, edges_df.shape[0] + 1)
    edges_df["pval_adj"] = edges_df["pval"] * n_pvals / edges_df["pval_rank"]

    return edges_df


def ks_test_screen(X_df, use_sig=True, sig=0.10, n_parents=50, verbose=False):
    """Runs a pre-screen on candidate edges using the KS test metric.

    Returns a binary mask indicating which edges to consider.
    """
    edges_df = compute_p_vals(X_df)
    if use_sig:
        valid_edges_df = edges_df[edges_df.pval_adj < sig]
        if verbose:
            print(
                f"Fraction edges valid under significance level {sig}: {len(valid_edges_df) / len(edges_df):.2f}"
            )
    else:
        G = X_df.shape[1] - 1
        if n_parents >= G:
            valid_edges_df = edges_df
        else:
            valid_edges_dfs = []
            sorted_groups = [
                df.sort_values("pval_adj", ascending=True)
                for _, df in edges_df.groupby(["target_gene_idx"])
            ]
            for g in sorted_groups:
                valid_edges_dfs.append(g[:n_parents])
            valid_edges_df = pd.concat(valid_edges_dfs)

    G = X_df.shape[1] - 1
    mask = np.zeros((G, G), dtype=int)
    for row in valid_edges_df.iterrows():
        mask[int(row[1]["candidate_parent_idx"]), int(row[1]["target_gene_idx"])] = 1

    return mask


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


def compute_min_dag_threshold(adjacency_matrix) -> float:
    def is_acyclic(m):
        return nx.is_directed_acyclic_graph(nx.DiGraph(m))

    def bisect(func, a, b, tol=1e-5):
        mid = (a + b) / 2.0
        while (b - a) / 2.0 > tol:
            if func(mid) is True:
                b = mid
            else:
                a = mid
            mid = (a + b) / 2.0
        return mid

    def is_dag_at_threshold(threshold):
        return is_acyclic(adjacency_matrix > threshold)

    min_dag_threshold = bisect(is_dag_at_threshold, 0, 10)
    return min_dag_threshold


def print_graph_from_weights(
    d, B_pred, B_true, thresholds=_THRESHOLDS, max_parents=50, max_nodes=50
):
    B_true_square = B_true @ B_true
    for i in range(min(d, max_nodes)):
        parents_weights = B_pred[:, i]
        parents = sorted(range(d), key=lambda j: parents_weights[j], reverse=True)
        parents_str = []
        for t in thresholds:
            if parents_weights[parents[0]] < t:
                parents_str.append("|")
        for idx, j in enumerate(parents[:max_parents]):
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
                        and parents_weights[parents[idx]]
                        > t
                        > parents_weights[parents[idx + 1]]
                    ),
                    (idx == d - 1 and parents_weights[parents[idx]] > t),
                ]
                if any(conditions):
                    parents_str.append("|")
        print(f"Node {i:2}: " + " ".join(parents_str))
    print("Thresholds t:")
    for t in thresholds:
        is_dag = nx.is_directed_acyclic_graph(nx.DiGraph(B_pred > t))
        diff = B_true != (B_pred > t)
        score = diff.sum()
        shd = score - ((((diff + diff.transpose()) == 0) & (diff != 0)).sum() / 2)
        print(f"\tt >{t}: is_dag={is_dag}, shd={shd}")
    print()
    print()
