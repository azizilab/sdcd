from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import scipy as sp
import torch
from torch.utils.data import Dataset, TensorDataset


def subset_interventions(
    X_df,
    n_interventions,
    maintain_dataset_size=False,
    obs_label="obs",
    perturbation_colname="perturbation_label",
):
    X_obs_df = X_df[X_df[perturbation_colname] == obs_label]
    size_obs = X_obs_df.shape[0]
    G = X_df[perturbation_colname].nunique() - 1
    size_interventions = (X_df[perturbation_colname] != obs_label).sum() // G
    subsample_intervention_idxs = np.random.choice(
        np.arange(G), size=n_interventions, replace=False
    )
    subsample_intervention_names = X_df.columns[subsample_intervention_idxs]
    X_subsets = []
    if maintain_dataset_size:
        X_subsets.append(
            X_df[X_df[perturbation_colname].isin(subsample_intervention_names)]
        )
        size_obs_subset = size_obs - size_interventions * n_interventions
        obs_subset_idxs = np.random.choice(
            np.arange(size_obs), size=size_obs_subset, replace=False
        )
        X_subsets.append(X_obs_df.iloc[obs_subset_idxs])
    else:
        X_subsets.append(
            X_df[
                X_df[perturbation_colname].isin(
                    np.append(subsample_intervention_names, [obs_label])
                )
            ]
        )
    X_subset_df = pd.concat(X_subsets)
    return X_subset_df


def create_intervention_dataset(
    X_df,
    perturbation_colname="perturbation_label",
    regime_format=True,
):
    X = torch.FloatTensor(
        X_df.drop(perturbation_colname, axis=1).to_numpy().astype(float)
    )
    # ensure interventions are ints mapping to index of column
    column_mapping = {str(c): i for i, c in enumerate(X_df.columns[:-1])}
    return _create_intervention_dataset_helper(
        X, X_df[perturbation_colname], column_mapping, regime_format=regime_format
    )


def create_intervention_dataset_anndata(
    adata,
    layer=None,
    perturbation_obsname="perturbation_label",
    regime_format=True,
):
    assert (
        perturbation_obsname in adata.obs.columns
    ), f"{perturbation_obsname} not found in adata.obs"

    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X
    X = X.toarray() if isinstance(X, sp.sparse.spmatrix) else X.copy()
    X = torch.FloatTensor(X.astype(float))

    # ensure interventions are ints mapping to index of column
    column_mapping = {str(c): i for i, c in enumerate(adata.var_names)}

    return _create_intervention_dataset_helper(
        X, adata.obs[perturbation_obsname], column_mapping, regime_format=regime_format
    )


def _create_intervention_dataset_helper(
    X,
    perturbations,
    column_mapping,
    regime_format=True,
):
    # Split the perturbation_colname by comma and map each value to its column index
    unstacked_perturbation_columns = (
        perturbations.map(str)
        .str.split(",", expand=True)
        .reset_index(drop=True)
        .stack()
        .map(column_mapping)
        .fillna(-1)
        .astype(int)
        .unstack(fill_value=-1)
    )
    combined_columns = unstacked_perturbation_columns.apply(
        lambda row: ",".join([str(val) for val in row if val != -1]), axis=1
    )

    if regime_format:
        # Split comma-separated strings and convert to a binary matrix
        def string_to_binary(row):
            if row == "":
                return np.ones(X.shape[1], dtype=int)
            else:
                indices = set(map(int, row.split(",")))
                return np.array([0 if i in indices else 1 for i in range(X.shape[1])])

        mask_interventions_oh = combined_columns.apply(string_to_binary)
        mask_interventions_oh = torch.LongTensor(
            np.vstack(mask_interventions_oh.to_numpy())
        )

        n_regimes = torch.LongTensor(X.shape[1] - mask_interventions_oh.sum(axis=1))

        return torch.utils.data.TensorDataset(X, mask_interventions_oh, n_regimes)

    max_perturbations = (
        unstacked_perturbation_columns.applymap(lambda x: x != -1).sum(axis=1).max()
    )
    if max_perturbations > 1:
        raise ValueError("Non regime format for multiple perturbations is unsupported")
    interventions = torch.LongTensor(
        pd.to_numeric(combined_columns, "coerce").fillna(-1).astype(int)
    )
    return TensorDataset(X, interventions)


def train_val_split(
    dataset: Dataset,
    flavor: Literal["random", "I-NLL", "train"] = "random",
    val_fraction: float = 0.2,
    seed: Optional[int] = None,
) -> Tuple[Dataset, Dataset]:
    if seed is not None:
        torch.manual_seed(seed)
    N = len(dataset)
    if flavor == "random":
        return torch.utils.data.random_split(
            dataset,
            [
                N - int(val_fraction * N),
                int(val_fraction * N),
            ],
        )
    elif flavor == "train":
        _, val_dataset = torch.utils.data.random_split(
            dataset,
            [
                N - int(val_fraction * N),
                int(val_fraction * N),
            ],
        )
        return dataset, val_dataset
    elif flavor == "I-NLL":
        if len(dataset.tensors) == 2:
            raise ValueError("I-NLL only works with regime datasets")
        mask_interventions_oh = dataset.tensors[1]
        n_regimes = dataset.tensors[2]
        unique_interventions = torch.unique(
            mask_interventions_oh[n_regimes > 0], dim=0
        )  # avoid observational samples
        # pick val fraction of these unique_interventions
        val_interventions = unique_interventions[
            torch.randperm(len(unique_interventions))[
                : int(val_fraction * len(unique_interventions))
            ]
        ]
        val_mask = torch.any(
            torch.all(
                torch.eq(mask_interventions_oh[:, None], val_interventions), dim=-1
            ),
            dim=-1,
        )
        train_mask = torch.logical_not(val_mask)
        return (
            TensorDataset(*[dataset.tensors[i][train_mask] for i in range(3)]),
            TensorDataset(*[dataset.tensors[i][val_mask] for i in range(3)]),
        )
    else:
        raise ValueError(f"Unknown train_val_split flavor: {flavor}")


def compute_metrics(B_pred_thresh, B_true):
    if B_true is not None:
        diff = B_true != B_pred_thresh
        score = diff.sum()
        shd = score - (((diff == diff.transpose()) & (diff != 0)).sum() / 2)
        recall = (B_true.astype(bool) & B_pred_thresh.astype(bool)).sum() / np.clip(
            B_true.sum(), 1, None
        )
        precision = (B_true.astype(bool) & B_pred_thresh.astype(bool)).sum() / np.clip(
            B_pred_thresh.sum(), 1, None
        )
    else:
        recall = "na"
        precision = "na"
        score = "na"
        shd = "na"

    n_edges_pred = (B_pred_thresh).sum()
    return {
        "score": score,
        "shd": shd,
        "precision": precision,
        "recall": recall,
        "n_edges_pred": n_edges_pred,
    }
