import numpy as np
import pandas as pd

import torch
import torch.nn as nn


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
    obs_label="obs",
    perturbation_colname="perturbation_label",
    regime_format=False,
):
    X = torch.FloatTensor(
        X_df.drop(perturbation_colname, axis=1).to_numpy().astype(float)
    )
    # ensure interventions are ints mapping to index of column
    column_mapping = {c: i for i, c in enumerate(X_df.columns[:-1])}
    column_mapping[obs_label] = -1
    interventions = torch.LongTensor(
        X_df[perturbation_colname].map(column_mapping).values
    ).reshape((-1, 1))

    if regime_format:
        regimes = interventions.clone()
        interventions[torch.where(interventions == -1)] = X_df.shape[1] - 1
        interventions_oh = nn.functional.one_hot(
            interventions.squeeze(), num_classes=X_df.shape[1]
        )[
            :, :-1
        ]  # cutoff obs
        mask_interventions_oh = 1 - interventions_oh
        return torch.utils.data.TensorDataset(X, mask_interventions_oh, regimes)

    return torch.utils.data.TensorDataset(X, interventions)


def create_intervention_dataloader(
    X_df, batch_size, obs_label="obs", perturbation_colname="perturbation_label"
):
    dataset = create_intervention_dataset(X_df, obs_label, perturbation_colname)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )


def compute_metrics(B_pred_thresh, B_true):
    if B_true is not None:
        diff = B_true != B_pred_thresh
        score = diff.sum()
        shd = score - ((((diff + diff.transpose()) == 0) & (diff != 0)).sum() / 2)
        recall = (B_true.astype(bool) & B_pred_thresh.astype(bool)).sum() / B_true.sum()
        precision = (B_true.astype(bool) & B_pred_thresh.astype(bool)).sum() / (
            B_pred_thresh
        ).sum()
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
