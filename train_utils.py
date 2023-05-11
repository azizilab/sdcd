import numpy as np
import pandas as pd

import torch 
import wandb
import networkx as nx

from utils import print_graph_from_weights

THRESHOLDS = [0.5, 0.3, 0.1]


def subset_interventions(X_df, n_interventions, maintain_dataset_size=False, obs_label="obs", perturbation_colname="perturbation_label"):
    X_obs_df = X_df[X_df[perturbation_colname] == obs_label]
    size_obs = X_obs_df.shape[0]
    G = (X_df[perturbation_colname].nunique() - 1)
    size_interventions = (X_df[perturbation_colname] != obs_label).sum() // G
    subsample_intervention_idxs = np.random.choice(
        np.arange(G), size=n_interventions, replace=False
    )
    subsample_intervention_names = X_df.columns[subsample_intervention_idxs]
    X_subsets = []
    if maintain_dataset_size:
        X_subsets.append(X_df[X_df[perturbation_colname].isin(subsample_intervention_names)])
        size_obs_subset = size_obs - size_interventions * n_interventions
        obs_subset_idxs = np.random.choice(
            np.arange(size_obs), size=size_obs_subset, replace=False
        )
        X_subsets.append(X_obs_df.iloc[obs_subset_idxs])
    else:
        X_subsets.append(X_df[X_df[perturbation_colname].isin(np.append(subsample_intervention_names, [obs_label]))])
    X_subset_df = pd.concat(X_subsets)
    return X_subset_df

def create_intervention_dataloader(X_df, batch_size, obs_label="obs", perturbation_colname="perturbation_label"):
    X = torch.FloatTensor(X_df.drop(perturbation_colname, axis=1).to_numpy().astype(float))
# ensure interventions are ints mapping to index of column
    column_mapping = {c: i for i, c in enumerate(X_df.columns[:-1])}
    column_mapping[obs_label] = -1
    interventions = torch.LongTensor(X_df[perturbation_colname].map(column_mapping).values).reshape((-1, 1))
    dataset = torch.utils.data.TensorDataset(X, interventions)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def train(model, data_loader, optimizer, config, freeze_gamma_at_dag=False, log_wandb=False, print_graph=True, B_true=None, start_wandb_epoch=0, n_epochs_check=1000):
    """Train the model. Assumes data_loader outputs batches alongside interventions."""
    # unpack config
    n_epochs = config["n_epochs"]
    alphas = config["alphas"]
    gammas = config["gammas"]
    beta = config["beta"]
    is_prescreen = model.dag_penalty_flavor == "none"

    n_observations = data_loader.batch_size * len(data_loader)
    d = data_loader.dataset[0][0].shape[0]

    gamma_cap = None
    for epoch in range(n_epochs):
        alpha = alphas[epoch]
        if gamma_cap is None:
            gamma = gammas[epoch]
        else:
            gamma = gamma_cap

        epoch_loss = 0
        epoch_loss_details = []
        for batch in data_loader:
            X_batch, interventions_batch = batch
            optimizer.zero_grad()
            loss, loss_details = model.loss(
                X_batch,
                alpha,
                beta,
                gamma,
                n_observations,
                interventions=interventions_batch,
                return_detailed_losses=True,
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_loss_details.append(loss_details)

        if epoch % n_epochs_check == 0:
            B_pred = model.get_adjacency_matrix().detach().numpy()

            if B_true is not None:
                score = (B_true != (B_pred > 0.3)).sum()
            else:
                score = "na"

            epoch_loss /= len(data_loader)
            print(f"Epoch {epoch}: loss={epoch_loss:.2f}, score={score}, gamma={gamma:.2f}")

            if epoch > max(0.05 * n_epochs, 10) and freeze_gamma_at_dag and gamma_cap is None:
                # Check dag if freeze_gamma_at_dag is True and beyond a warmup period of epochs to avoid seeing a trivial DAG.
                is_dag = nx.is_directed_acyclic_graph(nx.DiGraph(B_pred > 0.3))
                if is_dag:
                    gamma_cap = gamma

            if log_wandb:
                if B_true is not None:
                    recall = (B_true.astype(bool) & (B_pred > 0.3)).sum() / B_true.sum()
                    precision = (B_true.astype(bool) & (B_pred > 0.3)).sum() / (B_pred > 0.3).sum()
                else:
                    recall = "na"
                    precision = "na"

                n_edges_pred = (B_pred > 0.3).sum()
                epoch_loss_details = {k: sum(d[k].item() for d in epoch_loss_details) for k in epoch_loss_details[0]}
                epoch_loss_details = {k: v / len(data_loader) for k, v in epoch_loss_details.items()}

                wandb.log(
                    {
                        "epoch": epoch + start_wandb_epoch,
                        "epoch_loss": epoch_loss,
                        "score": score,
                        "precision": precision,
                        "recall": recall,
                        "n_edges_pred": n_edges_pred,
                        "gamma": gamma,
                        "alpha": alpha,
                        "is_prescreen": int(is_prescreen),
                        **epoch_loss_details,
                    }
                )

            if print_graph and B_true is not None:
                print_graph_from_weights(d, B_pred, B_true, THRESHOLDS)

    return model

