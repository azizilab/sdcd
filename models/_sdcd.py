import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import wandb
import networkx as nx

from modules import AutoEncoderLayers
from train_utils import (
    compute_metrics,
)
from utils import print_graph_from_weights

from .base._base_model import BaseModel


_DEFAULT_STAGE1_KWARGS = {
    "learning_rate": 2e-3,
    "n_epochs": 2_000,
    "alpha": 1e-2,
    "max_gamma": 0,
    "beta": 5e-3,
    "n_epochs_check": 100,
    "mask_threshold": 0.2,
}
_DEFAULT_STAGE2_KWARGS = {
    "learning_rate": 2e-2,
    "n_epochs": 3_000,
    "alphas": 5e-3,
    "max_gamma": 300,
    "beta": 5e-3,
    "freeze_gamma_at_dag": True,
    "freeze_gamma_threshold": 0.5,
    "threshold": 0.3,
    "n_epochs_check": 100,
    "dag_penalty_flavor": "scc",
}


class SDCD(BaseModel):
    def __init__(self):
        super().__init__()
        self._model_kwargs = None
        self._adj_matrix = None
        self._adj_matrix_thresh = None

    def train(
        self,
        dataset: Dataset,
        log_wandb: bool = False,
        wandb_project: str = "SDCD",
        wandb_config_dict: Optional[dict] = None,
        B_true: Optional[np.ndarray] = None,
        stage1_kwargs: Optional[dict] = None,
        stage2_kwargs: Optional[dict] = None,
    ):
        if log_wandb:
            wandb_config_dict = wandb_config_dict or {}
            wandb.init(
                project=wandb_project,
                name="SDCD",
                config=wandb_config_dict,
            )
        batch_size = 256
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        sample_batch = next(iter(dataloader))
        assert len(sample_batch) == 2, "Dataset should contain (X, intervention_labels)"
        d = sample_batch[0].shape[1]

        self._stage1_kwargs = _DEFAULT_STAGE1_KWARGS.update(stage1_kwargs or {})
        self._stage2_kwargs = _DEFAULT_STAGE2_KWARGS.update(stage2_kwargs or {})

        if log_wandb:
            wandb.init(
                project=wandb_project,
                name="SDCD",
                config={
                    "batch_size": batch_size,
                    "stage1_kwargs": self._stage1_kwargs,
                    "stage2_kwargs": self._stage2_kwargs,
                    **wandb_config_dict,
                },
            )

        start = time.time()
        # Stage 1: Pre-selection
        ps_model = AutoEncoderLayers(
            d,
            [10, 1],
            nn.Sigmoid(),
            shared_layers=False,
            adjacency_p=2.0,
            dag_penalty_flavor="none",
        )
        ps_optimizer = torch.optim.Adam(
            ps_model.parameters(), lr=self._stage1_kwargs["learning_rate"]
        )

        _train(
            ps_model,
            dataloader,
            ps_optimizer,
            stage1_kwargs,
            log_wandb=True,
            print_graph=True,
            B_true=B_true,
        )

        # Create mask for main algo
        mask_threshold = self._stage1_kwargs["mask_threshold"]
        mask = (
            ps_model.get_adjacency_matrix().detach().numpy() > mask_threshold
        ).astype(int)
        if B_true is not None:
            print(
                f"Recall of mask: {(B_true.astype(bool) & mask.astype(bool)).sum() / B_true.sum()}"
            )
        print(
            f"Fraction of possible edges in mask: {mask.sum() / (mask.shape[0] * mask.shape[1])}"
        )

        # Begin DAG training
        dag_penalty_flavor = self._stage2_kwargs["dag_penalty_flavor"]
        model = AutoEncoderLayers(
            d,
            [10, 1],
            nn.Sigmoid(),
            shared_layers=False,
            adjacency_p=2.0,
            dag_penalty_flavor=dag_penalty_flavor,
            mask=mask,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self._stage2_kwargs["learning_rate"]
        )

        _train(
            model,
            dataloader,
            optimizer,
            self._stage2_kwargs,
            log_wandb=True,
            print_graph=True,
            B_true=B_true,
            start_wandb_epoch=self._stage1_kwargs["n_epochs"],
        )
        self._train_runtime_in_sec = time.time() - start
        print(f"Finished training in {self._train_runtime_in_sec} seconds.")

    def get_adjacency_matrix(self, threshold: bool = True) -> np.ndarray:
        assert self._model is not None, "Model has not been trained yet."

        adj_matrix = self._model.get_adjacency_matrix().detach().numpy()
        return (adj_matrix > self.threshold).astype(int) if threshold else adj_matrix


def _train(
    model,
    dataloader,
    optimizer,
    config,
    log_wandb=False,
    print_graph=True,
    B_true=None,
    start_wandb_epoch=0,
):
    """Train the model. Assumes dataloader outputs batches alongside interventions."""
    # unpack config
    n_epochs = config["n_epochs"]
    alpha = config["alpha"]
    max_gamma = config["max_gamma"]
    gammas = np.linspace(0, max_gamma, n_epochs)
    beta = config["beta"]
    threshold = config["threshold"]
    freeze_gamma_at_dag = config["freeze_gamma_at_dag"]
    freeze_gamma_threshold = config["freeze_gamma_threshold"]
    n_epochs_check = config["n_epochs_check"]

    is_prescreen = model.dag_penalty_flavor == "none"

    n_observations = dataloader.batch_size * len(dataloader)
    d = dataloader.dataset[0][0].shape[0]

    gamma_cap = None
    for epoch in range(n_epochs):
        if gamma_cap is None:
            gamma = gammas[epoch]
        else:
            gamma = gamma_cap

        epoch_loss = 0
        epoch_loss_details = []
        for batch in dataloader:
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

            if (
                epoch > max(0.05 * n_epochs, 10)
                and freeze_gamma_at_dag
                and gamma_cap is None
            ):
                # Check dag if freeze_gamma_at_dag is True and beyond a warmup period of epochs to avoid seeing a trivial DAG.
                is_dag = nx.is_directed_acyclic_graph(
                    nx.DiGraph(B_pred > freeze_gamma_threshold)
                )
                if is_dag:
                    gamma_cap = gamma

            epoch_loss /= len(dataloader)
            if B_true is not None:
                metrics_dict = compute_metrics((B_pred > threshold).astype(int), B_true)
                print(
                    f"Epoch {epoch}: loss={epoch_loss:.2f}, score={metrics_dict['score']}, shd={metrics_dict['shd']}, gamma={gamma:.2f}"
                )
            else:
                metrics_dict = {}
                print(f"Epoch {epoch}: loss={epoch_loss:.2f}, gamma={gamma:.2f}")

            if log_wandb:
                epoch_loss_details = {
                    k: sum(d[k].item() for d in epoch_loss_details)
                    for k in epoch_loss_details[0]
                }
                epoch_loss_details = {
                    k: v / len(dataloader) for k, v in epoch_loss_details.items()
                }
                wandb.log(
                    {
                        "epoch": epoch + start_wandb_epoch,
                        "epoch_loss": epoch_loss,
                        "gamma": gamma,
                        "alpha": alpha,
                        "is_prescreen": int(is_prescreen),
                        **epoch_loss_details,
                        **metrics_dict,
                    }
                )

            if print_graph and B_true is not None:
                print_graph_from_weights(d, B_pred, B_true)

    return model
