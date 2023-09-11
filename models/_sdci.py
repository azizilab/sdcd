import copy
import time
from typing import Optional, Literal

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import wandb
import networkx as nx

from modules import AutoEncoderLayers
from utils import (
    print_graph_from_weights,
    move_modules_to_device,
    TorchStandardScaler,
    compute_metrics,
)

from .base._base_model import BaseModel


_DEFAULT_STAGE1_KWARGS = {
    "learning_rate": 2e-3,
    "batch_size": 256,
    "n_epochs": 2_000,
    "alpha": 1e-2,
    "max_gamma": 0,
    "beta": 5e-3,
    "n_epochs_check": 100,
    "mask_threshold": 0.2,
}
_DEFAULT_STAGE2_KWARGS = {
    "learning_rate": 2e-2,
    "batch_size": 512,
    "n_epochs": 1_000,
    "alpha": 5e-3,
    "max_gamma": 300,
    "gamma_schedule": "linear",
    "beta": 5e-3,
    "freeze_gamma_at_dag": True,
    "freeze_gamma_threshold": 0.5,
    "threshold": 0.3,
    "n_epochs_check": 100,
    "dag_penalty_flavor": "power_iteration",
}


class SDCI(BaseModel):
    def __init__(
        self,
        model_variance_flavor: Literal["unit", "nn", "parameter"] = "unit",
        standard_scale: bool = False,
    ):
        super().__init__()
        self.model_variance_flavor = model_variance_flavor
        self.standard_scale = standard_scale
        self._stage1_kwargs = None
        self._stage2_kwargs = None

    def train(
        self,
        dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        log_wandb: bool = False,
        wandb_project: str = "SDCI",
        wandb_name: str = "SDCI",
        wandb_config_dict: Optional[dict] = None,
        B_true: Optional[np.ndarray] = None,
        stage1_kwargs: Optional[dict] = None,
        stage2_kwargs: Optional[dict] = None,
        train_kwargs: Optional[dict] = None,
        verbose: bool = False,
        device: Optional[torch.device] = None,
        l2_on_dispatcher: bool = True,
    ):
        self._stage1_kwargs = {**_DEFAULT_STAGE1_KWARGS.copy(), **(stage1_kwargs or {})}
        self._stage2_kwargs = {**_DEFAULT_STAGE2_KWARGS.copy(), **(stage2_kwargs or {})}

        self.threshold = self._stage2_kwargs["threshold"]

        ps_batch_size = self._stage1_kwargs["batch_size"]
        batch_size = self._stage2_kwargs["batch_size"]

        if self.standard_scale:
            scaler = TorchStandardScaler()
            scaled_X = scaler.fit_transform(dataset[:][0])
            dataset = torch.utils.data.TensorDataset(scaled_X, *dataset[:][1:])
            val_dataset = torch.utils.data.TensorDataset(
                scaler.transform(val_dataset[:][0]), *val_dataset[:][1:]
            )

        val_dataloader = None
        if val_dataset is not None:
            val_dataloader = DataLoader(val_dataset, batch_size=ps_batch_size)

        ps_dataloader = DataLoader(dataset, batch_size=ps_batch_size, shuffle=True)
        sample_batch = next(iter(ps_dataloader))
        assert len(sample_batch) == 3, "Dataset should contain (X, masks, regimes)"
        self.d = sample_batch[0].shape[1]

        if log_wandb:
            if wandb.run is not None:
                wandb.finish()  # Close previous run

            wandb_config_dict = wandb_config_dict or {}
            wandb.init(
                project=wandb_project,
                name=wandb_name,
                config={
                    "batch_size": batch_size,
                    "stage1_kwargs": self._stage1_kwargs,
                    "stage2_kwargs": self._stage2_kwargs,
                    **wandb_config_dict,
                },
            )

        start = time.time()
        # Stage 1: Pre-selection
        self._ps_model = AutoEncoderLayers(
            self.d,
            [10],
            nn.Sigmoid(),
            model_variance_flavor=self.model_variance_flavor,
            shared_layers=False,
            adjacency_p=2.0,
            dag_penalty_flavor="none",
            l2_on_dispatcher=l2_on_dispatcher,
        )
        if device:
            move_modules_to_device(self._ps_model, device)

        ps_optimizer = torch.optim.Adam(
            self._ps_model.parameters(), lr=self._stage1_kwargs["learning_rate"]
        )

        ps_kwargs = {
            **self._stage1_kwargs,
            "threshold": self.threshold,
        }
        train_kwargs = train_kwargs or {}
        self._ps_model, next_epoch = _train(
            self._ps_model,
            ps_dataloader,
            ps_optimizer,
            ps_kwargs,
            val_dataloader=val_dataloader,
            log_wandb=log_wandb,
            print_graph=verbose,
            B_true=B_true,
            device=device,
            return_next_epoch=True,
            **train_kwargs,
        )

        # Create mask for main algo
        mask_threshold = self._stage1_kwargs["mask_threshold"]
        self._mask = (
            self._ps_model.get_adjacency_matrix().cpu().detach().numpy()
            > mask_threshold
        ).astype(int)
        if B_true is not None:
            print(
                f"Recall of mask: {(B_true.astype(bool) & self._mask.astype(bool)).sum() / B_true.sum()}"
            )
        print(
            f"Fraction of possible edges in mask: {self._mask.sum() / (self._mask.shape[0] * self._mask.shape[1])}"
        )

        # Begin DAG training
        dag_penalty_flavor = self._stage2_kwargs["dag_penalty_flavor"]
        self._model = AutoEncoderLayers(
            self.d,
            [10],
            nn.Sigmoid(),
            model_variance_flavor=self.model_variance_flavor,
            shared_layers=False,
            adjacency_p=2.0,
            dag_penalty_flavor=dag_penalty_flavor,
            mask=self._mask,
            l2_on_dispatcher=l2_on_dispatcher,
        )
        if device:
            move_modules_to_device(self._model, device)

        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._stage2_kwargs["learning_rate"]
        )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self._model = _train(
            self._model,
            dataloader,
            optimizer,
            self._stage2_kwargs,
            val_dataloader=val_dataloader,
            log_wandb=log_wandb,
            print_graph=verbose,
            B_true=B_true,
            start_wandb_epoch=next_epoch,
            device=device,
            **train_kwargs,
        )
        self._train_runtime_in_sec = time.time() - start
        print(f"Finished training in {self._train_runtime_in_sec} seconds.")

    def compute_min_dag_threshold(self) -> float:
        def is_acyclic(adj_matrix):
            return nx.is_directed_acyclic_graph(nx.DiGraph(adj_matrix))

        def bisect(func, a, b, tol=1e-5):
            mid = (a + b) / 2.0
            while (b - a) / 2.0 > tol:
                if func(mid) == True:
                    b = mid
                else:
                    a = mid
                mid = (a + b) / 2.0
            return mid

        adj_matrix = self._model.get_adjacency_matrix().cpu().detach().numpy()
        func = lambda threshold: is_acyclic(adj_matrix > threshold)
        self.threshold = bisect(func, 0, 1)
        return self.threshold

    def get_adjacency_matrix(self, threshold: bool = True) -> np.ndarray:
        assert self._model is not None, "Model has not been trained yet."

        adj_matrix = self._model.get_adjacency_matrix().cpu().detach().numpy()
        return (adj_matrix > self.threshold).astype(int) if threshold else adj_matrix


def _train(
    model,
    dataloader,
    optimizer,
    config,
    val_dataloader=None,
    log_wandb=False,
    print_graph=True,
    B_true=None,
    start_wandb_epoch=0,
    device=None,
    return_next_epoch=False,
    n_epochs_check_validation=20,
    early_stopping=True,
    early_stopping_patience=10,
):
    """Train the model. Assumes dataloader outputs batches alongside interventions."""
    # unpack config
    n_epochs = config["n_epochs"]
    alpha = config["alpha"]
    max_gamma = config["max_gamma"]
    gamma_schedule = config.get("gamma_schedule", "linear")
    if gamma_schedule == "linear":
        gammas = np.linspace(0, max_gamma, n_epochs)
    elif gamma_schedule == "exponential":
        gammas = np.exp(np.linspace(0, np.log(max_gamma), n_epochs))
    else:
        raise ValueError(f"Unknown gamma schedule {gamma_schedule}.")
    beta = config["beta"]
    threshold = config["threshold"]
    freeze_gamma_at_dag = config.get("freeze_gamma_at_dag", False)
    freeze_gamma_threshold = config.get("freeze_gamma_threshold", None)
    n_epochs_check = config["n_epochs_check"]

    is_prescreen = model.dag_penalty_flavor == "none"

    n_observations = dataloader.batch_size * len(dataloader)
    d = dataloader.dataset[0][0].shape[0]

    gamma_cap = None
    early_stopping_patience_counter = 0
    best_model = None
    best_val_loss = np.inf
    #######################
    # Begin training loop #
    #######################
    for epoch in range(n_epochs):
        if gamma_cap is None:
            gamma = gammas[epoch]
        else:
            gamma = gamma_cap

        #######################
        # Begin training step #
        #######################
        epoch_loss = 0
        epoch_loss_details = []
        model.train()
        for batch in dataloader:
            X_batch, mask_interventions_oh, _ = batch
            if device:
                X_batch = X_batch.to(device)
                mask_interventions_oh = mask_interventions_oh.to(device)

            optimizer.zero_grad()
            loss, loss_details = model.loss(
                X_batch,
                alpha,
                beta,
                gamma,
                n_observations,
                mask_interventions_oh=mask_interventions_oh,
                return_detailed_losses=True,
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_loss_details.append(loss_details)

        if epoch % n_epochs_check == 0:
            B_pred = model.get_adjacency_matrix().cpu().detach().numpy()

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
        #####################
        # End training step #
        #####################

        #########################
        # Begin validation step #
        #########################
        if epoch % n_epochs_check_validation == 0 and val_dataloader is not None:
            val_loss = 0.0
            model.eval()
            for batch in val_dataloader:
                X_batch, mask_interventions_oh, _ = batch
                if device:
                    X_batch = X_batch.to(device)
                    mask_interventions_oh = mask_interventions_oh.to(device)

                loss = model.reconstruction_loss(
                    X_batch,
                    mask_interventions_oh=mask_interventions_oh,
                )
                val_loss += loss.item()

            if log_wandb:
                wandb.log(
                    {
                        "epoch": epoch + start_wandb_epoch,
                        "validation_loss": val_loss,
                    }
                )

            if early_stopping:
                if val_loss < best_val_loss:
                    best_model = copy.deepcopy(model)
                    best_val_loss = val_loss
                    early_stopping_patience_counter = 0
                else:
                    early_stopping_patience_counter += 1

            if early_stopping_patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                model = best_model
                break
        #######################
        # End validation step #
        #######################

    #####################
    # End training loop #
    #####################

    if return_next_epoch:
        return model, epoch + 1
    return model
