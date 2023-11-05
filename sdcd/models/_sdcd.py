import copy
import time
from typing import Optional, Literal, Union
import itertools

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import wandb
import networkx as nx

from ..utils import (
    print_graph_from_weights,
    move_modules_to_device,
    TorchStandardScaler,
    compute_metrics,
    train_val_split,
    set_random_seed_all,
)

from .base._base_model import BaseModel
from .modules import AutoEncoderLayers


_DEFAULT_STAGE1_KWARGS = {
    "learning_rate": 2e-3,
    "batch_size": 256,
    "n_epochs": 2000,
    "alpha": 1e-2,
    "beta": 2e-4,
    "gamma_increment": 0,
    "n_epochs_check": 100,
    "mask_threshold": 0.2,
}
_DEFAULT_STAGE2_KWARGS = {
    "learning_rate": 1e-3,
    "batch_size": 256,
    "n_epochs": 2000,
    "alpha": 5e-4,
    "beta": 5e-3,
    "gamma_increment": 0.005,
    "gamma_schedule": "linear",
    "freeze_gamma_at_dag": True,
    "freeze_gamma_threshold": 0.01,
    "threshold": 0.1,
    "n_epochs_check": 100,
    "dag_penalty_flavor": "power_iteration",
}

_DEFAULT_MODEL_KWARGS = {
    "num_layers": 1,
    "dim_hidden": 10,
    "power_iteration_n_steps": 15,
}

NUM_WORKERS = 0


class SDCD(BaseModel):
    def __init__(
        self,
        model_variance_flavor: Literal["unit", "nn", "parameter"] = "nn",
        standard_scale: bool = False,
        use_gumbel: bool = False,
    ):
        super().__init__()
        self.model_variance_flavor = model_variance_flavor
        self.standard_scale = standard_scale
        self.use_gumbel = use_gumbel
        self._stage1_kwargs = None
        self._stage2_kwargs = None
        self._model_kwargs = None
        self._trained = False

    def train(
        self,
        dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        val_fraction: float = 0.2,
        log_wandb: bool = False,
        finetune: bool = False,
        wandb_project: str = "SDCD",
        wandb_name: str = "SDCD",
        wandb_config_dict: Optional[dict] = None,
        B_true: Optional[np.ndarray] = None,
        stage1_kwargs: Optional[dict] = None,
        stage2_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
        train_kwargs: Optional[dict] = None,
        verbose: bool = False,
        device: Optional[torch.device] = None,
        skip_stage1: bool = False,
        warm_start: bool = False,
        skip_masking: bool = False,
    ):
        set_random_seed_all(0)

        self._stage1_kwargs = {**_DEFAULT_STAGE1_KWARGS.copy(), **(stage1_kwargs or {})}
        self._stage2_kwargs = {**_DEFAULT_STAGE2_KWARGS.copy(), **(stage2_kwargs or {})}
        self._model_kwargs = {**_DEFAULT_MODEL_KWARGS.copy(), **(model_kwargs or {})}
        if skip_stage1:
            self._stage1_kwargs["n_epochs"] = 0
            self._stage1_kwargs["mask_threshold"] = -1.0

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

        if val_dataset is None:
            dataset, val_dataset = train_val_split(dataset, val_fraction=val_fraction)
        val_dataloader = DataLoader(
            val_dataset, batch_size=ps_batch_size, num_workers=NUM_WORKERS
        )

        ps_dataloader = DataLoader(
            dataset, batch_size=ps_batch_size, shuffle=True, num_workers=NUM_WORKERS
        )
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
                    "model_kwargs": self._model_kwargs,
                    "number_of_samples": len(dataset),
                    **wandb_config_dict,
                },
            )

        start = time.time()
        # Stage 1: Pre-selection
        self._ps_model = AutoEncoderLayers(
            self.d,
            [self._model_kwargs["dim_hidden"]] * self._model_kwargs["num_layers"],
            nn.Sigmoid(),
            model_variance_flavor=self.model_variance_flavor,
            shared_layers=False,
            adjacency_p=2.0,
            dag_penalty_flavor="none",
            use_gumbel=self.use_gumbel,
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
        if not skip_masking:
            self._mask = (
                self._ps_model.get_adjacency_matrix().cpu().detach().numpy()
                > mask_threshold
            ).astype(int) * (1 - np.eye(self.d, dtype=int))
        else:
            self._mask = 1 - np.eye(self.d, dtype=int)
        fraction_edges_mask = self._mask.sum() / (
            self._mask.shape[0] * self._mask.shape[1]
        )
        print(f"Fraction of possible edges in mask: {fraction_edges_mask}")
        if B_true is not None:
            recall_mask = (
                B_true.astype(bool) & self._mask.astype(bool)
            ).sum() / B_true.sum()
            print(f"Recall of mask: {recall_mask}")
        else:
            recall_mask = -1
        if log_wandb:
            wandb.log(
                {"fraction_edges_mask": fraction_edges_mask, "recall_mask": recall_mask}
            )

        # Begin DAG training
        dag_penalty_flavor = self._stage2_kwargs["dag_penalty_flavor"]
        print(dag_penalty_flavor)
        self._model = AutoEncoderLayers(
            self.d,
            [self._model_kwargs["dim_hidden"]] * self._model_kwargs["num_layers"],
            nn.Sigmoid(),
            model_variance_flavor=self.model_variance_flavor,
            shared_layers=False,
            adjacency_p=2.0,
            dag_penalty_flavor=dag_penalty_flavor,
            mask=self._mask,
            use_gumbel=self.use_gumbel,
            power_iteration_n_steps=self._model_kwargs["power_iteration_n_steps"],
        )
        if warm_start:
            warm_start_tensor = self._ps_model.layers[0]._weight.data
            self._model.layers[0]._weight.data = warm_start_tensor

        if device:
            move_modules_to_device(self._model, device)

        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._stage2_kwargs["learning_rate"]
        )

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
        )
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

        # Save unthresholded matrix because thresholding is destructive.
        self._adj_matrix = self._model.get_adjacency_matrix().cpu().detach().numpy()

        if finetune:
            print("Fixing adjacency matrix.")
            if self.use_gumbel:
                self.fix_gumbel_threshold()
            else:
                self._final_mask = self.adjacency_dag_at_threshold(
                    self._adj_matrix, self.threshold
                ).astype(int)
                self._model.update_mask(self._final_mask)

            print("Beginning finetune.")
            self._model = _train(
                self._model,
                dataloader,
                optimizer,
                {**self._stage2_kwargs, "gamma_increment": 0},
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

        self._trained = True

    def fix_gumbel_threshold(self):
        assert (
            self.use_gumbel
        ), "Not applicable for models that do not use Gumbel adjacency."
        with torch.no_grad():
            w_adj = self._model.get_adjacency_matrix()
            higher = (w_adj > self.threshold).type_as(w_adj)
            lower = (w_adj <= self.threshold).type_as(w_adj)
            self._model.layers[0].gumbel_adjacency.log_alpha.copy_(
                higher * 100 + lower * -100
            )
            self._model.layers[0].gumbel_adjacency.log_alpha.requires_grad = False
            self._model.layers[0].adjacency_mask.copy_(higher)

    @staticmethod
    def adjacency_dag_at_threshold(adjacency, threshold=0.1):
        """Threshold adjacency matrix at the threshold and removes edges that makes it cyclic."""
        edges = [
            (i, j, adjacency[i, j])
            for i, j in itertools.product(range(adjacency.shape[0]), repeat=2)
        ]
        edges.sort(key=lambda x: -x[2])
        g = nx.DiGraph()
        g.add_nodes_from(range(adjacency.shape[0]))
        for e in edges:
            if e[2] < threshold:
                break
            if nx.has_path(g, e[1], e[0]):
                continue
            else:
                g.add_edge(e[0], e[1])
        return nx.to_numpy_array(g)

    def get_adjacency_matrix(self, threshold: Union[bool, float] = True) -> np.ndarray:
        assert self._model is not None, "Model has not been trained yet."

        if threshold == False:
            return self._adj_matrix

        if type(threshold) == bool:
            threshold = self.threshold

        return self.adjacency_dag_at_threshold(self._adj_matrix, threshold).astype(int)

    def compute_nll(self, dataset: Dataset) -> float:
        total_loss = 0.0
        dataloader = DataLoader(dataset, batch_size=256)
        self._model.eval()
        for batch in dataloader:
            X_batch, mask_interventions_oh, _ = batch
            if self._model.device:
                device = self._model.device
                X_batch = X_batch.to(device)
                mask_interventions_oh = mask_interventions_oh.to(device)

            loss = (
                self._model.reconstruction_loss(
                    X_batch,
                    mask_interventions_oh=mask_interventions_oh,
                )
                * X_batch.shape[0]
            )
            total_loss += loss.item()
        return total_loss / len(dataset)


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
    gamma_increment = config["gamma_increment"]
    gamma_schedule = config.get("gamma_schedule", "linear")
    if gamma_schedule == "linear":
        gammas = np.linspace(0, gamma_increment * n_epochs, n_epochs)
    elif "power" in gamma_schedule:
        power = float(gamma_schedule.split("_")[1])
        gammas = np.power(np.linspace(0, gamma_increment * n_epochs, n_epochs), power)
    elif gamma_schedule == "exponential":
        gammas = np.exp(np.linspace(0, np.log(gamma_increment * n_epochs), n_epochs))
    else:
        raise ValueError(f"Unknown gamma schedule {gamma_schedule}.")
    beta = config["beta"]
    threshold = config["threshold"]
    freeze_gamma_at_dag = config.get("freeze_gamma_at_dag", False)
    freeze_gamma_threshold = config.get("freeze_gamma_threshold", None)
    n_epochs_check = config["n_epochs_check"]

    if freeze_gamma_at_dag:
        assert freeze_gamma_threshold <= threshold

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
    gamma_idx = 0
    epoch = 0
    for epoch in range(n_epochs):
        if gamma_cap is None:
            gamma = gammas[gamma_idx]
            gamma_idx += 1
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

            epoch_loss /= len(dataloader)
            if B_true is not None:
                adjacency = SDCD.adjacency_dag_at_threshold(B_pred, threshold)
                metrics_dict = compute_metrics(adjacency.astype(int), B_true)
                print(
                    f"Epoch {epoch}: loss={epoch_loss:.2f}, score={metrics_dict['score']}, shd={metrics_dict['shd']}, gamma={gamma:.2f}"
                )
                adjacency_half_threshold = SDCD.adjacency_dag_at_threshold(
                    B_pred, threshold / 2
                )
                metrics_dict_half = compute_metrics(
                    adjacency_half_threshold.astype(int), B_true
                )
                metrics_dict_half = {
                    k + "_half_th": v for k, v in metrics_dict_half.items()
                }
                metrics_dict.update(metrics_dict_half)

                adjacency_double_threshold = SDCD.adjacency_dag_at_threshold(
                    B_pred, threshold * 2
                )
                metrics_dict_double = compute_metrics(
                    adjacency_double_threshold.astype(int), B_true
                )
                metrics_dict_double = {
                    k + "_double_th": v for k, v in metrics_dict_double.items()
                }
                metrics_dict.update(metrics_dict_double)

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
            B_pred = model.get_adjacency_matrix().cpu().detach().numpy()

            # Check dag if freeze_gamma_at_dag is True and beyond a warmup period of epochs to avoid seeing a trivial DAG.
            if epoch > 1 and freeze_gamma_at_dag and gamma_cap is None:
                is_dag_freeze = nx.is_directed_acyclic_graph(
                    nx.DiGraph(B_pred > freeze_gamma_threshold)
                )
                if is_dag_freeze:
                    # If we hit a DAG, freeze the gamma value
                    gamma_cap = gamma
            elif freeze_gamma_at_dag and gamma_cap is not None:
                is_dag_thresh = nx.is_directed_acyclic_graph(
                    nx.DiGraph(B_pred > threshold)
                )
                if not is_dag_thresh:
                    # If we have frozen the gamma value but the graph is not a DAG, unfreeze it
                    gamma_cap = None
                    early_stopping_patience_counter = 0

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

            # Early stopping patience should only increase when gamma is frozen (the graph is still a DAG at threshold)
            if early_stopping and (not freeze_gamma_at_dag or gamma_cap is not None):
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
