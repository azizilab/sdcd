import os
import time

import click
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.utils.data
from torch import nn
import wandb
import networkx as nx
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import simulation
from modules import AutoEncoderLayers
from utils import set_random_seed_all
from train_utils import (
    train,
    create_intervention_dataloader,
    create_intervention_dataset,
    subset_interventions,
    compute_metrics,
)

from third_party.dcdi.model import MLPGaussianModel
from third_party.dcdfg.model import MLPModuleGaussianModel
from third_party.callback import (
    AugLagrangianCallback,
    ConditionalEarlyStopping,
    CustomProgressBar,
)


def generate_dataset(n, d, seed, frac_interventions):
    maintain_dataset_size = True
    n_edges = 5 * d
    knockdown_eff = 1.0

    set_random_seed_all(seed)

    B_true = simulation.simulate_dag(d, n_edges, "ER")
    X_full_df, param_dict = simulation.generate_full_interventional_set(
        B_true, n, "mlp", knockdown_eff=knockdown_eff, size_observational=n * (d + 1)
    )
    n_interventions = int(frac_interventions * d)
    X_df = subset_interventions(
        X_full_df, n_interventions, maintain_dataset_size=maintain_dataset_size
    )

    wandb_config_dict = {
        "seed": seed,
        "n": n,
        "d": d,
        "n_edges": n_edges,
        "knockdown_eff": knockdown_eff,
        "frac_interventions": frac_interventions,
        "maintain_dataset_size": maintain_dataset_size,
    }

    return X_df, B_true, wandb_config_dict


def run_sdcdi(X_df, B_true, wandb_config_dict):
    batch_size = 256
    data_loader = create_intervention_dataloader(X_df, batch_size=batch_size)

    ps_learning_rate = 2e-3

    alpha = 1e-2
    beta = 0.005
    n_epochs = 2_000
    n_epochs_check = 200

    use_prescreen = True

    prescreen_config = {
        "n_epochs": n_epochs,
        "alphas": [alpha] * n_epochs,
        "gammas": [0] * n_epochs,
        "beta": beta,
        "n_epochs_check": n_epochs_check,
    }

    mask_threshold = 0.2

    learning_rate = 2e-2

    alpha = 5e-3
    beta = 0.005
    n_epochs = 3_000
    # gamma = 10
    gammas = list(np.linspace(0, 300, n_epochs))
    threshold = 0.3
    freeze_gamma_at_dag = True
    freeze_gamma_threshold = 0.5
    n_epochs_check = 100

    config = {
        "n_epochs": n_epochs,
        "alphas": [alpha] * n_epochs,
        "gammas": gammas,
        "beta": beta,
        "freeze_gamma_at_dag": freeze_gamma_at_dag,
        "freeze_gamma_threshold": freeze_gamma_threshold,
        "threshold": threshold,
        "n_epochs_check": n_epochs_check,
    }

    n, d = wandb_config_dict["n"], wandb_config_dict["d"]
    frac_interventions = wandb_config_dict["frac_interventions"]
    prescreen_str = "_prescreen" if use_prescreen else ""
    name = f"{n}_{d}_{frac_interventions:.2f}_intervention_{prescreen_str}gamma_search_mask_{mask_threshold}_threshold_{threshold}_freeze_gamma_at_dag_{freeze_gamma_threshold}"
    # Log config
    wandb.init(
        project="full-pipeline-simulation",
        name=name,
        config={
            "batch_size": batch_size,
            "ps_learning_rate": ps_learning_rate,
            "use_prescreen": use_prescreen,
            "mask_threshold": mask_threshold,
            "prescreen_config": prescreen_config,
            "learning_rate": learning_rate,
            "config": config,
            **wandb_config_dict,
        },
    )

    start = time.time()
    # No DAG prescreen
    if use_prescreen:
        ps_model = AutoEncoderLayers(
            d,
            [10, 1],
            nn.Sigmoid(),
            shared_layers=False,
            adjacency_p=2.0,
            dag_penalty_flavor="none",
        )
        prescreen_optimizer = torch.optim.Adam(
            ps_model.parameters(), lr=ps_learning_rate
        )

        train(
            ps_model,
            data_loader,
            prescreen_optimizer,
            prescreen_config,
            log_wandb=True,
            print_graph=True,
            B_true=B_true,
        )

        # Create mask for main algo
        mask = (
            ps_model.get_adjacency_matrix().detach().numpy() > mask_threshold
        ).astype(int)
        # np.save(f"saved_mtxs/mask_{name}.npy", mask)
        print(
            f"Recall of mask: {(B_true.astype(bool) & mask.astype(bool)).sum() / B_true.sum()}"
        )
        print(
            f"Fraction of edges in mask: {mask.sum() / (mask.shape[0] * mask.shape[1])}"
        )
    else:
        mask = np.ones((d, d))

    # Begin DAG training
    # mask = np.load(f"saved_mtxs/common_agg_custom_mask.npy")

    dag_penalty_flavor = "scc"
    model = AutoEncoderLayers(
        d,
        [10, 1],
        nn.Sigmoid(),
        shared_layers=False,
        adjacency_p=2.0,
        dag_penalty_flavor=dag_penalty_flavor,
        mask=mask,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train(
        model,
        data_loader,
        optimizer,
        config,
        log_wandb=True,
        print_graph=True,
        B_true=B_true,
        start_wandb_epoch=prescreen_config["n_epochs"] if use_prescreen else 0,
    )
    train_time = time.time() - start

    pred_adj = model.get_adjacency_matrix().detach().numpy()
    pred_adj_thresh = (pred_adj > threshold).astype(int)
    metrics_dict = compute_metrics(pred_adj_thresh, B_true)
    metrics_dict["train_time"] = train_time

    wandb.log(metrics_dict)
    wandb.finish()

    return pred_adj


def run_dcdi(X_df, B_true, wandb_config_dict):
    wandb.init(
        project="full-pipeline-simulation-DCDI",
        name="DCDI",
        config=wandb_config_dict,
    )
    dataset = create_intervention_dataset(X_df, regime_format=True)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))]
    )

    start = time.time()
    model = MLPGaussianModel(
        B_true.shape[0],
        2,
        16,
        lr_init=1e-3,
        reg_coeff=0.1,
        constraint_mode="exp",
    )

    early_stop_1_callback = ConditionalEarlyStopping(
        monitor="Val/aug_lagrangian",
        min_delta=1e-4,
        patience=5,
        verbose=True,
        mode="min",
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        gpus=1,
        max_epochs=5000,
        logger=WandbLogger(
            project="full-pipeline-simulation-DCDI", log_model=True, reinit=True
        ),
        val_check_interval=1.0,
        callbacks=[AugLagrangianCallback(), early_stop_1_callback, CustomProgressBar()],
    )
    trainer.fit(
        model,
        torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=4),
        torch.utils.data.DataLoader(val_dataset, num_workers=8, batch_size=256),
    )

    train_time = time.time() - start

    model.module.threshold()
    pred_adj = np.array(model.module.get_w_adj().detach().cpu().numpy() > 0, dtype=int)
    metrics_dict = compute_metrics(pred_adj, B_true)
    metrics_dict["train_time"] = train_time

    wandb.log(metrics_dict)
    wandb.finish()

    return pred_adj


def run_dcdfg(X_df, B_true, wandb_config_dict):
    wandb.init(
        project="full-pipeline-simulation-DCDFG",
        name="DCDFG",
        config=wandb_config_dict,
    )
    dataset = create_intervention_dataset(X_df, regime_format=True)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))]
    )

    start = time.time()
    model = MLPGaussianModel(
        B_true.shape[0],
        2,
        16,
        lr_init=1e-3,
        reg_coeff=0.1,
        constraint_mode="exp",
    )
    model = MLPModuleGaussianModel(
        B_true.shape[0],
        2,
        20,
        16,
        lr_init=1e-3,
        reg_coeff=0.1,
        constraint_mode="exp",
    )

    early_stop_1_callback = ConditionalEarlyStopping(
        monitor="Val/aug_lagrangian",
        min_delta=1e-4,
        patience=5,
        verbose=True,
        mode="min",
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        gpus=1,
        max_epochs=5000,
        logger=WandbLogger(
            project="full-pipeline-simulation-DCDFG", log_model=True, reinit=True
        ),
        val_check_interval=1.0,
        callbacks=[AugLagrangianCallback(), early_stop_1_callback, CustomProgressBar()],
    )
    trainer.fit(
        model,
        torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=4),
        torch.utils.data.DataLoader(val_dataset, num_workers=8, batch_size=256),
    )

    train_time = time.time() - start

    model.module.threshold()
    pred_adj = np.array(model.module.weight_mask.detach().cpu().numpy() > 0, dtype=int)
    metrics_dict = compute_metrics(pred_adj, B_true)
    metrics_dict["train_time"] = train_time

    wandb.log(metrics_dict)
    wandb.finish()

    return pred_adj


def save_B_pred(
    B_pred, n, d, seed, frac_interventions, method_name, dirname="saved_mtxs/"
):
    filename = f"B_pred_{n}_{d}_{seed}_{frac_interventions}_{method_name}.npy"
    # check if dir exists
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filepath = os.path.join(dirname, filename)
    np.save(filepath, B_pred)


@click.command()
@click.option("--n", default=50, help="Per interventional subset")
@click.option("--d", type=int, help="Number of dimensions")
@click.option("--seed", default=0, help="Random seed")
@click.option("--frac_interventions", default=1.0, help="Fraction of interventions")
@click.option("--run_baselines", default=True, help="Run baselines")
def run_full_pipeline(n, d, seed, frac_interventions, run_baselines):
    X_df, B_true, wandb_config_dict = generate_dataset(n, d, seed, frac_interventions)

    B_pred = run_sdcdi(X_df, B_true, wandb_config_dict)
    save_B_pred(B_pred, n, d, seed, frac_interventions, "sdcdi")

    if run_baselines:
        #try:
        #    B_pred = run_dcdi(X_df, B_true, wandb_config_dict)
        #    save_B_pred(B_pred, n, d, seed, frac_interventions, "dcdi")
        #except ValueError:
        #    print("Skipping DCDI as it failed to scale.")

        B_pred = run_dcdfg(X_df, B_true, wandb_config_dict)
        save_B_pred(B_pred, n, d, seed, frac_interventions, "dcdfg")


if __name__ == "__main__":
    run_full_pipeline()
