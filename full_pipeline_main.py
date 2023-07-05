import os

import click
import numpy as np
import wandb

import simulation
from models._dagma import DAGMA
from utils import set_random_seed_all
from train_utils import (
    create_intervention_dataset,
    subset_interventions,
)

from models import SDCI, DCDI, DCDFG


def generate_dataset(n, d, seed, frac_interventions, n_edges_per_d=5):
    assert n_edges_per_d < d
    maintain_dataset_size = True
    n_edges = n_edges_per_d * d
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


def run_sdci(X_df, B_true, wandb_config_dict, wandb_project):
    dataset = create_intervention_dataset(X_df, regime_format=False)
    wandb_config_dict["model"] = "SDCI"
    model = SDCI()
    model.train(
        dataset,
        log_wandb=True,
        wandb_project=wandb_project,
        wandb_config_dict=wandb_config_dict,
        B_true=B_true,
    )
    metrics_dict = model.compute_metrics(B_true)
    metrics_dict["train_time"] = model._train_runtime_in_sec

    wandb.log(metrics_dict)
    wandb.finish()

    return model.get_adjacency_matrix()


def run_dcdi(X_df, B_true, wandb_config_dict, wandb_project):
    dataset = create_intervention_dataset(X_df, regime_format=True)
    wandb_config_dict["model"] = "DCDI"
    model = DCDI()
    model.train(
        dataset,
        log_wandb=True,
        wandb_project=wandb_project,
        wandb_config_dict=wandb_config_dict,
    )
    metrics_dict = model.compute_metrics(B_true)
    metrics_dict["train_time"] = model._train_runtime_in_sec

    wandb.log(metrics_dict)
    wandb.finish()

    return model.get_adjacency_matrix()


def run_dcdfg(X_df, B_true, wandb_config_dict, wandb_project):
    dataset = create_intervention_dataset(X_df, regime_format=False)
    wandb_config_dict["model"] = "DCDFG"
    model = DCDFG()
    model.train(
        dataset,
        log_wandb=True,
        wandb_project=wandb_project,
        wandb_config_dict=wandb_config_dict,
    )
    metrics_dict = model.compute_metrics(B_true)
    metrics_dict["train_time"] = model._train_runtime_in_sec

    wandb.log(metrics_dict)
    wandb.finish()

    return model.get_adjacency_matrix()


def run_dagma(X_df, B_true, wandb_config_dict, wandb_project):
    dataset = create_intervention_dataset(X_df, regime_format=True)
    wandb_config_dict["model"] = "DAGMA"
    model = DAGMA()
    model.train(
        dataset,
        log_wandb=True,
        wandb_project=wandb_project,
        wandb_config_dict=wandb_config_dict,
    )
    metrics_dict = model.compute_metrics(B_true)
    metrics_dict["train_time"] = model._train_runtime_in_sec

    wandb.log(metrics_dict)
    wandb.finish()

    return model.get_adjacency_matrix()


def save_B_pred(B_pred, n, d, seed, frac_interventions, method_name, dirname="saved_mtxs/"):
    filename = f"B_pred_{n}_{d}_{seed}_{frac_interventions}_{method_name}.npy"
    # check if dir exists
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filepath = os.path.join(dirname, filename)
    np.save(filepath, B_pred)


@click.command()
@click.option("--n", default=50, help="Per interventional subset")
@click.option("--d", default=20, type=int, help="Number of dimensions")
@click.option(
    "--n_edges_per_d", type=int, default=5, help="Number of edges per dimension"
)
@click.option("--seed", default=0, help="Random seed")
@click.option("--frac_interventions", default=1.0, help="Fraction of interventions")
@click.option(
    "--model", default="all", help="Model to run. Choices are [all, sdci, dcdi, dcdfg, dagma]"
)
@click.option(
    "--save_mtxs", default=True, help="Save matrices to saved_mtxs/ directory"
)
@click.option(
    "--wandb-project", default="full-pipeline-simulation", help="Wandb project name"
)
def run_full_pipeline(n, d, n_edges_per_d, seed, frac_interventions, model, save_mtxs, wandb_project):
    X_df, B_true, wandb_config_dict = generate_dataset(
        n, d, seed, frac_interventions, n_edges_per_d=n_edges_per_d
    )
    if save_mtxs:
        save_B_pred(B_true, n, d, seed, frac_interventions, "gt")

    if model == "all" or model == "sdci":
        B_pred = run_sdci(X_df, B_true, wandb_config_dict, wandb_project)
        if save_mtxs:
            save_B_pred(B_pred, n, d, seed, frac_interventions, "sdci")

    if model == "all" or model == "dcdi":
        try:
            B_pred = run_dcdi(X_df, B_true, wandb_config_dict, wandb_project)
            if save_mtxs:
                save_B_pred(B_pred, n, d, seed, frac_interventions, "dcdi")
        except ValueError:
            print("Skipping DCDI as it failed to scale.")
            wandb.finish()

    if model == "all" or model == "dcdfg":
        B_pred = run_dcdfg(X_df, B_true, wandb_config_dict, wandb_project)
        if save_mtxs:
            save_B_pred(B_pred, n, d, seed, frac_interventions, "dcdfg")

    if model == "all" or model == "dagma":
        B_pred = run_dagma(X_df, B_true, wandb_config_dict, wandb_project)
        if save_mtxs:
            save_B_pred(B_pred, n, d, seed, frac_interventions, "dagma")


if __name__ == "__main__":
    run_full_pipeline()
