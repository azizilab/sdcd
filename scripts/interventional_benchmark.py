import sys

sys.path.append("./")

import os

import click
import numpy as np
import pandas as pd
import wandb
import pprint
from torch.utils.data import DataLoader

from sdcd.utils import (
    set_random_seed_all,
    create_intervention_dataset,
    train_val_split,
)
from sdcd.simulated_data import random_model_gaussian_global_variance
from sdcd.models import SDCD, DCDFG, DCDI, GIES

MODEL_CLS_DCT = {
    model_cls.__name__: model_cls
    for model_cls in [
        SDCD,
        DCDFG,
        DCDI,
        GIES,
    ]
}


def generate_interventional_dataset(
    n,
    n_per_intervention,
    d,
    n_edges,
    seed,
    dataset="ER",
    scale=None,
    normalize=False,
    save_dir=None,
    **kwargs,
):
    assert n_edges <= d * (d - 1)

    if save_dir is not None:
        X_path = os.path.join(save_dir, f"X.csv")
        Btrue_path = os.path.join(save_dir, f"Btrue.csv")
        if os.path.exists(X_path) and os.path.exists(Btrue_path):
            X = pd.read_csv(X_path, index_col=0)
            B_true = np.loadtxt(Btrue_path, delimiter=",").astype(np.int64)
            return X, B_true

    set_random_seed_all(seed)
    if dataset == "ER" or dataset == "diag_band":
        scale = scale or 0.5
    elif dataset == "chain":
        scale = scale or (lambda depth: 2 * (d - depth) / d)
    else:
        raise ValueError(f"dataset type {dataset} not recognized")

    true_causal_model = random_model_gaussian_global_variance(
        d,
        n_edges,
        dag_type=dataset,
        scale=scale,
        hard=True,
        **kwargs,
    )
    B_true = true_causal_model.adjacency
    X_df = true_causal_model.generate_dataframe_from_all_distributions(
        n_samples_control=n,
        n_samples_per_intervention=n_per_intervention,
    )
    # normalize the data (except the last column, which is the intervention indicator)
    if normalize:
        X_df.iloc[:, :-1] = (X_df.iloc[:, :-1] - X_df.iloc[:, :-1].mean()) / X_df.iloc[
            :, :-1
        ].std()

    if save_dir is not None:
        X_df.to_csv(X_path)
        np.savetxt(Btrue_path, B_true, delimiter=",")

    return X_df, B_true


def run_model(
    model_cls,
    dataset,
    test_dataset,
    B_true,
    model_kwargs=None,
    wandb_project=None,
    wandb_config_dict=None,
    save_dir=None,
    force=False,
    **extra_kwargs,
):
    model_kwargs = model_kwargs or {}
    wandb_config_dict = wandb_config_dict or {}
    model_cls_name = model_cls.__name__

    if save_dir is not None:
        save_path = os.path.join(save_dir, f"{model_cls_name}.csv")
        if os.path.exists(save_path) and not force:
            print(f"Already ran {model_cls_name}, skipping. Use force=True to rerun.")
            return

    wandb_config_dict["model"] = model_cls_name
    model = model_cls()
    if model_cls_name == "SDCD":
        extra_kwargs["B_true"] = B_true

    model.train(
        dataset,
        finetune=False,
        log_wandb=True,
        wandb_project=wandb_project,
        wandb_config_dict=wandb_config_dict,
        **extra_kwargs,
    )
    metrics_dict = model.compute_metrics(B_true)
    metrics_dict["model"] = model_cls_name
    metrics_dict["train_time"] = model._train_runtime_in_sec

    # Compute I-NLL
    # i_nll = model.compute_nll(test_dataset)
    # metrics_dict["I-NLL"] = i_nll

    wandb.log(metrics_dict)
    wandb.finish()

    B_pred = model.get_adjacency_matrix()
    if save_dir is not None:
        np.savetxt(save_path, B_pred, delimiter=",")
    return metrics_dict


@click.command()
@click.option("--n", default=10000, help="Per interventional subset")
@click.option("--n_per_intervention", default=500, help="Per interventional subset")
@click.option("--d", default=10, type=int, help="Number of dimensions")
@click.option("--p", type=float, default=0.1, help="Expected edge density")
@click.option(
    "--s", type=int, default=-1, help="Number of edges per dimension.  (Overrides p)"
)
@click.option("--seed", default=0, help="Random seed")
@click.option("--model", type=str, default="all", help="Which models to run")
@click.option("--force", default=True, help="If results exist, redo anyways.")
@click.option("--sweep_frac", default=False, help="Sweep frac of interventions")
@click.option("--wandb_project_name", default="intervention", help="Wandb project name")
@click.option(
    "--save_mtxs", default=True, help="Save matrices to saved_mtxs/ directory"
)
def _run_full_pipeline(
    n,
    n_per_intervention,
    d,
    p,
    s,
    seed,
    model,
    force,
    sweep_frac,
    wandb_project_name,
    save_mtxs,
):
    if s != -1:
        n_edges = s * d
    else:
        n_edges = int(p * d * (d - 1))
    dataset_name = f"interventional_hard_n{n}_d{d}_edges{n_edges}_seed{seed}"
    save_dir = f"saved_mtxs/{dataset_name}"
    if save_mtxs:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    print(f"Using {n_edges} edges for {d} variables")
    X_df, B_true = generate_interventional_dataset(
        n,
        n_per_intervention,
        d,
        n_edges,
        seed,
        normalize=True,
        save_dir=save_dir,
    )
    X_dataset = create_intervention_dataset(X_df)

    # Hold out 10% interventions for I-NLL
    X_train_dataset, X_test_dataset = train_val_split(
        X_dataset, flavor="I-NLL", val_fraction=0.1, seed=seed
    )
    X_train_dataset = X_dataset

    if model == "all":
        model_classes = MODEL_CLS_DCT
    else:
        model_classes = {model: MODEL_CLS_DCT[model]}

    results_save_path = os.path.join(save_dir, "results.csv")
    results_df_rows = []
    if os.path.exists(results_save_path):
        results_df = pd.read_csv(results_save_path, index_col=0)
        results_df_rows = results_df.to_dict(orient="records")

    if sweep_frac:
        intervention_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    else:
        intervention_fractions = [1.0]  # of remaining interventions

    for intervention_frac in intervention_fractions:
        X_train_dataset_subset, _ = train_val_split(
            X_train_dataset,
            flavor="I-NLL",
            val_fraction=1 - intervention_frac,
            seed=seed,
        )
        wandb_config_dict = {
            "n": n,
            "d": d,
            "p": p,
            "s": s,
            "seed": seed,
            "intervention_frac": intervention_frac,
        }

        for model_cls_name, model_cls in model_classes.items():
            metrics_dict = run_model(
                model_cls,
                X_train_dataset_subset,
                X_test_dataset,
                B_true,
                model_kwargs={},
                wandb_project=wandb_project_name,
                wandb_config_dict=wandb_config_dict,
                save_dir=save_dir if save_mtxs else None,
                force=force,
            )
            if metrics_dict is None:
                continue

            pprint.pprint(metrics_dict)
            metrics_dict["intervention_fraction"] = intervention_frac
            results_df_rows.append(metrics_dict)
            results_df = pd.DataFrame.from_records(results_df_rows)
            intervention_dir = os.path.join(
                save_dir, f"intervention_frac{intervention_frac}"
            )
            if not os.path.exists(intervention_dir):
                os.makedirs(intervention_dir, exist_ok=True)
            results_df.to_csv(os.path.join(intervention_dir, f"{model_cls_name}.csv"))


if __name__ == "__main__":
    _run_full_pipeline()
