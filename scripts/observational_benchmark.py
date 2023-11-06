import os
import pprint

import click
import numpy as np
import pandas as pd
import torch

import wandb
from sdcd.models import (DAGMA, DCDFG, DCDI, GIES, NOBEARS, NOTEARS, SDCD,
                         Sortnregress)
from sdcd.simulated_data import random_model_gaussian_global_variance
from sdcd.utils import create_intervention_dataset, set_random_seed_all

MODEL_CLS_DCT = {
    model_cls.__name__: model_cls
    for model_cls in [
        SDCD,
        DAGMA,
        NOBEARS,
        NOTEARS,
        Sortnregress,
        DCDFG,
        DCDI,
        GIES,
    ]
}

# ablation study and GPU
MODEL_CLS_DCT["SDCD-GPU"] = SDCD
MODEL_CLS_DCT["SDCD-no-s1"] = SDCD
MODEL_CLS_DCT["SDCD-no-s1-2"] = SDCD
MODEL_CLS_DCT["SDCD-warm"] = SDCD
MODEL_CLS_DCT["SDCD-warm-nomask"] = SDCD


def generate_observational_dataset(
    n,
    d,
    n_edges,
    seed,
    dataset="ER",
    scale=None,
    normalize=False,
    save_dir=None,
):
    assert n_edges <= d * (d - 1)

    if save_dir is not None:
        X_path = os.path.join(save_dir, "X.csv")
        Btrue_path = os.path.join(save_dir, "Btrue.csv")
        if os.path.exists(X_path) and os.path.exists(Btrue_path):
            X = pd.read_csv(X_path, index_col=0)
            B_true = np.loadtxt(Btrue_path, delimiter=",").astype(np.int64)
            return X, B_true
    else:
        raise ValueError("should use save_dir")

    set_random_seed_all(seed)
    if dataset == "ER":
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
    )
    B_true = true_causal_model.adjacency
    X_df = true_causal_model.generate_dataframe_from_all_distributions(
        n_samples_control=n, n_samples_per_intervention=0, subset_interventions=[]
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
    model_cls_name,
    dataset,
    B_true,
    model_kwargs=None,
    wandb_project=None,
    wandb_config_dict=None,
    save_dir=None,
    force=False,
):
    model_kwargs = model_kwargs or {}
    wandb_config_dict = wandb_config_dict or {}
    model_cls = MODEL_CLS_DCT[model_cls_name]

    if save_dir is not None:
        save_path = os.path.join(save_dir, f"{model_cls_name}.csv")
        if os.path.exists(save_path) and not force:
            print(f"Already ran {model_cls_name}, skipping. Use force=True to rerun.")
            return

    wandb_config_dict["model"] = model_cls_name
    model = model_cls()
    extra_kwargs = {}
    # ablation study and GPU
    if "SDCD" in model_cls_name:
        extra_kwargs["B_true"] = B_true
        if model_cls_name == "SDCD-GPU":
            # run SDCD on GPU (and fail if gpu is unavailable)
            if not torch.cuda.is_available():
                print("CUDA not available, aborting.")
                return
            else:
                device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        extra_kwargs["device"] = device

        if model_cls_name == "SDCD-no-s1":
            # skip stage 1, stage 2 just has a mask for self-loops
            extra_kwargs["skip_stage1"] = True

        if model_cls_name == "SDCD-no-s1-2":
            # skip stage 1, stage 2 just has a mask for self-loops
            # but set alpha and beta of stage 2 like those from stage 1
            extra_kwargs["skip_stage1"] = True
            from sdcd.models._sdcd import _DEFAULT_STAGE1_KWARGS

            extra_kwargs["stage2_kwargs"] = {
                "alpha": _DEFAULT_STAGE1_KWARGS["alpha"],
                "beta": _DEFAULT_STAGE1_KWARGS["beta"],
            }

        if model_cls_name == "SDCD-warm":
            # warm start the input layer in stage 2 from stage 1
            extra_kwargs["warm_start"] = True

        if model_cls_name == "SDCD-warm-nomask":
            # warm start the input layer in stage 2 from stage 1,
            # but ignore the mask from stage 1
            extra_kwargs["warm_start"] = True
            extra_kwargs["skip_masking"] = True

        extra_kwargs["wandb_name"] = model_cls_name

    model.train(
        dataset,
        log_wandb=True,
        wandb_project=wandb_project,
        wandb_config_dict=wandb_config_dict,
        **extra_kwargs,
    )
    metrics_dict = model.compute_metrics(B_true)
    metrics_dict["model"] = model_cls_name
    metrics_dict["train_time"] = model._train_runtime_in_sec

    wandb.log(metrics_dict)
    wandb.finish()

    B_pred = model.get_adjacency_matrix()
    if save_path:
        np.savetxt(save_path, B_pred, delimiter=",")
    return metrics_dict


@click.command()
@click.option("--n", default=10000, help="Per interventional subset")
@click.option("--d", default=100, type=int, help="Number of dimensions")
@click.option(
    "--p",
    type=float,
    default=0.05,
    help="Expected edge density. (Ignored if s is specified)",
)
@click.option(
    "--s", type=int, default=4, help="Number of edges per dimension.  (Overrides p)"
)
@click.option("--seed", default=0, help="Random seed")
@click.option("--model", type=str, default="SDCD", help="Which models to run")
@click.option("--force", default=True, help="If results exist, redo anyways.")
@click.option(
    "--save_mtxs", default=True, help="Save matrices to saved_mtxs/ directory"
)
def _run_full_pipeline(n, d, p, s, seed, model, force, save_mtxs):
    if s != -1:
        n_edges = s * d
    else:
        n_edges = int(p * d * (d - 1))
    dataset_name = f"observational_n{n}_d{d}_edges{n_edges}_seed{seed}"
    save_dir = f"paper_experiments/observational/{dataset_name}"
    if save_mtxs:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    print(f"Using {n_edges} edges for {d} variables")
    X, B_true = generate_observational_dataset(
        n,
        d,
        n_edges,
        seed,
        normalize=True,
        save_dir=save_dir,
    )
    X_dataset = create_intervention_dataset(X)

    results_save_path = os.path.join(save_dir, "results.csv")
    results_df_rows = []
    if os.path.exists(results_save_path):
        results_df = pd.read_csv(results_save_path, index_col=0)
        results_df_rows = results_df.to_dict(orient="records")

    if model == "all":
        model_classes = MODEL_CLS_DCT
    else:
        model_classes = {model: MODEL_CLS_DCT[model]}
    wandb_config_dict = {
        "n": n,
        "d": d,
        "p": p,
        "s": s,
        "seed": seed,
    }
    for model_cls_name, model_cls in model_classes.items():
        # try:
        set_random_seed_all(0)
        metrics_dict = run_model(
            model_cls_name,
            X_dataset,
            B_true,
            model_kwargs={},
            wandb_project="observational_benchmark_v3",
            wandb_config_dict=wandb_config_dict,
            save_dir=save_dir if save_mtxs else None,
            force=force,
        )
        # except Exception as e:
        #     print(f"Failed to run {model_cls_name}")
        #     print(e)
        #     wandb.finish()
        #     continue
        if metrics_dict is None:
            continue

        pprint.pprint(metrics_dict)
        results_df_rows.append(metrics_dict)
        results_df = pd.DataFrame.from_records(results_df_rows)
        results_df.to_csv(f"saved_mtxs/{dataset_name}/{model_cls_name}.csv")


def create_data(n, d, s, seed):
    """Helper to just generate the data used on observational benchmark
    To make sure all models are evaluated on the same data."""
    n_edges = s * d
    dataset_name = f"observational_n{n}_d{d}_edges{n_edges}_seed{seed}"
    save_dir = f"paper_experiments/observational/{dataset_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print(f"Using {n_edges} edges for {d} variables")
    generate_observational_dataset(
        n,
        d,
        n_edges,
        seed,
        normalize=True,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    _run_full_pipeline()
    # ds = [10, 20, 30, 40, 50, 70, 100]
    # seeds = [0, 1, 2]
    # ss = [4]
    # ns = [10000]
    # for n, d, s, seed in itertools.product(ns, ds, ss, seeds):
    #     print(f"Creating data n={n}, d={d}, s={s}, seed={seed}")
    #     create_data(n, d, s, seed)
