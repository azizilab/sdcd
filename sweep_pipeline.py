import math
import pdb

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.utils.data
from torch import nn
import wandb
import networkx as nx

import simulation
from modules import AutoEncoderLayers
from utils import set_random_seed_all, print_graph_from_weights, ks_test_screen
from train_utils import train, create_intervention_dataloader, subset_interventions

dataset_seed = 0

def main():
    run_seed = np.random.randint(0, 2**32)
    set_random_seed_all(dataset_seed)

    n, d = 50, 100
    n_edges = 5 * d
    knockdown_eff = 1.0
    B_true = simulation.simulate_dag(d, n_edges, "ER")
    X_full_df, param_dict = simulation.generate_full_interventional_set(
        B_true, n, "mlp", knockdown_eff=1.0, size_observational=n * (d + 1)
    )

    set_random_seed_all(run_seed)

    wandb.init(
        project="sweep_pipeline",
    )

    maintain_dataset_size = True
    use_prescreen = True

    n_interventions = wandb.config.n_interventions
    X_df = subset_interventions(X_full_df, n_interventions, maintain_dataset_size=maintain_dataset_size)

    batch_size = 500
    data_loader = create_intervention_dataloader(X_df, batch_size=batch_size)

    ps_learning_rate = wandb.config.ps_learning_rate

    alpha = 1e-2
    beta = 0.005
    n_epochs = 2_000

    prescreen_config = {"n_epochs": n_epochs, "alphas": [alpha] * n_epochs, "gammas": [0] * n_epochs, "beta": beta}

    mask_threshold = wandb.config.mask_threshold

    learning_rate = wandb.config.learning_rate

    alpha = 5e-3
    beta = 0.005
    n_epochs = 5_000
    max_gamma = wandb.config.max_gamma
    gammas = list(np.linspace(0, max_gamma, n_epochs))
    freeze_gamma_at_dag = True

    config = {"n_epochs": n_epochs, "alphas": [alpha] * n_epochs, "gammas": gammas, "beta": beta}

    ps_model = AutoEncoderLayers(
        d, [10, 1], nn.Sigmoid(), shared_layers=False, adjacency_p=2.0,
        dag_penalty_flavor="none"
    )
    prescreen_optimizer = torch.optim.Adam(ps_model.parameters(), lr=ps_learning_rate)

    train(ps_model, data_loader, prescreen_optimizer, prescreen_config, log_wandb=True, print_graph=True, B_true=B_true, n_epochs_check=200)

    # Create mask for main algo
    mask = (ps_model.get_adjacency_matrix().detach().numpy() > mask_threshold).astype(int)
    print(f"Recall of mask: {(B_true.astype(bool) & mask.astype(bool)).sum() / B_true.sum()}")
    print(f"Fraction of edges in mask: {mask.sum() / (mask.shape[0] * mask.shape[1])}")

    # Begin DAG training

    dag_penalty_flavor = "scc"
    model = AutoEncoderLayers(
        d, [10, 1], nn.Sigmoid(), shared_layers=False, adjacency_p=2.0,
        dag_penalty_flavor=dag_penalty_flavor, mask=mask
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train(model, data_loader, optimizer, config, freeze_gamma_at_dag=freeze_gamma_at_dag, log_wandb=True, print_graph=True, B_true=B_true, n_epochs_check=100, start_wandb_epoch=prescreen_config["n_epochs"] if use_prescreen else 0)

sweep_configuration={
    "method": "bayes",
    "metric": {
        "name": "score",
        "goal": "minimize"
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 40,
    },
    "parameters": {
        "n_interventions": {
            "values": [0, 25, 50, 75, 100],
        },
        "ps_learning_rate": {
            "min": 1e-4,
            "max": 1e-1,
            "distribution": "log_uniform_values",
        },
        "learning_rate": {
            "min": 1e-4,
            "max": 1e-1,
            "distribution": "log_uniform_values",
        },
        "mask_threshold": {
            "min": 0.1,
            "max": 0.5,
        },
        "max_gamma": {
            "min": 50,
            "max": 200,
        }
    }
}

sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project="sweep_pipeline")
wandb.agent(sweep_id, function=main)


