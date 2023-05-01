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

seed = 0
set_random_seed_all(seed)

n, d = 50, 200
n_edges = 5 * d
knockdown_eff = 1.0
B_true = simulation.simulate_dag(d, n_edges, "ER")
X_full_df, param_dict = simulation.generate_full_interventional_set(
    B_true, n, "mlp", knockdown_eff=1.0, size_observational=n * (d + 1)
)

maintain_dataset_size = True
# use_prescreen = True

for use_prescreen in [False, True]:
    for n_interventions in reversed(range(0, d+1, 20)):
        X_df = subset_interventions(X_full_df, n_interventions, maintain_dataset_size=maintain_dataset_size)

        batch_size = 500
        data_loader = create_intervention_dataloader(X_df, batch_size=batch_size)

        ps_learning_rate = 2e-3

        alpha = 1e-2
        beta = 0.005
        n_epochs = 2_000

        prescreen_config = {"n_epochs": n_epochs, "alphas": [alpha] * n_epochs, "gammas": [0] * n_epochs, "beta": beta}

        mask_threshold = 0.3

        learning_rate = 2e-2

        alpha = 5e-3
        beta = 0.005
        n_epochs = 5_000
        # gamma = 10
        gammas = list(np.linspace(0, 100, n_epochs))
        freeze_gamma_at_dag = True

        config = {"n_epochs": n_epochs, "alphas": [alpha] * n_epochs, "gammas": gammas, "beta": beta}

        prescreen_str = "_prescreen" if use_prescreen else ""

        # Log config
        wandb.init(
            project="full-pipeline-simulation",
            name=f"{n_interventions / d:.2f}_interventions_{prescreen_str}gamma_search",
            config={
                "seed": seed,
                "n": n,
                "d": d,
                "n_edges": n_edges,
                "batch_size": batch_size,
                "knockdown_eff": knockdown_eff,
                "frac_interventions": n_interventions / d,
                "maintain_dataset_size": maintain_dataset_size,
                "ps_learning_rate": ps_learning_rate,
                "use_prescreen": use_prescreen,
                "prescreen_config": prescreen_config,
                "learning_rate": learning_rate,
                "config": config
            },
        )

        # No DAG prescreen
        if use_prescreen:
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
        else:
            mask = np.ones((d, d))

        # Begin DAG training

        dag_penalty_flavor = "scc"
        model = AutoEncoderLayers(
            d, [10, 1], nn.Sigmoid(), shared_layers=False, adjacency_p=2.0,
            dag_penalty_flavor=dag_penalty_flavor, mask=mask
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train(model, data_loader, optimizer, config, freeze_gamma_at_dag=freeze_gamma_at_dag, log_wandb=True, print_graph=True, B_true=B_true, n_epochs_check=100, start_wandb_epoch=prescreen_config["n_epochs"] if use_prescreen else 0)

        wandb.finish()



