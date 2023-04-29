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

n, d = 50, 20
n_edges = 5 * d
knockdown_eff = 1.0
B_true = simulation.simulate_dag(d, n_edges, "ER")
X_full_df, param_dict = simulation.generate_full_interventional_set(
    B_true, n, "mlp", knockdown_eff=1.0, size_observational=n * (d + 1)
)

batch_size = 500
data_loader = create_intervention_dataloader(X_df, batch_size=batch_size)

# No DAG prescreen
model = AutoEncoderLayers(
    d, [10, 1], nn.Sigmoid(), shared_layers=False, adjacency_p=2.0,
    dag_penalty_flavor="none"
)

learning_rate = 2e-2
prescreen_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

alpha = 5e-3
beta = 0.005
n_epochs = 3_000

prescreen_config = {"n_epochs": n_epochs, "alphas": [alpha] * n_epochs, "gammas": [0] * n_epochs, "beta": beta}
train(model, data_loader, prescreen_optimizer, prescreen_config, log_wandb=False, print_graph=True, B_true=B_true)

# Begin DAG training

dag_penalty_flavor = "scc"
model = AutoEncoderLayers(
    d, [10, 1], nn.Sigmoid(), shared_layers=False, adjacency_p=2.0,
    dag_penalty_flavor=dag_penalty_flavor
)
learning_rate = 2e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

alpha = 5e-3
beta = 0.005
gamma = 10
n_epochs = 25_000

config = {"n_epochs": n_epochs, "alphas": [alpha] * n_epochs, "gammas": [gamma] * n_epochs, "beta": beta}
train(model, data_loader, optimizer, config, log_wandb=False, print_graph=True, B_true=B_true)


# Log config
# wandb.init(
#     project="full-pipeline-simulation",
#     config={
#         "n": n,
#         "d": d,
#         "n_edges": n_edges,
#         "learning_rate": learning_rate,
#         "batch_size": batch_size,
#         "seed": seed,
#         "prescreen": prescreen,
#         "sig": sig,
#         "alpha": alpha_choices,
#         "alpha_schedule": alpha_update_flavor,
#         "beta": beta,
#         "knockdown_eff": knockdown_eff,
#         "gamma_update_flavor": gamma_update_flavor,
#         "gamma_update_config": gamma_update_config,
#         "use_interventions": use_interventions,
#     },
# )



