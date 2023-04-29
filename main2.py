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

seed = 0
set_random_seed_all(seed)

n, d = 50, 20
# n, d = 50, 200
n_edges = 5 * d
# n_edges = 10 * d
knockdown_eff = 1.0
B_true = simulation.simulate_dag(d, n_edges, "ER")
X_df, param_dict = simulation.generate_full_interventional_set(
    B_true, n, "mlp", knockdown_eff=1.0
)
X = torch.FloatTensor(X_df.to_numpy()[:, :-1].astype(float))

prescreen = False
n_parents = None
warmstart = True
sig = 0.2
mask = None
# Prescreen
if prescreen or warmstart:
    mask = ks_test_screen(X_df, use_sig=n_parents is None, sig=sig, n_parents=n_parents, verbose=True)
use_interventions = True

dag_penalty_flavor = "scc"
model = AutoEncoderLayers(
    d, [10, 1], nn.Sigmoid(), shared_layers=False, adjacency_p=2.0, mask=mask, warmstart=warmstart,
    dag_penalty_flavor=dag_penalty_flavor
)
# learning_rate = 2e-4
learning_rate = 1e-3
batch_size = 500
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if use_interventions:
    # ensure interventions are ints mapping to index of column
    column_mapping = {c: i for i, c in enumerate(X_df.columns[:-1])}
    column_mapping["obs"] = -1
    interventions = torch.LongTensor(X_df["perturbation_label"].map(column_mapping)).reshape((-1, 1))
    dataset = torch.utils.data.TensorDataset(X, interventions)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
else:
    data_loader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=True, drop_last=True)

n_epochs = 25_000

alpha_update_flavor = "fixed"
alpha_choices = [5e-2]
if alpha_update_flavor == "fixed":
    alphas = []
    for a in alpha_choices:
        alphas += [a] * int(n_epochs // len(alpha_choices))
    if len(alphas) != n_epochs:
        alphas += [alpha_choices[-1]] * int(n_epochs % alpha_choices)
elif alpha_update_flavor == "linear":
    alphas = np.linspace(0, alpha_choices[-1], num=n_epochs).tolist()

beta = 0.005

# Hyperparam tune config
gamma_update_flavor = "fixed"
gammas = []
# gamma_choices = [10, 100, 500, 500, 500, 500]
gamma_choices = [10]
for a in gamma_choices:
    gammas += [a] * int(n_epochs // len(gamma_choices))
if len(gammas) != n_epochs:
    gammas += [gamma_choices[-1]] * int(n_epochs % gamma_choices[-1])
gamma_update_interval_epochs = 10
gamma_update_config = None
if gamma_update_flavor == "fixed":
    gamma_update_config = {"gammas": gamma_choices}
elif gamma_update_flavor == "annealing":
    gamma_update_patience = 1000000
    gamma_update_delta = 0
    gamma_update_step = 20
    gamma_update_cycle_step = 100
    gamma_update_floor = 10
    gamma_update_ceil = 200
    gamma_update_max_cycle_epochs = 25000
    gamma_update_config = {
        "interval": gamma_update_interval_epochs,
        "patience": gamma_update_patience,
        "delta": gamma_update_delta,
        "step": gamma_update_step,
        "floor": gamma_update_floor,
        "ceil": gamma_update_ceil,
        "max_cycle": gamma_update_max_cycle_epochs,
    }

    _gamma_update_prev_adj = None
    _gamma_update_floor = gamma_update_floor
    _gamma_update_ceil = gamma_update_ceil
    _gamma_update_cycle_counter = 0
    _gamma_update_patience_counter = gamma_update_patience

# Log config
wandb.init(
    project="main2-simulation",
    config={
        "n": n,
        "d": d,
        "n_edges": n_edges,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "seed": seed,
        "prescreen": prescreen,
        "sig": sig,
        "alpha": alpha_choices,
        "alpha_schedule": alpha_update_flavor,
        "beta": beta,
        "knockdown_eff": knockdown_eff,
        "gamma_update_flavor": gamma_update_flavor,
        "gamma_update_config": gamma_update_config,
        "dag_penalty_flavor": dag_penalty_flavor,
        "use_interventions": use_interventions,
    },
)

thresholds = [0.5, 0.3, 0.1]
scores = []
n_observations = data_loader.batch_size * len(data_loader)

if gamma_update_flavor == "annealing":
    gamma = gamma_update_floor

for epoch in range(n_epochs):
    alpha = alphas[epoch]

    if gamma_update_flavor == "fixed":
        gamma = gammas[epoch]
    elif gamma_update_flavor == "annealing":
        _gamma_update_cycle_counter += 1
        cycle_frac = min(
            _gamma_update_cycle_counter / gamma_update_max_cycle_epochs
            if (n_epochs - epoch) >= gamma_update_max_cycle_epochs
            else _gamma_update_cycle_counter / gamma_update_max_cycle_epochs,
            1,
        )
        cos_frac = (
            math.cos((cycle_frac + 1) * math.pi) + 1
        ) / 2  # cos(x) from pi to 2pi remapped to [0,1] x [0,1]
        gamma = (
            cos_frac * (_gamma_update_ceil - _gamma_update_floor) + _gamma_update_floor
        )

    epoch_loss = 0
    for batch in data_loader:
        if use_interventions:
            X_batch, interventions_batch = batch
        else:
            X_batch = batch
            interventions_batch = None
        optimizer.zero_grad()
        loss = model.loss(X_batch, alpha, beta, gamma, n_observations, interventions = interventions_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch % gamma_update_interval_epochs == 0 and epoch > 0:
        if gamma_update_flavor == "annealing":
            B_pred = model.get_adjacency_matrix().detach().numpy()
            B_pred_thresh = B_pred > 0.3
            if _gamma_update_prev_adj is not None:
                # Check if the adjacency matrix has changed
                n_edges_change = (B_pred_thresh != _gamma_update_prev_adj).sum()
                is_dag = nx.is_directed_acyclic_graph(nx.DiGraph(B_pred > 0.1))
                wandb.log({"n_edges_change": n_edges_change, "is_dag": is_dag, "epoch": epoch})
                if is_dag and n_edges_change <= gamma_update_delta:
                    # Decrease patience if not changing much
                    _gamma_update_patience_counter -= 1
                else:
                    # Reset patience otherwise
                    _gamma_update_patience_counter = gamma_update_patience

            _gamma_update_prev_adj = B_pred_thresh

            if (n_epochs - epoch) < gamma_update_max_cycle_epochs:
                # If within last cycle, just finish up last cycle without logic
                pass
            elif _gamma_update_patience_counter == 0:
                print(f"Lost patience, resetting gamma to {_gamma_update_floor}")
                # If lost patience, reset gamma and reset everything
                _gamma_update_floor = max(
                    _gamma_update_floor - gamma_update_cycle_step, 0
                )
                _gamma_update_ceil = max(
                    _gamma_update_ceil - gamma_update_cycle_step,
                    _gamma_update_floor + gamma_update_step,
                )

                gamma = _gamma_update_floor

                _gamma_update_patience_counter = gamma_update_patience
                _gamma_update_cycle_counter = 0
            elif _gamma_update_cycle_counter >= gamma_update_max_cycle_epochs:
                print(f"Hit max cycle, resetting gamma to {_gamma_update_floor}")
                _gamma_update_ceil += gamma_update_step
                _gamma_update_floor += gamma_update_cycle_step

                gamma = _gamma_update_floor

                _gamma_update_cycle_counter = 0
                _gamma_update_patience_counter = gamma_update_patience

    if epoch % 1000 == 0:
        B_pred = model.get_adjacency_matrix().detach().numpy()
        score = (B_true != (B_pred > 0.3)).sum()
        recall = (B_true.astype(bool) & (B_pred > 0.3)).sum() / B_true.sum()
        precision = (B_true.astype(bool) & (B_pred > 0.3)).sum() / (B_pred > 0.3).sum()
        n_edges_pred = (B_pred > 0.3).sum()
        scores.append(score)
        wandb.log(
            {
                "epoch": epoch,
                "epoch_loss": epoch_loss,
                "score": score,
                "precision": precision,
                "recall": recall,
                "n_edges_pred": n_edges_pred,
                "gamma": gamma,
                "alpha": alpha,
            }
        )
        epoch_loss /= len(data_loader)
        print(f"Epoch {epoch}: loss={epoch_loss:.2f}, score={score}, gamma={gamma:.2f}")
        print_graph_from_weights(d, B_pred, B_true, thresholds)

# # plot both gammas and scores over epochs on the same plot with two axes
# fig, ax1 = plt.subplots()
# ax1.plot(gammas[::1000], color="tab:blue")
# ax1.set_xlabel("Epoch")
# ax1.set_ylabel("Gamma", color="tab:blue")
# ax1.tick_params(axis="y", labelcolor="tab:blue")
# # plot scores on the right y-axis
# ax2 = ax1.twinx()
# ax2.plot(scores, color="tab:red")
# ax2.set_ylabel("Score", color="tab:red")
# ax2.tick_params(axis="y", labelcolor="tab:red")
# plt.show()

# B_pred = model.get_adjacency_matrix().detach().numpy().round(3)
# # plot the true and predicted adjacency matrices as heatmaps next to each other
# fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# B_pred = B_pred > 0.3
# sns.heatmap(B_true, ax=ax[0])
# sns.heatmap(B_pred, ax=ax[1])
# plt.show()

# print((B_pred == B_true).sum() / (d * d))
