import math

import matplotlib.pyplot as plt
import seaborn as sns
import torch.utils.data
from torch import nn
import wandb

import simulation
from modules import AutoEncoderLayers
from utils import set_random_seed_all, print_graph_from_weights, ks_test_screen

seed = 0
set_random_seed_all(seed)

n, d = 50, 10
n_edges = 4 * d
knockdown_eff = 1.0
B_true = simulation.simulate_dag(d, n_edges, "ER")
X_df, param_dict = simulation.generate_full_interventional_set(
    B_true, n, "mlp", knockdown_eff=1.0
)
X = torch.FloatTensor(X_df.to_numpy()[:, :-1].astype(float))

prescreen = True
sig = 0.1
mask = None
# Prescreen
if prescreen:
    mask = ks_test_screen(X_df, verbose=True)

# Begin training
model = AutoEncoderLayers(
    d, [10, 1], nn.Sigmoid(), shared_layers=False, adjacency_p=2.0, mask=mask
)
learning_rate = 2e-4  # 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

data_loader = torch.utils.data.DataLoader(X, batch_size=n, shuffle=True, drop_last=True)

n_subepochs = 15_000
gammas = []
for a in [10, 100, 1000]:
    gammas += [a] * n_subepochs

n_epochs = len(gammas)
alpha = 0.02
beta = 0.005

# Hyperparam tune config
gamma_update_flavor = "fixed"
gamma_update_interval_epochs = 10
gamma_update_config = None
if gamma_update_flavor == "annealing":
    gamma_update_patience = 3
    gamma_update_delta = 3
    gamma_update_step = 10
    gamma_update_floor = 10
    gamma_update_ceil = 100
    gamma_update_max_cycle_epochs = 2000
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
    _gamma_update_floor = 10
    _gamma_update_ceil = 100
    _gamma_update_cycle_counter = 0
    _gamma_update_patience_counter = gamma_update_patience

# Log config
wandb.init(
    project="causal-perturbseq",
    config={
        "n": n,
        "d": d,
        "seed": seed,
        "prescreen": prescreen,
        "sig": sig,
        "alpha": alpha,
        "beta": beta,
        "gammas": gammas,
        "knockdown_eff": knockdown_eff,
    },
)

thresholds = [0.5, 0.3, 0.1]
scores = []
n_observations = data_loader.batch_size * len(data_loader)

if gamma_update_flavor == "annealing":
    gamma = gamma_update_floor

for epoch in range(n_epochs):
    if gamma_update_flavor == "fixed":
        gamma = gammas[epoch]
    elif gamma_update_flavor == "annealing":
        cycle_frac = _gamma_update_cycle_counter / gamma_update_max_cycle_epochs
        cos_frac = (
            math.cos((cycle_frac + 1) * math.pi) + 1
        ) / 2  # cos(x) from pi to 2pi remapped to [0,1] x [0,1]
        gamma = (
            cos_frac * (_gamma_update_ceil - _gamma_update_floor) + _gamma_update_floor
        )

    epoch_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        loss = model.loss(batch, alpha, beta, gamma, n_observations)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch % gamma_update_interval_epochs == 0 & epoch > 0:
        if gamma_update_flavor == "annealing":
            _gamma_update_cycle_counter += gamma_update_interval_epochs
            B_pred = model.get_adjacency_matrix().detach().numpy() > 0.3
            if _gamma_update_prev_adj is not None:
                n_edges_change = (B_pred != _gamma_update_prev_adj).sum()
                wandb.log({"n_edges_change": n_edges_change, "epoch": epoch})
                if n_edges_change <= gamma_update_delta:
                    _gamma_update_patience_counter -= 1
                else:
                    _gamma_update_patience_counter = gamma_update_patience

            _gamma_update_prev_adj = B_pred

            if _gamma_update_patience_counter == 0:
                gamma = _gamma_update_floor
                _gamma_update_patience_counter = gamma_update_patience
                _gamma_update_floor = gamma_update_floor + gamma_update_step
                _gamma_update_cycle_counter = 0
            elif _gamma_update_cycle_counter >= gamma_update_max_cycle_epochs:
                gamma = _gamma_update_floor
                _gamma_update_ceil += gamma_update_step
                _gamma_update_cycle_counter = 0
                _gamma_update_patience_counter = gamma_update_patience

    if epoch % 1000 == 0:
        B_pred = model.get_adjacency_matrix().detach().numpy()
        score = (B_true != (B_pred > 0.3)).sum()
        scores.append(score)
        wandb.log(
            {
                "epoch": epoch,
                "epoch_loss": epoch_loss,
                "score": score,
                "gamma": gamma,
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
