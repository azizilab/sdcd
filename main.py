import matplotlib.pyplot as plt
import seaborn as sns
import torch.utils.data
from torch import nn

import simulated_data.deprecated_simulation as deprecated_simulation
from modules import AutoEncoderLayers
from utils import set_random_seed_all, print_graph_from_weights

set_random_seed_all(0)

n, d = 1000, 20
n_edges = 4 * d
B_true = deprecated_simulation.simulate_dag(d, n_edges, "ER")
X, param_dict = deprecated_simulation.simulate_nonlinear_sem(B_true, n, "mlp")
X = torch.FloatTensor(X)

model = AutoEncoderLayers(
    d, [10, 1], nn.Sigmoid(), shared_layers=False, adjacency_p=2.0
)
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

data_loader = torch.utils.data.DataLoader(X, batch_size=n, shuffle=True, drop_last=True)
alpha = 0.05
beta = 0.005

n_subepochs = 15_000
gammas = []
for a in [10]:
    gammas += [a] * n_subepochs

n_epochs = len(gammas)

thresholds = [0.5, 0.3, 0.1]
scores = []
n_observations = data_loader.batch_size * len(data_loader)
for epoch in range(n_epochs):
    gamma = gammas[epoch]
    epoch_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        loss = model.loss(batch, alpha, beta, gamma, n_observations)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch % 1000 == 0:
        B_pred = model.get_adjacency_matrix().detach().numpy()
        score = (B_true != (B_pred > 0.3)).sum()
        scores.append(score)
        epoch_loss /= len(data_loader)
        print(f"Epoch {epoch}: loss={epoch_loss:.2f}, score={score}, gamma={gamma:.2f}")
        print_graph_from_weights(d, B_pred, B_true, thresholds)

# plot both gammas and scores over epochs on the same plot with two axes
fig, ax1 = plt.subplots()
ax1.plot(gammas[::1000], color="tab:blue")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Gamma", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
# plot scores on the right y-axis
ax2 = ax1.twinx()
ax2.plot(scores, color="tab:red")
ax2.set_ylabel("Score", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
plt.show()

B_pred = model.get_adjacency_matrix().detach().numpy().round(3)
# plot the true and predicted adjacency matrices as heatmaps next to each other
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
B_pred = B_pred > 0.3
sns.heatmap(B_true, ax=ax[0])
sns.heatmap(B_pred, ax=ax[1])
plt.show()

print((B_pred == B_true).sum() / (d * d))
