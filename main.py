import networkx as nx
import torch
from torch import nn
import torch.utils.data

import external.dagma.utils
from utils import set_random_seed_all, get_leading_left_and_right_eigenvectors
from autoencoder import AutoEncoder

import matplotlib.pyplot as plt
import seaborn as sns
# torch.set_default_dtype(torch.float32)

set_random_seed_all(0)

n, d, _, graph_type, sem_type = 1000, 10, -1, "ER", "mlp"
s0 = 4*d
B_true = external.dagma.utils.simulate_dag(d, s0, graph_type)
X = external.dagma.utils.simulate_nonlinear_sem(B_true, n, sem_type)
X = torch.FloatTensor(X)

model = AutoEncoder(d, 32, [], [], nn.ReLU(), adjacency_p=1)
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

data_loader = torch.utils.data.DataLoader(X, batch_size=128, shuffle=True, drop_last=True)
alpha = 1.0
beta = 1.0
initial_gamma = 1.0
target_gamma = 200_000

# initial_gamma = 10.0
target_gamma = 50
gamma = initial_gamma

n_subepochs = 3000
gammas = []
# for a in [1, 10, 20, 50, 10, 20, 50]:
for a in [1, 5, 20, 10, 20, 10, 20]:
    gammas += [a] * n_subepochs

n_epochs = len(gammas)

thresholds = [0.5, 0.3, 0.1]
scores = []
n_observations = data_loader.batch_size * len(data_loader)
for epoch in range(n_epochs):
    gamma = gammas[epoch]
    # gamma *= (target_gamma / initial_gamma) ** (1 / n_epochs)
    # gamma += (target_gamma - initial_gamma) / n_epochs

    epoch_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        # L1 loss only on the entries of the adjacency matrix that are zero in the true adjacency matrix B_true
        # reconstruction_loss = model.reconstruction_loss(batch)
        # l1_loss = (model.get_adjacency_matrix("normal", p=1) * torch.FloatTensor(1 - B_true)).abs().sum()
        # loss = l1_loss * gamma + reconstruction_loss
        # get left and right eigenvectors of the adjacency matrix

        # A = model.get_adjacency_matrix("normal", p=1)
        # l, r = get_leading_left_and_right_eigenvectors(A)
        # A_loss = (A * torch.FloatTensor(l[:, None] * r[None, :]) / l.dot(r)).sum()
        # trace_loss = torch.trace(A)
        # A_loss += trace_loss
        # loss = reconstruction_loss + A_loss * gamma + model.l1_reg() * 0.1

        loss = model.loss(batch, alpha, beta, gamma, n_observations)
        # loss += gamma * trace_loss

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(data_loader)
    B_pred = model.get_adjacency_matrix("normal").detach().numpy()
    score = (B_true != (B_pred > 0.3)).sum()
    scores.append(score)
    if epoch % 100 == 0:
        # print(
        #     f"Epoch {epoch}: "
        #     f"rec-loss={model.reconstruction_loss(data_loader.dataset):.2f}, "
        #     f"dag-loss={model.dag_loss()*gamma:.2f}, "
        #     f"largest-eig={torch.linalg.eigvals(model.get_adjacency_matrix('reverse', p=2)).abs().max():.2f}, "
        # )

        # for each node: list its parents in order of the predicted edge weights
        # if the parent is correct, print it in green, otherwise in red
        print("Epoch", epoch, "---", "gamma =", gamma)
        for i in range(d):
            parents_weights = B_pred[:, i]
            parents = sorted(range(d), key=lambda j: parents_weights[j], reverse=True)
            parents_str = []
            for t in thresholds:
                if parents_weights[parents[0]] < t:
                    parents_str.append("|")
            for idx, j in enumerate(parents):
                if B_true[j, i]:
                    parents_str.append(f"\033[92m{j}\033[0m")
                else:
                    parents_str.append(f"\033[91m{j}\033[0m")
                # add | if the parent weight is greater than one of the thresholds
                # and the next parent weight is less than the threshold
                for t in thresholds:
                    conditions = [(idx < d - 1 and parents_weights[parents[idx]] > t > parents_weights[parents[idx + 1]]),
                                  (idx == d - 1 and parents_weights[parents[idx]] > t)
                                ]
                    if any(conditions):
                        parents_str.append("|")
            print(f"Node {i}: " + " ".join(parents_str))
        # show a few cycles
        G = nx.DiGraph(B_pred > 0.1)
        # check if the graph is a DAG
        print("DAG:", nx.is_directed_acyclic_graph(G))
        for t in thresholds:
            print(f"SDH >{t}:", (B_true != (B_pred > t)).sum(), end="; ")
        print()
        print()

    # ask users to change gamma every 2000 epochs
    # if epoch % 2000 == 1999:
    #     valid_input = False
    #     while not valid_input:
    #         try:
    #             gamma = float(input("Enter gamma: "))
    #             valid_input = True
    #         except ValueError:
    #             print("Invalid input. Please try again.")

# plot both gammas and scores over epochs on the same plot with two axes
fig, ax1 = plt.subplots()
ax1.plot(gammas, color="tab:blue")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Gamma", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
# plot scores on the right y-axis
ax2 = ax1.twinx()
ax2.plot(scores, color="tab:red")
ax2.set_ylabel("Score", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
plt.show()


B_pred = model.get_adjacency_matrix("normal").detach().numpy().round(3)
print(sorted(B_pred.flatten())[::-1][:50])
# plot the true and predicted adjacency matrices as heatmaps next to each other
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
B_pred = B_pred > 0.3
sns.heatmap(B_true, ax=ax[0])
sns.heatmap(B_pred, ax=ax[1])

print((B_pred == B_true).sum() / (d * d))
plt.show()
