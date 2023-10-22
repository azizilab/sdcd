import abc
import time

import numpy as np
import pandas as pd
import scipy

import torch
import tqdm
from matplotlib import pyplot as plt

"""
penalty training: f(A) = ℓ(A) + h(A)
  > we need the gradient ∇h(A)
augmented lagrangian training: f(A) = ℓ(A) + γ h(A) + μ/2 h(A)²
  > we need the gradient ∇h(A) and the gradient of h(A)² which is 2h(A) ∇h(A)
the greek alphabet is: α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ, ν, ξ, ο, π, ρ, σ, τ, υ, φ, χ, ψ, ω

augmented lagrangian procedure:
  0. set η = 2, δ = 0.9, ε = 10⁻⁸
  1. initialize γ₀ = 0, μ₀ = 10⁻⁸
  2. repeat
    a. solve Aₜ = argmin_A ℓ(A) + γₜ h(A) + μₜ/2 h(A)²
    b. update γₜ₊₁ = γₜ + μₜ h(Aₜ)
    c. if h(Aₜ₊₁) > δ h(Aₜ) [has not decreased enough]
        then μₜ₊₁ = μₜ * η
        else μₜ₊₁ = μₜ
    d. t = t + 1
    e. if h(Aₜ) < ε # and (Aₜ > ξ) is acyclic
"""


class DAGConstraint:
    def __init__(self, d):
        self.d = d

    @abc.abstractmethod
    def __call__(self, matrix):
        pass

    @abc.abstractmethod
    def gradient(self, matrix):
        pass

    def gradient_pow2(self, matrix):
        return 2 * self(matrix) * self.gradient(matrix)


class L2Loss(DAGConstraint):
    def __init__(self, d, target_matrix):
        super().__init__(d)
        self.target_matrix = target_matrix

    def __call__(self, matrix):
        return np.power(matrix - self.target_matrix, 2).sum()

    def gradient(self, matrix):
        return 2 * (matrix - self.target_matrix)


class HExp(DAGConstraint):
    def __init__(self, d):
        super().__init__(d)
        self.identity = np.eye(d, dtype=np.float64)
        self.__name__ = "h_exp"

    def __call__(self, matrix):
        return np.trace(scipy.linalg.expm(matrix)) - self.d

    def gradient(self, matrix):
        return scipy.linalg.expm(matrix.T)


class HLog(DAGConstraint):
    def __init__(self, d):
        super().__init__(d)
        self.identity = np.eye(d, dtype=np.float64)
        self.__name__ = "h_log"

    def __call__(self, matrix):
        return -np.linalg.slogdet(self.identity - matrix)[1]

    def gradient(self, matrix):
        return scipy.linalg.inv(self.identity - matrix).T


class HInv(DAGConstraint):
    def __init__(self, d):
        super().__init__(d)
        self.identity = np.eye(d, dtype=np.float64)
        self.__name__ = "h_inv"

    def __call__(self, matrix):
        return np.trace(np.linalg.inv(self.identity - matrix)) - self.d

    def gradient(self, matrix):
        return np.linalg.matrix_power(self.identity - matrix.T, -2)


class HRho:
    def __init__(self, d, n_iter=5, name_suffix=""):
        self.d = d
        self.n_iter = n_iter
        self.u = np.ones(d, dtype=np.float64) / np.sqrt(d)  # eigenvector of A
        self.v = np.ones(d, dtype=np.float64) / np.sqrt(d)  # eigenvector of A.T
        self.__name__ = "h_rho"
        if name_suffix:
            self.__name__ += "_" + name_suffix

    def __call__(self, matrix):
        return np.dot(self.v, matrix @ self.u) / np.dot(self.v, self.u)

    def _update_eigenvectors(self, matrix):
        for _ in range(self.n_iter):
            self.u = matrix @ self.u
            self.u /= np.linalg.norm(self.u)
            self.v = matrix.T @ self.v
            self.v /= np.linalg.norm(self.v)

    def gradient(self, matrix, update=True):
        if update:
            self._update_eigenvectors(matrix + 1e-6)

        return np.outer(self.v, self.u) / np.dot(self.v, self.u)

    # def gradient_pow2(self, matrix):
    #     return 2 * self(matrix) * self.gradient(matrix, update=False)


class HRhoLog(HRho):
    def __init__(self, d, n_iter=5):
        super().__init__(d, n_iter)
        self.log = HLog(d)
        self.__name__ = "h_power_log"

        self._coeff = 0.5

    def __call__(self, matrix):
        rho = super().__call__(matrix)
        return self.log(matrix / rho * self._coeff)

    def gradient(self, matrix, update=True):
        self._update_eigenvectors(matrix + 1e-6)
        rho = super().__call__(matrix)
        return self.log.gradient(matrix / rho * self._coeff) / rho * self._coeff


def _is_acyclic(matrix: np.ndarray):
    d = matrix.shape[0]
    v = np.ones(d, dtype=matrix.dtype)
    nnz_old = np.count_nonzero(v)
    for i in range(d):
        v = matrix @ v
        nnz_new = np.count_nonzero(v)
        if nnz_new >= nnz_old:
            return False
        if nnz_new == 0:
            return True
        nnz_old = nnz_new
        v = v / (v + 1e-8)
    return True


def _threshold_to_dag(matrix: np.ndarray) -> float:
    # noinspection PyTypeChecker
    return scipy.optimize.bisect(
        lambda t: _is_acyclic(matrix >= t) - 0.5 if t > 0 else -0.5,
        -1,
        10,
        xtol=1e-10,
    )


def generate_random_sparse_matrix(d, sparsity=0.1, seed=0):
    np.random.seed(seed)
    matrix = np.random.rand(d, d).astype(np.float64) / d
    matrix = matrix * (np.random.rand(d, d) < sparsity)
    return matrix


def generate_random_sparse_torch_matrix(d, sparsity=0.1):
    torch.manual_seed(2)
    matrix = torch.rand((d, d), dtype=torch.float64) / d
    matrix = matrix * (torch.rand((d, d)) < sparsity)
    return matrix.numpy().astype(np.float64)


def training_penalty(
    matrix: np.array,
    loss,
    h,
    gamma,
    lr=1e-2,
    n_epochs=-1,
    n_epochs_val=100,
    tol=1e-5,
    max_time=2 * 60,
    zero_diag=False,
):
    # train with penalty: f(A) = ℓ(A) + γ h(A)
    # train until convergence if n_epochs == -1
    if n_epochs == -1:
        n_epochs = 1_000_000
    epoch = 0
    thresholds = [_threshold_to_dag(matrix)]
    hs = [h(matrix)]
    losses = [loss(matrix)]
    start = time.time()
    pbar = tqdm.tqdm(total=n_epochs)
    while epoch < n_epochs:
        # f = loss(matrix) + gamma * h(matrix)
        # gradient update
        grad = loss.gradient(matrix) + gamma * h.gradient(matrix)
        matrix = np.maximum(matrix - lr * grad, 0)
        if zero_diag:
            np.fill_diagonal(matrix, 0)
        # ###############

        loss.gradient(matrix)

        epoch += 1
        pbar.update(1)

        if epoch % n_epochs_val == 0:
            hs.append(h(matrix))
            threshold = _threshold_to_dag(matrix)
            thresholds.append(threshold)
            losses.append(loss(matrix))
            if threshold < tol:
                break
        if time.time() - start > max_time:
            break

    pbar.close()
    metrics = {
        "epoch": np.arange(len(hs)) * n_epochs_val,
        "threshold_to_dag": thresholds,
        "h": hs,
        "method": h.__name__,
    }
    return matrix, metrics


def training_augmented_lagrangian(
    matrix: np.array,
    loss,
    h,
    gamma=0,
    lr=1e-2,
    n_epochs=-1,
    n_epochs_val=100,
    tol=1e-5,
    max_time=2 * 60,
    zero_diag=False,
):
    # train with augmented lagrangian: f(A) = ℓ(A) + γ h(A) + μ/2 h(A)²
    # train until convergence if n_epochs == -1
    if n_epochs == -1:
        n_epochs = 1_000_000

    eta = 2
    delta = 0.9
    convergence_threshold = 1e-5
    gamma = gamma
    mu = 1

    epoch = 0
    thresholds = [_threshold_to_dag(matrix)]
    hs = [h(matrix)]
    mus = [mu]
    losses = [loss(matrix)]
    full_losses = [loss(matrix) + gamma * h(matrix) + mu / 2 * h(matrix) ** 2]
    gammas = [gamma]
    nnzs = [np.count_nonzero(matrix)]
    start = time.time()
    pbar = tqdm.tqdm(total=n_epochs)
    while epoch < n_epochs:
        # f = loss(matrix) + gamma * h(matrix) + mu / 2 * h(matrix) ** 2
        # gradient update
        h_grad = h.gradient(matrix)
        grad = loss.gradient(matrix) + gamma * h_grad + mu * h_grad * h(matrix)
        matrix = np.maximum(matrix - lr * grad, 0)
        full_losses.append(loss(matrix) + gamma * h(matrix) + mu / 2 * h(matrix) ** 2)
        if zero_diag:
            np.fill_diagonal(matrix, 0)
        # ###############

        epoch += 1
        pbar.update(1)

        if epoch % n_epochs_val == 0 and epoch > 0:
            hs.append(h(matrix))
            threshold = _threshold_to_dag(matrix)
            thresholds.append(threshold)
            losses.append(loss(matrix))
            nnzs.append(np.count_nonzero(matrix))
            full_losses_since_last = full_losses[-n_epochs_val:]
            if (
                len(hs) > 1
                and np.max(full_losses_since_last) - np.min(full_losses_since_last) < convergence_threshold
                # and np.abs(hs[-1] - hs[-2]) < convergence_threshold
            ):
                # we have converged, we update gamma and mu
                gamma += mu * hs[-1]
                if hs[-1] > delta * hs[-2]:
                    mu *= eta
                print(f"{epoch} - mu: {mu}, gamma: {gamma}, h: {hs[-1]}", {hs[-1] * mu})

            mus.append(mu)
            gammas.append(gamma)

            if threshold < tol:
                break

        if time.time() - start > max_time:
            break

    metrics = {
        "epoch": np.arange(len(hs)) * n_epochs_val,
        "threshold_to_dag": thresholds,
        "h": hs,
        "mu": mus,
        "gamma": gammas,
        "loss": losses,
        "nnz": nnzs,
        "method": h.__name__,
    }
    return matrix, metrics


def _paper_experiment_run(anchor, n_epochs, ax, mode):
    # experiment
    d = anchor.shape[0]
    loss = L2Loss(d, anchor)
    metrics = []
    n_epochs_val = 100
    if mode == "penalty":
        _train = training_penalty
        gamma = 10
    elif mode == "augmented_lagrangian":
        _train = training_augmented_lagrangian
        gamma = 0
    else:
        raise ValueError(f"Unknown mode {mode}")
    for h in [
        HExp(d),
        HLog(d),
        HInv(d),
        HRho(d),
        HRho(d, name_suffix="zero"),
    ]:
        matrix = anchor.copy()

        matrix, h_metrics = _train(
            matrix,
            loss,
            h,
            gamma=gamma,
            lr=0.001,
            n_epochs=n_epochs,
            n_epochs_val=n_epochs_val,
            tol=1e-4,
            max_time=6 * 60,
            zero_diag="zero" in h.__name__,
        )
        metrics.append(pd.DataFrame(h_metrics))
    metrics = pd.concat(metrics)

    method_labels = {
        "h_exp": r"$h_{\exp}$",
        "h_log": r"$h_{\log}$",
        "h_inv": r"$h_{\mathrm{inv}}$",
        "h_rho": r"$h_{\rho}$",
        "h_rho_zero": r"$h_{\rho}$+",
    }
    colors = ["C0", "C1", "C2", "C4", "C4"]
    for i, (method, method_label) in enumerate(method_labels.items()):
        tmp_data = metrics[metrics["method"] == method]
        ax.plot(
            tmp_data["epoch"],
            tmp_data["threshold_to_dag"],
            label=method_label,
            alpha=0.8,
            linewidth=2,
            color=colors[i],
            ls="-" if i < 4 else (0, (1, 0.5)),
        )
    ax.set_xlim(metrics["epoch"].min(), metrics["epoch"].max())
    ax.set_ylim(0, metrics["threshold_to_dag"].max())

    return metrics


def _plot_last_mus(metrics, ax):
    method_labels = {
        "h_exp": r"$h_{\exp}$",
        "h_log": r"$h_{\log}$",
        "h_inv": r"$h_{\mathrm{inv}}$",
        "h_rho": r"$h_{\rho}$",
        "h_rho_zero": r"$h_{\rho}$+",
    }
    colors = ["C0", "C1", "C2", "C4", "C4"]
    # bar plot showing the mu at the last epoch for each method
    # bars are horizontal, name of the method on the y axis, written vertically
    # x axis is the mu value, with a log scale
    # color is the same as the line plot
    # if zero is in the name, the bar is hatched
    for i, (method, method_label) in enumerate(method_labels.items()):
        tmp_data = metrics[metrics["method"] == method]
        ax.barh(
            i,
            tmp_data["mu"].iloc[-1],
            color=colors[i],
            alpha=0.8,
            height=0.8,
            hatch="///" if "zero" in method else None,
            edgecolor="white",
        )

        # ax.text(
        #     tmp_data["mu"].iloc[-1] + 0.01,
        #     i,
        #     f"{tmp_data['mu'].iloc[-1]:.2f}",
        #     va="center",
        #     fontsize=11,
        # )

    ax.set_yticks([])
    # ax.set_yticks(np.arange(len(method_labels)))
    # ax.set_yticklabels(method_labels.values(), fontsize=11)
    ax.set_xlabel(r"last $\mu$")
    ax.set_xscale("log")
    # ax.set_xlim(0.01, 100)
    ax.set_ylim(-0.5, len(method_labels) - 0.5)


def paper_experiment():
    d = 100
    m = 30
    anchor = generate_random_sparse_matrix(d, m / d) * 5

    n_cols = 2
    n_rows = 1
    width_ratios = [0.8, 1]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * 2, 2.5), width_ratios=width_ratios)
    _DEBUG = 1
    _paper_experiment_run(anchor, 8_000 // _DEBUG, axes[0], mode="penalty")
    metrics = _paper_experiment_run(anchor, 200_000 // _DEBUG, axes[1], mode="augmented_lagrangian")
    # _plot_last_mus(metrics, axes[2])

    ax = axes[0]
    ax.legend(
        fontsize=11,
        ncol=2,
        columnspacing=0.5,
        handletextpad=0.2,
        fancybox=False,
        framealpha=1,
        borderpad=0.2,
        labelspacing=0.2,
        handlelength=1,
    )

    ax.set_title("With Penalty")
    xticks = [0, 2_000, 4_000, 6_000]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x / 1000)}k" for x in xticks], fontsize=11)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Threshold to DAG", fontsize=14)

    ax = axes[1]
    ax.set_title("With Augmented Lagrangian")
    ax.set_ylabel("")
    ax.set_yticklabels([])
    xticks = [0, 50_000, 100_000, 150_000]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x / 1000)}k" for x in xticks], fontsize=11)
    ax.set_xlabel("Epoch", fontsize=14)

    for ax in axes.ravel():
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    plt.tight_layout(pad=0.5)
    plt.savefig("figures/training.stability.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    paper_experiment()
