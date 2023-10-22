from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch


def generate_random_torch_dag_matrix(d, sparsity=0.1):
    if sparsity <= 0:
        return torch.zeros((d, d), dtype=torch.float32)
    matrix = torch.rand((d, d), dtype=torch.float32)  # / d
    matrix = matrix * (torch.rand((d, d)) < sparsity)  # / sparsity
    # set lower triangle to zero
    matrix = matrix.triu(1)
    # shuffle rows and columns
    perm = torch.randperm(d)
    matrix = matrix[perm, :]
    matrix = matrix[:, perm]
    return matrix


# implement the constraint functions in numpy
# no need for speed for these functions


def h_exp(matrix):
    d = matrix.shape[0]
    # eigvals = scipy.linalg.eigvals(matrix*100)
    # return (np.exp(eigvals / 100) - 1).sum()
    return np.trace(scipy.linalg.expm(matrix) - np.eye(d))


def h_log(matrix):
    d = matrix.shape[0]
    i_d = np.eye(d)
    res = np.linalg.slogdet(i_d - matrix)
    if res[0] < 0:
        return np.nan
    else:
        return -res[1]


def h_inv(matrix):
    d = matrix.shape[0]
    i_d = np.eye(d)

    if h_eigen_max(matrix) > 1:
        return np.nan

    try:
        res = np.trace(np.linalg.inv(i_d - matrix)) - d
        if res < 0:
            return np.nan
        return res
    except np.linalg.LinAlgError:
        return np.nan


def h_power(matrix):
    d = matrix.shape[0]
    i_d = np.eye(d)
    return np.trace(np.linalg.matrix_power(i_d + matrix, d)) - d


def h_eigen_max(matrix):
    d = matrix.shape[0]
    x = np.ones(d)
    for _ in range(10):
        x = matrix @ x
        x = x / (np.linalg.norm(x) + 1e-8)
    return x @ matrix @ x


# create normed versions of the constraints, automatically
for func in [h_exp, h_log, h_inv, h_power]:
    func_normed = lambda matrix: func(matrix / np.linalg.norm(matrix, ord="fro"))
    func_normed.__name__ = f"{func.__name__}_normed"
    globals()[f"{func.__name__}_normed"] = func_normed


def generate_noise(d, noise_type):
    if noise_type == "single_cycle_length_3":
        noise = np.zeros((d, d), dtype=np.float64)
        noise[0, 1] = 1
        noise[1, 2] = 1
        noise[2, 0] = 1
    elif noise_type == "single_cycle_length_d":
        noise = np.zeros((d, d), dtype=np.float64)
        # cycle of length d 0 -> 1 -> 2 -> ... -> d-1 -> 0
        for i in range(d):
            noise[i, (i + 1) % d] = 1
    elif noise_type == "single_cycle_length_d/2":
        noise = np.zeros((d, d), dtype=np.float64)
        # cycle of length d 0 -> 1 -> 2 -> ... -> d//2 -> 0
        for i in range(d // 2):
            noise[i, (i + 1) % (d // 2)] = 1
    elif noise_type == "random_normal":
        noise = np.abs(np.random.randn(d, d))  # / d
    elif noise_type == "random_01":
        noise = np.random.uniform(0, 1, (d, d))  # / d
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    return noise


def plot_singularities(x, y, ax, i, **kwargs):
    kwargs["alpha"] = kwargs.get("alpha", 0.9)
    for xx, yy in zip(x, y):
        if np.isnan(yy):
            print(f"Singularity at {xx}")
            # ax.axvline(xx, 0, 1, **kwargs)
            ax.axvline(xx, 0, 1, ls=(5 * (i % 2), (4, 6)), lw=2, c=kwargs["color"])
            # ax.axvline(xx, 0, 1, lw=1, ls="-", c="gray")
            break


def analyze_over_range_of_d(
    penalties,
    range_d,
    m,
    epsilon=1e-6,
    noise_type="single_cycle_length_d",
):
    results = defaultdict(list)
    for d in range_d:
        dag = generate_random_torch_dag_matrix(d, m / d) / d
        noise = generate_noise(d, noise_type)

        matrix = (dag + epsilon * noise).numpy().astype(np.float64)
        for name, penalty_fn in penalties.items():
            tmp = penalty_fn(matrix)
            results[name].append(tmp)

    return results


def plot_analysis_over_range_of_d(
    penalties, colors, noise_type="single_cycle_length_d/2", epsilon=0.5, n_space=100, ax=None, ls="-"
):
    if ax is None:
        with plt.rc_context({"figure.figsize": (4, 3), "figure.dpi": 200}):
            ax = plt.gca()

    xx = [int(x) for x in np.logspace(np.log10(2), 3, n_space)]
    # results_d = analyze_over_range_of_d(
    #     penalties,
    #     range_d=xx,
    #     m=0,
    #     epsilon=0.5,
    #     noise_type="single_cycle_length_d",
    # )

    if noise_type == "random_01":
        # repeat multiple time and average
        results_d = defaultdict(list)
        for _ in range(10):
            results_d_ = analyze_over_range_of_d(
                penalties,
                range_d=xx,
                m=0,
                epsilon=epsilon,
                noise_type=noise_type,
            )
            for k, v in results_d_.items():
                results_d[k].append(v)
        for k, v in results_d.items():
            results_d[k] = np.mean(v, axis=0)
    else:
        results_d = analyze_over_range_of_d(
            penalties,
            range_d=xx,
            m=0,
            epsilon=epsilon,
            noise_type=noise_type,
        )

    for i, (name, result) in enumerate(results_d.items()):
        result_ = []
        xx_ = []
        for x, r in zip(xx, result):
            if np.isnan(r):
                xx_.append(x)
                result_.append(result_[-1] if len(result_) else 0)
                xx_.append(x)
                result_.append(10**30)
                break

            xx_.append(x)
            result_.append(r)
        lw = 3 if "rho" in name else 2
        ax.plot(xx_, result_, color=colors[i], label=name, linestyle=ls, lw=lw)
        # plot_singularities(xx, result, ax, color=colors[i], i=i)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([2, 10, 100, 1000], [2, 10, 100, 1000])
    ax.legend(fontsize=9)
    ax.set_xlabel("$d$", fontsize=16)
    ax.grid(True, which="major", ls="-", alpha=0.5)


def analyze_over_range_of_epsilon(
    penalties,
    range_epsilon,
    m,
    d,
    noise_type="single_cycle_length_d",
):
    results = defaultdict(list)
    dag = generate_random_torch_dag_matrix(d, m / d)
    noise = generate_noise(d, noise_type)
    for name, penalty_fn in penalties.items():
        singular = False
        for epsilon in range_epsilon:
            if singular:
                results[name].append(np.nan)
                continue
            matrix = dag + epsilon * noise
            tmp = penalty_fn(matrix.numpy().astype(np.float64))
            if np.isnan(tmp):
                singular = True

            results[name].append(tmp)

    return results


def plot_analysis_over_range_of_epsilon(
    penalties, colors, noise_type="single_cycle_length_d/2", range_eps=np.linspace(0, 1.0, 100), ax=None, ls="-"
):
    if ax is None:
        with plt.rc_context({"figure.figsize": (4, 3), "figure.dpi": 200}):
            ax = plt.gca()

    xx = range_eps
    results_d = analyze_over_range_of_epsilon(
        penalties,
        range_epsilon=xx,
        m=0,
        d=30,
        noise_type=noise_type,
    )

    for i, (name, result) in enumerate(results_d.items()):
        # result_ = []
        # is_singular = False
        # for r in result:
        #     is_singular |= np.isnan(r)
        #     result_.append(r if not is_singular else 10**30)
        result_ = []
        xx_ = []
        for x, r in zip(xx, result):
            if np.isnan(r):
                xx_.append(x)
                result_.append(result_[-1] if len(result_) else 0)
                xx_.append(x)
                result_.append(10**30)
                break

            xx_.append(x)
            result_.append(r)
        lw = 3 if "rho" in name else 2
        ax.plot(xx_[1:], result_[1:], color=colors[i], label=name, linestyle=ls, lw=lw)
        # plot_singularities(xx, result, ax=ax, color=colors[i], i=i)
        print(name, result[1:])
    ax.set_xscale("linear")
    ax.set_yscale("log")

    ax.set_xlabel("$\\epsilon$", fontsize=16)
    ax.grid(True, which="major", ls="-", alpha=0.5)


def plot_4():
    np.random.seed(0)
    torch.manual_seed(0)
    # increase font size and line width
    with plt.rc_context({"font.size": 13}):
        fig, axes = plt.subplots(2, 2, figsize=(7, 5), dpi=200)
    plot_analysis_over_range_of_d(noise_type="single_cycle_length_d/2", epsilon=0.5, ax=axes[0, 0])
    axes[0, 0].set_ylim(1e-16, 1e5)
    axes[0, 0].set_ylabel(r"$h(\epsilon \cdot$ d/2-cycle )")
    plot_analysis_over_range_of_d(noise_type="random_01", epsilon=0.5, ax=axes[1, 0])
    axes[1, 0].set_ylim(1e-1, 1e10)
    axes[1, 0].set_ylabel(r"$h(\epsilon \cdot$ [0,1]-noise )")
    plot_analysis_over_range_of_epsilon(noise_type="single_cycle_length_d/2", ax=axes[0, 1])
    axes[0, 1].set_ylim(1e-16, 1e5)
    axes[0, 1].set_ylabel(r"$h(\epsilon \cdot$ d/2-cycle)")
    plot_analysis_over_range_of_epsilon(noise_type="random_01", n_space=1000, ax=axes[1, 1])
    axes[1, 1].set_ylim(1e-5, 1e5)
    axes[1, 1].set_ylabel(r"$h(\epsilon \cdot$ [0,1]-noise)")

    axes[0, 0].legend().remove()
    axes[0, 1].legend().remove()
    axes[1, 0].legend().remove()
    for ax in axes.ravel():
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")

    axes[0, 0].set_xlabel("")
    axes[0, 1].set_xlabel("")
    axes[0, 0].set_xticklabels([])
    axes[0, 1].set_xticklabels([])
    axes[0, 1].set_yticklabels([])

    axes[0, 0].set_title("$\\epsilon = 0.5$", fontsize=16)
    # axes[1, 0].set_title("$\\epsilon = 0.5$")
    axes[0, 1].set_title("$d = 30$", fontsize=16)
    # axes[1, 1].set_title("$d = 25$")
    plt.tight_layout()
    plt.savefig("figures/constraints.asymptotic.pdf", bbox_inches="tight")
    plt.show()


def plot_2():
    np.random.seed(0)
    torch.manual_seed(0)
    # increase font size and line width
    with plt.rc_context({"font.size": 13}):
        fig, axes = plt.subplots(1, 2, figsize=(6, 2.5), dpi=200, squeeze=False, sharey=True)

    penalties = {
        r"$h_{\rho}$": h_eigen_max,
        r"$h_{\exp}$": h_exp,
        r"$h_{\log}$": h_log,
        r"$h_{\mathrm{inv}}$": h_inv,
        r"$h_{\mathrm{binom}}$": h_power,
    }
    colors = [
        "C1",
        "C0",
        "C2",
        "C3",
        "C4",
    ]
    plot_analysis_over_range_of_d(
        penalties, colors, noise_type="single_cycle_length_d/2", epsilon=0.5, ax=axes[0, 1], ls=(0, ()), n_space=50
    )
    plot_analysis_over_range_of_d(
        penalties, colors, noise_type="random_01", epsilon=0.5, ax=axes[0, 1], ls=(0, (5, 0.7, 1, 0.7)), n_space=50
    )
    axes[0, 0].set_ylim(1e-16, 1e16)
    axes[0, 0].set_ylabel(r"$h(A(\epsilon, d))$")
    range_epsilon = np.concatenate(
        [
            np.linspace(0.0, 0.7, 30),
        ]
    )

    plot_analysis_over_range_of_epsilon(
        penalties, colors, noise_type="single_cycle_length_d/2", range_eps=range_epsilon, ax=axes[0, 0], ls=(0, ())
    )
    plot_analysis_over_range_of_epsilon(
        penalties, colors, noise_type="random_01", range_eps=range_epsilon, ax=axes[0, 0], ls=(0, (5, 0.7, 1, 0.7))
    )
    axes[0, 1].set_ylim(1e-16, 1e16)

    axes[0, 0].legend().remove()

    legend = axes[0, 1].legend(
        fontsize=12,
        ncol=2,
        columnspacing=0.5,
        handletextpad=0.2,
        fancybox=False,
        framealpha=1,
        borderpad=0.2,
        bbox_to_anchor=(0.5, 1.5),
    )

    for ax in axes.ravel():
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    axes[0, 1].set_ylabel("")

    axes[0, 0].set_yticks([1e-15, 1e-10, 1e-5, 1e0, 1e5, 1e10, 1e15])

    axes[0, 0].set_title("$\\epsilon = 0.5$", fontsize=16)
    axes[0, 1].set_title("$d = 30$", fontsize=16)
    axes[0, 1].text(1, 1, "cycle")
    plt.tight_layout()
    plt.savefig("figures/constraints.asymptotic.pdf", bbox_inches="tight", pad_inches=0, bbox_extra_artists=[legend])
    plt.show()


if __name__ == "__main__":
    import matplotlib

    plt.style.use("default")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    plot_2()
