"""
Benchmark the speed of tensor operations.
"""
import abc
import time
import torch
import tqdm
from torch_sparse import SparseTensor


# models
# 1. dense matrix
# 2. sparse matrix torch
# 3. sparse matrix torch_sparse
# 4. own sparse matrix with fixed number of non-zero entries per row

# operations
# 1. matrix creation
# 2. matrix vector multiplication
# 3. matrix transpose vector multiplication
# 4. matrix matrix multiplication
# 5. matrix inversion


def time_it(func, n_loop=100, max_time=5, *args, **kwargs):
    total_time = 0
    # warm up with some iterations and some tensor operations
    for i in range(1000):
        a = torch.zeros(100) + 10

    for i in range(n_loop):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        total_time += end - start
        if total_time > max_time:
            n_loop = i + 1
            break

    average_time = total_time / n_loop
    return average_time


class Matrix(abc.ABC):
    def __init__(self, matrix: torch.Tensor):
        self._matrix = matrix
        self.matrix = None
        self.set_matrix(matrix)

    def benchmark(self, n_loop=100, max_time=5):
        benchmark = dict()

        # 1. matrix initialization
        benchmark["matrix_init"] = time_it(self.set_matrix, n_loop, max_time, self._matrix)

        # 2. matrix vector multiplication
        v = torch.randn(self._matrix.shape[1])
        benchmark["mv"] = time_it(self.mv, max_time, n_loop, v)

        # 3. matrix transpose vector multiplication
        v = torch.randn(self._matrix.shape[0])
        benchmark["mtv"] = time_it(self.mtv, max_time, n_loop, v)

        # 4. matrix matrix multiplication
        m = torch.randn(512, self._matrix.shape[0])
        benchmark["mm"] = time_it(self.mm, max_time, n_loop, m)

        # 5. matrix inversion
        # benchmark["inv"] = time_it(self.inv, max_time, n_loop)

        return benchmark

    @abc.abstractmethod
    def set_matrix(self, matrix: torch.Tensor):
        pass

    @abc.abstractmethod
    def mv(self, x):
        pass

    @abc.abstractmethod
    def mtv(self, x):
        pass

    @abc.abstractmethod
    def mm(self, x):
        pass

    def inv(self):
        pass


class DenseMatrix(Matrix):
    def set_matrix(self, matrix: torch.Tensor):
        self.matrix = matrix

    def mv(self, v):
        return self.matrix @ v

    def mtv(self, v):
        return self.matrix.T @ v

    def mm(self, m):
        return m @ self.matrix

    def inv(self):
        d = self.matrix.shape[0]
        if d > 10_000:
            return None
        return torch.inverse(torch.eye(d) - self.matrix)


class TorchCSR(Matrix):
    def set_matrix(self, matrix: torch.Tensor):
        self.matrix = matrix.to_sparse_csr()

    def mv(self, v):
        return self.matrix @ v

    def mtv(self, v):
        return v @ self.matrix

    def mm(self, m):
        return self.matrix @ m.T


class TorchCOO(Matrix):
    def set_matrix(self, matrix: torch.Tensor):
        self.matrix = matrix.to_sparse_coo().coalesce()

    def mv(self, v):
        return self.matrix @ v

    def mtv(self, v):
        return self.matrix.T @ v

    def mtv2(self, v):
        return v @ self.matrix

    def mm(self, m):
        return self.matrix @ m.T


class TorchSparse(Matrix):
    def set_matrix(self, matrix: torch.Tensor):
        self.matrix = SparseTensor.from_dense(matrix)

    def mv(self, v):
        if len(v.shape) == 1:
            v = v.unsqueeze(1)
        return self.matrix @ v

    def mtv(self, v):
        if len(v.shape) == 1:
            v = v.unsqueeze(1)
        return self.matrix.t() @ v

    def mm(self, m):
        return self.matrix @ m.T


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)

    benchmark_results = []
    for d in [100, 1_000, 2_000, 5_000, 10_000, 20_000, 30_000]:
        for k in tqdm.tqdm([10, 20, 50, 100, 200]):
            torch.manual_seed(0)
            M = torch.randn(d, d)
            mask = torch.rand(d, d) < k / d
            M = M * mask

            dense = DenseMatrix(M)
            sparse = TorchSparse(M)
            # coo = TorchCOO(M)
            # csr = TorchCSR(M)

            res = dict()
            for matrix in [dense, sparse]:
                b = matrix.benchmark(3)
                res[matrix.__class__.__name__] = b
            import pandas as pd

            df = pd.DataFrame(res).T
            df["k"] = k
            df["d"] = d
            benchmark_results.append(df)

    df = pd.concat(benchmark_results)
    print(df)

    # seaborn plot
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    sns.set_context("paper")
    sns.set(font_scale=1.5)
    sns.set_style("ticks")
    # lineplot
    # one subplot for each operation, hue is matrix type, x-axis is d
    # each row is a different density
    df = df.reset_index()
    df = df.melt(id_vars=["index", "k", "d"], var_name="operation", value_name="time")
    # do not share y-axis
    g = sns.FacetGrid(df, col="operation", hue="index", row="k", height=4, aspect=1.5, sharey=False)
    g.map(sns.lineplot, "d", "time")
    g.add_legend()
    g.set_titles("{row_name} - {col_name}")
    g.set_axis_labels("d", "time [s]")
    import matplotlib.pyplot as plt

    plt.show()
