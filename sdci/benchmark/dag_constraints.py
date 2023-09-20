import copy
from typing import Union, List

import networkx as nx
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import scipy.sparse.csgraph
import torch

from abc import ABC, abstractmethod
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import tqdm

"""
The benchmark class is a base class for all the dag constraint benchmarks.
It contains the following methods:
- set_matrix: sets the matrix to be optimized
- gradient_update: performs a gradient update given the gradient computed by the subclass
- compute_metrics: computes the metrics of interest


Each subclass should implement:
- a constructor that takes a matrix as input
- a method called compute_gradient that computes the gradient of the loss function

Parameters of the optimization:
- [4] the objective: 
    tr[exp(W)], tr[log(I-W)], tr[inv(I-W)], largest-eigenvalue magnitude, $Σ_{topk} | lambda_k |_p$

- [3+] the frequency of the gradient computation: 
    all, every k [5,20]
- the approximation:
    - the eigendecomposition approximation: 
        all, top-k
    - the low rank approximation:
        not sure how to benchmark this
    - the node-sampling approximation, sample k nodes according to some distribution:
        none, uniform, correlated by 

- [4] the re-normalization trick to divide A by some sub-multiplicative norm: 
    none, ||A||_F, ||A||_1, ||A||_2, ||A||_inf 
- [3] the projection trick on identical norm(riemannian gradient descent): 
    none, full projection, partial projection 
- [2] the preprocessing of the matrix: square coeff, abs coeff

Hyperparameters:
- the dimension of the matrix
- the sparsity of the matrix

Metrics:
- the time per epoch
- the number of non zero coefficients
- the dag-ness of the matrix:
    - the largest eigenvalue magnitude
    - the log determinant
    - the log determinant of the un-weighted matrix
- the number of edges
- the frobenius norm of the matrix
- the distance to the original matrix
- the distance to zero
"""


class MatrixWithGradient(ABC):
    """
    Base class for all the matrices with DAG constraint gradient.
    """

    def __init__(
        self,
        matrix,
        anchor_matrix=None,
        project_norm=0.0,
        normalization=None,
        regularization_to_anchor=0.0,
        name_suffix="",
    ):
        self.matrix = None
        self.anchor_matrix = anchor_matrix
        self.original_matrix = None
        self.d = None
        self.identity = None

        self.regularization_to_anchor = regularization_to_anchor
        self.project_norm = project_norm
        self.normalization = normalization

        self._gradient = None
        self.name_suffix = name_suffix

        self.total_time = None
        self.metrics = None
        self.n_updates = None
        self.reset_tracking()

        self.set_matrix(matrix)

    def reset_tracking(self):
        self.total_time = 0
        self.metrics = []
        self.n_updates = 0

    @abstractmethod
    def set_matrix(self, matrix: Union[torch.Tensor, np.ndarray, scipy.sparse.csr_matrix]):
        pass

    def gradient_update(self, lr=5e-3):
        success = True
        if self.regularization_to_anchor > 0:
            l2_gradient = self.regularization_to_anchor * 2 * (self.matrix - self.anchor_matrix)
        else:
            l2_gradient = 0

        start_time = time.time()
        try:
            gradient = self.compute_gradient()
            if self.project_norm > 0:
                gradient = self.project_gradient(gradient)
        except RuntimeError as e:
            gradient = 0
            print(e)
            success = False
        self.matrix = self.relu(self.matrix - lr * (gradient + l2_gradient))
        end_time = time.time()

        self.total_time += end_time - start_time
        self._gradient = gradient
        self.n_updates += 1
        return success

    @abstractmethod
    def relu(self, x):
        pass

    def project_gradient(self, gradient):
        if isinstance(self.matrix, torch.Tensor):
            matrix_l2 = torch.linalg.matrix_norm(self.matrix, ord="fro")
        elif isinstance(self.matrix, scipy.sparse.csr_matrix):
            matrix_l2 = scipy.sparse.linalg.norm(self.matrix, ord="fro")
        else:
            raise ValueError("Matrix type not supported")

        gradient -= (
            self.project_norm * (self.matrix * gradient).sum() / matrix_l2**2
        ) * self.matrix
        return gradient

    def train(self, n_epochs, test_n_epochs, lr=1e-3):
        for i in tqdm.tqdm(range(n_epochs)):
            if i % test_n_epochs == 0:
                self.compute_metrics()
            success = self.gradient_update(lr)
            if not success:
                break
        self.compute_metrics()

    def train_until_dag(
        self, test_n_epochs, lr=1e-3, tol=1e-4, max_epochs=100_000, max_time=60 * 5
    ):
        start = time.time()
        for i in tqdm.tqdm(range(max_epochs)):
            reached_dag = False
            if i % test_n_epochs == 0:
                self.compute_metrics()
                reached_dag = self.metrics[-1]["threshold_to_dag"] < tol
            success = self.gradient_update(lr)
            if not success or reached_dag:
                break
            if time.time() - start > max_time:
                break
        self.compute_metrics()

    def compute_metrics(self):
        # try:
        #     largest_eigen = torch.linalg.eigvals(self.matrix).abs().max().item()
        # except RuntimeError:
        largest_eigen = -1

        self.metrics.append(
            {
                "epoch": self.n_updates,
                "largest_eigen": largest_eigen,
                "time": self.total_time,
                "logdet": -1,  # self.compute_metric_logdet(),
                "log_cycles": -1,  # np.log(1 + self.upperbound_cycles()),
                "original_error_fro": matrix_norm(self.matrix - self.original_matrix),
                "original_error_1": matrix_norm(self.matrix - self.original_matrix, 1),
                "original_error_inf": matrix_norm(self.matrix - self.original_matrix, np.inf),
                "norm_fro": matrix_norm(self.matrix, "fro"),
                "norm_1": matrix_norm(self.matrix, 1),
                "norm_inf": matrix_norm(self.matrix, np.inf),
                "n_non_zero": count_non_zero_coordinates(self.matrix),
                "n_non_zero_gradient": count_non_zero_coordinates(self._gradient),
                "threshold_to_dag": self.compute_metric_threshold_to_dag(),
                "cycles_weights": self.compute_cycle_metrics(),
                "eigen": self.compute_eigen_metrics(),
                "matrix_history": copy.copy(self.matrix),
            }
        )

    @abstractmethod
    def compute_gradient(self):
        raise NotImplementedError

    def base_name(self):
        return self.__class__.__name__

    def name(self):
        base_name = self.base_name()
        if self.project_norm > 0:
            base_name += str("-PN%.2f" % self.project_norm)
        if self.normalization is not None:
            base_name += str("-NORM%s" % self.normalization)
        return base_name + self.name_suffix

    def get_matrix(self):
        if self.normalization is not None:
            return NotImplementedError
        return self.matrix

    # ############### metrics ###############
    def upperbound_cycles(self):
        return -1

    def compute_metric_logdet(self):
        return -1

    @abstractmethod
    def thresholded_matrix(self, threshold):
        pass

    def compute_metric_threshold_to_dag(self):
        return scipy.optimize.bisect(
            lambda t: is_dag(self.thresholded_matrix(t)) - 0.5 if t > 0 else -0.5,
            0,
            2,
            xtol=1e-10,
        )

    def get_matrix_numpy(self):
        return to_numpy(self.matrix)

    def compute_cycle_metrics(self):
        # compute only for matrix smaller than 20x20
        if self.d > 11 or count_non_zero_coordinates(self.get_matrix()) > 50:
            return None
        matrix = self.get_matrix_numpy()
        # compute list of all cycles
        graph = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
        cycles = list(nx.simple_cycles(graph))
        # compute log weight of each cycle
        cycle_weights = dict()
        for cycle in cycles:
            weight = np.prod([matrix[i, j] for i, j in zip(cycle, cycle[1:] + cycle[:1])])
            cycle_weights[tuple(cycle)] = np.log(weight)

        return cycle_weights

    def compute_eigen_metrics(self):
        # compute only for matrix smaller than 20x20
        if self.d > 20:
            return None
        matrix = self.get_matrix_numpy()
        # compute the eigen decomposition with right and left eigenvectors using scipy
        eigen_decomposition = scipy.linalg.eig(matrix, left=True, right=True)
        eigenvalues = eigen_decomposition[0]
        left_eigenvectors = eigen_decomposition[1]
        right_eigenvectors = eigen_decomposition[2]

        # compute rank one decomposition
        rank_one_decomposition = []
        for i in range(self.d):
            coefficient = eigenvalues[i] / right_eigenvectors[:, i].dot(
                left_eigenvectors[:, i].conj()
            )
            rank_one = (
                coefficient
                * right_eigenvectors[:, i][:, None]
                @ left_eigenvectors[:, i][None, :].conj()
            )
            rank_one_decomposition.append(rank_one)

        res = {
            "eigenvalues": eigenvalues,
            "right_eigenvectors": right_eigenvectors,
            "left_eigenvectors": left_eigenvectors,
            "rank_one_decomposition": rank_one_decomposition,
        }
        return res


def to_scipy_sparse(matrix):
    if isinstance(matrix, scipy.sparse.csr_matrix):
        return matrix
    elif isinstance(matrix, torch.Tensor):
        return scipy.sparse.csr_matrix(matrix.numpy())
    elif isinstance(matrix, np.ndarray):
        return scipy.sparse.csr_matrix(matrix)
    else:
        raise ValueError("Matrix type not supported")


def to_numpy(matrix):
    if isinstance(matrix, scipy.sparse.csr_matrix):
        return matrix.toarray()
    elif isinstance(matrix, torch.Tensor):
        return matrix.numpy()
    elif isinstance(matrix, np.ndarray):
        return matrix
    else:
        raise ValueError("Matrix type not supported")


def matrix_norm(matrix, ord: Union[str, int] = "fro"):
    if isinstance(matrix, scipy.sparse.csr_matrix):
        return scipy.sparse.linalg.norm(matrix, ord=ord)
    elif isinstance(matrix, torch.Tensor):
        return torch.linalg.matrix_norm(matrix, ord=ord).item()
    elif isinstance(matrix, np.ndarray):
        return np.linalg.norm(matrix, ord=ord)
    else:
        raise ValueError("Matrix type not supported")


def count_non_zero_coordinates(matrix):
    if isinstance(matrix, scipy.sparse.csr_matrix):
        return matrix.count_nonzero()
    elif isinstance(matrix, torch.Tensor):
        return torch.count_nonzero(matrix).item()
    elif isinstance(matrix, np.ndarray):
        return np.count_nonzero(matrix)
    elif matrix is None:
        return None
    else:
        raise ValueError("Matrix type not supported: " + str(type(matrix)))


def is_dag(matrix: Union[torch.Tensor, np.ndarray, scipy.sparse.csr_matrix]):
    """
    Check if a matrix is a DAG
    """
    d = matrix.shape[0]
    if isinstance(matrix, torch.Tensor):
        ones = torch.ones
        count_nonzero = torch.count_nonzero
    elif isinstance(matrix, np.ndarray) or isinstance(matrix, scipy.sparse.csr_matrix):
        ones = np.ones
        count_nonzero = np.count_nonzero
    else:
        raise ValueError("Matrix type not supported")
    v = ones(d, dtype=matrix.dtype)
    nnz_old = count_nonzero(v)
    for i in range(d):
        v = matrix @ v
        nnz_new = count_nonzero(v)
        if nnz_new >= nnz_old:
            return False
        if nnz_new == 0:
            return True
        nnz_old = nnz_new
        v = v / (v + 1e-8)
    return True


class MatrixWithGradientTorch(MatrixWithGradient, ABC):
    def set_matrix(self, matrix: Union[torch.Tensor, np.ndarray, scipy.sparse.csr_matrix]):
        self.d = matrix.shape[0]
        if isinstance(matrix, np.ndarray):
            self.matrix = torch.from_numpy(matrix.copy())
        elif isinstance(matrix, scipy.sparse.csr_matrix):
            self.matrix = torch.from_numpy(matrix.toarray().copy())
        elif isinstance(matrix, torch.Tensor):
            self.matrix = matrix.clone()
        else:
            raise ValueError("Matrix type not supported")
        self.original_matrix = self.matrix.clone()
        self.identity = torch.eye(self.d)

    def get_matrix(self):
        if self.normalization is not None:
            norm = torch.linalg.matrix_norm(self.matrix, ord=self.normalization)
            return self.matrix / norm
        else:
            return self.matrix

    def upperbound_cycles(self):
        matrix = (self.matrix > 0).float() / self.d
        # sum_{k=1,d} Tr[A^k] = Tr[A+A^2+...+A^d]
        inv = torch.linalg.inv(self.identity - matrix)
        return torch.trace(
            inv @ (self.identity - torch.linalg.matrix_power(matrix, self.d)) @ matrix
        ).item()

    def compute_metric_logdet(self):
        return -torch.linalg.slogdet(self.identity - self.matrix).logabsdet.item()

    def relu(self, x):
        return torch.relu(x)

    def get_skeleton(self):
        return (self.matrix > 0).float()

    def thresholded_matrix(self, threshold):
        return (self.matrix > threshold).float()


class MatrixWithGradientScipySparse(MatrixWithGradient, ABC):
    def set_matrix(self, matrix: Union[torch.Tensor, np.ndarray, scipy.sparse.csr_matrix]):
        self.d = matrix.shape[0]
        if isinstance(matrix, np.ndarray):
            self.matrix = scipy.sparse.csr_matrix(matrix, dtype=np.float32)
        elif isinstance(matrix, scipy.sparse.csr_matrix):
            self.matrix = matrix.copy()
        elif isinstance(matrix, torch.Tensor):
            self.matrix = scipy.sparse.csr_matrix(matrix.numpy())
        else:
            raise ValueError("Matrix type not supported")
        self.original_matrix = self.matrix.copy()
        self.identity = scipy.sparse.eye(self.d)

    def get_matrix(self, sub_matrix=None):
        if self.normalization is not None:
            norm = scipy.sparse.linalg.norm(self.matrix, ord=self.normalization)
            matrix = self.matrix / norm
        else:
            matrix = self.matrix

        if sub_matrix is None:
            return matrix
        else:
            return matrix[sub_matrix, :][:, sub_matrix]

    def relu(self, x):
        return x.maximum(0.0)

    def compute_metric_logdet(self):
        return -np.linalg.slogdet(self.identity.toarray() - self.matrix.toarray())[1]

    def thresholded_matrix(self, threshold):
        return (self.matrix > threshold).astype(np.float32)


class ExponentialGradientTorch(MatrixWithGradientTorch):
    def compute_gradient(self):
        return torch.linalg.matrix_exp(self.get_matrix()).T

    def base_name(self):
        return "T-Exp"


class LogGradientTorch(MatrixWithGradientTorch):
    def compute_gradient(self):
        gradient = torch.linalg.inv(self.identity - self.get_matrix()).T
        return gradient

    def base_name(self):
        return "T-Log"


class InverseGradientTorch(MatrixWithGradientTorch):
    def compute_gradient(self):
        return torch.linalg.matrix_power(self.identity - self.get_matrix(), -2).T

    def base_name(self):
        return "T-Inv"


def normalize(vector):
    return vector / torch.linalg.vector_norm(vector, ord=2)


class PowerIterationTorch(MatrixWithGradientTorch):
    def __init__(
        self,
        matrix,
        reset_cache_every=100,
        n_iterations_init=5,
        n_iterations_cache=2,
        identity_penalty=10,
        transpose_penalty=0,
        **kwargs,
    ):
        super().__init__(matrix, **kwargs)
        self.reset_cache_every = reset_cache_every
        self.n_iterations_init = n_iterations_init
        self.n_iterations_cache = n_iterations_cache
        self.identity_penalty = identity_penalty
        self.transpose_penalty = transpose_penalty
        self.v = None
        self.v_T = None
        self.initialize_eigenvector()

    def initialize_eigenvector(self):
        self.v = torch.ones(self.d) / np.sqrt(self.d)
        self.v_T = torch.ones(self.d) / np.sqrt(self.d)

    def iterate(self, n_iter=1):
        matrix = self.get_matrix() + 1e-6
        for i in range(n_iter):
            self.v = normalize(matrix @ self.v)
            self.v_T = normalize(self.v_T @ matrix)

    def compute_gradient(self):
        if self.reset_cache_every > 0 and self.n_updates % self.reset_cache_every == 0:
            self.initialize_eigenvector()
            self.iterate(self.n_iterations_init)
        else:
            self.iterate(self.n_iterations_cache)

        grad = torch.outer(self.v_T, self.v) / torch.inner(self.v_T, self.v)
        grad = grad + self.identity_penalty * self.identity
        grad = grad + self.transpose_penalty * self.matrix.T

        return grad

    def base_name(self):
        return "T-Power"


class SCCTopEigenvalueGradient(MatrixWithGradientScipySparse):
    def __init__(self, matrix, keep_v=True, eigen_transform=None, update_scc_freq=10, **kwargs):
        super().__init__(matrix, **kwargs)
        self.keep_v = keep_v
        if eigen_transform is None:
            eigen_transform = lambda x: x / np.abs(x)
        self.eigen_transform = eigen_transform
        self.update_scc_freq = update_scc_freq
        # store v0 from different scc in a single vector
        self.d = matrix.shape[0]
        self.v0 = np.ones(self.d) / np.sqrt(self.d)
        self.v0_T = np.ones(self.d) / np.sqrt(self.d)

        self.scc_list = []

    def get_skeleton(self, dtype=np.float32):
        return (self.matrix > 0).astype(dtype)

    def update_scc(self):
        n_components, labels = scipy.sparse.csgraph.connected_components(
            csgraph=self.get_matrix(), directed=True, return_labels=True, connection="strong"
        )
        self.scc_list = []
        for i in range(n_components):
            scc = np.where(labels == i)[0]
            self.scc_list.append(scc)

    def compute_gradient(self, scc=None):
        # fix when scc = everybody
        if scc is None:
            if self.n_updates % self.update_scc_freq == 0:
                self.update_scc()
            self.update_scc()

            gradient = scipy.sparse.csr_matrix(self.matrix.shape)
            for scc in self.scc_list:
                if len(scc) == self.d:
                    scc = slice(None)
                    idx = slice(None)
                else:
                    idx = np.ix_(scc, scc)

                scc_gradient = self.compute_gradient(scc)
                gradient[idx] = scc_gradient

            return gradient

        if type(scc) != slice and len(scc) == 1:
            return 1

        matrix = self.get_matrix(scc)

        double_matrix = scipy.sparse.block_diag([matrix, matrix.T], format="csr")
        v0 = np.concatenate([self.v0[scc], self.v0_T[scc]])
        eigen_res = scipy.sparse.linalg.eigs(double_matrix, k=1, which="LM", v0=v0, tol=1e-6, ncv=5)

        eigen_triplets = extract_eigen_triplets_from_double_eigen_res(eigen_res)
        eigen_triplet = eigen_triplets[0]
        eigen_triplet[0] = np.abs(eigen_triplet[0])
        eigen_triplet[1] = np.abs(eigen_triplet[1])

        if self.keep_v:
            self.v0[scc] = eigen_triplet[0]
            self.v0_T[scc] = eigen_triplet[1]

        print(eigen_triplet[2])
        gradient = form_approximation(
            [eigen_triplet],
            self.eigen_transform,
            self.get_skeleton(np.float32)[scc, :][:, scc],
            1,
        )

        if gradient.real.max() > 100:
            print("WARNING: gradient is too large")
            em = self.compute_eigen_metrics()
            print(sorted(em["eigenvalues"], key=np.abs, reverse=True))
            sns.heatmap(to_numpy(self.get_matrix()))
            plt.show()
            return 0

        return gradient.real

    def base_name(self):
        return "SCC-Top1"


class PowerIteration(MatrixWithGradientScipySparse):
    def __init__(self, matrix, **kwargs):
        super().__init__(matrix, **kwargs)
        self.d = matrix.shape[0]
        self.v0 = np.ones(self.d) / np.sqrt(self.d)
        self.v0_T = np.ones(self.d) / np.sqrt(self.d)

    def get_skeleton(self, dtype=np.float32):
        return (self.matrix > 0).astype(dtype)

    def power_method(self, n_iter=5):
        matrix = self.get_matrix().toarray() + 1e-6
        v = self.v0
        vT = self.v0_T
        for i in range(n_iter):
            v = matrix.dot(v)
            v = v / np.linalg.norm(v)
            vT = matrix.T.dot(vT)
            vT = vT / np.linalg.norm(vT)

        self.v0 = v
        self.v0_T = vT

    def compute_gradient(self):
        self.power_method(5)
        # skeleton = self.get_skeleton(np.float32)
        gradient = (
            np.outer(self.v0_T, self.v0)
            * self.get_skeleton(np.float32).toarray()
            / self.v0_T.dot(self.v0)
        )
        gradient = scipy.sparse.csr_matrix(gradient)
        return gradient

    def base_name(self):
        return "PowerIteration"


class PowerIterationSCCMask(MatrixWithGradientScipySparse):
    def __init__(self, matrix, update_scc_freq=200, **kwargs):
        super().__init__(matrix, **kwargs)
        self.d = matrix.shape[0]
        self.v0 = np.ones(self.d) / np.sqrt(self.d)
        self.v0_T = np.ones(self.d) / np.sqrt(self.d)
        self.update_scc_freq = update_scc_freq
        self.mask = None
        self.scc_list = None

    def get_skeleton(self, dtype=np.float32):
        return (self.matrix > 0).astype(dtype)

    def update_scc(self):
        n_components, labels = scipy.sparse.csgraph.connected_components(
            csgraph=self.get_matrix(), directed=True, return_labels=True, connection="strong"
        )
        self.scc_list = []
        for i in range(n_components):
            scc = np.where(labels == i)[0]
            self.scc_list.append(scc)

        if len(self.scc_list) == 1:
            mask = 1
        else:
            mask = np.zeros((self.d, self.d))
            for scc in self.scc_list:
                mask[np.ix_(scc, scc)] = 1
            # mask = scipy.sparse.csr_matrix(mask)
        self.mask = mask

        print(len(self.scc_list))
        print(max([len(scc) for scc in self.scc_list]), self.d)

    def power_method(self, n_iter=5):
        matrix = self.get_matrix()
        v = self.v0
        vT = self.v0_T
        for i in range(n_iter):
            v = matrix.dot(v) + 1e-6 * v.sum()
            v = v / np.linalg.norm(v)
            vT = matrix.T.dot(vT) + 1e-6 * vT.sum()
            vT = vT / np.linalg.norm(vT)

        self.v0 = v
        self.v0_T = vT

    def compute_gradient(self):
        if self.n_updates % self.update_scc_freq == 0:
            self.update_scc()
        self.power_method(5)
        gradient = (
            np.outer(self.v0_T, self.v0)
            * self.get_skeleton(np.float32).toarray()
            / self.v0_T.dot(self.v0)
        ) * self.mask
        gradient = scipy.sparse.csr_matrix(gradient)
        return gradient

    def base_name(self):
        return "PowerIterationSCCMask"


class SCCPowerIteration(MatrixWithGradientScipySparse):
    def __init__(self, matrix, update_scc_freq=100, **kwargs):
        super().__init__(matrix, **kwargs)
        self.d = matrix.shape[0]
        self.update_scc_freq = update_scc_freq
        self.v0 = np.ones(self.d) / np.sqrt(self.d)
        self.v0_T = np.ones(self.d) / np.sqrt(self.d)

        self.scc_list = None

    def update_scc(self):
        n_components, labels = scipy.sparse.csgraph.connected_components(
            csgraph=self.get_matrix(), directed=True, return_labels=True, connection="strong"
        )
        self.scc_list = []
        for i in range(n_components):
            scc = np.where(labels == i)[0]
            self.scc_list.append(scc)
        print(len(self.scc_list))
        print(max([len(scc) for scc in self.scc_list]), self.d)

    def get_skeleton(self, dtype=np.float32):
        return (self.matrix > 0).astype(dtype)

    def power_method(self, scc=slice(None), n_iter=5):
        matrix = self.get_matrix(scc)
        v = self.v0[scc]
        vT = self.v0_T[scc]
        for i in range(n_iter):
            v = matrix.dot(v) + 1e-6 * v.sum()
            v = v / np.linalg.norm(v)
            vT = matrix.T.dot(vT) + 1e-6 * vT.sum()
            vT = vT / np.linalg.norm(vT)

        self.v0[scc] = v
        self.v0_T[scc] = vT

    def compute_gradient(self, scc=None):
        if scc is None:
            if self.n_updates % self.update_scc_freq == 0:
                self.update_scc()
            gradient = scipy.sparse.csr_matrix(self.matrix.shape)
            for scc in self.scc_list:
                if len(scc) == self.d:
                    scc = slice(None)
                    idx = slice(None)
                else:
                    idx = np.ix_(scc, scc)

                scc_gradient = self.compute_gradient(scc)
                gradient[idx] = scc_gradient

            return gradient

        if type(scc) != slice and len(scc) == 1:
            return 1
        self.power_method(scc, 5)
        # skeleton = self.get_skeleton(np.float32)

        gradient = self.get_skeleton()[scc, :][:, scc]
        normalization = self.v0_T[scc].dot(self.v0[scc])
        gradient = gradient.multiply(self.v0_T[scc][:, None] / normalization)
        gradient = gradient.multiply(self.v0[scc][None, :])

        return gradient

    def base_name(self):
        return "SCCPowerIteration"


class SCCPowerIterationFaster(MatrixWithGradientScipySparse):
    def __init__(self, matrix, update_scc_freq=100, **kwargs):
        super().__init__(matrix, **kwargs)
        self.mask = None
        self.d = matrix.shape[0]
        self.update_scc_freq = update_scc_freq
        self.v0 = np.ones(self.d) / np.sqrt(self.d)
        self.v0_T = np.ones(self.d) / np.sqrt(self.d)

        self.scc_list = None

    def update_scc(self):
        n_components, labels = scipy.sparse.csgraph.connected_components(
            csgraph=self.get_matrix(), directed=True, return_labels=True, connection="strong"
        )
        self.scc_list = []
        for i in range(n_components):
            scc = np.where(labels == i)[0]
            self.scc_list.append(scc)
        print(len(self.scc_list))
        mask = np.zeros((self.d, self.d))
        if len(self.scc_list) == 1:
            mask = 1
        else:
            for scc in self.scc_list:
                mask[np.ix_(scc, scc)] = 1
            mask = scipy.sparse.csr_matrix(mask)
        self.mask = mask

    def get_skeleton(self, dtype=np.float32):
        return (self.matrix > 0).astype(dtype)

    def power_method(self, n_iter=5):
        def iterate(mat, v):
            v_old = v
            v_new = mat.dot(v_old)
            for scc in self.scc_list:

                if len(scc) == self.d:
                    v_new = v_new + 1e-6 * v_old.sum()
                    v_new = v_new / np.linalg.norm(v_new)
                elif len(scc) == 1:
                    continue
                else:
                    v_new[scc] += 1e-6 * v_old[scc].sum()
                    v_new[scc] = v_new[scc] / np.linalg.norm(v_new[scc])
            return v_new

        matrix = self.get_matrix() * self.mask
        v = self.v0
        vT = self.v0_T
        for i in range(n_iter):
            v = iterate(matrix, v)
            vT = iterate(matrix.T, vT)

        self.v0 = v
        self.v0_T = vT

    def compute_gradient(self, scc=None):
        if self.n_updates % self.update_scc_freq == 0:
            self.update_scc()

        self.power_method(5)
        gradient = self.get_skeleton(np.float32) * self.mask
        if len(self.scc_list) == 1:
            normalization = self.v0_T.dot(self.v0)
            gradient = gradient.multiply(self.v0_T[:, None] / normalization)
            gradient = gradient.multiply(self.v0[None, :])
        else:
            v = self.v0.copy()
            vT = self.v0_T
            for scc in self.scc_list:
                v[scc] /= v[scc].dot(vT[scc])
            gradient = gradient.multiply(self.v0_T[:, None])
            gradient = gradient.multiply(self.v0[None, :])
        return gradient.tocsr(False)

    def base_name(self):
        return "SCCPowerIterationF"


def extract_eigen_triplets_from_double_eigen_res(eigen_res):
    eigen_values = eigen_res[0]
    eigen_vectors = eigen_res[1]
    eigen_triplets = []
    found_eigenvalues = []
    d = eigen_vectors.shape[0] // 2
    for i in range(len(eigen_values)):
        if np.isclose(eigen_values[i], found_eigenvalues, rtol=1e-5).any():
            continue
        found_eigenvalues.append(eigen_values[i])
        eigen_triplets.append([eigen_vectors[:d, i], eigen_vectors[d:, i].conj(), eigen_values[i]])

    return eigen_triplets


def match_eigen_pairs(
    eig_values_right, eig_vectors_right, eig_values_left, eig_vectors_left, tol=1e-5
):
    eigen_triplets = []
    eig_values_left = eig_values_left.conj()
    for i in range(len(eig_values_right)):
        # match right eigenvalue to left eigenvalues
        j = np.argmin(np.abs(eig_values_left - eig_values_right[i]))
        # check that the eigenvalue is close enough
        if np.abs(eig_values_left[j] - eig_values_right[i]) < tol:
            eigen_triplets.append(
                [eig_vectors_right[:, i], eig_vectors_left[:, j], eig_values_right[i]]
            )
        else:
            print(
                "Warning: could not match eigenvalue %f to %f"
                % (eig_values_right[i], eig_values_left[j])
            )
    eigen_triplets.sort(key=lambda x: np.abs(x[2]), reverse=True)
    if len(eigen_triplets) < len(eig_values_right):
        print(
            "Warning: only matched %d/%d eigenvalues: "
            % (len(eigen_triplets), len(eig_values_right))
        )
    return eigen_triplets


def form_approximation(eigen_triplets, eigen_values_transform, skeleton, max_k=None):
    rank_ones = []
    eigen_triplets = sorted(eigen_triplets, key=lambda x: np.abs(x[2]), reverse=True)
    eigen_triplets = eigen_triplets[:max_k]
    eigen_triplets = [[x[0], x[1], eigen_values_transform(x[2])] for x in eigen_triplets]

    for eigen_vec_right, eigen_vec_left, eigen_val_transformed in eigen_triplets:
        rank_one = skeleton.copy()
        # rank_one = scipy.sparse.csr_matrix(np.ones(rank_one.shape))
        rank_one = rank_one.multiply(eigen_vec_left[:, None])
        rank_one = rank_one.multiply(eigen_vec_right[None, :].conjugate())
        rank_one = rank_one.multiply(
            eigen_val_transformed / eigen_vec_left.dot(eigen_vec_right.conjugate())
        )
        rank_ones.append(rank_one.real)
    return sum(rank_ones)


def generate_random_torch_matrix(d):
    matrix = torch.rand((d, d), dtype=torch.float32) / d
    return matrix


def generate_random_sparse_torch_matrix(d, sparsity=0.1):
    matrix = torch.rand((d, d), dtype=torch.float32) / d
    matrix = matrix * (torch.rand((d, d)) < sparsity) / sparsity
    matrix[torch.eye(d, dtype=torch.bool)] = 0
    return matrix


def plot_metrics(methods: List[MatrixWithGradient]):
    # plot pairs of metrics on the same figure, one subplot per pair, using seaborn
    # the metrics from different methods are plotted on the same figure

    metrics = []
    for method in methods:
        metrics.append(pd.DataFrame(method.metrics))
        metrics[-1]["method"] = method.name()
    metrics = pd.concat(metrics).reset_index(drop=True)
    title = "Metrics"

    pairs_of_metrics = [
        ("epoch", "time"),
        ("epoch", "threshold_to_dag"),
        ("time", "threshold_to_dag"),
        (
            "threshold_to_dag",
            "n_non_zero",
        ),
        (
            "threshold_to_dag",
            "norm_1",
        ),
    ]
    # plot pairs of metrics on the same figure, one subplot per pair, using seaborn
    n_cols = 3
    n_rows = int(np.ceil(len(pairs_of_metrics) / n_cols))
    fig, axes = plt.subplots(n_rows, 3, figsize=(4 * n_cols, 3 * n_rows))
    for (i, (x, y)) in enumerate(pairs_of_metrics):
        ax = axes[i // n_cols, i % n_cols]
        sns.lineplot(data=metrics, x=x, y=y, hue="method", ax=ax, alpha=0.8, legend=i == 0)
        ax.set_title("%s vs %s" % (y, x))
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig("metrics.pdf", bbox_inches="tight")
    plt.show()

    # plot all matrices as heatmap on the same figure with shared colorbar
    # fig, axes = plt.subplots(1, len(methods), figsize=(3 * len(methods), 3))
    # for i, method in enumerate(methods):
    #     sns.heatmap(to_numpy(method.get_matrix()), ax=axes[i], cbar=i == 0)
    #     axes[i].set_title(method.name())
    # fig.suptitle("Matrices")
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    torch.manual_seed(2)
    d = 1000
    m = 10
    W0 = generate_random_sparse_torch_matrix(d, m / d)
    W0 += torch.eye(d)
    anchor = generate_random_sparse_torch_matrix(d, m / d)
    anchor += torch.eye(d)

    anchor = W0.clone()

    # d = 4
    # W0 = (
    #     np.array([[0, 1, 0, 1], [0.7, 0, 0, 0], [0, 0, 0, 1], [0.1, 0, 0.9, 0]], dtype=np.float32)
    #     / 2
    # )
    #
    # d = 2
    # W0 = np.array(
    #     [[0, 1], [0.9, 0]], dtype=np.float32
    # ) / 2

    # W0 = torch.tensor(
    #     [
    #         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0.],
    #     ]
    # )/5

    gradient_methods = [
        # PowerIterationTorch(
        #     W0,
        #     regularization_to_original=1e-3,
        #     identity_penalty=0,
        #     name_suffix="no_identity_penalty",
        # ),
        # PowerIterationTorch(W0, regularization_to_anchor=1e-3, identity_penalty=10, anchor_matrix=anchor),
        # PowerIterationTorch(
        #     W0,
        #     regularization_to_anchor=1e-3,
        #     anchor_matrix=anchor,
        #     identity_penalty=10,
        #     reset_cache_every=1,
        #     name_suffix="reset_cache_every_1",
        # ),
        PowerIterationTorch(
            W0,
            regularization_to_anchor=1e-3,
            anchor_matrix=anchor,
            identity_penalty=10,
            n_iterations_cache=5,
            name_suffix="n_iterations_cache_5",
        ),
        PowerIterationTorch(
            W0,
            regularization_to_anchor=1e-3,
            anchor_matrix=anchor,
            identity_penalty=10,
            n_iterations_cache=1,
            name_suffix="n_iterations_cache_1",
        ),
        PowerIterationTorch(
            W0,
            regularization_to_anchor=1e-3,
            anchor_matrix=anchor,
            identity_penalty=10,
            reset_cache_every=1000000,
            name_suffix="never_reset",
        ),
        # ExponentialGradientTorch(W0, normalization=np.inf),
        # LogGradientTorch(W0, normalization=np.inf),
        # LogGradientTorch(W0),
        # InverseGradientTorch(W0, normalization=np.inf),
        # PowerIterationSCCMask(W0, update_scc_freq=500),
        # PowerIterationSCCMask(
        #     W0, update_scc_freq=500, regularization_to_original=1e-3, name_suffix="l2"
        # ),
    ]
    gradient_methods = [
        PowerIterationTorch(
            W0,
            anchor_matrix=anchor,
            regularization_to_anchor=1e-3,
            identity_penalty=0,
            name_suffix="no_identity_penalty",
        ),
        PowerIterationTorch(
            W0, regularization_to_anchor=1e-3, identity_penalty=10, anchor_matrix=anchor
        ),
        ExponentialGradientTorch(
            W0,
            anchor_matrix=anchor,
            regularization_to_anchor=1e-3,
        ),
        LogGradientTorch(
            W0,
            normalization=np.inf,
            anchor_matrix=anchor,
            regularization_to_anchor=1e-3,
        ),
        InverseGradientTorch(
            W0,
            normalization=np.inf,
            anchor_matrix=anchor,
            regularization_to_anchor=1e-3,
        ),
    ]
    for gradient_method in gradient_methods:
        # gradient_method.train(600, 10, lr=1e-3)
        gradient_method.train_until_dag(
            test_n_epochs=100, lr=1e-2, max_epochs=20_000, max_time=2 * 60
        )
    gm = gradient_methods[0]
    em = gm.compute_eigen_metrics()
    plot_metrics(gradient_methods)

    # plot eigenvalues of each method on the same figure, one subplot per method
    if d <= 20:
        fig, axes = plt.subplots(len(gradient_methods), 1, figsize=(3, 3 * len(gradient_methods)))
        for i, gradient_method in enumerate(gradient_methods):
            em = gradient_method.compute_eigen_metrics()
            eigenvalues = np.array(
                [metric["eigen"]["eigenvalues"] for metric in gradient_method.metrics]
            )
            colors = sns.color_palette(n_colors=d)
            ax = axes[i]
            for j in range(d):
                ax.scatter(eigenvalues.real[:, j], eigenvalues.imag[:, j], s=1, color=colors[j])
                ax.scatter(
                    eigenvalues.real[::10, j], eigenvalues.imag[::10, j], s=10, color=colors[j]
                )
            ax.set_title(gradient_method.name())
        plt.show()
    # plot matrix of each method on the same figure, one subplot per method
    fig, axes = plt.subplots(len(gradient_methods), 1, figsize=(3, 3 * len(gradient_methods)))
    for i, gradient_method in enumerate(gradient_methods):
        ax = axes[i]
        ax.set_title(gradient_method.name())
        sns.heatmap(to_numpy(gradient_method.get_matrix()), ax=ax)
    plt.show()
    print(gradient_methods[0].get_matrix())

    # gather cycles metrics
    cycle_weights = pd.DataFrame(
        [metric["cycles_weights"] for metric in gradient_methods[0].metrics]
    )
    cycle_weights["epoch"] = range(len(cycle_weights))
    # melt
    cycle_weights = cycle_weights.melt(id_vars=["epoch"], value_name="weight", var_name="cycle")
    # add cycle length
    cycle_weights["cycle_length"] = cycle_weights["cycle"]  # .apply(lambda x: len(x))

    # plot
    sns.lineplot(data=cycle_weights, x="epoch", y="weight", hue="cycle_length", style="cycle")
    plt.show()

# add A.T
