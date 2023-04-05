import random

import numpy as np
import torch


def set_random_seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)

def get_leading_left_and_right_eigenvectors(A):
    """Get the leading left and right eigenvectors of a matrix.
    Args:
        A (np.ndarray or torch.Tensor): A square matrix.
    Returns:
        (np.ndarray, np.ndarray): The leading left and right eigenvectors.
    """
    assert A.shape[0] == A.shape[1]
    if isinstance(A, torch.Tensor):
        A = A.detach().numpy()
    # right eigenvector
    w, v = np.linalg.eig(A)
    idx = np.argmax(np.abs(w))
    right_eigenvector = v[:, idx]
    # left eigenvector
    w, v = np.linalg.eig(A.T)
    idx = np.argmax(np.abs(w))
    left_eigenvector = v[:, idx]
    return left_eigenvector, right_eigenvector
