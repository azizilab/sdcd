from cdt.metrics import SHD
from sklearn import metrics


def fdr(pred, target):
    return 1 - metrics.precision_score(target.flatten(), pred.flatten())


def shd_metric(pred, target):
    """
    Calculates the structural hamming distance.

    Parameters:
    -----------
    pred: nx.DiGraph or ndarray
        The predicted adjacency matrix
    target: nx.DiGraph or ndarray
        The true adjacency matrix

    Returns:
    --------
    shd
    """
    return SHD(target, pred, double_for_anticausal=False)
