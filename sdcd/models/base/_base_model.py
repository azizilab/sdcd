from abc import abstractmethod
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from ...utils import compute_metrics, set_random_seed_all


class BaseModel:
    """
    Base model class. This class is abstract and should not be instantiated directly.
    It provides the basic structure for a model, including methods for training,
    getting the adjacency matrix, computing metrics and computing negative log likelihood.
    """

    def __init__(self):
        self._model = None
        self._model_kwargs = None
        self._trained = False
        set_random_seed_all(0)

    @abstractmethod
    def train(
        self,
        dataset: Dataset,
        log_wandb: bool = False,
        wandb_project: str = "undefined",
        wandb_config_dict: Optional[dict] = None,
        **kwargs,
    ):
        """
        Train the model.

        Parameters:
        dataset (Dataset): The dataset for training the model.
        log_wandb (bool, optional): If True, logs will be sent to Weights and Biases. Defaults to False.
        wandb_project (str, optional): The name of the Weights and Biases project. Defaults to "undefined".
        wandb_config_dict (dict, optional): A dictionary of configuration parameters for Weights and Biases. Defaults to None.
        **kwargs: Additional keyword arguments.

        Raises:
        NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def get_adjacency_matrix(self, threshold: bool = True) -> np.ndarray:
        """
        This method is used to get the adjacency matrix of the graph.

        Parameters:
        threshold (bool): If True, the adjacency matrix will be thresholded.
                           All values below the threshold will be set to 0,
                           and all values above the threshold will be set to 1.
                           If False, the weighted adjacency matrix will be returned.

        Returns:
        np.ndarray: The adjacency matrix of the graph.
        """
        raise NotImplementedError

    def compute_metrics(self, ground_truth_adjacency: np.ndarray) -> dict:
        """
        Compute the metrics for the model.

        Parameters:
        ground_truth_adjacency (np.ndarray): The ground truth adjacency matrix.

        Returns:
        dict: The computed metrics with respect to the thresholded adjacency matrix.
        """
        return compute_metrics(
            self.get_adjacency_matrix(threshold=True), ground_truth_adjacency
        )

    def compute_nll(self, dataset: Dataset) -> float:
        """
        Compute the negative log likelihood (NLL) for the model.

        Parameters:
        dataset (Dataset): The dataset for which to compute the NLL.

        Returns:
        float: The computed NLL.
        """
        raise NotImplementedError
