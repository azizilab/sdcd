from abc import abstractmethod
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from ...utils import compute_metrics, set_random_seed_all


class BaseModel:
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
        raise NotImplementedError

    @abstractmethod
    def get_adjacency_matrix(self, threshold: bool = True) -> np.ndarray:
        raise NotImplementedError

    def compute_metrics(self, ground_truth_adjacency: np.ndarray) -> dict:
        return compute_metrics(
            self.get_adjacency_matrix(threshold=True), ground_truth_adjacency
        )

    def compute_nll(self, dataset: Dataset) -> float:
        raise NotImplementedError
