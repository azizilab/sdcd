from abc import abstractmethod
from typing import Optional

import numpy as np
from torch.data import Dataset

from train_utils import (
    compute_metrics,
)


class BaseModel:
    def __init__(self):
        self._model = None

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
    def get_adjacency_pred(self, threshold: bool = True) -> np.ndarray:
        raise NotImplementedError

    def compute_ground_truth_metrics(self, ground_truth_adjacency: np.ndarray) -> dict:
        return compute_metrics(
            self.get_adjacency_pred(threshold=True), ground_truth_adjacency
        )
