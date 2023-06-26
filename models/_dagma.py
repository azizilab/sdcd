import time
from typing import Optional

import numpy as np
from torch.utils.data import Dataset
import wandb

from third_party.dagma import DagmaNonlinear, DagmaMLP

from .base._base_model import BaseModel

_DEFAULT_MODEL_KWARGS = dict(
    num_layers=2,
    num_modules=20,
    hid_dim=16,
    lambda1=0.02,
    lambda2=0.005,
    threshold=0.3,
)


class DAGMA(BaseModel):
    def __init__(self):
        super().__init__()
        self._adj_matrix = None
        self._adj_matrix_thresh = None

    def train(
        self,
        dataset: Dataset,
        log_wandb: bool = False,
        wandb_project: str = "DAGMA",
        wandb_config_dict: Optional[dict] = None,
        val_fraction: float = 0.2,
        **model_kwargs,
    ):
        if log_wandb:
            wandb_config_dict = wandb_config_dict or {}
            wandb.init(
                project=wandb_project,
                name="DAGMA",
                config=wandb_config_dict,
            )
        data = dataset.tensors[0]
        d = data.shape[1]

        start = time.time()
        self._model_kwargs = {**_DEFAULT_MODEL_KWARGS.copy(), **model_kwargs}
        eq_model = DagmaMLP(dims=[d, 10, 1], bias=True)
        self._model = DagmaNonlinear(eq_model)

        self._model.fit(
            data,
            lambda1=self._model_kwargs["lambda1"],
            lambda2=self._model_kwargs["lambda2"],
            w_threshold=self._model_kwargs["threshold"],
            log_wandb=log_wandb,
        )
        self._train_runtime_in_sec = time.time() - start
        print(f"Finished training in {self._train_runtime_in_sec} seconds.")

        self._adj_matrix = self._model.model.fc1_to_adj()
        self._adj_matrix_thresh = np.array(
            self._adj_matrix > self._model_kwargs["threshold"], dtype=int
        )

    def get_adjacency_matrix(self, threshold: bool = True) -> np.ndarray:
        assert self._model is not None, "Model has not been trained yet."
        return self._adj_matrix_thresh if threshold else self._adj_matrix
