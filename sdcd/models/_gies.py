import time
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

import wandb

from .base._base_model import BaseModel

_DEFAULT_MODEL_KWARGS = dict()


class GIES(BaseModel):
    def __init__(self):
        super().__init__()
        self._adj_matrix = None

    def train(
        self,
        dataset: Dataset,
        log_wandb: bool = False,
        wandb_project: str = "NOTEARS",
        wandb_config_dict: Optional[dict] = None,
        **model_kwargs,
    ):
        try:
            import gies
        except ImportError as e:
            raise ImportError(
                "You must install the 'benchmark' extra to use this class. Run `pip install sdcd[benchmark]`"
            ) from e
        gies.np.bool = bool  # bug in gies with newer np version

        assert len(dataset.tensors) == 3, "Dataset must be in regime format"

        if log_wandb:
            wandb_config_dict = wandb_config_dict or {}
            wandb.init(
                project=wandb_project,
                name="GIES",
                config=wandb_config_dict,
            )
        data = dataset.tensors[0].numpy()
        intervention_mask = dataset.tensors[1].numpy()
        intervention_strings = np.array(
            ["".join(map(str, row)) for row in intervention_mask]
        )

        data_envs = []
        intervention_list = []
        for intervention_id in list(set(intervention_strings)):
            intervention_idxs = np.where(intervention_strings == intervention_id)[0]
            data_envs.append(data[intervention_idxs])
            intervention_list.append(
                list(np.where(1 - intervention_mask[intervention_idxs[0]])[0])
            )

        start = time.time()
        self._model_kwargs = {**_DEFAULT_MODEL_KWARGS.copy(), **model_kwargs}
        self._model = -1
        self._adj_matrix, score = gies.fit_bic(data_envs, intervention_list)
        print(f"GIES score: {score}")
        self._train_runtime_in_sec = time.time() - start
        print(f"Finished training in {self._train_runtime_in_sec} seconds.")

    def get_adjacency_matrix(self, threshold: bool = True) -> np.ndarray:
        assert self._model is not None, "Model has not been trained yet."
        return self._adj_matrix
