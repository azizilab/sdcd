import time
from typing import Optional

import numpy as np
from torch.utils.data import Dataset
import wandb

from .base._base_model import BaseModel
from ..utils import compute_min_dag_threshold


_DEFAULT_MODEL_KWARGS = dict()  # dict(w_threshold=0.05)


class NOBEARS(BaseModel):
    def __init__(self):
        super().__init__()
        self._adj_matrix = None
        self._adj_matrix_thresh = None

    def train(
        self,
        dataset: Dataset,
        log_wandb: bool = False,
        wandb_project: str = "NOBEARS",
        wandb_config_dict: Optional[dict] = None,
        **model_kwargs,
    ):
        try:
            import tensorflow as tf
            from ..third_party.nobears import NoBearsTF, W_reg_init
        except ImportError as e:
            raise ImportError(
                "You must install the 'benchmark' extra to use this class. Run `pip install sdcd[benchmark]`"
            ) from e

        assert len(dataset.tensors) == 3, "Dataset must be in regime format"
        assert not dataset.tensors[2].any(), "Dataset must be fully observational"

        if log_wandb:
            wandb_config_dict = wandb_config_dict or {}
            wandb.init(
                project=wandb_project,
                name="NOBEARS",
                config=wandb_config_dict,
            )
        data = dataset.tensors[0].numpy()

        self._model_kwargs = {**_DEFAULT_MODEL_KWARGS.copy(), **model_kwargs}
        init_kwargs = self._model_kwargs.copy()
        start = time.time()

        self._W_init = W_reg_init(data).astype("float32")
        with tf.device("/gpu:0"):
            tf.compat.v1.reset_default_graph()

            self._model = NoBearsTF(**init_kwargs)
            self._model.construct_graph(data, self._W_init)

        sess = tf.compat.v1.Session()
        sess.run(self._model.graph_nodes["init_vars"])
        self._model.model_init_train(sess)

        self._model.model_train(sess)

        self._adj_matrix = np.abs(sess.run(self._model.graph_nodes["weight_ema"]))
        self._train_runtime_in_sec = time.time() - start
        print(f"Finished training in {self._train_runtime_in_sec} seconds.")

        w_threshold = compute_min_dag_threshold(self._adj_matrix)
        wandb.log({"w_threshold": w_threshold})
        self._adj_matrix_thresh = np.array(
            np.abs(self._adj_matrix) > w_threshold, dtype=int
        )

    def get_adjacency_matrix(self, threshold: bool = True) -> np.ndarray:
        assert self._model is not None, "Model has not been trained yet."
        return self._adj_matrix_thresh if threshold else self._adj_matrix
