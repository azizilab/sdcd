import time
from typing import Optional

import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from third_party.dcdfg import MLPModuleGaussianModel
from third_party.callback import (
    AugLagrangianCallback,
    ConditionalEarlyStopping,
    CustomProgressBar,
)

from .base._base_model import BaseModel

_DEFAULT_MODEL_KWARGS = dict(
    num_layers=2,
    num_modules=20,
    hid_dim=16,
    lr_init=1e-3,
    reg_coeff=0.1,
    constraint_mode="exp",
)


class DCDFG(BaseModel):
    def __init__(self):
        super().__init__()
        self._adj_matrix = None
        self._adj_matrix_thresh = None

    def train(
        self,
        dataset: Dataset,
        log_wandb: bool = False,
        wandb_project: str = "DCDFG",
        wandb_config_dict: Optional[dict] = None,
        val_fraction: float = 0.2,
        **model_kwargs,
    ):
        if log_wandb:
            wandb_config_dict = wandb_config_dict or {}
            wandb.init(
                project=wandb_project,
                name="DCDFG",
                config=wandb_config_dict,
            )
        train_dataset, val_dataset = random_split(
            dataset,
            [
                len(dataset) - int(val_fraction * len(dataset)),
                int(val_fraction * len(dataset)),
            ],
        )
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        sample_batch = next(iter(dataloader))
        assert len(sample_batch) == 3, "Dataset should contain (X, masks, regimes)"
        d = sample_batch[0].shape[1]

        start = time.time()
        self._model_kwargs = {**_DEFAULT_MODEL_KWARGS.copy(), **model_kwargs}
        self._model = MLPModuleGaussianModel(
            d,
            **self._model_kwargs,
        )

        early_stop_1_callback = ConditionalEarlyStopping(
            monitor="Val/aug_lagrangian",
            min_delta=1e-4,
            patience=5,
            verbose=True,
            mode="min",
        )
        trainer = pl.Trainer(
            max_epochs=60000,
            logger=WandbLogger(project=wandb_project, reinit=True)
            if log_wandb
            else False,
            val_check_interval=1.0,
            callbacks=[
                AugLagrangianCallback(),
                early_stop_1_callback,
                CustomProgressBar(),
            ],
        )
        trainer.fit(
            self._model,
            DataLoader(train_dataset, batch_size=128, num_workers=4),
            DataLoader(val_dataset, num_workers=8, batch_size=256),
        )

        self._train_runtime_in_sec = time.time() - start
        print(f"Finished training in {self._train_runtime_in_sec} seconds.")

        # Save unthresholded matrix because thresholding is destructive.
        self._adj_matrix = self._model.module.get_w_adj().detach().cpu().numpy()
        self._model.module.threshold()
        self._adj_matrix_thresh = np.array(
            self._model.module.weight_mask.detach().cpu().numpy() > 0, dtype=int
        )

    def get_adjacency_matrix(self, threshold: bool = True) -> np.ndarray:
        assert self._model is not None, "Model has not been trained yet."

        return self._adj_matrix_thresh if threshold else self._adj_matrix
