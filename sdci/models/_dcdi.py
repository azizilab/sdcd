import time
from typing import Optional

import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from ..third_party.dcdi import MLPGaussianModel
from ..third_party.callback import (
    AugLagrangianCallback,
    ConditionalEarlyStopping,
    CustomProgressBar,
)

from .base._base_model import BaseModel
from ..utils import set_random_seed_all

_DEFAULT_MODEL_KWARGS = dict(
    num_layers=2,
    hid_dim=16,
    lr_init=1e-3,
    reg_coeff=0.1,
    constraint_mode="exp",
    max_epochs=60000,
)


class DCDI(BaseModel):
    def __init__(self):
        super().__init__()
        self._adj_matrix = None
        self._adj_matrix_thresh = None
        self._trained = False

    def train(
        self,
        dataset: Dataset,
        log_wandb: bool = False,
        wandb_project: str = "DCDI",
        wandb_config_dict: Optional[dict] = None,
        val_fraction: float = 0.2,
        finetune: bool = False,
        **model_kwargs,
    ):
        set_random_seed_all(0)
        if log_wandb:
            wandb_config_dict = wandb_config_dict or {}
            wandb.init(
                project=wandb_project,
                name="DCDI",
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
        init_kwargs = self._model_kwargs.copy()
        max_epochs = init_kwargs.pop("max_epochs")
        self._model = MLPGaussianModel(
            d,
            **init_kwargs,
        )

        early_stop_1_callback = ConditionalEarlyStopping(
            monitor="Val/aug_lagrangian",
            min_delta=1e-4,
            patience=5,
            verbose=True,
            mode="min",
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs,
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

        # freeze and prune adjacency
        # Save unthresholded matrix because thresholding is destructive.
        self._adj_matrix = self._model.module.get_w_adj().detach().cpu().numpy()
        self._model.module.threshold()

        if finetune:
            # WE NEED THIS BECAUSE IF it's exactly a DAG THE POWER ITERATIONS DOESN'T CONVERGE
            # TODO Just refactor and remove constraint at validation time
            self._model.module.constraint_mode = "exp"
            # remove dag constraints: we have a prediction problem now!
            self._model.gamma = 0.0
            self._model.mu = 0.0

            # Step 2:fine tune weights with frozen model
            # LOG CONFIG
            early_stop_2_callback = EarlyStopping(
                monitor="Val/nll", min_delta=1e-6, patience=5, verbose=True, mode="min"
            )
            self.trainer_fine = pl.Trainer(
                max_epochs=max_epochs,
                logger=WandbLogger(project=wandb_project, reinit=True),
                val_check_interval=1.0,
                callbacks=[early_stop_2_callback, CustomProgressBar()],
            )
            self.trainer_fine.fit(
                self._model,
                DataLoader(train_dataset, batch_size=128),
                DataLoader(val_dataset, num_workers=2, batch_size=256),
            )

        self._train_runtime_in_sec = time.time() - start
        print(f"Finished training in {self._train_runtime_in_sec} seconds.")

        # Save unthresholded matrix because thresholding is destructive.
        self._adj_matrix = self._model.module.get_w_adj().detach().cpu().numpy()
        self._model.module.threshold()
        self._adj_matrix_thresh = np.array(
            self._model.module.adjacency.detach().cpu().numpy() > 0, dtype=int
        )

        self._trained = True

    def get_adjacency_matrix(self, threshold: bool = True) -> np.ndarray:
        assert self._model is not None, "Model has not been trained yet."

        return self._adj_matrix_thresh if threshold else self._adj_matrix

    def compute_nll(self, dataset: Dataset) -> float:
        assert self._trained
        pred = self.trainer_fine.predict(
            ckpt_path="best",
            dataloaders=DataLoader(dataset, num_workers=8, batch_size=256),
        )
        return np.mean([x.item() for x in pred])
