"""Example of running SDCI on synthetic data."""
import sys

sys.path.append("./")
import wandb

from sdci.models import SDCI
from sdci.utils import create_intervention_dataset

from deprecated.full_pipeline_main import generate_dataset


seed = 0
n = 40
d = 100
frac_interventions = 1.0
n_edges_per_d = 5
X_df, B_true, wandb_config_dict = generate_dataset(
    n, d, seed, frac_interventions, n_edges_per_d=n_edges_per_d
)
print("Dataset generated")
dataset = create_intervention_dataset(X_df, regime_format=True)
model = SDCI()
model.train(
    dataset,
    log_wandb=True,
    wandb_project="Test-SDCI",
    wandb_config_dict=wandb_config_dict,
    stage1_kwargs={
        "n_epochs": 1000,
        # "n_epochs_check": 10,
    },
    stage2_kwargs={
        "n_epochs": 1000,
        # "n_epochs_check": 10,
    },
    B_true=B_true,
)
# with more d: more independent weights? stronger regularization?
metrics_dict = model.compute_metrics(B_true)
metrics_dict["train_time"] = model._train_runtime_in_sec

wandb.log(metrics_dict)
wandb.finish()
