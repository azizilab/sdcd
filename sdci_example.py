"""Example of running SDCI on synthetic data."""
import wandb

from full_pipeline_main import generate_dataset
from models import SDCI
from train_utils import create_intervention_dataset


seed = 0
n = 40
d = 500
frac_interventions = 1.0
n_edges_per_d = 5
X_df, B_true, wandb_config_dict = generate_dataset(
    n, d, seed, frac_interventions, n_edges_per_d=n_edges_per_d
)
print("Dataset generated")
dataset = create_intervention_dataset(X_df, regime_format=False)


# sweep over hyperparameters:
# - n_epochs: 1000
# - learning rate: 1e-3, 2e-3, 5e-3, 1e-2
# - batch size: 512
# - beta: 1e-3, 1e-2
# - stage 1:
#   - alpha: 1e-2, 1e-1
# - stage 2:
#   - alpha: 1e-3, 1e-2, 1e-1
#   - freeze_gamma_at_dag: True, False

learning_rates = [5e-3]
betas = [1e-3, 1e-2, 1e-4]
alphas_stage1 = [1e-3, 1e-2, 1e-1, 1]
alphas_stage2 = [1e-3, 1e-2, 1e-4]
freeze_gamma_at_dag = [True]

for lr in learning_rates:
    for beta in betas:
        for a1 in alphas_stage1:
            for a2 in alphas_stage2:
                for f in freeze_gamma_at_dag:
                    model = SDCI()
                    model.train(
                        dataset,
                        log_wandb=True,
                        wandb_project="Test-SDCI-500",
                        wandb_config_dict=wandb_config_dict,
                        stage1_kwargs={
                            "n_epochs": 1500,
                            "learning_rate": lr,
                            "alpha": a1,
                            "beta": beta,
                        },
                        stage2_kwargs={
                            "n_epochs": 1500,
                            "learning_rate": lr,
                            "alpha": a2,
                            "beta": beta,
                            "freeze_gamma_at_dag": f,
                        },
                        B_true=B_true,
                        wandb_name=f"lr={lr:1.0e}|beta={beta:1.0e}|a1={a1:1.0e}|a2={a2:1.0e}",
                    )
                    # with more d: more independent weights? stronger regularization?
                    metrics_dict = model.compute_metrics(B_true)
                    metrics_dict["train_time"] = model._train_runtime_in_sec

                    wandb.log(metrics_dict)
                    wandb.finish()



