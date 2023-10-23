import numpy as np
from torch.utils.data import Dataset

from ..simulated_data import random_model_gaussian_global_variance
from ..utils import create_intervention_dataset


def generate_test_dataset(
    n,
    d,
    use_interventions=True,
) -> Dataset:
    n_edges = d // 2
    n_interventions = d if use_interventions else 0

    true_causal_model = random_model_gaussian_global_variance(
        d,
        n_edges,
    )
    interventions_names = np.random.choice(
        list(true_causal_model.interventions.keys()), n_interventions, replace=False
    )
    X_df = true_causal_model.generate_dataframe_from_all_distributions(
        n_samples_control=n * (d + 1),
        n_samples_per_intervention=n,
        subset_interventions=interventions_names,
    )

    return create_intervention_dataset(X_df, regime_format=True)
