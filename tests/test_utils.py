import pytest

import numpy as np
import scipy as sp
from anndata import AnnData

from sdcd.utils import create_intervention_dataset, create_intervention_dataset_anndata
from sdcd.simulated_data import random_model_gaussian_global_variance


def test_create_intervention_dataset():
    n = 10
    d = 5
    n_edges = d // 2
    n_interventions = d

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

    create_intervention_dataset(X_df, regime_format=True)


def test_create_intervention_dataset_anndata():
    n = 10
    d = 5
    n_edges = d // 2
    n_interventions = d

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

    adata = AnnData(
        X_df.iloc[:, :-1].values, obs={"perturbation_label_test": X_df.iloc[:, -1]}
    )
    create_intervention_dataset_anndata(
        adata, perturbation_obsname="perturbation_label_test", regime_format=True
    )

    # test with layer
    adata.layers["normalized"] = adata.X
    create_intervention_dataset_anndata(
        adata,
        layer="normalized",
        perturbation_obsname="perturbation_label_test",
        regime_format=True,
    )

    # test with sparse X
    adata.X = sp.sparse.csr_matrix(adata.X)
    create_intervention_dataset_anndata(
        adata, perturbation_obsname="perturbation_label_test", regime_format=True
    )
