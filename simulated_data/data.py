import pandas as pd

from causal_model.model import CausalModel


def generate_data(causal_graph: CausalModel, n_samples_control, n_samples_interventions: dict):
    """Generate data from a causal graph.
    Return a pandas DataFrame with the data as columns and a column "perturbation_label" that indicates the perturbation
    applied to each sample.
    """

    # Generate data from the control distribution
    X_control = causal_graph.sample_from_observational_distribution(n_samples_control)
    X_control["perturbation_label"] = "control"
    X = pd.DataFrame(X_control)

    # Generate data from the interventional distributions
    for intervention_name, n_samples in n_samples_interventions.items():
        X_intervention = causal_graph.sample_from_interventional_distribution(
            intervention_name, n_samples
        )
        X_intervention["perturbation_label"] = intervention_name
        X = X.append(X_intervention, ignore_index=True)

    return X
