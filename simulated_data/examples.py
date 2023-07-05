from causal_model import CausalModel, scale_mechanism
from simulated_data.graph import random_dag
from simulated_data.mechanisms import generate_gaussian_mlp_fixed_scale_mechanisms


def random_model_gaussian_global_variance(n_nodes, n_edges, knockdown=0.1, scale=1.0):
    dag = random_dag(n_nodes, n_edges)
    causal_model = CausalModel(dag)
    observational_mechanisms = generate_gaussian_mlp_fixed_scale_mechanisms(
        causal_model, [100], scale=scale, activation="sigmoid"
    )
    causal_model.set_causal_mechanisms(observational_mechanisms)

    for i in range(n_nodes):
        nodes = [i]
        new_intervened_mechanisms = {
            n: scale_mechanism(observational_mechanisms[n], knockdown) for n in nodes
        }
        causal_model.set_intervention(i, new_intervened_mechanisms)

    return causal_model
