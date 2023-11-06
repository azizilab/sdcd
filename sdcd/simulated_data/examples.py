import torch

from sdcd.causal_model import CausalModel, MarginalDistribution, scale_mechanism
from sdcd.simulated_data.graph import chain_graph, random_dag, random_diagonal_band_dag
from sdcd.simulated_data.mechanisms import generate_gaussian_mlp_fixed_scale_mechanisms


def random_model_gaussian_global_variance(
    n_nodes,
    n_edges,
    knockdown=0.1,
    hard=False,
    scale=1.0,
    scale_hard=0.1,
    dag_type="ER",
    **kwargs,
):
    if dag_type == "ER":
        dag = random_dag(n_nodes, n_edges)
    elif dag_type == "chain":
        dag = chain_graph(n_nodes)
    elif dag_type == "diag_band":
        dag = random_diagonal_band_dag(n_nodes, n_edges, **kwargs)
    else:
        raise ValueError(f"Unknown dag_type: {dag_type}")

    causal_model = CausalModel(dag)
    observational_mechanisms = generate_gaussian_mlp_fixed_scale_mechanisms(
        causal_model, [100], scale=scale, activation="sigmoid", bias=False
    )
    causal_model.set_causal_mechanisms(observational_mechanisms)

    for i in range(n_nodes):
        nodes = [i]
        if hard:
            # ignore knockdown, and set the mechanism to a marginal distribution
            new_intervened_mechanisms = {
                n: MarginalDistribution(torch.distributions.Normal(0, scale_hard))
                for n in nodes
            }
        else:
            new_intervened_mechanisms = {
                n: scale_mechanism(observational_mechanisms[n], knockdown)
                for n in nodes
            }
        causal_model.set_intervention(i, new_intervened_mechanisms)

    return causal_model
