import numpy as np
import igraph as ig
import sys
from sdcd.models._sdcd import SDCD
from sdcd.simulated_data.examples import random_model_gaussian_global_variance
from sdcd.utils.train_utils import create_intervention_dataset
import matplotlib.pyplot as plt


def format_ratio(ratio):
    return 0.5 if ratio == 0.5 else int(ratio)


def is_dag(W: np.ndarray) -> bool:
    """
    Returns ``True`` if ``W`` is a DAG, ``False`` otherwise.
    """
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def count_accuracy(B_true: np.ndarray, B_est: np.ndarray) -> dict:
    r"""
    Compute various accuracy metrics for B_est.

    | true positive = predicted association exists in condition in correct direction
    | reverse = predicted association exists in condition in opposite direction
    | false positive = predicted association does not exist in condition

    Parameters
    ----------
    B_true : np.ndarray
        :math:`[d, d]` ground truth graph, :math:`\{0, 1\}`.
    B_est : np.ndarray
        :math:`[d, d]` estimate, :math:`\{0, 1, -1\}`, -1 is undirected edge in CPDAG.

    Returns
    -------
    dict
        | fdr: (reverse + false positive) / prediction positive
        | tpr: (true positive) / condition positive
        | fpr: (reverse + false positive) / condition negative
        | shd: undirected extra + undirected missing + reverse
        | nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError("B_est should take value in {0,1,-1}")
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError("undirected edge should only appear once")
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError("B_est should take value in {0,1}")
        if not is_dag(B_est):
            raise ValueError("B_est should be a DAG")
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {"fdr": fdr, "tpr": tpr, "fpr": fpr, "shd": shd, "nnz": pred_size}


def sdcd_ev(n, d, n_edges, n_per_intervention):
    true_causal_model = random_model_gaussian_global_variance(
        d,
        n_edges,
        dag_type="ER",
        scale=0.5,
        hard=True,
    )
    X_df = true_causal_model.generate_dataframe_from_all_distributions(
        n_samples_control=n,
        n_samples_per_intervention=n_per_intervention,
    )
    X_df.iloc[:, :-1] = (X_df.iloc[:, :-1] - X_df.iloc[:, :-1].mean()) / X_df.iloc[
        :, :-1
    ].std()
    X_dataset = create_intervention_dataset(
        X_df, perturbation_colname="perturbation_label"
    )
    model = SDCD()
    model.train(X_dataset, finetune=True)
    adj_matrix = model.get_adjacency_matrix(threshold=True)
    acc = count_accuracy(true_causal_model.adjacency, adj_matrix)
    return acc


def run_one_experiment(trials, n, s0_ratio):
    noise_type = "gauss"
    error_var = "eq"
    num_nodes = [5, 10, 50, 100] if s0_ratio <= 2 else [10, 50, 100]
    methods = ["SDCD-orig"]
    shd_results = {method: {d: [] for d in num_nodes} for method in methods}

    for d in num_nodes:
        num_edges = int(s0_ratio * d)

        for i in range(trials):
            print(f"Running trial {i} for {d} nodes")
            try:
                result = sdcd_ev(n, d, num_edges, d)
                shd_results["SDCD-orig"][d].append(result["shd"] / d)

            except Exception as e:
                print(e)
                print(
                    f"trial with {d} nodes and {noise_type} noise and s0_ratio {s0_ratio} skipped due to error"
                )

    make_one_plot(
        s0_ratio,
        noise_type,
        methods,
        num_nodes,
        trials,
        n,
        error_var,
        shd_results,
        "shd",
    )


def make_one_plot(
    s0_ratio, noise_type, methods, num_nodes, trials, n, error_var, results, metric: str
):
    plt.figure(figsize=(8, 6))

    for method in methods:
        means = [
            np.mean(results[method][d]) if results[method][d] else None
            for d in num_nodes
        ]
        plt.plot(num_nodes, means, marker="o", label=method)

    noise_names = {"gauss": "Gaussian", "exp": "Exponential", "gumbel": "Gumbel"}

    plt.title(
        f"{noise_names[noise_type]} Noise, ER{s0_ratio}\n(n={n}, trials={trials}, error_var={error_var})"
    )
    plt.xlabel("d (Number of Nodes)")
    plt.ylabel(f"Normalized {metric.upper()}")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        f"sdcd_{metric}_ER{format_ratio(s0_ratio)}_noise={noise_type}_n={n}_var={error_var}.png"
    )

    output_filename = f"sdcd_{metric}_ER{format_ratio(s0_ratio)}_noise={noise_type}_n={n}_var={error_var}.txt"
    with open(output_filename, "w") as f:
        f.write(f"method,d,mean_normalized_{metric}\n")
        for method in methods:
            for d in num_nodes:
                mean_metric = (
                    np.mean(results[method][d]) if results[method][d] else None
                )
                if mean_metric is not None:
                    f.write(f"{method},{d},{mean_metric}\n")


nTrials = int(sys.argv[1])
nSamples = int(sys.argv[2])
s0_ratio = float(sys.argv[3])
run_one_experiment(nTrials, nSamples, s0_ratio)
