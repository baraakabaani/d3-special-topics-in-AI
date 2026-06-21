"""
AutoML search over HybridRetriever hyperparameters using Optuna.
done by Baraa

D2 fixes vs D1:
- k search space narrowed to [1,5] to match NDCG@5/Recall@5 evaluation
- Two samplers compared: TPESampler (Bayesian) vs RandomSampler (baseline)
- Best sampler selected by cross-validated NDCG@5
"""

import time
import warnings

import numpy as np
import optuna
import yaml
from sklearn.model_selection import KFold

from src.retriever import HybridRetriever
from src.evaluation import ndcg_at_k

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


def _objective(trial, corpus: list, queries: list, qrels: dict) -> float:
    """Optuna objective: 5-fold CV NDCG@5, zero if p95 latency > 100 ms."""
    k         = trial.suggest_int("k",       1, 5)          # top-5 max
    metric    = trial.suggest_categorical("metric", ["cosine", "euclidean", "dot_product"])
    svd_dim   = trial.suggest_int("svd_dim", 32, 512, log=True)
    norm      = trial.suggest_categorical("norm", ["l2", "none", "minmax"])
    alpha     = trial.suggest_float("alpha", 0.0, 1.0)

    kf     = KFold(n_splits=5, shuffle=True, random_state=42)
    q_list = list(queries)
    scores = []

    for train_idx, val_idx in kf.split(q_list):
        train_queries = [q_list[i] for i in train_idx]
        val_queries   = [q_list[i] for i in val_idx]

        retriever = HybridRetriever(
            k=k, metric=metric, svd_dim=svd_dim,
            normalization=norm, hybrid_weight=alpha,
        )
        retriever.fit(corpus)

        # latency check on val fold
        lats = []
        for q in val_queries:
            t0 = time.perf_counter()
            retriever.retrieve(q)
            lats.append((time.perf_counter() - t0) * 1000)
        if np.percentile(lats, 95) > 100:
            return 0.0

        fold_scores = [
            ndcg_at_k(retriever.retrieve(q), qrels.get(q, []))
            for q in val_queries
        ]
        scores.append(np.mean(fold_scores))

        trial.report(np.mean(scores), step=len(scores))
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def run_study(corpus: list, queries: list, qrels: dict,
              n_trials: int = 100) -> tuple:
    """
    Run Optuna with two samplers and return (best_params, study_tpe, study_random).
    Compares TPESampler (Bayesian, guided) vs RandomSampler (unguided baseline).
    """
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3)

    # --- TPE (Bayesian) ---
    study_tpe = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=pruner,
    )
    study_tpe.optimize(
        lambda t: _objective(t, corpus, queries, qrels),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    # --- Random (baseline) ---
    study_rand = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=42),
        pruner=pruner,
    )
    study_rand.optimize(
        lambda t: _objective(t, corpus, queries, qrels),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    # pick the sampler that found the higher NDCG@5
    if study_tpe.best_value >= study_rand.best_value:
        best_params  = study_tpe.best_params
        best_sampler = "TPE"
        best_value   = study_tpe.best_value
    else:
        best_params  = study_rand.best_params
        best_sampler = "Random"
        best_value   = study_rand.best_value

    print(f"\nSampler comparison (n_trials={n_trials} each):")
    print(f"  TPE    best NDCG@5 = {study_tpe.best_value:.4f}")
    print(f"  Random best NDCG@5 = {study_rand.best_value:.4f}")
    print(f"  Winner: {best_sampler}  NDCG@5={best_value:.4f}  params={best_params}")

    return best_params, study_tpe, study_rand


def export_run_card(params: dict, ndcg: float, recall: float,
                    out_path: str = "configs/run_card.yaml") -> None:
    card = {
        "component": "automl_hybrid_retriever",
        "sampler":   "TPE vs Random (best selected)",
        "pruner":    "MedianPruner(n_startup=10, n_warmup=3)",
        "cv_folds":  5,
        "best_params": {
            "k":       int(params["k"]),
            "metric":  params["metric"],
            "svd_dim": int(params["svd_dim"]),
            "norm":    params["norm"],
            "alpha":   round(float(params["alpha"]), 4),
        },
        "results": {
            "ndcg_at_5":   round(ndcg,   4),
            "recall_at_5": round(recall, 4),
        },
    }
    with open(out_path, "w") as f:
        yaml.dump(card, f, default_flow_style=False, sort_keys=False)
    print(f"Saved run card -> {out_path}")
