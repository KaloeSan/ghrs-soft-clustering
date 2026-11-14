"""Partition bound: the best strict RMSE any user partition can give under Algorithm 1.

See ghrs/bounds.py for why this limits every tuning of steps 1-5.

For each fold and K the script runs two searches:

    oracle  searches partitions directly on the test error (uses the test labels, so
            it only serves as a reference). It starts from several partitions (GHRS
            K-Means labels, random labels, K-Means on centred training ratings and
            user-mean quantiles), applies coordinate descent and, with
            ``--effort full``, simulated annealing and iterated local search.
    legit   the same search on a training-only objective (leave-one-rating-out
            error). Its test RMSE shows what a clustering based on ratings could reach.

Every final partition is scored again with ``ghrs.recommender.ClusterRatingModel``;
the script stops if the two results disagree.

Usage (from the repository root):
    python -m scripts.partition_bound                         # K=8, quick (1-2 min per fold)
    python -m scripts.partition_bound --effort full           # with annealing (about 10 min per fold)
    python -m scripts.partition_bound --k 2 4 8 16 32 --objective test

Results are written to outputs/bounds/partition_bound.csv.
"""

import ghrs  # noqa: F401 - sets KERAS_BACKEND before keras is imported

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ghrs.bounds import FoldStats, PartitionSearch
from ghrs.experiment import FOLDS_5CV, PAPER_TARGETS, ClusterConfig, Experiment, RunConfig
from ghrs.metrics import rmse
from ghrs.recommender import ClusterRatingModel, membership_matrix


def search(fs: FoldStats, inits: dict, K: int, objective: str, effort: str, chains: int, sa_sweeps: int,
           ils: int, seed: int) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    score = (lambda L: fs.rmse(L)) if objective == "test" else (lambda L: PartitionSearch(fs, L, K, "loro").F.sum())
    best_L, best, log = None, np.inf, {}
    for name, L0 in inits.items():
        s = PartitionSearch(fs, L0, K, objective)
        s.descend(rng)
        v = score(s.L)
        log[f"descend[{name}]"] = fs.rmse(s.L)
        if v < best:
            best, best_L = v, s.L.copy()
    if effort == "full":
        for ch in range(chains):
            L0 = best_L if ch == 0 else rng.integers(0, K, fs.NU)
            s = PartitionSearch(fs, L0, K, objective)
            s.anneal(rng, sa_sweeps)
            s.descend(rng)
            v = score(s.L)
            log[f"anneal[{ch}]"] = fs.rmse(s.L)
            if v < best:
                best, best_L = v, s.L.copy()
        for _ in range(ils):  # iterated local search: reassign 10% of the users at random, then descend
            L = best_L.copy()
            kick = rng.random(fs.NU) < 0.10
            L[kick] = rng.integers(0, K, kick.sum())
            s = PartitionSearch(fs, L, K, objective)
            s.descend(rng)
            v = score(s.L)
            if v < best:
                best, best_L = v, s.L.copy()
        log["after_ils"] = fs.rmse(best_L)
    return best_L, log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folds", nargs="+", default=list(FOLDS_5CV))
    ap.add_argument("--k", type=int, nargs="+", default=[8])
    ap.add_argument("--objective", choices=["test", "loro", "both"], default="both")
    ap.add_argument("--effort", choices=["quick", "full"], default="quick")
    ap.add_argument("--chains", type=int, default=3)
    ap.add_argument("--sa-sweeps", type=int, default=200)
    ap.add_argument("--ils", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/bounds"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ex = Experiment()
    objectives = ["test", "loro"] if args.objective == "both" else [args.objective]
    rows = []
    for fold in args.folds:
        train, test = ex.fold(fold)
        fs = FoldStats(train, test, ex.genres)
        Xc = np.where(fs.M > 0, fs.R - (fs.R.sum(1) / fs.M.sum(1))[:, None], 0.0)
        umean_rank = np.argsort(np.argsort(fs.R.sum(1) / fs.M.sum(1)))
        for K in args.k:
            ghrs_labels = ex.labels(ex.codes(ex.features(fold, RunConfig().graph, RunConfig().protocol),
                                             RunConfig().ae, 42)[0], ClusterConfig(k=K), 42)
            inits = {"ghrs": ghrs_labels,
                     "random": np.random.default_rng(args.seed).integers(0, K, fs.NU),
                     "kmeans_centred_ratings": KMeans(K, n_init=10, random_state=0).fit_predict(Xc),
                     "usermean_quantiles": np.minimum(umean_rank * K // fs.NU, K - 1)}
            for obj in objectives:
                t0 = time.perf_counter()
                L, log = search(fs, inits, K, obj, args.effort, args.chains, args.sa_sweeps, args.ils, args.seed)
                # Cross-check with the production model (UIDs there are 1-based).
                model = ClusterRatingModel().fit(train, membership_matrix(np.arange(1, fs.NU + 1), L, K), ex.genres)
                check = rmse(test["rate"], model.predict(test)[0])
                if abs(check - fs.rmse(L)) > 1e-9:
                    raise AssertionError(f"bound engine disagrees with ClusterRatingModel: {check} vs {fs.rmse(L)}")
                rows.append({"fold": fold, "K": K, "objective": obj, "effort": args.effort,
                             "test_rmse": check, "ghrs_rmse": fs.rmse(ghrs_labels),
                             "sizes": np.bincount(L, minlength=K).tolist(), **log,
                             "seconds": round(time.perf_counter() - t0, 1)})
                print(f"[{fold} K={K} {obj:4s}] best partition -> test RMSE {check:.4f} "
                      f"(GHRS {fs.rmse(ghrs_labels):.4f}; {time.perf_counter() - t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(args.out / "partition_bound.csv", index=False)

    df = pd.DataFrame(rows)
    print("\n=== mean test RMSE over folds ===")
    summary = df.groupby(["K", "objective"])[["test_rmse", "ghrs_rmse"]].mean()
    print(summary.to_string(float_format="%.4f"))
    # The search returns a partition that exists, so its RMSE is an upper bound on the
    # true minimum. A longer search could only lower it, hence the careful wording.
    for K, g in df[df.objective == "test"].groupby("K"):
        m = g["test_rmse"].mean()
        print(f"K={K}: best partition found with test-label access {m:.4f} vs paper {PAPER_TARGETS['rmse']} -> "
              + (f"no partition below the paper value was found ({args.effort} effort; evidence, not proof)"
                 if m > PAPER_TARGETS["rmse"] else "the paper value IS reachable by some partition at this K"))


if __name__ == "__main__":
    main()
