"""Hyperparameter search with Optuna: can any GHRS configuration reach RMSE 0.887?

The search runs 5-fold cross-validation on ML-100K (u1..u5) and should be read as a
sensitivity analysis. Steps 1-5 only decide how users are partitioned, and
``scripts/partition_bound.py`` shows that at K=8 even a partition chosen with the
test labels stays around 0.90. For larger K such partitions do go below 0.887, but
the real autoencoder + K-Means partitions get worse as K grows. The script shows how
far real configurations get, which hyperparameters matter, and (with ``--ci-source
full`` and similar options) what happens under a leaky protocol.

Search spaces:

    paper     only what the paper leaves open (hidden width, L1 and L2 strength, both
              non-zero; learning rate; batch size; the meaning of "similar items")
              plus the values the paper tuned itself (alpha on its grid, K in
              [2, 30]). Everything else is fixed as in the paper.
    extended  the paper space plus changes outside the paper: a relaxed graph rule,
              other activations, BatchNorm, clustering on scaled or raw features,
              larger K, and the extra estimators in ghrs/recommender.py.

Objectives:

    cv-test    mean RMSE over the five test folds. This selects models on the test
               data, so it is optimistic on purpose: if it cannot reach 0.887, honest
               tuning cannot either. This is the default.
    inner-val  each training fold is split 90/10 and the search scores the 10%
               part. The best trials are then re-scored on the real test folds; the
               summary reports the trial the search ranked best, and the lowest test
               score among the top trials separately.

Trials are pruned with a median pruner after each fold. The study is stored in an
Optuna journal file, so a run can be resumed and several worker processes can share it.

Usage (from the repository root):
    python -m scripts.optimize_phase1 --space paper --trials 150 --workers 4
    python -m scripts.optimize_phase1 --space extended --trials 300 --workers 4
    python -m scripts.optimize_phase1 --space paper --objective inner-val --trials 100
    python -m scripts.optimize_phase1 --space paper --ci-source full     # search under the H3 leak

Results are written to outputs/hpo/<study>/ (trials.csv, best.json, top_reeval.csv, summary.txt).
"""

import ghrs  # noqa: F401 - sets KERAS_BACKEND before keras is imported

import argparse
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from ghrs.autoencoder import AEConfig
from ghrs.experiment import (FOLDS_5CV, PAPER_TARGETS, ClusterConfig, Experiment, GraphConfig, Protocol,
                             RunConfig, summarize)
from ghrs.recommender import RecConfig

ALPHA_GRID = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]


def suggest(trial: optuna.Trial, space: str, protocol: Protocol) -> RunConfig:
    """Sample one configuration from the chosen search space."""
    if space == "paper":  # elastic net, so both penalties are present
        l1 = trial.suggest_float("l1", 1e-7, 1e-2, log=True)
        l2 = trial.suggest_float("l2", 1e-7, 1e-2, log=True)
    else:
        l1 = trial.suggest_float("l1", 1e-7, 1e-2, log=True) if trial.suggest_categorical("use_l1", [True, False]) else 0.0
        l2 = trial.suggest_float("l2", 1e-7, 1e-2, log=True) if trial.suggest_categorical("use_l2", [True, False]) else 0.0
    ae = AEConfig(
        hidden_dim=trial.suggest_categorical("hidden_dim", [4, 6, 8, 12, 16, 24, 32]),
        l1=l1, l2=l2,
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 3e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 754]),
    )
    graph = GraphConfig(alpha=trial.suggest_categorical("alpha", ALPHA_GRID))
    k = trial.suggest_int("k", 2, 30) if space == "paper" else trial.suggest_int("k", 2, 64, log=True)
    cluster = ClusterConfig(k=k)
    similar = ["genre_exact", "genre_jaccard", "item_cosine"] + (["none"] if space == "extended" else [])
    rec = RecConfig(similar=trial.suggest_categorical("similar", similar))

    if space == "extended":
        mode = trial.suggest_categorical("graph_mode", ["absolute", "corated"])
        tol = trial.suggest_categorical("tolerance", [0, 1])
        # The thresholds are kept fairly high because networkx betweenness and load
        # centrality are slow on dense graphs. The co-rated rule with tolerance 1 still
        # gives 100k-200k edges, which is slow the first time and cached afterwards.
        if mode == "corated":
            alpha = trial.suggest_categorical("rho", [0.6, 0.7, 0.8])
        else:
            alpha = graph.alpha * (trial.suggest_categorical("alpha_mult_tol1", [2, 3]) if tol else 1)
        graph = GraphConfig(alpha=alpha, tolerance=tol, mode=mode)
        ae = replace(ae, **{k: trial.suggest_categorical(k, v) for k, v in (
            ("output_activation", ["relu", "sigmoid", "linear"]),
            ("code_activation", ["relu", "linear", "tanh"]),
            ("batchnorm", [False, True]))})
        cluster = ClusterConfig(k=k,
                                space=trial.suggest_categorical("cluster_space", ["ae", "raw"]),
                                scale=trial.suggest_categorical("cluster_scale", ["minmax", "none", "standard"]))
        baseline = trial.suggest_categorical("baseline", ["none", "item", "user", "user_item"])
        rec = replace(rec, baseline=baseline,
                      shrink=trial.suggest_float("shrink", 0.0, 100.0),
                      fallback=trial.suggest_categorical("fallback", ["cluster", "baseline"]))
    return RunConfig(graph=graph, ae=ae, cluster=cluster, rec=rec, protocol=protocol)


def inner_folds(ex: Experiment, folds, seed: int = 0, frac: float = 0.1) -> list[str]:
    """Register '<fold>-inner' folds: 90% of the training ratings for training, 10% for validation."""
    names = []
    for f in folds:
        train, _ = ex.fold(f)
        val_mask = np.random.default_rng(seed).random(len(train)) < frac
        name = f"{f}-inner"
        ex._folds[name] = (train[~val_mask].reset_index(drop=True), train[val_mask].reset_index(drop=True))
        names.append(name)
    return names


def make_objective(ex: Experiment, space: str, protocol: Protocol, folds, seed: int):
    def objective(trial: optuna.Trial) -> float:
        cfg = suggest(trial, space, protocol)
        trial.set_user_attr("config", json.dumps(asdict(cfg)))
        rmses = []
        for step, fold in enumerate(folds):
            try:
                rmses.append(ex.run_fold(fold, cfg, seed)["rmse"])
            except Exception as e:  # e.g. an empty graph or a degenerate clustering
                trial.set_user_attr("error", repr(e))
                raise optuna.TrialPruned() from e
            trial.report(float(np.mean(rmses)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(rmses))
    return objective


def sampler_pruner(args, worker_id: int = 0):
    """Sampler and pruner shared by create_study and every worker.

    load_study does not store these settings, so each worker creates them again. Each
    worker gets its own sampler seed so the workers do not propose the same trials.
    """
    return (optuna.samplers.TPESampler(seed=args.sampler_seed + worker_id, multivariate=True, group=True),
            optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))


def storage_for(path: Path) -> optuna.storages.BaseStorage:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The default lock uses symlinks, which need admin rights on Windows (WinError 1314).
    j = optuna.storages.journal
    return optuna.storages.JournalStorage(j.JournalFileBackend(str(path), lock_obj=j.JournalFileOpenLock(str(path))))


def worker(args_dict: dict, n_trials: int, worker_id: int) -> None:
    args = argparse.Namespace(**args_dict)
    import torch
    torch.set_num_threads(args.threads or max(1, (os.cpu_count() or 2) // max(1, args.workers)))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ex = Experiment()
    protocol = Protocol(graph_source=args.graph_source, ci_source=args.ci_source, eval_on=args.eval_on)
    folds = list(FOLDS_5CV) if args.objective == "cv-test" else inner_folds(ex, FOLDS_5CV)
    sampler, pruner = sampler_pruner(args, worker_id)
    study = optuna.load_study(study_name=args.study, storage=storage_for(args.journal), sampler=sampler, pruner=pruner)
    t0 = time.time()

    def cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        done = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        best = min(t.value for t in done) if done else float("nan")  # best_value raises if nothing is complete
        val = f"{trial.value:.4f}" if trial.value is not None else trial.state.name
        print(f"[w{worker_id} {time.time() - t0:6.0f}s] trial {trial.number:4d}: {val}   best={best:.4f}", flush=True)

    study.optimize(make_objective(ex, args.space, protocol, folds, args.seed), n_trials=n_trials, callbacks=[cb])


def reevaluate(args, study: optuna.Study, out: Path) -> pd.DataFrame:
    """Re-score the best trials on the five test folds with several seeds.

    A single-seed search favours configurations that were lucky with their seed, so
    the final numbers come from this re-evaluation.
    """
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top = sorted(done, key=lambda t: t.value)[: args.top]
    ex = Experiment()
    rows = []
    for t in top:
        d = json.loads(t.user_attrs["config"])
        cfg = RunConfig(graph=GraphConfig(**d["graph"]), ae=AEConfig(**d["ae"]), cluster=ClusterConfig(**d["cluster"]),
                        rec=RecConfig(**d["rec"]), protocol=Protocol(**d["protocol"]))
        s = summarize(ex.run(cfg, FOLDS_5CV, tuple(args.reeval_seeds)))
        rows.append({"trial": t.number, "search_value": t.value, **s, **cfg.flat()})
        print(f"re-eval trial {t.number:4d}: search={t.value:.4f} -> 5-fold x {len(args.reeval_seeds)} seeds "
              f"RMSE={s['rmse']:.4f} (S2={s['rmse_S2']:.2e}) P={s['precision@3.5']:.3f} R={s['recall@3.5']:.3f}",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "top_reeval.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="GHRS Phase-1 hyperparameter search")
    ap.add_argument("--space", choices=["paper", "extended"], default="paper")
    ap.add_argument("--objective", choices=["cv-test", "inner-val"], default="cv-test")
    ap.add_argument("--trials", type=int, default=100, help="total number of trials (all workers)")
    ap.add_argument("--workers", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--threads", type=int, default=0, help="torch threads per worker (0 = cpu_count / workers)")
    ap.add_argument("--seed", type=int, default=42, help="AE/K-Means seed during the search")
    ap.add_argument("--sampler-seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=5, help="trials re-evaluated at the end")
    ap.add_argument("--reeval-seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--graph-source", default="train", choices=["train", "full", "authors_ua"])
    ap.add_argument("--ci-source", default="train", choices=["train", "full"])
    ap.add_argument("--eval-on", default="test", choices=["test", "train"])
    ap.add_argument("--study", default=None, help="study name (default: derived from the options)")
    args = ap.parse_args()

    proto = Protocol(graph_source=args.graph_source, ci_source=args.ci_source, eval_on=args.eval_on)
    tag = "strict" if proto.is_strict else f"g-{args.graph_source}_ci-{args.ci_source}_ev-{args.eval_on}"
    args.study = args.study or f"{args.space}_{args.objective}_{tag}"
    out = Path("outputs/hpo") / args.study
    out.mkdir(parents=True, exist_ok=True)
    args.journal = out / "journal.log"

    sampler, pruner = sampler_pruner(args)
    study = optuna.create_study(study_name=args.study, storage=storage_for(args.journal), direction="minimize",
                                load_if_exists=True, sampler=sampler, pruner=pruner)
    remaining = max(0, args.trials - len([t for t in study.trials if t.state.is_finished()]))
    print(f"study={args.study}  space={args.space}  objective={args.objective}  protocol={proto}  "
          f"trials to run={remaining}  workers={args.workers}", flush=True)

    if remaining:
        per = [remaining // args.workers + (i < remaining % args.workers) for i in range(args.workers)]
        if args.workers == 1:
            worker(vars(args), per[0], 0)
        else:
            ctx = mp.get_context("spawn")
            procs = [ctx.Process(target=worker, args=(vars(args), n, i)) for i, n in enumerate(per) if n]
            for p in procs:
                p.start()
            for p in procs:
                p.join()

    study = optuna.load_study(study_name=args.study, storage=storage_for(args.journal))
    trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "duration"))
    trials.to_csv(out / "trials.csv", index=False)
    best = study.best_trial
    (out / "best.json").write_text(json.dumps({"value": best.value, "params": best.params,
                                               "config": json.loads(best.user_attrs["config"])}, indent=2))
    print(f"\nbest search value = {best.value:.4f} (trial {best.number}); params = {best.params}")
    top = reevaluate(args, study, out)

    try:
        imp = optuna.importance.get_param_importances(study)
        imp_txt = "\n".join(f"  {k:22s} {v:.3f}" for k, v in imp.items())
    except Exception as e:  # needs enough completed trials
        imp_txt = f"  n/a ({e!r})"
    # Report the trial the search ranked best. The lowest test score among the top
    # trials is printed as well, but it is selected on the test data.
    best_re = top.sort_values("search_value").iloc[0]
    oracle_re = top.sort_values("rmse").iloc[0]
    states = [t.state for t in study.trials]
    summary = (f"study: {args.study}\nprotocol: {proto}\ntrials: "
               f"{states.count(optuna.trial.TrialState.COMPLETE)} complete, "
               f"{states.count(optuna.trial.TrialState.PRUNED)} pruned, "
               f"{states.count(optuna.trial.TrialState.FAIL)} failed\n"
               f"best search objective ({args.objective}): {best.value:.4f}\n"
               f"re-evaluated 5-fold RMSE of the search-best trial ({len(args.reeval_seeds)} seeds): "
               f"{best_re['rmse']:.4f} (trial {int(best_re['trial'])}; P {best_re['precision@3.5']:.3f}, "
               f"R {best_re['recall@3.5']:.3f})\n"
               f"min re-evaluated RMSE among the top {len(top)} (test-selected): {oracle_re['rmse']:.4f}\n"
               f"paper target: {PAPER_TARGETS['rmse']}  ->  gap = {best_re['rmse'] - PAPER_TARGETS['rmse']:+.4f}\n"
               f"parameter importances (fANOVA):\n{imp_txt}\n")
    (out / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()
