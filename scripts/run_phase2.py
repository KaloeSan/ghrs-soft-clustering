"""Phase 2: compare the novelty extensions with the strict Phase 1 baseline on ML-100K.

Every variant runs on the same folds (u1..u5) and the same seeds as the baseline, so
the comparison is paired. The extensions are:

- VAE: a variational autoencoder instead of the autoencoder in step 4,
- TF-IDF: graph edges also need a genre-profile cosine of at least ``--cos-threshold``,
- GMM / FCM: soft memberships instead of K-Means labels in step 5.

Controls separate the mechanisms behind a change in RMSE:

- "(argmax)": the same GMM/FCM partition used as hard labels, which shows how much
  of a soft variant's result comes from the softness itself,
- "Random-edge graph": as many co-rating edges as the TF-IDF rule keeps, chosen at
  random, which separates genre affinity from simply having a sparser graph,
- references without clustering: the training item mean and GHRS with K = 1. With
  membership-weighted cluster means, softer memberships move every prediction towards
  the K = 1 model, so a soft variant only shows a clustering gain if it beats K = 1.

Statistics. RMSE is averaged over seeds within each fold. Differences are tested on
the per-fold values with the corrected resampled t-test (Nadeau & Bengio 2003), which
accounts for the overlapping training sets of cross-validation, and the p-values are
Holm-adjusted over all Phase 2 variants and controls. With five folds they are
approximate, so the number of folds in which a variant is better is shown as well.
S^2 is the variance of the per-fold RMSE (as in the paper's Table 3) and sd_seeds the
spread of the overall RMSE over seeds.

The cold-start section removes all training ratings of a share of the users (as in
Fig. 13 of the paper) and scores only those users' test ratings. Each random draw of
cold users is one unit of comparison.

Usage (from the repository root):
    python -m scripts.run_phase2                              # everything with the defaults
    python -m scripts.run_phase2 --seeds 42 --cold-frac 0     # quick run, no cold start
    python -m scripts.run_phase2 --k-sweep 2 4 8 16 32 --cos-sweep 0.9 0.95

Results are written to outputs/phase2/: runs.csv, comparison.csv, tests.csv,
cold_start.csv, fuzziness_sweep.csv, k_sweep.csv, cos_sweep.csv and report.md.
"""

import ghrs  # noqa: F401 - sets KERAS_BACKEND before keras is imported

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ghrs.experiment import FOLDS_5CV, ClusterConfig, Experiment, Protocol, RunConfig, summarize
from ghrs.metrics import all_metrics

BASELINE = "Phase 1 baseline (AE + K-Means)"
ITEM_MEAN = "Reference: item mean"
K1 = "Reference: GHRS with K = 1"


def build_variants(args) -> dict[str, tuple[str, RunConfig]]:
    """Name -> (group, config) for the baseline, the Phase 2 variants and their controls."""
    base = RunConfig(cluster=ClusterConfig(k=args.k))
    tfidf = replace(base.graph, cos_threshold=args.cos_threshold)
    random_edges = replace(tfidf, edge_seed=0)

    def cfg(space="ae", method="kmeans", graph=None, argmax=False):
        cluster = replace(base.cluster, space=space, method=method, fuzziness=args.fuzziness, argmax=argmax)
        return replace(base, graph=graph or base.graph, cluster=cluster)

    return {
        BASELINE: ("baseline", base),
        "VAE + K-Means": ("phase 2", cfg(space="vae")),
        "AE + GMM": ("phase 2", cfg(method="gmm")),
        "AE + FCM": ("phase 2", cfg(method="fcm")),
        "TF-IDF graph + AE + K-Means": ("phase 2", cfg(graph=tfidf)),
        "VAE + GMM": ("phase 2", cfg(space="vae", method="gmm")),
        "VAE + FCM": ("phase 2", cfg(space="vae", method="fcm")),
        "VAE + GMM + TF-IDF": ("phase 2", cfg(space="vae", method="gmm", graph=tfidf)),
        "VAE + FCM + TF-IDF": ("phase 2", cfg(space="vae", method="fcm", graph=tfidf)),
        "AE + GMM (argmax)": ("control", cfg(method="gmm", argmax=True)),
        "AE + FCM (argmax)": ("control", cfg(method="fcm", argmax=True)),
        "VAE + GMM (argmax)": ("control", cfg(space="vae", method="gmm", argmax=True)),
        "VAE + FCM (argmax)": ("control", cfg(space="vae", method="fcm", argmax=True)),
        "Random-edge graph + AE + K-Means": ("control", cfg(graph=random_edges)),
    }


def item_mean_run(ex: Experiment, fold: str, protocol: Protocol) -> dict:
    """Training item mean (global mean for unseen items) scored like any other run."""
    _, ci, ev = ex.rating_sources(fold, protocol)
    pred = ev["MID"].map(ci.groupby("MID")["rate"].mean()).fillna(ci["rate"].mean()).to_numpy()
    return {"fold": fold, "seed": 0, "n_eval": len(ev),
            **all_metrics(ev["UID"].to_numpy(), ev["rate"].to_numpy(), pred)}


def reference_runs(ex: Experiment, folds, seed: int, protocol: Protocol = Protocol()) -> pd.DataFrame:
    """Both references; neither depends on the seed (K = 1 ignores the codes)."""
    items = pd.DataFrame([item_mean_run(ex, f, protocol) for f in folds]).assign(variant=ITEM_MEAN)
    one = ex.run(RunConfig(cluster=ClusterConfig(k=1), protocol=protocol), folds, (seed,)).assign(variant=K1)
    return pd.concat([items, one], ignore_index=True)


def corrected_ttest(d: np.ndarray, test_train_ratio: float) -> float:
    """Two-sided corrected resampled t-test (Nadeau & Bengio 2003) on per-fold differences."""
    k = len(d)
    var = float(np.var(d, ddof=1)) if k > 1 else 0.0
    if var == 0.0:
        return float("nan")
    t = float(np.mean(d)) / np.sqrt((1.0 / k + test_train_ratio) * var)
    return float(2 * stats.t.sf(abs(t), k - 1))


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjustment; NaN p-values are left out of the family."""
    valid = sorted((p, n) for n, p in pvalues.items() if not np.isnan(p))
    adjusted, running = {n: float("nan") for n in pvalues}, 0.0
    for i, (p, n) in enumerate(valid):
        running = max(running, min(1.0, (len(valid) - i) * p))
        adjusted[n] = running
    return adjusted


def test_table(runs: pd.DataFrame, groups: dict[str, str], ratio: float) -> pd.DataFrame:
    """Paired per-fold comparisons of every non-reference variant with the baseline and with K = 1."""
    per_fold = runs.groupby(["variant", "fold"])["rmse"].mean().unstack("variant")
    names = [n for n, g in groups.items() if g in ("phase 2", "control")]
    rows, p_base, p_k1 = [], {}, {}
    for name in names:
        d_base = (per_fold[name] - per_fold[BASELINE]).to_numpy()
        d_k1 = (per_fold[name] - per_fold[K1]).to_numpy()
        p_base[name], p_k1[name] = corrected_ttest(d_base, ratio), corrected_ttest(d_k1, ratio)
        rows.append({"variant": name, "group": groups[name],
                     "delta_vs_baseline": d_base.mean(), "folds_better_vs_baseline": f"{(d_base < 0).sum()}/{len(d_base)}",
                     "delta_vs_K1": d_k1.mean(), "folds_better_vs_K1": f"{(d_k1 < 0).sum()}/{len(d_k1)}"})
    adj_base, adj_k1 = holm(p_base), holm(p_k1)
    for r in rows:
        r["p_vs_baseline"], r["p_vs_K1"] = adj_base[r["variant"]], adj_k1[r["variant"]]
    cols = ["variant", "group", "delta_vs_baseline", "p_vs_baseline", "folds_better_vs_baseline",
            "delta_vs_K1", "p_vs_K1", "folds_better_vs_K1"]
    return pd.DataFrame(rows)[cols]


def accuracy_table(runs: pd.DataFrame, order: list[str], groups: dict[str, str]) -> pd.DataFrame:
    rows = []
    for name in order:
        df = runs[runs["variant"] == name]
        s = summarize(df)
        per_seed = df.groupby("seed")["rmse"].mean()
        rows.append({"variant": name, "group": groups.get(name, "reference"), "rmse": s["rmse"],
                     "sd_seeds": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else float("nan"),
                     "S2_folds": s["rmse_S2"], "mae": s["mae"],
                     "precision": s["precision@3.5"], "recall": s["recall@3.5"],
                     "max_membership": s.get("max_membership", float("nan"))})
    out = pd.DataFrame(rows)
    base = out.set_index("variant").loc[BASELINE]
    out["delta_S2"] = out["S2_folds"] - base["S2_folds"]
    out["delta_sd_seeds"] = out["sd_seeds"] - base["sd_seeds"]
    return out


def cold_start(ex: Experiment, variants: dict[str, tuple[str, RunConfig]], folds, args) -> pd.DataFrame:
    """Test RMSE of users whose training ratings were all removed, one row per run."""
    rows = []
    for cs in args.cold_seeds:
        proto = Protocol(cold_frac=args.cold_frac, cold_seed=cs, cold_eval="cold")
        for name, (_, cfg) in variants.items():
            rows.append(ex.run(replace(cfg, protocol=proto), folds, tuple(args.seeds)).assign(variant=name, cold_seed=cs))
        rows.append(reference_runs(ex, folds, args.seeds[0], proto).assign(cold_seed=cs))
        log(f"cold start draw {cs} done")
    return pd.concat(rows, ignore_index=True)


def cold_table(cold: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Per-draw RMSE (mean over folds and seeds) and paired differences across draws."""
    per_draw = cold.groupby(["variant", "cold_seed"])["rmse"].mean().unstack("variant")
    n_ratings = cold.groupby("variant")["n_eval"].mean()
    rows = []
    for name in order:
        d = per_draw[name] - per_draw[BASELINE]
        d1 = per_draw[name] - per_draw[K1]
        rows.append({"variant": name, "rmse_cold": per_draw[name].mean(),
                     "sd_draws": per_draw[name].std(ddof=1) if len(per_draw) > 1 else float("nan"),
                     "delta_vs_baseline": d.mean(), "draws_better_vs_baseline": f"{(d < 0).sum()}/{len(d)}",
                     "delta_vs_K1": d1.mean(), "draws_better_vs_K1": f"{(d1 < 0).sum()}/{len(d1)}",
                     "n_ratings_per_fold": int(round(n_ratings[name]))})
    return pd.DataFrame(rows)


def fuzziness_sweep(ex: Experiment, variants, folds, args) -> pd.DataFrame:
    """AE + FCM for several fuzzifiers: shows how softness alone moves the RMSE."""
    rows = []
    cfg0 = variants["AE + FCM"][1]
    for m in args.fuzziness_sweep:
        s = summarize(ex.run(replace(cfg0, cluster=replace(cfg0.cluster, fuzziness=m)), folds, tuple(args.seeds)))
        rows.append({"m": m, "rmse": s["rmse"], "max_membership": s["max_membership"]})
        log(f"fuzziness m={m:<5} RMSE={s['rmse']:.4f} max membership={s['max_membership']:.3f}")
    return pd.DataFrame(rows)


def k_sweep(ex: Experiment, variants, folds, args) -> pd.DataFrame:
    rows = []
    for name in (BASELINE, "AE + GMM", "AE + FCM", "VAE + FCM + TF-IDF"):
        cfg0 = variants[name][1]
        for k in args.k_sweep:
            s = summarize(ex.run(replace(cfg0, cluster=replace(cfg0.cluster, k=k)), folds, tuple(args.seeds)))
            rows.append({"variant": name, "k": k, "rmse": s["rmse"]})
            log(f"K sweep  {name:32s} K={k:3d} RMSE={s['rmse']:.4f}")
    return pd.DataFrame(rows)


def cos_sweep(ex: Experiment, variants, folds, args) -> pd.DataFrame:
    """TF-IDF graph against its random-edge control for several cosine thresholds."""
    rows = []
    base = variants[BASELINE][1]
    for tau in args.cos_sweep:
        res = {}
        for label, seed in (("tfidf", None), ("random_edges", 0)):
            cfg = replace(base, graph=replace(base.graph, cos_threshold=tau, edge_seed=seed))
            res[label] = summarize(ex.run(cfg, folds, tuple(args.seeds)))["rmse"]
        rows.append({"cos_threshold": tau, "rmse_tfidf": res["tfidf"], "rmse_random_edges": res["random_edges"],
                     "tfidf_minus_random": res["tfidf"] - res["random_edges"]})
        log(f"cosine threshold {tau}: TF-IDF {res['tfidf']:.4f}, random edges {res['random_edges']:.4f}")
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    def cell(v):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return ""
            return f"{v:.2e}" if 0 < abs(v) < 1e-3 else f"{v:.4f}"
        return str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * df.shape[1]]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def summary_lines(acc: pd.DataFrame, tests: pd.DataFrame, n_folds: int) -> list[str]:
    """Plain-language verdict for every Phase 2 variant, computed from the tables."""
    rmse = acc.set_index("variant")["rmse"]
    lines = []
    for r in tests[tests["group"] == "phase 2"].itertuples():
        vs_base = "lower" if r.delta_vs_baseline < 0 else "higher"
        vs_k1 = "lower" if r.delta_vs_K1 < 0 else "higher"
        lines.append(f"- {r.variant}: RMSE {rmse[r.variant]:.4f}, {abs(r.delta_vs_baseline):.4f} {vs_base} than the "
                     f"baseline (p = {r.p_vs_baseline:.3g}, better in {r.folds_better_vs_baseline} folds) and "
                     f"{abs(r.delta_vs_K1):.4f} {vs_k1} than GHRS with K = 1 (p = {r.p_vs_K1:.3g}, "
                     f"better in {r.folds_better_vs_K1} folds).")
    phase2 = acc[acc["group"] == "phase 2"]
    n_lower_s2 = int((phase2["delta_S2"] < 0).sum())
    n_lower_sd = int((phase2["delta_sd_seeds"] < 0).sum())
    lines += ["", f"Variance: S^2 over folds is lower than the baseline's for {n_lower_s2} of {len(phase2)} variants and "
              f"the spread over seeds for {n_lower_sd} of {len(phase2)}. {n_folds} folds are too few to "
              "tell such differences in variance apart from noise."]
    return lines


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="GHRS Phase 2 comparison (ML-100K, cross-validation on u1..u5)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3],
                    help="AE/VAE and clustering seeds (default: the seeds of the Phase 1 baseline)")
    ap.add_argument("--folds", nargs="+", default=list(FOLDS_5CV))
    ap.add_argument("--k", type=int, default=8, help="number of clusters for every variant")
    ap.add_argument("--fuzziness", type=float, default=2.0, help="FCM fuzzifier m")
    ap.add_argument("--cos-threshold", type=float, default=0.95,
                    help="minimum TF-IDF genre cosine for an edge (0.95 is about the median over "
                         "alpha-edges in the training folds)")
    ap.add_argument("--variants", nargs="+", default=None, help="subset of variant names to run")
    ap.add_argument("--cold-frac", type=float, default=0.1, help="share of cold-start users, 0 = skip")
    ap.add_argument("--cold-seeds", type=int, nargs="+", default=[0, 1, 2], help="random draws of cold users")
    ap.add_argument("--fuzziness-sweep", type=float, nargs="*", default=[1.1, 1.25, 1.5, 2.0, 3.0])
    ap.add_argument("--k-sweep", type=int, nargs="*", default=[])
    ap.add_argument("--cos-sweep", type=float, nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=Path("outputs/phase2"))
    args = ap.parse_args()

    all_variants = build_variants(args)
    if args.variants:
        unknown = set(args.variants) - set(all_variants)
        if unknown:
            ap.error(f"unknown variants {sorted(unknown)}; choose from {list(all_variants)}")
    names = [BASELINE] + [n for n in all_variants if n != BASELINE and (not args.variants or n in args.variants)]
    variants = {n: all_variants[n] for n in names}
    groups = {n: g for n, (g, _) in variants.items()}
    args.out.mkdir(parents=True, exist_ok=True)

    ex = Experiment()
    runs = []
    for name, (_, cfg) in variants.items():
        t0 = time.perf_counter()
        runs.append(ex.run(cfg, args.folds, tuple(args.seeds)).assign(variant=name))
        log(f"{name:34s} RMSE={runs[-1]['rmse'].mean():.4f}  ({time.perf_counter() - t0:.0f}s)")
    runs = pd.concat(runs + [reference_runs(ex, args.folds, args.seeds[0])], ignore_index=True)
    runs.to_csv(args.out / "runs.csv", index=False)

    order = list(variants) + [ITEM_MEAN, K1]
    acc = accuracy_table(runs, order, groups)
    acc.to_csv(args.out / "comparison.csv", index=False)
    ratio = float(np.mean([len(ex.fold(f)[1]) / len(ex.fold(f)[0]) for f in args.folds]))
    tests = test_table(runs, groups, ratio)
    tests.to_csv(args.out / "tests.csv", index=False)

    report = [
        "# Phase 2 comparison", "",
        f"{len(args.folds)}-fold cross-validation ({', '.join(args.folds)}), seeds {args.seeds}, K = {args.k}, "
        f"FCM m = {args.fuzziness}, TF-IDF cosine threshold = {args.cos_threshold}.", "",
        "## Accuracy", "",
        "sd_seeds: spread of the overall RMSE over seeds. S2_folds: variance of the per-fold RMSE. "
        "max_membership: mean largest membership weight (1 = hard clustering). "
        "delta_S2 and delta_sd_seeds are differences to the baseline.", "",
        md_table(acc), "",
        "## Paired tests", "",
        "Per-fold differences in RMSE (negative = better). p: corrected resampled t-test "
        f"(Nadeau & Bengio 2003, test/train ratio {ratio:.2f}), Holm-adjusted over the {len(tests)} "
        f"Phase 2 variants and controls. With {len(args.folds)} folds the p-values are approximate.", "",
        md_table(tests), "",
        "## Summary", "",
        *summary_lines(acc, tests, len(args.folds)), "",
        "With membership-weighted cluster means, softer memberships pull every prediction towards the "
        "K = 1 model. Compare each soft variant with its argmax control (same partition, hard labels) "
        "and with the K = 1 reference before attributing a gain to the clustering itself.", "",
    ]

    if args.cold_frac > 0:
        cold = cold_start(ex, variants, args.folds, args)
        cold.to_csv(args.out / "cold_start.csv", index=False)
        report += [f"## Cold start ({args.cold_frac:.0%} of users without training ratings, "
                   f"{len(args.cold_seeds)} random draws)", "",
                   "RMSE on the cold users' test ratings, averaged over folds and seeds within each draw; "
                   "differences are paired by draw.", "",
                   md_table(cold_table(cold, order)), ""]

    if args.fuzziness_sweep and "AE + FCM" in variants:
        fz = fuzziness_sweep(ex, variants, args.folds, args)
        fz.to_csv(args.out / "fuzziness_sweep.csv", index=False)
        report += ["## AE + FCM as a function of the fuzzifier m", "",
                   f"Reference RMSE: baseline {acc.set_index('variant').loc[BASELINE, 'rmse']:.4f}, "
                   f"K = 1 {acc.set_index('variant').loc[K1, 'rmse']:.4f}.", "", md_table(fz), ""]

    if args.k_sweep and all(n in variants for n in (BASELINE, "AE + GMM", "AE + FCM", "VAE + FCM + TF-IDF")):
        ks = k_sweep(ex, variants, args.folds, args)
        ks.to_csv(args.out / "k_sweep.csv", index=False)
        report += ["## RMSE as a function of K", "",
                   md_table(ks.pivot(index="k", columns="variant", values="rmse").reset_index()), ""]

    if args.cos_sweep:
        cs = cos_sweep(ex, variants, args.folds, args)
        cs.to_csv(args.out / "cos_sweep.csv", index=False)
        report += ["## TF-IDF threshold against the random-edge control (AE + K-Means)", "", md_table(cs), ""]

    (args.out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n" + "\n".join(report))
    print(f"Saved: {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
