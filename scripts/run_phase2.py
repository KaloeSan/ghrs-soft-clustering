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
    python -m scripts.run_phase2 --from-results               # rebuild report and figures only

The CSV tables (runs, comparison, tests, cold_start, fuzziness_sweep, k_sweep,
cos_sweep) and run_config.json go to outputs/phase2/. With that default --out, the
Markdown report is written to the git-tracked docs/phase2_results_report.md and the
figures (ghrs/figures.py) to the git-tracked figs/ directory. A run with any other
--out keeps its report and figures inside that directory, so test runs never
overwrite the thesis results.
"""

import ghrs  # noqa: F401 - sets KERAS_BACKEND before keras is imported

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ghrs.experiment import FOLDS_5CV, ClusterConfig, Experiment, Protocol, RunConfig, summarize
from ghrs.data import load_ratings
from ghrs.figures import FIGS_DIR, make_figures
from ghrs.metrics import all_metrics
from ghrs.paths import DOCS, OUTPUTS, ROOT
from ghrs.stats import corrected_ttest, holm
from ghrs.tables import markdown_table

BASELINE = "Phase 1 baseline (AE + K-Means)"
ITEM_MEAN = "Reference: item mean"
K1 = "Reference: GHRS with K = 1"
DEFAULT_OUT = OUTPUTS / "phase2"
TRACKED_REPORT = DOCS / "phase2_results_report.md"
RESULT_TABLES = ("cold_start", "fuzziness_sweep", "k_sweep", "cos_sweep")


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


def reference_runs(ex: Experiment, folds, seed: int, protocol: Protocol | None = None) -> pd.DataFrame:
    """Both references; neither depends on the seed (K = 1 ignores the codes)."""
    protocol = protocol or Protocol()
    items = pd.DataFrame([item_mean_run(ex, f, protocol) for f in folds]).assign(variant=ITEM_MEAN)
    one = ex.run(RunConfig(cluster=ClusterConfig(k=1), protocol=protocol), folds, (seed,)).assign(variant=K1)
    return pd.concat([items, one], ignore_index=True)


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
                     "delta_vs_baseline": d_base.mean(),
                     "folds_better_vs_baseline": f"{(d_base < 0).sum()}/{len(d_base)}",
                     "delta_vs_K1": d_k1.mean(),
                     "folds_better_vs_K1": f"{(d_k1 < 0).sum()}/{len(d_k1)}"})
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
            df = ex.run(replace(cfg, protocol=proto), folds, tuple(args.seeds))
            rows.append(df.assign(variant=name, cold_seed=cs))
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
    lines += ["", f"Variance: S^2 over folds is lower than the baseline's for {n_lower_s2} of {len(phase2)} "
              f"variants and the spread over seeds for {n_lower_sd} of {len(phase2)}. {n_folds} folds are too "
              "few to tell such differences in variance apart from noise."]
    return lines


def run_experiments(ex: Experiment, variants: dict[str, tuple[str, RunConfig]], args) -> dict[str, pd.DataFrame]:
    """Run the variants, the references and the optional sections; save every table as CSV."""
    runs = []
    for name, (_, cfg) in variants.items():
        t0 = time.perf_counter()
        runs.append(ex.run(cfg, args.folds, tuple(args.seeds)).assign(variant=name))
        log(f"{name:34s} RMSE={runs[-1]['rmse'].mean():.4f}  ({time.perf_counter() - t0:.0f}s)")
    results = {"runs": pd.concat(runs + [reference_runs(ex, args.folds, args.seeds[0])], ignore_index=True)}
    if args.cold_frac > 0:
        results["cold_start"] = cold_start(ex, variants, args.folds, args)
    if args.fuzziness_sweep and "AE + FCM" in variants:
        results["fuzziness_sweep"] = fuzziness_sweep(ex, variants, args.folds, args)
    if args.k_sweep and all(n in variants for n in (BASELINE, "AE + GMM", "AE + FCM", "VAE + FCM + TF-IDF")):
        results["k_sweep"] = k_sweep(ex, variants, args.folds, args)
    if args.cos_sweep:
        results["cos_sweep"] = cos_sweep(ex, variants, args.folds, args)
    for name, df in results.items():
        df.to_csv(args.out / f"{name}.csv", index=False)
    return results


def load_results(out: Path) -> dict[str, pd.DataFrame]:
    """Read the CSV tables of an earlier run (runs.csv is required, the rest optional)."""
    results = {"runs": pd.read_csv(out / "runs.csv")}
    for name in RESULT_TABLES:
        if (out / f"{name}.csv").exists():
            results[name] = pd.read_csv(out / f"{name}.csv")
    return results


def run_settings(args, variant_names: list[str]) -> dict:
    """The parameters a report needs, saved next to the results as run_config.json."""
    return {"folds": list(args.folds), "seeds": list(args.seeds), "k": args.k, "fuzziness": args.fuzziness,
            "cos_threshold": args.cos_threshold, "cold_frac": args.cold_frac,
            "cold_seeds": list(args.cold_seeds), "variants": variant_names}


def load_settings(out: Path, results: dict[str, pd.DataFrame], args) -> dict:
    """Settings of an earlier run: run_config.json if present, otherwise inferred from the tables.

    Folds, seeds, variants and cold-start draws can be read from the CSV files; K, m,
    the cosine threshold and the cold-start share then come from the command line and
    must match the earlier run.
    """
    path = out / "run_config.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    runs = results["runs"]
    names = [n for n in dict.fromkeys(runs["variant"]) if n not in (ITEM_MEAN, K1)]
    base = runs[runs["variant"] == BASELINE]
    cold = results.get("cold_start")
    log(f"{path} not found: folds, seeds and variants inferred from the CSV files, the other settings "
        "taken from the command line")
    args.folds, args.seeds = list(dict.fromkeys(base["fold"])), [int(s) for s in dict.fromkeys(base["seed"])]
    if cold is not None:
        args.cold_seeds = [int(s) for s in dict.fromkeys(cold["cold_seed"])]
    return run_settings(args, names)


def test_train_ratio(folds: list[str]) -> float:
    """Average size of the test set relative to the training set (for the corrected t-test)."""
    return float(np.mean([len(load_ratings(f"{f}.test")) / len(load_ratings(f"{f}.base")) for f in folds]))


def build_report(results: dict[str, pd.DataFrame], settings: dict, groups: dict[str, str]
                 ) -> tuple[list[str], pd.DataFrame]:
    """Markdown report from the result tables; also returns the accuracy table for the figures."""
    folds, runs = settings["folds"], results["runs"]
    order = list(settings["variants"]) + [ITEM_MEAN, K1]
    acc = accuracy_table(runs, order, groups)
    ratio = test_train_ratio(folds)
    tests = test_table(runs, groups, ratio)
    lines = [
        "# Phase 2 results", "",
        "Generated by `scripts/run_phase2.py`. Rerun the script (or `python -m scripts.run_phase2 "
        "--from-results` to rebuild it from the saved CSV files) instead of editing this file by hand.", "",
        f"{len(folds)}-fold cross-validation ({', '.join(folds)}), seeds {settings['seeds']}, "
        f"K = {settings['k']}, FCM m = {settings['fuzziness']}, TF-IDF cosine threshold = "
        f"{settings['cos_threshold']}.", "",
        "## Accuracy", "",
        "sd_seeds: spread of the overall RMSE over seeds. S2_folds: variance of the per-fold RMSE. "
        "max_membership: mean largest membership weight (1 = hard clustering). "
        "delta_S2 and delta_sd_seeds are differences to the baseline.", "",
        markdown_table(acc), "",
        "## Paired tests", "",
        "Per-fold differences in RMSE (negative = better). p: corrected resampled t-test "
        f"(Nadeau & Bengio 2003, test/train ratio {ratio:.2f}), Holm-adjusted over the {len(tests)} "
        f"Phase 2 variants and controls. With {len(folds)} folds the p-values are approximate.", "",
        markdown_table(tests), "",
        "## Summary", "",
        *summary_lines(acc, tests, len(folds)), "",
        "With membership-weighted cluster means, softer memberships pull every prediction towards the "
        "K = 1 model. Compare each soft variant with its argmax control (same partition, hard labels) "
        "and with the K = 1 reference before attributing a gain to the clustering itself.", "",
    ]
    if "cold_start" in results:
        lines += [f"## Cold start ({settings['cold_frac']:.0%} of users without training ratings, "
                  f"{len(settings['cold_seeds'])} random draws)", "",
                  "RMSE on the cold users' test ratings, averaged over folds and seeds within each draw; "
                  "differences are paired by draw.", "",
                  markdown_table(cold_table(results["cold_start"], order)), ""]
    if "fuzziness_sweep" in results:
        rmse = acc.set_index("variant")["rmse"]
        lines += ["## AE + FCM as a function of the fuzzifier m", "",
                  f"Reference RMSE: baseline {rmse[BASELINE]:.4f}, K = 1 {rmse[K1]:.4f}.", "",
                  markdown_table(results["fuzziness_sweep"]), ""]
    if "k_sweep" in results:
        table = results["k_sweep"].pivot(index="k", columns="variant", values="rmse").reset_index()
        lines += ["## RMSE as a function of K", "", markdown_table(table), ""]
    if "cos_sweep" in results:
        lines += ["## TF-IDF threshold against the random-edge control (AE + K-Means)", "",
                  markdown_table(results["cos_sweep"]), ""]
    return lines, acc


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def same_path(a: Path, b: Path) -> bool:
    return Path(a).resolve() == Path(b).resolve()


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
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="directory for the CSV result tables")
    ap.add_argument("--report", type=Path, default=None,
                    help=f"Markdown report (default: {TRACKED_REPORT.relative_to(ROOT)} for the default --out, "
                         "else <out>/report.md)")
    ap.add_argument("--figs-dir", type=Path, default=None,
                    help="where to save the figures (default: figs/ for the default --out, else <out>/figs)")
    ap.add_argument("--fig-formats", nargs="+", default=["png", "pdf"], help="figure file formats")
    ap.add_argument("--from-results", action="store_true",
                    help="rebuild the report and figures from the CSV files in --out without running anything")
    args = ap.parse_args()
    default_run = same_path(args.out, DEFAULT_OUT)
    args.report = args.report or (TRACKED_REPORT if default_run else args.out / "report.md")
    args.figs_dir = args.figs_dir or (FIGS_DIR if default_run else args.out / "figs")

    all_variants = build_variants(args)
    if args.from_results:
        results = load_results(args.out)
        settings = load_settings(args.out, results, args)
    else:
        if args.variants:
            unknown = set(args.variants) - set(all_variants)
            if unknown:
                ap.error(f"unknown variants {sorted(unknown)}; choose from {list(all_variants)}")
        names = [BASELINE] + [n for n in all_variants
                              if n != BASELINE and (not args.variants or n in args.variants)]
        args.out.mkdir(parents=True, exist_ok=True)
        results = run_experiments(Experiment(), {n: all_variants[n] for n in names}, args)
        settings = run_settings(args, names)
        (args.out / "run_config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    groups = {n: g for n, (g, _) in all_variants.items() if n in settings["variants"]}
    report, acc = build_report(results, settings, groups)
    acc.to_csv(args.out / "comparison.csv", index=False)
    test_table(results["runs"], groups, test_train_ratio(settings["folds"])).to_csv(args.out / "tests.csv",
                                                                                    index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report), encoding="utf-8")
    print("\n" + "\n".join(report))
    print(f"Saved: {args.report}")

    figures = make_figures(args.figs_dir, acc, results.get("fuzziness_sweep"), results.get("k_sweep"),
                           tuple(args.fig_formats), k=settings.get("k", args.k),
                           default_m=settings.get("fuzziness", args.fuzziness))
    for path in figures:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
