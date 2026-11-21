"""Final synthesis: our GHRS pipeline and Phase 2 clustering on top of SVD++.

R. Jarzyński's parallel work showed that the largest gain over GHRS comes from user
and item biases and from factorisation (SVD++ with GHRS side information), not from
the clustering. This script puts both lines of work into one comparison:

- the Phase 1 reproduction and the best Phase 2 variant (AE + FCM),
- two references without clustering: the item mean and ALS user/item biases,
- SVD++ with the 35 GHRS user features and the genre flags (ghrs.factorization),
- SVD++ with a cluster correction: the residuals of SVD++ are averaged per cluster
  and movie (K-Means, FCM, or VAE + FCM + TF-IDF graph) and shrunk towards zero,
  i.e. towards SVD++. The residuals come from out-of-fold SVD++ predictions.

Two evaluation protocols are used: the paper's random 5-fold split (u1..u5) and a
temporal split in which every user's last 20% of ratings form the test set. With
--temporal-global a third, mostly cold-start split is added.

A label-permutation audit (also from R. Jarzyński's work) checks that predictions do
not depend on the test ratings: the test labels of fold u1 are shuffled and the
predictions recomputed. The strict pipeline must not change; the deliberately leaky
protocol H3 (test ratings in the cluster means) must.

Usage (from the repository root):
    python -m scripts.run_final_synthesis
    python -m scripts.run_final_synthesis --seeds 42 --no-audit     # quick run

The CSV tables go to outputs/synthesis/. With that default --out the Markdown report
is written to the git-tracked docs/phase3_synthesis_results.md and figures 1 and 4
(ghrs/figures.py) to figs/; with any other --out both stay inside that directory.
"""

import ghrs  # noqa: F401 - sets KERAS_BACKEND before keras is imported

import argparse
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ghrs.experiment import FOLDS_5CV, ClusterConfig, Experiment, GraphConfig, Protocol, RunConfig, summarize
from ghrs.factorization import SVDppConfig
from ghrs.figures import FIGS_DIR, load_phase2_runs, make_synthesis_figures
from ghrs.metrics import all_metrics
from ghrs.paths import DOCS, OUTPUTS
from ghrs.recommender import RecConfig
from ghrs.stats import corrected_ttest, holm
from ghrs.tables import markdown_table

DEFAULT_OUT = OUTPUTS / "synthesis"
TRACKED_REPORT = DOCS / "phase3_synthesis_results.md"
GHRS = "GHRS reproduction (AE + K-Means)"
SVDPP = "SVD++ with GHRS features"
ITEM_MEAN = "Item mean"


def build_variants(shrink: float) -> dict[str, tuple[str, RunConfig, bool]]:
    """Name -> (group, config, depends on the clustering seed)."""
    base = RunConfig()
    no_cluster = ClusterConfig(k=1)
    pure_offset = RecConfig(baseline="offset", similar="none", fallback="baseline", shrink=1e12)
    residual = RecConfig(baseline="offset", similar="none", fallback="baseline", shrink=shrink)
    tfidf = GraphConfig(cos_threshold=0.95)
    return {
        GHRS: ("Phase 1", base, True),
        "AE + FCM (Phase 2)": ("Phase 2", replace(base, cluster=ClusterConfig(method="fcm")), True),
        "ALS user and item biases": ("reference", replace(base, cluster=no_cluster, rec=RecConfig(
            baseline="user_item", similar="none", fallback="baseline", shrink=1e12)), False),
        SVDPP: ("synthesis", replace(base, cluster=no_cluster, rec=pure_offset), False),
        "SVD++ + K-Means residual": ("synthesis", replace(base, rec=residual), True),
        "SVD++ + FCM residual": ("synthesis", replace(base, cluster=ClusterConfig(method="fcm"), rec=residual), True),
        "SVD++ + VAE + FCM + TF-IDF residual": ("synthesis", replace(
            base, graph=tfidf, cluster=ClusterConfig(space="vae", method="fcm"), rec=residual), True),
    }


def run_protocol(ex: Experiment, variants, folds, seeds) -> pd.DataFrame:
    rows = []
    for name, (_, cfg, seeded) in variants.items():
        t0 = time.perf_counter()
        df = ex.run(cfg, folds, tuple(seeds) if seeded else (seeds[0],)).assign(variant=name)
        rows.append(df)
        log(f"{name:38s} RMSE={df['rmse'].mean():.4f}  ({time.perf_counter() - t0:.0f}s)")
    items = []
    for fold in folds:
        _, ci, ev = ex.rating_sources(fold, Protocol())
        pred = ev["MID"].map(ci.groupby("MID")["rate"].mean()).fillna(ci["rate"].mean()).to_numpy()
        items.append({"fold": fold, "seed": 0, "n_eval": len(ev), "variant": ITEM_MEAN,
                      **all_metrics(ev["UID"].to_numpy(), ev["rate"].to_numpy(), pred)})
    return pd.concat(rows + [pd.DataFrame(items)], ignore_index=True)


def accuracy_table(runs: pd.DataFrame, groups: dict[str, str]) -> pd.DataFrame:
    rows = []
    for name in dict.fromkeys(runs["variant"]):
        df = runs[runs["variant"] == name]
        s = summarize(df)
        per_seed = df.groupby("seed")["rmse"].mean()
        rows.append({"variant": name, "group": groups.get(name, "reference"), "rmse": s["rmse"],
                     "sd_seeds": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else float("nan"),
                     "mae": s["mae"], "precision": s["precision@3.5"], "recall": s["recall@3.5"]})
    return pd.DataFrame(rows)


def test_table(runs: pd.DataFrame, groups: dict[str, str], ratio: float) -> pd.DataFrame:
    """Per-fold paired comparisons with the GHRS reproduction and with plain SVD++."""
    per_fold = runs.groupby(["variant", "fold"])["rmse"].mean().unstack("variant")
    names = [n for n in per_fold.columns if n not in (GHRS, SVDPP, ITEM_MEAN)] + [SVDPP]
    rows, p_ghrs, p_svd = [], {}, {}
    for name in names:
        d_g = (per_fold[name] - per_fold[GHRS]).to_numpy()
        p_ghrs[name] = corrected_ttest(d_g, ratio)
        row = {"variant": name, "delta_vs_GHRS": d_g.mean(), "folds_better_vs_GHRS": f"{(d_g < 0).sum()}/{len(d_g)}"}
        if groups.get(name) == "synthesis" and name != SVDPP:
            d_s = (per_fold[name] - per_fold[SVDPP]).to_numpy()
            p_svd[name] = corrected_ttest(d_s, ratio)
            row.update(delta_vs_SVDpp=d_s.mean(), folds_better_vs_SVDpp=f"{(d_s < 0).sum()}/{len(d_s)}")
        rows.append(row)
    adj_g, adj_s = holm(p_ghrs), holm(p_svd)
    for r in rows:
        r["p_vs_GHRS"] = adj_g[r["variant"]]
        r["p_vs_SVDpp"] = adj_s.get(r["variant"], float("nan"))
    cols = ["variant", "delta_vs_GHRS", "p_vs_GHRS", "folds_better_vs_GHRS",
            "delta_vs_SVDpp", "p_vs_SVDpp", "folds_better_vs_SVDpp"]
    return pd.DataFrame(rows).reindex(columns=cols)


def label_permutation_audit(cfg: RunConfig, fold: str = "u1", seed: int = 42) -> dict:
    """Largest prediction change after shuffling the test labels, next to run-to-run noise.

    Each run uses a fresh cache, so nothing computed with the original labels is reused.
    """
    preds = []
    for permute in (False, False, True):
        with tempfile.TemporaryDirectory() as tmp:
            ex = Experiment(cache_dir=Path(tmp))
            if permute:
                train, test = ex.fold(fold)
                shuffled = test.copy()
                shuffled["rate"] = np.random.default_rng(123).permutation(test["rate"].to_numpy())
                ex._folds[fold] = (train, shuffled)
            preds.append(ex.predict_fold(fold, cfg, seed)[1])
    return {"noise": float(np.abs(preds[0] - preds[1]).max()), "change": float(np.abs(preds[0] - preds[2]).max())}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Final synthesis: GHRS clustering on top of SVD++")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1], help="AE/VAE and clustering seeds")
    ap.add_argument("--shrink", type=float, default=25.0,
                    help="shrinkage of the cluster residuals towards SVD++ (fixed in advance, not tuned)")
    ap.add_argument("--temporal-global", action="store_true", help="also run the mostly cold-start global split")
    ap.add_argument("--no-audit", action="store_true", help="skip the label-permutation audit")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--figs-dir", type=Path, default=None,
                    help="where to save figures 1 and 4 (default: figs/ for the default --out, else <out>/figs)")
    ap.add_argument("--fig-formats", nargs="+", default=["png", "pdf"], help="figure file formats")
    args = ap.parse_args()
    default_run = Path(args.out).resolve() == DEFAULT_OUT.resolve()
    args.report = args.report or (TRACKED_REPORT if default_run else args.out / "report.md")
    args.figs_dir = args.figs_dir or (FIGS_DIR if default_run else args.out / "figs")
    args.out.mkdir(parents=True, exist_ok=True)

    variants = build_variants(args.shrink)
    groups = {n: g for n, (g, _, _) in variants.items()}
    protocols = {"random 5-fold (u1-u5)": list(FOLDS_5CV), "temporal per user (last 20%)": ["temporal_user"]}
    if args.temporal_global:
        protocols["temporal global (latest 20%)"] = ["temporal_global"]

    ex = Experiment()
    accuracy = {}
    report = [
        "# Phase 3: synthesis with SVD++", "",
        "Generated by `scripts/run_final_synthesis.py`; rerun the script instead of editing this file.", "",
        f"MovieLens 100K. Clustering seeds {args.seeds}; SVD++ predictions averaged over seeds "
        f"{list(SVDppConfig().seeds)} (k = {SVDppConfig().k}, lambda = {SVDppConfig().lam}). Cluster residuals "
        f"(K = 8) are shrunk towards SVD++ with lambda = {args.shrink}, fixed in advance.", "",
    ]
    for label, folds in protocols.items():
        log(f"=== {label} ===")
        runs = run_protocol(ex, variants, folds, args.seeds)
        slug = folds[0] if len(folds) == 1 else "random_5fold"
        runs.to_csv(args.out / f"runs_{slug}.csv", index=False)
        acc = accuracy_table(runs, groups)
        acc.to_csv(args.out / f"accuracy_{slug}.csv", index=False)
        accuracy[slug] = acc
        report += [f"## {label}", "", markdown_table(acc), ""]
        if len(folds) > 1:
            ratio = float(np.mean([len(ex.fold(f)[1]) / len(ex.fold(f)[0]) for f in folds]))
            tests = test_table(runs, groups, ratio)
            tests.to_csv(args.out / f"tests_{slug}.csv", index=False)
            report += ["Per-fold paired differences in RMSE (negative = better); p: corrected resampled "
                       f"t-test (Nadeau & Bengio 2003, test/train ratio {ratio:.2f}), Holm-adjusted per column.",
                       "", markdown_table(tests), ""]
        else:
            report += ["A single split: no fold-level test. sd_seeds is the spread over clustering seeds.", ""]

    if not args.no_audit:
        log("=== label-permutation audit (fold u1) ===")
        quick = replace(SVDppConfig(), seeds=(42,))
        strict = replace(variants["SVD++ + FCM residual"][1], svdpp=quick)
        leaky = replace(strict, protocol=Protocol(ci_source="full"))
        audit = pd.DataFrame([{"configuration": name, **label_permutation_audit(cfg)} for name, cfg in (
            ("SVD++ + FCM residual, strict protocol", strict),
            ("SVD++ + FCM residual, test ratings in the cluster means (H3)", leaky))])
        audit["verdict"] = np.where(audit["change"] < 1e-4, "no dependence on test labels", "depends on test labels")
        audit.to_csv(args.out / "audit.csv", index=False)
        report += ["## Label-permutation audit (fold u1)", "",
                   "Test labels shuffled, predictions recomputed from scratch. 'noise' is the largest "
                   "difference between two identical runs, 'change' the largest difference after shuffling.",
                   "", markdown_table(audit), ""]

    # Figure 1 also shows VAE + K-Means and AE + GMM from the Phase 2 run, if it exists.
    phase2 = load_phase2_runs(OUTPUTS / "phase2", args.seeds)
    for path in make_synthesis_figures(args.figs_dir, accuracy, tuple(args.fig_formats), phase2):
        print(f"Saved: {path}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report), encoding="utf-8")
    print("\n" + "\n".join(report))
    print(f"Saved: {args.report}")


if __name__ == "__main__":
    main()
