"""Leakage simulator: can an evaluation mistake explain the paper's RMSE of 0.887?

Each hypothesis runs the same pipeline (``ghrs.experiment``) with one protocol setting
changed. It is compared with everything the paper reports, not only the RMSE:

- Table 3: RMSE 0.887, S^2 = 1.595e-4 over 5 folds, precision 0.771, recall 0.799
- Fig. 7: RMSE as a function of alpha (0.887 at 0.01, about 1.29 at 0.045)
- Fig. 13: RMSE as a function of the share of users without ratings (about 1.095 at 20%)

A hypothesis that reaches 0.887 but does not reproduce the curves does not explain
the result.

Hypotheses (see ``Protocol`` in ghrs/experiment.py):

    H0   strict            graph, autoencoder, K-Means and CI use the training fold only
    H1   graph leak        similarity graph built from training and test ratings
    H1b  authors' pickles  the authors' feature files (built once from ua.base) reused
                           for every fold u1..u5
    H2   AE/K-Means leak   not a separate case: both only see the 35 user features, and
                           every user is in every training fold, so test ratings can
                           only reach them through the graph (same as H1)
    H3   target leak       cluster-item means computed from training and test ratings
    H4   in-sample         RMSE measured on the training ratings
    H5   combinations      H1+H3, H1b+H3, H1+H4
    H6   metric variants   MAE, Eq. 6 as printed, per-user RMSE and rounded predictions,
                           reported as extra columns
    H7   other split       ua/ub (10 test ratings per user) instead of u1..u5
    H8   single build      steps 1-6 run once on one base file and scored on every
                           u_k.test. ua.base contains 78-100% of each u_k.test (H8a),
                           u1.base contains all of u2..u5.test (H8b). The leak differs
                           between folds, so S^2 is the deciding number here.
    H9   ua/ub swap        train on ua.base and test on ub.test, and the other way
                           round; ub.test is a subset of ua.base

Usage (from the repository root):
    python -m scripts.leakage_simulator                    # all sections, seed 42
    python -m scripts.leakage_simulator --seeds 1 2 3      # average over seeds
    python -m scripts.leakage_simulator --sections table ksweep

Results are written to outputs/leakage/*.csv and outputs/leakage/report.md.
"""

import ghrs  # noqa: F401 - sets KERAS_BACKEND before keras is imported

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ghrs.experiment import FOLDS_5CV, PAPER_TARGETS, Experiment, GraphConfig, Protocol, RunConfig, summarize

# Values read off the paper's figures.
FIG7_PAPER = {0.005: 1.02, 0.01: 0.887, 0.015: 0.99, 0.02: 1.05, 0.025: 1.11,
              0.03: 1.10, 0.035: 1.15, 0.04: 1.27, 0.045: 1.29}
FIG13_PAPER = {0.0: 0.887, 0.05: 0.90, 0.10: 0.96, 0.15: 1.05, 0.20: 1.095}

STRICT = Protocol()
HYPOTHESES = {
    "H0 strict": (STRICT, FOLDS_5CV),
    "H1 graph leak (train+test graph)": (Protocol(graph_source="full"), FOLDS_5CV),
    "H1b authors' ua.base pickles reused": (Protocol(graph_source="authors_ua"), FOLDS_5CV),
    "H3 target leak (CI from train+test)": (Protocol(ci_source="full"), FOLDS_5CV),
    "H4 in-sample evaluation": (Protocol(eval_on="train"), FOLDS_5CV),
    "H5a H1 + H3": (Protocol(graph_source="full", ci_source="full"), FOLDS_5CV),
    "H5b H1b + H3": (Protocol(graph_source="authors_ua", ci_source="full"), FOLDS_5CV),
    "H5c H1 + H4": (Protocol(graph_source="full", eval_on="train"), FOLDS_5CV),
    "H7 strict on ua/ub": (STRICT, ("ua", "ub")),
    "H7b H3 on ua/ub": (Protocol(ci_source="full"), ("ua", "ub")),
    "H8a single build on ua.base": (Protocol(fixed_base="ua"), FOLDS_5CV),
    "H8b single build on u1.base": (Protocol(fixed_base="u1"), FOLDS_5CV),
    "H9 ua/ub file swap": (STRICT, ("ua.base:ub.test", "ub.base:ua.test")),
}
CURVE_HYPOTHESES = ["H0 strict", "H1 graph leak (train+test graph)",
                    "H3 target leak (CI from train+test)", "H4 in-sample evaluation", "H5a H1 + H3",
                    "H8a single build on ua.base"]
K_GRID = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64]
# The K at which each leaky variant reaches about 0.887 (K sweep, alpha=0.01, seed 42).
# The curves are computed both at the base K and at this K, because a hypothesis only
# explains the paper if one configuration matches Table 3 and the figures together.
K_MATCH = {"H3 target leak (CI from train+test)": 20, "H4 in-sample evaluation": 20,
           "H5a H1 + H3": 24, "H8a single build on ua.base": 32}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def section_table(ex: Experiment, base: RunConfig, seeds) -> pd.DataFrame:
    rows = []
    for name, (proto, folds) in HYPOTHESES.items():
        s = summarize(ex.run(replace(base, protocol=proto), folds, seeds))
        rows.append({"hypothesis": name, **s})
        log(f"{name:40s} RMSE={s['rmse']:.4f} MAE={s['mae']:.4f} P={s['precision@3.5']:.3f} "
            f"R={s['recall@3.5']:.3f} S2={s['rmse_S2']:.2e}")
    return pd.DataFrame(rows)


def section_ksweep(ex: Experiment, base: RunConfig, seeds) -> pd.DataFrame:
    rows = []
    for name in ["H0 strict", "H3 target leak (CI from train+test)", "H4 in-sample evaluation", "H5a H1 + H3",
                 "H8a single build on ua.base", "H9 ua/ub file swap"]:
        proto, folds = HYPOTHESES[name]
        for k in K_GRID:
            cfg = replace(base, protocol=proto, cluster=replace(base.cluster, k=k))
            s = summarize(ex.run(cfg, folds, seeds))
            rows.append({"hypothesis": name, "k": k, **s})
        sub = pd.DataFrame([r for r in rows if r["hypothesis"] == name])
        log(f"K-sweep {name:38s} " + " ".join(f"K{r.k}:{r.rmse:.3f}" for r in sub.itertuples()))
    return pd.DataFrame(rows)


def curve_variants(base: RunConfig):
    """Yield (label, folds, config) for each curve: at the base K and at the matching K."""
    for name in CURVE_HYPOTHESES:
        proto, folds = HYPOTHESES[name]
        for k in dict.fromkeys([base.cluster.k, K_MATCH.get(name, base.cluster.k)]):
            yield f"{name} [K={k}]", folds, replace(base, protocol=proto, cluster=replace(base.cluster, k=k))


def section_fig7(ex: Experiment, base: RunConfig, seeds) -> pd.DataFrame:
    rows = []
    for label, folds, cfg0 in curve_variants(base):
        for alpha in FIG7_PAPER:
            cfg = replace(cfg0, graph=replace(cfg0.graph, alpha=alpha))
            s = summarize(ex.run(cfg, folds, seeds))
            rows.append({"hypothesis": label, "alpha": alpha, "paper_rmse": FIG7_PAPER[alpha], **s})
        sub = pd.DataFrame([r for r in rows if r["hypothesis"] == label])
        log(f"Fig.7 {label:46s} " + " ".join(f"{r.alpha}:{r.rmse:.3f}" for r in sub.itertuples()))
    return pd.DataFrame(rows)


def section_fig13(ex: Experiment, base: RunConfig, seeds) -> pd.DataFrame:
    rows = []
    for label, folds, cfg0 in curve_variants(base):
        for frac in FIG13_PAPER:
            # Three random choices of the removed users, averaged to smooth the curve.
            dfs = [ex.run(replace(cfg0, protocol=replace(cfg0.protocol, cold_frac=frac, cold_seed=cs)), folds, seeds)
                   for cs in ((0,) if frac == 0 else (0, 1, 2))]
            s = summarize(pd.concat(dfs).groupby(["fold", "seed"], as_index=False).mean(numeric_only=True))
            rows.append({"hypothesis": label, "cold_frac": frac, "paper_rmse": FIG13_PAPER[frac], **s})
        sub = pd.DataFrame([r for r in rows if r["hypothesis"] == label])
        log(f"Fig.13 {label:45s} " + " ".join(f"{r.cold_frac:.2f}:{r.rmse:.3f}" for r in sub.itertuples()))
    return pd.DataFrame(rows)


def curve_fit(df: pd.DataFrame, x: str) -> pd.DataFrame:
    """Compare the shape of each simulated curve with the paper's curve.

    The level is already compared in Table 3, so this looks at the range of each curve,
    its correlation with the paper's curve, and the RMSE between the two curves after
    removing their means. The plain RMSE between the curves is included as well.
    """
    out = []
    for name, g in df.groupby("hypothesis", sort=False):
        ours, paper = g["rmse"].to_numpy(), g["paper_rmse"].to_numpy()
        out.append({"hypothesis": name,
                    "our_range": float(np.ptp(ours)), "paper_range": float(np.ptp(paper)),
                    "corr_with_paper": float(np.corrcoef(ours, paper)[0, 1]),
                    "shape_rmse": float(np.sqrt(np.mean(((ours - ours.mean()) - (paper - paper.mean())) ** 2))),
                    "raw_curve_rmse": float(np.sqrt(np.mean((ours - paper) ** 2))),
                    f"our_argmin_{x}": float(g.loc[g["rmse"].idxmin(), x])})
    return pd.DataFrame(out)


def md_table(df: pd.DataFrame) -> str:
    """Format a data frame as a Markdown table (avoids the optional `tabulate` dependency)."""
    cell = lambda v: f"{v:.4g}" if isinstance(v, float) and abs(v) < 1e-3 and v else (  # noqa: E731
        f"{v:.4f}" if isinstance(v, float) else str(v))
    rows = [list(df.columns), ["---"] * df.shape[1]] + [[cell(v) for v in r] for r in df.itertuples(index=False)]
    return "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)


def write_report(out: Path, results: dict[str, pd.DataFrame], base: RunConfig) -> None:
    fmt = md_table
    a, k0 = base.graph.alpha, base.cluster.k
    lines = ["# Leakage simulator report", "",
             f"Paper targets (Table 3): {PAPER_TARGETS}", ""]
    if "table" in results:
        t = results["table"]
        cols = ["hypothesis", "rmse", "rmse_S2", "precision@3.5", "recall@3.5", "maxP@R>=0.799", "mae",
                "per_user_rmse", "eq6_literal", "rmse_rounded", "dist_to_paper"]
        lines += [f"## Hypothesis table (alpha={a}, K={k0})", "", fmt(t[cols]), "",
                  "H2 (AE / K-Means fitted on the whole dataset) has no channel of its own: both "
                  "only consume the 943x35 user feature matrix, and all 943 users are present in "
                  "every training fold. Test ratings can only reach them through the graph, which "
                  "is exactly H1 / H1b.", ""]
    if "ksweep" in results:
        k = results["ksweep"]
        lines += [f"## K sweep (alpha={a})", "",
                  fmt(k[["hypothesis", "k", "rmse", "rmse_S2", "precision@3.5", "recall@3.5", "maxP@R>=0.799"]]), ""]
        for name, g in k.groupby("hypothesis", sort=False):
            hit = g[g["rmse"] <= PAPER_TARGETS["rmse"]]
            lines.append(f"- {name}: " + (f"first K with RMSE <= 0.887: K={int(hit['k'].iloc[0])}"
                                          if len(hit) else f"never reaches 0.887 (min {g['rmse'].min():.4f})"))
        lines.append("")
    for sec, x in (("fig7", "alpha"), ("fig13", "cold_frac")):
        if sec in results:
            lines += [f"## {sec.upper()} replication (K given per row)", "", fmt(curve_fit(results[sec], x)), ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="GHRS leakage simulator")
    ap.add_argument("--sections", nargs="+", default=["table", "ksweep", "fig7", "fig13"],
                    choices=["table", "ksweep", "fig7", "fig13"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("outputs/leakage"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ex = Experiment()
    base = RunConfig(graph=GraphConfig(alpha=args.alpha))
    base = replace(base, cluster=replace(base.cluster, k=args.k))
    fns = {"table": section_table, "ksweep": section_ksweep, "fig7": section_fig7, "fig13": section_fig13}
    results = {}
    for sec in args.sections:
        log(f"=== section {sec} ===")
        results[sec] = fns[sec](ex, base, tuple(args.seeds))
        results[sec].to_csv(args.out / f"{sec}.csv", index=False)
        write_report(args.out, results, base)  # rewritten after each section, so partial runs leave a report
    log(f"done -> {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
