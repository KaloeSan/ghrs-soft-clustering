"""Figures for the project: the dataset, the Phase 2 analyses and the final comparison.

- fig0_dataset_exploration: MovieLens 100K at a glance (needs no experiment results)
- fig1_rmse_comparison: the main result, from the GHRS baseline to SVD++ with the
  Phase 2 cluster residuals (scripts/run_final_synthesis.py)
- fig2_fuzziness, fig3_rmse_vs_k: AE + FCM against m and K (scripts/run_phase2.py)
- fig4_temporal_robustness: random 5-fold split against the temporal split
  (scripts/run_final_synthesis.py)

``scripts/run_phase2.py`` calls :func:`make_figures` and ``scripts/run_final_synthesis.py``
calls :func:`make_synthesis_figures` with the tables they have just computed; both save
to the git-tracked ``figs/`` directory. All figures can also be rebuilt from the CSV
files of earlier runs, without rerunning any experiment:

    python -m ghrs.figures

Figures 0, 1, 2 and 4 are drawn for the Polish defence slides, so their labels are in
Polish and numbers use a decimal comma; :func:`plot_fuzziness` also has an English
version (``lang="en"``). Figure 3 is not on the slides and keeps English labels. The Phase 1 reproduction is called "Baseline (AE + K-Means)" in every figure,
and bars carry only their RMSE value.

Sizes and fonts follow the Springer LNCS format: every figure fits the 12.2 cm text
width, text is set in a serif face at 7-8 pt so that it matches the body text when
the figure is included at its natural size, and PNGs are written at 600 dpi, the
resolution Springer asks for line art. Colours come from the Okabe-Ito palette,
which stays distinguishable for colour-blind readers and in greyscale print, and
every line also has its own marker.

PDFs embed their fonts as TrueType (Type 42) because publishers such as Springer
reject the Type 3 fonts matplotlib uses by default, and they carry no creation date,
so regenerating an unchanged figure does not show up as a change in git.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from .data import load_ratings
from .paths import FIGS, OUTPUTS

# Variant labels as written by scripts/run_phase2.py, and how the figures show them.
BASELINE = "Phase 1 baseline (AE + K-Means)"
ITEM_MEAN = "Reference: item mean"
BASELINE_LABEL = "Baseline (AE + K-Means)"

# Variant labels as written by scripts/run_final_synthesis.py.
SYN_GHRS = "GHRS reproduction (AE + K-Means)"
SYN_FCM = "AE + FCM (Phase 2)"
SYN_SVDPP = "SVD++ with GHRS features"
SYN_STACK = "SVD++ + VAE + FCM + TF-IDF residual"
SYNTHESIS_LABELS = {
    SYN_GHRS: "Baseline\n(AE + K-Means)",
    SYN_FCM: "AE + FCM",
    SYN_SVDPP: "SVD++\nz cechami GHRS",
    SYN_STACK: "SVD++ z rezyduami\nVAE + FCM + TF-IDF",
}
# Figure 1 adds two Phase 2 variants (from scripts/run_phase2.py) to the synthesis models.
PHASE2_IN_FIG1 = ("VAE + K-Means", "AE + GMM")
FIG1_ORDER = [SYN_GHRS, *PHASE2_IN_FIG1, SYN_FCM, SYN_SVDPP, SYN_STACK]
FIG1_LABELS = {
    SYN_GHRS: "Baseline\n(AE + K-Means)",
    "VAE + K-Means": "VAE +\nK-Means",
    "AE + GMM": "AE + GMM",
    SYN_FCM: "AE + FCM",
    SYN_SVDPP: "SVD++\nz cechami\nGHRS",
    SYN_STACK: "SVD++\nz rezyduami\nVAE + FCM + TF-IDF",
}
SYNTHESIS_PROTOCOLS = {
    "random_5fold": "Losowy 5-fold CV (u1-u5)",
    "temporal_user": "Podział temporalny\n(ostatnie 20% ocen użytkownika)",
}

FIGS_DIR = FIGS

TEXT_WIDTH_IN = 12.2 / 2.54
FIGSIZE = (TEXT_WIDTH_IN, 2.6)

COLORS = {
    BASELINE: "#595959",
    "VAE + K-Means": "#56B4E9",
    "AE + GMM": "#E69F00",
    "AE + FCM": "#009E73",
    SYN_GHRS: "#595959",
    SYN_FCM: "#009E73",
    SYN_SVDPP: "#0072B2",
    SYN_STACK: "#D55E00",
}
MARKERS = {BASELINE: "o", "AE + GMM": "s", "AE + FCM": "^"}
MEMBERSHIP_COLOR = "#CC79A7"
REFERENCE_COLOR = "#7F7F7F"

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "legend.frameon": False,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _new_figure() -> tuple[Figure, "matplotlib.axes.Axes"]:
    """Return a text-width figure with one set of axes.

    ``Figure`` is used directly instead of ``pyplot`` so that drawing needs no GUI
    backend and leaves no open figures behind in long experiment runs.
    """
    fig = Figure(figsize=FIGSIZE, layout="constrained")
    return fig, fig.subplots()


def _reference_rmse(comparison: pd.DataFrame, variant: str) -> float | None:
    """RMSE of ``variant`` in the comparison table, or None if it was not run."""
    rows = comparison.loc[comparison["variant"] == variant, "rmse"]
    return float(rows.iloc[0]) if len(rows) else None


FUZZINESS_TEXT = {
    "en": {"item_mean": "Item mean (reference)", "membership": "Mean max. membership",
           "x": "Fuzzifier $m$", "uniform": "uniform", "default": "main experiments"},
    "pl": {"item_mean": "Średnia ocen filmu (odniesienie)", "membership": "Średnia maks. przynależność",
           "x": "Współczynnik rozmycia $m$", "uniform": "wagi równe", "default": "główne eksperymenty"},
}


def plot_fuzziness(fuzziness: pd.DataFrame, comparison: pd.DataFrame, k: int = 8,
                   default_m: float = 2.0, lang: str = "pl") -> Figure:
    """AE + FCM against the fuzzifier m: RMSE on the left axis, max membership on the right.

    As m grows the memberships flatten (the mean largest weight falls towards 1/K),
    every user's prediction becomes an even mix of all cluster means, and the RMSE
    approaches that of the item mean. The baseline marks the hard-clustering end
    (m -> 1) and the item mean the fully smoothed end. A dashed vertical line marks the
    fuzzifier of the main experiments (``default_m``) and a dotted line on the right
    axis the uniform membership 1/K, the floor of the max-membership curve.

    ``lang`` is "pl" (the defence slides; decimal comma) or "en" (the paper).
    """
    text = FUZZINESS_TEXT[lang]
    num = (lambda v, d=3: _pl(v, d).rstrip("0").rstrip(",")) if lang == "pl" else (lambda v, d=3: f"{v:g}")
    fz = fuzziness.sort_values("m")
    m = fz["m"].to_numpy()

    fig, ax = _new_figure()
    handles = ax.plot(m, fz["rmse"], marker="^", color=COLORS["AE + FCM"], label="RMSE (AE + FCM)")
    for variant, style, label in ((BASELINE, ":", BASELINE_LABEL), (ITEM_MEAN, "--", text["item_mean"])):
        value = _reference_rmse(comparison, variant)
        if value is not None:
            handles.append(ax.axhline(value, color=REFERENCE_COLOR, linestyle=style, linewidth=0.9, label=label))
    ax.set_xlabel(text["x"])
    ax.set_ylabel("RMSE", color=COLORS["AE + FCM"])
    ax.tick_params(axis="y", colors=COLORS["AE + FCM"])
    ax.xaxis.set_major_locator(FixedLocator(m))
    ax.set_xticklabels([num(v) for v in m])

    right = ax.twinx()
    handles += right.plot(m, fz["max_membership"], marker="s", linestyle="--", color=MEMBERSHIP_COLOR,
                          label=text["membership"])
    right.set_ylim(0, 1.1)
    right.set_ylabel(text["membership"], color=MEMBERSHIP_COLOR)
    right.tick_params(axis="y", colors=MEMBERSHIP_COLOR)
    right.spines["right"].set_visible(True)
    right.grid(False)
    uniform = 1.0 / k
    right.axhline(uniform, color=MEMBERSHIP_COLOR, linestyle=":", linewidth=0.8, alpha=0.8)
    right.annotate(f"{text['uniform']}: $1/K$ = {num(uniform)}", (m[-1], uniform), xytext=(-2, 2),
                   textcoords="offset points", ha="right", va="bottom", fontsize=6.5, color=MEMBERSHIP_COLOR)

    # Headroom above the highest reference line, and the legend outside the axes: both
    # curves start near the top left and the reference lines span the full width.
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.08 * (high - low))
    if m.min() <= default_m <= m.max():
        ax.axvline(default_m, color=REFERENCE_COLOR, linestyle="--", linewidth=0.7, alpha=0.7, zorder=1)
        ax.annotate(f"{text['default']}: $m$ = {num(default_m)}", (default_m, 1.0),
                    xycoords=("data", "axes fraction"), xytext=(3, -3), textcoords="offset points",
                    ha="left", va="top", fontsize=6.5, color=REFERENCE_COLOR)
    if lang == "pl":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _pl(v, 2)))
        right.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _pl(v, 1)))
    fig.legend(handles=handles, loc="outside upper center", ncols=2)
    return fig


def plot_k_sweep(k_sweep: pd.DataFrame, comparison: pd.DataFrame) -> Figure:
    """RMSE against the number of clusters K (log scale) for hard and soft clustering.

    With hard labels each cluster mean rests on fewer users as K grows, so the
    baseline and GMM (whose memberships are nearly hard) degrade quickly. FCM
    spreads every user over all clusters and stays close to the item mean.
    """
    fig, ax = _new_figure()
    for variant in (BASELINE, "AE + GMM", "AE + FCM"):
        rows = k_sweep[k_sweep["variant"] == variant].sort_values("k")
        if len(rows):
            ax.plot(rows["k"], rows["rmse"], marker=MARKERS[variant], color=COLORS[variant],
                    label=BASELINE_LABEL if variant == BASELINE else variant)
    item_mean = _reference_rmse(comparison, ITEM_MEAN)
    if item_mean is not None:
        ax.axhline(item_mean, color=REFERENCE_COLOR, linestyle="--", linewidth=0.9, label="Item mean (reference)")

    ks = np.sort(k_sweep["k"].unique())
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(ks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Number of clusters $K$ (log scale)")
    ax.set_ylabel("RMSE")
    ax.legend(loc="upper left")
    return fig


def _pl(value: float, digits: int) -> str:
    """Number with a Polish decimal comma and a proper minus sign."""
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


def _polish_y_axis(ax) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _pl(v, 2)))


def _rmse_row(accuracy: pd.DataFrame, variant: str) -> tuple[float, float]:
    """RMSE and seed spread of ``variant``; the spread is 0 when only one seed was run."""
    row = accuracy.set_index("variant").loc[variant]
    sd = float(row["sd_seeds"])
    return float(row["rmse"]), sd if np.isfinite(sd) else 0.0


def phase2_accuracy(runs: pd.DataFrame, seeds) -> pd.DataFrame:
    """RMSE and seed spread of the baseline and the Phase 2 variants of figure 1.

    ``runs`` is outputs/phase2/runs.csv. Only the given seeds are used, so that the
    numbers are averaged exactly like the synthesis results they are drawn next to.
    """
    rows = []
    for variant in (BASELINE, *PHASE2_IN_FIG1):
        df = runs[(runs["variant"] == variant) & runs["seed"].isin(list(seeds))]
        if len(df):
            per_seed = df.groupby("seed")["rmse"].mean()
            rows.append({"variant": variant, "rmse": float(df["rmse"].mean()),
                         "sd_seeds": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else float("nan")})
    return pd.DataFrame(rows, columns=["variant", "rmse", "sd_seeds"])


def plot_rmse_comparison(accuracy: pd.DataFrame, phase2: pd.DataFrame | None = None) -> Figure:
    """The main result: RMSE from the GHRS baseline to SVD++ with the Phase 2 cluster residuals.

    ``accuracy`` is the random 5-fold accuracy table of scripts/run_final_synthesis.py;
    ``phase2`` (from :func:`phase2_accuracy`) adds VAE + K-Means and AE + GMM. The bars
    are separate models, not cumulative steps (SVD++ does not build on AE + FCM). Error
    bars show the spread over clustering seeds; SVD++ alone is already averaged over its
    seeds and has none. The y-axis does not start at zero.
    """
    table = accuracy if phase2 is None else pd.concat(
        [accuracy, phase2[phase2["variant"] != BASELINE]], ignore_index=True)
    order = [v for v in FIG1_ORDER if v in set(table["variant"])]
    rmse, sd = map(np.array, zip(*(_rmse_row(table, v) for v in order), strict=True))
    x = np.arange(len(order))

    fig, ax = _new_figure()
    ax.bar(x, rmse, width=0.6, color=[COLORS[v] for v in order], edgecolor="black", linewidth=0.5, zorder=2)
    ax.errorbar(x, rmse, yerr=sd, fmt="none", ecolor="black", elinewidth=0.6, capsize=2.5, capthick=0.6, zorder=3)
    for xi, value, err in zip(x, rmse, sd, strict=True):
        ax.annotate(_pl(value, 4), (xi, value + err), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5)

    ax.set_ylim(0.85, float(rmse.max()) + 0.03)
    ax.set_xticks(x, [FIG1_LABELS[v] for v in order])
    ax.tick_params(axis="x", length=0)
    ax.grid(False, axis="x")
    ax.set_ylabel("RMSE")
    _polish_y_axis(ax)
    return fig


def plot_temporal_robustness(accuracy: dict[str, pd.DataFrame]) -> Figure:
    """RMSE of three models on the random 5-fold split and on the per-user temporal split.

    ``accuracy`` maps the protocol keys of ``SYNTHESIS_PROTOCOLS`` to the accuracy
    tables of scripts/run_final_synthesis.py. Every model is worse on the temporal
    split, which predicts each user's latest ratings from the earlier ones, and the
    order of the models stays the same.
    """
    models = [SYN_GHRS, SYN_FCM, SYN_STACK]
    protocols = list(SYNTHESIS_PROTOCOLS)
    width = 0.24
    x = np.arange(len(protocols))

    fig, ax = _new_figure()
    for j, variant in enumerate(models):
        rmse, sd = map(np.array, zip(*(_rmse_row(accuracy[p], variant) for p in protocols), strict=True))
        pos = x + (j - 1) * width
        label = SYNTHESIS_LABELS[variant].replace("\n", " ")
        ax.bar(pos, rmse, width=width, color=COLORS[variant], edgecolor="black", linewidth=0.5,
               label=label, zorder=2)
        ax.errorbar(pos, rmse, yerr=sd, fmt="none", ecolor="black", elinewidth=0.6, capsize=2, capthick=0.6,
                    zorder=3)
        for p, value, err in zip(pos, rmse, sd, strict=True):
            ax.annotate(_pl(value, 3), (p, value + err), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5)

    ax.set_ylim(0.85, 1.16)
    ax.set_xticks(x, [SYNTHESIS_PROTOCOLS[p] for p in protocols])
    ax.tick_params(axis="x", length=0)
    ax.grid(False, axis="x")
    ax.set_ylabel("RMSE")
    _polish_y_axis(ax)
    fig.legend(loc="outside upper center", ncols=3)
    return fig


def plot_dataset_exploration(ratings: pd.DataFrame) -> Figure:
    """MovieLens 100K at a glance: the rating distribution and the long tail of activity.

    Left: the share of each rating value. Right: the number of ratings per user and per
    movie, each sorted in decreasing order and placed on a common 0-100% axis, on a log
    scale. Every user has at least 20 ratings, but many movies have only a handful, and
    the rating matrix is about 94% empty.
    """
    fig = Figure(figsize=FIGSIZE, layout="constrained")
    left, right = fig.subplots(1, 2, width_ratios=[1, 1.5])

    share = ratings["rate"].value_counts(normalize=True).sort_index()
    left.bar(share.index, 100 * share.to_numpy(), width=0.65, color="#0072B2", edgecolor="black",
             linewidth=0.5, zorder=2)
    for value, pct in zip(share.index, 100 * share.to_numpy(), strict=True):
        left.annotate(f"{_pl(pct, 1)}%", (value, pct), xytext=(0, 2), textcoords="offset points",
                      ha="center", va="bottom", fontsize=6.5)
    left.set_xticks(share.index)
    left.set_ylim(0, 100 * share.max() * 1.18)
    left.tick_params(axis="x", length=0)
    left.grid(False, axis="x")
    left.set_xlabel("Ocena")
    left.set_ylabel("Udział ocen [%]")
    left.set_title("Rozkład ocen", fontsize=8)

    n_users, n_items = ratings["UID"].nunique(), ratings["MID"].nunique()
    for key, color, marker, label in (("UID", "#D55E00", "o", "użytkownicy"), ("MID", "#009E73", "s", "filmy")):
        counts = np.sort(ratings.groupby(key).size().to_numpy())[::-1]
        x = 100 * np.arange(1, len(counts) + 1) / len(counts)
        median = float(np.median(counts))
        right.plot(x, counts, color=color, label=f"{label} (mediana {median:g})",
                   marker=marker, markevery=len(counts) // 6, zorder=2)
    right.set_yscale("log")
    right.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    right.set_xlim(0, 100)
    right.set_xlabel("Użytkownicy lub filmy, posortowane malejąco [%]")
    right.set_ylabel("Liczba ocen (skala log.)")
    right.set_title("Aktywność: długi ogon", fontsize=8)
    density = len(ratings) / (n_users * n_items)
    right.legend(loc="upper right", title=f"Wypełnienie macierzy: {_pl(100 * density, 1)}%",
                 title_fontsize=7)
    return fig


def _save(jobs, fig_dir: Path, formats: tuple[str, ...]) -> list[Path]:
    """Draw every (name, function) job in the house style and save it in each format."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with matplotlib.rc_context(STYLE):
        for name, draw in jobs:
            fig = draw()
            for ext in formats:
                path = fig_dir / f"{name}.{ext}"
                metadata = {"CreationDate": None} if ext == "pdf" else None
                fig.savefig(path, metadata=metadata)
                written.append(path)
    return written


def make_figures(fig_dir: Path, comparison: pd.DataFrame, fuzziness: pd.DataFrame | None = None,
                 k_sweep: pd.DataFrame | None = None, formats: tuple[str, ...] = ("png",),
                 k: int = 8, default_m: float = 2.0) -> list[Path]:
    """Draw the Phase 2 figures (fuzzifier and K sweeps) and return the paths written.

    ``comparison`` is the accuracy table (comparison.csv); it gives the reference lines.
    ``k`` and ``default_m`` are the number of clusters and the fuzzifier of the main
    run, marked in the fuzzifier figure. Each figure is drawn only when its table is
    given and not empty. Pass ``formats=("png", "pdf")`` to also get vector files for LaTeX.
    """
    jobs = []
    if fuzziness is not None and len(fuzziness):
        jobs.append(("fig2_fuzziness", lambda: plot_fuzziness(fuzziness, comparison, k, default_m)))
    if k_sweep is not None and len(k_sweep):
        jobs.append(("fig3_rmse_vs_k", lambda: plot_k_sweep(k_sweep, comparison)))
    return _save(jobs, fig_dir, formats)


def make_synthesis_figures(fig_dir: Path, accuracy: dict[str, pd.DataFrame],
                           formats: tuple[str, ...] = ("png",), phase2: pd.DataFrame | None = None) -> list[Path]:
    """Draw figures 1 and 4 from the accuracy tables of scripts/run_final_synthesis.py.

    ``accuracy`` maps "random_5fold" and, for figure 4, "temporal_user" to the tables
    saved as accuracy_<key>.csv. ``phase2`` (from :func:`phase2_accuracy`) adds the
    Phase 2 variants to figure 1, but only if its baseline equals the synthesis
    baseline, i.e. both were averaged over the same folds and seeds.
    """
    random = accuracy["random_5fold"]
    if phase2 is not None:
        base = phase2.loc[phase2["variant"] == BASELINE, "rmse"]
        if not len(base) or abs(float(base.iloc[0]) - _rmse_row(random, SYN_GHRS)[0]) > 1e-9:
            print("Phase 2 results do not match the synthesis baseline; figure 1 shows the synthesis models only")
            phase2 = None
    jobs = [("fig1_rmse_comparison", lambda: plot_rmse_comparison(random, phase2))]
    if all(p in accuracy for p in SYNTHESIS_PROTOCOLS):
        jobs.append(("fig4_temporal_robustness", lambda: plot_temporal_robustness(accuracy)))
    return _save(jobs, fig_dir, formats)


def make_dataset_figure(fig_dir: Path, ratings: pd.DataFrame | None = None,
                        formats: tuple[str, ...] = ("png",)) -> list[Path]:
    """Draw figure 0 from the full ratings file (``u.data`` unless ``ratings`` is given)."""
    ratings = load_ratings("u.data") if ratings is None else ratings
    return _save([("fig0_dataset_exploration", lambda: plot_dataset_exploration(ratings))], fig_dir, formats)


def load_phase2_runs(out_dir: Path, seeds) -> pd.DataFrame | None:
    """The Phase 2 rows of figure 1 from ``out_dir``/runs.csv, or None if there is no such file."""
    path = Path(out_dir) / "runs.csv"
    return phase2_accuracy(pd.read_csv(path), seeds) if path.exists() else None


def load_synthesis_results(out_dir: Path) -> dict[str, pd.DataFrame]:
    """Read the accuracy tables of the synthesis protocols that are present in ``out_dir``."""
    paths = {p: Path(out_dir) / f"accuracy_{p}.csv" for p in SYNTHESIS_PROTOCOLS}
    return {p: pd.read_csv(path) for p, path in paths.items() if path.exists()}


def load_results(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Read comparison.csv and, where present, fuzziness_sweep.csv and k_sweep.csv."""
    out_dir = Path(out_dir)

    def optional(name: str) -> pd.DataFrame | None:
        path = out_dir / name
        return pd.read_csv(path) if path.exists() else None

    return pd.read_csv(out_dir / "comparison.csv"), optional("fuzziness_sweep.csv"), optional("k_sweep.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild the dataset, Phase 2 and Phase 3 figures from saved results")
    ap.add_argument("results", type=Path, nargs="?", default=OUTPUTS / "phase2",
                    help="directory with comparison.csv and the sweep tables")
    ap.add_argument("--synthesis", type=Path, default=OUTPUTS / "synthesis",
                    help="directory with the accuracy_*.csv tables of run_final_synthesis.py")
    ap.add_argument("--figs-dir", type=Path, default=FIGS_DIR, help="where to save the figures")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"], help="file formats, e.g. png pdf")
    args = ap.parse_args()
    formats = tuple(args.formats)

    written = make_dataset_figure(args.figs_dir, formats=formats)
    if (args.results / "comparison.csv").exists():
        written += make_figures(args.figs_dir, *load_results(args.results), formats)
    else:
        print(f"No Phase 2 results in {args.results}; figures 2-3 skipped")
    synthesis = load_synthesis_results(args.synthesis)
    if "random_5fold" in synthesis:
        seeds = pd.read_csv(args.synthesis / "runs_random_5fold.csv")["seed"].unique()
        phase2 = load_phase2_runs(args.results, seeds)
        written += make_synthesis_figures(args.figs_dir, synthesis, formats, phase2)
    else:
        print(f"No synthesis results in {args.synthesis}; figures 1 and 4 skipped")
    for path in written:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
