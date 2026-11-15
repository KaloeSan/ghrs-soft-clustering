"""Publication figures for the Phase 2 results.

``scripts/run_phase2.py`` calls :func:`make_figures` with the tables it has just
computed and saves the figures to the git-tracked ``figs/`` directory. They can also
be rebuilt from the CSV files of an earlier run, without rerunning any experiment:

    python -m ghrs.figures

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
from matplotlib.ticker import FixedLocator, NullLocator

from .paths import FIGS, OUTPUTS

# Variant labels as written by scripts/run_phase2.py.
BASELINE = "Phase 1 baseline (AE + K-Means)"
ITEM_MEAN = "Reference: item mean"

FIGS_DIR = FIGS

TEXT_WIDTH_IN = 12.2 / 2.54
FIGSIZE = (TEXT_WIDTH_IN, 2.6)

COLORS = {
    BASELINE: "#595959",
    "VAE + K-Means": "#56B4E9",
    "AE + GMM": "#E69F00",
    "AE + FCM": "#009E73",
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


def plot_rmse_comparison(comparison: pd.DataFrame) -> Figure:
    """Bar chart of the RMSE of the Phase 1 baseline and three Phase 2 architectures.

    Error bars show the standard deviation of the overall RMSE over seeds
    (``sd_seeds``). The differences are a few hundredths of an RMSE point, so the
    y-axis starts just below the lowest bar rather than at zero; say so in the
    figure caption.
    """
    order = [BASELINE, "VAE + K-Means", "AE + GMM", "AE + FCM"]
    labels = {BASELINE: "Phase 1 baseline\n(AE + K-Means)"}
    rows = comparison.set_index("variant").reindex([n for n in order if n in set(comparison["variant"])])

    x = np.arange(len(rows))
    rmse = rows["rmse"].to_numpy()
    sd = rows["sd_seeds"].to_numpy(dtype=float)
    has_sd = np.isfinite(sd)

    fig, ax = _new_figure()
    ax.bar(x, rmse, width=0.6, color=[COLORS[n] for n in rows.index],
           edgecolor="black", linewidth=0.5, zorder=2)
    ax.errorbar(x[has_sd], rmse[has_sd], yerr=sd[has_sd], fmt="none",
                ecolor="black", elinewidth=0.6, capsize=2.5, capthick=0.6, zorder=3)

    for xi, value, err in zip(x, rmse, np.where(has_sd, sd, 0.0), strict=True):
        ax.annotate(f"{value:.4f}", (xi, value + err), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5)

    top = float(np.max(rmse + np.where(has_sd, sd, 0.0)))
    span = top - float(rmse.min())
    ax.set_ylim(np.floor((rmse.min() - 0.3 * span) * 100) / 100, top + 0.25 * span)
    ax.set_xticks(x, [labels.get(n, n) for n in rows.index])
    ax.tick_params(axis="x", length=0)
    ax.grid(False, axis="x")
    ax.set_ylabel("RMSE")
    return fig


def plot_fuzziness(fuzziness: pd.DataFrame, comparison: pd.DataFrame) -> Figure:
    """AE + FCM against the fuzzifier m: RMSE on the left axis, max membership on the right.

    As m grows the memberships flatten (the mean largest weight falls towards 1/K),
    every user's prediction becomes an even mix of all cluster means, and the RMSE
    approaches that of the item mean. The baseline marks the hard-clustering end
    (m -> 1) and the item mean the fully smoothed end.
    """
    fz = fuzziness.sort_values("m")
    m = fz["m"].to_numpy()

    fig, ax = _new_figure()
    handles = ax.plot(m, fz["rmse"], marker="^", color=COLORS["AE + FCM"], label="RMSE (AE + FCM)")
    for variant, style, label in ((BASELINE, ":", "Phase 1 baseline (AE + K-Means)"),
                                  (ITEM_MEAN, "--", "Item mean (reference)")):
        value = _reference_rmse(comparison, variant)
        if value is not None:
            handles.append(ax.axhline(value, color=REFERENCE_COLOR, linestyle=style, linewidth=0.9, label=label))
    ax.set_xlabel("Fuzzifier $m$")
    ax.set_ylabel("RMSE", color=COLORS["AE + FCM"])
    ax.tick_params(axis="y", colors=COLORS["AE + FCM"])
    ax.xaxis.set_major_locator(FixedLocator(m))
    ax.set_xticklabels([f"{v:g}" for v in m])

    right = ax.twinx()
    handles += right.plot(m, fz["max_membership"], marker="s", linestyle="--", color=MEMBERSHIP_COLOR,
                          label="Mean max. membership")
    right.set_ylim(0, 1.1)
    right.set_ylabel("Mean max. membership", color=MEMBERSHIP_COLOR)
    right.tick_params(axis="y", colors=MEMBERSHIP_COLOR)
    right.spines["right"].set_visible(True)
    right.grid(False)

    # Headroom above the highest reference line, and the legend outside the axes: both
    # curves start near the top left and the reference lines span the full width.
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.08 * (high - low))
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
            ax.plot(rows["k"], rows["rmse"], marker=MARKERS[variant], color=COLORS[variant], label=variant)
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


def make_figures(fig_dir: Path, comparison: pd.DataFrame, fuzziness: pd.DataFrame | None = None,
                 k_sweep: pd.DataFrame | None = None, formats: tuple[str, ...] = ("png",)) -> list[Path]:
    """Draw the Phase 2 figures and return the paths written.

    ``comparison`` is the accuracy table (comparison.csv). The fuzziness and K-sweep
    figures are drawn only when their tables are given and not empty. Pass
    ``formats=("png", "pdf")`` to also get vector files for LaTeX.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    jobs = [("fig1_rmse_comparison", lambda: plot_rmse_comparison(comparison))]
    if fuzziness is not None and len(fuzziness):
        jobs.append(("fig2_fuzziness", lambda: plot_fuzziness(fuzziness, comparison)))
    if k_sweep is not None and len(k_sweep):
        jobs.append(("fig3_rmse_vs_k", lambda: plot_k_sweep(k_sweep, comparison)))

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


def load_results(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Read comparison.csv and, where present, fuzziness_sweep.csv and k_sweep.csv."""
    out_dir = Path(out_dir)

    def optional(name: str) -> pd.DataFrame | None:
        path = out_dir / name
        return pd.read_csv(path) if path.exists() else None

    return pd.read_csv(out_dir / "comparison.csv"), optional("fuzziness_sweep.csv"), optional("k_sweep.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild the Phase 2 figures from saved results")
    ap.add_argument("results", type=Path, nargs="?", default=OUTPUTS / "phase2",
                    help="directory with comparison.csv and the sweep tables")
    ap.add_argument("--figs-dir", type=Path, default=FIGS_DIR, help="where to save the figures")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"], help="file formats, e.g. png pdf")
    args = ap.parse_args()

    comparison, fuzziness, k_sweep = load_results(args.results)
    for path in make_figures(args.figs_dir, comparison, fuzziness, k_sweep, tuple(args.formats)):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
