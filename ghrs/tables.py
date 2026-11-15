"""Plain-text Markdown tables for the experiment reports."""

import numpy as np
import pandas as pd


def _cell(v) -> str:
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        return f"{v:.2e}" if 0 < abs(v) < 1e-3 else f"{v:.4f}"
    return str(v)


def markdown_table(df: pd.DataFrame) -> str:
    """Format a data frame as a Markdown table whose columns line up in plain text.

    Every column is padded to its widest entry, so the table reads as a grid in the
    console and in a raw text editor. Numeric columns are right-aligned, which the
    ``---:`` marker also tells Markdown renderers; other columns are left-aligned.
    Floats get four decimals, very small values scientific notation, and NaN an
    empty cell.
    """
    header = [str(c) for c in df.columns]
    body = [[_cell(v) for v in row] for row in df.itertuples(index=False)]
    numeric = [pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes]
    widths = [max(3, len(h), *(len(r[i]) for r in body)) for i, h in enumerate(header)]

    def line(cells: list[str]) -> str:
        padded = [c.rjust(w) if num else c.ljust(w) for c, w, num in zip(cells, widths, numeric, strict=True)]
        return "| " + " | ".join(padded) + " |"

    rule = ["-" * (w - 1) + ":" if num else "-" * w for w, num in zip(widths, numeric, strict=True)]
    return "\n".join([line(header), line(rule)] + [line(r) for r in body])
