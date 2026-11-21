"""Loaders for the MovieLens 100K files.

The original script parsed the files with regex separators and ``engine='python'``.
Single-character separators produce the same frames with the faster C parser.
"""

from pathlib import Path

import pandas as pd

from .paths import DATA_DIR

N_ITEMS_100K = 1682  # hard-coded as `i_no` in the authors' script
GENRE_COLS = [f"g{i}" for i in range(19)]


def load_ratings(split: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load a ratings file such as ``u1.base`` or ``ua.test`` (columns UID, MID, rate, time)."""
    return pd.read_csv(data_dir / split, sep="\t", names=["UID", "MID", "rate", "time"])


def load_users(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "u.user", sep="|", names=["UID", "age", "gender", "job", "zip"])


def load_release_years(data_dir: Path = DATA_DIR) -> pd.Series:
    """Release year of every movie, indexed by MID; missing dates get the median year."""
    items = pd.read_csv(data_dir / "u.item", sep="|", header=None, encoding="latin-1")
    years = pd.to_datetime(items[2], format="%d-%b-%Y", errors="coerce").dt.year
    return pd.Series(years.fillna(years.median()).to_numpy(), index=items[0].to_numpy(), name="year")


def load_item_genres(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """The 19 binary genre flags of every movie, indexed by MID."""
    cols = ["MID", "title", "release", "video_release", "url", *GENRE_COLS]
    items = pd.read_csv(data_dir / "u.item", sep="|", names=cols, encoding="latin-1")
    return items.set_index("MID")[GENRE_COLS]
