"""Repository locations used as defaults by the package and the scripts.

The paths are built from this file's location, so the scripts behave the same
whatever the current working directory is.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets" / "ml-100k"
AUTHORS_FEATURES = ROOT / "original" / "data100k"
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs"
FIGS = ROOT / "figs"
