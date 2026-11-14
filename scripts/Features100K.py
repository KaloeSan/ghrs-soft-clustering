"""Steps 1-3 of GHRS: the authors' Features100K.py updated for Python 3.14.

Behaves like ``original/Features100K.py``: for every alpha it builds the user
similarity graph from ``ua.base`` and pickles the 943 x 35 feature matrix.

Usage (from the repository root):
    python -m scripts.Features100K [--split ua.base] [--out outputs/data100k]
"""

import argparse
import time
from pathlib import Path

from ghrs.data import load_ratings, load_users
from ghrs.features import build_feature_matrix, count_similar_pairs

ALPHA_COEFS = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="ua.base", help="training ratings file (default: ua.base, as authors)")
    ap.add_argument("--out", type=Path, default=Path("outputs/data100k"))
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHA_COEFS)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = load_ratings(args.split)
    df_user = load_users()
    counter = count_similar_pairs(df)  # does not depend on alpha, so it is computed once

    for alpha_coef in args.alphas:
        t0 = time.perf_counter()
        X_train = build_feature_matrix(df, df_user, alpha_coef, counter=counter)
        path = args.out / f"x_train_alpha({alpha_coef}).pkl"
        X_train.to_pickle(path)
        print(f"alpha={alpha_coef:<6} shape={X_train.shape} -> {path}  ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
