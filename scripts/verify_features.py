"""Compare regenerated features with the authors' pickles in ``original/data100k``.

Usage:
    python -m scripts.verify_features          # check outputs/data100k (from scripts.build_features_100k)
    python -m scripts.verify_features --fast   # check the vectorised implementation instead

The exit code is 0 only if every alpha matches (same columns, same shape, |diff| <= atol).
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ghrs.paths import AUTHORS_FEATURES, OUTPUTS


def compare(name: str, ref: pd.DataFrame, ours: pd.DataFrame, atol: float) -> bool:
    if list(ref.columns) != list(ours.columns) or ref.shape != ours.shape:
        print(f"FAIL     {name}: columns/shape differ {ref.shape} vs {ours.shape}")
        return False
    diff = np.abs(ref.to_numpy(float) - ours.to_numpy(float))
    worst = ref.columns[int(diff.max(axis=0).argmax())]
    ok = diff.max() <= atol
    print(f"{'OK  ' if ok else 'FAIL'}     {name}: max|diff|={diff.max():.2e} (worst column: {worst})")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, default=AUTHORS_FEATURES)
    ap.add_argument("--ours", type=Path, default=OUTPUTS / "data100k")
    ap.add_argument("--fast", action="store_true", help="verify the vectorised feature path instead")
    ap.add_argument("--atol", type=float, default=1e-6)
    args = ap.parse_args()

    if args.fast:
        from ghrs.data import load_ratings, load_users
        from ghrs.features import build_feature_matrix_fast, co_rating_matrices
        users, train = load_users(), load_ratings("ua.base")
        mats = co_rating_matrices(train, int(users["UID"].max()))

    ok = True
    for ref_path in sorted(args.ref.glob("x_train_alpha(*).pkl")):
        ref = pd.read_pickle(ref_path)
        if args.fast:
            alpha = float(re.search(r"\(([\d.]+)\)", ref_path.name).group(1))
            ours = build_feature_matrix_fast(train, users, alpha, mats=mats)
        else:
            ours_path = args.ours / ref_path.name
            if not ours_path.exists():
                print(f"MISSING  {ours_path}")
                ok = False
                continue
            ours = pd.read_pickle(ours_path)
        ok &= compare(ref_path.name, ref, ours, args.atol)
    print("ALL MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
