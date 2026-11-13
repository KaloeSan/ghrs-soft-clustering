"""Compare regenerated features with the authors' reference pickles (original/data100k).

Usage: python -m scripts.verify_features [--ours outputs/data100k] [--atol 1e-6]
Exit code 0 only if every alpha matches (same columns, same shape, |diff| <= atol).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, default=Path("original/data100k"))
    ap.add_argument("--ours", type=Path, default=Path("outputs/data100k"))
    ap.add_argument("--atol", type=float, default=1e-6)
    args = ap.parse_args()

    ok = True
    for ref_path in sorted(args.ref.glob("x_train_alpha(*).pkl")):
        ours_path = args.ours / ref_path.name
        if not ours_path.exists():
            print(f"MISSING  {ours_path}")
            ok = False
            continue
        ref, ours = pd.read_pickle(ref_path), pd.read_pickle(ours_path)
        if list(ref.columns) != list(ours.columns) or ref.shape != ours.shape:
            print(f"FAIL     {ref_path.name}: columns/shape differ {ref.shape} vs {ours.shape}")
            ok = False
            continue
        diff = np.abs(ref.to_numpy(float) - ours.to_numpy(float))
        per_col = pd.Series(diff.max(axis=0), index=ref.columns)
        worst = per_col.idxmax()
        status = "OK  " if diff.max() <= args.atol else "FAIL"
        ok &= status == "OK  "
        print(f"{status}     {ref_path.name}: max|diff|={diff.max():.2e} (worst column: {worst})")
    print("ALL MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
