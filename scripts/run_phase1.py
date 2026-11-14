"""Phase 1: the full GHRS pipeline on MovieLens 100K with 5-fold cross-validation (u1..u5).

For each fold the graph and side-information features are built from u{k}.base only,
then encoded by the autoencoder (35 to 4 dimensions), scaled, clustered with K-Means
(K=8), turned into cluster-item ratings by Algorithm 1 and evaluated on u{k}.test.
The global-mean and item-mean baselines are computed on the same folds for comparison.

Usage (from the repository root):
    python -m scripts.run_phase1                       # paper setting: alpha=0.01, K=8, seed 42
    python -m scripts.run_phase1 --seeds 1 2 3 4 5     # average over autoencoder/K-Means seeds
    python -m scripts.run_phase1 --select-k            # elbow and silhouette for K in [1, 30]
"""

import ghrs  # noqa: F401 - sets KERAS_BACKEND before keras is imported

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ghrs.autoencoder import AEConfig
from ghrs.clustering import elbow_and_silhouette, scale_codes
from ghrs.experiment import FOLDS_5CV, PAPER_TARGETS, ClusterConfig, Experiment, GraphConfig, RunConfig, summarize
from ghrs.metrics import rmse


def parse_args():
    d = AEConfig()
    ap = argparse.ArgumentParser(description="GHRS Phase 1 (ML-100K, 5-fold CV)")
    ap.add_argument("--alpha", type=float, default=0.01, help="paper optimum for ML-100K (Fig. 7)")
    ap.add_argument("--k", type=int, default=8, help="paper: Elbow K=8 for ML-100K (Table 2)")
    ap.add_argument("--folds", nargs="+", default=list(FOLDS_5CV))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--hidden", type=int, default=d.hidden_dim)
    ap.add_argument("--code", type=int, default=d.code_dim)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--batch", type=int, default=d.batch_size)
    ap.add_argument("--l1", type=float, default=d.l1)
    ap.add_argument("--l2", type=float, default=d.l2)
    ap.add_argument("--scale", default="minmax", choices=["minmax", "none", "standard"])
    ap.add_argument("--select-k", action="store_true", help="only run elbow/silhouette on the first fold")
    ap.add_argument("--out", type=Path, default=Path("outputs/results"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig(graph=GraphConfig(alpha=args.alpha),
                    ae=replace(AEConfig(), hidden_dim=args.hidden, code_dim=args.code, epochs=args.epochs,
                               batch_size=args.batch, l1=args.l1, l2=args.l2),
                    cluster=ClusterConfig(k=args.k, scale=args.scale))
    args.out.mkdir(parents=True, exist_ok=True)
    ex = Experiment()
    print(f"folds={args.folds}  seeds={args.seeds}\n{cfg.graph}\n{cfg.ae}\n{cfg.cluster}\n{cfg.rec}\n")

    if args.select_k:
        fold, seed = args.folds[0], args.seeds[0]
        F_e, _ = ex.codes(ex.features(fold, cfg.graph, cfg.protocol), cfg.ae, seed)
        res = pd.DataFrame(elbow_and_silhouette(scale_codes(F_e, args.scale), range(1, 31), seed))
        res.to_csv(args.out / f"select_k_{fold}_seed{seed}.csv", index=False)
        print(res.to_string(index=False, float_format="%.4f"))
        print(f"best silhouette K = {int(res.loc[res['silhouette'].idxmax(), 'k'])}")
        return

    rows = []
    for seed in args.seeds:
        for fold in args.folds:
            r = ex.run_fold(fold, cfg, seed)
            train, test = ex.fold(fold)
            r["rmse_global_mean"] = rmse(test["rate"], np.full(len(test), train["rate"].mean()))
            r["rmse_item_mean"] = rmse(test["rate"], test["MID"].map(train.groupby("MID")["rate"].mean())
                                       .fillna(train["rate"].mean()))
            rows.append(r)
            print(f"[{fold} seed={seed}] RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}  P={r['precision@3.5']:.3f} "
                  f"R={r['recall@3.5']:.3f}  (item-mean {r['rmse_item_mean']:.4f})  "
                  f"AE val_mse={r.get('ae_val_mse', float('nan')):.4f} active_dims={r.get('ae_active_code_dims')}",
                  flush=True)

    df = pd.DataFrame(rows)
    out_csv = args.out / f"phase1_alpha{args.alpha}_k{args.k}.csv"
    df.to_csv(out_csv, index=False)
    s = summarize(df)
    per_seed = df.groupby("seed")["rmse"].mean()
    print(f"\n=== GHRS ML-100K, {len(args.folds)}-fold CV, {len(args.seeds)} seed(s) ===")
    print(df.groupby("fold")[["rmse", "mae", "precision@3.5", "recall@3.5"]].mean().to_string(float_format="%.4f"))
    print(f"\nRMSE      = {s['rmse']:.4f}"
          + (f" +/- {per_seed.std(ddof=1):.4f} (sd over seeds)" if len(per_seed) > 1 else "")
          + f"   S^2 across folds = {s['rmse_S2']:.3e}   (paper: {PAPER_TARGETS['rmse']}, S^2 = 1.595e-4)")
    print(f"MAE       = {s['mae']:.4f}")
    print(f"Precision = {s['precision@3.5']:.4f}   (paper: {PAPER_TARGETS['precision']})")
    print(f"Recall    = {s['recall@3.5']:.4f}   (paper: {PAPER_TARGETS['recall']})")
    print(f"Baselines: global mean RMSE = {df['rmse_global_mean'].mean():.4f}, "
          f"item mean RMSE = {df['rmse_item_mean'].mean():.4f}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
