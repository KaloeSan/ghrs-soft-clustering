"""Phase 1: end-to-end GHRS on MovieLens 100K with 5-fold CV (u1..u5), as in the paper.

Per fold: graph + side-info features from u{k}.base only (no test leakage)
-> autoencoder (35 -> 4) -> K-Means (K=8) -> cluster-item ratings -> RMSE on u{k}.test.
Baselines (global mean, item mean) are reported on the same folds for context.

Usage (from the repo root):
    python -m scripts.run_phase1                       # paper setting: alpha=0.01, K=8
    python -m scripts.run_phase1 --seeds 1 2 3 4 5     # average over AE/K-Means seeds
    python -m scripts.run_phase1 --select-k            # elbow + silhouette, K in [1, 30]
"""

import ghrs  # noqa: F401  (sets KERAS_BACKEND before keras is imported)

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ghrs.autoencoder import AEConfig, encode_users
from ghrs.clustering import elbow_and_silhouette, kmeans_clusters
from ghrs.data import load_item_genres, load_ratings, load_users
from ghrs.features import build_feature_matrix
from ghrs.metrics import precision_recall, rmse
from ghrs.recommender import ClusterRatingModel


def parse_args():
    d = AEConfig()
    ap = argparse.ArgumentParser(description="GHRS Phase 1 (ML-100K, 5-fold CV)")
    ap.add_argument("--alpha", type=float, default=0.01, help="paper optimum for ML-100K (Fig. 7)")
    ap.add_argument("--k", type=int, default=8, help="paper: Elbow K=8 for ML-100K (Table 2)")
    ap.add_argument("--folds", nargs="+", default=["u1", "u2", "u3", "u4", "u5"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--hidden", type=int, default=d.hidden_dim)
    ap.add_argument("--code", type=int, default=d.code_dim)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--l1", type=float, default=d.l1)
    ap.add_argument("--l2", type=float, default=d.l2)
    ap.add_argument("--select-k", action="store_true", help="only run elbow/silhouette on fold 1")
    ap.add_argument("--out", type=Path, default=Path("outputs/results"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AEConfig(hidden_dim=args.hidden, code_dim=args.code, epochs=args.epochs, l1=args.l1, l2=args.l2)
    args.out.mkdir(parents=True, exist_ok=True)
    users, genres = load_users(), load_item_genres()
    uids = users["UID"].to_numpy()
    print(f"alpha={args.alpha}  K={args.k}  folds={args.folds}  seeds={args.seeds}\n{cfg}\n")

    rows = []
    for fold in args.folds:
        train, test = load_ratings(f"{fold}.base"), load_ratings(f"{fold}.test")
        t0 = time.perf_counter()
        F_t = build_feature_matrix(train, users, args.alpha)          # steps 1-3
        t_feat = time.perf_counter() - t0

        for seed in args.seeds:
            t0 = time.perf_counter()
            F_e, ae_info = encode_users(F_t.to_numpy(), cfg, seed)      # step 4

            if args.select_k:
                res = pd.DataFrame(elbow_and_silhouette(F_e, range(1, 31), seed))
                res.to_csv(args.out / f"select_k_{fold}_seed{seed}.csv", index=False)
                print(res.to_string(index=False, float_format="%.4f"))
                print(f"best silhouette K = {int(res.loc[res['silhouette'].idxmax(), 'k'])}")
                return

            labels = kmeans_clusters(F_e, args.k, seed)                  # step 5
            model = ClusterRatingModel().fit(train, pd.Series(labels, index=uids), genres)
            pred, tiers = model.predict(test)                             # steps 6-7
            p, r = precision_recall(test["rate"], pred)
            row = {
                "fold": fold, "seed": seed, "rmse": rmse(test["rate"], pred),
                "precision": p, "recall": r,
                "rmse_global_mean": rmse(test["rate"], np.full(len(test), train["rate"].mean())),
                "rmse_item_mean": rmse(test["rate"], test["MID"].map(train.groupby("MID")["rate"].mean())
                                       .fillna(train["rate"].mean())),
                **ae_info, **{f"tier_{k}": int(v) for k, v in tiers.items()},
                "cluster_sizes": np.bincount(labels, minlength=args.k).tolist(),
                "t_features_s": round(t_feat, 1), "t_model_s": round(time.perf_counter() - t0, 1),
            }
            rows.append(row)
            print(f"[{fold} seed={seed}] RMSE={row['rmse']:.4f}  P={p:.3f} R={r:.3f}  "
                  f"(item-mean {row['rmse_item_mean']:.4f})  AE val_loss={ae_info['val_loss']:.4f} "
                  f"active_dims={ae_info['active_code_dims']}  sizes={row['cluster_sizes']}")

    df = pd.DataFrame(rows)
    out_csv = args.out / f"phase1_alpha{args.alpha}_k{args.k}.csv"
    df.to_csv(out_csv, index=False)
    per_fold = df.groupby("fold")[["rmse", "precision", "recall"]].mean()
    print("\n=== GHRS ML-100K, 5-fold CV ===")
    print(per_fold.to_string(float_format="%.4f"))
    print(f"\nRMSE      = {per_fold['rmse'].mean():.4f}   S^2 across folds = {per_fold['rmse'].var(ddof=1):.3e}"
          f"   (paper: 0.887, S^2 = 1.595e-4)")
    print(f"Precision = {per_fold['precision'].mean():.4f}   (paper: 0.771)")
    print(f"Recall    = {per_fold['recall'].mean():.4f}   (paper: 0.799)")
    print(f"Baselines: global mean RMSE = {df['rmse_global_mean'].mean():.4f}, "
          f"item mean RMSE = {df['rmse_item_mean'].mean():.4f}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
