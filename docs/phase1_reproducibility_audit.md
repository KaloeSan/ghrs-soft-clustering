# GHRS Phase 1 — reproducibility audit (ML-100K)

Paper: Darban & Valipour, *GHRS: Graph-based Hybrid Recommendation System with Application to
Movie Recommendation* (arXiv:2111.11293). Claimed on ML-100K (Table 3): RMSE 0.887
(S² = 1.595e-4 over 5 folds), precision 0.771, recall 0.799.

**Bottom line.** Under a correct protocol the method gives **RMSE 1.076 ± 0.001**, and nothing
we tried closes the gap. We tested the AE design, every unspecified hyperparameter, alternative
readings of Algorithm 1, an oracle search over user partitions, and 13 leakage and
protocol-error hypotheses. Target leakage (test ratings inside the cluster-item means) is
the only mechanism that reaches 0.887. Even it needs K ≈ 20–24 instead of the paper's K = 8,
and it does not reproduce the paper's precision/recall, Fig. 7 or Fig. 13 at the same time.
The published numbers are not consistent with any single computation we could construct.

All numbers below come from 5-fold CV on u1..u5 with α = 0.01 and K = 8, unless stated
otherwise. Scripts: `scripts/run_phase1.py`, `scripts/leakage_simulator.py`,
`scripts/optimize_phase1.py`, `scripts/partition_bound.py`. Results are in `outputs/`.

## 1. What the authors released vs. what we reconstructed

|                                       Step | Source                                           | Status                                                  |
| -----------------------------------------: | ------------------------------------------------ | ------------------------------------------------------- |
| 1–3 graph + side information (35 features) | authors' `Features100K.py`                       | ported; matches the authors' pickles to 1.7e-15         |
|                4 autoencoder 35→16→4→16→35 | not released, rebuilt from Sec. 4.5 / Figs. 8–11 | Fig. 9 loss curves reproduced (log-RMS error 0.02–0.09) |
|                            5 K-Means K = 8 | not released                                     | min-max scaled codes (Fig. 10 axes run exactly 0–1)     |
|                            6–7 Algorithm 1 | not released                                     | matches an independent plain-Python version bit for bit |

Faithful strict result, 5 seeds × 5 folds: **RMSE 1.0761 ± 0.0014**, MAE 0.849,
P 0.664, R 0.724. Baselines on the same folds: global mean 1.126, item mean 1.025.

## 2. Flaws we found in our own Phase-1 implementation

None of these moves the result by more than 0.005 RMSE.

- `OneHotEncoder(sparse=)`, `drop(axis)`, chained `fillna(inplace)`, `KMeans` `n_init='auto'`,
  and TensorFlow missing on Python 3.14 were already fixed in the previous step.
- AE defaults recalibrated to the paper's figures: batch size 64 (Fig. 9) and min-max
  scaling of the codes (Figs. 10/11). Keras' `val_loss` includes the L1/L2 penalty, so the
  pure reconstruction MSE is now reported separately. Dead/collapsed code units are detected
  on the value range in float64.
- Algorithm 1 itself was correct. 97 % of test ratings are served by tier 1 (the
  cluster-item mean). Every reading of "similar items" lands at 1.0698–1.0767, and even a
  perfect oracle on all non-tier-1 rows only reaches 1.048.
- The real limitation is structural. The clusters encode demographics: NMI with age bin
  0.71, with gender × age 0.77, with the user's mean rating 0.016. They therefore barely beat
  a random partition (1.075 vs 1.087), and under Algorithm 1 RMSE rises steadily with K
  (K = 1: 1.025, K = 8: 1.075, K = 20: 1.113).

## 3. Can any configuration reach 0.887 under the strict protocol?

| Evidence                                                                                    |                                5-fold RMSE |
| ------------------------------------------------------------------------------------------- | -----------------------------------------: |
| 17 AE variants (activations, BatchNorm, L1/L2 0…1e-2, h = 8–32, lr, epochs, batch)          |                                1.070–1.079 |
| Optuna, paper space (unspecified knobs + α grid + K ∈ [2, 30]), test-selected               | 1.025 (collapsed code = item mean; see §5) |
| Best *train-only* partition (leave-one-rating-out search), K = 8                            |                                      1.001 |
| Best partition found **with the test labels** (oracle, full search), K = 8                  |                                  **0.902** |
| Oracle partition, quick search, K = 2 / 4 / 16 / 20 / 30                                    |      0.969 / 0.939 / 0.900 / 0.892 / 0.887 |
| Same K for the real GHRS pipeline                                                           |      1.034 / 1.049 / 1.104 / 1.113 / 1.136 |
| Beyond the paper: ALS user/item baseline + damped cluster residual (Optuna, extended space) |                                      0.943 |
| Context: biased MF / user-kNN / Koren baseline                                              |                      0.935 / 0.951 / 0.946 |

Steps 1–5 only choose the partition of the users. At the paper's K = 8, even a partition
chosen with access to the test labels gives about 0.90. This is an upper bound on the true
minimum, so it is evidence rather than proof. Oracle partitions go below 0.887 only from about
K ≥ 20–30, and at those K the real AE + K-Means pipeline is worse (≈ 1.11–1.14).
Reaching 0.887 with hard cluster means and no user-bias term would mean beating matrix
factorisation by 0.05.

## 4. Leakage hypotheses (`scripts/leakage_simulator.py`)

| Hypothesis (K = 8)                                     |      RMSE |         S² |     P@3.5 |     R@3.5 | max P at R ≥ 0.799 |
| ------------------------------------------------------ | --------: | ---------: | --------: | --------: | -----------------: |
| H0 strict                                              |     1.075 |     2.9e-5 |     0.663 |     0.726 |              0.642 |
| H1 graph from train+test                               |     1.067 |     4.7e-5 |     0.667 |     0.728 |              0.646 |
| H1b authors' `ua.base` features reused for every fold  |     1.063 |     2.8e-5 |     0.668 |     0.728 |              0.647 |
| H3 CI from train+test (target leak)                    |     0.942 |     6.1e-6 |     0.712 |     0.768 |              0.701 |
| H4 in-sample evaluation                                |     0.935 |     7.7e-6 |     0.715 |     0.771 |              0.705 |
| H8a steps 1–6 once on `ua.base`, scored on u1..u5.test |     0.960 | **1.7e-4** |     0.708 |     0.759 |              0.692 |
| H9 ua/ub file swap                                     |     0.946 |     7.5e-6 |     0.728 |     0.762 |              0.714 |
| **Paper**                                              | **0.887** | **1.6e-4** | **0.771** | **0.799** |                    |

H2 (AE and K-Means fitted "on the whole dataset") is not a separate channel: all 943 users are
in every training fold, and the AE sees only the 35 user features, so it reduces to H1.
Metric variants computed on H0: MAE 0.848, Eq. 6 as printed −0.08, per-user RMSE 1.044,
rounded predictions 1.111.

- **Graph/AE leakage, the hypothesis that motivated this audit, is ruled out.** It moves RMSE by
  −0.008 to −0.012.
- **Only target leakage reaches 0.887**, and only at a larger K: H3 at K ≈ 20–24
  (0.890 / 0.877), H4 at K ≈ 20, H9 at K ≈ 20, H8a at K ≈ 32–48.
- **At the K where RMSE ≈ 0.887, P/R do not match.** The threshold-free best precision at
  recall ≥ 0.799 is 0.73–0.75. The paper's 0.771 needs RMSE ≈ 0.80–0.84 in every leaky
  variant.
- **S² matches only for H8a (1.7e-4).** Its RMSE there is 0.960.
- **Fig. 7 (RMSE vs α) is not reproduced.** Every hypothesis, at K = 8 and at its matching K,
  is flat (range 0.003–0.024 vs the paper's 0.403). The paper's curve also contains values
  above the global-mean predictor (1.126).
- **Fig. 13 (cold start) is matched in direction only.** The target-leak variants rise
  monotonically like the paper (corr ≈ 0.99), but by 0.04–0.075 vs the paper's 0.208.

## 5. Hyperparameter search (`scripts/optimize_phase1.py`)

The objective is the mean RMSE on the 5 test folds. This is deliberately optimistic
("oracle HPO"), so failing to reach 0.887 is conservative.

- **Paper space** (strict, elastic net with both penalties > 0, K ∈ [2, 30], α on the paper
  grid; 120 trials: 99 complete, 21 pruned): best **1.0246**, unchanged when re-evaluated
  over 3 seeds (trial 45; P 0.684, R 0.719). This optimum is degenerate. The search drives
  L1 up (≈ 5e-3) and the learning rate down, the 4-d code collapses (0 active dimensions),
  and K-Means puts every user into one cluster. Algorithm 1 then reduces to the item-mean
  predictor (1.025). The best partially non-collapsed trials score 1.041–1.065. So the
  search's best move is to switch the clustering off. fANOVA: l2 0.28, learning rate 0.22,
  K 0.18, α 0.09, "similar" 0.08.
- **Extended space** (relaxed graph rules, other activations, BatchNorm, scaling,
  baselines, shrinkage): best 0.9432 ± seed noise. The gain comes entirely from the
  user/item baseline (fANOVA: shrink 0.50); the clusters add at most about 0.002.
  This run used Optuna's default (unseeded) TPE sampler; later runs use the seeded
  multivariate TPE.

## 6. Paper inconsistencies (forensics)

- Eq. 6 is printed without the square.
- "Average Silhouette = 18.93" lies outside the valid range [−1, 1].
- Fig. 7 has six of its eleven points above the global-mean predictor.
- Fig. 13's slope would need cold users to have RMSE ≈ 1.5–1.7; we measure ≈ 1.1–1.2.
- GHRS 0.887 equals AutoRec 0.887 in the same Table 4.
- Fig. 6 (feature sparsity vs α) is reproduced almost exactly (max |Δ| 0.0007) only by
  graphs built from an 80 % training fold.
- The authors never released steps 4–7. GitHub issue hadoov/GHRS#1 (open since 2023) reports
  an independent reimplementation at RMSE ≈ 1.08, with no reply from the authors.

## 7. Recommendation for Phase 2

Use **1.076 ± 0.001** (strict, faithful) as the GHRS baseline, alongside the item-mean
(1.025) and ALS-baseline (0.946) references. A soft-clustering or VAE contribution should
be compared on the same `ghrs.experiment` engine. `ClusterRatingModel` already accepts soft
UC matrices; its tiers are decided by membership mass. Keep Algorithm 1 (paper) and the
residual estimator (extension) as two clearly separated variants.
