# Final synthesis: our pipeline and R. Jarzyński's parallel work

This report compares our Phase 1 audit and Phase 2 extensions with the independent
GHRS reproduction by R. Jarzyński (workspace `GHRS-SVD-plus-plus-main/radek_testy`,
iterations `radek` to `radek9`), and describes how the two lines of work were merged.
All numbers are for MovieLens 100K. The generated result tables are in
[phase3_synthesis_results.md](phase3_synthesis_results.md).

## 1. Summary

- Both reproductions agree: the paper's pipeline gives RMSE 1.075-1.076 under a
  correct protocol, and 0.887 is only reached with test ratings leaking into the
  cluster means, at K of about 22 instead of 8.
- Radek found no error in our protocol, metrics or code. His precision/recall
  definition is equivalent to ours, and his Phase 1 number (1.0752) matches ours (1.0761).
- His main contribution is a different model class. User and item biases plus SVD++
  with the GHRS features as side information bring the RMSE from 1.076 to about 0.905.
  That is the real improvement over the paper, and the clustering plays almost no part in it.
- We integrated his SVD++ baseline, his per-user temporal split and his
  label-permutation leakage audit. Our soft clustering now runs on top of SVD++ as a
  residual correction. It improves SVD++ consistently but very little: 0.9080 → 0.9075
  (−0.0006, better in 5/5 folds, Holm-adjusted p = 0.04).

## 2. What Radek did

| Iteration | Content                                                              |                Best RMSE (5-fold) |
| --------- | -------------------------------------------------------------------- | --------------------------------: |
| `radek`   | GHRS rebuilt from the paper                                          |                            1.0752 |
| `radek2`  | about 30 GHRS variants; target-leak sweep over K                     |    1.0742 (leak: 0.887 at K ≈ 22) |
| `radek3`  | biases + matrix factorisation with GHRS features and genres          |                            0.9235 |
| `radek4`  | item text embeddings from a pretrained MiniLM model                  |                            0.9196 |
| `radek5`  | SVD++ with side features, item-kNN, uniform ensemble                 |                            0.9078 |
| `radek6`  | time biases (day of rating, movie age, user tenure)                  |                            0.9045 |
| `radek7`  | our Phase 2 ideas as features or ensemble members; cold start        |                            0.9040 |
| `radek8`  | strict version without the day term, 3 seeds, audit, temporal splits | 0.9048 (per-user temporal: 0.954) |
| `radek9`  | `radek6` turned step by step into our Phase 2 proposals              |      0.9019 (vs 0.9027, p ≈ 0.07) |

## 3. What he got right

- **The decisive diagnosis.** Algorithm 1 has no user bias, and the clusters mostly
  encode demographics. Adding μ + b_u + b_i alone brings the RMSE from 1.075 to 0.945
  (we found the same with our ALS baseline, 0.943, in the Phase 1 HPO).
- **A strong, leak-free model.** Hyperparameters and early stopping use a 5% split
  held out inside the training data; the test fold is used only for metrics.
- **A general leakage test.** Shuffling the test labels must not change the
  predictions. We adopted it; it passes for our strict pipeline and flags the leaky
  protocol H3 (section 6).
- **Self-criticism of his own time features.** The per-day user bias exploits ratings
  from the same day landing on both sides of a random split. He removed it in the
  strict version and showed that it adds almost nothing on a temporal split.
- **An independent check of our Phase 2 claim.** Soft clustering improves the
  cluster-mean predictor only by moving it towards the item mean. This is what our
  Phase 2 controls showed as well (FCM 1.026 vs K = 1 at 1.025).

## 4. Cross-examination

### Did Radek find a flaw in the paper that we missed?

No new mechanism. His leak sweep reproduces our hypothesis H3 (0.887 at K ≈ 20-24,
precision too low), and he cites our 13-hypothesis audit. Two of his diagnostics are
worth quoting in the paper: correct-protocol RMSE rises with K, leaky RMSE falls with
K; and the paper's 0.887 is also the level a strong, leak-free factorisation model
reaches only with time features and ensembling (about 0.90), which a clustering
method without user biases cannot get near.

### Did he find a flaw in our logic?

Not in the code or the protocol:

| Point checked              | Finding                                                                                       |
| -------------------------- | --------------------------------------------------------------------------------------------- |
| Metrics                    | Same RMSE; his precision/recall threshold (3.5 on both sides) equals ours for integer ratings |
| Data split, graph per fold | Same strict protocol; both build the graph from the training fold only                        |
| Phase 1 result             | 1.0752 (his) vs 1.0761 (ours, 5 seeds)                                                        |
| Cold start                 | Same conclusion: no model beats the item mean for users without ratings                       |

The one point to take seriously is about **framing**. His documents state plainly
that the Phase 2 gain equals switching the clustering off (K = 1 / item mean). Our
README and our generated report say the same, but the final presentation and the
README table no longer show that reference. An examiner who reads both repositories
will see the comparison, so the defence should be ready to explain it.

### Weak points in his work

| Issue                                                                                                    | Effect                                                  | How we handled it                                 |
| -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------- |
| MiniLM item embeddings bring in a pretrained language model (outside knowledge, titles include the year) | 0.002-0.004 RMSE, within noise by his own measurement   | left out                                          |
| Per-day user bias exploits random folds                                                                  | about 0.003 on random folds, none on the temporal split | left out (as in his strict version)               |
| `radek` to `radek7` use one seed and no significance test                                                | differences below about 0.003 are not reliable          | we use 3 seeds and corrected, Holm-adjusted tests |
| Seed-averaged predictions are compared with mean-over-seed scores elsewhere                              | averaging 3 seeds alone is worth about 0.0075 on u1     | we average SVD++ over seeds and say so            |
| Users and movies without training ratings keep random latent vectors                                     | slight noise in cold-start predictions                  | zero vectors in our port                          |
| `radek9` reports 7 p-values without multiplicity correction                                              | p ≈ 0.07 would not survive correction                   | Holm adjustment                                   |
| The cluster x movie member is blended with raw ratings, not residuals                                    | always hurts the ensemble (his `radek7` B)              | residuals after SVD++, out-of-fold                |

None of these invalidates his main result; the strict `radek8` numbers stand.

### What we did better

- One experiment engine with explicit protocols, caching and controls (argmax,
  random-edge graph, K = 1 and item-mean references), instead of numbered scripts.
- The 13-hypothesis leakage simulation, the oracle partition bound, Holm-adjusted
  corrected t-tests, cold-start statistics per random draw, and publication figures.

### Is the temporal split better than random 5-fold CV?

It answers a different question, so it complements the paper's protocol rather than
replacing it:

- **Random 5-fold (u1-u5)** is the paper's protocol and the only one that makes our
  numbers comparable with Table 3. It also gives five folds for paired tests.
- **Per-user temporal split** (last 20% of each user's ratings) predicts the future
  from the past and is closer to deployment. Every method gets worse (SVD++ 0.908 →
  0.955) and the ranking stays the same. It is a single split, so it gives no
  fold-level test.
- **Global temporal split** mostly measures cold start: 17,048 of 20,000 test
  ratings come from users without training ratings.

The random split is not flawed for our models: they use no timestamps, so the
same-day effect that inflated Radek's day term does not exist for them.

## 5. What was integrated

| Change                                    | Where                                                                 | Notes                                                                                                  |
| ----------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| SVD++ with GHRS side information          | `ghrs/factorization.py`                                               | port of his model; k = 32, λ = 0.1 (his most frequent inner-validation choice); no MiniLM, no day term |
| External baseline under the cluster model | `RecConfig(baseline='offset')` in `ghrs/recommender.py`               | cluster residual means shrunk towards zero, i.e. towards SVD++                                         |
| Out-of-fold SVD++ residuals               | `out_of_fold_predictions`                                             | 5 inner folds; residuals of a model on its own training data are too small to learn from               |
| Temporal splits                           | fold names `temporal_user`, `temporal_global` in `ghrs/experiment.py` | identical to his construction (same train/test sizes)                                                  |
| Label-permutation audit                   | `scripts/run_final_synthesis.py`                                      | fresh cache per run, numerical noise measured separately                                               |
| Shared statistics                         | `ghrs/stats.py`                                                       | corrected t-test and Holm, used by Phase 2 and Phase 3                                                 |

Not integrated: his kNN/MF ensemble. In his strict run the ensemble adds 0.0004 over
SVD++ alone (0.9048 vs 0.9052), which does not justify two more models.

The Phase 1 and Phase 2 code paths are unchanged: Phase 1 still gives 1.0796 on u1,
and the Phase 2 report and figures rebuild byte-identically.

## 6. Results of the synthesis

From [phase3_synthesis_results.md](phase3_synthesis_results.md) (3 clustering seeds,
SVD++ averaged over 3 seeds, residual shrinkage λ = 25 fixed in advance).

| Model                               | Random 5-fold | Temporal per user | Temporal global |
| ----------------------------------- | ------------: | ----------------: | --------------: |
| GHRS reproduction (AE + K-Means)    |        1.0761 |            1.1339 |          1.1466 |
| AE + FCM (Phase 2)                  |        1.0264 |            1.0766 |          1.0382 |
| ALS user and item biases            |        0.9458 |            0.9986 |          1.0300 |
| SVD++ with GHRS features            |        0.9080 |            0.9547 |          1.0281 |
| SVD++ + K-Means residual            |        0.9087 |            0.9544 |          1.0298 |
| SVD++ + FCM residual                |        0.9075 |            0.9535 |          1.0281 |
| SVD++ + VAE + FCM + TF-IDF residual |        0.9073 |            0.9534 |          1.0281 |

Paired tests on the random folds (corrected t-test, Holm):

- every SVD++ variant beats the GHRS reproduction by about 0.168 (5/5 folds, p < 1e-6);
- the FCM residual beats plain SVD++ by 0.0006 (5/5 folds, p = 0.04), the full Phase 2
  stack by 0.0007 (5/5 folds, p = 0.06), while the hard K-Means residual is 0.0006
  worse (1/5 folds better). The temporal split shows the same order (−0.0012 for FCM).

The label-permutation audit changes the strict predictions by at most 7.2e-6, below
the run-to-run noise of 1.1e-5, and the leaky H3 protocol by up to 4.1 rating points.

## 7. What can be claimed

1. GHRS as published is not reproducible: 1.076 under a correct protocol; 0.887
   appears only with target leakage and at the wrong K (both teams, independently).
2. The paper's architecture is limited by its prediction rule, not by its
   representation: adding biases and factorisation with the same GHRS features gives
   0.908 (SVD++) on the paper's own protocol, 0.168 better than the reproduction.
3. Soft clustering is a consistent but small refinement on top of a strong model.
   As a residual correction to SVD++, FCM beats both plain SVD++ and hard K-Means on
   every fold and on the temporal split, but the gain is 0.0006-0.0012 RMSE. On its
   own, without biases, its gain is a move towards the item mean.
4. VAE and the TF-IDF graph do not add measurably to FCM (0.9073 vs 0.9075).
5. No method beats the item mean for users without any ratings.

What should not be claimed: that soft clustering "drops the RMSE massively", or that
the Phase 2 result of 1.026 improves on GHRS for a reason other than smoothing.

## 8. Reproducing

```bash
python -m scripts.run_final_synthesis --temporal-global
```

Runtime about 15 minutes on a CPU (SVD++ fits are cached in `outputs/cache/svdpp`).
Radek's original scripts remain in `GHRS-SVD-plus-plus-main/radek_testy`.
