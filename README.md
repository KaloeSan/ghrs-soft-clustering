# GHRS reproduction on MovieLens 100K

This repository reimplements and evaluates **GHRS**, the graph-based hybrid recommender
system of Darban & Valipour (2022), on the MovieLens 100K dataset. It is the first phase
of a Master's project. The goal of this phase is to reproduce the paper's results and
build a clean baseline. Later phases will replace the hard K-Means step with soft
clustering.

The authors published only the feature extraction step
([hadoov/GHRS](https://github.com/hadoov/GHRS)). The remaining steps (autoencoder,
clustering and rating prediction) are rebuilt here from the paper.

![GHRS framework](Figs/ghrs-structure.png)

## Method

GHRS predicts ratings in seven steps:

1. Build a user similarity graph: two users are connected if they gave the same rating
   to at least `alpha * 1682` movies.
2. Compute six centrality measures for every user (PageRank, degree, closeness,
   betweenness, load, average neighbour degree).
3. Add one-hot demographic features (age group, gender, occupation), giving 35
   features per user.
4. Compress the features to 4 dimensions with an autoencoder (35-16-4-16-35).
5. Cluster the users with K-Means (K = 8).
6. Fill a cluster x movie matrix with average ratings (Algorithm 1 of the paper).
7. Predict a user's rating as the value for that user's cluster.

## Results

5-fold cross-validation on the standard splits `u1`-`u5`, alpha = 0.01, K = 8,
averaged over 5 random seeds:

| Model | RMSE | Precision | Recall |
|---|---|---|---|
| GHRS, as reported in the paper | 0.887 | 0.771 | 0.799 |
| GHRS, this implementation | 1.076 ± 0.001 | 0.664 | 0.724 |
| Item mean (reference) | 1.025 | | |
| Global mean (reference) | 1.126 | | |

The reported RMSE could not be reproduced. A tuning search over the settings the paper
does not specify, a search over user partitions, and a set of simulated evaluation
mistakes all fail to explain the gap. The details are in
[docs/phase1_reproducibility_audit.md](docs/phase1_reproducibility_audit.md).

## Repository structure

```
ghrs/                  the method as a Python package
  data.py              MovieLens 100K loaders
  features.py          steps 1-3: similarity graph, centralities, side information
  autoencoder.py       step 4: autoencoder
  clustering.py        step 5: K-Means and choice of K
  recommender.py       steps 6-7: cluster-item ratings and predictions
  metrics.py           RMSE, MAE, precision and recall
  experiment.py        shared pipeline, evaluation protocols and caching
  bounds.py            best RMSE reachable by any user partition
scripts/               command-line entry points
  Features100K.py      the authors' feature script, updated for current libraries
  verify_features.py   compares our features with the authors' files
  run_phase1.py        main experiment (5-fold cross-validation)
  leakage_simulator.py tests whether evaluation mistakes explain the paper's result
  optimize_phase1.py   hyperparameter search with Optuna
  partition_bound.py   partition search used as a reference bound
original/              the authors' unmodified script and feature files
datasets/ml-100k/      MovieLens 100K
docs/                  paper, presentation and the reproducibility report
outputs/               generated results (not tracked by git)
```

## Setup

Tested with Python 3.14 on Windows 11, CPU only. TensorFlow does not support
Python 3.14 yet, so Keras 3 runs on the PyTorch backend; this is set automatically
when the `ghrs` package is imported.

```bash
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run all commands from the repository root.

```bash
# check that the feature extraction matches the authors' files
python -m scripts.verify_features --fast

# main experiment: 5-fold cross-validation, 5 seeds
python -m scripts.run_phase1 --seeds 42 1 2 3 4

# choose K with the elbow and silhouette methods
python -m scripts.run_phase1 --select-k

# additional analyses
python -m scripts.leakage_simulator
python -m scripts.optimize_phase1 --space paper --trials 120 --workers 3
python -m scripts.partition_bound --k 8 --effort full
```

Results are written to `outputs/`. Feature matrices and autoencoder codes are cached in
`outputs/cache/`, so later runs are much faster than the first one.

## Citation

The method comes from:

```bibtex
@article{darban2022ghrs,
  title={GHRS: Graph-based hybrid recommendation system with application to movie recommendation},
  author={Darban, Zahra Zamanzadeh and Valipour, Mohammad Hadi},
  journal={Expert Systems with Applications},
  pages={116850},
  year={2022},
  publisher={Elsevier}
}
```

## References

1. Z. Z. Darban and M. H. Valipour. *GHRS: Graph-based hybrid recommendation system
   with application to movie recommendation.* Expert Systems with Applications, 2022.
   [doi:10.1016/j.eswa.2022.116850](https://doi.org/10.1016/j.eswa.2022.116850),
   preprint [arXiv:2111.11293](https://arxiv.org/abs/2111.11293).
2. F. M. Harper and J. A. Konstan. *The MovieLens datasets: History and context.*
   ACM Transactions on Interactive Intelligent Systems 5(4), 2015.
   [doi:10.1145/2827872](https://doi.org/10.1145/2827872).
