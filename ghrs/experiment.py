"""Shared experiment pipeline used by all scripts.

A single run builds the graph features, trains the autoencoder, clusters the users,
fits Algorithm 1 and computes the metrics for one fold. The ``Protocol`` decides which
ratings are used at each step. Its default is a correct 5-fold cross-validation; the
other settings introduce specific evaluation mistakes on purpose, so their effect can
be measured.

Feature matrices and autoencoder codes are cached as .npy files in ``outputs/cache``,
so repeated runs only recompute what has changed.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from .autoencoder import AEConfig, encode_users
from .clustering import N_INIT, kmeans_clusters, scale_codes
from .data import load_item_genres, load_ratings, load_users
from .features import build_feature_matrix_fast, co_rating_matrices
from .metrics import all_metrics
from .recommender import ClusterRatingModel, RecConfig, membership_matrix

PAPER_TARGETS = {"rmse": 0.887, "rmse_S2": 1.595e-4, "precision": 0.771, "recall": 0.799}
FOLDS_5CV = ("u1", "u2", "u3", "u4", "u5")


@dataclass(frozen=True)
class GraphConfig:
    alpha: float = 0.01          # best value for ML-100K in the paper
    tolerance: int = 0           # 0: identical ratings (authors' code), 1: ratings differ by at most 1
    mode: str = "absolute"       # 'absolute' (authors' code) or 'corated'


@dataclass(frozen=True)
class ClusterConfig:
    k: int = 8                   # elbow method, Table 2 of the paper
    space: str = "ae"            # 'ae' (paper) or 'raw' (K-Means on the 35 features)
    scale: str = "minmax"        # 'minmax' (paper), 'none' or 'standard'
    n_init: int = N_INIT


@dataclass(frozen=True)
class Protocol:
    """Which ratings each step may use. The defaults avoid any leakage.

    graph_source: 'train', 'full' (train and test ratings) or 'authors_ua' (the
        feature files published by the authors, built once from ua.base).
    ci_source: 'train' or 'full' (test ratings included in the cluster means).
    eval_on: 'test' or 'train' (in-sample evaluation).
    cold_frac, cold_seed, cold_eval: the cold-start experiment of Fig. 13. A fraction
        of the users loses all ratings, and the scored test ratings can be limited
        to 'cold' or 'warm' users.
    fixed_base: e.g. 'ua'. Steps 1-6 run once on ``<fixed_base>.base`` and every
        fold's test file is scored against that model, as the authors' script
        suggests. ua.base contains most of each u_k.test, so this leaks test data.
    """
    graph_source: str = "train"
    ci_source: str = "train"
    eval_on: str = "test"
    cold_frac: float = 0.0
    cold_seed: int = 0
    cold_eval: str = "all"
    fixed_base: str = ""

    @property
    def is_strict(self) -> bool:
        return (self.graph_source, self.ci_source, self.eval_on, self.fixed_base) == ("train", "train", "test", "")


@dataclass(frozen=True)
class RunConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    ae: AEConfig = field(default_factory=AEConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    rec: RecConfig = field(default_factory=RecConfig)
    protocol: Protocol = field(default_factory=Protocol)

    def flat(self) -> dict:
        return {f"{part}.{k}": v for part, d in asdict(self).items() for k, v in d.items()}


def _key(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    """Save via a temporary file, because parallel HPO workers share the cache."""
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


class Experiment:
    def __init__(self, cache_dir: Path = Path("outputs/cache"), authors_dir: Path = Path("original/data100k")):
        self.cache = cache_dir
        (self.cache / "features").mkdir(parents=True, exist_ok=True)
        (self.cache / "codes").mkdir(parents=True, exist_ok=True)
        self.authors_dir = authors_dir
        self.users, self.genres = load_users(), load_item_genres()
        self.uids = self.users["UID"].to_numpy()
        self._folds: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    def fold(self, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (train, test) for 'u1', or for an explicit pair such as 'ua.base:ub.test'."""
        if name not in self._folds:
            base, test = name.split(":") if ":" in name else (f"{name}.base", f"{name}.test")
            self._folds[name] = (load_ratings(base), load_ratings(test))
        return self._folds[name]

    def cold_users(self, p: Protocol) -> np.ndarray:
        n = int(round(p.cold_frac * len(self.uids)))
        return np.random.default_rng(p.cold_seed).choice(self.uids, n, replace=False) if n else np.array([], int)

    def rating_sources(self, fold: str, p: Protocol):
        """Ratings used for the graph, for the cluster means and for evaluation."""
        train, test = self.fold(fold)
        if p.fixed_base:
            if "full" in (p.graph_source, p.ci_source):
                raise ValueError("fixed_base cannot be combined with a 'full' source (ratings would be counted twice)")
            train = self.fold(p.fixed_base)[0]
        full = pd.concat([train, test], ignore_index=True)
        cold = self.cold_users(p)
        drop = lambda df: df[~df["UID"].isin(cold)] if len(cold) else df  # noqa: E731
        graph = drop(full if p.graph_source == "full" else train)
        ci = drop(full if p.ci_source == "full" else train)
        ev = train if p.eval_on == "train" else test
        if p.eval_on == "train":
            ev = drop(ev)  # cold users have no training ratings left to score
        if p.cold_eval == "cold":
            ev = ev[ev["UID"].isin(cold)]
        elif p.cold_eval == "warm":
            ev = ev[~ev["UID"].isin(cold)]
        return graph, ci, ev

    def features(self, fold: str, g: GraphConfig, p: Protocol) -> np.ndarray:
        if p.graph_source == "authors_ua":
            if g.tolerance or g.mode != "absolute" or p.cold_frac:
                raise ValueError("authors_ua features exist only for the original graph rule")
            path = self.authors_dir / f"x_train_alpha({g.alpha}).pkl"
            return pd.read_pickle(path).to_numpy(float)
        key = _key("feat-v1", p.fixed_base or fold, asdict(g), p.graph_source, p.cold_frac, p.cold_seed,
                   *([] if p.graph_source == "train" else [fold]))
        path = self.cache / "features" / f"{key}.npy"
        if path.exists():
            return np.load(path)
        graph_ratings, _, _ = self.rating_sources(fold, p)
        mats = co_rating_matrices(graph_ratings, int(self.uids.max()), g.tolerance)
        F = build_feature_matrix_fast(graph_ratings, self.users, g.alpha, g.tolerance, g.mode, mats)
        F = F.to_numpy(float)
        _atomic_save(path, F)
        return F

    def codes(self, F_t: np.ndarray, ae: AEConfig, seed: int) -> tuple[np.ndarray, dict]:
        key = _key("codes-v1", hashlib.sha1(np.ascontiguousarray(F_t)).hexdigest(), asdict(ae), seed)
        path = self.cache / "codes" / f"{key}.npy"
        meta = path.with_suffix(".json")
        if path.exists() and meta.exists():
            return np.load(path), json.loads(meta.read_text())
        F_e, info = encode_users(F_t, ae, seed)
        _atomic_save(path, F_e)
        tmp = meta.with_name(f"{meta.stem}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(info))
        os.replace(tmp, meta)
        return F_e, info

    def labels(self, X: np.ndarray, c: ClusterConfig, seed: int) -> np.ndarray:
        return kmeans_clusters(scale_codes(X, c.scale), c.k, seed, c.n_init)

    def run_fold(self, fold: str, cfg: RunConfig, seed: int = 42, labels: np.ndarray | None = None) -> dict:
        p = cfg.protocol
        _, ci_ratings, ev = self.rating_sources(fold, p)
        info: dict = {}
        if labels is None:
            F_t = self.features(fold, cfg.graph, p)
            if cfg.cluster.space == "ae":
                X, info = self.codes(F_t, cfg.ae, seed)
            else:
                X = F_t
            labels = self.labels(X, cfg.cluster, seed)
        UC = membership_matrix(self.uids, labels, cfg.cluster.k)
        model = ClusterRatingModel(cfg.rec).fit(ci_ratings, UC, self.genres)
        pred, tiers = model.predict(ev)
        out = {"fold": fold, "seed": seed, "n_eval": len(ev),
               **all_metrics(ev["UID"].to_numpy(), ev["rate"].to_numpy(), pred),
               **{f"ae_{k}": v for k, v in info.items()},
               **{f"tier_{k}": int(v) for k, v in tiers.items()}}
        return out

    def run(self, cfg: RunConfig, folds=FOLDS_5CV, seeds=(42,)) -> pd.DataFrame:
        return pd.DataFrame([self.run_fold(f, cfg, s) for s in seeds for f in folds])


def summarize(df: pd.DataFrame) -> dict:
    """Average the results over folds and seeds, in the format of the paper's Table 3.

    S^2 is the variance of the per-fold RMSE (each fold is first averaged over seeds).
    ``dist_to_paper`` is the relative distance to the paper's RMSE, precision and recall.
    """
    per_fold = df.groupby("fold").mean(numeric_only=True)
    s = {m: float(per_fold[m].mean()) for m in per_fold.columns if m not in ("seed", "n_eval")}
    s["rmse_S2"] = float(per_fold["rmse"].var(ddof=1)) if len(per_fold) > 1 else float("nan")
    s["n_folds"], s["n_seeds"] = len(per_fold), int(df["seed"].nunique())
    s["dist_to_paper"] = float(np.sqrt(
        ((s["rmse"] - PAPER_TARGETS["rmse"]) / PAPER_TARGETS["rmse"]) ** 2
        + ((s["precision@3.5"] - PAPER_TARGETS["precision"]) / PAPER_TARGETS["precision"]) ** 2
        + ((s["recall@3.5"] - PAPER_TARGETS["recall"]) / PAPER_TARGETS["recall"]) ** 2))
    return s
