"""Steps 6-7 of GHRS: the cluster-item rating matrix CI and the predictions UC x CI.

Algorithm 1 of the paper fills CI[c, i] for cluster c and item i with the first
available value from this list:

1. the mean rating of item i among the members of c,
2. the mean rating, among the members of c, of items similar to i,
3. the mean of all ratings given by the members of c.

A fourth level, the global mean, only protects against empty clusters. In our
experiments about 97% of the test ratings are predicted by level 1, so the exact
meaning of "similar items" changes the RMSE very little.

Implementation with matrices, for K clusters:

    E = r - b(u, i)                      residual after a baseline b (b = 0 in the paper)
    SUM[c, i], CNT[c, i]                 sum and count of E for the members of c who rated i
    CI[c, i] = (SUM + lam * P_i) / (CNT + lam)
    R_hat[u, i] = b(u, i) + (UC @ CI)[u, i]

With ``lam = 0`` this is the plain mean used in the paper. The prior P_i is the item
mean from the training data when there is no baseline and 0 otherwise. The similar-
items level uses ``SUM @ W.T`` and ``CNT @ W.T`` for a binary item-item matrix W.

``RecConfig()`` gives the model from the paper. The other options are extensions
that we only use in the hyperparameter search:

- ``similar``: identical genre vector (default), genre Jaccard similarity, top-k
  item cosine similarity, or no similar-items level at all.
- ``baseline``: none (paper), item bias, user bias, or user and item biases fitted
  by alternating least squares (as in Surprise's BaselineOnly).
- ``shrink``: pulls every cluster-item mean towards its prior P_i.
- ``fallback``: the cluster mean (paper) or the prior P_i.
- ``clip``: clip the predictions to [1, 5].

UC can contain hard K-Means labels or soft memberships (Phase 2: GMM, fuzzy c-means);
each user's row must sum to 1. With soft memberships SUM and CNT are membership-
weighted, so CI[c, i] is a weighted mean of the ratings of item i, and the prediction
sum_c UC[u, c] * CI[c, i] is the expected cluster-item mean under the user's membership
distribution. For one-hot rows both reduce to the paper's model. A cell uses level 1
only if its weighted rating count reaches ``min_mass`` (0.5 by default): for hard labels
this is exactly the paper's rule, and for soft memberships it keeps a tiny membership
from counting as "the cluster has rated this item".
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import GENRE_COLS

SIMILAR = ("genre_exact", "genre_jaccard", "item_cosine", "none")
BASELINES = ("none", "item", "user", "user_item")
FALLBACKS = ("cluster", "baseline")


@dataclass(frozen=True)
class RecConfig:
    similar: str = "genre_exact"
    baseline: str = "none"
    shrink: float = 0.0
    fallback: str = "cluster"
    clip: bool = False
    jaccard_min: float = 0.5
    knn: int = 20
    reg_i: float = 10.0
    reg_u: float = 15.0
    als_epochs: int = 10
    min_mass: float = 0.5       # weighted count needed before a level is used

    def __post_init__(self):
        if self.similar not in SIMILAR or self.baseline not in BASELINES or self.fallback not in FALLBACKS:
            raise ValueError(f"invalid RecConfig {self}")
        if self.shrink < 0 or self.min_mass <= 0:
            raise ValueError("shrink must be >= 0 and min_mass > 0")

    @property
    def is_paper(self) -> bool:
        # The paper does not define "similar items", so any choice of it still counts.
        return self == RecConfig(similar=self.similar)


def membership_matrix(uids: np.ndarray, labels: np.ndarray, k: int | None = None) -> np.ndarray:
    """Turn hard cluster labels into a one-hot matrix UC with one row per UID."""
    k = int(labels.max()) + 1 if k is None else k
    UC = np.zeros((int(uids.max()) + 1, k))
    UC[uids, labels] = 1.0
    return UC


class ClusterRatingModel:
    """Rating predictor based on cluster-item means (Algorithm 1 of the paper)."""

    def __init__(self, cfg: RecConfig = RecConfig()):
        self.cfg = cfg

    def _fit_baseline(self, u: np.ndarray, i: np.ndarray, r: np.ndarray) -> None:
        cfg, nU, nI = self.cfg, self.n_users, self.n_items
        n_i, n_u = np.bincount(i, minlength=nI), np.bincount(u, minlength=nU)
        self.b_u, self.b_i = np.zeros(nU), np.zeros(nI)
        self.mu = float(r.mean())
        if cfg.baseline == "none":
            # Prior for shrinkage: the item mean, or the global mean for unseen items.
            self.prior = np.where(n_i > 0, np.bincount(i, r, nI) / np.maximum(n_i, 1), self.mu)
            self.mu = 0.0
            return
        self.prior = np.zeros(nI)
        if cfg.baseline == "item":
            self.b_i = np.bincount(i, r - self.mu, nI) / (cfg.reg_i + n_i)
        elif cfg.baseline == "user":
            self.b_u = np.bincount(u, r - self.mu, nU) / (cfg.reg_u + n_u)
        else:  # user_item, fitted by alternating least squares
            for _ in range(cfg.als_epochs):
                self.b_i = np.bincount(i, r - self.mu - self.b_u[u], nI) / (cfg.reg_i + n_i)
                self.b_u = np.bincount(u, r - self.mu - self.b_i[i], nU) / (cfg.reg_u + n_u)

    def baseline(self, u: np.ndarray, i: np.ndarray) -> np.ndarray:
        return self.mu + self.b_u[u] + self.b_i[i]

    def _similarity_matrix(self, item_genres: pd.DataFrame, u, i, r) -> np.ndarray | None:
        """Binary item-item matrix W used by the similar-items level."""
        kind = self.cfg.similar
        if kind == "none":
            return None
        if kind in ("genre_exact", "genre_jaccard"):
            G = np.zeros((self.n_items, len(GENRE_COLS)))
            idx = item_genres.index.to_numpy()
            G[idx] = item_genres[GENRE_COLS].to_numpy()
            if kind == "genre_exact":
                _, sig = np.unique(G, axis=0, return_inverse=True)
                W = (sig[:, None] == sig[None, :]).astype(float)
            else:
                inter = G @ G.T
                union = G.sum(1)[:, None] + G.sum(1)[None, :] - inter
                W = (np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
                     >= self.cfg.jaccard_min).astype(float)
            W[:, ~np.isin(np.arange(self.n_items), idx)] = 0  # ids with no movie behind them
        else:  # item_cosine
            R = np.zeros((self.n_users, self.n_items))
            R[u, i] = r
            norms = np.linalg.norm(R, axis=0)
            S = (R.T @ R) / np.maximum(norms[:, None] * norms[None, :], 1e-12)
            np.fill_diagonal(S, -np.inf)
            W = np.zeros_like(S)
            top = np.argpartition(-S, self.cfg.knn, axis=1)[:, :self.cfg.knn]
            np.put_along_axis(W, top, 1.0, axis=1)
            W[norms == 0] = 0
        np.fill_diagonal(W, 0)
        return W

    def fit(self, train: pd.DataFrame, UC: np.ndarray, item_genres: pd.DataFrame,
            n_items: int | None = None) -> "ClusterRatingModel":
        """Fit CI from the training ratings.

        ``UC`` has one row per UID (row 0 is unused) and one column per cluster.
        """
        u, i, r = (train[c].to_numpy() for c in ("UID", "MID", "rate"))
        r = r.astype(float)
        self.UC = np.asarray(UC, float)
        if self.UC.ndim != 2 or self.UC.shape[0] <= u.max():
            raise ValueError(f"UC must be (max UID + 1) x K, got {self.UC.shape}")
        mass = self.UC.sum(axis=1)
        if (self.UC < 0).any() or not np.all(np.isclose(mass, 1.0) | (mass == 0)):
            raise ValueError("every used row of UC must be non-negative and sum to 1")
        self.n_users = self.UC.shape[0]
        self.n_items = int(max(n_items or 0, item_genres.index.max(), i.max())) + 1
        self._fit_baseline(u, i, r)
        e = r - self.baseline(u, i)

        K, lam = self.UC.shape[1], self.cfg.shrink
        w = self.UC[u]                                        # cluster weights of each rating
        SUM = np.stack([np.bincount(i, w[:, c] * e, self.n_items) for c in range(K)])
        CNT = np.stack([np.bincount(i, w[:, c], self.n_items) for c in range(K)])
        P = self.prior[None, :]

        CI = np.full((K, self.n_items), np.nan)
        # Which level filled each cell: 0 item, 1 similar, 2 cluster, 3 prior, 4 global.
        tier = np.full((K, self.n_items), 4, dtype=np.int8)
        mm = self.cfg.min_mass
        has = CNT >= mm
        # `where` limits the division to the selected cells; the others would be 0/0
        # and are filled by the lower levels anyway.
        np.divide(SUM + lam * P, CNT + lam, out=CI, where=has)
        tier[has] = 0

        W = self._similarity_matrix(item_genres, u, i, r)
        if W is not None:
            S_sim, N_sim = SUM @ W.T, CNT @ W.T
            m = ~has & (N_sim >= mm)
            np.divide(S_sim + lam * P, N_sim + lam, out=CI, where=m)
            tier[m] = 1

        m = np.isnan(CI)
        nonempty = (CNT.sum(1) >= mm)[:, None]
        if self.cfg.fallback == "cluster":
            c_mean = (SUM.sum(1) / np.maximum(CNT.sum(1), 1e-12))[:, None]
            CI = np.where(m & nonempty, c_mean, CI)
            tier[m & nonempty] = 2
        else:  # use the prior: the item mean without a baseline, 0 with one
            CI = np.where(m, np.broadcast_to(P, CI.shape), CI)
            tier[m] = 3
        # Anything still missing gets the global mean (0 when we predict residuals).
        g = float(e.mean()) if self.cfg.baseline == "none" else 0.0
        self.CI = np.where(np.isnan(CI), g, CI)
        self.tier = tier
        return self

    def predict(self, test: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
        """Predict the test ratings; also return how many came from each level."""
        u, i = test["UID"].to_numpy(), test["MID"].to_numpy()
        pred = self.baseline(u, i) + np.einsum("nk,kn->n", self.UC[u], self.CI[:, i])
        if self.cfg.clip:
            pred = np.clip(pred, 1.0, 5.0)
        hard = self.UC[u].argmax(1)
        names = np.array(["item", "similar", "cluster", "prior", "global"])
        return pred, pd.Series(names[self.tier[hard, i]]).value_counts()
