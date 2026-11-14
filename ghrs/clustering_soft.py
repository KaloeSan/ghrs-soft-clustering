"""Phase 2: soft clustering of the user codes.

Instead of a single K-Means label, every user gets K membership weights that sum to 1.

- Gaussian mixture model (GMM): the weights are the posterior probabilities of the
  mixture components. A temperature T > 1 makes them softer (softmax of log p / T).
- Fuzzy c-means (FCM, Bezdek 1981): u_ic is proportional to d_ic^(-2 / (m - 1)), where
  the fuzzifier m > 1 controls the softness; m close to 1 behaves like K-Means.

``ghrs.recommender.ClusterRatingModel`` turns the weights into predictions: the rating
of user u for item i is the expected cluster-item mean, sum_c UC[u, c] * CI[c, i].

About 45% of the users share their exact input vector with someone else (users without
graph edges and with the same demographics), so many codes are identical. A
full-covariance GMM can put a component on such a group and shrink its covariance to
the regularisation floor. ``reg_covar`` (1e-3 on the min-max scaled codes by default,
instead of scikit-learn's 1e-6) keeps those components from degenerating.
"""

import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.mixture import GaussianMixture

from .clustering import N_INIT

SOFT_METHODS = ("gmm", "fcm")


def gmm_memberships(X: np.ndarray, k: int, seed: int, covariance_type: str = "full",
                    temperature: float = 1.0, n_init: int = N_INIT, reg_covar: float = 1e-3) -> np.ndarray:
    """Posterior component probabilities of a Gaussian mixture fitted to X."""
    gm = GaussianMixture(n_components=k, covariance_type=covariance_type, n_init=n_init,
                         reg_covar=reg_covar, max_iter=500, random_state=seed).fit(X)
    p = gm.predict_proba(X)
    if temperature != 1.0:
        logp = np.log(np.clip(p, 1e-300, None)) / temperature
        p = np.exp(logp - logp.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
    return p


def _fcm_weights(d2: np.ndarray, m: float) -> np.ndarray:
    """FCM memberships from squared distances, computed in log space for stability."""
    logw = -np.log(np.maximum(d2, 1e-12)) / (m - 1.0)
    w = np.exp(logw - logw.max(axis=1, keepdims=True))
    return w / w.sum(axis=1, keepdims=True)


def _sq_dist(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    return ((X[:, None, :] - V[None, :, :]) ** 2).sum(axis=-1)


def fcm_memberships(X: np.ndarray, k: int, seed: int, m: float = 2.0, n_init: int = N_INIT,
                    max_iter: int = 300, tol: float = 1e-6) -> np.ndarray:
    """Fuzzy c-means memberships; the run with the lowest objective J_m out of ``n_init`` is kept."""
    if m <= 1.0:
        raise ValueError("the fuzzifier m must be greater than 1")
    rng = np.random.default_rng(seed)
    best_J, best_U = np.inf, None
    for _ in range(n_init):
        V, _ = kmeans_plusplus(X, k, random_state=int(rng.integers(2**31 - 1)))
        for _ in range(max_iter):
            Um = _fcm_weights(_sq_dist(X, V), m) ** m
            V_new = (Um.T @ X) / Um.sum(axis=0)[:, None]
            shift = np.abs(V_new - V).max()
            V = V_new
            if shift < tol:
                break
        d2 = _sq_dist(X, V)
        U = _fcm_weights(d2, m)
        J = float(((U ** m) * d2).sum())
        if J < best_J:
            best_J, best_U = J, U
    return best_U


def soft_memberships(X: np.ndarray, method: str, k: int, seed: int, fuzziness: float = 2.0,
                     covariance_type: str = "full", temperature: float = 1.0,
                     n_init: int = N_INIT, reg_covar: float = 1e-3) -> np.ndarray:
    """(n_users, k) membership matrix whose rows sum to 1.

    Constant columns (dead code units) are dropped first. If nothing is left, the code
    has collapsed and every user is put in cluster 0, exactly as K-Means does on a
    collapsed code (the model then equals GHRS with K = 1).
    """
    if method not in SOFT_METHODS:
        raise ValueError(f"unknown soft clustering method {method!r}")
    X = np.asarray(X, dtype=np.float64)
    X = X[:, X.max(axis=0) - X.min(axis=0) > 0]
    if k == 1 or X.shape[1] == 0:
        W = np.zeros((len(X), k))
        W[:, 0] = 1.0
        return W
    if method == "gmm":
        return gmm_memberships(X, k, seed, covariance_type, temperature, n_init, reg_covar)
    return fcm_memberships(X, k, seed, fuzziness, n_init)


def membership_stats(W: np.ndarray) -> dict:
    """How soft the memberships are: mean largest weight and mean entropy scaled to [0, 1]."""
    k = W.shape[1]
    ent = -(W * np.log(np.clip(W, 1e-300, None))).sum(axis=1)
    return {"max_membership": float(W.max(axis=1).mean()),
            "membership_entropy": float(ent.mean() / np.log(k)) if k > 1 else 0.0}
