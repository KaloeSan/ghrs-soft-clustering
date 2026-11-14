"""Step 5 of GHRS: K-Means clustering of the encoded users, plus tools for choosing K."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .autoencoder import active_dims

# scikit-learn 1.4 changed the default n_init from 10 to "auto", which means a single
# run with k-means++. The paper is older than that, so we keep the old default.
N_INIT = 10


def scale_codes(F_e: np.ndarray, how: str = "minmax") -> np.ndarray:
    """Scale the autoencoder codes before clustering.

    Min-max scaling is the closest match to the paper: every axis in Fig. 10 runs from
    0 to 1, and the distortion values in Fig. 11 only have the right magnitude for
    scaled codes. In practice the choice hardly affects the RMSE. Dead units are set
    to zero instead of being stretched to [0, 1], so a fully collapsed code turns into
    a single cluster.
    """
    if how == "none":
        return F_e
    act = active_dims(F_e)
    out = np.zeros(F_e.shape, dtype=float)
    X = F_e[:, act].astype(np.float64)
    if how == "minmax":
        lo, hi = X.min(axis=0), X.max(axis=0)
        out[:, act] = (X - lo) / (hi - lo)
    elif how == "standard":
        out[:, act] = (X - X.mean(axis=0)) / X.std(axis=0)
    else:
        raise ValueError(f"unknown scaling {how!r}")
    return out


def kmeans_clusters(F_e: np.ndarray, k: int, seed: int, n_init: int = N_INIT) -> np.ndarray:
    if k == 1:
        return np.zeros(len(F_e), dtype=int)
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(F_e)


def elbow_and_silhouette(F_e: np.ndarray, k_range: range, seed: int) -> list[dict]:
    """Distortion (inertia) and average silhouette for each K, as in Section 4.6.

    The paper reports an "average silhouette" of 18.93, which cannot be a silhouette
    coefficient since those lie in [-1, 1]. It looks more like the distance from the
    distortion curve to the straight line between its end points.
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(F_e)
        sil = silhouette_score(F_e, km.labels_) if 2 <= k < len(F_e) else np.nan
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return rows
