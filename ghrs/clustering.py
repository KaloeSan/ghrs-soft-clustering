"""GHRS step 5: K-Means user clustering on the encoded features F_e (+ K selection)."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# sklearn>=1.4 changed the KMeans default n_init from 10 to "auto" (= 1 run for
# k-means++). The paper predates that change, so the old default is pinned here.
N_INIT = 10


def kmeans_clusters(F_e: np.ndarray, k: int, seed: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit_predict(F_e)


def elbow_and_silhouette(F_e: np.ndarray, k_range: range, seed: int) -> list[dict]:
    """Paper Sec. 4.6: distortion (inertia) and average silhouette for K in [1, 30]."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(F_e)
        sil = silhouette_score(F_e, km.labels_) if 2 <= k < len(F_e) else np.nan
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return rows
