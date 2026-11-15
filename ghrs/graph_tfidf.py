"""Phase 2: user similarity graph enriched with TF-IDF genre profiles.

Every user is a "document" whose terms are the genres of the movies they rated: the
term frequency is how many rated movies have a genre, and the inverse document
frequency is estimated over users. Two users are connected only if they pass the
original co-rating rule and the cosine similarity of their genre profiles reaches
``cos_threshold``. The co-rating rule says that two users rate alike; the cosine
condition adds that they also watch the same kind of films.

On ML-100K almost every user has rated each common genre, so the IDF is nearly flat
for drama, comedy, action and the like and mainly boosts rare genres (documentary,
western, fantasy). The cosine therefore mostly compares each user's genre mix.

``edge_seed`` gives a control graph: instead of the edges with the highest genre
cosine, it keeps a random subset of the co-rating edges of the same size. Comparing the
two separates the effect of genre affinity from the effect of simply having fewer edges.

Only the ratings passed in are used, so with the training fold as input the graph
contains no information from the test set.
"""

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer

from .data import GENRE_COLS, N_ITEMS_100K
from .features import add_graph_features, build_side_info, co_rating_matrices, graph_from_pairs, similar_pairs


def genre_tfidf_profiles(df_ratings: pd.DataFrame, item_genres: pd.DataFrame, n_users: int) -> np.ndarray:
    """L2-normalised TF-IDF genre vectors, one row per UID (row 0 and users without ratings are zero)."""
    n_items = max(int(item_genres.index.max()), int(df_ratings["MID"].max())) + 1
    G = item_genres.reindex(np.arange(n_items))[GENRE_COLS].fillna(0).to_numpy(float)
    A = sp.csr_matrix((np.ones(len(df_ratings)), (df_ratings["UID"], df_ratings["MID"])),
                      shape=(n_users + 1, n_items))
    counts = np.asarray(A @ G)
    has_ratings = counts.sum(1) > 0
    # The IDF is estimated on users that have ratings, so empty rows do not shift it.
    tfidf = TfidfTransformer().fit(counts[has_ratings])
    return tfidf.transform(counts).toarray()


def genre_cosine(profiles: np.ndarray) -> np.ndarray:
    """Cosine similarity between users; the profiles are already L2-normalised."""
    return profiles @ profiles.T


def build_hybrid_graph(S: sp.csr_matrix, C: sp.csr_matrix, cos: np.ndarray, alpha_coef: float,
                       cos_threshold: float, mode: str = "absolute", n_items: int = N_ITEMS_100K,
                       edge_seed: int | None = None) -> nx.Graph:
    """Keep a co-rating edge only if the genre cosine of the two users is at least ``cos_threshold``.

    With ``edge_seed`` set, the same number of co-rating edges is kept at random instead
    (control graph).
    """
    rows, cols = similar_pairs(S, C, alpha_coef, mode, n_items)
    keep = cos[rows, cols] >= cos_threshold
    if edge_seed is not None:
        chosen = np.random.default_rng(edge_seed).choice(len(rows), int(keep.sum()), replace=False)
        keep = np.zeros(len(rows), dtype=bool)
        keep[chosen] = True
    return graph_from_pairs(rows[keep], cols[keep])


def build_feature_matrix_hybrid(df_ratings: pd.DataFrame, df_user_raw: pd.DataFrame,
                                item_genres: pd.DataFrame, alpha_coef: float, cos_threshold: float,
                                tolerance: int = 0, mode: str = "absolute",
                                mats: tuple | None = None, edge_seed: int | None = None) -> pd.DataFrame:
    """Steps 1-3 with the hybrid graph; same output layout as ``features.build_feature_matrix_fast``."""
    n_users = int(df_user_raw["UID"].max())
    if mats is None:
        mats = co_rating_matrices(df_ratings, n_users, tolerance)
    cos = genre_cosine(genre_tfidf_profiles(df_ratings, item_genres, n_users))
    df_user = build_side_info(df_user_raw)
    G = build_hybrid_graph(*mats, cos, alpha_coef, cos_threshold, mode, edge_seed=edge_seed)
    df_user = add_graph_features(df_user, G)
    return df_user[df_user.columns[1:]].fillna(0)
