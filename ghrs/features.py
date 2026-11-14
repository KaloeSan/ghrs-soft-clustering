"""Steps 1-3 of GHRS: user similarity graph, graph features and one-hot side information.

Ported from the authors' ``original/Features100K.py``. Only calls that no longer work
with current library versions were changed; the computation itself is the same.
The ``*_fast`` functions are a vectorised version used by the experiments.
"""

import collections
import itertools

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from .data import N_ITEMS_100K

GRAPH_FEATURES = ["PR", "CD", "CC", "CB", "LC", "AND"]


def convert_categorical(df_X: pd.DataFrame, _X: str) -> pd.DataFrame:
    """Replace column ``_X`` with one-hot columns ``_X1 .. _Xn`` placed right after UID."""
    values = np.array(df_X[_X])
    integer_encoded = LabelEncoder().fit_transform(values)
    # scikit-learn 1.4 renamed `sparse` to `sparse_output`.
    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # pandas 2.0 no longer accepts the axis as a positional argument.
    df_X = df_X.drop(columns=_X)
    for j in range(integer_encoded.max() + 1):
        df_X.insert(loc=j + 1, column=str(_X) + str(j + 1), value=onehot_encoded[:, j])
    return df_X


def build_side_info(df_user: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode job, gender and six age bins; the zip code is dropped."""
    df_user = convert_categorical(df_user, "job")
    df_user = convert_categorical(df_user, "gender")
    df_user["bin"] = pd.cut(df_user["age"], [0, 10, 20, 30, 40, 50, 100],
                            labels=["1", "2", "3", "4", "5", "6"])
    df_user["age"] = df_user["bin"]
    df_user = df_user.drop(columns="bin")
    df_user = convert_categorical(df_user, "age")
    df_user = df_user.drop(columns="zip")
    return df_user


def count_similar_pairs(df_ratings: pd.DataFrame) -> collections.Counter:
    """Count, for every pair of users, the items they gave exactly the same rating.

    Pairs come from ``itertools.combinations`` in file order, as in the original code.
    The .base files are sorted by UID, so each pair (a, b) has a < b.
    """
    pairs = []
    for _, group in df_ratings.groupby(["MID", "rate"]):
        pairs.extend(itertools.combinations(group["UID"], 2))
    return collections.Counter(pairs)


def build_similarity_graph(counter: collections.Counter, alpha_coef: float,
                           n_items: int = N_ITEMS_100K) -> nx.Graph:
    alpha = alpha_coef * n_items
    # The original rebuilt a second Counter from counter.elements() just to filter it.
    # This comprehension keeps the same pairs in the same order without that detour.
    edge_list = [pair for pair, cnt in counter.items() if cnt >= alpha]
    G = nx.Graph()
    for u, v in edge_list:
        G.add_edge(u, v, weight=1)
        G.add_edge(u, u, weight=1)  # the original model adds self-loops
        G.add_edge(v, v, weight=1)
    return G


def add_graph_features(df_user: pd.DataFrame, G: nx.Graph) -> pd.DataFrame:
    """Step 2: six centrality measures, each divided by its maximum over the users."""
    df_user = df_user.copy()
    measures = {
        "PR": lambda: nx.pagerank(G.to_directed()),
        "CD": lambda: nx.degree_centrality(G),
        "CC": lambda: nx.closeness_centrality(G),
        "CB": lambda: nx.betweenness_centrality(G),
        "LC": lambda: nx.load_centrality(G),
        "AND": lambda: nx.average_neighbor_degree(G, weight="weight"),
    }
    for col, fn in measures.items():
        df_user[col] = df_user["UID"].map(fn())
        df_user[col] /= float(df_user[col].max())
    return df_user


def co_rating_matrices(df_ratings: pd.DataFrame, n_users: int, tolerance: int = 0
                       ) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Pair statistics computed with sparse matrix products.

    Returns two symmetric matrices indexed by UID, with a zero diagonal:
    ``S[a, b]`` is the number of items both users rated with ``|r_a - r_b| <= tolerance``
    and ``C[a, b]`` is the number of items both users rated. With ``tolerance=0``,
    ``S`` holds the same counts as ``count_similar_pairs``.
    """
    u, m, r = (df_ratings[c].to_numpy() for c in ("UID", "MID", "rate"))
    shape = (n_users + 1, int(m.max()) + 1)
    B = {k: sp.csr_matrix((np.ones((r == k).sum()), (u[r == k], m[r == k])), shape=shape)
         for k in range(1, 6)}
    S = sum(B[k] @ B[k].T for k in B)
    for d in range(1, tolerance + 1):
        for k in range(1, 6 - d):
            X = B[k] @ B[k + d].T
            S = S + X + X.T
    A = sum(B.values())
    C = A @ A.T
    for M in (S, C):
        M.setdiag(0)
        M.eliminate_zeros()
    return S.tocsr(), C.tocsr()


def build_similarity_graph_fast(S: sp.csr_matrix, C: sp.csr_matrix, alpha_coef: float,
                                mode: str = "absolute", n_items: int = N_ITEMS_100K,
                                min_corated: int = 5) -> nx.Graph:
    """Build the similarity graph from the matrices of ``co_rating_matrices``.

    ``mode='absolute'`` is the authors' rule (``S >= alpha * n_items``). ``'corated'``
    is another possible reading of "more than alpha percent of items with similar
    ratings": ``S / C >= alpha``, with at least ``min_corated`` items rated by both.
    """
    S_up, C_up = sp.triu(S, k=1).tocoo(), sp.triu(C, k=1).tocsr()
    if mode == "absolute":
        keep = S_up.data >= alpha_coef * n_items
    elif mode == "corated":
        c = np.asarray(C_up[S_up.row, S_up.col]).ravel()
        keep = (c >= min_corated) & (S_up.data / np.maximum(c, 1) >= alpha_coef)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    rows, cols = S_up.row[keep].tolist(), S_up.col[keep].tolist()
    G = nx.Graph()
    G.add_edges_from(zip(rows, cols), weight=1)
    G.add_edges_from(((n, n) for n in set(rows) | set(cols)), weight=1)
    return G


def build_feature_matrix_fast(df_ratings: pd.DataFrame, df_user_raw: pd.DataFrame,
                              alpha_coef: float, tolerance: int = 0, mode: str = "absolute",
                              mats: tuple | None = None) -> pd.DataFrame:
    """Faster equivalent of ``build_feature_matrix``.

    Gives the same result (up to about 1e-12) for the default settings, and also
    supports the rating tolerance and the co-rated graph rule.
    """
    if mats is None:
        mats = co_rating_matrices(df_ratings, int(df_user_raw["UID"].max()), tolerance)
    df_user = build_side_info(df_user_raw)
    G = build_similarity_graph_fast(*mats, alpha_coef, mode=mode)
    df_user = add_graph_features(df_user, G)
    return df_user[df_user.columns[1:]].fillna(0)


def build_feature_matrix(df_ratings: pd.DataFrame, df_user_raw: pd.DataFrame,
                         alpha_coef: float, counter: collections.Counter | None = None
                         ) -> pd.DataFrame:
    """Steps 1-3 for a single alpha.

    Returns the 943 x 35 feature matrix without the UID column, in the row order of
    ``u.user``. Users that are not in the graph get zeros for the graph features.
    """
    if counter is None:
        counter = count_similar_pairs(df_ratings)
    df_user = build_side_info(df_user_raw)
    G = build_similarity_graph(counter, alpha_coef)
    df_user = add_graph_features(df_user, G)
    # The original called fillna(inplace=True) on a slice, which does nothing under
    # pandas 3 copy-on-write, so the result is returned directly instead.
    return df_user[df_user.columns[1:]].fillna(0)
