"""GHRS steps 1-3: user similarity graph -> graph features + one-hot side information.

Faithful port of the authors' ``original/Features100K.py``. Only API-level changes
were made; every mathematical / business rule is kept as-is (see inline notes).
"""

import collections
import itertools

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from .data import N_ITEMS_100K

GRAPH_FEATURES = ["PR", "CD", "CC", "CB", "LC", "AND"]


def convert_categorical(df_X: pd.DataFrame, _X: str) -> pd.DataFrame:
    """Replace column ``_X`` with one-hot columns ``_X1.._Xn`` inserted right after UID."""
    values = np.array(df_X[_X])
    integer_encoded = LabelEncoder().fit_transform(values)
    # sklearn>=1.4: `sparse` was removed in favour of `sparse_output`.
    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # pandas>=2.0: positional `axis` argument removed -> drop(columns=...).
    df_X = df_X.drop(columns=_X)
    for j in range(integer_encoded.max() + 1):
        df_X.insert(loc=j + 1, column=str(_X) + str(j + 1), value=onehot_encoded[:, j])
    return df_X


def build_side_info(df_user: pd.DataFrame) -> pd.DataFrame:
    """Categorised demographic features F_s (job, gender, 6 age bins); zip is dropped."""
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
    """Number of items each user pair rated with the *same* value.

    Pairs are ordered tuples from itertools.combinations in file order, exactly as
    in the original. MovieLens .base files are sorted by UID, so (a, b) has a < b.
    """
    pairs = []
    for _, group in df_ratings.groupby(["MID", "rate"]):
        pairs.extend(itertools.combinations(group["UID"], 2))
    return collections.Counter(pairs)


def build_similarity_graph(counter: collections.Counter, alpha_coef: float,
                           n_items: int = N_ITEMS_100K) -> nx.Graph:
    alpha = alpha_coef * n_items  # param * i_no
    # Original: Counter(el for el in counter.elements() if counter[el] >= alpha).keys()
    # That re-expands every pair `count` times only to keep the keys; the filter below
    # yields the same keys in the same (insertion) order, without the O(sum counts) pass.
    edge_list = [pair for pair, cnt in counter.items() if cnt >= alpha]
    G = nx.Graph()
    for u, v in edge_list:
        G.add_edge(u, v, weight=1)
        G.add_edge(u, u, weight=1)  # self-loops are part of the original model
        G.add_edge(v, v, weight=1)
    return G


def add_graph_features(df_user: pd.DataFrame, G: nx.Graph) -> pd.DataFrame:
    """Step 2: six centrality measures, each normalised by its max over users."""
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


def build_feature_matrix(df_ratings: pd.DataFrame, df_user_raw: pd.DataFrame,
                         alpha_coef: float, counter: collections.Counter | None = None
                         ) -> pd.DataFrame:
    """Steps 1-3 for one alpha. Returns the 943 x 35 matrix F_t (UID dropped, rows in
    u.user order); users absent from the graph get 0 for graph features."""
    if counter is None:
        counter = count_similar_pairs(df_ratings)
    df_user = build_side_info(df_user_raw)
    G = build_similarity_graph(counter, alpha_coef)
    df_user = add_graph_features(df_user, G)
    # Original: X_train = df_user[df_user.columns[1:]]; X_train.fillna(0, inplace=True)
    # (chained in-place op on a slice -> SettingWithCopy / CoW issue in pandas>=3).
    return df_user[df_user.columns[1:]].fillna(0)
