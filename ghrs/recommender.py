"""GHRS steps 6-7: cluster x item rating matrix CI and prediction R_hat = UC x CI.

Algorithm 1 (lines 11-19) of the paper, for cluster c and item i:
  1. mean rating of item i by users of cluster c                  (tier "item")
  2. else mean rating, by users of c, of items *similar* to i     (tier "similar")
  3. else mean of all ratings given by users of c                 (tier "cluster")
The paper does not define "similar items". Here they are items with an identical
genre vector in u.item (the item side information F_i), which is the simplest
content-based reading. Tier 4 (global mean) only guards against empty clusters.
"""

import numpy as np
import pandas as pd

from .data import GENRE_COLS


class ClusterRatingModel:
    def fit(self, train: pd.DataFrame, user_cluster: pd.Series,
            item_genres: pd.DataFrame) -> "ClusterRatingModel":
        self.user_cluster = user_cluster
        self.item_sig = item_genres[GENRE_COLS].astype(str).agg("".join, axis=1).rename("sig")
        t = train.assign(c=train["UID"].map(user_cluster), sig=train["MID"].map(self.item_sig))
        self.ci_item = t.groupby(["c", "MID"])["rate"].agg(["sum", "count"])
        self.ci_similar = t.groupby(["c", "sig"])["rate"].agg(["sum", "count"])
        self.c_mean = t.groupby("c")["rate"].mean().rename("p_cluster")
        self.global_mean = float(t["rate"].mean())
        return self

    def predict(self, test: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
        q = test[["UID", "MID"]].assign(c=test["UID"].map(self.user_cluster),
                                        sig=test["MID"].map(self.item_sig))
        q = q.join(self.ci_item.add_suffix("_item"), on=["c", "MID"])
        q = q.join(self.ci_similar.add_suffix("_sig"), on=["c", "sig"])
        q["p_item"] = q["sum_item"] / q["count_item"]  # NaN when c never rated i
        # "similar items": same genre signature in the same cluster. Item i itself has
        # no ratings in c whenever this tier is reached, so it never contributes.
        q["p_similar"] = (q["sum_sig"] / q["count_sig"]).where(q["count_sig"] > 0)
        q = q.join(self.c_mean, on="c")

        pred = q["p_item"].fillna(q["p_similar"]).fillna(q["p_cluster"]).fillna(self.global_mean)
        tier = np.select([q["p_item"].notna(), q["p_similar"].notna(), q["p_cluster"].notna()],
                         ["item", "similar", "cluster"], default="global")
        return pred.to_numpy(), pd.Series(tier).value_counts()
