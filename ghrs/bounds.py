"""Partition bound: the best RMSE any user partition can reach under Algorithm 1.

Everything before step 6 (graph, alpha, side information, the autoencoder and its
hyperparameters, K-Means) only decides how the 943 users are split into K clusters.
So the lowest test RMSE over all possible partitions limits what any tuning of
steps 1-5 can achieve for that K.

We estimate that minimum by searching partitions directly on the test error. This
uses the test labels and serves only as a reference point. The search returns an
actual partition, so its RMSE is an upper bound on the true minimum: a value above
0.887 is strong evidence, not a proof.

Both objectives are a sum of per-cluster terms that depend only on additive
statistics of the cluster members. Moving one user therefore changes only two terms,
which makes coordinate descent and simulated annealing cheap and exact.

- objective 'test': test SSE of Algorithm 1 (uses the test labels)
- objective 'loro': leave-one-rating-out SSE on the training ratings (legitimate)

``scripts/partition_bound.py`` re-checks every result against ``ClusterRatingModel``.
"""

import numpy as np
import pandas as pd

from .data import GENRE_COLS


def _ci_from(S, N, Sg, Ng, Sc, Nc, sig, mu):
    """Algorithm 1 levels 1-4 computed from the cluster statistics."""
    with np.errstate(divide="ignore", invalid="ignore"):
        Sgi, Ngi = Sg[..., sig], Ng[..., sig]
        p3 = np.where(Nc > 0, Sc / np.maximum(Nc, 1), mu)[..., None]
        p2 = np.where(Ngi > 0, Sgi / np.maximum(Ngi, 1), p3)
        return np.where(N > 0, S / np.maximum(N, 1), p2)


class FoldStats:
    """Per-user statistics of one train/test fold (0-based user and item indices)."""

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, item_genres: pd.DataFrame,
                 n_users: int = 943, n_items: int = 1682):
        self.NU, self.NI = n_users, n_items
        g = item_genres.reindex(np.arange(1, n_items + 1))[GENRE_COLS].fillna(0).astype(int)
        _, self.sig = np.unique(g.astype(str).agg("".join, axis=1).to_numpy(), return_inverse=True)
        G = int(self.sig.max()) + 1
        tru, tri, trr = train.UID.to_numpy() - 1, train.MID.to_numpy() - 1, train.rate.to_numpy(float)
        teu, tei, ter = test.UID.to_numpy() - 1, test.MID.to_numpy() - 1, test.rate.to_numpy(float)
        self.teu, self.tei, self.ter = teu, tei, ter
        self.mu = float(trr.mean())
        R, M = np.zeros((n_users, n_items)), np.zeros((n_users, n_items))
        R[tru, tri], M[tru, tri] = trr, 1
        self.R, self.M, self.Q = R, M, R ** 2
        self.Rg, self.Mg = np.zeros((n_users, G)), np.zeros((n_users, G))
        np.add.at(self.Rg, (tru, self.sig[tri]), trr)
        np.add.at(self.Mg, (tru, self.sig[tri]), 1)
        self.Rc, self.Mc = R.sum(1), M.sum(1)
        self.Tn, self.Ts = np.zeros((n_users, n_items)), np.zeros((n_users, n_items))
        self.Tn[teu, tei], self.Ts[teu, tei] = 1, ter
        self.Tq = self.Ts ** 2
        if (M * self.Tn).any():
            raise ValueError("train/test (user, item) overlap")

    def rmse(self, labels: np.ndarray) -> float:
        K = int(labels.max()) + 1
        H = np.zeros((K, self.NU))
        H[labels, np.arange(self.NU)] = 1
        CI = _ci_from(H @ self.R, H @ self.M, H @ self.Rg, H @ self.Mg, H @ self.Rc, H @ self.Mc, self.sig, self.mu)
        pred = CI[labels[self.teu], self.tei]
        return float(np.sqrt(np.mean((self.ter - pred) ** 2)))


class PartitionSearch:
    """Local search over user partitions for the 'test' or 'loro' objective."""

    def __init__(self, fs: FoldStats, labels: np.ndarray, K: int, objective: str = "test"):
        if objective not in ("test", "loro"):
            raise ValueError(objective)
        self.fs, self.K, self.obj = fs, K, objective
        self.L = np.asarray(labels).copy()
        H = np.zeros((K, fs.NU))
        H[self.L, np.arange(fs.NU)] = 1
        self.cl = [H @ x for x in (fs.R, fs.M, fs.Q, fs.Rg, fs.Mg, fs.Rc, fs.Mc, fs.Tn, fs.Ts, fs.Tq)]
        self.size = H.sum(1)
        self.F = self.f(*self.cl)

    def _user(self, u):
        fs = self.fs
        return (fs.R[u], fs.M[u], fs.Q[u], fs.Rg[u], fs.Mg[u], fs.Rc[u], fs.Mc[u], fs.Tn[u], fs.Ts[u], fs.Tq[u])

    def f(self, S, N, Q, Sg, Ng, Sc, Nc, Tn, Ts, Tq):
        fs = self.fs
        if self.obj == "test":
            CI = _ci_from(S, N, Sg, Ng, Sc, Nc, fs.sig, fs.mu)
            return (Tq - 2 * CI * Ts + CI * CI * Tn).sum(-1)
        with np.errstate(divide="ignore", invalid="ignore"):  # leave-one-rating-out SSE
            Nm = np.maximum(N - 1, 1)
            many = np.where(N >= 2, N / (Nm * Nm) * (N * Q - S * S), 0.0)
            Sgi, Ngi = Sg[..., fs.sig], Ng[..., fs.sig]
            p3 = np.where((Nc >= 2)[..., None], (Sc[..., None] - S) / np.maximum(Nc - 1, 1)[..., None], fs.mu)
            p = np.where(Ngi >= 2, (Sgi - S) / np.maximum(Ngi - 1, 1), p3)
            one = np.where(N == 1, (S - p) ** 2, 0.0)
        return (many + one).sum(-1)

    def _deltas(self, u):
        a, su = self.L[u], self._user(u)
        fa_minus = self.f(*[c[a] - s for c, s in zip(self.cl, su)])
        fb_plus = self.f(*[c + s for c, s in zip(self.cl, su)])
        delta = (fa_minus - self.F[a]) + (fb_plus - self.F)
        delta[a] = 0.0
        return a, su, fa_minus, fb_plus, delta

    def _move(self, u, a, b, su, fa_minus, fb_plus):
        for c, s in zip(self.cl, su):
            c[a] -= s
            c[b] += s
        self.F[a], self.F[b] = fa_minus, fb_plus[b]
        self.size[a] -= 1
        self.size[b] += 1
        self.L[u] = b

    def descend(self, rng, max_sweeps: int = 60) -> None:
        """Move each user to the cluster that lowers the objective most, until nothing improves."""
        for _ in range(max_sweeps):
            moves = 0
            for u in rng.permutation(self.fs.NU):
                if self.size[self.L[u]] <= 1:
                    continue
                a, su, fa, fb, delta = self._deltas(u)
                b = int(np.argmin(delta))
                if b != a and delta[b] < -1e-9:
                    self._move(u, a, b, su, fa, fb)
                    moves += 1
            if moves == 0:
                return

    def anneal(self, rng, sweeps: int, t0: float = 3.0, t1: float = 0.02) -> None:
        """Simulated annealing with heat-bath moves; temperatures are in SSE units."""
        for t in range(sweeps):
            T = t0 * (t1 / t0) ** (t / max(sweeps - 1, 1))
            for u in rng.permutation(self.fs.NU):
                if self.size[self.L[u]] <= 1:
                    continue
                a, su, fa, fb, delta = self._deltas(u)
                w = np.exp(-(delta - delta.min()) / T)
                b = int(rng.choice(self.K, p=w / w.sum()))
                if b != a:
                    self._move(u, a, b, su, fa, fb)
