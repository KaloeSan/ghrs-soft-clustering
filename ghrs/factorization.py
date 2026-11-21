"""SVD++ with GHRS side information, used as a strong baseline in the final synthesis.

    r_ui = mu + b_u + b_i + (p_u + |N(u)|^-1/2 sum_j y_j + A f_u) . (q_i + B g_i)
           + f_u . a + g_i . b  [+ b_age(movie age at rating) + b_tenure(user tenure)]

f_u are the 35 GHRS user features (standardised), g_i the 19 genre flags, N(u) the
movies the user rated in the training data. All parameters are fitted with full-batch
Adam on the training ratings; 5% of them are held out for early stopping, so the test
fold is never seen.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .data import GENRE_COLS, load_release_years

AGE_EDGES = np.array([0, 1, 2, 3, 5, 8, 12, 20, 40])
TENURE_BINS = 8


@dataclass(frozen=True)
class SVDppConfig:
    k: int = 32                  # latent dimension (chosen most often on inner validation)
    lam: float = 0.1             # weight of the L2 penalty
    lr: float = 0.03
    epochs: int = 400
    patience: int = 25
    val_fraction: float = 0.05
    time_biases: bool = False    # movie age at rating and user tenure (no per-day term)
    n_splits_oof: int = 5        # inner folds for out-of-fold training predictions
    seeds: tuple[int, ...] = (42, 0, 1)   # predictions are averaged over these seeds


def standardize(X: np.ndarray) -> np.ndarray:
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


class _SVDpp(torch.nn.Module):
    def __init__(self, n_users, n_items, F, G, N, k, time_biases, n_tenure):
        super().__init__()
        self.F, self.G, self.N, self.time_biases = F, G, N, time_biases
        rand = lambda *shape: torch.nn.Parameter(0.05 * torch.randn(*shape))  # noqa: E731
        zero = lambda *shape: torch.nn.Parameter(torch.zeros(*shape))  # noqa: E731
        self.bu, self.bi = zero(n_users), zero(n_items)
        self.P, self.Q, self.Y = rand(n_users, k), rand(n_items, k), rand(n_items, k)
        self.A, self.B = rand(F.shape[1], k), rand(G.shape[1], k)
        self.a, self.b = zero(F.shape[1]), zero(G.shape[1])
        self.b_age, self.b_ten = zero(len(AGE_EDGES) + 1), zero(n_tenure + 1)

    def vectors(self, u, i):
        implicit = torch.sparse.mm(self.N, self.Y)
        return self.P[u] + implicit[u] + self.F[u] @ self.A, self.Q[i] + self.G[i] @ self.B

    def forward(self, u, i, mu, age, ten):
        pu, qi = self.vectors(u, i)
        out = mu + self.bu[u] + self.bi[i] + (pu * qi).sum(1) + self.F[u] @ self.a + self.G[i] @ self.b
        if self.time_biases:
            out = out + self.b_age[age] + self.b_ten[ten]
        return out

    def penalty(self, u, i):
        pu, qi = self.vectors(u, i)
        pen = ((self.bu[u] ** 2 + self.bi[i] ** 2).mean() + (pu ** 2).sum(1).mean() + (qi ** 2).sum(1).mean()
               + 50 * ((self.a ** 2).sum() + (self.b ** 2).sum()) / len(u))
        if self.time_biases:
            pen = pen + (self.b_age ** 2).mean() + (self.b_ten ** 2).mean()
        return pen


class FittedSVDpp:
    """A trained model that predicts ratings for any (UID, MID, time) frame."""

    def __init__(self, model, mu, first_ts, tenure_edges, release, val_rmse):
        self.model, self.mu, self.first_ts = model, mu, first_ts
        self.tenure_edges, self.release, self.val_rmse = tenure_edges, release, val_rmse

    def _time_index(self, df: pd.DataFrame):
        t = df["time"].to_numpy()
        year = pd.to_datetime(t, unit="s").year.to_numpy()
        age = np.digitize(year - self.release[df["MID"].to_numpy()], AGE_EDGES)
        tenure = np.log1p(np.maximum(t - self.first_ts[df["UID"].to_numpy()], 0) / 86400.0)
        return torch.tensor(age), torch.tensor(np.digitize(tenure, self.tenure_edges))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        u = torch.tensor(df["UID"].to_numpy())
        i = torch.tensor(df["MID"].to_numpy())
        with torch.no_grad():
            return self.model(u, i, self.mu, *self._time_index(df)).clamp(1, 5).numpy().astype(float)


def _implicit_matrix(train: pd.DataFrame, n_users: int, n_items: int) -> torch.Tensor:
    u, i = train["UID"].to_numpy(), train["MID"].to_numpy()
    count = np.bincount(u, minlength=n_users)
    values = (1.0 / np.sqrt(np.maximum(count[u], 1))).astype(np.float32)
    return torch.sparse_coo_tensor(np.vstack([u, i]), values, (n_users, n_items), check_invariants=True).coalesce()


def item_features(item_genres: pd.DataFrame, n_items: int) -> np.ndarray:
    G = np.zeros((n_items, len(GENRE_COLS)), np.float32)
    G[item_genres.index.to_numpy()] = item_genres[GENRE_COLS].to_numpy()
    return G


def fit_svdpp(train: pd.DataFrame, user_features: np.ndarray, item_feats: np.ndarray,
              cfg: SVDppConfig, seed: int) -> FittedSVDpp:
    """Fit SVD++ on ``train``; ``user_features`` and ``item_feats`` are indexed by UID and MID."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_users, n_items = user_features.shape[0], item_feats.shape[0]
    F, G = torch.tensor(user_features, dtype=torch.float32), torch.tensor(item_feats, dtype=torch.float32)

    first_ts = np.full(n_users, np.inf)      # users without training ratings get tenure 0
    first = train.groupby("UID")["time"].min()
    first_ts[first.index.to_numpy()] = first.to_numpy()
    tenure = np.log1p((train["time"].to_numpy() - first_ts[train["UID"].to_numpy()]) / 86400.0)
    tenure_edges = np.quantile(tenure, np.linspace(0, 1, TENURE_BINS + 1)[1:-1])
    release = np.full(n_items, np.nan)
    years = load_release_years()
    release[years.index.to_numpy()] = years.to_numpy()
    release = np.where(np.isnan(release), np.nanmedian(release), release)

    model = _SVDpp(n_users, n_items, F, G, _implicit_matrix(train, n_users, n_items), cfg.k,
                   cfg.time_biases, len(tenure_edges))
    fitted = FittedSVDpp(model, 0.0, first_ts, tenure_edges, release, np.inf)
    u, i = torch.tensor(train["UID"].to_numpy()), torch.tensor(train["MID"].to_numpy())
    r = torch.tensor(train["rate"].to_numpy(), dtype=torch.float32)
    age, ten = fitted._time_index(train)
    val = torch.tensor(rng.random(len(r)) < cfg.val_fraction)
    fit = ~val
    mu = float(r[fit].mean())

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best, state, bad = np.inf, None, 0
    for _ in range(cfg.epochs):
        opt.zero_grad()
        pred = model(u[fit], i[fit], mu, age[fit], ten[fit])
        loss = ((pred - r[fit]) ** 2).mean() + cfg.lam * model.penalty(u[fit], i[fit])
        loss.backward()
        opt.step()
        with torch.no_grad():
            v = float(((model(u[val], i[val], mu, age[val], ten[val]).clamp(1, 5) - r[val]) ** 2).mean().sqrt())
        if v < best - 1e-5:
            best, bad = v, 0
            state = {n: p.detach().clone() for n, p in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(state)

    # Users and movies without training ratings were never updated: use zero latent
    # factors for them, so their predictions rest on the biases and side features only.
    unseen_users = np.setdiff1d(np.arange(n_users), train["UID"].unique())
    unseen_items = np.setdiff1d(np.arange(n_items), train["MID"].unique())
    with torch.no_grad():
        model.P[unseen_users] = 0.0
        model.Q[unseen_items] = 0.0
    fitted.mu, fitted.val_rmse = mu, best
    return fitted


def out_of_fold_predictions(train: pd.DataFrame, user_features: np.ndarray, item_feats: np.ndarray,
                            cfg: SVDppConfig, seed: int) -> np.ndarray:
    """Prediction for every training rating from a model that did not see that rating.

    Residuals of a flexible model on its own training data are far too small; these
    out-of-fold predictions give honest residuals for the cluster correction.
    """
    fold_id = np.random.default_rng(seed + 1).integers(0, cfg.n_splits_oof, len(train))
    oof = np.empty(len(train))
    for f in range(cfg.n_splits_oof):
        held = fold_id == f
        model = fit_svdpp(train[~held], user_features, item_feats, cfg, seed)
        oof[held] = model.predict(train[held])
    return oof
