"""Phase 2: a variational autoencoder (VAE) as an alternative to the step-4 autoencoder.

The encoder outputs a Gaussian posterior q(z | x) = N(mu(x), diag(sigma^2(x))) instead
of a single point. Training maximises the evidence lower bound: a reconstruction term
summed over the 35 features plus ``beta`` times the KL divergence to a standard normal
prior. The KL term keeps the latent space smooth and centred, which is expected to
help users with little information, such as cold-start users. Note that cold-start
users look like the many warm users without graph edges (all graph features are zero
for about half of the users at alpha = 0.01), so any benefit is measured empirically
in the cold-start section of ``scripts/run_phase2.py``. The posterior mean mu(x) is
used as the user's code for clustering.

The layer sizes, epochs, batch size and validation split follow the Phase 1
autoencoder (35-16-4-16-35). The optimizer defaults to Nadam. All input features lie
in [0, 1], so the default decoder has a sigmoid output trained with binary
cross-entropy; ``recon_loss='mse'`` with any output activation is also available.
"""

from dataclasses import dataclass

import keras
import numpy as np
from keras import layers, ops, regularizers
from sklearn.model_selection import train_test_split

from .autoencoder import PureValMSE, active_dims

OPTIMIZERS = {
    "nadam": keras.optimizers.Nadam,
    "adam": keras.optimizers.Adam,
    "rmsprop": keras.optimizers.RMSprop,
    "sgd": keras.optimizers.SGD,
}
RECON_LOSSES = ("bce", "mse")


@dataclass(frozen=True)
class VAEConfig:
    hidden_dim: int = 16
    code_dim: int = 4
    hidden_activation: str = "relu"
    output_activation: str = "sigmoid"
    recon_loss: str = "bce"          # summed over the features: 'bce' or 'mse'
    beta: float = 1.0                # weight of the KL term (1 = standard VAE)
    kl_warmup_epochs: int = 0        # linear KL annealing from beta / n to beta over n epochs, 0 = off
    l2: float = 0.0
    optimizer: str = "nadam"
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    val_fraction: float = 0.2

    def __post_init__(self):
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer {self.optimizer!r}, choose from {sorted(OPTIMIZERS)}")
        if self.recon_loss not in RECON_LOSSES:
            raise ValueError(f"unknown reconstruction loss {self.recon_loss!r}")
        if self.recon_loss == "bce" and self.output_activation != "sigmoid":
            raise ValueError("binary cross-entropy needs output_activation='sigmoid'")
        if self.beta < 0 or self.kl_warmup_epochs < 0 or self.l2 < 0:
            raise ValueError("beta, kl_warmup_epochs and l2 must be non-negative")
        if self.epochs < 1 or self.kl_warmup_epochs > self.epochs:
            raise ValueError("epochs must be positive and kl_warmup_epochs must not exceed it")


def make_optimizer(name: str, learning_rate: float) -> keras.optimizers.Optimizer:
    return OPTIMIZERS[name](learning_rate=learning_rate)


def _reconstruction_loss(kind: str):
    """Per-user reconstruction error summed over the features (Keras then averages over the batch)."""
    def bce(y_true, y_pred):
        p = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        return -ops.sum(y_true * ops.log(p) + (1.0 - y_true) * ops.log(1.0 - p), axis=-1)

    def mse(y_true, y_pred):
        return ops.sum(ops.square(y_true - y_pred), axis=-1)

    return bce if kind == "bce" else mse


class KLDivergence(layers.Layer):
    """Adds beta * KL(q(z|x) || N(0, I)) to the model loss and passes the mean through."""

    def __init__(self, beta: float, **kwargs):
        super().__init__(**kwargs)
        self.beta = self.add_weight(shape=(), initializer=keras.initializers.Constant(beta),
                                    trainable=False, name="beta")

    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl = -0.5 * ops.sum(1.0 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=-1)
        self.add_loss(self.beta * ops.mean(kl))
        return z_mean


class Sampling(layers.Layer):
    """Reparameterisation trick: z = mu + sigma * eps while training, z = mu otherwise."""

    def __init__(self, seed: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, inputs, training=None):
        z_mean, z_log_var = inputs
        if not training:
            return z_mean
        eps = keras.random.normal(ops.shape(z_mean), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * eps


class _KLWarmup(keras.callbacks.Callback):
    """Raises the KL weight linearly from beta / n to beta over the first n epochs."""

    def __init__(self, kl_layer: KLDivergence, beta: float, n_epochs: int):
        super().__init__()
        self.kl_layer, self.beta, self.n_epochs = kl_layer, beta, n_epochs

    def on_epoch_begin(self, epoch, logs=None):
        self.kl_layer.beta.assign(self.beta * min(1.0, (epoch + 1) / self.n_epochs))


def build_vae(input_dim: int, cfg: VAEConfig, seed: int | None = None
              ) -> tuple[keras.Model, keras.Model, KLDivergence]:
    """Return the trainable VAE, an encoder giving (mu, log sigma^2) and the KL layer."""
    reg = (lambda: regularizers.L2(cfg.l2)) if cfg.l2 else (lambda: None)
    act = cfg.hidden_activation

    inp = keras.Input(shape=(input_dim,), name="F_t")
    h = layers.Dense(cfg.hidden_dim, activation=act, kernel_regularizer=reg(), name="enc_hidden")(inp)
    z_mean = layers.Dense(cfg.code_dim, kernel_regularizer=reg(), name="z_mean")(h)
    z_log_var = layers.Dense(cfg.code_dim, kernel_regularizer=reg(), name="z_log_var")(h)
    kl_layer = KLDivergence(cfg.beta, name="kl")
    z = Sampling(seed, name="z")([kl_layer([z_mean, z_log_var]), z_log_var])
    h = layers.Dense(cfg.hidden_dim, activation=act, kernel_regularizer=reg(), name="dec_hidden")(z)
    out = layers.Dense(input_dim, activation=cfg.output_activation, kernel_regularizer=reg(),
                       name="reconstruction")(h)

    vae = keras.Model(inp, out, name="ghrs_vae")
    encoder = keras.Model(inp, [z_mean, z_log_var], name="ghrs_vae_encoder")
    vae.compile(optimizer=make_optimizer(cfg.optimizer, cfg.learning_rate),
                loss=_reconstruction_loss(cfg.recon_loss))
    return vae, encoder, kl_layer


def encode_users_vae(X: np.ndarray, cfg: VAEConfig, seed: int,
                     details: dict | None = None) -> tuple[np.ndarray, dict]:
    """Train the VAE on 80% of the users and return the posterior means of all users.

    The returned dict mirrors ``autoencoder.encode_users`` and adds the average KL
    divergence and the number of active units (variance of mu above 0.01, Burda et al.).
    """
    keras.utils.set_random_seed(seed)
    X = np.asarray(X, dtype="float32")
    X_tr, X_val = train_test_split(X, test_size=cfg.val_fraction, random_state=seed)
    vae, encoder, kl_layer = build_vae(X.shape[1], cfg, seed)
    mse_cb = PureValMSE(X_val)
    callbacks = [mse_cb]
    if cfg.kl_warmup_epochs:
        callbacks.append(_KLWarmup(kl_layer, cfg.beta, cfg.kl_warmup_epochs))
    hist = vae.fit(X_tr, X_tr, validation_data=(X_val, X_val), epochs=cfg.epochs,
                   batch_size=cfg.batch_size, shuffle=True, verbose=0, callbacks=callbacks)
    z_mean, z_log_var = (np.asarray(a, dtype=np.float64) for a in encoder.predict(X, batch_size=1024, verbose=0))
    kl = -0.5 * np.sum(1.0 + z_log_var - z_mean ** 2 - np.exp(z_log_var), axis=1)
    info = {
        "train_loss": float(hist.history["loss"][-1]),    # reconstruction + beta * KL
        "val_loss": float(hist.history["val_loss"][-1]),
        "val_mse": mse_cb.values[-1],                     # reconstruction from mu only
        "kl": float(kl.mean()),
        "active_units": int((z_mean.var(axis=0) > 0.01).sum()),
        "active_code_dims": int(active_dims(z_mean).sum()),
    }
    if details is not None:
        details.update(val_mse_curve=mse_cb.values, val_loss_curve=[float(v) for v in hist.history["val_loss"]])
    return z_mean.astype("float32"), info
