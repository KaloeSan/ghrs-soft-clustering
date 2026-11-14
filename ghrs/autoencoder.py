"""Step 4 of GHRS: an autoencoder that compresses the 35 user features to a 4-d code.

The authors did not publish this part. The architecture follows Section 4.5 and
Figs. 8-11 of the paper: input, hidden, bottleneck, hidden and output layers with
ReLU everywhere (the output layer included), MSE loss, Adam, 100 epochs, elastic-net
regularisation and 20% of the users held out for validation. After training, all
943 users are encoded.

A few values are not given in the paper, so they were chosen to match its figures:

- hidden width 16. Fig. 8 shows 8 hidden nodes, but it also shows 10 input nodes for
  35 inputs, so the drawing is schematic. Width 8 gives the same RMSE.
- batch size 64, which reproduces the validation curves of Fig. 9 best.
- L1 = L2 = 1e-5. Keras adds the penalty to ``val_loss``, and the starting value in
  Fig. 9 (about 0.107) only fits a small penalty. The old Keras 2 default of 1e-2
  is too strong and collapses the code.

None of these choices changes the 5-fold RMSE by more than about 0.005. The output
activation, code activation and BatchNorm options are extensions used only in the
hyperparameter search; their defaults reproduce the paper's model.
"""

from dataclasses import dataclass

import keras
import numpy as np
from keras import layers, regularizers
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class AEConfig:
    hidden_dim: int = 16           # not in the paper, see module docstring
    code_dim: int = 4              # Fig. 10
    l1: float = 1e-5               # elastic net, strength fitted to Fig. 9
    l2: float = 1e-5
    epochs: int = 100              # Fig. 9
    batch_size: int = 64           # not in the paper, fitted to Fig. 9
    learning_rate: float = 1e-3    # not in the paper, Adam default
    val_fraction: float = 0.2      # 20% of the users for ML-100K
    # Extensions; the defaults match the paper.
    output_activation: str = "relu"
    code_activation: str = "relu"
    batchnorm: bool = False


def build_autoencoder(input_dim: int, cfg: AEConfig) -> tuple[keras.Model, keras.Model]:
    reg = lambda: regularizers.L1L2(l1=cfg.l1, l2=cfg.l2)  # noqa: E731

    def dense(x, units, act, name):
        x = layers.Dense(units, activation=None if cfg.batchnorm else act,
                         kernel_regularizer=reg(), name=name)(x)
        if cfg.batchnorm:
            x = layers.Activation(act, name=f"{name}_act")(layers.BatchNormalization(name=f"{name}_bn")(x))
        return x

    inp = keras.Input(shape=(input_dim,), name="F_t")
    h = dense(inp, cfg.hidden_dim, "relu", "enc_hidden")
    code = dense(h, cfg.code_dim, cfg.code_activation, "code")
    h = dense(code, cfg.hidden_dim, "relu", "dec_hidden")
    out = layers.Dense(input_dim, activation=cfg.output_activation, kernel_regularizer=reg(),
                       name="reconstruction")(h)
    autoencoder = keras.Model(inp, out, name="ghrs_autoencoder")
    encoder = keras.Model(inp, code, name="ghrs_encoder")
    autoencoder.compile(optimizer=keras.optimizers.Adam(cfg.learning_rate), loss="mse")
    return autoencoder, encoder


def active_dims(F: np.ndarray, rel_tol: float = 1e-3, abs_tol: float = 1e-6) -> np.ndarray:
    """Boolean mask of the code units that are not dead.

    A unit counts as dead when its range of values is tiny compared with the widest
    unit, or below an absolute floor. If the whole code has collapsed, no unit is
    active. The check is done in float64 to avoid float32 rounding noise.
    """
    F = np.asarray(F, dtype=np.float64)
    spread = F.max(axis=0) - F.min(axis=0)
    return spread > max(rel_tol * float(spread.max(initial=0.0)), abs_tol)


class _PureValMSE(keras.callbacks.Callback):
    """Records the validation reconstruction MSE without the L1/L2 penalty."""

    def __init__(self, X_val: np.ndarray):
        super().__init__()
        self.X_val, self.values = X_val, []

    def on_epoch_end(self, epoch, logs=None):
        rec = keras.ops.convert_to_numpy(self.model(self.X_val, training=False))
        self.values.append(float(np.mean((rec - self.X_val) ** 2)))


def encode_users(X: np.ndarray, cfg: AEConfig, seed: int,
                 details: dict | None = None) -> tuple[np.ndarray, dict]:
    """Train the autoencoder on 80% of the users and encode all of them.

    Returns the raw codes (scaling for K-Means is done in ``clustering.scale_codes``)
    and a dict with the final losses. If ``details`` is a dict, the per-epoch
    validation curves are added to it as well.
    """
    keras.utils.set_random_seed(seed)  # seeds Python, NumPy and torch
    X = np.asarray(X, dtype="float32")
    X_tr, X_val = train_test_split(X, test_size=cfg.val_fraction, random_state=seed)
    autoencoder, encoder = build_autoencoder(X.shape[1], cfg)
    cb = _PureValMSE(X_val)
    hist = autoencoder.fit(X_tr, X_tr, validation_data=(X_val, X_val), epochs=cfg.epochs,
                           batch_size=cfg.batch_size, shuffle=True, verbose=0, callbacks=[cb])
    F_e = encoder.predict(X, batch_size=1024, verbose=0)
    info = {
        "train_loss": float(hist.history["loss"][-1]),    # MSE plus penalty
        "val_loss": float(hist.history["val_loss"][-1]),  # MSE plus penalty, as in Fig. 9
        "val_mse": cb.values[-1],                         # reconstruction error only
        "active_code_dims": int(active_dims(F_e).sum()),
    }
    if details is not None:
        details.update(val_mse_curve=cb.values, val_loss_curve=[float(v) for v in hist.history["val_loss"]])
    return F_e, info
