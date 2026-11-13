"""GHRS step 4: autoencoder dimension reduction F_t (35-d) -> F_e (4-d).

NOT released by the authors. Reconstructed from the paper (Sec. 4.5, Fig. 8, Fig. 10):
  * 5 layers: input -> hidden -> bottleneck -> hidden -> output (input = output = 35)
  * ReLU on all layers, MSE loss, Adam optimizer, 100 epochs
  * elastic-net (L1 + L2) regularisation
  * 20% of users randomly held out as validation data
  * bottleneck size 4 ("Dimension is reduced to 4", Fig. 10)
Values the paper does NOT give (hidden width, L1/L2 strength, batch size, learning
rate) are exposed in AEConfig with documented defaults.
"""

from dataclasses import dataclass

import keras
import numpy as np
from keras import layers, regularizers
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class AEConfig:
    hidden_dim: int = 16       # not in paper (assumption)
    code_dim: int = 4          # paper, Fig. 10
    l1: float = 1e-5           # paper: elastic net, strength not given (assumption)
    l2: float = 1e-5           # idem
    epochs: int = 100          # paper, Fig. 9
    batch_size: int = 32       # not in paper -> Keras default
    learning_rate: float = 1e-3  # not in paper -> Adam default
    val_fraction: float = 0.2  # paper: 20% of users for ML-100K


def build_autoencoder(input_dim: int, cfg: AEConfig) -> tuple[keras.Model, keras.Model]:
    reg = lambda: regularizers.L1L2(l1=cfg.l1, l2=cfg.l2)  # noqa: E731
    inp = keras.Input(shape=(input_dim,), name="F_t")
    h = layers.Dense(cfg.hidden_dim, activation="relu", kernel_regularizer=reg(), name="enc_hidden")(inp)
    code = layers.Dense(cfg.code_dim, activation="relu", kernel_regularizer=reg(), name="code")(h)
    h = layers.Dense(cfg.hidden_dim, activation="relu", kernel_regularizer=reg(), name="dec_hidden")(code)
    out = layers.Dense(input_dim, activation="relu", kernel_regularizer=reg(), name="reconstruction")(h)
    autoencoder = keras.Model(inp, out, name="ghrs_autoencoder")
    encoder = keras.Model(inp, code, name="ghrs_encoder")
    autoencoder.compile(optimizer=keras.optimizers.Adam(cfg.learning_rate), loss="mse")
    return autoencoder, encoder


def encode_users(X: np.ndarray, cfg: AEConfig, seed: int) -> tuple[np.ndarray, dict]:
    """Train the AE on 80% of users, validate on 20%, then encode *all* users."""
    keras.utils.set_random_seed(seed)  # seeds python, numpy and torch
    X = np.asarray(X, dtype="float32")
    X_tr, X_val = train_test_split(X, test_size=cfg.val_fraction, random_state=seed)
    autoencoder, encoder = build_autoencoder(X.shape[1], cfg)
    hist = autoencoder.fit(X_tr, X_tr, validation_data=(X_val, X_val), epochs=cfg.epochs,
                           batch_size=cfg.batch_size, shuffle=True, verbose=0)
    F_e = encoder.predict(X, batch_size=1024, verbose=0)
    info = {
        "train_loss": float(hist.history["loss"][-1]),
        "val_loss": float(hist.history["val_loss"][-1]),
        "active_code_dims": int((F_e.std(axis=0) > 1e-8).sum()),  # dead-ReLU check
    }
    return F_e, info
