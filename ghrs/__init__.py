"""GHRS (Darban & Valipour, 2022) — Phase 1 reproducibility port for MovieLens 100K.

Steps 1-3 (graph + side-information features) are a faithful port of the authors'
``Features100K.py``. Steps 4-7 (autoencoder, K-Means, cluster-based rating
prediction) were never released by the authors and are reconstructed from the
paper (arXiv:2111.11293, Sec. 3.1, 4.3-4.6, Algorithm 1).
"""

import os

# Keras 3 must know its backend before the first import. TensorFlow has no wheels
# for Python 3.14, so the PyTorch backend is used. The Keras API is unchanged.
os.environ.setdefault("KERAS_BACKEND", "torch")
