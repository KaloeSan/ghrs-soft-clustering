"""Reimplementation of GHRS (Darban & Valipour, 2022) for MovieLens 100K.

Steps 1-3 (similarity graph and side information) are ported from the authors'
``Features100K.py``. Steps 4-7 (autoencoder, K-Means and the cluster-based rating
prediction) were never published, so they are rebuilt from the paper
(arXiv:2111.11293, Sections 3.1 and 4.3-4.6, Algorithm 1).
"""

import os

# Keras 3 reads the backend name on first import. TensorFlow has no wheels for
# Python 3.14, so we use the PyTorch backend; the Keras API stays the same.
os.environ.setdefault("KERAS_BACKEND", "torch")
