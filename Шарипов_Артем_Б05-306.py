import numpy as np


def euclidean_distance(X, Y) -> np.ndarray:
    return np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=-1))


def cosine_distance(X, Y) -> np.ndarray:
    num = (X[:, None, :] * Y[None, :, :]).sum(axis=-1)
    den = np.sqrt((X ** 2).sum(axis=1)[:, None] * (Y ** 2).sum(axis=1)[None, :])
    cos_sim = num / den
    return 1 - cos_sim


def manhattan_distance(X, Y) -> np.ndarray:
    return ((abs(X[:, None, :] - Y[None, :, :])).sum(axis=-1))