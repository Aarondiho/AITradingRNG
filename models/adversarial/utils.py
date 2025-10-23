import numpy as np
import random
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix


def build_feature_matrix(records, windows=(5, 30, 60)):
    """
    Build a feature matrix X and label vector y from feature shard records.
    Only numeric fields are included.
    """
    X = []
    for r in records:
        row = []
        # Always include raw quote
        row.append(float(r.get("quote", 0.0)))
        # Rolling stats
        for w in windows:
            row.append(float(r.get(f"mean_{w}", 0.0)))
            row.append(float(r.get(f"var_{w}", 0.0)))
            row.append(float(r.get(f"momentum_{w}", 0.0)))
        X.append(row)
    return np.array(X, dtype=np.float64)


def zscore_fit(X):
    """
    Compute mean and std for each column of X.
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma[sigma == 0] = 1.0  # avoid division by zero
    return mu, sigma


def zscore_apply(X, mu, sigma):
    """
    Apply z-score normalization.
    """
    return (X - mu) / sigma


def synthetic_generate(X_real, mode="gaussian", seed=42):
    """
    Generate synthetic negatives with same shape as X_real.
    Modes:
      - gaussian: N(mean, std) per feature
      - uniform: U(min, max) per feature
      - shuffle: permute each column independently
    """
    np.random.seed(seed)
    X_synth = np.zeros_like(X_real)

    if mode == "gaussian":
        mu = np.mean(X_real, axis=0)
        sigma = np.std(X_real, axis=0)
        X_synth = np.random.normal(mu, sigma, size=X_real.shape)

    elif mode == "uniform":
        minv = np.min(X_real, axis=0)
        maxv = np.max(X_real, axis=0)
        X_synth = np.random.uniform(minv, maxv, size=X_real.shape)

    elif mode == "shuffle":
        X_synth = np.copy(X_real)
        for j in range(X_real.shape[1]):
            np.random.shuffle(X_synth[:, j])

    else:
        raise ValueError(f"Unknown synthetic mode: {mode}")

    return X_synth


def split_train_test(X, y, ratio=0.7, seed=42):
    """
    Deterministic train/test split.
    """
    np.random.seed(seed)
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    split = int(len(y) * ratio)
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def metrics_roc_auc(y_true, y_score):
    return float(roc_auc_score(y_true, y_score))


def metrics_pr_auc(y_true, y_score):
    return float(average_precision_score(y_true, y_score))


def metrics_confusion(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return cm
