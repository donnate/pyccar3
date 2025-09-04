import numpy as np
from typing import List, Optional, Tuple

# Prefer sparse SVD; fall back gracefully.
try:
    from scipy.sparse.linalg import svds as _svds
    _HAS_SVDS = True
except Exception:
    _HAS_SVDS = False

try:
    from sklearn.covariance import LedoitWolf
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception:
    _HAS_CVXPY = False


def soft_thresh(A: np.ndarray, lam: float) -> np.ndarray:
    """
    Elementwise soft-threshold: sign(A) * max(|A| - lam, 0)
    """
    return np.sign(A) * np.maximum(np.abs(A) - lam, 0.0)


def fnorm(A: np.ndarray) -> float:
    """Frobenius norm."""
    return float(np.sqrt(np.sum(A * A)))


def soft_thresh_group(A: np.ndarray, lam: float) -> np.ndarray:
    """
    Group-L2 soft threshold for a vector/array A.
    Returns A * max(0, 1 - lam / ||A||_2).
    """
    norm_A = float(np.sqrt(np.sum(A * A)))
    if norm_A == 0.0:
        return A
    scale = max(0.0, 1.0 - lam / norm_A)
    return A * scale


def soft_thresh2(A: np.ndarray, lam: float) -> np.ndarray:
    """
    Alias of soft_thresh_group (kept to mirror the R code).
    """
    return soft_thresh_group(A, lam)


def _standardize(X: np.ndarray) -> np.ndarray:
    """
    Column-wise standardization: (X - mean) / std.
    Columns with std==0 are left as-is to avoid divide-by-zero.
    """
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (X - mu) / sd


def _matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Alias for matrix multiplication (R's SMUT::eigenMapMatMult equivalent)."""
    return A @ B


def _top_r_svd(C: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute top-r SVD of C. Prefer scipy.sparse.linalg.svds when available.
    Returns (U, s, V) where C ~= U diag(s) V^T, s is descending.
    """
    p, q = C.shape
    if _HAS_SVDS and min(p, q) > r + 5:
        u, s, vt = _svds(C, k=r)
        # svds returns singular values in ascending order
        order = np.argsort(s)[::-1]
        s = s[order]; u = u[:, order]; vt = vt[order, :]
        v = vt.T
    else:
        U, S, VT = np.linalg.svd(C, full_matrices=False)
        u = U[:, :r]; s = S[:r]; v = VT[:r, :].T
    return u, s, v


def _safe_diag_inv(d: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    inv = np.zeros_like(d, dtype=float)
    mask = d > tol
    inv[mask] = 1.0 / d[mask]
    return inv


def _ledoit_wolf_cov(Y: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Ledoit-Wolf shrinkage of covariance of Y (n x q).
    Tries scikit-learn; otherwise falls back to sample covariance (Y^T Y / n).
    """
    n = Y.shape[0]
    if _HAS_SKLEARN:
        try:
            lw = LedoitWolf(store_precision=False, assume_centered=False)
            lw.fit(Y)  # sklearn expects samples as rows
            return lw.covariance_
        except Exception as e:
            if verbose:
                print(f"Ledoit-Wolf via sklearn failed ({e}); falling back to sample covariance.")
    # Fallback (note: assumes Y already centered if standardize=True)
    return (Y.T @ Y) / n

# --------------------------------------------------------------------
# Helpers: matrix square-root and inverse square-root (SPD-friendly)
# --------------------------------------------------------------------


def _symm_svd(S: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For a symmetric matrix S, return U, d, U^T such that S = U diag(d) U^T.
    We compute via eigen-decomposition for numerical symmetry.
    """
    d, U = np.linalg.eigh(S)
    # ascending -> descending (optional)
    order = np.argsort(d)[::-1]
    d = d[order]
    U = U[:, order]
    return U, d, U.T


def compute_sqrt_inv(S: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    """
    Returns S^{-1/2} using spectral mapping with thresholding on eigenvalues:
      lambda -> 1/sqrt(lambda) if lambda > threshold else 0
    """
    U, d, UT = _symm_svd(S)
    inv_sqrt_vals = np.where(d > threshold, 1.0 / np.sqrt(d), 0.0)
    return U @ np.diag(inv_sqrt_vals) @ UT


def compute_sqrt(S: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    """
    Returns S^{1/2} using spectral mapping with thresholding on eigenvalues:
      lambda -> sqrt(lambda) if lambda > threshold else 0
    """
    U, d, UT = _symm_svd(S)
    sqrt_vals = np.where(d > threshold, np.sqrt(d), 0.0)
    return U @ np.diag(sqrt_vals) @ UT


def _symm_eig(S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    d, U = np.linalg.eigh(S)
    order = np.argsort(d)[::-1]
    return U[:, order], d[order]
