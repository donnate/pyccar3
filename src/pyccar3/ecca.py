from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._helper import (
    _matmul, _standardize, _fnorm, _safe_diag_inv, soft_thresh,
    soft_thresh_group, soft_thresh2, _top_r_svd)

# Prefer sparse SVD; fall back gracefully.
try:
    from scipy.sparse.linalg import svds as _svds
    _HAS_SVDS = True
except Exception:
    _HAS_SVDS = False

# --------------------------
# Core algorithm (ADMM path)
# --------------------------

@dataclass
class ECCAResult:
    U: np.ndarray
    V: np.ndarray
    cor: np.ndarray
    loss: float


def ecca(
    X: np.ndarray,
    Y: np.ndarray,
    lambda_: float = 0.0,
    groups: Optional[List[np.ndarray]] = None,
    Sx: Optional[np.ndarray] = None,
    Sy: Optional[np.ndarray] = None,
    Sxy: Optional[np.ndarray] = None,
    r: int = 2,
    standardize: bool = False,
    rho: float = 1.0,
    B0: Optional[np.ndarray] = None,
    eps: float = 1e-4,
    maxiter: int = 500,
    verbose: bool = True,
) -> ECCAResult:
    """
    Python port of the R `ecca` (group-sparse reduced-rank regression for CCA).
    Matches the algebra and stopping rule closely.

    Parameters
    ----------
    X : (n, p) array
    Y : (n, q) array
    lambda_ : float
    groups : list of index arrays (see 'Groups format' below), or None for elementwise soft-threshold.
    Sx, Sy, Sxy : covariances/cross-covariances; if None they are computed from X,Y as in R.
    r : int, target rank
    standardize : bool, whether to standardize X and Y columns
    rho : float, ADMM parameter
    B0 : optional initial (p, q) array
    eps : float, relative change tolerance on B
    maxiter : int, maximum ADMM iterations
    verbose : bool

    Returns
    -------
    ECCAResult(U, V, cor, loss)

    Groups format
    -------------
    Pass a list where each element is a (k, 2) int array of zero-based pairs (row_idx, col_idx)
    identifying entries in the (p x q) matrix that belong to that group (mirrors the R code).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, p = X.shape[0], X.shape[1]
    q = Y.shape[1]

    if standardize:
        X = _standardize(X)
        Y = _standardize(Y)

    if Sxy is None:
        Sxy = _matmul(X.T, Y) / n
    if Sx is None:
        Sx = _matmul(X.T, X) / n
    if Sy is None:
        Sy = _matmul(Y.T, Y) / n

    B = np.zeros((p, q), dtype=float) if B0 is None else B0.astype(float, copy=True)

    # Symmetric eigendecompositions (ascending eigenvalues)
    Lx, Ux = np.linalg.eigh(Sx)
    Ly, Uy = np.linalg.eigh(Sy)

    # Build Sx^{1/2} and Sy^{1/2} via spectral mapping (clip small negatives)
    Lx = np.maximum(Lx, 0.0)
    Ly = np.maximum(Ly, 0.0)
    Sx12 = _matmul(_matmul(Ux, np.diag(np.sqrt(Lx))), Ux.T)
    Sy12 = _matmul(_matmul(Uy, np.diag(np.sqrt(Ly))), Uy.T)

    # Precompute terms for ADMM
    b = np.add.outer(Lx, Ly) + rho               # shape (p, q)
    B1 = _matmul(_matmul(Ux.T, Sxy), Uy)         # shape (p, q)

    H = np.zeros_like(B)
    Z = B.copy()
    delta = math.inf
    it = 0

    while (delta > eps) and (it < maxiter):
        it += 1
        B_prev = B.copy()

        # B update
        Btilde = B1 + rho * (_matmul(_matmul(Ux.T, (Z - H)), Uy))
        Btilde = Btilde / b
        B = _matmul(_matmul(Ux, Btilde), Uy.T)

        # Z update (prox)
        Z = B + H
        if groups is None:
            Z = soft_thresh(Z, lambda_ / rho)
        else:
            # Group-wise L2 soft threshold over specified entries
            for g_idx in groups:
                g_idx = np.asarray(g_idx, dtype=int)
                if g_idx.ndim != 2 or g_idx.shape[1] != 2:
                    raise ValueError("Each group must be an array of shape (k, 2) with (row, col) pairs.")
                rows = g_idx[:, 0]
                cols = g_idx[:, 1]
                subset = Z[rows, cols]
                lam_g = math.sqrt(len(rows)) * (lambda_ / rho)
                Z[rows, cols] = soft_thresh2(subset, lam_g)

        # H update
        H = H + rho * (B - Z)

        # Relative change
        sB0 = float(np.sum(B_prev ** 2))
        delta = float(np.sum((B - B_prev) ** 2) / sB0) if sB0 > 1e-20 else math.inf

        if verbose and it % 10 == 0:
            print(f"iter: {it:d} delta: {delta:.3e}")

    if it >= maxiter and verbose:
        print("ADMM did not converge!")
    elif verbose:
        print(f"ADMM converged in {it} iterations")

    # Map back
    B = Z
    C = _matmul(_matmul(Sx12, B), Sy12)

    U0, L0, V0 = _top_r_svd(C, r=r)
    inv_L0 = _safe_diag_inv(L0, tol=1e-8)

    if float(np.max(L0)) > 1e-8:
        U = _matmul(_matmul(_matmul(B, Sy12), V0), np.diag(inv_L0))
        V = _matmul(_matmul(_matmul(B.T, Sx12), U0), np.diag(inv_L0))
        # Match R loss: mean over components of sum of squared residuals
        resid = _matmul(X, U) - _matmul(Y, V)
        loss = float(np.mean(np.sum(resid ** 2, axis=0)))
        cor = np.diag(_matmul(_matmul(U.T, Sxy), V))
        return ECCAResult(U=U, V=V, cor=cor, loss=loss)
    else:
        U = np.full((p, r), np.nan, dtype=float)
        V = np.full((q, r), np.nan, dtype=float)
        return ECCAResult(U=U, V=V, cor=np.zeros(r, dtype=float), loss=math.inf)


def ecca_across_lambdas(
    X: np.ndarray,
    Y: np.ndarray,
    lambdas: Sequence[float],
    groups: Optional[List[np.ndarray]] = None,
    r: int = 2,
    Sx: Optional[np.ndarray] = None,
    Sy: Optional[np.ndarray] = None,
    Sxy: Optional[np.ndarray] = None,
    standardize: bool = True,
    rho: float = 1.0,
    B0: Optional[np.ndarray] = None,
    eps: float = 1e-4,
    maxiter: int = 500,
    verbose: bool = True,
) -> dict:
    """
    Fit ECCA across multiple lambda values (warm-start on B and H).
    Returns dict with 'U' and 'V' as lists (or arrays if len(lambdas)==1).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, p = X.shape[0], X.shape[1]
    q = Y.shape[1]

    if standardize:
        X = _standardize(X)
        Y = _standardize(Y)

    if Sxy is None:
        Sxy = _matmul(X.T, Y) / n
    if Sx is None:
        Sx = _matmul(X.T, X) / n
    if Sy is None:
        Sy = _matmul(Y.T, Y) / n

    B = np.zeros((p, q), dtype=float) if B0 is None else B0.astype(float, copy=True)

    # Eigendecompositions
    Lx, Ux = np.linalg.eigh(Sx)
    Ly, Uy = np.linalg.eigh(Sy)
    Lx = np.maximum(Lx, 0.0)
    Ly = np.maximum(Ly, 0.0)
    Sx12 = _matmul(_matmul(Ux, np.diag(np.sqrt(Lx))), Ux.T)
    Sy12 = _matmul(_matmul(Uy, np.diag(np.sqrt(Ly))), Uy.T)

    b = np.add.outer(Lx, Ly) + rho
    B1 = _matmul(_matmul(Ux.T, Sxy), Uy)

    H = np.zeros_like(B)
    U_list: List[np.ndarray] = []
    V_list: List[np.ndarray] = []

    for lam in lambdas:
        Z = B.copy()
        delta = math.inf
        it = 0
        while (delta > eps) and (it < maxiter):
            it += 1
            B_prev = B.copy()

            Btilde = B1 + rho * (_matmul(_matmul(Ux.T, (Z - H)), Uy))
            Btilde = Btilde / b
            B = _matmul(_matmul(Ux, Btilde), Uy.T)

            Z = B + H
            if groups is None:
                Z = soft_thresh(Z, lam / rho)
            else:
                for g_idx in groups:
                    g_idx = np.asarray(g_idx, dtype=int)
                    if g_idx.ndim != 2 or g_idx.shape[1] != 2:
                        raise ValueError("Each group must be an array of shape (k, 2) with (row, col) pairs.")
                    rows = g_idx[:, 0]
                    cols = g_idx[:, 1]
                    subset = Z[rows, cols]
                    lam_g = math.sqrt(len(rows)) * (lam / rho)
                    Z[rows, cols] = soft_thresh2(subset, lam_g)

            H = H + rho * (B - Z)

            sB0 = float(np.sum(B_prev ** 2))
            delta = float(np.sum((B - B_prev) ** 2) / sB0) if sB0 > 1e-20 else math.inf

            if verbose and it % 10 == 0:
                print(f"lambda={lam:.4g} iter: {it:d} delta: {delta:.3e}")

        if it >= maxiter and verbose:
            print(f"[lambda={lam:.4g}] ADMM did not converge!")
        elif verbose:
            print(f"[lambda={lam:.4g}] ADMM converged in {it} iterations")

        # map back
        B = Z
        C = _matmul(_matmul(Sx12, B), Sy12)
        U0, L0, V0 = _top_r_svd(C, r=r)

        if float(np.max(L0)) > 1e-8:
            inv_L0 = _safe_diag_inv(L0, tol=1e-8)
            U_k = _matmul(_matmul(_matmul(B, Sy12), V0), np.diag(inv_L0))
            V_k = _matmul(_matmul(_matmul(B.T, Sx12), U0), np.diag(inv_L0))
        else:
            U_k = np.full((p, r), np.nan, dtype=float)
            V_k = np.full((q, r), np.nan, dtype=float)

        U_list.append(U_k)
        V_list.append(V_k)

    if len(lambdas) == 1:
        return {"U": U_list[0], "V": V_list[0]}
    else:
        return {"U": U_list, "V": V_list}


def ecca_eval(
    X: np.ndarray,
    Y: np.ndarray,
    lambdas: Sequence[float],
    groups: Optional[List[np.ndarray]] = None,
    r: int = 2,
    standardize: bool = True,
    Sx: Optional[np.ndarray] = None,
    Sy: Optional[np.ndarray] = None,
    Sxy: Optional[np.ndarray] = None,
    rho: float = 1.0,
    B0: Optional[np.ndarray] = None,
    nfold: int = 5,
    eps: float = 1e-4,
    maxiter: int = 500,
    verbose: bool = True,
    scoring_method: str = "mse",
    cv_use_median: bool = False,
    set_seed_cv: Optional[int] = None,
) -> dict:
    """
    Cross-validation to select lambda.
    Returns: dict(scores=DataFrame, lambda.min, lambda.1se)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = X.shape[0]

    if n < 2 * nfold:
        if verbose:
            print(f"Warning: Sample size (n={n}) too small for {nfold}-fold CV. "
                  f"Skipping CV and using the first lambda.")
        return {
            "scores": None,
            "lambda.min": float(lambdas[0]),
            "lambda.1se": float(lambdas[0]),
        }

    if standardize:
        X = _standardize(X)
        Y = _standardize(Y)

    if Sxy is None:
        Sxy = _matmul(X.T, Y) / n
    if Sx is None:
        Sx = _matmul(X.T, X) / n
    if Sy is None:
        Sy = _matmul(Y.T, Y) / n

    # Folds
    if set_seed_cv is not None:
        rng = np.random.default_rng(set_seed_cv)
    else:
        rng = np.random.default_rng()

    idx = rng.permutation(n)
    folds = np.array_split(idx, nfold)

    # Accumulate scores (rows=lambdas, cols=successful folds)
    scores_by_fold: List[np.ndarray] = []

    for fold_id, fold in enumerate(folds, start=1):
        if verbose:
            print(f"\nFold {fold_id}/{nfold}")

        try:
            X_val = X[fold, :]
            Y_val = Y[fold, :]
            n_full = n
            n_train = n_full - len(fold)

            # Downdate trick for covariances
            Sx_train = (n_full * Sx - X_val.T @ X_val) / n_train
            Sy_train = (n_full * Sy - Y_val.T @ Y_val) / n_train
            Sxy_train = (n_full * Sxy - X_val.T @ Y_val) / n_train

            # Fit across lambdas on training covariances
            out = ecca_across_lambdas(
                X=np.delete(X, fold, axis=0),
                Y=np.delete(Y, fold, axis=0),
                lambdas=lambdas,
                groups=groups,
                r=r,
                Sx=Sx_train,
                Sy=Sy_train,
                Sxy=Sxy_train,
                standardize=False,
                rho=rho,
                B0=B0,
                eps=eps,
                maxiter=maxiter,
                verbose=verbose,
            )

            fold_scores = np.zeros(len(lambdas), dtype=float)
            for j, lam in enumerate(lambdas):
                U_j = out["U"][j] if isinstance(out["U"], list) else out["U"]
                V_j = out["V"][j] if isinstance(out["V"], list) else out["V"]

                if (U_j is None) or np.any(np.isnan(U_j)):
                    fold_scores[j] = np.inf
                else:
                    if scoring_method == "mse":
                        resid = (X_val @ U_j) - (Y_val @ V_j)
                        fold_scores[j] = float(np.mean(resid ** 2))
                    elif scoring_method == "trace":
                        fold_scores[j] = float(-np.trace(U_j.T @ (X_val.T @ Y_val) @ V_j))
                    else:
                        raise ValueError("Unknown scoring_method. Use 'mse' or 'trace'.")
            scores_by_fold.append(fold_scores)

        except Exception as e:
            if verbose:
                print(f"Fold {fold_id} failed with error: {e}")
            # Skip this fold

    if len(scores_by_fold) == 0:
        raise RuntimeError("All CV folds failed. Cannot select a lambda.")

    S = np.column_stack(scores_by_fold)  # shape: (len(lambdas), n_success)
    if cv_use_median:
        center = np.median(S, axis=1)
    else:
        center = np.mean(S, axis=1)
    se = np.std(S, axis=1, ddof=1) / math.sqrt(S.shape[1])

    scores_df = pd.DataFrame({"lambda": np.asarray(lambdas, dtype=float),
                              "mse": center,
                              "se": se})

    # lambda.min and lambda.1se
    idx_min = int(np.argmin(center))
    lambda_min = float(lambdas[idx_min])
    upper = center[idx_min] + se[idx_min]
    lambda_1se = float(np.max(np.asarray(lambdas)[center <= upper]))

    return {"scores": scores_df, "lambda.min": lambda_min, "lambda.1se": lambda_1se}


def ecca_cv(
    X: np.ndarray,
    Y: np.ndarray,
    lambdas: Sequence[float],
    groups: Optional[List[np.ndarray]] = None,
    r: int = 2,
    standardize: bool = False,
    rho: float = 1.0,
    B0: Optional[np.ndarray] = None,
    nfold: int = 5,
    select: str = "lambda.min",
    eps: float = 1e-4,
    maxiter: int = 500,
    verbose: bool = False,
    set_seed_cv: Optional[int] = None,
    scoring_method: str = "mse",
    cv_use_median: bool = False,
) -> dict:
    """
    Fit with cross-validated lambda and return U, V, cor, loss, lambda.opt, cv.scores.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = X.shape[0]

    if standardize:
        X = _standardize(X)
        Y = _standardize(Y)

    # Select lambda
    if len(lambdas) > 1:
        eval_out = ecca_eval(
            X, Y,
            lambdas=lambdas,
            groups=groups,
            r=r,
            standardize=False,  # already standardized above if requested
            Sx=None, Sy=None, Sxy=None,
            rho=rho,
            B0=B0,
            nfold=nfold,
            eps=eps,
            maxiter=maxiter,
            verbose=verbose,
            scoring_method=scoring_method,
            cv_use_median=cv_use_median,
            set_seed_cv=set_seed_cv,
        )
        lambda_opt = eval_out["lambda.1se"] if (select == "lambda.1se") else eval_out["lambda.min"]
        cv_scores = eval_out["scores"]
    else:
        lambda_opt = float(lambdas[0])
        cv_scores = None

    print(f"\nselected lambda: {lambda_opt}")

    fit = ecca(
        X, Y,
        lambda_=lambda_opt,
        groups=groups,
        r=r,
        standardize=False,  # already standardized above if requested
        rho=rho,
        B0=B0,
        eps=eps,
        maxiter=maxiter,
        verbose=verbose,
    )

    if np.any(np.isnan(fit.U)) or np.any(np.isnan(fit.V)):
        fit_cor = np.full(r, np.nan, dtype=float)
        fit_loss = math.inf
    else:
        Sxy = (X.T @ Y) / n
        fit_cor = np.diag(fit.U.T @ Sxy @ fit.V)
        resid = (X @ fit.U) - (Y @ fit.V)
        fit_loss = float(np.mean(np.sum(resid ** 2, axis=0)))

    return {
        "U": fit.U,
        "V": fit.V,
        "cor": fit_cor,
        "loss": fit_loss,
        "lambda.opt": lambda_opt,
        "cv.scores": cv_scores,
    }