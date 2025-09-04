# cca_graph_rrr.py
from __future__ import annotations

from typing import Optional, Sequence, List, Tuple
import numpy as np

# Optional dependencies
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


from ._helper import (
    _standardize,
     compute_sqrt_inv, compute_sqrt, _ledoit_wolf_cov
    )


# -------------------------------------------------------
# Core: Graph-regularized RRR for CCA (ADMM solver)
# -------------------------------------------------------

def cca_graph_rrr(
    X: np.ndarray,
    Y: np.ndarray,
    Gamma: np.ndarray,
    Sx: Optional[np.ndarray] = None,
    Sy: Optional[np.ndarray] = None,
    Sxy: Optional[np.ndarray] = None,   # not used; kept for API parity
    lambda_: float = 0.0,
    r: int = 2,
    standardize: bool = False,
    LW_Sy: bool = True,
    rho: float = 10.0,
    niter: int = 10_000,
    thresh: float = 1e-4,
    thresh_0: float = 1e-6,
    verbose: bool = False,
    Gamma_dagger: Optional[np.ndarray] = None,
) -> dict:
    """
    Graph-regularized Reduced-Rank Regression for CCA (ADMM solver).
    Returns: {"U","V","loss","cor"}.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)

    n, p = X.shape
    q = Y.shape[1]
    if Gamma.shape[1] != p:
        raise ValueError("Gamma must have shape (g, p) with p == X.shape[1].")
    if q > n:
        raise ValueError(
            "The X/Y swapping heuristic (for q > n) is incompatible with the graph "
            "constraint Gamma. Swap X and Y in your arguments."
        )

    if standardize:
        X = _standardize(X)
        Y = _standardize(Y)

    # Covariances
    if Sx is None:
        Sx = (X.T @ X) / n
    if Sy is None:
        Sy = _ledoit_wolf_cov(Y, verbose=verbose) if LW_Sy else (Y.T @ Y) / n

    # Whitening of Y
    sqrt_inv_Sy = compute_sqrt_inv(Sy)
    tilde_Y = Y @ sqrt_inv_Sy  # n x q

    # Graph operators
    if Gamma_dagger is None:
        Gamma_dagger = np.linalg.pinv(Gamma)   # p x g (Moore-Penrose)
    Pi = np.eye(p, dtype=float) - Gamma_dagger @ Gamma  # p x p
    XPi = X @ Pi
    XG = X @ Gamma_dagger                         # n x g

    # Remove projection on Pi: Projection = (XPi)^+ * tilde_Y
    Projection = np.linalg.pinv(XPi) @ tilde_Y    # p x q
    new_Ytilde = tilde_Y - XPi @ Projection       # n x q

    # ADMM in Gamma_dagger-space
    new_p = XG.shape[1]                           # g
    prod_xy = (XG.T @ new_Ytilde) / n             # g x q
    A = (XG.T @ XG) / n + rho * np.eye(new_p)     # g x g (SPD)

    B = np.zeros((new_p, q), dtype=float)
    Z = np.zeros_like(B)
    U = np.zeros_like(B)

    for it in range(1, niter + 1):
        B_prev = B.copy()
        Z_prev = Z.copy()

        # B update
        RHS = prod_xy + rho * (Z - U)
        B = np.linalg.solve(A, RHS)

        # Z update: row-wise (group) l2 shrinkage in Gamma_dagger-space
        Z = B + U
        norms = np.linalg.norm(Z, axis=1)              # (g,)
        shrink = np.maximum(0.0, 1.0 - (lambda_ / (rho * (norms + 1e-15))))
        Z = (shrink[:, None]) * Z

        # U update
        U = U + B - Z

        # Convergence check (scaled by sqrt(new_p), as in R)
        primal = np.linalg.norm(Z - B, ord="fro")
        dual = np.linalg.norm(Z_prev - Z, ord="fro")
        if verbose and (it % 50 == 0 or it == 1):
            print(f"ADMM iter {it} | primal {primal:.3e} | dual {dual:.3e}")
        if max(primal / np.sqrt(new_p), dual / np.sqrt(new_p)) < thresh:
            break

    # Reconstruct p x q coefficient matrix in original space
    B_opt = Gamma_dagger @ B + Pi @ Projection
    B_opt[np.abs(B_opt) < thresh_0] = 0.0

    # Final CCA step: SVD on sqrt(Sx) B_opt
    sqrt_Sx = compute_sqrt(Sx)
    sqrt_inv_Sx = compute_sqrt_inv(Sx)
    M = sqrt_Sx @ B_opt
    Uu, s, Vt = np.linalg.svd(M, full_matrices=False)  # M = Uu diag(s) Vt

    V = sqrt_inv_Sy @ Vt.T[:, :r]
    d = s[:r]
    inv_d = np.where(d > 1e-4, 1.0 / d, 0.0)
    U_mat = B_opt @ Vt.T[:, :r] @ np.diag(inv_d)

    # Metrics
    resid = (Y @ V) - (X @ U_mat)
    loss = float(np.mean(resid ** 2))
    XU = X @ U_mat
    YV = Y @ V
    cor = np.empty(r, dtype=float)
    for i in range(r):
        C = np.cov(XU[:, i], YV[:, i], bias=False)
        cor[i] = C[0, 1]

    return {"U": U_mat, "V": V, "loss": loss, "cor": cor}


# -------------------------------------------------------
# CV: single-lambda k-fold evaluation (with downdate)
# -------------------------------------------------------

def cca_graph_rrr_cv_folds(
    X: np.ndarray,
    Y: np.ndarray,
    Gamma: np.ndarray,
    Sx: Optional[np.ndarray] = None,
    Sy: Optional[np.ndarray] = None,
    kfolds: int = 5,
    lambda_: float = 0.01,
    r: int = 2,
    standardize: bool = False,
    LW_Sy: bool = False,
    rho: float = 10.0,
    niter: int = 10_000,
    thresh: float = 1e-4,
    thresh_0: float = 1e-6,
    Gamma_dagger: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
) -> float:
    """
    Evaluate one lambda via k-fold CV. Returns average MSE across folds.
    If Sx/Sy are provided, uses the downdate trick; otherwise recomputes from training data.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = X.shape[0]

    rng = np.random.default_rng(random_state)
    idx = rng.permutation(n)
    folds = np.array_split(idx, kfolds)

    mses: List[float] = []
    for test_idx in folds:
        train_idx = np.setdiff1d(np.arange(n), test_idx, assume_unique=False)
        X_train, X_val = X[train_idx], X[test_idx]
        Y_train, Y_val = Y[train_idx], Y[test_idx]

        n_full = n
        n_train = X_train.shape[0]

        # Downdate if full covariances given; else compute from training split
        if Sx is not None:
            Sx_train = (n_full * Sx - (X_val.T @ X_val)) / n_train
        else:
            Sx_train = (X_train.T @ X_train) / n_train

        if Sy is not None:
            Sy_train = (n_full * Sy - (Y_val.T @ Y_val)) / n_train
        else:
            Sy_train = (Y_train.T @ Y_train) / n_train

        try:
            fit = cca_graph_rrr(
                X_train, Y_train, Gamma,
                Sx=Sx_train, Sy=Sy_train,
                lambda_=lambda_, r=r,
                standardize=standardize, LW_Sy=LW_Sy, rho=rho,
                niter=niter, thresh=thresh, thresh_0=thresh_0,
                Gamma_dagger=Gamma_dagger, verbose=False
            )
            resid = (X_val @ fit["U"]) - (Y_val @ fit["V"])
            mses.append(float(np.mean(resid ** 2)))
        except Exception:
            mses.append(np.nan)

    mses = np.asarray(mses, dtype=float)
    if np.all(np.isnan(mses)):
        return 1e8
    return float(np.nanmean(mses))


# -------------------------------------------------------
# CV over a grid of lambdas + final refit
# -------------------------------------------------------

def cca_graph_rrr_cv(
    X: np.ndarray,
    Y: np.ndarray,
    Gamma: np.ndarray,
    r: int = 2,
    lambdas: Sequence[float] = tuple(np.logspace(-3, 1.5, 10)),
    kfolds: int = 5,
    parallelize: bool = False,
    standardize: bool = True,
    LW_Sy: bool = False,
    rho: float = 10.0,
    niter: int = 10_000,
    thresh: float = 1e-4,
    thresh_0: float = 1e-6,
    verbose: bool = False,
    Gamma_dagger: Optional[np.ndarray] = None,
    nb_cores: Optional[int] = None,
    random_state: Optional[int] = None,
) -> dict:
    """
    Cross-validated graph-regularized CCA via RRR.
    Returns: {"U","V","lambda","rmse","cor"}.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.shape[0] < min(X.shape[1], Y.shape[1]) and verbose:
        print("Warning: Both X and Y are high dimensional; method may fail.")

    # standardize for CV (like your R default)
    Xs = _standardize(X) if standardize else X
    Ys = _standardize(Y) if standardize else Y

    n = Xs.shape[0]
    Sx_full = (Xs.T @ Xs) / n
    Sy_full = (Ys.T @ Ys) / n

    def _cv_one(lam: float) -> float:
        return cca_graph_rrr_cv_folds(
            Xs, Ys, Gamma,
            Sx=Sx_full, Sy=Sy_full,
            kfolds=kfolds, lambda_=lam, r=r,
            standardize=False, LW_Sy=LW_Sy, rho=rho,
            niter=niter, thresh=thresh, thresh_0=thresh_0,
            Gamma_dagger=Gamma_dagger, random_state=random_state
        )

    if parallelize and _HAS_JOBLIB:
        n_jobs = nb_cores if (nb_cores is not None and nb_cores != 0) else -1
        rmses = Parallel(n_jobs=n_jobs)(delayed(_cv_one)(lam) for lam in lambdas)
    else:
        if parallelize and not _HAS_JOBLIB and verbose:
            print("joblib not found; running CV serially.")
        rmses = [_cv_one(lam) for lam in lambdas]

    rmses = np.asarray(rmses, dtype=float)
    rmses = np.where(np.isnan(rmses) | (rmses == 0.0), 1e8, rmses)
    mask = rmses > 1e-5
    lambdas_arr = np.asarray(lambdas, dtype=float)[mask]
    rmses = rmses[mask]

    if lambdas_arr.size == 0:
        lambdas_arr = np.asarray(lambdas, dtype=float)
        rmses = np.asarray([_cv_one(l) for l in lambdas_arr], dtype=float)

    # Select optimal lambda and refit
    idx = int(np.argmin(rmses))
    opt_lambda = float(lambdas_arr[idx]) if lambdas_arr.size > 0 else float(lambdas[0])
    if np.isnan(opt_lambda):
        opt_lambda = 0.1
    if verbose:
        print(f"Optimal lambda: {opt_lambda}")

    final = cca_graph_rrr(
        Xs, Ys, Gamma,
        Sx=None, Sy=None, lambda_=opt_lambda, r=r,
        standardize=False, LW_Sy=LW_Sy, rho=rho,
        niter=niter, thresh=thresh, thresh_0=thresh_0,
        Gamma_dagger=Gamma_dagger, verbose=verbose
    )

    # Canonical covariances on the final fit
    XU = Xs @ final["U"]
    YV = Ys @ final["V"]
    cor = np.empty(r, dtype=float)
    for i in range(r):
        C = np.cov(XU[:, i], YV[:, i], bias=False)
        cor[i] = C[0, 1]

    return {
        "U": final["U"],
        "V": final["V"],
        "lambda": opt_lambda,
        "rmse": rmses,
        "cor": cor,
    }
