# cca_rrr.py
from __future__ import annotations

from typing import Optional, Sequence, List, Tuple
import numpy as np

# Optional deps: used if available; otherwise fall back
try:
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception:
    _HAS_CVXPY = False

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
    _standardize, _safe_diag_inv,
     compute_sqrt_inv, compute_sqrt, _ledoit_wolf_cov,
    )




# --------------------------------------------------------------------
# ADMM-based group-sparse RRR solver (row-group penalty)
# --------------------------------------------------------------------

def solve_rrr_admm(
    X: np.ndarray,
    tilde_Y: np.ndarray,
    Sx: np.ndarray,
    lambda_: float,
    rho: float = 1.0,
    niter: int = 10_000,
    thresh: float = 1e-4,
    verbose: bool = False,
    thresh_0: float = 1e-6,
) -> np.ndarray:
    """
    Solve:  (1/n)||tilde_Y - X B||_F^2 + lambda * sum_i ||B_{i,:}||_2
    via ADMM using splitting on Z with row-wise group shrinkage.
    Matches the R logic and stopping rule.
    """
    X = np.asarray(X, dtype=float)
    tilde_Y = np.asarray(tilde_Y, dtype=float)
    Sx = np.asarray(Sx, dtype=float)

    n, p = X.shape
    q = tilde_Y.shape[1]

    # Precompute
    prod_xy = (X.T @ tilde_Y) / n  # p x q
    # Matrix to solve repeatedly: Sx + rho I (SPD if rho > 0)
    A = Sx + rho * np.eye(p, dtype=float)

    # ADMM variables
    B = np.zeros((p, q), dtype=float)
    Z = np.zeros_like(B)
    U = np.zeros_like(B)

    for it in range(1, niter + 1):
        B_old = B.copy()
        Z_old = Z.copy()

        # B-update: solve A * B = prod_xy + rho * (Z - U)
        RHS = prod_xy + rho * (Z - U)
        # Solve p x p system for each column of RHS
        # Using np.linalg.solve repeatedly is fine for clarity.
        # (One could Cholesky factorize A once for speed.)
        B = np.linalg.solve(A, RHS)

        # Z-update: proximal row-wise L2 shrinkage on B + U
        Z = B + U
        row_norms = np.sqrt(np.sum(Z * Z, axis=1))  # length p
        shrink = np.maximum(0.0, 1.0 - (lambda_ / rho) / (row_norms + 1e-15))  # avoid 0 division
        # expand to (p,1) for broadcasting across columns
        Z = (shrink[:, None]) * Z

        # U-update (dual)
        U = U + B - Z

        # Diagnostics / stopping
        primal = np.linalg.norm(Z - B)
        dual = np.linalg.norm(Z_old - Z)
        if verbose and (it % 50 == 0 or it == 1):
            print(f"ADMM iter {it} | primal {primal:.3e} | dual {dual:.3e}")

        # Stop when both scaled residuals small (matches R normalization by sqrt(p))
        if max(primal / np.sqrt(p), dual / np.sqrt(p)) < thresh:
            break

    B_opt = B
    B_opt[np.abs(B_opt) < thresh_0] = 0.0
    return B_opt


# --------------------------------------------------------------------
# CVXPY-based group-sparse solver (Python analogue of CVXR path)
# --------------------------------------------------------------------

def solve_rrr_cvxpy(
    X: np.ndarray,
    tilde_Y: np.ndarray,
    lambda_: float,
    thresh_0: float = 1e-6,
) -> np.ndarray:
    """
    Solve:  (1/n)||tilde_Y - X B||_F^2 + lambda * sum_i ||B_{i,:}||_2
    using CVXPY. If CVXPY is not installed, raises RuntimeError.
    """
    if not _HAS_CVXPY:
        raise RuntimeError("cvxpy is not installed; cannot use the 'CVX' solver.")

    X = np.asarray(X, dtype=float)
    tilde_Y = np.asarray(tilde_Y, dtype=float)

    n, p = X.shape
    q = tilde_Y.shape[1]

    B = cp.Variable((p, q))
    loss = (1.0 / n) * cp.sum_squares(tilde_Y - X @ B)
    # Sum of row-wise L2 norms (group-lasso over rows)
    reg = cp.sum(cp.norm(B, axis=1))
    problem = cp.Problem(cp.Minimize(loss + lambda_ * reg))
    problem.solve()  # you can pass a solver=... if you prefer (SCS, ECOS, etc.)

    B_opt = B.value
    if B_opt is None:
        raise RuntimeError("CVXPY did not return a solution (B.value is None).")
    B_opt[np.abs(B_opt) < thresh_0] = 0.0
    return B_opt


# --------------------------------------------------------------------
# Main: CCA via Reduced-Rank Regression (RRR)
# --------------------------------------------------------------------

def cca_rrr(
    X: np.ndarray,
    Y: np.ndarray,
    Sx: Optional[np.ndarray] = None,
    Sy: Optional[np.ndarray] = None,
    lambda_: float = 0.0,
    r: int = 2,
    highdim: bool = True,
    solver: str = "ADMM",         # "ADMM" | "CVX" (CVXPY) | "rrr" (not implemented)
    LW_Sy: bool = True,
    standardize: bool = True,
    rho: float = 1.0,
    niter: int = 10_000,
    thresh: float = 1e-4,
    thresh_0: float = 1e-6,
    verbose: bool = False,
) -> dict:
    """
    Canonical Correlation Analysis via Reduced Rank Regression (RRR).

    Returns dict with:
      U: (p x r), V: (q x r), loss: float, cor: length-r vector
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    n, p = X.shape
    q = Y.shape[1]

    if standardize:
        X = _standardize(X)
        Y = _standardize(Y)

    # Covariances (if not provided)
    if Sx is None:
        Sx = (X.T @ X) / n
    if Sy is None:
        if LW_Sy:
            Sy = _ledoit_wolf_cov(Y, verbose=verbose)
        else:
            Sy = (Y.T @ Y) / n

    sqrt_inv_Sy = compute_sqrt_inv(Sy)
    tilde_Y = Y @ sqrt_inv_Sy  # n x q
    Sx_tot = Sx

    if not highdim:
        if verbose:
            print("Not using highdim path (OLS + SVD).")
        # B_OLS = Sx^{-1} Sxy with Sxy = X^T tilde_Y / n
        Sxy = (X.T @ tilde_Y) / n
        B_OLS = np.linalg.solve(Sx_tot, Sxy)

        sqrt_Sx = compute_sqrt(Sx)
        sqrt_inv_Sx = compute_sqrt_inv(Sx)

        # SVD of sqrt(Sx) B_OLS
        M = sqrt_Sx @ B_OLS  # p x q
        Uu, s, VvT = np.linalg.svd(M, full_matrices=False)
        V_mat = sqrt_inv_Sy @ VvT.T[:, :r]
        U_mat = sqrt_inv_Sx @ Uu[:, :r]

    else:
        # High-dimensional with penalty (group lasso by row)
        if solver.upper() in ("CVX", "CVXR"):
            if verbose:
                print("Using CVXPY solver.")
            B_opt = solve_rrr_cvxpy(X, tilde_Y, lambda_=lambda_, thresh_0=thresh_0)
        elif solver.upper() == "ADMM":
            if verbose:
                print("Using ADMM solver.")
            B_opt = solve_rrr_admm(
                X, tilde_Y, Sx=Sx_tot, lambda_=lambda_, rho=rho,
                niter=niter, thresh=thresh, verbose=False, thresh_0=thresh_0
            )
        else:
            raise NotImplementedError("solver='rrr' (rrpack/gglasso path) is not implemented in Python.")

        # Threshold and active rows
        B_opt[np.abs(B_opt) < thresh_0] = 0.0
        row_active = np.sum(B_opt * B_opt, axis=1) > 0.0
        active_rows = np.where(row_active)[0]

        if active_rows.size > (r - 1):
            sqrt_Sx_sub = compute_sqrt(Sx[np.ix_(active_rows, active_rows)])
            # SVD of sqrt(Sx_sub) * B_opt[active_rows, :]
            M = sqrt_Sx_sub @ B_opt[active_rows, :]
            Uu, s, VvT = np.linalg.svd(M, full_matrices=False)
            V_mat = sqrt_inv_Sy @ VvT.T[:, :r]

            # inv_D: handle tiny singular values
            d = s[:r]
            inv_d = np.where(d < 1e-4, 0.0, 1.0 / d)
            U_mat = B_opt @ VvT.T[:, :r] @ np.diag(inv_d)
        else:
            # Not enough active rows; return zeros as in R
            U_mat = np.zeros((p, r), dtype=float)
            V_mat = np.zeros((q, r), dtype=float)

    # Loss and canonical covariances (use unbiased covariance like R's stats::cov)
    resid = (Y @ V_mat) - (X @ U_mat)
    loss = float(np.mean(resid ** 2))

    # Canonical covariances per component: cov(XU_i, YV_i)
    XU = X @ U_mat  # n x r
    YV = Y @ V_mat  # n x r
    cor = np.empty(r, dtype=float)
    for i in range(r):
        # np.cov defaults to unbiased (n-1 in denominator), like R's stats::cov
        C = np.cov(XU[:, i], YV[:, i], bias=False)
        cor[i] = C[0, 1]

    return {"U": U_mat, "V": V_mat, "loss": loss, "cor": cor}


# --------------------------------------------------------------------
# Cross-validated CCA via RRR
# --------------------------------------------------------------------

def cca_rrr_cv_folds(
    X: np.ndarray,
    Y: np.ndarray,
    Sx: Optional[np.ndarray],
    Sy: Optional[np.ndarray],
    kfolds: int = 5,
    lambda_: float = 0.01,
    r: int = 2,
    standardize: bool = False,
    solver: str = "ADMM",
    rho: float = 1.0,
    LW_Sy: bool = True,
    niter: int = 10_000,
    thresh_0: float = 1e-6,
    thresh: float = 1e-4,
    random_state: Optional[int] = None,
) -> float:
    """
    Single lambda evaluation via k-fold CV. Returns average MSE across folds.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = X.shape[0]

    # Build folds (like caret::createFolds)
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(n)
    folds = np.array_split(idx, kfolds)

    mses: List[float] = []
    for i, test_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(np.arange(n), test_idx, assume_unique=False)
        X_train, X_val = X[train_idx], X[test_idx]
        Y_train, Y_val = Y[train_idx], Y[test_idx]

        n_train = X_train.shape[0]

        # Downdate Sx if provided (as in your R code)
        if Sx is not None:
            # Sx (full) is (X^T X) / n; hence n * Sx = X^T X
            Sx_train = (n * Sx - (X_val.T @ X_val)) / n_train
        else:
            Sx_train = None

        try:
            fit = cca_rrr(
                X_train, Y_train,
                Sx=Sx_train, Sy=None,
                lambda_=lambda_, r=r, highdim=True, solver=solver,
                LW_Sy=LW_Sy, standardize=standardize, rho=rho, niter=niter,
                thresh=thresh, thresh_0=thresh_0, verbose=False
            )
            # MSE on validation set: mean((X_val U - Y_val V)^2)
            pred_resid = (X_val @ fit["U"]) - (Y_val @ fit["V"])
            mse = float(np.mean(pred_resid ** 2))
            mses.append(mse)
        except Exception as e:
            # Mimic R's tryCatch: return NA -> treated later
            mses.append(np.nan)

    mses = np.asarray(mses, dtype=float)
    if np.all(np.isnan(mses)):
        return 1e8  # as in R fallback
    return float(np.nanmean(mses))


def cca_rrr_cv(
    X: np.ndarray,
    Y: np.ndarray,
    r: int = 2,
    lambdas: Sequence[float] = tuple(np.logspace(-3, 1.5, 100)),
    kfolds: int = 14,
    solver: str = "ADMM",
    parallelize: bool = False,
    LW_Sy: bool = True,
    standardize: bool = True,
    rho: float = 1.0,
    thresh_0: float = 1e-6,
    niter: int = 10_000,
    thresh: float = 1e-4,
    verbose: bool = False,
    nb_cores: Optional[int] = None,
    random_state: Optional[int] = None,
) -> dict:
    """
    Cross-validated selection of lambda for CCA via RRR.

    Returns:
      {
        "U": p x r,
        "V": q x r,
        "lambda": optimal lambda,
        "rmse": vector of MSE values per lambda (same order),
        "cor": canonical covariances (on full fit with optimal lambda)
      }
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = X.shape[0]

    if standardize:
        Xs = _standardize(X)
        Ys = _standardize(Y)
    else:
        Xs, Ys = X, Y

    # Precompute Sx and Sy like in R
    Sx = (Xs.T @ Xs) / n
    if LW_Sy:
        Sy = _ledoit_wolf_cov(Ys, verbose=verbose)
    else:
        Sy = (Ys.T @ Ys) / n

    # CV evaluation function for a single lambda
    def _cv_fun(lam: float) -> float:
        return cca_rrr_cv_folds(
            Xs, Ys, Sx=Sx, Sy=None, kfolds=kfolds,
            lambda_=lam, r=r, standardize=False,
            solver=solver, rho=rho, LW_Sy=LW_Sy, niter=niter,
            thresh_0=thresh_0, thresh=thresh, random_state=random_state
        )

    if parallelize and solver.upper() in ("CVX", "CVXR", "ADMM") and _HAS_JOBLIB:
        n_jobs = nb_cores if (nb_cores is not None and nb_cores != 0) else -1
        rmses = Parallel(n_jobs=n_jobs)(
            delayed(_cv_fun)(lam) for lam in lambdas
        )
    else:
        if parallelize and not _HAS_JOBLIB and verbose:
            print("joblib is not installed; running CV in serial.")
        rmses = [ _cv_fun(lam) for lam in lambdas ]

    rmses = np.asarray(rmses, dtype=float)
    # Clean/guard like R code
    rmses = np.where(np.isnan(rmses) | (rmses == 0), 1e8, rmses)
    mask = rmses > 1e-5
    lambdas_arr = np.asarray(lambdas, dtype=float)
    lambdas_arr = lambdas_arr[mask]
    rmses = rmses[mask]

    if lambdas_arr.size == 0:
        # Fallback if everything filtered out
        lambdas_arr = np.asarray(lambdas, dtype=float)
        rmses = np.asarray([_cv_fun(lam) for lam in lambdas_arr], dtype=float)

    # Pick optimal lambda
    idx_opt = int(np.argmin(rmses))
    opt_lambda = float(lambdas_arr[idx_opt]) if lambdas_arr.size > 0 else float(lambdas[0])
    if np.isnan(opt_lambda):
        opt_lambda = 0.1
    if verbose:
        print(f"Optimal lambda: {opt_lambda}")

    # Final fit on full data
    final = cca_rrr(
        Xs, Ys, Sx=None, Sy=None, lambda_=opt_lambda, r=r,
        highdim=True, solver=solver, standardize=False, LW_Sy=LW_Sy,
        rho=rho, niter=niter, thresh=thresh, thresh_0=thresh_0, verbose=verbose
    )

    # Canonical covariances on full data (unbiased)
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
        "rmse": rmses,           # MSE per lambda (after filtering step above)
        "cor": cor
    }
