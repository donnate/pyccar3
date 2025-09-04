# cca_group_rrr.py
from __future__ import annotations

from typing import Optional, Sequence, List, Tuple
import numpy as np

# Optional deps
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

from ._helper import (
    _standardize,
     compute_sqrt_inv, compute_sqrt, _ledoit_wolf_cov
    )


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B


def _standardize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (X - mu) / sd




# --------------------------------------------------------------------
# ADMM and CVX solvers for grouped rows (groups of predictors)
# --------------------------------------------------------------------

def solve_group_rrr_admm(
    X: np.ndarray,
    tilde_Y: np.ndarray,
    Sx: np.ndarray,
    groups: List[np.ndarray],
    lambda_: float,
    rho: float = 1.0,
    niter: int = 10_000,
    thresh: float = 1e-5,
    thresh_0: float = 1e-6,
    verbose: bool = False,
) -> np.ndarray:
    """
    Solve (1/n)||tilde_Y - X B||_F^2 + lambda * sum_g ||B[groups[g], :]||_F
    with ADMM splitting (Z, U). Groups are lists of row-index arrays (0-based).
    """
    X = np.asarray(X, dtype=float)
    tilde_Y = np.asarray(tilde_Y, dtype=float)
    Sx = np.asarray(Sx, dtype=float)
    groups = [np.asarray(g, dtype=int) for g in groups]

    n, p = X.shape
    q = tilde_Y.shape[1]

    prod_xy = (X.T @ tilde_Y) / n            # p x q
    A = Sx + rho * np.eye(p, dtype=float)     # p x p (SPD)

    # ADMM state
    B = np.zeros((p, q), dtype=float)
    Z = np.zeros_like(B)
    U = np.zeros_like(B)

    for it in range(1, niter + 1):
        B_prev = B.copy()
        Z_prev = Z.copy()

        # B update
        RHS = prod_xy + rho * (Z - U)
        B = np.linalg.solve(A, RHS)   # could pre-factor A with cholesky for speed

        # Z update: grouped Frobenius shrinkage across specified row groups
        Z = B + U
        for g in groups:
            if g.size == 0:
                continue
            # Frobenius norm over the submatrix rows in group g
            block = Z[g, :]              # |g| x q
            norm_g = np.linalg.norm(block, ord="fro")
            tau = (lambda_ * np.sqrt(len(g))) / rho
            if norm_g <= tau:
                Z[g, :] = 0.0
            else:
                Z[g, :] = (1.0 - tau / (norm_g + 1e-15)) * block

        # U update
        U = U + B - Z

        # Convergence check (unscaled, as in your R code)
        primal = np.linalg.norm(Z - B, ord="fro")
        dual = np.linalg.norm(Z_prev - Z, ord="fro")
        if verbose and (it % 50 == 0 or it == 1):
            print(f"ADMM iter {it} | primal {primal:.3e} | dual {dual:.3e}")
        if max(primal, dual) < thresh:
            break

    B_opt = B
    B_opt[np.abs(B_opt) < thresh_0] = 0.0
    return B_opt


def solve_group_rrr_cvxpy(
    X: np.ndarray,
    tilde_Y: np.ndarray,
    groups: List[np.ndarray],
    lambda_: float,
    thresh_0: float = 1e-6,
) -> np.ndarray:
    """CVXPY analogue of the CVXR solver in R."""
    if not _HAS_CVXPY:
        raise RuntimeError("cvxpy is not installed; cannot use the 'CVX' solver.")

    X = np.asarray(X, dtype=float)
    tilde_Y = np.asarray(tilde_Y, dtype=float)
    n, p = X.shape
    q = tilde_Y.shape[1]
    groups = [np.asarray(g, dtype=int) for g in groups]

    B = cp.Variable((p, q))
    loss = (1.0 / n) * cp.sum_squares(tilde_Y - X @ B)

    # Sum of Frobenius norms over row groups
    group_norms = [cp.norm(B[g, :], "fro") for g in groups if len(g) > 0]
    reg = cp.sum(group_norms) if len(group_norms) else 0

    prob = cp.Problem(cp.Minimize(loss + lambda_ * reg))
    prob.solve()  # optionally pass solver=cp.SCS, etc.

    B_opt = B.value
    if B_opt is None:
        raise RuntimeError("CVXPY did not return a solution (B.value is None).")
    B_opt[np.abs(B_opt) < thresh_0] = 0.0
    return B_opt


# --------------------------------------------------------------------
# Main: Group-sparse CCA via RRR
# --------------------------------------------------------------------

def cca_group_rrr(
    X: np.ndarray,
    Y: np.ndarray,
    groups: List[np.ndarray],
    Sx: Optional[np.ndarray] = None,
    Sy: Optional[np.ndarray] = None,
    Sxy: Optional[np.ndarray] = None,   # not used
    lambda_: float = 0.0,
    r: int = 2,
    standardize: bool = False,
    LW_Sy: bool = True,
    solver: str = "ADMM",               # "ADMM" or "CVX"
    rho: float = 1.0,
    niter: int = 10_000,
    thresh: float = 1e-4,
    thresh_0: float = 1e-6,
    verbose: bool = False,
) -> dict:
    """Group-sparse CCA via reduced-rank regression."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    groups = [np.asarray(g, dtype=int) for g in groups]

    n, p = X.shape
    q = Y.shape[1]

    if standardize:
        X = _standardize(X)
        Y = _standardize(Y)

    if n < min(p, q) and verbose:
        print("Warning: Both X and Y are high-dimensional; method may be unstable.")

    if Sx is None:
        Sx = (X.T @ X) / n
    if Sy is None:
        Sy = _ledoit_wolf_cov(Y, verbose=verbose) if LW_Sy else (Y.T @ Y) / n

    # Whitening of Y
    sqrt_inv_Sy = compute_sqrt_inv(Sy)
    tilde_Y = Y @ sqrt_inv_Sy

    # Solve group-penalized regression for B
    if solver.upper() in ("ADMM",):
        B_opt = solve_group_rrr_admm(
            X, tilde_Y, Sx, groups=groups,
            lambda_=lambda_, rho=rho, niter=niter,
            thresh=thresh, thresh_0=thresh_0, verbose=verbose
        )
    elif solver.upper() in ("CVX"):
        B_opt = solve_group_rrr_cvxpy(
            X, tilde_Y, groups=groups, lambda_=lambda_, thresh_0=thresh_0
        )
    else:
        raise ValueError("Unsupported solver: choose either 'ADMM' or 'CVXR'.")

    # Final CCA step via SVD on sqrt(Sx) B
    sqrt_Sx = compute_sqrt(Sx)
    sqrt_inv_Sx = compute_sqrt_inv(Sx)

    M = sqrt_Sx @ B_opt
    Uu, s, Vt = np.linalg.svd(M, full_matrices=False)  # M = Uu diag(s) Vt
    V = sqrt_inv_Sy @ Vt.T[:, :r]

    d = s[:r]
    inv_d = np.where(d > 1e-4, 1.0 / d, 0.0)
    U_mat = B_opt @ Vt.T[:, :r] @ np.diag(inv_d)

    resid = (Y @ V) - (X @ U_mat)
    loss = float(np.mean(resid ** 2))

    # Canonical covariances cov(XU_i, YV_i)
    XU = X @ U_mat
    YV = Y @ V
    cor = np.empty(r, dtype=float)
    for i in range(r):
        C = np.cov(XU[:, i], YV[:, i], bias=False)
        cor[i] = C[0, 1]

    return {"U": U_mat, "V": V, "loss": loss, "cor": cor}


# --------------------------------------------------------------------
# Cross-validated loss for group-penalized CCA (single lambda)
# --------------------------------------------------------------------

def cca_group_rrr_cv_folds(
    X: np.ndarray,
    Y: np.ndarray,
    groups: List[np.ndarray],
    Sx: Optional[np.ndarray] = None,
    Sy: Optional[np.ndarray] = None,
    kfolds: int = 5,
    lambda_: float = 0.01,
    r: int = 2,
    standardize: bool = False,
    LW_Sy: bool = False,
    solver: str = "ADMM",
    rho: float = 1.0,
    niter: int = 10_000,
    thresh: float = 1e-4,
    thresh_0: float = 1e-6,
    verbose: bool = False,
    random_state: Optional[int] = None,
) -> float:
    """Average MSE across folds for a given lambda."""
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

        # Downdate Sx if provided
        if Sx is not None:
            Sx_train = (n * Sx - (X_val.T @ X_val)) / n_train
        else:
            Sx_train = (X_train.T @ X_train) / n_train

        try:
            fit = cca_group_rrr(
                X_train, Y_train, groups,
                Sx=Sx_train, Sy=None,
                lambda_=lambda_, r=r,
                standardize=standardize, LW_Sy=LW_Sy, solver=solver,
                rho=rho, niter=niter, thresh=thresh, thresh_0=thresh_0,
                verbose=False
            )
            resid = (X_val @ fit["U"]) - (Y_val @ fit["V"])
            mses.append(float(np.mean(resid ** 2)))
        except Exception:
            mses.append(np.nan)

    mses = np.asarray(mses, dtype=float)
    if np.all(np.isnan(mses)):
        return 1e8
    return float(np.nanmean(mses))


# --------------------------------------------------------------------
# Cross-validated selection of lambda and final fit
# --------------------------------------------------------------------

def cca_group_rrr_cv(
    X: np.ndarray,
    Y: np.ndarray,
    groups: List[np.ndarray],
    r: int = 2,
    lambdas: Sequence[float] = tuple(np.logspace(-3, 1.5, 10)),
    kfolds: int = 5,
    parallelize: bool = False,
    standardize: bool = False,
    LW_Sy: bool = True,
    solver: str = "ADMM",
    rho: float = 1.0,
    thresh_0: float = 1e-6,
    niter: int = 10_000,
    thresh: float = 1e-4,
    verbose: bool = False,
    nb_cores: Optional[int] = None,
    random_state: Optional[int] = None,
) -> dict:
    """
    Cross-validated group-sparse CCA via RRR.

    Returns:
      {
        "U": p x r,
        "V": q x r,
        "lambda": optimal lambda,
        "rmse": vector of MSEs per lambda (filtered),
        "cor": canonical covariances of final fit
      }
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.shape[0] < min(X.shape[1], Y.shape[1]) and verbose:
        print("Warning: Both X and Y are high-dimensional; method may be unstable.")

    if standardize:
        Xs = _standardize(X)
        Ys = _standardize(Y)
    else:
        Xs, Ys = X, Y

    n = Xs.shape[0]
    Sx_full = (Xs.T @ Xs) / n

    def _cv_one(lam: float) -> float:
        return cca_group_rrr_cv_folds(
            Xs, Ys, groups, Sx=Sx_full, Sy=None, kfolds=kfolds,
            lambda_=lam, r=r, standardize=False, LW_Sy=LW_Sy, solver=solver,
            rho=rho, niter=niter, thresh=thresh, thresh_0=thresh_0,
            verbose=False, random_state=random_state
        )

    if parallelize and _HAS_JOBLIB:
        n_jobs = nb_cores if (nb_cores is not None and nb_cores != 0) else -1
        rmses = Parallel(n_jobs=n_jobs)(delayed(_cv_one)(lam) for lam in lambdas)
    else:
        if parallelize and not _HAS_JOBLIB and verbose:
            print("joblib not found; running CV serially.")
        rmses = [ _cv_one(lam) for lam in lambdas ]

    rmses = np.asarray(rmses, dtype=float)
    rmses = np.where(np.isnan(rmses) | (rmses == 0.0), 1e8, rmses)
    mask = rmses > 1e-5
    lambdas_arr = np.asarray(lambdas, dtype=float)[mask]
    rmses = rmses[mask]

    if lambdas_arr.size == 0:
        lambdas_arr = np.asarray(lambdas, dtype=float)
        rmses = np.asarray([_cv_one(lam) for lam in lambdas_arr], dtype=float)

    idx_opt = int(np.argmin(rmses))
    opt_lambda = float(lambdas_arr[idx_opt]) if lambdas_arr.size > 0 else float(lambdas[0])
    if np.isnan(opt_lambda):
        opt_lambda = 0.1
    if verbose:
        print(f"Optimal lambda: {opt_lambda}")

    final = cca_group_rrr(
        Xs, Ys, groups, Sx=Sx_full, Sy=None, lambda_=opt_lambda,
        r=r, standardize=False, LW_Sy=LW_Sy, solver=solver,
        rho=rho, niter=niter, thresh=thresh, thresh_0=thresh_0, verbose=verbose
    )

    # Canonical covariances
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
