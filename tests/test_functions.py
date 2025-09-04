import numpy as np
import pytest

from pyccar3 import (
    ecca, ecca_across_lambdas, ecca_eval, ecca_cv,
    cca_rrr, cca_rrr_cv, cca_rrr_cv_folds,
    cca_graph_rrr, cca_graph_rrr_cv, cca_graph_rrr_cv_folds,
    cca_group_rrr, cca_group_rrr_cv, cca_group_rrr_cv_folds,
)

rng = np.random.default_rng(0)

def small_data(n=60, p=15, q=12, r=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    Y = rng.standard_normal((n, q))
    return X, Y, r

def test_ecca_shapes_and_cv():
    X, Y, r = small_data()
    out = ecca(X, Y, lambda_=0.0, r=r, standardize=True, verbose=False)
    assert out.U.shape == (X.shape[1], r)
    assert out.V.shape == (Y.shape[1], r)
    # across lambdas
    lams = [0.0, 0.05]
    out2 = ecca_across_lambdas(X, Y, lambdas=lams, r=r, standardize=True, verbose=False)
    assert isinstance(out2["U"], list) and len(out2["U"]) == len(lams)
    # eval + cv
    eval_out = ecca_eval(X, Y, lambdas=lams, r=r, standardize=True, verbose=False)
    assert "lambda.min" in eval_out and "lambda.1se" in eval_out
    cv_out = ecca_cv(X, Y, lambdas=lams, r=r, standardize=True, verbose=False)
    assert cv_out["U"].shape == (X.shape[1], r)
    assert cv_out["V"].shape == (Y.shape[1], r)

def test_cca_rrr_and_cv():
    X, Y, r = small_data(seed=1)
    lams = [1e-3, 1e-2]
    fit = cca_rrr(X, Y, lambda_=lams[0], r=r, standardize=True, solver="ADMM")
    assert fit["U"].shape == (X.shape[1], r)
    assert fit["V"].shape == (Y.shape[1], r)
    cv = cca_rrr_cv(X, Y, r=r, lambdas=lams, kfolds=3, solver="ADMM", standardize=True)
    assert "lambda" in cv and cv["U"].shape == (X.shape[1], r)

def test_cca_graph_rrr_and_cv():
    n, p, q, g, r = 80, 20, 10, 30, 2
    rng = np.random.default_rng(2)
    X = rng.standard_normal((n, p))
    Y = rng.standard_normal((n, q))
    Gamma = rng.standard_normal((g, p))
    fit = cca_graph_rrr(X, Y, Gamma, lambda_=1e-2, r=r, standardize=True, rho=5.0, niter=500)
    assert fit["U"].shape == (p, r)
    assert fit["V"].shape == (q, r)
    cv = cca_graph_rrr_cv(X, Y, Gamma, r=r, lambdas=[1e-3, 1e-2], kfolds=3, standardize=True, niter=500)
    assert "lambda" in cv and cv["U"].shape == (p, r)

def test_cca_group_rrr_and_cv():
    n, p, q, r = 90, 24, 16, 2
    rng = np.random.default_rng(3)
    X = rng.standard_normal((n, p))
    Y = rng.standard_normal((n, q))
    # create 4 contiguous groups over predictor rows
    groups = [np.arange(0,6), np.arange(6,12), np.arange(12,18), np.arange(18,24)]
    fit = cca_group_rrr(X, Y, groups, lambda_=1e-2, r=r, standardize=True, solver="ADMM", niter=500)
    assert fit["U"].shape == (p, r)
    assert fit["V"].shape == (q, r)
    cv = cca_group_rrr_cv(X, Y, groups, r=r, lambdas=[1e-3, 1e-2], kfolds=3, standardize=True, niter=500)
    assert "lambda" in cv and cv["U"].shape == (p, r)

def test_cv_fold_helpers_run():
    # row rrr folds
    X, Y, r = small_data(seed=4)
    val = cca_rrr_cv_folds(X, Y, Sx=None, Sy=None, kfolds=3, lambda_=1e-2, r=r, standardize=True, solver="ADMM", niter=300)
    assert isinstance(val, float)

    # graph rrr folds
    n, p, q, g, r2 = 60, 16, 10, 24, 2
    rng = np.random.default_rng(5)
    X = rng.standard_normal((n, p))
    Y = rng.standard_normal((n, q))
    Gamma = rng.standard_normal((g, p))
    val2 = cca_graph_rrr_cv_folds(X, Y, Gamma, Sx=None, Sy=None, kfolds=3, lambda_=1e-2, r=r2, standardize=True, niter=300)
    assert isinstance(val2, float)

    # grouped rows folds
    n, p, q, r3 = 60, 20, 12, 2
    rng = np.random.default_rng(6)
    X = rng.standard_normal((n, p))
    Y = rng.standard_normal((n, q))
    groups = [np.arange(0,5), np.arange(5,10), np.arange(10,15), np.arange(15,20)]
    val3 = cca_group_rrr_cv_folds(X, Y, groups, Sx=None, Sy=None, kfolds=3, lambda_=1e-2, r=r3, standardize=True, niter=300)
    assert isinstance(val3, float)
