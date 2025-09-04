import numpy as np
from pyccar3 import ecca_cv

def test_smoke_shapes():
    rng = np.random.default_rng(0)
    n, p, q, r = 40, 10, 8, 2
    X = rng.standard_normal((n, p))
    Y = rng.standard_normal((n, q))
    out = ecca_cv(X, Y, lambdas=[0.0], r=r, standardize=True, nfold=5, verbose=False)
    U, V = out["U"], out["V"]
    assert U.shape == (p, r)
    assert V.shape == (q, r)
