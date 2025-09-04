# Example: cca_group_rrr_cv (explicit groups of predictor rows)
import numpy as np
import matplotlib.pyplot as plt
from pyccar3 import cca_group_rrr_cv

rng = np.random.default_rng(3)
n, p, q, r = 180, 60, 40, 2
X = rng.standard_normal((n, p))
Y = rng.standard_normal((n, q))

# Build groups over predictor rows (0-based indices)
groups = [
    np.arange(0, 10),
    np.arange(10, 25),
    np.arange(25, 40),
    np.arange(40, 60),
]

lambdas = np.logspace(-3, 1, 12)
res = cca_group_rrr_cv(
    X, Y, groups,
    r=r, lambdas=lambdas, kfolds=5,
    standardize=True, solver="ADMM", rho=1.0, niter=2000, verbose=False
)
print("lambda* =", res["lambda"])

# Plot: RMSE vs lambda
plt.figure()
plt.semilogx(lambdas, res["rmse"])
plt.xlabel("lambda")
plt.ylabel("MSE")
plt.title("cca_group_rrr_cv: CV curve")
plt.tight_layout()
plt.show()
