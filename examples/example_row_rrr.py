# Example: cca_rrr_cv (row-group RRR-CCA, ADMM)
import numpy as np
import matplotlib.pyplot as plt
from pyccar3 import cca_rrr_cv, cca_rrr

rng = np.random.default_rng(1)
n, p, q, r = 200, 50, 35, 2
X = rng.standard_normal((n, p))
Y = rng.standard_normal((n, q))

lambdas = np.logspace(-3, 0, 12)
res = cca_rrr_cv(X, Y, r=r, lambdas=lambdas, kfolds=5, solver="ADMM", standardize=True, verbose=False)
print("lambda* =", res["lambda"])

# Plot: RMSE vs lambda
plt.figure()
plt.semilogx(lambdas, res["rmse"])
plt.xlabel("lambda")
plt.ylabel("MSE")
plt.title("cca_rrr_cv: CV curve")
plt.tight_layout()
plt.show()

# Fit once (already computed inside, but show direct use)
fit = cca_rrr(X, Y, lambda_=res["lambda"], r=r, standardize=True, solver="ADMM")
XU = X @ fit["U"]
YV = Y @ fit["V"]

# Scatter of first canonical variates
plt.figure()
plt.scatter(XU[:,0], YV[:,0], s=10)
plt.xlabel("XU[:,0]")
plt.ylabel("YV[:,0]")
plt.title("cca_rrr: first canonical variates")
plt.tight_layout()
plt.show()
