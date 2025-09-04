# Example: ecca_cv and ecca
import numpy as np
import matplotlib.pyplot as plt
from pyccar3 import ecca_cv, ecca

rng = np.random.default_rng(0)
n, p, q, r = 200, 40, 30, 2
X = rng.standard_normal((n, p))
Y = rng.standard_normal((n, q))
lambdas = np.linspace(0.0, 0.2, 10)

cv = ecca_cv(X, Y, lambdas=lambdas, r=r, standardize=True, verbose=False)
print("Selected lambda:", cv["lambda.opt"])

# Plot: CV curve (mse ± se) vs lambda
scores = cv["cv.scores"]
if scores is not None:
    lam = scores["lambda"].to_numpy()
    mse = scores["mse"].to_numpy()
    se  = scores["se"].to_numpy()
    plt.figure()
    plt.errorbar(lam, mse, yerr=se)
    plt.xlabel("lambda")
    plt.ylabel("CV MSE")
    plt.title("ecca_cv: cross-validation curve")
    plt.tight_layout()
    plt.show()

# Fit at selected lambda (already done inside ecca_cv, but show ecca usage)
fit = ecca(X, Y, lambda_=float(cv["lambda.opt"]), r=r, standardize=True, verbose=False)
XU = X @ fit.U
YV = Y @ fit.V

# Plot: first canonical variates scatter
plt.figure()
plt.scatter(XU[:,0], YV[:,0], s=10)
plt.xlabel("XU[:,0]")
plt.ylabel("YV[:,0]")
plt.title("First canonical variates (ecca)")
plt.tight_layout()
plt.show()

# Plot: canonical covariances
plt.figure()
plt.bar(range(r), fit.cor)
plt.xlabel("Component")
plt.ylabel("Covariance")
plt.title("Canonical covariances (ecca)")
plt.tight_layout()
plt.show()
