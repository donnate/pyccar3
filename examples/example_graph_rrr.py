# Example: cca_graph_rrr_cv (graph-regularized RRR-CCA)
import numpy as np
import matplotlib.pyplot as plt
from pyccar3 import cca_graph_rrr_cv

rng = np.random.default_rng(2)
n, p, q, g, r = 150, 40, 25, 60, 2
X = rng.standard_normal((n, p))
Y = rng.standard_normal((n, q))

# Example Gamma (g x p). Replace with your graph incidence/constraint matrix.
Gamma = rng.standard_normal((g, p))

lambdas = np.logspace(-3, 1, 10)
res = cca_graph_rrr_cv(
    X, Y, Gamma,
    r=r, lambdas=lambdas, kfolds=5,
    standardize=True, LW_Sy=False, rho=10.0, niter=2000, verbose=False
)
print("lambda* =", res["lambda"])

# Plot: RMSE vs lambda
plt.figure()
plt.semilogx(lambdas, res["rmse"])
plt.xlabel("lambda")
plt.ylabel("MSE")
plt.title("cca_graph_rrr_cv: CV curve")
plt.tight_layout()
plt.show()
