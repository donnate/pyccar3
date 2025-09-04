# pyccar3

Group-sparse reduced-rank regression for CCA via ADMM - a Python port of the R implementation.

## Installation

```bash
# Option 1: editable install from source directory
pip install -e .

# Option 2: install from the provided source zip
pip install pyccar3-0.1.0.zip
```

Requires Python >= 3.9 and packages: numpy, scipy, pandas.

## Usage

```python
import numpy as np
from pyccar3 import ecca_cv

n, p, q, r = 200, 50, 40, 2
rng = np.random.default_rng(0)
X = rng.standard_normal((n, p))
Y = rng.standard_normal((n, q))

# Elementwise penalty over B (no groups), small lambda path:
lambdas = [0.0, 0.05, 0.1]

fit = ecca_cv(
    X, Y,
    lambdas=lambdas,
    groups=None,      # or a list of (k,2) index arrays for group penalty
    r=r,
    standardize=True,
    rho=1.0,
    nfold=5,
    select="lambda.min",
    verbose=False
)

print("lambda.opt:", fit["lambda.opt"])
print("loss:", fit["loss"])
print("cor:", fit["cor"])
print("U shape:", fit["U"].shape, "V shape:", fit["V"].shape)
print("CV table:\n", fit["cv.scores"])  # pandas DataFrame (or None if only one lambda)
```

### Groups format (Python)

Each element of `groups` is a `(k, 2)` integer array of zero-based `(row, col)` pairs identifying entries in the `(p x q)` coefficient matrix that belong to that group. This mirrors the R code (which used two-column matrices of 1-based pairs).

Example:

```python
import numpy as np

# Suppose p=5, q=4. A group of 3 entries:
g1 = np.array([
    [0, 0],
    [1, 0],
    [2, 1],
], dtype=int)

# Another group of 2 entries:
g2 = np.array([
    [3, 2],
    [4, 3],
], dtype=int)

groups = [g1, g2]
```

### API

- `ecca(X, Y, lambda_, groups=None, Sx=None, Sy=None, Sxy=None, r=2, standardize=False, rho=1.0, B0=None, eps=1e-4, maxiter=500, verbose=True)`  
  Returns an `ECCAResult(U, V, cor, loss)`.

- `ecca_across_lambdas(X, Y, lambdas, groups=None, r=2, Sx=None, Sy=None, Sxy=None, standardize=True, rho=1.0, B0=None, eps=1e-4, maxiter=500, verbose=True)`  
  Returns `{"U": [...], "V": [...]}` (lists unless a single lambda).

- `ecca_eval(X, Y, lambdas, ..., scoring_method="mse"|"trace", cv_use_median=False)`  
  Returns `{"scores": DataFrame, "lambda.min": float, "lambda.1se": float}`.

- `ecca_cv(...)`  
  Fits with the selected lambda and returns `{"U","V","cor","loss","lambda.opt","cv.scores"}`.

### Notes

- Uses truncated SVD via `scipy.sparse.linalg.svds` when appropriate; falls back to full SVD otherwise.
- Eigenvalues are clipped to `>=0` before taking square roots.
- Stopping rule matches the R version's relative Frobenius change on `B`.
- Loss equals `mean(col_sums((XU - YV)^2))`, matching the R code (the docstring there mentioned 1/n but the code did not divide; we reproduce code behavior).

## Examples

See the `examples/` folder for runnable scripts that demonstrate:
- `ecca_cv` / `ecca` (sparse CCA via reduced-rank) — plots CV curve and canonical variates
- `cca_rrr_cv` (row-group RRR-CCA) — plots CV curve
- `cca_graph_rrr_cv` (graph-regularized RRR-CCA) — plots CV curve
- `cca_group_rrr_cv` (explicit groups of predictor rows) — plots CV curve

Run, for example:

```bash
python examples/example_ecca.py
python examples/example_row_rrr.py
python examples/example_graph_rrr.py
python examples/example_group_rrr.py
```
# pyccar3
