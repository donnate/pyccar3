__all__ = [
    # Core sparse CCA via reduced-rank (elementwise or entry-group penalties)
    "ecca",
    "ecca_across_lambdas",
    "ecca_eval",
    "ecca_cv",
    "soft_thresh",
    "soft_thresh_group",
    "soft_thresh2",
    # Row-group RRR-CCA (ADMM / CVX)
    "cca_rrr",
    "cca_rrr_cv",
    "cca_rrr_cv_folds",
    # Graph-regularized RRR-CCA
    "cca_graph_rrr",
    "cca_graph_rrr_cv",
    "cca_graph_rrr_cv_folds",
    # Grouped-by-rows RRR-CCA (explicit groups of predictor rows)
    "cca_group_rrr",
    "cca_group_rrr_cv",
    "cca_group_rrr_cv_folds",
]

__version__ = "0.2.0"

from .ecca import (
    ecca,
    ecca_across_lambdas,
    ecca_eval,
    ecca_cv,
    soft_thresh,
    soft_thresh_group,
    soft_thresh2,
)

from .cca_rrr import (
    cca_rrr,
    cca_rrr_cv,
    cca_rrr_cv_folds,
)

from .graph_cca_rrr import (
    cca_graph_rrr,
    cca_graph_rrr_cv,
    cca_graph_rrr_cv_folds,
)

from .cca_group_rrr import (
    cca_group_rrr,
    cca_group_rrr_cv,
    cca_group_rrr_cv_folds,
)