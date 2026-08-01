"""Certificate-gated parametric butterfly whitening."""

from .compressibility import complementary_action_probe_audit, complementary_rank_audit
from .losses import certificate_gated_action_loss
from .operator import (
    ButterflyEnsembleOperator,
    ButterflyOperator,
    ParametricButterflyEnsembleOperator,
    ParametricButterflyOperator,
)
from .training import (
    butterfly_loss_gradient,
    ensemble_loss_gradient,
    train_parametric_butterfly,
    train_parametric_ensemble,
)
from .whitening import (
    HODLROperator,
    SpectralLowRankOperator,
    WhiteningFamily,
    energy_score_perturbation_bound,
    operator_energy_score_error_bound,
    gaussian_frobenius_certificate,
    tie_aware_auc_perturbation_bound,
)

__all__ = [
    "ButterflyEnsembleOperator",
    "ButterflyOperator",
    "HODLROperator",
    "ParametricButterflyEnsembleOperator",
    "ParametricButterflyOperator",
    "SpectralLowRankOperator",
    "WhiteningFamily",
    "butterfly_loss_gradient",
    "certificate_gated_action_loss",
    "complementary_action_probe_audit",
    "complementary_rank_audit",
    "energy_score_perturbation_bound",
    "operator_energy_score_error_bound",
    "ensemble_loss_gradient",
    "gaussian_frobenius_certificate",
    "tie_aware_auc_perturbation_bound",
    "train_parametric_butterfly",
    "train_parametric_ensemble",
]

__version__ = "2.1.0"
