# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
E2: Non-IID Robustness (Real Implementation)
Tests AEGIS performance under non-IID data distributions using Dirichlet α.

Target:
- Accuracy drop from α=1.0 to α=0.1 ≤ 12%
- Detection AUC ≥ 0.85 at α=0.1
- FPR @ TPR=90% ≤ 0.15 at α=0.1
"""

from typing import Dict
from experiments.simulator import FLScenario, run_fl


def run_e2_noniid_robustness(config: Dict) -> Dict:
    """E2: Non-IID robustness test (real FL simulation).

    Tests how AEGIS handles non-IID data distributions (Dirichlet α).

    Args:
        config: Must contain:
            - alpha: Dirichlet concentration parameter (1.0=IID, 0.1=highly non-IID)
            - adversary_rate: Fraction of Byzantine clients (default 0.20)
            - seed: Random seed

    Returns:
        Metrics including clean_acc, robust_acc, auc, fpr_at_tpr90
    """
    alpha = config.get("alpha", 1.0)
    adversary_rate = config.get("adversary_rate", 0.20)
    seed = config.get("seed", 101)

    scenario = FLScenario(
        n_clients=50,
        byz_frac=adversary_rate,
        noniid_alpha=alpha,
        attack="model_replacement" if adversary_rate > 0 else "none",  # Stronger attack
        seed=seed,
        n_samples_per_client=100,
        n_features=50,
        n_classes=5,
        participation_rate=0.6,  # Tuned participation
        attack_lambda=10.0,  # Model replacement scaling
    )

    # Run FL simulation (using tuned defaults: 20 rounds, lr=0.05, 5 local_epochs, cosine decay)
    metrics = run_fl(
        scenario,
        aggregator="aegis",
    )

    return {
        "clean_acc": metrics["clean_acc"],
        "robust_acc": metrics["robust_acc"],
        "auc": metrics["auc"],
        "fpr_at_tpr90": metrics["fpr_at_tpr90"],
        "alpha": alpha,
        "adversary_rate": adversary_rate,
        "convergence_round": metrics["convergence_round"],
        "flags_per_round": metrics["flags_per_round"],
    }
