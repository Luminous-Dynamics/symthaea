# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Debug AEGIS detection to see why flagging isn't working."""
import sys
sys.path.insert(0, "experiments")

from simulator import FLScenario, run_fl
import numpy as np

# Create a simple backdoor scenario
scenario = FLScenario(
    n_clients=50,
    byz_frac=0.20,
    noniid_alpha=1.0,
    attack="backdoor",
    seed=101,
    n_features=50,
    n_classes=5,
    participation_rate=0.6,
    backdoor_target_label=0,
    backdoor_trigger_feature=0,
    backdoor_trigger_value=5.0,
    backdoor_triggered_frac=0.2,
)

print("Running FL with AEGIS (3 rounds for quick test)...")
print(f"Scenario: {scenario.n_clients} clients, {scenario.byz_frac*100}% Byzantine")
print(f"Attack: {scenario.attack}")
print(f"Participation: {scenario.participation_rate*100}%")
print()

# Monkey-patch to add debugging
original_apply_aegis = None

def debug_apply_aegis(*args, **kwargs):
    """Wrapper to print detection info."""
    scores, median_grad, flagged_indices, detection_info = original_apply_aegis(*args, **kwargs)
    
    print(f"\n=== AEGIS Detection Debug ===")
    print(f"Round: {kwargs.get('round_idx', 0)}")
    print(f"Clients: {len(args[0])}")
    print(f"q_frac_applied: {detection_info['q_frac_applied']:.3f}")
    print(f"Flagged: {len(flagged_indices)} (hard={detection_info['num_hard_blocked']}, soft={detection_info['num_soft_flagged']})")
    print(f"Scores: min={scores.min():.3f}, max={scores.max():.3f}, mean={scores.mean():.3f}")
    print(f"Cosine to median: mean={detection_info['cos_to_median_mean']:.3f}")
    print(f"MAD z-score: mean={detection_info['mad_z_mean']:.3f}, max={detection_info['mad_z_max']:.3f}")
    if len(flagged_indices) > 0:
        print(f"Flagged client indices: {flagged_indices[:5]}...")
    
    return scores, median_grad, flagged_indices, detection_info

# Monkey-patch
import simulator
original_apply_aegis = simulator.apply_aegis_detection
simulator.apply_aegis_detection = debug_apply_aegis

# Run
metrics = run_fl(
    scenario,
    aggregator="aegis",
    rounds=3,
    local_epochs=2,
    lr=0.05,
)

print("\n=== Final Metrics ===")
print(f"ASR: {metrics['asr']:.3f}")
print(f"Clean acc: {metrics['clean_acc']:.3f}")
print(f"AUC: {metrics['auc']:.3f}")
print(f"Flags per round: {metrics['flags_per_round']:.3f}")
