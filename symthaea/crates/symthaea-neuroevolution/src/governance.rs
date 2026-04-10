// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Governance gate for evolved thresholds.
//!
//! Evolved threshold parameters must pass consciousness-gated approval before
//! being applied to the cognitive loop. This prevents runaway self-modification
//! by requiring:
//!
//! 1. Minimum Phi (0.5) — the system must be sufficiently conscious to approve changes
//! 2. Threshold fitness ≥ 0.6 — evolved parameters must be internally consistent
//! 3. EMA blending — changes are applied gradually (alpha=0.1), not hard-switched
//! 4. Safety immutability — safety/moral thresholds cannot be modified
//!
//! # References
//!
//! - Schmidhuber (2003): Gödel machines — self-referential improvement with proof requirements
//! - Tononi (2004): Phi as a measure of consciousness quality

use serde::{Deserialize, Serialize};

use crate::threshold_genome::ThresholdPhenotype;

/// EMA blending alpha for threshold application (10 cycles to converge).
const THRESHOLD_BLEND_ALPHA: f32 = 0.1;

/// Minimum Phi required to approve threshold changes.
const MIN_PHI_FOR_APPROVAL: f64 = 0.5;

/// Minimum threshold fitness required for approval.
const MIN_THRESHOLD_FITNESS: f64 = 0.6;

/// Result of a governance gate check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GovernanceResult {
    /// Approved: thresholds can be applied (with EMA blending).
    Approved {
        /// The blended phenotype (current + evolved, weighted by alpha).
        blended: ThresholdPhenotype,
        /// Confidence in the approval (0.0-1.0).
        confidence: f64,
    },
    /// Rejected: thresholds cannot be applied.
    Rejected {
        /// Reason for rejection.
        reason: String,
    },
}

/// Gate evolved thresholds through consciousness-gated approval.
///
/// Returns `Approved` with a blended phenotype if all checks pass,
/// or `Rejected` with a reason if any check fails.
pub fn gate_threshold_proposal(
    current: &ThresholdPhenotype,
    proposed: &ThresholdPhenotype,
    current_phi: f64,
    threshold_fitness: f64,
) -> GovernanceResult {
    // Check 1: Minimum consciousness level
    if current_phi < MIN_PHI_FOR_APPROVAL {
        return GovernanceResult::Rejected {
            reason: format!(
                "Phi {:.3} below minimum {:.1} for threshold approval",
                current_phi, MIN_PHI_FOR_APPROVAL
            ),
        };
    }

    // Check 2: Threshold fitness (internal consistency)
    if threshold_fitness < MIN_THRESHOLD_FITNESS {
        return GovernanceResult::Rejected {
            reason: format!(
                "threshold fitness {:.3} below minimum {:.1}",
                threshold_fitness, MIN_THRESHOLD_FITNESS
            ),
        };
    }

    // Check 3: Apply EMA blending (gradual transition, not hard switch)
    let blended = blend_thresholds(current, proposed, THRESHOLD_BLEND_ALPHA);

    // Confidence based on fitness and Phi
    let confidence = (threshold_fitness * current_phi).clamp(0.0, 1.0);

    GovernanceResult::Approved {
        blended,
        confidence,
    }
}

/// Blend two threshold phenotypes via EMA: `result = (1-alpha)*current + alpha*proposed`.
pub fn blend_thresholds(
    current: &ThresholdPhenotype,
    proposed: &ThresholdPhenotype,
    alpha: f32,
) -> ThresholdPhenotype {
    let a = alpha;
    let b = 1.0 - alpha;
    ThresholdPhenotype {
        fep_surprise_scale: b * current.fep_surprise_scale + a * proposed.fep_surprise_scale,
        fep_lr_decay: b * current.fep_lr_decay + a * proposed.fep_lr_decay,
        dream_base_interval: lerp_u64(current.dream_base_interval, proposed.dream_base_interval, a),
        dream_min_interval: lerp_u64(current.dream_min_interval, proposed.dream_min_interval, a),
        neuromod_d2_baseline: b as f64 * current.neuromod_d2_baseline
            + a as f64 * proposed.neuromod_d2_baseline,
        neuromod_ne_phasic_threshold: b * current.neuromod_ne_phasic_threshold
            + a * proposed.neuromod_ne_phasic_threshold,
        neuromod_arousal_ema_decay: b * current.neuromod_arousal_ema_decay
            + a * proposed.neuromod_arousal_ema_decay,
        homeostasis_recalibrate_high: b * current.homeostasis_recalibrate_high
            + a * proposed.homeostasis_recalibrate_high,
        homeostasis_recalibrate_low: b * current.homeostasis_recalibrate_low
            + a * proposed.homeostasis_recalibrate_low,
        neuromod_ema_alpha: b * current.neuromod_ema_alpha + a * proposed.neuromod_ema_alpha,
        frustration_dampen_threshold: b as f64 * current.frustration_dampen_threshold
            + a as f64 * proposed.frustration_dampen_threshold,
        engagement_low_threshold: b as f64 * current.engagement_low_threshold
            + a as f64 * proposed.engagement_low_threshold,
        flow_exploration_increment: b * current.flow_exploration_increment
            + a * proposed.flow_exploration_increment,
        coherence_low: b * current.coherence_low + a * proposed.coherence_low,
        arousal_trap_threshold: b * current.arousal_trap_threshold
            + a * proposed.arousal_trap_threshold,
        self_model_weight_high: b * current.self_model_weight_high
            + a * proposed.self_model_weight_high,
        homeostasis_pull_cruise: b * current.homeostasis_pull_cruise
            + a * proposed.homeostasis_pull_cruise,
        confidence_crash_threshold: b as f64 * current.confidence_crash_threshold
            + a as f64 * proposed.confidence_crash_threshold,
    }
}

fn lerp_u64(a: u64, b: u64, alpha: f32) -> u64 {
    let result = (1.0 - alpha) * a as f32 + alpha * b as f32;
    result.round() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approved_with_good_phi_and_fitness() {
        let current = ThresholdPhenotype::default();
        let proposed = ThresholdPhenotype {
            fep_surprise_scale: 5.0,
            ..ThresholdPhenotype::default()
        };
        let result = gate_threshold_proposal(&current, &proposed, 0.7, 0.8);
        match result {
            GovernanceResult::Approved {
                blended,
                confidence,
            } => {
                // Blended should be 90% current + 10% proposed
                let expected = 0.9 * 3.0 + 0.1 * 5.0; // 3.2
                assert!((blended.fep_surprise_scale - expected).abs() < 0.01);
                assert!(confidence > 0.0);
            }
            GovernanceResult::Rejected { reason } => {
                panic!("should be approved, got rejected: {reason}");
            }
        }
    }

    #[test]
    fn test_rejected_low_phi() {
        let current = ThresholdPhenotype::default();
        let proposed = ThresholdPhenotype::default();
        let result = gate_threshold_proposal(&current, &proposed, 0.3, 0.8);
        assert!(matches!(result, GovernanceResult::Rejected { .. }));
    }

    #[test]
    fn test_rejected_low_fitness() {
        let current = ThresholdPhenotype::default();
        let proposed = ThresholdPhenotype::default();
        let result = gate_threshold_proposal(&current, &proposed, 0.7, 0.4);
        assert!(matches!(result, GovernanceResult::Rejected { .. }));
    }

    #[test]
    fn test_blend_identity() {
        let pheno = ThresholdPhenotype::default();
        let blended = blend_thresholds(&pheno, &pheno, 0.5);
        assert!((blended.fep_surprise_scale - pheno.fep_surprise_scale).abs() < 0.01);
        assert!((blended.fep_lr_decay - pheno.fep_lr_decay).abs() < 0.001);
    }

    #[test]
    fn test_blend_alpha_zero_returns_current() {
        let current = ThresholdPhenotype::default();
        let proposed = ThresholdPhenotype {
            fep_surprise_scale: 9.0,
            ..ThresholdPhenotype::default()
        };
        let blended = blend_thresholds(&current, &proposed, 0.0);
        assert!((blended.fep_surprise_scale - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_blend_alpha_one_returns_proposed() {
        let current = ThresholdPhenotype::default();
        let proposed = ThresholdPhenotype {
            fep_surprise_scale: 9.0,
            ..ThresholdPhenotype::default()
        };
        let blended = blend_thresholds(&current, &proposed, 1.0);
        assert!((blended.fep_surprise_scale - 9.0).abs() < 0.01);
    }
}
