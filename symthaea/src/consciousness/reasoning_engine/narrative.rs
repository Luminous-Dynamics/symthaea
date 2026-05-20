// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Best-Effort Narrative Generator
//!
//! Produces human-readable narratives from reasoning results.
//! Narrative generation is NOT budget-critical — it runs only in
//! Tier 2 when remaining budget > 3ms, and is always deferred if
//! budget is tight.

use crate::consciousness::counterfactual::CausalQueryOutcome;
use crate::consciousness::epistemic_conflict::{ConflictKind, ConflictMatrix};
use crate::consciousness::temporal_planning::types::MctsResult;
use crate::consciousness::tool_gate::types::GateResult;

/// Generate a narrative from reasoning components.
///
/// Returns `None` if there's nothing interesting to narrate.
pub fn generate_narrative(
    phi_eff: f64,
    reliability: f64,
    conflicts: &ConflictMatrix,
    plan: Option<&MctsResult>,
    gate: Option<&GateResult>,
    counterfactual: Option<&CausalQueryOutcome>,
) -> Option<String> {
    let mut parts = Vec::new();

    // Consciousness state
    let state_desc = if reliability > 0.8 {
        "High reliability: theories in strong consensus."
    } else if reliability > 0.5 {
        "Moderate reliability: some theoretical disagreement."
    } else if reliability > 0.3 {
        "Low reliability: significant theoretical conflict."
    } else {
        "Very low reliability: theories in severe disagreement. Epistemic caution engaged."
    };
    parts.push(format!(
        "Φ_eff={:.3} (R={:.2}). {}",
        phi_eff, reliability, state_desc
    ));

    // Dominant conflict
    if let Some(kind) = conflicts.dominant_kind {
        let conflict_desc = conflict_kind_narrative(kind);
        if conflicts.max_magnitude() > 0.3 {
            parts.push(format!("Dominant conflict: {}", conflict_desc));
        }
    }

    // Planning
    if let Some(plan) = plan {
        if plan.did_plan {
            parts.push(format!(
                "MCTS: {} iterations, confidence={:.2}, expected_value={:.2}.",
                plan.iterations, plan.confidence, plan.expected_value,
            ));
        }
    }

    // Tool gate
    if let Some(gate) = gate {
        if gate.is_allowed() {
            parts.push(format!("Tool gate: ALLOWED (risk={:?}).", gate.risk_level));
        } else {
            let fallback_desc = gate
                .fallback
                .as_ref()
                .map(|f| format!("{:?}", f))
                .unwrap_or_else(|| "none".to_string());
            parts.push(format!(
                "Tool gate: BLOCKED (risk={:?}, fallback={}).",
                gate.risk_level, fallback_desc,
            ));
        }
    }

    // Counterfactual
    if let Some(cf) = counterfactual {
        let cf_desc = match cf {
            CausalQueryOutcome::Identified {
                method, confidence, ..
            } => {
                format!(
                    "Causal effect identified via {:?} (conf={:.2}).",
                    method, confidence
                )
            }
            CausalQueryOutcome::Unidentified { reason, .. } => {
                format!("Causal effect unidentifiable: {:?}.", reason)
            }
            CausalQueryOutcome::AssumptionRequired {
                assumption,
                plausibility,
                ..
            } => {
                format!(
                    "Causal effect requires assumption '{}' (plausibility={:.2}).",
                    assumption.condition, plausibility,
                )
            }
        };
        parts.push(cf_desc);
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

/// Human-readable description of a conflict kind.
fn conflict_kind_narrative(kind: ConflictKind) -> &'static str {
    match kind {
        ConflictKind::IntegrationCollapse => {
            "Integration collapse — information integration has broken down."
        }
        ConflictKind::NoBroadcast => "No broadcast — global workspace is inactive.",
        ConflictKind::AttentionalInstability => "Attentional instability — focus is scattered.",
        ConflictKind::UnreliablePrediction => "Unreliable prediction — high surprise rate.",
        ConflictKind::ShallowRecurrence => "Shallow recurrence — processing depth insufficient.",
        ConflictKind::UngroundedRepresentation => {
            "Ungrounded representation — lacking embodied context."
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::epistemic_conflict::{ConflictScore, TheoryId};

    fn make_conflicts(magnitude: f64) -> ConflictMatrix {
        let conflicts: Vec<ConflictScore> = (0..15)
            .map(|i| {
                let a = TheoryId::ALL[i % 6];
                let b = TheoryId::ALL[(i + 1) % 6];
                ConflictScore::new(a, b, magnitude, ConflictKind::IntegrationCollapse)
            })
            .collect();
        ConflictMatrix::new(conflicts)
    }

    #[test]
    fn test_narrative_generation() {
        let conflicts = make_conflicts(0.5);
        let narrative = generate_narrative(0.4, 0.6, &conflicts, None, None, None);
        assert!(narrative.is_some());
        let text = narrative.unwrap();
        assert!(text.contains("Φ_eff=0.400"));
        assert!(text.contains("Dominant conflict"));
    }

    #[test]
    fn test_narrative_with_plan() {
        let conflicts = make_conflicts(0.1);
        let plan = MctsResult {
            best_action_idx: Some(0),
            confidence: 0.8,
            iterations: 50,
            tree_size: 50,
            expected_value: 0.7,
            did_plan: true,
        };
        let narrative = generate_narrative(0.6, 0.8, &conflicts, Some(&plan), None, None);
        assert!(narrative.is_some());
        let text = narrative.unwrap();
        assert!(text.contains("MCTS"));
        assert!(text.contains("50 iterations"));
    }

    #[test]
    fn test_low_reliability_narrative() {
        let conflicts = make_conflicts(0.8);
        let narrative = generate_narrative(0.1, 0.15, &conflicts, None, None, None);
        assert!(narrative.is_some());
        let text = narrative.unwrap();
        assert!(text.contains("Very low reliability"));
    }
}
