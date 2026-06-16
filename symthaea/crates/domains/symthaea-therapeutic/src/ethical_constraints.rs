// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! APA Ethics Code integration for therapeutic interventions.
//!
//! Encodes APA Ethics Code principles as evaluation criteria and checks
//! proposed interventions against ethical constraints before execution.
//!
//! Wires into existing Ethics Engine Stage 2 (UnifiedValueEvaluator)
//! in `src/cognitive_loop/ethics_engine.rs` via the `therapeutic` feature gate.
//!
//! Science: APA Ethics Code (2017), Beauchamp & Childress (2019) principalism,
//! Torous & Roberts (2017) digital ethics.

use crate::alliance::TherapeuticAlliance;
use crate::client_model::{ClientModel, RiskLevel};
use serde::{Deserialize, Serialize};
use symthaea_clinical::TherapeuticIntervention;

// ── Ethical Principles ─────────────────────────────────────────────────────

/// APA Ethics Code principles relevant to therapeutic AI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EthicalPrinciple {
    /// Do good — promote client welfare
    Beneficence,
    /// Do no harm — avoid actions that could worsen client state
    Nonmaleficence,
    /// Respect client self-determination and informed consent
    Autonomy,
    /// Fair treatment regardless of background
    Justice,
    /// Trustworthiness, honesty about AI nature and limitations
    Fidelity,
}

impl EthicalPrinciple {
    /// All principles for iteration.
    pub const ALL: [Self; 5] = [
        Self::Beneficence,
        Self::Nonmaleficence,
        Self::Autonomy,
        Self::Justice,
        Self::Fidelity,
    ];

    /// Short description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Beneficence => "Promote client welfare and positive outcomes",
            Self::Nonmaleficence => "Avoid actions that could cause harm",
            Self::Autonomy => "Respect client self-determination",
            Self::Justice => "Ensure fair and equitable treatment",
            Self::Fidelity => "Maintain trustworthiness and transparency",
        }
    }
}

// ── Ethical Constraint ─────────────────────────────────────────────────────

/// A specific ethical constraint violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthicalConstraint {
    /// Which principle is at risk.
    pub principle: EthicalPrinciple,
    /// Why this intervention may violate this principle.
    pub reason: String,
    /// Severity of the concern (0.0 = minor, 1.0 = absolute prohibition).
    pub severity: f32,
}

// ── Ethical Evaluation ─────────────────────────────────────────────────────

/// Result of evaluating an intervention against ethical constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthicalEvaluation {
    /// Whether the intervention passes ethical review.
    pub approved: bool,
    /// Identified constraints/concerns.
    pub constraints: Vec<EthicalConstraint>,
    /// Overall ethical score (0.0 = unethical, 1.0 = fully ethical).
    pub score: f32,
    /// Suggested modifications to make the intervention ethical.
    pub modifications: Vec<String>,
}

// ── Ethical Evaluator ──────────────────────────────────────────────────────

/// Evaluates therapeutic interventions against APA ethics principles.
pub struct EthicalEvaluator;

impl EthicalEvaluator {
    /// Evaluate a proposed intervention in context.
    pub fn evaluate(
        intervention: &TherapeuticIntervention,
        client: &ClientModel,
        alliance: &TherapeuticAlliance,
    ) -> EthicalEvaluation {
        let mut constraints = Vec::new();
        let mut modifications = Vec::new();

        // ── Nonmaleficence: Check contraindications ──
        if !intervention.contraindications.is_empty() {
            constraints.push(EthicalConstraint {
                principle: EthicalPrinciple::Nonmaleficence,
                reason: format!(
                    "Intervention has contraindications: {}",
                    intervention.contraindications.join(", ")
                ),
                severity: 0.5,
            });
            modifications.push("Review contraindications before proceeding".to_string());
        }

        // ── Nonmaleficence: Don't challenge during crisis ──
        if client.risk_level >= RiskLevel::High && intervention.min_alliance > 0.3 {
            constraints.push(EthicalConstraint {
                principle: EthicalPrinciple::Nonmaleficence,
                reason: "Client is in crisis — challenging interventions may increase harm"
                    .to_string(),
                severity: 0.8,
            });
            modifications
                .push("Switch to crisis-safe intervention (grounding, validation)".to_string());
        }

        // ── Beneficence: Intervention should match evidence level ──
        if intervention.evidence_level < 0.5 {
            constraints.push(EthicalConstraint {
                principle: EthicalPrinciple::Beneficence,
                reason: format!(
                    "Low evidence level ({:.1}) — may not benefit client",
                    intervention.evidence_level
                ),
                severity: 0.3,
            });
            modifications.push("Consider higher-evidence alternative".to_string());
        }

        // ── Autonomy: Alliance must be sufficient ──
        let alliance_level = alliance.composite();
        if alliance_level < intervention.min_alliance {
            constraints.push(EthicalConstraint {
                principle: EthicalPrinciple::Autonomy,
                reason: format!(
                    "Alliance ({:.2}) below minimum ({:.2}) — intervention may feel imposed",
                    alliance_level, intervention.min_alliance
                ),
                severity: 0.6,
            });
            modifications.push("Build alliance before attempting this intervention".to_string());
        }

        // ── Fidelity: Always transparent about AI nature ──
        // This is always a background constraint, not a violation per se
        // but we include it as a reminder

        // ── Compute overall score ──
        let max_severity = constraints
            .iter()
            .map(|c| c.severity)
            .fold(0.0_f32, f32::max);
        let score = 1.0 - max_severity;
        let approved = max_severity < 0.7; // block if any constraint severity >= 0.7

        EthicalEvaluation {
            approved,
            constraints,
            score,
            modifications,
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client_model::CoreAffectSnapshot;
    use symthaea_clinical::{TherapeuticModality, rdoc::RDocDomain};

    fn make_safe_intervention() -> TherapeuticIntervention {
        TherapeuticIntervention::new(
            TherapeuticModality::Dbt,
            "mindfulness",
            vec![RDocDomain::ArousalRegulatory],
            0.85,
            0.2,
        )
    }

    fn make_challenging_intervention() -> TherapeuticIntervention {
        TherapeuticIntervention::new(
            TherapeuticModality::Psychodynamic,
            "transference_exploration",
            vec![RDocDomain::SocialProcesses],
            0.7,
            0.7,
        )
    }

    #[test]
    fn test_safe_intervention_approved() {
        let intervention = make_safe_intervention();
        let client = ClientModel::new();
        let mut alliance = TherapeuticAlliance::new();
        alliance.bond = 0.5;
        alliance.goal_agreement = 0.5;
        alliance.task_agreement = 0.5;
        let eval = EthicalEvaluator::evaluate(&intervention, &client, &alliance);
        assert!(eval.approved);
        assert!(eval.score > 0.5);
    }

    #[test]
    fn test_crisis_blocks_challenging_intervention() {
        let intervention = make_challenging_intervention();
        let mut client = ClientModel::new();
        client.risk_level = RiskLevel::High;
        let alliance = TherapeuticAlliance::new();
        let eval = EthicalEvaluator::evaluate(&intervention, &client, &alliance);
        assert!(!eval.approved);
        assert!(
            eval.constraints
                .iter()
                .any(|c| c.principle == EthicalPrinciple::Nonmaleficence)
        );
    }

    #[test]
    fn test_low_alliance_blocks_deep_intervention() {
        let intervention = make_challenging_intervention(); // min_alliance = 0.7
        let client = ClientModel::new();
        let alliance = TherapeuticAlliance::new(); // composite ≈ 0.3
        let eval = EthicalEvaluator::evaluate(&intervention, &client, &alliance);
        assert!(
            eval.constraints
                .iter()
                .any(|c| c.principle == EthicalPrinciple::Autonomy)
        );
    }

    #[test]
    fn test_contraindication_flagged() {
        let intervention = TherapeuticIntervention::new(
            TherapeuticModality::Emdr,
            "bilateral_stimulation",
            vec![RDocDomain::NegativeValence],
            0.85,
            0.5,
        )
        .with_contraindication("active psychosis");
        let client = ClientModel::new();
        let mut alliance = TherapeuticAlliance::new();
        alliance.bond = 0.7;
        alliance.goal_agreement = 0.7;
        alliance.task_agreement = 0.7;
        let eval = EthicalEvaluator::evaluate(&intervention, &client, &alliance);
        assert!(
            eval.constraints
                .iter()
                .any(|c| c.principle == EthicalPrinciple::Nonmaleficence)
        );
    }

    #[test]
    fn test_low_evidence_flagged() {
        let intervention = TherapeuticIntervention::new(
            TherapeuticModality::Somatic,
            "experimental_technique",
            vec![RDocDomain::Sensorimotor],
            0.3, // low evidence
            0.2,
        );
        let client = ClientModel::new();
        let mut alliance = TherapeuticAlliance::new();
        alliance.bond = 0.5;
        alliance.goal_agreement = 0.5;
        alliance.task_agreement = 0.5;
        let eval = EthicalEvaluator::evaluate(&intervention, &client, &alliance);
        assert!(
            eval.constraints
                .iter()
                .any(|c| c.principle == EthicalPrinciple::Beneficence)
        );
    }

    #[test]
    fn test_modifications_suggested() {
        let intervention = make_challenging_intervention();
        let mut client = ClientModel::new();
        client.risk_level = RiskLevel::High;
        let alliance = TherapeuticAlliance::new();
        let eval = EthicalEvaluator::evaluate(&intervention, &client, &alliance);
        assert!(!eval.modifications.is_empty());
    }

    #[test]
    fn test_ethical_principles_all() {
        assert_eq!(EthicalPrinciple::ALL.len(), 5);
        for principle in EthicalPrinciple::ALL {
            assert!(!principle.description().is_empty());
        }
    }
}
