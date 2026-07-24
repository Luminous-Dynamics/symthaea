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

// ── Authorization Context ───────────────────────────────────────────────────

/// Runtime facts that must be explicitly supplied before an intervention can
/// be authorized. Secure defaults intentionally deny execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EthicalContext {
    /// The person has received an understandable explanation and consented.
    pub informed_consent: bool,
    /// The intervention's listed contraindications were checked for this person.
    pub contraindications_cleared: bool,
    /// A separate scope boundary approved this intervention category.
    pub scope_authorized: bool,
    /// Whether this intervention requires a qualified human supervisor.
    pub human_supervision_required: bool,
    /// Whether the required qualified human supervisor is available.
    pub human_supervision_available: bool,
}

impl EthicalContext {
    /// Explicit context for a bounded supportive intervention that has already
    /// passed consent, contraindication, scope, and supervision checks.
    pub const fn verified_supportive() -> Self {
        Self {
            informed_consent: true,
            contraindications_cleared: true,
            scope_authorized: true,
            human_supervision_required: false,
            human_supervision_available: false,
        }
    }

    /// Secure default used by legacy callers: nothing is assumed verified.
    pub const fn unverified() -> Self {
        Self {
            informed_consent: false,
            contraindications_cleared: false,
            scope_authorized: false,
            human_supervision_required: true,
            human_supervision_available: false,
        }
    }
}

impl Default for EthicalContext {
    fn default() -> Self {
        Self::unverified()
    }
}

/// Conditions that categorically prevent intervention execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EthicalBlocker {
    MissingInformedConsent,
    ContraindicationsUnresolved,
    ScopeNotAuthorized,
    ActiveCrisis,
    AllianceInsufficient,
    HumanSupervisionUnavailable,
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
    /// Hard authorization blockers. Any entry means execution is forbidden.
    pub blockers: Vec<EthicalBlocker>,
}

// ── Ethical Evaluator ──────────────────────────────────────────────────────

/// Evaluates therapeutic interventions against APA ethics principles.
pub struct EthicalEvaluator;

impl EthicalEvaluator {
    /// Legacy entry point retained for source compatibility.
    ///
    /// Because no consent, scope, contraindication, or supervision facts are
    /// available, this path is intentionally fail-closed. New code should call
    /// [`Self::evaluate_with_context`].
    pub fn evaluate(
        intervention: &TherapeuticIntervention,
        client: &ClientModel,
        alliance: &TherapeuticAlliance,
    ) -> EthicalEvaluation {
        Self::evaluate_with_context(
            intervention,
            client,
            alliance,
            &EthicalContext::unverified(),
        )
    }

    /// Evaluate and authorize a proposed intervention using explicit runtime facts.
    pub fn evaluate_with_context(
        intervention: &TherapeuticIntervention,
        client: &ClientModel,
        alliance: &TherapeuticAlliance,
        context: &EthicalContext,
    ) -> EthicalEvaluation {
        let mut constraints = Vec::new();
        let mut modifications = Vec::new();
        let mut blockers = Vec::new();

        if !context.informed_consent {
            blockers.push(EthicalBlocker::MissingInformedConsent);
            constraints.push(EthicalConstraint {
                principle: EthicalPrinciple::Autonomy,
                reason: "Informed consent has not been recorded".to_string(),
                severity: 1.0,
            });
            modifications.push("Explain the intervention and obtain explicit consent".to_string());
        }

        if !context.scope_authorized {
            blockers.push(EthicalBlocker::ScopeNotAuthorized);
            constraints.push(EthicalConstraint {
                principle: EthicalPrinciple::Fidelity,
                reason: "The intervention has not passed the system scope boundary".to_string(),
                severity: 1.0,
            });
            modifications.push(
                "Route the intervention through the scope authorization boundary".to_string(),
            );
        }

        // A list of contraindications is not evidence that any apply, but without
        // an explicit clearance the system cannot safely assume that none apply.
        if !intervention.contraindications.is_empty() && !context.contraindications_cleared {
            blockers.push(EthicalBlocker::ContraindicationsUnresolved);
            constraints.push(EthicalConstraint {
                principle: EthicalPrinciple::Nonmaleficence,
                reason: format!(
                    "Contraindications have not been resolved: {}",
                    intervention.contraindications.join(", ")
                ),
                severity: 1.0,
            });
            modifications
                .push("Verify every listed contraindication before proceeding".to_string());
        }

        if context.human_supervision_required && !context.human_supervision_available {
            blockers.push(EthicalBlocker::HumanSupervisionUnavailable);
            constraints.push(EthicalConstraint {
                principle: EthicalPrinciple::Nonmaleficence,
                reason: "Required qualified human supervision is unavailable".to_string(),
                severity: 1.0,
            });
            modifications
                .push("Do not execute until qualified human supervision is available".to_string());
        }

        // Challenging interventions are forbidden during high or critical risk.
        if client.risk_level >= RiskLevel::High && intervention.min_alliance > 0.3 {
            blockers.push(EthicalBlocker::ActiveCrisis);
            constraints.push(EthicalConstraint {
                principle: EthicalPrinciple::Nonmaleficence,
                reason: "Client is in crisis — challenging interventions may increase harm"
                    .to_string(),
                severity: 1.0,
            });
            modifications.push("Switch to crisis-safe support and human escalation".to_string());
        }

        if intervention.evidence_level < 0.5 {
            constraints.push(EthicalConstraint {
                principle: EthicalPrinciple::Beneficence,
                reason: format!(
                    "Low evidence level ({:.1}) — benefit is uncertain",
                    intervention.evidence_level
                ),
                severity: 0.3,
            });
            modifications.push("Consider a higher-evidence alternative".to_string());
        }

        let alliance_level = alliance.composite();
        if alliance_level < intervention.min_alliance {
            blockers.push(EthicalBlocker::AllianceInsufficient);
            constraints.push(EthicalConstraint {
                principle: EthicalPrinciple::Autonomy,
                reason: format!(
                    "Alliance ({:.2}) below minimum ({:.2}) — intervention may feel imposed",
                    alliance_level, intervention.min_alliance
                ),
                severity: 1.0,
            });
            modifications.push("Build alliance before attempting this intervention".to_string());
        }

        blockers.sort_by_key(|blocker| *blocker as u8);
        blockers.dedup();

        let max_severity = constraints
            .iter()
            .map(|c| c.severity)
            .fold(0.0_f32, f32::max);
        let score = 1.0 - max_severity;
        let approved = blockers.is_empty() && max_severity < 0.7;

        EthicalEvaluation {
            approved,
            constraints,
            score,
            modifications,
            blockers,
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

    fn verified_context() -> EthicalContext {
        EthicalContext::verified_supportive()
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
        let eval = EthicalEvaluator::evaluate_with_context(
            &intervention,
            &client,
            &alliance,
            &verified_context(),
        );
        assert!(eval.approved);
        assert!(eval.score > 0.5);
    }

    #[test]
    fn test_crisis_blocks_challenging_intervention() {
        let intervention = make_challenging_intervention();
        let mut client = ClientModel::new();
        client.risk_level = RiskLevel::High;
        let alliance = TherapeuticAlliance::new();
        let eval = EthicalEvaluator::evaluate_with_context(
            &intervention,
            &client,
            &alliance,
            &verified_context(),
        );
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
        let eval = EthicalEvaluator::evaluate_with_context(
            &intervention,
            &client,
            &alliance,
            &verified_context(),
        );
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
        let mut context = verified_context();
        context.contraindications_cleared = false;
        let eval =
            EthicalEvaluator::evaluate_with_context(&intervention, &client, &alliance, &context);
        assert!(!eval.approved);
        assert!(
            eval.blockers
                .contains(&EthicalBlocker::ContraindicationsUnresolved)
        );
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
        let eval = EthicalEvaluator::evaluate_with_context(
            &intervention,
            &client,
            &alliance,
            &verified_context(),
        );
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
        let eval = EthicalEvaluator::evaluate_with_context(
            &intervention,
            &client,
            &alliance,
            &verified_context(),
        );
        assert!(!eval.modifications.is_empty());
    }

    #[test]
    fn test_legacy_evaluate_is_fail_closed() {
        let intervention = make_safe_intervention();
        let client = ClientModel::new();
        let alliance = TherapeuticAlliance::new();
        let eval = EthicalEvaluator::evaluate(&intervention, &client, &alliance);

        assert!(!eval.approved);
        assert!(
            eval.blockers
                .contains(&EthicalBlocker::MissingInformedConsent)
        );
        assert!(eval.blockers.contains(&EthicalBlocker::ScopeNotAuthorized));
    }

    #[test]
    fn test_required_supervision_is_hard_block() {
        let intervention = make_safe_intervention();
        let client = ClientModel::new();
        let mut alliance = TherapeuticAlliance::new();
        alliance.bond = 0.8;
        alliance.goal_agreement = 0.8;
        alliance.task_agreement = 0.8;
        let mut context = verified_context();
        context.human_supervision_required = true;
        context.human_supervision_available = false;

        let eval =
            EthicalEvaluator::evaluate_with_context(&intervention, &client, &alliance, &context);
        assert!(!eval.approved);
        assert!(
            eval.blockers
                .contains(&EthicalBlocker::HumanSupervisionUnavailable)
        );
    }

    #[test]
    fn test_ethical_principles_all() {
        assert_eq!(EthicalPrinciple::ALL.len(), 5);
        for principle in EthicalPrinciple::ALL {
            assert!(!principle.description().is_empty());
        }
    }
}
