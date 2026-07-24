// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Data-minimized export surfaces for therapeutic state.
//!
//! Default persistence types may contain raw narratives, formulation text,
//! contacts, and other highly sensitive material. These summaries intentionally
//! preserve operational metrics while omitting raw text and contact details.

use crate::client_model::{ClientModel, CoreAffectSnapshot, RiskLevel};
use crate::formulation::CaseFormulation;
use crate::narrative_integration::TherapeuticNarrative;
use crate::safety::SafetyPlan;
use crate::shadow::ShadowDetector;
use serde::{Deserialize, Serialize};

/// Data sensitivity classification used by deployment storage policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TherapeuticDataClass {
    Operational,
    Sensitive,
    HighlySensitive,
}

/// Data categories that require separate retention decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TherapeuticDataCategory {
    AffectTrajectory,
    SymptomTrajectory,
    BehavioralPatterns,
    SafetyPlanContacts,
    NarrativeText,
    FormulationText,
    ShadowContent,
    AuditReceipts,
}

impl TherapeuticDataCategory {
    pub const fn class(self) -> TherapeuticDataClass {
        match self {
            Self::AuditReceipts | Self::AffectTrajectory => TherapeuticDataClass::Sensitive,
            Self::SymptomTrajectory
            | Self::BehavioralPatterns
            | Self::SafetyPlanContacts
            | Self::NarrativeText
            | Self::FormulationText
            | Self::ShadowContent => TherapeuticDataClass::HighlySensitive,
        }
    }
}

/// Text-free client-model export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedactedClientSnapshot {
    pub current_affect: CoreAffectSnapshot,
    pub affect_observation_count: usize,
    pub symptom_observation_count: usize,
    pub behavioral_pattern_count: usize,
    pub session_count: u32,
    pub cycle_count: u64,
    pub risk_level: RiskLevel,
    pub distress: f32,
    pub affect_trend: f32,
    pub mean_arousal: f32,
    pub rdoc_burden_index: f32,
}

/// Contact-free safety-plan export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedactedSafetyPlanSummary {
    pub warning_sign_count: usize,
    pub coping_strategy_count: usize,
    pub social_distraction_count: usize,
    pub support_contact_count: usize,
    pub professional_contact_count: usize,
    pub crisis_resource_count: usize,
    pub environmental_safety_step_count: usize,
    pub minimally_complete: bool,
}

/// Text-free case-formulation export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedactedFormulationSummary {
    pub predisposing_count: usize,
    pub precipitating_count: usize,
    pub perpetuating_count: usize,
    pub protective_count: usize,
    pub belief_chain_count: usize,
    pub mean_factor_confidence: f32,
    pub resilience_ratio: f32,
    pub actionable: bool,
}

/// Text-free narrative export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedactedNarrativeSummary {
    pub fragment_count: usize,
    pub traumatic_fragment_count: usize,
    pub trauma_proportion: f32,
    pub mean_integration: f32,
    pub coherence: f32,
    pub temporal_coherence: f32,
    pub narrative_arc_score: f32,
}

/// Text-free metadata for one shadow fragment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedactedShadowFragment {
    pub fragment_id: u64,
    pub first_detected_cycle: u64,
    pub last_active_cycle: u64,
    pub emotional_valence: f32,
    pub emotional_arousal: f32,
    pub recurrence_count: u32,
    pub cumulative_prediction_error: f32,
    pub pressure: f32,
    pub queued_for_dream: bool,
    pub dream_processing_count: u32,
    pub dream_phi_improvement: f32,
}

/// Raw-content-free shadow-state export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedactedShadowSummary {
    pub fragment_count: usize,
    pub dream_queue_depth: usize,
    pub total_pressure: f32,
    pub surfacing_indicated: bool,
    pub fragments: Vec<RedactedShadowFragment>,
}

impl ClientModel {
    pub fn redacted_snapshot(&self) -> RedactedClientSnapshot {
        RedactedClientSnapshot {
            current_affect: self.current_affect,
            affect_observation_count: self.affect_trajectory.len(),
            symptom_observation_count: self.symptom_trajectory.len(),
            behavioral_pattern_count: self.behavioral_patterns.len(),
            session_count: self.session_count,
            cycle_count: self.cycle_count,
            risk_level: self.risk_level,
            distress: self.distress(),
            affect_trend: self.affect_trend(),
            mean_arousal: self.mean_arousal(),
            rdoc_burden_index: self.rdoc_burden_index(),
        }
    }
}

impl SafetyPlan {
    pub fn redacted_summary(&self) -> RedactedSafetyPlanSummary {
        RedactedSafetyPlanSummary {
            warning_sign_count: self.warning_signs.len(),
            coping_strategy_count: self.coping_strategies.len(),
            social_distraction_count: self.social_distractions.len(),
            support_contact_count: self.support_contacts.len(),
            professional_contact_count: self.professional_contacts.len(),
            crisis_resource_count: self.crisis_resources.len(),
            environmental_safety_step_count: self.environmental_safety.len(),
            minimally_complete: self.is_complete(),
        }
    }
}

impl CaseFormulation {
    pub fn redacted_summary(&self) -> RedactedFormulationSummary {
        let factors = self
            .predisposing
            .iter()
            .chain(&self.precipitating)
            .chain(&self.perpetuating)
            .chain(&self.protective);
        let (confidence_sum, factor_count) = factors
            .fold((0.0_f32, 0_usize), |(sum, count), factor| {
                (sum + factor.confidence, count + 1)
            });
        RedactedFormulationSummary {
            predisposing_count: self.predisposing.len(),
            precipitating_count: self.precipitating.len(),
            perpetuating_count: self.perpetuating.len(),
            protective_count: self.protective.len(),
            belief_chain_count: self.belief_chains.len(),
            mean_factor_confidence: if factor_count == 0 {
                0.0
            } else {
                confidence_sum / factor_count as f32
            },
            resilience_ratio: self.resilience_ratio(),
            actionable: self.is_actionable(),
        }
    }
}

impl TherapeuticNarrative {
    pub fn redacted_summary(&self) -> RedactedNarrativeSummary {
        RedactedNarrativeSummary {
            fragment_count: self.fragments.len(),
            traumatic_fragment_count: self
                .fragments
                .iter()
                .filter(|fragment| fragment.is_traumatic)
                .count(),
            trauma_proportion: self.trauma_proportion(),
            mean_integration: self.mean_integration(),
            coherence: self.coherence,
            temporal_coherence: self.temporal_coherence(),
            narrative_arc_score: self.narrative_arc_score(),
        }
    }
}

impl ShadowDetector {
    pub fn redacted_summary(&self) -> RedactedShadowSummary {
        RedactedShadowSummary {
            fragment_count: self.fragment_count(),
            dream_queue_depth: self.dream_queue_depth(),
            total_pressure: self.total_pressure(),
            surfacing_indicated: self.surfacing_indicated(),
            fragments: self
                .fragments()
                .iter()
                .map(|fragment| RedactedShadowFragment {
                    fragment_id: fragment.fragment_id,
                    first_detected_cycle: fragment.first_detected_cycle,
                    last_active_cycle: fragment.last_active_cycle,
                    emotional_valence: fragment.emotional_valence,
                    emotional_arousal: fragment.emotional_arousal,
                    recurrence_count: fragment.recurrence_count,
                    cumulative_prediction_error: fragment.cumulative_prediction_error,
                    pressure: fragment.pressure(),
                    queued_for_dream: fragment.queued_for_dream,
                    dream_processing_count: fragment.dream_processing_count,
                    dream_phi_improvement: fragment.dream_phi_improvement,
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::narrative_integration::NarrativeFragment;

    #[test]
    fn narrative_summary_contains_no_raw_text() {
        let mut narrative = TherapeuticNarrative::new();
        narrative.integrate_fragment(NarrativeFragment::new(
            "highly sensitive narrative text",
            1,
            -0.8,
            true,
        ));
        let serialized = serde_json::to_string(&narrative.redacted_summary()).unwrap();
        assert!(!serialized.contains("highly sensitive"));
        assert_eq!(narrative.redacted_summary().fragment_count, 1);
    }

    #[test]
    fn safety_summary_omits_contacts_and_phone_numbers() {
        let mut plan = SafetyPlan::template();
        plan.support_contacts
            .push("Person +27 00 000 0000".to_string());
        let serialized = serde_json::to_string(&plan.redacted_summary()).unwrap();
        assert!(!serialized.contains("+27"));
        assert_eq!(plan.redacted_summary().support_contact_count, 1);
    }
}
