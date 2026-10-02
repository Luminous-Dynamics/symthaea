// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic physics/model consistency qualification.
//!
//! Sensor agreement is evidence quality, not physical truth. This module adds
//! an independent model trajectory so that a coherent sensor cohort can still
//! be challenged by a physics/model prediction.
//!
//! The result is deliberately an evidence-quality state, not a safety verdict.
//! Model mismatch can mean physical change, model error, stale calibration, or
//! an unmodelled operating regime. No single layer is allowed to silently turn
//! disagreement into recovery or operational authority.

use serde::{Deserialize, Serialize};

use crate::sensor_temporal::{TemporalFusionDecision, TemporalFusionState};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhysicsConsistencyState {
    Consistent,
    Inconsistent,
    InsufficientEvidence,
    Quarantined,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhysicsConsistencyIssue {
    EmptyComponentIdentity,
    ComponentMismatch,
    EmptyModelIdentity,
    EmptyModelVersion,
    EmptyEvidence,
    EmptyConfiguration,
    MissingTemporalEvidence,
    TemporalEvidenceNotCorroborated,
    MissingObservedDelta,
    StalePrediction,
    FuturePrediction,
    ConfigurationMismatch,
    PredictionResidualTooHigh,
    PredictionUncertaintyTooHigh,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicsPrediction {
    pub component_id: String,
    pub model_id: String,
    pub model_version: String,
    pub timestamp_ms: u64,
    pub predicted_delta: f64,
    pub uncertainty: f64,
    pub evidence_id: String,
    pub configuration_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicsConsistencyPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub maximum_prediction_age_ms: u64,
    pub maximum_prediction_residual_milli: u32,
    pub maximum_prediction_uncertainty_milli: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicsConsistencyDecision {
    pub schema_version: String,
    pub policy_id: String,
    pub component_id: String,
    pub model_id: String,
    pub model_version: String,
    pub state: PhysicsConsistencyState,
    pub observed_delta: Option<f64>,
    pub predicted_delta: Option<f64>,
    pub residual: Option<f64>,
    pub issues: Vec<PhysicsConsistencyIssue>,
}

#[derive(Debug, Clone)]
pub struct PhysicsConsistencyGate {
    policy: PhysicsConsistencyPolicy,
}

impl PhysicsConsistencyGate {
    pub fn new(policy: PhysicsConsistencyPolicy) -> Result<Self, &'static str> {
        if policy.schema_version.trim().is_empty()
            || policy.policy_id.trim().is_empty()
            || policy.maximum_prediction_age_ms == 0
            || policy.maximum_prediction_residual_milli == 0
            || policy.maximum_prediction_uncertainty_milli == 0
        {
            return Err("invalid physics consistency policy");
        }
        Ok(Self { policy })
    }

    pub fn assess(
        &self,
        component_id: &str,
        expected_configuration_digest: &str,
        temporal: &TemporalFusionDecision,
        prediction: &PhysicsPrediction,
        now_ms: u64,
    ) -> PhysicsConsistencyDecision {
        let mut issues = Vec::new();

        if component_id.trim().is_empty() {
            issues.push(PhysicsConsistencyIssue::EmptyComponentIdentity);
        }
        if prediction.component_id != component_id
            && !prediction.component_id.trim().is_empty()
            && !component_id.trim().is_empty()
        {
            issues.push(PhysicsConsistencyIssue::ComponentMismatch);
        }
        if prediction.model_id.trim().is_empty() {
            issues.push(PhysicsConsistencyIssue::EmptyModelIdentity);
        }
        if prediction.model_version.trim().is_empty() {
            issues.push(PhysicsConsistencyIssue::EmptyModelVersion);
        }
        if prediction.evidence_id.trim().is_empty() {
            issues.push(PhysicsConsistencyIssue::EmptyEvidence);
        }
        if prediction.configuration_digest.trim().is_empty()
            || expected_configuration_digest.trim().is_empty()
        {
            issues.push(PhysicsConsistencyIssue::EmptyConfiguration);
        }
        if prediction.timestamp_ms > now_ms {
            issues.push(PhysicsConsistencyIssue::FuturePrediction);
        } else if now_ms.saturating_sub(prediction.timestamp_ms)
            > self.policy.maximum_prediction_age_ms
        {
            issues.push(PhysicsConsistencyIssue::StalePrediction);
        }

        if prediction.configuration_digest != expected_configuration_digest
            && !prediction.configuration_digest.trim().is_empty()
            && !expected_configuration_digest.trim().is_empty()
        {
            issues.push(PhysicsConsistencyIssue::ConfigurationMismatch);
        }

        if temporal.state != TemporalFusionState::Corroborated {
            issues.push(PhysicsConsistencyIssue::TemporalEvidenceNotCorroborated);
        }

        let observed_delta = temporal.consensus_delta;
        if observed_delta.is_none() {
            issues.push(PhysicsConsistencyIssue::MissingObservedDelta);
        }

        if !prediction.predicted_delta.is_finite() || !prediction.uncertainty.is_finite() {
            issues.push(PhysicsConsistencyIssue::PredictionUncertaintyTooHigh);
        } else if prediction.uncertainty < 0.0
            || ((prediction.uncertainty * 1000.0).round() as u32)
                > self.policy.maximum_prediction_uncertainty_milli
        {
            issues.push(PhysicsConsistencyIssue::PredictionUncertaintyTooHigh);
        }

        let residual = observed_delta.map(|observed| (observed - prediction.predicted_delta).abs());
        if let Some(residual) = residual {
            if !residual.is_finite()
                || ((residual * 1000.0).round() as u32)
                    > self.policy.maximum_prediction_residual_milli
            {
                issues.push(PhysicsConsistencyIssue::PredictionResidualTooHigh);
            }
        }

        let hard_quarantine = issues.iter().any(|issue| {
            matches!(
                issue,
                PhysicsConsistencyIssue::EmptyComponentIdentity
                    | PhysicsConsistencyIssue::EmptyModelIdentity
                    | PhysicsConsistencyIssue::EmptyModelVersion
                    | PhysicsConsistencyIssue::EmptyEvidence
                    | PhysicsConsistencyIssue::EmptyConfiguration
                    | PhysicsConsistencyIssue::FuturePrediction
                    | PhysicsConsistencyIssue::StalePrediction
                    | PhysicsConsistencyIssue::ConfigurationMismatch
            )
        });

        let state = if hard_quarantine {
            PhysicsConsistencyState::Quarantined
        } else if temporal.state != TemporalFusionState::Corroborated
            || observed_delta.is_none()
            || issues.iter().any(|issue| {
                matches!(
                    issue,
                    PhysicsConsistencyIssue::PredictionUncertaintyTooHigh
                        | PhysicsConsistencyIssue::PredictionResidualTooHigh
                )
            })
        {
            if temporal.state != TemporalFusionState::Corroborated || observed_delta.is_none() {
                PhysicsConsistencyState::InsufficientEvidence
            } else {
                PhysicsConsistencyState::Inconsistent
            }
        } else {
            PhysicsConsistencyState::Consistent
        };

        PhysicsConsistencyDecision {
            schema_version: self.policy.schema_version.clone(),
            policy_id: self.policy.policy_id.clone(),
            component_id: component_id.into(),
            model_id: prediction.model_id.clone(),
            model_version: prediction.model_version.clone(),
            state,
            observed_delta,
            predicted_delta: Some(prediction.predicted_delta),
            residual,
            issues,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sensor_temporal::{TemporalFusionDecision, TemporalFusionIssue};

    fn temporal(state: TemporalFusionState, delta: Option<f64>) -> TemporalFusionDecision {
        TemporalFusionDecision {
            schema_version: "0.1".into(),
            policy_id: "temporal-v1".into(),
            state,
            trusted_sensor_ids: vec!["a".into(), "b".into()],
            consensus_delta: delta,
            sensor_deltas: vec![("a".into(), delta.unwrap_or(0.0)), ("b".into(), delta.unwrap_or(0.0))],
            issues: if state == TemporalFusionState::Corroborated {
                vec![]
            } else {
                vec![TemporalFusionIssue::TemporalDisagreement]
            },
        }
    }

    fn prediction(delta: f64) -> PhysicsPrediction {
        PhysicsPrediction {
            component_id: "wing-root".into(),
            model_id: "reduced-order-structural-v1".into(),
            model_version: "2026.10".into(),
            timestamp_ms: 2_000,
            predicted_delta: delta,
            uncertainty: 0.05,
            evidence_id: "model-e-2000".into(),
            configuration_digest: "cfg-wing-root".into(),
        }
    }

    fn gate() -> PhysicsConsistencyGate {
        PhysicsConsistencyGate::new(PhysicsConsistencyPolicy {
            schema_version: "0.1".into(),
            policy_id: "physics-v1".into(),
            maximum_prediction_age_ms: 10_000,
            maximum_prediction_residual_milli: 100,
            maximum_prediction_uncertainty_milli: 250,
        })
        .unwrap()
    }

    #[test]
    fn independent_model_agrees_with_sensor_trajectory() {
        let d = gate().assess(
            "wing-root",
            "cfg-wing-root",
            &temporal(TemporalFusionState::Corroborated, Some(1.0)),
            &prediction(1.0),
            2_000,
        );
        assert_eq!(d.state, PhysicsConsistencyState::Consistent);
        assert_eq!(d.residual, Some(0.0));
        assert!(d.issues.is_empty());
    }

    #[test]
    fn model_disagreement_remains_visible() {
        let d = gate().assess(
            "wing-root",
            "cfg-wing-root",
            &temporal(TemporalFusionState::Corroborated, Some(1.0)),
            &prediction(1.3),
            2_000,
        );
        assert_eq!(d.state, PhysicsConsistencyState::Inconsistent);
        assert!(d.issues.contains(&PhysicsConsistencyIssue::PredictionResidualTooHigh));
    }

    #[test]
    fn stale_model_evidence_is_quarantined() {
        let mut p = prediction(1.0);
        p.timestamp_ms = 0;
        let d = gate().assess(
            "wing-root",
            "cfg-wing-root",
            &temporal(TemporalFusionState::Corroborated, Some(1.0)),
            &p,
            20_001,
        );
        assert_eq!(d.state, PhysicsConsistencyState::Quarantined);
        assert!(d.issues.contains(&PhysicsConsistencyIssue::StalePrediction));
    }

    #[test]
    fn future_model_evidence_is_quarantined() {
        let mut p = prediction(1.0);
        p.timestamp_ms = 2_001;
        let d = gate().assess(
            "wing-root",
            "cfg-wing-root",
            &temporal(TemporalFusionState::Corroborated, Some(1.0)),
            &p,
            2_000,
        );
        assert_eq!(d.state, PhysicsConsistencyState::Quarantined);
        assert!(d.issues.contains(&PhysicsConsistencyIssue::FuturePrediction));
    }

    #[test]
    fn component_mismatch_is_quarantined() {
        let mut p = prediction(1.0);
        p.component_id = "tail-root".into();
        let d = gate().assess(
            "wing-root",
            "cfg-wing-root",
            &temporal(TemporalFusionState::Corroborated, Some(1.0)),
            &p,
            2_000,
        );
        assert_eq!(d.state, PhysicsConsistencyState::Quarantined);
        assert!(d.issues.contains(&PhysicsConsistencyIssue::ComponentMismatch));
    }

    #[test]
    fn configuration_mismatch_is_quarantined() {
        let d = gate().assess(
            "wing-root",
            "cfg-new",
            &temporal(TemporalFusionState::Corroborated, Some(1.0)),
            &prediction(1.0),
            2_000,
        );
        assert_eq!(d.state, PhysicsConsistencyState::Quarantined);
        assert!(d.issues.contains(&PhysicsConsistencyIssue::ConfigurationMismatch));
    }

    #[test]
    fn conflicted_temporal_evidence_cannot_be_promoted_by_model_agreement() {
        let d = gate().assess(
            "wing-root",
            "cfg-wing-root",
            &temporal(TemporalFusionState::Conflicted, Some(1.0)),
            &prediction(1.0),
            2_000,
        );
        assert_eq!(d.state, PhysicsConsistencyState::InsufficientEvidence);
        assert!(d.issues.contains(&PhysicsConsistencyIssue::TemporalEvidenceNotCorroborated));
    }

    #[test]
    fn model_uncertainty_is_a_first_class_gate() {
        let mut p = prediction(1.0);
        p.uncertainty = 0.5;
        let d = gate().assess(
            "wing-root",
            "cfg-wing-root",
            &temporal(TemporalFusionState::Corroborated, Some(1.0)),
            &p,
            2_000,
        );
        assert_eq!(d.state, PhysicsConsistencyState::Inconsistent);
        assert!(d.issues.contains(&PhysicsConsistencyIssue::PredictionUncertaintyTooHigh));
    }
}
