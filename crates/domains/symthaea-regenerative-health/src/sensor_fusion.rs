// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic multi-sensor corroboration.
//!
//! Fusion is an evidence-quality boundary, not a safety verdict. It prevents a
//! single trusted-looking channel from silently dominating when independent
//! sensing channels disagree or when too few qualified channels are available.

use serde::{Deserialize, Serialize};

use crate::sensor_health::{SensorHealthDecision, SensorHealthState, SensorObservation};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensorFusionState {
    Corroborated,
    InsufficientEvidence,
    Conflicted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensorFusionIssue {
    EmptyInput,
    InsufficientTrustedSensors,
    ResidualDisagreement,
    DuplicateSensor,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensorFusionPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub minimum_trusted_sensors: u16,
    pub maximum_residual_spread_milli: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorFusionInput {
    pub observation: SensorObservation,
    pub decision: SensorHealthDecision,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorFusionDecision {
    pub schema_version: String,
    pub policy_id: String,
    pub state: SensorFusionState,
    pub trusted_sensor_ids: Vec<String>,
    pub consensus_residual: Option<f64>,
    pub issues: Vec<SensorFusionIssue>,
}

#[derive(Debug, Clone)]
pub struct SensorFusionGate {
    policy: SensorFusionPolicy,
}

impl SensorFusionGate {
    pub fn new(policy: SensorFusionPolicy) -> Result<Self, &'static str> {
        if policy.schema_version.trim().is_empty()
            || policy.policy_id.trim().is_empty()
            || policy.minimum_trusted_sensors == 0
            || policy.maximum_residual_spread_milli == 0
        {
            return Err("invalid sensor fusion policy");
        }
        Ok(Self { policy })
    }

    pub fn assess(&self, inputs: &[SensorFusionInput]) -> SensorFusionDecision {
        let mut issues = Vec::new();

        if inputs.is_empty() {
            issues.push(SensorFusionIssue::EmptyInput);
        }

        let mut trusted = Vec::new();
        let mut sensor_ids = std::collections::BTreeSet::new();

        for input in inputs {
            if !sensor_ids.insert(input.observation.sensor_id.clone()) {
                issues.push(SensorFusionIssue::DuplicateSensor);
                continue;
            }
            if input.decision.state == SensorHealthState::Trusted {
                trusted.push(input);
            }
        }

        let required = self.policy.minimum_trusted_sensors as usize;
        if trusted.len() < required {
            issues.push(SensorFusionIssue::InsufficientTrustedSensors);
        }

        let mut values: Vec<f64> = trusted
            .iter()
            .map(|input| input.observation.normalized_residual)
            .collect();
        values.sort_by(|a, b| a.total_cmp(b));

        let consensus_residual = if values.is_empty() {
            None
        } else {
            let mid = values.len() / 2;
            Some(if values.len() % 2 == 0 {
                (values[mid - 1] + values[mid]) / 2.0
            } else {
                values[mid]
            })
        };

        if let (Some(min), Some(max)) = (values.first(), values.last()) {
            let spread_milli = ((max - min).max(0.0) * 1000.0).round() as u32;
            if spread_milli > self.policy.maximum_residual_spread_milli {
                issues.push(SensorFusionIssue::ResidualDisagreement);
            }
        }

        let state = if issues.iter().any(|issue| {
            matches!(
                issue,
                SensorFusionIssue::ResidualDisagreement | SensorFusionIssue::DuplicateSensor
            )
        }) {
            SensorFusionState::Conflicted
        } else if trusted.len() < required {
            SensorFusionState::InsufficientEvidence
        } else {
            SensorFusionState::Corroborated
        };

        SensorFusionDecision {
            schema_version: self.policy.schema_version.clone(),
            policy_id: self.policy.policy_id.clone(),
            state,
            trusted_sensor_ids: trusted
                .iter()
                .map(|input| input.observation.sensor_id.clone())
                .collect(),
            consensus_residual,
            issues,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sensor_health::{SensorHealthGate, SensorHealthPolicy};

    fn sensor_gate() -> SensorHealthGate {
        SensorHealthGate::new(SensorHealthPolicy {
            schema_version: "0.1".into(),
            policy_id: "sensor-v1".into(),
            degraded_residual_milli: 2_000,
            untrusted_residual_milli: 4_000,
            maximum_observation_age_ms: 1_000,
        })
        .unwrap()
    }

    fn observation(sensor_id: &str, residual: f64) -> SensorObservation {
        SensorObservation {
            observation_id: format!("obs-{sensor_id}"),
            sensor_id: sensor_id.into(),
            timestamp_ms: 1_000,
            normalized_residual: residual,
            uncertainty: 1.0,
            evidence_id: format!("e-{sensor_id}"),
            configuration_digest: "cfg-1".into(),
        }
    }

    fn input(sensor_id: &str, residual: f64) -> SensorFusionInput {
        let observation = observation(sensor_id, residual);
        let decision = sensor_gate().assess(&observation, Some("cfg-1"), 1_000);
        SensorFusionInput { observation, decision }
    }

    fn fusion_gate() -> SensorFusionGate {
        SensorFusionGate::new(SensorFusionPolicy {
            schema_version: "0.1".into(),
            policy_id: "fusion-v1".into(),
            minimum_trusted_sensors: 2,
            maximum_residual_spread_milli: 500,
        })
        .unwrap()
    }

    #[test]
    fn two_trusted_sensors_corroborate() {
        let d = fusion_gate().assess(&[input("strain-a", 0.5), input("strain-b", 0.7)]);
        assert_eq!(d.state, SensorFusionState::Corroborated);
        assert_eq!(d.consensus_residual, Some(0.6));
    }

    #[test]
    fn one_trusted_sensor_is_insufficient() {
        let d = fusion_gate().assess(&[input("strain-a", 0.5)]);
        assert_eq!(d.state, SensorFusionState::InsufficientEvidence);
        assert!(d.issues.contains(&SensorFusionIssue::InsufficientTrustedSensors));
    }

    #[test]
    fn conflicting_trusted_sensors_do_not_corroborate() {
        let d = fusion_gate().assess(&[input("strain-a", 0.5), input("strain-b", 1.5)]);
        assert_eq!(d.state, SensorFusionState::Conflicted);
        assert!(d.issues.contains(&SensorFusionIssue::ResidualDisagreement));
    }

    #[test]
    fn degraded_sensor_does_not_count_toward_quorum() {
        let d = fusion_gate().assess(&[input("strain-a", 0.5), input("strain-b", 2.5)]);
        assert_eq!(d.state, SensorFusionState::InsufficientEvidence);
        assert_eq!(d.trusted_sensor_ids, vec!["strain-a".to_string()]);
    }

    #[test]
    fn duplicate_sensor_ids_are_conflicted() {
        let d = fusion_gate().assess(&[input("strain-a", 0.5), input("strain-a", 0.5)]);
        assert_eq!(d.state, SensorFusionState::Conflicted);
        assert!(d.issues.contains(&SensorFusionIssue::DuplicateSensor));
    }
}
