// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic temporal corroboration for sensor fields.
//!
//! Cross-sectional agreement is insufficient against a slowly drifting or
//! stuck sensor. This module compares *changes* across independent sensors.
//! A genuine physical transition should move qualified sensors coherently;
//! an isolated sensing fault should produce a delta that departs from the
//! cohort's temporal consensus.
//!
//! This remains an evidence-quality boundary. It does not infer why the
//! physical state changed and does not certify that the physical system is safe.

use serde::{Deserialize, Serialize};

use crate::{sensor_health::{SensorHealthDecision, SensorHealthState, SensorObservation}, sensor_topology::{SensorTopologyAttestation, SensorTopologyAttestationIssue}};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalFusionState {
    Corroborated,
    InsufficientEvidence,
    Conflicted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalFusionIssue {
    EmptyInput,
    InsufficientTrustedSensors,
    InsufficientIndependentGroups,
    InvalidIndependenceProvenance,
    IndependenceConfigurationMismatch,
    TopologyAttestation(SensorTopologyAttestationIssue),
    DuplicateSensor,
    NonMonotonicTime,
    TemporalDisagreement,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalFusionPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub minimum_trusted_sensors: u16,
    pub minimum_independent_groups: u16,
    pub maximum_delta_disagreement_milli: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensorIndependenceBinding {
    pub schema_version: String,
    pub sensor_id: String,
    pub component_id: String,
    pub asset_id: String,
    /// Stable physical/common-mode dependency domain.
    pub independence_group: String,
    /// Qualified topology/dependency declaration governing this binding.
    pub topology_attestation: SensorTopologyAttestation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalSensorPair {
    pub previous: SensorObservation,
    pub previous_decision: SensorHealthDecision,
    pub current: SensorObservation,
    pub current_decision: SensorHealthDecision,
    pub independence: SensorIndependenceBinding,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalFusionDecision {
    pub schema_version: String,
    pub policy_id: String,
    pub state: TemporalFusionState,
    pub trusted_sensor_ids: Vec<String>,
    pub consensus_delta: Option<f64>,
    pub sensor_deltas: Vec<(String, f64)>,
    pub independent_group_count: usize,
    pub independent_group_ids: Vec<String>,
    pub issues: Vec<TemporalFusionIssue>,
}

#[derive(Debug, Clone)]
pub struct TemporalFusionGate {
    policy: TemporalFusionPolicy,
}

impl TemporalFusionGate {
    pub fn new(policy: TemporalFusionPolicy) -> Result<Self, &'static str> {
        if policy.schema_version.trim().is_empty()
            || policy.policy_id.trim().is_empty()
            || policy.minimum_trusted_sensors == 0
            || policy.minimum_independent_groups == 0
            || policy.maximum_delta_disagreement_milli == 0
        {
            return Err("invalid temporal fusion policy");
        }
        Ok(Self { policy })
    }

    pub fn assess(&self, pairs: &[TemporalSensorPair]) -> TemporalFusionDecision {
        let mut issues = Vec::new();
        let mut trusted = Vec::new();
        let mut sensor_ids = std::collections::BTreeSet::new();

        if pairs.is_empty() {
            issues.push(TemporalFusionIssue::EmptyInput);
        }

        for pair in pairs {
            if !sensor_ids.insert(pair.current.sensor_id.clone()) {
                issues.push(TemporalFusionIssue::DuplicateSensor);
                continue;
            }

            if pair.current.timestamp_ms <= pair.previous.timestamp_ms {
                issues.push(TemporalFusionIssue::NonMonotonicTime);
                continue;
            }

            if pair.previous.sensor_id != pair.current.sensor_id {
                issues.push(TemporalFusionIssue::DuplicateSensor);
                continue;
            }

            if pair.independence.schema_version.trim().is_empty()
                || pair.independence.sensor_id != pair.current.sensor_id
                || pair.independence.component_id.trim().is_empty()
                || pair.independence.asset_id.trim().is_empty()
                || pair.independence.independence_group.trim().is_empty()
            {
                issues.push(TemporalFusionIssue::InvalidIndependenceProvenance);
                continue;
            }
            if pair.previous.configuration_digest != pair.current.configuration_digest
                || pair.current.configuration_digest.trim().is_empty()
            {
                issues.push(TemporalFusionIssue::IndependenceConfigurationMismatch);
                continue;
            }

            if let Err(issue) = pair.independence.topology_attestation.validate(
                &pair.independence.asset_id,
                &pair.independence.component_id,
                &pair.current.configuration_digest,
                pair.previous.timestamp_ms,
            ) {
                issues.push(TemporalFusionIssue::TopologyAttestation(issue));
                continue;
            }
            if let Err(issue) = pair.independence.topology_attestation.validate(
                &pair.independence.asset_id,
                &pair.independence.component_id,
                &pair.current.configuration_digest,
                pair.current.timestamp_ms,
            ) {
                issues.push(TemporalFusionIssue::TopologyAttestation(issue));
                continue;
            }

            if pair.previous_decision.state == SensorHealthState::Trusted
                && pair.current_decision.state == SensorHealthState::Trusted
            {
                trusted.push(pair);
            }
        }

        let required = self.policy.minimum_trusted_sensors as usize;
        let independent_groups: std::collections::BTreeSet<_> = trusted
            .iter()
            .map(|pair| pair.independence.independence_group.as_str())
            .filter(|group| !group.trim().is_empty())
            .collect();
        let required_groups = self.policy.minimum_independent_groups as usize;
        if trusted.len() < required {
            issues.push(TemporalFusionIssue::InsufficientTrustedSensors);
        }
        if independent_groups.len() < required_groups {
            issues.push(TemporalFusionIssue::InsufficientIndependentGroups);
        }

        let mut deltas: Vec<(String, f64)> = trusted
            .iter()
            .map(|pair| {
                (
                    pair.current.sensor_id.clone(),
                    pair.current.normalized_residual - pair.previous.normalized_residual,
                )
            })
            .collect();
        deltas.sort_by(|a, b| a.0.cmp(&b.0));

        let mut sorted_deltas: Vec<f64> = deltas.iter().map(|(_, delta)| *delta).collect();
        sorted_deltas.sort_by(|a, b| a.total_cmp(b));

        let consensus_delta = if sorted_deltas.is_empty() {
            None
        } else {
            let mid = sorted_deltas.len() / 2;
            Some(if sorted_deltas.len() % 2 == 0 {
                (sorted_deltas[mid - 1] + sorted_deltas[mid]) / 2.0
            } else {
                sorted_deltas[mid]
            })
        };

        if let Some(consensus) = consensus_delta {
            if deltas.iter().any(|(_, delta)| {
                ((delta - consensus).abs() * 1000.0).round() as u32
                    > self.policy.maximum_delta_disagreement_milli
            }) {
                issues.push(TemporalFusionIssue::TemporalDisagreement);
            }
        }

        let state = if issues.iter().any(|issue| {
            matches!(
                issue,
                TemporalFusionIssue::DuplicateSensor
                    | TemporalFusionIssue::InvalidIndependenceProvenance
                    | TemporalFusionIssue::IndependenceConfigurationMismatch
                    | TemporalFusionIssue::TopologyAttestation(_)
                    | TemporalFusionIssue::NonMonotonicTime
                    | TemporalFusionIssue::TemporalDisagreement
            )
        }) {
            TemporalFusionState::Conflicted
        } else if trusted.len() < required || independent_groups.len() < required_groups {
            TemporalFusionState::InsufficientEvidence
        } else {
            TemporalFusionState::Corroborated
        };

        TemporalFusionDecision {
            schema_version: self.policy.schema_version.clone(),
            policy_id: self.policy.policy_id.clone(),
            state,
            trusted_sensor_ids: trusted
                .iter()
                .map(|pair| pair.current.sensor_id.clone())
                .collect(),
            consensus_delta,
            sensor_deltas: deltas,
            independent_group_count: independent_groups.len(),
            independent_group_ids: independent_groups.iter().map(|g| (*g).to_owned()).collect(),
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
            maximum_observation_age_ms: 10_000,
        })
        .unwrap()
    }

    fn observation(sensor_id: &str, timestamp_ms: u64, residual: f64) -> SensorObservation {
        SensorObservation {
            observation_id: format!("obs-{sensor_id}-{timestamp_ms}"),
            sensor_id: sensor_id.into(),
            timestamp_ms,
            normalized_residual: residual,
            uncertainty: 1.0,
            evidence_id: format!("e-{sensor_id}-{timestamp_ms}"),
            configuration_digest: "cfg-1".into(),
        }
    }

    fn pair(sensor_id: &str, previous: f64, current: f64) -> TemporalSensorPair {
        let gate = sensor_gate();
        let previous_observation = observation(sensor_id, 1_000, previous);
        let current_observation = observation(sensor_id, 2_000, current);
        let previous_decision = gate.assess(&previous_observation, Some("cfg-1"), 2_000);
        let current_decision = gate.assess(&current_observation, Some("cfg-1"), 2_000);
        TemporalSensorPair {
            previous: previous_observation,
            previous_decision,
            current: current_observation,
            current_decision,
            independence: SensorIndependenceBinding {
                schema_version: "0.1".into(),
                sensor_id: sensor_id.into(),
                component_id: "wing-root".into(),
                asset_id: "vehicle-1".into(),
                independence_group: format!("group-{sensor_id}"),
                topology_attestation: SensorTopologyAttestation {
                    schema_version: "0.1".into(),
                    asset_id: "vehicle-1".into(),
                    component_id: "wing-root".into(),
                    topology_id: "topology-wing-root".into(),
                    topology_version: "1".into(),
                    topology_digest: "topology-v1".into(),
                    configuration_digest: "cfg-1".into(),
                    issued_at_ms: 500,
                    valid_until_ms: 2_500,
                    evidence_id: format!("independence-{sensor_id}"),
                },
            },
        }
    }

    fn fusion_gate() -> TemporalFusionGate {
        TemporalFusionGate::new(TemporalFusionPolicy {
            schema_version: "0.1".into(),
            policy_id: "temporal-fusion-v1".into(),
            minimum_trusted_sensors: 2,
            minimum_independent_groups: 2,
            maximum_delta_disagreement_milli: 250,
        })
        .unwrap()
    }

    #[test]
    fn coherent_physical_change_remains_corroborated() {
        let d = fusion_gate().assess(&[
            pair("strain-a", 0.5, 1.5),
            pair("strain-b", 0.6, 1.6),
            pair("strain-c", 0.4, 1.4),
        ]);
        assert_eq!(d.state, TemporalFusionState::Corroborated);
        assert_eq!(d.consensus_delta, Some(1.0));
    }

    #[test]
    fn stuck_sensor_isolated_from_real_physical_change_is_conflicted() {
        let d = fusion_gate().assess(&[
            pair("strain-a", 0.5, 1.5),
            pair("strain-b", 0.6, 1.6),
            pair("strain-c", 0.4, 0.4),
        ]);
        assert_eq!(d.state, TemporalFusionState::Conflicted);
        assert!(d.issues.contains(&TemporalFusionIssue::TemporalDisagreement));
    }

    #[test]
    fn gradual_sensor_drift_is_detected_against_cohort_change() {
        let d = fusion_gate().assess(&[
            pair("strain-a", 0.5, 1.5),
            pair("strain-b", 0.6, 1.6),
            pair("strain-c", 0.4, 1.1),
        ]);
        assert_eq!(d.state, TemporalFusionState::Conflicted);
        assert!(d.issues.contains(&TemporalFusionIssue::TemporalDisagreement));
    }


    #[test]
    fn colocated_sensors_do_not_create_independent_quorum() {
        let mut a = pair("strain-a", 0.5, 1.5);
        let mut b = pair("strain-b", 0.6, 1.6);
        a.independence.independence_group = "wing-root-a".into();
        b.independence.independence_group = "wing-root-a".into();
        let d = fusion_gate().assess(&[a, b]);
        assert_eq!(d.state, TemporalFusionState::InsufficientEvidence);
        assert!(d.issues.contains(&TemporalFusionIssue::InsufficientIndependentGroups));
    }

    #[test]
    fn malformed_independence_binding_cannot_enter_quorum() {
        let mut p = pair("strain-a", 0.5, 1.5);
        p.independence.topology_attestation.topology_digest.clear();
        let d = fusion_gate().assess(&[p, pair("strain-b", 0.6, 1.6)]);
        assert_eq!(d.state, TemporalFusionState::Conflicted);
        assert!(d.issues.contains(&TemporalFusionIssue::InvalidIndependenceProvenance));
    }

    #[test]
    fn independence_binding_configuration_must_match_observation() {
        let mut p = pair("strain-a", 0.5, 1.5);
        p.independence.topology_attestation.configuration_digest = "cfg-attacker".into();
        let d = fusion_gate().assess(&[p, pair("strain-b", 0.6, 1.6)]);
        assert_eq!(d.state, TemporalFusionState::Conflicted);
        assert!(d.issues.contains(&TemporalFusionIssue::IndependenceConfigurationMismatch));
    }

    #[test]
    fn independent_group_identity_is_auditable() {
        let d = fusion_gate().assess(&[
            pair("strain-a", 0.5, 1.5),
            pair("strain-b", 0.6, 1.6),
        ]);
        assert_eq!(d.independent_group_count, 2);
        assert_eq!(
            d.independent_group_ids,
            vec!["group-strain-a", "group-strain-b"]
        );
    }

    #[test]
    fn stale_topology_attestation_cannot_create_quorum() {
        let mut p = pair("strain-a", 0.5, 1.5);
        p.independence.topology_attestation.valid_until_ms = 1_999;
        let d = fusion_gate().assess(&[p, pair("strain-b", 0.6, 1.6)]);
        assert_eq!(d.state, TemporalFusionState::Conflicted);
        assert!(d.issues.iter().any(|issue| matches!(
            issue,
            TemporalFusionIssue::TopologyAttestation(
                SensorTopologyAttestationIssue::StaleAttestation
            )
        )));
    }

    #[test]
    fn future_topology_attestation_cannot_create_quorum() {
        let mut p = pair("strain-a", 0.5, 1.5);
        p.independence.topology_attestation.issued_at_ms = 1_001;
        let d = fusion_gate().assess(&[p, pair("strain-b", 0.6, 1.6)]);
        assert_eq!(d.state, TemporalFusionState::Conflicted);
        assert!(d.issues.iter().any(|issue| matches!(
            issue,
            TemporalFusionIssue::TopologyAttestation(
                SensorTopologyAttestationIssue::FutureAttestation
            )
        )));
    }

    #[test]
    fn topology_substitution_cannot_enter_independence_quorum() {
        let mut p = pair("strain-a", 0.5, 1.5);
        p.independence.topology_attestation.component_id = "tail-root".into();
        let d = fusion_gate().assess(&[p, pair("strain-b", 0.6, 1.6)]);
        assert_eq!(d.state, TemporalFusionState::Conflicted);
        assert!(d.issues.iter().any(|issue| matches!(
            issue,
            TemporalFusionIssue::TopologyAttestation(
                SensorTopologyAttestationIssue::TopologyIdentityMismatch
            )
        )));
    }

    #[test]
    fn configuration_rotation_requires_new_topology_attestation() {
        let mut p = pair("strain-a", 0.5, 1.5);
        p.current.configuration_digest = "cfg-2".into();
        let d = fusion_gate().assess(&[p, pair("strain-b", 0.6, 1.6)]);
        assert_eq!(d.state, TemporalFusionState::Conflicted);
        assert!(d.issues.contains(&TemporalFusionIssue::IndependenceConfigurationMismatch));
    }

    #[test]
    fn non_monotonic_observation_is_not_admitted() {
        let mut p = pair("strain-a", 0.5, 1.5);
        p.current.timestamp_ms = 1_000;
        let d = fusion_gate().assess(&[p, pair("strain-b", 0.6, 1.6)]);
        assert_eq!(d.state, TemporalFusionState::Conflicted);
        assert!(d.issues.contains(&TemporalFusionIssue::NonMonotonicTime));
    }
}
