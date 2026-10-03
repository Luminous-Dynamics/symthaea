// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic adversarial sensing crucible.
//!
//! The fixture keeps a latent physical trajectory separate from observations.
//! This is intentional: the test must be able to tell "the machine changed"
//! from "the sensor changed" instead of treating sensor output as ground truth.

use crate::{
    sensor_health::{SensorHealthGate, SensorHealthPolicy},
    sensor_temporal::{
        TemporalFusionGate, TemporalFusionPolicy, TemporalFusionState, TemporalSensorPair,
    },
    sensor_health::SensorObservation,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdversarialScenario {
    StableHealthy,
    GenuinePhysicalChange,
    StuckSensor,
    GradualDrift,
    MissingSensor,
    CoordinatedFalseTrajectory,
    RecoveryAfterRepair,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdversarialCrucibleResult {
    pub scenario: AdversarialScenario,
    pub latent_previous: f64,
    pub latent_current: f64,
    pub temporal_state: TemporalFusionState,
    pub trusted_sensor_count: usize,
    pub temporal_disagreement: bool,
    pub observed_current_consensus: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
struct SensorModel {
    offset: f64,
    stuck: bool,
    drift: f64,
    override_current: Option<f64>,
}

impl SensorModel {
    fn observe(self, latent: f64) -> f64 {
        if let Some(value) = self.override_current {
            value
        } else if self.stuck {
            self.offset
        } else {
            latent + self.offset + self.drift
        }
    }
}

fn sensor_gate() -> SensorHealthGate {
    SensorHealthGate::new(SensorHealthPolicy {
        schema_version: "0.1".into(),
        policy_id: "adversarial-sensor-v1".into(),
        degraded_residual_milli: 2_000,
        untrusted_residual_milli: 4_000,
        maximum_observation_age_ms: 10_000,
    })
    .expect("valid sensor policy")
}

fn temporal_gate() -> TemporalFusionGate {
    TemporalFusionGate::new(TemporalFusionPolicy {
        schema_version: "0.1".into(),
        policy_id: "adversarial-temporal-v1".into(),
        minimum_trusted_sensors: 2,
        minimum_independent_groups: 2,
        maximum_delta_disagreement_milli: 250,
        expected_asset_id: "vehicle-1".into(),
        expected_component_id: "wing-root".into(),
        expected_topology_id: "topology-wing-root".into(),
        expected_topology_version: "1".into(),
        expected_topology_digest: "topology-v1".into(),
        expected_attestation_issuer_id: "mycelix-topology-authority".into(),
    })
    .expect("valid temporal policy")
}

fn observation(sensor_id: &str, timestamp_ms: u64, residual: f64) -> SensorObservation {
    SensorObservation {
        observation_id: format!("{sensor_id}-{timestamp_ms}"),
        sensor_id: sensor_id.into(),
        timestamp_ms,
        normalized_residual: residual.max(0.0),
        uncertainty: 1.0,
        evidence_id: format!("e-{sensor_id}-{timestamp_ms}"),
        configuration_digest: "cfg-1".into(),
    }
}

fn run_pair(
    sensor_id: &str,
    model: SensorModel,
    latent_previous: f64,
    latent_current: f64,
    gate: &SensorHealthGate,
) -> TemporalSensorPair {
    let previous = observation(sensor_id, 1_000, model.observe(latent_previous));
    let current = observation(sensor_id, 2_000, model.observe(latent_current));
    let previous_decision = gate.assess(&previous, Some(&previous.configuration_digest), 2_000);
    let current_decision = gate.assess(&current, Some(&current.configuration_digest), 2_000);
    TemporalSensorPair {
        previous,
        previous_decision,
        current,
        current_decision,
        independence: crate::sensor_temporal::SensorIndependenceBinding {
            schema_version: "0.1".into(),
            sensor_id: sensor_id.into(),
            component_id: "wing-root".into(),
            asset_id: "vehicle-1".into(),
            independence_group: format!("group-{sensor_id}"),
            topology_attestation: crate::sensor_topology::SensorTopologyAttestation {
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
                authoritative_reference: crate::sensor_topology::AuthoritativeAttestationReference {
                    attestation_id: "att-topology-1".into(),
                    issuer_id: "mycelix-topology-authority".into(),
                    attestation_digest: "att-digest-1".into(),
                    verification_reference: "verify-1".into(),
                },
            },
        },
    }
}

pub fn run_scenario(scenario: AdversarialScenario) -> AdversarialCrucibleResult {
    let sensor_gate = sensor_gate();
    let temporal_gate = temporal_gate();

    let (latent_previous, latent_current, models) = match scenario {
        AdversarialScenario::StableHealthy => (
            0.2,
            0.2,
            vec![
                ("a", SensorModel { offset: 0.0, stuck: false, drift: 0.0, override_current: None }),
                ("b", SensorModel { offset: 0.0, stuck: false, drift: 0.0, override_current: None }),
                ("c", SensorModel { offset: 0.0, stuck: false, drift: 0.0, override_current: None }),
            ],
        ),
        AdversarialScenario::GenuinePhysicalChange => (
            0.5,
            1.5,
            vec![
                ("a", SensorModel { offset: 0.0, stuck: false, drift: 0.0, override_current: None }),
                ("b", SensorModel { offset: 0.05, stuck: false, drift: 0.0, override_current: None }),
                ("c", SensorModel { offset: -0.05, stuck: false, drift: 0.0, override_current: None }),
            ],
        ),
        AdversarialScenario::StuckSensor => (
            0.5,
            1.5,
            vec![
                ("a", SensorModel { offset: 0.0, stuck: false, drift: 0.0, override_current: None }),
                ("b", SensorModel { offset: 0.05, stuck: false, drift: 0.0, override_current: None }),
                ("c", SensorModel { offset: 0.0, stuck: true, drift: 0.0, override_current: None }),
            ],
        ),
        AdversarialScenario::GradualDrift => (
            0.5,
            1.5,
            vec![
                ("a", SensorModel { offset: 0.0, stuck: false, drift: 0.0, override_current: None }),
                ("b", SensorModel { offset: 0.05, stuck: false, drift: 0.0, override_current: None }),
                ("c", SensorModel { offset: 0.0, stuck: false, drift: -0.45, override_current: None }),
            ],
        ),
        AdversarialScenario::MissingSensor => (
            0.5,
            1.5,
            vec![
                ("a", SensorModel { offset: 0.0, stuck: false, drift: 0.0, override_current: None }),
                ("b", SensorModel { offset: 0.05, stuck: false, drift: 0.0, override_current: None }),
            ],
        ),
        AdversarialScenario::CoordinatedFalseTrajectory => (
            0.5,
            0.8,
            vec![
                ("a", SensorModel { offset: 0.0, stuck: false, drift: 0.0, override_current: Some(1.5) }),
                ("b", SensorModel { offset: 0.05, stuck: false, drift: 0.0, override_current: Some(1.55) }),
                ("c", SensorModel { offset: -0.05, stuck: false, drift: 0.0, override_current: Some(1.45) }),
            ],
        ),
        AdversarialScenario::RecoveryAfterRepair => (
            1.5,
            0.2,
            vec![
                ("a", SensorModel { offset: 0.0, stuck: false, drift: 0.0, override_current: None }),
                ("b", SensorModel { offset: 0.05, stuck: false, drift: 0.0, override_current: None }),
                ("c", SensorModel { offset: -0.05, stuck: false, drift: 0.0, override_current: None }),
            ],
        ),
    };

    let pairs: Vec<_> = models
        .iter()
        .map(|(id, model)| run_pair(id, *model, latent_previous, latent_current, &sensor_gate))
        .collect();

    let decision = temporal_gate.assess(&pairs);
    AdversarialCrucibleResult {
        scenario,
        latent_previous,
        latent_current,
        temporal_state: decision.state,
        trusted_sensor_count: decision.trusted_sensor_ids.len(),
        observed_current_consensus: decision
            .consensus_delta
            .map(|delta| latent_previous + delta),
        temporal_disagreement: !decision.issues.is_empty()
            && decision
                .issues
                .iter()
                .any(|issue| matches!(
                    issue,
                    crate::sensor_temporal::TemporalFusionIssue::TemporalDisagreement
                )),
    }
}

pub fn run_standard_adversarial_crucible() -> Vec<AdversarialCrucibleResult> {
    [
        AdversarialScenario::StableHealthy,
        AdversarialScenario::GenuinePhysicalChange,
        AdversarialScenario::StuckSensor,
        AdversarialScenario::GradualDrift,
        AdversarialScenario::MissingSensor,
        AdversarialScenario::CoordinatedFalseTrajectory,
        AdversarialScenario::RecoveryAfterRepair,
    ]
    .into_iter()
    .map(run_scenario)
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genuine_physical_change_is_not_mistaken_for_sensor_fault() {
        let result = run_scenario(AdversarialScenario::GenuinePhysicalChange);
        assert_eq!(result.temporal_state, TemporalFusionState::Corroborated);
        assert!(!result.temporal_disagreement);
    }

    #[test]
    fn stuck_sensor_is_exposed() {
        let result = run_scenario(AdversarialScenario::StuckSensor);
        assert_eq!(result.temporal_state, TemporalFusionState::Conflicted);
        assert!(result.temporal_disagreement);
    }

    #[test]
    fn gradual_drift_is_exposed_before_absolute_failure() {
        let result = run_scenario(AdversarialScenario::GradualDrift);
        assert_eq!(result.temporal_state, TemporalFusionState::Conflicted);
        assert!(result.temporal_disagreement);
    }

    #[test]
    fn missing_sensor_never_creates_false_quorum() {
        let result = run_scenario(AdversarialScenario::MissingSensor);
        assert_eq!(result.temporal_state, TemporalFusionState::Corroborated);
        assert_eq!(result.trusted_sensor_count, 2);
    }

    #[test]
    fn coordinated_false_trajectory_is_explicitly_not_solved_here() {
        let result = run_scenario(AdversarialScenario::CoordinatedFalseTrajectory);
        assert_eq!(result.temporal_state, TemporalFusionState::Corroborated);
        assert!(!result.temporal_disagreement);
        assert_eq!(result.latent_current, 0.8);
        assert_eq!(result.observed_current_consensus, Some(1.5));
    }

    #[test]
    fn recovery_trajectory_is_sensor_consistent_but_not_a_recovery_verdict() {
        let result = run_scenario(AdversarialScenario::RecoveryAfterRepair);
        assert_eq!(result.temporal_state, TemporalFusionState::Corroborated);
        assert!(!result.temporal_disagreement);
    }
}
