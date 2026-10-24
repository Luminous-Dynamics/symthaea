// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic sensing-layer qualification for regenerative health.
//!
//! Structural health inference is only as trustworthy as the measurements that
//! feed it. This module therefore treats sensor integrity as an explicit,
//! upstream evidence boundary. It does not repair or certify a physical sensor;
//! it determines whether a sensor observation is admissible for downstream
//! structural-health reasoning.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SensorHealthState {
    Trusted,
    Degraded,
    Untrusted,
    Quarantined,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorObservation {
    pub observation_id: String,
    pub sensor_id: String,
    pub timestamp_ms: u64,
    /// Normalized disagreement with the physically admissible sensing manifold.
    pub normalized_residual: f64,
    pub uncertainty: f64,
    pub evidence_id: String,
    pub configuration_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensorHealthIssue {
    EmptyIdentity,
    InvalidResidual,
    InvalidUncertainty,
    MissingEvidence,
    StaleObservation,
    FutureObservation,
    ConfigurationMismatch,
    ResidualTooHigh,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensorHealthPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub degraded_residual_milli: u32,
    pub untrusted_residual_milli: u32,
    pub maximum_observation_age_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorHealthDecision {
    pub schema_version: String,
    pub policy_id: String,
    pub sensor_id: String,
    pub state: SensorHealthState,
    pub issues: Vec<SensorHealthIssue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensorHealthAdmissionError {
    NotTrusted(SensorHealthState),
}

/// Admit a structural-health observation only after the sensing layer has
/// independently qualified the measurement as trusted.
pub fn admit_health_observation(
    sensor: &SensorHealthDecision,
    observation: crate::HealthObservation,
) -> Result<crate::HealthObservation, SensorHealthAdmissionError> {
    if sensor.state != SensorHealthState::Trusted {
        return Err(SensorHealthAdmissionError::NotTrusted(sensor.state));
    }
    Ok(observation)
}

#[derive(Debug, Clone)]
pub struct SensorHealthGate {
    policy: SensorHealthPolicy,
}

impl SensorHealthGate {
    pub fn new(policy: SensorHealthPolicy) -> Result<Self, &'static str> {
        if policy.schema_version.trim().is_empty()
            || policy.policy_id.trim().is_empty()
            || policy.degraded_residual_milli == 0
            || policy.untrusted_residual_milli <= policy.degraded_residual_milli
            || policy.maximum_observation_age_ms == 0
        {
            return Err("invalid sensor health policy");
        }
        Ok(Self { policy })
    }

    pub fn assess(
        &self,
        observation: &SensorObservation,
        expected_configuration_digest: Option<&str>,
        now_ms: u64,
    ) -> SensorHealthDecision {
        let mut issues = Vec::new();

        if observation.observation_id.trim().is_empty()
            || observation.sensor_id.trim().is_empty()
            || observation.configuration_digest.trim().is_empty()
        {
            issues.push(SensorHealthIssue::EmptyIdentity);
        }
        if !observation.normalized_residual.is_finite()
            || observation.normalized_residual < 0.0
        {
            issues.push(SensorHealthIssue::InvalidResidual);
        }
        if !observation.uncertainty.is_finite() || observation.uncertainty <= 0.0 {
            issues.push(SensorHealthIssue::InvalidUncertainty);
        }
        if observation.evidence_id.trim().is_empty() {
            issues.push(SensorHealthIssue::MissingEvidence);
        }
        if observation.timestamp_ms > now_ms {
            issues.push(SensorHealthIssue::FutureObservation);
        } else if now_ms.saturating_sub(observation.timestamp_ms)
            > self.policy.maximum_observation_age_ms
        {
            issues.push(SensorHealthIssue::StaleObservation);
        }
        if let Some(expected) = expected_configuration_digest {
            if expected != observation.configuration_digest {
                issues.push(SensorHealthIssue::ConfigurationMismatch);
            }
        }

        let residual_milli = if observation.normalized_residual.is_finite()
            && observation.normalized_residual >= 0.0
        {
            (observation.normalized_residual * 1000.0).round() as u32
        } else {
            u32::MAX
        };

        let mut state = if !issues.is_empty() {
            SensorHealthState::Quarantined
        } else if residual_milli >= self.policy.untrusted_residual_milli {
            issues.push(SensorHealthIssue::ResidualTooHigh);
            SensorHealthState::Untrusted
        } else if residual_milli >= self.policy.degraded_residual_milli {
            SensorHealthState::Degraded
        } else {
            SensorHealthState::Trusted
        };

        // Invalid evidence must never be softened into a merely degraded sensor.
        if !issues.is_empty()
            && !matches!(
                state,
                SensorHealthState::Quarantined | SensorHealthState::Untrusted
            )
        {
            state = SensorHealthState::Quarantined;
        }

        SensorHealthDecision {
            schema_version: self.policy.schema_version.clone(),
            policy_id: self.policy.policy_id.clone(),
            sensor_id: observation.sensor_id.clone(),
            state,
            issues,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> SensorHealthGate {
        SensorHealthGate::new(SensorHealthPolicy {
            schema_version: "0.1".into(),
            policy_id: "sensor-v1".into(),
            degraded_residual_milli: 2_000,
            untrusted_residual_milli: 4_000,
            maximum_observation_age_ms: 1_000,
        })
        .unwrap()
    }

    fn observation(residual: f64) -> SensorObservation {
        SensorObservation {
            observation_id: "sensor-obs-1".into(),
            sensor_id: "strain-1".into(),
            timestamp_ms: 1_000,
            normalized_residual: residual,
            uncertainty: 1.0,
            evidence_id: "sensor-e-1".into(),
            configuration_digest: "cfg-1".into(),
        }
    }

    #[test]
    fn only_trusted_sensor_evidence_can_be_admitted() {
        let trusted = gate().assess(&observation(0.5), Some("cfg-1"), 1_000);
        let health = crate::HealthObservation {
            observation_id: "health-1".into(),
            component_id: "wing".into(),
            timestamp_ms: 1_000,
            normalized_residual: 0.5,
            uncertainty: 1.0,
            evidence_ids: vec!["health-e-1".into()],
            configuration_digest: "cfg-1".into(),
        };
        assert!(admit_health_observation(&trusted, health.clone()).is_ok());

        let degraded = gate().assess(&observation(2.5), Some("cfg-1"), 1_000);
        assert_eq!(
            admit_health_observation(&degraded, health),
            Err(SensorHealthAdmissionError::NotTrusted(
                SensorHealthState::Degraded
            ))
        );
    }

    #[test]
    fn healthy_sensor_is_trusted() {
        let d = gate().assess(&observation(0.5), Some("cfg-1"), 1_000);
        assert_eq!(d.state, SensorHealthState::Trusted);
        assert!(d.issues.is_empty());
    }

    #[test]
    fn elevated_sensor_residual_is_degraded() {
        let d = gate().assess(&observation(2.5), Some("cfg-1"), 1_000);
        assert_eq!(d.state, SensorHealthState::Degraded);
    }

    #[test]
    fn severe_sensor_residual_is_untrusted() {
        let d = gate().assess(&observation(4.5), Some("cfg-1"), 1_000);
        assert_eq!(d.state, SensorHealthState::Untrusted);
        assert!(d.issues.contains(&SensorHealthIssue::ResidualTooHigh));
    }

    #[test]
    fn stale_sensor_cannot_feed_trusted_health() {
        let d = gate().assess(&observation(0.5), Some("cfg-1"), 2_001);
        assert_eq!(d.state, SensorHealthState::Quarantined);
        assert!(d.issues.contains(&SensorHealthIssue::StaleObservation));
    }

    #[test]
    fn future_sensor_evidence_is_quarantined() {
        let mut o = observation(0.5);
        o.timestamp_ms = 1_001;
        let d = gate().assess(&o, Some("cfg-1"), 1_000);
        assert_eq!(d.state, SensorHealthState::Quarantined);
        assert!(d.issues.contains(&SensorHealthIssue::FutureObservation));
    }

    #[test]
    fn configuration_mismatch_is_quarantined() {
        let d = gate().assess(&observation(0.5), Some("cfg-attacker"), 1_000);
        assert_eq!(d.state, SensorHealthState::Quarantined);
        assert!(d.issues.contains(&SensorHealthIssue::ConfigurationMismatch));
    }

    #[test]
    fn missing_evidence_is_quarantined() {
        let mut o = observation(0.5);
        o.evidence_id.clear();
        let d = gate().assess(&o, Some("cfg-1"), 1_000);
        assert_eq!(d.state, SensorHealthState::Quarantined);
        assert!(d.issues.contains(&SensorHealthIssue::MissingEvidence));
    }
}
