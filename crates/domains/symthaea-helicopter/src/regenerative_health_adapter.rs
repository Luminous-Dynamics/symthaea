// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Projection from the helicopter digital-twin assurance report into the
//! shared regenerative-health evidence contract.
//!
//! This adapter does not certify recovery. It only converts a qualified,
//! complete twin report into a normalized observation that the deterministic
//! regenerative-health gate can evaluate.

use symthaea_regenerative_health::HealthObservation;

use crate::digital_twin_divergence::{
    DigitalTwinDivergenceReport, DigitalTwinDivergenceStatus,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwinHealthProjectionError {
    IncompleteTwinReport,
    NoSignalEvidence,
}

pub fn project_twin_report(
    report: &DigitalTwinDivergenceReport,
    component_id: impl Into<String>,
) -> Result<HealthObservation, TwinHealthProjectionError> {
    if report.status == DigitalTwinDivergenceStatus::Incomplete {
        return Err(TwinHealthProjectionError::IncompleteTwinReport);
    }
    if report.signals.is_empty() {
        return Err(TwinHealthProjectionError::NoSignalEvidence);
    }

    let peak = report
        .signals
        .iter()
        .map(|signal| signal.peak_normalized_residual)
        .fold(0.0_f64, f64::max);

    let evidence_id = report
        .digest_fnv1a64()
        .map_err(|_| TwinHealthProjectionError::NoSignalEvidence)?;

    Ok(HealthObservation {
        observation_id: format!("twin-report:{}", evidence_id),
        component_id: component_id.into(),
        timestamp_ms: report.assessed_at_ms,
        normalized_residual: peak,
        // The residual is already expressed in combined-sigma units.
        // Uncertainty therefore represents the normalized observation scale;
        // it is not a claim that the underlying physical uncertainty is 1σ.
        uncertainty: 1.0,
        evidence_ids: vec![evidence_id],
        configuration_digest: format!(
            "twin-policy:{}:{}",
            report.schema_version, report.policy_id
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::digital_twin_divergence::{
        DigitalTwinDivergenceReport, DigitalTwinDivergenceStatus, TwinSignal,
        TwinSignalDivergence,
    };

    fn report(status: DigitalTwinDivergenceStatus) -> DigitalTwinDivergenceReport {
        DigitalTwinDivergenceReport {
            schema_version: "1".into(),
            policy_id: "twin-v1".into(),
            assessed_at_ms: 42,
            status,
            signals: vec![TwinSignalDivergence {
                signal: TwinSignal::MainRotorSpeed,
                sample_count: 3,
                rms_normalized_residual: 1.5,
                peak_normalized_residual: 2.5,
                final_warning_streak: 0,
                final_unsafe_streak: 0,
                maximum_warning_streak: 0,
                maximum_unsafe_streak: 0,
            }],
            issues: Vec::new(),
        }
    }

    #[test]
    fn complete_report_projects_to_health_observation() {
        let observation = project_twin_report(&report(DigitalTwinDivergenceStatus::Aligned), "rotor")
            .expect("complete report should project");
        assert_eq!(observation.component_id, "rotor");
        assert_eq!(observation.normalized_residual, 2.5);
        assert_eq!(observation.timestamp_ms, 42);
        assert_eq!(observation.evidence_ids.len(), 1);
    }

    #[test]
    fn incomplete_report_cannot_be_promoted_to_health_evidence() {
        assert_eq!(
            project_twin_report(&report(DigitalTwinDivergenceStatus::Incomplete), "rotor"),
            Err(TwinHealthProjectionError::IncompleteTwinReport)
        );
    }
}
