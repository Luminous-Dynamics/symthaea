// SPDX-License-Identifier: AGPL-3.0-or-later
//! Thin projection from HAL-local runtime/admission evidence into maritime assurance.
//!
//! This module deliberately does not implement fleet rollout policy. The HAL's rollout,
//! quarantine, trusted-time and rollback protocols remain owned by their existing verifier/
//! deployment layer. This adapter consumes the already-decided facts and preserves their
//! failure modes in `symthaea-maritime-core`.

use crate::runtime::HealthStatus;
use symthaea_maritime_core::{
    ComponentHealth, FleetMemberReport, HealthSeverity, OperatingEnvelope, PlatformHealth,
};

/// State of the externally verified HAL fleet-assurance evidence presented to this adapter.
///
/// `Verified` means the caller has already checked the relevant admission evidence according
/// to HAL policy. The other variants remain distinct so missing, stale and mismatched evidence
/// cannot collapse into a generic success/failure bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalAssuranceEvidenceState {
    Verified,
    Missing,
    Stale,
    Mismatched,
}

impl HalAssuranceEvidenceState {
    pub fn is_verified(self) -> bool {
        self == Self::Verified
    }

    fn code(self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::Missing => "missing",
            Self::Stale => "stale",
            Self::Mismatched => "mismatched",
        }
    }
}

/// Minimal facts produced by the HAL fleet-admission layer and consumed by maritime assurance.
///
/// This is not a second admission policy. In particular, this adapter never decides whether a
/// rollout signature, trusted-time quorum, recovery artifact or rollback rehearsal is valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HalFleetAdmissionFacts {
    pub generation: u64,
    pub admitted: bool,
    pub quarantined: bool,
    pub evidence: HalAssuranceEvidenceState,
}

fn component(
    id: &str,
    healthy: bool,
    failure_severity: HealthSeverity,
    failure_code: &str,
    failure_detail: String,
) -> ComponentHealth {
    ComponentHealth {
        component_id: id.into(),
        severity: if healthy {
            HealthSeverity::Healthy
        } else {
            failure_severity
        },
        code: if healthy { "ok".into() } else { failure_code.into() },
        detail: if healthy { String::new() } else { failure_detail },
    }
}

/// Convert HAL-local readiness into explicit maritime component health.
///
/// The mapping is intentionally conservative. E-stop/interlock failures mean the platform is
/// not admissible for goal-directed operation; sensor, actuator-readiness and timing failures
/// are degraded rather than silently healthy. Unverified fleet evidence is represented as an
/// `Unknown` component even when the local runtime itself appears healthy.
pub fn maritime_health_from_hal(
    platform_id: impl Into<String>,
    observed_at_ms: u64,
    status: &HealthStatus,
    evidence: HalAssuranceEvidenceState,
) -> PlatformHealth {
    let platform_id = platform_id.into();
    let mut components = Vec::with_capacity(8);

    components.push(component(
        "hal.sensors",
        status.sensors_ok,
        HealthSeverity::Degraded,
        "sensor_health_failed",
        "one or more registered HAL sensors are unavailable".into(),
    ));
    components.push(component(
        "hal.servos_initialized",
        status.servos_initialized,
        HealthSeverity::Degraded,
        "actuators_not_initialized",
        "HAL actuator output has not completed initialization".into(),
    ));
    components.push(component(
        "hal.servos_enabled",
        status.servos_enabled,
        HealthSeverity::Degraded,
        "actuators_disabled",
        "HAL actuator output is disabled".into(),
    ));
    components.push(component(
        "hal.interlock",
        status.interlock_ok,
        HealthSeverity::Unsafe,
        "interlock_tripped",
        "HAL safety interlock is not healthy".into(),
    ));
    components.push(component(
        "hal.estop",
        !status.estop_active,
        HealthSeverity::Unsafe,
        "estop_active",
        "HAL emergency stop is active".into(),
    ));
    components.push(component(
        "hal.tick_rate",
        status.tick_rate_ok,
        HealthSeverity::Degraded,
        "control_timing_degraded",
        "HAL control-loop rate is outside its readiness band".into(),
    ));
    components.push(component(
        "hal.sensor_monitors",
        status.degraded_count == 0,
        HealthSeverity::Degraded,
        "degraded_monitors",
        format!("{} HAL sensor monitor(s) are degraded", status.degraded_count),
    ));

    components.push(ComponentHealth {
        component_id: "hal.fleet_assurance_evidence".into(),
        severity: if evidence.is_verified() {
            HealthSeverity::Healthy
        } else {
            HealthSeverity::Unknown
        },
        code: evidence.code().into(),
        detail: if evidence.is_verified() {
            String::new()
        } else {
            format!("HAL fleet-assurance evidence is {}", evidence.code())
        },
    });

    if !status.issues.is_empty() {
        components.push(ComponentHealth {
            component_id: "hal.runtime_issues".into(),
            severity: HealthSeverity::Advisory,
            code: "runtime_issues_reported".into(),
            detail: status.issues.join("; "),
        });
    }

    PlatformHealth {
        platform_id,
        observed_at_ms,
        components,
    }
}

/// Project verified HAL admission facts and runtime health into a maritime fleet member.
///
/// Missing/stale/mismatched evidence forces `admitted = false` even if an upstream boolean was
/// accidentally left true. Quarantine and generation are preserved exactly. The operating
/// envelope comes from the platform/domain layer because HAL does not own maritime navigation
/// or communications policy.
pub fn maritime_member_from_hal(
    platform_id: impl Into<String>,
    observed_at_ms: u64,
    facts: HalFleetAdmissionFacts,
    status: &HealthStatus,
    envelope: OperatingEnvelope,
) -> FleetMemberReport {
    let platform_id = platform_id.into();
    FleetMemberReport {
        health: maritime_health_from_hal(
            platform_id.clone(),
            observed_at_ms,
            status,
            facts.evidence,
        ),
        platform_id,
        generation: facts.generation,
        admitted: facts.admitted && facts.evidence.is_verified(),
        quarantined: facts.quarantined,
        envelope,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_maritime_core::{FleetAssuranceReport, HealthSeverity};
    use std::collections::BTreeMap;

    fn ready() -> HealthStatus {
        HealthStatus {
            sensors_ok: true,
            servos_initialized: true,
            servos_enabled: true,
            interlock_ok: true,
            estop_active: false,
            tick_rate_ok: true,
            degraded_count: 0,
            issues: Vec::new(),
        }
    }

    fn verified_facts() -> HalFleetAdmissionFacts {
        HalFleetAdmissionFacts {
            generation: 7,
            admitted: true,
            quarantined: false,
            evidence: HalAssuranceEvidenceState::Verified,
        }
    }

    #[test]
    fn healthy_verified_hal_projects_to_nominal_member() {
        let member = maritime_member_from_hal(
            "auv-1",
            100,
            verified_facts(),
            &ready(),
            OperatingEnvelope::Normal,
        );
        assert!(member.admitted);
        assert!(!member.quarantined);
        assert!(member.health.is_fully_observed_nominal());
    }

    #[test]
    fn missing_assurance_evidence_fails_closed() {
        let mut facts = verified_facts();
        facts.evidence = HalAssuranceEvidenceState::Missing;
        let member = maritime_member_from_hal(
            "auv-1",
            100,
            facts,
            &ready(),
            OperatingEnvelope::Normal,
        );
        assert!(!member.admitted);
        assert_eq!(member.health.worst_severity(), HealthSeverity::Unknown);
    }

    #[test]
    fn active_estop_is_visible_as_locally_unavailable_for_operation() {
        let mut health = ready();
        health.estop_active = true;
        let member = maritime_member_from_hal(
            "auv-1",
            100,
            verified_facts(),
            &health,
            OperatingEnvelope::FailStop,
        );
        assert_eq!(member.health.worst_severity(), HealthSeverity::Unsafe);
    }

    #[test]
    fn fleet_cannot_average_away_hal_evidence_failure() {
        let good = maritime_member_from_hal(
            "a",
            100,
            verified_facts(),
            &ready(),
            OperatingEnvelope::Normal,
        );
        let mut bad_facts = verified_facts();
        bad_facts.evidence = HalAssuranceEvidenceState::Stale;
        let bad = maritime_member_from_hal(
            "b",
            100,
            bad_facts,
            &ready(),
            OperatingEnvelope::Normal,
        );
        let report = FleetAssuranceReport {
            fleet_id: "test-fleet".into(),
            expected_generation: 7,
            members: BTreeMap::from([("a".into(), good), ("b".into(), bad)]),
        };
        assert!(!report.all_members_admitted());
        assert_eq!(report.degraded_members().len(), 1);
    }
}
