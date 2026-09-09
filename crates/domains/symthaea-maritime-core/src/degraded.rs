// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::{HealthSeverity, NavigationQuality, PlatformHealth};
use serde::{Deserialize, Serialize};

/// Coarse operating envelope used when dependencies disappear or become untrusted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OperatingEnvelope {
    Normal,
    ReducedCapability,
    SafeTransit,
    HoldOrLoiter,
    RecoverOrSurface,
    FailStop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DependencyState {
    pub fleet_link_available: bool,
    pub remote_operator_available: bool,
    pub trusted_time_available: bool,
    /// Informational dependency flag; `navigation` remains the decisive quality signal
    /// because underwater platforms normally operate without continuous external fixes.
    pub external_positioning_available: bool,
}

/// Deterministic, conservative baseline policy. Platform-specific controllers may
/// further restrict the returned envelope, but should not silently broaden it.
pub fn baseline_envelope(
    dependencies: DependencyState,
    navigation: NavigationQuality,
    health: &PlatformHealth,
) -> OperatingEnvelope {
    match health.worst_severity() {
        HealthSeverity::Unsafe => return OperatingEnvelope::FailStop,
        HealthSeverity::Critical => return OperatingEnvelope::RecoverOrSurface,
        HealthSeverity::Degraded => return OperatingEnvelope::SafeTransit,
        HealthSeverity::Unknown => return OperatingEnvelope::ReducedCapability,
        HealthSeverity::Healthy | HealthSeverity::Advisory => {}
    }

    if navigation == NavigationQuality::Unavailable {
        return OperatingEnvelope::HoldOrLoiter;
    }

    if !dependencies.trusted_time_available {
        return OperatingEnvelope::SafeTransit;
    }

    if !dependencies.fleet_link_available || !dependencies.remote_operator_available {
        return OperatingEnvelope::ReducedCapability;
    }

    OperatingEnvelope::Normal
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComponentHealth, PlatformHealth};

    fn health(severity: HealthSeverity) -> PlatformHealth {
        PlatformHealth {
            platform_id: "node".into(),
            observed_at_ms: 1,
            components: vec![ComponentHealth {
                component_id: "core".into(),
                severity,
                code: "test".into(),
                detail: String::new(),
            }],
        }
    }

    #[test]
    fn fleet_partition_degrades_without_forcing_failure() {
        let envelope = baseline_envelope(
            DependencyState {
                fleet_link_available: false,
                remote_operator_available: true,
                trusted_time_available: true,
                external_positioning_available: true,
            },
            NavigationQuality::Nominal,
            &health(HealthSeverity::Healthy),
        );
        assert_eq!(envelope, OperatingEnvelope::ReducedCapability);
    }

    #[test]
    fn unsafe_local_health_fail_stops_even_with_perfect_connectivity() {
        let envelope = baseline_envelope(
            DependencyState {
                fleet_link_available: true,
                remote_operator_available: true,
                trusted_time_available: true,
                external_positioning_available: true,
            },
            NavigationQuality::Nominal,
            &health(HealthSeverity::Unsafe),
        );
        assert_eq!(envelope, OperatingEnvelope::FailStop);
    }

    #[test]
    fn missing_health_evidence_never_returns_normal() {
        let envelope = baseline_envelope(
            DependencyState {
                fleet_link_available: true,
                remote_operator_available: true,
                trusted_time_available: true,
                external_positioning_available: true,
            },
            NavigationQuality::Nominal,
            &PlatformHealth {
                platform_id: "node".into(),
                observed_at_ms: 1,
                components: Vec::new(),
            },
        );
        assert_eq!(envelope, OperatingEnvelope::ReducedCapability);
    }
}
