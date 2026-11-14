// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::{HealthSeverity, NavigationQuality, PlatformHealth};
use serde::{Deserialize, Serialize};

/// Coarse operating envelope used when dependencies disappear or become untrusted.
///
/// Declaration order is also restriction order: larger values are never broader.
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
///
/// Independent degradations are composed by taking the most restrictive envelope.
/// This is load-bearing: an earlier, less severe fault must never mask a later,
/// more severe loss (for example Unknown health must not hide unavailable navigation).
pub fn baseline_envelope(
    dependencies: DependencyState,
    navigation: NavigationQuality,
    health: &PlatformHealth,
) -> OperatingEnvelope {
    let mut envelope = match health.worst_severity() {
        HealthSeverity::Unsafe => OperatingEnvelope::FailStop,
        HealthSeverity::Critical => OperatingEnvelope::RecoverOrSurface,
        HealthSeverity::Degraded => OperatingEnvelope::SafeTransit,
        HealthSeverity::Unknown => OperatingEnvelope::ReducedCapability,
        HealthSeverity::Healthy | HealthSeverity::Advisory => OperatingEnvelope::Normal,
    };

    if navigation == NavigationQuality::Unavailable {
        envelope = envelope.max(OperatingEnvelope::HoldOrLoiter);
    }

    if !dependencies.trusted_time_available {
        envelope = envelope.max(OperatingEnvelope::SafeTransit);
    }

    if !dependencies.fleet_link_available || !dependencies.remote_operator_available {
        envelope = envelope.max(OperatingEnvelope::ReducedCapability);
    }

    envelope
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

    fn nominal_dependencies() -> DependencyState {
        DependencyState {
            fleet_link_available: true,
            remote_operator_available: true,
            trusted_time_available: true,
            external_positioning_available: true,
        }
    }

    #[test]
    fn fleet_partition_degrades_without_forcing_failure() {
        let mut dependencies = nominal_dependencies();
        dependencies.fleet_link_available = false;
        let envelope = baseline_envelope(
            dependencies,
            NavigationQuality::Nominal,
            &health(HealthSeverity::Healthy),
        );
        assert_eq!(envelope, OperatingEnvelope::ReducedCapability);
    }

    #[test]
    fn unsafe_local_health_fail_stops_even_with_perfect_connectivity() {
        let envelope = baseline_envelope(
            nominal_dependencies(),
            NavigationQuality::Nominal,
            &health(HealthSeverity::Unsafe),
        );
        assert_eq!(envelope, OperatingEnvelope::FailStop);
    }

    #[test]
    fn missing_health_evidence_never_returns_normal() {
        let envelope = baseline_envelope(
            nominal_dependencies(),
            NavigationQuality::Nominal,
            &PlatformHealth {
                platform_id: "node".into(),
                observed_at_ms: 1,
                components: Vec::new(),
            },
        );
        assert_eq!(envelope, OperatingEnvelope::ReducedCapability);
    }

    #[test]
    fn combined_failures_choose_the_most_restrictive_envelope() {
        let mut dependencies = nominal_dependencies();
        dependencies.fleet_link_available = false;
        dependencies.trusted_time_available = false;

        // Unknown health alone is only ReducedCapability, but unavailable
        // navigation must still tighten the result to HoldOrLoiter.
        let unknown_health = PlatformHealth {
            platform_id: "node".into(),
            observed_at_ms: 1,
            components: Vec::new(),
        };
        assert_eq!(
            baseline_envelope(
                dependencies,
                NavigationQuality::Unavailable,
                &unknown_health,
            ),
            OperatingEnvelope::HoldOrLoiter
        );

        // Critical local health remains more restrictive than the same
        // dependency/navigation losses.
        assert_eq!(
            baseline_envelope(
                dependencies,
                NavigationQuality::Unavailable,
                &health(HealthSeverity::Critical),
            ),
            OperatingEnvelope::RecoverOrSurface
        );
    }
}
