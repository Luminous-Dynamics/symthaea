// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Transitional provenance contract for shell cognitive telemetry.
//!
//! The legacy IPC protocol can deliver real service observations, but it does not
//! yet carry the runtime identity + monotonic cursor required by the unified
//! interaction-runtime architecture to call a value `Live`. This module makes that
//! limitation explicit and, crucially, makes "no observation" unrepresentable as
//! a plausible numeric default.

use super::ipc_client::MetricsSnapshot;

/// Provenance of telemetry visible to legacy shell surfaces.
///
/// `LegacyObserved` means the value came from the connected service, but lacks the
/// runtime identity/cursor needed for canonical `Live` provenance. A disconnected
/// client may retain the last value only as `StaleLegacyObserved`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShellTelemetryProvenance {
    /// No telemetry observation is available.
    #[default]
    Unknown,
    /// Observed from protocol-v1 service traffic, without canonical runtime cursor.
    LegacyObserved,
    /// Previously observed from protocol-v1 service traffic, but connection is gone.
    StaleLegacyObserved,
    /// Deliberately synthesized locally for demos/tests; never service-observed.
    Simulation,
}

impl ShellTelemetryProvenance {
    /// Human-facing label intended for badges/status lines.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Unknown => "UNKNOWN",
            Self::LegacyObserved => "OBSERVED (LEGACY)",
            Self::StaleLegacyObserved => "STALE (LEGACY)",
            Self::Simulation => "SIMULATION",
        }
    }

    /// Whether the value originated in actual service traffic.
    pub const fn is_service_observed(self) -> bool {
        matches!(self, Self::LegacyObserved | Self::StaleLegacyObserved)
    }

    /// Canonical live provenance is intentionally impossible at this layer.
    ///
    /// Protocol v1 has no authoritative `RuntimeId + EventSeq/RuntimeCursor`, so
    /// callers must not promote these values to the interaction runtime's `Live`
    /// state merely because a Unix socket is connected.
    pub const fn is_canonical_live(self) -> bool {
        false
    }
}

/// One shell-visible telemetry state with provenance coupled to value presence.
///
/// `Unknown` always contains no metrics. Every state carrying metrics has an
/// explicit provenance class, preventing a `MetricsSnapshot::default()` from being
/// silently interpreted as an observed cognitive state.
#[derive(Debug, Clone, PartialEq)]
pub enum ShellTelemetryState {
    Unknown,
    LegacyObserved(MetricsSnapshot),
    StaleLegacyObserved(MetricsSnapshot),
    Simulation(MetricsSnapshot),
}

impl Default for ShellTelemetryState {
    fn default() -> Self {
        Self::Unknown
    }
}

impl ShellTelemetryState {
    /// Construct an unknown state. No numeric telemetry exists in this variant.
    pub const fn unknown() -> Self {
        Self::Unknown
    }

    /// Record telemetry received from the legacy protocol-v1 service.
    pub fn observe_legacy(metrics: MetricsSnapshot) -> Self {
        Self::LegacyObserved(metrics)
    }

    /// Record deliberately simulated telemetry.
    pub fn simulation(metrics: MetricsSnapshot) -> Self {
        Self::Simulation(metrics)
    }

    /// Current provenance classification.
    pub const fn provenance(&self) -> ShellTelemetryProvenance {
        match self {
            Self::Unknown => ShellTelemetryProvenance::Unknown,
            Self::LegacyObserved(_) => ShellTelemetryProvenance::LegacyObserved,
            Self::StaleLegacyObserved(_) => ShellTelemetryProvenance::StaleLegacyObserved,
            Self::Simulation(_) => ShellTelemetryProvenance::Simulation,
        }
    }

    /// Borrow the telemetry value when one actually exists.
    pub const fn metrics(&self) -> Option<&MetricsSnapshot> {
        match self {
            Self::Unknown => None,
            Self::LegacyObserved(metrics)
            | Self::StaleLegacyObserved(metrics)
            | Self::Simulation(metrics) => Some(metrics),
        }
    }

    /// Whether any numeric telemetry is present.
    pub const fn has_metrics(&self) -> bool {
        self.metrics().is_some()
    }

    /// Transition on transport disconnect.
    ///
    /// A real legacy observation is retained only as stale. Simulation remains
    /// simulation because its provenance is independent of transport. Unknown
    /// remains unknown.
    pub fn disconnected(self) -> Self {
        match self {
            Self::LegacyObserved(metrics) => Self::StaleLegacyObserved(metrics),
            Self::StaleLegacyObserved(metrics) => Self::StaleLegacyObserved(metrics),
            Self::Simulation(metrics) => Self::Simulation(metrics),
            Self::Unknown => Self::Unknown,
        }
    }

    /// Remove any retained telemetry and return to unknown.
    pub const fn clear(self) -> Self {
        Self::Unknown
    }

    /// Legacy helper for consumers that need the three headline values.
    ///
    /// Returns `None` for unknown instead of inventing `(phi, coherence, conscious)`.
    pub fn status_tuple(&self) -> Option<(f64, f64, bool)> {
        self.metrics()
            .map(|m| (m.phi, m.coherence, m.is_conscious))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(phi: f64) -> MetricsSnapshot {
        MetricsSnapshot {
            phi,
            coherence: 0.82,
            is_conscious: phi > 0.3,
            consciousness_level: phi,
            ..Default::default()
        }
    }

    #[test]
    fn default_state_contains_no_fabricated_metrics() {
        let state = ShellTelemetryState::default();
        assert_eq!(state.provenance(), ShellTelemetryProvenance::Unknown);
        assert!(!state.has_metrics());
        assert!(state.metrics().is_none());
        assert!(state.status_tuple().is_none());
    }

    #[test]
    fn legacy_observation_is_explicitly_not_canonical_live() {
        let state = ShellTelemetryState::observe_legacy(metrics(0.74));
        assert_eq!(
            state.provenance(),
            ShellTelemetryProvenance::LegacyObserved
        );
        assert!(state.provenance().is_service_observed());
        assert!(!state.provenance().is_canonical_live());
        assert_eq!(state.provenance().label(), "OBSERVED (LEGACY)");
        assert_eq!(state.status_tuple(), Some((0.74, 0.82, true)));
    }

    #[test]
    fn disconnect_demotes_observed_data_to_stale_without_changing_value() {
        let state = ShellTelemetryState::observe_legacy(metrics(0.68)).disconnected();
        assert_eq!(
            state.provenance(),
            ShellTelemetryProvenance::StaleLegacyObserved
        );
        assert_eq!(state.status_tuple(), Some((0.68, 0.82, true)));
        assert!(!state.provenance().is_canonical_live());
    }

    #[test]
    fn unknown_disconnect_stays_unknown() {
        let state = ShellTelemetryState::unknown().disconnected();
        assert_eq!(state, ShellTelemetryState::Unknown);
        assert!(state.metrics().is_none());
    }

    #[test]
    fn simulation_is_never_confused_with_service_observation() {
        let state = ShellTelemetryState::simulation(metrics(0.91));
        assert_eq!(state.provenance(), ShellTelemetryProvenance::Simulation);
        assert!(!state.provenance().is_service_observed());
        assert!(!state.provenance().is_canonical_live());
        assert_eq!(state.provenance().label(), "SIMULATION");

        let disconnected = state.disconnected();
        assert_eq!(
            disconnected.provenance(),
            ShellTelemetryProvenance::Simulation
        );
    }

    #[test]
    fn clear_removes_retained_numeric_state() {
        let state = ShellTelemetryState::observe_legacy(metrics(0.55))
            .disconnected()
            .clear();
        assert_eq!(state, ShellTelemetryState::Unknown);
        assert!(state.status_tuple().is_none());
    }
}
