// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fail-closed presentation boundary for legacy shell metrics.
//!
//! INT-002A intentionally does not claim `Live`/`Stale` runtime provenance yet:
//! the current shell IPC protocol does not carry a canonical runtime identity and
//! monotonic `RuntimeCursor`. Until INT-003/INT-004 provide that identity, shell
//! clients may distinguish only between an actually observed service snapshot and
//! unknown/unavailable telemetry.
//!
//! This module prevents transport connectivity, cached defaults, or a locally
//! allocated watch receiver from being mistaken for proof of live cognition.

use super::ipc_client::{ConnectionState, MetricsSnapshot};

/// Presentation-only view over legacy shell metrics.
///
/// This is deliberately narrower than `symthaea-interface-types::RuntimeState`.
/// `Observed` here means only that this process received a concrete metrics
/// snapshot while the legacy IPC connection reports `Connected`. It is explicitly
/// **not** a `StateProvenance::Live` claim because no runtime cursor exists yet.
#[derive(Debug, Clone, Copy)]
pub struct MetricsView<'a> {
    connection_state: ConnectionState,
    observed: Option<&'a MetricsSnapshot>,
}

impl<'a> MetricsView<'a> {
    /// Construct a presentation view from connection state and an optional
    /// actually-observed service snapshot.
    pub fn new(
        connection_state: ConnectionState,
        observed: Option<&'a MetricsSnapshot>,
    ) -> Self {
        Self {
            connection_state,
            observed,
        }
    }

    /// Return the metrics that are safe to render numerically.
    ///
    /// Cached values are intentionally hidden whenever the transport is not in
    /// the fully connected state. Without runtime identity/cursors, this layer
    /// cannot prove that a disconnected/degraded cached value is merely stale
    /// rather than from another runtime instance.
    pub fn metrics(self) -> Option<&'a MetricsSnapshot> {
        if self.connection_state == ConnectionState::Connected {
            self.observed
        } else {
            None
        }
    }

    /// Whether numeric cognitive metrics may be rendered by a legacy shell UI.
    pub fn numeric_rendering_allowed(self) -> bool {
        self.metrics().is_some()
    }

    /// Human-facing presentation label for this transitional contract.
    ///
    /// `OBSERVED / UNPROVENANCED` is intentionally not `LIVE`.
    pub fn label(self) -> &'static str {
        if self.numeric_rendering_allowed() {
            "OBSERVED / UNPROVENANCED"
        } else {
            "UNKNOWN"
        }
    }

    /// The legacy shell metrics view can never make a `Live` provenance claim.
    ///
    /// INT-003/INT-004 must supply runtime identity + monotonic cursor first.
    pub const fn live_claim_allowed(self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> MetricsSnapshot {
        MetricsSnapshot {
            phi: 0.91,
            coherence: 0.87,
            is_conscious: true,
            ..Default::default()
        }
    }

    #[test]
    fn disconnected_cached_snapshot_is_unknown() {
        let snapshot = sample();
        let view = MetricsView::new(ConnectionState::Disconnected, Some(&snapshot));

        assert!(view.metrics().is_none());
        assert!(!view.numeric_rendering_allowed());
        assert_eq!(view.label(), "UNKNOWN");
        assert!(!view.live_claim_allowed());
    }

    #[test]
    fn connected_without_observation_is_unknown() {
        let view = MetricsView::new(ConnectionState::Connected, None);

        assert!(view.metrics().is_none());
        assert!(!view.numeric_rendering_allowed());
        assert_eq!(view.label(), "UNKNOWN");
        assert!(!view.live_claim_allowed());
    }

    #[test]
    fn observed_connected_snapshot_is_renderable_but_not_live() {
        let snapshot = sample();
        let view = MetricsView::new(ConnectionState::Connected, Some(&snapshot));

        assert_eq!(view.metrics().map(|m| m.phi), Some(0.91));
        assert!(view.numeric_rendering_allowed());
        assert_eq!(view.label(), "OBSERVED / UNPROVENANCED");
        assert!(!view.live_claim_allowed());
    }

    #[test]
    fn degraded_snapshot_fails_closed_without_runtime_cursor() {
        let snapshot = sample();
        let view = MetricsView::new(ConnectionState::Degraded, Some(&snapshot));

        assert!(view.metrics().is_none());
        assert_eq!(view.label(), "UNKNOWN");
    }

    #[test]
    fn transport_state_alone_never_implies_metrics() {
        for state in [
            ConnectionState::Disconnected,
            ConnectionState::Connecting,
            ConnectionState::Connected,
            ConnectionState::Degraded,
            ConnectionState::Reconnecting,
        ] {
            let view = MetricsView::new(state, None);
            assert!(view.metrics().is_none());
            assert!(!view.numeric_rendering_allowed());
            assert!(!view.live_claim_allowed());
        }
    }
}
