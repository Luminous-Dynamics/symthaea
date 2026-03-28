// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Cross-Planetary Federated Learning: latency-aware model aggregation.
//!
//! Extends Mycelix's federated learning (0TML) with interplanetary constraints.
//! Communication latency means model updates between Earth and outer system
//! colonies arrive minutes to hours late. This module defines latency-weighted
//! aggregation strategies that account for stale gradients.
//!
//! # Physical Constraints
//!
//! - Earth ↔ Moon: 1.28s (negligible, standard FL works)
//! - Earth ↔ Mars: 4-22 min (async aggregation, epoch-based)
//! - Earth ↔ Europa: 33-54 min (deeply async, local models diverge)
//! - Earth ↔ Titan: 67-90 min (near-autonomous, periodic sync)
//! - Solar conjunction: 2-5 week blackout per synodic period (no sync at all)

use crate::earth_colony_protocol::PlanetaryBody;
use serde::{Deserialize, Serialize};

// ============================================================================
// Latency-Weighted Aggregation
// ============================================================================

/// Strategy for handling stale model updates from distant colonies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationStrategy {
    /// Standard FedAvg — all updates weighted equally. Only viable for <10s latency.
    Synchronous,
    /// Async FedAvg — updates weighted by freshness. For 1-30 min latency.
    AsyncWeighted,
    /// Epoch-based sync — aggregate every N ticks. For 30-90 min latency.
    EpochBased { ticks_per_epoch: u32 },
    /// Autonomous local — minimal sync, local models dominate. For >90 min or blackout.
    Autonomous,
}

impl AggregationStrategy {
    /// Select strategy based on communication latency.
    pub fn for_latency(one_way_delay_secs: f64) -> Self {
        if one_way_delay_secs < 10.0 {
            Self::Synchronous
        } else if one_way_delay_secs < 1800.0 { // <30 min
            Self::AsyncWeighted
        } else if one_way_delay_secs < 5400.0 { // <90 min
            // Sync every 12 ticks (1 year) for deep space
            Self::EpochBased { ticks_per_epoch: 12 }
        } else {
            Self::Autonomous
        }
    }

    /// Select strategy for a given planetary body.
    pub fn for_body(body: PlanetaryBody) -> Self {
        Self::for_latency(body.light_delay_to_earth_secs())
    }
}

/// Latency weighting for a model update from a specific colony.
///
/// Stale gradients from distant colonies are down-weighted to prevent
/// them from corrupting the global model. The staleness penalty follows
/// an exponential decay: `weight = base_weight * exp(-staleness / tau)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyWeighting {
    /// Source colony.
    pub source: PlanetaryBody,
    /// Base weight (from consciousness-gated FL, 0.0-1.0).
    pub base_weight: f64,
    /// Staleness in ticks since this update was computed at source.
    pub staleness_ticks: u32,
    /// Decay constant (higher = more tolerant of staleness).
    pub tau: f64,
    /// Effective weight after latency penalty.
    pub effective_weight: f64,
}

impl LatencyWeighting {
    /// Compute effective weight given staleness.
    pub fn compute(source: PlanetaryBody, base_weight: f64, staleness_ticks: u32) -> Self {
        // Tau scales with distance: nearby colonies are penalized more for staleness
        // because we expect fresher updates. Distant colonies get more slack.
        let tau = match source {
            PlanetaryBody::Moon => 1.0,    // Expect fresh updates
            PlanetaryBody::Mars => 3.0,    // 26-month windows
            PlanetaryBody::Europa => 6.0,  // Deeply async
            PlanetaryBody::Titan => 12.0,  // Near-autonomous
            _ => 2.0,
        };
        let effective_weight = base_weight * (-1.0 * staleness_ticks as f64 / tau).exp();
        Self { source, base_weight, staleness_ticks, tau, effective_weight }
    }
}

/// Cross-planetary model checkpoint.
///
/// Periodically synced between colonies during transfer windows.
/// Encodes the local model state plus metadata for staleness tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossPlanetaryCheckpoint {
    /// Source colony.
    pub source: PlanetaryBody,
    /// Tick when this checkpoint was created at source.
    pub source_tick: u32,
    /// Tick when this checkpoint arrives at destination.
    pub arrival_tick: u32,
    /// Model version (monotonically increasing per source).
    pub model_version: u64,
    /// Number of local training samples since last sync.
    pub local_samples: u64,
    /// Consciousness-gated quality score [0.0, 1.0].
    /// Higher-consciousness colonies produce more trustworthy models.
    pub quality_score: f64,
    /// Serialized model weights (opaque bytes for this layer).
    pub weights_hash: [u8; 32],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregation_strategy_selection() {
        assert_eq!(AggregationStrategy::for_body(PlanetaryBody::Moon), AggregationStrategy::Synchronous);
        assert_eq!(AggregationStrategy::for_body(PlanetaryBody::Mars), AggregationStrategy::AsyncWeighted);
        assert!(matches!(AggregationStrategy::for_body(PlanetaryBody::Europa),
            AggregationStrategy::EpochBased { .. }));
        // Titan at 4758s mean delay is in EpochBased range (<5400s threshold)
        assert!(matches!(AggregationStrategy::for_body(PlanetaryBody::Titan),
            AggregationStrategy::EpochBased { .. }));
    }

    #[test]
    fn test_staleness_penalty() {
        let fresh = LatencyWeighting::compute(PlanetaryBody::Mars, 1.0, 0);
        let stale = LatencyWeighting::compute(PlanetaryBody::Mars, 1.0, 6);
        assert!(fresh.effective_weight > stale.effective_weight);
        // Titan is more tolerant of staleness (higher tau)
        let titan_stale = LatencyWeighting::compute(PlanetaryBody::Titan, 1.0, 6);
        assert!(titan_stale.effective_weight > stale.effective_weight);
    }

    #[test]
    fn test_fresh_update_full_weight() {
        let w = LatencyWeighting::compute(PlanetaryBody::Earth, 0.8, 0);
        assert!((w.effective_weight - 0.8).abs() < 0.001);
    }
}
