// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Federated weight exchange for the Liquid-Mamba HDC↔SSM projection.
//!
//! Bridges `HdcSsmProjection` (symthaea-broca) with the existing
//! `FederatedAggregator` / `GradientMessage` protocol so projection
//! weights can be shared and averaged across the swarm.
//!
//! Three helper functions cover the full lifecycle:
//!
//! 1. **Export** — `projection_to_gradient_msg` flattens weights into a `GradientMessage`.
//! 2. **Import** — `apply_aggregated_weights` loads aggregated weights back.
//! 3. **Init**   — `create_projection_aggregator` creates an aggregator seeded
//!    with the projection's current weights.

use super::federated_cfc::{FederatedAggregator, GradientMessage};
use symthaea_broca::HdcSsmProjection;

/// Package a projection's current weights into a `GradientMessage` for swarm exchange.
///
/// # Arguments
/// * `proj` — the local projection whose weights to export
/// * `source_id` — 32-byte node identity
/// * `trust` — Holochain trust score (0.0–1.0)
/// * `version` — model version for stale-gradient detection
pub fn projection_to_gradient_msg(
    proj: &HdcSsmProjection,
    source_id: [u8; 32],
    trust: f32,
    version: u64,
) -> GradientMessage {
    GradientMessage::new(source_id, proj.flatten_weights(), trust).with_model_version(version)
}

/// Apply aggregated weights to a projection after federated averaging.
///
/// Returns `true` if the weights were applied, `false` if the dimension
/// doesn't match (indicating a version mismatch or corrupted message).
pub fn apply_aggregated_weights(proj: &mut HdcSsmProjection, aggregated: &[f32]) -> bool {
    if aggregated.len() != proj.num_params() {
        return false;
    }
    proj.load_weights(aggregated);
    true
}

/// Create a `FederatedAggregator` seeded with a projection's current weights.
///
/// The returned aggregator is ready to receive peer gradients and aggregate.
pub fn create_projection_aggregator(proj: &HdcSsmProjection) -> FederatedAggregator {
    FederatedAggregator::new(proj.flatten_weights())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::genesis::GenesisSeed;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-projection-federated")
    }

    fn small_projection() -> HdcSsmProjection {
        // Small dims for fast tests (4D HDC, 2D bottleneck, 3D SSM)
        HdcSsmProjection::new(&test_genesis(), 4, 2, 3)
    }

    #[test]
    fn test_roundtrip_flatten_aggregate_apply() {
        let proj = small_projection();
        let source_id = [42u8; 32];

        // Export weights as a gradient message
        let msg = projection_to_gradient_msg(&proj, source_id, 0.9, 1);
        assert_eq!(msg.gradient_data.len(), proj.num_params());
        assert_eq!(msg.source_id, source_id);
        assert!((msg.trust_score - 0.9).abs() < 1e-6);
        assert_eq!(msg.model_version, 1);

        // Create aggregator, feed single gradient, aggregate
        let mut aggregator = create_projection_aggregator(&proj);
        assert!(aggregator.receive_gradient(msg));
        let aggregated = aggregator
            .aggregate()
            .expect("should aggregate with 1 gradient");

        // Apply back — should succeed with matching dimensions
        let mut proj2 = small_projection();
        assert!(apply_aggregated_weights(&mut proj2, &aggregated));
    }

    #[test]
    fn test_dimension_mismatch_rejected() {
        let mut proj = small_projection();
        let wrong_weights = vec![0.0; proj.num_params() + 1]; // Wrong size
        assert!(!apply_aggregated_weights(&mut proj, &wrong_weights));
    }

    #[test]
    fn test_multi_peer_trust_weighted_averaging() {
        let genesis = test_genesis();
        let proj = HdcSsmProjection::new(&genesis, 4, 2, 3);
        let mut aggregator = create_projection_aggregator(&proj);

        // Two peers with different trust scores and slightly different weights
        let weights_a: Vec<f32> = proj.flatten_weights().iter().map(|w| w + 0.1).collect();
        let weights_b: Vec<f32> = proj.flatten_weights().iter().map(|w| w - 0.1).collect();

        let msg_a = GradientMessage::new([1u8; 32], weights_a.clone(), 0.9);
        let msg_b = GradientMessage::new([2u8; 32], weights_b.clone(), 0.1);

        assert!(aggregator.receive_gradient(msg_a));
        assert!(aggregator.receive_gradient(msg_b));

        let aggregated = aggregator
            .aggregate()
            .expect("should aggregate with 2 gradients");
        assert_eq!(aggregated.len(), proj.num_params());

        // All values should be finite
        assert!(aggregated.iter().all(|v| v.is_finite()));

        // Higher-trust peer should pull the result closer to its weights
        // (trust-weighted averaging: 0.9 * weights_a + 0.1 * weights_b)
        // The exact outcome depends on FederatedAggregator's weighting
        // — just verify it's between the two extremes
        for (i, &v) in aggregated.iter().enumerate() {
            let lo = weights_a[i].min(weights_b[i]);
            let hi = weights_a[i].max(weights_b[i]);
            assert!(
                v >= lo - 0.01 && v <= hi + 0.01,
                "aggregated[{i}] = {v} not in [{lo}, {hi}]"
            );
        }
    }
}
