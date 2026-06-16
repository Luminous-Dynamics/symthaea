// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Topological handshake for evaluating peer compatibility.
//!
//! After trust verification, peers exchange capability cards and compute
//! compatibility scores based on substrate, features, and Phi proximity.

use super::capability_card::CapabilityCard;

/// Result of a topological handshake evaluation.
#[derive(Debug, Clone)]
pub struct TopologicalHandshakeResult {
    pub substrate_compat: f64,
    pub feature_overlap: f64,
    pub phi_compat: f64,
    pub total_score: f64,
    pub approved: bool,
}

/// Configuration for handshake evaluation weights.
#[derive(Debug, Clone)]
pub struct HandshakeConfig {
    pub substrate_weight: f64,
    pub feature_weight: f64,
    pub phi_weight: f64,
    pub approval_threshold: f64,
}

impl Default for HandshakeConfig {
    fn default() -> Self {
        Self {
            substrate_weight: 0.4,
            feature_weight: 0.3,
            phi_weight: 0.3,
            approval_threshold: 0.3,
        }
    }
}

/// Evaluate compatibility between two capability cards.
///
/// Returns a result with substrate, feature, and Phi compatibility scores.
/// If either card fails hash verification, returns all zeros (not approved).
pub fn evaluate_compatibility(
    local: &CapabilityCard,
    remote: &CapabilityCard,
    config: &HandshakeConfig,
) -> TopologicalHandshakeResult {
    // Reject cards with bad hashes
    if !local.verify_hash() || !remote.verify_hash() {
        return TopologicalHandshakeResult {
            substrate_compat: 0.0,
            feature_overlap: 0.0,
            phi_compat: 0.0,
            total_score: 0.0,
            approved: false,
        };
    }

    // Substrate compatibility: 1.0 if same type, 0.5 otherwise
    let substrate_compat = if local.substrate_type == remote.substrate_type {
        1.0
    } else {
        0.5
    };

    // Feature overlap: Jaccard similarity
    let feature_overlap = if local.features.is_empty() && remote.features.is_empty() {
        1.0
    } else {
        let local_set: std::collections::HashSet<&str> =
            local.features.iter().map(|s| s.as_str()).collect();
        let remote_set: std::collections::HashSet<&str> =
            remote.features.iter().map(|s| s.as_str()).collect();
        let intersection = local_set.intersection(&remote_set).count() as f64;
        let union = local_set.union(&remote_set).count() as f64;
        if union > 0.0 {
            intersection / union
        } else {
            1.0
        }
    };

    // Phi compatibility: 1.0 when equal, decays with distance
    let phi_diff = (local.phi - remote.phi).abs();
    let phi_compat = (-phi_diff * 2.0).exp(); // e^(-2|Δφ|)

    let total_score = config.substrate_weight * substrate_compat
        + config.feature_weight * feature_overlap
        + config.phi_weight * phi_compat;

    TopologicalHandshakeResult {
        substrate_compat,
        feature_overlap,
        phi_compat,
        total_score,
        approved: total_score >= config.approval_threshold,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_loop::config::CognitiveLoopConfig;
    use crate::swarm::capability_card::CardStats;
    use crate::swarm::holochain::AgentPubKey;

    fn make_card(features: Vec<String>, phi: f64) -> CapabilityCard {
        let config = CognitiveLoopConfig::default();
        let stats = CardStats {
            generated_at: 1000,
            substrate_feasibility: 0.71,
            cycle_hz: 234.0,
            phi,
            features,
            physics_domains: vec![],
        };
        CapabilityCard::from_config(AgentPubKey::test_key(1), &config, &stats)
    }

    #[test]
    fn test_identical_configs_high_compat() {
        let card = make_card(vec!["a".into(), "b".into()], 0.8);
        let result = evaluate_compatibility(&card, &card, &HandshakeConfig::default());
        assert!(
            result.total_score > 0.9,
            "Identical cards should have high compat: {}",
            result.total_score
        );
        assert!(result.approved);
    }

    #[test]
    fn test_different_configs_still_positive() {
        let a = make_card(vec!["a".into()], 0.8);
        let b = make_card(vec!["b".into()], 0.3);
        let result = evaluate_compatibility(&a, &b, &HandshakeConfig::default());
        assert!(
            result.total_score > 0.0,
            "Different configs should still be positive"
        );
    }

    #[test]
    fn test_bad_hash_fails() {
        let a = make_card(vec![], 0.8);
        let mut b = make_card(vec![], 0.8);
        b.phi = 0.1; // tamper
        let result = evaluate_compatibility(&a, &b, &HandshakeConfig::default());
        assert_eq!(result.total_score, 0.0);
        assert!(!result.approved);
    }

    #[test]
    fn test_feature_jaccard() {
        let a = make_card(vec!["x".into(), "y".into(), "z".into()], 0.8);
        let b = make_card(vec!["y".into(), "z".into(), "w".into()], 0.8);
        let result = evaluate_compatibility(&a, &b, &HandshakeConfig::default());
        // Jaccard of {x,y,z} ∩ {y,z,w} = 2/4 = 0.5
        assert!((result.feature_overlap - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_phi_compatibility_close_values() {
        let a = make_card(vec![], 0.80);
        let b = make_card(vec![], 0.82);
        let result = evaluate_compatibility(&a, &b, &HandshakeConfig::default());
        assert!(
            result.phi_compat > 0.9,
            "Close phi should have high compat: {}",
            result.phi_compat
        );
    }
}
