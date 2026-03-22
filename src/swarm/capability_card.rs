// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Capability card for swarm peer discovery.
//!
//! A BLAKE3-hashed self-description of this node's capabilities,
//! used for peer discovery, reputation tracking, and compatibility evaluation.

use serde::{Deserialize, Serialize};

use super::holochain::AgentPubKey;
use crate::cognitive_loop::config::CognitiveLoopConfig;

/// Runtime snapshot stats for building a capability card.
#[derive(Debug, Clone, Default)]
pub struct CardStats {
    pub generated_at: u64,
    pub substrate_feasibility: f64,
    pub cycle_hz: f32,
    pub phi: f64,
    pub features: Vec<String>,
    pub physics_domains: Vec<String>,
}

/// A BLAKE3-hashed capability card describing a Symthaea node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityCard {
    pub format_version: u8,
    pub agent_key: AgentPubKey,
    pub generated_at: u64,
    pub substrate_type: String,
    pub substrate_feasibility: f64,
    pub cycle_hz: f32,
    pub phi: f64,
    pub features: Vec<String>,
    pub physics_domains: Vec<String>,
    pub hdc_dimension: usize,
    pub cfc_neurons: usize,
    pub temporal_backend: String,
    pub card_hash: [u8; 32],
}

impl CapabilityCard {
    /// Build a card from config + runtime stats, sealed with BLAKE3.
    pub fn from_config(
        agent_key: AgentPubKey,
        config: &CognitiveLoopConfig,
        stats: &CardStats,
    ) -> Self {
        let mut card = Self {
            format_version: 1,
            agent_key,
            generated_at: stats.generated_at,
            substrate_type: format!("{:?}", config.substrate_type.canonical()),
            substrate_feasibility: stats.substrate_feasibility,
            cycle_hz: stats.cycle_hz,
            phi: stats.phi,
            features: stats.features.clone(),
            physics_domains: stats.physics_domains.clone(),
            hdc_dimension: config.encoder_config.dimension,
            cfc_neurons: config.cfc_config.num_neurons,
            temporal_backend: format!("{:?}", config.temporal_backend),
            card_hash: [0u8; 32],
        };
        card.card_hash = card.compute_hash();
        card
    }

    /// Verify card integrity.
    pub fn verify_hash(&self) -> bool {
        let expected = self.compute_hash();
        self.card_hash == expected
    }

    fn compute_hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&[self.format_version]);
        hasher.update(self.agent_key.as_str().as_bytes());
        hasher.update(&self.generated_at.to_le_bytes());
        hasher.update(self.substrate_type.as_bytes());
        hasher.update(&self.substrate_feasibility.to_le_bytes());
        hasher.update(&self.cycle_hz.to_le_bytes());
        hasher.update(&self.phi.to_le_bytes());
        hasher.update(&self.hdc_dimension.to_le_bytes());
        hasher.update(&self.cfc_neurons.to_le_bytes());
        hasher.update(self.temporal_backend.as_bytes());
        for f in &self.features {
            hasher.update(f.as_bytes());
        }
        for d in &self.physics_domains {
            hasher.update(d.as_bytes());
        }
        *hasher.finalize().as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_card() -> CapabilityCard {
        let config = CognitiveLoopConfig::default();
        let stats = CardStats {
            generated_at: 1000,
            substrate_feasibility: 0.71,
            cycle_hz: 234.0,
            phi: 0.85,
            features: vec!["reasoning_engine".into()],
            physics_domains: vec!["mechanics".into()],
        };
        CapabilityCard::from_config(AgentPubKey::test_key(1), &config, &stats)
    }

    #[test]
    fn test_from_config_populates_fields() {
        let card = test_card();
        assert_eq!(card.format_version, 1);
        assert!(card.cycle_hz > 0.0);
        assert!(card.hdc_dimension > 0);
        assert!(card.cfc_neurons > 0);
    }

    #[test]
    fn test_hash_integrity() {
        let card = test_card();
        assert!(
            card.verify_hash(),
            "Fresh card should pass hash verification"
        );
    }

    #[test]
    fn test_hash_tamper_detection() {
        let mut card = test_card();
        card.phi = 9999.0;
        assert!(
            !card.verify_hash(),
            "Tampered card should fail hash verification"
        );
    }

    #[test]
    fn test_serde_roundtrip() {
        let card = test_card();
        let json = serde_json::to_string(&card).expect("serialize");
        let restored: CapabilityCard = serde_json::from_str(&json).expect("deserialize");
        assert!(
            restored.verify_hash(),
            "Deserialized card should pass hash verification"
        );
    }
}
