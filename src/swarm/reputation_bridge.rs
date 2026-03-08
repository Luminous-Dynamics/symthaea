//! Reputation bridge for processing capability cards.
//!
//! Tracks interaction counts and auto-vouches peers after sufficient
//! successful interactions above a Phi threshold.

use super::capability_card::CapabilityCard;

/// Result of processing a capability card.
#[derive(Debug, Clone, PartialEq)]
pub enum VouchDecision {
    /// Card had invalid hash — rejected.
    Rejected,
    /// Card valid but insufficient interactions for vouch.
    Accepted { interactions: u64, needed: u64 },
    /// Card valid and peer is vouched.
    Vouched,
}

/// Processes received capability cards and decides whether to vouch.
pub struct ReputationBridge {
    min_interactions: u64,
    phi_threshold: f64,
    interactions: std::collections::HashMap<String, u64>,
}

impl ReputationBridge {
    /// Create a new reputation bridge.
    /// `min_interactions`: minimum successful interactions before auto-vouch.
    /// `phi_threshold`: minimum Phi value on the card to consider vouching.
    pub fn new(min_interactions: u64, phi_threshold: f64) -> Self {
        Self {
            min_interactions,
            phi_threshold,
            interactions: std::collections::HashMap::new(),
        }
    }

    /// Process a received capability card.
    /// Returns Rejected if hash is invalid, Accepted if not enough interactions,
    /// or Vouched if the peer meets all criteria.
    pub fn process_card(&mut self, card: &CapabilityCard) -> VouchDecision {
        if !card.verify_hash() {
            return VouchDecision::Rejected;
        }

        let key = card.agent_key.as_str().to_string();
        let count = self.interactions.entry(key).or_insert(0);
        *count += 1;

        if card.phi < self.phi_threshold {
            return VouchDecision::Accepted {
                interactions: *count,
                needed: self.min_interactions,
            };
        }

        if *count >= self.min_interactions {
            VouchDecision::Vouched
        } else {
            VouchDecision::Accepted {
                interactions: *count,
                needed: self.min_interactions,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_loop::config::CognitiveLoopConfig;
    use crate::swarm::capability_card::CardStats;
    use crate::swarm::holochain::AgentPubKey;

    fn make_card(phi: f64) -> CapabilityCard {
        let config = CognitiveLoopConfig::default();
        let stats = CardStats {
            generated_at: 1000,
            substrate_feasibility: 0.71,
            cycle_hz: 234.0,
            phi,
            features: vec![],
            physics_domains: vec![],
        };
        CapabilityCard::from_config(AgentPubKey::test_key(1), &config, &stats)
    }

    #[test]
    fn test_bad_hash_rejected() {
        let mut bridge = ReputationBridge::new(3, 0.5);
        let mut card = make_card(0.9);
        card.phi = 0.1; // tamper
        assert_eq!(bridge.process_card(&card), VouchDecision::Rejected);
    }

    #[test]
    fn test_insufficient_interactions() {
        let mut bridge = ReputationBridge::new(3, 0.5);
        let card = make_card(0.9);
        let result = bridge.process_card(&card);
        assert!(matches!(
            result,
            VouchDecision::Accepted {
                interactions: 1,
                needed: 3
            }
        ));
    }

    #[test]
    fn test_auto_vouch_triggers() {
        let mut bridge = ReputationBridge::new(3, 0.5);
        let card = make_card(0.9);
        bridge.process_card(&card);
        bridge.process_card(&card);
        let result = bridge.process_card(&card);
        assert_eq!(result, VouchDecision::Vouched);
    }

    #[test]
    fn test_phi_below_threshold() {
        let mut bridge = ReputationBridge::new(1, 0.8);
        let card = make_card(0.3);
        // Even with enough interactions, low phi prevents vouch
        let result = bridge.process_card(&card);
        assert!(matches!(result, VouchDecision::Accepted { .. }));
    }
}
