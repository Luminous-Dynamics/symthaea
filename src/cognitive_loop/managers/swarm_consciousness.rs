// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Swarm cognition — collective Φ as a non-authoritative cognitive signal.
//!
//! When robots form coalitions with high mutual information, the collective
//! has a measurable Φ_swarm. That estimate may influence learning, attention,
//! information-seeking or proposal ranking. It never grants or transfers
//! execution authority; all protected effects require independent capabilities.
//!
//! Science: Tononi (2004) IIT applied to multi-agent systems,
//! Friston (2013) Markov blanket formation in collectives.

use crate::consciousness::fep_active_inference::SwarmCoalition;

/// Swarm consciousness state computed from coalition identification.
#[derive(Debug, Clone)]
pub struct SwarmConsciousness {
    /// The strongest coalition's collective Phi.
    pub phi_swarm: f64,
    /// Member Phis in the strongest coalition: (peer_id, phi).
    pub member_phis: Vec<(String, f64)>,
    /// Number of conscious collectives.
    pub conscious_collective_count: usize,
    /// Total robots in any coalition.
    pub total_coalition_members: usize,
    /// Collective cognitive influence for this robot; never execution authority.
    pub cognitive_influence: CollectiveCognitiveInfluence,
}

/// Non-authoritative cognitive influence from a collective.
///
/// This estimate may inform learning, attention or proposal ranking only.
/// It must never be consumed as a capability, permit, lease or effect grant.
#[derive(Debug, Clone)]
pub struct CollectiveCognitiveInfluence {
    /// This robot's individual Phi.
    pub individual_phi: f64,
    /// Collective Phi estimate from the strongest coalition.
    pub collective_phi: f64,
    /// Blended cognitive estimate; not an authority-bearing value.
    pub blended_cognitive_phi: f64,
    /// Weight assigned to the collective cognitive estimate.
    pub collective_influence_weight: f64,
    /// Whether this robot is part of a conscious collective.
    pub in_conscious_collective: bool,
}

impl SwarmConsciousness {
    /// Compute swarm consciousness from current coalitions and this robot's Phi.
    ///
    /// `coalitions`: active coalitions from SwarmManager
    /// `individual_phi`: this robot's consciousness level
    /// `self_id`: this robot's peer ID (if in a coalition)
    pub fn compute(
        coalitions: &[SwarmCoalition],
        individual_phi: f64,
        self_id: Option<&str>,
    ) -> Self {
        if coalitions.is_empty() {
            return Self {
                phi_swarm: 0.0,
                member_phis: Vec::new(),
                conscious_collective_count: 0,
                total_coalition_members: 0,
                cognitive_influence: CollectiveCognitiveInfluence::no_influence(individual_phi),
            };
        }

        // Find strongest coalition by collective_phi
        let strongest = coalitions
            .iter()
            .max_by(|a, b| a.collective_phi().total_cmp(&b.collective_phi()))
            .unwrap();

        let phi_swarm = strongest.collective_phi();
        let conscious_count = coalitions
            .iter()
            .filter(|c| c.is_conscious_collective())
            .count();
        let total_members: usize = coalitions.iter().map(|c| c.size()).sum();

        // Check if this robot is in the strongest coalition
        let in_strongest = self_id
            .map(|id| strongest.members.contains(&id.to_string()))
            .unwrap_or(false);

        let in_conscious = in_strongest && strongest.is_conscious_collective();

        // Collective signal influences cognition only; it never authorizes effects.
        let cognitive_influence = if in_conscious && phi_swarm > individual_phi {
            // Blend: 30% collective cognitive estimate + 70% individual estimate.
            let alpha = 0.3;
            let blended = alpha * phi_swarm + (1.0 - alpha) * individual_phi;
            let blended_cognitive_phi = blended.max(individual_phi);
            CollectiveCognitiveInfluence {
                individual_phi,
                collective_phi: phi_swarm,
                blended_cognitive_phi,
                collective_influence_weight: alpha,
                in_conscious_collective: true,
            }
        } else {
            CollectiveCognitiveInfluence::no_influence(individual_phi)
        };

        Self {
            phi_swarm,
            member_phis: strongest.members.iter().map(|m| (m.clone(), 0.0)).collect(),
            conscious_collective_count: conscious_count,
            total_coalition_members: total_members,
            cognitive_influence,
        }
    }
}

impl CollectiveCognitiveInfluence {
    /// No collective influence — individual Phi only.
    pub fn no_influence(individual_phi: f64) -> Self {
        Self {
            individual_phi,
            collective_phi: 0.0,
            blended_cognitive_phi: individual_phi,
            collective_influence_weight: 0.0,
            in_conscious_collective: false,
        }
    }
}

impl Default for SwarmConsciousness {
    fn default() -> Self {
        Self {
            phi_swarm: 0.0,
            member_phis: Vec::new(),
            conscious_collective_count: 0,
            total_coalition_members: 0,
            cognitive_influence: CollectiveCognitiveInfluence::no_influence(0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_coalition(members: Vec<&str>, internal_phi: f64, cohesion: f64) -> SwarmCoalition {
        SwarmCoalition {
            members: members.into_iter().map(|s| s.to_string()).collect(),
            internal_phi,
            boundary_phi: 0.1,
            cohesion,
            mean_internal_permeability: 0.8,
            mean_external_permeability: 0.2,
        }
    }

    #[test]
    fn test_no_coalitions() {
        let sc = SwarmConsciousness::compute(&[], 0.5, Some("self"));
        assert_eq!(sc.phi_swarm, 0.0);
        assert_eq!(sc.cognitive_influence.blended_cognitive_phi, 0.5);
        assert!(!sc.cognitive_influence.in_conscious_collective);
    }

    #[test]
    fn test_conscious_collective_influences_cognition_only() {
        let coalition = make_coalition(vec!["self", "peer1", "peer2"], 0.8, 0.9);
        let sc = SwarmConsciousness::compute(&[coalition], 0.3, Some("self"));

        assert!(sc.cognitive_influence.in_conscious_collective);
        assert!(
            sc.cognitive_influence.blended_cognitive_phi > 0.3,
            "Should include collective cognitive influence"
        );
        assert!(
            sc.cognitive_influence.blended_cognitive_phi < 0.8,
            "Should not fully equal swarm"
        );
        assert!(sc.cognitive_influence.collective_influence_weight > 0.0);
    }

    #[test]
    fn test_not_in_coalition_no_collective_influence() {
        let coalition = make_coalition(vec!["peer1", "peer2", "peer3"], 0.8, 0.9);
        let sc = SwarmConsciousness::compute(&[coalition], 0.3, Some("outsider"));

        assert!(!sc.cognitive_influence.in_conscious_collective);
        assert_eq!(sc.cognitive_influence.blended_cognitive_phi, 0.3);
    }

    #[test]
    fn test_high_individual_phi_not_downgraded() {
        let coalition = make_coalition(vec!["self", "peer1", "peer2"], 0.4, 0.9);
        let sc = SwarmConsciousness::compute(&[coalition], 0.9, Some("self"));

        // Individual Phi > swarm Phi — blended cognitive estimate should stay at the individual value
        assert!(sc.cognitive_influence.blended_cognitive_phi >= 0.9);
    }

    #[test]
    fn test_multiple_coalitions_picks_strongest() {
        let weak = make_coalition(vec!["a", "b", "c"], 0.3, 0.7);
        let strong = make_coalition(vec!["self", "d", "e"], 0.9, 0.95);
        let sc = SwarmConsciousness::compute(&[weak, strong], 0.4, Some("self"));

        assert!(sc.phi_swarm > 0.8, "Should pick strongest coalition");
        assert!(sc.cognitive_influence.in_conscious_collective);
    }
}
