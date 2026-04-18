// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Swarm Consciousness — collective Φ and authority delegation.
//!
//! When robots form coalitions with high mutual information, the collective
//! has a measurable Φ_swarm. A robot with low individual Φ can inherit
//! authority from a high-Φ_swarm coalition, allowing swarm-level decisions
//! that transcend individual capability.
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
    /// Authority delegation state for this robot.
    pub delegation: AuthorityDelegation,
}

/// Authority delegation from swarm to individual.
///
/// A robot with low individual Φ can operate at a higher effective Φ
/// when part of a conscious collective. The delegation ratio controls
/// how much authority flows from swarm → individual.
#[derive(Debug, Clone)]
pub struct AuthorityDelegation {
    /// This robot's individual Phi.
    pub individual_phi: f64,
    /// Phi inherited from the strongest coalition.
    pub inherited_phi: f64,
    /// Effective Phi = max(individual, blended).
    pub effective_phi: f64,
    /// Ratio of authority from swarm (0.0 = all individual, 1.0 = all swarm).
    pub delegation_ratio: f64,
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
                delegation: AuthorityDelegation::no_delegation(individual_phi),
            };
        }

        // Find strongest coalition by collective_phi
        let strongest = coalitions
            .iter()
            .max_by(|a, b| a.collective_phi().partial_cmp(&b.collective_phi()).unwrap())
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

        // Authority delegation
        let delegation = if in_conscious && phi_swarm > individual_phi {
            // Blend: 30% swarm + 70% individual (conservative delegation)
            let alpha = 0.3;
            let blended = alpha * phi_swarm + (1.0 - alpha) * individual_phi;
            let effective = blended.max(individual_phi);
            AuthorityDelegation {
                individual_phi,
                inherited_phi: phi_swarm,
                effective_phi: effective,
                delegation_ratio: alpha,
                in_conscious_collective: true,
            }
        } else {
            AuthorityDelegation::no_delegation(individual_phi)
        };

        Self {
            phi_swarm,
            member_phis: strongest.members.iter().map(|m| (m.clone(), 0.0)).collect(),
            conscious_collective_count: conscious_count,
            total_coalition_members: total_members,
            delegation,
        }
    }
}

impl AuthorityDelegation {
    /// No delegation — individual Phi only.
    pub fn no_delegation(individual_phi: f64) -> Self {
        Self {
            individual_phi,
            inherited_phi: 0.0,
            effective_phi: individual_phi,
            delegation_ratio: 0.0,
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
            delegation: AuthorityDelegation::no_delegation(0.0),
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
        assert_eq!(sc.delegation.effective_phi, 0.5);
        assert!(!sc.delegation.in_conscious_collective);
    }

    #[test]
    fn test_conscious_collective_delegates_authority() {
        let coalition = make_coalition(vec!["self", "peer1", "peer2"], 0.8, 0.9);
        let sc = SwarmConsciousness::compute(&[coalition], 0.3, Some("self"));

        assert!(sc.delegation.in_conscious_collective);
        assert!(
            sc.delegation.effective_phi > 0.3,
            "Should inherit from swarm"
        );
        assert!(
            sc.delegation.effective_phi < 0.8,
            "Should not fully equal swarm"
        );
        assert!(sc.delegation.delegation_ratio > 0.0);
    }

    #[test]
    fn test_not_in_coalition_no_delegation() {
        let coalition = make_coalition(vec!["peer1", "peer2", "peer3"], 0.8, 0.9);
        let sc = SwarmConsciousness::compute(&[coalition], 0.3, Some("outsider"));

        assert!(!sc.delegation.in_conscious_collective);
        assert_eq!(sc.delegation.effective_phi, 0.3);
    }

    #[test]
    fn test_high_individual_phi_not_downgraded() {
        let coalition = make_coalition(vec!["self", "peer1", "peer2"], 0.4, 0.9);
        let sc = SwarmConsciousness::compute(&[coalition], 0.9, Some("self"));

        // Individual Phi > swarm Phi — effective should stay at individual
        assert!(sc.delegation.effective_phi >= 0.9);
    }

    #[test]
    fn test_multiple_coalitions_picks_strongest() {
        let weak = make_coalition(vec!["a", "b", "c"], 0.3, 0.7);
        let strong = make_coalition(vec!["self", "d", "e"], 0.9, 0.95);
        let sc = SwarmConsciousness::compute(&[weak, strong], 0.4, Some("self"));

        assert!(sc.phi_swarm > 0.8, "Should pick strongest coalition");
        assert!(sc.delegation.in_conscious_collective);
    }
}
