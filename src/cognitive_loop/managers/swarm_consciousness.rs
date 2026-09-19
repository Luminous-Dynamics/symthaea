// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Swarm Consciousness — collective Φ as cognition-only context.
//!
//! When robots form coalitions with high mutual information, the collective
//! may expose a measurable Φ_swarm. This value may modulate cognition such as
//! attention, learning, epistemic caution, or coordination confidence.
//!
//! **Collective Φ never grants capabilities, authorization, safety approval,
//! or execution authority.** Those remain the responsibility of explicit
//! authority and safety systems outside this cognitive model.
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
    /// Cognition-only collective influence for this robot.
    ///
    /// The `delegation` field name is retained temporarily for source compatibility.
    /// This value does not delegate or confer execution authority.
    pub delegation: CollectiveCognitiveInfluence,
}

/// Cognition-only influence from a conscious collective.
///
/// A collective may provide epistemic/cognitive context to an individual agent.
/// This type deliberately carries no capability, permit, lease, safety admission,
/// or other authority-bearing state.
#[derive(Debug, Clone)]
pub struct CollectiveCognitiveInfluence {
    /// This robot's individual Phi.
    pub individual_phi: f64,
    /// Phi measured for the strongest relevant collective.
    pub collective_phi: f64,
    /// Cognition-only Phi used by cognitive modulation.
    ///
    /// This is not an authority level and must not be used as authorization evidence.
    pub effective_phi: f64,
    /// Fraction of the cognition-only blend contributed by the collective.
    /// Zero means the collective did not raise the effective cognitive Phi.
    pub influence_ratio: f64,
    /// Whether this robot is part of the strongest conscious collective.
    pub in_conscious_collective: bool,
}

/// Legacy type name retained for source compatibility.
///
/// This alias does not imply or carry execution authority. New code should use
/// [`CollectiveCognitiveInfluence`] and explicit authority types at effect boundaries.
#[deprecated(
    note = "use CollectiveCognitiveInfluence; collective Phi does not delegate authority"
)]
pub type AuthorityDelegation = CollectiveCognitiveInfluence;

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
                delegation: CollectiveCognitiveInfluence::individual_only(individual_phi),
            };
        }

        // Find strongest coalition by collective_phi.
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

        // Check if this robot is in the strongest coalition.
        let in_strongest = self_id
            .map(|id| strongest.members.contains(&id.to_string()))
            .unwrap_or(false);

        let in_conscious = in_strongest && strongest.is_conscious_collective();

        // Cognitive influence only. This calculation does not grant authority.
        let delegation = CollectiveCognitiveInfluence::from_collective(
            individual_phi,
            phi_swarm,
            in_conscious,
        );

        Self {
            phi_swarm,
            member_phis: strongest.members.iter().map(|m| (m.clone(), 0.0)).collect(),
            conscious_collective_count: conscious_count,
            total_coalition_members: total_members,
            delegation,
        }
    }
}

impl CollectiveCognitiveInfluence {
    const COLLECTIVE_BLEND_RATIO: f64 = 0.3;

    /// Individual cognition with no collective influence.
    pub fn individual_only(individual_phi: f64) -> Self {
        Self {
            individual_phi,
            collective_phi: 0.0,
            effective_phi: individual_phi,
            influence_ratio: 0.0,
            in_conscious_collective: false,
        }
    }

    fn from_collective(
        individual_phi: f64,
        collective_phi: f64,
        in_conscious_collective: bool,
    ) -> Self {
        if !in_conscious_collective {
            return Self::individual_only(individual_phi);
        }

        // Collective context may make cognition more conservative/informed, but it
        // never becomes an authority grant. Preserve membership even when the
        // collective does not raise the individual's effective cognitive Phi.
        let (effective_phi, influence_ratio) = if collective_phi > individual_phi {
            let alpha = Self::COLLECTIVE_BLEND_RATIO;
            let blended = alpha * collective_phi + (1.0 - alpha) * individual_phi;
            (blended.max(individual_phi), alpha)
        } else {
            (individual_phi, 0.0)
        };

        Self {
            individual_phi,
            collective_phi,
            effective_phi,
            influence_ratio,
            in_conscious_collective: true,
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
            delegation: CollectiveCognitiveInfluence::individual_only(0.0),
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
        assert_eq!(sc.delegation.collective_phi, 0.0);
        assert_eq!(sc.delegation.influence_ratio, 0.0);
        assert!(!sc.delegation.in_conscious_collective);
    }

    #[test]
    fn test_conscious_collective_provides_cognitive_influence() {
        let coalition = make_coalition(vec!["self", "peer1", "peer2"], 0.8, 0.9);
        let sc = SwarmConsciousness::compute(&[coalition], 0.3, Some("self"));

        assert!(sc.delegation.in_conscious_collective);
        assert_eq!(sc.delegation.collective_phi, sc.phi_swarm);
        assert!(
            sc.delegation.effective_phi > 0.3,
            "Collective context may raise cognition-only effective Phi"
        );
        assert!(
            sc.delegation.effective_phi < 0.8,
            "Cognitive blend should not fully equal collective Phi"
        );
        assert!(sc.delegation.influence_ratio > 0.0);
    }

    #[test]
    fn test_not_in_coalition_has_no_collective_influence() {
        let coalition = make_coalition(vec!["peer1", "peer2", "peer3"], 0.8, 0.9);
        let sc = SwarmConsciousness::compute(&[coalition], 0.3, Some("outsider"));

        assert!(!sc.delegation.in_conscious_collective);
        assert_eq!(sc.delegation.effective_phi, 0.3);
        assert_eq!(sc.delegation.influence_ratio, 0.0);
    }

    #[test]
    fn test_high_individual_phi_not_downgraded_by_collective() {
        let coalition = make_coalition(vec!["self", "peer1", "peer2"], 0.4, 0.9);
        let sc = SwarmConsciousness::compute(&[coalition], 0.9, Some("self"));

        assert!(sc.delegation.in_conscious_collective);
        assert_eq!(sc.delegation.effective_phi, 0.9);
        assert_eq!(sc.delegation.influence_ratio, 0.0);
        assert_eq!(sc.delegation.collective_phi, sc.phi_swarm);
    }

    #[test]
    fn test_multiple_coalitions_picks_strongest() {
        let weak = make_coalition(vec!["a", "b", "c"], 0.3, 0.7);
        let strong = make_coalition(vec!["self", "d", "e"], 0.9, 0.95);
        let sc = SwarmConsciousness::compute(&[weak, strong], 0.4, Some("self"));

        assert!(sc.phi_swarm > 0.8, "Should pick strongest coalition");
        assert!(sc.delegation.in_conscious_collective);
        assert_eq!(sc.delegation.collective_phi, sc.phi_swarm);
    }
}
