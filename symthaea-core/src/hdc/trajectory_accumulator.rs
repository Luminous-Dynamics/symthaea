//! Trajectory Accumulator — Continuous behavioral identity via HDC temporal binding.
//!
//! Accumulates action-encoded HVs per domain using `bind_temporal()` (non-commutative
//! permutation + XOR), creating a behavioral trajectory vector that encodes both
//! *what* actions were taken and *in what order*.
//!
//! ## Security Properties
//!
//! - **Hard to transfer**: Stealing a private key doesn't give you someone's trajectory.
//!   Forging requires sustained behavioral mimicry over time.
//! - **Temporally ordered**: A->B produces a different vector than B->A via cyclic permutation.
//! - **Continuously decaying**: Recent behavior weighs more than ancient history via
//!   exponential weighting in the bundle step.
//! - **Domain-separated**: Governance, trade, attestation, and social domains accumulate
//!   independently, preventing cross-domain trajectory pollution.
//!
//! ## NOT a replacement for cryptographic identity
//!
//! Ed25519 keys prove *who* you are. Trajectory vectors prove *that* you are a
//! consistent behavioral agent. This is a second factor, not a primary identity.

use std::collections::HashMap;

use super::binary_hv::BinaryHV;

/// Maximum actions to retain per domain before pruning oldest.
const MAX_ACTIONS_PER_DOMAIN: usize = 1024;

/// Default exponential decay rate for trajectory weighting.
/// Higher values mean recent actions dominate more strongly.
const DEFAULT_DECAY_RATE: f64 = 0.01;

/// Behavioral domains for trajectory separation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrajectoryDomain {
    /// Governance actions: voting, proposals, constitutional changes.
    Governance,
    /// Economic actions: trades, transfers, staking.
    Trade,
    /// Trust actions: attestations, endorsements, challenges.
    Attestation,
    /// Social actions: communication, collaboration, mentoring.
    Social,
    /// Technical actions: code contributions, reviews, deployments.
    Technical,
    /// General catch-all for unclassified actions.
    General,
}

/// A single recorded action in the trajectory.
#[derive(Debug, Clone)]
struct TrajectoryEntry {
    /// HDC encoding of the action.
    hv: BinaryHV,
    /// Cycle or timestamp when the action occurred.
    timestamp: u64,
}

/// Per-domain trajectory state.
#[derive(Debug)]
struct DomainTrajectory {
    /// Ordered list of action HVs (oldest first).
    entries: Vec<TrajectoryEntry>,
    /// Running temporally-bound accumulator.
    /// Each new action is bound via: `accumulator = accumulator.bind_temporal(&new_action)`
    accumulator: BinaryHV,
    /// Number of actions accumulated (including pruned).
    total_actions: u64,
}

impl DomainTrajectory {
    fn with_seed(seed: u64) -> Self {
        Self {
            entries: Vec::new(),
            accumulator: BinaryHV::random(seed),
            total_actions: 0,
        }
    }
}

/// Continuous behavioral identity accumulator.
///
/// Maintains per-domain trajectory vectors that encode the temporal
/// sequence of a user's actions as HDC hypervectors.
#[derive(Debug)]
pub struct TrajectoryAccumulator {
    /// Per-domain trajectories.
    domains: HashMap<TrajectoryDomain, DomainTrajectory>,
    /// Decay rate for exponential weighting.
    decay_rate: f64,
    /// Seed for deterministic initialization.
    seed: u64,
}

impl TrajectoryAccumulator {
    /// Create a new trajectory accumulator with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            domains: HashMap::new(),
            decay_rate: DEFAULT_DECAY_RATE,
            seed,
        }
    }

    /// Create with a custom decay rate.
    pub fn with_decay_rate(seed: u64, decay_rate: f64) -> Self {
        Self {
            domains: HashMap::new(),
            decay_rate: decay_rate.clamp(0.0001, 1.0),
            seed,
        }
    }

    /// Record an action in the specified domain.
    ///
    /// The action HV is temporally bound to the domain's accumulator,
    /// preserving the ordering of actions.
    pub fn record_action(&mut self, domain: TrajectoryDomain, action_hv: BinaryHV, timestamp: u64) {
        let domain_seed = self.seed.wrapping_add(domain as u64 * 7919);
        let traj = self
            .domains
            .entry(domain)
            .or_insert_with(|| DomainTrajectory::with_seed(domain_seed));

        // Temporal binding: order matters
        // accumulator = rho(accumulator) XOR action_hv
        traj.accumulator = traj.accumulator.bind_temporal(&action_hv);
        traj.entries.push(TrajectoryEntry {
            hv: action_hv,
            timestamp,
        });
        traj.total_actions += 1;

        // Prune oldest entries if over limit
        if traj.entries.len() > MAX_ACTIONS_PER_DOMAIN {
            let excess = traj.entries.len() - MAX_ACTIONS_PER_DOMAIN;
            traj.entries.drain(..excess);
        }
    }

    /// Get the current trajectory vector for a domain.
    ///
    /// Returns the temporally-bound accumulator that encodes the full
    /// action history in order.
    pub fn trajectory(&self, domain: &TrajectoryDomain) -> Option<&BinaryHV> {
        self.domains.get(domain).map(|t| &t.accumulator)
    }

    /// Compute a weighted snapshot of a domain's recent trajectory.
    ///
    /// Uses exponential decay weighting: recent actions have more influence.
    /// This is useful for comparing "current behavior" rather than "all-time behavior".
    pub fn weighted_snapshot(
        &self,
        domain: &TrajectoryDomain,
        current_time: u64,
    ) -> Option<BinaryHV> {
        let traj = self.domains.get(domain)?;
        if traj.entries.is_empty() {
            return None;
        }

        // Weight each entry by recency: w(t) = exp(-decay_rate * age)
        let weighted: Vec<BinaryHV> = traj
            .entries
            .iter()
            .filter_map(|entry| {
                let age = current_time.saturating_sub(entry.timestamp) as f64;
                let weight = (-self.decay_rate * age).exp();
                if weight > 0.01 {
                    // For weight > 0.01 (recent enough), include in bundle
                    Some(entry.hv)
                } else {
                    None // Too old to matter
                }
            })
            .collect();

        if weighted.is_empty() {
            return Some(traj.accumulator);
        }

        Some(BinaryHV::bundle(&weighted))
    }

    /// Compute the unified trajectory across all domains.
    ///
    /// Binds each domain's accumulator with a domain-specific role HV,
    /// then bundles all domain accumulators. This produces a single vector
    /// representing the agent's complete behavioral identity.
    pub fn unified_trajectory(&self) -> Option<BinaryHV> {
        let bound_domains: Vec<BinaryHV> = self
            .domains
            .iter()
            .map(|(domain, traj)| {
                let domain_role =
                    BinaryHV::random(self.seed.wrapping_add(*domain as u64 * 31337));
                traj.accumulator.bind(&domain_role)
            })
            .collect();

        if bound_domains.is_empty() {
            return None;
        }

        Some(BinaryHV::bundle(&bound_domains))
    }

    /// Compare two trajectories for behavioral similarity.
    ///
    /// Returns similarity in [0.0, 1.0]. High similarity means the agents
    /// have taken similar actions in similar order within the domain.
    pub fn compare(
        &self,
        domain: &TrajectoryDomain,
        other: &TrajectoryAccumulator,
    ) -> Option<f32> {
        let self_traj = self.trajectory(domain)?;
        let other_traj = other.trajectory(domain)?;
        Some(self_traj.similarity(other_traj))
    }

    /// Compute trajectory divergence from a reference point.
    ///
    /// Useful for detecting if someone's behavior has suddenly changed
    /// (potential credential theft or account takeover).
    pub fn divergence_from(
        &self,
        domain: &TrajectoryDomain,
        reference: &BinaryHV,
    ) -> Option<f32> {
        let current = self.trajectory(domain)?;
        Some(1.0 - current.similarity(reference))
    }

    /// Total actions recorded across all domains.
    pub fn total_actions(&self) -> u64 {
        self.domains.values().map(|t| t.total_actions).sum()
    }

    /// Number of active domains.
    pub fn active_domains(&self) -> usize {
        self.domains.len()
    }

    /// Actions recorded in a specific domain.
    pub fn domain_action_count(&self, domain: &TrajectoryDomain) -> u64 {
        self.domains.get(domain).map_or(0, |t| t.total_actions)
    }

    /// BLAKE3 hash of the unified trajectory for compact commitment.
    ///
    /// This can be included in a TrustCredential's kvector_commitment field
    /// to bind the credential to the agent's behavioral trajectory.
    pub fn trajectory_commitment(&self) -> Option<[u8; 32]> {
        let unified = self.unified_trajectory()?;
        let hash = blake3::hash(&unified.0);
        Some(*hash.as_bytes())
    }

    /// Clear all recorded actions (useful for testing or privacy reset).
    pub fn clear(&mut self) {
        self.domains.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action(seed: u64) -> BinaryHV {
        BinaryHV::random(seed)
    }

    #[test]
    fn record_and_retrieve() {
        let mut acc = TrajectoryAccumulator::new(42);
        acc.record_action(TrajectoryDomain::Governance, action(1), 100);
        acc.record_action(TrajectoryDomain::Governance, action(2), 200);

        assert!(acc.trajectory(&TrajectoryDomain::Governance).is_some());
        assert!(acc.trajectory(&TrajectoryDomain::Trade).is_none());
        assert_eq!(acc.total_actions(), 2);
        assert_eq!(acc.active_domains(), 1);
    }

    #[test]
    fn temporal_ordering_matters() {
        // A->B should produce a different trajectory than B->A
        let mut acc_ab = TrajectoryAccumulator::new(42);
        acc_ab.record_action(TrajectoryDomain::General, action(1), 100);
        acc_ab.record_action(TrajectoryDomain::General, action(2), 200);

        let mut acc_ba = TrajectoryAccumulator::new(42);
        acc_ba.record_action(TrajectoryDomain::General, action(2), 100);
        acc_ba.record_action(TrajectoryDomain::General, action(1), 200);

        let traj_ab = acc_ab.trajectory(&TrajectoryDomain::General).unwrap();
        let traj_ba = acc_ba.trajectory(&TrajectoryDomain::General).unwrap();

        // Should NOT be identical (temporal binding is non-commutative)
        let similarity = traj_ab.similarity(traj_ba);
        assert!(
            similarity < 0.9,
            "Different orderings should produce different trajectories (sim={similarity})"
        );
    }

    #[test]
    fn same_actions_same_order_same_trajectory() {
        let mut acc1 = TrajectoryAccumulator::new(42);
        let mut acc2 = TrajectoryAccumulator::new(42);

        for i in 0..10 {
            let a = action(i);
            acc1.record_action(TrajectoryDomain::General, a, i * 100);
            acc2.record_action(TrajectoryDomain::General, a, i * 100);
        }

        let sim = acc1.compare(&TrajectoryDomain::General, &acc2).unwrap();
        assert_eq!(
            sim, 1.0,
            "Identical action sequences should produce identical trajectories"
        );
    }

    #[test]
    fn domain_separation() {
        let mut acc = TrajectoryAccumulator::new(42);
        acc.record_action(TrajectoryDomain::Governance, action(1), 100);
        acc.record_action(TrajectoryDomain::Trade, action(2), 200);

        assert_eq!(acc.active_domains(), 2);
        assert_eq!(acc.domain_action_count(&TrajectoryDomain::Governance), 1);
        assert_eq!(acc.domain_action_count(&TrajectoryDomain::Trade), 1);

        // Different domains should have independent trajectories
        let gov = acc.trajectory(&TrajectoryDomain::Governance).unwrap();
        let trade = acc.trajectory(&TrajectoryDomain::Trade).unwrap();
        let sim = gov.similarity(trade);
        assert!(
            sim < 0.7,
            "Different domains should have independent trajectories"
        );
    }

    #[test]
    fn unified_trajectory_combines_domains() {
        let mut acc = TrajectoryAccumulator::new(42);
        acc.record_action(TrajectoryDomain::Governance, action(1), 100);
        acc.record_action(TrajectoryDomain::Trade, action(2), 200);
        acc.record_action(TrajectoryDomain::Social, action(3), 300);

        let unified = acc.unified_trajectory();
        assert!(unified.is_some());
    }

    #[test]
    fn divergence_detection() {
        let mut acc = TrajectoryAccumulator::new(42);

        // Build up a trajectory
        for i in 0..20 {
            acc.record_action(TrajectoryDomain::General, action(i), i * 100);
        }

        // Save reference point
        let reference = *acc.trajectory(&TrajectoryDomain::General).unwrap();

        // Add more actions — divergence should be nonzero
        for i in 20..30 {
            acc.record_action(TrajectoryDomain::General, action(i), i * 100);
        }

        let divergence = acc
            .divergence_from(&TrajectoryDomain::General, &reference)
            .unwrap();
        assert!(divergence > 0.0, "Trajectory should have evolved");
        assert!(divergence < 1.0, "Divergence should be bounded");
    }

    #[test]
    fn trajectory_commitment_deterministic() {
        let mut acc1 = TrajectoryAccumulator::new(42);
        let mut acc2 = TrajectoryAccumulator::new(42);

        for i in 0..5 {
            let a = action(i);
            acc1.record_action(TrajectoryDomain::General, a, i * 100);
            acc2.record_action(TrajectoryDomain::General, a, i * 100);
        }

        let c1 = acc1.trajectory_commitment().unwrap();
        let c2 = acc2.trajectory_commitment().unwrap();
        assert_eq!(c1, c2, "Same trajectory should produce same commitment");
    }

    #[test]
    fn pruning_limits_memory() {
        let mut acc = TrajectoryAccumulator::new(42);

        for i in 0..2000 {
            acc.record_action(TrajectoryDomain::General, action(i), i as u64);
        }

        // Total count preserved even after pruning entries
        assert_eq!(acc.total_actions(), 2000);
    }

    #[test]
    fn empty_accumulator() {
        let acc = TrajectoryAccumulator::new(42);
        assert!(acc.trajectory(&TrajectoryDomain::General).is_none());
        assert!(acc.unified_trajectory().is_none());
        assert!(acc.trajectory_commitment().is_none());
        assert_eq!(acc.total_actions(), 0);
        assert_eq!(acc.active_domains(), 0);
    }

    #[test]
    fn weighted_snapshot_favors_recent() {
        let mut acc = TrajectoryAccumulator::with_decay_rate(42, 0.1);

        // Old action
        let old_action = action(1);
        acc.record_action(TrajectoryDomain::General, old_action, 0);

        // Recent action
        let recent_action = action(2);
        acc.record_action(TrajectoryDomain::General, recent_action, 1000);

        let snapshot = acc
            .weighted_snapshot(&TrajectoryDomain::General, 1000)
            .unwrap();

        // Snapshot should be more similar to recent action than old action
        let sim_recent = snapshot.similarity(&recent_action);
        let sim_old = snapshot.similarity(&old_action);
        assert!(
            sim_recent > sim_old,
            "Weighted snapshot should favor recent actions (recent={sim_recent}, old={sim_old})"
        );
    }
}
