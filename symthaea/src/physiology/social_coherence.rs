// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Social Coherence Module
//!
//! Models social/collective coherence dynamics for consciousness systems.
//! Provides types for peer-to-peer coherence sharing and collective learning.

use std::collections::HashMap;
use std::time::Duration;

use super::coherence::{CoherenceState, TaskComplexity};
use super::endocrine::HormoneState;

/// A beacon broadcasting coherence state to peers
#[derive(Debug, Clone, Default)]
pub struct CoherenceBeacon {
    /// Source identifier
    pub source_id: String,
    /// Current coherence level (0.0-1.0)
    pub coherence_level: f64,
    /// Resonance frequency
    pub resonance_frequency: f64,
    /// Current task complexity
    pub current_task: Option<TaskComplexity>,
    /// Timestamp (ms since epoch)
    pub timestamp: u64,
    /// Optional message
    pub message: Option<String>,
}

impl CoherenceBeacon {
    /// Create a new coherence beacon
    pub fn new(source_id: impl Into<String>, coherence_level: f64) -> Self {
        Self {
            source_id: source_id.into(),
            coherence_level: coherence_level.clamp(0.0, 1.0),
            resonance_frequency: 1.0,
            current_task: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            message: None,
        }
    }

    /// Create from coherence state
    pub fn from_state(
        source_id: impl Into<String>,
        state: &CoherenceState,
        _hormones: &HormoneState,
        current_task: Option<TaskComplexity>,
    ) -> Self {
        Self {
            source_id: source_id.into(),
            coherence_level: state.coherence as f64,
            // Use relational_resonance as a proxy for resonance frequency
            resonance_frequency: state.relational_resonance as f64,
            current_task,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            message: None,
        }
    }
}

/// A coherence loan between peers
#[derive(Debug, Clone, Default)]
pub struct CoherenceLoan {
    /// Lender identifier
    pub lender_id: String,
    /// Borrower identifier
    pub borrower_id: String,
    /// Amount of coherence loaned
    pub amount: f32,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Interest rate (coherence growth factor)
    pub interest_rate: f32,
    /// Whether the loan is active
    pub active: bool,
    /// Time remaining
    pub remaining: Duration,
}

impl CoherenceLoan {
    /// Create a new coherence loan
    pub fn new(
        lender: impl Into<String>,
        borrower: impl Into<String>,
        amount: f32,
        duration: Duration,
    ) -> Self {
        Self {
            lender_id: lender.into(),
            borrower_id: borrower.into(),
            amount: amount.clamp(0.0, 1.0),
            duration_ms: duration.as_millis() as u64,
            interest_rate: 0.01,
            active: true,
            remaining: duration,
        }
    }
}

/// Social coherence field representing collective consciousness state
#[derive(Debug, Clone)]
pub struct SocialCoherenceField {
    /// Instance identifier
    pub instance_id: String,
    /// Individual coherence contributions
    pub contributions: HashMap<String, f64>,
    /// Aggregate field strength
    pub field_strength: f64,
    /// Resonance patterns
    pub resonance_patterns: Vec<f64>,
    /// Active beacons
    pub active_beacons: Vec<CoherenceBeacon>,
}

impl Default for SocialCoherenceField {
    fn default() -> Self {
        Self::new("default")
    }
}

impl SocialCoherenceField {
    /// Create a new social coherence field
    pub fn new(instance_id: impl Into<String>) -> Self {
        Self {
            instance_id: instance_id.into(),
            contributions: HashMap::new(),
            field_strength: 0.0,
            resonance_patterns: Vec::new(),
            active_beacons: Vec::new(),
        }
    }

    /// Add a contribution to the field
    pub fn add_contribution(&mut self, source: impl Into<String>, coherence: f64) {
        let source = source.into();
        self.contributions.insert(source, coherence.clamp(0.0, 1.0));
        self.update_field_strength();
    }

    /// Update aggregate field strength
    fn update_field_strength(&mut self) {
        if self.contributions.is_empty() {
            self.field_strength = 0.0;
        } else {
            let sum: f64 = self.contributions.values().sum();
            let count = self.contributions.len() as f64;
            // Collective coherence has synergistic effects
            self.field_strength = (sum / count) * (1.0 + 0.1 * count.ln().max(0.0));
        }
    }

    /// Broadcast current state as a beacon
    pub fn broadcast_state(
        &self,
        state: &CoherenceState,
        hormones: &HormoneState,
        current_task: Option<TaskComplexity>,
    ) -> CoherenceBeacon {
        CoherenceBeacon::from_state(&self.instance_id, state, hormones, current_task)
    }

    /// Receive a beacon and integrate it
    pub fn receive_beacon(&mut self, beacon: CoherenceBeacon) {
        self.add_contribution(&beacon.source_id, beacon.coherence_level);
        self.active_beacons.push(beacon);

        // Keep only recent beacons
        if self.active_beacons.len() > 100 {
            self.active_beacons.remove(0);
        }
    }

    /// Get alignment vector towards collective field
    /// Returns (target_coherence, target_resonance)
    pub fn get_alignment_vector(&self, _local_coherence: f32, local_resonance: f32) -> (f32, f32) {
        let target_coherence = self.field_strength as f32;
        let avg_resonance = if !self.active_beacons.is_empty() {
            self.active_beacons
                .iter()
                .map(|b| b.resonance_frequency as f32)
                .sum::<f32>()
                / self.active_beacons.len() as f32
        } else {
            local_resonance
        };
        (target_coherence, avg_resonance)
    }

    /// Apply synchronization, returning deltas
    /// Returns (coherence_delta, resonance_delta)
    pub fn apply_synchronization(
        &mut self,
        local_coherence: f32,
        local_resonance: f32,
    ) -> (f32, f32) {
        let (target_coherence, target_resonance) =
            self.get_alignment_vector(local_coherence, local_resonance);

        // Gradual alignment (10% per step)
        let coherence_delta = (target_coherence - local_coherence) * 0.1;
        let resonance_delta = (target_resonance - local_resonance) * 0.1;

        (coherence_delta, resonance_delta)
    }

    /// Calculate collective coherence including local
    pub fn calculate_collective_coherence(&mut self, local_coherence: f32) -> f32 {
        self.add_contribution(self.instance_id.clone(), local_coherence as f64);
        self.field_strength as f32
    }

    /// Get current field strength
    pub fn strength(&self) -> f64 {
        self.field_strength
    }
}

/// Protocol for lending coherence between peers
#[derive(Debug, Clone, Default)]
pub struct CoherenceLendingProtocol {
    /// Instance identifier
    pub instance_id: String,
    /// Active loans
    pub active_loans: Vec<CoherenceLoan>,
    /// Total coherence lent out
    pub total_lent: f32,
    /// Total coherence borrowed
    pub total_borrowed: f32,
    /// Trust scores for peers
    pub trust_scores: HashMap<String, f32>,
    /// Resonance boost accumulator
    pub resonance_boost: f32,
}

impl CoherenceLendingProtocol {
    /// Create a new lending protocol
    pub fn new(instance_id: impl Into<String>) -> Self {
        Self {
            instance_id: instance_id.into(),
            active_loans: Vec::new(),
            total_lent: 0.0,
            total_borrowed: 0.0,
            trust_scores: HashMap::new(),
            resonance_boost: 0.0,
        }
    }

    /// Check if we can lend the specified amount
    pub fn can_lend(&self, amount: f32, current_coherence: f32) -> bool {
        // Can lend if we have enough coherence and not too much lent out
        current_coherence > 0.5
            && self.total_lent + amount < current_coherence * 0.5
            && amount > 0.0
            && amount < 0.3
    }

    /// Grant a loan to a peer
    pub fn grant_loan(
        &mut self,
        to_peer: impl Into<String>,
        amount: f32,
        duration: Duration,
        _current_coherence: f32,
    ) -> Result<CoherenceLoan, crate::errors::SymthaeaError> {
        let to_peer = to_peer.into();
        let loan = CoherenceLoan::new(&self.instance_id, &to_peer, amount, duration);

        self.total_lent += amount;
        self.active_loans.push(loan.clone());

        // Generous lending creates resonance boost
        self.resonance_boost += 0.05;

        Ok(loan)
    }

    /// Calculate resonance boost from generous lending
    pub fn calculate_resonance_boost(&self) -> f32 {
        self.resonance_boost.min(0.2) // Cap at 0.2
    }

    /// Process loan repayments over time
    /// Returns (coherence_returned, coherence_lost)
    pub fn process_repayments(&mut self, dt: Duration) -> (f32, f32) {
        let mut returned = 0.0;
        let lost = 0.0; // Currently unused, reserved for future default/timeout scenarios

        // Update each loan
        for loan in &mut self.active_loans {
            if loan.active {
                if loan.remaining <= dt {
                    // Loan complete - coherence returns with interest
                    returned += loan.amount * (1.0 + loan.interest_rate);
                    loan.active = false;
                } else {
                    loan.remaining = loan.remaining.saturating_sub(dt);
                }
            }
        }

        // Remove completed loans
        self.active_loans.retain(|l| l.active);

        // Update totals
        self.total_lent -= returned;
        if self.total_lent < 0.0 {
            self.total_lent = 0.0;
        }

        // Decay resonance boost
        self.resonance_boost *= 0.99;

        (returned, lost)
    }

    /// Request a loan
    pub fn request_loan(&mut self, borrower: &str, amount: f32) -> Option<CoherenceLoan> {
        let loan = CoherenceLoan::new("pool", borrower, amount, Duration::from_secs(60));
        self.total_borrowed += amount;
        Some(loan)
    }

    /// Repay a loan
    pub fn repay(&mut self, borrower_id: &str) {
        self.active_loans
            .retain(|loan| loan.borrower_id != borrower_id);
    }

    /// Update trust score for a peer
    pub fn update_trust(&mut self, peer_id: impl Into<String>, delta: f32) {
        let peer_id = peer_id.into();
        let current = self.trust_scores.get(&peer_id).copied().unwrap_or(0.5);
        self.trust_scores
            .insert(peer_id, (current + delta).clamp(0.0, 1.0));
    }

    /// Accept an incoming loan from another peer (borrowing)
    pub fn accept_loan(&mut self, loan: CoherenceLoan) {
        self.total_borrowed += loan.amount;
        self.active_loans.push(loan);
    }
}

/// Collective learning system for shared consciousness growth
#[derive(Debug, Clone, Default)]
pub struct CollectiveLearning {
    /// Instance identifier
    pub instance_id: String,
    /// Shared knowledge pool
    pub knowledge_pool: Vec<String>,
    /// Learning rate multiplier from collective
    pub collective_boost: f64,
    /// Number of participants
    pub participant_count: usize,
    /// Synergy factor
    pub synergy: f64,
    /// Threshold observations per task type
    pub threshold_observations: HashMap<String, Vec<(f32, bool)>>,
}

impl CollectiveLearning {
    /// Create a new collective learning system
    pub fn new(instance_id: impl Into<String>) -> Self {
        Self {
            instance_id: instance_id.into(),
            knowledge_pool: Vec::new(),
            collective_boost: 1.0,
            participant_count: 1,
            synergy: 0.0,
            threshold_observations: HashMap::new(),
        }
    }

    /// Contribute a threshold observation
    pub fn contribute_threshold(&mut self, task: TaskComplexity, coherence: f32, success: bool) {
        let key = format!("{task:?}");
        self.threshold_observations
            .entry(key)
            .or_default()
            .push((coherence, success));
    }

    /// Query average threshold for a task
    pub fn query_threshold_average(&self, task: TaskComplexity) -> Option<f32> {
        let key = format!("{task:?}");
        let observations = self.threshold_observations.get(&key)?;

        if observations.is_empty() {
            return None;
        }

        // Find the minimum coherence that led to success
        let successful: Vec<f32> = observations
            .iter()
            .filter(|(_, success)| *success)
            .map(|(coherence, _)| *coherence)
            .collect();

        if successful.is_empty() {
            return None;
        }

        Some(successful.iter().sum::<f32>() / successful.len() as f32)
    }

    /// Add a participant
    pub fn add_participant(&mut self) {
        self.participant_count += 1;
        self.update_synergy();
    }

    /// Remove a participant
    pub fn remove_participant(&mut self) {
        self.participant_count = self.participant_count.saturating_sub(1);
        self.update_synergy();
    }

    /// Update synergy based on participant count
    fn update_synergy(&mut self) {
        if self.participant_count == 0 {
            self.synergy = 0.0;
            self.collective_boost = 1.0;
        } else {
            // Synergy grows logarithmically with participants
            self.synergy = (self.participant_count as f64).ln().max(0.0);
            self.collective_boost = 1.0 + 0.1 * self.synergy;
        }
    }

    /// Share knowledge with the collective (single knowledge item)
    pub fn share_knowledge(&mut self, knowledge: impl Into<String>) {
        self.knowledge_pool.push(knowledge.into());
        // More knowledge increases collective boost
        self.collective_boost += 0.01;
    }

    /// Merge knowledge from another CollectiveLearning instance
    pub fn merge_from(&mut self, other: &CollectiveLearning) {
        // Merge knowledge pool
        self.knowledge_pool
            .extend(other.knowledge_pool.iter().cloned());

        // Merge threshold observations
        for (complexity, observations) in &other.threshold_observations {
            self.threshold_observations
                .entry(complexity.clone())
                .or_default()
                .extend(observations.iter().cloned());
        }

        // Update participant count and synergy
        self.participant_count = self
            .participant_count
            .saturating_add(other.participant_count);
        self.synergy = (self.participant_count as f64).ln().max(0.0);
        self.collective_boost = 1.0 + 0.1 * self.synergy + 0.01 * other.knowledge_pool.len() as f64;
    }

    /// Get learning rate modifier
    pub fn learning_modifier(&self) -> f64 {
        self.collective_boost * (1.0 + self.synergy * 0.1)
    }

    /// Get statistics about collective learning
    /// Returns (task_types_count, total_observations, total_contributors)
    pub fn get_stats(&self) -> (usize, usize, usize) {
        let task_types = self.threshold_observations.len();
        let total_observations: usize = self.threshold_observations.values().map(|v| v.len()).sum();
        (task_types, total_observations, self.participant_count)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Collective Primitive Evolution
// ─────────────────────────────────────────────────────────────────────────────

use crate::consciousness::primitive_evolution::CandidatePrimitive;
use crate::hdc::primitive_system::PrimitiveTier;

/// Collective primitive sharing and evolution across system instances
///
/// This struct manages a shared pool of evolved primitives that can be
/// contributed to and queried by multiple reasoning instances.
#[derive(Debug, Clone)]
pub struct CollectivePrimitiveEvolution {
    /// System identifier
    pub system_id: String,
    /// Shared primitives in the collective pool
    primitives: Vec<(CandidatePrimitive, f32)>, // (primitive, score)
    /// Total contributions count
    contributions: usize,
    /// Successful applications
    successes: usize,
}

impl CollectivePrimitiveEvolution {
    /// Create a new collective primitive evolution system
    pub fn new(system_id: impl Into<String>) -> Self {
        Self {
            system_id: system_id.into(),
            primitives: Vec::new(),
            contributions: 0,
            successes: 0,
        }
    }

    /// Contribute a primitive to the collective pool
    pub fn contribute_primitive(
        &mut self,
        primitive: CandidatePrimitive,
        success: bool,
        phi_improvement: f32,
        harmonic_score: f32,
        epistemic_score: f32,
    ) {
        // Calculate overall score for ranking
        let score = if success {
            0.4 * phi_improvement.max(0.0) + 0.3 * harmonic_score + 0.3 * epistemic_score
        } else {
            0.1 * harmonic_score + 0.1 * epistemic_score
        };

        self.primitives.push((primitive, score));
        self.contributions += 1;
        if success {
            self.successes += 1;
        }

        // Keep pool manageable (top 1000 by score)
        if self.primitives.len() > 1000 {
            self.primitives
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            self.primitives.truncate(1000);
        }
    }

    /// Query top primitives of a given tier
    pub fn query_top_primitives(
        &self,
        tier: PrimitiveTier,
        count: usize,
    ) -> Vec<CandidatePrimitive> {
        let mut matching: Vec<_> = self
            .primitives
            .iter()
            .filter(|(p, _)| p.tier == tier)
            .collect();

        matching.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        matching
            .into_iter()
            .take(count)
            .map(|(p, _)| p.clone())
            .collect()
    }

    /// Merge knowledge from another collective
    pub fn merge_knowledge(&mut self, other: &CollectivePrimitiveEvolution) {
        for (prim, score) in &other.primitives {
            self.primitives.push((prim.clone(), *score));
        }
        self.contributions += other.contributions;
        self.successes += other.successes;

        // Deduplicate and keep top
        if self.primitives.len() > 1000 {
            self.primitives
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            self.primitives.truncate(1000);
        }
    }

    /// Get statistics: (success_rate, primitive_count, contribution_count)
    pub fn get_stats(&self) -> (f64, usize, usize) {
        let success_rate = if self.contributions > 0 {
            self.successes as f64 / self.contributions as f64
        } else {
            0.0
        };
        (success_rate, self.primitives.len(), self.contributions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coherence_beacon() {
        let beacon = CoherenceBeacon::new("test", 0.8);
        assert_eq!(beacon.source_id, "test");
        assert!((beacon.coherence_level - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_social_coherence_field() {
        let mut field = SocialCoherenceField::new("instance1");
        field.add_contribution("peer1", 0.8);
        field.add_contribution("peer2", 0.6);
        assert!(field.strength() > 0.0);
    }

    #[test]
    fn test_collective_learning() {
        let mut learning = CollectiveLearning::new("instance1");
        learning.add_participant();
        learning.add_participant();
        assert!(learning.synergy > 0.0);
        assert!(learning.collective_boost > 1.0);
    }

    #[test]
    fn test_lending_protocol() {
        let protocol = CoherenceLendingProtocol::new("instance1");
        assert!(protocol.can_lend(0.1, 0.8));
        assert!(!protocol.can_lend(0.5, 0.3)); // Not enough coherence
    }
}
