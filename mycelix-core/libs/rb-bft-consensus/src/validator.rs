// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Validator types and management for RB-BFT consensus

use serde::{Deserialize, Serialize};
use mycelix_core_types::{KVector, TrustScore};

use crate::error::{ConsensusError, ConsensusResult};
use crate::MIN_PARTICIPATION_REPUTATION;

/// A validator node in the consensus protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorNode {
    /// Unique identifier (typically agent pubkey)
    pub id: String,
    /// K-Vector trust representation
    pub k_vector: KVector,
    /// MATL trust score
    pub trust_score: TrustScore,
    /// Whether this validator is currently active
    pub active: bool,
    /// Number of successful rounds participated
    pub successful_rounds: u64,
    /// Number of failed/missed rounds
    pub failed_rounds: u64,
    /// Last activity timestamp (Unix epoch seconds)
    pub last_active: i64,
    /// Slashing record (number of times slashed)
    pub slash_count: u32,
}

impl ValidatorNode {
    /// Create a new validator with initial trust
    pub fn new(id: String) -> Self {
        Self {
            id,
            k_vector: KVector::neutral(),
            trust_score: TrustScore::default(),
            active: true,
            successful_rounds: 0,
            failed_rounds: 0,
            last_active: 0,
            slash_count: 0,
        }
    }

    /// Get the reputation score (primary component of K-Vector)
    pub fn reputation(&self) -> f32 {
        self.k_vector.k_r
    }

    /// Get the voting weight (reputation squared)
    ///
    /// This is the key innovation of RB-BFT: by squaring reputation,
    /// low-reputation nodes have diminished influence, enabling
    /// higher Byzantine tolerance.
    pub fn voting_weight(&self) -> f32 {
        self.reputation().powi(2)
    }

    /// Check if this validator can participate in consensus
    pub fn can_participate(&self) -> ConsensusResult<()> {
        if !self.active {
            return Err(ConsensusError::InvalidRoundState {
                expected: "active validator".to_string(),
                actual: "inactive validator".to_string(),
            });
        }

        if self.reputation() < MIN_PARTICIPATION_REPUTATION {
            return Err(ConsensusError::ReputationTooLow {
                reputation: self.reputation(),
                minimum: MIN_PARTICIPATION_REPUTATION,
            });
        }

        Ok(())
    }

    /// Update reputation based on round outcome
    pub fn update_reputation(&mut self, success: bool) {
        const LEARNING_RATE: f32 = 0.1;

        if success {
            self.successful_rounds += 1;
            // Increase reputation towards 1.0
            self.k_vector.k_r = self.k_vector.k_r + LEARNING_RATE * (1.0 - self.k_vector.k_r);
        } else {
            self.failed_rounds += 1;
            // Decrease reputation towards 0.0
            self.k_vector.k_r *= 1.0 - LEARNING_RATE;
        }

        // Update activity score
        self.k_vector.k_a = (self.k_vector.k_a + 0.1).min(1.0);
    }

    /// Apply slashing penalty
    pub fn slash(&mut self, severity: SlashingSeverity) {
        self.slash_count += 1;

        let penalty = match severity {
            SlashingSeverity::Minor => 0.1,
            SlashingSeverity::Moderate => 0.3,
            SlashingSeverity::Severe => 0.5,
            SlashingSeverity::Critical => 0.8,
        };

        // Apply penalty to reputation
        self.k_vector.k_r *= 1.0 - penalty;

        // Also affect integrity score
        self.k_vector.k_i *= 1.0 - penalty * 0.5;

        // Deactivate if reputation falls too low
        if self.k_vector.k_r < 0.05 {
            self.active = false;
        }
    }

    /// Calculate participation rate
    pub fn participation_rate(&self) -> f32 {
        let total = self.successful_rounds + self.failed_rounds;
        if total == 0 {
            return 0.0;
        }
        self.successful_rounds as f32 / total as f32
    }
}

/// Severity levels for slashing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlashingSeverity {
    /// Minor offense (e.g., slow response)
    Minor,
    /// Moderate offense (e.g., invalid vote)
    Moderate,
    /// Severe offense (e.g., double voting)
    Severe,
    /// Critical offense (e.g., provable Byzantine attack)
    Critical,
}

/// Set of validators for a consensus round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorSet {
    /// All validators in the set
    validators: Vec<ValidatorNode>,
    /// Total voting weight (sum of reputation²)
    total_weight: f32,
}

impl ValidatorSet {
    /// Create a new empty validator set
    pub fn new() -> Self {
        Self {
            validators: Vec::new(),
            total_weight: 0.0,
        }
    }

    /// Create from a list of validators
    pub fn from_validators(validators: Vec<ValidatorNode>) -> Self {
        let total_weight = validators.iter()
            .filter(|v| v.active && v.reputation() >= MIN_PARTICIPATION_REPUTATION)
            .map(|v| v.voting_weight())
            .sum();

        Self {
            validators,
            total_weight,
        }
    }

    /// Add a validator to the set
    pub fn add(&mut self, validator: ValidatorNode) {
        if validator.active && validator.reputation() >= MIN_PARTICIPATION_REPUTATION {
            self.total_weight += validator.voting_weight();
        }
        self.validators.push(validator);
    }

    /// Get a validator by ID
    pub fn get(&self, id: &str) -> Option<&ValidatorNode> {
        self.validators.iter().find(|v| v.id == id)
    }

    /// Get a mutable reference to a validator by ID
    pub fn get_mut(&mut self, id: &str) -> Option<&mut ValidatorNode> {
        self.validators.iter_mut().find(|v| v.id == id)
    }

    /// Get all active validators
    pub fn active_validators(&self) -> Vec<&ValidatorNode> {
        self.validators.iter()
            .filter(|v| v.active && v.reputation() >= MIN_PARTICIPATION_REPUTATION)
            .collect()
    }

    /// Get count of active validators
    pub fn active_count(&self) -> usize {
        self.active_validators().len()
    }

    /// Get total voting weight
    pub fn total_weight(&self) -> f32 {
        self.total_weight
    }

    /// Recalculate total weight (call after modifying validators)
    pub fn recalculate_weight(&mut self) {
        self.total_weight = self.validators.iter()
            .filter(|v| v.active && v.reputation() >= MIN_PARTICIPATION_REPUTATION)
            .map(|v| v.voting_weight())
            .sum();
    }

    /// Select leader for a round using weighted random selection
    ///
    /// Leader is selected proportional to voting weight.
    pub fn select_leader(&self, round: u64) -> Option<&ValidatorNode> {
        let active = self.active_validators();
        if active.is_empty() {
            return None;
        }

        // Deterministic selection based on round number
        // In production, use VRF (Verifiable Random Function)
        let index = (round as usize) % active.len();
        Some(active[index])
    }

    /// Calculate the threshold weight needed for consensus
    ///
    /// With reputation² weighting and 45% Byzantine tolerance,
    /// we need > 55% of weighted votes.
    pub fn consensus_threshold(&self) -> f32 {
        self.total_weight * 0.55
    }

    /// Check if validators can reach consensus
    pub fn can_reach_consensus(&self) -> ConsensusResult<()> {
        let active_count = self.active_count();
        if active_count < crate::MIN_VALIDATORS {
            return Err(ConsensusError::InsufficientValidators {
                have: active_count,
                need: crate::MIN_VALIDATORS,
            });
        }
        Ok(())
    }

    /// Get the number of validators in the set (including inactive)
    pub fn len(&self) -> usize {
        self.validators.len()
    }

    /// Check if the validator set is empty
    pub fn is_empty(&self) -> bool {
        self.validators.is_empty()
    }

    /// Get total voting weight (alias for total_weight)
    ///
    /// Sum of reputation² for all active, eligible validators.
    pub fn total_voting_weight(&self) -> f32 {
        self.total_weight
    }

    /// Iterator over all validators
    pub fn iter(&self) -> impl Iterator<Item = &ValidatorNode> {
        self.validators.iter()
    }
}

impl Default for ValidatorSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_validator(id: &str, reputation: f32) -> ValidatorNode {
        let mut v = ValidatorNode::new(id.to_string());
        v.k_vector.k_r = reputation;
        v
    }

    #[test]
    fn test_voting_weight_is_reputation_squared() {
        let v = create_test_validator("test", 0.8);
        assert!((v.voting_weight() - 0.64).abs() < 0.001); // 0.8² = 0.64
    }

    #[test]
    fn test_low_reputation_cannot_participate() {
        let mut v = create_test_validator("test", 0.05);
        v.active = true;
        assert!(v.can_participate().is_err());
    }

    #[test]
    fn test_reputation_update_on_success() {
        let mut v = create_test_validator("test", 0.5);
        v.update_reputation(true);
        assert!(v.reputation() > 0.5);
    }

    #[test]
    fn test_reputation_update_on_failure() {
        let mut v = create_test_validator("test", 0.5);
        v.update_reputation(false);
        assert!(v.reputation() < 0.5);
    }

    #[test]
    fn test_slashing_reduces_reputation() {
        let mut v = create_test_validator("test", 0.8);
        v.slash(SlashingSeverity::Moderate);
        assert!(v.reputation() < 0.8);
    }

    #[test]
    fn test_validator_set_weight() {
        let mut set = ValidatorSet::new();
        set.add(create_test_validator("a", 0.8)); // weight = 0.64
        set.add(create_test_validator("b", 0.6)); // weight = 0.36
        // Total = 1.0

        assert!((set.total_weight() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_consensus_threshold() {
        let mut set = ValidatorSet::new();
        set.add(create_test_validator("a", 1.0));
        set.add(create_test_validator("b", 1.0));
        // Total weight = 2.0, threshold = 1.1 (55%)

        assert!((set.consensus_threshold() - 1.1).abs() < 0.01);
    }
}
