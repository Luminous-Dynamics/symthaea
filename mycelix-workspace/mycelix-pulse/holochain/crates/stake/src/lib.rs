// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Stake-Based Trust System for Mycelix Mail
//!
//! Implements stake requirements for trust attestations to prevent Sybil attacks.
//! Agents must stake reputation to vouch for others, creating economic disincentives
//! for malicious attestations.

use hdi::prelude::*;
use hdk::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Stake system errors
#[derive(Error, Debug)]
pub enum StakeError {
    #[error("Insufficient stake: required {required}, available {available}")]
    InsufficientStake { required: u64, available: u64 },

    #[error("Stake already exists for this attestation")]
    StakeExists,

    #[error("Stake not found")]
    StakeNotFound,

    #[error("Cannot withdraw: stake is locked until {unlock_time}")]
    StakeLocked { unlock_time: u64 },

    #[error("Invalid stake amount: {0}")]
    InvalidAmount(String),

    #[error("Attestation validation failed: {0}")]
    ValidationFailed(String),
}

impl From<StakeError> for WasmError {
    fn from(e: StakeError) -> Self {
        wasm_error!(WasmErrorInner::Guest(e.to_string()))
    }
}

/// Stake configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StakeConfig {
    /// Minimum stake required for a basic attestation
    pub min_stake: u64,

    /// Maximum stake that can be applied to a single attestation
    pub max_stake: u64,

    /// Stake required per trust level increment (e.g., 100 per 0.1 trust)
    pub stake_per_trust_unit: u64,

    /// Lock period in seconds before stake can be withdrawn
    pub lock_period_secs: u64,

    /// Penalty percentage for negative attestation outcomes (0-100)
    pub penalty_percentage: u8,

    /// Initial stake granted to new agents
    pub initial_stake: u64,

    /// Daily stake regeneration rate
    pub daily_regeneration: u64,

    /// Maximum total stake an agent can hold
    pub max_total_stake: u64,
}

impl Default for StakeConfig {
    fn default() -> Self {
        Self {
            min_stake: 10,
            max_stake: 1000,
            stake_per_trust_unit: 100, // 100 stake per 0.1 trust level
            lock_period_secs: 7 * 24 * 60 * 60, // 7 days
            penalty_percentage: 50,
            initial_stake: 100,
            daily_regeneration: 10,
            max_total_stake: 10000,
        }
    }
}

impl StakeConfig {
    /// Configuration for high-security deployments
    pub fn high_security() -> Self {
        Self {
            min_stake: 50,
            max_stake: 5000,
            stake_per_trust_unit: 200,
            lock_period_secs: 30 * 24 * 60 * 60, // 30 days
            penalty_percentage: 75,
            initial_stake: 50,
            daily_regeneration: 5,
            max_total_stake: 20000,
        }
    }

    /// Configuration for testing/development
    pub fn development() -> Self {
        Self {
            min_stake: 1,
            max_stake: 1000,
            stake_per_trust_unit: 10,
            lock_period_secs: 60, // 1 minute
            penalty_percentage: 10,
            initial_stake: 1000,
            daily_regeneration: 100,
            max_total_stake: 100000,
        }
    }

    /// Calculate required stake for a given trust level
    pub fn required_stake_for_trust(&self, trust_level: f64) -> u64 {
        let trust_units = (trust_level * 10.0).ceil() as u64;
        let calculated = trust_units * self.stake_per_trust_unit;
        calculated.clamp(self.min_stake, self.max_stake)
    }

    /// Calculate stake weight for trust calculation
    pub fn stake_weight(&self, stake_amount: u64) -> f64 {
        // Logarithmic scaling to prevent plutocracy
        let normalized = (stake_amount as f64) / (self.max_stake as f64);
        (1.0 + normalized).ln() / (2.0_f64).ln()
    }
}

/// Agent's stake balance
#[hdk_entry_helper]
#[derive(Clone)]
pub struct StakeBalance {
    /// Agent this balance belongs to
    pub agent: AgentPubKey,

    /// Total available stake
    pub available: u64,

    /// Currently locked stake
    pub locked: u64,

    /// Pending stake (waiting for lock period)
    pub pending: u64,

    /// Timestamp of last balance update
    pub last_updated: u64,

    /// Timestamp of last regeneration
    pub last_regeneration: u64,
}

impl StakeBalance {
    /// Create initial balance for a new agent
    pub fn new(agent: AgentPubKey, initial: u64) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            agent,
            available: initial,
            locked: 0,
            pending: 0,
            last_updated: now,
            last_regeneration: now,
        }
    }

    /// Total stake (available + locked)
    pub fn total(&self) -> u64 {
        self.available + self.locked + self.pending
    }

    /// Check if agent can afford a stake
    pub fn can_afford(&self, amount: u64) -> bool {
        self.available >= amount
    }

    /// Lock stake for an attestation
    pub fn lock(&mut self, amount: u64) -> Result<(), StakeError> {
        if !self.can_afford(amount) {
            return Err(StakeError::InsufficientStake {
                required: amount,
                available: self.available,
            });
        }

        self.available -= amount;
        self.locked += amount;
        self.last_updated = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(())
    }

    /// Unlock stake (after lock period)
    pub fn unlock(&mut self, amount: u64) {
        let to_unlock = amount.min(self.locked);
        self.locked -= to_unlock;
        self.available += to_unlock;
        self.last_updated = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }

    /// Apply penalty (slash stake)
    pub fn apply_penalty(&mut self, amount: u64) {
        let to_slash = amount.min(self.locked);
        self.locked -= to_slash;
        // Slashed stake is burned (removed from circulation)
        self.last_updated = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }

    /// Regenerate stake based on time elapsed
    pub fn regenerate(&mut self, config: &StakeConfig) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let days_elapsed = (now - self.last_regeneration) / (24 * 60 * 60);
        if days_elapsed > 0 {
            let regeneration = days_elapsed * config.daily_regeneration;
            self.available = (self.available + regeneration).min(config.max_total_stake - self.locked);
            self.last_regeneration = now;
        }
    }
}

/// Stake record for an attestation
#[hdk_entry_helper]
#[derive(Clone)]
pub struct StakeRecord {
    /// Unique ID for this stake
    pub id: String,

    /// Agent who staked
    pub staker: AgentPubKey,

    /// Target agent of the attestation
    pub target: AgentPubKey,

    /// Amount staked
    pub amount: u64,

    /// Trust level claimed in attestation
    pub trust_level: f64,

    /// When the stake was created
    pub created_at: u64,

    /// When the stake can be unlocked
    pub unlock_at: u64,

    /// Current status
    pub status: StakeStatus,

    /// Hash of the attestation entry
    pub attestation_hash: ActionHash,
}

/// Stake status
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum StakeStatus {
    /// Stake is active and locked
    Active,

    /// Stake is pending unlock (lock period elapsed)
    PendingUnlock,

    /// Stake has been unlocked and returned
    Unlocked,

    /// Stake was slashed due to negative outcome
    Slashed { penalty_amount: u64 },

    /// Stake was revoked by staker
    Revoked,
}

/// Stake manager for handling stake operations
pub struct StakeManager {
    config: StakeConfig,
}

impl StakeManager {
    /// Create a new stake manager with the given configuration
    pub fn new(config: StakeConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(StakeConfig::default())
    }

    /// Get required stake for a trust attestation
    pub fn required_stake(&self, trust_level: f64) -> u64 {
        self.config.required_stake_for_trust(trust_level)
    }

    /// Validate that an attestation has sufficient stake
    pub fn validate_attestation_stake(
        &self,
        staker_balance: &StakeBalance,
        trust_level: f64,
        stake_amount: u64,
    ) -> Result<(), StakeError> {
        let required = self.required_stake(trust_level);

        if stake_amount < required {
            return Err(StakeError::InsufficientStake {
                required,
                available: stake_amount,
            });
        }

        if stake_amount > self.config.max_stake {
            return Err(StakeError::InvalidAmount(format!(
                "Stake {} exceeds maximum {}",
                stake_amount, self.config.max_stake
            )));
        }

        if !staker_balance.can_afford(stake_amount) {
            return Err(StakeError::InsufficientStake {
                required: stake_amount,
                available: staker_balance.available,
            });
        }

        Ok(())
    }

    /// Create a stake record for an attestation
    pub fn create_stake(
        &self,
        staker: AgentPubKey,
        target: AgentPubKey,
        trust_level: f64,
        stake_amount: u64,
        attestation_hash: ActionHash,
    ) -> StakeRecord {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        StakeRecord {
            id: format!("{:?}-{}", staker, now),
            staker,
            target,
            amount: stake_amount,
            trust_level,
            created_at: now,
            unlock_at: now + self.config.lock_period_secs,
            status: StakeStatus::Active,
            attestation_hash,
        }
    }

    /// Check if stake can be unlocked
    pub fn can_unlock(&self, stake: &StakeRecord) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        stake.status == StakeStatus::Active && now >= stake.unlock_at
    }

    /// Calculate penalty for a bad attestation
    pub fn calculate_penalty(&self, stake_amount: u64) -> u64 {
        (stake_amount * self.config.penalty_percentage as u64) / 100
    }

    /// Calculate stake weight for trust calculation
    pub fn stake_weight(&self, stake_amount: u64) -> f64 {
        self.config.stake_weight(stake_amount)
    }

    /// Get the configuration
    pub fn config(&self) -> &StakeConfig {
        &self.config
    }
}

/// Validation callback for attestation entries
/// Call this from the trust zome's validation
pub fn validate_attestation_with_stake(
    attestation_trust_level: f64,
    claimed_stake: u64,
    _staker: &AgentPubKey,
) -> ExternResult<ValidateCallbackResult> {
    let manager = StakeManager::default_config();

    // Check minimum stake requirement
    let required = manager.required_stake(attestation_trust_level);
    if claimed_stake < required {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Insufficient stake: {} required, {} provided",
            required, claimed_stake
        )));
    }

    // In a full implementation, we'd verify the staker's balance
    // by querying their stake balance entry

    Ok(ValidateCallbackResult::Valid)
}

/// Helper to initialize stake for a new agent
pub fn initialize_agent_stake(agent: AgentPubKey) -> ExternResult<StakeBalance> {
    let config = StakeConfig::default();
    let balance = StakeBalance::new(agent, config.initial_stake);
    // In production, commit this as an entry
    Ok(balance)
}

/// Query stake balance for an agent
pub fn get_stake_balance(agent: &AgentPubKey) -> ExternResult<StakeBalance> {
    // Query source chain for stake balance entry
    // For now, return a default balance
    let config = StakeConfig::default();
    Ok(StakeBalance::new(agent.clone(), config.initial_stake))
}

/// Query all active stakes for an agent
pub fn get_active_stakes(_agent: &AgentPubKey) -> ExternResult<Vec<StakeRecord>> {
    // Query DHT for stake records
    Ok(vec![])
}

/// Get total stake locked by an agent
pub fn get_locked_stake(_agent: &AgentPubKey) -> ExternResult<u64> {
    // Aggregate from stake records
    Ok(0)
}

/// Stake statistics for monitoring
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StakeStats {
    pub total_staked: u64,
    pub total_locked: u64,
    pub total_slashed: u64,
    pub active_stakes: u32,
    pub unique_stakers: u32,
}

/// Get global stake statistics
pub fn get_stake_stats() -> ExternResult<StakeStats> {
    // Aggregate from all stake records
    Ok(StakeStats {
        total_staked: 0,
        total_locked: 0,
        total_slashed: 0,
        active_stakes: 0,
        unique_stakers: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_required_stake_calculation() {
        let config = StakeConfig::default();

        // Trust 0.0 should require minimum
        assert_eq!(config.required_stake_for_trust(0.0), 10);

        // Trust 0.5 should require 500
        assert_eq!(config.required_stake_for_trust(0.5), 500);

        // Trust 1.0 should require max (clamped to 1000)
        assert_eq!(config.required_stake_for_trust(1.0), 1000);
    }

    #[test]
    fn test_stake_balance_lock() {
        let agent = AgentPubKey::from_raw_36(vec![0u8; 36]);
        let mut balance = StakeBalance::new(agent, 100);

        assert!(balance.can_afford(50));
        assert!(balance.lock(50).is_ok());
        assert_eq!(balance.available, 50);
        assert_eq!(balance.locked, 50);

        assert!(balance.lock(100).is_err());
    }

    #[test]
    fn test_stake_weight() {
        let config = StakeConfig::default();

        // No stake = weight 0
        let weight_zero = config.stake_weight(0);
        assert!(weight_zero < 0.1);

        // Max stake = weight 1
        let weight_max = config.stake_weight(1000);
        assert!(weight_max > 0.9 && weight_max <= 1.0);
    }

    #[test]
    fn test_penalty_calculation() {
        let manager = StakeManager::default_config();

        // Default penalty is 50%
        assert_eq!(manager.calculate_penalty(100), 50);
        assert_eq!(manager.calculate_penalty(1000), 500);
    }
}
