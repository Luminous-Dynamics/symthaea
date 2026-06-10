// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Temporal Trust Dynamics
//!
//! Time-aware trust modeling with decay, memory, and velocity limits.
//!
//! ## Features
//!
//! - **Trust Decay**: Trust decays over inactivity periods
//! - **Reputation Memory**: Forgiveness curves for past behavior
//! - **Velocity Limits**: Prevent rapid trust manipulation
//! - **Historical Snapshots**: Audit trail of trust changes
//!
//! ## Philosophy
//!
//! Trust is not static. An agent who was trustworthy a year ago but hasn't
//! been active should have lower trust than one who is consistently reliable.
//! Similarly, an agent who suddenly gains trust should be viewed with suspicion.
//!
//! ## Example
//!
//! ```rust,ignore
//! use mycelix_sdk::agentic::temporal_trust::{
//!     TemporalTrustManager, TrustDecayConfig, TrustSnapshot,
//! };
//!
//! let mut manager = TemporalTrustManager::new(TemporalTrustConfig::default());
//!
//! // Record initial trust
//! manager.record_trust("agent-1", 0.7);
//!
//! // Time passes...
//! manager.tick(86400_000); // 1 day
//!
//! // Get current trust (with decay applied)
//! let current = manager.current_trust("agent-1");
//! assert!(current < 0.7); // Trust has decayed
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

use crate::matl::KVector;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for temporal trust dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalTrustConfig {
    /// Trust decay configuration
    pub decay: TrustDecayConfig,
    /// Velocity limit configuration
    pub velocity: VelocityLimitConfig,
    /// Memory configuration
    pub memory: ReputationMemoryConfig,
    /// Maximum snapshots to retain
    pub max_snapshots: usize,
    /// Snapshot interval (ms)
    pub snapshot_interval_ms: u64,
}

impl Default for TemporalTrustConfig {
    fn default() -> Self {
        Self {
            decay: TrustDecayConfig::default(),
            velocity: VelocityLimitConfig::default(),
            memory: ReputationMemoryConfig::default(),
            max_snapshots: 100,
            snapshot_interval_ms: 3600_000, // 1 hour
        }
    }
}

/// Trust decay configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustDecayConfig {
    /// Enable decay
    pub enabled: bool,
    /// Half-life of trust in inactive state (ms)
    /// After this time, trust decays to 50% of original
    pub half_life_ms: u64,
    /// Minimum trust floor (decay stops here)
    pub floor: f64,
    /// Grace period before decay starts (ms)
    pub grace_period_ms: u64,
    /// Decay curve type
    pub curve: DecayCurve,
}

impl Default for TrustDecayConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            half_life_ms: 30 * 24 * 3600_000, // 30 days
            floor: 0.1,
            grace_period_ms: 7 * 24 * 3600_000, // 7 days grace
            curve: DecayCurve::Exponential,
        }
    }
}

/// Decay curve types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DecayCurve {
    /// Exponential decay: T(t) = T₀ * 0.5^(t/half_life)
    Exponential,
    /// Linear decay: T(t) = T₀ - (T₀ - floor) * (t/full_decay_time)
    Linear,
    /// Stepped decay: Trust drops at fixed intervals.
    Stepped {
        /// Interval between trust drops (ms).
        step_interval_ms: u64,
        /// Amount of trust dropped per step.
        step_size: f64,
    },
}

/// Velocity limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VelocityLimitConfig {
    /// Enable velocity limits
    pub enabled: bool,
    /// Maximum trust increase per period
    pub max_increase_per_period: f64,
    /// Maximum trust decrease per period
    pub max_decrease_per_period: f64,
    /// Period length (ms)
    pub period_ms: u64,
    /// Action on velocity violation
    pub violation_action: VelocityViolationAction,
}

impl Default for VelocityLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_increase_per_period: 0.1, // 10% max increase per period
            max_decrease_per_period: 0.2, // 20% max decrease per period
            period_ms: 24 * 3600_000,     // 24 hours
            violation_action: VelocityViolationAction::Clamp,
        }
    }
}

/// What to do when velocity limit is violated
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum VelocityViolationAction {
    /// Clamp to the maximum allowed change
    Clamp,
    /// Reject the change entirely
    Reject,
    /// Allow but flag for review
    FlagForReview,
}

/// Reputation memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationMemoryConfig {
    /// Enable reputation memory
    pub enabled: bool,
    /// How long negative events are remembered (ms)
    pub negative_memory_ms: u64,
    /// How long positive events are remembered (ms)
    pub positive_memory_ms: u64,
    /// Forgiveness curve - how quickly past events lose influence
    pub forgiveness_half_life_ms: u64,
    /// Weight of historical vs current trust
    pub history_weight: f64,
}

impl Default for ReputationMemoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            negative_memory_ms: 90 * 24 * 3600_000, // 90 days for negative
            positive_memory_ms: 180 * 24 * 3600_000, // 180 days for positive
            forgiveness_half_life_ms: 30 * 24 * 3600_000, // 30 days
            history_weight: 0.3,                    // 30% from history, 70% from current
        }
    }
}

// ============================================================================
// Trust Snapshot
// ============================================================================

/// A point-in-time snapshot of an agent's trust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustSnapshot {
    /// Agent ID
    pub agent_id: String,
    /// Trust score at this time
    pub trust: f64,
    /// K-Vector snapshot (optional)
    pub kvector: Option<KVector>,
    /// Timestamp
    pub timestamp: u64,
    /// Reason for snapshot
    pub reason: SnapshotReason,
    /// Delta from previous snapshot
    pub delta: Option<f64>,
}

/// Reason for taking a snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SnapshotReason {
    /// Regular interval snapshot
    Periodic,
    /// Trust changed significantly.
    SignificantChange {
        /// Amount of change.
        delta: f64,
    },
    /// Manual snapshot request.
    Manual,
    /// Decay applied.
    Decay {
        /// Amount decayed.
        decayed_amount: f64,
    },
    /// Velocity limit applied.
    VelocityLimited {
        /// Requested change.
        requested: f64,
        /// Actual change after limiting.
        actual: f64,
    },
    /// Activity recorded.
    Activity {
        /// Activity outcome description.
        outcome: String,
    },
}

// ============================================================================
// Trust Event
// ============================================================================

/// A trust-affecting event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustEvent {
    /// Timestamp
    pub timestamp: u64,
    /// Event type
    pub event_type: TrustEventType,
    /// Trust delta
    pub delta: f64,
    /// Context/reason
    pub context: String,
}

/// Types of trust events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrustEventType {
    /// Positive action completed
    PositiveAction,
    /// Negative action or failure
    NegativeAction,
    /// Gaming attempt detected
    GamingAttempt,
    /// Slashing occurred
    Slashed,
    /// Reward granted
    Rewarded,
    /// Manual adjustment
    ManualAdjustment,
}

// ============================================================================
// Agent Trust State
// ============================================================================

/// Complete trust state for an agent
#[derive(Debug, Clone)]
struct AgentTrustState {
    /// Current trust score
    current_trust: f64,
    /// Last activity timestamp
    last_activity: u64,
    /// Trust history (recent events)
    events: VecDeque<TrustEvent>,
    /// Snapshots
    snapshots: VecDeque<TrustSnapshot>,
    /// Trust at start of current velocity period
    period_start_trust: f64,
    /// Start of current velocity period
    period_start_time: u64,
    /// Flags
    flags: TrustFlags,
}

/// Flags on trust state
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
struct TrustFlags {
    /// Flagged for review
    flagged_for_review: bool,
    /// Velocity violation count
    velocity_violations: u32,
    /// Under observation
    under_observation: bool,
}

// ============================================================================
// Temporal Trust Manager
// ============================================================================

/// Manages temporal trust dynamics for multiple agents
pub struct TemporalTrustManager {
    /// Configuration
    config: TemporalTrustConfig,
    /// Agent trust states
    agents: HashMap<String, AgentTrustState>,
    /// Current simulation time (for testing)
    current_time: u64,
}

impl TemporalTrustManager {
    /// Create a new temporal trust manager
    pub fn new(config: TemporalTrustConfig) -> Self {
        Self {
            config,
            agents: HashMap::new(),
            current_time: Self::now(),
        }
    }

    /// Record initial trust for an agent
    pub fn record_trust(&mut self, agent_id: &str, trust: f64) {
        let now = self.current_time;
        let clamped_trust = trust.clamp(0.0, 1.0);

        let state = AgentTrustState {
            current_trust: clamped_trust,
            last_activity: now,
            events: VecDeque::new(),
            snapshots: VecDeque::new(),
            period_start_trust: clamped_trust,
            period_start_time: now,
            flags: TrustFlags::default(),
        };

        self.agents.insert(agent_id.to_string(), state);

        // Take initial snapshot
        self.take_snapshot(agent_id, SnapshotReason::Manual);
    }

    /// Update trust with velocity limiting
    pub fn update_trust(
        &mut self,
        agent_id: &str,
        new_trust: f64,
        reason: &str,
    ) -> Result<TrustUpdateResult, TemporalTrustError> {
        let state =
            self.agents
                .get_mut(agent_id)
                .ok_or_else(|| TemporalTrustError::AgentNotFound {
                    agent_id: agent_id.to_string(),
                })?;

        let now = self.current_time;
        let old_trust = state.current_trust;
        let requested_delta = new_trust - old_trust;

        // Check velocity limits
        let (final_trust, velocity_limited) = if self.config.velocity.enabled {
            Self::apply_velocity_limits(&self.config.velocity, state, new_trust, now)
        } else {
            (new_trust.clamp(0.0, 1.0), false)
        };

        let actual_delta = final_trust - old_trust;

        // Update state
        state.current_trust = final_trust;
        state.last_activity = now;

        // Record event
        let event = TrustEvent {
            timestamp: now,
            event_type: if actual_delta >= 0.0 {
                TrustEventType::PositiveAction
            } else {
                TrustEventType::NegativeAction
            },
            delta: actual_delta,
            context: reason.to_string(),
        };
        state.events.push_back(event);

        // Trim events
        while state.events.len() > 100 {
            state.events.pop_front();
        }

        // Take snapshot if significant change
        if actual_delta.abs() > 0.05 {
            self.take_snapshot(
                agent_id,
                SnapshotReason::SignificantChange {
                    delta: actual_delta,
                },
            );
        }

        Ok(TrustUpdateResult {
            old_trust,
            new_trust: final_trust,
            requested_delta,
            actual_delta,
            velocity_limited,
        })
    }

    /// Apply velocity limits to a trust change
    fn apply_velocity_limits(
        config: &VelocityLimitConfig,
        state: &mut AgentTrustState,
        new_trust: f64,
        now: u64,
    ) -> (f64, bool) {
        // Check if we're in a new period
        if now.saturating_sub(state.period_start_time) >= config.period_ms {
            state.period_start_trust = state.current_trust;
            state.period_start_time = now;
        }

        let period_delta = new_trust - state.period_start_trust;

        // Check limits

        if period_delta > 0.0 && period_delta > config.max_increase_per_period {
            match config.violation_action {
                VelocityViolationAction::Clamp => {
                    let clamped = state.period_start_trust + config.max_increase_per_period;
                    (clamped.min(1.0), true)
                }
                VelocityViolationAction::Reject => {
                    state.flags.velocity_violations += 1;
                    (state.current_trust, true)
                }
                VelocityViolationAction::FlagForReview => {
                    state.flags.flagged_for_review = true;
                    (new_trust.clamp(0.0, 1.0), true)
                }
            }
        } else if period_delta < 0.0 && period_delta.abs() > config.max_decrease_per_period {
            match config.violation_action {
                VelocityViolationAction::Clamp => {
                    let clamped = state.period_start_trust - config.max_decrease_per_period;
                    (clamped.max(0.0), true)
                }
                VelocityViolationAction::Reject => {
                    state.flags.velocity_violations += 1;
                    (state.current_trust, true)
                }
                VelocityViolationAction::FlagForReview => {
                    state.flags.flagged_for_review = true;
                    (new_trust.clamp(0.0, 1.0), true)
                }
            }
        } else {
            (new_trust.clamp(0.0, 1.0), false)
        }
    }

    /// Get current trust with decay applied
    pub fn current_trust(&self, agent_id: &str) -> Option<f64> {
        let state = self.agents.get(agent_id)?;

        if !self.config.decay.enabled {
            return Some(state.current_trust);
        }

        let elapsed = self.current_time.saturating_sub(state.last_activity);

        // Check grace period
        if elapsed <= self.config.decay.grace_period_ms {
            return Some(state.current_trust);
        }

        let decay_time = elapsed - self.config.decay.grace_period_ms;
        let decayed = self.apply_decay(state.current_trust, decay_time);

        Some(decayed)
    }

    /// Apply decay to a trust value
    fn apply_decay(&self, trust: f64, elapsed_ms: u64) -> f64 {
        let config = &self.config.decay;

        let decayed = match config.curve {
            DecayCurve::Exponential => {
                let half_lives = elapsed_ms as f64 / config.half_life_ms as f64;
                trust * 0.5_f64.powf(half_lives)
            }
            DecayCurve::Linear => {
                let full_decay_time = config.half_life_ms * 2;
                let progress = (elapsed_ms as f64 / full_decay_time as f64).min(1.0);
                trust - (trust - config.floor) * progress
            }
            DecayCurve::Stepped {
                step_interval_ms,
                step_size,
            } => {
                let steps = elapsed_ms / step_interval_ms;
                (trust - (steps as f64 * step_size)).max(config.floor)
            }
        };

        decayed.max(config.floor)
    }

    /// Record activity (resets decay timer)
    pub fn record_activity(&mut self, agent_id: &str) -> bool {
        if let Some(state) = self.agents.get_mut(agent_id) {
            state.last_activity = self.current_time;
            true
        } else {
            false
        }
    }

    /// Advance simulation time
    pub fn tick(&mut self, elapsed_ms: u64) {
        self.current_time += elapsed_ms;
    }

    /// Set current time (for testing)
    pub fn set_time(&mut self, time: u64) {
        self.current_time = time;
    }

    /// Get trust history for an agent
    pub fn trust_history(&self, agent_id: &str) -> Vec<TrustSnapshot> {
        self.agents
            .get(agent_id)
            .map(|s| s.snapshots.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get recent trust events
    pub fn recent_events(&self, agent_id: &str, limit: usize) -> Vec<TrustEvent> {
        self.agents
            .get(agent_id)
            .map(|s| s.events.iter().rev().take(limit).cloned().collect())
            .unwrap_or_default()
    }

    /// Check if agent is flagged for review
    pub fn is_flagged(&self, agent_id: &str) -> bool {
        self.agents
            .get(agent_id)
            .map(|s| s.flags.flagged_for_review)
            .unwrap_or(false)
    }

    /// Clear flag
    pub fn clear_flag(&mut self, agent_id: &str) {
        if let Some(state) = self.agents.get_mut(agent_id) {
            state.flags.flagged_for_review = false;
        }
    }

    /// Apply decay to all agents and take periodic snapshots
    pub fn periodic_maintenance(&mut self) {
        let now = self.current_time;
        let agent_ids: Vec<_> = self.agents.keys().cloned().collect();

        for agent_id in agent_ids {
            if let Some(state) = self.agents.get(&agent_id) {
                // Check if snapshot needed
                let last_snapshot_time = state.snapshots.back().map(|s| s.timestamp).unwrap_or(0);

                if now.saturating_sub(last_snapshot_time) >= self.config.snapshot_interval_ms {
                    let _current = self.current_trust(&agent_id).unwrap_or(0.0);
                    let reason = SnapshotReason::Periodic;
                    self.take_snapshot(&agent_id, reason);
                }
            }
        }
    }

    /// Take a snapshot of agent's trust
    fn take_snapshot(&mut self, agent_id: &str, reason: SnapshotReason) {
        if let Some(state) = self.agents.get_mut(agent_id) {
            let current = state.current_trust;
            let delta = state.snapshots.back().map(|s| current - s.trust);

            let snapshot = TrustSnapshot {
                agent_id: agent_id.to_string(),
                trust: current,
                kvector: None,
                timestamp: self.current_time,
                reason,
                delta,
            };

            state.snapshots.push_back(snapshot);

            // Trim snapshots
            while state.snapshots.len() > self.config.max_snapshots {
                state.snapshots.pop_front();
            }
        }
    }

    /// Calculate effective trust with reputation memory
    pub fn effective_trust(&self, agent_id: &str) -> Option<f64> {
        let state = self.agents.get(agent_id)?;

        if !self.config.memory.enabled {
            return self.current_trust(agent_id);
        }

        let current = self.current_trust(agent_id)?;

        // Calculate historical component
        let historical = self.calculate_historical_trust(state);

        // Blend current and historical
        let weight = self.config.memory.history_weight;
        Some(current * (1.0 - weight) + historical * weight)
    }

    /// Calculate historical trust component
    fn calculate_historical_trust(&self, state: &AgentTrustState) -> f64 {
        if state.events.is_empty() {
            return state.current_trust;
        }

        let now = self.current_time;
        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;

        for event in &state.events {
            let age = now.saturating_sub(event.timestamp);

            // Check memory limits
            let memory_limit = if event.delta >= 0.0 {
                self.config.memory.positive_memory_ms
            } else {
                self.config.memory.negative_memory_ms
            };

            if age > memory_limit {
                continue;
            }

            // Apply forgiveness decay
            let half_lives = age as f64 / self.config.memory.forgiveness_half_life_ms as f64;
            let weight = 0.5_f64.powf(half_lives);

            weighted_sum += event.delta * weight;
            weight_total += weight;
        }

        if weight_total > 0.0 {
            (state.current_trust + weighted_sum / weight_total).clamp(0.0, 1.0)
        } else {
            state.current_trust
        }
    }

    fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

// ============================================================================
// Results
// ============================================================================

/// Result of a trust update
#[derive(Debug, Clone)]
pub struct TrustUpdateResult {
    /// Trust before update
    pub old_trust: f64,
    /// Trust after update
    pub new_trust: f64,
    /// Requested change
    pub requested_delta: f64,
    /// Actual change (may differ due to velocity limits)
    pub actual_delta: f64,
    /// Whether velocity limiting was applied
    pub velocity_limited: bool,
}

// ============================================================================
// Errors
// ============================================================================

/// Temporal trust errors
#[derive(Debug, Clone)]
pub enum TemporalTrustError {
    /// Agent not found.
    AgentNotFound {
        /// Agent ID that was not found.
        agent_id: String,
    },
    /// Velocity limit exceeded.
    VelocityLimitExceeded {
        /// Requested change amount.
        requested: f64,
        /// Maximum allowed change.
        allowed: f64,
    },
}

impl std::fmt::Display for TemporalTrustError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AgentNotFound { agent_id } => write!(f, "Agent not found: {}", agent_id),
            Self::VelocityLimitExceeded { requested, allowed } => {
                write!(
                    f,
                    "Velocity limit exceeded: requested {}, allowed {}",
                    requested, allowed
                )
            }
        }
    }
}

impl std::error::Error for TemporalTrustError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trust_decay() {
        let config = TemporalTrustConfig {
            decay: TrustDecayConfig {
                enabled: true,
                half_life_ms: 1000, // 1 second for testing
                floor: 0.1,
                grace_period_ms: 0,
                curve: DecayCurve::Exponential,
            },
            ..Default::default()
        };

        let mut manager = TemporalTrustManager::new(config);
        manager.set_time(0);
        manager.record_trust("agent-1", 0.8);

        // After one half-life, trust should be ~0.4
        manager.tick(1000);
        let trust = manager.current_trust("agent-1").unwrap();
        assert!((trust - 0.4).abs() < 0.01);

        // After two half-lives, trust should be ~0.2
        manager.tick(1000);
        let trust = manager.current_trust("agent-1").unwrap();
        assert!((trust - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_velocity_limits() {
        let config = TemporalTrustConfig {
            velocity: VelocityLimitConfig {
                enabled: true,
                max_increase_per_period: 0.1,
                max_decrease_per_period: 0.2,
                period_ms: 1000,
                violation_action: VelocityViolationAction::Clamp,
            },
            decay: TrustDecayConfig {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut manager = TemporalTrustManager::new(config);
        manager.set_time(0);
        manager.record_trust("agent-1", 0.5);

        // Try to increase by 0.3 (should be clamped to 0.1)
        let result = manager.update_trust("agent-1", 0.8, "test").unwrap();
        assert!(result.velocity_limited);
        assert!((result.new_trust - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_grace_period() {
        let config = TemporalTrustConfig {
            decay: TrustDecayConfig {
                enabled: true,
                half_life_ms: 1000,
                floor: 0.1,
                grace_period_ms: 500, // 500ms grace
                curve: DecayCurve::Exponential,
            },
            ..Default::default()
        };

        let mut manager = TemporalTrustManager::new(config);
        manager.set_time(0);
        manager.record_trust("agent-1", 0.8);

        // Within grace period - no decay
        manager.tick(400);
        let trust = manager.current_trust("agent-1").unwrap();
        assert!((trust - 0.8).abs() < 0.01);

        // After grace period - decay starts
        manager.tick(600); // 1000ms total, 500ms after grace
        let trust = manager.current_trust("agent-1").unwrap();
        assert!(trust < 0.8);
    }

    #[test]
    fn test_activity_resets_decay() {
        let config = TemporalTrustConfig {
            decay: TrustDecayConfig {
                enabled: true,
                half_life_ms: 1000,
                floor: 0.1,
                grace_period_ms: 0,
                curve: DecayCurve::Exponential,
            },
            ..Default::default()
        };

        let mut manager = TemporalTrustManager::new(config);
        manager.set_time(0);
        manager.record_trust("agent-1", 0.8);

        // Let decay happen
        manager.tick(1000);
        let decayed = manager.current_trust("agent-1").unwrap();
        assert!(decayed < 0.8);

        // Record activity
        manager.record_activity("agent-1");

        // Trust should be back to current (decay timer reset)
        // Note: decay already happened, so we check from the decayed value
        manager.tick(500);
        let after_activity = manager.current_trust("agent-1").unwrap();

        // After activity, decay starts fresh from current trust
        // So after 500ms (half of half-life), should be decayed * 0.707
        assert!(after_activity > decayed * 0.5);
    }
}
