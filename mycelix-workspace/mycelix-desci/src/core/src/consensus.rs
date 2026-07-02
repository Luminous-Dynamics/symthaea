// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Temporal Consensus Tracking
//!
//! Tracks how scientific consensus on claims evolves over time, detecting
//! paradigm shifts, emerging consensus, and consensus stability.

use std::collections::HashMap;
use uuid::Uuid;

/// A snapshot of consensus at a point in time
#[derive(Debug, Clone)]
pub struct ConsensusSnapshot {
    /// Timestamp of this snapshot
    pub timestamp: i64,
    /// Support ratio (0.0-1.0)
    pub support_ratio: f64,
    /// Number of verifications included
    pub verification_count: usize,
    /// Weighted support (if using expertise weights)
    pub weighted_support: f64,
    /// Standard deviation of opinions
    pub opinion_variance: f64,
    /// Whether this was a significant change
    pub is_significant: bool,
}

/// Consensus state classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsensusState {
    /// No consensus (highly contested, ~50%)
    Contested,
    /// Emerging consensus (60-70%)
    Emerging,
    /// Moderate consensus (70-85%)
    Moderate,
    /// Strong consensus (85-95%)
    Strong,
    /// Near-universal consensus (>95%)
    NearUniversal,
    /// Insufficient data
    InsufficientData,
}

impl ConsensusState {
    /// Get state from support ratio
    pub fn from_ratio(ratio: f64, min_verifications: usize, actual_count: usize) -> Self {
        if actual_count < min_verifications {
            return ConsensusState::InsufficientData;
        }

        // Normalize to distance from 0.5 (contested)
        let distance = (ratio - 0.5).abs();

        if distance < 0.1 {
            ConsensusState::Contested
        } else if distance < 0.2 {
            ConsensusState::Emerging
        } else if distance < 0.35 {
            ConsensusState::Moderate
        } else if distance < 0.45 {
            ConsensusState::Strong
        } else {
            ConsensusState::NearUniversal
        }
    }

    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            ConsensusState::Contested => "Highly contested - no clear consensus",
            ConsensusState::Emerging => "Emerging consensus - beginning to form",
            ConsensusState::Moderate => "Moderate consensus - majority agreement",
            ConsensusState::Strong => "Strong consensus - clear majority",
            ConsensusState::NearUniversal => "Near-universal consensus",
            ConsensusState::InsufficientData => "Insufficient data for consensus",
        }
    }
}

/// Detected trend in consensus evolution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsensusTrend {
    /// Consensus is strengthening
    Strengthening,
    /// Consensus is weakening
    Weakening,
    /// Consensus is stable
    Stable,
    /// Rapid reversal in progress
    Reversing,
    /// Oscillating without clear direction
    Oscillating,
}

/// A paradigm shift event
#[derive(Debug, Clone)]
pub struct ParadigmShift {
    /// When the shift was detected
    pub detected_at: i64,
    /// Support ratio before the shift
    pub before_ratio: f64,
    /// Support ratio after the shift
    pub after_ratio: f64,
    /// Duration of the transition
    pub transition_duration: i64,
    /// Magnitude of the shift (0.0-1.0)
    pub magnitude: f64,
    /// Classification of the shift
    pub shift_type: ShiftType,
}

/// Type of paradigm shift
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftType {
    /// New consensus emerged from contested state
    ConsensusEmergence,
    /// Existing consensus collapsed
    ConsensusCollapse,
    /// Consensus flipped to opposite position
    ConsensusReversal,
    /// Gradual drift in position
    GradualDrift,
}

/// Temporal consensus history for a claim
#[derive(Debug, Clone)]
pub struct ConsensusHistory {
    /// Claim ID
    pub claim_id: Uuid,
    /// Historical snapshots (ordered by time)
    pub snapshots: Vec<ConsensusSnapshot>,
    /// Detected paradigm shifts
    pub paradigm_shifts: Vec<ParadigmShift>,
    /// Current consensus state
    pub current_state: ConsensusState,
    /// Current trend
    pub current_trend: ConsensusTrend,
    /// Time of first verification
    pub first_verification: i64,
    /// Time of last update
    pub last_update: i64,
}

impl ConsensusHistory {
    /// Create a new empty history
    pub fn new(claim_id: Uuid) -> Self {
        Self {
            claim_id,
            snapshots: Vec::new(),
            paradigm_shifts: Vec::new(),
            current_state: ConsensusState::InsufficientData,
            current_trend: ConsensusTrend::Stable,
            first_verification: 0,
            last_update: 0,
        }
    }

    /// Get the latest snapshot
    pub fn latest(&self) -> Option<&ConsensusSnapshot> {
        self.snapshots.last()
    }

    /// Get snapshots within a time range
    pub fn range(&self, start: i64, end: i64) -> Vec<&ConsensusSnapshot> {
        self.snapshots
            .iter()
            .filter(|s| s.timestamp >= start && s.timestamp <= end)
            .collect()
    }

    /// Calculate average support ratio over a time window
    pub fn average_support(&self, window_start: i64, window_end: i64) -> Option<f64> {
        let snapshots = self.range(window_start, window_end);
        if snapshots.is_empty() {
            return None;
        }

        let sum: f64 = snapshots.iter().map(|s| s.support_ratio).sum();
        Some(sum / snapshots.len() as f64)
    }

    /// Get consensus stability (inverse of variance)
    pub fn stability(&self, window_size: usize) -> f64 {
        if self.snapshots.len() < window_size {
            return 0.0;
        }

        let recent: Vec<f64> = self
            .snapshots
            .iter()
            .rev()
            .take(window_size)
            .map(|s| s.support_ratio)
            .collect();

        let mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance: f64 = recent.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / recent.len() as f64;

        // Stability = 1 / (1 + variance * 10)
        1.0 / (1.0 + variance * 10.0)
    }
}

/// Configuration for consensus tracking
#[derive(Debug, Clone)]
pub struct ConsensusTrackerConfig {
    /// Minimum verifications for valid consensus
    pub min_verifications: usize,
    /// Snapshot interval (in seconds)
    pub snapshot_interval: i64,
    /// Significant change threshold
    pub significant_change_threshold: f64,
    /// Paradigm shift detection threshold
    pub paradigm_shift_threshold: f64,
    /// Window size for trend detection
    pub trend_window_size: usize,
    /// Time decay for older snapshots
    pub time_decay_factor: f64,
}

impl Default for ConsensusTrackerConfig {
    fn default() -> Self {
        Self {
            min_verifications: 5,
            snapshot_interval: 86400, // 1 day
            significant_change_threshold: 0.1,
            paradigm_shift_threshold: 0.3,
            trend_window_size: 10,
            time_decay_factor: 0.95,
        }
    }
}

/// Consensus tracking engine
pub struct ConsensusTracker {
    config: ConsensusTrackerConfig,
    histories: HashMap<Uuid, ConsensusHistory>,
}

impl ConsensusTracker {
    /// Create a new tracker with default config
    pub fn new() -> Self {
        Self {
            config: ConsensusTrackerConfig::default(),
            histories: HashMap::new(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: ConsensusTrackerConfig) -> Self {
        Self {
            config,
            histories: HashMap::new(),
        }
    }

    /// Record a new verification and update consensus
    pub fn record_verification(
        &mut self,
        claim_id: Uuid,
        supports: bool,
        timestamp: i64,
        weight: f64,
    ) {
        self.histories
            .entry(claim_id)
            .or_insert_with(|| ConsensusHistory::new(claim_id));

        let should_snapshot = {
            let history = self.histories.get(&claim_id).unwrap();
            if history.first_verification == 0 {
                true // will set below
            } else {
                history.snapshots.is_empty()
                    || (timestamp - history.snapshots.last().unwrap().timestamp)
                        >= self.config.snapshot_interval
            }
        };

        {
            let history = self.histories.get_mut(&claim_id).unwrap();
            if history.first_verification == 0 {
                history.first_verification = timestamp;
            }
        }

        if should_snapshot {
            self.create_snapshot(claim_id, timestamp);
        }

        let history = self.histories.get_mut(&claim_id).unwrap();
        history.last_update = timestamp;
    }

    /// Create a consensus snapshot
    fn create_snapshot(&mut self, claim_id: Uuid, timestamp: i64) {
        let history = match self.histories.get_mut(&claim_id) {
            Some(h) => h,
            None => return,
        };

        // For now, create a simulated snapshot
        // In real implementation, this would aggregate actual verifications
        let prev_ratio = history
            .snapshots
            .last()
            .map(|s| s.support_ratio)
            .unwrap_or(0.5);

        let snapshot = ConsensusSnapshot {
            timestamp,
            support_ratio: prev_ratio, // Would be calculated from actual data
            verification_count: 0,
            weighted_support: prev_ratio,
            opinion_variance: 0.0,
            is_significant: false,
        };

        history.snapshots.push(snapshot);
    }

    /// Update consensus state and detect shifts
    pub fn update_consensus(
        &mut self,
        claim_id: Uuid,
        support_ratio: f64,
        verification_count: usize,
        weighted_support: f64,
        timestamp: i64,
    ) {
        // Ensure the history entry exists
        self.histories
            .entry(claim_id)
            .or_insert_with(|| ConsensusHistory::new(claim_id));

        // First pass: read-only computations (immutable borrow of self)
        let (prev_ratio, opinion_variance, trend) = {
            let history = self.histories.get(&claim_id).unwrap();
            let prev_ratio = history
                .snapshots
                .last()
                .map(|s| s.support_ratio)
                .unwrap_or(0.5);
            let opinion_variance = self.calculate_variance(history, support_ratio);
            let trend = self.detect_trend(history);
            (prev_ratio, opinion_variance, trend)
        };

        let change = (support_ratio - prev_ratio).abs();
        let is_significant = change >= self.config.significant_change_threshold;

        // Second pass: mutations (mutable borrow of history)
        {
            let history = self.histories.get_mut(&claim_id).unwrap();

            if history.first_verification == 0 {
                history.first_verification = timestamp;
            }

            let snapshot = ConsensusSnapshot {
                timestamp,
                support_ratio,
                verification_count,
                weighted_support,
                opinion_variance,
                is_significant,
            };

            history.snapshots.push(snapshot);

            // Update current state
            history.current_state =
                ConsensusState::from_ratio(support_ratio, self.config.min_verifications, verification_count);

            // Apply trend
            history.current_trend = trend;

            history.last_update = timestamp;
        }

        // Check for paradigm shift (needs &mut self)
        if change >= self.config.paradigm_shift_threshold {
            self.detect_paradigm_shift(claim_id, prev_ratio, support_ratio, timestamp);
        }
    }

    /// Calculate opinion variance
    fn calculate_variance(&self, history: &ConsensusHistory, new_ratio: f64) -> f64 {
        if history.snapshots.is_empty() {
            return 0.0;
        }

        let recent: Vec<f64> = history
            .snapshots
            .iter()
            .rev()
            .take(self.config.trend_window_size)
            .map(|s| s.support_ratio)
            .collect();

        let mut all_values = recent;
        all_values.push(new_ratio);

        let mean: f64 = all_values.iter().sum::<f64>() / all_values.len() as f64;
        all_values.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / all_values.len() as f64
    }

    /// Detect consensus trend from recent snapshots
    fn detect_trend(&self, history: &ConsensusHistory) -> ConsensusTrend {
        if history.snapshots.len() < 3 {
            return ConsensusTrend::Stable;
        }

        let window: Vec<f64> = history
            .snapshots
            .iter()
            .rev()
            .take(self.config.trend_window_size.min(history.snapshots.len()))
            .map(|s| s.support_ratio)
            .collect();

        if window.len() < 3 {
            return ConsensusTrend::Stable;
        }

        // Calculate linear regression slope
        let n = window.len() as f64;
        let sum_x: f64 = (0..window.len()).map(|i| i as f64).sum();
        let sum_y: f64 = window.iter().sum();
        let sum_xy: f64 = window.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_xx: f64 = (0..window.len()).map(|i| (i * i) as f64).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);

        // Detect oscillation (high variance, low net change)
        let first = window.first().unwrap();
        let last = window.last().unwrap();
        let net_change = (last - first).abs();
        let variance = self.calculate_variance(history, *first);

        if variance > 0.05 && net_change < 0.1 {
            return ConsensusTrend::Oscillating;
        }

        // Detect reversal (crossed 0.5 threshold)
        let crossed_threshold = (*first > 0.5 && *last < 0.5) || (*first < 0.5 && *last > 0.5);
        if crossed_threshold && net_change > 0.2 {
            return ConsensusTrend::Reversing;
        }

        // Trend based on slope
        if slope.abs() < 0.01 {
            ConsensusTrend::Stable
        } else if slope > 0.0 {
            // Moving toward 1.0 (consensus strengthening if already >0.5)
            // or weakening if <0.5
            if *last > 0.5 {
                ConsensusTrend::Strengthening
            } else {
                ConsensusTrend::Weakening
            }
        } else {
            // Moving toward 0.0
            if *last < 0.5 {
                ConsensusTrend::Strengthening // Strengthening in opposition
            } else {
                ConsensusTrend::Weakening
            }
        }
    }

    /// Detect and record a paradigm shift
    fn detect_paradigm_shift(
        &mut self,
        claim_id: Uuid,
        before: f64,
        after: f64,
        timestamp: i64,
    ) {
        let history = match self.histories.get_mut(&claim_id) {
            Some(h) => h,
            None => return,
        };

        let magnitude = (after - before).abs();

        let shift_type = if before > 0.4 && before < 0.6 && after > 0.7 {
            ShiftType::ConsensusEmergence
        } else if before > 0.7 && after < 0.6 {
            ShiftType::ConsensusCollapse
        } else if (before > 0.7 && after < 0.3) || (before < 0.3 && after > 0.7) {
            ShiftType::ConsensusReversal
        } else {
            ShiftType::GradualDrift
        };

        let transition_duration = history
            .snapshots
            .iter()
            .rev()
            .take(5)
            .map(|s| timestamp - s.timestamp)
            .max()
            .unwrap_or(0);

        history.paradigm_shifts.push(ParadigmShift {
            detected_at: timestamp,
            before_ratio: before,
            after_ratio: after,
            transition_duration,
            magnitude,
            shift_type,
        });
    }

    /// Get consensus history for a claim
    pub fn get_history(&self, claim_id: Uuid) -> Option<&ConsensusHistory> {
        self.histories.get(&claim_id)
    }

    /// Get current consensus state
    pub fn get_state(&self, claim_id: Uuid) -> ConsensusState {
        self.histories
            .get(&claim_id)
            .map(|h| h.current_state)
            .unwrap_or(ConsensusState::InsufficientData)
    }

    /// Get all claims with paradigm shifts
    pub fn get_shifting_claims(&self) -> Vec<(Uuid, &ParadigmShift)> {
        self.histories
            .iter()
            .filter_map(|(id, h)| h.paradigm_shifts.last().map(|s| (*id, s)))
            .collect()
    }

    /// Get claims in specific consensus state
    pub fn get_claims_by_state(&self, state: ConsensusState) -> Vec<Uuid> {
        self.histories
            .iter()
            .filter(|(_, h)| h.current_state == state)
            .map(|(id, _)| *id)
            .collect()
    }
}

impl Default for ConsensusTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consensus_state_classification() {
        assert_eq!(
            ConsensusState::from_ratio(0.52, 5, 10),
            ConsensusState::Contested
        );
        assert_eq!(
            ConsensusState::from_ratio(0.65, 5, 10),
            ConsensusState::Emerging
        );
        assert_eq!(
            ConsensusState::from_ratio(0.78, 5, 10),
            ConsensusState::Moderate
        );
        assert_eq!(
            ConsensusState::from_ratio(0.90, 5, 10),
            ConsensusState::Strong
        );
        assert_eq!(
            ConsensusState::from_ratio(0.97, 5, 10),
            ConsensusState::NearUniversal
        );
        assert_eq!(
            ConsensusState::from_ratio(0.90, 5, 3),
            ConsensusState::InsufficientData
        );
    }

    #[test]
    fn test_consensus_history() {
        let mut history = ConsensusHistory::new(Uuid::new_v4());

        history.snapshots.push(ConsensusSnapshot {
            timestamp: 1000,
            support_ratio: 0.6,
            verification_count: 10,
            weighted_support: 0.6,
            opinion_variance: 0.02,
            is_significant: false,
        });

        history.snapshots.push(ConsensusSnapshot {
            timestamp: 2000,
            support_ratio: 0.7,
            verification_count: 15,
            weighted_support: 0.7,
            opinion_variance: 0.01,
            is_significant: true,
        });

        assert_eq!(history.snapshots.len(), 2);
        assert_eq!(history.latest().unwrap().support_ratio, 0.7);

        let range = history.range(500, 1500);
        assert_eq!(range.len(), 1);
    }

    #[test]
    fn test_consensus_tracker() {
        let mut tracker = ConsensusTracker::new();
        let claim_id = Uuid::new_v4();

        // Update with emerging consensus
        tracker.update_consensus(claim_id, 0.65, 10, 0.65, 1000);
        assert_eq!(tracker.get_state(claim_id), ConsensusState::Emerging);

        // Update with strengthening consensus
        tracker.update_consensus(claim_id, 0.80, 20, 0.80, 2000);
        assert_eq!(tracker.get_state(claim_id), ConsensusState::Moderate);
    }

    #[test]
    fn test_paradigm_shift_detection() {
        let mut tracker = ConsensusTracker::with_config(ConsensusTrackerConfig {
            paradigm_shift_threshold: 0.3,
            ..Default::default()
        });
        let claim_id = Uuid::new_v4();

        // Initial contested state
        tracker.update_consensus(claim_id, 0.50, 10, 0.50, 1000);

        // Major shift to strong consensus
        tracker.update_consensus(claim_id, 0.85, 20, 0.85, 2000);

        let history = tracker.get_history(claim_id).unwrap();
        assert!(!history.paradigm_shifts.is_empty());
        assert_eq!(
            history.paradigm_shifts[0].shift_type,
            ShiftType::ConsensusEmergence
        );
    }

    #[test]
    fn test_stability_calculation() {
        let mut history = ConsensusHistory::new(Uuid::new_v4());

        // Add stable snapshots
        for i in 0..10 {
            history.snapshots.push(ConsensusSnapshot {
                timestamp: i * 1000,
                support_ratio: 0.75 + (i as f64 * 0.001), // Very slight variation
                verification_count: 10,
                weighted_support: 0.75,
                opinion_variance: 0.001,
                is_significant: false,
            });
        }

        let stability = history.stability(5);
        assert!(stability > 0.9); // High stability
    }
}
