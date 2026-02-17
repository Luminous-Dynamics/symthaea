//! Pattern Lifecycle Management
//!
//! Component 14 manages the lifecycle states of patterns (active, deprecated, archived).

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::collections::HashMap;

use super::PatternId;

// ==============================================================================
// COMPONENT 14: PATTERN LIFECYCLE MANAGEMENT
// ==============================================================================
//
// Manages the complete lifecycle of patterns from creation to retirement:
//
// 1. Active: Pattern is in regular use
// 2. Deprecated: Pattern marked for phase-out (use warned)
// 3. Archived: Pattern moved to cold storage (retrievable)
// 4. Retired: Pattern permanently removed from active consideration
//
// Lifecycle Transitions:
// - Active → Deprecated: Manual or auto (low usage + stale + low trust)
// - Deprecated → Archived: After deprecation period expires
// - Archived → Active: Manual resurrection
// - Deprecated → Active: Manual un-deprecation
// - Archived → Retired: Permanent removal
//
// Key Features:
// - Automatic deprecation based on configurable criteria
// - Deprecation warnings when using deprecated patterns
// - Archive with optional cold storage
// - Resurrection of archived patterns when needed
// - Complete lifecycle audit trail

/// Lifecycle state of a pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PatternLifecycleState {
    /// Pattern is active and in regular use
    Active,
    /// Pattern marked for phase-out (still usable, but warns)
    Deprecated,
    /// Pattern moved to cold storage (not in active queries)
    Archived,
    /// Pattern permanently retired (historical record only)
    Retired,
}

impl Default for PatternLifecycleState {
    fn default() -> Self {
        Self::Active
    }
}

impl PatternLifecycleState {
    /// Check if pattern is usable (Active or Deprecated)
    pub fn is_usable(&self) -> bool {
        matches!(self, Self::Active | Self::Deprecated)
    }

    /// Check if pattern is in cold storage
    pub fn is_cold(&self) -> bool {
        matches!(self, Self::Archived | Self::Retired)
    }

    /// Check if pattern should emit usage warnings
    pub fn should_warn(&self) -> bool {
        matches!(self, Self::Deprecated)
    }
}

/// Reason for a lifecycle transition
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LifecycleTransitionReason {
    /// Manual transition by operator
    Manual(String),
    /// Auto-deprecated due to low usage
    LowUsage { usage_count: u32, threshold: u32 },
    /// Auto-deprecated due to staleness
    Stale { days_since_use: f32, threshold: f32 },
    /// Auto-deprecated due to low trust
    LowTrust { trust_score: f32, threshold: f32 },
    /// Auto-deprecated due to combined factors
    Combined { score: f32, factors: Vec<String> },
    /// Superseded by a better pattern
    Superseded { replacement_id: PatternId },
    /// Deprecation period expired
    DeprecationExpired { deprecation_days: u32 },
    /// Resurrected from archive due to renewed need
    Resurrected { reason: String },
}

impl LifecycleTransitionReason {
    /// Create a manual transition reason
    pub fn manual(reason: impl Into<String>) -> Self {
        Self::Manual(reason.into())
    }

    /// Create a superseded reason
    pub fn superseded(replacement_id: PatternId) -> Self {
        Self::Superseded { replacement_id }
    }

    /// Create a resurrection reason
    pub fn resurrected(reason: impl Into<String>) -> Self {
        Self::Resurrected { reason: reason.into() }
    }
}

/// Record of a lifecycle transition
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LifecycleTransition {
    /// Pattern that transitioned
    pub pattern_id: PatternId,
    /// Previous state
    pub from_state: PatternLifecycleState,
    /// New state
    pub to_state: PatternLifecycleState,
    /// Reason for transition
    pub reason: LifecycleTransitionReason,
    /// When the transition occurred
    pub timestamp: u64,
    /// Who initiated the transition (agent ID or "system")
    pub initiator: String,
}

impl LifecycleTransition {
    /// Create a new transition record
    pub fn new(
        pattern_id: PatternId,
        from_state: PatternLifecycleState,
        to_state: PatternLifecycleState,
        reason: LifecycleTransitionReason,
        timestamp: u64,
        initiator: impl Into<String>,
    ) -> Self {
        Self {
            pattern_id,
            from_state,
            to_state,
            reason,
            timestamp,
            initiator: initiator.into(),
        }
    }
}

/// Configuration for automatic lifecycle management
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LifecycleConfig {
    /// Enable automatic deprecation
    pub auto_deprecate: bool,

    /// Minimum usage count before auto-deprecation (below this = deprecate)
    pub min_usage_threshold: u32,

    /// Days since last use before considered stale
    pub staleness_days: f32,

    /// Minimum trust score before auto-deprecation
    pub min_trust_threshold: f32,

    /// Weight for usage in combined score
    pub usage_weight: f32,

    /// Weight for staleness in combined score
    pub staleness_weight: f32,

    /// Weight for trust in combined score
    pub trust_weight: f32,

    /// Combined score threshold for auto-deprecation
    pub combined_threshold: f32,

    /// Days to keep deprecated before archiving
    pub deprecation_period_days: u32,

    /// Whether to automatically archive after deprecation period
    pub auto_archive: bool,

    /// Maximum archived patterns to keep (0 = unlimited)
    pub max_archived: usize,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            auto_deprecate: true,
            min_usage_threshold: 5,
            staleness_days: 90.0,
            min_trust_threshold: 0.3,
            usage_weight: 0.3,
            staleness_weight: 0.4,
            trust_weight: 0.3,
            combined_threshold: 0.4,
            deprecation_period_days: 30,
            auto_archive: true,
            max_archived: 1000,
        }
    }
}

/// Extended pattern info with lifecycle tracking
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatternLifecycleInfo {
    /// Current lifecycle state
    pub state: PatternLifecycleState,

    /// When the pattern was created
    pub created_at: u64,

    /// When state last changed
    pub state_changed_at: u64,

    /// Number of times pattern has been deprecated
    pub deprecation_count: u32,

    /// Number of times pattern has been resurrected
    pub resurrection_count: u32,

    /// If deprecated, when was it deprecated
    pub deprecated_at: Option<u64>,

    /// If deprecated, who deprecated it
    pub deprecated_by: Option<String>,

    /// If deprecated, why
    pub deprecation_reason: Option<LifecycleTransitionReason>,

    /// Replacement pattern (if superseded)
    pub replacement_id: Option<PatternId>,
}

impl PatternLifecycleInfo {
    /// Create new lifecycle info for a pattern
    pub fn new(timestamp: u64) -> Self {
        Self {
            state: PatternLifecycleState::Active,
            created_at: timestamp,
            state_changed_at: timestamp,
            deprecation_count: 0,
            resurrection_count: 0,
            deprecated_at: None,
            deprecated_by: None,
            deprecation_reason: None,
            replacement_id: None,
        }
    }

    /// Check if pattern has ever been deprecated
    pub fn was_deprecated(&self) -> bool {
        self.deprecation_count > 0
    }

    /// Check if pattern has been resurrected
    pub fn was_resurrected(&self) -> bool {
        self.resurrection_count > 0
    }

    /// Get age of pattern in days
    pub fn age_days(&self, current_time: u64) -> f32 {
        (current_time.saturating_sub(self.created_at)) as f32 / (24.0 * 60.0 * 60.0)
    }

    /// Get days since state change
    pub fn days_in_current_state(&self, current_time: u64) -> f32 {
        (current_time.saturating_sub(self.state_changed_at)) as f32 / (24.0 * 60.0 * 60.0)
    }
}

/// Registry for tracking pattern lifecycles
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatternLifecycleRegistry {
    /// Lifecycle info for each pattern
    lifecycles: HashMap<PatternId, PatternLifecycleInfo>,

    /// Transition history (recent transitions)
    transitions: Vec<LifecycleTransition>,

    /// Configuration
    pub config: LifecycleConfig,

    /// Maximum transitions to keep in history
    max_transitions: usize,
}

impl Default for PatternLifecycleRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternLifecycleRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            lifecycles: HashMap::new(),
            transitions: Vec::new(),
            config: LifecycleConfig::default(),
            max_transitions: 10_000,
        }
    }

    /// Create with custom config
    pub fn with_config(config: LifecycleConfig) -> Self {
        Self {
            lifecycles: HashMap::new(),
            transitions: Vec::new(),
            config,
            max_transitions: 10_000,
        }
    }

    /// Register a new pattern
    pub fn register(&mut self, pattern_id: PatternId, timestamp: u64) {
        self.lifecycles.insert(pattern_id, PatternLifecycleInfo::new(timestamp));
    }

    /// Get lifecycle info for a pattern
    pub fn get(&self, pattern_id: PatternId) -> Option<&PatternLifecycleInfo> {
        self.lifecycles.get(&pattern_id)
    }

    /// Get mutable lifecycle info
    pub fn get_mut(&mut self, pattern_id: PatternId) -> Option<&mut PatternLifecycleInfo> {
        self.lifecycles.get_mut(&pattern_id)
    }

    /// Get current state of a pattern
    pub fn state(&self, pattern_id: PatternId) -> PatternLifecycleState {
        self.lifecycles
            .get(&pattern_id)
            .map(|info| info.state)
            .unwrap_or(PatternLifecycleState::Active)
    }

    /// Check if pattern is usable
    pub fn is_usable(&self, pattern_id: PatternId) -> bool {
        self.state(pattern_id).is_usable()
    }

    /// Transition a pattern to a new state
    pub fn transition(
        &mut self,
        pattern_id: PatternId,
        new_state: PatternLifecycleState,
        reason: LifecycleTransitionReason,
        timestamp: u64,
        initiator: impl Into<String>,
    ) -> bool {
        let initiator_str = initiator.into();

        if let Some(info) = self.lifecycles.get_mut(&pattern_id) {
            let from_state = info.state;

            // Don't transition if already in target state
            if from_state == new_state {
                return false;
            }

            // Update state
            info.state = new_state;
            info.state_changed_at = timestamp;

            // Track deprecation-specific info
            match new_state {
                PatternLifecycleState::Deprecated => {
                    info.deprecation_count += 1;
                    info.deprecated_at = Some(timestamp);
                    info.deprecated_by = Some(initiator_str.clone());
                    info.deprecation_reason = Some(reason.clone());

                    if let LifecycleTransitionReason::Superseded { replacement_id } = &reason {
                        info.replacement_id = Some(*replacement_id);
                    }
                }
                PatternLifecycleState::Active if from_state == PatternLifecycleState::Archived => {
                    info.resurrection_count += 1;
                    info.deprecated_at = None;
                    info.deprecated_by = None;
                    info.deprecation_reason = None;
                }
                _ => {}
            }

            // Record transition
            let transition = LifecycleTransition::new(
                pattern_id,
                from_state,
                new_state,
                reason,
                timestamp,
                initiator_str,
            );
            self.add_transition(transition);

            true
        } else {
            false
        }
    }

    /// Deprecate a pattern
    pub fn deprecate(
        &mut self,
        pattern_id: PatternId,
        reason: LifecycleTransitionReason,
        timestamp: u64,
        initiator: impl Into<String>,
    ) -> bool {
        self.transition(
            pattern_id,
            PatternLifecycleState::Deprecated,
            reason,
            timestamp,
            initiator,
        )
    }

    /// Archive a pattern
    pub fn archive(
        &mut self,
        pattern_id: PatternId,
        reason: LifecycleTransitionReason,
        timestamp: u64,
        initiator: impl Into<String>,
    ) -> bool {
        self.transition(
            pattern_id,
            PatternLifecycleState::Archived,
            reason,
            timestamp,
            initiator,
        )
    }

    /// Resurrect an archived pattern
    pub fn resurrect(
        &mut self,
        pattern_id: PatternId,
        reason: String,
        timestamp: u64,
        initiator: impl Into<String>,
    ) -> bool {
        self.transition(
            pattern_id,
            PatternLifecycleState::Active,
            LifecycleTransitionReason::Resurrected { reason },
            timestamp,
            initiator,
        )
    }

    /// Retire a pattern permanently
    pub fn retire(
        &mut self,
        pattern_id: PatternId,
        reason: LifecycleTransitionReason,
        timestamp: u64,
        initiator: impl Into<String>,
    ) -> bool {
        self.transition(
            pattern_id,
            PatternLifecycleState::Retired,
            reason,
            timestamp,
            initiator,
        )
    }

    /// Calculate deprecation score for a pattern
    /// Returns (should_deprecate, score, factors)
    pub fn calculate_deprecation_score(
        &self,
        _pattern_id: PatternId,
        usage_count: u32,
        days_since_use: f32,
        trust_score: f32,
    ) -> (bool, f32, Vec<String>) {
        let mut factors = Vec::new();
        let mut score = 0.0;

        // Usage factor
        if usage_count < self.config.min_usage_threshold {
            let usage_factor = 1.0 - (usage_count as f32 / self.config.min_usage_threshold as f32);
            score += usage_factor * self.config.usage_weight;
            factors.push(format!("low_usage:{}", usage_count));
        }

        // Staleness factor
        if days_since_use > self.config.staleness_days {
            let staleness_factor = ((days_since_use - self.config.staleness_days) / self.config.staleness_days).min(1.0);
            score += staleness_factor * self.config.staleness_weight;
            factors.push(format!("stale:{:.0}d", days_since_use));
        }

        // Trust factor
        if trust_score < self.config.min_trust_threshold {
            let trust_factor = 1.0 - (trust_score / self.config.min_trust_threshold);
            score += trust_factor * self.config.trust_weight;
            factors.push(format!("low_trust:{:.2}", trust_score));
        }

        let should_deprecate = self.config.auto_deprecate && score >= self.config.combined_threshold;
        (should_deprecate, score, factors)
    }

    /// Get patterns that should be auto-deprecated
    pub fn patterns_needing_deprecation(&self) -> Vec<PatternId> {
        self.lifecycles
            .iter()
            .filter(|(_, info)| info.state == PatternLifecycleState::Active)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get deprecated patterns ready for archiving
    pub fn patterns_ready_for_archive(&self, current_time: u64) -> Vec<PatternId> {
        if !self.config.auto_archive {
            return Vec::new();
        }

        let threshold_secs = self.config.deprecation_period_days as u64 * 24 * 60 * 60;

        self.lifecycles
            .iter()
            .filter(|(_, info)| {
                info.state == PatternLifecycleState::Deprecated
                    && info.deprecated_at.map(|t| current_time.saturating_sub(t) >= threshold_secs).unwrap_or(false)
            })
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get patterns by state
    pub fn patterns_by_state(&self, state: PatternLifecycleState) -> Vec<PatternId> {
        self.lifecycles
            .iter()
            .filter(|(_, info)| info.state == state)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get active pattern count
    pub fn active_count(&self) -> usize {
        self.patterns_by_state(PatternLifecycleState::Active).len()
    }

    /// Get deprecated pattern count
    pub fn deprecated_count(&self) -> usize {
        self.patterns_by_state(PatternLifecycleState::Deprecated).len()
    }

    /// Get archived pattern count
    pub fn archived_count(&self) -> usize {
        self.patterns_by_state(PatternLifecycleState::Archived).len()
    }

    /// Add transition to history
    fn add_transition(&mut self, transition: LifecycleTransition) {
        self.transitions.push(transition);
        if self.transitions.len() > self.max_transitions {
            let to_remove = self.max_transitions / 10;
            self.transitions.drain(0..to_remove);
        }
    }

    /// Get recent transitions for a pattern
    pub fn recent_transitions(&self, pattern_id: PatternId, limit: usize) -> Vec<&LifecycleTransition> {
        self.transitions
            .iter()
            .rev()
            .filter(|t| t.pattern_id == pattern_id)
            .take(limit)
            .collect()
    }

    /// Get lifecycle statistics
    pub fn stats(&self) -> LifecycleStats {
        let total = self.lifecycles.len();
        let active = self.active_count();
        let deprecated = self.deprecated_count();
        let archived = self.archived_count();
        let retired = self.patterns_by_state(PatternLifecycleState::Retired).len();

        let total_resurrections: u32 = self.lifecycles.values().map(|i| i.resurrection_count).sum();
        let total_deprecations: u32 = self.lifecycles.values().map(|i| i.deprecation_count).sum();

        LifecycleStats {
            total_patterns: total,
            active_patterns: active,
            deprecated_patterns: deprecated,
            archived_patterns: archived,
            retired_patterns: retired,
            total_resurrections,
            total_deprecations,
            total_transitions: self.transitions.len(),
        }
    }

    /// Check if a pattern should be auto-deprecated based on usage, staleness, and trust
    /// Returns the deprecation reason if it should be deprecated, None otherwise
    pub fn check_auto_deprecation(
        &self,
        pattern_id: PatternId,
        usage_count: u32,
        last_used: u64,
        trust_score: f32,
        current_time: u64,
    ) -> Option<LifecycleTransitionReason> {
        // Only check active patterns
        if self.state(pattern_id) != PatternLifecycleState::Active {
            return None;
        }

        let days_since_use = (current_time.saturating_sub(last_used) as f32) / (24.0 * 60.0 * 60.0);
        let (should_deprecate, score, factors) =
            self.calculate_deprecation_score(pattern_id, usage_count, days_since_use, trust_score);

        if should_deprecate {
            Some(LifecycleTransitionReason::Combined { score, factors })
        } else if usage_count < self.config.min_usage_threshold && days_since_use > self.config.staleness_days * 0.5 {
            // Low usage alone might warrant deprecation
            Some(LifecycleTransitionReason::LowUsage {
                usage_count,
                threshold: self.config.min_usage_threshold,
            })
        } else if days_since_use > self.config.staleness_days * 2.0 {
            // Very stale alone warrants deprecation
            Some(LifecycleTransitionReason::Stale {
                days_since_use,
                threshold: self.config.staleness_days,
            })
        } else if trust_score < self.config.min_trust_threshold * 0.5 {
            // Very low trust alone warrants deprecation
            Some(LifecycleTransitionReason::LowTrust {
                trust_score,
                threshold: self.config.min_trust_threshold,
            })
        } else {
            None
        }
    }

    /// Auto-archive patterns whose deprecation period has expired
    /// Returns list of archived pattern IDs
    pub fn auto_archive_expired(&mut self, current_time: u64) -> Vec<PatternId> {
        let patterns_to_archive = self.patterns_ready_for_archive(current_time);
        let mut archived = Vec::new();

        for pattern_id in patterns_to_archive {
            let reason = LifecycleTransitionReason::DeprecationExpired {
                deprecation_days: self.config.deprecation_period_days,
            };
            if self.archive(pattern_id, reason, current_time, "auto") {
                archived.push(pattern_id);
            }
        }

        archived
    }

    /// Get all transitions for a specific pattern
    pub fn transitions_for(&self, pattern_id: PatternId) -> Vec<&LifecycleTransition> {
        self.transitions
            .iter()
            .filter(|t| t.pattern_id == pattern_id)
            .collect()
    }
}

/// Statistics about pattern lifecycle
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LifecycleStats {
    /// Total patterns tracked
    pub total_patterns: usize,
    /// Currently active patterns
    pub active_patterns: usize,
    /// Currently deprecated patterns
    pub deprecated_patterns: usize,
    /// Currently archived patterns
    pub archived_patterns: usize,
    /// Permanently retired patterns
    pub retired_patterns: usize,
    /// Total resurrection count across all patterns
    pub total_resurrections: u32,
    /// Total deprecation count across all patterns
    pub total_deprecations: u32,
    /// Total transitions recorded
    pub total_transitions: usize,
}

impl LifecycleStats {
    /// Get active ratio
    pub fn active_ratio(&self) -> f32 {
        if self.total_patterns == 0 {
            return 0.0;
        }
        self.active_patterns as f32 / self.total_patterns as f32
    }

    /// Get churn rate (deprecations / total)
    pub fn churn_rate(&self) -> f32 {
        if self.total_patterns == 0 {
            return 0.0;
        }
        self.total_deprecations as f32 / self.total_patterns as f32
    }

    /// Get resurrection rate
    pub fn resurrection_rate(&self) -> f32 {
        if self.total_deprecations == 0 {
            return 0.0;
        }
        self.total_resurrections as f32 / self.total_deprecations as f32
    }
}

