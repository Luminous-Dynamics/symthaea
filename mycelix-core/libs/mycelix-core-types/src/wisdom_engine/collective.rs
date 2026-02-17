//! Collective Pattern Integration
//!
//! Component 13 connects patterns to collective wisdom and group validation.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::collections::HashMap;

use super::{PatternId, SymthaeaId};

// ==============================================================================
// COMPONENT 13: COLLECTIVE PATTERN INTEGRATION
// ==============================================================================
//
// Connects patterns to the CollectiveMirror for group-validated learning.
// This is NOT about claiming wisdom - it's about observing how patterns
// perform in collective contexts and detecting emergent group behaviors.
//
// Key Insights:
// - A pattern that many agents independently discover is significant
// - A pattern used during high-agreement periods may be echo-chamber bait
// - Patterns that resolve group tension are valuable
// - Absent patterns (shadow) may reveal blind spots
//
// Philosophy: Mirror, not Oracle - we observe and reflect, not prescribe.

/// Configuration for collective pattern observation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CollectivePatternConfig {
    /// Minimum number of independent discoveries to flag as "emergent"
    pub emergence_threshold: usize,

    /// Agreement level above which to flag potential echo chamber
    pub echo_chamber_threshold: f32,

    /// Minimum tension reduction to mark pattern as "tension-resolving"
    pub tension_resolution_threshold: f32,

    /// Weight for collective signals in pattern scoring (0.0-1.0)
    pub collective_weight: f32,

    /// Enable collective observation
    pub enabled: bool,
}

impl Default for CollectivePatternConfig {
    fn default() -> Self {
        Self {
            emergence_threshold: 3, // 3+ independent discoveries
            echo_chamber_threshold: 0.9, // 90%+ agreement is suspicious
            tension_resolution_threshold: 0.2, // 20% tension reduction
            collective_weight: 0.15, // 15% weight in scoring
            enabled: true,
        }
    }
}

/// Collective observation context for a pattern
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CollectivePatternContext {
    /// Pattern ID this context belongs to
    pub pattern_id: PatternId,

    /// Number of agents who independently discovered this pattern
    pub independent_discoveries: usize,

    /// Agents who discovered this pattern
    pub discoverers: Vec<SymthaeaId>,

    /// Average agreement level when pattern was used
    pub avg_agreement_at_usage: f32,

    /// Usage count during high-agreement periods
    pub high_agreement_usages: u32,

    /// Usage count during low-agreement periods (more diverse)
    pub diverse_context_usages: u32,

    /// Times this pattern resolved group tension
    pub tension_resolutions: u32,

    /// Times this pattern increased group tension
    pub tension_increases: u32,

    /// Whether this pattern appears in group shadow (absent when needed)
    pub in_shadow: bool,

    /// Timestamp of last collective observation
    pub last_observed: u64,

    /// Collective confidence modifier
    pub collective_modifier: f32,
}

impl CollectivePatternContext {
    /// Create a new collective context for a pattern
    pub fn new(pattern_id: PatternId, discoverer: SymthaeaId, timestamp: u64) -> Self {
        Self {
            pattern_id,
            independent_discoveries: 1,
            discoverers: vec![discoverer],
            avg_agreement_at_usage: 0.5,
            high_agreement_usages: 0,
            diverse_context_usages: 0,
            tension_resolutions: 0,
            tension_increases: 0,
            in_shadow: false,
            last_observed: timestamp,
            collective_modifier: 1.0,
        }
    }

    /// Record an independent discovery of this pattern by a new agent
    pub fn record_discovery(&mut self, agent_id: SymthaeaId) {
        if !self.discoverers.contains(&agent_id) {
            self.discoverers.push(agent_id);
            self.independent_discoveries += 1;
        }
    }

    /// Check if pattern has emerged through independent discovery
    pub fn is_emergent(&self, threshold: usize) -> bool {
        self.independent_discoveries >= threshold
    }

    /// Record pattern usage in a collective context
    pub fn record_collective_usage(
        &mut self,
        agreement_level: f32,
        echo_threshold: f32,
        timestamp: u64,
    ) {
        // Update running average
        let total_usages = self.high_agreement_usages + self.diverse_context_usages + 1;
        self.avg_agreement_at_usage =
            (self.avg_agreement_at_usage * (total_usages - 1) as f32 + agreement_level)
            / total_usages as f32;

        if agreement_level >= echo_threshold {
            self.high_agreement_usages += 1;
        } else {
            self.diverse_context_usages += 1;
        }

        self.last_observed = timestamp;
    }

    /// Record tension change after pattern usage
    pub fn record_tension_change(&mut self, before: f32, after: f32, threshold: f32) {
        let change = before - after;
        if change >= threshold {
            self.tension_resolutions += 1;
        } else if change <= -threshold {
            self.tension_increases += 1;
        }
    }

    /// Calculate echo chamber risk (0.0 = healthy, 1.0 = very risky)
    pub fn echo_chamber_risk(&self) -> f32 {
        let total = self.high_agreement_usages + self.diverse_context_usages;
        if total == 0 {
            return 0.0;
        }
        self.high_agreement_usages as f32 / total as f32
    }

    /// Calculate tension resolution ratio
    pub fn tension_resolution_ratio(&self) -> f32 {
        let total = self.tension_resolutions + self.tension_increases;
        if total == 0 {
            return 0.5; // Neutral
        }
        self.tension_resolutions as f32 / total as f32
    }

    /// Update collective modifier based on observations
    pub fn update_modifier(&mut self, config: &CollectivePatternConfig) {
        let mut modifier = 1.0f32;

        // Boost for emergent patterns (independently discovered)
        if self.is_emergent(config.emergence_threshold) {
            modifier *= 1.0 + (self.independent_discoveries as f32 * 0.05).min(0.3);
        }

        // Penalty for high echo chamber risk
        let echo_risk = self.echo_chamber_risk();
        if echo_risk > 0.7 {
            modifier *= 1.0 - ((echo_risk - 0.7) * 0.5); // Up to 15% penalty
        }

        // Boost for tension-resolving patterns
        let resolution_ratio = self.tension_resolution_ratio();
        if resolution_ratio > 0.6 {
            modifier *= 1.0 + ((resolution_ratio - 0.5) * 0.2); // Up to 10% boost
        }

        // Penalty for tension-increasing patterns
        if resolution_ratio < 0.4 {
            modifier *= 1.0 - ((0.5 - resolution_ratio) * 0.2); // Up to 10% penalty
        }

        // Penalty for shadow patterns (absent when needed)
        if self.in_shadow {
            modifier *= 0.9; // 10% penalty for being underutilized
        }

        self.collective_modifier = modifier.clamp(0.5, 1.5);
    }
}

/// Types of collective pattern signals
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CollectiveSignal {
    /// Pattern was independently discovered by multiple agents
    EmergentDiscovery,

    /// Pattern was used during high-agreement period (echo risk)
    EchoChamberUsage,

    /// Pattern was used in diverse context (healthy)
    DiverseContextUsage,

    /// Pattern resolved group tension
    TensionResolved,

    /// Pattern increased group tension
    TensionIncreased,

    /// Pattern is in the group's shadow (needed but absent)
    InShadow,

    /// Pattern was validated by group outcome
    GroupValidated,
}

/// A collective observation about a pattern
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CollectiveObservation {
    /// Pattern that was observed
    pub pattern_id: PatternId,

    /// Type of signal observed
    pub signal: CollectiveSignal,

    /// Strength of the signal (0.0-1.0)
    pub strength: f32,

    /// Timestamp of observation
    pub timestamp: u64,

    /// Agreement level at time of observation
    pub agreement_level: f32,

    /// Number of participants in observation context
    pub participant_count: usize,

    /// Optional context description
    pub context: String,
}

impl CollectiveObservation {
    /// Create a new observation
    pub fn new(
        pattern_id: PatternId,
        signal: CollectiveSignal,
        strength: f32,
        timestamp: u64,
        agreement_level: f32,
        participant_count: usize,
    ) -> Self {
        Self {
            pattern_id,
            signal,
            strength: strength.clamp(0.0, 1.0),
            timestamp,
            agreement_level,
            participant_count,
            context: String::new(),
        }
    }

    /// Add context to observation
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = context.into();
        self
    }
}

/// Registry for collective pattern observations
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CollectivePatternRegistry {
    /// Collective context for each pattern
    contexts: HashMap<PatternId, CollectivePatternContext>,

    /// Recent observations (for analysis)
    observations: Vec<CollectiveObservation>,

    /// Configuration
    pub config: CollectivePatternConfig,

    /// Maximum observations to keep
    max_observations: usize,
}

impl CollectivePatternRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            contexts: HashMap::new(),
            observations: Vec::new(),
            config: CollectivePatternConfig::default(),
            max_observations: 10_000,
        }
    }

    /// Create with custom config
    pub fn with_config(config: CollectivePatternConfig) -> Self {
        Self {
            contexts: HashMap::new(),
            observations: Vec::new(),
            config,
            max_observations: 10_000,
        }
    }

    /// Get or create context for a pattern
    pub fn get_or_create(
        &mut self,
        pattern_id: PatternId,
        discoverer: SymthaeaId,
        timestamp: u64,
    ) -> &mut CollectivePatternContext {
        self.contexts
            .entry(pattern_id)
            .or_insert_with(|| CollectivePatternContext::new(pattern_id, discoverer, timestamp))
    }

    /// Get context for a pattern
    pub fn get(&self, pattern_id: PatternId) -> Option<&CollectivePatternContext> {
        self.contexts.get(&pattern_id)
    }

    /// Get mutable context
    pub fn get_mut(&mut self, pattern_id: PatternId) -> Option<&mut CollectivePatternContext> {
        self.contexts.get_mut(&pattern_id)
    }

    /// Record a discovery
    pub fn record_discovery(
        &mut self,
        pattern_id: PatternId,
        discoverer: SymthaeaId,
        timestamp: u64,
    ) {
        // First, ensure context exists and record discovery
        let ctx = self.get_or_create(pattern_id, discoverer, timestamp);
        ctx.record_discovery(discoverer);

        // Extract values we need before checking config
        let emergence_threshold = self.config.emergence_threshold;
        let (is_emergent, discoveries) = {
            let ctx = self.contexts.get(&pattern_id).unwrap();
            (ctx.is_emergent(emergence_threshold), ctx.independent_discoveries)
        };

        // Check for emergent discovery (now no borrow conflict)
        if is_emergent {
            self.add_observation(CollectiveObservation::new(
                pattern_id,
                CollectiveSignal::EmergentDiscovery,
                discoveries as f32 / 10.0,
                timestamp,
                0.5,
                discoveries,
            ));
        }
    }

    /// Record collective usage
    pub fn record_usage(
        &mut self,
        pattern_id: PatternId,
        agreement_level: f32,
        participant_count: usize,
        timestamp: u64,
    ) {
        // Extract config values before mutable borrow
        let echo_threshold = self.config.echo_chamber_threshold;

        if let Some(ctx) = self.contexts.get_mut(&pattern_id) {
            ctx.record_collective_usage(agreement_level, echo_threshold, timestamp);
        }

        // Now check if context exists and determine signal
        if self.contexts.contains_key(&pattern_id) {
            let signal = if agreement_level >= echo_threshold {
                CollectiveSignal::EchoChamberUsage
            } else {
                CollectiveSignal::DiverseContextUsage
            };

            self.add_observation(CollectiveObservation::new(
                pattern_id,
                signal,
                agreement_level,
                timestamp,
                agreement_level,
                participant_count,
            ));
        }
    }

    /// Record tension change
    pub fn record_tension_change(
        &mut self,
        pattern_id: PatternId,
        before: f32,
        after: f32,
        timestamp: u64,
    ) {
        // Extract config values before mutable borrow
        let tension_threshold = self.config.tension_resolution_threshold;

        if let Some(ctx) = self.contexts.get_mut(&pattern_id) {
            ctx.record_tension_change(before, after, tension_threshold);
        }

        // Now check conditions without holding mutable borrow
        if self.contexts.contains_key(&pattern_id) {
            let change = before - after;
            if change.abs() >= tension_threshold {
                let signal = if change > 0.0 {
                    CollectiveSignal::TensionResolved
                } else {
                    CollectiveSignal::TensionIncreased
                };

                self.add_observation(CollectiveObservation::new(
                    pattern_id,
                    signal,
                    change.abs(),
                    timestamp,
                    0.5,
                    0,
                ));
            }
        }
    }

    /// Mark pattern as in shadow (needed but absent)
    pub fn mark_in_shadow(&mut self, pattern_id: PatternId, timestamp: u64) {
        if let Some(ctx) = self.get_mut(pattern_id) {
            ctx.in_shadow = true;
            self.add_observation(CollectiveObservation::new(
                pattern_id,
                CollectiveSignal::InShadow,
                1.0,
                timestamp,
                0.5,
                0,
            ));
        }
    }

    /// Clear shadow status
    pub fn clear_shadow(&mut self, pattern_id: PatternId) {
        if let Some(ctx) = self.get_mut(pattern_id) {
            ctx.in_shadow = false;
        }
    }

    /// Check if a pattern has a collective context
    pub fn has_context(&self, pattern_id: PatternId) -> bool {
        self.contexts.contains_key(&pattern_id)
    }

    /// Register a pattern without an initial discoverer
    pub fn register_pattern(&mut self, pattern_id: PatternId, timestamp: u64) {
        if !self.contexts.contains_key(&pattern_id) {
            // Use pattern_id as discoverer placeholder (will be updated on first discovery)
            let ctx = CollectivePatternContext::new(pattern_id, pattern_id as SymthaeaId, timestamp);
            self.contexts.insert(pattern_id, ctx);
        }
    }

    /// Record usage with agreement level (simplified interface)
    pub fn record_usage_agreement(&mut self, pattern_id: PatternId, agreement_level: f32, timestamp: u64) {
        let echo_threshold = self.config.echo_chamber_threshold;
        if let Some(ctx) = self.contexts.get_mut(&pattern_id) {
            ctx.record_collective_usage(agreement_level, echo_threshold, timestamp);
        }
    }

    /// Set shadow status for a pattern
    pub fn set_shadow_status(&mut self, pattern_id: PatternId, in_shadow: bool, timestamp: u64) {
        if in_shadow {
            self.mark_in_shadow(pattern_id, timestamp);
        } else {
            self.clear_shadow(pattern_id);
        }
    }

    /// Get the collective modifier for a pattern (alias for collective_modifier)
    pub fn get_modifier(&self, pattern_id: PatternId) -> f32 {
        // Return 0.0 for neutral effect if no context (not 1.0 which would be a multiplier)
        self.contexts
            .get(&pattern_id)
            .map(|ctx| ctx.collective_modifier)
            .unwrap_or(0.0)
    }

    /// Update all modifiers
    pub fn update_all_modifiers(&mut self) {
        let config = self.config.clone();
        for ctx in self.contexts.values_mut() {
            ctx.update_modifier(&config);
        }
    }

    /// Get collective modifier for a pattern
    pub fn collective_modifier(&self, pattern_id: PatternId) -> f32 {
        self.contexts
            .get(&pattern_id)
            .map(|ctx| ctx.collective_modifier)
            .unwrap_or(1.0)
    }

    /// Get emergent patterns
    pub fn emergent_patterns(&self) -> Vec<PatternId> {
        self.contexts
            .iter()
            .filter(|(_, ctx)| ctx.is_emergent(self.config.emergence_threshold))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get patterns with high echo chamber risk (uses config threshold)
    pub fn echo_chamber_patterns(&self) -> Vec<PatternId> {
        self.echo_chamber_patterns_with_threshold(self.config.echo_chamber_threshold)
    }

    /// Get patterns with high echo chamber risk (custom threshold)
    pub fn echo_chamber_patterns_with_threshold(&self, threshold: f32) -> Vec<PatternId> {
        self.contexts
            .iter()
            .filter(|(_, ctx)| ctx.echo_chamber_risk() >= threshold)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get tension-resolving patterns
    pub fn tension_resolving_patterns(&self) -> Vec<PatternId> {
        self.contexts
            .iter()
            .filter(|(_, ctx)| ctx.tension_resolution_ratio() > 0.6)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get shadow patterns
    pub fn shadow_patterns(&self) -> Vec<PatternId> {
        self.contexts
            .iter()
            .filter(|(_, ctx)| ctx.in_shadow)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Add observation with overflow protection
    fn add_observation(&mut self, obs: CollectiveObservation) {
        self.observations.push(obs);
        if self.observations.len() > self.max_observations {
            // Remove oldest 10%
            let to_remove = self.max_observations / 10;
            self.observations.drain(0..to_remove);
        }
    }

    /// Get recent observations for a pattern
    pub fn recent_observations(&self, pattern_id: PatternId, limit: usize) -> Vec<&CollectiveObservation> {
        self.observations
            .iter()
            .rev()
            .filter(|o| o.pattern_id == pattern_id)
            .take(limit)
            .collect()
    }

    /// Get statistics
    pub fn stats(&self) -> CollectivePatternStats {
        let total = self.contexts.len();
        let emergent = self.emergent_patterns().len();
        let echo_risk = self.echo_chamber_patterns_with_threshold(0.7).len();
        let resolving = self.tension_resolving_patterns().len();
        let shadow = self.shadow_patterns().len();

        CollectivePatternStats {
            total_patterns_observed: total,
            emergent_patterns: emergent,
            echo_chamber_risk_patterns: echo_risk,
            tension_resolving_patterns: resolving,
            shadow_patterns: shadow,
            total_observations: self.observations.len(),
        }
    }
}

/// Statistics about collective pattern observations
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CollectivePatternStats {
    /// Total patterns with collective context
    pub total_patterns_observed: usize,

    /// Patterns discovered independently by multiple agents
    pub emergent_patterns: usize,

    /// Patterns with high echo chamber risk
    pub echo_chamber_risk_patterns: usize,

    /// Patterns that resolve group tension
    pub tension_resolving_patterns: usize,

    /// Patterns in the group shadow
    pub shadow_patterns: usize,

    /// Total observations recorded
    pub total_observations: usize,
}

impl CollectivePatternStats {
    /// Check if we have meaningful data
    pub fn has_data(&self) -> bool {
        self.total_patterns_observed > 0
    }

    /// Get emergence rate
    pub fn emergence_rate(&self) -> f32 {
        if self.total_patterns_observed == 0 {
            return 0.0;
        }
        self.emergent_patterns as f32 / self.total_patterns_observed as f32
    }

    /// Get echo chamber risk rate
    pub fn echo_risk_rate(&self) -> f32 {
        if self.total_patterns_observed == 0 {
            return 0.0;
        }
        self.echo_chamber_risk_patterns as f32 / self.total_patterns_observed as f32
    }
}

