//! Pattern Recommendations Engine
//!
//! Component 18: Synthesizes ALL signals to recommend optimal patterns for a given context.
//!
//! Unlike explainability (which explains *why*), this actively *selects* the best patterns
//! by combining trust scores, lifecycle state, collective signals, calibration accuracy,
//! dependency satisfaction, and domain relevance into a unified recommendation.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::collections::HashMap;

use super::{
    PatternId, DomainId, SymthaeaId,
    PatternLifecycleState,
    TrustLevel, Recommendation,
};

// ==============================================================================
// COMPONENT 18: PATTERN RECOMMENDATIONS ENGINE
// ==============================================================================
//
// This component synthesizes signals from ALL other components to recommend
// the best patterns for a given context. It acts as the "decision layer"
// that helps users choose which patterns to apply.
//
// Signal Sources:
// 1. Success Rate - Historical performance
// 2. Trust Score - K-Vector weighted credibility of pattern source
// 3. Lifecycle State - Prefer active, avoid deprecated
// 4. Collective Signals - Emergence, tension resolution, shadow detection
// 5. Calibration - How well-calibrated are the pattern's predictions
// 6. Dependencies - Are prerequisites satisfied?
// 7. Domain Relevance - How well does pattern match the context?
// 8. Recency - Recent patterns may be more relevant
//
// Philosophy: We recommend, not prescribe. Always provide multiple options
// with clear signal breakdowns so users can make informed choices.

/// Configuration for the recommendation engine
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RecommendationConfig {
    /// Weight for success rate signal (0.0-1.0)
    pub success_rate_weight: f32,

    /// Weight for trust score signal (0.0-1.0)
    pub trust_weight: f32,

    /// Weight for lifecycle state signal (0.0-1.0)
    pub lifecycle_weight: f32,

    /// Weight for collective signals (0.0-1.0)
    pub collective_weight: f32,

    /// Weight for calibration accuracy (0.0-1.0)
    pub calibration_weight: f32,

    /// Weight for dependency satisfaction (0.0-1.0)
    pub dependency_weight: f32,

    /// Weight for domain relevance (0.0-1.0)
    pub domain_weight: f32,

    /// Weight for recency (0.0-1.0)
    pub recency_weight: f32,

    /// Minimum composite score to include in recommendations
    pub min_score_threshold: f32,

    /// Maximum number of recommendations to return
    pub max_recommendations: usize,

    /// Penalize deprecated patterns
    pub penalize_deprecated: bool,

    /// Bonus for patterns validated in production
    pub production_validation_bonus: f32,

    /// Whether to include deprecated patterns in recommendations
    pub include_deprecated: bool,

    /// Diversity bonus - prefer diverse recommendations over similar ones
    pub diversity_bonus: f32,
}

impl Default for RecommendationConfig {
    fn default() -> Self {
        Self {
            success_rate_weight: 0.25,
            trust_weight: 0.15,
            lifecycle_weight: 0.10,
            collective_weight: 0.10,
            calibration_weight: 0.10,
            dependency_weight: 0.10,
            domain_weight: 0.15,
            recency_weight: 0.05,
            min_score_threshold: 0.3,
            max_recommendations: 10,
            penalize_deprecated: true,
            production_validation_bonus: 0.1,
            include_deprecated: false,
            diversity_bonus: 0.05,
        }
    }
}

impl RecommendationConfig {
    /// Create a config optimized for high-trust environments
    pub fn high_trust() -> Self {
        Self {
            trust_weight: 0.30,
            collective_weight: 0.20,
            success_rate_weight: 0.20,
            ..Default::default()
        }
    }

    /// Create a config optimized for production systems
    pub fn production() -> Self {
        Self {
            lifecycle_weight: 0.20,
            calibration_weight: 0.20,
            dependency_weight: 0.15,
            production_validation_bonus: 0.15,
            include_deprecated: false,
            ..Default::default()
        }
    }

    /// Create a config optimized for exploration/learning
    pub fn exploratory() -> Self {
        Self {
            recency_weight: 0.15,
            diversity_bonus: 0.15,
            include_deprecated: true,
            min_score_threshold: 0.2,
            ..Default::default()
        }
    }

    /// Total weight (should sum to ~1.0 for normalized scoring)
    pub fn total_weight(&self) -> f32 {
        self.success_rate_weight
            + self.trust_weight
            + self.lifecycle_weight
            + self.collective_weight
            + self.calibration_weight
            + self.dependency_weight
            + self.domain_weight
            + self.recency_weight
    }

    /// Normalize weights to sum to 1.0
    pub fn normalized(&self) -> Self {
        let total = self.total_weight();
        if total <= 0.0 {
            return self.clone();
        }
        Self {
            success_rate_weight: self.success_rate_weight / total,
            trust_weight: self.trust_weight / total,
            lifecycle_weight: self.lifecycle_weight / total,
            collective_weight: self.collective_weight / total,
            calibration_weight: self.calibration_weight / total,
            dependency_weight: self.dependency_weight / total,
            domain_weight: self.domain_weight / total,
            recency_weight: self.recency_weight / total,
            ..self.clone()
        }
    }
}

/// Breakdown of how each signal contributed to the composite score
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SignalBreakdown {
    /// Success rate contribution (raw value and weighted)
    pub success_rate: SignalContribution,

    /// Trust score contribution
    pub trust: SignalContribution,

    /// Lifecycle state contribution
    pub lifecycle: SignalContribution,

    /// Collective signals contribution
    pub collective: SignalContribution,

    /// Calibration accuracy contribution
    pub calibration: SignalContribution,

    /// Dependency satisfaction contribution
    pub dependencies: SignalContribution,

    /// Domain relevance contribution
    pub domain: SignalContribution,

    /// Recency contribution
    pub recency: SignalContribution,

    /// Bonuses applied (production validation, diversity, etc.)
    pub bonuses: f32,

    /// Penalties applied (deprecated, conflicts, etc.)
    pub penalties: f32,
}

impl SignalBreakdown {
    /// Get the total weighted contribution
    pub fn total(&self) -> f32 {
        self.success_rate.weighted
            + self.trust.weighted
            + self.lifecycle.weighted
            + self.collective.weighted
            + self.calibration.weighted
            + self.dependencies.weighted
            + self.domain.weighted
            + self.recency.weighted
            + self.bonuses
            - self.penalties
    }

    /// Get the strongest positive signal
    pub fn strongest_signal(&self) -> (&'static str, f32) {
        let signals = [
            ("success_rate", self.success_rate.weighted),
            ("trust", self.trust.weighted),
            ("lifecycle", self.lifecycle.weighted),
            ("collective", self.collective.weighted),
            ("calibration", self.calibration.weighted),
            ("dependencies", self.dependencies.weighted),
            ("domain", self.domain.weighted),
            ("recency", self.recency.weighted),
        ];
        signals
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(("none", 0.0))
    }

    /// Get the weakest signal (potential improvement area)
    pub fn weakest_signal(&self) -> (&'static str, f32) {
        let signals = [
            ("success_rate", self.success_rate.weighted),
            ("trust", self.trust.weighted),
            ("lifecycle", self.lifecycle.weighted),
            ("collective", self.collective.weighted),
            ("calibration", self.calibration.weighted),
            ("dependencies", self.dependencies.weighted),
            ("domain", self.domain.weighted),
            ("recency", self.recency.weighted),
        ];
        signals
            .into_iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(("none", 0.0))
    }

    /// Generate a human-readable summary
    pub fn summary(&self) -> String {
        let (strongest, strongest_val) = self.strongest_signal();
        let total = self.total();
        format!(
            "Score: {:.2} (strongest: {} at {:.2})",
            total, strongest, strongest_val
        )
    }
}

/// A single signal's contribution to the score
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SignalContribution {
    /// Raw signal value (0.0-1.0)
    pub raw: f32,

    /// Weight applied to this signal
    pub weight: f32,

    /// Weighted contribution (raw * weight)
    pub weighted: f32,

    /// Optional note about this signal
    pub note: Option<String>,
}

impl SignalContribution {
    /// Create a new signal contribution
    pub fn new(raw: f32, weight: f32) -> Self {
        Self {
            raw,
            weight,
            weighted: raw * weight,
            note: None,
        }
    }

    /// Create with a note
    pub fn with_note(raw: f32, weight: f32, note: &str) -> Self {
        Self {
            raw,
            weight,
            weighted: raw * weight,
            note: Some(note.to_string()),
        }
    }
}

/// Context for generating recommendations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RecommendationContext {
    /// Target domain(s) for recommendations
    pub target_domains: Vec<DomainId>,

    /// Currently active patterns (for dependency checking)
    pub active_patterns: Vec<PatternId>,

    /// Patterns to exclude from recommendations
    pub excluded_patterns: Vec<PatternId>,

    /// Requesting agent (for trust context)
    pub requesting_agent: Option<SymthaeaId>,

    /// Current timestamp
    pub timestamp: u64,

    /// Problem description (for semantic matching, future use)
    pub problem_description: Option<String>,

    /// Minimum trust level required
    pub min_trust_level: Option<TrustLevel>,

    /// Require production validation
    pub require_production_validated: bool,
}

impl RecommendationContext {
    /// Create a basic context with just timestamp
    pub fn new(timestamp: u64) -> Self {
        Self {
            target_domains: Vec::new(),
            active_patterns: Vec::new(),
            excluded_patterns: Vec::new(),
            requesting_agent: None,
            timestamp,
            problem_description: None,
            min_trust_level: None,
            require_production_validated: false,
        }
    }

    /// Create context for a specific domain
    pub fn for_domain(domain_id: DomainId, timestamp: u64) -> Self {
        Self {
            target_domains: vec![domain_id],
            ..Self::new(timestamp)
        }
    }

    /// Create context for multiple domains
    pub fn for_domains(domain_ids: Vec<DomainId>, timestamp: u64) -> Self {
        Self {
            target_domains: domain_ids,
            ..Self::new(timestamp)
        }
    }

    /// Add active patterns for dependency checking
    pub fn with_active_patterns(mut self, patterns: Vec<PatternId>) -> Self {
        self.active_patterns = patterns;
        self
    }

    /// Add patterns to exclude
    pub fn excluding(mut self, patterns: Vec<PatternId>) -> Self {
        self.excluded_patterns = patterns;
        self
    }

    /// Set requesting agent for trust context
    pub fn from_agent(mut self, agent_id: SymthaeaId) -> Self {
        self.requesting_agent = Some(agent_id);
        self
    }

    /// Set minimum trust level
    pub fn with_min_trust(mut self, level: TrustLevel) -> Self {
        self.min_trust_level = Some(level);
        self
    }

    /// Require production validation
    pub fn production_only(mut self) -> Self {
        self.require_production_validated = true;
        self
    }
}

/// A single pattern recommendation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatternRecommendation {
    /// The recommended pattern
    pub pattern_id: PatternId,

    /// Composite score (0.0-1.0)
    pub composite_score: f32,

    /// Confidence in this recommendation (0.0-1.0)
    pub confidence: f32,

    /// Rank in the recommendation list (1 = best)
    pub rank: u32,

    /// Detailed breakdown of contributing signals
    pub signal_breakdown: SignalBreakdown,

    /// Overall recommendation level
    pub recommendation_level: Recommendation,

    /// Brief explanation of why this pattern is recommended
    pub reason: String,

    /// Potential concerns or caveats
    pub caveats: Vec<String>,

    /// Whether this pattern is an alternative to others in the list
    pub is_alternative_to: Vec<PatternId>,

    /// Timestamp when recommendation was generated
    pub generated_at: u64,
}

impl PatternRecommendation {
    /// Create a new recommendation
    pub fn new(
        pattern_id: PatternId,
        composite_score: f32,
        confidence: f32,
        signal_breakdown: SignalBreakdown,
        timestamp: u64,
    ) -> Self {
        let recommendation_level = Recommendation::from_score(composite_score);
        Self {
            pattern_id,
            composite_score,
            confidence,
            rank: 0,
            signal_breakdown,
            recommendation_level,
            reason: String::new(),
            caveats: Vec::new(),
            is_alternative_to: Vec::new(),
            generated_at: timestamp,
        }
    }

    /// Set the reason for this recommendation
    pub fn with_reason(mut self, reason: &str) -> Self {
        self.reason = reason.to_string();
        self
    }

    /// Add a caveat
    pub fn with_caveat(mut self, caveat: &str) -> Self {
        self.caveats.push(caveat.to_string());
        self
    }

    /// Mark as alternative to other patterns
    pub fn as_alternative_to(mut self, patterns: Vec<PatternId>) -> Self {
        self.is_alternative_to = patterns;
        self
    }

    /// Get a short summary string
    pub fn summary(&self) -> String {
        format!(
            "#{} Pattern {} (score: {:.2}, confidence: {:.2}): {}",
            self.rank, self.pattern_id, self.composite_score, self.confidence,
            if self.reason.is_empty() { "No reason given" } else { &self.reason }
        )
    }

    /// Check if this is a strong recommendation
    pub fn is_strong(&self) -> bool {
        matches!(self.recommendation_level, Recommendation::StronglyRecommend)
    }

    /// Check if there are concerns
    pub fn has_concerns(&self) -> bool {
        !self.caveats.is_empty()
    }
}

/// A complete recommendation set for a query
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RecommendationSet {
    /// The recommendations, ordered by rank
    pub recommendations: Vec<PatternRecommendation>,

    /// Context used to generate these recommendations
    pub context: RecommendationContext,

    /// Total patterns considered
    pub patterns_considered: usize,

    /// Patterns filtered out (and why)
    pub filtered_count: usize,

    /// Summary of why patterns were filtered
    pub filter_reasons: HashMap<String, usize>,

    /// Overall confidence in this recommendation set
    pub set_confidence: f32,

    /// Timestamp when set was generated
    pub generated_at: u64,
}

impl RecommendationSet {
    /// Create a new recommendation set
    pub fn new(context: RecommendationContext, timestamp: u64) -> Self {
        Self {
            recommendations: Vec::new(),
            context,
            patterns_considered: 0,
            filtered_count: 0,
            filter_reasons: HashMap::new(),
            set_confidence: 0.0,
            generated_at: timestamp,
        }
    }

    /// Get the top recommendation
    pub fn top(&self) -> Option<&PatternRecommendation> {
        self.recommendations.first()
    }

    /// Get the top N recommendations
    pub fn top_n(&self, n: usize) -> Vec<&PatternRecommendation> {
        self.recommendations.iter().take(n).collect()
    }

    /// Check if the set is empty
    pub fn is_empty(&self) -> bool {
        self.recommendations.is_empty()
    }

    /// Get count of recommendations
    pub fn len(&self) -> usize {
        self.recommendations.len()
    }

    /// Get strong recommendations only
    pub fn strong_recommendations(&self) -> Vec<&PatternRecommendation> {
        self.recommendations.iter().filter(|r| r.is_strong()).collect()
    }

    /// Get recommendations with concerns
    pub fn with_concerns(&self) -> Vec<&PatternRecommendation> {
        self.recommendations.iter().filter(|r| r.has_concerns()).collect()
    }

    /// Format as human-readable string
    pub fn format(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!(
            "=== Recommendations ({} of {} patterns) ===\n",
            self.recommendations.len(),
            self.patterns_considered
        ));
        output.push_str(&format!("Set confidence: {:.1}%\n\n", self.set_confidence * 100.0));

        for rec in &self.recommendations {
            output.push_str(&format!("#{} Pattern {}\n", rec.rank, rec.pattern_id));
            output.push_str(&format!("   Score: {:.2} | Confidence: {:.2}\n",
                rec.composite_score, rec.confidence));
            output.push_str(&format!("   Level: {:?}\n", rec.recommendation_level));
            if !rec.reason.is_empty() {
                output.push_str(&format!("   Why: {}\n", rec.reason));
            }
            if !rec.caveats.is_empty() {
                output.push_str("   Caveats:\n");
                for caveat in &rec.caveats {
                    output.push_str(&format!("     - {}\n", caveat));
                }
            }
            output.push('\n');
        }

        if self.filtered_count > 0 {
            output.push_str(&format!("Filtered out: {} patterns\n", self.filtered_count));
            for (reason, count) in &self.filter_reasons {
                output.push_str(&format!("  - {}: {}\n", reason, count));
            }
        }

        output
    }
}

/// Input data for scoring a pattern
#[derive(Debug, Clone, Default)]
pub struct PatternSignals {
    /// Pattern success rate (0.0-1.0)
    pub success_rate: f32,

    /// Usage count
    pub usage_count: u64,

    /// Trust score from K-Vector (0.0-1.0)
    pub trust_score: f32,

    /// Trust level
    pub trust_level: Option<TrustLevel>,

    /// Lifecycle state
    pub lifecycle_state: Option<PatternLifecycleState>,

    /// Collective modifier (-1.0 to 1.0)
    pub collective_modifier: f32,

    /// Is emergent pattern
    pub is_emergent: bool,

    /// Resolves tension
    pub resolves_tension: bool,

    /// Calibration accuracy (0.0-1.0, higher is better)
    pub calibration_accuracy: f32,

    /// Dependencies satisfied ratio (0.0-1.0)
    pub dependencies_satisfied: f32,

    /// Has unmet required dependencies
    pub has_unmet_requirements: bool,

    /// Domain relevance score (0.0-1.0)
    pub domain_relevance: f32,

    /// Time since last use (for recency)
    pub time_since_last_use: u64,

    /// Validated in production
    pub production_validated: bool,

    /// Pattern age
    pub age: u64,
}

/// Statistics for the recommendation engine
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RecommendationStats {
    /// Total recommendations generated
    pub total_recommendations: u64,

    /// Total recommendation sets generated
    pub total_sets: u64,

    /// Average recommendations per set
    pub avg_recommendations_per_set: f32,

    /// Average confidence score
    pub avg_confidence: f32,

    /// Most recommended patterns
    pub most_recommended: Vec<(PatternId, u32)>,

    /// Patterns never recommended
    pub never_recommended_count: usize,

    /// Average filter rate
    pub avg_filter_rate: f32,

    /// Most common filter reasons
    pub common_filter_reasons: Vec<(String, u32)>,
}

/// The recommendation engine registry
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RecommendationRegistry {
    /// Configuration
    config: RecommendationConfig,

    /// Cache of recent recommendations (pattern_id -> last recommendation)
    cache: HashMap<PatternId, PatternRecommendation>,

    /// Cache TTL in time units
    cache_ttl: u64,

    /// Statistics tracking
    stats: RecommendationStats,

    /// Recommendation count per pattern
    recommendation_counts: HashMap<PatternId, u32>,
}

impl Default for RecommendationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl RecommendationRegistry {
    /// Create a new recommendation registry
    pub fn new() -> Self {
        Self {
            config: RecommendationConfig::default(),
            cache: HashMap::new(),
            cache_ttl: 3600, // 1 hour default
            stats: RecommendationStats::default(),
            recommendation_counts: HashMap::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: RecommendationConfig) -> Self {
        Self {
            config,
            ..Self::new()
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &RecommendationConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: RecommendationConfig) {
        self.config = config;
        self.cache.clear(); // Invalidate cache on config change
    }

    /// Score a single pattern given its signals
    pub fn score_pattern(
        &self,
        pattern_id: PatternId,
        signals: &PatternSignals,
        context: &RecommendationContext,
    ) -> Option<PatternRecommendation> {
        // Check exclusions
        if context.excluded_patterns.contains(&pattern_id) {
            return None;
        }

        // Check production validation requirement
        if context.require_production_validated && !signals.production_validated {
            return None;
        }

        // Check minimum trust level
        if let (Some(min_level), Some(actual_level)) = (&context.min_trust_level, &signals.trust_level) {
            if actual_level < min_level {
                return None;
            }
        }

        // Check lifecycle state - exclude archived/retired patterns
        if let Some(state) = &signals.lifecycle_state {
            if !state.is_usable() {
                return None;
            }
        }

        // Calculate signal contributions
        let config = self.config.normalized();
        let mut breakdown = SignalBreakdown::default();

        // Success rate signal
        let success_score = if signals.usage_count > 0 {
            signals.success_rate
        } else {
            0.5 // Neutral for unused patterns
        };
        breakdown.success_rate = SignalContribution::new(success_score, config.success_rate_weight);
        if signals.usage_count == 0 {
            breakdown.success_rate.note = Some("No usage data".to_string());
        }

        // Trust signal
        breakdown.trust = SignalContribution::new(signals.trust_score, config.trust_weight);

        // Lifecycle signal
        let lifecycle_score = match &signals.lifecycle_state {
            Some(PatternLifecycleState::Active) => 1.0,
            Some(PatternLifecycleState::Deprecated) => 0.3,
            Some(PatternLifecycleState::Archived) => 0.1,
            Some(PatternLifecycleState::Retired) => 0.0,
            None => 0.5,
        };
        breakdown.lifecycle = SignalContribution::new(lifecycle_score, config.lifecycle_weight);

        // Collective signal
        let collective_score = 0.5 + (signals.collective_modifier * 0.5); // Normalize to 0-1
        breakdown.collective = SignalContribution::new(collective_score, config.collective_weight);
        if signals.is_emergent {
            breakdown.collective.note = Some("Emergent pattern".to_string());
        } else if signals.resolves_tension {
            breakdown.collective.note = Some("Tension-resolving".to_string());
        }

        // Calibration signal
        breakdown.calibration = SignalContribution::new(signals.calibration_accuracy, config.calibration_weight);

        // Dependencies signal
        let dep_score = if signals.has_unmet_requirements {
            signals.dependencies_satisfied * 0.5 // Heavily penalize unmet requirements
        } else {
            signals.dependencies_satisfied
        };
        breakdown.dependencies = SignalContribution::new(dep_score, config.dependency_weight);
        if signals.has_unmet_requirements {
            breakdown.dependencies.note = Some("Has unmet requirements".to_string());
        }

        // Domain relevance signal
        breakdown.domain = SignalContribution::new(signals.domain_relevance, config.domain_weight);

        // Recency signal
        let recency_score = self.calculate_recency_score(signals.time_since_last_use);
        breakdown.recency = SignalContribution::new(recency_score, config.recency_weight);

        // Apply bonuses
        if signals.production_validated {
            breakdown.bonuses += self.config.production_validation_bonus;
        }

        // Apply penalties
        if self.config.penalize_deprecated {
            if let Some(PatternLifecycleState::Deprecated) = &signals.lifecycle_state {
                breakdown.penalties += 0.2;
            }
        }

        // Calculate composite score
        let composite_score = breakdown.total().clamp(0.0, 1.0);

        // Check minimum threshold
        if composite_score < self.config.min_score_threshold {
            return None;
        }

        // Calculate confidence based on data quality
        let confidence = self.calculate_confidence(signals);

        // Generate reason
        let (strongest, _) = breakdown.strongest_signal();
        let reason = self.generate_reason(strongest, signals);

        // Create recommendation
        let mut rec = PatternRecommendation::new(
            pattern_id,
            composite_score,
            confidence,
            breakdown,
            context.timestamp,
        );
        rec.reason = reason;

        // Add caveats
        if signals.usage_count < 10 {
            rec.caveats.push("Limited usage data".to_string());
        }
        if signals.has_unmet_requirements {
            rec.caveats.push("Has unmet dependencies".to_string());
        }
        if let Some(PatternLifecycleState::Deprecated) = &signals.lifecycle_state {
            rec.caveats.push("Pattern is deprecated".to_string());
        }

        Some(rec)
    }

    /// Calculate recency score from time since last use
    fn calculate_recency_score(&self, time_since_last_use: u64) -> f32 {
        // Exponential decay: score = e^(-time/halflife)
        // Halflife of 30 days (assuming time units are seconds)
        let halflife = 30 * 24 * 3600; // 30 days in seconds
        let decay_rate = 0.693 / halflife as f32; // ln(2) / halflife
        (-decay_rate * time_since_last_use as f32).exp()
    }

    /// Calculate confidence based on data quality
    fn calculate_confidence(&self, signals: &PatternSignals) -> f32 {
        let mut confidence = 0.5;

        // More usage = more confidence
        confidence += (signals.usage_count as f32 / 100.0).min(0.2);

        // Production validation = more confidence
        if signals.production_validated {
            confidence += 0.15;
        }

        // Good calibration = more confidence
        confidence += signals.calibration_accuracy * 0.1;

        // Known trust level = more confidence
        if signals.trust_level.is_some() {
            confidence += 0.05;
        }

        confidence.clamp(0.0, 1.0)
    }

    /// Generate a reason string based on strongest signal
    fn generate_reason(&self, strongest: &str, signals: &PatternSignals) -> String {
        match strongest {
            "success_rate" => format!(
                "High success rate ({:.0}% over {} uses)",
                signals.success_rate * 100.0,
                signals.usage_count
            ),
            "trust" => "From highly trusted source".to_string(),
            "lifecycle" => "Active and well-maintained pattern".to_string(),
            "collective" => {
                if signals.is_emergent {
                    "Emergent pattern gaining adoption".to_string()
                } else if signals.resolves_tension {
                    "Known to resolve conflicts".to_string()
                } else {
                    "Strong collective support".to_string()
                }
            }
            "calibration" => "Well-calibrated predictions".to_string(),
            "dependencies" => "All dependencies satisfied".to_string(),
            "domain" => "Highly relevant to your domain".to_string(),
            "recency" => "Recently validated and updated".to_string(),
            _ => "Recommended based on overall score".to_string(),
        }
    }

    /// Generate recommendations from a list of patterns and their signals
    pub fn generate_recommendations(
        &mut self,
        patterns: &[(PatternId, PatternSignals)],
        context: &RecommendationContext,
    ) -> RecommendationSet {
        let mut set = RecommendationSet::new(context.clone(), context.timestamp);
        set.patterns_considered = patterns.len();

        let mut recommendations: Vec<PatternRecommendation> = Vec::new();

        for (pattern_id, signals) in patterns {
            if let Some(rec) = self.score_pattern(*pattern_id, signals, context) {
                recommendations.push(rec);
            } else {
                set.filtered_count += 1;
                // Track filter reason
                if context.excluded_patterns.contains(pattern_id) {
                    *set.filter_reasons.entry("excluded".to_string()).or_insert(0) += 1;
                } else if context.require_production_validated && !signals.production_validated {
                    *set.filter_reasons.entry("not_production_validated".to_string()).or_insert(0) += 1;
                } else {
                    *set.filter_reasons.entry("below_threshold".to_string()).or_insert(0) += 1;
                }
            }
        }

        // Sort by composite score descending
        recommendations.sort_by(|a, b| {
            b.composite_score
                .partial_cmp(&a.composite_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply diversity bonus if configured
        if self.config.diversity_bonus > 0.0 {
            self.apply_diversity_bonus(&mut recommendations);
        }

        // Assign ranks and limit
        for (i, rec) in recommendations.iter_mut().enumerate() {
            rec.rank = (i + 1) as u32;
            *self.recommendation_counts.entry(rec.pattern_id).or_insert(0) += 1;
        }

        // Limit to max recommendations
        recommendations.truncate(self.config.max_recommendations);

        // Calculate set confidence
        set.set_confidence = if recommendations.is_empty() {
            0.0
        } else {
            recommendations.iter().map(|r| r.confidence).sum::<f32>() / recommendations.len() as f32
        };

        set.recommendations = recommendations;

        // Update stats
        self.stats.total_sets += 1;
        self.stats.total_recommendations += set.recommendations.len() as u64;

        set
    }

    /// Apply diversity bonus to recommendations
    fn apply_diversity_bonus(&self, recommendations: &mut [PatternRecommendation]) {
        // Simple diversity: penalize patterns that are alternatives to already-seen patterns
        // This is a placeholder for more sophisticated diversity algorithms
        let mut seen_domains: Vec<PatternId> = Vec::new();

        for rec in recommendations.iter_mut() {
            let diversity_penalty = seen_domains.iter()
                .filter(|&&id| rec.is_alternative_to.contains(&id))
                .count() as f32 * self.config.diversity_bonus;

            rec.composite_score = (rec.composite_score - diversity_penalty).max(0.0);
            seen_domains.push(rec.pattern_id);
        }
    }

    /// Clear the recommendation cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Invalidate cache for a specific pattern
    pub fn invalidate_cache(&mut self, pattern_id: PatternId) {
        self.cache.remove(&pattern_id);
    }

    /// Get statistics
    pub fn stats(&self) -> RecommendationStats {
        let mut stats = self.stats.clone();

        // Calculate average recommendations per set
        if stats.total_sets > 0 {
            stats.avg_recommendations_per_set =
                stats.total_recommendations as f32 / stats.total_sets as f32;
        }

        // Get most recommended patterns
        let mut counts: Vec<_> = self.recommendation_counts.iter().collect();
        counts.sort_by(|a, b| b.1.cmp(a.1));
        stats.most_recommended = counts.into_iter().take(10).map(|(&k, &v)| (k, v)).collect();

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recommendation_config_defaults() {
        let config = RecommendationConfig::default();
        assert!(config.total_weight() > 0.9);
        assert!(config.total_weight() < 1.1);
    }

    #[test]
    fn test_recommendation_config_normalized() {
        let config = RecommendationConfig {
            success_rate_weight: 0.5,
            trust_weight: 0.5,
            lifecycle_weight: 0.5,
            collective_weight: 0.5,
            calibration_weight: 0.5,
            dependency_weight: 0.5,
            domain_weight: 0.5,
            recency_weight: 0.5,
            ..Default::default()
        };
        let normalized = config.normalized();
        let total = normalized.total_weight();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_signal_contribution() {
        let contrib = SignalContribution::new(0.8, 0.25);
        assert_eq!(contrib.raw, 0.8);
        assert_eq!(contrib.weight, 0.25);
        assert!((contrib.weighted - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_signal_breakdown_total() {
        let mut breakdown = SignalBreakdown::default();
        breakdown.success_rate = SignalContribution::new(0.8, 0.25);
        breakdown.trust = SignalContribution::new(0.9, 0.25);
        breakdown.bonuses = 0.1;
        breakdown.penalties = 0.05;

        let total = breakdown.total();
        // 0.8*0.25 + 0.9*0.25 + 0.1 - 0.05 = 0.2 + 0.225 + 0.1 - 0.05 = 0.475
        assert!((total - 0.475).abs() < 0.01);
    }

    #[test]
    fn test_signal_breakdown_strongest() {
        let mut breakdown = SignalBreakdown::default();
        breakdown.success_rate = SignalContribution::new(0.5, 0.2);
        breakdown.trust = SignalContribution::new(0.9, 0.3); // Highest weighted
        breakdown.domain = SignalContribution::new(0.6, 0.2);

        let (name, _value) = breakdown.strongest_signal();
        assert_eq!(name, "trust");
    }

    #[test]
    fn test_recommendation_context_builder() {
        let ctx = RecommendationContext::for_domain(42, 1000)
            .with_active_patterns(vec![1, 2, 3])
            .excluding(vec![4, 5])
            .from_agent(99)
            .with_min_trust(TrustLevel::High)
            .production_only();

        assert_eq!(ctx.target_domains, vec![42]);
        assert_eq!(ctx.active_patterns, vec![1, 2, 3]);
        assert_eq!(ctx.excluded_patterns, vec![4, 5]);
        assert_eq!(ctx.requesting_agent, Some(99));
        assert_eq!(ctx.min_trust_level, Some(TrustLevel::High));
        assert!(ctx.require_production_validated);
    }

    #[test]
    fn test_pattern_recommendation_creation() {
        let breakdown = SignalBreakdown::default();
        let rec = PatternRecommendation::new(42, 0.85, 0.9, breakdown, 1000)
            .with_reason("Test reason")
            .with_caveat("Test caveat");

        assert_eq!(rec.pattern_id, 42);
        assert_eq!(rec.composite_score, 0.85);
        assert_eq!(rec.confidence, 0.9);
        assert_eq!(rec.reason, "Test reason");
        assert_eq!(rec.caveats.len(), 1);
        assert!(rec.is_strong());
    }

    #[test]
    fn test_recommendation_registry_score_pattern() {
        let registry = RecommendationRegistry::new();
        let context = RecommendationContext::new(1000);

        let signals = PatternSignals {
            success_rate: 0.8,
            usage_count: 100,
            trust_score: 0.9,
            trust_level: Some(TrustLevel::High),
            lifecycle_state: Some(PatternLifecycleState::Active),
            collective_modifier: 0.2,
            calibration_accuracy: 0.85,
            dependencies_satisfied: 1.0,
            domain_relevance: 0.9,
            production_validated: true,
            ..Default::default()
        };

        let rec = registry.score_pattern(1, &signals, &context);
        assert!(rec.is_some());
        let rec = rec.unwrap();
        assert!(rec.composite_score > 0.7);
        assert!(rec.confidence > 0.7);
    }

    #[test]
    fn test_recommendation_registry_excludes_pattern() {
        let registry = RecommendationRegistry::new();
        let context = RecommendationContext::new(1000).excluding(vec![42]);

        let signals = PatternSignals {
            success_rate: 0.9,
            ..Default::default()
        };

        let rec = registry.score_pattern(42, &signals, &context);
        assert!(rec.is_none());
    }

    #[test]
    fn test_recommendation_set_generation() {
        let mut registry = RecommendationRegistry::new();
        let context = RecommendationContext::new(1000);

        let patterns = vec![
            (1, PatternSignals { success_rate: 0.9, usage_count: 100, trust_score: 0.8, ..Default::default() }),
            (2, PatternSignals { success_rate: 0.7, usage_count: 50, trust_score: 0.9, ..Default::default() }),
            (3, PatternSignals { success_rate: 0.5, usage_count: 10, trust_score: 0.5, ..Default::default() }),
        ];

        let set = registry.generate_recommendations(&patterns, &context);

        assert!(!set.is_empty());
        assert_eq!(set.patterns_considered, 3);
        assert!(set.recommendations[0].rank == 1);

        // First recommendation should have highest score
        if set.recommendations.len() > 1 {
            assert!(set.recommendations[0].composite_score >= set.recommendations[1].composite_score);
        }
    }

    #[test]
    fn test_recommendation_set_format() {
        let mut registry = RecommendationRegistry::new();
        let context = RecommendationContext::new(1000);

        let patterns = vec![
            (1, PatternSignals { success_rate: 0.9, usage_count: 100, ..Default::default() }),
        ];

        let set = registry.generate_recommendations(&patterns, &context);
        let formatted = set.format();

        assert!(formatted.contains("Recommendations"));
        assert!(formatted.contains("Pattern 1"));
    }

    #[test]
    fn test_recency_score_calculation() {
        let registry = RecommendationRegistry::new();

        // Recent: high score
        let recent = registry.calculate_recency_score(0);
        assert!((recent - 1.0).abs() < 0.01);

        // Old: lower score
        let old = registry.calculate_recency_score(30 * 24 * 3600); // 30 days
        assert!(old < 0.6);
        assert!(old > 0.4);
    }
}
