//! Pattern Explainability: Human-Readable Explanations
//!
//! Component 17 provides human-readable explanations for why patterns are
//! recommended, succeed, or fail. This aligns with the project's emphasis on
//! "Justified, Wise Belief" - explanations are core to epistemic justification.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::collections::HashMap;

use super::{
    AgentTrustRegistry, CalibrationCurve, CollectivePatternRegistry, DependencyIssueType,
    DomainRegistry, PatternDependencyRegistry, PatternId, PatternLifecycleRegistry,
    PatternLifecycleState, SymthaeaPattern,
};

// ==============================================================================
// COMPONENT 17: PATTERN EXPLAINABILITY
// ==============================================================================
//
// This component provides human-readable explanations for pattern recommendations.
// The system generates explanations by analyzing multiple factors:
//
// 1. Success Rate - Historical performance of the pattern
// 2. Usage Count - How many times the pattern has been applied
// 3. Trust Score - K-Vector analysis of the pattern source
// 4. Lifecycle State - Whether pattern is active, deprecated, etc.
// 5. Domain Fit - How well the pattern matches the current domain
// 6. Dependency Status - Whether prerequisites are satisfied
// 7. Collective Signal - Swarm intelligence observations
// 8. Calibration Accuracy - How well predictions match outcomes
// 9. Recent Trends - Whether the pattern is improving or declining
//
// Philosophy: Explanations enable epistemic justification. A recommendation
// without explanation is mere assertion; a recommendation with explanation
// is justified belief.

/// Configuration for explanation generation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExplainabilityConfig {
    /// Include K-Vector trust analysis in explanations
    pub include_trust_factors: bool,

    /// Include lifecycle state and deprecation history
    pub include_lifecycle_factors: bool,

    /// Include dependency and prerequisite information
    pub include_dependency_factors: bool,

    /// Include domain context and fit analysis
    pub include_domain_factors: bool,

    /// Include collective/swarm intelligence signals
    pub include_collective_factors: bool,

    /// Include prediction calibration accuracy
    pub include_calibration_factors: bool,

    /// Include recent trend analysis (improvement/decline)
    pub include_trend_factors: bool,

    /// Maximum number of factors to include in explanation
    pub max_factors: usize,

    /// Minimum factor strength to include (0.0-1.0)
    pub min_factor_strength: f32,

    /// Cache TTL in seconds (0 = no caching)
    pub cache_ttl: u64,
}

impl Default for ExplainabilityConfig {
    fn default() -> Self {
        Self {
            include_trust_factors: true,
            include_lifecycle_factors: true,
            include_dependency_factors: true,
            include_domain_factors: true,
            include_collective_factors: true,
            include_calibration_factors: true,
            include_trend_factors: true,
            max_factors: 10,
            min_factor_strength: 0.1,
            cache_ttl: 300, // 5 minutes
        }
    }
}

/// Type of explanation factor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ExplanationFactorType {
    /// Pattern's historical success rate
    SuccessRate,
    /// Number of times pattern has been used
    UsageCount,
    /// Trust score of the pattern source (K-Vector)
    TrustScore,
    /// Pattern's lifecycle state (active, deprecated, etc.)
    LifecycleState,
    /// How well pattern fits the current domain
    DomainFit,
    /// Whether prerequisites/dependencies are satisfied
    DependencySatisfied,
    /// Collective/swarm intelligence signals
    CollectiveSignal,
    /// Prediction calibration accuracy
    CalibrationAccuracy,
    /// Recent performance trend
    RecentTrend,
    /// Production validation status
    ProductionValidated,
    /// Consciousness level at learning (Phi factor)
    PhiAtLearning,
    /// Superseded by another pattern
    Superseded,
}

impl ExplanationFactorType {
    /// Get human-readable name for this factor type
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::SuccessRate => "Success Rate",
            Self::UsageCount => "Usage Count",
            Self::TrustScore => "Source Trust",
            Self::LifecycleState => "Lifecycle State",
            Self::DomainFit => "Domain Fit",
            Self::DependencySatisfied => "Dependencies",
            Self::CollectiveSignal => "Collective Signal",
            Self::CalibrationAccuracy => "Calibration",
            Self::RecentTrend => "Recent Trend",
            Self::ProductionValidated => "Production Validated",
            Self::PhiAtLearning => "Learning Quality",
            Self::Superseded => "Superseded",
        }
    }
}

/// Impact direction of a factor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FactorImpact {
    /// Factor supports recommendation
    Positive,
    /// Factor opposes recommendation
    Negative,
    /// Factor is informational, no strong direction
    Neutral,
}

impl FactorImpact {
    /// Get symbol for display
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::Positive => "+",
            Self::Negative => "-",
            Self::Neutral => "~",
        }
    }
}

/// A single factor contributing to an explanation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExplanationFactor {
    /// Type of factor
    pub factor_type: ExplanationFactorType,

    /// Whether this factor supports or opposes the recommendation
    pub impact: FactorImpact,

    /// Importance weight of this factor (0.0-1.0)
    pub strength: f32,

    /// Human-readable description of the factor
    pub description: String,

    /// Optional supporting evidence or data
    pub evidence: Option<String>,
}

impl ExplanationFactor {
    /// Create a new explanation factor
    pub fn new(
        factor_type: ExplanationFactorType,
        impact: FactorImpact,
        strength: f32,
        description: impl Into<String>,
    ) -> Self {
        Self {
            factor_type,
            impact,
            strength: strength.clamp(0.0, 1.0),
            description: description.into(),
            evidence: None,
        }
    }

    /// Add evidence to the factor
    pub fn with_evidence(mut self, evidence: impl Into<String>) -> Self {
        self.evidence = Some(evidence.into());
        self
    }

    /// Format factor for display
    pub fn display(&self) -> String {
        let symbol = self.impact.symbol();
        format!(
            "  {} {} (strength: {:.0}%)",
            symbol,
            self.description,
            self.strength * 100.0
        )
    }
}

/// Recommendation level for a pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Recommendation {
    /// Strongly avoid using this pattern
    Avoid,
    /// Use with caution
    Caution,
    /// No strong opinion either way
    Neutral,
    /// Generally recommended
    Recommend,
    /// Highly recommended
    StronglyRecommend,
}

impl Recommendation {
    /// Get human-readable label
    pub fn label(&self) -> &'static str {
        match self {
            Self::Avoid => "AVOID",
            Self::Caution => "CAUTION",
            Self::Neutral => "NEUTRAL",
            Self::Recommend => "RECOMMEND",
            Self::StronglyRecommend => "STRONGLY RECOMMEND",
        }
    }

    /// Calculate recommendation from a weighted score (0.0-1.0)
    pub fn from_score(score: f32) -> Self {
        match score {
            s if s < 0.2 => Self::Avoid,
            s if s < 0.4 => Self::Caution,
            s if s < 0.6 => Self::Neutral,
            s if s < 0.8 => Self::Recommend,
            _ => Self::StronglyRecommend,
        }
    }
}

/// Complete explanation for a pattern
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatternExplanation {
    /// Pattern being explained
    pub pattern_id: PatternId,

    /// Brief 1-sentence summary
    pub summary: String,

    /// Overall recommendation
    pub recommendation: Recommendation,

    /// Confidence in this explanation (0.0-1.0)
    pub confidence: f32,

    /// Individual factors contributing to the explanation
    pub factors: Vec<ExplanationFactor>,

    /// Timestamp when explanation was generated
    pub generated_at: u64,
}

impl PatternExplanation {
    /// Create a new explanation
    pub fn new(pattern_id: PatternId, timestamp: u64) -> Self {
        Self {
            pattern_id,
            summary: String::new(),
            recommendation: Recommendation::Neutral,
            confidence: 0.5,
            factors: Vec::new(),
            generated_at: timestamp,
        }
    }

    /// Add a factor to the explanation
    pub fn add_factor(&mut self, factor: ExplanationFactor) {
        self.factors.push(factor);
    }

    /// Get positive factors only
    pub fn positive_factors(&self) -> Vec<&ExplanationFactor> {
        self.factors
            .iter()
            .filter(|f| f.impact == FactorImpact::Positive)
            .collect()
    }

    /// Get negative factors only
    pub fn negative_factors(&self) -> Vec<&ExplanationFactor> {
        self.factors
            .iter()
            .filter(|f| f.impact == FactorImpact::Negative)
            .collect()
    }

    /// Get neutral factors only
    pub fn neutral_factors(&self) -> Vec<&ExplanationFactor> {
        self.factors
            .iter()
            .filter(|f| f.impact == FactorImpact::Neutral)
            .collect()
    }

    /// Sort factors by strength (strongest first)
    pub fn sort_by_strength(&mut self) {
        self.factors.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Limit factors to top N
    pub fn limit_factors(&mut self, max: usize) {
        self.sort_by_strength();
        self.factors.truncate(max);
    }

    /// Format the full explanation for display
    pub fn format_full(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "Pattern #{}: \"{}\"\n\n",
            self.pattern_id, self.summary
        ));
        output.push_str(&format!(
            "Recommendation: {} (confidence: {:.0}%)\n\n",
            self.recommendation.label(),
            self.confidence * 100.0
        ));

        let positive = self.positive_factors();
        if !positive.is_empty() {
            output.push_str("Why this works:\n");
            for factor in positive {
                output.push_str(&factor.display());
                output.push('\n');
            }
            output.push('\n');
        }

        let negative = self.negative_factors();
        if !negative.is_empty() {
            output.push_str("Potential risks:\n");
            for factor in negative {
                output.push_str(&factor.display());
                output.push('\n');
            }
        }

        output
    }

    /// Format a brief one-liner explanation
    pub fn format_brief(&self) -> String {
        format!(
            "{}: {} (confidence: {:.0}%)",
            self.recommendation.label(),
            self.summary,
            self.confidence * 100.0
        )
    }
}

/// Comparison between two patterns
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatternComparison {
    /// First pattern
    pub pattern_a_id: PatternId,
    /// Second pattern
    pub pattern_b_id: PatternId,
    /// Which pattern is recommended
    pub preferred: Option<PatternId>,
    /// Reason for preference
    pub preference_reason: String,
    /// Explanation for pattern A
    pub explanation_a: PatternExplanation,
    /// Explanation for pattern B
    pub explanation_b: PatternExplanation,
    /// Key differentiating factors
    pub differentiators: Vec<String>,
}

impl PatternComparison {
    /// Create a new comparison
    pub fn new(explanation_a: PatternExplanation, explanation_b: PatternExplanation) -> Self {
        let pattern_a_id = explanation_a.pattern_id;
        let pattern_b_id = explanation_b.pattern_id;

        // Determine preference based on recommendation and confidence
        let score_a = (explanation_a.recommendation as u8 as f32) * explanation_a.confidence;
        let score_b = (explanation_b.recommendation as u8 as f32) * explanation_b.confidence;

        let (preferred, preference_reason) = if (score_a - score_b).abs() < 0.1 {
            (None, "Both patterns are roughly equivalent".to_string())
        } else if score_a > score_b {
            (
                Some(pattern_a_id),
                format!("Pattern {} has better overall score", pattern_a_id),
            )
        } else {
            (
                Some(pattern_b_id),
                format!("Pattern {} has better overall score", pattern_b_id),
            )
        };

        Self {
            pattern_a_id,
            pattern_b_id,
            preferred,
            preference_reason,
            explanation_a,
            explanation_b,
            differentiators: Vec::new(),
        }
    }

    /// Add a differentiating factor
    pub fn add_differentiator(&mut self, diff: impl Into<String>) {
        self.differentiators.push(diff.into());
    }
}

/// Cached explanation entry
#[derive(Debug, Clone)]
struct CachedExplanation {
    explanation: PatternExplanation,
    expires_at: u64,
}

/// Registry for tracking and caching pattern explanations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExplainabilityRegistry {
    /// Configuration
    pub config: ExplainabilityConfig,

    /// Explanation cache
    #[cfg_attr(feature = "serde", serde(skip))]
    cache: HashMap<PatternId, CachedExplanation>,

    /// Statistics tracking
    stats: ExplainabilityStatsInternal,
}

/// Internal statistics tracking
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct ExplainabilityStatsInternal {
    explanations_generated: u64,
    cache_hits: u64,
    cache_misses: u64,
    total_factors: u64,
    factor_counts: HashMap<ExplanationFactorType, u32>,
    positive_factor_counts: HashMap<ExplanationFactorType, u32>,
    negative_factor_counts: HashMap<ExplanationFactorType, u32>,
}

impl Default for ExplainabilityRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ExplainabilityRegistry {
    /// Create a new registry with default config
    pub fn new() -> Self {
        Self {
            config: ExplainabilityConfig::default(),
            cache: HashMap::new(),
            stats: ExplainabilityStatsInternal::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: ExplainabilityConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
            stats: ExplainabilityStatsInternal::default(),
        }
    }

    /// Generate an explanation for a pattern
    ///
    /// This is the main API method that queries all registries to build
    /// a comprehensive explanation.
    pub fn explain_pattern(
        &mut self,
        pattern: &SymthaeaPattern,
        trust_registry: &AgentTrustRegistry,
        lifecycle_registry: &PatternLifecycleRegistry,
        dependency_registry: &PatternDependencyRegistry,
        domain_registry: &DomainRegistry,
        collective_registry: &CollectivePatternRegistry,
        calibration: &CalibrationCurve,
        timestamp: u64,
    ) -> PatternExplanation {
        let pattern_id = pattern.pattern_id;

        // Check cache first
        if let Some(cached) = self.get_cached(pattern_id, timestamp) {
            self.stats.cache_hits += 1;
            return cached;
        }

        self.stats.cache_misses += 1;

        // Build explanation from scratch
        let mut explanation = PatternExplanation::new(pattern_id, timestamp);

        // Generate all factors
        let mut all_factors = Vec::new();

        // 1. Success rate factor
        all_factors.push(self.generate_success_rate_factor(pattern));

        // 2. Usage count factor
        all_factors.push(self.generate_usage_count_factor(pattern));

        // 3. Trust factors (K-Vector)
        if self.config.include_trust_factors {
            if let Some(factor) = self.generate_trust_factor(pattern, trust_registry) {
                all_factors.push(factor);
            }
        }

        // 4. Lifecycle factors
        if self.config.include_lifecycle_factors {
            if let Some(factor) = self.generate_lifecycle_factor(pattern, lifecycle_registry) {
                all_factors.push(factor);
            }
        }

        // 5. Dependency factors
        if self.config.include_dependency_factors {
            if let Some(factor) = self.generate_dependency_factor(pattern, dependency_registry) {
                all_factors.push(factor);
            }
        }

        // 6. Domain factors
        if self.config.include_domain_factors {
            if let Some(factor) = self.generate_domain_factor(pattern, domain_registry) {
                all_factors.push(factor);
            }
        }

        // 7. Collective factors
        if self.config.include_collective_factors {
            if let Some(factor) = self.generate_collective_factor(pattern, collective_registry) {
                all_factors.push(factor);
            }
        }

        // 8. Calibration factors
        if self.config.include_calibration_factors {
            if let Some(factor) = self.generate_calibration_factor(calibration) {
                all_factors.push(factor);
            }
        }

        // 9. Production validation factor
        all_factors.push(self.generate_production_validation_factor(pattern));

        // 10. Phi at learning factor
        all_factors.push(self.generate_phi_factor(pattern));

        // Filter by minimum strength and add to explanation
        for factor in all_factors {
            if factor.strength >= self.config.min_factor_strength {
                self.record_factor_stats(&factor);
                explanation.add_factor(factor);
            }
        }

        // Limit factors
        explanation.limit_factors(self.config.max_factors);

        // Calculate recommendation and confidence
        self.calculate_recommendation(&mut explanation);

        // Generate summary
        explanation.summary = self.generate_summary(pattern, &explanation);

        // Update stats
        self.stats.explanations_generated += 1;

        // Cache the result
        if self.config.cache_ttl > 0 {
            self.cache.insert(
                pattern_id,
                CachedExplanation {
                    explanation: explanation.clone(),
                    expires_at: timestamp + self.config.cache_ttl,
                },
            );
        }

        explanation
    }

    /// Get cached explanation if still valid
    fn get_cached(&self, pattern_id: PatternId, current_time: u64) -> Option<PatternExplanation> {
        self.cache
            .get(&pattern_id)
            .filter(|cached| cached.expires_at > current_time)
            .map(|cached| cached.explanation.clone())
    }

    /// Invalidate cache for a specific pattern
    pub fn invalidate_cache(&mut self, pattern_id: PatternId) {
        self.cache.remove(&pattern_id);
    }

    /// Clear all cached explanations
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Generate success rate factor
    fn generate_success_rate_factor(&self, pattern: &SymthaeaPattern) -> ExplanationFactor {
        let rate = pattern.success_rate;
        let (impact, description) = match rate {
            r if r >= 0.8 => (
                FactorImpact::Positive,
                format!("High success rate ({:.0}%)", r * 100.0),
            ),
            r if r >= 0.6 => (
                FactorImpact::Positive,
                format!("Good success rate ({:.0}%)", r * 100.0),
            ),
            r if r >= 0.4 => (
                FactorImpact::Neutral,
                format!("Moderate success rate ({:.0}%)", r * 100.0),
            ),
            r if r >= 0.2 => (
                FactorImpact::Negative,
                format!("Low success rate ({:.0}%)", r * 100.0),
            ),
            r => (
                FactorImpact::Negative,
                format!("Very low success rate ({:.0}%)", r * 100.0),
            ),
        };

        let strength = if rate >= 0.5 { rate } else { 1.0 - rate };

        ExplanationFactor::new(
            ExplanationFactorType::SuccessRate,
            impact,
            strength,
            description,
        )
        .with_evidence(format!(
            "{}/{} successful uses",
            pattern.success_count, pattern.usage_count
        ))
    }

    /// Generate usage count factor
    fn generate_usage_count_factor(&self, pattern: &SymthaeaPattern) -> ExplanationFactor {
        let count = pattern.usage_count;
        let (impact, description, strength) = match count {
            c if c >= 100 => (
                FactorImpact::Positive,
                format!("Well-tested ({} uses)", c),
                0.8,
            ),
            c if c >= 50 => (
                FactorImpact::Positive,
                format!("Moderately tested ({} uses)", c),
                0.6,
            ),
            c if c >= 10 => (
                FactorImpact::Neutral,
                format!("Some testing ({} uses)", c),
                0.4,
            ),
            c if c >= 1 => (
                FactorImpact::Neutral,
                format!("Limited testing ({} uses)", c),
                0.3,
            ),
            _ => (FactorImpact::Negative, "Never used".to_string(), 0.2),
        };

        ExplanationFactor::new(
            ExplanationFactorType::UsageCount,
            impact,
            strength,
            description,
        )
    }

    /// Generate trust factor from K-Vector analysis
    fn generate_trust_factor(
        &self,
        pattern: &SymthaeaPattern,
        trust_registry: &AgentTrustRegistry,
    ) -> Option<ExplanationFactor> {
        let agent_id = pattern.learned_by;
        let trust_context = trust_registry.get(agent_id)?;
        let trust_score = trust_context.trust_score();

        let (impact, description) = match trust_score {
            t if t >= 0.8 => (
                FactorImpact::Positive,
                format!("From highly trusted source (K={:.2})", t),
            ),
            t if t >= 0.6 => (
                FactorImpact::Positive,
                format!("From trusted source (K={:.2})", t),
            ),
            t if t >= 0.4 => (
                FactorImpact::Neutral,
                format!("From moderately trusted source (K={:.2})", t),
            ),
            t if t >= 0.2 => (
                FactorImpact::Negative,
                format!("From low-trust source (K={:.2})", t),
            ),
            t => (
                FactorImpact::Negative,
                format!("From untrusted source (K={:.2})", t),
            ),
        };

        Some(ExplanationFactor::new(
            ExplanationFactorType::TrustScore,
            impact,
            trust_score.abs().max(0.1), // Ensure minimum strength
            description,
        ))
    }

    /// Generate lifecycle factor
    fn generate_lifecycle_factor(
        &self,
        pattern: &SymthaeaPattern,
        lifecycle_registry: &PatternLifecycleRegistry,
    ) -> Option<ExplanationFactor> {
        let info = lifecycle_registry.get(pattern.pattern_id)?;

        let (impact, description, strength) = match info.state {
            PatternLifecycleState::Active => {
                if info.deprecation_count == 0 {
                    (
                        FactorImpact::Positive,
                        "Active pattern, never deprecated".to_string(),
                        0.7,
                    )
                } else {
                    (
                        FactorImpact::Neutral,
                        format!(
                            "Active pattern (resurrected {} times)",
                            info.resurrection_count
                        ),
                        0.5,
                    )
                }
            }
            PatternLifecycleState::Deprecated => {
                let reason = info
                    .deprecation_reason
                    .as_ref()
                    .map(|r| format!("{:?}", r))
                    .unwrap_or_else(|| "unknown reason".to_string());
                (
                    FactorImpact::Negative,
                    format!("Deprecated: {}", reason),
                    0.8,
                )
            }
            PatternLifecycleState::Archived => (
                FactorImpact::Negative,
                "Archived pattern (no longer maintained)".to_string(),
                0.9,
            ),
            PatternLifecycleState::Retired => (
                FactorImpact::Negative,
                "Retired pattern (do not use)".to_string(),
                1.0,
            ),
        };

        // Check for superseded
        if let Some(replacement_id) = info.replacement_id {
            return Some(ExplanationFactor::new(
                ExplanationFactorType::Superseded,
                FactorImpact::Negative,
                0.9,
                format!("Superseded by pattern #{}", replacement_id),
            ));
        }

        Some(ExplanationFactor::new(
            ExplanationFactorType::LifecycleState,
            impact,
            strength,
            description,
        ))
    }

    /// Generate dependency factor
    fn generate_dependency_factor(
        &self,
        pattern: &SymthaeaPattern,
        dependency_registry: &PatternDependencyRegistry,
    ) -> Option<ExplanationFactor> {
        // Pass empty active_patterns for basic resolution
        let resolution = dependency_registry.resolve(pattern.pattern_id, &[]);
        let total_deps = resolution.prerequisites.len() + resolution.required.len();

        if resolution.issues.is_empty() {
            if total_deps == 0 {
                // No dependencies
                Some(ExplanationFactor::new(
                    ExplanationFactorType::DependencySatisfied,
                    FactorImpact::Neutral,
                    0.3,
                    "No prerequisites required".to_string(),
                ))
            } else {
                // All dependencies satisfied
                Some(ExplanationFactor::new(
                    ExplanationFactorType::DependencySatisfied,
                    FactorImpact::Positive,
                    0.7,
                    format!("All {} prerequisites satisfied", total_deps),
                ))
            }
        } else {
            // Some dependencies not satisfied
            let missing_count = resolution
                .issues
                .iter()
                .filter(|i| matches!(i.issue_type, DependencyIssueType::MissingRequired))
                .count();
            let unmet_count = resolution
                .issues
                .iter()
                .filter(|i| matches!(i.issue_type, DependencyIssueType::UnmetPrerequisite))
                .count();

            let description = if missing_count > 0 {
                format!("{} required prerequisites missing", missing_count)
            } else {
                format!("{} prerequisites unmet", unmet_count)
            };

            let impact = if missing_count > 0 {
                FactorImpact::Negative
            } else {
                FactorImpact::Neutral
            };

            Some(ExplanationFactor::new(
                ExplanationFactorType::DependencySatisfied,
                impact,
                if missing_count > 0 { 0.8 } else { 0.3 },
                description,
            ))
        }
    }

    /// Generate domain factor
    fn generate_domain_factor(
        &self,
        pattern: &SymthaeaPattern,
        domain_registry: &DomainRegistry,
    ) -> Option<ExplanationFactor> {
        if pattern.domain_ids.is_empty() {
            return Some(ExplanationFactor::new(
                ExplanationFactorType::DomainFit,
                FactorImpact::Neutral,
                0.2,
                "No domain classification".to_string(),
            ));
        }

        // Get domain names for display
        let domain_names: Vec<String> = pattern
            .domain_ids
            .iter()
            .filter_map(|&id| domain_registry.get(id))
            .map(|d| d.name.clone())
            .collect();

        if domain_names.is_empty() {
            return None;
        }

        let description = format!("Matches domain: {}", domain_names.join(", "));

        Some(ExplanationFactor::new(
            ExplanationFactorType::DomainFit,
            FactorImpact::Positive,
            0.5,
            description,
        ))
    }

    /// Generate collective/swarm factor
    fn generate_collective_factor(
        &self,
        pattern: &SymthaeaPattern,
        collective_registry: &CollectivePatternRegistry,
    ) -> Option<ExplanationFactor> {
        let context = collective_registry.get(pattern.pattern_id)?;

        // Check for emergent pattern (multiple independent discoveries)
        if context.independent_discoveries >= 3 {
            return Some(ExplanationFactor::new(
                ExplanationFactorType::CollectiveSignal,
                FactorImpact::Positive,
                0.8,
                format!(
                    "Emerging pattern ({} independent discoveries)",
                    context.independent_discoveries
                ),
            ));
        }

        // Check for tension resolution
        if context.tension_resolutions > context.tension_increases {
            return Some(ExplanationFactor::new(
                ExplanationFactorType::CollectiveSignal,
                FactorImpact::Positive,
                0.6,
                "Pattern tends to resolve group tension".to_string(),
            ));
        }

        // Check for echo chamber warning
        if context.avg_agreement_at_usage > 0.9 && context.high_agreement_usages > 5 {
            return Some(ExplanationFactor::new(
                ExplanationFactorType::CollectiveSignal,
                FactorImpact::Negative,
                0.5,
                "Pattern may be echo-chamber reinforced".to_string(),
            ));
        }

        // Check for shadow pattern
        if context.in_shadow {
            return Some(ExplanationFactor::new(
                ExplanationFactorType::CollectiveSignal,
                FactorImpact::Neutral,
                0.4,
                "Pattern appears in collective shadow (often overlooked)".to_string(),
            ));
        }

        None
    }

    /// Generate calibration factor
    fn generate_calibration_factor(
        &self,
        calibration: &CalibrationCurve,
    ) -> Option<ExplanationFactor> {
        // Use brier_score as the calibration error measure (lower is better, 0 = perfect)
        let calibration_error = calibration.brier_score;

        // Skip if not enough data
        if calibration.total_predictions < 10 {
            return None;
        }

        let (impact, description) = match calibration_error {
            e if e < 0.1 => (
                FactorImpact::Positive,
                format!("Predictions well-calibrated (error: {:.1}%)", e * 100.0),
            ),
            e if e < 0.2 => (
                FactorImpact::Neutral,
                format!(
                    "Predictions moderately calibrated (error: {:.1}%)",
                    e * 100.0
                ),
            ),
            e => (
                FactorImpact::Negative,
                format!("Predictions poorly calibrated (error: {:.1}%)", e * 100.0),
            ),
        };

        Some(ExplanationFactor::new(
            ExplanationFactorType::CalibrationAccuracy,
            impact,
            (1.0_f32 - calibration_error).max(0.1),
            description,
        ))
    }

    /// Generate production validation factor
    fn generate_production_validation_factor(
        &self,
        pattern: &SymthaeaPattern,
    ) -> ExplanationFactor {
        if pattern.validated_in_production {
            ExplanationFactor::new(
                ExplanationFactorType::ProductionValidated,
                FactorImpact::Positive,
                0.6,
                "Validated in production".to_string(),
            )
        } else {
            ExplanationFactor::new(
                ExplanationFactorType::ProductionValidated,
                FactorImpact::Neutral,
                0.3,
                "Not yet validated in production".to_string(),
            )
        }
    }

    /// Generate phi (consciousness level) factor
    fn generate_phi_factor(&self, pattern: &SymthaeaPattern) -> ExplanationFactor {
        let phi = pattern.phi_at_learning;

        let (impact, description) = match phi {
            p if p >= 0.8 => (
                FactorImpact::Positive,
                format!("Learned at high awareness (Phi={:.2})", p),
            ),
            p if p >= 0.5 => (
                FactorImpact::Neutral,
                format!("Learned at moderate awareness (Phi={:.2})", p),
            ),
            p => (
                FactorImpact::Neutral,
                format!("Learned at low awareness (Phi={:.2})", p),
            ),
        };

        ExplanationFactor::new(
            ExplanationFactorType::PhiAtLearning,
            impact,
            phi.max(0.2),
            description,
        )
    }

    /// Record factor statistics
    fn record_factor_stats(&mut self, factor: &ExplanationFactor) {
        self.stats.total_factors += 1;
        *self
            .stats
            .factor_counts
            .entry(factor.factor_type)
            .or_insert(0) += 1;

        match factor.impact {
            FactorImpact::Positive => {
                *self
                    .stats
                    .positive_factor_counts
                    .entry(factor.factor_type)
                    .or_insert(0) += 1;
            }
            FactorImpact::Negative => {
                *self
                    .stats
                    .negative_factor_counts
                    .entry(factor.factor_type)
                    .or_insert(0) += 1;
            }
            FactorImpact::Neutral => {}
        }
    }

    /// Calculate recommendation based on factors
    fn calculate_recommendation(&self, explanation: &mut PatternExplanation) {
        if explanation.factors.is_empty() {
            explanation.recommendation = Recommendation::Neutral;
            explanation.confidence = 0.3;
            return;
        }

        let mut weighted_score = 0.0;
        let mut total_weight = 0.0;

        for factor in &explanation.factors {
            let base_score = match factor.impact {
                FactorImpact::Positive => 1.0,
                FactorImpact::Neutral => 0.5,
                FactorImpact::Negative => 0.0,
            };
            weighted_score += base_score * factor.strength;
            total_weight += factor.strength;
        }

        let final_score = if total_weight > 0.0 {
            weighted_score / total_weight
        } else {
            0.5
        };

        explanation.recommendation = Recommendation::from_score(final_score);

        // Confidence based on factor count and weight distribution
        let factor_count_bonus = (explanation.factors.len() as f32 / 10.0).min(1.0) * 0.3;
        explanation.confidence = (final_score * 0.7 + factor_count_bonus).min(1.0);
    }

    /// Generate summary text for explanation
    fn generate_summary(
        &self,
        pattern: &SymthaeaPattern,
        explanation: &PatternExplanation,
    ) -> String {
        let recommendation_text = match explanation.recommendation {
            Recommendation::StronglyRecommend => "Highly recommended",
            Recommendation::Recommend => "Recommended",
            Recommendation::Neutral => "Neutral",
            Recommendation::Caution => "Use with caution",
            Recommendation::Avoid => "Not recommended",
        };

        let domain_info = if pattern.problem_domain.is_empty() {
            pattern.solution.clone()
        } else {
            format!("{}: {}", pattern.problem_domain, pattern.solution)
        };

        format!("{} pattern for {}", recommendation_text, domain_info)
    }

    /// Compare two patterns
    pub fn compare_patterns(
        &mut self,
        pattern_a: &SymthaeaPattern,
        pattern_b: &SymthaeaPattern,
        trust_registry: &AgentTrustRegistry,
        lifecycle_registry: &PatternLifecycleRegistry,
        dependency_registry: &PatternDependencyRegistry,
        domain_registry: &DomainRegistry,
        collective_registry: &CollectivePatternRegistry,
        calibration: &CalibrationCurve,
        timestamp: u64,
    ) -> PatternComparison {
        let explanation_a = self.explain_pattern(
            pattern_a,
            trust_registry,
            lifecycle_registry,
            dependency_registry,
            domain_registry,
            collective_registry,
            calibration,
            timestamp,
        );

        let explanation_b = self.explain_pattern(
            pattern_b,
            trust_registry,
            lifecycle_registry,
            dependency_registry,
            domain_registry,
            collective_registry,
            calibration,
            timestamp,
        );

        let mut comparison = PatternComparison::new(explanation_a, explanation_b);

        // Find key differentiators
        if pattern_a.success_rate != pattern_b.success_rate {
            let diff = (pattern_a.success_rate - pattern_b.success_rate).abs() * 100.0;
            comparison.add_differentiator(format!("Success rate differs by {:.0}%", diff));
        }

        if pattern_a.usage_count != pattern_b.usage_count {
            comparison.add_differentiator(format!(
                "Pattern A has {} uses vs Pattern B with {} uses",
                pattern_a.usage_count, pattern_b.usage_count
            ));
        }

        comparison
    }

    /// Get user-friendly explanation for why a pattern is recommended
    pub fn explain_recommendation(&self, explanation: &PatternExplanation) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "Recommendation: {}\n",
            explanation.recommendation.label()
        ));
        output.push('\n');

        let positive = explanation.positive_factors();
        if !positive.is_empty() {
            output.push_str("Reasons to use this pattern:\n");
            for (i, factor) in positive.iter().enumerate().take(3) {
                output.push_str(&format!("  {}. {}\n", i + 1, factor.description));
            }
        }

        let negative = explanation.negative_factors();
        if !negative.is_empty() {
            output.push('\n');
            output.push_str("Points to consider:\n");
            for (i, factor) in negative.iter().enumerate().take(3) {
                output.push_str(&format!("  {}. {}\n", i + 1, factor.description));
            }
        }

        output
    }

    /// Get just the success factors as a formatted string
    pub fn explain_success_factors(&self, explanation: &PatternExplanation) -> String {
        let positive = explanation.positive_factors();
        if positive.is_empty() {
            return "No clear success factors identified.".to_string();
        }

        let mut output = String::from("Success factors:\n");
        for factor in positive {
            output.push_str(&format!("  - {}", factor.description));
            if let Some(evidence) = &factor.evidence {
                output.push_str(&format!(" ({})", evidence));
            }
            output.push('\n');
        }
        output
    }

    /// Get just the risk factors as a formatted string
    pub fn explain_risk_factors(&self, explanation: &PatternExplanation) -> String {
        let negative = explanation.negative_factors();
        if negative.is_empty() {
            return "No significant risks identified.".to_string();
        }

        let mut output = String::from("Risk factors:\n");
        for factor in negative {
            output.push_str(&format!("  - {}", factor.description));
            if let Some(evidence) = &factor.evidence {
                output.push_str(&format!(" ({})", evidence));
            }
            output.push('\n');
        }
        output
    }

    /// Get statistics about explanation generation
    pub fn stats(&self) -> ExplainabilityStats {
        // Find most common positive/negative factors
        let mut positive_vec: Vec<_> = self.stats.positive_factor_counts.iter().collect();
        positive_vec.sort_by(|a, b| b.1.cmp(a.1));

        let mut negative_vec: Vec<_> = self.stats.negative_factor_counts.iter().collect();
        negative_vec.sort_by(|a, b| b.1.cmp(a.1));

        let most_common_positive = positive_vec
            .iter()
            .take(5)
            .map(|(&ft, &count)| (ft, count))
            .collect();

        let most_common_negative = negative_vec
            .iter()
            .take(5)
            .map(|(&ft, &count)| (ft, count))
            .collect();

        let avg_factors = if self.stats.explanations_generated > 0 {
            self.stats.total_factors as f32 / self.stats.explanations_generated as f32
        } else {
            0.0
        };

        ExplainabilityStats {
            explanations_generated: self.stats.explanations_generated,
            cache_hits: self.stats.cache_hits,
            cache_misses: self.stats.cache_misses,
            avg_factors_per_explanation: avg_factors,
            most_common_positive_factors: most_common_positive,
            most_common_negative_factors: most_common_negative,
        }
    }

    // ==========================================================================
    // IMPROVEMENT PLAN GENERATION
    // ==========================================================================

    /// Generate an improvement plan for a pattern
    ///
    /// This analyzes the current explanation and generates:
    /// - Counterfactual factors (what would need to change)
    /// - Action suggestions (concrete steps to improve)
    /// - Feasibility assessment
    pub fn generate_improvement_plan(
        &mut self,
        pattern: &SymthaeaPattern,
        explanation: &PatternExplanation,
        timestamp: u64,
    ) -> ImprovementPlan {
        let mut plan = ImprovementPlan::new(
            pattern.pattern_id,
            explanation.recommendation,
            explanation.confidence,
            timestamp,
        );

        // Set target based on current state
        plan.target_recommendation = match explanation.recommendation {
            Recommendation::Avoid | Recommendation::Caution => Recommendation::Recommend,
            Recommendation::Neutral => Recommendation::Recommend,
            Recommendation::Recommend => Recommendation::StronglyRecommend,
            Recommendation::StronglyRecommend => Recommendation::StronglyRecommend,
        };

        // Generate counterfactuals and actions for each negative factor
        for factor in explanation.negative_factors() {
            if let Some(cf) = self.generate_counterfactual(pattern, factor) {
                plan.add_counterfactual(cf);
            }
            if let Some(action) = self.generate_action_for_factor(pattern, factor) {
                plan.add_action(action);
            }
        }

        // Also generate actions for neutral factors that could be improved
        for factor in explanation.neutral_factors() {
            if factor.strength > 0.4 {
                if let Some(action) = self.generate_action_for_factor(pattern, factor) {
                    plan.add_action(action);
                }
            }
        }

        // Sort actions by ROI
        plan.sort_actions_by_roi();

        // Calculate feasibility
        let score_gap = 0.8 - explanation.confidence; // Target 80% for StronglyRecommend
        let hard_count = plan
            .actions
            .iter()
            .filter(|a| a.effort == ActionEffortLevel::Hard)
            .count();
        plan.feasibility = PlanFeasibility::from_gap_and_effort(score_gap, hard_count);

        // Update projected score
        plan.projected_score = (plan.current_score + plan.total_potential_improvement()).min(1.0);
        plan.target_recommendation = Recommendation::from_score(plan.projected_score);

        plan
    }

    /// Generate a counterfactual for a negative factor
    fn generate_counterfactual(
        &self,
        pattern: &SymthaeaPattern,
        factor: &ExplanationFactor,
    ) -> Option<CounterfactualFactor> {
        match factor.factor_type {
            ExplanationFactorType::SuccessRate => {
                let current = format!("{:.0}%", pattern.success_rate * 100.0);
                let target = "75%+".to_string();
                let gap = (0.75 - pattern.success_rate).max(0.0);
                Some(
                    CounterfactualFactor::new(
                        factor.factor_type,
                        current,
                        target,
                        gap,
                        gap * 0.3, // Success rate has ~30% impact
                    )
                    .with_description(format!(
                        "Improve success rate from {:.0}% to 75%+ through testing and refinement",
                        pattern.success_rate * 100.0
                    )),
                )
            }
            ExplanationFactorType::UsageCount => {
                let current = format!("{} uses", pattern.usage_count);
                let target = "100+ uses".to_string();
                let gap = if pattern.usage_count < 100 {
                    (100 - pattern.usage_count) as f32 / 100.0
                } else {
                    0.0
                };
                Some(
                    CounterfactualFactor::new(
                        factor.factor_type,
                        current,
                        target,
                        gap,
                        gap * 0.15, // Usage has ~15% impact
                    )
                    .with_description(format!(
                        "Increase usage from {} to 100+ to build confidence",
                        pattern.usage_count
                    )),
                )
            }
            ExplanationFactorType::TrustScore => Some(
                CounterfactualFactor::new(
                    factor.factor_type,
                    "Low trust",
                    "High trust (K > 0.7)",
                    0.3,
                    0.15,
                )
                .with_description(
                    "Build trust through consistent positive contributions and vouching",
                ),
            ),
            ExplanationFactorType::LifecycleState => Some(
                CounterfactualFactor::new(
                    factor.factor_type,
                    "Deprecated/Archived",
                    "Active",
                    0.5,
                    0.2,
                )
                .with_description("Resurrect or replace with an active pattern variant"),
            ),
            ExplanationFactorType::DependencySatisfied => Some(
                CounterfactualFactor::new(
                    factor.factor_type,
                    "Dependencies missing",
                    "All dependencies satisfied",
                    0.4,
                    0.15,
                )
                .with_description("Resolve missing prerequisites before using this pattern"),
            ),
            ExplanationFactorType::ProductionValidated => Some(
                CounterfactualFactor::new(
                    factor.factor_type,
                    "Not validated",
                    "Production validated",
                    0.3,
                    0.1,
                )
                .with_description(
                    "Validate pattern in production environment to increase confidence",
                ),
            ),
            ExplanationFactorType::CalibrationAccuracy => Some(
                CounterfactualFactor::new(
                    factor.factor_type,
                    "Poorly calibrated",
                    "Well calibrated (<10% error)",
                    0.3,
                    0.1,
                )
                .with_description("Improve prediction accuracy through more outcome tracking"),
            ),
            _ => None, // Domain fit and collective signals are harder to change
        }
    }

    /// Generate an action suggestion for a factor
    fn generate_action_for_factor(
        &self,
        pattern: &SymthaeaPattern,
        factor: &ExplanationFactor,
    ) -> Option<ActionSuggestion> {
        match factor.factor_type {
            ExplanationFactorType::SuccessRate => {
                if pattern.success_rate < 0.5 {
                    Some(
                        ActionSuggestion::new(
                            "Review and fix failure cases causing low success rate",
                            9,
                            0.25,
                            ActionEffortLevel::Hard,
                            factor.factor_type,
                        )
                        .with_rationale(format!(
                            "Current {:.0}% success rate is below acceptable threshold",
                            pattern.success_rate * 100.0
                        )),
                    )
                } else if pattern.success_rate < 0.75 {
                    Some(
                        ActionSuggestion::new(
                            "Analyze edge cases to improve success rate",
                            7,
                            0.15,
                            ActionEffortLevel::Medium,
                            factor.factor_type,
                        )
                        .with_rationale("Moderate success rate has room for improvement"),
                    )
                } else {
                    None
                }
            }
            ExplanationFactorType::UsageCount => {
                if pattern.usage_count < 10 {
                    Some(
                        ActionSuggestion::new(
                            "Increase pattern usage to build statistical confidence",
                            5,
                            0.1,
                            ActionEffortLevel::Easy,
                            factor.factor_type,
                        )
                        .with_rationale(
                            "Low usage count means insufficient data for reliable assessment",
                        ),
                    )
                } else {
                    None
                }
            }
            ExplanationFactorType::TrustScore => Some(
                ActionSuggestion::new(
                    "Request vouching from trusted community members",
                    6,
                    0.12,
                    ActionEffortLevel::Medium,
                    factor.factor_type,
                )
                .with_rationale("Higher trust score improves pattern credibility"),
            ),
            ExplanationFactorType::LifecycleState => Some(
                ActionSuggestion::new(
                    "Consider creating an updated version of this pattern",
                    8,
                    0.2,
                    ActionEffortLevel::Hard,
                    factor.factor_type,
                )
                .with_rationale("Deprecated patterns should be replaced with improved versions"),
            ),
            ExplanationFactorType::DependencySatisfied => Some(
                ActionSuggestion::new(
                    "Resolve missing pattern dependencies",
                    9,
                    0.18,
                    ActionEffortLevel::Medium,
                    factor.factor_type,
                )
                .with_rationale("Missing dependencies prevent effective pattern application"),
            ),
            ExplanationFactorType::ProductionValidated => Some(
                ActionSuggestion::new(
                    "Deploy pattern in production with monitoring",
                    7,
                    0.12,
                    ActionEffortLevel::Medium,
                    factor.factor_type,
                )
                .with_rationale("Production validation significantly increases confidence"),
            ),
            ExplanationFactorType::CalibrationAccuracy => Some(
                ActionSuggestion::new(
                    "Track more outcomes to improve prediction calibration",
                    4,
                    0.08,
                    ActionEffortLevel::Easy,
                    factor.factor_type,
                )
                .with_rationale("Better calibration improves prediction reliability"),
            ),
            ExplanationFactorType::DomainFit => Some(
                ActionSuggestion::new(
                    "Add domain classification to improve discoverability",
                    3,
                    0.05,
                    ActionEffortLevel::Easy,
                    factor.factor_type,
                )
                .with_rationale("Proper domain classification helps users find relevant patterns"),
            ),
            ExplanationFactorType::CollectiveSignal => Some(
                ActionSuggestion::new(
                    "Promote pattern in relevant communities",
                    4,
                    0.08,
                    ActionEffortLevel::Medium,
                    factor.factor_type,
                )
                .with_rationale("Collective adoption signals increase pattern visibility"),
            ),
            ExplanationFactorType::RecentTrend => Some(
                ActionSuggestion::new(
                    "Investigate recent decline and address root causes",
                    8,
                    0.15,
                    ActionEffortLevel::Hard,
                    factor.factor_type,
                )
                .with_rationale("Declining trends indicate emerging problems"),
            ),
            ExplanationFactorType::PhiAtLearning => Some(
                ActionSuggestion::new(
                    "Improve pattern documentation for better initial understanding",
                    3,
                    0.05,
                    ActionEffortLevel::Easy,
                    factor.factor_type,
                )
                .with_rationale("Better phi (coherence) at learning time improves adoption"),
            ),
            ExplanationFactorType::Superseded => Some(
                ActionSuggestion::new(
                    "Migrate to the superseding pattern or create an improved version",
                    9,
                    0.2,
                    ActionEffortLevel::Hard,
                    factor.factor_type,
                )
                .with_rationale("This pattern has been superseded by a newer version"),
            ),
        }
    }

    /// Generate improvement plan directly from pattern with all registries
    pub fn generate_improvement_plan_full(
        &mut self,
        pattern: &SymthaeaPattern,
        trust_registry: &AgentTrustRegistry,
        lifecycle_registry: &PatternLifecycleRegistry,
        dependency_registry: &PatternDependencyRegistry,
        domain_registry: &DomainRegistry,
        collective_registry: &CollectivePatternRegistry,
        calibration: &CalibrationCurve,
        timestamp: u64,
    ) -> ImprovementPlan {
        let explanation = self.explain_pattern(
            pattern,
            trust_registry,
            lifecycle_registry,
            dependency_registry,
            domain_registry,
            collective_registry,
            calibration,
            timestamp,
        );
        self.generate_improvement_plan(pattern, &explanation, timestamp)
    }
}

/// Public statistics about explainability
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExplainabilityStats {
    /// Total explanations generated
    pub explanations_generated: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Average factors per explanation
    pub avg_factors_per_explanation: f32,
    /// Most common positive factors
    pub most_common_positive_factors: Vec<(ExplanationFactorType, u32)>,
    /// Most common negative factors
    pub most_common_negative_factors: Vec<(ExplanationFactorType, u32)>,
}

impl ExplainabilityStats {
    /// Calculate cache hit rate
    pub fn cache_hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f32 / total as f32
        }
    }
}

// ==============================================================================
// ENHANCEMENT: COUNTERFACTUAL EXPLANATIONS
// ==============================================================================
//
// Counterfactual explanations answer "What would need to change?"
// They help users understand:
// - Why a pattern isn't recommended
// - What improvements would change the recommendation
// - The gap between current state and target state

/// A counterfactual factor explaining what would need to change
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CounterfactualFactor {
    /// Which factor type this relates to
    pub factor_type: ExplanationFactorType,

    /// Current value description
    pub current_value: String,

    /// Target value needed for improvement
    pub target_value: String,

    /// Gap between current and target (0.0-1.0, where 1.0 = far from target)
    pub gap: f32,

    /// How much this would improve the recommendation score
    pub potential_improvement: f32,

    /// Human-readable description of the change needed
    pub description: String,
}

impl CounterfactualFactor {
    /// Create a new counterfactual factor
    pub fn new(
        factor_type: ExplanationFactorType,
        current_value: impl Into<String>,
        target_value: impl Into<String>,
        gap: f32,
        potential_improvement: f32,
    ) -> Self {
        let current = current_value.into();
        let target = target_value.into();
        let description = format!(
            "If {} changed from '{}' to '{}', recommendation would improve",
            factor_type.display_name(),
            current,
            target
        );

        Self {
            factor_type,
            current_value: current,
            target_value: target,
            gap: gap.clamp(0.0, 1.0),
            potential_improvement: potential_improvement.clamp(0.0, 1.0),
            description,
        }
    }

    /// Create with custom description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Format for display
    pub fn display(&self) -> String {
        format!(
            "  → {} → {} (potential +{:.0}% improvement)",
            self.current_value,
            self.target_value,
            self.potential_improvement * 100.0
        )
    }
}

// ==============================================================================
// ENHANCEMENT: ACTIONABLE SUGGESTIONS
// ==============================================================================
//
// Action suggestions provide concrete steps users can take to improve
// a pattern's standing. Each action is:
// - Specific and actionable
// - Prioritized by impact
// - Linked to the factor it addresses

/// Effort level for an action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ActionEffortLevel {
    /// Quick fix, minimal effort
    Easy,
    /// Moderate effort required
    Medium,
    /// Significant effort or resources needed
    Hard,
}

impl ActionEffortLevel {
    /// Get human-readable label
    pub fn label(&self) -> &'static str {
        match self {
            Self::Easy => "Easy",
            Self::Medium => "Medium",
            Self::Hard => "Hard",
        }
    }

    /// Get emoji indicator
    pub fn indicator(&self) -> &'static str {
        match self {
            Self::Easy => "[*]",
            Self::Medium => "[**]",
            Self::Hard => "[***]",
        }
    }
}

/// A specific action suggestion to improve a pattern
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ActionSuggestion {
    /// The action to take
    pub action: String,

    /// Priority (1-10, higher = more important)
    pub priority: u8,

    /// Estimated impact on recommendation score (0.0-1.0)
    pub estimated_impact: f32,

    /// How much effort this requires
    pub effort: ActionEffortLevel,

    /// Which factor type this addresses
    pub addresses_factor: ExplanationFactorType,

    /// Additional context or rationale
    pub rationale: Option<String>,
}

impl ActionSuggestion {
    /// Create a new action suggestion
    pub fn new(
        action: impl Into<String>,
        priority: u8,
        estimated_impact: f32,
        effort: ActionEffortLevel,
        addresses_factor: ExplanationFactorType,
    ) -> Self {
        Self {
            action: action.into(),
            priority: priority.clamp(1, 10),
            estimated_impact: estimated_impact.clamp(0.0, 1.0),
            effort,
            addresses_factor,
            rationale: None,
        }
    }

    /// Add rationale
    pub fn with_rationale(mut self, rationale: impl Into<String>) -> Self {
        self.rationale = Some(rationale.into());
        self
    }

    /// Calculate impact-to-effort ratio (higher = better ROI)
    pub fn roi_score(&self) -> f32 {
        let effort_multiplier = match self.effort {
            ActionEffortLevel::Easy => 1.0,
            ActionEffortLevel::Medium => 0.5,
            ActionEffortLevel::Hard => 0.25,
        };
        self.estimated_impact * effort_multiplier * (self.priority as f32 / 10.0)
    }

    /// Format for display
    pub fn display(&self) -> String {
        format!(
            "  {} {} (impact: +{:.0}%, effort: {})",
            self.effort.indicator(),
            self.action,
            self.estimated_impact * 100.0,
            self.effort.label()
        )
    }
}

// ==============================================================================
// IMPROVEMENT PLAN
// ==============================================================================
//
// An improvement plan combines counterfactuals and actions into a
// comprehensive roadmap for improving a pattern's recommendation.

/// Complete improvement plan for a pattern
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ImprovementPlan {
    /// Pattern this plan is for
    pub pattern_id: PatternId,

    /// Current recommendation level
    pub current_recommendation: Recommendation,

    /// Target recommendation level
    pub target_recommendation: Recommendation,

    /// Current weighted score
    pub current_score: f32,

    /// Estimated score after improvements
    pub projected_score: f32,

    /// Counterfactual factors (what would need to change)
    pub counterfactuals: Vec<CounterfactualFactor>,

    /// Concrete action suggestions
    pub actions: Vec<ActionSuggestion>,

    /// Overall feasibility assessment
    pub feasibility: PlanFeasibility,

    /// When this plan was generated
    pub generated_at: u64,
}

/// How feasible is this improvement plan?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PlanFeasibility {
    /// Easy to achieve with minor changes
    HighlyFeasible,
    /// Achievable with moderate effort
    Feasible,
    /// Challenging but possible
    Challenging,
    /// Would require significant changes
    Difficult,
    /// Pattern has fundamental issues
    Impractical,
}

impl PlanFeasibility {
    /// Get human-readable label
    pub fn label(&self) -> &'static str {
        match self {
            Self::HighlyFeasible => "Highly Feasible",
            Self::Feasible => "Feasible",
            Self::Challenging => "Challenging",
            Self::Difficult => "Difficult",
            Self::Impractical => "Impractical",
        }
    }

    /// Calculate from score gap and action count
    pub fn from_gap_and_effort(score_gap: f32, hard_action_count: usize) -> Self {
        match (score_gap, hard_action_count) {
            (gap, _) if gap < 0.1 => Self::HighlyFeasible,
            (gap, hard) if gap < 0.2 && hard == 0 => Self::Feasible,
            (gap, hard) if gap < 0.3 && hard <= 1 => Self::Challenging,
            (gap, hard) if gap < 0.5 || hard <= 2 => Self::Difficult,
            _ => Self::Impractical,
        }
    }
}

impl ImprovementPlan {
    /// Create a new improvement plan
    pub fn new(
        pattern_id: PatternId,
        current_recommendation: Recommendation,
        current_score: f32,
        timestamp: u64,
    ) -> Self {
        Self {
            pattern_id,
            current_recommendation,
            target_recommendation: Recommendation::StronglyRecommend,
            current_score,
            projected_score: current_score,
            counterfactuals: Vec::new(),
            actions: Vec::new(),
            feasibility: PlanFeasibility::Feasible,
            generated_at: timestamp,
        }
    }

    /// Add a counterfactual factor
    pub fn add_counterfactual(&mut self, cf: CounterfactualFactor) {
        self.projected_score += cf.potential_improvement;
        self.counterfactuals.push(cf);
    }

    /// Add an action suggestion
    pub fn add_action(&mut self, action: ActionSuggestion) {
        self.actions.push(action);
    }

    /// Sort actions by ROI (best return on investment first)
    pub fn sort_actions_by_roi(&mut self) {
        self.actions.sort_by(|a, b| {
            b.roi_score()
                .partial_cmp(&a.roi_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Sort actions by priority
    pub fn sort_actions_by_priority(&mut self) {
        self.actions.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Get top N actions by ROI
    pub fn top_actions(&self, n: usize) -> Vec<&ActionSuggestion> {
        let mut sorted: Vec<_> = self.actions.iter().collect();
        sorted.sort_by(|a, b| {
            b.roi_score()
                .partial_cmp(&a.roi_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(n).collect()
    }

    /// Get quick wins (high impact, low effort)
    pub fn quick_wins(&self) -> Vec<&ActionSuggestion> {
        self.actions
            .iter()
            .filter(|a| a.effort == ActionEffortLevel::Easy && a.estimated_impact >= 0.1)
            .collect()
    }

    /// Calculate total potential improvement
    pub fn total_potential_improvement(&self) -> f32 {
        self.counterfactuals
            .iter()
            .map(|cf| cf.potential_improvement)
            .sum()
    }

    /// Format as human-readable plan
    pub fn format(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "Improvement Plan for Pattern #{}\n",
            self.pattern_id
        ));
        output.push_str(&format!(
            "Current: {} → Target: {}\n",
            self.current_recommendation.label(),
            self.target_recommendation.label()
        ));
        output.push_str(&format!(
            "Score: {:.0}% → Projected: {:.0}%\n",
            self.current_score * 100.0,
            self.projected_score.min(1.0) * 100.0
        ));
        output.push_str(&format!("Feasibility: {}\n\n", self.feasibility.label()));

        if !self.counterfactuals.is_empty() {
            output.push_str("What would need to change:\n");
            for cf in &self.counterfactuals {
                output.push_str(&cf.display());
                output.push('\n');
            }
            output.push('\n');
        }

        if !self.actions.is_empty() {
            output.push_str("Recommended Actions:\n");
            for (i, action) in self.actions.iter().enumerate().take(5) {
                output.push_str(&format!("  {}. {}\n", i + 1, action.action));
                if let Some(rationale) = &action.rationale {
                    output.push_str(&format!("     Rationale: {}\n", rationale));
                }
            }
        }

        output
    }

    /// Format as brief summary
    pub fn format_brief(&self) -> String {
        let quick_wins = self.quick_wins();
        if quick_wins.is_empty() {
            format!(
                "Pattern #{}: {} → {} ({} actions needed)",
                self.pattern_id,
                self.current_recommendation.label(),
                self.target_recommendation.label(),
                self.actions.len()
            )
        } else {
            format!(
                "Pattern #{}: {} quick wins available, best: {}",
                self.pattern_id,
                quick_wins.len(),
                quick_wins
                    .first()
                    .map(|a| a.action.as_str())
                    .unwrap_or("none")
            )
        }
    }
}
