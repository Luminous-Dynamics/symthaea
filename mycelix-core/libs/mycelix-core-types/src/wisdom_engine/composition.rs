//! Pattern Composition: Combining Patterns for Synergy
//!
//! Component 11 enables combining multiple patterns to achieve synergistic effects.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::collections::HashMap;

use super::{CompositeId, DomainId, PatternId, SymthaeaId};

// ==============================================================================
// COMPONENT 11: PATTERN COMPOSITION - Combining Patterns for Synergy
// ==============================================================================
//
// Real-world solutions often combine multiple patterns. This component enables:
//
// 1. Creating composite patterns (A + B + C working together)
// 2. Tracking whether combinations outperform individuals (synergy)
// 3. Automatic discovery of patterns that work well together
// 4. Different composition modes (sequential, parallel, conditional)
//
// Key insight: A pattern with 70% success + another with 70% success
// might achieve 90% when combined (synergy > 1.0) or 50% (interference < 1.0)

/// Unique identifier for a pattern composite

/// How patterns in a composite are combined
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CompositionType {
    /// Apply patterns in sequence: A then B then C
    Sequential,

    /// Apply patterns together simultaneously
    Parallel,

    /// Apply first pattern, use second as fallback if first fails
    Fallback,

    /// Choose pattern based on context (requires selector)
    Conditional,

    /// All patterns must succeed for composite to succeed
    AllRequired,

    /// Any pattern succeeding counts as composite success
    AnyRequired,
}

impl Default for CompositionType {
    fn default() -> Self {
        Self::Parallel
    }
}

/// A composite pattern combining multiple individual patterns
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatternComposite {
    /// Unique identifier
    pub id: CompositeId,

    /// Human-readable name
    pub name: String,

    /// Description of why these patterns work together
    pub rationale: String,

    /// The patterns being combined (order matters for Sequential)
    pub pattern_ids: Vec<PatternId>,

    /// How the patterns are combined
    pub composition_type: CompositionType,

    /// Domains this composite applies to
    pub domain_ids: Vec<DomainId>,

    /// Total times this composite was used
    pub usage_count: u32,

    /// Times the composite succeeded
    pub success_count: u32,

    /// Synergy score: composite_rate / expected_rate
    /// > 1.0 = patterns enhance each other
    /// = 1.0 = no interaction
    /// < 1.0 = patterns interfere with each other
    pub synergy_score: f32,

    /// Expected success rate based on individual patterns
    /// (calculated based on composition type)
    pub expected_success_rate: f32,

    /// Actual observed success rate
    pub actual_success_rate: f32,

    /// Statistical confidence in synergy score (0.0-1.0)
    pub synergy_confidence: f32,

    /// When this composite was created
    pub created_at: u64,

    /// Who created this composite
    pub created_by: SymthaeaId,

    /// Whether this was auto-discovered vs manually created
    pub auto_discovered: bool,

    /// Last time this composite was used
    pub last_used: u64,
}

impl PatternComposite {
    /// Create a new pattern composite
    pub fn new(
        id: CompositeId,
        name: &str,
        pattern_ids: Vec<PatternId>,
        composition_type: CompositionType,
        created_at: u64,
        created_by: SymthaeaId,
    ) -> Self {
        Self {
            id,
            name: name.to_string(),
            rationale: String::new(),
            pattern_ids,
            composition_type,
            domain_ids: Vec::new(),
            usage_count: 0,
            success_count: 0,
            synergy_score: 1.0, // Neutral until we have data
            expected_success_rate: 0.0,
            actual_success_rate: 0.0,
            synergy_confidence: 0.0,
            created_at,
            created_by,
            auto_discovered: false,
            last_used: created_at,
        }
    }

    /// Add a rationale explaining why these patterns work together
    pub fn with_rationale(mut self, rationale: &str) -> Self {
        self.rationale = rationale.to_string();
        self
    }

    /// Add domain associations
    pub fn with_domains(mut self, domain_ids: Vec<DomainId>) -> Self {
        self.domain_ids = domain_ids;
        self
    }

    /// Mark as auto-discovered
    pub fn as_auto_discovered(mut self) -> Self {
        self.auto_discovered = true;
        self
    }

    /// Record a usage outcome
    pub fn record_outcome(&mut self, success: bool, timestamp: u64) {
        self.usage_count += 1;
        if success {
            self.success_count += 1;
        }
        self.last_used = timestamp;
        self.update_rates();
    }

    /// Update success rates and synergy score
    fn update_rates(&mut self) {
        if self.usage_count > 0 {
            self.actual_success_rate = self.success_count as f32 / self.usage_count as f32;

            // Calculate synergy if we have expected rate
            if self.expected_success_rate > 0.0 {
                self.synergy_score = self.actual_success_rate / self.expected_success_rate;
            }

            // Confidence increases with more observations
            // Using a simple sqrt scaling (caps around 100 uses)
            self.synergy_confidence = (self.usage_count as f32 / 100.0).sqrt().min(1.0);
        }
    }

    /// Calculate expected success rate based on individual pattern rates
    pub fn calculate_expected_rate(&mut self, individual_rates: &[f32]) {
        if individual_rates.is_empty() {
            self.expected_success_rate = 0.0;
            return;
        }

        self.expected_success_rate = match self.composition_type {
            CompositionType::Sequential | CompositionType::AllRequired => {
                // All must succeed: multiply probabilities
                individual_rates.iter().product()
            }
            CompositionType::Parallel => {
                // Average of individual rates (simplified model)
                individual_rates.iter().sum::<f32>() / individual_rates.len() as f32
            }
            CompositionType::Fallback | CompositionType::AnyRequired => {
                // At least one must succeed: 1 - P(all fail)
                1.0 - individual_rates.iter().map(|r| 1.0 - r).product::<f32>()
            }
            CompositionType::Conditional => {
                // Depends on selector, use average as approximation
                individual_rates.iter().sum::<f32>() / individual_rates.len() as f32
            }
        };

        // Recalculate synergy if we have actual data
        if self.usage_count > 0 {
            self.update_rates();
        }
    }

    /// Check if this composite shows significant synergy
    pub fn has_synergy(&self) -> bool {
        self.synergy_confidence >= 0.5 && self.synergy_score > 1.1
    }

    /// Check if this composite shows interference
    pub fn has_interference(&self) -> bool {
        self.synergy_confidence >= 0.5 && self.synergy_score < 0.9
    }

    /// Get the synergy status as a human-readable string
    pub fn synergy_status(&self) -> &'static str {
        if self.synergy_confidence < 0.3 {
            "insufficient data"
        } else if self.synergy_score > 1.2 {
            "strong synergy"
        } else if self.synergy_score > 1.05 {
            "mild synergy"
        } else if self.synergy_score > 0.95 {
            "neutral"
        } else if self.synergy_score > 0.8 {
            "mild interference"
        } else {
            "strong interference"
        }
    }

    /// Check if composite contains a specific pattern
    pub fn contains_pattern(&self, pattern_id: PatternId) -> bool {
        self.pattern_ids.contains(&pattern_id)
    }

    /// Check if composite is in a specific domain
    pub fn in_domain(&self, domain_id: DomainId) -> bool {
        self.domain_ids.contains(&domain_id)
    }
}

/// A candidate for automatic synergy discovery
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SynergyCandidate {
    /// Patterns that might work well together
    pub pattern_ids: Vec<PatternId>,

    /// Estimated synergy potential (0.0-2.0+)
    pub estimated_synergy: f32,

    /// Why we think these might synergize
    pub reason: SynergyReason,

    /// Confidence in this candidate (0.0-1.0)
    pub confidence: f32,

    /// Suggested composition type
    pub suggested_type: CompositionType,
}

/// Reasons why patterns might synergize
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SynergyReason {
    /// Patterns are frequently used together successfully
    CooccurrenceSuccess,

    /// Patterns are in related domains
    DomainAffinity,

    /// Patterns have complementary characteristics
    Complementary,

    /// One pattern's weakness is another's strength
    CompensatingWeakness,

    /// Patterns were explicitly marked as related
    ExplicitRelation,

    /// Causal discovery found a dependency
    CausalDependency,
}

impl std::fmt::Display for SynergyReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CooccurrenceSuccess => write!(f, "frequently succeed together"),
            Self::DomainAffinity => write!(f, "related domains"),
            Self::Complementary => write!(f, "complementary characteristics"),
            Self::CompensatingWeakness => write!(f, "compensating weaknesses"),
            Self::ExplicitRelation => write!(f, "explicitly related"),
            Self::CausalDependency => write!(f, "causal dependency"),
        }
    }
}

/// Tracker for pattern co-occurrence to discover synergies
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CooccurrenceTracker {
    /// Tracks (pattern_a, pattern_b) -> (times_together, successes_together)
    cooccurrences: HashMap<(PatternId, PatternId), (u32, u32)>,

    /// Minimum co-occurrences before considering for synergy
    pub min_cooccurrences: u32,

    /// Minimum success rate to consider patterns synergistic
    pub min_success_rate: f32,
}

impl CooccurrenceTracker {
    /// Create a new tracker with default settings
    pub fn new() -> Self {
        Self {
            cooccurrences: HashMap::new(),
            min_cooccurrences: 5,
            min_success_rate: 0.6,
        }
    }

    /// Record that patterns were used together
    pub fn record_cooccurrence(&mut self, patterns: &[PatternId], success: bool) {
        // Record all pairs
        for i in 0..patterns.len() {
            for j in (i + 1)..patterns.len() {
                let key = if patterns[i] < patterns[j] {
                    (patterns[i], patterns[j])
                } else {
                    (patterns[j], patterns[i])
                };

                let entry = self.cooccurrences.entry(key).or_insert((0, 0));
                entry.0 += 1;
                if success {
                    entry.1 += 1;
                }
            }
        }
    }

    /// Get co-occurrence stats for a pair of patterns
    pub fn get_stats(&self, a: PatternId, b: PatternId) -> Option<(u32, u32, f32)> {
        let key = if a < b { (a, b) } else { (b, a) };
        self.cooccurrences.get(&key).map(|&(count, successes)| {
            let rate = if count > 0 {
                successes as f32 / count as f32
            } else {
                0.0
            };
            (count, successes, rate)
        })
    }

    /// Find pattern pairs that frequently succeed together
    pub fn find_synergy_candidates(&self) -> Vec<(PatternId, PatternId, f32)> {
        self.cooccurrences
            .iter()
            .filter_map(|(&(a, b), &(count, successes))| {
                if count >= self.min_cooccurrences {
                    let rate = successes as f32 / count as f32;
                    if rate >= self.min_success_rate {
                        Some((a, b, rate))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get total tracked pairs
    pub fn pair_count(&self) -> usize {
        self.cooccurrences.len()
    }
}

/// Statistics about the composition system
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CompositionStats {
    /// Total number of composites in the system
    pub total_composites: usize,

    /// Composites showing positive synergy (score > 1.1)
    pub synergistic_composites: usize,

    /// Composites showing interference (score < 0.9)
    pub interfering_composites: usize,

    /// Composites that were auto-discovered vs manually created
    pub auto_discovered_composites: usize,

    /// Number of pattern pairs being tracked for potential synergy
    pub tracked_pattern_pairs: usize,

    /// Number of candidates ready for auto-composite creation
    pub pending_candidates: usize,
}

impl CompositionStats {
    /// Check if the system has meaningful composition data
    pub fn has_data(&self) -> bool {
        self.total_composites > 0 || self.tracked_pattern_pairs > 0
    }

    /// Get the synergy rate (proportion of composites with positive synergy)
    pub fn synergy_rate(&self) -> f32 {
        if self.total_composites == 0 {
            return 0.0;
        }
        self.synergistic_composites as f32 / self.total_composites as f32
    }

    /// Get the interference rate (proportion of composites with interference)
    pub fn interference_rate(&self) -> f32 {
        if self.total_composites == 0 {
            return 0.0;
        }
        self.interfering_composites as f32 / self.total_composites as f32
    }

    /// Get the auto-discovery rate (proportion auto-discovered)
    pub fn auto_discovery_rate(&self) -> f32 {
        if self.total_composites == 0 {
            return 0.0;
        }
        self.auto_discovered_composites as f32 / self.total_composites as f32
    }
}
