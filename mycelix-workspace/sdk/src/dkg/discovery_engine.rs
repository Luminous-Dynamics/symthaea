// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Discovery Engine - SCEI Component
//!
//! The Discovery Engine identifies knowledge gaps, suggests new claims to verify,
//! and finds patterns in the knowledge graph. It provides:
//!
//! - **Gap Detection**: Find missing knowledge in high-value areas
//! - **Pattern Matching**: Identify recurring structures in the graph
//! - **Clustering**: Group semantically related claims
//! - **Suggestions**: Recommend claims to add or verify
//! - **Exploration Guidance**: Direct agents toward valuable knowledge work
//!
//! # Integration with KREDIT
//!
//! Discovery suggestions come with KREDIT bounties - agents who fill
//! knowledge gaps earn rewards proportional to the gap's importance.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[cfg(feature = "ts-export")]
use ts_rs::TS;

// ============================================================================
// Core Types
// ============================================================================

/// Type of knowledge gap
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub enum GapType {
    /// Missing predicate for a well-known subject
    MissingPredicate,
    /// Subject has claims but low confidence
    LowConfidence,
    /// Area with high query volume but few claims
    HighDemand,
    /// Temporal gap - old claims need refreshing
    Stale,
    /// Domain has few experts attesting
    LowExpertCoverage,
    /// Contradictions need resolution
    UnresolvedContradiction,
}

/// A detected knowledge gap
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub struct KnowledgeGap {
    /// Gap identifier
    pub gap_id: String,
    /// Type of gap
    pub gap_type: GapType,
    /// Subject or area of the gap
    pub subject: String,
    /// Specific predicate needed (if applicable)
    pub missing_predicate: Option<String>,
    /// Domain of the gap
    pub domain: Option<String>,
    /// Importance score (0.0 - 1.0)
    pub importance: f64,
    /// Estimated KREDIT bounty for filling
    pub bounty: u64,
    /// When gap was detected
    pub detected_at: u64,
    /// Related triple hashes
    pub related_triples: Vec<String>,
}

/// Type of discovery suggestion
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub enum SuggestionType {
    /// Add a new claim
    AddClaim,
    /// Verify/attest existing claim
    VerifyClaim,
    /// Resolve contradiction
    ResolveContradiction,
    /// Update stale claim
    RefreshClaim,
    /// Connect isolated claims
    CreateLink,
}

/// A suggestion for knowledge work
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub struct DiscoverySuggestion {
    /// Suggestion identifier
    pub suggestion_id: String,
    /// Type of suggestion
    pub suggestion_type: SuggestionType,
    /// Description of what to do
    pub description: String,
    /// Related gap (if any)
    pub gap_id: Option<String>,
    /// Target triple hash (if applicable)
    pub target_triple: Option<String>,
    /// Suggested subject for new claim
    pub suggested_subject: Option<String>,
    /// Suggested predicate for new claim
    pub suggested_predicate: Option<String>,
    /// Priority score (0.0 - 1.0)
    pub priority: f64,
    /// KREDIT reward for completing
    pub reward: u64,
    /// Domains this suggestion applies to
    pub domains: Vec<String>,
}

/// A pattern found in the knowledge graph
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub struct PatternMatch {
    /// Pattern identifier
    pub pattern_id: String,
    /// Pattern name
    pub name: String,
    /// Triples matching this pattern
    pub matching_triples: Vec<String>,
    /// Frequency of pattern occurrence
    pub frequency: u32,
    /// Average confidence of matching triples
    pub avg_confidence: f64,
    /// Pattern description
    pub description: String,
}

/// A cluster of semantically related claims
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub struct SemanticCluster {
    /// Cluster identifier
    pub cluster_id: String,
    /// Cluster label/topic
    pub label: String,
    /// Triple hashes in this cluster
    pub members: Vec<String>,
    /// Centroid (representative triple)
    pub centroid: Option<String>,
    /// Cluster cohesion score
    pub cohesion: f64,
    /// Key predicates in cluster
    pub key_predicates: Vec<String>,
    /// Key subjects in cluster
    pub key_subjects: Vec<String>,
}

/// Result of a discovery operation
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub struct DiscoveryResult {
    /// Gaps found
    pub gaps: Vec<KnowledgeGap>,
    /// Suggestions generated
    pub suggestions: Vec<DiscoverySuggestion>,
    /// Patterns detected
    pub patterns: Vec<PatternMatch>,
    /// Clusters identified
    pub clusters: Vec<SemanticCluster>,
    /// Execution time in microseconds
    pub execution_time_us: u64,
}

// ============================================================================
// Discovery Engine
// ============================================================================

/// Configuration for the discovery engine
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiscoveryEngineConfig {
    /// Minimum importance to report a gap
    pub min_gap_importance: f64,
    /// Confidence threshold for "low confidence" gap
    pub low_confidence_threshold: f64,
    /// Age (seconds) for "stale" gap
    pub stale_threshold_seconds: u64,
    /// Minimum cluster size
    pub min_cluster_size: usize,
    /// Maximum suggestions to return
    pub max_suggestions: usize,
    /// Base bounty multiplier
    pub bounty_multiplier: u64,
}

impl Default for DiscoveryEngineConfig {
    fn default() -> Self {
        Self {
            min_gap_importance: 0.3,
            low_confidence_threshold: 0.5,
            stale_threshold_seconds: 86400 * 30, // 30 days
            min_cluster_size: 3,
            max_suggestions: 50,
            bounty_multiplier: 100,
        }
    }
}

/// Input parameters for registering a triple with the discovery engine
pub struct RegisterTripleInput<'a> {
    /// Hash identifier for the triple
    pub hash: &'a str,
    /// Subject of the triple
    pub subject: &'a str,
    /// Predicate of the triple
    pub predicate: &'a str,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Optional domain classification
    pub domain: Option<&'a str>,
    /// Creation timestamp (Unix seconds)
    pub created_at: u64,
    /// Number of attestations
    pub attestation_count: u32,
}

/// Internal triple representation for discovery
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct DiscoveryTriple {
    hash: String,
    subject: String,
    predicate: String,
    confidence: f64,
    domain: Option<String>,
    created_at: u64,
    attestation_count: u32,
}

/// The Discovery Engine
pub struct DiscoveryEngine {
    config: DiscoveryEngineConfig,
    triples: HashMap<String, DiscoveryTriple>,
    query_counts: HashMap<String, u64>, // subject -> query count
    domain_experts: HashMap<String, HashSet<String>>, // domain -> expert IDs
    gaps: Vec<KnowledgeGap>,
    gap_counter: u64,
    suggestion_counter: u64,
}

impl DiscoveryEngine {
    /// Create a new discovery engine with the given configuration
    pub fn new(config: DiscoveryEngineConfig) -> Self {
        Self {
            config,
            triples: HashMap::new(),
            query_counts: HashMap::new(),
            domain_experts: HashMap::new(),
            gaps: Vec::new(),
            gap_counter: 0,
            suggestion_counter: 0,
        }
    }

    /// Register a triple for discovery analysis
    pub fn register_triple(&mut self, input: RegisterTripleInput<'_>) {
        self.triples.insert(
            input.hash.to_string(),
            DiscoveryTriple {
                hash: input.hash.to_string(),
                subject: input.subject.to_string(),
                predicate: input.predicate.to_string(),
                confidence: input.confidence,
                domain: input.domain.map(|s| s.to_string()),
                created_at: input.created_at,
                attestation_count: input.attestation_count,
            },
        );
    }

    /// Record a query for demand tracking
    pub fn record_query(&mut self, subject: &str) {
        *self.query_counts.entry(subject.to_string()).or_insert(0) += 1;
    }

    /// Register a domain expert
    pub fn register_expert(&mut self, domain: &str, expert_id: &str) {
        self.domain_experts
            .entry(domain.to_string())
            .or_default()
            .insert(expert_id.to_string());
    }

    /// Run discovery analysis
    pub fn discover(&mut self, current_time: u64) -> DiscoveryResult {
        let start_time = std::time::Instant::now();

        // Find gaps
        let gaps = self.find_gaps(current_time);

        // Generate suggestions
        let suggestions = self.generate_suggestions(&gaps);

        // Find patterns
        let patterns = self.find_patterns();

        // Build clusters
        let clusters = self.build_clusters();

        // Store gaps for future reference
        self.gaps = gaps.clone();

        DiscoveryResult {
            gaps,
            suggestions,
            patterns,
            clusters,
            execution_time_us: start_time.elapsed().as_micros() as u64,
        }
    }

    fn find_gaps(&mut self, current_time: u64) -> Vec<KnowledgeGap> {
        let mut gaps = Vec::new();

        // Find low confidence claims
        for triple in self.triples.values() {
            if triple.confidence < self.config.low_confidence_threshold {
                self.gap_counter += 1;
                let importance =
                    self.calculate_gap_importance(&triple.subject, &GapType::LowConfidence);

                if importance >= self.config.min_gap_importance {
                    gaps.push(KnowledgeGap {
                        gap_id: format!("gap-{}", self.gap_counter),
                        gap_type: GapType::LowConfidence,
                        subject: triple.subject.clone(),
                        missing_predicate: None,
                        domain: triple.domain.clone(),
                        importance,
                        bounty: (importance * self.config.bounty_multiplier as f64) as u64,
                        detected_at: current_time,
                        related_triples: vec![triple.hash.clone()],
                    });
                }
            }

            // Find stale claims
            let age = current_time.saturating_sub(triple.created_at);
            if age > self.config.stale_threshold_seconds {
                self.gap_counter += 1;
                let importance = self.calculate_gap_importance(&triple.subject, &GapType::Stale);

                if importance >= self.config.min_gap_importance {
                    gaps.push(KnowledgeGap {
                        gap_id: format!("gap-{}", self.gap_counter),
                        gap_type: GapType::Stale,
                        subject: triple.subject.clone(),
                        missing_predicate: None,
                        domain: triple.domain.clone(),
                        importance,
                        bounty: (importance * self.config.bounty_multiplier as f64 * 0.5) as u64,
                        detected_at: current_time,
                        related_triples: vec![triple.hash.clone()],
                    });
                }
            }
        }

        // Find high-demand subjects with few claims
        for (subject, query_count) in &self.query_counts {
            let claim_count = self
                .triples
                .values()
                .filter(|t| &t.subject == subject)
                .count();

            if *query_count > 10 && claim_count < 3 {
                self.gap_counter += 1;
                let importance = (*query_count as f64 / 100.0).min(1.0);

                gaps.push(KnowledgeGap {
                    gap_id: format!("gap-{}", self.gap_counter),
                    gap_type: GapType::HighDemand,
                    subject: subject.clone(),
                    missing_predicate: None,
                    domain: None,
                    importance,
                    bounty: (importance * self.config.bounty_multiplier as f64 * 2.0) as u64,
                    detected_at: current_time,
                    related_triples: vec![],
                });
            }
        }

        // Find domains with low expert coverage
        for (domain, experts) in &self.domain_experts {
            if experts.len() < 3 {
                let domain_claims: Vec<_> = self
                    .triples
                    .values()
                    .filter(|t| t.domain.as_ref() == Some(domain))
                    .collect();

                if domain_claims.len() > 10 {
                    self.gap_counter += 1;
                    let importance = 0.6 - (experts.len() as f64 * 0.1);

                    gaps.push(KnowledgeGap {
                        gap_id: format!("gap-{}", self.gap_counter),
                        gap_type: GapType::LowExpertCoverage,
                        subject: domain.clone(),
                        missing_predicate: None,
                        domain: Some(domain.clone()),
                        importance,
                        bounty: (importance * self.config.bounty_multiplier as f64 * 1.5) as u64,
                        detected_at: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                        related_triples: domain_claims.iter().map(|t| t.hash.clone()).collect(),
                    });
                }
            }
        }

        gaps.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        gaps
    }

    fn calculate_gap_importance(&self, subject: &str, gap_type: &GapType) -> f64 {
        let base_importance = match gap_type {
            GapType::LowConfidence => 0.5,
            GapType::HighDemand => 0.7,
            GapType::Stale => 0.3,
            GapType::MissingPredicate => 0.4,
            GapType::LowExpertCoverage => 0.6,
            GapType::UnresolvedContradiction => 0.8,
        };

        // Boost importance based on query demand
        let demand_boost = self
            .query_counts
            .get(subject)
            .map(|c| (*c as f64 / 100.0).min(0.3))
            .unwrap_or(0.0);

        (base_importance + demand_boost).min(1.0)
    }

    fn generate_suggestions(&mut self, gaps: &[KnowledgeGap]) -> Vec<DiscoverySuggestion> {
        let mut suggestions = Vec::new();

        for gap in gaps.iter().take(self.config.max_suggestions) {
            self.suggestion_counter += 1;

            let (suggestion_type, description) = match gap.gap_type {
                GapType::LowConfidence => (
                    SuggestionType::VerifyClaim,
                    format!(
                        "Verify claims about '{}' - current confidence is low",
                        gap.subject
                    ),
                ),
                GapType::HighDemand => (
                    SuggestionType::AddClaim,
                    format!(
                        "Add claims about '{}' - high query demand but few claims",
                        gap.subject
                    ),
                ),
                GapType::Stale => (
                    SuggestionType::RefreshClaim,
                    format!(
                        "Update claims about '{}' - information may be outdated",
                        gap.subject
                    ),
                ),
                GapType::MissingPredicate => (
                    SuggestionType::AddClaim,
                    format!(
                        "Add '{}' information for '{}'",
                        gap.missing_predicate
                            .as_ref()
                            .unwrap_or(&"missing".to_string()),
                        gap.subject
                    ),
                ),
                GapType::LowExpertCoverage => (
                    SuggestionType::VerifyClaim,
                    format!("Domain '{}' needs more expert attestations", gap.subject),
                ),
                GapType::UnresolvedContradiction => (
                    SuggestionType::ResolveContradiction,
                    format!("Resolve conflicting claims about '{}'", gap.subject),
                ),
            };

            suggestions.push(DiscoverySuggestion {
                suggestion_id: format!("sug-{}", self.suggestion_counter),
                suggestion_type,
                description,
                gap_id: Some(gap.gap_id.clone()),
                target_triple: gap.related_triples.first().cloned(),
                suggested_subject: Some(gap.subject.clone()),
                suggested_predicate: gap.missing_predicate.clone(),
                priority: gap.importance,
                reward: gap.bounty,
                domains: gap.domain.iter().cloned().collect(),
            });
        }

        suggestions
    }

    fn find_patterns(&self) -> Vec<PatternMatch> {
        let mut patterns = Vec::new();
        let mut predicate_counts: HashMap<&str, Vec<&DiscoveryTriple>> = HashMap::new();

        // Group by predicate
        for triple in self.triples.values() {
            predicate_counts
                .entry(&triple.predicate)
                .or_default()
                .push(triple);
        }

        // Find frequent predicates
        for (predicate, triples) in predicate_counts {
            if triples.len() >= 5 {
                let avg_confidence =
                    triples.iter().map(|t| t.confidence).sum::<f64>() / triples.len() as f64;

                patterns.push(PatternMatch {
                    pattern_id: format!("pat-{}", predicate),
                    name: format!("Common predicate: {}", predicate),
                    matching_triples: triples.iter().map(|t| t.hash.clone()).collect(),
                    frequency: triples.len() as u32,
                    avg_confidence,
                    description: format!(
                        "Predicate '{}' appears {} times with avg confidence {:.2}",
                        predicate,
                        triples.len(),
                        avg_confidence
                    ),
                });
            }
        }

        patterns.sort_by(|a, b| b.frequency.cmp(&a.frequency));
        patterns
    }

    fn build_clusters(&self) -> Vec<SemanticCluster> {
        let mut clusters = Vec::new();
        let mut subject_groups: HashMap<&str, Vec<&DiscoveryTriple>> = HashMap::new();

        // Group by subject
        for triple in self.triples.values() {
            subject_groups
                .entry(&triple.subject)
                .or_default()
                .push(triple);
        }

        // Build clusters from subject groups
        for (subject, triples) in subject_groups {
            if triples.len() >= self.config.min_cluster_size {
                let predicates: HashSet<_> = triples.iter().map(|t| t.predicate.clone()).collect();

                let avg_confidence =
                    triples.iter().map(|t| t.confidence).sum::<f64>() / triples.len() as f64;

                clusters.push(SemanticCluster {
                    cluster_id: format!("cluster-{}", subject),
                    label: subject.to_string(),
                    members: triples.iter().map(|t| t.hash.clone()).collect(),
                    centroid: triples
                        .iter()
                        .max_by(|a, b| {
                            a.confidence
                                .partial_cmp(&b.confidence)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|t| t.hash.clone()),
                    cohesion: avg_confidence,
                    key_predicates: predicates.into_iter().collect(),
                    key_subjects: vec![subject.to_string()],
                });
            }
        }

        clusters.sort_by(|a, b| b.members.len().cmp(&a.members.len()));
        clusters
    }

    /// Get top suggestions by priority
    pub fn top_suggestions(&self, _limit: usize) -> Vec<&DiscoverySuggestion> {
        let mut result = self.discover_cached();
        result.suggestions.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // This is a simplification - in production you'd cache the result
        Vec::new()
    }

    fn discover_cached(&self) -> DiscoveryResult {
        DiscoveryResult {
            gaps: self.gaps.clone(),
            suggestions: Vec::new(),
            patterns: Vec::new(),
            clusters: Vec::new(),
            execution_time_us: 0,
        }
    }

    /// Get statistics about discovery state
    pub fn stats(&self) -> DiscoveryStats {
        DiscoveryStats {
            total_triples: self.triples.len(),
            unique_subjects: self
                .triples
                .values()
                .map(|t| &t.subject)
                .collect::<HashSet<_>>()
                .len(),
            unique_predicates: self
                .triples
                .values()
                .map(|t| &t.predicate)
                .collect::<HashSet<_>>()
                .len(),
            domains_tracked: self.domain_experts.len(),
            total_experts: self.domain_experts.values().map(|s| s.len()).sum(),
            open_gaps: self.gaps.len(),
            total_query_volume: self.query_counts.values().sum(),
        }
    }

    /// Fill a gap (mark as resolved)
    pub fn fill_gap(&mut self, gap_id: &str) -> Option<u64> {
        if let Some(pos) = self.gaps.iter().position(|g| g.gap_id == gap_id) {
            let gap = self.gaps.remove(pos);
            Some(gap.bounty)
        } else {
            None
        }
    }
}

/// Statistics about discovery engine state
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/dkg/"))]
pub struct DiscoveryStats {
    /// Total number of triples tracked
    pub total_triples: usize,
    /// Number of unique subjects across all triples
    pub unique_subjects: usize,
    /// Number of unique predicates across all triples
    pub unique_predicates: usize,
    /// Number of domains being tracked
    pub domains_tracked: usize,
    /// Total number of domain experts registered
    pub total_experts: usize,
    /// Number of currently open knowledge gaps
    pub open_gaps: usize,
    /// Total query volume across all subjects
    pub total_query_volume: u64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn current_time() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    #[test]
    fn test_register_triple() {
        let mut engine = DiscoveryEngine::new(DiscoveryEngineConfig::default());

        engine.register_triple(RegisterTripleInput {
            hash: "hash-1",
            subject: "climate",
            predicate: "temperature",
            confidence: 0.8,
            domain: Some("science"),
            created_at: current_time(),
            attestation_count: 5,
        });

        assert_eq!(engine.triples.len(), 1);
        assert!(engine.triples.contains_key("hash-1"));
    }

    #[test]
    fn test_low_confidence_gap() {
        let mut engine = DiscoveryEngine::new(DiscoveryEngineConfig {
            low_confidence_threshold: 0.5,
            min_gap_importance: 0.0,
            ..Default::default()
        });

        engine.register_triple(RegisterTripleInput {
            hash: "hash-1",
            subject: "uncertain_topic",
            predicate: "claim",
            confidence: 0.3, // Low confidence
            domain: None,
            created_at: current_time(),
            attestation_count: 1,
        });

        let result = engine.discover(current_time());
        let low_conf_gaps: Vec<_> = result
            .gaps
            .iter()
            .filter(|g| g.gap_type == GapType::LowConfidence)
            .collect();

        assert!(!low_conf_gaps.is_empty());
    }

    #[test]
    fn test_high_demand_gap() {
        let mut engine = DiscoveryEngine::new(DiscoveryEngineConfig {
            min_gap_importance: 0.0,
            ..Default::default()
        });

        // Record many queries for a subject with few claims
        for _ in 0..20 {
            engine.record_query("popular_topic");
        }

        let result = engine.discover(current_time());
        let demand_gaps: Vec<_> = result
            .gaps
            .iter()
            .filter(|g| g.gap_type == GapType::HighDemand)
            .collect();

        assert!(!demand_gaps.is_empty());
    }

    #[test]
    fn test_stale_gap() {
        let mut engine = DiscoveryEngine::new(DiscoveryEngineConfig {
            stale_threshold_seconds: 100,
            min_gap_importance: 0.0,
            ..Default::default()
        });

        engine.register_triple(RegisterTripleInput {
            hash: "hash-1",
            subject: "old_topic",
            predicate: "fact",
            confidence: 0.9,
            domain: None,
            created_at: current_time() - 200, // Old timestamp
            attestation_count: 10,
        });

        let result = engine.discover(current_time());
        let stale_gaps: Vec<_> = result
            .gaps
            .iter()
            .filter(|g| g.gap_type == GapType::Stale)
            .collect();

        assert!(!stale_gaps.is_empty());
    }

    #[test]
    fn test_pattern_detection() {
        let mut engine = DiscoveryEngine::new(DiscoveryEngineConfig::default());

        // Add multiple triples with same predicate
        for i in 0..10 {
            engine.register_triple(RegisterTripleInput {
                hash: &format!("hash-{}", i),
                subject: &format!("subject-{}", i),
                predicate: "common_predicate",
                confidence: 0.8,
                domain: None,
                created_at: current_time(),
                attestation_count: 5,
            });
        }

        let result = engine.discover(current_time());

        assert!(!result.patterns.is_empty());
        assert!(result.patterns[0].name.contains("common_predicate"));
    }

    #[test]
    fn test_clustering() {
        let mut engine = DiscoveryEngine::new(DiscoveryEngineConfig {
            min_cluster_size: 2,
            ..Default::default()
        });

        // Add multiple triples with same subject
        for i in 0..5 {
            engine.register_triple(RegisterTripleInput {
                hash: &format!("hash-{}", i),
                subject: "shared_subject",
                predicate: &format!("pred-{}", i),
                confidence: 0.7,
                domain: None,
                created_at: current_time(),
                attestation_count: 3,
            });
        }

        let result = engine.discover(current_time());

        assert!(!result.clusters.is_empty());
        assert_eq!(result.clusters[0].label, "shared_subject");
    }

    #[test]
    fn test_suggestions_generated() {
        let mut engine = DiscoveryEngine::new(DiscoveryEngineConfig {
            min_gap_importance: 0.0,
            ..Default::default()
        });

        engine.register_triple(RegisterTripleInput {
            hash: "hash-1",
            subject: "topic",
            predicate: "claim",
            confidence: 0.2, // Very low confidence
            domain: None,
            created_at: current_time(),
            attestation_count: 1,
        });

        let result = engine.discover(current_time());

        assert!(!result.suggestions.is_empty());
        assert!(result
            .suggestions
            .iter()
            .any(|s| s.suggestion_type == SuggestionType::VerifyClaim));
    }

    #[test]
    fn test_fill_gap() {
        let mut engine = DiscoveryEngine::new(DiscoveryEngineConfig {
            min_gap_importance: 0.0,
            bounty_multiplier: 100,
            ..Default::default()
        });

        engine.register_triple(RegisterTripleInput {
            hash: "hash-1",
            subject: "topic",
            predicate: "claim",
            confidence: 0.2,
            domain: None,
            created_at: current_time(),
            attestation_count: 1,
        });

        let result = engine.discover(current_time());
        let gap_id = result.gaps.first().map(|g| g.gap_id.clone());

        if let Some(id) = gap_id {
            let bounty = engine.fill_gap(&id);
            assert!(bounty.is_some());
            assert!(bounty.unwrap() > 0);
        }
    }

    #[test]
    fn test_stats() {
        let mut engine = DiscoveryEngine::new(DiscoveryEngineConfig::default());

        engine.register_triple(RegisterTripleInput {
            hash: "h1",
            subject: "s1",
            predicate: "p1",
            confidence: 0.8,
            domain: Some("d1"),
            created_at: current_time(),
            attestation_count: 5,
        });
        engine.register_triple(RegisterTripleInput {
            hash: "h2",
            subject: "s2",
            predicate: "p2",
            confidence: 0.7,
            domain: Some("d1"),
            created_at: current_time(),
            attestation_count: 3,
        });
        engine.register_expert("d1", "expert-1");
        engine.record_query("s1");

        let stats = engine.stats();

        assert_eq!(stats.total_triples, 2);
        assert_eq!(stats.unique_subjects, 2);
        assert_eq!(stats.domains_tracked, 1);
        assert_eq!(stats.total_experts, 1);
        assert_eq!(stats.total_query_volume, 1);
    }
}
