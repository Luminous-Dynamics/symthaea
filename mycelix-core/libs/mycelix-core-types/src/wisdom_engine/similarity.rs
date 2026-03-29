// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// =============================================================================
// Component 21: Pattern Similarity & Clustering
// =============================================================================
//
// Provides similarity metrics, clustering, and duplicate detection for patterns.
// This module enables the WisdomEngine to:
// - Find similar/duplicate patterns to prevent knowledge fragmentation
// - Group related patterns for better organization
// - Suggest pattern merges when appropriate
// - Provide similarity scores for the HDC associative learner
//
// Follows the Config + Registry + Stats pattern established in other components.
// =============================================================================

use std::collections::{HashMap, HashSet};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::symthaea::SymthaeaPattern;
use super::PatternId;

// =============================================================================
// Similarity Metrics
// =============================================================================

/// How to measure similarity between patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SimilarityMetric {
    /// Jaccard similarity on problem domains
    DomainJaccard,

    /// Cosine similarity on solution text (bag of words)
    SolutionCosine,

    /// Overlap in outcomes/success patterns
    OutcomeOverlap,

    /// Combined metric weighting all factors
    Composite,

    /// Structural similarity (dependencies, lifecycle)
    Structural,

    /// Semantic similarity (requires embeddings)
    Semantic,
}

impl Default for SimilarityMetric {
    fn default() -> Self {
        Self::Composite
    }
}

impl SimilarityMetric {
    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::DomainJaccard => "Jaccard similarity on problem domains",
            Self::SolutionCosine => "Cosine similarity on solution text",
            Self::OutcomeOverlap => "Overlap in outcome patterns",
            Self::Composite => "Weighted combination of all metrics",
            Self::Structural => "Similarity in structure and dependencies",
            Self::Semantic => "Deep semantic similarity via embeddings",
        }
    }
}

// =============================================================================
// Similarity Score
// =============================================================================

/// Detailed similarity score between two patterns
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimilarityScore {
    /// Pattern A
    pub pattern_a: PatternId,

    /// Pattern B
    pub pattern_b: PatternId,

    /// Overall similarity (0.0 = completely different, 1.0 = identical)
    pub overall: f32,

    /// Domain similarity component
    pub domain_similarity: f32,

    /// Solution similarity component
    pub solution_similarity: f32,

    /// Outcome similarity component
    pub outcome_similarity: f32,

    /// Structural similarity component
    pub structural_similarity: f32,

    /// Metric used for overall score
    pub metric_used: SimilarityMetric,

    /// Confidence in this score (affected by data availability)
    pub confidence: f32,

    /// When this score was computed
    pub computed_at: u64,
}

impl SimilarityScore {
    /// Create a new similarity score
    pub fn new(pattern_a: PatternId, pattern_b: PatternId, timestamp: u64) -> Self {
        Self {
            pattern_a,
            pattern_b,
            overall: 0.0,
            domain_similarity: 0.0,
            solution_similarity: 0.0,
            outcome_similarity: 0.0,
            structural_similarity: 0.0,
            metric_used: SimilarityMetric::Composite,
            confidence: 0.0,
            computed_at: timestamp,
        }
    }

    /// Check if patterns are highly similar (potential duplicates)
    pub fn is_highly_similar(&self) -> bool {
        self.overall >= 0.85
    }

    /// Check if patterns are moderately similar (related)
    pub fn is_related(&self) -> bool {
        self.overall >= 0.5 && self.overall < 0.85
    }

    /// Check if patterns are dissimilar
    pub fn is_dissimilar(&self) -> bool {
        self.overall < 0.3
    }

    /// Get the strongest similarity dimension
    pub fn strongest_dimension(&self) -> (&'static str, f32) {
        let dims = [
            ("domain", self.domain_similarity),
            ("solution", self.solution_similarity),
            ("outcome", self.outcome_similarity),
            ("structural", self.structural_similarity),
        ];
        dims.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(("none", 0.0))
    }
}

// =============================================================================
// Pattern Features (for similarity computation)
// =============================================================================

/// Extracted features from a pattern for similarity computation
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatternFeatures {
    /// Pattern ID
    pub pattern_id: PatternId,

    /// Problem domain identifier
    pub domain_id: u64,

    /// Tokenized solution words (lowercased)
    pub solution_tokens: HashSet<String>,

    /// N-gram features from solution
    pub solution_ngrams: HashSet<String>,

    /// Success rate
    pub success_rate: f32,

    /// Usage count
    pub usage_count: u64,

    /// Dependencies (other patterns this depends on)
    pub dependencies: HashSet<PatternId>,

    /// Dependents (patterns that depend on this)
    pub dependents: HashSet<PatternId>,

    /// Tags/categories
    pub tags: HashSet<String>,

    /// Phi value at learning
    pub phi_value: f32,

    /// Age in time units
    pub age: u64,

    /// Optional embedding vector (for semantic similarity)
    pub embedding: Option<Vec<f32>>,
}

impl PatternFeatures {
    /// Extract features from a SymthaeaPattern
    pub fn from_pattern(pattern: &SymthaeaPattern, timestamp: u64) -> Self {
        // Tokenize solution
        let solution_tokens: HashSet<String> = pattern
            .solution
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .map(|w| w.chars().filter(|c| c.is_alphanumeric()).collect())
            .filter(|w: &String| !w.is_empty())
            .collect();

        // Generate bigrams
        let words: Vec<&str> = pattern.solution.split_whitespace().collect();
        let solution_ngrams: HashSet<String> = words
            .windows(2)
            .map(|w| format!("{}_{}", w[0].to_lowercase(), w[1].to_lowercase()))
            .collect();

        // Use first structured domain_id if available, otherwise hash problem_domain string
        let domain_id = pattern.domain_ids.first().copied().unwrap_or_else(|| {
            // Simple hash of the problem_domain string
            pattern
                .problem_domain
                .bytes()
                .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64))
        });

        Self {
            pattern_id: pattern.pattern_id,
            domain_id,
            solution_tokens,
            solution_ngrams,
            success_rate: pattern.success_rate,
            usage_count: pattern.usage_count,
            dependencies: HashSet::new(),
            dependents: HashSet::new(),
            tags: HashSet::new(),
            phi_value: pattern.phi_at_learning,
            age: timestamp.saturating_sub(pattern.created_at),
            embedding: None,
        }
    }

    /// Add dependency information
    pub fn with_dependencies(mut self, deps: HashSet<PatternId>) -> Self {
        self.dependencies = deps;
        self
    }

    /// Add dependent information
    pub fn with_dependents(mut self, deps: HashSet<PatternId>) -> Self {
        self.dependents = deps;
        self
    }

    /// Add tags
    pub fn with_tags(mut self, tags: HashSet<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Add embedding
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

// =============================================================================
// Similarity Calculator
// =============================================================================

/// Calculates similarity between pattern features
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimilarityCalculator {
    /// Weight for domain similarity
    pub domain_weight: f32,

    /// Weight for solution similarity
    pub solution_weight: f32,

    /// Weight for outcome similarity
    pub outcome_weight: f32,

    /// Weight for structural similarity
    pub structural_weight: f32,
}

impl Default for SimilarityCalculator {
    fn default() -> Self {
        Self {
            domain_weight: 0.25,
            solution_weight: 0.35,
            outcome_weight: 0.20,
            structural_weight: 0.20,
        }
    }
}

impl SimilarityCalculator {
    /// Calculate similarity between two feature sets
    pub fn calculate(
        &self,
        features_a: &PatternFeatures,
        features_b: &PatternFeatures,
        timestamp: u64,
    ) -> SimilarityScore {
        let mut score =
            SimilarityScore::new(features_a.pattern_id, features_b.pattern_id, timestamp);

        // Domain similarity (exact match or not for now)
        score.domain_similarity = if features_a.domain_id == features_b.domain_id {
            1.0
        } else {
            0.0
        };

        // Solution similarity (Jaccard on tokens + ngrams)
        score.solution_similarity =
            self.jaccard_similarity(&features_a.solution_tokens, &features_b.solution_tokens) * 0.6
                + self.jaccard_similarity(&features_a.solution_ngrams, &features_b.solution_ngrams)
                    * 0.4;

        // Outcome similarity (success rate closeness)
        let rate_diff = (features_a.success_rate - features_b.success_rate).abs();
        score.outcome_similarity = 1.0 - rate_diff.min(1.0);

        // Structural similarity (shared dependencies/dependents)
        let dep_sim = self.jaccard_similarity(&features_a.dependencies, &features_b.dependencies);
        let dependent_sim = self.jaccard_similarity(&features_a.dependents, &features_b.dependents);
        score.structural_similarity = (dep_sim + dependent_sim) / 2.0;

        // Compute overall score
        score.overall = score.domain_similarity * self.domain_weight
            + score.solution_similarity * self.solution_weight
            + score.outcome_similarity * self.outcome_weight
            + score.structural_similarity * self.structural_weight;

        score.metric_used = SimilarityMetric::Composite;

        // Confidence based on data availability
        let mut confidence_factors = 0;
        let mut confidence_sum = 0.0;

        if !features_a.solution_tokens.is_empty() && !features_b.solution_tokens.is_empty() {
            confidence_factors += 1;
            confidence_sum += 1.0;
        }
        if features_a.usage_count > 10 && features_b.usage_count > 10 {
            confidence_factors += 1;
            confidence_sum += 1.0;
        }
        if !features_a.dependencies.is_empty() || !features_b.dependencies.is_empty() {
            confidence_factors += 1;
            confidence_sum += 0.8;
        }

        score.confidence = if confidence_factors > 0 {
            confidence_sum / confidence_factors as f32
        } else {
            0.3 // Low confidence if no data
        };

        score
    }

    /// Jaccard similarity for sets
    fn jaccard_similarity<T: Eq + std::hash::Hash>(&self, a: &HashSet<T>, b: &HashSet<T>) -> f32 {
        if a.is_empty() && b.is_empty() {
            return 1.0; // Both empty = identical
        }
        if a.is_empty() || b.is_empty() {
            return 0.0; // One empty = no similarity
        }

        let intersection = a.intersection(b).count();
        let union = a.union(b).count();

        intersection as f32 / union as f32
    }

    /// Cosine similarity for vectors
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }
}

// =============================================================================
// Pattern Cluster
// =============================================================================

/// A cluster of related patterns
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PatternCluster {
    /// Unique cluster identifier
    pub cluster_id: u64,

    /// Name/label for this cluster
    pub name: String,

    /// Description of what unifies these patterns
    pub description: String,

    /// Member pattern IDs
    pub members: HashSet<PatternId>,

    /// Centroid pattern (most representative)
    pub centroid_id: Option<PatternId>,

    /// Average internal similarity
    pub cohesion: f32,

    /// Average similarity to nearest other cluster
    pub separation: f32,

    /// When this cluster was formed
    pub formed_at: u64,

    /// Last time cluster was updated
    pub updated_at: u64,

    /// Dominant domain in this cluster
    pub dominant_domain: Option<u64>,

    /// Common tags across members
    pub common_tags: HashSet<String>,
}

impl PatternCluster {
    /// Create a new empty cluster
    pub fn new(cluster_id: u64, name: String, timestamp: u64) -> Self {
        Self {
            cluster_id,
            name,
            description: String::new(),
            members: HashSet::new(),
            centroid_id: None,
            cohesion: 0.0,
            separation: 0.0,
            formed_at: timestamp,
            updated_at: timestamp,
            dominant_domain: None,
            common_tags: HashSet::new(),
        }
    }

    /// Add a pattern to this cluster
    pub fn add_member(&mut self, pattern_id: PatternId, timestamp: u64) {
        self.members.insert(pattern_id);
        self.updated_at = timestamp;
    }

    /// Remove a pattern from this cluster
    pub fn remove_member(&mut self, pattern_id: PatternId, timestamp: u64) -> bool {
        let removed = self.members.remove(&pattern_id);
        if removed {
            self.updated_at = timestamp;
            if self.centroid_id == Some(pattern_id) {
                self.centroid_id = None;
            }
        }
        removed
    }

    /// Get cluster size
    pub fn size(&self) -> usize {
        self.members.len()
    }

    /// Check if cluster is empty
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Check if pattern is in this cluster
    pub fn contains(&self, pattern_id: PatternId) -> bool {
        self.members.contains(&pattern_id)
    }

    /// Get silhouette score (cluster quality metric)
    pub fn silhouette_score(&self) -> f32 {
        if self.cohesion + self.separation == 0.0 {
            return 0.0;
        }
        (self.separation - self.cohesion) / self.cohesion.max(self.separation)
    }
}

// =============================================================================
// Duplicate Candidate
// =============================================================================

/// A potential duplicate pattern pair
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DuplicateCandidate {
    /// Pattern A (typically older)
    pub pattern_a: PatternId,

    /// Pattern B (typically newer)
    pub pattern_b: PatternId,

    /// Similarity score
    pub similarity: f32,

    /// Confidence this is truly a duplicate
    pub confidence: f32,

    /// Reasons for flagging as duplicate
    pub reasons: Vec<String>,

    /// Suggested action
    pub suggestion: DuplicateSuggestion,

    /// When detected
    pub detected_at: u64,

    /// Whether this has been reviewed
    pub reviewed: bool,

    /// Review decision (if reviewed)
    pub decision: Option<DuplicateDecision>,
}

/// What to do with a duplicate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DuplicateSuggestion {
    /// Merge into single pattern
    Merge,
    /// Keep both but link them
    Link,
    /// Archive the newer one
    ArchiveNewer,
    /// Archive the older one
    ArchiveOlder,
    /// Needs manual review
    ManualReview,
}

/// Decision made about a duplicate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DuplicateDecision {
    /// Confirmed duplicate, merged
    Merged,
    /// Confirmed duplicate, one archived
    Archived,
    /// Not actually duplicates
    NotDuplicate,
    /// Linked as related
    Linked,
    /// Deferred for later
    Deferred,
}

impl DuplicateCandidate {
    /// Create a new duplicate candidate
    pub fn new(
        pattern_a: PatternId,
        pattern_b: PatternId,
        similarity: f32,
        timestamp: u64,
    ) -> Self {
        let suggestion = if similarity >= 0.95 {
            DuplicateSuggestion::Merge
        } else if similarity >= 0.85 {
            DuplicateSuggestion::Link
        } else {
            DuplicateSuggestion::ManualReview
        };

        Self {
            pattern_a,
            pattern_b,
            similarity,
            confidence: similarity, // Initial confidence = similarity
            reasons: Vec::new(),
            suggestion,
            detected_at: timestamp,
            reviewed: false,
            decision: None,
        }
    }

    /// Add a reason for duplicate detection
    pub fn add_reason(&mut self, reason: String) {
        self.reasons.push(reason);
    }

    /// Mark as reviewed with a decision
    pub fn review(&mut self, decision: DuplicateDecision) {
        self.reviewed = true;
        self.decision = Some(decision);
    }
}

// =============================================================================
// Merge Suggestion
// =============================================================================

/// Suggestion to merge similar patterns
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MergeSuggestion {
    /// Patterns to merge
    pub patterns: Vec<PatternId>,

    /// Suggested merged solution
    pub merged_solution: String,

    /// Suggested merged domain
    pub merged_domain: u64,

    /// Expected improvement from merge
    pub expected_improvement: f32,

    /// Confidence in this suggestion
    pub confidence: f32,

    /// Rationale for the merge
    pub rationale: String,

    /// When suggested
    pub suggested_at: u64,

    /// Status of the suggestion
    pub status: MergeSuggestionStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MergeSuggestionStatus {
    /// Pending review
    Pending,
    /// Approved
    Approved,
    /// Rejected
    Rejected,
    /// Executed
    Executed,
}

impl MergeSuggestion {
    /// Create a new merge suggestion
    pub fn new(patterns: Vec<PatternId>, timestamp: u64) -> Self {
        Self {
            patterns,
            merged_solution: String::new(),
            merged_domain: 0,
            expected_improvement: 0.0,
            confidence: 0.0,
            rationale: String::new(),
            suggested_at: timestamp,
            status: MergeSuggestionStatus::Pending,
        }
    }
}

// =============================================================================
// Similarity Configuration
// =============================================================================

/// Configuration for the similarity registry
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimilarityConfig {
    /// Threshold for considering patterns as duplicates
    pub duplicate_threshold: f32,

    /// Threshold for considering patterns as related
    pub related_threshold: f32,

    /// Minimum patterns to form a cluster
    pub min_cluster_size: usize,

    /// Maximum number of clusters
    pub max_clusters: usize,

    /// Whether to auto-detect duplicates
    pub auto_detect_duplicates: bool,

    /// Whether to auto-cluster patterns
    pub auto_cluster: bool,

    /// Cache TTL for similarity scores (in time units)
    pub cache_ttl: u64,

    /// Weight configuration for composite similarity
    pub weights: SimilarityWeights,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimilarityWeights {
    pub domain: f32,
    pub solution: f32,
    pub outcome: f32,
    pub structural: f32,
}

impl Default for SimilarityWeights {
    fn default() -> Self {
        Self {
            domain: 0.25,
            solution: 0.35,
            outcome: 0.20,
            structural: 0.20,
        }
    }
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        Self {
            duplicate_threshold: 0.85,
            related_threshold: 0.5,
            min_cluster_size: 2,
            max_clusters: 100,
            auto_detect_duplicates: true,
            auto_cluster: true,
            cache_ttl: 3600, // 1 hour
            weights: SimilarityWeights::default(),
        }
    }
}

// =============================================================================
// Similarity Registry
// =============================================================================

/// Main registry for pattern similarity operations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimilarityRegistry {
    /// Configuration
    config: SimilarityConfig,

    /// Cached similarity scores (pattern_a, pattern_b) -> score
    /// Uses (min, max) ordering to avoid duplicates
    similarity_cache: HashMap<(PatternId, PatternId), SimilarityScore>,

    /// Pattern features cache
    feature_cache: HashMap<PatternId, PatternFeatures>,

    /// Detected duplicate candidates
    duplicates: Vec<DuplicateCandidate>,

    /// Pattern clusters
    clusters: HashMap<u64, PatternCluster>,

    /// Pattern -> Cluster mapping
    pattern_to_cluster: HashMap<PatternId, u64>,

    /// Merge suggestions
    merge_suggestions: Vec<MergeSuggestion>,

    /// Next cluster ID
    next_cluster_id: u64,

    /// Calculator instance
    calculator: SimilarityCalculator,

    /// Statistics
    stats: SimilarityStats,
}

impl SimilarityRegistry {
    /// Create a new similarity registry
    pub fn new() -> Self {
        Self::with_config(SimilarityConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: SimilarityConfig) -> Self {
        let calculator = SimilarityCalculator {
            domain_weight: config.weights.domain,
            solution_weight: config.weights.solution,
            outcome_weight: config.weights.outcome,
            structural_weight: config.weights.structural,
        };

        Self {
            config,
            similarity_cache: HashMap::new(),
            feature_cache: HashMap::new(),
            duplicates: Vec::new(),
            clusters: HashMap::new(),
            pattern_to_cluster: HashMap::new(),
            merge_suggestions: Vec::new(),
            next_cluster_id: 1,
            calculator,
            stats: SimilarityStats::default(),
        }
    }

    /// Register pattern features
    pub fn register_features(&mut self, features: PatternFeatures) {
        self.feature_cache.insert(features.pattern_id, features);
        self.stats.patterns_tracked += 1;
    }

    /// Register features from a pattern
    pub fn register_pattern(&mut self, pattern: &SymthaeaPattern, timestamp: u64) {
        let features = PatternFeatures::from_pattern(pattern, timestamp);
        self.register_features(features);
    }

    /// Get similarity between two patterns
    pub fn get_similarity(
        &mut self,
        pattern_a: PatternId,
        pattern_b: PatternId,
        timestamp: u64,
    ) -> Option<SimilarityScore> {
        // Normalize order
        let (a, b) = if pattern_a <= pattern_b {
            (pattern_a, pattern_b)
        } else {
            (pattern_b, pattern_a)
        };

        // Check cache
        if let Some(cached) = self.similarity_cache.get(&(a, b)) {
            if timestamp - cached.computed_at < self.config.cache_ttl {
                self.stats.cache_hits += 1;
                return Some(cached.clone());
            }
        }

        self.stats.cache_misses += 1;

        // Calculate similarity
        let features_a = self.feature_cache.get(&a)?;
        let features_b = self.feature_cache.get(&b)?;

        let score = self.calculator.calculate(features_a, features_b, timestamp);
        self.stats.similarities_computed += 1;

        // Cache the result
        self.similarity_cache.insert((a, b), score.clone());

        Some(score)
    }

    /// Find patterns similar to a given pattern
    pub fn find_similar(
        &mut self,
        pattern_id: PatternId,
        min_similarity: f32,
        timestamp: u64,
    ) -> Vec<SimilarityScore> {
        let pattern_ids: Vec<PatternId> = self.feature_cache.keys().copied().collect();

        let mut results = Vec::new();
        for other_id in pattern_ids {
            if other_id == pattern_id {
                continue;
            }

            if let Some(score) = self.get_similarity(pattern_id, other_id, timestamp) {
                if score.overall >= min_similarity {
                    results.push(score);
                }
            }
        }

        // Sort by similarity descending
        results.sort_by(|a, b| {
            b.overall
                .partial_cmp(&a.overall)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// Detect potential duplicates
    pub fn detect_duplicates(&mut self, timestamp: u64) -> Vec<DuplicateCandidate> {
        let threshold = self.config.duplicate_threshold;
        let pattern_ids: Vec<PatternId> = self.feature_cache.keys().copied().collect();

        let mut new_duplicates = Vec::new();

        for i in 0..pattern_ids.len() {
            for j in (i + 1)..pattern_ids.len() {
                let a = pattern_ids[i];
                let b = pattern_ids[j];

                if let Some(score) = self.get_similarity(a, b, timestamp) {
                    if score.overall >= threshold {
                        let mut candidate = DuplicateCandidate::new(a, b, score.overall, timestamp);

                        // Add reasons
                        if score.solution_similarity >= 0.9 {
                            candidate.add_reason("Nearly identical solution text".to_string());
                        }
                        if score.domain_similarity == 1.0 {
                            candidate.add_reason("Same problem domain".to_string());
                        }
                        if score.outcome_similarity >= 0.95 {
                            candidate.add_reason("Very similar success rates".to_string());
                        }

                        new_duplicates.push(candidate);
                    }
                }
            }
        }

        self.stats.duplicates_detected += new_duplicates.len() as u64;
        self.duplicates.extend(new_duplicates.clone());

        new_duplicates
    }

    /// Get unreviewed duplicates
    pub fn pending_duplicates(&self) -> Vec<&DuplicateCandidate> {
        self.duplicates.iter().filter(|d| !d.reviewed).collect()
    }

    /// Review a duplicate candidate
    pub fn review_duplicate(
        &mut self,
        pattern_a: PatternId,
        pattern_b: PatternId,
        decision: DuplicateDecision,
    ) {
        for dup in &mut self.duplicates {
            if (dup.pattern_a == pattern_a && dup.pattern_b == pattern_b)
                || (dup.pattern_a == pattern_b && dup.pattern_b == pattern_a)
            {
                dup.review(decision);
                break;
            }
        }
    }

    /// Perform clustering on all patterns
    pub fn cluster_patterns(&mut self, timestamp: u64) -> Vec<PatternCluster> {
        // Simple agglomerative clustering
        let pattern_ids: Vec<PatternId> = self.feature_cache.keys().copied().collect();

        if pattern_ids.len() < self.config.min_cluster_size {
            return Vec::new();
        }

        // Start with each pattern in its own cluster
        let mut clusters: HashMap<PatternId, HashSet<PatternId>> = pattern_ids
            .iter()
            .map(|&id| (id, [id].into_iter().collect()))
            .collect();

        // Iteratively merge closest clusters
        loop {
            let mut best_merge: Option<(PatternId, PatternId, f32)> = None;

            let cluster_ids: Vec<PatternId> = clusters.keys().copied().collect();

            for i in 0..cluster_ids.len() {
                for j in (i + 1)..cluster_ids.len() {
                    let cluster_a = &clusters[&cluster_ids[i]];
                    let cluster_b = &clusters[&cluster_ids[j]];

                    // Average linkage
                    let mut total_sim = 0.0;
                    let mut count = 0;

                    for &a in cluster_a {
                        for &b in cluster_b {
                            if let Some(score) = self.get_similarity(a, b, timestamp) {
                                total_sim += score.overall;
                                count += 1;
                            }
                        }
                    }

                    if count > 0 {
                        let avg_sim = total_sim / count as f32;
                        if avg_sim >= self.config.related_threshold {
                            if best_merge.as_ref().map_or(true, |bm| avg_sim > bm.2) {
                                best_merge = Some((cluster_ids[i], cluster_ids[j], avg_sim));
                            }
                        }
                    }
                }
            }

            // Merge best pair or stop
            if let Some((a, b, _sim)) = best_merge {
                if clusters.len() <= self.config.max_clusters {
                    break;
                }

                let cluster_b = clusters.remove(&b)
                    .expect("cluster b exists since it was selected from cluster_ids");
                clusters.get_mut(&a)
                    .expect("cluster a exists since it was selected from cluster_ids")
                    .extend(cluster_b);
            } else {
                break;
            }
        }

        // Convert to PatternCluster objects
        let mut result = Vec::new();
        for (centroid, members) in clusters {
            if members.len() >= self.config.min_cluster_size {
                let mut cluster = PatternCluster::new(
                    self.next_cluster_id,
                    format!("Cluster {}", self.next_cluster_id),
                    timestamp,
                );
                self.next_cluster_id += 1;

                for member in &members {
                    cluster.add_member(*member, timestamp);
                    self.pattern_to_cluster.insert(*member, cluster.cluster_id);
                }
                cluster.centroid_id = Some(centroid);

                // Calculate cohesion
                let mut total_sim = 0.0;
                let mut count = 0;
                for &a in &members {
                    for &b in &members {
                        if a < b {
                            if let Some(score) = self.get_similarity(a, b, timestamp) {
                                total_sim += score.overall;
                                count += 1;
                            }
                        }
                    }
                }
                cluster.cohesion = if count > 0 {
                    total_sim / count as f32
                } else {
                    1.0
                };

                self.clusters.insert(cluster.cluster_id, cluster.clone());
                result.push(cluster);
            }
        }

        self.stats.clusters_formed = result.len() as u64;
        result
    }

    /// Get cluster for a pattern
    pub fn get_cluster(&self, pattern_id: PatternId) -> Option<&PatternCluster> {
        let cluster_id = self.pattern_to_cluster.get(&pattern_id)?;
        self.clusters.get(cluster_id)
    }

    /// Get all clusters
    pub fn all_clusters(&self) -> Vec<&PatternCluster> {
        self.clusters.values().collect()
    }

    /// Suggest merges based on similarity
    pub fn suggest_merges(&mut self, timestamp: u64) -> Vec<MergeSuggestion> {
        let mut suggestions = Vec::new();

        // Look at highly similar patterns within same domain
        for dup in &self.duplicates {
            if dup.reviewed {
                continue;
            }

            if dup.similarity >= 0.9 {
                let mut suggestion =
                    MergeSuggestion::new(vec![dup.pattern_a, dup.pattern_b], timestamp);
                suggestion.confidence = dup.confidence;
                suggestion.rationale = format!(
                    "Patterns have {:.0}% similarity: {}",
                    dup.similarity * 100.0,
                    dup.reasons.join(", ")
                );
                suggestions.push(suggestion);
            }
        }

        self.merge_suggestions.extend(suggestions.clone());
        suggestions
    }

    /// Get statistics
    pub fn stats(&self) -> &SimilarityStats {
        &self.stats
    }

    /// Clear the similarity cache
    pub fn clear_cache(&mut self) {
        self.similarity_cache.clear();
    }

    /// Get patterns that are candidates for clustering with a given pattern
    pub fn cluster_candidates(&mut self, pattern_id: PatternId, timestamp: u64) -> Vec<PatternId> {
        self.find_similar(pattern_id, self.config.related_threshold, timestamp)
            .into_iter()
            .map(|s| {
                if s.pattern_a == pattern_id {
                    s.pattern_b
                } else {
                    s.pattern_a
                }
            })
            .collect()
    }
}

impl Default for SimilarityRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Statistics
// =============================================================================

/// Statistics for the similarity registry
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimilarityStats {
    /// Total patterns tracked
    pub patterns_tracked: u64,

    /// Total similarities computed
    pub similarities_computed: u64,

    /// Cache hits
    pub cache_hits: u64,

    /// Cache misses
    pub cache_misses: u64,

    /// Duplicates detected
    pub duplicates_detected: u64,

    /// Clusters formed
    pub clusters_formed: u64,

    /// Merges suggested
    pub merges_suggested: u64,

    /// Merges executed
    pub merges_executed: u64,
}

impl SimilarityStats {
    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f32 / total as f32
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaccard_similarity() {
        let calc = SimilarityCalculator::default();

        let a: HashSet<String> = ["foo", "bar", "baz"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let b: HashSet<String> = ["foo", "bar", "qux"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let sim = calc.jaccard_similarity(&a, &b);
        assert!((sim - 0.5).abs() < 0.01); // 2 common / 4 total = 0.5
    }

    #[test]
    fn test_cosine_similarity() {
        let calc = SimilarityCalculator::default();

        let a = vec![1.0, 0.0, 1.0];
        let b = vec![1.0, 0.0, 1.0];
        assert!((calc.cosine_similarity(&a, &b) - 1.0).abs() < 0.01);

        let c = vec![0.0, 1.0, 0.0];
        assert!((calc.cosine_similarity(&a, &c) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_similarity_score_classification() {
        let mut score = SimilarityScore::new(1, 2, 1000);

        score.overall = 0.9;
        assert!(score.is_highly_similar());
        assert!(!score.is_related());
        assert!(!score.is_dissimilar());

        score.overall = 0.6;
        assert!(!score.is_highly_similar());
        assert!(score.is_related());
        assert!(!score.is_dissimilar());

        score.overall = 0.2;
        assert!(!score.is_highly_similar());
        assert!(!score.is_related());
        assert!(score.is_dissimilar());
    }

    #[test]
    fn test_pattern_cluster() {
        let mut cluster = PatternCluster::new(1, "Test Cluster".to_string(), 1000);

        assert!(cluster.is_empty());
        assert_eq!(cluster.size(), 0);

        cluster.add_member(1, 1001);
        cluster.add_member(2, 1002);

        assert!(!cluster.is_empty());
        assert_eq!(cluster.size(), 2);
        assert!(cluster.contains(1));
        assert!(cluster.contains(2));
        assert!(!cluster.contains(3));

        cluster.remove_member(1, 1003);
        assert_eq!(cluster.size(), 1);
        assert!(!cluster.contains(1));
    }

    #[test]
    fn test_duplicate_candidate() {
        let mut dup = DuplicateCandidate::new(1, 2, 0.95, 1000);

        assert!(!dup.reviewed);
        assert!(dup.decision.is_none());
        assert_eq!(dup.suggestion, DuplicateSuggestion::Merge);

        dup.add_reason("Test reason".to_string());
        assert_eq!(dup.reasons.len(), 1);

        dup.review(DuplicateDecision::Merged);
        assert!(dup.reviewed);
        assert_eq!(dup.decision, Some(DuplicateDecision::Merged));
    }

    #[test]
    fn test_similarity_registry_basic() {
        let mut registry = SimilarityRegistry::new();

        let features1 = PatternFeatures {
            pattern_id: 1,
            domain_id: 100,
            solution_tokens: ["use", "cache", "for", "speed"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            ..Default::default()
        };

        let features2 = PatternFeatures {
            pattern_id: 2,
            domain_id: 100,
            solution_tokens: ["use", "cache", "to", "improve", "speed"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            ..Default::default()
        };

        registry.register_features(features1);
        registry.register_features(features2);

        let score = registry.get_similarity(1, 2, 1000);
        assert!(score.is_some());

        let score = score.unwrap();
        assert!(score.overall > 0.5); // Should be fairly similar
        assert_eq!(score.domain_similarity, 1.0); // Same domain
    }
}
