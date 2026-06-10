// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Curiosity Engine - Active Exploration Module
//!
//! Extends the GIS Curiosity Engine with active exploration strategies
//! for resolving knowledge gaps through various knowledge sources.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                      Active Curiosity Engine                        │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │  ┌────────────┐    ┌────────────┐    ┌────────────┐                │
//! │  │ Exploration│───▶│ Resolution │───▶│ Integration│                │
//! │  │ Strategies │    │  Dispatch  │    │  & Learn   │                │
//! │  └────────────┘    └────────────┘    └────────────┘                │
//! │         │                │                  │                       │
//! │         ▼                ▼                  ▼                       │
//! │  ┌────────────┐    ┌────────────┐    ┌────────────┐                │
//! │  │ - Socratic │    │ - Memory   │    │ - Success  │                │
//! │  │ - Semantic │    │ - Web      │    │   Tracking │                │
//! │  │ - Analogic │    │ - Peer     │    │ - Strategy │                │
//! │  │ - Harmonic │    │ - Expert   │    │   Learning │                │
//! │  └────────────┘    └────────────┘    └────────────┘                │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Exploration Strategies
//!
//! 1. **Socratic** - Decompose complex questions into simpler ones
//! 2. **Semantic** - Find related concepts via HDC similarity
//! 3. **Analogical** - Map to known domains via structure preservation
//! 4. **Harmonic** - Explore based on which harmonies are affected
//!
//! ## Resolution Sources
//!
//! 1. **Memory** - Query internal databases (Qdrant, CozoDB, etc.)
//! 2. **Web** - Fetch from external web sources
//! 3. **Peer** - Ask other agents in the swarm
//! 4. **Expert** - Escalate to human experts

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

#[allow(unused_imports)] // Uncertainty3D used in tests
use super::{Domain, HarmonicIgnorance, Harmony, IgnoranceDetection, IgnoranceType, Uncertainty3D};

/// Exploration strategy for resolving ignorance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExplorationStrategy {
    /// Decompose complex questions into simpler sub-questions
    Socratic,
    /// Find semantically related concepts via HDC similarity
    Semantic,
    /// Map unknown to known domains via structural analogy
    Analogical,
    /// Explore based on affected harmonies
    Harmonic,
    /// Direct lookup in knowledge base
    Direct,
    /// Ask a clarifying question to the user
    Clarification,
}

impl ExplorationStrategy {
    /// Get recommended strategy based on ignorance type
    pub fn for_ignorance_type(ig_type: &IgnoranceType) -> Vec<Self> {
        match ig_type {
            IgnoranceType::None => vec![],
            IgnoranceType::Known => vec![Self::Direct, Self::Semantic],
            IgnoranceType::KnownUnknown => vec![Self::Semantic, Self::Analogical, Self::Socratic],
            IgnoranceType::Unknown => vec![Self::Socratic, Self::Clarification, Self::Harmonic],
            IgnoranceType::Impossible => vec![Self::Clarification, Self::Socratic],
        }
    }

    /// Base effectiveness score (0.0 - 1.0)
    pub fn base_effectiveness(&self) -> f32 {
        match self {
            Self::Direct => 0.9,
            Self::Semantic => 0.7,
            Self::Analogical => 0.6,
            Self::Socratic => 0.5,
            Self::Harmonic => 0.65,
            Self::Clarification => 0.8,
        }
    }
}

/// Knowledge source for resolution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KnowledgeSource {
    /// Internal memory systems (episodic, semantic, procedural)
    Memory,
    /// Vector database (Qdrant)
    VectorDB,
    /// Graph database (CozoDB)
    GraphDB,
    /// Analytics database (DuckDB)
    AnalyticsDB,
    /// Web search / external APIs
    Web,
    /// Other agents in the swarm
    Peer,
    /// Human expert escalation
    Expert,
    /// Internal reasoning/inference
    Inference,
}

impl KnowledgeSource {
    /// Latency estimate in milliseconds
    pub fn latency_estimate(&self) -> u64 {
        match self {
            Self::Inference => 10,
            Self::Memory => 50,
            Self::VectorDB => 100,
            Self::GraphDB => 150,
            Self::AnalyticsDB => 200,
            Self::Peer => 500,
            Self::Web => 2000,
            Self::Expert => 86400000, // 24 hours
        }
    }

    /// Reliability score (0.0 - 1.0)
    pub fn reliability(&self) -> f32 {
        match self {
            Self::Expert => 0.95,
            Self::Inference => 0.85,
            Self::Memory => 0.8,
            Self::GraphDB => 0.85,
            Self::VectorDB => 0.75,
            Self::AnalyticsDB => 0.8,
            Self::Peer => 0.7,
            Self::Web => 0.5,
        }
    }
}

/// Result of an exploration attempt
#[derive(Debug, Clone)]
pub struct ExplorationResult {
    /// Strategy used
    pub strategy: ExplorationStrategy,
    /// Source queried
    pub source: KnowledgeSource,
    /// Whether resolution was successful
    pub success: bool,
    /// Confidence in the resolution (0.0 - 1.0)
    pub confidence: f32,
    /// Resolved knowledge (if any)
    pub knowledge: Option<ResolvedKnowledge>,
    /// Sub-questions generated (for Socratic)
    pub sub_questions: Vec<String>,
    /// Related concepts found (for Semantic)
    pub related_concepts: Vec<String>,
    /// Time taken
    pub duration: Duration,
}

/// Knowledge resolved from exploration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedKnowledge {
    /// The original query
    pub query: String,
    /// The answer/resolution
    pub answer: String,
    /// Confidence level
    pub confidence: f32,
    /// Source of the knowledge
    pub source: KnowledgeSource,
    /// Supporting evidence
    pub evidence: Vec<String>,
    /// Any caveats or limitations
    pub caveats: Vec<String>,
    /// When resolved
    pub resolved_at: SystemTime,
}

/// Active exploration engine extending CuriosityEngine
pub struct ActiveExplorer {
    /// Strategy effectiveness history
    strategy_history: HashMap<ExplorationStrategy, StrategyStats>,
    /// Source effectiveness history
    source_history: HashMap<KnowledgeSource, SourceStats>,
    /// Configuration
    config: ActiveExplorerConfig,
    /// Pending explorations
    pending: Vec<PendingExploration>,
    /// Resolved items
    resolved: Vec<ExplorationResult>,
}

/// Statistics for strategy effectiveness
#[derive(Debug, Clone, Default)]
struct StrategyStats {
    attempts: u32,
    successes: u32,
    total_confidence: f32,
    avg_duration_ms: f64,
}

impl StrategyStats {
    fn success_rate(&self) -> f32 {
        if self.attempts == 0 {
            0.5 // Prior
        } else {
            self.successes as f32 / self.attempts as f32
        }
    }

    fn avg_confidence(&self) -> f32 {
        if self.successes == 0 {
            0.5
        } else {
            self.total_confidence / self.successes as f32
        }
    }

    fn record(&mut self, success: bool, confidence: f32, duration_ms: u64) {
        self.attempts += 1;
        if success {
            self.successes += 1;
            self.total_confidence += confidence;
        }
        // Running average for duration
        let alpha = 0.1;
        self.avg_duration_ms = (1.0 - alpha) * self.avg_duration_ms + alpha * duration_ms as f64;
    }
}

/// Statistics for source effectiveness
#[derive(Debug, Clone, Default)]
struct SourceStats {
    queries: u32,
    successes: u32,
    total_confidence: f32,
    avg_latency_ms: f64,
}

impl SourceStats {
    fn success_rate(&self) -> f32 {
        if self.queries == 0 {
            0.5
        } else {
            self.successes as f32 / self.queries as f32
        }
    }

    fn record(&mut self, success: bool, confidence: f32, latency_ms: u64) {
        self.queries += 1;
        if success {
            self.successes += 1;
            self.total_confidence += confidence;
        }
        let alpha = 0.1;
        self.avg_latency_ms = (1.0 - alpha) * self.avg_latency_ms + alpha * latency_ms as f64;
    }
}

/// Pending exploration
#[derive(Debug, Clone)]
#[allow(dead_code)] // RESERVED(future): exploration agenda tracking
struct PendingExploration {
    query: String,
    strategy: ExplorationStrategy,
    source: KnowledgeSource,
    priority: f32,
    started_at: SystemTime,
}

/// Configuration for active explorer
#[derive(Debug, Clone)]
pub struct ActiveExplorerConfig {
    /// Maximum concurrent explorations
    pub max_concurrent: usize,
    /// Timeout for exploration attempts
    pub timeout: Duration,
    /// Minimum confidence to accept resolution
    pub min_confidence: f32,
    /// Enable learning from results
    pub enable_learning: bool,
    /// Maximum sub-questions for Socratic
    pub max_sub_questions: usize,
}

impl Default for ActiveExplorerConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 4,
            timeout: Duration::from_secs(30),
            min_confidence: 0.5,
            enable_learning: true,
            max_sub_questions: 5,
        }
    }
}

impl ActiveExplorer {
    /// Create new active explorer
    pub fn new(config: ActiveExplorerConfig) -> Self {
        Self {
            strategy_history: HashMap::new(),
            source_history: HashMap::new(),
            config,
            pending: Vec::new(),
            resolved: Vec::new(),
        }
    }

    /// Plan exploration for a detection
    ///
    /// Returns ordered list of (strategy, source) pairs to try
    pub fn plan_exploration(
        &self,
        detection: &IgnoranceDetection,
    ) -> Vec<(ExplorationStrategy, KnowledgeSource)> {
        let mut plans = Vec::new();

        // Get recommended strategies for this ignorance type
        let strategies = ExplorationStrategy::for_ignorance_type(&detection.ignorance_type);

        for strategy in strategies {
            // Get best sources for this strategy
            let sources = self.sources_for_strategy(strategy, &detection.domain);

            for source in sources {
                let score = self.plan_score(strategy, source, detection);
                plans.push((strategy, source, score));
            }
        }

        // Sort by score descending
        plans.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Return top plans
        plans
            .into_iter()
            .take(self.config.max_concurrent * 2)
            .map(|(s, src, _)| (s, src))
            .collect()
    }

    /// Get suitable sources for a strategy
    fn sources_for_strategy(
        &self,
        strategy: ExplorationStrategy,
        _domain: &Domain,
    ) -> Vec<KnowledgeSource> {
        match strategy {
            ExplorationStrategy::Direct => vec![
                KnowledgeSource::Memory,
                KnowledgeSource::VectorDB,
                KnowledgeSource::GraphDB,
            ],
            ExplorationStrategy::Semantic => {
                vec![KnowledgeSource::VectorDB, KnowledgeSource::Memory]
            }
            ExplorationStrategy::Analogical => {
                vec![KnowledgeSource::GraphDB, KnowledgeSource::Inference]
            }
            ExplorationStrategy::Socratic => {
                vec![KnowledgeSource::Inference, KnowledgeSource::Memory]
            }
            ExplorationStrategy::Harmonic => {
                vec![KnowledgeSource::GraphDB, KnowledgeSource::Inference]
            }
            ExplorationStrategy::Clarification => vec![KnowledgeSource::Expert],
        }
    }

    /// Score a (strategy, source) plan
    fn plan_score(
        &self,
        strategy: ExplorationStrategy,
        source: KnowledgeSource,
        detection: &IgnoranceDetection,
    ) -> f32 {
        // Base effectiveness
        let strategy_base = strategy.base_effectiveness();
        let source_reliability = source.reliability();

        // Learned effectiveness (with prior)
        let strategy_learned = self
            .strategy_history
            .get(&strategy)
            .map(|s| s.success_rate())
            .unwrap_or(strategy_base);

        let source_learned = self
            .source_history
            .get(&source)
            .map(|s| s.success_rate())
            .unwrap_or(source_reliability);

        // Urgency factor (higher EIG = more urgent)
        let urgency = detection.eig;

        // Latency penalty (prefer faster sources for urgent queries)
        let latency_ms = source.latency_estimate();
        let latency_penalty = if urgency > 0.7 {
            1.0 / (1.0 + (latency_ms as f32 / 1000.0).ln())
        } else {
            1.0
        };

        // Combined score

        (strategy_base * 0.3 + strategy_learned * 0.3)
            * (source_reliability * 0.2 + source_learned * 0.2)
            * latency_penalty
            * (1.0 + urgency * 0.5)
    }

    /// Execute Socratic decomposition
    ///
    /// Breaks complex questions into simpler sub-questions
    pub fn socratic_decompose(&self, query: &str) -> Vec<String> {
        let mut sub_questions = Vec::new();

        // Simple heuristic decomposition
        // In production, would use LLM or semantic parsing

        // Check for compound questions (and/or)
        if query.contains(" and ") {
            for part in query.split(" and ") {
                if !part.trim().is_empty() {
                    sub_questions.push(format!("{}?", part.trim().trim_end_matches('?')));
                }
            }
        }

        // Check for causal questions
        if query.to_lowercase().contains("why") {
            sub_questions.push(format!("What is the context for: {query}?"));
            sub_questions.push(format!("What factors influence: {query}?"));
        }

        // Check for definitional questions
        if query.to_lowercase().contains("what is") {
            let term = query
                .to_lowercase()
                .replace("what is", "")
                .replace("?", "")
                .trim()
                .to_string();
            if !term.is_empty() {
                sub_questions.push(format!("What category does {term} belong to?"));
                sub_questions.push(format!("What are the key properties of {term}?"));
            }
        }

        // Check for comparative questions
        if query.to_lowercase().contains("difference between")
            || query.to_lowercase().contains("compare")
        {
            sub_questions.push("What are the key attributes to compare?".to_string());
            sub_questions.push("What context matters for this comparison?".to_string());
        }

        // Limit sub-questions
        sub_questions.truncate(self.config.max_sub_questions);

        // If no decomposition found, return original as single item
        if sub_questions.is_empty() {
            sub_questions.push(query.to_string());
        }

        sub_questions
    }

    /// Execute semantic exploration
    ///
    /// Finds related concepts via similarity
    pub fn semantic_explore(&self, query: &str, domain: &Domain) -> Vec<String> {
        // In production, would use actual HDC/vector similarity search
        // This is a placeholder implementation

        let mut related = Vec::new();

        // Domain-based related concepts
        match domain {
            Domain::Mathematics => {
                related.push("mathematical proof techniques".to_string());
                related.push("formal verification".to_string());
            }
            Domain::Physics => {
                related.push("physical laws and principles".to_string());
                related.push("experimental methods".to_string());
            }
            Domain::History => {
                related.push("historical context".to_string());
                related.push("primary sources".to_string());
            }
            Domain::Subjective => {
                related.push("perspective-taking".to_string());
                related.push("value systems".to_string());
            }
            _ => {
                related.push("general knowledge".to_string());
                related.push("common patterns".to_string());
            }
        }

        // Add query-specific terms (simple tokenization)
        for word in query.split_whitespace() {
            if word.len() > 4 {
                related.push(format!("{word} (related)"));
            }
        }

        related.truncate(10);
        related
    }

    /// Execute analogical exploration
    ///
    /// Maps unknown to known domains
    pub fn analogical_explore(&self, query: &str, source_domain: &Domain) -> Vec<(Domain, String)> {
        // Find structural analogies in other domains
        let mut analogies = Vec::new();

        // Simple analogy mapping
        match source_domain {
            Domain::Mathematics => {
                analogies.push((
                    Domain::Physics,
                    format!("Physical interpretation of: {query}"),
                ));
                analogies.push((Domain::General, format!("Everyday analogy for: {query}")));
            }
            Domain::Physics => {
                analogies.push((
                    Domain::Mathematics,
                    format!("Mathematical formulation of: {query}"),
                ));
                analogies.push((
                    Domain::General,
                    format!("Intuitive explanation of: {query}"),
                ));
            }
            Domain::History => {
                analogies.push((Domain::General, format!("Modern parallel to: {query}")));
            }
            _ => {
                analogies.push((Domain::General, format!("Simplified version of: {query}")));
            }
        }

        analogies
    }

    /// Execute harmonic exploration
    ///
    /// Explores based on affected harmonies
    pub fn harmonic_explore(
        &self,
        harmonic_ignorance: &HarmonicIgnorance,
    ) -> Vec<(Harmony, String)> {
        let mut explorations = Vec::new();

        for (harmony, impact) in &harmonic_ignorance.affected_harmonies {
            if *impact > 0.3 {
                let exploration = match harmony {
                    Harmony::ResonantCoherence => {
                        "How does this relate to coherence and integrity?".to_string()
                    }
                    Harmony::PanSentientFlourishing => {
                        "What are the well-being implications?".to_string()
                    }
                    Harmony::IntegralWisdom => "What does deep truth reveal here?".to_string(),
                    Harmony::InfinitePlay => "What creative possibilities emerge?".to_string(),
                    Harmony::UniversalInterconnectedness => {
                        "How is this connected to the larger whole?".to_string()
                    }
                    Harmony::SacredReciprocity => "What are the exchange dynamics?".to_string(),
                    Harmony::EvolutionaryProgression => {
                        "How does this serve growth and evolution?".to_string()
                    }
                    Harmony::SacredStillness => {
                        "What does contemplative stillness reveal here?".to_string()
                    }
                };
                explorations.push((*harmony, exploration));
            }
        }

        explorations
    }

    /// Record exploration result for learning
    pub fn record_result(&mut self, result: &ExplorationResult) {
        if !self.config.enable_learning {
            return;
        }

        // Update strategy stats
        let strategy_stats = self.strategy_history.entry(result.strategy).or_default();
        strategy_stats.record(
            result.success,
            result.confidence,
            result.duration.as_millis() as u64,
        );

        // Update source stats
        let source_stats = self.source_history.entry(result.source).or_default();
        source_stats.record(
            result.success,
            result.confidence,
            result.duration.as_millis() as u64,
        );

        // Store resolved result
        self.resolved.push(result.clone());

        // Trim old results
        if self.resolved.len() > 1000 {
            self.resolved.drain(0..500);
        }
    }

    /// Get best strategy for a given ignorance type based on learned history
    pub fn best_strategy(&self, ig_type: &IgnoranceType) -> Option<ExplorationStrategy> {
        let candidates = ExplorationStrategy::for_ignorance_type(ig_type);

        candidates.into_iter().max_by(|a, b| {
            let score_a = self
                .strategy_history
                .get(a)
                .map(|s| s.success_rate() * s.avg_confidence())
                .unwrap_or(a.base_effectiveness() * 0.5);
            let score_b = self
                .strategy_history
                .get(b)
                .map(|s| s.success_rate() * s.avg_confidence())
                .unwrap_or(b.base_effectiveness() * 0.5);
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get best source for a strategy based on learned history
    pub fn best_source(
        &self,
        strategy: ExplorationStrategy,
        domain: &Domain,
    ) -> Option<KnowledgeSource> {
        let candidates = self.sources_for_strategy(strategy, domain);

        candidates.into_iter().max_by(|a, b| {
            let score_a = self
                .source_history
                .get(a)
                .map(|s| s.success_rate() * (1.0 / (1.0 + s.avg_latency_ms as f32 / 1000.0)))
                .unwrap_or(a.reliability() * 0.5);
            let score_b = self
                .source_history
                .get(b)
                .map(|s| s.success_rate() * (1.0 / (1.0 + s.avg_latency_ms as f32 / 1000.0)))
                .unwrap_or(b.reliability() * 0.5);
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get exploration statistics
    pub fn statistics(&self) -> ExplorationStatistics {
        let total_attempts: u32 = self.strategy_history.values().map(|s| s.attempts).sum();
        let total_successes: u32 = self.strategy_history.values().map(|s| s.successes).sum();

        let best_strategy = self
            .strategy_history
            .iter()
            .max_by(|a, b| {
                let score_a = a.1.success_rate() * a.1.avg_confidence();
                let score_b = b.1.success_rate() * b.1.avg_confidence();
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(s, _)| *s);

        let best_source = self
            .source_history
            .iter()
            .max_by(|a, b| {
                a.1.success_rate()
                    .partial_cmp(&b.1.success_rate())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(s, _)| *s);

        ExplorationStatistics {
            total_explorations: total_attempts,
            successful_resolutions: total_successes,
            success_rate: if total_attempts > 0 {
                total_successes as f32 / total_attempts as f32
            } else {
                0.0
            },
            best_strategy,
            best_source,
            pending_count: self.pending.len(),
        }
    }
}

/// Exploration statistics
#[derive(Debug, Clone)]
pub struct ExplorationStatistics {
    pub total_explorations: u32,
    pub successful_resolutions: u32,
    pub success_rate: f32,
    pub best_strategy: Option<ExplorationStrategy>,
    pub best_source: Option<KnowledgeSource>,
    pub pending_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_for_ignorance_type() {
        // Known unknown should include semantic and socratic
        let strategies = ExplorationStrategy::for_ignorance_type(&IgnoranceType::KnownUnknown);
        assert!(strategies.contains(&ExplorationStrategy::Semantic));
        assert!(strategies.contains(&ExplorationStrategy::Socratic));

        // Unknown unknown should include clarification
        let strategies = ExplorationStrategy::for_ignorance_type(&IgnoranceType::Unknown);
        assert!(strategies.contains(&ExplorationStrategy::Clarification));
    }

    #[test]
    fn test_plan_exploration() {
        let explorer = ActiveExplorer::new(ActiveExplorerConfig::default());

        let detection = IgnoranceDetection {
            query: "What causes dark matter?".to_string(),
            ignorance_type: IgnoranceType::KnownUnknown,
            uncertainty: Uncertainty3D::new(0.6, 0.4, 0.3),
            domain: Domain::Physics,
            eig: 0.7,
            detected_at: SystemTime::now(),
        };

        let plans = explorer.plan_exploration(&detection);

        // Should have multiple plans
        assert!(!plans.is_empty());

        // Plans should include semantic exploration
        assert!(
            plans
                .iter()
                .any(|(s, _)| *s == ExplorationStrategy::Semantic)
        );
    }

    #[test]
    fn test_socratic_decompose() {
        let explorer = ActiveExplorer::new(ActiveExplorerConfig::default());

        // Compound question
        let subs = explorer.socratic_decompose("What is gravity and how does it work?");
        assert!(subs.len() >= 2);

        // Why question
        let subs = explorer.socratic_decompose("Why does water freeze?");
        assert!(
            subs.iter()
                .any(|q| q.contains("context") || q.contains("factors"))
        );
    }

    #[test]
    fn test_record_result() {
        let mut explorer = ActiveExplorer::new(ActiveExplorerConfig::default());

        let result = ExplorationResult {
            strategy: ExplorationStrategy::Semantic,
            source: KnowledgeSource::VectorDB,
            success: true,
            confidence: 0.8,
            knowledge: Some(ResolvedKnowledge {
                query: "test".to_string(),
                answer: "answer".to_string(),
                confidence: 0.8,
                source: KnowledgeSource::VectorDB,
                evidence: vec![],
                caveats: vec![],
                resolved_at: SystemTime::now(),
            }),
            sub_questions: vec![],
            related_concepts: vec!["concept1".to_string()],
            duration: Duration::from_millis(150),
        };

        explorer.record_result(&result);

        // Should have recorded the result
        let stats = explorer.statistics();
        assert_eq!(stats.total_explorations, 1);
        assert_eq!(stats.successful_resolutions, 1);
    }

    #[test]
    fn test_best_strategy_learning() {
        let mut explorer = ActiveExplorer::new(ActiveExplorerConfig::default());

        // Record some successes for semantic
        for _ in 0..5 {
            explorer.record_result(&ExplorationResult {
                strategy: ExplorationStrategy::Semantic,
                source: KnowledgeSource::VectorDB,
                success: true,
                confidence: 0.85,
                knowledge: None,
                sub_questions: vec![],
                related_concepts: vec![],
                duration: Duration::from_millis(100),
            });
        }

        // Record some failures for socratic
        for _ in 0..3 {
            explorer.record_result(&ExplorationResult {
                strategy: ExplorationStrategy::Socratic,
                source: KnowledgeSource::Inference,
                success: false,
                confidence: 0.2,
                knowledge: None,
                sub_questions: vec![],
                related_concepts: vec![],
                duration: Duration::from_millis(200),
            });
        }

        // Semantic should now be preferred for KnownUnknown
        let best = explorer.best_strategy(&IgnoranceType::KnownUnknown);
        assert_eq!(best, Some(ExplorationStrategy::Semantic));
    }

    #[test]
    fn test_harmonic_explore() {
        let explorer = ActiveExplorer::new(ActiveExplorerConfig::default());

        let mut harmonic_ignorance = HarmonicIgnorance::new(IgnoranceType::KnownUnknown);
        harmonic_ignorance.add_affected_harmony(Harmony::PanSentientFlourishing, 0.8);
        harmonic_ignorance.add_affected_harmony(Harmony::IntegralWisdom, 0.5);

        let explorations = explorer.harmonic_explore(&harmonic_ignorance);

        // Should have explorations for affected harmonies
        assert!(explorations.len() >= 2);
        assert!(
            explorations
                .iter()
                .any(|(h, _)| *h == Harmony::PanSentientFlourishing)
        );
    }

    #[test]
    fn test_source_latency_ordering() {
        // Sources should be ordered by latency for planning
        assert!(
            KnowledgeSource::Memory.latency_estimate() < KnowledgeSource::Web.latency_estimate()
        );
        assert!(
            KnowledgeSource::Inference.latency_estimate()
                < KnowledgeSource::Peer.latency_estimate()
        );
    }
}
