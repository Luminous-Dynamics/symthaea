//! F3: Wisdom Seed Transmission Research
//!
//! Addresses the question: Can wisdom be transmitted?
//!
//! Research areas:
//! - Seed influence tracking
//! - A/B testing of seed formats
//! - Citation network analysis
//! - Transmission effectiveness metrics

use crate::metrics::{BrierScore, StatisticalComparison};
use crate::simulation::SimpleRng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ============================================================================
// WISDOM SEED FORMATS
// ============================================================================

/// Different formats for presenting wisdom seeds
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum WisdomSeedFormat {
    /// Plain text lesson
    NaturalLanguage,
    /// Structured with explicit conditions
    Structured {
        has_conditions: bool,
        has_caveats: bool,
        has_examples: bool,
    },
    /// Minimal - just the core insight
    Minimal,
    /// Detailed - full context and reasoning
    Detailed,
    /// Narrative - story format
    Narrative,
    /// Checklist - actionable items
    Checklist,
    /// Question-based - prompts for reflection
    QuestionBased,
    /// Visual/diagram (simulated as having visual flag)
    Visual,
}

impl Default for WisdomSeedFormat {
    fn default() -> Self {
        Self::NaturalLanguage
    }
}

// ============================================================================
// WISDOM SEED DATA STRUCTURES
// ============================================================================

/// A wisdom seed for research purposes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchWisdomSeed {
    pub id: String,
    /// The creator of this wisdom
    pub creator_id: String,
    /// Domain this wisdom applies to
    pub domain: String,
    /// The core lesson
    pub lesson: String,
    /// Format of the seed
    pub format: WisdomSeedFormat,
    /// Confidence in the lesson
    pub confidence: f64,
    /// Conditions under which this applies
    pub conditions: Vec<String>,
    /// Known caveats/exceptions
    pub caveats: Vec<String>,
    /// Was the source prediction correct?
    pub source_was_correct: bool,
    /// Quality score of the original reasoning
    pub reasoning_quality: f64,
    /// Creation timestamp
    pub created_at: usize,
    /// Usage statistics
    pub usage_stats: WisdomUsageStats,
    /// Citation network data
    pub citations: WisdomCitations,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WisdomUsageStats {
    /// Number of times this seed was surfaced
    pub surface_count: usize,
    /// Number of times a user acknowledged viewing it
    pub view_count: usize,
    /// Number of times explicitly cited in a new prediction
    pub citation_count: usize,
    /// Usefulness ratings (0-1)
    pub usefulness_ratings: Vec<f64>,
    /// Average rating
    pub avg_usefulness: f64,
    /// Impact on viewer predictions
    pub prediction_impacts: Vec<PredictionImpact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionImpact {
    /// Viewer who was influenced
    pub viewer_id: String,
    /// Their prediction before seeing seed
    pub prediction_before: Option<f64>,
    /// Their prediction after seeing seed
    pub prediction_after: f64,
    /// Did they correctly adjust toward truth?
    pub improved: bool,
    /// Brier score change
    pub brier_change: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WisdomCitations {
    /// Seeds that cited this one
    pub cited_by: Vec<String>,
    /// Seeds this one cites
    pub cites: Vec<String>,
    /// Generation (0 = original, 1 = first generation of citations, etc.)
    pub generation: usize,
    /// PageRank-style influence score
    pub influence_score: f64,
}

// ============================================================================
// WISDOM TRANSMISSION STUDY
// ============================================================================

/// Configuration for wisdom transmission study
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WisdomStudyConfig {
    /// Formats to test
    pub formats: Vec<WisdomSeedFormat>,
    /// Number of participants per condition
    pub participants_per_condition: usize,
    /// Number of wisdom seeds to generate
    pub num_seeds: usize,
    /// Number of prediction rounds
    pub num_rounds: usize,
    /// Probability of surfacing a seed each round
    pub surfacing_probability: f64,
    /// Domains to include
    pub domains: Vec<String>,
    /// Seed
    pub seed: u64,
}

impl Default for WisdomStudyConfig {
    fn default() -> Self {
        Self {
            formats: vec![
                WisdomSeedFormat::NaturalLanguage,
                WisdomSeedFormat::Structured {
                    has_conditions: true,
                    has_caveats: true,
                    has_examples: true,
                },
                WisdomSeedFormat::Minimal,
                WisdomSeedFormat::Detailed,
                WisdomSeedFormat::QuestionBased,
            ],
            participants_per_condition: 30,
            num_seeds: 50,
            num_rounds: 100,
            surfacing_probability: 0.3,
            domains: vec![
                "politics".to_string(),
                "technology".to_string(),
                "science".to_string(),
            ],
            seed: 42,
        }
    }
}

/// A study participant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WisdomParticipant {
    pub id: String,
    /// Assigned format condition
    pub assigned_format: WisdomSeedFormat,
    /// Predictions made
    pub predictions: Vec<WisdomPrediction>,
    /// Seeds viewed
    pub seeds_viewed: Vec<String>,
    /// Seeds cited
    pub seeds_cited: Vec<String>,
    /// Current calibration
    pub calibration: f64,
    /// Running Brier score
    pub brier_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WisdomPrediction {
    pub round: usize,
    pub domain: String,
    pub predicted: f64,
    pub outcome: bool,
    pub brier: f64,
    /// Seed shown before prediction (if any)
    pub seed_shown: Option<String>,
    /// Whether the seed was cited
    pub cited_seed: bool,
    /// Prediction before seeing seed (if applicable)
    pub pre_seed_prediction: Option<f64>,
}

/// Run a wisdom transmission study
pub struct WisdomTransmissionStudy {
    pub config: WisdomStudyConfig,
    pub seeds: Vec<ResearchWisdomSeed>,
    pub participants: Vec<WisdomParticipant>,
    pub results: WisdomStudyResults,
    rng: SimpleRng,
}

impl WisdomTransmissionStudy {
    pub fn new(config: WisdomStudyConfig) -> Self {
        let rng = SimpleRng::new(config.seed);
        Self {
            config,
            seeds: vec![],
            participants: vec![],
            results: WisdomStudyResults::new(),
            rng,
        }
    }

    /// Initialize seeds and participants
    pub fn initialize(&mut self) {
        // Generate wisdom seeds
        for i in 0..self.config.num_seeds {
            let domain_idx = i % self.config.domains.len();
            let domain = self.config.domains[domain_idx].clone();

            let format_idx = i % self.config.formats.len();
            let format = self.config.formats[format_idx].clone();

            let seed = ResearchWisdomSeed {
                id: format!("seed_{}", i),
                creator_id: format!("creator_{}", i % 10),
                domain,
                lesson: format!("Lesson {} for domain considerations", i),
                format,
                confidence: 0.6 + self.rng.next_u64() as f64 / u64::MAX as f64 * 0.3,
                conditions: vec!["condition1".to_string()],
                caveats: vec!["caveat1".to_string()],
                source_was_correct: self.rng.next_u64() % 2 == 0,
                reasoning_quality: 0.5 + self.rng.next_u64() as f64 / u64::MAX as f64 * 0.4,
                created_at: 0,
                usage_stats: WisdomUsageStats::default(),
                citations: WisdomCitations::default(),
            };

            self.seeds.push(seed);
        }

        // Generate participants for each condition
        for (format_idx, format) in self.config.formats.iter().enumerate() {
            for i in 0..self.config.participants_per_condition {
                let participant = WisdomParticipant {
                    id: format!("participant_{}_{}", format_idx, i),
                    assigned_format: format.clone(),
                    predictions: vec![],
                    seeds_viewed: vec![],
                    seeds_cited: vec![],
                    calibration: 0.1 + self.rng.next_u64() as f64 / u64::MAX as f64 * 0.15,
                    brier_score: 0.2,
                };
                self.participants.push(participant);
            }
        }
    }

    /// Run the study
    pub fn run(&mut self) -> &WisdomStudyResults {
        self.initialize();

        for round in 0..self.config.num_rounds {
            self.run_round(round);
        }

        self.analyze_results();
        &self.results
    }

    fn run_round(&mut self, round: usize) {
        let domain_idx = round % self.config.domains.len();
        let domain = self.config.domains[domain_idx].clone();

        // Generate true probability for this round
        let true_prob = 0.2 + self.rng.next_u64() as f64 / u64::MAX as f64 * 0.6;

        for participant_idx in 0..self.participants.len() {
            // Decide if we show a seed
            let show_seed = (self.rng.next_u64() as f64 / u64::MAX as f64)
                < self.config.surfacing_probability;

            let seed_shown = if show_seed {
                self.select_seed_for_participant(participant_idx, &domain)
            } else {
                None
            };

            // Record pre-seed prediction if applicable
            let pre_seed = if seed_shown.is_some() {
                let noise = (self.rng.next_u64() as f64 / u64::MAX as f64 - 0.5)
                    * self.participants[participant_idx].calibration * 2.0;
                Some((true_prob + noise).clamp(0.01, 0.99))
            } else {
                None
            };

            // Generate prediction (possibly influenced by seed)
            let influence = if let Some(ref seed_id) = seed_shown {
                self.calculate_seed_influence(seed_id, &domain, true_prob)
            } else {
                0.0
            };

            let noise = (self.rng.next_u64() as f64 / u64::MAX as f64 - 0.5)
                * self.participants[participant_idx].calibration * 2.0;
            let predicted = (true_prob + noise + influence).clamp(0.01, 0.99);

            // Determine outcome
            let outcome = (self.rng.next_u64() as f64 / u64::MAX as f64) < true_prob;
            let brier = (predicted - if outcome { 1.0 } else { 0.0 }).powi(2);

            // Decide if participant cites the seed
            let cited = if let Some(ref seed_id) = seed_shown {
                let cite_prob = 0.2 + influence.abs() * 0.5;
                (self.rng.next_u64() as f64 / u64::MAX as f64) < cite_prob
            } else {
                false
            };

            // Update participant - first block, limited scope
            {
                let participant = &mut self.participants[participant_idx];
                participant.predictions.push(WisdomPrediction {
                    round,
                    domain: domain.clone(),
                    predicted,
                    outcome,
                    brier,
                    seed_shown: seed_shown.clone(),
                    cited_seed: cited,
                    pre_seed_prediction: pre_seed,
                });

                if let Some(ref seed_id) = seed_shown {
                    participant.seeds_viewed.push(seed_id.clone());
                    if cited {
                        participant.seeds_cited.push(seed_id.clone());
                    }
                }
            }

            // Update seed stats (requires mutable self)
            if let Some(ref seed_id) = seed_shown {
                self.update_seed_stats(seed_id, participant_idx, pre_seed, predicted, outcome);
            }

            // Update running Brier - second mutable borrow
            {
                let participant = &mut self.participants[participant_idx];
                let prediction_count = participant.predictions.len();
                let old_brier = participant.brier_score;
                participant.brier_score =
                    (old_brier * (prediction_count - 1) as f64 + brier) / prediction_count as f64;
            }
        }
    }

    fn select_seed_for_participant(&mut self, participant_idx: usize, domain: &str) -> Option<String> {
        let participant = &self.participants[participant_idx];

        // Select a seed matching format and domain
        let matching_seeds: Vec<&ResearchWisdomSeed> = self
            .seeds
            .iter()
            .filter(|s| {
                s.format == participant.assigned_format
                    && s.domain == domain
                    && !participant.seeds_viewed.contains(&s.id)
            })
            .collect();

        if matching_seeds.is_empty() {
            // Fall back to any seed in domain not yet viewed
            let fallback: Vec<&ResearchWisdomSeed> = self
                .seeds
                .iter()
                .filter(|s| s.domain == domain && !participant.seeds_viewed.contains(&s.id))
                .collect();

            fallback
                .get(self.rng.next_u64() as usize % fallback.len().max(1))
                .map(|s| s.id.clone())
        } else {
            matching_seeds
                .get(self.rng.next_u64() as usize % matching_seeds.len())
                .map(|s| s.id.clone())
        }
    }

    fn calculate_seed_influence(
        &self,
        seed_id: &str,
        _domain: &str,
        true_prob: f64,
    ) -> f64 {
        let seed = self.seeds.iter().find(|s| s.id == seed_id);

        if let Some(seed) = seed {
            // Seeds from correct predictions are more influential
            let correctness_bonus = if seed.source_was_correct { 0.05 } else { -0.02 };

            // Higher reasoning quality = more trust
            let quality_bonus = (seed.reasoning_quality - 0.5) * 0.1;

            // Higher confidence in seed = more influence
            let confidence_factor = seed.confidence * 0.05;

            // Direction: push toward 0.5 if cautionary, toward extreme if confident
            let direction = if seed.source_was_correct {
                // Correct prediction - push toward true prob
                (true_prob - 0.5).signum() * 0.03
            } else {
                // Incorrect prediction - lesson is cautionary
                -(true_prob - 0.5).signum() * 0.02
            };

            correctness_bonus + quality_bonus + confidence_factor + direction
        } else {
            0.0
        }
    }

    fn update_seed_stats(
        &mut self,
        seed_id: &str,
        participant_idx: usize,
        pre_seed: Option<f64>,
        prediction: f64,
        outcome: bool,
    ) {
        let seed = self.seeds.iter_mut().find(|s| s.id == seed_id);
        let participant = &self.participants[participant_idx];

        if let Some(seed) = seed {
            seed.usage_stats.surface_count += 1;
            seed.usage_stats.view_count += 1;

            if participant.seeds_cited.iter().any(|s| s == seed_id) {
                seed.usage_stats.citation_count += 1;
            }

            if let Some(pre) = pre_seed {
                let outcome_val = if outcome { 1.0 } else { 0.0 };
                let pre_brier = (pre - outcome_val).powi(2);
                let post_brier = (prediction - outcome_val).powi(2);
                let brier_change = pre_brier - post_brier;
                let improved = brier_change > 0.0;

                seed.usage_stats.prediction_impacts.push(PredictionImpact {
                    viewer_id: participant.id.clone(),
                    prediction_before: Some(pre),
                    prediction_after: prediction,
                    improved,
                    brier_change,
                });
            }
        }
    }

    fn analyze_results(&mut self) {
        // Analyze by format
        for format in &self.config.formats {
            let format_participants: Vec<&WisdomParticipant> = self
                .participants
                .iter()
                .filter(|p| p.assigned_format == *format)
                .collect();

            if format_participants.is_empty() {
                continue;
            }

            // Calculate influence metrics
            let total_citations: usize = format_participants.iter().map(|p| p.seeds_cited.len()).sum();
            let total_views: usize = format_participants.iter().map(|p| p.seeds_viewed.len()).sum();

            let citation_rate = if total_views > 0 {
                total_citations as f64 / total_views as f64
            } else {
                0.0
            };

            // Calculate Brier improvement when seeds were shown
            let mut brier_with_seed = Vec::new();
            let mut brier_without_seed = Vec::new();

            for participant in &format_participants {
                for pred in &participant.predictions {
                    if pred.seed_shown.is_some() {
                        brier_with_seed.push(pred.brier);
                    } else {
                        brier_without_seed.push(pred.brier);
                    }
                }
            }

            let avg_brier_with = if !brier_with_seed.is_empty() {
                brier_with_seed.iter().sum::<f64>() / brier_with_seed.len() as f64
            } else {
                0.0
            };

            let avg_brier_without = if !brier_without_seed.is_empty() {
                brier_without_seed.iter().sum::<f64>() / brier_without_seed.len() as f64
            } else {
                0.0
            };

            self.results.format_effectiveness.insert(
                format.clone(),
                FormatEffectiveness {
                    format: format.clone(),
                    citation_rate,
                    avg_brier_with_seed: avg_brier_with,
                    avg_brier_without_seed: avg_brier_without,
                    brier_improvement: avg_brier_without - avg_brier_with,
                    sample_size: format_participants.len(),
                },
            );
        }

        // Build citation network
        self.build_citation_network();

        // Calculate influence scores
        self.calculate_influence_scores();

        // Generate summary
        self.results.summary = self.generate_summary();
    }

    fn build_citation_network(&mut self) {
        // Track which seeds cite which (based on being viewed together)
        for participant in &self.participants {
            let viewed: Vec<&String> = participant.seeds_viewed.iter().collect();

            for (i, seed_id) in viewed.iter().enumerate() {
                // Seeds viewed after this one could be considered "influenced"
                for later_seed in viewed.iter().skip(i + 1) {
                    if let Some(seed) = self.seeds.iter_mut().find(|s| &s.id == *later_seed) {
                        if !seed.citations.cites.contains(*seed_id) {
                            seed.citations.cites.push((*seed_id).clone());
                        }
                    }
                    if let Some(seed) = self.seeds.iter_mut().find(|s| &s.id == *seed_id) {
                        if !seed.citations.cited_by.contains(*later_seed) {
                            seed.citations.cited_by.push((*later_seed).clone());
                        }
                    }
                }
            }
        }

        // Calculate generations
        self.calculate_generations();
    }

    fn calculate_generations(&mut self) {
        // BFS to calculate generation from root seeds
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: Vec<(String, usize)> = Vec::new();

        // Seeds with no inbound citations are generation 0
        for seed in &self.seeds {
            if seed.citations.cites.is_empty() {
                queue.push((seed.id.clone(), 0));
            }
        }

        while let Some((seed_id, gen)) = queue.pop() {
            if visited.contains(&seed_id) {
                continue;
            }
            visited.insert(seed_id.clone());

            if let Some(seed) = self.seeds.iter_mut().find(|s| s.id == seed_id) {
                seed.citations.generation = gen;

                for cited_by in seed.citations.cited_by.clone() {
                    if !visited.contains(&cited_by) {
                        queue.push((cited_by, gen + 1));
                    }
                }
            }
        }
    }

    fn calculate_influence_scores(&mut self) {
        // Simple PageRank-like algorithm
        let num_seeds = self.seeds.len();
        let damping = 0.85;
        let iterations = 20;

        // Initialize scores
        for seed in &mut self.seeds {
            seed.citations.influence_score = 1.0 / num_seeds as f64;
        }

        for _ in 0..iterations {
            let old_scores: HashMap<String, f64> = self
                .seeds
                .iter()
                .map(|s| (s.id.clone(), s.citations.influence_score))
                .collect();

            for seed in &mut self.seeds {
                let incoming_score: f64 = seed
                    .citations
                    .cites
                    .iter()
                    .filter_map(|citer_id| old_scores.get(citer_id))
                    .sum();

                let num_citers = seed.citations.cites.len().max(1);
                let new_score =
                    (1.0 - damping) / num_seeds as f64 + damping * incoming_score / num_citers as f64;

                seed.citations.influence_score = new_score;
            }
        }

        // Normalize
        let total: f64 = self.seeds.iter().map(|s| s.citations.influence_score).sum();
        if total > 0.0 {
            for seed in &mut self.seeds {
                seed.citations.influence_score /= total;
            }
        }
    }

    fn generate_summary(&self) -> WisdomSummary {
        // Find best format
        let best_format = self
            .results
            .format_effectiveness
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.brier_improvement
                    .partial_cmp(&b.brier_improvement)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(f, _)| f.clone())
            .unwrap_or(WisdomSeedFormat::NaturalLanguage);

        // Calculate transmission effectiveness
        let positive_impacts: usize = self
            .seeds
            .iter()
            .flat_map(|s| &s.usage_stats.prediction_impacts)
            .filter(|i| i.improved)
            .count();
        let total_impacts: usize = self
            .seeds
            .iter()
            .map(|s| s.usage_stats.prediction_impacts.len())
            .sum();

        let transmission_rate = if total_impacts > 0 {
            positive_impacts as f64 / total_impacts as f64
        } else {
            0.0
        };

        // Network statistics
        let max_generation = self.seeds.iter().map(|s| s.citations.generation).max().unwrap_or(0);
        let avg_citations: f64 = self.seeds.iter().map(|s| s.citations.cited_by.len()).sum::<usize>()
            as f64
            / self.seeds.len().max(1) as f64;

        // Top influencers
        let mut sorted_seeds = self.seeds.clone();
        sorted_seeds.sort_by(|a, b| {
            b.citations
                .influence_score
                .partial_cmp(&a.citations.influence_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_influencers: Vec<String> = sorted_seeds
            .iter()
            .take(5)
            .map(|s| s.id.clone())
            .collect();

        WisdomSummary {
            best_format,
            transmission_rate,
            avg_brier_improvement: self.calculate_avg_brier_improvement(),
            wisdom_is_transmissible: transmission_rate > 0.5,
            network_depth: max_generation,
            avg_citations_per_seed: avg_citations,
            top_influencer_seeds: top_influencers,
            recommendations: self.generate_recommendations(),
        }
    }

    fn calculate_avg_brier_improvement(&self) -> f64 {
        let improvements: Vec<f64> = self
            .seeds
            .iter()
            .flat_map(|s| &s.usage_stats.prediction_impacts)
            .map(|i| i.brier_change)
            .collect();

        if improvements.is_empty() {
            0.0
        } else {
            improvements.iter().sum::<f64>() / improvements.len() as f64
        }
    }

    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Format recommendation
        if let Some((format, eff)) = self
            .results
            .format_effectiveness
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.brier_improvement
                    .partial_cmp(&b.brier_improvement)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            recommendations.push(format!(
                "Best format: {:?} (Brier improvement: {:.4})",
                format, eff.brier_improvement
            ));
        }

        // Citation recommendation
        let avg_citation_rate: f64 = self
            .results
            .format_effectiveness
            .values()
            .map(|f| f.citation_rate)
            .sum::<f64>()
            / self.results.format_effectiveness.len().max(1) as f64;

        if avg_citation_rate < 0.2 {
            recommendations.push(
                "Low citation rate - consider prompting users to engage with wisdom seeds"
                    .to_string(),
            );
        }

        // Source correctness recommendation
        let correct_sources: Vec<&ResearchWisdomSeed> =
            self.seeds.iter().filter(|s| s.source_was_correct).collect();
        let correct_avg_citations = if !correct_sources.is_empty() {
            correct_sources
                .iter()
                .map(|s| s.usage_stats.citation_count)
                .sum::<usize>() as f64
                / correct_sources.len() as f64
        } else {
            0.0
        };

        let incorrect_sources: Vec<&ResearchWisdomSeed> =
            self.seeds.iter().filter(|s| !s.source_was_correct).collect();
        let incorrect_avg_citations = if !incorrect_sources.is_empty() {
            incorrect_sources
                .iter()
                .map(|s| s.usage_stats.citation_count)
                .sum::<usize>() as f64
                / incorrect_sources.len() as f64
        } else {
            0.0
        };

        if correct_avg_citations > incorrect_avg_citations * 1.5 {
            recommendations.push(
                "Wisdom from correct predictions is cited more - consider highlighting source accuracy"
                    .to_string(),
            );
        }

        recommendations
    }
}

// ============================================================================
// STUDY RESULTS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WisdomStudyResults {
    pub format_effectiveness: HashMap<WisdomSeedFormat, FormatEffectiveness>,
    pub citation_network_stats: CitationNetworkStats,
    pub summary: WisdomSummary,
}

impl WisdomStudyResults {
    pub fn new() -> Self {
        Self {
            format_effectiveness: HashMap::new(),
            citation_network_stats: CitationNetworkStats::default(),
            summary: WisdomSummary::default(),
        }
    }
}

impl Default for WisdomStudyResults {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatEffectiveness {
    pub format: WisdomSeedFormat,
    pub citation_rate: f64,
    pub avg_brier_with_seed: f64,
    pub avg_brier_without_seed: f64,
    pub brier_improvement: f64,
    pub sample_size: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CitationNetworkStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub avg_degree: f64,
    pub max_depth: usize,
    pub clustering_coefficient: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WisdomSummary {
    pub best_format: WisdomSeedFormat,
    pub transmission_rate: f64,
    pub avg_brier_improvement: f64,
    pub wisdom_is_transmissible: bool,
    pub network_depth: usize,
    pub avg_citations_per_seed: f64,
    pub top_influencer_seeds: Vec<String>,
    pub recommendations: Vec<String>,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wisdom_seed_creation() {
        let seed = ResearchWisdomSeed {
            id: "test".to_string(),
            creator_id: "creator".to_string(),
            domain: "politics".to_string(),
            lesson: "Test lesson".to_string(),
            format: WisdomSeedFormat::NaturalLanguage,
            confidence: 0.8,
            conditions: vec![],
            caveats: vec![],
            source_was_correct: true,
            reasoning_quality: 0.7,
            created_at: 0,
            usage_stats: WisdomUsageStats::default(),
            citations: WisdomCitations::default(),
        };

        assert_eq!(seed.id, "test");
        assert_eq!(seed.confidence, 0.8);
    }

    #[test]
    fn test_study_initialization() {
        let config = WisdomStudyConfig {
            participants_per_condition: 5,
            num_seeds: 10,
            num_rounds: 5,
            formats: vec![WisdomSeedFormat::NaturalLanguage],
            ..Default::default()
        };

        let mut study = WisdomTransmissionStudy::new(config);
        study.initialize();

        assert_eq!(study.seeds.len(), 10);
        assert_eq!(study.participants.len(), 5);
    }

    #[test]
    fn test_small_study() {
        let config = WisdomStudyConfig {
            participants_per_condition: 3,
            num_seeds: 5,
            num_rounds: 10,
            formats: vec![WisdomSeedFormat::NaturalLanguage, WisdomSeedFormat::Minimal],
            domains: vec!["test".to_string()],
            ..Default::default()
        };

        let mut study = WisdomTransmissionStudy::new(config);
        let results = study.run();

        assert!(!results.format_effectiveness.is_empty());
    }

    #[test]
    fn test_citation_generation() {
        let mut seed = ResearchWisdomSeed {
            id: "test".to_string(),
            creator_id: "creator".to_string(),
            domain: "test".to_string(),
            lesson: "Test".to_string(),
            format: WisdomSeedFormat::default(),
            confidence: 0.5,
            conditions: vec![],
            caveats: vec![],
            source_was_correct: true,
            reasoning_quality: 0.5,
            created_at: 0,
            usage_stats: WisdomUsageStats::default(),
            citations: WisdomCitations::default(),
        };

        assert_eq!(seed.citations.generation, 0);
        seed.citations.generation = 2;
        assert_eq!(seed.citations.generation, 2);
    }
}
