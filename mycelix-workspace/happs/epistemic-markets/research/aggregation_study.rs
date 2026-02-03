//! F1: Aggregation Effectiveness Research
//!
//! Addresses the foundational question: Does aggregation actually work?
//!
//! Research areas:
//! - Comparative analysis vs expert prediction
//! - Conditions where aggregation fails
//! - Minimum viable diversity
//! - Question-type specific effectiveness

use crate::metrics::{
    AggregationMetrics, BrierDecomposition, BrierScore, ExpectedCalibrationError,
    StatisticalComparison,
};
use crate::simulation::{
    AgentPopulationConfig, AgentType, SimulatedAgent, SimulatedMarket, SimulationConfig,
    SimulationEngine, SimpleRng,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// AGGREGATION METHODS
// ============================================================================

/// Different methods for aggregating predictions
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AggregationMethod {
    /// Simple arithmetic mean
    SimpleMean,
    /// Median prediction
    Median,
    /// Geometric mean (better for probabilities)
    GeometricMean,
    /// Weighted by past accuracy (Brier-weighted)
    BrierWeighted,
    /// Weighted by MATL score
    MATLWeighted,
    /// Weighted by stake amount
    StakeWeighted,
    /// Extremized aggregate (sharpen predictions)
    Extremized { factor: u32 },
    /// Trimmed mean (remove outliers)
    TrimmedMean { trim_percent: u32 },
    /// Logarithmic opinion pool
    LogarithmicPool,
    /// Bayesian aggregation (treat as independent signals)
    Bayesian,
}

/// Aggregate predictions using specified method
pub fn aggregate_predictions(
    predictions: &[(f64, f64)], // (prediction, weight)
    method: AggregationMethod,
) -> f64 {
    if predictions.is_empty() {
        return 0.5;
    }

    match method {
        AggregationMethod::SimpleMean => {
            predictions.iter().map(|(p, _)| *p).sum::<f64>() / predictions.len() as f64
        }

        AggregationMethod::Median => {
            let mut sorted: Vec<f64> = predictions.iter().map(|(p, _)| *p).collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                sorted[mid]
            }
        }

        AggregationMethod::GeometricMean => {
            let epsilon = 1e-10;
            let log_sum: f64 = predictions
                .iter()
                .map(|(p, _)| p.max(epsilon).ln())
                .sum();
            (log_sum / predictions.len() as f64).exp()
        }

        AggregationMethod::BrierWeighted => {
            let total_weight: f64 = predictions.iter().map(|(_, w)| *w).sum();
            if total_weight <= 0.0 {
                return aggregate_predictions(predictions, AggregationMethod::SimpleMean);
            }
            predictions.iter().map(|(p, w)| p * w).sum::<f64>() / total_weight
        }

        AggregationMethod::MATLWeighted => {
            // Same as Brier-weighted but weights are MATL scores
            aggregate_predictions(predictions, AggregationMethod::BrierWeighted)
        }

        AggregationMethod::StakeWeighted => {
            aggregate_predictions(predictions, AggregationMethod::BrierWeighted)
        }

        AggregationMethod::Extremized { factor } => {
            let mean = aggregate_predictions(predictions, AggregationMethod::SimpleMean);
            // Push away from 0.5
            let extremized = 0.5 + (mean - 0.5) * (1.0 + factor as f64 * 0.1);
            extremized.clamp(0.01, 0.99)
        }

        AggregationMethod::TrimmedMean { trim_percent } => {
            let mut sorted: Vec<f64> = predictions.iter().map(|(p, _)| *p).collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let trim_count = (sorted.len() as f64 * trim_percent as f64 / 100.0).floor() as usize;
            let trimmed = &sorted[trim_count..(sorted.len() - trim_count).max(trim_count + 1)];

            trimmed.iter().sum::<f64>() / trimmed.len() as f64
        }

        AggregationMethod::LogarithmicPool => {
            let epsilon = 1e-10;
            let n = predictions.len() as f64;

            // Log opinion pool: product of (p_i)^w_i normalized
            let log_yes: f64 = predictions
                .iter()
                .map(|(p, w)| w * p.max(epsilon).ln())
                .sum();
            let log_no: f64 = predictions
                .iter()
                .map(|(p, w)| w * (1.0 - p).max(epsilon).ln())
                .sum();

            let total_w: f64 = predictions.iter().map(|(_, w)| *w).sum();
            if total_w <= 0.0 {
                return 0.5;
            }

            let p_yes = (log_yes / total_w).exp();
            let p_no = (log_no / total_w).exp();

            p_yes / (p_yes + p_no)
        }

        AggregationMethod::Bayesian => {
            // Treat predictions as independent signals
            // Use log-odds aggregation
            let epsilon = 1e-10;

            let log_odds_sum: f64 = predictions
                .iter()
                .map(|(p, w)| {
                    let p_clamped = p.clamp(epsilon, 1.0 - epsilon);
                    let log_odds = (p_clamped / (1.0 - p_clamped)).ln();
                    log_odds * w
                })
                .sum();

            let total_w: f64 = predictions.iter().map(|(_, w)| *w).sum();
            let avg_log_odds = if total_w > 0.0 {
                log_odds_sum / total_w
            } else {
                0.0
            };

            // Convert back to probability
            1.0 / (1.0 + (-avg_log_odds).exp())
        }
    }
}

// ============================================================================
// AGGREGATION EFFECTIVENESS STUDY
// ============================================================================

/// Configuration for aggregation effectiveness study
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationStudyConfig {
    /// Methods to compare
    pub methods: Vec<AggregationMethod>,
    /// Number of simulation runs per condition
    pub runs_per_condition: usize,
    /// Agent count variations
    pub agent_counts: Vec<usize>,
    /// Diversity levels to test (0 = homogeneous, 1 = maximally diverse)
    pub diversity_levels: Vec<f64>,
    /// Independence levels (0 = fully correlated, 1 = fully independent)
    pub independence_levels: Vec<f64>,
    /// Question types to test
    pub question_types: Vec<QuestionType>,
    /// Include expert comparison
    pub compare_to_experts: bool,
    /// Number of experts to simulate
    pub num_experts: usize,
}

impl Default for AggregationStudyConfig {
    fn default() -> Self {
        Self {
            methods: vec![
                AggregationMethod::SimpleMean,
                AggregationMethod::Median,
                AggregationMethod::GeometricMean,
                AggregationMethod::Extremized { factor: 2 },
                AggregationMethod::TrimmedMean { trim_percent: 10 },
                AggregationMethod::LogarithmicPool,
            ],
            runs_per_condition: 100,
            agent_counts: vec![5, 10, 25, 50, 100, 500],
            diversity_levels: vec![0.1, 0.3, 0.5, 0.7, 0.9],
            independence_levels: vec![0.1, 0.3, 0.5, 0.7, 0.9],
            question_types: vec![
                QuestionType::Factual,
                QuestionType::Prediction,
                QuestionType::Estimation,
                QuestionType::Subjective,
            ],
            compare_to_experts: true,
            num_experts: 3,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum QuestionType {
    /// Questions with objectively verifiable answers
    Factual,
    /// Future event predictions
    Prediction,
    /// Numerical estimations
    Estimation,
    /// Subjective/opinion-based
    Subjective,
}

/// Run an aggregation effectiveness study
pub struct AggregationStudy {
    pub config: AggregationStudyConfig,
    pub results: AggregationStudyResults,
    rng: SimpleRng,
}

impl AggregationStudy {
    pub fn new(config: AggregationStudyConfig) -> Self {
        Self {
            config,
            results: AggregationStudyResults::new(),
            rng: SimpleRng::new(42),
        }
    }

    /// Run the full study
    pub fn run(&mut self) -> &AggregationStudyResults {
        // Test each condition combination
        for &agent_count in &self.config.agent_counts.clone() {
            for &diversity in &self.config.diversity_levels.clone() {
                for &independence in &self.config.independence_levels.clone() {
                    for question_type in &self.config.question_types.clone() {
                        self.run_condition(
                            agent_count,
                            diversity,
                            independence,
                            *question_type,
                        );
                    }
                }
            }
        }

        self.analyze_results();
        &self.results
    }

    fn run_condition(
        &mut self,
        agent_count: usize,
        diversity: f64,
        independence: f64,
        question_type: QuestionType,
    ) {
        let condition = StudyCondition::from_floats(
            agent_count,
            diversity,
            independence,
            question_type,
        );

        let mut condition_results = ConditionResults {
            condition: condition.clone(),
            method_scores: HashMap::new(),
            expert_comparison: None,
            individual_distribution: IndividualDistribution::default(),
        };

        // Run multiple trials
        let mut all_individual_briers = Vec::new();

        for _ in 0..self.config.runs_per_condition {
            // Generate agents for this trial
            let agents = self.generate_agents(agent_count, diversity);

            // Generate a question
            let true_prob = self.generate_question_truth(question_type);

            // Collect predictions with optional correlation
            let predictions = self.collect_predictions(&agents, true_prob, independence);

            // Determine actual outcome
            let outcome = (self.rng.next_u64() as f64 / u64::MAX as f64) < true_prob;
            let outcome_value = if outcome { 1.0 } else { 0.0 };

            // Test each aggregation method
            for &method in &self.config.methods {
                let weighted_preds: Vec<(f64, f64)> =
                    predictions.iter().map(|p| (*p, 1.0)).collect();
                let aggregate = aggregate_predictions(&weighted_preds, method);
                let brier = (aggregate - outcome_value).powi(2);

                condition_results
                    .method_scores
                    .entry(method)
                    .or_insert_with(Vec::new)
                    .push(brier);
            }

            // Track individual Brier scores
            for &pred in &predictions {
                let brier = (pred - outcome_value).powi(2);
                all_individual_briers.push(brier);
            }

            // Expert comparison
            if self.config.compare_to_experts {
                let expert_preds =
                    self.generate_expert_predictions(true_prob, self.config.num_experts);
                let expert_aggregate: f64 =
                    expert_preds.iter().sum::<f64>() / expert_preds.len() as f64;
                let expert_brier = (expert_aggregate - outcome_value).powi(2);

                condition_results
                    .expert_comparison
                    .get_or_insert_with(Vec::new)
                    .push(expert_brier);
            }
        }

        // Calculate individual distribution stats
        if !all_individual_briers.is_empty() {
            let mean = all_individual_briers.iter().sum::<f64>() / all_individual_briers.len() as f64;
            let min = all_individual_briers
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            let max = all_individual_briers
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let mut sorted = all_individual_briers.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = sorted[sorted.len() / 2];
            let p25 = sorted[sorted.len() / 4];
            let p75 = sorted[3 * sorted.len() / 4];

            condition_results.individual_distribution = IndividualDistribution {
                mean,
                median,
                min,
                max,
                p25,
                p75,
            };
        }

        self.results.condition_results.push(condition_results);
    }

    fn generate_agents(&mut self, count: usize, diversity: f64) -> Vec<SimulatedAgent> {
        let mut agents = Vec::with_capacity(count);

        // Distribution of agent types based on diversity
        let type_probs = if diversity < 0.3 {
            // Low diversity: mostly one type
            vec![
                (AgentType::Noise, 0.8),
                (AgentType::Informed, 0.2),
            ]
        } else if diversity < 0.7 {
            // Medium diversity
            vec![
                (AgentType::Noise, 0.4),
                (AgentType::Informed, 0.3),
                (AgentType::Expert, 0.2),
                (AgentType::Contrarian, 0.1),
            ]
        } else {
            // High diversity
            vec![
                (AgentType::Noise, 0.2),
                (AgentType::Informed, 0.25),
                (AgentType::Expert, 0.2),
                (AgentType::Contrarian, 0.15),
                (AgentType::Herder, 0.1),
                (AgentType::Novice, 0.1),
            ]
        };

        for i in 0..count {
            let agent_type = self.sample_agent_type(&type_probs);

            // Vary calibration and information quality based on diversity
            let base_calibration = 0.15;
            let calibration_var = 0.1 * diversity;
            let calibration = base_calibration
                + (self.rng.next_u64() as f64 / u64::MAX as f64 - 0.5) * calibration_var * 2.0;

            let base_info = 0.5;
            let info_var = 0.3 * diversity;
            let info_quality = base_info
                + (self.rng.next_u64() as f64 / u64::MAX as f64 - 0.5) * info_var * 2.0;

            agents.push(SimulatedAgent::new(
                format!("agent_{}", i),
                agent_type,
                calibration.max(0.0),
                info_quality.clamp(0.1, 0.9),
            ));
        }

        agents
    }

    fn sample_agent_type(&mut self, probs: &[(AgentType, f64)]) -> AgentType {
        let roll = self.rng.next_u64() as f64 / u64::MAX as f64;
        let mut cumulative = 0.0;
        for (t, p) in probs {
            cumulative += p;
            if roll < cumulative {
                return *t;
            }
        }
        probs.last().map(|(t, _)| *t).unwrap_or(AgentType::Noise)
    }

    fn generate_question_truth(&mut self, question_type: QuestionType) -> f64 {
        match question_type {
            QuestionType::Factual => {
                // Binary, often clear answer
                if self.rng.next_u64() % 2 == 0 {
                    0.1 + self.rng.next_u64() as f64 / u64::MAX as f64 * 0.2
                } else {
                    0.7 + self.rng.next_u64() as f64 / u64::MAX as f64 * 0.2
                }
            }
            QuestionType::Prediction => {
                // Full range of probabilities
                0.1 + self.rng.next_u64() as f64 / u64::MAX as f64 * 0.8
            }
            QuestionType::Estimation => {
                // Often near extreme values
                let extreme = self.rng.next_u64() % 4 == 0;
                if extreme {
                    if self.rng.next_u64() % 2 == 0 {
                        0.05 + self.rng.next_u64() as f64 / u64::MAX as f64 * 0.1
                    } else {
                        0.85 + self.rng.next_u64() as f64 / u64::MAX as f64 * 0.1
                    }
                } else {
                    0.3 + self.rng.next_u64() as f64 / u64::MAX as f64 * 0.4
                }
            }
            QuestionType::Subjective => {
                // Often near 0.5 (uncertain)
                0.3 + self.rng.next_u64() as f64 / u64::MAX as f64 * 0.4
            }
        }
    }

    fn collect_predictions(
        &mut self,
        agents: &[SimulatedAgent],
        true_prob: f64,
        independence: f64,
    ) -> Vec<f64> {
        let mut predictions = Vec::with_capacity(agents.len());

        // Generate a "common belief" that correlated agents will follow
        let common_belief = true_prob + (self.rng.next_u64() as f64 / u64::MAX as f64 - 0.5) * 0.4;

        for agent in agents {
            // Private signal based on information quality
            let private_signal = if (self.rng.next_u64() as f64 / u64::MAX as f64)
                < agent.information_quality
            {
                true_prob + ((self.rng.next_u64() as f64 / u64::MAX as f64) - 0.5) * 0.2
            } else {
                self.rng.next_u64() as f64 / u64::MAX as f64
            };

            // Blend private signal with common belief based on independence
            let base_estimate = independence * private_signal + (1.0 - independence) * common_belief;

            // Apply agent-specific noise/bias
            let noise = (self.rng.next_u64() as f64 / u64::MAX as f64 - 0.5)
                * agent.calibration_error
                * 2.0;

            let prediction = match agent.agent_type {
                AgentType::Contrarian => 1.0 - base_estimate + noise,
                AgentType::Herder => common_belief + noise * 0.5,
                _ => base_estimate + noise,
            };

            predictions.push(prediction.clamp(0.01, 0.99));
        }

        predictions
    }

    fn generate_expert_predictions(&mut self, true_prob: f64, num_experts: usize) -> Vec<f64> {
        let mut predictions = Vec::with_capacity(num_experts);

        for _ in 0..num_experts {
            // Experts have higher information quality but can still be wrong
            let noise = (self.rng.next_u64() as f64 / u64::MAX as f64 - 0.5) * 0.15;
            let pred = (true_prob + noise).clamp(0.05, 0.95);
            predictions.push(pred);
        }

        predictions
    }

    fn analyze_results(&mut self) {
        // Find best method per condition
        for result in &self.results.condition_results {
            let mut best_method = None;
            let mut best_brier = f64::INFINITY;

            for (method, scores) in &result.method_scores {
                let mean_brier: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
                if mean_brier < best_brier {
                    best_brier = mean_brier;
                    best_method = Some(*method);
                }
            }

            if let Some(method) = best_method {
                *self
                    .results
                    .best_method_by_condition
                    .entry(result.condition.clone())
                    .or_insert(method) = method;
            }
        }

        // Calculate wisdom of crowds ratio
        for result in &self.results.condition_results {
            let ind_mean = result.individual_distribution.mean;
            for (method, scores) in &result.method_scores {
                let agg_mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
                let ratio = if agg_mean > 0.0 {
                    ind_mean / agg_mean
                } else {
                    0.0
                };
                self.results.wisdom_ratio_by_method
                    .entry(*method)
                    .or_insert_with(Vec::new)
                    .push(ratio);
            }
        }

        // Expert vs aggregate comparison
        let mut aggregate_wins = 0;
        let mut total_comparisons = 0;

        for result in &self.results.condition_results {
            if let Some(expert_scores) = &result.expert_comparison {
                let expert_mean: f64 =
                    expert_scores.iter().sum::<f64>() / expert_scores.len() as f64;

                // Best aggregate score
                let best_agg = result
                    .method_scores
                    .values()
                    .map(|scores| scores.iter().sum::<f64>() / scores.len() as f64)
                    .fold(f64::INFINITY, f64::min);

                total_comparisons += 1;
                if best_agg < expert_mean {
                    aggregate_wins += 1;
                }
            }
        }

        if total_comparisons > 0 {
            self.results.aggregate_beats_expert_rate =
                aggregate_wins as f64 / total_comparisons as f64;
        }

        // Generate summary
        self.results.summary = self.generate_summary();
    }

    fn generate_summary(&self) -> AggregationSummary {
        // Identify conditions where aggregation fails
        let failure_conditions: Vec<StudyCondition> = self
            .results
            .condition_results
            .iter()
            .filter(|r| {
                // Aggregation "fails" if not better than individual mean
                let ind_mean = r.individual_distribution.mean;
                let best_agg = r
                    .method_scores
                    .values()
                    .map(|s| s.iter().sum::<f64>() / s.len() as f64)
                    .fold(f64::INFINITY, f64::min);
                best_agg >= ind_mean
            })
            .map(|r| r.condition.clone())
            .collect();

        // Identify minimum viable parameters
        let working_conditions: Vec<&ConditionResults> = self
            .results
            .condition_results
            .iter()
            .filter(|r| {
                let ind_mean = r.individual_distribution.mean;
                let best_agg = r
                    .method_scores
                    .values()
                    .map(|s| s.iter().sum::<f64>() / s.len() as f64)
                    .fold(f64::INFINITY, f64::min);
                best_agg < ind_mean * 0.9 // At least 10% improvement
            })
            .collect();

        let min_agents = working_conditions
            .iter()
            .map(|r| r.condition.agent_count)
            .min()
            .unwrap_or(0);

        let min_diversity = working_conditions
            .iter()
            .map(|r| r.condition.diversity as f64 / 1000.0)
            .fold(f64::INFINITY, f64::min);

        let min_independence = working_conditions
            .iter()
            .map(|r| r.condition.independence as f64 / 1000.0)
            .fold(f64::INFINITY, f64::min);

        AggregationSummary {
            overall_effectiveness: self.results.aggregate_beats_expert_rate,
            best_overall_method: self.find_best_overall_method(),
            failure_conditions,
            minimum_viable_agents: min_agents,
            minimum_viable_diversity: min_diversity,
            minimum_viable_independence: min_independence,
            question_type_findings: self.analyze_by_question_type(),
        }
    }

    fn find_best_overall_method(&self) -> AggregationMethod {
        let mut method_scores: HashMap<AggregationMethod, Vec<f64>> = HashMap::new();

        for result in &self.results.condition_results {
            for (method, scores) in &result.method_scores {
                let mean = scores.iter().sum::<f64>() / scores.len() as f64;
                method_scores
                    .entry(*method)
                    .or_insert_with(Vec::new)
                    .push(mean);
            }
        }

        method_scores
            .iter()
            .map(|(m, scores)| {
                let overall = scores.iter().sum::<f64>() / scores.len() as f64;
                (*m, overall)
            })
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(m, _)| m)
            .unwrap_or(AggregationMethod::SimpleMean)
    }

    fn analyze_by_question_type(&self) -> HashMap<QuestionType, QuestionTypeFinding> {
        let mut findings = HashMap::new();

        for qt in &self.config.question_types {
            let relevant_results: Vec<&ConditionResults> = self
                .results
                .condition_results
                .iter()
                .filter(|r| r.condition.question_type == *qt)
                .collect();

            if relevant_results.is_empty() {
                continue;
            }

            // Find best method for this question type
            let mut method_totals: HashMap<AggregationMethod, f64> = HashMap::new();
            let mut method_counts: HashMap<AggregationMethod, usize> = HashMap::new();

            for result in &relevant_results {
                for (method, scores) in &result.method_scores {
                    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
                    *method_totals.entry(*method).or_insert(0.0) += mean;
                    *method_counts.entry(*method).or_insert(0) += 1;
                }
            }

            let best_method = method_totals
                .iter()
                .map(|(m, total)| {
                    let count = method_counts.get(m).unwrap_or(&1);
                    (*m, total / *count as f64)
                })
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(m, _)| m)
                .unwrap_or(AggregationMethod::SimpleMean);

            // Calculate effectiveness rate
            let success_count = relevant_results
                .iter()
                .filter(|r| {
                    let ind = r.individual_distribution.mean;
                    let agg = r
                        .method_scores
                        .values()
                        .map(|s| s.iter().sum::<f64>() / s.len() as f64)
                        .fold(f64::INFINITY, f64::min);
                    agg < ind
                })
                .count();

            findings.insert(
                *qt,
                QuestionTypeFinding {
                    question_type: *qt,
                    best_method,
                    effectiveness_rate: success_count as f64 / relevant_results.len() as f64,
                    notes: self.generate_question_type_notes(*qt),
                },
            );
        }

        findings
    }

    fn generate_question_type_notes(&self, qt: QuestionType) -> String {
        match qt {
            QuestionType::Factual => {
                "Aggregation works well when individuals have partial information".to_string()
            }
            QuestionType::Prediction => {
                "Future predictions benefit most from diverse viewpoints".to_string()
            }
            QuestionType::Estimation => {
                "Extremized methods help counteract central tendency bias".to_string()
            }
            QuestionType::Subjective => {
                "Aggregation may not converge on subjective questions".to_string()
            }
        }
    }
}

// ============================================================================
// STUDY RESULTS
// ============================================================================

/// Results from aggregation effectiveness study
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationStudyResults {
    pub condition_results: Vec<ConditionResults>,
    pub best_method_by_condition: HashMap<StudyCondition, AggregationMethod>,
    pub wisdom_ratio_by_method: HashMap<AggregationMethod, Vec<f64>>,
    pub aggregate_beats_expert_rate: f64,
    pub summary: AggregationSummary,
}

impl AggregationStudyResults {
    pub fn new() -> Self {
        Self {
            condition_results: vec![],
            best_method_by_condition: HashMap::new(),
            wisdom_ratio_by_method: HashMap::new(),
            aggregate_beats_expert_rate: 0.0,
            summary: AggregationSummary::default(),
        }
    }
}

impl Default for AggregationStudyResults {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct StudyCondition {
    pub agent_count: usize,
    pub diversity: u64, // Stored as fixed point for Eq/Hash
    pub independence: u64,
    pub question_type: QuestionType,
}

impl StudyCondition {
    fn from_floats(agent_count: usize, diversity: f64, independence: f64, qt: QuestionType) -> Self {
        Self {
            agent_count,
            diversity: (diversity * 1000.0) as u64,
            independence: (independence * 1000.0) as u64,
            question_type: qt,
        }
    }
}

// Wrapper to allow float fields
impl From<(usize, f64, f64, QuestionType)> for StudyCondition {
    fn from((count, div, ind, qt): (usize, f64, f64, QuestionType)) -> Self {
        Self::from_floats(count, div, ind, qt)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionResults {
    pub condition: StudyCondition,
    pub method_scores: HashMap<AggregationMethod, Vec<f64>>,
    pub expert_comparison: Option<Vec<f64>>,
    pub individual_distribution: IndividualDistribution,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndividualDistribution {
    pub mean: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub p25: f64,
    pub p75: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregationSummary {
    pub overall_effectiveness: f64,
    pub best_overall_method: AggregationMethod,
    pub failure_conditions: Vec<StudyCondition>,
    pub minimum_viable_agents: usize,
    pub minimum_viable_diversity: f64,
    pub minimum_viable_independence: f64,
    pub question_type_findings: HashMap<QuestionType, QuestionTypeFinding>,
}

impl Default for AggregationMethod {
    fn default() -> Self {
        Self::SimpleMean
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionTypeFinding {
    pub question_type: QuestionType,
    pub best_method: AggregationMethod,
    pub effectiveness_rate: f64,
    pub notes: String,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_mean_aggregation() {
        let predictions = vec![(0.3, 1.0), (0.5, 1.0), (0.7, 1.0)];
        let result = aggregate_predictions(&predictions, AggregationMethod::SimpleMean);
        assert!((result - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_median_aggregation() {
        let predictions = vec![(0.1, 1.0), (0.5, 1.0), (0.9, 1.0)];
        let result = aggregate_predictions(&predictions, AggregationMethod::Median);
        assert!((result - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_weighted_aggregation() {
        let predictions = vec![(0.3, 3.0), (0.7, 1.0)];
        let result = aggregate_predictions(&predictions, AggregationMethod::BrierWeighted);
        // (0.3*3 + 0.7*1) / 4 = 1.6/4 = 0.4
        assert!((result - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_extremized_aggregation() {
        let predictions = vec![(0.6, 1.0), (0.7, 1.0)];
        let result = aggregate_predictions(&predictions, AggregationMethod::Extremized { factor: 2 });
        // Should be pushed away from 0.5
        assert!(result > 0.65);
    }

    #[test]
    fn test_study_initialization() {
        let config = AggregationStudyConfig {
            runs_per_condition: 2,
            agent_counts: vec![5],
            diversity_levels: vec![0.5],
            independence_levels: vec![0.5],
            question_types: vec![QuestionType::Prediction],
            ..Default::default()
        };

        let study = AggregationStudy::new(config);
        assert!(study.results.condition_results.is_empty());
    }
}
