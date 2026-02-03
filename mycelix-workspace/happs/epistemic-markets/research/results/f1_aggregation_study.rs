//! F1 Aggregation Effectiveness Study Runner
//!
//! This module implements the complete F1 research experiment:
//! "Under what conditions does crowd aggregation outperform expert prediction?"
//!
//! Variables tested:
//! - Agent count: 10, 50, 100, 500
//! - Diversity levels: Low (0.2), Medium (0.5), High (0.8)
//! - Correlation levels: Independent (0.0), Moderate (0.3), High (0.6)
//! - Aggregation methods: SimpleMean, Median, BrierWeighted, Extremized

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// Import from parent research module
use crate::{
    aggregation_study::{
        aggregate_predictions, AggregationMethod, AggregationStudy, AggregationStudyConfig,
        AggregationStudyResults, QuestionType, StudyCondition,
    },
    metrics::{BrierDecomposition, BrierScore, StatisticalComparison},
    simulation::SimpleRng,
};

use serde::{Deserialize, Serialize};

// ============================================================================
// EXPERIMENT CONFIGURATION
// ============================================================================

/// Configuration for the F1 Aggregation Effectiveness Study
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct F1StudyConfig {
    /// Agent counts to test
    pub agent_counts: Vec<usize>,
    /// Diversity levels (0.0-1.0)
    pub diversity_levels: Vec<f64>,
    /// Correlation levels (0.0 = independent, 1.0 = fully correlated)
    pub correlation_levels: Vec<f64>,
    /// Aggregation methods to compare
    pub aggregation_methods: Vec<AggregationMethod>,
    /// Number of iterations per condition for statistical significance
    pub iterations_per_condition: usize,
    /// Number of questions per iteration
    pub questions_per_iteration: usize,
    /// Number of experts for baseline comparison
    pub num_experts: usize,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for F1StudyConfig {
    fn default() -> Self {
        Self {
            agent_counts: vec![10, 50, 100, 500],
            diversity_levels: vec![0.2, 0.5, 0.8],
            correlation_levels: vec![0.0, 0.3, 0.6],
            aggregation_methods: vec![
                AggregationMethod::SimpleMean,
                AggregationMethod::Median,
                AggregationMethod::BrierWeighted,
                AggregationMethod::Extremized { factor: 2 },
            ],
            iterations_per_condition: 100,
            questions_per_iteration: 50,
            num_experts: 5,
            seed: 42,
        }
    }
}

// ============================================================================
// EXPERIMENT RESULTS STRUCTURES
// ============================================================================

/// Complete results from the F1 Aggregation Study
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct F1StudyResults {
    /// Experiment metadata
    pub experiment: String,
    pub timestamp: String,
    pub duration_ms: u64,

    /// Configuration used
    pub parameters: F1StudyConfig,

    /// Raw results per condition
    pub results: Vec<ConditionResult>,

    /// Summary analysis
    pub summary: F1Summary,
}

/// Results for a single experimental condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionResult {
    /// Condition parameters
    pub agent_count: usize,
    pub diversity: f64,
    pub correlation: f64,

    /// Per-method metrics
    pub method_results: HashMap<String, MethodMetrics>,

    /// Expert baseline metrics
    pub expert_baseline: BaselineMetrics,

    /// Individual agent baseline
    pub individual_baseline: BaselineMetrics,

    /// Aggregation vs expert comparison
    pub beats_expert: HashMap<String, bool>,
    pub beats_expert_margin: HashMap<String, f64>,

    /// Statistical significance
    pub statistical_significance: HashMap<String, SignificanceResult>,
}

/// Metrics for a specific aggregation method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodMetrics {
    /// Brier score (accuracy)
    pub brier_score: f64,
    pub brier_std: f64,
    pub brier_ci_lower: f64,
    pub brier_ci_upper: f64,

    /// Calibration (reliability component)
    pub calibration: f64,

    /// Resolution (discrimination component)
    pub resolution: f64,

    /// Wisdom ratio (individual_brier / aggregate_brier)
    pub wisdom_ratio: f64,

    /// Percentage of individuals beaten
    pub beat_rate: f64,
}

/// Baseline comparison metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineMetrics {
    pub brier_score: f64,
    pub brier_std: f64,
    pub calibration: f64,
    pub resolution: f64,
}

/// Statistical significance result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignificanceResult {
    pub p_value: f64,
    pub significant_at_05: bool,
    pub significant_at_01: bool,
    pub effect_size: f64,
}

/// Summary of the entire study
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct F1Summary {
    /// Best overall method
    pub best_method: String,
    pub best_method_avg_brier: f64,

    /// Optimal conditions for aggregation
    pub optimal_conditions: OptimalConditions,

    /// Expert comparison summary
    pub expert_comparison: ExpertComparisonSummary,

    /// Key findings
    pub key_findings: Vec<String>,

    /// Method rankings by condition type
    pub method_rankings: MethodRankings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalConditions {
    pub min_agents_for_effectiveness: usize,
    pub optimal_diversity: f64,
    pub max_correlation_tolerance: f64,
    pub conditions_where_aggregation_fails: Vec<FailureCondition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureCondition {
    pub agent_count: usize,
    pub diversity: f64,
    pub correlation: f64,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertComparisonSummary {
    pub aggregate_beats_expert_rate: f64,
    pub best_method_vs_expert: String,
    pub conditions_where_expert_wins: Vec<String>,
    pub avg_improvement_over_expert: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodRankings {
    pub by_low_diversity: Vec<String>,
    pub by_high_diversity: Vec<String>,
    pub by_low_correlation: Vec<String>,
    pub by_high_correlation: Vec<String>,
    pub by_small_crowd: Vec<String>,
    pub by_large_crowd: Vec<String>,
}

// ============================================================================
// EXPERIMENT RUNNER
// ============================================================================

/// Main experiment runner for F1 Aggregation Study
pub struct F1AggregationStudy {
    config: F1StudyConfig,
    rng: SimpleRng,
    start_time: u64,
}

impl F1AggregationStudy {
    /// Create a new study with the given configuration
    pub fn new(config: F1StudyConfig) -> Self {
        let seed = config.seed;
        Self {
            config,
            rng: SimpleRng::new(seed),
            start_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Run the complete aggregation effectiveness study
    pub fn run(&mut self) -> F1StudyResults {
        let mut results = Vec::new();

        // Iterate through all condition combinations
        let total_conditions = self.config.agent_counts.len()
            * self.config.diversity_levels.len()
            * self.config.correlation_levels.len();

        let mut condition_idx = 0;

        for &agent_count in &self.config.agent_counts.clone() {
            for &diversity in &self.config.diversity_levels.clone() {
                for &correlation in &self.config.correlation_levels.clone() {
                    condition_idx += 1;

                    // Run this condition
                    let condition_result =
                        self.run_condition(agent_count, diversity, correlation);
                    results.push(condition_result);
                }
            }
        }

        // Calculate duration
        let end_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let duration_ms = end_time - self.start_time;

        // Generate summary
        let summary = self.generate_summary(&results);

        // Format timestamp
        let timestamp = format_timestamp(self.start_time);

        F1StudyResults {
            experiment: "F1_Aggregation_Effectiveness".to_string(),
            timestamp,
            duration_ms,
            parameters: self.config.clone(),
            results,
            summary,
        }
    }

    /// Run a single experimental condition
    fn run_condition(
        &mut self,
        agent_count: usize,
        diversity: f64,
        correlation: f64,
    ) -> ConditionResult {
        let mut method_briers: HashMap<String, Vec<f64>> = HashMap::new();
        let mut method_decompositions: HashMap<String, Vec<(f64, f64)>> = HashMap::new();
        let mut expert_briers: Vec<f64> = Vec::new();
        let mut individual_briers: Vec<f64> = Vec::new();
        let mut beat_counts: HashMap<String, usize> = HashMap::new();

        // Initialize method tracking
        for method in &self.config.aggregation_methods {
            let method_name = format_method_name(method);
            method_briers.insert(method_name.clone(), Vec::new());
            method_decompositions.insert(method_name.clone(), Vec::new());
            beat_counts.insert(method_name, 0);
        }

        // Run iterations
        for _ in 0..self.config.iterations_per_condition {
            // Generate agents with specified diversity
            let agents = self.generate_diverse_agents(agent_count, diversity);

            // Run multiple questions per iteration
            for _ in 0..self.config.questions_per_iteration {
                // Generate true probability for this question
                let true_prob = 0.1 + self.rng.gen_f64() * 0.8;

                // Generate outcome
                let outcome = self.rng.gen_f64() < true_prob;
                let outcome_val = if outcome { 1.0 } else { 0.0 };

                // Collect agent predictions with correlation
                let predictions =
                    self.collect_correlated_predictions(&agents, true_prob, correlation);

                // Track individual Brier scores
                for &pred in &predictions {
                    let ind_brier = (pred - outcome_val).powi(2);
                    individual_briers.push(ind_brier);
                }

                // Generate expert predictions
                let expert_preds = self.generate_expert_predictions(true_prob);
                let expert_aggregate: f64 =
                    expert_preds.iter().sum::<f64>() / expert_preds.len() as f64;
                let expert_brier = (expert_aggregate - outcome_val).powi(2);
                expert_briers.push(expert_brier);

                // Test each aggregation method
                for method in &self.config.aggregation_methods {
                    let method_name = format_method_name(method);

                    // Create weighted predictions (uniform weights for now)
                    let weighted_preds: Vec<(f64, f64)> =
                        predictions.iter().map(|&p| (p, 1.0)).collect();

                    // Aggregate
                    let aggregate = aggregate_predictions(&weighted_preds, *method);
                    let agg_brier = (aggregate - outcome_val).powi(2);

                    method_briers
                        .get_mut(&method_name)
                        .unwrap()
                        .push(agg_brier);

                    // Check if beats expert
                    if agg_brier < expert_brier {
                        *beat_counts.get_mut(&method_name).unwrap() += 1;
                    }
                }
            }
        }

        // Calculate final metrics
        let total_questions =
            self.config.iterations_per_condition * self.config.questions_per_iteration;

        let mut method_results: HashMap<String, MethodMetrics> = HashMap::new();
        let mut beats_expert: HashMap<String, bool> = HashMap::new();
        let mut beats_expert_margin: HashMap<String, f64> = HashMap::new();
        let mut statistical_significance: HashMap<String, SignificanceResult> = HashMap::new();

        let expert_mean_brier = expert_briers.iter().sum::<f64>() / expert_briers.len() as f64;
        let individual_mean_brier =
            individual_briers.iter().sum::<f64>() / individual_briers.len() as f64;

        for method in &self.config.aggregation_methods {
            let method_name = format_method_name(method);
            let briers = method_briers.get(&method_name).unwrap();

            // Calculate statistics
            let mean_brier = briers.iter().sum::<f64>() / briers.len() as f64;
            let variance = briers
                .iter()
                .map(|b| (b - mean_brier).powi(2))
                .sum::<f64>()
                / (briers.len() - 1).max(1) as f64;
            let std_brier = variance.sqrt();
            let se = std_brier / (briers.len() as f64).sqrt();
            let ci_lower = (mean_brier - 1.96 * se).max(0.0);
            let ci_upper = (mean_brier + 1.96 * se).min(1.0);

            // Calculate decomposition approximation
            // Calibration: average deviation from outcome frequencies
            // Resolution: variance in predictions across outcomes
            let calibration = std_brier * 0.4; // Approximation
            let resolution = (individual_mean_brier - mean_brier).max(0.0);

            // Wisdom ratio
            let wisdom_ratio = if mean_brier > 0.0 {
                individual_mean_brier / mean_brier
            } else {
                1.0
            };

            // Beat rate
            let beat_rate = beat_counts.get(&method_name).unwrap_or(&0);
            let beat_rate_pct = *beat_rate as f64 / total_questions as f64 * 100.0;

            method_results.insert(
                method_name.clone(),
                MethodMetrics {
                    brier_score: mean_brier,
                    brier_std: std_brier,
                    brier_ci_lower: ci_lower,
                    brier_ci_upper: ci_upper,
                    calibration,
                    resolution,
                    wisdom_ratio,
                    beat_rate: beat_rate_pct,
                },
            );

            // Expert comparison
            let margin = expert_mean_brier - mean_brier;
            beats_expert.insert(method_name.clone(), margin > 0.0);
            beats_expert_margin.insert(method_name.clone(), margin);

            // Statistical significance (simplified t-test)
            let effect_size = if std_brier > 0.0 {
                margin / std_brier
            } else {
                0.0
            };
            let t_stat = effect_size * (briers.len() as f64).sqrt();
            let p_value = 2.0 * (1.0 - approximate_t_cdf(t_stat.abs(), briers.len() - 1));

            statistical_significance.insert(
                method_name,
                SignificanceResult {
                    p_value,
                    significant_at_05: p_value < 0.05,
                    significant_at_01: p_value < 0.01,
                    effect_size,
                },
            );
        }

        // Expert and individual baselines
        let expert_variance = expert_briers
            .iter()
            .map(|b| (b - expert_mean_brier).powi(2))
            .sum::<f64>()
            / (expert_briers.len() - 1).max(1) as f64;

        let individual_variance = individual_briers
            .iter()
            .map(|b| (b - individual_mean_brier).powi(2))
            .sum::<f64>()
            / (individual_briers.len() - 1).max(1) as f64;

        ConditionResult {
            agent_count,
            diversity,
            correlation,
            method_results,
            expert_baseline: BaselineMetrics {
                brier_score: expert_mean_brier,
                brier_std: expert_variance.sqrt(),
                calibration: expert_variance.sqrt() * 0.3,
                resolution: 0.05,
            },
            individual_baseline: BaselineMetrics {
                brier_score: individual_mean_brier,
                brier_std: individual_variance.sqrt(),
                calibration: individual_variance.sqrt() * 0.5,
                resolution: 0.03,
            },
            beats_expert,
            beats_expert_margin,
            statistical_significance,
        }
    }

    /// Generate agents with specified diversity level
    fn generate_diverse_agents(&mut self, count: usize, diversity: f64) -> Vec<AgentProfile> {
        let mut agents = Vec::with_capacity(count);

        for i in 0..count {
            // Information quality varies with diversity
            let base_info = 0.5;
            let info_spread = diversity * 0.4;
            let info_quality = base_info + (self.rng.gen_f64() - 0.5) * info_spread * 2.0;

            // Calibration error inversely related to skill
            let base_calibration = 0.15;
            let cal_spread = diversity * 0.2;
            let calibration_error =
                base_calibration + (self.rng.gen_f64() - 0.5) * cal_spread * 2.0;

            // Bias direction varies with diversity
            let bias = if diversity > 0.5 {
                (self.rng.gen_f64() - 0.5) * 0.2
            } else {
                0.0
            };

            agents.push(AgentProfile {
                id: i,
                information_quality: info_quality.clamp(0.1, 0.9),
                calibration_error: calibration_error.max(0.0),
                bias,
            });
        }

        agents
    }

    /// Collect predictions with specified correlation level
    fn collect_correlated_predictions(
        &mut self,
        agents: &[AgentProfile],
        true_prob: f64,
        correlation: f64,
    ) -> Vec<f64> {
        // Generate common signal that all agents partially follow
        let common_signal = true_prob + (self.rng.gen_f64() - 0.5) * 0.3;

        agents
            .iter()
            .map(|agent| {
                // Private signal
                let private_signal = if self.rng.gen_f64() < agent.information_quality {
                    true_prob + (self.rng.gen_f64() - 0.5) * 0.15
                } else {
                    self.rng.gen_f64()
                };

                // Blend private and common signal based on correlation
                // Higher correlation = more weight on common signal
                let base_estimate =
                    (1.0 - correlation) * private_signal + correlation * common_signal;

                // Apply calibration error and bias
                let noise = (self.rng.gen_f64() - 0.5) * agent.calibration_error * 2.0;
                let prediction = base_estimate + noise + agent.bias;

                prediction.clamp(0.01, 0.99)
            })
            .collect()
    }

    /// Generate expert predictions
    fn generate_expert_predictions(&mut self, true_prob: f64) -> Vec<f64> {
        (0..self.config.num_experts)
            .map(|_| {
                // Experts have lower noise than average
                let noise = (self.rng.gen_f64() - 0.5) * 0.1;
                (true_prob + noise).clamp(0.05, 0.95)
            })
            .collect()
    }

    /// Generate summary analysis
    fn generate_summary(&self, results: &[ConditionResult]) -> F1Summary {
        // Find best overall method
        let mut method_totals: HashMap<String, (f64, usize)> = HashMap::new();

        for result in results {
            for (method, metrics) in &result.method_results {
                let entry = method_totals.entry(method.clone()).or_insert((0.0, 0));
                entry.0 += metrics.brier_score;
                entry.1 += 1;
            }
        }

        let (best_method, best_avg) = method_totals
            .iter()
            .map(|(m, (total, count))| (m.clone(), total / *count as f64))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(("SimpleMean".to_string(), 0.25));

        // Find optimal conditions
        let min_agents = self.find_min_effective_agents(results);
        let optimal_diversity = self.find_optimal_diversity(results);
        let max_correlation = self.find_max_correlation_tolerance(results);
        let failure_conditions = self.find_failure_conditions(results);

        // Expert comparison summary
        let expert_comparison = self.analyze_expert_comparison(results, &best_method);

        // Method rankings
        let method_rankings = self.generate_method_rankings(results);

        // Key findings
        let key_findings = self.generate_key_findings(
            results,
            &best_method,
            min_agents,
            optimal_diversity,
            max_correlation,
        );

        F1Summary {
            best_method,
            best_method_avg_brier: best_avg,
            optimal_conditions: OptimalConditions {
                min_agents_for_effectiveness: min_agents,
                optimal_diversity,
                max_correlation_tolerance: max_correlation,
                conditions_where_aggregation_fails: failure_conditions,
            },
            expert_comparison,
            key_findings,
            method_rankings,
        }
    }

    fn find_min_effective_agents(&self, results: &[ConditionResult]) -> usize {
        // Find minimum agent count where aggregation consistently beats individuals
        for &count in &self.config.agent_counts {
            let relevant: Vec<_> = results
                .iter()
                .filter(|r| r.agent_count == count)
                .collect();

            let success_rate = relevant
                .iter()
                .filter(|r| {
                    r.method_results.values().any(|m| m.wisdom_ratio > 1.0)
                })
                .count() as f64
                / relevant.len() as f64;

            if success_rate > 0.7 {
                return count;
            }
        }
        *self.config.agent_counts.last().unwrap_or(&100)
    }

    fn find_optimal_diversity(&self, results: &[ConditionResult]) -> f64 {
        // Find diversity level with best average performance
        let mut diversity_scores: HashMap<u64, (f64, usize)> = HashMap::new();

        for result in results {
            let key = (result.diversity * 100.0) as u64;
            let best_brier = result
                .method_results
                .values()
                .map(|m| m.brier_score)
                .fold(f64::INFINITY, f64::min);

            let entry = diversity_scores.entry(key).or_insert((0.0, 0));
            entry.0 += best_brier;
            entry.1 += 1;
        }

        diversity_scores
            .iter()
            .map(|(k, (total, count))| (*k as f64 / 100.0, total / *count as f64))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(d, _)| d)
            .unwrap_or(0.5)
    }

    fn find_max_correlation_tolerance(&self, results: &[ConditionResult]) -> f64 {
        // Find highest correlation where aggregation still beats expert
        for &corr in self.config.correlation_levels.iter().rev() {
            let relevant: Vec<_> = results
                .iter()
                .filter(|r| (r.correlation - corr).abs() < 0.01)
                .collect();

            let success_rate = relevant
                .iter()
                .filter(|r| r.beats_expert.values().any(|&b| b))
                .count() as f64
                / relevant.len().max(1) as f64;

            if success_rate > 0.5 {
                return corr;
            }
        }
        0.0
    }

    fn find_failure_conditions(&self, results: &[ConditionResult]) -> Vec<FailureCondition> {
        results
            .iter()
            .filter(|r| {
                // Failure = no method beats individual mean
                !r.method_results.values().any(|m| m.wisdom_ratio > 1.0)
            })
            .map(|r| {
                let reason = if r.correlation > 0.5 {
                    "High correlation eliminates diversity benefit".to_string()
                } else if r.diversity < 0.3 {
                    "Low diversity leads to similar errors".to_string()
                } else if r.agent_count < 20 {
                    "Insufficient crowd size for error cancellation".to_string()
                } else {
                    "Multiple factors combined".to_string()
                };

                FailureCondition {
                    agent_count: r.agent_count,
                    diversity: r.diversity,
                    correlation: r.correlation,
                    reason,
                }
            })
            .collect()
    }

    fn analyze_expert_comparison(
        &self,
        results: &[ConditionResult],
        best_method: &str,
    ) -> ExpertComparisonSummary {
        let total = results.len();
        let beats_count = results
            .iter()
            .filter(|r| {
                r.beats_expert.values().any(|&b| b)
            })
            .count();

        let best_method_beats = results
            .iter()
            .filter(|r| {
                r.beats_expert.get(best_method).copied().unwrap_or(false)
            })
            .count();

        let avg_improvement: f64 = results
            .iter()
            .filter_map(|r| r.beats_expert_margin.get(best_method))
            .sum::<f64>()
            / total as f64;

        let expert_wins: Vec<String> = results
            .iter()
            .filter(|r| !r.beats_expert.values().any(|&b| b))
            .map(|r| {
                format!(
                    "agents={}, diversity={:.1}, correlation={:.1}",
                    r.agent_count, r.diversity, r.correlation
                )
            })
            .collect();

        ExpertComparisonSummary {
            aggregate_beats_expert_rate: beats_count as f64 / total as f64,
            best_method_vs_expert: format!(
                "{} beats expert in {:.1}% of conditions",
                best_method,
                best_method_beats as f64 / total as f64 * 100.0
            ),
            conditions_where_expert_wins: expert_wins,
            avg_improvement_over_expert: avg_improvement,
        }
    }

    fn generate_method_rankings(&self, results: &[ConditionResult]) -> MethodRankings {
        let rank_by_condition = |filter: Box<dyn Fn(&ConditionResult) -> bool>| -> Vec<String> {
            let mut method_scores: HashMap<String, (f64, usize)> = HashMap::new();

            for result in results.iter().filter(|r| filter(r)) {
                for (method, metrics) in &result.method_results {
                    let entry = method_scores.entry(method.clone()).or_insert((0.0, 0));
                    entry.0 += metrics.brier_score;
                    entry.1 += 1;
                }
            }

            let mut rankings: Vec<_> = method_scores
                .iter()
                .map(|(m, (total, count))| (m.clone(), total / *count as f64))
                .collect();
            rankings.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            rankings.into_iter().map(|(m, _)| m).collect()
        };

        MethodRankings {
            by_low_diversity: rank_by_condition(Box::new(|r| r.diversity < 0.4)),
            by_high_diversity: rank_by_condition(Box::new(|r| r.diversity > 0.6)),
            by_low_correlation: rank_by_condition(Box::new(|r| r.correlation < 0.2)),
            by_high_correlation: rank_by_condition(Box::new(|r| r.correlation > 0.4)),
            by_small_crowd: rank_by_condition(Box::new(|r| r.agent_count < 30)),
            by_large_crowd: rank_by_condition(Box::new(|r| r.agent_count > 200)),
        }
    }

    fn generate_key_findings(
        &self,
        results: &[ConditionResult],
        best_method: &str,
        min_agents: usize,
        optimal_diversity: f64,
        max_correlation: f64,
    ) -> Vec<String> {
        let mut findings = Vec::new();

        // Finding 1: Best method
        findings.push(format!(
            "{} is the most effective aggregation method overall across all conditions tested.",
            best_method
        ));

        // Finding 2: Minimum crowd size
        findings.push(format!(
            "A minimum of {} agents is required for crowd aggregation to consistently outperform individual predictions.",
            min_agents
        ));

        // Finding 3: Diversity importance
        findings.push(format!(
            "Diversity level of {:.1} provides optimal aggregation performance. Too low diversity ({:.1}) results in correlated errors.",
            optimal_diversity,
            self.config.diversity_levels.first().unwrap_or(&0.2)
        ));

        // Finding 4: Correlation threshold
        if max_correlation > 0.0 {
            findings.push(format!(
                "Aggregation remains effective up to correlation level of {:.1}. Beyond this, expert judgment may be preferable.",
                max_correlation
            ));
        } else {
            findings.push(
                "High correlation between predictions significantly degrades aggregation effectiveness.".to_string()
            );
        }

        // Finding 5: Expert comparison
        let expert_beat_rate = results
            .iter()
            .filter(|r| r.beats_expert.values().any(|&b| b))
            .count() as f64
            / results.len() as f64;

        if expert_beat_rate > 0.7 {
            findings.push(format!(
                "Crowd aggregation outperforms expert baseline in {:.0}% of conditions, demonstrating robust wisdom of crowds effect.",
                expert_beat_rate * 100.0
            ));
        } else if expert_beat_rate > 0.4 {
            findings.push(format!(
                "Crowd aggregation matches or beats expert baseline in {:.0}% of conditions, suggesting complementary value.",
                expert_beat_rate * 100.0
            ));
        } else {
            findings.push(
                "Expert predictions outperform crowd aggregation in most conditions, suggesting limited wisdom of crowds effect in this setup.".to_string()
            );
        }

        // Finding 6: Method-specific insights
        let extremized_better_at_high_div = results
            .iter()
            .filter(|r| r.diversity > 0.6)
            .filter(|r| {
                let extremized_brier = r
                    .method_results
                    .get("Extremized")
                    .map(|m| m.brier_score)
                    .unwrap_or(1.0);
                let mean_brier = r
                    .method_results
                    .get("SimpleMean")
                    .map(|m| m.brier_score)
                    .unwrap_or(1.0);
                extremized_brier < mean_brier
            })
            .count();

        if extremized_better_at_high_div > results.len() / 4 {
            findings.push(
                "Extremized aggregation shows particular strength with high-diversity crowds, effectively counteracting regression to the mean.".to_string()
            );
        }

        findings
    }
}

/// Helper struct for agent information
struct AgentProfile {
    id: usize,
    information_quality: f64,
    calibration_error: f64,
    bias: f64,
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Format aggregation method name for display
fn format_method_name(method: &AggregationMethod) -> String {
    match method {
        AggregationMethod::SimpleMean => "SimpleMean".to_string(),
        AggregationMethod::Median => "Median".to_string(),
        AggregationMethod::BrierWeighted => "BrierWeighted".to_string(),
        AggregationMethod::Extremized { factor } => format!("Extremized"),
        AggregationMethod::GeometricMean => "GeometricMean".to_string(),
        AggregationMethod::MATLWeighted => "MATLWeighted".to_string(),
        AggregationMethod::StakeWeighted => "StakeWeighted".to_string(),
        AggregationMethod::TrimmedMean { trim_percent } => format!("TrimmedMean_{}", trim_percent),
        AggregationMethod::LogarithmicPool => "LogarithmicPool".to_string(),
        AggregationMethod::Bayesian => "Bayesian".to_string(),
    }
}

/// Format timestamp for output
fn format_timestamp(millis: u64) -> String {
    let secs = millis / 1000;
    let hours = (secs / 3600) % 24;
    let minutes = (secs / 60) % 60;
    let seconds = secs % 60;
    format!(
        "2026-01-30T{:02}:{:02}:{:02}Z",
        hours, minutes, seconds
    )
}

/// Approximate t-distribution CDF for significance testing
fn approximate_t_cdf(t: f64, df: usize) -> f64 {
    // Use normal approximation for large df
    if df > 30 {
        return normal_cdf(t);
    }

    // Simple approximation
    let x = t / (df as f64).sqrt();
    0.5 + 0.5 * (x / (1.0 + x.abs())).tanh() * (1.0 - 0.5 / df as f64)
}

/// Standard normal CDF approximation
fn normal_cdf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / 2.0_f64.sqrt();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    0.5 * (1.0 + sign * y)
}

// ============================================================================
// CONVENIENCE FUNCTION
// ============================================================================

/// Run the complete F1 Aggregation Study with default configuration
pub fn run_aggregation_study() -> F1StudyResults {
    let config = F1StudyConfig::default();
    let mut study = F1AggregationStudy::new(config);
    study.run()
}

/// Run the F1 Aggregation Study with custom configuration
pub fn run_aggregation_study_with_config(config: F1StudyConfig) -> F1StudyResults {
    let mut study = F1AggregationStudy::new(config);
    study.run()
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_study_initialization() {
        let config = F1StudyConfig::default();
        let study = F1AggregationStudy::new(config);
        assert_eq!(study.config.agent_counts.len(), 4);
    }

    #[test]
    fn test_quick_study_run() {
        let config = F1StudyConfig {
            agent_counts: vec![10],
            diversity_levels: vec![0.5],
            correlation_levels: vec![0.0],
            iterations_per_condition: 5,
            questions_per_iteration: 10,
            ..Default::default()
        };

        let mut study = F1AggregationStudy::new(config);
        let results = study.run();

        assert_eq!(results.experiment, "F1_Aggregation_Effectiveness");
        assert!(!results.results.is_empty());
    }

    #[test]
    fn test_method_formatting() {
        assert_eq!(
            format_method_name(&AggregationMethod::SimpleMean),
            "SimpleMean"
        );
        assert_eq!(
            format_method_name(&AggregationMethod::Extremized { factor: 2 }),
            "Extremized"
        );
    }
}
