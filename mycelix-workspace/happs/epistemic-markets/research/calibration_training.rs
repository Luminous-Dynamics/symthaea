// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! F2: Calibration Training Research
//!
//! Addresses the question: Can calibration be taught?
//!
//! Research areas:
//! - Training method effectiveness
//! - Cross-domain transfer
//! - Persistence of calibration improvement
//! - Individual differences in learnability

use crate::metrics::{
    BrierScore, ExpectedCalibrationError, TimeSeriesMetrics, TransferAnalysis,
    PredictorDomainPerformance,
};
use crate::simulation::{SimpleRng, SimulatedAgent, AgentType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// TRAINING METHODS
// ============================================================================

/// Different calibration training methods to test
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TrainingMethod {
    /// No training (control group)
    Control,
    /// Immediate Brier score feedback after each prediction
    ImmediateFeedback,
    /// Calibration graph shown periodically
    CalibrationGraphs { frequency: usize },
    /// Base rate training (always start with base rates)
    BaseRateAnchor,
    /// Outside view prompting (consider reference class)
    OutsideView,
    /// Explicit consideration of alternatives
    ConsiderAlternatives,
    /// Fermi estimation training
    FermiEstimation,
    /// Decomposition training (break into sub-questions)
    Decomposition,
    /// Adversarial collaboration (argue both sides)
    AdversarialCollab,
    /// Historical accuracy review
    HistoricalReview { lookback_count: usize },
    /// Overconfidence warnings
    OverconfidenceWarning { threshold: u32 }, // threshold * 0.01
    /// Combined training protocol
    Combined(Vec<TrainingMethod>),
}

impl Default for TrainingMethod {
    fn default() -> Self {
        Self::Control
    }
}

// ============================================================================
// TRAINING SESSION
// ============================================================================

/// Configuration for a training experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExperimentConfig {
    /// Training methods to compare
    pub methods: Vec<TrainingMethod>,
    /// Number of participants per group
    pub participants_per_group: usize,
    /// Training session duration (number of predictions)
    pub training_duration: usize,
    /// Post-training assessment duration
    pub assessment_duration: usize,
    /// Follow-up assessments (timestamps)
    pub followup_assessments: Vec<usize>,
    /// Domains to test
    pub domains: Vec<String>,
    /// Include cross-domain transfer test
    pub test_transfer: bool,
    /// Random seed
    pub seed: u64,
}

impl Default for TrainingExperimentConfig {
    fn default() -> Self {
        Self {
            methods: vec![
                TrainingMethod::Control,
                TrainingMethod::ImmediateFeedback,
                TrainingMethod::CalibrationGraphs { frequency: 10 },
                TrainingMethod::BaseRateAnchor,
                TrainingMethod::OutsideView,
            ],
            participants_per_group: 50,
            training_duration: 100,
            assessment_duration: 50,
            followup_assessments: vec![200, 500, 1000],
            domains: vec![
                "politics".to_string(),
                "technology".to_string(),
                "sports".to_string(),
                "science".to_string(),
            ],
            test_transfer: true,
            seed: 42,
        }
    }
}

/// A single training participant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParticipant {
    pub id: String,
    /// Assigned training method
    pub training_method: TrainingMethod,
    /// Initial calibration error
    pub initial_ece: f64,
    /// Initial Brier score
    pub initial_brier: f64,
    /// Learning rate parameter (individual difference)
    pub learning_rate: f64,
    /// Current calibration state
    pub current_ece: f64,
    pub current_brier: f64,
    /// Calibration history over time
    pub ece_history: Vec<TimestampedCalibration>,
    /// Brier score history
    pub brier_history: Vec<TimestampedCalibration>,
    /// Domain-specific calibration
    pub domain_calibration: HashMap<String, f64>,
    /// Training predictions made
    pub training_predictions: Vec<TrainingPrediction>,
    /// Assessment predictions
    pub assessment_predictions: Vec<TrainingPrediction>,
    /// Has completed training
    pub training_complete: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedCalibration {
    pub step: usize,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingPrediction {
    pub step: usize,
    pub domain: String,
    pub predicted: f64,
    pub outcome: bool,
    pub brier: f64,
    /// Feedback received (if any)
    pub feedback: Option<TrainingFeedback>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingFeedback {
    pub feedback_type: FeedbackType,
    pub message: String,
    pub calibration_adjustment: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackType {
    BrierScore(f64),
    CalibrationGraph,
    BaseRateReminder(f64),
    OutsideViewPrompt(String),
    OverconfidenceWarning,
    Encouragement,
}

impl TrainingParticipant {
    pub fn new(id: String, method: TrainingMethod, initial_ece: f64, learning_rate: f64) -> Self {
        // Initial Brier roughly correlates with ECE
        let initial_brier = 0.15 + initial_ece * 0.3;

        Self {
            id,
            training_method: method,
            initial_ece,
            initial_brier,
            learning_rate,
            current_ece: initial_ece,
            current_brier: initial_brier,
            ece_history: vec![TimestampedCalibration {
                step: 0,
                value: initial_ece,
            }],
            brier_history: vec![TimestampedCalibration {
                step: 0,
                value: initial_brier,
            }],
            domain_calibration: HashMap::new(),
            training_predictions: vec![],
            assessment_predictions: vec![],
            training_complete: false,
        }
    }

    /// Process a training prediction
    pub fn process_prediction(
        &mut self,
        step: usize,
        domain: &str,
        true_prob: f64,
        rng: &mut SimpleRng,
    ) -> TrainingPrediction {
        // Generate prediction based on current calibration
        let noise = (rng.next_u64() as f64 / u64::MAX as f64 - 0.5) * self.current_ece * 2.0;
        let predicted = (true_prob + noise).clamp(0.01, 0.99);

        // Determine outcome
        let outcome = (rng.next_u64() as f64 / u64::MAX as f64) < true_prob;
        let brier = (predicted - if outcome { 1.0 } else { 0.0 }).powi(2);

        // Generate feedback based on training method
        let feedback = self.generate_feedback(step, predicted, outcome, brier, domain, rng);

        // Apply learning effect
        if feedback.is_some() {
            self.apply_learning(step, brier, &feedback);
        }

        let prediction = TrainingPrediction {
            step,
            domain: domain.to_string(),
            predicted,
            outcome,
            brier,
            feedback,
        };

        self.training_predictions.push(prediction.clone());
        prediction
    }

    fn generate_feedback(
        &self,
        step: usize,
        predicted: f64,
        outcome: bool,
        brier: f64,
        domain: &str,
        rng: &mut SimpleRng,
    ) -> Option<TrainingFeedback> {
        match &self.training_method {
            TrainingMethod::Control => None,

            TrainingMethod::ImmediateFeedback => Some(TrainingFeedback {
                feedback_type: FeedbackType::BrierScore(brier),
                message: format!(
                    "Your Brier score for this prediction: {:.3}",
                    brier
                ),
                calibration_adjustment: -0.002,
            }),

            TrainingMethod::CalibrationGraphs { frequency } => {
                if step % frequency == 0 && step > 0 {
                    Some(TrainingFeedback {
                        feedback_type: FeedbackType::CalibrationGraph,
                        message: "Review your calibration curve".to_string(),
                        calibration_adjustment: -0.005,
                    })
                } else {
                    None
                }
            }

            TrainingMethod::BaseRateAnchor => {
                // Remind of base rate for domain
                let base_rate = self.get_domain_base_rate(domain);
                Some(TrainingFeedback {
                    feedback_type: FeedbackType::BaseRateReminder(base_rate),
                    message: format!(
                        "Historical base rate for {} is {:.0}%",
                        domain,
                        base_rate * 100.0
                    ),
                    calibration_adjustment: -0.003,
                })
            }

            TrainingMethod::OutsideView => Some(TrainingFeedback {
                feedback_type: FeedbackType::OutsideViewPrompt(
                    "Consider similar historical cases".to_string(),
                ),
                message: "What do similar cases suggest?".to_string(),
                calibration_adjustment: -0.004,
            }),

            TrainingMethod::OverconfidenceWarning { threshold } => {
                let thresh = *threshold as f64 / 100.0;
                if predicted > (1.0 - thresh) || predicted < thresh {
                    Some(TrainingFeedback {
                        feedback_type: FeedbackType::OverconfidenceWarning,
                        message: "Warning: Extreme probability. Are you sure?".to_string(),
                        calibration_adjustment: -0.006,
                    })
                } else {
                    None
                }
            }

            TrainingMethod::ConsiderAlternatives => Some(TrainingFeedback {
                feedback_type: FeedbackType::OutsideViewPrompt(
                    "List 3 reasons you might be wrong".to_string(),
                ),
                message: "What could make this prediction wrong?".to_string(),
                calibration_adjustment: -0.003,
            }),

            TrainingMethod::HistoricalReview { lookback_count } => {
                if step % lookback_count == 0 && step > 0 {
                    let recent_brier = self.calculate_recent_brier(*lookback_count);
                    Some(TrainingFeedback {
                        feedback_type: FeedbackType::BrierScore(recent_brier),
                        message: format!(
                            "Your average Brier over last {} predictions: {:.3}",
                            lookback_count, recent_brier
                        ),
                        calibration_adjustment: -0.004,
                    })
                } else {
                    None
                }
            }

            TrainingMethod::Decomposition => Some(TrainingFeedback {
                feedback_type: FeedbackType::OutsideViewPrompt(
                    "Break this into component probabilities".to_string(),
                ),
                message: "What sub-questions does this depend on?".to_string(),
                calibration_adjustment: -0.003,
            }),

            TrainingMethod::FermiEstimation | TrainingMethod::AdversarialCollab => {
                Some(TrainingFeedback {
                    feedback_type: FeedbackType::Encouragement,
                    message: "Good analytical approach!".to_string(),
                    calibration_adjustment: -0.002,
                })
            }

            TrainingMethod::Combined(methods) => {
                // Apply strongest effect from combined methods
                let mut best_feedback: Option<TrainingFeedback> = None;
                let mut best_adjustment = 0.0f64;

                for method in methods {
                    let temp_participant = TrainingParticipant {
                        training_method: method.clone(),
                        ..self.clone()
                    };
                    if let Some(fb) = temp_participant.generate_feedback(
                        step, predicted, outcome, brier, domain, rng,
                    ) {
                        if fb.calibration_adjustment.abs() > best_adjustment.abs() {
                            best_adjustment = fb.calibration_adjustment;
                            best_feedback = Some(fb);
                        }
                    }
                }
                best_feedback
            }
        }
    }

    fn apply_learning(&mut self, step: usize, brier: f64, feedback: &Option<TrainingFeedback>) {
        if let Some(fb) = feedback {
            // Apply calibration adjustment based on learning rate
            let adjustment = fb.calibration_adjustment * self.learning_rate;
            self.current_ece = (self.current_ece + adjustment).max(0.01);
            self.current_brier = (self.current_brier + adjustment * 0.5).max(0.05);

            // Record history
            self.ece_history.push(TimestampedCalibration {
                step,
                value: self.current_ece,
            });
            self.brier_history.push(TimestampedCalibration {
                step,
                value: self.current_brier,
            });
        }
    }

    fn get_domain_base_rate(&self, domain: &str) -> f64 {
        // Simplified base rates
        match domain {
            "politics" => 0.5,
            "technology" => 0.6,
            "sports" => 0.5,
            "science" => 0.7,
            _ => 0.5,
        }
    }

    fn calculate_recent_brier(&self, count: usize) -> f64 {
        let recent: Vec<&TrainingPrediction> = self
            .training_predictions
            .iter()
            .rev()
            .take(count)
            .collect();

        if recent.is_empty() {
            return self.initial_brier;
        }

        recent.iter().map(|p| p.brier).sum::<f64>() / recent.len() as f64
    }

    /// Calculate improvement from initial to current
    pub fn calculate_improvement(&self) -> CalibrationImprovement {
        let ece_improvement = self.initial_ece - self.current_ece;
        let brier_improvement = self.initial_brier - self.current_brier;
        let ece_improvement_pct = if self.initial_ece > 0.0 {
            ece_improvement / self.initial_ece * 100.0
        } else {
            0.0
        };

        CalibrationImprovement {
            initial_ece: self.initial_ece,
            final_ece: self.current_ece,
            ece_improvement,
            ece_improvement_pct,
            initial_brier: self.initial_brier,
            final_brier: self.current_brier,
            brier_improvement,
            training_method: self.training_method.clone(),
            prediction_count: self.training_predictions.len(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationImprovement {
    pub initial_ece: f64,
    pub final_ece: f64,
    pub ece_improvement: f64,
    pub ece_improvement_pct: f64,
    pub initial_brier: f64,
    pub final_brier: f64,
    pub brier_improvement: f64,
    pub training_method: TrainingMethod,
    pub prediction_count: usize,
}

// ============================================================================
// TRAINING EXPERIMENT
// ============================================================================

/// Run a calibration training experiment
pub struct CalibrationTrainingExperiment {
    pub config: TrainingExperimentConfig,
    pub participants: Vec<TrainingParticipant>,
    pub current_step: usize,
    pub results: TrainingExperimentResults,
    rng: SimpleRng,
}

impl CalibrationTrainingExperiment {
    pub fn new(config: TrainingExperimentConfig) -> Self {
        let rng = SimpleRng::new(config.seed);
        Self {
            config,
            participants: vec![],
            current_step: 0,
            results: TrainingExperimentResults::new(),
            rng,
        }
    }

    /// Initialize participants
    pub fn initialize(&mut self) {
        for (method_idx, method) in self.config.methods.iter().enumerate() {
            for i in 0..self.config.participants_per_group {
                // Sample initial calibration (most people are poorly calibrated)
                let initial_ece = 0.1 + self.rng.next_u64() as f64 / u64::MAX as f64 * 0.2;

                // Sample learning rate (individual differences)
                let learning_rate = 0.5 + self.rng.next_u64() as f64 / u64::MAX as f64 * 1.0;

                let participant = TrainingParticipant::new(
                    format!("p_{}_{}", method_idx, i),
                    method.clone(),
                    initial_ece,
                    learning_rate,
                );

                self.participants.push(participant);
            }
        }
    }

    /// Run training phase
    pub fn run_training(&mut self) {
        for step in 0..self.config.training_duration {
            self.current_step = step;

            // Each participant makes a prediction
            for participant in &mut self.participants {
                // Random domain selection
                let domain_idx = self.rng.next_u64() as usize % self.config.domains.len();
                let domain = &self.config.domains[domain_idx];

                // Random true probability
                let true_prob = 0.1 + self.rng.next_u64() as f64 / u64::MAX as f64 * 0.8;

                participant.process_prediction(step, domain, true_prob, &mut self.rng);
            }

            // Periodic snapshot
            if step % 10 == 0 {
                self.record_snapshot(step);
            }
        }

        // Mark training complete
        for participant in &mut self.participants {
            participant.training_complete = true;
        }
    }

    /// Run assessment phase
    pub fn run_assessment(&mut self) {
        let start_step = self.config.training_duration;

        for step in 0..self.config.assessment_duration {
            let current = start_step + step;
            self.current_step = current;

            for participant in &mut self.participants {
                let domain_idx = self.rng.next_u64() as usize % self.config.domains.len();
                let domain = &self.config.domains[domain_idx];
                let true_prob = 0.1 + self.rng.next_u64() as f64 / u64::MAX as f64 * 0.8;

                // Assessment predictions (no feedback)
                let noise = (self.rng.next_u64() as f64 / u64::MAX as f64 - 0.5)
                    * participant.current_ece * 2.0;
                let predicted = (true_prob + noise).clamp(0.01, 0.99);
                let outcome = (self.rng.next_u64() as f64 / u64::MAX as f64) < true_prob;
                let brier = (predicted - if outcome { 1.0 } else { 0.0 }).powi(2);

                participant.assessment_predictions.push(TrainingPrediction {
                    step: current,
                    domain: domain.to_string(),
                    predicted,
                    outcome,
                    brier,
                    feedback: None,
                });
            }
        }
    }

    /// Run follow-up assessments
    pub fn run_followups(&mut self) {
        for &followup_time in &self.config.followup_assessments.clone() {
            // Simulate time passage with some decay
            for participant in &mut self.participants {
                // Calibration decays slightly without practice
                let decay = 0.01 * (followup_time as f64 / 100.0);
                participant.current_ece = (participant.current_ece + decay).min(participant.initial_ece);
            }

            // Run assessment at followup time
            for participant in &mut self.participants {
                for _ in 0..20 {
                    // Small follow-up assessment
                    let domain_idx = self.rng.next_u64() as usize % self.config.domains.len();
                    let domain = &self.config.domains[domain_idx];
                    let true_prob = 0.1 + self.rng.next_u64() as f64 / u64::MAX as f64 * 0.8;

                    let noise = (self.rng.next_u64() as f64 / u64::MAX as f64 - 0.5)
                        * participant.current_ece * 2.0;
                    let predicted = (true_prob + noise).clamp(0.01, 0.99);
                    let outcome = (self.rng.next_u64() as f64 / u64::MAX as f64) < true_prob;
                    let brier = (predicted - if outcome { 1.0 } else { 0.0 }).powi(2);

                    participant.assessment_predictions.push(TrainingPrediction {
                        step: followup_time,
                        domain: domain.to_string(),
                        predicted,
                        outcome,
                        brier,
                        feedback: None,
                    });
                }
            }

            self.record_snapshot(followup_time);
        }
    }

    /// Test cross-domain transfer
    pub fn test_transfer(&mut self) -> Vec<TransferAnalysis> {
        if !self.config.test_transfer {
            return vec![];
        }

        let mut analyses = Vec::new();

        // For each pair of domains
        let domains = self.config.domains.clone();
        for (i, source) in domains.iter().enumerate() {
            for target in domains.iter().skip(i + 1) {
                let predictor_performances: Vec<PredictorDomainPerformance> = self
                    .participants
                    .iter()
                    .map(|p| {
                        let source_brier = p
                            .training_predictions
                            .iter()
                            .filter(|pred| pred.domain == *source)
                            .map(|pred| pred.brier)
                            .sum::<f64>()
                            / p.training_predictions
                                .iter()
                                .filter(|pred| pred.domain == *source)
                                .count()
                                .max(1) as f64;

                        let target_brier = p
                            .assessment_predictions
                            .iter()
                            .filter(|pred| pred.domain == *target)
                            .map(|pred| pred.brier)
                            .sum::<f64>()
                            / p.assessment_predictions
                                .iter()
                                .filter(|pred| pred.domain == *target)
                                .count()
                                .max(1) as f64;

                        PredictorDomainPerformance {
                            predictor_id: p.id.clone(),
                            source_brier,
                            target_brier,
                        }
                    })
                    .collect();

                let analysis = TransferAnalysis::calculate(source, target, &predictor_performances);
                analyses.push(analysis);
            }
        }

        analyses
    }

    fn record_snapshot(&mut self, step: usize) {
        let mut method_ece: HashMap<TrainingMethod, Vec<f64>> = HashMap::new();

        for participant in &self.participants {
            method_ece
                .entry(participant.training_method.clone())
                .or_insert_with(Vec::new)
                .push(participant.current_ece);
        }

        self.results.snapshots.push(TrainingSnapshot {
            step,
            method_avg_ece: method_ece
                .iter()
                .map(|(m, values)| (m.clone(), values.iter().sum::<f64>() / values.len() as f64))
                .collect(),
        });
    }

    /// Analyze final results
    pub fn analyze(&mut self) -> &TrainingExperimentResults {
        // Group improvements by method
        for participant in &self.participants {
            let improvement = participant.calculate_improvement();
            self.results
                .improvements_by_method
                .entry(participant.training_method.clone())
                .or_insert_with(Vec::new)
                .push(improvement);
        }

        // Calculate method effectiveness
        for (method, improvements) in &self.results.improvements_by_method {
            let avg_improvement = improvements.iter().map(|i| i.ece_improvement).sum::<f64>()
                / improvements.len() as f64;
            let improvement_rate = improvements
                .iter()
                .filter(|i| i.ece_improvement > 0.0)
                .count() as f64
                / improvements.len() as f64;

            self.results.method_effectiveness.insert(
                method.clone(),
                MethodEffectiveness {
                    method: method.clone(),
                    avg_ece_improvement: avg_improvement,
                    improvement_rate,
                    participant_count: improvements.len(),
                    best_improvement: improvements
                        .iter()
                        .map(|i| i.ece_improvement)
                        .fold(f64::NEG_INFINITY, f64::max),
                    worst_improvement: improvements
                        .iter()
                        .map(|i| i.ece_improvement)
                        .fold(f64::INFINITY, f64::min),
                },
            );
        }

        // Transfer analysis
        self.results.transfer_analyses = self.test_transfer();

        // Persistence analysis
        self.analyze_persistence();

        // Generate summary
        self.results.summary = self.generate_summary();

        &self.results
    }

    fn analyze_persistence(&mut self) {
        // Compare calibration at different follow-up times
        for participant in &self.participants {
            let initial_improvement = participant.initial_ece - participant.current_ece;

            // Calculate retention at each follow-up
            let mut retention_rates = Vec::new();
            let followup_count = self.config.followup_assessments.len();

            if followup_count > 0 {
                // Look at ECE history to find follow-up measurements
                let training_end_ece = participant
                    .ece_history
                    .iter()
                    .find(|h| h.step >= self.config.training_duration)
                    .map(|h| h.value)
                    .unwrap_or(participant.current_ece);

                for (i, &followup) in self.config.followup_assessments.iter().enumerate() {
                    let followup_ece = participant
                        .ece_history
                        .iter()
                        .find(|h| h.step >= followup)
                        .map(|h| h.value)
                        .unwrap_or(training_end_ece);

                    let improvement_retained = participant.initial_ece - followup_ece;
                    let retention = if initial_improvement > 0.0 {
                        (improvement_retained / initial_improvement).clamp(0.0, 1.0)
                    } else {
                        1.0
                    };

                    retention_rates.push((followup, retention));
                }
            }

            self.results
                .persistence_data
                .entry(participant.training_method.clone())
                .or_insert_with(Vec::new)
                .push(retention_rates);
        }
    }

    fn generate_summary(&self) -> TrainingSummary {
        // Find best method
        let best_method = self
            .results
            .method_effectiveness
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.avg_ece_improvement
                    .partial_cmp(&b.avg_ece_improvement)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(m, _)| m.clone())
            .unwrap_or(TrainingMethod::Control);

        // Calculate overall improvement rate
        let total_improved: usize = self
            .results
            .improvements_by_method
            .values()
            .flat_map(|v| v.iter())
            .filter(|i| i.ece_improvement > 0.0)
            .count();
        let total_participants: usize = self
            .results
            .improvements_by_method
            .values()
            .map(|v| v.len())
            .sum();

        let overall_improvement_rate = total_improved as f64 / total_participants.max(1) as f64;

        // Analyze transfer
        let positive_transfer_pairs: Vec<(String, String)> = self
            .results
            .transfer_analyses
            .iter()
            .filter(|t| t.positive_transfer)
            .map(|t| (t.source_domain.clone(), t.target_domain.clone()))
            .collect();

        // Analyze persistence
        let avg_retention = self.calculate_average_retention();

        TrainingSummary {
            best_method,
            overall_improvement_rate,
            avg_ece_improvement: self.calculate_overall_avg_improvement(),
            calibration_is_trainable: overall_improvement_rate > 0.5,
            positive_transfer_pairs,
            transfer_coefficient_avg: self.calculate_avg_transfer_coefficient(),
            persistence_half_life: self.estimate_half_life(),
            avg_retention_at_followup: avg_retention,
            recommendations: self.generate_recommendations(),
        }
    }

    fn calculate_overall_avg_improvement(&self) -> f64 {
        let all_improvements: Vec<f64> = self
            .results
            .improvements_by_method
            .values()
            .flat_map(|v| v.iter().map(|i| i.ece_improvement))
            .collect();

        if all_improvements.is_empty() {
            0.0
        } else {
            all_improvements.iter().sum::<f64>() / all_improvements.len() as f64
        }
    }

    fn calculate_avg_transfer_coefficient(&self) -> f64 {
        if self.results.transfer_analyses.is_empty() {
            return 0.0;
        }

        self.results
            .transfer_analyses
            .iter()
            .map(|t| t.transfer_coefficient)
            .filter(|c| !c.is_nan())
            .sum::<f64>()
            / self.results.transfer_analyses.len() as f64
    }

    fn calculate_average_retention(&self) -> HashMap<usize, f64> {
        let mut retention_by_time: HashMap<usize, Vec<f64>> = HashMap::new();

        for retention_series in self.results.persistence_data.values().flatten() {
            for &(time, retention) in retention_series {
                retention_by_time
                    .entry(time)
                    .or_insert_with(Vec::new)
                    .push(retention);
            }
        }

        retention_by_time
            .iter()
            .map(|(time, values)| (*time, values.iter().sum::<f64>() / values.len() as f64))
            .collect()
    }

    fn estimate_half_life(&self) -> usize {
        // Simple estimation: find time when average retention drops below 50%
        let avg_retention = self.calculate_average_retention();
        let mut times: Vec<_> = avg_retention.iter().collect();
        times.sort_by_key(|(t, _)| *t);

        for (time, retention) in times {
            if *retention < 0.5 {
                return *time;
            }
        }

        // If never drops below 50%, return last follow-up time
        self.config.followup_assessments.last().copied().unwrap_or(1000)
    }

    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Method recommendation
        if let Some((best_method, effectiveness)) = self
            .results
            .method_effectiveness
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.avg_ece_improvement
                    .partial_cmp(&b.avg_ece_improvement)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            recommendations.push(format!(
                "Best training method: {:?} (avg improvement: {:.3})",
                best_method, effectiveness.avg_ece_improvement
            ));
        }

        // Transfer recommendation
        if !self.results.transfer_analyses.is_empty() {
            let best_transfer = self
                .results
                .transfer_analyses
                .iter()
                .max_by(|a, b| {
                    a.transfer_coefficient
                        .partial_cmp(&b.transfer_coefficient)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

            if let Some(transfer) = best_transfer {
                recommendations.push(format!(
                    "Best transfer: {} -> {} (coefficient: {:.2})",
                    transfer.source_domain, transfer.target_domain, transfer.transfer_coefficient
                ));
            }
        }

        // Persistence recommendation
        let half_life = self.estimate_half_life();
        recommendations.push(format!(
            "Calibration half-life: {} steps - regular practice recommended",
            half_life
        ));

        recommendations
    }

    /// Run the full experiment
    pub fn run(&mut self) -> &TrainingExperimentResults {
        self.initialize();
        self.run_training();
        self.run_assessment();
        self.run_followups();
        self.analyze()
    }
}

// ============================================================================
// EXPERIMENT RESULTS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExperimentResults {
    pub improvements_by_method: HashMap<TrainingMethod, Vec<CalibrationImprovement>>,
    pub method_effectiveness: HashMap<TrainingMethod, MethodEffectiveness>,
    pub snapshots: Vec<TrainingSnapshot>,
    pub transfer_analyses: Vec<TransferAnalysis>,
    pub persistence_data: HashMap<TrainingMethod, Vec<Vec<(usize, f64)>>>,
    pub summary: TrainingSummary,
}

impl TrainingExperimentResults {
    pub fn new() -> Self {
        Self {
            improvements_by_method: HashMap::new(),
            method_effectiveness: HashMap::new(),
            snapshots: vec![],
            transfer_analyses: vec![],
            persistence_data: HashMap::new(),
            summary: TrainingSummary::default(),
        }
    }
}

impl Default for TrainingExperimentResults {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodEffectiveness {
    pub method: TrainingMethod,
    pub avg_ece_improvement: f64,
    pub improvement_rate: f64,
    pub participant_count: usize,
    pub best_improvement: f64,
    pub worst_improvement: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSnapshot {
    pub step: usize,
    pub method_avg_ece: HashMap<TrainingMethod, f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingSummary {
    pub best_method: TrainingMethod,
    pub overall_improvement_rate: f64,
    pub avg_ece_improvement: f64,
    pub calibration_is_trainable: bool,
    pub positive_transfer_pairs: Vec<(String, String)>,
    pub transfer_coefficient_avg: f64,
    pub persistence_half_life: usize,
    pub avg_retention_at_followup: HashMap<usize, f64>,
    pub recommendations: Vec<String>,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_participant_creation() {
        let participant = TrainingParticipant::new(
            "test".to_string(),
            TrainingMethod::ImmediateFeedback,
            0.15,
            1.0,
        );
        assert_eq!(participant.initial_ece, 0.15);
        assert!(!participant.training_complete);
    }

    #[test]
    fn test_training_experiment_initialization() {
        let config = TrainingExperimentConfig {
            participants_per_group: 5,
            methods: vec![TrainingMethod::Control, TrainingMethod::ImmediateFeedback],
            ..Default::default()
        };

        let mut experiment = CalibrationTrainingExperiment::new(config);
        experiment.initialize();

        assert_eq!(experiment.participants.len(), 10);
    }

    #[test]
    fn test_improvement_calculation() {
        let mut participant = TrainingParticipant::new(
            "test".to_string(),
            TrainingMethod::ImmediateFeedback,
            0.2,
            1.0,
        );

        // Simulate improvement
        participant.current_ece = 0.15;
        participant.current_brier = 0.18;

        let improvement = participant.calculate_improvement();
        assert!(improvement.ece_improvement > 0.0);
        assert!(improvement.ece_improvement_pct > 0.0);
    }

    #[test]
    fn test_small_experiment() {
        let config = TrainingExperimentConfig {
            participants_per_group: 3,
            methods: vec![TrainingMethod::Control, TrainingMethod::ImmediateFeedback],
            training_duration: 10,
            assessment_duration: 5,
            followup_assessments: vec![20],
            domains: vec!["test".to_string()],
            test_transfer: false,
            ..Default::default()
        };

        let mut experiment = CalibrationTrainingExperiment::new(config);
        let results = experiment.run();

        assert!(!results.method_effectiveness.is_empty());
        assert!(results.summary.overall_improvement_rate >= 0.0);
    }
}
