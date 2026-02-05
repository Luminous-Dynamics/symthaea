//! Prediction-Gate Experiment: Validating Learning, Not Φ
//!
//! **Key Insight**: Our Φ-Gate experiment showed no correlation (r ≈ 0) between Φ and
//! reasoning accuracy. This is because Φ measures integration, not understanding.
//!
//! **New Approach**: Test whether the system can LEARN to predict better.
//! Understanding = accurate prediction + generalization to novel inputs
//!
//! **Hypothesis**: After training, prediction error should decrease AND the system
//! should generalize to unseen patterns.
//!
//! **Metrics**:
//! 1. Prediction Error Reduction (PER): How much does MSE decrease with training?
//! 2. Generalization Gap (GG): Performance on held-out test vs training data
//! 3. Learning Rate (LR): How quickly does the system improve?
//! 4. Retention (R): Does it remember after consolidation?
//!
//! A system "passes" the Prediction-Gate if:
//! - PER > 50% (significant learning)
//! - GG < 30% (reasonable generalization)
//! - LR shows consistent improvement
//! - R > 80% after consolidation

use std::time::Instant;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::learnable_ltc::{LearnableLTC, LearnableLTCConfig};

/// Configuration for Prediction-Gate experiment
#[derive(Debug, Clone)]
pub struct PredictionGateConfig {
    /// Number of training episodes
    pub num_episodes: usize,

    /// Number of examples per episode
    pub examples_per_episode: usize,

    /// Test set size (held out)
    pub test_size: usize,

    /// Number of consolidation cycles
    pub consolidation_cycles: usize,

    /// Input dimension for patterns
    pub pattern_dim: usize,

    /// Output dimension
    pub output_dim: usize,

    /// LTC configuration
    pub ltc_config: LearnableLTCConfig,
}

impl Default for PredictionGateConfig {
    fn default() -> Self {
        let ltc_config = LearnableLTCConfig {
            num_neurons: 128,
            input_dim: 64,
            output_dim: 16,
            num_steps: 20,
            learning_rate: 0.005,
            ..Default::default()
        };

        Self {
            num_episodes: 10,
            examples_per_episode: 50,
            test_size: 20,
            consolidation_cycles: 3,
            pattern_dim: 64,
            output_dim: 16,
            ltc_config,
        }
    }
}

/// A pattern prediction task
#[derive(Debug, Clone)]
pub struct PatternTask {
    /// Task name
    pub name: String,

    /// Task description
    pub description: String,

    /// Generator function (seed -> input, target)
    pub task_type: TaskType,
}

/// Types of prediction tasks
#[derive(Debug, Clone)]
pub enum TaskType {
    /// Predict the sequence continuation
    SequencePrediction,

    /// Classify patterns into categories
    PatternClassification,

    /// Transform input to output (encoding)
    Transformation,

    /// Predict which dimension is dominant
    DimensionDetection,

    /// Detect periodicity in input
    PeriodicityDetection,
}

impl TaskType {
    /// Generate training example
    fn generate(&self, dim: usize, output_dim: usize, _seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut rng = rand::thread_rng();

        match self {
            TaskType::SequencePrediction => {
                // Input: sequence values, Target: next values
                let mut input = vec![0.0f32; dim];
                let mut target = vec![0.0f32; output_dim];

                // Generate a simple arithmetic or geometric pattern
                let start: f32 = rng.gen_range(-1.0..1.0);
                let step: f32 = rng.gen_range(-0.1..0.1);

                for i in 0..dim {
                    input[i] = (start + step * i as f32).tanh();
                }

                // Target: continuation of pattern
                for i in 0..output_dim {
                    target[i] = (start + step * (dim + i) as f32).tanh();
                }

                (input, target)
            }

            TaskType::PatternClassification => {
                // Input: pattern, Target: one-hot class
                let mut input = vec![0.0f32; dim];
                let mut target = vec![0.0f32; output_dim];

                // Generate patterns with different signatures
                let class = rng.gen_range(0..output_dim);

                // Each class has a different frequency pattern
                let freq = 2.0 * std::f32::consts::PI * (class + 1) as f32 / dim as f32;
                for i in 0..dim {
                    input[i] = (freq * i as f32).sin() + rng.gen_range(-0.1..0.1);
                }

                target[class] = 1.0;

                (input, target)
            }

            TaskType::Transformation => {
                // Input: raw values, Target: transformed values
                let mut input = vec![0.0f32; dim];
                let mut target = vec![0.0f32; output_dim];

                // Random input
                for i in 0..dim {
                    input[i] = rng.gen_range(-1.0..1.0);
                }

                // Target: some transformation (e.g., dimensionality reduction + nonlinearity)
                for i in 0..output_dim {
                    let chunk_size = dim / output_dim;
                    let start = i * chunk_size;
                    let end = (start + chunk_size).min(dim);

                    let sum: f32 = input[start..end].iter().sum();
                    target[i] = sum.tanh();
                }

                (input, target)
            }

            TaskType::DimensionDetection => {
                // Input: vector with one dominant dimension, Target: which one
                let mut input = vec![0.0f32; dim];
                let mut target = vec![0.0f32; output_dim];

                // Pick a dominant dimension
                let dominant = rng.gen_range(0..dim.min(output_dim));

                // Fill with noise
                for i in 0..dim {
                    input[i] = rng.gen_range(-0.3..0.3);
                }

                // Make one dimension strong
                input[dominant] = rng.gen_range(0.7..1.0) * if rng.gen() { 1.0 } else { -1.0 };

                // Target: one-hot for dominant dimension (mapped to output_dim)
                let target_idx = dominant % output_dim;
                target[target_idx] = 1.0;

                (input, target)
            }

            TaskType::PeriodicityDetection => {
                // Input: signal, Target: periodicity features
                let mut input = vec![0.0f32; dim];
                let mut target = vec![0.0f32; output_dim];

                // Generate signal with some periodicity
                let num_periods: usize = rng.gen_range(1..5);
                let freq = 2.0 * std::f32::consts::PI * num_periods as f32 / dim as f32;
                let amplitude: f32 = rng.gen_range(0.5..1.0);
                let phase: f32 = rng.gen_range(0.0..2.0 * std::f32::consts::PI);

                for i in 0..dim {
                    input[i] = amplitude * (freq * i as f32 + phase).sin()
                        + rng.gen_range(-0.1..0.1);
                }

                // Target: encode periodicity info
                target[0] = (num_periods as f32 / 5.0).min(1.0);  // Period count
                target[1] = amplitude;  // Amplitude
                if output_dim > 2 {
                    target[2] = (phase / (2.0 * std::f32::consts::PI)).fract();  // Phase
                }

                (input, target)
            }
        }
    }
}

/// Results from one training episode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeResult {
    pub episode: usize,
    pub train_loss: f32,
    pub test_loss: f32,
    pub train_accuracy: f32,
    pub test_accuracy: f32,
    pub tau_mean: f32,
    pub tau_std: f32,
    pub duration_ms: u64,
}

/// Complete results from Prediction-Gate experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionGateResults {
    /// Task being tested
    pub task_name: String,

    /// Episode-by-episode results
    pub episodes: Vec<EpisodeResult>,

    /// Initial prediction error (before training)
    pub initial_error: f32,

    /// Final prediction error (after training)
    pub final_error: f32,

    /// Prediction Error Reduction (%)
    pub per: f32,

    /// Generalization Gap (test_loss - train_loss) / train_loss
    pub generalization_gap: f32,

    /// Learning rate (slope of improvement)
    pub learning_rate: f32,

    /// Retention after consolidation (%)
    pub retention: f32,

    /// Post-consolidation error
    pub post_consolidation_error: f32,

    /// Did we pass the Prediction-Gate?
    pub passed: bool,

    /// Detailed reasoning
    pub reasoning: String,

    /// Total time
    pub total_time_ms: u64,
}

/// Run the Prediction-Gate experiment
pub struct PredictionGate {
    config: PredictionGateConfig,
    ltc: LearnableLTC,
}

impl PredictionGate {
    pub fn new(config: PredictionGateConfig) -> anyhow::Result<Self> {
        let ltc = LearnableLTC::new(config.ltc_config.clone())?;
        Ok(Self { config, ltc })
    }

    /// Run experiment on a single task
    pub fn run_task(&mut self, task: &PatternTask) -> anyhow::Result<PredictionGateResults> {
        let start_time = Instant::now();

        // Generate test set (held out)
        let test_set: Vec<(Vec<f32>, Vec<f32>)> = (0..self.config.test_size)
            .map(|i| task.task_type.generate(
                self.config.pattern_dim,
                self.config.output_dim,
                i as u64 + 10000  // Different seed for test
            ))
            .collect();

        // Measure initial error (before training)
        let initial_error = self.evaluate(&test_set)?;

        let mut episodes = Vec::new();

        // Training episodes
        for episode in 0..self.config.num_episodes {
            let episode_start = Instant::now();

            // Generate training examples for this episode
            let train_set: Vec<(Vec<f32>, Vec<f32>)> = (0..self.config.examples_per_episode)
                .map(|i| task.task_type.generate(
                    self.config.pattern_dim,
                    self.config.output_dim,
                    (episode * 1000 + i) as u64
                ))
                .collect();

            // Train on this episode
            let mut train_loss = 0.0;
            for (input, target) in &train_set {
                let loss = self.ltc.train_step(input, target)?;
                train_loss += loss;
                self.ltc.reset_state();
            }
            train_loss /= train_set.len() as f32;

            // Evaluate on test set
            let test_loss = self.evaluate(&test_set)?;

            // Calculate accuracy (for classification tasks)
            let train_accuracy = self.calculate_accuracy(&train_set)?;
            let test_accuracy = self.calculate_accuracy(&test_set)?;

            // Get tau distribution
            let (tau_mean, tau_std, _, _) = self.ltc.get_tau_distribution();

            episodes.push(EpisodeResult {
                episode,
                train_loss,
                test_loss,
                train_accuracy,
                test_accuracy,
                tau_mean,
                tau_std,
                duration_ms: episode_start.elapsed().as_millis() as u64,
            });
        }

        // Final error after training
        let final_error = self.evaluate(&test_set)?;

        // Simulate consolidation (replay training data)
        let _consolidation_start = Instant::now();
        for _cycle in 0..self.config.consolidation_cycles {
            // Replay subset of training data
            for i in 0..10 {
                let (input, target) = task.task_type.generate(
                    self.config.pattern_dim,
                    self.config.output_dim,
                    (i + 5000) as u64
                );
                let _ = self.ltc.train_step(&input, &target);
                self.ltc.reset_state();
            }
        }

        let post_consolidation_error = self.evaluate(&test_set)?;

        // Calculate metrics
        let per = if initial_error > 0.0 {
            ((initial_error - final_error) / initial_error * 100.0).max(0.0)
        } else {
            0.0
        };

        let last_train_loss = episodes.last().map(|e| e.train_loss).unwrap_or(final_error);
        let generalization_gap = if last_train_loss > 0.001 {
            ((final_error - last_train_loss) / last_train_loss * 100.0).max(0.0)
        } else {
            0.0
        };

        // Calculate learning rate (slope of test_loss decrease)
        let learning_rate = if episodes.len() >= 2 {
            let first = episodes.first().unwrap().test_loss;
            let last = episodes.last().unwrap().test_loss;
            (first - last) / episodes.len() as f32
        } else {
            0.0
        };

        // Retention
        let retention = if final_error > 0.001 {
            (1.0 - (post_consolidation_error - final_error).abs() / final_error) * 100.0
        } else {
            100.0
        };

        // Determine if passed
        let per_pass = per > 50.0;
        let gg_pass = generalization_gap < 30.0;
        let lr_pass = learning_rate > 0.0;
        let retention_pass = retention > 80.0;

        let passed = per_pass && gg_pass && lr_pass && retention_pass;

        let reasoning = format!(
            "PER: {:.1}% (>{} 50%)  |  GG: {:.1}% (<{} 30%)  |  LR: {:.4} (>{} 0)  |  R: {:.1}% (>{} 80%)",
            per, if per_pass { "✓" } else { "✗" },
            generalization_gap, if gg_pass { "✓" } else { "✗" },
            learning_rate, if lr_pass { "✓" } else { "✗" },
            retention, if retention_pass { "✓" } else { "✗" }
        );

        Ok(PredictionGateResults {
            task_name: task.name.clone(),
            episodes,
            initial_error,
            final_error,
            per,
            generalization_gap,
            learning_rate,
            retention,
            post_consolidation_error,
            passed,
            reasoning,
            total_time_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    /// Evaluate on a dataset
    fn evaluate(&mut self, dataset: &[(Vec<f32>, Vec<f32>)]) -> anyhow::Result<f32> {
        let mut total_loss = 0.0;

        for (input, target) in dataset {
            let (output, _) = self.ltc.forward(input)?;

            // MSE loss
            let loss: f32 = output.iter().zip(target.iter())
                .map(|(o, t)| (o - t).powi(2))
                .sum::<f32>() / target.len() as f32;

            total_loss += loss;
            self.ltc.reset_state();
        }

        Ok(total_loss / dataset.len() as f32)
    }

    /// Calculate accuracy (for classification-like tasks)
    fn calculate_accuracy(&mut self, dataset: &[(Vec<f32>, Vec<f32>)]) -> anyhow::Result<f32> {
        let mut correct = 0;

        for (input, target) in dataset {
            let (output, _) = self.ltc.forward(input)?;

            // Find argmax for both
            let pred_max = output.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            let target_max = target.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            if pred_max == target_max {
                correct += 1;
            }

            self.ltc.reset_state();
        }

        Ok(correct as f32 / dataset.len() as f32)
    }

    /// Run on all task types
    pub fn run_all_tasks(&mut self) -> anyhow::Result<Vec<PredictionGateResults>> {
        let tasks = vec![
            PatternTask {
                name: "Sequence Prediction".to_string(),
                description: "Predict continuation of arithmetic sequences".to_string(),
                task_type: TaskType::SequencePrediction,
            },
            PatternTask {
                name: "Pattern Classification".to_string(),
                description: "Classify sinusoidal patterns by frequency".to_string(),
                task_type: TaskType::PatternClassification,
            },
            PatternTask {
                name: "Transformation".to_string(),
                description: "Learn input-to-output mapping".to_string(),
                task_type: TaskType::Transformation,
            },
            PatternTask {
                name: "Dimension Detection".to_string(),
                description: "Identify dominant input dimension".to_string(),
                task_type: TaskType::DimensionDetection,
            },
            PatternTask {
                name: "Periodicity Detection".to_string(),
                description: "Extract period, amplitude, phase from signals".to_string(),
                task_type: TaskType::PeriodicityDetection,
            },
        ];

        let mut results = Vec::new();

        for task in tasks {
            // Reset LTC for each task
            self.ltc = LearnableLTC::new(self.config.ltc_config.clone())?;

            let result = self.run_task(&task)?;
            results.push(result);
        }

        Ok(results)
    }
}

/// Print summary of Prediction-Gate results
pub fn print_prediction_gate_summary(results: &[PredictionGateResults]) {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║        PREDICTION-GATE EXPERIMENT RESULTS                    ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Testing: Does the system LEARN to predict better?            ║");
    println!("║ Criteria: PER>50%, GG<30%, LR>0, Retention>80%              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut passed_count = 0;
    let mut total_per = 0.0;

    for result in results {
        let status = if result.passed { "✓ PASS" } else { "✗ FAIL" };

        println!("┌─────────────────────────────────────────────────────────────┐");
        println!("│ Task: {:40} {:8} │", result.task_name, status);
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│ Initial Error: {:.4}  →  Final Error: {:.4}               │",
                 result.initial_error, result.final_error);
        println!("│ {}     │", result.reasoning);
        println!("│ Time: {}ms                                              │", result.total_time_ms);
        println!("└─────────────────────────────────────────────────────────────┘\n");

        if result.passed {
            passed_count += 1;
        }
        total_per += result.per;
    }

    let avg_per = total_per / results.len() as f32;
    let pass_rate = passed_count as f32 / results.len() as f32 * 100.0;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║ OVERALL: {}/{} tasks passed ({:.1}%)                          ║",
             passed_count, results.len(), pass_rate);
    println!("║ Average Prediction Error Reduction: {:.1}%                    ║", avg_per);

    if passed_count >= (results.len() as f32 * 0.6) as usize {
        println!("║                                                              ║");
        println!("║  🎉 PREDICTION-GATE: PASSED                                  ║");
        println!("║  The system demonstrates genuine learning ability!           ║");
    } else {
        println!("║                                                              ║");
        println!("║  ⚠️  PREDICTION-GATE: NOT YET PASSED                         ║");
        println!("║  Learning needs improvement before reasoning tasks           ║");
    }

    println!("╚══════════════════════════════════════════════════════════════╝\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_generation() {
        let dim = 32;
        let output_dim = 8;

        for task_type in [
            TaskType::SequencePrediction,
            TaskType::PatternClassification,
            TaskType::Transformation,
            TaskType::DimensionDetection,
            TaskType::PeriodicityDetection,
        ] {
            let (input, target) = task_type.generate(dim, output_dim, 42);
            assert_eq!(input.len(), dim);
            assert_eq!(target.len(), output_dim);
        }
    }

    #[test]
    fn test_prediction_gate_single_task() {
        let config = PredictionGateConfig {
            num_episodes: 3,
            examples_per_episode: 10,
            test_size: 5,
            consolidation_cycles: 1,
            ..Default::default()
        };

        let mut gate = PredictionGate::new(config).unwrap();

        let task = PatternTask {
            name: "Test".to_string(),
            description: "Test task".to_string(),
            task_type: TaskType::Transformation,
        };

        let result = gate.run_task(&task).unwrap();

        assert!(!result.task_name.is_empty());
        assert_eq!(result.episodes.len(), 3);
    }
}
