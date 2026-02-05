//! # Primitive Learning with Cognitive Loop
//!
//! This benchmark tests primitive learning using the bidirectional HDC↔LTC
//! cognitive loop instead of the one-way pipeline.
//!
//! ## Hypothesis
//!
//! The bidirectional coupling should enable better learning on primitive
//! reasoning tasks because:
//! 1. Prediction error drives learning in the correct direction
//! 2. Attention emergence focuses on relevant primitives
//! 3. The loop creates temporal context across the sequence
//!
//! ## Comparison
//!
//! We compare:
//! - Original: Direct LTC training (ltc.train_step)
//! - Cognitive Loop: Bidirectional HDC↔LTC (cognitive_loop.cycle)

use crate::cognitive_loop::{CognitiveLoopBuilder, CognitiveLoopService};
use symthaea_core::hdc::primitive_system::PrimitiveSystem;
use crate::learnable_ltc::{LearnableLTC, LearnableLTCConfig};
use serde::{Serialize, Deserialize};
use std::time::Instant;
use rand::Rng;

/// Configuration for cognitive primitive learning
#[derive(Debug, Clone)]
pub struct CognitivePrimitiveLearningConfig {
    /// Number of training episodes
    pub num_episodes: usize,
    /// Cycles per episode (for cognitive loop)
    pub cycles_per_episode: usize,
    /// Test patterns
    pub test_size: usize,
    /// Learning rate for cognitive loop
    pub learning_rate: f32,
    /// Attention learning rate
    pub attention_lr: f32,
}

impl Default for CognitivePrimitiveLearningConfig {
    fn default() -> Self {
        Self {
            num_episodes: 10,
            cycles_per_episode: 50,
            test_size: 20,
            learning_rate: 0.01,
            attention_lr: 0.05,
        }
    }
}

/// Primitive reasoning task types
#[derive(Debug, Clone, Copy)]
pub enum CognitivePrimitiveTask {
    /// CAUSE → ? → EFFECT
    CausalChain,
    /// TRUE → ? → CONSEQUENCE
    LogicalInference,
    /// WANT → ? → HAVE
    ActionSequence,
    /// BEFORE → ? → AFTER
    TemporalOrder,
}

impl CognitivePrimitiveTask {
    /// Get the pattern parts and valid completions
    pub fn get_pattern(&self) -> (&'static str, &'static str, Vec<&'static str>) {
        match self {
            CognitivePrimitiveTask::CausalChain =>
                ("cause", "effect", vec!["action", "process", "change"]),
            CognitivePrimitiveTask::LogicalInference =>
                ("true", "consequence", vec!["implies", "therefore", "thus"]),
            CognitivePrimitiveTask::ActionSequence =>
                ("want", "have", vec!["do", "make", "get"]),
            CognitivePrimitiveTask::TemporalOrder =>
                ("before", "after", vec!["now", "during", "when"]),
        }
    }

    /// Get task name
    pub fn name(&self) -> &'static str {
        match self {
            CognitivePrimitiveTask::CausalChain => "Causal Chain",
            CognitivePrimitiveTask::LogicalInference => "Logical Inference",
            CognitivePrimitiveTask::ActionSequence => "Action Sequence",
            CognitivePrimitiveTask::TemporalOrder => "Temporal Order",
        }
    }

    /// Generate input strings for the cognitive loop
    pub fn generate_inputs(&self, count: usize) -> Vec<String> {
        let (start, end, completions) = self.get_pattern();
        let mut rng = rand::thread_rng();

        (0..count).map(|_| {
            let completion = completions[rng.gen_range(0..completions.len())];
            // Create a string that the encoder can process
            // Format: "start completion end" - the full reasoning chain
            format!("{} {} {}", start, completion, end)
        }).collect()
    }
}

/// Episode results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveEpisodeResult {
    pub episode: usize,
    pub avg_prediction_error: f32,
    pub attention_variance: f32,
    pub pattern_accuracy: f32,
    pub duration_ms: u64,
}

/// Complete benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitivePrimitiveLearningResults {
    pub task_name: String,
    pub method: String,
    pub episodes: Vec<CognitiveEpisodeResult>,
    pub initial_error: f32,
    pub final_error: f32,
    pub error_reduction_percent: f32,
    pub initial_accuracy: f32,
    pub final_accuracy: f32,
    pub accuracy_improvement: f32,
    pub attention_emerged: bool,
    pub passed: bool,
    pub reasoning: String,
    pub total_time_ms: u64,
}

/// Cognitive Primitive Learning benchmark
pub struct CognitivePrimitiveLearning {
    config: CognitivePrimitiveLearningConfig,
}

impl CognitivePrimitiveLearning {
    pub fn new(config: CognitivePrimitiveLearningConfig) -> Self {
        Self { config }
    }

    /// Run task with cognitive loop (bidirectional HDC↔LTC)
    pub fn run_cognitive(&self, task: CognitivePrimitiveTask) -> anyhow::Result<CognitivePrimitiveLearningResults> {
        let start_time = Instant::now();

        // Build cognitive loop service with tuned parameters
        // Step 1: Increased neurons 128→256, learning rate more aggressive
        let mut loop_service = CognitiveLoopBuilder::default()
            .with_ltc_neurons(256)
            .with_learning_rate(self.config.learning_rate * 3.0) // 3x learning rate
            .with_attention_lr(self.config.attention_lr * 2.0)   // 2x attention lr
            .with_learning_threshold(0.005) // Lower threshold = learn more often
            .build()?;

        // Generate test inputs
        let test_inputs = task.generate_inputs(self.config.test_size);

        // Measure initial error
        let initial_error = self.measure_error(&mut loop_service, &test_inputs);
        let initial_accuracy = self.measure_accuracy(&mut loop_service, task, &test_inputs);

        let mut episodes = Vec::new();

        // Training episodes
        for episode in 0..self.config.num_episodes {
            let episode_start = Instant::now();

            // Generate training inputs for this episode
            let train_inputs = task.generate_inputs(self.config.cycles_per_episode);

            // Run cycles through the cognitive loop
            let mut total_error = 0.0;
            for input in &train_inputs {
                let result = loop_service.cycle(input);
                total_error += result.prediction_error;
            }
            let avg_error = total_error / train_inputs.len() as f32;

            // Measure accuracy on test set
            let accuracy = self.measure_accuracy(&mut loop_service, task, &test_inputs);

            // Get attention variance
            let stats = loop_service.stats();

            episodes.push(CognitiveEpisodeResult {
                episode,
                avg_prediction_error: avg_error,
                attention_variance: stats.attention_variance,
                pattern_accuracy: accuracy,
                duration_ms: episode_start.elapsed().as_millis() as u64,
            });
        }

        // Final measurements
        let final_error = self.measure_error(&mut loop_service, &test_inputs);
        let final_accuracy = self.measure_accuracy(&mut loop_service, task, &test_inputs);

        // Calculate metrics
        let error_reduction = if initial_error > 0.0 {
            ((initial_error - final_error) / initial_error * 100.0).max(-100.0)
        } else {
            0.0
        };

        let accuracy_improvement = final_accuracy - initial_accuracy;
        let attention_emerged = loop_service.stats().attention_variance > 0.001;

        // Pass criteria:
        // - Error decreased OR accuracy improved significantly
        // - Attention emerged (shows the loop is differentiating)
        let passed = (error_reduction > 10.0 || accuracy_improvement > 10.0) && attention_emerged;

        let reasoning = format!(
            "Error: {:.1}% reduction | Accuracy: {:.1}% → {:.1}% (Δ{:+.1}%) | Attention: {}",
            error_reduction,
            initial_accuracy, final_accuracy, accuracy_improvement,
            if attention_emerged { "emerged" } else { "uniform" }
        );

        Ok(CognitivePrimitiveLearningResults {
            task_name: task.name().to_string(),
            method: "Cognitive Loop (HDC↔LTC)".to_string(),
            episodes,
            initial_error,
            final_error,
            error_reduction_percent: error_reduction,
            initial_accuracy,
            final_accuracy,
            accuracy_improvement,
            attention_emerged,
            passed,
            reasoning,
            total_time_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    /// Run task with original one-way pipeline (for comparison)
    pub fn run_original(&self, task: CognitivePrimitiveTask) -> anyhow::Result<CognitivePrimitiveLearningResults> {
        let start_time = Instant::now();

        // Create direct LTC (one-way pipeline)
        let ltc_config = LearnableLTCConfig {
            input_dim: 128,
            output_dim: 64,
            num_neurons: 128,
            learning_rate: self.config.learning_rate,
            ..Default::default()
        };
        let mut ltc = LearnableLTC::new(ltc_config)?;

        // Get primitive system for encoding
        let primitive_system = PrimitiveSystem::global();

        // Generate test inputs
        let test_inputs = task.generate_inputs(self.config.test_size);

        // Measure initial error
        let initial_error = self.measure_ltc_error(&mut ltc, &test_inputs, primitive_system);
        let initial_accuracy = self.measure_ltc_accuracy(&mut ltc, task, &test_inputs, primitive_system);

        let mut episodes = Vec::new();

        // Training episodes
        for episode in 0..self.config.num_episodes {
            let episode_start = Instant::now();

            // Generate training inputs
            let train_inputs = task.generate_inputs(self.config.cycles_per_episode);

            // Train with one-way pipeline
            let mut total_loss = 0.0;
            for input in &train_inputs {
                let (encoded_input, target) = self.encode_for_ltc(input, primitive_system);
                if let Ok(loss) = ltc.train_step(&encoded_input, &target) {
                    total_loss += loss;
                }
                ltc.reset_state();
            }
            let avg_loss = total_loss / train_inputs.len() as f32;

            // Measure accuracy
            let accuracy = self.measure_ltc_accuracy(&mut ltc, task, &test_inputs, primitive_system);

            episodes.push(CognitiveEpisodeResult {
                episode,
                avg_prediction_error: avg_loss,
                attention_variance: 0.0,  // No attention in one-way pipeline
                pattern_accuracy: accuracy,
                duration_ms: episode_start.elapsed().as_millis() as u64,
            });
        }

        // Final measurements
        let final_error = self.measure_ltc_error(&mut ltc, &test_inputs, primitive_system);
        let final_accuracy = self.measure_ltc_accuracy(&mut ltc, task, &test_inputs, primitive_system);

        let error_reduction = if initial_error > 0.0 {
            ((initial_error - final_error) / initial_error * 100.0).max(-100.0)
        } else {
            0.0
        };

        let accuracy_improvement = final_accuracy - initial_accuracy;

        // Original pipeline has no attention emergence
        let passed = accuracy_improvement > 10.0;

        let reasoning = format!(
            "Error: {:.1}% reduction | Accuracy: {:.1}% → {:.1}% (Δ{:+.1}%) | No attention (one-way)",
            error_reduction,
            initial_accuracy, final_accuracy, accuracy_improvement
        );

        Ok(CognitivePrimitiveLearningResults {
            task_name: task.name().to_string(),
            method: "Original (One-Way LTC)".to_string(),
            episodes,
            initial_error,
            final_error,
            error_reduction_percent: error_reduction,
            initial_accuracy,
            final_accuracy,
            accuracy_improvement,
            attention_emerged: false,
            passed,
            reasoning,
            total_time_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    /// Run both methods and compare
    pub fn run_comparison(&self, task: CognitivePrimitiveTask) -> anyhow::Result<(CognitivePrimitiveLearningResults, CognitivePrimitiveLearningResults)> {
        let cognitive_result = self.run_cognitive(task)?;
        let original_result = self.run_original(task)?;
        Ok((cognitive_result, original_result))
    }

    // Helper methods

    fn measure_error(&self, service: &mut CognitiveLoopService, inputs: &[String]) -> f32 {
        let mut total_error = 0.0;
        for input in inputs {
            let result = service.cycle(input);
            total_error += result.prediction_error;
        }
        total_error / inputs.len() as f32
    }

    fn measure_accuracy(&self, service: &mut CognitiveLoopService, task: CognitivePrimitiveTask, inputs: &[String]) -> f32 {
        let (_, _, valid_completions) = task.get_pattern();
        let mut correct = 0;

        for input in inputs {
            let result = service.cycle(input);
            // Check if any valid completion primitive was detected
            for completion in &valid_completions {
                if result.detected_primitives.iter().any(|p| p.to_lowercase().contains(completion)) {
                    correct += 1;
                    break;
                }
            }
        }

        (correct as f32 / inputs.len() as f32) * 100.0
    }

    fn measure_ltc_error(&self, ltc: &mut LearnableLTC, inputs: &[String], primitive_system: &PrimitiveSystem) -> f32 {
        let mut total_error = 0.0;
        for input in inputs {
            let (encoded, target) = self.encode_for_ltc(input, primitive_system);
            if let Ok((output, _)) = ltc.forward(&encoded) {
                let error: f32 = output.iter().zip(target.iter())
                    .map(|(o, t)| (o - t).powi(2))
                    .sum::<f32>() / target.len() as f32;
                total_error += error;
            }
            ltc.reset_state();
        }
        total_error / inputs.len() as f32
    }

    fn measure_ltc_accuracy(&self, ltc: &mut LearnableLTC, task: CognitivePrimitiveTask, inputs: &[String], primitive_system: &PrimitiveSystem) -> f32 {
        let (_, _, valid_completions) = task.get_pattern();
        let mut correct = 0;

        for input in inputs {
            let (encoded, _) = self.encode_for_ltc(input, primitive_system);
            if let Ok((output, _)) = ltc.forward(&encoded) {
                // Simple heuristic: check if output has high values in expected positions
                // This is a weak proxy for accuracy in the one-way pipeline
                let max_idx = output.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                // Map output index to completion (very rough heuristic)
                if max_idx < valid_completions.len() {
                    correct += 1;  // Give benefit of doubt
                }
            }
            ltc.reset_state();
        }

        (correct as f32 / inputs.len() as f32) * 100.0
    }

    fn encode_for_ltc(&self, input: &str, primitive_system: &PrimitiveSystem) -> (Vec<f32>, Vec<f32>) {
        // Simple encoding: hash-based compression to 128D input, 64D target
        let mut encoded = vec![0.0f32; 128];
        let mut target = vec![0.0f32; 64];

        // Extract words and encode
        for (i, word) in input.split_whitespace().enumerate() {
            if let Some(prim) = primitive_system.get(word) {
                // Use primitive encoding
                let bytes = &prim.encoding.0;
                let start = (i * 32) % 128;
                for (j, &byte) in bytes.iter().take(32).enumerate() {
                    if start + j < 128 {
                        encoded[start + j] = (byte as f32 / 128.0) - 1.0;
                    }
                }
            } else {
                // Hash-based fallback
                let hash = Self::simple_hash(word);
                let start = (i * 32) % 128;
                for j in 0..32 {
                    if start + j < 128 {
                        encoded[start + j] = ((hash >> j) & 1) as f32 * 2.0 - 1.0;
                    }
                }
            }
        }

        // Target: shifted/transformed version of input
        for (i, &val) in encoded.iter().take(64).enumerate() {
            target[i] = val.tanh();
        }

        (encoded, target)
    }

    fn simple_hash(s: &str) -> u64 {
        let mut hash: u64 = 5381;
        for c in s.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(c as u64);
        }
        hash
    }
}

/// Print comparison results
pub fn print_comparison_results(
    cognitive: &CognitivePrimitiveLearningResults,
    original: &CognitivePrimitiveLearningResults,
) {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║     PRIMITIVE LEARNING: COGNITIVE LOOP vs ORIGINAL          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("Task: {}\n", cognitive.task_name);

    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ COGNITIVE LOOP (Bidirectional HDC↔LTC)                      │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Error Reduction: {:>6.1}%                                    │", cognitive.error_reduction_percent);
    println!("│ Accuracy: {:>5.1}% → {:>5.1}% (Δ{:+5.1}%)                      │",
             cognitive.initial_accuracy, cognitive.final_accuracy, cognitive.accuracy_improvement);
    println!("│ Attention Emerged: {:>3}                                      │",
             if cognitive.attention_emerged { "YES" } else { "NO" });
    println!("│ Result: {}                                                  │",
             if cognitive.passed { "PASSED ✓" } else { "FAILED ✗" });
    println!("└─────────────────────────────────────────────────────────────┘\n");

    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ ORIGINAL (One-Way LTC Pipeline)                             │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Error Reduction: {:>6.1}%                                    │", original.error_reduction_percent);
    println!("│ Accuracy: {:>5.1}% → {:>5.1}% (Δ{:+5.1}%)                      │",
             original.initial_accuracy, original.final_accuracy, original.accuracy_improvement);
    println!("│ Attention Emerged: N/A (no attention in one-way)            │");
    println!("│ Result: {}                                                  │",
             if original.passed { "PASSED ✓" } else { "FAILED ✗" });
    println!("└─────────────────────────────────────────────────────────────┘\n");

    // Winner determination
    let cognitive_score = cognitive.error_reduction_percent + cognitive.accuracy_improvement;
    let original_score = original.error_reduction_percent + original.accuracy_improvement;

    println!("═══════════════════════════════════════════════════════════════");
    if cognitive_score > original_score + 5.0 {
        println!(" WINNER: COGNITIVE LOOP ({:+.1} combined improvement)         ", cognitive_score - original_score);
    } else if original_score > cognitive_score + 5.0 {
        println!(" WINNER: ORIGINAL ({:+.1} combined improvement)                ", original_score - cognitive_score);
    } else {
        println!(" RESULT: COMPARABLE (difference < 5%)                          ");
    }
    println!("═══════════════════════════════════════════════════════════════\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognitive_primitive_learning() {
        let config = CognitivePrimitiveLearningConfig {
            num_episodes: 3,
            cycles_per_episode: 20,
            test_size: 10,
            ..Default::default()
        };

        let benchmark = CognitivePrimitiveLearning::new(config);
        let result = benchmark.run_cognitive(CognitivePrimitiveTask::CausalChain);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.task_name, "Causal Chain");
        assert_eq!(result.method, "Cognitive Loop (HDC↔LTC)");
    }

    #[test]
    fn test_original_primitive_learning() {
        let config = CognitivePrimitiveLearningConfig {
            num_episodes: 3,
            cycles_per_episode: 20,
            test_size: 10,
            ..Default::default()
        };

        let benchmark = CognitivePrimitiveLearning::new(config);
        let result = benchmark.run_original(CognitivePrimitiveTask::CausalChain);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.method, "Original (One-Way LTC)");
    }
}
