//! # Primitive Learning Gate - HDC+LTC Combined Learning
//!
//! This benchmark tests whether the system can learn to predict and generate
//! sequences of semantic primitives - the bridge between HDC pattern matching
//! and LTC temporal learning.
//!
//! ## Hypothesis
//!
//! If HDC provides semantic structure and LTC provides temporal learning,
//! then learning on primitive sequences should be more effective than
//! learning on raw numerical patterns.
//!
//! ## Tasks
//!
//! 1. **Reasoning Chain Completion**: Learn to complete "CAUSE → ? → EFFECT"
//! 2. **Primitive Transformation**: Learn mappings like "ACTION → RESULT"
//! 3. **Compositional Pattern**: Learn "A ∘ B → C" structures
//! 4. **Analogy Completion**: If X:Y then A:?

use symthaea_core::hdc::binary_hv::HV16;
use symthaea_core::hdc::primitive_system::PrimitiveSystem;
use crate::learnable_ltc::{LearnableLTC, LearnableLTCConfig};
use serde::{Serialize, Deserialize};
use std::time::Instant;
use rand::Rng;

/// Configuration for primitive learning experiments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimitiveLearningConfig {
    /// Number of training episodes
    pub num_episodes: usize,
    /// Examples per episode
    pub examples_per_episode: usize,
    /// Test set size
    pub test_size: usize,
    /// LTC configuration
    pub ltc_config: LearnableLTCConfig,
    /// Number of primitives to use in sequences
    pub sequence_length: usize,
}

impl Default for PrimitiveLearningConfig {
    fn default() -> Self {
        Self {
            num_episodes: 20,
            examples_per_episode: 50,
            test_size: 25,
            ltc_config: LearnableLTCConfig {
                input_dim: 128,  // Compressed primitive encoding
                output_dim: 64,  // Prediction target
                num_neurons: 128,
                learning_rate: 0.01,
                tau_bounds: (0.5, 5.0),
                ..Default::default()
            },
            sequence_length: 3,
        }
    }
}

/// A primitive reasoning pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningPattern {
    /// Pattern name
    pub name: String,
    /// Template: e.g., ["CAUSE", "_", "EFFECT"]
    pub template: Vec<String>,
    /// Valid completions for the blank
    pub valid_completions: Vec<String>,
}

/// Task types for primitive learning
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PrimitiveTask {
    /// Complete a causal chain: CAUSE → ? → EFFECT
    CausalChain,
    /// Complete logical inference: PREMISE → ? → CONCLUSION
    LogicalInference,
    /// Complete action sequence: GOAL → ? → RESULT
    ActionSequence,
    /// Complete temporal relation: BEFORE → ? → AFTER
    TemporalOrder,
    /// Complete quantity relation: MORE → ? → LESS
    QuantityRelation,
}

impl PrimitiveTask {
    /// Get the reasoning pattern for this task
    pub fn get_pattern(&self) -> ReasoningPattern {
        match self {
            PrimitiveTask::CausalChain => ReasoningPattern {
                name: "Causal Chain".to_string(),
                template: vec!["CAUSE".to_string(), "_".to_string(), "EFFECT".to_string()],
                valid_completions: vec![
                    "ACTION".to_string(),
                    "PROCESS".to_string(),
                    "MECHANISM".to_string(),
                ],
            },
            PrimitiveTask::LogicalInference => ReasoningPattern {
                name: "Logical Inference".to_string(),
                template: vec!["TRUE".to_string(), "_".to_string(), "CONSEQUENCE".to_string()],
                valid_completions: vec![
                    "IMPLIES".to_string(),
                    "ENTAILS".to_string(),
                    "THEREFORE".to_string(),
                ],
            },
            PrimitiveTask::ActionSequence => ReasoningPattern {
                name: "Action Sequence".to_string(),
                template: vec!["WANT".to_string(), "_".to_string(), "HAVE".to_string()],
                valid_completions: vec![
                    "DO".to_string(),
                    "MAKE".to_string(),
                    "GET".to_string(),
                ],
            },
            PrimitiveTask::TemporalOrder => ReasoningPattern {
                name: "Temporal Order".to_string(),
                template: vec!["BEFORE".to_string(), "_".to_string(), "AFTER".to_string()],
                valid_completions: vec![
                    "NOW".to_string(),
                    "DURING".to_string(),
                    "WHEN".to_string(),
                ],
            },
            PrimitiveTask::QuantityRelation => ReasoningPattern {
                name: "Quantity Relation".to_string(),
                template: vec!["MORE".to_string(), "_".to_string(), "LESS".to_string()],
                valid_completions: vec![
                    "SAME".to_string(),
                    "PART".to_string(),
                    "SOME".to_string(),
                ],
            },
        }
    }
}

/// Results from one primitive learning episode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimitiveLearningEpisode {
    pub episode: usize,
    pub train_loss: f32,
    pub test_loss: f32,
    pub completion_accuracy: f32,
    pub tau_mean: f32,
    pub duration_ms: u64,
}

/// Complete results from primitive learning experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimitiveLearningResults {
    pub task_name: String,
    pub episodes: Vec<PrimitiveLearningEpisode>,
    pub initial_accuracy: f32,
    pub final_accuracy: f32,
    pub accuracy_improvement: f32,
    pub passed: bool,
    pub reasoning: String,
    pub total_time_ms: u64,
}

/// Primitive Learning Gate - tests HDC+LTC combined learning
pub struct PrimitiveLearningGate {
    config: PrimitiveLearningConfig,
    primitive_system: &'static PrimitiveSystem,
    ltc: LearnableLTC,
    // Cache of primitive encodings (compressed to 128D for LTC input)
    primitive_cache: std::collections::HashMap<String, Vec<f32>>,
}

impl PrimitiveLearningGate {
    pub fn new(config: PrimitiveLearningConfig) -> anyhow::Result<Self> {
        let ltc = LearnableLTC::new(config.ltc_config.clone())?;
        let primitive_system = PrimitiveSystem::global();

        Ok(Self {
            config,
            primitive_system,
            ltc,
            primitive_cache: std::collections::HashMap::new(),
        })
    }

    /// Encode a primitive name to a vector for LTC input
    fn encode_primitive(&mut self, name: &str) -> Vec<f32> {
        if let Some(cached) = self.primitive_cache.get(name) {
            return cached.clone();
        }

        // Get HV16 encoding from primitive system
        let hv = if let Some(prim) = self.primitive_system.get(name) {
            prim.encoding
        } else {
            // Fallback: hash-based encoding
            HV16::random(Self::name_to_seed(name))
        };

        // Compress 2048 bytes to 128 floats by averaging groups
        // HV16 is a newtype wrapping [u8; 2048], access via .0
        let bytes = &hv.0;
        let mut compressed = vec![0.0f32; 128];
        let group_size = bytes.len() / 128;  // 16 bytes per group

        for (i, chunk) in bytes.chunks(group_size).enumerate() {
            if i < 128 {
                // Count set bits in this chunk
                let mut popcount: u32 = 0;
                for byte in chunk {
                    popcount += byte.count_ones();
                }
                // Normalize to [-1, 1] range
                compressed[i] = (popcount as f32 / (group_size * 8) as f32) * 2.0 - 1.0;
            }
        }

        self.primitive_cache.insert(name.to_string(), compressed.clone());
        compressed
    }

    fn name_to_seed(name: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        name.hash(&mut hasher);
        hasher.finish()
    }

    /// Generate training example for a reasoning pattern
    fn generate_example(&mut self, pattern: &ReasoningPattern) -> (Vec<f32>, Vec<f32>) {
        let mut rng = rand::thread_rng();

        // Encode context (primitives around the blank)
        let mut input = vec![0.0f32; self.config.ltc_config.input_dim];

        for (i, part) in pattern.template.iter().enumerate() {
            if part != "_" {
                let encoding = self.encode_primitive(part);
                // Place each primitive in a section of the input
                let start = i * 32;
                for (j, &val) in encoding.iter().take(32).enumerate() {
                    if start + j < input.len() {
                        input[start + j] = val;
                    }
                }
            }
        }

        // Pick a random valid completion as target
        let target_name = &pattern.valid_completions[rng.gen_range(0..pattern.valid_completions.len())];
        let target_encoding = self.encode_primitive(target_name);

        // Compress target to output_dim
        let mut target = vec![0.0f32; self.config.ltc_config.output_dim];
        for (i, &val) in target_encoding.iter().take(target.len()).enumerate() {
            target[i] = val;
        }

        (input, target)
    }

    /// Run experiment on a single task
    pub fn run_task(&mut self, task: PrimitiveTask) -> anyhow::Result<PrimitiveLearningResults> {
        let start_time = Instant::now();
        let pattern = task.get_pattern();

        // Generate test set
        let test_set: Vec<(Vec<f32>, Vec<f32>)> = (0..self.config.test_size)
            .map(|_| self.generate_example(&pattern))
            .collect();

        // Measure initial accuracy
        let initial_accuracy = self.evaluate_accuracy(&test_set)?;

        let mut episodes = Vec::new();

        // Training episodes
        for episode in 0..self.config.num_episodes {
            let episode_start = Instant::now();

            // Generate training examples
            let train_set: Vec<(Vec<f32>, Vec<f32>)> = (0..self.config.examples_per_episode)
                .map(|_| self.generate_example(&pattern))
                .collect();

            // Train
            let mut train_loss = 0.0;
            for (input, target) in &train_set {
                let loss = self.ltc.train_step(input, target)?;
                train_loss += loss;
                self.ltc.reset_state();
            }
            train_loss /= train_set.len() as f32;

            // Evaluate
            let test_loss = self.evaluate_loss(&test_set)?;
            let completion_accuracy = self.evaluate_accuracy(&test_set)?;

            let (tau_mean, _, _, _) = self.ltc.get_tau_distribution();

            episodes.push(PrimitiveLearningEpisode {
                episode,
                train_loss,
                test_loss,
                completion_accuracy,
                tau_mean,
                duration_ms: episode_start.elapsed().as_millis() as u64,
            });
        }

        // Final accuracy
        let final_accuracy = self.evaluate_accuracy(&test_set)?;
        let accuracy_improvement = final_accuracy - initial_accuracy;

        // Pass criteria: >20% accuracy improvement (since task is harder)
        let passed = accuracy_improvement > 20.0 && final_accuracy > 40.0;

        let reasoning = format!(
            "Initial: {:.1}% → Final: {:.1}% (Δ = {:.1}%)",
            initial_accuracy, final_accuracy, accuracy_improvement
        );

        Ok(PrimitiveLearningResults {
            task_name: pattern.name,
            episodes,
            initial_accuracy,
            final_accuracy,
            accuracy_improvement,
            passed,
            reasoning,
            total_time_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    /// Evaluate loss on dataset
    fn evaluate_loss(&mut self, dataset: &[(Vec<f32>, Vec<f32>)]) -> anyhow::Result<f32> {
        let mut total_loss = 0.0;

        for (input, target) in dataset {
            let (output, _) = self.ltc.forward(input)?;

            let loss: f32 = output.iter().zip(target.iter())
                .map(|(o, t)| (o - t).powi(2))
                .sum::<f32>() / target.len() as f32;

            total_loss += loss;
            self.ltc.reset_state();
        }

        Ok(total_loss / dataset.len() as f32)
    }

    /// Evaluate completion accuracy
    fn evaluate_accuracy(&mut self, dataset: &[(Vec<f32>, Vec<f32>)]) -> anyhow::Result<f32> {
        let mut correct = 0;

        for (input, target) in dataset {
            let (output, _) = self.ltc.forward(input)?;

            // Check if output is close enough to target
            let similarity: f32 = output.iter().zip(target.iter())
                .map(|(o, t)| o * t)
                .sum::<f32>() / (output.len() as f32);

            if similarity > 0.5 {
                correct += 1;
            }

            self.ltc.reset_state();
        }

        Ok(correct as f32 / dataset.len() as f32 * 100.0)
    }

    /// Run all primitive learning tasks
    pub fn run_all_tasks(&mut self) -> anyhow::Result<Vec<PrimitiveLearningResults>> {
        let tasks = vec![
            PrimitiveTask::CausalChain,
            PrimitiveTask::LogicalInference,
            PrimitiveTask::ActionSequence,
            PrimitiveTask::TemporalOrder,
            PrimitiveTask::QuantityRelation,
        ];

        let mut results = Vec::new();
        for task in tasks {
            // Reset LTC between tasks
            self.ltc = LearnableLTC::new(self.config.ltc_config.clone())?;
            self.primitive_cache.clear();

            let result = self.run_task(task)?;
            results.push(result);
        }

        Ok(results)
    }
}
