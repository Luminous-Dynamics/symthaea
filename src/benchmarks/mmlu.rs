//! # MMLU Benchmark Adapter for Φ-Accuracy Correlation
//!
//! This module implements an adapter for the MMLU (Massive Multitask Language Understanding)
//! benchmark, enabling measurement of the correlation between integrated information (Φ)
//! and reasoning accuracy.
//!
//! ## Purpose
//!
//! The Φ-Gate experiment tests the hypothesis:
//! > Does Φ (integrated information) correlate with reasoning ability?
//! > Target: Pearson r > 0.3
//!
//! ## Architecture
//!
//! ```text
//! MMLU Dataset → MMLUQuestion → TaskState → Reasoning Chain → Answer
//!                                   ↓
//!                          ConsciousnessGraph → Φ measurement
//!                                   ↓
//!                          (accuracy, Φ) pairs → correlation analysis
//! ```

use crate::hdc::binary_hv::HV16;
use crate::hdc::bootstrapping::{PrimitiveBootstrapper, ReasoningCategory};
use crate::domains::task::{TaskState, TaskAction, TaskDynamics};
use crate::consciousness::ConsciousnessGraph;
use crate::core::domain_traits::WorldModel;
use crate::hdc::phi_real::RealPhiCalculator;

use serde::{Serialize, Deserialize};

/// An MMLU question with multiple choice answers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MMLUQuestion {
    /// The question text
    pub question: String,
    /// Subject/category (e.g., "physics", "philosophy")
    pub subject: String,
    /// The four answer choices (A, B, C, D)
    pub choices: [String; 4],
    /// The correct answer index (0-3)
    pub correct_answer: usize,
    /// Optional explanation
    pub explanation: Option<String>,
}

impl MMLUQuestion {
    /// Create a new MMLU question
    pub fn new(
        question: String,
        subject: String,
        choices: [String; 4],
        correct_answer: usize,
    ) -> Self {
        assert!(correct_answer < 4, "correct_answer must be 0-3");
        Self {
            question,
            subject,
            choices,
            correct_answer,
            explanation: None,
        }
    }

    /// Add an explanation
    pub fn with_explanation(mut self, explanation: String) -> Self {
        self.explanation = Some(explanation);
        self
    }
}

/// Result of reasoning on a single question
#[derive(Debug, Clone)]
pub struct ReasoningResult {
    /// The question that was answered
    pub question: MMLUQuestion,
    /// The selected answer (0-3)
    pub selected_answer: usize,
    /// Whether the answer was correct
    pub is_correct: bool,
    /// Φ (integrated information) measured during reasoning
    pub phi: f64,
    /// Number of reasoning steps taken
    pub reasoning_steps: usize,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Time taken in milliseconds
    pub time_ms: u128,
}

/// Aggregate results from a benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// All individual reasoning results
    pub results: Vec<ReasoningResult>,
    /// Overall accuracy
    pub accuracy: f64,
    /// Mean Φ across all questions
    pub mean_phi: f64,
    /// Std deviation of Φ
    pub std_phi: f64,
    /// Pearson correlation between Φ and accuracy
    pub phi_accuracy_correlation: f64,
    /// Whether the Φ-Gate was passed (r > 0.3)
    pub phi_gate_passed: bool,
}

impl BenchmarkResults {
    /// Calculate aggregate statistics from results
    pub fn from_results(results: Vec<ReasoningResult>) -> Self {
        let n = results.len() as f64;
        if n == 0.0 {
            return Self {
                results,
                accuracy: 0.0,
                mean_phi: 0.0,
                std_phi: 0.0,
                phi_accuracy_correlation: 0.0,
                phi_gate_passed: false,
            };
        }

        // Calculate accuracy
        let correct_count = results.iter().filter(|r| r.is_correct).count() as f64;
        let accuracy = correct_count / n;

        // Calculate mean Φ
        let mean_phi = results.iter().map(|r| r.phi).sum::<f64>() / n;

        // Calculate std Φ
        let variance: f64 = results.iter()
            .map(|r| (r.phi - mean_phi).powi(2))
            .sum::<f64>() / n;
        let std_phi = variance.sqrt();

        // Calculate Pearson correlation between Φ and correctness
        let phi_values: Vec<f64> = results.iter().map(|r| r.phi).collect();
        let correct_values: Vec<f64> = results.iter().map(|r| if r.is_correct { 1.0 } else { 0.0 }).collect();

        let phi_accuracy_correlation = pearson_correlation(&phi_values, &correct_values);

        // Check Φ-Gate threshold
        let phi_gate_passed = phi_accuracy_correlation > 0.3;

        Self {
            results,
            accuracy,
            mean_phi,
            std_phi,
            phi_accuracy_correlation,
            phi_gate_passed,
        }
    }

    /// Generate a summary report
    pub fn summary(&self) -> String {
        format!(
            "=== MMLU Benchmark Results ===\n\
             Questions: {}\n\
             Accuracy: {:.1}%\n\
             Mean Φ: {:.4}\n\
             Std Φ: {:.4}\n\
             Φ-Accuracy Correlation: r = {:.4}\n\
             Φ-Gate (r > 0.3): {}\n",
            self.results.len(),
            self.accuracy * 100.0,
            self.mean_phi,
            self.std_phi,
            self.phi_accuracy_correlation,
            if self.phi_gate_passed { "PASSED ✓" } else { "FAILED ✗" }
        )
    }
}

/// Calculate Pearson correlation coefficient
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 || x.len() != y.len() {
        return 0.0;
    }

    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }

    cov / denom
}

/// The MMLU benchmark runner
pub struct MMLUBenchmark {
    /// Primitive bootstrapper for reasoning setup
    bootstrapper: PrimitiveBootstrapper,
    /// Task dynamics for reasoning steps
    dynamics: TaskDynamics,
    /// Φ calculator
    phi_calculator: RealPhiCalculator,
    /// Maximum reasoning steps per question
    max_steps: usize,
    /// Similarity threshold for answer matching
    answer_threshold: f64,
}

impl MMLUBenchmark {
    /// Create a new MMLU benchmark
    pub fn new() -> Self {
        Self {
            bootstrapper: PrimitiveBootstrapper::new(),
            dynamics: TaskDynamics::new(),
            phi_calculator: RealPhiCalculator::new(),
            max_steps: 20,
            answer_threshold: 0.6,
        }
    }

    /// Configure maximum reasoning steps
    pub fn with_max_steps(mut self, steps: usize) -> Self {
        self.max_steps = steps;
        self
    }

    /// Encode a text string into an HV16 hypervector
    fn encode_text(&self, text: &str) -> HV16 {
        // Simple character-based encoding (could be enhanced with embeddings)
        let mut hv = HV16::random(text_hash(text));

        // Bundle with word-level encodings for richer representation
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() > 1 {
            let mut all_hvs = vec![hv.clone()];
            for w in words.iter().take(20) {
                all_hvs.push(HV16::random(text_hash(w)));
            }
            hv = HV16::bundle(&all_hvs);
        }

        hv
    }

    /// Run reasoning on a single MMLU question
    pub fn reason_on_question(&self, question: &MMLUQuestion) -> ReasoningResult {
        let start = std::time::Instant::now();

        // 1. Encode the question as HV16
        let question_hv = self.encode_text(&question.question);

        // 2. Encode answer choices
        let answer_hvs: Vec<HV16> = question.choices.iter()
            .map(|choice| self.encode_text(choice))
            .collect();

        // 3. Bootstrap working memory with relevant primitives
        let initial_memory = self.bootstrapper.bootstrap_working_memory(
            &question.subject,
            &question_hv
        );

        // 4. Create initial task state
        let mut state = TaskState::from_question(initial_memory);

        // 5. Create consciousness graph for Φ measurement
        let mut graph = ConsciousnessGraph::new();

        // 6. Run reasoning steps
        let categories = ReasoningCategory::for_subject(&question.subject);
        let primitives = self.bootstrapper.primitives_for_subject(&question.subject);

        let mut step_count = 0;
        for step in 0..self.max_steps {
            step_count = step + 1;

            // Select a reasoning primitive based on current state
            let action = self.select_action(&state, &primitives, step);

            // Apply the action
            state = self.dynamics.predict(&state, &action);

            // Record state in consciousness graph
            let semantic_hv: Vec<f32> = state.working_memory().0.iter()
                .map(|&b| b as f32 / 255.0)
                .collect();
            let dynamic_hv: Vec<f32> = vec![state.confidence() as f32; 32];
            graph.add_state(semantic_hv, dynamic_hv, state.confidence() as f32);

            // Check if we've reached sufficient confidence
            if state.confidence() > 0.8 {
                break;
            }
        }

        // 7. Measure Φ from consciousness graph
        let phi = self.measure_phi(&graph);

        // 8. Select best matching answer
        let (selected_answer, confidence) = self.select_answer(&state, &answer_hvs);

        let time_ms = start.elapsed().as_millis();

        ReasoningResult {
            question: question.clone(),
            selected_answer,
            is_correct: selected_answer == question.correct_answer,
            phi,
            reasoning_steps: step_count,
            confidence,
            time_ms,
        }
    }

    /// Select an action based on current state and available primitives
    fn select_action(&self, state: &TaskState, primitives: &[(String, HV16)], step: usize) -> TaskAction {
        // Cycle through primitives based on step number
        if primitives.is_empty() {
            return TaskAction::Evaluate;
        }

        let idx = step % primitives.len();
        let (name, _) = &primitives[idx];

        match step % 4 {
            0 => TaskAction::ApplyPrimitive(name.clone()),
            1 => TaskAction::QueryKnowledge(name.clone()),
            2 => TaskAction::Attend(name.clone()),
            _ => TaskAction::Evaluate,
        }
    }

    /// Measure Φ from the consciousness graph
    fn measure_phi(&self, graph: &ConsciousnessGraph) -> f64 {
        // Use graph properties as a proxy for Φ
        let size = graph.size();
        if size < 2 {
            return 0.0;
        }

        // Combine multiple factors for Φ estimation:
        // 1. Current consciousness level
        let consciousness = graph.current_consciousness() as f64;

        // 2. Graph complexity (edges per node) - proxy for integration
        let complexity = graph.complexity() as f64;

        // 3. Self-reference count (autopoietic measure)
        let self_loops = graph.self_loop_count() as f64;
        let loop_ratio = if size > 0 {
            self_loops / size as f64
        } else {
            0.0
        };

        // Combined Φ proxy: weighted average of factors
        let phi = consciousness * 0.4 + complexity.min(1.0) * 0.3 + loop_ratio * 0.3;

        phi.clamp(0.0, 1.0)
    }

    /// Select the best matching answer based on working memory similarity
    fn select_answer(&self, state: &TaskState, answer_hvs: &[HV16]) -> (usize, f64) {
        let working_memory = state.working_memory();

        let similarities: Vec<f64> = answer_hvs.iter()
            .map(|ahv| {
                let sim = working_memory.similarity(ahv);
                // Normalize to [0, 1]
                (sim as f64 + 1.0) / 2.0
            })
            .collect();

        // Find best matching answer
        let mut best_idx = 0;
        let mut best_sim = similarities[0];
        for (i, &sim) in similarities.iter().enumerate() {
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        (best_idx, best_sim)
    }

    /// Run the full benchmark on a set of questions
    pub fn run_benchmark(&self, questions: &[MMLUQuestion]) -> BenchmarkResults {
        let results: Vec<ReasoningResult> = questions.iter()
            .map(|q| self.reason_on_question(q))
            .collect();

        BenchmarkResults::from_results(results)
    }
}

impl Default for MMLUBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple hash function for text to seed
fn text_hash(text: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

/// Generate sample MMLU questions for testing
pub fn sample_questions() -> Vec<MMLUQuestion> {
    vec![
        MMLUQuestion::new(
            "What is the result of 2 + 3?".to_string(),
            "mathematics".to_string(),
            ["4".to_string(), "5".to_string(), "6".to_string(), "7".to_string()],
            1, // B = 5
        ),
        MMLUQuestion::new(
            "If all A are B, and all B are C, then:".to_string(),
            "philosophy".to_string(),
            [
                "Some A are not C".to_string(),
                "All A are C".to_string(),
                "No A are C".to_string(),
                "Cannot determine".to_string(),
            ],
            1, // B = All A are C (transitive)
        ),
        MMLUQuestion::new(
            "Newton's first law states that an object at rest:".to_string(),
            "physics".to_string(),
            [
                "Will accelerate".to_string(),
                "Will remain at rest unless acted upon".to_string(),
                "Has no mass".to_string(),
                "Is always in motion".to_string(),
            ],
            1, // B
        ),
        MMLUQuestion::new(
            "The concept of utility in economics refers to:".to_string(),
            "economics".to_string(),
            [
                "Money saved".to_string(),
                "Satisfaction from consumption".to_string(),
                "Cost of production".to_string(),
                "Total revenue".to_string(),
            ],
            1, // B
        ),
        MMLUQuestion::new(
            "In psychology, 'cognitive dissonance' occurs when:".to_string(),
            "psychology".to_string(),
            [
                "Two people disagree".to_string(),
                "Beliefs and behaviors conflict".to_string(),
                "Memory fails".to_string(),
                "Perception is altered".to_string(),
            ],
            1, // B
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmlu_question_creation() {
        let q = MMLUQuestion::new(
            "What is 2+2?".to_string(),
            "mathematics".to_string(),
            ["3".to_string(), "4".to_string(), "5".to_string(), "6".to_string()],
            1,
        );
        assert_eq!(q.correct_answer, 1);
        assert_eq!(q.choices[1], "4");
    }

    #[test]
    fn test_pearson_correlation() {
        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 0.001, "Perfect positive correlation should be 1.0");

        // Perfect negative correlation
        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let r_neg = pearson_correlation(&x, &y_neg);
        assert!((r_neg + 1.0).abs() < 0.001, "Perfect negative correlation should be -1.0");

        // No correlation (random)
        let y_rand = vec![5.0, 3.0, 8.0, 1.0, 7.0];
        let r_rand = pearson_correlation(&x, &y_rand);
        assert!(r_rand.abs() < 0.5, "Random data should have low correlation");
    }

    #[test]
    fn test_benchmark_creation() {
        let benchmark = MMLUBenchmark::new();
        assert_eq!(benchmark.max_steps, 20);
    }

    #[test]
    fn test_text_encoding() {
        let benchmark = MMLUBenchmark::new();
        let hv1 = benchmark.encode_text("hello world");
        let hv2 = benchmark.encode_text("hello world");
        let hv3 = benchmark.encode_text("goodbye moon");

        // Same text should produce same encoding
        assert_eq!(hv1.hamming_distance(&hv2), 0);

        // Different text should produce different encoding
        assert!(hv1.hamming_distance(&hv3) > 0);
    }

    #[test]
    fn test_sample_questions() {
        let questions = sample_questions();
        assert!(!questions.is_empty());
        assert!(questions.len() >= 5);

        for q in &questions {
            assert!(q.correct_answer < 4);
            assert_eq!(q.choices.len(), 4);
        }
    }

    #[test]
    fn test_reason_on_single_question() {
        let benchmark = MMLUBenchmark::new().with_max_steps(5);
        let question = sample_questions()[0].clone();

        let result = benchmark.reason_on_question(&question);

        assert!(result.selected_answer < 4);
        assert!(result.phi >= 0.0);
        assert!(result.reasoning_steps > 0);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_benchmark_results() {
        let benchmark = MMLUBenchmark::new().with_max_steps(3);
        let questions = sample_questions();

        let results = benchmark.run_benchmark(&questions);

        assert_eq!(results.results.len(), questions.len());
        assert!(results.accuracy >= 0.0 && results.accuracy <= 1.0);
        assert!(results.mean_phi >= 0.0);

        let summary = results.summary();
        assert!(summary.contains("Accuracy:"));
        assert!(summary.contains("Mean Φ:"));
        assert!(summary.contains("Φ-Accuracy Correlation:"));
    }
}
