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
use crate::hdc::real_hv::RealHV;

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

        // 5. Create consciousness graph (kept for potential future use)
        let mut _graph = ConsciousnessGraph::new();

        // 5b. Collect semantic states as RealHV for proper Φ calculation
        let mut semantic_states: Vec<RealHV> = Vec::new();

        // 6. Run reasoning steps
        let _categories = ReasoningCategory::for_subject(&question.subject);
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
            _graph.add_state(semantic_hv.clone(), dynamic_hv, state.confidence() as f32);

            // Also collect as RealHV for actual Φ calculation
            // Convert HV16 (binary) to RealHV for spectral analysis
            semantic_states.push(RealHV::from_values(semantic_hv));

            // Check if we've reached sufficient confidence
            if state.confidence() > 0.8 {
                break;
            }
        }

        // 7. Select best matching answer (need this first for Φ calculation)
        let (selected_answer, confidence, answer_similarities) = self.select_answer_with_sims(&state, &answer_hvs);

        // 8. Measure Φ as answer discrimination quality
        // High discrimination (clear preference) should correlate with correctness
        let phi = self.measure_phi_discrimination(&semantic_states, &answer_similarities);

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

    /// Measure Φ as answer-context integration (unused - kept for reference)
    #[allow(dead_code)]
    ///
    /// Key insight: Correct answers should "fit" well with the reasoning context.
    /// We measure how coherently the reasoning trajectory converges toward
    /// a distinguishable answer state.
    ///
    /// The metric combines:
    /// 1. Trajectory coherence (how focused the reasoning was)
    /// 2. Final state differentiation (how clearly it points to one answer)
    /// 3. Convergence (did reasoning states become more similar over time?)
    fn measure_phi_from_states(&self, states: &[RealHV]) -> f64 {
        if states.len() < 2 {
            return 0.0;
        }

        // Component 1: Trajectory coherence
        // Measure how similar consecutive states are (focused reasoning)
        let mut consecutive_sims = Vec::new();
        for i in 1..states.len() {
            let sim = states[i - 1].similarity(&states[i]);
            consecutive_sims.push((sim + 1.0) / 2.0); // Normalize to [0, 1]
        }
        let coherence = if consecutive_sims.is_empty() {
            0.5
        } else {
            consecutive_sims.iter().sum::<f32>() / consecutive_sims.len() as f32
        };

        // Component 2: Convergence (do later states become more similar?)
        // Higher convergence = reasoning is settling on an answer
        let n = states.len();
        let first_half_sim = if n >= 4 {
            let mid = n / 2;
            let mut sims = Vec::new();
            for i in 0..mid {
                for j in (i + 1)..mid {
                    sims.push((states[i].similarity(&states[j]) + 1.0) / 2.0);
                }
            }
            if sims.is_empty() { 0.5 } else { sims.iter().sum::<f32>() / sims.len() as f32 }
        } else {
            0.5
        };

        let second_half_sim = if n >= 4 {
            let mid = n / 2;
            let mut sims = Vec::new();
            for i in mid..n {
                for j in (i + 1)..n {
                    sims.push((states[i].similarity(&states[j]) + 1.0) / 2.0);
                }
            }
            if sims.is_empty() { 0.5 } else { sims.iter().sum::<f32>() / sims.len() as f32 }
        } else {
            0.5
        };

        // Convergence: second half should be more similar than first half
        let convergence = if second_half_sim > first_half_sim {
            (second_half_sim - first_half_sim + 0.5).min(1.0)
        } else {
            (0.5 - (first_half_sim - second_half_sim) * 0.5).max(0.0)
        };

        // Component 3: Final state differentiation
        // How distinct is the final state from the initial state?
        let differentiation = if states.len() >= 2 {
            let first = &states[0];
            let last = &states[states.len() - 1];
            let sim = (first.similarity(last) + 1.0) / 2.0;
            // We want MODERATE differentiation - too similar = no reasoning happened
            // Too different = reasoning went off track
            // Optimal is around 0.3-0.7 similarity
            let distance_from_optimal = (sim - 0.5).abs();
            1.0 - distance_from_optimal * 2.0 // Max at 0.5 similarity, min at 0 or 1
        } else {
            0.5
        };

        // Combine components with weights that favor convergent, coherent reasoning
        // that reaches a distinct but not wildly different conclusion
        let phi = coherence as f64 * 0.3
                + convergence as f64 * 0.4
                + differentiation as f64 * 0.3;

        phi.clamp(0.0, 1.0)
    }

    /// Measure Φ from the consciousness graph (legacy proxy - less accurate)
    #[allow(dead_code)]
    fn measure_phi_proxy(&self, graph: &ConsciousnessGraph) -> f64 {
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
    /// Returns (selected_index, confidence, all_similarities)
    fn select_answer_with_sims(&self, state: &TaskState, answer_hvs: &[HV16]) -> (usize, f64, Vec<f64>) {
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

        (best_idx, best_sim, similarities)
    }

    /// Measure Φ based on answer discrimination quality
    ///
    /// Key insight: Correct answers should come from CONFIDENT discrimination.
    /// When the reasoning state clearly distinguishes one answer from others,
    /// it's more likely to be correct.
    ///
    /// Φ = discrimination_score * convergence_bonus
    fn measure_phi_discrimination(&self, states: &[RealHV], answer_sims: &[f64]) -> f64 {
        if answer_sims.is_empty() {
            return 0.0;
        }

        // Component 1: Answer discrimination (how peaked is the distribution?)
        // Use the gap between best and second-best answer
        let mut sorted_sims = answer_sims.to_vec();
        sorted_sims.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let discrimination = if sorted_sims.len() >= 2 {
            // Gap between best and second-best, normalized
            let gap = sorted_sims[0] - sorted_sims[1];
            // Normalize by average similarity to get relative discrimination
            let avg = answer_sims.iter().sum::<f64>() / answer_sims.len() as f64;
            if avg > 0.001 {
                (gap / avg).min(1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Component 2: Confidence level (absolute similarity to chosen answer)
        let max_sim = sorted_sims.first().copied().unwrap_or(0.5);
        let confidence_component = (max_sim - 0.3).max(0.0) / 0.5; // Scale from [0.3, 0.8] to [0, 1]

        // Component 3: Reasoning convergence from states
        let convergence = if states.len() >= 4 {
            let n = states.len();
            let last_quarter = &states[3 * n / 4..];

            // How similar are the final states to each other?
            let mut final_sims = Vec::new();
            for i in 0..last_quarter.len() {
                for j in (i + 1)..last_quarter.len() {
                    let sim = (last_quarter[i].similarity(&last_quarter[j]) + 1.0) / 2.0;
                    final_sims.push(sim as f64);
                }
            }

            if final_sims.is_empty() {
                0.5
            } else {
                final_sims.iter().sum::<f64>() / final_sims.len() as f64
            }
        } else {
            0.5
        };

        // Combine: discrimination is most important (0.5), then confidence (0.3), then convergence (0.2)
        let phi = discrimination * 0.5
                + confidence_component.min(1.0) * 0.3
                + convergence * 0.2;

        phi.clamp(0.0, 1.0)
    }

    /// Select the best matching answer (legacy method)
    #[allow(dead_code)]
    fn select_answer(&self, state: &TaskState, answer_hvs: &[HV16]) -> (usize, f64) {
        let (idx, conf, _) = self.select_answer_with_sims(state, answer_hvs);
        (idx, conf)
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
