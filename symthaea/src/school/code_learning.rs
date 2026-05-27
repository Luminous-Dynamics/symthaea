// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Code Learning Loop — Phase 4 School Integration
//!
//! Maps School curriculum objectives to concrete code generation tasks,
//! runs them through CodeGenerator → CodeExecutor → mastery tracking.
//!
//! # Pipeline
//!
//! ```text
//! School::recommend_next()
//!     ↓ objective_to_spec()
//! CodeSpec (language, name, purpose, signature, constraints)
//!     ↓ CodeGenerator::generate()
//! GeneratedCode { source, plan_steps, phi_score, ... }
//!     ↓ CodeExecutor::execute_rust()
//! ExecutionResult { compiled, tests_passed/failed, surprise }
//!     ↓ try_auto_fix() + retry (up to MAX_RETRIES)
//! LessonOutcome { compiled, tests_passed, surprise, mastery_delta }
//!     ↓ distillation_target() → cache for Broca SSM
//! ```

use std::collections::HashMap;

use crate::language::code_executor::{CodeExecutor, ExecutionResult, try_auto_fix};
use crate::language::code_generator::{CodeContext, CodeGenerator};
use crate::language::code_intent::{CodeIntent, CodeSpec, CodeTarget, EntityKind};

/// Maximum auto-fix retry attempts per generation
const MAX_RETRIES: usize = 3;

/// Maximum LLM retry attempts when feeding back compiler errors
const MAX_LLM_RETRIES: usize = 2;

/// Minimum plan coverage to consider a generation "good enough" for mastery
const MIN_PLAN_COVERAGE: f32 = 0.5;

/// Energy cost thresholds for metabolic budgeting
const ENERGY_TIER1_NATIVE: f32 = 1.0;
const ENERGY_TIER2_LLM_CALL: f32 = 10.0;
const ENERGY_LLM_RETRY: f32 = 8.0;
const ENERGY_AUTO_FIX: f32 = 0.5;

/// Maximum distinct error patterns to track before evicting old ones
const MAX_ERROR_PATTERNS: usize = 128;

/// How many times an error must recur before flagging for pattern evolution
const ERROR_PATTERN_EVOLUTION_THRESHOLD: usize = 3;

// ═══════════════════════════════════════════════════════════════════════════════
// OBJECTIVE → SPEC MAPPING
// ═══════════════════════════════════════════════════════════════════════════════

/// A concrete code task derived from a curriculum objective.
#[derive(Debug, Clone)]
pub struct CodeLesson {
    /// Curriculum objective ID this lesson exercises
    pub objective_id: String,
    /// The code specification to generate
    pub spec: CodeSpec,
    /// Optional test source to validate the generated code
    pub test_source: Option<String>,
}

/// Result of running a single code lesson through the pipeline.
#[derive(Debug, Clone)]
pub struct LessonOutcome {
    /// Which objective this exercises
    pub objective_id: String,
    /// The generated source code
    pub source: String,
    /// Whether execution was simulated rather than actually compiled/tested
    pub simulated_execution: bool,
    /// Whether the code compiled
    pub compiled: bool,
    /// Tests passed count
    pub tests_passed: usize,
    /// Tests failed count
    pub tests_failed: usize,
    /// FEP surprise signal (0.0 = perfect, 1.0 = total failure)
    pub surprise: f32,
    /// Number of auto-fix retries used
    pub retries_used: usize,
    /// Plan coverage from the generator (0.0-1.0)
    pub plan_coverage: f32,
    /// Phi score from primitive execution
    pub phi_score: f32,
    /// Whether this was a Tier 2 (LLM-assisted) generation
    pub used_llm: bool,
    /// Number of LLM retries with error feedback
    pub llm_retries_used: usize,
    /// Eligible for distillation into Broca SSM
    pub distillation_eligible: bool,
    /// Metabolic energy spent on this lesson
    pub energy_spent: f32,
    /// Predicted quality before committing energy (Item #1: Lookahead)
    pub predicted_quality: Option<f32>,
    /// Actual quality computed after execution
    pub actual_quality: f32,
    /// Whether the prediction was a hallucination (predicted high, actual low)
    pub was_hallucination: bool,
}

impl LessonOutcome {
    /// Whether the lesson was fully successful
    pub fn is_success(&self) -> bool {
        !self.simulated_execution && self.compiled && self.tests_failed == 0
    }

    /// Mastery signal: 1.0 for perfect, scaled down for partial success
    pub fn mastery_signal(&self) -> f32 {
        if self.simulated_execution || !self.compiled {
            return 0.0;
        }
        let total_tests = self.tests_passed + self.tests_failed;
        if total_tests == 0 {
            // Compiled but no tests — partial credit
            0.5
        } else {
            self.tests_passed as f32 / total_tests as f32
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// METABOLIC ENERGY BUDGET
// ═══════════════════════════════════════════════════════════════════════════════

/// Metabolic energy budget for a learning session.
///
/// Models the cognitive cost of code generation: native emission is cheap,
/// LLM calls are expensive, retries add up. The agent should prefer native
/// patterns when possible and only invoke the LLM for genuinely hard tasks.
///
/// Inspired by biological metabolic constraints: the brain uses ~20% of
/// the body's energy despite being ~2% of its mass. Cognitive effort is
/// a limited resource that should be allocated wisely.
#[derive(Debug, Clone)]
pub struct MetabolicBudget {
    /// Total energy available for this session
    pub total_budget: f32,
    /// Energy spent so far
    pub spent: f32,
    /// Energy reserved for critical tasks (can't be spent on easy ones)
    pub reserved: f32,
}

impl MetabolicBudget {
    /// Create a new budget with the given total energy.
    pub fn new(total: f32) -> Self {
        Self {
            total_budget: total,
            spent: 0.0,
            reserved: total * 0.2, // Reserve 20% for hard tasks
        }
    }

    /// Default budget for a learning session (~20 lessons).
    pub fn default_session() -> Self {
        // Budget: 20 native + 5 LLM calls + 3 retries = ~100 energy units
        Self::new(100.0)
    }

    /// Available energy (total - spent - reserved)
    pub fn available(&self) -> f32 {
        (self.total_budget - self.spent - self.reserved).max(0.0)
    }

    /// Available including reserves (for critical tasks)
    pub fn available_with_reserves(&self) -> f32 {
        (self.total_budget - self.spent).max(0.0)
    }

    /// Whether we can afford a native generation
    pub fn can_afford_native(&self) -> bool {
        self.available() >= ENERGY_TIER1_NATIVE
    }

    /// Whether we can afford an LLM call (dips into reserves if needed)
    pub fn can_afford_llm(&self) -> bool {
        self.available_with_reserves() >= ENERGY_TIER2_LLM_CALL
    }

    /// Record energy expenditure
    pub fn spend(&mut self, amount: f32) {
        self.spent += amount;
    }

    /// Fraction of budget consumed (0.0 - 1.0+)
    pub fn utilization(&self) -> f32 {
        if self.total_budget > 0.0 {
            self.spent / self.total_budget
        } else {
            1.0
        }
    }

    /// Efficiency: mastery gained per unit energy spent
    pub fn efficiency(&self, mastery_gained: f32) -> f32 {
        if self.spent > 0.0 {
            mastery_gained / self.spent
        } else {
            0.0
        }
    }
}

impl Default for MetabolicBudget {
    fn default() -> Self {
        Self::default_session()
    }
}

/// Summary of a full learning session across multiple objectives.
#[derive(Debug, Clone, Default)]
pub struct SessionSummary {
    /// Total lessons attempted
    pub lessons_attempted: usize,
    /// Lessons whose execution was simulated rather than actually compiled/tested
    pub simulated_lessons: usize,
    /// Lessons that compiled
    pub lessons_compiled: usize,
    /// Lessons where all tests passed
    pub lessons_passed: usize,
    /// Total auto-fix retries across all lessons
    pub total_retries: usize,
    /// Total LLM retries across all lessons
    pub total_llm_retries: usize,
    /// Average surprise across all lessons
    pub avg_surprise: f32,
    /// Average plan coverage
    pub avg_plan_coverage: f32,
    /// Lessons eligible for distillation
    pub distillation_eligible: usize,
    /// Total metabolic energy spent
    pub total_energy_spent: f32,
    /// Per-objective outcomes
    pub outcomes: Vec<LessonOutcome>,
    /// Average prediction error across lessons with predictions
    pub avg_prediction_error: f32,
    /// Fraction of predictions that were hallucinations
    pub hallucination_rate: f32,
    /// Number of error patterns ready for native pattern evolution
    pub error_patterns_learned: usize,
}

impl SessionSummary {
    /// Fraction of lessons that were simulated rather than actually verified.
    pub fn simulated_rate(&self) -> f32 {
        if self.lessons_attempted == 0 {
            return 0.0;
        }
        self.simulated_lessons as f32 / self.lessons_attempted as f32 * 100.0
    }

    /// Compile rate as a percentage
    pub fn compile_rate(&self) -> f32 {
        if self.lessons_attempted == 0 {
            return 0.0;
        }
        self.lessons_compiled as f32 / self.lessons_attempted as f32 * 100.0
    }

    /// Pass rate as a percentage (of attempted)
    pub fn pass_rate(&self) -> f32 {
        if self.lessons_attempted == 0 {
            return 0.0;
        }
        self.lessons_passed as f32 / self.lessons_attempted as f32 * 100.0
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LESSON BANK — Maps objective IDs to concrete code tasks
// ═══════════════════════════════════════════════════════════════════════════════

/// Build the lesson bank: a mapping from objective IDs to concrete code tasks.
///
/// Each objective gets multiple concrete exercises to prevent overfitting.
/// Tier 1-2 exercises are designed so the native emitter can handle them.
/// Tier 3-4 exercises require LLM (Tier 2) fallback.
pub fn build_lesson_bank() -> HashMap<String, Vec<CodeLesson>> {
    let mut bank = HashMap::new();

    // ── Tier 1: Foundations ──────────────────────────────────────────────

    bank.insert(
        "codegen_simple_arithmetic".into(),
        vec![
            lesson(
                "codegen_simple_arithmetic",
                "add_numbers",
                "fn add_numbers(a: i32, b: i32) -> i32",
                "Add two numbers together",
                Some(
                    r#"
    #[test]
    fn test_add() { assert_eq!(add_numbers(2, 3), 5); }
    #[test]
    fn test_add_neg() { assert_eq!(add_numbers(-1, 1), 0); }
"#,
                ),
            ),
            lesson(
                "codegen_simple_arithmetic",
                "multiply",
                "fn multiply(a: i32, b: i32) -> i32",
                "Multiply two numbers",
                Some(
                    r#"
    #[test]
    fn test_mul() { assert_eq!(multiply(3, 4), 12); }
    #[test]
    fn test_mul_zero() { assert_eq!(multiply(5, 0), 0); }
"#,
                ),
            ),
            lesson(
                "codegen_simple_arithmetic",
                "absolute_value",
                "fn absolute_value(x: i32) -> i32",
                "Return the absolute value of a number",
                Some(
                    r#"
    #[test]
    fn test_pos() { assert_eq!(absolute_value(5), 5); }
    #[test]
    fn test_neg() { assert_eq!(absolute_value(-3), 3); }
    #[test]
    fn test_zero() { assert_eq!(absolute_value(0), 0); }
"#,
                ),
            ),
        ],
    );

    bank.insert(
        "codegen_string_ops".into(),
        vec![
            lesson(
                "codegen_string_ops",
                "reverse_string",
                "fn reverse_string(s: &str) -> String",
                "Reverse a string",
                Some(
                    r#"
    #[test]
    fn test_rev() { assert_eq!(reverse_string("hello"), "olleh"); }
    #[test]
    fn test_rev_empty() { assert_eq!(reverse_string(""), ""); }
"#,
                ),
            ),
            lesson(
                "codegen_string_ops",
                "to_uppercase",
                "fn to_uppercase(s: &str) -> String",
                "Convert a string to uppercase",
                Some(
                    r#"
    #[test]
    fn test_upper() { assert_eq!(to_uppercase("hello"), "HELLO"); }
"#,
                ),
            ),
            lesson(
                "codegen_string_ops",
                "count_vowels",
                "fn count_vowels(s: &str) -> usize",
                "Count the vowels in a string",
                Some(
                    r#"
    #[test]
    fn test_vowels() { assert_eq!(count_vowels("hello"), 2); }
    #[test]
    fn test_none() { assert_eq!(count_vowels("xyz"), 0); }
"#,
                ),
            ),
        ],
    );

    bank.insert(
        "codegen_boolean_checks".into(),
        vec![
            lesson(
                "codegen_boolean_checks",
                "is_even",
                "fn is_even(n: i32) -> bool",
                "Check if a number is even",
                Some(
                    r#"
    #[test]
    fn test_even() { assert!(is_even(4)); }
    #[test]
    fn test_odd() { assert!(!is_even(3)); }
"#,
                ),
            ),
            lesson(
                "codegen_boolean_checks",
                "is_positive",
                "fn is_positive(n: i32) -> bool",
                "Check if a number is positive",
                Some(
                    r#"
    #[test]
    fn test_pos() { assert!(is_positive(1)); }
    #[test]
    fn test_neg() { assert!(!is_positive(-1)); }
    #[test]
    fn test_zero() { assert!(!is_positive(0)); }
"#,
                ),
            ),
            lesson(
                "codegen_boolean_checks",
                "is_palindrome",
                "fn is_palindrome(s: &str) -> bool",
                "Check if a string is a palindrome",
                Some(
                    r#"
    #[test]
    fn test_pal() { assert!(is_palindrome("racecar")); }
    #[test]
    fn test_not_pal() { assert!(!is_palindrome("hello")); }
"#,
                ),
            ),
        ],
    );

    bank.insert(
        "codegen_basic_collections".into(),
        vec![
            lesson(
                "codegen_basic_collections",
                "sum_vec",
                "fn sum_vec(nums: &[i32]) -> i32",
                "Sum all numbers in a slice",
                Some(
                    r#"
    #[test]
    fn test_sum() { assert_eq!(sum_vec(&[1, 2, 3]), 6); }
    #[test]
    fn test_empty() { assert_eq!(sum_vec(&[]), 0); }
"#,
                ),
            ),
            lesson(
                "codegen_basic_collections",
                "find_max",
                "fn find_max(nums: &[i32]) -> Option<i32>",
                "Find the maximum value in a slice",
                Some(
                    r#"
    #[test]
    fn test_max() { assert_eq!(find_max(&[1, 5, 3]), Some(5)); }
    #[test]
    fn test_empty() { assert_eq!(find_max(&[]), None); }
"#,
                ),
            ),
            lesson(
                "codegen_basic_collections",
                "filter_positive",
                "fn filter_positive(nums: Vec<i32>) -> Vec<i32>",
                "Filter a vector to keep only positive numbers",
                Some(
                    r#"
    #[test]
    fn test_filter() { assert_eq!(filter_positive(vec![-1, 2, -3, 4]), vec![2, 4]); }
    #[test]
    fn test_all_neg() { assert_eq!(filter_positive(vec![-1, -2]), vec![]); }
"#,
                ),
            ),
        ],
    );

    // ── Tier 2: Composition ─────────────────────────────────────────────

    bank.insert(
        "codegen_composed_chains".into(),
        vec![
            lesson(
                "codegen_composed_chains",
                "sum_of_squares",
                "fn sum_of_squares(nums: Vec<i32>) -> i32",
                "Compute the sum of squares of all numbers using iterator chains",
                Some(
                    r#"
    #[test]
    fn test_sos() { assert_eq!(sum_of_squares(vec![1, 2, 3]), 14); }
    #[test]
    fn test_empty() { assert_eq!(sum_of_squares(vec![]), 0); }
"#,
                ),
            ),
            lesson(
                "codegen_composed_chains",
                "unique_sorted",
                "fn unique_sorted(nums: Vec<i32>) -> Vec<i32>",
                "Deduplicate and sort a vector of numbers",
                Some(
                    r#"
    #[test]
    fn test_dedup() { assert_eq!(unique_sorted(vec![3, 1, 2, 1, 3]), vec![1, 2, 3]); }
"#,
                ),
            ),
        ],
    );

    bank.insert(
        "codegen_struct_impl".into(),
        vec![lesson(
            "codegen_struct_impl",
            "Counter",
            "struct Counter { count: i32 }",
            "A simple counter with increment, decrement, and get methods",
            Some(
                r#"
    #[test]
    fn test_counter() {
        let mut c = Counter { count: 0 };
        c.increment();
        c.increment();
        assert_eq!(c.get(), 2);
        c.decrement();
        assert_eq!(c.get(), 1);
    }
"#,
            ),
        )],
    );

    bank.insert(
        "codegen_error_handling".into(),
        vec![lesson(
            "codegen_error_handling",
            "parse_integer",
            "fn parse_integer(s: &str) -> Result<i32, String>",
            "Parse a string to integer, returning Err with a message on failure",
            Some(
                r#"
    #[test]
    fn test_ok() { assert_eq!(parse_integer("42"), Ok(42)); }
    #[test]
    fn test_err() { assert!(parse_integer("abc").is_err()); }
"#,
            ),
        )],
    );

    bank.insert(
        "codegen_closures_hof".into(),
        vec![lesson(
            "codegen_closures_hof",
            "apply_twice",
            "fn apply_twice(f: impl Fn(i32) -> i32, x: i32) -> i32",
            "Apply a function to a value twice: f(f(x))",
            Some(
                r#"
    #[test]
    fn test_twice() { assert_eq!(apply_twice(|x| x + 1, 5), 7); }
    #[test]
    fn test_twice_double() { assert_eq!(apply_twice(|x| x * 2, 3), 12); }
"#,
            ),
        )],
    );

    bank.insert(
        "codegen_test_generation".into(),
        vec![lesson(
            "codegen_test_generation",
            "clamp",
            "fn clamp(value: i32, min: i32, max: i32) -> i32",
            "Clamp a value between min and max",
            Some(
                r#"
    #[test]
    fn test_below() { assert_eq!(clamp(-5, 0, 10), 0); }
    #[test]
    fn test_above() { assert_eq!(clamp(15, 0, 10), 10); }
    #[test]
    fn test_within() { assert_eq!(clamp(5, 0, 10), 5); }
"#,
            ),
        )],
    );

    // ── Tier 3: Advanced (these may need LLM fallback) ──────────────────

    bank.insert(
        "codegen_algorithm_sorting".into(),
        vec![lesson(
            "codegen_algorithm_sorting",
            "bubble_sort",
            "fn bubble_sort(arr: &mut Vec<i32>)",
            "Sort a vector in-place using bubble sort",
            Some(
                r#"
    #[test]
    fn test_sort() {
        let mut v = vec![3, 1, 4, 1, 5];
        bubble_sort(&mut v);
        assert_eq!(v, vec![1, 1, 3, 4, 5]);
    }
    #[test]
    fn test_empty() {
        let mut v: Vec<i32> = vec![];
        bubble_sort(&mut v);
        assert!(v.is_empty());
    }
"#,
            ),
        )],
    );

    bank.insert(
        "codegen_algorithm_search".into(),
        vec![lesson(
            "codegen_algorithm_search",
            "binary_search",
            "fn binary_search(arr: &[i32], target: i32) -> Option<usize>",
            "Find the index of a target value in a sorted slice using binary search",
            Some(
                r#"
    #[test]
    fn test_found() { assert_eq!(binary_search(&[1, 3, 5, 7, 9], 5), Some(2)); }
    #[test]
    fn test_not_found() { assert_eq!(binary_search(&[1, 3, 5, 7, 9], 4), None); }
    #[test]
    fn test_empty() { assert_eq!(binary_search(&[], 1), None); }
"#,
            ),
        )],
    );

    bank.insert(
        "codegen_algorithm_dp".into(),
        vec![lesson(
            "codegen_algorithm_dp",
            "fibonacci",
            "fn fibonacci(n: u32) -> u64",
            "Compute the nth Fibonacci number using dynamic programming",
            Some(
                r#"
    #[test]
    fn test_fib_0() { assert_eq!(fibonacci(0), 0); }
    #[test]
    fn test_fib_1() { assert_eq!(fibonacci(1), 1); }
    #[test]
    fn test_fib_10() { assert_eq!(fibonacci(10), 55); }
"#,
            ),
        )],
    );

    bank.insert("codegen_code_modification".into(), vec![]);
    bank.insert("codegen_multi_entity".into(), vec![]);
    bank.insert("codegen_import_inference".into(), vec![]);

    // ── Tier 4: Mastery (LLM-required) ──────────────────────────────────

    bank.insert("codegen_graph_algorithms".into(), vec![]);
    bank.insert("codegen_concurrent_code".into(), vec![]);
    bank.insert("codegen_generic_abstractions".into(), vec![]);
    bank.insert("codegen_self_correction".into(), vec![]);
    bank.insert("codegen_cross_language".into(), vec![]);

    bank
}

/// Helper: create a CodeLesson from components.
fn lesson(
    objective_id: &str,
    name: &str,
    signature: &str,
    purpose: &str,
    test_source: Option<&str>,
) -> CodeLesson {
    let spec = CodeSpec::new("rust", name, purpose).with_signature(signature);

    CodeLesson {
        objective_id: objective_id.into(),
        spec,
        test_source: test_source.map(|s| s.to_string()),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PREDICTION TRACKER (Item #1: Lookahead + Reality Check)
// ═══════════════════════════════════════════════════════════════════════════════

/// Lightweight quality predictor for code generation lessons.
///
/// Tracks per-objective success rates and calibrates predictions using EMA
/// of prediction errors. Feeds into the full LookaheadEngine when available.
pub struct PredictionTracker {
    /// Per-objective-prefix success rate (EMA)
    objective_rates: HashMap<String, f32>,
    /// Global prediction error EMA (|predicted - actual|)
    error_ema: f32,
    /// Total predictions made
    prediction_count: usize,
    /// Hallucination count (predicted >= 0.7, actual < 0.3)
    hallucination_count: usize,
}

impl PredictionTracker {
    pub fn new() -> Self {
        Self {
            objective_rates: HashMap::new(),
            error_ema: 0.0,
            prediction_count: 0,
            hallucination_count: 0,
        }
    }

    /// Predict quality for a given objective (0.0-1.0).
    ///
    /// Returns the EMA success rate for this objective prefix,
    /// or 0.7 (optimistic prior) if no history exists.
    pub fn predict(&self, objective_id: &str) -> f32 {
        let prefix = objective_prefix(objective_id);
        *self.objective_rates.get(&prefix).unwrap_or(&0.7)
    }

    /// Record an actual outcome and update calibration.
    pub fn record(&mut self, objective_id: &str, predicted: f32, actual: f32) {
        let prefix = objective_prefix(objective_id);
        let alpha = 0.3; // EMA decay

        // Update per-objective rate
        let rate = self.objective_rates.entry(prefix).or_insert(0.7);
        *rate = *rate * (1.0 - alpha) + actual * alpha;

        // Update global error EMA
        let error = (predicted - actual).abs();
        self.error_ema = self.error_ema * (1.0 - alpha) + error * alpha;

        self.prediction_count += 1;

        // Hallucination: predicted high (≥0.7) but actual was low (<0.3)
        if predicted >= 0.7 && actual < 0.3 {
            self.hallucination_count += 1;
        }
    }

    /// Average prediction error (EMA).
    pub fn avg_error(&self) -> f32 {
        self.error_ema
    }

    /// Fraction of predictions that were hallucinations.
    pub fn hallucination_rate(&self) -> f32 {
        if self.prediction_count == 0 {
            0.0
        } else {
            self.hallucination_count as f32 / self.prediction_count as f32
        }
    }

    /// Total predictions made.
    pub fn prediction_count(&self) -> usize {
        self.prediction_count
    }
}

/// Extract objective prefix for grouping (e.g., "codegen_simple_arithmetic" → "codegen_simple").
fn objective_prefix(id: &str) -> String {
    let parts: Vec<&str> = id.splitn(3, '_').collect();
    if parts.len() >= 2 {
        format!("{}_{}", parts[0], parts[1])
    } else {
        id.to_string()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ERROR PATTERN TRACKER (Item #3: Error-Driven Pattern Evolution)
// ═══════════════════════════════════════════════════════════════════════════════

/// A recurring compiler error pattern.
#[derive(Debug, Clone)]
pub struct ErrorPattern {
    /// Normalized error key (paths/line numbers stripped)
    pub error_key: String,
    /// Auto-fix hint (if one worked)
    pub fix_hint: Option<String>,
    /// Number of times this error has occurred
    pub occurrences: usize,
    /// Whether this has been evolved into a native pattern
    pub evolved: bool,
}

/// Tracks recurring compiler errors across lessons.
///
/// When an error recurs enough times (ERROR_PATTERN_EVOLUTION_THRESHOLD),
/// it's flagged as ready for native pattern evolution — meaning the emitter
/// should learn to avoid this class of error entirely.
pub struct ErrorPatternTracker {
    patterns: HashMap<String, ErrorPattern>,
}

impl ErrorPatternTracker {
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
        }
    }

    /// Record compiler errors from a lesson execution.
    pub fn record(&mut self, errors: &[String], fix_hint: Option<&str>) {
        for error in errors {
            let key = normalize_error(error);
            if key.is_empty() {
                continue;
            }
            let entry = self.patterns.entry(key.clone()).or_insert(ErrorPattern {
                error_key: key,
                fix_hint: None,
                occurrences: 0,
                evolved: false,
            });
            entry.occurrences += 1;
            if entry.fix_hint.is_none() {
                entry.fix_hint = fix_hint.map(|s| s.to_string());
            }
        }

        // Evict oldest if over capacity
        while self.patterns.len() > MAX_ERROR_PATTERNS {
            if let Some(min_key) = self
                .patterns
                .iter()
                .filter(|(_, p)| !p.evolved)
                .min_by_key(|(_, p)| p.occurrences)
                .map(|(k, _)| k.clone())
            {
                self.patterns.remove(&min_key);
            } else {
                break;
            }
        }
    }

    /// Patterns that have reached the evolution threshold but haven't been evolved yet.
    pub fn ready_for_evolution(&self) -> Vec<&ErrorPattern> {
        self.patterns
            .values()
            .filter(|p| p.occurrences >= ERROR_PATTERN_EVOLUTION_THRESHOLD && !p.evolved)
            .collect()
    }

    /// Mark a pattern as evolved into the native emitter.
    pub fn mark_evolved(&mut self, error_key: &str) {
        if let Some(p) = self.patterns.get_mut(error_key) {
            p.evolved = true;
        }
    }

    /// Total tracked patterns.
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Returns true if no patterns are tracked.
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Number of evolved patterns.
    pub fn evolved_count(&self) -> usize {
        self.patterns.values().filter(|p| p.evolved).count()
    }

    /// All tracked patterns.
    pub fn patterns(&self) -> impl Iterator<Item = &ErrorPattern> {
        self.patterns.values()
    }
}

/// Normalize a compiler error for stable pattern matching.
///
/// Strips file paths, line numbers, and column numbers to find the
/// structural error pattern.
fn normalize_error(error: &str) -> String {
    // Strip paths like /tmp/xxx/main.rs:123:45
    let re_path = regex::Regex::new(r"[/\\][\w./\\-]+\.rs:\d+:\d+").unwrap();
    let s = re_path.replace_all(error, "<SRC>");
    // Strip standalone line:col references
    let re_linecol = regex::Regex::new(r"\b\d+:\d+\b").unwrap();
    let s = re_linecol.replace_all(&s, "<POS>");
    s.trim().to_string()
}

// ═══════════════════════════════════════════════════════════════════════════════
// DISTILLATION RECORD (Item #4: SSM Distillation Prep)
// ═══════════════════════════════════════════════════════════════════════════════

/// Rich distillation record for Broca SSM training data.
///
/// Goes beyond the simple (purpose, source, quality) triple to include
/// metadata needed for curriculum-aware training: objective provenance,
/// whether the code was native-only, retry count, etc.
#[derive(Debug, Clone)]
pub struct DistillationRecord {
    /// What the code does (human-readable purpose)
    pub purpose: String,
    /// The generated source code
    pub source: String,
    /// Quality score (0.0-1.0)
    pub quality: f32,
    /// Curriculum objective that produced this
    pub objective_id: String,
    /// Whether this was generated purely by native emission (no LLM)
    pub native_only: bool,
    /// Total retries (auto-fix + LLM) needed
    pub total_retries: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// CODE LEARNING ENGINE
// ═══════════════════════════════════════════════════════════════════════════════

/// The code learning engine: runs lessons through generate → compile → test.
pub struct CodeLearningEngine {
    generator: CodeGenerator,
    executor: CodeExecutor,
    lesson_bank: HashMap<String, Vec<CodeLesson>>,
    /// Successful (spec, source, quality) triples for Broca SSM distillation
    distillation_cache: Vec<(String, String, f32)>,
    /// Past successful examples for few-shot context
    past_examples: Vec<(String, String)>,
    /// Metabolic energy budget for the current session
    budget: MetabolicBudget,
    /// Optional LLM prompt generator for Tier 2 fallback
    llm_prompt_fn: Option<Box<dyn Fn(&CodeSpec, &[String]) -> String + Send>>,
    /// Quality prediction tracker (Item #1)
    predictions: PredictionTracker,
    /// Recurring error pattern tracker (Item #3)
    error_tracker: ErrorPatternTracker,
    /// Rich distillation records for Broca SSM (Item #4)
    distillation_records: Vec<DistillationRecord>,
    /// Optional real-time distillation collector — writes training pairs to JSONL.
    /// When set, successful native generations are streamed to disk immediately
    /// (in addition to being cached in `distillation_cache`).
    /// Distillation collector requires ssm_language feature (symthaea-broca).
    /// When the feature is disabled, this is a unit-type placeholder.
    #[cfg(feature = "ssm_language")]
    distillation_collector: Option<crate::language::distillation::DistillationCollector>,
    #[cfg(not(feature = "ssm_language"))]
    distillation_collector: (),
    /// Optional coding experience store — persists error patterns and fix strategies
    /// across sessions. When set, compilation failures and fixes are recorded,
    /// and learned templates are stored for future native generation.
    experience_store: Option<crate::coding_experience::CodingExperienceStore>,
}

impl CodeLearningEngine {
    /// Create a new code learning engine.
    ///
    /// Uses simulation-mode executor by default. Call `with_real_execution()`
    /// for actual compilation.
    pub fn new(generator: CodeGenerator) -> Self {
        Self {
            generator,
            executor: CodeExecutor::new(),
            lesson_bank: build_lesson_bank(),
            distillation_cache: Vec::new(),
            past_examples: Vec::new(),
            budget: MetabolicBudget::default_session(),
            llm_prompt_fn: None,
            predictions: PredictionTracker::new(),
            error_tracker: ErrorPatternTracker::new(),
            distillation_records: Vec::new(),
            #[cfg(feature = "ssm_language")]
            distillation_collector: crate::language::distillation::DistillationCollector::from_env(
            ),
            #[cfg(not(feature = "ssm_language"))]
            distillation_collector: (),
            experience_store: None,
        }
    }

    /// Create with real (non-simulated) code execution.
    pub fn with_real_execution(generator: CodeGenerator) -> Self {
        Self {
            generator,
            executor: CodeExecutor::with_real_execution(),
            lesson_bank: build_lesson_bank(),
            distillation_cache: Vec::new(),
            past_examples: Vec::new(),
            budget: MetabolicBudget::default_session(),
            llm_prompt_fn: None,
            predictions: PredictionTracker::new(),
            error_tracker: ErrorPatternTracker::new(),
            distillation_records: Vec::new(),
            #[cfg(feature = "ssm_language")]
            distillation_collector: crate::language::distillation::DistillationCollector::from_env(
            ),
            #[cfg(not(feature = "ssm_language"))]
            distillation_collector: (),
            experience_store: None,
        }
    }

    /// Whether this engine can perform real compilation and test execution.
    pub fn supports_real_execution(&self) -> bool {
        self.executor.supports_real_execution()
    }

    /// Set a coding experience store for persistent error/fix tracking.
    pub fn with_experience_store(
        mut self,
        store: crate::coding_experience::CodingExperienceStore,
    ) -> Self {
        self.experience_store = Some(store);
        self
    }

    /// Set an explicit distillation collector (overrides env-var auto-detection).
    #[cfg(feature = "ssm_language")]
    pub fn with_distillation_collector(
        mut self,
        collector: crate::language::distillation::DistillationCollector,
    ) -> Self {
        self.distillation_collector = Some(collector);
        self
    }

    /// Set a custom metabolic budget for this session.
    pub fn with_budget(mut self, budget: MetabolicBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Set the LLM prompt generator for Tier 2 fallback.
    ///
    /// The function receives (spec, compiler_errors) and returns a prompt string.
    /// When set, lessons that produce `todo!()` will call this to get LLM-generated code.
    pub fn with_llm_prompt(
        mut self,
        prompt_fn: impl Fn(&CodeSpec, &[String]) -> String + Send + 'static,
    ) -> Self {
        self.llm_prompt_fn = Some(Box::new(prompt_fn));
        self
    }

    /// Get the current metabolic budget state.
    pub fn budget(&self) -> &MetabolicBudget {
        &self.budget
    }

    /// Run a single lesson through the full pipeline.
    ///
    /// Pipeline:
    /// 1. Generate via native emitter (Tier 1) — cost: 1.0 energy
    /// 2. If source contains `todo!()` and LLM is available — cost: 10.0 energy
    /// 3. Compile and auto-fix loop — cost: 0.5 per retry
    /// 4. If compilation still fails, LLM retry with error feedback — cost: 8.0 per retry
    /// 5. Record mastery and cache distillation targets
    pub fn run_lesson(&mut self, lesson: &CodeLesson) -> LessonOutcome {
        let mut energy_spent = 0.0;

        // Check budget before starting
        if !self.budget.can_afford_native() {
            return LessonOutcome::budget_exhausted(&lesson.objective_id);
        }

        // Item #1: Predict quality before committing energy
        let predicted_quality = self.predictions.predict(&lesson.objective_id);

        // 1. Build intent and context
        let intent = CodeIntent::Create {
            target: CodeTarget::new("code_learning", EntityKind::Module),
            spec: lesson.spec.clone(),
        };

        let context = CodeContext {
            past_examples: self.past_examples.clone(),
            ..Default::default()
        };

        // 2. Generate code (Tier 1: native emission)
        let generated = self.generator.generate(&intent, &context);
        let mut source = generated.source.clone();
        energy_spent += ENERGY_TIER1_NATIVE;

        // 3. If native emitter yields todo!(), try LLM fallback (Tier 2)
        let mut used_llm = false;
        let mut llm_retries = 0;

        if source.contains("todo!(") || source.contains("unimplemented!()") {
            if let Some(llm_source) = self.try_llm_generation(&lesson.spec, &[]) {
                source = llm_source;
                used_llm = true;
                energy_spent += ENERGY_TIER2_LLM_CALL;
            }
        }

        // 3.5. Proactive Symbolic Gating (Track 4)
        // Autonomously derive safety invariants and run Z3 checks before execution.
        let mut symbolic_veto = false;
        if let Ok(encoding) = crate::language::rust_ast_hdc::encode_rust_ast_hdc(&source, 512) {
            let invariants = encoding.derive_code_invariants();
            if !invariants.is_empty() {
                let z3 = symthaea_runtime::formal::z3_bridge::Z3Bridge::new();
                for inv in invariants {
                    // Try to find a counterexample (negate the safety invariant)
                    let query = format!("(assert (not {}))", inv.trim_start_matches("(assert ").trim_end_matches(')'));
                    if z3.verify_satisfiable(&query).is_sat() {
                        tracing::warn!("🚨 Symbolic Safety Veto: Potential invariant violation detected: {}", inv);
                        symbolic_veto = true;
                        break;
                    }
                }
            }
        }

        if symbolic_veto {
            return LessonOutcome {
                objective_id: lesson.objective_id.clone(),
                source,
                simulated_execution: true,
                compiled: false,
                tests_passed: 0,
                tests_failed: 0,
                surprise: 1.0, // Maximum surprise for safety violation
                retries_used: 0,
                plan_coverage: 0.0,
                phi_score: 0.0,
                used_llm,
                llm_retries_used: 0,
                distillation_eligible: false,
                energy_spent,
                predicted_quality,
                actual_quality: 0.0,
            };
        }

        // 4. Execute and auto-fix retry loop
        let mut retries = 0;
        let mut exec_result = self.execute_lesson(&source, lesson.test_source.as_deref());
        let mut simulated_execution = false;
        if exec_result.simulated {
            simulated_execution = true;
            exec_result = Self::mark_simulated_execution_failed(exec_result);
        }

        while !exec_result.compiled && retries < MAX_RETRIES {
            if simulated_execution {
                break;
            }
            // Item #3: Track errors before attempting fix
            let fix_applied = try_auto_fix(&source, &exec_result.compile_errors);
            // Item #3: Track errors (fix_hint is "auto-fix applied" if a fix was found)
            self.error_tracker.record(
                &exec_result.compile_errors,
                if fix_applied.is_some() {
                    Some("auto-fix")
                } else {
                    None
                },
            );

            // Record error in experience store for cross-session learning
            if let Some(ref mut store) = self.experience_store {
                let error_sig = exec_result.compile_errors.join("; ");
                if let Some(ref _fixed) = fix_applied {
                    store.store_fix_strategy(&error_sig, "auto-fix applied", None);
                }
            }

            if let Some(fixed) = fix_applied {
                source = fixed;
                retries += 1;
                energy_spent += ENERGY_AUTO_FIX;
                exec_result = self.execute_lesson(&source, lesson.test_source.as_deref());
                if exec_result.simulated {
                    simulated_execution = true;
                    exec_result = Self::mark_simulated_execution_failed(exec_result);
                    break;
                }
            } else {
                break;
            }
        }

        // 5. If still failing and we have LLM, retry with error feedback
        if !simulated_execution
            && !exec_result.compiled
            && self.llm_prompt_fn.is_some()
            && self.budget.can_afford_llm()
        {
            while !exec_result.compiled && llm_retries < MAX_LLM_RETRIES {
                if let Some(llm_source) =
                    self.try_llm_generation(&lesson.spec, &exec_result.compile_errors)
                {
                    source = llm_source;
                    used_llm = true;
                    llm_retries += 1;
                    energy_spent += ENERGY_LLM_RETRY;
                    exec_result = self.execute_lesson(&source, lesson.test_source.as_deref());
                    if exec_result.simulated {
                        simulated_execution = true;
                        exec_result = Self::mark_simulated_execution_failed(exec_result);
                        break;
                    }

                    // Also try auto-fix on LLM output
                    if !exec_result.compiled {
                        if let Some(fixed) = try_auto_fix(&source, &exec_result.compile_errors) {
                            source = fixed;
                            energy_spent += ENERGY_AUTO_FIX;
                            exec_result =
                                self.execute_lesson(&source, lesson.test_source.as_deref());
                            if exec_result.simulated {
                                simulated_execution = true;
                                exec_result = Self::mark_simulated_execution_failed(exec_result);
                                break;
                            }
                        }
                    }
                } else {
                    break;
                }
            }
        }

        // Record energy spent
        self.budget.spend(energy_spent);

        // 6. Build outcome
        let distillation_eligible = !simulated_execution
            && exec_result.compiled
            && exec_result.tests_failed == 0
            && generated.plan_coverage >= MIN_PLAN_COVERAGE;

        // Item #1: Compute actual quality and reality-check the prediction
        let actual_quality = if simulated_execution || !exec_result.compiled {
            0.0
        } else if exec_result.tests_failed > 0 {
            let total = exec_result.tests_passed + exec_result.tests_failed;
            exec_result.tests_passed as f32 / total as f32 * 0.5 // max 0.5 for partial
        } else {
            1.0 - (retries as f32 + llm_retries as f32) * 0.1 // small penalty per retry
        }
        .clamp(0.0, 1.0);

        let was_hallucination = predicted_quality >= 0.7 && actual_quality < 0.3;
        self.predictions
            .record(&lesson.objective_id, predicted_quality, actual_quality);

        let outcome = LessonOutcome {
            objective_id: lesson.objective_id.clone(),
            source: source.clone(),
            simulated_execution,
            compiled: exec_result.compiled,
            tests_passed: exec_result.tests_passed,
            tests_failed: exec_result.tests_failed,
            surprise: exec_result.to_surprise(),
            retries_used: retries,
            plan_coverage: generated.plan_coverage,
            phi_score: generated.phi_score,
            used_llm,
            llm_retries_used: llm_retries,
            distillation_eligible,
            energy_spent,
            predicted_quality: Some(predicted_quality),
            actual_quality,
            was_hallucination,
        };

        // 7. Cache successes for distillation and few-shot context
        if distillation_eligible {
            if let Some((_, src, quality)) =
                self.generator.distillation_target(&lesson.spec, &generated)
            {
                self.distillation_cache
                    .push((lesson.spec.purpose.clone(), src.clone(), quality));

                // Item #4: Build rich distillation record
                self.distillation_records.push(DistillationRecord {
                    purpose: lesson.spec.purpose.clone(),
                    source: src.clone(),
                    quality,
                    objective_id: lesson.objective_id.clone(),
                    native_only: !used_llm,
                    total_retries: retries + llm_retries,
                });

                // Stream to distillation collector if active (requires ssm_language)
                #[cfg(feature = "ssm_language")]
                if let Some(ref collector) = self.distillation_collector {
                    let channels = symthaea_broca::encoder::ThoughtChannels::default();
                    collector.record(&channels, &src);
                }

                // Store learned template in experience store
                if let Some(ref mut store) = self.experience_store {
                    store.store_learned_template(&lesson.spec.purpose, &src, None);
                }
            }
            // Add to past examples (capped at 16)
            if self.past_examples.len() < 16 {
                self.past_examples
                    .push((lesson.spec.purpose.clone(), source));
            }
        }

        outcome
    }

    /// Try to generate code via the LLM prompt function.
    ///
    /// Returns `Some(source)` if the LLM was called and produced output,
    /// `None` if no LLM is configured or the call failed.
    fn try_llm_generation(&self, spec: &CodeSpec, errors: &[String]) -> Option<String> {
        let prompt_fn = self.llm_prompt_fn.as_ref()?;
        let prompt = prompt_fn(spec, errors);
        if prompt.is_empty() {
            return None;
        }
        // The prompt is ready — the caller must use an async runtime to
        // actually call the LLM. For now, we return the prompt as a marker
        // that indicates LLM should be invoked. The actual async call
        // happens in the integration test / runner.
        //
        // In synchronous mode, we use a blocking Ollama HTTP call.
        self.call_ollama_sync(&prompt)
    }

    /// Synchronous Ollama call for the learning loop.
    ///
    /// Uses `curl` via `std::process::Command` — avoids needing reqwest's
    /// blocking feature. Acceptable for batch learning where latency doesn't
    /// matter (each call takes 5-30s anyway for a 7B model).
    fn call_ollama_sync(&self, prompt: &str) -> Option<String> {
        let model =
            std::env::var("SYMTHAEA_LLM_MODEL").unwrap_or_else(|_| "qwen2.5-coder:7b".to_string());

        // Escape the prompt for JSON embedding
        let escaped_prompt = prompt
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n");

        let body = format!(
            r#"{{"model":"{}","prompt":"{}","stream":false,"options":{{"temperature":0.3,"num_predict":1024}}}}"#,
            model, escaped_prompt
        );

        let output = std::process::Command::new("curl")
            .args([
                "-s",
                "--max-time",
                "120",
                "-X",
                "POST",
                "http://localhost:11434/api/generate",
                "-H",
                "Content-Type: application/json",
                "-d",
                &body,
            ])
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let response_str = String::from_utf8(output.stdout).ok()?;
        let json: serde_json::Value = serde_json::from_str(&response_str).ok()?;
        let text = json.get("response")?.as_str()?;

        extract_code_block(text)
    }

    /// Run all lessons for a given objective.
    pub fn run_objective(&mut self, objective_id: &str) -> Vec<LessonOutcome> {
        let lessons = match self.lesson_bank.get(objective_id) {
            Some(lessons) if !lessons.is_empty() => lessons.clone(),
            _ => return Vec::new(),
        };

        lessons.iter().map(|l| self.run_lesson(l)).collect()
    }

    /// Run a full learning session across all objectives with available lessons.
    ///
    /// Processes objectives in tier order (Tier 1 first).
    pub fn run_session(&mut self, objective_ids: &[&str]) -> SessionSummary {
        let mut summary = SessionSummary::default();
        let mut total_surprise = 0.0;
        let mut total_coverage = 0.0;

        for obj_id in objective_ids {
            // Check budget before starting each objective
            if !self.budget.can_afford_native() {
                break;
            }
            let outcomes = self.run_objective(obj_id);
            for outcome in outcomes {
                summary.lessons_attempted += 1;
                if outcome.simulated_execution {
                    summary.simulated_lessons += 1;
                }
                if outcome.compiled {
                    summary.lessons_compiled += 1;
                }
                if outcome.is_success() {
                    summary.lessons_passed += 1;
                }
                if outcome.distillation_eligible {
                    summary.distillation_eligible += 1;
                }
                summary.total_retries += outcome.retries_used;
                summary.total_llm_retries += outcome.llm_retries_used;
                summary.total_energy_spent += outcome.energy_spent;
                total_surprise += outcome.surprise;
                total_coverage += outcome.plan_coverage;
                summary.outcomes.push(outcome);
            }
        }

        if summary.lessons_attempted > 0 {
            summary.avg_surprise = total_surprise / summary.lessons_attempted as f32;
            summary.avg_plan_coverage = total_coverage / summary.lessons_attempted as f32;
        }

        // Item #1: Prediction calibration stats
        summary.avg_prediction_error = self.predictions.avg_error();
        summary.hallucination_rate = self.predictions.hallucination_rate();
        // Item #3: Error patterns ready for evolution
        summary.error_patterns_learned = self.error_tracker.ready_for_evolution().len();

        summary
    }

    /// Record all session outcomes into the CodingExperienceStore.
    ///
    /// This is the error-driven learning loop:
    /// - Compilation failures → encoded as error patterns for future fix hints
    /// - Successful generations → stored as learned templates for future native gen
    /// - Fix strategies → persisted for cross-session error recovery
    ///
    /// Call this after `run_session()` to persist learning across sessions.
    pub async fn persist_session_learning(&mut self, summary: &SessionSummary) {
        let store = match self.experience_store.as_mut() {
            Some(s) => s,
            None => return,
        };

        for outcome in &summary.outcomes {
            let experience = crate::coding_experience::CodingExperience {
                task: outcome.objective_id.clone(),
                detail: if outcome.compiled {
                    format!(
                        "compiled, {}/{} tests passed",
                        outcome.tests_passed,
                        outcome.tests_passed + outcome.tests_failed
                    )
                } else {
                    format!("compilation failed after {} retries", outcome.retries_used)
                },
                success: outcome.is_success(),
                tier: if outcome.used_llm {
                    "LLM".to_string()
                } else {
                    "Native".to_string()
                },
                fix_hint: if outcome.retries_used > 0 && outcome.compiled {
                    Some("auto-fix succeeded".to_string())
                } else {
                    None
                },
                diagnostic_hv: None,
            };
            store.store(experience).await;
        }

        store.flush().await;
    }

    /// Get the distillation cache (for Broca SSM training).
    pub fn distillation_cache(&self) -> &[(String, String, f32)] {
        &self.distillation_cache
    }

    /// Get past successful examples (for few-shot prompting).
    pub fn past_examples(&self) -> &[(String, String)] {
        &self.past_examples
    }

    /// Export the distillation cache to a JSONL file for Broca SSM training.
    ///
    /// This is the distillation feedback loop:
    /// 1. School runs lessons → successful code gets cached
    /// 2. This method exports to JSONL with ThoughtChannels
    /// 3. Broca can be trained on this data to learn code generation natively
    /// 4. Over time, native generation improves, reducing LLM dependency
    ///
    /// Returns the number of training pairs exported.
    pub fn export_for_broca_training(&self, path: &std::path::Path) -> std::io::Result<usize> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;

        let mut count = 0;
        for (purpose, source, quality) in &self.distillation_cache {
            // Only export high-quality generations
            if *quality < 0.5 {
                continue;
            }

            // Build a minimal ThoughtChannels-like record
            let channels: Vec<f32> = vec![0.0; 24];
            let record = serde_json::json!({
                "channels": channels,
                "target_text": source,
                "metadata": {
                    "purpose": purpose,
                    "quality": quality,
                    "domain": "code",
                }
            });

            if let Ok(json) = serde_json::to_string(&record) {
                writeln!(file, "{}", json)?;
                count += 1;
            }
        }

        Ok(count)
    }

    /// Get the native vs LLM generation ratio for this session.
    ///
    /// Tracks improvement over time: as Broca learns from distilled code,
    /// the native_ratio should increase and llm_ratio should decrease.
    pub fn generation_ratio(&self) -> (f32, f32) {
        let native = self
            .distillation_records
            .iter()
            .filter(|r| r.native_only)
            .count();
        let total = self.distillation_records.len();
        if total == 0 {
            return (0.5, 0.5); // uninformative prior
        }
        let native_ratio = native as f32 / total as f32;
        (native_ratio, 1.0 - native_ratio)
    }

    /// Number of distillation-eligible generations so far.
    pub fn distillation_count(&self) -> usize {
        self.distillation_cache.len()
    }

    fn execute_lesson(&mut self, source: &str, test_source: Option<&str>) -> ExecutionResult {
        self.executor.execute_rust(source, test_source)
    }

    fn mark_simulated_execution_failed(mut result: ExecutionResult) -> ExecutionResult {
        result.compiled = false;
        result.tests_passed = 0;
        result.tests_failed = 0;
        if result.compile_errors.is_empty() {
            result.compile_errors.push(
                "Execution was simulated; lesson results cannot count as real compilation or test verification"
                    .to_string(),
            );
        }
        result.runtime_error.get_or_insert_with(|| {
            "Execution was simulated; rerun with real execution enabled".to_string()
        });
        result
    }

    /// Reset the metabolic budget for a new session.
    pub fn reset_budget(&mut self) {
        self.budget = MetabolicBudget::default_session();
    }

    // ── Item #1: Prediction accessors ────────────────────────────────────

    /// Get the prediction tracker.
    pub fn predictions(&self) -> &PredictionTracker {
        &self.predictions
    }

    /// Average prediction error across all lessons.
    pub fn avg_prediction_error(&self) -> f32 {
        self.predictions.avg_error()
    }

    /// Fraction of predictions that were hallucinations.
    pub fn hallucination_rate(&self) -> f32 {
        self.predictions.hallucination_rate()
    }

    // ── Item #3: Error pattern accessors ─────────────────────────────────

    /// Get the error pattern tracker.
    pub fn error_tracker(&self) -> &ErrorPatternTracker {
        &self.error_tracker
    }

    /// Error patterns ready for evolution into native emitter patterns.
    pub fn error_patterns_ready_for_evolution(&self) -> Vec<&ErrorPattern> {
        self.error_tracker.ready_for_evolution()
    }

    // ── Item #4: Distillation record accessors ───────────────────────────

    /// Get the rich distillation records.
    pub fn distillation_records(&self) -> &[DistillationRecord] {
        &self.distillation_records
    }

    /// Export distillation records as JSONL (one JSON object per line).
    pub fn export_distillation_jsonl(&self) -> String {
        let mut output = String::new();
        for record in &self.distillation_records {
            let escaped_source = record
                .source
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n");
            let line = format!(
                r#"{{"purpose":"{}","source":"{}","quality":{:.3},"objective":"{}","native_only":{},"retries":{}}}"#,
                record.purpose,
                escaped_source,
                record.quality,
                record.objective_id,
                record.native_only,
                record.total_retries,
            );
            output.push_str(&line);
            output.push('\n');
        }
        output
    }

    /// Save distillation records to a JSONL file.
    pub fn save_distillation(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::write(path, self.export_distillation_jsonl())
    }
}

impl LessonOutcome {
    /// Create a "budget exhausted" outcome — skipped due to energy constraints.
    fn budget_exhausted(objective_id: &str) -> Self {
        Self {
            objective_id: objective_id.to_string(),
            source: String::new(),
            simulated_execution: false,
            compiled: false,
            tests_passed: 0,
            tests_failed: 0,
            surprise: 1.0,
            retries_used: 0,
            plan_coverage: 0.0,
            phi_score: 0.0,
            used_llm: false,
            llm_retries_used: 0,
            distillation_eligible: false,
            energy_spent: 0.0,
            predicted_quality: None,
            actual_quality: 0.0,
            was_hallucination: false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LLM PROMPT GENERATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Default LLM prompt generator for Tier 2 code generation.
///
/// Produces a focused prompt that includes the function signature, purpose,
/// and any compiler errors from a previous attempt.
pub fn default_llm_prompt(spec: &CodeSpec, errors: &[String]) -> String {
    let mut prompt = String::with_capacity(512);
    prompt.push_str("Write a Rust function. Output ONLY the function code, no explanation.\n\n");

    if let Some(sig) = &spec.signature {
        prompt.push_str(&format!("Signature: {}\n", sig));
    }
    prompt.push_str(&format!("Purpose: {}\n", spec.purpose));

    if !spec.constraints.is_empty() {
        prompt.push_str("Constraints:\n");
        for c in &spec.constraints {
            prompt.push_str(&format!("- {}\n", c));
        }
    }

    if !spec.examples.is_empty() {
        prompt.push_str("Examples:\n");
        for (input, output) in &spec.examples {
            prompt.push_str(&format!("  {} -> {}\n", input, output));
        }
    }

    if !errors.is_empty() {
        prompt.push_str("\nPrevious attempt had these compiler errors. Fix them:\n");
        for (i, err) in errors.iter().enumerate().take(5) {
            prompt.push_str(&format!("{}. {}\n", i + 1, err));
        }
    }

    prompt
}

/// Extract a code block from LLM output.
///
/// Handles ```rust ... ```, ``` ... ```, or raw code.
fn extract_code_block(text: &str) -> Option<String> {
    // Try ```rust ... ``` first
    if let Some(start) = text.find("```rust") {
        let code_start = start + 7;
        // Skip optional newline after ```rust
        let code_start = if text[code_start..].starts_with('\n') {
            code_start + 1
        } else {
            code_start
        };
        if let Some(end) = text[code_start..].find("```") {
            return Some(text[code_start..code_start + end].trim().to_string());
        }
    }

    // Try ``` ... ```
    if let Some(start) = text.find("```") {
        let code_start = start + 3;
        let code_start = if text[code_start..].starts_with('\n') {
            code_start + 1
        } else {
            code_start
        };
        if let Some(end) = text[code_start..].find("```") {
            return Some(text[code_start..code_start + end].trim().to_string());
        }
    }

    // No code block — return raw text if it looks like code
    let trimmed = text.trim();
    if trimmed.contains("fn ") || trimmed.contains("struct ") || trimmed.contains("impl ") {
        Some(trimmed.to_string())
    } else if !trimmed.is_empty() {
        Some(trimmed.to_string())
    } else {
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TIER 1 OBJECTIVE IDS (for convenient session setup)
// ═══════════════════════════════════════════════════════════════════════════════

/// All Tier 1 objective IDs
pub const TIER1_OBJECTIVES: &[&str] = &[
    "codegen_simple_arithmetic",
    "codegen_string_ops",
    "codegen_boolean_checks",
    "codegen_basic_collections",
];

/// All Tier 2 objective IDs
pub const TIER2_OBJECTIVES: &[&str] = &[
    "codegen_composed_chains",
    "codegen_struct_impl",
    "codegen_error_handling",
    "codegen_closures_hof",
    "codegen_test_generation",
];

/// All Tier 3 objective IDs (with lessons)
pub const TIER3_OBJECTIVES: &[&str] = &[
    "codegen_algorithm_sorting",
    "codegen_algorithm_search",
    "codegen_algorithm_dp",
];

// ═══════════════════════════════════════════════════════════════════════════════
// AUTO-CURRICULUM FROM FAILURES (Phase 5, Item 6)
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate a targeted [`CodeLesson`] from a recurring compiler error pattern.
pub fn lesson_from_failure(
    error_pattern: &str,
    task_description: &str,
    _failure_count: usize,
) -> Option<CodeLesson> {
    use crate::mind::structured_thought::EpistemicStatus;

    let error_lower = error_pattern.to_lowercase();
    let (name, signature, purpose, test) =
        if error_lower.contains("e0308") || error_lower.contains("mismatched types") {
            (
                "type_conversion_drill",
                "fn convert(input: &str) -> i64",
                "Parse string to integer with proper error handling",
                "assert_eq!(convert(\"42\"), 42);",
            )
        } else if error_lower.contains("e0382")
            || error_lower.contains("moved value")
            || error_lower.contains("borrow")
        {
            (
                "ownership_transfer_drill",
                "fn process(data: Vec<u8>) -> (Vec<u8>, usize)",
                "Return owned data after processing to maintain ownership",
                "let (d, n) = process(vec![1,2,3]); assert_eq!(n, 3);",
            )
        } else if error_lower.contains("e0106") || error_lower.contains("lifetime") {
            (
                "lifetime_annotation_drill",
                "fn longest<'a>(a: &'a str, b: &'a str) -> &'a str",
                "Return the longer of two string slices with correct lifetimes",
                "assert_eq!(longest(\"ab\", \"abc\"), \"abc\");",
            )
        } else if error_lower.contains("e0412") || error_lower.contains("cannot find type") {
            (
                "import_resolution_drill",
                "fn create_map() -> std::collections::HashMap<String, i32>",
                "Use fully qualified paths or proper imports",
                "let m = create_map(); assert!(m.is_empty());",
            )
        } else if error_lower.contains("e0277") || error_lower.contains("trait bound") {
            (
                "trait_bound_drill",
                "fn print_all<T: std::fmt::Display>(items: &[T])",
                "Apply correct trait bounds for generic functions",
                "print_all(&[1, 2, 3]);",
            )
        } else if error_lower.contains("overflow") {
            (
                "safe_arithmetic_drill",
                "fn safe_add(a: u32, b: u32) -> Option<u32>",
                "Use checked arithmetic to prevent overflow",
                "assert_eq!(safe_add(u32::MAX, 1), None);",
            )
        } else {
            return None;
        };

    Some(CodeLesson {
        objective_id: format!("auto_drill_{}", name),
        spec: CodeSpec {
            language: "rust".to_string(),
            name: format!("auto_{}", name),
            purpose: format!("{} (auto-generated from: {})", purpose, task_description),
            purpose_hv: None,
            signature: Some(signature.to_string()),
            constraints: Vec::new(),
            examples: Vec::new(),
            epistemic_status: EpistemicStatus::Probable,
            metadata: HashMap::new(),
        },
        test_source: Some(test.to_string()),
    })
}

/// Generate lessons from a batch of failure patterns, sorted by frequency.
pub fn lessons_from_failures(failures: &[(String, String, usize)], max: usize) -> Vec<CodeLesson> {
    let mut sorted: Vec<_> = failures.to_vec();
    sorted.sort_by(|a, b| b.2.cmp(&a.2));
    sorted
        .iter()
        .filter_map(|(pattern, task, count)| lesson_from_failure(pattern, task, *count))
        .take(max)
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::code_encoder::CodeHDEncoder;

    fn make_engine() -> CodeLearningEngine {
        let encoder = CodeHDEncoder::new(256);
        let generator = CodeGenerator::new(encoder);
        // Simulation mode — no real compilation
        CodeLearningEngine::new(generator)
    }

    #[test]
    fn test_lesson_bank_has_all_tier1() {
        let bank = build_lesson_bank();
        for id in TIER1_OBJECTIVES {
            let lessons = bank.get(*id);
            assert!(
                lessons.is_some() && !lessons.unwrap().is_empty(),
                "Missing lessons for Tier 1 objective: {id}"
            );
        }
    }

    #[test]
    fn test_lesson_bank_has_tier2() {
        let bank = build_lesson_bank();
        for id in TIER2_OBJECTIVES {
            assert!(
                bank.contains_key(*id),
                "Missing entry for Tier 2 objective: {id}"
            );
        }
    }

    #[test]
    fn test_lesson_bank_tier1_has_tests() {
        let bank = build_lesson_bank();
        for id in TIER1_OBJECTIVES {
            for lesson in bank.get(*id).unwrap() {
                assert!(
                    lesson.test_source.is_some(),
                    "Tier 1 lesson {}/{} should have test source",
                    id,
                    lesson.spec.name,
                );
            }
        }
    }

    #[test]
    fn test_lesson_bank_total_count() {
        let bank = build_lesson_bank();
        let total: usize = bank.values().map(|v| v.len()).sum();
        assert!(
            total >= 15,
            "Expected at least 15 concrete lessons, got {total}"
        );
    }

    #[test]
    fn test_run_single_lesson_simulated() {
        let mut engine = make_engine();
        let bank = build_lesson_bank();
        let lesson = &bank["codegen_simple_arithmetic"][0];

        let outcome = engine.run_lesson(lesson);
        assert!(outcome.simulated_execution);
        assert!(
            !outcome.compiled,
            "Simulated execution must not count as a real compile"
        );
        assert_eq!(outcome.objective_id, "codegen_simple_arithmetic");
    }

    #[test]
    fn test_run_objective_simulated() {
        let mut engine = make_engine();
        let outcomes = engine.run_objective("codegen_simple_arithmetic");
        assert_eq!(outcomes.len(), 3, "Should run all 3 arithmetic lessons");
        for outcome in &outcomes {
            assert!(outcome.simulated_execution);
            assert!(!outcome.compiled);
        }
    }

    #[test]
    fn test_run_session_tier1() {
        let mut engine = make_engine();
        let summary = engine.run_session(TIER1_OBJECTIVES);
        assert!(
            summary.lessons_attempted >= 12,
            "Tier 1 has 12 lessons total, got {}",
            summary.lessons_attempted,
        );
        assert_eq!(summary.simulated_lessons, summary.lessons_attempted);
        assert_eq!(summary.lessons_compiled, 0);
    }

    #[test]
    fn test_run_empty_objective() {
        let mut engine = make_engine();
        let outcomes = engine.run_objective("codegen_graph_algorithms");
        assert!(outcomes.is_empty(), "No lessons yet for graph algorithms");
    }

    #[test]
    fn test_lesson_outcome_mastery_signal() {
        let success = LessonOutcome {
            objective_id: "test".into(),
            source: String::new(),
            simulated_execution: false,
            compiled: true,
            tests_passed: 3,
            tests_failed: 0,
            surprise: 0.0,
            retries_used: 0,
            plan_coverage: 1.0,
            phi_score: 0.5,
            used_llm: false,
            llm_retries_used: 0,
            distillation_eligible: true,
            energy_spent: 1.0,
            predicted_quality: Some(0.7),
            actual_quality: 1.0,
            was_hallucination: false,
        };
        assert_eq!(success.mastery_signal(), 1.0);

        let partial = LessonOutcome {
            tests_passed: 2,
            tests_failed: 1,
            ..success.clone()
        };
        assert!((partial.mastery_signal() - 0.667).abs() < 0.01);

        let fail = LessonOutcome {
            compiled: false,
            ..success
        };
        assert_eq!(fail.mastery_signal(), 0.0);
    }

    #[test]
    fn test_session_summary_rates() {
        let summary = SessionSummary {
            lessons_attempted: 10,
            simulated_lessons: 2,
            lessons_compiled: 8,
            lessons_passed: 6,
            ..Default::default()
        };
        assert!((summary.simulated_rate() - 20.0).abs() < 0.01);
        assert!((summary.compile_rate() - 80.0).abs() < 0.01);
        assert!((summary.pass_rate() - 60.0).abs() < 0.01);
    }

    #[test]
    fn test_distillation_cache_starts_empty() {
        let engine = make_engine();
        assert_eq!(engine.distillation_count(), 0);
        assert!(engine.distillation_cache().is_empty());
    }

    // ── Metabolic budget tests ──────────────────────────────────────────

    #[test]
    fn test_metabolic_budget_default() {
        let budget = MetabolicBudget::default_session();
        assert_eq!(budget.total_budget, 100.0);
        assert_eq!(budget.spent, 0.0);
        assert_eq!(budget.reserved, 20.0); // 20% reserved
        assert!(budget.can_afford_native());
        assert!(budget.can_afford_llm());
    }

    #[test]
    fn test_metabolic_budget_spending() {
        let mut budget = MetabolicBudget::new(50.0);
        assert_eq!(budget.available(), 40.0); // 50 - 0 - 10 (20% reserve)

        budget.spend(30.0);
        assert_eq!(budget.available(), 10.0);
        assert!(budget.can_afford_native()); // 10 >= 1.0

        budget.spend(10.0);
        assert_eq!(budget.available(), 0.0);
        assert!(!budget.can_afford_native()); // 0 < 1.0
        assert!(budget.can_afford_llm()); // 10 reserve >= 10 LLM cost
    }

    #[test]
    fn test_metabolic_budget_exhaustion() {
        let mut budget = MetabolicBudget::new(50.0);
        budget.spend(50.0);
        assert!(!budget.can_afford_native());
        assert!(!budget.can_afford_llm());
        assert!(budget.utilization() >= 1.0);
    }

    #[test]
    fn test_metabolic_budget_efficiency() {
        let mut budget = MetabolicBudget::new(100.0);
        budget.spend(20.0);
        // 0.5 mastery / 20 energy = 0.025 efficiency
        assert!((budget.efficiency(0.5) - 0.025).abs() < 0.001);
    }

    #[test]
    fn test_budget_exhausted_outcome() {
        let outcome = LessonOutcome::budget_exhausted("test_obj");
        assert!(!outcome.compiled);
        assert_eq!(outcome.surprise, 1.0);
        assert_eq!(outcome.energy_spent, 0.0);
        assert!(!outcome.distillation_eligible);
    }

    #[test]
    fn test_session_respects_budget() {
        let encoder = CodeHDEncoder::new(256);
        let generator = CodeGenerator::new(encoder);
        // Tiny budget: only enough for 2 native generations
        let mut engine = CodeLearningEngine::new(generator).with_budget(MetabolicBudget::new(2.5));

        let summary = engine.run_session(TIER1_OBJECTIVES);
        // Should stop early due to budget
        assert!(
            summary.lessons_attempted <= 3,
            "Budget should limit lessons, got {}",
            summary.lessons_attempted,
        );
    }

    // ── LLM prompt tests ────────────────────────────────────────────────

    #[test]
    fn test_default_llm_prompt_basic() {
        let spec = CodeSpec::new("rust", "fibonacci", "Compute fibonacci numbers")
            .with_signature("fn fibonacci(n: u32) -> u64");
        let prompt = default_llm_prompt(&spec, &[]);
        assert!(prompt.contains("fn fibonacci(n: u32) -> u64"));
        assert!(prompt.contains("Compute fibonacci"));
        assert!(!prompt.contains("compiler errors"));
    }

    #[test]
    fn test_default_llm_prompt_with_errors() {
        let spec = CodeSpec::new("rust", "test", "test function");
        let errors = vec!["error[E0308]: mismatched types".to_string()];
        let prompt = default_llm_prompt(&spec, &errors);
        assert!(prompt.contains("compiler errors"));
        assert!(prompt.contains("E0308"));
    }

    // ── Code block extraction tests ─────────────────────────────────────

    #[test]
    fn test_extract_code_block_rust_fenced() {
        let text = "Here's the code:\n```rust\nfn add(a: i32, b: i32) -> i32 { a + b }\n```\nDone.";
        let code = extract_code_block(text).unwrap();
        assert!(code.contains("fn add"));
        assert!(!code.contains("```"));
    }

    #[test]
    fn test_extract_code_block_plain_fenced() {
        let text = "```\nfn foo() {}\n```";
        let code = extract_code_block(text).unwrap();
        assert!(code.contains("fn foo"));
    }

    #[test]
    fn test_extract_code_block_raw() {
        let text = "fn bar(x: i32) -> i32 { x * 2 }";
        let code = extract_code_block(text).unwrap();
        assert!(code.contains("fn bar"));
    }

    #[test]
    fn test_energy_tracking_in_session() {
        let mut engine = make_engine();
        let summary = engine.run_session(TIER1_OBJECTIVES);
        // Each lesson costs at least ENERGY_TIER1_NATIVE (1.0)
        assert!(
            summary.total_energy_spent >= summary.lessons_attempted as f32,
            "Energy {} should be >= lessons {}",
            summary.total_energy_spent,
            summary.lessons_attempted,
        );
    }

    // ── Item #1: Prediction tracker tests ────────────────────────────────

    #[test]
    fn test_prediction_tracker_initial_prior() {
        let tracker = PredictionTracker::new();
        // Unknown objective should return optimistic prior
        assert_eq!(tracker.predict("codegen_unknown_foo"), 0.7);
        assert_eq!(tracker.prediction_count(), 0);
        assert_eq!(tracker.hallucination_rate(), 0.0);
    }

    #[test]
    fn test_prediction_tracker_learns_from_outcomes() {
        let mut tracker = PredictionTracker::new();
        // Record several successes
        for _ in 0..5 {
            tracker.record("codegen_simple_add", 0.7, 1.0);
        }
        // Prediction should move toward 1.0
        let pred = tracker.predict("codegen_simple_add");
        assert!(
            pred > 0.8,
            "Prediction should be high after successes, got {pred}"
        );
    }

    #[test]
    fn test_prediction_tracker_detects_hallucinations() {
        let mut tracker = PredictionTracker::new();
        // Predicted high (0.7), actual low (0.0) = hallucination
        tracker.record("codegen_hard_thing", 0.7, 0.0);
        assert_eq!(tracker.hallucination_rate(), 1.0);

        // Record a non-hallucination
        tracker.record("codegen_hard_thing", 0.5, 0.8);
        assert_eq!(tracker.hallucination_rate(), 0.5);
    }

    #[test]
    fn test_prediction_in_session() {
        let mut engine = make_engine();
        let summary = engine.run_session(TIER1_OBJECTIVES);
        // After a full session, predictions should be calibrated
        assert!(
            engine.predictions().prediction_count() >= 12,
            "Should have 12+ predictions, got {}",
            engine.predictions().prediction_count(),
        );
        assert!(
            summary.hallucination_rate > 0.0,
            "Simulated lessons should not be credited as real success"
        );
    }

    // ── Item #3: Error pattern tracker tests ─────────────────────────────

    #[test]
    fn test_error_pattern_tracker_basic() {
        let mut tracker = ErrorPatternTracker::new();
        assert_eq!(tracker.len(), 0);

        tracker.record(
            &["error[E0308]: mismatched types at /tmp/x.rs:1:5".to_string()],
            None,
        );
        assert_eq!(tracker.len(), 1);
        assert!(tracker.ready_for_evolution().is_empty());
    }

    #[test]
    fn test_error_pattern_reaches_threshold() {
        let mut tracker = ErrorPatternTracker::new();
        let error = "error[E0308]: mismatched types at /tmp/x.rs:1:5".to_string();

        for _ in 0..ERROR_PATTERN_EVOLUTION_THRESHOLD {
            tracker.record(&[error.clone()], Some("add type annotation"));
        }

        let ready = tracker.ready_for_evolution();
        assert_eq!(ready.len(), 1, "One pattern should be ready for evolution");
        assert_eq!(ready[0].occurrences, ERROR_PATTERN_EVOLUTION_THRESHOLD);
        assert!(ready[0].fix_hint.is_some());
    }

    #[test]
    fn test_error_pattern_mark_evolved() {
        let mut tracker = ErrorPatternTracker::new();
        let error = "error[E0308]: mismatched types".to_string();
        for _ in 0..3 {
            tracker.record(&[error.clone()], None);
        }

        let key = tracker.ready_for_evolution()[0].error_key.clone();
        tracker.mark_evolved(&key);

        assert!(tracker.ready_for_evolution().is_empty());
        assert_eq!(tracker.evolved_count(), 1);
    }

    #[test]
    fn test_normalize_error_strips_paths() {
        let raw = "error[E0308]: mismatched types at /tmp/code_abc123/main.rs:42:10";
        let norm = normalize_error(raw);
        assert!(!norm.contains("/tmp/"));
        assert!(!norm.contains("42:10"));
        assert!(norm.contains("E0308"));
    }

    #[test]
    fn test_session_tracks_error_patterns() {
        let mut engine = make_engine();
        let summary = engine.run_session(TIER1_OBJECTIVES);
        assert_eq!(
            summary.error_patterns_learned, 0,
            "No error patterns in simulation mode"
        );
    }

    // ── Item #4: Distillation record tests ───────────────────────────────

    #[test]
    fn test_distillation_records_populated() {
        let mut engine = make_engine();
        let _summary = engine.run_session(TIER1_OBJECTIVES);
        assert!(
            engine.distillation_records().is_empty(),
            "Simulated sessions must not produce distillation records"
        );
    }

    #[test]
    fn test_distillation_jsonl_export() {
        let mut engine = make_engine();
        let _summary = engine.run_session(TIER1_OBJECTIVES);
        let jsonl = engine.export_distillation_jsonl();
        assert!(jsonl.is_empty());
    }

    #[test]
    fn test_distillation_save_to_file() {
        let mut engine = make_engine();
        let _summary = engine.run_session(TIER1_OBJECTIVES);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        engine.save_distillation(tmp.path()).unwrap();
        let contents = std::fs::read_to_string(tmp.path()).unwrap();
        assert!(contents.is_empty());
    }

    #[test]
    fn test_objective_prefix() {
        assert_eq!(
            objective_prefix("codegen_simple_arithmetic"),
            "codegen_simple"
        );
        assert_eq!(objective_prefix("codegen_string_ops"), "codegen_string");
        assert_eq!(objective_prefix("solo"), "solo");
    }

    #[test]
    fn test_lesson_from_borrow_error() {
        let lesson = lesson_from_failure(
            "error[E0382]: use of moved value: `data`",
            "process data after sending",
            2,
        );
        assert!(lesson.is_some());
        assert!(lesson.unwrap().spec.name.contains("ownership"));
    }

    #[test]
    fn test_lesson_from_type_mismatch() {
        let lesson = lesson_from_failure(
            "error[E0308]: mismatched types expected i32 found &str",
            "parse user input",
            3,
        );
        assert!(lesson.is_some());
        let l = lesson.unwrap();
        assert!(l.spec.name.contains("type_conversion"));
    }

    #[test]
    fn test_lesson_from_lifetime_error() {
        let lesson = lesson_from_failure(
            "error[E0106]: missing lifetime specifier",
            "return reference from function",
            1,
        );
        assert!(lesson.is_some());
        assert!(lesson.unwrap().spec.name.contains("lifetime"));
    }

    #[test]
    fn test_lesson_from_unknown_error_returns_none() {
        let lesson =
            lesson_from_failure("some completely unknown error pattern", "do something", 5);
        assert!(lesson.is_none());
    }

    #[test]
    fn test_lessons_from_failures_sorted_by_frequency() {
        let failures = vec![
            ("E0382 moved".into(), "task a".into(), 1usize),
            ("E0308 mismatch".into(), "task b".into(), 5),
            ("E0106 lifetime".into(), "task c".into(), 3),
        ];
        let lessons = lessons_from_failures(&failures, 10);
        assert_eq!(lessons.len(), 3);
        assert!(lessons[0].spec.name.contains("type_conversion"));
    }

    #[test]
    fn test_lessons_from_failures_max_cap() {
        let failures = vec![
            ("E0382 moved".into(), "a".into(), 2usize),
            ("E0308 mismatch".into(), "b".into(), 4),
            ("E0106 lifetime".into(), "c".into(), 3),
        ];
        let lessons = lessons_from_failures(&failures, 2);
        assert_eq!(lessons.len(), 2);
    }
}