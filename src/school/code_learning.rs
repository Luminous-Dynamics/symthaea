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

use crate::language::code_executor::{try_auto_fix, CodeExecutor, ExecutionResult};
use crate::language::code_generator::{CodeContext, CodeGenerator, GeneratedCode};
use crate::language::code_intent::{CodeIntent, CodeSpec, CodeTarget};

/// Maximum auto-fix retry attempts per generation
const MAX_RETRIES: usize = 3;

/// Minimum plan coverage to consider a generation "good enough" for mastery
const MIN_PLAN_COVERAGE: f32 = 0.5;

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
    /// Eligible for distillation into Broca SSM
    pub distillation_eligible: bool,
}

impl LessonOutcome {
    /// Whether the lesson was fully successful
    pub fn is_success(&self) -> bool {
        self.compiled && self.tests_failed == 0
    }

    /// Mastery signal: 1.0 for perfect, scaled down for partial success
    pub fn mastery_signal(&self) -> f32 {
        if !self.compiled {
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

/// Summary of a full learning session across multiple objectives.
#[derive(Debug, Clone, Default)]
pub struct SessionSummary {
    /// Total lessons attempted
    pub lessons_attempted: usize,
    /// Lessons that compiled
    pub lessons_compiled: usize,
    /// Lessons where all tests passed
    pub lessons_passed: usize,
    /// Total auto-fix retries across all lessons
    pub total_retries: usize,
    /// Average surprise across all lessons
    pub avg_surprise: f32,
    /// Average plan coverage
    pub avg_plan_coverage: f32,
    /// Lessons eligible for distillation
    pub distillation_eligible: usize,
    /// Per-objective outcomes
    pub outcomes: Vec<LessonOutcome>,
}

impl SessionSummary {
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
            lesson("codegen_simple_arithmetic", "add_numbers",
                "fn add_numbers(a: i32, b: i32) -> i32",
                "Add two numbers together",
                Some(r#"
    #[test]
    fn test_add() { assert_eq!(add_numbers(2, 3), 5); }
    #[test]
    fn test_add_neg() { assert_eq!(add_numbers(-1, 1), 0); }
"#)),
            lesson("codegen_simple_arithmetic", "multiply",
                "fn multiply(a: i32, b: i32) -> i32",
                "Multiply two numbers",
                Some(r#"
    #[test]
    fn test_mul() { assert_eq!(multiply(3, 4), 12); }
    #[test]
    fn test_mul_zero() { assert_eq!(multiply(5, 0), 0); }
"#)),
            lesson("codegen_simple_arithmetic", "absolute_value",
                "fn absolute_value(x: i32) -> i32",
                "Return the absolute value of a number",
                Some(r#"
    #[test]
    fn test_pos() { assert_eq!(absolute_value(5), 5); }
    #[test]
    fn test_neg() { assert_eq!(absolute_value(-3), 3); }
    #[test]
    fn test_zero() { assert_eq!(absolute_value(0), 0); }
"#)),
        ],
    );

    bank.insert(
        "codegen_string_ops".into(),
        vec![
            lesson("codegen_string_ops", "reverse_string",
                "fn reverse_string(s: &str) -> String",
                "Reverse a string",
                Some(r#"
    #[test]
    fn test_rev() { assert_eq!(reverse_string("hello"), "olleh"); }
    #[test]
    fn test_rev_empty() { assert_eq!(reverse_string(""), ""); }
"#)),
            lesson("codegen_string_ops", "to_uppercase",
                "fn to_uppercase(s: &str) -> String",
                "Convert a string to uppercase",
                Some(r#"
    #[test]
    fn test_upper() { assert_eq!(to_uppercase("hello"), "HELLO"); }
"#)),
            lesson("codegen_string_ops", "count_vowels",
                "fn count_vowels(s: &str) -> usize",
                "Count the vowels in a string",
                Some(r#"
    #[test]
    fn test_vowels() { assert_eq!(count_vowels("hello"), 2); }
    #[test]
    fn test_none() { assert_eq!(count_vowels("xyz"), 0); }
"#)),
        ],
    );

    bank.insert(
        "codegen_boolean_checks".into(),
        vec![
            lesson("codegen_boolean_checks", "is_even",
                "fn is_even(n: i32) -> bool",
                "Check if a number is even",
                Some(r#"
    #[test]
    fn test_even() { assert!(is_even(4)); }
    #[test]
    fn test_odd() { assert!(!is_even(3)); }
"#)),
            lesson("codegen_boolean_checks", "is_positive",
                "fn is_positive(n: i32) -> bool",
                "Check if a number is positive",
                Some(r#"
    #[test]
    fn test_pos() { assert!(is_positive(1)); }
    #[test]
    fn test_neg() { assert!(!is_positive(-1)); }
    #[test]
    fn test_zero() { assert!(!is_positive(0)); }
"#)),
            lesson("codegen_boolean_checks", "is_palindrome",
                "fn is_palindrome(s: &str) -> bool",
                "Check if a string is a palindrome",
                Some(r#"
    #[test]
    fn test_pal() { assert!(is_palindrome("racecar")); }
    #[test]
    fn test_not_pal() { assert!(!is_palindrome("hello")); }
"#)),
        ],
    );

    bank.insert(
        "codegen_basic_collections".into(),
        vec![
            lesson("codegen_basic_collections", "sum_vec",
                "fn sum_vec(nums: &[i32]) -> i32",
                "Sum all numbers in a slice",
                Some(r#"
    #[test]
    fn test_sum() { assert_eq!(sum_vec(&[1, 2, 3]), 6); }
    #[test]
    fn test_empty() { assert_eq!(sum_vec(&[]), 0); }
"#)),
            lesson("codegen_basic_collections", "find_max",
                "fn find_max(nums: &[i32]) -> Option<i32>",
                "Find the maximum value in a slice",
                Some(r#"
    #[test]
    fn test_max() { assert_eq!(find_max(&[1, 5, 3]), Some(5)); }
    #[test]
    fn test_empty() { assert_eq!(find_max(&[]), None); }
"#)),
            lesson("codegen_basic_collections", "filter_positive",
                "fn filter_positive(nums: Vec<i32>) -> Vec<i32>",
                "Filter a vector to keep only positive numbers",
                Some(r#"
    #[test]
    fn test_filter() { assert_eq!(filter_positive(vec![-1, 2, -3, 4]), vec![2, 4]); }
    #[test]
    fn test_all_neg() { assert_eq!(filter_positive(vec![-1, -2]), vec![]); }
"#)),
        ],
    );

    // ── Tier 2: Composition ─────────────────────────────────────────────

    bank.insert(
        "codegen_composed_chains".into(),
        vec![
            lesson("codegen_composed_chains", "sum_of_squares",
                "fn sum_of_squares(nums: Vec<i32>) -> i32",
                "Compute the sum of squares of all numbers using iterator chains",
                Some(r#"
    #[test]
    fn test_sos() { assert_eq!(sum_of_squares(vec![1, 2, 3]), 14); }
    #[test]
    fn test_empty() { assert_eq!(sum_of_squares(vec![]), 0); }
"#)),
            lesson("codegen_composed_chains", "unique_sorted",
                "fn unique_sorted(nums: Vec<i32>) -> Vec<i32>",
                "Deduplicate and sort a vector of numbers",
                Some(r#"
    #[test]
    fn test_dedup() { assert_eq!(unique_sorted(vec![3, 1, 2, 1, 3]), vec![1, 2, 3]); }
"#)),
        ],
    );

    bank.insert(
        "codegen_struct_impl".into(),
        vec![
            lesson("codegen_struct_impl", "Counter",
                "struct Counter { count: i32 }",
                "A simple counter with increment, decrement, and get methods",
                Some(r#"
    #[test]
    fn test_counter() {
        let mut c = Counter { count: 0 };
        c.increment();
        c.increment();
        assert_eq!(c.get(), 2);
        c.decrement();
        assert_eq!(c.get(), 1);
    }
"#)),
        ],
    );

    bank.insert(
        "codegen_error_handling".into(),
        vec![
            lesson("codegen_error_handling", "parse_integer",
                "fn parse_integer(s: &str) -> Result<i32, String>",
                "Parse a string to integer, returning Err with a message on failure",
                Some(r#"
    #[test]
    fn test_ok() { assert_eq!(parse_integer("42"), Ok(42)); }
    #[test]
    fn test_err() { assert!(parse_integer("abc").is_err()); }
"#)),
        ],
    );

    bank.insert(
        "codegen_closures_hof".into(),
        vec![
            lesson("codegen_closures_hof", "apply_twice",
                "fn apply_twice(f: impl Fn(i32) -> i32, x: i32) -> i32",
                "Apply a function to a value twice: f(f(x))",
                Some(r#"
    #[test]
    fn test_twice() { assert_eq!(apply_twice(|x| x + 1, 5), 7); }
    #[test]
    fn test_twice_double() { assert_eq!(apply_twice(|x| x * 2, 3), 12); }
"#)),
        ],
    );

    bank.insert(
        "codegen_test_generation".into(),
        vec![
            lesson("codegen_test_generation", "clamp",
                "fn clamp(value: i32, min: i32, max: i32) -> i32",
                "Clamp a value between min and max",
                Some(r#"
    #[test]
    fn test_below() { assert_eq!(clamp(-5, 0, 10), 0); }
    #[test]
    fn test_above() { assert_eq!(clamp(15, 0, 10), 10); }
    #[test]
    fn test_within() { assert_eq!(clamp(5, 0, 10), 5); }
"#)),
        ],
    );

    // ── Tier 3: Advanced (these may need LLM fallback) ──────────────────

    bank.insert(
        "codegen_algorithm_sorting".into(),
        vec![
            lesson("codegen_algorithm_sorting", "bubble_sort",
                "fn bubble_sort(arr: &mut Vec<i32>)",
                "Sort a vector in-place using bubble sort",
                Some(r#"
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
"#)),
        ],
    );

    bank.insert(
        "codegen_algorithm_search".into(),
        vec![
            lesson("codegen_algorithm_search", "binary_search",
                "fn binary_search(arr: &[i32], target: i32) -> Option<usize>",
                "Find the index of a target value in a sorted slice using binary search",
                Some(r#"
    #[test]
    fn test_found() { assert_eq!(binary_search(&[1, 3, 5, 7, 9], 5), Some(2)); }
    #[test]
    fn test_not_found() { assert_eq!(binary_search(&[1, 3, 5, 7, 9], 4), None); }
    #[test]
    fn test_empty() { assert_eq!(binary_search(&[], 1), None); }
"#)),
        ],
    );

    bank.insert(
        "codegen_algorithm_dp".into(),
        vec![
            lesson("codegen_algorithm_dp", "fibonacci",
                "fn fibonacci(n: u32) -> u64",
                "Compute the nth Fibonacci number using dynamic programming",
                Some(r#"
    #[test]
    fn test_fib_0() { assert_eq!(fibonacci(0), 0); }
    #[test]
    fn test_fib_1() { assert_eq!(fibonacci(1), 1); }
    #[test]
    fn test_fib_10() { assert_eq!(fibonacci(10), 55); }
"#)),
        ],
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
    let spec = CodeSpec::new("rust", name, purpose)
        .with_signature(signature);

    CodeLesson {
        objective_id: objective_id.into(),
        spec,
        test_source: test_source.map(|s| s.to_string()),
    }
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
        }
    }

    /// Run a single lesson through the full pipeline.
    pub fn run_lesson(&mut self, lesson: &CodeLesson) -> LessonOutcome {
        // 1. Build intent and context
        let intent = CodeIntent::Create {
            target: CodeTarget::new("code_learning"),
            spec: lesson.spec.clone(),
        };

        let context = CodeContext {
            past_examples: self.past_examples.clone(),
            ..Default::default()
        };

        // 2. Generate code
        let generated = self.generator.generate(&intent, &context);
        let mut source = generated.source.clone();
        let mut retries = 0;

        // 3. Execute and retry loop
        let mut exec_result = self.execute_lesson(&source, lesson.test_source.as_deref());

        while !exec_result.compiled && retries < MAX_RETRIES {
            // Try auto-fix
            if let Some(fixed) = try_auto_fix(&source, &exec_result.compile_errors) {
                source = fixed;
                retries += 1;
                exec_result = self.execute_lesson(&source, lesson.test_source.as_deref());
            } else {
                break;
            }
        }

        // 4. Check for todo!() — indicates Tier 2 (LLM) would be needed
        let used_llm = source.contains("todo!()") || source.contains("unimplemented!()");

        // 5. Build outcome
        let distillation_eligible = exec_result.compiled
            && exec_result.tests_failed == 0
            && generated.plan_coverage >= MIN_PLAN_COVERAGE
            && !used_llm;

        let outcome = LessonOutcome {
            objective_id: lesson.objective_id.clone(),
            source: source.clone(),
            compiled: exec_result.compiled,
            tests_passed: exec_result.tests_passed,
            tests_failed: exec_result.tests_failed,
            surprise: exec_result.to_surprise(),
            retries_used: retries,
            plan_coverage: generated.plan_coverage,
            phi_score: generated.phi_score,
            used_llm,
            distillation_eligible,
        };

        // 6. Cache successes for distillation and few-shot context
        if distillation_eligible {
            if let Some((_, src, quality)) =
                self.generator.distillation_target(&lesson.spec, &generated)
            {
                self.distillation_cache.push((
                    lesson.spec.purpose.clone(),
                    src,
                    quality,
                ));
            }
            // Add to past examples (capped at 16)
            if self.past_examples.len() < 16 {
                self.past_examples
                    .push((lesson.spec.purpose.clone(), source));
            }
        }

        outcome
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
            let outcomes = self.run_objective(obj_id);
            for outcome in outcomes {
                summary.lessons_attempted += 1;
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
                total_surprise += outcome.surprise;
                total_coverage += outcome.plan_coverage;
                summary.outcomes.push(outcome);
            }
        }

        if summary.lessons_attempted > 0 {
            summary.avg_surprise = total_surprise / summary.lessons_attempted as f32;
            summary.avg_plan_coverage = total_coverage / summary.lessons_attempted as f32;
        }

        summary
    }

    /// Get the distillation cache (for Broca SSM training).
    pub fn distillation_cache(&self) -> &[(String, String, f32)] {
        &self.distillation_cache
    }

    /// Get past successful examples (for few-shot prompting).
    pub fn past_examples(&self) -> &[(String, String)] {
        &self.past_examples
    }

    /// Number of distillation-eligible generations so far.
    pub fn distillation_count(&self) -> usize {
        self.distillation_cache.len()
    }

    fn execute_lesson(&mut self, source: &str, test_source: Option<&str>) -> ExecutionResult {
        self.executor.execute_rust(source, test_source)
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
        // In simulation mode, should always "compile"
        assert!(outcome.compiled, "Simulated execution should succeed");
        assert_eq!(outcome.objective_id, "codegen_simple_arithmetic");
    }

    #[test]
    fn test_run_objective_simulated() {
        let mut engine = make_engine();
        let outcomes = engine.run_objective("codegen_simple_arithmetic");
        assert_eq!(outcomes.len(), 3, "Should run all 3 arithmetic lessons");
        for outcome in &outcomes {
            assert!(outcome.compiled);
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
        // All should compile in simulation mode
        assert_eq!(summary.lessons_compiled, summary.lessons_attempted);
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
            compiled: true,
            tests_passed: 3,
            tests_failed: 0,
            surprise: 0.0,
            retries_used: 0,
            plan_coverage: 1.0,
            phi_score: 0.5,
            used_llm: false,
            distillation_eligible: true,
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
            lessons_compiled: 8,
            lessons_passed: 6,
            ..Default::default()
        };
        assert_eq!(summary.compile_rate(), 80.0);
        assert_eq!(summary.pass_rate(), 60.0);
    }

    #[test]
    fn test_distillation_cache_starts_empty() {
        let engine = make_engine();
        assert_eq!(engine.distillation_count(), 0);
        assert!(engine.distillation_cache().is_empty());
    }
}
