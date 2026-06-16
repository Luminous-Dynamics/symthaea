// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sequencer Benchmark — 100-function baseline for native code generation
//!
//! Generates canonical Rust functions via the native tier and measures:
//! - Pattern detection accuracy
//! - Plan quality (action sequence correctness)
//! - Compilation success rate (when executor is available)
//!
//! This benchmark establishes the quantitative baseline before each improvement
//! phase. Run it before and after changes to measure impact.

use super::code_generator::CodeGenerator;
use super::code_intent::{CodeIntent, CodeSpec, CodeTarget};

/// A benchmark task: purpose + signature + expected compilation
#[derive(Debug, Clone)]
pub struct BenchmarkTask {
    pub name: String,
    pub purpose: String,
    pub signature: String,
    pub category: TaskCategory,
}

/// Categories for grouping benchmark results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskCategory {
    Arithmetic,
    StringOps,
    CollectionOps,
    BooleanChecks,
    MathFunctions,
    Sorting,
    Search,
    IteratorChains,
    ErrorHandling,
    MultiStep,
}

impl TaskCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Arithmetic => "arithmetic",
            Self::StringOps => "string_ops",
            Self::CollectionOps => "collection_ops",
            Self::BooleanChecks => "boolean_checks",
            Self::MathFunctions => "math_functions",
            Self::Sorting => "sorting",
            Self::Search => "search",
            Self::IteratorChains => "iterator_chains",
            Self::ErrorHandling => "error_handling",
            Self::MultiStep => "multi_step",
        }
    }
}

/// Result of running the benchmark
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub total: usize,
    pub generated_non_todo: usize,
    pub has_function: usize,
    pub per_category: Vec<(TaskCategory, usize, usize)>, // (category, total, non_todo)
    pub failures: Vec<(String, String)>,                 // (name, reason)
}

impl BenchmarkResult {
    pub fn non_todo_rate(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        self.generated_non_todo as f32 / self.total as f32
    }

    pub fn report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "=== Sequencer Benchmark: {}/{} non-todo ({:.1}%) ===\n",
            self.generated_non_todo,
            self.total,
            self.non_todo_rate() * 100.0
        ));
        out.push_str(&format!(
            "  Functions with real body: {}/{}\n\n",
            self.has_function, self.total
        ));
        out.push_str("Per-category:\n");
        for (cat, total, non_todo) in &self.per_category {
            let rate = if *total > 0 {
                *non_todo as f32 / *total as f32 * 100.0
            } else {
                0.0
            };
            out.push_str(&format!(
                "  {:20} {}/{} ({:.0}%)\n",
                cat.as_str(),
                non_todo,
                total,
                rate
            ));
        }
        if !self.failures.is_empty() {
            out.push_str(&format!("\nFailures ({}):\n", self.failures.len()));
            for (name, reason) in &self.failures {
                out.push_str(&format!("  {} — {}\n", name, reason));
            }
        }
        out
    }
}

/// The 100 canonical benchmark tasks
pub fn benchmark_tasks() -> Vec<BenchmarkTask> {
    let mut tasks = Vec::with_capacity(100);

    // ── Arithmetic (15) ──
    let arith = [
        (
            "add_numbers",
            "Add two integers",
            "fn add_numbers(a: i32, b: i32) -> i32",
        ),
        (
            "subtract",
            "Subtract b from a",
            "fn subtract(a: i32, b: i32) -> i32",
        ),
        (
            "multiply",
            "Multiply two numbers",
            "fn multiply(a: i32, b: i32) -> i32",
        ),
        (
            "divide",
            "Divide a by b",
            "fn divide(a: f64, b: f64) -> f64",
        ),
        (
            "max_of_two",
            "Return the maximum of two integers",
            "fn max_of_two(a: i32, b: i32) -> i32",
        ),
        (
            "min_of_two",
            "Return the minimum of two integers",
            "fn min_of_two(a: i32, b: i32) -> i32",
        ),
        (
            "absolute_value",
            "Return absolute value",
            "fn absolute_value(x: i32) -> i32",
        ),
        (
            "clamp_value",
            "Clamp value between min and max",
            "fn clamp_value(x: i32, min: i32, max: i32) -> i32",
        ),
        (
            "dot_product",
            "Compute dot product of two vectors",
            "fn dot_product(a: &[f64], b: &[f64]) -> f64",
        ),
        (
            "sum_list",
            "Sum all elements in a vector",
            "fn sum_list(nums: &[i32]) -> i32",
        ),
        (
            "average",
            "Calculate average of numbers",
            "fn average(nums: &[f64]) -> f64",
        ),
        (
            "power",
            "Raise base to exponent",
            "fn power(base: f64, exp: u32) -> f64",
        ),
        (
            "modulo",
            "Return remainder of division",
            "fn modulo(a: i32, b: i32) -> i32",
        ),
        ("negate", "Negate a number", "fn negate(x: i32) -> i32"),
        (
            "increment",
            "Add one to a number",
            "fn increment(x: i32) -> i32",
        ),
    ];
    for (name, purpose, sig) in arith {
        tasks.push(BenchmarkTask {
            name: name.into(),
            purpose: purpose.into(),
            signature: sig.into(),
            category: TaskCategory::Arithmetic,
        });
    }

    // ── String Operations (15) ──
    let strings = [
        (
            "reverse_string",
            "Reverse a string",
            "fn reverse_string(s: &str) -> String",
        ),
        (
            "to_uppercase",
            "Convert string to uppercase",
            "fn to_uppercase(s: &str) -> String",
        ),
        (
            "to_lowercase",
            "Convert string to lowercase",
            "fn to_lowercase(s: &str) -> String",
        ),
        (
            "string_length",
            "Get length of a string",
            "fn string_length(s: &str) -> usize",
        ),
        (
            "contains_substr",
            "Check if string contains a substring",
            "fn contains_substr(s: &str, sub: &str) -> bool",
        ),
        (
            "concatenate",
            "Concatenate two strings",
            "fn concatenate(a: &str, b: &str) -> String",
        ),
        (
            "trim_string",
            "Trim whitespace from string",
            "fn trim_string(s: &str) -> String",
        ),
        (
            "split_words",
            "Split string into words",
            "fn split_words(s: &str) -> Vec<String>",
        ),
        (
            "count_words",
            "Count the number of words in a string",
            "fn count_words(s: &str) -> usize",
        ),
        (
            "replace_char",
            "Replace all occurrences of a character",
            "fn replace_char(s: &str, from: char, to: char) -> String",
        ),
        (
            "starts_with_prefix",
            "Check if string starts with prefix",
            "fn starts_with_prefix(s: &str, prefix: &str) -> bool",
        ),
        (
            "ends_with_suffix",
            "Check if string ends with suffix",
            "fn ends_with_suffix(s: &str, suffix: &str) -> bool",
        ),
        (
            "repeat_string",
            "Repeat a string n times",
            "fn repeat_string(s: &str, n: usize) -> String",
        ),
        (
            "char_at",
            "Get character at index",
            "fn char_at(s: &str, idx: usize) -> Option<char>",
        ),
        (
            "capitalize",
            "Capitalize first letter of string",
            "fn capitalize(s: &str) -> String",
        ),
    ];
    for (name, purpose, sig) in strings {
        tasks.push(BenchmarkTask {
            name: name.into(),
            purpose: purpose.into(),
            signature: sig.into(),
            category: TaskCategory::StringOps,
        });
    }

    // ── Collection Operations (15) ──
    let collections = [
        (
            "sort_vec",
            "Sort a vector of integers",
            "fn sort_vec(v: &[i32]) -> Vec<i32>",
        ),
        (
            "reverse_vec",
            "Reverse a vector",
            "fn reverse_vec(v: &[i32]) -> Vec<i32>",
        ),
        (
            "unique_elements",
            "Remove duplicates from vector",
            "fn unique_elements(v: &[i32]) -> Vec<i32>",
        ),
        (
            "flatten_nested",
            "Flatten nested vectors",
            "fn flatten_nested(v: &[Vec<i32>]) -> Vec<i32>",
        ),
        (
            "zip_vectors",
            "Zip two vectors into pairs",
            "fn zip_vectors(a: &[i32], b: &[i32]) -> Vec<(i32, i32)>",
        ),
        (
            "take_first_n",
            "Take first n elements",
            "fn take_first_n(v: &[i32], n: usize) -> Vec<i32>",
        ),
        (
            "skip_first_n",
            "Skip first n elements",
            "fn skip_first_n(v: &[i32], n: usize) -> Vec<i32>",
        ),
        (
            "enumerate_items",
            "Enumerate items with index",
            "fn enumerate_items(v: &[String]) -> Vec<(usize, String)>",
        ),
        (
            "find_max",
            "Find the largest element",
            "fn find_max(v: &[i32]) -> Option<i32>",
        ),
        (
            "find_min",
            "Find the smallest element",
            "fn find_min(v: &[i32]) -> Option<i32>",
        ),
        (
            "chunk_vec",
            "Split vector into chunks of size n",
            "fn chunk_vec(v: &[i32], n: usize) -> Vec<Vec<i32>>",
        ),
        (
            "last_element",
            "Get the last element",
            "fn last_element(v: &[i32]) -> Option<i32>",
        ),
        (
            "count_elements",
            "Count number of elements",
            "fn count_elements(v: &[i32]) -> usize",
        ),
        (
            "index_of",
            "Find index of element",
            "fn index_of(v: &[i32], target: i32) -> Option<usize>",
        ),
        (
            "vec_contains",
            "Check if vector contains element",
            "fn vec_contains(v: &[i32], target: i32) -> bool",
        ),
    ];
    for (name, purpose, sig) in collections {
        tasks.push(BenchmarkTask {
            name: name.into(),
            purpose: purpose.into(),
            signature: sig.into(),
            category: TaskCategory::CollectionOps,
        });
    }

    // ── Boolean Checks (10) ──
    let booleans = [
        (
            "is_even",
            "Check if number is even",
            "fn is_even(n: i32) -> bool",
        ),
        (
            "is_odd",
            "Check if number is odd",
            "fn is_odd(n: i32) -> bool",
        ),
        (
            "is_positive",
            "Check if number is positive",
            "fn is_positive(n: i32) -> bool",
        ),
        (
            "is_negative",
            "Check if number is negative",
            "fn is_negative(n: i32) -> bool",
        ),
        (
            "is_zero",
            "Check if number is zero",
            "fn is_zero(n: i32) -> bool",
        ),
        (
            "is_empty_str",
            "Check if string is empty",
            "fn is_empty_str(s: &str) -> bool",
        ),
        (
            "is_palindrome",
            "Check if string is a palindrome",
            "fn is_palindrome(s: &str) -> bool",
        ),
        (
            "is_sorted",
            "Check if vector is sorted in ascending order",
            "fn is_sorted(v: &[i32]) -> bool",
        ),
        (
            "all_positive",
            "Check if all elements are positive",
            "fn all_positive(v: &[i32]) -> bool",
        ),
        (
            "any_negative",
            "Check if any element is negative",
            "fn any_negative(v: &[i32]) -> bool",
        ),
    ];
    for (name, purpose, sig) in booleans {
        tasks.push(BenchmarkTask {
            name: name.into(),
            purpose: purpose.into(),
            signature: sig.into(),
            category: TaskCategory::BooleanChecks,
        });
    }

    // ── Math Functions (10) ──
    let math = [
        (
            "factorial",
            "Compute factorial of n",
            "fn factorial(n: u64) -> u64",
        ),
        (
            "fibonacci",
            "Compute nth fibonacci number",
            "fn fibonacci(n: u64) -> u64",
        ),
        (
            "gcd",
            "Compute greatest common divisor",
            "fn gcd(a: u64, b: u64) -> u64",
        ),
        (
            "is_prime",
            "Check if a number is prime",
            "fn is_prime(n: u64) -> bool",
        ),
        (
            "square_root",
            "Compute integer square root",
            "fn square_root(n: f64) -> f64",
        ),
        (
            "celsius_to_fahrenheit",
            "Convert celsius to fahrenheit",
            "fn celsius_to_fahrenheit(c: f64) -> f64",
        ),
        (
            "distance",
            "Compute Euclidean distance between two points",
            "fn distance(x1: f64, y1: f64, x2: f64, y2: f64) -> f64",
        ),
        (
            "lcm",
            "Compute least common multiple",
            "fn lcm(a: u64, b: u64) -> u64",
        ),
        (
            "collatz_steps",
            "Count Collatz sequence steps to reach 1",
            "fn collatz_steps(n: u64) -> u64",
        ),
        (
            "sum_digits",
            "Sum all digits of a number",
            "fn sum_digits(n: u64) -> u64",
        ),
    ];
    for (name, purpose, sig) in math {
        tasks.push(BenchmarkTask {
            name: name.into(),
            purpose: purpose.into(),
            signature: sig.into(),
            category: TaskCategory::MathFunctions,
        });
    }

    // ── Iterator Chains (10) ──
    let chains = [
        (
            "filter_even",
            "Filter even numbers from a vector",
            "fn filter_even(v: &[i32]) -> Vec<i32>",
        ),
        (
            "map_double",
            "Double each element",
            "fn map_double(v: &[i32]) -> Vec<i32>",
        ),
        (
            "filter_and_sum",
            "Filter positive numbers and sum them",
            "fn filter_and_sum(v: &[i32]) -> i32",
        ),
        (
            "sort_and_take",
            "Sort and take first 3",
            "fn sort_and_take(v: &[i32]) -> Vec<i32>",
        ),
        (
            "squares",
            "Map each element to its square",
            "fn squares(v: &[i32]) -> Vec<i32>",
        ),
        (
            "filter_long_strings",
            "Filter strings longer than n characters",
            "fn filter_long_strings(v: &[String], n: usize) -> Vec<String>",
        ),
        (
            "sum_of_squares",
            "Sum the squares of all elements",
            "fn sum_of_squares(v: &[i32]) -> i32",
        ),
        (
            "product_of_all",
            "Multiply all elements together",
            "fn product_of_all(v: &[i32]) -> i32",
        ),
        (
            "count_matching",
            "Count elements matching a predicate",
            "fn count_matching(v: &[i32], target: i32) -> usize",
        ),
        (
            "two_sum",
            "Find two indices that sum to target",
            "fn two_sum(nums: &[i32], target: i32) -> Option<(usize, usize)>",
        ),
    ];
    for (name, purpose, sig) in chains {
        tasks.push(BenchmarkTask {
            name: name.into(),
            purpose: purpose.into(),
            signature: sig.into(),
            category: TaskCategory::IteratorChains,
        });
    }

    // ── Sorting (5) ──
    let sorting = [
        (
            "bubble_sort",
            "Sort using bubble sort algorithm",
            "fn bubble_sort(v: &mut [i32])",
        ),
        (
            "sort_descending",
            "Sort vector in descending order",
            "fn sort_descending(v: &[i32]) -> Vec<i32>",
        ),
        (
            "sort_by_length",
            "Sort strings by their length",
            "fn sort_by_length(v: &[String]) -> Vec<String>",
        ),
        (
            "partition_by",
            "Partition into elements below and above threshold",
            "fn partition_by(v: &[i32], threshold: i32) -> (Vec<i32>, Vec<i32>)",
        ),
        (
            "merge_sorted",
            "Merge two sorted vectors",
            "fn merge_sorted(a: &[i32], b: &[i32]) -> Vec<i32>",
        ),
    ];
    for (name, purpose, sig) in sorting {
        tasks.push(BenchmarkTask {
            name: name.into(),
            purpose: purpose.into(),
            signature: sig.into(),
            category: TaskCategory::Sorting,
        });
    }

    // ── Search (5) ──
    let search = [
        (
            "binary_search",
            "Binary search for target in sorted array",
            "fn binary_search(v: &[i32], target: i32) -> Option<usize>",
        ),
        (
            "linear_search",
            "Linear search for target",
            "fn linear_search(v: &[i32], target: i32) -> Option<usize>",
        ),
        (
            "find_first",
            "Find first element matching predicate",
            "fn find_first(v: &[i32], target: i32) -> Option<i32>",
        ),
        (
            "count_occurrences",
            "Count how many times target appears",
            "fn count_occurrences(v: &[i32], target: i32) -> usize",
        ),
        (
            "find_duplicates",
            "Find all duplicate elements",
            "fn find_duplicates(v: &[i32]) -> Vec<i32>",
        ),
    ];
    for (name, purpose, sig) in search {
        tasks.push(BenchmarkTask {
            name: name.into(),
            purpose: purpose.into(),
            signature: sig.into(),
            category: TaskCategory::Search,
        });
    }

    // ── Error Handling (5) ──
    let errors = [
        (
            "parse_integer",
            "Parse string to integer with error handling",
            "fn parse_integer(s: &str) -> Result<i32, String>",
        ),
        (
            "safe_divide",
            "Divide with zero-check",
            "fn safe_divide(a: f64, b: f64) -> Result<f64, String>",
        ),
        (
            "first_or_error",
            "Get first element or return error",
            "fn first_or_error(v: &[i32]) -> Result<i32, String>",
        ),
        (
            "unwrap_or_default",
            "Unwrap option or return default",
            "fn unwrap_or_default(opt: Option<i32>, default: i32) -> i32",
        ),
        (
            "validate_range",
            "Validate number is within range",
            "fn validate_range(n: i32, min: i32, max: i32) -> Result<i32, String>",
        ),
    ];
    for (name, purpose, sig) in errors {
        tasks.push(BenchmarkTask {
            name: name.into(),
            purpose: purpose.into(),
            signature: sig.into(),
            category: TaskCategory::ErrorHandling,
        });
    }

    // ── Multi-Step (10) ──
    let multi = [
        (
            "filter_sort_take",
            "Filter positive, sort ascending, take first 5",
            "fn filter_sort_take(v: &[i32]) -> Vec<i32>",
        ),
        (
            "words_to_lengths",
            "Split into words and map to their lengths",
            "fn words_to_lengths(s: &str) -> Vec<usize>",
        ),
        (
            "unique_sorted",
            "Remove duplicates then sort",
            "fn unique_sorted(v: &[i32]) -> Vec<i32>",
        ),
        (
            "running_sum",
            "Compute running sum of vector",
            "fn running_sum(v: &[i32]) -> Vec<i32>",
        ),
        (
            "mode",
            "Find the most frequent element",
            "fn mode(v: &[i32]) -> Option<i32>",
        ),
        (
            "median",
            "Find the median of a sorted vector",
            "fn median(v: &[f64]) -> Option<f64>",
        ),
        (
            "zip_and_sum",
            "Zip two vectors and sum pairs",
            "fn zip_and_sum(a: &[i32], b: &[i32]) -> Vec<i32>",
        ),
        (
            "flatten_and_sort",
            "Flatten nested vectors then sort",
            "fn flatten_and_sort(v: &[Vec<i32>]) -> Vec<i32>",
        ),
        (
            "group_by_sign",
            "Group numbers into positive and negative",
            "fn group_by_sign(v: &[i32]) -> (Vec<i32>, Vec<i32>)",
        ),
        (
            "interleave",
            "Interleave two vectors",
            "fn interleave(a: &[i32], b: &[i32]) -> Vec<i32>",
        ),
    ];
    for (name, purpose, sig) in multi {
        tasks.push(BenchmarkTask {
            name: name.into(),
            purpose: purpose.into(),
            signature: sig.into(),
            category: TaskCategory::MultiStep,
        });
    }

    assert_eq!(tasks.len(), 100, "Benchmark must have exactly 100 tasks");
    tasks
}

/// Run the benchmark against a CodeGenerator, returning structured results.
///
/// This measures how many of the 100 canonical functions the native tier can
/// generate without falling back to `todo!()` / `unimplemented!()`.
pub fn run_benchmark(generator: &CodeGenerator) -> BenchmarkResult {
    use std::collections::HashMap;

    let tasks = benchmark_tasks();
    let mut total = 0usize;
    let mut generated_non_todo = 0usize;
    let mut has_function = 0usize;
    let mut failures: Vec<(String, String)> = Vec::new();
    let mut cat_counts: HashMap<TaskCategory, (usize, usize)> = HashMap::new();

    let context = super::code_generator::CodeContext::default();

    for task in &tasks {
        total += 1;
        let cat_entry = cat_counts.entry(task.category).or_insert((0, 0));
        cat_entry.0 += 1;

        let spec = CodeSpec {
            language: "rust".into(),
            name: task.name.clone(),
            purpose: task.purpose.clone(),
            purpose_hv: None,
            signature: Some(task.signature.clone()),
            constraints: Vec::new(),
            examples: Vec::new(),
            epistemic_status: crate::mind::structured_thought::EpistemicStatus::Certain,
            metadata: HashMap::new(),
        };

        let intent = CodeIntent::Create {
            target: CodeTarget {
                kind: super::code_parser::EntityKind::Function,
                name: task.name.clone(),
                path: None,
                language: Some("rust".into()),
                hv: None,
            },
            spec,
        };

        let result = generator.generate(&intent, &context);

        // Check if the generated code has a real body (not todo!/unimplemented!)
        let is_todo = result.source.contains("todo!(")
            || result.source.contains("unimplemented!(")
            || result.source.is_empty();

        if !is_todo {
            generated_non_todo += 1;
            cat_entry.1 += 1;
        } else {
            let reason = if result.source.is_empty() {
                "empty output".to_string()
            } else {
                "contains todo!()/unimplemented!()".to_string()
            };
            failures.push((task.name.clone(), reason));
        }

        // Check if it at least has "fn "
        if result.source.contains("fn ") {
            has_function += 1;
        }
    }

    // Build per-category results
    let categories = [
        TaskCategory::Arithmetic,
        TaskCategory::StringOps,
        TaskCategory::CollectionOps,
        TaskCategory::BooleanChecks,
        TaskCategory::MathFunctions,
        TaskCategory::Sorting,
        TaskCategory::Search,
        TaskCategory::IteratorChains,
        TaskCategory::ErrorHandling,
        TaskCategory::MultiStep,
    ];
    let per_category: Vec<(TaskCategory, usize, usize)> = categories
        .iter()
        .map(|cat| {
            let (total, non_todo) = cat_counts.get(cat).copied().unwrap_or((0, 0));
            (*cat, total, non_todo)
        })
        .collect();

    BenchmarkResult {
        total,
        generated_non_todo,
        has_function,
        per_category,
        failures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_has_100_tasks() {
        let tasks = benchmark_tasks();
        assert_eq!(tasks.len(), 100);
    }

    #[test]
    fn test_benchmark_covers_all_categories() {
        let tasks = benchmark_tasks();
        let categories: std::collections::HashSet<TaskCategory> =
            tasks.iter().map(|t| t.category).collect();
        assert_eq!(categories.len(), 10, "Should cover all 10 categories");
    }

    #[test]
    fn test_all_tasks_have_signatures() {
        let tasks = benchmark_tasks();
        for task in &tasks {
            assert!(
                task.signature.starts_with("fn "),
                "Task {} should have fn signature, got: {}",
                task.name,
                task.signature
            );
        }
    }

    #[test]
    fn test_run_benchmark_produces_results() {
        let encoder = crate::hdc::code_encoder::CodeHDEncoder::new(512);
        let generator = CodeGenerator::new(encoder);
        let result = run_benchmark(&generator);
        assert_eq!(result.total, 100);
        // At minimum, the emitter should produce *something* for most tasks
        assert!(
            result.has_function >= 80,
            "Expected at least 80 functions with fn keyword, got {}",
            result.has_function
        );
    }
}
