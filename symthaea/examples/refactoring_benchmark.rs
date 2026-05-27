// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-File Refactoring Benchmark Suite
//!
//! Tests Symthaea's ability to perform structural code transformations:
//! - Extract function
//! - Rename symbol
//! - Change type signature (verify callers update)
//! - Move code between files
//! - Add error handling
//!
//! Each task provides before/after source files and validates that the
//! transformation preserves semantics (tests still pass after refactoring).
//!
//! Run:
//!   `cargo run --example refactoring_benchmark --features code_generation`

use std::time::Instant;

/// A refactoring benchmark task.
struct RefactorTask {
    /// Human-readable task description
    description: &'static str,
    /// Category of refactoring
    category: RefactorCategory,
    /// Source code before refactoring
    before_source: &'static str,
    /// Expected key patterns in the refactored code
    expected_patterns: &'static [&'static str],
    /// Patterns that should NOT appear after refactoring
    forbidden_patterns: &'static [&'static str],
    /// Test source that should compile and pass after refactoring
    test_source: &'static str,
}

#[derive(Debug, Clone, Copy)]
enum RefactorCategory {
    ExtractFunction,
    RenameSymbol,
    ChangeSignature,
    AddErrorHandling,
    SimplifyLogic,
}

impl std::fmt::Display for RefactorCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExtractFunction => write!(f, "ExtractFn"),
            Self::RenameSymbol => write!(f, "Rename"),
            Self::ChangeSignature => write!(f, "ChangeSig"),
            Self::AddErrorHandling => write!(f, "AddErrH"),
            Self::SimplifyLogic => write!(f, "Simplify"),
        }
    }
}

fn define_tasks() -> Vec<RefactorTask> {
    vec![
        // === Extract Function ===
        RefactorTask {
            description: "Extract repeated validation logic into a validate_age() function",
            category: RefactorCategory::ExtractFunction,
            before_source: r#"
pub fn create_user(name: &str, age: u32) -> Result<String, String> {
    if age < 1 || age > 150 {
        return Err("Invalid age".to_string());
    }
    Ok(format!("User: {}, Age: {}", name, age))
}

pub fn update_age(current: u32, new_age: u32) -> Result<u32, String> {
    if new_age < 1 || new_age > 150 {
        return Err("Invalid age".to_string());
    }
    Ok(new_age)
}
"#,
            expected_patterns: &["fn validate_age", "validate_age("],
            forbidden_patterns: &[],
            test_source: r#"
#[test]
fn test_create_user() {
    assert!(create_user("Alice", 25).is_ok());
    assert!(create_user("Bob", 0).is_err());
    assert!(create_user("Carol", 200).is_err());
}
#[test]
fn test_update_age() {
    assert_eq!(update_age(25, 30), Ok(30));
    assert!(update_age(25, 0).is_err());
}
"#,
        },
        RefactorTask {
            description: "Extract sorting comparison into a compare_by_priority() function",
            category: RefactorCategory::ExtractFunction,
            before_source: r#"
pub fn sort_tasks(tasks: &mut [(String, u32)]) {
    tasks.sort_by(|a, b| {
        if a.1 == b.1 {
            a.0.cmp(&b.0)
        } else {
            b.1.cmp(&a.1)
        }
    });
}
"#,
            expected_patterns: &["fn compare_by_priority", "compare_by_priority("],
            forbidden_patterns: &[],
            test_source: r#"
#[test]
fn test_sort_tasks() {
    let mut tasks = vec![
        ("low".to_string(), 1),
        ("high".to_string(), 3),
        ("med".to_string(), 2),
    ];
    sort_tasks(&mut tasks);
    assert_eq!(tasks[0].0, "high");
    assert_eq!(tasks[2].0, "low");
}
"#,
        },
        // === Rename Symbol ===
        RefactorTask {
            description: "Rename function 'calc' to 'calculate_total' across all call sites",
            category: RefactorCategory::RenameSymbol,
            before_source: r#"
pub fn calc(items: &[f64], tax_rate: f64) -> f64 {
    let subtotal: f64 = items.iter().sum();
    subtotal * (1.0 + tax_rate)
}

pub fn invoice(items: &[f64]) -> String {
    let total = calc(items, 0.08);
    format!("Total: ${:.2}", total)
}

pub fn receipt(items: &[f64], rate: f64) -> f64 {
    calc(items, rate)
}
"#,
            expected_patterns: &[
                "fn calculate_total",
                "calculate_total(items",
                "calculate_total(items, 0.08)",
                "calculate_total(items, rate)",
            ],
            forbidden_patterns: &["fn calc(", "calc(items"],
            test_source: r#"
#[test]
fn test_calculate_total() {
    let items = vec![10.0, 20.0];
    let total = calculate_total(&items, 0.1);
    assert!((total - 33.0).abs() < 0.01);
}
#[test]
fn test_invoice() {
    let items = vec![100.0];
    let inv = invoice(&items);
    assert!(inv.contains("108.00"));
}
"#,
        },
        RefactorTask {
            description: "Rename struct 'Data' to 'SensorReading' and update all usages",
            category: RefactorCategory::RenameSymbol,
            before_source: r#"
pub struct Data {
    pub value: f64,
    pub timestamp: u64,
}

impl Data {
    pub fn new(value: f64, timestamp: u64) -> Data {
        Data { value, timestamp }
    }
    pub fn is_valid(&self) -> bool {
        self.value.is_finite() && self.timestamp > 0
    }
}

pub fn average(readings: &[Data]) -> f64 {
    if readings.is_empty() { return 0.0; }
    readings.iter().map(|d| d.value).sum::<f64>() / readings.len() as f64
}
"#,
            expected_patterns: &[
                "struct SensorReading",
                "SensorReading {",
                "SensorReading::new",
                "&[SensorReading]",
            ],
            forbidden_patterns: &["struct Data", "Data {", "Data::new"],
            test_source: r#"
#[test]
fn test_sensor_reading() {
    let r = SensorReading::new(42.0, 1000);
    assert!(r.is_valid());
    assert!((r.value - 42.0).abs() < 0.01);
}
#[test]
fn test_average() {
    let readings = vec![
        SensorReading::new(10.0, 1),
        SensorReading::new(20.0, 2),
    ];
    assert!((average(&readings) - 15.0).abs() < 0.01);
}
"#,
        },
        // === Change Signature ===
        RefactorTask {
            description: "Change process() to return Result<String, ProcessError> instead of String",
            category: RefactorCategory::ChangeSignature,
            before_source: r#"
pub fn process(input: &str) -> String {
    if input.is_empty() {
        return "empty".to_string();
    }
    input.to_uppercase()
}

pub fn batch_process(items: &[&str]) -> Vec<String> {
    items.iter().map(|i| process(i)).collect()
}
"#,
            expected_patterns: &["Result<String", "ProcessError", "fn process("],
            forbidden_patterns: &[],
            test_source: r#"
#[test]
fn test_process_ok() {
    assert_eq!(process("hello").unwrap(), "HELLO");
}
#[test]
fn test_process_empty_err() {
    assert!(process("").is_err());
}
"#,
        },
        // === Add Error Handling ===
        RefactorTask {
            description: "Add proper error handling to divide() — return Result instead of panicking",
            category: RefactorCategory::AddErrorHandling,
            before_source: r#"
pub fn divide(a: f64, b: f64) -> f64 {
    a / b
}

pub fn safe_ratio(values: &[f64]) -> f64 {
    let sum: f64 = values.iter().sum();
    divide(sum, values.len() as f64)
}
"#,
            expected_patterns: &["Result<f64", "Err("],
            forbidden_patterns: &[],
            test_source: r#"
#[test]
fn test_divide_ok() {
    assert!((divide(10.0, 2.0).unwrap() - 5.0).abs() < 0.01);
}
#[test]
fn test_divide_by_zero() {
    assert!(divide(10.0, 0.0).is_err());
}
"#,
        },
        RefactorTask {
            description: "Add bounds checking to get_element() — return Option instead of indexing",
            category: RefactorCategory::AddErrorHandling,
            before_source: r#"
pub fn get_element(arr: &[i32], idx: usize) -> i32 {
    arr[idx]
}
"#,
            expected_patterns: &["Option<i32>", "fn get_element"],
            forbidden_patterns: &["arr[idx]"],
            test_source: r#"
#[test]
fn test_get_element_valid() {
    assert_eq!(get_element(&[1, 2, 3], 1), Some(2));
}
#[test]
fn test_get_element_oob() {
    assert_eq!(get_element(&[1, 2, 3], 10), None);
}
"#,
        },
        // === Simplify Logic ===
        RefactorTask {
            description: "Simplify nested if-else chain into a match expression",
            category: RefactorCategory::SimplifyLogic,
            before_source: r#"
pub fn grade(score: u32) -> &'static str {
    if score >= 90 {
        "A"
    } else if score >= 80 {
        "B"
    } else if score >= 70 {
        "C"
    } else if score >= 60 {
        "D"
    } else {
        "F"
    }
}
"#,
            expected_patterns: &["match", "=>"],
            forbidden_patterns: &["else if"],
            test_source: r#"
#[test]
fn test_grades() {
    assert_eq!(grade(95), "A");
    assert_eq!(grade(85), "B");
    assert_eq!(grade(75), "C");
    assert_eq!(grade(65), "D");
    assert_eq!(grade(50), "F");
}
"#,
        },
        RefactorTask {
            description: "Replace manual loop with iterator chain (map/filter/collect)",
            category: RefactorCategory::SimplifyLogic,
            before_source: r#"
pub fn even_squares(nums: &[i32]) -> Vec<i32> {
    let mut result = Vec::new();
    for n in nums {
        if n % 2 == 0 {
            result.push(n * n);
        }
    }
    result
}
"#,
            expected_patterns: &[".filter(", ".map(", ".collect()"],
            forbidden_patterns: &["let mut result", "result.push"],
            test_source: r#"
#[test]
fn test_even_squares() {
    assert_eq!(even_squares(&[1, 2, 3, 4, 5]), vec![4, 16]);
    assert_eq!(even_squares(&[]), vec![]);
}
"#,
        },
    ]
}

#[derive(Debug)]
struct TaskResult {
    description: String,
    category: String,
    has_expected: bool,
    has_no_forbidden: bool,
    would_compile: bool,
    elapsed_ms: u128,
}

fn main() {
    let tasks = define_tasks();
    println!("=== Multi-File Refactoring Benchmark ===");
    println!("{} refactoring tasks\n", tasks.len());

    let mut results: Vec<TaskResult> = Vec::new();

    for (i, task) in tasks.iter().enumerate() {
        let start = Instant::now();

        // For now, this benchmark validates the task definitions themselves.
        // When wired to the CodingAgent, each task would be:
        // 1. Write before_source to a temp file
        // 2. Ask agent to perform the refactoring
        // 3. Check expected_patterns present in output
        // 4. Check forbidden_patterns absent
        // 5. Compile with test_source and verify tests pass

        // Validate task structure
        let has_expected = !task.expected_patterns.is_empty();
        let has_no_forbidden = true; // Will check against actual output when agent is wired

        // Check that before_source compiles (baseline validity)
        // For now just validate the task is well-formed
        let would_compile =
            task.before_source.contains("fn ") && task.test_source.contains("#[test]");

        let result = TaskResult {
            description: task.description.to_string(),
            category: format!("{}", task.category),
            has_expected,
            has_no_forbidden,
            would_compile,
            elapsed_ms: start.elapsed().as_millis(),
        };

        let status = if result.would_compile && result.has_expected {
            "READY"
        } else {
            "ISSUE"
        };

        println!(
            "  [{:>2}] [{}] {:>10} | {}",
            i + 1,
            status,
            result.category,
            &result.description[..result.description.len().min(70)]
        );

        results.push(result);
    }

    // Summary
    println!("\n=== SUMMARY ===");
    let ready = results
        .iter()
        .filter(|r| r.would_compile && r.has_expected)
        .count();
    println!("Tasks ready: {}/{}", ready, results.len());

    // Category breakdown
    let categories: Vec<&str> = vec!["ExtractFn", "Rename", "ChangeSig", "AddErrH", "Simplify"];
    for cat in categories {
        let count = results.iter().filter(|r| r.category == cat).count();
        if count > 0 {
            println!("  {}: {} tasks", cat, count);
        }
    }

    println!("\nNote: This benchmark currently validates task definitions.");
    println!("Wire to CodingAgent for end-to-end refactoring measurement.");
    println!("See `examples/coding_agent_benchmark.rs` for the agent integration pattern.");
}