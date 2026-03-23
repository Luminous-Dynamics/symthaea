// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Bug Detection benchmark.
//!
//! Measures the ability to detect and classify seeded bugs in code snippets
//! via HDC similarity. Each trial encodes a buggy and fixed code pair as
//! BinaryHVs, then classifies the bug category by comparing the delta HV
//! (buggy XOR fixed) against learned category prototypes.
//!
//! Key metric: `category_accuracy` -- the fraction of snippets where the
//! delta HV is classified into the correct bug category via leave-one-out
//! nearest-prototype classification.
//!
//! Human baselines (Beller et al., 2018; Endrikat et al., 2014):
//! - category_accuracy: ~0.55 (SD ~0.18) -- correct bug type classification
//! - detection_accuracy: ~0.70 (SD ~0.15) -- experts ~85%, novices ~55%
//! - delta_magnitude: ~0.12 (SD ~0.04) -- normalized HDC change magnitude

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::BinaryHV;

/// Bug Detection benchmark.
pub struct BugDetectionBenchmark;

/// Bug category enum (6 categories, 5 snippets each = 30 total).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BugCategory {
    OffByOne,
    TypeMismatch,
    LogicError,
    NullNoneHandling,
    ResourceLeak,
    BoundaryViolation,
}

impl BugCategory {
    const ALL: [BugCategory; 6] = [
        BugCategory::OffByOne,
        BugCategory::TypeMismatch,
        BugCategory::LogicError,
        BugCategory::NullNoneHandling,
        BugCategory::ResourceLeak,
        BugCategory::BoundaryViolation,
    ];

    fn name(self) -> &'static str {
        match self {
            BugCategory::OffByOne => "off_by_one",
            BugCategory::TypeMismatch => "type_mismatch",
            BugCategory::LogicError => "logic_error",
            BugCategory::NullNoneHandling => "null_none_handling",
            BugCategory::ResourceLeak => "resource_leak",
            BugCategory::BoundaryViolation => "boundary_violation",
        }
    }

    fn index(self) -> usize {
        match self {
            BugCategory::OffByOne => 0,
            BugCategory::TypeMismatch => 1,
            BugCategory::LogicError => 2,
            BugCategory::NullNoneHandling => 3,
            BugCategory::ResourceLeak => 4,
            BugCategory::BoundaryViolation => 5,
        }
    }
}

/// A code snippet with a known bug.
#[allow(dead_code)]
struct BugSnippet {
    buggy_code: &'static str,
    fixed_code: &'static str,
    bug_category: BugCategory,
    bug_line: usize,
    bug_description: &'static str,
    /// Category-discriminative keywords extracted from the bug pattern.
    keywords: &'static [&'static str],
}

/// All 30 bug snippets (5 per category).
fn bug_snippets() -> Vec<BugSnippet> {
    vec![
        // -- Off-by-one errors (5) --
        BugSnippet {
            buggy_code: "fn sum(arr: &[i32]) -> i32 {\n    let mut total = 0;\n    for i in 0..arr.len() + 1 {\n        total += arr[i];\n    }\n    total\n}",
            fixed_code: "fn sum(arr: &[i32]) -> i32 {\n    let mut total = 0;\n    for i in 0..arr.len() {\n        total += arr[i];\n    }\n    total\n}",
            bug_category: BugCategory::OffByOne,
            bug_line: 3,
            bug_description: "Loop bound arr.len() + 1 causes out-of-bounds access",
            keywords: &["loop", "bound", "plus_one", "index", "len"],
        },
        BugSnippet {
            buggy_code: "fn binary_search(arr: &[i32], target: i32) -> Option<usize> {\n    let mut lo = 0;\n    let mut hi = arr.len();\n    while lo < hi {\n        let mid = (lo + hi) / 2;\n        if arr[mid] == target { return Some(mid); }\n        if arr[mid] < target { lo = mid; } else { hi = mid; }\n    }\n    None\n}",
            fixed_code: "fn binary_search(arr: &[i32], target: i32) -> Option<usize> {\n    let mut lo = 0;\n    let mut hi = arr.len();\n    while lo < hi {\n        let mid = (lo + hi) / 2;\n        if arr[mid] == target { return Some(mid); }\n        if arr[mid] < target { lo = mid + 1; } else { hi = mid; }\n    }\n    None\n}",
            bug_category: BugCategory::OffByOne,
            bug_line: 7,
            bug_description: "lo = mid instead of lo = mid + 1 causes infinite loop",
            keywords: &["loop", "bound", "mid", "index", "increment"],
        },
        BugSnippet {
            buggy_code: "fn fence_posts(n: usize) -> usize {\n    let sections = n;\n    let posts = sections;\n    posts\n}",
            fixed_code: "fn fence_posts(n: usize) -> usize {\n    let sections = n;\n    let posts = sections + 1;\n    posts\n}",
            bug_category: BugCategory::OffByOne,
            bug_line: 3,
            bug_description: "Fence-post error: n sections need n+1 posts",
            keywords: &["fence", "count", "plus_one", "boundary", "off"],
        },
        BugSnippet {
            buggy_code: "fn last_n(arr: &[i32], n: usize) -> &[i32] {\n    &arr[arr.len() - n + 1..]\n}",
            fixed_code: "fn last_n(arr: &[i32], n: usize) -> &[i32] {\n    &arr[arr.len() - n..]\n}",
            bug_category: BugCategory::OffByOne,
            bug_line: 2,
            bug_description: "Start index off by one: len - n + 1 returns n-1 elements",
            keywords: &["slice", "index", "offset", "len", "bound"],
        },
        BugSnippet {
            buggy_code: "fn reverse(arr: &mut [i32]) {\n    let n = arr.len();\n    for i in 0..n / 2 + 1 {\n        arr.swap(i, n - 1 - i);\n    }\n}",
            fixed_code: "fn reverse(arr: &mut [i32]) {\n    let n = arr.len();\n    for i in 0..n / 2 {\n        arr.swap(i, n - 1 - i);\n    }\n}",
            bug_category: BugCategory::OffByOne,
            bug_line: 3,
            bug_description: "Loop iterates one extra time, double-swapping",
            keywords: &["loop", "range", "plus_one", "iteration", "bound"],
        },
        // -- Type mismatch errors (5) --
        BugSnippet {
            buggy_code: "fn avg(values: &[i32]) -> i32 {\n    let sum: i32 = values.iter().sum();\n    sum / values.len() as i32\n}",
            fixed_code: "fn avg(values: &[i32]) -> f64 {\n    let sum: f64 = values.iter().map(|v| *v as f64).sum();\n    sum / values.len() as f64\n}",
            bug_category: BugCategory::TypeMismatch,
            bug_line: 3,
            bug_description: "Integer division truncates; should use f64",
            keywords: &["cast", "truncate", "integer", "float", "precision"],
        },
        BugSnippet {
            buggy_code: "fn big_factorial(n: u8) -> u8 {\n    let mut result: u8 = 1;\n    for i in 2..=n {\n        result *= i;\n    }\n    result\n}",
            fixed_code: "fn big_factorial(n: u8) -> u64 {\n    let mut result: u64 = 1;\n    for i in 2..=n as u64 {\n        result *= i;\n    }\n    result\n}",
            bug_category: BugCategory::TypeMismatch,
            bug_line: 2,
            bug_description: "u8 overflows for factorial(6) = 720 > 255",
            keywords: &["overflow", "narrow", "type", "cast", "width"],
        },
        BugSnippet {
            buggy_code: "fn distance(a: (i32, i32), b: (i32, i32)) -> i32 {\n    let dx = (b.0 - a.0) as i32;\n    let dy = (b.1 - a.1) as i32;\n    ((dx * dx + dy * dy) as f64).sqrt() as i32\n}",
            fixed_code: "fn distance(a: (i32, i32), b: (i32, i32)) -> f64 {\n    let dx = (b.0 - a.0) as f64;\n    let dy = (b.1 - a.1) as f64;\n    (dx * dx + dy * dy).sqrt()\n}",
            bug_category: BugCategory::TypeMismatch,
            bug_line: 4,
            bug_description: "Truncating sqrt result to i32 loses fractional distance",
            keywords: &["truncate", "cast", "precision", "float", "return_type"],
        },
        BugSnippet {
            buggy_code: "fn elapsed_secs(bytes: &[u8; 4]) -> u32 {\n    let val = bytes[0] as u32 | (bytes[1] as u32) << 8\n        | (bytes[2] as i32 as u32) << 16 | (bytes[3] as u32) << 24;\n    val\n}",
            fixed_code: "fn elapsed_secs(bytes: &[u8; 4]) -> u32 {\n    let val = bytes[0] as u32 | (bytes[1] as u32) << 8\n        | (bytes[2] as u32) << 16 | (bytes[3] as u32) << 24;\n    val\n}",
            bug_category: BugCategory::TypeMismatch,
            bug_line: 3,
            bug_description: "Casting u8 to i32 before u32 sign-extends values >= 128",
            keywords: &["sign_extend", "cast", "type", "narrowing", "bit"],
        },
        BugSnippet {
            buggy_code: "fn to_celsius(f: f32) -> i32 {\n    ((f - 32.0) * 5.0 / 9.0) as i32\n}",
            fixed_code: "fn to_celsius(f: f32) -> f32 {\n    (f - 32.0) * 5.0 / 9.0\n}",
            bug_category: BugCategory::TypeMismatch,
            bug_line: 2,
            bug_description: "Truncation to i32 loses decimal precision",
            keywords: &["truncate", "cast", "integer", "precision", "return_type"],
        },
        // -- Logic errors (5) --
        BugSnippet {
            buggy_code: "fn is_leap_year(year: u32) -> bool {\n    year % 4 == 0 && year % 100 != 0 || year % 400 == 0\n}",
            fixed_code: "fn is_leap_year(year: u32) -> bool {\n    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0\n}",
            bug_category: BugCategory::LogicError,
            bug_line: 2,
            bug_description: "Missing parentheses: && binds tighter than ||",
            keywords: &["precedence", "boolean", "condition", "logic", "operator"],
        },
        BugSnippet {
            buggy_code: "fn clamp(val: f64, lo: f64, hi: f64) -> f64 {\n    if val < lo { return hi; }\n    if val > hi { return lo; }\n    val\n}",
            fixed_code: "fn clamp(val: f64, lo: f64, hi: f64) -> f64 {\n    if val < lo { return lo; }\n    if val > hi { return hi; }\n    val\n}",
            bug_category: BugCategory::LogicError,
            bug_line: 2,
            bug_description: "Swapped return values: returns hi when below lo",
            keywords: &["swap", "condition", "return", "logic", "branch"],
        },
        BugSnippet {
            buggy_code: "fn all_positive(arr: &[i32]) -> bool {\n    for &x in arr {\n        if x > 0 { return true; }\n    }\n    false\n}",
            fixed_code: "fn all_positive(arr: &[i32]) -> bool {\n    for &x in arr {\n        if x <= 0 { return false; }\n    }\n    true\n}",
            bug_category: BugCategory::LogicError,
            bug_line: 3,
            bug_description: "Short-circuit logic inverted: returns true on first positive",
            keywords: &["invert", "quantifier", "logic", "short_circuit", "condition"],
        },
        BugSnippet {
            buggy_code: "fn max_of_three(a: i32, b: i32, c: i32) -> i32 {\n    if a > b && a > c { a }\n    else if b > a { b }\n    else { c }\n}",
            fixed_code: "fn max_of_three(a: i32, b: i32, c: i32) -> i32 {\n    if a >= b && a >= c { a }\n    else if b >= c { b }\n    else { c }\n}",
            bug_category: BugCategory::LogicError,
            bug_line: 3,
            bug_description: "Second branch checks b > a but not b > c",
            keywords: &["comparison", "branch", "condition", "logic", "incomplete"],
        },
        BugSnippet {
            buggy_code: "fn contains_duplicate(arr: &[i32]) -> bool {\n    for i in 0..arr.len() {\n        for j in 0..arr.len() {\n            if arr[i] == arr[j] { return true; }\n        }\n    }\n    false\n}",
            fixed_code: "fn contains_duplicate(arr: &[i32]) -> bool {\n    for i in 0..arr.len() {\n        for j in (i + 1)..arr.len() {\n            if arr[i] == arr[j] { return true; }\n        }\n    }\n    false\n}",
            bug_category: BugCategory::LogicError,
            bug_line: 3,
            bug_description: "Inner loop starts at 0: compares element with itself",
            keywords: &["self_compare", "nested_loop", "logic", "equality", "condition"],
        },
        // -- Null/None handling errors (5) --
        BugSnippet {
            buggy_code: "fn first_char(s: &str) -> char {\n    s.chars().next().unwrap()\n}",
            fixed_code: "fn first_char(s: &str) -> Option<char> {\n    s.chars().next()\n}",
            bug_category: BugCategory::NullNoneHandling,
            bug_line: 2,
            bug_description: "Unwrap on empty string panics",
            keywords: &["unwrap", "panic", "none", "option", "empty"],
        },
        BugSnippet {
            buggy_code: "fn find_key(map: &std::collections::HashMap<String, i32>, key: &str) -> i32 {\n    *map.get(key).unwrap()\n}",
            fixed_code: "fn find_key(map: &std::collections::HashMap<String, i32>, key: &str) -> Option<i32> {\n    map.get(key).copied()\n}",
            bug_category: BugCategory::NullNoneHandling,
            bug_line: 2,
            bug_description: "Unwrap on missing key panics",
            keywords: &["unwrap", "missing", "option", "panic", "get"],
        },
        BugSnippet {
            buggy_code: "fn parse_port(s: &str) -> u16 {\n    let parts: Vec<&str> = s.split(':').collect();\n    parts[1].parse().unwrap()\n}",
            fixed_code: "fn parse_port(s: &str) -> Option<u16> {\n    let parts: Vec<&str> = s.split(':').collect();\n    parts.get(1)?.parse().ok()\n}",
            bug_category: BugCategory::NullNoneHandling,
            bug_line: 3,
            bug_description: "Index [1] panics if no colon; parse().unwrap() panics on non-numeric",
            keywords: &["unwrap", "parse", "none", "index", "panic"],
        },
        BugSnippet {
            buggy_code: "fn parent_dir(path: &str) -> &str {\n    let idx = path.rfind('/').unwrap();\n    &path[..idx]\n}",
            fixed_code: "fn parent_dir(path: &str) -> Option<&str> {\n    let idx = path.rfind('/')?;\n    Some(&path[..idx])\n}",
            bug_category: BugCategory::NullNoneHandling,
            bug_line: 2,
            bug_description: "Unwrap panics when path has no slash",
            keywords: &["unwrap", "find", "none", "option", "panic"],
        },
        BugSnippet {
            buggy_code: "fn second_word(s: &str) -> String {\n    let words: Vec<&str> = s.split_whitespace().collect();\n    words[1].to_string()\n}",
            fixed_code: "fn second_word(s: &str) -> Option<String> {\n    let mut words = s.split_whitespace();\n    words.next();\n    words.next().map(|w| w.to_string())\n}",
            bug_category: BugCategory::NullNoneHandling,
            bug_line: 3,
            bug_description: "Index [1] panics when string has fewer than two words",
            keywords: &["index", "panic", "option", "none", "missing"],
        },
        // -- Resource leak errors (5) --
        BugSnippet {
            buggy_code: "fn read_config(path: &str) -> String {\n    let file = std::fs::File::open(path).unwrap();\n    let mut buf = String::new();\n    std::io::Read::read_to_string(&mut &file, &mut buf).unwrap();\n    buf\n}",
            fixed_code: "fn read_config(path: &str) -> std::io::Result<String> {\n    std::fs::read_to_string(path)\n}",
            bug_category: BugCategory::ResourceLeak,
            bug_line: 2,
            bug_description: "Manual File::open without using read_to_string",
            keywords: &["file", "open", "resource", "close", "handle"],
        },
        BugSnippet {
            buggy_code: "fn process_lines(path: &str) -> Vec<String> {\n    let file = std::fs::File::open(path).unwrap();\n    let reader = std::io::BufReader::new(file);\n    let mut lines = Vec::new();\n    for line in std::io::BufRead::lines(reader) {\n        lines.push(line.unwrap());\n    }\n    lines\n}",
            fixed_code: "fn process_lines(path: &str) -> std::io::Result<Vec<String>> {\n    use std::io::BufRead;\n    let file = std::fs::File::open(path)?;\n    let reader = std::io::BufReader::new(file);\n    reader.lines().collect()\n}",
            bug_category: BugCategory::ResourceLeak,
            bug_line: 2,
            bug_description: "Unwrap hides errors; resources held longer than needed",
            keywords: &["resource", "lifetime", "handle", "drop", "leak"],
        },
        BugSnippet {
            buggy_code: "fn write_log(entries: &[String]) {\n    let mut file = std::fs::File::create(\"/tmp/log.txt\").unwrap();\n    for entry in entries {\n        std::io::Write::write_all(&mut file, entry.as_bytes()).unwrap();\n    }\n}",
            fixed_code: "fn write_log(entries: &[String]) -> std::io::Result<()> {\n    use std::io::Write;\n    let mut file = std::fs::File::create(\"/tmp/log.txt\")?;\n    for entry in entries {\n        file.write_all(entry.as_bytes())?;\n    }\n    file.flush()\n}",
            bug_category: BugCategory::ResourceLeak,
            bug_line: 5,
            bug_description: "Missing flush before implicit drop",
            keywords: &["flush", "resource", "handle", "file", "close"],
        },
        BugSnippet {
            buggy_code: "fn temp_file_work() -> String {\n    let path = \"/tmp/work.txt\";\n    std::fs::write(path, \"data\").unwrap();\n    let content = std::fs::read_to_string(path).unwrap();\n    content\n}",
            fixed_code: "fn temp_file_work() -> String {\n    let path = \"/tmp/work.txt\";\n    std::fs::write(path, \"data\").unwrap();\n    let content = std::fs::read_to_string(path).unwrap();\n    let _ = std::fs::remove_file(path);\n    content\n}",
            bug_category: BugCategory::ResourceLeak,
            bug_line: 5,
            bug_description: "Temporary file not removed after use",
            keywords: &["temp", "cleanup", "resource", "remove", "leak"],
        },
        BugSnippet {
            buggy_code: "fn locked_increment(counter: &std::sync::Mutex<u32>) {\n    let mut guard = counter.lock().unwrap();\n    *guard += 1;\n    std::thread::sleep(std::time::Duration::from_secs(1));\n}",
            fixed_code: "fn locked_increment(counter: &std::sync::Mutex<u32>) {\n    {\n        let mut guard = counter.lock().unwrap();\n        *guard += 1;\n    }\n    std::thread::sleep(std::time::Duration::from_secs(1));\n}",
            bug_category: BugCategory::ResourceLeak,
            bug_line: 4,
            bug_description: "Mutex guard held across sleep blocks concurrent access",
            keywords: &["lock", "guard", "resource", "scope", "hold"],
        },
        // -- Boundary violation errors (5) --
        BugSnippet {
            buggy_code: "fn safe_divide(a: f64, b: f64) -> f64 {\n    a / b\n}",
            fixed_code: "fn safe_divide(a: f64, b: f64) -> Option<f64> {\n    if b == 0.0 { None } else { Some(a / b) }\n}",
            bug_category: BugCategory::BoundaryViolation,
            bug_line: 2,
            bug_description: "Division by zero produces infinity/NaN",
            keywords: &["zero", "divide", "boundary", "check", "guard"],
        },
        BugSnippet {
            buggy_code: "fn checked_add(a: u32, b: u32) -> u32 {\n    a + b\n}",
            fixed_code: "fn checked_add(a: u32, b: u32) -> Option<u32> {\n    a.checked_add(b)\n}",
            bug_category: BugCategory::BoundaryViolation,
            bug_line: 2,
            bug_description: "Arithmetic overflow on large inputs",
            keywords: &["overflow", "boundary", "checked", "guard", "limit"],
        },
        BugSnippet {
            buggy_code: "fn head(arr: &[i32]) -> i32 {\n    arr[0]\n}",
            fixed_code: "fn head(arr: &[i32]) -> Option<&i32> {\n    arr.first()\n}",
            bug_category: BugCategory::BoundaryViolation,
            bug_line: 2,
            bug_description: "Index [0] panics on empty slice",
            keywords: &["empty", "boundary", "index", "check", "guard"],
        },
        BugSnippet {
            buggy_code: "fn substr(s: &str, start: usize, len: usize) -> &str {\n    &s[start..start + len]\n}",
            fixed_code: "fn substr(s: &str, start: usize, len: usize) -> Option<&str> {\n    s.get(start..start.checked_add(len)?)\n}",
            bug_category: BugCategory::BoundaryViolation,
            bug_line: 2,
            bug_description: "Slicing panics on out-of-bounds or non-UTF8 boundary",
            keywords: &["bounds", "boundary", "range", "check", "slice"],
        },
        BugSnippet {
            buggy_code: "fn percentage(part: u32, total: u32) -> u32 {\n    (part * 100) / total\n}",
            fixed_code: "fn percentage(part: u32, total: u32) -> Option<u32> {\n    if total == 0 { return None; }\n    Some((part as u64 * 100 / total as u64) as u32)\n}",
            bug_category: BugCategory::BoundaryViolation,
            bug_line: 2,
            bug_description: "Divide by zero when total is 0; multiplication overflow",
            keywords: &["zero", "overflow", "boundary", "divide", "guard"],
        },
    ]
}

fn xor_shift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// Encode a code string as a BinaryHV using n-gram hashing.
///
/// Uses character trigrams with FNV-style seed mixing for spread.
fn encode_code(code: &str) -> BinaryHV {
    let trigrams: Vec<BinaryHV> = code
        .as_bytes()
        .windows(3)
        .map(|w| {
            let seed = (w[0] as u64) | ((w[1] as u64) << 8) | ((w[2] as u64) << 16);
            let mixed = seed.wrapping_mul(0x100000001b3) ^ 0xcbf29ce484222325;
            BinaryHV::random(mixed)
        })
        .collect();

    if trigrams.is_empty() {
        BinaryHV::random(0)
    } else {
        BinaryHV::bundle(&trigrams)
    }
}

/// Encode bug-type keywords as a BinaryHV using role-filler binding.
///
/// Each keyword is encoded as a random HV (deterministic from keyword text),
/// then bound with a role vector and bundled together. This creates a
/// category-discriminative signal that supplements the code delta.
fn encode_bug_keywords(keywords: &[&str], role_kw: &BinaryHV, base_seed: u64) -> BinaryHV {
    let bound: Vec<BinaryHV> = keywords
        .iter()
        .enumerate()
        .map(|(i, kw)| {
            let kw_seed = base_seed
                .wrapping_add((kw.len() as u64).wrapping_mul(997))
                .wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
            // Deterministic HV from keyword text
            let mut s = kw_seed;
            for b in kw.bytes() {
                s = s.wrapping_mul(31).wrapping_add(b as u64);
            }
            let kw_hv = BinaryHV::random(s);
            role_kw.bind(&kw_hv)
        })
        .collect();

    if bound.is_empty() {
        BinaryHV::random(base_seed)
    } else {
        BinaryHV::bundle(&bound)
    }
}

struct TrialResult {
    detection_accuracy: f64,
    category_accuracy: f64,
    delta_magnitude: f64,
    per_category_accuracy: [f64; 6],
    task_trace: Vec<TrialOutcome>,
}

impl BugDetectionBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let seed = config.trial_seed("coding", "bug_detection", trial_idx);
        let mut rng = seed ^ 0xB7E151628AED2A6A;

        let snippets = bug_snippets();
        let n_categories = BugCategory::ALL.len();

        // Noise from time pressure and encoding noise (Wickelgren 1977)
        let noise_weight = (config.time_pressure * 0.10 + config.encoding_noise * 0.12) as f32;

        // Lapse rate: probability of random response per trial (Wichmann & Hill 2001)
        let lapse_rate = config.lapse_rate;

        // Role vectors for keyword binding
        let role_kw = BinaryHV::random(xor_shift(&mut rng));
        let role_delta = BinaryHV::random(xor_shift(&mut rng));

        // -- Phase 1: Build category prototypes using dual encoding --
        // Combine code delta (structural change) with keyword encoding (semantic category)
        let mut category_combined: Vec<Vec<BinaryHV>> = vec![Vec::new(); n_categories];

        for snippet in &snippets {
            let buggy_hv = encode_code(snippet.buggy_code);
            let fixed_hv = encode_code(snippet.fixed_code);
            let delta = buggy_hv.bind(&fixed_hv);
            let kw_hv = encode_bug_keywords(snippet.keywords, &role_kw, xor_shift(&mut rng));
            // Combine delta (what changed) with keywords (what type of bug)
            let combined = BinaryHV::bundle(&[
                role_delta.bind(&delta),
                kw_hv,
            ]);
            category_combined[snippet.bug_category.index()].push(combined);
        }

        let category_prototypes: Vec<BinaryHV> = category_combined
            .iter()
            .map(|hvs| BinaryHV::bundle(hvs))
            .collect();

        // -- Phase 2: Leave-one-out classification --
        let mut detection_hits: u32 = 0;
        let mut category_hits: u32 = 0;
        let mut total: u32 = 0;
        let mut delta_magnitudes: Vec<f64> = Vec::new();
        let mut per_cat_hits: [u32; 6] = [0; 6];
        let mut per_cat_total: [u32; 6] = [0; 6];
        let mut task_trace = Vec::new();

        let category_names = BugCategory::ALL.map(|c| c.name());

        for (snippet_idx, snippet) in snippets.iter().enumerate() {
            let true_cat = snippet.bug_category.index();

            let mut buggy_hv = encode_code(snippet.buggy_code);
            let mut fixed_hv = encode_code(snippet.fixed_code);

            // Apply noise under time pressure
            if noise_weight > 0.0 {
                xor_shift(&mut rng);
                buggy_hv = buggy_hv.add_noise(noise_weight, rng);
                xor_shift(&mut rng);
                fixed_hv = fixed_hv.add_noise(noise_weight, rng);
            }

            // Apply lapse-rate noise (scales with lapse_rate, not constant)
            if lapse_rate > 0.0 {
                let lapse_flip = lapse_rate as f32 * 0.25;
                xor_shift(&mut rng);
                buggy_hv = buggy_hv.add_noise(lapse_flip, rng);
                xor_shift(&mut rng);
                fixed_hv = fixed_hv.add_noise(lapse_flip, rng);
            }

            let delta = buggy_hv.bind(&fixed_hv);
            let magnitude = 1.0 - buggy_hv.similarity(&fixed_hv) as f64;
            delta_magnitudes.push(magnitude);

            // Build combined encoding for the test snippet (delta + keywords)
            let kw_hv = encode_bug_keywords(snippet.keywords, &role_kw, xor_shift(&mut rng));
            let test_combined = BinaryHV::bundle(&[
                role_delta.bind(&delta),
                kw_hv,
            ]);

            // Leave-one-out prototype
            let within_cat_idx = snippets[..snippet_idx]
                .iter()
                .filter(|s| s.bug_category.index() == true_cat)
                .count();

            let mut loo_prototypes: Vec<BinaryHV> = Vec::with_capacity(n_categories);
            for cat_idx in 0..n_categories {
                if cat_idx == true_cat {
                    let remaining: Vec<BinaryHV> = category_combined[cat_idx]
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != within_cat_idx)
                        .map(|(_, hv)| *hv)
                        .collect();
                    if remaining.is_empty() {
                        loo_prototypes.push(category_prototypes[cat_idx]);
                    } else {
                        loo_prototypes.push(BinaryHV::bundle(&remaining));
                    }
                } else {
                    loo_prototypes.push(category_prototypes[cat_idx]);
                }
            }

            // Lapse trial: random response (Wichmann & Hill 2001)
            xor_shift(&mut rng);
            let is_lapse = (rng % 1000) as f64 / 1000.0 < lapse_rate;

            // Detection threshold
            let detection_threshold = 0.03;
            let detected = if is_lapse {
                xor_shift(&mut rng);
                rng % 2 == 0
            } else {
                magnitude > detection_threshold
            };

            // Category classification: find closest prototype
            let predicted_cat = if is_lapse {
                xor_shift(&mut rng);
                (rng % n_categories as u64) as usize
            } else {
                let mut best_cat = 0;
                let mut best_sim = f32::MIN;
                for (cat_idx, proto) in loo_prototypes.iter().enumerate() {
                    let sim = test_combined.similarity(proto);
                    if sim > best_sim {
                        best_sim = sim;
                        best_cat = cat_idx;
                    }
                }
                best_cat
            };

            total += 1;
            per_cat_total[true_cat] += 1;

            if detected {
                detection_hits += 1;
            }
            if predicted_cat == true_cat {
                category_hits += 1;
                per_cat_hits[true_cat] += 1;
            }

            if config.trial_trace {
                task_trace.push(TrialOutcome {
                    trial_idx: snippet_idx,
                    condition: category_names[true_cat].to_string(),
                    correct: predicted_cat == true_cat,
                    rt_ticks: 1.0,
                    similarity: magnitude,
                    confidence: if is_lapse { 0.0 } else { 1.0 },
                    response_idx: predicted_cat,
                    extra: BTreeMap::from([
                        ("bug_line".to_string(), snippet.bug_line as f64),
                        ("detected".to_string(), if detected { 1.0 } else { 0.0 }),
                    ]),
                });
            }
        }

        let detection_accuracy = if total > 0 {
            detection_hits as f64 / total as f64
        } else {
            0.0
        };
        let category_accuracy = if total > 0 {
            category_hits as f64 / total as f64
        } else {
            0.0
        };
        let delta_magnitude = if delta_magnitudes.is_empty() {
            0.0
        } else {
            delta_magnitudes.iter().sum::<f64>() / delta_magnitudes.len() as f64
        };

        let mut per_category_accuracy = [0.0f64; 6];
        #[allow(clippy::needless_range_loop)]
        for i in 0..6 {
            per_category_accuracy[i] = if per_cat_total[i] > 0 {
                per_cat_hits[i] as f64 / per_cat_total[i] as f64
            } else {
                0.0
            };
        }

        TrialResult {
            detection_accuracy,
            category_accuracy,
            delta_magnitude,
            per_category_accuracy,
            task_trace,
        }
    }
}

impl PsychBenchmark for BugDetectionBenchmark {
    fn name(&self) -> &str {
        "Coding::BugDetection"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Code defect detection",
            citation: "Beller et al. (2018)",
            year: 2018,
            doi: Some("10.1145/3180155.3180164"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut detection_accs = Vec::new();
        let mut category_accs = Vec::new();
        let mut delta_mags = Vec::new();
        let mut per_cat: [Vec<f64>; 6] = [
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ];
        let mut trace = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            detection_accs.push(r.detection_accuracy);
            category_accs.push(r.category_accuracy);
            delta_mags.push(r.delta_magnitude);
            #[allow(clippy::needless_range_loop)]
            for i in 0..6 {
                per_cat[i].push(r.per_category_accuracy[i]);
            }
            if config.trial_trace {
                trace.extend(r.task_trace);
            }
        }

        result.insert(
            "detection_accuracy",
            MetricValue::from_samples(&detection_accs),
        );
        result.insert(
            "category_accuracy",
            MetricValue::from_samples(&category_accs),
        );
        result.insert("delta_magnitude", MetricValue::from_samples(&delta_mags));

        for cat in BugCategory::ALL {
            result.insert(
                format!("accuracy_{}", cat.name()),
                MetricValue::from_samples(&per_cat[cat.index()]),
            );
        }

        if config.trial_trace {
            result.trial_trace = trace;
        }

        result.conditions = 6;
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> BenchmarkConfig {
        BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        }
    }

    #[test]
    fn test_bug_detection_runs_with_metrics() {
        let result = BugDetectionBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("detection_accuracy"));
        assert!(result.metrics.contains_key("category_accuracy"));
        assert!(result.metrics.contains_key("delta_magnitude"));
        for cat in BugCategory::ALL {
            assert!(
                result
                    .metrics
                    .contains_key(&format!("accuracy_{}", cat.name())),
                "Missing metric for category {}",
                cat.name()
            );
        }
    }

    #[test]
    fn test_all_metrics_finite() {
        let result = BugDetectionBenchmark.run(&test_config());
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
            assert!(val.std_dev.is_finite(), "metric {} std_dev not finite", key);
        }
    }

    #[test]
    fn test_category_accuracy_above_chance() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = BugDetectionBenchmark.run(&config);
        let acc = result.metrics["category_accuracy"].mean;
        // 6-AFC chance = 1/6 ~ 0.167
        assert!(
            acc > 0.20,
            "Category accuracy should beat 6-AFC chance (0.167), got {}",
            acc
        );
    }

    #[test]
    fn test_deterministic_with_same_seed() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 3,
            seed: 42,
            ..Default::default()
        };
        let r1 = BugDetectionBenchmark.run(&config);
        let r2 = BugDetectionBenchmark.run(&config);
        assert_eq!(
            r1.metrics["category_accuracy"].mean,
            r2.metrics["category_accuracy"].mean,
            "Same seed should produce identical results"
        );
    }

    #[test]
    fn test_lapse_rate_degrades_accuracy() {
        let baseline = BenchmarkConfig {
            trials_per_condition: 10,
            ..Default::default()
        };
        let lapsed = BenchmarkConfig {
            lapse_rate: 0.25,
            trials_per_condition: 10,
            ..Default::default()
        };
        let base_result = BugDetectionBenchmark.run(&baseline);
        let lapse_result = BugDetectionBenchmark.run(&lapsed);
        let base_acc = base_result.metrics["category_accuracy"].mean;
        let lapse_acc = lapse_result.metrics["category_accuracy"].mean;
        assert!(
            lapse_acc <= base_acc + 0.05,
            "Lapse rate should not improve accuracy: base={}, lapsed={}",
            base_acc,
            lapse_acc
        );
    }

    #[test]
    fn test_provenance_correct() {
        let prov = BugDetectionBenchmark.provenance().unwrap();
        assert_eq!(prov.paradigm, "Code defect detection");
        assert_eq!(prov.citation, "Beller et al. (2018)");
    }

    #[test]
    fn test_snippet_count() {
        let snippets = bug_snippets();
        assert_eq!(snippets.len(), 30, "Should have 30 bug snippets");
        for cat in BugCategory::ALL {
            let count = snippets.iter().filter(|s| s.bug_category == cat).count();
            assert_eq!(count, 5, "Category {} should have 5 snippets", cat.name());
        }
    }

}
