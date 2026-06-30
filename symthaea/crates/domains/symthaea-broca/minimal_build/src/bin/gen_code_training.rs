// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Generate structurally-derived Rust code training data for Broca.
//!
//! Produces 1000 Rust code snippets across 7 algorithm families and 3 complexity
//! levels. Each snippet is deterministically composed (not template-filled) from
//! the selected family, complexity, and random variation seed.
//!
//! Channels encode:
//! - 0-7: intent one-hot (Answer=1 for code output)
//! - 8: epistemic status (Certain=0.0 for correct code)
//! - 9-11: neutral emotion
//! - 12-14: consciousness
//! - 24-27: CODE channels (syntax_complexity, type_confidence, algorithm_pattern, error_likelihood)
//! - 28-42: epistemic cube (E3, N2, M2, H2 for verified code)
//!
//! Usage:
//!   cargo run --release -p symthaea-broca --bin r#gen-code-training
//!
//! Outputs:
//!   data/train-code-v1.jsonl  (80%)
//!   data/eval-code-v1.jsonl   (20%)

use std::io::Write;
use symthaea_broca::encoder::{
    EPISTEMIC_CUBE_BASE, EPISTEMIC_CUBE_CHANNELS, NUM_CHANNELS, ThoughtChannels,
};

// ============================================================================
// Deterministic RNG (matches gen_epistemic_training.rs)
// ============================================================================

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() % 10000) as f32 / 10000.0
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        if hi <= lo {
            lo
        } else {
            lo + (self.next_u64() as usize % (hi - lo))
        }
    }
    fn pick<'a>(&mut self, items: &'a [&str]) -> &'a str {
        items[self.range(0, items.len())]
    }
}

// ============================================================================
// Algorithm families
// ============================================================================

#[derive(Debug, Clone, Copy)]
enum Family {
    Sorting,
    Searching,
    Iteration,
    DataStructures,
    ErrorHandling,
    StringProcessing,
    Io,
}

const ALL_FAMILIES: [Family; 7] = [
    Family::Sorting,
    Family::Searching,
    Family::Iteration,
    Family::DataStructures,
    Family::ErrorHandling,
    Family::StringProcessing,
    Family::Io,
];

#[derive(Debug, Clone, Copy)]
enum Complexity {
    Simple,  // 0.2
    Medium,  // 0.5
    Complex, // 0.8
}

const ALL_COMPLEXITIES: [Complexity; 3] =
    [Complexity::Simple, Complexity::Medium, Complexity::Complex];

impl Complexity {
    fn value(self) -> f32 {
        match self {
            Complexity::Simple => 0.2,
            Complexity::Medium => 0.5,
            Complexity::Complex => 0.8,
        }
    }
}

// ============================================================================
// Variable name pools
// ============================================================================

const VAR_NAMES: &[&str] = &[
    "x", "v", "n", "val", "item", "acc", "tmp", "buf", "key", "out",
];
const TYPE_NAMES: &[&str] = &["i32", "i64", "u32", "u64", "usize", "f64"];
const GENERIC_BOUNDS: &[&str] = &[
    "Ord",
    "PartialOrd",
    "Clone",
    "Copy + Ord",
    "PartialOrd + Clone",
];

// ============================================================================
// Sorting family
// ============================================================================

fn gen_sorting(rng: &mut Rng, cx: Complexity) -> String {
    match cx {
        Complexity::Simple => {
            let ty = rng.pick(TYPE_NAMES);
            let variant = rng.range(0, 3);
            match variant {
                0 => format!(
                    "fn bubble_sort(arr: &mut [{ty}]) {{\n\
                     \x20   let n = arr.len();\n\
                     \x20   for i in 0..n {{\n\
                     \x20       for j in 0..n - 1 - i {{\n\
                     \x20           if arr[j] > arr[j + 1] {{\n\
                     \x20               arr.swap(j, j + 1);\n\
                     \x20           }}\n\
                     \x20       }}\n\
                     \x20   }}\n\
                     }}"
                ),
                1 => format!(
                    "fn insertion_sort(arr: &mut [{ty}]) {{\n\
                     \x20   for i in 1..arr.len() {{\n\
                     \x20       let mut j = i;\n\
                     \x20       while j > 0 && arr[j - 1] > arr[j] {{\n\
                     \x20           arr.swap(j - 1, j);\n\
                     \x20           j -= 1;\n\
                     \x20       }}\n\
                     \x20   }}\n\
                     }}"
                ),
                _ => format!(
                    "fn selection_sort(arr: &mut [{ty}]) {{\n\
                     \x20   let n = arr.len();\n\
                     \x20   for i in 0..n {{\n\
                     \x20       let min = (i..n).min_by_key(|&k| &arr[k]).unwrap();\n\
                     \x20       arr.swap(i, min);\n\
                     \x20   }}\n\
                     }}"
                ),
            }
        }
        Complexity::Medium => {
            let bound = rng.pick(GENERIC_BOUNDS);
            let variant = rng.range(0, 2);
            match variant {
                0 => format!(
                    "fn quicksort<T: {bound}>(arr: &mut [T]) {{\n\
                     \x20   if arr.len() <= 1 {{ return; }}\n\
                     \x20   let pivot = arr.len() - 1;\n\
                     \x20   let mut i = 0;\n\
                     \x20   for j in 0..pivot {{\n\
                     \x20       if arr[j] <= arr[pivot] {{\n\
                     \x20           arr.swap(i, j);\n\
                     \x20           i += 1;\n\
                     \x20       }}\n\
                     \x20   }}\n\
                     \x20   arr.swap(i, pivot);\n\
                     \x20   quicksort(&mut arr[..i]);\n\
                     \x20   quicksort(&mut arr[i + 1..]);\n\
                     }}"
                ),
                _ => format!(
                    "fn merge_sorted<T: {bound}>(a: &[T], b: &[T]) -> Vec<T> {{\n\
                     \x20   let mut out = Vec::with_capacity(a.len() + b.len());\n\
                     \x20   let (mut i, mut j) = (0, 0);\n\
                     \x20   while i < a.len() && j < b.len() {{\n\
                     \x20       if a[i] <= b[j] {{ out.push(a[i].clone()); i += 1; }}\n\
                     \x20       else {{ out.push(b[j].clone()); j += 1; }}\n\
                     \x20   }}\n\
                     \x20   out.extend_from_slice(&a[i..]);\n\
                     \x20   out.extend_from_slice(&b[j..]);\n\
                     \x20   out\n\
                     }}"
                ),
            }
        }
        Complexity::Complex => {
            let variant = rng.range(0, 2);
            match variant {
                0 => "fn merge_sort<T: Ord + Clone>(arr: &mut [T]) {\n\
                     \x20   if arr.len() <= 1 { return; }\n\
                     \x20   let mid = arr.len() / 2;\n\
                     \x20   let mut left = arr[..mid].to_vec();\n\
                     \x20   let mut right = arr[mid..].to_vec();\n\
                     \x20   merge_sort(&mut left);\n\
                     \x20   merge_sort(&mut right);\n\
                     \x20   let (mut i, mut j, mut k) = (0, 0, 0);\n\
                     \x20   while i < left.len() && j < right.len() {\n\
                     \x20       if left[i] <= right[j] { arr[k] = left[i].clone(); i += 1; }\n\
                     \x20       else { arr[k] = right[j].clone(); j += 1; }\n\
                     \x20       k += 1;\n\
                     \x20   }\n\
                     \x20   while i < left.len() { arr[k] = left[i].clone(); i += 1; k += 1; }\n\
                     \x20   while j < right.len() { arr[k] = right[j].clone(); j += 1; k += 1; }\n\
                     }".to_string(),
                _ => "fn heapsort<T: Ord>(arr: &mut [T]) {\n\
                     \x20   let n = arr.len();\n\
                     \x20   for i in (0..n / 2).rev() { sift_down(arr, i, n); }\n\
                     \x20   for end in (1..n).rev() {\n\
                     \x20       arr.swap(0, end);\n\
                     \x20       sift_down(arr, 0, end);\n\
                     \x20   }\n\
                     }\n\
                     fn sift_down<T: Ord>(arr: &mut [T], mut root: usize, end: usize) {\n\
                     \x20   while 2 * root + 1 < end {\n\
                     \x20       let child = 2 * root + 1;\n\
                     \x20       let swap = if child + 1 < end && arr[child] < arr[child + 1] { child + 1 } else { child };\n\
                     \x20       if arr[root] < arr[swap] { arr.swap(root, swap); root = swap; }\n\
                     \x20       else { break; }\n\
                     \x20   }\n\
                     }".to_string(),
            }
        }
    }
}

// ============================================================================
// Searching family
// ============================================================================

fn gen_searching(rng: &mut Rng, cx: Complexity) -> String {
    match cx {
        Complexity::Simple => {
            let ty = rng.pick(TYPE_NAMES);
            let variant = rng.range(0, 3);
            match variant {
                0 => format!(
                    "fn linear_search(arr: &[{ty}], target: {ty}) -> Option<usize> {{\n\
                     \x20   for (i, &val) in arr.iter().enumerate() {{\n\
                     \x20       if val == target {{ return Some(i); }}\n\
                     \x20   }}\n\
                     \x20   None\n\
                     }}"
                ),
                1 => format!(
                    "fn contains(arr: &[{ty}], target: {ty}) -> bool {{\n\
                     \x20   arr.iter().any(|&x| x == target)\n\
                     }}"
                ),
                _ => format!(
                    "fn find_max(arr: &[{ty}]) -> Option<{ty}> {{\n\
                     \x20   arr.iter().copied().max()\n\
                     }}"
                ),
            }
        }
        Complexity::Medium => {
            let variant = rng.range(0, 2);
            match variant {
                0 => "fn binary_search<T: Ord>(arr: &[T], target: &T) -> Option<usize> {\n\
                     \x20   let (mut lo, mut hi) = (0, arr.len());\n\
                     \x20   while lo < hi {\n\
                     \x20       let mid = lo + (hi - lo) / 2;\n\
                     \x20       match arr[mid].cmp(target) {\n\
                     \x20           std::cmp::Ordering::Equal => return Some(mid),\n\
                     \x20           std::cmp::Ordering::Less => lo = mid + 1,\n\
                     \x20           std::cmp::Ordering::Greater => hi = mid,\n\
                     \x20       }\n\
                     \x20   }\n\
                     \x20   None\n\
                     }"
                .to_string(),
                _ => "fn find_first<T: PartialEq>(arr: &[T], target: &T) -> Option<usize> {\n\
                     \x20   arr.iter().position(|x| x == target)\n\
                     }"
                .to_string(),
            }
        }
        Complexity::Complex => {
            let variant = rng.range(0, 2);
            match variant {
                0 => "fn binary_search_by<T, F>(arr: &[T], mut f: F) -> Option<usize>\n\
                     where\n\
                     \x20   F: FnMut(&T) -> std::cmp::Ordering,\n\
                     {\n\
                     \x20   let (mut lo, mut hi) = (0, arr.len());\n\
                     \x20   while lo < hi {\n\
                     \x20       let mid = lo + (hi - lo) / 2;\n\
                     \x20       match f(&arr[mid]) {\n\
                     \x20           std::cmp::Ordering::Equal => return Some(mid),\n\
                     \x20           std::cmp::Ordering::Less => lo = mid + 1,\n\
                     \x20           std::cmp::Ordering::Greater => hi = mid,\n\
                     \x20       }\n\
                     \x20   }\n\
                     \x20   None\n\
                     }"
                .to_string(),
                _ => "fn find_all<'a, T: PartialEq>(arr: &'a [T], target: &T) -> Vec<&'a T> {\n\
                     \x20   arr.iter().filter(|x| *x == target).collect()\n\
                     }"
                .to_string(),
            }
        }
    }
}

// ============================================================================
// Iteration family
// ============================================================================

fn gen_iteration(rng: &mut Rng, cx: Complexity) -> String {
    let var = rng.pick(VAR_NAMES);
    match cx {
        Complexity::Simple => {
            let ty = rng.pick(TYPE_NAMES);
            let variant = rng.range(0, 4);
            match variant {
                0 => format!(
                    "fn double_all(arr: &[{ty}]) -> Vec<{ty}> {{\n\
                     \x20   arr.iter().map(|{var}| {var} * 2).collect()\n\
                     }}"
                ),
                1 => format!(
                    "fn sum_all(arr: &[{ty}]) -> {ty} {{\n\
                     \x20   arr.iter().copied().fold(0, |acc, {var}| acc + {var})\n\
                     }}"
                ),
                2 => format!(
                    "fn keep_positive(arr: &[{ty}]) -> Vec<{ty}> {{\n\
                     \x20   arr.iter().copied().filter(|&{var}| {var} > 0).collect()\n\
                     }}"
                ),
                _ => format!(
                    "fn print_all(arr: &[{ty}]) {{\n\
                     \x20   arr.iter().for_each(|{var}| println!(\"{{}}\"  , {var}));\n\
                     }}"
                ),
            }
        }
        Complexity::Medium => {
            let variant = rng.range(0, 3);
            match variant {
                0 => format!(
                    "fn filter_map_example(arr: &[&str]) -> Vec<usize> {{\n\
                     \x20   arr.iter()\n\
                     \x20       .filter(|{var}| !{var}.is_empty())\n\
                     \x20       .map(|{var}| {var}.len())\n\
                     \x20       .collect()\n\
                     }}"
                ),
                1 => "fn enumerate_sum(arr: &[f64]) -> f64 {\n\
                     \x20   arr.iter()\n\
                     \x20       .enumerate()\n\
                     \x20       .map(|(i, v)| (i as f64 + 1.0) * v)\n\
                     \x20       .sum()\n\
                     }"
                .to_string(),
                _ => "fn chain_example(a: &[i32], b: &[i32]) -> Vec<i32> {\n\
                     \x20   a.iter().chain(b.iter()).copied().collect()\n\
                     }"
                .to_string(),
            }
        }
        Complexity::Complex => {
            let variant = rng.range(0, 2);
            match variant {
                0 => "fn group_by_len<'a>(words: &[&'a str]) -> std::collections::HashMap<usize, Vec<&'a str>> {\n\
                     \x20   let mut map = std::collections::HashMap::new();\n\
                     \x20   for &w in words {\n\
                     \x20       map.entry(w.len()).or_insert_with(Vec::new).push(w);\n\
                     \x20   }\n\
                     \x20   map\n\
                     }".to_string(),
                _ => "fn sliding_window<T: Clone>(arr: &[T], size: usize) -> Vec<Vec<T>> {\n\
                     \x20   arr.windows(size)\n\
                     \x20       .map(|w| w.to_vec())\n\
                     \x20       .collect()\n\
                     }".to_string(),
            }
        }
    }
}

// ============================================================================
// Data structures family
// ============================================================================

fn gen_data_structures(rng: &mut Rng, cx: Complexity) -> String {
    match cx {
        Complexity::Simple => {
            let ty = rng.pick(TYPE_NAMES);
            let variant = rng.range(0, 3);
            match variant {
                0 => format!(
                    "fn build_vec() -> Vec<{ty}> {{\n\
                     \x20   let mut v = Vec::new();\n\
                     \x20   v.push(1);\n\
                     \x20   v.push(2);\n\
                     \x20   v.push(3);\n\
                     \x20   v\n\
                     }}"
                ),
                1 => format!(
                    "fn last_element(v: &[{ty}]) -> Option<&{ty}> {{\n\
                     \x20   v.last()\n\
                     }}"
                ),
                _ => format!(
                    "fn pop_last(v: &mut Vec<{ty}>) -> Option<{ty}> {{\n\
                     \x20   v.pop()\n\
                     }}"
                ),
            }
        }
        Complexity::Medium => {
            let variant = rng.range(0, 3);
            match variant {
                0 => "fn word_count(words: &[&str]) -> std::collections::HashMap<&str, usize> {\n\
                     \x20   let mut map = std::collections::HashMap::new();\n\
                     \x20   for &w in words {\n\
                     \x20       *map.entry(w).or_insert(0) += 1;\n\
                     \x20   }\n\
                     \x20   map\n\
                     }".to_string(),
                1 => "fn sorted_keys(map: &std::collections::BTreeMap<String, i32>) -> Vec<String> {\n\
                     \x20   map.keys().cloned().collect()\n\
                     }".to_string(),
                _ => "fn rotate_deque(n: usize) -> std::collections::VecDeque<usize> {\n\
                     \x20   let mut dq = std::collections::VecDeque::new();\n\
                     \x20   for i in 0..n { dq.push_back(i); }\n\
                     \x20   dq.rotate_left(1);\n\
                     \x20   dq\n\
                     }".to_string(),
            }
        }
        Complexity::Complex => {
            let variant = rng.range(0, 2);
            match variant {
                0 => "fn invert_map<K, V>(map: &std::collections::HashMap<K, V>)\n\
                     \x20   -> std::collections::HashMap<V, Vec<K>>\n\
                     where\n\
                     \x20   K: Clone + Eq + std::hash::Hash,\n\
                     \x20   V: Clone + Eq + std::hash::Hash,\n\
                     {\n\
                     \x20   let mut out = std::collections::HashMap::new();\n\
                     \x20   for (k, v) in map {\n\
                     \x20       out.entry(v.clone()).or_insert_with(Vec::new).push(k.clone());\n\
                     \x20   }\n\
                     \x20   out\n\
                     }"
                .to_string(),
                _ => "fn lru_insert<K: Eq + std::hash::Hash + Clone, V>(\n\
                     \x20   cache: &mut std::collections::HashMap<K, V>,\n\
                     \x20   order: &mut std::collections::VecDeque<K>,\n\
                     \x20   key: K,\n\
                     \x20   value: V,\n\
                     \x20   cap: usize,\n\
                     ) {\n\
                     \x20   if cache.len() >= cap {\n\
                     \x20       if let Some(old) = order.pop_front() { cache.remove(&old); }\n\
                     \x20   }\n\
                     \x20   order.push_back(key.clone());\n\
                     \x20   cache.insert(key, value);\n\
                     }"
                .to_string(),
            }
        }
    }
}

// ============================================================================
// Error handling family
// ============================================================================

fn gen_error_handling(rng: &mut Rng, cx: Complexity) -> String {
    match cx {
        Complexity::Simple => {
            let variant = rng.range(0, 3);
            match variant {
                0 => "fn safe_div(a: f64, b: f64) -> Option<f64> {\n\
                     \x20   if b == 0.0 { None } else { Some(a / b) }\n\
                     }"
                .to_string(),
                1 => "fn unwrap_or_default(val: Option<i32>) -> i32 {\n\
                     \x20   val.unwrap_or(0)\n\
                     }"
                .to_string(),
                _ => "fn parse_int(s: &str) -> Result<i32, std::num::ParseIntError> {\n\
                     \x20   s.trim().parse()\n\
                     }"
                .to_string(),
            }
        }
        Complexity::Medium => {
            let variant = rng.range(0, 3);
            match variant {
                0 => "fn read_and_parse(s: &str) -> Result<Vec<i32>, String> {\n\
                     \x20   s.split(',')\n\
                     \x20       .map(|p| p.trim().parse::<i32>().map_err(|e| e.to_string()))\n\
                     \x20       .collect()\n\
                     }"
                .to_string(),
                1 => "fn first_even(arr: &[i32]) -> Result<i32, &'static str> {\n\
                     \x20   arr.iter()\n\
                     \x20       .find(|x| *x % 2 == 0)\n\
                     \x20       .copied()\n\
                     \x20       .ok_or(\"no even number found\")\n\
                     }"
                .to_string(),
                _ => "fn try_both(a: Option<i32>, b: Option<i32>) -> Option<i32> {\n\
                     \x20   match (a, b) {\n\
                     \x20       (Some(x), Some(y)) => Some(x + y),\n\
                     \x20       (Some(x), None) | (None, Some(x)) => Some(x),\n\
                     \x20       _ => None,\n\
                     \x20   }\n\
                     }"
                .to_string(),
            }
        }
        Complexity::Complex => {
            let variant = rng.range(0, 2);
            match variant {
                0 => "fn chain_results(input: &str) -> Result<u64, Box<dyn std::error::Error>> {\n\
                     \x20   let trimmed = input.trim();\n\
                     \x20   let without_prefix = trimmed\n\
                     \x20       .strip_prefix(\"0x\")\n\
                     \x20       .ok_or(\"missing 0x prefix\")?;\n\
                     \x20   let value = u64::from_str_radix(without_prefix, 16)?;\n\
                     \x20   Ok(value)\n\
                     }"
                .to_string(),
                _ => "#[derive(Debug)]\n\
                     enum AppError {\n\
                     \x20   NotFound(String),\n\
                     \x20   ParseFailed(std::num::ParseIntError),\n\
                     }\n\
                     impl std::fmt::Display for AppError {\n\
                     \x20   fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {\n\
                     \x20       match self {\n\
                     \x20           Self::NotFound(s) => write!(f, \"not found: {}\", s),\n\
                     \x20           Self::ParseFailed(e) => write!(f, \"parse error: {}\", e),\n\
                     \x20       }\n\
                     \x20   }\n\
                     }\n\
                     impl std::error::Error for AppError {}"
                    .to_string(),
            }
        }
    }
}

// ============================================================================
// String processing family
// ============================================================================

fn gen_string_processing(rng: &mut Rng, cx: Complexity) -> String {
    match cx {
        Complexity::Simple => {
            let variant = rng.range(0, 4);
            match variant {
                0 => "fn greet(name: &str) -> String {\n\
                     \x20   format!(\"Hello, {}!\", name)\n\
                     }"
                .to_string(),
                1 => "fn is_blank(s: &str) -> bool {\n\
                     \x20   s.trim().is_empty()\n\
                     }"
                .to_string(),
                2 => "fn first_word(s: &str) -> &str {\n\
                     \x20   s.split_whitespace().next().unwrap_or(\"\")\n\
                     }"
                .to_string(),
                _ => "fn shout(s: &str) -> String {\n\
                     \x20   s.to_uppercase()\n\
                     }"
                .to_string(),
            }
        }
        Complexity::Medium => {
            let variant = rng.range(0, 3);
            match variant {
                0 => "fn snake_to_camel(s: &str) -> String {\n\
                     \x20   s.split('_')\n\
                     \x20       .enumerate()\n\
                     \x20       .map(|(i, w)| {\n\
                     \x20           if i == 0 { w.to_string() }\n\
                     \x20           else {\n\
                     \x20               let mut c = w.chars();\n\
                     \x20               match c.next() {\n\
                     \x20                   None => String::new(),\n\
                     \x20                   Some(f) => f.to_uppercase().to_string() + c.as_str(),\n\
                     \x20               }\n\
                     \x20           }\n\
                     \x20       })\n\
                     \x20       .collect()\n\
                     }"
                .to_string(),
                1 => "fn truncate(s: &str, max_len: usize) -> String {\n\
                     \x20   if s.len() <= max_len { s.to_string() }\n\
                     \x20   else { format!(\"{}...\", &s[..max_len]) }\n\
                     }"
                .to_string(),
                _ => "fn csv_fields(line: &str) -> Vec<String> {\n\
                     \x20   line.split(',')\n\
                     \x20       .map(|f| f.trim().to_string())\n\
                     \x20       .collect()\n\
                     }"
                .to_string(),
            }
        }
        Complexity::Complex => {
            let variant = rng.range(0, 2);
            match variant {
                0 => "fn replace_all_owned(haystack: &str, needle: &str, replacement: &str) -> String {\n\
                     \x20   let mut result = String::with_capacity(haystack.len());\n\
                     \x20   let mut last = 0;\n\
                     \x20   for (start, _) in haystack.match_indices(needle) {\n\
                     \x20       result.push_str(&haystack[last..start]);\n\
                     \x20       result.push_str(replacement);\n\
                     \x20       last = start + needle.len();\n\
                     \x20   }\n\
                     \x20   result.push_str(&haystack[last..]);\n\
                     \x20   result\n\
                     }".to_string(),
                _ => "fn extract_between<'a>(s: &'a str, open: char, close: char) -> Vec<&'a str> {\n\
                     \x20   let mut results = Vec::new();\n\
                     \x20   let mut start = None;\n\
                     \x20   for (i, c) in s.char_indices() {\n\
                     \x20       if c == open { start = Some(i + c.len_utf8()); }\n\
                     \x20       else if c == close {\n\
                     \x20           if let Some(begin) = start.take() { results.push(&s[begin..i]); }\n\
                     \x20       }\n\
                     \x20   }\n\
                     \x20   results\n\
                     }".to_string(),
            }
        }
    }
}

// ============================================================================
// IO family
// ============================================================================

fn gen_io(rng: &mut Rng, cx: Complexity) -> String {
    match cx {
        Complexity::Simple => {
            let variant = rng.range(0, 3);
            match variant {
                0 => "fn read_file(path: &str) -> std::io::Result<String> {\n\
                     \x20   std::fs::read_to_string(path)\n\
                     }"
                .to_string(),
                1 => "fn write_file(path: &str, data: &str) -> std::io::Result<()> {\n\
                     \x20   std::fs::write(path, data)\n\
                     }"
                .to_string(),
                _ => "fn file_exists(path: &str) -> bool {\n\
                     \x20   std::path::Path::new(path).exists()\n\
                     }"
                .to_string(),
            }
        }
        Complexity::Medium => {
            let variant = rng.range(0, 2);
            match variant {
                0 => "fn read_lines(path: &str) -> std::io::Result<Vec<String>> {\n\
                     \x20   use std::io::BufRead;\n\
                     \x20   let file = std::fs::File::open(path)?;\n\
                     \x20   let reader = std::io::BufReader::new(file);\n\
                     \x20   reader.lines().collect()\n\
                     }"
                .to_string(),
                _ => "fn append_line(path: &str, line: &str) -> std::io::Result<()> {\n\
                     \x20   use std::io::Write;\n\
                     \x20   let mut f = std::fs::OpenOptions::new()\n\
                     \x20       .append(true).create(true).open(path)?;\n\
                     \x20   writeln!(f, \"{}\", line)\n\
                     }"
                .to_string(),
            }
        }
        Complexity::Complex => {
            let variant = rng.range(0, 2);
            match variant {
                0 => "fn copy_filtered<P: AsRef<std::path::Path>>(\n\
                     \x20   src: P,\n\
                     \x20   dst: P,\n\
                     \x20   predicate: impl Fn(&str) -> bool,\n\
                     ) -> std::io::Result<usize> {\n\
                     \x20   use std::io::{BufRead, Write};\n\
                     \x20   let reader = std::io::BufReader::new(std::fs::File::open(src)?);\n\
                     \x20   let mut writer = std::io::BufWriter::new(std::fs::File::create(dst)?);\n\
                     \x20   let mut count = 0;\n\
                     \x20   for line in reader.lines() {\n\
                     \x20       let line = line?;\n\
                     \x20       if predicate(&line) {\n\
                     \x20           writeln!(writer, \"{}\", line)?;\n\
                     \x20           count += 1;\n\
                     \x20       }\n\
                     \x20   }\n\
                     \x20   Ok(count)\n\
                     }".to_string(),
                _ => "fn read_csv_records(path: &str) -> std::io::Result<Vec<Vec<String>>> {\n\
                     \x20   use std::io::BufRead;\n\
                     \x20   let file = std::fs::File::open(path)?;\n\
                     \x20   let reader = std::io::BufReader::new(file);\n\
                     \x20   let records = reader.lines()\n\
                     \x20       .map(|l| {\n\
                     \x20           l.map(|line| line.split(',').map(|f| f.trim().to_string()).collect())\n\
                     \x20       })\n\
                     \x20       .collect::<std::io::Result<Vec<_>>>()?;\n\
                     \x20   Ok(records)\n\
                     }".to_string(),
            }
        }
    }
}

// ============================================================================
// Dispatch
// ============================================================================

fn generate_snippet(rng: &mut Rng, family: Family, cx: Complexity) -> String {
    match family {
        Family::Sorting => gen_sorting(rng, cx),
        Family::Searching => gen_searching(rng, cx),
        Family::Iteration => gen_iteration(rng, cx),
        Family::DataStructures => gen_data_structures(rng, cx),
        Family::ErrorHandling => gen_error_handling(rng, cx),
        Family::StringProcessing => gen_string_processing(rng, cx),
        Family::Io => gen_io(rng, cx),
    }
}

/// Map family to algorithm_pattern channel value (0.0-1.0).
fn family_pattern(family: Family) -> f32 {
    match family {
        Family::Sorting => 0.9,
        Family::Searching => 0.85,
        Family::Iteration => 0.7,
        Family::DataStructures => 0.6,
        Family::ErrorHandling => 0.5,
        Family::StringProcessing => 0.4,
        Family::Io => 0.3,
    }
}

// ============================================================================
// Channel construction
// ============================================================================

fn build_channels(rng: &mut Rng, family: Family, cx: Complexity) -> ThoughtChannels {
    let mut ch = ThoughtChannels::default();

    let complexity = cx.value();

    // Intent: Answer (index 1) for code output
    for i in 0..8 {
        ch.channels[i] = if i == 1 { 1.0 } else { 0.0 };
    }

    // Epistemic status: Certain for correct code (small noise for variety)
    let epistemic = match cx {
        Complexity::Simple => 0.0,
        Complexity::Medium => 0.5 + rng.next_f32() * 0.5,
        Complexity::Complex => 1.0 + rng.next_f32() * 1.0,
    };
    ch.set_epistemic(epistemic);

    // Neutral emotion (code is emotionally flat)
    let valence = (rng.next_f32() - 0.5) * 0.2; // -0.1..0.1
    let arousal = 0.2 + rng.next_f32() * 0.2; // 0.2..0.4
    let dominance = 0.4 + rng.next_f32() * 0.2; // 0.4..0.6
    ch.set_emotion(valence, arousal, dominance);

    // Consciousness: moderate and stable for code generation
    let phi = 0.5 + rng.next_f32() * 0.2; // 0.5..0.7
    let coherence = 0.4 + rng.next_f32() * 0.2; // 0.4..0.6
    let depth = 0.6 + rng.next_f32() * 0.2; // 0.6..0.8
    ch.set_consciousness(phi, coherence, depth);

    // Mood temperature: cool/analytical
    ch.channels[17] = 0.6 + rng.next_f32() * 0.3; // 0.6..0.9

    // Domain familiarity: high for code
    ch.channels[21] = 0.7 + rng.next_f32() * 0.3; // 0.7..1.0

    // Response confidence from complexity (simpler = more confident)
    ch.channels[23] = (1.0 - complexity * 0.5) + rng.next_f32() * 0.2;

    // CODE channels (24-27)
    let syntax_complexity = complexity + (rng.next_f32() - 0.5) * 0.15;
    let type_confidence = match cx {
        Complexity::Simple => 0.9 + rng.next_f32() * 0.1,
        Complexity::Medium => 0.7 + rng.next_f32() * 0.2,
        Complexity::Complex => 0.5 + rng.next_f32() * 0.3,
    };
    let algorithm_pattern = family_pattern(family) + (rng.next_f32() - 0.5) * 0.1;
    let error_likelihood = match cx {
        Complexity::Simple => rng.next_f32() * 0.1,
        Complexity::Medium => 0.05 + rng.next_f32() * 0.15,
        Complexity::Complex => 0.1 + rng.next_f32() * 0.2,
    };
    ch.set_code(
        syntax_complexity,
        type_confidence,
        algorithm_pattern,
        error_likelihood,
    );

    // Epistemic cube: E3 (proven), N2 (network), M2 (persistent), H ~0.5 for verified code
    // Vary slightly per complexity
    let e_tier: u8 = match cx {
        Complexity::Simple => 4,  // reproducible
        Complexity::Medium => 3,  // proven
        Complexity::Complex => 3, // proven (still correct, just complex)
    };
    let n_tier: u8 = 2; // network — code is widely shared knowledge
    let m_tier: u8 = 2; // persistent — algorithms don't change
    let h_value = match cx {
        Complexity::Simple => 0.3 + rng.next_f32() * 0.2,
        Complexity::Medium => 0.4 + rng.next_f32() * 0.2,
        Complexity::Complex => 0.6 + rng.next_f32() * 0.3,
    };
    let quality = match cx {
        Complexity::Simple => 0.8 + rng.next_f32() * 0.2,
        Complexity::Medium => 0.6 + rng.next_f32() * 0.3,
        Complexity::Complex => 0.5 + rng.next_f32() * 0.3,
    };
    ch.set_epistemic_cube(e_tier, n_tier, m_tier, h_value, quality);

    ch
}

// ============================================================================
// Training pair
// ============================================================================

#[derive(serde::Serialize)]
struct TrainingPair {
    channels: Vec<f32>,
    target_text: String,
    #[serde(default)]
    target_ids: Vec<u32>,
    family: String,
    complexity: String,
}

fn generate_pairs(rng: &mut Rng, count: usize) -> Vec<TrainingPair> {
    let mut pairs = Vec::with_capacity(count);

    // Systematic: 7 families x 3 complexities x ~48 variants each = ~1008
    // We iterate families x complexities and generate multiple variants per cell
    let cells = ALL_FAMILIES.len() * ALL_COMPLEXITIES.len(); // 21
    let per_cell = count / cells; // ~47
    let mut remainder = count - per_cell * cells;

    for &family in &ALL_FAMILIES {
        for &cx in &ALL_COMPLEXITIES {
            let n = per_cell
                + if remainder > 0 {
                    remainder -= 1;
                    1
                } else {
                    0
                };
            for _ in 0..n {
                let snippet = generate_snippet(rng, family, cx);
                let channels = build_channels(rng, family, cx);

                pairs.push(TrainingPair {
                    channels: channels.channels.to_vec(),
                    target_text: snippet,
                    target_ids: Vec::new(),
                    family: format!("{:?}", family),
                    complexity: format!("{:?}", cx),
                });

                if pairs.len() >= count {
                    return pairs;
                }
            }
        }
    }

    // Fill any remaining with random picks
    while pairs.len() < count {
        let family = ALL_FAMILIES[rng.range(0, ALL_FAMILIES.len())];
        let cx = ALL_COMPLEXITIES[rng.range(0, ALL_COMPLEXITIES.len())];
        let snippet = generate_snippet(rng, family, cx);
        let channels = build_channels(rng, family, cx);

        pairs.push(TrainingPair {
            channels: channels.channels.to_vec(),
            target_text: snippet,
            target_ids: Vec::new(),
            family: format!("{:?}", family),
            complexity: format!("{:?}", cx),
        });
    }

    pairs
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let mut rng = Rng::new(42);

    let total = 1000;
    let pairs = generate_pairs(&mut rng, total);

    let split = (total as f32 * 0.8) as usize;
    let (train, eval) = pairs.split_at(split);

    // Ensure data directory exists
    let data_dir = std::path::Path::new("data");
    std::fs::create_dir_all(data_dir).expect("create data/");

    // Write training set
    {
        let path = data_dir.join("train-code-v1.jsonl");
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path).expect("create train"));
        for pair in train {
            serde_json::to_writer(&mut f, pair).expect("serialize");
            writeln!(f).expect("newline");
        }
        eprintln!("Wrote {} training pairs to {}", train.len(), path.display());
    }

    // Write eval set
    {
        let path = data_dir.join("eval-code-v1.jsonl");
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path).expect("create eval"));
        for pair in eval {
            serde_json::to_writer(&mut f, pair).expect("serialize");
            writeln!(f).expect("newline");
        }
        eprintln!("Wrote {} eval pairs to {}", eval.len(), path.display());
    }

    // Summary statistics
    let mut family_counts = std::collections::HashMap::new();
    let mut complexity_counts = std::collections::HashMap::new();
    for p in &pairs {
        *family_counts.entry(p.family.as_str()).or_insert(0usize) += 1;
        *complexity_counts
            .entry(p.complexity.as_str())
            .or_insert(0usize) += 1;
    }

    eprintln!(
        "\nGenerated {} training pairs, {} eval pairs",
        train.len(),
        eval.len()
    );
    eprintln!("\nFamily coverage:");
    let mut fam_sorted: Vec<_> = family_counts.iter().collect();
    fam_sorted.sort_by_key(|(k, _)| *k);
    for (k, v) in &fam_sorted {
        eprintln!("  {}: {}", k, v);
    }
    eprintln!("\nComplexity coverage:");
    let mut cx_sorted: Vec<_> = complexity_counts.iter().collect();
    cx_sorted.sort_by_key(|(k, _)| *k);
    for (k, v) in &cx_sorted {
        eprintln!("  {}: {}", k, v);
    }
    eprintln!("\nTotal:  {} pairs ({} channels each)", total, NUM_CHANNELS);
    eprintln!("Code channels at indices 24-27");
    eprintln!(
        "Epistemic cube at indices {}-{}",
        EPISTEMIC_CUBE_BASE,
        EPISTEMIC_CUBE_BASE + EPISTEMIC_CUBE_CHANNELS - 1
    );
}
