// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Golden-reference Rust snippets for logic distillation.

pub fn rust_golden_for(prompt: &str) -> Option<&'static str> {
    match prompt.to_lowercase().as_str() {
        "calculate fibonacci sequence iteratively" => Some(FIBONACCI_ITERATIVE),
        "binary search on a sorted slice" => Some(BINARY_SEARCH),
        "quicksort algorithm" => Some(QUICKSORT),
        "factorial using recursion" => Some(FACTORIAL_RECURSIVE),
        _ => None,
    }
}

pub const RUST_HARVEST_PROMPTS: &[&str] = &[
    "calculate fibonacci sequence iteratively",
    "binary search on a sorted slice",
    "quicksort algorithm",
    "factorial using recursion",
];

const FIBONACCI_ITERATIVE: &str = r#"pub fn fibonacci(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    let mut a = 0;
    let mut b = 1;
    for _ in 0..n {
        let temp = a + b;
        a = b;
        b = temp;
    }
    a
}
"#;

const BINARY_SEARCH: &str = r#"pub fn binary_search(arr: &[i32], target: i32) -> Option<usize> {
    let mut low = 0;
    let mut high = arr.len();

    while low < high {
        let mid = low + (high - low) / 2;
        if arr[mid] == target {
            return Some(mid);
        } else if arr[mid] < target {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    None
}
"#;

const QUICKSORT: &str = r#"pub fn quicksort<T: Ord>(arr: &mut [T]) {
    let len = arr.len();
    if len <= 1 {
        return;
    }
    let pivot_index = partition(arr);
    quicksort(&mut arr[0..pivot_index]);
    quicksort(&mut arr[pivot_index + 1..len]);
}

fn partition<T: Ord>(arr: &mut [T]) -> usize {
    let len = arr.len();
    let pivot_index = len / 2;
    let last_index = len - 1;

    arr.swap(pivot_index, last_index);

    let mut store_index = 0;
    for i in 0..last_index {
        if arr[i] <= arr[last_index] {
            arr.swap(i, store_index);
            store_index += 1;
        }
    }

    arr.swap(store_index, len - 1);
    store_index
}
"#;

const FACTORIAL_RECURSIVE: &str = r#"pub fn factorial(n: u64) -> u64 {
    if n == 0 {
        1
    } else {
        n * factorial(n - 1)
    }
}
"#;
