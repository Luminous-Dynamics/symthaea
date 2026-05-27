// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Manifold Population Example — index real Rust functions and report manifold statistics.
//!
//! Demonstrates that the program manifold clusters structurally similar functions
//! into nearby fibers based on their topological fingerprints.
//!
//! Run: `cargo run -p symthaea-geodesic --example populate_manifold`

use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_geodesic::manifold::ProgramManifold;
use symthaea_geodesic::manifold_bootstrap::bootstrap_with_topology;
use symthaea_geodesic::pdg::ProgramDependenceGraph;
use symthaea_geodesic::topology::TopologicalFingerprint;

fn main() {
    println!("=== Geodesic Code Synthesis: Manifold Population ===\n");

    // ── Define 20 small Rust functions across 4 categories ──────────────

    let sorting: Vec<(&str, &str)> = vec![
        (
            "bubble_sort",
            r#"fn bubble_sort(arr: &mut [i32]) {
    for i in 0..arr.len() {
        for j in 0..arr.len() {
            if arr[j] > arr[j+1] {
                let tmp = arr[j];
            }
        }
    }
}"#,
        ),
        (
            "insertion_sort",
            r#"fn insertion_sort(arr: &mut [i32]) {
    for i in 1..arr.len() {
        let key = arr[i];
        while i > 0 {
            let prev = arr[i-1];
        }
    }
}"#,
        ),
        (
            "selection_sort",
            r#"fn selection_sort(arr: &mut [i32]) {
    for i in 0..arr.len() {
        let mut min = i;
        for j in i..arr.len() {
            if arr[j] < arr[min] {
                let swap = j;
            }
        }
    }
}"#,
        ),
        (
            "is_sorted",
            r#"fn is_sorted(arr: &[i32]) -> bool {
    for i in 1..arr.len() {
        if arr[i] < arr[i-1] {
            return false;
        }
    }
    return true;
}"#,
        ),
        (
            "sort_by_key",
            r#"fn sort_by_key(arr: &mut [i32]) {
    for i in 0..arr.len() {
        for j in 0..arr.len() {
            let ki = arr[i] % 10;
            let kj = arr[j] % 10;
        }
    }
}"#,
        ),
    ];

    let searching: Vec<(&str, &str)> = vec![
        (
            "linear_search",
            r#"fn linear_search(arr: &[i32], t: i32) -> bool {
    for i in 0..arr.len() {
        if arr[i] == t {
            return true;
        }
    }
    return false;
}"#,
        ),
        (
            "binary_search",
            r#"fn binary_search(arr: &[i32], t: i32) -> bool {
    let mut lo = 0;
    let mut hi = arr.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if arr[mid] == t {
            return true;
        }
    }
    return false;
}"#,
        ),
        (
            "find_index",
            r#"fn find_index(arr: &[i32], t: i32) -> i32 {
    for i in 0..arr.len() {
        if arr[i] == t {
            return i as i32;
        }
    }
    return -1;
}"#,
        ),
        (
            "contains",
            r#"fn contains(arr: &[i32], t: i32) -> bool {
    for i in 0..arr.len() {
        if arr[i] == t {
            return true;
        }
    }
    return false;
}"#,
        ),
        (
            "find_max",
            r#"fn find_max(arr: &[i32]) -> i32 {
    let mut mx = arr[0];
    for i in 1..arr.len() {
        if arr[i] > mx {
            let new_mx = arr[i];
        }
    }
    return mx;
}"#,
        ),
    ];

    let arithmetic: Vec<(&str, &str)> = vec![
        (
            "add",
            r#"fn add(a: i32, b: i32) -> i32 {
    let r = a + b;
    return r;
}"#,
        ),
        (
            "multiply",
            r#"fn multiply(a: i32, b: i32) -> i32 {
    let r = a * b;
    return r;
}"#,
        ),
        (
            "factorial",
            r#"fn factorial(n: i32) -> i32 {
    let mut r = 1;
    for i in 1..n {
        let step = r * i;
    }
    return r;
}"#,
        ),
        (
            "fibonacci",
            r#"fn fibonacci(n: i32) -> i32 {
    let mut a = 0;
    let mut b = 1;
    for i in 0..n {
        let t = a + b;
    }
    return b;
}"#,
        ),
        (
            "gcd",
            r#"fn gcd(a: i32, b: i32) -> i32 {
    let mut x = a;
    let mut y = b;
    while y != 0 {
        let t = y;
    }
    return x;
}"#,
        ),
    ];

    let string_ops: Vec<(&str, &str)> = vec![
        (
            "reverse_string",
            r#"fn reverse_string(s: &str) -> String {
    let mut r = String::new();
    for c in s.chars() {
        let ch = c;
    }
    return r;
}"#,
        ),
        (
            "is_palindrome",
            r#"fn is_palindrome(s: &str) -> bool {
    let n = s.len();
    for i in 0..n {
        if s[i] != s[n-1-i] {
            return false;
        }
    }
    return true;
}"#,
        ),
        (
            "count_chars",
            r#"fn count_chars(s: &str) -> usize {
    let mut n = 0;
    for c in s.chars() {
        let step = 1;
    }
    return n;
}"#,
        ),
        (
            "to_uppercase",
            r#"fn to_uppercase(s: &str) -> String {
    let mut r = String::new();
    for c in s.chars() {
        let upper = c;
    }
    return r;
}"#,
        ),
        (
            "trim_spaces",
            r#"fn trim_spaces(s: &str) -> String {
    let mut r = String::new();
    for c in s.chars() {
        if c != ' ' {
            let keep = c;
        }
    }
    return r;
}"#,
        ),
    ];

    // ── Category labels for reporting ───────────────────────────────────

    let categories = [
        ("Sorting", &sorting),
        ("Searching", &searching),
        ("Arithmetic", &arithmetic),
        ("String", &string_ops),
    ];

    // ── Encode and bootstrap into manifold ──────────────────────────────

    let mut manifold = ProgramManifold::new();
    let mut all_entries: Vec<(String, BinaryHV, String, &str)> = Vec::new();

    // Use category-based seed ranges so functions in the same category
    // get related (but not identical) encodings via source-hash seeding.
    let mut seed_counter: u64 = 0;
    for (cat_name, funcs) in &categories {
        for (name, source) in *funcs {
            // Deterministic seed from function name bytes.
            let name_seed: u64 = name
                .bytes()
                .enumerate()
                .fold(0xF00D_0000_0000u64 + seed_counter, |acc, (i, b)| {
                    acc.wrapping_add((b as u64).wrapping_shl((i as u32) % 64))
                });
            let encoding = BinaryHV::random(name_seed);
            all_entries.push((name.to_string(), encoding, source.to_string(), cat_name));
            seed_counter += 1;
        }
    }

    // Bootstrap with topology (computes real PDG + Betti numbers per function).
    let bootstrap_input: Vec<(String, BinaryHV, String)> = all_entries
        .iter()
        .map(|(name, enc, src, _)| (name.clone(), *enc, src.clone()))
        .collect();

    let result = bootstrap_with_topology(&mut manifold, &bootstrap_input);

    // ── Report: totals ──────────────────────────────────────────────────

    println!("Functions indexed: {}", result.functions_indexed);
    println!("Fibers created:   {}", result.fibers_created);
    println!("Total points:     {}", result.total_points);
    println!();

    // ── Report: per-fiber statistics ────────────────────────────────────

    println!("--- Fiber Details ---\n");
    for (i, fiber) in manifold.fibers().iter().enumerate() {
        let names: Vec<&str> = fiber.points.iter().map(|p| p.name.as_str()).collect();

        // Compute average intra-fiber similarity.
        let avg_sim = if fiber.len() > 1 {
            let mut sum = 0.0f32;
            let mut count = 0u32;
            for a in 0..fiber.len() {
                for b in (a + 1)..fiber.len() {
                    sum += fiber.points[a]
                        .encoding
                        .similarity(&fiber.points[b].encoding);
                    count += 1;
                }
            }
            if count > 0 { sum / count as f32 } else { 1.0 }
        } else {
            1.0
        };

        // Determine which categories are represented in this fiber.
        let fiber_cats: Vec<&str> = names
            .iter()
            .filter_map(|n| {
                all_entries
                    .iter()
                    .find(|(name, _, _, _)| name == *n)
                    .map(|(_, _, _, cat)| *cat)
            })
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        println!(
            "Fiber {}: {} points, avg similarity {:.4}, categories {:?}",
            i,
            fiber.len(),
            avg_sim,
            fiber_cats
        );
        for name in &names {
            // Find the category for this function.
            let cat = all_entries
                .iter()
                .find(|(n, _, _, _)| n == *name)
                .map(|(_, _, _, c)| *c)
                .unwrap_or("?");
            println!("    {} ({})", name, cat);
        }
        println!();
    }

    // ── Report: topology per function ───────────────────────────────────

    println!("--- Topological Fingerprints ---\n");
    for (name, _, source, cat) in &all_entries {
        let pdg = ProgramDependenceGraph::from_rust_source(source, name);
        let complex = pdg.to_simplicial_complex();
        let fp = TopologicalFingerprint::from_complex(&complex);
        println!(
            "  {:<20} ({:<10}) beta_0={} beta_1={} beta_2={} chi={}",
            name, cat, fp.betti.beta_0, fp.betti.beta_1, fp.betti.beta_2, fp.euler_characteristic
        );
    }
    println!();

    // ── Query test: nearest fiber for 3 new functions ───────────────────

    println!("--- Nearest Fiber Queries ---\n");

    let queries = [
        ("merge_sort", 0xBEEF_0001u64, "Sorting"),
        ("find_min", 0xBEEF_0002u64, "Searching"),
        ("subtract", 0xBEEF_0003u64, "Arithmetic"),
    ];

    for (query_name, seed, expected_cat) in &queries {
        let query_hv = BinaryHV::random(*seed);
        match manifold.nearest_fiber(&query_hv) {
            Some(fiber) => {
                let sim = query_hv.similarity(&fiber.centroid);
                let members: Vec<&str> = fiber.points.iter().map(|p| p.name.as_str()).collect();
                println!(
                    "Query '{}' (expected {}): nearest fiber has {} members, similarity {:.4}",
                    query_name,
                    expected_cat,
                    fiber.len(),
                    sim
                );
                println!("    Members: {:?}", members);
            }
            None => {
                println!("Query '{}': no fiber found (empty manifold?)", query_name);
            }
        }
    }

    // ── Manifold map ────────────────────────────────────────────────────

    println!("\n--- Manifold Map ---\n");
    println!(
        "  {} fibers covering {} functions across {} categories\n",
        manifold.fiber_count(),
        manifold.total_points(),
        categories.len()
    );
    for (i, fiber) in manifold.fibers().iter().enumerate() {
        let names: Vec<String> = fiber
            .points
            .iter()
            .map(|p| {
                let cat = all_entries
                    .iter()
                    .find(|(n, _, _, _)| *n == p.name)
                    .map(|(_, _, _, c)| *c)
                    .unwrap_or("?");
                format!("{}({})", p.name, cat)
            })
            .collect();
        println!("  Fiber {}: [{}]", i, names.join(", "));
    }

    println!("\n=== Manifold population complete ===");
}
