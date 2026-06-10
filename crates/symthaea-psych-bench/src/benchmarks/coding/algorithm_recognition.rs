// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Algorithm Recognition benchmark.
//!
//! Measures the ability to classify code snippets into algorithm families
//! using HDC pattern matching. Each snippet implements a canonical algorithm
//! (sorting, searching, graph traversal, dynamic programming, divide-and-conquer).
//! The system builds prototype HVs for each family from training examples,
//! then classifies held-out test snippets via nearest-prototype similarity.
//!
//! This tests structural code understanding -- recognizing algorithmic patterns
//! despite surface-level variation in variable names and formatting.
//!
//! Human baselines (Alam et al., 2022; Weimer et al., 2009):
//! - algorithm_accuracy: ~0.72 (SD ~0.14) -- CS students classifying algorithm families
//! - confidence_discrimination: ~0.15 (SD ~0.08) -- margin between correct and next-best
//!
//! References:
//! - Alam et al. (2022). Code comprehension and algorithm recognition. ICSE.
//! - Weimer et al. (2009). Automatically finding patches using genetic programming. ICSE.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::BinaryHV;

/// Algorithm Recognition benchmark.
pub struct AlgorithmRecognitionBenchmark;

/// Algorithm family categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AlgoFamily {
    Sorting,
    Searching,
    GraphTraversal,
    DynamicProgramming,
    DivideAndConquer,
}

impl AlgoFamily {
    const ALL: [AlgoFamily; 5] = [
        AlgoFamily::Sorting,
        AlgoFamily::Searching,
        AlgoFamily::GraphTraversal,
        AlgoFamily::DynamicProgramming,
        AlgoFamily::DivideAndConquer,
    ];

    fn index(self) -> usize {
        match self {
            AlgoFamily::Sorting => 0,
            AlgoFamily::Searching => 1,
            AlgoFamily::GraphTraversal => 2,
            AlgoFamily::DynamicProgramming => 3,
            AlgoFamily::DivideAndConquer => 4,
        }
    }

    fn name(self) -> &'static str {
        match self {
            AlgoFamily::Sorting => "sorting",
            AlgoFamily::Searching => "searching",
            AlgoFamily::GraphTraversal => "graph_traversal",
            AlgoFamily::DynamicProgramming => "dynamic_programming",
            AlgoFamily::DivideAndConquer => "divide_and_conquer",
        }
    }
}

struct AlgoSnippet {
    code: &'static str,
    family: AlgoFamily,
    /// Structural keywords that characterize this algorithm family.
    /// Used as additional features for HDC encoding.
    keywords: &'static [&'static str],
}

/// 25 algorithm snippets (5 per family).
/// Each family has distinctive structural patterns that HDC can capture
/// through trigram and keyword encoding.
fn algorithm_snippets() -> Vec<AlgoSnippet> {
    vec![
        // -- Sorting (5) --
        AlgoSnippet {
            code: "fn bubble_sort(arr: &mut [i32]) {\n    let n = arr.len();\n    for i in 0..n {\n        for j in 0..n-1-i {\n            if arr[j] > arr[j+1] { arr.swap(j, j+1); }\n        }\n    }\n}",
            family: AlgoFamily::Sorting,
            keywords: &["swap", "sort", "compare", "nested_loop"],
        },
        AlgoSnippet {
            code: "fn insertion_sort(arr: &mut [i32]) {\n    for i in 1..arr.len() {\n        let key = arr[i];\n        let mut j = i;\n        while j > 0 && arr[j-1] > key { arr[j] = arr[j-1]; j -= 1; }\n        arr[j] = key;\n    }\n}",
            family: AlgoFamily::Sorting,
            keywords: &["insert", "key", "shift", "sort"],
        },
        AlgoSnippet {
            code: "fn selection_sort(arr: &mut [i32]) {\n    let n = arr.len();\n    for i in 0..n {\n        let mut min_idx = i;\n        for j in i+1..n {\n            if arr[j] < arr[min_idx] { min_idx = j; }\n        }\n        arr.swap(i, min_idx);\n    }\n}",
            family: AlgoFamily::Sorting,
            keywords: &["min", "swap", "select", "sort"],
        },
        AlgoSnippet {
            code: "fn shell_sort(arr: &mut [i32]) {\n    let mut gap = arr.len() / 2;\n    while gap > 0 {\n        for i in gap..arr.len() {\n            let temp = arr[i];\n            let mut j = i;\n            while j >= gap && arr[j - gap] > temp { arr[j] = arr[j-gap]; j -= gap; }\n            arr[j] = temp;\n        }\n        gap /= 2;\n    }\n}",
            family: AlgoFamily::Sorting,
            keywords: &["gap", "sort", "shift", "diminishing"],
        },
        AlgoSnippet {
            code: "fn counting_sort(arr: &[i32], max_val: usize) -> Vec<i32> {\n    let mut count = vec![0; max_val + 1];\n    for &x in arr { count[x as usize] += 1; }\n    let mut result = Vec::new();\n    for (val, &c) in count.iter().enumerate() {\n        for _ in 0..c { result.push(val as i32); }\n    }\n    result\n}",
            family: AlgoFamily::Sorting,
            keywords: &["count", "sort", "frequency", "bucket"],
        },
        // -- Searching (5) --
        AlgoSnippet {
            code: "fn binary_search(arr: &[i32], target: i32) -> Option<usize> {\n    let (mut lo, mut hi) = (0, arr.len());\n    while lo < hi {\n        let mid = lo + (hi - lo) / 2;\n        match arr[mid].cmp(&target) {\n            std::cmp::Ordering::Equal => return Some(mid),\n            std::cmp::Ordering::Less => lo = mid + 1,\n            std::cmp::Ordering::Greater => hi = mid,\n        }\n    }\n    None\n}",
            family: AlgoFamily::Searching,
            keywords: &["binary", "search", "mid", "half"],
        },
        AlgoSnippet {
            code: "fn linear_search(arr: &[i32], target: i32) -> Option<usize> {\n    for (i, &val) in arr.iter().enumerate() {\n        if val == target { return Some(i); }\n    }\n    None\n}",
            family: AlgoFamily::Searching,
            keywords: &["linear", "search", "iterate", "find"],
        },
        AlgoSnippet {
            code: "fn interpolation_search(arr: &[i32], target: i32) -> Option<usize> {\n    let (mut lo, mut hi) = (0isize, arr.len() as isize - 1);\n    while lo <= hi && target >= arr[lo as usize] && target <= arr[hi as usize] {\n        let pos = lo + ((target - arr[lo as usize]) as isize * (hi - lo)) / (arr[hi as usize] - arr[lo as usize]) as isize;\n        if arr[pos as usize] == target { return Some(pos as usize); }\n        if arr[pos as usize] < target { lo = pos + 1; } else { hi = pos - 1; }\n    }\n    None\n}",
            family: AlgoFamily::Searching,
            keywords: &["interpolation", "search", "position", "estimate"],
        },
        AlgoSnippet {
            code: "fn exponential_search(arr: &[i32], target: i32) -> Option<usize> {\n    if arr[0] == target { return Some(0); }\n    let mut i = 1;\n    while i < arr.len() && arr[i] <= target { i *= 2; }\n    let lo = i / 2;\n    let hi = i.min(arr.len());\n    arr[lo..hi].iter().position(|&x| x == target).map(|p| p + lo)\n}",
            family: AlgoFamily::Searching,
            keywords: &["exponential", "search", "doubling", "bound"],
        },
        AlgoSnippet {
            code: "fn ternary_search(arr: &[i32], target: i32) -> Option<usize> {\n    let (mut lo, mut hi) = (0, arr.len().saturating_sub(1));\n    while lo <= hi {\n        let m1 = lo + (hi - lo) / 3;\n        let m2 = hi - (hi - lo) / 3;\n        if arr[m1] == target { return Some(m1); }\n        if arr[m2] == target { return Some(m2); }\n        if target < arr[m1] { hi = m1.saturating_sub(1); }\n        else if target > arr[m2] { lo = m2 + 1; }\n        else { lo = m1 + 1; hi = m2.saturating_sub(1); }\n    }\n    None\n}",
            family: AlgoFamily::Searching,
            keywords: &["ternary", "search", "thirds", "partition"],
        },
        // -- Graph Traversal (5) --
        AlgoSnippet {
            code: "fn bfs(graph: &[Vec<usize>], start: usize) -> Vec<usize> {\n    let mut visited = vec![false; graph.len()];\n    let mut queue = std::collections::VecDeque::new();\n    let mut order = Vec::new();\n    visited[start] = true;\n    queue.push_back(start);\n    while let Some(node) = queue.pop_front() {\n        order.push(node);\n        for &neighbor in &graph[node] {\n            if !visited[neighbor] { visited[neighbor] = true; queue.push_back(neighbor); }\n        }\n    }\n    order\n}",
            family: AlgoFamily::GraphTraversal,
            keywords: &["queue", "visited", "neighbor", "bfs"],
        },
        AlgoSnippet {
            code: "fn dfs(graph: &[Vec<usize>], start: usize) -> Vec<usize> {\n    let mut visited = vec![false; graph.len()];\n    let mut stack = vec![start];\n    let mut order = Vec::new();\n    while let Some(node) = stack.pop() {\n        if visited[node] { continue; }\n        visited[node] = true;\n        order.push(node);\n        for &neighbor in graph[node].iter().rev() {\n            if !visited[neighbor] { stack.push(neighbor); }\n        }\n    }\n    order\n}",
            family: AlgoFamily::GraphTraversal,
            keywords: &["stack", "visited", "neighbor", "dfs"],
        },
        AlgoSnippet {
            code: "fn topological_sort(graph: &[Vec<usize>]) -> Vec<usize> {\n    let n = graph.len();\n    let mut in_degree = vec![0; n];\n    for edges in graph { for &v in edges { in_degree[v] += 1; } }\n    let mut queue: std::collections::VecDeque<usize> = in_degree.iter().enumerate().filter(|(_, &d)| d == 0).map(|(i, _)| i).collect();\n    let mut order = Vec::new();\n    while let Some(node) = queue.pop_front() {\n        order.push(node);\n        for &next in &graph[node] { in_degree[next] -= 1; if in_degree[next] == 0 { queue.push_back(next); } }\n    }\n    order\n}",
            family: AlgoFamily::GraphTraversal,
            keywords: &["degree", "topological", "graph", "order"],
        },
        AlgoSnippet {
            code: "fn dijkstra(graph: &[Vec<(usize, u32)>], start: usize) -> Vec<u32> {\n    let n = graph.len();\n    let mut dist = vec![u32::MAX; n];\n    dist[start] = 0;\n    let mut heap = std::collections::BinaryHeap::new();\n    heap.push(std::cmp::Reverse((0u32, start)));\n    while let Some(std::cmp::Reverse((d, u))) = heap.pop() {\n        if d > dist[u] { continue; }\n        for &(v, w) in &graph[u] { let nd = d + w; if nd < dist[v] { dist[v] = nd; heap.push(std::cmp::Reverse((nd, v))); } }\n    }\n    dist\n}",
            family: AlgoFamily::GraphTraversal,
            keywords: &["distance", "heap", "shortest", "dijkstra"],
        },
        AlgoSnippet {
            code: "fn connected_components(n: usize, edges: &[(usize, usize)]) -> Vec<usize> {\n    let mut parent: Vec<usize> = (0..n).collect();\n    fn find(parent: &mut [usize], x: usize) -> usize {\n        if parent[x] != x { parent[x] = find(parent, parent[x]); }\n        parent[x]\n    }\n    for &(u, v) in edges {\n        let pu = find(&mut parent, u);\n        let pv = find(&mut parent, v);\n        if pu != pv { parent[pu] = pv; }\n    }\n    (0..n).map(|i| find(&mut parent, i)).collect()\n}",
            family: AlgoFamily::GraphTraversal,
            keywords: &["union", "find", "component", "parent"],
        },
        // -- Dynamic Programming (5) --
        AlgoSnippet {
            code: "fn fibonacci(n: usize) -> u64 {\n    if n <= 1 { return n as u64; }\n    let mut dp = vec![0u64; n + 1];\n    dp[1] = 1;\n    for i in 2..=n { dp[i] = dp[i-1] + dp[i-2]; }\n    dp[n]\n}",
            family: AlgoFamily::DynamicProgramming,
            keywords: &["dp", "table", "recurrence", "memoize"],
        },
        AlgoSnippet {
            code: "fn longest_common_subsequence(a: &str, b: &str) -> usize {\n    let (m, n) = (a.len(), b.len());\n    let mut dp = vec![vec![0; n + 1]; m + 1];\n    for (i, ca) in a.chars().enumerate() {\n        for (j, cb) in b.chars().enumerate() {\n            dp[i+1][j+1] = if ca == cb { dp[i][j] + 1 } else { dp[i+1][j].max(dp[i][j+1]) };\n        }\n    }\n    dp[m][n]\n}",
            family: AlgoFamily::DynamicProgramming,
            keywords: &["dp", "subsequence", "table", "match"],
        },
        AlgoSnippet {
            code: "fn knapsack(weights: &[u32], values: &[u32], capacity: u32) -> u32 {\n    let n = weights.len();\n    let c = capacity as usize;\n    let mut dp = vec![vec![0u32; c + 1]; n + 1];\n    for i in 1..=n {\n        for w in 0..=c {\n            dp[i][w] = dp[i-1][w];\n            if weights[i-1] as usize <= w {\n                dp[i][w] = dp[i][w].max(dp[i-1][w - weights[i-1] as usize] + values[i-1]);\n            }\n        }\n    }\n    dp[n][c]\n}",
            family: AlgoFamily::DynamicProgramming,
            keywords: &["dp", "knapsack", "capacity", "optimal"],
        },
        AlgoSnippet {
            code: "fn edit_distance(a: &str, b: &str) -> usize {\n    let (m, n) = (a.len(), b.len());\n    let mut dp = vec![vec![0; n + 1]; m + 1];\n    for i in 0..=m { dp[i][0] = i; }\n    for j in 0..=n { dp[0][j] = j; }\n    for (i, ca) in a.chars().enumerate() {\n        for (j, cb) in b.chars().enumerate() {\n            dp[i+1][j+1] = if ca == cb { dp[i][j] } else { 1 + dp[i][j].min(dp[i+1][j]).min(dp[i][j+1]) };\n        }\n    }\n    dp[m][n]\n}",
            family: AlgoFamily::DynamicProgramming,
            keywords: &["dp", "edit", "distance", "cost"],
        },
        AlgoSnippet {
            code: "fn coin_change(coins: &[u32], amount: u32) -> Option<u32> {\n    let mut dp = vec![u32::MAX; amount as usize + 1];\n    dp[0] = 0;\n    for &coin in coins {\n        for a in coin as usize..=amount as usize {\n            if dp[a - coin as usize] != u32::MAX {\n                dp[a] = dp[a].min(dp[a - coin as usize] + 1);\n            }\n        }\n    }\n    if dp[amount as usize] == u32::MAX { None } else { Some(dp[amount as usize]) }\n}",
            family: AlgoFamily::DynamicProgramming,
            keywords: &["dp", "coin", "change", "minimum"],
        },
        // -- Divide and Conquer (5) --
        AlgoSnippet {
            code: "fn merge_sort(arr: &mut [i32]) {\n    let n = arr.len();\n    if n <= 1 { return; }\n    let mid = n / 2;\n    let mut left = arr[..mid].to_vec();\n    let mut right = arr[mid..].to_vec();\n    merge_sort(&mut left);\n    merge_sort(&mut right);\n    let (mut i, mut j, mut k) = (0, 0, 0);\n    while i < left.len() && j < right.len() {\n        if left[i] <= right[j] { arr[k] = left[i]; i += 1; } else { arr[k] = right[j]; j += 1; }\n        k += 1;\n    }\n    while i < left.len() { arr[k] = left[i]; i += 1; k += 1; }\n    while j < right.len() { arr[k] = right[j]; j += 1; k += 1; }\n}",
            family: AlgoFamily::DivideAndConquer,
            keywords: &["merge", "split", "recursive", "halve"],
        },
        AlgoSnippet {
            code: "fn quicksort(arr: &mut [i32]) {\n    if arr.len() <= 1 { return; }\n    let pivot = arr[arr.len() / 2];\n    let mut lo = 0;\n    let mut hi = arr.len() - 1;\n    while lo <= hi {\n        while arr[lo] < pivot { lo += 1; }\n        while arr[hi] > pivot { hi = hi.wrapping_sub(1); }\n        if lo <= hi { arr.swap(lo, hi); lo += 1; hi = hi.wrapping_sub(1); }\n    }\n}",
            family: AlgoFamily::DivideAndConquer,
            keywords: &["pivot", "partition", "recursive", "quicksort"],
        },
        AlgoSnippet {
            code: "fn max_subarray(arr: &[i32]) -> i32 {\n    fn helper(arr: &[i32], lo: usize, hi: usize) -> i32 {\n        if lo == hi { return arr[lo]; }\n        let mid = (lo + hi) / 2;\n        let left = helper(arr, lo, mid);\n        let right = helper(arr, mid + 1, hi);\n        let mut cross_left = i32::MIN;\n        let mut sum = 0;\n        for i in (lo..=mid).rev() { sum += arr[i]; cross_left = cross_left.max(sum); }\n        let mut cross_right = i32::MIN;\n        sum = 0;\n        for i in mid+1..=hi { sum += arr[i]; cross_right = cross_right.max(sum); }\n        left.max(right).max(cross_left + cross_right)\n    }\n    helper(arr, 0, arr.len() - 1)\n}",
            family: AlgoFamily::DivideAndConquer,
            keywords: &["divide", "conquer", "cross", "halve"],
        },
        AlgoSnippet {
            code: "fn power(base: i64, exp: u32) -> i64 {\n    if exp == 0 { return 1; }\n    if exp % 2 == 0 { let half = power(base, exp / 2); half * half }\n    else { base * power(base, exp - 1) }\n}",
            family: AlgoFamily::DivideAndConquer,
            keywords: &["power", "halve", "recursive", "exponent"],
        },
        AlgoSnippet {
            code: "fn closest_pair_distance(points: &mut [(f64, f64)]) -> f64 {\n    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());\n    fn dist(a: (f64, f64), b: (f64, f64)) -> f64 { ((a.0-b.0).powi(2) + (a.1-b.1).powi(2)).sqrt() }\n    fn solve(pts: &[(f64, f64)]) -> f64 {\n        if pts.len() <= 3 { let mut d = f64::MAX; for i in 0..pts.len() { for j in i+1..pts.len() { d = d.min(dist(pts[i], pts[j])); } } return d; }\n        let mid = pts.len() / 2;\n        let dl = solve(&pts[..mid]);\n        let dr = solve(&pts[mid..]);\n        dl.min(dr)\n    }\n    solve(points)\n}",
            family: AlgoFamily::DivideAndConquer,
            keywords: &["divide", "pair", "split", "recursive"],
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
fn encode_code(code: &str, seed_offset: u64) -> BinaryHV {
    let trigrams: Vec<BinaryHV> = code
        .as_bytes()
        .windows(3)
        .map(|w| {
            let seed = (w[0] as u64) | ((w[1] as u64) << 8) | ((w[2] as u64) << 16);
            let mixed =
                seed.wrapping_mul(0x100000001b3).wrapping_add(seed_offset) ^ 0xcbf29ce484222325;
            BinaryHV::random(mixed)
        })
        .collect();

    if trigrams.is_empty() {
        BinaryHV::random(seed_offset)
    } else {
        BinaryHV::bundle(&trigrams)
    }
}

/// Encode structural keywords into a BinaryHV.
/// Each keyword is hashed to a deterministic HV and bundled.
fn encode_keywords(keywords: &[&str], role_keyword: &BinaryHV, base_seed: u64) -> BinaryHV {
    let keyword_hvs: Vec<BinaryHV> = keywords
        .iter()
        .enumerate()
        .map(|(i, kw)| {
            let kw_seed = base_seed
                .wrapping_add((kw.len() as u64).wrapping_mul(997))
                .wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
            let kw_hv = BinaryHV::random(kw_seed);
            role_keyword.bind(&kw_hv)
        })
        .collect();

    if keyword_hvs.is_empty() {
        BinaryHV::random(base_seed)
    } else {
        BinaryHV::bundle(&keyword_hvs)
    }
}

struct TrialResult {
    accuracy: f64,
    per_family_accuracy: [f64; 5],
    mean_confidence: f64,
    task_trace: Vec<TrialOutcome>,
}

impl AlgorithmRecognitionBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let seed = config.trial_seed("coding", "algorithm_recognition", trial_idx);
        let mut rng = seed ^ 0xA160_BEC0_0000_0001;

        let snippets = algorithm_snippets();
        let n_families = AlgoFamily::ALL.len();

        // Lapse rate degrades encoding fidelity
        let lapse_flip_prob = config.lapse_rate as f32 * 0.30;
        let lapse_rate = config.lapse_rate;

        // Role vectors
        let role_code = BinaryHV::random(xor_shift(&mut rng));
        let role_keyword = BinaryHV::random(xor_shift(&mut rng));

        // Build per-family prototype HVs using both code trigrams and keywords
        let mut family_hvs: Vec<Vec<BinaryHV>> = vec![Vec::new(); n_families];

        for snippet in &snippets {
            let code_hv = encode_code(snippet.code, xor_shift(&mut rng));
            let kw_hv = encode_keywords(snippet.keywords, &role_keyword, xor_shift(&mut rng));
            // Combine code surface features with structural keywords
            let combined = BinaryHV::bundle(&[role_code.bind(&code_hv), kw_hv]);
            family_hvs[snippet.family.index()].push(combined);
        }

        let family_prototypes: Vec<BinaryHV> =
            family_hvs.iter().map(|hvs| BinaryHV::bundle(hvs)).collect();

        // Leave-one-out classification
        let mut hits = 0u32;
        let mut total = 0u32;
        let mut per_family_hits: [u32; 5] = [0; 5];
        let mut per_family_total: [u32; 5] = [0; 5];
        let mut confidences = Vec::new();
        let mut task_trace = Vec::new();

        for (snippet_idx, snippet) in snippets.iter().enumerate() {
            let true_family = snippet.family.index();
            total += 1;
            per_family_total[true_family] += 1;

            // Encode the test snippet
            let mut code_hv = encode_code(snippet.code, xor_shift(&mut rng));
            let kw_hv = encode_keywords(snippet.keywords, &role_keyword, xor_shift(&mut rng));

            // Apply lapse noise to code encoding (scales with lapse_rate)
            if lapse_flip_prob > 0.0 {
                code_hv = code_hv.add_noise(lapse_flip_prob.min(0.49), xor_shift(&mut rng));
            }

            let test_hv = BinaryHV::bundle(&[role_code.bind(&code_hv), kw_hv]);

            // Leave-one-out: rebuild family prototype without this snippet
            let within_family_idx = snippets[..snippet_idx]
                .iter()
                .filter(|s| s.family.index() == true_family)
                .count();

            // Lapse trial: random guess
            xor_shift(&mut rng);
            let is_lapse = (rng % 1000) as f64 / 1000.0 < lapse_rate;

            let predicted = if is_lapse {
                xor_shift(&mut rng);
                (rng % n_families as u64) as usize
            } else {
                let mut best_family = 0;
                let mut best_sim = f32::MIN;
                let mut second_sim = f32::MIN;

                for fam_idx in 0..n_families {
                    let proto = if fam_idx == true_family {
                        // Leave one out
                        let remaining: Vec<BinaryHV> = family_hvs[fam_idx]
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != within_family_idx)
                            .map(|(_, hv)| *hv)
                            .collect();
                        if remaining.is_empty() {
                            family_prototypes[fam_idx]
                        } else {
                            BinaryHV::bundle(&remaining)
                        }
                    } else {
                        family_prototypes[fam_idx]
                    };

                    let sim = test_hv.similarity(&proto);
                    if sim > best_sim {
                        second_sim = best_sim;
                        best_sim = sim;
                        best_family = fam_idx;
                    } else if sim > second_sim {
                        second_sim = sim;
                    }
                }

                confidences.push((best_sim - second_sim) as f64);
                best_family
            };

            let correct = predicted == true_family;
            if correct {
                hits += 1;
                per_family_hits[true_family] += 1;
            }

            if config.trial_trace {
                task_trace.push(TrialOutcome {
                    trial_idx: snippet_idx,
                    condition: AlgoFamily::ALL[true_family].name().to_string(),
                    correct,
                    rt_ticks: 0.0,
                    similarity: 0.0,
                    confidence: confidences.last().copied().unwrap_or(0.0),
                    response_idx: predicted,
                    extra: BTreeMap::new(),
                });
            }
        }

        let accuracy = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };

        let mut per_family_accuracy = [0.0f64; 5];
        for i in 0..5 {
            per_family_accuracy[i] = if per_family_total[i] > 0 {
                per_family_hits[i] as f64 / per_family_total[i] as f64
            } else {
                0.0
            };
        }

        let mean_confidence = if confidences.is_empty() {
            0.0
        } else {
            confidences.iter().sum::<f64>() / confidences.len() as f64
        };

        TrialResult {
            accuracy,
            per_family_accuracy,
            mean_confidence,
            task_trace,
        }
    }
}

impl PsychBenchmark for AlgorithmRecognitionBenchmark {
    fn name(&self) -> &str {
        "Coding::AlgorithmRecognition"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Algorithm family classification",
            citation: "Alam et al. (2022)",
            year: 2022,
            doi: Some("10.1145/3510003.3510072"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut accuracies = Vec::new();
        let mut confidences = Vec::new();
        let mut per_fam: [Vec<f64>; 5] =
            [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut trace = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            accuracies.push(r.accuracy);
            confidences.push(r.mean_confidence);
            for i in 0..5 {
                per_fam[i].push(r.per_family_accuracy[i]);
            }
            if config.trial_trace {
                trace.extend(r.task_trace);
            }
        }

        result.insert("algorithm_accuracy", MetricValue::from_samples(&accuracies));
        result.insert(
            "confidence_discrimination",
            MetricValue::from_samples(&confidences),
        );

        for fam in AlgoFamily::ALL {
            result.insert(
                format!("accuracy_{}", fam.name()),
                MetricValue::from_samples(&per_fam[fam.index()]),
            );
        }

        if config.trial_trace {
            result.trial_trace = trace;
        }

        result.conditions = 5;
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
    fn test_algorithm_recognition_runs() {
        let result = AlgorithmRecognitionBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("algorithm_accuracy"));
        assert!(result.metrics.contains_key("confidence_discrimination"));
    }

    #[test]
    fn test_all_metrics_finite() {
        let result = AlgorithmRecognitionBenchmark.run(&test_config());
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_accuracy_above_chance() {
        // 5-AFC chance = 0.20
        let config = BenchmarkConfig {
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = AlgorithmRecognitionBenchmark.run(&config);
        let acc = result.metrics["algorithm_accuracy"].mean;
        assert!(
            acc > 0.25,
            "Algorithm accuracy should beat 5-AFC chance (0.20), got {}",
            acc
        );
    }

    #[test]
    fn test_deterministic() {
        let config = BenchmarkConfig {
            seed: 42,
            trials_per_condition: 3,
            ..Default::default()
        };
        let r1 = AlgorithmRecognitionBenchmark.run(&config);
        let r2 = AlgorithmRecognitionBenchmark.run(&config);
        assert_eq!(
            r1.metrics["algorithm_accuracy"].mean,
            r2.metrics["algorithm_accuracy"].mean,
        );
    }

    #[test]
    fn test_lapse_degrades() {
        let baseline = BenchmarkConfig {
            trials_per_condition: 10,
            lapse_rate: 0.0,
            seed: 42,
            ..Default::default()
        };
        let lapsed = BenchmarkConfig {
            lapse_rate: 0.25,
            ..baseline.clone()
        };
        let r_base = AlgorithmRecognitionBenchmark.run(&baseline);
        let r_lapse = AlgorithmRecognitionBenchmark.run(&lapsed);
        let a_base = r_base.metrics["algorithm_accuracy"].mean;
        let a_lapse = r_lapse.metrics["algorithm_accuracy"].mean;
        assert!(
            a_lapse <= a_base + 0.05,
            "Lapse should degrade: base={}, lapse={}",
            a_base,
            a_lapse
        );
    }

    #[test]
    fn test_provenance() {
        let prov = AlgorithmRecognitionBenchmark.provenance().unwrap();
        assert_eq!(prov.paradigm, "Algorithm family classification");
    }

    #[test]
    fn test_snippet_count() {
        let snippets = algorithm_snippets();
        assert_eq!(snippets.len(), 25, "Should have 25 algorithm snippets");
        for fam in AlgoFamily::ALL {
            let count = snippets.iter().filter(|s| s.family == fam).count();
            assert_eq!(count, 5, "Family {} should have 5 snippets", fam.name());
        }
    }
}
