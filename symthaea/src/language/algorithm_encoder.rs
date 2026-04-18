// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Abstract Algorithm Encoder — structural code features → 16,384D HDC vectors.
//!
//! Follows the Broca ThoughtEncoder pattern: define channels with normalized
//! ranges, thermometer-code each channel, bind with per-channel basis vectors,
//! bundle into a single high-dimensional representation.
//!
//! The key insight: we encode *algorithmic structure* (loop depth, data flow
//! shape, type graph) rather than surface syntax (tokens, identifiers). This
//! makes the representation language-agnostic — the same DP problem produces
//! similar HDC vectors whether the source is Rust or Python.

use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

// ─── Channel Schema ────────────────────────────────────────────────────────

/// Number of algorithm channels.
pub const NUM_CHANNELS: usize = 32;

/// Number of thermometer levels per channel.
const NUM_LEVELS: usize = 16;

/// Channel names — used as seeds for deterministic basis vector generation.
const CHANNEL_NAMES: [&str; NUM_CHANNELS] = [
    // Structural topology (0-5)
    "loop_depth",         // 0: max nesting depth of loops
    "branch_depth",       // 1: max nesting depth of if/match
    "recursion",          // 2: whether function calls itself
    "closure_count",      // 3: number of closures/lambdas
    "iterator_chain_len", // 4: longest iterator method chain
    "block_count",        // 5: number of {} blocks
    // Type features (6-11)
    "input_arity",           // 6: number of parameters
    "returns_option_result", // 7: return type wrapping level
    "generic_count",         // 8: number of type parameters
    "has_lifetime",          // 9: uses lifetime annotations
    "trait_bound_count",     // 10: number of trait bounds
    "type_complexity",       // 11: nesting depth of types (Vec<Option<T>>)
    // Algorithm class — one-hot (12-19)
    "class_sorting",        // 12
    "class_search",         // 13
    "class_dp",             // 14: dynamic programming
    "class_graph",          // 15
    "class_string",         // 16: string processing
    "class_math",           // 17: mathematical/numerical
    "class_data_structure", // 18
    "class_io_transform",   // 19: I/O transformation / format conversion
    // Data flow (20-25)
    "map_count",        // 20: map-like operations
    "filter_count",     // 21: filter operations
    "fold_count",       // 22: fold/reduce/accumulate
    "mutation_level",   // 23: 0=pure, 1=local mut, 2=&mut self, 3=global
    "allocation_level", // 24: 0=stack, 1=Vec, 2=HashMap, 3=Box/Rc
    "error_handling",   // 25: 0=none, 1=Option, 2=Result, 3=custom Error
    // Complexity (26-31)
    "line_count",            // 26: lines of code (normalized)
    "cyclomatic_complexity", // 27: independent paths
    "state_variables",       // 28: number of mutable bindings
    "helper_functions",      // 29: inner fn or separate fn count
    "pattern_match_arms",    // 30: number of match arms
    "unsafe_blocks",         // 31: number of unsafe blocks
];

/// [min, max] range for each channel.
const CHANNEL_RANGES: [[f32; 2]; NUM_CHANNELS] = [
    // Structural topology
    [0.0, 5.0],  // loop_depth
    [0.0, 5.0],  // branch_depth
    [0.0, 1.0],  // recursion (boolean)
    [0.0, 5.0],  // closure_count
    [0.0, 8.0],  // iterator_chain_len
    [0.0, 20.0], // block_count
    // Type features
    [0.0, 8.0], // input_arity
    [0.0, 2.0], // returns_option_result (0=plain, 1=Option, 2=Result)
    [0.0, 5.0], // generic_count
    [0.0, 1.0], // has_lifetime
    [0.0, 5.0], // trait_bound_count
    [0.0, 5.0], // type_complexity
    // Algorithm class (one-hot, each 0 or 1)
    [0.0, 1.0], // sorting
    [0.0, 1.0], // search
    [0.0, 1.0], // dp
    [0.0, 1.0], // graph
    [0.0, 1.0], // string
    [0.0, 1.0], // math
    [0.0, 1.0], // data_structure
    [0.0, 1.0], // io_transform
    // Data flow
    [0.0, 5.0], // map_count
    [0.0, 5.0], // filter_count
    [0.0, 3.0], // fold_count
    [0.0, 3.0], // mutation_level
    [0.0, 3.0], // allocation_level
    [0.0, 3.0], // error_handling
    // Complexity
    [0.0, 100.0], // line_count
    [0.0, 20.0],  // cyclomatic_complexity
    [0.0, 10.0],  // state_variables
    [0.0, 5.0],   // helper_functions
    [0.0, 20.0],  // pattern_match_arms
    [0.0, 3.0],   // unsafe_blocks
];

// ─── AlgorithmChannels ─────────────────────────────────────────────────────

/// 32-channel representation of abstract algorithm structure.
#[derive(Clone, Debug)]
pub struct AlgorithmChannels {
    pub channels: [f32; NUM_CHANNELS],
}

impl Default for AlgorithmChannels {
    fn default() -> Self {
        Self {
            channels: [0.0; NUM_CHANNELS],
        }
    }
}

impl AlgorithmChannels {
    // ── Structural setters ──

    pub fn set_loop_depth(&mut self, v: f32) {
        self.channels[0] = v;
    }
    pub fn set_branch_depth(&mut self, v: f32) {
        self.channels[1] = v;
    }
    pub fn set_recursion(&mut self, v: bool) {
        self.channels[2] = if v { 1.0 } else { 0.0 };
    }
    pub fn set_closure_count(&mut self, v: f32) {
        self.channels[3] = v;
    }
    pub fn set_iterator_chain_len(&mut self, v: f32) {
        self.channels[4] = v;
    }
    pub fn set_block_count(&mut self, v: f32) {
        self.channels[5] = v;
    }

    // ── Type setters ──

    pub fn set_input_arity(&mut self, v: f32) {
        self.channels[6] = v;
    }
    pub fn set_returns_option_result(&mut self, v: f32) {
        self.channels[7] = v;
    }
    pub fn set_generic_count(&mut self, v: f32) {
        self.channels[8] = v;
    }
    pub fn set_has_lifetime(&mut self, v: bool) {
        self.channels[9] = if v { 1.0 } else { 0.0 };
    }
    pub fn set_trait_bound_count(&mut self, v: f32) {
        self.channels[10] = v;
    }
    pub fn set_type_complexity(&mut self, v: f32) {
        self.channels[11] = v;
    }

    // ── Algorithm class (one-hot) ──

    pub fn set_class(&mut self, class: AlgorithmClass) {
        // Clear all class channels
        for i in 12..20 {
            self.channels[i] = 0.0;
        }
        let idx = match class {
            AlgorithmClass::Sorting => 12,
            AlgorithmClass::Search => 13,
            AlgorithmClass::DynamicProgramming => 14,
            AlgorithmClass::Graph => 15,
            AlgorithmClass::StringProcessing => 16,
            AlgorithmClass::Mathematical => 17,
            AlgorithmClass::DataStructure => 18,
            AlgorithmClass::IoTransform => 19,
        };
        self.channels[idx] = 1.0;
    }

    // ── Data flow setters ──

    pub fn set_map_count(&mut self, v: f32) {
        self.channels[20] = v;
    }
    pub fn set_filter_count(&mut self, v: f32) {
        self.channels[21] = v;
    }
    pub fn set_fold_count(&mut self, v: f32) {
        self.channels[22] = v;
    }
    pub fn set_mutation_level(&mut self, v: f32) {
        self.channels[23] = v;
    }
    pub fn set_allocation_level(&mut self, v: f32) {
        self.channels[24] = v;
    }
    pub fn set_error_handling(&mut self, v: f32) {
        self.channels[25] = v;
    }

    // ── Complexity setters ──

    pub fn set_line_count(&mut self, v: f32) {
        self.channels[26] = v;
    }
    pub fn set_cyclomatic_complexity(&mut self, v: f32) {
        self.channels[27] = v;
    }
    pub fn set_state_variables(&mut self, v: f32) {
        self.channels[28] = v;
    }
    pub fn set_helper_functions(&mut self, v: f32) {
        self.channels[29] = v;
    }
    pub fn set_pattern_match_arms(&mut self, v: f32) {
        self.channels[30] = v;
    }
    pub fn set_unsafe_blocks(&mut self, v: f32) {
        self.channels[31] = v;
    }
}

// ─── Algorithm Classification ──────────────────────────────────────────────

/// High-level algorithm family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AlgorithmClass {
    Sorting,
    Search,
    DynamicProgramming,
    Graph,
    StringProcessing,
    Mathematical,
    DataStructure,
    IoTransform,
}

impl AlgorithmClass {
    pub const ALL: [Self; 8] = [
        Self::Sorting,
        Self::Search,
        Self::DynamicProgramming,
        Self::Graph,
        Self::StringProcessing,
        Self::Mathematical,
        Self::DataStructure,
        Self::IoTransform,
    ];

    /// Channel index for this class (12-19).
    pub fn channel_index(&self) -> usize {
        match self {
            Self::Sorting => 12,
            Self::Search => 13,
            Self::DynamicProgramming => 14,
            Self::Graph => 15,
            Self::StringProcessing => 16,
            Self::Mathematical => 17,
            Self::DataStructure => 18,
            Self::IoTransform => 19,
        }
    }

    /// Read the class from a channel vector.
    pub fn from_channels(channels: &AlgorithmChannels) -> Self {
        let mut best = Self::IoTransform;
        let mut best_val = -1.0f32;
        for class in Self::ALL {
            let val = channels.channels[class.channel_index()];
            if val > best_val {
                best_val = val;
                best = class;
            }
        }
        best
    }
}

// ─── Feature Extractor ─────────────────────────────────────────────────────

/// Extract abstract algorithm features from source code.
///
/// Uses pattern matching on source text — intentionally avoids tree-sitter
/// to keep the encoder language-agnostic. The patterns capture STRUCTURAL
/// properties (loops, branches, data flow) rather than syntax.
pub fn extract_features(source: &str) -> AlgorithmChannels {
    let mut ch = AlgorithmChannels::default();

    // ── Structural topology ──

    // Loop depth: count max nesting of for/while/loop
    let mut max_loop_depth = 0u32;
    let mut current_loop_depth = 0u32;
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("for ")
            || trimmed.starts_with("while ")
            || trimmed == "loop {"
            || trimmed.starts_with("loop {")
        {
            current_loop_depth += 1;
            max_loop_depth = max_loop_depth.max(current_loop_depth);
        }
        // Heuristic: closing brace at low indentation reduces loop depth
        if trimmed == "}" && current_loop_depth > 0 {
            current_loop_depth = current_loop_depth.saturating_sub(1);
        }
    }
    ch.set_loop_depth(max_loop_depth as f32);

    // Branch depth: count if/match nesting
    let mut max_branch = 0u32;
    let mut branch_depth = 0u32;
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("if ")
            || trimmed.starts_with("match ")
            || trimmed.starts_with("} else if ")
            || trimmed.starts_with("else {")
        {
            branch_depth += 1;
            max_branch = max_branch.max(branch_depth);
        }
        if trimmed == "}" && branch_depth > 0 {
            branch_depth = branch_depth.saturating_sub(1);
        }
    }
    ch.set_branch_depth(max_branch as f32);

    // Recursion: function calls itself
    // Find the first `fn name(` and check if `name(` appears in the body
    let has_recursion = detect_recursion(source);
    ch.set_recursion(has_recursion);

    // Closure count
    let closure_count = source.matches("|").count() / 2; // pairs of pipes
    ch.set_closure_count(closure_count as f32);

    // Iterator chain length: longest sequence of .method() calls
    let max_chain = source
        .lines()
        .map(|line| {
            line.matches('.')
                .count()
                .saturating_sub(if line.contains("..") { 1 } else { 0 })
        })
        .max()
        .unwrap_or(0);
    ch.set_iterator_chain_len(max_chain as f32);

    // Block count
    ch.set_block_count(source.matches('{').count() as f32);

    // ── Type features ──

    // Input arity: count commas in first fn signature + 1
    if let Some(sig) = find_first_fn_signature(source) {
        let params = sig
            .split('(')
            .nth(1)
            .and_then(|s| s.split(')').next())
            .unwrap_or("");
        let arity =
            if params.trim().is_empty() || params.trim() == "&self" || params.trim() == "&mut self"
            {
                0
            } else {
                params.split(',').count()
            };
        ch.set_input_arity(arity as f32);

        // Return type analysis
        if sig.contains("-> Option<") {
            ch.set_returns_option_result(1.0);
        } else if sig.contains("-> Result<") || sig.contains("Result<") {
            ch.set_returns_option_result(2.0);
        }

        // Generics
        let generic_count = sig.matches('<').count();
        ch.set_generic_count(generic_count as f32);

        // Lifetimes
        ch.set_has_lifetime(sig.contains("'a") || sig.contains("'_") || sig.contains("'b"));

        // Trait bounds
        let trait_bounds = source.matches("where").count() + source.matches(": Fn").count();
        ch.set_trait_bound_count(trait_bounds as f32);
    }

    // Type complexity: nesting depth of angle brackets
    let max_type_depth = source
        .lines()
        .map(|l| {
            let mut depth = 0u32;
            let mut max_d = 0u32;
            for c in l.chars() {
                match c {
                    '<' => {
                        depth += 1;
                        max_d = max_d.max(depth);
                    }
                    '>' => depth = depth.saturating_sub(1),
                    _ => {}
                }
            }
            max_d
        })
        .max()
        .unwrap_or(0);
    ch.set_type_complexity(max_type_depth as f32);

    // ── Algorithm class (inferred from keywords) ──

    let lower = source.to_lowercase();
    let class = classify_algorithm(&lower, source);
    ch.set_class(class);

    // ── Data flow ──

    ch.set_map_count(count_pattern(source, ".map(") as f32);
    ch.set_filter_count(count_pattern(source, ".filter(") as f32);
    ch.set_fold_count(
        (count_pattern(source, ".fold(")
            + count_pattern(source, ".reduce(")
            + count_pattern(source, ".sum()")) as f32,
    );

    // Mutation level
    let mutation = if source.contains("unsafe") {
        3.0
    } else if source.contains("&mut self") {
        2.0
    } else if source.contains("let mut ") {
        1.0
    } else {
        0.0
    };
    ch.set_mutation_level(mutation);

    // Allocation level
    let alloc = if source.contains("Rc<") || source.contains("Box<") || source.contains("Arc<") {
        3.0
    } else if source.contains("HashMap")
        || source.contains("HashSet")
        || source.contains("BTreeMap")
    {
        2.0
    } else if source.contains("Vec<") || source.contains("vec![") || source.contains("String") {
        1.0
    } else {
        0.0
    };
    ch.set_allocation_level(alloc);

    // Error handling
    let err = if source.contains("enum Error") || source.contains("pub enum Error") {
        3.0
    } else if source.contains("Result<") {
        2.0
    } else if source.contains("Option<") {
        1.0
    } else {
        0.0
    };
    ch.set_error_handling(err);

    // ── Complexity ──

    ch.set_line_count(source.lines().count() as f32);
    ch.set_cyclomatic_complexity(
        (count_pattern(source, "if ")
            + count_pattern(source, "else")
            + count_pattern(source, "for ")
            + count_pattern(source, "while ")
            + count_pattern(source, "match ")
            + count_pattern(source, "&&")
            + count_pattern(source, "||")
            + 1) as f32,
    );

    let mut_count = count_pattern(source, "let mut ");
    ch.set_state_variables(mut_count as f32);

    let fn_count = count_pattern(source, "fn ").saturating_sub(1); // first fn is the main one
    ch.set_helper_functions(fn_count as f32);

    // Match arms: count `=>` occurrences
    ch.set_pattern_match_arms(count_pattern(source, "=>") as f32);

    ch.set_unsafe_blocks(count_pattern(source, "unsafe") as f32);

    ch
}

// ─── Helpers ───────────────────────────────────────────────────────────────

fn count_pattern(source: &str, pattern: &str) -> usize {
    source.matches(pattern).count()
}

fn find_first_fn_signature(source: &str) -> Option<String> {
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("pub fn ") || trimmed.starts_with("fn ") {
            return Some(trimmed.to_string());
        }
    }
    None
}

fn detect_recursion(source: &str) -> bool {
    // Find the name of the first fn and check if it appears in the body
    let mut fn_name = None;
    let mut in_body = false;
    for line in source.lines() {
        let trimmed = line.trim();
        if fn_name.is_none() && (trimmed.starts_with("pub fn ") || trimmed.starts_with("fn ")) {
            let after_fn = if trimmed.starts_with("pub fn ") {
                &trimmed[7..]
            } else {
                &trimmed[3..]
            };
            let name: String = after_fn
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                fn_name = Some(name);
                in_body = true;
                continue;
            }
        }
        if in_body {
            if let Some(ref name) = fn_name {
                let call_pattern = format!("{name}(");
                if trimmed.contains(&call_pattern) {
                    return true;
                }
            }
        }
    }
    false
}

fn classify_algorithm(lower: &str, source: &str) -> AlgorithmClass {
    // Score each class by keyword presence
    let scores = [
        (
            AlgorithmClass::Sorting,
            kw_score(
                lower,
                &[
                    "sort",
                    "swap",
                    "partition",
                    "pivot",
                    "merge_sorted",
                    "bubble",
                ],
            ),
        ),
        (
            AlgorithmClass::Search,
            kw_score(
                lower,
                &[
                    "search",
                    "find",
                    "binary_search",
                    "bfs",
                    "dfs",
                    "lookup",
                    "index",
                ],
            ),
        ),
        (
            AlgorithmClass::DynamicProgramming,
            kw_score(
                lower,
                &["dp[", "dp =", "memo", "tabul", "knapsack", "subsequence"],
            ) + if source.contains("dp[") || source.contains("dp =") {
                2
            } else {
                0
            },
        ),
        (
            AlgorithmClass::Graph,
            kw_score(
                lower,
                &[
                    "graph", "node", "edge", "adjacen", "vertex", "path", "visited", "queue",
                ],
            ),
        ),
        (
            AlgorithmClass::StringProcessing,
            kw_score(
                lower,
                &[
                    "chars()", "string", "str", "parse", "format!", "split", "join", "trim",
                ],
            ) + if source.contains(".chars()") || source.contains("&str") {
                1
            } else {
                0
            },
        ),
        (
            AlgorithmClass::Mathematical,
            kw_score(
                lower,
                &[
                    "sqrt",
                    "pow",
                    "prime",
                    "gcd",
                    "factorial",
                    "modulo",
                    "fibonacci",
                    "sum",
                ],
            ),
        ),
        (
            AlgorithmClass::DataStructure,
            kw_score(
                lower,
                &[
                    "struct", "impl", "push", "pop", "insert", "remove", "capacity", "len()",
                ],
            ),
        ),
        (
            AlgorithmClass::IoTransform,
            kw_score(
                lower,
                &[
                    "read",
                    "write",
                    "encode",
                    "decode",
                    "convert",
                    "transform",
                    "serialize",
                ],
            ),
        ),
    ];

    scores
        .iter()
        .max_by_key(|(_, s)| *s)
        .map(|(c, _)| *c)
        .unwrap_or(AlgorithmClass::IoTransform)
}

fn kw_score(text: &str, keywords: &[&str]) -> usize {
    keywords.iter().filter(|kw| text.contains(*kw)).count()
}

// ─── HDC Encoder ───────────────────────────────────────────────────────────

/// Encodes AlgorithmChannels into a 16,384D ContinuousHV.
///
/// Follows the Broca ThoughtLanguageEncoder pattern:
/// 1. For each channel, normalize to [0,1]
/// 2. Thermometer-code the normalized value (bundle first k level vectors)
/// 3. Bind with per-channel basis vector
/// 4. Bundle all bound channel vectors
pub struct AlgorithmEncoder {
    base_vectors: Vec<ContinuousHV>,
    level_vectors: Vec<ContinuousHV>,
}

impl AlgorithmEncoder {
    /// Create encoder with deterministic basis vectors.
    pub fn new() -> Self {
        // Use deterministic seeds for reproducibility
        let base_vectors: Vec<ContinuousHV> = CHANNEL_NAMES
            .iter()
            .map(|name| {
                ContinuousHV::random(HDC_DIMENSION, hash_seed(&format!("algo::channel::{name}")))
            })
            .collect();

        let level_vectors: Vec<ContinuousHV> = (0..NUM_LEVELS)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, hash_seed(&format!("algo::level::{i}"))))
            .collect();

        Self {
            base_vectors,
            level_vectors,
        }
    }

    /// Encode algorithm channels into a single HDC vector.
    pub fn encode(&self, channels: &AlgorithmChannels) -> ContinuousHV {
        let mut bound_hvs: Vec<ContinuousHV> = Vec::with_capacity(NUM_CHANNELS);

        for (i, &raw) in channels.channels.iter().enumerate() {
            let value = if raw.is_finite() { raw } else { 0.0 };
            let normalized = normalize_channel(i, value);
            let level_hv = self.encode_level(normalized);
            bound_hvs.push(self.base_vectors[i].bind(&level_hv));
        }

        let refs: Vec<&ContinuousHV> = bound_hvs.iter().collect();
        ContinuousHV::bundle(&refs)
    }

    /// Thermometer-code a normalized [0,1] value.
    fn encode_level(&self, normalized: f32) -> ContinuousHV {
        let k = ((normalized * NUM_LEVELS as f32) as usize).min(NUM_LEVELS - 1);
        if k == 0 {
            self.level_vectors[0].clone()
        } else {
            let refs: Vec<&ContinuousHV> = self.level_vectors[..=k].iter().collect();
            ContinuousHV::bundle(&refs)
        }
    }

    /// Compute similarity between two algorithm encodings.
    pub fn similarity(a: &ContinuousHV, b: &ContinuousHV) -> f32 {
        a.similarity(b)
    }
}

fn normalize_channel(channel: usize, value: f32) -> f32 {
    let [min, max] = CHANNEL_RANGES[channel];
    if (max - min).abs() < 1e-8 {
        return 0.5;
    }
    ((value - min) / (max - min)).clamp(0.0, 1.0)
}

fn hash_seed(label: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
    for byte in label.bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3); // FNV-1a prime
    }
    h
}

// ─── Training Data ─────────────────────────────────────────────────────────

/// A training pair: problem features → algorithm class + plan actions.
#[derive(Clone, Debug)]
pub struct AlgorithmTrainingPair {
    /// Exercise name (for diagnostics).
    pub name: String,
    /// Extracted structural features.
    pub channels: AlgorithmChannels,
    /// Encoded HDC vector.
    pub hv: ContinuousHV,
    /// Ground-truth algorithm class.
    pub class: AlgorithmClass,
    /// Problem purpose/description (for the decoder path).
    pub purpose: String,
    /// Source code for 1-NN body retrieval.
    pub source: String,
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_simple_function() {
        let source = r#"pub fn reverse(input: &str) -> String {
    input.chars().rev().collect()
}"#;
        let ch = extract_features(source);
        assert_eq!(ch.channels[0], 0.0, "no loops");
        assert!(ch.channels[4] > 0.0, "has iterator chain");
        assert_eq!(ch.channels[6], 1.0, "one parameter");
    }

    #[test]
    fn test_extract_loop_heavy() {
        let source = r#"pub fn sieve(n: u64) -> Vec<u64> {
    let mut is_prime = vec![true; n as usize + 1];
    for i in 2..=n as usize {
        if is_prime[i] {
            for j in (i*i..=n as usize).step_by(i) {
                is_prime[j] = false;
            }
        }
    }
    (2..=n as usize).filter(|&i| is_prime[i]).map(|i| i as u64).collect()
}"#;
        let ch = extract_features(source);
        assert!(ch.channels[0] >= 2.0, "nested loops: {}", ch.channels[0]);
        assert!(ch.channels[23] >= 1.0, "has mutation");
    }

    #[test]
    fn test_extract_recursive() {
        let source = r#"pub fn factorial(n: u64) -> u64 {
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}"#;
        let ch = extract_features(source);
        assert_eq!(ch.channels[2], 1.0, "detects recursion");
    }

    #[test]
    fn test_extract_dp() {
        let source = r#"pub fn knapsack(max_weight: u32, items: &[Item]) -> u32 {
    let w = max_weight as usize;
    let mut dp = vec![0u32; w + 1];
    for item in items {
        for cap in (item.weight as usize..=w).rev() {
            dp[cap] = dp[cap].max(dp[cap - item.weight as usize] + item.value);
        }
    }
    dp[w]
}"#;
        let ch = extract_features(source);
        assert_eq!(
            AlgorithmClass::from_channels(&ch),
            AlgorithmClass::DynamicProgramming,
            "classifies as DP"
        );
    }

    #[test]
    fn test_encoder_determinism() {
        let enc1 = AlgorithmEncoder::new();
        let enc2 = AlgorithmEncoder::new();
        let ch = AlgorithmChannels::default();
        let hv1 = enc1.encode(&ch);
        let hv2 = enc2.encode(&ch);
        assert!(
            (hv1.similarity(&hv2) - 1.0).abs() < 1e-5,
            "deterministic encoding"
        );
    }

    #[test]
    fn test_different_algorithms_produce_different_hvs() {
        let enc = AlgorithmEncoder::new();

        // Rich sorting profile
        let mut ch_sort = AlgorithmChannels::default();
        ch_sort.set_class(AlgorithmClass::Sorting);
        ch_sort.set_loop_depth(2.0);
        ch_sort.set_mutation_level(2.0);
        ch_sort.set_line_count(20.0);
        ch_sort.set_state_variables(2.0);
        ch_sort.set_input_arity(1.0);

        // Rich DP profile
        let mut ch_dp = AlgorithmChannels::default();
        ch_dp.set_class(AlgorithmClass::DynamicProgramming);
        ch_dp.set_loop_depth(2.0);
        ch_dp.set_allocation_level(1.0);
        ch_dp.set_line_count(15.0);
        ch_dp.set_input_arity(2.0);
        ch_dp.set_cyclomatic_complexity(5.0);

        let hv_sort = enc.encode(&ch_sort);
        let hv_dp = enc.encode(&ch_dp);

        let sim = hv_sort.similarity(&hv_dp);
        assert!(
            sim < 0.95,
            "different algorithms should produce dissimilar HVs: {sim}"
        );
    }

    #[test]
    fn test_similar_algorithms_cluster() {
        let enc = AlgorithmEncoder::new();

        // Two sorting variants should be more similar than sorting vs DP
        let mut ch_sort1 = AlgorithmChannels::default();
        ch_sort1.set_class(AlgorithmClass::Sorting);
        ch_sort1.set_loop_depth(2.0);
        ch_sort1.set_mutation_level(1.0);

        let mut ch_sort2 = AlgorithmChannels::default();
        ch_sort2.set_class(AlgorithmClass::Sorting);
        ch_sort2.set_loop_depth(1.0);
        ch_sort2.set_mutation_level(2.0);

        let mut ch_dp = AlgorithmChannels::default();
        ch_dp.set_class(AlgorithmClass::DynamicProgramming);
        ch_dp.set_loop_depth(2.0);
        ch_dp.set_allocation_level(1.0);

        let hv_sort1 = enc.encode(&ch_sort1);
        let hv_sort2 = enc.encode(&ch_sort2);
        let hv_dp = enc.encode(&ch_dp);

        let sim_sorts = hv_sort1.similarity(&hv_sort2);
        let sim_sort_dp = hv_sort1.similarity(&hv_dp);

        assert!(
            sim_sorts > sim_sort_dp,
            "sorting variants should cluster: sort-sort={sim_sorts:.3} > sort-dp={sim_sort_dp:.3}"
        );
    }

    #[test]
    fn test_channel_ranges_valid() {
        for (i, [min, max]) in CHANNEL_RANGES.iter().enumerate() {
            assert!(
                max > min || (max - min).abs() < 1e-8,
                "channel {i} ({}) has invalid range [{min}, {max}]",
                CHANNEL_NAMES[i]
            );
        }
    }
}
