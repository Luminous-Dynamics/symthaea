// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Prompt building and templates for the coding agent.

use super::*;

impl CodingAgent {
    /// Build the prompt sent to the LLM for code generation.
    pub(super) fn build_generation_prompt(&self) -> String {
        let mut prompt = String::with_capacity(2048);

        prompt.push_str(&format!("Task: {}\n\n", self.task));

        // Classify task type for structured guidance
        {
            use crate::cognitive_loop::routing::CodeTaskDetector;
            let detector = CodeTaskDetector::new();
            let task_type = detector.detect_task_type(&self.task);
            let guidance = match task_type {
                crate::cognitive_loop::routing::CodeTaskType::Create => {
                    "Type: CREATE — Write a complete, compilable Rust implementation.\n\
                     Rules: Output ONLY the code (no fn main wrapper for library items). \
                     Include all necessary `use` statements. Declare generic type parameters \
                     explicitly (e.g., `<T>` on struct/fn/impl). Add `pub` to public items.\n"
                }
                crate::cognitive_loop::routing::CodeTaskType::Debug => {
                    "Type: DEBUG — Fix the broken code.\n\
                     Rules: Preserve the existing API. Only change what's needed to fix the error. \
                     Show the complete fixed file, not just the changed lines.\n"
                }
                crate::cognitive_loop::routing::CodeTaskType::Refactor => {
                    "Type: REFACTOR — Restructure without changing behavior.\n\
                     Rules: Keep the same public API. Improve clarity and performance. \
                     Show the complete refactored file.\n"
                }
                _ => "",
            };
            if !guidance.is_empty() {
                prompt.push_str(guidance);
                prompt.push('\n');
            }
        }

        // HDC context from indexed codebase memory (similarity search results)
        let hdc_ctx = self.build_hdc_context_prompt();
        if !hdc_ctx.is_empty() {
            prompt.push_str(&hdc_ctx);
            prompt.push('\n');
        }

        // Dynamic re-query: in Fixing phase, query CodebaseMemory for types/functions
        // mentioned in error messages (supplements the static code_context)
        let dynamic_ctx = self.build_dynamic_error_context();
        if !dynamic_ctx.is_empty() {
            prompt.push_str("## Additional context from error analysis\n");
            prompt.push_str(&dynamic_ctx);
            prompt.push('\n');
        }

        // Include codebase context from CodebaseMemory
        if !self.code_context.is_empty() {
            prompt.push_str("Relevant code from the project:\n");
            for ctx in &self.code_context {
                prompt.push_str(&format!("---\n{}\n", ctx));
            }
            prompt.push_str("---\n\n");
        }

        // Include observations (file contents read, etc.)
        if !self.observations.is_empty() {
            prompt.push_str("Context from prior analysis:\n");
            for obs in self.observations.iter().rev().take(5).rev() {
                prompt.push_str(&format!("- {}\n", obs));
            }
            prompt.push('\n');
        }

        // In Fixing phase, include both raw error and structured test failure analysis
        if self.phase == TaskPhase::Fixing {
            if let Some(ref test_output) = self.last_test_output {
                // Parse structured test failures for targeted fixing
                let structured = Self::parse_test_failures(test_output);
                if !structured.is_empty() {
                    prompt.push_str(&Self::format_structured_test_failures(&structured));
                }
                prompt.push_str(&format!(
                    "The previous code failed with this error:\n```\n{}\n```\n\nFix the code.\n",
                    test_output
                ));
            }
            // Include retry strategy hints
            match &self.retry_state.current_strategy {
                RetryStrategy::DifferentTemplate => {
                    prompt.push_str("\nTry a completely different implementation approach.\n");
                }
                RetryStrategy::SimplifyScope => {
                    prompt.push_str(
                        "\nSimplify the implementation — use the minimal viable approach.\n",
                    );
                }
                _ => {}
            }
        }

        // Inject semantically-ranked fix suggestions from the error knowledge graph.
        // These are Bayesian-ranked strategies that worked for similar errors in the past.
        if self.phase == TaskPhase::Fixing {
            if let Some(ref test_output) = self.last_test_output {
                let structured =
                    crate::language::code_executor::parse_structured_errors(test_output);
                let mut suggestions_added = 0;
                for error in structured.iter().take(3) {
                    let error_code = error.code.clone().unwrap_or_default();
                    let category =
                        super::error_knowledge::ErrorCategory::from_error_code(&error_code);
                    let fixes = self.error_knowledge.suggest_fixes(&error_code, category);
                    if !fixes.is_empty() {
                        if suggestions_added == 0 {
                            prompt.push_str("## Learned fix strategies (ranked by success rate)\n");
                        }
                        for fix in fixes.iter().take(2) {
                            prompt.push_str(&format!(
                                "- {} (success rate: {:.0}%, {} attempts): {}\n",
                                error_code,
                                fix.success_rate * 100.0,
                                fix.attempts,
                                fix.strategy
                            ));
                        }
                        suggestions_added += 1;
                    }
                }
                if suggestions_added > 0 {
                    prompt.push('\n');
                }
            }
        }

        // Inject failure patterns from THIS run — these are errors we've already
        // seen, so the generator should avoid repeating them.
        if !self.failure_patterns.is_empty() {
            prompt.push_str("Errors encountered in this session (AVOID these patterns):\n");
            for (pattern, count) in self.failure_patterns.iter().take(5) {
                prompt.push_str(&format!("- ({count}x) {pattern}\n"));
            }
            prompt.push('\n');
        }

        // Inject GCS topological/structural violations as hard constraints.
        // These come from the previous generation's GCS verification pass and
        // MUST be addressed by the next generation attempt.
        if !self.gcs_violations.is_empty() {
            prompt.push_str("## Topological/Structural Violations (MUST FIX)\n");
            for v in &self.gcs_violations {
                prompt.push_str(&format!("- {}\n", v));
            }
            prompt.push('\n');
        }

        // Inject experience hints from persistent store (cross-session learning)
        let hints = self.retrieve_experience_hints();
        if !hints.is_empty() {
            prompt.push_str("Relevant patterns from past experience:\n");
            for (pattern, hint) in hints.iter().take(3) {
                prompt.push_str(&format!("- Error: {} -> Fix: {}\n", pattern, hint));
            }
            prompt.push('\n');
        }

        // Inject successful code patterns from past sessions
        let successes = self.retrieve_success_patterns();
        if !successes.is_empty() {
            prompt.push_str("Successful implementations from prior sessions:\n");
            for (task, code_summary) in successes.iter().take(2) {
                prompt.push_str(&format!("- Task: {} -> Code: {}\n", task, code_summary));
            }
            prompt.push('\n');
        }

        // Inject plan constraints (energy, Phi, risk level)
        if let Some(ref plan) = self.current_plan {
            prompt.push_str("## Plan constraints\n");
            prompt.push_str(&format!(
                "- Energy budget remaining: {:.0}\n- Min Phi required: {:.2}\n- Risk level: {:?}\n- Reversible: {}\n- Steps: {}\n\n",
                self.energy_budget,
                plan.min_phi,
                plan.max_destructiveness,
                if plan.fully_reversible { "yes" } else { "no" },
                plan.step_count,
            ));
        }

        // Infer language from target file extension
        let target = self.resolve_target_file();
        let lang = target
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("rust");
        prompt.push_str(&format!("Generate valid {} code.\n", lang));

        prompt
    }

    /// Retrieve relevant error hints from the persistent experience store.
    pub(super) fn retrieve_experience_hints(&self) -> Vec<(String, String)> {
        if let Some(ref store) = self.experience_store {
            if let Ok(rt) = tokio::runtime::Runtime::new() {
                return rt.block_on(async { store.error_hints_for(&self.task, 3).await });
            }
        }
        Vec::new()
    }

    /// Retrieve successful code patterns from the persistent experience store.
    ///
    /// Returns `(task_description, code_summary)` pairs from past sessions
    /// where similar tasks were completed successfully.
    pub(super) fn retrieve_success_patterns(&self) -> Vec<(String, String)> {
        if let Some(ref store) = self.experience_store {
            // First check cache (fast, no async)
            let self_task_lower = self.task.to_lowercase();
            let self_words: std::collections::HashSet<String> = self_task_lower
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();
            let cached: Vec<(String, String)> = store
                .cached_successes()
                .iter()
                .filter(|(task, _)| {
                    let task_lower = task.to_lowercase();
                    let task_words: std::collections::HashSet<String> = task_lower
                        .split_whitespace()
                        .map(|s| s.to_string())
                        .collect();
                    task_words.intersection(&self_words).count() >= 2
                })
                .take(3)
                .cloned()
                .collect();

            if !cached.is_empty() {
                return cached;
            }

            // Fall back to similarity search
            if let Ok(rt) = tokio::runtime::Runtime::new() {
                return rt.block_on(async {
                    let results = store.query_similar(&self.task, 5).await;
                    results
                        .into_iter()
                        .filter(|r| r.record.valence > 0.0 && r.similarity > 0.5)
                        .map(|r| {
                            let summary = r.record.topics.first().cloned().unwrap_or_default();
                            (r.record.content, summary)
                        })
                        .take(3)
                        .collect()
                });
            }
        }
        Vec::new()
    }

    /// Generate code using native (non-LLM) pattern-aware templates.
    ///
    /// Matches task descriptions against a library of common coding patterns
    /// (arithmetic, string ops, sorting, searching, data structures, etc.)
    /// and generates compilable, tested implementations. Returns `None` if
    /// the task doesn't match any known pattern — the caller should escalate
    /// to an LLM tier rather than produce a TODO stub.
    pub(super) fn native_code_template(&self) -> Option<String> {
        let task_lower = self.task.to_lowercase();

        // Phase 1: Hardcoded pattern templates (exact keyword matches — fast, reliable)
        // These templates are Rust source verbatim (doc comments, `pub fn`, etc.) —
        // only apply them when the target language is actually Rust, otherwise they
        // leak Rust syntax into e.g. generated Python.
        if self.target_language() == "rust" {
            if let Some(code) = Self::match_native_pattern(&task_lower) {
                return Some(code);
            }
        }

        // Phase 2: HDC Program Algebra (semantic matching -> generated code)
        // This is the neuro-symbolic path: encode task -> find similar pattern -> translate to code
        // Runs after hardcoded patterns so exact matches always win over fuzzy HDC similarity.
        {
            use crate::language::program_node_translator;
            use symthaea_core::hdc::program_algebra::{
                ProgramPatternLibrary, encode_task_description,
            };

            let task_hv = encode_task_description(&task_lower);
            let lib = ProgramPatternLibrary::standard();
            // BinaryHV similarity: 0.5 = random, > 0.52 = meaningful match
            if let Some((entry, similarity)) = lib.find_similar(&task_hv, 0.52) {
                let function_name =
                    Self::extract_function_name(&task_lower).unwrap_or_else(|| entry.name.clone());
                let language = self.target_language();

                if let Some(code) =
                    program_node_translator::translate(entry, language, &function_name)
                {
                    tracing::info!(
                        target: "symthaea::coding_agent",
                        pattern = %entry.name,
                        similarity = similarity,
                        language = language,
                        "HDC program algebra match -> code generated"
                    );
                    return Some(code);
                }
            }
        }

        // Try learned templates from past successful LLM generations
        if let Some(ref store) = self.experience_store {
            if let Some(template) = store.lookup_learned_template(&self.task) {
                tracing::info!(
                    target: "symthaea::coding_agent",
                    task = %self.task,
                    template_len = template.len(),
                    "Using learned template from past LLM generation"
                );
                return Some(template.to_string());
            }
        }

        // No match — signal that native generation can't handle this task.
        // The caller should escalate to an LLM tier.
        None
    }

    /// Match task against known coding patterns and return compilable code.
    pub(super) fn match_native_pattern(task: &str) -> Option<String> {
        // -- Arithmetic / Math --
        if task.contains("fibonacci") || task.contains("fib") {
            return Some(
                "/// Compute the nth Fibonacci number.\npub fn fibonacci(n: u64) -> u64 {\n    match n {\n        0 => 0,\n        1 => 1,\n        _ => {\n            let (mut a, mut b) = (0u64, 1u64);\n            for _ in 2..=n {\n                let c = a.saturating_add(b);\n                a = b;\n                b = c;\n            }\n            b\n        }\n    }\n}\n"
                    .to_string(),
            );
        }
        if task.contains("factorial") {
            return Some(
                "/// Compute the factorial of n.\npub fn factorial(n: u64) -> u64 {\n    (1..=n).fold(1u64, |acc, x| acc.saturating_mul(x))\n}\n"
                    .to_string(),
            );
        }
        if task.contains("gcd") || task.contains("greatest common") {
            return Some(
                "/// Compute the greatest common divisor using Euclid's algorithm.\npub fn gcd(mut a: u64, mut b: u64) -> u64 {\n    while b != 0 {\n        let t = b;\n        b = a % b;\n        a = t;\n    }\n    a\n}\n"
                    .to_string(),
            );
        }
        if task.contains("is_prime") || task.contains("prime check") || task.contains("primality") {
            return Some(
                "/// Check if a number is prime.\npub fn is_prime(n: u64) -> bool {\n    if n < 2 { return false; }\n    if n < 4 { return true; }\n    if n % 2 == 0 || n % 3 == 0 { return false; }\n    let mut i = 5;\n    while i * i <= n {\n        if n % i == 0 || n % (i + 2) == 0 { return false; }\n        i += 6;\n    }\n    true\n}\n"
                    .to_string(),
            );
        }
        if task.contains("absolute") || task.contains("abs ") {
            return Some(
                "/// Return the absolute value.\npub fn absolute(x: i64) -> u64 {\n    x.unsigned_abs()\n}\n"
                    .to_string(),
            );
        }

        // -- String operations --
        if task.contains("hello") {
            return Some(
                "/// Return a greeting.\npub fn hello() -> &'static str {\n    \"Hello, world!\"\n}\n"
                    .to_string(),
            );
        }
        if task.contains("reverse") && task.contains("string") {
            return Some(
                "/// Reverse a string.\npub fn reverse_string(s: &str) -> String {\n    s.chars().rev().collect()\n}\n"
                    .to_string(),
            );
        }
        if task.contains("palindrome") {
            return Some(
                "/// Check if a string is a palindrome.\npub fn is_palindrome(s: &str) -> bool {\n    let s: String = s.chars().filter(|c| c.is_alphanumeric()).flat_map(|c| c.to_lowercase()).collect();\n    s == s.chars().rev().collect::<String>()\n}\n"
                    .to_string(),
            );
        }
        if task.contains("count") && task.contains("vowel") {
            return Some(
                "/// Count vowels in a string.\npub fn count_vowels(s: &str) -> usize {\n    s.chars().filter(|c| \"aeiouAEIOU\".contains(*c)).count()\n}\n"
                    .to_string(),
            );
        }
        if task.contains("uppercase") || task.contains("to_upper") {
            return Some(
                "/// Convert a string to uppercase.\npub fn to_uppercase(s: &str) -> String {\n    s.to_uppercase()\n}\n"
                    .to_string(),
            );
        }

        // -- Sorting --
        if task.contains("bubble sort") {
            return Some(
                "/// Sort a slice using bubble sort.\npub fn bubble_sort<T: Ord>(arr: &mut [T]) {\n    let n = arr.len();\n    for i in 0..n {\n        let mut swapped = false;\n        for j in 0..n.saturating_sub(i + 1) {\n            if arr[j] > arr[j + 1] {\n                arr.swap(j, j + 1);\n                swapped = true;\n            }\n        }\n        if !swapped { break; }\n    }\n}\n"
                    .to_string(),
            );
        }
        if task.contains("insertion sort") {
            return Some(
                "/// Sort a slice using insertion sort.\npub fn insertion_sort<T: Ord + Copy>(arr: &mut [T]) {\n    for i in 1..arr.len() {\n        let key = arr[i];\n        let mut j = i;\n        while j > 0 && arr[j - 1] > key {\n            arr[j] = arr[j - 1];\n            j -= 1;\n        }\n        arr[j] = key;\n    }\n}\n"
                    .to_string(),
            );
        }
        if task.contains("merge sort") {
            return Some(
                "/// Sort a slice using merge sort.\npub fn merge_sort<T: Ord + Clone>(arr: &mut [T]) {\n    let len = arr.len();\n    if len <= 1 { return; }\n    let mid = len / 2;\n    let mut left = arr[..mid].to_vec();\n    let mut right = arr[mid..].to_vec();\n    merge_sort(&mut left);\n    merge_sort(&mut right);\n    let (mut i, mut j, mut k) = (0, 0, 0);\n    while i < left.len() && j < right.len() {\n        if left[i] <= right[j] { arr[k] = left[i].clone(); i += 1; }\n        else { arr[k] = right[j].clone(); j += 1; }\n        k += 1;\n    }\n    while i < left.len() { arr[k] = left[i].clone(); i += 1; k += 1; }\n    while j < right.len() { arr[k] = right[j].clone(); j += 1; k += 1; }\n}\n"
                    .to_string(),
            );
        }
        if task.contains("sort") {
            return Some(
                "/// Sort a vector and return it.\npub fn sort_vec<T: Ord>(mut v: Vec<T>) -> Vec<T> {\n    v.sort();\n    v\n}\n"
                    .to_string(),
            );
        }

        // -- Searching --
        if task.contains("binary search") {
            return Some(
                "/// Binary search for a value in a sorted slice. Returns the index if found.\npub fn binary_search<T: Ord>(arr: &[T], target: &T) -> Option<usize> {\n    let mut lo = 0usize;\n    let mut hi = arr.len();\n    while lo < hi {\n        let mid = lo + (hi - lo) / 2;\n        match arr[mid].cmp(target) {\n            std::cmp::Ordering::Equal => return Some(mid),\n            std::cmp::Ordering::Less => lo = mid + 1,\n            std::cmp::Ordering::Greater => hi = mid,\n        }\n    }\n    None\n}\n"
                    .to_string(),
            );
        }
        if task.contains("linear search") || (task.contains("search") && task.contains("find")) {
            return Some(
                "/// Linear search for a value in a slice. Returns the index if found.\npub fn linear_search<T: PartialEq>(arr: &[T], target: &T) -> Option<usize> {\n    arr.iter().position(|x| x == target)\n}\n"
                    .to_string(),
            );
        }

        // -- Collections --
        if task.contains("sum")
            && (task.contains("vec") || task.contains("list") || task.contains("array"))
        {
            return Some("/// Sum all elements in a slice.\npub fn sum_vec(v: &[i64]) -> i64 {\n    v.iter().sum()\n}\n".to_string());
        }
        if task.contains("max")
            && (task.contains("vec")
                || task.contains("list")
                || task.contains("array")
                || task.contains("find"))
        {
            return Some("/// Find the maximum value in a slice.\npub fn find_max(v: &[i64]) -> Option<i64> {\n    v.iter().copied().max()\n}\n".to_string());
        }
        if task.contains("min")
            && (task.contains("vec")
                || task.contains("list")
                || task.contains("array")
                || task.contains("find"))
        {
            return Some("/// Find the minimum value in a slice.\npub fn find_min(v: &[i64]) -> Option<i64> {\n    v.iter().copied().min()\n}\n".to_string());
        }
        if task.contains("is_even") || (task.contains("even") && task.contains("check")) {
            return Some("/// Check if a number is even.\npub fn is_even(n: i64) -> bool {\n    n % 2 == 0\n}\n".to_string());
        }
        if task.contains("flatten") {
            return Some("/// Flatten a nested vector.\npub fn flatten<T: Clone>(nested: &[Vec<T>]) -> Vec<T> {\n    nested.iter().flat_map(|v| v.iter().cloned()).collect()\n}\n".to_string());
        }

        // -- Data structures --
        if task.contains("stack") {
            return Some("/// A simple stack backed by a Vec.\npub struct Stack<T> {\n    data: Vec<T>,\n}\n\nimpl<T> Stack<T> {\n    pub fn new() -> Self { Self { data: Vec::new() } }\n    pub fn push(&mut self, val: T) { self.data.push(val); }\n    pub fn pop(&mut self) -> Option<T> { self.data.pop() }\n    pub fn peek(&self) -> Option<&T> { self.data.last() }\n    pub fn is_empty(&self) -> bool { self.data.is_empty() }\n    pub fn len(&self) -> usize { self.data.len() }\n}\n\nimpl<T> Default for Stack<T> {\n    fn default() -> Self { Self::new() }\n}\n".to_string());
        }
        if task.contains("ring buffer") {
            return Some("/// A fixed-capacity ring buffer.\npub struct RingBuffer<T> {\n    buf: Vec<Option<T>>,\n    head: usize,\n    len: usize,\n}\n\nimpl<T> RingBuffer<T> {\n    pub fn new(capacity: usize) -> Self {\n        let mut buf = Vec::with_capacity(capacity);\n        for _ in 0..capacity { buf.push(None); }\n        Self { buf, head: 0, len: 0 }\n    }\n    pub fn push(&mut self, val: T) {\n        let cap = self.buf.len();\n        let idx = (self.head + self.len) % cap;\n        self.buf[idx] = Some(val);\n        if self.len == cap { self.head = (self.head + 1) % cap; }\n        else { self.len += 1; }\n    }\n    pub fn pop(&mut self) -> Option<T> {\n        if self.len == 0 { return None; }\n        let val = self.buf[self.head].take();\n        self.head = (self.head + 1) % self.buf.len();\n        self.len -= 1;\n        val\n    }\n    pub fn len(&self) -> usize { self.len }\n    pub fn is_empty(&self) -> bool { self.len == 0 }\n}\n".to_string());
        }
        if task.contains("linked list") {
            return Some("/// A singly linked list.\npub struct LinkedList<T> {\n    head: Option<Box<Node<T>>>,\n}\n\nstruct Node<T> {\n    value: T,\n    next: Option<Box<Node<T>>>,\n}\n\nimpl<T> LinkedList<T> {\n    pub fn new() -> Self { Self { head: None } }\n    pub fn push_front(&mut self, val: T) {\n        let node = Box::new(Node { value: val, next: self.head.take() });\n        self.head = Some(node);\n    }\n    pub fn pop_front(&mut self) -> Option<T> {\n        let node = self.head.take()?;\n        self.head = node.next;\n        Some(node.value)\n    }\n    pub fn is_empty(&self) -> bool { self.head.is_none() }\n    pub fn len(&self) -> usize {\n        let mut count = 0;\n        let mut current = &self.head;\n        while let Some(node) = current { count += 1; current = &node.next; }\n        count\n    }\n}\n\nimpl<T> Default for LinkedList<T> {\n    fn default() -> Self { Self::new() }\n}\n".to_string());
        }
        if task.contains("bloom filter") {
            return Some("use std::collections::hash_map::DefaultHasher;\nuse std::hash::{Hash, Hasher};\n\n/// A simple Bloom filter.\npub struct BloomFilter {\n    bits: Vec<bool>,\n    num_hashes: usize,\n}\n\nimpl BloomFilter {\n    pub fn new(size: usize, num_hashes: usize) -> Self {\n        Self { bits: vec![false; size], num_hashes }\n    }\n    pub fn insert<T: Hash>(&mut self, item: &T) {\n        for i in 0..self.num_hashes {\n            let idx = self.hash_idx(item, i);\n            self.bits[idx] = true;\n        }\n    }\n    pub fn contains<T: Hash>(&self, item: &T) -> bool {\n        (0..self.num_hashes).all(|i| self.bits[self.hash_idx(item, i)])\n    }\n    fn hash_idx<T: Hash>(&self, item: &T, seed: usize) -> usize {\n        let mut hasher = DefaultHasher::new();\n        item.hash(&mut hasher);\n        seed.hash(&mut hasher);\n        hasher.finish() as usize % self.bits.len()\n    }\n}\n".to_string());
        }

        // The remaining patterns (caesar, rle, pascal, levenshtein, permutations,
        // dfs, dijkstra, matrix multiply, tokenizer, email validator, csv parser,
        // lru cache, state machine, trie, regex, s-expression, convex hull,
        // expression evaluator, http parser, bytecode interpreter, b-tree,
        // n-queens, priority queue, concurrent map, skip list, thread pool)
        // are all handled by the same keyword-matching pattern.
        // For brevity they are included in the full match_native_pattern implementation.

        // -- Medium algorithms and more data structures --
        if task.contains("caesar") || task.contains("cipher") {
            return Some("/// Encrypt text using a Caesar cipher with the given shift.\npub fn encrypt(text: &str, shift: u8) -> String {\n    text.chars().map(|c| {\n        if c.is_ascii_lowercase() {\n            (b'a' + (c as u8 - b'a' + shift) % 26) as char\n        } else if c.is_ascii_uppercase() {\n            (b'A' + (c as u8 - b'A' + shift) % 26) as char\n        } else { c }\n    }).collect()\n}\n\n/// Decrypt text using a Caesar cipher with the given shift.\npub fn decrypt(text: &str, shift: u8) -> String {\n    encrypt(text, 26 - (shift % 26))\n}\n".to_string());
        }
        if task.contains("run-length") || task.contains("rle") {
            return Some("/// Run-length encode a string: \"aaabbc\" -> \"3a2b1c\".\npub fn rle_encode(s: &str) -> String {\n    if s.is_empty() { return String::new(); }\n    let mut result = String::new();\n    let chars: Vec<char> = s.chars().collect();\n    let mut count = 1usize;\n    for i in 1..chars.len() {\n        if chars[i] == chars[i - 1] { count += 1; }\n        else {\n            result.push_str(&count.to_string());\n            result.push(chars[i - 1]);\n            count = 1;\n        }\n    }\n    result.push_str(&count.to_string());\n    result.push(*chars.last().unwrap());\n    result\n}\n".to_string());
        }
        if task.contains("pascal") && task.contains("triangle") {
            return Some("/// Generate Pascal's triangle with n rows.\npub fn pascal_triangle(n: usize) -> Vec<Vec<u64>> {\n    let mut tri: Vec<Vec<u64>> = Vec::with_capacity(n);\n    for i in 0..n {\n        let mut row = vec![1u64; i + 1];\n        for j in 1..i {\n            row[j] = tri[i - 1][j - 1] + tri[i - 1][j];\n        }\n        tri.push(row);\n    }\n    tri\n}\n".to_string());
        }
        if task.contains("levenshtein") || task.contains("edit distance") {
            return Some("/// Compute the Levenshtein (edit) distance between two strings.\npub fn levenshtein(a: &str, b: &str) -> usize {\n    let a: Vec<char> = a.chars().collect();\n    let b: Vec<char> = b.chars().collect();\n    let (m, n) = (a.len(), b.len());\n    let mut dp = vec![vec![0usize; n + 1]; m + 1];\n    for i in 0..=m { dp[i][0] = i; }\n    for j in 0..=n { dp[0][j] = j; }\n    for i in 1..=m {\n        for j in 1..=n {\n            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };\n            dp[i][j] = (dp[i - 1][j] + 1)\n                .min(dp[i][j - 1] + 1)\n                .min(dp[i - 1][j - 1] + cost);\n        }\n    }\n    dp[m][n]\n}\n".to_string());
        }
        if task.contains("permutation") {
            return Some("/// Generate all permutations of a string.\npub fn permutations(s: &str) -> Vec<String> {\n    let chars: Vec<char> = s.chars().collect();\n    let mut results = Vec::new();\n    let mut current = chars.clone();\n    permute(&mut current, 0, &mut results);\n    results\n}\n\nfn permute(chars: &mut Vec<char>, start: usize, results: &mut Vec<String>) {\n    if start == chars.len() {\n        results.push(chars.iter().collect());\n        return;\n    }\n    for i in start..chars.len() {\n        chars.swap(start, i);\n        permute(chars, start + 1, results);\n        chars.swap(start, i);\n    }\n}\n".to_string());
        }
        if task.contains("depth-first") || task.contains("dfs") {
            return Some("use std::collections::HashSet;\n\n/// Perform depth-first search on an adjacency list graph.\n/// Returns visited nodes in DFS order.\npub fn dfs(graph: &[Vec<usize>], start: usize) -> Vec<usize> {\n    let mut visited = HashSet::new();\n    let mut order = Vec::new();\n    dfs_visit(graph, start, &mut visited, &mut order);\n    order\n}\n\nfn dfs_visit(graph: &[Vec<usize>], node: usize, visited: &mut HashSet<usize>, order: &mut Vec<usize>) {\n    if !visited.insert(node) { return; }\n    order.push(node);\n    if node < graph.len() {\n        for &neighbor in &graph[node] {\n            dfs_visit(graph, neighbor, visited, order);\n        }\n    }\n}\n".to_string());
        }
        if task.contains("dijkstra") {
            return Some("use std::collections::BinaryHeap;\nuse std::cmp::Reverse;\n\n/// Dijkstra's shortest path from `start` on a weighted adjacency list.\n/// Returns distances to all nodes (usize::MAX = unreachable).\npub fn dijkstra(graph: &[Vec<(usize, u64)>], start: usize) -> Vec<u64> {\n    let n = graph.len();\n    let mut dist = vec![u64::MAX; n];\n    dist[start] = 0;\n    let mut heap = BinaryHeap::new();\n    heap.push(Reverse((0u64, start)));\n    while let Some(Reverse((d, u))) = heap.pop() {\n        if d > dist[u] { continue; }\n        for &(v, w) in &graph[u] {\n            let nd = d + w;\n            if nd < dist[v] {\n                dist[v] = nd;\n                heap.push(Reverse((nd, v)));\n            }\n        }\n    }\n    dist\n}\n".to_string());
        }
        if task.contains("matrix") && task.contains("multiply") {
            return Some("/// Multiply two matrices (Vec of rows).\npub fn multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {\n    let m = a.len();\n    let n = b[0].len();\n    let k = b.len();\n    let mut result = vec![vec![0.0f64; n]; m];\n    for i in 0..m {\n        for j in 0..n {\n            for p in 0..k {\n                result[i][j] += a[i][p] * b[p][j];\n            }\n        }\n    }\n    result\n}\n".to_string());
        }
        if task.contains("tokenize") || task.contains("tokenizer") {
            return Some("/// Token types for arithmetic expressions.\n#[derive(Debug, Clone, PartialEq)]\npub enum Token {\n    Number(f64),\n    Plus,\n    Minus,\n    Star,\n    Slash,\n    LParen,\n    RParen,\n}\n\n/// Tokenize an arithmetic expression string.\npub fn tokenize(input: &str) -> Vec<Token> {\n    let mut tokens = Vec::new();\n    let mut chars = input.chars().peekable();\n    while let Some(&c) = chars.peek() {\n        match c {\n            ' ' | '\\t' => { chars.next(); }\n            '+' => { tokens.push(Token::Plus); chars.next(); }\n            '-' => { tokens.push(Token::Minus); chars.next(); }\n            '*' => { tokens.push(Token::Star); chars.next(); }\n            '/' => { tokens.push(Token::Slash); chars.next(); }\n            '(' => { tokens.push(Token::LParen); chars.next(); }\n            ')' => { tokens.push(Token::RParen); chars.next(); }\n            '0'..='9' | '.' => {\n                let mut num = String::new();\n                while let Some(&d) = chars.peek() {\n                    if d.is_ascii_digit() || d == '.' { num.push(d); chars.next(); }\n                    else { break; }\n                }\n                if let Ok(n) = num.parse::<f64>() { tokens.push(Token::Number(n)); }\n            }\n            _ => { chars.next(); }\n        }\n    }\n    tokens\n}\n".to_string());
        }
        if task.contains("email") && task.contains("validat") {
            return Some("/// Validate an email address (basic RFC 5321 check).\npub fn validate_email(email: &str) -> bool {\n    let parts: Vec<&str> = email.splitn(2, '@').collect();\n    if parts.len() != 2 { return false; }\n    let (local, domain) = (parts[0], parts[1]);\n    if local.is_empty() || local.len() > 64 { return false; }\n    if domain.is_empty() || domain.len() > 255 { return false; }\n    if !domain.contains('.') { return false; }\n    let valid_char = |c: char| c.is_alphanumeric() || \".-_+\".contains(c);\n    local.chars().all(valid_char) && domain.chars().all(|c| c.is_alphanumeric() || \".-\".contains(c))\n}\n".to_string());
        }
        if task.contains("csv") && task.contains("parse") {
            return Some("/// Parse a CSV line into fields, respecting quoted strings.\npub fn parse_csv(line: &str) -> Vec<String> {\n    let mut fields = Vec::new();\n    let mut current = String::new();\n    let mut in_quotes = false;\n    let mut chars = line.chars().peekable();\n    while let Some(c) = chars.next() {\n        match c {\n            '\"' => {\n                if in_quotes {\n                    if chars.peek() == Some(&'\"') { current.push('\"'); chars.next(); }\n                    else { in_quotes = false; }\n                } else { in_quotes = true; }\n            }\n            ',' if !in_quotes => {\n                fields.push(current.clone());\n                current.clear();\n            }\n            _ => current.push(c),\n        }\n    }\n    fields.push(current);\n    fields\n}\n".to_string());
        }
        if task.contains("lru") && task.contains("cache") {
            return Some("use std::collections::HashMap;\n\n/// A simple LRU cache with fixed capacity.\npub struct LruCache<K: std::hash::Hash + Eq + Clone, V> {\n    capacity: usize,\n    order: Vec<K>,\n    map: HashMap<K, V>,\n}\n\nimpl<K: std::hash::Hash + Eq + Clone, V> LruCache<K, V> {\n    pub fn new(capacity: usize) -> Self {\n        Self { capacity, order: Vec::new(), map: HashMap::new() }\n    }\n    pub fn get(&mut self, key: &K) -> Option<&V> {\n        if self.map.contains_key(key) {\n            self.order.retain(|k| k != key);\n            self.order.push(key.clone());\n            self.map.get(key)\n        } else { None }\n    }\n    pub fn put(&mut self, key: K, value: V) {\n        if self.map.contains_key(&key) {\n            self.order.retain(|k| k != &key);\n        } else if self.map.len() >= self.capacity {\n            if let Some(oldest) = self.order.first().cloned() {\n                self.order.remove(0);\n                self.map.remove(&oldest);\n            }\n        }\n        self.order.push(key.clone());\n        self.map.insert(key, value);\n    }\n    pub fn len(&self) -> usize { self.map.len() }\n    pub fn is_empty(&self) -> bool { self.map.is_empty() }\n}\n".to_string());
        }
        // Additional patterns: state machine, trie, regex, s-expression, convex hull,
        // expression evaluator, http parser, bytecode interpreter, b-tree, n-queens,
        // priority queue, concurrent map, skip list, thread pool
        // These are intentionally omitted for brevity — they follow the same pattern.
        // The original implementations are preserved in the full match.

        if task.contains("state machine") {
            return Some("use std::collections::HashMap;\n\n/// A simple finite state machine.\npub struct StateMachine {\n    current: String,\n    transitions: HashMap<(String, String), String>,\n}\n\nimpl StateMachine {\n    pub fn new(initial: &str) -> Self {\n        Self { current: initial.to_string(), transitions: HashMap::new() }\n    }\n    pub fn add_transition(&mut self, from: &str, event: &str, to: &str) {\n        self.transitions.insert((from.to_string(), event.to_string()), to.to_string());\n    }\n    pub fn send(&mut self, event: &str) -> bool {\n        let key = (self.current.clone(), event.to_string());\n        if let Some(next) = self.transitions.get(&key) {\n            self.current = next.clone();\n            true\n        } else { false }\n    }\n    pub fn state(&self) -> &str { &self.current }\n}\n".to_string());
        }
        if task.contains("trie") {
            return Some("use std::collections::HashMap;\n\n/// A trie (prefix tree) for string storage and lookup.\npub struct Trie {\n    children: HashMap<char, Trie>,\n    is_end: bool,\n}\n\nimpl Trie {\n    pub fn new() -> Self { Self { children: HashMap::new(), is_end: false } }\n    pub fn insert(&mut self, word: &str) {\n        let mut node = self;\n        for c in word.chars() {\n            node = node.children.entry(c).or_insert_with(Trie::new);\n        }\n        node.is_end = true;\n    }\n    pub fn search(&self, word: &str) -> bool {\n        let mut node = self;\n        for c in word.chars() {\n            match node.children.get(&c) {\n                Some(next) => node = next,\n                None => return false,\n            }\n        }\n        node.is_end\n    }\n    pub fn starts_with(&self, prefix: &str) -> bool {\n        let mut node = self;\n        for c in prefix.chars() {\n            match node.children.get(&c) {\n                Some(next) => node = next,\n                None => return false,\n            }\n        }\n        true\n    }\n}\n\nimpl Default for Trie {\n    fn default() -> Self { Self::new() }\n}\n".to_string());
        }
        if task.contains("regex") && task.contains("match") {
            return Some("/// Match a simple regex pattern supporting '.' (any char) and '*' (zero or more of previous).\npub fn regex_match(text: &str, pattern: &str) -> bool {\n    let t: Vec<char> = text.chars().collect();\n    let p: Vec<char> = pattern.chars().collect();\n    let (m, n) = (t.len(), p.len());\n    let mut dp = vec![vec![false; n + 1]; m + 1];\n    dp[0][0] = true;\n    for j in 1..=n {\n        if p[j - 1] == '*' && j >= 2 { dp[0][j] = dp[0][j - 2]; }\n    }\n    for i in 1..=m {\n        for j in 1..=n {\n            if p[j - 1] == '.' || p[j - 1] == t[i - 1] {\n                dp[i][j] = dp[i - 1][j - 1];\n            } else if p[j - 1] == '*' && j >= 2 {\n                dp[i][j] = dp[i][j - 2];\n                if p[j - 2] == '.' || p[j - 2] == t[i - 1] {\n                    dp[i][j] = dp[i][j] || dp[i - 1][j];\n                }\n            }\n        }\n    }\n    dp[m][n]\n}\n".to_string());
        }
        if task.contains("s-expression") || task.contains("sexp") || task.contains("s expression") {
            return Some("/// A simple S-expression tree.\n#[derive(Debug, Clone, PartialEq)]\npub enum Sexp {\n    Atom(String),\n    List(Vec<Sexp>),\n}\n\n/// Parse an S-expression string into a tree.\npub fn parse_sexp(input: &str) -> Result<Sexp, String> {\n    let tokens = tokenize_sexp(input);\n    let mut pos = 0;\n    let result = parse_tokens(&tokens, &mut pos)?;\n    Ok(result)\n}\n\nfn tokenize_sexp(input: &str) -> Vec<String> {\n    let mut tokens = Vec::new();\n    let mut current = String::new();\n    for c in input.chars() {\n        match c {\n            '(' | ')' => {\n                if !current.is_empty() { tokens.push(current.clone()); current.clear(); }\n                tokens.push(c.to_string());\n            }\n            ' ' | '\\t' | '\\n' | '\\r' => {\n                if !current.is_empty() { tokens.push(current.clone()); current.clear(); }\n            }\n            _ => current.push(c),\n        }\n    }\n    if !current.is_empty() { tokens.push(current); }\n    tokens\n}\n\nfn parse_tokens(tokens: &[String], pos: &mut usize) -> Result<Sexp, String> {\n    if *pos >= tokens.len() { return Err(\"unexpected end of input\".into()); }\n    if tokens[*pos] == \"(\" {\n        *pos += 1;\n        let mut list = Vec::new();\n        while *pos < tokens.len() && tokens[*pos] != \")\" {\n            list.push(parse_tokens(tokens, pos)?);\n        }\n        if *pos >= tokens.len() { return Err(\"missing closing parenthesis\".into()); }\n        *pos += 1;\n        Ok(Sexp::List(list))\n    } else if tokens[*pos] == \")\" {\n        Err(\"unexpected )\".into())\n    } else {\n        let atom = tokens[*pos].clone();\n        *pos += 1;\n        Ok(Sexp::Atom(atom))\n    }\n}\n".to_string());
        }
        if task.contains("convex hull") || task.contains("convex_hull") {
            return Some("/// A 2D point.\n#[derive(Debug, Clone, Copy, PartialEq)]\npub struct Point {\n    pub x: f64,\n    pub y: f64,\n}\n\n/// Compute the convex hull of a set of points using Graham scan.\npub fn convex_hull(mut points: Vec<Point>) -> Vec<Point> {\n    let n = points.len();\n    if n < 3 { return points; }\n    let mut min_idx = 0;\n    for i in 1..n {\n        if points[i].y < points[min_idx].y || (points[i].y == points[min_idx].y && points[i].x < points[min_idx].x) {\n            min_idx = i;\n        }\n    }\n    points.swap(0, min_idx);\n    let pivot = points[0];\n    points[1..].sort_by(|a, b| {\n        let ca = cross(pivot, *a, *b);\n        if ca == 0.0 { dist(pivot, *a).partial_cmp(&dist(pivot, *b)).unwrap() }\n        else if ca > 0.0 { std::cmp::Ordering::Less }\n        else { std::cmp::Ordering::Greater }\n    });\n    let mut hull: Vec<Point> = Vec::new();\n    for p in &points {\n        while hull.len() > 1 && cross(hull[hull.len() - 2], hull[hull.len() - 1], *p) <= 0.0 {\n            hull.pop();\n        }\n        hull.push(*p);\n    }\n    hull\n}\n\nfn cross(o: Point, a: Point, b: Point) -> f64 {\n    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)\n}\n\nfn dist(a: Point, b: Point) -> f64 {\n    (a.x - b.x).powi(2) + (a.y - b.y).powi(2)\n}\n".to_string());
        }
        if task.contains("evaluat") && (task.contains("expression") || task.contains("arithmetic"))
        {
            return Some("/// Evaluate an arithmetic expression with +, -, *, /, and parentheses.\npub fn evaluate(expr: &str) -> f64 {\n    let mut pos = 0;\n    let chars: Vec<char> = expr.chars().filter(|c| !c.is_whitespace()).collect();\n    parse_expr(&chars, &mut pos)\n}\n\nfn parse_expr(chars: &[char], pos: &mut usize) -> f64 {\n    let mut result = parse_term(chars, pos);\n    while *pos < chars.len() && (chars[*pos] == '+' || chars[*pos] == '-') {\n        let op = chars[*pos];\n        *pos += 1;\n        let right = parse_term(chars, pos);\n        result = if op == '+' { result + right } else { result - right };\n    }\n    result\n}\n\nfn parse_term(chars: &[char], pos: &mut usize) -> f64 {\n    let mut result = parse_factor(chars, pos);\n    while *pos < chars.len() && (chars[*pos] == '*' || chars[*pos] == '/') {\n        let op = chars[*pos];\n        *pos += 1;\n        let right = parse_factor(chars, pos);\n        result = if op == '*' { result * right } else { result / right };\n    }\n    result\n}\n\nfn parse_factor(chars: &[char], pos: &mut usize) -> f64 {\n    if *pos < chars.len() && chars[*pos] == '(' {\n        *pos += 1;\n        let result = parse_expr(chars, pos);\n        if *pos < chars.len() && chars[*pos] == ')' { *pos += 1; }\n        return result;\n    }\n    let start = *pos;\n    if *pos < chars.len() && chars[*pos] == '-' { *pos += 1; }\n    while *pos < chars.len() && (chars[*pos].is_ascii_digit() || chars[*pos] == '.') {\n        *pos += 1;\n    }\n    let s: String = chars[start..*pos].iter().collect();\n    s.parse::<f64>().unwrap_or(0.0)\n}\n".to_string());
        }
        if task.contains("http") && task.contains("pars") {
            return Some("/// A parsed HTTP request.\n#[derive(Debug, Clone)]\npub struct HttpRequest {\n    pub method: String,\n    pub path: String,\n    pub version: String,\n    pub headers: Vec<(String, String)>,\n    pub body: String,\n}\n\n/// Parse a raw HTTP request string.\npub fn parse_request(raw: &str) -> Result<HttpRequest, String> {\n    let mut parts = raw.splitn(2, \"\\r\\n\\r\\n\");\n    let header_section = parts.next().ok_or(\"empty request\")?;\n    let body = parts.next().unwrap_or(\"\").to_string();\n    let mut lines = header_section.lines();\n    let request_line = lines.next().ok_or(\"missing request line\")?;\n    let mut rl_parts = request_line.splitn(3, ' ');\n    let method = rl_parts.next().ok_or(\"missing method\")?.to_string();\n    let path = rl_parts.next().ok_or(\"missing path\")?.to_string();\n    let version = rl_parts.next().unwrap_or(\"HTTP/1.1\").to_string();\n    let mut headers = Vec::new();\n    for line in lines {\n        if line.is_empty() { break; }\n        if let Some((key, val)) = line.split_once(':') {\n            headers.push((key.trim().to_string(), val.trim().to_string()));\n        }\n    }\n    Ok(HttpRequest { method, path, version, headers, body })\n}\n".to_string());
        }
        if task.contains("bytecode") && task.contains("interpret") {
            return Some("/// Bytecode opcodes for a simple stack machine.\n#[derive(Debug, Clone, Copy)]\npub enum Op {\n    Push(i64),\n    Add,\n    Sub,\n    Mul,\n    Div,\n    Dup,\n    Pop,\n    Halt,\n}\n\n/// Interpret a bytecode program on a stack machine. Returns the top of stack.\npub fn interpret(program: &[Op]) -> i64 {\n    let mut stack: Vec<i64> = Vec::new();\n    let mut pc = 0;\n    while pc < program.len() {\n        match program[pc] {\n            Op::Push(v) => stack.push(v),\n            Op::Add => { let b = stack.pop().unwrap_or(0); let a = stack.pop().unwrap_or(0); stack.push(a + b); }\n            Op::Sub => { let b = stack.pop().unwrap_or(0); let a = stack.pop().unwrap_or(0); stack.push(a - b); }\n            Op::Mul => { let b = stack.pop().unwrap_or(0); let a = stack.pop().unwrap_or(0); stack.push(a * b); }\n            Op::Div => { let b = stack.pop().unwrap_or(0); let a = stack.pop().unwrap_or(0); if b != 0 { stack.push(a / b); } else { stack.push(0); } }\n            Op::Dup => { if let Some(&top) = stack.last() { stack.push(top); } }\n            Op::Pop => { stack.pop(); }\n            Op::Halt => break,\n        }\n        pc += 1;\n    }\n    stack.pop().unwrap_or(0)\n}\n".to_string());
        }
        if task.contains("b-tree") || (task.contains("btree") && task.contains("insert")) {
            return Some("/// A simple B-tree node.\n#[derive(Debug, Clone)]\npub struct BTreeNode {\n    keys: Vec<i64>,\n    children: Vec<BTreeNode>,\n    leaf: bool,\n    t: usize,\n}\n\nimpl BTreeNode {\n    pub fn new(t: usize, leaf: bool) -> Self {\n        Self { keys: Vec::new(), children: Vec::new(), leaf, t }\n    }\n}\n\n/// A B-tree with minimum degree `t`.\n#[derive(Debug, Clone)]\npub struct BTree {\n    root: BTreeNode,\n    t: usize,\n}\n\nimpl BTree {\n    pub fn new(t: usize) -> Self { Self { root: BTreeNode::new(t, true), t } }\n    pub fn insert(&mut self, key: i64) {\n        let t = self.t;\n        if self.root.keys.len() == 2 * t - 1 {\n            let mut new_root = BTreeNode::new(t, false);\n            let old_root = std::mem::replace(&mut self.root, BTreeNode::new(t, true));\n            new_root.children.push(old_root);\n            Self::split_child(&mut new_root, 0, t);\n            Self::insert_non_full(&mut new_root, key, t);\n            self.root = new_root;\n        } else { Self::insert_non_full(&mut self.root, key, t); }\n    }\n    fn split_child(parent: &mut BTreeNode, i: usize, t: usize) {\n        let full = &mut parent.children[i];\n        let mut right = BTreeNode::new(t, full.leaf);\n        right.keys = full.keys.split_off(t);\n        let median = full.keys.pop().unwrap();\n        if !full.leaf { right.children = full.children.split_off(t); }\n        parent.keys.insert(i, median);\n        parent.children.insert(i + 1, right);\n    }\n    fn insert_non_full(node: &mut BTreeNode, key: i64, t: usize) {\n        if node.leaf {\n            let pos = node.keys.iter().position(|&k| k > key).unwrap_or(node.keys.len());\n            node.keys.insert(pos, key);\n        } else {\n            let mut i = node.keys.iter().position(|&k| k > key).unwrap_or(node.keys.len());\n            if node.children[i].keys.len() == 2 * t - 1 {\n                Self::split_child(node, i, t);\n                if key > node.keys[i] { i += 1; }\n            }\n            Self::insert_non_full(&mut node.children[i], key, t);\n        }\n    }\n    pub fn search(&self, key: i64) -> bool { Self::search_node(&self.root, key) }\n    fn search_node(node: &BTreeNode, key: i64) -> bool {\n        match node.keys.binary_search(&key) {\n            Ok(_) => true,\n            Err(i) => if node.leaf { false } else { Self::search_node(&node.children[i], key) }\n        }\n    }\n}\n".to_string());
        }
        if task.contains("n-queen") || task.contains("queens") {
            return Some("/// Solve the N-Queens problem. Returns all valid board configurations.\npub fn solve_queens(n: usize) -> Vec<Vec<usize>> {\n    let mut solutions = Vec::new();\n    let mut board = Vec::with_capacity(n);\n    solve_queens_bt(n, &mut board, &mut solutions);\n    solutions\n}\n\nfn solve_queens_bt(n: usize, board: &mut Vec<usize>, solutions: &mut Vec<Vec<usize>>) {\n    if board.len() == n { solutions.push(board.clone()); return; }\n    let row = board.len();\n    for col in 0..n {\n        if board.iter().enumerate().all(|(r, &c)| c != col && (row - r) != col.abs_diff(c)) {\n            board.push(col);\n            solve_queens_bt(n, board, solutions);\n            board.pop();\n        }\n    }\n}\n".to_string());
        }
        if task.contains("task queue") || task.contains("priority queue") {
            return Some("use std::collections::BinaryHeap;\nuse std::cmp::Reverse;\n\n/// A priority task queue where lower priority number = higher urgency.\n#[derive(Debug)]\npub struct TaskQueue<T> {\n    heap: BinaryHeap<Reverse<(u32, usize, T)>>,\n    counter: usize,\n}\n\nimpl<T: Ord> TaskQueue<T> {\n    pub fn new() -> Self { Self { heap: BinaryHeap::new(), counter: 0 } }\n    pub fn push(&mut self, priority: u32, task: T) {\n        self.heap.push(Reverse((priority, self.counter, task)));\n        self.counter += 1;\n    }\n    pub fn pop(&mut self) -> Option<T> { self.heap.pop().map(|Reverse((_, _, task))| task) }\n    pub fn peek(&self) -> Option<&T> { self.heap.peek().map(|Reverse((_, _, task))| task) }\n    pub fn len(&self) -> usize { self.heap.len() }\n    pub fn is_empty(&self) -> bool { self.heap.is_empty() }\n}\n\nimpl<T: Ord> Default for TaskQueue<T> {\n    fn default() -> Self { Self::new() }\n}\n".to_string());
        }
        if task.contains("concurrent") && (task.contains("map") || task.contains("hashmap")) {
            return Some("use std::collections::HashMap;\nuse std::sync::Mutex;\n\n/// A concurrent hashmap using shard-level locking.\npub struct ConcurrentMap<K, V> {\n    shards: Vec<Mutex<HashMap<K, V>>>,\n    num_shards: usize,\n}\n\nimpl<K: std::hash::Hash + Eq + Clone, V: Clone> ConcurrentMap<K, V> {\n    pub fn new(num_shards: usize) -> Self {\n        let shards = (0..num_shards).map(|_| Mutex::new(HashMap::new())).collect();\n        Self { shards, num_shards }\n    }\n    fn shard_idx(&self, key: &K) -> usize {\n        use std::hash::{Hash, Hasher};\n        let mut hasher = std::collections::hash_map::DefaultHasher::new();\n        key.hash(&mut hasher);\n        hasher.finish() as usize % self.num_shards\n    }\n    pub fn insert(&self, key: K, value: V) { let idx = self.shard_idx(&key); self.shards[idx].lock().unwrap().insert(key, value); }\n    pub fn get(&self, key: &K) -> Option<V> { let idx = self.shard_idx(key); self.shards[idx].lock().unwrap().get(key).cloned() }\n    pub fn remove(&self, key: &K) -> Option<V> { let idx = self.shard_idx(key); self.shards[idx].lock().unwrap().remove(key) }\n    pub fn len(&self) -> usize { self.shards.iter().map(|s| s.lock().unwrap().len()).sum() }\n    pub fn is_empty(&self) -> bool { self.len() == 0 }\n}\n".to_string());
        }
        if task.contains("skip list") || task.contains("skiplist") {
            return Some("use rand::Rng;\n\nconst MAX_LEVEL: usize = 16;\n\n#[derive(Debug)]\nstruct SkipNode { value: i64, forward: Vec<Option<usize>> }\n\n#[derive(Debug)]\npub struct SkipList { nodes: Vec<SkipNode>, head: usize, level: usize }\n\nimpl SkipList {\n    pub fn new() -> Self { let head_node = SkipNode { value: i64::MIN, forward: vec![None; MAX_LEVEL] }; Self { nodes: vec![head_node], head: 0, level: 0 } }\n    fn random_level() -> usize { let mut lvl = 0; let mut rng = rand::rng(); while rng.random::<bool>() && lvl < MAX_LEVEL - 1 { lvl += 1; } lvl }\n    pub fn insert(&mut self, value: i64) {\n        let new_level = Self::random_level();\n        if new_level > self.level { self.level = new_level; }\n        let new_idx = self.nodes.len();\n        self.nodes.push(SkipNode { value, forward: vec![None; new_level + 1] });\n        let mut current = self.head;\n        for lvl in (0..=self.level).rev() {\n            while let Some(next) = self.nodes[current].forward[lvl] {\n                if self.nodes[next].value < value { current = next; } else { break; }\n            }\n            if lvl <= new_level {\n                self.nodes[new_idx].forward[lvl] = self.nodes[current].forward[lvl];\n                self.nodes[current].forward[lvl] = Some(new_idx);\n            }\n        }\n    }\n    pub fn search(&self, value: i64) -> bool {\n        let mut current = self.head;\n        for lvl in (0..=self.level).rev() {\n            while let Some(next) = self.nodes[current].forward[lvl] {\n                if self.nodes[next].value < value { current = next; }\n                else if self.nodes[next].value == value { return true; }\n                else { break; }\n            }\n        }\n        false\n    }\n    pub fn len(&self) -> usize { self.nodes.len() - 1 }\n    pub fn is_empty(&self) -> bool { self.len() == 0 }\n}\nimpl Default for SkipList { fn default() -> Self { Self::new() } }\n".to_string());
        }
        if task.contains("thread pool") || task.contains("threadpool") {
            return Some("use std::sync::{mpsc, Arc, Mutex};\nuse std::thread;\n\ntype Job = Box<dyn FnOnce() + Send + 'static>;\n\npub struct ThreadPool {\n    workers: Vec<thread::JoinHandle<()>>,\n    sender: Option<mpsc::Sender<Job>>,\n}\n\nimpl ThreadPool {\n    pub fn new(size: usize) -> Self {\n        let (sender, receiver) = mpsc::channel::<Job>();\n        let receiver = Arc::new(Mutex::new(receiver));\n        let mut workers = Vec::with_capacity(size);\n        for _ in 0..size {\n            let rx = Arc::clone(&receiver);\n            workers.push(thread::spawn(move || { loop { match rx.lock().unwrap().recv() { Ok(f) => f(), Err(_) => break } } }));\n        }\n        Self { workers, sender: Some(sender) }\n    }\n    pub fn execute<F: FnOnce() + Send + 'static>(&self, f: F) {\n        if let Some(ref sender) = self.sender { sender.send(Box::new(f)).unwrap(); }\n    }\n}\n\nimpl Drop for ThreadPool {\n    fn drop(&mut self) { drop(self.sender.take()); for worker in self.workers.drain(..) { let _ = worker.join(); } }\n}\n".to_string());
        }

        None
    }

    /// Extract a likely function name from a task description.
    pub(super) fn extract_function_name(task: &str) -> Option<String> {
        // Prefer an explicit function declaration already present in the task text
        // (e.g. HumanEval-style prompts embed the real `def name(...)`/`fn name(...)`
        // signature) over the prose heuristics below. Those heuristics can misfire
        // when a docstring's prose happens to contain "function " as plain English
        // ("...to this function is a string...") rather than a declaration keyword —
        // that specific collision previously extracted "is" as a function name.
        for marker in ["def ", "fn "] {
            let mut search_from = 0;
            while let Some(rel_idx) = task[search_from..].find(marker) {
                let idx = search_from + rel_idx;
                let after = &task[idx + marker.len()..];
                let name: String = after
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                let rest = after[name.len()..].trim_start();
                if !name.is_empty() && rest.starts_with('(') {
                    return Some(name.to_lowercase());
                }
                search_from = idx + marker.len();
            }
        }

        let stop_words = [
            "a", "an", "the", "my", "our", "new", "simple", "basic", "is", "are", "was", "were",
        ];
        // Look for explicit function names: "add a X function", "implement X", "create X"
        let patterns = [
            "function ",
            "fn ",
            "method ",
            "implement ",
            "create ",
            "add ",
        ];
        for pattern in &patterns {
            if let Some(idx) = task.find(pattern) {
                let after = &task[idx + pattern.len()..];
                // Skip stop words (e.g., "add a fibonacci" -> skip "a")
                let name: String = after
                    .split_whitespace()
                    .find(|w| !stop_words.contains(&w.to_lowercase().as_str()))
                    .unwrap_or("")
                    .chars()
                    .filter(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty() && name.len() < 50 {
                    return Some(name.to_lowercase());
                }
            }
        }

        // Fallback: use first multi-char word that looks like an identifier
        for word in task.split_whitespace() {
            let clean: String = word
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if clean.len() >= 3
                && clean.chars().next().map_or(false, |c| c.is_alphabetic())
                && ![
                    "the", "and", "for", "add", "that", "with", "from", "this", "into",
                ]
                .contains(&clean.to_lowercase().as_str())
            {
                return Some(clean.to_lowercase());
            }
        }
        None
    }

    /// Build HDC context prompt from indexed codebase memory.
    #[cfg(feature = "code_generation")]
    pub(super) fn build_hdc_context_prompt(&self) -> String {
        use crate::hdc::code_encoder::CodeHDEncoder;

        let code_memory = match &self.code_memory {
            Some(m) => m,
            None => return String::new(),
        };

        let encoder = CodeHDEncoder::new(16_384);
        let query_hv = encoder.encode_name(&self.task);
        let matches = code_memory.query(&query_hv, 5);

        if matches.is_empty() {
            return String::new();
        }

        let coherence = code_memory.codebase_coherence();
        let mut prompt = format!(
            "## Codebase context (HDC similarity search, coherence={:.2})\n",
            coherence
        );
        for m in &matches {
            prompt.push_str(&format!("- {} (similarity={:.3})\n", m.name, m.similarity));
            if let Some(src) = self.source_cache.get(&m.path) {
                let snippet = Self::extract_entity_source(src, &m.name, m.kind);
                if !snippet.contains("(source not found)") {
                    let truncated: String = snippet.chars().take(200).collect();
                    prompt.push_str(&format!("  ```\n  {}\n  ```\n", truncated));
                }
            }
        }
        prompt
    }

    #[cfg(not(feature = "code_generation"))]
    pub(super) fn build_hdc_context_prompt(&self) -> String {
        String::new()
    }

    /// Dynamically query CodebaseMemory for context relevant to current errors.
    #[cfg(feature = "code_generation")]
    pub(super) fn build_dynamic_error_context(&self) -> String {
        let code_memory = match (&self.phase, &self.code_memory) {
            (TaskPhase::Fixing, Some(m)) => m,
            _ => return String::new(),
        };
        let test_output = match &self.last_test_output {
            Some(o) => o,
            None => return String::new(),
        };

        let mut query_terms: Vec<String> = Vec::new();
        for line in test_output.lines() {
            let mut rest = line;
            while let Some(start) = rest.find('`') {
                let after = &rest[start + 1..];
                if let Some(end) = after.find('`') {
                    let name = &after[..end];
                    let clean = name.trim_end_matches("()");
                    if !clean.is_empty()
                        && clean.len() < 60
                        && !clean.contains(' ')
                        && !clean.starts_with("error")
                        && !clean.starts_with("help")
                    {
                        if !query_terms.iter().any(|t| t == clean) {
                            query_terms.push(clean.to_string());
                        }
                    }
                    rest = &after[end + 1..];
                } else {
                    break;
                }
            }
        }

        if query_terms.is_empty() {
            return String::new();
        }

        let encoder = code_memory.encoder();
        let mut seen_names = std::collections::HashSet::new();
        let mut result = String::new();

        for term in query_terms.iter().take(5) {
            let query_hv = encoder.encode_name(term);
            let matches = code_memory.query(&query_hv, 3);

            for m in &matches {
                if m.similarity < 0.25 || !seen_names.insert(m.name.clone()) {
                    continue;
                }
                if let Some(src) = self.source_cache.get(&m.path) {
                    let snippet = Self::extract_entity_source(src, &m.name, m.kind);
                    if !snippet.contains("(source not found)") {
                        let truncated: String = snippet.chars().take(300).collect();
                        result.push_str(&format!(
                            "- `{}` ({:?}, {}): ```\n{}\n```\n",
                            m.name,
                            m.kind,
                            m.path.display(),
                            truncated
                        ));
                    }
                } else {
                    result.push_str(&format!(
                        "- `{}` ({:?}, {}, sim={:.2})\n",
                        m.name,
                        m.kind,
                        m.path.display(),
                        m.similarity
                    ));
                }
            }
        }

        result
    }

    #[cfg(not(feature = "code_generation"))]
    pub(super) fn build_dynamic_error_context(&self) -> String {
        String::new()
    }

    /// Build a language-appropriate system prompt for the LLM.
    pub(super) fn codegen_system_prompt(&self) -> String {
        match self.target_language() {
            "python" => "You are a Python code generator. Output ONLY valid Python code. \
                No explanations, no markdown fences, no comments outside the function. \
                Complete the function body directly."
                .into(),
            "nix" => "You are a Nix code generator. Output ONLY valid Nix expressions.".into(),
            _ => "You are a code generator. Output ONLY valid source code, no explanations.".into(),
        }
    }
}
