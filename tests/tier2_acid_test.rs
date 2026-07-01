#![cfg(feature = "code_generation")]

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================================
// Tier 2 Acid Test — Novel Algorithm Synthesis via Ollama
// ==================================================================================
//
// Three algorithmic challenges that are GUARANTEED to bypass Symthaea's 80+ hardcoded
// templates and force the Tier 2 LLM fallback (Ollama) to synthesize real code:
//
//   1. LRU Cache      — stateful data structure with HashMap + VecDeque
//   2. Topological Sort — graph algorithm requiring CS logic synthesis
//   3. Worker Pool     — concurrent channels + thread spawning
//
// Success criteria (the full lifecycle):
//   ✅ The Yield:  Phase 3 sequencer emits todo!() — no template exists
//   ✅ The Call:   System formats CodeContext and sends to Ollama
//   ✅ The Synthesis: LLM returns valid Rust code filling the todo!()
//   ✅ The Compile: rustc accepts the code (actual compilation, not simulation)
//   ✅ The Loop:   Success writes to distillation cache (dry pipes → flowing)
//
// Run: SYMTHAEA_LLM_MODEL=mistral:7b cargo test --features code_generation \
//        --test tier2_acid_test -- --nocapture
// ==================================================================================

use std::process::Command;
use std::time::Instant;
use symthaea::language::code_generator::{CodeContext, CodeGenerator};
use symthaea::language::code_intent::{CodeIntent, CodeSpec, CodeTarget};
use symthaea::language::code_parser::EntityKind;
use symthaea::language::consciousness_prompts::CODE_GENERATION_SYSTEM_PROMPT;
use symthaea::language::llm_backend::{GenerationParams, LLMBackend, OllamaBackend};
use symthaea::mind::structured_thought::{CodeContext as ThoughtCodeContext, StructuredThought};

// ==================================================================================
// Helpers
// ==================================================================================

fn ollama_available() -> bool {
    std::net::TcpStream::connect_timeout(
        &"127.0.0.1:11434".parse().unwrap(),
        std::time::Duration::from_secs(2),
    )
    .is_ok()
}

fn needs_llm(source: &str) -> bool {
    source.contains("todo!(") || source.contains("NotImplementedError")
}

/// Extract code from a fenced code block in LLM response.
fn extract_code_block(response: &str) -> String {
    // Try to find ```rust ... ``` or ``` ... ```
    let markers = ["```rust\n", "```rs\n", "```\n"];
    for marker in markers {
        if let Some(start) = response.find(marker) {
            let code_start = start + marker.len();
            if let Some(end) = response[code_start..].find("```") {
                return response[code_start..code_start + end].to_string();
            }
        }
    }
    // No code block found — return the raw response (might be just code)
    response.to_string()
}

/// Try to compile a Rust source file. Returns (success, stderr).
fn try_compile(source: &str, test_name: &str) -> (bool, String) {
    let dir = std::path::Path::new("/tmp/symthaea-acid-test");
    std::fs::create_dir_all(dir).ok();
    let src_path = dir.join(format!("{}.rs", test_name));
    let bin_path = dir.join(test_name);

    std::fs::write(&src_path, source).expect("write source");

    let output = Command::new("rustc")
        .arg("--edition=2021")
        .arg("-o")
        .arg(&bin_path)
        .arg(&src_path)
        .output()
        .expect("run rustc");

    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    (output.status.success(), stderr)
}

/// Try to compile with --test flag and run tests. Returns (compiled, tests_passed, stderr).
fn try_compile_and_test(source: &str, test_name: &str) -> (bool, bool, String) {
    let dir = std::path::Path::new("/tmp/symthaea-acid-test");
    std::fs::create_dir_all(dir).ok();
    let src_path = dir.join(format!("{}.rs", test_name));
    let bin_path = dir.join(format!("{}_test", test_name));

    std::fs::write(&src_path, source).expect("write source");

    // Compile as test binary
    let compile = Command::new("rustc")
        .arg("--edition=2021")
        .arg("--test")
        .arg("-o")
        .arg(&bin_path)
        .arg(&src_path)
        .output()
        .expect("run rustc");

    let compile_stderr = String::from_utf8_lossy(&compile.stderr).to_string();
    if !compile.status.success() {
        return (false, false, compile_stderr);
    }

    // Run tests
    let test_run = Command::new(&bin_path).output().expect("run tests");
    let test_stderr = String::from_utf8_lossy(&test_run.stderr).to_string();
    let test_stdout = String::from_utf8_lossy(&test_run.stdout).to_string();
    let combined = format!("{}\n{}", test_stdout, test_stderr);
    (true, test_run.status.success(), combined)
}

/// Run a single acid test case through the full Tier 2 pipeline.
async fn run_acid_test(
    name: &str,
    purpose: &str,
    signature: &str,
    constraints: &[&str],
    test_code: &str,
    impl_markers: &[&str],
) -> AcidTestResult {
    let start = Instant::now();
    let r#gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();

    // ── Step 1: THE YIELD ──────────────────────────────────────────────
    // Generate code via Tier 1 (native emitter). Should produce todo!().
    let mut spec = CodeSpec::new("rust", name, purpose).with_signature(signature);
    for c in constraints {
        spec = spec.with_constraint(*c);
    }
    let intent = CodeIntent::Create {
        target: CodeTarget::new(name, EntityKind::Function).with_language("rust"),
        spec: spec.clone(),
    };

    let generated = r#gen.generate(&intent, &ctx);
    let yielded_todo = needs_llm(&generated.source);

    eprintln!("\n============================================================");
    eprintln!("ACID TEST: {}", name);
    eprintln!("============================================================");
    eprintln!("  YIELD: todo!() emitted = {}", yielded_todo);
    eprintln!(
        "  Native source ({} bytes):\n    {}",
        generated.source.len(),
        generated
            .source
            .lines()
            .take(5)
            .collect::<Vec<_>>()
            .join("\n    ")
    );

    if !yielded_todo {
        // If native emitter handled it (unlikely for these cases), try compiling directly
        let full_source = format!("{}\n\n{}", generated.source, test_code);
        let (compiled, tests_ok, output) = try_compile_and_test(&full_source, name);
        eprintln!("  SURPRISE: Native emitter handled '{}' directly!", name);
        eprintln!("  Compiled: {}, Tests: {}", compiled, tests_ok);
        return AcidTestResult {
            name: name.to_string(),
            yielded_todo: false,
            called_llm: false,
            llm_responded: false,
            compiled: compiled,
            tests_passed: tests_ok,
            native_handled: true,
            elapsed: start.elapsed(),
            llm_code: generated.source,
            compile_errors: if compiled { String::new() } else { output },
        };
    }

    // ── Step 2: THE CALL ───────────────────────────────────────────────
    // Build structured thought and send to Ollama.
    let mut thought = StructuredThought::default();
    thought.code_context = Some(ThoughtCodeContext {
        language: "rust".to_string(),
        spec_purpose: Some(purpose.to_string()),
        spec_signature: Some(signature.to_string()),
        spec_constraints: constraints.iter().map(|s| s.to_string()).collect(),
        spec_examples: vec![],
        plan_steps: generated
            .plan_steps
            .iter()
            .map(|s| format!("{:?}", s.action))
            .collect(),
        generated_code: Some(generated.source.clone()),
        phi_score: Some(generated.phi_score),
        intent_similarity: Some(generated.intent_similarity),
        syntactically_valid: None,
        notes: generated.notes.clone(),
        needs_llm_completion: true,
    });

    let prompt = thought.to_translation_prompt();
    eprintln!("  CALL: Prompt constructed ({} bytes)", prompt.len());
    eprintln!(
        "  CALL: Contains NEEDS_COMPLETION: {}",
        prompt.contains("NEEDS_COMPLETION: true")
    );

    let model = std::env::var("SYMTHAEA_LLM_MODEL").unwrap_or_else(|_| "mistral:7b".to_string());
    let backend = OllamaBackend::with_timeouts(
        "http://localhost:11434",
        &model,
        std::time::Duration::from_secs(5),
        std::time::Duration::from_secs(180),
    );

    if !backend.is_available().await {
        eprintln!("  SKIP: Ollama not available");
        return AcidTestResult {
            name: name.to_string(),
            yielded_todo: true,
            called_llm: false,
            llm_responded: false,
            compiled: false,
            tests_passed: false,
            native_handled: false,
            elapsed: start.elapsed(),
            llm_code: String::new(),
            compile_errors: "Ollama not available".to_string(),
        };
    }

    let params = GenerationParams {
        temperature: 0.2,
        max_tokens: 4096,
        system_prompt: Some(CODE_GENERATION_SYSTEM_PROMPT.to_string()),
        consciousness_context: None,
    };

    // ── Step 3: THE SYNTHESIS ──────────────────────────────────────────
    let response = backend.generate(&prompt, &params).await;
    let llm_responded = response.is_ok();
    eprintln!("  SYNTHESIS: LLM responded = {}", llm_responded);

    if !llm_responded {
        eprintln!("  ERROR: {:?}", response.err());
        return AcidTestResult {
            name: name.to_string(),
            yielded_todo: true,
            called_llm: true,
            llm_responded: false,
            compiled: false,
            tests_passed: false,
            native_handled: false,
            elapsed: start.elapsed(),
            llm_code: String::new(),
            compile_errors: "LLM generation failed".to_string(),
        };
    }

    let raw_response = response.unwrap();
    let llm_code = extract_code_block(&raw_response);
    eprintln!("  SYNTHESIS: Received {} bytes of code", llm_code.len());
    eprintln!(
        "  First 10 lines:\n    {}",
        llm_code.lines().take(10).collect::<Vec<_>>().join("\n    ")
    );

    // Check for implementation markers
    let lower = llm_code.to_lowercase();
    for marker in impl_markers {
        eprintln!(
            "  MARKER '{}': {}",
            marker,
            if lower.contains(&marker.to_lowercase()) {
                "FOUND"
            } else {
                "MISSING"
            }
        );
    }

    // ── Step 4: THE COMPILE ───────────────────────────────────────────
    // Combine LLM code with our test code and compile
    let full_source = format!("{}\n\n{}", llm_code, test_code);
    let (compiled, tests_ok, output) = try_compile_and_test(&full_source, name);
    eprintln!("  COMPILE: {}", if compiled { "SUCCESS" } else { "FAIL" });
    if !compiled {
        eprintln!(
            "  ERRORS:\n    {}",
            output.lines().take(15).collect::<Vec<_>>().join("\n    ")
        );
    }
    eprintln!(
        "  TESTS: {}",
        if tests_ok {
            "ALL PASSED"
        } else if compiled {
            "SOME FAILED"
        } else {
            "SKIPPED (compile failed)"
        }
    );
    if compiled && !tests_ok {
        eprintln!(
            "  TEST OUTPUT:\n    {}",
            output.lines().take(15).collect::<Vec<_>>().join("\n    ")
        );
    }

    // ── Step 5: THE LOOP ──────────────────────────────────────────────
    if compiled && tests_ok {
        eprintln!("  LOOP: ✅ Ready for distillation cache");
    } else if compiled {
        eprintln!("  LOOP: ⚠️  Compiled but tests failed — partial success");
    } else {
        eprintln!("  LOOP: ❌ Compilation failed — no cache entry");
    }

    eprintln!("  ELAPSED: {:.1}s", start.elapsed().as_secs_f64());

    AcidTestResult {
        name: name.to_string(),
        yielded_todo: true,
        called_llm: true,
        llm_responded: true,
        compiled,
        tests_passed: tests_ok,
        native_handled: false,
        elapsed: start.elapsed(),
        llm_code,
        compile_errors: if compiled { String::new() } else { output },
    }
}

/// Result of a single acid test case.
#[derive(Debug)]
struct AcidTestResult {
    name: String,
    yielded_todo: bool,
    called_llm: bool,
    llm_responded: bool,
    compiled: bool,
    tests_passed: bool,
    native_handled: bool,
    elapsed: std::time::Duration,
    llm_code: String,
    compile_errors: String,
}

impl AcidTestResult {
    fn lifecycle_score(&self) -> (usize, usize) {
        let mut passed = 0;
        let total = 4;
        if self.yielded_todo || self.native_handled {
            passed += 1;
        }
        if self.called_llm || self.native_handled {
            passed += 1;
        }
        if self.llm_responded || self.native_handled {
            passed += 1;
        }
        if self.compiled {
            passed += 1;
        }
        (passed, total)
    }
}

// ==================================================================================
// Test 1: LRU Cache (Data Structure)
// ==================================================================================

#[tokio::test]
async fn acid_test_lru_cache() {
    if !ollama_available() {
        eprintln!("SKIP: Ollama not available");
        return;
    }

    let result = run_acid_test(
        "lru_cache",
        "Implement a Least Recently Used (LRU) cache with get and put operations. \
         Use a HashMap for O(1) lookup and a VecDeque for ordering. \
         When the cache exceeds capacity, evict the least recently used entry.",
        "fn lru_new(capacity: usize) -> LruCache",
        &[
            "Must define struct LruCache with capacity, map, and order fields",
            "Must implement get(&mut self, key: i32) -> Option<i32>",
            "Must implement put(&mut self, key: i32, value: i32)",
            "get() must move the key to most-recently-used position",
            "put() must evict least-recently-used when at capacity",
            "Use std::collections::HashMap and std::collections::VecDeque",
        ],
        r#"
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_put_get() {
        let mut cache = LruCache::new(2);
        cache.put(1, 10);
        cache.put(2, 20);
        assert_eq!(cache.get(1), Some(10));
        assert_eq!(cache.get(2), Some(20));
        assert_eq!(cache.get(3), None);
    }

    #[test]
    fn test_eviction() {
        let mut cache = LruCache::new(2);
        cache.put(1, 10);
        cache.put(2, 20);
        cache.put(3, 30); // evicts key 1
        assert_eq!(cache.get(1), None);
        assert_eq!(cache.get(2), Some(20));
        assert_eq!(cache.get(3), Some(30));
    }

    #[test]
    fn test_access_refreshes() {
        let mut cache = LruCache::new(2);
        cache.put(1, 10);
        cache.put(2, 20);
        cache.get(1); // refresh key 1
        cache.put(3, 30); // should evict key 2 (not 1)
        assert_eq!(cache.get(1), Some(10));
        assert_eq!(cache.get(2), None);
        assert_eq!(cache.get(3), Some(30));
    }

    #[test]
    fn test_overwrite() {
        let mut cache = LruCache::new(2);
        cache.put(1, 10);
        cache.put(1, 100); // overwrite
        assert_eq!(cache.get(1), Some(100));
    }
}
"#,
        &["HashMap", "VecDeque", "capacity", "get", "put"],
    )
    .await;

    let (passed, total) = result.lifecycle_score();
    eprintln!(
        "\n  ACID RESULT [{}]: {}/{} lifecycle stages passed",
        result.name, passed, total
    );

    // The minimum bar: the pipeline must at least yield todo!() and call the LLM
    assert!(
        result.yielded_todo || result.native_handled,
        "LRU cache should either yield todo!() or be handled natively"
    );
    assert!(
        result.called_llm || result.native_handled,
        "LRU cache should call LLM when template is missing"
    );
}

// ==================================================================================
// Test 2: Topological Sort (Graph Algorithm)
// ==================================================================================

#[tokio::test]
async fn acid_test_topological_sort() {
    if !ollama_available() {
        eprintln!("SKIP: Ollama not available");
        return;
    }

    let result = run_acid_test(
        "topo_sort",
        "Implement topological sort using Kahn's algorithm (BFS-based). \
         Given a directed acyclic graph as an adjacency list, return a valid \
         topological ordering. Return None if the graph has a cycle.",
        "fn topological_sort(num_nodes: usize, edges: &[(usize, usize)]) -> Option<Vec<usize>>",
        &[
            "Use Kahn's algorithm (BFS with in-degree tracking)",
            "Build adjacency list and in-degree count from edges",
            "Start with all nodes that have in-degree 0",
            "Return None if not all nodes are visited (cycle detected)",
        ],
        r#"
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_chain() {
        // 0 -> 1 -> 2 -> 3
        let result = topological_sort(4, &[(0, 1), (1, 2), (2, 3)]);
        assert_eq!(result, Some(vec![0, 1, 2, 3]));
    }

    #[test]
    fn test_diamond() {
        // 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        let result = topological_sort(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]).unwrap();
        assert_eq!(result[0], 0); // 0 must be first
        assert_eq!(result[3], 3); // 3 must be last
    }

    #[test]
    fn test_no_edges() {
        let result = topological_sort(3, &[]).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_cycle_detected() {
        // 0 -> 1 -> 2 -> 0 (cycle)
        let result = topological_sort(3, &[(0, 1), (1, 2), (2, 0)]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_single_node() {
        let result = topological_sort(1, &[]);
        assert_eq!(result, Some(vec![0]));
    }
}
"#,
        &["in_degree", "queue", "adjacency", "visited"],
    )
    .await;

    let (passed, total) = result.lifecycle_score();
    eprintln!(
        "\n  ACID RESULT [{}]: {}/{} lifecycle stages passed",
        result.name, passed, total
    );

    assert!(
        result.yielded_todo || result.native_handled,
        "Topological sort should yield todo!() — no template exists"
    );
    assert!(
        result.called_llm || result.native_handled,
        "Topological sort should call LLM"
    );
}

// ==================================================================================
// Test 3: Worker Pool (Concurrency)
// ==================================================================================

#[tokio::test]
async fn acid_test_worker_pool() {
    if !ollama_available() {
        eprintln!("SKIP: Ollama not available");
        return;
    }

    let result = run_acid_test(
        "worker_pool",
        "Implement a simple thread pool that can execute closures in parallel. \
         Use std::sync::mpsc channels and std::thread for the workers. \
         The pool should have a fixed number of worker threads.",
        "fn worker_pool_new(num_threads: usize) -> WorkerPool",
        &[
            "Must define struct WorkerPool with workers and sender fields",
            "Must implement execute(&self, f: impl FnOnce() + Send + 'static)",
            "Workers receive jobs via mpsc::channel",
            "Must implement Drop to join all threads on shutdown",
            "Use std::sync::mpsc, std::thread, std::sync::Arc, std::sync::Mutex",
        ],
        r#"
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

    #[test]
    fn test_basic_execution() {
        let counter = Arc::new(AtomicUsize::new(0));
        {
            let pool = WorkerPool::new(4);
            for _ in 0..8 {
                let c = Arc::clone(&counter);
                pool.execute(move || {
                    c.fetch_add(1, Ordering::SeqCst);
                });
            }
        } // pool drops here, joining threads
        assert_eq!(counter.load(Ordering::SeqCst), 8);
    }

    #[test]
    fn test_single_thread() {
        let counter = Arc::new(AtomicUsize::new(0));
        {
            let pool = WorkerPool::new(1);
            for _ in 0..5 {
                let c = Arc::clone(&counter);
                pool.execute(move || {
                    c.fetch_add(1, Ordering::SeqCst);
                });
            }
        }
        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }
}
"#,
        &["mpsc", "thread", "Arc", "Mutex", "execute"],
    )
    .await;

    let (passed, total) = result.lifecycle_score();
    eprintln!(
        "\n  ACID RESULT [{}]: {}/{} lifecycle stages passed",
        result.name, passed, total
    );

    assert!(
        result.yielded_todo || result.native_handled,
        "Worker pool should yield todo!() — no template exists"
    );
    assert!(
        result.called_llm || result.native_handled,
        "Worker pool should call LLM"
    );
}

// ==================================================================================
// Summary
// ==================================================================================

#[tokio::test]
async fn acid_test_summary() {
    if !ollama_available() {
        eprintln!("SKIP: Ollama not available — cannot run acid test summary");
        return;
    }

    eprintln!("\n");
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║              TIER 2 ACID TEST — SUMMARY                    ║");
    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!("║ Run individual tests above for detailed per-case output.   ║");
    eprintln!("║ This test just verifies the pipeline is structurally sound.║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    // Quick smoke test: generate code for a simple novel case and verify
    // the prompt construction works end-to-end
    let r#gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();

    let intent = CodeIntent::Create {
        target: CodeTarget::new("trie_insert", EntityKind::Function).with_language("rust"),
        spec: CodeSpec::new(
            "rust",
            "trie_insert",
            "Insert a string into a trie data structure",
        )
        .with_signature("fn trie_insert(root: &mut TrieNode, word: &str)"),
    };

    let generated = r#gen.generate(&intent, &ctx);
    let has_todo = needs_llm(&generated.source);

    eprintln!("  Smoke test (trie_insert): todo!() = {}", has_todo);
    eprintln!("  Plan steps: {}", generated.plan_steps.len());
    eprintln!("  Plan coverage: {:.2}", generated.plan_coverage);
    eprintln!("  Intent similarity: {:.3}", generated.intent_similarity);

    // Build structured thought
    let mut thought = StructuredThought::default();
    thought.code_context = Some(ThoughtCodeContext {
        language: "rust".to_string(),
        spec_purpose: Some("Insert into a trie".to_string()),
        spec_signature: Some("fn trie_insert(root: &mut TrieNode, word: &str)".to_string()),
        spec_constraints: vec![],
        spec_examples: vec![],
        plan_steps: generated
            .plan_steps
            .iter()
            .map(|s| format!("{:?}", s.action))
            .collect(),
        generated_code: Some(generated.source.clone()),
        phi_score: Some(generated.phi_score),
        intent_similarity: Some(generated.intent_similarity),
        syntactically_valid: None,
        notes: generated.notes.clone(),
        needs_llm_completion: has_todo,
    });

    let prompt = thought.to_translation_prompt();

    // Verify prompt construction
    assert!(prompt.contains("CODE_LANGUAGE: rust"));
    assert!(prompt.contains("SPEC_PURPOSE:"));
    assert!(prompt.contains("GENERATED_CODE:"));
    if has_todo {
        assert!(
            prompt.contains("NEEDS_COMPLETION: true"),
            "Prompt must signal needs completion"
        );
    }

    eprintln!("  Prompt: {} bytes, well-formed ✅", prompt.len());
    eprintln!("\n  To run the full acid tests with LLM synthesis:");
    eprintln!("    SYMTHAEA_LLM_MODEL=mistral:7b cargo test --features code_generation \\");
    eprintln!("      --test tier2_acid_test -- --nocapture --test-threads=1");
}