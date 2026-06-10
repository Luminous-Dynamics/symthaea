// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Code generation subsystem methods.
//!
//! All items in this module are gated behind `#[cfg(feature = "code_generation")]`.

use super::Symthaea;

impl Symthaea {
    /// Extract function name, entity kind, and optional signature from NL input.
    ///
    /// Parses patterns like:
    /// - "Write a function that reverses a string" -> (reverse, Function, Some("fn reverse(s: &str) -> String"))
    /// - "Create a Point struct with x and y" -> (Point, Struct, None)
    /// - "Implement fibonacci" -> (fibonacci, Function, None)
    pub(super) fn extract_code_metadata(
        content: &str,
        lang: &str,
    ) -> (
        String,
        crate::language::code_parser::EntityKind,
        Option<String>,
    ) {
        use crate::language::code_parser::EntityKind;
        let lower = content.to_lowercase();
        let words: Vec<&str> = content.split_whitespace().collect();

        // Detect entity kind
        let entity_kind =
            if lower.contains("struct") || lower.contains("class") || lower.contains("type ") {
                EntityKind::Struct
            } else if lower.contains("trait") || lower.contains("interface") {
                EntityKind::Trait
            } else if lower.contains("module") || lower.contains("mod ") {
                EntityKind::Module
            } else {
                EntityKind::Function
            };

        // Extract function name — look for known patterns
        let func_name = Self::extract_func_name_from_nl(&lower, &words);

        // Try to infer a signature from NL description
        let signature = if entity_kind == EntityKind::Function {
            Self::infer_signature_from_nl(&lower, &func_name, lang)
        } else {
            None
        };

        (func_name, entity_kind, signature)
    }

    /// Extract a plausible function/entity name from natural language.
    pub(super) fn extract_func_name_from_nl(lower: &str, words: &[&str]) -> String {
        // Pattern 1: explicit "called X" or "named X"
        for (i, w) in words.iter().enumerate() {
            let wl = w.to_lowercase();
            if (wl == "called" || wl == "named") && i + 1 < words.len() {
                let name = words[i + 1].trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
                if !name.is_empty() {
                    return name.to_lowercase();
                }
            }
        }

        // Pattern 2: look for a verb phrase that maps to a function name
        let verb_mappings: &[(&[&str], &str)] = &[
            (&["reverse", "reverses", "reversing"], "reverse"),
            (&["sort", "sorts", "sorting"], "sort"),
            (&["add", "adds", "adding", "sum"], "add"),
            (&["subtract", "subtracts"], "subtract"),
            (&["multiply", "multiplies"], "multiply"),
            (&["divide", "divides"], "divide"),
            (&["check if even", "checks if even", "is even"], "is_even"),
            (&["check if odd", "checks if odd", "is odd"], "is_odd"),
            (&["check if empty", "is empty"], "is_empty"),
            (&["check if positive", "is positive"], "is_positive"),
            (&["check if negative", "is negative"], "is_negative"),
            (&["factorial"], "factorial"),
            (&["fibonacci"], "fibonacci"),
            (&["uppercase", "to uppercase", "upper case"], "to_uppercase"),
            (&["lowercase", "to lowercase", "lower case"], "to_lowercase"),
            (&["contains", "includes"], "contains"),
            (&["starts with", "begins with"], "starts_with"),
            (&["ends with"], "ends_with"),
            (&["trim", "strip"], "trim"),
            (&["replace"], "replace"),
            (&["split"], "split"),
            (&["join", "concatenate"], "join"),
            (&["flatten"], "flatten"),
            (&["unique", "deduplicate"], "unique"),
            (&["filter"], "filter"),
            (&["clamp"], "clamp"),
            (&["absolute value", "abs"], "abs"),
            (&["power", "exponent"], "power"),
            (&["square root", "sqrt"], "sqrt"),
            (&["greatest common", "gcd"], "gcd"),
            (&["average", "mean"], "average"),
            (&["binary search", "bsearch"], "binary_search"),
            (&["dijkstra"], "dijkstra"),
            (&["knapsack"], "solve_knapsack"),
            (&["capitalize"], "capitalize"),
            (&["repeat"], "repeat"),
            (&["enumerate"], "enumerate"),
            (&["zip"], "zip"),
            (&["count"], "count"),
            (&["length", "len"], "length"),
        ];

        for (triggers, name) in verb_mappings {
            for trigger in *triggers {
                if lower.contains(trigger) {
                    return name.to_string();
                }
            }
        }

        // Pattern 3: "function/fn X" or "implement X"
        let prefix_words = [
            "function",
            "fn",
            "implement",
            "create",
            "write",
            "build",
            "make",
        ];
        for (i, w) in words.iter().enumerate() {
            let wl = w.to_lowercase();
            if prefix_words.contains(&wl.as_str()) && i + 1 < words.len() {
                // Skip articles: "a", "an", "the", "that"
                let mut j = i + 1;
                while j < words.len() {
                    let next = words[j].to_lowercase();
                    if ["a", "an", "the", "that", "which", "to"].contains(&next.as_str()) {
                        j += 1;
                    } else {
                        break;
                    }
                }
                if j < words.len() {
                    let candidate = words[j]
                        .trim_matches(|c: char| !c.is_alphanumeric() && c != '_')
                        .to_lowercase();
                    if candidate.len() >= 2
                        && candidate.chars().all(|c| c.is_alphanumeric() || c == '_')
                    {
                        return candidate;
                    }
                }
            }
        }

        // Fallback: use first meaningful word after removing stop words
        let stop = [
            "write",
            "create",
            "implement",
            "make",
            "build",
            "a",
            "an",
            "the",
            "that",
            "which",
            "to",
            "for",
            "in",
            "rust",
            "python",
            "function",
            "method",
            "struct",
            "class",
            "new",
        ];
        for w in words {
            let wl = w.to_lowercase();
            let clean = wl.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
            if clean.len() >= 2 && !stop.contains(&clean) {
                return clean.to_string();
            }
        }

        "generated".to_string()
    }

    /// Infer a Rust/Python function signature from NL description.
    ///
    /// Matches patterns like "takes two integers", "returns a boolean",
    /// "accepts a string and returns a vector of integers".
    pub(super) fn infer_signature_from_nl(
        lower: &str,
        func_name: &str,
        lang: &str,
    ) -> Option<String> {
        // Only infer for Rust currently
        if lang != "rust" {
            return None;
        }

        // Detect parameter types from NL
        let mut params: Vec<(&str, &str)> = Vec::new();

        // "two numbers/integers" -> (a: i32, b: i32)
        if lower.contains("two number")
            || lower.contains("two integer")
            || lower.contains("2 number")
        {
            params.push(("a", "i32"));
            params.push(("b", "i32"));
        } else if lower.contains("two float") || lower.contains("two decimal") {
            params.push(("a", "f64"));
            params.push(("b", "f64"));
        } else if lower.contains("two string") {
            params.push(("a", "&str"));
            params.push(("b", "&str"));
        } else if lower.contains("a string")
            || lower.contains("a str")
            || lower.contains("given string")
        {
            params.push(("s", "&str"));
        } else if lower.contains("a number")
            || lower.contains("an integer")
            || lower.contains("given number")
        {
            params.push(("n", "i32"));
        } else if lower.contains("a vector")
            || lower.contains("a list")
            || lower.contains("an array")
            || lower.contains("a vec")
        {
            if lower.contains("string") || lower.contains("str") {
                params.push(("items", "Vec<String>"));
            } else {
                params.push(("items", "Vec<i32>"));
            }
        } else if lower.contains("three number") || lower.contains("three integer") {
            params.push(("a", "i32"));
            params.push(("b", "i32"));
            params.push(("c", "i32"));
        }

        if params.is_empty() {
            return None;
        }

        // Detect return type from NL
        let ret = if lower.contains("return") && lower.contains("bool")
            || lower.contains("check if")
            || lower.contains("is even")
            || lower.contains("is odd")
            || lower.contains("is empty")
            || lower.contains("is positive")
            || lower.contains("is negative")
        {
            " -> bool"
        } else if lower.contains("return") && lower.contains("string")
            || lower.contains("reverse a string")
            || lower.contains("uppercase")
            || lower.contains("lowercase")
            || lower.contains("capitalize")
        {
            " -> String"
        } else if lower.contains("return") && lower.contains("vector")
            || lower.contains("return") && lower.contains("vec")
            || lower.contains("sort") && params.iter().any(|(_, t)| t.contains("Vec"))
        {
            " -> Vec<i32>"
        } else if lower.contains("return") && lower.contains("float") {
            " -> f64"
        } else if params.iter().any(|(_, t)| t.contains("Vec"))
            && (lower.contains("sum")
                || lower.contains("count")
                || lower.contains("max")
                || lower.contains("min"))
        {
            " -> i32"
        } else if params.iter().any(|(_, t)| *t == "i32" || *t == "f64") {
            if params[0].1 == "f64" {
                " -> f64"
            } else {
                " -> i32"
            }
        } else {
            ""
        };

        let params_str: Vec<String> = params
            .iter()
            .map(|(n, t)| format!("{}: {}", n, t))
            .collect();

        Some(format!(
            "fn {}({}){}",
            func_name,
            params_str.join(", "),
            ret
        ))
    }

    /// Extract a fenced code block from LLM output.
    ///
    /// Returns the content between the first ``` and the closing ```,
    /// stripping the optional language tag. Falls back to the full text
    /// if no fenced block is found.
    pub(super) fn extract_code_block(text: &str) -> String {
        if let Some(start) = text.find("```") {
            let after_fence = &text[start + 3..];
            // Skip the language tag (first line after ```)
            let code_start = after_fence.find('\n').map(|i| i + 1).unwrap_or(0);
            let code_region = &after_fence[code_start..];
            if let Some(end) = code_region.find("```") {
                return code_region[..end].trim().to_string();
            }
        }
        text.to_string()
    }

    /// Parse code using the appropriate tree-sitter parser for verification.
    pub(super) fn parse_code_for_verification(
        lang: &str,
        source: &str,
    ) -> Option<crate::language::code_parser::ParsedCode> {
        use crate::language::code_parser::CodeParser;
        match lang {
            "rust" => {
                let mut parser = crate::language::rust_parser::RustParser::new();
                parser.parse(source).ok()
            }
            "python" => {
                let mut parser = crate::language::python_parser::PythonParser::new();
                parser.parse(source).ok()
            }
            _ => None,
        }
    }

    /// Index a project directory into CodebaseMemory for semantic code search.
    ///
    /// Walks the directory tree (respecting common ignore patterns), parses each
    /// source file, and encodes its AST into HDC vectors. After indexing, the
    /// code generator can query for relevant functions/types when generating new code.
    ///
    /// Returns `(files_indexed, parse_errors)`.
    pub fn index_project(&mut self, root: &std::path::Path) -> (usize, usize) {
        use crate::language::parser_registry::ParserRegistry;

        let mut parser_registry = ParserRegistry::new();
        let mut files_indexed = 0usize;
        let mut parse_errors = 0usize;
        let start = std::time::Instant::now();

        // Collect source files recursively (skip hidden, target, node_modules, etc.)
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let entries = match std::fs::read_dir(&dir) {
                Ok(e) => e,
                Err(_) => continue,
            };
            for entry in entries.filter_map(|e| e.ok()) {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with('.')
                    || name_str == "target"
                    || name_str == "node_modules"
                    || name_str == "venv"
                    || name_str == "__pycache__"
                {
                    continue;
                }
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.is_file() {
                    let filename = path.file_name().and_then(|n| n.to_str());
                    // Quick extension check before reading file
                    if filename.is_none() {
                        continue;
                    }
                    let ext = path.extension().and_then(|e| e.to_str());
                    let is_parseable = matches!(ext, Some("rs") | Some("py") | Some("nix"));
                    if !is_parseable {
                        continue;
                    }
                    match std::fs::read_to_string(&path) {
                        Ok(source) => match parser_registry.parse(&source, None, filename) {
                            Ok(parsed) => {
                                self.code_memory.index_file(&path, &parsed);
                                files_indexed += 1;
                            }
                            Err(_) => parse_errors += 1,
                        },
                        Err(_) => parse_errors += 1,
                    }
                }
            }
        }

        let elapsed = start.elapsed();
        tracing::info!(
            target: "symthaea::code_memory",
            files = files_indexed,
            errors = parse_errors,
            functions = self.code_memory.function_count(),
            types = self.code_memory.type_count(),
            elapsed_ms = elapsed.as_millis(),
            "Project indexed"
        );

        (files_indexed, parse_errors)
    }

    /// Query the codebase memory for functions/types similar to a natural language query.
    ///
    /// Returns up to `top_k` matches with similarity scores. Requires prior `index_project()`.
    pub fn query_codebase(
        &self,
        query: &str,
        top_k: usize,
    ) -> Vec<crate::hdc::code_memory::CodeMatch> {
        let query_hv = self.code_memory.encoder().encode_name(query);
        self.code_memory.query(&query_hv, top_k)
    }

    /// Get the codebase coherence score (0.0 = fragmented, 1.0 = highly cohesive).
    pub fn codebase_coherence(&self) -> f32 {
        self.code_memory.codebase_coherence()
    }

    /// Access the code memory directly for advanced queries.
    pub fn code_memory(&self) -> &crate::hdc::code_memory::CodebaseMemory {
        &self.code_memory
    }

    /// Run a coding task through the consciousness-gated agentic loop.
    ///
    /// This is the primary entry point for coding AI functionality. It:
    /// 1. Queries `CodebaseMemory` for relevant context (if indexed)
    /// 2. Creates a `CodingAgent` with the project's working directory
    /// 3. Feeds codebase context into the agent's generation prompts
    /// 4. Runs the multi-step loop (understand -> plan -> generate -> test -> fix)
    /// 5. Records the outcome for backend stats learning
    ///
    /// Call `index_project()` first for codebase-aware generation.
    pub fn run_coding_task(&mut self, task: &str) -> crate::coding_agent::AgentResult {
        use crate::coding_agent::{CodingAgent, CodingAgentConfig};

        let working_dir = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));

        let config = CodingAgentConfig {
            working_dir: working_dir.clone(),
            ..Default::default()
        };

        let mut agent = CodingAgent::new(config).unwrap_or_else(|e| {
            tracing::error!(target: "symthaea::coding", error = %e, "Failed to create CodingAgent");
            // Fallback: create with default config (will use current dir)
            CodingAgent::new(CodingAgentConfig::default()).expect("CodingAgent default must work")
        });

        // Query CodebaseMemory for relevant context
        let context: Vec<String> = self
            .query_codebase(task, 5)
            .into_iter()
            .map(|m| {
                format!(
                    "// {}::{} (similarity: {:.2})\n// file: {}",
                    m.kind,
                    m.name,
                    m.similarity,
                    m.path.display()
                )
            })
            .collect();

        if !context.is_empty() {
            tracing::info!(
                target: "symthaea::coding",
                matches = context.len(),
                "Injecting codebase context into agent"
            );
            agent.set_code_context(context);
        }

        // Run the agent
        let result = agent.run(task);

        // Record outcome into error pattern memory for future generations
        if let Some(false) = result.tests_passed {
            for err in &result.errors {
                if err.len() > 10 {
                    // Extract a short pattern from the error
                    let pattern = err.chars().take(120).collect::<String>();
                    self.error_pattern_memory.push((pattern, task.to_string()));
                    // Cap error memory at 64 entries
                    if self.error_pattern_memory.len() > 64 {
                        self.error_pattern_memory.remove(0);
                    }
                }
            }
        }

        // Cache successful generations
        if result.tests_passed == Some(true) && !result.files_modified.is_empty() {
            self.code_generation_cache
                .push((task.to_string(), format!("{:?}", result.files_modified)));
            if self.code_generation_cache.len() > 32 {
                self.code_generation_cache.remove(0);
            }
        }

        tracing::info!(
            target: "symthaea::coding",
            task = task,
            iterations = result.iterations_used,
            files = result.files_modified.len(),
            phase = %result.final_phase,
            tiers = ?result.generation_tiers,
            energy = result.total_energy,
            "Coding task complete"
        );

        result
    }

    /// Run a coding task with a custom configuration.
    pub fn run_coding_task_with_config(
        &mut self,
        task: &str,
        config: crate::coding_agent::CodingAgentConfig,
    ) -> crate::coding_agent::AgentResult {
        use crate::coding_agent::CodingAgent;

        let mut agent = CodingAgent::new(config)
            .unwrap_or_else(|_| CodingAgent::new(Default::default()).expect("default agent"));

        let context: Vec<String> = self
            .query_codebase(task, 5)
            .into_iter()
            .map(|m| {
                format!(
                    "// {}::{} (similarity: {:.2})\n// file: {}",
                    m.kind,
                    m.name,
                    m.similarity,
                    m.path.display()
                )
            })
            .collect();

        if !context.is_empty() {
            agent.set_code_context(context);
        }

        let result = agent.run(task);

        // Same outcome recording as run_coding_task
        if let Some(false) = result.tests_passed {
            for err in &result.errors {
                if err.len() > 10 {
                    let pattern = err.chars().take(120).collect::<String>();
                    self.error_pattern_memory.push((pattern, task.to_string()));
                    if self.error_pattern_memory.len() > 64 {
                        self.error_pattern_memory.remove(0);
                    }
                }
            }
        }

        if result.tests_passed == Some(true) && !result.files_modified.is_empty() {
            self.code_generation_cache
                .push((task.to_string(), format!("{:?}", result.files_modified)));
            if self.code_generation_cache.len() > 32 {
                self.code_generation_cache.remove(0);
            }
        }

        result
    }
}
