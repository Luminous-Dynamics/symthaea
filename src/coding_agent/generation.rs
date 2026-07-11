// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Generation and fixing methods for the coding agent.

use super::*;

impl CodingAgent {
    fn code_signals(&self) -> super::consciousness_bridge::CodeSignals {
        super::consciousness_bridge::CodeSignals::from_agent_state(
            &self.failure_patterns,
            self.iteration,
            self.phase_failures,
            self.generated_code.as_deref(),
            self.energy_budget,
            100.0,
            self.native_exhausted,
        )
    }

    /// Generate code via the IntelligentDispatcher and write to disk.
    pub(super) fn do_generation(&mut self) {
        // Consciousness gate: defer generation if Phi is below plan requirement
        let current_phi = self.phi_trace.last().copied().unwrap_or(0.0);
        if let Some(ref plan) = self.current_plan {
            if !plan.phi_sufficient(current_phi) {
                tracing::debug!(
                    target: "symthaea::coding_agent",
                    current_phi = current_phi,
                    min_phi = plan.min_phi,
                    "Consciousness gate: deferring generation (Phi too low)"
                );
                self.observations.push(format!(
                    "Generation deferred: consciousness level {:.3} below plan minimum {:.2}",
                    current_phi, plan.min_phi
                ));
                self.consciousness_deferrals += 1;
                return;
            }
        }

        // Try the unified CodeOrchestrator first when wired in (feature + config
        // gated via `use_orchestrator`). The orchestrator runs native/analogy/LLM
        // backends internally, each accepted only after its own compiler/test
        // verification, so a positive result here supersedes the raw
        // IntelligentDispatcher path for this iteration. Falls through to the
        // existing dispatcher logic below when disabled or unaccepted, so default
        // behavior (use_orchestrator=false) is unchanged.
        #[cfg(feature = "code_generation")]
        if self.try_orchestrator_generation() {
            return;
        }

        // Get consciousness state for dispatch routing
        let confidence = self.cognitive_loop.prediction_confidence();
        let phi = self.phi_trace.last().copied().unwrap_or(0.5) as f64;
        let prediction_error = self.cognitive_loop.prediction_confidence(); // inverse proxy

        // If native generation was exhausted (returned None), override consciousness
        // state to force the dispatcher toward LLM tier:
        // - Epistemic -> Uncertain (triggers LLM selection)
        // - Prediction error -> 0.7+ (confirms need for external help)
        // - Phi -> 0.5 (bypasses the consciousness < 0.2 -> Native override)
        let (epistemic, prediction_error, phi) = if self.native_exhausted {
            (
                EpistemicStatus::Uncertain,
                0.7_f64.max(prediction_error as f64),
                0.5,
            )
        } else {
            (
                Self::confidence_to_epistemic(confidence),
                prediction_error as f64,
                phi,
            )
        };

        // Build the generation prompt and system prompt before borrowing dispatcher
        let prompt = self.build_generation_prompt();
        let sys_prompt = self.codegen_system_prompt();

        let code_signals = self.code_signals();

        // Call the dispatcher (async -> sync bridge)
        let dispatch_result = if let Some(ref mut dispatcher) = self.dispatcher {
            // Consciousness-informed temperature: higher prediction error -> more exploration
            let pe = self.prediction_error_history.last().copied().unwrap_or(0.3);
            let temperature = (0.3 + pe * 0.3).min(0.9);

            // Apply forced backend tier from retry strategy
            if let RetryStrategy::DifferentBackend(tier) = &self.retry_state.current_strategy {
                dispatcher.force_next_tier(*tier);
            }

            // Build consciousness context for the LLM backend
            let consciousness_ctx = crate::language::llm_backend::ConsciousnessContext {
                epistemic_status: format!("{:?}", epistemic),
                phi: current_phi,
                type_confidence: code_signals.type_confidence as f32,
                algorithm_pattern: code_signals.algorithm_pattern,
                error_likelihood: code_signals.error_likelihood,
                syntax_complexity: code_signals.syntax_complexity,
            };

            let params = GenerationParams {
                temperature,
                max_tokens: 4096, // was 1024 — truncated code+prose mid-function; see CAPABILITY_LADDER.md
                system_prompt: Some(sys_prompt.clone()),
                consciousness_context: Some(consciousness_ctx),
            };

            // Sync bridge for async dispatcher
            let result = Self::block_on_dispatch(
                dispatcher,
                &prompt,
                &params,
                epistemic,
                prediction_error,
                phi,
            );
            Some(result)
        } else {
            None
        };

        // Process the dispatch result
        if let Some(result) = dispatch_result {
            self.generation_tiers.push(result.tier);
            self.energy_budget -= result.energy_cost as f32;

            tracing::debug!(
                target: "symthaea::coding_agent",
                tier = %result.tier,
                native_exhausted = self.native_exhausted,
                success = result.success,
                output_len = result.output.len(),
                energy_remaining = self.energy_budget,
                "Dispatch result"
            );

            // Fast-fail: if native is exhausted and the "LLM" returned simulated
            // or signal output, there's no real backend to escalate to. Stop early
            // instead of looping through remaining iterations.
            if self.native_exhausted
                && result.tier != BackendTier::Native
                && (result.output.contains("simulated")
                    || result.output.contains("[NATIVE:")
                    || result.output.is_empty())
            {
                tracing::info!(
                    target: "symthaea::coding_agent",
                    task = %self.task,
                    tier = %result.tier,
                    "No real LLM available — fast-failing"
                );
                self.observations
                    .push("Fast-fail: no real LLM backend available for this task".into());
                self.phase = TaskPhase::Done;
                self.last_dispatch = Some(result);
                return;
            }

            if result.success && result.tier != BackendTier::Native {
                // LLM-generated code — write to disk
                let target = self.resolve_target_file();
                self.write_code_to_disk(&target, &result.output);
                self.generated_code = Some(Self::sanitize_generated_code(
                    &Self::strip_code_fences(&result.output),
                ));
                // LLM succeeded — clear native_exhausted (task was handled)
                self.native_exhausted = false;

                tracing::info!(
                    target: "symthaea::coding_agent",
                    tier = %result.tier,
                    energy = result.energy_cost,
                    target = %target.display(),
                    "Code generated and written"
                );
            } else if result.tier == BackendTier::Native {
                // Native tier — try pattern-aware generation
                if let Some(code) = self.native_code_template() {
                    let target = self.resolve_target_file();
                    self.write_code_to_disk(&target, &code);
                    self.generated_code = Some(Self::sanitize_generated_code(
                        &Self::strip_code_fences(&code),
                    ));
                } else {
                    // Native can't handle this — immediately escalate to LLM
                    // within the SAME iteration (don't wait for next cycle).
                    self.native_exhausted = true;
                    if let Some(ref mut dispatcher) = self.dispatcher {
                        dispatcher.record_outcome_with_category(
                            BackendTier::Native,
                            false,
                            &self.task,
                        );

                        // Re-dispatch with overridden state to force LLM tier
                        let params = GenerationParams {
                            temperature: 0.4,
                            max_tokens: 4096, // was 1024 — truncated code+prose mid-function; see CAPABILITY_LADDER.md
                            system_prompt: Some(sys_prompt.clone()),
                            consciousness_context: None, // escalation path — no context needed
                        };
                        let llm_result = Self::block_on_dispatch(
                            dispatcher,
                            &prompt,
                            &params,
                            EpistemicStatus::Uncertain,
                            0.7,
                            0.5, // bypass consciousness < 0.2 check
                        );
                        self.generation_tiers.push(llm_result.tier);
                        tracing::info!(
                            target: "symthaea::coding_agent",
                            tier = %llm_result.tier,
                            success = llm_result.success,
                            output_len = llm_result.output.len(),
                            "Native->LLM escalation"
                        );
                        if llm_result.success && llm_result.tier != BackendTier::Native {
                            let target = self.resolve_target_file();
                            self.write_code_to_disk(&target, &llm_result.output);
                            self.generated_code = Some(Self::sanitize_generated_code(
                                &Self::strip_code_fences(&llm_result.output),
                            ));
                            self.native_exhausted = false;
                        } else {
                            self.observations
                                .push("Native exhausted, LLM escalation attempted".into());
                        }
                        self.last_dispatch = Some(llm_result);
                        return; // already processed
                    }
                    self.observations.push(
                        "Native generation: no matching pattern, no dispatcher available".into(),
                    );
                }
            } else {
                self.errors.push(format!(
                    "Generation failed ({}): {}",
                    result.tier, result.output
                ));
            }

            self.last_dispatch = Some(result);
        }
    }

    /// Try to auto-fix the last compilation error using structured (line-aware) fixes.
    ///
    /// Parses `last_test_output` into structured errors with file/line/column info,
    /// then applies targeted fixes (type conversions, clone insertion, lifetime
    /// annotations, derive attributes). If a fix is applied, writes the fixed code
    /// to disk and sets `generated_code` — skipping the LLM entirely.
    ///
    /// Returns `true` if a fix was applied (caller should skip LLM generation).
    pub(super) fn try_structured_auto_fix(&mut self) -> bool {
        // Need both the error output and the generated code to fix
        let (test_output, code) = match (&self.last_test_output, &self.generated_code) {
            (Some(output), Some(code)) => (output.clone(), code.clone()),
            _ => return false,
        };

        // Parse structured errors
        let structured = crate::language::code_executor::parse_structured_errors(&test_output);
        if structured.is_empty() {
            return false;
        }

        // Check error knowledge graph for semantically-ranked fix strategies
        for error in &structured {
            let error_code = error.code.clone().unwrap_or_default();
            let category = super::error_knowledge::ErrorCategory::from_error_code(&error_code);
            if let Some(best) = self.error_knowledge.best_fix(&error_code, category) {
                tracing::info!(
                    target: "symthaea::coding_agent",
                    error_code = %error_code,
                    best_fix = %best,
                    "Knowledge graph suggests fix strategy"
                );
                self.observations.push(format!(
                    "Knowledge graph: best fix for {} is '{}'",
                    error_code, best
                ));
            }
        }

        // Check experience store for cached fix strategies before re-parsing
        if let Some(ref store) = self.experience_store {
            for error in &structured {
                let sig = Self::normalize_error_pattern(&error.message);
                if let Some(strategy) = store.lookup_fix_strategy(&sig) {
                    tracing::debug!(
                        target: "symthaea::coding_agent",
                        error_sig = %sig,
                        strategy = %strategy,
                        "Found cached fix strategy"
                    );
                    self.observations.push(format!(
                        "Cached fix strategy for {}: {}",
                        error.code.as_deref().unwrap_or("unknown"),
                        strategy
                    ));
                }
            }
        }

        // Try compiler-suggested replacements first (highest fidelity — from rustc itself)
        // These come from JSON diagnostics if the output contains JSON lines
        let json_errors = crate::language::code_executor::parse_json_diagnostics(&test_output);
        let has_suggestions = json_errors
            .iter()
            .any(|e| e.suggested_replacement.is_some());
        if has_suggestions {
            let suggestion_key = "compiler-suggestion-fix".to_string();
            if !self.attempted_fixes.contains(&suggestion_key) {
                if let Some(fixed) = Self::try_apply_compiler_suggestions(&code, &json_errors) {
                    self.attempted_fixes.insert(suggestion_key);
                    let target = self.resolve_target_file();
                    self.write_code_to_disk(&target, &fixed);
                    self.generated_code = Some(Self::strip_code_fences(&fixed));
                    self.observations.push(format!(
                        "Applied {} compiler-suggested fix(es)",
                        json_errors
                            .iter()
                            .filter(|e| e.suggested_replacement.is_some())
                            .count()
                    ));
                    tracing::info!(
                        target: "symthaea::coding_agent",
                        "Applied compiler-suggested fixes from JSON diagnostics"
                    );
                    return true;
                }
            }
        }

        // Build dedup key from first error signature
        let error_sig = structured
            .first()
            .map(|e| Self::normalize_error_pattern(&e.message))
            .unwrap_or_default();

        // Chained fix pipeline: apply ALL applicable fix strategies in sequence.
        // Each strategy transforms the code, and the next strategy operates on the
        // already-fixed output. This catches cascading errors (e.g., stripping
        // fn main() reveals an undeclared generic underneath).
        let mut current_code = code.clone();
        let mut any_chained_fix = false;
        let mut fix_descriptions: Vec<String> = Vec::new();

        // Stage 1: Structured (line-aware) fixes
        let structured_key = format!("{}:structured-line-fix", error_sig);
        if self.attempted_fixes.contains(&structured_key) {
            self.dedup_skips += 1;
        } else if let Some(fixed) =
            crate::language::code_executor::try_auto_fix_structured(&current_code, &structured)
        {
            self.attempted_fixes.insert(structured_key);
            current_code = fixed;
            any_chained_fix = true;
            fix_descriptions.push(format!("structured-line-fix ({} errors)", structured.len()));
            self.store_fix_strategies(&structured, "structured-line-fix");
            self.observe_errors_for_self_mod(&structured, "structured-line-fix", true);
        }

        // Stage 2: Category-aware fixes (operates on output of stage 1)
        let category_key = format!("{}:category-aware-fix", error_sig);
        if self.attempted_fixes.contains(&category_key) {
            self.dedup_skips += 1;
        } else if let Some(fixed) = self.try_category_aware_fix(&current_code, &structured) {
            self.attempted_fixes.insert(category_key);
            current_code = fixed;
            any_chained_fix = true;
            fix_descriptions.push(format!(
                "category-aware-fix ({})",
                structured
                    .iter()
                    .map(|e| format!("{:?}", e.category))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            self.store_fix_strategies(&structured, "category-aware-fix");
            self.observe_errors_for_self_mod(&structured, "category-aware-fix", true);
        }

        // Commit chained fixes if any stage succeeded
        if any_chained_fix {
            let target = self.resolve_target_file();
            self.write_code_to_disk(&target, &current_code);
            self.generated_code = Some(Self::strip_code_fences(&current_code));
            let desc = fix_descriptions.join(" -> ");
            self.observations
                .push(format!("Chained auto-fix applied: {}", desc));
            tracing::info!(
                target: "symthaea::coding_agent",
                pipeline = %desc,
                "Chained auto-fix pipeline applied, skipping LLM"
            );
            return true;
        }

        // Fall back to basic (non-line-aware) fix
        let flat_errors: Vec<String> = structured.iter().map(|e| e.message.clone()).collect();
        let basic_key = format!("{}:basic-pattern-fix", error_sig);
        if self.attempted_fixes.contains(&basic_key) {
            self.dedup_skips += 1;
        } else if let Some(fixed) =
            crate::language::code_executor::try_auto_fix(&code, &flat_errors)
        {
            self.attempted_fixes.insert(basic_key);
            let target = self.resolve_target_file();
            self.write_code_to_disk(&target, &fixed);
            self.generated_code = Some(Self::strip_code_fences(&fixed));

            // Store successful fix strategies for future reuse
            self.store_fix_strategies(&structured, "basic-pattern-fix");
            self.observe_errors_for_self_mod(&structured, "basic-pattern-fix", true);

            self.observations.push("Basic auto-fix applied".into());
            tracing::info!(
                target: "symthaea::coding_agent",
                "Basic auto-fix applied, skipping LLM"
            );
            return true;
        }

        // Log which categories will escalate to LLM (for observability)
        let categories: Vec<_> = structured
            .iter()
            .map(|e| format!("{:?}", e.category))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        tracing::info!(
            target: "symthaea::coding_agent",
            ?categories,
            "Auto-fix exhausted, escalating to LLM"
        );
        self.observe_errors_for_self_mod(&structured, "escalate-to-llm", false);

        false
    }

    /// Apply compiler-suggested replacements from JSON diagnostics.
    ///
    /// When `CompileError.suggested_replacement` is populated (from
    /// `parse_json_diagnostics()`), this applies the compiler's own fix suggestions
    /// directly — the most reliable auto-fix possible since rustc computed them.
    ///
    /// Returns `Some(fixed_code)` if any replacement was applied.
    pub(super) fn try_apply_compiler_suggestions(
        code: &str,
        errors: &[crate::language::code_executor::CompileError],
    ) -> Option<String> {
        let mut lines: Vec<String> = code.lines().map(|l| l.to_string()).collect();
        let mut any_fix = false;

        // Apply in reverse line order to preserve line numbers
        let mut fixes: Vec<_> = errors
            .iter()
            .filter_map(|e| {
                let replacement = e.suggested_replacement.as_ref()?;
                let line = e.line?;
                let col = e.column.unwrap_or(1);
                Some((line, col, replacement.clone()))
            })
            .collect();
        fixes.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));

        for (line_num, _col, replacement) in &fixes {
            let idx = line_num.saturating_sub(1);
            if idx < lines.len() && !replacement.is_empty() {
                lines[idx] = replacement.clone();
                any_fix = true;
            }
        }

        if any_fix {
            Some(lines.join("\n"))
        } else {
            None
        }
    }

    /// Attempt category-aware fixes that the line-level auto-fix missed.
    ///
    /// Unlike `try_auto_fix_structured` (which does line-targeted patches), this
    /// applies whole-file transformations based on error category:
    /// - `MissingImport` -> scan for unresolved names and add `use` statements
    /// - `UnusedCode` -> prefix unused variables with `_`
    /// - `BorrowError` ("cannot borrow as mutable") -> add `mut` to bindings
    /// - `MissingImpl` -> add common derives when the struct is in our code
    pub(super) fn try_category_aware_fix(
        &self,
        code: &str,
        errors: &[crate::language::code_executor::CompileError],
    ) -> Option<String> {
        use crate::language::code_executor::ErrorCategory;

        let mut lines: Vec<String> = code.lines().map(|l| l.to_string()).collect();
        let mut any_fix = false;

        for error in errors {
            match error.category {
                ErrorCategory::MissingImport => {
                    // Extract the unresolved name from "cannot find X in this scope"
                    if let Some(name) = Self::extract_unresolved_name(&error.message) {
                        // Common std imports
                        let use_stmt = match name.as_str() {
                            "HashMap" => Some("use std::collections::HashMap;"),
                            "HashSet" => Some("use std::collections::HashSet;"),
                            "BTreeMap" => Some("use std::collections::BTreeMap;"),
                            "BTreeSet" => Some("use std::collections::BTreeSet;"),
                            "VecDeque" => Some("use std::collections::VecDeque;"),
                            "BinaryHeap" => Some("use std::collections::BinaryHeap;"),
                            "Rc" => Some("use std::rc::Rc;"),
                            "Arc" => Some("use std::sync::Arc;"),
                            "Mutex" => Some("use std::sync::Mutex;"),
                            "RwLock" => Some("use std::sync::RwLock;"),
                            "Cell" => Some("use std::cell::Cell;"),
                            "RefCell" => Some("use std::cell::RefCell;"),
                            "Path" => Some("use std::path::Path;"),
                            "PathBuf" => Some("use std::path::PathBuf;"),
                            "File" => Some("use std::fs::File;"),
                            "Read" => Some("use std::io::Read;"),
                            "Write" => Some("use std::io::Write;"),
                            "Display" => Some("use std::fmt::Display;"),
                            "Formatter" => Some("use std::fmt::Formatter;"),
                            _ => None,
                        };
                        if let Some(stmt) = use_stmt {
                            if !lines.iter().any(|l| l.trim() == stmt) {
                                lines.insert(0, stmt.to_string());
                                any_fix = true;
                            }
                        } else {
                            // Fallback: query CodebaseMemory for project-specific types
                            #[cfg(feature = "code_generation")]
                            if let Some(ref memory) = self.code_memory {
                                use crate::language::code_parser::EntityKind;
                                let encoder = crate::hdc::code_encoder::CodeHDEncoder::new(16384);
                                let name_hv = encoder.encode_name(&name);
                                let matches = memory.query_types(&name_hv, 3);
                                if let Some(best) = matches.iter().find(|m| {
                                    m.similarity > 0.4
                                        && matches!(
                                            m.kind,
                                            EntityKind::Struct
                                                | EntityKind::Enum
                                                | EntityKind::Trait
                                                | EntityKind::TypeAlias
                                        )
                                }) {
                                    let path_str = best.path.to_string_lossy();
                                    let mod_path = path_str
                                        .trim_start_matches("src/")
                                        .trim_end_matches(".rs")
                                        .trim_end_matches("/mod")
                                        .replace('/', "::");
                                    let use_line =
                                        format!("use crate::{}::{};", mod_path, best.name);
                                    if !lines.iter().any(|l| l.trim() == use_line) {
                                        lines.insert(0, use_line);
                                        any_fix = true;
                                    }
                                }
                            }
                        }
                    }
                }

                ErrorCategory::UnusedCode => {
                    // "unused variable: `x`" -> rename to `_x`
                    if let Some(var_name) = Self::extract_unused_var(&error.message) {
                        if !var_name.starts_with('_') {
                            let prefixed = format!("_{}", var_name);
                            if let Some(idx) = error
                                .line
                                .map(|l| l.saturating_sub(1))
                                .filter(|&i| i < lines.len())
                            {
                                let line = &lines[idx];
                                // Only rename in let bindings, not usage sites
                                if line.contains("let ") || line.contains("let mut ") {
                                    let new_line = line.replacen(&var_name, &prefixed, 1);
                                    if new_line != *line {
                                        lines[idx] = new_line;
                                        any_fix = true;
                                    }
                                }
                            }
                        }
                    }
                }

                // "cannot borrow `x` as mutable, as it is not declared as mutable"
                ErrorCategory::BorrowError if error.message.contains("not declared as mutable") => {
                    if let Some(var_name) = Self::extract_between_backticks(&error.message) {
                        // Find the `let x` binding and add `mut`
                        let let_pattern = format!("let {}", var_name);
                        for line in &mut lines {
                            if line.contains(&let_pattern) && !line.contains("let mut ") {
                                *line = line.replacen(
                                    &let_pattern,
                                    &format!("let mut {}", var_name),
                                    1,
                                );
                                any_fix = true;
                                break;
                            }
                        }
                    }
                }

                ErrorCategory::UndeclaredGeneric => {
                    // Re-run the generic fixer on the whole file
                    let fixed = Self::fix_undeclared_generics(&lines.join("\n"));
                    let new_lines: Vec<String> = fixed.lines().map(|l| l.to_string()).collect();
                    if new_lines != lines {
                        lines = new_lines;
                        any_fix = true;
                    }
                }

                ErrorCategory::UnwantedMain => {
                    // Strip fn main() wrapper if present
                    let fixed = Self::strip_main_wrapper(&lines.join("\n"));
                    let new_lines: Vec<String> = fixed.lines().map(|l| l.to_string()).collect();
                    if new_lines != lines {
                        lines = new_lines;
                        any_fix = true;
                    }
                }

                // TypeMismatch, LifetimeError, MissingImpl, SyntaxError — already
                // handled well by try_auto_fix_structured, or need LLM for complex cases
                _ => {}
            }
        }

        if any_fix {
            Some(lines.join("\n"))
        } else {
            None
        }
    }

    /// Extract unresolved name from "cannot find type/value/module `X`" messages.
    pub(super) fn extract_unresolved_name(msg: &str) -> Option<String> {
        // "cannot find type `HashMap` in this scope"
        // "cannot find value `x` in this scope"
        Self::extract_between_backticks(msg)
    }

    /// Extract variable name from "unused variable: `x`" messages.
    pub(super) fn extract_unused_var(msg: &str) -> Option<String> {
        if msg.contains("unused variable") || msg.contains("unused mut") {
            Self::extract_between_backticks(msg)
        } else {
            None
        }
    }

    /// Extract text between first pair of backticks in a message.
    pub(super) fn extract_between_backticks(msg: &str) -> Option<String> {
        let start = msg.find('`')? + 1;
        let rest = &msg[start..];
        let end = rest.find('`')?;
        Some(rest[..end].to_string())
    }

    /// Store fix strategies from successful auto-fixes into both the experience store
    /// and the semantic error knowledge graph.
    pub(super) fn store_fix_strategies(
        &mut self,
        errors: &[crate::language::code_executor::CompileError],
        strategy: &str,
    ) {
        for error in errors {
            let sig = Self::normalize_error_pattern(&error.message);
            let strategy_desc = format!(
                "{} ({})",
                strategy,
                error.code.as_deref().unwrap_or("unknown")
            );

            // Store in flat experience store
            if let Some(ref mut store) = self.experience_store {
                store.store_fix_strategy(&sig, &strategy_desc, None);
            }

            // Store in semantic error knowledge graph (richer: tracks success rates)
            let error_code = error.code.clone().unwrap_or_default();
            let category = super::error_knowledge::ErrorCategory::from_error_code(&error_code);
            let context: String = self
                .generated_code
                .as_deref()
                .unwrap_or("")
                .chars()
                .take(200)
                .collect();

            self.error_knowledge
                .record_fix(super::error_knowledge::CodeErrorFact {
                    error_code,
                    category,
                    pattern_signature: sig,
                    fix_strategy: strategy_desc,
                    compiled: true,     // we're recording a fix that was applied
                    tests_passed: None, // not yet known — will be updated after testing
                    context_snippet: context,
                });
        }
    }

    /// Observe real compiler errors into the self-modification error-cluster
    /// tracker (`FixRuleGenerator`), closing the wiring gap noted in the coding-agent
    /// audit: `FixRuleGenerator::observe_error()` previously had zero callers anywhere.
    ///
    /// This is an **observation-only** call site: it never applies or promotes a
    /// generated rule. Rule *generation* (`try_generate_rules()`) additionally
    /// requires `config.enable_self_modification` (default `false`); rule
    /// *application*/*promotion* (`try_apply_rule`, `record_rule_outcome`) is not
    /// wired into the live agent loop at all — this is a self-modification
    /// pipeline and must never silently mutate the agent's own fix repertoire.
    pub(super) fn observe_errors_for_self_mod(
        &mut self,
        errors: &[crate::language::code_executor::CompileError],
        strategy: &str,
        success: bool,
    ) {
        for error in errors {
            let sig = Self::normalize_error_pattern(&error.message);
            let error_code = error.code.clone().unwrap_or_default();
            let category = super::error_knowledge::ErrorCategory::from_error_code(&error_code);
            let context: String = self
                .generated_code
                .as_deref()
                .unwrap_or("")
                .chars()
                .take(200)
                .collect();
            self.fix_rule_generator.observe_error(
                &error_code,
                &format!("{:?}", category),
                &sig,
                strategy,
                success,
                &context,
            );
        }

        if self.config.enable_self_modification {
            let phi = self.phi_trace.last().copied().unwrap_or(0.0);
            let calibration = self.current_calibration_quality();
            let generated = self.fix_rule_generator.try_generate_rules(phi, calibration);
            if !generated.is_empty() {
                tracing::info!(
                    target: "symthaea::coding_agent",
                    count = generated.len(),
                    phi = phi,
                    calibration = calibration,
                    "FixRuleGenerator hypothesized new fix rule(s) (observation-only, not applied)"
                );
                self.observations.push(format!(
                    "Self-modification: hypothesized {} new fix rule(s) (not auto-applied)",
                    generated.len()
                ));
            }
        }
    }

    /// Current MAGI calibration quality (1 - running Brier score), used only to
    /// gate `FixRuleGenerator::try_generate_rules()`. Without a live orchestrator
    /// (`magi_bridge` unpopulated), this returns a neutral 0.5, which sits below
    /// `FixRuleGenerator`'s default 0.8 calibration-gate threshold — so rule
    /// generation naturally stays inert without real calibration evidence.
    #[cfg(feature = "code_generation")]
    fn current_calibration_quality(&self) -> f32 {
        self.magi_bridge
            .as_ref()
            .map(|bridge| (1.0 - bridge.stats().running_brier as f32).clamp(0.0, 1.0))
            .unwrap_or(0.5)
    }

    #[cfg(not(feature = "code_generation"))]
    fn current_calibration_quality(&self) -> f32 {
        0.5
    }

    /// Synchronously call the async dispatcher.
    pub(super) fn block_on_dispatch(
        dispatcher: &mut IntelligentDispatcher,
        prompt: &str,
        params: &GenerationParams,
        epistemic: EpistemicStatus,
        prediction_error: f64,
        phi: f64,
    ) -> DispatchResult {
        // Try existing tokio runtime first, fall back to a temporary one
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| {
                handle.block_on(dispatcher.generate(
                    prompt,
                    params,
                    epistemic,
                    prediction_error,
                    phi,
                ))
            }),
            Err(_) => {
                // No runtime available — create a lightweight current-thread runtime
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("failed to create tokio runtime for code generation");
                rt.block_on(dispatcher.generate(prompt, params, epistemic, prediction_error, phi))
            }
        }
    }

    /// Attempt generation via the unified `CodeOrchestrator` (native CodeGenerator +
    /// CodeAlgebra analogy + LLM fallback, each accepted only after compiler/test
    /// verification). Only active when `config.use_orchestrator` is set — this is
    /// the real, live call site closing the wiring gap between `CodeOrchestrator`/
    /// `MagiCodeBridge` and `CodingAgent`, without changing default behavior
    /// (`IntelligentDispatcher` remains the default route, `use_orchestrator`
    /// defaults to `false`).
    ///
    /// Returns `true` if the orchestrator produced accepted code (already written
    /// to disk and recorded in `generated_code`/`generation_tiers`) — the caller
    /// should skip the legacy dispatch path for this iteration. Returns `false` to
    /// fall through to `IntelligentDispatcher` when the orchestrator is disabled or
    /// did not produce an accepted candidate.
    #[cfg(feature = "code_generation")]
    pub(super) fn try_orchestrator_generation(&mut self) -> bool {
        if self.orchestrator.is_none() {
            return false;
        }

        let target = self.resolve_target_file();
        let name = target
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("generated")
            .to_string();
        let language = self.target_language().to_string();
        let current_phi = self.phi_trace.last().copied().unwrap_or(0.5);

        let mut request =
            symthaea_core::synthesis_trait::SynthesisRequest::new(&language, &name, &self.task);
        request.consciousness_level = current_phi;
        // If the task text embeds an explicit signature ("fn name(...) -> Ret"),
        // pass it through — it materially improves native/LLM acceptance odds.
        if let Some(sig) = Self::extract_fn_signature(&self.task) {
            request = request.with_signature(&sig);
        }

        // MAGI world-prediction: predict before generating, resolve immediately
        // after — the orchestrator's own internal compiler/test verification means
        // compiled/tests_passed are both already known from `response.accepted`,
        // so there's no need to defer resolution to the agent's own Testing phase.
        let prediction_id = self.magi_bridge.as_mut().map(|bridge| {
            bridge.predict_generation(&name, "orchestrator", 0.6, "code will compile and verify")
        });

        let response = self
            .orchestrator
            .as_ref()
            .expect("checked is_none above")
            .synthesize(&request);

        if let Some(id) = prediction_id {
            if let Some(ref mut bridge) = self.magi_bridge {
                bridge.resolve_prediction(&id, response.accepted, Some(response.accepted));
            }
        }

        if !response.accepted || response.source.trim().is_empty() {
            self.observations.push(format!(
                "Orchestrator: no accepted candidate ({})",
                response.narrative.as_deref().unwrap_or("unverified")
            ));
            return false;
        }

        let tier = Self::orchestrator_backend_tier(&response.backend_name);
        self.write_code_to_disk(&target, &response.source);
        self.generated_code = Some(Self::sanitize_generated_code(&Self::strip_code_fences(
            &response.source,
        )));
        self.generation_tiers.push(tier);
        self.native_exhausted = false;
        self.observations.push(format!(
            "Orchestrator accepted via {} (confidence {:.2})",
            response.backend_name, response.confidence
        ));

        tracing::info!(
            target: "symthaea::coding_agent",
            backend = %response.backend_name,
            confidence = response.confidence,
            target = %target.display(),
            "CodeOrchestrator accepted generation"
        );

        true
    }

    /// Map an orchestrator backend name to the existing `BackendTier` telemetry enum.
    #[cfg(feature = "code_generation")]
    pub(super) fn orchestrator_backend_tier(backend_name: &str) -> BackendTier {
        if backend_name.starts_with("LLM:") {
            BackendTier::LocalLlm
        } else if backend_name == "HardwareDriverEmitter" || backend_name == "HdlEmitter" {
            BackendTier::Hardware
        } else {
            // CodeGenerator, CodeAlgebra::analogy, GeodesicSkeleton — all native,
            // compiler-verified, zero-external-call tiers.
            BackendTier::Native
        }
    }

    /// Extract a Rust-style `fn name(...) -> Ret` signature embedded in free text,
    /// if present. Used to enrich orchestrator requests when the task description
    /// already specifies an exact signature.
    #[cfg(feature = "code_generation")]
    pub(super) fn extract_fn_signature(task: &str) -> Option<String> {
        let start = task.find("fn ")?;
        let rest = &task[start..];
        let end = rest.find(['{', '\n'])?;
        let sig = rest[..end].trim_end();
        if sig.contains('(') {
            Some(sig.to_string())
        } else {
            None
        }
    }
}
