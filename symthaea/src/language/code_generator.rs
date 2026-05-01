// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness-Aware Code Generator
//!
//! The generation pipeline: HDC thinks, CfC sequences, LLM translates.
//!
//! # Pipeline
//!
//! ```text
//! CodeIntent
//!     ↓ encode_intent()
//! Intent HV (16,384D)
//!     ↓ find_similar() — query codebase memory
//! Context HVs
//!     ↓ CfC sequencer — plan structure
//! CodePlanStep[]
//!     ↓ emitters — generate source code
//! Source code string
//!     ↓ verifier — round-trip check
//! GeneratedCode (verified)
//! ```

use symthaea_core::hdc::ContinuousHV;

use super::code_intent::{CodeChange, CodeIntent, CodeSpec, CodeTarget};
use super::code_parser::EntityKind;
use super::emitters::{CodeEmitter, NixEmitter, PythonEmitter, RustEmitter};
use crate::consciousness::code_primitives::{
    CodeExecutionResult, CodeOperation, CodePrimitiveExecutor,
};
use crate::dynamics::cfc_code_sequencer::{CfCCodeSequencer, CodePlanStep};
use crate::hdc::code_algebra::CodeAlgebra;
use crate::hdc::code_encoder::CodeHDEncoder;
use crate::hdc::code_memory::CodebaseMemory;
use crate::mind::structured_thought::EpistemicStatus;

/// Basic dataflow information extracted from generated code.
///
/// Tracks variable definitions, uses, and potential issues like unused
/// variables or use-before-definition. This is a lightweight static
/// analysis — not a full CFG, but catches common generation errors.
#[derive(Debug, Clone, Default)]
pub struct DataflowInfo {
    /// Variables defined in the code (name, line)
    pub definitions: Vec<(String, usize)>,
    /// Variables used in the code (name, line)
    pub uses: Vec<(String, usize)>,
    /// Variables defined but never used
    pub unused: Vec<String>,
    /// Variables used before definition
    pub use_before_def: Vec<String>,
}

impl DataflowInfo {
    /// Analyze a Rust source string for basic dataflow.
    pub fn analyze_rust(source: &str) -> Self {
        let mut defs = Vec::new();
        let mut uses = Vec::new();

        for (line_num, line) in source.lines().enumerate() {
            let trimmed = line.trim();

            // Track let bindings
            if let Some(rest) = trimmed.strip_prefix("let ") {
                let rest = rest.strip_prefix("mut ").unwrap_or(rest);
                // Extract variable name (up to : or = or whitespace)
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty() && name != "_" {
                    defs.push((name, line_num));
                }
            }

            // Track function parameter bindings
            if trimmed.starts_with("fn ") || trimmed.starts_with("pub fn ") {
                if let Some(paren_start) = trimmed.find('(') {
                    if let Some(paren_end) = trimmed.find(')') {
                        let params = &trimmed[paren_start + 1..paren_end];
                        for param in params.split(',') {
                            let param = param.trim();
                            let name: String = param
                                .chars()
                                .take_while(|c| c.is_alphanumeric() || *c == '_')
                                .collect();
                            if !name.is_empty()
                                && name != "_"
                                && name != "self"
                                && name != "&self"
                                && name != "mut"
                            {
                                defs.push((name, line_num));
                            }
                        }
                    }
                }
            }

            // Track identifier uses (simple heuristic: word chars in expressions)
            // Skip comment lines and string literals for now
            if !trimmed.starts_with("//") && !trimmed.starts_with("let ") {
                for word in trimmed.split(|c: char| !c.is_alphanumeric() && c != '_') {
                    if !word.is_empty()
                        && word.chars().next().map_or(false, |c| c.is_lowercase())
                        && !is_rust_keyword(word)
                    {
                        uses.push((word.to_string(), line_num));
                    }
                }
            }
        }

        // Compute unused and use-before-def
        let def_names: std::collections::HashSet<_> = defs.iter().map(|(n, _)| n.clone()).collect();
        let use_names: std::collections::HashSet<_> = uses.iter().map(|(n, _)| n.clone()).collect();

        let unused: Vec<String> = def_names.difference(&use_names).cloned().collect();

        let mut use_before_def = Vec::new();
        for (use_name, use_line) in &uses {
            let first_def = defs
                .iter()
                .filter(|(n, _)| n == use_name)
                .map(|(_, l)| *l)
                .min();
            if let Some(def_line) = first_def {
                if *use_line < def_line {
                    use_before_def.push(use_name.clone());
                }
            }
        }

        Self {
            definitions: defs,
            uses,
            unused,
            use_before_def,
        }
    }

    /// Returns true if no issues were found
    pub fn is_clean(&self) -> bool {
        self.use_before_def.is_empty()
    }
}

/// Lightweight control-flow analysis for generated Rust code.
#[derive(Debug, Clone, Default)]
pub struct ControlFlowInfo {
    pub warnings: Vec<String>,
}

impl ControlFlowInfo {
    pub fn analyze_rust(source: &str, return_type: Option<&str>) -> Self {
        let mut warnings = Vec::new();
        let lines: Vec<&str> = source.lines().collect();
        let ret = return_type.unwrap_or("");
        let is_unit_return = ret.is_empty() || ret == "()" || ret == "-> ()";

        // Check for unreachable code after return/break/continue
        let mut prev_was_terminator = false;
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") || trimmed == "}" || trimmed == "{" {
                continue;
            }
            if prev_was_terminator && !trimmed.starts_with('}') && !trimmed.starts_with("//") {
                warnings.push(format!(
                    "UNREACHABLE_CODE: line {} after return/break",
                    i + 1
                ));
            }
            prev_was_terminator = trimmed.starts_with("return ")
                || trimmed == "return;"
                || trimmed.starts_with("break")
                || trimmed.starts_with("continue");
        }

        // Check if-without-else when return type is non-unit
        if !is_unit_return {
            let if_count = lines.iter().filter(|l| l.trim().starts_with("if ")).count();
            let else_count = lines.iter().filter(|l| l.trim().contains("else")).count();
            if if_count > 0 && else_count == 0 {
                warnings.push("MISSING_ELSE: non-unit return but if-without-else".to_string());
            }
        }

        // Check that non-unit functions end with an expression
        if !is_unit_return {
            let last_expr = lines.iter().rev().find(|l| {
                let t = l.trim();
                !t.is_empty() && t != "}" && !t.starts_with("//")
            });
            if let Some(last) = last_expr {
                let t = last.trim();
                if t.ends_with(';') && !t.starts_with("return") {
                    warnings.push(
                        "MISSING_RETURN: last line is statement but return type is non-unit"
                            .to_string(),
                    );
                }
            }
        }

        Self { warnings }
    }

    pub fn is_clean(&self) -> bool {
        self.warnings.is_empty()
    }
}

/// Nested Rust type parser.
#[derive(Debug, Clone, PartialEq)]
pub enum ParsedType {
    Simple(String),
    Generic {
        base: String,
        params: Vec<ParsedType>,
    },
    Tuple(Vec<ParsedType>),
    Reference {
        mutable: bool,
        inner: Box<ParsedType>,
    },
}

impl ParsedType {
    pub fn parse(s: &str) -> Self {
        let s = s.trim();
        if s.starts_with("&mut ") {
            return ParsedType::Reference {
                mutable: true,
                inner: Box::new(Self::parse(&s[5..])),
            };
        }
        if s.starts_with('&') {
            return ParsedType::Reference {
                mutable: false,
                inner: Box::new(Self::parse(&s[1..])),
            };
        }
        if s.starts_with('(') && s.ends_with(')') {
            let parts = split_type_params(&s[1..s.len() - 1]);
            return ParsedType::Tuple(parts.into_iter().map(|p| Self::parse(&p)).collect());
        }
        if let Some(angle) = s.find('<') {
            if s.ends_with('>') {
                let base = s[..angle].to_string();
                let parts = split_type_params(&s[angle + 1..s.len() - 1]);
                return ParsedType::Generic {
                    base,
                    params: parts.into_iter().map(|p| Self::parse(&p)).collect(),
                };
            }
        }
        ParsedType::Simple(s.to_string())
    }

    pub fn base_name(&self) -> &str {
        match self {
            ParsedType::Simple(s) => s,
            ParsedType::Generic { base, .. } => base,
            ParsedType::Tuple(_) => "()",
            ParsedType::Reference { inner, .. } => inner.base_name(),
        }
    }

    pub fn is_collection(&self) -> bool {
        matches!(
            self.base_name(),
            "Vec" | "HashSet" | "BTreeSet" | "VecDeque"
        )
    }

    pub fn is_map(&self) -> bool {
        matches!(self.base_name(), "HashMap" | "BTreeMap")
    }

    pub fn inner_type(&self) -> Option<&ParsedType> {
        match self {
            ParsedType::Generic { params, .. } => params.first(),
            _ => None,
        }
    }
}

fn split_type_params(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let (mut depth, mut paren) = (0i32, 0i32);
    let mut current = String::new();
    for ch in s.chars() {
        match ch {
            '<' => {
                depth += 1;
                current.push(ch);
            }
            '>' => {
                depth -= 1;
                current.push(ch);
            }
            '(' => {
                paren += 1;
                current.push(ch);
            }
            ')' => {
                paren -= 1;
                current.push(ch);
            }
            ',' if depth == 0 && paren == 0 => {
                parts.push(current.trim().to_string());
                current = String::new();
            }
            _ => current.push(ch),
        }
    }
    let last = current.trim().to_string();
    if !last.is_empty() {
        parts.push(last);
    }
    parts
}

/// Maps rustc error patterns to fix hints for error-driven pattern refinement.
#[derive(Debug, Clone)]
pub struct CompileErrorFix {
    pub error_pattern: &'static str,
    pub fix_hint: &'static str,
    pub category: &'static str,
}

pub fn diagnose_compile_error(stderr: &str) -> Vec<CompileErrorFix> {
    let patterns: &[(&str, &str, &str)] = &[
        (
            "expected expression, found `)`",
            "Empty closure body — need concrete expression",
            "empty_closure",
        ),
        (
            "type annotations needed",
            "Add explicit type with turbofish ::<Type>",
            "type_inference",
        ),
        (
            "no method named",
            "Method doesn't exist — check .iter() or different method",
            "wrong_method",
        ),
        (
            "expected one of `,` or `>`",
            "Type parameter syntax error — tuple/generic nesting",
            "type_syntax",
        ),
        (
            "mismatched types",
            "Return type mismatch — check .cloned()/.copied()/conversion",
            "type_mismatch",
        ),
        (
            "cannot borrow",
            "Borrow error — try .clone() or restructure ownership",
            "borrow_error",
        ),
        (
            "use of moved value",
            "Value moved — use .clone() or avoid double use",
            "moved_value",
        ),
        (
            "cannot find value",
            "Variable not in scope — check spelling or add let",
            "undefined_var",
        ),
        (
            "cannot find type",
            "Type not imported — add use or fully qualify",
            "undefined_type",
        ),
        (
            "the trait bound",
            "Missing trait impl — may need derive or implement",
            "missing_trait",
        ),
        (
            "lifetime",
            "Lifetime issue — consider owned types",
            "lifetime",
        ),
    ];
    patterns
        .iter()
        .filter(|(p, _, _)| stderr.contains(p))
        .map(|(p, h, c)| CompileErrorFix {
            error_pattern: p,
            fix_hint: h,
            category: c,
        })
        .collect()
}

fn is_rust_keyword(word: &str) -> bool {
    matches!(
        word,
        "fn" | "let"
            | "mut"
            | "if"
            | "else"
            | "for"
            | "in"
            | "while"
            | "loop"
            | "match"
            | "return"
            | "break"
            | "continue"
            | "pub"
            | "use"
            | "mod"
            | "struct"
            | "enum"
            | "trait"
            | "impl"
            | "self"
            | "super"
            | "crate"
            | "as"
            | "ref"
            | "true"
            | "false"
            | "type"
            | "where"
            | "async"
            | "await"
            | "move"
            | "dyn"
            | "const"
            | "static"
            | "extern"
            | "unsafe"
            | "iter"
            | "map"
            | "filter"
            | "collect"
            | "clone"
            | "len"
            | "push"
            | "new"
            | "some"
            | "none"
            | "ok"
            | "err"
            | "vec"
            | "format"
            | "println"
            | "assert"
            | "assert_eq"
            | "todo"
            | "unimplemented"
            | "unwrap"
            | "expect"
            | "cloned"
            | "into"
            | "from"
            | "default"
            | "sum"
            | "product"
            | "min"
            | "max"
            | "sort"
            | "rev"
            | "take"
            | "skip"
            | "zip"
            | "entry"
            | "or_insert"
            | "contains"
            | "is_empty"
            | "to_string"
            | "to_vec"
            | "to_lowercase"
            | "to_uppercase"
            | "trim"
            | "chars"
            | "lines"
            | "split"
            | "join"
            | "replace"
            | "starts_with"
            | "ends_with"
    )
}

/// Result of code generation
#[derive(Debug, Clone)]
pub struct GeneratedCode {
    /// The generated source code
    pub source: String,
    /// Language of the generated code
    pub language: String,
    /// Plan steps that produced this code
    pub plan_steps: Vec<CodePlanStep>,
    /// Epistemic status — how confident we are in this generation
    pub epistemic_status: EpistemicStatus,
    /// Similarity between intent and generated code in HDC space
    pub intent_similarity: f32,
    /// Notes or warnings about the generation
    pub notes: Vec<String>,
    /// Phi score from primitive execution (consciousness integration measure)
    pub phi_score: f32,
    /// Primitives that were composed for this generation
    pub primitives_used: Vec<String>,
    /// Dataflow analysis of generated code
    pub dataflow: Option<DataflowInfo>,
    /// Plan depth metric: fraction of plan steps that produced actual code (0.0-1.0).
    /// Low values indicate the planner is generating steps that the emitter can't use.
    /// Can be fed to FEP as a learning signal (plan_gap = 1.0 - plan_coverage).
    pub plan_coverage: f32,
}

/// Context provided to the code generator
pub struct CodeContext<'a> {
    /// The codebase memory for pattern retrieval
    pub memory: Option<&'a CodebaseMemory>,
    /// Additional context HVs (e.g., from conversation)
    pub context_hvs: Vec<ContinuousHV>,
    /// Additional source files for context
    pub source_files: Vec<(String, String)>,
    /// Past successful code generations (from episodic memory).
    /// Each entry is (purpose, source_code) for few-shot context.
    pub past_examples: Vec<(String, String)>,
    /// MCTS plan confidence from the reasoning engine (0.0-1.0).
    /// When > 0, modulates the CfC sequencer's completion threshold —
    /// higher MCTS confidence means more ambitious plans.
    pub mcts_plan_confidence: f32,
    /// Error hints from the CodingExperienceStore: (error_pattern, fix_hint).
    /// Populated by callers who have access to the experience store.
    /// Used during auto-fix retry and to inform native generation.
    pub error_hints: Vec<(String, String)>,
    /// HDC-encoded diagnostics for high-precision neuro-symbolic repair.
    /// These are produced by DiagnosticHDEncoder and capture the "geometry" of compiler errors.
    pub diagnostic_hvs: Vec<ContinuousHV>,
    /// Learned code template from the CodingExperienceStore.
    /// If a similar task was previously completed via LLM, the template
    /// is injected here so native generation can use it directly.
    pub learned_template: Option<String>,
}

impl<'a> Default for CodeContext<'a> {
    fn default() -> Self {
        Self {
            memory: None,
            context_hvs: Vec::new(),
            source_files: Vec::new(),
            past_examples: Vec::new(),
            mcts_plan_confidence: 0.0,
            error_hints: Vec::new(),
            diagnostic_hvs: Vec::new(),
            learned_template: None,
        }
    }
}

/// The main code generation engine
pub struct CodeGenerator {
    encoder: CodeHDEncoder,
    algebra: CodeAlgebra,
    sequencer: CfCCodeSequencer,
    rust_emitter: RustEmitter,
    python_emitter: PythonEmitter,
    nix_emitter: NixEmitter,
    /// Primitive executor for consciousness-aware code operations
    primitive_executor: CodePrimitiveExecutor,
}

impl CodeGenerator {
    /// Create a new code generator
    pub fn new(encoder: CodeHDEncoder) -> Self {
        let dim = encoder.dim();
        let algebra = CodeAlgebra::new(CodeHDEncoder::new(dim));
        let sequencer = CfCCodeSequencer::new(
            crate::dynamics::cfc_code_sequencer::CfCCodeSequencerConfig {
                hdc_dim: dim,
                ..Default::default()
            },
        );
        let primitive_executor = CodePrimitiveExecutor::new(dim);

        Self {
            encoder,
            algebra,
            sequencer,
            rust_emitter: RustEmitter,
            python_emitter: PythonEmitter,
            nix_emitter: NixEmitter,
            primitive_executor,
        }
    }

    /// Map a CodeIntent to the appropriate CodeOperation for primitive selection
    fn intent_to_operation(intent: &CodeIntent) -> CodeOperation {
        match intent {
            CodeIntent::Create { .. } => CodeOperation::Generate,
            CodeIntent::Modify { .. } => CodeOperation::Modify,
            CodeIntent::Explain { .. } => CodeOperation::Explain,
            CodeIntent::Find { .. } => CodeOperation::FindSimilar,
            CodeIntent::Refactor { .. } => CodeOperation::Refactor,
            CodeIntent::Debug { .. } => CodeOperation::Debug,
        }
    }

    /// Execute primitives for a code operation and return the result
    fn execute_primitives(&self, intent: &CodeIntent) -> CodeExecutionResult {
        let op = Self::intent_to_operation(intent);
        self.primitive_executor.execute(op)
    }

    /// Create with default dimension
    pub fn with_default_dim() -> Self {
        Self::new(CodeHDEncoder::new(512))
    }

    /// Get the encoder
    pub fn encoder(&self) -> &CodeHDEncoder {
        &self.encoder
    }

    /// Get the code algebra for external similarity/analogy queries
    pub fn algebra(&self) -> &CodeAlgebra {
        &self.algebra
    }

    /// Generate code from an intent
    ///
    /// This is the main entry point for consciousness-aware code generation.
    /// Primitives are executed first to measure integration (Phi), then the
    /// specific generation path is taken.
    pub fn generate(&self, intent: &CodeIntent, context: &CodeContext) -> GeneratedCode {
        // Execute primitives first - this gives us Phi and the composed primitive HV
        let primitive_result = self.execute_primitives(intent);

        match intent {
            CodeIntent::Create { target, spec } => {
                self.generate_create(target, spec, context, &primitive_result)
            }
            CodeIntent::Modify { target, changes } => {
                self.generate_modify(target, changes, context, &primitive_result)
            }
            CodeIntent::Explain { target, depth } => {
                self.generate_explanation(target, *depth, context, &primitive_result)
            }
            CodeIntent::Find { pattern_hv, scope } => {
                self.generate_search_result(pattern_hv, scope, context, &primitive_result)
            }
            CodeIntent::Refactor { target, strategy } => {
                self.generate_refactor(target, strategy, context, &primitive_result)
            }
            CodeIntent::Debug { target, symptoms } => {
                self.generate_debug(target, symptoms, context, &primitive_result)
            }
        }
    }

    /// Get the primitive executor for direct access
    pub fn primitive_executor(&self) -> &CodePrimitiveExecutor {
        &self.primitive_executor
    }

    /// Check if a generated result needs LLM completion (contains `todo!()` or `unimplemented!()`).
    pub fn needs_llm_completion(result: &GeneratedCode) -> bool {
        result.source.contains("todo!(") || result.source.contains("unimplemented!(")
    }

    /// Build an LLM prompt to complete code that has `todo!()` placeholders.
    ///
    /// Returns a prompt string suitable for sending to any LLM backend.
    /// The prompt includes the generated scaffold, the spec, error hints,
    /// and instructions to replace `todo!()` with real implementations.
    pub fn build_llm_completion_prompt(spec: &CodeSpec, result: &GeneratedCode) -> String {
        let mut prompt = String::new();
        prompt.push_str("Complete the following Rust code by replacing all `todo!()` ");
        prompt.push_str("and `unimplemented!()` macros with real implementations.\n\n");
        prompt.push_str(&format!("## Purpose\n{}\n\n", spec.purpose));
        if let Some(ref sig) = spec.signature {
            prompt.push_str(&format!("## Signature\n{}\n\n", sig));
        }
        if !spec.examples.is_empty() {
            prompt.push_str("## Examples\n");
            for (input, output) in &spec.examples {
                prompt.push_str(&format!("  {} => {}\n", input, output));
            }
            prompt.push_str("\n");
        }
        if !spec.constraints.is_empty() {
            prompt.push_str("## Constraints\n");
            for c in &spec.constraints {
                prompt.push_str(&format!("  - {}\n", c));
            }
            prompt.push_str("\n");
        }
        // Include error hints from experience store
        for note in &result.notes {
            if note.starts_with("ERROR_HINT") {
                prompt.push_str(&format!("## Known Issue\n{}\n\n", note));
            }
        }
        let repo_notes: Vec<&String> = result
            .notes
            .iter()
            .filter(|note| note.starts_with("REPO_SNIPPET"))
            .collect();
        if !repo_notes.is_empty() {
            prompt.push_str("## Repository Context\n");
            for note in repo_notes {
                prompt.push_str(note);
                prompt.push_str("\n\n");
            }
        }
        prompt.push_str("## Code to Complete\n```rust\n");
        prompt.push_str(&result.source);
        prompt.push_str("\n```\n\n");
        prompt.push_str("Return ONLY the completed Rust code, no explanations.\n");
        prompt
    }

    /// Generate code with automatic LLM fallback for `todo!()` placeholders.
    ///
    /// Pipeline:
    /// 1. Native generation via HDC+CfC emitters
    /// 2. If result contains `todo!()`, build LLM completion prompt
    /// 3. Return (result, Option<llm_prompt>) — caller dispatches to LLM backend
    ///
    /// This closes the two-tier architecture loop: native patterns handle ~70%
    /// of functions, LLM handles the rest, and successful LLM outputs get
    /// distilled back into Broca SSM for future native generation.
    pub fn generate_with_fallback(
        &self,
        intent: &CodeIntent,
        context: &CodeContext,
    ) -> (GeneratedCode, Option<String>) {
        let result = self.generate(intent, context);

        if Self::needs_llm_completion(&result) {
            // Extract spec from intent for prompt building
            let prompt = match intent {
                CodeIntent::Create { spec, .. } => {
                    Some(Self::build_llm_completion_prompt(spec, &result))
                }
                _ => None, // LLM fallback only for Create intents
            };
            (result, prompt)
        } else {
            (result, None)
        }
    }

    /// Extract an SSM distillation target from a successful native generation.
    ///
    /// Returns `Some((purpose_embedding, source_code, quality))` when the generated
    /// code is high-confidence native (no `todo!()`, high intent similarity).
    /// The cognitive loop can feed these to Broca's SSM as training targets,
    /// teaching the language model to reproduce native-quality code patterns.
    pub fn distillation_target(
        &self,
        spec: &CodeSpec,
        result: &GeneratedCode,
    ) -> Option<(ContinuousHV, String, f32)> {
        // Only distill native Rust code that was generated with high confidence
        if spec.language != "rust" {
            return None;
        }
        if result.source.contains("todo!") || result.source.contains("unimplemented!") {
            return None; // LLM fallback — not a native pattern
        }
        if result.intent_similarity < 0.5 {
            return None; // Low confidence generation
        }
        // Check dataflow cleanliness
        if let Some(ref df) = result.dataflow {
            if !df.is_clean() {
                return None; // Has use-before-def issues
            }
        }
        // Check no type warnings
        if result.notes.iter().any(|n| n.starts_with("TYPE_MISMATCH")) {
            return None;
        }

        let purpose_hv = self.encode_spec(spec);
        // Quality signal: intent_similarity weighted by phi
        let quality = result.intent_similarity * 0.7 + result.phi_score.min(1.0) * 0.3;

        Some((purpose_hv, result.source.clone(), quality))
    }

    /// Validate type consistency between spec signature and generated code.
    ///
    /// Checks:
    /// - Return type in generated code matches spec's return type
    /// - Parameter types referenced in body are compatible with spec
    /// - Iterator chain return types (e.g., .sum() → numeric, .collect() → Vec)
    ///
    /// Returns a list of type warnings (empty = all good).
    pub fn validate_types(spec: &CodeSpec, source: &str) -> Vec<String> {
        let mut warnings = Vec::new();

        let sig = match spec.signature.as_deref() {
            Some(s) => s,
            None => return warnings,
        };

        let parsed = match super::emitters::parse_rust_signature_pub(sig) {
            Some(p) => p,
            None => return warnings,
        };

        let ret_type = parsed.return_type.as_deref().unwrap_or("");

        // Check iterator chain return type consistency
        if source.contains(".collect()") || source.contains(".collect::<") {
            if !ret_type.contains("Vec")
                && !ret_type.contains("HashMap")
                && !ret_type.contains("HashSet")
                && !ret_type.contains("String")
                && !ret_type.is_empty()
            {
                warnings.push(format!(
                    "TYPE_MISMATCH: .collect() used but return type is '{}' (expected Vec/HashMap/HashSet/String)",
                    ret_type
                ));
            }
        }

        if source.contains(".sum()") {
            if !ret_type.contains("i32")
                && !ret_type.contains("i64")
                && !ret_type.contains("u32")
                && !ret_type.contains("u64")
                && !ret_type.contains("f32")
                && !ret_type.contains("f64")
                && !ret_type.contains("usize")
                && !ret_type.is_empty()
            {
                warnings.push(format!(
                    "TYPE_MISMATCH: .sum() used but return type is '{}' (expected numeric)",
                    ret_type
                ));
            }
        }

        if source.contains(".len()") && !ret_type.is_empty() {
            if !ret_type.contains("usize") && !ret_type.contains("u32") && !ret_type.contains("i32")
            {
                warnings.push(format!(
                    "TYPE_HINT: .len() returns usize but return type is '{}'",
                    ret_type
                ));
            }
        }

        // Check if bool operations match bool return type
        let has_bool_ops = source.contains(" == ")
            || source.contains(" != ")
            || source.contains(" > ")
            || source.contains(" < ")
            || source.contains(" >= ")
            || source.contains(" <= ");
        if has_bool_ops && ret_type == "bool" {
            // Good: bool ops with bool return — no warning
        } else if has_bool_ops
            && !ret_type.contains("bool")
            && !ret_type.is_empty()
            && !source.contains("if ")
            && !source.contains("filter")
            && !source.contains("while")
        {
            // Bool ops but non-bool return and not in a conditional — might be OK
            // (e.g., inside .filter()), so only warn if it looks like the final expression
            if source.lines().last().map_or(false, |l| {
                let t = l.trim();
                (t.contains("==") || t.contains("!=") || t.contains("> ") || t.contains("< "))
                    && !t.starts_with("//")
            }) {
                warnings.push(format!(
                    "TYPE_MISMATCH: final expression is boolean but return type is '{}'",
                    ret_type
                ));
            }
        }

        // Phase 5: Borrow-check heuristics
        let borrow_warnings = Self::check_borrow_heuristics(source);
        warnings.extend(borrow_warnings);

        // Phase 5: Iterator type consistency
        let iter_warnings = Self::check_iterator_consistency(source, ret_type);
        warnings.extend(iter_warnings);

        warnings
    }

    /// Phase 5: Lightweight borrow-check heuristics.
    ///
    /// Catches common Rust borrow errors without running rustc:
    /// - Use after move (variable used after being passed by value)
    /// - Mutable + immutable borrow conflicts (simple cases)
    /// - Return of reference to local variable
    fn check_borrow_heuristics(source: &str) -> Vec<String> {
        let mut warnings = Vec::new();
        let lines: Vec<&str> = source.lines().collect();

        // Track variables that have been moved (passed to functions/methods consuming ownership)
        let mut moved_vars: Vec<String> = Vec::new();

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Skip comments
            if trimmed.starts_with("//") {
                continue;
            }

            // Detect moves: into_iter(), sort() on owned, passing to function that takes T (not &T)
            if trimmed.contains(".into_iter()") || trimmed.contains(".into_sorted()") {
                // Extract the variable name before the method call
                if let Some(var) = trimmed.split('.').next() {
                    let var = var.trim().trim_start_matches("let ").trim();
                    if !var.is_empty() && var.chars().all(|c| c.is_alphanumeric() || c == '_') {
                        moved_vars.push(var.to_string());
                    }
                }
            }

            // Check if a moved variable is used again
            for moved_var in &moved_vars {
                if i > 0 && trimmed.contains(moved_var.as_str()) {
                    // Only warn if it's a USE, not the move itself
                    let prev_lines = &lines[..i];
                    let was_moved_before = prev_lines.iter().any(|l| {
                        l.contains(&format!("{}.into_iter()", moved_var))
                            || l.contains(&format!("{}.into_sorted()", moved_var))
                    });
                    if was_moved_before && !trimmed.contains("into_iter") {
                        warnings.push(format!(
                            "BORROW_HINT(line {}): '{}' may have been moved earlier (into_iter/sort consumes ownership)",
                            i + 1,
                            moved_var
                        ));
                    }
                }
            }

            // Detect return of reference to local
            if trimmed.starts_with("&")
                && !trimmed.contains("&self")
                && !trimmed.contains("&mut self")
            {
                // Check if this looks like a return of a local ref
                let is_last_expr = i == lines.len() - 2; // second to last (before closing brace)
                if is_last_expr && !trimmed.contains("&str") && !trimmed.contains("&[") {
                    warnings.push(format!(
                        "BORROW_HINT(line {}): returning a reference — ensure the referent outlives the return",
                        i + 1
                    ));
                }
            }

            // Detect simultaneous mutable and immutable borrows (simple heuristic)
            if trimmed.contains("&mut ") {
                // Check if an immutable borrow of the same variable exists nearby
                if let Some(var_after_mut) = trimmed.split("&mut ").nth(1) {
                    let var_name: String = var_after_mut
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect();
                    if !var_name.is_empty() {
                        // Check surrounding lines for immutable borrow
                        let window_start = i.saturating_sub(3);
                        let window_end = (i + 4).min(lines.len());
                        for j in window_start..window_end {
                            if j != i {
                                let other = lines[j].trim();
                                if other.contains(&format!("&{}", var_name))
                                    && !other.contains("&mut")
                                    && !other.starts_with("//")
                                {
                                    warnings.push(format!(
                                        "BORROW_HINT(lines {}-{}): possible simultaneous &mut and & borrow of '{}'",
                                        i + 1,
                                        j + 1,
                                        var_name
                                    ));
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        warnings
    }

    /// Phase 5: Check iterator chain type consistency.
    ///
    /// Verifies that iterator adapters produce types compatible with the
    /// return type and with each other.
    fn check_iterator_consistency(source: &str, ret_type: &str) -> Vec<String> {
        let mut warnings = Vec::new();

        // .iter() produces &T, but .collect::<Vec<T>>() needs owned T
        // Common fix: .cloned() or .copied() before .collect()
        if source.contains(".iter()") && source.contains(".collect()") {
            if !source.contains(".cloned()")
                && !source.contains(".copied()")
                && !source.contains(".map(")
                && !source.contains("to_vec()")
                && !source.contains("to_string()")
            {
                // Check if return type expects owned values
                if ret_type.contains("Vec<") && !ret_type.contains("Vec<&") {
                    warnings.push(
                        "TYPE_HINT: .iter().collect() produces Vec<&T> but return expects Vec<T>. Consider .cloned() or .copied()".to_string()
                    );
                }
            }
        }

        // .filter() preserves reference level — if source has .filter() without .cloned()
        // and collects to Vec<T>, that's a type error
        if source.contains(".filter(") && source.contains(".collect()") {
            if source.contains(".iter().filter(")
                && !source.contains(".cloned()")
                && !source.contains(".copied()")
            {
                if ret_type.contains("Vec<") && !ret_type.contains("Vec<&") {
                    warnings.push(
                        "TYPE_HINT: .iter().filter().collect() produces Vec<&&T>. Add .cloned() before .collect()".to_string()
                    );
                }
            }
        }

        // .sum() requires Sum trait — usually works on numeric types
        if source.contains(".sum()") && source.contains(".iter()") {
            if !source.contains("::<") {
                // May need turbofish: .sum::<i32>()
                if !ret_type.is_empty() && !ret_type.contains("()") {
                    warnings.push(format!(
                        "TYPE_HINT: .iter().sum() may need turbofish .sum::<{}>() for type inference",
                        ret_type
                    ));
                }
            }
        }

        warnings
    }

    /// Generate ONLY the test assertions for a spec, without the implementation.
    ///
    /// Used for test-first generation: produce behavioral expectations before
    /// generating the code. Tests come from spec examples + purpose-based auto-tests.
    pub fn generate_tests_only(&self, spec: &CodeSpec) -> Option<String> {
        let _emitter: &dyn CodeEmitter = match spec.language.as_str() {
            "rust" => &self.rust_emitter,
            "python" => &self.python_emitter,
            "nix" => &self.nix_emitter,
            _ => &self.rust_emitter,
        };

        if spec.language == "rust" {
            // Generate a test module with a stub function signature
            let sig = spec.signature.as_deref()?;
            let parsed = super::emitters::parse_rust_signature_pub(sig)?;

            let mut tests = Vec::new();

            // From explicit examples
            for (i, (input, output)) in spec.examples.iter().enumerate() {
                tests.push(format!("    #[test]"));
                tests.push(format!("    fn test_example_{}() {{", i));
                tests.push(format!("        assert_eq!({}, {});", input, output));
                tests.push(format!("    }}"));
            }

            // From auto-generation (example-based)
            let auto = super::emitters::generate_auto_tests_pub(
                &parsed.name,
                &spec.purpose,
                Some(&parsed),
            );
            for (i, assertion) in auto.iter().enumerate() {
                tests.push(format!("    #[test]"));
                tests.push(format!("    fn test_auto_{}() {{", i));
                tests.push(format!("        {}", assertion));
                tests.push(format!("    }}"));
            }

            // From property-based generation (invariant tests)
            let props = super::emitters::generate_property_tests_pub(
                &parsed.name,
                &spec.purpose,
                Some(&parsed),
            );
            for (i, assertion) in props.iter().enumerate() {
                tests.push(format!("    #[test]"));
                tests.push(format!("    fn test_property_{}() {{", i));
                tests.push(format!("        {}", assertion));
                tests.push(format!("    }}"));
            }

            if tests.is_empty() {
                return None;
            }

            Some(format!(
                "#[cfg(test)]\nmod tests {{\n    use super::*;\n\n{}\n}}",
                tests.join("\n")
            ))
        } else {
            None // Python test-first not yet supported
        }
    }

    /// Generate new code from a specification
    fn generate_create(
        &self,
        _target: &CodeTarget,
        spec: &CodeSpec,
        context: &CodeContext,
        primitive_result: &CodeExecutionResult,
    ) -> GeneratedCode {
        // 1. Encode intent as HDC
        let intent_hv = self.encode_spec(spec);

        // 2. Query codebase for similar patterns
        let similar_hvs = self.find_similar_context(&intent_hv, context);
        let similar_refs: Vec<&ContinuousHV> = similar_hvs.iter().collect();

        // 3. CfC plan (informed by primitive composition + MCTS confidence)
        // Use purpose text for direct keyword-based pattern detection (more reliable
        // than HDC similarity with byte-hash encoding)
        let mut plan =
            self.sequencer
                .plan_structure_with_purpose(&intent_hv, &similar_refs, &spec.purpose);

        // If MCTS confidence is high, boost low-confidence plan steps
        // (the reasoning engine has already vetted this direction)
        if context.mcts_plan_confidence > 0.5 {
            let boost = (context.mcts_plan_confidence - 0.5) * 0.4; // up to 0.2 boost
            for step in &mut plan {
                step.confidence = (step.confidence + boost).min(1.0);
            }
        }

        // 3.5. Check for learned template from CodingExperienceStore
        // If a previous LLM-generated solution exists for a similar task,
        // use it directly instead of the emitter's pattern library.
        let source = if let Some(ref template) = context.learned_template {
            template.clone()
        } else {
            // 4. Emit code using language-specific emitter
            self.emit_from_plan(&plan, spec, &spec.language)
        };

        // 5. Compute intent similarity (combine plan coverage + primitive phi + MCTS)
        let intent_similarity = if !source.is_empty() {
            let coverage = plan.len() as f32 / 5.0;
            let mcts_bonus = context.mcts_plan_confidence * 0.1; // up to 0.1 bonus
                                                                 // Weight: 60% plan coverage, 30% primitive phi, 10% MCTS confidence
            (coverage * 0.6 + primitive_result.phi * 0.3 + mcts_bonus).min(1.0)
        } else {
            0.0
        };

        // Include AST/HDC-retrieved source snippets as notes for LLM context.
        let mut notes = Vec::new();
        for (label, snippet) in &context.source_files {
            notes.push(format!(
                "REPO_SNIPPET({}):\n{}",
                label,
                snippet.chars().take(4_000).collect::<String>()
            ));
        }

        // Include past successful examples as notes for LLM few-shot context
        for (purpose, code) in &context.past_examples {
            notes.push(format!("PAST_EXAMPLE({}):\n{}", purpose, code));
        }

        // Include error hints from CodingExperienceStore for informed generation
        for (pattern, hint) in &context.error_hints {
            notes.push(format!("ERROR_HINT({}): {}", pattern, hint));
        }

        // Dataflow analysis for Rust code
        let dataflow = if spec.language == "rust" && !source.contains("todo!") {
            let df = DataflowInfo::analyze_rust(&source);
            if !df.use_before_def.is_empty() {
                notes.push(format!(
                    "DATAFLOW_WARNING: possible use-before-def: {:?}",
                    df.use_before_def
                ));
            }
            Some(df)
        } else {
            None
        };

        // Type consistency validation
        if spec.language == "rust" && !source.contains("todo!") {
            let type_warnings = Self::validate_types(spec, &source);
            for w in type_warnings {
                notes.push(w);
            }
        }

        // Control-flow validation
        if spec.language == "rust" && !source.contains("todo!") {
            let ret_type = spec
                .signature
                .as_deref()
                .and_then(|s| super::emitters::parse_rust_signature_pub(s))
                .and_then(|p| p.return_type);
            let cf = ControlFlowInfo::analyze_rust(&source, ret_type.as_deref());
            for w in cf.warnings {
                notes.push(w);
            }
        }

        // Compute plan coverage: how many plan steps produced visible code artifacts.
        // A DefineFunction step that results in a fn declaration counts; a DefineStruct
        // step that doesn't produce struct code doesn't count.
        let plan_coverage = if plan.is_empty() {
            0.0
        } else {
            use crate::dynamics::cfc_code_sequencer::PlanAction;
            let covered = plan
                .iter()
                .filter(|step| match step.action {
                    PlanAction::DefineFunction => source.contains("fn "),
                    PlanAction::DefineStruct => source.contains("struct "),
                    PlanAction::DefineTrait => source.contains("trait "),
                    PlanAction::AddField => source.contains(':'),
                    PlanAction::AddMethod => source.contains("fn "),
                    PlanAction::ImplTrait => source.contains("impl "),
                    PlanAction::AddImport => source.contains("use "),
                    PlanAction::AddDocumentation => source.contains("///"),
                    PlanAction::AddErrorHandling => {
                        source.contains("Result") || source.contains("?")
                    }
                    PlanAction::Complete => true, // always counts
                    _ => !source.is_empty(),      // conservative: if we have code, count it
                })
                .count();
            covered as f32 / plan.len() as f32
        };

        GeneratedCode {
            source,
            language: spec.language.clone(),
            plan_steps: plan,
            epistemic_status: spec.epistemic_status,
            intent_similarity,
            notes,
            phi_score: primitive_result.phi,
            primitives_used: primitive_result
                .primitives
                .iter()
                .map(|p| p.primitive.name.clone())
                .collect(),
            dataflow,
            plan_coverage,
        }
    }

    /// Generate code modification by applying structural transforms.
    ///
    /// When existing source code is available (via target.path or context),
    /// performs real transformations. Otherwise generates a modification spec.
    fn generate_modify(
        &self,
        target: &CodeTarget,
        changes: &[CodeChange],
        context: &CodeContext,
        primitive_result: &CodeExecutionResult,
    ) -> GeneratedCode {
        let mut notes = Vec::new();
        let lang = target
            .language
            .clone()
            .unwrap_or_else(|| "rust".to_string());

        // Try to find existing source code in context
        let existing_source = context
            .source_files
            .iter()
            .find(|(path, _)| {
                path.contains(&target.name)
                    || target.path.as_ref().map_or(false, |p| path.contains(p))
            })
            .map(|(_, src)| src.as_str());

        let source = if let Some(existing) = existing_source {
            // Apply modifications to existing source
            Self::apply_modifications(existing, &target.name, changes, &lang)
        } else {
            // No existing source — generate a modification spec
            Self::generate_modification_spec(&target.name, changes, &lang)
        };

        notes.push(format!("Modification applied to `{}`", target.name));

        GeneratedCode {
            source,
            language: lang,
            plan_steps: Vec::new(),
            epistemic_status: EpistemicStatus::Probable,
            intent_similarity: 0.6,
            notes,
            phi_score: primitive_result.phi,
            primitives_used: primitive_result
                .primitives
                .iter()
                .map(|p| p.primitive.name.clone())
                .collect(),
            dataflow: None,
            plan_coverage: 0.0,
        }
    }

    /// Apply structural modifications to existing Rust source code.
    fn apply_modifications(
        source: &str,
        func_name: &str,
        changes: &[CodeChange],
        _lang: &str,
    ) -> String {
        let mut result = source.to_string();

        for change in changes {
            match change {
                CodeChange::AddParameter { name, param_type } => {
                    // Find function signature and add parameter
                    let fn_pattern = format!("fn {}(", func_name);
                    if let Some(pos) = result.find(&fn_pattern) {
                        let after = &result[pos + fn_pattern.len()..];
                        if let Some(close_paren) = after.find(')') {
                            let params = &after[..close_paren];
                            let new_param = format!("{}: {}", name, param_type);
                            let new_params = if params.trim().is_empty() {
                                new_param
                            } else {
                                format!("{}, {}", params, new_param)
                            };
                            let insert_pos = pos + fn_pattern.len();
                            result = format!(
                                "{}{}{}",
                                &result[..insert_pos],
                                new_params,
                                &result[insert_pos + close_paren..]
                            );
                        }
                    }
                }
                CodeChange::ChangeReturnType { new_type } => {
                    // Find "-> OldType {" and replace return type
                    let fn_pattern = format!("fn {}(", func_name);
                    if let Some(fn_pos) = result.find(&fn_pattern) {
                        let after_fn = &result[fn_pos..];
                        if let Some(arrow_pos) = after_fn.find("->") {
                            let after_arrow = &after_fn[arrow_pos + 2..];
                            if let Some(brace_pos) = after_arrow.find('{') {
                                let abs_arrow = fn_pos + arrow_pos + 2;
                                let abs_brace = abs_arrow + brace_pos;
                                result = format!(
                                    "{} {} {}",
                                    &result[..abs_arrow],
                                    new_type,
                                    &result[abs_brace..]
                                );
                            }
                        }
                    }
                }
                CodeChange::Rename { new_name } => {
                    // Replace all occurrences of the function name
                    result = result.replace(func_name, new_name);
                }
                CodeChange::AddDocumentation { content } => {
                    // Add doc comment before the function
                    let fn_pattern = format!("fn {}(", func_name);
                    if let Some(pos) = result.find(&fn_pattern) {
                        // Find start of line
                        let line_start = result[..pos].rfind('\n').map(|p| p + 1).unwrap_or(0);
                        let indent = &result[line_start..pos];
                        let whitespace: String =
                            indent.chars().take_while(|c| c.is_whitespace()).collect();
                        result.insert_str(line_start, &format!("{}/// {}\n", whitespace, content));
                    }
                }
                CodeChange::AddErrorHandling {
                    strategy: _strategy,
                } => {
                    // Wrap return type in Result if not already
                    let fn_pattern = format!("fn {}(", func_name);
                    if let Some(fn_pos) = result.find(&fn_pattern) {
                        let after_fn = &result[fn_pos..];
                        if let Some(arrow_pos) = after_fn.find("->") {
                            let after_arrow = &after_fn[arrow_pos + 2..];
                            if let Some(brace_pos) = after_arrow.find('{') {
                                let ret_type = after_arrow[..brace_pos].trim();
                                if !ret_type.starts_with("Result") {
                                    let abs_arrow = fn_pos + arrow_pos + 2;
                                    let abs_brace = abs_arrow + brace_pos;
                                    result = format!(
                                        "{} Result<{}, Box<dyn std::error::Error>> {}",
                                        &result[..abs_arrow],
                                        ret_type,
                                        &result[abs_brace..]
                                    );
                                }
                            }
                        }
                    }
                    // Error handling transformation applied above
                }
                CodeChange::RemoveParameter { name } => {
                    // Find and remove parameter from signature
                    let fn_pattern = format!("fn {}(", func_name);
                    if let Some(pos) = result.find(&fn_pattern) {
                        let after = &result[pos + fn_pattern.len()..];
                        if let Some(close_paren) = after.find(')') {
                            let params = &after[..close_paren];
                            let new_params: Vec<&str> =
                                params.split(',').filter(|p| !p.contains(name)).collect();
                            let insert_pos = pos + fn_pattern.len();
                            result = format!(
                                "{}{}{}",
                                &result[..insert_pos],
                                new_params.join(","),
                                &result[insert_pos + close_paren..]
                            );
                        }
                    }
                }
                CodeChange::Custom { description } => {
                    // Can't auto-apply custom changes — add as comment
                    result = format!("// MODIFICATION: {}\n{}", description, result);
                }
            }
        }

        result
    }

    /// Generate a modification spec when no existing source is available.
    fn generate_modification_spec(func_name: &str, changes: &[CodeChange], _lang: &str) -> String {
        let mut lines = Vec::new();
        lines.push(format!("// Modification plan for `{}`:", func_name));

        for change in changes {
            match change {
                CodeChange::AddParameter { name, param_type } => {
                    lines.push(format!("// - Add parameter `{}: {}`", name, param_type));
                }
                CodeChange::ChangeReturnType { new_type } => {
                    lines.push(format!("// - Change return type to `{}`", new_type));
                }
                CodeChange::Rename { new_name } => {
                    lines.push(format!("// - Rename to `{}`", new_name));
                }
                CodeChange::AddDocumentation { content } => {
                    lines.push(format!("/// {}", content));
                }
                CodeChange::AddErrorHandling { strategy } => {
                    lines.push(format!("// - Add error handling: {}", strategy));
                }
                CodeChange::RemoveParameter { name } => {
                    lines.push(format!("// - Remove parameter `{}`", name));
                }
                CodeChange::Custom { description } => {
                    lines.push(format!("// - {}", description));
                }
            }
        }

        lines.join("\n")
    }

    /// Generate code explanation with structural analysis.
    ///
    /// Brief: one-liner. Standard: structure + location. Detailed: full
    /// breakdown of entities, complexity hints, and algorithm patterns detected
    /// via HDC encoding.
    fn generate_explanation(
        &self,
        target: &CodeTarget,
        depth: super::code_intent::ExplanationDepth,
        context: &CodeContext,
        primitive_result: &CodeExecutionResult,
    ) -> GeneratedCode {
        let mut notes = Vec::new();

        let explanation = match depth {
            super::code_intent::ExplanationDepth::Brief => {
                format!("// `{}` is a {:?}", target.name, target.kind)
            }
            super::code_intent::ExplanationDepth::Standard => {
                let name_hv = self.encoder.encode_name(&target.name);
                let similar = self.find_similar_context(&name_hv, context);
                if !similar.is_empty() {
                    notes.push(format!(
                        "SIMILAR_ENTITIES: {} related patterns found in codebase",
                        similar.len()
                    ));
                }
                format!(
                    "// `{}` is a {:?}\n// Located in: {}",
                    target.name,
                    target.kind,
                    target.path.as_deref().unwrap_or("unknown")
                )
            }
            _ => {
                let name_hv = self.encoder.encode_name(&target.name);
                let similar = self.find_similar_context(&name_hv, context);

                let mut lines = Vec::new();
                lines.push(format!("// === {} ({:?}) ===", target.name, target.kind));
                lines.push(format!(
                    "// Path: {}",
                    target.path.as_deref().unwrap_or("unknown")
                ));
                lines.push(format!(
                    "// Language: {}",
                    target.language.as_deref().unwrap_or("unknown")
                ));

                // Algorithm pattern detection via HDC
                if let Some(pattern) =
                    crate::dynamics::cfc_code_sequencer::CfCCodeSequencer::detect_algorithm_pattern(
                        &name_hv,
                    )
                {
                    lines.push(format!("// Detected algorithm family: {:?}", pattern));
                    notes.push(format!("ALGORITHM_PATTERN:{:?}", pattern));
                }

                // Complexity hints from entity kind
                let complexity_hint = match target.kind {
                    EntityKind::Function => "O(?) — analyze body for loops/recursion",
                    EntityKind::Struct => "data structure — check field count and nesting",
                    EntityKind::Trait => "interface — check implementors for dispatch cost",
                    EntityKind::TraitImpl => "implementation — check method count",
                    EntityKind::Enum => "sum type — check variant count for match exhaustiveness",
                    _ => "unknown entity kind",
                };
                lines.push(format!("// Complexity hint: {}", complexity_hint));

                if !similar.is_empty() {
                    lines.push(format!(
                        "// Related patterns: {} found in codebase context",
                        similar.len()
                    ));
                }

                lines.push(format!("// Phi (integration): {:.3}", primitive_result.phi));

                lines.join("\n")
            }
        };

        GeneratedCode {
            source: explanation,
            language: target
                .language
                .clone()
                .unwrap_or_else(|| "rust".to_string()),
            plan_steps: Vec::new(),
            epistemic_status: EpistemicStatus::Probable,
            intent_similarity: 0.7,
            notes,
            phi_score: primitive_result.phi,
            primitives_used: primitive_result
                .primitives
                .iter()
                .map(|p| p.primitive.name.clone())
                .collect(),
            dataflow: None,
            plan_coverage: 0.0,
        }
    }

    /// Generate search results
    fn generate_search_result(
        &self,
        pattern_hv: &ContinuousHV,
        _scope: &super::code_intent::SearchScope,
        context: &CodeContext,
        primitive_result: &CodeExecutionResult,
    ) -> GeneratedCode {
        let mut results = Vec::new();

        if let Some(memory) = context.memory {
            let matches = memory.query(pattern_hv, 5);
            for m in &matches {
                results.push(format!(
                    "// Match: {} ({:?}) in {} — similarity: {:.3}",
                    m.name,
                    m.kind,
                    m.path.display(),
                    m.similarity
                ));
            }
        }

        if results.is_empty() {
            results.push("// No matches found in codebase memory".to_string());
        }

        GeneratedCode {
            source: results.join("\n"),
            language: "text".to_string(),
            plan_steps: Vec::new(),
            epistemic_status: EpistemicStatus::Probable,
            intent_similarity: 0.6,
            notes: Vec::new(),
            phi_score: primitive_result.phi,
            primitives_used: primitive_result
                .primitives
                .iter()
                .map(|p| p.primitive.name.clone())
                .collect(),
            dataflow: None,
            plan_coverage: 0.0,
        }
    }

    /// Generate refactoring suggestion
    fn generate_refactor(
        &self,
        target: &CodeTarget,
        strategy: &super::code_intent::RefactorStrategy,
        _context: &CodeContext,
        primitive_result: &CodeExecutionResult,
    ) -> GeneratedCode {
        let suggestion = match strategy {
            super::code_intent::RefactorStrategy::ExtractFunction { name } => {
                format!(
                    "// Refactor: Extract function `{}` from `{}`",
                    name, target.name
                )
            }
            super::code_intent::RefactorStrategy::InlineFunction => {
                format!("// Refactor: Inline `{}` at call sites", target.name)
            }
            super::code_intent::RefactorStrategy::Rename { new_name } => {
                format!("// Refactor: Rename `{}` → `{}`", target.name, new_name)
            }
            super::code_intent::RefactorStrategy::ExtractTrait { name } => {
                format!(
                    "// Refactor: Extract trait `{}` from `{}`",
                    name, target.name
                )
            }
            super::code_intent::RefactorStrategy::SplitFunction => {
                format!(
                    "// Refactor: Split `{}` into smaller functions",
                    target.name
                )
            }
            super::code_intent::RefactorStrategy::ConvertPattern { from, to } => {
                format!(
                    "// Refactor: Convert `{}` pattern from `{}` to `{}`",
                    target.name, from, to
                )
            }
        };

        GeneratedCode {
            source: suggestion,
            language: target
                .language
                .clone()
                .unwrap_or_else(|| "rust".to_string()),
            plan_steps: Vec::new(),
            epistemic_status: EpistemicStatus::Uncertain,
            intent_similarity: 0.5,
            notes: vec!["Refactoring skeleton — needs verification".to_string()],
            phi_score: primitive_result.phi,
            primitives_used: primitive_result
                .primitives
                .iter()
                .map(|p| p.primitive.name.clone())
                .collect(),
            dataflow: None,
            plan_coverage: 0.0,
        }
    }

    /// Generate debug analysis
    fn generate_debug(
        &self,
        target: &CodeTarget,
        symptoms: &[String],
        _context: &CodeContext,
        primitive_result: &CodeExecutionResult,
    ) -> GeneratedCode {
        let mut analysis = vec![format!("// Debug analysis for `{}`", target.name)];

        for symptom in symptoms {
            analysis.push(format!("// Symptom: {}", symptom));
        }

        analysis.push("// (Full diagnosis requires LLM organ analysis)".to_string());

        GeneratedCode {
            source: analysis.join("\n"),
            language: target
                .language
                .clone()
                .unwrap_or_else(|| "rust".to_string()),
            plan_steps: Vec::new(),
            epistemic_status: EpistemicStatus::Uncertain,
            intent_similarity: 0.4,
            notes: vec!["Debug skeleton — requires LLM organ for full analysis".to_string()],
            phi_score: primitive_result.phi,
            primitives_used: primitive_result
                .primitives
                .iter()
                .map(|p| p.primitive.name.clone())
                .collect(),
            dataflow: None,
            plan_coverage: 0.0,
        }
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    /// Encode a code spec into an HDC vector
    fn encode_spec(&self, spec: &CodeSpec) -> ContinuousHV {
        if let Some(ref hv) = spec.purpose_hv {
            return hv.clone();
        }

        // Combine name + purpose into an encoding
        let name_hv = self.encoder.encode_name(&spec.name);
        let purpose_hv = self.encoder.encode_name(&spec.purpose);
        let lang_hv = self.encoder.encode_name(&spec.language);

        ContinuousHV::bundle_owned(&[name_hv, purpose_hv, lang_hv])
    }

    /// Find similar patterns from context, including analogy-based retrieval.
    ///
    /// If codebase memory provides at least 2 results, uses CodeAlgebra.analogy()
    /// to find "A:B :: intent:?" patterns — transferring relationships from known
    /// code to the current intent.
    fn find_similar_context(
        &self,
        intent_hv: &ContinuousHV,
        context: &CodeContext,
    ) -> Vec<ContinuousHV> {
        let mut results = Vec::new();

        // Add context HVs directly
        for hv in &context.context_hvs {
            results.push(hv.clone());
        }

        // Query codebase memory if available
        if let Some(memory) = context.memory {
            let matches = memory.query(intent_hv, 5);
            if !matches.is_empty() {
                // Encode matched names as additional context
                let mut match_hvs = Vec::new();
                for m in &matches {
                    let hv = self.encoder.encode_name(&m.name);
                    match_hvs.push(hv.clone());
                    results.push(hv);
                }

                // Analogy-based retrieval: if we have at least 2 matches,
                // use "match[0]:match[1] :: intent:?" to find analogous patterns
                if match_hvs.len() >= 2 {
                    let analogy_matches = self.algebra.analogy(
                        &match_hvs[0],
                        &match_hvs[1],
                        intent_hv,
                        &match_hvs[2..]
                            .iter()
                            .collect::<Vec<_>>()
                            .iter()
                            .map(|h| (*h).clone())
                            .collect::<Vec<_>>(),
                    );
                    for am in analogy_matches.iter().take(2) {
                        if am.similarity > 0.3 {
                            if let Some(hv) = match_hvs.get(am.index + 2) {
                                results.push(hv.clone());
                            }
                        }
                    }
                }
            }
        }

        results
    }

    /// Emit source code from a plan using the appropriate language emitter
    fn emit_from_plan(&self, plan: &[CodePlanStep], spec: &CodeSpec, language: &str) -> String {
        let emitter: &dyn CodeEmitter = match language {
            "rust" => &self.rust_emitter,
            "python" => &self.python_emitter,
            "nix" => &self.nix_emitter,
            _ => &self.rust_emitter, // default to Rust
        };

        emitter.emit_from_spec(spec, plan)
    }

    /// Attempt automatic repair using CodingExperienceStore hints first,
    /// then falling back to pattern-based diagnosis.
    ///
    /// `error_hints` are (error_pattern, fix_hint) pairs from the experience store.
    /// If a matching hint is found, applies the hint-suggested fix before trying
    /// the general-purpose `diagnose_compile_error` path.
    pub fn try_auto_fix_with_hints(
        &self,
        source: &str,
        stderr: &str,
        error_hints: &[(String, String)],
    ) -> Option<String> {
        // Try experience-store hints first — these are learned from prior sessions
        for (pattern, hint) in error_hints {
            if stderr.contains(pattern.as_str())
                || pattern
                    .split_whitespace()
                    .filter(|w| w.len() > 3)
                    .any(|w| stderr.contains(w))
            {
                // Apply hint-based fix
                if let Some(fixed) = Self::apply_hint_fix(source, hint) {
                    return Some(fixed);
                }
            }
        }
        // Fall back to pattern-based auto-fix
        self.try_auto_fix(source, stderr)
    }

    /// Apply a fix hint from the CodingExperienceStore.
    ///
    /// Hints are strings like "add `as u32`", "use `&str` instead of `String`",
    /// "add `#[derive(Clone)]`". Returns `Some(fixed)` if the hint was actionable.
    fn apply_hint_fix(source: &str, hint: &str) -> Option<String> {
        let hint_lower = hint.to_lowercase();
        let mut fixed = source.to_string();

        if hint_lower.contains("derive") {
            // Extract derive name and add it
            if let Some(start) = hint.find("Derive(").or_else(|| hint.find("derive(")) {
                let derive_part = &hint[start..];
                if let Some(end) = derive_part.find(')') {
                    let derive = &derive_part[..=end];
                    if !fixed.contains(derive) {
                        // Add derive before the first struct/enum
                        if let Some(pos) = fixed
                            .find("pub struct")
                            .or_else(|| fixed.find("struct"))
                            .or_else(|| fixed.find("pub enum"))
                            .or_else(|| fixed.find("enum"))
                        {
                            fixed.insert_str(pos, &format!("#[{}]\n", derive));
                            return Some(fixed);
                        }
                    }
                }
            }
        }

        if hint_lower.contains("as u32")
            || hint_lower.contains("as i32")
            || hint_lower.contains("as usize")
        {
            // Already handled by diagnose_compile_error, skip to avoid double-fixing
            return None;
        }

        if hint_lower.contains(".clone()") && !fixed.contains(".clone()") {
            // Generic: can't auto-apply without knowing WHERE to add .clone()
            return None;
        }

        None // Hint wasn't actionable — fall through to pattern-based
    }

    /// Attempt automatic repair of generated Rust code based on compiler error output.
    ///
    /// Uses `diagnose_compile_error` to identify the error category, then applies
    /// targeted source-level fixes. Returns `Some(fixed_source)` if a fix was applied.
    pub fn try_auto_fix(&self, source: &str, stderr: &str) -> Option<String> {
        let fixes = diagnose_compile_error(stderr);
        if fixes.is_empty() {
            return None;
        }

        let mut fixed = source.to_string();
        let mut applied = false;

        for fix in &fixes {
            match fix.category {
                "type_inference" => {
                    // Add turbofish to bare .parse() calls
                    if fixed.contains(".parse()") && !fixed.contains(".parse::<") {
                        // Try to infer type from context: look for -> Type or : Type
                        if let Some(ret) = Self::extract_return_type_from_source(&fixed) {
                            fixed = fixed.replace(".parse()", &format!(".parse::<{}>()", ret));
                            applied = true;
                        }
                    }
                }
                "wrong_method" => {
                    // .dedup() on iterator → restructure to Vec method
                    if stderr.contains("dedup") && fixed.contains(".dedup()") {
                        // Check if dedup is chained on an iterator (not on a Vec binding)
                        if fixed.contains(".iter()") && fixed.contains(".dedup()") {
                            // Cannot auto-fix complex chains — flag for regeneration
                        }
                    }
                }
                "type_mismatch" => {
                    // &T vs T: add .copied() or .cloned() before .collect()
                    if stderr.contains("expected") && fixed.contains(".iter()") {
                        if !fixed.contains(".cloned()") && !fixed.contains(".copied()") {
                            fixed =
                                fixed.replace(".iter().collect()", ".iter().cloned().collect()");
                            fixed = fixed.replace(
                                ".iter().enumerate().collect()",
                                ".into_iter().enumerate().collect()",
                            );
                            fixed = fixed.replace(".iter().zip(", ".into_iter().zip(");
                            applied = true;
                        }
                    }
                }
                "empty_closure" => {
                    // Replace empty closures |x| ) with a default
                    if fixed.contains("|x| )") || fixed.contains("|x| /* ") {
                        fixed = fixed.replace("|x| )", "|x| true)");
                        applied = true;
                    }
                }
                _ => {}
            }
        }

        if applied {
            Some(fixed)
        } else {
            None
        }
    }

    /// Extract return type from generated Rust source by scanning for `-> Type {`.
    fn extract_return_type_from_source(source: &str) -> Option<String> {
        for line in source.lines() {
            let trimmed = line.trim();
            if (trimmed.starts_with("pub fn ") || trimmed.starts_with("fn "))
                && trimmed.contains("->")
            {
                if let Some(arrow) = trimmed.find("->") {
                    let after = trimmed[arrow + 2..].trim();
                    let ret = after.trim_end_matches('{').trim();
                    if !ret.is_empty() {
                        return Some(ret.to_string());
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::code_intent::{CodeIntentCategory, CodeSpec, CodeTarget};

    #[test]
    fn test_create_generator() {
        let gen = CodeGenerator::with_default_dim();
        assert_eq!(gen.encoder().dim(), 512);
    }

    #[test]
    fn test_generate_create_rust() {
        let gen = CodeGenerator::with_default_dim();
        let intent = CodeIntent::Create {
            target: CodeTarget::new("sort_numbers", EntityKind::Function),
            spec: CodeSpec::new("rust", "sort_numbers", "Sort a vector of numbers")
                .with_signature("fn sort_numbers(data: &mut Vec<i32>)")
                .with_epistemic(EpistemicStatus::Certain),
        };

        let result = gen.generate(&intent, &CodeContext::default());
        assert_eq!(result.language, "rust");
        assert!(!result.source.is_empty());
    }

    #[test]
    fn test_generate_create_attaches_repo_snippets() {
        let gen = CodeGenerator::with_default_dim();
        let intent = CodeIntent::Create {
            target: CodeTarget::new("normalize_user", EntityKind::Function),
            spec: CodeSpec::new("rust", "normalize_user", "Normalize a user name")
                .with_signature("fn normalize_user(name: &str) -> String"),
        };
        let context = CodeContext {
            source_files: vec![(
                "src/users.rs:4:normalize_name".to_string(),
                "pub fn normalize_name(name: &str) -> String {\n    name.trim().to_lowercase()\n}"
                    .to_string(),
            )],
            ..CodeContext::default()
        };

        let result = gen.generate(&intent, &context);

        assert!(result.notes.iter().any(|note| {
            note.starts_with("REPO_SNIPPET(src/users.rs:4:normalize_name)")
                && note.contains("trim().to_lowercase()")
        }));
    }

    #[test]
    fn test_llm_completion_prompt_includes_repo_snippets() {
        let spec = CodeSpec::new("rust", "normalize_user", "Normalize a user name")
            .with_signature("fn normalize_user(name: &str) -> String");
        let result = GeneratedCode {
            source: "pub fn normalize_user(name: &str) -> String {\n    todo!()\n}".to_string(),
            language: "rust".to_string(),
            plan_steps: Vec::new(),
            epistemic_status: EpistemicStatus::Uncertain,
            intent_similarity: 0.1,
            notes: vec![
                "REPO_SNIPPET(src/users.rs:4:normalize_name):\npub fn normalize_name(name: &str) -> String {\n    name.trim().to_lowercase()\n}".to_string(),
            ],
            phi_score: 0.0,
            primitives_used: Vec::new(),
            dataflow: None,
            plan_coverage: 0.0,
        };

        let prompt = CodeGenerator::build_llm_completion_prompt(&spec, &result);

        assert!(prompt.contains("## Repository Context"));
        assert!(prompt.contains("normalize_name"));
        assert!(prompt.contains("trim().to_lowercase()"));
    }

    #[test]
    fn test_generate_create_python() {
        let gen = CodeGenerator::with_default_dim();
        let intent = CodeIntent::Create {
            target: CodeTarget::new("hello", EntityKind::Function),
            spec: CodeSpec::new("python", "hello", "Print hello world"),
        };

        let result = gen.generate(&intent, &CodeContext::default());
        assert_eq!(result.language, "python");
        assert!(!result.source.is_empty());
    }

    #[test]
    fn test_generate_explanation() {
        let gen = CodeGenerator::with_default_dim();
        let intent = CodeIntent::Explain {
            target: CodeTarget::new("parse", EntityKind::Function)
                .with_language("rust")
                .with_path("src/parser.rs"),
            depth: crate::language::code_intent::ExplanationDepth::Standard,
        };

        let result = gen.generate(&intent, &CodeContext::default());
        assert!(result.source.contains("parse"));
    }

    #[test]
    fn test_generate_modify() {
        let gen = CodeGenerator::with_default_dim();
        let intent = CodeIntent::Modify {
            target: CodeTarget::new("process", EntityKind::Function).with_language("rust"),
            changes: vec![
                CodeChange::AddParameter {
                    name: "timeout".to_string(),
                    param_type: "Duration".to_string(),
                },
                CodeChange::ChangeReturnType {
                    new_type: "Result<()>".to_string(),
                },
            ],
        };

        let result = gen.generate(&intent, &CodeContext::default());
        assert!(result.source.contains("timeout"));
        assert!(result.source.contains("Result<()>"));
    }

    #[test]
    fn test_control_flow_unreachable() {
        let source = "return 42;\nlet x = 10;";
        let cf = ControlFlowInfo::analyze_rust(source, Some("i32"));
        assert!(!cf.is_clean(), "Should detect unreachable code");
        assert!(cf.warnings.iter().any(|w| w.contains("UNREACHABLE")));
    }

    #[test]
    fn test_control_flow_missing_else() {
        let source = "if x > 0 {\n    x\n}";
        let cf = ControlFlowInfo::analyze_rust(source, Some("i32"));
        assert!(cf.warnings.iter().any(|w| w.contains("MISSING_ELSE")));
    }

    #[test]
    fn test_control_flow_clean() {
        let source = "x + y";
        let cf = ControlFlowInfo::analyze_rust(source, Some("i32"));
        assert!(cf.is_clean());
    }

    #[test]
    fn test_parsed_type_simple() {
        assert_eq!(
            ParsedType::parse("i32"),
            ParsedType::Simple("i32".to_string())
        );
    }

    #[test]
    fn test_parsed_type_generic() {
        let t = ParsedType::parse("Vec<i32>");
        match t {
            ParsedType::Generic {
                ref base,
                ref params,
            } => {
                assert_eq!(base, "Vec");
                assert_eq!(params.len(), 1);
                assert_eq!(params[0], ParsedType::Simple("i32".to_string()));
            }
            _ => panic!("Expected Generic, got {:?}", t),
        }
    }

    #[test]
    fn test_parsed_type_nested() {
        let t = ParsedType::parse("HashMap<String, Vec<f64>>");
        match t {
            ParsedType::Generic {
                ref base,
                ref params,
            } => {
                assert_eq!(base, "HashMap");
                assert_eq!(params.len(), 2);
                assert!(matches!(&params[1], ParsedType::Generic { base, .. } if base == "Vec"));
            }
            _ => panic!("Expected Generic, got {:?}", t),
        }
    }

    #[test]
    fn test_parsed_type_tuple() {
        let t = ParsedType::parse("(i32, String)");
        match t {
            ParsedType::Tuple(ref parts) => {
                assert_eq!(parts.len(), 2);
            }
            _ => panic!("Expected Tuple, got {:?}", t),
        }
    }

    #[test]
    fn test_diagnose_compile_error() {
        let fixes = diagnose_compile_error("error: expected expression, found `)`");
        assert!(!fixes.is_empty());
        assert_eq!(fixes[0].category, "empty_closure");

        let fixes2 = diagnose_compile_error("error[E0308]: mismatched types");
        assert!(!fixes2.is_empty());
        assert_eq!(fixes2[0].category, "type_mismatch");

        let fixes3 = diagnose_compile_error("all good, no errors");
        assert!(fixes3.is_empty());
    }

    #[test]
    fn test_auto_fix_type_inference() {
        let gen = CodeGenerator::with_default_dim();
        let source = "pub fn parse_int(s: &str) -> i32 {\n    s.parse().unwrap_or_default()\n}";
        let stderr = "error[E0282]: type annotations needed";
        let fixed = gen.try_auto_fix(source, stderr);
        assert!(fixed.is_some(), "Should auto-fix type inference");
        let fixed = fixed.unwrap();
        assert!(
            fixed.contains(".parse::<i32>()"),
            "Should add turbofish: {}",
            fixed
        );
    }

    #[test]
    fn test_auto_fix_type_mismatch_iter() {
        let gen = CodeGenerator::with_default_dim();
        let source = "a.iter().zip(b.iter()).collect()";
        let stderr = "error[E0308]: mismatched types\nexpected (i32, i32), found (&i32, &i32)";
        let fixed = gen.try_auto_fix(source, stderr);
        assert!(fixed.is_some(), "Should auto-fix iter to into_iter");
        let fixed = fixed.unwrap();
        assert!(
            fixed.contains("into_iter()"),
            "Should switch to into_iter: {}",
            fixed
        );
    }

    #[test]
    fn test_auto_fix_no_fix_needed() {
        let gen = CodeGenerator::with_default_dim();
        let result = gen.try_auto_fix("fn good() { }", "all good");
        assert!(result.is_none(), "Should return None when no fix needed");
    }

    #[test]
    fn test_e2e_generate_and_validate() {
        // End-to-end: text intent → generation → validation → distillation
        let gen = CodeGenerator::with_default_dim();
        let spec = CodeSpec::new("rust", "add", "Add two numbers")
            .with_signature("fn add(a: i32, b: i32) -> i32");
        let intent = CodeIntent::Create {
            target: CodeTarget::new("add", EntityKind::Function).with_language("rust"),
            spec: spec.clone(),
        };

        let result = gen.generate(&intent, &CodeContext::default());

        // Generation should produce valid code
        assert!(!result.source.is_empty(), "Should generate source");
        assert!(
            !result.source.contains("todo!"),
            "Should not have todo: {}",
            result.source
        );
        assert!(
            result.source.contains("a + b"),
            "Should have add body: {}",
            result.source
        );

        // Type validation should be clean
        let type_warnings = CodeGenerator::validate_types(&spec, &result.source);
        assert!(
            type_warnings.is_empty(),
            "No type warnings expected: {:?}",
            type_warnings
        );

        // Dataflow should be clean
        if let Some(ref df) = result.dataflow {
            assert!(
                df.is_clean(),
                "Dataflow should be clean: {:?}",
                df.use_before_def
            );
        }

        // Distillation target should be available
        let target = gen.distillation_target(&spec, &result);
        assert!(
            target.is_some(),
            "Should produce distillation target for clean native code"
        );
        let (_, src, quality) = target.unwrap();
        assert!(!src.is_empty());
        assert!(quality > 0.0, "Quality should be positive: {}", quality);
    }
}
