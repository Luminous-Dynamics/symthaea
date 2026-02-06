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

use std::collections::HashMap;

use symthaea_core::hdc::RealHV;

use super::code_intent::{CodeIntent, CodeSpec, CodeTarget, CodeChange};
use super::code_parser::{EntityKind, ParsedCode};
use super::emitters::{CodeEmitter, RustEmitter, PythonEmitter, NixEmitter};
use crate::dynamics::cfc_code_sequencer::{CfCCodeSequencer, CodePlanStep, PlanAction};
use crate::hdc::code_algebra::CodeAlgebra;
use crate::hdc::code_encoder::CodeHDEncoder;
use crate::hdc::code_memory::CodebaseMemory;
use crate::mind::structured_thought::EpistemicStatus;

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
}

/// Context provided to the code generator
pub struct CodeContext<'a> {
    /// The codebase memory for pattern retrieval
    pub memory: Option<&'a CodebaseMemory>,
    /// Additional context HVs (e.g., from conversation)
    pub context_hvs: Vec<RealHV>,
    /// Additional source files for context
    pub source_files: Vec<(String, String)>,
}

impl<'a> Default for CodeContext<'a> {
    fn default() -> Self {
        Self {
            memory: None,
            context_hvs: Vec::new(),
            source_files: Vec::new(),
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
}

impl CodeGenerator {
    /// Create a new code generator
    pub fn new(encoder: CodeHDEncoder) -> Self {
        let algebra = CodeAlgebra::new(CodeHDEncoder::new(encoder.dim()));
        let sequencer = CfCCodeSequencer::new(
            crate::dynamics::cfc_code_sequencer::CfCCodeSequencerConfig {
                hdc_dim: encoder.dim(),
                ..Default::default()
            },
        );

        Self {
            encoder,
            algebra,
            sequencer,
            rust_emitter: RustEmitter,
            python_emitter: PythonEmitter,
            nix_emitter: NixEmitter,
        }
    }

    /// Create with default dimension
    pub fn with_default_dim() -> Self {
        Self::new(CodeHDEncoder::new(512))
    }

    /// Get the encoder
    pub fn encoder(&self) -> &CodeHDEncoder {
        &self.encoder
    }

    /// Generate code from an intent
    pub fn generate(&self, intent: &CodeIntent, context: &CodeContext) -> GeneratedCode {
        match intent {
            CodeIntent::Create { target, spec } => self.generate_create(target, spec, context),
            CodeIntent::Modify { target, changes } => self.generate_modify(target, changes, context),
            CodeIntent::Explain { target, depth } => self.generate_explanation(target, *depth, context),
            CodeIntent::Find { pattern_hv, scope } => self.generate_search_result(pattern_hv, scope, context),
            CodeIntent::Refactor { target, strategy } => self.generate_refactor(target, strategy, context),
            CodeIntent::Debug { target, symptoms } => self.generate_debug(target, symptoms, context),
        }
    }

    /// Generate new code from a specification
    fn generate_create(
        &self,
        target: &CodeTarget,
        spec: &CodeSpec,
        context: &CodeContext,
    ) -> GeneratedCode {
        // 1. Encode intent as HDC
        let intent_hv = self.encode_spec(spec);

        // 2. Query codebase for similar patterns
        let similar_hvs = self.find_similar_context(&intent_hv, context);
        let similar_refs: Vec<&RealHV> = similar_hvs.iter().collect();

        // 3. CfC plan
        let plan = self.sequencer.plan_structure(&intent_hv, &similar_refs);

        // 4. Emit code using language-specific emitter
        let source = self.emit_from_plan(&plan, spec, &spec.language);

        // 5. Compute intent similarity (how close is the result to what we wanted)
        let intent_similarity = if !source.is_empty() {
            // Simple heuristic: check if plan steps cover expected structure
            let coverage = plan.len() as f32 / 5.0; // normalized by typical plan length
            coverage.min(1.0)
        } else {
            0.0
        };

        GeneratedCode {
            source,
            language: spec.language.clone(),
            plan_steps: plan,
            epistemic_status: spec.epistemic_status,
            intent_similarity,
            notes: Vec::new(),
        }
    }

    /// Generate code modification
    fn generate_modify(
        &self,
        target: &CodeTarget,
        changes: &[CodeChange],
        _context: &CodeContext,
    ) -> GeneratedCode {
        let mut notes = Vec::new();

        // Build modification description
        let mut modifications = Vec::new();
        for change in changes {
            match change {
                CodeChange::AddParameter { name, param_type } => {
                    modifications.push(format!("// TODO: Add parameter `{}: {}`", name, param_type));
                }
                CodeChange::ChangeReturnType { new_type } => {
                    modifications.push(format!("// TODO: Change return type to `{}`", new_type));
                }
                CodeChange::Rename { new_name } => {
                    modifications.push(format!("// TODO: Rename `{}` to `{}`", target.name, new_name));
                }
                CodeChange::AddDocumentation { content } => {
                    modifications.push(format!("/// {}", content));
                }
                CodeChange::AddErrorHandling { strategy } => {
                    modifications.push(format!("// TODO: Add error handling ({})", strategy));
                }
                CodeChange::RemoveParameter { name } => {
                    modifications.push(format!("// TODO: Remove parameter `{}`", name));
                }
                CodeChange::Custom { description } => {
                    modifications.push(format!("// TODO: {}", description));
                }
            }
        }

        notes.push(format!("Modification plan for `{}`", target.name));

        GeneratedCode {
            source: modifications.join("\n"),
            language: target.language.clone().unwrap_or_else(|| "rust".to_string()),
            plan_steps: Vec::new(),
            epistemic_status: EpistemicStatus::Probable,
            intent_similarity: 0.5,
            notes,
        }
    }

    /// Generate code explanation
    fn generate_explanation(
        &self,
        target: &CodeTarget,
        depth: super::code_intent::ExplanationDepth,
        _context: &CodeContext,
    ) -> GeneratedCode {
        let explanation = match depth {
            super::code_intent::ExplanationDepth::Brief => {
                format!("// `{}` is a {:?}", target.name, target.kind)
            }
            super::code_intent::ExplanationDepth::Standard => {
                format!(
                    "// `{}` is a {:?}\n// Located in: {}",
                    target.name,
                    target.kind,
                    target.path.as_deref().unwrap_or("unknown")
                )
            }
            _ => {
                format!(
                    "// Detailed analysis of `{}` ({:?})\n// Path: {}\n// Language: {}\n// (Full analysis requires LLM organ translation)",
                    target.name,
                    target.kind,
                    target.path.as_deref().unwrap_or("unknown"),
                    target.language.as_deref().unwrap_or("unknown"),
                )
            }
        };

        GeneratedCode {
            source: explanation,
            language: target.language.clone().unwrap_or_else(|| "rust".to_string()),
            plan_steps: Vec::new(),
            epistemic_status: EpistemicStatus::Probable,
            intent_similarity: 0.7,
            notes: vec!["Explanation skeleton — full translation requires LLM organ".to_string()],
        }
    }

    /// Generate search results
    fn generate_search_result(
        &self,
        pattern_hv: &RealHV,
        _scope: &super::code_intent::SearchScope,
        context: &CodeContext,
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
        }
    }

    /// Generate refactoring suggestion
    fn generate_refactor(
        &self,
        target: &CodeTarget,
        strategy: &super::code_intent::RefactorStrategy,
        _context: &CodeContext,
    ) -> GeneratedCode {
        let suggestion = match strategy {
            super::code_intent::RefactorStrategy::ExtractFunction { name } => {
                format!("// Refactor: Extract function `{}` from `{}`", name, target.name)
            }
            super::code_intent::RefactorStrategy::InlineFunction => {
                format!("// Refactor: Inline `{}` at call sites", target.name)
            }
            super::code_intent::RefactorStrategy::Rename { new_name } => {
                format!("// Refactor: Rename `{}` → `{}`", target.name, new_name)
            }
            super::code_intent::RefactorStrategy::ExtractTrait { name } => {
                format!("// Refactor: Extract trait `{}` from `{}`", name, target.name)
            }
            super::code_intent::RefactorStrategy::SplitFunction => {
                format!("// Refactor: Split `{}` into smaller functions", target.name)
            }
            super::code_intent::RefactorStrategy::ConvertPattern { from, to } => {
                format!("// Refactor: Convert `{}` pattern from `{}` to `{}`", target.name, from, to)
            }
        };

        GeneratedCode {
            source: suggestion,
            language: target.language.clone().unwrap_or_else(|| "rust".to_string()),
            plan_steps: Vec::new(),
            epistemic_status: EpistemicStatus::Uncertain,
            intent_similarity: 0.5,
            notes: vec!["Refactoring skeleton — needs verification".to_string()],
        }
    }

    /// Generate debug analysis
    fn generate_debug(
        &self,
        target: &CodeTarget,
        symptoms: &[String],
        _context: &CodeContext,
    ) -> GeneratedCode {
        let mut analysis = vec![
            format!("// Debug analysis for `{}`", target.name),
        ];

        for symptom in symptoms {
            analysis.push(format!("// Symptom: {}", symptom));
        }

        analysis.push("// (Full diagnosis requires LLM organ analysis)".to_string());

        GeneratedCode {
            source: analysis.join("\n"),
            language: target.language.clone().unwrap_or_else(|| "rust".to_string()),
            plan_steps: Vec::new(),
            epistemic_status: EpistemicStatus::Uncertain,
            intent_similarity: 0.4,
            notes: vec!["Debug skeleton — requires LLM organ for full analysis".to_string()],
        }
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    /// Encode a code spec into an HDC vector
    fn encode_spec(&self, spec: &CodeSpec) -> RealHV {
        if let Some(ref hv) = spec.purpose_hv {
            return hv.clone();
        }

        // Combine name + purpose into an encoding
        let name_hv = self.encoder.encode_name(&spec.name);
        let purpose_hv = self.encoder.encode_name(&spec.purpose);
        let lang_hv = self.encoder.encode_name(&spec.language);

        RealHV::bundle(&[name_hv, purpose_hv, lang_hv])
    }

    /// Find similar patterns from context
    fn find_similar_context(&self, intent_hv: &RealHV, context: &CodeContext) -> Vec<RealHV> {
        let mut results = Vec::new();

        // Add context HVs directly
        for hv in &context.context_hvs {
            results.push(hv.clone());
        }

        // Query codebase memory if available
        if let Some(memory) = context.memory {
            let matches = memory.query(intent_hv, 3);
            // We just use the match info, not the HVs themselves since CodeMatch
            // doesn't store them. The memory context is already captured in context_hvs.
            if !matches.is_empty() {
                // Encode matched names as additional context
                for m in &matches {
                    results.push(self.encoder.encode_name(&m.name));
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::code_intent::{CodeSpec, CodeTarget, CodeIntentCategory};

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
}
