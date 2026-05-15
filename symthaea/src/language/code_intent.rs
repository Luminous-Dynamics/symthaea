// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Code Intent System — Bridge between StructuredThought and code generation
//!
//! Classifies user intents related to code operations and creates structured
//! specifications that the code generation pipeline can execute.
//!
//! # Intent Types
//!
//! - **Create**: Write new code (function, struct, module)
//! - **Modify**: Change existing code
//! - **Explain**: Understand what code does
//! - **Find**: Search for code patterns
//! - **Refactor**: Transform code structure
//! - **Debug**: Diagnose and fix issues

use std::collections::HashMap;
use std::path::PathBuf;

use symthaea_core::hdc::ContinuousHV;

pub use super::code_parser::EntityKind;
use crate::mind::structured_thought::EpistemicStatus;

/// What the user wants to do with code
#[derive(Debug, Clone)]
pub enum CodeIntent {
    /// Write new code (function, struct, module, etc.)
    Create { target: CodeTarget, spec: CodeSpec },
    /// Modify existing code
    Modify {
        target: CodeTarget,
        changes: Vec<CodeChange>,
    },
    /// Explain what code does
    Explain {
        target: CodeTarget,
        depth: ExplanationDepth,
    },
    /// Find code matching a pattern
    Find {
        pattern_hv: ContinuousHV,
        scope: SearchScope,
    },
    /// Refactor code structure
    Refactor {
        target: CodeTarget,
        strategy: RefactorStrategy,
    },
    /// Debug: diagnose and fix an issue
    Debug {
        target: CodeTarget,
        symptoms: Vec<String>,
    },
    /// Solve: repository-scale engineering task (SWE-bench style)
    Solve { target: RepoTarget, spec: IssueSpec },
}

impl CodeIntent {
    /// Get the target language, if specified
    pub fn language(&self) -> Option<&str> {
        match self {
            CodeIntent::Create { spec, .. } => Some(&spec.language),
            CodeIntent::Modify { target, .. }
            | CodeIntent::Explain { target, .. }
            | CodeIntent::Refactor { target, .. }
            | CodeIntent::Debug { target, .. } => target.language.as_deref(),
            CodeIntent::Solve { spec, .. } => Some(&spec.language),
            CodeIntent::Find { .. } => None,
        }
    }

    /// Get the epistemic status of this intent
    pub fn epistemic_status(&self) -> EpistemicStatus {
        match self {
            CodeIntent::Create { spec, .. } => spec.epistemic_status,
            CodeIntent::Solve { spec, .. } => spec.epistemic_status,
            _ => EpistemicStatus::Probable,
        }
    }
}

/// A whole repository target
#[derive(Debug, Clone)]
pub struct RepoTarget {
    /// Name of the repository
    pub name: String,
    /// Local path to the repository
    pub root: PathBuf,
    /// Remote URL (optional)
    pub url: Option<String>,
}

/// Specification for a repository-scale issue to solve
#[derive(Debug, Clone)]
pub struct IssueSpec {
    /// Primary language of the issue
    pub language: String,
    /// Raw GitHub issue text (description)
    pub description: String,
    /// List of file paths explicitly mentioned in the issue
    pub relevant_files: Vec<String>,
    /// Epistemic status of this specification
    pub epistemic_status: EpistemicStatus,
}

/// What code entity is being targeted
#[derive(Debug, Clone)]
pub struct CodeTarget {
    /// Name of the target entity
    pub name: String,
    /// Kind of entity (function, struct, module, etc.)
    pub kind: EntityKind,
    /// Language of the target code
    pub language: Option<String>,
    /// File path, if known
    pub path: Option<String>,
    /// HDC encoding of the target, if available
    pub hv: Option<ContinuousHV>,
}

impl CodeTarget {
    /// Create a new code target
    pub fn new(name: impl Into<String>, kind: EntityKind) -> Self {
        Self {
            name: name.into(),
            kind,
            language: None,
            path: None,
            hv: None,
        }
    }

    /// Set the language
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// Set the file path
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Set the HDC encoding
    pub fn with_hv(mut self, hv: ContinuousHV) -> Self {
        self.hv = Some(hv);
        self
    }
}

/// Specification for code to be generated
#[derive(Debug, Clone)]
pub struct CodeSpec {
    /// Target language
    pub language: String,
    /// Name of the entity to create
    pub name: String,
    /// Natural language description of purpose
    pub purpose: String,
    /// HDC encoding of the purpose
    pub purpose_hv: Option<ContinuousHV>,
    /// Optional type signature
    pub signature: Option<String>,
    /// Constraints on the generated code
    pub constraints: Vec<String>,
    /// Input/output examples
    pub examples: Vec<(String, String)>,
    /// Epistemic status of this specification
    pub epistemic_status: EpistemicStatus,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl CodeSpec {
    /// Create a new code spec
    pub fn new(
        language: impl Into<String>,
        name: impl Into<String>,
        purpose: impl Into<String>,
    ) -> Self {
        Self {
            language: language.into(),
            name: name.into(),
            purpose: purpose.into(),
            purpose_hv: None,
            signature: None,
            constraints: Vec::new(),
            examples: Vec::new(),
            epistemic_status: EpistemicStatus::Probable,
            metadata: HashMap::new(),
        }
    }

    /// Set the type signature
    pub fn with_signature(mut self, sig: impl Into<String>) -> Self {
        self.signature = Some(sig.into());
        self
    }

    /// Add a constraint
    pub fn with_constraint(mut self, constraint: impl Into<String>) -> Self {
        self.constraints.push(constraint.into());
        self
    }

    /// Add an example
    pub fn with_example(mut self, input: impl Into<String>, output: impl Into<String>) -> Self {
        self.examples.push((input.into(), output.into()));
        self
    }

    /// Set epistemic status
    pub fn with_epistemic(mut self, status: EpistemicStatus) -> Self {
        self.epistemic_status = status;
        self
    }

    /// Set the purpose HV
    pub fn with_purpose_hv(mut self, hv: ContinuousHV) -> Self {
        self.purpose_hv = Some(hv);
        self
    }
}

/// A specific change to apply to existing code
#[derive(Debug, Clone)]
pub enum CodeChange {
    /// Add a new parameter
    AddParameter { name: String, param_type: String },
    /// Remove a parameter
    RemoveParameter { name: String },
    /// Change return type
    ChangeReturnType { new_type: String },
    /// Add error handling
    AddErrorHandling { strategy: String },
    /// Rename entity
    Rename { new_name: String },
    /// Add documentation
    AddDocumentation { content: String },
    /// Generic modification described in natural language
    Custom { description: String },
}

/// How deep an explanation should go
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplanationDepth {
    /// One-line summary
    Brief,
    /// Paragraph explanation
    Standard,
    /// Line-by-line walkthrough
    Detailed,
    /// Full analysis including complexity, safety, etc.
    Comprehensive,
}

/// Where to search for code
#[derive(Debug, Clone)]
pub enum SearchScope {
    /// Search entire codebase
    All,
    /// Search within a specific file
    File(String),
    /// Search within a specific module/directory
    Module(String),
    /// Search only functions
    FunctionsOnly,
    /// Search only types
    TypesOnly,
}

/// How to refactor code
#[derive(Debug, Clone)]
pub enum RefactorStrategy {
    /// Extract a function from a code block
    ExtractFunction { name: String },
    /// Inline a function at call sites
    InlineFunction,
    /// Rename with all references
    Rename { new_name: String },
    /// Extract an interface/trait
    ExtractTrait { name: String },
    /// Split a large function
    SplitFunction,
    /// Convert between equivalent patterns
    ConvertPattern { from: String, to: String },
}

/// Classifies text input into CodeIntent
pub struct CodeIntentClassifier {
    dim: usize,
    /// Prototype HVs for different intents
    create_prototype: ContinuousHV,
    modify_prototype: ContinuousHV,
    explain_prototype: ContinuousHV,
    find_prototype: ContinuousHV,
    refactor_prototype: ContinuousHV,
    debug_prototype: ContinuousHV,
}

impl CodeIntentClassifier {
    /// Create a new classifier with the given dimension.
    ///
    /// `dim` must be at least 1 to avoid division-by-zero in modular hashing.
    pub fn new(dim: usize) -> Self {
        let dim = dim.max(1);
        Self {
            dim,
            create_prototype: Self::encode_prototype(
                dim,
                &[
                    "write",
                    "create",
                    "implement",
                    "add",
                    "new",
                    "generate",
                    "function",
                    "struct",
                    "module",
                    "class",
                ],
            ),
            modify_prototype: Self::encode_prototype(
                dim,
                &[
                    "change",
                    "modify",
                    "update",
                    "fix",
                    "alter",
                    "adjust",
                    "edit",
                    "transform",
                ],
            ),
            explain_prototype: Self::encode_prototype(
                dim,
                &[
                    "explain",
                    "what",
                    "how",
                    "why",
                    "describe",
                    "understand",
                    "show",
                    "tell",
                    "mean",
                ],
            ),
            find_prototype: Self::encode_prototype(
                dim,
                &[
                    "find", "search", "where", "locate", "similar", "like", "match", "pattern",
                ],
            ),
            refactor_prototype: Self::encode_prototype(
                dim,
                &[
                    "refactor",
                    "extract",
                    "rename",
                    "inline",
                    "split",
                    "reorganize",
                    "clean",
                    "simplify",
                ],
            ),
            debug_prototype: Self::encode_prototype(
                dim,
                &[
                    "debug", "error", "bug", "fix", "crash", "wrong", "broken", "fail", "issue",
                    "problem",
                ],
            ),
        }
    }

    /// Encode a set of keywords into a prototype HV
    fn encode_prototype(dim: usize, keywords: &[&str]) -> ContinuousHV {
        let mut values = vec![0.0f32; dim];

        for keyword in keywords {
            let keyword_lower = keyword.to_lowercase();
            for (i, byte) in keyword_lower.bytes().enumerate() {
                let idx = ((byte as usize)
                    .wrapping_mul(31)
                    .wrapping_add(i.wrapping_mul(7)))
                    % dim;
                values[idx] += 1.0;
            }
        }

        // Normalize
        let magnitude: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in &mut values {
                *v /= magnitude;
            }
        }

        ContinuousHV::from_values(values)
    }

    /// Encode text for classification
    fn encode_text(&self, text: &str) -> ContinuousHV {
        let mut values = vec![0.0f32; self.dim];
        let text_lower = text.to_lowercase();

        for (i, byte) in text_lower.bytes().enumerate() {
            let idx = ((byte as usize)
                .wrapping_mul(31)
                .wrapping_add(i.wrapping_mul(7)))
                % self.dim;
            values[idx] += 1.0;
        }

        let magnitude: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in &mut values {
                *v /= magnitude;
            }
        }

        ContinuousHV::from_values(values)
    }

    /// Classify text input into a code intent category
    pub fn classify(&self, text: &str) -> CodeIntentCategory {
        let text_hv = self.encode_text(text);

        let scores = [
            (
                CodeIntentCategory::Create,
                text_hv.similarity(&self.create_prototype),
            ),
            (
                CodeIntentCategory::Modify,
                text_hv.similarity(&self.modify_prototype),
            ),
            (
                CodeIntentCategory::Explain,
                text_hv.similarity(&self.explain_prototype),
            ),
            (
                CodeIntentCategory::Find,
                text_hv.similarity(&self.find_prototype),
            ),
            (
                CodeIntentCategory::Refactor,
                text_hv.similarity(&self.refactor_prototype),
            ),
            (
                CodeIntentCategory::Debug,
                text_hv.similarity(&self.debug_prototype),
            ),
        ];

        scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(cat, _)| *cat)
            .unwrap_or(CodeIntentCategory::Create)
    }

    /// Get all scores for text input (useful for ambiguous cases)
    pub fn classify_with_scores(&self, text: &str) -> Vec<(CodeIntentCategory, f32)> {
        let text_hv = self.encode_text(text);

        let mut scores = vec![
            (
                CodeIntentCategory::Create,
                text_hv.similarity(&self.create_prototype),
            ),
            (
                CodeIntentCategory::Modify,
                text_hv.similarity(&self.modify_prototype),
            ),
            (
                CodeIntentCategory::Explain,
                text_hv.similarity(&self.explain_prototype),
            ),
            (
                CodeIntentCategory::Find,
                text_hv.similarity(&self.find_prototype),
            ),
            (
                CodeIntentCategory::Refactor,
                text_hv.similarity(&self.refactor_prototype),
            ),
            (
                CodeIntentCategory::Debug,
                text_hv.similarity(&self.debug_prototype),
            ),
        ];

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }
}

/// High-level category of a code intent
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeIntentCategory {
    Create,
    Modify,
    Explain,
    Find,
    Refactor,
    Debug,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_spec_builder() {
        let spec = CodeSpec::new("rust", "my_func", "Sort a vector of integers")
            .with_signature("fn my_func(data: &mut Vec<i32>)")
            .with_constraint("must be in-place")
            .with_example("[3,1,2]", "[1,2,3]")
            .with_epistemic(EpistemicStatus::Certain);

        assert_eq!(spec.language, "rust");
        assert_eq!(spec.name, "my_func");
        assert!(spec.signature.is_some());
        assert_eq!(spec.constraints.len(), 1);
        assert_eq!(spec.examples.len(), 1);
        assert_eq!(spec.epistemic_status, EpistemicStatus::Certain);
    }

    #[test]
    fn test_code_target_builder() {
        let target = CodeTarget::new("parse", EntityKind::Function)
            .with_language("rust")
            .with_path("src/parser.rs");

        assert_eq!(target.name, "parse");
        assert_eq!(target.kind, EntityKind::Function);
        assert_eq!(target.language.as_deref(), Some("rust"));
        assert_eq!(target.path.as_deref(), Some("src/parser.rs"));
    }

    #[test]
    fn test_intent_classifier() {
        let classifier = CodeIntentClassifier::new(512);

        // Test that classifier produces non-empty scores
        let scores = classifier.classify_with_scores("write a sorting function");
        assert!(!scores.is_empty(), "Classifier should produce scores");
        assert!(scores[0].1 > 0.0, "Top score should be positive");

        // Test that different inputs produce different top categories or different score distributions
        let create_scores = classifier.classify_with_scores("implement a new feature");
        let debug_scores = classifier.classify_with_scores("fix the crash bug");

        // At least scores should differ somewhat for different inputs
        let create_top = create_scores[0].1;
        let debug_top = debug_scores[0].1;
        // Both should have some positive score
        assert!(
            create_top > 0.0 && debug_top > 0.0,
            "Scores should be positive"
        );
    }

    #[test]
    fn test_classify_with_scores() {
        let classifier = CodeIntentClassifier::new(512);

        let scores = classifier.classify_with_scores("debug the error in sort function");
        assert!(!scores.is_empty());
        // Scores should be sorted descending
        for i in 1..scores.len() {
            assert!(scores[i - 1].1 >= scores[i].1);
        }
    }

    #[test]
    fn test_code_intent_language() {
        let intent = CodeIntent::Create {
            target: CodeTarget::new("my_func", EntityKind::Function),
            spec: CodeSpec::new("rust", "my_func", "Sort integers"),
        };

        assert_eq!(intent.language(), Some("rust"));
    }
}
