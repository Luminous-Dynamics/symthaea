// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Epistemic-Aware Code Generation
//!
//! Generates code with explicit uncertainty tracking. Never generates code
//! with false confidence — if we don't know, we say so.
//!
//! # Epistemic Levels
//!
//! - **Certain**: High confidence — generate directly
//! - **Probable**: Moderate confidence — generate with TODO markers
//! - **Uncertain**: Low confidence — generate skeleton with questions
//! - **Unknown**: Cannot generate — explain what we'd need to know

use super::code_generator::{CodeContext, CodeGenerator, GeneratedCode};
use super::code_intent::{CodeIntent, CodeIntentClassifier};
use crate::mind::structured_thought::EpistemicStatus;

/// Code with epistemic status attached
#[derive(Debug)]
pub enum EpistemicCode {
    /// High confidence — code is ready to use
    Confident(GeneratedCode),
    /// Moderate confidence — code with uncertainty markers
    Probable(GeneratedCode, Vec<String>),
    /// Low confidence — skeleton that needs human input
    NeedsInput(GeneratedCode, Vec<String>),
    /// Cannot generate — explanation of what's missing
    CannotGenerate(String),
}

impl EpistemicCode {
    /// Get the source code, if any
    pub fn source(&self) -> Option<&str> {
        match self {
            EpistemicCode::Confident(g) => Some(&g.source),
            EpistemicCode::Probable(g, _) => Some(&g.source),
            EpistemicCode::NeedsInput(g, _) => Some(&g.source),
            EpistemicCode::CannotGenerate(_) => None,
        }
    }

    /// Get the epistemic status
    pub fn epistemic_status(&self) -> EpistemicStatus {
        match self {
            EpistemicCode::Confident(_) => EpistemicStatus::Certain,
            EpistemicCode::Probable(_, _) => EpistemicStatus::Probable,
            EpistemicCode::NeedsInput(_, _) => EpistemicStatus::Uncertain,
            EpistemicCode::CannotGenerate(_) => EpistemicStatus::Unknown,
        }
    }

    /// Get uncertainty notes/questions
    pub fn notes(&self) -> Vec<&str> {
        match self {
            EpistemicCode::Confident(_) => Vec::new(),
            EpistemicCode::Probable(_, notes) => notes.iter().map(|s| s.as_str()).collect(),
            EpistemicCode::NeedsInput(_, questions) => {
                questions.iter().map(|s| s.as_str()).collect()
            }
            EpistemicCode::CannotGenerate(reason) => vec![reason.as_str()],
        }
    }

    /// Is this code ready to use without human review?
    pub fn is_ready(&self) -> bool {
        matches!(self, EpistemicCode::Confident(_))
    }
}

/// Epistemic-aware code generator wrapper
pub struct EpistemicCodeGenerator {
    generator: CodeGenerator,
    _intent_classifier: CodeIntentClassifier,
}

impl EpistemicCodeGenerator {
    /// Create a new epistemic code generator
    pub fn new(generator: CodeGenerator, dim: usize) -> Self {
        Self {
            generator,
            _intent_classifier: CodeIntentClassifier::new(dim),
        }
    }

    /// Generate code with explicit confidence tracking
    pub fn generate_with_confidence(
        &self,
        intent: &CodeIntent,
        context: &CodeContext,
    ) -> EpistemicCode {
        let epistemic = intent.epistemic_status();

        match epistemic {
            EpistemicStatus::Certain => {
                let code = self.generator.generate(intent, context);
                EpistemicCode::Confident(code)
            }
            EpistemicStatus::Probable => {
                let mut code = self.generator.generate(intent, context);
                let uncertainties = self.identify_uncertainties(intent);

                // Add TODO markers to source
                if !uncertainties.is_empty() {
                    let markers: String = uncertainties
                        .iter()
                        .map(|u| format!("// TODO: Verify — {}", u))
                        .collect::<Vec<_>>()
                        .join("\n");
                    code.source = format!("{}\n\n{}", markers, code.source);
                }

                EpistemicCode::Probable(code, uncertainties)
            }
            EpistemicStatus::Uncertain => {
                let code = self.generator.generate(intent, context);
                let questions = self.generate_questions(intent);
                EpistemicCode::NeedsInput(code, questions)
            }
            EpistemicStatus::Unknown | EpistemicStatus::OutOfDomain => {
                let reasoning = self.explain_unknowns(intent);
                EpistemicCode::CannotGenerate(reasoning)
            }
        }
    }

    /// Identify what we're uncertain about in this intent
    fn identify_uncertainties(&self, intent: &CodeIntent) -> Vec<String> {
        let mut uncertainties = Vec::new();

        match intent {
            CodeIntent::Create { spec, .. } => {
                if spec.signature.is_none() {
                    uncertainties
                        .push("No type signature specified — using best guess".to_string());
                }
                if spec.examples.is_empty() {
                    uncertainties.push("No examples provided — verify correctness".to_string());
                }
                if spec.constraints.is_empty() {
                    uncertainties.push("No constraints specified — review edge cases".to_string());
                }
            }
            CodeIntent::Modify { changes, .. } if changes.len() > 3 => {
                uncertainties
                    .push("Multiple simultaneous changes — verify interactions".to_string());
            }
            CodeIntent::Refactor { .. } => {
                uncertainties.push("Refactoring may change behavior — run tests".to_string());
            }
            CodeIntent::Solve { spec, .. } => {
                if spec.relevant_files.is_empty() {
                    uncertainties.push(
                        "No relevant files specified — navigation will be purely discovery-based"
                            .to_string(),
                    );
                }
                uncertainties
                    .push("Repository-scale fix requires cross-file verification".to_string());
            }
            _ => {}
        }

        uncertainties
    }

    /// Generate questions for low-confidence intents
    fn generate_questions(&self, intent: &CodeIntent) -> Vec<String> {
        let mut questions = Vec::new();

        match intent {
            CodeIntent::Create { spec, .. } => {
                if spec.signature.is_none() {
                    questions.push("What should the function signature be?".to_string());
                }
                questions.push(format!("What edge cases should {} handle?", spec.name));
                if spec.examples.is_empty() {
                    questions.push("Can you provide input/output examples?".to_string());
                }
            }
            CodeIntent::Debug { symptoms, .. } => {
                if symptoms.is_empty() {
                    questions.push("What symptoms are you observing?".to_string());
                }
                questions.push("Can you reproduce the issue consistently?".to_string());
            }
            CodeIntent::Modify { target, .. } => {
                questions.push(format!(
                    "Should modifications to `{}` preserve backward compatibility?",
                    target.name
                ));
            }
            CodeIntent::Solve { spec, .. } => {
                if spec.relevant_files.is_empty() {
                    questions
                        .push("Which files do you think are related to this issue?".to_string());
                }
                questions.push("Are there existing tests that reproduce this issue?".to_string());
            }
            _ => {
                questions.push("Can you provide more context for this request?".to_string());
            }
        }

        questions
    }

    /// Explain why we can't generate code for this intent
    fn explain_unknowns(&self, intent: &CodeIntent) -> String {
        match intent {
            CodeIntent::Create { spec, .. } => {
                format!(
                    "Cannot confidently generate `{}`: {}. \
                     Need: specific requirements, type signatures, and examples.",
                    spec.name, spec.purpose
                )
            }
            CodeIntent::Debug { target, .. } => {
                format!(
                    "Cannot diagnose `{}` without more information. \
                     Need: error messages, stack traces, and reproduction steps.",
                    target.name
                )
            }
            CodeIntent::Solve { spec, .. } => {
                format!(
                    "Cannot autonomously solve issue without repository context. \
                     Description: {}. Need: repository access and clearly defined issue.",
                    spec.description
                )
            }
            _ => "Insufficient information to proceed. Please provide more context.".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::code_encoder::CodeHDEncoder;
    use crate::language::code_intent::{CodeSpec, CodeTarget};
    use crate::language::code_parser::EntityKind;

    fn make_generator() -> EpistemicCodeGenerator {
        let generated = CodeGenerator::with_default_dim();
        EpistemicCodeGenerator::new(generated, 512)
    }

    #[test]
    fn test_confident_generation() {
        let generated = make_generator();
        let intent = CodeIntent::Create {
            target: CodeTarget::new("add", EntityKind::Function),
            spec: CodeSpec::new("rust", "add", "Add two numbers")
                .with_signature("fn add(a: i32, b: i32) -> i32")
                .with_example("add(1, 2)", "3")
                .with_epistemic(EpistemicStatus::Certain),
        };

        let result = generated.generate_with_confidence(&intent, &CodeContext::default());
        assert!(result.is_ready());
        assert!(result.source().is_some());
        assert_eq!(result.epistemic_status(), EpistemicStatus::Certain);
    }

    #[test]
    fn test_probable_generation() {
        let generated = make_generator();
        let intent = CodeIntent::Create {
            target: CodeTarget::new("process", EntityKind::Function),
            spec: CodeSpec::new("rust", "process", "Process data")
                .with_epistemic(EpistemicStatus::Probable),
        };

        let result = generated.generate_with_confidence(&intent, &CodeContext::default());
        assert!(!result.is_ready());
        assert!(result.source().is_some());
        assert!(!result.notes().is_empty());
    }

    #[test]
    fn test_unknown_generation() {
        let generated = make_generator();
        let intent = CodeIntent::Create {
            target: CodeTarget::new("mystery", EntityKind::Function),
            spec: CodeSpec::new("rust", "mystery", "Do something")
                .with_epistemic(EpistemicStatus::Unknown),
        };

        let result = generated.generate_with_confidence(&intent, &CodeContext::default());
        assert!(!result.is_ready());
        assert!(result.source().is_none());
        assert_eq!(result.epistemic_status(), EpistemicStatus::Unknown);
    }

    #[test]
    fn test_uncertain_asks_questions() {
        let generated = make_generator();
        let intent = CodeIntent::Create {
            target: CodeTarget::new("transform", EntityKind::Function),
            spec: CodeSpec::new("rust", "transform", "Transform data")
                .with_epistemic(EpistemicStatus::Uncertain),
        };

        let result = generated.generate_with_confidence(&intent, &CodeContext::default());
        assert!(!result.is_ready());
        assert!(!result.notes().is_empty()); // Should have questions
    }
}
