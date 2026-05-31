// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Coding prediction-error signals for the verified generation loop.
//!
//! This is the narrow "compiler as environment" bridge: convert rustc/test
//! feedback into typed surprise events, diagnostic HDC geometry, and repair
//! hints that can guide the next generation attempt.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use symthaea_core::hdc::ContinuousHV;

use super::code_executor::{
    CompileError, ErrorCategory, ExecutionResult, parse_json_diagnostics, parse_structured_errors,
};
use crate::hdc::diagnostic_encoder::DiagnosticHDEncoder;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodingFeedbackKind {
    Compile,
    Test,
    Runtime,
    Structural,
}

#[derive(Debug, Clone)]
pub struct CodingPredictionError {
    pub key: String,
    pub kind: CodingFeedbackKind,
    pub category: String,
    pub diagnostic: String,
    pub error_code: Option<String>,
    pub surprise: f32,
    pub retry_number: usize,
    pub diagnostic_hv: ContinuousHV,
}

pub fn prediction_errors_from_execution(
    result: &ExecutionResult,
    retry_number: usize,
) -> Vec<CodingPredictionError> {
    let surprise = result.to_surprise();

    if !result.compiled {
        return compile_prediction_errors(&result.compile_errors, retry_number, surprise);
    }

    if result.tests_failed > 0 {
        return test_prediction_errors(result, retry_number, surprise);
    }

    if let Some(runtime_error) = &result.runtime_error {
        return vec![text_prediction_error(
            CodingFeedbackKind::Runtime,
            "runtime_error",
            runtime_error,
            None,
            retry_number,
            surprise.max(0.5),
        )];
    }

    Vec::new()
}

pub fn structural_prediction_error_from_ast_parse(
    diagnostic: impl Into<String>,
    retry_number: usize,
) -> CodingPredictionError {
    let diagnostic = diagnostic.into();
    text_prediction_error(
        CodingFeedbackKind::Structural,
        "ast_parse_failure",
        &diagnostic,
        None,
        retry_number,
        0.75,
    )
}

pub fn structural_prediction_error_from_prior(
    diagnostic: impl Into<String>,
    retry_number: usize,
    surprise: f32,
) -> CodingPredictionError {
    let diagnostic = diagnostic.into();
    text_prediction_error(
        CodingFeedbackKind::Structural,
        "structural_prior_mismatch",
        &diagnostic,
        None,
        retry_number,
        surprise.clamp(0.0, 1.0),
    )
}

pub fn prediction_error_hints(errors: &[CodingPredictionError]) -> Vec<(String, String)> {
    if std::env::var_os("SYMTHAEA_DISABLE_FEP_REPAIR_HINTS").is_some() {
        return Vec::new();
    }

    errors
        .iter()
        .enumerate()
        .map(|(idx, error)| {
            let label = format!(
                "fep_prediction_error_{}_{}",
                idx,
                sanitize_label(&error.category)
            );
            let code = error
                .error_code
                .as_deref()
                .map(|code| format!(" ({code})"))
                .unwrap_or_default();
            let hint = format!(
                "Prediction-error signal {:?}/{}{} at retry {} has surprise {:.2}. Repair the smallest local cause: {}",
                error.kind,
                error.category,
                code,
                error.retry_number,
                error.surprise,
                error.diagnostic
            );
            (label, hint)
        })
        .collect()
}

pub fn prediction_error_diagnostics(errors: &[CodingPredictionError]) -> Vec<String> {
    errors
        .iter()
        .map(|error| error.diagnostic.clone())
        .collect()
}

pub fn prediction_error_categories(errors: &[CodingPredictionError]) -> Vec<String> {
    errors.iter().map(|error| error.category.clone()).collect()
}

fn compile_prediction_errors(
    compile_errors: &[String],
    retry_number: usize,
    surprise: f32,
) -> Vec<CodingPredictionError> {
    let stderr = compile_errors.join("\n");
    let structured = structured_compile_errors(&stderr);
    let encoder = DiagnosticHDEncoder::default_dim();

    if !structured.is_empty() {
        return structured
            .into_iter()
            .map(|error| structured_prediction_error(&encoder, error, retry_number, surprise))
            .collect();
    }

    compile_errors
        .iter()
        .map(|error| {
            text_prediction_error(
                CodingFeedbackKind::Compile,
                categorize_raw_compile_error(error),
                error,
                None,
                retry_number,
                surprise,
            )
        })
        .collect()
}

pub fn structured_compile_errors(stderr: &str) -> Vec<CompileError> {
    let json_errors = parse_json_diagnostics(stderr);
    if !json_errors.is_empty() {
        json_errors
    } else {
        parse_structured_errors(stderr)
    }
}

fn structured_prediction_error(
    encoder: &DiagnosticHDEncoder,
    error: CompileError,
    retry_number: usize,
    surprise: f32,
) -> CodingPredictionError {
    let diagnostic_hv = encoder.encode_diagnostic(&error);
    let diagnostic = match (&error.file, error.line, error.column) {
        (Some(file), Some(line), Some(column)) => {
            format!("{} at {file}:{line}:{column}", error.message)
        }
        _ => error.message.clone(),
    };

    CodingPredictionError {
        key: prediction_error_key(
            CodingFeedbackKind::Compile,
            error_category_label(error.category),
            error.code.as_deref(),
            &diagnostic,
        ),
        kind: CodingFeedbackKind::Compile,
        category: error_category_label(error.category).to_string(),
        diagnostic,
        error_code: error.code,
        surprise,
        retry_number,
        diagnostic_hv,
    }
}

fn test_prediction_errors(
    result: &ExecutionResult,
    retry_number: usize,
    surprise: f32,
) -> Vec<CodingPredictionError> {
    let constraints = result.failure_constraints();
    let diagnostics = if constraints.is_empty() {
        vec![result.test_output.clone()]
    } else {
        constraints
    };

    diagnostics
        .into_iter()
        .filter(|diagnostic| !diagnostic.trim().is_empty())
        .map(|diagnostic| {
            text_prediction_error(
                CodingFeedbackKind::Test,
                "test_failure",
                &diagnostic,
                None,
                retry_number,
                surprise,
            )
        })
        .collect()
}

fn text_prediction_error(
    kind: CodingFeedbackKind,
    category: &str,
    diagnostic: &str,
    error_code: Option<String>,
    retry_number: usize,
    surprise: f32,
) -> CodingPredictionError {
    CodingPredictionError {
        key: prediction_error_key(kind, category, error_code.as_deref(), diagnostic),
        kind,
        category: category.to_string(),
        diagnostic: diagnostic.to_string(),
        error_code,
        surprise,
        retry_number,
        diagnostic_hv: text_hv(category, diagnostic),
    }
}

fn prediction_error_key(
    kind: CodingFeedbackKind,
    category: &str,
    error_code: Option<&str>,
    diagnostic: &str,
) -> String {
    let mut hasher = DefaultHasher::new();
    diagnostic.hash(&mut hasher);
    format!(
        "{:?}:{}:{}:{:016x}",
        kind,
        category,
        error_code.unwrap_or("none"),
        hasher.finish()
    )
}

fn text_hv(category: &str, diagnostic: &str) -> ContinuousHV {
    let mut hasher = DefaultHasher::new();
    category.hash(&mut hasher);
    diagnostic.hash(&mut hasher);
    ContinuousHV::random(
        symthaea_core::hdc::unified_hv::HDC_DIMENSION,
        hasher.finish(),
    )
}

fn error_category_label(category: ErrorCategory) -> &'static str {
    match category {
        ErrorCategory::TypeMismatch => "type_mismatch",
        ErrorCategory::MissingImport => "missing_import",
        ErrorCategory::BorrowError => "borrow_error",
        ErrorCategory::MovedValue => "moved_value",
        ErrorCategory::LifetimeError => "lifetime_error",
        ErrorCategory::VisibilityError => "visibility_error",
        ErrorCategory::UnusedCode => "unused_code",
        ErrorCategory::MissingImpl => "missing_impl",
        ErrorCategory::UndeclaredGeneric => "undeclared_generic",
        ErrorCategory::UnwantedMain => "unwanted_main",
        ErrorCategory::SyntaxError => "syntax_error",
        ErrorCategory::Timeout => "timeout",
        ErrorCategory::LinkerError => "linker_error",
        ErrorCategory::SandboxError => "sandbox_error",
        ErrorCategory::Other => "rustc_error",
    }
}

fn categorize_raw_compile_error(error: &str) -> &'static str {
    let lower = error.to_ascii_lowercase();
    if lower.contains("e0308") || lower.contains("mismatched types") {
        "type_mismatch"
    } else if lower.contains("e0425") || lower.contains("not found in this scope") {
        "unresolved_identifier"
    } else if lower.contains("e0382") || lower.contains("use of moved value") {
        "use_after_move"
    } else if lower.contains("e0507") || lower.contains("cannot move out") {
        "move_out_of_borrow"
    } else if lower.contains("e0596") || lower.contains("cannot borrow") {
        "borrow_mutability"
    } else if lower.contains("lifetime") {
        "lifetime_error"
    } else {
        "rustc_error"
    }
}

fn sanitize_label(label: &str) -> String {
    label
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn compiler_diagnostic_becomes_prediction_error() {
        let result = ExecutionResult {
            compiled: false,
            compile_errors: vec![
                "error[E0308]: mismatched types\n --> generated.rs:2:5\n  |\n2 |     \"x\"\n  |     ^^^ expected `i32`, found `&str`"
                    .to_string(),
            ],
            tests_passed: 0,
            tests_failed: 0,
            test_output: String::new(),
            runtime_error: None,
            elapsed: Duration::from_millis(1),
            simulated: false,
            binary_path: None,
            test_failures: Vec::new(),
        };

        let errors = prediction_errors_from_execution(&result, 1);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, CodingFeedbackKind::Compile);
        assert_eq!(errors[0].category, "type_mismatch");
        assert_eq!(errors[0].error_code.as_deref(), Some("E0308"));
        assert!(errors[0].surprise > 0.8);
    }

    #[test]
    fn test_failure_becomes_prediction_error_hint() {
        let mut result = ExecutionResult {
            compiled: true,
            compile_errors: Vec::new(),
            tests_passed: 1,
            tests_failed: 1,
            test_output: "---- tests::case stdout ----\nthread 'tests::case' panicked at 'assertion failed'\n  left: 1\n right: 2"
                .to_string(),
            runtime_error: None,
            elapsed: Duration::from_millis(1),
            simulated: false,
            binary_path: None,
            test_failures: Vec::new(),
        };
        result.parse_test_failures();

        let errors = prediction_errors_from_execution(&result, 2);
        let hints = prediction_error_hints(&errors);

        assert_eq!(errors[0].kind, CodingFeedbackKind::Test);
        assert_eq!(errors[0].category, "test_failure");
        assert!(hints[0].1.contains("Prediction-error signal"));
        assert!(hints[0].1.contains("surprise"));
    }

    #[test]
    fn ast_parse_failure_becomes_structural_prediction_error() {
        let error = structural_prediction_error_from_ast_parse("expected expression", 3);

        assert_eq!(error.kind, CodingFeedbackKind::Structural);
        assert_eq!(error.category, "ast_parse_failure");
        assert_eq!(error.retry_number, 3);
        assert!(error.surprise >= 0.75);
    }

    #[test]
    fn low_structural_prior_becomes_prediction_error() {
        let error = structural_prediction_error_from_prior("low AST prior similarity", 2, 0.42);

        assert_eq!(error.kind, CodingFeedbackKind::Structural);
        assert_eq!(error.category, "structural_prior_mismatch");
        assert_eq!(error.retry_number, 2);
        assert_eq!(error.surprise, 0.42);
    }
}
