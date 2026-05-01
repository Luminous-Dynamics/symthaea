// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Diagnostic HDC Encoder — Structured errors → Hypervectors
//!
//! Encodes compiler diagnostics and test failures into 16,384-D hypervectors
//! to provide the generator with a "mathematical geometry" of the error state.

use crate::language::code_executor::{CompileError, ErrorCategory};
use std::collections::HashMap;
use symthaea_core::hdc::ContinuousHV;

/// Roles in the diagnostic hypergraph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiagnosticRole {
    ErrorCode,
    Category,
    File,
    Line,
    Message,
    ExpectedType,
    FoundType,
    SymbolName,
    SymbolKind,
}

/// Encoder for translating structured diagnostics into HDC space
pub struct DiagnosticHDEncoder {
    dim: usize,
    /// Base vectors for each role in a diagnostic
    role_vectors: HashMap<DiagnosticRole, ContinuousHV>,
    /// Base vectors for each error category
    category_vectors: HashMap<ErrorCategory, ContinuousHV>,
}

impl DiagnosticHDEncoder {
    /// Create a new encoder with the given dimension
    pub fn new(dim: usize) -> Self {
        let mut encoder = Self {
            dim,
            role_vectors: HashMap::new(),
            category_vectors: HashMap::new(),
        };
        encoder.init_role_vectors();
        encoder.init_category_vectors();
        encoder
    }

    /// Create with default 16,384-D dimension
    pub fn default_dim() -> Self {
        Self::new(symthaea_core::hdc::unified_hv::HDC_DIMENSION)
    }

    fn init_role_vectors(&mut self) {
        let roles = [
            DiagnosticRole::ErrorCode,
            DiagnosticRole::Category,
            DiagnosticRole::File,
            DiagnosticRole::Line,
            DiagnosticRole::Message,
            DiagnosticRole::ExpectedType,
            DiagnosticRole::FoundType,
            DiagnosticRole::SymbolName,
            DiagnosticRole::SymbolKind,
        ];
        for (i, role) in roles.iter().enumerate() {
            let seed = (i as u64 + 5000) * 2_654_435_761;
            self.role_vectors
                .insert(*role, ContinuousHV::random(self.dim, seed));
        }
    }

    fn init_category_vectors(&mut self) {
        let categories = [
            ErrorCategory::TypeMismatch,
            ErrorCategory::MissingImport,
            ErrorCategory::BorrowError,
            ErrorCategory::MovedValue,
            ErrorCategory::LifetimeError,
            ErrorCategory::VisibilityError,
            ErrorCategory::UndeclaredGeneric,
            ErrorCategory::MissingImpl,
            ErrorCategory::UnusedCode,
            ErrorCategory::LinkerError,
            ErrorCategory::SandboxError,
            ErrorCategory::Other,
        ];
        for (i, cat) in categories.iter().enumerate() {
            let seed = (i as u64 + 6000) * 3_141_592_653;
            self.category_vectors
                .insert(*cat, ContinuousHV::random(self.dim, seed));
        }
    }

    /// Encode a single structured compilation error into a hypervector
    pub fn encode_diagnostic(&self, error: &CompileError) -> ContinuousHV {
        let mut bundle = Vec::new();

        // 1. Error Code (e.g., E0308)
        if let Some(code) = &error.code {
            let code_hv = self.bind_string(code);
            bundle.push(self.role_vectors[&DiagnosticRole::ErrorCode].bind(&code_hv));
        }

        // 2. Category (e.g., TypeMismatch)
        if let Some(cat_hv) = self.category_vectors.get(&error.category) {
            bundle.push(self.role_vectors[&DiagnosticRole::Category].bind(cat_hv));
        }

        // 3. Message (HDC semantic hash)
        let msg_hv = self.bind_string(&error.message);
        bundle.push(self.role_vectors[&DiagnosticRole::Message].bind(&msg_hv));

        // 4. File & Line (Spatial context)
        if let Some(file) = &error.file {
            let file_hv = self.bind_string(file);
            bundle.push(self.role_vectors[&DiagnosticRole::File].bind(&file_hv));
        }
        if let Some(line) = error.line {
            let line_hv = ContinuousHV::random(self.dim, line as u64);
            bundle.push(self.role_vectors[&DiagnosticRole::Line].bind(&line_hv));
        }

        // 5. Advanced Semantic Extraction (Type Mismatch)
        if error.category == ErrorCategory::TypeMismatch {
            if let Some((expected, found)) = extract_types_from_message(&error.message) {
                let exp_hv = self.bind_string(&expected);
                let fnd_hv = self.bind_string(&found);
                bundle.push(self.role_vectors[&DiagnosticRole::ExpectedType].bind(&exp_hv));
                bundle.push(self.role_vectors[&DiagnosticRole::FoundType].bind(&fnd_hv));
            }
        }

        if bundle.is_empty() {
            return ContinuousHV::random(self.dim, 999);
        }

        ContinuousHV::bundle(&bundle)
    }

    /// Encode multiple diagnostics into a single "Surprise" hypervector
    pub fn encode_diagnostics(&self, errors: &[CompileError]) -> ContinuousHV {
        if errors.is_empty() {
            return ContinuousHV::random(self.dim, 0);
        }
        let hvs: Vec<ContinuousHV> = errors.iter().map(|e| self.encode_diagnostic(e)).collect();
        ContinuousHV::bundle(&hvs)
    }

    /// Deterministically map a string to a hypervector
    fn bind_string(&self, s: &str) -> ContinuousHV {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        ContinuousHV::random(self.dim, hasher.finish())
    }
}

/// Helper to extract "expected `X`, found `Y`" patterns from rustc messages
fn extract_types_from_message(msg: &str) -> Option<(String, String)> {
    // Basic heuristic: look for "expected `...`, found `...`"
    if let (Some(exp_start), Some(fnd_start)) = (msg.find("expected `"), msg.find("found `")) {
        let exp_end = msg[exp_start + 10..].find('`')?;
        let fnd_end = msg[fnd_start + 7..].find('`')?;

        let expected = msg[exp_start + 10..exp_start + 10 + exp_end].to_string();
        let found = msg[fnd_start + 7..fnd_start + 7 + fnd_end].to_string();

        return Some((expected, found));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::code_executor::ErrorCategory;

    #[test]
    fn test_diagnostic_encoding_similarity() {
        let encoder = DiagnosticHDEncoder::new(1024);

        let err1 = CompileError {
            message: "expected `String`, found `&str`".to_string(),
            code: Some("E0308".to_string()),
            file: Some("lib.rs".to_string()),
            line: Some(10),
            column: Some(5),
            category: ErrorCategory::TypeMismatch,
            suggested_replacement: None,
        };

        let err2 = CompileError {
            message: "expected `i32`, found `u32`".to_string(),
            code: Some("E0308".to_string()),
            file: Some("lib.rs".to_string()),
            line: Some(12),
            column: Some(8),
            category: ErrorCategory::TypeMismatch,
            suggested_replacement: None,
        };

        let hv1 = encoder.encode_diagnostic(&err1);
        let hv2 = encoder.encode_diagnostic(&err2);

        // Should have high similarity due to same code and category
        let sim = hv1.similarity(&hv2);
        assert!(
            sim > 0.4,
            "Similarity should be significant for same error code: {sim}"
        );
    }
}
