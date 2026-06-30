// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Codebase Bridge — Read-only indexing of her own substrate.
//!
//! Allows Broca to map her repository structure into a high-dimensional
//! structural manifold for strategic refactoring.

use crate::rust_walker::{LanguageWalker, RustWalker};
use symthaea_core::hdc::ContinuousHV;

/// A single entry in the codebase manifold.
#[derive(Clone)]
pub struct CodebaseElement {
    pub path: String,
    pub symbol: String,
    pub kind: String,
    pub structural_hv: ContinuousHV,
}

#[derive(Clone)]
pub struct CodebaseBridge {
    pub root_dir: String,
    pub elements: Vec<CodebaseElement>,
}

impl CodebaseBridge {
    pub fn new(root: &str) -> Self {
        Self {
            root_dir: root.to_string(),
            elements: Vec::new(),
        }
    }

    /// Index a Rust file into the structural manifold.
    pub fn index_file(&mut self, relative_path: &str, code: &str) -> usize {
        let mut walker = RustWalker::new();
        let extracted = walker.extract_elements(code);
        let count = extracted.len();

        for elem in extracted {
            // Generate a structural HV based on the symbol name and kind
            // (Simplified: using a random HV derived from a hash)
            let hv = ContinuousHV::random(16384, elem.dotted_path.len() as u64);

            self.elements.push(CodebaseElement {
                path: relative_path.to_string(),
                symbol: elem.dotted_path,
                kind: elem.kind,
                structural_hv: hv,
            });
        }
        count
    }

    /// Find elements similar to a given semantic intent.
    pub fn search_symbols(&self, query_hv: &ContinuousHV, top_k: usize) -> Vec<&CodebaseElement> {
        let mut scored: Vec<(&CodebaseElement, f32)> = self
            .elements
            .iter()
            .map(|e| (e, e.structural_hv.similarity(query_hv)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(top_k).map(|(e, _)| e).collect()
    }
}
