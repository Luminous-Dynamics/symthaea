// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generic Structural Scorer Integration
//! Example showing how to wire the new walkers into a GenericStructuralScorer.

use crate::go_walker::GoWalker;
use crate::python_walker::PythonWalker;
use symthaea_broca::rust_walker::{LanguageWalker, RustWalker, StructuralElement};
use std::collections::HashSet;

pub struct GenericStructuralScorer {
    rust_walker: RustWalker,
    python_walker: PythonWalker,
    go_walker: GoWalker,
}

impl GenericStructuralScorer {
    pub fn new_with_all() -> Self {
        Self {
            rust_walker: RustWalker::new(),
            python_walker: PythonWalker::new(),
            go_walker: GoWalker::new(),
        }
    }

    pub fn score_rust(&mut self, generated: &str, golden: &str) -> StructuralVerdict {
        let gen_elems = self.rust_walker.extract_elements(generated);
        let gold_elems = self.rust_walker.extract_elements(golden);
        Self::compare_elements(&gen_elems, &gold_elems)
    }

    pub fn score_python(&mut self, generated: &str, golden: &str) -> StructuralVerdict {
        let gen_elems = self.python_walker.extract_elements(generated);
        let gold_elems = self.python_walker.extract_elements(golden);
        Self::compare_elements(&gen_elems, &gold_elems)
    }

    pub fn score_go(&mut self, generated: &str, golden: &str) -> StructuralVerdict {
        let gen_elems = self.go_walker.extract_elements(generated);
        let gold_elems = self.go_walker.extract_elements(golden);
        Self::compare_elements(&gen_elems, &gold_elems)
    }

    fn compare_elements(
        r#gen: &[StructuralElement],
        gold: &[StructuralElement],
    ) -> StructuralVerdict {
        // Simple Jaccard + missing path detection
        let gen_set: HashSet<_> = r#gen.iter().map(|e| e.dotted_path.clone()).collect();
        let gold_set: HashSet<_> = gold.iter().map(|e| e.dotted_path.clone()).collect();

        let missing: Vec<_> = gold_set.difference(&gen_set).cloned().collect();
        let extra: Vec<_> = gen_set.difference(&gold_set).cloned().collect();

        let jaccard = if gen_set.is_empty() && gold_set.is_empty() {
            1.0
        } else {
            let inter = gen_set.intersection(&gold_set).count() as f32;
            inter / (gen_set.len() + gold_set.len() - inter as usize) as f32
        };

        StructuralVerdict {
            pass: missing.is_empty() && extra.is_empty(),
            missing_required: missing,
            extraneous: extra,
            jaccard_similarity: jaccard,
        }
    }
}

#[derive(Debug, Default)]
pub struct StructuralVerdict {
    pub pass: bool,
    pub missing_required: Vec<String>,
    pub extraneous: Vec<String>,
    pub jaccard_similarity: f32,
}
