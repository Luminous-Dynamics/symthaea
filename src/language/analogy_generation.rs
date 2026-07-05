// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Analogy-Based Code Generation — Solve new problems by structural similarity
//!
//! Instead of keyword matching or LLM generation, this module:
//! 1. Indexes a library of known-good solutions (e.g., Exercism canonical)
//! 2. For a new problem: finds the most structurally similar solved problem
//! 3. Retrieves that solution's source code as a template
//!
//! This is genuine cognitive reasoning — not statistical generation, not
//! keyword lookup. The system understands "this new problem looks like
//! that solved one" through HDC similarity in hypervector space.
//!
//! # Architecture
//!
//! ```text
//! Known Solutions Library
//!     ↓ index (purpose_hv → solution_source)
//! SolutionMemory { entries: Vec<(purpose_hv, source)> }
//!     ↓ query(new_purpose)
//! encode(new_purpose) → find_similar → retrieve solution
//!     ↓ adapt
//! Rename functions/types to match new problem's signature
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use quote::ToTokens;
use symthaea_core::hdc::ContinuousHV;
use syn::visit::{self, Visit};
use syn::visit_mut::{self, VisitMut};
use syn::{FnArg, Ident, Item, ItemFn, Pat, PatIdent};

use crate::hdc::code_encoder::CodeHDEncoder;

/// A known-good solution indexed for analogy retrieval
#[derive(Debug, Clone)]
struct SolutionEntry {
    /// Exercise/problem name
    name: String,
    /// Purpose description (from docs/instructions.md)
    purpose: String,
    /// HDC encoding of the purpose
    purpose_hv: ContinuousHV,
    /// The canonical solution source code
    solution_source: String,
    /// Function signature from the stub
    signature: String,
}

/// Library of known solutions for analogy-based code generation.
pub struct SolutionLibrary {
    encoder: CodeHDEncoder,
    entries: Vec<SolutionEntry>,
}

/// Result of an analogy query
#[derive(Debug, Clone)]
pub struct AnalogyResult {
    /// The matched solution's name
    pub matched_name: String,
    /// Similarity score (0.0-1.0)
    pub similarity: f32,
    /// The adapted solution source code
    pub source: String,
    /// The original solution's purpose (for debugging)
    pub original_purpose: String,
}

impl SolutionLibrary {
    /// Create a new empty solution library
    pub fn new(dim: usize) -> Self {
        Self {
            encoder: CodeHDEncoder::new(dim),
            entries: Vec::new(),
        }
    }

    /// Add a solution to the library
    pub fn add_solution(&mut self, name: &str, purpose: &str, signature: &str, source: &str) {
        let purpose_hv = self.encoder.encode_name(purpose);
        self.entries.push(SolutionEntry {
            name: name.to_string(),
            purpose: purpose.to_string(),
            purpose_hv,
            solution_source: source.to_string(),
            signature: signature.to_string(),
        });
    }

    /// Number of indexed solutions
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the library is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Find the most similar solution for a given purpose.
    ///
    /// Returns the top-k matches ranked by cosine similarity between
    /// the query purpose and indexed solution purposes.
    pub fn find_similar(&self, purpose: &str, top_k: usize) -> Vec<AnalogyResult> {
        if self.entries.is_empty() {
            return Vec::new();
        }

        let query_hv = self.encoder.encode_name(purpose);

        let mut scored: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, entry)| (i, query_hv.similarity(&entry.purpose_hv)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(top_k)
            .map(|(idx, sim)| {
                let entry = &self.entries[idx];
                AnalogyResult {
                    matched_name: entry.name.clone(),
                    similarity: sim,
                    source: entry.solution_source.clone(),
                    original_purpose: entry.purpose.clone(),
                }
            })
            .collect()
    }

    /// Generate code for a new problem by finding the most similar solved problem
    /// and adapting its solution.
    ///
    /// Returns `Some(source)` if a sufficiently similar solution is found,
    /// `None` otherwise (falls through to other generation methods).
    pub fn generate_by_analogy(
        &self,
        purpose: &str,
        target_fn_name: &str,
        target_signature: &str,
        min_similarity: f32,
    ) -> Option<AnalogyResult> {
        let matches = self.find_similar(purpose, 5);

        // Find the best match that is:
        // 1. Above similarity threshold
        // 2. A simple solution (single function, not multi-struct module)
        // 3. Not too complex (< 30 lines — complex solutions break when transplanted)
        for candidate in matches {
            if candidate.similarity < min_similarity {
                break; // sorted by similarity, so all remaining are lower
            }

            let source = &candidate.source;

            // Skip complex solutions — they won't adapt well
            let line_count = source.lines().count();
            let struct_count = source.matches("struct ").count();
            let impl_count = source.matches("impl ").count();
            let fn_count = source.matches("pub fn ").count() + source.matches("fn ").count();

            // Only use simple solutions: single function, < 30 lines, no structs
            if line_count > 30 || struct_count > 0 || impl_count > 1 || fn_count > 3 {
                continue; // Too complex — try next match
            }

            let entry = self
                .entries
                .iter()
                .find(|e| e.name == candidate.matched_name)?;

            let adapted_source =
                match adapt_solution(source, &entry.signature, target_fn_name, target_signature) {
                    Some(source) => source,
                    None => continue,
                };

            // Sanity check: adapted source should have balanced braces
            let opens = adapted_source.matches('{').count();
            let closes = adapted_source.matches('}').count();
            if opens != closes {
                continue; // Malformed — try next match
            }

            return Some(AnalogyResult {
                source: adapted_source,
                ..candidate
            });
        }

        None
    }

    /// Load solutions from an Exercism exercises directory.
    ///
    /// Scans all exercise directories for:
    /// - `.meta/example.rs` — canonical solution
    /// - `.docs/instructions.md` — problem description
    /// - `src/lib.rs` — function signature
    pub fn load_exercism_solutions(&mut self, exercises_dir: &Path) -> usize {
        let mut count = 0;

        let entries = match std::fs::read_dir(exercises_dir) {
            Ok(entries) => entries,
            Err(_) => return 0,
        };

        for entry in entries.flatten() {
            if !entry.file_type().map_or(false, |t| t.is_dir()) {
                continue;
            }

            let dir = entry.path();
            let name = dir
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();

            // Read canonical solution
            let solution_path = dir.join(".meta/example.rs");
            let solution = match std::fs::read_to_string(&solution_path) {
                Ok(s) => s,
                Err(_) => continue,
            };

            // Read problem description
            let instructions_path = dir.join(".docs/instructions.md");
            let purpose = if let Ok(instructions) = std::fs::read_to_string(&instructions_path) {
                // Extract first meaningful paragraph
                instructions
                    .lines()
                    .find(|l| !l.trim().starts_with('#') && !l.trim().is_empty())
                    .unwrap_or(&name)
                    .to_string()
            } else {
                name.replace('-', " ")
            };

            // Read function signature from stub
            let lib_path = dir.join("src/lib.rs");
            let signature = if let Ok(lib) = std::fs::read_to_string(&lib_path) {
                lib.lines()
                    .find(|l| l.trim().starts_with("pub fn "))
                    .map(|l| l.split('{').next().unwrap_or(l).trim().to_string())
                    .unwrap_or_default()
            } else {
                String::new()
            };

            self.add_solution(&name, &purpose, &signature, &solution);
            count += 1;
        }

        count
    }
}

/// Adapt a solution from one problem to another using AST-level grafting.
///
/// The donor body is preserved, but the function visibility/signature are
/// replaced by the target signature. Parameter identifiers are renamed through
/// `syn` identifier nodes, never by substring replacement.
fn adapt_solution(
    source: &str,
    _original_sig: &str,
    target_fn_name: &str,
    target_signature: &str,
) -> Option<String> {
    let mut donor_fn = parse_first_function(source)?;
    let target_fn = parse_signature_as_function(target_signature, target_fn_name)?;

    let param_renames = parameter_renames(&donor_fn, &target_fn);
    if !param_renames.is_empty() {
        let shadowed = collect_body_bindings(&donor_fn);
        if param_renames.keys().any(|name| shadowed.contains_key(name)) {
            return None;
        }

        ParamRenamer {
            renames: &param_renames,
        }
        .visit_block_mut(&mut donor_fn.block);
    }

    donor_fn.vis = target_fn.vis;
    donor_fn.sig = target_fn.sig;

    Some(donor_fn.into_token_stream().to_string())
}

fn parse_first_function(source: &str) -> Option<ItemFn> {
    if let Ok(item_fn) = syn::parse_str::<ItemFn>(source) {
        return Some(item_fn);
    }

    let file = syn::parse_file(source).ok()?;
    file.items.into_iter().find_map(|item| match item {
        Item::Fn(item_fn) => Some(item_fn),
        _ => None,
    })
}

fn parse_signature_as_function(signature: &str, fallback_name: &str) -> Option<ItemFn> {
    let sig = signature.trim().trim_end_matches(';');
    let mut item_fn = syn::parse_str::<ItemFn>(&format!("{sig} {{}}")).ok()?;

    if item_fn.sig.ident.to_string().is_empty() {
        item_fn.sig.ident = Ident::new(fallback_name, item_fn.sig.ident.span());
    }

    Some(item_fn)
}

fn parameter_renames(donor_fn: &ItemFn, target_fn: &ItemFn) -> HashMap<String, Ident> {
    donor_fn
        .sig
        .inputs
        .iter()
        .filter_map(fn_arg_ident)
        .zip(target_fn.sig.inputs.iter().filter_map(fn_arg_ident))
        .filter_map(|(from, to)| {
            let from_name = from.to_string();
            if *to == from_name {
                None
            } else {
                Some((from_name, to.clone()))
            }
        })
        .collect()
}

fn fn_arg_ident(arg: &FnArg) -> Option<&Ident> {
    match arg {
        FnArg::Typed(pat_type) => pat_ident(&pat_type.pat),
        FnArg::Receiver(_) => None,
    }
}

fn pat_ident(pat: &Pat) -> Option<&Ident> {
    match pat {
        Pat::Ident(PatIdent { ident, .. }) => Some(ident),
        _ => None,
    }
}

fn collect_body_bindings(item_fn: &ItemFn) -> HashMap<String, ()> {
    let mut collector = BodyBindingCollector {
        bindings: HashMap::new(),
    };
    collector.visit_block(&item_fn.block);
    collector.bindings
}

struct BodyBindingCollector {
    bindings: HashMap<String, ()>,
}

impl<'ast> Visit<'ast> for BodyBindingCollector {
    fn visit_pat_ident(&mut self, pat_ident: &'ast PatIdent) {
        self.bindings.insert(pat_ident.ident.to_string(), ());
        visit::visit_pat_ident(self, pat_ident);
    }
}

struct ParamRenamer<'a> {
    renames: &'a HashMap<String, Ident>,
}

impl VisitMut for ParamRenamer<'_> {
    fn visit_expr_path_mut(&mut self, expr_path: &mut syn::ExprPath) {
        if expr_path.qself.is_none() && expr_path.path.segments.len() == 1 {
            let segment = &mut expr_path.path.segments[0];
            if let Some(new_ident) = self.renames.get(&segment.ident.to_string()) {
                segment.ident = new_ident.clone();
            }
        }

        visit_mut::visit_expr_path_mut(self, expr_path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_find() {
        let mut lib = SolutionLibrary::new(512);
        lib.add_solution(
            "reverse-string",
            "Reverse a given string",
            "pub fn reverse(input: &str) -> String",
            "pub fn reverse(input: &str) -> String {\n    input.chars().rev().collect()\n}",
        );
        lib.add_solution(
            "leap",
            "Determine whether a given year is a leap year",
            "pub fn is_leap_year(year: u64) -> bool",
            "pub fn is_leap_year(year: u64) -> bool {\n    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0\n}",
        );

        assert_eq!(lib.len(), 2);

        // Query for something similar to reverse
        let matches = lib.find_similar("Reverse a string", 2);
        assert!(!matches.is_empty());
        assert_eq!(matches[0].matched_name, "reverse-string");
    }

    #[test]
    fn test_generate_by_analogy() {
        let mut lib = SolutionLibrary::new(512);
        lib.add_solution(
            "leap",
            "Determine whether a given year is a leap year",
            "pub fn is_leap_year(year: u64) -> bool",
            "pub fn is_leap_year(year: u64) -> bool {\n    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0\n}",
        );

        // Query with very similar purpose
        let result = lib.generate_by_analogy(
            "Check if a year is a leap year",
            "check_leap",
            "pub fn check_leap(y: u64) -> bool",
            0.3,
        );

        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.source.contains("pub fn check_leap"));
    }

    #[test]
    fn test_low_similarity_returns_none() {
        let mut lib = SolutionLibrary::new(512);
        lib.add_solution(
            "leap",
            "Determine whether a given year is a leap year",
            "pub fn is_leap_year(year: u64) -> bool",
            "pub fn is_leap_year(year: u64) -> bool { true }",
        );

        // Completely unrelated query with high threshold
        let result = lib.generate_by_analogy(
            "Parse a complex JSON document into a tree structure",
            "parse_json",
            "pub fn parse_json(s: &str) -> Tree",
            0.9, // Very high threshold
        );

        assert!(result.is_none());
    }

    #[test]
    fn test_adapt_solution() {
        let source =
            "pub fn reverse(input: &str) -> String {\n    input.chars().rev().collect()\n}";
        let adapted = adapt_solution(
            source,
            "pub fn reverse(input: &str) -> String",
            "my_reverse",
            "pub fn my_reverse(s: &str) -> String",
        )
        .expect("AST graft should succeed");
        assert!(adapted.contains("pub fn my_reverse"));
        assert!(adapted.contains("s . chars"));
        syn::parse_str::<ItemFn>(&adapted).expect("adapted source should remain a Rust function");
    }

    #[test]
    fn test_adapt_solution_preserves_literals_while_renaming_identifiers() {
        let source = r#"pub fn label(input: &str) -> String {
    format!("input={}", input)
}"#;

        let adapted = adapt_solution(
            source,
            "pub fn label(input: &str) -> String",
            "tag",
            "pub fn tag(value: &str) -> String",
        )
        .expect("AST graft should succeed");

        assert!(adapted.contains("pub fn tag"));
        assert!(adapted.contains("\"input={}\""));
        assert!(adapted.contains("value"));
        assert!(!adapted.contains("format ! (\"value={}\""));
        syn::parse_str::<ItemFn>(&adapted).expect("adapted source should remain a Rust function");
    }

    #[test]
    fn test_adapt_solution_handles_multiline_generic_signature() {
        let source = r#"pub fn first<'a, T: Clone>(
    items: &'a [T],
) -> Option<T> {
    items.first().cloned()
}"#;

        let adapted = adapt_solution(
            source,
            "pub fn first<'a, T: Clone>(items: &'a [T]) -> Option<T>",
            "head",
            "pub fn head<'b, U: Clone + Default>(values: &'b [U]) -> Option<U>",
        )
        .expect("AST graft should succeed");

        assert!(adapted.contains("pub fn head"));
        assert!(adapted.contains("U : Clone + Default"));
        assert!(adapted.contains("values . first"));
        syn::parse_str::<ItemFn>(&adapted).expect("adapted source should remain a Rust function");
    }

    #[test]
    fn test_adapt_solution_rejects_shadowed_parameter_rename() {
        let source = r#"pub fn donor(input: &str) -> usize {
    let input = input.trim();
    input.len()
}"#;

        let adapted = adapt_solution(
            source,
            "pub fn donor(input: &str) -> usize",
            "target",
            "pub fn target(value: &str) -> usize",
        );

        assert!(
            adapted.is_none(),
            "shadowed parameter grafts should be rejected rather than silently rewritten"
        );
    }
}
