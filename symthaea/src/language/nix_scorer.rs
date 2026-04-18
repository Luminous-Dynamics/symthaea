// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Structural scorer for NixEval benchmark (P1 of nix-scoring plan).
//!
//! The legacy benchmark in `examples/nix_eval_benchmark.rs` uses substring
//! containment to judge generated Nix against a curated problem set. That
//! scheme admits false positives — e.g. `services.postgresql.enable = false;
//! # pgvector needed` passes when the required substrings are
//! `postgresql`, `enable`, and `pgvector`.
//!
//! This module walks both the generated code and a **golden reference**
//! through the `rnix` parser, extracts `(attrpath, canonical_value)` pairs,
//! and compares them structurally. Comments cannot satisfy a path because
//! they never produce a `NODE_KEY_VALUE` AST node.
//!
//! Only compiled when `code_generation` feature is on (matches corpus/benchmark).

use rnix::{NodeOrToken, Root, SyntaxKind, SyntaxNode};
use std::collections::{BTreeMap, BTreeSet};

/// Canonicalized value for an option assignment. The scorer is deliberately
/// conservative — unknown shapes fall through to `Opaque(raw)` and compare
/// by trimmed source text. Refinements (full list equality modulo order,
/// `with pkgs;` unwrapping, etc.) can land incrementally without breaking
/// callers since the verdict only reports mismatches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonValue {
    Bool(bool),
    Int(i64),
    /// String literal with quotes stripped.
    Str(String),
    /// Raw source of the RHS, trimmed and whitespace-collapsed.
    /// Used for anything we don't canonicalize further yet.
    Opaque(String),
}

impl CanonValue {
    /// Human-friendly rendering for mismatch reports.
    pub fn display(&self) -> String {
        match self {
            CanonValue::Bool(b) => b.to_string(),
            CanonValue::Int(i) => i.to_string(),
            CanonValue::Str(s) => format!("\"{}\"", s),
            CanonValue::Opaque(s) => s.clone(),
        }
    }
}

/// A single flattened `attrpath = value` assignment from a Nix expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatOption {
    /// Dotted path, e.g. `["services", "postgresql", "enable"]`.
    pub path: Vec<String>,
    pub value: CanonValue,
}

impl FlatOption {
    fn path_key(&self) -> String {
        self.path.join(".")
    }
}

/// Result of comparing generated Nix against a golden reference.
#[derive(Debug, Clone, Default)]
pub struct StructuralVerdict {
    /// Jaccard overlap of attrpath sets: |A ∩ B| / |A ∪ B|.
    pub path_jaccard: f32,
    /// Paths present in both sides but with differing canonical values.
    pub value_mismatches: Vec<ValueMismatch>,
    /// Paths in the golden that are absent from the generated code.
    pub missing_required: Vec<String>,
    /// Paths in the generated code that are absent from the golden.
    /// Treated as *warning* only — safe extras (firewall ports, comments)
    /// should not penalize a generation.
    pub extraneous: Vec<String>,
    /// A parse error in either side; if present, the verdict is
    /// `pass = false` regardless of the other fields.
    pub parse_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueMismatch {
    pub path: String,
    pub got: CanonValue,
    pub want: CanonValue,
}

impl StructuralVerdict {
    /// Pass criteria: no parse error, every required (golden) path is
    /// present, and no value mismatches on the intersection.
    ///
    /// Notably `path_jaccard` is **not** a pass gate — it would penalize
    /// generations that add safe extras (e.g. firewall ports, default
    /// settings). The plan explicitly says extraneous is warning-only, so
    /// the gate is gold-coverage (`missing_required.is_empty()`) instead.
    /// `path_jaccard` stays on the verdict for telemetry/reporting.
    pub fn pass(&self) -> bool {
        self.parse_error.is_none()
            && self.value_mismatches.is_empty()
            && self.missing_required.is_empty()
    }

    /// Short one-line summary for benchmark tables.
    pub fn summary(&self) -> String {
        if let Some(err) = &self.parse_error {
            return format!("PARSE-ERR: {}", err);
        }
        if self.pass() {
            "PASS".to_string()
        } else {
            format!(
                "FAIL: jaccard={:.2} mismatches={} missing={}",
                self.path_jaccard,
                self.value_mismatches.len(),
                self.missing_required.len()
            )
        }
    }
}

/// Walk a parsed Nix tree and collect every `(attrpath, value)` leaf.
///
/// Flattens nested attrset forms: `services.nginx = { enable = true; };`
/// produces `services.nginx.enable = true` — matching the flat form
/// `services.nginx.enable = true;`. This is essential because NixOS
/// module generators use both forms interchangeably.
///
/// Only **static** attrpaths are collected — dynamic keys like `${foo}`
/// or `"literal str"` as a key are skipped. A real Nix evaluator would
/// need them; a scorer for curated option configs does not.
fn walk_attrpaths(root: &SyntaxNode) -> Vec<FlatOption> {
    let mut out = Vec::new();
    walk_with_prefix(root, &[], &mut out);
    out
}

/// Recursive walker that accumulates the enclosing attrpath prefix.
/// Visits `NODE_ATTRPATH_VALUE` leaves and recurses through structural
/// nodes (attrset literals, let-bindings, lambdas, function applications,
/// parens, with-scopes) without consuming path context.
fn walk_with_prefix(node: &SyntaxNode, prefix: &[String], out: &mut Vec<FlatOption>) {
    for child in node.children() {
        match child.kind() {
            SyntaxKind::NODE_ATTRPATH_VALUE => {
                let mut sub_children = child.children();
                let Some(key_node) = sub_children.next() else {
                    continue;
                };
                if key_node.kind() != SyntaxKind::NODE_ATTRPATH {
                    continue;
                }
                let Some(value_node) = sub_children.next() else {
                    continue;
                };
                let Some(segs) = extract_static_attrpath(&key_node) else {
                    continue;
                };

                let mut full_path: Vec<String> = prefix.to_vec();
                full_path.extend(segs);

                if value_node.kind() == SyntaxKind::NODE_ATTR_SET {
                    // Nested attrset: recurse with the extended prefix.
                    // Don't emit a leaf for the container itself — the
                    // leaves inside carry the semantic content.
                    walk_with_prefix(&value_node, &full_path, out);
                } else {
                    out.push(FlatOption {
                        path: full_path,
                        value: canonicalize_value(&value_node),
                    });
                }
            }
            // Structural nodes: recurse but keep prefix as-is. These
            // wrap the module body without adding attrpath context.
            SyntaxKind::NODE_ATTR_SET
            | SyntaxKind::NODE_LET_IN
            | SyntaxKind::NODE_LAMBDA
            | SyntaxKind::NODE_APPLY
            | SyntaxKind::NODE_PAREN
            | SyntaxKind::NODE_WITH => {
                walk_with_prefix(&child, prefix, out);
            }
            _ => {
                // Leaf-ish nodes (literals, strings, selects, etc.) —
                // don't descend.
            }
        }
    }
}

/// NODE_ATTRPATH children are a sequence of NODE_IDENTs (for `a.b.c = x`)
/// separated by `.` tokens, optionally interleaved with NODE_DYNAMIC or
/// NODE_STRING segments. Returns `None` if any segment is dynamic/quoted —
/// the scorer only handles static attrpaths, which is all the corpus uses.
fn extract_static_attrpath(key_node: &SyntaxNode) -> Option<Vec<String>> {
    let mut segs = Vec::new();
    for child in key_node.children_with_tokens() {
        match child {
            NodeOrToken::Node(n) => match n.kind() {
                SyntaxKind::NODE_IDENT => {
                    segs.push(n.text().to_string());
                }
                SyntaxKind::NODE_DYNAMIC | SyntaxKind::NODE_STRING => {
                    // Dynamic or quoted segment: bail.
                    return None;
                }
                _ => {
                    // Unexpected node kind inside NODE_ATTRPATH — skip it
                    // rather than fail, so exotic constructs don't
                    // crash the scorer.
                }
            },
            NodeOrToken::Token(_) => {
                // `.` separators and whitespace — ignored.
            }
        }
    }
    if segs.is_empty() {
        None
    } else {
        Some(segs)
    }
}

/// Canonicalize a value node to one of the supported CanonValue variants.
/// Unrecognized shapes fall through to Opaque with whitespace-collapsed source.
fn canonicalize_value(node: &SyntaxNode) -> CanonValue {
    let raw = node.text().to_string();
    let trimmed = raw.trim();

    // Literal bool
    if trimmed == "true" {
        return CanonValue::Bool(true);
    }
    if trimmed == "false" {
        return CanonValue::Bool(false);
    }

    // Literal int (no decimal point, optional leading sign, pure digits)
    if let Ok(i) = trimmed.parse::<i64>() {
        return CanonValue::Int(i);
    }

    // Literal quoted string: `"..."` (no interpolation).
    // rnix would give us NODE_STRING; we just strip quotes here.
    if trimmed.starts_with('"')
        && trimmed.ends_with('"')
        && trimmed.len() >= 2
        && !trimmed[1..trimmed.len() - 1].contains("${")
    {
        let inner = &trimmed[1..trimmed.len() - 1];
        // No escape processing — if two goldens differ in `\n` vs literal
        // newline that's a real mismatch worth surfacing.
        return CanonValue::Str(inner.to_string());
    }

    // Fallback: canonicalize formatting. First pad structural punctuation
    // so `[a b]` and `[ a b ]` tokenize identically, then collapse runs
    // of whitespace. Crude but catches the common cosmetic differences
    // between hand-written goldens and generator output.
    let padded = trimmed
        .replace('[', " [ ")
        .replace(']', " ] ")
        .replace('{', " { ")
        .replace('}', " } ")
        .replace(';', " ; ")
        .replace(',', " , ");
    let collapsed: String = padded.split_whitespace().collect::<Vec<_>>().join(" ");
    CanonValue::Opaque(collapsed)
}

/// Strip line-comments (`#…` to EOL) before parsing. Block comments
/// (`/* … */`) are handled by rnix itself. We do this so that a scorer
/// *pre-parse* pass cannot be satisfied by a commented-out assignment,
/// but more importantly so that doc-string comments embedded in option
/// values don't get compared as part of `CanonValue::Opaque`.
fn strip_line_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    for line in src.lines() {
        // Crude pass: `#` inside a string is rare in NixOS configs.
        // If it breaks a real problem we'll swap to a state-machine
        // aware of quote context. For the 95 curated problems this
        // heuristic is known-safe.
        if let Some(idx) = line.find('#') {
            out.push_str(&line[..idx]);
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    out
}

/// Top-level scorer: parse both snippets, walk, compare.
pub fn score(generated: &str, golden: &str) -> StructuralVerdict {
    let mut verdict = StructuralVerdict::default();

    let gen_src = strip_line_comments(generated);
    let gold_src = strip_line_comments(golden);

    let gen_parse = Root::parse(&gen_src);
    if !gen_parse.errors().is_empty() {
        verdict.parse_error = Some(format!("generated: {}", gen_parse.errors()[0]));
        return verdict;
    }
    let gold_parse = Root::parse(&gold_src);
    if !gold_parse.errors().is_empty() {
        verdict.parse_error = Some(format!("golden: {}", gold_parse.errors()[0]));
        return verdict;
    }

    let gen_opts = walk_attrpaths(&gen_parse.syntax());
    let gold_opts = walk_attrpaths(&gold_parse.syntax());

    // Build path→value maps for quick lookup. Duplicate paths (possible in
    // nested `{ a.b = 1; a.b = 2; }` — a Nix error but let's be robust)
    // keep the last seen value, matching Nix eval semantics.
    let gen_map: BTreeMap<String, CanonValue> = gen_opts
        .iter()
        .map(|o| (o.path_key(), o.value.clone()))
        .collect();
    let gold_map: BTreeMap<String, CanonValue> = gold_opts
        .iter()
        .map(|o| (o.path_key(), o.value.clone()))
        .collect();

    let gen_paths: BTreeSet<&String> = gen_map.keys().collect();
    let gold_paths: BTreeSet<&String> = gold_map.keys().collect();

    let intersection: BTreeSet<&&String> = gen_paths.intersection(&gold_paths).collect();
    let union: BTreeSet<&&String> = gen_paths.union(&gold_paths).collect();

    verdict.path_jaccard = if union.is_empty() {
        1.0
    } else {
        intersection.len() as f32 / union.len() as f32
    };

    for path in &intersection {
        let key: &String = **path;
        let got = &gen_map[key];
        let want = &gold_map[key];
        if got != want {
            verdict.value_mismatches.push(ValueMismatch {
                path: key.clone(),
                got: got.clone(),
                want: want.clone(),
            });
        }
    }
    for path in gold_paths.difference(&gen_paths) {
        verdict.missing_required.push((*path).clone());
    }
    for path in gen_paths.difference(&gold_paths) {
        verdict.extraneous.push((*path).clone());
    }

    verdict
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = "{ config, pkgs, ... }: ";

    fn wrap(body: &str) -> String {
        format!("{}{{\n{}\n}}", HEADER, body)
    }

    #[test]
    fn identical_inputs_pass() {
        let nix = wrap("services.nginx.enable = true;");
        let v = score(&nix, &nix);
        assert!(v.pass(), "identical configs must pass; got {:?}", v);
        assert!((v.path_jaccard - 1.0).abs() < 1e-6);
    }

    #[test]
    fn enable_false_does_not_satisfy_enable_true_golden() {
        // The exact false-positive case from the plan: substring scorer
        // would pass this because "services.postgresql", "enable = true"
        // (once the comment is there) and "pgvector" all appear.
        // Structural scorer must catch the value mismatch on `enable`.
        let generated = wrap(
            "services.postgresql.enable = false;\n\
             services.postgresql.extensions = \"pgvector\";",
        );
        let golden = wrap(
            "services.postgresql.enable = true;\n\
             services.postgresql.extensions = \"pgvector\";",
        );
        let v = score(&generated, &golden);
        assert!(!v.pass(), "enable=false vs enable=true must not pass");
        assert_eq!(v.value_mismatches.len(), 1);
        assert_eq!(v.value_mismatches[0].path, "services.postgresql.enable");
        assert_eq!(v.value_mismatches[0].got, CanonValue::Bool(false));
        assert_eq!(v.value_mismatches[0].want, CanonValue::Bool(true));
    }

    #[test]
    fn comment_cannot_satisfy_option() {
        // A comment mentioning `services.postgresql.enable = true` must not
        // register as an assignment. Only a real AST node does.
        let generated = wrap("# services.postgresql.enable = true;");
        let golden = wrap("services.postgresql.enable = true;");
        let v = score(&generated, &golden);
        assert!(!v.pass(), "commented-out option must not satisfy golden");
        assert!(v
            .missing_required
            .contains(&"services.postgresql.enable".to_string()));
    }

    #[test]
    fn reordered_attrsets_pass() {
        let gen = wrap(
            "services.nginx.enable = true;\n\
             networking.firewall.enable = true;",
        );
        let gold = wrap(
            "networking.firewall.enable = true;\n\
             services.nginx.enable = true;",
        );
        let v = score(&gen, &gold);
        assert!(v.pass(), "reordering should not fail; got {:?}", v);
    }

    #[test]
    fn missing_required_fails() {
        let gen = wrap("services.nginx.enable = true;");
        let gold = wrap(
            "services.nginx.enable = true;\n\
             services.postgresql.enable = true;",
        );
        let v = score(&gen, &gold);
        assert!(!v.pass());
        assert!(v
            .missing_required
            .contains(&"services.postgresql.enable".to_string()));
    }

    #[test]
    fn extraneous_is_warning_not_fail() {
        // Generated adds firewall ports that golden doesn't specify.
        // This must still pass (warning only).
        let gen = wrap(
            "services.nginx.enable = true;\n\
             networking.firewall.allowedTCPPorts = [ 80 443 ];",
        );
        let gold = wrap("services.nginx.enable = true;");
        let v = score(&gen, &gold);
        assert!(v.pass(), "extraneous should not fail; got {:?}", v);
        assert!(!v.extraneous.is_empty(), "must record extraneous paths");
    }

    #[test]
    fn value_type_mismatch_fails() {
        // Golden uses int, generated uses string — real mistake.
        let gen = wrap("services.postgresql.port = \"5432\";");
        let gold = wrap("services.postgresql.port = 5432;");
        let v = score(&gen, &gold);
        assert!(!v.pass());
        assert_eq!(v.value_mismatches.len(), 1);
    }

    #[test]
    fn parse_error_is_reported() {
        let gen = wrap("services.nginx.enable =;"); // syntax error
        let gold = wrap("services.nginx.enable = true;");
        let v = score(&gen, &gold);
        assert!(!v.pass());
        assert!(v.parse_error.is_some());
    }

    #[test]
    fn nested_attrset_equals_flat_dotted_form() {
        // `services.nginx = { enable = true; }` must be structurally
        // equivalent to `services.nginx.enable = true`. Both are valid
        // NixOS module idioms; a hand-written golden in one form must
        // not reject a generator that uses the other.
        let flat = wrap("services.nginx.enable = true;");
        let nested = wrap("services.nginx = {\n    enable = true;\n  };");
        let v = score(&nested, &flat);
        assert!(v.pass(), "nested form must match flat form; got {:?}", v);
    }

    #[test]
    fn deep_nesting_flattens_correctly() {
        // Triple-nested: `a = { b = { c = 1; }; }` → a.b.c = 1
        let nested = wrap("a = {\n    b = {\n      c = 1;\n    };\n  };");
        let flat = wrap("a.b.c = 1;");
        let v = score(&nested, &flat);
        assert!(v.pass(), "deep nesting should flatten; got {:?}", v);
    }

    #[test]
    fn whitespace_collapse_in_opaque_values() {
        // `[ 80  443 ]` and `[80 443]` should canonicalize identically.
        let gen = wrap("networking.firewall.allowedTCPPorts = [ 80  443 ];");
        let gold = wrap("networking.firewall.allowedTCPPorts = [80 443];");
        let v = score(&gen, &gold);
        assert!(
            v.pass(),
            "whitespace differences should not fail; got {:?}",
            v
        );
    }
}
