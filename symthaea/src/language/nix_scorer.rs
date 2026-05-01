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
    /// A list of package identifiers. Captured from `with pkgs; [ a b c ]`
    /// and bare `[ a b c ]` forms. BTreeSet so order is canonical.
    ///
    /// **Satisfaction semantics**: generated list satisfies golden iff
    /// `golden ⊆ generated` — extras are accepted (more packages are
    /// strictly more useful in a dev shell). This matches the "extraneous
    /// is warning, not fail" principle applied at path-set level.
    PackageList(std::collections::BTreeSet<String>),
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
            CanonValue::PackageList(pkgs) => {
                let joined: Vec<&str> = pkgs.iter().map(|s| s.as_str()).collect();
                format!("with pkgs; [ {} ]", joined.join(" "))
            }
            CanonValue::Opaque(s) => s.clone(),
        }
    }

    /// Directional comparison: does `self` (the generated value) satisfy
    /// the `want` (golden) value?
    ///
    /// For Bool/Int/Str/Opaque this is structural equality. For
    /// PackageList it's "superset OK": a generated list satisfies a
    /// golden list iff every package in the golden is present in the
    /// generated list (extras are fine). Mixed-type compares are never
    /// satisfied — an int value can't meet a string expectation, etc.
    pub fn satisfies(&self, want: &CanonValue) -> bool {
        match (self, want) {
            (CanonValue::PackageList(got), CanonValue::PackageList(wanted)) => {
                wanted.is_subset(got)
            }
            _ => self == want,
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
    /// Set when the golden parses but flattens to zero static attrpaths —
    /// happens when the golden uses only dynamic keys like
    /// `services.redis.servers."".enable = true;`. Without static paths
    /// there's nothing to require of the generation, so every parse-valid
    /// generation would trivially "pass" (empty missing_required set).
    /// That's a scorer failure, not a real pass — `pass()` returns false.
    pub golden_unscorable: bool,
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
            && !self.golden_unscorable
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

/// Public helper: return the set of attrpaths present in a Nix
/// snippet. Used by the self-repair loop (no-golden mode) to compare
/// against intent-derived expected paths. Parse failures return an
/// empty set — callers should treat that as "trust the generator."
pub fn attrpath_set_of(code: &str) -> std::collections::BTreeSet<String> {
    let stripped = strip_line_comments(code);
    let parsed = Root::parse(&stripped);
    if !parsed.errors().is_empty() {
        return std::collections::BTreeSet::new();
    }
    walk_attrpaths(&parsed.syntax())
        .into_iter()
        .map(|opt| opt.path.join("."))
        .collect()
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
                SyntaxKind::NODE_STRING => {
                    let text = n.text().to_string();
                    if text == "\"\"" {
                        segs.push("".to_string());
                    } else {
                        // Other quoted segments: bail.
                        return None;
                    }
                }
                SyntaxKind::NODE_DYNAMIC => {
                    // Dynamic segment: bail.
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

    // PackageList: `with pkgs; [ a b c ]` or plain `[ a b c ]` with only
    // identifiers inside. Checked before Opaque so subset semantics
    // apply where they naturally should (dev-shell buildInputs).
    if let Some(pkgs) = try_parse_package_list(trimmed) {
        return CanonValue::PackageList(pkgs);
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

/// Recognize `with pkgs; [ ident ident ]` and plain `[ ident ident ]`
/// forms. Returns `None` if the list contains anything that isn't a
/// plain identifier (falls through to Opaque).
fn try_parse_package_list(src: &str) -> Option<std::collections::BTreeSet<String>> {
    let body = if let Some(rest) = src.strip_prefix("with ") {
        // `with <scope>;` — find the `;` and skip past it.
        let semi = rest.find(';')?;
        rest[semi + 1..].trim_start()
    } else {
        src
    };
    let body = body.strip_prefix('[')?.strip_suffix(']')?;

    let mut pkgs = std::collections::BTreeSet::new();
    for token in body.split_whitespace() {
        if !is_package_identifier(token) {
            return None;
        }
        pkgs.insert(token.to_string());
    }
    if pkgs.is_empty() {
        None
    } else {
        Some(pkgs)
    }
}

/// Identifier shape: `[A-Za-z_][\w.-]*`. Rejects anything that could be
/// an expression (quoted strings, interpolations, calls, etc.) so the
/// subset-check can't accidentally compare semantically-different values.
fn is_package_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-')
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

    // Guard: if the golden has zero static attrpaths we can't evaluate
    // the generation against it — any parse-valid output would trivially
    // satisfy an empty `missing_required`. Mark the verdict unscorable
    // and return early so `pass()` fails closed. This catches goldens
    // using dynamic keys like `services.redis.servers."".enable`.
    if gold_opts.is_empty() {
        verdict.golden_unscorable = true;
        return verdict;
    }

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
        // `satisfies` is directional: for package lists, extras on the
        // generated side are OK (subset semantics). For scalars it's
        // still structural equality.
        if !got.satisfies(want) {
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
    fn package_list_superset_satisfies_golden() {
        // A dev-shell generator might emit a more complete package set
        // than the hand-written golden. As long as every required pkg
        // is present, extras (rustfmt, clippy) should NOT fail the check.
        let gen =
            wrap("buildInputs = with pkgs; [ rustc cargo rustfmt clippy rust-analyzer mold ];");
        let gold = wrap("buildInputs = with pkgs; [ rustc cargo rust-analyzer mold ];");
        let v = score(&gen, &gold);
        assert!(
            v.pass(),
            "generator superset must satisfy golden; got {:?}",
            v
        );
    }

    #[test]
    fn package_list_missing_pkg_fails() {
        // Conversely: if a required package is absent, the structural
        // scorer must report a mismatch — this is the legitimate
        // generator-bug case that matters to catch.
        let gen = wrap("buildInputs = with pkgs; [ rustc cargo ];");
        let gold = wrap("buildInputs = with pkgs; [ rustc cargo rust-analyzer ];");
        let v = score(&gen, &gold);
        assert!(!v.pass(), "missing required package must fail");
        assert_eq!(v.value_mismatches.len(), 1);
    }

    #[test]
    fn package_list_order_independent() {
        // BTreeSet canonicalization means [ b a c ] == [ a b c ].
        let gen = wrap("buildInputs = with pkgs; [ cargo rustc rust-analyzer ];");
        let gold = wrap("buildInputs = with pkgs; [ rustc cargo rust-analyzer ];");
        let v = score(&gen, &gold);
        assert!(v.pass(), "list order should not matter; got {:?}", v);
    }

    #[test]
    fn non_identifier_list_falls_through_to_opaque() {
        // Lists containing non-identifier expressions (strings, numbers)
        // must NOT become PackageList — subset semantics don't make sense
        // for config values.
        let gen = wrap("networking.firewall.allowedTCPPorts = [ 80 443 ];");
        let gold = wrap("networking.firewall.allowedTCPPorts = [ 80 443 ];");
        let v = score(&gen, &gold);
        assert!(v.pass(), "integer list should still match; got {:?}", v);
        // Confirm it canonicalized as Opaque (not PackageList) by adding
        // an extra port that SHOULD fail — if it became PackageList, the
        // subset rule would let this through.
        let gen_extra = wrap("networking.firewall.allowedTCPPorts = [ 80 443 8080 ];");
        let v2 = score(&gen_extra, &gold);
        assert!(
            !v2.pass(),
            "extra integer in opaque-int-list must fail (only package lists are subset-tolerant); got {:?}",
            v2
        );
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

    #[test]
    fn golden_with_only_dynamic_attrpaths_fails_closed() {
        // Regression: the redis golden uses `services.redis.servers."".enable`
        // where `""` is a dynamic (quoted) segment. `walk_attrpaths` correctly
        // skips dynamic paths, which left the golden's flat-options empty —
        // and that let any parse-valid gibberish trivially satisfy the empty
        // `missing_required` set, producing a false PASS.
        //
        // This test locks in the fail-closed behavior: when the golden
        // flattens to zero static paths, `pass()` must return false even if
        // the generated output parses cleanly.
        let golden = "{ services.redis.servers.\"\".enable = true; }";
        let generated = "{ unrelated.thing.x = true; }";
        let v = score(generated, golden);
        assert!(
            !v.pass(),
            "golden with only dynamic paths should NOT silently pass; got {:?}",
            v
        );
        assert!(
            v.golden_unscorable,
            "verdict should flag golden_unscorable; got {:?}",
            v
        );
    }
}
