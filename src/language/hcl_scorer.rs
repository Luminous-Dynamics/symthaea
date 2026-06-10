// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Structural scorer for Terraform HCL (Phase 3 of the coding-AI
//! roadmap — second substrate-independence port after Docker
//! Compose).
//!
//! Mirrors `nix_scorer.rs` and `compose_scorer.rs`: parse generated +
//! golden, flatten to dotted paths, compare.
//!
//! Built on `hcl-rs` 0.18 — a battle-tested HCL parser that handles
//! everything the spec supports (blocks, nested blocks, labels,
//! attributes, heredocs, templates, conditionals, for-expressions,
//! traversals). We only canonicalize scalars directly; anything with
//! interpolation/computation falls through to `Opaque(source-style
//! debug-print)`. That's the conservative choice — two HCL snippets
//! with differently-structured-but-semantically-identical for-expressions
//! would appear mismatched. Real Terraform diff/verify would use the
//! post-evaluation Value, not the Expression AST; our use case is
//! "generated code matches hand-written golden," so sticking to
//! Expression equality is correct.
//!
//! Path layout (Terraform canonical):
//! - `resource "aws_s3_bucket" "example" { bucket = "x" }` flattens
//!   to `resource.aws_s3_bucket.example.bucket`.
//! - Nested blocks get the same treatment: `resource.X.Y.tags`
//!   becomes the parent prefix, its attributes extend it.
//! - Top-level attributes (rare in modules, common in vars files)
//!   land at their key.

use hcl::{Block, BlockLabel, Body, Expression, Structure};
use std::collections::BTreeMap;

/// Canonicalized HCL value. Scalars get their own variants;
/// anything with computation/template/reference semantics falls to
/// `Opaque(display-string)`. Structural scorer needs equality, not
/// evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum HclValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    /// Quoted string with outer `"` stripped.
    Str(String),
    /// Bare identifier (variable reference, `null`).
    Ident(String),
    /// Anything not directly canonicalized — lists, objects,
    /// template exprs, traversals, function calls. Comparison by
    /// trimmed display-string.
    Opaque(String),
}

impl HclValue {
    pub fn display(&self) -> String {
        match self {
            HclValue::Bool(b) => b.to_string(),
            HclValue::Int(i) => i.to_string(),
            HclValue::Float(f) => f.to_string(),
            HclValue::Str(s) => format!("\"{}\"", s),
            HclValue::Ident(i) => i.clone(),
            HclValue::Opaque(s) => s.clone(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct HclVerdict {
    pub path_jaccard: f32,
    pub value_mismatches: Vec<HclMismatch>,
    pub missing_required: Vec<String>,
    pub extraneous: Vec<String>,
    pub parse_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HclMismatch {
    pub path: String,
    pub got: HclValue,
    pub want: HclValue,
}

impl HclVerdict {
    pub fn pass(&self) -> bool {
        self.parse_error.is_none()
            && self.value_mismatches.is_empty()
            && self.missing_required.is_empty()
    }

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

/// Public entry: parse both sides, walk, compare.
pub fn score(generated: &str, golden: &str) -> HclVerdict {
    let mut verdict = HclVerdict::default();
    let gen_map = match flatten(generated) {
        Ok(m) => m,
        Err(e) => {
            verdict.parse_error = Some(format!("generated: {}", e));
            return verdict;
        }
    };
    let gold_map = match flatten(golden) {
        Ok(m) => m,
        Err(e) => {
            verdict.parse_error = Some(format!("golden: {}", e));
            return verdict;
        }
    };

    let gen_paths: std::collections::BTreeSet<&String> = gen_map.keys().collect();
    let gold_paths: std::collections::BTreeSet<&String> = gold_map.keys().collect();
    let intersection: std::collections::BTreeSet<&&String> =
        gen_paths.intersection(&gold_paths).collect();
    let union: std::collections::BTreeSet<&&String> = gen_paths.union(&gold_paths).collect();
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
            verdict.value_mismatches.push(HclMismatch {
                path: key.clone(),
                got: got.clone(),
                want: want.clone(),
            });
        }
    }
    for p in gold_paths.difference(&gen_paths) {
        verdict.missing_required.push((*p).clone());
    }
    for p in gen_paths.difference(&gold_paths) {
        verdict.extraneous.push((*p).clone());
    }
    verdict
}

// ─── hcl-rs → path-map flattener ────────────────────────────────────────

fn flatten(src: &str) -> Result<BTreeMap<String, HclValue>, String> {
    let body: Body = hcl::from_str(src).map_err(|e| e.to_string())?;
    let mut out = BTreeMap::new();
    walk_body(&body, "", &mut out);
    Ok(out)
}

fn walk_body(body: &Body, prefix: &str, out: &mut BTreeMap<String, HclValue>) {
    for structure in body.iter() {
        match structure {
            Structure::Attribute(attr) => {
                let path = if prefix.is_empty() {
                    attr.key.to_string()
                } else {
                    format!("{}.{}", prefix, attr.key)
                };
                out.insert(path, expr_to_value(&attr.expr));
            }
            Structure::Block(block) => {
                walk_block(block, prefix, out);
            }
        }
    }
}

fn walk_block(block: &Block, prefix: &str, out: &mut BTreeMap<String, HclValue>) {
    let mut segs = vec![block.identifier.to_string()];
    for label in &block.labels {
        let l = match label {
            BlockLabel::Identifier(i) => i.to_string(),
            BlockLabel::String(s) => s.clone(),
        };
        segs.push(l);
    }
    let block_path = if prefix.is_empty() {
        segs.join(".")
    } else {
        format!("{}.{}", prefix, segs.join("."))
    };
    walk_body(&block.body, &block_path, out);
}

/// Convert an `hcl::Expression` to a `HclValue`. Scalars get typed
/// variants; compound expressions go to Opaque with a debug-printed
/// body so two differently-written-but-identical Opaques compare
/// equal.
fn expr_to_value(e: &Expression) -> HclValue {
    match e {
        Expression::Null => HclValue::Ident("null".into()),
        Expression::Bool(b) => HclValue::Bool(*b),
        Expression::Number(n) => {
            if let Some(i) = n.as_i64() {
                HclValue::Int(i)
            } else if let Some(f) = n.as_f64() {
                HclValue::Float(f)
            } else {
                HclValue::Opaque(n.to_string())
            }
        }
        Expression::String(s) => HclValue::Str(s.clone()),
        Expression::Array(items) => {
            let parts: Vec<String> = items.iter().map(|e| expr_to_value(e).display()).collect();
            HclValue::Opaque(format!("[{}]", parts.join(", ")))
        }
        _ => {
            // hcl::Expression is `#[non_exhaustive]` — catch everything
            // else (Object, TemplateExpr, Variable, Traversal,
            // FuncCall, Parenthesis, Conditional, Operation, ForExpr,
            // plus any future additions). Collapse to display form;
            // equality is source-shape. For Variables and Traversals
            // the Display impl yields the canonical `var.x` /
            // `module.y.z` form so two identical references compare
            // equal.
            HclValue::Opaque(format!("{}", e))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_hcl_passes() {
        let hcl = r#"
resource "aws_s3_bucket" "example" {
  bucket = "my-bucket"
  acl    = "private"
}
"#;
        let v = score(hcl, hcl);
        assert!(v.pass(), "identical should pass; got {:?}", v);
    }

    #[test]
    fn bucket_name_mismatch_fails() {
        let generated = r#"
resource "aws_s3_bucket" "example" {
  bucket = "wrong-bucket"
  acl    = "private"
}
"#;
        let gold = r#"
resource "aws_s3_bucket" "example" {
  bucket = "right-bucket"
  acl    = "private"
}
"#;
        let v = score(generated, gold);
        assert!(!v.pass());
        assert_eq!(v.value_mismatches.len(), 1);
        assert_eq!(
            v.value_mismatches[0].path,
            "resource.aws_s3_bucket.example.bucket"
        );
    }

    #[test]
    fn missing_resource_fails() {
        let generated = r#"
resource "aws_s3_bucket" "a" {
  bucket = "only-one"
}
"#;
        let gold = r#"
resource "aws_s3_bucket" "a" {
  bucket = "only-one"
}
resource "aws_instance" "server" {
  ami = "ami-12345"
}
"#;
        let v = score(generated, gold);
        assert!(!v.pass());
        assert!(
            v.missing_required
                .iter()
                .any(|p| p.contains("aws_instance"))
        );
    }

    #[test]
    fn extra_resource_is_warning_only() {
        let generated = r#"
resource "aws_s3_bucket" "a" { bucket = "one" }
resource "aws_s3_bucket" "b" { bucket = "two" }
"#;
        let gold = r#"
resource "aws_s3_bucket" "a" { bucket = "one" }
"#;
        let v = score(generated, gold);
        assert!(v.pass(), "extra resource is warning; got {:?}", v);
        assert!(!v.extraneous.is_empty());
    }

    #[test]
    fn bool_vs_string_fails() {
        let generated = r#"
resource "aws_s3_bucket" "example" {
  versioning = "true"
}
"#;
        let gold = r#"
resource "aws_s3_bucket" "example" {
  versioning = true
}
"#;
        let v = score(generated, gold);
        assert!(!v.pass(), "string vs bool must be flagged");
    }

    #[test]
    fn line_comments_do_not_create_paths() {
        // HCL supports `#` and `//` comments — hcl-rs parser drops
        // them, so commented-out blocks don't contribute paths.
        let generated = r#"
# resource "aws_s3_bucket" "fake" {
#   bucket = "fake"
# }
resource "aws_s3_bucket" "real" {
  bucket = "real"
}
"#;
        let gold = r#"
resource "aws_s3_bucket" "real" {
  bucket = "real"
}
"#;
        let v = score(generated, gold);
        assert!(
            v.pass(),
            "comment-only differences must not fail; got {:?}",
            v
        );
    }

    #[test]
    fn nested_blocks_flatten() {
        let hcl = r#"
resource "aws_instance" "web" {
  ami = "ami-12345"
  tags {
    Environment = "production"
    Name        = "web-server"
  }
}
"#;
        let v = score(hcl, hcl);
        assert!(v.pass(), "identical nested blocks should pass; got {:?}", v);

        // Removing a nested attribute makes it appear as extraneous on
        // the generated side — warning, not fail.
        let gold_minus_tag = r#"
resource "aws_instance" "web" {
  ami = "ami-12345"
  tags {
    Environment = "production"
  }
}
"#;
        let v2 = score(hcl, gold_minus_tag);
        assert!(v2.pass(), "extra tag is warning-only; got {:?}", v2);
        assert!(v2.extraneous.iter().any(|p| p.contains("Name")));
    }

    #[test]
    fn int_port_values_match() {
        let hcl = r#"
resource "aws_security_group_rule" "http" {
  from_port = 80
  to_port   = 80
  protocol  = "tcp"
}
"#;
        let v = score(hcl, hcl);
        assert!(v.pass());
    }

    #[test]
    fn unterminated_string_parse_errors() {
        let generated = r#"
resource "aws_s3_bucket" "broken" {
  bucket = "no-close
}
"#;
        let gold = r#"
resource "aws_s3_bucket" "broken" {
  bucket = "fine"
}
"#;
        let v = score(generated, gold);
        assert!(!v.pass());
        assert!(v.parse_error.is_some());
    }

    // ── Tests that ONLY work with a real HCL parser ────────────────
    // These prove the hcl-rs choice bought us real coverage:

    #[test]
    fn heredoc_strings_parse_and_compare() {
        // Heredocs are common in Terraform for multi-line policies,
        // user_data scripts. Hand-rolled parsers almost always
        // tokenize them incorrectly. hcl-rs handles them natively.
        let hcl = r#"
resource "aws_iam_policy" "example" {
  policy = <<EOT
{
  "Version": "2012-10-17",
  "Statement": []
}
EOT
}
"#;
        let v = score(hcl, hcl);
        assert!(v.pass(), "heredoc identity should pass; got {:?}", v);
    }

    #[test]
    fn variable_references_compare_via_opaque() {
        // `var.environment` is a Traversal expression. Two identical
        // traversals should compare equal; different ones should not.
        let identical_gen = r#"
resource "aws_instance" "web" {
  ami = var.ami_id
}
"#;
        let v = score(identical_gen, identical_gen);
        assert!(v.pass());

        let diff = r#"
resource "aws_instance" "web" {
  ami = var.different_id
}
"#;
        let v2 = score(identical_gen, diff);
        assert!(!v2.pass(), "different traversals must fail");
    }

    #[test]
    fn for_expressions_compare_structurally() {
        // Real Terraform feature: `[for x in var.list : x.id]`
        // Hand-rolled parser would not handle this at all.
        let hcl = r#"
locals {
  ids = [for inst in var.instances : inst.id]
}
"#;
        let v = score(hcl, hcl);
        assert!(v.pass(), "for-expr identity; got {:?}", v);
    }

    #[test]
    fn conditional_expressions_compare() {
        let hcl = r#"
resource "aws_instance" "web" {
  instance_type = var.large ? "t3.large" : "t3.small"
}
"#;
        let v = score(hcl, hcl);
        assert!(v.pass(), "conditional identity; got {:?}", v);
    }
}
