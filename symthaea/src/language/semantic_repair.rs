// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic semantic AST repairs for generated Rust.
//!
//! This module is intentionally conservative. It only applies transforms when
//! a diagnostic points to a narrow Rust fact we can repair directly in the AST.

use quote::{ToTokens, quote};
use syn::visit_mut::{self, VisitMut};

/// Try one deterministic semantic repair before falling back to probabilistic
/// regeneration.
pub fn try_semantic_ast_repair(source: &str, diagnostics: &[String]) -> Option<String> {
    let joined = diagnostics.join("\n").to_ascii_lowercase();

    if mentions_result_mismatch(&joined) {
        if let Some(repaired) = wrap_result_tail_expression(source) {
            return Some(repaired);
        }
    }

    if mentions_missing_mut(&joined) {
        if let Some(binding) = binding_name_from_backticks(&diagnostics.join("\n")) {
            if let Some(repaired) = add_mut_to_binding(source, &binding) {
                return Some(repaired);
            }
        }
    }

    None
}

fn mentions_result_mismatch(diagnostics: &str) -> bool {
    diagnostics.contains("expected enum `result")
        || diagnostics.contains("expected result")
        || diagnostics.contains("return type `result")
        || (diagnostics.contains("mismatched types") && diagnostics.contains("ok("))
}

fn mentions_missing_mut(diagnostics: &str) -> bool {
    diagnostics.contains("requires `mut`")
        || diagnostics.contains("cannot assign")
        || diagnostics.contains("cannot borrow")
}

fn wrap_result_tail_expression(source: &str) -> Option<String> {
    let mut file = syn::parse_file(source).ok()?;
    let mut changed = false;

    for item in &mut file.items {
        let syn::Item::Fn(item_fn) = item else {
            continue;
        };
        if !function_returns_result(&item_fn.sig) {
            continue;
        }
        let Some(syn::Stmt::Expr(tail, None)) = item_fn.block.stmts.last_mut() else {
            continue;
        };
        if is_result_constructor(tail) {
            continue;
        }
        let original = tail.clone();
        *tail = syn::parse_quote!(Ok(#original));
        changed = true;
    }

    changed.then(|| file.into_token_stream().to_string())
}

fn add_mut_to_binding(source: &str, binding: &str) -> Option<String> {
    let mut file = syn::parse_file(source).ok()?;
    let mut visitor = AddMutVisitor {
        binding,
        changed: false,
    };
    visitor.visit_file_mut(&mut file);
    visitor
        .changed
        .then(|| file.into_token_stream().to_string())
}

struct AddMutVisitor<'a> {
    binding: &'a str,
    changed: bool,
}

impl VisitMut for AddMutVisitor<'_> {
    fn visit_pat_ident_mut(&mut self, pat_ident: &mut syn::PatIdent) {
        if pat_ident.ident == self.binding && pat_ident.mutability.is_none() {
            pat_ident.mutability = Some(syn::parse_quote!(mut));
            self.changed = true;
        }
        visit_mut::visit_pat_ident_mut(self, pat_ident);
    }
}

fn function_returns_result(sig: &syn::Signature) -> bool {
    match &sig.output {
        syn::ReturnType::Type(_, ty) => {
            let normalized = ty.to_token_stream().to_string().replace(' ', "");
            normalized.starts_with("Result<")
                || normalized.starts_with("std::result::Result<")
                || normalized.starts_with("core::result::Result<")
        }
        syn::ReturnType::Default => false,
    }
}

fn is_result_constructor(expr: &syn::Expr) -> bool {
    let syn::Expr::Call(call) = expr else {
        return false;
    };
    let syn::Expr::Path(path) = call.func.as_ref() else {
        return false;
    };
    path.path
        .segments
        .last()
        .is_some_and(|segment| matches!(segment.ident.to_string().as_str(), "Ok" | "Err"))
}

fn binding_name_from_backticks(diagnostic: &str) -> Option<String> {
    let start = diagnostic.find('`')? + 1;
    let end = diagnostic[start..].find('`')? + start;
    let name = &diagnostic[start..end];
    (!name.is_empty()
        && name
            .chars()
            .all(|ch| ch == '_' || ch.is_ascii_alphanumeric()))
    .then(|| name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_result_tail_expression_in_ok() {
        let source = "pub fn parse() -> Result<i32, String> { 42 }";
        let repaired = try_semantic_ast_repair(
            source,
            &["mismatched types: expected enum `Result<i32, String>`, found integer".into()],
        )
        .unwrap();

        assert!(repaired.contains("Ok"));
        syn::parse_file(&repaired).unwrap();
    }

    #[test]
    fn adds_mut_to_named_binding() {
        let source = "pub fn inc() -> i32 { let value = 1; value += 1; value }";
        let repaired = try_semantic_ast_repair(
            source,
            &["assignment to `value` requires `mut` binding".into()],
        )
        .unwrap();

        assert!(repaired.contains("let mut value"));
        syn::parse_file(&repaired).unwrap();
    }
}
