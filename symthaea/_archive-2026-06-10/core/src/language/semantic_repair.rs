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

    // NEW: Compiler Help-Driven Return Injection
    if joined.contains("you might have meant to return this value") {
        if let Some(repaired) = inject_explicit_return(source) {
            return Some(repaired);
        }
    }

    if joined.contains("use the `?` operator to extract")
        || joined.contains("propagating a `result::err` value")
    {
        if let Some(repaired) = try_append_question_mark_to_ok_arg(source) {
            return Some(repaired);
        }
    }

    // NEW: Missing Import Injection
    if joined.contains("cannot find") && joined.contains("in this scope") {
        if let Some(repaired) = try_inject_missing_imports(source, &joined) {
            return Some(repaired);
        }
    }

    // NEW: Vector/Slice Conversion
    if mentions_vec_slice_mismatch(&joined) {
        if let Some(repaired) = try_fix_vec_slice_mismatch(source, &joined) {
            return Some(repaired);
        }
    }

    // NEW: Lifetime Return Fix
    if joined.contains("missing lifetime specifier")
        || joined.contains("explicit lifetime name needed")
    {
        if let Some(repaired) = try_fix_missing_lifetime(source) {
            return Some(repaired);
        }
    }

    // NEW: Missing Reference Fix
    if joined.contains("expected `&") && !joined.contains("found `&") {
        if let Some(repaired) = try_fix_missing_reference(source, &joined) {
            return Some(repaired);
        }
    }

    None
}

fn mentions_vec_slice_mismatch(diagnostics: &str) -> bool {
    (diagnostics.contains("expected `&[") && diagnostics.contains("found `&vec<"))
        || (diagnostics.contains("expected `vec<") && diagnostics.contains("found `&["))
}

fn try_inject_missing_imports(source: &str, diagnostics: &str) -> Option<String> {
    let mut file = syn::parse_file(source).ok()?;
    let mut changed = false;

    let missing_types = [
        ("HashMap", "std::collections::HashMap"),
        ("BTreeMap", "std::collections::BTreeMap"),
        ("Arc", "std::sync::Arc"),
        ("Mutex", "std::sync::Mutex"),
        ("RefCell", "std::cell::RefCell"),
        ("Instant", "std::time::Instant"),
        ("Duration", "std::time::Duration"),
    ];

    for (name, path) in missing_types {
        let name_lower = name.to_lowercase();
        if diagnostics.contains(&format!("cannot find type `{name_lower}`"))
            || diagnostics.contains(&format!("cannot find value `{name_lower}`"))
        {
            if !has_import(&file, name) {
                let path_segments: Vec<syn::Ident> = path
                    .split("::")
                    .map(|s| syn::parse_str(s).unwrap())
                    .collect();
                let use_item: syn::ItemUse = syn::parse_quote!(use #(#path_segments)::* ;);
                file.items.insert(0, syn::Item::Use(use_item));
                changed = true;
            }
        }
    }

    changed.then(|| file.into_token_stream().to_string())
}

fn has_import(file: &syn::File, name: &str) -> bool {
    for item in &file.items {
        if let syn::Item::Use(item_use) = item {
            let use_str = item_use.to_token_stream().to_string();
            if use_str.contains(name) || use_str.contains(&format!("::{}", name)) {
                return true;
            }
        }
    }
    false
}

fn try_fix_vec_slice_mismatch(source: &str, diagnostics: &str) -> Option<String> {
    let mut file = syn::parse_file(source).ok()?;
    let mut visitor = VecSliceFixVisitor {
        diagnostics: diagnostics.to_string(),
        changed: false,
    };
    visitor.visit_file_mut(&mut file);
    visitor
        .changed
        .then(|| file.into_token_stream().to_string())
}

struct VecSliceFixVisitor {
    diagnostics: String,
    changed: bool,
}

impl VisitMut for VecSliceFixVisitor {
    fn visit_expr_mut(&mut self, expr: &mut syn::Expr) {
        if self.changed {
            return;
        }

        // expected `Vec<...>`, found `&[...]` -> .to_vec()
        if self.diagnostics.contains("expected `vec") && self.diagnostics.contains("found `&[") {
            if matches!(expr, syn::Expr::Reference(_) | syn::Expr::Path(_)) {
                let original = expr.clone();
                *expr = syn::parse_quote!(#original.to_vec());
                self.changed = true;
                return;
            }
        }

        // expected `&[...]`, found `&Vec<...>` -> .as_slice()
        if self.diagnostics.contains("expected `&[") && self.diagnostics.contains("found `&vec") {
            if let syn::Expr::Reference(r) = expr {
                let inner = &r.expr;
                *expr = syn::parse_quote!(#inner.as_slice());
                self.changed = true;
                return;
            }
        }

        visit_mut::visit_expr_mut(self, expr);
    }
}

fn try_fix_missing_lifetime(source: &str) -> Option<String> {
    let mut file = syn::parse_file(source).ok()?;
    let mut visitor = LifetimeFixVisitor { changed: false };
    visitor.visit_file_mut(&mut file);
    visitor
        .changed
        .then(|| file.into_token_stream().to_string())
}

struct LifetimeFixVisitor {
    changed: bool,
}

impl VisitMut for LifetimeFixVisitor {
    fn visit_item_fn_mut(&mut self, i: &mut syn::ItemFn) {
        // Collect all available lifetimes from parameters
        let mut available_lifetimes = Vec::new();
        for input in &i.sig.inputs {
            if let syn::FnArg::Typed(pat_type) = input {
                let type_str = pat_type.ty.to_token_stream().to_string();
                if let Some(start) = type_str.find('\'') {
                    let end = type_str[start..]
                        .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                        .unwrap_or(type_str[start..].len())
                        + start;
                    available_lifetimes.push(type_str[start..end].to_string());
                }
            }
        }

        // If no explicit lifetimes found, check for the default 'a pattern or use 'static if no inputs
        let target_lifetime = if let Some(lt) = available_lifetimes.first() {
            lt.clone()
        } else if i.sig.inputs.is_empty() {
            "'static".to_string()
        } else {
            // Default to 'a if it exists in generics, otherwise nothing
            if i.sig
                .generics
                .params
                .iter()
                .any(|p| matches!(p, syn::GenericParam::Lifetime(_)))
            {
                "'a".to_string()
            } else {
                return;
            }
        };

        let lt: syn::Lifetime = syn::parse_str(&target_lifetime).unwrap();

        // Apply to return type if it's a reference lacking a lifetime
        if let syn::ReturnType::Type(_, ty) = &mut i.sig.output {
            if inject_lifetime_if_missing(ty.as_mut(), &lt) {
                self.changed = true;
            }
        }

        visit_mut::visit_item_fn_mut(self, i);
    }
}

fn inject_lifetime_if_missing(ty: &mut syn::Type, lt: &syn::Lifetime) -> bool {
    match ty {
        syn::Type::Reference(tr) => {
            if tr.lifetime.is_none() {
                tr.lifetime = Some(lt.clone());
                return true;
            }
        }
        syn::Type::Path(tp) => {
            // Check for Result<T, E> or Option<T>
            if let Some(last) = tp.path.segments.last_mut() {
                if let syn::PathArguments::AngleBracketed(args) = &mut last.arguments {
                    let mut changed = false;
                    for arg in &mut args.args {
                        if let syn::GenericArgument::Type(inner_ty) = arg {
                            if inject_lifetime_if_missing(inner_ty, lt) {
                                changed = true;
                            }
                        }
                    }
                    return changed;
                }
            }
        }
        _ => {}
    }
    false
}

fn try_fix_missing_reference(source: &str, diagnostics: &str) -> Option<String> {
    let mut file = syn::parse_file(source).ok()?;
    // Heuristic: extract the 'found' type or variable name if possible
    let name = binding_name_from_backticks(diagnostics);

    let mut visitor = ReferenceFixVisitor {
        name,
        changed: false,
    };
    visitor.visit_file_mut(&mut file);
    visitor
        .changed
        .then(|| file.into_token_stream().to_string())
}

struct ReferenceFixVisitor {
    name: Option<String>,
    changed: bool,
}

impl VisitMut for ReferenceFixVisitor {
    fn visit_expr_mut(&mut self, expr: &mut syn::Expr) {
        if self.changed {
            return;
        }

        if let Some(ref target_name) = self.name {
            match expr {
                syn::Expr::Path(p) if p.path.is_ident(target_name) => {
                    let original = expr.clone();
                    *expr = syn::parse_quote!(&#original);
                    self.changed = true;
                    return;
                }
                _ => {}
            }
        }

        visit_mut::visit_expr_mut(self, expr);
    }
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

fn inject_explicit_return(source: &str) -> Option<String> {
    let mut file = syn::parse_file(source).ok()?;
    let mut visitor = ExplicitReturnVisitor { changed: false };
    visitor.visit_file_mut(&mut file);
    visitor
        .changed
        .then(|| file.into_token_stream().to_string())
}

struct ExplicitReturnVisitor {
    changed: bool,
}

impl VisitMut for ExplicitReturnVisitor {
    fn visit_block_mut(&mut self, block: &mut syn::Block) {
        if let Some(syn::Stmt::Expr(expr, None)) = block.stmts.last_mut() {
            if !matches!(
                expr,
                syn::Expr::Return(_)
                    | syn::Expr::While(_)
                    | syn::Expr::ForLoop(_)
                    | syn::Expr::Loop(_)
                    | syn::Expr::If(_)
            ) {
                let original = expr.clone();
                *expr = syn::parse_quote!(return #original);
                self.changed = true;
            }
        }
        visit_mut::visit_block_mut(self, block);
    }
}

fn try_append_question_mark_to_ok_arg(source: &str) -> Option<String> {
    let mut file = syn::parse_file(source).ok()?;
    let mut visitor = OkArgQuestionMarkVisitor { changed: false };
    visitor.visit_file_mut(&mut file);
    visitor
        .changed
        .then(|| file.into_token_stream().to_string())
}

struct OkArgQuestionMarkVisitor {
    changed: bool,
}

impl VisitMut for OkArgQuestionMarkVisitor {
    fn visit_expr_call_mut(&mut self, call: &mut syn::ExprCall) {
        if let syn::Expr::Path(path) = call.func.as_ref() {
            if path.path.is_ident("Ok") {
                if let Some(arg) = call.args.first_mut() {
                    if !matches!(arg, syn::Expr::Try(_)) {
                        let original = arg.clone();
                        *arg = syn::parse_quote!(#original?);
                        self.changed = true;
                    }
                }
            }
        }
        visit_mut::visit_expr_call_mut(self, call);
    }
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
    if !name.is_empty()
        && name
            .chars()
            .all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
    {
        Some(name.to_string())
    } else {
        None
    }
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

    #[test]
    fn injects_explicit_return_based_on_compiler_help() {
        let source = r#"
            pub fn first_positive(items: &[i32]) -> Result<i32, &'static str> {
                let mut result = Ok(0);
                if false {
                    // check branch
                } else {
                    result
                }
                result
            }
        "#;
        let diagnostic = "help: you might have meant to return this value";
        let repaired = try_semantic_ast_repair(source, &[diagnostic.into()]).unwrap();
        assert!(repaired.contains("return result"));
        syn::parse_file(&repaired).unwrap();
    }

    #[test]
    fn appends_question_mark_to_ok_argument_on_help_signal() {
        let source = r#"
            pub fn first_positive(items: &[i32]) -> Result<i32, &'static str> {
                Ok(items.first().copied().ok_or_else(|| "empty"))
            }
        "#;
        let diagnostic = "help: use the `?` operator to extract the `Result<i32, String>` value";
        let repaired = try_semantic_ast_repair(source, &[diagnostic.into()]).unwrap();

        // Strip out spacing artifacts to survive any local token formatting configurations safely
        let normalized = repaired.replace(' ', "");
        assert!(normalized.contains("Ok(items.first().copied().ok_or_else(||\"empty\")?)"));
        syn::parse_file(&repaired).unwrap();
    }

    #[test]
    fn injects_missing_hashmap_import() {
        let source = "pub fn new_map() { let mut m = HashMap::new(); }";
        let diagnostic =
            "error[E0433]: failed to resolve: cannot find type `HashMap` in this scope";
        let repaired = try_semantic_ast_repair(source, &[diagnostic.into()]).unwrap();

        let normalized = repaired.replace(' ', "");
        assert!(normalized.contains("usestd::collections::HashMap;"));
        syn::parse_file(&repaired).unwrap();
    }

    #[test]
    fn fixes_vec_to_slice_mismatch() {
        let source =
            "pub fn take_slice(s: &[i32]) {} pub fn main() { let v = vec![1]; take_slice(&v); }";
        let diagnostic = "expected `&[i32]`, found `&Vec<i32>`";
        let repaired = try_semantic_ast_repair(source, &[diagnostic.into()]).unwrap();

        assert!(repaired.replace(' ', "").contains("v.as_slice()"));
        syn::parse_file(&repaired).unwrap();
    }

    #[test]
    fn fixes_slice_to_vec_mismatch() {
        let source = "pub fn take_vec(v: Vec<i32>) {} pub fn main() { let s = &[1]; take_vec(s); }";
        let diagnostic = "expected `Vec<i32>`, found `&[i32]`";
        let repaired = try_semantic_ast_repair(source, &[diagnostic.into()]).unwrap();

        let normalized = repaired.replace(' ', "");
        assert!(normalized.contains("s.to_vec()") || normalized.contains("[1].to_vec()"));
        syn::parse_file(&repaired).unwrap();
    }

    #[test]
    fn fixes_missing_lifetime_in_return() {
        let source = "pub fn get_ref<'a>(data: &'a [i32]) -> &i32 { &data[0] }";
        let diagnostic = "error[E0106]: missing lifetime specifier";
        let repaired = try_semantic_ast_repair(source, &[diagnostic.into()]).unwrap();

        assert!(repaired.contains("-> &'a i32"));
        syn::parse_file(&repaired).unwrap();
    }

    #[test]
    fn fixes_missing_reference_operator() {
        let source = "pub fn take_ref(s: &str) {} pub fn main() { let val = \"hi\".to_string(); take_ref(val); }";
        let diagnostic = "expected `&str`, found `String` (variable `val`)";
        let repaired = try_semantic_ast_repair(source, &[diagnostic.into()]).unwrap();

        assert!(repaired.contains("take_ref(&val)"));
        syn::parse_file(&repaired).unwrap();
    }
}
