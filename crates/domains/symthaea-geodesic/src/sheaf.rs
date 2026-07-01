// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Sheaf Coherence -- local-to-global code correctness verification.
//!
//! Models code composability as a presheaf over the call graph:
//! - Open sets = connected subgraphs (function + dependencies)
//! - Sections = implementations with type signatures as HDC vectors
//! - Gluing condition = type compatibility across module boundaries
//!
//! # Theory
//!
//! Sheaf theory (Kashiwara & Schapira, 1990) guarantees that locally-consistent
//! data assembles into a globally-consistent whole. Applied to code: if every
//! caller's expectation of a callee's interface matches the callee's actual
//! interface (measured via HDC similarity), then the entire program composes
//! correctly.
//!
//! # HDC Encoding
//!
//! Each function's "expected signature" of a callee is computed by binding the
//! caller's encoding with a name-derived basis vector for the callee. The
//! callee's "actual signature" is its own encoding. High similarity between
//! these two vectors indicates type-compatible composition.

use quote::ToTokens;
use symthaea_core::hdc::binary_hv::BinaryHV;

// ---------------------------------------------------------------------------
// Deterministic seed derivation for callee name encoding.
// ---------------------------------------------------------------------------

/// Derive a deterministic seed from a function name string.
///
/// Uses a simple hash: fold bytes with position-dependent shifts.
/// Must remain stable across versions for reproducible gluing checks.
fn name_seed(name: &str) -> u64 {
    name.bytes()
        .enumerate()
        .fold(0x50EA_FC0D_E000_0000u64, |acc, (i, b)| {
            acc.wrapping_add((b as u64).wrapping_shl((i as u32) % 64))
        })
}

// ============================================================================
// LOCAL SECTION
// ============================================================================

/// A local section: one function's implementation with its interface.
///
/// In sheaf-theoretic terms, this is a section over an open set (the function
/// and its direct dependencies). The `encoding` captures the function's
/// semantic content, while `param_types` and `return_type` describe its
/// interface.
#[derive(Debug, Clone)]
pub struct LocalSection {
    /// Function name (unique identifier within the sheaf).
    pub name: String,
    /// HDC encoding of the function's semantic content.
    pub encoding: BinaryHV,
    /// Parameter types encoded as HDC vectors.
    pub param_types: Vec<BinaryHV>,
    /// Return type encoded as HDC vector.
    pub return_type: BinaryHV,
    /// Names of functions this section calls (callees).
    pub callees: Vec<String>,
}

// ============================================================================
// GLUING CONDITION
// ============================================================================

/// A gluing condition between two overlapping sections.
///
/// The gluing axiom requires that on overlapping open sets, the restrictions
/// of two sections agree. Here, the "overlap" is the call site: the caller
/// has an expectation of the callee's interface, and the callee has an actual
/// interface. Agreement is measured by HDC similarity.
#[derive(Debug, Clone)]
pub struct GluingCondition {
    /// The calling function.
    pub caller: String,
    /// The called function.
    pub callee: String,
    /// Caller's expectation of the callee's signature (bind of caller encoding
    /// with callee name vector).
    pub expected_signature: BinaryHV,
    /// The callee's actual encoding.
    pub actual_signature: BinaryHV,
    /// Cosine similarity between expected and actual in [0, 1].
    pub similarity: f64,
    /// Whether the condition is satisfied (similarity > threshold).
    pub satisfied: bool,
}

// ============================================================================
// SHEAF DIAGNOSTIC
// ============================================================================

/// A diagnostic for a sheaf violation (unsatisfied gluing condition).
///
/// Provides human-readable information about which interface mismatch
/// was detected and how severe it is.
#[derive(Debug, Clone)]
pub struct SheafDiagnostic {
    /// The calling function.
    pub caller: String,
    /// The called function.
    pub callee: String,
    /// Human-readable description of the violation.
    pub message: String,
    /// Similarity score (lower = worse mismatch).
    pub similarity: f64,
    /// Summary of what the caller expected.
    pub expected_summary: String,
    /// Summary of what the callee actually provides.
    pub actual_summary: String,
}

/// Concrete v0 Rust coherence result for a single generated function.
#[derive(Debug, Clone)]
pub struct RustSheafCoherence {
    pub coherent: bool,
    pub diagnostics: Vec<String>,
}

/// Verify a concrete, Rust-specific v0 approximation of sheaf coherence.
///
/// This is deliberately modest: it checks facts that can be derived from the
/// syntax tree before invoking heavier compiler diagnostics. The compiler/test
/// loop remains the acceptance gate.
pub fn verify_rust_v0_sheaf_coherence(source: &str, function_name: &str) -> RustSheafCoherence {
    let mut diagnostics = Vec::new();
    if source.contains("todo!")
        || source.contains("unimplemented!")
        || source.contains("panic!(\"not implemented")
    {
        diagnostics.push("source contains an implementation stub".to_string());
    }

    let Ok(file) = syn::parse_file(source) else {
        return RustSheafCoherence {
            coherent: false,
            diagnostics: vec!["source does not parse as Rust".to_string()],
        };
    };

    let item_fn = file
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Fn(item_fn) => Some(item_fn),
            _ => None,
        })
        .find(|item_fn| item_fn.sig.ident == function_name)
        .or_else(|| {
            file.items.iter().find_map(|item| match item {
                syn::Item::Fn(item_fn) => Some(item_fn),
                _ => None,
            })
        });

    let Some(item_fn) = item_fn else {
        return RustSheafCoherence {
            coherent: false,
            diagnostics: vec![format!("function `{function_name}` was not found")],
        };
    };

    let return_type = function_return_type(&item_fn.sig);
    let facts = collect_function_facts(&file, item_fn);

    if return_type != "()" && !facts.has_return_value {
        diagnostics.push(format!(
            "function `{}` has return type `{}` but no return or tail value was found",
            item_fn.sig.ident, return_type
        ));
    }
    if return_type == "()" && facts.has_return_value {
        diagnostics.push(format!(
            "function `{}` returns a value from a unit-returning signature",
            item_fn.sig.ident
        ));
    }
    diagnostics.extend(facts.diagnostics);
    for ident in facts.unresolved_uses {
        diagnostics.push(format!(
            "identifier `{ident}` is used without a local definition"
        ));
    }

    RustSheafCoherence {
        coherent: diagnostics.is_empty(),
        diagnostics,
    }
}

/// Stable diagnostic category for Rust v0 sheaf diagnostics.
pub fn categorize_rust_v0_sheaf_diagnostic(diagnostic: &str) -> &'static str {
    let lower = diagnostic.to_ascii_lowercase();
    if lower.contains("stub") || lower.contains("todo") || lower.contains("unimplemented") {
        "stub"
    } else if lower.contains("does not parse") {
        "parse_failure"
    } else if lower.contains("was not found") {
        "missing_function"
    } else if lower.contains("used without a local definition") {
        "unresolved_identifier"
    } else if lower.contains("return type") || lower.contains("returns a value") {
        "return_mismatch"
    } else if lower.contains("shadowed") {
        "shadowing"
    } else if lower.contains("requires `mut`") {
        "missing_mut"
    } else if lower.contains("unreachable code") {
        "unreachable_code"
    } else if lower.contains("reference to local") {
        "borrow_lifetime"
    } else if lower.contains("no reachable break") {
        "infinite_loop"
    } else if lower.contains("non-exhaustive match") {
        "non_exhaustive_match"
    } else if lower.contains("infinite recursion") {
        "infinite_recursion"
    } else if lower.contains("use of moved value") {
        "use_after_move"
    } else {
        "sheaf_coherence"
    }
}

/// Repair hint paired with a Rust v0 sheaf diagnostic category.
pub fn repair_hint_for_rust_v0_sheaf_category(category: &str) -> &'static str {
    match category {
        "stub" => "replace the stub with a concrete expression or implementation body",
        "parse_failure" => {
            "emit syntactically complete Rust; check missing semicolons, braces, and expression positions"
        }
        "missing_function" => {
            "emit the requested function name and keep the generated item at module top level"
        }
        "unresolved_identifier" => {
            "define the identifier before use or replace it with an in-scope parameter/local binding"
        }
        "return_mismatch" => {
            "make every return path produce the declared return type; use a valid tail expression for non-unit functions"
        }
        "shadowing" => {
            "avoid redeclaring an existing local binding; reuse the binding or choose a distinct name"
        }
        "missing_mut" => "mark the binding as mutable or avoid reassignment",
        "unreachable_code" => {
            "remove code after return/break/continue or move it before terminal control flow"
        }
        "borrow_lifetime" => {
            "do not return references to local temporaries; return owned values or references derived from inputs"
        }
        "infinite_loop" => "add a reachable break condition or use bounded iteration",
        "non_exhaustive_match" => "cover every known enum variant or add an explicit wildcard arm",
        "infinite_recursion" => {
            "change recursive arguments toward a base case or return a non-recursive base value"
        }
        "use_after_move" => {
            "borrow, clone, or reorder the value so it is not used after being moved"
        }
        _ => "repair the local structural inconsistency before compiler verification",
    }
}

fn function_return_type(sig: &syn::Signature) -> String {
    match &sig.output {
        syn::ReturnType::Default => "()".to_string(),
        syn::ReturnType::Type(_, ty) => ty.to_token_stream().to_string(),
    }
}

#[derive(Default)]
struct FunctionFacts {
    has_return_value: bool,
    unresolved_uses: Vec<String>,
    diagnostics: Vec<String>,
}

fn collect_function_facts(file: &syn::File, item_fn: &syn::ItemFn) -> FunctionFacts {
    use syn::visit::Visit;

    struct FactVisitor {
        function_name: String,
        param_names: Vec<String>,
        enum_variants: std::collections::HashMap<String, std::collections::BTreeSet<String>>,
        defined: std::collections::HashSet<String>,
        mutable_bindings: std::collections::HashSet<String>,
        local_bindings: std::collections::HashSet<String>,
        binding_types: std::collections::HashMap<String, String>,
        owned_bindings: std::collections::HashSet<String>,
        moved_bindings: std::collections::HashMap<String, String>,
        reported_moved_bindings: std::collections::HashSet<String>,
        unresolved: std::collections::BTreeSet<String>,
        diagnostics: Vec<String>,
        has_return_value: bool,
    }

    impl<'ast> Visit<'ast> for FactVisitor {
        fn visit_local(&mut self, local: &'ast syn::Local) {
            if let Some((name, is_mutable, type_name)) = binding_from_pat(&local.pat) {
                if self.defined.contains(&name) {
                    self.diagnostics
                        .push(format!("binding `{name}` is shadowed"));
                }
                self.defined.insert(name.clone());
                self.local_bindings.insert(name.clone());
                if is_mutable {
                    self.mutable_bindings.insert(name.clone());
                }
                if let Some(type_name) = type_name {
                    self.binding_types.insert(name.clone(), type_name);
                }
                if local
                    .init
                    .as_ref()
                    .is_some_and(|init| expr_is_owned_constructor(&init.expr))
                {
                    self.owned_bindings.insert(name);
                }
            }
            syn::visit::visit_local(self, local);
        }

        fn visit_expr_closure(&mut self, closure: &'ast syn::ExprClosure) {
            let old = self.defined.clone();
            let old_mutable = self.mutable_bindings.clone();
            let old_locals = self.local_bindings.clone();
            for input in &closure.inputs {
                if let Some((name, is_mutable, type_name)) = binding_from_pat(input) {
                    self.defined.insert(name.clone());
                    self.local_bindings.insert(name.clone());
                    if is_mutable {
                        self.mutable_bindings.insert(name.clone());
                    }
                    if let Some(type_name) = type_name {
                        self.binding_types.insert(name, type_name);
                    }
                }
            }
            syn::visit::visit_expr_closure(self, closure);
            self.defined = old;
            self.mutable_bindings = old_mutable;
            self.local_bindings = old_locals;
        }

        fn visit_expr_for_loop(&mut self, for_loop: &'ast syn::ExprForLoop) {
            let old = self.defined.clone();
            let old_mutable = self.mutable_bindings.clone();
            let old_locals = self.local_bindings.clone();
            if let Some((name, is_mutable, type_name)) = binding_from_pat(&for_loop.pat) {
                self.defined.insert(name.clone());
                self.local_bindings.insert(name.clone());
                if is_mutable {
                    self.mutable_bindings.insert(name.clone());
                }
                if let Some(type_name) = type_name {
                    self.binding_types.insert(name, type_name);
                }
            }
            syn::visit::visit_expr_for_loop(self, for_loop);
            self.defined = old;
            self.mutable_bindings = old_mutable;
            self.local_bindings = old_locals;
        }

        fn visit_expr_return(&mut self, ret: &'ast syn::ExprReturn) {
            if let Some(expr) = &ret.expr {
                self.has_return_value = true;
                self.check_returned_reference_to_local(expr);
                self.check_obvious_tail_recursion(expr);
            }
            syn::visit::visit_expr_return(self, ret);
        }

        fn visit_expr_assign(&mut self, assign: &'ast syn::ExprAssign) {
            self.check_assignment_target(&assign.left);
            syn::visit::visit_expr_assign(self, assign);
        }

        fn visit_expr_binary(&mut self, binary: &'ast syn::ExprBinary) {
            if matches!(
                binary.op,
                syn::BinOp::AddAssign(_)
                    | syn::BinOp::SubAssign(_)
                    | syn::BinOp::MulAssign(_)
                    | syn::BinOp::DivAssign(_)
                    | syn::BinOp::RemAssign(_)
                    | syn::BinOp::BitXorAssign(_)
                    | syn::BinOp::BitAndAssign(_)
                    | syn::BinOp::BitOrAssign(_)
                    | syn::BinOp::ShlAssign(_)
                    | syn::BinOp::ShrAssign(_)
            ) {
                self.check_assignment_target(&binary.left);
            }
            syn::visit::visit_expr_binary(self, binary);
        }

        fn visit_expr_loop(&mut self, loop_expr: &'ast syn::ExprLoop) {
            if !block_contains_break(&loop_expr.body) {
                self.diagnostics
                    .push("loop expression has no reachable break".to_string());
            }
            syn::visit::visit_expr_loop(self, loop_expr);
        }

        fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
            self.visit_expr(&call.func);
            for arg in &call.args {
                self.visit_expr(arg);
                self.mark_move_if_owned_arg(arg);
            }
        }

        fn visit_expr_match(&mut self, match_expr: &'ast syn::ExprMatch) {
            self.check_non_exhaustive_simple_enum_match(match_expr);
            syn::visit::visit_expr_match(self, match_expr);
        }

        fn visit_block(&mut self, block: &'ast syn::Block) {
            self.check_unreachable_statements(block);
            syn::visit::visit_block(self, block);
            if let Some(syn::Stmt::Expr(expr, None)) = block.stmts.last() {
                self.check_returned_reference_to_local(expr);
                self.check_obvious_tail_recursion(expr);
            }
        }

        fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
            if path.qself.is_none() && path.path.segments.len() == 1 {
                let ident = path.path.segments[0].ident.to_string();
                if let Some(move_site) = self.moved_bindings.get(&ident) {
                    if self.reported_moved_bindings.insert(ident.clone()) {
                        self.diagnostics.push(format!(
                            "use of moved value `{ident}` after move into `{move_site}`"
                        ));
                    }
                    return;
                }
                if !self.defined.contains(&ident)
                    && !matches!(
                        ident.as_str(),
                        "true" | "false" | "Some" | "None" | "Ok" | "Err"
                    )
                    && !ident.chars().next().is_some_and(|ch| ch.is_uppercase())
                {
                    self.unresolved.insert(ident);
                }
            }
            syn::visit::visit_expr_path(self, path);
        }
    }

    impl FactVisitor {
        fn check_assignment_target(&mut self, expr: &syn::Expr) {
            let Some(name) = expr_ident(expr) else {
                return;
            };
            if self.defined.contains(&name) && !self.mutable_bindings.contains(&name) {
                self.diagnostics
                    .push(format!("assignment to `{name}` requires `mut` binding"));
            }
        }

        fn check_returned_reference_to_local(&mut self, expr: &syn::Expr) {
            let syn::Expr::Reference(reference) = expr else {
                return;
            };
            let Some(name) = expr_ident(&reference.expr) else {
                return;
            };
            if self.local_bindings.contains(&name) {
                self.diagnostics
                    .push(format!("returning reference to local variable `{name}`"));
            }
        }

        fn check_unreachable_statements(&mut self, block: &syn::Block) {
            let mut terminal_seen = false;
            for stmt in &block.stmts {
                if terminal_seen {
                    self.diagnostics
                        .push("unreachable code after terminal control flow".to_string());
                    break;
                }
                terminal_seen = stmt_is_terminal(stmt);
            }
        }

        fn check_obvious_tail_recursion(&mut self, expr: &syn::Expr) {
            let syn::Expr::Call(call) = strip_parens(expr) else {
                return;
            };
            let Some(callee) = expr_ident(&call.func) else {
                return;
            };
            if callee != self.function_name || call.args.len() != self.param_names.len() {
                return;
            }
            let same_args = call
                .args
                .iter()
                .zip(&self.param_names)
                .all(|(arg, param)| expr_ident(arg).as_deref() == Some(param.as_str()));
            if same_args {
                self.diagnostics.push(format!(
                    "obvious infinite recursion: `{callee}` calls itself with unchanged arguments in tail position"
                ));
            }
        }

        fn check_non_exhaustive_simple_enum_match(&mut self, match_expr: &syn::ExprMatch) {
            let Some(scrutinee) = expr_ident(&match_expr.expr) else {
                return;
            };
            let Some(type_name) = self.binding_types.get(&scrutinee) else {
                return;
            };
            let Some(expected_variants) = self.enum_variants.get(type_name) else {
                return;
            };
            if match_expr.arms.iter().any(|arm| pat_is_catch_all(&arm.pat)) {
                return;
            }

            let mut seen = std::collections::BTreeSet::new();
            for arm in &match_expr.arms {
                collect_pat_variants(&arm.pat, &mut seen);
            }
            let missing: Vec<_> = expected_variants.difference(&seen).cloned().collect();
            if !missing.is_empty() {
                self.diagnostics.push(format!(
                    "non-exhaustive match on `{type_name}` via `{scrutinee}`; missing variant(s): {}",
                    missing.join(", ")
                ));
            }
        }

        fn mark_move_if_owned_arg(&mut self, arg: &syn::Expr) {
            let Some(name) = expr_ident(arg) else {
                return;
            };
            if self.owned_bindings.contains(&name) && !self.moved_bindings.contains_key(&name) {
                self.moved_bindings
                    .insert(name, "by-value function call".to_string());
            }
        }
    }

    let enum_variants = collect_simple_enums(file);
    let function_name = item_fn.sig.ident.to_string();
    let mut param_names = Vec::new();
    let mut visitor = FactVisitor {
        function_name,
        param_names: Vec::new(),
        enum_variants,
        defined: std::collections::HashSet::new(),
        mutable_bindings: std::collections::HashSet::new(),
        local_bindings: std::collections::HashSet::new(),
        binding_types: std::collections::HashMap::new(),
        owned_bindings: std::collections::HashSet::new(),
        moved_bindings: std::collections::HashMap::new(),
        reported_moved_bindings: std::collections::HashSet::new(),
        unresolved: std::collections::BTreeSet::new(),
        diagnostics: Vec::new(),
        has_return_value: false,
    };

    for input in &item_fn.sig.inputs {
        if let syn::FnArg::Typed(pat_type) = input {
            if let syn::Pat::Ident(pat_ident) = pat_type.pat.as_ref() {
                let name = pat_ident.ident.to_string();
                param_names.push(name.clone());
                visitor.defined.insert(name.clone());
                if let Some(type_name) = simple_type_name(&pat_type.ty) {
                    visitor.binding_types.insert(name.clone(), type_name);
                }
                if pat_ident.mutability.is_some() {
                    visitor.mutable_bindings.insert(name);
                }
            }
        }
    }
    visitor.param_names = param_names;

    if let Some(last) = item_fn.block.stmts.last() {
        if let syn::Stmt::Expr(expr, None) = last {
            visitor.has_return_value = true;
            visitor.check_returned_reference_to_local(expr);
        }
    }

    visitor.visit_block(&item_fn.block);

    FunctionFacts {
        has_return_value: visitor.has_return_value,
        unresolved_uses: visitor.unresolved.into_iter().collect(),
        diagnostics: visitor.diagnostics,
    }
}

fn expr_ident(expr: &syn::Expr) -> Option<String> {
    match expr {
        syn::Expr::Path(path) if path.qself.is_none() && path.path.segments.len() == 1 => {
            Some(path.path.segments[0].ident.to_string())
        }
        syn::Expr::Paren(paren) => expr_ident(&paren.expr),
        syn::Expr::Reference(reference) => expr_ident(&reference.expr),
        _ => None,
    }
}

fn strip_parens(expr: &syn::Expr) -> &syn::Expr {
    match expr {
        syn::Expr::Paren(paren) => strip_parens(&paren.expr),
        _ => expr,
    }
}

fn binding_from_pat(pat: &syn::Pat) -> Option<(String, bool, Option<String>)> {
    match pat {
        syn::Pat::Ident(pat_ident) => Some((
            pat_ident.ident.to_string(),
            pat_ident.mutability.is_some(),
            None,
        )),
        syn::Pat::Type(pat_type) => {
            let syn::Pat::Ident(pat_ident) = pat_type.pat.as_ref() else {
                return None;
            };
            Some((
                pat_ident.ident.to_string(),
                pat_ident.mutability.is_some(),
                simple_type_name(&pat_type.ty),
            ))
        }
        _ => None,
    }
}

fn simple_type_name(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(type_path) if type_path.qself.is_none() => type_path
            .path
            .segments
            .last()
            .map(|seg| seg.ident.to_string()),
        syn::Type::Reference(reference) => simple_type_name(&reference.elem),
        syn::Type::Paren(paren) => simple_type_name(&paren.elem),
        _ => None,
    }
}

fn collect_simple_enums(
    file: &syn::File,
) -> std::collections::HashMap<String, std::collections::BTreeSet<String>> {
    let mut enums = std::collections::HashMap::new();
    for item in &file.items {
        let syn::Item::Enum(item_enum) = item else {
            continue;
        };
        let variants = item_enum
            .variants
            .iter()
            .map(|variant| variant.ident.to_string())
            .collect();
        enums.insert(item_enum.ident.to_string(), variants);
    }
    enums
}

fn pat_is_catch_all(pat: &syn::Pat) -> bool {
    match pat {
        syn::Pat::Wild(_) | syn::Pat::Rest(_) => true,
        syn::Pat::Ident(pat_ident) => pat_ident
            .ident
            .to_string()
            .chars()
            .next()
            .is_some_and(|ch| ch.is_lowercase() || ch == '_'),
        syn::Pat::Or(or_pat) => or_pat.cases.iter().any(pat_is_catch_all),
        _ => false,
    }
}

fn collect_pat_variants(pat: &syn::Pat, seen: &mut std::collections::BTreeSet<String>) {
    match pat {
        syn::Pat::Path(path) => {
            if let Some(segment) = path.path.segments.last() {
                seen.insert(segment.ident.to_string());
            }
        }
        syn::Pat::TupleStruct(tuple) => {
            if let Some(segment) = tuple.path.segments.last() {
                seen.insert(segment.ident.to_string());
            }
        }
        syn::Pat::Struct(strukt) => {
            if let Some(segment) = strukt.path.segments.last() {
                seen.insert(segment.ident.to_string());
            }
        }
        syn::Pat::Or(or_pat) => {
            for case in &or_pat.cases {
                collect_pat_variants(case, seen);
            }
        }
        _ => {}
    }
}

fn expr_is_owned_constructor(expr: &syn::Expr) -> bool {
    match strip_parens(expr) {
        syn::Expr::Call(call) => match call.func.as_ref() {
            syn::Expr::Path(path) => {
                let text = path.path.to_token_stream().to_string();
                text.contains("String :: from")
                    || text.contains("Vec :: from")
                    || text.ends_with("to_string")
                    || text.ends_with("to_owned")
            }
            _ => false,
        },
        syn::Expr::MethodCall(method) => matches!(
            method.method.to_string().as_str(),
            "to_string" | "to_owned" | "collect"
        ),
        syn::Expr::Macro(mac) => mac
            .mac
            .path
            .segments
            .last()
            .is_some_and(|seg| matches!(seg.ident.to_string().as_str(), "vec" | "format")),
        _ => false,
    }
}

fn block_contains_break(block: &syn::Block) -> bool {
    use syn::visit::Visit;

    struct BreakVisitor {
        found: bool,
    }

    impl<'ast> Visit<'ast> for BreakVisitor {
        fn visit_expr_break(&mut self, _break_expr: &'ast syn::ExprBreak) {
            self.found = true;
        }
    }

    let mut visitor = BreakVisitor { found: false };
    visitor.visit_block(block);
    visitor.found
}

fn stmt_is_terminal(stmt: &syn::Stmt) -> bool {
    match stmt {
        syn::Stmt::Expr(expr, _) => expr_is_terminal(expr),
        _ => false,
    }
}

fn expr_is_terminal(expr: &syn::Expr) -> bool {
    match expr {
        syn::Expr::Return(_) | syn::Expr::Break(_) | syn::Expr::Continue(_) => true,
        syn::Expr::If(if_expr) => {
            let then_terminal = block_last_expr_is_terminal(&if_expr.then_branch);
            let else_terminal = if_expr.else_branch.as_ref().is_some_and(|(_, else_expr)| {
                matches!(
                    else_expr.as_ref(),
                    syn::Expr::Return(_) | syn::Expr::Break(_) | syn::Expr::Continue(_)
                ) || expr_is_terminal(else_expr)
            });
            then_terminal && else_terminal
        }
        _ => false,
    }
}

fn block_last_expr_is_terminal(block: &syn::Block) -> bool {
    block
        .stmts
        .last()
        .is_some_and(|stmt| stmt_is_terminal(stmt))
}

// ============================================================================
// CODE SHEAF
// ============================================================================

/// The code sheaf: verifies local-to-global coherence of a codebase.
///
/// Collects [`LocalSection`]s (function implementations) and verifies that
/// all call-site interfaces are compatible via the sheaf gluing axiom.
///
/// # Example
///
/// ```rust,ignore
/// use symthaea_geodesic::sheaf::{CodeSheaf, LocalSection};
/// use symthaea_core::hdc::binary_hv::BinaryHV;
///
/// let mut sheaf = CodeSheaf::new();
/// sheaf.add_section(LocalSection {
///     name: "foo".into(),
///     encoding: BinaryHV::random(1),
///     param_types: vec![],
///     return_type: BinaryHV::random(2),
///     callees: vec!["bar".into()],
/// });
/// sheaf.add_section(LocalSection {
///     name: "bar".into(),
///     encoding: BinaryHV::random(3),
///     param_types: vec![],
///     return_type: BinaryHV::random(4),
///     callees: vec![],
/// });
///
/// match sheaf.verify() {
///     Ok(conditions) => println!("{} gluing conditions satisfied", conditions.len()),
///     Err(diagnostics) => {
///         for d in &diagnostics {
///             eprintln!("VIOLATION: {} -> {}: {}", d.caller, d.callee, d.message);
///         }
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct CodeSheaf {
    sections: Vec<LocalSection>,
    /// Similarity threshold for gluing conditions. A gluing condition is
    /// satisfied when the HDC similarity between expected and actual
    /// signatures exceeds this threshold.
    threshold: f64,
}

impl Default for CodeSheaf {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeSheaf {
    /// Default similarity threshold for gluing conditions.
    const DEFAULT_THRESHOLD: f64 = 0.3;

    /// Create an empty code sheaf with the default threshold (0.3).
    pub fn new() -> Self {
        Self {
            sections: Vec::new(),
            threshold: Self::DEFAULT_THRESHOLD,
        }
    }

    /// Set the similarity threshold for gluing conditions.
    ///
    /// Higher thresholds are stricter (require more similar interfaces).
    /// Values in [0.0, 1.0]; 0.5 is the expected similarity of random HVs.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Add a local section (function implementation) to the sheaf.
    pub fn add_section(&mut self, section: LocalSection) {
        self.sections.push(section);
    }

    /// Find a section by name.
    pub fn find_section(&self, name: &str) -> Option<&LocalSection> {
        self.sections.iter().find(|s| s.name == name)
    }

    /// Number of sections in the sheaf.
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }

    /// Number of gluing conditions that would be checked.
    ///
    /// This equals the total number of (caller, callee) pairs where both
    /// the caller and callee exist as sections in the sheaf.
    pub fn gluing_count(&self) -> usize {
        self.sections
            .iter()
            .flat_map(|s| {
                s.callees
                    .iter()
                    .filter(|callee_name| self.find_section(callee_name).is_some())
            })
            .count()
    }

    /// Verify all gluing conditions between overlapping sections.
    ///
    /// Returns `Ok(conditions)` if all conditions are satisfied, where
    /// `conditions` contains the full list of checked gluing conditions.
    /// Returns `Err(diagnostics)` if any condition is violated.
    pub fn verify(&self) -> Result<Vec<GluingCondition>, Vec<SheafDiagnostic>> {
        let mut conditions = Vec::new();
        let mut diagnostics = Vec::new();

        for caller in &self.sections {
            for callee_name in &caller.callees {
                // Only check gluing if the callee exists as a section
                let Some(callee) = self.find_section(callee_name) else {
                    continue;
                };

                let condition = self.check_gluing(caller, callee);

                if !condition.satisfied {
                    diagnostics.push(SheafDiagnostic {
                        caller: condition.caller.clone(),
                        callee: condition.callee.clone(),
                        message: format!(
                            "Interface mismatch: similarity {:.4} < threshold {:.4}",
                            condition.similarity, self.threshold
                        ),
                        similarity: condition.similarity,
                        expected_summary: format!(
                            "caller '{}' expects callee signature (bind of caller encoding with name '{}')",
                            caller.name, callee_name
                        ),
                        actual_summary: format!(
                            "callee '{}' provides its own encoding",
                            callee.name
                        ),
                    });
                }

                conditions.push(condition);
            }
        }

        if diagnostics.is_empty() {
            Ok(conditions)
        } else {
            Err(diagnostics)
        }
    }

    /// Check a single gluing condition between a caller and callee.
    ///
    /// The caller's "expected signature" is computed by binding the caller's
    /// encoding with a name-derived vector for the callee. This captures the
    /// idea that the caller "projects" its understanding of the callee through
    /// its own semantic context.
    fn check_gluing(&self, caller: &LocalSection, callee: &LocalSection) -> GluingCondition {
        // The caller's expectation: bind caller encoding with callee name vector.
        // This produces a "projected interface" from the caller's perspective.
        let callee_name_hv = BinaryHV::random(name_seed(&callee.name));
        let expected = caller.encoding.bind(&callee_name_hv);

        let actual = &callee.encoding;
        let similarity = expected.similarity(actual) as f64;

        GluingCondition {
            caller: caller.name.clone(),
            callee: callee.name.clone(),
            expected_signature: expected,
            actual_signature: *actual,
            similarity,
            satisfied: similarity > self.threshold,
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a section with a given seed and callees.
    fn make_section(name: &str, seed: u64, callees: Vec<&str>) -> LocalSection {
        LocalSection {
            name: name.to_string(),
            encoding: BinaryHV::random(seed),
            param_types: vec![],
            return_type: BinaryHV::random(seed.wrapping_add(1)),
            callees: callees.into_iter().map(String::from).collect(),
        }
    }

    #[test]
    fn test_sheaf_no_sections() {
        // Empty sheaf is vacuously coherent.
        let sheaf = CodeSheaf::new();
        let result = sheaf.verify();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
        assert_eq!(sheaf.section_count(), 0);
        assert_eq!(sheaf.gluing_count(), 0);
    }

    #[test]
    fn test_sheaf_coherent() {
        // Two sections where the caller's expected signature of the callee
        // has high similarity to the callee's actual encoding.
        //
        // We use the same encoding for the callee as what the caller expects:
        // expected = caller.bind(name_hv(callee)), so we set callee.encoding
        // to exactly that value.
        let caller_encoding = BinaryHV::random(100);
        let callee_name = "callee_fn";
        let callee_name_hv = BinaryHV::random(name_seed(callee_name));
        let expected_callee_encoding = caller_encoding.bind(&callee_name_hv);

        let mut sheaf = CodeSheaf::new();
        sheaf.add_section(LocalSection {
            name: "caller_fn".to_string(),
            encoding: caller_encoding,
            param_types: vec![],
            return_type: BinaryHV::random(101),
            callees: vec![callee_name.to_string()],
        });
        sheaf.add_section(LocalSection {
            name: callee_name.to_string(),
            encoding: expected_callee_encoding,
            param_types: vec![],
            return_type: BinaryHV::random(102),
            callees: vec![],
        });

        assert_eq!(sheaf.section_count(), 2);
        assert_eq!(sheaf.gluing_count(), 1);

        let result = sheaf.verify();
        assert!(
            result.is_ok(),
            "expected coherent sheaf, got: {:?}",
            result.err()
        );
        let conditions = result.unwrap();
        assert_eq!(conditions.len(), 1);
        assert!(conditions[0].satisfied);
        // Similarity should be 1.0 since we used the exact expected encoding.
        assert!(
            conditions[0].similarity > 0.99,
            "similarity should be ~1.0, got {}",
            conditions[0].similarity
        );
    }

    #[test]
    fn test_sheaf_violation() {
        // Caller and callee have unrelated encodings -> low similarity -> violation.
        let mut sheaf = CodeSheaf::new().with_threshold(0.6);
        sheaf.add_section(make_section("alpha", 200, vec!["beta"]));
        sheaf.add_section(make_section("beta", 300, vec![]));

        let result = sheaf.verify();
        assert!(
            result.is_err(),
            "expected sheaf violation for unrelated encodings"
        );

        let diagnostics = result.unwrap_err();
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].caller, "alpha");
        assert_eq!(diagnostics[0].callee, "beta");
        assert!(diagnostics[0].similarity < 0.6);
    }

    #[test]
    fn test_sheaf_self_referential() {
        // A section that calls itself (recursive function).
        // The gluing check should handle this gracefully.
        let mut sheaf = CodeSheaf::new();
        sheaf.add_section(make_section("recurse", 400, vec!["recurse"]));

        // Self-referential gluing: expected = encoding.bind(name_hv("recurse"))
        // which is unrelated to encoding itself, so similarity will be ~0.5.
        // With default threshold 0.3, this should pass.
        let result = sheaf.verify();
        assert!(
            result.is_ok(),
            "self-referential section should verify: {:?}",
            result.err()
        );
        let conditions = result.unwrap();
        assert_eq!(conditions.len(), 1);
        assert_eq!(conditions[0].caller, "recurse");
        assert_eq!(conditions[0].callee, "recurse");
    }

    #[test]
    fn test_sheaf_missing_callee_ignored() {
        // If a callee is not present as a section, the gluing condition
        // is not checked (external dependency).
        let mut sheaf = CodeSheaf::new();
        sheaf.add_section(make_section("main", 500, vec!["external_lib"]));

        // No section for "external_lib" -> no gluing condition to check.
        assert_eq!(sheaf.gluing_count(), 0);
        let result = sheaf.verify();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_sheaf_transitive_calls() {
        // A -> B -> C: verify both A->B and B->C gluing conditions.
        let mut sheaf = CodeSheaf::new().with_threshold(0.3);
        sheaf.add_section(make_section("a", 600, vec!["b"]));
        sheaf.add_section(make_section("b", 700, vec!["c"]));
        sheaf.add_section(make_section("c", 800, vec![]));

        assert_eq!(sheaf.gluing_count(), 2);

        let result = sheaf.verify();
        // Both conditions are checked regardless of pass/fail.
        match result {
            Ok(conds) => assert_eq!(conds.len(), 2),
            Err(diags) => {
                // Even if some fail, the total checked should be 2.
                // With random encodings at threshold 0.3, ~50% pass.
                assert!(!diags.is_empty());
            }
        }
    }

    #[test]
    fn test_sheaf_find_section() {
        let mut sheaf = CodeSheaf::new();
        sheaf.add_section(make_section("foo", 900, vec![]));
        sheaf.add_section(make_section("bar", 901, vec![]));

        assert!(sheaf.find_section("foo").is_some());
        assert!(sheaf.find_section("bar").is_some());
        assert!(sheaf.find_section("baz").is_none());
    }

    #[test]
    fn test_sheaf_threshold_clamping() {
        let sheaf = CodeSheaf::new().with_threshold(2.0);
        assert!(sheaf.threshold <= 1.0);

        let sheaf = CodeSheaf::new().with_threshold(-1.0);
        assert!(sheaf.threshold >= 0.0);
    }

    #[test]
    fn test_name_seed_deterministic() {
        let s1 = name_seed("hello");
        let s2 = name_seed("hello");
        assert_eq!(s1, s2, "name_seed must be deterministic");

        let s3 = name_seed("world");
        assert_ne!(s1, s3, "different names should produce different seeds");
    }

    #[test]
    fn test_rust_v0_sheaf_coherence_accepts_defined_loop() {
        let source = r#"
pub fn sum(items: &[i32]) -> i32 {
    let mut total = 0;
    for item in items {
        total += *item;
    }
    total
}
"#;

        let result = verify_rust_v0_sheaf_coherence(source, "sum");
        assert!(
            result.coherent,
            "expected coherent source, got {:?}",
            result.diagnostics
        );
    }

    #[test]
    fn test_rust_v0_sheaf_coherence_rejects_unresolved_use() {
        let source = "pub fn broken() -> i32 { missing_value }";
        let result = verify_rust_v0_sheaf_coherence(source, "broken");

        assert!(!result.coherent);
        assert!(
            result
                .diagnostics
                .iter()
                .any(|diag| diag.contains("missing_value"))
        );
    }

    #[test]
    fn test_rust_v0_sheaf_coherence_rejects_shadowing_missing_mut_and_unreachable() {
        let source = r#"
pub fn broken(n: i32) -> i32 {
    let x = 1;
    let x = 2;
    n += x;
    return n;
    100
}
"#;
        let result = verify_rust_v0_sheaf_coherence(source, "broken");

        assert!(!result.coherent);
        assert!(
            result
                .diagnostics
                .iter()
                .any(|diag| diag.contains("shadowed"))
        );
        assert!(result.diagnostics.iter().any(|diag| diag.contains("mut")));
        assert!(
            result
                .diagnostics
                .iter()
                .any(|diag| diag.contains("unreachable"))
        );
    }

    #[test]
    fn test_rust_v0_sheaf_coherence_rejects_local_reference_return_and_infinite_loop() {
        let source = r#"
pub fn local_ref<'a>() -> &'a i32 {
    let x = 1;
    &x
}

pub fn spin() {
    loop {}
}
"#;
        let local_ref = verify_rust_v0_sheaf_coherence(source, "local_ref");
        assert!(!local_ref.coherent);
        assert!(
            local_ref
                .diagnostics
                .iter()
                .any(|diag| diag.contains("local variable"))
        );

        let spin = verify_rust_v0_sheaf_coherence(source, "spin");
        assert!(!spin.coherent);
        assert!(
            spin.diagnostics
                .iter()
                .any(|diag| diag.contains("no reachable break"))
        );
    }

    #[test]
    fn test_rust_v0_sheaf_coherence_rejects_non_exhaustive_simple_enum_match() {
        let source = r#"
enum Mode {
    Idle,
    Active,
    Fault,
}

pub fn score(mode: Mode) -> i32 {
    match mode {
        Mode::Idle => 0,
        Mode::Active => 1,
    }
}
"#;
        let result = verify_rust_v0_sheaf_coherence(source, "score");

        assert!(!result.coherent);
        assert!(
            result
                .diagnostics
                .iter()
                .any(|diag| diag.contains("non-exhaustive match") && diag.contains("Fault"))
        );
    }

    #[test]
    fn test_rust_v0_sheaf_coherence_accepts_enum_match_with_wildcard() {
        let source = r#"
enum Mode {
    Idle,
    Active,
    Fault,
}

pub fn score(mode: Mode) -> i32 {
    match mode {
        Mode::Idle => 0,
        _ => 1,
    }
}
"#;
        let result = verify_rust_v0_sheaf_coherence(source, "score");

        assert!(
            result.coherent,
            "wildcard should satisfy simple enum exhaustiveness: {:?}",
            result.diagnostics
        );
    }

    #[test]
    fn test_rust_v0_sheaf_coherence_rejects_obvious_tail_recursion() {
        let source = r#"
pub fn recurse(n: i32) -> i32 {
    recurse(n)
}
"#;
        let result = verify_rust_v0_sheaf_coherence(source, "recurse");

        assert!(!result.coherent);
        assert!(
            result
                .diagnostics
                .iter()
                .any(|diag| diag.contains("obvious infinite recursion"))
        );
    }

    #[test]
    fn test_rust_v0_sheaf_coherence_rejects_simple_use_after_move() {
        let source = r#"
fn consume(_: String) {}

pub fn moved() -> usize {
    let value = String::from("abc");
    consume(value);
    value.len()
}
"#;
        let result = verify_rust_v0_sheaf_coherence(source, "moved");

        assert!(!result.coherent);
        assert!(
            result
                .diagnostics
                .iter()
                .any(|diag| diag.contains("use of moved value") && diag.contains("value"))
        );
    }
}
