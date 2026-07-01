// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Program Emitter — convert ProgramNode trees to compilable Rust source code.
//!
//! This is the final step of the resonant explorer pipeline: once the noise-driven
//! search converges on a ProgramNode that matches the desired behaviour, the emitter
//! translates it to human-readable Rust.
//!
//! Operators (ADD, SUB, MUL, etc.) are emitted as infix expressions; function
//! applications use standard call syntax; higher-order combinators map to idiomatic
//! iterator chains.

use symthaea_core::hdc::program_algebra::ProgramNode;

// ═══════════════════════════════════════════════════════════════════════════════
// Operator mapping
// ═══════════════════════════════════════════════════════════════════════════════

/// Map known operator atoms to their Rust infix symbol.
fn op_symbol(name: &str) -> Option<&'static str> {
    match name.to_uppercase().as_str() {
        "ADD" => Some("+"),
        "SUB" => Some("-"),
        "MUL" => Some("*"),
        "DIV" => Some("/"),
        "MOD" => Some("%"),
        "EQ" => Some("=="),
        "LT" => Some("<"),
        "GT" => Some(">"),
        "AND" => Some("&&"),
        "OR" => Some("||"),
        _ => None,
    }
}

/// Check if a name refers to the unary NOT operator.
fn is_not_op(name: &str) -> bool {
    name.eq_ignore_ascii_case("NOT")
}

// ═══════════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════════

/// Emit a complete Rust function from a ProgramNode tree.
///
/// If `signature` is `None`, a default `fn function_name()` signature is used.
///
/// # Example
/// ```ignore
/// let code = emit_rust(&node, "compute", Some("fn compute(a: i64, b: i64) -> i64"));
/// // => "pub fn compute(a: i64, b: i64) -> i64 {\n    a + b\n}"
/// ```
pub fn emit_rust(node: &ProgramNode, function_name: &str, signature: Option<&str>) -> String {
    let sig = match signature {
        Some(s) => format!("pub {s}"),
        None => format!("pub fn {function_name}()"),
    };
    let body = emit_expression(node);
    format!("{sig} {{\n    {body}\n}}")
}

/// Emit a ProgramNode as a Rust expression (no function wrapper).
pub fn emit_expression(node: &ProgramNode) -> String {
    match node {
        ProgramNode::Atom(name) => name.clone(),

        ProgramNode::Typed(name, _ty) => {
            // Emit just the name; type annotations appear in signatures, not expressions.
            name.clone()
        }

        ProgramNode::Apply { func, args } => emit_apply(func, args),

        ProgramNode::Sequence(steps) => steps
            .iter()
            .map(|s| emit_expression(s))
            .collect::<Vec<_>>()
            .join(";\n    "),

        ProgramNode::Branch {
            condition,
            then_branch,
            else_branch,
        } => {
            let cond = emit_expression(condition);
            let then_b = emit_expression(then_branch);
            let else_b = emit_expression(else_branch);
            format!("if {cond} {{ {then_b} }} else {{ {else_b} }}")
        }

        ProgramNode::Map { func, collection } => {
            let f = emit_expression(func);
            let col = emit_expression(collection);
            format!("{col}.iter().map(|x| {f}(x)).collect::<Vec<_>>()")
        }

        ProgramNode::Filter {
            predicate,
            collection,
        } => {
            let pred = emit_expression(predicate);
            let col = emit_expression(collection);
            format!("{col}.iter().filter(|x| {pred}(x)).collect::<Vec<_>>()")
        }

        ProgramNode::Reduce {
            func,
            initial,
            collection,
        } => {
            let f = emit_expression(func);
            let init = emit_expression(initial);
            let col = emit_expression(collection);
            format!("{col}.iter().fold({init}, |acc, x| {f}(acc, x))")
        }

        ProgramNode::Iterate {
            init,
            step,
            condition,
        } => {
            let init_s = emit_expression(init);
            let step_s = emit_expression(step);
            let cond_s = emit_expression(condition);
            format!("{init_s};\n    while {cond_s} {{\n        {step_s};\n    }}")
        }

        ProgramNode::Recurse {
            base_case,
            recursive_step: _,
        } => {
            // The base_case typically contains a Branch that handles both cases.
            emit_expression(base_case)
        }

        ProgramNode::Compose(f, g) => {
            let f_s = emit_expression(f);
            let g_s = emit_expression(g);
            format!("{g_s}({f_s}(x))")
        }

        ProgramNode::Collect(source) => {
            let src = emit_expression(source);
            format!("{src}.collect::<Vec<_>>()")
        }

        ProgramNode::Abstract(_examples) => "todo!(\"abstract pattern\")".to_string(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Emit an Apply node, choosing infix for known operators.
fn emit_apply(func: &ProgramNode, args: &[ProgramNode]) -> String {
    let func_name = match func {
        ProgramNode::Atom(name) => name.as_str(),
        ProgramNode::Typed(name, _) => name.as_str(),
        _ => "",
    };

    // Unary NOT
    if is_not_op(func_name) && args.len() == 1 {
        let a = emit_expression(&args[0]);
        return format!("!{a}");
    }

    // Binary infix operators
    if let Some(sym) = op_symbol(func_name) {
        if args.len() == 2 {
            let lhs = emit_expression(&args[0]);
            let rhs = emit_expression(&args[1]);
            return format!("{lhs} {sym} {rhs}");
        }
    }

    // General function call
    let f = emit_expression(func);
    let arg_strs: Vec<String> = args.iter().map(|a| emit_expression(a)).collect();
    format!("{f}({})", arg_strs.join(", "))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_atom() {
        let node = ProgramNode::atom("x");
        assert_eq!(emit_expression(&node), "x");
    }

    #[test]
    fn test_emit_apply_op() {
        let node = ProgramNode::apply(
            ProgramNode::op("ADD"),
            vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
        );
        assert_eq!(emit_expression(&node), "a + b");
    }

    #[test]
    fn test_emit_apply_func() {
        let node = ProgramNode::apply(ProgramNode::atom("compute"), vec![ProgramNode::atom("a")]);
        assert_eq!(emit_expression(&node), "compute(a)");
    }

    #[test]
    fn test_emit_branch() {
        let node = ProgramNode::branch(
            ProgramNode::atom("cond"),
            ProgramNode::atom("then"),
            ProgramNode::atom("els"),
        );
        assert_eq!(emit_expression(&node), "if cond { then } else { els }");
    }

    #[test]
    fn test_emit_iterate() {
        let node = ProgramNode::iterate(
            ProgramNode::atom("i = 0"),
            ProgramNode::atom("i += 1"),
            ProgramNode::atom("i < n"),
        );
        let code = emit_expression(&node);
        assert!(code.contains("i = 0"), "should contain init");
        assert!(code.contains("while"), "should contain while");
        assert!(code.contains("i += 1"), "should contain step");
    }

    #[test]
    fn test_emit_full_function() {
        let node = ProgramNode::apply(
            ProgramNode::op("ADD"),
            vec![ProgramNode::atom("a"), ProgramNode::atom("b")],
        );
        let code = emit_rust(&node, "add", Some("fn add(a: i64, b: i64) -> i64"));
        assert!(code.contains("pub fn add(a: i64, b: i64) -> i64"));
        assert!(code.contains("a + b"));
    }

    #[test]
    fn test_emit_not_op() {
        let node = ProgramNode::apply(ProgramNode::op("NOT"), vec![ProgramNode::atom("flag")]);
        assert_eq!(emit_expression(&node), "!flag");
    }

    #[test]
    fn test_emit_map() {
        let node = ProgramNode::map(ProgramNode::atom("double"), ProgramNode::atom("items"));
        let code = emit_expression(&node);
        assert!(code.contains("iter().map"), "should use iterator map");
        assert!(code.contains("double"), "should reference the function");
    }

    #[test]
    fn test_emit_reduce() {
        let node = ProgramNode::reduce(
            ProgramNode::op("ADD"),
            ProgramNode::typed("0", "INT"),
            ProgramNode::atom("arr"),
        );
        let code = emit_expression(&node);
        assert!(code.contains("fold"), "should use fold for reduce");
    }
}
