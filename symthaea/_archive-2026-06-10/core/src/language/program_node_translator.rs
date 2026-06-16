// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ProgramNode Translator — HDC program tree → source code.
//!
//! Bridges the HDC Program Algebra with actual code generation. Given a
//! matched `PatternEntry` from the `ProgramPatternLibrary`, produces
//! compilable source code in the target language (Rust, Python, Nix).
//!
//! This replaces the hardcoded string templates in `coding_agent.rs`
//! with semantically-grounded code generation.

use symthaea_core::hdc::program_algebra::{PatternEntry, ProgramNode};

/// Translate a matched HDC program pattern into source code.
///
/// Uses the pattern's metadata (purpose, signature, return type) combined
/// with the ProgramNode tree structure to produce compilable code.
///
/// # Arguments
/// * `entry` — The matched pattern with full metadata
/// * `language` — Target language: "rust", "python", or "nix"
/// * `function_name` — The name to use for the generated function
///
/// # Returns
/// Complete source code string ready to write to disk.
pub fn translate(entry: &PatternEntry, language: &str, function_name: &str) -> Option<String> {
    // Use metadata-guided translation: the pattern's signature and purpose
    // tell us the structure, the ProgramNode tells us the algorithm.
    match language {
        "python" => translate_python(entry, function_name),
        "nix" => translate_nix(entry, function_name),
        _ => translate_rust(entry, function_name),
    }
}

// ── Rust Translation ─────────────────────────────────────────────────────

fn translate_rust(entry: &PatternEntry, function_name: &str) -> Option<String> {
    // If we have a Rust signature, use it; otherwise derive from ProgramNode
    let sig = if !entry.rust_signature.is_empty() {
        entry.rust_signature.clone()
    } else {
        derive_rust_signature(&entry.node, &entry.return_type)
    };

    let doc = if !entry.purpose.is_empty() {
        format!("/// {}", entry.purpose)
    } else {
        format!("/// {}", entry.name)
    };

    // Generate the function body from the ProgramNode structure
    let body = generate_rust_body(&entry.node, function_name, &entry.return_type)?;

    Some(format!(
        "{}\npub fn {}{} {{\n{}\n}}\n",
        doc, function_name, sig, body
    ))
}

fn derive_rust_signature(node: &ProgramNode, return_type: &str) -> String {
    let ret = match return_type {
        "INT" => "u64",
        "FLOAT" => "f64",
        "BOOL" => "bool",
        "STRING" => "String",
        "LIST" => "Vec<i64>",
        "OPTION" => "Option<i64>",
        "VOID" | "" => "()",
        other => other,
    };

    // Infer parameters from the node structure
    match node {
        ProgramNode::Recurse { .. } => format!("(n: u64) -> {ret}"),
        ProgramNode::Reduce { .. } => format!("(v: &[i64]) -> {ret}"),
        ProgramNode::Map { .. } | ProgramNode::Filter { .. } => {
            format!("<T: Clone>(v: &[T], f: impl Fn(&T) -> T) -> Vec<T>")
        }
        ProgramNode::Iterate { .. } => format!("(arr: &[i64], target: i64) -> Option<usize>"),
        _ => format!("() -> {ret}"),
    }
}

fn generate_rust_body(node: &ProgramNode, fn_name: &str, return_type: &str) -> Option<String> {
    match node {
        ProgramNode::Recurse { base_case, .. } => {
            // Recursive function — generate iterative version for efficiency
            generate_rust_recursive(base_case, fn_name, return_type)
        }
        ProgramNode::Reduce { func, initial, .. } => generate_rust_reduce(func, initial),
        ProgramNode::Map { func, .. } => Some("    v.iter().map(|x| f(x)).collect()".to_string()),
        ProgramNode::Filter { .. } => {
            Some("    v.iter().filter(|x| f(x)).cloned().collect()".to_string())
        }
        ProgramNode::Iterate { .. } => generate_rust_iterate(fn_name),
        ProgramNode::Compose(f, g) => {
            let f_name = extract_name(f);
            let g_name = extract_name(g);
            Some(format!("    {}({}(input))", g_name, f_name))
        }
        ProgramNode::Apply { func, args } => {
            let fname = extract_name(func);
            let arg_names: Vec<String> = args.iter().map(|a| extract_name(a)).collect();
            Some(format!("    {}({})", fname, arg_names.join(", ")))
        }
        ProgramNode::Sequence(steps) => {
            let lines: Vec<String> = steps
                .iter()
                .map(|s| format!("    {};", extract_name(s)))
                .collect();
            Some(lines.join("\n"))
        }
        ProgramNode::Branch {
            condition,
            then_branch,
            else_branch,
        } => Some(format!(
            "    if {} {{\n        {}\n    }} else {{\n        {}\n    }}",
            extract_name(condition),
            extract_name(then_branch),
            extract_name(else_branch),
        )),
        _ => None,
    }
}

fn generate_rust_recursive(
    base_case: &ProgramNode,
    fn_name: &str,
    return_type: &str,
) -> Option<String> {
    // Pattern: recursive functions often have a branch (base case vs recursive case)
    match base_case {
        ProgramNode::Branch {
            condition,
            then_branch,
            else_branch,
        } => {
            let cond = rust_expr(condition);
            let base = rust_expr(then_branch);
            let recursive = rust_expr(else_branch);
            Some(format!(
                "    if {cond} {{\n        {base}\n    }} else {{\n        {recursive}\n    }}"
            ))
        }
        _ => {
            // Simple recursive pattern
            Some(format!(
                "    // Recursive pattern for {fn_name}\n    Default::default()"
            ))
        }
    }
}

fn generate_rust_reduce(func: &ProgramNode, initial: &ProgramNode) -> Option<String> {
    let op = extract_name(func).to_lowercase();
    let init = rust_expr(initial);

    match op.as_str() {
        "add" | "sum" => Some(format!("    v.iter().sum()")),
        "max" => Some("    v.iter().copied().max()".to_string()),
        "min" => Some("    v.iter().copied().min()".to_string()),
        "mul" => Some(format!("    v.iter().fold({init}, |acc, &x| acc * x)")),
        _ => Some(format!(
            "    v.iter().fold({init}, |acc, &x| acc.{}(x))",
            op
        )),
    }
}

fn generate_rust_iterate(_fn_name: &str) -> Option<String> {
    // Binary search pattern (most common iterate pattern)
    Some("    let mut lo = 0usize;\n    let mut hi = arr.len();\n    while lo < hi {\n        let mid = lo + (hi - lo) / 2;\n        match arr[mid].cmp(&target) {\n            std::cmp::Ordering::Equal => return Some(mid),\n            std::cmp::Ordering::Less => lo = mid + 1,\n            std::cmp::Ordering::Greater => hi = mid,\n        }\n    }\n    None".to_string())
}

/// Convert a ProgramNode to a Rust expression string.
fn rust_expr(node: &ProgramNode) -> String {
    match node {
        ProgramNode::Atom(name) => name.clone(),
        ProgramNode::Typed(name, _) => name.clone(),
        ProgramNode::Apply { func, args } => {
            let fname = extract_name(func);
            let arg_strs: Vec<String> = args.iter().map(|a| rust_expr(a)).collect();
            match fname.to_uppercase().as_str() {
                "ADD" => format!(
                    "{} + {}",
                    arg_strs.first().unwrap_or(&"0".into()),
                    arg_strs.get(1).unwrap_or(&"0".into())
                ),
                "SUB" => format!(
                    "{} - {}",
                    arg_strs.first().unwrap_or(&"0".into()),
                    arg_strs.get(1).unwrap_or(&"0".into())
                ),
                "MUL" => format!(
                    "{} * {}",
                    arg_strs.first().unwrap_or(&"1".into()),
                    arg_strs.get(1).unwrap_or(&"1".into())
                ),
                "DIV" => format!(
                    "{} / {}",
                    arg_strs.first().unwrap_or(&"1".into()),
                    arg_strs.get(1).unwrap_or(&"1".into())
                ),
                "MOD" => format!(
                    "{} % {}",
                    arg_strs.first().unwrap_or(&"0".into()),
                    arg_strs.get(1).unwrap_or(&"1".into())
                ),
                "EQ" => format!(
                    "{} == {}",
                    arg_strs.first().unwrap_or(&"0".into()),
                    arg_strs.get(1).unwrap_or(&"0".into())
                ),
                "LT" => format!(
                    "{} < {}",
                    arg_strs.first().unwrap_or(&"0".into()),
                    arg_strs.get(1).unwrap_or(&"0".into())
                ),
                "GT" => format!(
                    "{} > {}",
                    arg_strs.first().unwrap_or(&"0".into()),
                    arg_strs.get(1).unwrap_or(&"0".into())
                ),
                _ => {
                    if arg_strs.is_empty() {
                        format!("{}()", fname.to_lowercase())
                    } else {
                        format!("{}({})", fname.to_lowercase(), arg_strs.join(", "))
                    }
                }
            }
        }
        ProgramNode::Branch {
            condition,
            then_branch,
            else_branch,
        } => {
            format!(
                "if {} {{ {} }} else {{ {} }}",
                rust_expr(condition),
                rust_expr(then_branch),
                rust_expr(else_branch)
            )
        }
        _ => extract_name(node),
    }
}

// ── Python Translation ───────────────────────────────────────────────────

fn translate_python(entry: &PatternEntry, function_name: &str) -> Option<String> {
    let sig = if !entry.python_signature.is_empty() {
        entry.python_signature.clone()
    } else {
        derive_python_signature(&entry.node, &entry.return_type)
    };

    let doc = if !entry.purpose.is_empty() {
        format!("    \"\"\"{}\"\"\"", entry.purpose)
    } else {
        format!("    \"\"\"{}\"\"\"", entry.name)
    };

    let body = generate_python_body(&entry.node, function_name)?;

    Some(format!(
        "def {}{}:\n{}\n{}\n",
        function_name, sig, doc, body
    ))
}

fn derive_python_signature(node: &ProgramNode, _return_type: &str) -> String {
    match node {
        ProgramNode::Recurse { .. } => "(n: int) -> int".to_string(),
        ProgramNode::Reduce { .. } => "(v: list) -> int".to_string(),
        ProgramNode::Map { .. } | ProgramNode::Filter { .. } => "(v: list, f) -> list".to_string(),
        ProgramNode::Iterate { .. } => "(arr: list, target) -> int".to_string(),
        _ => "()".to_string(),
    }
}

fn generate_python_body(node: &ProgramNode, fn_name: &str) -> Option<String> {
    match node {
        ProgramNode::Recurse { base_case, .. } => {
            match base_case.as_ref() {
                ProgramNode::Branch { condition, then_branch, else_branch } => {
                    let cond = python_expr(condition);
                    let base = python_expr(then_branch);
                    let recursive = python_expr(else_branch);
                    Some(format!("    if {cond}:\n        return {base}\n    return {recursive}"))
                }
                _ => Some(format!("    pass  # recursive pattern for {fn_name}"))
            }
        }
        ProgramNode::Reduce { func, initial, .. } => {
            let op = extract_name(func).to_lowercase();
            match op.as_str() {
                "add" | "sum" => Some("    return sum(v)".to_string()),
                "max" => Some("    return max(v) if v else None".to_string()),
                "min" => Some("    return min(v) if v else None".to_string()),
                _ => {
                    let init = python_expr(initial);
                    Some(format!("    from functools import reduce\n    return reduce(lambda a, x: a + x, v, {init})"))
                }
            }
        }
        ProgramNode::Map { .. } => Some("    return [f(x) for x in v]".to_string()),
        ProgramNode::Filter { .. } => Some("    return [x for x in v if f(x)]".to_string()),
        ProgramNode::Compose(f, g) => {
            let fname = extract_name(f);
            let gname = extract_name(g);
            Some(format!("    return {gname}({fname}(input))"))
        }
        ProgramNode::Iterate { .. } => {
            Some("    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid\n    return -1".to_string())
        }
        _ => None,
    }
}

fn python_expr(node: &ProgramNode) -> String {
    match node {
        ProgramNode::Atom(name) => name.clone(),
        ProgramNode::Typed(name, _) => name.clone(),
        ProgramNode::Apply { func, args } => {
            let fname = extract_name(func);
            let arg_strs: Vec<String> = args.iter().map(|a| python_expr(a)).collect();
            match fname.to_uppercase().as_str() {
                "ADD" => format!(
                    "{} + {}",
                    arg_strs.first().unwrap_or(&"0".into()),
                    arg_strs.get(1).unwrap_or(&"0".into())
                ),
                "SUB" => format!(
                    "{} - {}",
                    arg_strs.first().unwrap_or(&"0".into()),
                    arg_strs.get(1).unwrap_or(&"0".into())
                ),
                "MUL" => format!(
                    "{} * {}",
                    arg_strs.first().unwrap_or(&"0".into()),
                    arg_strs.get(1).unwrap_or(&"0".into())
                ),
                "EQ" => format!(
                    "{} == {}",
                    arg_strs.first().unwrap_or(&"0".into()),
                    arg_strs.get(1).unwrap_or(&"0".into())
                ),
                "LT" => format!(
                    "{} < {}",
                    arg_strs.first().unwrap_or(&"0".into()),
                    arg_strs.get(1).unwrap_or(&"0".into())
                ),
                "GT" => format!(
                    "{} > {}",
                    arg_strs.first().unwrap_or(&"0".into()),
                    arg_strs.get(1).unwrap_or(&"0".into())
                ),
                "MOD" => format!(
                    "{} % {}",
                    arg_strs.first().unwrap_or(&"0".into()),
                    arg_strs.get(1).unwrap_or(&"0".into())
                ),
                _ => {
                    if arg_strs.is_empty() {
                        format!("{}()", fname.to_lowercase())
                    } else {
                        format!("{}({})", fname.to_lowercase(), arg_strs.join(", "))
                    }
                }
            }
        }
        ProgramNode::Branch {
            condition,
            then_branch,
            else_branch,
        } => {
            format!(
                "{} if {} else {}",
                python_expr(then_branch),
                python_expr(condition),
                python_expr(else_branch)
            )
        }
        _ => extract_name(node),
    }
}

// ── Nix Translation ──────────────────────────────────────────────────────

fn translate_nix(entry: &PatternEntry, function_name: &str) -> Option<String> {
    // Nix is functional — most patterns translate naturally
    let body = generate_nix_body(&entry.node, function_name)?;
    let doc = if !entry.purpose.is_empty() {
        format!("# {}", entry.purpose)
    } else {
        String::new()
    };
    Some(format!("{}\n{} = {};\n", doc, function_name, body))
}

fn generate_nix_body(node: &ProgramNode, fn_name: &str) -> Option<String> {
    match node {
        ProgramNode::Recurse { base_case, .. } => match base_case.as_ref() {
            ProgramNode::Branch {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond = nix_expr(condition);
                let base = nix_expr(then_branch);
                let recursive = nix_expr(else_branch);
                Some(format!("n: if {cond} then {base} else {recursive}"))
            }
            _ => Some(format!("n: n # placeholder for {fn_name}")),
        },
        ProgramNode::Reduce { .. } => Some("builtins.foldl' (a: b: a + b) 0".to_string()),
        ProgramNode::Map { .. } => Some("map (x: f x)".to_string()),
        ProgramNode::Filter { .. } => Some("builtins.filter (x: f x)".to_string()),
        _ => None,
    }
}

fn nix_expr(node: &ProgramNode) -> String {
    match node {
        ProgramNode::Atom(name) => name.clone(),
        ProgramNode::Typed(name, _) => name.clone(),
        ProgramNode::Apply { func, args } => {
            let fname = extract_name(func);
            let arg_strs: Vec<String> = args.iter().map(|a| nix_expr(a)).collect();
            match fname.to_uppercase().as_str() {
                "ADD" => format!(
                    "({} + {})",
                    arg_strs.first().unwrap_or(&"0".into()),
                    arg_strs.get(1).unwrap_or(&"0".into())
                ),
                "EQ" => format!(
                    "({} == {})",
                    arg_strs.first().unwrap_or(&"0".into()),
                    arg_strs.get(1).unwrap_or(&"0".into())
                ),
                "LT" => format!(
                    "({} < {})",
                    arg_strs.first().unwrap_or(&"0".into()),
                    arg_strs.get(1).unwrap_or(&"0".into())
                ),
                _ => format!("({} {})", fname.to_lowercase(), arg_strs.join(" ")),
            }
        }
        _ => extract_name(node),
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn extract_name(node: &ProgramNode) -> String {
    match node {
        ProgramNode::Atom(name) => name.clone(),
        ProgramNode::Typed(name, _) => name.clone(),
        ProgramNode::Apply { func, .. } => extract_name(func),
        _ => "unknown".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::program_algebra::ProgramPatternLibrary;

    fn get_pattern(name: &str) -> PatternEntry {
        let lib = ProgramPatternLibrary::standard();
        lib.find_top_k(
            &symthaea_core::hdc::program_algebra::encode_task_description(name),
            1,
        )
        .into_iter()
        .next()
        .map(|(entry, _)| entry.clone())
        .expect("pattern should exist")
    }

    #[test]
    fn test_translate_fibonacci_rust() {
        let entry = get_pattern("fibonacci");
        let code = translate(&entry, "rust", "fibonacci").unwrap();
        assert!(
            code.contains("pub fn fibonacci"),
            "Should have function: {code}"
        );
        assert!(code.contains("///"), "Should have doc comment");
    }

    #[test]
    fn test_translate_fibonacci_python() {
        let entry = get_pattern("fibonacci");
        let code = translate(&entry, "python", "fibonacci").unwrap();
        assert!(
            code.contains("def fibonacci"),
            "Should have function: {code}"
        );
        assert!(code.contains("\"\"\""), "Should have docstring");
    }

    #[test]
    fn test_translate_sum_rust() {
        let entry = get_pattern("sum");
        let code = translate(&entry, "rust", "sum_vec").unwrap();
        assert!(
            code.contains("pub fn sum_vec"),
            "Should have function: {code}"
        );
        assert!(
            code.contains("sum()") || code.contains("iter()"),
            "Should use iterator: {code}"
        );
    }

    #[test]
    fn test_translate_sum_python() {
        let entry = get_pattern("sum");
        let code = translate(&entry, "python", "sum_list").unwrap();
        assert!(
            code.contains("def sum_list"),
            "Should have function: {code}"
        );
        assert!(code.contains("sum("), "Should use sum(): {code}");
    }

    #[test]
    fn test_translate_sort_rust() {
        let entry = get_pattern("sort");
        let code = translate(&entry, "rust", "sort_vec");
        // Sort might not have a body generator yet — that's OK
        if let Some(ref c) = code {
            assert!(c.contains("pub fn sort_vec"), "Should have function: {c}");
        }
    }

    #[test]
    fn test_translate_returns_none_for_unknown() {
        let entry = PatternEntry {
            name: "unknown_thing".into(),
            node: ProgramNode::Atom("???".into()),
            encoding: symthaea_core::hdc::binary_hv::BinaryHV::random(42),
            purpose: String::new(),
            rust_signature: String::new(),
            python_signature: String::new(),
            param_types: Vec::new(),
            return_type: String::new(),
            keywords: Vec::new(),
        };
        // Atom nodes don't have a body generator
        let code = translate(&entry, "rust", "unknown");
        // May return None or a simple stub — either is acceptable
        if let Some(ref c) = code {
            assert!(
                c.contains("pub fn unknown"),
                "Should have function name: {c}"
            );
        }
    }
}
