// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Lightweight Rust code analysis helpers for Broca evaluation.

/// Functions discovered in a Rust source fragment.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RustFunctionExtraction {
    pub functions: Vec<String>,
    pub parse_error: Option<String>,
}

/// Extract top-level functions, inline module functions, and `impl` methods
/// from a Rust fragment.
///
/// The parser first tries a complete Rust file. If that fails, it still returns
/// best-effort function names found by a conservative lexical fallback so evals
/// can report parse failures without silently dropping cases.
pub fn extract_rust_functions(source: &str) -> RustFunctionExtraction {
    let wrapped = wrap_rust_fragment(source);
    match syn::parse_file(&wrapped) {
        Ok(file) => {
            let mut functions = Vec::new();
            collect_item_functions(&file.items, &mut functions);
            functions.sort();
            functions.dedup();
            RustFunctionExtraction {
                functions,
                parse_error: None,
            }
        }
        Err(error) => RustFunctionExtraction {
            functions: fallback_rust_function_names(source),
            parse_error: Some(error.to_string()),
        },
    }
}

/// Categorize a code-sheaf diagnostic into a stable bucket for dashboards and
/// quality gates.
pub fn categorize_code_sheaf_diagnostic(diagnostic: &str) -> String {
    let lower = diagnostic.to_ascii_lowercase();
    let category = if lower.contains("lifetime") {
        "lifetime_error"
    } else if lower.contains("borrow") {
        "borrow_checker"
    } else if lower.contains("trait bound") || lower.contains("trait") {
        "trait_bound"
    } else if lower.contains("async") || lower.contains("await") {
        "async_mismatch"
    } else if lower.contains("private") || lower.contains("visibility") {
        "visibility_issue"
    } else if lower.contains("parse") || lower.contains("does not parse") {
        "parse_failure"
    } else if lower.contains("implementation stub") {
        "stub"
    } else if lower.contains("missing target function") || lower.contains("was not found") {
        "missing_function"
    } else if lower.contains("non-exhaustive match") {
        "non_exhaustive_match"
    } else if lower.contains("use of moved value") {
        "use_after_move"
    } else if lower.contains("obvious infinite recursion") {
        "infinite_recursion"
    } else if lower.contains("return type") || lower.contains("returns a value") {
        "return_mismatch"
    } else if lower.contains("without a local definition") {
        "unresolved_identifier"
    } else if lower.contains("shadowed") {
        "shadowing"
    } else if lower.contains("requires `mut`") {
        "missing_mut"
    } else if lower.contains("unreachable") {
        "unreachable_code"
    } else {
        "other"
    };
    category.to_string()
}

/// Suggest a stable repair action for a code-sheaf diagnostic category.
///
/// These hints are intentionally broad: they are meant for repair prompts,
/// dashboards, and checkpoint selection metadata, not as compiler replacements.
pub fn repair_hint_for_code_sheaf_category(category: &str) -> Option<&'static str> {
    match category {
        "parse_failure" => Some("regenerate syntactically complete Rust before semantic repair"),
        "stub" => Some("replace placeholder bodies with concrete expressions or control flow"),
        "missing_function" => Some("emit the requested function name and signature exactly"),
        "return_mismatch" => Some("align every return path with the declared return type"),
        "unresolved_identifier" => Some("introduce a local binding or parameter before each use"),
        "shadowing" => Some("rename one binding or reuse the existing binding intentionally"),
        "missing_mut" => Some("mark the binding as mutable or avoid reassignment"),
        "unreachable_code" => Some("remove statements after terminal control flow"),
        "borrow_checker" => Some("avoid aliasing mutable borrows and preserve ownership"),
        "lifetime_error" => Some("return owned values or tie references to input lifetimes"),
        "use_after_move" => Some("borrow, clone, or reorder uses before moving the value"),
        "non_exhaustive_match" => Some("add missing match arms or a wildcard arm"),
        "infinite_recursion" => Some("add a base case or change recursive arguments"),
        "trait_bound" => Some("add the required bound or use operations supported by the type"),
        "async_mismatch" => {
            Some("match async signatures, awaits, and returned future/output types")
        }
        "visibility_issue" => Some("adjust item visibility to match the requested API surface"),
        _ => None,
    }
}

/// Wrap likely expression/block fragments in a synthetic function body while
/// leaving item-like Rust intact.
pub fn wrap_rust_fragment(source: &str) -> String {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return trimmed.to_string();
    }

    if looks_like_rust_item(trimmed) || contains_rust_item(trimmed) {
        trimmed.to_string()
    } else {
        format!("fn __symthaea_eval_fragment__() {{\n{trimmed}\n}}")
    }
}

fn looks_like_rust_item(source: &str) -> bool {
    let source = source.trim_start();
    source.starts_with("fn ")
        || source.starts_with("pub fn ")
        || source.starts_with("async fn ")
        || source.starts_with("pub async fn ")
        || source.starts_with("const fn ")
        || source.starts_with("pub const fn ")
        || source.starts_with("unsafe fn ")
        || source.starts_with("pub unsafe fn ")
        || source.starts_with("impl ")
        || source.starts_with("mod ")
        || source.starts_with("pub mod ")
        || source.starts_with("use ")
        || source.starts_with("pub use ")
        || source.starts_with("struct ")
        || source.starts_with("pub struct ")
        || source.starts_with("enum ")
        || source.starts_with("pub enum ")
        || source.starts_with("trait ")
        || source.starts_with("pub trait ")
        || source.starts_with("#[")
}

fn contains_rust_item(source: &str) -> bool {
    ["fn ", "impl ", "mod ", "struct ", "enum ", "trait "]
        .iter()
        .any(|needle| source.contains(needle))
}

fn collect_item_functions(items: &[syn::Item], functions: &mut Vec<String>) {
    for item in items {
        match item {
            syn::Item::Fn(item_fn) => functions.push(item_fn.sig.ident.to_string()),
            syn::Item::Impl(item_impl) => {
                for impl_item in &item_impl.items {
                    if let syn::ImplItem::Fn(method) = impl_item {
                        functions.push(method.sig.ident.to_string());
                    }
                }
            }
            syn::Item::Mod(item_mod) => {
                if let Some((_, nested_items)) = &item_mod.content {
                    collect_item_functions(nested_items, functions);
                }
            }
            _ => {}
        }
    }
}

fn fallback_rust_function_names(source: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut rest = source;
    while let Some((_, after_fn)) = rest.split_once("fn ") {
        let after_fn = after_fn.trim_start();
        let name: String = after_fn
            .chars()
            .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
            .collect();
        if !name.is_empty() {
            names.push(name);
        }
        rest = after_fn;
    }
    names.sort();
    names.dedup();
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_top_level_mod_and_impl_functions() {
        let source = r#"
            fn top_level() -> bool { true }

            mod nested {
                pub fn inside_mod() -> usize { 1 }
            }

            impl Worker {
                pub fn run(&self) {}
            }
        "#;

        let extracted = extract_rust_functions(source);
        assert_eq!(extracted.parse_error, None);
        assert!(extracted.functions.contains(&"top_level".to_string()));
        assert!(extracted.functions.contains(&"inside_mod".to_string()));
        assert!(extracted.functions.contains(&"run".to_string()));
    }

    #[test]
    fn extracts_async_generic_trait_and_nested_functions() {
        let source = r#"
            pub trait Loader<T> {
                fn load(&self) -> T;
            }

            mod outer {
                pub mod inner {
                    pub async fn fetch<T: Clone>(value: T) -> T { value.clone() }
                }
            }

            impl<T> Loader<T> for Cache<T>
            where
                T: Clone,
            {
                fn load(&self) -> T { self.value.clone() }
            }
        "#;

        let extracted = extract_rust_functions(source);
        assert_eq!(extracted.parse_error, None);
        assert!(extracted.functions.contains(&"fetch".to_string()));
        assert!(extracted.functions.contains(&"load".to_string()));
    }

    #[test]
    fn extracts_names_from_partially_broken_rust() {
        let extracted = extract_rust_functions("fn broken(value: i32) -> i32 { value + }");
        assert!(extracted.parse_error.is_some());
        assert_eq!(extracted.functions, vec!["broken".to_string()]);
    }

    #[test]
    fn wraps_expression_fragments_without_treating_them_as_items() {
        let wrapped = wrap_rust_fragment("items.iter().count()");
        assert!(wrapped.starts_with("fn __symthaea_eval_fragment__()"));
    }

    #[test]
    fn diagnostic_categories_are_stable() {
        assert_eq!(
            categorize_code_sheaf_diagnostic("source does not parse as Rust"),
            "parse_failure"
        );
        assert_eq!(
            categorize_code_sheaf_diagnostic("generated output missing target function `foo`"),
            "missing_function"
        );
        assert_eq!(
            categorize_code_sheaf_diagnostic("use of moved value `value` after move"),
            "use_after_move"
        );
        assert_eq!(
            categorize_code_sheaf_diagnostic("borrow checker rejected mutable alias"),
            "borrow_checker"
        );
    }

    #[test]
    fn repair_hints_cover_actionable_categories() {
        assert_eq!(
            repair_hint_for_code_sheaf_category("missing_mut"),
            Some("mark the binding as mutable or avoid reassignment")
        );
        assert!(repair_hint_for_code_sheaf_category("other").is_none());
    }
}
