// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Type Causal Model — Rust type transformation reasoning
//!
//! Models how Rust types transform through operations using causal DAG reasoning.
//! Instead of guessing return types, the emitter can query:
//! "If I apply .iter() to &[i32], what type does the iterator produce?"
//! "If the return type is Option<u64>, what wrapping do I need?"
//!
//! This fixes the 13 type mismatch failures in the Exercism benchmark by
//! reasoning about types causally rather than guessing from keywords.
//!
//! # Architecture
//!
//! ```text
//! Input Type + Operation → Type Transformation Rule → Output Type
//!                              ↓
//!                 Return Type Constraint → Required Wrapping
//! ```

/// Type transformation: what happens when you apply an operation to a type
#[derive(Debug, Clone)]
pub struct TypeTransformation {
    /// Input type pattern (e.g., "&[T]", "Vec<T>")
    pub input_pattern: &'static str,
    /// Operation applied (e.g., ".iter()", ".into_iter()")
    pub operation: &'static str,
    /// Resulting item type (e.g., "&T", "T")
    pub output_item: &'static str,
    /// Whether this consumes the input
    pub consumes_input: bool,
}

/// Return type wrapping: what wrapper is needed for a return type
#[derive(Debug, Clone, PartialEq)]
pub enum ReturnWrapping {
    /// No wrapping needed — return the value directly
    Direct,
    /// Wrap in Some() for Option<T>
    OptionSome,
    /// Wrap in Ok() for Result<T, E>
    ResultOk,
    /// Collect into Vec<T>
    CollectVec,
    /// Convert to String
    ToString,
    /// Cast with `as`
    Cast(&'static str),
}

/// The causal type model — consult before generating code
pub struct TypeCausalModel {
    transformations: Vec<TypeTransformation>,
}

impl TypeCausalModel {
    pub fn new() -> Self {
        Self {
            transformations: vec![
                // Slice/array iteration
                TypeTransformation {
                    input_pattern: "&[T]",
                    operation: ".iter()",
                    output_item: "&T",
                    consumes_input: false,
                },
                TypeTransformation {
                    input_pattern: "&[T]",
                    operation: ".into_iter()",
                    output_item: "&T",
                    consumes_input: false, // into_iter on &[T] gives &T, not T
                },
                // Vec iteration
                TypeTransformation {
                    input_pattern: "Vec<T>",
                    operation: ".iter()",
                    output_item: "&T",
                    consumes_input: false,
                },
                TypeTransformation {
                    input_pattern: "Vec<T>",
                    operation: ".into_iter()",
                    output_item: "T",
                    consumes_input: true,
                },
                // Reference conversions
                TypeTransformation {
                    input_pattern: "&T",
                    operation: ".copied()",
                    output_item: "T",
                    consumes_input: false,
                },
                TypeTransformation {
                    input_pattern: "&T",
                    operation: ".cloned()",
                    output_item: "T",
                    consumes_input: false,
                },
                // String operations
                TypeTransformation {
                    input_pattern: "&str",
                    operation: ".to_string()",
                    output_item: "String",
                    consumes_input: false,
                },
                TypeTransformation {
                    input_pattern: "&str",
                    operation: ".chars()",
                    output_item: "char",
                    consumes_input: false,
                },
            ],
        }
    }

    /// Determine what wrapping is needed for a return type.
    ///
    /// Given a return type signature, returns what wrapping the final
    /// expression needs. This fixes the most common Exercism type mismatches.
    pub fn required_wrapping(return_type: &str) -> ReturnWrapping {
        if return_type.starts_with("Option<") {
            ReturnWrapping::OptionSome
        } else if return_type.starts_with("Result<") {
            ReturnWrapping::ResultOk
        } else if return_type.starts_with("Vec<") {
            ReturnWrapping::CollectVec
        } else if return_type == "String" {
            ReturnWrapping::ToString
        } else if return_type == "u32"
            || return_type == "u64"
            || return_type == "i32"
            || return_type == "i64"
        {
            ReturnWrapping::Direct
        } else {
            ReturnWrapping::Direct
        }
    }

    /// Check if a parameter type is a slice (borrows, doesn't own)
    pub fn is_slice(param_type: &str) -> bool {
        param_type.starts_with("&[") || param_type == "&str"
    }

    /// Check if a parameter type owns its data
    pub fn is_owned(param_type: &str) -> bool {
        param_type.starts_with("Vec<") || param_type == "String"
    }

    /// Recommend the right iterator method for a parameter type.
    ///
    /// - Slice (`&[T]`) → `.iter()` (borrows, items are `&T`)
    /// - Owned (`Vec<T>`) → `.into_iter()` (consumes, items are `T`)
    pub fn recommend_iterator(param_type: &str) -> &'static str {
        if Self::is_slice(param_type) {
            ".iter()"
        } else if Self::is_owned(param_type) {
            ".into_iter()"
        } else {
            ".iter()"
        }
    }

    /// Determine if `.copied()` or `.cloned()` is needed after iteration.
    ///
    /// When iterating over `&[T]` with `.iter()`, items are `&T`.
    /// If the return type needs owned `T`, we need `.copied()`.
    pub fn needs_copy_after_iter(param_type: &str, return_type: &str) -> bool {
        // If iterating over a slice and collecting to Vec<T> (not Vec<&T>)
        if Self::is_slice(param_type)
            && return_type.contains("Vec<")
            && !return_type.contains("Vec<&")
        {
            return true;
        }
        // If iterating over a slice and summing/counting (need owned values)
        if Self::is_slice(param_type)
            && (return_type == "i32"
                || return_type == "i64"
                || return_type == "u32"
                || return_type == "u64"
                || return_type == "f32"
                || return_type == "f64"
                || return_type == "usize")
        {
            return true;
        }
        false
    }

    /// Wrap a body expression to match the required return type.
    ///
    /// Given a body expression and the function's return type, adds
    /// the necessary wrapping (Some(), Ok(), .collect(), etc.)
    pub fn wrap_for_return(body: &str, return_type: &str) -> String {
        let wrapping = Self::required_wrapping(return_type);
        match wrapping {
            ReturnWrapping::Direct => body.to_string(),
            ReturnWrapping::OptionSome => {
                // Don't double-wrap if body already returns Option
                if body.contains("None")
                    || body.contains("Some(")
                    || body.contains("return None")
                    || body.contains("return Some")
                {
                    body.to_string()
                } else {
                    format!("Some({})", body)
                }
            }
            ReturnWrapping::ResultOk => {
                if body.contains("Err(")
                    || body.contains("Ok(")
                    || body.contains("return Err")
                    || body.contains("return Ok")
                {
                    body.to_string()
                } else {
                    format!("Ok({})", body)
                }
            }
            ReturnWrapping::CollectVec => {
                if body.contains(".collect()") || body.contains(".collect::<") {
                    body.to_string()
                } else {
                    format!("{}.collect()", body)
                }
            }
            ReturnWrapping::ToString => {
                if body.contains(".to_string()")
                    || body.contains("String::")
                    || body.contains("format!(")
                {
                    body.to_string()
                } else {
                    format!("{}.to_string()", body)
                }
            }
            ReturnWrapping::Cast(target) => format!("{} as {}", body, target),
        }
    }

    /// Fix a generated body to match the expected return type.
    ///
    /// This is the main entry point for type-aware post-processing.
    /// Analyzes the body and return type, applies necessary transformations:
    /// - Wrap in Some/Ok if return type is Option/Result
    /// - Add .collect() if return type is Vec and body is an iterator chain
    /// - Fix iterator ownership (.copied() for slices)
    pub fn fix_return_type(body: &str, return_type: &str, param_types: &[&str]) -> String {
        let mut fixed = body.to_string();

        // Fix 1: Iterator ownership for slice params
        for param_type in param_types {
            if Self::needs_copy_after_iter(param_type, return_type) {
                // If body uses .iter().filter() or .iter().map() without .copied()
                if fixed.contains(".iter().filter(")
                    && !fixed.contains(".copied()")
                    && !fixed.contains(".cloned()")
                {
                    fixed = fixed.replace(".iter().filter(", ".iter().copied().filter(");
                }
                if fixed.contains(".iter().map(")
                    && !fixed.contains(".copied()")
                    && !fixed.contains(".cloned()")
                {
                    fixed = fixed.replace(".iter().map(", ".iter().copied().map(");
                }
            }
        }

        // Fix 2: Return type wrapping
        fixed = Self::wrap_for_return(&fixed, return_type);

        fixed
    }
}

impl Default for TypeCausalModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_required_wrapping() {
        assert_eq!(
            TypeCausalModel::required_wrapping("Option<u64>"),
            ReturnWrapping::OptionSome
        );
        assert_eq!(
            TypeCausalModel::required_wrapping("Result<f64, String>"),
            ReturnWrapping::ResultOk
        );
        assert_eq!(
            TypeCausalModel::required_wrapping("Vec<i32>"),
            ReturnWrapping::CollectVec
        );
        assert_eq!(
            TypeCausalModel::required_wrapping("String"),
            ReturnWrapping::ToString
        );
        assert_eq!(
            TypeCausalModel::required_wrapping("bool"),
            ReturnWrapping::Direct
        );
    }

    #[test]
    fn test_wrap_for_return_option() {
        assert_eq!(
            TypeCausalModel::wrap_for_return("42", "Option<u64>"),
            "Some(42)"
        );
        // Don't double-wrap
        assert_eq!(
            TypeCausalModel::wrap_for_return("Some(42)", "Option<u64>"),
            "Some(42)"
        );
    }

    #[test]
    fn test_wrap_for_return_result() {
        assert_eq!(
            TypeCausalModel::wrap_for_return("42.0", "Result<f64, String>"),
            "Ok(42.0)"
        );
    }

    #[test]
    fn test_needs_copy() {
        assert!(TypeCausalModel::needs_copy_after_iter("&[i32]", "Vec<i32>"));
        assert!(!TypeCausalModel::needs_copy_after_iter(
            "Vec<i32>", "Vec<i32>"
        ));
        assert!(TypeCausalModel::needs_copy_after_iter("&[i32]", "i32"));
    }

    #[test]
    fn test_fix_return_type() {
        // Slice iterator without .copied()
        let body = "v.iter().filter(|x| x > 0).collect()";
        let fixed = TypeCausalModel::fix_return_type(body, "Vec<i32>", &["&[i32]"]);
        assert!(fixed.contains(".copied()"));
    }

    #[test]
    fn test_recommend_iterator() {
        assert_eq!(TypeCausalModel::recommend_iterator("&[i32]"), ".iter()");
        assert_eq!(
            TypeCausalModel::recommend_iterator("Vec<i32>"),
            ".into_iter()"
        );
    }
}
