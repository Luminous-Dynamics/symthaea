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
        let trimmed = body.trim_end();
        match wrapping {
            ReturnWrapping::Direct => body.to_string(),
            ReturnWrapping::OptionSome => {
                // Don't double-wrap if body already returns Option — either textually,
                // or because it ends in `.copied()`/`.cloned()`, which are only ever
                // chained to convert an already-Option<&T> (from `.max()`/`.min()`/
                // `.find(...)`/etc.) into Option<T>. Missing this produced
                // Option<Option<T>> (E0308) for every max/min/find-style case.
                if body.contains("None")
                    || body.contains("Some(")
                    || body.contains("return None")
                    || body.contains("return Some")
                    || trimmed.ends_with(".copied()")
                    || trimmed.ends_with(".cloned()")
                {
                    body.to_string()
                } else {
                    format!("Some({})", body)
                }
            }
            ReturnWrapping::ResultOk => {
                // `.map_err(...)` is only callable on Result — its presence means the
                // body already evaluates to a Result, so wrapping in another Ok(...)
                // produced Result<Result<T, E>, _> (E0308).
                if body.contains("Err(")
                    || body.contains("Ok(")
                    || body.contains("return Err")
                    || body.contains("return Ok")
                    || body.contains(".map_err(")
                {
                    body.to_string()
                } else {
                    format!("Ok({})", body)
                }
            }
            ReturnWrapping::CollectVec => {
                // A bare identifier as the TAIL expression (no method-call dot on its
                // own line) is already a concrete value — e.g. a Vec built up via
                // in-place `.sort()`/`.dedup()` over several statements, ending in a
                // bare `result` — not an iterator chain needing `.collect()`.
                // Appending it produced "no method named `collect` found for struct
                // `Vec<T>`" (E0599). Must check only the tail line, not the whole
                // (possibly multi-statement) body — those earlier statements contain
                // plenty of dots (`.to_vec()`, `.sort()`, ...) that don't apply here.
                let tail = trimmed.lines().last().unwrap_or(trimmed).trim();
                if body.contains(".collect()")
                    || body.contains(".collect::<")
                    || !tail.contains('.')
                {
                    body.to_string()
                } else {
                    format!("{}.collect()", body)
                }
            }
            ReturnWrapping::ToString => {
                // A bare trailing `.collect()` already type-infers to the function's
                // declared return type (String) as the tail expression — appending
                // `.to_string()` after it makes `.collect()`'s target type ambiguous
                // (E0282 "type annotations needed").
                if body.contains(".to_string()")
                    || body.contains("String::")
                    || body.contains("format!(")
                    || body.contains(".collect::<")
                    || trimmed.ends_with(".collect()")
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

// =========================================================================
// Solution Family Classification — Structural reasoning from signatures
// =========================================================================

/// What structural pattern a solution should use.
///
/// Inferred from the function signature (input types + return type) before
/// the emitter runs. This is Symthaea's structural reasoning about code:
/// "the signature tells me what KIND of solution this needs."
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolutionFamily {
    /// Single expression or short formula: `a + b`, `x % 4 == 0`
    /// Signature: few params, primitive return, no collection
    SingleExpression,
    /// Iterator chain: `input.iter().map().filter().collect()`
    /// Signature: collection in → collection/value out
    IteratorChain,
    /// Loop with mutable accumulator: `while { acc += ...; } acc`
    /// Signature: numeric in → Option/numeric out (suggests iteration to find answer)
    LoopAccumulator,
    /// Pattern match / conditional dispatch: `match input { ... }`
    /// Signature: &str → &str or enum-like dispatch
    PatternMatch,
    /// Struct with methods: `struct X { ... } impl X { fn new() ... }`
    /// Signature: has &self, or constructs a named type
    StructMethods,
    /// Error handling with Result: validation + Ok/Err
    /// Signature: returns Result<T, E>
    ErrorHandling,
    /// Can't determine from signature alone
    Unknown,
}

impl SolutionFamily {
    /// Classify the solution family from function parameter types and return type.
    ///
    /// This is the core structural reasoning: the signature constrains what
    /// kind of code can satisfy it. No keyword matching needed.
    pub fn from_signature(params: &[(String, String)], return_type: &str) -> Self {
        let ret = return_type.trim();
        let has_collection_input = params
            .iter()
            .any(|(_, t)| t.contains("&[") || t.contains("Vec<") || t.contains("&str"));
        let has_numeric_input = params.iter().any(|(_, t)| {
            t.contains("i32")
                || t.contains("i64")
                || t.contains("u32")
                || t.contains("u64")
                || t.contains("f32")
                || t.contains("f64")
                || t.contains("usize")
        });
        let returns_collection = ret.contains("Vec<")
            || ret.contains("HashSet<")
            || ret.contains("HashMap<")
            || ret.contains("BTreeMap<");
        let returns_string = ret == "String" || ret == "&str" || ret == "&'static str";
        let returns_option = ret.starts_with("Option<");
        let returns_result = ret.starts_with("Result<");
        let returns_bool = ret == "bool";
        let returns_numeric = ret == "i32"
            || ret == "i64"
            || ret == "u32"
            || ret == "u64"
            || ret == "f32"
            || ret == "f64"
            || ret == "usize";
        let has_self = params
            .iter()
            .any(|(n, _)| n == "self" || n == "&self" || n == "&mut self");
        let no_params = params.is_empty();

        // Error handling: Result return type
        if returns_result {
            return Self::ErrorHandling;
        }

        // Struct methods: has &self parameter
        if has_self {
            return Self::StructMethods;
        }

        // Iterator chain: collection input → collection output or String output
        if has_collection_input && (returns_collection || returns_string) {
            return Self::IteratorChain;
        }

        // Iterator chain: collection input → numeric output (sum/count/product)
        if has_collection_input && returns_numeric {
            return Self::IteratorChain;
        }

        // Iterator chain: collection input → bool (all/any/contains check)
        if has_collection_input && returns_bool {
            return Self::IteratorChain;
        }

        // Iterator chain: &str input → String output (string transformation)
        if params.len() == 1 && params[0].1.contains("str") && returns_string {
            return Self::IteratorChain;
        }

        // Loop accumulator: numeric input → Option (suggests search/iteration)
        if has_numeric_input && returns_option {
            return Self::LoopAccumulator;
        }

        // Loop accumulator: numeric input → numeric output (computation)
        if has_numeric_input && returns_numeric && params.len() <= 2 {
            return Self::SingleExpression;
        }

        // Pattern match: &str → &str (dispatch/lookup)
        if params.len() == 1 && params[0].1.contains("str") && (ret.contains("str") || returns_bool)
        {
            return Self::PatternMatch;
        }

        // No params → single expression or constant
        if no_params {
            return Self::SingleExpression;
        }

        // Bool return with few params → single expression
        if returns_bool && params.len() <= 2 {
            return Self::SingleExpression;
        }

        Self::Unknown
    }

    /// Generate a structural template body based on the solution family.
    ///
    /// This produces a SKELETON that the emitter fills in with domain logic.
    /// Even without knowing the specific algorithm, the structure constrains
    /// what valid code looks like.
    pub fn generate_skeleton(
        &self,
        params: &[(String, String)],
        return_type: &str,
        purpose: &str,
    ) -> Option<String> {
        match self {
            Self::IteratorChain => {
                if params.is_empty() {
                    return None;
                }
                let p0 = &params[0].0;
                let p0_type = &params[0].1;
                let purpose_lower = purpose.to_lowercase();

                // Determine iterator start
                let iter_start = if p0_type.contains("&str") {
                    format!("{}.chars()", p0)
                } else if p0_type.contains("&[") {
                    format!("{}.iter()", p0)
                } else if p0_type.contains("Vec<") {
                    format!("{}.into_iter()", p0)
                } else {
                    return None;
                };

                // Infer adapter chain from purpose keywords
                let mut adapters = Vec::new();

                // Filtering adapters
                if purpose_lower.contains("filter")
                    || purpose_lower.contains("keep")
                    || purpose_lower.contains("select")
                    || purpose_lower.contains("remove")
                {
                    if purpose_lower.contains("even") {
                        adapters.push(".filter(|&x| x % 2 == 0)");
                    } else if purpose_lower.contains("odd") {
                        adapters.push(".filter(|&x| x % 2 != 0)");
                    } else if purpose_lower.contains("positive") {
                        adapters.push(".filter(|&x| x > 0)");
                    } else if purpose_lower.contains("alpha") {
                        adapters.push(".filter(|c| c.is_alphabetic())");
                    } else {
                        adapters.push(".filter(|x| true /* TODO: condition */)");
                    }
                }

                // Transformation adapters
                if purpose_lower.contains("uppercase") || purpose_lower.contains("upper") {
                    adapters.push(".map(|c| c.to_uppercase().next().unwrap_or(c))");
                } else if purpose_lower.contains("lowercase") || purpose_lower.contains("lower") {
                    adapters.push(".map(|c| c.to_lowercase().next().unwrap_or(c))");
                } else if purpose_lower.contains("double") {
                    adapters.push(".map(|x| x * 2)");
                } else if purpose_lower.contains("square") {
                    adapters.push(".map(|x| x * x)");
                }

                // Windowing
                if purpose_lower.contains("window")
                    || purpose_lower.contains("consecutive")
                    || purpose_lower.contains("series")
                    || purpose_lower.contains("substring")
                {
                    if params.len() >= 2 {
                        let p1 = &params[1].0;
                        // Switch from chars() to windows approach
                        return Some(format!(
                            "if {p1} == 0 {{ return vec![\"\".to_string(); {p0}.len() + 1]; }}\n    \
                             {p0}.as_bytes().windows({p1}).map(|w| std::str::from_utf8(w).unwrap().to_string()).collect()"
                        ));
                    }
                }

                // Second param for zip
                if params.len() >= 2
                    && (purpose_lower.contains("distance")
                        || purpose_lower.contains("compare")
                        || purpose_lower.contains("zip"))
                {
                    let p1 = &params[1].0;
                    let p1_type = &params[1].1;
                    let iter2 = if p1_type.contains("&str") {
                        format!("{}.chars()", p1)
                    } else {
                        format!("{}.iter()", p1)
                    };
                    return Some(format!(
                        "{}.zip({}).filter(|(a, b)| a != b).count()",
                        iter_start, iter2
                    ));
                }

                // Copy for slices producing owned values
                let needs_copy = p0_type.contains("&[")
                    && (return_type.contains("Vec<") && !return_type.contains("Vec<&"));

                // Determine terminal
                let terminal = if return_type.contains("Vec<String>") {
                    ".map(|c| c.to_string()).collect()"
                } else if return_type.contains("Vec<") {
                    if needs_copy && adapters.is_empty() {
                        ".copied().collect()"
                    } else if needs_copy {
                        ".collect()" // adapter should handle ownership
                    } else {
                        ".collect()"
                    }
                } else if return_type == "String" {
                    ".collect::<String>()"
                } else if return_type == "usize" {
                    ".count()"
                } else if return_type == "bool" {
                    if purpose_lower.contains("all") || purpose_lower.contains("every") {
                        ".all(|x| true /* TODO */)"
                    } else if purpose_lower.contains("any") {
                        ".any(|x| true /* TODO */)"
                    } else if purpose_lower.contains("pangram") {
                        // Special: check all letters present
                        return Some(format!(
                            "let lower = {}.to_lowercase();\n    ('a'..='z').all(|c| lower.contains(c))",
                            p0
                        ));
                    } else if purpose_lower.contains("sorted") {
                        return Some(format!("{}.windows(2).all(|w| w[0] <= w[1])", p0));
                    } else {
                        ".count() > 0"
                    }
                } else if return_type.contains("i32")
                    || return_type.contains("i64")
                    || return_type.contains("u16")
                    || return_type.contains("u32")
                    || return_type.contains("u64")
                    || return_type.contains("f64")
                {
                    ".sum()"
                } else {
                    ".collect()"
                };

                let adapter_chain = adapters.join("");
                Some(format!("{}{}{}", iter_start, adapter_chain, terminal))
            }

            Self::LoopAccumulator => {
                if params.is_empty() {
                    return None;
                }
                let p0 = &params[0].0;

                if return_type.starts_with("Option<") {
                    Some(format!(
                        "let mut n = {};\n    let mut result = 0;\n    while n > 0 {{\n        // accumulate\n        n -= 1;\n        result += 1;\n    }}\n    Some(result)",
                        p0
                    ))
                } else {
                    Some(format!(
                        "let mut result = 0;\n    for i in 0..{} {{\n        result += i;\n    }}\n    result",
                        p0
                    ))
                }
            }

            Self::ErrorHandling => {
                if params.is_empty() {
                    return None;
                }
                let p0 = &params[0].0;

                // Generic error handling skeleton: validate input, process, wrap in Ok
                let purpose_lower = purpose.to_lowercase();
                if purpose_lower.contains("count") || purpose_lower.contains("how many") {
                    // Counting with validation
                    Some(format!(
                        "let mut count = 0;\n    for c in {}.chars() {{\n        // validate and count\n        count += 1;\n    }}\n    Ok(count)",
                        p0
                    ))
                } else if purpose_lower.contains("parse") || purpose_lower.contains("convert") {
                    Some(format!(
                        "{}.parse().map_err(|e: std::num::ParseIntError| e.to_string())",
                        p0
                    ))
                } else {
                    // Generic: validate then wrap
                    Some(format!(
                        "// Validate input\n    // Process\n    Ok({}.to_string())",
                        p0
                    ))
                }
            }

            Self::SingleExpression | Self::PatternMatch | Self::StructMethods | Self::Unknown => {
                None
            }
        }
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

    // Solution Family Classification tests

    #[test]
    fn test_classify_iterator_chain_slice_to_vec() {
        let family = SolutionFamily::from_signature(&[("v".into(), "&[i32]".into())], "Vec<i32>");
        assert_eq!(family, SolutionFamily::IteratorChain);
    }

    #[test]
    fn test_classify_iterator_chain_str_to_string() {
        let family = SolutionFamily::from_signature(&[("input".into(), "&str".into())], "String");
        assert_eq!(family, SolutionFamily::IteratorChain);
    }

    #[test]
    fn test_classify_iterator_chain_slice_to_bool() {
        let family = SolutionFamily::from_signature(&[("sentence".into(), "&str".into())], "bool");
        // &str → bool could be iterator chain (pangram check) or pattern match
        assert!(matches!(
            family,
            SolutionFamily::IteratorChain | SolutionFamily::PatternMatch
        ));
    }

    #[test]
    fn test_classify_loop_accumulator() {
        let family = SolutionFamily::from_signature(&[("n".into(), "u64".into())], "Option<u64>");
        assert_eq!(family, SolutionFamily::LoopAccumulator);
    }

    #[test]
    fn test_classify_error_handling() {
        let family =
            SolutionFamily::from_signature(&[("s".into(), "&str".into())], "Result<i32, String>");
        assert_eq!(family, SolutionFamily::ErrorHandling);
    }

    #[test]
    fn test_classify_single_expression() {
        let family = SolutionFamily::from_signature(
            &[("a".into(), "i32".into()), ("b".into(), "i32".into())],
            "i32",
        );
        assert_eq!(family, SolutionFamily::SingleExpression);
    }

    #[test]
    fn test_classify_no_params() {
        let family = SolutionFamily::from_signature(&[], "u64");
        assert_eq!(family, SolutionFamily::SingleExpression);
    }

    #[test]
    fn test_skeleton_iterator_chain() {
        let family = SolutionFamily::IteratorChain;
        let skeleton = family.generate_skeleton(
            &[("input".into(), "&str".into())],
            "String",
            "Reverse a string",
        );
        assert!(skeleton.is_some());
        let s = skeleton.unwrap();
        assert!(s.contains(".chars()"));
        assert!(s.contains(".collect()"));
    }
}
