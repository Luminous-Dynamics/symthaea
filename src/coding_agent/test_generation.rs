// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Example and Invocation-Smoke Test Generation
//!
//! Generates tests from caller-supplied input/output examples and a narrow set of
//! type-directed invocation smoke probes. Smoke probes demonstrate only that a
//! constructed call can be formed; they do not establish behavioral correctness.
//! Unsupported inputs and missing behavioral oracles are reported explicitly.
//!
//! ```text
//! Function signature: fn fibonacci(n: u64) -> u64
//! Examples: fibonacci(0)=0, fibonacci(10)=55
//!     ↓
//! TestSuite {
//!     tests: [
//!         Test { name: "test_fibonacci_example_0", assert: "fibonacci(0)", expected: "0" },
//!         Test { name: "test_fibonacci_example_1", assert: "fibonacci(10)", expected: "55" },
//!         Test { name: "test_fibonacci_boundary_zero", assert: "fibonacci(0)", expected: "0" },
//!         Test { name: "test_fibonacci_boundary_one", assert: "fibonacci(1)", expected: "1" },
//!         Test { name: "test_fibonacci_large", assert: "fibonacci(20)", expected: "6765" },
//!     ]
//! }
//! ```
//!

use symthaea_core::synthesis_trait::SynthesisRequest;
use quote::ToTokens;
use syn::{FnArg, Pat, Type};

/// A generated test case.
#[derive(Debug, Clone)]
pub struct GeneratedTest {
    /// Test function name (e.g., "test_fibonacci_example_0")
    pub name: String,
    /// The assertion expression (e.g., "assert_eq!(fibonacci(0), 0)")
    pub assertion: String,
    /// Category of this test
    pub category: TestCategory,
}

/// Category of generated test — informs which testing patterns to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestCategory {
    /// Explicit expected input/output example supplied by the request author.
    Example,
    /// A call was formed for a selected boundary input; no output oracle is implied.
    InvocationSmoke,
    /// Reserved for properties justified by an independent behavioral contract.
    Property,
    /// Reserved until an error contract and valid inputs are supplied.
    ErrorCase,
    /// Reserved for checks whose expected behavior is backed by an explicit task specification.
    SpecificationBacked,
}

/// A complete test suite generated for a function.
#[derive(Debug, Clone)]
pub struct GeneratedTestSuite {
    /// The function being tested
    pub function_name: String,
    /// Generated test cases
    pub tests: Vec<GeneratedTest>,
    /// The complete Rust test module source code
    pub source: String,
    /// Checks not generated because their inputs or oracle were not established.
    pub unsupported_reasons: Vec<String>,
}

/// Generates test suites from function specifications.
pub struct TestGenerator;

impl TestGenerator {
    /// Generate tests from explicit examples and safe invocation-only smoke probes.
    /// A return type alone does not define a behavioral property or error oracle.
    pub fn generate(request: &SynthesisRequest) -> GeneratedTestSuite {
        let mut tests = Vec::new();
        let mut unsupported_reasons = Vec::new();

        // Explicit examples carry the expected-value oracle in this API.
        for (i, (input, expected)) in request.examples.iter().enumerate() {
            tests.push(GeneratedTest {
                name: format!("test_{}_example_{}", request.name, i),
                assertion: format!("assert_eq!({}, {});", input, expected),
                category: TestCategory::Example,
            });
        }

        if let Some(signature) = request.signature.as_deref() {
            match Self::parse_signature(&request.name, signature) {
                Ok(parsed) => {
                    let (boundary_tests, boundary_reasons) =
                        Self::generate_boundary_tests(&request.name, &parsed.parameters);
                    tests.extend(boundary_tests);
                    for reason in boundary_reasons {
                        Self::push_unique_reason(&mut unsupported_reasons, reason);
                    }

                    if matches!(
                        parsed.return_type.as_ref(),
                        Some(Type::Path(path))
                            if path.path.segments.last().is_some_and(|segment| {
                                matches!(segment.ident.to_string().as_str(), "bool" | "Option" | "Result")
                            })
                    ) {
                        Self::push_unique_reason(
                            &mut unsupported_reasons,
                            "No property/error oracle was generated from the return type alone. Supply explicit expected examples or an independently sourced behavioral contract.".to_string(),
                        );
                    }
                }
                Err(reason) => Self::push_unique_reason(&mut unsupported_reasons, reason),
            }
        } else if request.examples.is_empty() {
            Self::push_unique_reason(
                &mut unsupported_reasons,
                "No Rust signature or explicit input/output examples were provided; no tests were generated.".to_string(),
            );
        }

        let source = Self::render_test_module(&request.name, &tests);
        GeneratedTestSuite {
            function_name: request.name.clone(),
            tests,
            source,
            unsupported_reasons,
        }
    }

    /// Parse the declaration with Rust's AST rather than delimiter/string splitting.
    /// Generic/method/async/unsafe/extern/variadic contexts are deliberately unsupported.
    fn parse_signature(fn_name: &str, signature: &str) -> Result<ParsedSignature, String> {
        let declaration = signature
            .trim()
            .trim_end_matches(';')
            .trim()
            .trim_end_matches('{')
            .trim();
        if declaration.is_empty() {
            return Err("Signature was empty; no signature-derived tests were generated.".into());
        }

        let source = format!("{declaration} {{}}");
        let item = syn::parse_str::<syn::ItemFn>(&source).map_err(|error| {
            format!("Signature could not be parsed as a Rust function: {error}; no signature-derived tests were generated.")
        })?;

        if item.sig.ident.to_string() != fn_name {
            return Err(format!(
                "Signature names function {}, but request names {}; no signature-derived tests were generated.",
                item.sig.ident, fn_name
            ));
        }
        if !item.sig.generics.params.is_empty() || item.sig.generics.where_clause.is_some() {
            return Err("Generic function signatures require caller-supplied typed examples; no guessed generic arguments were generated.".into());
        }
        if item.sig.asyncness.is_some()
            || item.sig.unsafety.is_some()
            || item.sig.abi.is_some()
            || item.sig.variadic.is_some()
        {
            return Err("Async, unsafe, extern or variadic signatures require a specialized invocation harness; no signature-derived tests were generated.".into());
        }

        let mut parameters = Vec::new();
        for input in &item.sig.inputs {
            match input {
                FnArg::Receiver(_) => {
                    return Err("Method receivers require an instance construction contract; no signature-derived tests were generated.".into());
                }
                FnArg::Typed(argument) => {
                    let Pat::Ident(pattern) = argument.pat.as_ref() else {
                        return Err("Destructured parameter patterns require explicit typed examples; no signature-derived tests were generated.".into());
                    };
                    if pattern.by_ref.is_some() || pattern.subpat.is_some() {
                        return Err("ref/subpattern parameters require an explicit invocation contract; no signature-derived tests were generated.".into());
                    }
                    parameters.push(ParsedParameter {
                        name: pattern.ident.to_string(),
                        ty: argument.ty.as_ref().clone(),
                    });
                }
            }
        }

        let return_type = match &item.sig.output {
            syn::ReturnType::Default => None,
            syn::ReturnType::Type(_, ty) => Some(ty.as_ref().clone()),
        };
        Ok(ParsedSignature { parameters, return_type })
    }

    /// Generate invocation-only smoke probes. These do not assert semantic correctness.
    fn generate_boundary_tests(
        fn_name: &str,
        params: &[ParsedParameter],
    ) -> (Vec<GeneratedTest>, Vec<String>) {
        let mut tests = Vec::new();
        let mut unsupported_reasons = Vec::new();

        for parameter in params {
            let Some(boundaries) = Self::boundary_values_for_type(&parameter.ty) else {
                Self::push_unique_reason(
                    &mut unsupported_reasons,
                    format!(
                        "No safe boundary constructor is defined for parameter {} of type {}; no guessed Default::default() call was emitted.",
                        parameter.name,
                        Self::type_name(&parameter.ty)
                    ),
                );
                continue;
            };

            for (suffix, value) in boundaries {
                let Some(arguments) = Self::make_call_with_value(params, &parameter.name, &value)
                else {
                    Self::push_unique_reason(
                        &mut unsupported_reasons,
                        format!(
                            "Boundary probe for parameter {} was skipped because at least one other parameter has no safe constructor.",
                            parameter.name
                        ),
                    );
                    continue;
                };

                tests.push(GeneratedTest {
                    name: format!("test_{fn_name}_boundary_{}_{}", parameter.name, suffix),
                    assertion: format!("let _ = {fn_name}({arguments});"),
                    category: TestCategory::InvocationSmoke,
                });
            }
        }
        (tests, unsupported_reasons)
    }

    /// Construct a call only when every other argument has a known type-directed expression.
    fn make_call_with_value(
        params: &[ParsedParameter],
        target_param: &str,
        value: &str,
    ) -> Option<String> {
        params
            .iter()
            .map(|parameter| {
                if parameter.name == target_param {
                    Some(value.to_string())
                } else {
                    Self::default_value_for_type(&parameter.ty)
                }
            })
            .collect::<Option<Vec<_>>>()
            .map(|arguments| arguments.join(", "))
    }

    /// Conservative constructors for a small explicit set of Rust input types.
    fn default_value_for_type(ty: &Type) -> Option<String> {
        match ty {
            Type::Reference(reference) if reference.mutability.is_none() => {
                match reference.elem.as_ref() {
                    Type::Path(path) if path.path.is_ident("str") => Some("\"\"".into()),
                    Type::Path(path) if path.path.segments.last().is_some_and(|segment| segment.ident.to_string() == "String") => Some("&String::new()".into()),
                    Type::Slice(_) => Some("&[]".into()),
                    _ => None,
                }
            }
            Type::Path(path) if path.qself.is_none() => {
                let ident = path.path.segments.last()?.ident.to_string();
                match ident.as_str() {
                    "u8" | "u16" | "u32" | "u64" | "u128" | "usize"
                    | "i8" | "i16" | "i32" | "i64" | "i128" | "isize" => Some("0".into()),
                    "f32" => Some("0.0f32".into()),
                    "f64" => Some("0.0f64".into()),
                    "bool" => Some("false".into()),
                    "char" => Some("'\\0'".into()),
                    "String" => Some("String::new()".into()),
                    "Vec" => Some("Vec::new()".into()),
                    "Option" => Some("None".into()),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Representative input expressions for a target parameter.
    fn boundary_values_for_type(ty: &Type) -> Option<Vec<(&'static str, String)>> {
        let values = match ty {
            Type::Reference(reference) if reference.mutability.is_none() => {
                match reference.elem.as_ref() {
                    Type::Path(path) if path.path.is_ident("str") => vec![("empty", "\"\"".into())],
                    Type::Path(path) if path.path.segments.last().is_some_and(|segment| segment.ident.to_string() == "String") => vec![("empty", "&String::new()".into())],
                    Type::Slice(_) => vec![("empty", "&[]".into())],
                    _ => return None,
                }
            }
            Type::Path(path) if path.qself.is_none() => {
                let ident = path.path.segments.last()?.ident.to_string();
                match ident.as_str() {
                    "u8" | "u16" | "u32" | "u64" | "u128" | "usize" => vec![("zero", "0".into()), ("one", "1".into())],
                    "i8" | "i16" | "i32" | "i64" | "i128" | "isize" => vec![("negative", "-1".into()), ("zero", "0".into())],
                    "f32" => vec![("zero", "0.0f32".into())],
                    "f64" => vec![("zero", "0.0f64".into())],
                    "bool" => vec![("false", "false".into()), ("true", "true".into())],
                    "char" => vec![("null", "'\\0'".into()), ("letter", "'x'".into())],
                    "String" => vec![("empty", "String::new()".into())],
                    "Vec" => vec![("empty", "Vec::new()".into())],
                    "Option" => vec![("none", "None".into())],
                    _ => return None,
                }
            }
            _ => return None,
        };
        Some(values)
    }

    fn type_name(ty: &Type) -> String {
        ty.to_token_stream().to_string()
    }

    fn push_unique_reason(reasons: &mut Vec<String>, reason: String) {
        if !reasons.contains(&reason) {
            reasons.push(reason);
        }
    }

    /// Render tests into a Rust `#[cfg(test)] mod tests` block.
    fn render_test_module(fn_name: &str, tests: &[GeneratedTest]) -> String {
        if tests.is_empty() {
            return format!("// No tests generated for {fn_name}\n");
        }

        let mut source = String::new();
        source.push_str("#[cfg(test)]\n");
        source.push_str("mod tests {\n");
        source.push_str("    use super::*;\n\n");

        for test in tests {
            source.push_str("    #[test]\n");
            source.push_str(&format!("    fn {}() {{\n", test.name));
            source.push_str(&format!("        {}\n", test.assertion));
            source.push_str("    }\n\n");
        }

        source.push_str("}\n");
        source
    }

}

#[derive(Debug, Clone)]
struct ParsedParameter {
    name: String,
    ty: Type,
}

#[derive(Debug, Clone)]
struct ParsedSignature {
    parameters: Vec<ParsedParameter>,
    return_type: Option<Type>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_from_examples() {
        let request = SynthesisRequest::new("rust", "add", "Add two integers")
            .with_signature("fn add(a: i32, b: i32) -> i32")
            .with_example("add(2, 3)", "5")
            .with_example("add(-1, 1)", "0");

        let suite = TestGenerator::generate(&request);

        assert_eq!(suite.function_name, "add");
        assert!(suite.tests.len() >= 2); // At least the 2 examples
        assert!(suite.source.contains("#[cfg(test)]"));
        assert!(suite.source.contains("test_add_example_0"));
        assert!(suite.source.contains("assert_eq!(add(2, 3), 5)"));
    }

    #[test]
    fn test_generate_boundary_unsigned() {
        let request = SynthesisRequest::new("rust", "factorial", "Compute factorial")
            .with_signature("fn factorial(n: u64) -> u64");

        let suite = TestGenerator::generate(&request);

        let boundary_tests: Vec<_> = suite
            .tests
            .iter()
            .filter(|t| t.category == TestCategory::InvocationSmoke)
            .collect();

        assert!(
            !boundary_tests.is_empty(),
            "Should generate boundary tests for u64 param"
        );
        assert!(suite.source.contains("boundary_n_zero"));
    }

    #[test]
    fn test_generate_boundary_signed() {
        let request = SynthesisRequest::new("rust", "abs", "Absolute value")
            .with_signature("fn abs(x: i32) -> i32");

        let suite = TestGenerator::generate(&request);

        let boundary_tests: Vec<_> = suite
            .tests
            .iter()
            .filter(|t| t.category == TestCategory::Boundary)
            .collect();

        assert!(boundary_tests.len() >= 2); // At least negative + zero
        assert!(suite.source.contains("boundary_x_negative"));
    }

    #[test]
    fn test_generate_boundary_string() {
        let request = SynthesisRequest::new("rust", "reverse", "Reverse a string")
            .with_signature("fn reverse(s: &str) -> String");

        let suite = TestGenerator::generate(&request);

        assert!(suite.source.contains("boundary_s_empty"));
    }

    #[test]
    fn test_generate_boundary_vec() {
        let request = SynthesisRequest::new("rust", "sum", "Sum a vector")
            .with_signature("fn sum(v: &[i32]) -> i32");

        let suite = TestGenerator::generate(&request);

        assert!(suite.source.contains("boundary_v_empty"));
    }

    #[test]
    fn test_signature_parser_uses_rust_ast_for_nested_types() {
        let parsed = TestGenerator::parse_signature(
            "find",
            "fn find(haystack: &[u8], needle: Vec<(u16, Option<u8>)>) -> Option<usize>",
        )
        .unwrap();

        assert_eq!(parsed.parameters.len(), 2);
        assert_eq!(parsed.parameters[0].name, "haystack");
        assert!(matches!(
            &parsed.parameters[0].ty,
            Type::Reference(reference) if matches!(reference.elem.as_ref(), Type::Slice(_))
        ));
        assert_eq!(parsed.parameters[1].name, "needle");
        assert!(matches!(
            &parsed.parameters[1].ty,
            Type::Path(path) if path.path.segments.last().is_some_and(|segment| segment.ident.to_string() == "Vec")
        ));
        assert!(matches!(parsed.return_type, Some(Type::Path(_))));
    }

    #[test]
    fn test_signature_parser_rejects_generic_and_receiver_contexts() {
        assert!(TestGenerator::parse_signature("first", "fn first<T>(items: Vec<T>) -> Option<T>").is_err());
        assert!(TestGenerator::parse_signature("ready", "fn ready(&self) -> bool").is_err());
    }

    #[test]
    fn test_property_like_return_types_do_not_invent_oracles() {
        let cases = [
            ("is_ready", "fn is_ready(value: NonDefault) -> bool"),
            ("parse", "fn parse(input: NonDefault) -> Result<u8, String>"),
            ("first", "fn first(input: NonDefault) -> Option<u8>"),
        ];

        for (name, signature) in cases {
            let request = SynthesisRequest::new("rust", name, "Check a value")
                .with_signature(signature);
            let suite = TestGenerator::generate(&request);

            assert!(suite.tests.is_empty(), "{name} must not receive a guessed one-argument property test");
            assert!(!suite.source.contains("Default::default()"));
            assert!(
                suite.unsupported_reasons.iter().any(|reason| reason.contains("No safe boundary constructor"))
                    || suite.unsupported_reasons.iter().any(|reason| reason.contains("No property/error oracle")),
                "unsupported construction/oracle must be explicit: {:?}",
                suite.unsupported_reasons
            );
        }
    }

    #[test]
    fn test_multi_parameter_smoke_calls_preserve_arity() {
        let request = SynthesisRequest::new("rust", "clamp", "Clamp a value")
            .with_signature("fn clamp(value: i32, min: i32, max: i32) -> i32");
        let suite = TestGenerator::generate(&request);

        assert_eq!(suite.tests.len(), 6);
        for test in &suite.tests {
            assert_eq!(test.category, TestCategory::InvocationSmoke);
            let args = test.assertion.split_once("clamp(").unwrap().1.trim_end_matches(");");
            assert_eq!(args.split(',').count(), 3, "{}", test.assertion);
        }
        assert!(suite.unsupported_reasons.is_empty());
    }

    #[test]
    fn test_unknown_parameter_prevents_guessed_default_argument() {
        let request = SynthesisRequest::new("rust", "check", "Check a value")
            .with_signature("fn check(value: i32, context: NonDefault) -> bool");
        let suite = TestGenerator::generate(&request);

        assert!(suite.tests.is_empty());
        assert!(!suite.source.contains("Default::default()"));
        assert!(!suite.source.contains("check("));
        assert!(suite.unsupported_reasons.iter().any(|reason| reason.contains("no safe constructor")));
    }

    #[test]
    fn test_option_return_does_not_create_comment_only_test() {
        let request = SynthesisRequest::new("rust", "lookup", "Look up an item")
            .with_signature("fn lookup(key: u32) -> Option<String>");
        let suite = TestGenerator::generate(&request);

        assert!(!suite.source.contains("can_return_none"));
        assert!(!suite.source.contains("Option return type"));
        assert!(suite.tests.iter().all(|test| !test.assertion.trim_start().starts_with("//")));
        assert!(suite.unsupported_reasons.iter().any(|reason| reason.contains("No property/error oracle")));
    }

    #[test]
    fn test_no_tests_for_empty_request() {
        let request = SynthesisRequest::new("rust", "mystery", "Unknown function");
        let suite = TestGenerator::generate(&request);
        // No signature, no examples → minimal or no tests
        assert!(suite.tests.is_empty() || suite.source.contains("No tests generated"));
    }

    #[test]
    fn test_render_module_format() {
        let request = SynthesisRequest::new("rust", "double", "Double a number")
            .with_signature("fn double(x: i32) -> i32")
            .with_example("double(5)", "10");

        let suite = TestGenerator::generate(&request);

        assert!(suite.source.contains("#[cfg(test)]"));
        assert!(suite.source.contains("mod tests {"));
        assert!(suite.source.contains("use super::*;"));
        assert!(suite.source.contains("#[test]"));
        assert!(suite.source.ends_with("}\n"));
    }
}
