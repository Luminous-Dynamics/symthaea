// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Self-Test Generation — Agent Writes Its Own Tests
//!
//! Generates example-based assertions and conservative invocation smoke checks.
//!
//! A generated call is not a behavioral oracle: it can establish that a call
//! shape compiles and that the candidate does not panic for one input, but it
//! cannot establish the requested result without an independent expected value.
//!
//! ```text
//! Function signature: fn fibonacci(n: u64) -> u64
//! Examples: fibonacci(0)=0, fibonacci(10)=55
//!     ↓
//! TestSuite {
//!     tests: [
//!         Test { assertion: "assert_eq!(fibonacci(10), 55)", oracle: ProvidedExample },
//!         Test { assertion: "let _ = fibonacci(0)", oracle: TypeDerivedInvocationSmoke },
//!     ]
//! }
//! ```
//!
//! Return types alone do not specify behavior. This generator therefore does
//! not invent Property/Result/Option assertions without an independent oracle.

use symthaea_core::synthesis_trait::SynthesisRequest;

/// A generated test case.
#[derive(Debug, Clone)]
pub struct GeneratedTest {
    /// Test function name (e.g., "test_fibonacci_example_0")
    pub name: String,
    /// The assertion expression (e.g., "assert_eq!(fibonacci(0), 0)")
    pub assertion: String,
    /// Category of this test
    pub category: TestCategory,
    /// Provenance and strength of the oracle; smoke checks are not correctness proofs.
    pub oracle_kind: TestOracleKind,
}

/// Provenance of the assertion/oracle attached to a generated test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestOracleKind {
    /// Explicit input/output pair supplied with the synthesis request.
    ProvidedExample,
    /// Type-derived input only; checks invocation/compilation, not expected behavior.
    TypeDerivedInvocationSmoke,
}

/// Category of generated test — informs which testing patterns to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestCategory {
    /// Derived from user-provided examples
    Example,
    /// Type-derived boundary input. This category is an invocation smoke check,
    /// not a specification-backed assertion about the returned value.
    Boundary,
    /// Reserved for future properties supplied with an independent specification.
    Property,
    /// Reserved for future error cases supplied with an independent specification.
    ErrorCase,
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
}

/// Generates test suites from function specifications.
pub struct TestGenerator;

impl TestGenerator {
    /// Generate a test suite from a SynthesisRequest.
    ///
    /// Produces tests from three sources:
    /// 1. User-provided examples (exact I/O pairs)
    /// 2. Boundary values (inferred from parameter types)
    /// 3. Property-based checks (inferred from return type)
    pub fn generate(request: &SynthesisRequest) -> GeneratedTestSuite {
        let mut tests = Vec::new();

        // 1. Example-based tests with explicitly supplied expected values
        for (i, (input, expected)) in request.examples.iter().enumerate() {
            tests.push(GeneratedTest {
                name: format!("test_{}_example_{}", request.name, i),
                assertion: format!("assert_eq!({}, {});", input, expected),
                category: TestCategory::Example,
                oracle_kind: TestOracleKind::ProvidedExample,
            });
        }

        // 2. Boundary tests (inferred from signature)
        if let Some(ref sig) = request.signature {
            let boundary_tests = Self::generate_boundary_tests(&request.name, sig);
            tests.extend(boundary_tests);
        }

        // 3. Do not infer behavioral properties from return types alone.
        // An Option<T> result does not imply that None is expected, and bool or
        // Result<T, E> does not supply valid arguments or an expected outcome.
        // Build the test module source
        let source = Self::render_test_module(&request.name, &tests);

        GeneratedTestSuite {
            function_name: request.name.clone(),
            tests,
            source,
        }
    }

    /// Generate conservative invocation smoke checks using known boundary inputs.
    ///
    /// Generic functions are skipped because an empty/default argument may not
    /// provide enough information to infer their type parameters. Every non-target
    /// argument must also have a known literal constructor.
    fn generate_boundary_tests(fn_name: &str, signature: &str) -> Vec<GeneratedTest> {
        let Some(parsed_signature) = Self::parse_signature(signature) else {
            return Vec::new();
        };
        if !parsed_signature.generics.params.is_empty() {
            return Vec::new();
        }

        let params = Self::extract_param_types(signature);
        let mut tests = Vec::new();

        for (param_name, param_type) in &params {
            for (suffix, value) in Self::boundary_values_for_type(param_type) {
                let Some(arguments) =
                    Self::make_call_with_value(&params, param_name, &value)
                else {
                    // At least one other parameter has no known safe constructor.
                    continue;
                };

                tests.push(GeneratedTest {
                    name: format!("test_{fn_name}_boundary_{param_name}_{suffix}"),
                    assertion: format!("let _ = {fn_name}({arguments});"),
                    category: TestCategory::Boundary,
                    oracle_kind: TestOracleKind::TypeDerivedInvocationSmoke,
                });
            }
        }

        tests
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

    /// Parse a function signature through syn so nested function-pointer,
    /// tuple and generic argument types are not split by string heuristics.
    fn parse_signature(signature: &str) -> Option<syn::Signature> {
        let signature = signature.trim().trim_end_matches('{').trim();
        let item_source = format!("{signature} {{}}");
        syn::parse_str::<syn::ItemFn>(&item_source)
            .ok()
            .map(|item| item.sig)
    }

    /// Extract simple identifier parameters with their parsed Rust types.
    ///
    /// Receiver arguments and destructuring patterns are intentionally omitted:
    /// this generator does not guess how to construct them.
    fn extract_param_types(signature: &str) -> Vec<(String, syn::Type)> {
        let Some(signature) = Self::parse_signature(signature) else {
            return Vec::new();
        };

        signature
            .inputs
            .into_iter()
            .filter_map(|argument| match argument {
                syn::FnArg::Typed(argument) => match *argument.pat {
                    syn::Pat::Ident(pattern)
                        if pattern.by_ref.is_none() && pattern.subpat.is_none() =>
                    {
                        Some((pattern.ident.to_string(), *argument.ty))
                    }
                    _ => None,
                },
                syn::FnArg::Receiver(_) => None,
            })
            .collect()
    }

    /// Generate representative values only for a conservative set of concrete types.
    fn boundary_values_for_type(typ: &syn::Type) -> Vec<(&'static str, String)> {
        use syn::Type;

        match typ {
            Type::Reference(reference) if reference.mutability.is_none() => {
                match reference.elem.as_ref() {
                    Type::Path(_) if Self::path_type_ident(&reference.elem).as_deref() == Some("str") => {
                        vec![("empty", "\"\"".to_string())]
                    }
                    Type::Path(_) if Self::path_type_ident(&reference.elem).as_deref() == Some("String") => {
                        vec![("empty", "&String::new()".to_string())]
                    }
                    Type::Slice(slice) if Self::is_supported_element_type(&slice.elem) => {
                        vec![("empty", "&[]".to_string())]
                    }
                    inner if Self::is_supported_vec_type(inner) => {
                        vec![("empty", "&Vec::new()".to_string())]
                    }
                    _ => Vec::new(),
                }
            }
            Type::Path(_) => match Self::path_type_ident(typ).as_deref() {
                Some("u8" | "u16" | "u32" | "u64" | "usize" | "i8" | "i16" | "i32" | "i64" | "isize") => {
                    vec![("zero", "0".to_string()), ("one", "1".to_string())]
                }
                Some("f32") => vec![("zero", "0.0f32".to_string())],
                Some("f64") => vec![("zero", "0.0f64".to_string())],
                Some("bool") => vec![
                    ("false", "false".to_string()),
                    ("true", "true".to_string()),
                ],
                Some("String") => vec![("empty", "String::new()".to_string())],
                _ if Self::is_supported_vec_type(typ) => {
                    vec![("empty", "vec![]".to_string())]
                }
                _ => Vec::new(),
            },
            _ => Vec::new(),
        }
    }

    /// Return the final path identifier for a concrete path type.
    fn path_type_ident(typ: &syn::Type) -> Option<String> {
        match typ {
            syn::Type::Path(path) if path.qself.is_none() => {
                path.path.segments.last().map(|segment| segment.ident.to_string())
            }
            _ => None,
        }
    }

    /// Keep empty collection construction to well-understood concrete element types.
    fn is_supported_element_type(typ: &syn::Type) -> bool {
        matches!(
            Self::path_type_ident(typ).as_deref(),
            Some(
                "u8" | "u16" | "u32" | "u64" | "usize"
                    | "i8" | "i16" | "i32" | "i64" | "isize"
                    | "f32" | "f64" | "bool" | "char" | "String"
            )
        )
    }

    fn is_supported_vec_type(typ: &syn::Type) -> bool {
        let syn::Type::Path(path) = typ else {
            return false;
        };
        let Some(segment) = path.path.segments.last() else {
            return false;
        };
        if segment.ident.to_string() != "Vec" {
            return false;
        }
        let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
            return false;
        };
        let Some(syn::GenericArgument::Type(element_type)) = arguments.args.first() else {
            return false;
        };
        Self::is_supported_element_type(element_type)
    }

    /// Create a function call only when every argument can be constructed safely.
    fn make_call_with_value(
        params: &[(String, syn::Type)],
        target_param: &str,
        value: &str,
    ) -> Option<String> {
        let arguments = params
            .iter()
            .map(|(name, typ)| {
                if name == target_param {
                    Some(value.to_string())
                } else {
                    Self::default_value_for_type(typ)
                }
            })
            .collect::<Option<Vec<_>>>()?;

        Some(arguments.join(", "))
    }

    /// Return a conservative literal for a small set of well-understood types.
    ///
    /// Unknown, generic, mutable-reference and custom types return None. Guessing
    /// that an arbitrary type implements Default would create invalid test calls.
    fn default_value_for_type(typ: &syn::Type) -> Option<String> {
        use syn::Type;

        match typ {
            Type::Reference(reference) if reference.mutability.is_none() => {
                match reference.elem.as_ref() {
                    Type::Path(_) if Self::path_type_ident(&reference.elem).as_deref() == Some("str") => {
                        Some("\"\"".to_string())
                    }
                    Type::Path(_) if Self::path_type_ident(&reference.elem).as_deref() == Some("String") => {
                        Some("&String::new()".to_string())
                    }
                    Type::Slice(slice) if Self::is_supported_element_type(&slice.elem) => {
                        Some("&[]".to_string())
                    }
                    inner if Self::is_supported_vec_type(inner) => {
                        Some("&Vec::new()".to_string())
                    }
                    _ => None,
                }
            }
            Type::Path(_) => match Self::path_type_ident(typ).as_deref() {
                Some("u8" | "u16" | "u32" | "u64" | "usize" | "i8" | "i16" | "i32" | "i64" | "isize") => {
                    Some("0".to_string())
                }
                Some("f32") => Some("0.0f32".to_string()),
                Some("f64") => Some("0.0f64".to_string()),
                Some("bool") => Some("false".to_string()),
                Some("String") => Some("String::new()".to_string()),
                _ if Self::is_supported_vec_type(typ) => Some("vec![]".to_string()),
                _ => None,
            },
            _ => None,
        }
    }
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
        assert!(suite
            .tests
            .iter()
            .filter(|test| test.category == TestCategory::Example)
            .all(|test| test.oracle_kind == TestOracleKind::ProvidedExample));
    }

    #[test]
    fn test_generate_boundary_unsigned() {
        let request = SynthesisRequest::new("rust", "factorial", "Compute factorial")
            .with_signature("fn factorial(n: u64) -> u64");

        let suite = TestGenerator::generate(&request);

        let boundary_tests: Vec<_> = suite
            .tests
            .iter()
            .filter(|t| t.category == TestCategory::Boundary)
            .collect();

        assert!(
            !boundary_tests.is_empty(),
            "Should generate boundary tests for u64 param"
        );
        assert!(suite.source.contains("boundary_n_zero"));
        assert!(boundary_tests
            .iter()
            .all(|test| test.oracle_kind == TestOracleKind::TypeDerivedInvocationSmoke));
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
    fn test_extract_param_types() {
        let params = TestGenerator::extract_param_types("fn add(a: i32, b: i32) -> i32");
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].0, "a");
        assert_eq!(TestGenerator::path_type_ident(&params[0].1).as_deref(), Some("i32"));
        assert_eq!(params[1].0, "b");
        assert_eq!(TestGenerator::path_type_ident(&params[1].1).as_deref(), Some("i32"));
    }

    #[test]
    fn test_extract_concrete_generic_argument_types() {
        let params = TestGenerator::extract_param_types(
            "fn find(haystack: &[u8], needle: Vec<u8>) -> Option<usize>",
        );
        assert_eq!(params.len(), 2);
        assert!(matches!(
            &params[0].1,
            syn::Type::Reference(reference)
                if matches!(reference.elem.as_ref(), syn::Type::Slice(_))
        ));
        assert!(TestGenerator::is_supported_vec_type(&params[1].1));
    }

    #[test]
    fn nested_function_pointer_arguments_are_parsed_without_splitting() {
        let params = TestGenerator::extract_param_types(
            "fn register(callback: fn(u32, Result<(), Error>), enabled: bool)",
        );

        assert_eq!(params.len(), 2);
        assert_eq!(params[0].0, "callback");
        assert_eq!(params[1].0, "enabled");
    }

    #[test]
    fn generic_functions_do_not_receive_unresolved_smoke_inputs() {
        let request = SynthesisRequest::new("rust", "collect", "Collect generic values")
            .with_signature("fn collect<T>(values: Vec<T>) -> usize");

        let suite = TestGenerator::generate(&request);

        assert!(suite.tests.is_empty());
        assert!(suite.source.contains("No tests generated"));
    }

    #[test]
    fn generated_option_tests_never_contain_comment_only_passing_oracles() {
        let request = SynthesisRequest::new("rust", "lookup", "Find a value")
            .with_signature("fn lookup(key: &str) -> Option<String>");

        let suite = TestGenerator::generate(&request);

        assert!(suite.tests.iter().all(|test| {
            test.category != TestCategory::Property
                && test.category != TestCategory::ErrorCase
        }));
        assert!(!suite.source.contains("should handle None gracefully"));
        assert!(!suite.source.contains("test_lookup_can_return_none"));
    }

    #[test]
    fn return_type_alone_never_generates_default_argument_property_tests() {
        let bool_request = SynthesisRequest::new("rust", "allowed", "Check access")
            .with_signature("fn allowed(user: User, resource: Resource) -> bool");
        let result_request = SynthesisRequest::new("rust", "load", "Load an item")
            .with_signature("fn load(key: Key, store: Store) -> Result<Value, Error>");

        for request in [&bool_request, &result_request] {
            let suite = TestGenerator::generate(request);
            assert!(suite.tests.is_empty());
            assert!(!suite.source.contains("Default::default()"));
            assert!(suite.source.contains("No tests generated"));
        }
    }

    #[test]
    fn boundary_smoke_calls_skip_unknown_non_target_types() {
        let request = SynthesisRequest::new("rust", "check", "Check a value")
            .with_signature("fn check(value: u32, policy: CustomPolicy) -> bool");

        let suite = TestGenerator::generate(&request);

        assert!(suite.tests.is_empty());
        assert!(!suite.source.contains("Default::default()"));
    }

    #[test]
    fn boundary_smoke_calls_construct_all_known_arguments() {
        let request = SynthesisRequest::new("rust", "check", "Check a value")
            .with_signature("fn check(value: u32, enabled: bool) -> bool");

        let suite = TestGenerator::generate(&request);

        assert!(suite.source.contains("check(0, false)"));
        assert!(suite.tests.iter().all(|test| {
            test.oracle_kind == TestOracleKind::TypeDerivedInvocationSmoke
        }));
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
