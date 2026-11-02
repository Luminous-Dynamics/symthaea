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
//!         Test { name: "test_fibonacci_example_0", assert: "fibonacci(0)", expected: "0" },
//!         Test { name: "test_fibonacci_example_1", assert: "fibonacci(10)", expected: "55" },
//!         Test { name: "test_fibonacci_boundary_zero", assert: "fibonacci(0)", expected: "0" },
//!         Test { name: "test_fibonacci_boundary_one", assert: "fibonacci(1)", expected: "1" },
//!         Test { name: "test_fibonacci_large", assert: "fibonacci(20)", expected: "6765" },
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
    /// Every non-target argument must have an explicit, known literal constructor.
    /// Unknown types are skipped rather than guessed with `Default::default()`.
    /// These calls do not assert return-value semantics.
    fn generate_boundary_tests(fn_name: &str, signature: &str) -> Vec<GeneratedTest> {
        let mut tests = Vec::new();
        let params = Self::extract_param_types(signature);

        for (param_name, param_type) in &params {
            let cases: &[(&str, &str)] = match param_type.as_str() {
                "u8" | "u16" | "u32" | "u64" | "usize" => {
                    &[("zero", "0"), ("one", "1")]
                }
                "i8" | "i16" | "i32" | "i64" | "isize" => {
                    &[("negative", "-1"), ("zero", "0")]
                }
                "f32" => &[("zero", "0.0f32")],
                "f64" => &[("zero", "0.0f64")],
                "&str" => &[("empty", "\"\"\"")],
                "&String" => &[("empty", "&String::new()")],
                "String" => &[("empty", "String::new()")],
                t if t.starts_with("&[") => &[("empty", "&[]")],
                t if t.starts_with("&Vec<") => &[("empty", "&Vec::new()")],
                t if t.starts_with("Vec<") => &[("empty", "vec![]")],
                _ => continue,
            };

            for (suffix, value) in cases {
                let Some(arguments) =
                    Self::make_call_with_value(&params, param_name, value)
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

    /// Extract parameter names and types from a Rust function signature.
    fn extract_param_types(signature: &str) -> Vec<(String, String)> {
        let mut params = Vec::new();

        // Find the parameter list between ( and )
        let open = match signature.find('(') {
            Some(i) => i + 1,
            None => return params,
        };
        let close = match signature.find(')') {
            Some(i) => i,
            None => return params,
        };

        let param_str = &signature[open..close];
        if param_str.trim().is_empty() {
            return params;
        }

        // Split by comma, handling generic angle brackets
        for param in Self::split_params(param_str) {
            let param = param.trim();
            if param.is_empty() || param == "&self" || param == "&mut self" || param == "self" {
                continue;
            }

            // Split by `: ` to get name and type
            if let Some(colon_pos) = param.find(':') {
                let name = param[..colon_pos].trim().to_string();
                let typ = param[colon_pos + 1..].trim().to_string();
                params.push((name, typ));
            }
        }

        params
    }

    /// Split parameter string by commas, respecting angle bracket nesting.
    fn split_params(s: &str) -> Vec<String> {
        let mut parts = Vec::new();
        let mut current = String::new();
        let mut depth = 0;

        for ch in s.chars() {
            match ch {
                '<' => {
                    depth += 1;
                    current.push(ch);
                }
                '>' => {
                    depth -= 1;
                    current.push(ch);
                }
                ',' if depth == 0 => {
                    parts.push(current.clone());
                    current.clear();
                }
                _ => current.push(ch),
            }
        }
        if !current.is_empty() {
            parts.push(current);
        }
        parts
    }

    /// Create a function call only when every argument can be constructed safely.
    fn make_call_with_value(
        params: &[(String, String)],
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
    /// Returning None is intentional: guessing that an arbitrary user type
    /// implements Default can produce uncompilable tests and false confidence.
    fn default_value_for_type(typ: &str) -> Option<String> {
        match typ.trim() {
            "u8" | "u16" | "u32" | "u64" | "usize"
            | "i8" | "i16" | "i32" | "i64" | "isize" => Some("0".to_string()),
            "f32" => Some("0.0f32".to_string()),
            "f64" => Some("0.0f64".to_string()),
            "bool" => Some("false".to_string()),
            "&str" => Some("\"\"".to_string()),
            "&String" => Some("&String::new()".to_string()),
            "String" => Some("String::new()".to_string()),
            t if t.starts_with("&[") => Some("&[]".to_string()),
            t if t.starts_with("&Vec<") => Some("&Vec::new()".to_string()),
            t if t.starts_with("Vec<") => Some("vec![]".to_string()),
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
        assert_eq!(params[0].1, "i32");
        assert_eq!(params[1].0, "b");
        assert_eq!(params[1].1, "i32");
    }

    #[test]
    fn test_extract_param_types_generic() {
        let params = TestGenerator::extract_param_types(
            "fn find(haystack: &[u8], needle: Vec<u8>) -> Option<usize>",
        );
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].1, "&[u8]");
        assert_eq!(params[1].1, "Vec<u8>");
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
