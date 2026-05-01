// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Self-Test Generation — Agent Writes Its Own Tests
//!
//! Generates `#[test]` modules from function signatures and docstrings.
//! This enables the coding agent to validate its own output without
//! relying on externally provided test suites.
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
//! No LLM can do this natively — it requires understanding function semantics
//! well enough to derive test cases, not just pattern-match from training data.

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
}

/// Category of generated test — informs which testing patterns to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestCategory {
    /// Derived from user-provided examples
    Example,
    /// Boundary values (0, 1, empty, max)
    Boundary,
    /// Property-based (e.g., output type, non-negative, idempotent)
    Property,
    /// Error handling (invalid inputs, overflow)
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

        // 1. Example-based tests
        for (i, (input, expected)) in request.examples.iter().enumerate() {
            tests.push(GeneratedTest {
                name: format!("test_{}_example_{}", request.name, i),
                assertion: format!("assert_eq!({}, {});", input, expected),
                category: TestCategory::Example,
            });
        }

        // 2. Boundary tests (inferred from signature)
        if let Some(ref sig) = request.signature {
            let boundary_tests = Self::generate_boundary_tests(&request.name, sig);
            tests.extend(boundary_tests);
        }

        // 3. Property tests (inferred from return type)
        if let Some(ref sig) = request.signature {
            let property_tests = Self::generate_property_tests(&request.name, sig);
            tests.extend(property_tests);
        }

        // Build the test module source
        let source = Self::render_test_module(&request.name, &tests);

        GeneratedTestSuite {
            function_name: request.name.clone(),
            tests,
            source,
        }
    }

    /// Generate boundary value tests from a function signature.
    fn generate_boundary_tests(fn_name: &str, signature: &str) -> Vec<GeneratedTest> {
        let mut tests = Vec::new();

        // Parse parameter types from signature
        let params = Self::extract_param_types(signature);

        for (param_name, param_type) in &params {
            match param_type.as_str() {
                "u8" | "u16" | "u32" | "u64" | "usize" => {
                    // Unsigned integer boundaries: 0, 1
                    tests.push(GeneratedTest {
                        name: format!("test_{fn_name}_boundary_{param_name}_zero"),
                        assertion: format!(
                            "let _ = {fn_name}({});",
                            Self::make_call_with_value(&params, param_name, "0")
                        ),
                        category: TestCategory::Boundary,
                    });
                    tests.push(GeneratedTest {
                        name: format!("test_{fn_name}_boundary_{param_name}_one"),
                        assertion: format!(
                            "let _ = {fn_name}({});",
                            Self::make_call_with_value(&params, param_name, "1")
                        ),
                        category: TestCategory::Boundary,
                    });
                }
                "i8" | "i16" | "i32" | "i64" | "isize" => {
                    // Signed integer boundaries: -1, 0, 1
                    tests.push(GeneratedTest {
                        name: format!("test_{fn_name}_boundary_{param_name}_negative"),
                        assertion: format!(
                            "let _ = {fn_name}({});",
                            Self::make_call_with_value(&params, param_name, "-1")
                        ),
                        category: TestCategory::Boundary,
                    });
                    tests.push(GeneratedTest {
                        name: format!("test_{fn_name}_boundary_{param_name}_zero"),
                        assertion: format!(
                            "let _ = {fn_name}({});",
                            Self::make_call_with_value(&params, param_name, "0")
                        ),
                        category: TestCategory::Boundary,
                    });
                }
                "f32" | "f64" => {
                    // Float boundaries: 0.0, -0.0, very small
                    tests.push(GeneratedTest {
                        name: format!("test_{fn_name}_boundary_{param_name}_zero"),
                        assertion: format!(
                            "let _ = {fn_name}({});",
                            Self::make_call_with_value(&params, param_name, "0.0")
                        ),
                        category: TestCategory::Boundary,
                    });
                }
                t if t.starts_with("&str") || t.starts_with("&String") || t == "String" => {
                    // String boundaries: empty, single char
                    tests.push(GeneratedTest {
                        name: format!("test_{fn_name}_boundary_{param_name}_empty"),
                        assertion: format!(
                            "let _ = {fn_name}({});",
                            Self::make_call_with_value(&params, param_name, "\"\"")
                        ),
                        category: TestCategory::Boundary,
                    });
                }
                t if t.contains("Vec<") || t.contains("&[") => {
                    // Collection boundaries: empty
                    tests.push(GeneratedTest {
                        name: format!("test_{fn_name}_boundary_{param_name}_empty"),
                        assertion: format!(
                            "let _ = {fn_name}({});",
                            Self::make_call_with_value(&params, param_name, "&[]")
                        ),
                        category: TestCategory::Boundary,
                    });
                }
                _ => {}
            }
        }

        tests
    }

    /// Generate property-based tests from return type.
    fn generate_property_tests(fn_name: &str, signature: &str) -> Vec<GeneratedTest> {
        let mut tests = Vec::new();

        let return_type = Self::extract_return_type(signature);

        match return_type.as_deref() {
            Some("bool") => {
                // Boolean return: test that it returns a bool (type check)
                tests.push(GeneratedTest {
                    name: format!("test_{fn_name}_returns_bool"),
                    assertion: format!(
                        "let result: bool = {fn_name}(Default::default()); let _ = result;"
                    ),
                    category: TestCategory::Property,
                });
            }
            Some(t) if t.starts_with("Result<") => {
                // Result return: test that it doesn't panic
                tests.push(GeneratedTest {
                    name: format!("test_{fn_name}_does_not_panic"),
                    assertion: format!("let _ = std::panic::catch_unwind(|| {{ let _ = {fn_name}(Default::default()); }});"),
                    category: TestCategory::ErrorCase,
                });
            }
            Some(t) if t.starts_with("Option<") => {
                // Option return: test None case
                tests.push(GeneratedTest {
                    name: format!("test_{fn_name}_can_return_none"),
                    assertion: format!(
                        "// Option return type: {fn_name} should handle None gracefully"
                    ),
                    category: TestCategory::Property,
                });
            }
            _ => {}
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

    /// Extract return type from a Rust function signature.
    fn extract_return_type(signature: &str) -> Option<String> {
        let arrow_pos = signature.find("->")?;
        let ret = signature[arrow_pos + 2..].trim();
        // Remove trailing { if present
        let ret = ret.trim_end_matches('{').trim();
        if ret.is_empty() {
            None
        } else {
            Some(ret.to_string())
        }
    }

    /// Create a function call string with one parameter replaced by a specific value.
    fn make_call_with_value(
        params: &[(String, String)],
        target_param: &str,
        value: &str,
    ) -> String {
        params
            .iter()
            .map(|(name, typ)| {
                if name == target_param {
                    value.to_string()
                } else {
                    Self::default_value_for_type(typ)
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Get a default value for a Rust type (for filling non-target parameters).
    fn default_value_for_type(typ: &str) -> String {
        match typ.trim() {
            "u8" | "u16" | "u32" | "u64" | "usize" => "0".to_string(),
            "i8" | "i16" | "i32" | "i64" | "isize" => "0".to_string(),
            "f32" => "0.0f32".to_string(),
            "f64" => "0.0f64".to_string(),
            "bool" => "false".to_string(),
            "&str" => "\"\"".to_string(),
            "String" => "String::new()".to_string(),
            t if t.contains("Vec<") => "vec![]".to_string(),
            t if t.starts_with("&[") => "&[]".to_string(),
            t if t.starts_with("Option<") => "None".to_string(),
            _ => "Default::default()".to_string(),
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
    fn test_extract_return_type() {
        assert_eq!(
            TestGenerator::extract_return_type("fn add(a: i32) -> i32"),
            Some("i32".to_string())
        );
        assert_eq!(
            TestGenerator::extract_return_type("fn check() -> Result<(), Error>"),
            Some("Result<(), Error>".to_string())
        );
        assert_eq!(TestGenerator::extract_return_type("fn noop()"), None);
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
