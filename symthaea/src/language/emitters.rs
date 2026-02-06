//! Language-Specific Code Emitters
//!
//! Convert abstract code plans into language-specific source code.
//! Each emitter follows the conventions of its target language.
//!
//! # Emitters
//!
//! - **RustEmitter**: Generates Rust code following rustfmt conventions
//! - **PythonEmitter**: Generates Python code following PEP 8/black conventions
//! - **NixEmitter**: Generates Nix expressions following nixfmt conventions

use super::code_intent::CodeSpec;
use crate::dynamics::cfc_code_sequencer::{CodePlanStep, PlanAction};

/// Trait for language-specific code emitters
pub trait CodeEmitter: Send + Sync {
    /// Emit code from a specification and plan
    fn emit_from_spec(&self, spec: &CodeSpec, plan: &[CodePlanStep]) -> String;

    /// Emit a simple function skeleton
    fn emit_function(&self, name: &str, params: &str, return_type: &str, body: &str) -> String;

    /// Emit a struct/class skeleton
    fn emit_struct(&self, name: &str, fields: &[(String, String)]) -> String;

    /// Emit an import statement
    fn emit_import(&self, module: &str) -> String;

    /// Get the language name
    fn language(&self) -> &str;
}

// ============================================================================
// Rust Emitter
// ============================================================================

/// Emitter for Rust source code
pub struct RustEmitter;

impl CodeEmitter for RustEmitter {
    fn emit_from_spec(&self, spec: &CodeSpec, plan: &[CodePlanStep]) -> String {
        let mut parts = Vec::new();

        // Analyze plan to determine what to emit
        let mut has_struct = false;
        let mut has_function = false;
        let mut has_trait = false;
        let mut has_impl = false;

        for step in plan {
            match step.action {
                PlanAction::DefineStruct => has_struct = true,
                PlanAction::DefineFunction | PlanAction::AddMethod => has_function = true,
                PlanAction::DefineTrait => has_trait = true,
                PlanAction::ImplTrait => has_impl = true,
                PlanAction::AddImport => {
                    // Add a relevant import based on spec
                    if spec.purpose.contains("error") || spec.purpose.contains("Result") {
                        parts.push("use std::error::Error;".to_string());
                    }
                }
                _ => {}
            }
        }

        // Emit doc comment
        if !spec.purpose.is_empty() {
            parts.push(format!("/// {}", spec.purpose));
        }

        // Emit based on plan analysis
        if has_struct {
            parts.push(format!("pub struct {} {{", spec.name));
            parts.push("    // TODO: Add fields".to_string());
            parts.push("}".to_string());
            parts.push(String::new());
        }

        if has_trait {
            parts.push(format!("pub trait {} {{", spec.name));
            parts.push("    // TODO: Add trait methods".to_string());
            parts.push("}".to_string());
            parts.push(String::new());
        }

        if has_function || (!has_struct && !has_trait) {
            // Default to emitting a function
            let sig = spec.signature.as_deref().unwrap_or_else(|| "");
            if sig.is_empty() {
                parts.push(format!("pub fn {}() {{", spec.name));
            } else {
                parts.push(format!("{} {{", sig));
            }

            if !spec.constraints.is_empty() {
                for c in &spec.constraints {
                    parts.push(format!("    // Constraint: {}", c));
                }
            }

            parts.push("    todo!(\"Implementation\")".to_string());
            parts.push("}".to_string());
        }

        if has_impl {
            parts.push(String::new());
            parts.push(format!("impl {} {{", spec.name));
            parts.push("    // TODO: Add implementation".to_string());
            parts.push("}".to_string());
        }

        // Add tests if examples are provided
        if !spec.examples.is_empty() {
            parts.push(String::new());
            parts.push("#[cfg(test)]".to_string());
            parts.push("mod tests {".to_string());
            parts.push("    use super::*;".to_string());
            parts.push(String::new());
            for (i, (input, output)) in spec.examples.iter().enumerate() {
                parts.push(format!("    #[test]"));
                parts.push(format!("    fn test_example_{}() {{", i));
                parts.push(format!("        // Input: {}", input));
                parts.push(format!("        // Expected: {}", output));
                parts.push("        todo!(\"Implement test\")".to_string());
                parts.push("    }".to_string());
            }
            parts.push("}".to_string());
        }

        parts.join("\n")
    }

    fn emit_function(&self, name: &str, params: &str, return_type: &str, body: &str) -> String {
        let ret = if return_type.is_empty() {
            String::new()
        } else {
            format!(" -> {}", return_type)
        };

        format!(
            "pub fn {}({}){} {{\n    {}\n}}",
            name, params, ret, body
        )
    }

    fn emit_struct(&self, name: &str, fields: &[(String, String)]) -> String {
        let field_lines: Vec<String> = fields.iter()
            .map(|(n, t)| format!("    pub {}: {},", n, t))
            .collect();

        format!(
            "pub struct {} {{\n{}\n}}",
            name,
            field_lines.join("\n")
        )
    }

    fn emit_import(&self, module: &str) -> String {
        format!("use {};", module)
    }

    fn language(&self) -> &str {
        "rust"
    }
}

// ============================================================================
// Python Emitter
// ============================================================================

/// Emitter for Python source code
pub struct PythonEmitter;

impl CodeEmitter for PythonEmitter {
    fn emit_from_spec(&self, spec: &CodeSpec, plan: &[CodePlanStep]) -> String {
        let mut parts = Vec::new();

        let mut has_class = false;
        let mut has_function = false;

        for step in plan {
            match step.action {
                PlanAction::DefineStruct | PlanAction::DefineTrait => has_class = true,
                PlanAction::DefineFunction | PlanAction::AddMethod => has_function = true,
                PlanAction::AddImport => {
                    parts.push("from typing import Optional, List".to_string());
                }
                _ => {}
            }
        }

        if has_class {
            parts.push(format!("class {}:", spec.name));
            parts.push(format!("    \"\"\"{}\"\"\"", spec.purpose));
            parts.push(String::new());
            parts.push("    def __init__(self):".to_string());
            parts.push("        pass  # TODO: Initialize".to_string());
            parts.push(String::new());
        }

        if has_function || (!has_class) {
            let sig = spec.signature.as_deref().unwrap_or("");
            if sig.is_empty() {
                parts.push(format!("def {}():", spec.name));
            } else {
                parts.push(format!("{}:", sig));
            }

            parts.push(format!("    \"\"\"{}\"\"\"", spec.purpose));

            if !spec.constraints.is_empty() {
                for c in &spec.constraints {
                    parts.push(format!("    # Constraint: {}", c));
                }
            }

            parts.push("    pass  # TODO: Implement".to_string());
        }

        // Add tests if examples exist
        if !spec.examples.is_empty() {
            parts.push(String::new());
            parts.push(String::new());
            for (i, (input, output)) in spec.examples.iter().enumerate() {
                parts.push(format!("def test_example_{}():", i));
                parts.push(format!("    # Input: {}", input));
                parts.push(format!("    # Expected: {}", output));
                parts.push("    pass  # TODO: Implement test".to_string());
                parts.push(String::new());
            }
        }

        parts.join("\n")
    }

    fn emit_function(&self, name: &str, params: &str, return_type: &str, body: &str) -> String {
        let ret = if return_type.is_empty() {
            String::new()
        } else {
            format!(" -> {}", return_type)
        };

        format!(
            "def {}({}){}:\n    {}",
            name, params, ret, body
        )
    }

    fn emit_struct(&self, name: &str, fields: &[(String, String)]) -> String {
        let mut lines = vec![format!("class {}:", name)];
        if fields.is_empty() {
            lines.push("    pass".to_string());
        } else {
            lines.push("    def __init__(self):".to_string());
            for (n, t) in fields {
                lines.push(format!("        self.{}: {} = None", n, t));
            }
        }
        lines.join("\n")
    }

    fn emit_import(&self, module: &str) -> String {
        format!("import {}", module)
    }

    fn language(&self) -> &str {
        "python"
    }
}

// ============================================================================
// Nix Emitter
// ============================================================================

/// Emitter for Nix expressions
pub struct NixEmitter;

impl CodeEmitter for NixEmitter {
    fn emit_from_spec(&self, spec: &CodeSpec, _plan: &[CodePlanStep]) -> String {
        let mut parts = Vec::new();

        parts.push(format!("# {}", spec.purpose));

        // Nix code is typically a function or attrset
        if spec.name.contains("derivation") || spec.name.contains("package") {
            parts.push(format!("{{ lib, stdenv, ... }}:"));
            parts.push(String::new());
            parts.push("stdenv.mkDerivation {".to_string());
            parts.push(format!("  pname = \"{}\";", spec.name));
            parts.push("  version = \"0.1.0\";".to_string());
            parts.push(String::new());
            parts.push("  src = ./.;".to_string());
            parts.push(String::new());
            parts.push("  # TODO: Add build inputs".to_string());
            parts.push("}".to_string());
        } else if spec.name.contains("module") || spec.name.contains("config") {
            parts.push("{ config, lib, pkgs, ... }:".to_string());
            parts.push(String::new());
            parts.push("{".to_string());
            parts.push(format!("  # {}", spec.purpose));
            parts.push("  # TODO: Add module options and config".to_string());
            parts.push("}".to_string());
        } else {
            // Default: function
            parts.push(format!("{} =", spec.name));
            if !spec.constraints.is_empty() {
                for c in &spec.constraints {
                    parts.push(format!("  # Constraint: {}", c));
                }
            }
            parts.push("  # TODO: Implement".to_string());
            parts.push("  null;".to_string());
        }

        parts.join("\n")
    }

    fn emit_function(&self, name: &str, params: &str, _return_type: &str, body: &str) -> String {
        if params.is_empty() {
            format!("{} = {};", name, body)
        } else {
            format!("{} = {}: {};", name, params, body)
        }
    }

    fn emit_struct(&self, name: &str, fields: &[(String, String)]) -> String {
        let field_lines: Vec<String> = fields.iter()
            .map(|(n, _t)| format!("  {} = null; # TODO", n))
            .collect();

        format!(
            "{} = {{\n{}\n}};",
            name,
            field_lines.join("\n")
        )
    }

    fn emit_import(&self, module: &str) -> String {
        format!("imports = [ {} ];", module)
    }

    fn language(&self) -> &str {
        "nix"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_emit_function() {
        let emitter = RustEmitter;
        let result = emitter.emit_function("add", "a: i32, b: i32", "i32", "a + b");
        assert!(result.contains("pub fn add"));
        assert!(result.contains("-> i32"));
        assert!(result.contains("a + b"));
    }

    #[test]
    fn test_rust_emit_struct() {
        let emitter = RustEmitter;
        let fields = vec![
            ("name".to_string(), "String".to_string()),
            ("age".to_string(), "u32".to_string()),
        ];
        let result = emitter.emit_struct("Person", &fields);
        assert!(result.contains("pub struct Person"));
        assert!(result.contains("pub name: String"));
        assert!(result.contains("pub age: u32"));
    }

    #[test]
    fn test_python_emit_function() {
        let emitter = PythonEmitter;
        let result = emitter.emit_function("add", "a: int, b: int", "int", "return a + b");
        assert!(result.contains("def add"));
        assert!(result.contains("-> int"));
    }

    #[test]
    fn test_python_emit_struct() {
        let emitter = PythonEmitter;
        let fields = vec![
            ("name".to_string(), "str".to_string()),
        ];
        let result = emitter.emit_struct("Person", &fields);
        assert!(result.contains("class Person"));
        assert!(result.contains("self.name: str"));
    }

    #[test]
    fn test_nix_emit_function() {
        let emitter = NixEmitter;
        let result = emitter.emit_function("greet", "name", "", "\"Hello ${name}\"");
        assert!(result.contains("greet = name:"));
    }

    #[test]
    fn test_rust_emit_from_spec() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "sort_vec", "Sort a vector in place");
        let plan = vec![
            CodePlanStep {
                action: PlanAction::DefineFunction,
                name: None,
                context: Vec::new(),
                confidence: 0.9,
            },
        ];
        let result = emitter.emit_from_spec(&spec, &plan);
        assert!(result.contains("sort_vec"));
        assert!(result.contains("Sort a vector"));
    }

    #[test]
    fn test_python_emit_from_spec_with_examples() {
        let emitter = PythonEmitter;
        let spec = CodeSpec::new("python", "add", "Add two numbers")
            .with_example("add(1, 2)", "3");
        let plan = vec![
            CodePlanStep {
                action: PlanAction::DefineFunction,
                name: None,
                context: Vec::new(),
                confidence: 0.9,
            },
        ];
        let result = emitter.emit_from_spec(&spec, &plan);
        assert!(result.contains("def add"));
        assert!(result.contains("test_example_0"));
    }

    #[test]
    fn test_nix_emit_derivation() {
        let emitter = NixEmitter;
        let spec = CodeSpec::new("nix", "my-derivation", "Build a custom package");
        let result = emitter.emit_from_spec(&spec, &[]);
        assert!(result.contains("mkDerivation"));
    }
}
