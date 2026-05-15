// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Python-specific code emitter
//!
//! Generates Python source code following PEP 8/black conventions from
//! abstract code plans and specifications.

use super::super::code_intent::CodeSpec;
use super::{CodeEmitter, extract_fields_from_text};
use crate::dynamics::cfc_code_sequencer::{CodePlanStep, PlanAction};

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

            let fields = extract_fields_from_text(&spec.purpose);
            if !fields.is_empty() {
                let params: Vec<String> = fields
                    .iter()
                    .map(|(n, t)| format!("{}: {}", n, t))
                    .collect();
                parts.push(format!("    def __init__(self, {}):", params.join(", ")));
                for (n, _) in &fields {
                    parts.push(format!("        self.{} = {}", n, n));
                }
            } else {
                parts.push("    def __init__(self):".to_string());
                parts.push("        pass".to_string());
            }
            parts.push(String::new());
        }

        if has_function || !has_class {
            let sig = spec.signature.as_deref().unwrap_or("");
            if sig.is_empty() {
                parts.push(format!("def {}():", spec.name));
            } else {
                // Ensure it ends with ':'
                let sig_str = if sig.ends_with(':') {
                    sig.to_string()
                } else {
                    format!("{}:", sig)
                };
                parts.push(sig_str);
            }

            parts.push(format!("    \"\"\"{}\"\"\"", spec.purpose));

            let purpose_lower = spec.purpose.to_lowercase();
            // Try to generate a real body
            if purpose_lower.contains("reverse") {
                parts.push("    return s[::-1]".to_string());
            } else if purpose_lower.contains("add") || purpose_lower.contains("sum") {
                parts.push("    return a + b".to_string());
            } else if purpose_lower.contains("length") || purpose_lower.contains("len") {
                parts.push("    return len(s)".to_string());
            } else if purpose_lower.contains("sort") {
                parts.push("    return sorted(items)".to_string());
            } else if purpose_lower.contains("uppercase") {
                parts.push("    return s.upper()".to_string());
            } else if purpose_lower.contains("lowercase") {
                parts.push("    return s.lower()".to_string());
            } else if purpose_lower.contains("factorial") {
                parts.push("    import math".to_string());
                parts.push("    return math.factorial(n)".to_string());
            } else if purpose_lower.contains("subtract") || purpose_lower.contains("difference") {
                parts.push("    return a - b".to_string());
            } else if purpose_lower.contains("multiply") || purpose_lower.contains("product") {
                parts.push("    return a * b".to_string());
            } else if purpose_lower.contains("divide") || purpose_lower.contains("quotient") {
                parts.push("    return a / b".to_string());
            } else if purpose_lower.contains("maximum")
                || purpose_lower.contains("max of")
                || purpose_lower.contains("larger")
            {
                parts.push("    return max(a, b)".to_string());
            } else if purpose_lower.contains("minimum")
                || purpose_lower.contains("min of")
                || purpose_lower.contains("smaller")
            {
                parts.push("    return min(a, b)".to_string());
            } else if purpose_lower.contains("absolute") || purpose_lower.contains("abs") {
                parts.push("    return abs(n)".to_string());
            } else if purpose_lower.contains("clamp") {
                parts.push("    return max(min_val, min(max_val, n))".to_string());
            } else if purpose_lower.contains("contains") || purpose_lower.contains("has") {
                parts.push("    return needle in haystack".to_string());
            } else if purpose_lower.contains("concatenate") || purpose_lower.contains("join") {
                parts.push("    return a + b".to_string());
            } else if purpose_lower.contains("split") {
                parts.push("    return s.split(sep)".to_string());
            } else if purpose_lower.contains("trim") || purpose_lower.contains("strip") {
                parts.push("    return s.strip()".to_string());
            } else if purpose_lower.contains("replace") {
                parts.push("    return s.replace(old, new)".to_string());
            } else if purpose_lower.contains("starts_with") || purpose_lower.contains("prefix") {
                parts.push("    return s.startswith(prefix)".to_string());
            } else if purpose_lower.contains("ends_with") || purpose_lower.contains("suffix") {
                parts.push("    return s.endswith(suffix)".to_string());
            } else if purpose_lower.contains("filter") && purpose_lower.contains("even") {
                parts.push("    return [x for x in items if x % 2 == 0]".to_string());
            } else if purpose_lower.contains("filter") && purpose_lower.contains("odd") {
                parts.push("    return [x for x in items if x % 2 != 0]".to_string());
            } else if purpose_lower.contains("is_empty") {
                parts.push("    return len(s) == 0".to_string());
            } else if purpose_lower.contains("is_even") {
                parts.push("    return n % 2 == 0".to_string());
            } else if purpose_lower.contains("is_odd") {
                parts.push("    return n % 2 != 0".to_string());
            } else if purpose_lower.contains("is_positive") {
                parts.push("    return n > 0".to_string());
            } else if purpose_lower.contains("is_negative") {
                parts.push("    return n < 0".to_string());
            } else if purpose_lower.contains("fibonacci") {
                parts.push("    a, b = 0, 1".to_string());
                parts.push("    for _ in range(n):".to_string());
                parts.push("        a, b = b, a + b".to_string());
                parts.push("    return a".to_string());
            } else if purpose_lower.contains("power") || purpose_lower.contains("pow") {
                parts.push("    return a ** b".to_string());
            } else if purpose_lower.contains("sqrt") || purpose_lower.contains("square root") {
                parts.push("    import math".to_string());
                parts.push("    return math.sqrt(n)".to_string());
            } else if purpose_lower.contains("flatten") {
                parts.push("    return [x for sub in items for x in sub]".to_string());
            } else if purpose_lower.contains("unique") || purpose_lower.contains("deduplicate") {
                parts.push("    return list(set(items))".to_string());
            } else if !spec.constraints.is_empty() {
                for c in &spec.constraints {
                    parts.push(format!("    # {}", c));
                }
                parts.push(format!(
                    "    raise NotImplementedError(\"{}\")",
                    spec.purpose
                ));
            } else {
                parts.push(format!(
                    "    raise NotImplementedError(\"{}\")",
                    spec.purpose
                ));
            }
        }

        // Generate real tests from examples
        if !spec.examples.is_empty() {
            parts.push(String::new());
            parts.push(String::new());
            for (i, (input, output)) in spec.examples.iter().enumerate() {
                parts.push(format!("def test_example_{}():", i));
                parts.push(format!("    assert {} == {}", input, output));
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

        format!("def {}({}){}:\n    {}", name, params, ret, body)
    }

    fn emit_struct(&self, name: &str, fields: &[(String, String)]) -> String {
        let mut lines = vec![format!("class {}:", name)];
        if fields.is_empty() {
            lines.push("    pass".to_string());
        } else {
            let params: Vec<String> = fields
                .iter()
                .map(|(n, t)| format!("{}: {}", n, t))
                .collect();
            lines.push(format!("    def __init__(self, {}):", params.join(", ")));
            for (n, _) in fields {
                lines.push(format!("        self.{} = {}", n, n));
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
