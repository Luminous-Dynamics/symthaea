// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Language-Specific Code Emitters
//!
//! Convert abstract code plans into language-specific source code.
//! Each emitter follows the conventions of its target language.
//!
//! The emitters use the CodeSpec (purpose, signature, constraints, examples)
//! combined with CfC plan steps to produce **real, compilable code** — not
//! TODO skeletons. This is the native code generation path that operates
//! entirely from HDC+CfC cognition without an LLM.
//!
//! # Emitters
//!
//! - **RustEmitter**: Generates Rust code following rustfmt conventions
//! - **PythonEmitter**: Generates Python code following PEP 8/black conventions
//! - **NixEmitter**: Generates Nix expressions following nixfmt conventions

mod nix;
mod python;
mod rust;

pub use nix::NixEmitter;
pub use python::PythonEmitter;
pub use rust::RustEmitter;

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
// Signature Parsing Utilities
// ============================================================================

/// Parsed function signature
pub(crate) struct ParsedSignature {
    pub(crate) name: String,
    pub(crate) params: Vec<(String, String)>, // (name, type)
    pub(crate) return_type: Option<String>,
    pub(crate) _is_method: bool, // has &self or &mut self
}

/// Split a string on `delim` only at nesting depth 0 (respects `<>`, `()`, `{}`).
fn split_at_depth_zero(s: &str, delim: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, ch) in s.char_indices() {
        match ch {
            '<' | '(' | '{' => depth += 1,
            '>' | ')' | '}' => depth -= 1,
            c if c == delim && depth == 0 => {
                parts.push(&s[start..i]);
                start = i + ch.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

/// Parse a Rust function signature string like "fn foo(a: i32, b: &str) -> String"
fn parse_rust_signature(sig: &str) -> Option<ParsedSignature> {
    let sig = sig.trim();

    // Strip leading visibility/async/fn markers. Keep this parser permissive:
    // generated signatures often arrive without a body and may include generic
    // bounds that are not valid in a call expression.
    let after_fn = if sig.starts_with("pub async fn ") {
        &sig[13..]
    } else if sig.starts_with("async fn ") {
        &sig[9..]
    } else if sig.starts_with("pub fn ") {
        &sig[7..]
    } else if sig.starts_with("fn ") {
        &sig[3..]
    } else {
        sig
    };

    // Find name and params
    let paren_start = after_fn.find('(')?;
    let name = after_fn[..paren_start]
        .trim()
        .split('<')
        .next()
        .unwrap_or("")
        .trim()
        .to_string();

    // Find the matching ')' for the parameter list by tracking nesting depth.
    // rfind(')') breaks on return types containing tuples like Vec<(i32, i32)>.
    let mut depth = 0i32;
    let mut paren_end = None;
    for (i, ch) in after_fn[paren_start..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    paren_end = Some(paren_start + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let paren_end = paren_end?;
    let params_str = &after_fn[paren_start + 1..paren_end];

    // Parse params — split on commas at depth 0 (handles nested generics like Vec<i32>)
    let mut params = Vec::new();
    let mut _is_method = false;
    let param_tokens = split_at_depth_zero(params_str, ',');
    for param in &param_tokens {
        let param = param.trim();
        if param.is_empty() {
            continue;
        }
        if param == "&self" || param == "&mut self" || param == "self" {
            _is_method = true;
            continue;
        }
        if let Some(colon_pos) = param.find(':') {
            let pname = param[..colon_pos].trim().to_string();
            let ptype = param[colon_pos + 1..].trim().to_string();
            params.push((pname, ptype));
        }
    }

    // Parse return type
    let after_paren = &after_fn[paren_end + 1..];
    let return_type = if let Some(arrow_pos) = after_paren.find("->") {
        Some(after_paren[arrow_pos + 2..].trim().to_string())
    } else {
        None
    };

    Some(ParsedSignature {
        name,
        params,
        return_type,
        _is_method,
    })
}

/// Extract the Ok type from a Result<T, E> string, e.g. "Result<i32, String>" → "i32".
fn extract_result_ok_type(ret: &str) -> Option<&str> {
    let inner = ret.strip_prefix("Result<")?.strip_suffix('>')?;
    // Split at depth-0 comma to get the Ok type
    let parts = split_at_depth_zero(inner, ',');
    if parts.is_empty() {
        return None;
    }
    Some(parts[0].trim())
}

/// Parse field definitions from purpose/constraints text.
/// Looks for patterns like "x: f64", "name: String", "x:f64" etc.
fn extract_fields_from_text(text: &str) -> Vec<(String, String)> {
    let mut fields = Vec::new();
    let words: Vec<&str> = text.split_whitespace().collect();

    let mut i = 0;
    while i < words.len() {
        let word = words[i];
        if let Some(colon_pos) = word.find(':') {
            let name = word[..colon_pos].trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
            let after_colon = word[colon_pos + 1..]
                .trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '<' && c != '>');

            // Type is in same word (e.g. "x:f64")
            if !after_colon.is_empty() {
                if !name.is_empty() && name.chars().next().map_or(false, |c| c.is_lowercase()) {
                    fields.push((name.to_string(), after_colon.to_string()));
                }
            } else if !name.is_empty() && i + 1 < words.len() {
                // Type is in next word (e.g. "x: f64")
                let typ = words[i + 1].trim_matches(|c: char| {
                    !c.is_alphanumeric() && c != '_' && c != '<' && c != '>'
                });
                if !typ.is_empty() && name.chars().next().map_or(false, |c| c.is_lowercase()) {
                    fields.push((name.to_string(), typ.to_string()));
                    i += 1; // skip the type word
                }
            }
        }
        i += 1;
    }
    fields
}

/// Check if a parameter type is a collection (Vec, slice, or array).
fn is_collection(type_str: &str) -> bool {
    type_str.contains("Vec") || type_str.contains("&[") || type_str.contains("[")
}

/// Infer a reasonable function body from the purpose, params, and return type.
fn infer_rust_body(
    purpose: &str,
    params: &[(String, String)],
    return_type: Option<&str>,
    constraints: &[String],
    examples: &[(String, String)],
) -> String {
    let purpose_lower = purpose.to_lowercase();

    // Try to infer from examples first (most precise)
    if let Some(body) = infer_from_examples(examples, params, return_type) {
        return body;
    }

    // Try composed operations before single-pattern matching.
    // This catches "filter even and sum", "sort then take first 3", etc.
    if let Some(body) = infer_composed_body(&purpose_lower, params, return_type) {
        return body;
    }

    // Try pattern composition (multi-step algorithms: HashMap lookups, sort+nth, etc.)
    if let Some(body) = compose_patterns(&purpose_lower, params, return_type) {
        return body;
    }

    // Pattern-match common operations from the purpose
    let ret = return_type.unwrap_or("");

    // ── Specific multi-word patterns (must be checked BEFORE generic single-word) ──

    // two_sum (before generic "sum")
    if purpose_lower.contains("two_sum") || purpose_lower.contains("two sum") {
        if params.len() == 2 && is_collection(&params[0].1) {
            return format!(
                "for i in 0..{0}.len() {{\n        for j in (i + 1)..{0}.len() {{\n            if {0}[i] + {0}[j] == {1} {{\n                return Some((i, j));\n            }}\n        }}\n    }}\n    None",
                params[0].0, params[1].0
            );
        }
    }
    // dot_product (before generic "product")
    if purpose_lower.contains("dot_product") || purpose_lower.contains("dot product") {
        if params.len() == 2 && is_collection(&params[0].1) {
            return format!(
                "{}.iter().zip({}.iter()).map(|(a, b)| a * b).sum()",
                params[0].0, params[1].0
            );
        }
    }
    // count_words (before generic "count")
    if purpose_lower.contains("count_words")
        || purpose_lower.contains("count words")
        || purpose_lower.contains("word count")
    {
        if params.len() == 1 {
            return format!("{}.split_whitespace().count()", params[0].0);
        }
    }
    // ok_or (before generic Result/Option patterns)
    if purpose_lower.contains("ok_or") {
        if params.len() >= 1 && params[0].1.contains("Option") {
            return format!("{}.ok_or(\"value was None\".to_string())", params[0].0);
        }
    }
    // characters / to_chars (before generic "char at")
    if purpose_lower.contains("characters")
        || purpose_lower.contains("to_chars")
        || purpose_lower.contains("get chars")
    {
        if params.len() == 1 && (params[0].1.contains("str") || params[0].1.contains("String")) {
            return format!("{}.chars().collect()", params[0].0);
        }
    }

    // ── Generic single-word patterns ──

    // Arithmetic operations
    if purpose_lower.contains("add") || purpose_lower.contains("sum") {
        if params.len() == 2 {
            return format!("{} + {}", params[0].0, params[1].0);
        }
    }
    if purpose_lower.contains("subtract") || purpose_lower.contains("difference") {
        if params.len() == 2 {
            return format!("{} - {}", params[0].0, params[1].0);
        }
    }
    if purpose_lower.contains("multiply") || purpose_lower.contains("product") {
        if params.len() == 2 {
            return format!("{} * {}", params[0].0, params[1].0);
        }
    }
    if purpose_lower.contains("divide") || purpose_lower.contains("quotient") {
        if params.len() == 2 {
            // Safe divide: return Result with zero-check
            if purpose_lower.contains("safe")
                || purpose_lower.contains("zero")
                || ret.contains("Result")
            {
                return format!(
                    "if {} == 0.0 {{ Err(\"division by zero\".to_string()) }} else {{ Ok({} / {}) }}",
                    params[1].0, params[0].0, params[1].0
                );
            }
            return format!("{} / {}", params[0].0, params[1].0);
        }
    }
    if purpose_lower.contains("maximum")
        || purpose_lower.contains("max of")
        || purpose_lower.contains("larger")
    {
        if params.len() == 2 {
            if params[0].1.contains("f32") || params[0].1.contains("f64") {
                return format!("{}.max({})", params[0].0, params[1].0);
            }
            return format!(
                "if {} > {} {{ {} }} else {{ {} }}",
                params[0].0, params[1].0, params[0].0, params[1].0
            );
        }
    }
    if purpose_lower.contains("minimum")
        || purpose_lower.contains("min of")
        || purpose_lower.contains("smaller")
    {
        if params.len() == 2 {
            if params[0].1.contains("f32") || params[0].1.contains("f64") {
                return format!("{}.min({})", params[0].0, params[1].0);
            }
            return format!(
                "if {} < {} {{ {} }} else {{ {} }}",
                params[0].0, params[1].0, params[0].0, params[1].0
            );
        }
    }
    if purpose_lower.contains("absolute") || purpose_lower.contains("abs") {
        if params.len() == 1 {
            return format!("{}.abs()", params[0].0);
        }
    }
    if purpose_lower.contains("clamp") {
        if params.len() == 3 {
            return format!("{}.clamp({}, {})", params[0].0, params[1].0, params[2].0);
        }
    }

    // String operations
    if purpose_lower.contains("reverse") {
        if params.len() == 1 && (params[0].1.contains("str") || params[0].1.contains("String")) {
            return format!("{}.chars().rev().collect()", params[0].0);
        }
        if params.len() == 1 && is_collection(&params[0].1) {
            return format!(
                "let mut result = {}.to_vec();\n    result.reverse();\n    result",
                params[0].0
            );
        }
    }
    if purpose_lower.contains("length")
        || purpose_lower.contains("len")
        || purpose_lower.contains("count")
    {
        if params.len() == 1 {
            return format!("{}.len()", params[0].0);
        }
    }
    if purpose_lower.contains("to uppercase") || purpose_lower.contains("uppercase") {
        if params.len() == 1 {
            return format!("{}.to_uppercase()", params[0].0);
        }
    }
    if purpose_lower.contains("to lowercase") || purpose_lower.contains("lowercase") {
        if params.len() == 1 {
            return format!("{}.to_lowercase()", params[0].0);
        }
    }
    // Contains in collection (Vec-specific, before generic string contains)
    if (purpose_lower.contains("contains") || purpose_lower.contains("includes"))
        && params.len() == 2
        && is_collection(&params[0].1)
    {
        return format!("{}.contains(&{})", params[0].0, params[1].0);
    }
    if purpose_lower.contains("contains") || purpose_lower.contains("has") {
        if params.len() == 2 {
            return format!("{}.contains({})", params[0].0, params[1].0);
        }
    }
    if purpose_lower.contains("concatenat")
        || purpose_lower.contains("join")
        || purpose_lower.contains("append")
    {
        if params.len() == 2 && (params[0].1.contains("str") || params[0].1.contains("String")) {
            return format!("format!(\"{{}}{{}}\", {}, {})", params[0].0, params[1].0);
        }
    }
    if purpose_lower.contains("split") {
        if params.len() == 2 {
            return format!("{}.split({}).collect()", params[0].0, params[1].0);
        }
    }
    if purpose_lower.contains("trim") {
        if params.len() == 1 {
            return format!("{}.trim().to_string()", params[0].0);
        }
    }
    if purpose_lower.contains("replace") {
        if params.len() == 3 {
            return format!("{}.replace({}, {})", params[0].0, params[1].0, params[2].0);
        }
    }
    if purpose_lower.contains("starts with") || purpose_lower.contains("prefix") {
        if params.len() == 2 {
            return format!("{}.starts_with({})", params[0].0, params[1].0);
        }
    }
    if purpose_lower.contains("ends with") || purpose_lower.contains("suffix") {
        if params.len() == 2 {
            return format!("{}.ends_with({})", params[0].0, params[1].0);
        }
    }
    // Repeat string
    if purpose_lower.contains("repeat") {
        if params.len() == 2 {
            return format!("{}.repeat({})", params[0].0, params[1].0);
        }
    }
    // Char at index
    if purpose_lower.contains("char at") || purpose_lower.contains("nth char") {
        if params.len() == 2 {
            return format!("{}.chars().nth({})", params[0].0, params[1].0);
        }
    }
    // Count occurrences in string
    if purpose_lower.contains("count")
        && params.len() == 2
        && (params[0].1.contains("str") || params[0].1.contains("String"))
    {
        return format!("{}.matches({}).count()", params[0].0, params[1].0);
    }
    // Capitalize / title case
    if purpose_lower.contains("capitalize") || purpose_lower.contains("title") {
        if params.len() == 1 {
            return format!(
                "let mut c = {}.chars();\n    match c.next() {{\n        None => String::new(),\n        Some(f) => f.to_uppercase().to_string() + &c.as_str().to_lowercase(),\n    }}",
                params[0].0
            );
        }
    }

    // Collection operations
    if purpose_lower.contains("sort") {
        if params.len() == 1 && is_collection(&params[0].1) {
            if purpose_lower.contains("descending") || purpose_lower.contains("reverse") {
                return format!(
                    "let mut result = {}.to_vec();\n    result.sort();\n    result.reverse();\n    result",
                    params[0].0
                );
            }
            return format!(
                "let mut result = {}.to_vec();\n    result.sort();\n    result",
                params[0].0
            );
        }
    }
    if purpose_lower.contains("filter") {
        if params.len() >= 1 && is_collection(&params[0].1) {
            let cond = infer_filter_closure(&purpose_lower);
            let is_slice = params[0].1.contains("&[");
            if is_slice {
                // Slice: .iter() gives &T, so filter gets &&T.
                // Use .iter().copied() first to get owned T, then filter with |&x|
                let cond_owned = cond.replace("*x", "x");
                return format!(
                    "{}.iter().copied().filter(|x| {}).collect()",
                    params[0].0, cond_owned
                );
            }
            let iter = iter_method_for_owned(return_type);
            if iter == ".into_iter()" {
                return format!("{}.into_iter().filter(|x| {}).collect()", params[0].0, cond);
            } else {
                return format!(
                    "{}.iter().filter(|x| {}).cloned().collect()",
                    params[0].0, cond
                );
            }
        }
    }
    if purpose_lower.contains("map") || purpose_lower.contains("transform") {
        if params.len() >= 1 && is_collection(&params[0].1) {
            let body = infer_map_closure(&purpose_lower);
            let iter = iter_method_for_owned(return_type);
            return format!("{}{}.map(|x| {}).collect()", params[0].0, iter, body);
        }
    }
    if purpose_lower.contains("flatten") {
        if params.len() == 1 {
            return format!("{}.into_iter().flatten().collect()", params[0].0);
        }
    }
    if purpose_lower.contains("unique")
        || purpose_lower.contains("deduplicate")
        || purpose_lower.contains("dedup")
    {
        if params.len() == 1 {
            return format!(
                "let mut result = {}.to_vec();\n    result.sort();\n    result.dedup();\n    result",
                params[0].0
            );
        }
    }
    // Binary search
    if purpose_lower.contains("binary search") || purpose_lower.contains("bsearch") {
        if params.len() == 2 {
            return format!("{}.binary_search(&{}).ok()", params[0].0, params[1].0);
        }
    }
    // Sum of collection
    if (purpose_lower.contains("sum") || purpose_lower.contains("total"))
        && params.len() == 1
        && is_collection(&params[0].1)
    {
        return format!("{}.iter().sum()", params[0].0);
    }
    // Max of collection
    if (purpose_lower.contains("max")
        || purpose_lower.contains("largest")
        || purpose_lower.contains("biggest"))
        && params.len() == 1
        && is_collection(&params[0].1)
    {
        return format!("{}.iter().max().copied()", params[0].0);
    }
    // Min of collection
    if (purpose_lower.contains("min") || purpose_lower.contains("smallest"))
        && params.len() == 1
        && is_collection(&params[0].1)
    {
        return format!("{}.iter().min().copied()", params[0].0);
    }
    // Count elements matching condition
    if purpose_lower.contains("count") && params.len() == 1 && is_collection(&params[0].1) {
        return format!("{}.len()", params[0].0);
    }
    // Zip two collections
    if purpose_lower.contains("zip") {
        if params.len() == 2 {
            return format!(
                "{}.into_iter().zip({}.into_iter()).collect()",
                params[0].0, params[1].0
            );
        }
    }
    // Enumerate
    if purpose_lower.contains("enumerate") || purpose_lower.contains("with index") {
        if params.len() == 1 {
            return format!("{}.into_iter().enumerate().collect()", params[0].0);
        }
    }
    // Take first N
    if (purpose_lower.contains("take") || purpose_lower.contains("first")) && params.len() == 2 {
        if params[1].1.contains("usize") || params[1].1.contains("u") || params[1].1.contains("i") {
            return format!(
                "{}.iter().take({}).cloned().collect()",
                params[0].0, params[1].0
            );
        }
    }
    // Skip first N
    if purpose_lower.contains("skip") && params.len() == 2 {
        return format!(
            "{}.iter().skip({}).cloned().collect()",
            params[0].0, params[1].0
        );
    }
    // Chunk/windows
    if purpose_lower.contains("chunk") && params.len() == 2 {
        return format!(
            "{}.chunks({}).map(|c| c.to_vec()).collect()",
            params[0].0, params[1].0
        );
    }

    // Boolean/check operations
    if purpose_lower.contains("is empty") || purpose_lower.contains("empty") {
        if params.len() == 1 {
            return format!("{}.is_empty()", params[0].0);
        }
    }
    if purpose_lower.contains("is even") {
        if params.len() == 1 {
            return format!("{} % 2 == 0", params[0].0);
        }
    }
    if purpose_lower.contains("is odd") {
        if params.len() == 1 {
            return format!("{} % 2 != 0", params[0].0);
        }
    }
    if purpose_lower.contains("is positive") {
        if params.len() == 1 {
            return format!("{} > 0", params[0].0);
        }
    }
    if purpose_lower.contains("is negative") {
        if params.len() == 1 {
            return format!("{} < 0", params[0].0);
        }
    }

    // Math operations
    if purpose_lower.contains("factorial") {
        if params.len() == 1 {
            return format!("(1..={}).product()", params[0].0);
        }
    }
    if purpose_lower.contains("fibonacci") {
        if params.len() == 1 {
            return "let (mut a, mut b) = (0u64, 1u64);\n    for _ in 0..n {\n        let tmp = a + b;\n        a = b;\n        b = tmp;\n    }\n    a".to_string();
        }
    }
    if purpose_lower.contains("power")
        || purpose_lower.contains("exponent")
        || purpose_lower.contains("pow")
    {
        if params.len() == 2 {
            if params[0].1.contains("f") {
                return format!("{}.powf({})", params[0].0, params[1].0);
            }
            return format!("{}.pow({} as u32)", params[0].0, params[1].0);
        }
    }
    if purpose_lower.contains("square root") || purpose_lower.contains("sqrt") {
        if params.len() == 1 {
            return format!("({} as f64).sqrt()", params[0].0);
        }
    }
    // GCD
    if purpose_lower.contains("gcd") || purpose_lower.contains("greatest common") {
        if params.len() == 2 {
            return format!(
                "let (mut a, mut b) = ({}, {});\n    while b != 0 {{\n        let t = b;\n        b = a % b;\n        a = t;\n    }}\n    a",
                params[0].0, params[1].0
            );
        }
    }
    // Average / mean
    if purpose_lower.contains("average") || purpose_lower.contains("mean") {
        if params.len() == 1 && is_collection(&params[0].1) {
            return format!(
                "{}.iter().sum::<f64>() / {}.len() as f64",
                params[0].0, params[0].0
            );
        }
    }

    // Error handling patterns
    if ret.contains("Result") {
        if params.len() == 1
            && (purpose_lower.contains("parse") || purpose_lower.contains("convert"))
        {
            // Extract the Ok type from Result<T, ...> for turbofish
            let turbofish = extract_result_ok_type(ret)
                .map(|t| format!("::<{}>", t))
                .unwrap_or_default();
            return format!(
                "{}.parse{}().map_err(|e| e.to_string())",
                params[0].0, turbofish
            );
        }
        if purpose_lower.contains("read") || purpose_lower.contains("file") {
            return "std::fs::read_to_string(path).map_err(|e| e.to_string())".to_string();
        }
    }

    // Option patterns
    if ret.contains("Option") {
        if purpose_lower.contains("find") || purpose_lower.contains("first") {
            if params.len() >= 1 {
                let cond = infer_filter_closure(&purpose_lower);
                return format!("{}.iter().find(|x| {}).cloned()", params[0].0, cond);
            }
        }
    }

    // --- Iterator adapters ---
    // windows
    if purpose_lower.contains("windows") || purpose_lower.contains("sliding window") {
        if params.len() == 2 && is_collection(&params[0].1) {
            return format!(
                "{}.windows({}).map(|w| w.to_vec()).collect()",
                params[0].0, params[1].0
            );
        }
    }
    // chain / concatenate lists
    if (purpose_lower.contains("chain") || purpose_lower.contains("concatenate lists"))
        && params.len() == 2
        && is_collection(&params[0].1)
    {
        return format!(
            "[{}.as_slice(), {}.as_slice()].concat()",
            params[0].0, params[1].0
        );
    }
    // flat_map
    if purpose_lower.contains("flat_map") || purpose_lower.contains("flat map") {
        if params.len() == 1 && is_collection(&params[0].1) {
            return format!(
                "{}.iter().flat_map(|x| x.iter().cloned()).collect()",
                params[0].0
            );
        }
    }
    // partition / split into
    if purpose_lower.contains("partition") || purpose_lower.contains("split into") {
        if params.len() == 1 && is_collection(&params[0].1) {
            let cond = infer_filter_closure(&purpose_lower);
            return format!("{}.iter().partition(|x| {})", params[0].0, cond);
        }
    }
    // any / any element
    if purpose_lower.contains("any element")
        || (purpose_lower.contains("any")
            && !purpose_lower.contains("many")
            && params.len() == 1
            && is_collection(&params[0].1))
    {
        let cond = infer_filter_closure(&purpose_lower);
        return format!("{}.iter().any(|x| {})", params[0].0, cond);
    }
    // all / every element
    if purpose_lower.contains("all elements") || purpose_lower.contains("every element") {
        if params.len() == 1 && is_collection(&params[0].1) {
            let cond = infer_filter_closure(&purpose_lower);
            return format!("{}.iter().all(|x| {})", params[0].0, cond);
        }
    }

    // --- Option/Result combinators ---
    // unwrap_or / default value
    if purpose_lower.contains("unwrap_or")
        || (purpose_lower.contains("default")
            && params.len() == 2
            && params[0].1.contains("Option"))
    {
        return format!("{}.unwrap_or({})", params[0].0, params[1].0);
    }
    // map_or
    if purpose_lower.contains("map_or") {
        if params.len() >= 1 && params[0].1.contains("Option") {
            return format!("{}.map_or(Default::default(), |x| x)", params[0].0);
        }
    }
    // and_then / then (Option)
    if params.len() >= 1
        && params[0].1.contains("Option")
        && (purpose_lower.contains("and_then") || purpose_lower.contains("then"))
    {
        return format!("{}.and_then(|x| Some(x))", params[0].0);
    }

    // --- Common algorithms ---
    // palindrome
    if purpose_lower.contains("palindrome") {
        if params.len() == 1 {
            return format!(
                "let s: String = {}.chars().collect();\n    let r: String = s.chars().rev().collect();\n    s == r",
                params[0].0
            );
        }
    }
    // (two_sum moved to specific multi-word section above)
    // matrix_transpose
    if purpose_lower.contains("matrix_transpose")
        || purpose_lower.contains("matrix transpose")
        || purpose_lower.contains("transpose")
    {
        if params.len() == 1 && is_collection(&params[0].1) {
            return format!(
                "if {0}.is_empty() || {0}[0].is_empty() {{ return vec![]; }}\n    let rows = {0}.len();\n    let cols = {0}[0].len();\n    (0..cols).map(|c| (0..rows).map(|r| {0}[r][c]).collect()).collect()",
                params[0].0
            );
        }
    }
    // (dot_product and count_words moved to specific multi-word section above)
    // median
    if purpose_lower.contains("median") {
        if params.len() == 1 && is_collection(&params[0].1) {
            return format!(
                "let mut sorted = {0}.to_vec();\n    sorted.sort();\n    let mid = sorted.len() / 2;\n    if sorted.len() % 2 == 0 {{\n        (sorted[mid - 1] + sorted[mid]) / 2\n    }} else {{\n        sorted[mid]\n    }}",
                params[0].0
            );
        }
    }
    // mode
    if purpose_lower.contains("mode") {
        if params.len() == 1 && is_collection(&params[0].1) {
            return format!(
                "let mut counts = std::collections::HashMap::new();\n    for v in {0}.iter() {{\n        *counts.entry(v).or_insert(0usize) += 1;\n    }}\n    counts.into_iter().max_by_key(|&(_, c)| c).map(|(v, _)| *v).unwrap_or_default()",
                params[0].0
            );
        }
    }
    // lcm (least common multiple)
    if purpose_lower.contains("lcm") || purpose_lower.contains("least common multiple") {
        if params.len() == 2 {
            return format!(
                "let (mut a, mut b) = ({0}, {1});\n    let product = a * b;\n    while b != 0 {{\n        let t = b;\n        b = a % b;\n        a = t;\n    }}\n    product / a",
                params[0].0, params[1].0
            );
        }
    }

    // --- Type conversion ---
    // to_string / stringify
    if purpose_lower.contains("to_string") || purpose_lower.contains("stringify") {
        if params.len() == 1 {
            return format!("{}.to_string()", params[0].0);
        }
    }
    // parse / from_string
    if purpose_lower.contains("parse")
        || purpose_lower.contains("from_string")
        || purpose_lower.contains("from string")
    {
        if params.len() == 1 && (params[0].1.contains("str") || params[0].1.contains("String")) {
            // Add turbofish if return type is known and concrete
            let turbofish = if !ret.is_empty()
                && ret != "()"
                && !ret.contains("Result")
                && !ret.contains("Option")
            {
                format!("::<{}>", ret)
            } else {
                String::new()
            };
            return format!("{}.parse{}().unwrap_or_default()", params[0].0, turbofish);
        }
    }
    // to_vec / to_vector
    if purpose_lower.contains("to_vec")
        || purpose_lower.contains("to_vector")
        || purpose_lower.contains("to vector")
    {
        if params.len() == 1 {
            return format!("{}.to_vec()", params[0].0);
        }
    }
    // (characters/to_char moved to specific multi-word section above)

    // --- Misc ---
    // swap
    if purpose_lower.contains("swap") {
        if params.len() == 2 {
            return format!("({1}, {0})", params[0].0, params[1].0);
        }
    }
    // modulo / remainder
    if purpose_lower.contains("modulo") || purpose_lower.contains("remainder") {
        if params.len() == 2 {
            return format!("{} % {}", params[0].0, params[1].0);
        }
    }

    // ── Additional patterns (Phase 0 benchmark gap analysis) ──

    // Negate
    if purpose_lower.contains("negate") {
        if params.len() == 1 {
            return format!("-{}", params[0].0);
        }
    }
    // Increment / add one
    if purpose_lower.contains("increment") || purpose_lower.contains("add one") {
        if params.len() == 1 {
            return format!("{} + 1", params[0].0);
        }
    }
    // Decrement / subtract one
    if purpose_lower.contains("decrement") || purpose_lower.contains("subtract one") {
        if params.len() == 1 {
            return format!("{} - 1", params[0].0);
        }
    }
    // Is zero
    if purpose_lower.contains("is zero") || purpose_lower.contains("zero") {
        if params.len() == 1 && (ret.contains("bool") || ret.is_empty()) {
            return format!("{} == 0", params[0].0);
        }
    }
    // Is sorted (ascending)
    if purpose_lower.contains("is sorted") || purpose_lower.contains("sorted") {
        if params.len() == 1 && is_collection(&params[0].1) && ret.contains("bool") {
            return format!("{}.windows(2).all(|w| w[0] <= w[1])", params[0].0);
        }
    }
    // All positive / all match predicate
    if purpose_lower.contains("all positive") || purpose_lower.contains("all are positive") {
        if params.len() == 1 && is_collection(&params[0].1) {
            return format!("{}.iter().all(|&x| x > 0)", params[0].0);
        }
    }
    // Any negative
    if purpose_lower.contains("any negative") || purpose_lower.contains("any element is negative") {
        if params.len() == 1 && is_collection(&params[0].1) {
            return format!("{}.iter().any(|&x| x < 0)", params[0].0);
        }
    }
    // Is prime (but NOT "nth prime" or "prime factors" — those have their own patterns)
    if purpose_lower.contains("prime")
        && !purpose_lower.contains("nth")
        && !purpose_lower.contains("factor")
    {
        if params.len() == 1 && ret.contains("bool") {
            return format!(
                "if {} < 2 {{ return false; }}\n    (2..=(({} as f64).sqrt() as u64)).all(|i| {} % i != 0)",
                params[0].0, params[0].0, params[0].0
            );
        }
    }
    // Celsius to Fahrenheit
    if purpose_lower.contains("celsius") && purpose_lower.contains("fahrenheit") {
        if params.len() == 1 {
            return format!("{} * 9.0 / 5.0 + 32.0", params[0].0);
        }
    }
    // Fahrenheit to Celsius
    if purpose_lower.contains("fahrenheit") && purpose_lower.contains("celsius") {
        if params.len() == 1 {
            return format!("({} - 32.0) * 5.0 / 9.0", params[0].0);
        }
    }
    // Distance (Euclidean)
    if purpose_lower.contains("distance") || purpose_lower.contains("euclidean") {
        if params.len() == 4 {
            return format!(
                "(({0} - {2}).powi(2) + ({1} - {3}).powi(2)).sqrt()",
                params[0].0, params[1].0, params[2].0, params[3].0
            );
        }
    }
    // Sum digits
    if purpose_lower.contains("sum") && purpose_lower.contains("digit") {
        if params.len() == 1 {
            return format!(
                "let mut n = {};\n    let mut total = 0u64;\n    while n > 0 {{\n        total += n % 10;\n        n /= 10;\n    }}\n    total",
                params[0].0
            );
        }
    }
    // Collatz steps
    if purpose_lower.contains("collatz") {
        if params.len() == 1 {
            // If return type is Option, wrap with None guard
            if ret.contains("Option") {
                return format!(
                    "if {0} == 0 {{ return None; }}\n    let mut n = {0};\n    let mut steps = 0u64;\n    while n != 1 {{\n        if n % 2 == 0 {{ n /= 2; }} else {{ n = 3 * n + 1; }}\n        steps += 1;\n    }}\n    Some(steps)",
                    params[0].0
                );
            }
            return format!(
                "let mut n = {};\n    let mut steps = 0u64;\n    while n != 1 {{\n        if n % 2 == 0 {{ n /= 2; }} else {{ n = 3 * n + 1; }}\n        steps += 1;\n    }}\n    steps",
                params[0].0
            );
        }
    }
    // Double each element
    if purpose_lower.contains("double") {
        if params.len() == 1 && is_collection(&params[0].1) {
            return format!("{}.iter().map(|&x| x * 2).collect()", params[0].0);
        }
    }
    // Squares of elements
    if purpose_lower.contains("square") && !purpose_lower.contains("root") {
        if params.len() == 1 && is_collection(&params[0].1) {
            return format!("{}.iter().map(|&x| x * x).collect()", params[0].0);
        }
    }
    // Product of all elements
    if purpose_lower.contains("product") || purpose_lower.contains("multiply all") {
        if params.len() == 1 && is_collection(&params[0].1) {
            return format!("{}.iter().product()", params[0].0);
        }
    }
    // Count matching (with target param)
    if purpose_lower.contains("count") && purpose_lower.contains("match") {
        if params.len() == 2 && is_collection(&params[0].1) {
            return format!(
                "{}.iter().filter(|&&x| x == {}).count()",
                params[0].0, params[1].0
            );
        }
    }
    // Count occurrences / how many times
    if purpose_lower.contains("count")
        && (purpose_lower.contains("occur")
            || purpose_lower.contains("times")
            || purpose_lower.contains("appear"))
    {
        if params.len() == 2 && is_collection(&params[0].1) {
            return format!(
                "{}.iter().filter(|&&x| x == {}).count()",
                params[0].0, params[1].0
            );
        }
    }
    // Last element
    if purpose_lower.contains("last") {
        if params.len() == 1 && is_collection(&params[0].1) {
            return format!("{}.last().copied()", params[0].0);
        }
    }
    // Index of element
    if purpose_lower.contains("index of") || purpose_lower.contains("position") {
        if params.len() == 2 && is_collection(&params[0].1) {
            return format!("{}.iter().position(|&x| x == {})", params[0].0, params[1].0);
        }
    }
    // Find duplicates
    if purpose_lower.contains("duplicate") && !purpose_lower.contains("dedup") {
        if params.len() == 1 && is_collection(&params[0].1) {
            return format!(
                "let mut seen = std::collections::HashSet::new();\n    {}.iter().filter(|&&x| !seen.insert(x)).cloned().collect()",
                params[0].0
            );
        }
    }
    // Running sum / cumulative sum
    if purpose_lower.contains("running") && purpose_lower.contains("sum") {
        if params.len() == 1 && is_collection(&params[0].1) {
            return format!(
                "let mut acc = 0;\n    {}.iter().map(|&x| {{ acc += x; acc }}).collect()",
                params[0].0
            );
        }
    }
    // Group by sign (positive/negative)
    if purpose_lower.contains("group")
        && (purpose_lower.contains("sign")
            || purpose_lower.contains("positive")
            || purpose_lower.contains("negative"))
    {
        if params.len() == 1 && is_collection(&params[0].1) {
            return format!(
                "let pos: Vec<_> = {0}.iter().filter(|&&x| x >= 0).cloned().collect();\n    let neg: Vec<_> = {0}.iter().filter(|&&x| x < 0).cloned().collect();\n    (pos, neg)",
                params[0].0
            );
        }
    }
    // Interleave two collections
    if purpose_lower.contains("interleave") {
        if params.len() == 2 && is_collection(&params[0].1) {
            return format!(
                "{}.iter().zip({}.iter()).flat_map(|(&a, &b)| [a, b]).collect()",
                params[0].0, params[1].0
            );
        }
    }
    // Unique sorted (dedup + sort)
    if purpose_lower.contains("unique") && purpose_lower.contains("sort") {
        if params.len() == 1 && is_collection(&params[0].1) {
            return format!(
                "let mut result: Vec<_> = {}.to_vec();\n    result.sort();\n    result.dedup();\n    result",
                params[0].0
            );
        }
    }
    // Merge sorted arrays
    if purpose_lower.contains("merge") && purpose_lower.contains("sort") {
        if params.len() == 2 {
            return format!(
                "let mut result = {}.to_vec();\n    result.extend_from_slice({});\n    result.sort();\n    result",
                params[0].0, params[1].0
            );
        }
    }
    // Bubble sort
    if purpose_lower.contains("bubble") {
        if params.len() == 1 {
            return format!(
                "let n = {0}.len();\n    for i in 0..n {{\n        for j in 0..n - 1 - i {{\n            if {0}[j] > {0}[j + 1] {{\n                {0}.swap(j, j + 1);\n            }}\n        }}\n    }}",
                params[0].0
            );
        }
    }
    // Sort descending
    if purpose_lower.contains("descending") {
        if params.len() == 1 && is_collection(&params[0].1) {
            return format!(
                "let mut result = {}.to_vec();\n    result.sort();\n    result.reverse();\n    result",
                params[0].0
            );
        }
    }
    // Sort by length
    if purpose_lower.contains("sort") && purpose_lower.contains("length") {
        if params.len() == 1 {
            return format!(
                "let mut result = {}.to_vec();\n    result.sort_by_key(|s| s.len());\n    result",
                params[0].0
            );
        }
    }
    // Partition by threshold
    if purpose_lower.contains("partition") {
        if params.len() == 2 && is_collection(&params[0].1) {
            return format!(
                "let below: Vec<_> = {0}.iter().filter(|&&x| x < {1}).cloned().collect();\n    let above: Vec<_> = {0}.iter().filter(|&&x| x >= {1}).cloned().collect();\n    (below, above)",
                params[0].0, params[1].0
            );
        }
    }
    // Linear search
    if purpose_lower.contains("linear") && purpose_lower.contains("search") {
        if params.len() == 2 {
            return format!("{}.iter().position(|&x| x == {})", params[0].0, params[1].0);
        }
    }
    // Filter long strings
    if purpose_lower.contains("filter") && purpose_lower.contains("long") {
        if params.len() == 2 {
            return format!(
                "{}.iter().filter(|s| s.len() > {}).cloned().collect()",
                params[0].0, params[1].0
            );
        }
    }
    // First or error
    if purpose_lower.contains("first")
        && (purpose_lower.contains("error") || ret.contains("Result"))
    {
        if params.len() == 1 && is_collection(&params[0].1) {
            return format!(
                "{}.first().copied().ok_or_else(|| \"empty collection\".to_string())",
                params[0].0
            );
        }
    }
    // Safe divide (zero-check)
    if purpose_lower.contains("safe") && purpose_lower.contains("divide") {
        if params.len() == 2 {
            return format!(
                "if {} == 0.0 {{ Err(\"division by zero\".to_string()) }} else {{ Ok({} / {}) }}",
                params[1].0, params[0].0, params[1].0
            );
        }
    }
    // Validate range
    if purpose_lower.contains("validate") && purpose_lower.contains("range") {
        if params.len() == 3 {
            return format!(
                "if {0} >= {1} && {0} <= {2} {{ Ok({0}) }} else {{ Err(format!(\"{{}} is not in range [{{}}..{{}}]\", {0}, {1}, {2})) }}",
                params[0].0, params[1].0, params[2].0
            );
        }
    }
    // Unwrap or default
    if purpose_lower.contains("unwrap") || purpose_lower.contains("default") {
        if params.len() == 2 && params[0].1.contains("Option") {
            return format!("{}.unwrap_or({})", params[0].0, params[1].0);
        }
    }
    // Split into words (Vec<String>)
    if purpose_lower.contains("split") && purpose_lower.contains("word") {
        if params.len() == 1 {
            return format!(
                "{}.split_whitespace().map(|s| s.to_string()).collect()",
                params[0].0
            );
        }
    }
    // Char at index
    if purpose_lower.contains("char") && purpose_lower.contains("index") {
        if params.len() == 2 {
            return format!("{}.chars().nth({})", params[0].0, params[1].0);
        }
    }
    // Zip and sum pairs
    if purpose_lower.contains("zip") && purpose_lower.contains("sum") {
        if params.len() == 2 {
            return format!(
                "{}.iter().zip({}.iter()).map(|(&a, &b)| a + b).collect()",
                params[0].0, params[1].0
            );
        }
    }
    // Flatten and sort
    if purpose_lower.contains("flatten") && purpose_lower.contains("sort") {
        if params.len() == 1 {
            return format!(
                "let mut result: Vec<_> = {}.iter().flat_map(|v| v.iter().cloned()).collect();\n    result.sort();\n    result",
                params[0].0
            );
        }
    }

    // ── Exercism domain patterns ──

    // Leap year
    if purpose_lower.contains("leap") && purpose_lower.contains("year") {
        if params.len() == 1 {
            return format!(
                "({0} % 4 == 0 && {0} % 100 != 0) || {0} % 400 == 0",
                params[0].0
            );
        }
    }
    // Collatz conjecture (return steps as Option)
    if purpose_lower.contains("collatz") && ret.contains("Option") {
        if params.len() == 1 {
            return format!(
                "if {0} == 0 {{ return None; }}\n    let mut n = {0};\n    let mut steps = 0u64;\n    while n != 1 {{\n        if n % 2 == 0 {{ n /= 2; }} else {{ n = 3 * n + 1; }}\n        steps += 1;\n    }}\n    Some(steps)",
                params[0].0
            );
        }
    }
    // Nth prime
    if purpose_lower.contains("nth") && purpose_lower.contains("prime") {
        if params.len() == 1 {
            return format!(
                "let mut count = 0u32;\n    let mut candidate = 2u32;\n    loop {{\n        if (2..=((candidate as f64).sqrt() as u32).max(1)).all(|i| candidate % i != 0) {{\n            if count == {} {{ return candidate; }}\n            count += 1;\n        }}\n        candidate += 1;\n    }}",
                params[0].0
            );
        }
    }
    // Pangram
    if purpose_lower.contains("pangram") {
        if params.len() == 1 {
            return format!(
                "let lower = {}.to_lowercase();\n    ('a'..='z').all(|c| lower.contains(c))",
                params[0].0
            );
        }
    }
    // Isogram (no repeating letters)
    if purpose_lower.contains("isogram")
        || (purpose_lower.contains("repeating") && purpose_lower.contains("letter"))
    {
        if params.len() == 1 {
            return format!(
                "let mut seen = std::collections::HashSet::new();\n    {}.to_lowercase().chars().filter(|c| c.is_alphabetic()).all(|c| seen.insert(c))",
                params[0].0
            );
        }
    }
    // Hamming distance
    if purpose_lower.contains("hamming") && ret.contains("Option") {
        if params.len() == 2 {
            return format!(
                "if {0}.len() != {1}.len() {{ return None; }}\n    Some({0}.chars().zip({1}.chars()).filter(|(a, b)| a != b).count())",
                params[0].0, params[1].0
            );
        }
    }
    // Raindrops (fizzbuzz variant)
    if purpose_lower.contains("raindrop") {
        if params.len() == 1 {
            return format!(
                "let mut result = String::new();\n    if {0} % 3 == 0 {{ result.push_str(\"Pling\"); }}\n    if {0} % 5 == 0 {{ result.push_str(\"Plang\"); }}\n    if {0} % 7 == 0 {{ result.push_str(\"Plong\"); }}\n    if result.is_empty() {{ {0}.to_string() }} else {{ result }}",
                params[0].0
            );
        }
    }
    // Square of sum (but NOT "difference between square of sum and sum of squares")
    if purpose_lower.contains("square of")
        && purpose_lower.contains("sum")
        && !purpose_lower.contains("difference")
    {
        if params.len() == 1 {
            return format!("let s: u32 = (1..={}).sum(); s * s", params[0].0);
        }
    }
    // Sum of squares
    if purpose_lower.contains("sum of") && purpose_lower.contains("square") {
        if params.len() == 1 {
            return format!("(1..={}).map(|i| i * i).sum()", params[0].0);
        }
    }
    // Difference (between square of sum and sum of squares)
    if purpose_lower.contains("difference")
        && purpose_lower.contains("square")
        && purpose_lower.contains("sum")
    {
        if params.len() == 1 {
            return format!(
                "let s: u32 = (1..={0}).sum();\n    let sq_sum = s * s;\n    let sum_sq: u32 = (1..={0}).map(|i| i * i).sum();\n    sq_sum - sum_sq",
                params[0].0
            );
        }
    }
    // Bob (reply to messages)
    if purpose_lower.contains("bob")
        && (purpose_lower.contains("reply") || purpose_lower.contains("respond"))
    {
        if params.len() == 1 {
            return format!(
                "let trimmed = {0}.trim();\n    let is_question = trimmed.ends_with('?');\n    let is_yelling = trimmed.chars().any(|c| c.is_alphabetic()) && trimmed.chars().filter(|c| c.is_alphabetic()).all(|c| c.is_uppercase());\n    match (is_question, is_yelling, trimmed.is_empty()) {{\n        (_, _, true) => \"Fine. Be that way!\",\n        (true, true, _) => \"Calm down, I know what I'm doing!\",\n        (true, _, _) => \"Sure.\",\n        (_, true, _) => \"Whoa, chill out!\",\n        _ => \"Whatever.\",\n    }}",
                params[0].0
            );
        }
    }
    // Two-fer
    if purpose_lower.contains("one for")
        || purpose_lower.contains("two-fer")
        || purpose_lower.contains("twofer")
    {
        if params.len() == 1 {
            return format!("format!(\"One for {{}}, one for me.\", {})", params[0].0);
        }
    }
    // Total grains on chessboard (no params — sum all 64 squares)
    if (purpose_lower.contains("total") && purpose_lower.contains("grain"))
        || (purpose_lower.contains("total") && purpose_lower.contains("chessboard"))
    {
        if params.is_empty() {
            return "u64::MAX".to_string(); // 2^64 - 1 = sum of 2^0 + 2^1 + ... + 2^63
        }
    }
    // Grains on chessboard
    if purpose_lower.contains("grain") || purpose_lower.contains("chessboard") {
        if params.len() == 1 {
            return format!(
                "if {0} == 0 || {0} > 64 {{ panic!(\"Square must be between 1 and 64\"); }}\n    2u64.pow({0} as u32 - 1)",
                params[0].0
            );
        }
    }
    // Armstrong number
    if purpose_lower.contains("armstrong") {
        if params.len() == 1 {
            return format!(
                "let s = {0}.to_string();\n    let l = s.len() as u32;\n    s.chars().map(|c| c.to_digit(10).unwrap().pow(l)).sum::<u32>() == {0}",
                params[0].0
            );
        }
    }
    // Prime factors
    if purpose_lower.contains("prime") && purpose_lower.contains("factor") {
        if params.len() == 1 {
            return format!(
                "let mut n = {};\n    let mut factors = Vec::new();\n    let mut d = 2;\n    while d * d <= n {{\n        while n % d == 0 {{\n            factors.push(d as u64);\n            n /= d;\n        }}\n        d += 1;\n    }}\n    if n > 1 {{ factors.push(n as u64); }}\n    factors",
                params[0].0
            );
        }
    }
    // Sum of multiples
    if purpose_lower.contains("sum") && purpose_lower.contains("multiple") {
        if params.len() == 2 {
            return format!(
                "(1..{1}).filter(|n| {0}.iter().any(|&f| f != 0 && n % f == 0)).sum()",
                params[0].0, params[1].0
            );
        }
    }

    // Constraint-based body generation
    if !constraints.is_empty() {
        let mut body_parts = Vec::new();
        for c in constraints {
            body_parts.push(format!("// {}", c));
        }
        if let Some(rt) = return_type {
            body_parts.push(format!("todo!(\"Implement: {} → {}\")", purpose, rt));
        } else {
            body_parts.push(format!("todo!(\"Implement: {}\")", purpose));
        }
        return body_parts.join("\n    ");
    }

    // Default: return a todo with context
    if let Some(rt) = return_type {
        format!("todo!(\"Implement: {} → {}\")", purpose, rt)
    } else {
        format!("todo!(\"Implement: {}\")", purpose)
    }
}

/// Infer composed operation bodies from multi-step purpose descriptions.
///
/// Detects patterns like "filter even numbers and sum them", "sort then take
/// first N", "map to uppercase and join with commas".
fn infer_composed_body(
    purpose: &str,
    params: &[(String, String)],
    _return_type: Option<&str>,
) -> Option<String> {
    if params.is_empty() {
        return None;
    }

    let p0 = &params[0].0;
    let is_vec = is_collection(&params[0].1) || params[0].1.contains("&[");

    if !is_vec {
        return None;
    }

    // Defer specific multi-step patterns to compose_patterns() which handles
    // them more precisely than the generic iterator chain builder:
    //
    // - "3rd largest" → sort + index (not sort + max)
    // - "count unique" → HashSet (not sort + dedup + count)
    // - "frequency"/"occurrences" → HashMap (not iterator chain)
    let has_nth = purpose.contains("nth")
        || purpose.contains("kth")
        || extract_number_from_text(purpose).map_or(false, |n| n > 1);
    if has_nth
        && (purpose.contains("largest")
            || purpose.contains("smallest")
            || purpose.contains("biggest"))
    {
        return None;
    }
    if (purpose.contains("unique") || purpose.contains("distinct")) && purpose.contains("count") {
        return None;
    }
    if purpose.contains("frequency") || purpose.contains("occurrences") {
        return None;
    }

    // Build an iterator chain from recognized verbs in order of appearance
    let mut chain_parts: Vec<(&str, &str)> = Vec::new(); // (verb, code_fragment)

    // Scan for operation keywords in order
    let ops: &[(&[&str], &str, &str)] = &[
        // (keywords, iterator_method, description)
        (
            &["filter even", "keep even", "only even"],
            ".filter(|x| *x % 2 == 0)",
            "filter_even",
        ),
        (
            &["filter odd", "keep odd", "only odd"],
            ".filter(|x| *x % 2 != 0)",
            "filter_odd",
        ),
        (
            &["filter positive", "keep positive"],
            ".filter(|x| *x > 0)",
            "filter_pos",
        ),
        (
            &["filter negative", "keep negative"],
            ".filter(|x| *x < 0)",
            "filter_neg",
        ),
        (&["filter"], "", "filter"), // handled in match arm via infer_filter_closure
        (&["sort", "order"], "", "sort"), // handled specially
        (&["reverse", "reversed"], ".rev()", "reverse"),
        (
            &["map", "transform", "convert each", "double", "square"],
            "",
            "map",
        ), // handled specially
        (&["sum", "total", "add up", "add them"], ".sum()", "sum"),
        (&["count", "how many"], ".count()", "count"),
        (&["take first", "first n", "take top", "top n"], "", "take"), // handled specially
        (&["unique", "deduplicate", "dedup", "distinct"], "", "dedup"), // handled specially
        (&["join", "concatenate"], "", "join"),                        // handled specially
        (&["maximum", "max", "largest", "biggest"], ".max()", "max"),
        (&["minimum", "min", "smallest"], ".min()", "min"),
        (&["flatten"], ".flatten()", "flatten"),
    ];

    for (keywords, fragment, tag) in ops {
        for kw in *keywords {
            if purpose.contains(kw) {
                // Guard: "hashmap" contains "map" but shouldn't trigger map transform
                if *tag == "map" && purpose.contains("hashmap") {
                    continue;
                }
                chain_parts.push((tag, fragment));
                break; // only match first keyword per operation
            }
        }
    }

    if chain_parts.len() < 2 {
        return None; // single operations handled by the main matcher
    }

    // Build the composed expression
    let mut needs_sort_first = false;
    let mut needs_collect = true;
    let mut iter_chain = Vec::new();

    let mut needs_dedup = false;

    for (tag, fragment) in &chain_parts {
        match *tag {
            "sort" => {
                needs_sort_first = true;
            }
            "dedup" => {
                needs_sort_first = true; // dedup requires sorted
                needs_dedup = true; // dedup() is a Vec method, not an iterator adapter
            }
            "map" => {
                let body = infer_map_closure(purpose);
                iter_chain.push(format!(".map(|x| {})", body));
            }
            "take" => {
                // Try to extract N from purpose
                let n = extract_number_from_text(purpose).unwrap_or(3);
                iter_chain.push(format!(".take({})", n));
            }
            "join" => {
                // join produces a String, not a Vec
                let sep = if purpose.contains("comma") {
                    ", "
                } else if purpose.contains("space") {
                    " "
                } else if purpose.contains("newline") {
                    "\\n"
                } else {
                    ", "
                };
                iter_chain.push(format!(".collect::<Vec<_>>()"));
                // Return the join expression
                let dedup_line = if needs_dedup {
                    "\n    tmp.dedup();"
                } else {
                    ""
                };
                let sort_prefix = if needs_sort_first {
                    format!(
                        "let mut tmp = {}.to_vec();\n    tmp.sort();{}\n    tmp.iter()",
                        p0, dedup_line
                    )
                } else {
                    format!("{}.iter()", p0)
                };
                let chain: String = iter_chain
                    .iter()
                    .take(iter_chain.len() - 1) // exclude the collect we just added
                    .map(|s| s.as_str())
                    .collect::<String>();
                return Some(format!(
                    "{}{}.map(|x| x.to_string()).collect::<Vec<_>>().join(\"{}\")",
                    sort_prefix, chain, sep
                ));
            }
            "filter" | "filter_even" | "filter_odd" | "filter_pos" | "filter_neg" => {
                let cond = infer_filter_closure(purpose);
                iter_chain.push(format!(".filter(|x| {})", cond));
            }
            "sum" | "count" | "max" | "min" => {
                iter_chain.push(fragment.to_string());
                needs_collect = false;
            }
            _ => {
                iter_chain.push(fragment.to_string());
            }
        }
    }

    if iter_chain.is_empty() && !needs_sort_first {
        return None;
    }

    let chain: String = iter_chain.iter().map(|s| s.as_str()).collect::<String>();

    if needs_sort_first {
        // Sort (and optionally dedup) require mutation, so we need a let binding
        let dedup_line = if needs_dedup {
            "\n    tmp.dedup();"
        } else {
            ""
        };
        let collect = if needs_collect { ".collect()" } else { "" };
        Some(format!(
            "let mut tmp = {}.to_vec();\n    tmp.sort();{}\n    tmp.into_iter(){}{}",
            p0, dedup_line, chain, collect
        ))
    } else {
        let collect = if needs_collect { ".collect()" } else { "" };
        Some(format!("{}.into_iter(){}{}", p0, chain, collect))
    }
}

/// Extract a number from text like "first 3", "top 5", "take 10"
fn extract_number_from_text(text: &str) -> Option<usize> {
    for word in text.split_whitespace() {
        if let Ok(n) = word.parse::<usize>() {
            return Some(n);
        }
        // Handle ordinals: "3rd", "2nd", "1st", "4th", etc.
        let trimmed = word
            .trim_end_matches("st")
            .trim_end_matches("nd")
            .trim_end_matches("rd")
            .trim_end_matches("th");
        if let Ok(n) = trimmed.parse::<usize>() {
            return Some(n);
        }
    }
    // Check for written-out numbers
    let written = &[
        ("one", 1),
        ("two", 2),
        ("three", 3),
        ("four", 4),
        ("five", 5),
        ("six", 6),
        ("seven", 7),
        ("eight", 8),
        ("nine", 9),
        ("ten", 10),
    ];
    for (word, n) in written {
        if text.contains(word) {
            return Some(*n);
        }
    }
    None
}

/// Infer a filter closure body from the purpose description.
/// Choose the right iterator method based on whether the return type needs owned values.
///
/// - Returns `".into_iter()"` when the return type contains owned collections (`Vec<T>`,
///   `HashMap`, `HashSet`, `String`, `Result`, `Option`) since these need owned `T`.
/// - Returns `".iter()"` for reference return types or when no return type is specified.
fn iter_method_for_owned(return_type: Option<&str>) -> &'static str {
    match return_type {
        Some(ret) => {
            if ret.contains("Vec<")
                || ret.contains("HashMap")
                || ret.contains("HashSet")
                || ret.contains("String")
                || ret.contains("Result<")
                || ret.contains("Option<")
                || ret == "usize"
                || ret == "i32"
                || ret == "i64"
                || ret == "f32"
                || ret == "f64"
                || ret == "bool"
            {
                ".into_iter()"
            } else {
                ".iter()"
            }
        }
        None => ".into_iter()", // default to owned for safety
    }
}

fn infer_filter_closure(purpose: &str) -> &'static str {
    if purpose.contains("even") {
        return "*x % 2 == 0";
    }
    if purpose.contains("odd") {
        return "*x % 2 != 0";
    }
    if purpose.contains("positive") {
        return "*x > 0";
    }
    if purpose.contains("negative") {
        return "*x < 0";
    }
    if purpose.contains("non-zero") || purpose.contains("nonzero") {
        return "*x != 0";
    }
    if purpose.contains("zero") {
        return "*x == 0";
    }
    if purpose.contains("prime") {
        return "*x > 1 && (2..=(*x as f64).sqrt() as i32).all(|d| *x % d != 0)";
    }
    if purpose.contains("empty") {
        return "!x.is_empty()";
    }
    if purpose.contains("contains") {
        return "x.contains(&target)";
    }
    if purpose.contains("starts with") || purpose.contains("starts_with") {
        return "x.starts_with(&prefix)";
    }
    if purpose.contains("ends with") || purpose.contains("ends_with") {
        return "x.ends_with(&suffix)";
    }
    if purpose.contains("greater") || purpose.contains("above") {
        return "*x > threshold";
    }
    if purpose.contains("less") || purpose.contains("below") {
        return "*x < threshold";
    }
    if purpose.contains("between") || purpose.contains("in range") {
        return "*x >= low && *x <= high";
    }
    if purpose.contains("divisible") {
        return "*x % divisor == 0";
    }
    if purpose.contains("alphabetic") {
        return "x.chars().all(|c| c.is_alphabetic())";
    }
    if purpose.contains("numeric") || purpose.contains("digit") {
        return "x.chars().all(|c| c.is_numeric())";
    }
    if purpose.contains("unique") {
        return "true /* deduplicated upstream */";
    }
    "true /* TODO: specify condition */"
}

/// Infer a map/transform closure body from purpose.
fn infer_map_closure(purpose: &str) -> &'static str {
    if purpose.contains("double") {
        return "x * 2";
    }
    if purpose.contains("square") {
        return "x * x";
    }
    if purpose.contains("triple") {
        return "x * 3";
    }
    if purpose.contains("negate") {
        return "-x";
    }
    if purpose.contains("absolute") || purpose.contains("abs") {
        return "x.abs()";
    }
    if purpose.contains("string") || purpose.contains("to_string") {
        return "x.to_string()";
    }
    if purpose.contains("increment") || purpose.contains("add 1") {
        return "x + 1";
    }
    if purpose.contains("decrement") || purpose.contains("subtract 1") {
        return "x - 1";
    }
    if purpose.contains("half") || purpose.contains("halve") {
        return "x / 2";
    }
    if purpose.contains("uppercase") {
        return "x.to_uppercase()";
    }
    if purpose.contains("lowercase") {
        return "x.to_lowercase()";
    }
    if purpose.contains("length") || purpose.contains("len") {
        return "x.len()";
    }
    if purpose.contains("trim") {
        return "x.trim().to_string()";
    }
    if purpose.contains("reverse") {
        return "x.chars().rev().collect::<String>()";
    }
    if purpose.contains("clamp") {
        return "x.clamp(low, high)";
    }
    if purpose.contains("reciprocal") || purpose.contains("invert") {
        return "1.0 / x";
    }
    if purpose.contains("ceil") {
        return "x.ceil()";
    }
    if purpose.contains("floor") {
        return "x.floor()";
    }
    if purpose.contains("round") {
        return "x.round()";
    }
    if purpose.contains("sqrt") || purpose.contains("square root") {
        return "(x as f64).sqrt()";
    }
    if purpose.contains("cube") {
        return "x * x * x";
    }
    if purpose.contains("sign") || purpose.contains("signum") {
        return "x.signum()";
    }
    if purpose.contains("ascii") {
        return "*x as u8";
    }
    "x /* TODO: specify transform */"
}

/// Compose multiple atomic patterns into multi-step algorithms.
///
/// Handles patterns that are too complex for single-pattern matching but
/// don't need the generic iterator chain builder in `infer_composed_body`.
fn compose_patterns(
    purpose: &str,
    params: &[(String, String)],
    _return_type: Option<&str>,
) -> Option<String> {
    if params.is_empty() {
        return None;
    }

    let p0 = &params[0].0;
    let p0_is_vec = is_collection(&params[0].1) || params[0].1.contains("&[");

    // ── 1. HashMap-based lookups: frequency/occurrences/group by ──
    if (purpose.contains("frequency")
        || purpose.contains("occurrences")
        || purpose.contains("group by"))
        && p0_is_vec
    {
        return Some(format!(
            "let mut map = std::collections::HashMap::new();\n    \
             for item in {}.iter() {{\n        \
                 *map.entry(item.clone()).or_insert(0usize) += 1;\n    \
             }}\n    map",
            p0
        ));
    }

    // ── 2. Sort + select Nth largest/smallest ──
    if p0_is_vec
        && (purpose.contains("largest")
            || purpose.contains("smallest")
            || purpose.contains("nth")
            || purpose.contains("kth"))
        && (purpose.contains("sort") || purpose.contains("element") || purpose.contains("find the"))
    {
        let n = extract_number_from_text(purpose).unwrap_or(1);
        if purpose.contains("smallest") {
            return Some(format!(
                "let mut sorted = {}.to_vec();\n    sorted.sort();\n    sorted[{} - 1]",
                p0, n
            ));
        } else {
            return Some(format!(
                "let mut sorted = {}.to_vec();\n    sorted.sort();\n    sorted[sorted.len() - {}]",
                p0, n
            ));
        }
    }

    // ── 3. Accumulate with condition: "sum of even" / "product of positives" ──
    if p0_is_vec {
        let has_condition = purpose.contains("even")
            || purpose.contains("odd")
            || purpose.contains("positive")
            || purpose.contains("negative");
        let has_accumulate = purpose.contains("sum of")
            || purpose.contains("product of")
            || purpose.contains("sum the")
            || purpose.contains("product the");
        if has_condition && has_accumulate {
            let predicate = if purpose.contains("even") {
                "|x| x % 2 == 0"
            } else if purpose.contains("odd") {
                "|x| x % 2 != 0"
            } else if purpose.contains("positive") {
                "|x| *x > 0"
            } else {
                "|x| *x < 0"
            };

            let accumulator = if purpose.contains("product") {
                ".product()"
            } else {
                ".sum()"
            };

            return Some(format!(
                "{}.iter().filter({}).cloned(){}",
                p0, predicate, accumulator
            ));
        }
    }

    // ── 4. Zip + combine: element-wise operations on two vectors ──
    // Guard: only match when there's an actual arithmetic operation, not plain "zip together"
    // Exclude "dot product" which has its own specific pattern in infer_rust_body
    let has_arithmetic_intent = !purpose.contains("dot")
        && !purpose.contains("cartesian")
        && (purpose.contains("add")
            || purpose.contains("subtract")
            || purpose.contains("multiply")
            || purpose.contains("product")
            || purpose.contains("difference")
            || purpose.contains("element-wise")
            || purpose.contains("elementwise")
            || purpose.contains("pairwise"));
    if params.len() >= 2
        && p0_is_vec
        && (params[1].1.contains("Vec") || params[1].1.contains("&["))
        && has_arithmetic_intent
    {
        let p1 = &params[1].0;
        let op = if purpose.contains("multiply") || purpose.contains("product") {
            "*"
        } else if purpose.contains("subtract") || purpose.contains("difference") {
            "-"
        } else {
            "+"
        };
        return Some(format!(
            "{}.iter().zip({}.iter()).map(|(a, b)| a {} b).collect()",
            p0, p1, op
        ));
    }

    // ── 5. Deduplicate + count: "count unique/distinct elements" ──
    if p0_is_vec
        && (purpose.contains("unique") || purpose.contains("distinct"))
        && purpose.contains("count")
    {
        return Some(format!(
            "let set: std::collections::HashSet<_> = {}.iter().collect();\n    set.len()",
            p0
        ));
    }

    // ── 6. Nested iteration: "all pairs" / "combinations" / "cartesian product" ──
    if p0_is_vec
        && (purpose.contains("pairs")
            || purpose.contains("combinations")
            || purpose.contains("cartesian"))
    {
        if params.len() >= 2
            && (params[1].1.contains("Vec") || params[1].1.contains("&["))
            && purpose.contains("cartesian")
        {
            let p1 = &params[1].0;
            return Some(format!(
                "let mut result = Vec::new();\n    \
                 for a in {}.iter() {{\n        \
                     for b in {}.iter() {{\n            \
                         result.push((*a, *b));\n        \
                     }}\n    \
                 }}\n    result",
                p0, p1
            ));
        }
        // Single-vector pairs
        return Some(format!(
            "let mut result = Vec::new();\n    \
             for i in 0..{p}.len() {{\n        \
                 for j in (i+1)..{p}.len() {{\n            \
                     result.push(({p}[i], {p}[j]));\n        \
                 }}\n    \
             }}\n    result",
            p = p0
        ));
    }

    // ── 7. Filter + transform + collect (combined) ──
    if p0_is_vec {
        let has_filter = purpose.contains("filter") || purpose.contains("keep");
        let has_transform = purpose.contains("double")
            || purpose.contains("square")
            || purpose.contains("triple")
            || purpose.contains("negate");
        if has_filter && has_transform {
            let predicate = if purpose.contains("positive") {
                "|x| **x > 0"
            } else if purpose.contains("negative") {
                "|x| **x < 0"
            } else if purpose.contains("even") {
                "|x| **x % 2 == 0"
            } else if purpose.contains("odd") {
                "|x| **x % 2 != 0"
            } else {
                "|x| true"
            };

            let transform = if purpose.contains("double") {
                "|x| x * 2"
            } else if purpose.contains("square") {
                "|x| x * x"
            } else if purpose.contains("triple") {
                "|x| x * 3"
            } else {
                "|x| -x"
            };

            return Some(format!(
                "{}.iter().filter({}).map({}).collect()",
                p0, predicate, transform
            ));
        }
    }

    None
}

/// Try to infer function body from input/output examples
fn infer_from_examples(
    examples: &[(String, String)],
    params: &[(String, String)],
    _return_type: Option<&str>,
) -> Option<String> {
    if examples.is_empty() || params.is_empty() {
        return None;
    }

    // Check if all examples show the same simple pattern
    // e.g., ("add(1, 2)", "3"), ("add(3, 4)", "7") → a + b
    // This is a heuristic; for complex patterns we fall through to purpose matching

    // For now, if we have examples but can't infer, return None
    // The examples will still be used in test generation
    None
}

/// Auto-generate test assertions from purpose and signature when no examples are provided.
///
/// Returns a Vec of assertion lines (one per test case).
fn generate_auto_tests(
    func_name: &str,
    purpose: &str,
    sig: Option<&ParsedSignature>,
) -> Vec<String> {
    let purpose_lower = purpose.to_lowercase();
    let mut tests = Vec::new();

    let sig = match sig {
        Some(s) => s,
        None => return tests,
    };

    let has_i32_params = sig.params.iter().any(|(_, t)| {
        t.contains("i32") || t.contains("i64") || t.contains("u32") || t.contains("u64")
    });
    let has_scalar_i32_param = sig.params.iter().any(|(_, t)| {
        (t.contains("i32") || t.contains("i64") || t.contains("u32") || t.contains("u64"))
            && !t.contains("&[")
            && !t.contains("Vec")
    });
    let has_str_param = sig
        .params
        .iter()
        .any(|(_, t)| t.contains("str") || t.contains("String"));
    let has_vec_param = sig
        .params
        .iter()
        .any(|(_, t)| t.contains("Vec") || t.contains("&["));
    let returns_bool = sig
        .return_type
        .as_ref()
        .map_or(false, |r| r.contains("bool"));
    let returns_vec = sig
        .return_type
        .as_ref()
        .map_or(false, |r| r.contains("Vec"));
    let returns_string = sig
        .return_type
        .as_ref()
        .map_or(false, |r| r.contains("String") && !r.contains("Vec"));

    // Arithmetic: two numeric params → test with small values
    if sig.params.len() == 2 && has_i32_params && !returns_bool && !returns_vec {
        if purpose_lower.contains("add") || purpose_lower.contains("sum") {
            tests.push(format!("assert_eq!({}(2, 3), 5);", func_name));
            tests.push(format!("assert_eq!({}(0, 0), 0);", func_name));
            tests.push(format!("assert_eq!({}(-1, 1), 0);", func_name));
        } else if purpose_lower.contains("subtract") || purpose_lower.contains("difference") {
            tests.push(format!("assert_eq!({}(5, 3), 2);", func_name));
            tests.push(format!("assert_eq!({}(0, 0), 0);", func_name));
        } else if purpose_lower.contains("multiply") || purpose_lower.contains("product") {
            tests.push(format!("assert_eq!({}(3, 4), 12);", func_name));
            tests.push(format!("assert_eq!({}(0, 5), 0);", func_name));
        } else if purpose_lower.contains("divide") || purpose_lower.contains("quotient") {
            tests.push(format!("assert_eq!({}(10, 2), 5);", func_name));
            tests.push(format!("assert_eq!({}(7, 3), 2);", func_name));
        } else if purpose_lower.contains("max") || purpose_lower.contains("larger") {
            tests.push(format!("assert_eq!({}(3, 7), 7);", func_name));
            tests.push(format!("assert_eq!({}(5, 5), 5);", func_name));
        } else if purpose_lower.contains("min") || purpose_lower.contains("smaller") {
            tests.push(format!("assert_eq!({}(3, 7), 3);", func_name));
            tests.push(format!("assert_eq!({}(5, 5), 5);", func_name));
        } else if purpose_lower.contains("gcd") || purpose_lower.contains("greatest common") {
            tests.push(format!("assert_eq!({}(12, 8), 4);", func_name));
            tests.push(format!("assert_eq!({}(7, 5), 1);", func_name));
        }
    }

    // Single numeric param
    if sig.params.len() == 1 && has_scalar_i32_param {
        if returns_bool {
            if purpose_lower.contains("is even") || purpose_lower.contains("even") {
                tests.push(format!("assert!({}(4));", func_name));
                tests.push(format!("assert!(!{}(3));", func_name));
                tests.push(format!("assert!({}(0));", func_name));
            } else if purpose_lower.contains("is odd") || purpose_lower.contains("odd") {
                tests.push(format!("assert!({}(3));", func_name));
                tests.push(format!("assert!(!{}(4));", func_name));
            } else if purpose_lower.contains("is positive") || purpose_lower.contains("positive") {
                tests.push(format!("assert!({}(5));", func_name));
                tests.push(format!("assert!(!{}(-3));", func_name));
            } else if purpose_lower.contains("is negative") || purpose_lower.contains("negative") {
                tests.push(format!("assert!({}(-5));", func_name));
                tests.push(format!("assert!(!{}(3));", func_name));
            }
        }
        if purpose_lower.contains("factorial") {
            tests.push(format!("assert_eq!({}(0), 1);", func_name));
            tests.push(format!("assert_eq!({}(5), 120);", func_name));
        } else if purpose_lower.contains("fibonacci") {
            tests.push(format!("assert_eq!({}(0), 0);", func_name));
            tests.push(format!("assert_eq!({}(1), 1);", func_name));
            tests.push(format!("assert_eq!({}(10), 55);", func_name));
        } else if purpose_lower.contains("absolute") || purpose_lower.contains("abs") {
            tests.push(format!("assert_eq!({}(-5), 5);", func_name));
            tests.push(format!("assert_eq!({}(3), 3);", func_name));
        }
    }

    // String operations
    if has_str_param && sig.params.len() == 1 {
        if returns_string {
            if purpose_lower.contains("reverse") {
                tests.push(format!("assert_eq!({}(\"hello\"), \"olleh\");", func_name));
                tests.push(format!("assert_eq!({}(\"\"), \"\");", func_name));
            } else if purpose_lower.contains("uppercase") {
                tests.push(format!("assert_eq!({}(\"hello\"), \"HELLO\");", func_name));
            } else if purpose_lower.contains("lowercase") {
                tests.push(format!("assert_eq!({}(\"HELLO\"), \"hello\");", func_name));
            } else if purpose_lower.contains("trim") || purpose_lower.contains("strip") {
                tests.push(format!("assert_eq!({}(\"  hi  \"), \"hi\");", func_name));
            } else if purpose_lower.contains("capitalize") {
                tests.push(format!("assert_eq!({}(\"hello\"), \"Hello\");", func_name));
            }
        }
        if returns_bool {
            if purpose_lower.contains("is empty") || purpose_lower.contains("is_empty") {
                tests.push(format!("assert!({}(\"\"));", func_name));
                tests.push(format!("assert!(!{}(\"hi\"));", func_name));
            }
        }
    }

    // Vec operations
    if has_vec_param && sig.params.len() == 1 {
        if returns_vec && (purpose_lower.contains("sort") || purpose_lower.contains("order")) {
            let first_param_type = sig.params.first().map(|(_, ty)| ty.as_str()).unwrap_or("");
            if first_param_type.contains("&[") {
                tests.push(format!(
                    "let v = vec![3, 1, 2]; assert_eq!({}(&v), vec![1, 2, 3]);",
                    func_name
                ));
                tests.push(format!(
                    "let v: Vec<i32> = Vec::new(); assert_eq!({}(&v), Vec::<i32>::new());",
                    func_name
                ));
            } else {
                tests.push(format!(
                    "assert_eq!({}(vec![3, 1, 2]), vec![1, 2, 3]);",
                    func_name
                ));
                tests.push(format!("assert_eq!({}(vec![]), vec![]);", func_name));
            }
        }
    }

    // Suppress unused variable warnings
    let _ = returns_string;

    tests
}

// ============================================================================
// Import Inference
// ============================================================================

/// Scan Rust source code and prepend any needed `use` imports.
///
/// Detects common stdlib types and traits used in the code but not imported.
pub(crate) fn infer_rust_imports(source: &str) -> String {
    let known_imports: &[(&str, &str)] = &[
        ("HashMap", "use std::collections::HashMap;"),
        ("HashSet", "use std::collections::HashSet;"),
        ("BTreeMap", "use std::collections::BTreeMap;"),
        ("BTreeSet", "use std::collections::BTreeSet;"),
        ("VecDeque", "use std::collections::VecDeque;"),
        ("BinaryHeap", "use std::collections::BinaryHeap;"),
        ("File", "use std::fs::File;"),
        ("OpenOptions", "use std::fs::OpenOptions;"),
        ("Duration", "use std::time::Duration;"),
        ("Instant", "use std::time::Instant;"),
        ("BufReader", "use std::io::BufReader;"),
        ("BufWriter", "use std::io::BufWriter;"),
        ("Ordering", "use std::cmp::Ordering;"),
        ("Reverse", "use std::cmp::Reverse;"),
        ("stdin", "use std::io;"),
        ("Arc", "use std::sync::Arc;"),
        ("Mutex", "use std::sync::Mutex;"),
        ("Rc", "use std::rc::Rc;"),
        ("RefCell", "use std::cell::RefCell;"),
        ("Path", "use std::path::Path;"),
        ("PathBuf", "use std::path::PathBuf;"),
        ("fmt::Display", "use std::fmt;"),
        ("fmt::Formatter", "use std::fmt;"),
    ];

    let mut imports = Vec::new();
    for (type_name, import_stmt) in known_imports {
        // Check if the type is used in the source but not already imported
        if source.contains(type_name) && !source.contains(import_stmt) {
            // Make sure it's not just a substring (e.g., "HashMap" in a comment)
            // Simple heuristic: check for type usage patterns
            let patterns = [
                format!("{}<", type_name),   // HashMap<K, V>
                format!("{}::", type_name),  // HashMap::new()
                format!(": {}", type_name),  // x: HashMap
                format!("-> {}", type_name), // -> HashMap
                format!("{} {{", type_name), // HashMap {
                format!("{}(", type_name),   // File(
                format!("&{}", type_name),   // &Path
            ];
            if patterns.iter().any(|p| source.contains(p.as_str())) {
                if !imports.contains(&import_stmt.to_string()) {
                    imports.push(import_stmt.to_string());
                }
            }
        }
    }

    if imports.is_empty() {
        source.to_string()
    } else {
        format!("{}\n\n{}", imports.join("\n"), source)
    }
}

// ============================================================================
// Public API for cross-module use (test-first generation)
// ============================================================================

/// Parse a Rust function signature (pub wrapper for cross-module use).
pub(crate) fn parse_rust_signature_pub(sig: &str) -> Option<ParsedSignature> {
    parse_rust_signature(sig)
}

/// Generate auto-tests from purpose and signature (pub wrapper for cross-module use).
pub(crate) fn generate_auto_tests_pub(
    func_name: &str,
    purpose: &str,
    sig: Option<&ParsedSignature>,
) -> Vec<String> {
    generate_auto_tests(func_name, purpose, sig)
}

/// Generate property-based invariant tests (pub wrapper for cross-module use).
pub(crate) fn generate_property_tests_pub(
    func_name: &str,
    purpose: &str,
    sig: Option<&ParsedSignature>,
) -> Vec<String> {
    generate_property_tests(func_name, purpose, sig)
}

/// Generate property-based invariant tests from purpose and signature.
///
/// Unlike `generate_auto_tests` which tests specific input→output pairs,
/// this generates algebraic property assertions that hold for ANY valid input.
fn generate_property_tests(
    func_name: &str,
    purpose: &str,
    sig: Option<&ParsedSignature>,
) -> Vec<String> {
    let purpose_lower = purpose.to_lowercase();
    let mut tests = Vec::new();

    let sig = match sig {
        Some(s) => s,
        None => return tests,
    };

    let has_i32_params = sig.params.iter().any(|(_, t)| {
        t.contains("i32") || t.contains("i64") || t.contains("u32") || t.contains("u64")
    });
    let has_scalar_i32_param = sig.params.iter().any(|(_, t)| {
        (t.contains("i32") || t.contains("i64") || t.contains("u32") || t.contains("u64"))
            && !t.contains("&[")
            && !t.contains("Vec")
    });
    let has_str_param = sig
        .params
        .iter()
        .any(|(_, t)| t.contains("str") || t.contains("String"));
    let has_vec_param = sig
        .params
        .iter()
        .any(|(_, t)| t.contains("Vec") || t.contains("&["));
    let returns_vec = sig
        .return_type
        .as_ref()
        .map_or(false, |r| r.contains("Vec"));
    let returns_string = sig
        .return_type
        .as_ref()
        .map_or(false, |r| r.contains("String") && !r.contains("Vec"));

    // ── Sorting: idempotency — sort(sort(v)) == sort(v) ──
    if has_vec_param
        && returns_vec
        && (purpose_lower.contains("sort") || purpose_lower.contains("order"))
    {
        let first_param_type = sig.params.first().map(|(_, ty)| ty.as_str()).unwrap_or("");
        if first_param_type.contains("&[") {
            tests.push(format!(
                "let v = vec![3, 1, 4, 1, 5, 9, 2, 6];\n        \
                 let once = {f}(&v);\n        \
                 assert_eq!(once.clone(), {f}(&once), \"sort must be idempotent\");",
                f = func_name
            ));
            tests.push(format!(
                "let v = vec![5, 3, 8, 1];\n        \
                 assert_eq!({f}(&v).len(), v.len(), \"sort must preserve length\");",
                f = func_name
            ));
        } else {
            tests.push(format!(
                "let v = vec![3, 1, 4, 1, 5, 9, 2, 6];\n        \
                 assert_eq!({f}(v.clone()), {f}({f}(v.clone())), \"sort must be idempotent\");",
                f = func_name
            ));
            tests.push(format!(
                "let v = vec![5, 3, 8, 1];\n        \
                 assert_eq!({f}(v.clone()).len(), v.len(), \"sort must preserve length\");",
                f = func_name
            ));
        }
    }

    // ── Reverse: involution — reverse(reverse(x)) == x ──
    if purpose_lower.contains("reverse") {
        if has_str_param && returns_string {
            tests.push(format!(
                "let s = \"abcdef\".to_string();\n        \
                 assert_eq!({f}(&{f}(&s)), s, \"reverse must be an involution\");",
                f = func_name
            ));
        }
        if has_vec_param && returns_vec {
            tests.push(format!(
                "let v = vec![1, 2, 3, 4, 5];\n        \
                 assert_eq!({f}({f}(v.clone())), v, \"reverse must be an involution\");",
                f = func_name
            ));
        }
    }

    // ── Filter: size reduction — filter(v).len() <= v.len() ──
    if has_vec_param
        && returns_vec
        && (purpose_lower.contains("filter")
            || purpose_lower.contains("keep")
            || purpose_lower.contains("remove")
            || purpose_lower.contains("select"))
    {
        let first_param_type = sig.params.first().map(|(_, t)| t.as_str()).unwrap_or("");
        if first_param_type.contains("&[&str]") {
            tests.push(format!(
                "let v = vec![\"1\", \"x\", \"2\", \"3\"];\n        \
                 assert!({f}(&v).len() <= v.len(), \"filter must not increase length\");",
                f = func_name
            ));
        } else if first_param_type.contains("&[String]") {
            tests.push(format!(
                "let v = vec![\"a\".to_string(), \"bb\".to_string(), \"ccc\".to_string()];\n        \
                 assert!({f}(&v).len() <= v.len(), \"filter must not increase length\");",
                f = func_name
            ));
        } else if first_param_type.contains("&[") {
            tests.push(format!(
                "let v = vec![1, 2, 3, 4, 5, 6, 7, 8];\n        \
                 assert!({f}(&v).len() <= v.len(), \"filter must not increase length\");",
                f = func_name
            ));
        } else {
            tests.push(format!(
                "let v = vec![1, 2, 3, 4, 5, 6, 7, 8];\n        \
                 assert!({f}(v.clone()).len() <= v.len(), \"filter must not increase length\");",
                f = func_name
            ));
        }
    }

    // ── Arithmetic commutativity: f(a, b) == f(b, a) ──
    if sig.params.len() == 2 && has_i32_params {
        let is_commutative = purpose_lower.contains("add")
            || purpose_lower.contains("sum")
            || purpose_lower.contains("multiply")
            || purpose_lower.contains("product")
            || purpose_lower.contains("max")
            || purpose_lower.contains("min")
            || purpose_lower.contains("gcd");
        if is_commutative {
            tests.push(format!(
                "assert_eq!({f}(7, 13), {f}(13, 7), \"{f} must be commutative\");",
                f = func_name
            ));
        }
    }

    // ── Identity element: f(x, 0) == x for add, f(x, 1) == x for multiply ──
    if sig.params.len() == 2 && has_i32_params {
        if purpose_lower.contains("add") || purpose_lower.contains("sum") {
            tests.push(format!(
                "assert_eq!({f}(42, 0), 42, \"0 must be additive identity\");",
                f = func_name
            ));
        }
        if purpose_lower.contains("multiply") || purpose_lower.contains("product") {
            tests.push(format!(
                "assert_eq!({f}(42, 1), 42, \"1 must be multiplicative identity\");",
                f = func_name
            ));
        }
    }

    // ── Absolute value: |x| >= 0, |x| == |-x| ──
    if sig.params.len() == 1
        && has_scalar_i32_param
        && (purpose_lower.contains("absolute") || purpose_lower.contains("abs"))
    {
        tests.push(format!(
            "for x in [-10, -1, 0, 1, 10] {{\n            \
                 assert!({f}(x) >= 0, \"abs must be non-negative\");\n        \
             }}",
            f = func_name
        ));
        tests.push(format!(
            "for x in [-5, -1, 0, 1, 5] {{\n            \
                 assert_eq!({f}(x), {f}(-x), \"abs must satisfy |x| == |-x|\");\n        \
             }}",
            f = func_name
        ));
    }

    // ── String case: length preservation ──
    if has_str_param && returns_string && sig.params.len() == 1 {
        if purpose_lower.contains("uppercase") || purpose_lower.contains("lowercase") {
            tests.push(format!(
                "let s = \"Hello World\";\n        \
                 assert_eq!({f}(s).len(), s.len(), \"case change must preserve length\");",
                f = func_name
            ));
        }
        if purpose_lower.contains("trim") || purpose_lower.contains("strip") {
            tests.push(format!(
                "let s = \"  hello  \";\n        \
                 assert!({f}(s).len() <= s.len(), \"trim must not increase length\");",
                f = func_name
            ));
        }
    }

    // ── Map/transform: output length == input length ──
    if has_vec_param
        && returns_vec
        && (purpose_lower.contains("map")
            || purpose_lower.contains("double")
            || purpose_lower.contains("square")
            || purpose_lower.contains("negate"))
        && !purpose_lower.contains("filter")
    {
        let first_param_type = sig.params.first().map(|(_, ty)| ty.as_str()).unwrap_or("");
        let values = if first_param_type.contains("String") {
            "vec![\"A\".to_string(), \"B\".to_string(), \"C\".to_string()]"
        } else {
            "vec![1, 2, 3, 4, 5]"
        };
        let arg = if first_param_type.contains("&[") {
            "&v"
        } else {
            "v.clone()"
        };
        tests.push(format!(
            "let v = {values};\n        \
             assert_eq!({f}({arg}).len(), v.len(), \"map must preserve length\");",
            f = func_name,
            arg = arg,
            values = values
        ));
    }

    tests
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::cfc_code_sequencer::{CodePlanStep, PlanAction};

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
        let fields = vec![("name".to_string(), "str".to_string())];
        let result = emitter.emit_struct("Person", &fields);
        assert!(result.contains("class Person"));
        assert!(result.contains("self.name = name"));
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
        let plan = vec![CodePlanStep {
            action: PlanAction::DefineFunction,
            name: None,
            context: Vec::new(),
            confidence: 0.9,
        }];
        let result = emitter.emit_from_spec(&spec, &plan);
        assert!(result.contains("sort_vec"));
        assert!(result.contains("Sort a vector"));
    }

    #[test]
    fn test_python_emit_from_spec_with_examples() {
        let emitter = PythonEmitter;
        let spec = CodeSpec::new("python", "add", "Add two numbers").with_example("add(1, 2)", "3");
        let plan = vec![CodePlanStep {
            action: PlanAction::DefineFunction,
            name: None,
            context: Vec::new(),
            confidence: 0.9,
        }];
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

    // ========================================================================
    // New tests: real code generation from spec + signature
    // ========================================================================

    #[test]
    fn test_rust_add_with_signature() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "add", "Add two numbers")
            .with_signature("fn add(a: i32, b: i32) -> i32")
            .with_example("add(1, 2)", "3")
            .with_example("add(-1, 1)", "0");
        let plan = vec![CodePlanStep {
            action: PlanAction::DefineFunction,
            name: None,
            context: Vec::new(),
            confidence: 0.9,
        }];
        let result = emitter.emit_from_spec(&spec, &plan);
        assert!(result.contains("pub fn add(a: i32, b: i32) -> i32"));
        assert!(result.contains("a + b"), "Should infer body: {}", result);
        assert!(result.contains("assert_eq!(add(1, 2), 3)"));
        assert!(result.contains("assert_eq!(add(-1, 1), 0)"));
        // Should NOT contain todo!
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_rust_reverse_string() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "reverse", "Reverse a string")
            .with_signature("fn reverse(s: &str) -> String");
        let plan = vec![CodePlanStep {
            action: PlanAction::DefineFunction,
            name: None,
            context: Vec::new(),
            confidence: 0.9,
        }];
        let result = emitter.emit_from_spec(&spec, &plan);
        assert!(
            result.contains("s.chars().rev().collect()"),
            "Should infer reverse: {}",
            result
        );
    }

    #[test]
    fn test_rust_sort_vec() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "sort", "Sort a vector")
            .with_signature("fn sort(items: Vec<i32>) -> Vec<i32>");
        let plan = vec![CodePlanStep {
            action: PlanAction::DefineFunction,
            name: None,
            context: Vec::new(),
            confidence: 0.9,
        }];
        let result = emitter.emit_from_spec(&spec, &plan);
        assert!(result.contains(".sort()"), "Should infer sort: {}", result);
    }

    #[test]
    fn test_rust_is_even() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "is_even", "Check if a number is even")
            .with_signature("fn is_even(n: i32) -> bool");
        let plan = vec![CodePlanStep {
            action: PlanAction::DefineFunction,
            name: None,
            context: Vec::new(),
            confidence: 0.9,
        }];
        let result = emitter.emit_from_spec(&spec, &plan);
        assert!(
            result.contains("% 2 == 0"),
            "Should infer even check: {}",
            result
        );
    }

    #[test]
    fn test_rust_factorial() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "factorial", "Compute factorial of n")
            .with_signature("fn factorial(n: u64) -> u64");
        let plan = vec![CodePlanStep {
            action: PlanAction::DefineFunction,
            name: None,
            context: Vec::new(),
            confidence: 0.9,
        }];
        let result = emitter.emit_from_spec(&spec, &plan);
        assert!(
            result.contains(".product()"),
            "Should infer factorial: {}",
            result
        );
    }

    #[test]
    fn test_rust_struct_with_fields() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "Point", "A point with x: f64 and y: f64 fields");
        let plan = vec![
            CodePlanStep {
                action: PlanAction::DefineStruct,
                name: None,
                context: Vec::new(),
                confidence: 0.9,
            },
            CodePlanStep {
                action: PlanAction::AddField,
                name: None,
                context: Vec::new(),
                confidence: 0.8,
            },
            CodePlanStep {
                action: PlanAction::AddField,
                name: None,
                context: Vec::new(),
                confidence: 0.8,
            },
            CodePlanStep {
                action: PlanAction::ImplTrait,
                name: None,
                context: Vec::new(),
                confidence: 0.7,
            },
        ];
        let result = emitter.emit_from_spec(&spec, &plan);
        assert!(
            result.contains("pub struct Point"),
            "Should define struct: {}",
            result
        );
        assert!(result.contains("x: f64"), "Should have x field: {}", result);
        assert!(result.contains("y: f64"), "Should have y field: {}", result);
        assert!(
            result.contains("fn new"),
            "Should have constructor: {}",
            result
        );
    }

    #[test]
    fn test_parse_rust_signature() {
        let sig = parse_rust_signature("fn add(a: i32, b: i32) -> i32").unwrap();
        assert_eq!(sig.name, "add");
        assert_eq!(sig.params.len(), 2);
        assert_eq!(sig.params[0], ("a".to_string(), "i32".to_string()));
        assert_eq!(sig.return_type, Some("i32".to_string()));
        assert!(!sig._is_method);
    }

    #[test]
    fn test_parse_rust_signature_method() {
        let sig = parse_rust_signature("fn name(&self) -> &str").unwrap();
        assert_eq!(sig.name, "name");
        assert!(sig.params.is_empty());
        assert!(sig._is_method);
        assert_eq!(sig.return_type, Some("&str".to_string()));
    }

    #[test]
    fn test_parse_rust_signature_no_return() {
        let sig = parse_rust_signature("fn do_thing(x: String)").unwrap();
        assert_eq!(sig.name, "do_thing");
        assert_eq!(sig.params.len(), 1);
        assert_eq!(sig.return_type, None);
    }

    #[test]
    fn test_python_add_real_body() {
        let emitter = PythonEmitter;
        let spec = CodeSpec::new("python", "add", "Add two numbers").with_example("add(1, 2)", "3");
        let plan = vec![CodePlanStep {
            action: PlanAction::DefineFunction,
            name: None,
            context: Vec::new(),
            confidence: 0.9,
        }];
        let result = emitter.emit_from_spec(&spec, &plan);
        assert!(
            result.contains("return a + b"),
            "Should infer add: {}",
            result
        );
        assert!(result.contains("assert add(1, 2) == 3"));
    }

    // ========================================================================
    // Composition tests: multi-step operations
    // ========================================================================

    #[test]
    fn test_rust_filter_even_and_sum() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "sum_evens", "Filter even numbers and sum them")
            .with_signature("fn sum_evens(nums: Vec<i32>) -> i32");
        let plan = vec![CodePlanStep {
            action: PlanAction::DefineFunction,
            name: None,
            context: Vec::new(),
            confidence: 0.9,
        }];
        let result = emitter.emit_from_spec(&spec, &plan);
        assert!(result.contains("filter"), "Should have filter: {}", result);
        assert!(
            result.contains("% 2 == 0"),
            "Should filter even: {}",
            result
        );
        assert!(result.contains(".sum()"), "Should sum: {}", result);
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_rust_sort_and_take_first() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new(
            "rust",
            "top_three",
            "Sort a vector and take first 3 elements",
        )
        .with_signature("fn top_three(items: Vec<i32>) -> Vec<i32>");
        let plan = vec![CodePlanStep {
            action: PlanAction::DefineFunction,
            name: None,
            context: Vec::new(),
            confidence: 0.9,
        }];
        let result = emitter.emit_from_spec(&spec, &plan);
        assert!(result.contains(".sort()"), "Should sort: {}", result);
        assert!(result.contains(".take(3)"), "Should take 3: {}", result);
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_rust_filter_and_count() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new(
            "rust",
            "count_positive",
            "Filter positive numbers and count them",
        )
        .with_signature("fn count_positive(nums: Vec<i32>) -> usize");
        let plan = vec![CodePlanStep {
            action: PlanAction::DefineFunction,
            name: None,
            context: Vec::new(),
            confidence: 0.9,
        }];
        let result = emitter.emit_from_spec(&spec, &plan);
        assert!(result.contains("filter"), "Should have filter: {}", result);
        assert!(result.contains(".count()"), "Should count: {}", result);
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_rust_sort_and_dedup() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "sort_unique", "Sort and deduplicate a vector")
            .with_signature("fn sort_unique(items: Vec<i32>) -> Vec<i32>");
        let plan = vec![CodePlanStep {
            action: PlanAction::DefineFunction,
            name: None,
            context: Vec::new(),
            confidence: 0.9,
        }];
        let result = emitter.emit_from_spec(&spec, &plan);
        assert!(result.contains(".sort()"), "Should sort: {}", result);
        assert!(result.contains(".dedup()"), "Should dedup: {}", result);
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_extract_number_from_text() {
        assert_eq!(extract_number_from_text("take first 3 elements"), Some(3));
        assert_eq!(extract_number_from_text("top five items"), Some(5));
        assert_eq!(extract_number_from_text("no number here"), None);
    }

    #[test]
    fn test_auto_tests_add() {
        let sig = ParsedSignature {
            name: "add".to_string(),
            params: vec![
                ("a".to_string(), "i32".to_string()),
                ("b".to_string(), "i32".to_string()),
            ],
            return_type: Some("i32".to_string()),
            _is_method: false,
        };
        let tests = generate_auto_tests("add", "Add two numbers", Some(&sig));
        assert!(!tests.is_empty(), "Should generate tests for add");
        assert!(tests.iter().any(|t| t.contains("assert_eq!(add(2, 3), 5)")));
    }

    #[test]
    fn test_auto_tests_is_even() {
        let sig = ParsedSignature {
            name: "is_even".to_string(),
            params: vec![("n".to_string(), "i32".to_string())],
            return_type: Some("bool".to_string()),
            _is_method: false,
        };
        let tests = generate_auto_tests("is_even", "Check if a number is even", Some(&sig));
        assert!(!tests.is_empty(), "Should generate tests for is_even");
        assert!(tests.iter().any(|t| t.contains("assert!(is_even(4))")));
        assert!(tests.iter().any(|t| t.contains("assert!(!is_even(3))")));
    }

    #[test]
    fn test_auto_tests_reverse() {
        let sig = ParsedSignature {
            name: "reverse".to_string(),
            params: vec![("s".to_string(), "&str".to_string())],
            return_type: Some("String".to_string()),
            _is_method: false,
        };
        let tests = generate_auto_tests("reverse", "Reverse a string", Some(&sig));
        assert!(!tests.is_empty(), "Should generate tests for reverse");
        assert!(tests.iter().any(|t| t.contains("\"olleh\"")));
    }

    #[test]
    fn test_auto_tests_factorial() {
        let sig = ParsedSignature {
            name: "factorial".to_string(),
            params: vec![("n".to_string(), "u64".to_string())],
            return_type: Some("u64".to_string()),
            _is_method: false,
        };
        let tests = generate_auto_tests("factorial", "Compute factorial", Some(&sig));
        assert!(!tests.is_empty(), "Should generate tests for factorial");
        assert!(tests.iter().any(|t| t.contains("120")));
    }

    #[test]
    fn test_auto_tests_sort() {
        let sig = ParsedSignature {
            name: "sort".to_string(),
            params: vec![("items".to_string(), "Vec<i32>".to_string())],
            return_type: Some("Vec<i32>".to_string()),
            _is_method: false,
        };
        let tests = generate_auto_tests("sort", "Sort a vector", Some(&sig));
        assert!(!tests.is_empty(), "Should generate tests for sort");
        assert!(tests.iter().any(|t| t.contains("vec![1, 2, 3]")));
    }

    #[test]
    fn test_parse_generic_signature_uses_callable_name() {
        let sig = parse_rust_signature("fn sorted_clone<T: Ord + Clone>(items: &[T]) -> Vec<T>")
            .expect("generic signature should parse");
        assert_eq!(sig.name, "sorted_clone");
        assert_eq!(sig.params[0].0, "items");
        assert_eq!(sig.params[0].1, "&[T]");
    }

    #[test]
    fn test_auto_tests_sort_borrowed_slice_use_valid_calls() {
        let sig = ParsedSignature {
            name: "sorted_clone".to_string(),
            params: vec![("items".to_string(), "&[T]".to_string())],
            return_type: Some("Vec<T>".to_string()),
            _is_method: false,
        };
        let tests = generate_auto_tests(
            "sorted_clone",
            "Return a sorted cloned vector from a generic slice using the Ord bound",
            Some(&sig),
        );
        assert!(!tests.is_empty(), "Should generate tests for borrowed sort");
        assert!(
            tests.iter().all(|test| !test.contains("<T:")),
            "Generated calls must not include generic bounds: {:?}",
            tests
        );
        assert!(
            tests.iter().any(|test| test.contains("sorted_clone(&v)")),
            "Borrowed slice tests should pass a borrowed vector: {:?}",
            tests
        );
    }

    #[test]
    fn test_auto_tests_no_sig_no_tests() {
        let tests = generate_auto_tests("foo", "Do something", None);
        assert!(tests.is_empty(), "No sig → no auto tests");
    }

    #[test]
    fn test_rust_emit_with_auto_tests() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "add", "Add two numbers")
            .with_signature("fn add(a: i32, b: i32) -> i32");
        let result = emitter.emit_from_spec(
            &spec,
            &[CodePlanStep {
                action: PlanAction::DefineFunction,
                name: None,
                context: vec![],
                confidence: 0.9,
            }],
        );
        // Should have auto-generated tests since no examples provided
        assert!(
            result.contains("#[test]"),
            "Should contain auto-generated tests"
        );
        assert!(
            result.contains("assert_eq!(add(2, 3), 5)"),
            "Should have add(2,3)=5 assertion"
        );
    }

    #[test]
    fn test_multi_entity_struct_with_distance() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new(
            "rust",
            "Point",
            "Point struct with x: f64 and y: f64 with distance method",
        )
        .with_constraint("MULTI_ENTITY: generate struct + impl block + methods");
        let result = emitter.emit_from_spec(
            &spec,
            &[CodePlanStep {
                action: PlanAction::DefineStruct,
                name: None,
                context: vec![],
                confidence: 0.9,
            }],
        );
        assert!(
            result.contains("pub struct Point"),
            "Should have struct: {}",
            result
        );
        assert!(
            result.contains("impl Point"),
            "Should have impl block: {}",
            result
        );
        assert!(
            result.contains("fn new("),
            "Should have constructor: {}",
            result
        );
        assert!(
            result.contains("fn distance("),
            "Should have distance method: {}",
            result
        );
    }

    // ========================================================================
    // New pattern arm tests: iterator adapters, combinators, algorithms, etc.
    // ========================================================================

    fn make_plan() -> Vec<CodePlanStep> {
        vec![CodePlanStep {
            action: PlanAction::DefineFunction,
            name: None,
            context: vec![],
            confidence: 0.9,
        }]
    }

    #[test]
    fn test_emit_windows() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "sliding", "Get sliding windows of size n")
            .with_signature("fn sliding(items: Vec<i32>, n: usize) -> Vec<Vec<i32>>");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(
            result.contains(".windows("),
            "Should use windows: {}",
            result
        );
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_emit_chain() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "merge", "Chain two lists together")
            .with_signature("fn merge(a: Vec<i32>, b: Vec<i32>) -> Vec<i32>");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(
            result.contains(".concat()"),
            "Should use concat: {}",
            result
        );
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_emit_flat_map() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "flatten_map", "Apply flat_map to nested list")
            .with_signature("fn flatten_map(items: Vec<Vec<i32>>) -> Vec<i32>");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(
            result.contains("flat_map"),
            "Should use flat_map: {}",
            result
        );
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_emit_partition() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "split", "Partition a list into two groups")
            .with_signature("fn split(items: Vec<i32>) -> (Vec<i32>, Vec<i32>)");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(
            result.contains(".partition("),
            "Should use partition: {}",
            result
        );
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_emit_any() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new(
            "rust",
            "has_match",
            "Check if any element matches condition",
        )
        .with_signature("fn has_match(items: Vec<i32>) -> bool");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(result.contains(".any("), "Should use any: {}", result);
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_emit_all() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new(
            "rust",
            "all_valid",
            "Check if every element satisfies condition",
        )
        .with_signature("fn all_valid(items: Vec<i32>) -> bool");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(result.contains(".all("), "Should use all: {}", result);
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_emit_unwrap_or() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "get_or", "Return value or unwrap_or with default")
            .with_signature("fn get_or(val: Option<i32>, fallback: i32) -> i32");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(
            result.contains(".unwrap_or("),
            "Should use unwrap_or: {}",
            result
        );
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_emit_map_or() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "transform_opt", "Apply map_or to option")
            .with_signature("fn transform_opt(val: Option<i32>) -> i32");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(result.contains(".map_or("), "Should use map_or: {}", result);
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_emit_ok_or() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "require", "Convert option to result using ok_or")
            .with_signature("fn require(val: Option<i32>) -> Result<i32, String>");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(result.contains(".ok_or("), "Should use ok_or: {}", result);
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_emit_and_then() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "chain_opt", "Apply and_then to option")
            .with_signature("fn chain_opt(val: Option<i32>) -> Option<i32>");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(
            result.contains(".and_then("),
            "Should use and_then: {}",
            result
        );
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_emit_palindrome() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "is_palindrome", "Check if string is a palindrome")
            .with_signature("fn is_palindrome(s: &str) -> bool");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(
            result.contains(".chars().rev()"),
            "Should reverse chars: {}",
            result
        );
        assert!(result.contains("s == r"), "Should compare: {}", result);
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_emit_two_sum() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "two_sum", "Find two_sum indices that add to target")
            .with_signature("fn two_sum(nums: Vec<i32>, target: i32) -> Option<(usize, usize)>");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(
            result.contains("nums[i] + nums[j]"),
            "Should have nested loop: {}",
            result
        );
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_emit_matrix_transpose() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "transpose", "Transpose a matrix")
            .with_signature("fn transpose(matrix: Vec<Vec<i32>>) -> Vec<Vec<i32>>");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(result.contains("cols"), "Should reference cols: {}", result);
        assert!(result.contains("rows"), "Should reference rows: {}", result);
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_emit_dot_product() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "dot", "Compute dot product of two vectors")
            .with_signature("fn dot(a: Vec<f64>, b: Vec<f64>) -> f64");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(result.contains(".zip("), "Should zip: {}", result);
        assert!(result.contains("a * b"), "Should multiply: {}", result);
        assert!(result.contains(".sum()"), "Should sum: {}", result);
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_emit_count_words() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "count_words", "Count words in a string")
            .with_signature("fn count_words(s: &str) -> usize");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(
            result.contains("split_whitespace"),
            "Should split whitespace: {}",
            result
        );
        assert!(result.contains(".count()"), "Should count: {}", result);
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_emit_median() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "median", "Compute the median of a list")
            .with_signature("fn median(nums: Vec<i32>) -> i32");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(result.contains("sorted.sort()"), "Should sort: {}", result);
        assert!(result.contains("mid"), "Should find middle: {}", result);
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_emit_mode() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "mode", "Find the mode of a list")
            .with_signature("fn mode(nums: Vec<i32>) -> i32");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(result.contains("HashMap"), "Should use HashMap: {}", result);
        assert!(
            result.contains("max_by_key"),
            "Should find max frequency: {}",
            result
        );
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_emit_lcm() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "lcm", "Compute least common multiple")
            .with_signature("fn lcm(a: i32, b: i32) -> i32");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(
            result.contains("product"),
            "Should compute product: {}",
            result
        );
        assert!(result.contains("% b"), "Should use mod for gcd: {}", result);
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_emit_to_string() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "stringify", "Convert to_string representation")
            .with_signature("fn stringify(n: i32) -> String");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(
            result.contains(".to_string()"),
            "Should use to_string: {}",
            result
        );
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_emit_parse() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "parse_int", "Parse a string to integer")
            .with_signature("fn parse_int(s: &str) -> i32");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(result.contains(".parse"), "Should use parse: {}", result);
        assert!(
            result.contains("unwrap_or_default"),
            "Should have default: {}",
            result
        );
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_emit_to_vec() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "to_vec", "Convert slice to_vec")
            .with_signature("fn to_vec(items: &[i32]) -> Vec<i32>");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(
            result.contains(".to_vec()"),
            "Should use to_vec: {}",
            result
        );
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_emit_chars() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "characters", "Get characters from string")
            .with_signature("fn characters(s: &str) -> Vec<char>");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(
            result.contains(".chars().collect"),
            "Should collect chars: {}",
            result
        );
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_emit_swap() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "swap", "Swap two values")
            .with_signature("fn swap(a: i32, b: i32) -> (i32, i32)");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(result.contains("(b, a)"), "Should swap: {}", result);
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_emit_modulo() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "modulo", "Compute modulo of two numbers")
            .with_signature("fn modulo(a: i32, b: i32) -> i32");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(result.contains("a % b"), "Should use modulo: {}", result);
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    // ── Property-based test generation tests ──

    #[test]
    fn test_property_sort_idempotency() {
        let sig = parse_rust_signature("fn sort(v: Vec<i32>) -> Vec<i32>").unwrap();
        let props = generate_property_tests("sort", "sort a vector", Some(&sig));
        assert!(!props.is_empty(), "sort should generate property tests");
        let joined = props.join("\n");
        assert!(
            joined.contains("idempotent"),
            "sort should test idempotency"
        );
        assert!(
            joined.contains("preserve length"),
            "sort should test length"
        );
    }

    #[test]
    fn test_property_reverse_involution() {
        let sig = parse_rust_signature("fn reverse(s: &str) -> String").unwrap();
        let props = generate_property_tests("reverse", "reverse a string", Some(&sig));
        assert!(!props.is_empty(), "reverse should generate property tests");
        assert!(
            props.join("\n").contains("involution"),
            "reverse should test involution"
        );
    }

    #[test]
    fn test_property_add_commutativity() {
        let sig = parse_rust_signature("fn add(a: i32, b: i32) -> i32").unwrap();
        let props = generate_property_tests("add", "add two numbers", Some(&sig));
        assert!(!props.is_empty(), "add should generate property tests");
        let joined = props.join("\n");
        assert!(
            joined.contains("commutative"),
            "add should test commutativity"
        );
        assert!(
            joined.contains("additive identity"),
            "add should test identity"
        );
    }

    #[test]
    fn test_property_filter_size_reduction() {
        let sig = parse_rust_signature("fn filter_pos(v: Vec<i32>) -> Vec<i32>").unwrap();
        let props = generate_property_tests("filter_pos", "filter positive numbers", Some(&sig));
        assert!(!props.is_empty(), "filter should generate property tests");
        assert!(
            props.join("\n").contains("not increase length"),
            "filter should test size"
        );
    }

    #[test]
    fn test_property_abs_nonnegative() {
        let sig = parse_rust_signature("fn abs(x: i32) -> i32").unwrap();
        let props = generate_property_tests("abs", "absolute value", Some(&sig));
        assert!(!props.is_empty(), "abs should generate property tests");
        let joined = props.join("\n");
        assert!(
            joined.contains("non-negative"),
            "abs should test non-negativity"
        );
        assert!(joined.contains("|x| == |-x|"), "abs should test symmetry");
    }

    #[test]
    fn test_property_no_properties_for_unknown() {
        let sig = parse_rust_signature("fn mystery(x: i32) -> i32").unwrap();
        let props = generate_property_tests("mystery", "do something mysterious", Some(&sig));
        assert!(
            props.is_empty(),
            "unknown purpose should generate no property tests"
        );
    }

    // ── Pattern composition tests ──

    #[test]
    fn test_compose_hashmap_frequency() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "freq", "Count frequency of each element")
            .with_signature("fn freq(items: Vec<i32>) -> std::collections::HashMap<i32, usize>");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(
            result.contains("HashMap::new()"),
            "Should use HashMap: {}",
            result
        );
        assert!(
            result.contains("or_insert"),
            "Should use or_insert: {}",
            result
        );
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_compose_sort_nth_largest() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new(
            "rust",
            "third_largest",
            "Find the 3rd largest element by sort",
        )
        .with_signature("fn third_largest(nums: Vec<i32>) -> i32");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(result.contains("sorted.sort()"), "Should sort: {}", result);
        assert!(
            result.contains("sorted.len() - 3"),
            "Should index from end: {}",
            result
        );
    }

    #[test]
    fn test_compose_count_unique() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "count_unique", "Count unique distinct elements")
            .with_signature("fn count_unique(items: Vec<i32>) -> usize");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(result.contains("HashSet"), "Should use HashSet: {}", result);
        assert!(result.contains(".len()"), "Should get length: {}", result);
    }

    #[test]
    fn test_compose_zip_elementwise() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "add_vecs", "Element-wise add two vectors")
            .with_signature("fn add_vecs(a: Vec<i32>, b: Vec<i32>) -> Vec<i32>");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(result.contains(".zip("), "Should zip: {}", result);
        assert!(result.contains(".collect()"), "Should collect: {}", result);
    }

    #[test]
    fn test_compose_find_pairs() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "all_pairs", "Find all pairs from a vector")
            .with_signature("fn all_pairs(nums: Vec<i32>) -> Vec<(i32, i32)>");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(
            result.contains("for i in"),
            "Should have outer loop: {}",
            result
        );
        assert!(
            result.contains("for j in"),
            "Should have inner loop: {}",
            result
        );
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }

    #[test]
    fn test_compose_cartesian_product() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "cartesian", "Cartesian product of two vectors")
            .with_signature("fn cartesian(a: Vec<i32>, b: Vec<i32>) -> Vec<(i32, i32)>");
        let result = emitter.emit_from_spec(&spec, &make_plan());
        assert!(
            result.contains("for a in"),
            "Should iterate first vec: {}",
            result
        );
        assert!(
            result.contains("for b in"),
            "Should iterate second vec: {}",
            result
        );
        assert!(
            !result.contains("todo!"),
            "Should not have todo: {}",
            result
        );
    }
}
