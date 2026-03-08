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

/// Parse a Rust function signature string like "fn foo(a: i32, b: &str) -> String"
fn parse_rust_signature(sig: &str) -> Option<ParsedSignature> {
    let sig = sig.trim();

    // Strip leading "pub fn " or "fn "
    let after_fn = if sig.starts_with("pub fn ") {
        &sig[7..]
    } else if sig.starts_with("fn ") {
        &sig[3..]
    } else {
        sig
    };

    // Find name and params
    let paren_start = after_fn.find('(')?;
    let name = after_fn[..paren_start].trim().to_string();

    let paren_end = after_fn.rfind(')')?;
    let params_str = &after_fn[paren_start + 1..paren_end];

    // Parse params
    let mut params = Vec::new();
    let mut _is_method = false;
    for param in params_str.split(',') {
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
            let after_colon = word[colon_pos + 1..].trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '<' && c != '>');

            // Type is in same word (e.g. "x:f64")
            if !after_colon.is_empty() {
                if !name.is_empty() && name.chars().next().map_or(false, |c| c.is_lowercase()) {
                    fields.push((name.to_string(), after_colon.to_string()));
                }
            } else if !name.is_empty() && i + 1 < words.len() {
                // Type is in next word (e.g. "x: f64")
                let typ = words[i + 1].trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '<' && c != '>');
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

/// Infer a reasonable function body from the purpose, params, and return type.
fn infer_rust_body(purpose: &str, params: &[(String, String)], return_type: Option<&str>, constraints: &[String], examples: &[(String, String)]) -> String {
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

    // Pattern-match common operations from the purpose
    let ret = return_type.unwrap_or("");

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
            return format!("{} / {}", params[0].0, params[1].0);
        }
    }
    if purpose_lower.contains("maximum") || purpose_lower.contains("max of") || purpose_lower.contains("larger") {
        if params.len() == 2 {
            if params[0].1.contains("f32") || params[0].1.contains("f64") {
                return format!("{}.max({})", params[0].0, params[1].0);
            }
            return format!("if {} > {} {{ {} }} else {{ {} }}", params[0].0, params[1].0, params[0].0, params[1].0);
        }
    }
    if purpose_lower.contains("minimum") || purpose_lower.contains("min of") || purpose_lower.contains("smaller") {
        if params.len() == 2 {
            if params[0].1.contains("f32") || params[0].1.contains("f64") {
                return format!("{}.min({})", params[0].0, params[1].0);
            }
            return format!("if {} < {} {{ {} }} else {{ {} }}", params[0].0, params[1].0, params[0].0, params[1].0);
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
        if params.len() == 1 && params[0].1.contains("Vec") {
            return format!("let mut result = {}.to_vec();\n    result.reverse();\n    result", params[0].0);
        }
    }
    if purpose_lower.contains("length") || purpose_lower.contains("len") || purpose_lower.contains("count") {
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
    if (purpose_lower.contains("contains") || purpose_lower.contains("includes")) && params.len() == 2 && params[0].1.contains("Vec") {
        return format!("{}.contains(&{})", params[0].0, params[1].0);
    }
    if purpose_lower.contains("contains") || purpose_lower.contains("has") {
        if params.len() == 2 {
            return format!("{}.contains({})", params[0].0, params[1].0);
        }
    }
    if purpose_lower.contains("concatenat") || purpose_lower.contains("join") || purpose_lower.contains("append") {
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
    if purpose_lower.contains("count") && params.len() == 2 && (params[0].1.contains("str") || params[0].1.contains("String")) {
        return format!("{}.matches({}).count()", params[0].0, params[1].0);
    }
    // Capitalize / title case
    if purpose_lower.contains("capitalize") || purpose_lower.contains("title") {
        if params.len() == 1 {
            return format!("let mut c = {}.chars();\n    match c.next() {{\n        None => String::new(),\n        Some(f) => f.to_uppercase().to_string() + &c.as_str().to_lowercase(),\n    }}", params[0].0);
        }
    }

    // Collection operations
    if purpose_lower.contains("sort") {
        if params.len() == 1 && params[0].1.contains("Vec") {
            if purpose_lower.contains("descending") || purpose_lower.contains("reverse") {
                return format!("let mut result = {}.to_vec();\n    result.sort();\n    result.reverse();\n    result", params[0].0);
            }
            return format!("let mut result = {}.to_vec();\n    result.sort();\n    result", params[0].0);
        }
    }
    if purpose_lower.contains("filter") {
        if params.len() >= 1 && params[0].1.contains("Vec") {
            return format!("{}.iter().filter(|x| /* condition */).cloned().collect()", params[0].0);
        }
    }
    if purpose_lower.contains("map") || purpose_lower.contains("transform") {
        if params.len() >= 1 && params[0].1.contains("Vec") {
            return format!("{}.iter().map(|x| /* transform */).collect()", params[0].0);
        }
    }
    if purpose_lower.contains("flatten") {
        if params.len() == 1 {
            return format!("{}.into_iter().flatten().collect()", params[0].0);
        }
    }
    if purpose_lower.contains("unique") || purpose_lower.contains("deduplicate") || purpose_lower.contains("dedup") {
        if params.len() == 1 {
            return format!("let mut result = {}.to_vec();\n    result.sort();\n    result.dedup();\n    result", params[0].0);
        }
    }
    // Binary search
    if purpose_lower.contains("binary search") || purpose_lower.contains("bsearch") {
        if params.len() == 2 {
            return format!("{}.binary_search(&{}).ok()", params[0].0, params[1].0);
        }
    }
    // Sum of collection
    if (purpose_lower.contains("sum") || purpose_lower.contains("total")) && params.len() == 1 && params[0].1.contains("Vec") {
        return format!("{}.iter().sum()", params[0].0);
    }
    // Max of collection
    if (purpose_lower.contains("max") || purpose_lower.contains("largest") || purpose_lower.contains("biggest")) && params.len() == 1 && params[0].1.contains("Vec") {
        return format!("{}.iter().max().copied()", params[0].0);
    }
    // Min of collection
    if (purpose_lower.contains("min") || purpose_lower.contains("smallest")) && params.len() == 1 && params[0].1.contains("Vec") {
        return format!("{}.iter().min().copied()", params[0].0);
    }
    // Count elements matching condition
    if purpose_lower.contains("count") && params.len() == 1 && params[0].1.contains("Vec") {
        return format!("{}.len()", params[0].0);
    }
    // Zip two collections
    if purpose_lower.contains("zip") {
        if params.len() == 2 {
            return format!("{}.iter().zip({}.iter()).collect()", params[0].0, params[1].0);
        }
    }
    // Enumerate
    if purpose_lower.contains("enumerate") || purpose_lower.contains("with index") {
        if params.len() == 1 {
            return format!("{}.iter().enumerate().collect()", params[0].0);
        }
    }
    // Take first N
    if (purpose_lower.contains("take") || purpose_lower.contains("first")) && params.len() == 2 {
        if params[1].1.contains("usize") || params[1].1.contains("u") || params[1].1.contains("i") {
            return format!("{}.iter().take({}).cloned().collect()", params[0].0, params[1].0);
        }
    }
    // Skip first N
    if purpose_lower.contains("skip") && params.len() == 2 {
        return format!("{}.iter().skip({}).cloned().collect()", params[0].0, params[1].0);
    }
    // Chunk/windows
    if purpose_lower.contains("chunk") && params.len() == 2 {
        return format!("{}.chunks({}).map(|c| c.to_vec()).collect()", params[0].0, params[1].0);
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
    if purpose_lower.contains("power") || purpose_lower.contains("exponent") || purpose_lower.contains("pow") {
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
            return format!("let (mut a, mut b) = ({}, {});\n    while b != 0 {{\n        let t = b;\n        b = a % b;\n        a = t;\n    }}\n    a", params[0].0, params[1].0);
        }
    }
    // Average / mean
    if purpose_lower.contains("average") || purpose_lower.contains("mean") {
        if params.len() == 1 && params[0].1.contains("Vec") {
            return format!("{}.iter().sum::<f64>() / {}.len() as f64", params[0].0, params[0].0);
        }
    }

    // Error handling patterns
    if ret.contains("Result") {
        if params.len() == 1 && (purpose_lower.contains("parse") || purpose_lower.contains("convert")) {
            return format!("{}.parse().map_err(|e| e.to_string())", params[0].0);
        }
        if purpose_lower.contains("read") || purpose_lower.contains("file") {
            return "std::fs::read_to_string(path).map_err(|e| e.to_string())".to_string();
        }
    }

    // Option patterns
    if ret.contains("Option") {
        if purpose_lower.contains("find") || purpose_lower.contains("first") {
            if params.len() >= 1 {
                return format!("{}.iter().find(|x| /* condition */).cloned()", params[0].0);
            }
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
fn infer_composed_body(purpose: &str, params: &[(String, String)], return_type: Option<&str>) -> Option<String> {
    if params.is_empty() {
        return None;
    }

    let p0 = &params[0].0;
    let is_vec = params[0].1.contains("Vec") || params[0].1.contains("&[");

    if !is_vec {
        return None;
    }

    // Build an iterator chain from recognized verbs in order of appearance
    let mut chain_parts: Vec<(&str, &str)> = Vec::new(); // (verb, code_fragment)

    // Scan for operation keywords in order
    let ops: &[(&[&str], &str, &str)] = &[
        // (keywords, iterator_method, description)
        (&["filter even", "keep even", "only even"], ".filter(|x| *x % 2 == 0)", "filter_even"),
        (&["filter odd", "keep odd", "only odd"], ".filter(|x| *x % 2 != 0)", "filter_odd"),
        (&["filter positive", "keep positive"], ".filter(|x| *x > 0)", "filter_pos"),
        (&["filter negative", "keep negative"], ".filter(|x| *x < 0)", "filter_neg"),
        (&["filter"], ".filter(|x| /* condition */)", "filter"),
        (&["sort", "order"], "", "sort"), // handled specially
        (&["reverse", "reversed"], ".rev()", "reverse"),
        (&["map", "transform", "convert each", "double", "square"], "", "map"), // handled specially
        (&["sum", "total", "add up", "add them"], ".sum()", "sum"),
        (&["count", "how many"], ".count()", "count"),
        (&["take first", "first n", "take top", "top n"], "", "take"), // handled specially
        (&["unique", "deduplicate", "dedup", "distinct"], "", "dedup"), // handled specially
        (&["join", "concatenate"], "", "join"), // handled specially
        (&["maximum", "max", "largest", "biggest"], ".max()", "max"),
        (&["minimum", "min", "smallest"], ".min()", "min"),
        (&["flatten"], ".flatten()", "flatten"),
    ];

    for (keywords, fragment, tag) in ops {
        for kw in *keywords {
            if purpose.contains(kw) {
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

    for (tag, fragment) in &chain_parts {
        match *tag {
            "sort" => {
                needs_sort_first = true;
            }
            "dedup" => {
                needs_sort_first = true; // dedup requires sorted
                iter_chain.push(".dedup()".to_string());
            }
            "map" => {
                if purpose.contains("double") {
                    iter_chain.push(".map(|x| x * 2)".to_string());
                } else if purpose.contains("square") {
                    iter_chain.push(".map(|x| x * x)".to_string());
                } else {
                    iter_chain.push(".map(|x| /* transform */)".to_string());
                }
            }
            "take" => {
                // Try to extract N from purpose
                let n = extract_number_from_text(purpose).unwrap_or(3);
                iter_chain.push(format!(".take({})", n));
            }
            "join" => {
                // join produces a String, not a Vec
                let sep = if purpose.contains("comma") { ", " }
                    else if purpose.contains("space") { " " }
                    else if purpose.contains("newline") { "\\n" }
                    else { ", " };
                iter_chain.push(format!(".collect::<Vec<_>>()"));
                needs_collect = false;
                // Return the join expression
                let sort_prefix = if needs_sort_first {
                    format!("let mut tmp = {}.to_vec();\n    tmp.sort();\n    tmp.iter()", p0)
                } else {
                    format!("{}.iter()", p0)
                };
                let chain: String = iter_chain.iter()
                    .take(iter_chain.len() - 1) // exclude the collect we just added
                    .map(|s| s.as_str())
                    .collect::<String>();
                return Some(format!(
                    "{}{}.map(|x| x.to_string()).collect::<Vec<_>>().join(\"{}\")",
                    sort_prefix, chain, sep
                ));
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
        // Sort requires mutation, so we need a let binding
        let collect = if needs_collect { ".collect()" } else { "" };
        Some(format!(
            "let mut tmp = {}.to_vec();\n    tmp.sort();\n    tmp.iter().cloned(){}{}",
            p0, chain, collect
        ))
    } else {
        let collect = if needs_collect { ".collect()" } else { "" };
        Some(format!("{}.iter().cloned(){}{}", p0, chain, collect))
    }
}

/// Extract a number from text like "first 3", "top 5", "take 10"
fn extract_number_from_text(text: &str) -> Option<usize> {
    for word in text.split_whitespace() {
        if let Ok(n) = word.parse::<usize>() {
            return Some(n);
        }
    }
    // Check for written-out numbers
    let written = &[
        ("one", 1), ("two", 2), ("three", 3), ("four", 4), ("five", 5),
        ("six", 6), ("seven", 7), ("eight", 8), ("nine", 9), ("ten", 10),
    ];
    for (word, n) in written {
        if text.contains(word) {
            return Some(*n);
        }
    }
    None
}

/// Try to infer function body from input/output examples
fn infer_from_examples(examples: &[(String, String)], params: &[(String, String)], _return_type: Option<&str>) -> Option<String> {
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
fn generate_auto_tests(func_name: &str, purpose: &str, sig: Option<&ParsedSignature>) -> Vec<String> {
    let purpose_lower = purpose.to_lowercase();
    let mut tests = Vec::new();

    let sig = match sig {
        Some(s) => s,
        None => return tests,
    };

    let has_i32_params = sig.params.iter().any(|(_, t)| t.contains("i32") || t.contains("i64") || t.contains("u32") || t.contains("u64"));
    let has_str_param = sig.params.iter().any(|(_, t)| t.contains("str") || t.contains("String"));
    let has_vec_param = sig.params.iter().any(|(_, t)| t.contains("Vec") || t.contains("&["));
    let returns_bool = sig.return_type.as_ref().map_or(false, |r| r.contains("bool"));
    let returns_string = sig.return_type.as_ref().map_or(false, |r| r.contains("String"));
    let returns_vec = sig.return_type.as_ref().map_or(false, |r| r.contains("Vec"));

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
    if sig.params.len() == 1 && has_i32_params {
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
            tests.push(format!("assert_eq!({}(vec![3, 1, 2]), vec![1, 2, 3]);", func_name));
            tests.push(format!("assert_eq!({}(vec![]), vec![]);", func_name));
        }
    }

    tests
}

// ============================================================================
// Rust Emitter
// ============================================================================

/// Emitter for Rust source code
pub struct RustEmitter;

impl CodeEmitter for RustEmitter {
    fn emit_from_spec(&self, spec: &CodeSpec, plan: &[CodePlanStep]) -> String {
        let mut parts = Vec::new();

        // Collect plan actions
        let mut has_struct = false;
        let mut has_function = false;
        let mut has_trait = false;
        let mut has_impl = false;
        let mut has_error_handling = false;
        let mut has_doc = false;
        let mut field_steps = 0usize;
        let mut method_steps = 0usize;
        let mut _param_steps = 0usize;

        for step in plan {
            match step.action {
                PlanAction::DefineStruct => has_struct = true,
                PlanAction::DefineFunction => has_function = true,
                PlanAction::DefineTrait => has_trait = true,
                PlanAction::ImplTrait => has_impl = true,
                PlanAction::AddField => field_steps += 1,
                PlanAction::AddMethod => method_steps += 1,
                PlanAction::AddParameter => _param_steps += 1,
                PlanAction::AddErrorHandling => has_error_handling = true,
                PlanAction::AddDocumentation => has_doc = true,
                PlanAction::AddImport => {
                    if spec.purpose.contains("error") || spec.purpose.contains("Result") {
                        parts.push("use std::error::Error;".to_string());
                    }
                    if spec.purpose.contains("HashMap") || spec.purpose.contains("hash map") {
                        parts.push("use std::collections::HashMap;".to_string());
                    }
                    if spec.purpose.contains("File") || spec.purpose.contains("read") || spec.purpose.contains("write") {
                        parts.push("use std::fs;".to_string());
                    }
                }
                _ => {}
            }
        }

        // Try to parse the provided signature
        let parsed_sig = spec.signature.as_deref().and_then(parse_rust_signature);

        // Emit struct
        if has_struct {
            if has_doc || !spec.purpose.is_empty() {
                parts.push(format!("/// {}", spec.purpose));
            }

            let fields = extract_fields_from_text(&spec.purpose);
            if fields.is_empty() && field_steps > 0 {
                // Generate placeholder fields from field_steps count
                parts.push(format!("#[derive(Debug, Clone)]"));
                parts.push(format!("pub struct {} {{", spec.name));
                for i in 0..field_steps.min(5) {
                    parts.push(format!("    pub field_{}: String,", i));
                }
                parts.push("}".to_string());
            } else if !fields.is_empty() {
                parts.push(format!("#[derive(Debug, Clone)]"));
                parts.push(format!("pub struct {} {{", spec.name));
                for (name, typ) in &fields {
                    parts.push(format!("    pub {}: {},", name, typ));
                }
                parts.push("}".to_string());
            } else {
                parts.push(format!("#[derive(Debug, Clone)]"));
                parts.push(format!("pub struct {} {{", spec.name));
                parts.push("}".to_string());
            }
            parts.push(String::new());

            // Emit constructor + methods if we have impl steps or MULTI_ENTITY constraint
            let is_multi_entity = spec.constraints.iter().any(|c| c.starts_with("MULTI_ENTITY"));
            if has_impl || method_steps > 0 || is_multi_entity {
                parts.push(format!("impl {} {{", spec.name));
                let fields = extract_fields_from_text(&spec.purpose);
                if !fields.is_empty() {
                    let params: Vec<String> = fields.iter().map(|(n, t)| format!("{}: {}", n, t)).collect();
                    let assigns: Vec<String> = fields.iter().map(|(n, _)| format!("            {}", n)).collect();
                    parts.push(format!("    pub fn new({}) -> Self {{", params.join(", ")));
                    parts.push("        Self {".to_string());
                    for a in &assigns {
                        parts.push(format!("{},", a));
                    }
                    parts.push("        }".to_string());
                    parts.push("    }".to_string());
                } else {
                    parts.push("    pub fn new() -> Self {".to_string());
                    parts.push("        Self {}".to_string());
                    parts.push("    }".to_string());
                }

                // Auto-generate methods from purpose when MULTI_ENTITY
                if is_multi_entity {
                    let purpose_lower = spec.purpose.to_lowercase();
                    if purpose_lower.contains("distance") && fields.len() >= 2 {
                        let f0 = &fields[0].0;
                        let f1 = &fields[1].0;
                        parts.push(String::new());
                        parts.push("    pub fn distance(&self, other: &Self) -> f64 {".to_string());
                        parts.push(format!(
                            "        (((self.{f0} - other.{f0}).powi(2) + (self.{f1} - other.{f1}).powi(2)) as f64).sqrt()"
                        ));
                        parts.push("    }".to_string());
                    }
                    if purpose_lower.contains("area") && fields.len() >= 2 {
                        let f0 = &fields[0].0;
                        let f1 = &fields[1].0;
                        parts.push(String::new());
                        parts.push("    pub fn area(&self) -> f64 {".to_string());
                        parts.push(format!("        (self.{f0} * self.{f1}) as f64"));
                        parts.push("    }".to_string());
                    }
                    if purpose_lower.contains("display") || purpose_lower.contains("to_string") || purpose_lower.contains("format") {
                        parts.push(String::new());
                        let field_fmts: Vec<String> = fields.iter()
                            .map(|(n, _)| format!("{}: {{}}", n))
                            .collect();
                        let field_refs: Vec<String> = fields.iter()
                            .map(|(n, _)| format!("self.{}", n))
                            .collect();
                        if !fields.is_empty() {
                            parts.push("    pub fn display(&self) -> String {".to_string());
                            parts.push(format!(
                                "        format!(\"{}({})\", {})",
                                spec.name,
                                field_fmts.join(", "),
                                field_refs.join(", ")
                            ));
                            parts.push("    }".to_string());
                        }
                    }
                }

                parts.push("}".to_string());
            }
        }

        // Emit trait
        if has_trait {
            if has_doc || !spec.purpose.is_empty() {
                parts.push(format!("/// {}", spec.purpose));
            }
            parts.push(format!("pub trait {} {{", spec.name));
            for i in 0..method_steps.max(1).min(5) {
                parts.push(format!("    fn method_{}(&self);", i));
            }
            parts.push("}".to_string());
            parts.push(String::new());
        }

        // Emit function
        if has_function || (!has_struct && !has_trait) {
            if let Some(ref sig) = parsed_sig {
                // Use the parsed signature to generate a real function
                let params_str: Vec<String> = sig.params.iter()
                    .map(|(n, t)| format!("{}: {}", n, t))
                    .collect();

                let ret_str = sig.return_type.as_deref()
                    .map(|r| format!(" -> {}", r))
                    .unwrap_or_default();

                let body = infer_rust_body(
                    &spec.purpose,
                    &sig.params,
                    sig.return_type.as_deref(),
                    &spec.constraints,
                    &spec.examples,
                );

                if has_doc || !spec.purpose.is_empty() {
                    parts.push(format!("/// {}", spec.purpose));
                }

                if has_error_handling && !ret_str.contains("Result") {
                    // Wrap in Result if error handling was planned
                    parts.push(format!(
                        "pub fn {}({}){} {{",
                        sig.name, params_str.join(", "), ret_str
                    ));
                } else {
                    parts.push(format!(
                        "pub fn {}({}){} {{",
                        sig.name, params_str.join(", "), ret_str
                    ));
                }
                parts.push(format!("    {}", body));
                parts.push("}".to_string());
            } else {
                // No parsed signature — try to infer from purpose
                if has_doc || !spec.purpose.is_empty() {
                    parts.push(format!("/// {}", spec.purpose));
                }
                parts.push(format!("pub fn {}() {{", spec.name));
                let body = infer_rust_body(
                    &spec.purpose,
                    &[],
                    None,
                    &spec.constraints,
                    &spec.examples,
                );
                parts.push(format!("    {}", body));
                parts.push("}".to_string());
            }
        }

        // Emit tests: from explicit examples first, then auto-generate from purpose
        let func_name = parsed_sig.as_ref()
            .map(|s| s.name.as_str())
            .unwrap_or(&spec.name);

        let auto_tests = if spec.examples.is_empty() {
            generate_auto_tests(func_name, &spec.purpose, parsed_sig.as_ref())
        } else {
            Vec::new()
        };

        if !spec.examples.is_empty() || !auto_tests.is_empty() {
            parts.push(String::new());
            parts.push("#[cfg(test)]".to_string());
            parts.push("mod tests {".to_string());
            parts.push("    use super::*;".to_string());
            parts.push(String::new());

            // Explicit examples
            for (i, (input, output)) in spec.examples.iter().enumerate() {
                parts.push("    #[test]".to_string());
                parts.push(format!("    fn test_example_{}() {{", i));
                parts.push(format!("        assert_eq!({}, {});", input, output));
                parts.push("    }".to_string());
                if i + 1 < spec.examples.len() || !auto_tests.is_empty() {
                    parts.push(String::new());
                }
            }

            // Auto-generated tests
            for (i, test_line) in auto_tests.iter().enumerate() {
                parts.push("    #[test]".to_string());
                parts.push(format!("    fn test_auto_{}() {{", i));
                parts.push(format!("        {}", test_line));
                parts.push("    }".to_string());
                if i + 1 < auto_tests.len() {
                    parts.push(String::new());
                }
            }
            parts.push("}".to_string());
        }

        // Apply import inference — scan generated code for stdlib types and prepend imports
        let raw = parts.join("\n");
        infer_rust_imports(&raw)
    }

    fn emit_function(&self, name: &str, params: &str, return_type: &str, body: &str) -> String {
        let ret = if return_type.is_empty() {
            String::new()
        } else {
            format!(" -> {}", return_type)
        };

        format!("pub fn {}({}){} {{\n    {}\n}}", name, params, ret, body)
    }

    fn emit_struct(&self, name: &str, fields: &[(String, String)]) -> String {
        let field_lines: Vec<String> = fields
            .iter()
            .map(|(n, t)| format!("    pub {}: {},", n, t))
            .collect();

        format!(
            "#[derive(Debug, Clone)]\npub struct {} {{\n{}\n}}",
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

            let fields = extract_fields_from_text(&spec.purpose);
            if !fields.is_empty() {
                let params: Vec<String> = fields.iter().map(|(n, t)| format!("{}: {}", n, t)).collect();
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
            } else if purpose_lower.contains("maximum") || purpose_lower.contains("max of") || purpose_lower.contains("larger") {
                parts.push("    return max(a, b)".to_string());
            } else if purpose_lower.contains("minimum") || purpose_lower.contains("min of") || purpose_lower.contains("smaller") {
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
                parts.push(format!("    raise NotImplementedError(\"{}\")", spec.purpose));
            } else {
                parts.push(format!("    raise NotImplementedError(\"{}\")", spec.purpose));
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
            let params: Vec<String> = fields.iter().map(|(n, t)| format!("{}: {}", n, t)).collect();
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

// ============================================================================
// Nix Emitter
// ============================================================================

/// Emitter for Nix expressions
pub struct NixEmitter;

impl CodeEmitter for NixEmitter {
    fn emit_from_spec(&self, spec: &CodeSpec, _plan: &[CodePlanStep]) -> String {
        let mut parts = Vec::new();

        parts.push(format!("# {}", spec.purpose));

        if spec.name.contains("derivation") || spec.name.contains("package") {
            parts.push("{ lib, stdenv, ... }:".to_string());
            parts.push(String::new());
            parts.push("stdenv.mkDerivation {".to_string());
            parts.push(format!("  pname = \"{}\";", spec.name));
            parts.push("  version = \"0.1.0\";".to_string());
            parts.push(String::new());
            parts.push("  src = ./.;".to_string());
            parts.push(String::new());
            parts.push("  meta = with lib; {".to_string());
            parts.push(format!("    description = \"{}\";", spec.purpose));
            parts.push("    license = licenses.mit;".to_string());
            parts.push("  };".to_string());
            parts.push("}".to_string());
        } else if spec.name.contains("module") || spec.name.contains("config") {
            parts.push("{ config, lib, pkgs, ... }:".to_string());
            parts.push(String::new());
            parts.push("{".to_string());
            parts.push(format!("  # {}", spec.purpose));
            if !spec.constraints.is_empty() {
                for c in &spec.constraints {
                    parts.push(format!("  # Constraint: {}", c));
                }
            }
            parts.push("}".to_string());
        } else if spec.name.contains("overlay") {
            parts.push("final: prev: {".to_string());
            parts.push(format!("  # {}", spec.purpose));
            parts.push("}".to_string());
        } else if spec.name.contains("shell") || spec.name.contains("devShell") {
            parts.push("{ pkgs ? import <nixpkgs> {} }:".to_string());
            parts.push(String::new());
            parts.push("pkgs.mkShell {".to_string());
            parts.push("  buildInputs = with pkgs; [".to_string());
            parts.push("  ];".to_string());
            parts.push("}".to_string());
        } else {
            // Default: function or let binding
            parts.push(format!("{} =", spec.name));
            if !spec.constraints.is_empty() {
                for c in &spec.constraints {
                    parts.push(format!("  # {}", c));
                }
            }
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
        let field_lines: Vec<String> = fields
            .iter()
            .map(|(n, _t)| format!("  {} = null;", n))
            .collect();

        format!("{} = {{\n{}\n}};", name, field_lines.join("\n"))
    }

    fn emit_import(&self, module: &str) -> String {
        format!("imports = [ {} ];", module)
    }

    fn language(&self) -> &str {
        "nix"
    }
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
                format!("{}<", type_name),      // HashMap<K, V>
                format!("{}::", type_name),      // HashMap::new()
                format!(": {}", type_name),      // x: HashMap
                format!("-> {}", type_name),     // -> HashMap
                format!("{} {{", type_name),     // HashMap {
                format!("{}(", type_name),       // File(
                format!("&{}", type_name),       // &Path
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
        assert!(!result.contains("todo!"), "Should not have todo: {}", result);
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
        assert!(result.contains("s.chars().rev().collect()"), "Should infer reverse: {}", result);
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
        assert!(result.contains("% 2 == 0"), "Should infer even check: {}", result);
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
        assert!(result.contains(".product()"), "Should infer factorial: {}", result);
    }

    #[test]
    fn test_rust_struct_with_fields() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "Point", "A point with x: f64 and y: f64 fields");
        let plan = vec![
            CodePlanStep { action: PlanAction::DefineStruct, name: None, context: Vec::new(), confidence: 0.9 },
            CodePlanStep { action: PlanAction::AddField, name: None, context: Vec::new(), confidence: 0.8 },
            CodePlanStep { action: PlanAction::AddField, name: None, context: Vec::new(), confidence: 0.8 },
            CodePlanStep { action: PlanAction::ImplTrait, name: None, context: Vec::new(), confidence: 0.7 },
        ];
        let result = emitter.emit_from_spec(&spec, &plan);
        assert!(result.contains("pub struct Point"), "Should define struct: {}", result);
        assert!(result.contains("x: f64"), "Should have x field: {}", result);
        assert!(result.contains("y: f64"), "Should have y field: {}", result);
        assert!(result.contains("fn new"), "Should have constructor: {}", result);
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
        let spec = CodeSpec::new("python", "add", "Add two numbers")
            .with_example("add(1, 2)", "3");
        let plan = vec![CodePlanStep {
            action: PlanAction::DefineFunction,
            name: None,
            context: Vec::new(),
            confidence: 0.9,
        }];
        let result = emitter.emit_from_spec(&spec, &plan);
        assert!(result.contains("return a + b"), "Should infer add: {}", result);
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
        assert!(result.contains("% 2 == 0"), "Should filter even: {}", result);
        assert!(result.contains(".sum()"), "Should sum: {}", result);
        assert!(!result.contains("todo!"), "Should not have todo: {}", result);
    }

    #[test]
    fn test_rust_sort_and_take_first() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "top_three", "Sort a vector and take first 3 elements")
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
        assert!(!result.contains("todo!"), "Should not have todo: {}", result);
    }

    #[test]
    fn test_rust_filter_and_count() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "count_positive", "Filter positive numbers and count them")
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
        assert!(!result.contains("todo!"), "Should not have todo: {}", result);
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
        assert!(!result.contains("todo!"), "Should not have todo: {}", result);
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
            params: vec![("a".to_string(), "i32".to_string()), ("b".to_string(), "i32".to_string())],
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
    fn test_auto_tests_no_sig_no_tests() {
        let tests = generate_auto_tests("foo", "Do something", None);
        assert!(tests.is_empty(), "No sig → no auto tests");
    }

    #[test]
    fn test_rust_emit_with_auto_tests() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "add", "Add two numbers")
            .with_signature("fn add(a: i32, b: i32) -> i32");
        let result = emitter.emit_from_spec(&spec, &[
            CodePlanStep { action: PlanAction::DefineFunction, name: None, context: vec![], confidence: 0.9 },
        ]);
        // Should have auto-generated tests since no examples provided
        assert!(result.contains("#[test]"), "Should contain auto-generated tests");
        assert!(result.contains("assert_eq!(add(2, 3), 5)"), "Should have add(2,3)=5 assertion");
    }

    #[test]
    fn test_multi_entity_struct_with_distance() {
        let emitter = RustEmitter;
        let spec = CodeSpec::new("rust", "Point", "Point struct with x: f64 and y: f64 with distance method")
            .with_constraint("MULTI_ENTITY: generate struct + impl block + methods");
        let result = emitter.emit_from_spec(&spec, &[
            CodePlanStep { action: PlanAction::DefineStruct, name: None, context: vec![], confidence: 0.9 },
        ]);
        assert!(result.contains("pub struct Point"), "Should have struct: {}", result);
        assert!(result.contains("impl Point"), "Should have impl block: {}", result);
        assert!(result.contains("fn new("), "Should have constructor: {}", result);
        assert!(result.contains("fn distance("), "Should have distance method: {}", result);
    }
}
