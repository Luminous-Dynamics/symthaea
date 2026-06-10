// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Lightweight coding-feedback classification.
//!
//! This crate intentionally has no external dependencies. It keeps semantic
//! repair taxonomy and test-failure classification fast to test and reusable
//! across Symthaea's generator, Broca evaluator, and future training pipelines.

pub fn repair_hint_for_prediction_error_category(category: &str) -> &'static str {
    match category {
        "off_by_one" => {
            "Check inclusive/exclusive bounds, index starts, loop ranges, and final accumulator adjustment."
        }
        "boolean_inversion" => {
            "Invert the predicate or swap the true/false branch while preserving the function signature."
        }
        "numeric_value_mismatch" => {
            "Re-evaluate the arithmetic formula, accumulator update, and operator choice against the examples."
        }
        "empty_result_mismatch" => {
            "Check initialization, early returns, filters, and whether matching items are pushed into the result."
        }
        "order_mismatch" => {
            "Preserve the required ordering or sort deterministically before returning."
        }
        "partial_value_mismatch" => {
            "Check string/list assembly for missing, extra, or truncated components."
        }
        "semantic_value_mismatch" => {
            "Treat the expected/actual pair as the semantic contract and adjust the core transformation."
        }
        "test_timeout" => {
            "Look for non-terminating loops or algorithms that do not reduce their search space."
        }
        "test_panic" => {
            "Remove unchecked indexing/unwrap paths or handle invalid inputs before executing the operation."
        }
        _ => "Use the diagnostic as an executable semantic constraint.",
    }
}

pub fn categorize_rejection(reason: &str) -> &'static str {
    let lower = reason.to_ascii_lowercase();
    if lower.contains("energy budget") {
        "energy_budget"
    } else if lower.contains("stub") || lower.contains("todo") || lower.contains("unimplemented") {
        "stub"
    } else if lower.contains("does not parse")
        || lower.contains("parse")
        || lower.contains("expected one of")
    {
        "parse_failure"
    } else if lower.contains("not found in this scope")
        || lower.contains("cannot find value")
        || lower.contains("unresolved")
    {
        "unresolved_identifier"
    } else if lower.contains("mismatched types")
        || lower.contains("type mismatch")
        || lower.contains("returns a value")
        || lower.contains("unit-returning")
        || lower.contains("return type")
        || lower.contains("signature expects")
    {
        "type_mismatch"
    } else if lower.contains("borrow") || lower.contains("moved") || lower.contains("move") {
        "ownership"
    } else if lower.contains("test") {
        "test_failure"
    } else if lower.contains("compile") || lower.contains("rustc") {
        "compile_failure"
    } else if lower.contains("sheaf") || lower.contains("coherence") {
        "sheaf_failure"
    } else if lower.contains("no similar") || lower.contains("analogy") {
        "analogy_miss"
    } else if lower.contains("llm") {
        "llm_failure"
    } else {
        "other"
    }
}

pub fn repair_hint_for_rejection_category(category: &str) -> &'static str {
    match category {
        "type_mismatch" | "compile_failure" => {
            "regenerate with the declared signature as the source of truth for arguments and return type"
        }
        "unresolved_identifier" => {
            "only use parameters and locals that are in scope, or introduce the binding first"
        }
        "ownership" => {
            "prefer borrowing, cloning, copying, or statement-only mutation according to the signature"
        }
        "stub" => "replace placeholders with concrete expressions or accumulator statements",
        "parse_failure" => {
            "emit complete Rust syntax with balanced delimiters and valid separators"
        }
        "sheaf_failure" => {
            "repair local data-flow facts: definitions, return shape, ownership, and stubs"
        }
        "off_by_one" => {
            "check inclusive/exclusive bounds, index origins, and final accumulator adjustment"
        }
        "boolean_inversion" => {
            "invert the predicate or swap branch bodies so the boolean semantic contract matches"
        }
        "numeric_value_mismatch" => {
            "repair the arithmetic formula, operator, or accumulator update against the examples"
        }
        "empty_result_mismatch" => {
            "check early returns, initial values, filters, and whether matching items are appended"
        }
        "order_mismatch" => "preserve insertion order or sort deterministically before returning",
        "partial_value_mismatch" => {
            "repair string/list assembly so no required component is missing, extra, or truncated"
        }
        "semantic_value_mismatch" => {
            "treat expected/actual values as the core semantic contract, not only a syntax issue"
        }
        "test_failure" => "treat examples and generated tests as executable semantic constraints",
        "analogy_miss" => "fall back to direct synthesis from the signature and task purpose",
        "energy_budget" => {
            "prefer low-cost deterministic templates before expensive fallback tiers"
        }
        _ => "use the previous failure as a concrete negative example for the next candidate",
    }
}

pub fn repair_lesson_for_rejection(reason: &str) -> String {
    if let Some(hint) = extract_embedded_repair_hint(reason) {
        return hint;
    }
    let category =
        extract_embedded_category(reason).unwrap_or_else(|| categorize_rejection(reason));
    repair_hint_for_rejection_category(category).to_string()
}

pub fn extract_embedded_category(reason: &str) -> Option<&str> {
    let start = reason.find("category=")? + "category=".len();
    let rest = &reason[start..];
    let end = rest.find([';', ']']).unwrap_or(rest.len());
    let category = rest[..end].trim();
    (!category.is_empty()).then_some(category)
}

pub fn extract_embedded_repair_hint(reason: &str) -> Option<String> {
    let start = reason.find("repair_hint=")? + "repair_hint=".len();
    let rest = &reason[start..];
    let end = rest.find(']').unwrap_or(rest.len());
    let hint = rest[..end].trim();
    (!hint.is_empty()).then(|| hint.to_string())
}

pub fn categorize_test_failure_diagnostic(diagnostic: &str) -> &'static str {
    let lower = diagnostic.to_ascii_lowercase();
    let Some((expected, actual)) = expected_actual_from_constraint(diagnostic) else {
        return if lower.contains("timeout") || lower.contains("timed out") {
            "test_timeout"
        } else if lower.contains("panic") || lower.contains("panicked") {
            "test_panic"
        } else {
            "test_failure"
        };
    };

    let expected_trimmed = normalize_test_value(&expected);
    let actual_trimmed = normalize_test_value(&actual);
    let expected_lower = expected_trimmed.to_ascii_lowercase();
    let actual_lower = actual_trimmed.to_ascii_lowercase();

    if matches!(
        (expected_lower.as_str(), actual_lower.as_str()),
        ("true", "false") | ("false", "true")
    ) {
        return "boolean_inversion";
    }

    if let (Some(expected_number), Some(actual_number)) = (
        parse_numeric_value(&expected_trimmed),
        parse_numeric_value(&actual_trimmed),
    ) {
        if (expected_number - actual_number).abs() == 1 {
            return "off_by_one";
        }
        return "numeric_value_mismatch";
    }

    if is_empty_value(&expected_trimmed) != is_empty_value(&actual_trimmed) {
        return "empty_result_mismatch";
    }

    if looks_like_collection(&expected_trimmed)
        && looks_like_collection(&actual_trimmed)
        && collection_items_sorted(&expected_trimmed) == collection_items_sorted(&actual_trimmed)
    {
        return "order_mismatch";
    }

    if !expected_trimmed.is_empty()
        && !actual_trimmed.is_empty()
        && (expected_lower.contains(&actual_lower) || actual_lower.contains(&expected_lower))
    {
        return "partial_value_mismatch";
    }

    "semantic_value_mismatch"
}

fn expected_actual_from_constraint(diagnostic: &str) -> Option<(String, String)> {
    let (_, rest) = diagnostic.split_once(" expected ")?;
    let (expected, actual) = rest.split_once(" but got ")?;
    Some((expected.trim().to_string(), actual.trim().to_string()))
}

fn normalize_test_value(value: &str) -> String {
    value
        .trim()
        .trim_matches('`')
        .trim_matches('"')
        .trim_matches('\'')
        .trim()
        .to_string()
}

fn parse_numeric_value(value: &str) -> Option<i128> {
    value
        .trim()
        .trim_end_matches(|ch: char| ch.is_ascii_alphabetic())
        .parse::<i128>()
        .ok()
}

fn is_empty_value(value: &str) -> bool {
    matches!(value.trim(), "[]" | "{}" | "\"\"" | "''" | "None" | "null")
}

fn looks_like_collection(value: &str) -> bool {
    let trimmed = value.trim();
    (trimmed.starts_with('[') && trimmed.ends_with(']'))
        || (trimmed.starts_with('{') && trimmed.ends_with('}'))
}

fn collection_items_sorted(value: &str) -> Vec<String> {
    let trimmed = value
        .trim()
        .trim_start_matches(['[', '{'])
        .trim_end_matches([']', '}']);
    let mut items = trimmed
        .split(',')
        .map(normalize_test_value)
        .filter(|item| !item.is_empty())
        .collect::<Vec<_>>();
    items.sort();
    items
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_semantic_failure_shapes() {
        assert_eq!(
            categorize_test_failure_diagnostic("CONSTRAINT: case expected 42 but got 41"),
            "off_by_one"
        );
        assert_eq!(
            categorize_test_failure_diagnostic("CONSTRAINT: case expected true but got false"),
            "boolean_inversion"
        );
        assert_eq!(
            categorize_test_failure_diagnostic(
                "CONSTRAINT: case expected [1, 2, 3] but got [3, 2, 1]"
            ),
            "order_mismatch"
        );
        assert_eq!(
            categorize_test_failure_diagnostic("CONSTRAINT: case expected [] but got [1]"),
            "empty_result_mismatch"
        );
        assert_eq!(
            categorize_test_failure_diagnostic(
                "CONSTRAINT: case expected \"foobar\" but got \"foo\""
            ),
            "partial_value_mismatch"
        );
    }

    #[test]
    fn classifies_non_value_test_failures() {
        assert_eq!(
            categorize_test_failure_diagnostic("test timed out after 30s"),
            "test_timeout"
        );
        assert_eq!(
            categorize_test_failure_diagnostic("thread panicked at index out of bounds"),
            "test_panic"
        );
        assert_eq!(
            categorize_test_failure_diagnostic("assertion failed"),
            "test_failure"
        );
    }

    #[test]
    fn semantic_failure_hints_are_actionable() {
        assert!(
            repair_hint_for_prediction_error_category("off_by_one").contains("inclusive/exclusive")
        );
        assert!(repair_hint_for_prediction_error_category("boolean_inversion").contains("Invert"));
        assert!(repair_hint_for_prediction_error_category("test_panic").contains("unchecked"));
    }

    #[test]
    fn classifies_rejection_shapes() {
        assert_eq!(
            categorize_rejection("candidate contains todo! placeholder"),
            "stub"
        );
        assert_eq!(
            categorize_rejection("cannot find value `items` in this scope"),
            "unresolved_identifier"
        );
        assert_eq!(
            categorize_rejection("borrow of moved value in generated candidate"),
            "ownership"
        );
        assert_eq!(
            categorize_rejection("CONSTRAINT: test expected 3 but got 4"),
            "test_failure"
        );
    }

    #[test]
    fn embedded_repair_lessons_take_precedence() {
        assert_eq!(
            extract_embedded_category("[category=off_by_one; repair_hint=check bounds]").unwrap(),
            "off_by_one"
        );
        assert_eq!(
            extract_embedded_repair_hint("[category=off_by_one; repair_hint=check bounds]")
                .unwrap(),
            "check bounds"
        );
        assert_eq!(
            repair_lesson_for_rejection("[category=off_by_one; repair_hint=check bounds]"),
            "check bounds"
        );
    }

    #[test]
    fn rejection_hints_cover_semantic_categories() {
        assert!(
            repair_hint_for_rejection_category("numeric_value_mismatch").contains("arithmetic")
        );
        assert!(repair_hint_for_rejection_category("energy_budget").contains("low-cost"));
    }
}
