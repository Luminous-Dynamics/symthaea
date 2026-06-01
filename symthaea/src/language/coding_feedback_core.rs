// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Lightweight coding-feedback classification.
//!
//! This module intentionally avoids the full `code_generation` stack so semantic
//! repair taxonomy can be tested quickly and reused by Broca/evaluation code.

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
    fn classifies_semantic_failure_shapes_without_codegen_feature() {
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
    fn classifies_non_value_test_failures_without_codegen_feature() {
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
    fn semantic_failure_hints_are_actionable_without_codegen_feature() {
        assert!(
            repair_hint_for_prediction_error_category("off_by_one").contains("inclusive/exclusive")
        );
        assert!(repair_hint_for_prediction_error_category("boolean_inversion").contains("Invert"));
        assert!(repair_hint_for_prediction_error_category("test_panic").contains("unchecked"));
    }
}
