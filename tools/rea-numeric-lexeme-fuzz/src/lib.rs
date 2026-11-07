//! Deterministic properties wrapped around the raw numeric-lexeme auditor.

use rea_numeric_lexeme_audit::{AuditError, NumericLexemeAudit, audit_json};

pub const MAX_FUZZ_INPUT_BYTES: usize = 1024 * 1024;

/// Exercise a raw JSON document. Invalid or over-limit inputs are ordinary
/// parser outcomes; accepted inputs must be deterministic and whitespace
/// invariant except for their exact-source digest.
pub fn exercise_raw_input(input: &[u8]) -> Result<(), AuditError> {
    if input.len() > MAX_FUZZ_INPUT_BYTES {
        return Ok(());
    }

    let Ok(first) = audit_json(input) else {
        return Ok(());
    };
    assert_deterministic(input, &first);

    let mut with_outer_whitespace = Vec::with_capacity(input.len() + 3);
    with_outer_whitespace.push(b' ');
    with_outer_whitespace.extend_from_slice(input);
    with_outer_whitespace.extend_from_slice(b"\n\t");

    let spaced = audit_json(&with_outer_whitespace)?;
    assert_eq!(
        first.visited_values, spaced.visited_values,
        "outer whitespace changed JSON value traversal"
    );
    assert_eq!(
        first.numeric_precision_warnings, spaced.numeric_precision_warnings,
        "outer whitespace changed raw integer warning paths or lexemes"
    );
    assert_ne!(
        first.source_raw_sha256, spaced.source_raw_sha256,
        "the exact-source digest must bind added whitespace bytes"
    );
    Ok(())
}

fn assert_deterministic(input: &[u8], first: &NumericLexemeAudit) {
    let second = audit_json(input).expect("the same accepted source must remain accepted");
    assert_eq!(*first, second, "repeated audit was not deterministic");
    assert!(
        first
            .numeric_precision_warnings
            .windows(2)
            .all(|pair| {
                pair[0]
                    .json_pointer
                    .cmp(&pair[1].json_pointer)
                    .then_with(|| pair[0].integer_lexeme.cmp(&pair[1].integer_lexeme))
                    .is_le()
            }),
        "precision warnings must be sorted by pointer and lexeme"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepted_integer_document_is_deterministic_and_whitespace_invariant() {
        exercise_raw_input(br#"{"large":9007199254740993,"nested":[-9007199254740992]}"#)
            .unwrap();
    }

    #[test]
    fn decimal_exponent_and_string_controls_remain_unwarned() {
        exercise_raw_input(br#"{"decimal":1.0,"exp":1e+30,"string":"9007199254740993"}"#)
            .unwrap();
    }

    #[test]
    fn malformed_or_duplicate_documents_are_safe_parser_outcomes() {
        exercise_raw_input(br#"{"n":1,"n":9007199254740992}"#).unwrap();
        exercise_raw_input(br#"{"n":01}"#).unwrap();
        exercise_raw_input(b"").unwrap();
    }

    #[test]
    fn oversized_fuzz_inputs_are_skipped_before_audit() {
        let input = vec![b' '; MAX_FUZZ_INPUT_BYTES + 1];
        exercise_raw_input(&input).unwrap();
    }
}
