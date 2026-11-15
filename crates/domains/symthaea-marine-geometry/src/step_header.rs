//! Minimal ISO 10303-21 Part 21 envelope/header inspection.
//!
//! This module inspects the exchange envelope and exact FILE_SCHEMA strings.
//! It is not a complete Part 21 parser and does not validate DATA entities,
//! B-rep topology, geometry, PMI or downstream engineering semantics.

use serde::{Deserialize, Serialize};

use crate::cad_exchange::StepSchemaEdition;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepHeaderViolation {
    pub code: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepHeaderProfile {
    pub id: String,
    /// Declared schema edition is policy metadata; FILE_SCHEMA strings are
    /// compared exactly against this profile's explicit allow-list.
    pub required_edition: StepSchemaEdition,
    pub expected_file_schema_identifiers: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepHeaderInspection {
    pub profile_id: String,
    pub profile_required_edition: StepSchemaEdition,
    pub file_digest_blake3: String,
    pub file_schema_identifiers: Vec<String>,
    pub header_section_present: bool,
    pub data_section_present: bool,
    pub end_marker_present: bool,
}

fn add(
    violations: &mut Vec<StepHeaderViolation>,
    code: &str,
    detail: impl Into<String>,
) {
    violations.push(StepHeaderViolation {
        code: code.to_owned(),
        detail: detail.into(),
    });
}

fn skip_ignored(bytes: &[u8], mut index: usize) -> Result<usize, StepHeaderViolation> {
    loop {
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        if index + 1 < bytes.len() && bytes[index] == b'/' && bytes[index + 1] == b'*' {
            index += 2;
            let mut closed = false;
            while index + 1 < bytes.len() {
                if bytes[index] == b'*' && bytes[index + 1] == b'/' {
                    index += 2;
                    closed = true;
                    break;
                }
                index += 1;
            }
            if !closed {
                return Err(StepHeaderViolation {
                    code: "unterminated_comment".into(),
                    detail: "Part 21 header contains an unterminated block comment".into(),
                });
            }
            continue;
        }
        return Ok(index);
    }
}

fn keyword_positions(bytes: &[u8], keyword: &[u8]) -> Vec<usize> {
    let mut found = Vec::new();
    let mut index = 0;
    let mut in_string = false;
    let mut in_comment = false;

    while index < bytes.len() {
        if in_comment {
            if index + 1 < bytes.len() && bytes[index] == b'*' && bytes[index + 1] == b'/' {
                in_comment = false;
                index += 2;
            } else {
                index += 1;
            }
            continue;
        }
        if in_string {
            if bytes[index] == b'\'' {
                if index + 1 < bytes.len() && bytes[index + 1] == b'\'' {
                    index += 2;
                } else {
                    in_string = false;
                    index += 1;
                }
            } else {
                index += 1;
            }
            continue;
        }
        if index + 1 < bytes.len() && bytes[index] == b'/' && bytes[index + 1] == b'*' {
            in_comment = true;
            index += 2;
            continue;
        }
        if bytes[index] == b'\'' {
            in_string = true;
            index += 1;
            continue;
        }
        if index + keyword.len() <= bytes.len()
            && bytes[index..index + keyword.len()]
                .iter()
                .zip(keyword)
                .all(|(left, right)| left.eq_ignore_ascii_case(right))
        {
            let before_ok = index == 0
                || !(bytes[index - 1].is_ascii_alphanumeric() || bytes[index - 1] == b'_');
            let after = index + keyword.len();
            let after_ok = after == bytes.len()
                || !(bytes[after].is_ascii_alphanumeric() || bytes[after] == b'_');
            if before_ok && after_ok {
                found.push(index);
                index += keyword.len();
                continue;
            }
        }
        index += 1;
    }
    found
}

fn expect_byte(
    bytes: &[u8],
    index: usize,
    expected: u8,
    code: &'static str,
) -> Result<usize, StepHeaderViolation> {
    let index = skip_ignored(bytes, index)?;
    if bytes.get(index).copied() == Some(expected) {
        Ok(index + 1)
    } else {
        Err(StepHeaderViolation {
            code: code.into(),
            detail: format!(
                "expected '{}' at byte offset {index}",
                expected as char
            ),
        })
    }
}

fn parse_step_string(
    bytes: &[u8],
    mut index: usize,
) -> Result<(String, usize), StepHeaderViolation> {
    index = skip_ignored(bytes, index)?;
    if bytes.get(index).copied() != Some(b'\'') {
        return Err(StepHeaderViolation {
            code: "file_schema_string".into(),
            detail: format!("expected a quoted FILE_SCHEMA identifier at byte offset {index}"),
        });
    }
    index += 1;
    let mut value = Vec::new();
    while index < bytes.len() {
        if bytes[index] == b'\'' {
            if index + 1 < bytes.len() && bytes[index + 1] == b'\'' {
                value.push(b'\'');
                index += 2;
                continue;
            }
            let parsed = String::from_utf8(value).map_err(|_| StepHeaderViolation {
                code: "file_schema_encoding".into(),
                detail: "FILE_SCHEMA identifier is not valid UTF-8/ASCII-compatible text".into(),
            })?;
            return Ok((parsed, index + 1));
        }
        value.push(bytes[index]);
        index += 1;
    }
    Err(StepHeaderViolation {
        code: "file_schema_string".into(),
        detail: "unterminated quoted FILE_SCHEMA identifier".into(),
    })
}

fn parse_file_schema_statement(
    bytes: &[u8],
    keyword_index: usize,
) -> Result<Vec<String>, StepHeaderViolation> {
    let mut index = keyword_index + b"FILE_SCHEMA".len();
    index = expect_byte(bytes, index, b'(', "file_schema_open")?;
    index = expect_byte(bytes, index, b'(', "file_schema_inner_open")?;

    let mut identifiers = Vec::new();
    loop {
        let (identifier, next) = parse_step_string(bytes, index)?;
        if identifier.trim().is_empty() {
            return Err(StepHeaderViolation {
                code: "file_schema_empty_identifier".into(),
                detail: "FILE_SCHEMA contains an empty schema identifier".into(),
            });
        }
        identifiers.push(identifier);
        index = skip_ignored(bytes, next)?;

        match bytes.get(index).copied() {
            Some(b',') => index += 1,
            Some(b')') => {
                index = expect_byte(bytes, index, b')', "file_schema_inner_close")?;
                index = expect_byte(bytes, index, b')', "file_schema_outer_close")?;
                index = expect_byte(bytes, index, b';', "file_schema_semicolon")?;
                break;
            }
            _ => {
                return Err(StepHeaderViolation {
                    code: "file_schema_separator".into(),
                    detail: format!(
                        "expected comma or closing parenthesis in FILE_SCHEMA at byte offset {index}"
                    ),
                })
            }
        }
    }

    if identifiers.is_empty() {
        return Err(StepHeaderViolation {
            code: "file_schema_empty".into(),
            detail: "FILE_SCHEMA must declare at least one schema identifier".into(),
        });
    }
    Ok(identifiers)
}

/// Inspect a Part 21 file's outer envelope and exact FILE_SCHEMA identifiers.
///
/// This does not parse DATA records or certify the declared schema edition.
/// The caller supplies a frozen profile containing the exact identifiers
/// expected from the selected exporter/conformance configuration.
pub fn inspect_step_part21_header(
    bytes: &[u8],
    profile: &StepHeaderProfile,
) -> Result<StepHeaderInspection, Vec<StepHeaderViolation>> {
    let mut violations = Vec::new();

    if profile.id.trim().is_empty() || profile.expected_file_schema_identifiers.is_empty() {
        add(
            &mut violations,
            "profile_identity",
            "profile ID and expected FILE_SCHEMA allow-list must be non-empty",
        );
    }
    if profile.required_edition != StepSchemaEdition::Ap2422025Edition4 {
        add(
            &mut violations,
            "unsupported_step_edition",
            "the initial Part 21 inspection profile requires AP242:2025 Edition 4",
        );
    }

    let start = b"ISO-10303-21;";
    let end = b"END-ISO-10303-21;";
    if !bytes.starts_with(start) {
        add(
            &mut violations,
            "part21_start",
            "file does not begin with ISO-10303-21;",
        );
    }
    if !bytes.ends_with(end) {
        add(
            &mut violations,
            "part21_end",
            "file does not end with END-ISO-10303-21;",
        );
    }

    let header_positions = keyword_positions(bytes, b"HEADER");
    let data_positions = keyword_positions(bytes, b"DATA");
    let file_schema_positions = keyword_positions(bytes, b"FILE_SCHEMA");
    let endsec_positions = keyword_positions(bytes, b"ENDSEC");

    if header_positions.len() != 1 {
        add(
            &mut violations,
            "header_section_count",
            format!("expected exactly one HEADER keyword, found {}", header_positions.len()),
        );
    }
    if data_positions.len() != 1 {
        add(
            &mut violations,
            "data_section_count",
            format!("expected exactly one DATA keyword, found {}", data_positions.len()),
        );
    }
    if file_schema_positions.len() != 1 {
        add(
            &mut violations,
            "file_schema_count",
            format!(
                "expected exactly one FILE_SCHEMA statement, found {}",
                file_schema_positions.len()
            ),
        );
    }
    if endsec_positions.len() != 2 {
        add(
            &mut violations,
            "endsec_count",
            format!("expected two ENDSEC keywords, found {}", endsec_positions.len()),
        );
    }

    let mut schema_identifiers = Vec::new();
    if let (Some(header), Some(data), Some(schema)) = (
        header_positions.first(),
        data_positions.first(),
        file_schema_positions.first(),
    ) {
        if !(header < schema && schema < data) {
            add(
                &mut violations,
                "header_schema_order",
                "HEADER, FILE_SCHEMA and DATA appear in an invalid order",
            );
        }
        match parse_file_schema_statement(bytes, *schema) {
            Ok(identifiers) => schema_identifiers = identifiers,
            Err(error) => violations.push(error),
        }
    }

    if !schema_identifiers.is_empty() {
        let mut observed = schema_identifiers.clone();
        let mut expected = profile.expected_file_schema_identifiers.clone();
        observed.sort();
        expected.sort();
        if observed != expected {
            add(
                &mut violations,
                "file_schema_mismatch",
                format!(
                    "observed FILE_SCHEMA identifiers {:?} do not exactly match profile {:?}",
                    schema_identifiers, profile.expected_file_schema_identifiers
                ),
            );
        }
    }

    if let (Some(header), Some(data), Some(first_end), Some(second_end)) = (
        header_positions.first(),
        data_positions.first(),
        endsec_positions.first(),
        endsec_positions.get(1),
    ) {
        if !(header < first_end && first_end < data && data < second_end) {
            add(
                &mut violations,
                "section_order",
                "Part 21 HEADER/DATA/ENDSEC order is invalid",
            );
        }
    }

    if !violations.is_empty() {
        return Err(violations);
    }

    Ok(StepHeaderInspection {
        profile_id: profile.id.clone(),
        profile_required_edition: profile.required_edition,
        file_digest_blake3: blake3::hash(bytes).to_hex().to_string(),
        file_schema_identifiers: schema_identifiers,
        header_section_present: true,
        data_section_present: true,
        end_marker_present: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const AP242_SCHEMA: &str =
        "AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF";

    fn profile() -> StepHeaderProfile {
        StepHeaderProfile {
            id: "marine-ap242e4-header-v1".into(),
            required_edition: StepSchemaEdition::Ap2422025Edition4,
            expected_file_schema_identifiers: vec![AP242_SCHEMA.into()],
        }
    }

    fn step(schema: &str) -> Vec<u8> {
        format!(
            "ISO-10303-21;\\nHEADER;\\nFILE_DESCRIPTION(('fixture'),'2;1');\\nFILE_NAME('fixture.step','2026-11-15T00:00:00',('test'),('test'),'writer','preprocessor','');\\nFILE_SCHEMA(('{schema}'));\\nENDSEC;\\nDATA;\\n#1=PRODUCT('fixture','','',());\\nENDSEC;\\nEND-ISO-10303-21;"
        )
        .into_bytes()
    }

    #[test]
    fn explicit_ap242_schema_identifier_matches_profile() {
        let inspected = inspect_step_part21_header(&step(AP242_SCHEMA), &profile()).unwrap();
        assert_eq!(inspected.file_schema_identifiers, vec![AP242_SCHEMA]);
        assert_eq!(
            inspected.profile_required_edition,
            StepSchemaEdition::Ap2422025Edition4
        );
    }

    #[test]
    fn ap214_schema_does_not_satisfy_ap242_profile() {
        let error =
            inspect_step_part21_header(&step("AUTOMOTIVE_DESIGN"), &profile()).unwrap_err();
        assert!(error.iter().any(|item| item.code == "file_schema_mismatch"));
    }

    #[test]
    fn file_schema_text_inside_comment_or_string_is_not_counted() {
        let bytes = format!(
            "ISO-10303-21;\\nHEADER;\\n/* FILE_SCHEMA(('AUTOMOTIVE_DESIGN')); */\\nFILE_DESCRIPTION(('FILE_SCHEMA'),'2;1');\\nFILE_SCHEMA(('{AP242_SCHEMA}'));\\nENDSEC;\\nDATA;\\nENDSEC;\\nEND-ISO-10303-21;"
        )
        .into_bytes();
        let inspected = inspect_step_part21_header(&bytes, &profile()).unwrap();
        assert_eq!(inspected.file_schema_identifiers, vec![AP242_SCHEMA]);
    }

    #[test]
    fn duplicate_file_schema_statements_are_rejected() {
        let bytes = format!(
            "ISO-10303-21;\\nHEADER;\\nFILE_SCHEMA(('{AP242_SCHEMA}'));\\nFILE_SCHEMA(('{AP242_SCHEMA}'));\\nENDSEC;\\nDATA;\\nENDSEC;\\nEND-ISO-10303-21;"
        )
        .into_bytes();
        let error = inspect_step_part21_header(&bytes, &profile()).unwrap_err();
        assert!(error.iter().any(|item| item.code == "file_schema_count"));
    }

    #[test]
    fn incomplete_part21_envelope_is_rejected() {
        let bytes = format!(
            "ISO-10303-21;\\nHEADER;\\nFILE_SCHEMA(('{AP242_SCHEMA}'));\\nENDSEC;\\nDATA;\\nENDSEC;"
        )
        .into_bytes();
        let error = inspect_step_part21_header(&bytes, &profile()).unwrap_err();
        assert!(error.iter().any(|item| item.code == "part21_end"));
    }

    #[test]
    fn only_released_edition_four_profile_is_accepted() {
        let mut wrong = profile();
        wrong.required_edition = StepSchemaEdition::Ap242Edition5CommitteeDraft;
        let error = inspect_step_part21_header(&step(AP242_SCHEMA), &wrong).unwrap_err();
        assert!(error.iter().any(|item| item.code == "unsupported_step_edition"));
    }
}
