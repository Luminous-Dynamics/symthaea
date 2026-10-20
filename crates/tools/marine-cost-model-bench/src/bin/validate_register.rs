//! Fail-closed validator for the marine procurement source-register CSV.
//!
//! This checks record shape and declared admission rules; it does not fetch, authenticate,
//! hash, or scientifically validate source documents. A valid row is not proof its claims are true.

use std::collections::HashSet;
use std::env;
use std::fs;
use std::process::ExitCode;

const HEADERS: [&str; 23] = [
    "anchor_id",
    "country",
    "buyer_or_source",
    "source_record",
    "source_kind",
    "award_or_issue_date",
    "vessel_length_m",
    "vessel_material",
    "vessel_or_propulsion_description",
    "quantity",
    "total_contract_value",
    "currency",
    "price_scope",
    "explicitly_included",
    "unknown_or_unverified",
    "comparability_disposition",
    "source_url",
    "notes",
    "normalized_unit_price",
    "normalized_currency_basis_id",
    "normalization_receipt_sha256",
    "normalization_method_ref",
    "normalized_scope_id",
];

const CONTEXT_ONLY: &str = "ContextOnlyNotComparable";
const SPEC_ONLY: &str = "SpecificationOnlyNoPrice";
const DISCOVERY_ONLY: &str = "DiscoverySourceNotPriceEvidence";
const COMPARABLE: &str = "ComparablePriceEvidence";

#[derive(Debug, PartialEq, Eq)]
struct AuditReport {
    records: usize,
    comparable_price_records: usize,
    non_comparable_records: usize,
}

fn parse_csv(input: &str) -> Result<Vec<Vec<String>>, String> {
    let chars: Vec<char> = input.chars().collect();
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut just_closed_quote = false;
    let mut index = 0;

    while index < chars.len() {
        let mut current = chars[index];
        if current == '\r' {
            if chars.get(index + 1) != Some(&'\n') {
                return Err(format!("unexpected carriage return at character {}", index + 1));
            }
            current = '\n';
            index += 1;
        }

        if in_quotes {
            if current == '"' {
                if chars.get(index + 1) == Some(&'"') {
                    field.push('"');
                    index += 2;
                    continue;
                }
                in_quotes = false;
                just_closed_quote = true;
            } else {
                field.push(current);
            }
            index += 1;
            continue;
        }

        if just_closed_quote {
            match current {
                ',' => {
                    row.push(std::mem::take(&mut field));
                    just_closed_quote = false;
                }
                '\n' => {
                    row.push(std::mem::take(&mut field));
                    if !row.iter().all(|value| value.is_empty()) {
                        rows.push(std::mem::take(&mut row));
                    } else {
                        row.clear();
                    }
                    just_closed_quote = false;
                }
                _ => return Err(format!(
                    "unexpected character after closing quote at character {}",
                    index + 1
                )),
            }
            index += 1;
            continue;
        }

        match current {
            '"' if field.is_empty() => in_quotes = true,
            '"' => return Err(format!(
                "quote inside an unquoted field at character {}",
                index + 1
            )),
            ',' => row.push(std::mem::take(&mut field)),
            '\n' => {
                row.push(std::mem::take(&mut field));
                if !row.iter().all(|value| value.is_empty()) {
                    rows.push(std::mem::take(&mut row));
                } else {
                    row.clear();
                }
            }
            _ => field.push(current),
        }
        index += 1;
    }

    if in_quotes {
        return Err("unterminated quoted CSV field".to_owned());
    }
    if !field.is_empty() || !row.is_empty() || just_closed_quote {
        row.push(field);
        if !row.iter().all(|value| value.is_empty()) {
            rows.push(row);
        }
    }

    Ok(rows)
}

fn cell<'a>(row: &'a [String], index: usize) -> &'a str {
    row.get(index).map(String::as_str).unwrap_or("").trim()
}

fn is_http_url(value: &str) -> bool {
    (value.starts_with("https://") || value.starts_with("http://"))
        && !value.chars().any(char::is_whitespace)
        && value.len() > if value.starts_with("https://") { 8 } else { 7 }
}

fn validate_csv(input: &str) -> Result<AuditReport, Vec<String>> {
    let rows = parse_csv(input).map_err(|error| vec![format!("CSV parse error: {error}")])?;
    if rows.is_empty() {
        return Err(vec!["CSV is empty; expected a header row".to_owned()]);
    }
    let header = &rows[0];
    let actual: Vec<&str> = header.iter().map(String::as_str).collect();
    if actual.as_slice() != HEADERS.as_slice() {
        return Err(vec![format!(
            "header mismatch; expected exact v0 schema with {} columns, found {}",
            HEADERS.len(),
            actual.len()
        )]);
    }

    let mut errors = Vec::new();
    let mut seen_ids = HashSet::new();
    let mut valid_records = 0;
    let mut comparable_price_records = 0;
    let mut non_comparable_records = 0;

    for (row_index, row) in rows.iter().enumerate().skip(1) {
        let record_number = row_index + 1;
        let mut row_errors = Vec::new();
        if row.len() != HEADERS.len() {
            row_errors.push(format!(
                "row {record_number}: expected {} columns, found {}",
                HEADERS.len(),
                row.len()
            ));
            errors.extend(row_errors);
            continue;
        }

        let anchor_id = cell(row, 0);
        let disposition = cell(row, 15);
        let price_text = cell(row, 10);
        let currency = cell(row, 11);
        let has_price = !price_text.is_empty();

        for (index, field_name) in [
            (0, "anchor_id"),
            (1, "country"),
            (2, "buyer_or_source"),
            (3, "source_record"),
            (4, "source_kind"),
            (15, "comparability_disposition"),
            (16, "source_url"),
            (17, "notes"),
        ] {
            if cell(row, index).is_empty() {
                row_errors.push(format!(
                    "row {record_number} ({anchor_id}): required field {field_name} is blank"
                ));
            }
        }

        if !anchor_id.is_empty() && !seen_ids.insert(anchor_id.to_owned()) {
            row_errors.push(format!(
                "row {record_number} ({anchor_id}): duplicate anchor_id"
            ));
        }
        if !is_http_url(cell(row, 16)) {
            row_errors.push(format!(
                "row {record_number} ({anchor_id}): source_url must be an absolute HTTP(S) URL without whitespace"
            ));
        }

        if has_price {
            match price_text.parse::<f64>() {
                Ok(value) if value.is_finite() && value > 0.0 => {
                    if currency.is_empty() {
                        row_errors.push(format!(
                            "row {record_number} ({anchor_id}): priced record has no currency"
                        ));
                    }
                }
                _ => row_errors.push(format!(
                    "row {record_number} ({anchor_id}): total_contract_value must be a finite positive number or blank"
                )),
            }
        }

        match disposition {
            CONTEXT_ONLY => {
                non_comparable_records += 1;
            }
            SPEC_ONLY | DISCOVERY_ONLY => {
                non_comparable_records += 1;
                if has_price {
                    row_errors.push(format!(
                        "row {record_number} ({anchor_id}): {disposition} cannot declare total_contract_value"
                    ));
                }
            }
            COMPARABLE => {
                let price_valid = match price_text.parse::<f64>() {
                    Ok(value) => value.is_finite() && value > 0.0 && !currency.is_empty(),
                    Err(_) => false,
                };
                for (index, field_name) in [
                    (5, "award_or_issue_date"),
                    (8, "vessel_or_propulsion_description"),
                    (9, "quantity"),
                    (12, "price_scope"),
                    (13, "explicitly_included"),
                ] {
                    if cell(row, index).is_empty() {
                        row_errors.push(format!(
                            "row {record_number} ({anchor_id}): ComparablePriceEvidence requires {field_name}"
                        ));
                    }
                }
                match cell(row, 9).parse::<f64>() {
                    Ok(value) if value.is_finite() && value > 0.0 => {}
                    _ => row_errors.push(format!(
                        "row {record_number} ({anchor_id}): ComparablePriceEvidence requires a known positive numeric quantity"
                    )),
                }
                let unresolved = cell(row, 14);
                if !unresolved.is_empty()
                    && !unresolved.eq_ignore_ascii_case("none")
                    && !unresolved.eq_ignore_ascii_case("none identified")
                {
                    row_errors.push(format!(
                        "row {record_number} ({anchor_id}): unresolved fields remain; record must stay non-comparable"
                    ));
                }
                if !price_valid {
                    row_errors.push(format!(
                        "row {record_number} ({anchor_id}): ComparablePriceEvidence requires a valid positive total_contract_value and currency"
                    ));
                }

                match cell(row, 18).parse::<f64>() {
                    Ok(value) if value.is_finite() && value > 0.0 => {}
                    _ => row_errors.push(format!(
                        "row {record_number} ({anchor_id}): ComparablePriceEvidence requires positive normalized_unit_price"
                    )),
                }
                if cell(row, 19).is_empty() {
                    row_errors.push(format!(
                        "row {record_number} ({anchor_id}): ComparablePriceEvidence requires normalized_currency_basis_id (currency, price year and geography)"
                    ));
                }
                let receipt = cell(row, 20);
                if receipt.len() != 64 || !receipt.chars().all(|character| character.is_ascii_hexdigit()) {
                    row_errors.push(format!(
                        "row {record_number} ({anchor_id}): normalization_receipt_sha256 must contain 64 hexadecimal characters"
                    ));
                }
                if cell(row, 21).is_empty() {
                    row_errors.push(format!(
                        "row {record_number} ({anchor_id}): ComparablePriceEvidence requires normalization_method_ref"
                    ));
                }
                if cell(row, 22).is_empty() {
                    row_errors.push(format!(
                        "row {record_number} ({anchor_id}): ComparablePriceEvidence requires normalized_scope_id"
                    ));
                }
                if row_errors.is_empty() {
                    comparable_price_records += 1;
                }
            }
            _ => row_errors.push(format!(
                "row {record_number} ({anchor_id}): unsupported comparability_disposition {disposition:?}"
            )),
        }

        if row_errors.is_empty() {
            valid_records += 1;
            if disposition == CONTEXT_ONLY
                || disposition == SPEC_ONLY
                || disposition == DISCOVERY_ONLY
            {
                // These rows can be valid register entries but never admitted price points.
            }
        } else {
            errors.extend(row_errors);
        }
    }

    if !errors.is_empty() {
        errors.sort();
        return Err(errors);
    }

    Ok(AuditReport {
        records: valid_records,
        comparable_price_records,
        non_comparable_records,
    })
}

fn main() -> ExitCode {
    let path = match env::args().nth(1) {
        Some(path) => path,
        None => {
            eprintln!(
                "usage: validate_register <path-to-public-procurement-anchors.csv>"
            );
            return ExitCode::from(2);
        }
    };
    let input = match fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(error) => {
            eprintln!("cannot read {path}: {error}");
            return ExitCode::from(2);
        }
    };

    match validate_csv(&input) {
        Ok(report) => {
            println!("schema,records,comparable_price_records,non_comparable_records");
            println!(
                "marine-procurement-register-v0,{},{},{}",
                report.records, report.comparable_price_records, report.non_comparable_records
            );
            println!("Notice: validation checks structure and declared admission rules only; it does not authenticate source documents or normalize prices.");
            ExitCode::SUCCESS
        }
        Err(errors) => {
            for error in errors {
                eprintln!("{error}");
            }
            eprintln!("register validation failed; no records were admitted as a successful audit");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_row() -> Vec<String> {
        vec![String::new(); HEADERS.len()]
    }

    fn register_row(disposition: &str) -> Vec<String> {
        let mut row = empty_row();
        row[0] = "X-1".to_owned();
        row[1] = "South Africa".to_owned();
        row[2] = "Buyer".to_owned();
        row[3] = "AWARD-1".to_owned();
        row[4] = "Contract award".to_owned();
        row[5] = "2026-01-01".to_owned();
        row[8] = "6m utility workboat".to_owned();
        row[15] = disposition.to_owned();
        row[16] = "https://example.org/award".to_owned();
        row[17] = "Test fixture".to_owned();
        row
    }

    fn csv_for_rows(rows: &[Vec<String>]) -> String {
        let mut output = HEADERS.join(",");
        output.push('\n');
        for row in rows {
            output.push_str(&row.join(","));
            output.push('\n');
        }
        output
    }

    fn fully_normalized_row() -> Vec<String> {
        let mut row = register_row(COMPARABLE);
        row[9] = "1".to_owned();
        row[10] = "100000".to_owned();
        row[11] = "ZAR".to_owned();
        row[12] = "Delivered and commissioned vessel".to_owned();
        row[13] = "Hull; propulsion; safety outfit".to_owned();
        row[14] = "None".to_owned();
        row[18] = "100000".to_owned();
        row[19] = "ZAR-2026-SA-yard-gate-v1".to_owned();
        row[20] = "a".repeat(64);
        row[21] = "normalization-method-v1".to_owned();
        row[22] = "delivered-commissioned-vessel-v1".to_owned();
        row
    }

    #[test]
    fn csv_parser_handles_quoted_commas_and_escaped_quotes() {
        let rows = parse_csv(
            r#"a,b
"hello, world","say ""yes"""
"#
        ).unwrap();
        assert_eq!(rows[1], vec!["hello, world", "say \"yes\""]);
    }

    #[test]
    fn csv_parser_rejects_unterminated_quotes() {
        let input = r#"a,b
"unfinished,b
"#;
        assert!(parse_csv(input).is_err());
    }

    #[test]
    fn current_register_is_valid_but_admits_no_market_price_points() {
        let input = include_str!("../../references/public-procurement-anchors.csv");
        let report = validate_csv(input).unwrap();
        assert_eq!(report.records, 6);
        assert_eq!(report.comparable_price_records, 0);
        assert_eq!(report.non_comparable_records, 6);
    }

    #[test]
    fn complete_normalized_record_is_admitted_as_one_price_point() {
        let input = csv_for_rows(&[fully_normalized_row()]);
        let report = validate_csv(&input).unwrap();
        assert_eq!(report.records, 1);
        assert_eq!(report.comparable_price_records, 1);
        assert_eq!(report.non_comparable_records, 0);
    }

    #[test]
    fn comparable_price_without_quantity_is_rejected() {
        let mut row = fully_normalized_row();
        row[9] = String::new();
        let errors = validate_csv(&csv_for_rows(&[row])).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("known positive numeric quantity")));
    }

    #[test]
    fn blank_price_cannot_be_admitted_as_comparable() {
        let mut row = fully_normalized_row();
        row[10] = String::new();
        let errors = validate_csv(&csv_for_rows(&[row])).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("valid positive total_contract_value")));
    }

    #[test]
    fn missing_normalization_receipt_or_basis_is_rejected() {
        let mut row = fully_normalized_row();
        row[19] = String::new();
        row[20] = "not-a-digest".to_owned();
        let errors = validate_csv(&csv_for_rows(&[row])).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("normalized_currency_basis_id")));
        assert!(errors.iter().any(|error| error.contains("normalization_receipt_sha256")));
    }

    #[test]
    fn discovery_record_cannot_claim_a_price() {
        let mut row = register_row(DISCOVERY_ONLY);
        row[10] = "1000".to_owned();
        row[11] = "ZAR".to_owned();
        let errors = validate_csv(&csv_for_rows(&[row])).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("cannot declare total_contract_value")));
    }

    #[test]
    fn duplicate_anchor_ids_are_rejected() {
        let first = register_row(CONTEXT_ONLY);
        let mut second = register_row(CONTEXT_ONLY);
        second[3] = "DATASET-2".to_owned();
        second[16] = "https://example.org/data-2".to_owned();
        let errors = validate_csv(&csv_for_rows(&[first, second])).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("duplicate anchor_id")));
    }

    #[test]
    fn spec_only_row_cannot_include_an_award_value() {
        let mut row = register_row(SPEC_ONLY);
        row[10] = "1000".to_owned();
        row[11] = "ZAR".to_owned();
        let errors = validate_csv(&csv_for_rows(&[row])).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("cannot declare total_contract_value")));
    }
}
