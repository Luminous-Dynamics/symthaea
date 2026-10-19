use anyhow::{Context, Result, anyhow};
use csv::{ReaderBuilder, StringRecord};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

const EXPECTED_SOURCE_ID: &str = "DS-010";
const EXPECTED_SOURCE_URI: &str = "https://zenodo.org/records/16745911";

const REQUIRED_COLUMNS: &[&str] = &[
    "Date",
    "Plant",
    "Treatment",
    "Number of Fruits",
    "Numer of Harvested Fruits",
    "Total Weight of Harvest Fruits",
];
const GROUP_FIELDS: &[&str] = &["Plant", "Date", "Treatment"];

#[derive(Debug, Serialize)]
struct ColumnAudit {
    name: String,
    blank_cells: usize,
    zero_cells: usize,
    numeric_parseable_cells: usize,
}

#[derive(Debug, Serialize)]
struct GroupFieldAudit {
    field: String,
    present: bool,
    unique_nonblank_values: usize,
    missing_rows: usize,
    example_values: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CsvAuditReceipt {
    receipt_schema: String,
    stage: String,
    source_id: String,
    source_uri: String,
    input_file_name: String,
    content_sha256: String,
    expected_sha256: Option<String>,
    digest_matches_expected: Option<bool>,
    byte_length: usize,
    delimiter: char,
    header_field_count: usize,
    data_record_count: usize,
    duplicate_header_names: Vec<String>,
    blank_header_positions: Vec<usize>,
    missing_required_columns: Vec<String>,
    rows_with_width_mismatch: usize,
    rows_with_extra_fields: usize,
    total_extra_fields: usize,
    duplicate_plant_date_keys: usize,
    columns: Vec<ColumnAudit>,
    group_fields: Vec<GroupFieldAudit>,
    unit_review_warnings: Vec<String>,
    expected_data_records: Option<usize>,
    record_count_matches_expectation: Option<bool>,
    training_authorized: bool,
    notes: Vec<String>,
}

fn normalized_header(raw: &str) -> String {
    raw.trim_start_matches('\u{feff}').trim().to_owned()
}

fn header_index(headers: &[String], sought: &str) -> Option<usize> {
    headers.iter().position(|header| header == sought)
}

fn looks_like_measurement_without_unit(header: &str) -> bool {
    let lower = header.to_ascii_lowercase();
    let physical_quantity = [
        "weight",
        "width",
        "height",
        "temperature",
        "radiation",
        "humidity",
        "moisture",
        "conductivity",
    ]
    .iter()
    .any(|term| lower.contains(term));
    physical_quantity && !header.contains('[') && !header.contains('(')
}

/// Inspect bytes supplied by the caller. This audit never authorizes training and
/// never downloads data or normalizes values. It records the exact file digest.
fn audit_csv_bytes(
    bytes: &[u8],
    delimiter: u8,
    source_id: &str,
    source_uri: &str,
    input_file_name: &str,
    expected_data_records: Option<usize>,
    expected_sha256: Option<&str>,
) -> Result<CsvAuditReceipt> {
    if bytes.is_empty() {
        return Err(anyhow!("CSV input is empty"));
    }
    if source_id != EXPECTED_SOURCE_ID {
        return Err(anyhow!(
            "this v0 audit profile is specific to source {EXPECTED_SOURCE_ID}"
        ));
    }
    if source_uri != EXPECTED_SOURCE_URI {
        return Err(anyhow!(
            "source_uri must be the canonical record {EXPECTED_SOURCE_URI}"
        ));
    }
    if let Some(expected) = expected_sha256 {
        if expected.len() != 64 || !expected.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(anyhow!(
                "expected SHA-256 must be exactly 64 hexadecimal characters"
            ));
        }
    }

    let mut reader = ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(true)
        .flexible(true)
        .from_reader(bytes);

    let raw_headers: StringRecord = reader
        .headers()
        .context("CSV header could not be read")?
        .clone();
    let headers: Vec<String> = raw_headers.iter().map(normalized_header).collect();
    if headers.is_empty() || headers.iter().all(|header| header.is_empty()) {
        return Err(anyhow!("CSV has no non-empty header fields"));
    }

    let mut header_counts = BTreeMap::<String, usize>::new();
    for header in &headers {
        if !header.is_empty() {
            *header_counts.entry(header.clone()).or_default() += 1;
        }
    }
    let duplicate_header_names: Vec<String> = header_counts
        .iter()
        .filter_map(|(name, count)| (*count > 1).then_some(name.clone()))
        .collect();

    let blank_header_positions = headers
        .iter()
        .enumerate()
        .filter_map(|(index, header)| header.is_empty().then_some(index + 1))
        .collect::<Vec<_>>();

    let missing_required_columns = REQUIRED_COLUMNS
        .iter()
        .filter(|required| !headers.iter().any(|header| header == **required))
        .map(|required| (*required).to_owned())
        .collect::<Vec<_>>();

    let mut columns: Vec<ColumnAudit> = headers
        .iter()
        .map(|name| ColumnAudit {
            name: name.clone(),
            blank_cells: 0,
            zero_cells: 0,
            numeric_parseable_cells: 0,
        })
        .collect();

    let mut group_values: BTreeMap<String, BTreeSet<String>> = GROUP_FIELDS
        .iter()
        .map(|field| ((*field).to_owned(), BTreeSet::new()))
        .collect();
    let mut group_missing: BTreeMap<String, usize> = GROUP_FIELDS
        .iter()
        .map(|field| ((*field).to_owned(), 0))
        .collect();
    let mut group_examples: BTreeMap<String, Vec<String>> = GROUP_FIELDS
        .iter()
        .map(|field| ((*field).to_owned(), Vec::new()))
        .collect();

    let plant_index = header_index(&headers, "Plant");
    let date_index = header_index(&headers, "Date");
    let mut plant_date_keys = BTreeSet::<(String, String)>::new();
    let mut duplicate_plant_date_keys = 0usize;

    let mut data_record_count = 0usize;
    let mut rows_with_width_mismatch = 0usize;
    let mut rows_with_extra_fields = 0usize;
    let mut total_extra_fields = 0usize;

    for (record_index, row) in reader.records().enumerate() {
        let row =
            row.with_context(|| format!("invalid CSV record at data row {}", record_index + 1))?;
        data_record_count += 1;

        if row.len() != headers.len() {
            rows_with_width_mismatch += 1;
        }
        if row.len() > headers.len() {
            rows_with_extra_fields += 1;
            total_extra_fields += row.len() - headers.len();
        }

        for (column_index, column) in columns.iter_mut().enumerate() {
            let value = row.get(column_index).unwrap_or("").trim();
            if value.is_empty() {
                column.blank_cells += 1;
            }
            if let Ok(number) = value.parse::<f64>() {
                if number.is_finite() {
                    column.numeric_parseable_cells += 1;
                    if number == 0.0 {
                        column.zero_cells += 1;
                    }
                }
            }
        }

        for field in GROUP_FIELDS {
            let Some(index) = header_index(&headers, field) else {
                continue;
            };
            let value = row.get(index).unwrap_or("").trim();
            if value.is_empty() {
                *group_missing
                    .get_mut(*field)
                    .expect("group field initialized") += 1;
            } else {
                group_values
                    .get_mut(*field)
                    .expect("group field initialized")
                    .insert(value.to_owned());
                let examples = group_examples
                    .get_mut(*field)
                    .expect("group field initialized");
                if examples.len() < 5 && !examples.iter().any(|existing| existing == value) {
                    examples.push(value.to_owned());
                }
            }
        }

        if let (Some(plant_index), Some(date_index)) = (plant_index, date_index) {
            let plant = row.get(plant_index).unwrap_or("").trim();
            let date = row.get(date_index).unwrap_or("").trim();
            if !plant.is_empty() && !date.is_empty() {
                if !plant_date_keys.insert((plant.to_owned(), date.to_owned())) {
                    duplicate_plant_date_keys += 1;
                }
            }
        }
    }

    let group_fields = GROUP_FIELDS
        .iter()
        .map(|field| {
            let name = (*field).to_owned();
            let values = group_values.get(&name).expect("group field initialized");
            GroupFieldAudit {
                field: name.clone(),
                present: header_index(&headers, field).is_some(),
                unique_nonblank_values: values.len(),
                missing_rows: *group_missing.get(&name).expect("group field initialized"),
                example_values: group_examples
                    .get(&name)
                    .expect("group field initialized")
                    .clone(),
            }
        })
        .collect();

    let unit_review_warnings = headers
        .iter()
        .filter(|header| looks_like_measurement_without_unit(header))
        .map(|header| {
            format!(
                "measurement header {header} has no explicit unit annotation; do not infer units from observed magnitude"
            )
        })
        .collect();

    let digest = Sha256::digest(bytes);
    let content_sha256 = format!("{digest:x}");
    let digest_matches_expected =
        expected_sha256.map(|expected| expected.eq_ignore_ascii_case(&content_sha256));
    let record_count_matches_expectation =
        expected_data_records.map(|expected| expected == data_record_count);

    Ok(CsvAuditReceipt {
        receipt_schema: "cropfoundry-csv-audit-receipt-v0".to_owned(),
        stage: "local_schema_audit_only".to_owned(),
        source_id: source_id.to_owned(),
        source_uri: source_uri.to_owned(),
        input_file_name: input_file_name.to_owned(),
        content_sha256,
        expected_sha256: expected_sha256.map(str::to_ascii_lowercase),
        digest_matches_expected,
        byte_length: bytes.len(),
        delimiter: delimiter as char,
        header_field_count: headers.len(),
        data_record_count,
        duplicate_header_names,
        blank_header_positions,
        missing_required_columns,
        rows_with_width_mismatch,
        rows_with_extra_fields,
        total_extra_fields,
        duplicate_plant_date_keys,
        columns,
        group_fields,
        unit_review_warnings,
        expected_data_records,
        record_count_matches_expectation,
        training_authorized: false,
        notes: vec![
            "The digest identifies these exact local bytes; it does not establish artifact provenance, license, or access rights.".to_owned(),
            "digest_matches_expected is null when no expected digest was provided; a match is only as trustworthy as the supplied checksum source.".to_owned(),
            "This receipt does not authorize training, prove agronomic validity, or normalize raw values.".to_owned(),
            "Dates, units, treatment encodings, and missing/zero semantics require a source-backed data dictionary review.".to_owned(),
        ],
    })
}

/// Audit an explicitly supplied local CSV and print a JSON receipt. No network or
/// training action occurs; callers may optionally save the receipt to a file.
pub fn run(
    input: PathBuf,
    source_id: String,
    source_uri: String,
    delimiter: char,
    expected_data_records: Option<usize>,
    expected_sha256: Option<String>,
    out: Option<PathBuf>,
) -> Result<()> {
    if !delimiter.is_ascii() {
        return Err(anyhow!("CSV delimiter must be a single ASCII character"));
    }
    let bytes =
        fs::read(&input).with_context(|| format!("cannot read local CSV {}", input.display()))?;
    let file_name = input
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("input.csv")
        .to_owned();
    let receipt = audit_csv_bytes(
        &bytes,
        delimiter as u8,
        &source_id,
        &source_uri,
        &file_name,
        expected_data_records,
        expected_sha256.as_deref(),
    )?;
    let mut failures = Vec::new();
    if !receipt.duplicate_header_names.is_empty() {
        failures.push(format!(
            "duplicate headers: {:?}",
            receipt.duplicate_header_names
        ));
    }
    if !receipt.blank_header_positions.is_empty() {
        failures.push(format!(
            "unnamed header fields at 1-based positions {:?}; map only with source-backed documentation",
            receipt.blank_header_positions
        ));
    }
    if !receipt.missing_required_columns.is_empty() {
        failures.push(format!(
            "missing required columns: {:?}",
            receipt.missing_required_columns
        ));
    }
    if receipt.rows_with_width_mismatch > 0 {
        failures.push(format!(
            "{} rows have a field count different from the header",
            receipt.rows_with_width_mismatch
        ));
    }
    if receipt.record_count_matches_expectation == Some(false) {
        failures.push(format!(
            "data record count {} does not match the declared expectation {}",
            receipt.data_record_count,
            receipt.expected_data_records.unwrap_or_default()
        ));
    }
    if receipt.digest_matches_expected == Some(false) {
        failures.push(
            "local file SHA-256 does not match the independently supplied expected digest"
                .to_owned(),
        );
    }

    let rendered = serde_json::to_string_pretty(&receipt)?;
    println!("{rendered}");
    if let Some(path) = out {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .with_context(|| format!("cannot create output directory {}", parent.display()))?;
        }
        fs::write(&path, format!("{rendered}\n"))
            .with_context(|| format!("cannot write audit receipt {}", path.display()))?;
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(anyhow!(
            "CSV schema audit found blocking issues; receipt emitted/saved, no PASS: {}",
            failures.join("; ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_ID: &str = "DS-010";
    const SOURCE_URI: &str = "https://zenodo.org/records/16745911";

    fn audit(input: &str, expected: Option<usize>) -> CsvAuditReceipt {
        audit_csv_bytes(
            input.as_bytes(),
            b';',
            SOURCE_ID,
            SOURCE_URI,
            "synthetic-fixture.csv",
            expected,
            None,
        )
        .unwrap()
    }

    #[test]
    fn small_semicolon_csv_has_exact_digest_and_dimensions() {
        let input = "Date;Plant;Treatment;Total Weight of Harvest Fruits;Number of Fruits\n\
                     01/01/2024;P1;T1;0;0\n\
                     08/01/2024;P1;T1;12;2\n\
                     01/01/2024;P2;T2;;3\n";
        let receipt = audit(input, Some(3));
        assert_eq!(receipt.data_record_count, 3);
        assert_eq!(receipt.header_field_count, 5);
        assert_eq!(receipt.content_sha256.len(), 64);
        assert_eq!(receipt.record_count_matches_expectation, Some(true));
        assert!(!receipt.training_authorized);
    }

    #[test]
    fn repeated_plant_date_pair_is_detected() {
        let input = "Date;Plant;Treatment\n01/01/2024;P1;T1\n01/01/2024;P1;T2\n";
        let receipt = audit(input, None);
        assert_eq!(receipt.duplicate_plant_date_keys, 1);
    }

    #[test]
    fn row_width_mismatch_is_reported_not_silently_discarded() {
        let input = "A;B\n1;2\n3;4;5\n";
        let receipt = audit(input, None);
        assert_eq!(receipt.rows_with_width_mismatch, 1);
        assert_eq!(receipt.rows_with_extra_fields, 1);
        assert_eq!(receipt.total_extra_fields, 1);
    }

    #[test]
    fn duplicate_headers_are_reported() {
        let input = "Date;Plant;Plant\n01/01/2024;P1;P1\n";
        let receipt = audit(input, None);
        assert_eq!(receipt.duplicate_header_names, vec!["Plant".to_owned()]);
    }

    #[test]
    fn unnamed_header_is_reported_not_silently_renamed() {
        let input = ";Date;Plant;Treatment\n0;01/01/2024;P1;T1\n";
        let receipt = audit(input, None);
        assert_eq!(receipt.blank_header_positions, vec![1]);
    }

    #[test]
    fn source_profile_reports_missing_required_fields() {
        let input = "date;plant;yield\n01/01/2024;P1;1\n";
        let receipt = audit(input, None);
        assert!(
            receipt
                .missing_required_columns
                .contains(&"Date".to_owned())
        );
        assert!(
            receipt
                .missing_required_columns
                .contains(&"Treatment".to_owned())
        );
    }

    #[test]
    fn blanks_and_numeric_zero_counts_remain_separate() {
        let input = "Date;Plant;Treatment;Total Weight of Harvest Fruits\n\
                     01/01/2024;P1;T1;0\n\
                     08/01/2024;P1;T1;\n";
        let receipt = audit(input, None);
        let weight = receipt
            .columns
            .iter()
            .find(|column| column.name == "Total Weight of Harvest Fruits")
            .unwrap();
        assert_eq!(weight.zero_cells, 1);
        assert_eq!(weight.blank_cells, 1);
    }

    #[test]
    fn unitless_physical_quantity_headers_are_flagged_for_review() {
        let input = "Plant;Date;Treatment;Total Weight of Harvest Fruits\nP1;01/01/2024;T1;12\n";
        let receipt = audit(input, None);
        assert!(
            receipt
                .unit_review_warnings
                .iter()
                .any(|warning| warning.contains("Total Weight of Harvest Fruits"))
        );
    }

    #[test]
    fn explicit_expected_record_count_mismatch_is_not_a_pass() {
        let input = "Date;Plant;Treatment\n01/01/2024;P1;T1\n";
        let receipt = audit(input, Some(2));
        assert_eq!(receipt.record_count_matches_expectation, Some(false));
    }

    #[test]
    fn empty_csv_is_rejected() {
        let result = audit_csv_bytes(b"", b';', SOURCE_ID, SOURCE_URI, "empty.csv", None, None);
        assert!(result.is_err());
    }

    #[test]
    fn expected_sha256_verifies_exact_local_bytes() {
        let input = "Date;Plant;Treatment\n01/01/2024;P1;T1\n";
        let expected = format!("{:x}", Sha256::digest(input.as_bytes()));
        let receipt = audit_csv_bytes(
            input.as_bytes(),
            b';',
            SOURCE_ID,
            SOURCE_URI,
            "digest.csv",
            None,
            Some(&expected),
        )
        .unwrap();
        assert_eq!(receipt.digest_matches_expected, Some(true));
    }

    #[test]
    fn mismatched_expected_sha256_does_not_verify_artifact() {
        let input = "Date;Plant;Treatment\n01/01/2024;P1;T1\n";
        let expected = "a".repeat(64);
        let receipt = audit_csv_bytes(
            input.as_bytes(),
            b';',
            SOURCE_ID,
            SOURCE_URI,
            "digest-mismatch.csv",
            None,
            Some(&expected),
        )
        .unwrap();
        assert_eq!(receipt.digest_matches_expected, Some(false));
    }

    #[test]
    fn malformed_expected_sha256_is_rejected() {
        let input = "Date;Plant;Treatment\n01/01/2024;P1;T1\n";
        let result = audit_csv_bytes(
            input.as_bytes(),
            b';',
            SOURCE_ID,
            SOURCE_URI,
            "bad-digest.csv",
            None,
            Some("not-a-digest"),
        );
        assert!(result.is_err());
    }

    #[test]
    fn source_identity_cannot_be_relabelled() {
        let bytes = b"Date;Plant;Treatment\n01/01/2024;P1;T1\n";
        let result = audit_csv_bytes(
            bytes,
            b';',
            "DS-011",
            SOURCE_URI,
            "wrong-source.csv",
            None,
            None,
        );
        assert!(result.is_err());

        let wrong_uri = audit_csv_bytes(
            bytes,
            b';',
            SOURCE_ID,
            "https://example.invalid/other-source",
            "wrong-uri.csv",
            None,
            None,
        );
        assert!(wrong_uri.is_err());
    }
}
