// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Provenance-aware structural audit of the public UCI Wine Quality CSV format.
//!
//! The declared source and license are not cryptographically authenticated here.
//! This tool audits local bytes and basic field semantics; it does not train a
//! model or establish batch/vintage/site generalization.

use serde::Serialize;
use serde_json::to_vec_pretty;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, env, fs, path::PathBuf, process::ExitCode};

const SOURCE_URI: &str = "https://archive.ics.uci.edu/dataset/186/wine%2Bquality";
const SOURCE_DOI: &str = "https://doi.org/10.24432/C56S3T";
const REPORTED_LICENSE: &str =
    "CC BY 4.0 (as stated by UCI; verify current record before redistribution)";
const EXPECTED_COLUMNS: [&str; 12] = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
];

#[derive(Debug, Serialize)]
struct ColumnSummary {
    name: String,
    valid_value_count: u64,
    min: f64,
    max: f64,
    mean: f64,
}

#[derive(Debug)]
struct ColumnAccumulator {
    name: &'static str,
    count: u64,
    min: f64,
    max: f64,
    sum: f64,
}

impl ColumnAccumulator {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            count: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            sum: 0.0,
        }
    }

    fn add(&mut self, value: f64) {
        self.count += 1;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.sum += value;
    }

    fn summary(&self) -> Option<ColumnSummary> {
        (self.count > 0).then(|| ColumnSummary {
            name: self.name.to_owned(),
            valid_value_count: self.count,
            min: self.min,
            max: self.max,
            mean: self.sum / self.count as f64,
        })
    }
}

#[derive(Debug, Serialize)]
struct AuditReceipt {
    receipt_schema: &'static str,
    status: &'static str,
    qualification_scope: &'static str,
    source_uri: &'static str,
    source_doi: &'static str,
    source_identity_status: &'static str,
    reported_license: &'static str,
    input_file: String,
    declared_color: String,
    input_bytes: usize,
    input_sha256: String,
    header_valid: bool,
    rows_total: u64,
    rows_valid: u64,
    rows_invalid: u64,
    quality_score_counts: BTreeMap<i64, u64>,
    columns: Vec<ColumnSummary>,
    errors: Vec<String>,
    claim_ceiling: &'static str,
}

fn hex_sha256(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

fn invalid_color(color: &str) -> bool {
    color != "red" && color != "white"
}

fn row_values(record: &csv::StringRecord) -> Result<Vec<f64>, Vec<String>> {
    if record.len() != EXPECTED_COLUMNS.len() {
        return Err(vec!["column_count_mismatch".to_owned()]);
    }

    let mut values = Vec::with_capacity(EXPECTED_COLUMNS.len());
    let mut errors = Vec::new();
    for (index, column) in EXPECTED_COLUMNS.iter().enumerate() {
        let text = record.get(index).unwrap_or_default().trim();
        match text.parse::<f64>() {
            Ok(value) if value.is_finite() => values.push(value),
            Ok(_) => {
                errors.push(format!("non_finite_value:{column}"));
                values.push(f64::NAN);
            }
            Err(_) => {
                errors.push(format!("missing_or_non_numeric_value:{column}"));
                values.push(f64::NAN);
            }
        }
    }

    if errors.is_empty() {
        for (index, column) in EXPECTED_COLUMNS.iter().enumerate().take(11) {
            if values[index] < 0.0 {
                errors.push(format!("negative_measurement:{column}"));
            }
        }
        if values[7] <= 0.0 {
            errors.push("density_must_be_positive".to_owned());
        }
        if !(0.0..=14.0).contains(&values[8]) {
            errors.push("ph_outside_declared_physical_range".to_owned());
        }
        if !(0.0..=100.0).contains(&values[10]) {
            errors.push("alcohol_outside_percent_range".to_owned());
        }
        if values[11].fract() != 0.0 || !(0.0..=10.0).contains(&values[11]) {
            errors.push("quality_must_be_integer_0_to_10".to_owned());
        }
    }

    if errors.is_empty() {
        Ok(values)
    } else {
        Err(errors)
    }
}

fn audit_csv(bytes: &[u8], color: &str, filename: &str) -> AuditReceipt {
    let mut errors = Vec::new();
    if invalid_color(color) {
        errors.push("declared_color_must_be_red_or_white".to_owned());
    }

    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b';')
        .has_headers(true)
        .trim(csv::Trim::All)
        .from_reader(bytes);
    let header = match reader.headers() {
        Ok(record) => record.iter().map(str::to_owned).collect::<Vec<_>>(),
        Err(error) => {
            errors.push(format!("csv_header_parse_error:{error}"));
            Vec::new()
        }
    };
    let header_valid = header.len() == EXPECTED_COLUMNS.len()
        && header
            .iter()
            .zip(EXPECTED_COLUMNS.iter())
            .all(|(actual, expected)| actual == expected);
    if !header_valid
        && !errors
            .iter()
            .any(|e| e.starts_with("csv_header_parse_error:"))
    {
        errors.push("header_mismatch_or_wrong_column_order".to_owned());
    }

    let mut accumulators = EXPECTED_COLUMNS.map(ColumnAccumulator::new);
    let mut quality_score_counts = BTreeMap::new();
    let mut rows_total = 0_u64;
    let mut rows_valid = 0_u64;
    let mut rows_invalid = 0_u64;

    if header_valid {
        for record in reader.records() {
            rows_total += 1;
            match record {
                Ok(record) => match row_values(&record) {
                    Ok(values) => {
                        rows_valid += 1;
                        for (accumulator, value) in accumulators.iter_mut().zip(values.iter()) {
                            accumulator.add(*value);
                        }
                        *quality_score_counts
                            .entry(values[11] as i64)
                            .or_insert(0_u64) += 1;
                    }
                    Err(row_errors) => {
                        rows_invalid += 1;
                        for error in row_errors {
                            errors.push(format!("row_{}_{}", rows_total, error));
                        }
                    }
                },
                Err(_) => {
                    rows_invalid += 1;
                    errors.push(format!("row_{}_csv_record_parse_error", rows_total));
                }
            }
        }
    }

    if header_valid && rows_total == 0 {
        errors.push("no_data_rows".to_owned());
    }
    let columns = accumulators
        .iter()
        .filter_map(ColumnAccumulator::summary)
        .collect::<Vec<_>>();
    let passed = header_valid
        && !invalid_color(color)
        && rows_total > 0
        && rows_invalid == 0
        && errors.is_empty();

    AuditReceipt {
        receipt_schema: "uci-wine-quality-audit-v1",
        status: if passed { "PASS_AUDIT" } else { "FAIL" },
        qualification_scope: "local_csv_integrity_and_basic_field_semantics_only",
        source_uri: SOURCE_URI,
        source_doi: SOURCE_DOI,
        source_identity_status: "CALLER_DECLARED_NOT_AUTHENTICATED",
        reported_license: REPORTED_LICENSE,
        input_file: filename.to_owned(),
        declared_color: color.to_owned(),
        input_bytes: bytes.len(),
        input_sha256: hex_sha256(bytes),
        header_valid,
        rows_total,
        rows_valid,
        rows_invalid,
        quality_score_counts,
        columns,
        errors,
        claim_ceiling: "A valid receipt establishes only structural consistency of these local bytes. It does not authenticate the download, independently verify license terms, prove source record identity, establish training/test independence, demonstrate predictive accuracy, prove vintage/site transfer, or validate food safety.",
    }
}

fn parse_args() -> Result<(PathBuf, String, PathBuf), String> {
    let mut input = None;
    let mut color = None;
    let mut output = None;
    let mut args = env::args().skip(1);

    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing_value_for_{flag}"))?;
        match flag.as_str() {
            "--input" => input = Some(PathBuf::from(value)),
            "--color" => color = Some(value),
            "--out" => output = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown_argument_{flag}")),
        }
    }

    Ok((
        input.ok_or_else(|| "missing_--input".to_owned())?,
        color.ok_or_else(|| "missing_--color".to_owned())?,
        output.ok_or_else(|| "missing_--out".to_owned())?,
    ))
}

fn run() -> Result<ExitCode, String> {
    let (input_path, color, output_path) = parse_args()?;
    let bytes = fs::read(&input_path).map_err(|error| format!("input_read_failed: {error}"))?;
    let filename = input_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("<non-utf8>");
    let receipt = audit_csv(&bytes, &color, filename);
    let passed = receipt.status == "PASS_AUDIT";
    let serialized = to_vec_pretty(&receipt)
        .map_err(|error| format!("receipt_serialization_failed: {error}"))?;
    fs::write(&output_path, serialized)
        .map_err(|error| format!("receipt_write_failed: {error}"))?;
    if passed {
        Ok(ExitCode::SUCCESS)
    } else {
        Ok(ExitCode::FAILURE)
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("wine-quality-source-audit: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = "fixed acidity;volatile acidity;citric acid;residual sugar;chlorides;free sulfur dioxide;total sulfur dioxide;density;pH;sulphates;alcohol;quality";
    const ROW_1: &str = "7.4;0.7;0.0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;5";
    const ROW_2: &str = "7.8;0.88;0.0;2.6;0.098;25;67;0.9968;3.2;0.68;9.8;6";

    #[test]
    fn valid_semicolon_dataset_produces_audit_receipt() {
        let bytes = format!("{HEADER}\n{ROW_1}\n{ROW_2}\n");
        let receipt = audit_csv(bytes.as_bytes(), "red", "fixture-red.csv");
        assert_eq!(receipt.status, "PASS_AUDIT");
        assert!(receipt.header_valid);
        assert_eq!(receipt.rows_total, 2);
        assert_eq!(receipt.rows_valid, 2);
        assert_eq!(receipt.rows_invalid, 0);
        assert_eq!(receipt.quality_score_counts.get(&5), Some(&1));
        assert_eq!(receipt.quality_score_counts.get(&6), Some(&1));
        assert_eq!(receipt.input_sha256.len(), 64);
        assert_eq!(receipt.columns.len(), 12);
    }

    #[test]
    fn non_finite_rows_fail_without_silent_imputation() {
        let bytes = format!("{HEADER}\n7.4;0.7;0.0;1.9;0.076;11;34;0.9978;NaN;0.56;9.4;5\n");
        let receipt = audit_csv(bytes.as_bytes(), "red", "bad.csv");
        assert_eq!(receipt.status, "FAIL");
        assert_eq!(receipt.rows_total, 1);
        assert_eq!(receipt.rows_invalid, 1);
        assert!(
            receipt
                .errors
                .iter()
                .any(|error| error.contains("non_finite_value:pH"))
        );
    }

    #[test]
    fn wrong_header_or_delimiter_fails_closed() {
        let bytes = format!("{HEADER}\n{ROW_1}\n").replace(';', ",");
        let receipt = audit_csv(bytes.as_bytes(), "red", "wrong-delimiter.csv");
        assert_eq!(receipt.status, "FAIL");
        assert!(!receipt.header_valid);
        assert_eq!(receipt.rows_total, 0);
    }

    #[test]
    fn invalid_quality_values_fail_closed() {
        let bytes = format!("{HEADER}\n7.4;0.7;0.0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;5.5\n");
        let receipt = audit_csv(bytes.as_bytes(), "red", "bad-quality.csv");
        assert_eq!(receipt.status, "FAIL");
        assert!(
            receipt
                .errors
                .iter()
                .any(|error| error.contains("quality_must_be_integer_0_to_10"))
        );
    }

    #[test]
    fn input_color_is_explicit_and_checked() {
        let bytes = format!("{HEADER}\n{ROW_1}\n");
        let receipt = audit_csv(bytes.as_bytes(), "rose", "unknown-color.csv");
        assert_eq!(receipt.status, "FAIL");
        assert!(
            receipt
                .errors
                .iter()
                .any(|error| error == "declared_color_must_be_red_or_white")
        );
    }

    #[test]
    fn digest_binds_exact_raw_input_bytes() {
        let first = format!("{HEADER}\n{ROW_1}\n");
        let second = format!("{HEADER}\n{ROW_1}\n\n");
        let a = audit_csv(first.as_bytes(), "red", "fixture.csv");
        let b = audit_csv(second.as_bytes(), "red", "fixture.csv");
        assert_ne!(a.input_sha256, b.input_sha256);
    }
}
