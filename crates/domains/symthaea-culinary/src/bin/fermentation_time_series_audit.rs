// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Intake auditor for the published Dryad wine-fermentation long-form TSV files.
//!
//! It audits local bytes, not source authenticity or scientific validity. NA and
//! ND remain distinct missingness categories; they are never converted to zero.
//! Technical replicate IDs are preserved and must not be treated as independent
//! biological fermentations.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::PathBuf,
    process::ExitCode,
};

const SOURCE_URI: &str = "https://doi.org/10.5061/dryad.cfxpnvxp3";
const SOURCE_LICENSE: &str =
    "CC0 as declared by the Dryad record; verify the exact source version on intake";
const FIELD1: &[&str] = &[
    "time_days",
    "treatment_temp",
    "units_treatment_temp",
    "treatment_inoculation",
    "analyte",
    "units_analyte",
    "rep",
    "value",
];
const FIELD2: &[&str] = &[
    "time_days",
    "treatment_temp",
    "units_treatment_temp",
    "analyte",
    "units_analyte",
    "rep",
    "value",
];
const FIELD3: &[&str] = &[
    "treatment",
    "time_hours",
    "analyte",
    "units_analyte",
    "rep",
    "value",
];
const FIELDS1: &[&str] = &[
    "time_days",
    "treatment_temp",
    "treatment_inoculation",
    "analyte",
    "rep",
];
const FIELDS2: &[&str] = &["time_days", "treatment_temp", "analyte", "rep"];
const FIELDS3: &[&str] = &["treatment", "time_hours", "analyte", "rep"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DatasetKind {
    Figure1,
    Figure2,
    FigureS1,
}

impl DatasetKind {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "figure1" => Ok(Self::Figure1),
            "figure2" => Ok(Self::Figure2),
            "figure-s1" => Ok(Self::FigureS1),
            _ => Err("dataset_must_be_figure1_figure2_or_figure-s1".to_owned()),
        }
    }

    fn filename(self) -> &'static str {
        match self {
            Self::Figure1 => "Onetto2026_Figure1_raw_data_final.txt",
            Self::Figure2 => "Onetto2026_Figure2_raw_data_final.txt",
            Self::FigureS1 => "Onetto2026_FigureS1_raw_data_final.txt",
        }
    }

    fn expected_fields(self) -> &'static [&'static str] {
        match self {
            Self::Figure1 => FIELD1,
            Self::Figure2 => FIELD2,
            Self::FigureS1 => FIELD3,
        }
    }

    fn key_fields(self) -> &'static [&'static str] {
        match self {
            Self::Figure1 => FIELDS1,
            Self::Figure2 => FIELDS2,
            Self::FigureS1 => FIELDS3,
        }
    }

    fn time_field(self) -> &'static str {
        match self {
            Self::Figure1 | Self::Figure2 => "time_days",
            Self::FigureS1 => "time_hours",
        }
    }

    fn condition_fields(self) -> &'static [&'static str] {
        match self {
            Self::Figure1 => &[
                "treatment_temp",
                "units_treatment_temp",
                "treatment_inoculation",
            ],
            Self::Figure2 => &["treatment_temp", "units_treatment_temp"],
            Self::FigureS1 => &["treatment"],
        }
    }
}

#[derive(Debug, Serialize)]
struct NumericSummary {
    count: u64,
    min: f64,
    max: f64,
    mean: f64,
}

#[derive(Debug)]
struct NumericAccumulator {
    count: u64,
    min: f64,
    max: f64,
    sum: f64,
}

impl NumericAccumulator {
    fn new() -> Self {
        Self {
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

    fn summary(&self) -> Option<NumericSummary> {
        (self.count > 0).then(|| NumericSummary {
            count: self.count,
            min: self.min,
            max: self.max,
            mean: self.sum / self.count as f64,
        })
    }
}

impl Default for NumericAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Serialize)]
struct AnalyteSummary {
    analyte: String,
    units: String,
    observed_numeric_values: u64,
    missing_na: u64,
    missing_nd: u64,
    numeric_summary: Option<NumericSummary>,
}

#[derive(Debug, Serialize)]
struct IntakeReceipt {
    receipt_schema: &'static str,
    status: &'static str,
    qualification_scope: &'static str,
    source_uri: &'static str,
    source_license_declaration: &'static str,
    source_identity_status: &'static str,
    dataset_kind: String,
    expected_source_filename: &'static str,
    input_file: String,
    input_bytes: usize,
    input_sha256: String,
    header_valid: bool,
    header_order_matches_reference: bool,
    rows_total: u64,
    rows_valid: u64,
    rows_invalid: u64,
    missing_na_total: u64,
    missing_nd_total: u64,
    duplicate_key_rows: u64,
    distinct_technical_replicates: u64,
    distinct_treatments: u64,
    analyte_summaries: Vec<AnalyteSummary>,
    errors: Vec<String>,
    warnings: Vec<String>,
    claim_ceiling: &'static str,
}

#[derive(Default)]
struct MutableAnalyteSummary {
    values: NumericAccumulator,
    missing_na: u64,
    missing_nd: u64,
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

fn column_indexes(
    header: &csv::StringRecord,
    expected: &[&str],
) -> Option<BTreeMap<String, usize>> {
    if header.len() != expected.len() {
        return None;
    }
    let mut indexes = BTreeMap::new();
    for (index, name) in header.iter().enumerate() {
        if indexes.insert(name.to_owned(), index).is_some() {
            return None;
        }
    }
    if expected.iter().all(|name| indexes.contains_key(*name)) {
        Some(indexes)
    } else {
        None
    }
}

fn required_cell<'a>(
    record: &'a csv::StringRecord,
    indexes: &BTreeMap<String, usize>,
    field: &str,
) -> Result<&'a str, String> {
    let index = indexes
        .get(field)
        .ok_or_else(|| format!("required_column_missing:{field}"))?;
    let value = record
        .get(*index)
        .ok_or_else(|| format!("required_cell_missing:{field}"))?
        .trim();
    if value.is_empty() {
        Err(format!("required_cell_blank:{field}"))
    } else {
        Ok(value)
    }
}

fn required_nonnegative_number(
    record: &csv::StringRecord,
    indexes: &BTreeMap<String, usize>,
    field: &str,
) -> Result<f64, String> {
    let text = required_cell(record, indexes, field)?;
    let value = text
        .parse::<f64>()
        .map_err(|_| format!("non_numeric_{field}"))?;
    if !value.is_finite() {
        return Err(format!("non_finite_{field}"));
    }
    if value < 0.0 {
        return Err(format!("negative_{field}"));
    }
    Ok(value)
}

fn audit_tsv(bytes: &[u8], kind: DatasetKind, filename: &str) -> IntakeReceipt {
    let mut errors = Vec::new();
    let mut warnings = vec![
        "source_identity_is_caller_declared_not_authenticated".to_owned(),
        "rep_is_a_technical_replicate_in_the_source_description_not_proof_of_independent_batches"
            .to_owned(),
        "no_prediction_or_scientific_validity_claim_is_evaluated".to_owned(),
    ];

    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .flexible(false)
        .trim(csv::Trim::All)
        .from_reader(bytes);

    let header = match reader.headers() {
        Ok(value) => value.clone(),
        Err(error) => {
            errors.push(format!("header_parse_error:{error}"));
            csv::StringRecord::new()
        }
    };
    let indexes = column_indexes(&header, kind.expected_fields());
    let header_valid = indexes.is_some();
    let header_order_matches_reference = header.len() == kind.expected_fields().len()
        && header
            .iter()
            .zip(kind.expected_fields().iter())
            .all(|(actual, expected)| actual == *expected);

    if !header_valid
        && !errors
            .iter()
            .any(|error| error.starts_with("header_parse_error:"))
    {
        errors.push("header_columns_missing_extra_or_duplicated".to_owned());
    }
    if header_valid && !header_order_matches_reference {
        warnings.push(
            "header_order_differs_from_source_reference_but_named_columns_were_resolved".to_owned(),
        );
    }

    let mut rows_total = 0_u64;
    let mut rows_valid = 0_u64;
    let mut rows_invalid = 0_u64;
    let mut missing_na_total = 0_u64;
    let mut missing_nd_total = 0_u64;
    let mut duplicate_key_rows = 0_u64;
    let mut row_keys = BTreeSet::new();
    let mut replicates = BTreeSet::new();
    let mut treatments = BTreeSet::new();
    let mut summaries: BTreeMap<(String, String), MutableAnalyteSummary> = BTreeMap::new();

    if let Some(indexes) = indexes {
        for result in reader.records() {
            rows_total += 1;
            let record = match result {
                Ok(value) => value,
                Err(error) => {
                    rows_invalid += 1;
                    errors.push(format!("row_{rows_total}_tsv_parse_error:{error}"));
                    continue;
                }
            };

            let mut row_errors = Vec::new();
            let mut key_parts = Vec::new();
            for field in kind.key_fields() {
                match required_cell(&record, &indexes, field) {
                    Ok(value) => key_parts.push(value.to_owned()),
                    Err(error) => row_errors.push(error),
                }
            }

            let time_field = kind.time_field();
            if let Err(error) = required_nonnegative_number(&record, &indexes, time_field) {
                row_errors.push(error);
            }
            if matches!(kind, DatasetKind::Figure1 | DatasetKind::Figure2) {
                if let Err(error) = required_nonnegative_number(&record, &indexes, "treatment_temp")
                {
                    row_errors.push(error);
                }
                if let Err(error) = required_cell(&record, &indexes, "units_treatment_temp") {
                    row_errors.push(error);
                }
            }

            let analyte = required_cell(&record, &indexes, "analyte")
                .map(str::to_owned)
                .unwrap_or_else(|error| {
                    row_errors.push(error);
                    "<invalid-analyte>".to_owned()
                });
            let units = required_cell(&record, &indexes, "units_analyte")
                .map(str::to_owned)
                .unwrap_or_else(|error| {
                    row_errors.push(error);
                    "<invalid-units>".to_owned()
                });
            let replicate = required_cell(&record, &indexes, "rep")
                .map(str::to_owned)
                .unwrap_or_else(|error| {
                    row_errors.push(error);
                    "<invalid-rep>".to_owned()
                });
            if let Ok(value) = replicate.parse::<u64>() {
                if value == 0 {
                    row_errors.push("replicate_must_be_positive".to_owned());
                }
                replicates.insert(value);
            } else {
                row_errors.push("replicate_not_positive_integer".to_owned());
            }

            let condition_parts = kind
                .condition_fields()
                .iter()
                .filter_map(|field| indexes.get(*field).and_then(|index| record.get(*index)))
                .map(str::trim)
                .collect::<Vec<_>>();
            if condition_parts.iter().any(|value| value.is_empty()) {
                row_errors.push("treatment_condition_field_blank".to_owned());
            } else {
                treatments.insert(condition_parts.join("\u{1f}"));
            }

            if row_errors.is_empty() {
                let key = key_parts.join("\u{1f}");
                if !row_keys.insert(key) {
                    duplicate_key_rows += 1;
                    row_errors.push("duplicate_time_treatment_analyte_replicate_key".to_owned());
                }
            }

            let value = required_cell(&record, &indexes, "value");
            match value {
                Ok("NA") => {
                    missing_na_total += 1;
                    summaries
                        .entry((analyte.clone(), units.clone()))
                        .or_default()
                        .missing_na += 1;
                }
                Ok("ND") => {
                    missing_nd_total += 1;
                    summaries
                        .entry((analyte.clone(), units.clone()))
                        .or_default()
                        .missing_nd += 1;
                }
                Ok(text) => match text.parse::<f64>() {
                    Ok(number) if number.is_finite() => {
                        summaries
                            .entry((analyte.clone(), units.clone()))
                            .or_default()
                            .values
                            .add(number);
                    }
                    Ok(_) => row_errors.push("value_non_finite".to_owned()),
                    Err(_) => {
                        row_errors.push("value_not_numeric_or_declared_missing_code".to_owned())
                    }
                },
                Err(error) => row_errors.push(error),
            }

            if row_errors.is_empty() {
                rows_valid += 1;
            } else {
                rows_invalid += 1;
                errors.extend(
                    row_errors
                        .into_iter()
                        .map(|error| format!("row_{rows_total}_{error}")),
                );
            }
        }
    }

    if header_valid && rows_total == 0 {
        errors.push("no_data_rows".to_owned());
    }
    if duplicate_key_rows > 0 {
        errors.push("duplicate_observation_keys_detected".to_owned());
    }
    let analyte_summaries = summaries
        .into_iter()
        .map(|((analyte, units), summary)| AnalyteSummary {
            analyte,
            units,
            observed_numeric_values: summary.values.count,
            missing_na: summary.missing_na,
            missing_nd: summary.missing_nd,
            numeric_summary: summary.values.summary(),
        })
        .collect::<Vec<_>>();

    let passed = header_valid
        && rows_total > 0
        && rows_invalid == 0
        && duplicate_key_rows == 0
        && errors.is_empty();

    IntakeReceipt {
        receipt_schema: "dryad-wine-fermentation-intake-v1",
        status: if passed { "AUDIT_OK" } else { "FAIL" },
        qualification_scope: "local_raw_file_structure_missingness_and_key_integrity_only",
        source_uri: SOURCE_URI,
        source_license_declaration: SOURCE_LICENSE,
        source_identity_status: "CALLER_DECLARED_NOT_AUTHENTICATED",
        dataset_kind: match kind {
            DatasetKind::Figure1 => "figure1",
            DatasetKind::Figure2 => "figure2",
            DatasetKind::FigureS1 => "figure-s1",
        }
        .to_owned(),
        expected_source_filename: kind.filename(),
        input_file: filename.to_owned(),
        input_bytes: bytes.len(),
        input_sha256: hex_sha256(bytes),
        header_valid,
        header_order_matches_reference,
        rows_total,
        rows_valid,
        rows_invalid,
        missing_na_total,
        missing_nd_total,
        duplicate_key_rows,
        distinct_technical_replicates: replicates.len() as u64,
        distinct_treatments: treatments.len() as u64,
        analyte_summaries,
        errors,
        warnings,
        claim_ceiling: "AUDIT_OK establishes only local byte digest, declared column contract, row-level parsing, preserved NA/ND missingness, and duplicate-key checks. It does not authenticate the download, establish biological independence, validate experimental design, prove scientific claims, or qualify fermentation forecasting, sensory quality, process control, production savings or safety.",
    }
}

fn parse_args() -> Result<(PathBuf, DatasetKind, PathBuf), String> {
    let mut input = None;
    let mut dataset = None;
    let mut output = None;
    let mut args = env::args().skip(1);

    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing_value_for_{flag}"))?;
        match flag.as_str() {
            "--input" => input = Some(PathBuf::from(value)),
            "--dataset" => dataset = Some(DatasetKind::parse(&value)?),
            "--out" => output = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown_argument_{flag}")),
        }
    }
    Ok((
        input.ok_or_else(|| "missing_--input".to_owned())?,
        dataset.ok_or_else(|| "missing_--dataset".to_owned())?,
        output.ok_or_else(|| "missing_--out".to_owned())?,
    ))
}

fn run() -> Result<ExitCode, String> {
    let (input_path, kind, output_path) = parse_args()?;
    let bytes = fs::read(&input_path).map_err(|error| format!("input_read_failed: {error}"))?;
    let filename = input_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("<non-utf8>");
    let receipt = audit_tsv(&bytes, kind, filename);
    let passed = receipt.status == "AUDIT_OK";
    let serialized = serde_json::to_vec_pretty(&receipt)
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
            eprintln!("dryad-wine-fermentation-intake: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER1: &str = "time_days\ttreatment_temp\tunits_treatment_temp\ttreatment_inoculation\tanalyte\tunits_analyte\trep\tvalue";
    const HEADER2: &str =
        "time_days\ttreatment_temp\tunits_treatment_temp\tanalyte\tunits_analyte\trep\tvalue";
    const HEADER3: &str = "treatment\ttime_hours\tanalyte\tunits_analyte\trep\tvalue";

    #[test]
    fn figure1_valid_rows_and_numeric_summary() {
        let data = format!(
            "{HEADER1}\n0\t17\tdegC\tHu\tcell_count\tcells_ml\t1\t1.5\n1\t17\tdegC\tHu\tcell_count\tcells_ml\t1\t2.5\n"
        );
        let receipt = audit_tsv(data.as_bytes(), DatasetKind::Figure1, "fixture.txt");
        assert_eq!(receipt.status, "AUDIT_OK");
        assert_eq!(receipt.rows_total, 2);
        assert_eq!(receipt.rows_valid, 2);
        assert_eq!(receipt.rows_invalid, 0);
        assert_eq!(receipt.input_sha256.len(), 64);
        assert_eq!(receipt.analyte_summaries.len(), 1);
        let summary = receipt.analyte_summaries.first().expect("one analyte");
        let numbers = summary.numeric_summary.as_ref().expect("numeric values");
        assert_eq!(numbers.count, 2);
        assert!((numbers.mean - 2.0).abs() < 1e-12);
    }

    #[test]
    fn na_and_nd_are_preserved_as_distinct_missingness() {
        let data = format!(
            "{HEADER2}\n0\t17\tdegC\tcell_count\tcells_ml\t1\tNA\n1\t17\tdegC\tcell_count\tcells_ml\t1\tND\n"
        );
        let receipt = audit_tsv(data.as_bytes(), DatasetKind::Figure2, "fixture.txt");
        assert_eq!(receipt.status, "AUDIT_OK");
        assert_eq!(receipt.missing_na_total, 1);
        assert_eq!(receipt.missing_nd_total, 1);
        assert_eq!(receipt.rows_valid, 2);
        let summary = receipt.analyte_summaries.first().expect("one analyte");
        assert_eq!(summary.missing_na, 1);
        assert_eq!(summary.missing_nd, 1);
        assert!(summary.numeric_summary.is_none());
    }

    #[test]
    fn duplicate_observation_key_fails_closed() {
        let data = format!(
            "{HEADER2}\n0\t17\tdegC\tcell_count\tcells_ml\t1\t1.5\n0\t17\tdegC\tcell_count\tcells_ml\t1\t1.7\n"
        );
        let receipt = audit_tsv(data.as_bytes(), DatasetKind::Figure2, "fixture.txt");
        assert_eq!(receipt.status, "FAIL");
        assert_eq!(receipt.duplicate_key_rows, 1);
        assert!(
            receipt
                .errors
                .iter()
                .any(|error| error == "duplicate_observation_keys_detected")
        );
    }

    #[test]
    fn unknown_missing_code_and_nonfinite_value_fail() {
        let data = format!("{HEADER3}\nfresh\t0\tethanol\tg_L\t1\tNaN\n");
        let receipt = audit_tsv(data.as_bytes(), DatasetKind::FigureS1, "fixture.txt");
        assert_eq!(receipt.status, "FAIL");
        assert!(
            receipt
                .errors
                .iter()
                .any(|error| error.ends_with("value_non_finite"))
        );
    }

    #[test]
    fn wrong_header_fails_closed() {
        let data = "time_days,value\n0,1\n";
        let receipt = audit_tsv(data.as_bytes(), DatasetKind::Figure1, "wrong.csv");
        assert_eq!(receipt.status, "FAIL");
        assert!(!receipt.header_valid);
    }

    #[test]
    fn negative_time_fails_closed() {
        let data = format!("{HEADER3}\nfrozen\t-1\tethanol\tg_L\t1\t0.5\n");
        let receipt = audit_tsv(data.as_bytes(), DatasetKind::FigureS1, "fixture.txt");
        assert_eq!(receipt.status, "FAIL");
        assert!(
            receipt
                .errors
                .iter()
                .any(|error| error.ends_with("negative_time_hours"))
        );
    }

    #[test]
    fn column_reordering_is_resolved_by_name_and_reported() {
        let data = "value\trep\tunits_analyte\tanalyte\ttime_days\tunits_treatment_temp\ttreatment_temp\n1.0\t1\tcells_ml\tcell_count\t0\tdegC\t17\n";
        let receipt = audit_tsv(data.as_bytes(), DatasetKind::Figure2, "reordered.txt");
        assert_eq!(receipt.status, "AUDIT_OK");
        assert!(!receipt.header_order_matches_reference);
        assert!(
            receipt
                .warnings
                .iter()
                .any(|warning| warning.contains("header_order_differs"))
        );
    }
}
