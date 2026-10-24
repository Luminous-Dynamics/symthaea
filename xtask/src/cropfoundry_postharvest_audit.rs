use anyhow::{Context, Result, anyhow};
use csv::{ReaderBuilder, StringRecord};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

const REQUIRED_COLUMNS: &[&str] = &[
    "lot_id",
    "stage_id",
    "replicate_id",
    "measurement_time",
    "mass_unit",
    "measurement_method",
    "input_mass_kg",
    "marketable_output_mass_kg",
    "edible_discard_mass_kg",
    "inedible_removed_mass_kg",
    "moisture_loss_mass_kg",
    "handling_loss_mass_kg",
];

const OUTFLOW_COLUMNS: &[&str] = &[
    "marketable_output_mass_kg",
    "edible_discard_mass_kg",
    "inedible_removed_mass_kg",
    "moisture_loss_mass_kg",
    "handling_loss_mass_kg",
];

#[derive(Debug, Serialize)]
struct RowMassAudit {
    csv_record_number: usize,
    lot_id: String,
    stage_id: String,
    replicate_id: String,
    measurement_time: String,
    input_mass_kg: Option<f64>,
    accounted_outflow_mass_kg: Option<f64>,
    unexplained_mass_residual_kg: Option<f64>,
    tolerance_kg: Option<f64>,
    residual_fraction_of_input: Option<f64>,
    marketable_mass_fraction: Option<f64>,
    within_mass_balance_tolerance: bool,
    issues: Vec<String>,
}

#[derive(Debug, Serialize)]
struct PostharvestAuditReceipt {
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
    missing_required_columns: Vec<String>,
    duplicate_header_names: Vec<String>,
    blank_header_positions: Vec<usize>,
    rows_with_width_mismatch: usize,
    duplicate_lot_stage_replicate_time_keys: usize,
    global_issues: Vec<String>,
    rows: Vec<RowMassAudit>,
    row_checks_pass: bool,
    training_authorized: bool,
    food_safety_validated: bool,
    causal_effects_identified: bool,
    notes: Vec<String>,
}

fn normalized_header(raw: &str) -> String {
    raw.trim_start_matches('\u{feff}').trim().to_owned()
}

fn parse_nonnegative_mass(value: &str, field: &str, issues: &mut Vec<String>) -> Option<f64> {
    match value.trim().parse::<f64>() {
        Ok(number) if number.is_finite() && number >= 0.0 => Some(number),
        _ => {
            issues.push(format!(
                "{field} must be a finite non-negative number in kg"
            ));
            None
        }
    }
}

fn header_index(headers: &[String], sought: &str) -> Option<usize> {
    headers.iter().position(|header| header == sought)
}

fn record_field(record: &StringRecord, index: Option<usize>) -> &str {
    index.and_then(|i| record.get(i)).unwrap_or("").trim()
}

fn audit_bytes(
    bytes: &[u8],
    delimiter: u8,
    source_id: &str,
    source_uri: &str,
    input_file_name: &str,
    absolute_tolerance_kg: f64,
    relative_tolerance: f64,
    expected_sha256: Option<&str>,
) -> Result<PostharvestAuditReceipt> {
    if bytes.is_empty() {
        return Err(anyhow!("postharvest audit input is empty"));
    }
    if source_id.trim().is_empty() || source_uri.trim().is_empty() {
        return Err(anyhow!("source id and source URI must be explicit"));
    }
    if !absolute_tolerance_kg.is_finite()
        || absolute_tolerance_kg < 0.0
        || !relative_tolerance.is_finite()
        || relative_tolerance < 0.0
    {
        return Err(anyhow!(
            "mass-balance tolerances must be finite and non-negative"
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
        .context("postharvest CSV header could not be read")?
        .clone();
    let headers: Vec<String> = raw_headers.iter().map(normalized_header).collect();
    let mut global_issues = Vec::new();
    if headers.is_empty() || headers.iter().all(|header| header.is_empty()) {
        global_issues.push("CSV has no non-empty header fields".to_owned());
    }

    let mut header_counts = BTreeMap::<String, usize>::new();
    for header in &headers {
        if !header.is_empty() {
            *header_counts.entry(header.clone()).or_default() += 1;
        }
    }
    let duplicate_header_names = header_counts
        .iter()
        .filter_map(|(name, count)| (*count > 1).then_some(name.clone()))
        .collect::<Vec<_>>();
    let blank_header_positions = headers
        .iter()
        .enumerate()
        .filter_map(|(index, header)| header.is_empty().then_some(index + 1))
        .collect::<Vec<_>>();
    let missing_required_columns = REQUIRED_COLUMNS
        .iter()
        .filter(|name| !headers.iter().any(|header| header == **name))
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();

    if !duplicate_header_names.is_empty() {
        global_issues.push(format!(
            "duplicate header names: {duplicate_header_names:?}"
        ));
    }
    if !blank_header_positions.is_empty() {
        global_issues.push(format!(
            "blank header fields at 1-based positions: {blank_header_positions:?}"
        ));
    }
    if !missing_required_columns.is_empty() {
        global_issues.push(format!(
            "missing required columns: {missing_required_columns:?}"
        ));
    }

    let input_index = header_index(&headers, "input_mass_kg");
    let outflow_indices: Vec<Option<usize>> = OUTFLOW_COLUMNS
        .iter()
        .map(|field| header_index(&headers, field))
        .collect();
    let unit_index = header_index(&headers, "mass_unit");
    let method_index = header_index(&headers, "measurement_method");
    let lot_index = header_index(&headers, "lot_id");
    let stage_index = header_index(&headers, "stage_id");
    let replicate_index = header_index(&headers, "replicate_id");
    let time_index = header_index(&headers, "measurement_time");

    let mut records = Vec::new();
    let mut rows_with_width_mismatch = 0usize;
    let mut seen_keys = BTreeSet::<(String, String, String, String)>::new();
    let mut duplicate_lot_stage_replicate_time_keys = 0usize;

    for (record_index, record_result) in reader.records().enumerate() {
        let record = record_result
            .with_context(|| format!("invalid postharvest CSV record {}", record_index + 1))?;
        if record.len() != headers.len() {
            rows_with_width_mismatch += 1;
        }
        let lot_id = record_field(&record, lot_index).to_owned();
        let stage_id = record_field(&record, stage_index).to_owned();
        let replicate_id = record_field(&record, replicate_index).to_owned();
        let measurement_time = record_field(&record, time_index).to_owned();
        let mut issues = Vec::new();

        if record.len() != headers.len() {
            issues.push(format!(
                "row has {} fields but header has {}",
                record.len(),
                headers.len()
            ));
        }
        for (label, value) in [
            ("lot_id", lot_id.as_str()),
            ("stage_id", stage_id.as_str()),
            ("replicate_id", replicate_id.as_str()),
            ("measurement_time", measurement_time.as_str()),
            ("measurement_method", record_field(&record, method_index)),
        ] {
            if value.is_empty() {
                issues.push(format!("{label} must be non-empty"));
            }
        }
        if !record_field(&record, unit_index).eq_ignore_ascii_case("kg") {
            issues.push(
                "mass_unit must be exactly kg for all mass columns in this v0 profile".to_owned(),
            );
        }

        if !lot_id.is_empty()
            && !stage_id.is_empty()
            && !replicate_id.is_empty()
            && !measurement_time.is_empty()
            && !seen_keys.insert((
                lot_id.clone(),
                stage_id.clone(),
                replicate_id.clone(),
                measurement_time.clone(),
            ))
        {
            duplicate_lot_stage_replicate_time_keys += 1;
            issues.push("duplicate lot/stage/replicate/measurement_time key".to_owned());
        }

        let input_mass = input_index.and_then(|index| {
            parse_nonnegative_mass(
                record_field(&record, Some(index)),
                "input_mass_kg",
                &mut issues,
            )
        });
        if input_mass.is_some_and(|mass| mass <= 0.0) {
            issues.push("input_mass_kg must be greater than zero".to_owned());
        }
        let mut outflow_total = 0.0;
        let mut all_outflows_valid = true;
        let mut parsed_outflows = Vec::with_capacity(OUTFLOW_COLUMNS.len());
        for (index, name) in outflow_indices.iter().zip(OUTFLOW_COLUMNS) {
            let parsed = index.and_then(|i| {
                parse_nonnegative_mass(record_field(&record, Some(i)), name, &mut issues)
            });
            if let Some(mass) = parsed {
                outflow_total += mass;
            } else {
                all_outflows_valid = false;
            }
            parsed_outflows.push(parsed);
        }
        let accounted_outflow_mass = all_outflows_valid.then_some(outflow_total);
        let residual = input_mass
            .zip(accounted_outflow_mass)
            .map(|(input, output)| input - output);
        let tolerance =
            input_mass.map(|input| absolute_tolerance_kg.max(input * relative_tolerance));
        let within_tolerance = residual
            .zip(tolerance)
            .is_some_and(|(delta, tolerance)| delta.abs() <= tolerance)
            && !issues.iter().any(|issue| {
                issue.contains("mass_unit")
                    || issue.contains("mass_kg")
                    || issue.contains("duplicate lot/")
                    || issue.contains("row has")
            });
        if residual
            .zip(tolerance)
            .is_some_and(|(delta, tolerance)| delta.abs() > tolerance)
        {
            issues.push("mass-balance residual exceeds declared tolerance".to_owned());
        }

        let residual_fraction = residual
            .zip(input_mass)
            .and_then(|(delta, input)| (input > 0.0).then_some(delta / input));
        let marketable_mass = parsed_outflows.first().copied().flatten();
        let marketable_fraction = marketable_mass
            .zip(input_mass)
            .and_then(|(marketable, input)| (input > 0.0).then_some(marketable / input));

        records.push(RowMassAudit {
            csv_record_number: record_index + 1,
            lot_id,
            stage_id,
            replicate_id,
            measurement_time,
            input_mass_kg: input_mass,
            accounted_outflow_mass_kg: accounted_outflow_mass,
            unexplained_mass_residual_kg: residual,
            tolerance_kg: tolerance,
            residual_fraction_of_input: residual_fraction,
            marketable_mass_fraction: marketable_fraction,
            within_mass_balance_tolerance: within_tolerance,
            issues,
        });
    }

    if rows_with_width_mismatch > 0 {
        global_issues.push(format!(
            "{rows_with_width_mismatch} records have a field width different from the header"
        ));
    }
    if duplicate_lot_stage_replicate_time_keys > 0 {
        global_issues.push(format!(
            "{duplicate_lot_stage_replicate_time_keys} duplicate lot/stage/replicate/time keys detected"
        ));
    }
    let digest = Sha256::digest(bytes);
    let content_sha256 = format!("{digest:x}");
    let digest_matches_expected =
        expected_sha256.map(|expected| expected.eq_ignore_ascii_case(&content_sha256));
    if digest_matches_expected == Some(false) {
        global_issues.push(
            "local byte SHA-256 mismatches the separately supplied expected digest".to_owned(),
        );
    }
    if records.is_empty() {
        global_issues.push("CSV has no data records".to_owned());
    }
    let row_checks_pass = global_issues.is_empty()
        && records
            .iter()
            .all(|row| row.within_mass_balance_tolerance && row.issues.is_empty());

    Ok(PostharvestAuditReceipt {
        receipt_schema: "cropfoundry-postharvest-mass-balance-receipt-v0".to_owned(),
        stage: "local_schema_and_mass_balance_audit_only".to_owned(),
        source_id: source_id.to_owned(),
        source_uri: source_uri.to_owned(),
        input_file_name: input_file_name.to_owned(),
        content_sha256,
        expected_sha256: expected_sha256.map(str::to_ascii_lowercase),
        digest_matches_expected,
        byte_length: bytes.len(),
        delimiter: delimiter as char,
        header_field_count: headers.len(),
        data_record_count: records.len(),
        missing_required_columns,
        duplicate_header_names,
        blank_header_positions,
        rows_with_width_mismatch,
        duplicate_lot_stage_replicate_time_keys,
        global_issues,
        rows: records,
        row_checks_pass,
        training_authorized: false,
        food_safety_validated: false,
        causal_effects_identified: false,
        notes: vec![
            "The digest identifies exact local bytes; it does not establish source authenticity, rights, or permission.".to_owned(),
            "All listed mass components must be non-overlapping; added materials, wash water, packaging, gas exchange and unmeasured residues require a revised balance profile.".to_owned(),
            "Mass-balance closure does not prove marketability, freshness, microbiological safety, nutrition retention, or causal effect.".to_owned(),
            "A passing receipt only indicates schema/key checks and residual tolerance for this declared accounting equation.".to_owned(),
            "Training authorization remains false regardless of audit result.".to_owned(),
        ],
    })
}

/// Audit a local stage-by-stage CSV and emit a receipt. No download, training, or safety claim occurs.
pub fn run(
    input: PathBuf,
    source_id: String,
    source_uri: String,
    delimiter: char,
    absolute_tolerance_kg: f64,
    relative_tolerance: f64,
    expected_sha256: Option<String>,
    out: Option<PathBuf>,
) -> Result<()> {
    if !delimiter.is_ascii() {
        return Err(anyhow!("CSV delimiter must be a single ASCII character"));
    }
    let bytes = fs::read(&input)
        .with_context(|| format!("cannot read local postharvest CSV {}", input.display()))?;
    let file_name = input
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("postharvest.csv")
        .to_owned();
    let receipt = audit_bytes(
        &bytes,
        delimiter as u8,
        &source_id,
        &source_uri,
        &file_name,
        absolute_tolerance_kg,
        relative_tolerance,
        expected_sha256.as_deref(),
    )?;
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
    if receipt.row_checks_pass {
        Ok(())
    } else {
        Err(anyhow!(
            "postharvest mass-balance audit failed; receipt emitted/saved but does not assert data quality or food safety"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_ID: &str = "SYNTHETIC-FIXTURE";
    const SOURCE_URI: &str = "urn:cropfoundry:postharvest-fixture-v0";
    const HEADER: &str = "lot_id;stage_id;replicate_id;measurement_time;mass_unit;measurement_method;input_mass_kg;marketable_output_mass_kg;edible_discard_mass_kg;inedible_removed_mass_kg;moisture_loss_mass_kg;handling_loss_mass_kg";

    fn audit(input: &str) -> PostharvestAuditReceipt {
        audit_bytes(
            input.as_bytes(),
            b';',
            SOURCE_ID,
            SOURCE_URI,
            "fixture.csv",
            0.02,
            0.01,
            None,
        )
        .unwrap()
    }

    #[test]
    fn cropfoundry_postharvest_balanced_lot_stage_emits_a_non_authorizing_receipt() {
        let input = format!(
            "{HEADER}\nlot-1;harvest;rep-1;2026-01-01T08:00:00Z;kg;calibrated_scale;10;8;1;0.5;0.4;0.1\n"
        );
        let receipt = audit(&input);
        assert!(receipt.row_checks_pass);
        assert_eq!(receipt.rows.len(), 1);
        assert!(receipt.training_authorized == false);
        assert!(receipt.food_safety_validated == false);
        assert!(receipt.causal_effects_identified == false);
        assert_eq!(receipt.rows[0].unexplained_mass_residual_kg, Some(0.0));
    }

    #[test]
    fn cropfoundry_postharvest_unexplained_mass_residual_fails_closed() {
        let input = format!(
            "{HEADER}\nlot-1;storage;rep-1;2026-01-02T08:00:00Z;kg;calibrated_scale;10;7;1;0.5;0.5;0.2\n"
        );
        let receipt = audit(&input);
        assert!(!receipt.row_checks_pass);
        assert!(
            receipt.rows[0]
                .issues
                .iter()
                .any(|issue| issue.contains("residual exceeds"))
        );
    }

    #[test]
    fn cropfoundry_postharvest_negative_mass_is_rejected() {
        let input = format!("{HEADER}\nlot-1;harvest;rep-1;2026-01-01;kg;scale;-1;0;0;0;0;0\n");
        let receipt = audit(&input);
        assert!(!receipt.row_checks_pass);
        assert!(
            receipt.rows[0]
                .issues
                .iter()
                .any(|issue| issue.contains("input_mass_kg"))
        );
    }

    #[test]
    fn cropfoundry_postharvest_wrong_mass_unit_is_rejected() {
        let input = format!(
            "{HEADER}\nlot-1;harvest;rep-1;2026-01-01;g;scale;10000;8000;1000;500;400;100\n"
        );
        let receipt = audit(&input);
        assert!(!receipt.row_checks_pass);
        assert!(
            receipt.rows[0]
                .issues
                .iter()
                .any(|issue| issue.contains("mass_unit"))
        );
    }

    #[test]
    fn cropfoundry_postharvest_duplicate_lot_stage_replicate_time_fails_closed() {
        let row = "lot-1;harvest;rep-1;2026-01-01;kg;scale;10;8;1;0.5;0.4;0.1";
        let input = format!("{HEADER}\n{row}\n{row}\n");
        let receipt = audit(&input);
        assert!(!receipt.row_checks_pass);
        assert_eq!(receipt.duplicate_lot_stage_replicate_time_keys, 1);
    }

    #[test]
    fn cropfoundry_postharvest_missing_required_column_fails_closed() {
        let input = "lot_id;stage_id;input_mass_kg\nlot-1;harvest;10\n";
        let receipt = audit(&input);
        assert!(!receipt.row_checks_pass);
        assert!(
            receipt
                .missing_required_columns
                .contains(&"marketable_output_mass_kg".to_owned())
        );
    }

    #[test]
    fn cropfoundry_postharvest_wrong_external_digest_does_not_pass() {
        let row = "lot-1;harvest;rep-1;2026-01-01;kg;scale;10;8;1;0.5;0.4;0.1";
        let input = format!("{HEADER}\n{row}\n");
        let expected = "a".repeat(64);
        let receipt = audit_bytes(
            input.as_bytes(),
            b';',
            SOURCE_ID,
            SOURCE_URI,
            "fixture.csv",
            0.02,
            0.01,
            Some(&expected),
        )
        .unwrap();
        assert_eq!(receipt.digest_matches_expected, Some(false));
        assert!(!receipt.row_checks_pass);
    }

    #[test]
    fn cropfoundry_postharvest_source_identity_is_preserved_not_authorized() {
        let input =
            format!("{HEADER}\nlot-1;harvest;rep-1;2026-01-01;kg;scale;10;8;1;0.5;0.4;0.1\n");
        let receipt = audit(&input);
        assert_eq!(receipt.source_id, SOURCE_ID);
        assert_eq!(receipt.source_uri, SOURCE_URI);
        assert!(!receipt.training_authorized);
    }
}
