//! Structural and provenance intake for the vendored CC BY 4.0 tire-health dataset.
//!
//! The upstream data covers passenger-car pressure/temperature/speed monitoring.
//! It is not evidence of motorcycle tire grip, wear, wet braking or qualification.
//! The source file is split into text shards so their concatenation reconstructs
//! the upstream CSV without row filtering or added separators.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Cursor;
use std::path::Path;

const SHARD_COUNT: usize = 9;
const EXPECTED_ROWS: usize = 210_128;
const EXPECTED_SEGMENTS: usize = 1_755;
const EXPECTED_NORMAL: usize = 196_260;
const EXPECTED_ANOMALOUS: usize = 13_868;
const EXPECTED_METHOD_NORMAL: usize = 182_777;
const EXPECTED_METHOD_EXPERIMENTAL: usize = 26_029;
const EXPECTED_METHOD_NATURAL: usize = 1_322;
const SOURCE_BLOB_SHA1: &str = "0029b6f156847ecc052f57ae45fd025ad032d086";

const EXPECTED_HEADER: &[&str] = &[
    "Unit_ID",
    "Time_Step",
    "Pressure",
    "Temperature",
    "Speed",
    "Label",
    "Collection_Method",
];

#[derive(Debug, Serialize)]
struct RangeSummary {
    min: f64,
    max: f64,
}

#[derive(Debug, Serialize)]
struct DataSummary {
    data_rows: usize,
    segment_count: usize,
    label_counts: BTreeMap<String, usize>,
    collection_method_counts: BTreeMap<String, usize>,
    pressure_kpa: RangeSummary,
    temperature_c: RangeSummary,
    speed_kmh: RangeSummary,
}

#[derive(Debug, Serialize)]
struct IntakeReport {
    report_version: String,
    dataset_id: String,
    upstream_uri: String,
    upstream_blob_sha1: String,
    reconstructed_csv_sha256: String,
    shard_count: usize,
    summary: DataSummary,
    source_profile_match: bool,
    intake_integrity_pass: bool,
    claim_ceiling: String,
}

fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= 1e-6
}

fn inspect_csv(raw: &[u8]) -> Result<DataSummary> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .from_reader(Cursor::new(raw));

    let headers = reader.headers().context("reading tire dataset CSV header")?;
    let normalized: Vec<String> = headers
        .iter()
        .enumerate()
        .map(|(index, value)| {
            if index == 0 {
                value.trim_start_matches('\u{feff}').to_owned()
            } else {
                value.to_owned()
            }
        })
        .collect();
    let header_matches = normalized.len() == EXPECTED_HEADER.len()
        && normalized.iter().zip(EXPECTED_HEADER.iter())
            .all(|(actual, expected)| actual.as_str() == *expected);
    if !header_matches {
        bail!("unexpected tire-monitoring CSV header: {normalized:?}");
    }

    let mut rows = 0usize;
    let mut segments = BTreeSet::new();
    let mut label_counts = BTreeMap::<String, usize>::new();
    let mut method_counts = BTreeMap::<String, usize>::new();
    let mut pressure = RangeSummary { min: f64::INFINITY, max: f64::NEG_INFINITY };
    let mut temperature = RangeSummary { min: f64::INFINITY, max: f64::NEG_INFINITY };
    let mut speed = RangeSummary { min: f64::INFINITY, max: f64::NEG_INFINITY };

    for (index, result) in reader.records().enumerate() {
        let record_number = index + 2;
        let record = result.with_context(|| format!("parsing CSV record {record_number}"))?;
        if record.len() != EXPECTED_HEADER.len() {
            bail!("record {record_number} has {} columns, expected {}", record.len(), EXPECTED_HEADER.len());
        }

        let unit_id = record.get(0).context("missing Unit_ID")?.trim();
        if unit_id.is_empty() || unit_id.parse::<u64>().is_err() {
            bail!("record {record_number} has invalid Unit_ID");
        }
        let step = record.get(1).context("missing Time_Step")?.trim();
        if step.is_empty() || step.parse::<u64>().is_err() {
            bail!("record {record_number} has invalid Time_Step");
        }

        let read_number = |column: usize, name: &str| -> Result<f64> {
            let value = record.get(column).with_context(|| format!("record {record_number} missing {name}"))?.trim();
            let value = value.parse::<f64>()
                .with_context(|| format!("record {record_number} invalid {name}: {value}"))?;
            if !value.is_finite() {
                bail!("record {record_number} has non-finite {name}");
            }
            Ok(value)
        };

        let row_pressure = read_number(2, "Pressure")?;
        let row_temperature = read_number(3, "Temperature")?;
        let row_speed = read_number(4, "Speed")?;
        if row_pressure <= 0.0 {
            bail!("record {record_number} has non-positive pressure");
        }
        if row_speed < 0.0 {
            bail!("record {record_number} has negative speed");
        }

        let label = record.get(5).context("missing Label")?.trim();
        if label != "0" && label != "1" {
            bail!("record {record_number} has unsupported label {label}");
        }
        let method = record.get(6).context("missing Collection_Method")?.trim();
        if !["Normal", "Experimental", "Natural"].contains(&method) {
            bail!("record {record_number} has unsupported collection method {method}");
        }

        rows += 1;
        segments.insert(unit_id.to_owned());
        *label_counts.entry(label.to_owned()).or_default() += 1;
        *method_counts.entry(method.to_owned()).or_default() += 1;
        pressure.min = pressure.min.min(row_pressure);
        pressure.max = pressure.max.max(row_pressure);
        temperature.min = temperature.min.min(row_temperature);
        temperature.max = temperature.max.max(row_temperature);
        speed.min = speed.min.min(row_speed);
        speed.max = speed.max.max(row_speed);
    }

    if rows == 0 {
        bail!("tire-monitoring CSV contains no data rows");
    }

    Ok(DataSummary {
        data_rows: rows,
        segment_count: segments.len(),
        label_counts,
        collection_method_counts: method_counts,
        pressure_kpa: pressure,
        temperature_c: temperature,
        speed_kmh: speed,
    })
}

fn matches_source_profile(summary: &DataSummary) -> bool {
    summary.data_rows == EXPECTED_ROWS
        && summary.segment_count == EXPECTED_SEGMENTS
        && summary.label_counts.get("0") == Some(&EXPECTED_NORMAL)
        && summary.label_counts.get("1") == Some(&EXPECTED_ANOMALOUS)
        && summary.collection_method_counts.get("Normal") == Some(&EXPECTED_METHOD_NORMAL)
        && summary.collection_method_counts.get("Experimental") == Some(&EXPECTED_METHOD_EXPERIMENTAL)
        && summary.collection_method_counts.get("Natural") == Some(&EXPECTED_METHOD_NATURAL)
        && close(summary.pressure_kpa.min, 197.37)
        && close(summary.pressure_kpa.max, 289.31)
        && close(summary.temperature_c.min, 1.6)
        && close(summary.temperature_c.max, 45.94)
        && close(summary.speed_kmh.min, 2.84)
        && close(summary.speed_kmh.max, 144.94)
}

pub fn run(input_dir: &Path, output_path: Option<&Path>) -> Result<()> {
    let mut raw = Vec::new();
    for index in 1..=SHARD_COUNT {
        let path = input_dir.join(format!("tire_monitoring_dataset.csv.part-{index:04}"));
        let part = fs::read(&path)
            .with_context(|| format!("reading ordered tire dataset shard {}", path.display()))?;
        if part.is_empty() {
            bail!("tire dataset shard {} is empty", path.display());
        }
        raw.extend_from_slice(&part);
    }

    let summary = inspect_csv(&raw)?;
    let source_profile_match = matches_source_profile(&summary);
    let report = IntakeReport {
        report_version: "0.1.0".into(),
        dataset_id: "PASSENGER_TIRE_TPMS_CC_BY".into(),
        upstream_uri: "https://github.com/kerzeyc/tire-monitoring-dataset".into(),
        upstream_blob_sha1: SOURCE_BLOB_SHA1.into(),
        reconstructed_csv_sha256: format!("{:x}", Sha256::digest(&raw)),
        shard_count: SHARD_COUNT,
        summary,
        source_profile_match,
        intake_integrity_pass: source_profile_match,
        claim_ceiling: "CSV schema, content digest, source-profile counts and ranges only. This is passenger-car tire pressure/temperature/speed anomaly data, not motorcycle grip, tire-force, wear, wet-braking, structural or roadworthiness evidence.".into(),
    };
    let encoded = serde_json::to_string_pretty(&report)?;
    if let Some(path) = output_path {
        if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating report directory {}", parent.display()))?;
        }
        fs::write(path, format!("{encoded}\n"))
            .with_context(|| format!("writing tire intake report {}", path.display()))?;
    }
    println!("{encoded}");
    if !report.intake_integrity_pass {
        bail!("tire dataset intake checks did not match the frozen source profile");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bom_and_counts_segments_without_leaking_them() {
        let raw = b"\xef\xbb\xbfUnit_ID,Time_Step,Pressure,Temperature,Speed,Label,Collection_Method\r\n1,1,252,19.59,31.36,0,Normal\r\n1,2,257.19,22.91,57.48,1,Experimental\r\n2,1,250,20,40,0,Natural\r\n";
        let summary = inspect_csv(raw).expect("small fixture parses");
        assert_eq!(summary.data_rows, 3);
        assert_eq!(summary.segment_count, 2);
        assert_eq!(summary.label_counts.get("0"), Some(&2));
        assert_eq!(summary.label_counts.get("1"), Some(&1));
        assert_eq!(summary.collection_method_counts.get("Normal"), Some(&1));
        assert_eq!(summary.collection_method_counts.get("Experimental"), Some(&1));
        assert_eq!(summary.collection_method_counts.get("Natural"), Some(&1));
    }

    #[test]
    fn rejects_unknown_label_instead_of_relabeling() {
        let raw = b"Unit_ID,Time_Step,Pressure,Temperature,Speed,Label,Collection_Method\n1,1,252,19.59,31.36,2,Normal\n";
        assert!(inspect_csv(raw).is_err());
    }

    #[test]
    fn rejects_non_finite_measurements() {
        let raw = b"Unit_ID,Time_Step,Pressure,Temperature,Speed,Label,Collection_Method\n1,1,252,NaN,31.36,0,Normal\n";
        assert!(inspect_csv(raw).is_err());
    }

    #[test]
    fn rejects_missing_or_reordered_columns() {
        let raw = b"Unit_ID,Pressure,Time_Step,Temperature,Speed,Label,Collection_Method\n1,252,1,19.59,31.36,0,Normal\n";
        assert!(inspect_csv(raw).is_err());
    }

    #[test]
    fn expected_profile_fails_closed_on_changed_source_counts() {
        let summary = DataSummary {
            data_rows: 3,
            segment_count: 2,
            label_counts: BTreeMap::from([("0".into(), 2), ("1".into(), 1)]),
            collection_method_counts: BTreeMap::from([
                ("Normal".into(), 1),
                ("Experimental".into(), 1),
                ("Natural".into(), 1),
            ]),
            pressure_kpa: RangeSummary { min: 197.37, max: 289.31 },
            temperature_c: RangeSummary { min: 1.6, max: 45.94 },
            speed_kmh: RangeSummary { min: 2.84, max: 144.94 },
        };
        assert!(!matches_source_profile(&summary));
    }
}
