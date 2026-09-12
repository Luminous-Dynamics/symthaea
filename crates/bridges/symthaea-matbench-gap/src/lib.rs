// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pinned, network-free Matbench experimental-gap truth adapter.
//!
//! The official entry point accepts only the exact compressed artifact pinned by
//! matminer's metadata for `matbench_expt_gap`. Artifact identity is checked
//! before decompression; parsing then enforces the dataframe schema, row count,
//! normalized composition identity, and overlap with Symthaea's exposed
//! band-gap training compositions before producing Benchmark Zero truth.

#![forbid(unsafe_code)]

use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::io::Read;
use symthaea_bandgap::periodic_table::atomic_number;
use symthaea_energy_benchmark_zero::{BandgapTruthSet, DataSliceId, DatasetProvenance};
use symthaea_process_discovery::formula::parse_formula;
use thiserror::Error;

pub const MATBENCH_EXPT_GAP_DATASET_ID: &str = "matbench_expt_gap";
pub const MATBENCH_EXPT_GAP_ROWS: usize = 4_604;
pub const MATBENCH_EXPT_GAP_URL: &str =
    "https://ml.materialsproject.org/projects/matbench_expt_gap.json.gz";
pub const MATBENCH_EXPT_GAP_SHA256: &str =
    "783e7d1461eb83b00b2f2942da4b95fda5e58a0d1ae26b581c24cf8a82ca75b2";
/// Matminer's pinned metadata does not currently declare a dataset-specific
/// license for this entry. Preserve that absence instead of inventing one.
pub const MATBENCH_EXPT_GAP_LICENSE_DISCLOSURE: &str =
    "UNKNOWN: dataset-specific license not declared in pinned matminer metadata";

const MAX_COMPRESSED_BYTES: usize = 2 * 1024 * 1024;
const MAX_DECOMPRESSED_BYTES: usize = 8 * 1024 * 1024;
const FRACTION_SCALE: f64 = 1_000_000_000.0;
const ROW_ORDER_DIGEST_DOMAIN: &[u8] = b"symthaea.matbench-gap.row-order.v0\0";

/// A content pin for a split-oriented experimental-gap artifact.
///
/// The generic form exists for deterministic parser fixtures and future pinned
/// revisions. Benchmark Zero should use [`parse_official_matbench_expt_gap`]
/// when claiming compatibility with the current official Matbench artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedGapArtifact {
    pub dataset_id: String,
    pub source_uri: String,
    pub expected_sha256: String,
    pub expected_rows: usize,
    pub license_disclosure: String,
}

impl PinnedGapArtifact {
    pub fn official_matbench_expt_gap() -> Self {
        Self {
            dataset_id: MATBENCH_EXPT_GAP_DATASET_ID.to_owned(),
            source_uri: MATBENCH_EXPT_GAP_URL.to_owned(),
            expected_sha256: MATBENCH_EXPT_GAP_SHA256.to_owned(),
            expected_rows: MATBENCH_EXPT_GAP_ROWS,
            license_disclosure: MATBENCH_EXPT_GAP_LICENSE_DISCLOSURE.to_owned(),
        }
    }

    fn validate(&self) -> Result<(), AdapterError> {
        if self.dataset_id.trim().is_empty()
            || self.source_uri.trim().is_empty()
            || self.license_disclosure.trim().is_empty()
        {
            return Err(AdapterError::InvalidPin(
                "dataset id, source URI, and license disclosure must be non-empty".into(),
            ));
        }
        if self.expected_rows == 0 {
            return Err(AdapterError::InvalidPin(
                "expected row count must be positive".into(),
            ));
        }
        if self.expected_sha256.len() != 64
            || !self.expected_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(AdapterError::InvalidPin(
                "expected SHA-256 must be exactly 64 hexadecimal characters".into(),
            ));
        }
        Ok(())
    }
}

/// Phase-insensitive atomic-fraction composition identity.
///
/// Fractions are normalized to sum to one then quantized to 1e-9. This is a
/// deliberately conservative leakage identity: polymorphs with identical
/// stoichiometry collide even when their display labels differ.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CompositionFingerprint(pub Vec<(u8, u64)>);

impl CompositionFingerprint {
    fn from_amounts(amounts: BTreeMap<u8, f64>) -> Result<Self, AdapterError> {
        if amounts.is_empty() {
            return Err(AdapterError::InvalidComposition(
                "composition cannot be empty".into(),
            ));
        }

        let total: f64 = amounts.values().sum();
        if !total.is_finite() || total <= 0.0 {
            return Err(AdapterError::InvalidComposition(
                "composition total must be finite and positive".into(),
            ));
        }

        let mut fingerprint = Vec::with_capacity(amounts.len());
        for (atomic_number, amount) in amounts {
            if !amount.is_finite() || amount <= 0.0 {
                return Err(AdapterError::InvalidComposition(format!(
                    "element Z={atomic_number} has non-finite or non-positive amount"
                )));
            }
            let normalized = amount / total;
            let quantized = (normalized * FRACTION_SCALE).round();
            if !quantized.is_finite() || quantized <= 0.0 || quantized > u64::MAX as f64 {
                return Err(AdapterError::InvalidComposition(format!(
                    "element Z={atomic_number} cannot be represented in canonical fraction space"
                )));
            }
            fingerprint.push((atomic_number, quantized as u64));
        }

        Ok(Self(fingerprint))
    }

    fn candidate_id(&self) -> String {
        let mut value = String::from("composition:");
        for (index, (atomic_number, fraction)) in self.0.iter().enumerate() {
            if index > 0 {
                value.push('|');
            }
            write!(&mut value, "Z{atomic_number}:{fraction}")
                .expect("writing to String cannot fail");
        }
        value
    }
}

/// One normalized truth record in exact artifact order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatbenchGapRecord {
    pub candidate_id: String,
    pub composition: CompositionFingerprint,
    pub experimental_gap_ev: f64,
}

/// One conservative overlap between Matbench truth and Symthaea's exposed
/// band-gap training data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingOverlap {
    pub candidate_id: String,
    pub symthaea_training_labels: Vec<String>,
}

/// Parsed, pinned truth plus order-preserving source identity.
#[derive(Debug, Clone, PartialEq)]
pub struct MatbenchGapDataset {
    pub truth: BandgapTruthSet,
    pub ordered_records: Vec<MatbenchGapRecord>,
    /// SHA-256 of the exact compressed source artifact.
    pub compressed_sha256: String,
    /// Domain-separated SHA-256 of ordered normalized records after schema
    /// validation.
    pub row_order_sha256: String,
}

impl MatbenchGapDataset {
    /// Composition-level overlap against Symthaea's currently exposed curated
    /// band-gap training table.
    pub fn symthaea_training_overlap(&self) -> Result<Vec<TrainingOverlap>, AdapterError> {
        let mut training_by_fingerprint: BTreeMap<CompositionFingerprint, Vec<String>> =
            BTreeMap::new();

        for entry in symthaea_bandgap::training_data::load_training_data() {
            let fingerprint = fingerprint_from_atomic_fractions(&entry.composition)?;
            training_by_fingerprint
                .entry(fingerprint)
                .or_default()
                .push(entry.formula.to_owned());
        }

        let mut overlaps = Vec::new();
        for record in &self.ordered_records {
            if let Some(labels) = training_by_fingerprint.get(&record.composition) {
                let mut labels = labels.clone();
                labels.sort();
                labels.dedup();
                overlaps.push(TrainingOverlap {
                    candidate_id: record.candidate_id.clone(),
                    symthaea_training_labels: labels,
                });
            }
        }
        Ok(overlaps)
    }

    /// Fail closed when any normalized composition in the truth artifact also
    /// occurs in Symthaea's exposed band-gap training data.
    ///
    /// Passing this theorem does not establish absence of all historical prior
    /// knowledge used to tune hand-written baselines.
    pub fn ensure_no_symthaea_training_overlap(&self) -> Result<(), AdapterError> {
        let overlap = self.symthaea_training_overlap()?;
        if overlap.is_empty() {
            Ok(())
        } else {
            Err(AdapterError::TrainingCompositionOverlap(overlap))
        }
    }
}

#[derive(Debug, Deserialize)]
struct PandasSplitFrame {
    columns: Vec<String>,
    index: Vec<Value>,
    data: Vec<Vec<Value>>,
}

/// Parse only the exact current upstream artifact pinned by matminer's
/// `matbench_expt_gap` metadata.
pub fn parse_official_matbench_expt_gap(
    compressed_bytes: &[u8],
) -> Result<MatbenchGapDataset, AdapterError> {
    parse_pinned_gap_artifact(
        compressed_bytes,
        &PinnedGapArtifact::official_matbench_expt_gap(),
    )
}

/// Parse any explicitly pinned artifact with the same dataframe schema.
///
/// This function is content-addressed but makes no claim that a caller-defined
/// pin is an official Matbench release. Use the official wrapper above for that
/// claim.
pub fn parse_pinned_gap_artifact(
    compressed_bytes: &[u8],
    pin: &PinnedGapArtifact,
) -> Result<MatbenchGapDataset, AdapterError> {
    pin.validate()?;

    if compressed_bytes.len() > MAX_COMPRESSED_BYTES {
        return Err(AdapterError::CompressedTooLarge {
            actual: compressed_bytes.len(),
            limit: MAX_COMPRESSED_BYTES,
        });
    }

    let actual_sha256 = sha256_hex(compressed_bytes);
    if !actual_sha256.eq_ignore_ascii_case(&pin.expected_sha256) {
        return Err(AdapterError::DigestMismatch {
            expected: pin.expected_sha256.clone(),
            actual: actual_sha256,
        });
    }

    let mut decoder = GzDecoder::new(compressed_bytes);
    let mut json_bytes = Vec::new();
    decoder
        .by_ref()
        .take((MAX_DECOMPRESSED_BYTES + 1) as u64)
        .read_to_end(&mut json_bytes)?;
    if json_bytes.len() > MAX_DECOMPRESSED_BYTES {
        return Err(AdapterError::DecompressedTooLarge {
            actual: json_bytes.len(),
            limit: MAX_DECOMPRESSED_BYTES,
        });
    }

    parse_verified_split_json(&json_bytes, pin, actual_sha256)
}

fn parse_verified_split_json(
    json_bytes: &[u8],
    pin: &PinnedGapArtifact,
    compressed_sha256: String,
) -> Result<MatbenchGapDataset, AdapterError> {
    let frame: PandasSplitFrame = serde_json::from_slice(json_bytes)?;

    let expected_columns = ["composition", "gap expt"];
    if frame.columns.len() != expected_columns.len()
        || frame
            .columns
            .iter()
            .map(String::as_str)
            .ne(expected_columns.into_iter())
    {
        return Err(AdapterError::InvalidShape(format!(
            "expected columns {:?}, got {:?}",
            expected_columns, frame.columns
        )));
    }

    if frame.data.len() != pin.expected_rows || frame.index.len() != pin.expected_rows {
        return Err(AdapterError::RowCountMismatch {
            expected: pin.expected_rows,
            data_rows: frame.data.len(),
            index_rows: frame.index.len(),
        });
    }

    let mut index_ids = BTreeSet::new();
    for index_value in &frame.index {
        let encoded = serde_json::to_string(index_value)?;
        if !index_ids.insert(encoded.clone()) {
            return Err(AdapterError::DuplicateIndex(encoded));
        }
    }

    let mut truth_by_candidate = BTreeMap::new();
    let mut ordered_records = Vec::with_capacity(pin.expected_rows);
    let mut seen_compositions = BTreeSet::new();

    for (row_index, row) in frame.data.into_iter().enumerate() {
        if row.len() != 2 {
            return Err(AdapterError::InvalidShape(format!(
                "row {row_index} has {} values; expected 2",
                row.len()
            )));
        }

        let composition = composition_fingerprint_from_json(&row[0]).map_err(|error| {
            AdapterError::InvalidRow(format!("row {row_index} composition: {error}"))
        })?;
        if !seen_compositions.insert(composition.clone()) {
            return Err(AdapterError::DuplicateNormalizedComposition(
                composition.candidate_id(),
            ));
        }

        let gap = row[1].as_f64().ok_or_else(|| {
            AdapterError::InvalidRow(format!(
                "row {row_index} experimental gap must be numeric"
            ))
        })?;
        if !gap.is_finite() || gap < 0.0 {
            return Err(AdapterError::InvalidRow(format!(
                "row {row_index} experimental gap must be finite and non-negative"
            )));
        }

        let candidate_id = composition.candidate_id();
        if truth_by_candidate.insert(candidate_id.clone(), gap).is_some() {
            return Err(AdapterError::DuplicateNormalizedComposition(candidate_id));
        }
        ordered_records.push(MatbenchGapRecord {
            candidate_id,
            composition,
            experimental_gap_ev: gap,
        });
    }

    let row_order_sha256 = domain_separated_sha256(ROW_ORDER_DIGEST_DOMAIN, &ordered_records)?;
    let truth = BandgapTruthSet {
        provenance: DatasetProvenance {
            slice: DataSliceId::new(
                pin.dataset_id.clone(),
                format!("full-canonical-order@sha256:{}", pin.expected_sha256),
            )?,
            source_uri: pin.source_uri.clone(),
            content_digest: format!("sha256:{}", pin.expected_sha256),
            license: pin.license_disclosure.clone(),
        },
        experimental_gap_ev: truth_by_candidate,
    };
    truth.validate()?;

    Ok(MatbenchGapDataset {
        truth,
        ordered_records,
        compressed_sha256,
        row_order_sha256,
    })
}

fn composition_fingerprint_from_json(value: &Value) -> Result<CompositionFingerprint, AdapterError> {
    match value {
        Value::String(formula) => fingerprint_from_formula(formula),
        Value::Object(amounts) => fingerprint_from_symbol_amount_map(amounts),
        _ => Err(AdapterError::InvalidComposition(
            "expected formula string or element->amount object".into(),
        )),
    }
}

fn fingerprint_from_formula(formula: &str) -> Result<CompositionFingerprint, AdapterError> {
    let compact: String = formula.chars().filter(|character| !character.is_whitespace()).collect();
    let parsed = parse_formula(&compact).ok_or_else(|| {
        AdapterError::InvalidComposition(format!("unsupported or malformed formula {formula:?}"))
    })?;

    let mut amounts = BTreeMap::new();
    for (symbol, amount) in parsed {
        let atomic_number = atomic_number(&symbol).ok_or_else(|| {
            AdapterError::InvalidComposition(format!("unknown element symbol {symbol:?}"))
        })?;
        amounts.insert(atomic_number, f64::from(amount));
    }
    CompositionFingerprint::from_amounts(amounts)
}

fn fingerprint_from_symbol_amount_map(
    amounts: &Map<String, Value>,
) -> Result<CompositionFingerprint, AdapterError> {
    let mut atomic_amounts = BTreeMap::new();
    for (symbol, raw_amount) in amounts {
        if symbol.starts_with('@') {
            return Err(AdapterError::InvalidComposition(format!(
                "unexpected Monty metadata key {symbol:?}; expected Composition.as_dict element mapping"
            )));
        }
        let atomic_number = atomic_number(symbol).ok_or_else(|| {
            AdapterError::InvalidComposition(format!("unknown element symbol {symbol:?}"))
        })?;
        let amount = raw_amount.as_f64().ok_or_else(|| {
            AdapterError::InvalidComposition(format!(
                "amount for element {symbol:?} must be numeric"
            ))
        })?;
        if atomic_amounts.insert(atomic_number, amount).is_some() {
            return Err(AdapterError::InvalidComposition(format!(
                "duplicate element identity for {symbol:?}"
            )));
        }
    }
    CompositionFingerprint::from_amounts(atomic_amounts)
}

fn fingerprint_from_atomic_fractions(
    composition: &[(u8, f64)],
) -> Result<CompositionFingerprint, AdapterError> {
    let mut amounts = BTreeMap::new();
    for &(atomic_number, fraction) in composition {
        let entry = amounts.entry(atomic_number).or_insert(0.0);
        *entry += fraction;
    }
    CompositionFingerprint::from_amounts(amounts)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn domain_separated_sha256<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<String, AdapterError> {
    let encoded = serde_json::to_vec(value)?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(encoded);
    let digest = hasher.finalize();
    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    Ok(output)
}

#[derive(Debug, Error)]
pub enum AdapterError {
    #[error("invalid artifact pin: {0}")]
    InvalidPin(String),
    #[error("compressed artifact is too large: {actual} bytes > {limit} byte limit")]
    CompressedTooLarge { actual: usize, limit: usize },
    #[error("decompressed artifact is too large: {actual} bytes > {limit} byte limit")]
    DecompressedTooLarge { actual: usize, limit: usize },
    #[error("artifact SHA-256 mismatch: expected {expected}, got {actual}")]
    DigestMismatch { expected: String, actual: String },
    #[error("gzip decode failed: {0}")]
    Gzip(#[from] std::io::Error),
    #[error("JSON decode/encode failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid split-dataframe shape: {0}")]
    InvalidShape(String),
    #[error(
        "artifact row count mismatch: expected {expected}, data rows {data_rows}, index rows {index_rows}"
    )]
    RowCountMismatch {
        expected: usize,
        data_rows: usize,
        index_rows: usize,
    },
    #[error("duplicate dataframe index value {0}")]
    DuplicateIndex(String),
    #[error("duplicate normalized composition {0}")]
    DuplicateNormalizedComposition(String),
    #[error("invalid composition: {0}")]
    InvalidComposition(String),
    #[error("invalid dataset row: {0}")]
    InvalidRow(String),
    #[error("benchmark provenance contract rejected dataset: {0}")]
    Benchmark(#[from] symthaea_energy_benchmark_zero::BenchmarkError),
    #[error("truth compositions overlap Symthaea band-gap training data: {0:?}")]
    TrainingCompositionOverlap(Vec<TrainingOverlap>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    fn gzip(json: &[u8]) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(json).unwrap();
        encoder.finish().unwrap()
    }

    fn fixture_pin(bytes: &[u8], rows: usize) -> PinnedGapArtifact {
        PinnedGapArtifact {
            dataset_id: "fixture_gap".into(),
            source_uri: "https://example.invalid/fixture.json.gz".into(),
            expected_sha256: sha256_hex(bytes),
            expected_rows: rows,
            license_disclosure: "test-fixture".into(),
        }
    }

    #[test]
    fn official_constants_match_pinned_upstream_metadata() {
        let pin = PinnedGapArtifact::official_matbench_expt_gap();
        assert_eq!(pin.dataset_id, "matbench_expt_gap");
        assert_eq!(pin.expected_rows, 4_604);
        assert_eq!(pin.expected_sha256.len(), 64);
        assert_eq!(pin.expected_sha256, MATBENCH_EXPT_GAP_SHA256);
    }

    #[test]
    fn digest_mismatch_fails_before_gzip_decode() {
        let error = parse_official_matbench_expt_gap(b"not even gzip").unwrap_err();
        assert!(matches!(error, AdapterError::DigestMismatch { .. }));
    }

    #[test]
    fn split_artifact_accepts_pymatgen_composition_dicts_and_preserves_order() {
        let json = br#"{
            "columns":["composition","gap expt"],
            "index":[0,1],
            "data":[[{"Ga":1.0,"As":1.0},1.42],[{"Cd":1.0,"Te":1.0},1.44]]
        }"#;
        let bytes = gzip(json);
        let pin = fixture_pin(&bytes, 2);
        let dataset = parse_pinned_gap_artifact(&bytes, &pin).unwrap();

        assert_eq!(dataset.ordered_records.len(), 2);
        assert_eq!(dataset.ordered_records[0].experimental_gap_ev, 1.42);
        assert_eq!(dataset.ordered_records[1].experimental_gap_ev, 1.44);
        assert_eq!(dataset.compressed_sha256, sha256_hex(&bytes));
        assert!(!dataset.row_order_sha256.is_empty());
        assert_eq!(dataset.truth.provenance.slice.dataset_id, "fixture_gap");
    }

    #[test]
    fn formula_string_fallback_normalizes_whitespace() {
        let a = composition_fingerprint_from_json(&Value::String("GaAs".into())).unwrap();
        let b = composition_fingerprint_from_json(&Value::String("Ga As".into())).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn phase_label_training_overlap_is_detected_by_composition() {
        let json = br#"{
            "columns":["composition","gap expt"],
            "index":[0],
            "data":[[{"Si":1.0,"C":1.0},2.5]]
        }"#;
        let bytes = gzip(json);
        let dataset = parse_pinned_gap_artifact(&bytes, &fixture_pin(&bytes, 1)).unwrap();
        let overlap = dataset.symthaea_training_overlap().unwrap();

        assert_eq!(overlap.len(), 1);
        assert!(
            overlap[0]
                .symthaea_training_labels
                .iter()
                .any(|label| label.starts_with("SiC-"))
        );
        assert!(matches!(
            dataset.ensure_no_symthaea_training_overlap(),
            Err(AdapterError::TrainingCompositionOverlap(_))
        ));
    }

    #[test]
    fn equivalent_ratios_produce_identical_fingerprints() {
        let a = composition_fingerprint_from_json(&serde_json::json!({"Ga": 1.0, "As": 1.0})).unwrap();
        let b = composition_fingerprint_from_json(&serde_json::json!({"Ga": 2.0, "As": 2.0})).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn wrong_columns_fail_closed() {
        let json = br#"{
            "columns":["gap expt","composition"],
            "index":[0],
            "data":[[1.42,{"Ga":1.0,"As":1.0}]]
        }"#;
        let bytes = gzip(json);
        assert!(matches!(
            parse_pinned_gap_artifact(&bytes, &fixture_pin(&bytes, 1)),
            Err(AdapterError::InvalidShape(_))
        ));
    }

    #[test]
    fn duplicate_normalized_compositions_fail_closed() {
        let json = br#"{
            "columns":["composition","gap expt"],
            "index":[0,1],
            "data":[[{"Ga":1.0,"As":1.0},1.42],[{"Ga":2.0,"As":2.0},1.43]]
        }"#;
        let bytes = gzip(json);
        assert!(matches!(
            parse_pinned_gap_artifact(&bytes, &fixture_pin(&bytes, 2)),
            Err(AdapterError::DuplicateNormalizedComposition(_))
        ));
    }

    #[test]
    fn duplicate_indices_fail_closed() {
        let json = br#"{
            "columns":["composition","gap expt"],
            "index":[0,0],
            "data":[[{"Ga":1.0,"As":1.0},1.42],[{"Cd":1.0,"Te":1.0},1.44]]
        }"#;
        let bytes = gzip(json);
        assert!(matches!(
            parse_pinned_gap_artifact(&bytes, &fixture_pin(&bytes, 2)),
            Err(AdapterError::DuplicateIndex(_))
        ));
    }

    #[test]
    fn invalid_element_or_amount_fails_closed() {
        assert!(composition_fingerprint_from_json(&serde_json::json!({"Xx": 1.0})).is_err());
        assert!(composition_fingerprint_from_json(&serde_json::json!({"Ga": 0.0, "As": 1.0})).is_err());
    }

    #[test]
    fn row_order_identity_changes_when_rows_swap() {
        let first = gzip(
            br#"{"columns":["composition","gap expt"],"index":[0,1],"data":[[{"Ga":1.0,"As":1.0},1.0],[{"Cd":1.0,"Te":1.0},2.0]]}"#,
        );
        let second = gzip(
            br#"{"columns":["composition","gap expt"],"index":[0,1],"data":[[{"Cd":1.0,"Te":1.0},2.0],[{"Ga":1.0,"As":1.0},1.0]]}"#,
        );
        let first_dataset =
            parse_pinned_gap_artifact(&first, &fixture_pin(&first, 2)).unwrap();
        let second_dataset =
            parse_pinned_gap_artifact(&second, &fixture_pin(&second, 2)).unwrap();

        assert_ne!(first_dataset.row_order_sha256, second_dataset.row_order_sha256);
    }
}
