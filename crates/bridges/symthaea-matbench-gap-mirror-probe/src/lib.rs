// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exploratory overlap analysis for third-party mirrors of Matbench experimental-gap data.
//!
//! This crate is intentionally incapable of producing `BandgapTruthSet`.
//! It exists only to answer a narrower question before official-artifact qualification:
//! which normalized compositions in a supplied mirror also occur in Symthaea's exposed
//! band-gap training table?

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use symthaea_bandgap::periodic_table::atomic_number;
use symthaea_matbench_gap::CompositionFingerprint;
use thiserror::Error;

pub const CAPABILITY_CLASSIFICATION: &str =
    "EXPLORATORY THIRD-PARTY MATBENCH MIRROR OVERLAP ONLY -- this receipt is not official Matbench truth, not a Benchmark Zero truth slice, not a clean holdout certificate, and not scientific validation.";

pub const KNOWN_MIRROR_REPOSITORY: &str = "Zhang-NJ-Lab/Datasets";
pub const KNOWN_MIRROR_PATH: &str = "matbench_expt_gap.csv";
pub const KNOWN_MIRROR_OBSERVED_GIT_BLOB_SHA1: &str =
    "35943edef66ae36412d3f184c71869941eb57087";
pub const KNOWN_MIRROR_EXPECTED_ROWS: usize = 4_604;

const FRACTION_SCALE: f64 = 1_000_000_000.0;
const ROW_ORDER_DIGEST_DOMAIN: &[u8] = b"symthaea.matbench-gap.mirror-row-order.v0\0";
const RECEIPT_DIGEST_DOMAIN: &[u8] = b"symthaea.matbench-gap.mirror-overlap-receipt.v0\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExploratoryMirrorSource {
    pub repository: String,
    pub path: String,
    /// Locator metadata observed from the hosting Git service. The probe does
    /// not derive Git's SHA-1 from the supplied bytes; `csv_sha256` below is the
    /// locally derived byte identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_git_blob_sha1: Option<String>,
}

impl ExploratoryMirrorSource {
    pub fn known_github_mirror() -> Self {
        Self {
            repository: KNOWN_MIRROR_REPOSITORY.into(),
            path: KNOWN_MIRROR_PATH.into(),
            observed_git_blob_sha1: Some(KNOWN_MIRROR_OBSERVED_GIT_BLOB_SHA1.into()),
        }
    }

    fn validate(&self) -> Result<(), ProbeError> {
        if self.repository.trim().is_empty() || self.path.trim().is_empty() {
            return Err(ProbeError::InvalidSource(
                "repository and path must be non-empty".into(),
            ));
        }
        if let Some(sha1) = &self.observed_git_blob_sha1 {
            if sha1.len() != 40 || !sha1.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(ProbeError::InvalidSource(
                    "observed Git blob SHA-1 must be exactly 40 hexadecimal characters".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirrorOccurrence {
    pub row_index: usize,
    pub formula: String,
    pub experimental_gap_ev: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExploratoryTrainingOverlap {
    pub composition: CompositionFingerprint,
    pub mirror_occurrences: Vec<MirrorOccurrence>,
    pub symthaea_training_labels: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExploratoryMirrorOverlapReceipt {
    pub schema: String,
    pub capability_classification: String,
    pub source: ExploratoryMirrorSource,
    /// SHA-256 derived directly from the caller-supplied CSV bytes.
    pub csv_sha256: String,
    /// Domain-separated digest of ordered normalized row identities.
    pub row_order_sha256: String,
    pub row_count: usize,
    pub unique_composition_count: usize,
    pub overlap_row_count: usize,
    pub overlap_composition_count: usize,
    pub overlaps: Vec<ExploratoryTrainingOverlap>,
    pub source_identity_disclosure: String,
    pub holdout_disclosure: String,
}

impl ExploratoryMirrorOverlapReceipt {
    pub fn validate(&self) -> Result<(), ProbeError> {
        if self.schema != "symthaea.matbench-gap.mirror-overlap-receipt.v0"
            || self.capability_classification != CAPABILITY_CLASSIFICATION
        {
            return Err(ProbeError::InvalidReceipt(
                "receipt schema/capability classification was altered".into(),
            ));
        }
        self.source.validate()?;
        validate_sha256(&self.csv_sha256, "CSV SHA-256")?;
        validate_sha256(&self.row_order_sha256, "row-order SHA-256")?;
        if self.row_count == 0 || self.unique_composition_count == 0 {
            return Err(ProbeError::InvalidReceipt(
                "row and unique-composition counts must be positive".into(),
            ));
        }
        if self.unique_composition_count > self.row_count {
            return Err(ProbeError::InvalidReceipt(
                "unique-composition count cannot exceed row count".into(),
            ));
        }
        if self.overlap_composition_count != self.overlaps.len() {
            return Err(ProbeError::InvalidReceipt(
                "overlap-composition count differs from overlap records".into(),
            ));
        }
        if self.overlap_composition_count > self.unique_composition_count
            || self.overlap_row_count > self.row_count
        {
            return Err(ProbeError::InvalidReceipt(
                "overlap counts exceed source counts".into(),
            ));
        }

        let mut previous_composition: Option<&CompositionFingerprint> = None;
        let mut observed_rows = BTreeSet::new();
        let mut row_total = 0usize;
        for overlap in &self.overlaps {
            if overlap.mirror_occurrences.is_empty()
                || overlap.symthaea_training_labels.is_empty()
            {
                return Err(ProbeError::InvalidReceipt(
                    "overlap records require mirror occurrences and training labels".into(),
                ));
            }
            if previous_composition.is_some_and(|previous| previous >= &overlap.composition) {
                return Err(ProbeError::InvalidReceipt(
                    "overlap records must be in canonical composition order".into(),
                ));
            }
            previous_composition = Some(&overlap.composition);

            let mut previous_row = None;
            for occurrence in &overlap.mirror_occurrences {
                if occurrence.row_index >= self.row_count
                    || !occurrence.experimental_gap_ev.is_finite()
                    || occurrence.experimental_gap_ev < 0.0
                    || occurrence.formula.trim().is_empty()
                {
                    return Err(ProbeError::InvalidReceipt(
                        "overlap occurrence is outside validated source semantics".into(),
                    ));
                }
                if previous_row.is_some_and(|previous| occurrence.row_index <= previous) {
                    return Err(ProbeError::InvalidReceipt(
                        "mirror occurrences must be in increasing row order".into(),
                    ));
                }
                previous_row = Some(occurrence.row_index);
                if !observed_rows.insert(occurrence.row_index) {
                    return Err(ProbeError::InvalidReceipt(
                        "one mirror row appears in multiple overlap compositions".into(),
                    ));
                }
                row_total += 1;
            }

            if overlap
                .symthaea_training_labels
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            {
                return Err(ProbeError::InvalidReceipt(
                    "training labels must be sorted and unique".into(),
                ));
            }
        }
        if row_total != self.overlap_row_count {
            return Err(ProbeError::InvalidReceipt(
                "overlap-row count differs from overlap occurrences".into(),
            ));
        }
        if self.source_identity_disclosure.trim().is_empty()
            || self.holdout_disclosure.trim().is_empty()
        {
            return Err(ProbeError::InvalidReceipt(
                "authority disclosures cannot be empty".into(),
            ));
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, ProbeError> {
        self.validate()?;
        domain_separated_sha256(RECEIPT_DIGEST_DOMAIN, self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct OrderedMirrorRecord {
    row_index: usize,
    formula: String,
    composition: CompositionFingerprint,
    experimental_gap_ev: f64,
}

/// Probe the currently observed GitHub mirror. This is deliberately an
/// exploratory receipt, even when all 4,604 rows parse successfully.
pub fn probe_known_github_mirror(
    csv_bytes: &[u8],
) -> Result<ExploratoryMirrorOverlapReceipt, ProbeError> {
    probe_exploratory_mirror_csv(
        csv_bytes,
        ExploratoryMirrorSource::known_github_mirror(),
        KNOWN_MIRROR_EXPECTED_ROWS,
    )
}

/// Parse a caller-supplied third-party CSV mirror and compare normalized
/// compositions with Symthaea's exposed band-gap training table.
///
/// This API never returns Benchmark Zero truth. A successful receipt means only
/// that the supplied bytes were structurally parsed and overlap was measured.
pub fn probe_exploratory_mirror_csv(
    csv_bytes: &[u8],
    source: ExploratoryMirrorSource,
    expected_rows: usize,
) -> Result<ExploratoryMirrorOverlapReceipt, ProbeError> {
    source.validate()?;
    if expected_rows == 0 {
        return Err(ProbeError::InvalidSource(
            "expected row count must be positive".into(),
        ));
    }
    let text = std::str::from_utf8(csv_bytes)
        .map_err(|error| ProbeError::InvalidCsv(format!("CSV must be UTF-8: {error}")))?;
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| ProbeError::InvalidCsv("CSV is empty".into()))?
        .trim_end_matches('\r');
    if header != ",composition,gap expt" {
        return Err(ProbeError::InvalidCsv(format!(
            "unexpected header {header:?}; expected \",composition,gap expt\""
        )));
    }

    let mut records = Vec::with_capacity(expected_rows);
    let mut by_composition: BTreeMap<CompositionFingerprint, Vec<MirrorOccurrence>> =
        BTreeMap::new();

    for (expected_index, raw_line) in lines.enumerate() {
        let line = raw_line.trim_end_matches('\r');
        if line.is_empty() {
            return Err(ProbeError::InvalidCsv(format!(
                "blank data row at logical index {expected_index}"
            )));
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 3 {
            return Err(ProbeError::InvalidCsv(format!(
                "row {expected_index} has {} comma-separated fields; expected 3",
                fields.len()
            )));
        }
        let row_index = fields[0].parse::<usize>().map_err(|error| {
            ProbeError::InvalidCsv(format!("row index {:?} is invalid: {error}", fields[0]))
        })?;
        if row_index != expected_index {
            return Err(ProbeError::InvalidCsv(format!(
                "row-order/index mismatch: expected {expected_index}, found {row_index}"
            )));
        }
        let formula = fields[1].trim();
        if formula.is_empty() {
            return Err(ProbeError::InvalidCsv(format!(
                "row {row_index} composition is empty"
            )));
        }
        let gap = fields[2].parse::<f64>().map_err(|error| {
            ProbeError::InvalidCsv(format!(
                "row {row_index} gap {:?} is invalid: {error}",
                fields[2]
            ))
        })?;
        if !gap.is_finite() || gap < 0.0 {
            return Err(ProbeError::InvalidCsv(format!(
                "row {row_index} gap must be finite and non-negative"
            )));
        }

        let composition = mirror_formula_fingerprint(formula).map_err(|error| {
            ProbeError::InvalidCsv(format!("row {row_index} formula {formula:?}: {error}"))
        })?;
        let occurrence = MirrorOccurrence {
            row_index,
            formula: formula.into(),
            experimental_gap_ev: gap,
        };
        by_composition
            .entry(composition.clone())
            .or_default()
            .push(occurrence);
        records.push(OrderedMirrorRecord {
            row_index,
            formula: formula.into(),
            composition,
            experimental_gap_ev: gap,
        });
    }

    if records.len() != expected_rows {
        return Err(ProbeError::RowCountMismatch {
            expected: expected_rows,
            actual: records.len(),
        });
    }

    let training = symthaea_training_index()?;
    let mut overlaps = Vec::new();
    for (composition, occurrences) in &by_composition {
        if let Some(labels) = training.get(composition) {
            overlaps.push(ExploratoryTrainingOverlap {
                composition: composition.clone(),
                mirror_occurrences: occurrences.clone(),
                symthaea_training_labels: labels.clone(),
            });
        }
    }
    let overlap_row_count = overlaps
        .iter()
        .map(|overlap| overlap.mirror_occurrences.len())
        .sum();

    let receipt = ExploratoryMirrorOverlapReceipt {
        schema: "symthaea.matbench-gap.mirror-overlap-receipt.v0".into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        source,
        csv_sha256: sha256_hex(csv_bytes),
        row_order_sha256: domain_separated_sha256(ROW_ORDER_DIGEST_DOMAIN, &records)?,
        row_count: records.len(),
        unique_composition_count: by_composition.len(),
        overlap_row_count,
        overlap_composition_count: overlaps.len(),
        overlaps,
        source_identity_disclosure: "csv_sha256 is derived from the supplied bytes. observed_git_blob_sha1 is locator metadata reported by the Git host and is not independently recomputed by this crate.".into(),
        holdout_disclosure: "An exploratory third-party mirror overlap result cannot establish official Matbench artifact identity, canonical row equivalence, licensing, a clean holdout, or Benchmark Zero truth authority.".into(),
    };
    receipt.validate()?;
    Ok(receipt)
}

/// Parse the decimal/parenthesized composition grammar used by the CSV mirror
/// and return the same public fingerprint type used by `symthaea-matbench-gap`.
pub fn mirror_formula_fingerprint(formula: &str) -> Result<CompositionFingerprint, ProbeError> {
    let compact: String = formula
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    if compact.is_empty() {
        return Err(ProbeError::InvalidFormula("formula cannot be empty".into()));
    }
    if !compact.is_ascii() {
        return Err(ProbeError::InvalidFormula(
            "mirror formula grammar is ASCII-only".into(),
        ));
    }
    let mut parser = FormulaParser::new(compact.as_bytes());
    let amounts = parser.parse_sequence(false)?;
    if parser.pos != parser.bytes.len() {
        return Err(ProbeError::InvalidFormula(format!(
            "unexpected trailing input at byte {}",
            parser.pos
        )));
    }
    fingerprint_from_amounts(amounts)
}

fn symthaea_training_index(
) -> Result<BTreeMap<CompositionFingerprint, Vec<String>>, ProbeError> {
    let mut index: BTreeMap<CompositionFingerprint, Vec<String>> = BTreeMap::new();
    for entry in symthaea_bandgap::training_data::load_training_data() {
        let fingerprint = fingerprint_from_atomic_fractions(&entry.composition)?;
        index.entry(fingerprint).or_default().push(entry.formula.into());
    }
    for labels in index.values_mut() {
        labels.sort();
        labels.dedup();
    }
    Ok(index)
}

fn fingerprint_from_atomic_fractions(
    composition: &[(u8, f64)],
) -> Result<CompositionFingerprint, ProbeError> {
    let mut amounts = BTreeMap::new();
    for &(atomic_number, amount) in composition {
        *amounts.entry(atomic_number).or_insert(0.0) += amount;
    }
    fingerprint_from_amounts(amounts)
}

fn fingerprint_from_amounts(
    amounts: BTreeMap<u8, f64>,
) -> Result<CompositionFingerprint, ProbeError> {
    if amounts.is_empty() {
        return Err(ProbeError::InvalidFormula(
            "composition cannot be empty".into(),
        ));
    }
    let total: f64 = amounts.values().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(ProbeError::InvalidFormula(
            "composition total must be finite and positive".into(),
        ));
    }
    let mut fingerprint = Vec::with_capacity(amounts.len());
    for (atomic_number, amount) in amounts {
        if !amount.is_finite() || amount <= 0.0 {
            return Err(ProbeError::InvalidFormula(format!(
                "element Z={atomic_number} has invalid amount {amount}"
            )));
        }
        let quantized = ((amount / total) * FRACTION_SCALE).round();
        if !quantized.is_finite() || quantized <= 0.0 || quantized > u64::MAX as f64 {
            return Err(ProbeError::InvalidFormula(format!(
                "element Z={atomic_number} cannot be represented in fingerprint space"
            )));
        }
        fingerprint.push((atomic_number, quantized as u64));
    }
    Ok(CompositionFingerprint(fingerprint))
}

struct FormulaParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> FormulaParser<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn parse_sequence(&mut self, stop_at_close: bool) -> Result<BTreeMap<u8, f64>, ProbeError> {
        let mut amounts = BTreeMap::new();
        let mut terms = 0usize;
        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b')' if stop_at_close => break,
                b')' => {
                    return Err(ProbeError::InvalidFormula(format!(
                        "unexpected ')' at byte {}",
                        self.pos
                    )));
                }
                b'(' => {
                    self.pos += 1;
                    let group = self.parse_sequence(true)?;
                    if self.pos >= self.bytes.len() || self.bytes[self.pos] != b')' {
                        return Err(ProbeError::InvalidFormula(
                            "unterminated parenthesized group".into(),
                        ));
                    }
                    self.pos += 1;
                    let multiplier = self.parse_number_or_one()?;
                    for (atomic_number, amount) in group {
                        *amounts.entry(atomic_number).or_insert(0.0) += amount * multiplier;
                    }
                    terms += 1;
                }
                byte if byte.is_ascii_uppercase() => {
                    let atomic_number = self.parse_element()?;
                    let amount = self.parse_number_or_one()?;
                    *amounts.entry(atomic_number).or_insert(0.0) += amount;
                    terms += 1;
                }
                byte => {
                    return Err(ProbeError::InvalidFormula(format!(
                        "unexpected byte {:?} at position {}",
                        char::from(byte),
                        self.pos
                    )));
                }
            }
        }
        if stop_at_close && (self.pos >= self.bytes.len() || self.bytes[self.pos] != b')') {
            return Err(ProbeError::InvalidFormula(
                "unterminated parenthesized group".into(),
            ));
        }
        if terms == 0 {
            return Err(ProbeError::InvalidFormula(
                "formula/group cannot be empty".into(),
            ));
        }
        Ok(amounts)
    }

    fn parse_element(&mut self) -> Result<u8, ProbeError> {
        let start = self.pos;
        self.pos += 1;
        if self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_lowercase() {
            self.pos += 1;
        }
        let symbol = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|error| ProbeError::InvalidFormula(error.to_string()))?;
        atomic_number(symbol).ok_or_else(|| {
            ProbeError::InvalidFormula(format!("unknown element symbol {symbol:?}"))
        })
    }

    fn parse_number_or_one(&mut self) -> Result<f64, ProbeError> {
        if self.pos >= self.bytes.len()
            || (!self.bytes[self.pos].is_ascii_digit() && self.bytes[self.pos] != b'.')
        {
            return Ok(1.0);
        }
        let start = self.pos;
        let mut digits_before = 0usize;
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
            self.pos += 1;
            digits_before += 1;
        }
        let mut digits_after = 0usize;
        if self.pos < self.bytes.len() && self.bytes[self.pos] == b'.' {
            self.pos += 1;
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                self.pos += 1;
                digits_after += 1;
            }
        }
        if digits_before == 0 && digits_after == 0 {
            return Err(ProbeError::InvalidFormula(format!(
                "invalid numeric amount at byte {start}"
            )));
        }
        let raw = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|error| ProbeError::InvalidFormula(error.to_string()))?;
        let value = raw.parse::<f64>().map_err(|error| {
            ProbeError::InvalidFormula(format!("invalid amount {raw:?}: {error}"))
        })?;
        if !value.is_finite() || value <= 0.0 {
            return Err(ProbeError::InvalidFormula(format!(
                "amount must be finite and positive, got {raw:?}"
            )));
        }
        Ok(value)
    }
}

fn validate_sha256(value: &str, name: &str) -> Result<(), ProbeError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ProbeError::InvalidReceipt(format!(
            "{name} must be exactly 64 hexadecimal characters"
        )));
    }
    Ok(())
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
) -> Result<String, ProbeError> {
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
pub enum ProbeError {
    #[error("invalid exploratory mirror source: {0}")]
    InvalidSource(String),
    #[error("invalid exploratory mirror CSV: {0}")]
    InvalidCsv(String),
    #[error("invalid mirror formula: {0}")]
    InvalidFormula(String),
    #[error("mirror row count mismatch: expected {expected}, got {actual}")]
    RowCountMismatch { expected: usize, actual: usize },
    #[error("invalid exploratory overlap receipt: {0}")]
    InvalidReceipt(String),
    #[error("JSON encode/decode failed: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp(formula: &str) -> CompositionFingerprint {
        mirror_formula_fingerprint(formula).unwrap()
    }

    #[test]
    fn decimal_stoichiometry_normalizes_equivalent_ratios() {
        assert_eq!(fp("Ag0.5Ge1Pb1.75S4"), fp("Ag2Ge4Pb7S16"));
    }

    #[test]
    fn parenthesized_groups_expand_with_multiplier() {
        assert_eq!(fp("Ag(W3Br7)2"), fp("AgW6Br14"));
    }

    #[test]
    fn nested_parentheses_are_supported() {
        assert_eq!(fp("Ca((OH)2)2"), fp("CaO4H4"));
    }

    #[test]
    fn adjacent_uppercase_symbols_are_not_collapsed() {
        assert_eq!(fp("Ag7(SI)2"), fp("Ag7S2I2"));
        assert_ne!(fp("Ag7(SI)2"), fp("Ag7Si2"));
    }

    #[test]
    fn phase_insensitive_fingerprint_matches_training_identity() {
        let target = fp("SiC");
        let training = symthaea_training_index().unwrap();
        let labels = training.get(&target).unwrap();
        assert!(labels.iter().any(|label| label == "SiC-3C"));
        assert!(labels.iter().any(|label| label == "SiC-4H"));
        assert!(labels.iter().any(|label| label == "SiC-6H"));
    }

    #[test]
    fn small_csv_aggregates_duplicate_normalized_compositions() {
        let csv = b",composition,gap expt\n0,SiC,2.4\n1,Si2C2,2.5\n2,GaAs,1.42\n";
        let receipt = probe_exploratory_mirror_csv(
            csv,
            ExploratoryMirrorSource {
                repository: "fixture/repo".into(),
                path: "fixture.csv".into(),
                observed_git_blob_sha1: None,
            },
            3,
        )
        .unwrap();
        assert_eq!(receipt.row_count, 3);
        assert_eq!(receipt.unique_composition_count, 2);
        assert_eq!(receipt.overlap_row_count, 3);
        assert_eq!(receipt.overlap_composition_count, 2);
        receipt.validate().unwrap();
    }

    #[test]
    fn discontinuous_indices_fail_closed() {
        let csv = b",composition,gap expt\n0,SiC,2.4\n2,GaAs,1.42\n";
        assert!(matches!(
            probe_exploratory_mirror_csv(
                csv,
                ExploratoryMirrorSource {
                    repository: "fixture/repo".into(),
                    path: "fixture.csv".into(),
                    observed_git_blob_sha1: None,
                },
                2,
            ),
            Err(ProbeError::InvalidCsv(_))
        ));
    }

    #[test]
    fn official_json_adapter_and_mirror_probe_remain_type_separated() {
        let receipt = ExploratoryMirrorOverlapReceipt {
            schema: "symthaea.matbench-gap.mirror-overlap-receipt.v0".into(),
            capability_classification: CAPABILITY_CLASSIFICATION.into(),
            source: ExploratoryMirrorSource::known_github_mirror(),
            csv_sha256: "a".repeat(64),
            row_order_sha256: "b".repeat(64),
            row_count: 1,
            unique_composition_count: 1,
            overlap_row_count: 0,
            overlap_composition_count: 0,
            overlaps: vec![],
            source_identity_disclosure: "fixture".into(),
            holdout_disclosure: "fixture".into(),
        };
        receipt.validate().unwrap();
        // The type exposes no `truth` field and no conversion to BandgapTruthSet.
        assert_eq!(receipt.capability_classification, CAPABILITY_CLASSIFICATION);
    }
}
