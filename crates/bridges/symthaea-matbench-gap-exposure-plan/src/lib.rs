// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic exposure planning for the pinned Matbench gap artifact.
//!
//! This crate deliberately does **not** produce a benchmark truth slice. It
//! partitions the exact SHA-gated official composition universe into retained
//! versus exposed-training exclusions without copying experimental gap values.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use symthaea_matbench_gap::{
    parse_official_matbench_expt_gap, CompositionFingerprint, MatbenchGapDataset,
    TrainingOverlap,
};
use thiserror::Error;

pub const PLAN_SCHEMA: &str = "symthaea.matbench-gap.exposed-training-exclusion-plan.v0";
pub const CAPABILITY_CLASSIFICATION: &str =
    "TRUTH-FREE SCREENING-UNIVERSE EXPOSURE PLAN ONLY -- not a clean holdout certificate, not Benchmark Zero truth, and not a scientific result.";
pub const PRIOR_KNOWLEDGE_DISCLOSURE: &str =
    "This plan excludes only compositions present in Symthaea's currently exposed band-gap training table. It does not establish absence of historical/manual prior knowledge, hand-tuned baseline knowledge, external pretraining exposure, publication familiarity, or other leakage channels.";

const TRAINING_SNAPSHOT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.matbench-gap.exposed-training-snapshot.v0\0";
const COMPOSITION_UNIVERSE_DIGEST_DOMAIN: &[u8] =
    b"symthaea.matbench-gap.composition-universe-order.v0\0";
const PARTITION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.matbench-gap.exposure-partition.v0\0";
const RETAINED_UNIVERSE_DIGEST_DOMAIN: &[u8] =
    b"symthaea.matbench-gap.retained-screening-universe.v0\0";
const EXCLUSION_SET_DIGEST_DOMAIN: &[u8] =
    b"symthaea.matbench-gap.exposed-training-exclusion-set.v0\0";
const PLAN_DIGEST_DOMAIN: &[u8] = b"symthaea.matbench-gap.exposed-training-exclusion-plan.v0\0";

/// Why one official composition row is or is not available to screening.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ExposureDisposition {
    /// No exact normalized-composition collision was found in the exposed
    /// Symthaea band-gap training table.
    Retained,
    /// Exact normalized composition already occurs in exposed training data.
    ExcludedExposedTraining {
        symthaea_training_labels: Vec<String>,
    },
}

/// One truth-free row in the exact official composition order.
///
/// Experimental gap values are intentionally absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedComposition {
    pub source_row_index: usize,
    pub candidate_id: String,
    pub composition: CompositionFingerprint,
    pub disposition: ExposureDisposition,
}

/// Content-addressed partition of the official Matbench composition universe.
///
/// A screening implementation can consume [`Self::retained_rows`] without
/// opening `BandgapTruthSet` merely to discover which candidate identities are
/// admissible. Truth remains a separate later evaluation input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExposedTrainingExclusionPlan {
    pub schema: String,
    pub capability_classification: String,
    pub source_dataset_id: String,
    pub source_content_digest: String,
    pub source_compressed_sha256: String,
    /// Parent-adapter identity of complete normalized rows. This includes the
    /// experimental values belonging to the exact artifact.
    pub source_row_order_sha256: String,
    /// Identity of ordered `(candidate_id, composition)` pairs only.
    pub source_composition_order_sha256: String,
    /// Content identity of the exact currently exposed Symthaea training table
    /// used when this partition was derived.
    pub symthaea_training_snapshot_sha256: String,
    pub source_row_count: usize,
    pub excluded_composition_count: usize,
    pub retained_composition_count: usize,
    /// Complete source-row partition in exact canonical source order.
    pub rows: Vec<PlannedComposition>,
    /// Domain-separated identity of the complete retain/exclude partition.
    pub partition_sha256: String,
    /// Truth-free identity of retained source-row/candidate/composition tuples.
    pub retained_universe_sha256: String,
    /// Identity of excluded rows including the exposed-training labels that
    /// caused each exclusion.
    pub exclusion_set_sha256: String,
    pub prior_knowledge_disclosure: String,
}

impl ExposedTrainingExclusionPlan {
    /// Validate only the serialized plan's internal consistency.
    ///
    /// This does not authenticate the remote Matbench source and does not prove
    /// that the named training snapshot is still current. Use replay and the
    /// explicit current-training check for those separate theorems.
    pub fn validate(&self) -> Result<(), PlanError> {
        if self.schema != PLAN_SCHEMA || self.capability_classification != CAPABILITY_CLASSIFICATION {
            return Err(PlanError::InvalidPlan(
                "schema or capability classification was altered".into(),
            ));
        }
        if self.source_dataset_id.trim().is_empty() || self.source_content_digest.trim().is_empty() {
            return Err(PlanError::InvalidPlan(
                "source dataset identity must be non-empty".into(),
            ));
        }
        for (name, value) in [
            ("source compressed SHA-256", self.source_compressed_sha256.as_str()),
            ("source row-order SHA-256", self.source_row_order_sha256.as_str()),
            (
                "source composition-order SHA-256",
                self.source_composition_order_sha256.as_str(),
            ),
            (
                "Symthaea training snapshot SHA-256",
                self.symthaea_training_snapshot_sha256.as_str(),
            ),
            ("partition SHA-256", self.partition_sha256.as_str()),
            (
                "retained-universe SHA-256",
                self.retained_universe_sha256.as_str(),
            ),
            ("exclusion-set SHA-256", self.exclusion_set_sha256.as_str()),
        ] {
            validate_sha256(value, name)?;
        }

        let expected_content_digest = format!("sha256:{}", self.source_compressed_sha256);
        if self.source_content_digest != expected_content_digest {
            return Err(PlanError::InvalidPlan(
                "source content digest does not bind the compressed artifact SHA-256".into(),
            ));
        }
        if self.source_row_count == 0 || self.rows.len() != self.source_row_count {
            return Err(PlanError::InvalidPlan(
                "complete partition must contain exactly source_row_count rows".into(),
            ));
        }
        if self
            .excluded_composition_count
            .checked_add(self.retained_composition_count)
            != Some(self.source_row_count)
        {
            return Err(PlanError::InvalidPlan(
                "excluded + retained counts must equal source row count without overflow".into(),
            ));
        }
        if self.prior_knowledge_disclosure != PRIOR_KNOWLEDGE_DISCLOSURE {
            return Err(PlanError::InvalidPlan(
                "prior-knowledge disclosure was altered".into(),
            ));
        }

        let mut candidate_ids = BTreeSet::new();
        let mut retained = 0usize;
        let mut excluded = 0usize;
        for (expected_row_index, row) in self.rows.iter().enumerate() {
            if row.source_row_index != expected_row_index {
                return Err(PlanError::InvalidPlan(
                    "partition rows must cover every source row exactly once in canonical order"
                        .into(),
                ));
            }
            if row.candidate_id.trim().is_empty() || !candidate_ids.insert(row.candidate_id.as_str()) {
                return Err(PlanError::InvalidPlan(
                    "partition candidate ids must be non-empty and unique".into(),
                ));
            }
            validate_composition(&row.composition)?;
            match &row.disposition {
                ExposureDisposition::Retained => retained += 1,
                ExposureDisposition::ExcludedExposedTraining {
                    symthaea_training_labels,
                } => {
                    excluded += 1;
                    validate_training_labels(symthaea_training_labels)?;
                }
            }
        }
        if retained != self.retained_composition_count
            || excluded != self.excluded_composition_count
        {
            return Err(PlanError::InvalidPlan(
                "serialized retain/exclude counts differ from the complete partition".into(),
            ));
        }

        let expected_partition = domain_separated_sha256(PARTITION_DIGEST_DOMAIN, &self.rows)?;
        if self.partition_sha256 != expected_partition {
            return Err(PlanError::InvalidPlan(
                "partition digest does not match partition rows".into(),
            ));
        }
        let expected_retained = retained_universe_digest(&self.rows)?;
        if self.retained_universe_sha256 != expected_retained {
            return Err(PlanError::InvalidPlan(
                "retained-universe digest does not match retained rows".into(),
            ));
        }
        let expected_exclusions = exclusion_set_digest(&self.rows)?;
        if self.exclusion_set_sha256 != expected_exclusions {
            return Err(PlanError::InvalidPlan(
                "exclusion-set digest does not match excluded rows".into(),
            ));
        }
        Ok(())
    }

    /// Require that this plan names the exact exposed training table in the
    /// current build. This is deliberately separate from structural validation.
    pub fn validate_against_current_training_snapshot(&self) -> Result<(), PlanError> {
        self.validate()?;
        let current = current_training_snapshot_sha256()?;
        if self.symthaea_training_snapshot_sha256 != current {
            return Err(PlanError::TrainingSnapshotMismatch {
                expected: self.symthaea_training_snapshot_sha256.clone(),
                current,
            });
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, PlanError> {
        self.validate()?;
        domain_separated_sha256(PLAN_DIGEST_DOMAIN, self)
    }

    /// Complete truth-free candidate surface admitted for later screening.
    pub fn retained_rows(&self) -> impl Iterator<Item = &PlannedComposition> {
        self.rows
            .iter()
            .filter(|row| matches!(&row.disposition, ExposureDisposition::Retained))
    }

    pub fn retained_candidate_ids(&self) -> impl Iterator<Item = &str> {
        self.retained_rows().map(|row| row.candidate_id.as_str())
    }

    pub fn excluded_rows(&self) -> impl Iterator<Item = &PlannedComposition> {
        self.rows.iter().filter(|row| {
            matches!(
                &row.disposition,
                ExposureDisposition::ExcludedExposedTraining { .. }
            )
        })
    }
}

#[derive(Serialize)]
struct CompositionUniverseRecord<'a> {
    source_row_index: usize,
    candidate_id: &'a str,
    composition: &'a CompositionFingerprint,
}

#[derive(Serialize)]
struct ExclusionDigestRecord<'a> {
    source_row_index: usize,
    candidate_id: &'a str,
    composition: &'a CompositionFingerprint,
    symthaea_training_labels: &'a [String],
}

/// Parse the exact current official Matbench artifact through the SHA-gated
/// parent adapter and derive the complete truth-free retain/exclude partition.
pub fn plan_official_matbench_exposed_training_exclusions(
    compressed_bytes: &[u8],
) -> Result<ExposedTrainingExclusionPlan, PlanError> {
    let dataset = parse_official_matbench_expt_gap(compressed_bytes)?;
    plan_from_parsed_dataset(&dataset)
}

/// Replay the official parser and planner from exact compressed bytes and
/// require both the same current training snapshot and exact plan equality.
pub fn verify_official_exposure_plan(
    compressed_bytes: &[u8],
    expected: &ExposedTrainingExclusionPlan,
) -> Result<(), PlanError> {
    expected.validate_against_current_training_snapshot()?;
    let observed = plan_official_matbench_exposed_training_exclusions(compressed_bytes)?;
    if &observed != expected {
        return Err(PlanError::ReplayMismatch);
    }
    Ok(())
}

fn plan_from_parsed_dataset(
    dataset: &MatbenchGapDataset,
) -> Result<ExposedTrainingExclusionPlan, PlanError> {
    dataset.truth.validate()?;
    validate_sha256(&dataset.compressed_sha256, "dataset compressed SHA-256")?;
    validate_sha256(&dataset.row_order_sha256, "dataset row-order SHA-256")?;
    if dataset.ordered_records.is_empty() {
        return Err(PlanError::InvalidDataset(
            "parsed dataset contains no ordered records".into(),
        ));
    }

    let truth_keys: BTreeSet<&str> = dataset
        .truth
        .experimental_gap_ev
        .keys()
        .map(String::as_str)
        .collect();
    let ordered_keys: BTreeSet<&str> = dataset
        .ordered_records
        .iter()
        .map(|record| record.candidate_id.as_str())
        .collect();
    if truth_keys != ordered_keys || truth_keys.len() != dataset.ordered_records.len() {
        return Err(PlanError::InvalidDataset(
            "truth candidate set and ordered-record candidate set differ".into(),
        ));
    }

    let expected_content_digest = format!("sha256:{}", dataset.compressed_sha256);
    if dataset.truth.provenance.content_digest != expected_content_digest {
        return Err(PlanError::InvalidDataset(
            "truth provenance does not bind the parsed compressed artifact".into(),
        ));
    }

    let overlaps = dataset.symthaea_training_overlap()?;
    let overlap_by_candidate = overlap_index(&overlaps)?;
    let mut rows = Vec::with_capacity(dataset.ordered_records.len());
    for (source_row_index, record) in dataset.ordered_records.iter().enumerate() {
        let disposition = if let Some(labels) = overlap_by_candidate.get(record.candidate_id.as_str()) {
            ExposureDisposition::ExcludedExposedTraining {
                symthaea_training_labels: (*labels).to_vec(),
            }
        } else {
            ExposureDisposition::Retained
        };
        rows.push(PlannedComposition {
            source_row_index,
            candidate_id: record.candidate_id.clone(),
            composition: record.composition.clone(),
            disposition,
        });
    }

    let excluded_composition_count = rows
        .iter()
        .filter(|row| {
            matches!(
                &row.disposition,
                ExposureDisposition::ExcludedExposedTraining { .. }
            )
        })
        .count();
    if excluded_composition_count != overlaps.len() {
        return Err(PlanError::InvalidDataset(
            "training-overlap records did not map one-to-one onto source rows".into(),
        ));
    }
    let source_row_count = rows.len();
    let retained_composition_count = source_row_count
        .checked_sub(excluded_composition_count)
        .ok_or_else(|| PlanError::InvalidDataset("exclusion count exceeds row count".into()))?;

    let source_composition_order_sha256 = composition_universe_digest(&rows)?;
    let symthaea_training_snapshot_sha256 = current_training_snapshot_sha256()?;
    let partition_sha256 = domain_separated_sha256(PARTITION_DIGEST_DOMAIN, &rows)?;
    let retained_universe_sha256 = retained_universe_digest(&rows)?;
    let exclusion_set_sha256 = exclusion_set_digest(&rows)?;

    let plan = ExposedTrainingExclusionPlan {
        schema: PLAN_SCHEMA.into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        source_dataset_id: dataset.truth.provenance.slice.dataset_id.clone(),
        source_content_digest: dataset.truth.provenance.content_digest.clone(),
        source_compressed_sha256: dataset.compressed_sha256.clone(),
        source_row_order_sha256: dataset.row_order_sha256.clone(),
        source_composition_order_sha256,
        symthaea_training_snapshot_sha256,
        source_row_count,
        excluded_composition_count,
        retained_composition_count,
        rows,
        partition_sha256,
        retained_universe_sha256,
        exclusion_set_sha256,
        prior_knowledge_disclosure: PRIOR_KNOWLEDGE_DISCLOSURE.into(),
    };
    plan.validate_against_current_training_snapshot()?;
    Ok(plan)
}

fn overlap_index<'a>(
    overlaps: &'a [TrainingOverlap],
) -> Result<BTreeMap<&'a str, &'a [String]>, PlanError> {
    let mut index = BTreeMap::new();
    for overlap in overlaps {
        if overlap.candidate_id.trim().is_empty() {
            return Err(PlanError::InvalidDataset(
                "training-overlap record has an empty candidate id".into(),
            ));
        }
        validate_training_labels(&overlap.symthaea_training_labels)?;
        if index
            .insert(
                overlap.candidate_id.as_str(),
                overlap.symthaea_training_labels.as_slice(),
            )
            .is_some()
        {
            return Err(PlanError::InvalidDataset(
                "duplicate candidate id in training-overlap records".into(),
            ));
        }
    }
    Ok(index)
}

fn current_training_snapshot_sha256() -> Result<String, PlanError> {
    let training = symthaea_bandgap::training_data::load_training_data();
    if training.is_empty() {
        return Err(PlanError::InvalidDataset(
            "Symthaea band-gap training table is empty".into(),
        ));
    }
    domain_separated_sha256(TRAINING_SNAPSHOT_DIGEST_DOMAIN, &training)
}

fn composition_universe_digest(rows: &[PlannedComposition]) -> Result<String, PlanError> {
    let universe: Vec<_> = rows
        .iter()
        .map(|row| CompositionUniverseRecord {
            source_row_index: row.source_row_index,
            candidate_id: row.candidate_id.as_str(),
            composition: &row.composition,
        })
        .collect();
    domain_separated_sha256(COMPOSITION_UNIVERSE_DIGEST_DOMAIN, &universe)
}

fn retained_universe_digest(rows: &[PlannedComposition]) -> Result<String, PlanError> {
    let retained: Vec<_> = rows
        .iter()
        .filter(|row| matches!(&row.disposition, ExposureDisposition::Retained))
        .map(|row| CompositionUniverseRecord {
            source_row_index: row.source_row_index,
            candidate_id: row.candidate_id.as_str(),
            composition: &row.composition,
        })
        .collect();
    domain_separated_sha256(RETAINED_UNIVERSE_DIGEST_DOMAIN, &retained)
}

fn exclusion_set_digest(rows: &[PlannedComposition]) -> Result<String, PlanError> {
    let mut excluded = Vec::new();
    for row in rows {
        if let ExposureDisposition::ExcludedExposedTraining {
            symthaea_training_labels,
        } = &row.disposition
        {
            excluded.push(ExclusionDigestRecord {
                source_row_index: row.source_row_index,
                candidate_id: row.candidate_id.as_str(),
                composition: &row.composition,
                symthaea_training_labels,
            });
        }
    }
    domain_separated_sha256(EXCLUSION_SET_DIGEST_DOMAIN, &excluded)
}

fn validate_training_labels(labels: &[String]) -> Result<(), PlanError> {
    if labels.is_empty()
        || labels.iter().any(|label| label.trim().is_empty())
        || labels.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(PlanError::InvalidPlan(
            "training labels must be non-empty, non-blank, sorted, and unique".into(),
        ));
    }
    Ok(())
}

fn validate_composition(composition: &CompositionFingerprint) -> Result<(), PlanError> {
    if composition.0.is_empty() {
        return Err(PlanError::InvalidPlan(
            "composition fingerprint cannot be empty".into(),
        ));
    }
    let mut previous_atomic_number = None;
    for &(atomic_number, fraction) in &composition.0 {
        if atomic_number == 0 || atomic_number > 118 || fraction == 0 {
            return Err(PlanError::InvalidPlan(
                "composition contains an invalid atomic number or zero fraction".into(),
            ));
        }
        if previous_atomic_number.is_some_and(|previous| atomic_number <= previous) {
            return Err(PlanError::InvalidPlan(
                "composition fingerprint must use strictly increasing atomic-number order".into(),
            ));
        }
        previous_atomic_number = Some(atomic_number);
    }
    Ok(())
}

fn validate_sha256(value: &str, name: &str) -> Result<(), PlanError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(PlanError::InvalidPlan(format!(
            "{name} must be exactly 64 hexadecimal characters"
        )));
    }
    Ok(())
}

fn domain_separated_sha256<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<String, PlanError> {
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
pub enum PlanError {
    #[error("official Matbench parser rejected artifact: {0}")]
    Matbench(#[from] symthaea_matbench_gap::AdapterError),
    #[error("benchmark truth provenance rejected parsed dataset: {0}")]
    Benchmark(#[from] symthaea_energy_benchmark_zero::BenchmarkError),
    #[error("invalid parsed dataset for exposure planning: {0}")]
    InvalidDataset(String),
    #[error("invalid exposure plan: {0}")]
    InvalidPlan(String),
    #[error("training snapshot changed: plan {expected}, current build {current}")]
    TrainingSnapshotMismatch { expected: String, current: String },
    #[error("exposure plan replay differs from the expected plan")]
    ReplayMismatch,
    #[error("JSON encode/decode failed: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    use symthaea_matbench_gap::{parse_pinned_gap_artifact, PinnedGapArtifact};

    fn fixture_sha256(bytes: &[u8]) -> String {
        let digest = Sha256::digest(bytes);
        let mut output = String::with_capacity(64);
        for byte in digest {
            write!(&mut output, "{byte:02x}").unwrap();
        }
        output
    }

    fn fixture_dataset_with_gaps(si_c_gap: f64, ga_as_gap: f64, h_gap: f64) -> MatbenchGapDataset {
        let json = format!(
            r#"{{
                "columns":["composition","gap expt"],
                "index":[0,1,2],
                "data":[
                    [{{"Si":1.0,"C":1.0}},{si_c_gap}],
                    [{{"Ga":1.0,"As":1.0}},{ga_as_gap}],
                    [{{"H":1.0}},{h_gap}]
                ]
            }}"#
        );
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(json.as_bytes()).unwrap();
        let bytes = encoder.finish().unwrap();
        let pin = PinnedGapArtifact {
            dataset_id: "fixture_gap".into(),
            source_uri: "https://example.invalid/fixture.json.gz".into(),
            expected_sha256: fixture_sha256(&bytes),
            expected_rows: 3,
            license_disclosure: "test-fixture".into(),
        };
        parse_pinned_gap_artifact(&bytes, &pin).unwrap()
    }

    fn fixture_dataset() -> MatbenchGapDataset {
        fixture_dataset_with_gaps(2.5, 1.42, 0.0)
    }

    #[test]
    fn plan_partitions_every_row_without_copying_truth_labels() {
        let plan = plan_from_parsed_dataset(&fixture_dataset()).unwrap();
        assert_eq!(plan.source_row_count, 3);
        assert_eq!(plan.rows.len(), 3);
        assert_eq!(plan.excluded_composition_count, 2);
        assert_eq!(plan.retained_composition_count, 1);
        assert_eq!(plan.rows[0].source_row_index, 0);
        assert_eq!(plan.rows[1].source_row_index, 1);
        assert_eq!(plan.rows[2].source_row_index, 2);

        match &plan.rows[0].disposition {
            ExposureDisposition::ExcludedExposedTraining {
                symthaea_training_labels,
            } => assert!(symthaea_training_labels
                .iter()
                .any(|label| label.starts_with("SiC-"))),
            ExposureDisposition::Retained => panic!("SiC must be excluded"),
        }
        match &plan.rows[1].disposition {
            ExposureDisposition::ExcludedExposedTraining {
                symthaea_training_labels,
            } => assert!(symthaea_training_labels.iter().any(|label| label == "GaAs")),
            ExposureDisposition::Retained => panic!("GaAs must be excluded"),
        }
        assert!(matches!(
            &plan.rows[2].disposition,
            ExposureDisposition::Retained
        ));
        assert_eq!(plan.retained_rows().count(), 1);
        assert_eq!(plan.retained_candidate_ids().count(), 1);

        let encoded = serde_json::to_string(&plan).unwrap();
        assert!(!encoded.contains("experimental_gap_ev"));
        assert!(!encoded.contains("2.5"));
        assert!(!encoded.contains("1.42"));
        plan.validate_against_current_training_snapshot().unwrap();
    }

    #[test]
    fn exclusion_decision_and_screening_universe_are_invariant_to_truth_values() {
        let first = plan_from_parsed_dataset(&fixture_dataset_with_gaps(2.5, 1.42, 0.0)).unwrap();
        let second =
            plan_from_parsed_dataset(&fixture_dataset_with_gaps(9.9, 0.01, 7.7)).unwrap();

        assert_eq!(first.rows, second.rows);
        assert_eq!(
            first.source_composition_order_sha256,
            second.source_composition_order_sha256
        );
        assert_eq!(
            first.symthaea_training_snapshot_sha256,
            second.symthaea_training_snapshot_sha256
        );
        assert_eq!(first.partition_sha256, second.partition_sha256);
        assert_eq!(first.retained_universe_sha256, second.retained_universe_sha256);
        assert_eq!(first.exclusion_set_sha256, second.exclusion_set_sha256);
        assert_ne!(first.source_compressed_sha256, second.source_compressed_sha256);
        assert_ne!(first.source_row_order_sha256, second.source_row_order_sha256);
    }

    #[test]
    fn incomplete_or_reordered_partition_fails_closed() {
        let mut plan = plan_from_parsed_dataset(&fixture_dataset()).unwrap();
        plan.rows.swap(0, 1);
        assert!(matches!(plan.validate(), Err(PlanError::InvalidPlan(_))));
    }

    #[test]
    fn partition_and_exclusion_digests_detect_tampering() {
        let mut plan = plan_from_parsed_dataset(&fixture_dataset()).unwrap();
        match &mut plan.rows[0].disposition {
            ExposureDisposition::ExcludedExposedTraining {
                symthaea_training_labels,
            } => symthaea_training_labels[0] = "forged-label".into(),
            ExposureDisposition::Retained => panic!("fixture row must be excluded"),
        }
        assert!(plan.validate().is_err());
    }

    #[test]
    fn stale_training_snapshot_is_distinct_from_structural_failure() {
        let mut plan = plan_from_parsed_dataset(&fixture_dataset()).unwrap();
        plan.symthaea_training_snapshot_sha256 = "a".repeat(64);
        assert!(plan.validate().is_ok());
        assert!(matches!(
            plan.validate_against_current_training_snapshot(),
            Err(PlanError::TrainingSnapshotMismatch { .. })
        ));
    }

    #[test]
    fn plan_digest_is_stable_for_same_dataset() {
        let dataset = fixture_dataset();
        let first = plan_from_parsed_dataset(&dataset).unwrap();
        let second = plan_from_parsed_dataset(&dataset).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.sha256().unwrap(), second.sha256().unwrap());
    }

    #[test]
    fn fixed_disclosures_cannot_be_weakened() {
        let mut plan = plan_from_parsed_dataset(&fixture_dataset()).unwrap();
        plan.prior_knowledge_disclosure = "clean holdout".into();
        assert!(matches!(plan.validate(), Err(PlanError::InvalidPlan(_))));
    }

    #[test]
    fn count_validation_is_overflow_safe() {
        let mut plan = plan_from_parsed_dataset(&fixture_dataset()).unwrap();
        plan.excluded_composition_count = usize::MAX;
        plan.retained_composition_count = 1;
        assert!(matches!(plan.validate(), Err(PlanError::InvalidPlan(_))));
    }

    #[test]
    fn replay_verifier_rejects_nonofficial_bytes_before_plan_comparison() {
        let plan = plan_from_parsed_dataset(&fixture_dataset()).unwrap();
        let error = verify_official_exposure_plan(b"not official gzip", &plan).unwrap_err();
        assert!(matches!(error, PlanError::Matbench(_)));
    }
}
