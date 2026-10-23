// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Frozen dataset/split manifest and leakage checks for safety benchmarks.
//!
//! This module never fetches or redistributes source data. It records provenance
//! needed to freeze an eligible dataset snapshot and validates split assignments
//! before a benchmark run. A passing structural check is not a data license,
//! source-truth verification, or scientific model validation.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const MANIFEST_SCHEMA_VERSION: &str = "1.0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetReuseStatus {
    VerifiedPublicDomain,
    VerifiedPermitted,
    AttributionRequired,
    ReferenceOnly,
    Unverified,
    Prohibited,
}

impl DatasetReuseStatus {
    fn permits_frozen_benchmark(&self) -> bool {
        matches!(
            self,
            Self::VerifiedPublicDomain | Self::VerifiedPermitted | Self::AttributionRequired
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitStrategy {
    Random,
    ChemicalScaffold,
    GroupedBySource,
    TemporalHoldout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataPartition {
    Train,
    Validation,
    Test,
}

/// Record-level split assignment. The grouping keys are precomputed with
/// versioned methods and should never be guessed from display names.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestRecord {
    pub record_id: String,
    /// Stable canonical identity grouping key; exact-compound records stay in one split.
    pub compound_identity_key: String,
    /// Required for ChemicalScaffold strategy.
    pub scaffold_group_key: Option<String>,
    /// Required for GroupedBySource strategy (for example source paper or assay series).
    pub source_group_key: Option<String>,
    /// ISO-8601 date of publication/source release, required for temporal holdout.
    pub source_date: Option<String>,
}

/// Immutable, source-linked dataset snapshot and explicit split assignments.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PharmacologyDatasetManifest {
    pub schema_version: String,
    pub dataset_id: String,
    pub dataset_version: String,
    pub source_uri: String,
    pub acquired_on: String,
    pub reuse_status: DatasetReuseStatus,
    pub reuse_terms_reference: String,
    pub source_content_sha256: String,
    pub label_definition: String,
    pub label_definition_sha256: String,
    pub split_strategy: SplitStrategy,
    pub split_seed: Option<u64>,
    /// Required for TemporalHoldout. Train/validation dates must be <= cutoff;
    /// test dates must be strictly after cutoff.
    pub temporal_cutoff: Option<String>,
    pub grouping_method_version: String,
    pub train: Vec<ManifestRecord>,
    pub validation: Vec<ManifestRecord>,
    pub test: Vec<ManifestRecord>,
    /// BLAKE3 digest over the version and sorted split assignments.
    pub split_assignment_digest_blake3: String,
}

impl PharmacologyDatasetManifest {
    /// Compute a stable digest over split assignment content, independent of
    /// vector ordering. This digest binds declared assignments, not source truth.
    pub fn compute_split_assignment_digest(&self) -> Result<String, String> {
        let mut train = self.train.clone();
        let mut validation = self.validation.clone();
        let mut test = self.test.clone();
        train.sort_by(|a, b| a.record_id.cmp(&b.record_id));
        validation.sort_by(|a, b| a.record_id.cmp(&b.record_id));
        test.sort_by(|a, b| a.record_id.cmp(&b.record_id));
        let bytes = serde_json::to_vec(&(
            self.schema_version.as_str(),
            self.dataset_id.as_str(),
            self.dataset_version.as_str(),
            self.split_strategy,
            self.split_seed,
            self.temporal_cutoff.as_deref(),
            self.grouping_method_version.as_str(),
            train,
            validation,
            test,
        ))
        .map_err(|error| format!("serialize split assignments: {error}"))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }

    /// Validate manifest provenance, reuse eligibility, and split leakage rules.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        require_equal(
            &self.schema_version,
            MANIFEST_SCHEMA_VERSION,
            "schema_version",
            &mut errors,
        );
        for (field, value) in [
            ("dataset_id", self.dataset_id.as_str()),
            ("dataset_version", self.dataset_version.as_str()),
            ("source_uri", self.source_uri.as_str()),
            ("reuse_terms_reference", self.reuse_terms_reference.as_str()),
            ("label_definition", self.label_definition.as_str()),
            ("grouping_method_version", self.grouping_method_version.as_str()),
        ] {
            require_nonempty(value, field, &mut errors);
        }
        require_date(&self.acquired_on, "acquired_on", &mut errors);
        require_digest(&self.source_content_sha256, "source_content_sha256", &mut errors);
        require_digest(
            &self.label_definition_sha256,
            "label_definition_sha256",
            &mut errors,
        );
        require_digest(
            &self.split_assignment_digest_blake3,
            "split_assignment_digest_blake3",
            &mut errors,
        );
        if !self.reuse_status.permits_frozen_benchmark() {
            errors.push(format!(
                "reuse_status {:?} does not permit inclusion in a frozen benchmark dataset",
                self.reuse_status
            ));
        }

        if self.train.is_empty() || self.validation.is_empty() || self.test.is_empty() {
            errors.push("train, validation, and test partitions must all be non-empty".into());
        }
        if self.split_strategy == SplitStrategy::TemporalHoldout {
            match self.temporal_cutoff.as_deref() {
                Some(cutoff) if chrono::NaiveDate::parse_from_str(cutoff, "%Y-%m-%d").is_ok() => {}
                _ => errors.push("temporal holdout requires a valid YYYY-MM-DD temporal_cutoff".into()),
            }
        } else if self.temporal_cutoff.is_some() {
            errors.push("temporal_cutoff must only be set for TemporalHoldout strategy".into());
        }
        if self.split_strategy == SplitStrategy::Random && self.split_seed.is_none() {
            errors.push("random split must record split_seed".into());
        }

        let partitions = [
            (DataPartition::Train, self.train.as_slice()),
            (DataPartition::Validation, self.validation.as_slice()),
            (DataPartition::Test, self.test.as_slice()),
        ];
        let mut id_partition: BTreeMap<&str, DataPartition> = BTreeMap::new();
        let mut compound_partition: BTreeMap<&str, DataPartition> = BTreeMap::new();
        let mut scaffold_partition: BTreeMap<&str, DataPartition> = BTreeMap::new();
        let mut source_partition: BTreeMap<&str, DataPartition> = BTreeMap::new();

        for (partition, records) in partitions {
            for record in records {
                for (field, value) in [
                    ("record_id", record.record_id.as_str()),
                    ("compound_identity_key", record.compound_identity_key.as_str()),
                ] {
                    require_nonempty(value, field, &mut errors);
                }

                if let Some(previous) = id_partition.insert(&record.record_id, partition) {
                    errors.push(format!(
                        "record_id {:?} occurs more than once (previous partition {:?}, current {:?})",
                        record.record_id, previous, partition
                    ));
                }

                if let Some(previous) =
                    compound_partition.insert(&record.compound_identity_key, partition)
                {
                    if previous != partition {
                        errors.push(format!(
                            "compound identity leakage for key {:?}: {:?} and {:?}",
                            record.compound_identity_key, previous, partition
                        ));
                    }
                }

                if self.split_strategy == SplitStrategy::ChemicalScaffold {
                    match record.scaffold_group_key.as_deref() {
                        Some(key) if !key.trim().is_empty() => {
                            if let Some(previous) = scaffold_partition.insert(key, partition) {
                                if previous != partition {
                                    errors.push(format!(
                                        "scaffold leakage for key {:?}: {:?} and {:?}",
                                        key, previous, partition
                                    ));
                                }
                            }
                        }
                        _ => errors.push(format!(
                            "record {:?} requires scaffold_group_key for scaffold split",
                            record.record_id
                        )),
                    }
                }

                if self.split_strategy == SplitStrategy::GroupedBySource {
                    match record.source_group_key.as_deref() {
                        Some(key) if !key.trim().is_empty() => {
                            if let Some(previous) = source_partition.insert(key, partition) {
                                if previous != partition {
                                    errors.push(format!(
                                        "source-group leakage for key {:?}: {:?} and {:?}",
                                        key, previous, partition
                                    ));
                                }
                            }
                        }
                        _ => errors.push(format!(
                            "record {:?} requires source_group_key for grouped-by-source split",
                            record.record_id
                        )),
                    }
                }

                if self.split_strategy == SplitStrategy::TemporalHoldout {
                    match (self.temporal_cutoff.as_deref(), record.source_date.as_deref()) {
                        (Some(cutoff), Some(date)) => {
                            let cutoff = chrono::NaiveDate::parse_from_str(cutoff, "%Y-%m-%d");
                            let date = chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d");
                            match (cutoff, date) {
                                (Ok(cutoff), Ok(date)) => {
                                    if partition == DataPartition::Test && date <= cutoff {
                                        errors.push(format!(
                                            "temporal test record {:?} has source date {} not after cutoff",
                                            record.record_id, date
                                        ));
                                    }
                                    if partition != DataPartition::Test && date > cutoff {
                                        errors.push(format!(
                                            "temporal {:?} record {:?} has source date {} after cutoff",
                                            partition, record.record_id, date
                                        ));
                                    }
                                }
                                _ => errors.push(format!(
                                    "record {:?} must have a valid YYYY-MM-DD source_date for temporal holdout",
                                    record.record_id
                                )),
                            }
                        }
                        _ => errors.push(format!(
                            "record {:?} requires source_date for temporal holdout",
                            record.record_id
                        )),
                    }
                }
            }
        }

        match self.compute_split_assignment_digest() {
            Ok(computed) if computed == self.split_assignment_digest_blake3 => {}
            Ok(_) => errors.push("split_assignment_digest_blake3 does not match current split assignments".into()),
            Err(error) => errors.push(error),
        }

        if errors.is_empty() {
            Ok(())
        } else {
            // Identical findings in deterministic order improve reproducibility.
            errors.sort();
            errors.dedup();
            Err(errors)
        }
    }
}

fn require_equal(actual: &str, expected: &str, field: &str, errors: &mut Vec<String>) {
    if actual != expected {
        errors.push(format!("{field} must equal {expected:?}"));
    }
}

fn require_nonempty(value: &str, field: &str, errors: &mut Vec<String>) {
    if value.trim().is_empty() {
        errors.push(format!("{field} must not be empty"));
    }
}

fn require_date(value: &str, field: &str, errors: &mut Vec<String>) {
    if chrono::NaiveDate::parse_from_str(value, "%Y-%m-%d").is_err() {
        errors.push(format!("{field} must be a valid YYYY-MM-DD date"));
    }
}

fn is_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn require_digest(value: &str, field: &str, errors: &mut Vec<String>) {
    if !is_digest(value) {
        errors.push(format!("{field} must be a 64-character hexadecimal digest"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: &str, compound: &str, scaffold: &str, source: &str, date: &str) -> ManifestRecord {
        ManifestRecord {
            record_id: id.to_string(),
            compound_identity_key: compound.to_string(),
            scaffold_group_key: Some(scaffold.to_string()),
            source_group_key: Some(source.to_string()),
            source_date: Some(date.to_string()),
        }
    }

    fn manifest() -> PharmacologyDatasetManifest {
        let mut value = PharmacologyDatasetManifest {
            schema_version: MANIFEST_SCHEMA_VERSION.to_string(),
            dataset_id: "public-admet-pilot".to_string(),
            dataset_version: "frozen-example-1".to_string(),
            source_uri: "https://example.org/dataset".to_string(),
            acquired_on: "2026-10-20".to_string(),
            reuse_status: DatasetReuseStatus::VerifiedPermitted,
            reuse_terms_reference: "https://example.org/license".to_string(),
            source_content_sha256: "a".repeat(64),
            label_definition: "positive is the prespecified hazard label".to_string(),
            label_definition_sha256: "b".repeat(64),
            split_strategy: SplitStrategy::ChemicalScaffold,
            split_seed: Some(42),
            temporal_cutoff: None,
            grouping_method_version: "scaffold-v1".to_string(),
            train: vec![
                record("r1", "compound-1", "scaffold-1", "paper-1", "2020-01-01"),
                record("r2", "compound-2", "scaffold-2", "paper-2", "2021-01-01"),
            ],
            validation: vec![record("r3", "compound-3", "scaffold-3", "paper-3", "2022-01-01")],
            test: vec![record("r4", "compound-4", "scaffold-4", "paper-4", "2023-01-01")],
            split_assignment_digest_blake3: String::new(),
        };
        value.split_assignment_digest_blake3 = value
            .compute_split_assignment_digest()
            .expect("calculate test manifest digest");
        value
    }

    #[test]
    fn valid_scaffold_manifest_passes_and_has_digest() {
        let value = manifest();
        assert!(value.validate().is_ok());
        assert_eq!(value.split_assignment_digest_blake3.len(), 64);
    }

    #[test]
    fn split_digest_is_order_independent_but_content_sensitive() {
        let mut first = manifest();
        let mut reordered = first.clone();
        reordered.train.reverse();
        assert_eq!(
            first.compute_split_assignment_digest().expect("digest"),
            reordered.compute_split_assignment_digest().expect("digest")
        );
        reordered.train[0].compound_identity_key = "changed-compound".to_string();
        assert_ne!(
            first.compute_split_assignment_digest().expect("digest"),
            reordered.compute_split_assignment_digest().expect("digest")
        );
    }

    #[test]
    fn exact_compound_and_scaffold_cross_split_leakage_are_rejected() {
        let mut value = manifest();
        value.test[0].compound_identity_key = value.train[0].compound_identity_key.clone();
        value.test[0].scaffold_group_key = value.train[0].scaffold_group_key.clone();
        value.split_assignment_digest_blake3 = value.compute_split_assignment_digest().expect("digest");
        let errors = value.validate().expect_err("leakage must fail");
        assert!(errors.iter().any(|e| e.contains("compound identity leakage")));
        assert!(errors.iter().any(|e| e.contains("scaffold leakage")));
    }

    #[test]
    fn duplicate_records_and_stale_assignment_digest_fail() {
        let mut value = manifest();
        value.validation.push(value.train[0].clone());
        let errors = value.validate().expect_err("duplicated record and digest mismatch must fail");
        assert!(errors.iter().any(|e| e.contains("occurs more than once")));
        assert!(errors.iter().any(|e| e.contains("does not match current split assignments")));
    }

    #[test]
    fn temporal_holdout_enforces_date_cutoff() {
        let mut value = manifest();
        value.split_strategy = SplitStrategy::TemporalHoldout;
        value.temporal_cutoff = Some("2022-12-31".to_string());
        value.train[0].source_date = Some("2020-01-01".to_string());
        value.train[1].source_date = Some("2021-01-01".to_string());
        value.validation[0].source_date = Some("2022-01-01".to_string());
        value.test[0].source_date = Some("2023-01-01".to_string());
        value.split_assignment_digest_blake3 = value.compute_split_assignment_digest().expect("digest");
        assert!(value.validate().is_ok());

        value.test[0].source_date = Some("2022-03-01".to_string());
        value.split_assignment_digest_blake3 = value.compute_split_assignment_digest().expect("digest");
        assert!(value.validate().expect_err("test set before cutoff must fail").iter().any(|e| e.contains("not after cutoff")));
    }

    #[test]
    fn unverified_reuse_rights_fail_closed() {
        let mut value = manifest();
        value.reuse_status = DatasetReuseStatus::Unverified;
        value.split_assignment_digest_blake3 = value.compute_split_assignment_digest().expect("digest");
        assert!(value.validate().expect_err("unknown reuse rights must fail").iter().any(|e| e.contains("does not permit inclusion")));
    }

    #[test]
    fn grouped_by_source_rejects_source_group_leakage() {
        let mut value = manifest();
        value.split_strategy = SplitStrategy::GroupedBySource;
        value.test[0].source_group_key = value.train[0].source_group_key.clone();
        value.split_assignment_digest_blake3 = value.compute_split_assignment_digest().expect("digest");
        assert!(value.validate().expect_err("source leakage must fail").iter().any(|e| e.contains("source-group leakage")));
    }
}
