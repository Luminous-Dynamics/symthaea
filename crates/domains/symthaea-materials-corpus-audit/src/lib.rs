// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Historical-corpus contamination auditing for materials benchmarks.
//!
//! A held-out target is only a meaningful rediscovery target when the relevant
//! information was not already present in the pre-cutoff corpus. This crate
//! classifies targets before model scoring, distinguishing composition exposure,
//! structure exposure, property-label exposure, and unresolved identity.
//!
//! The audit never deletes contaminated targets. It records contamination as
//! benchmark evidence so downstream interpretation cannot improve by silently
//! filtering inconvenient cases.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use thiserror::Error;

/// Exact property-label identity used for contamination matching.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PropertyLabelRef {
    /// Stable scientific property identifier.
    pub property_id: String,
    /// Exact condition signature under which the property label applies.
    pub condition_signature: String,
}

impl PropertyLabelRef {
    fn validate(&self) -> Result<(), CorpusAuditError> {
        nonempty("property_id", &self.property_id)?;
        nonempty("condition_signature", &self.condition_signature)
    }
}

/// One material record present in the historical source corpus.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoricalCorpusRecord {
    /// Stable source-record identifier.
    pub record_id: String,
    /// Exact canonical composition identity, normally a SHA-256 of canonical composition bytes.
    pub composition_sha256: String,
    /// Exact canonical structure identity when known.
    pub structure_sha256: Option<String>,
    /// Property labels actually present for this exact source record.
    pub property_labels: Vec<PropertyLabelRef>,
}

impl HistoricalCorpusRecord {
    fn validate(&self) -> Result<(), CorpusAuditError> {
        nonempty("record_id", &self.record_id)?;
        sha256(&self.composition_sha256)?;
        if let Some(structure) = &self.structure_sha256 {
            sha256(structure)?;
        }
        unique_labels(&self.property_labels)
    }
}

/// One benchmark target to audit against the historical corpus.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoricalBenchmarkTarget {
    /// Stable benchmark target identifier.
    pub target_id: String,
    /// Exact canonical composition identity.
    pub composition_sha256: String,
    /// Exact canonical structure identity when the benchmark claims a structure-specific target.
    pub structure_sha256: Option<String>,
    /// Property labels the benchmark plans to score for this target.
    pub scored_property_labels: Vec<PropertyLabelRef>,
}

impl HistoricalBenchmarkTarget {
    fn validate(&self) -> Result<(), CorpusAuditError> {
        nonempty("target_id", &self.target_id)?;
        sha256(&self.composition_sha256)?;
        if let Some(structure) = &self.structure_sha256 {
            sha256(structure)?;
        }
        if self.scored_property_labels.is_empty() {
            return Err(CorpusAuditError::NoScoredProperties(self.target_id.clone()));
        }
        unique_labels(&self.scored_property_labels)
    }
}

/// Conservative contamination classification for one historical benchmark target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContaminationClass {
    /// No matching composition is present in the pre-cutoff corpus.
    AbsentFromPrecutoffCorpus,
    /// Composition exists, but the exact benchmark structure is not present.
    CompositionPresentStructureHeldOut,
    /// Exact structure exists, but none of the benchmark-scored property labels are present.
    StructurePresentLabelHeldOut,
    /// Exact structure and at least one benchmark-scored property label are already present.
    PropertyLabelPresent {
        /// Property labels already available in the pre-cutoff corpus.
        labels: Vec<PropertyLabelRef>,
    },
    /// Identity is not strong enough to distinguish structure exposure conservatively.
    AmbiguousIdentity {
        /// Human-readable machine-generated reason.
        reason: String,
    },
}

/// One audited target result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetContaminationResult {
    /// Benchmark target identifier.
    pub target_id: String,
    /// Contamination classification.
    pub class: ContaminationClass,
    /// Source records relevant to the classification.
    pub matching_record_ids: Vec<String>,
}

/// Complete deterministic audit over one immutable historical corpus snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusContaminationAudit {
    /// Audit schema version.
    pub schema_version: u32,
    /// Exact historical corpus snapshot or compact-extract digest.
    pub corpus_snapshot_sha256: String,
    /// Exact target-set artifact digest.
    pub target_set_sha256: String,
    /// Per-target results sorted by target ID.
    pub results: Vec<TargetContaminationResult>,
    /// Count by classification name for reporting.
    pub class_counts: BTreeMap<String, u32>,
}

impl CorpusContaminationAudit {
    /// Deterministic SHA-256 of the full contamination audit.
    pub fn audit_sha256(&self) -> Result<String, CorpusAuditError> {
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Audit a frozen target set against a frozen pre-cutoff corpus.
pub fn audit_historical_corpus(
    corpus_snapshot_sha256: &str,
    target_set_sha256: &str,
    corpus: &[HistoricalCorpusRecord],
    targets: &[HistoricalBenchmarkTarget],
) -> Result<CorpusContaminationAudit, CorpusAuditError> {
    sha256(corpus_snapshot_sha256)?;
    sha256(target_set_sha256)?;
    if corpus.is_empty() {
        return Err(CorpusAuditError::EmptyCorpus);
    }
    if targets.is_empty() {
        return Err(CorpusAuditError::EmptyTargetSet);
    }

    let mut record_ids = HashSet::new();
    for record in corpus {
        record.validate()?;
        if !record_ids.insert(record.record_id.as_str()) {
            return Err(CorpusAuditError::DuplicateRecordId(record.record_id.clone()));
        }
    }

    let mut target_ids = HashSet::new();
    for target in targets {
        target.validate()?;
        if !target_ids.insert(target.target_id.as_str()) {
            return Err(CorpusAuditError::DuplicateTargetId(target.target_id.clone()));
        }
    }

    let mut ordered_targets: Vec<&HistoricalBenchmarkTarget> = targets.iter().collect();
    ordered_targets.sort_by(|a, b| a.target_id.cmp(&b.target_id));

    let mut results = Vec::with_capacity(ordered_targets.len());
    let mut class_counts = BTreeMap::new();
    for target in ordered_targets {
        let result = classify_target(corpus, target)?;
        let class_name = class_name(&result.class).to_string();
        *class_counts.entry(class_name).or_insert(0) += 1;
        results.push(result);
    }

    Ok(CorpusContaminationAudit {
        schema_version: 1,
        corpus_snapshot_sha256: corpus_snapshot_sha256.to_ascii_lowercase(),
        target_set_sha256: target_set_sha256.to_ascii_lowercase(),
        results,
        class_counts,
    })
}

fn classify_target(
    corpus: &[HistoricalCorpusRecord],
    target: &HistoricalBenchmarkTarget,
) -> Result<TargetContaminationResult, CorpusAuditError> {
    let composition_matches: Vec<&HistoricalCorpusRecord> = corpus
        .iter()
        .filter(|record| record.composition_sha256.eq_ignore_ascii_case(&target.composition_sha256))
        .collect();

    if composition_matches.is_empty() {
        return Ok(TargetContaminationResult {
            target_id: target.target_id.clone(),
            class: ContaminationClass::AbsentFromPrecutoffCorpus,
            matching_record_ids: Vec::new(),
        });
    }

    let mut matching_ids: Vec<String> = composition_matches
        .iter()
        .map(|record| record.record_id.clone())
        .collect();
    matching_ids.sort();

    let Some(target_structure) = &target.structure_sha256 else {
        return Ok(TargetContaminationResult {
            target_id: target.target_id.clone(),
            class: ContaminationClass::AmbiguousIdentity {
                reason: "target composition is present but target structure identity is unspecified"
                    .to_string(),
            },
            matching_record_ids: matching_ids,
        });
    };

    if composition_matches
        .iter()
        .any(|record| record.structure_sha256.is_none())
    {
        return Ok(TargetContaminationResult {
            target_id: target.target_id.clone(),
            class: ContaminationClass::AmbiguousIdentity {
                reason: "pre-cutoff corpus contains matching composition with unresolved structure identity"
                    .to_string(),
            },
            matching_record_ids: matching_ids,
        });
    }

    let structure_matches: Vec<&HistoricalCorpusRecord> = composition_matches
        .iter()
        .copied()
        .filter(|record| {
            record
                .structure_sha256
                .as_ref()
                .is_some_and(|structure| structure.eq_ignore_ascii_case(target_structure))
        })
        .collect();

    if structure_matches.is_empty() {
        return Ok(TargetContaminationResult {
            target_id: target.target_id.clone(),
            class: ContaminationClass::CompositionPresentStructureHeldOut,
            matching_record_ids: matching_ids,
        });
    }

    let target_labels: BTreeSet<PropertyLabelRef> =
        target.scored_property_labels.iter().cloned().collect();
    let mut present_labels = BTreeSet::new();
    for record in &structure_matches {
        for label in &record.property_labels {
            if target_labels.contains(label) {
                present_labels.insert(label.clone());
            }
        }
    }

    let mut exact_structure_ids: Vec<String> = structure_matches
        .iter()
        .map(|record| record.record_id.clone())
        .collect();
    exact_structure_ids.sort();

    if present_labels.is_empty() {
        Ok(TargetContaminationResult {
            target_id: target.target_id.clone(),
            class: ContaminationClass::StructurePresentLabelHeldOut,
            matching_record_ids: exact_structure_ids,
        })
    } else {
        Ok(TargetContaminationResult {
            target_id: target.target_id.clone(),
            class: ContaminationClass::PropertyLabelPresent {
                labels: present_labels.into_iter().collect(),
            },
            matching_record_ids: exact_structure_ids,
        })
    }
}

fn class_name(class: &ContaminationClass) -> &'static str {
    match class {
        ContaminationClass::AbsentFromPrecutoffCorpus => "AbsentFromPrecutoffCorpus",
        ContaminationClass::CompositionPresentStructureHeldOut => {
            "CompositionPresentStructureHeldOut"
        }
        ContaminationClass::StructurePresentLabelHeldOut => "StructurePresentLabelHeldOut",
        ContaminationClass::PropertyLabelPresent { .. } => "PropertyLabelPresent",
        ContaminationClass::AmbiguousIdentity { .. } => "AmbiguousIdentity",
    }
}

fn unique_labels(labels: &[PropertyLabelRef]) -> Result<(), CorpusAuditError> {
    let mut seen = BTreeSet::new();
    for label in labels {
        label.validate()?;
        if !seen.insert(label.clone()) {
            return Err(CorpusAuditError::DuplicatePropertyLabel {
                property_id: label.property_id.clone(),
                condition_signature: label.condition_signature.clone(),
            });
        }
    }
    Ok(())
}

fn sha256(value: &str) -> Result<(), CorpusAuditError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Err(CorpusAuditError::InvalidSha256)
    } else {
        Ok(())
    }
}

fn nonempty(field: &'static str, value: &str) -> Result<(), CorpusAuditError> {
    if value.trim().is_empty() {
        Err(CorpusAuditError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Historical-corpus contamination audit failure.
#[derive(Debug, Error)]
pub enum CorpusAuditError {
    /// Required field empty.
    #[error("required corpus-audit field is empty: {0}")]
    EmptyField(&'static str),
    /// SHA-256 malformed.
    #[error("invalid corpus-audit SHA-256")]
    InvalidSha256,
    /// Historical corpus empty.
    #[error("historical corpus is empty")]
    EmptyCorpus,
    /// Benchmark target set empty.
    #[error("historical benchmark target set is empty")]
    EmptyTargetSet,
    /// Duplicate source-record identity.
    #[error("duplicate historical corpus record ID: {0}")]
    DuplicateRecordId(String),
    /// Duplicate benchmark target identity.
    #[error("duplicate historical benchmark target ID: {0}")]
    DuplicateTargetId(String),
    /// Target defines no property to score.
    #[error("historical benchmark target has no scored properties: {0}")]
    NoScoredProperties(String),
    /// Property label repeated inside one target/record.
    #[error("duplicate property label: {property_id} @ {condition_signature}")]
    DuplicatePropertyLabel {
        /// Property identifier.
        property_id: String,
        /// Exact condition signature.
        condition_signature: String,
    },
    /// JSON serialization failure.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    const A64: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B64: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C64: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const D64: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const E64: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn label(property: &str, condition: &str) -> PropertyLabelRef {
        PropertyLabelRef {
            property_id: property.to_string(),
            condition_signature: condition.to_string(),
        }
    }

    fn target(
        id: &str,
        composition: &str,
        structure: Option<&str>,
        labels: Vec<PropertyLabelRef>,
    ) -> HistoricalBenchmarkTarget {
        HistoricalBenchmarkTarget {
            target_id: id.to_string(),
            composition_sha256: composition.to_string(),
            structure_sha256: structure.map(ToString::to_string),
            scored_property_labels: labels,
        }
    }

    fn record(
        id: &str,
        composition: &str,
        structure: Option<&str>,
        labels: Vec<PropertyLabelRef>,
    ) -> HistoricalCorpusRecord {
        HistoricalCorpusRecord {
            record_id: id.to_string(),
            composition_sha256: composition.to_string(),
            structure_sha256: structure.map(ToString::to_string),
            property_labels: labels,
        }
    }

    #[test]
    fn absent_composition_is_cleanly_absent() {
        let corpus = vec![record("known", A64, Some(B64), vec![])];
        let target = target("candidate", C64, Some(D64), vec![label("k1", "0K|SOC")]);
        let result = classify_target(&corpus, &target).unwrap();
        assert_eq!(result.class, ContaminationClass::AbsentFromPrecutoffCorpus);
    }

    #[test]
    fn known_composition_but_new_structure_is_distinguished() {
        let corpus = vec![record("known", A64, Some(B64), vec![])];
        let target = target("candidate", A64, Some(C64), vec![label("k1", "0K|SOC")]);
        let result = classify_target(&corpus, &target).unwrap();
        assert_eq!(
            result.class,
            ContaminationClass::CompositionPresentStructureHeldOut
        );
    }

    #[test]
    fn exact_structure_without_scored_label_remains_label_holdout() {
        let corpus = vec![record(
            "known",
            A64,
            Some(B64),
            vec![label("formation_energy", "0K|PBE")],
        )];
        let target = target("candidate", A64, Some(B64), vec![label("k1", "0K|SOC")]);
        let result = classify_target(&corpus, &target).unwrap();
        assert_eq!(result.class, ContaminationClass::StructurePresentLabelHeldOut);
    }

    #[test]
    fn same_property_under_different_conditions_is_not_target_label_contamination() {
        let corpus = vec![record(
            "known",
            A64,
            Some(B64),
            vec![label("magnetization", "300K|experiment")],
        )];
        let target = target(
            "candidate",
            A64,
            Some(B64),
            vec![label("magnetization", "0K|spin-DFT")],
        );
        let result = classify_target(&corpus, &target).unwrap();
        assert_eq!(result.class, ContaminationClass::StructurePresentLabelHeldOut);
    }

    #[test]
    fn exact_structure_and_scored_label_is_property_contaminated() {
        let wanted = label("k1", "0K|SOC");
        let corpus = vec![record("known", A64, Some(B64), vec![wanted.clone()])];
        let target = target("candidate", A64, Some(B64), vec![wanted.clone()]);
        let result = classify_target(&corpus, &target).unwrap();
        assert_eq!(
            result.class,
            ContaminationClass::PropertyLabelPresent {
                labels: vec![wanted]
            }
        );
    }

    #[test]
    fn unresolved_structure_identity_fails_conservatively_ambiguous() {
        let corpus = vec![record("known", A64, None, vec![label("k1", "0K|SOC")])];
        let target = target("candidate", A64, Some(B64), vec![label("k1", "0K|SOC")]);
        let result = classify_target(&corpus, &target).unwrap();
        assert!(matches!(result.class, ContaminationClass::AmbiguousIdentity { .. }));
    }

    #[test]
    fn composition_only_target_is_ambiguous_when_composition_is_already_known() {
        let corpus = vec![record("known", A64, Some(B64), vec![])];
        let target = target("candidate", A64, None, vec![label("formation_energy", "0K|PBE")]);
        let result = classify_target(&corpus, &target).unwrap();
        assert!(matches!(result.class, ContaminationClass::AmbiguousIdentity { .. }));
    }

    #[test]
    fn audit_preserves_every_target_and_has_stable_class_counts() {
        let corpus = vec![
            record("known-a", A64, Some(B64), vec![label("k1", "0K|SOC")]),
            record("known-c", C64, Some(D64), vec![]),
        ];
        let targets = vec![
            target("z-property", A64, Some(B64), vec![label("k1", "0K|SOC")]),
            target("a-absent", E64, Some(B64), vec![label("k1", "0K|SOC")]),
            target("m-structure", C64, Some(B64), vec![label("k1", "0K|SOC")]),
        ];
        let audit = audit_historical_corpus(A64, B64, &corpus, &targets).unwrap();
        assert_eq!(audit.results.len(), 3);
        assert_eq!(audit.results[0].target_id, "a-absent");
        assert_eq!(audit.results[1].target_id, "m-structure");
        assert_eq!(audit.results[2].target_id, "z-property");
        assert_eq!(audit.class_counts["AbsentFromPrecutoffCorpus"], 1);
        assert_eq!(audit.class_counts["CompositionPresentStructureHeldOut"], 1);
        assert_eq!(audit.class_counts["PropertyLabelPresent"], 1);
        assert_eq!(audit.audit_sha256().unwrap().len(), 64);
    }

    #[test]
    fn corpus_snapshot_identity_is_part_of_audit_identity() {
        let corpus = vec![record("known", A64, Some(B64), vec![])];
        let targets = vec![target(
            "candidate",
            C64,
            Some(D64),
            vec![label("formation_energy", "0K|PBE")],
        )];
        let a = audit_historical_corpus(A64, B64, &corpus, &targets).unwrap();
        let b = audit_historical_corpus(C64, B64, &corpus, &targets).unwrap();
        assert_ne!(a.audit_sha256().unwrap(), b.audit_sha256().unwrap());
    }
}
