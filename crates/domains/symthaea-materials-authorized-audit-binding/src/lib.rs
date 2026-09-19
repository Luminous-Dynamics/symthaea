// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bind publication-authorized targets to the exact historical contamination audit.
//!
//! This is the convergence theorem between the historical-data lineage and the
//! publication-disclosure lineage. It proves that the exact labels authorized by
//! captured source evidence are the same labels audited against the exact pre-cutoff
//! corpus used by the verified historical run.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use symthaea_materials_corpus_audit::{ContaminationClass, CorpusContaminationAudit};
use symthaea_materials_history_tool::{
    VerifiedHistoricalRun, canonical_target_set_sha256,
};
use symthaea_materials_target_set::AuthorizedHistoricalTargetSet;
use thiserror::Error;

/// Verified convergence between authorized publication targets and historical audit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorizedAuditBinding {
    /// Binding schema version.
    pub schema_version: u32,
    /// Exact verified historical-run identity.
    pub verified_run_sha256: String,
    /// Exact historical input/evidence bundle identity.
    pub historical_bundle_sha256: String,
    /// Exact publication-authorized target-set identity.
    pub authorized_target_set_sha256: String,
    /// Exact disclosure manifest that authorized the target set.
    pub disclosure_manifest_sha256: String,
    /// Exact captured publication/source artifact.
    pub source_artifact_sha256: String,
    /// Narrow contamination-audit projection SHA used by MAG-DATA-010.
    pub audit_projection_sha256: String,
    /// Exact contamination audit identity.
    pub contamination_audit_sha256: String,
    /// Exact historical corpus audited.
    pub compact_corpus_sha256: String,
    /// Number of authorized/audited targets.
    pub target_count: u64,
}

impl AuthorizedAuditBinding {
    /// Deterministic digest of the complete convergence proof.
    pub fn binding_sha256(&self) -> Result<String, AuthorizedAuditError> {
        if self.schema_version != 1 {
            return Err(AuthorizedAuditError::UnsupportedSchema(self.schema_version));
        }
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// Bind one authority-derived target set to one verified historical run.
pub fn bind_authorized_targets_to_historical_run(
    run: &VerifiedHistoricalRun,
    authorized: &AuthorizedHistoricalTargetSet,
) -> Result<AuthorizedAuditBinding, AuthorizedAuditError> {
    authorized
        .validate()
        .map_err(|error| AuthorizedAuditError::TargetAuthority(error.to_string()))?;
    validate_stored_audit(&run.contamination_audit)?;

    let audit_targets = authorized
        .to_audit_targets()
        .map_err(|error| AuthorizedAuditError::TargetAuthority(error.to_string()))?;
    let projection_sha = canonical_target_set_sha256(audit_targets)
        .map_err(|error| AuthorizedAuditError::HistoricalRun(error.to_string()))?;

    if !projection_sha.eq_ignore_ascii_case(&run.bundle.target_set_sha256) {
        return Err(AuthorizedAuditError::TargetProjectionMismatch);
    }
    if !projection_sha.eq_ignore_ascii_case(&run.contamination_audit.target_set_sha256) {
        return Err(AuthorizedAuditError::AuditTargetSetMismatch);
    }
    if !run
        .bundle
        .compact_corpus_sha256
        .eq_ignore_ascii_case(&run.contamination_audit.corpus_snapshot_sha256)
    {
        return Err(AuthorizedAuditError::AuditCorpusMismatch);
    }
    if run.bundle.target_count != authorized.targets.len() as u64 {
        return Err(AuthorizedAuditError::TargetCountMismatch {
            historical: run.bundle.target_count,
            authorized: authorized.targets.len() as u64,
        });
    }
    if run.contamination_audit.results.len() as u64 != run.bundle.target_count {
        return Err(AuthorizedAuditError::AuditResultCountMismatch {
            audit: run.contamination_audit.results.len() as u64,
            bundle: run.bundle.target_count,
        });
    }

    let audit_sha = run
        .contamination_audit
        .audit_sha256()
        .map_err(|error| AuthorizedAuditError::HistoricalRun(error.to_string()))?;
    if !audit_sha.eq_ignore_ascii_case(&run.bundle.contamination_audit_sha256) {
        return Err(AuthorizedAuditError::AuditDigestMismatch);
    }

    let authorized_ids: BTreeSet<&str> = authorized
        .targets
        .iter()
        .map(|target| target.target_id.as_str())
        .collect();
    let audited_ids: BTreeSet<&str> = run
        .contamination_audit
        .results
        .iter()
        .map(|result| result.target_id.as_str())
        .collect();
    if authorized_ids != audited_ids {
        return Err(AuthorizedAuditError::AuditedTargetIdsMismatch);
    }

    let binding = AuthorizedAuditBinding {
        schema_version: 1,
        verified_run_sha256: run
            .verified_run_sha256()
            .map_err(|error| AuthorizedAuditError::HistoricalRun(error.to_string()))?,
        historical_bundle_sha256: run
            .bundle
            .bundle_sha256()
            .map_err(|error| AuthorizedAuditError::HistoricalRun(error.to_string()))?,
        authorized_target_set_sha256: authorized
            .target_set_sha256()
            .map_err(|error| AuthorizedAuditError::TargetAuthority(error.to_string()))?,
        disclosure_manifest_sha256: authorized.disclosure_manifest_sha256.clone(),
        source_artifact_sha256: authorized.source_artifact_sha256.clone(),
        audit_projection_sha256: projection_sha,
        contamination_audit_sha256: audit_sha,
        compact_corpus_sha256: run.bundle.compact_corpus_sha256.clone(),
        target_count: run.bundle.target_count,
    };
    binding.binding_sha256()?;
    Ok(binding)
}

/// Revalidate the internal canonical shape of a deserialized contamination audit.
fn validate_stored_audit(audit: &CorpusContaminationAudit) -> Result<(), AuthorizedAuditError> {
    if audit.schema_version != 1 {
        return Err(AuthorizedAuditError::UnsupportedAuditSchema(
            audit.schema_version,
        ));
    }
    if audit.results.is_empty() {
        return Err(AuthorizedAuditError::EmptyAuditResults);
    }

    let mut previous_target: Option<&str> = None;
    let mut target_ids = HashSet::new();
    let mut computed_counts: BTreeMap<String, u32> = BTreeMap::new();

    for result in &audit.results {
        if result.target_id.trim().is_empty() {
            return Err(AuthorizedAuditError::EmptyAuditedTargetId);
        }
        if !target_ids.insert(result.target_id.as_str()) {
            return Err(AuthorizedAuditError::DuplicateAuditedTargetId(
                result.target_id.clone(),
            ));
        }
        if previous_target.is_some_and(|prior| result.target_id.as_str() <= prior) {
            return Err(AuthorizedAuditError::NonCanonicalAuditTargetOrder);
        }
        previous_target = Some(&result.target_id);

        validate_sorted_unique_strings(&result.matching_record_ids, &result.target_id)?;
        validate_class_payload(&result.class, &result.target_id)?;

        let key = class_name(&result.class).to_string();
        let count = computed_counts.entry(key).or_insert(0);
        *count = count
            .checked_add(1)
            .ok_or(AuthorizedAuditError::AuditClassCountOverflow)?;
    }

    if computed_counts != audit.class_counts {
        return Err(AuthorizedAuditError::AuditClassCountsMismatch);
    }
    Ok(())
}

fn validate_sorted_unique_strings(
    values: &[String],
    target_id: &str,
) -> Result<(), AuthorizedAuditError> {
    let mut previous: Option<&str> = None;
    for value in values {
        if value.trim().is_empty() {
            return Err(AuthorizedAuditError::InvalidMatchingRecordIds(
                target_id.to_string(),
            ));
        }
        if previous.is_some_and(|prior| value.as_str() <= prior) {
            return Err(AuthorizedAuditError::InvalidMatchingRecordIds(
                target_id.to_string(),
            ));
        }
        previous = Some(value);
    }
    Ok(())
}

fn validate_class_payload(
    class: &ContaminationClass,
    target_id: &str,
) -> Result<(), AuthorizedAuditError> {
    match class {
        ContaminationClass::PropertyLabelPresent { labels } => {
            if labels.is_empty() {
                return Err(AuthorizedAuditError::InvalidClassPayload(
                    target_id.to_string(),
                ));
            }
            let mut previous = None;
            for label in labels {
                if label.property_id.trim().is_empty() || label.condition_signature.trim().is_empty() {
                    return Err(AuthorizedAuditError::InvalidClassPayload(
                        target_id.to_string(),
                    ));
                }
                let key = (label.property_id.as_str(), label.condition_signature.as_str());
                if previous.is_some_and(|prior| key <= prior) {
                    return Err(AuthorizedAuditError::InvalidClassPayload(
                        target_id.to_string(),
                    ));
                }
                previous = Some(key);
            }
        }
        ContaminationClass::AmbiguousIdentity { reason } if reason.trim().is_empty() => {
            return Err(AuthorizedAuditError::InvalidClassPayload(
                target_id.to_string(),
            ));
        }
        _ => {}
    }
    Ok(())
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

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Publication-target / historical-audit convergence failure.
#[derive(Debug, Error)]
pub enum AuthorizedAuditError {
    /// Target authority layer rejected the supplied target set.
    #[error("authorized target set rejected input: {0}")]
    TargetAuthority(String),
    /// Historical-run layer rejected the supplied run.
    #[error("verified historical run rejected input: {0}")]
    HistoricalRun(String),
    /// Authorized target audit projection differs from historical bundle target set.
    #[error("authorized target audit projection differs from historical run target-set SHA")]
    TargetProjectionMismatch,
    /// Contamination audit names another target-set projection.
    #[error("contamination audit target-set SHA differs from authorized projection")]
    AuditTargetSetMismatch,
    /// Contamination audit names another historical corpus.
    #[error("contamination audit corpus SHA differs from verified historical bundle")]
    AuditCorpusMismatch,
    /// Historical bundle target count differs from authorized target count.
    #[error("target-count mismatch: historical={historical}, authorized={authorized}")]
    TargetCountMismatch {
        /// Count in verified historical bundle.
        historical: u64,
        /// Count in authorized target set.
        authorized: u64,
    },
    /// Audit result count differs from the bundle target count.
    #[error("audit-result-count mismatch: audit={audit}, bundle={bundle}")]
    AuditResultCountMismatch {
        /// Audit result count.
        audit: u64,
        /// Bundle target count.
        bundle: u64,
    },
    /// Contamination audit bytes differ from the audit digest bound by the run.
    #[error("contamination audit digest differs from historical bundle")]
    AuditDigestMismatch,
    /// Audited target IDs differ from the authority-derived target IDs.
    #[error("audited target IDs differ from authorized target IDs")]
    AuditedTargetIdsMismatch,
    /// Stored audit schema unsupported.
    #[error("unsupported contamination-audit schema {0}")]
    UnsupportedAuditSchema(u32),
    /// Stored audit contains no results.
    #[error("stored contamination audit contains no results")]
    EmptyAuditResults,
    /// Stored audit target ID empty.
    #[error("stored contamination audit contains an empty target ID")]
    EmptyAuditedTargetId,
    /// Stored audit repeats a target ID.
    #[error("stored contamination audit repeats target ID: {0}")]
    DuplicateAuditedTargetId(String),
    /// Stored audit target results are not in strict canonical target-ID order.
    #[error("stored contamination audit results are not in strict target-ID order")]
    NonCanonicalAuditTargetOrder,
    /// Matching record IDs are empty, duplicated, or noncanonical.
    #[error("stored audit matching record IDs are noncanonical for target {0}")]
    InvalidMatchingRecordIds(String),
    /// Class-specific payload is malformed/noncanonical.
    #[error("stored audit class payload is invalid for target {0}")]
    InvalidClassPayload(String),
    /// Recomputed class counts differ from stored summary counts.
    #[error("stored contamination audit class_counts do not match result classes")]
    AuditClassCountsMismatch,
    /// Class count overflowed.
    #[error("stored contamination audit class count overflowed")]
    AuditClassCountOverflow,
    /// Binding schema unsupported.
    #[error("unsupported authorized-audit binding schema {0}")]
    UnsupportedSchema(u32),
    /// Serialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_materials_corpus_audit::{
        ContaminationClass, CorpusContaminationAudit, TargetContaminationResult,
    };
    use symthaea_materials_history_tool::HistoricalRunBundle;
    use symthaea_materials_target_set::{
        AuthorizedHistoricalTarget, AuthorizedQuantitativeLabel,
    };

    fn hex(ch: char) -> String {
        ch.to_string().repeat(64)
    }

    fn authorized() -> AuthorizedHistoricalTargetSet {
        AuthorizedHistoricalTargetSet {
            schema_version: 1,
            disclosure_manifest_sha256: hex('1'),
            source_artifact_sha256: hex('2'),
            disclosure_date: "2026-02-09".to_string(),
            targets: vec![AuthorizedHistoricalTarget {
                target_id: "candidate".to_string(),
                composition_sha256: hex('3'),
                structure_sha256: hex('4'),
                structure_artifact_sha256: hex('5'),
                structure_fact_id: "01-structure".to_string(),
                properties: vec![AuthorizedQuantitativeLabel {
                    disclosure_fact_id: "02-k1".to_string(),
                    property_id: "k1".to_string(),
                    value: "1.1".to_string(),
                    unit: "MJ/m^3".to_string(),
                    condition_signature: "0K|SOC".to_string(),
                    method_artifact_sha256: hex('6'),
                }],
            }],
        }
    }

    fn run(authorized: &AuthorizedHistoricalTargetSet) -> VerifiedHistoricalRun {
        let projection = canonical_target_set_sha256(authorized.to_audit_targets().unwrap()).unwrap();
        let audit = CorpusContaminationAudit {
            schema_version: 1,
            corpus_snapshot_sha256: hex('7'),
            target_set_sha256: projection.clone(),
            results: vec![TargetContaminationResult {
                target_id: "candidate".to_string(),
                class: ContaminationClass::AbsentFromPrecutoffCorpus,
                matching_record_ids: Vec::new(),
            }],
            class_counts: BTreeMap::from([("AbsentFromPrecutoffCorpus".to_string(), 1)]),
        };
        let audit_sha = audit.audit_sha256().unwrap();
        VerifiedHistoricalRun {
            bundle: HistoricalRunBundle {
                schema_version: 1,
                verifier_artifact_sha256: hex('8'),
                protocol_sha256: hex('9'),
                compressed_snapshot_sha256: hex('a'),
                compressed_snapshot_bytes: 1,
                acquisition_receipt_sha256: hex('b'),
                import_profile_sha256: hex('c'),
                import_receipt_sha256: hex('d'),
                schema_inventory_sha256: hex('e'),
                row_count_inventory_sha256: hex('f'),
                import_extraction_binding_sha256: hex('1'),
                extraction_receipt_sha256: hex('2'),
                compact_corpus_sha256: hex('7'),
                compact_corpus_record_count: 1,
                target_set_sha256: projection,
                target_count: 1,
                contamination_audit_sha256: audit_sha,
            },
            contamination_audit: audit,
        }
    }

    fn refresh_audit_digest(run: &mut VerifiedHistoricalRun) {
        run.bundle.contamination_audit_sha256 = run.contamination_audit.audit_sha256().unwrap();
    }

    #[test]
    fn exact_authorized_projection_binds_to_audit() {
        let authorized = authorized();
        let run = run(&authorized);
        let binding = bind_authorized_targets_to_historical_run(&run, &authorized).unwrap();
        assert_eq!(binding.target_count, 1);
        assert_eq!(binding.binding_sha256().unwrap().len(), 64);
    }

    #[test]
    fn arbitrary_historical_target_projection_is_rejected() {
        let authorized = authorized();
        let mut run = run(&authorized);
        run.bundle.target_set_sha256 = hex('f');
        assert!(matches!(
            bind_authorized_targets_to_historical_run(&run, &authorized),
            Err(AuthorizedAuditError::TargetProjectionMismatch)
        ));
    }

    #[test]
    fn duplicate_audit_target_ids_are_rejected_even_if_outer_digest_is_refreshed() {
        let authorized = authorized();
        let mut run = run(&authorized);
        run.contamination_audit.results.push(run.contamination_audit.results[0].clone());
        run.contamination_audit.class_counts.insert(
            "AbsentFromPrecutoffCorpus".to_string(),
            2,
        );
        refresh_audit_digest(&mut run);
        assert!(matches!(
            bind_authorized_targets_to_historical_run(&run, &authorized),
            Err(AuthorizedAuditError::DuplicateAuditedTargetId(_))
        ));
    }

    #[test]
    fn forged_class_counts_are_rejected_even_if_outer_digest_is_refreshed() {
        let authorized = authorized();
        let mut run = run(&authorized);
        run.contamination_audit.class_counts.insert(
            "AbsentFromPrecutoffCorpus".to_string(),
            9,
        );
        refresh_audit_digest(&mut run);
        assert!(matches!(
            bind_authorized_targets_to_historical_run(&run, &authorized),
            Err(AuthorizedAuditError::AuditClassCountsMismatch)
        ));
    }
}
