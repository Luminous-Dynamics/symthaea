use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use symthaea_assurance_tpm2_measured_boot_replay::{MeasuredBootReplayRecord, REPLAY_SCOPE};

use crate::measurements::{
    MeasurementExtractionVerificationReceipt, NormalizedMeasurementSet,
};
use crate::reference::{
    ReferenceIntegrityManifest, ReferenceIntegrityPolicy, RimSignatureVerificationReceipt,
};
use crate::util::{nonempty_refs, push_field, valid_digest};
use crate::REFERENCE_INTEGRITY_SCOPE;

const EVALUATION_REPORT_DIGEST_SCHEMA: &[u8] =
    b"symthaea-reference-integrity-evaluation-report-v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferenceIntegrityStatus {
    Approved,
    Rejected,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferenceIntegrityIssue {
    ManifestNotYetValid,
    ManifestExpired,
    ReplayEvidenceTooOld,
    ManifestVerificationTooOld,
    ExtractionVerificationTooOld,
    ExplicitlyDenied {
        measurement_id: String,
        rule_id: String,
    },
    UnknownCritical {
        measurement_id: String,
    },
    UnknownNoncritical {
        measurement_id: String,
    },
    MissingRequiredMeasurement {
        rule_id: String,
        observed: u32,
        minimum: u32,
    },
    TooManyMeasurements {
        rule_id: String,
        observed: u32,
        maximum: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceIntegrityEvaluationReport {
    pub schema_version: String,
    pub policy_id: String,
    pub replay_record_digest: String,
    pub manifest_digest: String,
    pub manifest_signature_receipt_digest: String,
    pub measurement_set_digest: String,
    pub extraction_receipt_digest: String,
    pub status: ReferenceIntegrityStatus,
    pub matched_rule_ids: Vec<String>,
    pub rejected_measurement_ids: Vec<String>,
    pub unknown_measurement_ids: Vec<String>,
    pub missing_rule_ids: Vec<String>,
    pub issues: Vec<ReferenceIntegrityIssue>,
    pub evaluated_at_ms: u64,
    pub scope: String,
    pub evidence_refs: Vec<String>,
}

impl ReferenceIntegrityEvaluationReport {
    pub fn report_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(EVALUATION_REPORT_DIGEST_SCHEMA);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.replay_record_digest.as_str(),
            self.manifest_digest.as_str(),
            self.manifest_signature_receipt_digest.as_str(),
            self.measurement_set_digest.as_str(),
            self.extraction_receipt_digest.as_str(),
        ] {
            push_field(&mut hasher, value);
        }
        push_field(
            &mut hasher,
            match self.status {
                ReferenceIntegrityStatus::Approved => "approved",
                ReferenceIntegrityStatus::Rejected => "rejected",
                ReferenceIntegrityStatus::Incomplete => "incomplete",
            },
        );
        for value in &self.matched_rule_ids {
            push_field(&mut hasher, &format!("matched:{value}"));
        }
        for value in &self.rejected_measurement_ids {
            push_field(&mut hasher, &format!("rejected:{value}"));
        }
        for value in &self.unknown_measurement_ids {
            push_field(&mut hasher, &format!("unknown:{value}"));
        }
        for value in &self.missing_rule_ids {
            push_field(&mut hasher, &format!("missing:{value}"));
        }
        for issue in &self.issues {
            push_issue(&mut hasher, issue);
        }
        push_field(&mut hasher, &self.evaluated_at_ms.to_string());
        push_field(&mut hasher, &self.scope);

        let mut refs = self.evidence_refs.clone();
        refs.sort();
        for reference in refs {
            push_field(&mut hasher, &reference);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferenceIntegrityError {
    InvalidPolicy,
    InvalidReplayRecord,
    ReplayRecordMismatch,
    InvalidManifest,
    ManifestMismatch,
    ManifestSubjectMismatch,
    InvalidManifestSignatureReceipt,
    ManifestVerifierMismatch,
    InvalidMeasurementSet,
    MeasurementReplayMismatch,
    InvalidExtractionReceipt,
    ExtractionVerifierMismatch,
    EvaluationBeforeEvidence,
}

pub fn evaluate_reference_integrity(
    policy: &ReferenceIntegrityPolicy,
    replay: &MeasuredBootReplayRecord,
    manifest: &ReferenceIntegrityManifest,
    manifest_signature: &RimSignatureVerificationReceipt,
    measurements: &NormalizedMeasurementSet,
    extraction: &MeasurementExtractionVerificationReceipt,
    evaluated_at_ms: u64,
) -> Result<ReferenceIntegrityEvaluationReport, ReferenceIntegrityError> {
    if !policy.validate() {
        return Err(ReferenceIntegrityError::InvalidPolicy);
    }
    if !valid_replay_record(replay) {
        return Err(ReferenceIntegrityError::InvalidReplayRecord);
    }
    let replay_digest = replay.replay_record_digest();
    if replay_digest != policy.expected_replay_record_digest {
        return Err(ReferenceIntegrityError::ReplayRecordMismatch);
    }

    if !manifest.validate() {
        return Err(ReferenceIntegrityError::InvalidManifest);
    }
    let manifest_digest = manifest.manifest_digest();
    if manifest_digest != policy.expected_manifest_digest {
        return Err(ReferenceIntegrityError::ManifestMismatch);
    }
    if manifest.subject_class_ref != policy.expected_subject_class_ref
        || manifest.release_ref != policy.expected_release_ref
    {
        return Err(ReferenceIntegrityError::ManifestSubjectMismatch);
    }
    if !manifest_signature.validate_for(manifest) {
        return Err(ReferenceIntegrityError::InvalidManifestSignatureReceipt);
    }
    if manifest_signature.verifier_ref != policy.expected_manifest_verifier_ref
        || manifest_signature.verification_tool_digest
            != policy.expected_manifest_verification_tool_digest
    {
        return Err(ReferenceIntegrityError::ManifestVerifierMismatch);
    }

    if !measurements.validate() {
        return Err(ReferenceIntegrityError::InvalidMeasurementSet);
    }
    if measurements.replay_record_digest != replay_digest
        || measurements.log_bundle_digest != replay.log_bundle_digest
    {
        return Err(ReferenceIntegrityError::MeasurementReplayMismatch);
    }
    if !extraction.validate_for(measurements) {
        return Err(ReferenceIntegrityError::InvalidExtractionReceipt);
    }
    if extraction.verifier_ref != policy.expected_extraction_verifier_ref
        || extraction.extraction_tool_digest != policy.expected_extraction_tool_digest
    {
        return Err(ReferenceIntegrityError::ExtractionVerifierMismatch);
    }
    if evaluated_at_ms < replay.qualified_at_ms
        || evaluated_at_ms < manifest_signature.verified_at_ms
        || evaluated_at_ms < extraction.verified_at_ms
    {
        return Err(ReferenceIntegrityError::EvaluationBeforeEvidence);
    }

    let mut issues = freshness_issues(
        policy,
        replay,
        manifest,
        manifest_signature,
        extraction,
        evaluated_at_ms,
    );

    let rules: BTreeMap<_, _> = manifest
        .rules
        .iter()
        .map(|rule| (rule.selector.clone(), rule))
        .collect();
    let critical: BTreeSet<_> = manifest.critical_selectors.iter().cloned().collect();
    let mut occurrence_counts: BTreeMap<String, u32> = BTreeMap::new();
    let mut allowed_counts: BTreeMap<String, u32> = BTreeMap::new();
    let mut matched_rule_ids = BTreeSet::new();
    let mut rejected_measurement_ids = BTreeSet::new();
    let mut unknown_measurement_ids = BTreeSet::new();

    for measurement in &measurements.measurements {
        match rules.get(&measurement.selector) {
            Some(rule) => {
                *occurrence_counts.entry(rule.rule_id.clone()).or_default() += 1;
                if rule.denied_digests.contains(&measurement.digest) {
                    rejected_measurement_ids.insert(measurement.measurement_id.clone());
                    issues.push(ReferenceIntegrityIssue::ExplicitlyDenied {
                        measurement_id: measurement.measurement_id.clone(),
                        rule_id: rule.rule_id.clone(),
                    });
                } else if rule.allowed_digests.contains(&measurement.digest) {
                    *allowed_counts.entry(rule.rule_id.clone()).or_default() += 1;
                    matched_rule_ids.insert(rule.rule_id.clone());
                } else if rule.critical {
                    rejected_measurement_ids.insert(measurement.measurement_id.clone());
                    issues.push(ReferenceIntegrityIssue::UnknownCritical {
                        measurement_id: measurement.measurement_id.clone(),
                    });
                } else {
                    unknown_measurement_ids.insert(measurement.measurement_id.clone());
                    if !policy.allow_unknown_noncritical {
                        issues.push(ReferenceIntegrityIssue::UnknownNoncritical {
                            measurement_id: measurement.measurement_id.clone(),
                        });
                    }
                }
            }
            None => {
                if critical.contains(&measurement.selector) {
                    rejected_measurement_ids.insert(measurement.measurement_id.clone());
                    issues.push(ReferenceIntegrityIssue::UnknownCritical {
                        measurement_id: measurement.measurement_id.clone(),
                    });
                } else {
                    unknown_measurement_ids.insert(measurement.measurement_id.clone());
                    if !policy.allow_unknown_noncritical {
                        issues.push(ReferenceIntegrityIssue::UnknownNoncritical {
                            measurement_id: measurement.measurement_id.clone(),
                        });
                    }
                }
            }
        }
    }

    let mut missing_rule_ids = BTreeSet::new();
    let mut sorted_rules = manifest.rules.iter().collect::<Vec<_>>();
    sorted_rules.sort_by(|left, right| left.selector.cmp(&right.selector));
    for rule in sorted_rules {
        let allowed = allowed_counts.get(&rule.rule_id).copied().unwrap_or(0);
        let observed = occurrence_counts.get(&rule.rule_id).copied().unwrap_or(0);
        if allowed < rule.minimum_occurrences {
            missing_rule_ids.insert(rule.rule_id.clone());
            issues.push(ReferenceIntegrityIssue::MissingRequiredMeasurement {
                rule_id: rule.rule_id.clone(),
                observed: allowed,
                minimum: rule.minimum_occurrences,
            });
        }
        if let Some(maximum) = rule.maximum_occurrences {
            if observed > maximum {
                issues.push(ReferenceIntegrityIssue::TooManyMeasurements {
                    rule_id: rule.rule_id.clone(),
                    observed,
                    maximum,
                });
            }
        }
    }

    let status = status_from_issues(&issues);
    let mut evidence_refs = policy.evidence_refs.clone();
    evidence_refs.push(format!("manifest:{manifest_digest}"));
    evidence_refs.push(format!(
        "manifest-signature:{}",
        manifest_signature.receipt_digest()
    ));
    evidence_refs.push(format!(
        "measurements:{}",
        measurements.measurement_set_digest()
    ));
    evidence_refs.push(format!("extraction:{}", extraction.receipt_digest()));

    Ok(ReferenceIntegrityEvaluationReport {
        schema_version: "1".into(),
        policy_id: policy.policy_id.clone(),
        replay_record_digest: replay_digest,
        manifest_digest,
        manifest_signature_receipt_digest: manifest_signature.receipt_digest(),
        measurement_set_digest: measurements.measurement_set_digest(),
        extraction_receipt_digest: extraction.receipt_digest(),
        status,
        matched_rule_ids: matched_rule_ids.into_iter().collect(),
        rejected_measurement_ids: rejected_measurement_ids.into_iter().collect(),
        unknown_measurement_ids: unknown_measurement_ids.into_iter().collect(),
        missing_rule_ids: missing_rule_ids.into_iter().collect(),
        issues,
        evaluated_at_ms,
        scope: REFERENCE_INTEGRITY_SCOPE.into(),
        evidence_refs,
    })
}

fn freshness_issues(
    policy: &ReferenceIntegrityPolicy,
    replay: &MeasuredBootReplayRecord,
    manifest: &ReferenceIntegrityManifest,
    signature: &RimSignatureVerificationReceipt,
    extraction: &MeasurementExtractionVerificationReceipt,
    now_ms: u64,
) -> Vec<ReferenceIntegrityIssue> {
    let mut issues = Vec::new();
    if now_ms < manifest.valid_from_ms {
        issues.push(ReferenceIntegrityIssue::ManifestNotYetValid);
    }
    if manifest.valid_until_ms.is_some_and(|until| now_ms > until) {
        issues.push(ReferenceIntegrityIssue::ManifestExpired);
    }
    if now_ms - replay.qualified_at_ms > policy.max_replay_to_evaluation_ms {
        issues.push(ReferenceIntegrityIssue::ReplayEvidenceTooOld);
    }
    if now_ms - signature.verified_at_ms > policy.max_manifest_verification_age_ms {
        issues.push(ReferenceIntegrityIssue::ManifestVerificationTooOld);
    }
    if now_ms - extraction.verified_at_ms > policy.max_extraction_verification_age_ms {
        issues.push(ReferenceIntegrityIssue::ExtractionVerificationTooOld);
    }
    issues
}

fn status_from_issues(issues: &[ReferenceIntegrityIssue]) -> ReferenceIntegrityStatus {
    if issues.iter().any(|issue| {
        matches!(
            issue,
            ReferenceIntegrityIssue::ExplicitlyDenied { .. }
                | ReferenceIntegrityIssue::UnknownCritical { .. }
                | ReferenceIntegrityIssue::TooManyMeasurements { .. }
        )
    }) {
        ReferenceIntegrityStatus::Rejected
    } else if issues.is_empty() {
        ReferenceIntegrityStatus::Approved
    } else {
        ReferenceIntegrityStatus::Incomplete
    }
}

fn valid_replay_record(record: &MeasuredBootReplayRecord) -> bool {
    !record.schema_version.trim().is_empty()
        && !record.policy_id.trim().is_empty()
        && valid_digest(&record.possession_digest)
        && valid_digest(&record.quote_artifact_digest)
        && valid_digest(&record.log_bundle_digest)
        && valid_digest(&record.replay_receipt_digest)
        && valid_digest(&record.platform_qualification_digest)
        && valid_digest(&record.pcr_selection_digest)
        && valid_digest(&record.quoted_pcr_values_digest)
        && record.scope == REPLAY_SCOPE
        && nonempty_refs(&record.evidence_refs)
}

fn push_issue(hasher: &mut blake3::Hasher, issue: &ReferenceIntegrityIssue) {
    match issue {
        ReferenceIntegrityIssue::ManifestNotYetValid => {
            push_field(hasher, "issue:manifest-not-yet-valid")
        }
        ReferenceIntegrityIssue::ManifestExpired => push_field(hasher, "issue:manifest-expired"),
        ReferenceIntegrityIssue::ReplayEvidenceTooOld => push_field(hasher, "issue:replay-too-old"),
        ReferenceIntegrityIssue::ManifestVerificationTooOld => {
            push_field(hasher, "issue:manifest-verification-too-old")
        }
        ReferenceIntegrityIssue::ExtractionVerificationTooOld => {
            push_field(hasher, "issue:extraction-verification-too-old")
        }
        ReferenceIntegrityIssue::ExplicitlyDenied {
            measurement_id,
            rule_id,
        } => {
            push_field(hasher, "issue:explicitly-denied");
            push_field(hasher, measurement_id);
            push_field(hasher, rule_id);
        }
        ReferenceIntegrityIssue::UnknownCritical { measurement_id } => {
            push_field(hasher, "issue:unknown-critical");
            push_field(hasher, measurement_id);
        }
        ReferenceIntegrityIssue::UnknownNoncritical { measurement_id } => {
            push_field(hasher, "issue:unknown-noncritical");
            push_field(hasher, measurement_id);
        }
        ReferenceIntegrityIssue::MissingRequiredMeasurement {
            rule_id,
            observed,
            minimum,
        } => {
            push_field(hasher, "issue:missing-required");
            push_field(hasher, rule_id);
            push_field(hasher, &observed.to_string());
            push_field(hasher, &minimum.to_string());
        }
        ReferenceIntegrityIssue::TooManyMeasurements {
            rule_id,
            observed,
            maximum,
        } => {
            push_field(hasher, "issue:too-many");
            push_field(hasher, rule_id);
            push_field(hasher, &observed.to_string());
            push_field(hasher, &maximum.to_string());
        }
    }
}
