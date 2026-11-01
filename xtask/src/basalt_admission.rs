//! Reference evaluator for BASALT claim-scoped three-valued admission.
//!
//! This module reduces independently verified obligation outcomes. It does not
//! verify DSSE signatures, Rekor proofs, witness signatures, etcd freshness,
//! observer completeness, or storage durability. Those must be verified by
//! separate adapters and represented as typed obligation outcomes here.
//!
//! Admission is allowed only when every required obligation is valid and the
//! exact subject/use binding is valid. Unknown is a fail-closed result.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use serde::{Deserialize, Serialize};

const INPUT_SCHEMA_VERSION: u32 = 1;
const CAPSULE_SCHEMA_VERSION: u32 = 1;
const MAX_IDENTIFIER_BYTES: usize = 128;

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdmissionInput {
    pub schema_version: u32,
    pub profile: AdmissionProfile,
    pub expected_binding: ClaimBinding,
    pub capsule: EvidenceCapsule,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdmissionProfile {
    pub profile_id: String,
    pub profile_version: u32,
    pub required_obligations: Vec<String>,
    #[serde(default)]
    pub optional_obligations: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ClaimBinding {
    pub claim_id: String,
    pub subject_digest: String,
    pub intended_use_id: String,
    pub context_digest: String,
    pub policy_digest: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvidenceCapsule {
    pub schema_version: u32,
    pub profile_id: String,
    pub profile_version: u32,
    pub binding: ClaimBinding,
    pub obligations: Vec<ObligationEvaluation>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ObligationOutcome {
    Valid,
    Invalid,
    Unknown,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObligationEvaluation {
    pub id: String,
    pub outcome: ObligationOutcome,
    pub reason_code: String,
    #[serde(default)]
    pub evidence_ref: Option<String>,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionStatus {
    Valid,
    Invalid,
    Unknown,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct AdmissionReason {
    pub code: String,
    pub detail: String,
    pub obligation_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct AdmissionReport {
    pub report_version: u32,
    /// This evaluator reduces supplied outcomes; it does not authenticate the profile or verify evidence.
    pub reference_only: bool,
    pub status: AdmissionStatus,
    pub admission_allowed: bool,
    pub obligations: Vec<ObligationEvaluation>,
    pub reasons: Vec<AdmissionReason>,
}

/// Evaluate a capsule against the exact expected claim binding and profile.
///
/// This function trusts neither a caller-provided aggregate verdict nor a
/// cached admission_allowed field: neither is accepted in the input model.
/// The caller must authenticate the selected profile and obtain obligation
/// outcomes from independent verifiers before using this reduction. This
/// reference module itself verifies neither profile authenticity nor evidence.
pub fn evaluate(input: &AdmissionInput) -> AdmissionReport {
    let mut reasons = Vec::new();
    let mut obligations = Vec::new();
    let mut has_invalid = false;
    let mut has_unknown = false;

    if input.schema_version != INPUT_SCHEMA_VERSION {
        reasons.push(global_reason(
            "unsupported_input_schema",
            format!(
                "input schema version {} is not supported (expected {})",
                input.schema_version, INPUT_SCHEMA_VERSION
            ),
        ));
        return report(AdmissionStatus::Invalid, obligations, reasons);
    }

    if input.capsule.schema_version != CAPSULE_SCHEMA_VERSION {
        reasons.push(global_reason(
            "unsupported_capsule_schema",
            format!(
                "capsule schema version {} is not supported (expected {})",
                input.capsule.schema_version, CAPSULE_SCHEMA_VERSION
            ),
        ));
        return report(AdmissionStatus::Invalid, obligations, reasons);
    }

    if !valid_identifier(&input.profile.profile_id)
        || input.profile.profile_version == 0
        || input.profile.required_obligations.is_empty()
    {
        reasons.push(global_reason(
            "invalid_admission_profile",
            "profile ID/version must be valid and at least one mandatory obligation is required",
        ));
        return report(AdmissionStatus::Invalid, obligations, reasons);
    }

    let mut required = BTreeSet::new();
    for id in &input.profile.required_obligations {
        if !valid_identifier(id) || !required.insert(id.clone()) {
            reasons.push(AdmissionReason {
                code: "invalid_or_duplicate_required_obligation".to_string(),
                detail: "required obligation IDs must be valid and unique".to_string(),
                obligation_id: Some(id.clone()),
            });
            has_invalid = true;
        }
    }

    let mut optional = BTreeSet::new();
    for id in &input.profile.optional_obligations {
        if !valid_identifier(id)
            || !optional.insert(id.clone())
            || required.contains(id)
        {
            reasons.push(AdmissionReason {
                code: "invalid_optional_obligation_profile".to_string(),
                detail: "optional IDs must be valid, unique, and disjoint from required IDs"
                    .to_string(),
                obligation_id: Some(id.clone()),
            });
            has_invalid = true;
        }
    }

    if input.capsule.profile_id != input.profile.profile_id {
        reasons.push(global_reason(
            "profile_identity_mismatch",
            "capsule profile ID does not match the requested admission profile",
        ));
        has_invalid = true;
    }

    if input.capsule.profile_version != input.profile.profile_version {
        reasons.push(global_reason(
            "profile_version_reverification_required",
            "capsule was evaluated under a different profile version",
        ));
        has_unknown = true;
    }

    if !binding_fields_valid(&input.expected_binding)
        || !binding_fields_valid(&input.capsule.binding)
    {
        reasons.push(global_reason(
            "malformed_claim_binding",
            "all claim-binding fields must be non-empty",
        ));
        has_invalid = true;
    }

    compare_binding(
        &input.expected_binding,
        &input.capsule.binding,
        &mut reasons,
        &mut has_invalid,
        &mut has_unknown,
    );

    let mut by_id: BTreeMap<String, ObligationEvaluation> = BTreeMap::new();
    let mut duplicate_ids = BTreeSet::new();

    for evaluation in &input.capsule.obligations {
        if !valid_identifier(&evaluation.id) {
            reasons.push(AdmissionReason {
                code: "malformed_obligation_id".to_string(),
                detail: "obligation ID is empty, oversized, or contains unsupported characters"
                    .to_string(),
                obligation_id: Some(evaluation.id.clone()),
            });
            has_invalid = true;
            continue;
        }

        if !required.contains(&evaluation.id) && !optional.contains(&evaluation.id) {
            reasons.push(AdmissionReason {
                code: "unexpected_obligation".to_string(),
                detail: "capsule contains an obligation not declared by the selected profile"
                    .to_string(),
                obligation_id: Some(evaluation.id.clone()),
            });
            has_invalid = true;
            continue;
        }

        if by_id.contains_key(&evaluation.id) {
            duplicate_ids.insert(evaluation.id.clone());
        } else {
            by_id.insert(evaluation.id.clone(), evaluation.clone());
        }
    }

    for id in duplicate_ids {
        by_id.insert(
            id.clone(),
            ObligationEvaluation {
                id: id.clone(),
                outcome: ObligationOutcome::Invalid,
                reason_code: "duplicate_obligation_result".to_string(),
                evidence_ref: None,
            },
        );
        reasons.push(AdmissionReason {
            code: "duplicate_obligation_result".to_string(),
            detail: "an obligation has more than one result in the capsule".to_string(),
            obligation_id: Some(id),
        });
        has_invalid = true;
    }

    for id in &required {
        match by_id.remove(id) {
            Some(mut evaluation) => {
                normalize_required_evaluation(
                    &mut evaluation,
                    &mut reasons,
                    &mut has_invalid,
                    &mut has_unknown,
                );
                obligations.push(evaluation);
            }
            None => {
                obligations.push(ObligationEvaluation {
                    id: id.clone(),
                    outcome: ObligationOutcome::Unknown,
                    reason_code: "missing_required_obligation".to_string(),
                    evidence_ref: None,
                });
                reasons.push(AdmissionReason {
                    code: "missing_required_obligation".to_string(),
                    detail: "mandatory evidence was not supplied".to_string(),
                    obligation_id: Some(id.clone()),
                });
                has_unknown = true;
            }
        }
    }

    // Optional results are preserved for audit, but cannot compensate for or
    // invalidate a missing/failed mandatory obligation by definition.
    for id in &optional {
        if let Some(evaluation) = by_id.remove(id) {
            obligations.push(evaluation);
        }
    }

    // Keep the report deterministic regardless of input obligation ordering.
    obligations.sort_by(|left, right| left.id.cmp(&right.id));
    reasons.sort_by(|left, right| {
        left.obligation_id
            .cmp(&right.obligation_id)
            .then_with(|| left.code.cmp(&right.code))
            .then_with(|| left.detail.cmp(&right.detail))
    });

    let status = if has_invalid {
        AdmissionStatus::Invalid
    } else if has_unknown {
        AdmissionStatus::Unknown
    } else {
        AdmissionStatus::Valid
    };

    report(status, obligations, reasons)
}

fn normalize_required_evaluation(
    evaluation: &mut ObligationEvaluation,
    reasons: &mut Vec<AdmissionReason>,
    has_invalid: &mut bool,
    has_unknown: &mut bool,
) {
    if evaluation.reason_code.trim().is_empty() {
        evaluation.outcome = ObligationOutcome::Invalid;
        evaluation.reason_code = "missing_reason_code".to_string();
        reasons.push(AdmissionReason {
            code: "missing_reason_code".to_string(),
            detail: "mandatory obligation result must carry a reason code".to_string(),
            obligation_id: Some(evaluation.id.clone()),
        });
        *has_invalid = true;
        return;
    }

    match evaluation.outcome {
        ObligationOutcome::Valid => {
            if evaluation
                .evidence_ref
                .as_deref()
                .is_none_or(|value| value.trim().is_empty())
            {
                evaluation.outcome = ObligationOutcome::Unknown;
                evaluation.reason_code = "missing_evidence_reference".to_string();
                reasons.push(AdmissionReason {
                    code: "missing_evidence_reference".to_string(),
                    detail: "a valid mandatory result must identify the evidence it verified"
                        .to_string(),
                    obligation_id: Some(evaluation.id.clone()),
                });
                *has_unknown = true;
            }
        }
        ObligationOutcome::Invalid => {
            reasons.push(AdmissionReason {
                code: evaluation.reason_code.clone(),
                detail: "mandatory obligation was affirmatively rejected".to_string(),
                obligation_id: Some(evaluation.id.clone()),
            });
            *has_invalid = true;
        }
        ObligationOutcome::Unknown => {
            reasons.push(AdmissionReason {
                code: evaluation.reason_code.clone(),
                detail: "mandatory obligation could not be established".to_string(),
                obligation_id: Some(evaluation.id.clone()),
            });
            *has_unknown = true;
        }
    }
}

fn compare_binding(
    expected: &ClaimBinding,
    actual: &ClaimBinding,
    reasons: &mut Vec<AdmissionReason>,
    has_invalid: &mut bool,
    has_unknown: &mut bool,
) {
    for (field, expected_value, actual_value) in [
        ("claim_id", &expected.claim_id, &actual.claim_id),
        ("subject_digest", &expected.subject_digest, &actual.subject_digest),
        (
            "intended_use_id",
            &expected.intended_use_id,
            &actual.intended_use_id,
        ),
    ] {
        if expected_value != actual_value {
            reasons.push(global_reason(
                "claim_binding_mismatch",
                format!("{field} does not match the requested claim"),
            ));
            *has_invalid = true;
        }
    }

    for (field, expected_value, actual_value) in [
        ("context_digest", &expected.context_digest, &actual.context_digest),
        ("policy_digest", &expected.policy_digest, &actual.policy_digest),
    ] {
        if expected_value != actual_value {
            reasons.push(global_reason(
                "claim_reverification_required",
                format!("{field} differs; prior verification cannot be reused as current"),
            ));
            *has_unknown = true;
        }
    }
}

fn binding_fields_valid(binding: &ClaimBinding) -> bool {
    [
        &binding.claim_id,
        &binding.subject_digest,
        &binding.intended_use_id,
        &binding.context_digest,
        &binding.policy_digest,
    ]
    .iter()
    .all(|value| !value.trim().is_empty())
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_IDENTIFIER_BYTES
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'-' | b'_' | b'.')
        })
}

fn global_reason(code: &str, detail: impl Into<String>) -> AdmissionReason {
    AdmissionReason {
        code: code.to_string(),
        detail: detail.into(),
        obligation_id: None,
    }
}

fn report(
    status: AdmissionStatus,
    obligations: Vec<ObligationEvaluation>,
    reasons: Vec<AdmissionReason>,
) -> AdmissionReport {
    AdmissionReport {
        report_version: 1,
        reference_only: true,
        status,
        admission_allowed: status == AdmissionStatus::Valid,
        obligations,
        reasons,
    }
}

/// Evaluate a JSON input file and print a machine-readable report.
///
/// A non-valid result is printed before returning a non-zero error so callers
/// can retain the typed denial/unknown report in automation logs.
pub fn check(input_path: PathBuf) -> anyhow::Result<()> {
    let raw = fs::read(&input_path)
        .with_context(|| format!("failed to read BASALT admission input {}", input_path.display()))?;
    let input: AdmissionInput = serde_json::from_slice(&raw)
        .with_context(|| format!("failed to parse BASALT admission input {}", input_path.display()))?;
    let result = evaluate(&input);
    println!("{}", serde_json::to_string_pretty(&result)?);

    if !result.admission_allowed {
        anyhow::bail!("BASALT admission denied with status {:?}", result.status);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding() -> ClaimBinding {
        ClaimBinding {
            claim_id: "runtime-claim-01".to_string(),
            subject_digest: "sha256:subject-01".to_string(),
            intended_use_id: "ci-promotion".to_string(),
            context_digest: "sha256:context-01".to_string(),
            policy_digest: "sha256:policy-01".to_string(),
        }
    }

    fn evaluation(
        id: &str,
        outcome: ObligationOutcome,
        reason_code: &str,
    ) -> ObligationEvaluation {
        ObligationEvaluation {
            id: id.to_string(),
            outcome,
            reason_code: reason_code.to_string(),
            evidence_ref: Some(format!("sha256:evidence-{id}")),
        }
    }

    fn valid_input() -> AdmissionInput {
        let binding = binding();
        AdmissionInput {
            schema_version: INPUT_SCHEMA_VERSION,
            profile: AdmissionProfile {
                profile_id: "basalt-runtime-v1".to_string(),
                profile_version: 1,
                required_obligations: vec![
                    "payload-binding".to_string(),
                    "signer-authorization".to_string(),
                ],
                optional_obligations: vec!["diagnostic-provenance".to_string()],
            },
            expected_binding: binding.clone(),
            capsule: EvidenceCapsule {
                schema_version: CAPSULE_SCHEMA_VERSION,
                profile_id: "basalt-runtime-v1".to_string(),
                profile_version: 1,
                binding,
                obligations: vec![
                    evaluation("payload-binding", ObligationOutcome::Valid, "verified"),
                    evaluation("signer-authorization", ObligationOutcome::Valid, "verified"),
                ],
            },
        }
    }

    #[test]
    fn all_required_valid_is_admissible() {
        let result = evaluate(&valid_input());
        assert_eq!(result.status, AdmissionStatus::Valid);
        assert!(result.admission_allowed);
        assert!(result.reasons.is_empty());
    }

    #[test]
    fn missing_required_obligation_is_unknown_and_fails_closed() {
        let mut input = valid_input();
        input.capsule.obligations.pop();
        let result = evaluate(&input);
        assert_eq!(result.status, AdmissionStatus::Unknown);
        assert!(!result.admission_allowed);
        assert!(result
            .reasons
            .iter()
            .any(|reason| reason.code == "missing_required_obligation"));
    }

    #[test]
    fn invalid_dominates_unknown_in_conjunction() {
        let mut input = valid_input();
        input.capsule.obligations = vec![
            evaluation("payload-binding", ObligationOutcome::Invalid, "digest-mismatch"),
            evaluation("signer-authorization", ObligationOutcome::Unknown, "revocation-unavailable"),
        ];
        let result = evaluate(&input);
        assert_eq!(result.status, AdmissionStatus::Invalid);
        assert!(!result.admission_allowed);
        assert_eq!(result.reasons.len(), 2);
    }

    #[test]
    fn cross_subject_or_intended_use_is_invalid() {
        let mut input = valid_input();
        input.capsule.binding.subject_digest = "sha256:other-subject".to_string();
        let result = evaluate(&input);
        assert_eq!(result.status, AdmissionStatus::Invalid);
        assert!(!result.admission_allowed);
    }

    #[test]
    fn changed_context_or_policy_requires_reverification() {
        let mut input = valid_input();
        input.capsule.binding.context_digest = "sha256:new-context".to_string();
        let result = evaluate(&input);
        assert_eq!(result.status, AdmissionStatus::Unknown);
        assert!(!result.admission_allowed);
    }

    #[test]
    fn duplicate_obligation_result_is_invalid() {
        let mut input = valid_input();
        input.capsule.obligations.push(evaluation(
            "payload-binding",
            ObligationOutcome::Valid,
            "second-verdict",
        ));
        let result = evaluate(&input);
        assert_eq!(result.status, AdmissionStatus::Invalid);
        assert!(result
            .reasons
            .iter()
            .any(|reason| reason.code == "duplicate_obligation_result"));
    }

    #[test]
    fn optional_unknown_is_preserved_but_does_not_block_required_admission() {
        let mut input = valid_input();
        input.capsule.obligations.push(ObligationEvaluation {
            id: "diagnostic-provenance".to_string(),
            outcome: ObligationOutcome::Unknown,
            reason_code: "diagnostic-source-unavailable".to_string(),
            evidence_ref: None,
        });
        let result = evaluate(&input);
        assert_eq!(result.status, AdmissionStatus::Valid);
        assert!(result.admission_allowed);
        assert!(result
            .obligations
            .iter()
            .any(|item| item.id == "diagnostic-provenance"
                && item.outcome == ObligationOutcome::Unknown));
    }

    #[test]
    fn empty_required_profile_is_invalid() {
        let mut input = valid_input();
        input.profile.required_obligations.clear();
        let result = evaluate(&input);
        assert_eq!(result.status, AdmissionStatus::Invalid);
    }

    #[test]
    fn unsupported_capsule_schema_is_invalid() {
        let mut input = valid_input();
        input.capsule.schema_version = CAPSULE_SCHEMA_VERSION + 1;
        let result = evaluate(&input);
        assert_eq!(result.status, AdmissionStatus::Invalid);
    }

    #[test]
    fn obligation_order_does_not_change_the_report() {
        let mut input = valid_input();
        input.capsule.obligations.push(evaluation(
            "diagnostic-provenance",
            ObligationOutcome::Unknown,
            "diagnostic-source-unavailable",
        ));
        let expected = evaluate(&input);
        input.capsule.obligations.reverse();
        let actual = evaluate(&input);
        assert_eq!(actual, expected);
    }

    #[test]
    fn unexpected_obligation_is_invalid() {
        let mut input = valid_input();
        input
            .capsule
            .obligations
            .push(evaluation("unregistered-obligation", ObligationOutcome::Valid, "ok"));
        let result = evaluate(&input);
        assert_eq!(result.status, AdmissionStatus::Invalid);
        assert!(result
            .reasons
            .iter()
            .any(|reason| reason.code == "unexpected_obligation"));
    }

    #[test]
    fn valid_mandatory_result_without_evidence_reference_is_unknown() {
        let mut input = valid_input();
        input.capsule.obligations[0].evidence_ref = None;
        let result = evaluate(&input);
        assert_eq!(result.status, AdmissionStatus::Unknown);
        assert!(result
            .reasons
            .iter()
            .any(|reason| reason.code == "missing_evidence_reference"));
    }
}
