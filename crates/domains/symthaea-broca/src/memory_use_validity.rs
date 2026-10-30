// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Decision-specific evidence use-validity reference model.
//!
//! A fact can be true, correctly sourced, and current while still being
//! ineligible for a particular decision. This module evaluates a deliberately
//! small set of typed governing-state contracts. It is a reference evaluator,
//! not a general scientific policy engine or an authorization verifier.
//!
//! Source eligibility is evaluated at the valid_at / known_at coordinates.
//! Decision-use constraints are evaluated only after every required source
//! revision passes temporal and exact-scope checks. Scope equality is not
//! authorization.
//!
//! The certificate is a deterministic integrity envelope, not a digital
//! signature. A deployment must independently verify source authenticity,
//! authorization receipts, policy approval, and rule provenance.

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

fn non_empty(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(format!("{field} must be non-empty and trimmed"));
    }
    Ok(())
}

fn valid_sha256(value: &str, field: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{field} must be 64 lowercase hexadecimal characters"));
    }
    Ok(())
}

/// Immutable source revision reference bound into a decision-use certificate.
/// The bitemporal source ledger remains authoritative for event history; this
/// projection is the minimal eligibility envelope consumed by this evaluator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceSourceRef {
    pub revision_id: String,
    pub payload_sha256: String,
    pub scope: String,
    pub valid_from: NaiveDate,
    /// Exclusive valid-time end.
    pub valid_to: Option<NaiveDate>,
    pub recorded_at: DateTime<Utc>,
    /// Transaction time at which this revision was retracted or superseded.
    pub retracted_at: Option<DateTime<Utc>>,
}

impl EvidenceSourceRef {
    fn validate(&self) -> Result<(), String> {
        non_empty(&self.revision_id, "source revision ID")?;
        non_empty(&self.scope, "source scope")?;
        valid_sha256(&self.payload_sha256, "source payload_sha256")?;
        if self.valid_to.is_some_and(|end| end <= self.valid_from) {
            return Err("source valid_to must be strictly after valid_from".into());
        }
        if self.retracted_at.is_some_and(|retracted| retracted < self.recorded_at) {
            return Err("source retracted_at cannot precede recorded_at".into());
        }
        Ok(())
    }

    fn valid_at(&self, date: NaiveDate) -> bool {
        self.valid_from <= date && self.valid_to.is_none_or(|end| date < end)
    }
}

/// The request makes the decision and its temporal/scope context explicit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceUseRequest {
    pub decision_id: String,
    pub requested_action: String,
    pub scope: String,
    pub valid_at: NaiveDate,
    pub known_at: DateTime<Utc>,
    pub rule_id: String,
    pub rule_version: String,
}

impl EvidenceUseRequest {
    fn validate(&self) -> Result<(), String> {
        for (field, value) in [
            ("decision_id", self.decision_id.as_str()),
            ("requested_action", self.requested_action.as_str()),
            ("scope", self.scope.as_str()),
            ("rule_id", self.rule_id.as_str()),
            ("rule_version", self.rule_version.as_str()),
        ] {
            non_empty(value, field)?;
        }
        Ok(())
    }
}

/// Distinguishes failed or absent measurements from actual measured zero.
/// Censored and below-detection results remain observations, but with limits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum ObservationState {
    NotRun,
    Failed,
    Censored { censoring_rule_id: String },
    BelowDetection { detection_limit_sha256: String },
    Undetected { detector_revision_id: String },
    MeasuredZero { measurement_id: String },
    MeasuredValue { measurement_id: String, value_sha256: String },
}

/// Decision-governing information needed for the first executable validity slice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GoverningEvidence {
    /// A decision maker who accessed the outcome before plan lock cannot
    /// treat the result as confirmatory under this simple reference rule.
    PlanLock {
        plan_locked_at: DateTime<Utc>,
        first_outcome_access_at: Option<DateTime<Utc>>,
        outcome_accessed_by_decision_maker: bool,
    },
    /// Selection/stopping metadata for a predeclared analysis family.
    OutcomeSelection {
        registered_family_size: u32,
        inspected_outcome_count: u32,
        selected_after_inspection: bool,
        stopping_rule_preregistered: bool,
    },
    /// Same-name metrics require identical definitions and units, or an
    /// explicit prevalidated conversion-receipt digest.
    MetricComparability {
        left_definition_sha256: String,
        right_definition_sha256: String,
        left_unit: String,
        right_unit: String,
        conversion_receipt_sha256: Option<String>,
    },
    Observation {
        state: ObservationState,
    },
    /// Duplicate source revisions do not increase the number of independent
    /// evidence families. The lineage digest must come from a trusted
    /// duplicate-lineage analyzer; this module only binds the digest.
    EvidenceIndependence {
        independent_family_ids: BTreeSet<String>,
        minimum_independent_families: u32,
        lineage_receipt_sha256: Option<String>,
    },
    /// Prospective choice intentionally contains no sunk-cost field.
    /// Historical spend must not affect this decision surface.
    ProspectiveChoice {
        alternatives: Vec<ProspectiveAlternative>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProspectiveAlternative {
    pub alternative_id: String,
    pub action_id: String,
    pub future_benefit_units: i64,
    pub future_cost_units: i64,
}

impl ProspectiveAlternative {
    fn net_future_value(&self) -> Result<i64, String> {
        self.future_benefit_units
            .checked_sub(self.future_cost_units)
            .ok_or_else(|| format!("future net value overflow for {:?}", self.alternative_id))
    }
}

impl GoverningEvidence {
    fn validate(&self) -> Result<(), String> {
        match self {
            Self::PlanLock { .. } => Ok(()),
            Self::OutcomeSelection {
                registered_family_size,
                inspected_outcome_count,
                ..
            } => {
                if *registered_family_size == 0 || *inspected_outcome_count == 0 {
                    return Err("outcome-selection counts must be non-zero".into());
                }
                Ok(())
            }
            Self::MetricComparability {
                left_definition_sha256,
                right_definition_sha256,
                left_unit,
                right_unit,
                conversion_receipt_sha256,
            } => {
                valid_sha256(left_definition_sha256, "left metric definition digest")?;
                valid_sha256(right_definition_sha256, "right metric definition digest")?;
                non_empty(left_unit, "left metric unit")?;
                non_empty(right_unit, "right metric unit")?;
                if let Some(receipt) = conversion_receipt_sha256 {
                    valid_sha256(receipt, "conversion receipt digest")?;
                }
                Ok(())
            }
            Self::Observation { state } => match state {
                ObservationState::NotRun | ObservationState::Failed => Ok(()),
                ObservationState::Censored { censoring_rule_id } => {
                    non_empty(censoring_rule_id, "censoring rule ID")
                }
                ObservationState::BelowDetection { detection_limit_sha256 } => {
                    valid_sha256(detection_limit_sha256, "detection-limit digest")
                }
                ObservationState::Undetected { detector_revision_id } => {
                    non_empty(detector_revision_id, "detector revision ID")
                }
                ObservationState::MeasuredZero { measurement_id } => {
                    non_empty(measurement_id, "measurement ID")
                }
                ObservationState::MeasuredValue {
                    measurement_id,
                    value_sha256,
                } => {
                    non_empty(measurement_id, "measurement ID")?;
                    valid_sha256(value_sha256, "measured value digest")
                }
            },
            Self::EvidenceIndependence {
                independent_family_ids,
                minimum_independent_families,
                lineage_receipt_sha256,
            } => {
                if *minimum_independent_families == 0 {
                    return Err("minimum independent family count must be non-zero".into());
                }
                if independent_family_ids.is_empty() {
                    return Err("independent family set must not be empty".into());
                }
                for family_id in independent_family_ids {
                    non_empty(family_id, "evidence family ID")?;
                }
                if let Some(receipt) = lineage_receipt_sha256 {
                    valid_sha256(receipt, "lineage receipt digest")?;
                }
                Ok(())
            }
            Self::ProspectiveChoice { alternatives } => {
                if alternatives.len() < 2 {
                    return Err("prospective choice requires at least two alternatives".into());
                }
                let mut ids = BTreeSet::new();
                for alternative in alternatives {
                    non_empty(&alternative.alternative_id, "alternative ID")?;
                    non_empty(&alternative.action_id, "alternative action ID")?;
                    if !ids.insert(alternative.alternative_id.as_str()) {
                        return Err(format!(
                            "duplicate prospective alternative ID {:?}",
                            alternative.alternative_id
                        ));
                    }
                    alternative.net_future_value()?;
                }
                Ok(())
            }
        }
    }
}

/// Separate states are intentional: none is interpreted as generic success or
/// generic failure, and all machine-readable reason codes are retained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UseValidityStatus {
    Admissible,
    AdmissibleWithLimitations,
    ExploratoryOnly,
    NotComparable,
    MissingOrFailed,
    Contested,
    NotKnownAtQueryTime,
    Retracted,
    OutsideValidity,
    OutOfScope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UseValidityReason {
    SourceNotKnownAtQueryTime,
    SourceRetractedByQueryTime,
    SourceOutsideValidTime,
    SourceScopeMismatch,
    OutcomeAccessBeforePlanLock,
    AdaptiveSelectionWithoutPreregisteredStoppingRule,
    InspectedMoreOutcomesThanRegistered,
    MetricDefinitionsDiffer,
    ConversionReceiptApplied,
    ObservationNotRun,
    ObservationFailed,
    ObservationCensored,
    ObservationBelowDetection,
    ObservationUndetected,
    IndependentFamilyCountInsufficient,
    EvidenceFamilyLineageUnverified,
    ProspectiveValueTie,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceUseAssessment {
    pub status: UseValidityStatus,
    pub reasons: Vec<UseValidityReason>,
    pub proposed_action: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceUseInput {
    pub source_revisions: Vec<EvidenceSourceRef>,
    pub governing_evidence: GoverningEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceUseCertificate {
    pub request: EvidenceUseRequest,
    pub source_revisions: Vec<EvidenceSourceRef>,
    pub governing_evidence: GoverningEvidence,
    pub governing_state_sha256: String,
    pub assessment: EvidenceUseAssessment,
    pub certificate_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct CertificatePayload<'a> {
    request: &'a EvidenceUseRequest,
    source_revisions: &'a [EvidenceSourceRef],
    governing_evidence: &'a GoverningEvidence,
    governing_state_sha256: &'a str,
    assessment: &'a EvidenceUseAssessment,
}

fn sha256_json<T: Serialize>(value: &T, field: &str) -> Result<String, String> {
    let encoded = serde_json::to_vec(value)
        .map_err(|error| format!("failed to serialize {field}: {error}"))?;
    Ok(format!("{:x}", Sha256::digest(encoded)))
}

fn canonical_source_revisions(
    request: &EvidenceUseRequest,
    sources: &[EvidenceSourceRef],
) -> Result<Vec<EvidenceSourceRef>, String> {
    if sources.is_empty() {
        return Err("at least one source revision is required".into());
    }
    let mut canonical = sources.to_vec();
    for source in &canonical {
        source.validate()?;
    }
    canonical.sort_by(|left, right| left.revision_id.cmp(&right.revision_id));
    if canonical.windows(2).any(|pair| pair[0].revision_id == pair[1].revision_id) {
        return Err("duplicate source revision ID in evidence-use input".into());
    }
    // Validate request scope as a partition key, but do not treat equality
    // as proof of authorization.
    non_empty(&request.scope, "request scope")?;
    Ok(canonical)
}

fn source_eligibility(
    request: &EvidenceUseRequest,
    sources: &[EvidenceSourceRef],
) -> Option<EvidenceUseAssessment> {
    let mut reasons = BTreeSet::new();
    for source in sources {
        if source.recorded_at > request.known_at {
            reasons.insert(UseValidityReason::SourceNotKnownAtQueryTime);
        }
        if source.retracted_at.is_some_and(|retracted| retracted <= request.known_at) {
            reasons.insert(UseValidityReason::SourceRetractedByQueryTime);
        }
        if source.scope != request.scope {
            reasons.insert(UseValidityReason::SourceScopeMismatch);
        }
        if !source.valid_at(request.valid_at) {
            reasons.insert(UseValidityReason::SourceOutsideValidTime);
        }
    }
    if reasons.is_empty() {
        return None;
    }

    // Retain every detected defect, but choose one deterministic primary state.
    let status = if reasons.contains(&UseValidityReason::SourceNotKnownAtQueryTime) {
        UseValidityStatus::NotKnownAtQueryTime
    } else if reasons.contains(&UseValidityReason::SourceRetractedByQueryTime) {
        UseValidityStatus::Retracted
    } else if reasons.contains(&UseValidityReason::SourceScopeMismatch) {
        UseValidityStatus::OutOfScope
    } else {
        UseValidityStatus::OutsideValidity
    };
    Some(EvidenceUseAssessment {
        status,
        reasons: reasons.into_iter().collect(),
        proposed_action: None,
    })
}

fn assess_governing_evidence(evidence: &GoverningEvidence) -> Result<EvidenceUseAssessment, String> {
    let mut reasons = BTreeSet::new();
    let mut proposed_action = None;

    let status = match evidence {
        GoverningEvidence::PlanLock {
            plan_locked_at,
            first_outcome_access_at,
            outcome_accessed_by_decision_maker,
        } => {
            if *outcome_accessed_by_decision_maker
                && first_outcome_access_at.is_some_and(|access| access < *plan_locked_at)
            {
                reasons.insert(UseValidityReason::OutcomeAccessBeforePlanLock);
                UseValidityStatus::ExploratoryOnly
            } else {
                UseValidityStatus::Admissible
            }
        }
        GoverningEvidence::OutcomeSelection {
            registered_family_size,
            inspected_outcome_count,
            selected_after_inspection,
            stopping_rule_preregistered,
        } => {
            if inspected_outcome_count > registered_family_size {
                reasons.insert(UseValidityReason::InspectedMoreOutcomesThanRegistered);
                UseValidityStatus::ExploratoryOnly
            } else if *selected_after_inspection && !*stopping_rule_preregistered {
                reasons.insert(UseValidityReason::AdaptiveSelectionWithoutPreregisteredStoppingRule);
                UseValidityStatus::ExploratoryOnly
            } else if *selected_after_inspection {
                reasons.insert(UseValidityReason::AdaptiveSelectionWithoutPreregisteredStoppingRule);
                UseValidityStatus::AdmissibleWithLimitations
            } else {
                UseValidityStatus::Admissible
            }
        }
        GoverningEvidence::MetricComparability {
            left_definition_sha256,
            right_definition_sha256,
            left_unit,
            right_unit,
            conversion_receipt_sha256,
        } => {
            if left_definition_sha256 == right_definition_sha256 && left_unit == right_unit {
                UseValidityStatus::Admissible
            } else if conversion_receipt_sha256.is_some() {
                reasons.insert(UseValidityReason::MetricDefinitionsDiffer);
                reasons.insert(UseValidityReason::ConversionReceiptApplied);
                UseValidityStatus::AdmissibleWithLimitations
            } else {
                reasons.insert(UseValidityReason::MetricDefinitionsDiffer);
                UseValidityStatus::NotComparable
            }
        }
        GoverningEvidence::Observation { state } => match state {
            ObservationState::NotRun => {
                reasons.insert(UseValidityReason::ObservationNotRun);
                UseValidityStatus::MissingOrFailed
            }
            ObservationState::Failed => {
                reasons.insert(UseValidityReason::ObservationFailed);
                UseValidityStatus::MissingOrFailed
            }
            ObservationState::Censored { .. } => {
                reasons.insert(UseValidityReason::ObservationCensored);
                UseValidityStatus::AdmissibleWithLimitations
            }
            ObservationState::BelowDetection { .. } => {
                reasons.insert(UseValidityReason::ObservationBelowDetection);
                UseValidityStatus::AdmissibleWithLimitations
            }
            ObservationState::Undetected { .. } => {
                reasons.insert(UseValidityReason::ObservationUndetected);
                UseValidityStatus::AdmissibleWithLimitations
            }
            ObservationState::MeasuredZero { .. } | ObservationState::MeasuredValue { .. } => {
                UseValidityStatus::Admissible
            }
        },
        GoverningEvidence::EvidenceIndependence {
            independent_family_ids,
            minimum_independent_families,
            lineage_receipt_sha256,
        } => {
            if lineage_receipt_sha256.is_none() {
                reasons.insert(UseValidityReason::EvidenceFamilyLineageUnverified);
                UseValidityStatus::Contested
            } else if independent_family_ids.len() < *minimum_independent_families as usize {
                reasons.insert(UseValidityReason::IndependentFamilyCountInsufficient);
                UseValidityStatus::ExploratoryOnly
            } else {
                UseValidityStatus::Admissible
            }
        }
        GoverningEvidence::ProspectiveChoice { alternatives } => {
            let mut ranked: Vec<_> = alternatives
                .iter()
                .map(|alternative| {
                    Ok((
                        alternative.net_future_value()?,
                        alternative.alternative_id.as_str(),
                        alternative.action_id.as_str(),
                    ))
                })
                .collect::<Result<_, String>>()?;
            ranked.sort_by(|left, right| {
                right.0.cmp(&left.0).then_with(|| left.1.cmp(right.1))
            });
            if ranked[0].0 == ranked[1].0 {
                reasons.insert(UseValidityReason::ProspectiveValueTie);
                UseValidityStatus::Contested
            } else {
                proposed_action = Some(ranked[0].2.to_owned());
                UseValidityStatus::Admissible
            }
        }
    };

    Ok(EvidenceUseAssessment {
        status,
        reasons: reasons.into_iter().collect(),
        proposed_action,
    })
}

/// Compile a certificate from source lineage and one typed governing-state
/// contract. Every required source must be eligible at (valid_at, known_at).
pub fn assess_evidence_use(
    request: EvidenceUseRequest,
    input: EvidenceUseInput,
) -> Result<EvidenceUseCertificate, String> {
    request.validate()?;
    input.governing_evidence.validate()?;
    let source_revisions = canonical_source_revisions(&request, &input.source_revisions)?;

    let assessment = match source_eligibility(&request, &source_revisions) {
        Some(source_failure) => source_failure,
        None => assess_governing_evidence(&input.governing_evidence)?,
    };
    let governing_state_sha256 = sha256_json(&input.governing_evidence, "governing evidence")?;
    let payload = CertificatePayload {
        request: &request,
        source_revisions: &source_revisions,
        governing_evidence: &input.governing_evidence,
        governing_state_sha256: &governing_state_sha256,
        assessment: &assessment,
    };
    let certificate_sha256 = sha256_json(&payload, "evidence-use certificate")?;

    Ok(EvidenceUseCertificate {
        request,
        source_revisions,
        governing_evidence: input.governing_evidence,
        governing_state_sha256,
        assessment,
        certificate_sha256,
    })
}

/// Independently re-evaluate the public reference semantics and compare every
/// certificate field. This detects mutation/tampering; it is not a signature
/// or authenticity check and must not replace those production controls.
pub fn verify_evidence_use_certificate(
    certificate: &EvidenceUseCertificate,
) -> Result<(), String> {
    let reconstructed = assess_evidence_use(
        certificate.request.clone(),
        EvidenceUseInput {
            source_revisions: certificate.source_revisions.clone(),
            governing_evidence: certificate.governing_evidence.clone(),
        },
    )?;
    if &reconstructed != certificate {
        return Err("evidence-use certificate differs from independently recomputed result".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(value: &str) -> NaiveDate {
        NaiveDate::parse_from_str(value, "%Y-%m-%d").unwrap()
    }

    fn instant(value: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(value).unwrap().with_timezone(&Utc)
    }

    fn source(id: &str) -> EvidenceSourceRef {
        EvidenceSourceRef {
            revision_id: id.into(),
            payload_sha256: "a".repeat(64),
            scope: "tenant-1".into(),
            valid_from: date("2026-01-01"),
            valid_to: None,
            recorded_at: instant("2026-01-02T00:00:00Z"),
            retracted_at: None,
        }
    }

    fn request() -> EvidenceUseRequest {
        EvidenceUseRequest {
            decision_id: "decision-1".into(),
            requested_action: "publish-confirmatory-claim".into(),
            scope: "tenant-1".into(),
            valid_at: date("2026-02-01"),
            known_at: instant("2026-02-02T00:00:00Z"),
            rule_id: "use-validity-v1".into(),
            rule_version: "1".into(),
        }
    }

    fn assess(state: GoverningEvidence) -> EvidenceUseCertificate {
        assess_evidence_use(
            request(),
            EvidenceUseInput {
                source_revisions: vec![source("revision-1")],
                governing_evidence: state,
            },
        )
        .unwrap()
    }

    #[test]
    fn outcome_access_before_plan_lock_is_exploratory() {
        let result = assess(GoverningEvidence::PlanLock {
            plan_locked_at: instant("2026-01-03T00:00:00Z"),
            first_outcome_access_at: Some(instant("2026-01-02T00:00:00Z")),
            outcome_accessed_by_decision_maker: true,
        });
        assert_eq!(result.assessment.status, UseValidityStatus::ExploratoryOnly);
        assert_eq!(
            result.assessment.reasons,
            vec![UseValidityReason::OutcomeAccessBeforePlanLock]
        );
    }

    #[test]
    fn plan_lock_flip_changes_only_admissibility() {
        let early = assess(GoverningEvidence::PlanLock {
            plan_locked_at: instant("2026-01-03T00:00:00Z"),
            first_outcome_access_at: Some(instant("2026-01-02T00:00:00Z")),
            outcome_accessed_by_decision_maker: true,
        });
        let locked_first = assess(GoverningEvidence::PlanLock {
            plan_locked_at: instant("2026-01-01T00:00:00Z"),
            first_outcome_access_at: Some(instant("2026-01-02T00:00:00Z")),
            outcome_accessed_by_decision_maker: true,
        });
        assert_eq!(early.assessment.status, UseValidityStatus::ExploratoryOnly);
        assert_eq!(locked_first.assessment.status, UseValidityStatus::Admissible);
    }

    #[test]
    fn adaptive_selection_requires_a_preregistered_stopping_rule() {
        let result = assess(GoverningEvidence::OutcomeSelection {
            registered_family_size: 20,
            inspected_outcome_count: 20,
            selected_after_inspection: true,
            stopping_rule_preregistered: false,
        });
        assert_eq!(result.assessment.status, UseValidityStatus::ExploratoryOnly);
    }

    #[test]
    fn inspected_outcomes_cannot_exceed_the_registered_family() {
        let result = assess(GoverningEvidence::OutcomeSelection {
            registered_family_size: 10,
            inspected_outcome_count: 20,
            selected_after_inspection: false,
            stopping_rule_preregistered: true,
        });
        assert_eq!(result.assessment.status, UseValidityStatus::ExploratoryOnly);
        assert!(result.assessment.reasons.contains(
            &UseValidityReason::InspectedMoreOutcomesThanRegistered
        ));
    }

    #[test]
    fn incompatible_metric_definitions_are_not_comparable_without_receipt() {
        let result = assess(GoverningEvidence::MetricComparability {
            left_definition_sha256: "a".repeat(64),
            right_definition_sha256: "b".repeat(64),
            left_unit: "percent".into(),
            right_unit: "ratio".into(),
            conversion_receipt_sha256: None,
        });
        assert_eq!(result.assessment.status, UseValidityStatus::NotComparable);
    }

    #[test]
    fn metric_conversion_receipt_downgrades_to_limited_comparability() {
        let result = assess(GoverningEvidence::MetricComparability {
            left_definition_sha256: "a".repeat(64),
            right_definition_sha256: "b".repeat(64),
            left_unit: "percent".into(),
            right_unit: "ratio".into(),
            conversion_receipt_sha256: Some("c".repeat(64)),
        });
        assert_eq!(
            result.assessment.status,
            UseValidityStatus::AdmissibleWithLimitations
        );
    }

    #[test]
    fn failed_observation_is_not_a_measured_zero() {
        let failed = assess(GoverningEvidence::Observation {
            state: ObservationState::Failed,
        });
        let zero = assess(GoverningEvidence::Observation {
            state: ObservationState::MeasuredZero {
                measurement_id: "measurement-0".into(),
            },
        });
        assert_eq!(failed.assessment.status, UseValidityStatus::MissingOrFailed);
        assert_eq!(zero.assessment.status, UseValidityStatus::Admissible);
    }

    #[test]
    fn censored_observation_is_present_but_limited() {
        let result = assess(GoverningEvidence::Observation {
            state: ObservationState::Censored {
                censoring_rule_id: "right-censor-v1".into(),
            },
        });
        assert_eq!(
            result.assessment.status,
            UseValidityStatus::AdmissibleWithLimitations
        );
    }

    #[test]
    fn duplicate_source_copies_do_not_count_as_independent_families() {
        let result = assess(GoverningEvidence::EvidenceIndependence {
            independent_family_ids: BTreeSet::from(["family-A".into()]),
            minimum_independent_families: 3,
            lineage_receipt_sha256: Some("d".repeat(64)),
        });
        assert_eq!(result.assessment.status, UseValidityStatus::ExploratoryOnly);
    }

    #[test]
    fn evidence_family_lineage_without_a_receipt_is_contested() {
        let result = assess(GoverningEvidence::EvidenceIndependence {
            independent_family_ids: BTreeSet::from([
                "family-A".into(),
                "family-B".into(),
                "family-C".into(),
            ]),
            minimum_independent_families: 3,
            lineage_receipt_sha256: None,
        });
        assert_eq!(result.assessment.status, UseValidityStatus::Contested);
    }

    #[test]
    fn prospective_choice_uses_future_net_value_only() {
        let result = assess(GoverningEvidence::ProspectiveChoice {
            alternatives: vec![
                ProspectiveAlternative {
                    alternative_id: "A".into(),
                    action_id: "choose-A".into(),
                    future_benefit_units: 40,
                    future_cost_units: 20,
                },
                ProspectiveAlternative {
                    alternative_id: "B".into(),
                    action_id: "choose-B".into(),
                    future_benefit_units: 45,
                    future_cost_units: 30,
                },
            ],
        });
        assert_eq!(result.assessment.status, UseValidityStatus::Admissible);
        assert_eq!(result.assessment.proposed_action.as_deref(), Some("choose-A"));
    }

    #[test]
    fn equal_prospective_values_are_contested_not_tie_broken_by_id() {
        let result = assess(GoverningEvidence::ProspectiveChoice {
            alternatives: vec![
                ProspectiveAlternative {
                    alternative_id: "A".into(),
                    action_id: "choose-A".into(),
                    future_benefit_units: 40,
                    future_cost_units: 20,
                },
                ProspectiveAlternative {
                    alternative_id: "B".into(),
                    action_id: "choose-B".into(),
                    future_benefit_units: 50,
                    future_cost_units: 30,
                },
            ],
        });
        assert_eq!(result.assessment.status, UseValidityStatus::Contested);
        assert_eq!(result.assessment.proposed_action, None);
    }

    #[test]
    fn future_source_revision_cannot_leak_into_as_known_decision() {
        let mut input = EvidenceUseInput {
            source_revisions: vec![source("revision-1")],
            governing_evidence: GoverningEvidence::Observation {
                state: ObservationState::MeasuredZero {
                    measurement_id: "measurement-0".into(),
                },
            },
        };
        input.source_revisions[0].recorded_at = instant("2026-03-01T00:00:00Z");
        let result = assess_evidence_use(request(), input).unwrap();
        assert_eq!(result.assessment.status, UseValidityStatus::NotKnownAtQueryTime);
    }

    #[test]
    fn delayed_retraction_is_visible_only_after_its_transaction_time() {
        let mut input = EvidenceUseInput {
            source_revisions: vec![source("revision-1")],
            governing_evidence: GoverningEvidence::Observation {
                state: ObservationState::MeasuredZero {
                    measurement_id: "measurement-0".into(),
                },
            },
        };
        input.source_revisions[0].retracted_at = Some(instant("2026-03-01T00:00:00Z"));
        let before = assess_evidence_use(request(), input.clone()).unwrap();
        assert_eq!(before.assessment.status, UseValidityStatus::Admissible);

        let mut after_request = request();
        after_request.known_at = instant("2026-03-02T00:00:00Z");
        let after = assess_evidence_use(after_request, input).unwrap();
        assert_eq!(after.assessment.status, UseValidityStatus::Retracted);
    }

    #[test]
    fn valid_time_uses_half_open_interval_boundaries() {
        let mut input = EvidenceUseInput {
            source_revisions: vec![source("revision-1")],
            governing_evidence: GoverningEvidence::Observation {
                state: ObservationState::MeasuredZero {
                    measurement_id: "measurement-0".into(),
                },
            },
        };
        input.source_revisions[0].valid_from = date("2026-02-01");
        input.source_revisions[0].valid_to = Some(date("2026-03-01"));
        let before = assess_evidence_use(
            EvidenceUseRequest { valid_at: date("2026-01-31"), ..request() },
            input.clone(),
        ).unwrap();
        let start = assess_evidence_use(
            EvidenceUseRequest { valid_at: date("2026-02-01"), ..request() },
            input.clone(),
        ).unwrap();
        let end = assess_evidence_use(
            EvidenceUseRequest { valid_at: date("2026-03-01"), ..request() },
            input,
        ).unwrap();

        assert_eq!(before.assessment.status, UseValidityStatus::OutsideValidity);
        assert_eq!(start.assessment.status, UseValidityStatus::Admissible);
        assert_eq!(end.assessment.status, UseValidityStatus::OutsideValidity);
    }

    #[test]
    fn exact_scope_mismatch_fails_closed() {
        let mut input = EvidenceUseInput {
            source_revisions: vec![source("revision-1")],
            governing_evidence: GoverningEvidence::Observation {
                state: ObservationState::MeasuredZero {
                    measurement_id: "measurement-0".into(),
                },
            },
        };
        input.source_revisions[0].scope = "tenant-2".into();
        let result = assess_evidence_use(request(), input).unwrap();
        assert_eq!(result.assessment.status, UseValidityStatus::OutOfScope);
    }

    #[test]
    fn source_order_does_not_change_certificate_digest() {
        let mut input = EvidenceUseInput {
            source_revisions: vec![source("revision-B"), source("revision-A")],
            governing_evidence: GoverningEvidence::Observation {
                state: ObservationState::MeasuredZero {
                    measurement_id: "measurement-0".into(),
                },
            },
        };
        let left = assess_evidence_use(request(), input.clone()).unwrap();
        input.source_revisions.reverse();
        let right = assess_evidence_use(request(), input).unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn independent_verifier_rejects_certificate_mutation() {
        let mut result = assess(GoverningEvidence::Observation {
            state: ObservationState::MeasuredZero {
                measurement_id: "measurement-0".into(),
            },
        });
        verify_evidence_use_certificate(&result).unwrap();
        result.assessment.status = UseValidityStatus::AdmissibleWithLimitations;
        assert!(verify_evidence_use_certificate(&result).is_err());
    }

    #[test]
    fn duplicate_source_revision_ids_are_rejected() {
        let result = assess_evidence_use(
            request(),
            EvidenceUseInput {
                source_revisions: vec![source("revision-1"), source("revision-1")],
                governing_evidence: GoverningEvidence::Observation {
                    state: ObservationState::MeasuredZero {
                        measurement_id: "measurement-0".into(),
                    },
                },
            },
        );
        assert!(result.unwrap_err().contains("duplicate source revision ID"));
    }

    #[test]
    fn invalid_metric_digest_is_rejected_before_certification() {
        let result = assess_evidence_use(
            request(),
            EvidenceUseInput {
                source_revisions: vec![source("revision-1")],
                governing_evidence: GoverningEvidence::MetricComparability {
                    left_definition_sha256: "bad".into(),
                    right_definition_sha256: "b".repeat(64),
                    left_unit: "seconds".into(),
                    right_unit: "seconds".into(),
                    conversion_receipt_sha256: None,
                },
            },
        );
        assert!(result.unwrap_err().contains("left metric definition digest"));
    }
}
