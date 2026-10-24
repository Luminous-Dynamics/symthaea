// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed evidence gates for aviation public-data benchmarks.
//!
//! This module checks metadata contracts, not the truth of a source system.
//! Passing means only that supplied metadata is internally consistent. It is
//! not proof that timestamps are authentic or that a feature is correctly derived.

use std::collections::BTreeSet;

/// Semantic role of a field in the frozen model-input schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldRole {
    /// Permitted predictor, subject to availability-time checks.
    ModelInput,
    /// Identity/provenance field; not a predictor by default.
    IdentifierOnly,
    /// Ex-post label used for scoring, never a predictor.
    TargetOnly,
    /// Value produced during or after the operation.
    PostEventOutcome,
}

/// One field declaration from a frozen feature schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureSpec {
    pub name: String,
    pub role: FieldRole,
    /// Earliest instant this value was available to the decision-maker.
    /// None means availability has not been evidenced.
    pub available_at_epoch_seconds: Option<i64>,
}

/// Machine-readable result status for a feature availability audit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditStatus {
    /// All supplied fields satisfy the declared schema contract.
    Accept,
    /// The schema contains a known violation; fail closed.
    Reject,
    /// Required evidence is absent; do not promote the run as accepted.
    Indeterminate,
}

/// Reason a feature audit did not accept the schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditCode {
    InvalidDecisionTime,
    EmptyFeatureSet,
    EmptyFieldName,
    DuplicateFieldName,
    NonInputFieldInModelSchema,
    MissingAvailabilityTimestamp,
    InvalidAvailabilityTimestamp,
    FeatureAvailableAfterDecision,
}

/// A specific field-level issue found during validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditFinding {
    pub field: String,
    pub code: AuditCode,
}

/// Result of a feature-availability contract check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureAuditReport {
    pub status: AuditStatus,
    pub findings: Vec<AuditFinding>,
}

/// Check that each model-input field was available by the declared decision time.
///
/// Availability is checked against the decision instant, not flight date,
/// scheduled departure, or the timestamp when an outcome was eventually stored.
/// A known target/outcome/identifier in the model schema is rejected even if
/// its timestamp precedes the cutoff. Missing timestamps yield Indeterminate,
/// never Accept. A future timestamp is a hard rejection.
pub fn audit_feature_availability(
    decision_time_epoch_seconds: i64,
    features: &[FeatureSpec],
) -> FeatureAuditReport {
    let mut findings = Vec::new();

    if decision_time_epoch_seconds < 0 {
        findings.push(AuditFinding {
            field: String::new(),
            code: AuditCode::InvalidDecisionTime,
        });
    }
    if features.is_empty() {
        findings.push(AuditFinding {
            field: String::new(),
            code: AuditCode::EmptyFeatureSet,
        });
    }

    let mut names = BTreeSet::new();
    for feature in features {
        let name = feature.name.trim();
        if name.is_empty() {
            findings.push(AuditFinding {
                field: feature.name.clone(),
                code: AuditCode::EmptyFieldName,
            });
        }
        if !names.insert(name.to_owned()) {
            findings.push(AuditFinding {
                field: feature.name.clone(),
                code: AuditCode::DuplicateFieldName,
            });
        }
        if feature.role != FieldRole::ModelInput {
            findings.push(AuditFinding {
                field: feature.name.clone(),
                code: AuditCode::NonInputFieldInModelSchema,
            });
        }

        match feature.available_at_epoch_seconds {
            None => findings.push(AuditFinding {
                field: feature.name.clone(),
                code: AuditCode::MissingAvailabilityTimestamp,
            }),
            Some(ts) if ts < 0 => findings.push(AuditFinding {
                field: feature.name.clone(),
                code: AuditCode::InvalidAvailabilityTimestamp,
            }),
            Some(ts) if ts > decision_time_epoch_seconds => findings.push(AuditFinding {
                field: feature.name.clone(),
                code: AuditCode::FeatureAvailableAfterDecision,
            }),
            Some(_) => {}
        }
    }

    let rejected = findings.iter().any(|finding| {
        matches!(
            finding.code,
            AuditCode::InvalidDecisionTime
                | AuditCode::EmptyFeatureSet
                | AuditCode::EmptyFieldName
                | AuditCode::DuplicateFieldName
                | AuditCode::NonInputFieldInModelSchema
                | AuditCode::InvalidAvailabilityTimestamp
                | AuditCode::FeatureAvailableAfterDecision
        )
    });
    let indeterminate = findings
        .iter()
        .any(|finding| finding.code == AuditCode::MissingAvailabilityTimestamp);

    let status = if rejected {
        AuditStatus::Reject
    } else if indeterminate {
        AuditStatus::Indeterminate
    } else {
        AuditStatus::Accept
    };

    FeatureAuditReport { status, findings }
}

/// Evidence boundary for a benchmark result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceTier {
    /// Public retrospective records, such as BTS flight outcomes.
    PublicRetrospective,
    /// Synthetic cases evaluated under a declared simulator/action-effect model.
    SyntheticReplay,
    /// Authorized operator logs and outcome records.
    AuthorizedOperator,
}

/// Claim requested for a result artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimKind {
    /// Predictive performance against observed held-out labels.
    ForecastQuality,
    /// Policy value conditional on a declared objective and action-effect model.
    ConditionalPolicyReplayValue,
    /// A causal statement that an intervention changed a real outcome.
    OperationalEffect,
}

/// Fail-closed decision from the claim/evidence boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimDecision {
    Permitted,
    RejectedWrongEvidenceTier,
    IndeterminateMissingActionEffectModel,
    IndeterminateMissingOperatorAuthorization,
    IndeterminateMissingCausalIdentification,
}

/// Evidence flags attached to a run manifest or release review.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClaimEvidence {
    pub tier: EvidenceTier,
    pub action_effect_model_declared: bool,
    pub operator_authorization_confirmed: bool,
    pub causal_identification_present: bool,
}

/// Decide whether a claim is compatible with the declared evidence category.
///
/// This is a metadata gate, not an independent audit of the data, causal design,
/// or simulator. Permitted means only that the claim is compatible with the
/// declared evidence class; the underlying evidence must still be reviewed.
pub fn evaluate_claim(kind: ClaimKind, evidence: ClaimEvidence) -> ClaimDecision {
    match kind {
        ClaimKind::ForecastQuality => match evidence.tier {
            EvidenceTier::PublicRetrospective | EvidenceTier::AuthorizedOperator => {
                ClaimDecision::Permitted
            }
            EvidenceTier::SyntheticReplay => ClaimDecision::RejectedWrongEvidenceTier,
        },
        ClaimKind::ConditionalPolicyReplayValue => {
            if !evidence.action_effect_model_declared {
                ClaimDecision::IndeterminateMissingActionEffectModel
            } else {
                match evidence.tier {
                    EvidenceTier::SyntheticReplay | EvidenceTier::AuthorizedOperator => {
                        ClaimDecision::Permitted
                    }
                    EvidenceTier::PublicRetrospective => ClaimDecision::RejectedWrongEvidenceTier,
                }
            }
        }
        ClaimKind::OperationalEffect => {
            if evidence.tier != EvidenceTier::AuthorizedOperator {
                ClaimDecision::RejectedWrongEvidenceTier
            } else if !evidence.operator_authorization_confirmed {
                ClaimDecision::IndeterminateMissingOperatorAuthorization
            } else if !evidence.causal_identification_present {
                ClaimDecision::IndeterminateMissingCausalIdentification
            } else {
                ClaimDecision::Permitted
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CUTOFF: i64 = 1_800_000_000;

    fn feature(name: &str, available_at: Option<i64>) -> FeatureSpec {
        FeatureSpec {
            name: name.to_owned(),
            role: FieldRole::ModelInput,
            available_at_epoch_seconds: available_at,
        }
    }

    #[test]
    fn accepts_inputs_available_by_decision_time() {
        let report = audit_feature_availability(
            CUTOFF,
            &[
                feature("scheduled_departure", Some(CUTOFF - 60)),
                feature("historical_route_rate", Some(CUTOFF)),
            ],
        );
        assert_eq!(report.status, AuditStatus::Accept);
        assert!(report.findings.is_empty());
    }

    #[test]
    fn future_feature_canary_is_rejected() {
        let report =
            audit_feature_availability(CUTOFF, &[feature("actual_arrival_delay", Some(CUTOFF + 1))]);
        assert_eq!(report.status, AuditStatus::Reject);
        assert!(report
            .findings
            .iter()
            .any(|f| f.code == AuditCode::FeatureAvailableAfterDecision));
    }

    #[test]
    fn outcome_roles_are_rejected_even_with_old_timestamps() {
        let mut outcome = feature("cancelled", Some(CUTOFF - 500));
        outcome.role = FieldRole::TargetOnly;
        let report = audit_feature_availability(CUTOFF, &[outcome]);
        assert_eq!(report.status, AuditStatus::Reject);
        assert!(report
            .findings
            .iter()
            .any(|f| f.code == AuditCode::NonInputFieldInModelSchema));
    }

    #[test]
    fn missing_availability_evidence_is_indeterminate() {
        let report = audit_feature_availability(CUTOFF, &[feature("route_history", None)]);
        assert_eq!(report.status, AuditStatus::Indeterminate);
        assert!(report
            .findings
            .iter()
            .any(|f| f.code == AuditCode::MissingAvailabilityTimestamp));
    }

    #[test]
    fn duplicate_and_blank_names_are_rejected() {
        let report = audit_feature_availability(
            CUTOFF,
            &[
                feature("airport_history", Some(CUTOFF)),
                feature("airport_history", Some(CUTOFF)),
                feature("  ", Some(CUTOFF)),
            ],
        );
        assert_eq!(report.status, AuditStatus::Reject);
        assert!(report.findings.iter().any(|f| f.code == AuditCode::DuplicateFieldName));
        assert!(report.findings.iter().any(|f| f.code == AuditCode::EmptyFieldName));
    }

    #[test]
    fn public_observations_do_not_authorize_a_causal_operational_claim() {
        let evidence = ClaimEvidence {
            tier: EvidenceTier::PublicRetrospective,
            action_effect_model_declared: false,
            operator_authorization_confirmed: false,
            causal_identification_present: false,
        };
        assert_eq!(
            evaluate_claim(ClaimKind::OperationalEffect, evidence),
            ClaimDecision::RejectedWrongEvidenceTier
        );
    }

    #[test]
    fn synthetic_replay_value_requires_an_action_effect_model() {
        let evidence = ClaimEvidence {
            tier: EvidenceTier::SyntheticReplay,
            action_effect_model_declared: false,
            operator_authorization_confirmed: false,
            causal_identification_present: false,
        };
        assert_eq!(
            evaluate_claim(ClaimKind::ConditionalPolicyReplayValue, evidence),
            ClaimDecision::IndeterminateMissingActionEffectModel
        );
    }

    #[test]
    fn operator_data_without_causal_design_is_not_an_operational_effect_claim() {
        let evidence = ClaimEvidence {
            tier: EvidenceTier::AuthorizedOperator,
            action_effect_model_declared: true,
            operator_authorization_confirmed: true,
            causal_identification_present: false,
        };
        assert_eq!(
            evaluate_claim(ClaimKind::OperationalEffect, evidence),
            ClaimDecision::IndeterminateMissingCausalIdentification
        );
    }

    #[test]
    fn authorized_operator_data_with_causal_design_can_support_claim_review() {
        let evidence = ClaimEvidence {
            tier: EvidenceTier::AuthorizedOperator,
            action_effect_model_declared: true,
            operator_authorization_confirmed: true,
            causal_identification_present: true,
        };
        assert_eq!(
            evaluate_claim(ClaimKind::OperationalEffect, evidence),
            ClaimDecision::Permitted
        );
    }
}
