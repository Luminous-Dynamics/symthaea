// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Prospective forecast-attempt evidence protocol.
//!
//! The lower-level v2 schema provides time-neutral forecast payloads and generic
//! commitment/resolution records. This module adds the stricter protocol needed
//! for genuinely prospective evaluation:
//!
//! - the evaluation/scoring protocol is committed before an outcome exists;
//! - a forecast attempt is either a distribution or a typed abstention;
//! - abstentions remain in the evidence population instead of disappearing;
//! - a later resolution has no API for choosing a scoring rule post-outcome;
//! - replayed resolutions can be validated against the exact attempt metadata.
//!
//! As with v2, these types do not themselves provide cryptographic wall-clock
//! precedence. A durable registry still needs immutable unique IDs and an
//! append-only/content-addressed persistence boundary.

use std::cmp::Ordering;
use std::fmt;

use serde::{Deserialize, Deserializer, Serialize};
use symthaea_futures_core::{AbstentionReason, ForecastPayload, OutcomeRegion, OutcomeSpaceId};

use crate::v2::{
    ExternalReference, ForecastCommitmentId, ForecastCoordinate, ForecastResolutionId,
    ForecastWindow, LedgerLabel, LedgerV2Error, ObservationLineage, ResolutionScore,
};

#[derive(Debug, Clone, PartialEq)]
pub enum ProspectiveError {
    Ledger(LedgerV2Error),
    AttemptDecisionMismatch,
    AttemptIdMismatch,
    ForecastTargetMismatch,
    OutcomeSpaceMismatch,
    EvaluationProtocolMismatch,
    AbstentionReasonMismatch,
}

impl fmt::Display for ProspectiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ledger(error) => write!(f, "{error}"),
            Self::AttemptDecisionMismatch => {
                write!(f, "resolution kind does not match the committed forecast attempt")
            }
            Self::AttemptIdMismatch => write!(f, "resolution references a different attempt id"),
            Self::ForecastTargetMismatch => {
                write!(f, "resolution forecast target differs from the committed target")
            }
            Self::OutcomeSpaceMismatch => {
                write!(f, "resolution outcome space differs from the committed forecast")
            }
            Self::EvaluationProtocolMismatch => {
                write!(f, "resolution evaluation protocol differs from the committed protocol")
            }
            Self::AbstentionReasonMismatch => {
                write!(f, "resolution abstention reason differs from the committed reason")
            }
        }
    }
}

impl std::error::Error for ProspectiveError {}

impl From<LedgerV2Error> for ProspectiveError {
    fn from(value: LedgerV2Error) -> Self {
        Self::Ledger(value)
    }
}

fn compare_coordinates(
    first: &ForecastCoordinate,
    second: &ForecastCoordinate,
) -> Result<Ordering, ProspectiveError> {
    match (first, second) {
        (ForecastCoordinate::SimulationTick(a), ForecastCoordinate::SimulationTick(b)) => {
            Ok(a.cmp(b))
        }
        (ForecastCoordinate::UnixMillis(a), ForecastCoordinate::UnixMillis(b)) => Ok(a.cmp(b)),
        (
            ForecastCoordinate::Ordinal {
                axis: first_axis,
                index: first_index,
            },
            ForecastCoordinate::Ordinal {
                axis: second_axis,
                index: second_index,
            },
        ) if first_axis == second_axis => Ok(first_index.cmp(second_index)),
        _ => Err(LedgerV2Error::TimeAxisMismatch.into()),
    }
}

/// Evaluation choices that must be fixed before the outcome exists.
///
/// `protocol_version` can identify the complete evaluation specification while
/// `scoring_rule` makes the primary proper scoring rule explicit. The abstention
/// policy is separately versioned because selection/coverage statistics are not
/// the same quantity as conditional forecast accuracy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationProtocol {
    protocol_version: LedgerLabel,
    scoring_rule: LedgerLabel,
    abstention_policy_version: LedgerLabel,
}

impl EvaluationProtocol {
    pub fn new(
        protocol_version: impl Into<String>,
        scoring_rule: impl Into<String>,
        abstention_policy_version: impl Into<String>,
    ) -> Result<Self, ProspectiveError> {
        Ok(Self {
            protocol_version: LedgerLabel::new("evaluation protocol version", protocol_version)?,
            scoring_rule: LedgerLabel::new("scoring rule", scoring_rule)?,
            abstention_policy_version: LedgerLabel::new(
                "abstention policy version",
                abstention_policy_version,
            )?,
        })
    }

    pub fn protocol_version(&self) -> &str {
        self.protocol_version.as_str()
    }

    pub fn scoring_rule(&self) -> &str {
        self.scoring_rule.as_str()
    }

    pub fn abstention_policy_version(&self) -> &str {
        self.abstention_policy_version.as_str()
    }
}

/// The scientifically relevant result of one forecast attempt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ForecastAttemptDecision {
    Forecast(ForecastPayload),
    Abstain(AbstentionReason),
}

#[derive(Serialize, Deserialize)]
struct ProspectiveAttemptCommitmentRepr {
    id: ForecastCommitmentId,
    observation_lineage: ObservationLineage,
    observation_cutoff: ForecastCoordinate,
    forecast_window: ForecastWindow,
    observation_policy_version: LedgerLabel,
    input_snapshot_hash: LedgerLabel,
    model_versions: Vec<LedgerLabel>,
    trajectory_generator_ids: Vec<LedgerLabel>,
    branch_clustering_method: Option<LedgerLabel>,
    external_references: Vec<ExternalReference>,
    evaluation_protocol: EvaluationProtocol,
    decision: ForecastAttemptDecision,
    notes: String,
}

/// One pre-outcome forecast attempt, including typed abstention.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProspectiveAttemptCommitment {
    id: ForecastCommitmentId,
    observation_lineage: ObservationLineage,
    observation_cutoff: ForecastCoordinate,
    forecast_window: ForecastWindow,
    observation_policy_version: LedgerLabel,
    input_snapshot_hash: LedgerLabel,
    model_versions: Vec<LedgerLabel>,
    trajectory_generator_ids: Vec<LedgerLabel>,
    branch_clustering_method: Option<LedgerLabel>,
    external_references: Vec<ExternalReference>,
    evaluation_protocol: EvaluationProtocol,
    decision: ForecastAttemptDecision,
    notes: String,
}

impl ProspectiveAttemptCommitment {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: ForecastCommitmentId,
        observation_lineage: ObservationLineage,
        observation_cutoff: ForecastCoordinate,
        forecast_window: ForecastWindow,
        observation_policy_version: impl Into<String>,
        input_snapshot_hash: impl Into<String>,
        model_versions: Vec<String>,
        trajectory_generator_ids: Vec<String>,
        branch_clustering_method: Option<String>,
        external_references: Vec<ExternalReference>,
        evaluation_protocol: EvaluationProtocol,
        decision: ForecastAttemptDecision,
        notes: impl Into<String>,
    ) -> Result<Self, ProspectiveError> {
        if compare_coordinates(&observation_cutoff, forecast_window.issued_at())?
            == Ordering::Greater
        {
            return Err(LedgerV2Error::ObservationCutoffAfterIssue.into());
        }
        if model_versions.is_empty() {
            return Err(LedgerV2Error::EmptyList {
                field: "model versions",
            }
            .into());
        }
        if trajectory_generator_ids.is_empty() {
            return Err(LedgerV2Error::EmptyList {
                field: "trajectory generator ids",
            }
            .into());
        }

        let model_versions = model_versions
            .into_iter()
            .map(|value| LedgerLabel::new("model version", value))
            .collect::<Result<Vec<_>, _>>()?;
        let trajectory_generator_ids = trajectory_generator_ids
            .into_iter()
            .map(|value| LedgerLabel::new("trajectory generator id", value))
            .collect::<Result<Vec<_>, _>>()?;
        let branch_clustering_method = branch_clustering_method
            .map(|value| LedgerLabel::new("branch clustering method", value))
            .transpose()?;

        Ok(Self {
            id,
            observation_lineage,
            observation_cutoff,
            forecast_window,
            observation_policy_version: LedgerLabel::new(
                "observation policy version",
                observation_policy_version,
            )?,
            input_snapshot_hash: LedgerLabel::new("input snapshot hash", input_snapshot_hash)?,
            model_versions,
            trajectory_generator_ids,
            branch_clustering_method,
            external_references,
            evaluation_protocol,
            decision,
            notes: notes.into(),
        })
    }

    pub fn id(&self) -> &ForecastCommitmentId {
        &self.id
    }

    pub fn observation_lineage(&self) -> &ObservationLineage {
        &self.observation_lineage
    }

    pub fn observation_cutoff(&self) -> &ForecastCoordinate {
        &self.observation_cutoff
    }

    pub fn forecast_window(&self) -> &ForecastWindow {
        &self.forecast_window
    }

    pub fn evaluation_protocol(&self) -> &EvaluationProtocol {
        &self.evaluation_protocol
    }

    pub fn decision(&self) -> &ForecastAttemptDecision {
        &self.decision
    }

    pub fn external_references(&self) -> &[ExternalReference] {
        &self.external_references
    }
}

impl TryFrom<ProspectiveAttemptCommitmentRepr> for ProspectiveAttemptCommitment {
    type Error = ProspectiveError;

    fn try_from(repr: ProspectiveAttemptCommitmentRepr) -> Result<Self, Self::Error> {
        Self::new(
            repr.id,
            repr.observation_lineage,
            repr.observation_cutoff,
            repr.forecast_window,
            repr.observation_policy_version.as_str().to_string(),
            repr.input_snapshot_hash.as_str().to_string(),
            repr.model_versions
                .into_iter()
                .map(|value| value.as_str().to_string())
                .collect(),
            repr.trajectory_generator_ids
                .into_iter()
                .map(|value| value.as_str().to_string())
                .collect(),
            repr.branch_clustering_method
                .map(|value| value.as_str().to_string()),
            repr.external_references,
            repr.evaluation_protocol,
            repr.decision,
            repr.notes,
        )
    }
}

impl<'de> Deserialize<'de> for ProspectiveAttemptCommitment {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = ProspectiveAttemptCommitmentRepr::deserialize(deserializer)?;
        Self::try_from(repr).map_err(serde::de::Error::custom)
    }
}

#[derive(Serialize, Deserialize)]
enum ProspectiveAttemptResolutionRepr {
    Forecast {
        id: ForecastResolutionId,
        attempt_id: ForecastCommitmentId,
        forecast_target: ForecastCoordinate,
        outcome_space: OutcomeSpaceId,
        outcome_lineage: ObservationLineage,
        outcome_cutoff: ForecastCoordinate,
        actual_continuation: OutcomeRegion,
        evaluation_protocol: EvaluationProtocol,
        score: ResolutionScore,
        calibration_bucket: Option<LedgerLabel>,
        notes: String,
    },
    AbstentionObserved {
        id: ForecastResolutionId,
        attempt_id: ForecastCommitmentId,
        forecast_target: ForecastCoordinate,
        outcome_lineage: ObservationLineage,
        outcome_cutoff: ForecastCoordinate,
        actual_continuation: OutcomeRegion,
        evaluation_protocol: EvaluationProtocol,
        reason: AbstentionReason,
        notes: String,
    },
}

/// Outcome observation for a committed forecast attempt.
///
/// Distribution attempts are scored using the already-committed evaluation
/// protocol. Abstentions are resolved too, but never receive a fake forecast
/// score; they remain available for coverage/selective-risk analysis.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ProspectiveAttemptResolution {
    Forecast {
        id: ForecastResolutionId,
        attempt_id: ForecastCommitmentId,
        forecast_target: ForecastCoordinate,
        outcome_space: OutcomeSpaceId,
        outcome_lineage: ObservationLineage,
        outcome_cutoff: ForecastCoordinate,
        actual_continuation: OutcomeRegion,
        evaluation_protocol: EvaluationProtocol,
        score: ResolutionScore,
        calibration_bucket: Option<LedgerLabel>,
        notes: String,
    },
    AbstentionObserved {
        id: ForecastResolutionId,
        attempt_id: ForecastCommitmentId,
        forecast_target: ForecastCoordinate,
        outcome_lineage: ObservationLineage,
        outcome_cutoff: ForecastCoordinate,
        actual_continuation: OutcomeRegion,
        evaluation_protocol: EvaluationProtocol,
        reason: AbstentionReason,
        notes: String,
    },
}

impl ProspectiveAttemptResolution {
    #[allow(clippy::too_many_arguments)]
    pub fn resolve_forecast(
        id: ForecastResolutionId,
        attempt: &ProspectiveAttemptCommitment,
        outcome_lineage: ObservationLineage,
        outcome_cutoff: ForecastCoordinate,
        actual_continuation: OutcomeRegion,
        score: f64,
        calibration_bucket: Option<String>,
        notes: impl Into<String>,
    ) -> Result<Self, ProspectiveError> {
        let ForecastAttemptDecision::Forecast(payload) = attempt.decision() else {
            return Err(ProspectiveError::AttemptDecisionMismatch);
        };
        let forecast_target = attempt.forecast_window().target()?;
        if compare_coordinates(&outcome_cutoff, &forecast_target)? == Ordering::Less {
            return Err(LedgerV2Error::OutcomeBeforeForecastTarget.into());
        }

        Ok(Self::Forecast {
            id,
            attempt_id: attempt.id().clone(),
            forecast_target,
            outcome_space: payload.outcome_space().clone(),
            outcome_lineage,
            outcome_cutoff,
            actual_continuation,
            evaluation_protocol: attempt.evaluation_protocol().clone(),
            score: ResolutionScore::new(score)?,
            calibration_bucket: calibration_bucket
                .map(|value| LedgerLabel::new("calibration bucket", value))
                .transpose()?,
            notes: notes.into(),
        })
    }

    pub fn resolve_abstention(
        id: ForecastResolutionId,
        attempt: &ProspectiveAttemptCommitment,
        outcome_lineage: ObservationLineage,
        outcome_cutoff: ForecastCoordinate,
        actual_continuation: OutcomeRegion,
        notes: impl Into<String>,
    ) -> Result<Self, ProspectiveError> {
        let ForecastAttemptDecision::Abstain(reason) = attempt.decision() else {
            return Err(ProspectiveError::AttemptDecisionMismatch);
        };
        let forecast_target = attempt.forecast_window().target()?;
        if compare_coordinates(&outcome_cutoff, &forecast_target)? == Ordering::Less {
            return Err(LedgerV2Error::OutcomeBeforeForecastTarget.into());
        }

        Ok(Self::AbstentionObserved {
            id,
            attempt_id: attempt.id().clone(),
            forecast_target,
            outcome_lineage,
            outcome_cutoff,
            actual_continuation,
            evaluation_protocol: attempt.evaluation_protocol().clone(),
            reason: *reason,
            notes: notes.into(),
        })
    }

    pub fn attempt_id(&self) -> &ForecastCommitmentId {
        match self {
            Self::Forecast { attempt_id, .. } | Self::AbstentionObserved { attempt_id, .. } => {
                attempt_id
            }
        }
    }

    pub fn evaluation_protocol(&self) -> &EvaluationProtocol {
        match self {
            Self::Forecast {
                evaluation_protocol,
                ..
            }
            | Self::AbstentionObserved {
                evaluation_protocol,
                ..
            } => evaluation_protocol,
        }
    }

    /// Re-establish cross-record integrity after loading a resolution and its
    /// referenced attempt from durable storage.
    pub fn validate_against(
        &self,
        attempt: &ProspectiveAttemptCommitment,
    ) -> Result<(), ProspectiveError> {
        if self.attempt_id() != attempt.id() {
            return Err(ProspectiveError::AttemptIdMismatch);
        }
        if self.evaluation_protocol() != attempt.evaluation_protocol() {
            return Err(ProspectiveError::EvaluationProtocolMismatch);
        }

        let expected_target = attempt.forecast_window().target()?;
        let stored_target = match self {
            Self::Forecast {
                forecast_target, ..
            }
            | Self::AbstentionObserved {
                forecast_target, ..
            } => forecast_target,
        };
        if stored_target != &expected_target {
            return Err(ProspectiveError::ForecastTargetMismatch);
        }

        match (self, attempt.decision()) {
            (
                Self::Forecast { outcome_space, .. },
                ForecastAttemptDecision::Forecast(payload),
            ) => {
                if outcome_space != payload.outcome_space() {
                    return Err(ProspectiveError::OutcomeSpaceMismatch);
                }
            }
            (
                Self::AbstentionObserved { reason, .. },
                ForecastAttemptDecision::Abstain(expected_reason),
            ) => {
                if reason != expected_reason {
                    return Err(ProspectiveError::AbstentionReasonMismatch);
                }
            }
            _ => return Err(ProspectiveError::AttemptDecisionMismatch),
        }

        Ok(())
    }
}

impl TryFrom<ProspectiveAttemptResolutionRepr> for ProspectiveAttemptResolution {
    type Error = ProspectiveError;

    fn try_from(repr: ProspectiveAttemptResolutionRepr) -> Result<Self, Self::Error> {
        let resolution = match repr {
            ProspectiveAttemptResolutionRepr::Forecast {
                id,
                attempt_id,
                forecast_target,
                outcome_space,
                outcome_lineage,
                outcome_cutoff,
                actual_continuation,
                evaluation_protocol,
                score,
                calibration_bucket,
                notes,
            } => {
                if compare_coordinates(&outcome_cutoff, &forecast_target)? == Ordering::Less {
                    return Err(LedgerV2Error::OutcomeBeforeForecastTarget.into());
                }
                Self::Forecast {
                    id,
                    attempt_id,
                    forecast_target,
                    outcome_space,
                    outcome_lineage,
                    outcome_cutoff,
                    actual_continuation,
                    evaluation_protocol,
                    score,
                    calibration_bucket,
                    notes,
                }
            }
            ProspectiveAttemptResolutionRepr::AbstentionObserved {
                id,
                attempt_id,
                forecast_target,
                outcome_lineage,
                outcome_cutoff,
                actual_continuation,
                evaluation_protocol,
                reason,
                notes,
            } => {
                if compare_coordinates(&outcome_cutoff, &forecast_target)? == Ordering::Less {
                    return Err(LedgerV2Error::OutcomeBeforeForecastTarget.into());
                }
                Self::AbstentionObserved {
                    id,
                    attempt_id,
                    forecast_target,
                    outcome_lineage,
                    outcome_cutoff,
                    actual_continuation,
                    evaluation_protocol,
                    reason,
                    notes,
                }
            }
        };
        Ok(resolution)
    }
}

impl<'de> Deserialize<'de> for ProspectiveAttemptResolution {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = ProspectiveAttemptResolutionRepr::deserialize(deserializer)?;
        Self::try_from(repr).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::{ForecastSpan, ObservedSnapshotRef};

    fn lineage(snapshot: &str) -> ObservationLineage {
        ObservationLineage::observed(vec![
            ObservedSnapshotRef::new("public-statistics", snapshot, "sha256:abc123").unwrap(),
        ])
        .unwrap()
    }

    fn window() -> ForecastWindow {
        ForecastWindow::new(
            ForecastCoordinate::ordinal("calendar-month", 100).unwrap(),
            ForecastSpan::ordinal_steps("calendar-month", 3).unwrap(),
        )
        .unwrap()
    }

    fn protocol() -> EvaluationProtocol {
        EvaluationProtocol::new("econ-eval-v1", "brier", "selective-risk-v1").unwrap()
    }

    fn payload() -> ForecastPayload {
        ForecastPayload::try_from_raw(
            OutcomeSpaceId("inflation-target".into()),
            vec![
                (0.7, OutcomeRegion::Boolean(true), vec![]),
                (0.3, OutcomeRegion::Boolean(false), vec![]),
            ],
            0.0,
        )
        .unwrap()
    }

    fn attempt(decision: ForecastAttemptDecision) -> ProspectiveAttemptCommitment {
        ProspectiveAttemptCommitment::new(
            ForecastCommitmentId::new("attempt-1").unwrap(),
            lineage("input-vintage"),
            ForecastCoordinate::ordinal("calendar-month", 100).unwrap(),
            window(),
            "observation-policy-v1",
            "sha256:inputs",
            vec!["model-v1".into()],
            vec!["generator-v1".into()],
            None,
            vec![ExternalReference::new("test.claim", "claim-1").unwrap()],
            protocol(),
            decision,
            "",
        )
        .unwrap()
    }

    #[test]
    fn scoring_rule_is_fixed_before_resolution() {
        let attempt = attempt(ForecastAttemptDecision::Forecast(payload()));
        assert_eq!(attempt.evaluation_protocol().scoring_rule(), "brier");

        let resolution = ProspectiveAttemptResolution::resolve_forecast(
            ForecastResolutionId::new("resolution-1").unwrap(),
            &attempt,
            lineage("outcome-vintage"),
            ForecastCoordinate::ordinal("calendar-month", 103).unwrap(),
            OutcomeRegion::Boolean(true),
            0.18,
            None,
            "",
        )
        .unwrap();

        assert_eq!(resolution.evaluation_protocol().scoring_rule(), "brier");
        resolution.validate_against(&attempt).unwrap();
    }

    #[test]
    fn abstention_is_committed_and_resolved_without_fake_score() {
        let attempt = attempt(ForecastAttemptDecision::Abstain(
            AbstentionReason::OutOfDistributionScenario,
        ));
        let resolution = ProspectiveAttemptResolution::resolve_abstention(
            ForecastResolutionId::new("resolution-abstain").unwrap(),
            &attempt,
            lineage("outcome-vintage"),
            ForecastCoordinate::ordinal("calendar-month", 103).unwrap(),
            OutcomeRegion::Boolean(false),
            "outcome observed despite abstention",
        )
        .unwrap();

        assert!(matches!(
            resolution,
            ProspectiveAttemptResolution::AbstentionObserved {
                reason: AbstentionReason::OutOfDistributionScenario,
                ..
            }
        ));
        resolution.validate_against(&attempt).unwrap();
    }

    #[test]
    fn forecast_resolution_rejects_abstained_attempt() {
        let attempt = attempt(ForecastAttemptDecision::Abstain(
            AbstentionReason::InsufficientObservationHistory,
        ));
        let result = ProspectiveAttemptResolution::resolve_forecast(
            ForecastResolutionId::new("bad-resolution").unwrap(),
            &attempt,
            lineage("outcome-vintage"),
            ForecastCoordinate::ordinal("calendar-month", 103).unwrap(),
            OutcomeRegion::Boolean(true),
            0.5,
            None,
            "",
        );
        assert_eq!(result, Err(ProspectiveError::AttemptDecisionMismatch));
    }

    #[test]
    fn abstention_resolution_rejects_forecast_attempt() {
        let attempt = attempt(ForecastAttemptDecision::Forecast(payload()));
        let result = ProspectiveAttemptResolution::resolve_abstention(
            ForecastResolutionId::new("bad-abstention-resolution").unwrap(),
            &attempt,
            lineage("outcome-vintage"),
            ForecastCoordinate::ordinal("calendar-month", 103).unwrap(),
            OutcomeRegion::Boolean(true),
            "",
        );
        assert_eq!(result, Err(ProspectiveError::AttemptDecisionMismatch));
    }

    #[test]
    fn cross_record_validation_detects_different_attempt() {
        let attempt = attempt(ForecastAttemptDecision::Forecast(payload()));
        let resolution = ProspectiveAttemptResolution::resolve_forecast(
            ForecastResolutionId::new("resolution-1").unwrap(),
            &attempt,
            lineage("outcome-vintage"),
            ForecastCoordinate::ordinal("calendar-month", 103).unwrap(),
            OutcomeRegion::Boolean(true),
            0.18,
            None,
            "",
        )
        .unwrap();

        let other = ProspectiveAttemptCommitment::new(
            ForecastCommitmentId::new("attempt-2").unwrap(),
            lineage("input-vintage"),
            ForecastCoordinate::ordinal("calendar-month", 100).unwrap(),
            window(),
            "observation-policy-v1",
            "sha256:inputs",
            vec!["model-v1".into()],
            vec!["generator-v1".into()],
            None,
            vec![],
            protocol(),
            ForecastAttemptDecision::Forecast(payload()),
            "",
        )
        .unwrap();

        assert_eq!(
            resolution.validate_against(&other),
            Err(ProspectiveError::AttemptIdMismatch)
        );
    }

    #[test]
    fn cross_record_validation_detects_protocol_drift() {
        let attempt = attempt(ForecastAttemptDecision::Forecast(payload()));
        let resolution = ProspectiveAttemptResolution::resolve_forecast(
            ForecastResolutionId::new("resolution-1").unwrap(),
            &attempt,
            lineage("outcome-vintage"),
            ForecastCoordinate::ordinal("calendar-month", 103).unwrap(),
            OutcomeRegion::Boolean(true),
            0.18,
            None,
            "",
        )
        .unwrap();

        let drifted = ProspectiveAttemptCommitment::new(
            attempt.id().clone(),
            lineage("input-vintage"),
            ForecastCoordinate::ordinal("calendar-month", 100).unwrap(),
            window(),
            "observation-policy-v1",
            "sha256:inputs",
            vec!["model-v1".into()],
            vec!["generator-v1".into()],
            None,
            vec![],
            EvaluationProtocol::new("econ-eval-v2", "log-score", "selective-risk-v1").unwrap(),
            ForecastAttemptDecision::Forecast(payload()),
            "",
        )
        .unwrap();

        assert_eq!(
            resolution.validate_against(&drifted),
            Err(ProspectiveError::EvaluationProtocolMismatch)
        );
    }

    #[test]
    fn outcome_before_target_fails_closed_for_both_decisions() {
        let forecast_attempt = attempt(ForecastAttemptDecision::Forecast(payload()));
        assert!(matches!(
            ProspectiveAttemptResolution::resolve_forecast(
                ForecastResolutionId::new("early-forecast").unwrap(),
                &forecast_attempt,
                lineage("outcome-vintage"),
                ForecastCoordinate::ordinal("calendar-month", 102).unwrap(),
                OutcomeRegion::Boolean(true),
                0.2,
                None,
                "",
            ),
            Err(ProspectiveError::Ledger(
                LedgerV2Error::OutcomeBeforeForecastTarget
            ))
        ));

        let abstention_attempt = attempt(ForecastAttemptDecision::Abstain(
            AbstentionReason::ModelDisagreementTooHigh,
        ));
        assert!(matches!(
            ProspectiveAttemptResolution::resolve_abstention(
                ForecastResolutionId::new("early-abstention").unwrap(),
                &abstention_attempt,
                lineage("outcome-vintage"),
                ForecastCoordinate::ordinal("calendar-month", 102).unwrap(),
                OutcomeRegion::Boolean(false),
                "",
            ),
            Err(ProspectiveError::Ledger(
                LedgerV2Error::OutcomeBeforeForecastTarget
            ))
        ));
    }
}
