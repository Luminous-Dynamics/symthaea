// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Qualification spike for precommitted outcome-observation semantics.
//!
//! The prospective protocol already commits scoring and abstention semantics
//! before reveal. For external data, the definition of the *observed outcome*
//! must be fixed too: source, measured series/outcome identity, release/vintage
//! selection, transformation identity, and missing-outcome disposition.
//!
//! This file is test-only. It does not change the current raw resolution API and
//! does not claim that source metadata, timing, custody, bytes, or transforms are
//! independently verified.

use symthaea_futures_core::{AssumptionId, ForecastPayload, OutcomeRegion, OutcomeSpaceId};
use symthaea_futures_ledger::prospective::{
    EvaluationProtocol, ForecastAttemptDecision, ProspectiveAttemptCommitment,
    ProspectiveAttemptResolution,
};
use symthaea_futures_ledger::provenance::{ContentAddressedRef, ContentDigest};
use symthaea_futures_ledger::v2::{
    ForecastCommitmentId, ForecastCoordinate, ForecastResolutionId, ForecastSpan, ForecastWindow,
    ObservationLineage, ObservedSnapshotRef,
};

const ZERO_SHA256: &str =
    "sha256:0000000000000000000000000000000000000000000000000000000000000000";
const ONE_SHA256: &str =
    "sha256:1111111111111111111111111111111111111111111111111111111111111111";

#[derive(Debug, Clone, PartialEq, Eq)]
enum OutcomePolicyError {
    EmptyText(&'static str),
    InvalidReleaseOrdinal,
    AttemptIdMismatch,
    SourceMismatch,
    MeasureMismatch,
    ReleaseSelectionMismatch,
    TransformMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ReleaseSelector {
    /// Evaluate against the first published release for the target observation.
    FirstPublished,
    /// Evaluate against one exact one-based release ordinal.
    Ordinal(u16),
    /// Evaluate against one exact, pre-known vintage identity.
    FixedVintage(String),
}

impl ReleaseSelector {
    fn validate(&self) -> Result<(), OutcomePolicyError> {
        match self {
            Self::FirstPublished => Ok(()),
            Self::Ordinal(value) => {
                if *value == 0 {
                    Err(OutcomePolicyError::InvalidReleaseOrdinal)
                } else {
                    Ok(())
                }
            }
            Self::FixedVintage(vintage) if vintage.trim().is_empty() => {
                Err(OutcomePolicyError::EmptyText("fixed vintage id"))
            }
            Self::FixedVintage(_) => Ok(()),
        }
    }

    fn matches(&self, descriptor: &ObservedOutcomeDescriptor) -> bool {
        match self {
            Self::FirstPublished => descriptor.release_ordinal == 1,
            Self::Ordinal(expected) => descriptor.release_ordinal == *expected,
            Self::FixedVintage(expected) => descriptor.vintage_id.as_str() == expected,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MissingOutcomeDisposition {
    /// Keep the attempt unresolved until the precommitted outcome becomes available.
    RemainUnresolved,
    /// Close as missing without manufacturing a forecast score.
    ResolveMissingWithoutScore,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OutcomeObservationPolicy {
    policy_version: String,
    source_id: String,
    measure_id: String,
    release_selector: ReleaseSelector,
    transform: Option<ContentAddressedRef>,
    missing_disposition: MissingOutcomeDisposition,
}

impl OutcomeObservationPolicy {
    fn new(
        policy_version: impl Into<String>,
        source_id: impl Into<String>,
        measure_id: impl Into<String>,
        release_selector: ReleaseSelector,
        transform: Option<ContentAddressedRef>,
        missing_disposition: MissingOutcomeDisposition,
    ) -> Result<Self, OutcomePolicyError> {
        let policy_version = policy_version.into();
        let source_id = source_id.into();
        let measure_id = measure_id.into();
        if policy_version.trim().is_empty() {
            return Err(OutcomePolicyError::EmptyText("outcome policy version"));
        }
        if source_id.trim().is_empty() {
            return Err(OutcomePolicyError::EmptyText("outcome source id"));
        }
        if measure_id.trim().is_empty() {
            return Err(OutcomePolicyError::EmptyText("outcome measure id"));
        }
        release_selector.validate()?;
        Ok(Self {
            policy_version,
            source_id,
            measure_id,
            release_selector,
            transform,
            missing_disposition,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PrecommittedEvaluationContract {
    attempt_id: ForecastCommitmentId,
    evaluation_protocol: EvaluationProtocol,
    outcome_policy: OutcomeObservationPolicy,
}

impl PrecommittedEvaluationContract {
    fn new(
        attempt: &ProspectiveAttemptCommitment,
        outcome_policy: OutcomeObservationPolicy,
    ) -> Self {
        Self {
            attempt_id: attempt.id().clone(),
            evaluation_protocol: attempt.evaluation_protocol().clone(),
            outcome_policy,
        }
    }

    fn qualify_outcome(
        &self,
        attempt: &ProspectiveAttemptCommitment,
        descriptor: &ObservedOutcomeDescriptor,
    ) -> Result<(), OutcomePolicyError> {
        if attempt.id() != &self.attempt_id {
            return Err(OutcomePolicyError::AttemptIdMismatch);
        }
        if descriptor.source_id != self.outcome_policy.source_id {
            return Err(OutcomePolicyError::SourceMismatch);
        }
        if descriptor.measure_id != self.outcome_policy.measure_id {
            return Err(OutcomePolicyError::MeasureMismatch);
        }
        if !self.outcome_policy.release_selector.matches(descriptor) {
            return Err(OutcomePolicyError::ReleaseSelectionMismatch);
        }
        if descriptor.transform != self.outcome_policy.transform {
            return Err(OutcomePolicyError::TransformMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ObservedOutcomeDescriptor {
    source_id: String,
    measure_id: String,
    vintage_id: String,
    release_ordinal: u16,
    transform: Option<ContentAddressedRef>,
}

impl ObservedOutcomeDescriptor {
    fn new(
        source_id: impl Into<String>,
        measure_id: impl Into<String>,
        vintage_id: impl Into<String>,
        release_ordinal: u16,
        transform: Option<ContentAddressedRef>,
    ) -> Result<Self, OutcomePolicyError> {
        let source_id = source_id.into();
        let measure_id = measure_id.into();
        let vintage_id = vintage_id.into();
        if source_id.trim().is_empty() {
            return Err(OutcomePolicyError::EmptyText("observed source id"));
        }
        if measure_id.trim().is_empty() {
            return Err(OutcomePolicyError::EmptyText("observed measure id"));
        }
        if vintage_id.trim().is_empty() {
            return Err(OutcomePolicyError::EmptyText("observed vintage id"));
        }
        if release_ordinal == 0 {
            return Err(OutcomePolicyError::InvalidReleaseOrdinal);
        }
        Ok(Self {
            source_id,
            measure_id,
            vintage_id,
            release_ordinal,
            transform,
        })
    }
}

fn digest(value: &str) -> ContentDigest {
    ContentDigest::parse(value).unwrap()
}

fn transform(id: &str, value: &str) -> ContentAddressedRef {
    ContentAddressedRef::new("outcome-transform", id, digest(value)).unwrap()
}

fn payload() -> ForecastPayload {
    ForecastPayload::try_from_raw(
        OutcomeSpaceId("gdp-growth-sign".into()),
        vec![
            (
                0.6,
                OutcomeRegion::Discrete("positive".into()),
                vec![AssumptionId("claim:gdp-positive".into())],
            ),
            (
                0.4,
                OutcomeRegion::Discrete("non-positive".into()),
                vec![],
            ),
        ],
        0.0,
    )
    .unwrap()
}

fn attempt(id: &str) -> ProspectiveAttemptCommitment {
    ProspectiveAttemptCommitment::new(
        ForecastCommitmentId::new(id).unwrap(),
        ObservationLineage::seeded_simulation("outcome-policy-fixture", 7).unwrap(),
        ForecastCoordinate::SimulationTick(90),
        ForecastWindow::new(
            ForecastCoordinate::SimulationTick(100),
            ForecastSpan::SimulationTicks(10),
        )
        .unwrap(),
        "observation-policy-v1",
        "sha256:legacy-input-label",
        vec!["model-v1".into()],
        vec!["generator-v1".into()],
        None,
        vec![],
        EvaluationProtocol::new("eval-v1", "brier", "abstention-v1").unwrap(),
        ForecastAttemptDecision::Forecast(payload()),
        "pre-outcome attempt",
    )
    .unwrap()
}

fn outcome_lineage(snapshot_id: &str) -> ObservationLineage {
    ObservationLineage::observed(vec![
        ObservedSnapshotRef::new("official-statistics", snapshot_id, "sha256:raw-label").unwrap(),
    ])
    .unwrap()
}

#[test]
fn current_resolution_api_accepts_multiple_posthoc_outcome_vintages() {
    let attempt = attempt("attempt-current-negative-control");

    let first_release = ProspectiveAttemptResolution::resolve_forecast(
        ForecastResolutionId::new("resolution-first").unwrap(),
        &attempt,
        outcome_lineage("gdp-advance"),
        ForecastCoordinate::SimulationTick(110),
        OutcomeRegion::Discrete("positive".into()),
        0.2,
        None,
        "first release",
    );
    let second_release = ProspectiveAttemptResolution::resolve_forecast(
        ForecastResolutionId::new("resolution-second").unwrap(),
        &attempt,
        outcome_lineage("gdp-second-estimate"),
        ForecastCoordinate::SimulationTick(110),
        OutcomeRegion::Discrete("positive".into()),
        0.2,
        None,
        "second release",
    );

    // Valid by current raw protocol design: outcome lineage is supplied after
    // reveal and no outcome-vintage selector is stored in the attempt itself.
    assert!(first_release.is_ok());
    assert!(second_release.is_ok());
}

#[test]
fn first_published_policy_rejects_later_revision_after_reveal() {
    let attempt = attempt("attempt-first-release");
    let contract = PrecommittedEvaluationContract::new(
        &attempt,
        OutcomeObservationPolicy::new(
            "gdp-outcome-v1",
            "bea.gdp",
            "real-gdp-growth",
            ReleaseSelector::FirstPublished,
            None,
            MissingOutcomeDisposition::RemainUnresolved,
        )
        .unwrap(),
    );

    let first = ObservedOutcomeDescriptor::new(
        "bea.gdp",
        "real-gdp-growth",
        "2026q4-advance",
        1,
        None,
    )
    .unwrap();
    let second = ObservedOutcomeDescriptor::new(
        "bea.gdp",
        "real-gdp-growth",
        "2026q4-second-estimate",
        2,
        None,
    )
    .unwrap();

    assert_eq!(contract.outcome_policy.policy_version, "gdp-outcome-v1");
    assert_eq!(contract.qualify_outcome(&attempt, &first), Ok(()));
    assert_eq!(
        contract.qualify_outcome(&attempt, &second),
        Err(OutcomePolicyError::ReleaseSelectionMismatch)
    );
}

#[test]
fn exact_release_ordinal_and_fixed_vintage_selectors_are_distinct() {
    let attempt = attempt("attempt-selector-semantics");
    let second_release = ObservedOutcomeDescriptor::new(
        "bea.gdp",
        "real-gdp-growth",
        "2026q4-second-estimate",
        2,
        None,
    )
    .unwrap();

    let ordinal_contract = PrecommittedEvaluationContract::new(
        &attempt,
        OutcomeObservationPolicy::new(
            "gdp-outcome-second-v1",
            "bea.gdp",
            "real-gdp-growth",
            ReleaseSelector::Ordinal(2),
            None,
            MissingOutcomeDisposition::RemainUnresolved,
        )
        .unwrap(),
    );
    assert_eq!(
        ordinal_contract.qualify_outcome(&attempt, &second_release),
        Ok(())
    );

    let fixed_contract = PrecommittedEvaluationContract::new(
        &attempt,
        OutcomeObservationPolicy::new(
            "gdp-outcome-fixed-v1",
            "bea.gdp",
            "real-gdp-growth",
            ReleaseSelector::FixedVintage("2026q4-second-estimate".into()),
            None,
            MissingOutcomeDisposition::ResolveMissingWithoutScore,
        )
        .unwrap(),
    );
    assert_eq!(
        fixed_contract.qualify_outcome(&attempt, &second_release),
        Ok(())
    );
    assert_eq!(
        fixed_contract.outcome_policy.missing_disposition,
        MissingOutcomeDisposition::ResolveMissingWithoutScore
    );
}

#[test]
fn source_measure_and_transform_substitution_fail_closed() {
    let attempt = attempt("attempt-binding");
    let expected_transform = transform("annualize-v1", ZERO_SHA256);
    let contract = PrecommittedEvaluationContract::new(
        &attempt,
        OutcomeObservationPolicy::new(
            "gdp-outcome-v1",
            "bea.gdp",
            "real-gdp-growth",
            ReleaseSelector::FirstPublished,
            Some(expected_transform.clone()),
            MissingOutcomeDisposition::RemainUnresolved,
        )
        .unwrap(),
    );

    let wrong_source = ObservedOutcomeDescriptor::new(
        "alternate.gdp",
        "real-gdp-growth",
        "advance",
        1,
        Some(expected_transform.clone()),
    )
    .unwrap();
    assert_eq!(
        contract.qualify_outcome(&attempt, &wrong_source),
        Err(OutcomePolicyError::SourceMismatch)
    );

    let wrong_measure = ObservedOutcomeDescriptor::new(
        "bea.gdp",
        "nominal-gdp-growth",
        "advance",
        1,
        Some(expected_transform.clone()),
    )
    .unwrap();
    assert_eq!(
        contract.qualify_outcome(&attempt, &wrong_measure),
        Err(OutcomePolicyError::MeasureMismatch)
    );

    let wrong_transform = ObservedOutcomeDescriptor::new(
        "bea.gdp",
        "real-gdp-growth",
        "advance",
        1,
        Some(transform("annualize-v1", ONE_SHA256)),
    )
    .unwrap();
    assert_eq!(
        contract.qualify_outcome(&attempt, &wrong_transform),
        Err(OutcomePolicyError::TransformMismatch)
    );
}

#[test]
fn precommitted_contract_is_bound_to_one_attempt_and_evaluation_protocol() {
    let attempt_a = attempt("attempt-a");
    let attempt_b = attempt("attempt-b");
    let contract = PrecommittedEvaluationContract::new(
        &attempt_a,
        OutcomeObservationPolicy::new(
            "gdp-outcome-v1",
            "bea.gdp",
            "real-gdp-growth",
            ReleaseSelector::FirstPublished,
            None,
            MissingOutcomeDisposition::RemainUnresolved,
        )
        .unwrap(),
    );
    let observed = ObservedOutcomeDescriptor::new(
        "bea.gdp",
        "real-gdp-growth",
        "advance",
        1,
        None,
    )
    .unwrap();

    assert_eq!(
        contract.qualify_outcome(&attempt_b, &observed),
        Err(OutcomePolicyError::AttemptIdMismatch)
    );
    assert_eq!(
        &contract.evaluation_protocol,
        attempt_a.evaluation_protocol()
    );
}

#[test]
fn invalid_policy_or_observation_identities_fail_before_selection() {
    assert_eq!(
        OutcomeObservationPolicy::new(
            "policy",
            "source",
            "measure",
            ReleaseSelector::Ordinal(0),
            None,
            MissingOutcomeDisposition::RemainUnresolved,
        ),
        Err(OutcomePolicyError::InvalidReleaseOrdinal)
    );
    assert_eq!(
        OutcomeObservationPolicy::new(
            "policy",
            "source",
            "measure",
            ReleaseSelector::FixedVintage("   ".into()),
            None,
            MissingOutcomeDisposition::RemainUnresolved,
        ),
        Err(OutcomePolicyError::EmptyText("fixed vintage id"))
    );
    assert_eq!(
        ObservedOutcomeDescriptor::new("source", "measure", "vintage", 0, None),
        Err(OutcomePolicyError::InvalidReleaseOrdinal)
    );
}
