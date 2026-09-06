// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Qualification spike for real-world forecast timing.
//!
//! The existing v2/prospective schema intentionally uses one `ForecastCoordinate`
//! family for observation cutoffs, issuance, targets, and outcome cutoffs. That is
//! sufficient for seeded simulations and single-axis experiments, but a real-world
//! forecast needs independent notions of:
//!
//! 1. **information time** — when exact bytes/releases were publicly available;
//! 2. **custody time** — when the forecasting process actually acquired those bytes;
//! 3. **semantic/reference time** — which period/event the forecast or observation
//!    is about (for example, `2026-Q4`).
//!
//! Historical reconstruction of an old public vintage is not proof of live
//! prospective custody. This file is test-only: it qualifies that distinction
//! before changing a production wire format.

use std::cmp::Ordering;

use symthaea_futures_core::{AssumptionId, ForecastPayload, OutcomeRegion, OutcomeSpaceId};
use symthaea_futures_ledger::prospective::{
    EvaluationProtocol, ForecastAttemptDecision, ProspectiveAttemptCommitment, ProspectiveError,
};
use symthaea_futures_ledger::v2::{
    ForecastCommitmentId, ForecastCoordinate, ForecastSpan, ForecastWindow, LedgerV2Error,
    ObservationLineage,
};

#[derive(Debug, Clone, PartialEq, Eq)]
enum DualClockError {
    EmptyIdentity(&'static str),
    InformationCutoffAfterIssue,
    SourceReleaseAfterInformationCutoff,
    AcquisitionBeforeSourceAvailability,
    AcquisitionAfterInformationCutoff,
    OutcomeSourceAlreadyAvailableAtIssue,
    OutcomeBeforeSemanticTarget,
    SemanticAxisMismatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TimingAdmissionClass {
    /// After-the-fact reconstruction can establish that a source vintage was
    /// publicly available, but not that a live forecast process possessed it.
    HistoricalReconstructionOnly,
    /// Exact bytes were acquired by the forecasting process before its cutoff.
    LiveProspectiveEligible,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DualClockAttemptTiming {
    /// Exact wall-clock instant through which information is admissible.
    information_cutoff_unix_ms: i64,
    /// Exact wall-clock instant at which the attempt is issued/committed.
    issued_at_unix_ms: i64,
    /// Semantic forecast axis, e.g. calendar-quarter, election-cycle, simulation tick.
    forecast_window: ForecastWindow,
}

impl DualClockAttemptTiming {
    fn new(
        information_cutoff_unix_ms: i64,
        issued_at_unix_ms: i64,
        forecast_window: ForecastWindow,
    ) -> Result<Self, DualClockError> {
        if information_cutoff_unix_ms > issued_at_unix_ms {
            return Err(DualClockError::InformationCutoffAfterIssue);
        }
        Ok(Self {
            information_cutoff_unix_ms,
            issued_at_unix_ms,
            forecast_window,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReleaseCustody {
    /// The bytes were retrieved later for a historical real-time reconstruction.
    HistoricalReconstruction { retrieved_at_unix_ms: i64 },
    /// The live forecasting process acquired the exact bytes at this instant.
    LiveAcquisition { acquired_at_unix_ms: i64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RealtimeReleaseRef {
    source_id: String,
    vintage_id: String,
    /// Period/event described by the release. It is deliberately not compared
    /// with wall-clock publication or acquisition time.
    reference_coordinate: ForecastCoordinate,
    /// First public/source-authoritative availability of this exact vintage.
    source_available_at_unix_ms: i64,
    custody: ReleaseCustody,
}

impl RealtimeReleaseRef {
    fn new(
        source_id: impl Into<String>,
        vintage_id: impl Into<String>,
        reference_coordinate: ForecastCoordinate,
        source_available_at_unix_ms: i64,
        custody: ReleaseCustody,
    ) -> Result<Self, DualClockError> {
        let source_id = source_id.into();
        let vintage_id = vintage_id.into();
        if source_id.trim().is_empty() {
            return Err(DualClockError::EmptyIdentity("source id"));
        }
        if vintage_id.trim().is_empty() {
            return Err(DualClockError::EmptyIdentity("vintage id"));
        }
        Ok(Self {
            source_id,
            vintage_id,
            reference_coordinate,
            source_available_at_unix_ms,
            custody,
        })
    }

    fn admit_against(
        &self,
        attempt: &DualClockAttemptTiming,
    ) -> Result<TimingAdmissionClass, DualClockError> {
        if self.source_available_at_unix_ms > attempt.information_cutoff_unix_ms {
            return Err(DualClockError::SourceReleaseAfterInformationCutoff);
        }

        match self.custody {
            ReleaseCustody::HistoricalReconstruction {
                retrieved_at_unix_ms,
            } => {
                if retrieved_at_unix_ms < self.source_available_at_unix_ms {
                    return Err(DualClockError::AcquisitionBeforeSourceAvailability);
                }
                Ok(TimingAdmissionClass::HistoricalReconstructionOnly)
            }
            ReleaseCustody::LiveAcquisition {
                acquired_at_unix_ms,
            } => {
                if acquired_at_unix_ms < self.source_available_at_unix_ms {
                    return Err(DualClockError::AcquisitionBeforeSourceAvailability);
                }
                if acquired_at_unix_ms > attempt.information_cutoff_unix_ms {
                    return Err(DualClockError::AcquisitionAfterInformationCutoff);
                }
                Ok(TimingAdmissionClass::LiveProspectiveEligible)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OutcomeObservationTiming {
    /// Semantic period/event actually observed.
    reference_coordinate: ForecastCoordinate,
    /// First source-authoritative availability of the outcome.
    source_available_at_unix_ms: i64,
    /// Exact instant at which the evaluator acquired the outcome bytes.
    acquired_at_unix_ms: i64,
}

impl OutcomeObservationTiming {
    fn validate_against(&self, attempt: &DualClockAttemptTiming) -> Result<(), DualClockError> {
        if self.source_available_at_unix_ms <= attempt.issued_at_unix_ms {
            return Err(DualClockError::OutcomeSourceAlreadyAvailableAtIssue);
        }
        if self.acquired_at_unix_ms < self.source_available_at_unix_ms {
            return Err(DualClockError::AcquisitionBeforeSourceAvailability);
        }
        let target = attempt
            .forecast_window
            .target()
            .map_err(|_| DualClockError::SemanticAxisMismatch)?;
        if compare_semantic_coordinates(&self.reference_coordinate, &target)? == Ordering::Less {
            return Err(DualClockError::OutcomeBeforeSemanticTarget);
        }
        Ok(())
    }
}

fn compare_semantic_coordinates(
    first: &ForecastCoordinate,
    second: &ForecastCoordinate,
) -> Result<Ordering, DualClockError> {
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
        _ => Err(DualClockError::SemanticAxisMismatch),
    }
}

fn payload() -> ForecastPayload {
    ForecastPayload::try_from_raw(
        OutcomeSpaceId("growth-sign".into()),
        vec![
            (
                0.6,
                OutcomeRegion::Discrete("positive".into()),
                vec![AssumptionId("claim:growth-positive".into())],
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

fn quarter(index: i64) -> ForecastCoordinate {
    ForecastCoordinate::ordinal("calendar-quarter", index).unwrap()
}

fn semantic_window() -> ForecastWindow {
    ForecastWindow::new(
        quarter(2026 * 4 + 2),
        ForecastSpan::ordinal_steps("calendar-quarter", 2).unwrap(),
    )
    .unwrap()
}

fn live_release(
    vintage_id: &str,
    reference_coordinate: ForecastCoordinate,
    source_available_at_unix_ms: i64,
    acquired_at_unix_ms: i64,
) -> RealtimeReleaseRef {
    RealtimeReleaseRef::new(
        "bea.gdp",
        vintage_id,
        reference_coordinate,
        source_available_at_unix_ms,
        ReleaseCustody::LiveAcquisition {
            acquired_at_unix_ms,
        },
    )
    .unwrap()
}

#[test]
fn current_prospective_schema_rejects_wall_clock_cutoff_with_semantic_issue_axis() {
    let result = ProspectiveAttemptCommitment::new(
        ForecastCommitmentId::new("dual-clock-negative-control").unwrap(),
        ObservationLineage::seeded_simulation("timing-fixture", 1).unwrap(),
        ForecastCoordinate::UnixMillis(1_780_000_000_000),
        semantic_window(),
        "observation-policy-v1",
        "sha256:legacy-label-only",
        vec!["model-v1".into()],
        vec!["generator-v1".into()],
        None,
        vec![],
        EvaluationProtocol::new("eval-v1", "brier", "abstention-v1").unwrap(),
        ForecastAttemptDecision::Forecast(payload()),
        "qualification negative control",
    );

    assert!(matches!(
        result,
        Err(ProspectiveError::Ledger(LedgerV2Error::TimeAxisMismatch))
    ));
}

#[test]
fn information_clock_and_semantic_forecast_axis_can_be_independent() {
    let timing = DualClockAttemptTiming::new(
        1_780_000_000_000,
        1_780_000_300_000,
        semantic_window(),
    )
    .unwrap();

    assert!(matches!(
        timing.forecast_window.target().unwrap(),
        ForecastCoordinate::Ordinal { ref axis, index }
            if axis.as_str() == "calendar-quarter" && index == 2026 * 4 + 4
    ));
}

#[test]
fn release_identity_reference_availability_and_custody_remain_explicit() {
    let release = live_release(
        "2026q2-advance",
        quarter(2026 * 4 + 1),
        1_779_000_000_000,
        1_779_000_010_000,
    );

    assert_eq!(release.source_id, "bea.gdp");
    assert_eq!(release.vintage_id, "2026q2-advance");
    assert!(matches!(
        release.reference_coordinate,
        ForecastCoordinate::Ordinal { ref axis, index }
            if axis.as_str() == "calendar-quarter" && index == 2026 * 4 + 1
    ));
    assert_eq!(release.source_available_at_unix_ms, 1_779_000_000_000);
    assert_eq!(
        release.custody,
        ReleaseCustody::LiveAcquisition {
            acquired_at_unix_ms: 1_779_000_010_000
        }
    );

    assert_eq!(
        RealtimeReleaseRef::new(
            "",
            "v1",
            quarter(1),
            1,
            ReleaseCustody::HistoricalReconstruction {
                retrieved_at_unix_ms: 2,
            },
        ),
        Err(DualClockError::EmptyIdentity("source id"))
    );
}

#[test]
fn historical_reconstruction_does_not_mint_live_prospective_custody() {
    let timing = DualClockAttemptTiming::new(
        1_780_000_000_000,
        1_780_000_300_000,
        semantic_window(),
    )
    .unwrap();
    let reconstructed_later = RealtimeReleaseRef::new(
        "bea.gdp",
        "2026q2-advance",
        quarter(2026 * 4 + 1),
        1_779_000_000_000,
        ReleaseCustody::HistoricalReconstruction {
            retrieved_at_unix_ms: 1_900_000_000_000,
        },
    )
    .unwrap();

    assert_eq!(
        reconstructed_later.admit_against(&timing),
        Ok(TimingAdmissionClass::HistoricalReconstructionOnly)
    );
}

#[test]
fn live_bytes_must_be_acquired_before_the_information_cutoff() {
    let timing = DualClockAttemptTiming::new(
        1_780_000_000_000,
        1_780_000_300_000,
        semantic_window(),
    )
    .unwrap();
    let on_time = live_release(
        "2026q2-advance",
        quarter(2026 * 4 + 1),
        1_779_000_000_000,
        1_779_500_000_000,
    );
    let acquired_too_late = live_release(
        "2026q2-advance-late-copy",
        quarter(2026 * 4 + 1),
        1_779_000_000_000,
        1_780_000_000_001,
    );

    assert_eq!(
        on_time.admit_against(&timing),
        Ok(TimingAdmissionClass::LiveProspectiveEligible)
    );
    assert_eq!(
        acquired_too_late.admit_against(&timing),
        Err(DualClockError::AcquisitionAfterInformationCutoff)
    );
}

#[test]
fn acquisition_cannot_precede_source_availability() {
    let timing = DualClockAttemptTiming::new(
        1_780_000_000_000,
        1_780_000_300_000,
        semantic_window(),
    )
    .unwrap();
    let impossible = live_release(
        "2026q2-advance-impossible",
        quarter(2026 * 4 + 1),
        1_779_000_000_000,
        1_778_999_999_999,
    );

    assert_eq!(
        impossible.admit_against(&timing),
        Err(DualClockError::AcquisitionBeforeSourceAvailability)
    );
}

#[test]
fn revised_vintage_is_admitted_only_after_its_actual_source_release_time() {
    let early_cutoff = DualClockAttemptTiming::new(
        1_780_000_000_000,
        1_780_000_300_000,
        semantic_window(),
    )
    .unwrap();

    let advance = live_release(
        "2026q2-advance",
        quarter(2026 * 4 + 1),
        1_779_000_000_000,
        1_779_000_010_000,
    );
    let revised = live_release(
        "2026q2-second-estimate",
        quarter(2026 * 4 + 1),
        1_781_000_000_000,
        1_781_000_010_000,
    );

    assert_eq!(
        advance.admit_against(&early_cutoff),
        Ok(TimingAdmissionClass::LiveProspectiveEligible)
    );
    assert_eq!(
        revised.admit_against(&early_cutoff),
        Err(DualClockError::SourceReleaseAfterInformationCutoff)
    );
}

#[test]
fn old_reference_period_does_not_make_a_future_revision_historically_available() {
    let timing = DualClockAttemptTiming::new(
        1_780_000_000_000,
        1_780_000_300_000,
        semantic_window(),
    )
    .unwrap();
    let revision_of_old_period = RealtimeReleaseRef::new(
        "statistics.population",
        "2025-rebenchmark-v2",
        quarter(2025 * 4),
        1_790_000_000_000,
        ReleaseCustody::HistoricalReconstruction {
            retrieved_at_unix_ms: 1_900_000_000_000,
        },
    )
    .unwrap();

    assert_eq!(
        revision_of_old_period.admit_against(&timing),
        Err(DualClockError::SourceReleaseAfterInformationCutoff)
    );
}

#[test]
fn information_cutoff_cannot_follow_attempt_issuance() {
    assert_eq!(
        DualClockAttemptTiming::new(
            1_780_000_300_001,
            1_780_000_300_000,
            semantic_window(),
        ),
        Err(DualClockError::InformationCutoffAfterIssue)
    );
}

#[test]
fn outcome_reference_source_availability_and_acquisition_are_checked_separately() {
    let timing = DualClockAttemptTiming::new(
        1_780_000_000_000,
        1_780_000_300_000,
        semantic_window(),
    )
    .unwrap();

    let valid = OutcomeObservationTiming {
        reference_coordinate: quarter(2026 * 4 + 4),
        source_available_at_unix_ms: 1_800_000_000_000,
        acquired_at_unix_ms: 1_800_000_010_000,
    };
    assert_eq!(valid.validate_against(&timing), Ok(()));

    let semantically_too_early = OutcomeObservationTiming {
        reference_coordinate: quarter(2026 * 4 + 3),
        source_available_at_unix_ms: 1_800_000_000_000,
        acquired_at_unix_ms: 1_800_000_010_000,
    };
    assert_eq!(
        semantically_too_early.validate_against(&timing),
        Err(DualClockError::OutcomeBeforeSemanticTarget)
    );

    let source_leaked_before_issue = OutcomeObservationTiming {
        reference_coordinate: quarter(2026 * 4 + 4),
        source_available_at_unix_ms: 1_780_000_300_000,
        acquired_at_unix_ms: 1_800_000_010_000,
    };
    assert_eq!(
        source_leaked_before_issue.validate_against(&timing),
        Err(DualClockError::OutcomeSourceAlreadyAvailableAtIssue)
    );

    let impossible_acquisition = OutcomeObservationTiming {
        reference_coordinate: quarter(2026 * 4 + 4),
        source_available_at_unix_ms: 1_800_000_000_000,
        acquired_at_unix_ms: 1_799_999_999_999,
    };
    assert_eq!(
        impossible_acquisition.validate_against(&timing),
        Err(DualClockError::AcquisitionBeforeSourceAvailability)
    );
}

#[test]
fn semantic_axis_mismatch_fails_without_reinterpreting_wall_clock_time() {
    let timing = DualClockAttemptTiming::new(
        1_780_000_000_000,
        1_780_000_300_000,
        semantic_window(),
    )
    .unwrap();
    let outcome = OutcomeObservationTiming {
        reference_coordinate: ForecastCoordinate::ordinal("calendar-month", 2026 * 12 + 11).unwrap(),
        source_available_at_unix_ms: 1_800_000_000_000,
        acquired_at_unix_ms: 1_800_000_010_000,
    };

    assert_eq!(
        outcome.validate_against(&timing),
        Err(DualClockError::SemanticAxisMismatch)
    );
}
