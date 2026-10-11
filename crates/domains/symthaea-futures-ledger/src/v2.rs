// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Futures evidence schema v2.
//!
//! v1 records prediction, realized outcome, and score in one post-hoc record and
//! assumes a seeded simulation world. v2 separates the lifecycle into an
//! immutable pre-outcome [`ForecastCommitment`] and a later
//! [`ForecastResolution`]. It can represent seeded simulations or immutable
//! external observation snapshots without sentinel seeds/ticks.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Deserializer, Serialize};
use symthaea_futures_core::{ForecastPayload, OutcomeRegion, OutcomeSpaceId};

#[derive(Debug, Clone, PartialEq)]
pub enum LedgerV2Error {
    EmptyText { field: &'static str },
    EmptyList { field: &'static str },
    DuplicateSnapshot { source_id: String, snapshot_id: String },
    ZeroForecastSpan,
    TimeAxisMismatch,
    CoordinateOverflow,
    ObservationCutoffAfterIssue,
    OutcomeBeforeForecastTarget,
    NonFiniteScore { value: f64 },
}

impl fmt::Display for LedgerV2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyText { field } => write!(f, "{field} must not be empty"),
            Self::EmptyList { field } => write!(f, "{field} must not be empty"),
            Self::DuplicateSnapshot {
                source_id,
                snapshot_id,
            } => write!(f, "duplicate observed snapshot {source_id}/{snapshot_id}"),
            Self::ZeroForecastSpan => write!(f, "forecast span must be greater than zero"),
            Self::TimeAxisMismatch => write!(f, "forecast coordinates use incompatible time axes"),
            Self::CoordinateOverflow => write!(f, "forecast target coordinate overflowed"),
            Self::ObservationCutoffAfterIssue => {
                write!(f, "observation cutoff occurs after forecast issuance")
            }
            Self::OutcomeBeforeForecastTarget => {
                write!(f, "outcome cutoff occurs before the forecast target")
            }
            Self::NonFiniteScore { value } => write!(f, "resolution score must be finite, got {value}"),
        }
    }
}

impl std::error::Error for LedgerV2Error {}

/// Reusable validated non-empty text used for provenance labels and hashes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct LedgerLabel(String);

impl LedgerLabel {
    pub fn new(field: &'static str, value: impl Into<String>) -> Result<Self, LedgerV2Error> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(LedgerV2Error::EmptyText { field });
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for LedgerLabel {
    type Error = LedgerV2Error;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new("ledger label", value)
    }
}

impl From<LedgerLabel> for String {
    fn from(value: LedgerLabel) -> Self {
        value.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ForecastCommitmentId(String);

impl ForecastCommitmentId {
    pub fn new(value: impl Into<String>) -> Result<Self, LedgerV2Error> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(LedgerV2Error::EmptyText {
                field: "forecast commitment id",
            });
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ForecastCommitmentId {
    type Error = LedgerV2Error;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<ForecastCommitmentId> for String {
    fn from(value: ForecastCommitmentId) -> Self {
        value.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ForecastResolutionId(String);

impl ForecastResolutionId {
    pub fn new(value: impl Into<String>) -> Result<Self, LedgerV2Error> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(LedgerV2Error::EmptyText {
                field: "forecast resolution id",
            });
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ForecastResolutionId {
    type Error = LedgerV2Error;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<ForecastResolutionId> for String {
    fn from(value: ForecastResolutionId) -> Self {
        value.0
    }
}

/// One immutable external source snapshot admitted to an observation lineage.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ObservedSnapshotRef {
    source_id: LedgerLabel,
    snapshot_id: LedgerLabel,
    content_hash: LedgerLabel,
}

impl ObservedSnapshotRef {
    pub fn new(
        source_id: impl Into<String>,
        snapshot_id: impl Into<String>,
        content_hash: impl Into<String>,
    ) -> Result<Self, LedgerV2Error> {
        Ok(Self {
            source_id: LedgerLabel::new("source id", source_id)?,
            snapshot_id: LedgerLabel::new("snapshot id", snapshot_id)?,
            content_hash: LedgerLabel::new("snapshot content hash", content_hash)?,
        })
    }

    pub fn source_id(&self) -> &str {
        self.source_id.as_str()
    }

    pub fn snapshot_id(&self) -> &str {
        self.snapshot_id.as_str()
    }

    pub fn content_hash(&self) -> &str {
        self.content_hash.as_str()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "Vec<ObservedSnapshotRef>", into = "Vec<ObservedSnapshotRef>")]
pub struct ObservedSnapshotSet(Vec<ObservedSnapshotRef>);

impl ObservedSnapshotSet {
    pub fn new(snapshots: Vec<ObservedSnapshotRef>) -> Result<Self, LedgerV2Error> {
        if snapshots.is_empty() {
            return Err(LedgerV2Error::EmptyList {
                field: "observed snapshots",
            });
        }

        let mut seen = BTreeSet::new();
        for snapshot in &snapshots {
            let key = (snapshot.source_id.clone(), snapshot.snapshot_id.clone());
            if !seen.insert(key) {
                return Err(LedgerV2Error::DuplicateSnapshot {
                    source_id: snapshot.source_id().to_string(),
                    snapshot_id: snapshot.snapshot_id().to_string(),
                });
            }
        }

        Ok(Self(snapshots))
    }

    pub fn as_slice(&self) -> &[ObservedSnapshotRef] {
        &self.0
    }
}

impl TryFrom<Vec<ObservedSnapshotRef>> for ObservedSnapshotSet {
    type Error = LedgerV2Error;

    fn try_from(value: Vec<ObservedSnapshotRef>) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<ObservedSnapshotSet> for Vec<ObservedSnapshotRef> {
    fn from(value: ObservedSnapshotSet) -> Self {
        value.0
    }
}

/// Where observations came from. No sentinel seed represents external reality.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObservationLineage {
    SeededSimulation {
        scenario_family: LedgerLabel,
        world_seed: u64,
    },
    ObservedSnapshots(ObservedSnapshotSet),
}

impl ObservationLineage {
    pub fn seeded_simulation(
        scenario_family: impl Into<String>,
        world_seed: u64,
    ) -> Result<Self, LedgerV2Error> {
        Ok(Self::SeededSimulation {
            scenario_family: LedgerLabel::new("scenario family", scenario_family)?,
            world_seed,
        })
    }

    pub fn observed(snapshots: Vec<ObservedSnapshotRef>) -> Result<Self, LedgerV2Error> {
        Ok(Self::ObservedSnapshots(ObservedSnapshotSet::new(snapshots)?))
    }
}

/// Absolute coordinate of an observation/forecast boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForecastCoordinate {
    SimulationTick(u64),
    UnixMillis(i64),
    Ordinal { axis: LedgerLabel, index: i64 },
}

impl ForecastCoordinate {
    pub fn ordinal(axis: impl Into<String>, index: i64) -> Result<Self, LedgerV2Error> {
        Ok(Self::Ordinal {
            axis: LedgerLabel::new("ordinal time axis", axis)?,
            index,
        })
    }

    fn compare(&self, other: &Self) -> Result<Ordering, LedgerV2Error> {
        match (self, other) {
            (Self::SimulationTick(a), Self::SimulationTick(b)) => Ok(a.cmp(b)),
            (Self::UnixMillis(a), Self::UnixMillis(b)) => Ok(a.cmp(b)),
            (
                Self::Ordinal { axis: a_axis, index: a },
                Self::Ordinal { axis: b_axis, index: b },
            ) if a_axis == b_axis => Ok(a.cmp(b)),
            _ => Err(LedgerV2Error::TimeAxisMismatch),
        }
    }
}

/// Relative distance from issuance to the target outcome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForecastSpan {
    SimulationTicks(u64),
    Millis(u64),
    OrdinalSteps { axis: LedgerLabel, steps: u64 },
}

impl ForecastSpan {
    pub fn ordinal_steps(axis: impl Into<String>, steps: u64) -> Result<Self, LedgerV2Error> {
        Ok(Self::OrdinalSteps {
            axis: LedgerLabel::new("ordinal time axis", axis)?,
            steps,
        })
    }

    fn is_zero(&self) -> bool {
        match self {
            Self::SimulationTicks(value) | Self::Millis(value) => *value == 0,
            Self::OrdinalSteps { steps, .. } => *steps == 0,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct ForecastWindowRepr {
    issued_at: ForecastCoordinate,
    span: ForecastSpan,
}

/// Validated issuance/target window on one explicit time axis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForecastWindow {
    issued_at: ForecastCoordinate,
    span: ForecastSpan,
}

impl ForecastWindow {
    pub fn new(
        issued_at: ForecastCoordinate,
        span: ForecastSpan,
    ) -> Result<Self, LedgerV2Error> {
        if span.is_zero() {
            return Err(LedgerV2Error::ZeroForecastSpan);
        }

        match (&issued_at, &span) {
            (ForecastCoordinate::SimulationTick(_), ForecastSpan::SimulationTicks(_))
            | (ForecastCoordinate::UnixMillis(_), ForecastSpan::Millis(_)) => {}
            (
                ForecastCoordinate::Ordinal { axis: issue_axis, .. },
                ForecastSpan::OrdinalSteps { axis: span_axis, .. },
            ) if issue_axis == span_axis => {}
            _ => return Err(LedgerV2Error::TimeAxisMismatch),
        }

        Ok(Self { issued_at, span })
    }

    pub fn issued_at(&self) -> &ForecastCoordinate {
        &self.issued_at
    }

    pub fn span(&self) -> &ForecastSpan {
        &self.span
    }

    pub fn target(&self) -> Result<ForecastCoordinate, LedgerV2Error> {
        match (&self.issued_at, &self.span) {
            (ForecastCoordinate::SimulationTick(at), ForecastSpan::SimulationTicks(delta)) => at
                .checked_add(*delta)
                .map(ForecastCoordinate::SimulationTick)
                .ok_or(LedgerV2Error::CoordinateOverflow),
            (ForecastCoordinate::UnixMillis(at), ForecastSpan::Millis(delta)) => {
                let delta = i64::try_from(*delta).map_err(|_| LedgerV2Error::CoordinateOverflow)?;
                at.checked_add(delta)
                    .map(ForecastCoordinate::UnixMillis)
                    .ok_or(LedgerV2Error::CoordinateOverflow)
            }
            (
                ForecastCoordinate::Ordinal { axis, index },
                ForecastSpan::OrdinalSteps {
                    axis: span_axis,
                    steps,
                },
            ) if axis == span_axis => {
                let delta = i64::try_from(*steps).map_err(|_| LedgerV2Error::CoordinateOverflow)?;
                index
                    .checked_add(delta)
                    .map(|target| ForecastCoordinate::Ordinal {
                        axis: axis.clone(),
                        index: target,
                    })
                    .ok_or(LedgerV2Error::CoordinateOverflow)
            }
            _ => Err(LedgerV2Error::TimeAxisMismatch),
        }
    }
}

impl<'de> Deserialize<'de> for ForecastWindow {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = ForecastWindowRepr::deserialize(deserializer)?;
        Self::new(repr.issued_at, repr.span).map_err(serde::de::Error::custom)
    }
}

/// Neutral cross-domain reference. Economics can bind ETIR claim IDs here
/// without making the Futures ledger depend on the economics crate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalReference {
    namespace: LedgerLabel,
    id: LedgerLabel,
}

impl ExternalReference {
    pub fn new(
        namespace: impl Into<String>,
        id: impl Into<String>,
    ) -> Result<Self, LedgerV2Error> {
        Ok(Self {
            namespace: LedgerLabel::new("external reference namespace", namespace)?,
            id: LedgerLabel::new("external reference id", id)?,
        })
    }

    pub fn namespace(&self) -> &str {
        self.namespace.as_str()
    }

    pub fn id(&self) -> &str {
        self.id.as_str()
    }
}

#[derive(Serialize, Deserialize)]
struct ForecastCommitmentRepr {
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
    forecast: ForecastPayload,
    notes: String,
}

/// Immutable pre-outcome record. It intentionally contains no realized outcome,
/// score, or calibration result.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ForecastCommitment {
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
    forecast: ForecastPayload,
    notes: String,
}

impl ForecastCommitment {
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
        forecast: ForecastPayload,
        notes: impl Into<String>,
    ) -> Result<Self, LedgerV2Error> {
        if observation_cutoff.compare(forecast_window.issued_at())? == Ordering::Greater {
            return Err(LedgerV2Error::ObservationCutoffAfterIssue);
        }
        if model_versions.is_empty() {
            return Err(LedgerV2Error::EmptyList {
                field: "model versions",
            });
        }
        if trajectory_generator_ids.is_empty() {
            return Err(LedgerV2Error::EmptyList {
                field: "trajectory generator ids",
            });
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
            forecast,
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

    pub fn forecast(&self) -> &ForecastPayload {
        &self.forecast
    }

    pub fn external_references(&self) -> &[ExternalReference] {
        &self.external_references
    }
}

impl TryFrom<ForecastCommitmentRepr> for ForecastCommitment {
    type Error = LedgerV2Error;

    fn try_from(repr: ForecastCommitmentRepr) -> Result<Self, Self::Error> {
        Self::new(
            repr.id,
            repr.observation_lineage,
            repr.observation_cutoff,
            repr.forecast_window,
            repr.observation_policy_version.0,
            repr.input_snapshot_hash.0,
            repr.model_versions.into_iter().map(|value| value.0).collect(),
            repr.trajectory_generator_ids
                .into_iter()
                .map(|value| value.0)
                .collect(),
            repr.branch_clustering_method.map(|value| value.0),
            repr.external_references,
            repr.forecast,
            repr.notes,
        )
    }
}

impl<'de> Deserialize<'de> for ForecastCommitment {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = ForecastCommitmentRepr::deserialize(deserializer)?;
        Self::try_from(repr).map_err(serde::de::Error::custom)
    }
}

/// Finite score stored by a resolution. Lower/higher semantics remain the
/// scoring rule's responsibility; the ledger only rejects NaN/infinity.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct ResolutionScore(f64);

impl ResolutionScore {
    pub fn new(value: f64) -> Result<Self, LedgerV2Error> {
        if value.is_finite() {
            Ok(Self(value))
        } else {
            Err(LedgerV2Error::NonFiniteScore { value })
        }
    }

    pub fn get(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for ResolutionScore {
    type Error = LedgerV2Error;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<ResolutionScore> for f64 {
    fn from(value: ResolutionScore) -> Self {
        value.0
    }
}

#[derive(Serialize, Deserialize)]
struct ForecastResolutionRepr {
    id: ForecastResolutionId,
    commitment_id: ForecastCommitmentId,
    forecast_target: ForecastCoordinate,
    outcome_space: OutcomeSpaceId,
    outcome_lineage: ObservationLineage,
    outcome_cutoff: ForecastCoordinate,
    actual_continuation: OutcomeRegion,
    scoring_rule: LedgerLabel,
    score: ResolutionScore,
    calibration_bucket: Option<LedgerLabel>,
    notes: String,
}

/// Post-outcome record linked to one pre-outcome commitment.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ForecastResolution {
    id: ForecastResolutionId,
    commitment_id: ForecastCommitmentId,
    forecast_target: ForecastCoordinate,
    outcome_space: OutcomeSpaceId,
    outcome_lineage: ObservationLineage,
    outcome_cutoff: ForecastCoordinate,
    actual_continuation: OutcomeRegion,
    scoring_rule: LedgerLabel,
    score: ResolutionScore,
    calibration_bucket: Option<LedgerLabel>,
    notes: String,
}

impl ForecastResolution {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: ForecastResolutionId,
        commitment: &ForecastCommitment,
        outcome_lineage: ObservationLineage,
        outcome_cutoff: ForecastCoordinate,
        actual_continuation: OutcomeRegion,
        scoring_rule: impl Into<String>,
        score: f64,
        calibration_bucket: Option<String>,
        notes: impl Into<String>,
    ) -> Result<Self, LedgerV2Error> {
        let forecast_target = commitment.forecast_window.target()?;
        if outcome_cutoff.compare(&forecast_target)? == Ordering::Less {
            return Err(LedgerV2Error::OutcomeBeforeForecastTarget);
        }

        Ok(Self {
            id,
            commitment_id: commitment.id.clone(),
            forecast_target,
            outcome_space: commitment.forecast.outcome_space().clone(),
            outcome_lineage,
            outcome_cutoff,
            actual_continuation,
            scoring_rule: LedgerLabel::new("scoring rule", scoring_rule)?,
            score: ResolutionScore::new(score)?,
            calibration_bucket: calibration_bucket
                .map(|value| LedgerLabel::new("calibration bucket", value))
                .transpose()?,
            notes: notes.into(),
        })
    }

    pub fn id(&self) -> &ForecastResolutionId {
        &self.id
    }

    pub fn commitment_id(&self) -> &ForecastCommitmentId {
        &self.commitment_id
    }

    pub fn forecast_target(&self) -> &ForecastCoordinate {
        &self.forecast_target
    }

    pub fn outcome_space(&self) -> &OutcomeSpaceId {
        &self.outcome_space
    }

    pub fn outcome_lineage(&self) -> &ObservationLineage {
        &self.outcome_lineage
    }

    pub fn outcome_cutoff(&self) -> &ForecastCoordinate {
        &self.outcome_cutoff
    }

    pub fn actual_continuation(&self) -> &OutcomeRegion {
        &self.actual_continuation
    }

    pub fn scoring_rule(&self) -> &str {
        self.scoring_rule.as_str()
    }

    pub fn score(&self) -> f64 {
        self.score.get()
    }
}

impl TryFrom<ForecastResolutionRepr> for ForecastResolution {
    type Error = LedgerV2Error;

    fn try_from(repr: ForecastResolutionRepr) -> Result<Self, Self::Error> {
        if repr.outcome_cutoff.compare(&repr.forecast_target)? == Ordering::Less {
            return Err(LedgerV2Error::OutcomeBeforeForecastTarget);
        }

        Ok(Self {
            id: repr.id,
            commitment_id: repr.commitment_id,
            forecast_target: repr.forecast_target,
            outcome_space: repr.outcome_space,
            outcome_lineage: repr.outcome_lineage,
            outcome_cutoff: repr.outcome_cutoff,
            actual_continuation: repr.actual_continuation,
            scoring_rule: repr.scoring_rule,
            score: repr.score,
            calibration_bucket: repr.calibration_bucket,
            notes: repr.notes,
        })
    }
}

impl<'de> Deserialize<'de> for ForecastResolution {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = ForecastResolutionRepr::deserialize(deserializer)?;
        Self::try_from(repr).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_futures_core::{AssumptionId, OutcomeRegion};

    fn payload() -> ForecastPayload {
        ForecastPayload::try_from_raw(
            OutcomeSpaceId("inflation_band".into()),
            vec![
                (
                    0.6,
                    OutcomeRegion::Discrete("inside_target".into()),
                    vec![AssumptionId("claim:inflation-v1".into())],
                ),
                (
                    0.4,
                    OutcomeRegion::Discrete("outside_target".into()),
                    vec![],
                ),
            ],
            0.0,
        )
        .unwrap()
    }

    fn observed_lineage(snapshot_id: &str) -> ObservationLineage {
        ObservationLineage::observed(vec![
            ObservedSnapshotRef::new("public-statistics", snapshot_id, "sha256:abc123").unwrap(),
        ])
        .unwrap()
    }

    fn commitment() -> ForecastCommitment {
        ForecastCommitment::new(
            ForecastCommitmentId::new("forecast-001").unwrap(),
            observed_lineage("2026-09-vintage"),
            ForecastCoordinate::ordinal("calendar-month", 2026 * 12 + 8).unwrap(),
            ForecastWindow::new(
                ForecastCoordinate::ordinal("calendar-month", 2026 * 12 + 8).unwrap(),
                ForecastSpan::ordinal_steps("calendar-month", 3).unwrap(),
            )
            .unwrap(),
            "econ-observation-policy-v1",
            "sha256:model-inputs",
            vec!["economic-model-v1".into()],
            vec!["heterogeneous-agent".into()],
            None,
            vec![ExternalReference::new(
                "symthaea.economics.etir.claim",
                "demand_employment",
            )
            .unwrap()],
            payload(),
            "pre-outcome commitment",
        )
        .unwrap()
    }

    #[test]
    fn external_observation_lineage_needs_no_sentinel_seed() {
        let commitment = commitment();
        assert!(matches!(
            commitment.observation_lineage(),
            ObservationLineage::ObservedSnapshots(_)
        ));
        assert_eq!(
            commitment.external_references()[0].namespace(),
            "symthaea.economics.etir.claim"
        );
    }

    #[test]
    fn seeded_simulation_remains_first_class() {
        let lineage = ObservationLineage::seeded_simulation("ecological-collapse", 91).unwrap();
        assert!(matches!(
            lineage,
            ObservationLineage::SeededSimulation { world_seed: 91, .. }
        ));
    }

    #[test]
    fn observed_snapshot_set_rejects_duplicates() {
        let first = ObservedSnapshotRef::new("source", "snapshot", "sha256:a").unwrap();
        let second = ObservedSnapshotRef::new("source", "snapshot", "sha256:b").unwrap();
        assert!(matches!(
            ObservedSnapshotSet::new(vec![first, second]),
            Err(LedgerV2Error::DuplicateSnapshot { .. })
        ));
    }

    #[test]
    fn forecast_window_rejects_zero_or_mixed_axes() {
        assert!(matches!(
            ForecastWindow::new(
                ForecastCoordinate::UnixMillis(100),
                ForecastSpan::Millis(0),
            ),
            Err(LedgerV2Error::ZeroForecastSpan)
        ));
        assert!(matches!(
            ForecastWindow::new(
                ForecastCoordinate::UnixMillis(100),
                ForecastSpan::SimulationTicks(1),
            ),
            Err(LedgerV2Error::TimeAxisMismatch)
        ));
    }

    #[test]
    fn commitment_rejects_observation_cutoff_after_issue() {
        let result = ForecastCommitment::new(
            ForecastCommitmentId::new("bad-cutoff").unwrap(),
            observed_lineage("v1"),
            ForecastCoordinate::UnixMillis(200),
            ForecastWindow::new(
                ForecastCoordinate::UnixMillis(100),
                ForecastSpan::Millis(50),
            )
            .unwrap(),
            "policy-v1",
            "sha256:inputs",
            vec!["model-v1".into()],
            vec!["generator-v1".into()],
            None,
            vec![],
            payload(),
            "",
        );
        assert!(matches!(
            result,
            Err(LedgerV2Error::ObservationCutoffAfterIssue)
        ));
    }

    #[test]
    fn resolution_cannot_precede_forecast_target() {
        let commitment = commitment();
        let result = ForecastResolution::new(
            ForecastResolutionId::new("resolution-too-early").unwrap(),
            &commitment,
            observed_lineage("2026-10-vintage"),
            ForecastCoordinate::ordinal("calendar-month", 2026 * 12 + 9).unwrap(),
            OutcomeRegion::Discrete("inside_target".into()),
            "brier",
            0.2,
            None,
            "",
        );
        assert!(matches!(
            result,
            Err(LedgerV2Error::OutcomeBeforeForecastTarget)
        ));
    }

    #[test]
    fn resolution_links_commitment_and_copies_outcome_space() {
        let commitment = commitment();
        let resolution = ForecastResolution::new(
            ForecastResolutionId::new("resolution-001").unwrap(),
            &commitment,
            observed_lineage("2026-12-vintage"),
            ForecastCoordinate::ordinal("calendar-month", 2026 * 12 + 11).unwrap(),
            OutcomeRegion::Discrete("inside_target".into()),
            "brier",
            0.32,
            Some("0.5-0.7".into()),
            "resolved after target",
        )
        .unwrap();

        assert_eq!(resolution.commitment_id(), commitment.id());
        assert_eq!(resolution.outcome_space(), commitment.forecast().outcome_space());
        assert_eq!(resolution.score(), 0.32);
    }

    #[test]
    fn resolution_rejects_nonfinite_score() {
        let commitment = commitment();
        let result = ForecastResolution::new(
            ForecastResolutionId::new("resolution-nan").unwrap(),
            &commitment,
            observed_lineage("2026-12-vintage"),
            ForecastCoordinate::ordinal("calendar-month", 2026 * 12 + 11).unwrap(),
            OutcomeRegion::Discrete("inside_target".into()),
            "brier",
            f64::NAN,
            None,
            "",
        );
        assert!(matches!(result, Err(LedgerV2Error::NonFiniteScore { .. })));
    }
}
