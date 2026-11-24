// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hard sample-health admissibility for advisory Mycelial active sensing.
//!
//! This module is deliberately downstream of value-of-information (VOI)
//! evaluation. It never converts specimen-health limits into utility costs:
//! an action that breaches a declared hard bound is ineligible regardless
//! of its VOI. It also never actuates microscope hardware or establishes
//! biological safety; limits and exposure estimates must be qualified for
//! the particular assay, specimen and instrument.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::mycelial_active_sensing::{
    MYCELIAL_ACTIVE_SENSING_SCHEMA_VERSION, MycelialActiveSensingReport,
    MycelialAcquisitionAction, MycelialAcquisitionActionKind,
    MycelialAcquisitionDecision, MycelialAcquisitionOutcome,
    MycelialHypothesisLikelihood, MycelialHypothesisLoss, evaluate_mycelial_active_sensing,
};
use crate::mycelial_fusion_counterfactual::{
    MycelialFusionBridgeLengthBasis, MycelialFusionCandidate, MycelialFusionTarget,
};
use crate::mycelial_graph_hypothesis_posterior::{
    MycelialGraphHypothesisPosterior, evaluate_mycelial_graph_hypothesis_posterior,
};
use crate::mycelial_reconstruction_metrics::{
    AnnotatedHyphaSegment, AnnotatedJunction, AnnotatedJunctionKind,
    AnnotatedMycelialSkeleton, MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION,
    ReconstructionDimensionality, ReconstructionLengthUnit, ReconstructionPoint3,
};

pub const MYCELIAL_SAMPLE_HEALTH_BUDGET_SCHEMA_VERSION: u16 = 1;
pub const MAX_MYCELIAL_HEALTH_BUDGET_ACTIONS: usize = 64;
pub const MAX_MYCELIAL_HEALTH_BUDGET_CELLS_PER_ACTION: usize = 16_384;
pub const MAX_MYCELIAL_HEALTH_BUDGET_CHANNELS: usize = 64;
pub const MAX_MYCELIAL_HEALTH_EXPOSURE_RECORDS: usize = 1_000_000;
pub const MAX_MYCELIAL_HEALTH_BUDGET_EVALUATIONS: usize = 1_000_000;
pub const MAX_MYCELIAL_HEALTH_ID_BYTES: usize = 128;
const DIGEST_HEX_BYTES: usize = 64;

/// Immutable input bundle for the upstream decision-value acquisition report.
///
/// Construction independently recomputes the report against the exact source
/// skeleton, candidate inventory, posterior, decision table and action models.
/// The fields remain private so downstream specimen-health filtering cannot
/// receive an unqualified or post-construction-mutated VOI result.
#[derive(Debug, Clone)]
pub struct MycelialSampleHealthActiveSensingEvidence {
    report: MycelialActiveSensingReport,
    skeleton: AnnotatedMycelialSkeleton,
    candidates: Vec<MycelialFusionCandidate>,
    posterior: MycelialGraphHypothesisPosterior,
    decisions: Vec<MycelialAcquisitionDecision>,
    actions: Vec<MycelialAcquisitionAction>,
}

impl MycelialSampleHealthActiveSensingEvidence {
    pub fn new(
        report: MycelialActiveSensingReport,
        skeleton: AnnotatedMycelialSkeleton,
        candidates: Vec<MycelialFusionCandidate>,
        posterior: MycelialGraphHypothesisPosterior,
        decisions: Vec<MycelialAcquisitionDecision>,
        actions: Vec<MycelialAcquisitionAction>,
    ) -> Result<Self, MycelialSampleHealthBudgetError> {
        report
            .validate_against(&skeleton, &candidates, &posterior, &decisions, &actions)
            .map_err(|_| {
                MycelialSampleHealthBudgetError::InvalidActiveSensingReport(
                    "report failed independent source/posterior/decision/action recomputation",
                )
            })?;
        Ok(Self {
            report,
            skeleton,
            candidates,
            posterior,
            decisions,
            actions,
        })
    }

    pub fn report(&self) -> &MycelialActiveSensingReport {
        &self.report
    }

    /// Digest of the exact upstream scientific evidence used to qualify this report.
    pub fn source_evidence_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-mycelial-sample-health-active-sensing-evidence-v1");
        hash_string(&mut hasher, &self.report.source_graph_digest);
        hash_string(&mut hasher, &self.report.candidate_inventory_digest);
        hash_string(&mut hasher, &self.report.posterior_inventory_digest);
        hash_string(&mut hasher, &self.report.decision_inventory_digest);
        hash_string(&mut hasher, &self.report.acquisition_action_inventory_digest);
        hash_string(&mut hasher, &self.skeleton.source_observation_digest);
        hash_string(&mut hasher, &self.skeleton.calibration_digest);
        hasher.update(&(self.candidates.len() as u64).to_le_bytes());
        hasher.update(&(self.posterior.hypotheses.len() as u64).to_le_bytes());
        hasher.update(&(self.decisions.len() as u64).to_le_bytes());
        hasher.update(&(self.actions.len() as u64).to_le_bytes());
        finish_digest(hasher)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialPhotonDoseLimit {
    pub channel_id: String,
    pub minimum_wavelength_nm: u16,
    pub maximum_wavelength_nm: u16,
    /// Digest of the channel-specific calibration used to convert instrument settings to dose.
    pub calibration_digest: String,
    /// Maximum irradiance at the specimen plane, in W/cm².
    pub maximum_irradiance_watts_per_cm2: f64,
    /// Maximum uninterrupted illumination time for one action on this channel.
    pub maximum_continuous_illumination_duration_ns: u64,
    /// Minimum sample-wide recovery interval after prior optical exposure.
    pub minimum_recovery_interval_ns: u64,
    /// Maximum cumulative incident photon dose per spatial cell, in photons/µm².
    /// This is an assay-specific exposure bound, not a universal phototoxicity threshold.
    pub maximum_cumulative_photons_per_um2: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialSampleHealthBudget {
    pub schema_version: u16,
    pub policy_id: String,
    pub policy_version: String,
    pub coordinate_frame_id: String,
    /// Calibration/model bundle used to establish this specimen-health policy.
    pub health_model_digest: String,
    pub maximum_session_duration_ns: u64,
    pub maximum_actions_per_session: u32,
    pub maximum_roi_area_um2: f64,
    pub maximum_local_temperature_rise_millikelvin: u32,
    /// Exactly one wavelength band and cumulative dose limit per channel.
    pub photon_dose_limits: Vec<MycelialPhotonDoseLimit>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialPhotonDoseEstimate {
    pub channel_id: String,
    pub wavelength_nm: u16,
    /// Conservative upper bound at the specimen plane, not a mean or nominal estimate.
    pub photons_per_um2_upper: f64,
    /// Conservative upper bound on irradiance at the specimen plane, in W/cm².
    pub irradiance_watts_per_cm2_upper: f64,
    /// Actual continuous illumination duration for this channel in the proposed plan.
    pub illumination_duration_ns: u64,
    pub calibration_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialAcquisitionBudgetProposal {
    pub action_id: String,
    pub session_id: String,
    pub sample_id: String,
    /// Exact hardware-neutral intent compiled for the proposed acquisition.
    pub acquisition_plan_digest: String,
    /// Exact device/software/firmware capability manifest used for plan compilation.
    pub device_capability_manifest_digest: String,
    pub coordinate_frame_id: String,
    /// Planned start time relative to the recorded session's monotonic timebase.
    /// Must be at or after the state's elapsed session duration.
    pub planned_start_offset_ns: u64,
    /// Canonical sorted, unique spatial cells. Cell identities are stable across
    /// acquisitions so overlapping ROIs accumulate dose on shared cells.
    pub spatial_footprint_cell_ids: Vec<String>,
    pub spatial_footprint_digest: String,
    /// Exact, sorted channels expected to illuminate this action. The estimate
    /// inventory must cover every required channel.
    pub required_illumination_channel_ids: Vec<String>,
    pub roi_area_um2: f64,
    pub duration_ns: u64,
    /// Conservative upper bound on the maximum local temperature rise.
    pub local_temperature_rise_upper_millikelvin: u32,
    /// Must match the corresponding active-sensing action's model digest.
    pub acquisition_model_digest: String,
    pub cost_evidence_digest: String,
    /// Calibration/model identity for temperature and footprint projections.
    pub health_model_digest: String,
    /// Empty only for explicitly non-optical HumanAnnotation actions.
    pub photon_dose_estimates: Vec<MycelialPhotonDoseEstimate>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialCumulativePhotonExposure {
    pub sample_id: String,
    pub spatial_cell_id: String,
    pub channel_id: String,
    pub wavelength_nm: u16,
    pub photons_per_um2: f64,
    pub source_session_id: String,
    /// Evidence receipt and calibration that support this historical exposure.
    pub evidence_receipt_digest: String,
    pub calibration_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialSampleHealthState {
    pub sample_id: String,
    pub session_id: String,
    /// Active device capability manifest for this recorded acquisition session.
    pub device_capability_manifest_digest: String,
    pub coordinate_frame_id: String,
    pub elapsed_session_duration_ns: u64,
    pub completed_actions_in_session: u32,
    /// True only when all prior specimen optical exposures are represented in the ledger.
    pub exposure_history_complete: bool,
    /// Elapsed time since the most recent optical exposure, from a qualified timebase.
    /// None means the complete exposure history contains no prior optical exposure.
    pub elapsed_since_last_optical_exposure_ns: Option<u64>,
    /// Exposure history can span sessions; duration and action count are session-local.
    pub cumulative_photon_exposures: Vec<MycelialCumulativePhotonExposure>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialSampleHealthRejectionReason {
    SessionDurationExceeded,
    SessionActionCountExceeded,
    RoiAreaExceeded,
    TemperatureRiseExceeded,
    MissingSpatialFootprint,
    UnknownIlluminationChannel,
    WavelengthOutsideCalibratedBand,
    MissingOpticalDoseEstimate,
    CumulativePhotonDoseExceeded,
    PhotonCalibrationMismatch,
    IrradianceExceeded,
    ContinuousIlluminationDurationExceeded,
    RecoveryIntervalNotMet,
    ExposureHistoryIncomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialSampleHealthRejection {
    pub action_id: String,
    pub acquisition_plan_digest: String,
    pub device_capability_manifest_digest: String,
    pub spatial_footprint_digest: String,
    pub reasons: Vec<MycelialSampleHealthRejectionReason>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialSampleHealthEligibleAction {
    pub action_id: String,
    pub action_kind: MycelialAcquisitionActionKind,
    pub acquisition_plan_digest: String,
    pub device_capability_manifest_digest: String,
    pub spatial_footprint_digest: String,
    pub health_model_digest: String,
    pub net_value_of_information: f64,
    pub planned_start_offset_ns: u64,
    pub projected_session_duration_ns: u64,
    pub projected_action_count: u32,
    pub maximum_projected_photons_per_um2_by_channel: Vec<MycelialProjectedChannelDose>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialProjectedChannelDose {
    pub channel_id: String,
    pub maximum_projected_photons_per_um2: f64,
    pub irradiance_watts_per_cm2_upper: f64,
    pub illumination_duration_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MycelialSampleHealthBudgetReport {
    pub schema_version: u16,
    pub active_sensing_report_digest: String,
    pub active_sensing_evidence_digest: String,
    pub policy_digest: String,
    pub sample_health_state_digest: String,
    pub proposal_inventory_digest: String,
    pub recommended_action_id: Option<String>,
    pub eligible_actions: Vec<MycelialSampleHealthEligibleAction>,
    pub rejected_actions: Vec<MycelialSampleHealthRejection>,
    /// This report only filters advisory proposals; it never executes an action.
    pub advisory_only: bool,
    pub canonical_digest: String,
}

impl MycelialSampleHealthBudgetReport {
    /// Recomputes the complete sample-health decision against frozen evidence.
    ///
    /// This checks more than the output hash: it replays the upstream VOI source
    /// validation, policy admissibility, cumulative exposure calculation and
    /// final recommendation using the caller's exact inputs.
    pub fn validate_against(
        &self,
        active_sensing_evidence: &MycelialSampleHealthActiveSensingEvidence,
        budget: &MycelialSampleHealthBudget,
        state: &MycelialSampleHealthState,
        proposals: &[MycelialAcquisitionBudgetProposal],
    ) -> Result<(), MycelialSampleHealthBudgetError> {
        let expected = evaluate_mycelial_sample_health_budget(
            active_sensing_evidence,
            budget,
            state,
            proposals,
        )?;
        if self != &expected {
            return Err(MycelialSampleHealthBudgetError::ReportMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MycelialSampleHealthBudgetError {
    InvalidSchemaVersion(u16),
    InvalidIdentifier(&'static str),
    InvalidDigest(&'static str),
    InvalidBudget(&'static str),
    InvalidState(&'static str),
    InvalidProposal(&'static str),
    InvalidActiveSensingReport(&'static str),
    ReportMismatch,
    DuplicateId(&'static str),
    InventoryMismatch,
    ResourceLimitExceeded(&'static str),
    ExposureArithmeticOverflow,
}

impl fmt::Display for MycelialSampleHealthBudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchemaVersion(version) => {
                write!(f, "unsupported Mycelial sample-health budget schema version {version}")
            }
            Self::InvalidIdentifier(field) => {
                write!(f, "invalid Mycelial sample-health identifier: {field}")
            }
            Self::InvalidDigest(field) => {
                write!(f, "invalid Mycelial sample-health evidence digest: {field}")
            }
            Self::InvalidBudget(field) => {
                write!(f, "invalid Mycelial sample-health budget: {field}")
            }
            Self::InvalidState(field) => {
                write!(f, "invalid Mycelial sample-health state: {field}")
            }
            Self::InvalidProposal(field) => {
                write!(f, "invalid Mycelial acquisition budget proposal: {field}")
            }
            Self::InvalidActiveSensingReport(field) => {
                write!(f, "invalid active-sensing report for sample-health filtering: {field}")
            }
            Self::ReportMismatch => {
                write!(f, "sample-health budget report does not match independent recomputation")
            }
            Self::DuplicateId(field) => {
                write!(f, "duplicate Mycelial sample-health identity: {field}")
            }
            Self::InventoryMismatch => {
                write!(f, "sample-health proposal IDs do not exactly match active-sensing action IDs")
            }
            Self::ResourceLimitExceeded(field) => {
                write!(f, "Mycelial sample-health resource limit exceeded: {field}")
            }
            Self::ExposureArithmeticOverflow => {
                write!(f, "cumulative photon-dose arithmetic became non-finite")
            }
        }
    }
}

impl Error for MycelialSampleHealthBudgetError {}

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value.len() <= MAX_MYCELIAL_HEALTH_ID_BYTES
        && !value.chars().any(char::is_control)
}

fn valid_digest(value: &str) -> bool {
    value.len() == DIGEST_HEX_BYTES
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hash_f64(hasher: &mut blake3::Hasher, value: f64) {
    hasher.update(&value.to_bits().to_le_bytes());
}

fn finish_digest(hasher: blake3::Hasher) -> String {
    hasher.finalize().to_hex().to_string()
}

fn validate_identifier(
    value: &str,
    field: &'static str,
) -> Result<(), MycelialSampleHealthBudgetError> {
    if !valid_id(value) {
        return Err(MycelialSampleHealthBudgetError::InvalidIdentifier(field));
    }
    Ok(())
}

fn validate_digest(
    value: &str,
    field: &'static str,
) -> Result<(), MycelialSampleHealthBudgetError> {
    if !valid_digest(value) {
        return Err(MycelialSampleHealthBudgetError::InvalidDigest(field));
    }
    Ok(())
}

fn footprint_digest(
    coordinate_frame_id: &str,
    acquisition_plan_digest: &str,
    device_capability_manifest_digest: &str,
    cell_ids: &[String],
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-spatial-footprint-v2");
    hash_string(&mut hasher, coordinate_frame_id);
    hash_string(&mut hasher, acquisition_plan_digest);
    hash_string(&mut hasher, device_capability_manifest_digest);
    hasher.update(&(cell_ids.len() as u64).to_le_bytes());
    for cell_id in cell_ids {
        hash_string(&mut hasher, cell_id);
    }
    finish_digest(hasher)
}

fn validate_budget(
    budget: &MycelialSampleHealthBudget,
) -> Result<(), MycelialSampleHealthBudgetError> {
    if budget.schema_version != MYCELIAL_SAMPLE_HEALTH_BUDGET_SCHEMA_VERSION {
        return Err(MycelialSampleHealthBudgetError::InvalidSchemaVersion(
            budget.schema_version,
        ));
    }
    validate_identifier(&budget.policy_id, "policy_id")?;
    validate_identifier(&budget.policy_version, "policy_version")?;
    validate_identifier(&budget.coordinate_frame_id, "coordinate_frame_id")?;
    validate_digest(&budget.health_model_digest, "health_model_digest")?;
    if budget.maximum_session_duration_ns == 0
        || budget.maximum_actions_per_session == 0
        || !budget.maximum_roi_area_um2.is_finite()
        || budget.maximum_roi_area_um2 <= 0.0
        || budget.photon_dose_limits.len() > MAX_MYCELIAL_HEALTH_BUDGET_CHANNELS
    {
        return Err(MycelialSampleHealthBudgetError::InvalidBudget(
            "nonzero bounded duration, action count, ROI area and channel inventory are required",
        ));
    }
    let mut channels = BTreeSet::new();
    for limit in &budget.photon_dose_limits {
        validate_identifier(&limit.channel_id, "photon_dose_limit.channel_id")?;
        validate_digest(&limit.calibration_digest, "photon_dose_limit.calibration_digest")?;
        if !channels.insert(limit.channel_id.as_str()) {
            return Err(MycelialSampleHealthBudgetError::DuplicateId(
                "photon_dose_limit.channel_id",
            ));
        }
        if limit.minimum_wavelength_nm == 0
            || limit.minimum_wavelength_nm > limit.maximum_wavelength_nm
            || !limit.maximum_cumulative_photons_per_um2.is_finite()
            || limit.maximum_cumulative_photons_per_um2 <= 0.0
            || !limit.maximum_irradiance_watts_per_cm2.is_finite()
            || limit.maximum_irradiance_watts_per_cm2 <= 0.0
            || limit.maximum_continuous_illumination_duration_ns == 0
        {
            return Err(MycelialSampleHealthBudgetError::InvalidBudget(
                "invalid wavelength band or cumulative photon-dose limit",
            ));
        }
    }
    Ok(())
}

fn validate_state(
    state: &MycelialSampleHealthState,
    budget: &MycelialSampleHealthBudget,
) -> Result<(), MycelialSampleHealthBudgetError> {
    validate_identifier(&state.sample_id, "state.sample_id")?;
    validate_identifier(&state.session_id, "state.session_id")?;
    validate_digest(
        &state.device_capability_manifest_digest,
        "state.device_capability_manifest_digest",
    )?;
    validate_identifier(&state.coordinate_frame_id, "state.coordinate_frame_id")?;
    if state.coordinate_frame_id != budget.coordinate_frame_id {
        return Err(MycelialSampleHealthBudgetError::InvalidState(
            "coordinate frame differs from policy",
        ));
    }
    validate_digest(
        &state.exposure_history_evidence_digest,
        "state.exposure_history_evidence_digest",
    )?;
    if state.elapsed_session_duration_ns > budget.maximum_session_duration_ns
        || state.completed_actions_in_session > budget.maximum_actions_per_session
    {
        return Err(MycelialSampleHealthBudgetError::InvalidState(
            "existing session usage already exceeds policy",
        ));
    }
    if state.cumulative_photon_exposures.len() > MAX_MYCELIAL_HEALTH_EXPOSURE_RECORDS {
        return Err(MycelialSampleHealthBudgetError::ResourceLimitExceeded(
            "cumulative photon exposure records",
        ));
    }
    if state.exposure_history_complete {
        let has_prior_optical_exposure = !state.cumulative_photon_exposures.is_empty();
        if has_prior_optical_exposure != state.elapsed_since_last_optical_exposure_ns.is_some() {
            return Err(MycelialSampleHealthBudgetError::InvalidState(
                "complete exposure history and recovery clock disagree",
            ));
        }
    }
    let limits_by_channel = budget
        .photon_dose_limits
        .iter()
        .map(|limit| (limit.channel_id.as_str(), limit))
        .collect::<BTreeMap<_, _>>();
    for exposure in &state.cumulative_photon_exposures {
        validate_identifier(&exposure.sample_id, "exposure.sample_id")?;
        validate_identifier(&exposure.spatial_cell_id, "exposure.spatial_cell_id")?;
        validate_identifier(&exposure.channel_id, "exposure.channel_id")?;
        validate_identifier(&exposure.source_session_id, "exposure.source_session_id")?;
        validate_digest(&exposure.evidence_receipt_digest, "exposure.evidence_receipt_digest")?;
        validate_digest(&exposure.calibration_digest, "exposure.calibration_digest")?;
        if exposure.sample_id != state.sample_id
            || exposure.wavelength_nm == 0
            || !exposure.photons_per_um2.is_finite()
            || exposure.photons_per_um2 < 0.0
        {
            return Err(MycelialSampleHealthBudgetError::InvalidState(
                "exposure identity, wavelength or dose is invalid",
            ));
        }
        let Some(limit) = limits_by_channel.get(exposure.channel_id.as_str()) else {
            return Err(MycelialSampleHealthBudgetError::InvalidState(
                "prior exposure references an unknown illumination channel",
            ));
        };
        if exposure.wavelength_nm < limit.minimum_wavelength_nm
            || exposure.wavelength_nm > limit.maximum_wavelength_nm
        {
            return Err(MycelialSampleHealthBudgetError::InvalidState(
                "prior exposure wavelength is outside the policy's calibrated band",
            ));
        }
        if exposure.calibration_digest != limit.calibration_digest {
            return Err(MycelialSampleHealthBudgetError::InvalidState(
                "prior exposure calibration differs from the current channel policy",
            ));
        }
    }
    Ok(())
}

fn prior_exposure_by_cell_channel(
    state: &MycelialSampleHealthState,
) -> Result<BTreeMap<(String, String), f64>, MycelialSampleHealthBudgetError> {
    let mut totals = BTreeMap::new();
    for exposure in &state.cumulative_photon_exposures {
        let total = totals
            .entry((
                exposure.spatial_cell_id.clone(),
                exposure.channel_id.clone(),
            ))
            .or_insert(0.0_f64);
        *total += exposure.photons_per_um2;
        if !total.is_finite() {
            return Err(MycelialSampleHealthBudgetError::ExposureArithmeticOverflow);
        }
    }
    Ok(totals)
}

fn validate_active_sensing_report(
    report: &MycelialActiveSensingReport,
) -> Result<(), MycelialSampleHealthBudgetError> {
    if report.schema_version != MYCELIAL_ACTIVE_SENSING_SCHEMA_VERSION {
        return Err(
            MycelialSampleHealthBudgetError::InvalidActiveSensingReport(
                "schema version mismatch",
            ),
        );
    }
    if !report.advisory_only {
        return Err(
            MycelialSampleHealthBudgetError::InvalidActiveSensingReport(
                "physical or non-advisory reports cannot be filtered",
            ),
        );
    }
    for (digest, field) in [
        (report.source_graph_digest.as_str(), "source_graph_digest"),
        (
            report.candidate_inventory_digest.as_str(),
            "candidate_inventory_digest",
        ),
        (
            report.posterior_inventory_digest.as_str(),
            "posterior_inventory_digest",
        ),
        (
            report.decision_inventory_digest.as_str(),
            "decision_inventory_digest",
        ),
        (
            report.acquisition_action_inventory_digest.as_str(),
            "acquisition_action_inventory_digest",
        ),
    ] {
        validate_digest(digest, field)?;
    }
    if report.actions.len() > MAX_MYCELIAL_HEALTH_BUDGET_ACTIONS
        || report.actions.is_empty()
        || report.posterior_hypothesis_count == 0
        || !report.posterior_entropy_nats.is_finite()
        || report.posterior_entropy_nats < 0.0
        || !valid_id(&report.loss_scale_id)
        || !valid_id(&report.baseline_best_decision_id)
        || !report.baseline_bayes_risk.is_finite()
        || report.baseline_bayes_risk < 0.0
    {
        return Err(
            MycelialSampleHealthBudgetError::InvalidActiveSensingReport(
                "empty, unbounded or invalid action inventory",
            ),
        );
    }
    let mut action_ids = BTreeSet::new();
    for action in &report.actions {
        validate_identifier(&action.action_id, "active_sensing.action_id")?;
        validate_identifier(&action.acquisition_model_id, "active_sensing.acquisition_model_id")?;
        validate_identifier(
            &action.acquisition_model_version,
            "active_sensing.acquisition_model_version",
        )?;
        validate_digest(
            &action.acquisition_model_digest,
            "active_sensing.acquisition_model_digest",
        )?;
        validate_digest(
            &action.cost_evidence_digest,
            "active_sensing.cost_evidence_digest",
        )?;
        if !action_ids.insert(action.action_id.as_str())
            || !action.net_value_of_information.is_finite()
            || !action.gross_value_of_information.is_finite()
            || !action.cost_in_loss_units.is_finite()
            || action.cost_in_loss_units < 0.0
            || !action.expected_posterior_bayes_risk.is_finite()
            || action.expected_posterior_bayes_risk < 0.0
        {
            return Err(
                MycelialSampleHealthBudgetError::InvalidActiveSensingReport(
                    "duplicate action ID or non-finite action value",
                ),
            );
        }
    }
    Ok(())
}

fn validate_proposal(
    proposal: &MycelialAcquisitionBudgetProposal,
    action_kind: MycelialAcquisitionActionKind,
    action_model_digest: &str,
    cost_evidence_digest: &str,
    state: &MycelialSampleHealthState,
    budget: &MycelialSampleHealthBudget,
) -> Result<(), MycelialSampleHealthBudgetError> {
    for (value, field) in [
        (proposal.action_id.as_str(), "proposal.action_id"),
        (proposal.session_id.as_str(), "proposal.session_id"),
        (proposal.sample_id.as_str(), "proposal.sample_id"),
        (
            proposal.coordinate_frame_id.as_str(),
            "proposal.coordinate_frame_id",
        ),
    ] {
        validate_identifier(value, field)?;
    }
    for (digest, field) in [
        (
            proposal.spatial_footprint_digest.as_str(),
            "proposal.spatial_footprint_digest",
        ),
        (
            proposal.acquisition_plan_digest.as_str(),
            "proposal.acquisition_plan_digest",
        ),
        (
            proposal.device_capability_manifest_digest.as_str(),
            "proposal.device_capability_manifest_digest",
        ),
        (
            proposal.acquisition_model_digest.as_str(),
            "proposal.acquisition_model_digest",
        ),
        (
            proposal.cost_evidence_digest.as_str(),
            "proposal.cost_evidence_digest",
        ),
        (
            proposal.health_model_digest.as_str(),
            "proposal.health_model_digest",
        ),
    ] {
        validate_digest(digest, field)?;
    }
    if proposal.session_id != state.session_id || proposal.sample_id != state.sample_id {
        return Err(MycelialSampleHealthBudgetError::InvalidProposal(
            "sample or session identity differs from current state",
        ));
    }
    if proposal.device_capability_manifest_digest != state.device_capability_manifest_digest {
        return Err(MycelialSampleHealthBudgetError::InvalidProposal(
            "device capability manifest differs from the active acquisition session",
        ));
    }
    if proposal.coordinate_frame_id != state.coordinate_frame_id
        || proposal.coordinate_frame_id != budget.coordinate_frame_id
    {
        return Err(MycelialSampleHealthBudgetError::InvalidProposal(
            "coordinate frame differs from current state or policy",
        ));
    }
    if proposal.acquisition_model_digest != action_model_digest
        || proposal.cost_evidence_digest != cost_evidence_digest
        || proposal.health_model_digest != budget.health_model_digest
    {
        return Err(MycelialSampleHealthBudgetError::InvalidProposal(
            "acquisition, cost or health-model evidence differs from policy and active-sensing action",
        ));
    }
    if proposal.spatial_footprint_cell_ids.len() > MAX_MYCELIAL_HEALTH_BUDGET_CELLS_PER_ACTION
    {
        return Err(MycelialSampleHealthBudgetError::ResourceLimitExceeded(
            "proposal spatial footprint",
        ));
    }
    let mut previous_required_channel: Option<&str> = None;
    for channel_id in &proposal.required_illumination_channel_ids {
        validate_identifier(channel_id, "proposal.required_illumination_channel_id")?;
        if previous_required_channel.is_some_and(|prior| prior >= channel_id.as_str()) {
            return Err(MycelialSampleHealthBudgetError::InvalidProposal(
                "required illumination channels must be strictly sorted and unique",
            ));
        }
        previous_required_channel = Some(channel_id);
    }
    let mut previous: Option<&str> = None;
    for cell_id in &proposal.spatial_footprint_cell_ids {
        validate_identifier(cell_id, "proposal.spatial_footprint_cell_id")?;
        if previous.is_some_and(|prior| prior >= cell_id.as_str()) {
            return Err(MycelialSampleHealthBudgetError::InvalidProposal(
                "spatial footprint cell IDs must be strictly sorted and unique",
            ));
        }
        previous = Some(cell_id);
    }
    if footprint_digest(
        &proposal.coordinate_frame_id,
        &proposal.acquisition_plan_digest,
        &proposal.device_capability_manifest_digest,
        &proposal.spatial_footprint_cell_ids,
    ) != proposal.spatial_footprint_digest
    {
        return Err(MycelialSampleHealthBudgetError::InvalidDigest(
            "proposal.spatial_footprint_digest",
        ));
    }
    if !proposal.roi_area_um2.is_finite()
        || proposal.roi_area_um2 <= 0.0
        || proposal.duration_ns == 0
        || proposal.planned_start_offset_ns < state.elapsed_session_duration_ns
    {
        return Err(MycelialSampleHealthBudgetError::InvalidProposal(
            "positive finite ROI area and acquisition duration are required",
        ));
    }
    if action_kind == MycelialAcquisitionActionKind::HumanAnnotation {
        if !proposal.photon_dose_estimates.is_empty()
            || !proposal.required_illumination_channel_ids.is_empty()
        {
            return Err(MycelialSampleHealthBudgetError::InvalidProposal(
                "human annotation must not carry optical exposure requirements",
            ));
        }
    }
    let mut previous_dose_channel: Option<&str> = None;
    for dose in &proposal.photon_dose_estimates {
        validate_identifier(&dose.channel_id, "dose.channel_id")?;
        validate_digest(&dose.calibration_digest, "dose.calibration_digest")?;
        if dose.wavelength_nm == 0
            || !dose.photons_per_um2_upper.is_finite()
            || dose.photons_per_um2_upper < 0.0
            || !dose.irradiance_watts_per_cm2_upper.is_finite()
            || dose.irradiance_watts_per_cm2_upper < 0.0
            || dose.illumination_duration_ns == 0
            || dose.illumination_duration_ns > proposal.duration_ns
        {
            return Err(MycelialSampleHealthBudgetError::InvalidProposal(
                "invalid photon-dose, irradiance or illumination-duration estimate",
            ));
        }
        if previous_dose_channel.is_some_and(|prior| prior >= dose.channel_id.as_str()) {
            return Err(MycelialSampleHealthBudgetError::InvalidProposal(
                "photon-dose estimates must be strictly sorted and unique by channel",
            ));
        }
        previous_dose_channel = Some(&dose.channel_id);
    }
    Ok(())
}

fn policy_digest(budget: &MycelialSampleHealthBudget) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-sample-health-budget-v2");
    hasher.update(&budget.schema_version.to_le_bytes());
    hash_string(&mut hasher, &budget.policy_id);
    hash_string(&mut hasher, &budget.policy_version);
    hash_string(&mut hasher, &budget.coordinate_frame_id);
    hash_string(&mut hasher, &budget.health_model_digest);
    hasher.update(&budget.maximum_session_duration_ns.to_le_bytes());
    hasher.update(&budget.maximum_actions_per_session.to_le_bytes());
    hash_f64(&mut hasher, budget.maximum_roi_area_um2);
    hasher.update(&budget.maximum_local_temperature_rise_millikelvin.to_le_bytes());
    let mut limits: Vec<_> = budget.photon_dose_limits.iter().collect();
    limits.sort_by(|left, right| left.channel_id.cmp(&right.channel_id));
    hasher.update(&(limits.len() as u64).to_le_bytes());
    for limit in limits {
        hash_string(&mut hasher, &limit.channel_id);
        hasher.update(&limit.minimum_wavelength_nm.to_le_bytes());
        hasher.update(&limit.maximum_wavelength_nm.to_le_bytes());
        hash_string(&mut hasher, &limit.calibration_digest);
        hash_f64(&mut hasher, limit.maximum_irradiance_watts_per_cm2);
        hasher.update(&limit.maximum_continuous_illumination_duration_ns.to_le_bytes());
        hasher.update(&limit.minimum_recovery_interval_ns.to_le_bytes());
        hash_f64(&mut hasher, limit.maximum_cumulative_photons_per_um2);
    }
    finish_digest(hasher)
}

fn state_digest(state: &MycelialSampleHealthState) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-sample-health-state-v2");
    hash_string(&mut hasher, &state.sample_id);
    hash_string(&mut hasher, &state.session_id);
    hash_string(&mut hasher, &state.device_capability_manifest_digest);
    hash_string(&mut hasher, &state.coordinate_frame_id);
    hasher.update(&state.elapsed_session_duration_ns.to_le_bytes());
    hasher.update(&state.completed_actions_in_session.to_le_bytes());
    hasher.update(&[u8::from(state.exposure_history_complete)]);
    hash_string(&mut hasher, &state.exposure_history_evidence_digest);
    match state.elapsed_since_last_optical_exposure_ns {
        Some(elapsed) => {
            hasher.update(&[1]);
            hasher.update(&elapsed.to_le_bytes());
        }
        None => hasher.update(&[0]),
    }
    let mut exposures: Vec<_> = state.cumulative_photon_exposures.iter().collect();
    exposures.sort_by(|left, right| {
        (
            &left.sample_id,
            &left.spatial_cell_id,
            &left.channel_id,
            left.wavelength_nm,
            &left.source_session_id,
        )
            .cmp(&(
                &right.sample_id,
                &right.spatial_cell_id,
                &right.channel_id,
                right.wavelength_nm,
                &right.source_session_id,
            ))
            .then_with(|| left.photons_per_um2.total_cmp(&right.photons_per_um2))
            .then_with(|| left.evidence_receipt_digest.cmp(&right.evidence_receipt_digest))
            .then_with(|| left.calibration_digest.cmp(&right.calibration_digest))
    });
    hasher.update(&(exposures.len() as u64).to_le_bytes());
    for exposure in exposures {
        hash_string(&mut hasher, &exposure.sample_id);
        hash_string(&mut hasher, &exposure.spatial_cell_id);
        hash_string(&mut hasher, &exposure.channel_id);
        hasher.update(&exposure.wavelength_nm.to_le_bytes());
        hash_f64(&mut hasher, exposure.photons_per_um2);
        hash_string(&mut hasher, &exposure.source_session_id);
        hash_string(&mut hasher, &exposure.evidence_receipt_digest);
        hash_string(&mut hasher, &exposure.calibration_digest);
    }
    finish_digest(hasher)
}

fn proposal_inventory_digest(proposals: &[MycelialAcquisitionBudgetProposal]) -> String {
    let mut ordered: Vec<_> = proposals.iter().collect();
    ordered.sort_by(|left, right| left.action_id.cmp(&right.action_id));
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-acquisition-budget-proposals-v2");
    hasher.update(&(ordered.len() as u64).to_le_bytes());
    for proposal in ordered {
        hash_string(&mut hasher, &proposal.action_id);
        hash_string(&mut hasher, &proposal.session_id);
        hash_string(&mut hasher, &proposal.sample_id);
        hash_string(&mut hasher, &proposal.acquisition_plan_digest);
        hash_string(&mut hasher, &proposal.device_capability_manifest_digest);
        hash_string(&mut hasher, &proposal.coordinate_frame_id);
        hasher.update(&proposal.planned_start_offset_ns.to_le_bytes());
        hash_string(&mut hasher, &proposal.spatial_footprint_digest);
        hasher.update(&(proposal.required_illumination_channel_ids.len() as u64).to_le_bytes());
        for channel_id in &proposal.required_illumination_channel_ids {
            hash_string(&mut hasher, channel_id);
        }
        hasher.update(&proposal.roi_area_um2.to_le_bytes());
        hasher.update(&proposal.duration_ns.to_le_bytes());
        hasher.update(&proposal.local_temperature_rise_upper_millikelvin.to_le_bytes());
        hash_string(&mut hasher, &proposal.acquisition_model_digest);
        hash_string(&mut hasher, &proposal.cost_evidence_digest);
        hash_string(&mut hasher, &proposal.health_model_digest);
        let mut doses: Vec<_> = proposal.photon_dose_estimates.iter().collect();
        doses.sort_by(|left, right| {
            (&left.channel_id, left.wavelength_nm).cmp(&(&right.channel_id, right.wavelength_nm))
        });
        hasher.update(&(doses.len() as u64).to_le_bytes());
        for dose in doses {
            hash_string(&mut hasher, &dose.channel_id);
            hasher.update(&dose.wavelength_nm.to_le_bytes());
            hash_f64(&mut hasher, dose.photons_per_um2_upper);
            hash_f64(&mut hasher, dose.irradiance_watts_per_cm2_upper);
            hasher.update(&dose.illumination_duration_ns.to_le_bytes());
            hash_string(&mut hasher, &dose.calibration_digest);
        }
        hasher.update(&(proposal.spatial_footprint_cell_ids.len() as u64).to_le_bytes());
        for cell_id in &proposal.spatial_footprint_cell_ids {
            hash_string(&mut hasher, cell_id);
        }
    }
    finish_digest(hasher)
}

fn active_sensing_report_digest(report: &MycelialActiveSensingReport) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-active-sensing-source-v2");
    hasher.update(&report.schema_version.to_le_bytes());
    hash_string(&mut hasher, &report.source_graph_digest);
    hash_string(&mut hasher, &report.candidate_inventory_digest);
    hash_string(&mut hasher, &report.posterior_inventory_digest);
    hasher.update(&(report.posterior_hypothesis_count as u64).to_le_bytes());
    hash_f64(&mut hasher, report.posterior_entropy_nats);
    hash_string(&mut hasher, &report.loss_scale_id);
    hash_string(&mut hasher, &report.decision_inventory_digest);
    hash_string(&mut hasher, &report.acquisition_action_inventory_digest);
    hash_string(&mut hasher, &report.baseline_best_decision_id);
    hash_f64(&mut hasher, report.baseline_bayes_risk);
    match &report.recommended_action_id {
        Some(action_id) => {
            hasher.update(&[1]);
            hash_string(&mut hasher, action_id);
        }
        None => hasher.update(&[0]),
    }
    let mut actions: Vec<_> = report.actions.iter().collect();
    actions.sort_by(|left, right| left.action_id.cmp(&right.action_id));
    hasher.update(&(actions.len() as u64).to_le_bytes());
    for action in actions {
        hash_string(&mut hasher, &action.action_id);
        hasher.update(&[match action.action_kind {
            MycelialAcquisitionActionKind::AdditionalZPlane => 0,
            MycelialAcquisitionActionKind::HighResolutionCrop => 1,
            MycelialAcquisitionActionKind::AlternateIllumination => 2,
            MycelialAcquisitionActionKind::NewTimepoint => 3,
            MycelialAcquisitionActionKind::HumanAnnotation => 4,
            MycelialAcquisitionActionKind::Other => 5,
        }]);
        hash_string(&mut hasher, &action.acquisition_model_id);
        hash_string(&mut hasher, &action.acquisition_model_version);
        hash_string(&mut hasher, &action.acquisition_model_digest);
        hash_string(&mut hasher, &action.cost_evidence_digest);
        hash_f64(&mut hasher, action.cost_in_loss_units);
        hash_f64(&mut hasher, action.expected_posterior_bayes_risk);
        hash_f64(&mut hasher, action.gross_value_of_information);
        hash_f64(&mut hasher, action.net_value_of_information);
        let mut outcomes: Vec<_> = action.outcomes.iter().collect();
        outcomes.sort_by(|left, right| left.outcome_id.cmp(&right.outcome_id));
        hasher.update(&(outcomes.len() as u64).to_le_bytes());
        for outcome in outcomes {
            hash_string(&mut hasher, &outcome.outcome_id);
            hash_f64(&mut hasher, outcome.predictive_probability);
            match &outcome.best_decision_id {
                Some(decision_id) => {
                    hasher.update(&[1]);
                    hash_string(&mut hasher, decision_id);
                }
                None => hasher.update(&[0]),
            }
            match outcome.conditional_bayes_risk {
                Some(value) => {
                    hasher.update(&[1]);
                    hash_f64(&mut hasher, value);
                }
                None => hasher.update(&[0]),
            }
            hash_f64(&mut hasher, outcome.expected_risk_contribution);
        }
    }
    hasher.update(&[u8::from(report.advisory_only)]);
    finish_digest(hasher)
}

fn add_reason(
    reasons: &mut Vec<MycelialSampleHealthRejectionReason>,
    reason: MycelialSampleHealthRejectionReason,
) {
    if !reasons.contains(&reason) {
        reasons.push(reason);
    }
}

fn evaluate_proposal(
    action_id: &str,
    action_kind: MycelialAcquisitionActionKind,
    net_voi: f64,
    proposal: &MycelialAcquisitionBudgetProposal,
    budget: &MycelialSampleHealthBudget,
    state: &MycelialSampleHealthState,
    prior_dose_by_cell_channel: &BTreeMap<(String, String), f64>,
) -> Result<
    Result<MycelialSampleHealthEligibleAction, MycelialSampleHealthRejection>,
    MycelialSampleHealthBudgetError,
> {
    let mut reasons = Vec::new();
    let projected_duration = proposal
        .planned_start_offset_ns
        .checked_add(proposal.duration_ns)
        .unwrap_or(u64::MAX);
    let wait_before_action_ns = proposal
        .planned_start_offset_ns
        .saturating_sub(state.elapsed_session_duration_ns);
    let projected_action_count = state.completed_actions_in_session.saturating_add(1);

    if projected_duration > budget.maximum_session_duration_ns {
        add_reason(
            &mut reasons,
            MycelialSampleHealthRejectionReason::SessionDurationExceeded,
        );
    }
    if projected_action_count > budget.maximum_actions_per_session {
        add_reason(
            &mut reasons,
            MycelialSampleHealthRejectionReason::SessionActionCountExceeded,
        );
    }
    if proposal.roi_area_um2 > budget.maximum_roi_area_um2 {
        add_reason(
            &mut reasons,
            MycelialSampleHealthRejectionReason::RoiAreaExceeded,
        );
    }
    if proposal.local_temperature_rise_upper_millikelvin
        > budget.maximum_local_temperature_rise_millikelvin
    {
        add_reason(
            &mut reasons,
            MycelialSampleHealthRejectionReason::TemperatureRiseExceeded,
        );
    }

    if proposal.spatial_footprint_cell_ids.is_empty() {
        add_reason(
            &mut reasons,
            MycelialSampleHealthRejectionReason::MissingSpatialFootprint,
        );
    }

    let mut projected_by_channel: BTreeMap<String, f64> = BTreeMap::new();
    let mut irradiance_by_channel: BTreeMap<String, f64> = BTreeMap::new();
    let mut illumination_duration_by_channel: BTreeMap<String, u64> = BTreeMap::new();
    if action_kind == MycelialAcquisitionActionKind::HumanAnnotation {
        if !proposal.photon_dose_estimates.is_empty() {
            return Err(MycelialSampleHealthBudgetError::InvalidProposal(
                "human annotation unexpectedly contains photon-dose estimates",
            ));
        }
    } else if proposal.photon_dose_estimates.is_empty()
        || proposal.required_illumination_channel_ids.is_empty()
    {
        add_reason(
            &mut reasons,
            MycelialSampleHealthRejectionReason::MissingOpticalDoseEstimate,
        );
    } else {
        if !state.exposure_history_complete {
            add_reason(
                &mut reasons,
                MycelialSampleHealthRejectionReason::ExposureHistoryIncomplete,
            );
        }
        let required_channels = proposal
            .required_illumination_channel_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let estimated_channels = proposal
            .photon_dose_estimates
            .iter()
            .map(|dose| dose.channel_id.as_str())
            .collect::<BTreeSet<_>>();
        if !required_channels.is_subset(&estimated_channels) {
            add_reason(
                &mut reasons,
                MycelialSampleHealthRejectionReason::MissingOpticalDoseEstimate,
            );
        }
        for required_channel in &proposal.required_illumination_channel_ids {
            if !budget
                .photon_dose_limits
                .iter()
                .any(|limit| limit.channel_id == *required_channel)
            {
                add_reason(
                    &mut reasons,
                    MycelialSampleHealthRejectionReason::UnknownIlluminationChannel,
                );
            }
        }
        let mut increment_by_channel: BTreeMap<String, f64> = BTreeMap::new();
        for dose in &proposal.photon_dose_estimates {
            if !required_channels.contains(dose.channel_id.as_str()) {
                add_reason(
                    &mut reasons,
                    MycelialSampleHealthRejectionReason::UnknownIlluminationChannel,
                );
                continue;
            }
            let Some(limit) = budget
                .photon_dose_limits
                .iter()
                .find(|limit| limit.channel_id == dose.channel_id)
            else {
                add_reason(
                    &mut reasons,
                    MycelialSampleHealthRejectionReason::UnknownIlluminationChannel,
                );
                continue;
            };
            if dose.wavelength_nm < limit.minimum_wavelength_nm
                || dose.wavelength_nm > limit.maximum_wavelength_nm
            {
                add_reason(
                    &mut reasons,
                    MycelialSampleHealthRejectionReason::WavelengthOutsideCalibratedBand,
                );
                continue;
            }
            if dose.calibration_digest != limit.calibration_digest {
                add_reason(
                    &mut reasons,
                    MycelialSampleHealthRejectionReason::PhotonCalibrationMismatch,
                );
                continue;
            }
            if dose.irradiance_watts_per_cm2_upper > limit.maximum_irradiance_watts_per_cm2 {
                add_reason(
                    &mut reasons,
                    MycelialSampleHealthRejectionReason::IrradianceExceeded,
                );
            }
            if dose.illumination_duration_ns
                > limit.maximum_continuous_illumination_duration_ns
            {
                add_reason(
                    &mut reasons,
                    MycelialSampleHealthRejectionReason::ContinuousIlluminationDurationExceeded,
                );
            }
            if let Some(elapsed_since_last) = state.elapsed_since_last_optical_exposure_ns {
                let available_recovery_ns =
                    elapsed_since_last.saturating_add(wait_before_action_ns);
                if available_recovery_ns < limit.minimum_recovery_interval_ns {
                    add_reason(
                        &mut reasons,
                        MycelialSampleHealthRejectionReason::RecoveryIntervalNotMet,
                    );
                }
            }
            irradiance_by_channel.insert(
                dose.channel_id.clone(),
                dose.irradiance_watts_per_cm2_upper,
            );
            illumination_duration_by_channel.insert(
                dose.channel_id.clone(),
                dose.illumination_duration_ns,
            );
            let increment = increment_by_channel
                .entry(dose.channel_id.clone())
                .or_insert(0.0);
            *increment += dose.photons_per_um2_upper;
            if !increment.is_finite() {
                return Err(MycelialSampleHealthBudgetError::ExposureArithmeticOverflow);
            }
        }

        for limit in &budget.photon_dose_limits {
            if let Some(increment) = increment_by_channel.get(&limit.channel_id) {
                let mut maximum_projected = 0.0_f64;
                for cell_id in &proposal.spatial_footprint_cell_ids {
                    let prior = prior_dose_by_cell_channel
                        .get(&(cell_id.clone(), limit.channel_id.clone()))
                        .copied()
                        .unwrap_or(0.0);
                    let projected = prior + *increment;
                    if !projected.is_finite() {
                        return Err(MycelialSampleHealthBudgetError::ExposureArithmeticOverflow);
                    }
                    maximum_projected = maximum_projected.max(projected);
                    if projected > limit.maximum_cumulative_photons_per_um2 {
                        add_reason(
                            &mut reasons,
                            MycelialSampleHealthRejectionReason::CumulativePhotonDoseExceeded,
                        );
                    }
                }
                projected_by_channel.insert(limit.channel_id.clone(), maximum_projected);
            }
        }
    }

    if !reasons.is_empty() {
        return Ok(Err(MycelialSampleHealthRejection {
            action_id: action_id.to_string(),
            acquisition_plan_digest: proposal.acquisition_plan_digest.clone(),
            device_capability_manifest_digest: proposal.device_capability_manifest_digest.clone(),
            spatial_footprint_digest: proposal.spatial_footprint_digest.clone(),
            reasons,
        }));
    }

    let mut dose_summaries = projected_by_channel
        .into_iter()
        .map(
            |(channel_id, maximum_projected_photons_per_um2)| MycelialProjectedChannelDose {
                irradiance_watts_per_cm2_upper: irradiance_by_channel
                    .get(&channel_id)
                    .copied()
                    .unwrap_or(0.0),
                illumination_duration_ns: illumination_duration_by_channel
                    .get(&channel_id)
                    .copied()
                    .unwrap_or(0),
                channel_id,
                maximum_projected_photons_per_um2,
            },
        )
        .collect::<Vec<_>>();
    dose_summaries.sort_by(|left, right| left.channel_id.cmp(&right.channel_id));

    Ok(Ok(MycelialSampleHealthEligibleAction {
        action_id: action_id.to_string(),
        action_kind,
        acquisition_plan_digest: proposal.acquisition_plan_digest.clone(),
        device_capability_manifest_digest: proposal.device_capability_manifest_digest.clone(),
        spatial_footprint_digest: proposal.spatial_footprint_digest.clone(),
        health_model_digest: proposal.health_model_digest.clone(),
        net_value_of_information: net_voi,
        planned_start_offset_ns: proposal.planned_start_offset_ns,
        projected_session_duration_ns: projected_duration,
        projected_action_count,
        maximum_projected_photons_per_um2_by_channel: dose_summaries,
    }))
}

/// Filters VOI-ranked proposals through hard, non-monetizable specimen-health bounds.
///
/// Exactly one proposal is required for every active-sensing action. A proposal
/// that breaches a bound is returned with typed reasons and cannot be recommended.
/// Among eligible proposals, the highest positive net VOI is recommended.
pub fn evaluate_mycelial_sample_health_budget(
    active_sensing_evidence: &MycelialSampleHealthActiveSensingEvidence,
    budget: &MycelialSampleHealthBudget,
    state: &MycelialSampleHealthState,
    proposals: &[MycelialAcquisitionBudgetProposal],
) -> Result<MycelialSampleHealthBudgetReport, MycelialSampleHealthBudgetError> {
    validate_budget(budget)?;
    validate_state(state, budget)?;
    let active_sensing_report = active_sensing_evidence.report();
    validate_active_sensing_report(active_sensing_report)?;

    if proposals.len() != active_sensing_report.actions.len()
        || proposals.len() > MAX_MYCELIAL_HEALTH_BUDGET_ACTIONS
    {
        return Err(MycelialSampleHealthBudgetError::InventoryMismatch);
    }

    let mut proposal_by_id = BTreeMap::new();
    for proposal in proposals {
        if proposal_by_id.insert(proposal.action_id.as_str(), proposal).is_some() {
            return Err(MycelialSampleHealthBudgetError::DuplicateId(
                "proposal.action_id",
            ));
        }
    }

    let action_ids = active_sensing_report
        .actions
        .iter()
        .map(|action| action.action_id.as_str())
        .collect::<BTreeSet<_>>();
    if proposal_by_id.keys().copied().collect::<BTreeSet<_>>() != action_ids {
        return Err(MycelialSampleHealthBudgetError::InventoryMismatch);
    }

    let evaluations = proposals.iter().try_fold(0usize, |total, proposal| {
        total.checked_add(
            proposal
                .spatial_footprint_cell_ids
                .len()
                .saturating_mul(budget.photon_dose_limits.len().max(1)),
        )
    });
    if evaluations.is_none_or(|count| count > MAX_MYCELIAL_HEALTH_BUDGET_EVALUATIONS) {
        return Err(MycelialSampleHealthBudgetError::ResourceLimitExceeded(
            "sample-health action × footprint × channel evaluation budget",
        ));
    }

    for action in &active_sensing_report.actions {
        let proposal = proposal_by_id
            .get(action.action_id.as_str())
            .expect("proposal inventory equality was checked");
        validate_proposal(
            proposal,
            action.action_kind,
            &action.acquisition_model_digest,
            &action.cost_evidence_digest,
            state,
            budget,
        )?;
    }

    let prior_dose_by_cell_channel = prior_exposure_by_cell_channel(state)?;
    let mut eligible_actions = Vec::new();
    let mut rejected_actions = Vec::new();
    for action in &active_sensing_report.actions {
        let proposal = proposal_by_id
            .get(action.action_id.as_str())
            .expect("proposal inventory equality was checked");
        match evaluate_proposal(
            &action.action_id,
            action.action_kind,
            action.net_value_of_information,
            proposal,
            budget,
            state,
            &prior_dose_by_cell_channel,
        )? {
            Ok(eligible) => eligible_actions.push(eligible),
            Err(rejected) => rejected_actions.push(rejected),
        }
    }

    eligible_actions.sort_by(|left, right| {
        right
            .net_value_of_information
            .total_cmp(&left.net_value_of_information)
            .then_with(|| left.action_id.cmp(&right.action_id))
    });
    rejected_actions.sort_by(|left, right| left.action_id.cmp(&right.action_id));

    let recommended_action_id = eligible_actions
        .iter()
        .find(|action| action.net_value_of_information > 0.0)
        .map(|action| action.action_id.clone());

    let mut report = MycelialSampleHealthBudgetReport {
        schema_version: MYCELIAL_SAMPLE_HEALTH_BUDGET_SCHEMA_VERSION,
        active_sensing_report_digest: active_sensing_report_digest(active_sensing_report),
        active_sensing_evidence_digest: active_sensing_evidence.source_evidence_digest(),
        policy_digest: policy_digest(budget),
        sample_health_state_digest: state_digest(state),
        proposal_inventory_digest: proposal_inventory_digest(proposals),
        recommended_action_id,
        eligible_actions,
        rejected_actions,
        advisory_only: true,
        canonical_digest: String::new(),
    };
    report.canonical_digest = mycelial_sample_health_budget_report_digest(&report);
    Ok(report)
}

/// Recomputes the output digest from the complete deterministic report inventory.
pub fn mycelial_sample_health_budget_report_digest(
    report: &MycelialSampleHealthBudgetReport,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-sample-health-budget-report-v2");
    hasher.update(&report.schema_version.to_le_bytes());
    hash_string(&mut hasher, &report.active_sensing_report_digest);
    hash_string(&mut hasher, &report.active_sensing_evidence_digest);
    hash_string(&mut hasher, &report.policy_digest);
    hash_string(&mut hasher, &report.sample_health_state_digest);
    hash_string(&mut hasher, &report.proposal_inventory_digest);
    match &report.recommended_action_id {
        Some(action_id) => {
            hasher.update(&[1]);
            hash_string(&mut hasher, action_id);
        }
        None => hasher.update(&[0]),
    }
    hasher.update(&(report.eligible_actions.len() as u64).to_le_bytes());
    for action in &report.eligible_actions {
        hash_string(&mut hasher, &action.action_id);
        hash_string(&mut hasher, &action.acquisition_plan_digest);
        hash_string(&mut hasher, &action.device_capability_manifest_digest);
        hash_string(&mut hasher, &action.spatial_footprint_digest);
        hash_string(&mut hasher, &action.health_model_digest);
        hasher.update(&[match action.action_kind {
            MycelialAcquisitionActionKind::AdditionalZPlane => 0,
            MycelialAcquisitionActionKind::HighResolutionCrop => 1,
            MycelialAcquisitionActionKind::AlternateIllumination => 2,
            MycelialAcquisitionActionKind::NewTimepoint => 3,
            MycelialAcquisitionActionKind::HumanAnnotation => 4,
            MycelialAcquisitionActionKind::Other => 5,
        }]);
        hash_f64(&mut hasher, action.net_value_of_information);
        hasher.update(&action.planned_start_offset_ns.to_le_bytes());
        hasher.update(&action.projected_session_duration_ns.to_le_bytes());
        hasher.update(&action.projected_action_count.to_le_bytes());
        hasher.update(
            &(action.maximum_projected_photons_per_um2_by_channel.len() as u64).to_le_bytes(),
        );
        for dose in &action.maximum_projected_photons_per_um2_by_channel {
            hash_string(&mut hasher, &dose.channel_id);
            hash_f64(&mut hasher, dose.maximum_projected_photons_per_um2);
            hash_f64(&mut hasher, dose.irradiance_watts_per_cm2_upper);
            hasher.update(&dose.illumination_duration_ns.to_le_bytes());
        }
    }
    hasher.update(&(report.rejected_actions.len() as u64).to_le_bytes());
    for rejected in &report.rejected_actions {
        hash_string(&mut hasher, &rejected.action_id);
        hash_string(&mut hasher, &rejected.acquisition_plan_digest);
        hash_string(&mut hasher, &rejected.device_capability_manifest_digest);
        hash_string(&mut hasher, &rejected.spatial_footprint_digest);
        hasher.update(&(rejected.reasons.len() as u64).to_le_bytes());
        for reason in &rejected.reasons {
            hasher.update(&[*reason as u8]);
        }
    }
    hasher.update(&[u8::from(report.advisory_only)]);
    finish_digest(hasher)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mycelial_active_sensing::{
        MycelialAcquisitionActionValue, MycelialAcquisitionOutcomeValue,
    };

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, DIGEST_HEX_BYTES).collect()
    }

    fn budget() -> MycelialSampleHealthBudget {
        MycelialSampleHealthBudget {
            schema_version: MYCELIAL_SAMPLE_HEALTH_BUDGET_SCHEMA_VERSION,
            policy_id: "assay-policy-001".to_string(),
            policy_version: "1.0.0".to_string(),
            coordinate_frame_id: "specimen-frame-v1".to_string(),
            health_model_digest: digest('e'),
            maximum_session_duration_ns: 100,
            maximum_actions_per_session: 4,
            maximum_roi_area_um2: 50.0,
            maximum_local_temperature_rise_millikelvin: 500,
            photon_dose_limits: vec![MycelialPhotonDoseLimit {
                channel_id: "gfp".to_string(),
                minimum_wavelength_nm: 480,
                maximum_wavelength_nm: 510,
                calibration_digest: digest('f'),
                maximum_irradiance_watts_per_cm2: 0.01,
                maximum_continuous_illumination_duration_ns: 20,
                minimum_recovery_interval_ns: 5,
                maximum_cumulative_photons_per_um2: 10.0,
            }],
        }
    }

    fn state() -> MycelialSampleHealthState {
        MycelialSampleHealthState {
            sample_id: "sample-001".to_string(),
            session_id: "session-001".to_string(),
            device_capability_manifest_digest: digest('j'),
            coordinate_frame_id: "specimen-frame-v1".to_string(),
            elapsed_session_duration_ns: 20,
            completed_actions_in_session: 1,
            exposure_history_complete: true,
            exposure_history_evidence_digest: digest('l'),
            elapsed_since_last_optical_exposure_ns: Some(10),
            cumulative_photon_exposures: vec![MycelialCumulativePhotonExposure {
                sample_id: "sample-001".to_string(),
                spatial_cell_id: "cell-a".to_string(),
                channel_id: "gfp".to_string(),
                wavelength_nm: 488,
                photons_per_um2: 7.0,
                source_session_id: "session-000".to_string(),
                evidence_receipt_digest: digest('g'),
                calibration_digest: digest('f'),
            }],
        }
    }

    fn action(
        action_id: &str,
        kind: MycelialAcquisitionActionKind,
        net_voi: f64,
    ) -> MycelialAcquisitionActionValue {
        MycelialAcquisitionActionValue {
            action_id: action_id.to_string(),
            action_kind: kind,
            acquisition_model_id: "response-model".to_string(),
            acquisition_model_version: "1.0.0".to_string(),
            acquisition_model_digest: digest('a'),
            cost_evidence_digest: digest('b'),
            cost_in_loss_units: 1.0,
            expected_posterior_bayes_risk: 0.5,
            gross_value_of_information: net_voi + 1.0,
            net_value_of_information: net_voi,
            outcomes: vec![MycelialAcquisitionOutcomeValue {
                outcome_id: "outcome-1".to_string(),
                predictive_probability: 1.0,
                best_decision_id: Some("decision-1".to_string()),
                conditional_bayes_risk: Some(0.5),
                expected_risk_contribution: 0.5,
            }],
        }
    }

    fn test_skeleton() -> AnnotatedMycelialSkeleton {
        AnnotatedMycelialSkeleton {
            schema_version: MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION,
            frame_id: "frame-health".to_string(),
            coordinate_frame_id: "microscope-health".to_string(),
            calibration_digest: digest('b'),
            source_observation_digest: digest('a'),
            dimensionality: ReconstructionDimensionality::TwoD,
            length_unit: ReconstructionLengthUnit::Micrometer,
            producer_id: "human-annotation".to_string(),
            producer_version: "v1".to_string(),
            junctions: vec![
                AnnotatedJunction {
                    id: "A".to_string(),
                    kind: AnnotatedJunctionKind::Origin,
                    position: ReconstructionPoint3 { x: 0.0, y: 0.0, z: 0.0 },
                },
                AnnotatedJunction {
                    id: "B".to_string(),
                    kind: AnnotatedJunctionKind::Tip,
                    position: ReconstructionPoint3 { x: 1.0, y: 0.0, z: 0.0 },
                },
                AnnotatedJunction {
                    id: "C".to_string(),
                    kind: AnnotatedJunctionKind::Origin,
                    position: ReconstructionPoint3 { x: 10.0, y: 0.0, z: 0.0 },
                },
                AnnotatedJunction {
                    id: "D".to_string(),
                    kind: AnnotatedJunctionKind::Tip,
                    position: ReconstructionPoint3 { x: 11.0, y: 0.0, z: 0.0 },
                },
            ],
            segments: vec![
                AnnotatedHyphaSegment {
                    id: "ab".to_string(),
                    start_junction_id: "A".to_string(),
                    end_junction_id: "B".to_string(),
                    length: 1.0,
                },
                AnnotatedHyphaSegment {
                    id: "cd".to_string(),
                    start_junction_id: "C".to_string(),
                    end_junction_id: "D".to_string(),
                    length: 1.0,
                },
            ],
        }
    }

    fn test_candidate() -> MycelialFusionCandidate {
        MycelialFusionCandidate {
            candidate_id: "bridge".to_string(),
            source_tip_junction_id: "B".to_string(),
            target: MycelialFusionTarget::TipJunction {
                junction_id: "D".to_string(),
            },
            bridge_length: 1.0,
            bridge_length_unit: ReconstructionLengthUnit::Micrometer,
            bridge_length_basis: MycelialFusionBridgeLengthBasis::Measured,
            bridge_length_evidence_digest: digest('e'),
            fusion_probability: 0.5,
            calibration_policy_id: "calibration-v1".to_string(),
            probability_evidence_digest: digest('f'),
            conflict_group_id: None,
        }
    }

    fn active_report(
        action_values: Vec<MycelialAcquisitionActionValue>,
    ) -> MycelialSampleHealthActiveSensingEvidence {
        let skeleton = test_skeleton();
        let candidates = vec![test_candidate()];
        let posterior =
            evaluate_mycelial_graph_hypothesis_posterior(&skeleton, &candidates).unwrap();
        let decisions = vec![
            MycelialAcquisitionDecision {
                decision_id: "accept-fusion".to_string(),
                loss_scale_id: "decision-loss-v1".to_string(),
                losses_by_hypothesis: posterior
                    .hypotheses
                    .iter()
                    .map(|hypothesis| MycelialHypothesisLoss {
                        accepted_candidate_ids: hypothesis.accepted_candidate_ids.clone(),
                        loss: if hypothesis.accepted_candidate_ids.contains(&"bridge".to_string()) {
                            0.0
                        } else {
                            1_000.0
                        },
                    })
                    .collect(),
            },
            MycelialAcquisitionDecision {
                decision_id: "reject-fusion".to_string(),
                loss_scale_id: "decision-loss-v1".to_string(),
                losses_by_hypothesis: posterior
                    .hypotheses
                    .iter()
                    .map(|hypothesis| MycelialHypothesisLoss {
                        accepted_candidate_ids: hypothesis.accepted_candidate_ids.clone(),
                        loss: if hypothesis.accepted_candidate_ids.contains(&"bridge".to_string()) {
                            1_000.0
                        } else {
                            0.0
                        },
                    })
                    .collect(),
            },
        ];
        let actions = action_values
            .iter()
            .map(|value| MycelialAcquisitionAction {
                action_id: value.action_id.clone(),
                action_kind: value.action_kind,
                acquisition_model_id: value.acquisition_model_id.clone(),
                acquisition_model_version: value.acquisition_model_version.clone(),
                acquisition_model_digest: value.acquisition_model_digest.clone(),
                cost_evidence_digest: value.cost_evidence_digest.clone(),
                cost_in_loss_units: 500.0 - value.net_value_of_information,
                cost_scale_id: "decision-loss-v1".to_string(),
                outcomes: vec![
                    MycelialAcquisitionOutcome {
                        outcome_id: "no-fusion".to_string(),
                        likelihoods_by_hypothesis: posterior
                            .hypotheses
                            .iter()
                            .map(|hypothesis| MycelialHypothesisLikelihood {
                                accepted_candidate_ids: hypothesis.accepted_candidate_ids.clone(),
                                likelihood: if hypothesis.accepted_candidate_ids.contains(&"bridge".to_string()) {
                                    0.0
                                } else {
                                    1.0
                                },
                            })
                            .collect(),
                    },
                    MycelialAcquisitionOutcome {
                        outcome_id: "fusion".to_string(),
                        likelihoods_by_hypothesis: posterior
                            .hypotheses
                            .iter()
                            .map(|hypothesis| MycelialHypothesisLikelihood {
                                accepted_candidate_ids: hypothesis.accepted_candidate_ids.clone(),
                                likelihood: if hypothesis.accepted_candidate_ids.contains(&"bridge".to_string()) {
                                    1.0
                                } else {
                                    0.0
                                },
                            })
                            .collect(),
                    },
                ],
            })
            .collect::<Vec<_>>();
        let report = evaluate_mycelial_active_sensing(
            &skeleton,
            &candidates,
            &posterior,
            &decisions,
            &actions,
        )
        .unwrap();
        MycelialSampleHealthActiveSensingEvidence::new(
            report,
            skeleton,
            candidates,
            posterior,
            decisions,
            actions,
        )
        .unwrap()
    }

    fn proposal(
        action_id: &str,
        cells: &[&str],
        dose: f64,
        kind: MycelialAcquisitionActionKind,
    ) -> MycelialAcquisitionBudgetProposal {
        let cell_ids = cells.iter().map(|value| value.to_string()).collect::<Vec<_>>();
        MycelialAcquisitionBudgetProposal {
            action_id: action_id.to_string(),
            session_id: "session-001".to_string(),
            sample_id: "sample-001".to_string(),
            acquisition_plan_digest: digest('k'),
            device_capability_manifest_digest: digest('j'),
            coordinate_frame_id: "specimen-frame-v1".to_string(),
            spatial_footprint_digest: footprint_digest(
                "specimen-frame-v1",
                &digest('k'),
                &digest('j'),
                &cell_ids,
            ),
            spatial_footprint_cell_ids: cell_ids,
            required_illumination_channel_ids: if kind == MycelialAcquisitionActionKind::HumanAnnotation {
                Vec::new()
            } else {
                vec!["gfp".to_string()]
            },
            roi_area_um2: 20.0,
            planned_start_offset_ns: 20,
            duration_ns: 10,
            local_temperature_rise_upper_millikelvin: 100,
            acquisition_model_digest: digest('a'),
            cost_evidence_digest: digest('b'),
            health_model_digest: digest('e'),
            photon_dose_estimates: if kind == MycelialAcquisitionActionKind::HumanAnnotation {
                Vec::new()
            } else {
                vec![MycelialPhotonDoseEstimate {
                    channel_id: "gfp".to_string(),
                    wavelength_nm: 488,
                    photons_per_um2_upper: dose,
                    irradiance_watts_per_cm2_upper: 0.001,
                    illumination_duration_ns: 5,
                    calibration_digest: digest('f'),
                }]
            },
        }
    }

    #[test]
    fn active_sensing_evidence_bundle_rejects_tampered_voI_report() {
        let qualified = active_report(vec![action(
            "qualified",
            MycelialAcquisitionActionKind::HumanAnnotation,
            2.0,
        )]);
        let mut tampered_report = qualified.report.clone();
        tampered_report.actions[0].net_value_of_information += 0.25;
        let result = MycelialSampleHealthActiveSensingEvidence::new(
            tampered_report,
            qualified.skeleton.clone(),
            qualified.candidates.clone(),
            qualified.posterior.clone(),
            qualified.decisions.clone(),
            qualified.actions.clone(),
        );
        assert!(matches!(
            result,
            Err(MycelialSampleHealthBudgetError::InvalidActiveSensingReport(_))
        ));
    }

    #[test]
    fn high_voi_action_is_rejected_when_cumulative_photon_dose_exceeds_limit() {
        let report = active_report(vec![action(
            "high-voi",
            MycelialAcquisitionActionKind::AdditionalZPlane,
            10.0,
        )]);
        let proposal = proposal(
            "high-voi",
            &["cell-a"],
            4.0,
            MycelialAcquisitionActionKind::AdditionalZPlane,
        );
        let result =
            evaluate_mycelial_sample_health_budget(&report, &budget(), &state(), &[proposal])
                .unwrap();
        assert_eq!(result.recommended_action_id, None);
        assert_eq!(result.rejected_actions.len(), 1);
        assert_eq!(
            result.rejected_actions[0].reasons,
            vec![MycelialSampleHealthRejectionReason::CumulativePhotonDoseExceeded]
        );
    }

    #[test]
    fn overlapping_rois_accumulate_dose_on_shared_spatial_cells() {
        let report = active_report(vec![
            action("roi-a", MycelialAcquisitionActionKind::HighResolutionCrop, 3.0),
            action("roi-b", MycelialAcquisitionActionKind::HighResolutionCrop, 5.0),
        ]);
        let proposals = vec![
            proposal("roi-a", &["cell-a", "cell-b"], 1.0, MycelialAcquisitionActionKind::HighResolutionCrop),
            proposal("roi-b", &["cell-a", "cell-c"], 1.0, MycelialAcquisitionActionKind::HighResolutionCrop),
        ];
        let mut current = state();
        current.cumulative_photon_exposures[0].photons_per_um2 = 8.0;
        current.cumulative_photon_exposures.push(MycelialCumulativePhotonExposure {
            sample_id: "sample-001".to_string(),
            spatial_cell_id: "cell-a".to_string(),
            channel_id: "gfp".to_string(),
            wavelength_nm: 488,
            photons_per_um2: 1.5,
            source_session_id: "session-previous-other".to_string(),
            evidence_receipt_digest: digest('h'),
            calibration_digest: digest('f'),
        });
        let result =
            evaluate_mycelial_sample_health_budget(&report, &budget(), &current, &proposals)
                .unwrap();
        assert_eq!(result.rejected_actions.len(), 2);
        assert!(result.rejected_actions.iter().all(|item| {
            item.reasons.contains(
                &MycelialSampleHealthRejectionReason::CumulativePhotonDoseExceeded
            )
        }));
    }

    #[test]
    fn non_overlapping_spatial_cells_do_not_share_prior_dose() {
        let report = active_report(vec![action(
            "new-cell",
            MycelialAcquisitionActionKind::AdditionalZPlane,
            4.0,
        )]);
        let proposal = proposal(
            "new-cell",
            &["cell-b"],
            4.0,
            MycelialAcquisitionActionKind::AdditionalZPlane,
        );
        let result =
            evaluate_mycelial_sample_health_budget(&report, &budget(), &state(), &[proposal])
                .unwrap();
        assert_eq!(result.recommended_action_id.as_deref(), Some("new-cell"));
    }

    #[test]
    fn missing_spatial_footprint_is_explicitly_rejected() {
        let report = active_report(vec![action(
            "no-footprint",
            MycelialAcquisitionActionKind::HumanAnnotation,
            4.0,
        )]);
        let proposal = proposal(
            "no-footprint",
            &[],
            0.0,
            MycelialAcquisitionActionKind::HumanAnnotation,
        );
        let result =
            evaluate_mycelial_sample_health_budget(&report, &budget(), &state(), &[proposal])
                .unwrap();
        assert_eq!(result.recommended_action_id, None);
        assert!(result.rejected_actions[0].reasons.contains(
            &MycelialSampleHealthRejectionReason::MissingSpatialFootprint
        ));
    }

    #[test]
    fn exact_photon_dose_boundary_is_admissible() {
        let report = active_report(vec![action(
            "boundary",
            MycelialAcquisitionActionKind::AdditionalZPlane,
            2.0,
        )]);
        let proposal = proposal(
            "boundary",
            &["cell-a"],
            3.0,
            MycelialAcquisitionActionKind::AdditionalZPlane,
        );
        let result =
            evaluate_mycelial_sample_health_budget(&report, &budget(), &state(), &[proposal])
                .unwrap();
        assert_eq!(result.recommended_action_id.as_deref(), Some("boundary"));
    }

    #[test]
    fn rejected_high_voi_action_cannot_outrank_feasible_lower_voi_action() {
        let report = active_report(vec![
            action("unsafe-high", MycelialAcquisitionActionKind::AdditionalZPlane, 100.0),
            action("safe-low", MycelialAcquisitionActionKind::HumanAnnotation, 1.0),
        ]);
        let proposals = vec![
            proposal("unsafe-high", &["cell-a"], 4.0, MycelialAcquisitionActionKind::AdditionalZPlane),
            proposal("safe-low", &["cell-b"], 0.0, MycelialAcquisitionActionKind::HumanAnnotation),
        ];
        let result =
            evaluate_mycelial_sample_health_budget(&report, &budget(), &state(), &proposals)
                .unwrap();
        assert_eq!(result.recommended_action_id.as_deref(), Some("safe-low"));
    }

    #[test]
    fn human_annotation_consumes_time_but_no_photon_dose() {
        let report = active_report(vec![action(
            "human-review",
            MycelialAcquisitionActionKind::HumanAnnotation,
            2.0,
        )]);
        let proposal = proposal(
            "human-review",
            &["cell-a"],
            0.0,
            MycelialAcquisitionActionKind::HumanAnnotation,
        );
        let result =
            evaluate_mycelial_sample_health_budget(&report, &budget(), &state(), &[proposal])
                .unwrap();
        assert_eq!(result.recommended_action_id.as_deref(), Some("human-review"));
        assert!(result.eligible_actions[0]
            .maximum_projected_photons_per_um2_by_channel
            .is_empty());
    }

    #[test]
    fn session_time_roi_and_temperature_are_hard_limits() {
        let report = active_report(vec![action(
            "over-budget",
            MycelialAcquisitionActionKind::AdditionalZPlane,
            50.0,
        )]);
        let mut proposal = proposal(
            "over-budget",
            &["cell-b"],
            1.0,
            MycelialAcquisitionActionKind::AdditionalZPlane,
        );
        proposal.duration_ns = 90;
        proposal.roi_area_um2 = 100.0;
        proposal.local_temperature_rise_upper_millikelvin = 600;
        let result =
            evaluate_mycelial_sample_health_budget(&report, &budget(), &state(), &[proposal])
                .unwrap();
        assert_eq!(result.recommended_action_id, None);
        assert!(result.rejected_actions[0].reasons.contains(
            &MycelialSampleHealthRejectionReason::SessionDurationExceeded
        ));
        assert!(result.rejected_actions[0].reasons.contains(
            &MycelialSampleHealthRejectionReason::RoiAreaExceeded
        ));
        assert!(result.rejected_actions[0].reasons.contains(
            &MycelialSampleHealthRejectionReason::TemperatureRiseExceeded
        ));
    }

    #[test]
    fn irradiance_upper_bound_is_enforced_independently_of_total_dose() {
        let evidence = active_report(vec![action(
            "high-irradiance",
            MycelialAcquisitionActionKind::AlternateIllumination,
            8.0,
        )]);
        let mut proposal = proposal(
            "high-irradiance",
            &["cell-b"],
            1.0,
            MycelialAcquisitionActionKind::AlternateIllumination,
        );
        proposal.photon_dose_estimates[0].irradiance_watts_per_cm2_upper = 0.02;
        let result =
            evaluate_mycelial_sample_health_budget(&evidence, &budget(), &state(), &[proposal])
                .unwrap();
        assert_eq!(result.recommended_action_id, None);
        assert!(result.rejected_actions[0]
            .reasons
            .contains(&MycelialSampleHealthRejectionReason::IrradianceExceeded));
    }

    #[test]
    fn continuous_illumination_duration_is_a_separate_hard_limit() {
        let evidence = active_report(vec![action(
            "long-dwell",
            MycelialAcquisitionActionKind::AlternateIllumination,
            8.0,
        )]);
        let mut proposal = proposal(
            "long-dwell",
            &["cell-b"],
            1.0,
            MycelialAcquisitionActionKind::AlternateIllumination,
        );
        proposal.duration_ns = 30;
        proposal.photon_dose_estimates[0].illumination_duration_ns = 21;
        let result =
            evaluate_mycelial_sample_health_budget(&evidence, &budget(), &state(), &[proposal])
                .unwrap();
        assert_eq!(result.recommended_action_id, None);
        assert!(result.rejected_actions[0].reasons.contains(
            &MycelialSampleHealthRejectionReason::ContinuousIlluminationDurationExceeded
        ));
    }

    #[test]
    fn recovery_interval_can_be_met_by_delaying_the_next_acquisition() {
        let evidence = active_report(vec![action(
            "recovery-wait",
            MycelialAcquisitionActionKind::AlternateIllumination,
            8.0,
        )]);
        let mut proposal = proposal(
            "recovery-wait",
            &["cell-b"],
            1.0,
            MycelialAcquisitionActionKind::AlternateIllumination,
        );
        let mut current = state();
        current.elapsed_since_last_optical_exposure_ns = Some(1);
        let rejected =
            evaluate_mycelial_sample_health_budget(
                &evidence,
                &budget(),
                &current,
                &[proposal.clone()],
            )
            .unwrap();
        assert!(rejected.rejected_actions[0]
            .reasons
            .contains(&MycelialSampleHealthRejectionReason::RecoveryIntervalNotMet));

        proposal.planned_start_offset_ns = 24;
        proposal.spatial_footprint_digest = footprint_digest(
            &proposal.coordinate_frame_id,
            &proposal.acquisition_plan_digest,
            &proposal.device_capability_manifest_digest,
            &proposal.spatial_footprint_cell_ids,
        );
        let eligible =
            evaluate_mycelial_sample_health_budget(&evidence, &budget(), &current, &[proposal])
                .unwrap();
        assert_eq!(eligible.recommended_action_id.as_deref(), Some("recovery-wait"));
    }

    #[test]
    fn incomplete_exposure_history_blocks_optical_but_not_human_review() {
        let mut current = state();
        current.exposure_history_complete = false;
        let optical_evidence = active_report(vec![action(
            "unknown-exposure-history",
            MycelialAcquisitionActionKind::AdditionalZPlane,
            8.0,
        )]);
        let optical_proposal = proposal(
            "unknown-exposure-history",
            &["cell-b"],
            1.0,
            MycelialAcquisitionActionKind::AdditionalZPlane,
        );
        let blocked = evaluate_mycelial_sample_health_budget(
            &optical_evidence,
            &budget(),
            &current,
            &[optical_proposal],
        )
        .unwrap();
        assert!(blocked.rejected_actions[0].reasons.contains(
            &MycelialSampleHealthRejectionReason::ExposureHistoryIncomplete
        ));

        let review_evidence = active_report(vec![action(
            "human-review-with-incomplete-history",
            MycelialAcquisitionActionKind::HumanAnnotation,
            1.0,
        )]);
        let review_proposal = proposal(
            "human-review-with-incomplete-history",
            &["cell-b"],
            0.0,
            MycelialAcquisitionActionKind::HumanAnnotation,
        );
        let review = evaluate_mycelial_sample_health_budget(
            &review_evidence,
            &budget(),
            &current,
            &[review_proposal],
        )
        .unwrap();
        assert_eq!(
            review.recommended_action_id.as_deref(),
            Some("human-review-with-incomplete-history")
        );
    }

    #[test]
    fn complete_exposure_history_requires_consistent_recovery_clock() {
        let evidence = active_report(vec![action(
            "inconsistent-clock",
            MycelialAcquisitionActionKind::HumanAnnotation,
            1.0,
        )]);
        let proposal = proposal(
            "inconsistent-clock",
            &["cell-b"],
            0.0,
            MycelialAcquisitionActionKind::HumanAnnotation,
        );
        let mut current = state();
        current.elapsed_since_last_optical_exposure_ns = None;
        assert!(matches!(
            evaluate_mycelial_sample_health_budget(
                &evidence,
                &budget(),
                &current,
                &[proposal]
            ),
            Err(MycelialSampleHealthBudgetError::InvalidState(_))
        ));
    }

    #[test]
    fn unknown_channel_and_out_of_band_wavelength_are_rejected() {
        let report = active_report(vec![action(
            "wrong-channel",
            MycelialAcquisitionActionKind::AlternateIllumination,
            1.0,
        )]);
        let mut proposal = proposal(
            "wrong-channel",
            &["cell-b"],
            1.0,
            MycelialAcquisitionActionKind::AlternateIllumination,
        );
        proposal.photon_dose_estimates[0].channel_id = "unknown".to_string();
        let result =
            evaluate_mycelial_sample_health_budget(&report, &budget(), &state(), &[proposal])
                .unwrap();
        assert!(result.rejected_actions[0].reasons.contains(
            &MycelialSampleHealthRejectionReason::UnknownIlluminationChannel
        ));

        let mut proposal = proposal(
            "wrong-channel",
            &["cell-b"],
            1.0,
            MycelialAcquisitionActionKind::AlternateIllumination,
        );
        proposal.photon_dose_estimates[0].wavelength_nm = 550;
        let result =
            evaluate_mycelial_sample_health_budget(&report, &budget(), &state(), &[proposal])
                .unwrap();
        assert!(result.rejected_actions[0].reasons.contains(
            &MycelialSampleHealthRejectionReason::WavelengthOutsideCalibratedBand
        ));
    }

    #[test]
    fn every_required_illumination_channel_needs_a_dose_estimate() {
        let report = active_report(vec![action(
            "missing-one-channel",
            MycelialAcquisitionActionKind::AlternateIllumination,
            8.0,
        )]);
        let mut proposal = proposal(
            "missing-one-channel",
            &["cell-b"],
            1.0,
            MycelialAcquisitionActionKind::AlternateIllumination,
        );
        proposal.required_illumination_channel_ids =
            vec!["gfp".to_string(), "red".to_string()];
        let mut policy = budget();
        policy.photon_dose_limits.push(MycelialPhotonDoseLimit {
            channel_id: "red".to_string(),
            minimum_wavelength_nm: 600,
            maximum_wavelength_nm: 650,
            calibration_digest: digest('i'),
            maximum_irradiance_watts_per_cm2: 0.01,
            maximum_continuous_illumination_duration_ns: 20,
            minimum_recovery_interval_ns: 5,
            maximum_cumulative_photons_per_um2: 10.0,
        });
        let result =
            evaluate_mycelial_sample_health_budget(&report, &policy, &state(), &[proposal])
                .unwrap();
        assert!(result.rejected_actions[0].reasons.contains(
            &MycelialSampleHealthRejectionReason::MissingOpticalDoseEstimate
        ));
        assert_eq!(result.recommended_action_id, None);
    }

    #[test]
    fn proposal_device_capability_must_match_active_session() {
        let evidence = active_report(vec![action(
            "wrong-device",
            MycelialAcquisitionActionKind::HumanAnnotation,
            1.0,
        )]);
        let mut proposal = proposal(
            "wrong-device",
            &["cell-a"],
            0.0,
            MycelialAcquisitionActionKind::HumanAnnotation,
        );
        proposal.device_capability_manifest_digest = digest('z');
        assert!(matches!(
            evaluate_mycelial_sample_health_budget(&evidence, &budget(), &state(), &[proposal]),
            Err(MycelialSampleHealthBudgetError::InvalidProposal(_))
        ));
    }

    #[test]
    fn policy_and_proposal_health_model_digests_must_match() {
        let report = active_report(vec![action(
            "wrong-health-model",
            MycelialAcquisitionActionKind::HumanAnnotation,
            2.0,
        )]);
        let mut proposal = proposal(
            "wrong-health-model",
            &["cell-a"],
            0.0,
            MycelialAcquisitionActionKind::HumanAnnotation,
        );
        proposal.health_model_digest = digest('x');
        assert!(matches!(
            evaluate_mycelial_sample_health_budget(&report, &budget(), &state(), &[proposal]),
            Err(MycelialSampleHealthBudgetError::InvalidProposal(_))
        ));
    }

    #[test]
    fn dose_estimate_must_use_the_policy_bound_channel_calibration() {
        let report = active_report(vec![action(
            "wrong-calibration",
            MycelialAcquisitionActionKind::AdditionalZPlane,
            5.0,
        )]);
        let mut proposal = proposal(
            "wrong-calibration",
            &["cell-b"],
            1.0,
            MycelialAcquisitionActionKind::AdditionalZPlane,
        );
        proposal.photon_dose_estimates[0].calibration_digest = digest('x');
        let result =
            evaluate_mycelial_sample_health_budget(&report, &budget(), &state(), &[proposal])
                .unwrap();
        assert_eq!(result.recommended_action_id, None);
        assert!(result.rejected_actions[0].reasons.contains(
            &MycelialSampleHealthRejectionReason::PhotonCalibrationMismatch
        ));
    }

    #[test]
    fn historical_exposure_must_use_the_policy_bound_channel_calibration() {
        let report = active_report(vec![action(
            "historical-calibration",
            MycelialAcquisitionActionKind::AdditionalZPlane,
            5.0,
        )]);
        let proposal = proposal(
            "historical-calibration",
            &["cell-b"],
            1.0,
            MycelialAcquisitionActionKind::AdditionalZPlane,
        );
        let mut current = state();
        current.cumulative_photon_exposures[0].calibration_digest = digest('x');
        assert!(matches!(
            evaluate_mycelial_sample_health_budget(&report, &budget(), &current, &[proposal]),
            Err(MycelialSampleHealthBudgetError::InvalidState(_))
        ));
    }

    #[test]
    fn missing_optical_dose_model_fails_closed() {
        let report = active_report(vec![action(
            "missing-dose",
            MycelialAcquisitionActionKind::AdditionalZPlane,
            10.0,
        )]);
        let mut proposal = proposal(
            "missing-dose",
            &["cell-b"],
            1.0,
            MycelialAcquisitionActionKind::AdditionalZPlane,
        );
        proposal.photon_dose_estimates.clear();
        let result =
            evaluate_mycelial_sample_health_budget(&report, &budget(), &state(), &[proposal])
                .unwrap();
        assert_eq!(result.recommended_action_id, None);
        assert!(result.rejected_actions[0].reasons.contains(
            &MycelialSampleHealthRejectionReason::MissingOpticalDoseEstimate
        ));
    }

    #[test]
    fn no_positive_net_value_action_is_recommended() {
        let report = active_report(vec![action(
            "negative-value",
            MycelialAcquisitionActionKind::HumanAnnotation,
            -0.1,
        )]);
        let proposal = proposal(
            "negative-value",
            &["cell-b"],
            0.0,
            MycelialAcquisitionActionKind::HumanAnnotation,
        );
        let result =
            evaluate_mycelial_sample_health_budget(&report, &budget(), &state(), &[proposal])
                .unwrap();
        assert_eq!(result.recommended_action_id, None);
        assert_eq!(result.eligible_actions.len(), 1);
    }

    #[test]
    fn action_count_budget_is_enforced() {
        let report = active_report(vec![action(
            "one-too-many",
            MycelialAcquisitionActionKind::HumanAnnotation,
            3.0,
        )]);
        let mut current = state();
        current.completed_actions_in_session = 4;
        let proposal = proposal(
            "one-too-many",
            &["cell-b"],
            0.0,
            MycelialAcquisitionActionKind::HumanAnnotation,
        );
        let result =
            evaluate_mycelial_sample_health_budget(&report, &budget(), &current, &[proposal])
                .unwrap();
        assert!(result.rejected_actions[0].reasons.contains(
            &MycelialSampleHealthRejectionReason::SessionActionCountExceeded
        ));
    }

    #[test]
    fn output_digest_is_stable_and_recomputable() {
        let report = active_report(vec![
            action("first", MycelialAcquisitionActionKind::HumanAnnotation, 2.0),
            action("second", MycelialAcquisitionActionKind::HumanAnnotation, 1.0),
        ]);
        let proposals = vec![
            proposal("first", &["cell-a"], 0.0, MycelialAcquisitionActionKind::HumanAnnotation),
            proposal("second", &["cell-b"], 0.0, MycelialAcquisitionActionKind::HumanAnnotation),
        ];
        let policy = budget();
        let state = state();
        let first =
            evaluate_mycelial_sample_health_budget(&report, &policy, &state, &proposals).unwrap();
        let second =
            evaluate_mycelial_sample_health_budget(&report, &policy, &state, &proposals).unwrap();
        assert_eq!(first.canonical_digest, second.canonical_digest);
        assert_eq!(
            first.canonical_digest,
            mycelial_sample_health_budget_report_digest(&first)
        );
    }

    #[test]
    fn report_replay_rejects_tampered_recommendation_and_digest() {
        let evidence = active_report(vec![action(
            "replay",
            MycelialAcquisitionActionKind::HumanAnnotation,
            2.0,
        )]);
        let policy = budget();
        let current = state();
        let proposals = vec![proposal(
            "replay",
            &["cell-b"],
            0.0,
            MycelialAcquisitionActionKind::HumanAnnotation,
        )];
        let mut report =
            evaluate_mycelial_sample_health_budget(&evidence, &policy, &current, &proposals)
                .unwrap();
        report
            .validate_against(&evidence, &policy, &current, &proposals)
            .unwrap();

        report.recommended_action_id = None;
        assert_eq!(
            report.validate_against(&evidence, &policy, &current, &proposals),
            Err(MycelialSampleHealthBudgetError::ReportMismatch)
        );
    }

    #[test]
    fn proposal_inventory_must_exactly_match_active_actions() {
        let report = active_report(vec![action(
            "action-one",
            MycelialAcquisitionActionKind::HumanAnnotation,
            2.0,
        )]);
        let proposal = proposal(
            "other-action",
            &["cell-b"],
            0.0,
            MycelialAcquisitionActionKind::HumanAnnotation,
        );
        assert_eq!(
            evaluate_mycelial_sample_health_budget(&report, &budget(), &state(), &[proposal]),
            Err(MycelialSampleHealthBudgetError::InventoryMismatch)
        );
    }

    #[test]
    fn policy_rejects_duplicate_channels_and_invalid_limits() {
        let mut invalid = budget();
        invalid.photon_dose_limits.push(invalid.photon_dose_limits[0].clone());
        assert!(matches!(
            validate_budget(&invalid),
            Err(MycelialSampleHealthBudgetError::DuplicateId(_))
        ));
        let mut invalid = budget();
        invalid.photon_dose_limits[0].maximum_cumulative_photons_per_um2 = f64::NAN;
        assert!(matches!(
            validate_budget(&invalid),
            Err(MycelialSampleHealthBudgetError::InvalidBudget(_))
        ));
    }
}
