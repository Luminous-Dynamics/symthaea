// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-aware peak resource usage measurement and forecast drift (SX-035).
//!
//! Observations are compared against an exact, freshly re-evaluated SX-034
//! reservation plan. The evaluator distinguishes declared forecasts from
//! observations and never authenticates sensor evidence or promotes maturity.
//! It has no actuation, scheduling, or hardware-qualification authority.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

use crate::suit_mission_configuration::MissionConfiguration;
use crate::suit_mission_tasks::{MissionTaskError, MissionTaskReceipt, MissionTaskRequirement};
use crate::suit_peak_reservations::{
    evaluate_peak_resource_reservations, MissionTimeWindow, PeakQuantityInterval,
    PeakResourceKind, PeakResourceReservationPlan, PeakResourceReservationReceipt,
    PeakResourceReservationError, PeakResourceReservationReport, PeakResourceReservationStatus,
};
use crate::suit_profile::SuitEvidenceLevel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ResourceMeasurementMethod {
    SimulatedTrace,
    BenchInstrumented,
    HumanInLoopInstrumented,
    OperationalTelemetry,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeakResourceUsageObservation {
    pub observation_id: String,
    pub pool_id: String,
    pub resource_kind: PeakResourceKind,
    pub provider_component_id: String,
    pub provider_hardware_revision: String,
    pub window: MissionTimeWindow,
    /// Peak-over-window interval, not an average, instantaneous sample, or total.
    pub observed_peak: PeakQuantityInterval,
    pub sample_count: u64,
    /// Declared maximum gap/resolution between samples, in mission-relative ms.
    pub sampling_interval_ms: u64,
    pub measurement_method: ResourceMeasurementMethod,
    pub evidence_level: SuitEvidenceLevel,
    /// Immutable artifact URI, test record, telemetry batch or digest; unauthenticated.
    pub evidence_reference: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeakResourceUsageMeasurementPlan {
    pub schema_version: u32,
    pub mission_id: String,
    pub configuration_revision: String,
    pub expected_configuration_digest_sha256: String,
    pub expected_task_plan_digest_sha256: String,
    pub expected_reservation_plan_digest_sha256: String,
    pub expected_reservation_report_digest_sha256: String,
    pub expected_reservation_receipt_digest_sha256: String,
    pub minimum_evidence_level: SuitEvidenceLevel,
    pub minimum_sample_count: u64,
    pub maximum_sampling_interval_ms: u64,
    pub observations: Vec<PeakResourceUsageObservation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PeakUsageFindingCode {
    ProviderComponentMismatch,
    ProviderHardwareRevisionMismatch,
    MeasurementEvidenceBelowPolicy,
    SampleCountBelowPolicy,
    SamplingIntervalAbovePolicy,
    NoDeclaredForecast,
    ForecastMayBeUnderestimated,
    ForecastDefinitelyUnderestimated,
    PossibleBudgetLimitExceeded,
    DefiniteBudgetLimitExceeded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PeakUsageFindingSeverity {
    Degradation,
    Blocker,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeakUsageFinding {
    pub observation_id: String,
    pub pool_id: String,
    pub code: PeakUsageFindingCode,
    pub severity: PeakUsageFindingSeverity,
    pub forecast_peak_upper: Option<u64>,
    pub usable_capacity: u64,
    pub observed_peak: PeakQuantityInterval,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PeakUsageObservationStatus {
    WithinDeclaredForecast,
    ForecastMayBeUnderestimated,
    ForecastDefinitelyUnderestimated,
    PossibleBudgetLimitExceeded,
    DefiniteBudgetLimitExceeded,
    UnmodeledUsage,
    EvidenceInsufficient,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeakUsageObservationEvaluation {
    pub observation_id: String,
    pub pool_id: String,
    pub resource_kind: PeakResourceKind,
    pub provider_component_id: String,
    pub provider_hardware_revision: String,
    pub window: MissionTimeWindow,
    pub observed_peak: PeakQuantityInterval,
    pub sample_count: u64,
    pub sampling_interval_ms: u64,
    pub measurement_method: ResourceMeasurementMethod,
    pub evidence_level: SuitEvidenceLevel,
    pub evidence_reference: String,
    pub forecast_window_count: usize,
    pub forecast_peak_upper: Option<u64>,
    pub usable_capacity: u64,
    pub findings: Vec<PeakUsageFinding>,
    pub status: PeakUsageObservationStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PeakResourceUsageMeasurementStatus {
    Blocked,
    Degraded,
    MeetsDeclaredMeasurementPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeakResourceUsageMeasurementReport {
    pub schema_version: u32,
    pub mission_id: String,
    pub configuration_revision: String,
    pub configuration_digest_sha256: String,
    pub task_plan_digest_sha256: String,
    pub reservation_plan_digest_sha256: String,
    pub reservation_report_digest_sha256: String,
    pub measurement_plan_digest_sha256: String,
    pub baseline_reservation_status: PeakResourceReservationStatus,
    pub observations: Vec<PeakUsageObservationEvaluation>,
    pub findings: Vec<PeakUsageFinding>,
    pub status: PeakResourceUsageMeasurementStatus,
    pub observation_count: usize,
    pub forecasted_observation_count: usize,
    pub unmodeled_observation_count: usize,
    pub forecast_drift_count: usize,
    pub possible_budget_limit_exceedance_count: usize,
    pub definite_budget_limit_exceedance_count: usize,
    /// Always false: measurement bookkeeping is not physical qualification.
    pub represents_physical_qualification: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeakResourceUsageMeasurementReceipt {
    pub configuration_digest_sha256: String,
    pub task_plan_digest_sha256: String,
    pub task_receipt_digest_sha256: String,
    pub reservation_plan_digest_sha256: String,
    pub reservation_receipt_digest_sha256: String,
    pub measurement_plan_digest_sha256: String,
    pub report_digest_sha256: String,
    /// Deterministic change evidence, not a digital signature/authentication.
    pub receipt_digest_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PeakResourceUsageMeasurementError {
    UnsupportedSchemaVersion,
    InvalidPlanIdentity,
    InvalidPinnedDigest,
    InvalidSamplingPolicy,
    EmptyObservationSet,
    InvalidObservation,
    InvalidTimeWindow,
    InvalidQuantityInterval,
    DuplicateObservationId,
    MissingReferencedPool,
    ResourceKindMismatch,
    MissionIdentityMismatch,
    ConfigurationDigestMismatch,
    TaskPlanDigestMismatch,
    ReservationPlanDigestMismatch,
    ReservationReportDigestMismatch,
    ReservationReceiptDigestMismatch,
    TaskReceiptMismatch,
    TaskEvaluationFailed(MissionTaskError),
    ReservationBaselineEvaluationFailed(PeakResourceReservationError),
}

pub fn evaluate_peak_resource_usage(
    configuration: &MissionConfiguration,
    tasks: &[MissionTaskRequirement],
    task_receipt: &MissionTaskReceipt,
    reservation_plan: &PeakResourceReservationPlan,
    reservation_report: &PeakResourceReservationReport,
    reservation_receipt: &PeakResourceReservationReceipt,
    measurement_plan: &PeakResourceUsageMeasurementPlan,
) -> Result<
    (
        PeakResourceUsageMeasurementReport,
        PeakResourceUsageMeasurementReceipt,
    ),
    PeakResourceUsageMeasurementError,
> {
    validate_measurement_plan(measurement_plan)?;

    let (fresh_report, fresh_receipt) = evaluate_peak_resource_reservations(
        configuration,
        tasks,
        task_receipt,
        reservation_plan,
    )
    .map_err(PeakResourceUsageMeasurementError::ReservationBaselineEvaluationFailed)?;
    if &fresh_report != reservation_report {
        return Err(PeakResourceUsageMeasurementError::ReservationReportDigestMismatch);
    }
    if &fresh_receipt != reservation_receipt {
        return Err(PeakResourceUsageMeasurementError::ReservationReceiptDigestMismatch);
    }
    if fresh_report.status != reservation_report.status {
        return Err(PeakResourceUsageMeasurementError::ReservationReportDigestMismatch);
    }

    if measurement_plan.mission_id != configuration.mission_id
        || measurement_plan.configuration_revision != configuration.configuration_revision
    {
        return Err(PeakResourceUsageMeasurementError::MissionIdentityMismatch);
    }
    if measurement_plan.expected_configuration_digest_sha256
        != fresh_report.configuration_digest_sha256
    {
        return Err(PeakResourceUsageMeasurementError::ConfigurationDigestMismatch);
    }
    if measurement_plan.expected_task_plan_digest_sha256 != fresh_report.task_plan_digest_sha256 {
        return Err(PeakResourceUsageMeasurementError::TaskPlanDigestMismatch);
    }
    if measurement_plan.expected_reservation_plan_digest_sha256
        != fresh_report.reservation_plan_digest_sha256
    {
        return Err(PeakResourceUsageMeasurementError::ReservationPlanDigestMismatch);
    }
    if measurement_plan.expected_reservation_report_digest_sha256
        != reservation_receipt.report_digest_sha256
    {
        return Err(PeakResourceUsageMeasurementError::ReservationReportDigestMismatch);
    }
    if measurement_plan.expected_reservation_receipt_digest_sha256
        != reservation_receipt.receipt_digest_sha256
    {
        return Err(PeakResourceUsageMeasurementError::ReservationReceiptDigestMismatch);
    }
    validate_observation_references(measurement_plan, reservation_plan)?;

    let pools: BTreeMap<&str, _> = reservation_plan
        .pools
        .iter()
        .map(|pool| (pool.pool_id.as_str(), pool))
        .collect();
    let pool_evaluations: BTreeMap<&str, _> = reservation_report
        .pools
        .iter()
        .map(|pool| (pool.pool_id.as_str(), pool))
        .collect();
    let measurement_plan_digest_sha256 = digest_measurement_plan(measurement_plan);
    let mut ordered: Vec<&PeakResourceUsageObservation> =
        measurement_plan.observations.iter().collect();
    ordered.sort_by(|left, right| left.observation_id.cmp(&right.observation_id));

    let mut observations = Vec::with_capacity(ordered.len());
    let mut findings = Vec::new();
    let mut forecasted_observation_count = 0;
    let mut unmodeled_observation_count = 0;
    let mut forecast_drift_count = 0;
    let mut possible_budget_limit_exceedance_count = 0;
    let mut definite_budget_limit_exceedance_count = 0;

    for observation in ordered {
        let pool = pools[observation.pool_id.as_str()];
        let pool_evaluation = pool_evaluations[observation.pool_id.as_str()];
        let mut forecast_peak_upper: Option<u64> = None;
        let mut forecast_window_count = 0;
        for window in &reservation_report.windows {
            if window.pool_id != observation.pool_id
                || !windows_overlap(window.window, observation.window)
            {
                continue;
            }
            forecast_window_count += 1;
            let total_upper = window
                .required_demand_upper
                .checked_add(window.optional_demand_upper)
                .ok_or(PeakResourceUsageMeasurementError::InvalidQuantityInterval)?;
            forecast_peak_upper =
                Some(forecast_peak_upper.map_or(total_upper, |current| current.max(total_upper)));
        }

        let mut observation_findings = Vec::new();
        let mut add_finding = |code: PeakUsageFindingCode,
                               severity: PeakUsageFindingSeverity,
                               forecast: Option<u64>| {
            observation_findings.push(PeakUsageFinding {
                observation_id: observation.observation_id.clone(),
                pool_id: observation.pool_id.clone(),
                code,
                severity,
                forecast_peak_upper: forecast,
                usable_capacity: pool_evaluation.usable_capacity,
                observed_peak: observation.observed_peak,
            });
        };

        if observation.provider_component_id != pool.provider_component_id {
            add_finding(
                PeakUsageFindingCode::ProviderComponentMismatch,
                PeakUsageFindingSeverity::Degradation,
                forecast_peak_upper,
            );
        }
        if observation.provider_hardware_revision != pool.provider_hardware_revision {
            add_finding(
                PeakUsageFindingCode::ProviderHardwareRevisionMismatch,
                PeakUsageFindingSeverity::Degradation,
                forecast_peak_upper,
            );
        }
        if observation.evidence_level < measurement_plan.minimum_evidence_level {
            add_finding(
                PeakUsageFindingCode::MeasurementEvidenceBelowPolicy,
                PeakUsageFindingSeverity::Degradation,
                forecast_peak_upper,
            );
        }
        if observation.sample_count < measurement_plan.minimum_sample_count {
            add_finding(
                PeakUsageFindingCode::SampleCountBelowPolicy,
                PeakUsageFindingSeverity::Degradation,
                forecast_peak_upper,
            );
        }
        if observation.sampling_interval_ms > measurement_plan.maximum_sampling_interval_ms {
            add_finding(
                PeakUsageFindingCode::SamplingIntervalAbovePolicy,
                PeakUsageFindingSeverity::Degradation,
                forecast_peak_upper,
            );
        }

        // Compare against SX-034's usable capacity after reserve. This is a
        // declared budget boundary, not a claim about the hardware's physical limit.
        let definite_capacity_exceedance =
            observation.observed_peak.lower > pool_evaluation.usable_capacity;
        let possible_capacity_exceedance =
            observation.observed_peak.upper > pool_evaluation.usable_capacity
                && !definite_capacity_exceedance;
        if definite_capacity_exceedance {
            definite_budget_limit_exceedance_count += 1;
            add_finding(
                PeakUsageFindingCode::DefiniteBudgetLimitExceeded,
                PeakUsageFindingSeverity::Blocker,
                forecast_peak_upper,
            );
        } else if possible_capacity_exceedance {
            possible_budget_limit_exceedance_count += 1;
            add_finding(
                PeakUsageFindingCode::PossibleBudgetLimitExceeded,
                PeakUsageFindingSeverity::Degradation,
                forecast_peak_upper,
            );
        }

        if let Some(forecast_upper) = forecast_peak_upper {
            forecasted_observation_count += 1;
            if observation.observed_peak.lower > forecast_upper {
                forecast_drift_count += 1;
                add_finding(
                    PeakUsageFindingCode::ForecastDefinitelyUnderestimated,
                    PeakUsageFindingSeverity::Degradation,
                    Some(forecast_upper),
                );
            } else if observation.observed_peak.upper > forecast_upper {
                forecast_drift_count += 1;
                add_finding(
                    PeakUsageFindingCode::ForecastMayBeUnderestimated,
                    PeakUsageFindingSeverity::Degradation,
                    Some(forecast_upper),
                );
            }
        } else {
            unmodeled_observation_count += 1;
            add_finding(
                PeakUsageFindingCode::NoDeclaredForecast,
                PeakUsageFindingSeverity::Degradation,
                None,
            );
        }

        observation_findings.sort_by(|left, right| {
            left.code
                .cmp(&right.code)
                .then(left.severity.cmp(&right.severity))
        });
        let status = if definite_capacity_exceedance {
            PeakUsageObservationStatus::DefiniteBudgetLimitExceeded
        } else if possible_capacity_exceedance {
            PeakUsageObservationStatus::PossibleBudgetLimitExceeded
        } else if forecast_peak_upper.is_none() && observation.observed_peak.upper > 0 {
            PeakUsageObservationStatus::UnmodeledUsage
        } else if forecast_peak_upper.is_none() || observation_findings.iter().any(|finding| {
            finding.code == PeakUsageFindingCode::NoDeclaredForecast
        }) {
            PeakUsageObservationStatus::EvidenceInsufficient
        } else if observation_findings.iter().any(|finding| {
            finding.code == PeakUsageFindingCode::ForecastDefinitelyUnderestimated
        }) {
            PeakUsageObservationStatus::ForecastDefinitelyUnderestimated
        } else if observation_findings.iter().any(|finding| {
            finding.code == PeakUsageFindingCode::ForecastMayBeUnderestimated
        }) {
            PeakUsageObservationStatus::ForecastMayBeUnderestimated
        } else if !observation_findings.is_empty() {
            PeakUsageObservationStatus::EvidenceInsufficient
        } else {
            PeakUsageObservationStatus::WithinDeclaredForecast
        };

        findings.extend(observation_findings.iter().cloned());
        observations.push(PeakUsageObservationEvaluation {
            observation_id: observation.observation_id.clone(),
            pool_id: observation.pool_id.clone(),
            resource_kind: observation.resource_kind,
            provider_component_id: observation.provider_component_id.clone(),
            provider_hardware_revision: observation.provider_hardware_revision.clone(),
            window: observation.window,
            observed_peak: observation.observed_peak,
            sample_count: observation.sample_count,
            sampling_interval_ms: observation.sampling_interval_ms,
            measurement_method: observation.measurement_method,
            evidence_level: observation.evidence_level,
            evidence_reference: observation.evidence_reference.clone(),
            forecast_window_count,
            forecast_peak_upper,
            usable_capacity: pool_evaluation.usable_capacity,
            findings: observation_findings,
            status,
        });
    }

    findings.sort_by(|left, right| {
        left.pool_id
            .cmp(&right.pool_id)
            .then(left.observation_id.cmp(&right.observation_id))
            .then(left.code.cmp(&right.code))
            .then(left.severity.cmp(&right.severity))
    });

    let blocker_count = findings
        .iter()
        .filter(|finding| finding.severity == PeakUsageFindingSeverity::Blocker)
        .count();
    let has_degradation = !findings.is_empty();
    let status = if reservation_report.status == PeakResourceReservationStatus::Blocked
        || blocker_count > 0
    {
        PeakResourceUsageMeasurementStatus::Blocked
    } else if reservation_report.status == PeakResourceReservationStatus::Degraded
        || has_degradation
    {
        PeakResourceUsageMeasurementStatus::Degraded
    } else {
        PeakResourceUsageMeasurementStatus::MeetsDeclaredMeasurementPolicy
    };

    let report = PeakResourceUsageMeasurementReport {
        schema_version: 1,
        mission_id: measurement_plan.mission_id.clone(),
        configuration_revision: measurement_plan.configuration_revision.clone(),
        configuration_digest_sha256: measurement_plan.expected_configuration_digest_sha256.clone(),
        task_plan_digest_sha256: measurement_plan.expected_task_plan_digest_sha256.clone(),
        reservation_plan_digest_sha256: measurement_plan
            .expected_reservation_plan_digest_sha256
            .clone(),
        reservation_report_digest_sha256: measurement_plan
            .expected_reservation_report_digest_sha256
            .clone(),
        measurement_plan_digest_sha256: measurement_plan_digest_sha256.clone(),
        baseline_reservation_status: reservation_report.status,
        observation_count: observations.len(),
        forecasted_observation_count,
        unmodeled_observation_count,
        forecast_drift_count,
        possible_budget_limit_exceedance_count,
        definite_budget_limit_exceedance_count,
        observations,
        findings,
        status,
        represents_physical_qualification: false,
    };

    let report_digest_sha256 = digest_report(&report);
    let mut receipt = PeakResourceUsageMeasurementReceipt {
        configuration_digest_sha256: measurement_plan.expected_configuration_digest_sha256.clone(),
        task_plan_digest_sha256: measurement_plan.expected_task_plan_digest_sha256.clone(),
        task_receipt_digest_sha256: digest_task_receipt(task_receipt),
        reservation_plan_digest_sha256: measurement_plan
            .expected_reservation_plan_digest_sha256
            .clone(),
        reservation_receipt_digest_sha256: reservation_receipt.receipt_digest_sha256.clone(),
        measurement_plan_digest_sha256,
        report_digest_sha256,
        receipt_digest_sha256: String::new(),
    };
    receipt.receipt_digest_sha256 = digest_receipt(&receipt);
    Ok((report, receipt))
}

pub fn verify_peak_resource_usage_receipt(
    configuration: &MissionConfiguration,
    tasks: &[MissionTaskRequirement],
    task_receipt: &MissionTaskReceipt,
    reservation_plan: &PeakResourceReservationPlan,
    reservation_report: &PeakResourceReservationReport,
    reservation_receipt: &PeakResourceReservationReceipt,
    measurement_plan: &PeakResourceUsageMeasurementPlan,
    expected_report: &PeakResourceUsageMeasurementReport,
    expected_receipt: &PeakResourceUsageMeasurementReceipt,
) -> bool {
    match evaluate_peak_resource_usage(
        configuration,
        tasks,
        task_receipt,
        reservation_plan,
        reservation_report,
        reservation_receipt,
        measurement_plan,
    ) {
        Ok((report, receipt)) => report == *expected_report && receipt == *expected_receipt,
        Err(_) => false,
    }
}

fn validate_measurement_plan(
    plan: &PeakResourceUsageMeasurementPlan,
) -> Result<(), PeakResourceUsageMeasurementError> {
    if plan.schema_version != 1 {
        return Err(PeakResourceUsageMeasurementError::UnsupportedSchemaVersion);
    }
    if !clean_identifier(&plan.mission_id) || !clean_identifier(&plan.configuration_revision) {
        return Err(PeakResourceUsageMeasurementError::InvalidPlanIdentity);
    }
    if !valid_sha256(&plan.expected_configuration_digest_sha256)
        || !valid_sha256(&plan.expected_task_plan_digest_sha256)
        || !valid_sha256(&plan.expected_reservation_plan_digest_sha256)
        || !valid_sha256(&plan.expected_reservation_report_digest_sha256)
        || !valid_sha256(&plan.expected_reservation_receipt_digest_sha256)
    {
        return Err(PeakResourceUsageMeasurementError::InvalidPinnedDigest);
    }
    if plan.minimum_sample_count == 0 || plan.maximum_sampling_interval_ms == 0 {
        return Err(PeakResourceUsageMeasurementError::InvalidSamplingPolicy);
    }
    if plan.observations.is_empty() {
        return Err(PeakResourceUsageMeasurementError::EmptyObservationSet);
    }

    let mut observation_ids = BTreeSet::new();
    for observation in &plan.observations {
        if !clean_identifier(&observation.observation_id)
            || !clean_identifier(&observation.pool_id)
            || !clean_identifier(&observation.provider_component_id)
            || !clean_identifier(&observation.provider_hardware_revision)
            || !clean_identifier(&observation.evidence_reference)
            || observation.sample_count == 0
            || observation.sampling_interval_ms == 0
        {
            return Err(PeakResourceUsageMeasurementError::InvalidObservation);
        }
        if !observation_ids.insert(observation.observation_id.as_str()) {
            return Err(PeakResourceUsageMeasurementError::DuplicateObservationId);
        }
        if !observation.window.is_valid() {
            return Err(PeakResourceUsageMeasurementError::InvalidTimeWindow);
        }
        if !observation.observed_peak.is_valid() {
            return Err(PeakResourceUsageMeasurementError::InvalidQuantityInterval);
        }
    }
    Ok(())
}

fn validate_observation_references(
    plan: &PeakResourceUsageMeasurementPlan,
    reservation_plan: &PeakResourceReservationPlan,
) -> Result<(), PeakResourceUsageMeasurementError> {
    let pools: BTreeMap<&str, _> = reservation_plan
        .pools
        .iter()
        .map(|pool| (pool.pool_id.as_str(), pool))
        .collect();
    for observation in &plan.observations {
        let pool = pools
            .get(observation.pool_id.as_str())
            .ok_or(PeakResourceUsageMeasurementError::MissingReferencedPool)?;
        if pool.resource_kind != observation.resource_kind {
            return Err(PeakResourceUsageMeasurementError::ResourceKindMismatch);
        }
    }
    Ok(())
}

fn windows_overlap(left: MissionTimeWindow, right: MissionTimeWindow) -> bool {
    left.start_ms < right.end_ms && right.start_ms < left.end_ms
}

fn digest_measurement_plan(plan: &PeakResourceUsageMeasurementPlan) -> String {
    let mut writer = HashWriter::new("symthaea.sx035.measurement-plan.v1");
    writer.u64(plan.schema_version as u64);
    writer.string(&plan.mission_id);
    writer.string(&plan.configuration_revision);
    writer.string(&plan.expected_configuration_digest_sha256);
    writer.string(&plan.expected_task_plan_digest_sha256);
    writer.string(&plan.expected_reservation_plan_digest_sha256);
    writer.string(&plan.expected_reservation_report_digest_sha256);
    writer.string(&plan.expected_reservation_receipt_digest_sha256);
    writer.string(evidence_tag(plan.minimum_evidence_level));
    writer.u64(plan.minimum_sample_count);
    writer.u64(plan.maximum_sampling_interval_ms);
    let mut observations: Vec<_> = plan.observations.iter().collect();
    observations.sort_by(|left, right| left.observation_id.cmp(&right.observation_id));
    writer.usize(observations.len());
    for observation in observations {
        writer.string(&observation.observation_id);
        writer.string(&observation.pool_id);
        writer.string(resource_kind_tag(observation.resource_kind));
        writer.string(&observation.provider_component_id);
        writer.string(&observation.provider_hardware_revision);
        writer.u64(observation.window.start_ms);
        writer.u64(observation.window.end_ms);
        write_interval(&mut writer, observation.observed_peak);
        writer.u64(observation.sample_count);
        writer.u64(observation.sampling_interval_ms);
        writer.string(method_tag(observation.measurement_method));
        writer.string(evidence_tag(observation.evidence_level));
        writer.string(&observation.evidence_reference);
    }
    writer.finish()
}

fn digest_report(report: &PeakResourceUsageMeasurementReport) -> String {
    let mut writer = HashWriter::new("symthaea.sx035.measurement-report.v1");
    writer.u64(report.schema_version as u64);
    writer.string(&report.mission_id);
    writer.string(&report.configuration_revision);
    writer.string(&report.configuration_digest_sha256);
    writer.string(&report.task_plan_digest_sha256);
    writer.string(&report.reservation_plan_digest_sha256);
    writer.string(&report.reservation_report_digest_sha256);
    writer.string(&report.measurement_plan_digest_sha256);
    writer.string(reservation_status_tag(report.baseline_reservation_status));
    writer.string(measurement_status_tag(report.status));
    writer.boolean(report.represents_physical_qualification);
    writer.usize(report.observation_count);
    writer.usize(report.forecasted_observation_count);
    writer.usize(report.unmodeled_observation_count);
    writer.usize(report.forecast_drift_count);
    writer.usize(report.possible_budget_limit_exceedance_count);
    writer.usize(report.definite_budget_limit_exceedance_count);

    writer.usize(report.observations.len());
    for observation in &report.observations {
        writer.string(&observation.observation_id);
        writer.string(&observation.pool_id);
        writer.string(resource_kind_tag(observation.resource_kind));
        writer.string(&observation.provider_component_id);
        writer.string(&observation.provider_hardware_revision);
        writer.u64(observation.window.start_ms);
        writer.u64(observation.window.end_ms);
        write_interval(&mut writer, observation.observed_peak);
        writer.u64(observation.sample_count);
        writer.u64(observation.sampling_interval_ms);
        writer.string(method_tag(observation.measurement_method));
        writer.string(evidence_tag(observation.evidence_level));
        writer.string(&observation.evidence_reference);
        writer.usize(observation.forecast_window_count);
        writer.option_u64(observation.forecast_peak_upper);
        writer.u64(observation.usable_capacity);
        writer.string(observation_status_tag(observation.status));
        writer.usize(observation.findings.len());
        for finding in &observation.findings {
            write_finding(&mut writer, finding);
        }
    }
    writer.usize(report.findings.len());
    for finding in &report.findings {
        write_finding(&mut writer, finding);
    }
    writer.finish()
}

fn write_finding(writer: &mut HashWriter, finding: &PeakUsageFinding) {
    writer.string(&finding.observation_id);
    writer.string(&finding.pool_id);
    writer.string(finding_code_tag(finding.code));
    writer.string(finding_severity_tag(finding.severity));
    writer.option_u64(finding.forecast_peak_upper);
    writer.u64(finding.usable_capacity);
    write_interval(writer, finding.observed_peak);
}

fn digest_receipt(receipt: &PeakResourceUsageMeasurementReceipt) -> String {
    let mut writer = HashWriter::new("symthaea.sx035.measurement-receipt.v1");
    writer.string(&receipt.configuration_digest_sha256);
    writer.string(&receipt.task_plan_digest_sha256);
    writer.string(&receipt.task_receipt_digest_sha256);
    writer.string(&receipt.reservation_plan_digest_sha256);
    writer.string(&receipt.reservation_receipt_digest_sha256);
    writer.string(&receipt.measurement_plan_digest_sha256);
    writer.string(&receipt.report_digest_sha256);
    writer.finish()
}

fn digest_task_receipt(receipt: &MissionTaskReceipt) -> String {
    let mut writer = HashWriter::new("symthaea.sx035.bound-task-receipt.v1");
    writer.string(&receipt.configuration_digest_sha256);
    writer.string(&receipt.task_plan_digest_sha256);
    writer.string(&receipt.mission_configuration_report_digest_sha256);
    writer.string(&receipt.task_report_digest_sha256);
    writer.string(&receipt.receipt_digest_sha256);
    writer.finish()
}

fn write_interval(writer: &mut HashWriter, interval: PeakQuantityInterval) {
    writer.u64(interval.lower);
    writer.u64(interval.nominal);
    writer.u64(interval.upper);
}

fn clean_identifier(value: &str) -> bool {
    !value.trim().is_empty() && value.trim() == value
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value[7..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

struct HashWriter(Sha256);

impl HashWriter {
    fn new(domain: &str) -> Self {
        let mut writer = Self(Sha256::new());
        writer.string(domain);
        writer
    }

    fn raw(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    fn u64(&mut self, value: u64) {
        self.raw(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }

    fn boolean(&mut self, value: bool) {
        self.raw(&[u8::from(value)]);
    }

    fn string(&mut self, value: &str) {
        self.usize(value.len());
        self.raw(value.as_bytes());
    }

    fn option_u64(&mut self, value: Option<u64>) {
        match value {
            Some(value) => {
                self.boolean(true);
                self.u64(value);
            }
            None => self.boolean(false),
        }
    }

    fn finish(self) -> String {
        let digest = self.0.finalize();
        format!(
            "sha256:{}",
            digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>()
        )
    }
}

fn resource_kind_tag(value: PeakResourceKind) -> &'static str {
    match value {
        PeakResourceKind::ElectricalPowerMilliwatts => "electrical_power_milliwatts",
        PeakResourceKind::CoolantFlowMicrolitersPerSecond => "coolant_flow_microliters_per_second",
        PeakResourceKind::DataRateBytesPerSecond => "data_rate_bytes_per_second",
        PeakResourceKind::ComputeBudgetUnitsPerSecond => "compute_budget_units_per_second",
        PeakResourceKind::SharedEquipmentSlots => "shared_equipment_slots",
    }
}

fn evidence_tag(value: SuitEvidenceLevel) -> &'static str {
    match value {
        SuitEvidenceLevel::Declared => "declared",
        SuitEvidenceLevel::Simulated => "simulated",
        SuitEvidenceLevel::BenchTested => "bench_tested",
        SuitEvidenceLevel::HumanInLoop => "human_in_loop",
        SuitEvidenceLevel::Qualified => "qualified",
    }
}

fn method_tag(value: ResourceMeasurementMethod) -> &'static str {
    match value {
        ResourceMeasurementMethod::SimulatedTrace => "simulated_trace",
        ResourceMeasurementMethod::BenchInstrumented => "bench_instrumented",
        ResourceMeasurementMethod::HumanInLoopInstrumented => "human_in_loop_instrumented",
        ResourceMeasurementMethod::OperationalTelemetry => "operational_telemetry",
    }
}

fn reservation_status_tag(value: PeakResourceReservationStatus) -> &'static str {
    match value {
        PeakResourceReservationStatus::Blocked => "blocked",
        PeakResourceReservationStatus::Degraded => "degraded",
        PeakResourceReservationStatus::MeetsDeclaredPeakResourcePolicy => {
            "meets_declared_peak_resource_policy"
        }
    }
}

fn measurement_status_tag(value: PeakResourceUsageMeasurementStatus) -> &'static str {
    match value {
        PeakResourceUsageMeasurementStatus::Blocked => "blocked",
        PeakResourceUsageMeasurementStatus::Degraded => "degraded",
        PeakResourceUsageMeasurementStatus::MeetsDeclaredMeasurementPolicy => {
            "meets_declared_measurement_policy"
        }
    }
}

fn observation_status_tag(value: PeakUsageObservationStatus) -> &'static str {
    match value {
        PeakUsageObservationStatus::WithinDeclaredForecast => "within_declared_forecast",
        PeakUsageObservationStatus::ForecastMayBeUnderestimated => "forecast_may_be_underestimated",
        PeakUsageObservationStatus::ForecastDefinitelyUnderestimated => {
            "forecast_definitely_underestimated"
        }
        PeakUsageObservationStatus::PossibleBudgetLimitExceeded => "possible_budget_limit_exceeded",
        PeakUsageObservationStatus::DefiniteBudgetLimitExceeded => "definite_budget_limit_exceeded",
        PeakUsageObservationStatus::UnmodeledUsage => "unmodeled_usage",
        PeakUsageObservationStatus::EvidenceInsufficient => "evidence_insufficient",
    }
}

fn finding_code_tag(value: PeakUsageFindingCode) -> &'static str {
    match value {
        PeakUsageFindingCode::ProviderComponentMismatch => "provider_component_mismatch",
        PeakUsageFindingCode::ProviderHardwareRevisionMismatch => {
            "provider_hardware_revision_mismatch"
        },
        PeakUsageFindingCode::MeasurementEvidenceBelowPolicy => "measurement_evidence_below_policy",
        PeakUsageFindingCode::SampleCountBelowPolicy => "sample_count_below_policy",
        PeakUsageFindingCode::SamplingIntervalAbovePolicy => "sampling_interval_above_policy",
        PeakUsageFindingCode::NoDeclaredForecast => "no_declared_forecast",
        PeakUsageFindingCode::ForecastMayBeUnderestimated => "forecast_may_be_underestimated",
        PeakUsageFindingCode::ForecastDefinitelyUnderestimated => {
            "forecast_definitely_underestimated"
        },
        PeakUsageFindingCode::PossibleBudgetLimitExceeded => "possible_budget_limit_exceeded",
        PeakUsageFindingCode::DefiniteBudgetLimitExceeded => "definite_budget_limit_exceeded",
    }
}

fn finding_severity_tag(value: PeakUsageFindingSeverity) -> &'static str {
    match value {
        PeakUsageFindingSeverity::Degradation => "degradation",
        PeakUsageFindingSeverity::Blocker => "blocker",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suit_interop::{
        SuitInterfaceEndpoint, SuitInterfaceKind, SuitInterfaceManifest, SuitInterfaceRole,
        SuitOperatingEnvironment,
    };
    use crate::suit_mission_configuration::{
        MissionComponent, MissionComponentKind, MissionConnection,
    };
    use crate::suit_peak_reservations::{PeakResourcePool, PeakResourceReservation};
    use crate::suit_profile::{
        SuitCapability, SuitCapabilityClaim, SuitMissionProfile, SuitVariantDeclaration,
    };

    fn fixture() -> (
        MissionConfiguration,
        Vec<MissionTaskRequirement>,
        MissionTaskReceipt,
        PeakResourceReservationPlan,
        PeakResourceReservationReport,
        PeakResourceReservationReceipt,
        PeakResourceUsageMeasurementPlan,
    ) {
        let variant_id = "measurement-suit";
        let suit_revision = "suit-m1";
        let vehicle_revision = "vehicle-m1";
        let environments =
            BTreeSet::from([SuitOperatingEnvironment::LunarShadowOrPolarRegion]);
        let manifest = |component_id: &str, revision: &str, role: SuitInterfaceRole| {
            SuitInterfaceManifest {
                component_id: component_id.into(),
                hardware_revision: revision.into(),
                endpoints: vec![SuitInterfaceEndpoint {
                    interface_id: "power.v1".into(),
                    endpoint_id: format!("{component_id}.power"),
                    kind: SuitInterfaceKind::ElectricalPower,
                    role,
                    contract_revision: "1.0.0".into(),
                    schema_digest_sha256: format!("sha256:{}", "9".repeat(64)),
                    connector_profile_id: "synthetic-power-v1".into(),
                    applicable_environments: environments.clone(),
                    evidence_level: SuitEvidenceLevel::BenchTested,
                    evidence_reference: format!("synthetic-interface:{component_id}"),
                    required: true,
                }],
            }
        };
        let claims = SuitMissionProfile::LunarSurfaceEva
            .required_capabilities()
            .into_iter()
            .map(|capability| SuitCapabilityClaim {
                capability,
                applies_to_variant_id: variant_id.into(),
                applies_to_hardware_revision: suit_revision.into(),
                implementation_id: format!("synthetic-impl:{capability:?}"),
                evidence_level: SuitEvidenceLevel::BenchTested,
                evidence_reference: format!("synthetic-evidence:{capability:?}"),
            })
            .collect();
        let configuration = MissionConfiguration {
            schema_version: 1,
            mission_id: "synthetic-measurement-mission".into(),
            configuration_revision: "config-meas-r1".into(),
            suit_component_id: "suit".into(),
            suit_profile: SuitMissionProfile::LunarSurfaceEva,
            suit_variant: SuitVariantDeclaration {
                variant_id: variant_id.into(),
                hardware_revision: suit_revision.into(),
                capability_claims: claims,
            },
            minimum_capability_evidence: SuitEvidenceLevel::BenchTested,
            required_component_kinds: BTreeSet::from([
                MissionComponentKind::Suit,
                MissionComponentKind::LanderOrVehicle,
            ]),
            required_connection_ids: BTreeSet::from(["suit-vehicle-power".into()]),
            components: vec![
                MissionComponent {
                    component_id: "suit".into(),
                    kind: MissionComponentKind::Suit,
                    hardware_revision: suit_revision.into(),
                    interfaces: manifest("suit", suit_revision, SuitInterfaceRole::Consumer),
                },
                MissionComponent {
                    component_id: "vehicle".into(),
                    kind: MissionComponentKind::LanderOrVehicle,
                    hardware_revision: vehicle_revision.into(),
                    interfaces: manifest("vehicle", vehicle_revision, SuitInterfaceRole::Provider),
                },
            ],
            connections: vec![MissionConnection {
                connection_id: "suit-vehicle-power".into(),
                left_component_id: "suit".into(),
                right_component_id: "vehicle".into(),
                policy: crate::suit_interop::SuitInteropPolicy {
                    minimum_evidence_level: SuitEvidenceLevel::Simulated,
                    required_environments: environments.clone(),
                },
            }],
        };
        let tasks = vec![MissionTaskRequirement {
            task_id: "required-life-support".into(),
            required: true,
            required_capabilities: BTreeSet::from([SuitCapability::ThermalManagement]),
            required_component_ids: BTreeSet::from(["vehicle".into()]),
            required_connection_ids: BTreeSet::from(["suit-vehicle-power".into()]),
            required_environments: environments.clone(),
        }];
        let (task_report, task_receipt) = crate::suit_mission_tasks::evaluate_mission_tasks(
            &configuration, &tasks
        ).unwrap();
        let reservation_plan = PeakResourceReservationPlan {
            schema_version: 1,
            mission_id: configuration.mission_id.clone(),
            configuration_revision: configuration.configuration_revision.clone(),
            expected_configuration_digest_sha256: task_report.configuration_digest_sha256,
            expected_task_plan_digest_sha256: task_report.task_plan_digest_sha256,
            minimum_evidence_level: SuitEvidenceLevel::BenchTested,
            reserve_basis_points: 1_000,
            pools: vec![PeakResourcePool {
                pool_id: "vehicle-power".into(),
                resource_kind: PeakResourceKind::ElectricalPowerMilliwatts,
                provider_component_id: "vehicle".into(),
                provider_hardware_revision: vehicle_revision.into(),
                required_connection_ids: BTreeSet::from(["suit-vehicle-power".into()]),
                required_interface_kinds: BTreeSet::from([SuitInterfaceKind::ElectricalPower]),
                capacity: PeakQuantityInterval { lower: 1_000, nominal: 1_100, upper: 1_200 },
                evidence_level: SuitEvidenceLevel::BenchTested,
                evidence_reference: "synthetic-capacity:power".into(),
                applicable_environments: environments.clone(),
            }],
            reservations: vec![PeakResourceReservation {
                reservation_id: "life-support-load".into(),
                task_id: "required-life-support".into(),
                pool_id: "vehicle-power".into(),
                resource_kind: PeakResourceKind::ElectricalPowerMilliwatts,
                window: MissionTimeWindow { start_ms: 0, end_ms: 100 },
                demand: PeakQuantityInterval { lower: 300, nominal: 400, upper: 500 },
                evidence_level: SuitEvidenceLevel::BenchTested,
                evidence_reference: "synthetic-forecast:life-support".into(),
            }],
        };
        let (reservation_report, reservation_receipt) = evaluate_peak_resource_reservations(
            &configuration, &tasks, &task_receipt, &reservation_plan
        ).unwrap();
        let measurement_plan = measurement_plan_for(
            &configuration,
            &task_report.task_plan_digest_sha256,
            &reservation_report,
            &reservation_receipt,
            vec![observation("measurement-1", PeakQuantityInterval {
                lower: 400, nominal: 420, upper: 450
            })],
        );
        (
            configuration, tasks, task_receipt, reservation_plan,
            reservation_report, reservation_receipt, measurement_plan
        )
    }

    fn observation(
        id: &str,
        observed_peak: PeakQuantityInterval,
    ) -> PeakResourceUsageObservation {
        PeakResourceUsageObservation {
            observation_id: id.into(),
            pool_id: "vehicle-power".into(),
            resource_kind: PeakResourceKind::ElectricalPowerMilliwatts,
            provider_component_id: "vehicle".into(),
            provider_hardware_revision: "vehicle-m1".into(),
            window: MissionTimeWindow { start_ms: 25, end_ms: 75 },
            observed_peak,
            sample_count: 100,
            sampling_interval_ms: 10,
            measurement_method: ResourceMeasurementMethod::BenchInstrumented,
            evidence_level: SuitEvidenceLevel::BenchTested,
            evidence_reference: format!("synthetic-observation:{id}"),
        }
    }

    fn measurement_plan_for(
        configuration: &MissionConfiguration,
        task_digest: &str,
        report: &PeakResourceReservationReport,
        receipt: &PeakResourceReservationReceipt,
        observations: Vec<PeakResourceUsageObservation>,
    ) -> PeakResourceUsageMeasurementPlan {
        PeakResourceUsageMeasurementPlan {
            schema_version: 1,
            mission_id: configuration.mission_id.clone(),
            configuration_revision: configuration.configuration_revision.clone(),
            expected_configuration_digest_sha256: report.configuration_digest_sha256.clone(),
            expected_task_plan_digest_sha256: task_digest.into(),
            expected_reservation_plan_digest_sha256: report.reservation_plan_digest_sha256.clone(),
            expected_reservation_report_digest_sha256: receipt.report_digest_sha256.clone(),
            expected_reservation_receipt_digest_sha256: receipt.receipt_digest_sha256.clone(),
            minimum_evidence_level: SuitEvidenceLevel::BenchTested,
            minimum_sample_count: 10,
            maximum_sampling_interval_ms: 50,
            observations,
        }
    }

    #[test]
    fn observed_peak_inside_forecast_and_capacity_meets_policy() {
        let (
            config, tasks, task_receipt, reserve_plan, reserve_report, reserve_receipt, plan,
        ) = fixture();
        let (report, receipt) = evaluate_peak_resource_usage(
            &config, &tasks, &task_receipt, &reserve_plan, &reserve_report, &reserve_receipt, &plan
        ).unwrap();
        assert_eq!(
            report.status,
            PeakResourceUsageMeasurementStatus::MeetsDeclaredMeasurementPolicy
        );
        assert_eq!(report.observations[0].forecast_peak_upper, Some(500));
        assert_eq!(
            report.observations[0].status,
            PeakUsageObservationStatus::WithinDeclaredForecast
        );
        assert!(!report.represents_physical_qualification);
        assert!(verify_peak_resource_usage_receipt(
            &config, &tasks, &task_receipt, &reserve_plan, &reserve_report,
            &reserve_receipt, &plan, &report, &receipt
        ));
    }

    #[test]
    fn definite_forecast_underprediction_degrades_without_claiming_capacity_failure() {
        let (
            config, tasks, task_receipt, reserve_plan, reserve_report, reserve_receipt, mut plan,
        ) = fixture();
        plan.observations[0].observed_peak = PeakQuantityInterval {
            lower: 600,
            nominal: 650,
            upper: 700,
        };
        let (report, _) = evaluate_peak_resource_usage(
            &config, &tasks, &task_receipt, &reserve_plan, &reserve_report, &reserve_receipt, &plan
        ).unwrap();
        assert_eq!(report.status, PeakResourceUsageMeasurementStatus::Degraded);
        assert_eq!(report.definite_budget_limit_exceedance_count, 0);
        assert!(report.findings.iter().any(|finding| {
            finding.code == PeakUsageFindingCode::ForecastDefinitelyUnderestimated
        }));
    }

    #[test]
    fn definite_budget_limit_exceedance_blocks() {
        let (
            config, tasks, task_receipt, reserve_plan, reserve_report, reserve_receipt, mut plan,
        ) = fixture();
        plan.observations[0].observed_peak = PeakQuantityInterval {
            lower: 950,
            nominal: 1_000,
            upper: 1_050,
        };
        let (report, _) = evaluate_peak_resource_usage(
            &config, &tasks, &task_receipt, &reserve_plan, &reserve_report, &reserve_receipt, &plan
        ).unwrap();
        assert_eq!(report.status, PeakResourceUsageMeasurementStatus::Blocked);
        assert_eq!(report.definite_budget_limit_exceedance_count, 1);
        assert!(report.findings.iter().any(|finding| {
            finding.code == PeakUsageFindingCode::DefiniteBudgetLimitExceeded
                && finding.severity == PeakUsageFindingSeverity::Blocker
        }));
    }

    #[test]
    fn possible_budget_limit_exceedance_is_review_not_definite_failure() {
        let (
            config, tasks, task_receipt, reserve_plan, reserve_report, reserve_receipt, mut plan,
        ) = fixture();
        plan.observations[0].observed_peak = PeakQuantityInterval {
            lower: 850,
            nominal: 900,
            upper: 950,
        };
        let (report, _) = evaluate_peak_resource_usage(
            &config, &tasks, &task_receipt, &reserve_plan, &reserve_report, &reserve_receipt, &plan
        ).unwrap();
        assert_eq!(report.status, PeakResourceUsageMeasurementStatus::Degraded);
        assert_eq!(report.possible_budget_limit_exceedance_count, 1);
        assert_eq!(report.definite_budget_limit_exceedance_count, 0);
    }

    #[test]
    fn low_sample_count_and_slow_sampling_are_explicit_degradations() {
        let (
            config, tasks, task_receipt, reserve_plan, reserve_report, reserve_receipt, mut plan,
        ) = fixture();
        plan.observations[0].sample_count = 2;
        plan.observations[0].sampling_interval_ms = 200;
        let (report, _) = evaluate_peak_resource_usage(
            &config, &tasks, &task_receipt, &reserve_plan, &reserve_report, &reserve_receipt, &plan
        ).unwrap();
        assert_eq!(report.status, PeakResourceUsageMeasurementStatus::Degraded);
        assert!(report.findings.iter().any(|finding| {
            finding.code == PeakUsageFindingCode::SampleCountBelowPolicy
        }));
        assert!(report.findings.iter().any(|finding| {
            finding.code == PeakUsageFindingCode::SamplingIntervalAbovePolicy
        }));
    }

    #[test]
    fn provider_revision_mismatch_does_not_masquerade_as_current_evidence() {
        let (
            config, tasks, task_receipt, reserve_plan, reserve_report, reserve_receipt, mut plan,
        ) = fixture();
        plan.observations[0].provider_hardware_revision = "vehicle-m0".into();
        let (report, _) = evaluate_peak_resource_usage(
            &config, &tasks, &task_receipt, &reserve_plan, &reserve_report, &reserve_receipt, &plan
        ).unwrap();
        assert_eq!(report.status, PeakResourceUsageMeasurementStatus::Degraded);
        assert!(report.findings.iter().any(|finding| {
            finding.code == PeakUsageFindingCode::ProviderHardwareRevisionMismatch
        }));
    }

    #[test]
    fn no_declared_forecast_for_positive_observation_is_reported() {
        let (
            config, tasks, task_receipt, reserve_plan, reserve_report, reserve_receipt, mut plan,
        ) = fixture();
        plan.observations[0].window = MissionTimeWindow { start_ms: 200, end_ms: 300 };
        let (report, _) = evaluate_peak_resource_usage(
            &config, &tasks, &task_receipt, &reserve_plan, &reserve_report, &reserve_receipt, &plan
        ).unwrap();
        assert_eq!(report.status, PeakResourceUsageMeasurementStatus::Degraded);
        assert_eq!(report.unmodeled_observation_count, 1);
        assert!(report.findings.iter().any(|finding| {
            finding.code == PeakUsageFindingCode::NoDeclaredForecast
        }));
    }

    #[test]
    fn stale_baseline_receipt_is_rejected() {
        let (
            config, tasks, task_receipt, reserve_plan, reserve_report, mut reserve_receipt, plan,
        ) = fixture();
        reserve_receipt.receipt_digest_sha256.push_str("tampered");
        assert_eq!(
            evaluate_peak_resource_usage(
                &config,
                &tasks,
                &task_receipt,
                &reserve_plan,
                &reserve_report,
                &reserve_receipt,
                &plan
            ).unwrap_err(),
            PeakResourceUsageMeasurementError::ReservationReceiptDigestMismatch
        );
    }

    #[test]
    fn wrong_resource_kind_and_invalid_observation_window_are_rejected() {
        let (
            config, tasks, task_receipt, reserve_plan, reserve_report, reserve_receipt, mut plan,
        ) = fixture();
        plan.observations[0].resource_kind = PeakResourceKind::DataRateBytesPerSecond;
        assert_eq!(
            evaluate_peak_resource_usage(
                &config,
                &tasks,
                &task_receipt,
                &reserve_plan,
                &reserve_report,
                &reserve_receipt,
                &plan
            ).unwrap_err(),
            PeakResourceUsageMeasurementError::ResourceKindMismatch
        );
        plan.observations[0].resource_kind = PeakResourceKind::ElectricalPowerMilliwatts;
        plan.observations[0].window.end_ms = plan.observations[0].window.start_ms;
        assert_eq!(
            evaluate_peak_resource_usage(
                &config,
                &tasks,
                &task_receipt,
                &reserve_plan,
                &reserve_report,
                &reserve_receipt,
                &plan
            ).unwrap_err(),
            PeakResourceUsageMeasurementError::InvalidTimeWindow
        );
    }

    #[test]
    fn seeded_2000_case_corpus_checks_forecast_and_capacity_boundaries() {
        let (
            config,
            tasks,
            task_receipt,
            reserve_plan,
            reserve_report,
            reserve_receipt,
            mut plan,
        ) = fixture();

        for seed in 0_u64..2_000 {
            let lower = seed.wrapping_mul(97) % 1_001;
            let width = seed.wrapping_mul(53).wrapping_add(7) % 201;
            let upper = lower + width;
            let nominal = lower + (width / 2);
            plan.observations[0].observed_peak = PeakQuantityInterval {
                lower,
                nominal,
                upper,
            };
            let (report, _) = evaluate_peak_resource_usage(
                &config,
                &tasks,
                &task_receipt,
                &reserve_plan,
                &reserve_report,
                &reserve_receipt,
                &plan,
            )
            .unwrap();

            let expected = if lower > 900 {
                PeakUsageObservationStatus::DefiniteBudgetLimitExceeded
            } else if upper > 900 {
                PeakUsageObservationStatus::PossibleBudgetLimitExceeded
            } else if lower > 500 {
                PeakUsageObservationStatus::ForecastDefinitelyUnderestimated
            } else if upper > 500 {
                PeakUsageObservationStatus::ForecastMayBeUnderestimated
            } else {
                PeakUsageObservationStatus::WithinDeclaredForecast
            };
            assert_eq!(
                report.observations[0].status,
                expected,
                "seed {seed}, observed [{lower}, {nominal}, {upper}]"
            );
            assert_eq!(
                report.definite_budget_limit_exceedance_count,
                usize::from(lower > 900),
                "seed {seed}"
            );
            assert_eq!(
                report.possible_budget_limit_exceedance_count,
                usize::from(lower <= 900 && upper > 900),
                "seed {seed}"
            );
            assert_eq!(report.forecasted_observation_count, 1, "seed {seed}");

            if seed % 251 == 0 {
                assert!(verify_peak_resource_usage_receipt(
                    &config,
                    &tasks,
                    &task_receipt,
                    &reserve_plan,
                    &reserve_report,
                    &reserve_receipt,
                    &plan,
                    &report,
                    &evaluate_peak_resource_usage(
                        &config,
                        &tasks,
                        &task_receipt,
                        &reserve_plan,
                        &reserve_report,
                        &reserve_receipt,
                        &plan,
                    )
                    .unwrap()
                    .1,
                ));
            }
        }
    }

    #[test]
    fn input_order_is_invariant_and_mutation_breaks_receipt() {
        let (config, tasks, task_receipt, reserve_plan, reserve_report, reserve_receipt, mut plan) =
            fixture();
        let mut second = plan.observations[0].clone();
        second.observation_id = "measurement-2".into();
        second.evidence_reference = "synthetic-observation:measurement-2".into();
        plan.observations.push(second);
        let (report_a, receipt_a) = evaluate_peak_resource_usage(
            &config, &tasks, &task_receipt, &reserve_plan, &reserve_report, &reserve_receipt, &plan
        ).unwrap();
        plan.observations.reverse();
        let (report_b, receipt_b) = evaluate_peak_resource_usage(
            &config, &tasks, &task_receipt, &reserve_plan, &reserve_report, &reserve_receipt, &plan
        ).unwrap();
        assert_eq!(report_a, report_b);
        assert_eq!(receipt_a, receipt_b);

        plan.observations[0].observed_peak.upper += 1;
        assert!(!verify_peak_resource_usage_receipt(
            &config, &tasks, &task_receipt, &reserve_plan, &reserve_report,
            &reserve_receipt, &plan, &report_a, &receipt_a
        ));
    }
}
