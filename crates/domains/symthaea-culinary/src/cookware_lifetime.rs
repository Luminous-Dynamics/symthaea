// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Longitudinal cookware condition, repair, and service-life evidence.
//!
//! This contract composes the v1 matched benchmark manifest instead of
//! re-defining its measurements or evidence classes. Each stage assessment
//! points to exactly one existing benchmark run; use, maintenance, repair,
//! failure, and retirement events preserve the history of the physical unit.
//!
//! Validation is structural. It does not authenticate references, certify
//! food-contact safety, prove an accelerated duty cycle equals calendar years,
//! or establish a product's actual service life.

use super::cookware::{
    CookwareBenchmarkManifestV1, CookwareCandidateV1, CookwareRunV1,
    CookwareValidationReport, validate_cookware_benchmark,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const COOKWARE_LIFETIME_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CookwareLifecycleStage {
    Baseline,
    InServiceCheckpoint,
    PostMaintenance,
    PostRepair,
    EndOfLife,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CookwareLifecycleCondition {
    New,
    InService,
    Worn,
    Failed,
    Retired,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CookwareLifecycleEventKind {
    StageAssessment,
    UseBatchCompleted,
    MaintenanceCompleted,
    RepairCompleted,
    FailureObserved,
    Retired,
}

/// Cumulative exposure counters for one physical specimen. A counter must
/// never decrease; only a completed-use event may increase these counters.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct CookwareDutyCycleCountersV1 {
    pub thermal_cycles: u64,
    pub cooking_load_cycles: u64,
    pub cleaning_cycles: u64,
    pub abrasion_cycles: u64,
    pub declared_misuse_cycles: u64,
}

impl CookwareDutyCycleCountersV1 {
    fn is_zero(self) -> bool {
        self.thermal_cycles == 0
            && self.cooking_load_cycles == 0
            && self.cleaning_cycles == 0
            && self.abrasion_cycles == 0
            && self.declared_misuse_cycles == 0
    }

    fn is_componentwise_at_least(self, prior: Self) -> bool {
        self.thermal_cycles >= prior.thermal_cycles
            && self.cooking_load_cycles >= prior.cooking_load_cycles
            && self.cleaning_cycles >= prior.cleaning_cycles
            && self.abrasion_cycles >= prior.abrasion_cycles
            && self.declared_misuse_cycles >= prior.declared_misuse_cycles
    }

    fn strictly_increased_from(self, prior: Self) -> bool {
        self.is_componentwise_at_least(prior) && self != prior
    }
}

/// One immutable event in a specimen's timeline.
///
/// `sequence_index` starts at zero for the baseline and is contiguous per
/// specimen. Run references are only permitted on StageAssessment events.
/// Repair cost uses minor currency units plus an explicit ISO-4217-style code;
/// this module checks shape only and does not verify economic evidence.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CookwareLifecycleEventV1 {
    pub event_id: String,
    pub cookware_unit_id: String,
    pub candidate_id: String,
    pub sequence_index: u32,
    pub kind: CookwareLifecycleEventKind,
    pub stage: CookwareLifecycleStage,
    pub condition_after: CookwareLifecycleCondition,
    pub cumulative_cycles: CookwareDutyCycleCountersV1,
    pub evidence_ref: String,
    pub method_ref: String,
    pub run_id: Option<String>,
    pub failure_mode: Option<String>,
    pub repair_parts: Vec<String>,
    pub repair_duration_seconds: Option<u64>,
    pub repair_cost_minor_units: Option<u64>,
    pub repair_currency_code: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CookwareLifetimeProtocolV1 {
    pub protocol_id: String,
    pub duty_cycle_profile_id: String,
    pub minimum_independent_units_per_candidate: u32,
    /// Minimum number of non-baseline instrumented checkpoints required per
    /// physical unit. Pilot protocols may use one; superiority claims need a
    /// study-specific sample-size and analysis plan.
    pub minimum_in_service_checkpoints_per_unit: u32,
    /// Optional calendar-life extrapolation. Must remain absent unless an
    /// independently justified cycle-to-calendar equivalence record exists.
    pub calendar_life_claim_years: Option<u32>,
    pub cycle_to_calendar_equivalence_ref: Option<String>,
}

/// A longitudinal report retains the underlying benchmark validation result.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CookwareLifetimeValidationReport {
    pub study_id: String,
    pub benchmark_report: CookwareValidationReport,
    pub violations: Vec<CookwareLifetimeViolation>,
}

impl CookwareLifetimeValidationReport {
    pub fn is_valid(&self) -> bool {
        self.benchmark_report.is_valid() && self.violations.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CookwareLifetimeViolationCode {
    UnsupportedSchemaVersion,
    MissingRequiredValue,
    InvalidProtocol,
    DuplicateEventIdentifier,
    InvalidSequence,
    UnknownCandidate,
    UnknownCookwareUnit,
    UnitIdentityConflict,
    UnknownBenchmarkRun,
    RunIdentityMismatch,
    MissingRunLink,
    UnexpectedRunLink,
    DuplicateRunLink,
    UnreferencedBenchmarkRun,
    MissingUnitTimeline,
    MissingBaseline,
    InvalidBaseline,
    MissingUseBatch,
    InsufficientInServiceCheckpoints,
    CounterRegression,
    CounterAdvancedOutsideUseBatch,
    UseBatchDidNotAdvanceCounters,
    InvalidStageEvent,
    InvalidConditionTransition,
    MissingRepairEvidence,
    MissingMaintenanceEvent,
    MissingRepairEvent,
    UnexpectedRepairEvidence,
    MissingFailureMode,
    UnexpectedFailureMode,
    RetiredUnitHasLaterEvents,
    InsufficientIndependentUnits,
    UnsupportedCalendarLifeExtrapolation,
}

/// Machine-readable lifecycle violation. The validator never silently drops a
/// failed specimen or converts repeated measurements into additional units.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CookwareLifetimeViolation {
    pub code: CookwareLifetimeViolationCode,
    pub candidate_id: Option<String>,
    pub cookware_unit_id: Option<String>,
    pub event_id: Option<String>,
    pub detail: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CookwareLifetimeManifestV1 {
    pub schema_version: u32,
    pub study_id: String,
    pub protocol: CookwareLifetimeProtocolV1,
    /// Contains all measured stages, grouped in matched benchmark blocks.
    /// Every run must be bound to exactly one StageAssessment event below.
    pub benchmark: CookwareBenchmarkManifestV1,
    pub events: Vec<CookwareLifecycleEventV1>,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn add_violation(
    violations: &mut Vec<CookwareLifetimeViolation>,
    code: CookwareLifetimeViolationCode,
    candidate_id: Option<&str>,
    unit_id: Option<&str>,
    event_id: Option<&str>,
    detail: impl Into<String>,
) {
    violations.push(CookwareLifetimeViolation {
        code,
        candidate_id: candidate_id.map(str::to_owned),
        cookware_unit_id: unit_id.map(str::to_owned),
        event_id: event_id.map(str::to_owned),
        detail: detail.into(),
    });
}

fn event_has_repair_fields(event: &CookwareLifecycleEventV1) -> bool {
    !event.repair_parts.is_empty()
        || event.repair_duration_seconds.is_some()
        || event.repair_cost_minor_units.is_some()
        || event.repair_currency_code.is_some()
}

fn valid_currency_code(code: &str) -> bool {
    code.len() == 3 && code.bytes().all(|byte| byte.is_ascii_uppercase())
}

fn validate_event_shape(
    event: &CookwareLifecycleEventV1,
    violations: &mut Vec<CookwareLifetimeViolation>,
) {
    let candidate_id = Some(event.candidate_id.as_str());
    let unit_id = Some(event.cookware_unit_id.as_str());
    let event_id = Some(event.event_id.as_str());

    if !nonempty(&event.event_id)
        || !nonempty(&event.cookware_unit_id)
        || !nonempty(&event.candidate_id)
        || !nonempty(&event.evidence_ref)
        || !nonempty(&event.method_ref)
    {
        add_violation(
            violations,
            CookwareLifetimeViolationCode::MissingRequiredValue,
            candidate_id,
            unit_id,
            event_id,
            "event, unit, candidate, evidence, and method identifiers must be non-empty",
        );
    }

    match event.kind {
        CookwareLifecycleEventKind::StageAssessment => {
            if event.run_id.as_deref().is_none_or(|id| !nonempty(id)) {
                add_violation(
                    violations,
                    CookwareLifetimeViolationCode::MissingRunLink,
                    candidate_id,
                    unit_id,
                    event_id,
                    "stage assessments must reference one benchmark run",
                );
            }
        }
        _ if event.run_id.is_some() => add_violation(
            violations,
            CookwareLifetimeViolationCode::UnexpectedRunLink,
            candidate_id,
            unit_id,
            event_id,
            "only stage assessment events may reference a benchmark run",
        ),
        _ => {}
    }

    match event.kind {
        CookwareLifecycleEventKind::UseBatchCompleted => {
            if event.stage != CookwareLifecycleStage::InServiceCheckpoint
                || !matches!(
                    event.condition_after,
                    CookwareLifecycleCondition::InService | CookwareLifecycleCondition::Worn
                )
            {
                add_violation(
                    violations,
                    CookwareLifetimeViolationCode::InvalidStageEvent,
                    candidate_id,
                    unit_id,
                    event_id,
                    "use batches must be recorded at an in-service checkpoint in an in-service or worn state",
                );
            }
        }
        CookwareLifecycleEventKind::MaintenanceCompleted => {
            if event.stage != CookwareLifecycleStage::PostMaintenance
                || !matches!(
                    event.condition_after,
                    CookwareLifecycleCondition::InService | CookwareLifecycleCondition::Worn
                )
            {
                add_violation(
                    violations,
                    CookwareLifetimeViolationCode::InvalidStageEvent,
                    candidate_id,
                    unit_id,
                    event_id,
                    "maintenance events must record a post-maintenance state",
                );
            }
        }
        CookwareLifecycleEventKind::RepairCompleted => {
            if event.stage != CookwareLifecycleStage::PostRepair
                || !matches!(
                    event.condition_after,
                    CookwareLifecycleCondition::InService
                        | CookwareLifecycleCondition::Worn
                        | CookwareLifecycleCondition::Failed
                )
            {
                add_violation(
                    violations,
                    CookwareLifetimeViolationCode::InvalidStageEvent,
                    candidate_id,
                    unit_id,
                    event_id,
                    "repair events must record the post-repair in-service, worn, or failed outcome",
                );
            }
            let repair_valid = !event.repair_parts.is_empty()
                && event.repair_parts.iter().all(|part| nonempty(part))
                && event.repair_duration_seconds.is_some_and(|seconds| seconds > 0)
                && event.repair_cost_minor_units.is_some()
                && event
                    .repair_currency_code
                    .as_deref()
                    .is_some_and(valid_currency_code);
            if !repair_valid {
                add_violation(
                    violations,
                    CookwareLifetimeViolationCode::MissingRepairEvidence,
                    candidate_id,
                    unit_id,
                    event_id,
                    "repairs require named replaced/repaired parts, positive duration, recorded cost, and a three-letter uppercase currency code",
                );
            }
        }
        CookwareLifecycleEventKind::FailureObserved => {
            if event.condition_after != CookwareLifecycleCondition::Failed
                || event.stage == CookwareLifecycleStage::Baseline
            {
                add_violation(
                    violations,
                    CookwareLifetimeViolationCode::InvalidStageEvent,
                    candidate_id,
                    unit_id,
                    event_id,
                    "failure events must record a post-baseline failed state",
                );
            }
            if event.failure_mode.as_deref().is_none_or(|mode| !nonempty(mode)) {
                add_violation(
                    violations,
                    CookwareLifetimeViolationCode::MissingFailureMode,
                    candidate_id,
                    unit_id,
                    event_id,
                    "failure events require a declared failure mode",
                );
            }
        }
        CookwareLifecycleEventKind::Retired => {
            if event.stage != CookwareLifecycleStage::EndOfLife
                || event.condition_after != CookwareLifecycleCondition::Retired
            {
                add_violation(
                    violations,
                    CookwareLifetimeViolationCode::InvalidStageEvent,
                    candidate_id,
                    unit_id,
                    event_id,
                    "retirement must be recorded as an end-of-life event",
                );
            }
        }
        CookwareLifecycleEventKind::StageAssessment => {
            if event.stage == CookwareLifecycleStage::Baseline
                && event.condition_after != CookwareLifecycleCondition::New
            {
                add_violation(
                    violations,
                    CookwareLifetimeViolationCode::InvalidBaseline,
                    candidate_id,
                    unit_id,
                    event_id,
                    "baseline assessments must identify a new specimen",
                );
            }
            if event.stage != CookwareLifecycleStage::Baseline
                && event.condition_after == CookwareLifecycleCondition::New
            {
                add_violation(
                    violations,
                    CookwareLifetimeViolationCode::InvalidStageEvent,
                    candidate_id,
                    unit_id,
                    event_id,
                    "post-baseline assessments cannot return to the new condition state",
                );
            }
            if event.stage == CookwareLifecycleStage::EndOfLife
                && !matches!(
                    event.condition_after,
                    CookwareLifecycleCondition::Failed | CookwareLifecycleCondition::Retired
                )
            {
                add_violation(
                    violations,
                    CookwareLifetimeViolationCode::InvalidStageEvent,
                    candidate_id,
                    unit_id,
                    event_id,
                    "end-of-life assessments must record failed or retired condition",
                );
            }
        }
    }

    if event.kind != CookwareLifecycleEventKind::RepairCompleted && event_has_repair_fields(event) {
        add_violation(
            violations,
            CookwareLifetimeViolationCode::UnexpectedRepairEvidence,
            candidate_id,
            unit_id,
            event_id,
            "repair-only fields may only appear on RepairCompleted events",
        );
    }
    if event.kind != CookwareLifecycleEventKind::FailureObserved && event.failure_mode.is_some() {
        add_violation(
            violations,
            CookwareLifetimeViolationCode::UnexpectedFailureMode,
            candidate_id,
            unit_id,
            event_id,
            "failure_mode may only appear on FailureObserved events",
        );
    }
}

/// Validate the lifecycle history and its one-to-one links to measured runs.
///
/// A valid result means that the declared longitudinal record is structurally
/// consistent. It does not prove measurement quality, external evidence
/// authenticity, food-contact compliance, or real-world lifespan.
pub fn validate_cookware_lifetime(
    manifest: &CookwareLifetimeManifestV1,
) -> CookwareLifetimeValidationReport {
    let benchmark_report = validate_cookware_benchmark(&manifest.benchmark);
    let mut violations = Vec::new();

    if manifest.schema_version != COOKWARE_LIFETIME_SCHEMA_VERSION {
        add_violation(
            &mut violations,
            CookwareLifetimeViolationCode::UnsupportedSchemaVersion,
            None,
            None,
            None,
            format!(
                "lifecycle schema version {} is unsupported; expected {}",
                manifest.schema_version, COOKWARE_LIFETIME_SCHEMA_VERSION
            ),
        );
    }
    if !nonempty(&manifest.study_id)
        || !nonempty(&manifest.protocol.protocol_id)
        || !nonempty(&manifest.protocol.duty_cycle_profile_id)
    {
        add_violation(
            &mut violations,
            CookwareLifetimeViolationCode::MissingRequiredValue,
            None,
            None,
            None,
            "study, lifecycle protocol, and duty-cycle profile identifiers are required",
        );
    }
    if manifest.protocol.minimum_independent_units_per_candidate == 0
        || manifest.protocol.minimum_in_service_checkpoints_per_unit == 0
    {
        add_violation(
            &mut violations,
            CookwareLifetimeViolationCode::InvalidProtocol,
            None,
            None,
            None,
            "minimum independent units and in-service checkpoints must be positive",
        );
    }
    match (
        manifest.protocol.calendar_life_claim_years,
        manifest.protocol.cycle_to_calendar_equivalence_ref.as_deref(),
    ) {
        (Some(years), Some(reference)) if years > 0 && nonempty(reference) => {}
        (Some(_), _) => add_violation(
            &mut violations,
            CookwareLifetimeViolationCode::UnsupportedCalendarLifeExtrapolation,
            None,
            None,
            None,
            "calendar service-life claims require a positive year count and a declared cycle-to-calendar equivalence reference",
        ),
        (None, Some(_)) => add_violation(
            &mut violations,
            CookwareLifetimeViolationCode::InvalidProtocol,
            None,
            None,
            None,
            "a cycle-to-calendar equivalence reference must not be supplied without a calendar-life claim",
        ),
        (None, None) => {}
    }

    let candidate_by_id: BTreeMap<&str, &CookwareCandidateV1> = manifest
        .benchmark
        .candidates
        .iter()
        .map(|candidate| (candidate.candidate_id.as_str(), candidate))
        .collect();
    let run_by_id: BTreeMap<&str, &CookwareRunV1> = manifest
        .benchmark
        .runs
        .iter()
        .map(|run| (run.run_id.as_str(), run))
        .collect();

    let mut unit_candidate: BTreeMap<String, String> = BTreeMap::new();
    let mut units_by_candidate: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for run in &manifest.benchmark.runs {
        match unit_candidate.get(&run.cookware_unit_id) {
            Some(existing) if existing != &run.candidate_id => add_violation(
                &mut violations,
                CookwareLifetimeViolationCode::UnitIdentityConflict,
                Some(&run.candidate_id),
                Some(&run.cookware_unit_id),
                None,
                "one physical cookware unit cannot belong to multiple candidate identities",
            ),
            _ => {
                unit_candidate.insert(run.cookware_unit_id.clone(), run.candidate_id.clone());
            }
        }
        units_by_candidate
            .entry(run.candidate_id.clone())
            .or_default()
            .insert(run.cookware_unit_id.clone());
    }

    let mut event_ids = BTreeSet::new();
    let mut events_by_unit: BTreeMap<String, Vec<&CookwareLifecycleEventV1>> = BTreeMap::new();
    let mut run_link_counts: BTreeMap<String, usize> = BTreeMap::new();

    for event in &manifest.events {
        if !event_ids.insert(event.event_id.as_str()) {
            add_violation(
                &mut violations,
                CookwareLifetimeViolationCode::DuplicateEventIdentifier,
                Some(&event.candidate_id),
                Some(&event.cookware_unit_id),
                Some(&event.event_id),
                "lifecycle event IDs must be unique",
            );
        }
        validate_event_shape(event, &mut violations);

        if !candidate_by_id.contains_key(event.candidate_id.as_str()) {
            add_violation(
                &mut violations,
                CookwareLifetimeViolationCode::UnknownCandidate,
                Some(&event.candidate_id),
                Some(&event.cookware_unit_id),
                Some(&event.event_id),
                "event refers to an unknown cookware candidate",
            );
        }
        match unit_candidate.get(&event.cookware_unit_id) {
            None => add_violation(
                &mut violations,
                CookwareLifetimeViolationCode::UnknownCookwareUnit,
                Some(&event.candidate_id),
                Some(&event.cookware_unit_id),
                Some(&event.event_id),
                "event unit ID does not appear in any benchmark run",
            ),
            Some(expected_candidate) if expected_candidate != &event.candidate_id => add_violation(
                &mut violations,
                CookwareLifetimeViolationCode::UnitIdentityConflict,
                Some(&event.candidate_id),
                Some(&event.cookware_unit_id),
                Some(&event.event_id),
                "event candidate does not match the candidate assigned to this physical unit",
            ),
            Some(_) => {}
        }

        if event.kind == CookwareLifecycleEventKind::StageAssessment {
            if let Some(run_id) = event.run_id.as_deref() {
                match run_by_id.get(run_id) {
                    None => add_violation(
                        &mut violations,
                        CookwareLifetimeViolationCode::UnknownBenchmarkRun,
                        Some(&event.candidate_id),
                        Some(&event.cookware_unit_id),
                        Some(&event.event_id),
                        "stage assessment references a run absent from the benchmark manifest",
                    ),
                    Some(run) => {
                        *run_link_counts.entry(run_id.to_owned()).or_default() += 1;
                        if run.candidate_id != event.candidate_id
                            || run.cookware_unit_id != event.cookware_unit_id
                        {
                            add_violation(
                                &mut violations,
                                CookwareLifetimeViolationCode::RunIdentityMismatch,
                                Some(&event.candidate_id),
                                Some(&event.cookware_unit_id),
                                Some(&event.event_id),
                                "assessment run candidate/unit identity differs from the lifecycle event",
                            );
                        }
                    }
                }
            }
        }
        events_by_unit
            .entry(event.cookware_unit_id.clone())
            .or_default()
            .push(event);
    }

    for run in &manifest.benchmark.runs {
        match run_link_counts.get(&run.run_id).copied().unwrap_or_default() {
            0 => add_violation(
                &mut violations,
                CookwareLifetimeViolationCode::UnreferencedBenchmarkRun,
                Some(&run.candidate_id),
                Some(&run.cookware_unit_id),
                None,
                "every benchmark run must be linked to one lifecycle stage assessment",
            ),
            1 => {}
            _ => add_violation(
                &mut violations,
                CookwareLifetimeViolationCode::DuplicateRunLink,
                Some(&run.candidate_id),
                Some(&run.cookware_unit_id),
                None,
                "a benchmark run may be linked to only one stage assessment",
            ),
        }
    }

    for (candidate_id, units) in &units_by_candidate {
        if units.len() < manifest.protocol.minimum_independent_units_per_candidate as usize {
            add_violation(
                &mut violations,
                CookwareLifetimeViolationCode::InsufficientIndependentUnits,
                Some(candidate_id),
                None,
                None,
                format!(
                    "candidate has {} distinct physical units; protocol requires {}",
                    units.len(),
                    manifest.protocol.minimum_independent_units_per_candidate
                ),
            );
        }
        for unit_id in units {
            if !events_by_unit.contains_key(unit_id) {
                add_violation(
                    &mut violations,
                    CookwareLifetimeViolationCode::MissingUnitTimeline,
                    Some(candidate_id),
                    Some(unit_id),
                    None,
                    "every physical unit in the comparison requires an explicit lifecycle timeline",
                );
            }
        }
    }

    for (unit_id, timeline) in &mut events_by_unit {
        timeline.sort_by_key(|event| event.sequence_index);
        let candidate_id = timeline.first().map(|event| event.candidate_id.as_str());
        let baseline_events: Vec<_> = timeline
            .iter()
            .filter(|event| {
                event.kind == CookwareLifecycleEventKind::StageAssessment
                    && event.stage == CookwareLifecycleStage::Baseline
            })
            .collect();
        if baseline_events.len() != 1 {
            add_violation(
                &mut violations,
                CookwareLifetimeViolationCode::MissingBaseline,
                candidate_id,
                Some(unit_id),
                None,
                "each physical unit requires exactly one baseline stage assessment",
            );
        }

        let use_batch_count = timeline
            .iter()
            .filter(|event| event.kind == CookwareLifecycleEventKind::UseBatchCompleted)
            .count();
        if use_batch_count == 0 {
            add_violation(
                &mut violations,
                CookwareLifetimeViolationCode::MissingUseBatch,
                candidate_id,
                Some(unit_id),
                None,
                "each physical unit requires at least one completed-use batch",
            );
        }

        let in_service_assessment_count = timeline
            .iter()
            .filter(|event| {
                event.kind == CookwareLifecycleEventKind::StageAssessment
                    && event.stage == CookwareLifecycleStage::InServiceCheckpoint
            })
            .count();
        if in_service_assessment_count
            < manifest.protocol.minimum_in_service_checkpoints_per_unit as usize
        {
            add_violation(
                &mut violations,
                CookwareLifetimeViolationCode::InsufficientInServiceCheckpoints,
                candidate_id,
                Some(unit_id),
                None,
                format!(
                    "unit has {} in-service assessments; protocol requires {}",
                    in_service_assessment_count,
                    manifest.protocol.minimum_in_service_checkpoints_per_unit
                ),
            );
        }

        if let Some(first) = timeline.first() {
            if first.sequence_index != 0
                || first.kind != CookwareLifecycleEventKind::StageAssessment
                || first.stage != CookwareLifecycleStage::Baseline
                || !first.cumulative_cycles.is_zero()
                || first.condition_after != CookwareLifecycleCondition::New
            {
                add_violation(
                    &mut violations,
                    CookwareLifetimeViolationCode::InvalidBaseline,
                    Some(&first.candidate_id),
                    Some(unit_id),
                    Some(&first.event_id),
                    "the first event must be a new, zero-cycle baseline assessment at sequence index zero",
                );
            }
        }

        let mut previous_counters: Option<CookwareDutyCycleCountersV1> = None;
        let mut previous_condition: Option<CookwareLifecycleCondition> = None;
        let mut retired = false;

        for (index, event) in timeline.iter().enumerate() {
            if event.sequence_index as usize != index {
                add_violation(
                    &mut violations,
                    CookwareLifetimeViolationCode::InvalidSequence,
                    Some(&event.candidate_id),
                    Some(unit_id),
                    Some(&event.event_id),
                    "sequence indexes must be contiguous, unique, and start at zero for each unit",
                );
            }
            if retired {
                add_violation(
                    &mut violations,
                    CookwareLifetimeViolationCode::RetiredUnitHasLaterEvents,
                    Some(&event.candidate_id),
                    Some(unit_id),
                    Some(&event.event_id),
                    "no lifecycle event may follow retirement of a physical unit",
                );
            }

            if event.kind == CookwareLifecycleEventKind::StageAssessment {
                let required_prior_event = match event.stage {
                    CookwareLifecycleStage::PostMaintenance => Some((
                        CookwareLifecycleEventKind::MaintenanceCompleted,
                        CookwareLifetimeViolationCode::MissingMaintenanceEvent,
                        "post-maintenance assessment requires an earlier maintenance event",
                    )),
                    CookwareLifecycleStage::PostRepair => Some((
                        CookwareLifecycleEventKind::RepairCompleted,
                        CookwareLifetimeViolationCode::MissingRepairEvent,
                        "post-repair assessment requires an earlier repair event",
                    )),
                    _ => None,
                };
                if let Some((required_kind, code, detail)) = required_prior_event {
                    let has_prior_event = timeline.iter().take(index).any(|prior| {
                        prior.kind == required_kind
                            && prior.sequence_index < event.sequence_index
                    });
                    if !has_prior_event {
                        add_violation(
                            &mut violations,
                            code,
                            Some(&event.candidate_id),
                            Some(unit_id),
                            Some(&event.event_id),
                            detail,
                        );
                    }
                }
            }

            if let Some(previous) = previous_counters {
                if !event.cumulative_cycles.is_componentwise_at_least(previous) {
                    add_violation(
                        &mut violations,
                        CookwareLifetimeViolationCode::CounterRegression,
                        Some(&event.candidate_id),
                        Some(unit_id),
                        Some(&event.event_id),
                        "cumulative duty-cycle counters cannot decrease",
                    );
                }
                if event.kind == CookwareLifecycleEventKind::UseBatchCompleted {
                    if !event.cumulative_cycles.strictly_increased_from(previous) {
                        add_violation(
                            &mut violations,
                            CookwareLifetimeViolationCode::UseBatchDidNotAdvanceCounters,
                            Some(&event.candidate_id),
                            Some(unit_id),
                            Some(&event.event_id),
                            "a completed-use batch must advance at least one cumulative counter",
                        );
                    }
                } else if event.cumulative_cycles != previous {
                    add_violation(
                        &mut violations,
                        CookwareLifetimeViolationCode::CounterAdvancedOutsideUseBatch,
                        Some(&event.candidate_id),
                        Some(unit_id),
                        Some(&event.event_id),
                        "only a completed-use event may increment duty-cycle counters",
                    );
                }

                if previous_condition == Some(CookwareLifecycleCondition::Failed)
                    && event.condition_after != CookwareLifecycleCondition::Failed
                    && event.kind != CookwareLifecycleEventKind::RepairCompleted
                    && event.kind != CookwareLifecycleEventKind::Retired
                {
                    add_violation(
                        &mut violations,
                        CookwareLifetimeViolationCode::InvalidConditionTransition,
                        Some(&event.candidate_id),
                        Some(unit_id),
                        Some(&event.event_id),
                        "a failed unit may return to service only through an explicit repair event",
                    );
                }
            }

            if event.kind == CookwareLifecycleEventKind::Retired
                || event.condition_after == CookwareLifecycleCondition::Retired
            {
                retired = true;
            }
            previous_counters = Some(event.cumulative_cycles);
            previous_condition = Some(event.condition_after);
        }
    }

    CookwareLifetimeValidationReport {
        study_id: manifest.study_id.clone(),
        benchmark_report,
        violations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cookware::{
        CookwareBenchmarkManifestV1, CookwareCandidateV1, CookwareEvidenceClass, CookwareMetric,
        CookwareObservationV1, CookwareProtocolV1, CookwareRole, CookwareRunV1,
        CraftExperienceRecordV1, FoodContactEvidenceStatus,
    };

    fn candidate(candidate_id: &str, role: CookwareRole) -> CookwareCandidateV1 {
        CookwareCandidateV1 {
            candidate_id: candidate_id.into(),
            design_revision_id: format!("{candidate_id}-rev1"),
            role,
            construction_family_id: format!("{candidate_id}-construction"),
            intended_task_profile_id: "saute-task-v1".into(),
            material_dossier_ref: format!("dossier://{candidate_id}"),
            food_contact_status: FoodContactEvidenceStatus::Unverified,
            food_contact_qualification_ref: None,
            food_contact_jurisdiction: None,
        }
    }

    fn run(
        run_id: &str,
        candidate_id: &str,
        unit_id: &str,
        block_id: &str,
        order_index: u32,
    ) -> CookwareRunV1 {
        CookwareRunV1 {
            run_id: run_id.into(),
            candidate_id: candidate_id.into(),
            cookware_unit_id: unit_id.into(),
            block_id: block_id.into(),
            order_index,
            protocol_id: "thermal-v1".into(),
            task_profile_id: "saute-task-v1".into(),
            geometry_profile_id: "pan-28cm-v1".into(),
            heat_source_profile_id: "induction-2kw-v1".into(),
            load_profile_id: "standardized-250g-load-v1".into(),
            ambient_profile_id: "lab-20c-v1".into(),
            observations: vec![CookwareObservationV1 {
                metric: CookwareMetric::SurfaceTemperatureSpreadMilliCelsius,
                value: 1_000,
                evidence_class: CookwareEvidenceClass::ObservedInstrumental,
                evidence_ref: format!("evidence://{run_id}"),
                measurement_method_ref: "method://temperature-grid-v1".into(),
                calibration_ref: Some("calibration://thermocouples-2026-10".into()),
                uncertainty_abs: Some(25),
            }],
        }
    }

    fn assessment(
        event_id: &str,
        unit_id: &str,
        candidate_id: &str,
        sequence_index: u32,
        stage: CookwareLifecycleStage,
        condition: CookwareLifecycleCondition,
        counters: CookwareDutyCycleCountersV1,
        run_id: &str,
    ) -> CookwareLifecycleEventV1 {
        CookwareLifecycleEventV1 {
            event_id: event_id.into(),
            cookware_unit_id: unit_id.into(),
            candidate_id: candidate_id.into(),
            sequence_index,
            kind: CookwareLifecycleEventKind::StageAssessment,
            stage,
            condition_after: condition,
            cumulative_cycles: counters,
            evidence_ref: format!("evidence://{event_id}"),
            method_ref: "method://lifetime-observation-v1".into(),
            run_id: Some(run_id.into()),
            failure_mode: None,
            repair_parts: vec![],
            repair_duration_seconds: None,
            repair_cost_minor_units: None,
            repair_currency_code: None,
        }
    }

    fn use_batch(
        event_id: &str,
        unit_id: &str,
        candidate_id: &str,
        sequence_index: u32,
        counters: CookwareDutyCycleCountersV1,
    ) -> CookwareLifecycleEventV1 {
        CookwareLifecycleEventV1 {
            event_id: event_id.into(),
            cookware_unit_id: unit_id.into(),
            candidate_id: candidate_id.into(),
            sequence_index,
            kind: CookwareLifecycleEventKind::UseBatchCompleted,
            stage: CookwareLifecycleStage::InServiceCheckpoint,
            condition_after: CookwareLifecycleCondition::InService,
            cumulative_cycles: counters,
            evidence_ref: format!("evidence://{event_id}"),
            method_ref: "method://duty-cycle-v1".into(),
            run_id: None,
            failure_mode: None,
            repair_parts: vec![],
            repair_duration_seconds: None,
            repair_cost_minor_units: None,
            repair_currency_code: None,
        }
    }

    fn fixture() -> CookwareLifetimeManifestV1 {
        let baseline_counters = CookwareDutyCycleCountersV1::default();
        let used_counters = CookwareDutyCycleCountersV1 {
            thermal_cycles: 1,
            cooking_load_cycles: 1,
            cleaning_cycles: 1,
            abrasion_cycles: 0,
            declared_misuse_cycles: 0,
        };
        CookwareLifetimeManifestV1 {
            schema_version: COOKWARE_LIFETIME_SCHEMA_VERSION,
            study_id: "cookware-lifetime-fixture-v1".into(),
            protocol: CookwareLifetimeProtocolV1 {
                protocol_id: "lifecycle-protocol-v1".into(),
                duty_cycle_profile_id: "saute-clean-cool-v1".into(),
                minimum_independent_units_per_candidate: 1,
                minimum_in_service_checkpoints_per_unit: 1,
                calendar_life_claim_years: None,
                cycle_to_calendar_equivalence_ref: None,
            },
            benchmark: CookwareBenchmarkManifestV1 {
                schema_version: 1,
                benchmark_id: "matched-lifetime-fixture-v1".into(),
                protocol: CookwareProtocolV1 {
                    protocol_id: "thermal-v1".into(),
                    task_profile_id: "saute-task-v1".into(),
                    geometry_profile_id: "pan-28cm-v1".into(),
                    heat_source_profile_id: "induction-2kw-v1".into(),
                    load_profile_id: "standardized-250g-load-v1".into(),
                    ambient_profile_id: "lab-20c-v1".into(),
                    jurisdiction: "ZA".into(),
                    randomization_method_ref: "method://balanced-order-v1".into(),
                    minimum_blocks_per_candidate: 1,
                    minimum_independent_units_per_candidate: 1,
                    required_metrics: vec![CookwareMetric::SurfaceTemperatureSpreadMilliCelsius],
                    empirical_comparison: true,
                    claims_food_contact_suitability: false,
                },
                candidates: vec![
                    candidate("candidate-pan", CookwareRole::Candidate),
                    candidate("reference-pan", CookwareRole::Reference),
                ],
                runs: vec![
                    run("candidate-base", "candidate-pan", "candidate-unit-1", "baseline", 0),
                    run("reference-base", "reference-pan", "reference-unit-1", "baseline", 1),
                    run("candidate-used", "candidate-pan", "candidate-unit-1", "in-service", 0),
                    run("reference-used", "reference-pan", "reference-unit-1", "in-service", 1),
                ],
                craft_experience: Vec::<CraftExperienceRecordV1>::new(),
            },
            events: vec![
                assessment(
                    "candidate-baseline",
                    "candidate-unit-1",
                    "candidate-pan",
                    0,
                    CookwareLifecycleStage::Baseline,
                    CookwareLifecycleCondition::New,
                    baseline_counters,
                    "candidate-base",
                ),
                use_batch(
                    "candidate-use",
                    "candidate-unit-1",
                    "candidate-pan",
                    1,
                    used_counters,
                ),
                assessment(
                    "candidate-in-service",
                    "candidate-unit-1",
                    "candidate-pan",
                    2,
                    CookwareLifecycleStage::InServiceCheckpoint,
                    CookwareLifecycleCondition::InService,
                    used_counters,
                    "candidate-used",
                ),
                assessment(
                    "reference-baseline",
                    "reference-unit-1",
                    "reference-pan",
                    0,
                    CookwareLifecycleStage::Baseline,
                    CookwareLifecycleCondition::New,
                    baseline_counters,
                    "reference-base",
                ),
                use_batch(
                    "reference-use",
                    "reference-unit-1",
                    "reference-pan",
                    1,
                    used_counters,
                ),
                assessment(
                    "reference-in-service",
                    "reference-unit-1",
                    "reference-pan",
                    2,
                    CookwareLifecycleStage::InServiceCheckpoint,
                    CookwareLifecycleCondition::InService,
                    used_counters,
                    "reference-used",
                ),
            ],
        }
    }

    #[test]
    fn matched_longitudinal_fixture_is_structurally_valid() {
        assert!(validate_cookware_lifetime(&fixture()).is_valid());
    }

    #[test]
    fn calendar_year_claim_requires_equivalence_evidence() {
        let mut manifest = fixture();
        manifest.protocol.calendar_life_claim_years = Some(5);
        let report = validate_cookware_lifetime(&manifest);
        assert!(report.violations.iter().any(|violation| {
            violation.code
                == CookwareLifetimeViolationCode::UnsupportedCalendarLifeExtrapolation
        }));
    }

    #[test]
    fn use_batch_must_advance_at_least_one_counter() {
        let mut manifest = fixture();
        manifest.events[1].cumulative_cycles = CookwareDutyCycleCountersV1::default();
        let report = validate_cookware_lifetime(&manifest);
        assert!(report.violations.iter().any(|violation| {
            violation.code == CookwareLifetimeViolationCode::UseBatchDidNotAdvanceCounters
        }));
    }

    #[test]
    fn duty_cycle_counter_regression_is_rejected() {
        let mut manifest = fixture();
        manifest.events[2].cumulative_cycles.thermal_cycles = 0;
        let report = validate_cookware_lifetime(&manifest);
        assert!(report.violations.iter().any(|violation| {
            violation.code == CookwareLifetimeViolationCode::CounterRegression
        }));
    }

    #[test]
    fn baseline_must_start_at_zero_cycles() {
        let mut manifest = fixture();
        manifest.events[0].cumulative_cycles.thermal_cycles = 1;
        let report = validate_cookware_lifetime(&manifest);
        assert!(report.violations.iter().any(|violation| {
            violation.code == CookwareLifetimeViolationCode::InvalidBaseline
        }));
    }

    #[test]
    fn all_measured_runs_must_be_linked_exactly_once() {
        let mut manifest = fixture();
        manifest.events[2].run_id = Some("reference-used".into());
        let report = validate_cookware_lifetime(&manifest);
        assert!(report.violations.iter().any(|violation| {
            violation.code == CookwareLifetimeViolationCode::DuplicateRunLink
        }));
        assert!(report.violations.iter().any(|violation| {
            violation.code == CookwareLifetimeViolationCode::UnreferencedBenchmarkRun
        }));
    }

    #[test]
    fn unit_and_candidate_identity_must_match_benchmark_run() {
        let mut manifest = fixture();
        manifest.events[2].candidate_id = "reference-pan".into();
        let report = validate_cookware_lifetime(&manifest);
        assert!(report.violations.iter().any(|violation| {
            violation.code == CookwareLifetimeViolationCode::RunIdentityMismatch
        }));
    }

    #[test]
    fn repair_event_requires_parts_duration_cost_and_currency() {
        let mut manifest = fixture();
        let mut repair = manifest.events[2].clone();
        repair.event_id = "candidate-repair".into();
        repair.sequence_index = 3;
        repair.kind = CookwareLifecycleEventKind::RepairCompleted;
        repair.stage = CookwareLifecycleStage::PostRepair;
        repair.run_id = None;
        repair.repair_parts = vec![];
        manifest.events.insert(3, repair);
        // Re-number this specimen's later events so the injected event occupies
        // a distinct sequence position and the test isolates repair evidence.
        for event in manifest.events.iter_mut().filter(|event| {
            event.cookware_unit_id == "candidate-unit-1" && event.sequence_index >= 3
        }) {
            if event.event_id != "candidate-repair" {
                event.sequence_index += 1;
            }
        }
        let report = validate_cookware_lifetime(&manifest);
        assert!(report.violations.iter().any(|violation| {
            violation.code == CookwareLifetimeViolationCode::MissingRepairEvidence
        }));
    }

    #[test]
    fn retired_unit_cannot_have_subsequent_events() {
        let mut manifest = fixture();
        let mut retirement = manifest.events[2].clone();
        retirement.event_id = "candidate-retired".into();
        retirement.sequence_index = 3;
        retirement.kind = CookwareLifecycleEventKind::Retired;
        retirement.stage = CookwareLifecycleStage::EndOfLife;
        retirement.condition_after = CookwareLifecycleCondition::Retired;
        retirement.run_id = None;
        manifest.events.push(retirement);
        let mut after_retirement = manifest.events[2].clone();
        after_retirement.event_id = "candidate-after-retirement".into();
        after_retirement.sequence_index = 4;
        manifest.events.push(after_retirement);
        let report = validate_cookware_lifetime(&manifest);
        assert!(report.violations.iter().any(|violation| {
            violation.code == CookwareLifetimeViolationCode::RetiredUnitHasLaterEvents
        }));
    }

    #[test]
    fn post_maintenance_and_repair_assessments_require_interventions() {
        let mut manifest = fixture();
        let mut post_maintenance = assessment(
            "candidate-post-maintenance",
            "candidate-unit-1",
            "candidate-pan",
            3,
            CookwareLifecycleStage::PostMaintenance,
            CookwareLifecycleCondition::InService,
            manifest.events[2].cumulative_cycles,
            "unknown-maintenance-run",
        );
        post_maintenance.run_id = Some("unknown-maintenance-run".into());
        let mut post_repair = assessment(
            "candidate-post-repair",
            "candidate-unit-1",
            "candidate-pan",
            4,
            CookwareLifecycleStage::PostRepair,
            CookwareLifecycleCondition::InService,
            manifest.events[2].cumulative_cycles,
            "unknown-repair-run",
        );
        post_repair.run_id = Some("unknown-repair-run".into());
        manifest.events.push(post_maintenance);
        manifest.events.push(post_repair);

        let report = validate_cookware_lifetime(&manifest);
        assert!(report.violations.iter().any(|violation| {
            violation.code == CookwareLifetimeViolationCode::MissingMaintenanceEvent
        }));
        assert!(report.violations.iter().any(|violation| {
            violation.code == CookwareLifetimeViolationCode::MissingRepairEvent
        }));
    }

    #[test]
    fn lifecycle_event_sequence_must_be_contiguous() {
        let mut manifest = fixture();
        manifest.events[2].sequence_index = 4;
        let report = validate_cookware_lifetime(&manifest);
        assert!(report.violations.iter().any(|violation| {
            violation.code == CookwareLifetimeViolationCode::InvalidSequence
        }));
    }
}
