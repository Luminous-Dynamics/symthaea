// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic, effects-disabled farm-platform screening kernel.
//!
//! This crate intentionally does not connect to actuators, crop treatment,
//! irrigation, procurement, or live inventory. A result is a candidate-screening
//! artifact, not permission to operate machinery or proof of agronomic benefit.

use std::collections::{BTreeMap, BTreeSet};

pub const CONTRACT_VERSION: &str = "AGRI-ROBOTICS-BENCH-V1";
pub const MAX_ANNUAL_OPERATING_HOURS: u64 = 8_760;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskKind {
    FieldScouting,
    SoilMoistureMonitoring,
    MechanicalWeeding,
    BroadAreaTreatment,
    GreenhouseHarvest,
    ProduceTransport,
    NurseryHandling,
    LivestockMonitoring,
}

impl TaskKind {
    pub const fn service_unit(self) -> ServiceUnit {
        match self {
            Self::FieldScouting
            | Self::SoilMoistureMonitoring
            | Self::MechanicalWeeding
            | Self::BroadAreaTreatment => ServiceUnit::SquareMetersPerDay,
            Self::GreenhouseHarvest | Self::ProduceTransport => ServiceUnit::KilogramsPerDay,
            Self::NurseryHandling => ServiceUnit::PlantsPerDay,
            Self::LivestockMonitoring => ServiceUnit::AnimalsPerDay,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ServiceUnit {
    SquareMetersPerDay,
    KilogramsPerDay,
    PlantsPerDay,
    AnimalsPerDay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlatformClass {
    ManualCurrentPractice,
    FixedSensorWorkflow,
    TractorImplement,
    ModularGroundRobot,
    RailOrGantry,
    AerialSurvey,
    LivestockSpecific,
    PackhouseAutomation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EvidenceDomain {
    TaskDefinition,
    SiteGeometry,
    TerrainEnvelope,
    NavigationEnvelope,
    SafetyEnvelope,
    CostModel,
    Serviceability,
}

impl EvidenceDomain {
    pub const ALL: [Self; 7] = [
        Self::TaskDefinition,
        Self::SiteGeometry,
        Self::TerrainEnvelope,
        Self::NavigationEnvelope,
        Self::SafetyEnvelope,
        Self::CostModel,
        Self::Serviceability,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TaskDefinition => "task_definition",
            Self::SiteGeometry => "site_geometry",
            Self::TerrainEnvelope => "terrain_envelope",
            Self::NavigationEnvelope => "navigation_envelope",
            Self::SafetyEnvelope => "safety_envelope",
            Self::CostModel => "cost_model",
            Self::Serviceability => "serviceability",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceState {
    Verified,
    Estimated,
    Missing,
    Stale,
    Conflicting,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceRecord {
    pub domain: EvidenceDomain,
    pub state: EvidenceState,
    /// A source URI, report identifier, or explicit synthetic-fixture reference.
    pub reference: Option<String>,
    /// SHA-256 of the exact referenced artifact when one is available.
    pub artifact_sha256: Option<String>,
}

impl EvidenceRecord {
    pub fn verified(
        domain: EvidenceDomain,
        reference: impl Into<String>,
        artifact_sha256: impl Into<String>,
    ) -> Self {
        Self {
            domain,
            state: EvidenceState::Verified,
            reference: Some(reference.into()),
            artifact_sha256: Some(artifact_sha256.into()),
        }
    }

    pub fn estimated(domain: EvidenceDomain, reference: impl Into<String>) -> Self {
        Self {
            domain,
            state: EvidenceState::Estimated,
            reference: Some(reference.into()),
            artifact_sha256: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScenarioProfile {
    pub id: String,
    pub task: TaskKind,
    /// Required capacity in the unit implied by the task, not an interchangeable
    /// generic score. For example, scouting is square metres/day and harvesting
    /// is kilograms/day.
    pub required_daily_capacity: u64,
    pub site_slope_permille: u32,
    pub available_row_clearance_cm: u32,
    pub available_turning_radius_cm: u32,
    pub site_roughness_mm: u32,
    pub gnss_available: bool,
    pub max_upfront_budget_cents: u64,
    pub max_annualized_cost_cents: u64,
    pub annual_operating_hours: u64,
    pub require_local_repair: bool,
    pub max_repair_lead_days: u32,
    pub evidence: Vec<EvidenceRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlatformProfile {
    pub id: String,
    pub class: PlatformClass,
    pub supported_tasks: Vec<TaskKind>,
    pub service_unit: ServiceUnit,
    pub max_slope_permille: u32,
    pub width_cm: u32,
    pub turning_radius_cm: u32,
    pub max_roughness_mm: u32,
    pub requires_gnss: bool,
    pub gnss_denied_capable: bool,
    pub daily_capacity_units: u64,
    pub acquisition_cost_cents: u64,
    pub operating_cost_cents_per_hour: u64,
    pub supervision_minutes_per_hour: u64,
    pub maintenance_minutes_per_100_hours: u64,
    pub useful_life_years: u32,
    pub local_repair_available: bool,
    pub repair_lead_days: u32,
    /// A task- and operating-envelope-specific safety case was reviewed.
    pub bounded_safety_case_verified: bool,
    /// A person can stop or take over the operation in the declared envelope.
    pub human_override_available: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssessmentStatus {
    Feasible,
    Provisional,
    Rejected,
    Blocked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BlockerCode {
    InvalidScenario,
    InvalidPlatformProfile,
    MalformedEvidence,
    DuplicateEvidenceDomain,
    MissingEvidenceRecord,
    EvidenceMissing(EvidenceDomain),
    EvidenceStale(EvidenceDomain),
    EvidenceConflicting(EvidenceDomain),
    UnsupportedTask,
    IncompatibleServiceUnit,
    SlopeExceeded,
    RowClearanceTooNarrow,
    TurningRadiusExceeded,
    TerrainRoughnessExceeded,
    GnssUnavailable,
    CapacityInsufficient,
    SafetyCaseMissing,
    HumanOverrideMissing,
    UpfrontBudgetExceeded,
    AnnualizedCostExceeded,
    LocalRepairUnavailable,
    RepairLeadTimeExceeded,
    ArithmeticOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerformanceMetrics {
    pub annualized_total_cost_cents: u64,
    pub annual_supervision_minutes: u64,
    pub annual_maintenance_minutes: u64,
    pub repair_lead_days: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assessment {
    pub candidate_id: String,
    pub status: AssessmentStatus,
    pub blockers: Vec<BlockerCode>,
    pub uncertain_domains: Vec<EvidenceDomain>,
    /// Indicative cost/operational metrics. Present for verified or estimated
    /// cost inputs; status must be checked before using them for a decision.
    pub metrics: Option<PerformanceMetrics>,
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn nonempty(value: &Option<String>) -> bool {
    value.as_deref().is_some_and(|s| !s.trim().is_empty())
}

fn evidence_record_is_well_formed(record: &EvidenceRecord) -> bool {
    if let Some(digest) = record.artifact_sha256.as_deref() {
        if !valid_sha256(digest) {
            return false;
        }
    }

    match record.state {
        EvidenceState::Verified => {
            nonempty(&record.reference)
                && record.artifact_sha256.as_deref().is_some_and(valid_sha256)
        }
        EvidenceState::Estimated => nonempty(&record.reference),
        EvidenceState::Missing => true,
        EvidenceState::Stale | EvidenceState::Conflicting => nonempty(&record.reference),
    }
}

fn evidence_map(
    records: &[EvidenceRecord],
) -> Result<BTreeMap<EvidenceDomain, EvidenceState>, BTreeSet<BlockerCode>> {
    let mut states = BTreeMap::new();
    let mut problems = BTreeSet::new();

    for record in records {
        if !evidence_record_is_well_formed(record) {
            problems.insert(BlockerCode::MalformedEvidence);
        }
        if states.insert(record.domain, record.state).is_some() {
            problems.insert(BlockerCode::DuplicateEvidenceDomain);
        }
    }

    for domain in EvidenceDomain::ALL {
        if !states.contains_key(&domain) {
            problems.insert(BlockerCode::MissingEvidenceRecord);
        }
    }

    if problems.is_empty() {
        Ok(states)
    } else {
        Err(problems)
    }
}

fn blocked(
    candidate_id: &str,
    blockers: BTreeSet<BlockerCode>,
    uncertain_domains: Vec<EvidenceDomain>,
) -> Assessment {
    Assessment {
        candidate_id: candidate_id.to_owned(),
        status: AssessmentStatus::Blocked,
        blockers: blockers.into_iter().collect(),
        uncertain_domains,
        metrics: None,
    }
}

fn rejected(
    candidate_id: &str,
    blockers: BTreeSet<BlockerCode>,
    uncertain_domains: Vec<EvidenceDomain>,
) -> Assessment {
    Assessment {
        candidate_id: candidate_id.to_owned(),
        status: AssessmentStatus::Rejected,
        blockers: blockers.into_iter().collect(),
        uncertain_domains,
        metrics: None,
    }
}

fn ceil_div(value: u64, divisor: u64) -> Option<u64> {
    if divisor == 0 {
        return None;
    }
    let quotient = value / divisor;
    if value % divisor == 0 {
        Some(quotient)
    } else {
        quotient.checked_add(1)
    }
}

fn multiply_divide_ceil(value: u64, multiplier: u64, divisor: u64) -> Option<u64> {
    ceil_div(value.checked_mul(multiplier)?, divisor)
}

fn calculate_metrics(
    scenario: &ScenarioProfile,
    platform: &PlatformProfile,
) -> Option<PerformanceMetrics> {
    if platform.useful_life_years == 0 {
        return None;
    }
    let annualized_capital =
        ceil_div(platform.acquisition_cost_cents, u64::from(platform.useful_life_years))?;
    let annual_operating_cost = platform
        .operating_cost_cents_per_hour
        .checked_mul(scenario.annual_operating_hours)?;
    let annualized_total_cost_cents = annualized_capital.checked_add(annual_operating_cost)?;
    let annual_supervision_minutes = platform
        .supervision_minutes_per_hour
        .checked_mul(scenario.annual_operating_hours)?;
    let annual_maintenance_minutes = multiply_divide_ceil(
        scenario.annual_operating_hours,
        platform.maintenance_minutes_per_100_hours,
        100,
    )?;

    Some(PerformanceMetrics {
        annualized_total_cost_cents,
        annual_supervision_minutes,
        annual_maintenance_minutes,
        repair_lead_days: platform.repair_lead_days,
    })
}

/// Assess one platform against one declared site/task scenario.
///
/// Feasible means only that the candidate passed this screening contract for
/// the provided profile and evidence states. It is not a safety certification,
/// procurement recommendation, agronomic result, or authorization to operate.
pub fn assess(
    scenario: &ScenarioProfile,
    platform: &PlatformProfile,
) -> Assessment {
    if scenario.required_daily_capacity == 0
        || scenario.annual_operating_hours > MAX_ANNUAL_OPERATING_HOURS
    {
        return blocked(
            &platform.id,
            BTreeSet::from([BlockerCode::InvalidScenario]),
            Vec::new(),
        );
    }
    if platform.useful_life_years == 0 {
        return blocked(
            &platform.id,
            BTreeSet::from([BlockerCode::InvalidPlatformProfile]),
            Vec::new(),
        );
    }

    let states = match evidence_map(&scenario.evidence) {
        Ok(states) => states,
        Err(problems) => return blocked(&platform.id, problems, Vec::new()),
    };

    let mut evidence_blockers = BTreeSet::new();
    let mut uncertain_domains = Vec::new();

    for domain in EvidenceDomain::ALL {
        match states.get(&domain).copied() {
            Some(EvidenceState::Verified) => {}
            Some(EvidenceState::Estimated) => uncertain_domains.push(domain),
            Some(EvidenceState::Missing) => {
                evidence_blockers.insert(BlockerCode::EvidenceMissing(domain));
            }
            Some(EvidenceState::Stale) => {
                evidence_blockers.insert(BlockerCode::EvidenceStale(domain));
            }
            Some(EvidenceState::Conflicting) => {
                evidence_blockers.insert(BlockerCode::EvidenceConflicting(domain));
            }
            None => {
                evidence_blockers.insert(BlockerCode::MissingEvidenceRecord);
            }
        }
    }

    // Unknown, stale, or conflicting evidence is never silently replaced by the
    // typed numeric fields carried by a scenario. Stop before screening.
    if !evidence_blockers.is_empty() {
        return blocked(&platform.id, evidence_blockers, uncertain_domains);
    }

    let state = |domain: EvidenceDomain| states.get(&domain).copied();
    let mut violations = BTreeSet::new();

    if state(EvidenceDomain::TaskDefinition) == Some(EvidenceState::Verified) {
        if !platform.supported_tasks.contains(&scenario.task) {
            violations.insert(BlockerCode::UnsupportedTask);
        }
        if platform.service_unit != scenario.task.service_unit() {
            violations.insert(BlockerCode::IncompatibleServiceUnit);
        }
        if platform.daily_capacity_units < scenario.required_daily_capacity {
            violations.insert(BlockerCode::CapacityInsufficient);
        }
    }

    if state(EvidenceDomain::SiteGeometry) == Some(EvidenceState::Verified) {
        if platform.width_cm > scenario.available_row_clearance_cm {
            violations.insert(BlockerCode::RowClearanceTooNarrow);
        }
        if platform.turning_radius_cm > scenario.available_turning_radius_cm {
            violations.insert(BlockerCode::TurningRadiusExceeded);
        }
    }

    if state(EvidenceDomain::TerrainEnvelope) == Some(EvidenceState::Verified) {
        if scenario.site_slope_permille > platform.max_slope_permille {
            violations.insert(BlockerCode::SlopeExceeded);
        }
        if scenario.site_roughness_mm > platform.max_roughness_mm {
            violations.insert(BlockerCode::TerrainRoughnessExceeded);
        }
    }

    if state(EvidenceDomain::NavigationEnvelope) == Some(EvidenceState::Verified)
        && platform.requires_gnss
        && !scenario.gnss_available
        && !platform.gnss_denied_capable
    {
        violations.insert(BlockerCode::GnssUnavailable);
    }

    if state(EvidenceDomain::SafetyEnvelope) == Some(EvidenceState::Verified) {
        if !platform.bounded_safety_case_verified {
            violations.insert(BlockerCode::SafetyCaseMissing);
        }
        if !platform.human_override_available {
            violations.insert(BlockerCode::HumanOverrideMissing);
        }
    }

    if !violations.is_empty() {
        return rejected(&platform.id, violations, uncertain_domains);
    }

    let metrics = match calculate_metrics(scenario, platform) {
        Some(metrics) => metrics,
        None => {
            return blocked(
                &platform.id,
                BTreeSet::from([BlockerCode::ArithmeticOverflow]),
                uncertain_domains,
            );
        }
    };

    let mut economic_or_service_violations = BTreeSet::new();

    if state(EvidenceDomain::CostModel) == Some(EvidenceState::Verified) {
        if platform.acquisition_cost_cents > scenario.max_upfront_budget_cents {
            economic_or_service_violations.insert(BlockerCode::UpfrontBudgetExceeded);
        }
        if metrics.annualized_total_cost_cents > scenario.max_annualized_cost_cents {
            economic_or_service_violations.insert(BlockerCode::AnnualizedCostExceeded);
        }
    }

    if state(EvidenceDomain::Serviceability) == Some(EvidenceState::Verified) {
        if scenario.require_local_repair && !platform.local_repair_available {
            economic_or_service_violations.insert(BlockerCode::LocalRepairUnavailable);
        }
        if platform.repair_lead_days > scenario.max_repair_lead_days {
            economic_or_service_violations.insert(BlockerCode::RepairLeadTimeExceeded);
        }
    }

    if !economic_or_service_violations.is_empty() {
        return Assessment {
            candidate_id: platform.id.clone(),
            status: AssessmentStatus::Rejected,
            blockers: economic_or_service_violations.into_iter().collect(),
            uncertain_domains,
            metrics: Some(metrics),
        };
    }

    Assessment {
        candidate_id: platform.id.clone(),
        status: if uncertain_domains.is_empty() {
            AssessmentStatus::Feasible
        } else {
            AssessmentStatus::Provisional
        },
        blockers: Vec::new(),
        uncertain_domains,
        metrics: Some(metrics),
    }
}

fn dominates(a: &PerformanceMetrics, b: &PerformanceMetrics) -> bool {
    let no_worse = a.annualized_total_cost_cents <= b.annualized_total_cost_cents
        && a.annual_supervision_minutes <= b.annual_supervision_minutes
        && a.annual_maintenance_minutes <= b.annual_maintenance_minutes
        && a.repair_lead_days <= b.repair_lead_days;
    let strictly_better = a.annualized_total_cost_cents < b.annualized_total_cost_cents
        || a.annual_supervision_minutes < b.annual_supervision_minutes
        || a.annual_maintenance_minutes < b.annual_maintenance_minutes
        || a.repair_lead_days < b.repair_lead_days;
    no_worse && strictly_better
}

/// Return the non-dominated, fully feasible candidates in stable ID order.
///
/// No weighted sum is used. A trade-off between cost, human supervision,
/// maintenance, and repair lead time stays visible on the Pareto frontier.
/// Provisional, blocked, and rejected candidates are never promoted.
pub fn pareto_frontier(assessments: &[Assessment]) -> Vec<String> {
    let mut frontier = Vec::new();

    for (index, candidate) in assessments.iter().enumerate() {
        if candidate.status != AssessmentStatus::Feasible {
            continue;
        }
        let Some(candidate_metrics) = candidate.metrics.as_ref() else {
            continue;
        };

        let is_dominated = assessments.iter().enumerate().any(|(other_index, other)| {
            if index == other_index || other.status != AssessmentStatus::Feasible {
                return false;
            }
            other
                .metrics
                .as_ref()
                .is_some_and(|other_metrics| dominates(other_metrics, candidate_metrics))
        });

        if !is_dominated {
            frontier.push(candidate.candidate_id.clone());
        }
    }

    frontier.sort();
    frontier.dedup();
    frontier
}

#[cfg(test)]
mod tests {
    use super::*;

    const SYNTHETIC_FIXTURE_REFERENCE: &str =
        "inline-fixture:synthetic-agri-platform-bench-fixture-v1";
    const SYNTHETIC_FIXTURE_SHA256: &str =
        "db5822bd2544c966f3d8d5348e55551aef8e4d4f48cc562edf6a0deada73c3bf";

    #[derive(Debug, Clone, Copy)]
    enum Mutation {
        Baseline,
        UnsupportedTask,
        IncompatibleServiceUnit,
        SlopeExceeded,
        ClearanceTooNarrow,
        TurningRoomTooSmall,
        RoughnessExceeded,
        GnssUnavailable,
        CapacityShortfall,
        UpfrontBudgetExceeded,
        AnnualCostExceeded,
        LocalRepairUnavailable,
        RepairLeadTooLong,
        SafetyCaseMissing,
        HumanOverrideMissing,
        MissingTaskEvidence,
        StaleGeometryEvidence,
        ConflictingTerrainEvidence,
        MissingNavigationEvidence,
        EstimatedSafetyEvidence,
        EstimatedCostEvidence,
        EstimatedServiceabilityEvidence,
        EstimatedTerrainEvidence,
        MalformedVerifiedDigest,
        DuplicateEvidenceDomain,
        MissingEvidenceDomain,
        ArithmeticOverflow,
        GnssDeniedFallback,
        ManualCurrentPracticeFeasible,
        ExactCapacityBoundary,
        ExactSlopeBoundary,
        ExactClearanceBoundary,
        ZeroAnnualHours,
        KnownSlopePlusEstimatedCost,
        StaleCostEvidence,
        ConflictingServiceabilityEvidence,
        EmptyVerifiedSourceReference,
        InvalidOperatingHours,
    }

    struct FrozenCase {
        id: &'static str,
        mutation: Mutation,
        expected: AssessmentStatus,
    }

    const CASES: &[FrozenCase] = &[
        FrozenCase { id: "baseline-feasible", mutation: Mutation::Baseline, expected: AssessmentStatus::Feasible },
        FrozenCase { id: "unsupported-task", mutation: Mutation::UnsupportedTask, expected: AssessmentStatus::Rejected },
        FrozenCase { id: "incompatible-service-unit", mutation: Mutation::IncompatibleServiceUnit, expected: AssessmentStatus::Rejected },
        FrozenCase { id: "slope-exceeded", mutation: Mutation::SlopeExceeded, expected: AssessmentStatus::Rejected },
        FrozenCase { id: "clearance-too-narrow", mutation: Mutation::ClearanceTooNarrow, expected: AssessmentStatus::Rejected },
        FrozenCase { id: "turning-room-too-small", mutation: Mutation::TurningRoomTooSmall, expected: AssessmentStatus::Rejected },
        FrozenCase { id: "roughness-exceeded", mutation: Mutation::RoughnessExceeded, expected: AssessmentStatus::Rejected },
        FrozenCase { id: "gnss-unavailable", mutation: Mutation::GnssUnavailable, expected: AssessmentStatus::Rejected },
        FrozenCase { id: "capacity-shortfall", mutation: Mutation::CapacityShortfall, expected: AssessmentStatus::Rejected },
        FrozenCase { id: "upfront-budget-exceeded", mutation: Mutation::UpfrontBudgetExceeded, expected: AssessmentStatus::Rejected },
        FrozenCase { id: "annual-cost-exceeded", mutation: Mutation::AnnualCostExceeded, expected: AssessmentStatus::Rejected },
        FrozenCase { id: "local-repair-required", mutation: Mutation::LocalRepairUnavailable, expected: AssessmentStatus::Rejected },
        FrozenCase { id: "repair-lead-too-long", mutation: Mutation::RepairLeadTooLong, expected: AssessmentStatus::Rejected },
        FrozenCase { id: "safety-case-missing", mutation: Mutation::SafetyCaseMissing, expected: AssessmentStatus::Rejected },
        FrozenCase { id: "human-override-missing", mutation: Mutation::HumanOverrideMissing, expected: AssessmentStatus::Rejected },
        FrozenCase { id: "missing-task-evidence", mutation: Mutation::MissingTaskEvidence, expected: AssessmentStatus::Blocked },
        FrozenCase { id: "stale-geometry-evidence", mutation: Mutation::StaleGeometryEvidence, expected: AssessmentStatus::Blocked },
        FrozenCase { id: "conflicting-terrain-evidence", mutation: Mutation::ConflictingTerrainEvidence, expected: AssessmentStatus::Blocked },
        FrozenCase { id: "missing-navigation-evidence", mutation: Mutation::MissingNavigationEvidence, expected: AssessmentStatus::Blocked },
        FrozenCase { id: "estimated-safety-evidence", mutation: Mutation::EstimatedSafetyEvidence, expected: AssessmentStatus::Provisional },
        FrozenCase { id: "estimated-cost-evidence", mutation: Mutation::EstimatedCostEvidence, expected: AssessmentStatus::Provisional },
        FrozenCase { id: "estimated-serviceability-evidence", mutation: Mutation::EstimatedServiceabilityEvidence, expected: AssessmentStatus::Provisional },
        FrozenCase { id: "estimated-terrain-evidence", mutation: Mutation::EstimatedTerrainEvidence, expected: AssessmentStatus::Provisional },
        FrozenCase { id: "malformed-verified-digest", mutation: Mutation::MalformedVerifiedDigest, expected: AssessmentStatus::Blocked },
        FrozenCase { id: "duplicate-evidence-domain", mutation: Mutation::DuplicateEvidenceDomain, expected: AssessmentStatus::Blocked },
        FrozenCase { id: "missing-evidence-domain", mutation: Mutation::MissingEvidenceDomain, expected: AssessmentStatus::Blocked },
        FrozenCase { id: "arithmetic-overflow", mutation: Mutation::ArithmeticOverflow, expected: AssessmentStatus::Blocked },
        FrozenCase { id: "gnss-denied-fallback", mutation: Mutation::GnssDeniedFallback, expected: AssessmentStatus::Feasible },
        FrozenCase { id: "manual-current-practice-feasible", mutation: Mutation::ManualCurrentPracticeFeasible, expected: AssessmentStatus::Feasible },
        FrozenCase { id: "exact-capacity-boundary", mutation: Mutation::ExactCapacityBoundary, expected: AssessmentStatus::Feasible },
        FrozenCase { id: "exact-slope-boundary", mutation: Mutation::ExactSlopeBoundary, expected: AssessmentStatus::Feasible },
        FrozenCase { id: "exact-clearance-boundary", mutation: Mutation::ExactClearanceBoundary, expected: AssessmentStatus::Feasible },
        FrozenCase { id: "zero-annual-hours", mutation: Mutation::ZeroAnnualHours, expected: AssessmentStatus::Feasible },
        FrozenCase { id: "known-slope-plus-estimated-cost", mutation: Mutation::KnownSlopePlusEstimatedCost, expected: AssessmentStatus::Rejected },
        FrozenCase { id: "stale-cost-evidence", mutation: Mutation::StaleCostEvidence, expected: AssessmentStatus::Blocked },
        FrozenCase { id: "conflicting-serviceability-evidence", mutation: Mutation::ConflictingServiceabilityEvidence, expected: AssessmentStatus::Blocked },
        FrozenCase { id: "empty-verified-source-reference", mutation: Mutation::EmptyVerifiedSourceReference, expected: AssessmentStatus::Blocked },
        FrozenCase { id: "invalid-operating-hours", mutation: Mutation::InvalidOperatingHours, expected: AssessmentStatus::Blocked },
    ];

    fn baseline_scenario() -> ScenarioProfile {
        ScenarioProfile {
            id: "synthetic-highveld-vegetable-scouting".into(),
            task: TaskKind::FieldScouting,
            required_daily_capacity: 1_000,
            site_slope_permille: 100,
            available_row_clearance_cm: 120,
            available_turning_radius_cm: 150,
            site_roughness_mm: 40,
            gnss_available: true,
            max_upfront_budget_cents: 1_000_000,
            max_annualized_cost_cents: 500_000,
            annual_operating_hours: 100,
            require_local_repair: true,
            max_repair_lead_days: 14,
            evidence: EvidenceDomain::ALL
                .into_iter()
                .map(|domain| {
                    EvidenceRecord::verified(
                        domain,
                        SYNTHETIC_FIXTURE_REFERENCE,
                        SYNTHETIC_FIXTURE_SHA256,
                    )
                })
                .collect(),
        }
    }

    fn baseline_platform() -> PlatformProfile {
        PlatformProfile {
            id: "candidate-modular-ugv".into(),
            class: PlatformClass::ModularGroundRobot,
            supported_tasks: vec![TaskKind::FieldScouting],
            service_unit: ServiceUnit::SquareMetersPerDay,
            max_slope_permille: 300,
            width_cm: 80,
            turning_radius_cm: 100,
            max_roughness_mm: 100,
            requires_gnss: true,
            gnss_denied_capable: false,
            daily_capacity_units: 2_000,
            acquisition_cost_cents: 100_000,
            operating_cost_cents_per_hour: 500,
            supervision_minutes_per_hour: 10,
            maintenance_minutes_per_100_hours: 60,
            useful_life_years: 5,
            local_repair_available: true,
            repair_lead_days: 7,
            bounded_safety_case_verified: true,
            human_override_available: true,
        }
    }

    fn set_state(scenario: &mut ScenarioProfile, domain: EvidenceDomain, state: EvidenceState) {
        let record = scenario.evidence.iter_mut().find(|r| r.domain == domain).unwrap();
        record.state = state;
        record.reference = match state {
            EvidenceState::Missing => None,
            EvidenceState::Stale => Some("fixture://stale-evidence".into()),
            EvidenceState::Conflicting => Some("fixture://conflicting-evidence".into()),
            EvidenceState::Verified | EvidenceState::Estimated => {
                Some(SYNTHETIC_FIXTURE_REFERENCE.into())
            }
        };
        record.artifact_sha256 = if state == EvidenceState::Verified {
            Some(SYNTHETIC_FIXTURE_SHA256.into())
        } else {
            None
        };
    }

    fn instantiate(case: &FrozenCase) -> (ScenarioProfile, PlatformProfile) {
        let mut scenario = baseline_scenario();
        let mut platform = baseline_platform();

        match case.mutation {
            Mutation::Baseline => {}
            Mutation::UnsupportedTask => platform.supported_tasks = vec![TaskKind::ProduceTransport],
            Mutation::IncompatibleServiceUnit => platform.service_unit = ServiceUnit::KilogramsPerDay,
            Mutation::SlopeExceeded => scenario.site_slope_permille = 301,
            Mutation::ClearanceTooNarrow => scenario.available_row_clearance_cm = 79,
            Mutation::TurningRoomTooSmall => scenario.available_turning_radius_cm = 99,
            Mutation::RoughnessExceeded => scenario.site_roughness_mm = 101,
            Mutation::GnssUnavailable => scenario.gnss_available = false,
            Mutation::CapacityShortfall => platform.daily_capacity_units = 999,
            Mutation::UpfrontBudgetExceeded => scenario.max_upfront_budget_cents = 99_999,
            Mutation::AnnualCostExceeded => scenario.max_annualized_cost_cents = 22_000,
            Mutation::LocalRepairUnavailable => platform.local_repair_available = false,
            Mutation::RepairLeadTooLong => platform.repair_lead_days = 15,
            Mutation::SafetyCaseMissing => platform.bounded_safety_case_verified = false,
            Mutation::HumanOverrideMissing => platform.human_override_available = false,
            Mutation::MissingTaskEvidence => set_state(&mut scenario, EvidenceDomain::TaskDefinition, EvidenceState::Missing),
            Mutation::StaleGeometryEvidence => set_state(&mut scenario, EvidenceDomain::SiteGeometry, EvidenceState::Stale),
            Mutation::ConflictingTerrainEvidence => set_state(&mut scenario, EvidenceDomain::TerrainEnvelope, EvidenceState::Conflicting),
            Mutation::MissingNavigationEvidence => set_state(&mut scenario, EvidenceDomain::NavigationEnvelope, EvidenceState::Missing),
            Mutation::EstimatedSafetyEvidence => set_state(&mut scenario, EvidenceDomain::SafetyEnvelope, EvidenceState::Estimated),
            Mutation::EstimatedCostEvidence => set_state(&mut scenario, EvidenceDomain::CostModel, EvidenceState::Estimated),
            Mutation::EstimatedServiceabilityEvidence => set_state(&mut scenario, EvidenceDomain::Serviceability, EvidenceState::Estimated),
            Mutation::EstimatedTerrainEvidence => set_state(&mut scenario, EvidenceDomain::TerrainEnvelope, EvidenceState::Estimated),
            Mutation::MalformedVerifiedDigest => {
                let record = scenario.evidence.iter_mut().find(|r| r.domain == EvidenceDomain::CostModel).unwrap();
                record.artifact_sha256 = Some("bad-digest".into());
            }
            Mutation::DuplicateEvidenceDomain => {
                let duplicate = scenario.evidence[0].clone();
                scenario.evidence.push(duplicate);
            },
            Mutation::MissingEvidenceDomain => scenario.evidence.retain(|r| r.domain != EvidenceDomain::TaskDefinition),
            Mutation::ArithmeticOverflow => {
                platform.operating_cost_cents_per_hour = u64::MAX;
                scenario.annual_operating_hours = 2;
            }
            Mutation::GnssDeniedFallback => {
                scenario.gnss_available = false;
                platform.gnss_denied_capable = true;
            }
            Mutation::ManualCurrentPracticeFeasible => {
                platform.id = "manual-current-practice".into();
                platform.class = PlatformClass::ManualCurrentPractice;
                platform.acquisition_cost_cents = 0;
                platform.operating_cost_cents_per_hour = 100;
                platform.supervision_minutes_per_hour = 60;
                platform.maintenance_minutes_per_100_hours = 0;
                platform.useful_life_years = 1;
                platform.requires_gnss = false;
            }
            Mutation::ExactCapacityBoundary => platform.daily_capacity_units = scenario.required_daily_capacity,
            Mutation::ExactSlopeBoundary => scenario.site_slope_permille = platform.max_slope_permille,
            Mutation::ExactClearanceBoundary => {
                scenario.available_row_clearance_cm = platform.width_cm;
                scenario.available_turning_radius_cm = platform.turning_radius_cm;
            }
            Mutation::ZeroAnnualHours => scenario.annual_operating_hours = 0,
            Mutation::KnownSlopePlusEstimatedCost => {
                scenario.site_slope_permille = platform.max_slope_permille + 1;
                set_state(&mut scenario, EvidenceDomain::CostModel, EvidenceState::Estimated);
            }
            Mutation::StaleCostEvidence => set_state(&mut scenario, EvidenceDomain::CostModel, EvidenceState::Stale),
            Mutation::ConflictingServiceabilityEvidence => set_state(&mut scenario, EvidenceDomain::Serviceability, EvidenceState::Conflicting),
            Mutation::EmptyVerifiedSourceReference => {
                let record = scenario.evidence.iter_mut().find(|r| r.domain == EvidenceDomain::TaskDefinition).unwrap();
                record.reference = Some("   ".into());
            }
            Mutation::InvalidOperatingHours => scenario.annual_operating_hours = MAX_ANNUAL_OPERATING_HOURS + 1,
        }

        (scenario, platform)
    }

    fn status_name(status: AssessmentStatus) -> &'static str {
        match status {
            AssessmentStatus::Feasible => "FEASIBLE",
            AssessmentStatus::Provisional => "PROVISIONAL",
            AssessmentStatus::Rejected => "REJECTED",
            AssessmentStatus::Blocked => "BLOCKED",
        }
    }

    #[test]
    fn frozen_manifest_matches_independent_expected_dispositions() {
        let manifest = include_str!("../fixtures/scenario_manifest.tsv");
        let rows: Vec<_> = manifest
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
            .collect();
        assert_eq!(rows.first().copied(), Some("case_id\texpected_status\trationale"));
        let data_rows = &rows[1..];
        assert_eq!(data_rows.len(), CASES.len(), "manifest/corpus case count drift");

        for (row, case) in data_rows.iter().zip(CASES.iter()) {
            let columns: Vec<_> = row.split('\t').collect();
            assert_eq!(columns.len(), 3, "malformed manifest row: {row}");
            assert_eq!(columns[0], case.id, "case order/identity drift");
            assert_eq!(columns[1], status_name(case.expected), "oracle disposition drift");

            let (scenario, platform) = instantiate(case);
            let result = assess(&scenario, &platform);
            assert_eq!(
                result.status, case.expected,
                "case {} returned {:?}; blockers={:?}; uncertain={:?}",
                case.id, result.status, result.blockers, result.uncertain_domains
            );
        }
    }

    #[test]
    fn unknown_numeric_values_are_not_promoted_over_stale_evidence() {
        let case = CASES.iter().find(|c| c.id == "stale-geometry-evidence").unwrap();
        let (mut scenario, platform) = instantiate(case);
        scenario.site_slope_permille = u32::MAX;
        let result = assess(&scenario, &platform);
        assert_eq!(result.status, AssessmentStatus::Blocked);
        assert!(result.blockers.contains(&BlockerCode::EvidenceStale(EvidenceDomain::SiteGeometry)));
        assert!(!result.blockers.contains(&BlockerCode::SlopeExceeded));
    }

    #[test]
    fn no_robot_remains_a_valid_option_when_robot_exceeds_budget() {
        let mut scenario = baseline_scenario();
        scenario.max_upfront_budget_cents = 0;
        let robot = baseline_platform();
        let mut manual = baseline_platform();
        manual.id = "manual-current-practice".into();
        manual.class = PlatformClass::ManualCurrentPractice;
        manual.acquisition_cost_cents = 0;
        manual.operating_cost_cents_per_hour = 100;
        manual.supervision_minutes_per_hour = 60;
        manual.maintenance_minutes_per_100_hours = 0;
        manual.useful_life_years = 1;
        manual.requires_gnss = false;

        let robot_result = assess(&scenario, &robot);
        let manual_result = assess(&scenario, &manual);
        assert_eq!(robot_result.status, AssessmentStatus::Rejected);
        assert!(robot_result.blockers.contains(&BlockerCode::UpfrontBudgetExceeded));
        assert_eq!(manual_result.status, AssessmentStatus::Feasible);
    }

    #[test]
    fn frontier_preserves_tradeoffs_and_excludes_nonqualified_candidates() {
        let metrics = |cost, supervision, maintenance, repair| PerformanceMetrics {
            annualized_total_cost_cents: cost,
            annual_supervision_minutes: supervision,
            annual_maintenance_minutes: maintenance,
            repair_lead_days: repair,
        };
        let assessment = |id: &str, status, m| Assessment {
            candidate_id: id.into(),
            status,
            blockers: Vec::new(),
            uncertain_domains: Vec::new(),
            metrics: Some(m),
        };
        let candidates = vec![
            assessment("low-cost-high-supervision", AssessmentStatus::Feasible, metrics(200, 1_000, 100, 5)),
            assessment("balanced-robot", AssessmentStatus::Feasible, metrics(500, 400, 50, 2)),
            assessment("dominated-robot", AssessmentStatus::Feasible, metrics(800, 900, 80, 4)),
            assessment("uncertain-option", AssessmentStatus::Provisional, metrics(100, 1, 1, 1)),
            assessment("blocked-option", AssessmentStatus::Blocked, metrics(10, 1, 1, 1)),
        ];
        assert_eq!(
            pareto_frontier(&candidates),
            vec![
                "balanced-robot".to_string(),
                "low-cost-high-supervision".to_string()
            ]
        );
    }

    #[test]
    fn annualized_cost_rounds_capital_up_and_keeps_units_explicit() {
        let scenario = baseline_scenario();
        let platform = baseline_platform();
        let result = assess(&scenario, &platform);
        let metrics = result.metrics.unwrap();
        // ceil(100000 / 5) + (500 * 100) = 20000 + 50000 cents.
        assert_eq!(metrics.annualized_total_cost_cents, 70_000);
        assert_eq!(metrics.annual_supervision_minutes, 1_000);
        assert_eq!(metrics.annual_maintenance_minutes, 60);
    }

    #[test]
    fn same_metrics_yield_stable_sorted_frontier() {
        let m = PerformanceMetrics {
            annualized_total_cost_cents: 100,
            annual_supervision_minutes: 100,
            annual_maintenance_minutes: 10,
            repair_lead_days: 2,
        };
        let make = |id: &str| Assessment {
            candidate_id: id.into(),
            status: AssessmentStatus::Feasible,
            blockers: Vec::new(),
            uncertain_domains: Vec::new(),
            metrics: Some(m.clone()),
        };
        assert_eq!(
            pareto_frontier(&[make("zeta"), make("alpha")]),
            vec!["alpha".to_string(), "zeta".to_string()]
        );
    }
}
