// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Conservative cumulative mission-resource budgeting (SX-033).
//!
//! Resource quantities are integer values in the canonical unit named by the
//! ResourceKind enum. The evaluator uses the lower supply bound and upper
//! demand bounds, then adds an explicit reserve for required-task demand.
//! Evidence intervals and maturity labels are caller-declared claims. This is
//! not a physical resource model, flight-readiness determination, or safety
//! authority. Peak-power scheduling and coupled resource physics are out of scope.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::suit_interop::{
    SuitInteropReportStatus, SuitOperatingEnvironment,
};
use crate::suit_mission_configuration::{
    MissionConfiguration, MissionConfigurationReport,
};
use crate::suit_mission_tasks::{
    evaluate_mission_tasks, MissionTaskError, MissionTaskPlanReport,
    MissionTaskPlanStatus, MissionTaskReceipt, MissionTaskRequirement,
};
use crate::suit_profile::SuitEvidenceLevel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum MissionResourceKind {
    OxygenMassMicrograms,
    WaterVolumeMicroliters,
    ConsumableMassMicrograms,
    ElectricalEnergyMicrowattHours,
    DataVolumeBytes,
    CrewTimeMilliseconds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceQuantityInterval {
    pub lower: u64,
    pub nominal: u64,
    pub upper: u64,
}

impl ResourceQuantityInterval {
    pub fn is_valid(&self) -> bool {
        self.lower <= self.nominal && self.nominal <= self.upper
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissionResourcePool {
    pub pool_id: String,
    pub resource_kind: MissionResourceKind,
    pub provider_component_id: String,
    pub provider_hardware_revision: String,
    pub available: ResourceQuantityInterval,
    pub evidence_level: SuitEvidenceLevel,
    /// Artifact URI, test record, requirement ID, or digest; not authenticated here.
    pub evidence_reference: String,
    pub applicable_environments: BTreeSet<SuitOperatingEnvironment>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissionResourceDemand {
    pub demand_id: String,
    pub task_id: String,
    pub pool_id: String,
    pub resource_kind: MissionResourceKind,
    pub quantity: ResourceQuantityInterval,
    pub evidence_level: SuitEvidenceLevel,
    pub evidence_reference: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissionResourceBudgetPlan {
    pub schema_version: u32,
    pub mission_id: String,
    pub configuration_revision: String,
    /// Must equal SX-031's canonical mission configuration digest.
    pub expected_configuration_digest_sha256: String,
    /// Must equal SX-032's canonical task plan digest.
    pub expected_task_plan_digest_sha256: String,
    pub minimum_evidence_level: SuitEvidenceLevel,
    /// Reserve against required-task worst-case demand, in basis points.
    /// 10_000 = 100%; greater than 100_000 is rejected as likely misconfiguration.
    pub reserve_basis_points: u32,
    pub pools: Vec<MissionResourcePool>,
    pub demands: Vec<MissionResourceDemand>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MissionResourceFindingCode {
    MissingProviderComponent,
    ProviderHardwareRevisionMismatch,
    ProviderNotReachable,
    EnvironmentScopeMismatch,
    SupplyEvidenceBelowPolicy,
    DemandEvidenceBelowPolicy,
    RequiredDemandExceedsSupply,
    OptionalDemandExceedsRemainingCapacity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MissionResourceFindingSeverity {
    Degradation,
    Blocker,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissionResourceFinding {
    pub pool_id: String,
    pub task_id: Option<String>,
    pub demand_id: Option<String>,
    pub code: MissionResourceFindingCode,
    pub severity: MissionResourceFindingSeverity,
    pub observed_evidence_level: Option<SuitEvidenceLevel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MissionResourcePoolStatus {
    MeetsDeclaredBudget,
    RequiredDemandExceedsSupply,
    OptionalDemandExceedsRemainingCapacity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissionResourcePoolEvaluation {
    pub pool_id: String,
    pub resource_kind: MissionResourceKind,
    pub provider_component_id: String,
    pub provider_hardware_revision: String,
    pub available: ResourceQuantityInterval,
    pub required_demand_upper: u64,
    pub optional_demand_upper: u64,
    pub reserve_quantity: u64,
    pub required_capacity: u64,
    pub required_margin: Option<u64>,
    pub required_shortfall: u64,
    pub total_planned_capacity: u64,
    pub remaining_after_all_demand: Option<u64>,
    pub optional_shortfall: u64,
    pub status: MissionResourcePoolStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MissionResourceBudgetStatus {
    Blocked,
    Degraded,
    MeetsDeclaredResourcePolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissionResourceBudgetReport {
    pub schema_version: u32,
    pub mission_id: String,
    pub configuration_revision: String,
    pub configuration_digest_sha256: String,
    pub task_plan_digest_sha256: String,
    pub resource_plan_digest_sha256: String,
    pub mission_task_plan_status: MissionTaskPlanStatus,
    pub pools: Vec<MissionResourcePoolEvaluation>,
    pub findings: Vec<MissionResourceFinding>,
    pub status: MissionResourceBudgetStatus,
    pub required_pool_count: usize,
    pub required_pool_count_within_budget: usize,
    pub required_pool_count_over_budget: usize,
    pub optional_pool_count_over_budget: usize,
    pub required_demand_count: usize,
    pub optional_demand_count: usize,
    /// Always false: arithmetic over declarations is not physical qualification.
    pub represents_physical_qualification: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissionResourceBudgetReceipt {
    pub configuration_digest_sha256: String,
    pub task_plan_digest_sha256: String,
    pub task_receipt_digest_sha256: String,
    pub resource_plan_digest_sha256: String,
    pub report_digest_sha256: String,
    /// Deterministic change-evidence; not a signature or authentication claim.
    pub receipt_digest_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissionResourceBudgetError {
    UnsupportedSchemaVersion,
    InvalidPlanIdentity,
    InvalidPinnedDigest,
    InvalidReservePolicy,
    EmptyResourcePlan,
    InvalidPool,
    DuplicatePoolId,
    DuplicateProviderResourcePool,
    InvalidDemand,
    DuplicateDemandId,
    MissingReferencedTask,
    MissingReferencedPool,
    ResourceKindMismatch,
    UnusedPool,
    InvalidQuantityInterval,
    QuantityOverflow,
    ReserveOverflow,
    TaskEvaluationFailed(MissionTaskError),
    TaskReceiptMismatch,
    ExpectedConfigurationDigestMismatch,
    ExpectedTaskPlanDigestMismatch,
    MissionIdentityMismatch,
}

#[derive(Debug, Clone, Copy, Default)]
struct PoolDemandTotals {
    required_upper: u64,
    optional_upper: u64,
    required_count: usize,
    optional_count: usize,
}

/// Evaluate resource pools and demand claims against freshly re-evaluated SX-032
/// task requirements. The supplied task receipt must match the exact current
/// configuration and task plan, so budget findings cannot silently reuse a
/// receipt from another head/revision.
pub fn evaluate_mission_resource_budget(
    configuration: &MissionConfiguration,
    tasks: &[MissionTaskRequirement],
    task_receipt: &MissionTaskReceipt,
    plan: &MissionResourceBudgetPlan,
) -> Result<
    (MissionResourceBudgetReport, MissionResourceBudgetReceipt),
    MissionResourceBudgetError,
> {
    validate_plan_structure(plan)?;

    let (task_report, recomputed_task_receipt) =
        evaluate_mission_tasks(configuration, tasks)
            .map_err(MissionResourceBudgetError::TaskEvaluationFailed)?;
    if &recomputed_task_receipt != task_receipt {
        return Err(MissionResourceBudgetError::TaskReceiptMismatch);
    }
    if plan.expected_configuration_digest_sha256
        != task_report.configuration_digest_sha256
    {
        return Err(MissionResourceBudgetError::ExpectedConfigurationDigestMismatch);
    }
    if plan.expected_task_plan_digest_sha256 != task_report.task_plan_digest_sha256 {
        return Err(MissionResourceBudgetError::ExpectedTaskPlanDigestMismatch);
    }
    if plan.mission_id != configuration.mission_id
        || plan.configuration_revision != configuration.configuration_revision
        || plan.mission_id != task_report.mission_id
        || plan.configuration_revision != task_report.configuration_revision
    {
        return Err(MissionResourceBudgetError::MissionIdentityMismatch);
    }

    validate_plan_references(plan, &task_report)?;
    let resource_plan_digest_sha256 = digest_resource_plan(plan);
    let task_by_id: BTreeMap<&str, _> = task_report
        .task_evaluations
        .iter()
        .map(|task| (task.task_id.as_str(), task))
        .collect();
    let pool_by_id: BTreeMap<&str, _> = plan
        .pools
        .iter()
        .map(|pool| (pool.pool_id.as_str(), pool))
        .collect();
    let component_by_id: BTreeMap<&str, _> = configuration
        .components
        .iter()
        .map(|component| (component.component_id.as_str(), component))
        .collect();

    let mut totals_by_pool: BTreeMap<String, PoolDemandTotals> = BTreeMap::new();
    for demand in &plan.demands {
        let task = task_by_id[demand.task_id.as_str()];
        let totals = totals_by_pool.entry(demand.pool_id.clone()).or_default();
        if task.required {
            totals.required_upper = totals.required_upper
                .checked_add(demand.quantity.upper)
                .ok_or(MissionResourceBudgetError::QuantityOverflow)?;
            totals.required_count += 1;
        } else {
            totals.optional_upper = totals.optional_upper
                .checked_add(demand.quantity.upper)
                .ok_or(MissionResourceBudgetError::QuantityOverflow)?;
            totals.optional_count += 1;
        }
    }

    let mut pools = Vec::with_capacity(plan.pools.len());
    let mut findings = Vec::new();
    let mut required_pool_count = 0;
    let mut required_pool_count_within_budget = 0;
    let mut required_pool_count_over_budget = 0;
    let mut optional_pool_count_over_budget = 0;

    let mut ordered_pools: Vec<&MissionResourcePool> = plan.pools.iter().collect();
    ordered_pools.sort_by(|left, right| left.pool_id.cmp(&right.pool_id));
    for pool in ordered_pools {
        let totals = totals_by_pool[&pool.pool_id];
        let reserve_quantity = compute_reserve(totals.required_upper, plan.reserve_basis_points)?;
        let required_capacity = totals.required_upper
            .checked_add(reserve_quantity)
            .ok_or(MissionResourceBudgetError::QuantityOverflow)?;
        let total_planned_capacity = required_capacity
            .checked_add(totals.optional_upper)
            .ok_or(MissionResourceBudgetError::QuantityOverflow)?;
        let required_margin = pool.available.lower.checked_sub(required_capacity);
        let required_shortfall = required_capacity.saturating_sub(pool.available.lower);
        let remaining_after_all_demand = pool.available.lower.checked_sub(total_planned_capacity);
        let optional_shortfall = total_planned_capacity.saturating_sub(pool.available.lower);
        let status = if required_shortfall > 0 {
            required_pool_count_over_budget += 1;
            MissionResourcePoolStatus::RequiredDemandExceedsSupply
        } else if optional_shortfall > 0 {
            optional_pool_count_over_budget += 1;
            MissionResourcePoolStatus::OptionalDemandExceedsRemainingCapacity
        } else {
            MissionResourcePoolStatus::MeetsDeclaredBudget
        };
        if totals.required_count > 0 {
            required_pool_count += 1;
            if required_shortfall == 0 {
                required_pool_count_within_budget += 1;
            }
        }
        if required_shortfall > 0 {
            findings.push(MissionResourceFinding {
                pool_id: pool.pool_id.clone(),
                task_id: None,
                demand_id: None,
                code: MissionResourceFindingCode::RequiredDemandExceedsSupply,
                severity: MissionResourceFindingSeverity::Blocker,
                observed_evidence_level: None,
            });
        } else if optional_shortfall > 0 {
            findings.push(MissionResourceFinding {
                pool_id: pool.pool_id.clone(),
                task_id: None,
                demand_id: None,
                code: MissionResourceFindingCode::OptionalDemandExceedsRemainingCapacity,
                severity: MissionResourceFindingSeverity::Degradation,
                observed_evidence_level: None,
            });
        }
        pools.push(MissionResourcePoolEvaluation {
            pool_id: pool.pool_id.clone(),
            resource_kind: pool.resource_kind,
            provider_component_id: pool.provider_component_id.clone(),
            provider_hardware_revision: pool.provider_hardware_revision.clone(),
            available: pool.available,
            required_demand_upper: totals.required_upper,
            optional_demand_upper: totals.optional_upper,
            reserve_quantity,
            required_capacity,
            required_margin,
            required_shortfall,
            total_planned_capacity,
            remaining_after_all_demand,
            optional_shortfall,
            status,
        });
    }

    let mut reachability_cache: BTreeMap<
        BTreeSet<SuitOperatingEnvironment>,
        BTreeSet<String>,
    > = BTreeMap::new();
    let mission_report = &task_report.mission_configuration_report;
    let mut ordered_demands: Vec<&MissionResourceDemand> = plan.demands.iter().collect();
    ordered_demands.sort_by(|left, right| left.demand_id.cmp(&right.demand_id));
    for demand in ordered_demands {
        let task = task_by_id[demand.task_id.as_str()];
        let pool = pool_by_id[demand.pool_id.as_str()];
        let severity = if task.required {
            MissionResourceFindingSeverity::Blocker
        } else {
            MissionResourceFindingSeverity::Degradation
        };
        let provider = component_by_id.get(pool.provider_component_id.as_str()).copied();
        match provider {
            None => findings.push(MissionResourceFinding {
                pool_id: pool.pool_id.clone(),
                task_id: Some(task.task_id.clone()),
                demand_id: Some(demand.demand_id.clone()),
                code: MissionResourceFindingCode::MissingProviderComponent,
                severity,
                observed_evidence_level: None,
            }),
            Some(component) => {
                if component.hardware_revision != pool.provider_hardware_revision {
                    findings.push(MissionResourceFinding {
                        pool_id: pool.pool_id.clone(),
                        task_id: Some(task.task_id.clone()),
                        demand_id: Some(demand.demand_id.clone()),
                        code: MissionResourceFindingCode::ProviderHardwareRevisionMismatch,
                        severity,
                        observed_evidence_level: None,
                    });
                }

                if !task.required_environments.is_subset(&pool.applicable_environments) {
                    findings.push(MissionResourceFinding {
                        pool_id: pool.pool_id.clone(),
                        task_id: Some(task.task_id.clone()),
                        demand_id: Some(demand.demand_id.clone()),
                        code: MissionResourceFindingCode::EnvironmentScopeMismatch,
                        severity,
                        observed_evidence_level: None,
                    });
                }

                if !reachability_cache.contains_key(&task.required_environments) {
                    let reachable = reachable_for_environment(
                        &configuration.suit_component_id,
                        &configuration.components,
                        mission_report,
                        &task.required_environments,
                    );
                    reachability_cache.insert(task.required_environments.clone(), reachable);
                }
                let reachable = reachability_cache
                    .get(&task.required_environments)
                    .expect("reachability cache entry was just inserted");
                if !reachable.contains(&pool.provider_component_id) {
                    findings.push(MissionResourceFinding {
                        pool_id: pool.pool_id.clone(),
                        task_id: Some(task.task_id.clone()),
                        demand_id: Some(demand.demand_id.clone()),
                        code: MissionResourceFindingCode::ProviderNotReachable,
                        severity,
                        observed_evidence_level: None,
                    });
                }
            }
        }

        if pool.evidence_level < plan.minimum_evidence_level {
            findings.push(MissionResourceFinding {
                pool_id: pool.pool_id.clone(),
                task_id: Some(task.task_id.clone()),
                demand_id: Some(demand.demand_id.clone()),
                code: MissionResourceFindingCode::SupplyEvidenceBelowPolicy,
                severity,
                observed_evidence_level: Some(pool.evidence_level),
            });
        }
        if demand.evidence_level < plan.minimum_evidence_level {
            findings.push(MissionResourceFinding {
                pool_id: pool.pool_id.clone(),
                task_id: Some(task.task_id.clone()),
                demand_id: Some(demand.demand_id.clone()),
                code: MissionResourceFindingCode::DemandEvidenceBelowPolicy,
                severity,
                observed_evidence_level: Some(demand.evidence_level),
            });
        }
    }

    findings.sort_by(|left, right| {
        left.pool_id
            .cmp(&right.pool_id)
            .then(left.task_id.cmp(&right.task_id))
            .then(left.demand_id.cmp(&right.demand_id))
            .then(left.code.cmp(&right.code))
            .then(left.severity.cmp(&right.severity))
            .then(left.observed_evidence_level.cmp(&right.observed_evidence_level))
    });

    let blocker_count = findings.iter()
        .filter(|finding| finding.severity == MissionResourceFindingSeverity::Blocker)
        .count();
    let degradation_count = findings.iter()
        .filter(|finding| finding.severity == MissionResourceFindingSeverity::Degradation)
        .count();
    let status = if task_report.status == MissionTaskPlanStatus::Blocked
        || blocker_count > 0
        || required_pool_count_over_budget > 0
    {
        MissionResourceBudgetStatus::Blocked
    } else if task_report.status == MissionTaskPlanStatus::Degraded
        || degradation_count > 0
        || optional_pool_count_over_budget > 0
    {
        MissionResourceBudgetStatus::Degraded
    } else {
        MissionResourceBudgetStatus::MeetsDeclaredResourcePolicy
    };

    let report = MissionResourceBudgetReport {
        schema_version: 1,
        mission_id: plan.mission_id.clone(),
        configuration_revision: plan.configuration_revision.clone(),
        configuration_digest_sha256: plan.expected_configuration_digest_sha256.clone(),
        task_plan_digest_sha256: plan.expected_task_plan_digest_sha256.clone(),
        resource_plan_digest_sha256: resource_plan_digest_sha256.clone(),
        mission_task_plan_status: task_report.status,
        pools,
        findings,
        status,
        required_pool_count,
        required_pool_count_within_budget,
        required_pool_count_over_budget,
        optional_pool_count_over_budget,
        required_demand_count: plan.demands.iter()
            .filter(|demand| task_by_id[demand.task_id.as_str()].required)
            .count(),
        optional_demand_count: plan.demands.iter()
            .filter(|demand| !task_by_id[demand.task_id.as_str()].required)
            .count(),
        represents_physical_qualification: false,
    };

    let report_digest_sha256 = digest_resource_report(&report);
    let mut receipt = MissionResourceBudgetReceipt {
        configuration_digest_sha256: plan.expected_configuration_digest_sha256.clone(),
        task_plan_digest_sha256: plan.expected_task_plan_digest_sha256.clone(),
        task_receipt_digest_sha256: digest_task_receipt(task_receipt),
        resource_plan_digest_sha256,
        report_digest_sha256,
        receipt_digest_sha256: String::new(),
    };
    receipt.receipt_digest_sha256 = digest_resource_receipt(&receipt);
    Ok((report, receipt))
}

/// Re-evaluate configuration, task plan, and budget from exact inputs; all
/// supplied receipts and typed reports must match the fresh deterministic result.
pub fn verify_mission_resource_budget_receipt(
    configuration: &MissionConfiguration,
    tasks: &[MissionTaskRequirement],
    task_receipt: &MissionTaskReceipt,
    plan: &MissionResourceBudgetPlan,
    expected_report: &MissionResourceBudgetReport,
    expected_receipt: &MissionResourceBudgetReceipt,
) -> bool {
    match evaluate_mission_resource_budget(configuration, tasks, task_receipt, plan) {
        Ok((report, receipt)) => report == *expected_report && receipt == *expected_receipt,
        Err(_) => false,
    }
}

fn validate_plan_structure(
    plan: &MissionResourceBudgetPlan,
) -> Result<(), MissionResourceBudgetError> {
    if plan.schema_version != 1 {
        return Err(MissionResourceBudgetError::UnsupportedSchemaVersion);
    }
    if !clean_identifier(&plan.mission_id)
        || !clean_identifier(&plan.configuration_revision)
    {
        return Err(MissionResourceBudgetError::InvalidPlanIdentity);
    }
    if !valid_sha256(&plan.expected_configuration_digest_sha256)
        || !valid_sha256(&plan.expected_task_plan_digest_sha256)
    {
        return Err(MissionResourceBudgetError::InvalidPinnedDigest);
    }
    if plan.reserve_basis_points > 100_000 {
        return Err(MissionResourceBudgetError::InvalidReservePolicy);
    }
    if plan.pools.is_empty() || plan.demands.is_empty() {
        return Err(MissionResourceBudgetError::EmptyResourcePlan);
    }

    let mut pool_ids = BTreeSet::new();
    let mut provider_resources = BTreeSet::new();
    for pool in &plan.pools {
        if !clean_identifier(&pool.pool_id)
            || !clean_identifier(&pool.provider_component_id)
            || !clean_identifier(&pool.provider_hardware_revision)
            || !clean_identifier(&pool.evidence_reference)
            || pool.applicable_environments.is_empty()
        {
            return Err(MissionResourceBudgetError::InvalidPool);
        }
        if !pool.available.is_valid() {
            return Err(MissionResourceBudgetError::InvalidQuantityInterval);
        }
        if !pool_ids.insert(pool.pool_id.as_str()) {
            return Err(MissionResourceBudgetError::DuplicatePoolId);
        }
        if !provider_resources.insert((
            pool.provider_component_id.as_str(),
            pool.resource_kind,
        )) {
            return Err(MissionResourceBudgetError::DuplicateProviderResourcePool);
        }
    }

    let mut demand_ids = BTreeSet::new();
    let mut used_pools = BTreeSet::new();
    for demand in &plan.demands {
        if !clean_identifier(&demand.demand_id)
            || !clean_identifier(&demand.task_id)
            || !clean_identifier(&demand.pool_id)
            || !clean_identifier(&demand.evidence_reference)
            || demand.quantity.upper == 0
        {
            return Err(MissionResourceBudgetError::InvalidDemand);
        }
        if !demand.quantity.is_valid() {
            return Err(MissionResourceBudgetError::InvalidQuantityInterval);
        }
        if !demand_ids.insert(demand.demand_id.as_str()) {
            return Err(MissionResourceBudgetError::DuplicateDemandId);
        }
        used_pools.insert(demand.pool_id.as_str());
    }
    if plan.pools.iter().any(|pool| !used_pools.contains(pool.pool_id.as_str())) {
        return Err(MissionResourceBudgetError::UnusedPool);
    }
    Ok(())
}

fn validate_plan_references(
    plan: &MissionResourceBudgetPlan,
    task_report: &MissionTaskPlanReport,
) -> Result<(), MissionResourceBudgetError> {
    let task_ids: BTreeSet<&str> = task_report
        .task_evaluations
        .iter()
        .map(|task| task.task_id.as_str())
        .collect();
    let pools: BTreeMap<&str, &MissionResourcePool> = plan
        .pools
        .iter()
        .map(|pool| (pool.pool_id.as_str(), pool))
        .collect();
    for demand in &plan.demands {
        if !task_ids.contains(demand.task_id.as_str()) {
            return Err(MissionResourceBudgetError::MissingReferencedTask);
        }
        let pool = pools
            .get(demand.pool_id.as_str())
            .ok_or(MissionResourceBudgetError::MissingReferencedPool)?;
        if pool.resource_kind != demand.resource_kind {
            return Err(MissionResourceBudgetError::ResourceKindMismatch);
        }
    }
    Ok(())
}

fn compute_reserve(
    required_demand_upper: u64,
    reserve_basis_points: u32,
) -> Result<u64, MissionResourceBudgetError> {
    let product = u128::from(required_demand_upper)
        .checked_mul(u128::from(reserve_basis_points))
        .ok_or(MissionResourceBudgetError::ReserveOverflow)?;
    let rounded = product
        .checked_add(9_999)
        .ok_or(MissionResourceBudgetError::ReserveOverflow)?
        / 10_000;
    u64::try_from(rounded).map_err(|_| MissionResourceBudgetError::ReserveOverflow)
}

fn reachable_for_environment(
    suit_component_id: &str,
    components: &[crate::suit_mission_configuration::MissionComponent],
    report: &MissionConfigurationReport,
    environments: &BTreeSet<SuitOperatingEnvironment>,
) -> BTreeSet<String> {
    let mut adjacency: BTreeMap<String, BTreeSet<String>> = components
        .iter()
        .map(|component| (component.component_id.clone(), BTreeSet::new()))
        .collect();
    for connection in &report.connection_evaluations {
        if connection.interoperability.status
            != SuitInteropReportStatus::MeetsDeclaredCompatibilityPolicy
            || !environments.is_subset(
                &connection.interoperability.policy.required_environments,
            )
        {
            continue;
        }
        adjacency
            .entry(connection.left_component_id.clone())
            .or_default()
            .insert(connection.right_component_id.clone());
        adjacency
            .entry(connection.right_component_id.clone())
            .or_default()
            .insert(connection.left_component_id.clone());
    }
    let mut reached = BTreeSet::new();
    let mut pending = VecDeque::from([suit_component_id.to_string()]);
    while let Some(component_id) = pending.pop_front() {
        if !reached.insert(component_id.clone()) {
            continue;
        }
        if let Some(neighbors) = adjacency.get(&component_id) {
            pending.extend(neighbors.iter().cloned());
        }
    }
    reached
}

fn digest_resource_plan(plan: &MissionResourceBudgetPlan) -> String {
    let mut writer = HashWriter::new("symthaea.sx033.resource-plan.v1");
    writer.u64(plan.schema_version as u64);
    writer.string(&plan.mission_id);
    writer.string(&plan.configuration_revision);
    writer.string(&plan.expected_configuration_digest_sha256);
    writer.string(&plan.expected_task_plan_digest_sha256);
    writer.string(evidence_tag(plan.minimum_evidence_level));
    writer.u64(u64::from(plan.reserve_basis_points));

    let mut pools: Vec<&MissionResourcePool> = plan.pools.iter().collect();
    pools.sort_by(|left, right| left.pool_id.cmp(&right.pool_id));
    writer.usize(pools.len());
    for pool in pools {
        writer.string(&pool.pool_id);
        writer.string(resource_kind_tag(pool.resource_kind));
        writer.string(&pool.provider_component_id);
        writer.string(&pool.provider_hardware_revision);
        write_interval(&mut writer, pool.available);
        writer.string(evidence_tag(pool.evidence_level));
        writer.string(&pool.evidence_reference);
        writer.usize(pool.applicable_environments.len());
        for environment in &pool.applicable_environments {
            writer.string(environment_tag(*environment));
        }
    }

    let mut demands: Vec<&MissionResourceDemand> = plan.demands.iter().collect();
    demands.sort_by(|left, right| left.demand_id.cmp(&right.demand_id));
    writer.usize(demands.len());
    for demand in demands {
        writer.string(&demand.demand_id);
        writer.string(&demand.task_id);
        writer.string(&demand.pool_id);
        writer.string(resource_kind_tag(demand.resource_kind));
        write_interval(&mut writer, demand.quantity);
        writer.string(evidence_tag(demand.evidence_level));
        writer.string(&demand.evidence_reference);
    }
    writer.finish()
}

fn digest_resource_report(report: &MissionResourceBudgetReport) -> String {
    let mut writer = HashWriter::new("symthaea.sx033.resource-report.v1");
    writer.u64(report.schema_version as u64);
    writer.string(&report.mission_id);
    writer.string(&report.configuration_revision);
    writer.string(&report.configuration_digest_sha256);
    writer.string(&report.task_plan_digest_sha256);
    writer.string(&report.resource_plan_digest_sha256);
    writer.string(task_status_tag(report.mission_task_plan_status));
    writer.string(resource_budget_status_tag(report.status));
    writer.boolean(report.represents_physical_qualification);

    writer.usize(report.pools.len());
    for pool in &report.pools {
        writer.string(&pool.pool_id);
        writer.string(resource_kind_tag(pool.resource_kind));
        writer.string(&pool.provider_component_id);
        writer.string(&pool.provider_hardware_revision);
        write_interval(&mut writer, pool.available);
        writer.u64(pool.required_demand_upper);
        writer.u64(pool.optional_demand_upper);
        writer.u64(pool.reserve_quantity);
        writer.u64(pool.required_capacity);
        writer.option_u64(pool.required_margin);
        writer.u64(pool.required_shortfall);
        writer.u64(pool.total_planned_capacity);
        writer.option_u64(pool.remaining_after_all_demand);
        writer.u64(pool.optional_shortfall);
        writer.string(pool_status_tag(pool.status));
    }
    writer.usize(report.findings.len());
    for finding in &report.findings {
        writer.string(&finding.pool_id);
        writer.option_string(finding.task_id.as_deref());
        writer.option_string(finding.demand_id.as_deref());
        writer.string(finding_code_tag(finding.code));
        writer.string(finding_severity_tag(finding.severity));
        match finding.observed_evidence_level {
            Some(level) => {
                writer.boolean(true);
                writer.string(evidence_tag(level));
            }
            None => writer.boolean(false),
        }
    }
    writer.usize(report.required_pool_count);
    writer.usize(report.required_pool_count_within_budget);
    writer.usize(report.required_pool_count_over_budget);
    writer.usize(report.optional_pool_count_over_budget);
    writer.usize(report.required_demand_count);
    writer.usize(report.optional_demand_count);
    writer.finish()
}

fn digest_task_receipt(receipt: &MissionTaskReceipt) -> String {
    let mut writer = HashWriter::new("symthaea.sx033.bound-task-receipt.v1");
    writer.string(&receipt.configuration_digest_sha256);
    writer.string(&receipt.task_plan_digest_sha256);
    writer.string(&receipt.mission_configuration_report_digest_sha256);
    writer.string(&receipt.task_report_digest_sha256);
    writer.string(&receipt.receipt_digest_sha256);
    writer.finish()
}

fn digest_resource_receipt(receipt: &MissionResourceBudgetReceipt) -> String {
    let mut writer = HashWriter::new("symthaea.sx033.resource-receipt.v1");
    writer.string(&receipt.configuration_digest_sha256);
    writer.string(&receipt.task_plan_digest_sha256);
    writer.string(&receipt.task_receipt_digest_sha256);
    writer.string(&receipt.resource_plan_digest_sha256);
    writer.string(&receipt.report_digest_sha256);
    writer.finish()
}

fn write_interval(writer: &mut HashWriter, interval: ResourceQuantityInterval) {
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

    fn option_string(&mut self, value: Option<&str>) {
        match value {
            Some(value) => {
                self.boolean(true);
                self.string(value);
            }
            None => self.boolean(false),
        }
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

fn resource_kind_tag(value: MissionResourceKind) -> &'static str {
    match value {
        MissionResourceKind::OxygenMassMicrograms => "oxygen_mass_micrograms",
        MissionResourceKind::WaterVolumeMicroliters => "water_volume_microliters",
        MissionResourceKind::ConsumableMassMicrograms => "consumable_mass_micrograms",
        MissionResourceKind::ElectricalEnergyMicrowattHours => "electrical_energy_microwatt_hours",
        MissionResourceKind::DataVolumeBytes => "data_volume_bytes",
        MissionResourceKind::CrewTimeMilliseconds => "crew_time_milliseconds",
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

fn environment_tag(value: SuitOperatingEnvironment) -> &'static str {
    match value {
        SuitOperatingEnvironment::MicrogravityVacuum => "microgravity_vacuum",
        SuitOperatingEnvironment::LunarSunlitSurface => "lunar_sunlit_surface",
        SuitOperatingEnvironment::LunarShadowOrPolarRegion => "lunar_shadow_or_polar_region",
        SuitOperatingEnvironment::MarsSurface => "mars_surface",
        SuitOperatingEnvironment::PressurizedCabin => "pressurized_cabin",
        SuitOperatingEnvironment::ThermalVacuumGroundTest => "thermal_vacuum_ground_test",
        SuitOperatingEnvironment::GroundLaboratory => "ground_laboratory",
    }
}

fn task_status_tag(value: MissionTaskPlanStatus) -> &'static str {
    match value {
        MissionTaskPlanStatus::Blocked => "blocked",
        MissionTaskPlanStatus::Degraded => "degraded",
        MissionTaskPlanStatus::MeetsDeclaredTaskPolicy => "meets_declared_task_policy",
    }
}

fn resource_budget_status_tag(value: MissionResourceBudgetStatus) -> &'static str {
    match value {
        MissionResourceBudgetStatus::Blocked => "blocked",
        MissionResourceBudgetStatus::Degraded => "degraded",
        MissionResourceBudgetStatus::MeetsDeclaredResourcePolicy => {
            "meets_declared_resource_policy"
        }
    }
}

fn pool_status_tag(value: MissionResourcePoolStatus) -> &'static str {
    match value {
        MissionResourcePoolStatus::MeetsDeclaredBudget => "meets_declared_budget",
        MissionResourcePoolStatus::RequiredDemandExceedsSupply => "required_demand_exceeds_supply",
        MissionResourcePoolStatus::OptionalDemandExceedsRemainingCapacity => {
            "optional_demand_exceeds_remaining_capacity"
        }
    }
}

fn finding_code_tag(value: MissionResourceFindingCode) -> &'static str {
    match value {
        MissionResourceFindingCode::MissingProviderComponent => "missing_provider_component",
        MissionResourceFindingCode::ProviderHardwareRevisionMismatch => {
            "provider_hardware_revision_mismatch"
        }
        MissionResourceFindingCode::ProviderNotReachable => "provider_not_reachable",
        MissionResourceFindingCode::EnvironmentScopeMismatch => "environment_scope_mismatch",
        MissionResourceFindingCode::SupplyEvidenceBelowPolicy => "supply_evidence_below_policy",
        MissionResourceFindingCode::DemandEvidenceBelowPolicy => "demand_evidence_below_policy",
        MissionResourceFindingCode::RequiredDemandExceedsSupply => "required_demand_exceeds_supply",
        MissionResourceFindingCode::OptionalDemandExceedsRemainingCapacity => {
            "optional_demand_exceeds_remaining_capacity"
        }
    }
}

fn finding_severity_tag(value: MissionResourceFindingSeverity) -> &'static str {
    match value {
        MissionResourceFindingSeverity::Degradation => "degradation",
        MissionResourceFindingSeverity::Blocker => "blocker",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suit_interop::{
        SuitInterfaceEndpoint, SuitInterfaceKind, SuitInterfaceManifest, SuitInterfaceRole,
    };
    use crate::suit_mission_configuration::{
        MissionComponent, MissionComponentKind, MissionConnection,
    };
    use crate::suit_mission_tasks::MissionTaskRequirement;
    use crate::suit_profile::{
        SuitCapability, SuitCapabilityClaim, SuitMissionProfile, SuitVariantDeclaration,
    };

    fn fixture() -> (
        MissionConfiguration,
        Vec<MissionTaskRequirement>,
        MissionTaskReceipt,
        MissionResourceBudgetPlan,
    ) {
        let variant_id = "synthetic-resource-suit";
        let suit_revision = "suit-r5";
        let lander_revision = "lander-r4";
        let environments = BTreeSet::from([
            SuitOperatingEnvironment::LunarShadowOrPolarRegion,
        ]);
        let make_manifest = |component_id: &str, revision: &str, role: SuitInterfaceRole| {
            SuitInterfaceManifest {
                component_id: component_id.into(),
                hardware_revision: revision.into(),
                endpoints: vec![SuitInterfaceEndpoint {
                    interface_id: "resource-service.v1".into(),
                    endpoint_id: format!("{component_id}.resource-service"),
                    kind: SuitInterfaceKind::ConsumableTransfer,
                    role,
                    contract_revision: "1.0.0".into(),
                    schema_digest_sha256: format!("sha256:{}", "c".repeat(64)),
                    connector_profile_id: "synthetic-resource-connector-v1".into(),
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
                implementation_id: format!("synthetic-capability:{capability:?}"),
                evidence_level: SuitEvidenceLevel::BenchTested,
                evidence_reference: format!("synthetic-capability-evidence:{capability:?}"),
            })
            .collect();
        let configuration = MissionConfiguration {
            schema_version: 1,
            mission_id: "synthetic-resource-mission".into(),
            configuration_revision: "config-r1".into(),
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
            required_connection_ids: BTreeSet::from(["suit-lander-service".into()]),
            components: vec![
                MissionComponent {
                    component_id: "suit".into(),
                    kind: MissionComponentKind::Suit,
                    hardware_revision: suit_revision.into(),
                    interfaces: make_manifest("suit", suit_revision, SuitInterfaceRole::Consumer),
                },
                MissionComponent {
                    component_id: "lander".into(),
                    kind: MissionComponentKind::LanderOrVehicle,
                    hardware_revision: lander_revision.into(),
                    interfaces: make_manifest(
                        "lander",
                        lander_revision,
                        SuitInterfaceRole::Provider,
                    ),
                },
            ],
            connections: vec![MissionConnection {
                connection_id: "suit-lander-service".into(),
                left_component_id: "suit".into(),
                right_component_id: "lander".into(),
                policy: crate::suit_interop::SuitInteropPolicy {
                    minimum_evidence_level: SuitEvidenceLevel::Simulated,
                    required_environments: environments.clone(),
                },
            }],
        };
        let tasks = vec![MissionTaskRequirement {
            task_id: "shadow-service-work".into(),
            required: true,
            required_capabilities: BTreeSet::from([
                SuitCapability::ElectrostaticChargeManagement,
            ]),
            required_component_ids: BTreeSet::from(["lander".into()]),
            required_connection_ids: BTreeSet::from(["suit-lander-service".into()]),
            required_environments: environments.clone(),
        }];
        let (task_report, task_receipt) =
            evaluate_mission_tasks(&configuration, &tasks).unwrap();
        let plan = MissionResourceBudgetPlan {
            schema_version: 1,
            mission_id: configuration.mission_id.clone(),
            configuration_revision: configuration.configuration_revision.clone(),
            expected_configuration_digest_sha256: task_report.configuration_digest_sha256,
            expected_task_plan_digest_sha256: task_report.task_plan_digest_sha256,
            minimum_evidence_level: SuitEvidenceLevel::BenchTested,
            reserve_basis_points: 2_000,
            pools: vec![MissionResourcePool {
                pool_id: "lander-oxygen".into(),
                resource_kind: MissionResourceKind::OxygenMassMicrograms,
                provider_component_id: "lander".into(),
                provider_hardware_revision: lander_revision.into(),
                available: ResourceQuantityInterval {
                    lower: 1_000,
                    nominal: 1_100,
                    upper: 1_200,
                },
                evidence_level: SuitEvidenceLevel::BenchTested,
                evidence_reference: "synthetic-budget-evidence:oxygen-pool".into(),
                applicable_environments: environments.clone(),
            }],
            demands: vec![MissionResourceDemand {
                demand_id: "task-oxygen-1".into(),
                task_id: "shadow-service-work".into(),
                pool_id: "lander-oxygen".into(),
                resource_kind: MissionResourceKind::OxygenMassMicrograms,
                quantity: ResourceQuantityInterval {
                    lower: 100,
                    nominal: 200,
                    upper: 400,
                },
                evidence_level: SuitEvidenceLevel::BenchTested,
                evidence_reference: "synthetic-demand-estimate:oxygen-1".into(),
            }],
        };
        (configuration, tasks, task_receipt, plan)
    }

    #[test]
    fn conservative_interval_reserve_and_receipt_are_deterministic() {
        let (configuration, tasks, task_receipt, plan) = fixture();
        let (report, receipt) =
            evaluate_mission_resource_budget(&configuration, &tasks, &task_receipt, &plan)
                .unwrap();
        let pool = &report.pools[0];
        assert_eq!(pool.required_demand_upper, 400);
        assert_eq!(pool.reserve_quantity, 80);
        assert_eq!(pool.required_capacity, 480);
        assert_eq!(pool.required_margin, Some(520));
        assert_eq!(pool.required_shortfall, 0);
        assert_eq!(
            report.status,
            MissionResourceBudgetStatus::MeetsDeclaredResourcePolicy
        );
        assert!(!report.represents_physical_qualification);
        assert!(verify_mission_resource_budget_receipt(
            &configuration,
            &tasks,
            &task_receipt,
            &plan,
            &report,
            &receipt,
        ));
    }

    #[test]
    fn required_worst_case_demand_plus_reserve_blocks_when_supply_is_short() {
        let (configuration, tasks, task_receipt, mut plan) = fixture();
        plan.pools[0].available.lower = 479;
        plan.pools[0].available.nominal = 500;
        plan.pools[0].available.upper = 600;
        let (report, _) =
            evaluate_mission_resource_budget(&configuration, &tasks, &task_receipt, &plan).unwrap();
        assert_eq!(report.status, MissionResourceBudgetStatus::Blocked);
        assert_eq!(report.pools[0].required_shortfall, 1);
        assert!(report.findings.iter().any(|finding| {
            finding.code == MissionResourceFindingCode::RequiredDemandExceedsSupply
                && finding.severity == MissionResourceFindingSeverity::Blocker
        }));
    }

    #[test]
    fn optional_overage_degrades_without_blocking_required_budget() {
        let (configuration, mut tasks, _, mut plan) = fixture();
        tasks.push(MissionTaskRequirement {
            task_id: "optional-survey".into(),
            required: false,
            required_capabilities: BTreeSet::from([SuitCapability::ElectrostaticChargeManagement]),
            required_component_ids: BTreeSet::from(["lander".into()]),
            required_connection_ids: BTreeSet::from(["suit-lander-service".into()]),
            required_environments: BTreeSet::from([
                SuitOperatingEnvironment::LunarShadowOrPolarRegion,
            ]),
        });
        let (new_task_report, new_task_receipt) =
            evaluate_mission_tasks(&configuration, &tasks).unwrap();
        plan.expected_task_plan_digest_sha256 = new_task_report.task_plan_digest_sha256;
        plan.demands.push(MissionResourceDemand {
            demand_id: "optional-oxygen".into(),
            task_id: "optional-survey".into(),
            pool_id: "lander-oxygen".into(),
            resource_kind: MissionResourceKind::OxygenMassMicrograms,
            quantity: ResourceQuantityInterval {
                lower: 500,
                nominal: 600,
                upper: 700,
            },
            evidence_level: SuitEvidenceLevel::BenchTested,
            evidence_reference: "synthetic-demand-estimate:optional".into(),
        });
        let (report, _) = evaluate_mission_resource_budget(
            &configuration,
            &tasks,
            &new_task_receipt,
            &plan,
        )
        .unwrap();
        assert_eq!(report.status, MissionResourceBudgetStatus::Degraded);
        assert_eq!(report.pools[0].required_shortfall, 0);
        assert_eq!(report.pools[0].optional_shortfall, 180);
        assert!(report.findings.iter().any(|finding| {
            finding.code == MissionResourceFindingCode::OptionalDemandExceedsRemainingCapacity
                && finding.severity == MissionResourceFindingSeverity::Degradation
        }));

    }

    #[test]
    fn supply_and_demand_intervals_must_be_ordered() {
        let (configuration, tasks, task_receipt, mut plan) = fixture();
        plan.demands[0].quantity = ResourceQuantityInterval {
            lower: 100,
            nominal: 500,
            upper: 400,
        };
        assert_eq!(
            evaluate_mission_resource_budget(&configuration, &tasks, &task_receipt, &plan)
                .unwrap_err(),
            MissionResourceBudgetError::InvalidDemand
        );
    }

    #[test]
    fn provider_revision_mismatch_blocks_required_pool() {
        let (configuration, tasks, task_receipt, mut plan) = fixture();
        plan.pools[0].provider_hardware_revision = "lander-r5".into();
        let (report, _) =
            evaluate_mission_resource_budget(&configuration, &tasks, &task_receipt, &plan).unwrap();
        assert_eq!(report.status, MissionResourceBudgetStatus::Blocked);
        assert!(report.findings.iter().any(|finding| {
            finding.code == MissionResourceFindingCode::ProviderHardwareRevisionMismatch
                && finding.severity == MissionResourceFindingSeverity::Blocker
        }));
    }

    #[test]
    fn wrong_resource_kind_cannot_be_silently_coerced() {
        let (configuration, tasks, task_receipt, mut plan) = fixture();
        plan.demands[0].resource_kind = MissionResourceKind::DataVolumeBytes;
        assert_eq!(
            evaluate_mission_resource_budget(&configuration, &tasks, &task_receipt, &plan)
                .unwrap_err(),
            MissionResourceBudgetError::ResourceKindMismatch
        );
    }

    #[test]
    fn stale_task_receipt_and_stale_revision_are_rejected() {
        let (configuration, tasks, mut task_receipt, mut plan) = fixture();
        task_receipt.receipt_digest_sha256.push_str("tampered");
        assert_eq!(
            evaluate_mission_resource_budget(&configuration, &tasks, &task_receipt, &plan)
                .unwrap_err(),
            MissionResourceBudgetError::TaskReceiptMismatch
        );
        let (_, fresh_receipt) = evaluate_mission_tasks(&configuration, &tasks).unwrap();
        plan.configuration_revision = "config-r2".into();
        assert_eq!(
            evaluate_mission_resource_budget(&configuration, &tasks, &fresh_receipt, &plan)
                .unwrap_err(),
            MissionResourceBudgetError::MissionIdentityMismatch
        );
    }

    #[test]
    fn task_and_vector_order_do_not_change_budget_receipt() {
        let (mut configuration, tasks, task_receipt, mut plan) = fixture();
        let (report_a, receipt_a) =
            evaluate_mission_resource_budget(&configuration, &tasks, &task_receipt, &plan).unwrap();
        configuration.components.reverse();
        configuration.connections.reverse();
        configuration.suit_variant.capability_claims.reverse();
        plan.pools.reverse();
        plan.demands.reverse();
        let (report_b, receipt_b) =
            evaluate_mission_resource_budget(&configuration, &tasks, &task_receipt, &plan).unwrap();
        assert_eq!(report_a, report_b);
        assert_eq!(receipt_a, receipt_b);
    }

    #[test]
    fn reserve_rounds_up_and_extreme_arithmetic_fails_closed() {
        assert_eq!(compute_reserve(1, 1).unwrap(), 1);
        assert_eq!(compute_reserve(101, 100).unwrap(), 2);
        assert_eq!(
            compute_reserve(u64::MAX, 100_000).unwrap_err(),
            MissionResourceBudgetError::ReserveOverflow
        );
    }

    #[test]
    fn duplicate_provider_resource_pools_are_rejected_to_avoid_double_counting() {
        let (configuration, tasks, task_receipt, mut plan) = fixture();
        let mut duplicate = plan.pools[0].clone();
        duplicate.pool_id = "second-copy-of-same-supply".into();
        plan.pools.push(duplicate);
        assert_eq!(
            evaluate_mission_resource_budget(&configuration, &tasks, &task_receipt, &plan)
                .unwrap_err(),
            MissionResourceBudgetError::DuplicateProviderResourcePool
        );
    }

    #[test]
    fn changed_demand_invalidates_resource_receipt() {
        let (configuration, tasks, task_receipt, plan) = fixture();
        let (report, receipt) =
            evaluate_mission_resource_budget(&configuration, &tasks, &task_receipt, &plan).unwrap();
        let mut changed = plan.clone();
        changed.demands[0].quantity.upper += 1;
        assert!(!verify_mission_resource_budget_receipt(
            &configuration, &tasks, &task_receipt, &changed, &report, &receipt
        ));
    }
}
