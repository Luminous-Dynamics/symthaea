// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mission-relative peak resource reservations and oversubscription analysis (SX-034).
//!
//! This module is deliberately separate from SX-033 cumulative resource budgets.
//! It evaluates declared integer capacity/demand over half-open time windows
//! [start_ms, end_ms), using lower capacity bounds and upper reservation bounds.
//! It is a deterministic software analysis, not a physical model, real-time
//! scheduler, controller, or flight/safety qualification.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::suit_interop::{
    SuitInteropFindingStatus, SuitInterfaceKind, SuitInteropReportStatus,
    SuitOperatingEnvironment,
};
use crate::suit_mission_configuration::{
    MissionComponent, MissionConfiguration, MissionConfigurationReport,
};
use crate::suit_mission_tasks::{
    evaluate_mission_tasks, MissionTaskError, MissionTaskPlanStatus, MissionTaskReceipt,
    MissionTaskRequirement,
};
use crate::suit_profile::SuitEvidenceLevel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PeakResourceKind {
    ElectricalPowerMilliwatts,
    CoolantFlowMicrolitersPerSecond,
    DataRateBytesPerSecond,
    ComputeBudgetUnitsPerSecond,
    SharedEquipmentSlots,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeakQuantityInterval {
    pub lower: u64,
    pub nominal: u64,
    pub upper: u64,
}

impl PeakQuantityInterval {
    pub fn is_valid(&self) -> bool {
        self.lower <= self.nominal && self.nominal <= self.upper
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissionTimeWindow {
    /// Mission-relative monotonically increasing milliseconds, not wall time.
    pub start_ms: u64,
    /// Exclusive end. A window is active on [start_ms, end_ms).
    pub end_ms: u64,
}

impl MissionTimeWindow {
    pub fn is_valid(&self) -> bool {
        self.start_ms < self.end_ms
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeakResourcePool {
    pub pool_id: String,
    pub resource_kind: PeakResourceKind,
    pub provider_component_id: String,
    pub provider_hardware_revision: String,
    /// Exact SX-030 contract IDs that provide access to this resource pool.
    pub required_connection_ids: BTreeSet<String>,
    /// Interface kinds that constitute this resource path; empty is invalid.
    pub required_interface_kinds: BTreeSet<SuitInterfaceKind>,
    /// Conservative capacity envelope; capacity lower bound is budgeted.
    pub capacity: PeakQuantityInterval,
    pub evidence_level: SuitEvidenceLevel,
    pub evidence_reference: String,
    pub applicable_environments: BTreeSet<SuitOperatingEnvironment>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeakResourceReservation {
    pub reservation_id: String,
    pub task_id: String,
    pub pool_id: String,
    pub resource_kind: PeakResourceKind,
    pub window: MissionTimeWindow,
    /// Worst-case concurrent demand uses this interval's upper bound.
    pub demand: PeakQuantityInterval,
    pub evidence_level: SuitEvidenceLevel,
    pub evidence_reference: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeakResourceReservationPlan {
    pub schema_version: u32,
    pub mission_id: String,
    pub configuration_revision: String,
    /// Must match the fresh SX-031 graph report.
    pub expected_configuration_digest_sha256: String,
    /// Must match the fresh SX-032 task plan.
    pub expected_task_plan_digest_sha256: String,
    pub minimum_evidence_level: SuitEvidenceLevel,
    /// Fraction of the conservative capacity kept unavailable as reserve.
    /// 1_000 basis points = 10%; maximum accepted is 10_000 basis points.
    pub reserve_basis_points: u32,
    pub pools: Vec<PeakResourcePool>,
    pub reservations: Vec<PeakResourceReservation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PeakResourceFindingCode {
    MissingProviderComponent,
    ProviderHardwareRevisionMismatch,
    ProviderNotReachable,
    MissingProviderConnection,
    ProviderConnectionDoesNotBindProvider,
    ProviderConnectionResourceKindMismatch,
    ProviderConnectionBlocked,
    ProviderConnectionEnvironmentMismatch,
    EnvironmentScopeMismatch,
    SupplyEvidenceBelowPolicy,
    ReservationEvidenceBelowPolicy,
    RequiredOversubscription,
    OptionalOversubscription,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PeakResourceFindingSeverity {
    Degradation,
    Blocker,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeakResourceFinding {
    pub pool_id: String,
    pub task_id: Option<String>,
    pub reservation_id: Option<String>,
    pub window: Option<MissionTimeWindow>,
    pub code: PeakResourceFindingCode,
    pub severity: PeakResourceFindingSeverity,
    pub observed_evidence_level: Option<SuitEvidenceLevel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PeakWindowStatus {
    WithinDeclaredCapacity,
    RequiredOversubscribed,
    OptionalOversubscribed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeakResourceWindowEvaluation {
    pub pool_id: String,
    pub resource_kind: PeakResourceKind,
    pub window: MissionTimeWindow,
    /// Count of active reservations, including non-overbooked slices.
    pub active_reservation_count: usize,
    /// Stable, sorted IDs are retained for overbooked slices; empty otherwise.
    pub active_reservation_ids: Vec<String>,
    pub required_demand_upper: u64,
    pub optional_demand_upper: u64,
    pub reserve_quantity: u64,
    pub usable_capacity: u64,
    pub required_shortfall: u64,
    pub total_shortfall: u64,
    pub status: PeakWindowStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeakResourcePoolEvaluation {
    pub pool_id: String,
    pub resource_kind: PeakResourceKind,
    pub provider_component_id: String,
    pub provider_hardware_revision: String,
    pub capacity: PeakQuantityInterval,
    pub reserve_quantity: u64,
    pub usable_capacity: u64,
    pub peak_required_demand_upper: u64,
    pub peak_optional_demand_upper: u64,
    pub peak_total_demand_upper: u64,
    pub minimum_required_headroom: u64,
    pub maximum_required_shortfall: u64,
    pub maximum_total_shortfall: u64,
    pub required_oversubscribed_window_count: usize,
    pub optional_oversubscribed_window_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PeakResourceReservationStatus {
    Blocked,
    Degraded,
    MeetsDeclaredPeakResourcePolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeakResourceReservationReport {
    pub schema_version: u32,
    pub mission_id: String,
    pub configuration_revision: String,
    pub configuration_digest_sha256: String,
    pub task_plan_digest_sha256: String,
    pub reservation_plan_digest_sha256: String,
    pub task_plan_status: MissionTaskPlanStatus,
    pub pools: Vec<PeakResourcePoolEvaluation>,
    pub windows: Vec<PeakResourceWindowEvaluation>,
    pub findings: Vec<PeakResourceFinding>,
    pub status: PeakResourceReservationStatus,
    pub pool_count: usize,
    pub reservation_count: usize,
    pub evaluated_window_count: usize,
    pub required_oversubscribed_window_count: usize,
    pub optional_oversubscribed_window_count: usize,
    /// Always false: time-window arithmetic does not qualify physical systems.
    pub represents_physical_qualification: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeakResourceReservationReceipt {
    pub configuration_digest_sha256: String,
    pub task_plan_digest_sha256: String,
    pub task_receipt_digest_sha256: String,
    pub reservation_plan_digest_sha256: String,
    pub report_digest_sha256: String,
    /// Deterministic change evidence, not a digital signature or authentication.
    pub receipt_digest_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PeakResourceReservationError {
    UnsupportedSchemaVersion,
    InvalidPlanIdentity,
    InvalidPinnedDigest,
    InvalidReservePolicy,
    EmptyReservationPlan,
    InvalidPool,
    DuplicatePoolId,
    DuplicateProviderResourcePool,
    InvalidReservation,
    DuplicateReservationId,
    MissingReferencedTask,
    MissingReferencedPool,
    ResourceKindMismatch,
    UnusedPool,
    InvalidQuantityInterval,
    InvalidTimeWindow,
    QuantityOverflow,
    ReserveOverflow,
    TaskEvaluationFailed(MissionTaskError),
    TaskReceiptMismatch,
    ExpectedConfigurationDigestMismatch,
    ExpectedTaskPlanDigestMismatch,
    MissionIdentityMismatch,
}

#[derive(Debug, Clone, Copy, Default)]
struct LoadTotals {
    required_upper: u64,
    optional_upper: u64,
}

pub fn evaluate_peak_resource_reservations(
    configuration: &MissionConfiguration,
    tasks: &[MissionTaskRequirement],
    task_receipt: &MissionTaskReceipt,
    plan: &PeakResourceReservationPlan,
) -> Result<
    (PeakResourceReservationReport, PeakResourceReservationReceipt),
    PeakResourceReservationError,
> {
    validate_plan_structure(plan)?;

    let (task_report, fresh_task_receipt) = evaluate_mission_tasks(configuration, tasks)
        .map_err(PeakResourceReservationError::TaskEvaluationFailed)?;
    if &fresh_task_receipt != task_receipt {
        return Err(PeakResourceReservationError::TaskReceiptMismatch);
    }
    if plan.expected_configuration_digest_sha256 != task_report.configuration_digest_sha256 {
        return Err(PeakResourceReservationError::ExpectedConfigurationDigestMismatch);
    }
    if plan.expected_task_plan_digest_sha256 != task_report.task_plan_digest_sha256 {
        return Err(PeakResourceReservationError::ExpectedTaskPlanDigestMismatch);
    }
    if plan.mission_id != configuration.mission_id
        || plan.configuration_revision != configuration.configuration_revision
        || plan.mission_id != task_report.mission_id
        || plan.configuration_revision != task_report.configuration_revision
    {
        return Err(PeakResourceReservationError::MissionIdentityMismatch);
    }
    validate_references(plan, &task_report)?;

    let reservation_plan_digest_sha256 = digest_plan(plan);
    let task_by_id: BTreeMap<&str, _> = task_report
        .task_evaluations
        .iter()
        .map(|task| (task.task_id.as_str(), task))
        .collect();
    let pools_by_id: BTreeMap<&str, &PeakResourcePool> = plan
        .pools
        .iter()
        .map(|pool| (pool.pool_id.as_str(), pool))
        .collect();
    let component_by_id: BTreeMap<&str, &MissionComponent> = configuration
        .components
        .iter()
        .map(|component| (component.component_id.as_str(), component))
        .collect();

    let mut findings = Vec::new();
    let mut windows = Vec::new();
    let mut pool_evaluations = Vec::new();
    let mut required_oversubscribed_window_count = 0;
    let mut optional_oversubscribed_window_count = 0;

    let mut ordered_pools: Vec<&PeakResourcePool> = plan.pools.iter().collect();
    ordered_pools.sort_by(|left, right| left.pool_id.cmp(&right.pool_id));
    for pool in ordered_pools {
        let reserve_quantity = compute_reserve(
            pool.capacity.lower,
            plan.reserve_basis_points,
        )?;
        let usable_capacity = pool.capacity.lower.saturating_sub(reserve_quantity);
        let reservations: Vec<&PeakResourceReservation> = plan
            .reservations
            .iter()
            .filter(|reservation| reservation.pool_id == pool.pool_id)
            .collect();
        let mut boundaries = BTreeSet::new();
        let mut starts_at: BTreeMap<u64, Vec<&PeakResourceReservation>> = BTreeMap::new();
        let mut ends_at: BTreeMap<u64, Vec<&PeakResourceReservation>> = BTreeMap::new();
        for reservation in &reservations {
            boundaries.insert(reservation.window.start_ms);
            boundaries.insert(reservation.window.end_ms);
            starts_at.entry(reservation.window.start_ms).or_default().push(*reservation);
            ends_at.entry(reservation.window.end_ms).or_default().push(*reservation);
        }
        for reservations_at_tick in starts_at.values_mut().chain(ends_at.values_mut()) {
            reservations_at_tick.sort_by(|left, right| {
                left.reservation_id.cmp(&right.reservation_id)
            });
        }
        let ordered_boundaries: Vec<u64> = boundaries.into_iter().collect();

        let mut peak_required_demand_upper = 0;
        let mut peak_optional_demand_upper = 0;
        let mut peak_total_demand_upper = 0;
        let mut minimum_required_headroom = u64::MAX;
        let mut maximum_required_shortfall = 0;
        let mut maximum_total_shortfall = 0;
        let mut pool_required_oversubscribed_count = 0;
        let mut pool_optional_oversubscribed_count = 0;

        let mut active: BTreeMap<String, &PeakResourceReservation> = BTreeMap::new();
        let mut totals = LoadTotals::default();
        for index in 0..ordered_boundaries.len().saturating_sub(1) {
            let start_ms = ordered_boundaries[index];
            let end_ms = ordered_boundaries[index + 1];
            if start_ms >= end_ms {
                continue;
            }

            // Process endings before starts at each tick: windows are [start,end).
            if let Some(ending) = ends_at.get(&start_ms) {
                for reservation in ending {
                    if let Some(removed) = active.remove(&reservation.reservation_id) {
                        let task = task_by_id[removed.task_id.as_str()];
                        if task.required {
                            totals.required_upper = totals.required_upper
                                .checked_sub(removed.demand.upper)
                                .ok_or(PeakResourceReservationError::QuantityOverflow)?;
                        } else {
                            totals.optional_upper = totals.optional_upper
                                .checked_sub(removed.demand.upper)
                                .ok_or(PeakResourceReservationError::QuantityOverflow)?;
                        }
                    }
                }
            }
            if let Some(starting) = starts_at.get(&start_ms) {
                for reservation in starting {
                    let task = task_by_id[reservation.task_id.as_str()];
                    if task.required {
                        totals.required_upper = totals.required_upper
                            .checked_add(reservation.demand.upper)
                            .ok_or(PeakResourceReservationError::QuantityOverflow)?;
                    } else {
                        totals.optional_upper = totals.optional_upper
                            .checked_add(reservation.demand.upper)
                            .ok_or(PeakResourceReservationError::QuantityOverflow)?;
                    }
                    active.insert(reservation.reservation_id.clone(), *reservation);
                }
            }
            if active.is_empty() {
                continue;
            }
            let total_upper = totals
                .required_upper
                .checked_add(totals.optional_upper)
                .ok_or(PeakResourceReservationError::QuantityOverflow)?;
            peak_required_demand_upper = peak_required_demand_upper.max(totals.required_upper);
            peak_optional_demand_upper = peak_optional_demand_upper.max(totals.optional_upper);
            peak_total_demand_upper = peak_total_demand_upper.max(total_upper);

            let required_shortfall = totals.required_upper.saturating_sub(usable_capacity);
            let total_shortfall = total_upper.saturating_sub(usable_capacity);
            let status = if required_shortfall > 0 {
                pool_required_oversubscribed_count += 1;
                required_oversubscribed_window_count += 1;
                maximum_required_shortfall = maximum_required_shortfall.max(required_shortfall);
                maximum_total_shortfall = maximum_total_shortfall.max(total_shortfall);
                minimum_required_headroom = 0;
                PeakWindowStatus::RequiredOversubscribed
            } else if total_shortfall > 0 {
                pool_optional_oversubscribed_count += 1;
                optional_oversubscribed_window_count += 1;
                maximum_total_shortfall = maximum_total_shortfall.max(total_shortfall);
                minimum_required_headroom =
                    minimum_required_headroom.min(usable_capacity - totals.required_upper);
                PeakWindowStatus::OptionalOversubscribed
            } else {
                minimum_required_headroom =
                    minimum_required_headroom.min(usable_capacity - totals.required_upper);
                PeakWindowStatus::WithinDeclaredCapacity
            };

            let active_reservation_ids = if status == PeakWindowStatus::WithinDeclaredCapacity {
                Vec::new()
            } else {
                active.keys().cloned().collect()
            };
            windows.push(PeakResourceWindowEvaluation {
                pool_id: pool.pool_id.clone(),
                resource_kind: pool.resource_kind,
                window: MissionTimeWindow { start_ms, end_ms },
                active_reservation_count: active.len(),
                active_reservation_ids,
                required_demand_upper: totals.required_upper,
                optional_demand_upper: totals.optional_upper,
                reserve_quantity,
                usable_capacity,
                required_shortfall,
                total_shortfall,
                status,
            });
        }

        if minimum_required_headroom == u64::MAX {
            minimum_required_headroom = usable_capacity;
        }

        pool_evaluations.push(PeakResourcePoolEvaluation {
            pool_id: pool.pool_id.clone(),
            resource_kind: pool.resource_kind,
            provider_component_id: pool.provider_component_id.clone(),
            provider_hardware_revision: pool.provider_hardware_revision.clone(),
            capacity: pool.capacity,
            reserve_quantity,
            usable_capacity,
            peak_required_demand_upper,
            peak_optional_demand_upper,
            peak_total_demand_upper,
            minimum_required_headroom,
            maximum_required_shortfall,
            maximum_total_shortfall,
            required_oversubscribed_window_count: pool_required_oversubscribed_count,
            optional_oversubscribed_window_count: pool_optional_oversubscribed_count,
        });
    }

    // Verify provider identity, evidence and reachability for every reservation.
    // The task environment is cached to avoid repeating graph traversal.
    let mut reachability_cache: BTreeMap<
        BTreeSet<SuitOperatingEnvironment>,
        BTreeSet<String>,
    > = BTreeMap::new();
    let mut ordered_reservations: Vec<&PeakResourceReservation> =
        plan.reservations.iter().collect();
    ordered_reservations.sort_by(|left, right| left.reservation_id.cmp(&right.reservation_id));
    for reservation in ordered_reservations {
        let task = task_by_id[reservation.task_id.as_str()];
        let pool = pools_by_id[reservation.pool_id.as_str()];
        let severity = if task.required {
            PeakResourceFindingSeverity::Blocker
        } else {
            PeakResourceFindingSeverity::Degradation
        };
        let scope = task.required_environments.is_subset(&pool.applicable_environments);
        if !scope {
            findings.push(PeakResourceFinding {
                pool_id: pool.pool_id.clone(),
                task_id: Some(task.task_id.clone()),
                reservation_id: Some(reservation.reservation_id.clone()),
                window: Some(reservation.window),
                code: PeakResourceFindingCode::EnvironmentScopeMismatch,
                severity,
                observed_evidence_level: None,
            });
        }

        // A generic path to a provider is not enough: pin the exact interface
        // contracts intended to supply this peak resource.
        for connection_id in &pool.required_connection_ids {
            match task_report
                .mission_configuration_report
                .connection_evaluations
                .iter()
                .find(|connection| connection.connection_id == *connection_id)
            {
                None => findings.push(PeakResourceFinding {
                    pool_id: pool.pool_id.clone(),
                    task_id: Some(task.task_id.clone()),
                    reservation_id: Some(reservation.reservation_id.clone()),
                    window: Some(reservation.window),
                    code: PeakResourceFindingCode::MissingProviderConnection,
                    severity,
                    observed_evidence_level: None,
                }),
                Some(connection) => {
                    if connection.left_component_id != pool.provider_component_id
                        && connection.right_component_id != pool.provider_component_id
                    {
                        findings.push(PeakResourceFinding {
                            pool_id: pool.pool_id.clone(),
                            task_id: Some(task.task_id.clone()),
                            reservation_id: Some(reservation.reservation_id.clone()),
                            window: Some(reservation.window),
                            code: PeakResourceFindingCode::ProviderConnectionDoesNotBindProvider,
                            severity,
                            observed_evidence_level: None,
                        });
                    }
                    let resource_interface_matches = connection.interoperability.findings.iter()
                        .any(|finding| {
                            finding.status == SuitInteropFindingStatus::CompatibleUnderPolicy
                                && pool.required_interface_kinds.contains(&finding.interface_kind)
                        });
                    if !resource_interface_matches {
                        findings.push(PeakResourceFinding {
                            pool_id: pool.pool_id.clone(),
                            task_id: Some(task.task_id.clone()),
                            reservation_id: Some(reservation.reservation_id.clone()),
                            window: Some(reservation.window),
                            code: PeakResourceFindingCode::ProviderConnectionResourceKindMismatch,
                            severity,
                            observed_evidence_level: None,
                        });
                    }
                    if connection.interoperability.status
                        != SuitInteropReportStatus::MeetsDeclaredCompatibilityPolicy
                    {
                        findings.push(PeakResourceFinding {
                            pool_id: pool.pool_id.clone(),
                            task_id: Some(task.task_id.clone()),
                            reservation_id: Some(reservation.reservation_id.clone()),
                            window: Some(reservation.window),
                            code: PeakResourceFindingCode::ProviderConnectionBlocked,
                            severity,
                            observed_evidence_level: None,
                        });
                    }
                    if !task.required_environments.is_subset(
                        &connection.interoperability.policy.required_environments,
                    ) {
                        findings.push(PeakResourceFinding {
                            pool_id: pool.pool_id.clone(),
                            task_id: Some(task.task_id.clone()),
                            reservation_id: Some(reservation.reservation_id.clone()),
                            window: Some(reservation.window),
                            code: PeakResourceFindingCode::ProviderConnectionEnvironmentMismatch,
                            severity,
                            observed_evidence_level: None,
                        });
                    }
                }
            }
        }

        match component_by_id.get(pool.provider_component_id.as_str()) {
            None => findings.push(PeakResourceFinding {
                pool_id: pool.pool_id.clone(),
                task_id: Some(task.task_id.clone()),
                reservation_id: Some(reservation.reservation_id.clone()),
                window: Some(reservation.window),
                code: PeakResourceFindingCode::MissingProviderComponent,
                severity,
                observed_evidence_level: None,
            }),
            Some(component) => {
                if component.hardware_revision != pool.provider_hardware_revision {
                    findings.push(PeakResourceFinding {
                        pool_id: pool.pool_id.clone(),
                        task_id: Some(task.task_id.clone()),
                        reservation_id: Some(reservation.reservation_id.clone()),
                        window: Some(reservation.window),
                        code: PeakResourceFindingCode::ProviderHardwareRevisionMismatch,
                        severity,
                        observed_evidence_level: None,
                    });
                }
                if !reachability_cache.contains_key(&task.required_environments) {
                    let reachable = reachable_for_environment(
                        &configuration.suit_component_id,
                        &configuration.components,
                        &task_report.mission_configuration_report,
                        &task.required_environments,
                    );
                    reachability_cache.insert(task.required_environments.clone(), reachable);
                }
                if !reachability_cache[&task.required_environments]
                    .contains(&pool.provider_component_id)
                {
                    findings.push(PeakResourceFinding {
                        pool_id: pool.pool_id.clone(),
                        task_id: Some(task.task_id.clone()),
                        reservation_id: Some(reservation.reservation_id.clone()),
                        window: Some(reservation.window),
                        code: PeakResourceFindingCode::ProviderNotReachable,
                        severity,
                        observed_evidence_level: None,
                    });
                }
            }
        }

        if pool.evidence_level < plan.minimum_evidence_level {
            findings.push(PeakResourceFinding {
                pool_id: pool.pool_id.clone(),
                task_id: Some(task.task_id.clone()),
                reservation_id: Some(reservation.reservation_id),
                window: Some(reservation.window),
                code: PeakResourceFindingCode::SupplyEvidenceBelowPolicy,
                severity,
                observed_evidence_level: Some(pool.evidence_level),
            });
        }
        if reservation.evidence_level < plan.minimum_evidence_level {
            findings.push(PeakResourceFinding {
                pool_id: pool.pool_id.clone(),
                task_id: Some(task.task_id.clone()),
                reservation_id: Some(reservation.reservation_id.clone()),
                window: Some(reservation.window),
                code: PeakResourceFindingCode::ReservationEvidenceBelowPolicy,
                severity,
                observed_evidence_level: Some(reservation.evidence_level),
            });
        }
    }

    // Reclassify oversubscription windows from the active task criticality, not
    // merely the pool totals, and emit one stable finding per active reservation.
    let mut windows_by_pool: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, window) in windows.iter().enumerate() {
        windows_by_pool.entry(window.pool_id.clone()).or_default().push(index);
    }
    for (pool_id, indices) in windows_by_pool {
        for index in indices {
            let window = &windows[index];
            match window.status {
                PeakWindowStatus::RequiredOversubscribed => {
                    for reservation_id in &window.active_reservation_ids {
                        let reservation = plan.reservations.iter()
                            .find(|item| item.reservation_id == *reservation_id)
                            .expect("active reservation ID is from validated plan");
                        let task = task_by_id[reservation.task_id.as_str()];
                        if task.required && reservation.demand.upper > 0 {
                            findings.push(PeakResourceFinding {
                                pool_id: pool_id.clone(),
                                task_id: Some(task.task_id.clone()),
                                reservation_id: Some(reservation.reservation_id.clone()),
                                window: Some(window.window),
                                code: PeakResourceFindingCode::RequiredOversubscription,
                                severity: PeakResourceFindingSeverity::Blocker,
                                observed_evidence_level: Some(reservation.evidence_level),
                            });
                        }
                    }
                }
                PeakWindowStatus::OptionalOversubscribed => {
                    for reservation_id in &window.active_reservation_ids {
                        let reservation = plan.reservations.iter()
                            .find(|item| item.reservation_id == *reservation_id)
                            .expect("active reservation ID is from validated plan");
                        let task = task_by_id[reservation.task_id.as_str()];
                        if !task.required && reservation.demand.upper > 0 {
                            findings.push(PeakResourceFinding {
                                pool_id: pool_id.clone(),
                                task_id: Some(task.task_id.clone()),
                                reservation_id: Some(reservation.reservation_id.clone()),
                                window: Some(window.window),
                                code: PeakResourceFindingCode::OptionalOversubscription,
                                severity: PeakResourceFindingSeverity::Degradation,
                                observed_evidence_level: Some(reservation.evidence_level),
                            });
                        }
                    }
                }
                PeakWindowStatus::WithinDeclaredCapacity => {}
            }
        }
    }

    findings.sort_by(|left, right| {
        left.pool_id
            .cmp(&right.pool_id)
            .then(left.window.cmp(&right.window))
            .then(left.task_id.cmp(&right.task_id))
            .then(left.reservation_id.cmp(&right.reservation_id))
            .then(left.code.cmp(&right.code))
            .then(left.severity.cmp(&right.severity))
            .then(left.observed_evidence_level.cmp(&right.observed_evidence_level))
    });
    windows.sort_by(|left, right| {
        left.pool_id
            .cmp(&right.pool_id)
            .then(left.window.start_ms.cmp(&right.window.start_ms))
            .then(left.window.end_ms.cmp(&right.window.end_ms))
    });

    let blocker_count = findings.iter()
        .filter(|finding| finding.severity == PeakResourceFindingSeverity::Blocker)
        .count();
    let degradation_count = findings.iter()
        .filter(|finding| finding.severity == PeakResourceFindingSeverity::Degradation)
        .count();
    let status = if task_report.status == MissionTaskPlanStatus::Blocked
        || blocker_count > 0
        || required_oversubscribed_window_count > 0
    {
        PeakResourceReservationStatus::Blocked
    } else if task_report.status == MissionTaskPlanStatus::Degraded
        || degradation_count > 0
        || optional_oversubscribed_window_count > 0
    {
        PeakResourceReservationStatus::Degraded
    } else {
        PeakResourceReservationStatus::MeetsDeclaredPeakResourcePolicy
    };

    let report = PeakResourceReservationReport {
        schema_version: 1,
        mission_id: plan.mission_id.clone(),
        configuration_revision: plan.configuration_revision.clone(),
        configuration_digest_sha256: plan.expected_configuration_digest_sha256.clone(),
        task_plan_digest_sha256: plan.expected_task_plan_digest_sha256.clone(),
        reservation_plan_digest_sha256: reservation_plan_digest_sha256.clone(),
        task_plan_status: task_report.status,
        pool_count: pool_evaluations.len(),
        reservation_count: plan.reservations.len(),
        evaluated_window_count: windows.len(),
        required_oversubscribed_window_count,
        optional_oversubscribed_window_count,
        pools: pool_evaluations,
        windows,
        findings,
        status,
        represents_physical_qualification: false,
    };

    let report_digest_sha256 = digest_report(&report);
    let mut receipt = PeakResourceReservationReceipt {
        configuration_digest_sha256: plan.expected_configuration_digest_sha256.clone(),
        task_plan_digest_sha256: plan.expected_task_plan_digest_sha256.clone(),
        task_receipt_digest_sha256: digest_task_receipt(task_receipt),
        reservation_plan_digest_sha256,
        report_digest_sha256,
        receipt_digest_sha256: String::new(),
    };
    receipt.receipt_digest_sha256 = digest_receipt(&receipt);
    Ok((report, receipt))
}

pub fn verify_peak_resource_reservation_receipt(
    configuration: &MissionConfiguration,
    tasks: &[MissionTaskRequirement],
    task_receipt: &MissionTaskReceipt,
    plan: &PeakResourceReservationPlan,
    expected_report: &PeakResourceReservationReport,
    expected_receipt: &PeakResourceReservationReceipt,
) -> bool {
    match evaluate_peak_resource_reservations(configuration, tasks, task_receipt, plan) {
        Ok((report, receipt)) => report == *expected_report && receipt == *expected_receipt,
        Err(_) => false,
    }
}

fn validate_plan_structure(
    plan: &PeakResourceReservationPlan,
) -> Result<(), PeakResourceReservationError> {
    if plan.schema_version != 1 {
        return Err(PeakResourceReservationError::UnsupportedSchemaVersion);
    }
    if !clean_identifier(&plan.mission_id) || !clean_identifier(&plan.configuration_revision) {
        return Err(PeakResourceReservationError::InvalidPlanIdentity);
    }
    if !valid_sha256(&plan.expected_configuration_digest_sha256)
        || !valid_sha256(&plan.expected_task_plan_digest_sha256)
    {
        return Err(PeakResourceReservationError::InvalidPinnedDigest);
    }
    if plan.reserve_basis_points > 10_000 {
        return Err(PeakResourceReservationError::InvalidReservePolicy);
    }
    if plan.pools.is_empty() || plan.reservations.is_empty() {
        return Err(PeakResourceReservationError::EmptyReservationPlan);
    }

    let mut pool_ids = BTreeSet::new();
    let mut provider_resource_pairs = BTreeSet::new();
    for pool in &plan.pools {
        if !clean_identifier(&pool.pool_id)
            || !clean_identifier(&pool.provider_component_id)
            || !clean_identifier(&pool.provider_hardware_revision)
            || !clean_identifier(&pool.evidence_reference)
            || pool.applicable_environments.is_empty()
            || pool.required_connection_ids.is_empty()
            || pool.required_connection_ids.iter().any(|id| !clean_identifier(id))
            || pool.required_interface_kinds.is_empty()
        {
            return Err(PeakResourceReservationError::InvalidPool);
        }
        if !pool.capacity.is_valid() {
            return Err(PeakResourceReservationError::InvalidQuantityInterval);
        }
        if !pool_ids.insert(pool.pool_id.as_str()) {
            return Err(PeakResourceReservationError::DuplicatePoolId);
        }
        if !provider_resource_pairs.insert((
            pool.provider_component_id.as_str(),
            pool.resource_kind,
        )) {
            return Err(PeakResourceReservationError::DuplicateProviderResourcePool);
        }
    }

    let mut reservation_ids = BTreeSet::new();
    let mut used_pools = BTreeSet::new();
    for reservation in &plan.reservations {
        if !clean_identifier(&reservation.reservation_id)
            || !clean_identifier(&reservation.task_id)
            || !clean_identifier(&reservation.pool_id)
            || !clean_identifier(&reservation.evidence_reference)
        {
            return Err(PeakResourceReservationError::InvalidReservation);
        }
        if !reservation_ids.insert(reservation.reservation_id.as_str()) {
            return Err(PeakResourceReservationError::DuplicateReservationId);
        }
        if !reservation.window.is_valid() {
            return Err(PeakResourceReservationError::InvalidTimeWindow);
        }
        if !reservation.demand.is_valid() || reservation.demand.upper == 0 {
            return Err(PeakResourceReservationError::InvalidQuantityInterval);
        }
        used_pools.insert(reservation.pool_id.as_str());
    }
    if plan.pools.iter().any(|pool| !used_pools.contains(pool.pool_id.as_str())) {
        return Err(PeakResourceReservationError::UnusedPool);
    }
    Ok(())
}

fn validate_references(
    plan: &PeakResourceReservationPlan,
    task_report: &crate::suit_mission_tasks::MissionTaskPlanReport,
) -> Result<(), PeakResourceReservationError> {
    let tasks: BTreeSet<&str> = task_report
        .task_evaluations
        .iter()
        .map(|task| task.task_id.as_str())
        .collect();
    let pools: BTreeMap<&str, &PeakResourcePool> = plan
        .pools
        .iter()
        .map(|pool| (pool.pool_id.as_str(), pool))
        .collect();
    for reservation in &plan.reservations {
        if !tasks.contains(reservation.task_id.as_str()) {
            return Err(PeakResourceReservationError::MissingReferencedTask);
        }
        let pool = pools.get(reservation.pool_id.as_str())
            .ok_or(PeakResourceReservationError::MissingReferencedPool)?;
        if pool.resource_kind != reservation.resource_kind {
            return Err(PeakResourceReservationError::ResourceKindMismatch);
        }
    }
    Ok(())
}

fn compute_reserve(
    capacity_lower: u64,
    reserve_basis_points: u32,
) -> Result<u64, PeakResourceReservationError> {
    let product = u128::from(capacity_lower)
        .checked_mul(u128::from(reserve_basis_points))
        .ok_or(PeakResourceReservationError::ReserveOverflow)?;
    let rounded = product
        .checked_add(9_999)
        .ok_or(PeakResourceReservationError::ReserveOverflow)?
        / 10_000;
    u64::try_from(rounded).map_err(|_| PeakResourceReservationError::ReserveOverflow)
}

fn reachable_for_environment(
    suit_component_id: &str,
    components: &[MissionComponent],
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
            || !environments.is_subset(&connection.interoperability.policy.required_environments)
        {
            continue;
        }
        adjacency.entry(connection.left_component_id.clone())
            .or_default()
            .insert(connection.right_component_id.clone());
        adjacency.entry(connection.right_component_id.clone())
            .or_default()
            .insert(connection.left_component_id.clone());
    }
    let mut reached = BTreeSet::new();
    let mut queue = VecDeque::from([suit_component_id.to_string()]);
    while let Some(node) = queue.pop_front() {
        if !reached.insert(node.clone()) {
            continue;
        }
        if let Some(neighbors) = adjacency.get(&node) {
            queue.extend(neighbors.iter().cloned());
        }
    }
    reached
}

fn digest_plan(plan: &PeakResourceReservationPlan) -> String {
    let mut writer = HashWriter::new("symthaea.sx034.reservation-plan.v1");
    writer.u64(plan.schema_version as u64);
    writer.string(&plan.mission_id);
    writer.string(&plan.configuration_revision);
    writer.string(&plan.expected_configuration_digest_sha256);
    writer.string(&plan.expected_task_plan_digest_sha256);
    writer.string(evidence_tag(plan.minimum_evidence_level));
    writer.u64(u64::from(plan.reserve_basis_points));

    let mut pools: Vec<_> = plan.pools.iter().collect();
    pools.sort_by(|left, right| left.pool_id.cmp(&right.pool_id));
    writer.usize(pools.len());
    for pool in pools {
        writer.string(&pool.pool_id);
        writer.string(resource_kind_tag(pool.resource_kind));
        writer.string(&pool.provider_component_id);
        writer.string(&pool.provider_hardware_revision);
        writer.usize(pool.required_connection_ids.len());
        for connection_id in &pool.required_connection_ids {
            writer.string(connection_id);
        }
        writer.usize(pool.required_interface_kinds.len());
        for interface_kind in &pool.required_interface_kinds {
            writer.string(interface_kind_tag(*interface_kind));
        }
        write_interval(
            &mut writer,
            pool.capacity.lower,
            pool.capacity.nominal,
            pool.capacity.upper,
        );
        writer.string(evidence_tag(pool.evidence_level));
        writer.string(&pool.evidence_reference);
        writer.usize(pool.applicable_environments.len());
        for environment in &pool.applicable_environments {
            writer.string(environment_tag(*environment));
        }
    }

    let mut reservations: Vec<_> = plan.reservations.iter().collect();
    reservations.sort_by(|left, right| left.reservation_id.cmp(&right.reservation_id));
    writer.usize(reservations.len());
    for reservation in reservations {
        writer.string(&reservation.reservation_id);
        writer.string(&reservation.task_id);
        writer.string(&reservation.pool_id);
        writer.string(resource_kind_tag(reservation.resource_kind));
        writer.u64(reservation.window.start_ms);
        writer.u64(reservation.window.end_ms);
        write_interval(
            &mut writer,
            reservation.demand.lower,
            reservation.demand.nominal,
            reservation.demand.upper,
        );
        writer.string(evidence_tag(reservation.evidence_level));
        writer.string(&reservation.evidence_reference);
    }
    writer.finish()
}

fn digest_report(report: &PeakResourceReservationReport) -> String {
    let mut writer = HashWriter::new("symthaea.sx034.reservation-report.v1");
    writer.u64(report.schema_version as u64);
    writer.string(&report.mission_id);
    writer.string(&report.configuration_revision);
    writer.string(&report.configuration_digest_sha256);
    writer.string(&report.task_plan_digest_sha256);
    writer.string(&report.reservation_plan_digest_sha256);
    writer.string(task_plan_status_tag(report.task_plan_status));
    writer.string(reservation_status_tag(report.status));
    writer.boolean(report.represents_physical_qualification);
    writer.usize(report.pool_count);
    writer.usize(report.reservation_count);
    writer.usize(report.evaluated_window_count);
    writer.usize(report.required_oversubscribed_window_count);
    writer.usize(report.optional_oversubscribed_window_count);

    writer.usize(report.pools.len());
    for pool in &report.pools {
        writer.string(&pool.pool_id);
        writer.string(resource_kind_tag(pool.resource_kind));
        writer.string(&pool.provider_component_id);
        writer.string(&pool.provider_hardware_revision);
        write_interval(
            &mut writer,
            pool.capacity.lower,
            pool.capacity.nominal,
            pool.capacity.upper,
        );
        writer.u64(pool.reserve_quantity);
        writer.u64(pool.usable_capacity);
        writer.u64(pool.peak_required_demand_upper);
        writer.u64(pool.peak_optional_demand_upper);
        writer.u64(pool.peak_total_demand_upper);
        writer.u64(pool.minimum_required_headroom);
        writer.u64(pool.maximum_required_shortfall);
        writer.u64(pool.maximum_total_shortfall);
        writer.usize(pool.required_oversubscribed_window_count);
        writer.usize(pool.optional_oversubscribed_window_count);
    }

    writer.usize(report.windows.len());
    for window in &report.windows {
        writer.string(&window.pool_id);
        writer.string(resource_kind_tag(window.resource_kind));
        writer.u64(window.window.start_ms);
        writer.u64(window.window.end_ms);
        writer.usize(window.active_reservation_count);
        writer.usize(window.active_reservation_ids.len());
        for id in &window.active_reservation_ids {
            writer.string(id);
        }
        writer.u64(window.required_demand_upper);
        writer.u64(window.optional_demand_upper);
        writer.u64(window.reserve_quantity);
        writer.u64(window.usable_capacity);
        writer.u64(window.required_shortfall);
        writer.u64(window.total_shortfall);
        writer.string(window_status_tag(window.status));
    }

    writer.usize(report.findings.len());
    for finding in &report.findings {
        writer.string(&finding.pool_id);
        writer.option_string(finding.task_id.as_deref());
        writer.option_string(finding.reservation_id.as_deref());
        match finding.window {
            Some(window) => {
                writer.boolean(true);
                writer.u64(window.start_ms);
                writer.u64(window.end_ms);
            }
            None => writer.boolean(false),
        }
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
    writer.finish()
}

fn digest_receipt(receipt: &PeakResourceReservationReceipt) -> String {
    let mut writer = HashWriter::new("symthaea.sx034.reservation-receipt.v1");
    writer.string(&receipt.configuration_digest_sha256);
    writer.string(&receipt.task_plan_digest_sha256);
    writer.string(&receipt.task_receipt_digest_sha256);
    writer.string(&receipt.reservation_plan_digest_sha256);
    writer.string(&receipt.report_digest_sha256);
    writer.finish()
}

fn digest_task_receipt(
    receipt: &MissionTaskReceipt,
) -> String {
    let mut writer = HashWriter::new("symthaea.sx034.bound-task-receipt.v1");
    writer.string(&receipt.configuration_digest_sha256);
    writer.string(&receipt.task_plan_digest_sha256);
    writer.string(&receipt.mission_configuration_report_digest_sha256);
    writer.string(&receipt.task_report_digest_sha256);
    writer.string(&receipt.receipt_digest_sha256);
    writer.finish()
}

fn write_interval(writer: &mut HashWriter, lower: u64, nominal: u64, upper: u64) {
    writer.u64(lower);
    writer.u64(nominal);
    writer.u64(upper);
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

fn task_plan_status_tag(value: MissionTaskPlanStatus) -> &'static str {
    match value {
        MissionTaskPlanStatus::Blocked => "blocked",
        MissionTaskPlanStatus::Degraded => "degraded",
        MissionTaskPlanStatus::MeetsDeclaredTaskPolicy => "meets_declared_task_policy",
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

fn window_status_tag(value: PeakWindowStatus) -> &'static str {
    match value {
        PeakWindowStatus::WithinDeclaredCapacity => "within_declared_capacity",
        PeakWindowStatus::RequiredOversubscribed => "required_oversubscribed",
        PeakWindowStatus::OptionalOversubscribed => "optional_oversubscribed",
    }
}

fn finding_code_tag(value: PeakResourceFindingCode) -> &'static str {
    match value {
        PeakResourceFindingCode::MissingProviderComponent => "missing_provider_component",
        PeakResourceFindingCode::ProviderHardwareRevisionMismatch => {
            "provider_hardware_revision_mismatch"
        }
        PeakResourceFindingCode::ProviderNotReachable => "provider_not_reachable",
        PeakResourceFindingCode::MissingProviderConnection => "missing_provider_connection",
        PeakResourceFindingCode::ProviderConnectionDoesNotBindProvider => {
            "provider_connection_does_not_bind_provider"
        }
        PeakResourceFindingCode::ProviderConnectionResourceKindMismatch => {
            "provider_connection_resource_kind_mismatch"
        }
        PeakResourceFindingCode::ProviderConnectionBlocked => "provider_connection_blocked",
        PeakResourceFindingCode::ProviderConnectionEnvironmentMismatch => {
            "provider_connection_environment_mismatch"
        }
        PeakResourceFindingCode::EnvironmentScopeMismatch => "environment_scope_mismatch",
        PeakResourceFindingCode::SupplyEvidenceBelowPolicy => "supply_evidence_below_policy",
        PeakResourceFindingCode::ReservationEvidenceBelowPolicy => {
            "reservation_evidence_below_policy"
        }
        PeakResourceFindingCode::RequiredOversubscription => "required_oversubscription",
        PeakResourceFindingCode::OptionalOversubscription => "optional_oversubscription",
    }
}

fn finding_severity_tag(value: PeakResourceFindingSeverity) -> &'static str {
    match value {
        PeakResourceFindingSeverity::Degradation => "degradation",
        PeakResourceFindingSeverity::Blocker => "blocker",
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
        PeakResourceReservationPlan,
    ) {
        let variant_id = "synthetic-peak-suit";
        let suit_revision = "suit-r6";
        let vehicle_revision = "vehicle-r6";
        let environments =
            BTreeSet::from([SuitOperatingEnvironment::LunarShadowOrPolarRegion]);
        let manifest = |component_id: &str, revision: &str, role: SuitInterfaceRole| {
            SuitInterfaceManifest {
                component_id: component_id.into(),
                hardware_revision: revision.into(),
                endpoints: vec![SuitInterfaceEndpoint {
                    interface_id: "power-distribution.v1".into(),
                    endpoint_id: format!("{component_id}.power"),
                    kind: SuitInterfaceKind::ElectricalPower,
                    role,
                    contract_revision: "1.0.0".into(),
                    schema_digest_sha256: format!("sha256:{}", "e".repeat(64)),
                    connector_profile_id: "synthetic-power-bus-v1".into(),
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
                implementation_id: format!("synthetic-implementation:{capability:?}"),
                evidence_level: SuitEvidenceLevel::BenchTested,
                evidence_reference: format!("synthetic-capability-evidence:{capability:?}"),
            })
            .collect();
        let configuration = MissionConfiguration {
            schema_version: 1,
            mission_id: "synthetic-peak-mission".into(),
            configuration_revision: "cfg-peak-r1".into(),
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
        let tasks = vec![
            MissionTaskRequirement {
                task_id: "required-life-support".into(),
                required: true,
                required_capabilities: BTreeSet::from([
                    SuitCapability::ThermalManagement,
                ]),
                required_component_ids: BTreeSet::from(["vehicle".into()]),
                required_connection_ids: BTreeSet::from(["suit-vehicle-power".into()]),
                required_environments: environments.clone(),
            },
            MissionTaskRequirement {
                task_id: "optional-science".into(),
                required: false,
                required_capabilities: BTreeSet::from([
                    SuitCapability::ElectrostaticChargeManagement,
                ]),
                required_component_ids: BTreeSet::from(["vehicle".into()]),
                required_connection_ids: BTreeSet::from(["suit-vehicle-power".into()]),
                required_environments: environments.clone(),
            },
        ];
        let (_, task_receipt) = evaluate_mission_tasks(&configuration, &tasks).unwrap();
        let task_report = evaluate_mission_tasks(&configuration, &tasks).unwrap().0;
        let plan = PeakResourceReservationPlan {
            schema_version: 1,
            mission_id: configuration.mission_id.clone(),
            configuration_revision: configuration.configuration_revision.clone(),
            expected_configuration_digest_sha256: task_report.configuration_digest_sha256,
            expected_task_plan_digest_sha256: task_report.task_plan_digest_sha256,
            minimum_evidence_level: SuitEvidenceLevel::BenchTested,
            reserve_basis_points: 1_000,
            pools: vec![PeakResourcePool {
                pool_id: "vehicle-peak-power".into(),
                resource_kind: PeakResourceKind::ElectricalPowerMilliwatts,
                provider_component_id: "vehicle".into(),
                provider_hardware_revision: vehicle_revision.into(),
                required_connection_ids: BTreeSet::from(["suit-vehicle-power".into()]),
                required_interface_kinds: BTreeSet::from([
                    crate::suit_interop::SuitInterfaceKind::ElectricalPower,
                ]),
                capacity: PeakQuantityInterval {
                    lower: 1_000,
                    nominal: 1_100,
                    upper: 1_200,
                },
                evidence_level: SuitEvidenceLevel::BenchTested,
                evidence_reference: "synthetic-capacity:power".into(),
                applicable_environments: environments,
            }],
            reservations: vec![
                reservation(
                    "life-support-peak",
                    "required-life-support",
                    "vehicle-peak-power",
                    0,
                    100,
                    500,
                ),
                reservation(
                    "science-peak",
                    "optional-science",
                    "vehicle-peak-power",
                    50,
                    150,
                    600,
                ),
            ],
        };
        (configuration, tasks, task_receipt, plan)
    }

    fn reservation(
        id: &str,
        task_id: &str,
        pool_id: &str,
        start_ms: u64,
        end_ms: u64,
        upper: u64,
    ) -> PeakResourceReservation {
        PeakResourceReservation {
            reservation_id: id.into(),
            task_id: task_id.into(),
            pool_id: pool_id.into(),
            resource_kind: PeakResourceKind::ElectricalPowerMilliwatts,
            window: MissionTimeWindow { start_ms, end_ms },
            demand: PeakQuantityInterval {
                lower: upper / 2,
                nominal: (upper * 3) / 4,
                upper,
            },
            evidence_level: SuitEvidenceLevel::BenchTested,
            evidence_reference: format!("synthetic-reservation:{id}"),
        }
    }

    #[test]
    fn overlapping_required_plus_optional_reservations_degrade_only() {
        let (configuration, tasks, task_receipt, plan) = fixture();
        let (report, receipt) =
            evaluate_peak_resource_reservations(&configuration, &tasks, &task_receipt, &plan)
                .unwrap();
        assert_eq!(report.status, PeakResourceReservationStatus::Degraded);
        assert_eq!(report.pools[0].usable_capacity, 900);
        let overlapping = report.windows.iter().find(|window| {
            window.window.start_ms == 50 && window.window.end_ms == 100
        }).unwrap();
        assert_eq!(overlapping.required_demand_upper, 500);
        assert_eq!(overlapping.optional_demand_upper, 600);
        assert_eq!(overlapping.total_shortfall, 200);
        assert_eq!(overlapping.status, PeakWindowStatus::OptionalOversubscribed);
        assert!(!report.represents_physical_qualification);
        assert!(verify_peak_resource_reservation_receipt(
            &configuration, &tasks, &task_receipt, &plan, &report, &receipt
        ));
    }

    #[test]
    fn required_oversubscription_blocks_even_when_optional_work_is_removed() {
        let (configuration, tasks, task_receipt, mut plan) = fixture();
        plan.reservations.retain(|item| item.task_id == "required-life-support");
        let mut second = reservation(
            "life-support-second-load",
            "required-life-support",
            "vehicle-peak-power",
            50,
            80,
            500,
        );
        second.demand.lower = 250;
        second.demand.nominal = 375;
        plan.reservations.push(second);
        let (report, _) =
            evaluate_peak_resource_reservations(&configuration, &tasks, &task_receipt, &plan)
                .unwrap();
        assert_eq!(report.status, PeakResourceReservationStatus::Blocked);
        assert!(report.findings.iter().any(|finding| {
            finding.code == PeakResourceFindingCode::RequiredOversubscription
                && finding.severity == PeakResourceFindingSeverity::Blocker
        }));
    }

    #[test]
    fn adjacent_half_open_reservations_do_not_overlap() {
        let (configuration, tasks, task_receipt, mut plan) = fixture();
        plan.reservations = vec![
            reservation("a", "required-life-support", "vehicle-peak-power", 0, 50, 700),
            reservation("b", "required-life-support", "vehicle-peak-power", 50, 100, 700),
        ];
        let (report, _) =
            evaluate_peak_resource_reservations(&configuration, &tasks, &task_receipt, &plan)
                .unwrap();
        assert_eq!(report.status, PeakResourceReservationStatus::MeetsDeclaredPeakResourcePolicy);
        assert!(report.windows.iter().all(|window| window.required_demand_upper == 700));
        assert!(report.windows.iter().all(|window| window.total_shortfall == 0));
        assert!(report.windows.iter().all(|window| window.active_reservation_count == 1));
        assert!(report.windows.iter().all(|window| window.active_reservation_ids.is_empty()));
    }

    #[test]
    fn reserve_reduces_capacity_and_rounds_up() {
        let (configuration, tasks, task_receipt, mut plan) = fixture();
        plan.reserve_basis_points = 1;
        plan.reservations.retain(|item| item.task_id == "required-life-support");
        plan.reservations[0].demand = PeakQuantityInterval {
            lower: 999,
            nominal: 999,
            upper: 999,
        };
        let (report, _) =
            evaluate_peak_resource_reservations(&configuration, &tasks, &task_receipt, &plan)
                .unwrap();
        assert_eq!(report.pools[0].reserve_quantity, 1);
        assert_eq!(report.pools[0].usable_capacity, 999);
        assert_eq!(report.status, PeakResourceReservationStatus::MeetsDeclaredPeakResourcePolicy);
    }

    #[test]
    fn empty_time_window_is_rejected() {
        let (configuration, tasks, task_receipt, mut plan) = fixture();
        plan.reservations[0].window.end_ms = plan.reservations[0].window.start_ms;
        assert_eq!(
            evaluate_peak_resource_reservations(&configuration, &tasks, &task_receipt, &plan)
                .unwrap_err(),
            PeakResourceReservationError::InvalidTimeWindow
        );
    }

    #[test]
    fn invalid_capacity_interval_is_rejected() {
        let (configuration, tasks, task_receipt, mut plan) = fixture();
        plan.pools[0].capacity = PeakQuantityInterval {
            lower: 1_000,
            nominal: 900,
            upper: 1_200,
        };
        assert_eq!(
            evaluate_peak_resource_reservations(&configuration, &tasks, &task_receipt, &plan)
                .unwrap_err(),
            PeakResourceReservationError::InvalidQuantityInterval
        );
    }

    #[test]
    fn mismatched_kind_is_not_coerced() {
        let (configuration, tasks, task_receipt, mut plan) = fixture();
        plan.reservations[0].resource_kind = PeakResourceKind::DataRateBytesPerSecond;
        assert_eq!(
            evaluate_peak_resource_reservations(&configuration, &tasks, &task_receipt, &plan)
                .unwrap_err(),
            PeakResourceReservationError::ResourceKindMismatch
        );
    }

    #[test]
    fn stale_task_receipt_and_revision_are_rejected() {
        let (configuration, tasks, mut task_receipt, mut plan) = fixture();
        task_receipt.receipt_digest_sha256.push_str("tampered");
        assert_eq!(
            evaluate_peak_resource_reservations(&configuration, &tasks, &task_receipt, &plan)
                .unwrap_err(),
            PeakResourceReservationError::TaskReceiptMismatch
        );
        let (_, fresh_receipt) = evaluate_mission_tasks(&configuration, &tasks).unwrap();
        plan.configuration_revision = "config-r2".into();
        assert_eq!(
            evaluate_peak_resource_reservations(&configuration, &tasks, &fresh_receipt, &plan)
                .unwrap_err(),
            PeakResourceReservationError::MissionIdentityMismatch
        );
    }

    #[test]
    fn provider_revision_mismatch_blocks_required_reservation() {
        let (configuration, tasks, task_receipt, mut plan) = fixture();
        plan.pools[0].provider_hardware_revision = "vehicle-r9".into();
        let (report, _) =
            evaluate_peak_resource_reservations(&configuration, &tasks, &task_receipt, &plan)
                .unwrap();
        assert_eq!(report.status, PeakResourceReservationStatus::Blocked);
        assert!(report.findings.iter().any(|finding| {
            finding.code == PeakResourceFindingCode::ProviderHardwareRevisionMismatch
                && finding.severity == PeakResourceFindingSeverity::Blocker
        }));
    }

    #[test]
    fn task_environment_must_be_covered_by_pool_scope() {
        let (configuration, tasks, task_receipt, mut plan) = fixture();
        plan.pools[0].applicable_environments =
            BTreeSet::from([SuitOperatingEnvironment::LunarSunlitSurface]);
        let (report, _) =
            evaluate_peak_resource_reservations(&configuration, &tasks, &task_receipt, &plan)
                .unwrap();
        assert_eq!(report.status, PeakResourceReservationStatus::Blocked);
        assert!(report.findings.iter().any(|finding| {
            finding.code == PeakResourceFindingCode::EnvironmentScopeMismatch
                && finding.severity == PeakResourceFindingSeverity::Blocker
        }));
    }

    #[test]
    fn resource_pool_requires_a_matching_declared_interface_kind() {
        let (configuration, tasks, task_receipt, mut plan) = fixture();
        plan.pools[0].required_interface_kinds =
            BTreeSet::from([crate::suit_interop::SuitInterfaceKind::GroundingAndEsd]);
        let (report, _) =
            evaluate_peak_resource_reservations(&configuration, &tasks, &task_receipt, &plan)
                .unwrap();
        assert_eq!(report.status, PeakResourceReservationStatus::Blocked);
        assert!(report.findings.iter().any(|finding| {
            finding.code == PeakResourceFindingCode::ProviderConnectionResourceKindMismatch
                && finding.severity == PeakResourceFindingSeverity::Blocker
        }));
    }

    #[test]
    fn missing_exact_provider_connection_is_a_blocker() {
        let (configuration, tasks, task_receipt, mut plan) = fixture();
        plan.pools[0].required_connection_ids =
            BTreeSet::from(["unlisted-power-contract".into()]);
        let (report, _) =
            evaluate_peak_resource_reservations(&configuration, &tasks, &task_receipt, &plan)
                .unwrap();
        assert_eq!(report.status, PeakResourceReservationStatus::Blocked);
        assert!(report.findings.iter().any(|finding| {
            finding.code == PeakResourceFindingCode::MissingProviderConnection
                && finding.severity == PeakResourceFindingSeverity::Blocker
        }));
    }

    #[test]
    fn duplicate_provider_resource_pool_is_rejected() {
        let (configuration, tasks, task_receipt, mut plan) = fixture();
        let mut duplicate = plan.pools[0].clone();
        duplicate.pool_id = "alias-pool".into();
        plan.pools.push(duplicate);
        assert_eq!(
            evaluate_peak_resource_reservations(&configuration, &tasks, &task_receipt, &plan)
                .unwrap_err(),
            PeakResourceReservationError::DuplicateProviderResourcePool
        );
    }

    #[test]
    fn reservation_order_is_invariant_and_receipt_detects_mutation() {
        let (configuration, tasks, task_receipt, mut plan) = fixture();
        let (report_a, receipt_a) =
            evaluate_peak_resource_reservations(&configuration, &tasks, &task_receipt, &plan)
                .unwrap();
        plan.reservations.reverse();
        plan.pools.reverse();
        let (report_b, receipt_b) =
            evaluate_peak_resource_reservations(&configuration, &tasks, &task_receipt, &plan)
                .unwrap();
        assert_eq!(report_a, report_b);
        assert_eq!(receipt_a, receipt_b);

        plan.reservations[0].window.end_ms += 1;
        assert!(!verify_peak_resource_reservation_receipt(
            &configuration, &tasks, &task_receipt, &plan, &report_a, &receipt_a
        ));
    }

    #[test]
    fn seeded_sweep_boundary_corpus_has_exact_peak_load_classification() {
        let (configuration, tasks, task_receipt, mut plan) = fixture();
        plan.reservations.retain(|item| item.task_id == "required-life-support");
        for seed in 0_u64..2_000 {
            let start = seed % 31;
            let duration = (seed.wrapping_mul(17) % 9) + 1;
            let amount_a = (seed.wrapping_mul(19) % 500) + 1;
            let amount_b = (seed.wrapping_mul(23).wrapping_add(13) % 500) + 1;
            plan.reservations = vec![
                reservation("seed-a", "required-life-support", "vehicle-peak-power",
                    start, start + duration, amount_a),
                reservation("seed-b", "required-life-support", "vehicle-peak-power",
                    start + duration / 2, start + duration + 1, amount_b),
            ];
            let (report, _) =
                evaluate_peak_resource_reservations(&configuration, &tasks, &task_receipt, &plan)
                    .unwrap();

            let overlap_start = start.max(start + duration / 2);
            let overlap_end = (start + duration).min(start + duration + 1);
            let overlaps = overlap_start < overlap_end;
            let maximum_expected = if overlaps {
                amount_a + amount_b
            } else {
                amount_a.max(amount_b)
            };
            assert_eq!(
                report.pools[0].peak_required_demand_upper,
                maximum_expected,
                "seed {seed}"
            );
            let expected_status = if maximum_expected > report.pools[0].usable_capacity {
                PeakResourceReservationStatus::Blocked
            } else {
                PeakResourceReservationStatus::MeetsDeclaredPeakResourcePolicy
            };
            assert_eq!(report.status, expected_status, "seed {seed}");
        }
    }
}
