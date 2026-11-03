//! System-window accounting for overlapping InstantClean service requests.
//!
//! A system window is measured on one monotonic clock domain. Request durations
//! are never added to obtain elapsed time: throughput uses the fixed wall-clock
//! observation window, while concurrency is computed from the union of request
//! intervals. Item releases require an upstream service report that qualifies
//! that exact item using independent endpoint assays.
//!
//! System resource meters are authoritative for system totals. Per-request
//! service-ledger resource totals are deliberately not summed here because
//! shared preheat, standby, fill, purge or parent/branch meters could otherwise
//! be counted more than once. This module validates declared meter scopes and
//! overlap; it cannot prove that a physical boundary declaration is truthful.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::service::ServiceAccountingReport;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SystemResourceKind {
    ApplianceWallElectricity,
    UpstreamHotWaterEnergy,
    FreshWater,
    DetergentMass,
    RinseAidMass,
    ActiveOperatorTime,
}

impl SystemResourceKind {
    pub const fn required_unit(self) -> &'static str {
        match self {
            Self::ApplianceWallElectricity | Self::UpstreamHotWaterEnergy => "kWh",
            Self::FreshWater => "L",
            Self::DetergentMass | Self::RinseAidMass => "g",
            Self::ActiveOperatorTime => "s",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowRequestState {
    Completed { closed_at_s: u64 },
    /// Terminal cancellation, either while queued or after physical work starts.
    Cancelled { cancelled_at_s: u64 },
    /// The observation is right-censored at the system-window boundary.
    Unresolved { observed_until_s: u64 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ServiceWindowRequest {
    pub service_request_id: String,
    pub clock_domain_id: String,
    pub accepted_at_s: u64,
    /// None means the request is still queued and has not started.
    pub started_at_s: Option<u64>,
    pub state: WindowRequestState,
    /// Present once the request has produced at least one physical attempt.
    pub service_report: Option<ServiceAccountingReport>,
    /// Item identities are scoped to this service request.
    pub requested_item_ids: Vec<String>,
    pub compartment_id: String,
    pub compartment_configuration_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ItemReleaseEvent {
    pub service_request_id: String,
    pub item_id: String,
    pub released_at_s: u64,
    /// Link to the independently qualified item/endpoint evidence.
    pub evidence_ref: String,
    pub evidence_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SystemWindowResourceMeasurement {
    pub source_window_id: String,
    pub clock_domain_id: String,
    pub measurement_id: String,
    pub kind: SystemResourceKind,
    /// Stable declared physical scope. For active operator time use a
    /// pseudonymous per-person identity, not a shared team identity.
    pub measurement_boundary_id: String,
    pub value: f64,
    pub unit: String,
    pub start_at_s: u64,
    pub end_at_s: u64,
    pub instrument_id: Option<String>,
    pub calibration_id: Option<String>,
    pub method_id: String,
    pub evidence_ref: String,
    pub evidence_sha256: String,
    /// If this meter already includes downstream boundaries, list their IDs.
    /// Overlapping parent and child measurements are rejected. This declaration
    /// still requires an independently audited physical meter/topology map.
    pub covers_boundary_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemWindowSpec {
    pub window_id: String,
    pub clock_domain_id: String,
    /// Observation interval is half-open: [start_at_s, end_at_s).
    pub start_at_s: u64,
    pub end_at_s: u64,
    pub required_resources: Vec<SystemResourceKind>,
    pub machine_unit_id: String,
    pub system_configuration_sha256: String,
    pub utility_topology_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompartmentWindowStats {
    pub compartment_id: String,
    pub compartment_configuration_sha256: String,
    pub request_record_count: usize,
    pub requests_accepted_in_window: usize,
    pub requests_started_in_window: usize,
    pub requests_completed_in_window: usize,
    pub cancelled_requests_in_window: usize,
    pub unresolved_requests_at_window_end: usize,
    pub verified_item_release_count: usize,
    pub active_request_seconds: u64,
    pub maximum_concurrent_requests: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SystemWindowReport {
    pub window_id: String,
    pub machine_unit_id: String,
    pub system_configuration_sha256: String,
    pub utility_topology_id: String,
    pub elapsed_wall_seconds: u64,
    pub request_record_count: usize,
    pub physical_attempt_count_total: usize,
    pub rewash_attempt_count_total: usize,
    pub requests_with_rewash_attempts_total: usize,
    pub item_attempt_visits_total: usize,
    pub reattempted_item_visits_total: usize,
    pub carry_in_requests_at_window_start: usize,
    pub requests_accepted_in_window: usize,
    pub requests_started_in_window: usize,
    pub requests_completed_in_window: usize,
    pub cancelled_requests_in_window: usize,
    pub cancelled_items_in_accepted_cohort: usize,
    pub unresolved_requests_at_window_end: usize,
    pub requested_items_in_accepted_cohort: usize,
    pub items_without_verified_qualification_in_accepted_unresolved_requests: usize,
    pub verified_item_release_count: usize,
    pub unqualified_items_in_service_reports: usize,
    pub first_pass_verified_items_in_completed_cohort: usize,
    pub items_in_completed_cohort: usize,
    pub first_pass_yield: Option<f64>,
    pub verified_items_per_hour: f64,
    pub time_to_first_release_seconds: Option<u64>,
    pub active_request_seconds: u64,
    pub maximum_concurrent_requests: usize,
    /// Percentiles use nearest-rank over completed requests accepted in-window.
    /// Right-censored requests are reported separately and never imputed.
    pub request_latency_p50_seconds: Option<u64>,
    pub request_latency_p95_seconds: Option<u64>,
    pub queue_wait_p50_seconds: Option<u64>,
    pub queue_wait_p95_seconds: Option<u64>,
    pub total_resource_measurement_count: usize,
    pub resource_totals: Vec<(SystemResourceKind, f64, &'static str)>,
    /// Window resource intensity per release; not a per-request lifecycle cost.
    pub window_resources_per_verified_release: Option<Vec<(SystemResourceKind, f64, &'static str)>>,
    pub compartment_stats: Vec<CompartmentWindowStats>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SystemWindowError {
    InvalidWindow,
    InvalidIdentity(&'static str),
    InvalidSystemIdentity(&'static str),
    CompartmentConfigurationMismatch { compartment_id: String },
    InvalidClockDomain { request_id: String },
    InvalidResourceClockDomain { measurement_id: String },
    EmptyRequests,
    DuplicateRequestId(String),
    EmptyRequestedItemSet(String),
    DuplicateRequestedItem { request_id: String, item_id: String },
    RequestOutsideWindow(String),
    RequestFinishBeforeStart(String),
    CancelledRequestBeforeAcceptance(String),
    CancelledRequestFinishBeforeStart(String),
    RequestStartedBeforeAcceptance(String),
    StartedRequestMissingServiceReport(String),
    QueuedRequestHasServiceReport(String),
    CompletedRequestNotStarted(String),
    CompletedRequestMissingServiceReport(String),
    UnresolvedRequestNotCensoredAtWindowEnd(String),
    ServiceRequestIdentityMismatch(String),
    InconsistentServiceReport { request_id: String, field: &'static str },
    ServiceReportItemSetMismatch(String),
    ReleaseForUnknownRequest(String),
    ReleaseForUnrequestedItem { request_id: String, item_id: String },
    ReleaseNotQualified { request_id: String, item_id: String },
    DuplicateItemRelease { request_id: String, item_id: String },
    ReleaseOutsideService { request_id: String, item_id: String },
    InvalidEvidenceReference(&'static str),
    InvalidEvidenceDigest(&'static str),
    EmptyRequiredResources,
    DuplicateRequiredResource(SystemResourceKind),
    MissingRequiredResource(SystemResourceKind),
    DuplicateMeasurementId(String),
    InvalidResourceInterval(String),
    ResourceOutsideWindow(String),
    InvalidResourceValue(String),
    ResourceUnitMismatch { measurement_id: String, expected: String, observed: String },
    InvalidResourceCalibration(String),
    OverlappingResourceBoundaryIntervals {
        kind: SystemResourceKind,
        boundary_id: String,
        first_measurement_id: String,
        second_measurement_id: String,
    },
    OverlappingInstrumentStreams {
        kind: SystemResourceKind,
        instrument_id: String,
        first_measurement_id: String,
        second_measurement_id: String,
    },
    InvalidCoveredBoundary(String),
    NestedBoundaryDoubleCount {
        kind: SystemResourceKind,
        parent_measurement_id: String,
        child_measurement_id: String,
    },
    ResourceTotalOverflow(SystemResourceKind),
    CounterOverflow(&'static str),
}

fn canonical_non_empty(value: &str) -> bool {
    !value.trim().is_empty() && value.trim() == value
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn percentile_nearest_rank(values: &mut [u64], percentile: f64) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    let rank = ((percentile * values.len() as f64).ceil() as usize)
        .max(1)
        .min(values.len());
    Some(values[rank - 1])
}

#[derive(Debug, Default)]
struct CompartmentAccumulator {
    configuration_sha256: String,
    request_record_count: usize,
    requests_accepted_in_window: usize,
    requests_started_in_window: usize,
    requests_completed_in_window: usize,
    cancelled_requests_in_window: usize,
    unresolved_requests_at_window_end: usize,
    verified_item_release_count: usize,
    active_request_seconds: u64,
    concurrency_events: Vec<(u64, i8)>,
}

fn maximum_concurrency(events: &mut [(u64, i8)]) -> Result<usize, SystemWindowError> {
    events.sort_unstable_by_key(|(timestamp, delta)| (*timestamp, *delta));
    let mut current = 0_isize;
    let mut maximum = 0_usize;
    for (_, delta) in events.iter() {
        current += *delta as isize;
        if current < 0 {
            return Err(SystemWindowError::InvalidWindow);
        }
        maximum = maximum.max(current as usize);
    }
    if current != 0 {
        return Err(SystemWindowError::InvalidWindow);
    }
    Ok(maximum)
}

fn build_compartment_stats(
    spec: &SystemWindowSpec,
    requests: &[ServiceWindowRequest],
    releases: &[ItemReleaseEvent],
) -> Result<Vec<CompartmentWindowStats>, SystemWindowError> {
    let mut accumulators: BTreeMap<String, CompartmentAccumulator> = BTreeMap::new();
    let mut request_by_id = HashMap::with_capacity(requests.len());

    for request in requests {
        request_by_id.insert(request.service_request_id.as_str(), request);
        let stats = accumulators
            .entry(request.compartment_id.clone())
            .or_insert_with(|| CompartmentAccumulator {
                configuration_sha256: request.compartment_configuration_sha256.clone(),
                ..CompartmentAccumulator::default()
            });
        if stats.configuration_sha256 != request.compartment_configuration_sha256 {
            return Err(SystemWindowError::CompartmentConfigurationMismatch {
                compartment_id: request.compartment_id.clone(),
            });
        }
        stats.request_record_count = stats.request_record_count.checked_add(1)
            .ok_or(SystemWindowError::CounterOverflow("compartment_request_record_count"))?;

        let accepted_in_window =
            request.accepted_at_s >= spec.start_at_s && request.accepted_at_s < spec.end_at_s;
        if accepted_in_window {
            stats.requests_accepted_in_window = stats.requests_accepted_in_window.checked_add(1)
                .ok_or(SystemWindowError::CounterOverflow("compartment_accepted_count"))?;
        }

        if let Some(started_at_s) = request.started_at_s {
            if started_at_s >= spec.start_at_s && started_at_s < spec.end_at_s {
                stats.requests_started_in_window = stats.requests_started_in_window.checked_add(1)
                    .ok_or(SystemWindowError::CounterOverflow("compartment_started_count"))?;
            }
            let service_end = match request.state {
                WindowRequestState::Completed { closed_at_s } => closed_at_s,
                WindowRequestState::Cancelled { cancelled_at_s } => cancelled_at_s,
                WindowRequestState::Unresolved { .. } => spec.end_at_s,
            };
            let start = started_at_s.max(spec.start_at_s);
            let end = service_end.min(spec.end_at_s);
            if end > start {
                let duration = end - start;
                stats.active_request_seconds = stats.active_request_seconds.checked_add(duration)
                    .ok_or(SystemWindowError::CounterOverflow("compartment_active_request_seconds"))?;
                stats.concurrency_events.push((start, 1));
                stats.concurrency_events.push((end, -1));
            }
        }

        match request.state {
            WindowRequestState::Completed { closed_at_s }
                if closed_at_s >= spec.start_at_s && closed_at_s < spec.end_at_s =>
            {
                stats.requests_completed_in_window = stats.requests_completed_in_window.checked_add(1)
                    .ok_or(SystemWindowError::CounterOverflow("compartment_completed_count"))?;
            }
            WindowRequestState::Cancelled { cancelled_at_s }
                if cancelled_at_s >= spec.start_at_s && cancelled_at_s < spec.end_at_s =>
            {
                stats.cancelled_requests_in_window = stats.cancelled_requests_in_window.checked_add(1)
                    .ok_or(SystemWindowError::CounterOverflow("compartment_cancelled_count"))?;
            }
            WindowRequestState::Unresolved { .. } => {
                stats.unresolved_requests_at_window_end = stats.unresolved_requests_at_window_end.checked_add(1)
                    .ok_or(SystemWindowError::CounterOverflow("compartment_unresolved_count"))?;
            }
            _ => {}
        }
    }

    for release in releases {
        let request = request_by_id.get(release.service_request_id.as_str())
            .ok_or_else(|| SystemWindowError::ReleaseForUnknownRequest(
                release.service_request_id.clone()
            ))?;
        let stats = accumulators.get_mut(&request.compartment_id)
            .ok_or(SystemWindowError::InvalidIdentity("release_compartment"))?;
        stats.verified_item_release_count = stats.verified_item_release_count.checked_add(1)
            .ok_or(SystemWindowError::CounterOverflow("compartment_release_count"))?;
    }

    accumulators.into_iter().map(|(compartment_id, mut stats)| {
        let maximum_concurrent_requests = maximum_concurrency(&mut stats.concurrency_events)?;
        Ok(CompartmentWindowStats {
            compartment_id,
            compartment_configuration_sha256: stats.configuration_sha256,
            request_record_count: stats.request_record_count,
            requests_accepted_in_window: stats.requests_accepted_in_window,
            requests_started_in_window: stats.requests_started_in_window,
            requests_completed_in_window: stats.requests_completed_in_window,
            cancelled_requests_in_window: stats.cancelled_requests_in_window,
            unresolved_requests_at_window_end: stats.unresolved_requests_at_window_end,
            verified_item_release_count: stats.verified_item_release_count,
            active_request_seconds: stats.active_request_seconds,
            maximum_concurrent_requests,
        })
    }).collect()
}

fn validate_service_report(
    request: &ServiceWindowRequest,
    completed: bool,
) -> Result<(), SystemWindowError> {
    let report = request.service_report.as_ref().ok_or_else(|| {
        SystemWindowError::CompletedRequestMissingServiceReport(request.service_request_id.clone())
    })?;

    if report.service_request_id != request.service_request_id {
        return Err(SystemWindowError::ServiceRequestIdentityMismatch(
            request.service_request_id.clone(),
        ));
    }

    let mut verified = BTreeSet::new();
    for item_id in &report.verified_successful_item_ids {
        if !canonical_non_empty(item_id) || !verified.insert(item_id.as_str()) {
            return Err(SystemWindowError::InconsistentServiceReport {
                request_id: request.service_request_id.clone(),
                field: "verified_successful_item_ids",
            });
        }
    }
    if report.verified_successful_item_count != verified.len() {
        return Err(SystemWindowError::InconsistentServiceReport {
            request_id: request.service_request_id.clone(),
            field: "verified_successful_item_count",
        });
    }

    let mut unqualified = BTreeSet::new();
    for item_id in &report.unqualified_item_ids {
        if !canonical_non_empty(item_id)
            || verified.contains(item_id.as_str())
            || !unqualified.insert(item_id.as_str())
        {
            return Err(SystemWindowError::InconsistentServiceReport {
                request_id: request.service_request_id.clone(),
                field: "unqualified_item_ids",
            });
        }
    }
    if report.unique_item_count != verified.len() + unqualified.len() {
        return Err(SystemWindowError::InconsistentServiceReport {
            request_id: request.service_request_id.clone(),
            field: "unique_item_count",
        });
    }

    let mut first_pass = BTreeSet::new();
    for item_id in &report.first_pass_verified_item_ids {
        if !verified.contains(item_id.as_str()) || !first_pass.insert(item_id.as_str()) {
            return Err(SystemWindowError::InconsistentServiceReport {
                request_id: request.service_request_id.clone(),
                field: "first_pass_verified_item_ids",
            });
        }
    }
    if report.first_pass_verified_item_count != first_pass.len() {
        return Err(SystemWindowError::InconsistentServiceReport {
            request_id: request.service_request_id.clone(),
            field: "first_pass_verified_item_count",
        });
    }
    if report.attempt_count == 0 || report.physical_run_count != report.attempt_count {
        return Err(SystemWindowError::InconsistentServiceReport {
            request_id: request.service_request_id.clone(),
            field: "attempt_count/physical_run_count",
        });
    }
    if report.rewash_attempt_count > report.attempt_count
        || report.rewash_attempt_count > report.reattempted_item_visits
        || (report.rewash_attempt_count == 0) != (report.reattempted_item_visits == 0)
    {
        return Err(SystemWindowError::InconsistentServiceReport {
            request_id: request.service_request_id.clone(),
            field: "rewash_attempt_count",
        });
    }
    if report.item_attempt_visits < report.unique_item_count
        || report.reattempted_item_visits != report.item_attempt_visits - report.unique_item_count
    {
        return Err(SystemWindowError::InconsistentServiceReport {
            request_id: request.service_request_id.clone(),
            field: "item_attempt_visits",
        });
    }

    let requested: BTreeSet<&str> = request.requested_item_ids.iter().map(String::as_str).collect();
    if !verified.iter().all(|item| requested.contains(item))
        || !unqualified.iter().all(|item| requested.contains(item))
    {
        return Err(SystemWindowError::ServiceReportItemSetMismatch(
            request.service_request_id.clone(),
        ));
    }
    if completed && report.unique_item_count != request.requested_item_ids.len() {
        return Err(SystemWindowError::ServiceReportItemSetMismatch(
            request.service_request_id.clone(),
        ));
    }
    Ok(())
}

fn validate_request(
    request: &ServiceWindowRequest,
    spec: &SystemWindowSpec,
) -> Result<(), SystemWindowError> {
    if !canonical_non_empty(&request.service_request_id) {
        return Err(SystemWindowError::InvalidIdentity("service_request_id"));
    }
    if !canonical_non_empty(&request.compartment_id) {
        return Err(SystemWindowError::InvalidIdentity("compartment_id"));
    }
    if !valid_digest(&request.compartment_configuration_sha256) {
        return Err(SystemWindowError::InvalidEvidenceDigest("compartment_configuration_sha256"));
    }
    if !canonical_non_empty(&request.clock_domain_id)
        || request.clock_domain_id != spec.clock_domain_id
    {
        return Err(SystemWindowError::InvalidClockDomain {
            request_id: request.service_request_id.clone(),
        });
    }
    if request.accepted_at_s >= spec.end_at_s {
        return Err(SystemWindowError::RequestOutsideWindow(
            request.service_request_id.clone(),
        ));
    }

    if request.requested_item_ids.is_empty() {
        return Err(SystemWindowError::EmptyRequestedItemSet(
            request.service_request_id.clone(),
        ));
    }
    let mut requested = HashSet::new();
    for item_id in &request.requested_item_ids {
        if !canonical_non_empty(item_id) {
            return Err(SystemWindowError::InvalidIdentity("requested_item_id"));
        }
        if !requested.insert(item_id.as_str()) {
            return Err(SystemWindowError::DuplicateRequestedItem {
                request_id: request.service_request_id.clone(),
                item_id: item_id.clone(),
            });
        }
    }

    if let Some(started_at_s) = request.started_at_s {
        if started_at_s < request.accepted_at_s {
            return Err(SystemWindowError::RequestStartedBeforeAcceptance(
                request.service_request_id.clone(),
            ));
        }
        if started_at_s >= spec.end_at_s {
            return Err(SystemWindowError::RequestOutsideWindow(
                request.service_request_id.clone(),
            ));
        }
        if request.service_report.is_none() {
            return Err(SystemWindowError::StartedRequestMissingServiceReport(
                request.service_request_id.clone(),
            ));
        }
    } else if request.service_report.is_some() {
        return Err(SystemWindowError::QueuedRequestHasServiceReport(
            request.service_request_id.clone(),
        ));
    }

    match request.state {
        WindowRequestState::Completed { closed_at_s } => {
            let started_at_s = request.started_at_s.ok_or_else(|| {
                SystemWindowError::CompletedRequestNotStarted(request.service_request_id.clone())
            })?;
            if closed_at_s < started_at_s || closed_at_s >= spec.end_at_s {
                return Err(SystemWindowError::RequestFinishBeforeStart(
                    request.service_request_id.clone(),
                ));
            }
            if closed_at_s <= spec.start_at_s {
                return Err(SystemWindowError::RequestOutsideWindow(
                    request.service_request_id.clone(),
                ));
            }
            validate_service_report(request, true)?;
        }
        WindowRequestState::Cancelled { cancelled_at_s } => {
            if cancelled_at_s < request.accepted_at_s {
                return Err(SystemWindowError::CancelledRequestBeforeAcceptance(
                    request.service_request_id.clone(),
                ));
            }
            if cancelled_at_s >= spec.end_at_s {
                return Err(SystemWindowError::RequestOutsideWindow(
                    request.service_request_id.clone(),
                ));
            }
            if let Some(started_at_s) = request.started_at_s {
                if cancelled_at_s < started_at_s {
                    return Err(SystemWindowError::CancelledRequestFinishBeforeStart(
                        request.service_request_id.clone(),
                    ));
                }
                validate_service_report(request, false)?;
            }
        }
        WindowRequestState::Unresolved { observed_until_s } => {
            if observed_until_s != spec.end_at_s {
                return Err(SystemWindowError::UnresolvedRequestNotCensoredAtWindowEnd(
                    request.service_request_id.clone(),
                ));
            }
            if request.started_at_s.is_some() {
                validate_service_report(request, false)?;
            }
        }
    }
    Ok(())
}

/// Validate one system observation window.
///
/// System-level meter rows are the authoritative resource numerator. The
/// request-level reports supply item qualification, first-pass yield and
/// service-attempt context, but their resource totals are not added again.
/// Every required resource dimension needs an explicit row, including an
/// independently measured zero.
pub fn account_system_window(
    spec: &SystemWindowSpec,
    requests: &[ServiceWindowRequest],
    releases: &[ItemReleaseEvent],
    resources: &[SystemWindowResourceMeasurement],
) -> Result<SystemWindowReport, SystemWindowError> {
    if !canonical_non_empty(&spec.window_id)
        || !canonical_non_empty(&spec.clock_domain_id)
        || spec.end_at_s <= spec.start_at_s
    {
        return Err(SystemWindowError::InvalidWindow);
    }
    if !canonical_non_empty(&spec.machine_unit_id) {
        return Err(SystemWindowError::InvalidSystemIdentity("machine_unit_id"));
    }
    if !valid_digest(&spec.system_configuration_sha256) {
        return Err(SystemWindowError::InvalidSystemIdentity("system_configuration_sha256"));
    }
    if !canonical_non_empty(&spec.utility_topology_id) {
        return Err(SystemWindowError::InvalidSystemIdentity("utility_topology_id"));
    }
    if requests.is_empty() {
        return Err(SystemWindowError::EmptyRequests);
    }
    if spec.required_resources.is_empty() {
        return Err(SystemWindowError::EmptyRequiredResources);
    }

    let elapsed_wall_seconds = spec.end_at_s - spec.start_at_s;
    let mut required_resources = BTreeSet::new();
    for kind in &spec.required_resources {
        if !required_resources.insert(*kind) {
            return Err(SystemWindowError::DuplicateRequiredResource(*kind));
        }
    }

    let mut request_by_id = HashMap::with_capacity(requests.len());
    for request in requests {
        validate_request(request, spec)?;
        if request_by_id.insert(request.service_request_id.as_str(), request).is_some() {
            return Err(SystemWindowError::DuplicateRequestId(
                request.service_request_id.clone(),
            ));
        }
    }

    let mut carry_in_count = 0_usize;
    let mut accepted_count = 0_usize;
    let mut started_count = 0_usize;
    let mut physical_attempt_count_total = 0_usize;
    let mut rewash_attempt_count_total = 0_usize;
    let mut requests_with_rewash_attempts_total = 0_usize;
    let mut item_attempt_visits_total = 0_usize;
    let mut reattempted_item_visits_total = 0_usize;
    let mut completed_count = 0_usize;
    let mut cancelled_count = 0_usize;
    let mut cancelled_item_count = 0_usize;
    let mut unresolved_count = 0_usize;
    let mut requested_item_count = 0_usize;
    let mut unresolved_item_count = 0_usize;
    let mut unqualified_item_count = 0_usize;
    let mut first_pass_item_count = 0_usize;
    let mut completed_cohort_item_count = 0_usize;
    let mut active_request_seconds = 0_u64;
    let mut latency_values = Vec::new();
    let mut queue_wait_values = Vec::new();
    let mut concurrency_events: Vec<(u64, i8)> = Vec::new();

    for request in requests {
        let request_end_at_s = match request.state {
            WindowRequestState::Completed { closed_at_s } => closed_at_s,
            WindowRequestState::Cancelled { cancelled_at_s } => cancelled_at_s,
            WindowRequestState::Unresolved { .. } => spec.end_at_s,
        };
        if request.accepted_at_s < spec.start_at_s && request_end_at_s > spec.start_at_s {
            carry_in_count = carry_in_count.checked_add(1)
                .ok_or(SystemWindowError::CounterOverflow("carry_in_requests"))?;
        }
        let accepted_in_window =
            request.accepted_at_s >= spec.start_at_s && request.accepted_at_s < spec.end_at_s;
        if accepted_in_window {
            accepted_count = accepted_count.checked_add(1)
                .ok_or(SystemWindowError::CounterOverflow("requests_accepted_in_window"))?;
            requested_item_count = requested_item_count.checked_add(request.requested_item_ids.len())
                .ok_or(SystemWindowError::CounterOverflow("requested_item_count"))?;
        }

        if let Some(started_at_s) = request.started_at_s {
            if started_at_s >= spec.start_at_s && started_at_s < spec.end_at_s {
                started_count = started_count.checked_add(1)
                    .ok_or(SystemWindowError::CounterOverflow("requests_started_in_window"))?;
            }
            if accepted_in_window {
                queue_wait_values.push(started_at_s.saturating_sub(request.accepted_at_s));
            }

            let service_end = match request.state {
                WindowRequestState::Completed { closed_at_s } => closed_at_s,
                WindowRequestState::Cancelled { cancelled_at_s } => cancelled_at_s,
                WindowRequestState::Unresolved { .. } => spec.end_at_s,
            };
            let intersection_start = started_at_s.max(spec.start_at_s);
            let intersection_end = service_end.min(spec.end_at_s);
            if intersection_end > intersection_start {
                let duration = intersection_end - intersection_start;
                active_request_seconds = active_request_seconds.checked_add(duration)
                    .ok_or(SystemWindowError::CounterOverflow("active_request_seconds"))?;
                concurrency_events.push((intersection_start, 1));
                concurrency_events.push((intersection_end, -1));
            }
        }

        match request.state {
            WindowRequestState::Completed { closed_at_s } => {
                if closed_at_s >= spec.start_at_s && closed_at_s < spec.end_at_s {
                    completed_count = completed_count.checked_add(1)
                        .ok_or(SystemWindowError::CounterOverflow("requests_completed_in_window"))?;
                }
                if accepted_in_window {
                    let latency = closed_at_s - request.accepted_at_s;
                    latency_values.push(latency);
                    let report = request.service_report.as_ref().ok_or_else(|| {
                        SystemWindowError::CompletedRequestMissingServiceReport(
                            request.service_request_id.clone()
                        )
                    })?;
                    first_pass_item_count = first_pass_item_count.checked_add(
                        report.first_pass_verified_item_count
                    ).ok_or(SystemWindowError::CounterOverflow("first_pass_item_count"))?;
                    completed_cohort_item_count = completed_cohort_item_count.checked_add(
                        request.requested_item_ids.len()
                    ).ok_or(SystemWindowError::CounterOverflow("completed_cohort_item_count"))?;
                }
            }
            WindowRequestState::Cancelled { cancelled_at_s } => {
                if cancelled_at_s >= spec.start_at_s && cancelled_at_s < spec.end_at_s {
                    cancelled_count = cancelled_count.checked_add(1)
                        .ok_or(SystemWindowError::CounterOverflow("cancelled_requests"))?;
                }
                if accepted_in_window {
                    cancelled_item_count = cancelled_item_count.checked_add(request.requested_item_ids.len())
                        .ok_or(SystemWindowError::CounterOverflow("cancelled_items"))?;
                }
            }
            WindowRequestState::Unresolved { .. } => {
                unresolved_count = unresolved_count.checked_add(1)
                    .ok_or(SystemWindowError::CounterOverflow("unresolved_requests"))?;
                if accepted_in_window {
                    let already_qualified = request.service_report.as_ref()
                        .map(|report| report.verified_successful_item_count)
                        .unwrap_or(0);
                    let pending_qualification = request.requested_item_ids.len()
                        .checked_sub(already_qualified)
                        .ok_or(SystemWindowError::CounterOverflow("unresolved_item_count"))?;
                    unresolved_item_count = unresolved_item_count.checked_add(
                        pending_qualification
                    ).ok_or(SystemWindowError::CounterOverflow("unresolved_item_count"))?;
                }
            }
        }

        if let Some(report) = &request.service_report {
            physical_attempt_count_total = physical_attempt_count_total.checked_add(report.attempt_count)
                .ok_or(SystemWindowError::CounterOverflow("physical_attempt_count_total"))?;
            rewash_attempt_count_total = rewash_attempt_count_total.checked_add(report.rewash_attempt_count)
                .ok_or(SystemWindowError::CounterOverflow("rewash_attempt_count_total"))?;
            item_attempt_visits_total = item_attempt_visits_total.checked_add(report.item_attempt_visits)
                .ok_or(SystemWindowError::CounterOverflow("item_attempt_visits_total"))?;
            reattempted_item_visits_total = reattempted_item_visits_total.checked_add(report.reattempted_item_visits)
                .ok_or(SystemWindowError::CounterOverflow("reattempted_item_visits_total"))?;
            if report.rewash_attempt_count > 0 {
                requests_with_rewash_attempts_total = requests_with_rewash_attempts_total.checked_add(1)
                    .ok_or(SystemWindowError::CounterOverflow("requests_with_rewash_attempts_total"))?;
            }
            unqualified_item_count = unqualified_item_count.checked_add(report.unqualified_item_ids.len())
                .ok_or(SystemWindowError::CounterOverflow("unqualified_item_count"))?;
        }
    }

    concurrency_events.sort_unstable_by_key(|(timestamp, delta)| (*timestamp, *delta));
    let mut current_concurrency = 0_isize;
    let mut maximum_concurrent_requests = 0_usize;
    for (_, delta) in concurrency_events {
        current_concurrency += delta as isize;
        if current_concurrency < 0 {
            return Err(SystemWindowError::InvalidWindow);
        }
        maximum_concurrent_requests = maximum_concurrent_requests.max(current_concurrency as usize);
    }
    if current_concurrency != 0 {
        return Err(SystemWindowError::InvalidWindow);
    }

    let mut released_items = HashSet::new();
    let mut first_release_at_s: Option<u64> = None;
    for release in releases {
        if !canonical_non_empty(&release.service_request_id)
            || !canonical_non_empty(&release.item_id)
            || !canonical_non_empty(&release.evidence_ref)
        {
            return Err(SystemWindowError::InvalidIdentity("item_release"));
        }
        if !valid_digest(&release.evidence_sha256) {
            return Err(SystemWindowError::InvalidEvidenceDigest("item_release"));
        }
        if release.released_at_s < spec.start_at_s || release.released_at_s >= spec.end_at_s {
            return Err(SystemWindowError::ReleaseOutsideService {
                request_id: release.service_request_id.clone(),
                item_id: release.item_id.clone(),
            });
        }
        let request = request_by_id.get(release.service_request_id.as_str())
            .ok_or_else(|| SystemWindowError::ReleaseForUnknownRequest(
                release.service_request_id.clone()
            ))?;
        if !request.requested_item_ids.iter().any(|item| item == &release.item_id) {
            return Err(SystemWindowError::ReleaseForUnrequestedItem {
                request_id: release.service_request_id.clone(),
                item_id: release.item_id.clone(),
            });
        }
        let started_at_s = request.started_at_s.ok_or_else(|| {
            SystemWindowError::ReleaseOutsideService {
                request_id: release.service_request_id.clone(),
                item_id: release.item_id.clone(),
            }
        })?;
        if release.released_at_s < request.accepted_at_s
            || release.released_at_s < started_at_s
        {
            return Err(SystemWindowError::ReleaseOutsideService {
                request_id: release.service_request_id.clone(),
                item_id: release.item_id.clone(),
            });
        }
        match request.state {
            WindowRequestState::Completed { closed_at_s } if release.released_at_s > closed_at_s => {
                return Err(SystemWindowError::ReleaseOutsideService {
                    request_id: release.service_request_id.clone(),
                    item_id: release.item_id.clone(),
                });
            }
            WindowRequestState::Cancelled { cancelled_at_s }
                if release.released_at_s >= cancelled_at_s => {
                return Err(SystemWindowError::ReleaseOutsideService {
                    request_id: release.service_request_id.clone(),
                    item_id: release.item_id.clone(),
                });
            }
            WindowRequestState::Unresolved { .. } => {}
            _ => {}
        }
        let report = request.service_report.as_ref().ok_or_else(|| {
            SystemWindowError::ReleaseNotQualified {
                request_id: release.service_request_id.clone(),
                item_id: release.item_id.clone(),
            }
        })?;
        if !report.verified_successful_item_ids.iter().any(|item| item == &release.item_id) {
            return Err(SystemWindowError::ReleaseNotQualified {
                request_id: release.service_request_id.clone(),
                item_id: release.item_id.clone(),
            });
        }
        if !released_items.insert((release.service_request_id.as_str(), release.item_id.as_str())) {
            return Err(SystemWindowError::DuplicateItemRelease {
                request_id: release.service_request_id.clone(),
                item_id: release.item_id.clone(),
            });
        }
        first_release_at_s = Some(first_release_at_s.map_or(
            release.released_at_s,
            |previous| previous.min(release.released_at_s),
        ));
    }

    let mut measurement_ids = HashSet::new();
    let mut present_resources = BTreeSet::new();
    let mut resource_totals: BTreeMap<SystemResourceKind, f64> = BTreeMap::new();
    let mut boundary_intervals: BTreeMap<
        (SystemResourceKind, String),
        Vec<(u64, u64, String)>,
    > = BTreeMap::new();
    // instrument_id identifies one calibrated metering stream, not a
    // multi-channel DAQ chassis. A single stream cannot be summed over
    // overlapping intervals assigned to different declared boundaries.
    let mut instrument_intervals: BTreeMap<
        (SystemResourceKind, String),
        Vec<(u64, u64, String)>,
    > = BTreeMap::new();

    for measurement in resources {
        if measurement.source_window_id != spec.window_id {
            return Err(SystemWindowError::InvalidIdentity("resource_source_window_id"));
        }
        if !canonical_non_empty(&measurement.clock_domain_id)
            || measurement.clock_domain_id != spec.clock_domain_id
        {
            return Err(SystemWindowError::InvalidResourceClockDomain {
                measurement_id: measurement.measurement_id.clone(),
            });
        }
        for (field, value) in [
            ("measurement_id", measurement.measurement_id.as_str()),
            ("measurement_boundary_id", measurement.measurement_boundary_id.as_str()),
            ("resource_method_id", measurement.method_id.as_str()),
            ("resource_evidence_ref", measurement.evidence_ref.as_str()),
        ] {
            if !canonical_non_empty(value) {
                return Err(SystemWindowError::InvalidIdentity(field));
            }
        }
        if !measurement_ids.insert(measurement.measurement_id.as_str()) {
            return Err(SystemWindowError::DuplicateMeasurementId(
                measurement.measurement_id.clone(),
            ));
        }
        if !valid_digest(&measurement.evidence_sha256) {
            return Err(SystemWindowError::InvalidEvidenceDigest("resource_measurement"));
        }
        if measurement.end_at_s <= measurement.start_at_s {
            return Err(SystemWindowError::InvalidResourceInterval(
                measurement.measurement_id.clone(),
            ));
        }
        if measurement.start_at_s < spec.start_at_s || measurement.end_at_s > spec.end_at_s {
            return Err(SystemWindowError::ResourceOutsideWindow(
                measurement.measurement_id.clone(),
            ));
        }
        if !measurement.value.is_finite() || measurement.value < 0.0 {
            return Err(SystemWindowError::InvalidResourceValue(
                measurement.measurement_id.clone(),
            ));
        }
        if measurement.unit != measurement.kind.required_unit() {
            return Err(SystemWindowError::ResourceUnitMismatch {
                measurement_id: measurement.measurement_id.clone(),
                expected: measurement.kind.required_unit().to_owned(),
                observed: measurement.unit.clone(),
            });
        }
        match (&measurement.instrument_id, &measurement.calibration_id) {
            (Some(instrument), Some(calibration))
                if canonical_non_empty(instrument) && canonical_non_empty(calibration) => {}
            (None, None) if measurement.kind == SystemResourceKind::ActiveOperatorTime => {}
            _ => return Err(SystemWindowError::InvalidResourceCalibration(
                measurement.measurement_id.clone(),
            )),
        }

        let mut covers = BTreeSet::new();
        for boundary_id in &measurement.covers_boundary_ids {
            if !canonical_non_empty(boundary_id)
                || boundary_id == &measurement.measurement_boundary_id
                || !covers.insert(boundary_id.as_str())
            {
                return Err(SystemWindowError::InvalidCoveredBoundary(
                    measurement.measurement_id.clone(),
                ));
            }
        }

        if let Some(instrument) = &measurement.instrument_id {
            let instrument_key = (measurement.kind, instrument.clone());
            let stream_intervals = instrument_intervals.entry(instrument_key).or_default();
            if let Some((_, _, first_id)) = stream_intervals.iter().find(|(start, end, _)| {
                measurement.start_at_s < *end && *start < measurement.end_at_s
            }) {
                return Err(SystemWindowError::OverlappingInstrumentStreams {
                    kind: measurement.kind,
                    instrument_id: instrument.clone(),
                    first_measurement_id: first_id.clone(),
                    second_measurement_id: measurement.measurement_id.clone(),
                });
            }
            stream_intervals.push((
                measurement.start_at_s,
                measurement.end_at_s,
                measurement.measurement_id.clone(),
            ));
        }

        let key = (measurement.kind, measurement.measurement_boundary_id.clone());
        let intervals = boundary_intervals.entry(key).or_default();
        if let Some((_, _, first_id)) = intervals.iter().find(|(start, end, _)| {
            measurement.start_at_s < *end && *start < measurement.end_at_s
        }) {
            return Err(SystemWindowError::OverlappingResourceBoundaryIntervals {
                kind: measurement.kind,
                boundary_id: measurement.measurement_boundary_id.clone(),
                first_measurement_id: first_id.clone(),
                second_measurement_id: measurement.measurement_id.clone(),
            });
        }
        intervals.push((
            measurement.start_at_s,
            measurement.end_at_s,
            measurement.measurement_id.clone(),
        ));

        let total = resource_totals.entry(measurement.kind).or_insert(0.0);
        *total += measurement.value;
        if !total.is_finite() {
            return Err(SystemWindowError::ResourceTotalOverflow(measurement.kind));
        }
        present_resources.insert(measurement.kind);
    }

    for index in 0..resources.len() {
        for child_index in 0..resources.len() {
            if index == child_index {
                continue;
            }
            let parent = &resources[index];
            let child = &resources[child_index];
            if parent.kind != child.kind
                || !parent.covers_boundary_ids.iter().any(
                    |boundary_id| boundary_id == &child.measurement_boundary_id
                )
            {
                continue;
            }
            let overlaps = parent.start_at_s < child.end_at_s
                && child.start_at_s < parent.end_at_s;
            if overlaps {
                return Err(SystemWindowError::NestedBoundaryDoubleCount {
                    kind: parent.kind,
                    parent_measurement_id: parent.measurement_id.clone(),
                    child_measurement_id: child.measurement_id.clone(),
                });
            }
        }
    }

    for kind in required_resources {
        if !present_resources.contains(&kind) {
            return Err(SystemWindowError::MissingRequiredResource(*kind));
        }
    }

    let verified_item_release_count = released_items.len();
    let verified_items_per_hour = verified_item_release_count as f64
        * 3_600.0 / elapsed_wall_seconds as f64;
    let first_pass_yield = (completed_cohort_item_count > 0).then(|| {
        first_pass_item_count as f64 / completed_cohort_item_count as f64
    });
    let resource_totals: Vec<_> = resource_totals.into_iter()
        .map(|(kind, value)| (kind, value, kind.required_unit()))
        .collect();
    let resources_per_verified_item = (verified_item_release_count > 0).then(|| {
        resource_totals.iter().map(|(kind, value, unit)| {
            (*kind, *value / verified_item_release_count as f64, *unit)
        }).collect()
    });

    let compartment_stats = build_compartment_stats(spec, requests, releases)?;

    Ok(SystemWindowReport {
        window_id: spec.window_id.clone(),
        machine_unit_id: spec.machine_unit_id.clone(),
        system_configuration_sha256: spec.system_configuration_sha256.clone(),
        utility_topology_id: spec.utility_topology_id.clone(),
        elapsed_wall_seconds,
        request_record_count: requests.len(),
        physical_attempt_count_total,
        rewash_attempt_count_total,
        requests_with_rewash_attempts_total,
        item_attempt_visits_total,
        reattempted_item_visits_total,
        carry_in_requests_at_window_start: carry_in_count,
        requests_accepted_in_window: accepted_count,
        requests_started_in_window: started_count,
        requests_completed_in_window: completed_count,
        cancelled_requests_in_window: cancelled_count,
        cancelled_items_in_accepted_cohort: cancelled_item_count,
        unresolved_requests_at_window_end: unresolved_count,
        requested_items_in_accepted_cohort: requested_item_count,
        items_without_verified_qualification_in_accepted_unresolved_requests: unresolved_item_count,
        verified_item_release_count,
        unqualified_items_in_service_reports: unqualified_item_count,
        first_pass_verified_items_in_completed_cohort: first_pass_item_count,
        items_in_completed_cohort: completed_cohort_item_count,
        first_pass_yield,
        verified_items_per_hour,
        time_to_first_release_seconds: first_release_at_s
            .map(|timestamp| timestamp - spec.start_at_s),
        active_request_seconds,
        maximum_concurrent_requests,
        request_latency_p50_seconds: percentile_nearest_rank(&mut latency_values, 0.50),
        request_latency_p95_seconds: percentile_nearest_rank(&mut latency_values, 0.95),
        queue_wait_p50_seconds: percentile_nearest_rank(&mut queue_wait_values, 0.50),
        queue_wait_p95_seconds: percentile_nearest_rank(&mut queue_wait_values, 0.95),
        total_resource_measurement_count: resources.len(),
        resource_totals,
        window_resources_per_verified_release: resources_per_verified_item,
        compartment_stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service::AttemptResources;

    const CLOCK: &str = "monotonic-clock-01";

    fn digest(ch: char) -> String {
        std::iter::repeat(ch).take(64).collect()
    }

    fn report(
        request_id: &str,
        verified: &[&str],
        unqualified: &[&str],
        first_pass: &[&str],
    ) -> ServiceAccountingReport {
        let verified_ids: Vec<String> = verified.iter().map(|id| (*id).to_owned()).collect();
        let unqualified_ids: Vec<String> = unqualified.iter().map(|id| (*id).to_owned()).collect();
        let first_pass_ids: Vec<String> = first_pass.iter().map(|id| (*id).to_owned()).collect();
        let unique_item_count = verified_ids.len() + unqualified_ids.len();
        let resources = AttemptResources::default();
        let resources_per_verified_item = if verified_ids.is_empty() { None } else { Some(resources) };
        ServiceAccountingReport {
            service_request_id: request_id.to_owned(),
            attempt_count: 1,
            physical_run_count: 1,
            rewash_attempt_count: 0,
            unique_item_count,
            item_attempt_visits: unique_item_count,
            reattempted_item_visits: 0,
            verified_successful_item_count: verified_ids.len(),
            verified_successful_item_ids: verified_ids,
            first_pass_verified_item_count: first_pass_ids.len(),
            first_pass_verified_item_ids: first_pass_ids,
            unqualified_item_ids: unqualified_ids,
            total_resources: resources,
            resources_per_verified_item,
            service_makespan_s: 60,
        }
    }

    fn request(
        id: &str,
        accepted: u64,
        started: Option<u64>,
        state: WindowRequestState,
        item_ids: &[&str],
        service_report: Option<ServiceAccountingReport>,
    ) -> ServiceWindowRequest {
        ServiceWindowRequest {
            service_request_id: id.to_owned(),
            clock_domain_id: CLOCK.to_owned(),
            accepted_at_s: accepted,
            started_at_s: started,
            state,
            service_report,
            requested_item_ids: item_ids.iter().map(|id| (*id).to_owned()).collect(),
            compartment_id: "cell-01".to_owned(),
            compartment_configuration_sha256: digest('c'),
        }
    }

    fn release(request_id: &str, item_id: &str, at: u64) -> ItemReleaseEvent {
        ItemReleaseEvent {
            service_request_id: request_id.to_owned(),
            item_id: item_id.to_owned(),
            released_at_s: at,
            evidence_ref: format!("evidence://release/{request_id}/{item_id}"),
            evidence_sha256: digest('a'),
        }
    }

    fn measurement(
        kind: SystemResourceKind,
        id: &str,
        boundary: &str,
        value: f64,
        unit: &str,
        instrument: Option<&str>,
        calibration: Option<&str>,
        covers: &[&str],
    ) -> SystemWindowResourceMeasurement {
        SystemWindowResourceMeasurement {
            source_window_id: "window-01".to_owned(),
            clock_domain_id: CLOCK.to_owned(),
            measurement_id: id.to_owned(),
            kind,
            measurement_boundary_id: boundary.to_owned(),
            value,
            unit: unit.to_owned(),
            start_at_s: 0,
            end_at_s: 100,
            instrument_id: instrument.map(str::to_owned),
            calibration_id: calibration.map(str::to_owned),
            method_id: "window-meter-method-v1".to_owned(),
            evidence_ref: format!("evidence://system-resource/{id}"),
            evidence_sha256: digest('b'),
            covers_boundary_ids: covers.iter().map(|id| (*id).to_owned()).collect(),
        }
    }

    fn spec(required: &[SystemResourceKind]) -> SystemWindowSpec {
        SystemWindowSpec {
            window_id: "window-01".to_owned(),
            clock_domain_id: CLOCK.to_owned(),
            start_at_s: 0,
            end_at_s: 100,
            required_resources: required.to_vec(),
            machine_unit_id: "machine-01".to_owned(),
            system_configuration_sha256: digest('d'),
            utility_topology_id: "two-cell-isolated-wet-loops-shared-thermal-bus-v1".to_owned(),
        }
    }

    fn basic_resources() -> Vec<SystemWindowResourceMeasurement> {
        vec![
            measurement(
                SystemResourceKind::ApplianceWallElectricity,
                "power-main",
                "appliance-main-feed",
                0.12,
                "kWh",
                Some("power-meter-01"),
                Some("power-cal-01"),
                &[],
            ),
            measurement(
                SystemResourceKind::FreshWater,
                "water-main",
                "fresh-water-inlet",
                1.8,
                "L",
                Some("water-meter-01"),
                Some("water-cal-01"),
                &[],
            ),
            measurement(
                SystemResourceKind::ActiveOperatorTime,
                "operator-one",
                "operator-pseudonym-one",
                30.0,
                "s",
                None,
                None,
                &[],
            ),
            measurement(
                SystemResourceKind::ActiveOperatorTime,
                "operator-two",
                "operator-pseudonym-two",
                30.0,
                "s",
                None,
                None,
                &[],
            ),
        ]
    }

    const REQUIRED: [SystemResourceKind; 3] = [
        SystemResourceKind::ApplianceWallElectricity,
        SystemResourceKind::FreshWater,
        SystemResourceKind::ActiveOperatorTime,
    ];

    #[test]
    fn concurrent_requests_use_wall_window_and_report_actual_overlap() {
        let requests = [
            request(
                "service-a",
                0,
                Some(0),
                WindowRequestState::Completed { closed_at_s: 60 },
                &["cup-a"],
                Some(report("service-a", &["cup-a"], &[], &["cup-a"])),
            ),
            request(
                "service-b",
                20,
                Some(30),
                WindowRequestState::Completed { closed_at_s: 90 },
                &["cup-b"],
                Some(report("service-b", &["cup-b"], &[], &["cup-b"])),
            ),
        ];
        let report = account_system_window(
            &spec(&REQUIRED),
            &requests,
            &[release("service-a", "cup-a", 50), release("service-b", "cup-b", 80)],
            &basic_resources(),
        ).unwrap();

        assert_eq!(report.elapsed_wall_seconds, 100);
        assert_eq!(report.verified_item_release_count, 2);
        assert_eq!(report.verified_items_per_hour, 72.0);
        assert_eq!(report.active_request_seconds, 120);
        assert_eq!(report.maximum_concurrent_requests, 2);
        assert_eq!(report.request_latency_p50_seconds, Some(60));
        assert_eq!(report.request_latency_p95_seconds, Some(70));
        assert_eq!(report.first_pass_yield, Some(1.0));
        assert_eq!(report.time_to_first_release_seconds, Some(50));
    }

    #[test]
    fn sequential_requests_do_not_create_false_concurrency() {
        let requests = [
            request(
                "service-a",
                0,
                Some(0),
                WindowRequestState::Completed { closed_at_s: 60 },
                &["cup-a"],
                Some(report("service-a", &["cup-a"], &[], &["cup-a"])),
            ),
            request(
                "service-b",
                60,
                Some(60),
                WindowRequestState::Completed { closed_at_s: 90 },
                &["cup-b"],
                Some(report("service-b", &["cup-b"], &[], &["cup-b"])),
            ),
        ];
        let result = account_system_window(
            &spec(&REQUIRED),
            &requests,
            &[release("service-a", "cup-a", 50), release("service-b", "cup-b", 80)],
            &basic_resources(),
        ).unwrap();
        assert_eq!(result.maximum_concurrent_requests, 1);
        assert_eq!(result.active_request_seconds, 90);
    }

    #[test]
    fn reports_carry_in_requests_separately_from_new_arrivals() {
        let requests = [
            request(
                "carry-in",
                0,
                Some(0),
                WindowRequestState::Completed { closed_at_s: 40 },
                &["cup-old"],
                Some(report("carry-in", &["cup-old"], &[], &["cup-old"])),
            ),
            request(
                "new-arrival",
                20,
                Some(30),
                WindowRequestState::Completed { closed_at_s: 80 },
                &["cup-new"],
                Some(report("new-arrival", &["cup-new"], &[], &["cup-new"])),
            ),
        ];
        let mut window = spec(&REQUIRED);
        window.start_at_s = 10;
        window.end_at_s = 110;
        let mut resource_rows = basic_resources();
        for row in &mut resource_rows {
            row.start_at_s = 10;
            row.end_at_s = 110;
        }
        let result = account_system_window(
            &window,
            &requests,
            &[release("carry-in", "cup-old", 35), release("new-arrival", "cup-new", 70)],
            &resource_rows,
        ).unwrap();
        assert_eq!(result.carry_in_requests_at_window_start, 1);
        assert_eq!(result.requests_accepted_in_window, 1);
        assert_eq!(result.verified_item_release_count, 2);
        assert_eq!(result.items_in_completed_cohort, 1);
    }

    #[test]
    fn queued_unresolved_request_is_right_censored_and_never_released() {
        let requests = [request(
            "service-queued",
            40,
            None,
            WindowRequestState::Unresolved { observed_until_s: 100 },
            &["cup-queued"],
            None,
        )];
        let result = account_system_window(
            &spec(&REQUIRED),
            &requests,
            &[],
            &basic_resources(),
        ).unwrap();
        assert_eq!(result.unresolved_requests_at_window_end, 1);
        assert_eq!(result.items_without_verified_qualification_in_accepted_unresolved_requests, 1);
        assert_eq!(result.verified_item_release_count, 0);
        assert_eq!(result.verified_items_per_hour, 0.0);
        assert!(result.window_resources_per_verified_release.is_none());
        assert!(result.first_pass_yield.is_none());
    }

    #[test]
    fn started_but_unresolved_request_can_release_only_assayed_items() {
        let requests = [request(
            "service-partial",
            5,
            Some(10),
            WindowRequestState::Unresolved { observed_until_s: 100 },
            &["cup-passed", "cup-pending"],
            Some(report("service-partial", &["cup-passed"], &[], &["cup-passed"])),
        )];
        let result = account_system_window(
            &spec(&REQUIRED),
            &requests,
            &[release("service-partial", "cup-passed", 80)],
            &basic_resources(),
        ).unwrap();
        assert_eq!(result.verified_item_release_count, 1);
        assert_eq!(result.items_without_verified_qualification_in_accepted_unresolved_requests, 1);
        assert_eq!(result.unresolved_requests_at_window_end, 1);
    }

    #[test]
    fn release_without_independent_service_qualification_is_rejected() {
        let requests = [request(
            "service-failed",
            0,
            Some(0),
            WindowRequestState::Completed { closed_at_s: 60 },
            &["cup-failed"],
            Some(report("service-failed", &[], &["cup-failed"], &[])),
        )];
        assert!(matches!(
            account_system_window(
                &spec(&REQUIRED),
                &requests,
                &[release("service-failed", "cup-failed", 50)],
                &basic_resources(),
            ),
            Err(SystemWindowError::ReleaseNotQualified { .. })
        ));
    }

    #[test]
    fn duplicate_release_event_for_same_request_item_is_rejected() {
        let requests = [request(
            "service-one",
            0,
            Some(0),
            WindowRequestState::Completed { closed_at_s: 60 },
            &["cup-one"],
            Some(report("service-one", &["cup-one"], &[], &["cup-one"])),
        )];
        assert!(matches!(
            account_system_window(
                &spec(&REQUIRED),
                &requests,
                &[release("service-one", "cup-one", 50), release("service-one", "cup-one", 55)],
                &basic_resources(),
            ),
            Err(SystemWindowError::DuplicateItemRelease { .. })
        ));
    }

    #[test]
    fn request_clock_domain_must_match_system_window() {
        let mut one = request(
            "service-one",
            0,
            Some(0),
            WindowRequestState::Completed { closed_at_s: 60 },
            &["cup-one"],
            Some(report("service-one", &["cup-one"], &[], &["cup-one"])),
        );
        one.clock_domain_id = "different-clock".to_owned();
        assert!(matches!(
            account_system_window(&spec(&REQUIRED), &[one], &[], &basic_resources()),
            Err(SystemWindowError::InvalidClockDomain { .. })
        ));
    }

    #[test]
    fn overlapping_meters_on_same_boundary_are_rejected_even_when_instruments_differ() {
        let requests = [request(
            "service-one",
            0,
            Some(0),
            WindowRequestState::Completed { closed_at_s: 60 },
            &["cup-one"],
            Some(report("service-one", &["cup-one"], &[], &["cup-one"])),
        )];
        let mut resources = basic_resources();
        resources.push(measurement(
            SystemResourceKind::ApplianceWallElectricity,
            "power-redundant",
            "appliance-main-feed",
            0.02,
            "kWh",
            Some("power-meter-02"),
            Some("power-cal-02"),
            &[],
        ));
        assert!(matches!(
            account_system_window(&spec(&REQUIRED), &requests, &[], &resources),
            Err(SystemWindowError::OverlappingResourceBoundaryIntervals { .. })
        ));
    }

    #[test]
    fn resource_meter_must_share_the_window_clock_domain() {
        let requests = [request(
            "service-one",
            0,
            Some(0),
            WindowRequestState::Completed { closed_at_s: 60 },
            &["cup-one"],
            Some(report("service-one", &["cup-one"], &[], &["cup-one"])),
        )];
        let mut resources = basic_resources();
        resources[0].clock_domain_id = "other-clock".to_owned();
        assert!(matches!(
            account_system_window(&spec(&REQUIRED), &requests, &[], &resources),
            Err(SystemWindowError::InvalidResourceClockDomain { .. })
        ));
    }

    #[test]
    fn one_meter_stream_cannot_overlap_across_two_declared_boundaries() {
        let requests = [request(
            "service-one",
            0,
            Some(0),
            WindowRequestState::Completed { closed_at_s: 60 },
            &["cup-one"],
            Some(report("service-one", &["cup-one"], &[], &["cup-one"])),
        )];
        let mut resources = basic_resources();
        resources.push(measurement(
            SystemResourceKind::ApplianceWallElectricity,
            "power-second-boundary",
            "secondary-feed",
            0.02,
            "kWh",
            Some("power-meter-01"),
            Some("power-cal-01"),
            &[],
        ));
        assert!(matches!(
            account_system_window(&spec(&REQUIRED), &requests, &[], &resources),
            Err(SystemWindowError::OverlappingInstrumentStreams { .. })
        ));
    }

    #[test]
    fn aggregate_and_child_meters_are_not_summed_when_coverage_is_declared() {
        let requests = [request(
            "service-one",
            0,
            Some(0),
            WindowRequestState::Completed { closed_at_s: 60 },
            &["cup-one"],
            Some(report("service-one", &["cup-one"], &[], &["cup-one"])),
        )];
        let mut resources = basic_resources();
        resources.push(measurement(
            SystemResourceKind::ApplianceWallElectricity,
            "power-parent",
            "appliance-total",
            0.12,
            "kWh",
            Some("power-parent-meter"),
            Some("power-parent-cal"),
            &["heater-branch"],
        ));
        resources.push(measurement(
            SystemResourceKind::ApplianceWallElectricity,
            "power-child",
            "heater-branch",
            0.03,
            "kWh",
            Some("power-child-meter"),
            Some("power-child-cal"),
            &[],
        ));
        assert!(matches!(
            account_system_window(&spec(&REQUIRED), &requests, &[], &resources),
            Err(SystemWindowError::NestedBoundaryDoubleCount { .. })
        ));
    }

    #[test]
    fn parallel_operator_time_is_counted_as_person_seconds_not_duplicate_team_time() {
        let requests = [request(
            "service-one",
            0,
            Some(0),
            WindowRequestState::Completed { closed_at_s: 60 },
            &["cup-one"],
            Some(report("service-one", &["cup-one"], &[], &["cup-one"])),
        )];
        let result = account_system_window(
            &spec(&REQUIRED),
            &requests,
            &[],
            &basic_resources(),
        ).unwrap();
        let operator_seconds = result.resource_totals.iter()
            .find(|(kind, _, _)| *kind == SystemResourceKind::ActiveOperatorTime)
            .map(|(_, value, _)| *value)
            .unwrap();
        assert_eq!(operator_seconds, 60.0);
    }

    #[test]
    fn missing_required_resource_dimension_is_rejected_even_when_zero_was_intended() {
        let requests = [request(
            "service-one",
            0,
            Some(0),
            WindowRequestState::Completed { closed_at_s: 60 },
            &["cup-one"],
            Some(report("service-one", &["cup-one"], &[], &["cup-one"])),
        )];
        let resources = vec![
            basic_resources()[0].clone(),
            basic_resources()[1].clone(),
        ];
        assert!(matches!(
            account_system_window(&spec(&REQUIRED), &requests, &[], &resources),
            Err(SystemWindowError::MissingRequiredResource(SystemResourceKind::ActiveOperatorTime))
        ));
    }

    #[test]
    fn first_pass_yield_excludes_pending_requests_and_includes_failed_items() {
        let requests = [
            request(
                "service-complete",
                0,
                Some(0),
                WindowRequestState::Completed { closed_at_s: 60 },
                &["cup-pass", "cup-fail"],
                Some(report("service-complete", &["cup-pass"], &["cup-fail"], &["cup-pass"])),
            ),
            request(
                "service-pending",
                20,
                None,
                WindowRequestState::Unresolved { observed_until_s: 100 },
                &["cup-pending"],
                None,
            ),
        ];
        let result = account_system_window(
            &spec(&REQUIRED),
            &requests,
            &[release("service-complete", "cup-pass", 50)],
            &basic_resources(),
        ).unwrap();
        assert_eq!(result.first_pass_verified_items_in_completed_cohort, 1);
        assert_eq!(result.items_in_completed_cohort, 2);
        assert_eq!(result.first_pass_yield, Some(0.5));
        assert_eq!(result.unresolved_requests_at_window_end, 1);
    }

    #[test]
    fn reports_per_compartment_activity_and_release_counts() {
        let requests = [
            request(
                "service-cell-a",
                0,
                Some(0),
                WindowRequestState::Completed { closed_at_s: 60 },
                &["cup-a"],
                Some(report("service-cell-a", &["cup-a"], &[], &["cup-a"])),
            ),
            {
                let mut second = request(
                    "service-cell-b",
                    10,
                    Some(20),
                    WindowRequestState::Completed { closed_at_s: 80 },
                    &["cup-b"],
                    Some(report("service-cell-b", &["cup-b"], &[], &["cup-b"])),
                );
                second.compartment_id = "cell-02".to_owned();
                second.compartment_configuration_sha256 = digest('e');
                second
            },
        ];
        let result = account_system_window(
            &spec(&REQUIRED),
            &requests,
            &[release("service-cell-a", "cup-a", 50), release("service-cell-b", "cup-b", 70)],
            &basic_resources(),
        ).unwrap();

        assert_eq!(result.maximum_concurrent_requests, 2);
        assert_eq!(result.compartment_stats.len(), 2);
        let cell_a = result.compartment_stats.iter().find(|cell| cell.compartment_id == "cell-01").unwrap();
        let cell_b = result.compartment_stats.iter().find(|cell| cell.compartment_id == "cell-02").unwrap();
        assert_eq!(cell_a.verified_item_release_count, 1);
        assert_eq!(cell_b.verified_item_release_count, 1);
        assert_eq!(cell_a.maximum_concurrent_requests, 1);
        assert_eq!(cell_b.maximum_concurrent_requests, 1);
        assert_eq!(cell_a.compartment_configuration_sha256, digest('c'));
        assert_eq!(cell_b.compartment_configuration_sha256, digest('e'));
    }

    #[test]
    fn rejects_one_compartment_id_reused_with_different_configuration_digests() {
        let requests = [
            request(
                "service-a",
                0,
                Some(0),
                WindowRequestState::Completed { closed_at_s: 60 },
                &["cup-a"],
                Some(report("service-a", &["cup-a"], &[], &["cup-a"])),
            ),
            {
                let mut second = request(
                    "service-b",
                    10,
                    Some(20),
                    WindowRequestState::Completed { closed_at_s: 80 },
                    &["cup-b"],
                    Some(report("service-b", &["cup-b"], &[], &["cup-b"])),
                );
                second.compartment_configuration_sha256 = digest('e');
                second
            },
        ];
        assert!(matches!(
            account_system_window(&spec(&REQUIRED), &requests, &[], &basic_resources()),
            Err(SystemWindowError::CompartmentConfigurationMismatch { .. })
        ));
    }

    #[test]
    fn aggregates_attempts_and_rewash_burden_across_requests() {
        let requests = [
            request(
                "service-a",
                0,
                Some(0),
                WindowRequestState::Completed { closed_at_s: 60 },
                &["cup-a"],
                Some(report("service-a", &["cup-a"], &[], &["cup-a"])),
            ),
            {
                let mut service_report = report("service-b", &["cup-b"], &[], &[]);
                service_report.attempt_count = 2;
                service_report.physical_run_count = 2;
                service_report.rewash_attempt_count = 1;
                service_report.item_attempt_visits = 2;
                service_report.reattempted_item_visits = 1;
                let mut second = request(
                    "service-b",
                    10,
                    Some(20),
                    WindowRequestState::Completed { closed_at_s: 80 },
                    &["cup-b"],
                    Some(service_report),
                );
                second
            },
        ];
        let result = account_system_window(&spec(&REQUIRED), &requests, &[], &basic_resources()).unwrap();
        assert_eq!(result.physical_attempt_count_total, 3);
        assert_eq!(result.rewash_attempt_count_total, 1);
        assert_eq!(result.requests_with_rewash_attempts_total, 1);
        assert_eq!(result.item_attempt_visits_total, 3);
        assert_eq!(result.reattempted_item_visits_total, 1);
    }

    #[test]
    fn inconsistent_first_pass_item_list_is_rejected() {
        let mut bad_report = report("service-one", &["cup-one"], &[], &[]);
        bad_report.first_pass_verified_item_count = 1;
        let requests = [request(
            "service-one",
            0,
            Some(0),
            WindowRequestState::Completed { closed_at_s: 60 },
            &["cup-one"],
            Some(bad_report),
        )];
        assert!(matches!(
            account_system_window(&spec(&REQUIRED), &requests, &[], &basic_resources()),
            Err(SystemWindowError::InconsistentServiceReport { field: "first_pass_verified_item_count", .. })
        ));
    }
}
