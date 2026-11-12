//! Deterministic constrained inference admission for optional robot-fleet requests.
//!
//! This reference only admits optional remote inference work. It does not
//! schedule or authorize local control, RTA, interlocks, safe-stop or actuators.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

/// Request-specific evidence required before optional remote inference is considered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceRequest {
    /// Unique request identity.
    pub request_id: String,
    /// Robot whose optional remote request is being considered.
    pub robot_id: String,
    /// Versioned loss scale for this request's EVSI.
    pub loss_scale_id: Option<String>,
    /// Expected value of sample information in task-loss units.
    pub evsi_units: Option<u64>,
    /// Conservative p99 remote result return latency.
    pub p99_return_ticks: Option<u64>,
    /// Time until the local action buffer reaches its qualified fallback boundary.
    pub action_buffer_slack_ticks: Option<u64>,
    /// Remaining task/actuation deadline slack.
    pub deadline_slack_ticks: Option<u64>,
    /// Local projection/dispatch guard after remote result return.
    pub local_projection_dispatch_guard_ticks: u64,
    /// Qualified conservative source-to-actuator AoI upper bound.
    pub actuation_aoi_upper_ticks: Option<u64>,
    /// Maximum task-specific AoI permitted by the consumer contract.
    pub max_actuation_aoi_ticks: u64,
    /// Conservative upper bound on probability of missing the declared actuation window.
    pub poma_upper_ppm: Option<u32>,
    /// Per-request tolerated PoMA upper bound in parts per million.
    pub poma_limit_ppm: u32,
    /// GPU service demand.
    pub gpu_micros: u64,
    /// Declared memory-bandwidth service demand.
    pub memory_bandwidth_units: u64,
    /// Network payload/transfer demand in bytes.
    pub network_bytes: u64,
    /// Incremental KV-cache allocation required for this request; None is Unknown.
    pub kv_cache_growth_tokens: Option<u64>,
    /// Endpoint/session/model/request identity evidence is valid.
    pub identity_valid: bool,
    /// Source and decision receipt digests are complete and match.
    pub evidence_digest_valid: bool,
    /// Clock-domain relation is qualified for the supplied age and latency bounds.
    pub clock_mapping_valid: bool,
    /// A qualified local fallback remains available.
    pub local_fallback_ready: bool,
    /// Remote transmission is allowed by the request's privacy/data policy.
    pub privacy_policy_allows_remote_data: bool,
    /// Admitting this request would not preempt or displace any local hard-real-time safety work.
    pub local_hard_safety_preserved: bool,
    /// This robot is owed a declared optional-inference service floor.
    pub service_floor_due: bool,
    /// Current consecutive control ticks without a useful admitted remote result.
    pub starvation_ticks: u64,
    /// Maximum permitted starvation before a service-floor obligation is breached.
    pub max_starvation_ticks: u64,
}

/// Simultaneous resource budget for one admission batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceBudget {
    /// GPU execution budget in microseconds.
    pub gpu_micros: u64,
    /// Memory-bandwidth service budget in caller-defined calibrated units.
    pub memory_bandwidth_units: u64,
    /// Network transfer budget in bytes.
    pub network_bytes: u64,
    /// Available additional KV-cache token capacity.
    pub kv_cache_growth_tokens: u64,
}

/// Resources consumed by an admitted batch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResourceUse {
    /// GPU execution used in microseconds.
    pub gpu_micros: u64,
    /// Memory-bandwidth service used in calibrated units.
    pub memory_bandwidth_units: u64,
    /// Network bytes admitted.
    pub network_bytes: u64,
    /// Additional KV-cache tokens reserved by this batch.
    pub kv_cache_growth_tokens: u64,
}

impl ResourceUse {
    fn checked_add_request(&mut self, request: &InferenceRequest) -> Result<(), ReferenceError> {
        self.gpu_micros = self
            .gpu_micros
            .checked_add(request.gpu_micros)
            .ok_or(ReferenceError::ResourceArithmeticOverflow)?;
        self.memory_bandwidth_units = self
            .memory_bandwidth_units
            .checked_add(request.memory_bandwidth_units)
            .ok_or(ReferenceError::ResourceArithmeticOverflow)?;
        self.network_bytes = self
            .network_bytes
            .checked_add(request.network_bytes)
            .ok_or(ReferenceError::ResourceArithmeticOverflow)?;
        self.kv_cache_growth_tokens = self
            .kv_cache_growth_tokens
            .checked_add(request.kv_cache_growth_tokens.ok_or(
                ReferenceError::UnknownKvCacheDemand(request.request_id.clone()),
            )?)
            .ok_or(ReferenceError::ResourceArithmeticOverflow)?;
        Ok(())
    }

    fn fits(self, budget: ResourceBudget) -> bool {
        self.gpu_micros <= budget.gpu_micros
            && self.memory_bandwidth_units <= budget.memory_bandwidth_units
            && self.network_bytes <= budget.network_bytes
            && self.kv_cache_growth_tokens <= budget.kv_cache_growth_tokens
    }
}

/// Deterministic policy arms implemented by this reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerPolicy {
    /// Robot-order rotation from the declared cursor.
    RoundRobin,
    /// Earliest minimum of action-buffer slack and deadline slack.
    EarliestDeadlineFirst,
    /// Highest EVSI per GPU-microsecond; requires one common task-loss scale.
    EvsiPerCompute,
    /// Constrained lexicographic mission-aware candidate.
    Luminous,
}

/// Round-robin cursor: requests from robot IDs lexically after this robot are visited first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerCursor {
    /// Last robot ID serviced before the current scheduling decision.
    pub last_robot_id: Option<String>,
}

/// Request-level disposition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestDisposition {
    /// Request was selected for optional remote inference.
    Admitted,
    /// Request was blocked by one or more hard eligibility requirements.
    Rejected(RejectReason),
    /// Request was eligible but did not fit in remaining batch resources.
    ResourceCapacity,
}

/// Reason a request cannot be admitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectReason {
    /// The request identity or endpoint evidence is invalid.
    InvalidIdentity,
    /// The source/decision receipt digest is incomplete or mismatched.
    InvalidEvidenceDigest,
    /// Clock mapping is not qualified.
    UnknownClockRelation,
    /// Actuation-side age upper bound is missing.
    UnknownActuationAge,
    /// Incremental KV-cache growth demand is missing.
    UnknownKvCacheDemand,
    /// Actuation-side age exceeds the task limit.
    StaleActuationAge,
    /// p99 return latency or action-buffer slack is missing.
    UnknownTimingBound,
    /// p99 result plus guard cannot fit in the action-buffer/deadline window.
    ResultReturnsTooLate,
    /// PoMA upper bound is missing.
    UnknownMissRisk,
    /// PoMA upper bound exceeds the request-specific limit.
    MissRiskTooHigh,
    /// Supplied probability-of-miss upper bound exceeds one million ppm.
    InvalidProbabilityBound(u32),
    /// Request probability threshold exceeds one million ppm.
    InvalidProbabilityLimit(u32),
    /// Local fallback is not ready.
    NoLocalFallback,
    /// Request data is not allowed to leave the robot/declared trust boundary.
    PrivacyPolicyDenied,
    /// Optional compute would displace local hard-real-time safety work.
    LocalSafetyWouldBeDisplaced,
    /// EVSI or loss-scale identity is missing.
    UnknownDecisionValue,
    /// Optional remote request has exactly zero task-loss value.
    ZeroEvsi,
    /// The declared maximum starvation interval is invalid or zero.
    StarvationEvidenceInvalid,
}

/// Typed input or policy error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferenceError {
    /// A request or robot identity is empty.
    EmptyIdentity,
    /// A request identity is duplicated in the same candidate batch.
    DuplicateRequestId(String),
    /// A request's upper probability bound is outside [0, 1].
    InvalidProbabilityBound(u32),
    /// A request-specific probability threshold is outside [0, 1].
    InvalidProbabilityLimit(u32),
    /// A zero GPU demand makes EVSI-per-compute undefined.
    ZeroComputeDemand(String),
    /// EVSI-per-compute cannot compare unrelated task-loss scales.
    IncomparableLossScales,
    /// Resource sum overflowed its integer representation.
    ResourceArithmeticOverflow,
    /// KV-cache growth demand was not supplied for the request.
    UnknownKvCacheDemand(String),
}

/// One request's final disposition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestResult {
    /// Request identity.
    pub request_id: String,
    /// Robot identity.
    pub robot_id: String,
    /// Request outcome.
    pub disposition: RequestDisposition,
}

/// Result of one optional inference admission decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionReport {
    /// Policy that produced the report.
    pub policy: SchedulerPolicy,
    /// Request outcomes in the original input order.
    pub requests: Vec<RequestResult>,
    /// Requests admitted to the optional remote inference batch.
    pub admitted_request_ids: Vec<String>,
    /// Resources consumed by the admitted batch.
    pub resource_use: ResourceUse,
    /// Robots with service-floor obligations not fulfilled by this batch.
    pub service_floor_violations: Vec<String>,
    /// Robots whose maximum starvation interval was already exceeded before this batch.
    pub preexisting_starvation_breaches: Vec<String>,
    /// Whether all supplied service-floor obligations were met.
    pub service_floors_met: bool,
    /// Robot cursor to supply to the next Round Robin admission call.
    pub next_cursor_robot_id: Option<String>,
    /// Explicitly false: this report never authorizes actuator application.
    pub actuation_authorized: bool,
}

fn eligibility(request: &InferenceRequest) -> Result<(), RejectReason> {
    if !request.identity_valid {
        return Err(RejectReason::InvalidIdentity);
    }
    if !request.evidence_digest_valid {
        return Err(RejectReason::InvalidEvidenceDigest);
    }
    if !request.clock_mapping_valid {
        return Err(RejectReason::UnknownClockRelation);
    }

    let age = request
        .actuation_aoi_upper_ticks
        .ok_or(RejectReason::UnknownActuationAge)?;
    if age > request.max_actuation_aoi_ticks {
        return Err(RejectReason::StaleActuationAge);
    }

    let p99 = request
        .p99_return_ticks
        .ok_or(RejectReason::UnknownTimingBound)?;
    let buffer_slack = request
        .action_buffer_slack_ticks
        .ok_or(RejectReason::UnknownTimingBound)?;
    let deadline_slack = request
        .deadline_slack_ticks
        .ok_or(RejectReason::UnknownTimingBound)?;
    let available_slack = buffer_slack.min(deadline_slack);
    let result_and_guard = p99
        .checked_add(request.local_projection_dispatch_guard_ticks)
        .ok_or(RejectReason::ResultReturnsTooLate)?;
    if result_and_guard > available_slack {
        return Err(RejectReason::ResultReturnsTooLate);
    }

    let poma = request
        .poma_upper_ppm
        .ok_or(RejectReason::UnknownMissRisk)?;
    if poma > 1_000_000 {
        return Err(RejectReason::InvalidProbabilityBound(poma));
    }
    if request.poma_limit_ppm > 1_000_000 {
        return Err(RejectReason::InvalidProbabilityLimit(
            request.poma_limit_ppm,
        ));
    }
    if poma > request.poma_limit_ppm {
        return Err(RejectReason::MissRiskTooHigh);
    }

    if !request.local_fallback_ready {
        return Err(RejectReason::NoLocalFallback);
    }
    if !request.privacy_policy_allows_remote_data {
        return Err(RejectReason::PrivacyPolicyDenied);
    }
    if !request.local_hard_safety_preserved {
        return Err(RejectReason::LocalSafetyWouldBeDisplaced);
    }
    if request.max_starvation_ticks == 0 {
        return Err(RejectReason::StarvationEvidenceInvalid);
    }

    if request.kv_cache_growth_tokens.is_none() {
        return Err(RejectReason::UnknownKvCacheDemand);
    }

    let evsi = request
        .evsi_units
        .ok_or(RejectReason::UnknownDecisionValue)?;
    if request
        .loss_scale_id
        .as_ref()
        .is_none_or(|value| value.trim().is_empty())
    {
        return Err(RejectReason::UnknownDecisionValue);
    }
    if evsi == 0 {
        return Err(RejectReason::ZeroEvsi);
    }

    Ok(())
}

fn slack(request: &InferenceRequest) -> u64 {
    request
        .action_buffer_slack_ticks
        .unwrap_or(u64::MAX)
        .min(request.deadline_slack_ticks.unwrap_or(u64::MAX))
}

fn ratio_cmp(a_num: u64, a_den: u64, b_num: u64, b_den: u64) -> std::cmp::Ordering {
    (a_num as u128 * b_den as u128).cmp(&(b_num as u128 * a_den as u128))
}

fn compare_request(
    policy: SchedulerPolicy,
    a: &InferenceRequest,
    b: &InferenceRequest,
) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match policy {
        SchedulerPolicy::RoundRobin => a
            .robot_id
            .cmp(&b.robot_id)
            .then(a.request_id.cmp(&b.request_id)),
        SchedulerPolicy::EarliestDeadlineFirst => slack(a)
            .cmp(&slack(b))
            .then(a.request_id.cmp(&b.request_id)),
        SchedulerPolicy::EvsiPerCompute => {
            let ae = a.evsi_units.unwrap_or(0);
            let be = b.evsi_units.unwrap_or(0);
            let ag = a.gpu_micros.max(1);
            let bg = b.gpu_micros.max(1);
            ratio_cmp(be, bg, ae, ag).then(a.request_id.cmp(&b.request_id))
        }
        SchedulerPolicy::Luminous => {
            // A lexicographic contract, not a scalar reward:
            // service floor, starvation debt, minimum slack, PoMA urgency,
            // then EVSI density only within the same declared loss scale.
            let floor = b.service_floor_due.cmp(&a.service_floor_due);
            if floor != Ordering::Equal {
                return floor;
            }

            let starvation = ratio_cmp(
                b.starvation_ticks,
                b.max_starvation_ticks.max(1),
                a.starvation_ticks,
                a.max_starvation_ticks.max(1),
            );
            if starvation != Ordering::Equal {
                return starvation;
            }

            let slack_order = slack(a).cmp(&slack(b));
            if slack_order != Ordering::Equal {
                return slack_order;
            }

            let risk_order = b
                .poma_upper_ppm
                .unwrap_or(0)
                .cmp(&a.poma_upper_ppm.unwrap_or(0));
            if risk_order != Ordering::Equal {
                return risk_order;
            }

            if a.loss_scale_id == b.loss_scale_id && a.gpu_micros > 0 && b.gpu_micros > 0 {
                let density = ratio_cmp(
                    b.evsi_units.unwrap_or(0),
                    b.gpu_micros,
                    a.evsi_units.unwrap_or(0),
                    a.gpu_micros,
                );
                if density != Ordering::Equal {
                    return density;
                }
            }

            a.robot_id
                .cmp(&b.robot_id)
                .then(a.request_id.cmp(&b.request_id))
        }
    }
}

/// Admit a batch of optional remote inference requests under the selected policy.
///
/// The function performs hard eligibility before sorting, enforces three
/// simultaneous resource budgets, and returns request-level dispositions.
/// It never authorizes use of the returned model result or any actuator command.
pub fn admit_batch(
    policy: SchedulerPolicy,
    cursor: &SchedulerCursor,
    budget: ResourceBudget,
    requests: &[InferenceRequest],
) -> Result<AdmissionReport, ReferenceError> {
    if requests.is_empty() {
        return Ok(AdmissionReport {
            policy,
            requests: Vec::new(),
            admitted_request_ids: Vec::new(),
            resource_use: ResourceUse::default(),
            service_floor_violations: Vec::new(),
            preexisting_starvation_breaches: Vec::new(),
            service_floors_met: true,
            next_cursor_robot_id: None,
            actuation_authorized: false,
        });
    }

    let mut seen_ids = BTreeSet::new();
    for request in requests {
        if request.request_id.trim().is_empty() || request.robot_id.trim().is_empty() {
            return Err(ReferenceError::EmptyIdentity);
        }
        if !seen_ids.insert(request.request_id.as_str()) {
            return Err(ReferenceError::DuplicateRequestId(
                request.request_id.clone(),
            ));
        }
        if let Some(poma) = request.poma_upper_ppm
            && poma > 1_000_000
        {
            return Err(ReferenceError::InvalidProbabilityBound(poma));
        }
        if request.poma_limit_ppm > 1_000_000 {
            return Err(ReferenceError::InvalidProbabilityLimit(
                request.poma_limit_ppm,
            ));
        }
    }

    let mut eligible = Vec::new();
    let mut outcomes: BTreeMap<String, RequestDisposition> = BTreeMap::new();
    for request in requests {
        match eligibility(request) {
            Ok(()) => eligible.push(request),
            Err(reason) => {
                outcomes.insert(
                    request.request_id.clone(),
                    RequestDisposition::Rejected(reason),
                );
            }
        }
    }

    if policy == SchedulerPolicy::EvsiPerCompute {
        let scales: BTreeSet<&str> = eligible
            .iter()
            .filter_map(|request| request.loss_scale_id.as_deref())
            .collect();
        if scales.len() > 1 {
            return Err(ReferenceError::IncomparableLossScales);
        }
        for request in &eligible {
            if request.gpu_micros == 0 {
                return Err(ReferenceError::ZeroComputeDemand(
                    request.request_id.clone(),
                ));
            }
        }
    }

    let mut ordered: Vec<&InferenceRequest> = eligible;
    if policy == SchedulerPolicy::RoundRobin {
        if let Some(last) = cursor.last_robot_id.as_deref() {
            ordered.sort_by(|a, b| {
                let a_after = a.robot_id.as_str() > last;
                let b_after = b.robot_id.as_str() > last;
                b_after
                    .cmp(&a_after)
                    .then_with(|| compare_request(policy, a, b))
            });
        } else {
            ordered.sort_by(|a, b| compare_request(policy, a, b));
        }
    } else {
        ordered.sort_by(|a, b| compare_request(policy, a, b));
    }

    let mut resource_use = ResourceUse::default();
    let mut admitted_request_ids = Vec::new();
    let mut admitted_robots = BTreeSet::new();
    let mut next_cursor_robot_id = None;

    for request in ordered {
        let mut proposed = resource_use;
        proposed.checked_add_request(request)?;
        if proposed.fits(budget) {
            resource_use = proposed;
            admitted_request_ids.push(request.request_id.clone());
            admitted_robots.insert(request.robot_id.clone());
            next_cursor_robot_id = Some(request.robot_id.clone());
            outcomes.insert(request.request_id.clone(), RequestDisposition::Admitted);
        } else {
            outcomes.insert(
                request.request_id.clone(),
                RequestDisposition::ResourceCapacity,
            );
        }
    }

    if next_cursor_robot_id.is_none() {
        next_cursor_robot_id = cursor.last_robot_id.clone();
    }

    let service_floor_robots: BTreeSet<String> = requests
        .iter()
        .filter(|request| request.service_floor_due)
        .map(|request| request.robot_id.clone())
        .collect();
    let service_floor_violations: Vec<String> = service_floor_robots
        .iter()
        .filter(|robot_id| !admitted_robots.contains(*robot_id))
        .cloned()
        .collect();
    let preexisting_starvation_breaches: Vec<String> = requests
        .iter()
        .filter(|request| request.starvation_ticks > request.max_starvation_ticks)
        .map(|request| request.robot_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let results = requests
        .iter()
        .map(|request| RequestResult {
            request_id: request.request_id.clone(),
            robot_id: request.robot_id.clone(),
            disposition: outcomes
                .get(&request.request_id)
                .cloned()
                .unwrap_or(RequestDisposition::ResourceCapacity),
        })
        .collect();

    Ok(AdmissionReport {
        policy,
        requests: results,
        admitted_request_ids,
        resource_use,
        service_floors_met: service_floor_violations.is_empty()
            && preexisting_starvation_breaches.is_empty(),
        service_floor_violations,
        preexisting_starvation_breaches,
        next_cursor_robot_id,
        actuation_authorized: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(id: &str, robot: &str) -> InferenceRequest {
        InferenceRequest {
            request_id: id.to_owned(),
            robot_id: robot.to_owned(),
            loss_scale_id: Some("task-loss-v1".to_owned()),
            evsi_units: Some(10),
            p99_return_ticks: Some(2),
            action_buffer_slack_ticks: Some(10),
            deadline_slack_ticks: Some(10),
            local_projection_dispatch_guard_ticks: 0,
            actuation_aoi_upper_ticks: Some(3),
            max_actuation_aoi_ticks: 5,
            poma_upper_ppm: Some(10_000),
            poma_limit_ppm: 50_000,
            gpu_micros: 10,
            memory_bandwidth_units: 10,
            network_bytes: 10,
            kv_cache_growth_tokens: Some(10),
            identity_valid: true,
            evidence_digest_valid: true,
            clock_mapping_valid: true,
            local_fallback_ready: true,
            privacy_policy_allows_remote_data: true,
            local_hard_safety_preserved: true,
            service_floor_due: false,
            starvation_ticks: 1,
            max_starvation_ticks: 10,
        }
    }

    fn budget() -> ResourceBudget {
        ResourceBudget {
            gpu_micros: 100,
            memory_bandwidth_units: 100,
            network_bytes: 100,
            kv_cache_growth_tokens: 100,
        }
    }

    fn cursor() -> SchedulerCursor {
        SchedulerCursor {
            last_robot_id: None,
        }
    }

    #[test]
    fn admits_eligible_positive_value_request() {
        let report = admit_batch(
            SchedulerPolicy::Luminous,
            &cursor(),
            budget(),
            &[request("q1", "r1")],
        )
        .unwrap();
        assert_eq!(report.admitted_request_ids, vec!["q1"]);
        assert!(report.service_floors_met);
        assert!(!report.actuation_authorized);
    }

    #[test]
    fn zero_evsi_is_declined_even_when_fresh() {
        let mut q = request("q1", "r1");
        q.evsi_units = Some(0);
        let report = admit_batch(SchedulerPolicy::Luminous, &cursor(), budget(), &[q]).unwrap();
        assert!(report.admitted_request_ids.is_empty());
        assert_eq!(
            report.requests[0].disposition,
            RequestDisposition::Rejected(RejectReason::ZeroEvsi)
        );
    }

    #[test]
    fn unknown_age_fails_closed() {
        let mut q = request("q1", "r1");
        q.actuation_aoi_upper_ticks = None;
        let report = admit_batch(SchedulerPolicy::Luminous, &cursor(), budget(), &[q]).unwrap();
        assert_eq!(
            report.requests[0].disposition,
            RequestDisposition::Rejected(RejectReason::UnknownActuationAge)
        );
    }

    #[test]
    fn stale_age_is_rejected() {
        let mut q = request("q1", "r1");
        q.actuation_aoi_upper_ticks = Some(6);
        let report = admit_batch(SchedulerPolicy::Luminous, &cursor(), budget(), &[q]).unwrap();
        assert_eq!(
            report.requests[0].disposition,
            RequestDisposition::Rejected(RejectReason::StaleActuationAge)
        );
    }

    #[test]
    fn unknown_timing_is_rejected() {
        let mut q = request("q1", "r1");
        q.p99_return_ticks = None;
        let report = admit_batch(SchedulerPolicy::Luminous, &cursor(), budget(), &[q]).unwrap();
        assert_eq!(
            report.requests[0].disposition,
            RequestDisposition::Rejected(RejectReason::UnknownTimingBound)
        );
    }

    #[test]
    fn late_result_cannot_enter_batch() {
        let mut q = request("q1", "r1");
        q.p99_return_ticks = Some(11);
        q.action_buffer_slack_ticks = Some(10);
        let report = admit_batch(SchedulerPolicy::Luminous, &cursor(), budget(), &[q]).unwrap();
        assert_eq!(
            report.requests[0].disposition,
            RequestDisposition::Rejected(RejectReason::ResultReturnsTooLate)
        );
    }

    #[test]
    fn projection_dispatch_guard_is_reserved_before_admission() {
        let mut q = request("q1", "r1");
        q.p99_return_ticks = Some(8);
        q.action_buffer_slack_ticks = Some(10);
        q.deadline_slack_ticks = Some(12);
        q.local_projection_dispatch_guard_ticks = 3;
        let report = admit_batch(SchedulerPolicy::Luminous, &cursor(), budget(), &[q]).unwrap();
        assert_eq!(
            report.requests[0].disposition,
            RequestDisposition::Rejected(RejectReason::ResultReturnsTooLate)
        );
    }

    #[test]
    fn result_and_guard_exactly_at_slack_is_admissible() {
        let mut q = request("q1", "r1");
        q.p99_return_ticks = Some(8);
        q.action_buffer_slack_ticks = Some(10);
        q.deadline_slack_ticks = Some(12);
        q.local_projection_dispatch_guard_ticks = 2;
        let report = admit_batch(SchedulerPolicy::Luminous, &cursor(), budget(), &[q]).unwrap();
        assert_eq!(report.admitted_request_ids, vec!["q1"]);
    }

    #[test]
    fn excessive_poma_is_rejected() {
        let mut q = request("q1", "r1");
        q.poma_upper_ppm = Some(80_000);
        let report = admit_batch(SchedulerPolicy::Luminous, &cursor(), budget(), &[q]).unwrap();
        assert_eq!(
            report.requests[0].disposition,
            RequestDisposition::Rejected(RejectReason::MissRiskTooHigh)
        );
    }

    #[test]
    fn safety_work_cannot_be_displaced() {
        let mut q = request("q1", "r1");
        q.local_hard_safety_preserved = false;
        let report = admit_batch(SchedulerPolicy::Luminous, &cursor(), budget(), &[q]).unwrap();
        assert_eq!(
            report.requests[0].disposition,
            RequestDisposition::Rejected(RejectReason::LocalSafetyWouldBeDisplaced)
        );
    }

    #[test]
    fn local_fallback_is_mandatory() {
        let mut q = request("q1", "r1");
        q.local_fallback_ready = false;
        let report = admit_batch(SchedulerPolicy::Luminous, &cursor(), budget(), &[q]).unwrap();
        assert_eq!(
            report.requests[0].disposition,
            RequestDisposition::Rejected(RejectReason::NoLocalFallback)
        );
    }

    #[test]
    fn privacy_denial_is_mandatory() {
        let mut q = request("q1", "r1");
        q.privacy_policy_allows_remote_data = false;
        let report = admit_batch(SchedulerPolicy::Luminous, &cursor(), budget(), &[q]).unwrap();
        assert_eq!(
            report.requests[0].disposition,
            RequestDisposition::Rejected(RejectReason::PrivacyPolicyDenied)
        );
    }

    #[test]
    fn service_floor_precedes_optional_urgency() {
        let mut floor = request("a_floor", "r_floor");
        floor.service_floor_due = true;
        floor.starvation_ticks = 7;
        let mut urgent = request("b_urgent", "r_urgent");
        urgent.action_buffer_slack_ticks = Some(1);
        urgent.deadline_slack_ticks = Some(1);
        let tight = ResourceBudget {
            gpu_micros: 10,
            memory_bandwidth_units: 10,
            network_bytes: 10,
            kv_cache_growth_tokens: 10,
        };
        let report = admit_batch(
            SchedulerPolicy::Luminous,
            &cursor(),
            tight,
            &[urgent, floor],
        )
        .unwrap();
        assert_eq!(report.admitted_request_ids, vec!["a_floor"]);
        assert!(report.service_floors_met);
    }

    #[test]
    fn already_overdue_service_is_admitted_and_breach_is_reported() {
        let mut q = request("q_due", "r1");
        q.service_floor_due = true;
        q.starvation_ticks = 12;
        q.max_starvation_ticks = 10;
        let report = admit_batch(SchedulerPolicy::Luminous, &cursor(), budget(), &[q]).unwrap();
        assert_eq!(report.admitted_request_ids, vec!["q_due"]);
        assert!(report.service_floor_violations.is_empty());
        assert_eq!(report.preexisting_starvation_breaches, vec!["r1"]);
        assert!(!report.service_floors_met);
    }

    #[test]
    fn zero_starvation_limit_is_invalid_evidence() {
        let mut q = request("q_due", "r1");
        q.max_starvation_ticks = 0;
        let report = admit_batch(SchedulerPolicy::Luminous, &cursor(), budget(), &[q]).unwrap();
        assert_eq!(
            report.requests[0].disposition,
            RequestDisposition::Rejected(RejectReason::StarvationEvidenceInvalid)
        );
    }

    #[test]
    fn service_floor_miss_is_reported_not_hidden_by_throughput() {
        let mut floor = request("q_floor", "r1");
        floor.service_floor_due = true;
        let report = admit_batch(
            SchedulerPolicy::Luminous,
            &cursor(),
            ResourceBudget {
                gpu_micros: 0,
                memory_bandwidth_units: 0,
                network_bytes: 0,
                kv_cache_growth_tokens: 0,
            },
            &[floor],
        )
        .unwrap();
        assert!(!report.service_floors_met);
        assert_eq!(report.service_floor_violations, vec!["r1"]);
    }

    #[test]
    fn unknown_kv_cache_growth_is_rejected() {
        let mut q = request("q1", "r1");
        q.kv_cache_growth_tokens = None;
        let report = admit_batch(SchedulerPolicy::Luminous, &cursor(), budget(), &[q]).unwrap();
        assert_eq!(
            report.requests[0].disposition,
            RequestDisposition::Rejected(RejectReason::UnknownKvCacheDemand)
        );
    }

    #[test]
    fn kv_cache_growth_is_an_independent_capacity_constraint() {
        let mut q1 = request("q1", "r1");
        let mut q2 = request("q2", "r2");
        q1.kv_cache_growth_tokens = Some(60);
        q2.kv_cache_growth_tokens = Some(60);
        let tight = ResourceBudget {
            gpu_micros: 100,
            memory_bandwidth_units: 100,
            network_bytes: 100,
            kv_cache_growth_tokens: 100,
        };
        let report = admit_batch(SchedulerPolicy::RoundRobin, &cursor(), tight, &[q1, q2]).unwrap();
        assert_eq!(report.admitted_request_ids.len(), 1);
        assert_eq!(report.resource_use.kv_cache_growth_tokens, 60);
    }

    #[test]
    fn memory_bandwidth_is_a_first_class_batch_constraint() {
        let mut q1 = request("q1", "r1");
        let mut q2 = request("q2", "r2");
        q1.memory_bandwidth_units = 60;
        q2.memory_bandwidth_units = 60;
        let tight = ResourceBudget {
            gpu_micros: 100,
            memory_bandwidth_units: 100,
            network_bytes: 100,
            kv_cache_growth_tokens: 100,
        };
        let report = admit_batch(SchedulerPolicy::RoundRobin, &cursor(), tight, &[q1, q2]).unwrap();
        assert_eq!(report.admitted_request_ids.len(), 1);
        assert!(report.resource_use.memory_bandwidth_units <= 100);
    }

    #[test]
    fn evsi_per_compute_refuses_unrelated_loss_scales() {
        let a = request("q1", "r1");
        let mut b = request("q2", "r2");
        b.loss_scale_id = Some("other-task-v1".to_owned());
        assert_eq!(
            admit_batch(
                SchedulerPolicy::EvsiPerCompute,
                &cursor(),
                budget(),
                &[a, b],
            ),
            Err(ReferenceError::IncomparableLossScales)
        );
    }

    #[test]
    fn evsi_per_compute_orders_higher_value_density_first() {
        let mut a = request("q1", "r1");
        let mut b = request("q2", "r2");
        a.evsi_units = Some(10);
        a.gpu_micros = 20;
        b.evsi_units = Some(9);
        b.gpu_micros = 10;
        let tight = ResourceBudget {
            gpu_micros: 10,
            memory_bandwidth_units: 20,
            network_bytes: 20,
            kv_cache_growth_tokens: 10,
        };
        let report =
            admit_batch(SchedulerPolicy::EvsiPerCompute, &cursor(), tight, &[a, b]).unwrap();
        assert_eq!(report.admitted_request_ids, vec!["q2"]);
    }

    #[test]
    fn edf_uses_minimum_of_action_buffer_and_deadline_slack() {
        let mut long = request("q_long", "r1");
        long.action_buffer_slack_ticks = Some(8);
        long.deadline_slack_ticks = Some(20);
        let mut short = request("q_short", "r2");
        short.action_buffer_slack_ticks = Some(12);
        short.deadline_slack_ticks = Some(4);
        let tight = ResourceBudget {
            gpu_micros: 10,
            memory_bandwidth_units: 10,
            network_bytes: 10,
            kv_cache_growth_tokens: 10,
        };
        let report = admit_batch(
            SchedulerPolicy::EarliestDeadlineFirst,
            &cursor(),
            tight,
            &[long, short],
        )
        .unwrap();
        assert_eq!(report.admitted_request_ids, vec!["q_short"]);
    }

    #[test]
    fn round_robin_cursor_rotates_robot_priority() {
        let a = request("q_a", "a");
        let b = request("q_b", "b");
        let c = request("q_c", "c");
        let tight = ResourceBudget {
            gpu_micros: 10,
            memory_bandwidth_units: 10,
            network_bytes: 10,
            kv_cache_growth_tokens: 10,
        };
        let report = admit_batch(
            SchedulerPolicy::RoundRobin,
            &SchedulerCursor {
                last_robot_id: Some("a".to_owned()),
            },
            tight,
            &[a, b, c],
        )
        .unwrap();
        assert_eq!(report.admitted_request_ids, vec!["q_b"]);
    }

    #[test]
    fn duplicate_request_ids_are_rejected() {
        let a = request("same", "r1");
        let b = request("same", "r2");
        assert_eq!(
            admit_batch(SchedulerPolicy::Luminous, &cursor(), budget(), &[a, b],),
            Err(ReferenceError::DuplicateRequestId("same".to_owned()))
        );
    }

    #[test]
    fn round_robin_cursor_is_preserved_when_no_request_is_admitted() {
        let report = admit_batch(
            SchedulerPolicy::RoundRobin,
            &SchedulerCursor {
                last_robot_id: Some("robot-m".to_owned()),
            },
            ResourceBudget {
                gpu_micros: 0,
                memory_bandwidth_units: 0,
                network_bytes: 0,
                kv_cache_growth_tokens: 0,
            },
            &[request("q1", "robot-a")],
        )
        .unwrap();
        assert!(report.admitted_request_ids.is_empty());
        assert_eq!(report.next_cursor_robot_id.as_deref(), Some("robot-m"));
    }

    #[test]
    fn empty_batch_is_a_valid_noop() {
        let report = admit_batch(SchedulerPolicy::Luminous, &cursor(), budget(), &[]).unwrap();
        assert!(report.admitted_request_ids.is_empty());
        assert!(report.service_floors_met);
        assert!(!report.actuation_authorized);
    }
}
