//! Independent evaluation of essential-service continuity and accepted restoration.
//! This is a contract evaluator, not a reliability predictor, safety case,
//! certification tool, or physical-plant simulator. Results remain per-service.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum EventKind {
    ComponentFailure, BufferSupplyActive, RepairStarted, RepairCompleted,
    PostTestPassed, PostTestFailed, ConfigurationUpdated, AuthorizedRelease,
    ServiceUnavailable, SpareArrived, CommonDependencyFailure, SpareRejected,
    /// Global authorization gate; its rate field is not a flow measurement.
    SafetyGateFailed,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ServiceContract {
    pub service_id: String, pub unit: String,
    pub minimum_rate: f64, pub nominal_rate: f64, pub initial_rate: f64,
}
#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceEvent {
    pub event_id: String, pub at_seconds: u64, pub kind: EventKind,
    pub service_id: String, pub component_id: String,
    /// Exact revision installed or tested at the event time.
    pub component_revision_digest: String,
    /// Work-order identity for maintenance, spare, test and release events.
    pub maintenance_task_id: Option<String>,
    /// Stable identity for one service/component fault-and-restoration episode.
    pub fault_episode_id: Option<String>,
    pub delivered_rate_after: f64, pub evidence_ref: String,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpareState { NotRequired, CompatibleOnboard, CompatibleArrivesLater, WrongRevision, Unknown }
#[derive(Clone, Debug, PartialEq)]
pub struct MaintenanceTask {
    pub task_id: String, pub service_id: String, pub component_id: String,
    /// Exact fault episode this work order is intended to restore.
    pub fault_episode_id: String,
    pub procedure_approved: bool, pub qualified_personnel_present: bool,
    /// Revision on the failed component at fault observation.
    pub faulted_component_revision_digest: String,
    /// Exact replacement/retained revision authorized by this maintenance task.
    pub target_component_revision_digest: String,
    /// Revision identity of the spare, when the task uses one.
    pub spare_component_revision_digest: Option<String>,
    pub spare_state: SpareState, pub required_tests: Vec<String>,
    pub completed_tests: Vec<String>, pub configuration_updated: bool,
    pub authorized_release: bool,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ServiceContinuityCase {
    pub case_id: String, pub timeline_seconds: u64,
    pub services: Vec<ServiceContract>, pub events: Vec<EvidenceEvent>,
    pub maintenance_tasks: Vec<MaintenanceTask>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContinuityDisposition { NoFloorBreach, FloorBreach, UnsafeNotAuthorized }
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EvaluationError {
    EmptyCaseId, ZeroTimeline, NoServices, EmptyServiceId, EmptyServiceUnit,
    DuplicateServiceId(String), InvalidServiceRate(String), EmptyEventId,
    DuplicateEventId(String), EventOutsideTimeline(String), UnknownEventService(String),
    EmptyComponentId(String), EmptyEvidenceReference(String), InvalidEventRate(String),
    ConflictingRatesAtSameInstant { service_id: String, at_seconds: u64 },
    EmptyTaskId, DuplicateTaskId(String), UnknownTaskService(String), EmptyTaskComponent(String),
    InvalidRevisionDigest(String), UnknownMaintenanceTask(String), MaintenanceTaskIdentityMismatch(String),
    EmptyFaultEpisodeId(String), UnknownFaultEpisode(String), FaultEpisodeIdentityMismatch(String),
    MaintenanceFaultEpisodeMismatch(String), DuplicateFaultEpisodeTask(String),
}
/// Deficit integral is in that service's unit-hours; never sum across unlike services.
#[derive(Clone, Debug, PartialEq)]
pub struct ServiceEvaluation {
    pub service_id: String, pub unit: String, pub minimum_rate: f64,
    pub restoration_episodes: Vec<RestorationEpisodeEvaluation>,
    pub minimum_observed_rate: f64, pub time_below_floor_seconds: u64,
    pub floor_breach_episodes: u32, pub first_floor_breach_seconds: Option<u64>,
    /// 0 if the floor was never breached; Some(t) for the latest recovery;
    /// None if the service ends the observation window below its floor.
    pub last_floor_recovery_seconds: Option<u64>,
    pub shortfall_rate_hours: f64, pub first_fault_seconds: Option<u64>,
    pub repair_started_seconds: Option<u64>, pub spare_arrived_seconds: Option<u64>,
    pub repair_completed_seconds: Option<u64>, pub post_test_passed_seconds: Option<u64>,
    pub configuration_updated_seconds: Option<u64>,
    pub accepted_restoration_seconds: Option<u64>, pub tasr_seconds: Option<u64>,
    pub rejected_release_attempts: u32,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RestorationEpisodeDisposition {
    Restored,
    Open,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RestorationEpisodeEvaluation {
    pub fault_episode_id: String,
    pub component_id: String,
    pub fault_observed_seconds: u64,
    pub repair_started_seconds: Option<u64>,
    pub spare_arrived_seconds: Option<u64>,
    pub repair_completed_seconds: Option<u64>,
    pub post_test_passed_seconds: Option<u64>,
    pub configuration_updated_seconds: Option<u64>,
    pub accepted_restoration_seconds: Option<u64>,
    pub tasr_seconds: Option<u64>,
    pub rejected_release_attempts: u32,
    pub disposition: RestorationEpisodeDisposition,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServiceContinuityEvaluation {
    pub case_id: String, pub disposition: ContinuityDisposition,
    pub safety_gate_failed: bool, pub service_floor_breach_count: u32,
    pub first_floor_breach_seconds: Option<u64>,
    /// Latest time at which every declared service was at or above its floor.
    /// None means at least one service remained below floor or a safety gate failed.
    pub all_declared_service_floors_restored_seconds: Option<u64>,
    /// Earliest accepted restoration among services; inspect per-service values too.
    pub earliest_service_accepted_restoration_seconds: Option<u64>,
    pub services: Vec<ServiceEvaluation>,
}

/// Recompute service-floor exposure and accepted restoration from event evidence.
pub fn evaluate_service_continuity(
    case: &ServiceContinuityCase,
) -> Result<ServiceContinuityEvaluation, EvaluationError> {
    validate_case(case)?;
    let safety_gate_events: Vec<&EvidenceEvent> = case.events.iter()
        .filter(|event| event.kind == EventKind::SafetyGateFailed)
        .collect();
    let safety_failed = !safety_gate_events.is_empty();
    let mut services = Vec::with_capacity(case.services.len());
    for contract in &case.services {
        let mut events: Vec<&EvidenceEvent> = case.events.iter()
            .filter(|e| e.service_id == contract.service_id).collect();
        events.sort_by(|a,b| a.at_seconds.cmp(&b.at_seconds)
            .then_with(|| a.kind.cmp(&b.kind)).then_with(|| a.event_id.cmp(&b.event_id)));
        let tasks: Vec<&MaintenanceTask> = case.maintenance_tasks.iter()
            .filter(|t| t.service_id == contract.service_id).collect();
        services.push(evaluate_service(
            contract,
            &events,
            &tasks,
            &safety_gate_events,
            case.timeline_seconds,
        ));
    }
    let breaches: u32 = services.iter().map(|s| s.floor_breach_episodes).sum();
    let first_breach = services.iter().filter_map(|s| s.first_floor_breach_seconds).min();
    let earliest_restoration = services.iter().filter_map(|s| s.accepted_restoration_seconds).min();
    let all_floors_restored = if safety_failed { None } else {
        services.iter().map(|s| s.last_floor_recovery_seconds)
            .collect::<Option<Vec<u64>>>().map(|times| times.into_iter().max().unwrap_or(0))
    };
    let disposition = if safety_failed { ContinuityDisposition::UnsafeNotAuthorized }
        else if breaches > 0 { ContinuityDisposition::FloorBreach }
        else { ContinuityDisposition::NoFloorBreach };
    Ok(ServiceContinuityEvaluation {
        case_id: case.case_id.clone(), disposition, safety_gate_failed: safety_failed,
        service_floor_breach_count: breaches, first_floor_breach_seconds: first_breach,
        all_declared_service_floors_restored_seconds: all_floors_restored,
        earliest_service_accepted_restoration_seconds: earliest_restoration, services,
    })
}

fn is_sha256_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn validate_case(case: &ServiceContinuityCase) -> Result<(), EvaluationError> {
    if case.case_id.trim().is_empty() { return Err(EvaluationError::EmptyCaseId); }
    if case.timeline_seconds == 0 { return Err(EvaluationError::ZeroTimeline); }
    if case.services.is_empty() { return Err(EvaluationError::NoServices); }
    let mut service_ids = BTreeSet::new();
    let mut max_rates = BTreeMap::new();
    for s in &case.services {
        if s.service_id.trim().is_empty() { return Err(EvaluationError::EmptyServiceId); }
        if s.unit.trim().is_empty() { return Err(EvaluationError::EmptyServiceUnit); }
        if !service_ids.insert(s.service_id.as_str()) {
            return Err(EvaluationError::DuplicateServiceId(s.service_id.clone()));
        }
        let rates = [s.minimum_rate, s.nominal_rate, s.initial_rate];
        if rates.iter().any(|v| !v.is_finite() || *v < 0.0)
            || s.nominal_rate <= 0.0 || s.minimum_rate > s.nominal_rate
            || s.initial_rate > s.nominal_rate {
            return Err(EvaluationError::InvalidServiceRate(s.service_id.clone()));
        }
        max_rates.insert(s.service_id.as_str(), s.nominal_rate);
    }
    let mut event_ids = BTreeSet::new();
    let mut rates_at_time: BTreeMap<(String,u64),u64> = BTreeMap::new();
    for e in &case.events {
        if e.event_id.trim().is_empty() { return Err(EvaluationError::EmptyEventId); }
        if !event_ids.insert(e.event_id.as_str()) {
            return Err(EvaluationError::DuplicateEventId(e.event_id.clone()));
        }
        if e.at_seconds > case.timeline_seconds {
            return Err(EvaluationError::EventOutsideTimeline(e.event_id.clone()));
        }
        let Some(max_rate) = max_rates.get(e.service_id.as_str()) else {
            return Err(EvaluationError::UnknownEventService(e.service_id.clone()));
        };
        if e.component_id.trim().is_empty() { return Err(EvaluationError::EmptyComponentId(e.event_id.clone())); }
        if !is_sha256_digest(&e.component_revision_digest) {
            return Err(EvaluationError::InvalidRevisionDigest(e.event_id.clone()));
        }
        if e.evidence_ref.trim().is_empty() {
            return Err(EvaluationError::EmptyEvidenceReference(e.event_id.clone()));
        }
        if !e.delivered_rate_after.is_finite() || e.delivered_rate_after < 0.0
            || e.delivered_rate_after > *max_rate {
            return Err(EvaluationError::InvalidEventRate(e.event_id.clone()));
        }
        if e.kind != EventKind::SafetyGateFailed {
            let key = (e.service_id.clone(), e.at_seconds);
            if let Some(previous) = rates_at_time.insert(key, e.delivered_rate_after.to_bits()) {
                if previous != e.delivered_rate_after.to_bits() {
                    return Err(EvaluationError::ConflictingRatesAtSameInstant {
                        service_id: e.service_id.clone(), at_seconds: e.at_seconds,
                    });
                }
            }
        }
    }
    let mut fault_episode_identity: BTreeMap<&str, (&str, &str)> = BTreeMap::new();
    for event in &case.events {
        let is_fault = matches!(
            event.kind,
            EventKind::ComponentFailure
                | EventKind::ServiceUnavailable
                | EventKind::CommonDependencyFailure
        );
        if is_fault {
            let Some(episode_id) = event.fault_episode_id.as_deref() else {
                return Err(EvaluationError::EmptyFaultEpisodeId(event.event_id.clone()));
            };
            if episode_id.trim().is_empty() {
                return Err(EvaluationError::EmptyFaultEpisodeId(event.event_id.clone()));
            }
            match fault_episode_identity.get(episode_id) {
                Some((service_id, component_id))
                    if *service_id != event.service_id || *component_id != event.component_id =>
                {
                    return Err(EvaluationError::FaultEpisodeIdentityMismatch(
                        episode_id.to_owned(),
                    ));
                }
                Some(_) => {}
                None => {
                    fault_episode_identity.insert(
                        episode_id,
                        (event.service_id.as_str(), event.component_id.as_str()),
                    );
                }
            }
        }
    }

    let mut task_ids = BTreeSet::new();
    let mut task_episode_ids = BTreeSet::new();
    let task_by_id: BTreeMap<&str, &MaintenanceTask> = case.maintenance_tasks.iter()
        .map(|task| (task.task_id.as_str(), task)).collect();
    for task in &case.maintenance_tasks {
        if task.task_id.trim().is_empty() { return Err(EvaluationError::EmptyTaskId); }
        if !task_ids.insert(task.task_id.as_str()) {
            return Err(EvaluationError::DuplicateTaskId(task.task_id.clone()));
        }
        if !service_ids.contains(task.service_id.as_str()) {
            return Err(EvaluationError::UnknownTaskService(task.service_id.clone()));
        }
        if task.component_id.trim().is_empty() {
            return Err(EvaluationError::EmptyTaskComponent(task.task_id.clone()));
        }
        if task.fault_episode_id.trim().is_empty() {
            return Err(EvaluationError::EmptyFaultEpisodeId(task.task_id.clone()));
        }
        if !task_episode_ids.insert(task.fault_episode_id.as_str()) {
            return Err(EvaluationError::DuplicateFaultEpisodeTask(
                task.fault_episode_id.clone(),
            ));
        }
        let Some((fault_service, fault_component)) =
            fault_episode_identity.get(task.fault_episode_id.as_str())
        else {
            return Err(EvaluationError::UnknownFaultEpisode(task.task_id.clone()));
        };
        if *fault_service != task.service_id.as_str() || *fault_component != task.component_id.as_str() {
            return Err(EvaluationError::FaultEpisodeIdentityMismatch(
                task.fault_episode_id.clone(),
            ));
        }
        if !is_sha256_digest(&task.faulted_component_revision_digest)
            || !is_sha256_digest(&task.target_component_revision_digest)
            || task.spare_component_revision_digest.as_deref()
                .is_some_and(|digest| !is_sha256_digest(digest))
        {
            return Err(EvaluationError::InvalidRevisionDigest(task.task_id.clone()));
        }
    }

    for event in &case.events {
        if let Some(task_id) = event.maintenance_task_id.as_deref() {
            let Some(task) = task_by_id.get(task_id) else {
                return Err(EvaluationError::UnknownMaintenanceTask(task_id.to_owned()));
            };
            if task.service_id != event.service_id || task.component_id != event.component_id {
                return Err(EvaluationError::MaintenanceTaskIdentityMismatch(
                    event.event_id.clone(),
                ));
            }
            if event.fault_episode_id.as_deref() != Some(task.fault_episode_id.as_str()) {
                return Err(EvaluationError::MaintenanceFaultEpisodeMismatch(
                    event.event_id.clone(),
                ));
            }
            let requires_target_revision = matches!(
                event.kind,
                EventKind::RepairCompleted
                    | EventKind::PostTestPassed
                    | EventKind::PostTestFailed
                    | EventKind::ConfigurationUpdated
                    | EventKind::SpareArrived
                    | EventKind::AuthorizedRelease
            );
            if requires_target_revision
                && event.component_revision_digest != task.target_component_revision_digest
            {
                return Err(EvaluationError::InvalidRevisionDigest(event.event_id.clone()));
            }
            if event.kind == EventKind::ComponentFailure
                && event.component_revision_digest != task.faulted_component_revision_digest
            {
                return Err(EvaluationError::InvalidRevisionDigest(event.event_id.clone()));
            }
        }
    }
    Ok(())
}
fn first_time(events: &[&EvidenceEvent], kind: EventKind) -> Option<u64> {
    events.iter().filter(|e| e.kind == kind).map(|e| e.at_seconds).min()
}
fn integrate(
    c: &ServiceContract, rate: f64, start: u64, end: u64, active: &mut bool,
    seconds_below: &mut u64, episodes: &mut u32, first: &mut Option<u64>, deficit_hours: &mut f64,
) {
    if end <= start || rate >= c.minimum_rate { return; }
    if !*active {
        *episodes = episodes.saturating_add(1); first.get_or_insert(start); *active = true;
    }
    let dt = end - start;
    *seconds_below = seconds_below.saturating_add(dt);
    *deficit_hours += (c.minimum_rate - rate) * (dt as f64 / 3600.0);
}
fn evaluate_service(
    c: &ServiceContract,
    events: &[&EvidenceEvent],
    tasks: &[&MaintenanceTask],
    safety_gate_events: &[&EvidenceEvent],
    timeline: u64,
) -> ServiceEvaluation {
    let first_fault = events.iter().filter(|e| matches!(e.kind, EventKind::ComponentFailure
        | EventKind::ServiceUnavailable | EventKind::CommonDependencyFailure))
        .map(|e| e.at_seconds).min();
    let mut rate = c.initial_rate;
    let mut min_rate = rate;
    let mut prev = 0_u64;
    let mut active = c.initial_rate < c.minimum_rate;
    let mut seconds_below = 0_u64;
    let mut episodes = u32::from(active);
    let mut first_breach = active.then_some(0);
    let mut last_floor_recovery = (!active).then_some(0);
    let mut deficit_hours = 0.0;
    let mut restoration = None;
    let mut rejected = 0_u32;
    let mut i = 0;
    while i < events.len() {
        let t = events[i].at_seconds;
        integrate(c, rate, prev, t, &mut active, &mut seconds_below,
            &mut episodes, &mut first_breach, &mut deficit_hours);
        let mut j = i + 1;
        while j < events.len() && events[j].at_seconds == t { j += 1; }
        for e in &events[i..j] {
            match e.kind {
                EventKind::SafetyGateFailed => {}, // global gate, not a flow observation
                EventKind::AuthorizedRelease => {
                    if e.delivered_rate_after >= c.minimum_rate
                        && release_accepted(e, events, tasks, safety_gate_events) {
                        rate = e.delivered_rate_after; restoration.get_or_insert(t);
                    } else { rejected = rejected.saturating_add(1); }
                }
                // Maintenance workflow evidence does not itself measure delivered
                // service or restore operational flow. Only an accepted release can
                // transition service delivery after the repair/requalification chain.
                EventKind::RepairStarted
                | EventKind::RepairCompleted
                | EventKind::PostTestPassed
                | EventKind::PostTestFailed
                | EventKind::ConfigurationUpdated
                | EventKind::SpareArrived
                | EventKind::SpareRejected => {},
                _ => rate = e.delivered_rate_after,
            }
        }
        min_rate = min_rate.min(rate);
        if rate >= c.minimum_rate {
            if active { last_floor_recovery = Some(t); }
            active = false;
        } else if !active {
            episodes = episodes.saturating_add(1);
            first_breach.get_or_insert(t);
            active = true;
        }
        prev = t; i = j;
    }
    integrate(c, rate, prev, timeline, &mut active, &mut seconds_below,
        &mut episodes, &mut first_breach, &mut deficit_hours);
    let tasr = match (first_fault, restoration) {
        (Some(fault), Some(done)) => done.checked_sub(fault), _ => None,
    };
    ServiceEvaluation {
        service_id: c.service_id.clone(), unit: c.unit.clone(), minimum_rate: c.minimum_rate,
        minimum_observed_rate: min_rate, time_below_floor_seconds: seconds_below,
        floor_breach_episodes: episodes, first_floor_breach_seconds: first_breach,
        last_floor_recovery_seconds: if active { None } else { last_floor_recovery },
        shortfall_rate_hours: deficit_hours, first_fault_seconds: first_fault,
        repair_started_seconds: first_time(events, EventKind::RepairStarted),
        spare_arrived_seconds: first_time(events, EventKind::SpareArrived),
        repair_completed_seconds: events.iter().filter(|e| e.kind == EventKind::RepairCompleted)
            .map(|e| e.at_seconds).max(),
        post_test_passed_seconds: events.iter().filter(|e| e.kind == EventKind::PostTestPassed)
            .map(|e| e.at_seconds).max(),
        configuration_updated_seconds: events.iter().filter(|e| e.kind == EventKind::ConfigurationUpdated)
            .map(|e| e.at_seconds).max(),
        accepted_restoration_seconds: restoration, tasr_seconds: tasr,
        rejected_release_attempts: rejected,
        restoration_episodes: evaluate_restoration_episodes(
            events,
            tasks,
            safety_gate_events,
            c.minimum_rate,
        ),
    }
}
fn release_accepted(
    release: &EvidenceEvent,
    events: &[&EvidenceEvent],
    tasks: &[&MaintenanceTask],
    safety_gate_events: &[&EvidenceEvent],
) -> bool {
    // Authorization is evaluated against the case-wide gate prefix available at this
    // release, even when the gate record is attached to a different service's event log.
    // A later safety-gate failure does not retroactively veto this earlier release.
    if safety_gate_events.iter().any(|event| event.at_seconds <= release.at_seconds) {
        return false;
    }
    let Some(task_id) = release.maintenance_task_id.as_deref() else { return false; };
    let Some(episode_id) = release.fault_episode_id.as_deref() else { return false; };
    let Some(task) = tasks.iter().copied().find(|task| {
        task.task_id == task_id
            && task.fault_episode_id == episode_id
            && task.service_id == release.service_id
            && task.component_id == release.component_id
    }) else {
        return false;
    };
    if release.component_revision_digest != task.target_component_revision_digest {
        return false;
    }

    let evidence: Vec<&EvidenceEvent> = events.iter().copied()
        .filter(|event| {
            event.fault_episode_id.as_deref() == Some(episode_id)
                && event.component_id == release.component_id
                && event.at_seconds < release.at_seconds
        })
        .collect();
    let latest_fault = evidence.iter().filter(|event| matches!(
        event.kind,
        EventKind::ComponentFailure
            | EventKind::ServiceUnavailable
            | EventKind::CommonDependencyFailure
    )).max_by_key(|event| event.at_seconds);
    let Some(fault) = latest_fault else { return false; };
    if fault.component_revision_digest != task.faulted_component_revision_digest {
        return false;
    }

    let task_evidence: Vec<&EvidenceEvent> = evidence.iter().copied()
        .filter(|event| {
            event.maintenance_task_id.as_deref() == Some(task_id)
                && event.component_revision_digest == task.target_component_revision_digest
        })
        .collect();
    let first_repair_started = task_evidence.iter()
        .filter(|event| event.kind == EventKind::RepairStarted)
        .map(|event| event.at_seconds).min();
    let Some(first_repair_started) = first_repair_started else { return false; };
    let last_repair_started = task_evidence.iter()
        .filter(|event| event.kind == EventKind::RepairStarted)
        .map(|event| event.at_seconds).max();
    let Some(last_repair_started) = last_repair_started else { return false; };
    let repair_completed = task_evidence.iter()
        .filter(|event| event.kind == EventKind::RepairCompleted)
        .map(|event| event.at_seconds).max();
    let Some(repair) = repair_completed else { return false; };
    if first_repair_started <= fault.at_seconds || repair <= last_repair_started {
        return false;
    }

    let last_failed_test = task_evidence.iter()
        .filter(|event| event.kind == EventKind::PostTestFailed)
        .map(|event| event.at_seconds).max();
    let pass = task_evidence.iter()
        .filter(|event| event.kind == EventKind::PostTestPassed)
        .map(|event| event.at_seconds).max();
    let Some(pass) = pass else { return false; };
    if pass <= repair || last_failed_test.is_some_and(|failed| failed >= pass) {
        return false;
    }
    if !task_evidence.iter().any(|event| {
        event.kind == EventKind::ConfigurationUpdated && event.at_seconds > pass
    }) {
        return false;
    }

    let spare_arrival_before_repair = task_evidence.iter()
        .filter(|event| event.kind == EventKind::SpareArrived)
        .map(|event| event.at_seconds)
        .any(|arrival| arrival < first_repair_started);
    let spare_matches = match task.spare_state {
        SpareState::NotRequired => task.spare_component_revision_digest.is_none(),
        SpareState::CompatibleOnboard => task.spare_component_revision_digest.as_deref()
            == Some(task.target_component_revision_digest.as_str()),
        SpareState::CompatibleArrivesLater => spare_arrival_before_repair
            && task.spare_component_revision_digest.as_deref()
                == Some(task.target_component_revision_digest.as_str()),
        SpareState::WrongRevision | SpareState::Unknown => false,
    };
    let tests_ok = task.required_tests.iter().all(|required|
        task.completed_tests.iter().any(|done| done == required));
    task.procedure_approved && task.qualified_personnel_present && tests_ok
        && task.configuration_updated && task.authorized_release && spare_matches
}

fn evaluate_restoration_episodes(
    events: &[&EvidenceEvent],
    tasks: &[&MaintenanceTask],
    safety_gate_events: &[&EvidenceEvent],
    minimum_rate: f64,
) -> Vec<RestorationEpisodeEvaluation> {
    let mut fault_events: BTreeMap<&str, Vec<&EvidenceEvent>> = BTreeMap::new();
    for event in events.iter().copied().filter(|event| matches!(
        event.kind,
        EventKind::ComponentFailure
            | EventKind::ServiceUnavailable
            | EventKind::CommonDependencyFailure
    )) {
        if let Some(episode_id) = event.fault_episode_id.as_deref() {
            fault_events.entry(episode_id).or_default().push(event);
        }
    }

    let mut reports = Vec::with_capacity(fault_events.len());
    for (episode_id, episode_faults) in fault_events {
        let first_fault = episode_faults
            .iter()
            .min_by_key(|event| (event.at_seconds, event.event_id.as_str()))
            .expect("fault episode inventory cannot contain an empty event set");
        let task = tasks.iter().copied()
            .find(|task| task.fault_episode_id == episode_id);
        let task_events: Vec<&EvidenceEvent> = task.map(|task| {
            events.iter().copied().filter(|event| {
                event.maintenance_task_id.as_deref() == Some(task.task_id.as_str())
                    && event.fault_episode_id.as_deref() == Some(episode_id)
            }).collect()
        }).unwrap_or_default();
        let repair_started = task_events.iter()
            .filter(|event| event.kind == EventKind::RepairStarted)
            .map(|event| event.at_seconds).min();
        let spare_arrived = task_events.iter()
            .filter(|event| event.kind == EventKind::SpareArrived)
            .map(|event| event.at_seconds).min();
        let repair_completed = task_events.iter()
            .filter(|event| event.kind == EventKind::RepairCompleted)
            .map(|event| event.at_seconds).max();
        let post_test_passed = task_events.iter()
            .filter(|event| event.kind == EventKind::PostTestPassed)
            .map(|event| event.at_seconds).max();
        let configuration_updated = task_events.iter()
            .filter(|event| event.kind == EventKind::ConfigurationUpdated)
            .map(|event| event.at_seconds).max();

        let mut accepted_release: Option<u64> = None;
        let mut rejected_releases = 0_u32;
        for release in events.iter().copied().filter(|event| {
            event.kind == EventKind::AuthorizedRelease
                && event.fault_episode_id.as_deref() == Some(episode_id)
                && event.component_id == first_fault.component_id
        }) {
            if release.delivered_rate_after >= minimum_rate
                && release_accepted(release, events, tasks, safety_gate_events)
            {
                accepted_release = Some(
                    accepted_release.map_or(release.at_seconds, |time| time.min(release.at_seconds))
                );
            } else {
                rejected_releases = rejected_releases.saturating_add(1);
            }
        }
        let tasr_seconds = accepted_release
            .and_then(|done| done.checked_sub(first_fault.at_seconds));
        reports.push(RestorationEpisodeEvaluation {
            fault_episode_id: episode_id.to_owned(),
            component_id: first_fault.component_id.clone(),
            fault_observed_seconds: first_fault.at_seconds,
            repair_started_seconds: repair_started,
            spare_arrived_seconds: spare_arrived,
            repair_completed_seconds: repair_completed,
            post_test_passed_seconds: post_test_passed,
            configuration_updated_seconds: configuration_updated,
            accepted_restoration_seconds: accepted_release,
            tasr_seconds,
            rejected_release_attempts: rejected_releases,
            disposition: if accepted_release.is_some() {
                RestorationEpisodeDisposition::Restored
            } else {
                RestorationEpisodeDisposition::Open
            },
        });
    }
    reports.sort_by(|left, right| {
        left.fault_observed_seconds.cmp(&right.fault_observed_seconds)
            .then_with(|| left.fault_episode_id.cmp(&right.fault_episode_id))
    });
    reports
}

#[cfg(test)]
mod tests {
    use super::*;

    const TASK_ID: &str = "replace_maker_A";
    const REV_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const REV_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn service() -> ServiceContract {
        ServiceContract { service_id:"water".into(), unit:"L/h".into(),
            minimum_rate:40.0, nominal_rate:80.0, initial_rate:80.0 }
    }
    fn event_with_revision(
        id:&str, t:u64, kind:EventKind, rate:f64, revision:&str,
    ) -> EvidenceEvent {
        let fault_episode_id = matches!(
            kind,
            EventKind::ComponentFailure
                | EventKind::ServiceUnavailable
                | EventKind::CommonDependencyFailure
        ).then(|| id.to_owned());
        EvidenceEvent { event_id:id.into(), at_seconds:t, kind, service_id:"water".into(),
            component_id:"maker_A".into(), component_revision_digest:revision.into(),
            maintenance_task_id:None, fault_episode_id,
            delivered_rate_after:rate, evidence_ref:format!("fixture:{id}") }
    }
    fn event(id:&str, t:u64, kind:EventKind, rate:f64) -> EvidenceEvent {
        event_with_revision(id,t,kind,rate,REV_A)
    }
    fn task_event_for(
        task_id: &str, episode_id: &str, id: &str, t: u64,
        kind: EventKind, rate: f64, revision: &str,
    ) -> EvidenceEvent {
        let mut event = event_with_revision(id, t, kind, rate, revision);
        event.maintenance_task_id = Some(task_id.into());
        event.fault_episode_id = Some(episode_id.into());
        event
    }
    fn task_event_with_revision(
        id: &str, t: u64, kind: EventKind, rate: f64, revision: &str,
    ) -> EvidenceEvent {
        task_event_for(TASK_ID, "failure", id, t, kind, rate, revision)
    }
    fn task_event(id: &str, t: u64, kind: EventKind, rate: f64) -> EvidenceEvent {
        task_event_with_revision(id, t, kind, rate, REV_A)
    }
    fn task_for(task_id: &str, episode_id: &str) -> MaintenanceTask {
        MaintenanceTask {
            task_id: task_id.into(),
            service_id: "water".into(),
            component_id: "maker_A".into(),
            fault_episode_id: episode_id.into(),
            procedure_approved: true,
            qualified_personnel_present: true,
            faulted_component_revision_digest: REV_A.into(),
            target_component_revision_digest: REV_A.into(),
            spare_component_revision_digest: Some(REV_A.into()),
            spare_state: SpareState::CompatibleOnboard,
            required_tests: vec!["flow_quality".into()],
            completed_tests: vec!["flow_quality".into()],
            configuration_updated: true,
            authorized_release: true,
        }
    }
    fn task() -> MaintenanceTask {
        task_for(TASK_ID, "failure")
    }
    fn case(events:Vec<EvidenceEvent>, tasks:Vec<MaintenanceTask>, timeline:u64) -> ServiceContinuityCase {
        ServiceContinuityCase { case_id:"synthetic".into(), timeline_seconds:timeline,
            services:vec![service()], events, maintenance_tasks:tasks }
    }

    #[test]
    fn buffer_holds_floor_and_tasr_ends_at_accepted_release() {
        let events=vec![
            event("failure",3600,EventKind::ComponentFailure,45.0),
            event("buffer",3600,EventKind::BufferSupplyActive,45.0),
            task_event("repair_start",7200,EventKind::RepairStarted,45.0),
            task_event("repair_done",14400,EventKind::RepairCompleted,45.0),
            task_event("test",18000,EventKind::PostTestPassed,45.0),
            task_event("config",18600,EventKind::ConfigurationUpdated,45.0),
            task_event("release",19200,EventKind::AuthorizedRelease,80.0)];
        let out=evaluate_service_continuity(&case(events,vec![task()],86400)).unwrap();
        let s=&out.services[0];
        assert_eq!(out.disposition,ContinuityDisposition::NoFloorBreach);
        assert_eq!(s.floor_breach_episodes,0);
        assert_eq!(s.accepted_restoration_seconds,Some(19200));
        assert_eq!(s.tasr_seconds,Some(15600));
        assert_eq!(s.time_below_floor_seconds,0);
    }

    #[test]
    fn exhausted_buffer_breaches_at_86400_and_does_not_restore() {
        let mut exhausted = event("empty",86400,EventKind::ServiceUnavailable,0.0);
        exhausted.fault_episode_id = Some("failure".into());
        let events=vec![
            event("failure",3600,EventKind::ComponentFailure,45.0),
            exhausted,
            task_event("spare",108000,EventKind::SpareArrived,0.0)];
        let mut t=task();
        t.spare_state=SpareState::CompatibleArrivesLater;
        t.completed_tests.clear(); t.configuration_updated=false; t.authorized_release=false;
        let out=evaluate_service_continuity(&case(events,vec![t],172800)).unwrap();
        assert_eq!(out.services[0].first_floor_breach_seconds,Some(86400));
        assert_eq!(out.services[0].time_below_floor_seconds,86400);
        assert_eq!(out.services[0].accepted_restoration_seconds,None);
    }

    #[test]
    fn common_dependency_failure_defeats_nominal_redundancy() {
        let out=evaluate_service_continuity(&case(
            vec![event("controller",1800,EventKind::CommonDependencyFailure,0.0)],vec![],43200)).unwrap();
        assert_eq!(out.services[0].first_floor_breach_seconds,Some(1800));
        assert_eq!(out.services[0].time_below_floor_seconds,41400);
    }

    #[test]
    fn failed_post_test_rejects_release_even_if_a_release_is_attempted() {
        let events=vec![
            event("failure",1800,EventKind::ComponentFailure,0.0),
            task_event("repair_start",2700,EventKind::RepairStarted,0.0),
            task_event("repair",3600,EventKind::RepairCompleted,0.0),
            task_event("test_fail",5400,EventKind::PostTestFailed,0.0),
            task_event("config",6000,EventKind::ConfigurationUpdated,0.0),
            task_event("release",7200,EventKind::AuthorizedRelease,80.0)];
        let mut t=task();
        t.completed_tests.clear(); t.configuration_updated=false; t.authorized_release=false;
        let out=evaluate_service_continuity(&case(events,vec![t],86400)).unwrap();
        assert_eq!(out.services[0].accepted_restoration_seconds,None);
        assert_eq!(out.services[0].rejected_release_attempts,1);
    }

    #[test]
    fn wrong_revision_spare_is_rejected() {
        let events=vec![
            event("failure",1800,EventKind::ComponentFailure,0.0),
            task_event("repair_start",2700,EventKind::RepairStarted,0.0),
            task_event("repair",3600,EventKind::RepairCompleted,0.0),
            task_event("test",5400,EventKind::PostTestPassed,0.0),
            task_event("config",6000,EventKind::ConfigurationUpdated,0.0),
            task_event("release",7200,EventKind::AuthorizedRelease,80.0)];
        let mut t=task();
        t.spare_state=SpareState::WrongRevision;
        t.spare_component_revision_digest=Some(REV_B.into());
        let out=evaluate_service_continuity(&case(events,vec![t],86400)).unwrap();
        assert_eq!(out.services[0].accepted_restoration_seconds,None);
        assert_eq!(out.services[0].rejected_release_attempts,1);
    }

    #[test]
    fn release_is_bound_to_exact_target_revision_not_only_component_name() {
        let mut t=task();
        t.target_component_revision_digest=REV_B.into();
        t.spare_component_revision_digest=Some(REV_B.into());
        let events=vec![
            event_with_revision("failure",1800,EventKind::ComponentFailure,0.0,REV_A),
            task_event_with_revision("repair_start",3000,EventKind::RepairStarted,0.0,REV_B),
            task_event_with_revision("repair",3600,EventKind::RepairCompleted,0.0,REV_B),
            task_event_with_revision("test",5400,EventKind::PostTestPassed,0.0,REV_B),
            task_event_with_revision("config",6000,EventKind::ConfigurationUpdated,0.0,REV_B),
            task_event_with_revision("release",7200,EventKind::AuthorizedRelease,80.0,REV_B)];
        let out=evaluate_service_continuity(&case(events,vec![t],86400)).unwrap();
        assert_eq!(out.services[0].accepted_restoration_seconds,Some(7200));
    }

    #[test]
    fn mismatched_spare_digest_is_rejected_even_when_spare_state_claims_compatible() {
        let events=vec![
            event("failure",1800,EventKind::ComponentFailure,0.0),
            task_event("repair",3600,EventKind::RepairCompleted,0.0),
            task_event("test",5400,EventKind::PostTestPassed,0.0),
            task_event("config",6000,EventKind::ConfigurationUpdated,0.0),
            task_event("release",7200,EventKind::AuthorizedRelease,80.0)];
        let mut t=task();
        t.spare_component_revision_digest=Some(REV_B.into());
        let out=evaluate_service_continuity(&case(events,vec![t],86400)).unwrap();
        assert_eq!(out.services[0].accepted_restoration_seconds,None);
    }

    #[test]
    fn later_safety_gate_does_not_retroactively_veto_earlier_accepted_release() {
        let mut events = vec![
            event("failure", 10, EventKind::ComponentFailure, 0.0),
            task_event("repair_start", 15, EventKind::RepairStarted, 0.0),
            task_event("repair", 20, EventKind::RepairCompleted, 0.0),
            task_event("test", 30, EventKind::PostTestPassed, 0.0),
            task_event("config", 40, EventKind::ConfigurationUpdated, 0.0),
            task_event("release", 50, EventKind::AuthorizedRelease, 80.0),
            event("later_safety_gate", 80, EventKind::SafetyGateFailed, 80.0),
        ];
        let out = evaluate_service_continuity(&case(events.clone(), vec![task()], 100)).unwrap();

        assert_eq!(out.disposition, ContinuityDisposition::UnsafeNotAuthorized);
        assert!(out.safety_gate_failed);
        assert_eq!(out.services[0].accepted_restoration_seconds, Some(50));
        assert_eq!(out.services[0].tasr_seconds, Some(40));
        assert_eq!(out.services[0].rejected_release_attempts, 0);

        // The same gate, moved before release, must reject that release.
        events.retain(|event| event.event_id != "later_safety_gate");
        events.push(event("prior_safety_gate", 45, EventKind::SafetyGateFailed, 0.0));
        let prior = evaluate_service_continuity(&case(events, vec![task()], 100)).unwrap();
        assert_eq!(prior.disposition, ContinuityDisposition::UnsafeNotAuthorized);
        assert_eq!(prior.services[0].accepted_restoration_seconds, None);
        assert_eq!(prior.services[0].rejected_release_attempts, 1);
    }

    #[test]
    fn case_wide_safety_gate_applies_across_services_at_each_release_time() {
        let mut events = vec![
            event("water_failure", 10, EventKind::ComponentFailure, 0.0),
            task_event_for(TASK_ID, "water_failure", "water_start", 20,
                EventKind::RepairStarted, 0.0, REV_A),
            task_event_for(TASK_ID, "water_failure", "water_repair", 30,
                EventKind::RepairCompleted, 0.0, REV_A),
            task_event_for(TASK_ID, "water_failure", "water_test", 40,
                EventKind::PostTestPassed, 0.0, REV_A),
            task_event_for(TASK_ID, "water_failure", "water_config", 45,
                EventKind::ConfigurationUpdated, 0.0, REV_A),
            task_event_for(TASK_ID, "water_failure", "water_release", 50,
                EventKind::AuthorizedRelease, 80.0, REV_A),
            event("global_safety_gate", 60, EventKind::SafetyGateFailed, 0.0),
        ];
        let mut electrical_fault = event(
            "electrical_fault", 12, EventKind::ComponentFailure, 0.0,
        );
        electrical_fault.service_id = "electrical_distribution".into();
        electrical_fault.component_id = "dc_controller".into();
        events.push(electrical_fault);

        for (id, tick, kind, rate) in [
            ("electrical_start", 25, EventKind::RepairStarted, 0.0),
            ("electrical_repair", 35, EventKind::RepairCompleted, 0.0),
            ("electrical_test", 45, EventKind::PostTestPassed, 0.0),
            ("electrical_config", 55, EventKind::ConfigurationUpdated, 0.0),
            ("electrical_release", 70, EventKind::AuthorizedRelease, 5.0),
        ] {
            let mut event = task_event_for(
                "electrical_task",
                "electrical_fault",
                id,
                tick,
                kind,
                rate,
                REV_A,
            );
            event.service_id = "electrical_distribution".into();
            event.component_id = "dc_controller".into();
            events.push(event);
        }

        let mut electrical_task = task_for("electrical_task", "electrical_fault");
        electrical_task.service_id = "electrical_distribution".into();
        electrical_task.component_id = "dc_controller".into();
        let mut input = case(
            events,
            vec![task_for(TASK_ID, "water_failure"), electrical_task],
            100,
        );
        input.services.push(ServiceContract {
            service_id: "electrical_distribution".into(),
            unit: "kW".into(),
            minimum_rate: 2.0,
            nominal_rate: 5.0,
            initial_rate: 5.0,
        });

        let out = evaluate_service_continuity(&input).unwrap();
        assert_eq!(out.disposition, ContinuityDisposition::UnsafeNotAuthorized);
        // Water release predates the global gate and remains historically accepted.
        assert_eq!(out.services[0].accepted_restoration_seconds, Some(50));
        assert_eq!(out.services[0].restoration_episodes[0].accepted_restoration_seconds, Some(50));
        // Electrical release follows the global gate, despite its log living in another service.
        assert_eq!(out.services[1].accepted_restoration_seconds, None);
        assert_eq!(out.services[1].restoration_episodes[0].disposition, RestorationEpisodeDisposition::Open);
    }

    #[test]
    fn repair_and_post_test_events_do_not_restore_delivery_before_release() {
        let events = vec![
            event("failure", 10, EventKind::ComponentFailure, 0.0),
            task_event("repair", 20, EventKind::RepairCompleted, 80.0),
            task_event("test", 30, EventKind::PostTestPassed, 80.0),
            task_event("config", 40, EventKind::ConfigurationUpdated, 80.0),
        ];
        let out = evaluate_service_continuity(&case(events, vec![task()], 60)).unwrap();
        assert_eq!(out.services[0].accepted_restoration_seconds, None);
        assert_eq!(out.services[0].last_floor_recovery_seconds, None);
        assert_eq!(out.services[0].time_below_floor_seconds, 50);
        assert_eq!(out.services[0].floor_breach_episodes, 1);
    }

    #[test]
    fn safety_gate_is_not_misreported_as_a_flow_breach() {
        let out=evaluate_service_continuity(&case(
            vec![event("safety",0,EventKind::SafetyGateFailed,0.0)],vec![],86400)).unwrap();
        assert_eq!(out.disposition,ContinuityDisposition::UnsafeNotAuthorized);
        assert_eq!(out.service_floor_breach_count,0);
    }

    #[test]
    fn permutation_does_not_change_results() {
        let mut events=vec![
            event("failure",3600,EventKind::ComponentFailure,45.0),
            task_event("repair_start",7200,EventKind::RepairStarted,45.0),
            task_event("repair",14400,EventKind::RepairCompleted,45.0),
            task_event("test",18000,EventKind::PostTestPassed,45.0),
            task_event("config",18600,EventKind::ConfigurationUpdated,45.0),
            task_event("release",19200,EventKind::AuthorizedRelease,80.0)];
        let a=evaluate_service_continuity(&case(events.clone(),vec![task()],86400)).unwrap();
        events.reverse();
        let b=evaluate_service_continuity(&case(events,vec![task()],86400)).unwrap();
        assert_eq!(a,b);
    }

    #[test]
    fn conflicting_same_time_rates_and_orphans_fail_closed() {
        let err=evaluate_service_continuity(&case(vec![
            event("a",3600,EventKind::ComponentFailure,0.0),
            event("b",3600,EventKind::BufferSupplyActive,45.0)],vec![],86400)).unwrap_err();
        assert_eq!(err,EvaluationError::ConflictingRatesAtSameInstant {
            service_id:"water".into(),at_seconds:3600});
        let mut orphan=event("orphan",1,EventKind::ComponentFailure,0.0);
        orphan.service_id="missing".into();
        assert_eq!(evaluate_service_continuity(&case(vec![orphan],vec![],100)).unwrap_err(),
            EvaluationError::UnknownEventService("missing".into()));
    }

    #[test]
    fn out_of_window_event_and_release_without_task_are_rejected() {
        let late=event("late",101,EventKind::ComponentFailure,0.0);
        assert_eq!(evaluate_service_continuity(&case(vec![late],vec![],100)).unwrap_err(),
            EvaluationError::EventOutsideTimeline("late".into()));
        let events=vec![
            event("failure", 5, EventKind::ComponentFailure, 0.0),
            task_event("repair",10,EventKind::RepairCompleted,0.0),
            task_event("test",20,EventKind::PostTestPassed,0.0),
            task_event("config",30,EventKind::ConfigurationUpdated,0.0),
            event("release",40,EventKind::AuthorizedRelease,80.0)];
        // Event has no task id, so a claim of release cannot be linked to a work order.
        let mut t=task();
        let out=evaluate_service_continuity(&case(events,vec![t],100)).unwrap();
        assert_eq!(out.services[0].accepted_restoration_seconds,None);
        assert_eq!(out.services[0].rejected_release_attempts,1);
    }

    #[test]
    fn surplus_in_one_service_cannot_compensate_for_another_service_breach() {
        let mut input=case(vec![
            event("power_failure",10,EventKind::CommonDependencyFailure,0.0)
        ],vec![],3610);
        input.services.push(ServiceContract {
            service_id:"electrical_distribution".into(), unit:"kW".into(),
            minimum_rate:2.0, nominal_rate:5.0, initial_rate:5.0,
        });
        input.events[0].service_id="electrical_distribution".into();
        input.events[0].component_id="dc_controller".into();
        let out=evaluate_service_continuity(&input).unwrap();
        assert_eq!(out.disposition,ContinuityDisposition::FloorBreach);
        assert_eq!(out.services[0].time_below_floor_seconds,0);
        assert_eq!(out.services[1].time_below_floor_seconds,3600);
        assert_eq!(out.services[0].last_floor_recovery_seconds,Some(0));
        assert_eq!(out.services[1].last_floor_recovery_seconds,None);
        assert_eq!(out.all_declared_service_floors_restored_seconds,None);
        assert!((out.services[1].shortfall_rate_hours-2.0).abs()<1e-12);
    }

    #[test]
    fn repeated_faults_have_separate_restoration_episodes_and_tasr() {
        let events = vec![
            event("fault_a", 10, EventKind::ComponentFailure, 0.0),
            task_event_for("task_a", "fault_a", "start_a", 20, EventKind::RepairStarted, 0.0, REV_A),
            task_event_for("task_a", "fault_a", "repair_a", 30, EventKind::RepairCompleted, 0.0, REV_A),
            task_event_for("task_a", "fault_a", "test_a", 40, EventKind::PostTestPassed, 0.0, REV_A),
            task_event_for("task_a", "fault_a", "config_a", 45, EventKind::ConfigurationUpdated, 0.0, REV_A),
            task_event_for("task_a", "fault_a", "release_a", 50, EventKind::AuthorizedRelease, 80.0, REV_A),
            event("fault_b", 60, EventKind::ComponentFailure, 0.0),
            task_event_for("task_b", "fault_b", "start_b", 65, EventKind::RepairStarted, 0.0, REV_A),
            task_event_for("task_b", "fault_b", "repair_b", 70, EventKind::RepairCompleted, 0.0, REV_A),
            task_event_for("task_b", "fault_b", "test_b", 75, EventKind::PostTestPassed, 0.0, REV_A),
            task_event_for("task_b", "fault_b", "config_b", 78, EventKind::ConfigurationUpdated, 0.0, REV_A),
            task_event_for("task_b", "fault_b", "release_b", 80, EventKind::AuthorizedRelease, 80.0, REV_A),
        ];
        let tasks = vec![task_for("task_a", "fault_a"), task_for("task_b", "fault_b")];
        let out = evaluate_service_continuity(&case(events, tasks, 100)).unwrap();
        let episodes = &out.services[0].restoration_episodes;
        assert_eq!(episodes.len(), 2);
        assert_eq!(episodes[0].fault_episode_id, "fault_a");
        assert_eq!(episodes[0].tasr_seconds, Some(40));
        assert_eq!(episodes[0].disposition, RestorationEpisodeDisposition::Restored);
        assert_eq!(episodes[1].fault_episode_id, "fault_b");
        assert_eq!(episodes[1].tasr_seconds, Some(20));
        assert_eq!(episodes[1].disposition, RestorationEpisodeDisposition::Restored);
        assert_eq!(out.services[0].time_below_floor_seconds, 60);
        assert_eq!(out.services[0].accepted_restoration_seconds, Some(50));
    }

    #[test]
    fn fault_without_accepted_release_remains_an_open_episode() {
        let out = evaluate_service_continuity(&case(
            vec![event("unresolved_fault", 10, EventKind::ComponentFailure, 0.0)],
            vec![],
            100,
        )).unwrap();
        assert_eq!(out.services[0].restoration_episodes.len(), 1);
        let episode = &out.services[0].restoration_episodes[0];
        assert_eq!(episode.disposition, RestorationEpisodeDisposition::Open);
        assert_eq!(episode.accepted_restoration_seconds, None);
        assert_eq!(episode.tasr_seconds, None);
    }

    #[test]
    fn compatible_spare_arriving_before_first_repair_start_can_qualify() {
        let events = vec![
            event("failure", 10, EventKind::ComponentFailure, 0.0),
            task_event("spare_arrived", 15, EventKind::SpareArrived, 0.0),
            task_event("repair_start", 20, EventKind::RepairStarted, 0.0),
            task_event("repair", 30, EventKind::RepairCompleted, 0.0),
            task_event("test", 40, EventKind::PostTestPassed, 0.0),
            task_event("config", 45, EventKind::ConfigurationUpdated, 0.0),
            task_event("release", 50, EventKind::AuthorizedRelease, 80.0),
        ];
        let mut maintenance = task();
        maintenance.spare_state = SpareState::CompatibleArrivesLater;
        let out = evaluate_service_continuity(&case(events, vec![maintenance], 60)).unwrap();
        assert_eq!(out.services[0].accepted_restoration_seconds, Some(50));
        assert_eq!(out.services[0].restoration_episodes[0].disposition, RestorationEpisodeDisposition::Restored);
    }

    #[test]
    fn repair_completion_without_repair_start_cannot_restore_service() {
        let events = vec![
            event("failure", 10, EventKind::ComponentFailure, 0.0),
            task_event("repair", 20, EventKind::RepairCompleted, 0.0),
            task_event("test", 30, EventKind::PostTestPassed, 0.0),
            task_event("config", 40, EventKind::ConfigurationUpdated, 0.0),
            task_event("release", 50, EventKind::AuthorizedRelease, 80.0),
        ];
        let out = evaluate_service_continuity(&case(events, vec![task()], 60)).unwrap();
        assert_eq!(out.services[0].accepted_restoration_seconds, None);
        assert_eq!(out.services[0].restoration_episodes[0].disposition, RestorationEpisodeDisposition::Open);
    }

    #[test]
    fn spare_arrival_after_repair_cannot_qualify_release() {
        let events = vec![
            event("failure", 10, EventKind::ComponentFailure, 0.0),
            task_event("repair_start", 20, EventKind::RepairStarted, 0.0),
            task_event("repair", 30, EventKind::RepairCompleted, 0.0),
            task_event("spare_late", 35, EventKind::SpareArrived, 0.0),
            task_event("test", 40, EventKind::PostTestPassed, 0.0),
            task_event("config", 45, EventKind::ConfigurationUpdated, 0.0),
            task_event("release", 50, EventKind::AuthorizedRelease, 80.0),
        ];
        let mut maintenance = task();
        maintenance.spare_state = SpareState::CompatibleArrivesLater;
        let out = evaluate_service_continuity(&case(events, vec![maintenance], 60)).unwrap();
        assert_eq!(out.services[0].accepted_restoration_seconds, None);
        assert_eq!(out.services[0].restoration_episodes[0].disposition, RestorationEpisodeDisposition::Open);
    }

    #[test]
    fn procedure_and_personnel_gates_are_independently_enforced() {
        let events=vec![
            event("failure",10,EventKind::ComponentFailure,0.0),
            task_event("repair",20,EventKind::RepairCompleted,0.0),
            task_event("test",30,EventKind::PostTestPassed,0.0),
            task_event("config",40,EventKind::ConfigurationUpdated,0.0),
            task_event("release",50,EventKind::AuthorizedRelease,80.0)];
        let mut unapproved=task();
        unapproved.procedure_approved=false;
        let out=evaluate_service_continuity(&case(events.clone(),vec![unapproved],100)).unwrap();
        assert_eq!(out.services[0].accepted_restoration_seconds,None);
        let mut unqualified=task();
        unqualified.qualified_personnel_present=false;
        let out=evaluate_service_continuity(&case(events,vec![unqualified],100)).unwrap();
        assert_eq!(out.services[0].accepted_restoration_seconds,None);
    }

    #[test]
    fn all_declared_service_floors_require_every_service_to_recover() {
        let mut input=case(vec![
            event("water_failure",10,EventKind::ComponentFailure,0.0),
            task_event_for(TASK_ID, "water_failure", "water_repair", 30,
                EventKind::RepairCompleted, 0.0, REV_A),
            task_event_for(TASK_ID, "water_failure", "water_test", 40,
                EventKind::PostTestPassed, 0.0, REV_A),
            task_event_for(TASK_ID, "water_failure", "water_config", 50,
                EventKind::ConfigurationUpdated, 0.0, REV_A),
            task_event_for(TASK_ID, "water_failure", "water_release", 60,
                EventKind::AuthorizedRelease, 80.0, REV_A),
        ],vec![task_for(TASK_ID, "water_failure")],100);
        input.services.push(ServiceContract {
            service_id:"electrical_distribution".into(), unit:"kW".into(),
            minimum_rate:2.0, nominal_rate:5.0, initial_rate:5.0,
        });
        let out=evaluate_service_continuity(&input).unwrap();
        assert_eq!(out.all_declared_service_floors_restored_seconds,Some(60));
        assert_eq!(out.services[1].last_floor_recovery_seconds,Some(0));
    }

    #[test]
    fn duplicate_event_ids_are_rejected() {
        let err=evaluate_service_continuity(&case(vec![
            event("same",10,EventKind::ComponentFailure,0.0),
            event("same",20,EventKind::ServiceUnavailable,0.0),
        ],vec![],100)).unwrap_err();
        assert_eq!(err,EvaluationError::DuplicateEventId("same".into()));
    }

    #[test]
    fn malformed_revision_digest_fails_closed() {
        let mut bad=event("bad_digest",10,EventKind::ComponentFailure,0.0);
        bad.component_revision_digest="not-a-digest".into();
        assert_eq!(evaluate_service_continuity(&case(vec![bad],vec![],100)).unwrap_err(),
            EvaluationError::InvalidRevisionDigest("bad_digest".into()));
    }
}
