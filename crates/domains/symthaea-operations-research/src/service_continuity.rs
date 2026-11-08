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
    pub delivered_rate_after: f64, pub evidence_ref: String,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpareState { CompatibleOnboard, CompatibleArrivesLater, WrongRevision, Unknown }
#[derive(Clone, Debug, PartialEq)]
pub struct MaintenanceTask {
    pub task_id: String, pub service_id: String, pub component_id: String,
    pub procedure_approved: bool, pub qualified_personnel_present: bool,
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
}
/// Deficit integral is in that service's unit-hours; never sum across unlike services.
#[derive(Clone, Debug, PartialEq)]
pub struct ServiceEvaluation {
    pub service_id: String, pub unit: String, pub minimum_rate: f64,
    pub minimum_observed_rate: f64, pub time_below_floor_seconds: u64,
    pub floor_breach_episodes: u32, pub first_floor_breach_seconds: Option<u64>,
    pub shortfall_rate_hours: f64, pub first_fault_seconds: Option<u64>,
    pub repair_started_seconds: Option<u64>, pub spare_arrived_seconds: Option<u64>,
    pub repair_completed_seconds: Option<u64>, pub post_test_passed_seconds: Option<u64>,
    pub configuration_updated_seconds: Option<u64>,
    pub accepted_restoration_seconds: Option<u64>, pub tasr_seconds: Option<u64>,
    pub rejected_release_attempts: u32,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ServiceContinuityEvaluation {
    pub case_id: String, pub disposition: ContinuityDisposition,
    pub safety_gate_failed: bool, pub service_floor_breach_count: u32,
    pub first_floor_breach_seconds: Option<u64>,
    pub authorized_restoration_seconds: Option<u64>, pub services: Vec<ServiceEvaluation>,
}

/// Recompute service-floor exposure and accepted restoration from event evidence.
pub fn evaluate_service_continuity(
    case: &ServiceContinuityCase,
) -> Result<ServiceContinuityEvaluation, EvaluationError> {
    validate_case(case)?;
    let safety_failed = case.events.iter().any(|e| e.kind == EventKind::SafetyGateFailed);
    let mut services = Vec::with_capacity(case.services.len());
    for contract in &case.services {
        let mut events: Vec<&EvidenceEvent> = case.events.iter()
            .filter(|e| e.service_id == contract.service_id).collect();
        events.sort_by(|a,b| a.at_seconds.cmp(&b.at_seconds)
            .then_with(|| a.kind.cmp(&b.kind)).then_with(|| a.event_id.cmp(&b.event_id)));
        let tasks: Vec<&MaintenanceTask> = case.maintenance_tasks.iter()
            .filter(|t| t.service_id == contract.service_id).collect();
        services.push(evaluate_service(contract, &events, &tasks, case.timeline_seconds, safety_failed));
    }
    let breaches: u32 = services.iter().map(|s| s.floor_breach_episodes).sum();
    let first_breach = services.iter().filter_map(|s| s.first_floor_breach_seconds).min();
    let restored = services.iter().filter_map(|s| s.accepted_restoration_seconds).min();
    let disposition = if safety_failed { ContinuityDisposition::UnsafeNotAuthorized }
        else if breaches > 0 { ContinuityDisposition::FloorBreach }
        else { ContinuityDisposition::NoFloorBreach };
    Ok(ServiceContinuityEvaluation {
        case_id: case.case_id.clone(), disposition, safety_gate_failed: safety_failed,
        service_floor_breach_count: breaches, first_floor_breach_seconds: first_breach,
        authorized_restoration_seconds: restored, services,
    })
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
    let mut task_ids = BTreeSet::new();
    for t in &case.maintenance_tasks {
        if t.task_id.trim().is_empty() { return Err(EvaluationError::EmptyTaskId); }
        if !task_ids.insert(t.task_id.as_str()) {
            return Err(EvaluationError::DuplicateTaskId(t.task_id.clone()));
        }
        if !service_ids.contains(t.service_id.as_str()) {
            return Err(EvaluationError::UnknownTaskService(t.service_id.clone()));
        }
        if t.component_id.trim().is_empty() { return Err(EvaluationError::EmptyTaskComponent(t.task_id.clone())); }
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
    c: &ServiceContract, events: &[&EvidenceEvent], tasks: &[&MaintenanceTask],
    timeline: u64, safety_failed: bool,
) -> ServiceEvaluation {
    let first_fault = events.iter().filter(|e| matches!(e.kind, EventKind::ComponentFailure
        | EventKind::ServiceUnavailable | EventKind::CommonDependencyFailure))
        .map(|e| e.at_seconds).min();
    let mut rate = c.initial_rate;
    let mut min_rate = rate;
    let mut prev = 0_u64;
    let mut active = false;
    let mut seconds_below = 0_u64;
    let mut episodes = 0_u32;
    let mut first_breach = None;
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
                    if !safety_failed && e.delivered_rate_after >= c.minimum_rate
                        && release_accepted(e, events, tasks) {
                        rate = e.delivered_rate_after; restoration.get_or_insert(t);
                    } else { rejected = rejected.saturating_add(1); }
                }
                _ => rate = e.delivered_rate_after,
            }
        }
        min_rate = min_rate.min(rate);
        if rate >= c.minimum_rate { active = false; }
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
    }
}
fn release_accepted(
    release: &EvidenceEvent, events: &[&EvidenceEvent], tasks: &[&MaintenanceTask],
) -> bool {
    let matches: Vec<&MaintenanceTask> = tasks.iter().copied()
        .filter(|t| t.component_id == release.component_id).collect();
    if matches.is_empty() { return false; }
    let evidence: Vec<&EvidenceEvent> = events.iter().copied()
        .filter(|e| e.component_id == release.component_id && e.at_seconds < release.at_seconds)
        .collect();
    let last_fault = evidence.iter().filter(|e| matches!(e.kind, EventKind::ComponentFailure
        | EventKind::ServiceUnavailable | EventKind::CommonDependencyFailure))
        .map(|e| e.at_seconds).max();
    let last_repair = evidence.iter().filter(|e| e.kind == EventKind::RepairCompleted)
        .map(|e| e.at_seconds).max();
    let (Some(fault), Some(repair)) = (last_fault, last_repair) else { return false; };
    if repair <= fault { return false; }
    let last_failed_test = evidence.iter().filter(|e| e.kind == EventKind::PostTestFailed)
        .map(|e| e.at_seconds).max();
    let pass = evidence.iter().filter(|e| e.kind == EventKind::PostTestPassed)
        .map(|e| e.at_seconds).max();
    let Some(pass) = pass else { return false; };
    if pass <= repair || last_failed_test.is_some_and(|failed| failed >= pass) { return false; }
    if !evidence.iter().any(|e| e.kind == EventKind::ConfigurationUpdated && e.at_seconds > pass) {
        return false;
    }
    let spare_arrived = evidence.iter().any(|e| e.kind == EventKind::SpareArrived);
    matches.iter().all(|t| {
        let tests_ok = t.required_tests.iter().all(|required|
            t.completed_tests.iter().any(|done| done == required));
        t.procedure_approved && t.qualified_personnel_present && tests_ok
            && t.configuration_updated && t.authorized_release
            && match t.spare_state {
                SpareState::CompatibleOnboard => true,
                SpareState::CompatibleArrivesLater => spare_arrived,
                SpareState::WrongRevision | SpareState::Unknown => false,
            }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn service() -> ServiceContract {
        ServiceContract { service_id:"water".into(), unit:"L/h".into(),
            minimum_rate:40.0, nominal_rate:80.0, initial_rate:80.0 }
    }
    fn event(id:&str, t:u64, kind:EventKind, rate:f64) -> EvidenceEvent {
        EvidenceEvent { event_id:id.into(), at_seconds:t, kind, service_id:"water".into(),
            component_id:"maker_A".into(), delivered_rate_after:rate, evidence_ref:format!("fixture:{id}") }
    }
    fn task() -> MaintenanceTask {
        MaintenanceTask { task_id:"replace_maker_A".into(), service_id:"water".into(),
            component_id:"maker_A".into(), procedure_approved:true, qualified_personnel_present:true,
            spare_state:SpareState::CompatibleOnboard, required_tests:vec!["flow_quality".into()],
            completed_tests:vec!["flow_quality".into()], configuration_updated:true, authorized_release:true }
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
            event("repair_start",7200,EventKind::RepairStarted,45.0),
            event("repair_done",14400,EventKind::RepairCompleted,45.0),
            event("test",18000,EventKind::PostTestPassed,45.0),
            event("config",18600,EventKind::ConfigurationUpdated,45.0),
            event("release",19200,EventKind::AuthorizedRelease,80.0)];
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
        let events=vec![
            event("failure",3600,EventKind::ComponentFailure,45.0),
            event("empty",86400,EventKind::ServiceUnavailable,0.0),
            event("spare",108000,EventKind::SpareArrived,0.0)];
        let mut t=task(); t.spare_state=SpareState::CompatibleArrivesLater;
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
    fn failed_post_test_and_wrong_revision_reject_release() {
        let events=vec![
            event("failure",1800,EventKind::ComponentFailure,0.0),
            event("repair",3600,EventKind::RepairCompleted,0.0),
            event("test_fail",5400,EventKind::PostTestFailed,0.0),
            event("config",6000,EventKind::ConfigurationUpdated,0.0),
            event("release",7200,EventKind::AuthorizedRelease,80.0)];
        let mut t=task(); t.completed_tests.clear(); t.configuration_updated=false; t.authorized_release=false;
        let out=evaluate_service_continuity(&case(events.clone(),vec![t],86400)).unwrap();
        assert_eq!(out.services[0].accepted_restoration_seconds,None);
        let mut wrong=task(); wrong.spare_state=SpareState::WrongRevision;
        let mut passing=events; passing[2]=event("test_pass",5400,EventKind::PostTestPassed,0.0);
        let out=evaluate_service_continuity(&case(passing,vec![wrong],86400)).unwrap();
        assert_eq!(out.services[0].accepted_restoration_seconds,None);
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
            event("repair",14400,EventKind::RepairCompleted,45.0),
            event("test",18000,EventKind::PostTestPassed,45.0),
            event("config",18600,EventKind::ConfigurationUpdated,45.0),
            event("release",19200,EventKind::AuthorizedRelease,80.0)];
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
            event("repair",10,EventKind::RepairCompleted,0.0),
            event("test",20,EventKind::PostTestPassed,0.0),
            event("config",30,EventKind::ConfigurationUpdated,0.0),
            event("release",40,EventKind::AuthorizedRelease,80.0)];
        let out=evaluate_service_continuity(&case(events,vec![],100)).unwrap();
        assert_eq!(out.services[0].accepted_restoration_seconds,None);
        assert_eq!(out.services[0].rejected_release_attempts,1);
    }
}
