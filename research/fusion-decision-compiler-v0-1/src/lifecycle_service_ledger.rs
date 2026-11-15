//! Event-based availability and signed lifecycle net-energy ledger.
//!
//! V0.1 is a deterministic accounting contract. It does not estimate failure
//! probabilities or infer maintenance durations. It verifies a complete
//! time-partition, exact event incidence and signed net-energy integration.

use std::collections::{BTreeMap, BTreeSet};

pub const LIFECYCLE_SERVICE_VERSION: &str = "fusion-lifecycle-service/0.1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum OutageCategory {
    Planned,
    Unplanned,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum ServiceImpact {
    FullOutage,
    Derated,
    NoServiceLoss,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum LifecycleEvidenceClass {
    Synthetic,
    Assumed,
    Modelled,
    MockupValidated,
    Measured,
    IndependentlyVerified,
    PhysicallyQualified,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum PowerOperatingState {
    PowerGeneration,
    ReducedPower,
    HotStandby,
    PlannedMaintenance,
    ForcedOutage,
    ConcurrentOutage,
    Restart,
}

impl PowerOperatingState {
    fn is_outage(self) -> bool {
        matches!(
            self,
            Self::PlannedMaintenance | Self::ForcedOutage | Self::ConcurrentOutage
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AvailabilityEvent {
    pub event_id: String,
    pub category: OutageCategory,
    pub impact: ServiceImpact,
    pub subsystem_id: String,
    pub failure_mode_id: String,
    pub start_second: u64,
    pub end_second: u64,
    pub evidence_ref: String,
    pub evidence_class: LifecycleEvidenceClass,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PowerStateSegment {
    pub segment_id: String,
    pub start_second: u64,
    pub end_second: u64,
    /// Signed net electricity at the declared grid boundary: export positive,
    /// import negative. It must include the state-specific plant loads.
    pub net_power_mwe: f64,
    pub state: PowerOperatingState,
    /// Exact active availability-event IDs for the entire half-open segment.
    pub active_event_ids: Vec<String>,
    /// Reference to a qualified case-level power-balance output.
    pub power_balance_report_ref: String,
    pub output_sha256: String,
    pub evidence_ref: String,
    pub evidence_class: LifecycleEvidenceClass,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LifecycleServicePlan {
    pub campaign_id: String,
    pub horizon_ref: String,
    pub horizon_start_second: u64,
    pub horizon_end_second: u64,
    pub service_boundary_ref: String,
    pub unit_ref: String,
    pub sign_convention_ref: String,
    /// Events may be supplied in any order; the report canonicalizes by ID.
    pub events: Vec<AvailabilityEvent>,
    /// Segments must be in chronological order and partition the full horizon.
    pub power_segments: Vec<PowerStateSegment>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LifecycleServiceDisposition {
    NotEvaluable,
    ValidDeterministicLedger,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LifecycleDiagnostic {
    EmptyCampaignIdentity,
    InvalidHorizon,
    EmptyServiceBoundary,
    UnitMismatch,
    SignConventionMismatch,
    NoPowerSegments,
    DuplicateEventId,
    EmptyEventIdentity,
    EmptyEventEvidence,
    EventOutsideHorizon,
    NonPositiveEventDuration,
    DuplicateSegmentId,
    EmptySegmentIdentity,
    EmptySegmentEvidence,
    InvalidSegmentOutputDigest,
    MissingPowerBalanceReceipt,
    NonFiniteNetPower,
    SegmentOutsideHorizon,
    NonPositiveSegmentDuration,
    ScheduleDoesNotStartAtHorizon,
    ScheduleDoesNotEndAtHorizon,
    ScheduleGap,
    ScheduleOverlap,
    MissingEventBoundary,
    DuplicateActiveEventId,
    ActiveEventSetMismatch,
    OutageStateMismatch,
    OutageStateWithoutFullOutageEvent,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StateEnergy {
    pub state: PowerOperatingState,
    pub signed_net_energy_mwh: f64,
    pub export_energy_mwh: f64,
    pub import_energy_mwh: f64,
    pub duration_seconds: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventEvidenceReceipt {
    pub event_id: String,
    pub category: OutageCategory,
    pub impact: ServiceImpact,
    pub subsystem_id: String,
    pub failure_mode_id: String,
    pub start_second: u64,
    pub end_second: u64,
    pub evidence_ref: String,
    pub evidence_class: LifecycleEvidenceClass,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentEvidenceReceipt {
    pub segment_id: String,
    pub start_second: u64,
    pub end_second: u64,
    pub active_event_ids: Vec<String>,
    pub power_balance_report_ref: String,
    pub output_sha256: String,
    pub evidence_ref: String,
    pub evidence_class: LifecycleEvidenceClass,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LifecycleServiceReport {
    pub contract_version: &'static str,
    pub campaign_id: String,
    pub disposition: LifecycleServiceDisposition,
    pub diagnostics: Vec<LifecycleDiagnostic>,
    pub horizon_seconds: Option<u64>,
    pub available_seconds: Option<u64>,
    pub planned_only_unavailable_seconds: Option<u64>,
    pub unplanned_only_unavailable_seconds: Option<u64>,
    pub concurrent_unavailable_seconds: Option<u64>,
    pub total_unavailable_seconds: Option<u64>,
    pub availability_fraction: Option<f64>,
    pub signed_net_energy_mwh: Option<f64>,
    pub export_energy_mwh: Option<f64>,
    /// Positive quantity consumed from the grid during outages/standby.
    pub import_energy_mwh: Option<f64>,
    pub energy_by_state: Vec<StateEnergy>,
    pub event_receipts: Vec<EventEvidenceReceipt>,
    pub segment_receipts: Vec<SegmentEvidenceReceipt>,
    pub claim_ceiling: &'static str,
}

/// Evaluate a deterministic event-based availability and net-energy ledger.
///
/// Downtime is classified as planned-only, unplanned-only or concurrent.
/// Concurrent time is counted exactly once in total unavailability. Net
/// energy is integrated across all states, including negative imports during
/// hot standby, planned maintenance and forced outages.
pub fn evaluate_lifecycle_service_budget(plan: &LifecycleServicePlan) -> LifecycleServiceReport {
    let mut diagnostics = Vec::new();
    let mut event_by_id: BTreeMap<&str, &AvailabilityEvent> = BTreeMap::new();
    let mut segment_ids = BTreeSet::new();

    if plan.campaign_id.trim().is_empty() || plan.horizon_ref.trim().is_empty() {
        diagnostics.push(LifecycleDiagnostic::EmptyCampaignIdentity);
    }
    if plan.horizon_end_second <= plan.horizon_start_second {
        diagnostics.push(LifecycleDiagnostic::InvalidHorizon);
    }
    if plan.service_boundary_ref.trim().is_empty() {
        diagnostics.push(LifecycleDiagnostic::EmptyServiceBoundary);
    }
    if plan.unit_ref != "MWe" {
        diagnostics.push(LifecycleDiagnostic::UnitMismatch);
    }
    if plan.sign_convention_ref != "net-export-positive-import-negative-v1" {
        diagnostics.push(LifecycleDiagnostic::SignConventionMismatch);
    }
    if plan.power_segments.is_empty() {
        diagnostics.push(LifecycleDiagnostic::NoPowerSegments);
    }

    for event in &plan.events {
        if event.event_id.trim().is_empty()
            || event.subsystem_id.trim().is_empty()
            || event.failure_mode_id.trim().is_empty()
        {
            diagnostics.push(LifecycleDiagnostic::EmptyEventIdentity);
        }
        if event.evidence_ref.trim().is_empty() {
            diagnostics.push(LifecycleDiagnostic::EmptyEventEvidence);
        }
        if event.end_second <= event.start_second {
            diagnostics.push(LifecycleDiagnostic::NonPositiveEventDuration);
        }
        if event.start_second < plan.horizon_start_second
            || event.end_second > plan.horizon_end_second
        {
            diagnostics.push(LifecycleDiagnostic::EventOutsideHorizon);
        }
        if event_by_id.insert(event.event_id.as_str(), event).is_some() {
            diagnostics.push(LifecycleDiagnostic::DuplicateEventId);
        }
    }

    let mut previous_end = None;
    for segment in &plan.power_segments {
        if segment.segment_id.trim().is_empty() {
            diagnostics.push(LifecycleDiagnostic::EmptySegmentIdentity);
        }
        if !segment_ids.insert(segment.segment_id.as_str()) {
            diagnostics.push(LifecycleDiagnostic::DuplicateSegmentId);
        }
        if segment.power_balance_report_ref.trim().is_empty() {
            diagnostics.push(LifecycleDiagnostic::MissingPowerBalanceReceipt);
        }
        if segment.evidence_ref.trim().is_empty() {
            diagnostics.push(LifecycleDiagnostic::EmptySegmentEvidence);
        }
        if !is_sha256(&segment.output_sha256) {
            diagnostics.push(LifecycleDiagnostic::InvalidSegmentOutputDigest);
        }
        if !segment.net_power_mwe.is_finite() {
            diagnostics.push(LifecycleDiagnostic::NonFiniteNetPower);
        }
        if segment.end_second <= segment.start_second {
            diagnostics.push(LifecycleDiagnostic::NonPositiveSegmentDuration);
        }
        if segment.start_second < plan.horizon_start_second
            || segment.end_second > plan.horizon_end_second
        {
            diagnostics.push(LifecycleDiagnostic::SegmentOutsideHorizon);
        }

        if let Some(end) = previous_end {
            if segment.start_second > end {
                diagnostics.push(LifecycleDiagnostic::ScheduleGap);
            } else if segment.start_second < end {
                diagnostics.push(LifecycleDiagnostic::ScheduleOverlap);
            }
        } else if segment.start_second != plan.horizon_start_second {
            diagnostics.push(LifecycleDiagnostic::ScheduleDoesNotStartAtHorizon);
        }
        previous_end = Some(segment.end_second);
    }
    if plan
        .power_segments
        .last()
        .is_some_and(|segment| segment.end_second != plan.horizon_end_second)
    {
        diagnostics.push(LifecycleDiagnostic::ScheduleDoesNotEndAtHorizon);
    }

    let mut segment_boundaries = BTreeSet::new();
    for segment in &plan.power_segments {
        segment_boundaries.insert(segment.start_second);
        segment_boundaries.insert(segment.end_second);
    }
    for event in &plan.events {
        if !segment_boundaries.contains(&event.start_second)
            || !segment_boundaries.contains(&event.end_second)
        {
            diagnostics.push(LifecycleDiagnostic::MissingEventBoundary);
        }
    }

    for segment in &plan.power_segments {
        let mut actual_ids = segment.active_event_ids.clone();
        actual_ids.sort();
        let before = actual_ids.len();
        actual_ids.dedup();
        if actual_ids.len() != before {
            diagnostics.push(LifecycleDiagnostic::DuplicateActiveEventId);
        }

        let mut expected_ids: Vec<String> = plan
            .events
            .iter()
            .filter(|event| {
                event.start_second <= segment.start_second && event.end_second >= segment.end_second
            })
            .map(|event| event.event_id.clone())
            .collect();
        expected_ids.sort();
        if actual_ids != expected_ids {
            diagnostics.push(LifecycleDiagnostic::ActiveEventSetMismatch);
        }

        let active_full_outages: Vec<&AvailabilityEvent> = plan
            .events
            .iter()
            .filter(|event| {
                event.impact == ServiceImpact::FullOutage
                    && event.start_second <= segment.start_second
                    && event.end_second >= segment.end_second
            })
            .collect();
        let planned = active_full_outages
            .iter()
            .any(|event| event.category == OutageCategory::Planned);
        let unplanned = active_full_outages
            .iter()
            .any(|event| event.category == OutageCategory::Unplanned);
        let expected_state = match (planned, unplanned) {
            (true, true) => Some(PowerOperatingState::ConcurrentOutage),
            (true, false) => Some(PowerOperatingState::PlannedMaintenance),
            (false, true) => Some(PowerOperatingState::ForcedOutage),
            (false, false) => None,
        };
        match expected_state {
            Some(state) if segment.state != state => {
                diagnostics.push(LifecycleDiagnostic::OutageStateMismatch);
            }
            None if segment.state.is_outage() => {
                diagnostics.push(LifecycleDiagnostic::OutageStateWithoutFullOutageEvent);
            }
            _ => {}
        }
    }

    let event_receipts = canonical_event_receipts(plan);
    let segment_receipts = canonical_segment_receipts(plan);
    if !diagnostics.is_empty() {
        return invalid_report(plan, diagnostics, event_receipts, segment_receipts);
    }

    let horizon = plan.horizon_end_second - plan.horizon_start_second;
    let mut available = 0_u64;
    let mut planned_only = 0_u64;
    let mut unplanned_only = 0_u64;
    let mut concurrent = 0_u64;

    let mut energy_by_state: BTreeMap<PowerOperatingState, StateEnergy> = BTreeMap::new();
    let mut signed_energy = 0.0_f64;
    let mut export_energy = 0.0_f64;
    let mut import_energy = 0.0_f64;

    for segment in &plan.power_segments {
        let duration = segment.end_second - segment.start_second;
        let active_full: Vec<&AvailabilityEvent> = plan
            .events
            .iter()
            .filter(|event| {
                event.impact == ServiceImpact::FullOutage
                    && event.start_second <= segment.start_second
                    && event.end_second >= segment.end_second
            })
            .collect();
        let planned = active_full
            .iter()
            .any(|event| event.category == OutageCategory::Planned);
        let unplanned = active_full
            .iter()
            .any(|event| event.category == OutageCategory::Unplanned);
        match (planned, unplanned) {
            (true, true) => concurrent += duration,
            (true, false) => planned_only += duration,
            (false, true) => unplanned_only += duration,
            (false, false) => available += duration,
        }

        let hours = duration as f64 / 3600.0;
        let energy = segment.net_power_mwe * hours;
        signed_energy += energy;
        if energy >= 0.0 {
            export_energy += energy;
        } else {
            import_energy += -energy;
        }

        let entry = energy_by_state.entry(segment.state).or_insert(StateEnergy {
            state: segment.state,
            signed_net_energy_mwh: 0.0,
            export_energy_mwh: 0.0,
            import_energy_mwh: 0.0,
            duration_seconds: 0,
        });
        entry.signed_net_energy_mwh += energy;
        entry.duration_seconds += duration;
        if energy >= 0.0 {
            entry.export_energy_mwh += energy;
        } else {
            entry.import_energy_mwh += -energy;
        }
    }

    let unavailable = planned_only + unplanned_only + concurrent;
    let availability_fraction = available as f64 / horizon as f64;
    let energy_by_state = energy_by_state.into_values().collect();

    LifecycleServiceReport {
        contract_version: LIFECYCLE_SERVICE_VERSION,
        campaign_id: plan.campaign_id.clone(),
        disposition: LifecycleServiceDisposition::ValidDeterministicLedger,
        diagnostics,
        horizon_seconds: Some(horizon),
        available_seconds: Some(available),
        planned_only_unavailable_seconds: Some(planned_only),
        unplanned_only_unavailable_seconds: Some(unplanned_only),
        concurrent_unavailable_seconds: Some(concurrent),
        total_unavailable_seconds: Some(unavailable),
        availability_fraction: Some(availability_fraction),
        signed_net_energy_mwh: Some(signed_energy),
        export_energy_mwh: Some(export_energy),
        import_energy_mwh: Some(import_energy),
        energy_by_state,
        event_receipts,
        segment_receipts,
        claim_ceiling: "deterministic time/energy ledger under supplied events and state powers only",
    }
}

fn canonical_event_receipts(plan: &LifecycleServicePlan) -> Vec<EventEvidenceReceipt> {
    let mut events = plan.events.clone();
    events.sort_by(|left, right| left.event_id.cmp(&right.event_id));
    events
        .into_iter()
        .map(|event| EventEvidenceReceipt {
            event_id: event.event_id,
            category: event.category,
            impact: event.impact,
            subsystem_id: event.subsystem_id,
            failure_mode_id: event.failure_mode_id,
            start_second: event.start_second,
            end_second: event.end_second,
            evidence_ref: event.evidence_ref,
            evidence_class: event.evidence_class,
        })
        .collect()
}

fn canonical_segment_receipts(plan: &LifecycleServicePlan) -> Vec<SegmentEvidenceReceipt> {
    let mut segments = plan.power_segments.clone();
    segments.sort_by(|left, right| {
        (left.start_second, left.end_second, left.segment_id.as_str()).cmp(&(
            right.start_second,
            right.end_second,
            right.segment_id.as_str(),
        ))
    });
    segments
        .into_iter()
        .map(|segment| {
            let mut active_event_ids = segment.active_event_ids;
            active_event_ids.sort();
            SegmentEvidenceReceipt {
                segment_id: segment.segment_id,
                start_second: segment.start_second,
                end_second: segment.end_second,
                active_event_ids,
                power_balance_report_ref: segment.power_balance_report_ref,
                output_sha256: segment.output_sha256,
                evidence_ref: segment.evidence_ref,
                evidence_class: segment.evidence_class,
            }
        })
        .collect()
}

fn invalid_report(
    plan: &LifecycleServicePlan,
    diagnostics: Vec<LifecycleDiagnostic>,
    event_receipts: Vec<EventEvidenceReceipt>,
    segment_receipts: Vec<SegmentEvidenceReceipt>,
) -> LifecycleServiceReport {
    LifecycleServiceReport {
        contract_version: LIFECYCLE_SERVICE_VERSION,
        campaign_id: plan.campaign_id.clone(),
        disposition: LifecycleServiceDisposition::NotEvaluable,
        diagnostics,
        horizon_seconds: None,
        available_seconds: None,
        planned_only_unavailable_seconds: None,
        unplanned_only_unavailable_seconds: None,
        concurrent_unavailable_seconds: None,
        total_unavailable_seconds: None,
        availability_fraction: None,
        signed_net_energy_mwh: None,
        export_energy_mwh: None,
        import_energy_mwh: None,
        energy_by_state: Vec::new(),
        event_receipts,
        segment_receipts,
        claim_ceiling: "deterministic time/energy ledger under supplied events and state powers only",
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn event(
        id: &str,
        category: OutageCategory,
        impact: ServiceImpact,
        start: u64,
        end: u64,
    ) -> AvailabilityEvent {
        AvailabilityEvent {
            event_id: id.to_string(),
            category,
            impact,
            subsystem_id: format!("subsystem-{id}"),
            failure_mode_id: format!("failure-{id}"),
            start_second: start,
            end_second: end,
            evidence_ref: format!("event-evidence-{id}"),
            evidence_class: LifecycleEvidenceClass::Modelled,
        }
    }

    fn segment(
        id: &str,
        start: u64,
        end: u64,
        power: f64,
        state: PowerOperatingState,
        active: &[&str],
    ) -> PowerStateSegment {
        PowerStateSegment {
            segment_id: id.to_string(),
            start_second: start,
            end_second: end,
            net_power_mwe: power,
            state,
            active_event_ids: active.iter().map(|value| (*value).to_string()).collect(),
            power_balance_report_ref: format!("power-balance-{id}"),
            output_sha256: sha('a'),
            evidence_ref: format!("segment-evidence-{id}"),
            evidence_class: LifecycleEvidenceClass::Modelled,
        }
    }

    fn base_plan(
        events: Vec<AvailabilityEvent>,
        power_segments: Vec<PowerStateSegment>,
    ) -> LifecycleServicePlan {
        LifecycleServicePlan {
            campaign_id: "synthetic-lifecycle-case-v1".to_string(),
            horizon_ref: "synthetic-1000-second-horizon".to_string(),
            horizon_start_second: 0,
            horizon_end_second: 1000,
            service_boundary_ref: "plant-terminal-net-electric-v1".to_string(),
            unit_ref: "MWe".to_string(),
            sign_convention_ref: "net-export-positive-import-negative-v1".to_string(),
            events,
            power_segments,
        }
    }

    fn overlapping_planned_plan() -> LifecycleServicePlan {
        base_plan(
            vec![
                event(
                    "planned-a",
                    OutageCategory::Planned,
                    ServiceImpact::FullOutage,
                    100,
                    400,
                ),
                event(
                    "planned-b",
                    OutageCategory::Planned,
                    ServiceImpact::FullOutage,
                    300,
                    600,
                ),
            ],
            vec![
                segment(
                    "s0",
                    0,
                    100,
                    100.0,
                    PowerOperatingState::PowerGeneration,
                    &[],
                ),
                segment(
                    "s1",
                    100,
                    300,
                    -5.0,
                    PowerOperatingState::PlannedMaintenance,
                    &["planned-a"],
                ),
                segment(
                    "s2",
                    300,
                    400,
                    -8.0,
                    PowerOperatingState::PlannedMaintenance,
                    &["planned-a", "planned-b"],
                ),
                segment(
                    "s3",
                    400,
                    600,
                    -5.0,
                    PowerOperatingState::PlannedMaintenance,
                    &["planned-b"],
                ),
                segment(
                    "s4",
                    600,
                    1000,
                    100.0,
                    PowerOperatingState::PowerGeneration,
                    &[],
                ),
            ],
        )
    }

    fn concurrent_planned_forced_plan() -> LifecycleServicePlan {
        base_plan(
            vec![
                event(
                    "planned-a",
                    OutageCategory::Planned,
                    ServiceImpact::FullOutage,
                    100,
                    500,
                ),
                event(
                    "forced-b",
                    OutageCategory::Unplanned,
                    ServiceImpact::FullOutage,
                    300,
                    700,
                ),
            ],
            vec![
                segment(
                    "s0",
                    0,
                    100,
                    100.0,
                    PowerOperatingState::PowerGeneration,
                    &[],
                ),
                segment(
                    "s1",
                    100,
                    300,
                    -5.0,
                    PowerOperatingState::PlannedMaintenance,
                    &["planned-a"],
                ),
                segment(
                    "s2",
                    300,
                    500,
                    -8.0,
                    PowerOperatingState::ConcurrentOutage,
                    &["forced-b", "planned-a"],
                ),
                segment(
                    "s3",
                    500,
                    700,
                    -7.0,
                    PowerOperatingState::ForcedOutage,
                    &["forced-b"],
                ),
                segment(
                    "s4",
                    700,
                    1000,
                    100.0,
                    PowerOperatingState::PowerGeneration,
                    &[],
                ),
            ],
        )
    }

    #[test]
    fn overlapping_planned_outages_are_counted_once() {
        let report = evaluate_lifecycle_service_budget(&overlapping_planned_plan());
        assert_eq!(
            report.disposition,
            LifecycleServiceDisposition::ValidDeterministicLedger
        );
        assert_eq!(report.available_seconds, Some(500));
        assert_eq!(report.planned_only_unavailable_seconds, Some(500));
        assert_eq!(report.unplanned_only_unavailable_seconds, Some(0));
        assert_eq!(report.concurrent_unavailable_seconds, Some(0));
        assert_eq!(report.total_unavailable_seconds, Some(500));
        assert_eq!(report.availability_fraction, Some(0.5));
    }

    #[test]
    fn planned_and_forced_overlap_is_separated_and_not_double_counted() {
        let report = evaluate_lifecycle_service_budget(&concurrent_planned_forced_plan());
        assert_eq!(report.planned_only_unavailable_seconds, Some(200));
        assert_eq!(report.unplanned_only_unavailable_seconds, Some(200));
        assert_eq!(report.concurrent_unavailable_seconds, Some(200));
        assert_eq!(report.total_unavailable_seconds, Some(600));
        assert_eq!(report.available_seconds, Some(400));
    }

    #[test]
    fn schedule_gap_is_not_evaluable() {
        let mut plan = overlapping_planned_plan();
        plan.power_segments[1].start_second = 101;
        let report = evaluate_lifecycle_service_budget(&plan);
        assert_eq!(
            report.disposition,
            LifecycleServiceDisposition::NotEvaluable
        );
        assert!(
            report
                .diagnostics
                .contains(&LifecycleDiagnostic::ScheduleGap)
        );
    }

    #[test]
    fn schedule_overlap_is_not_evaluable() {
        let mut plan = overlapping_planned_plan();
        plan.power_segments[1].start_second = 99;
        let report = evaluate_lifecycle_service_budget(&plan);
        assert_eq!(
            report.disposition,
            LifecycleServiceDisposition::NotEvaluable
        );
        assert!(
            report
                .diagnostics
                .contains(&LifecycleDiagnostic::ScheduleOverlap)
        );
    }

    #[test]
    fn event_boundary_must_split_the_power_schedule() {
        let mut plan = overlapping_planned_plan();
        plan.power_segments.remove(2);
        let report = evaluate_lifecycle_service_budget(&plan);
        assert_eq!(
            report.disposition,
            LifecycleServiceDisposition::NotEvaluable
        );
        assert!(
            report
                .diagnostics
                .contains(&LifecycleDiagnostic::MissingEventBoundary)
        );
    }

    #[test]
    fn declared_active_event_ids_must_match_interval_derived_ids() {
        let mut plan = overlapping_planned_plan();
        plan.power_segments[2].active_event_ids = vec!["planned-a".to_string()];
        let report = evaluate_lifecycle_service_budget(&plan);
        assert_eq!(
            report.disposition,
            LifecycleServiceDisposition::NotEvaluable
        );
        assert!(
            report
                .diagnostics
                .contains(&LifecycleDiagnostic::ActiveEventSetMismatch)
        );
    }

    #[test]
    fn event_outside_horizon_is_refused() {
        let mut plan = overlapping_planned_plan();
        plan.events[0].start_second = 999;
        plan.events[0].end_second = 1001;
        let report = evaluate_lifecycle_service_budget(&plan);
        assert_eq!(
            report.disposition,
            LifecycleServiceDisposition::NotEvaluable
        );
        assert!(
            report
                .diagnostics
                .contains(&LifecycleDiagnostic::EventOutsideHorizon)
        );
    }

    #[test]
    fn duplicate_event_identity_is_refused() {
        let mut plan = overlapping_planned_plan();
        plan.events.push(plan.events[0].clone());
        let report = evaluate_lifecycle_service_budget(&plan);
        assert_eq!(
            report.disposition,
            LifecycleServiceDisposition::NotEvaluable
        );
        assert!(
            report
                .diagnostics
                .contains(&LifecycleDiagnostic::DuplicateEventId)
        );
    }

    #[test]
    fn missing_event_evidence_is_refused() {
        let mut plan = overlapping_planned_plan();
        plan.events[0].evidence_ref.clear();
        let report = evaluate_lifecycle_service_budget(&plan);
        assert_eq!(
            report.disposition,
            LifecycleServiceDisposition::NotEvaluable
        );
        assert!(
            report
                .diagnostics
                .contains(&LifecycleDiagnostic::EmptyEventEvidence)
        );
    }

    #[test]
    fn negative_hot_standby_power_is_counted_as_import_energy() {
        let plan = base_plan(
            Vec::new(),
            vec![segment(
                "standby",
                0,
                1000,
                -10.0,
                PowerOperatingState::HotStandby,
                &[],
            )],
        );
        let report = evaluate_lifecycle_service_budget(&plan);
        assert_eq!(
            report.disposition,
            LifecycleServiceDisposition::ValidDeterministicLedger
        );
        assert_eq!(report.signed_net_energy_mwh, Some(-10.0 / 3.6));
        assert_eq!(report.export_energy_mwh, Some(0.0));
        assert_eq!(report.import_energy_mwh, Some(10.0 / 3.6));
    }

    #[test]
    fn energy_is_integrated_across_each_signed_power_segment() {
        let plan = base_plan(
            Vec::new(),
            vec![
                segment(
                    "generation",
                    0,
                    600,
                    120.0,
                    PowerOperatingState::PowerGeneration,
                    &[],
                ),
                segment(
                    "standby",
                    600,
                    1000,
                    -9.0,
                    PowerOperatingState::HotStandby,
                    &[],
                ),
            ],
        );
        let report = evaluate_lifecycle_service_budget(&plan);
        assert_eq!(report.export_energy_mwh, Some(20.0));
        assert_eq!(report.import_energy_mwh, Some(1.0));
        assert_eq!(report.signed_net_energy_mwh, Some(19.0));
    }

    #[test]
    fn half_open_event_intervals_do_not_overlap_at_touching_boundaries() {
        let plan = base_plan(
            vec![
                event(
                    "planned-a",
                    OutageCategory::Planned,
                    ServiceImpact::FullOutage,
                    100,
                    300,
                ),
                event(
                    "forced-b",
                    OutageCategory::Unplanned,
                    ServiceImpact::FullOutage,
                    300,
                    500,
                ),
            ],
            vec![
                segment(
                    "s0",
                    0,
                    100,
                    80.0,
                    PowerOperatingState::PowerGeneration,
                    &[],
                ),
                segment(
                    "s1",
                    100,
                    300,
                    -5.0,
                    PowerOperatingState::PlannedMaintenance,
                    &["planned-a"],
                ),
                segment(
                    "s2",
                    300,
                    500,
                    -6.0,
                    PowerOperatingState::ForcedOutage,
                    &["forced-b"],
                ),
                segment(
                    "s3",
                    500,
                    1000,
                    80.0,
                    PowerOperatingState::PowerGeneration,
                    &[],
                ),
            ],
        );
        let report = evaluate_lifecycle_service_budget(&plan);
        assert_eq!(report.planned_only_unavailable_seconds, Some(200));
        assert_eq!(report.unplanned_only_unavailable_seconds, Some(200));
        assert_eq!(report.concurrent_unavailable_seconds, Some(0));
    }

    #[test]
    fn event_input_order_does_not_change_canonical_report() {
        let first = overlapping_planned_plan();
        let mut second = overlapping_planned_plan();
        second.events.reverse();
        let report_a = evaluate_lifecycle_service_budget(&first);
        let report_b = evaluate_lifecycle_service_budget(&second);
        assert_eq!(report_a, report_b);
    }

    #[test]
    fn derated_events_reduce_energy_without_becoming_full_unavailability() {
        let plan = base_plan(
            vec![event(
                "pump-derate",
                OutageCategory::Unplanned,
                ServiceImpact::Derated,
                200,
                800,
            )],
            vec![
                segment(
                    "full",
                    0,
                    200,
                    100.0,
                    PowerOperatingState::PowerGeneration,
                    &[],
                ),
                segment(
                    "derated",
                    200,
                    800,
                    60.0,
                    PowerOperatingState::ReducedPower,
                    &["pump-derate"],
                ),
                segment(
                    "full-again",
                    800,
                    1000,
                    100.0,
                    PowerOperatingState::PowerGeneration,
                    &[],
                ),
            ],
        );
        let report = evaluate_lifecycle_service_budget(&plan);
        assert_eq!(report.availability_fraction, Some(1.0));
        let expected = 100.0 * 400.0 / 3600.0 + 60.0 * 600.0 / 3600.0;
        assert!((report.signed_net_energy_mwh.unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn outage_state_without_full_outage_event_is_refused() {
        let plan = base_plan(
            Vec::new(),
            vec![segment(
                "false-maintenance",
                0,
                1000,
                -5.0,
                PowerOperatingState::PlannedMaintenance,
                &[],
            )],
        );
        let report = evaluate_lifecycle_service_budget(&plan);
        assert_eq!(
            report.disposition,
            LifecycleServiceDisposition::NotEvaluable
        );
        assert!(
            report
                .diagnostics
                .contains(&LifecycleDiagnostic::OutageStateWithoutFullOutageEvent)
        );
    }

    #[test]
    fn full_outage_category_and_operating_state_must_agree() {
        let mut plan = overlapping_planned_plan();
        plan.power_segments[1].state = PowerOperatingState::ForcedOutage;
        let report = evaluate_lifecycle_service_budget(&plan);
        assert_eq!(
            report.disposition,
            LifecycleServiceDisposition::NotEvaluable
        );
        assert!(
            report
                .diagnostics
                .contains(&LifecycleDiagnostic::OutageStateMismatch)
        );
    }

    #[test]
    fn every_segment_requires_power_balance_receipt_and_output_digest() {
        let mut plan = overlapping_planned_plan();
        plan.power_segments[0].power_balance_report_ref.clear();
        plan.power_segments[1].output_sha256 = "bad".to_string();
        let report = evaluate_lifecycle_service_budget(&plan);
        assert_eq!(
            report.disposition,
            LifecycleServiceDisposition::NotEvaluable
        );
        assert!(
            report
                .diagnostics
                .contains(&LifecycleDiagnostic::MissingPowerBalanceReceipt)
        );
        assert!(
            report
                .diagnostics
                .contains(&LifecycleDiagnostic::InvalidSegmentOutputDigest)
        );
    }

    #[test]
    fn zero_event_horizon_is_fully_available_and_reconciles_all_time() {
        let plan = base_plan(
            Vec::new(),
            vec![segment(
                "generation",
                0,
                1000,
                90.0,
                PowerOperatingState::PowerGeneration,
                &[],
            )],
        );
        let report = evaluate_lifecycle_service_budget(&plan);
        assert_eq!(report.available_seconds, Some(1000));
        assert_eq!(report.total_unavailable_seconds, Some(0));
        assert_eq!(report.availability_fraction, Some(1.0));
    }

    #[test]
    fn evidence_class_is_preserved_without_promotion() {
        let mut plan = overlapping_planned_plan();
        plan.events[0].evidence_class = LifecycleEvidenceClass::Assumed;
        let report = evaluate_lifecycle_service_budget(&plan);
        assert_eq!(
            report.disposition,
            LifecycleServiceDisposition::ValidDeterministicLedger
        );
        assert_eq!(
            report.event_receipts[0].evidence_class,
            LifecycleEvidenceClass::Assumed
        );
    }

    #[test]
    fn non_finite_state_power_is_not_treated_as_zero() {
        let plan = base_plan(
            Vec::new(),
            vec![segment(
                "unknown-power",
                0,
                1000,
                f64::NAN,
                PowerOperatingState::HotStandby,
                &[],
            )],
        );
        let report = evaluate_lifecycle_service_budget(&plan);
        assert_eq!(
            report.disposition,
            LifecycleServiceDisposition::NotEvaluable
        );
        assert!(
            report
                .diagnostics
                .contains(&LifecycleDiagnostic::NonFiniteNetPower)
        );
    }
}
