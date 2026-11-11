//! Offline cross-system causal mission trace replay.
//!
//! This reference checks structural lineage over supplied observation,
//! decision, admission, action-prefix, assurance, actuator, reserve and
//! terminal records. It never authorizes actuator application.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

/// Bounded timestamp on the mission's declared monotonic clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimestampInterval {
    /// Earliest possible event time, in integer nanoseconds.
    pub lower_ns: u64,
    /// Latest possible event time, in integer nanoseconds.
    pub upper_ns: u64,
}

/// Common causal identity on every trace event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventStamp {
    /// Unique event identity.
    pub event_id: String,
    /// Strictly ordered event sequence within the mission trace.
    pub sequence: u64,
    /// Robot control tick associated with this event.
    pub tick: u64,
    /// Robot identity.
    pub robot_id: String,
    /// Clock identity.
    pub clock_id: String,
    /// Boot epoch identity.
    pub boot_epoch: u64,
    /// Bounded event timestamp.
    pub timestamp: TimestampInterval,
}

/// Mission-level identity and evidence scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissionScope {
    /// Unique mission-run identity.
    pub mission_id: String,
    /// Robot identity.
    pub robot_id: String,
    /// Declared monotonic clock identity.
    pub clock_id: String,
    /// Declared boot epoch.
    pub boot_epoch: u64,
    /// Frozen scenario/configuration manifest digest.
    pub scenario_manifest_digest: String,
    /// Initial persistent physical-reserve state digest.
    pub initial_reserve_state_digest: String,
    /// Complete required thermal-node identity manifest for this embodiment.
    pub required_thermal_node_ids: Vec<String>,
}

/// One source observation consumed by a decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceObservation {
    /// Source-observation identity.
    pub observation_id: String,
    /// Common event identity/time.
    pub stamp: EventStamp,
    /// Sensor modality.
    pub modality: String,
    /// Monotonic sensor sequence number.
    pub source_sequence: u64,
    /// Sensor calibration / identity digest.
    pub sensor_digest: String,
}

/// Qualification state of the decision calibration receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationDisposition {
    /// Upstream calibration receipt is qualified within scope.
    Qualified,
    /// Required calibration evidence is unavailable or stale.
    Unknown,
    /// Upstream calibration receipt is known invalid.
    Invalid,
}

/// A selected action and the exact evidence/calibration scope used to justify it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionRecord {
    /// Decision identity.
    pub decision_id: String,
    /// Common event identity/time.
    pub stamp: EventStamp,
    /// Selected action identity.
    pub selected_action_id: String,
    /// Exact ordered source-observation dependency identities.
    pub source_observation_ids: Vec<String>,
    /// Loss model digest.
    pub loss_model_digest: String,
    /// Task-loss scale identity.
    pub loss_scale_id: String,
    /// Candidate action-domain manifest digest.
    pub action_domain_digest: String,
    /// Calibration receipt identity.
    pub calibration_id: String,
    /// Calibration receipt digest.
    pub calibration_digest: String,
    /// Upstream calibration status.
    pub calibration_disposition: CalibrationDisposition,
    /// Conservative stale-action regret upper bound in task-loss units.
    pub regret_upper_units: u64,
    /// Task-specific tolerated regret in the same loss scale.
    pub epsilon_task_loss_units: u64,
    /// Last control tick for which this decision remains sufficient.
    pub valid_through_tick: u64,
}

/// Admission state for an optional remote inference request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionDisposition {
    /// Request was admitted by the upstream constrained scheduler.
    Admitted,
    /// Request was rejected or not admitted.
    Rejected,
    /// Admission evidence is missing or inconclusive.
    Unknown,
}

/// Optional remote inference request and its returned response event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteInferenceRecord {
    /// Request identity.
    pub request_id: String,
    /// Request event stamp.
    pub request_stamp: EventStamp,
    /// Upstream admission status.
    pub admission: AdmissionDisposition,
    /// Optional response event stamp.
    pub response_stamp: Option<EventStamp>,
    /// Model identity returned by the remote service.
    pub response_model_digest: String,
    /// Context/prefix digest supplied to the remote service.
    pub context_digest: String,
    /// Last tick at which the response could still be used.
    pub expiry_tick: u64,
}

/// Proposed next action chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionChunkRecord {
    /// Proposal identity.
    pub proposal_id: String,
    /// Proposal event stamp.
    pub stamp: EventStamp,
    /// Decision that generated the proposal.
    pub decision_id: String,
    /// Optional remote request responsible for this result.
    pub request_id: Option<String>,
    /// Proposed action identity.
    pub action_id: String,
    /// Policy/model digest.
    pub model_digest: String,
    /// Candidate action-domain digest.
    pub action_domain_digest: String,
    /// Prefix digest the proposal expected to have executed.
    pub expected_executed_prefix_digest: String,
    /// Prefix digest independently observed as actually executed at proposal time.
    pub observed_executed_prefix_digest: String,
}

/// Result of an independent projection/certificate/controller-boundary gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssuranceDisposition {
    /// All required upstream gates reported the proposal eligible for application.
    Admitted,
    /// An upstream gate rejected the proposal.
    Rejected,
    /// Required assurance evidence is absent or inconclusive.
    Unknown,
}

/// Independent upstream disposition for one action proposal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssuranceRecord {
    /// Proposal identity.
    pub proposal_id: String,
    /// Assurance event stamp.
    pub stamp: EventStamp,
    /// Upstream result; this replay does not recompute it.
    pub disposition: AssuranceDisposition,
}

/// Command actually applied to an actuator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActuatorApplication {
    /// Unique command/application identity.
    pub command_id: String,
    /// Actual actuator application event.
    pub stamp: EventStamp,
    /// Action proposal applied.
    pub proposal_id: String,
    /// Decision lineage.
    pub decision_id: String,
    /// Applied action identity.
    pub action_id: String,
    /// Exact ordered source-observation dependencies.
    pub source_observation_ids: Vec<String>,
    /// Optional remote request lineage.
    pub request_id: Option<String>,
    /// Actuator identity.
    pub actuator_id: String,
    /// Controller/actuator application receipt digest.
    pub application_receipt_digest: String,
}

/// Post-application topology evidence state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyDisposition {
    /// Topology was independently recomputed/confirmed for this transition.
    Known,
    /// Topology is Unknown and cannot be copied forward as unchanged.
    Unknown,
}

/// Persistent physical reserve state transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalTransition {
    /// Transition identity.
    pub transition_id: String,
    /// Transition event stamp.
    pub stamp: EventStamp,
    /// Command whose physical effect is reconciled.
    pub command_id: String,
    /// Exact predecessor reserve-state digest.
    pub predecessor_state_digest: String,
    /// Resulting reserve-state digest.
    pub resulting_state_digest: String,
    /// Thermal model identity.
    pub thermal_model_digest: String,
    /// Thermal snapshot digest.
    pub thermal_snapshot_digest: String,
    /// Actual thermal-node IDs included in the snapshot.
    pub thermal_node_ids: Vec<String>,
    /// Topology receipt disposition.
    pub topology: TopologyDisposition,
}

/// Terminal mission-closure state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalDisposition {
    /// Qualified terminal safe-stop or retreat was independently evidenced.
    SafeStopQualified,
    /// Terminal state is unknown or incomplete.
    Unknown,
    /// Mission ended without qualified terminal safe-stop or retreat.
    NotQualified,
}

/// Final mission terminal receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalRecord {
    /// Terminal event stamp.
    pub stamp: EventStamp,
    /// Terminal safe-stop disposition.
    pub disposition: TerminalDisposition,
    /// Reserve-state digest at terminal closure.
    pub reserve_state_digest: String,
}

/// Complete finite mission trace supplied to the offline auditor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissionTrace {
    /// Frozen mission scope.
    pub scope: MissionScope,
    /// Source observations.
    pub observations: Vec<SourceObservation>,
    /// Decision records.
    pub decisions: Vec<DecisionRecord>,
    /// Optional remote inference requests/responses.
    pub remote_requests: Vec<RemoteInferenceRecord>,
    /// Action proposals.
    pub chunks: Vec<ActionChunkRecord>,
    /// Independent assurance results.
    pub assurance: Vec<AssuranceRecord>,
    /// Actual actuator applications.
    pub applications: Vec<ActuatorApplication>,
    /// Persistent physical reserve transitions.
    pub physical_transitions: Vec<PhysicalTransition>,
    /// Terminal mission closure receipt.
    pub terminal: Option<TerminalRecord>,
}

/// Severity of a trace finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FindingSeverity {
    /// A deterministic structural or causal contract was violated.
    Invalid,
    /// Supplied evidence cannot establish the required relationship.
    Unknown,
}

/// Reason-coded causal replay finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayFinding {
    /// Invalid or Unknown classification.
    pub severity: FindingSeverity,
    /// Stable reason code.
    pub code: &'static str,
    /// Primary event or record identity.
    pub record_id: String,
}

/// Actuation-side age result for one applied command / source observation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AoIBound {
    /// Command/application identity.
    pub command_id: String,
    /// Source observation identity.
    pub observation_id: String,
    /// Minimum possible source-to-actuator age, in nanoseconds.
    pub minimum_age_ns: u64,
    /// Maximum possible source-to-actuator age, in nanoseconds.
    pub maximum_age_ns: u64,
}

/// Overall result of the offline causal replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayDisposition {
    /// All required structural relationships in this supplied trace are closed.
    Closed,
    /// No deterministic invalidity found, but required evidence is inconclusive.
    Unknown,
    /// At least one deterministic structural or causal violation exists.
    Rejected,
}

/// Causal replay report. This report never grants actuator authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayReport {
    /// Final replay disposition.
    pub disposition: ReplayDisposition,
    /// Reason-coded findings.
    pub findings: Vec<ReplayFinding>,
    /// Same-clock source-to-actuator age bounds.
    pub aoi_bounds: Vec<AoIBound>,
    /// Number of actuator applications examined.
    pub applications_examined: usize,
    /// Number of applications with a complete linked chain.
    pub applications_linked: usize,
    /// Whether a qualified terminal safe-stop/retreat receipt closed the mission.
    pub terminal_safe_stop_qualified: bool,
    /// Explicitly false; replay is retrospective and has no control authority.
    pub actuation_authorized: bool,
}

struct StampIndex {
    ids: BTreeSet<String>,
    sequences: BTreeSet<u64>,
}

fn push_finding(
    findings: &mut Vec<ReplayFinding>,
    severity: FindingSeverity,
    code: &'static str,
    record_id: impl Into<String>,
) {
    findings.push(ReplayFinding {
        severity,
        code,
        record_id: record_id.into(),
    });
}

/// Check whether timestamp intervals support an already-declared causal sequence edge.
fn check_time_order(
    earlier: &EventStamp,
    later: &EventStamp,
    record_id: &str,
    findings: &mut Vec<ReplayFinding>,
) -> bool {
    if earlier.clock_id != later.clock_id || earlier.boot_epoch != later.boot_epoch {
        push_finding(
            findings,
            FindingSeverity::Unknown,
            "CAUSAL_TIME_CROSS_CLOCK_OR_BOOT",
            record_id,
        );
        return false;
    }
    if later.timestamp.upper_ns < earlier.timestamp.lower_ns {
        push_finding(
            findings,
            FindingSeverity::Invalid,
            "CAUSAL_TIMESTAMP_CONTRADICTS_SEQUENCE",
            record_id,
        );
        return false;
    }
    if later.timestamp.lower_ns < earlier.timestamp.upper_ns {
        push_finding(
            findings,
            FindingSeverity::Unknown,
            "CAUSAL_TIMESTAMP_ORDER_UNCERTAIN",
            record_id,
        );
        return false;
    }
    true
}

fn register_stamp(
    stamp: &EventStamp,
    scope: &MissionScope,
    index: &mut StampIndex,
    findings: &mut Vec<ReplayFinding>,
) {
    if stamp.event_id.trim().is_empty() {
        push_finding(findings, FindingSeverity::Invalid, "EMPTY_EVENT_ID", "<empty>");
    }
    if !index.ids.insert(stamp.event_id.clone()) {
        push_finding(findings, FindingSeverity::Invalid, "DUPLICATE_EVENT_ID", &stamp.event_id);
    }
    if !index.sequences.insert(stamp.sequence) {
        push_finding(findings, FindingSeverity::Invalid, "DUPLICATE_EVENT_SEQUENCE", &stamp.event_id);
    }
    if stamp.robot_id != scope.robot_id
        || stamp.clock_id != scope.clock_id
        || stamp.boot_epoch != scope.boot_epoch
    {
        push_finding(
            findings,
            FindingSeverity::Unknown,
            "ROBOT_CLOCK_OR_BOOT_SCOPE_MISMATCH",
            &stamp.event_id,
        );
    }
    if stamp.timestamp.lower_ns > stamp.timestamp.upper_ns {
        push_finding(
            findings,
            FindingSeverity::Invalid,
            "INVALID_TIMESTAMP_INTERVAL",
            &stamp.event_id,
        );
    }
}

fn stamp_index(trace: &MissionTrace, findings: &mut Vec<ReplayFinding>) {
    let mut index = StampIndex {
        ids: BTreeSet::new(),
        sequences: BTreeSet::new(),
    };
    for record in &trace.observations {
        register_stamp(&record.stamp, &trace.scope, &mut index, findings);
    }
    for record in &trace.decisions {
        register_stamp(&record.stamp, &trace.scope, &mut index, findings);
    }
    for record in &trace.remote_requests {
        register_stamp(&record.request_stamp, &trace.scope, &mut index, findings);
        if let Some(response) = &record.response_stamp {
            register_stamp(response, &trace.scope, &mut index, findings);
        }
    }
    for record in &trace.chunks {
        register_stamp(&record.stamp, &trace.scope, &mut index, findings);
    }
    for record in &trace.assurance {
        register_stamp(&record.stamp, &trace.scope, &mut index, findings);
    }
    for record in &trace.applications {
        register_stamp(&record.stamp, &trace.scope, &mut index, findings);
    }
    for record in &trace.physical_transitions {
        register_stamp(&record.stamp, &trace.scope, &mut index, findings);
    }
    if let Some(terminal) = &trace.terminal {
        register_stamp(&terminal.stamp, &trace.scope, &mut index, findings);
    }
}

/// Replay all cross-system causal relationships in one finite mission trace.
///
/// This validates supplied record relationships, not the physical truth of those
/// records. It never produces a controller permit or actuator authority.
pub fn replay_mission(trace: &MissionTrace) -> ReplayReport {
    let mut findings = Vec::new();
    let mut aoi_bounds = Vec::new();
    stamp_index(trace, &mut findings);

    for (field, value) in [
        ("mission_id", trace.scope.mission_id.as_str()),
        ("robot_id", trace.scope.robot_id.as_str()),
        ("clock_id", trace.scope.clock_id.as_str()),
        ("scenario_manifest_digest", trace.scope.scenario_manifest_digest.as_str()),
        ("initial_reserve_state_digest", trace.scope.initial_reserve_state_digest.as_str()),
    ] {
        if value.trim().is_empty() {
            push_finding(
                &mut findings,
                FindingSeverity::Invalid,
                "EMPTY_MISSION_SCOPE_FIELD",
                field,
            );
        }
    }

    let mut required_nodes = BTreeSet::new();
    for node_id in &trace.scope.required_thermal_node_ids {
        if node_id.trim().is_empty() || !required_nodes.insert(node_id.as_str()) {
            push_finding(
                &mut findings,
                FindingSeverity::Invalid,
                "INVALID_REQUIRED_THERMAL_MANIFEST",
                node_id,
            );
        }
    }
    if required_nodes.is_empty() {
        push_finding(
            &mut findings,
            FindingSeverity::Unknown,
            "EMPTY_REQUIRED_THERMAL_MANIFEST",
            &trace.scope.mission_id,
        );
    }

    let observations: BTreeMap<&str, &SourceObservation> = trace
        .observations
        .iter()
        .map(|record| (record.observation_id.as_str(), record))
        .collect();
    let decisions: BTreeMap<&str, &DecisionRecord> = trace
        .decisions
        .iter()
        .map(|record| (record.decision_id.as_str(), record))
        .collect();
    let requests: BTreeMap<&str, &RemoteInferenceRecord> = trace
        .remote_requests
        .iter()
        .map(|record| (record.request_id.as_str(), record))
        .collect();
    let chunks: BTreeMap<&str, &ActionChunkRecord> = trace
        .chunks
        .iter()
        .map(|record| (record.proposal_id.as_str(), record))
        .collect();
    let assurance: BTreeMap<&str, &AssuranceRecord> = trace
        .assurance
        .iter()
        .map(|record| (record.proposal_id.as_str(), record))
        .collect();

    if observations.len() != trace.observations.len()
        || decisions.len() != trace.decisions.len()
        || requests.len() != trace.remote_requests.len()
        || chunks.len() != trace.chunks.len()
        || assurance.len() != trace.assurance.len()
    {
        push_finding(
            &mut findings,
            FindingSeverity::Invalid,
            "DUPLICATE_RECORD_ID",
            &trace.scope.mission_id,
        );
    }

    for observation in &trace.observations {
        if observation.observation_id.trim().is_empty()
            || observation.modality.trim().is_empty()
            || observation.sensor_digest.trim().is_empty()
        {
            push_finding(
                &mut findings,
                FindingSeverity::Unknown,
                "INCOMPLETE_SOURCE_OBSERVATION",
                &observation.observation_id,
            );
        }
    }

    for decision in &trace.decisions {
        if decision.decision_id.trim().is_empty()
            || decision.selected_action_id.trim().is_empty()
            || decision.loss_model_digest.trim().is_empty()
            || decision.loss_scale_id.trim().is_empty()
            || decision.action_domain_digest.trim().is_empty()
            || decision.calibration_id.trim().is_empty()
            || decision.calibration_digest.trim().is_empty()
        {
            push_finding(
                &mut findings,
                FindingSeverity::Unknown,
                "INCOMPLETE_DECISION_RECEIPT",
                &decision.decision_id,
            );
        }
        if decision.source_observation_ids.is_empty() {
            push_finding(
                &mut findings,
                FindingSeverity::Unknown,
                "DECISION_HAS_NO_SOURCE_OBSERVATIONS",
                &decision.decision_id,
            );
        }
        if decision.regret_upper_units > decision.epsilon_task_loss_units {
            push_finding(
                &mut findings,
                FindingSeverity::Invalid,
                "DECISION_REGRET_EXCEEDS_EPSILON",
                &decision.decision_id,
            );
        }
        match decision.calibration_disposition {
            CalibrationDisposition::Qualified => {}
            CalibrationDisposition::Unknown => push_finding(
                &mut findings,
                FindingSeverity::Unknown,
                "DECISION_CALIBRATION_UNKNOWN",
                &decision.decision_id,
            ),
            CalibrationDisposition::Invalid => push_finding(
                &mut findings,
                FindingSeverity::Invalid,
                "DECISION_CALIBRATION_INVALID",
                &decision.decision_id,
            ),
        }

        let mut seen_sources = BTreeSet::new();
        for observation_id in &decision.source_observation_ids {
            if !seen_sources.insert(observation_id.as_str()) {
                push_finding(
                    &mut findings,
                    FindingSeverity::Invalid,
                    "DUPLICATE_DECISION_SOURCE",
                    &decision.decision_id,
                );
            }
            match observations.get(observation_id.as_str()) {
                Some(observation) => {
                    if observation.stamp.sequence >= decision.stamp.sequence
                        || observation.stamp.tick > decision.stamp.tick
                    {
                        push_finding(
                            &mut findings,
                            FindingSeverity::Invalid,
                            "DECISION_LOOKAHEAD_SOURCE",
                            &decision.decision_id,
                        );
                    } else {
                        check_time_order(
                            &observation.stamp,
                            &decision.stamp,
                            &decision.decision_id,
                            &mut findings,
                        );
                    }
                }
                None => push_finding(
                    &mut findings,
                    FindingSeverity::Invalid,
                    "MISSING_DECISION_SOURCE",
                    &decision.decision_id,
                ),
            }
        }
    }

    for chunk in &trace.chunks {
        let decision = match decisions.get(chunk.decision_id.as_str()) {
            Some(value) => *value,
            None => {
                push_finding(
                    &mut findings,
                    FindingSeverity::Invalid,
                    "MISSING_CHUNK_DECISION",
                    &chunk.proposal_id,
                );
                continue;
            }
        };
        if chunk.stamp.sequence <= decision.stamp.sequence
            || chunk.stamp.tick > decision.valid_through_tick
            || chunk.action_id != decision.selected_action_id
            || chunk.model_digest != decision.loss_model_digest
            || chunk.action_domain_digest != decision.action_domain_digest
        {
            push_finding(
                &mut findings,
                FindingSeverity::Invalid,
                "CHUNK_DECISION_BINDING_MISMATCH",
                &chunk.proposal_id,
            );
        } else {
            check_time_order(
                &decision.stamp,
                &chunk.stamp,
                &chunk.proposal_id,
                &mut findings,
            );
        }
        if chunk.expected_executed_prefix_digest.trim().is_empty()
            || chunk.observed_executed_prefix_digest.trim().is_empty()
        {
            push_finding(
                &mut findings,
                FindingSeverity::Unknown,
                "MISSING_EXECUTED_PREFIX_DIGEST",
                &chunk.proposal_id,
            );
        } else if chunk.expected_executed_prefix_digest != chunk.observed_executed_prefix_digest {
            push_finding(
                &mut findings,
                FindingSeverity::Invalid,
                "EXECUTED_PREFIX_MISMATCH",
                &chunk.proposal_id,
            );
        }

        if let Some(request_id) = &chunk.request_id {
            match requests.get(request_id.as_str()) {
                Some(request) => {
                    match request.admission {
                        AdmissionDisposition::Admitted => {}
                        AdmissionDisposition::Rejected => push_finding(
                            &mut findings,
                            FindingSeverity::Invalid,
                            "REMOTE_REQUEST_NOT_ADMITTED",
                            &chunk.proposal_id,
                        ),
                        AdmissionDisposition::Unknown => push_finding(
                            &mut findings,
                            FindingSeverity::Unknown,
                            "REMOTE_REQUEST_NOT_ADMITTED",
                            &chunk.proposal_id,
                        ),
                    }
                    match &request.response_stamp {
                        Some(response) => {
                            if request.request_stamp.sequence <= decision.stamp.sequence
                                || request.request_stamp.sequence >= response.sequence
                                || response.sequence >= chunk.stamp.sequence
                                || response.tick > request.expiry_tick
                            {
                                push_finding(
                                    &mut findings,
                                    FindingSeverity::Invalid,
                                    "REMOTE_RESPONSE_CAUSAL_OR_EXPIRY_VIOLATION",
                                    &chunk.proposal_id,
                                );
                            }
                            check_time_order(
                                &decision.stamp,
                                &request.request_stamp,
                                &chunk.proposal_id,
                                &mut findings,
                            );
                            check_time_order(
                                &request.request_stamp,
                                response,
                                &chunk.proposal_id,
                                &mut findings,
                            );
                            check_time_order(
                                response,
                                &chunk.stamp,
                                &chunk.proposal_id,
                                &mut findings,
                            );
                            if request.response_model_digest != chunk.model_digest
                                || request.context_digest != chunk.observed_executed_prefix_digest
                            {
                                push_finding(
                                    &mut findings,
                                    FindingSeverity::Invalid,
                                    "REMOTE_RESPONSE_MODEL_OR_CONTEXT_MISMATCH",
                                    &chunk.proposal_id,
                                );
                            }
                        }
                        None => push_finding(
                            &mut findings,
                            FindingSeverity::Unknown,
                            "REMOTE_RESPONSE_MISSING",
                            &chunk.proposal_id,
                        ),
                    }
                }
                None => push_finding(
                    &mut findings,
                    FindingSeverity::Invalid,
                    "MISSING_REMOTE_REQUEST",
                    &chunk.proposal_id,
                ),
            }
        }
    }

    let mut applications_linked = 0_usize;
    let mut application_by_command: BTreeMap<&str, &ActuatorApplication> = BTreeMap::new();
    for application in &trace.applications {
        let findings_before = findings.len();
        if application.command_id.trim().is_empty()
            || application.actuator_id.trim().is_empty()
            || application.application_receipt_digest.trim().is_empty()
        {
            push_finding(
                &mut findings,
                FindingSeverity::Unknown,
                "INCOMPLETE_ACTUATOR_APPLICATION",
                &application.command_id,
            );
        }
        if application_by_command
            .insert(application.command_id.as_str(), application)
            .is_some()
        {
            push_finding(
                &mut findings,
                FindingSeverity::Invalid,
                "DUPLICATE_COMMAND_APPLICATION",
                &application.command_id,
            );
        }

        let chunk = match chunks.get(application.proposal_id.as_str()) {
            Some(value) => *value,
            None => {
                push_finding(
                    &mut findings,
                    FindingSeverity::Invalid,
                    "APPLICATION_WITHOUT_PROPOSAL",
                    &application.command_id,
                );
                continue;
            }
        };
        let decision = match decisions.get(application.decision_id.as_str()) {
            Some(value) => *value,
            None => {
                push_finding(
                    &mut findings,
                    FindingSeverity::Invalid,
                    "APPLICATION_WITHOUT_DECISION",
                    &application.command_id,
                );
                continue;
            }
        };
        if chunk.decision_id != application.decision_id
            || chunk.action_id != application.action_id
            || chunk.stamp.sequence >= application.stamp.sequence
            || decision.selected_action_id != application.action_id
            || decision.source_observation_ids != application.source_observation_ids
            || chunk.request_id != application.request_id
        {
            push_finding(
                &mut findings,
                FindingSeverity::Invalid,
                "APPLICATION_LINEAGE_MISMATCH",
                &application.command_id,
            );
        }
        if application.stamp.tick > decision.valid_through_tick {
            push_finding(
                &mut findings,
                FindingSeverity::Invalid,
                "APPLICATION_AFTER_DECISION_HORIZON",
                &application.command_id,
            );
        }

        let gate_ok = match assurance.get(application.proposal_id.as_str()) {
            Some(gate) => {
                if gate.stamp.sequence <= chunk.stamp.sequence
                    || gate.stamp.sequence >= application.stamp.sequence
                {
                    push_finding(
                        &mut findings,
                        FindingSeverity::Invalid,
                        "ASSURANCE_EVENT_ORDER_INVALID",
                        &application.command_id,
                    );
                } else {
                    check_time_order(
                        &chunk.stamp,
                        &gate.stamp,
                        &application.command_id,
                        &mut findings,
                    );
                    check_time_order(
                        &gate.stamp,
                        &application.stamp,
                        &application.command_id,
                        &mut findings,
                    );
                }
                match gate.disposition {
                    AssuranceDisposition::Admitted => true,
                    AssuranceDisposition::Rejected => {
                        push_finding(
                            &mut findings,
                            FindingSeverity::Invalid,
                            "APPLICATION_AFTER_ASSURANCE_REJECTION",
                            &application.command_id,
                        );
                        false
                    }
                    AssuranceDisposition::Unknown => {
                        push_finding(
                            &mut findings,
                            FindingSeverity::Unknown,
                            "APPLICATION_ASSURANCE_UNKNOWN",
                            &application.command_id,
                        );
                        false
                    }
                }
            }
            None => {
                push_finding(
                    &mut findings,
                    FindingSeverity::Unknown,
                    "APPLICATION_ASSURANCE_MISSING",
                    &application.command_id,
                );
                false
            }
        };

        let mut command_aoi_complete = true;
        for observation_id in &application.source_observation_ids {
            match observations.get(observation_id.as_str()) {
                Some(observation) => {
                    let source_stamp = &observation.stamp;
                    let app_stamp = &application.stamp;
                    if source_stamp.clock_id != app_stamp.clock_id
                        || source_stamp.boot_epoch != app_stamp.boot_epoch
                    {
                        command_aoi_complete = false;
                        push_finding(
                            &mut findings,
                            FindingSeverity::Unknown,
                            "AOI_CROSS_CLOCK_OR_BOOT",
                            &application.command_id,
                        );
                        continue;
                    }
                    if !check_time_order(
                        source_stamp,
                        app_stamp,
                        &application.command_id,
                        &mut findings,
                    ) {
                        command_aoi_complete = false;
                        continue;
                    }
                    aoi_bounds.push(AoIBound {
                        command_id: application.command_id.clone(),
                        observation_id: observation_id.clone(),
                        minimum_age_ns: app_stamp.timestamp.lower_ns
                            - source_stamp.timestamp.upper_ns,
                        maximum_age_ns: app_stamp.timestamp.upper_ns
                            - source_stamp.timestamp.lower_ns,
                    });
                }
                None => {
                    command_aoi_complete = false;
                    push_finding(
                        &mut findings,
                        FindingSeverity::Invalid,
                        "APPLICATION_SOURCE_MISSING",
                        &application.command_id,
                    );
                }
            }
        }

        let remote_ok = match &application.request_id {
            Some(request_id) => match requests.get(request_id.as_str()) {
                Some(request) => {
                    if request.admission != AdmissionDisposition::Admitted {
                        false
                    } else if application.stamp.tick > request.expiry_tick {
                        push_finding(
                            &mut findings,
                            FindingSeverity::Invalid,
                            "REMOTE_RESULT_APPLIED_AFTER_EXPIRY",
                            &application.command_id,
                        );
                        false
                    } else {
                        request.response_stamp.as_ref().is_some_and(|response| {
                            request.request_stamp.sequence < response.sequence
                                && response.sequence < chunk.stamp.sequence
                                && response.tick <= request.expiry_tick
                                && request.response_model_digest == chunk.model_digest
                                && request.context_digest == chunk.observed_executed_prefix_digest
                        })
                    }
                }
                None => false,
            },
            None => true,
        };
        let decision_source_order_ok = decision.source_observation_ids.iter().all(|id| {
            observations.get(id.as_str()).is_some_and(|observation| {
                observation.stamp.sequence < decision.stamp.sequence
                    && observation.stamp.clock_id == application.stamp.clock_id
                    && observation.stamp.boot_epoch == application.stamp.boot_epoch
                    && !observation.sensor_digest.trim().is_empty()
            })
        });
        let decision_receipt_complete = !decision.calibration_id.trim().is_empty()
            && !decision.calibration_digest.trim().is_empty()
            && !decision.loss_model_digest.trim().is_empty()
            && !decision.loss_scale_id.trim().is_empty()
            && !decision.action_domain_digest.trim().is_empty();
        let chunk_binding_ok = chunk.decision_id == application.decision_id
            && chunk.action_id == application.action_id
            && chunk.model_digest == decision.loss_model_digest
            && chunk.action_domain_digest == decision.action_domain_digest
            && chunk.expected_executed_prefix_digest == chunk.observed_executed_prefix_digest;

        if findings.len() == findings_before
            && command_aoi_complete
            && gate_ok
            && remote_ok
            && decision_source_order_ok
            && decision_receipt_complete
            && chunk_binding_ok
            && decision.regret_upper_units <= decision.epsilon_task_loss_units
            && decision.calibration_disposition == CalibrationDisposition::Qualified
        {
            applications_linked += 1;
        }
    }

    let mut transitions = trace.physical_transitions.iter().collect::<Vec<_>>();
    transitions.sort_by_key(|record| record.stamp.sequence);
    let mut expected_predecessor = trace.scope.initial_reserve_state_digest.as_str();
    let mut transition_commands = BTreeSet::new();
    let mut last_physical_sequence = 0_u64;
    let mut last_physical_stamp: Option<&EventStamp> = None;

    for transition in transitions {
        last_physical_sequence = last_physical_sequence.max(transition.stamp.sequence);
        if last_physical_stamp.is_none_or(|previous| previous.sequence < transition.stamp.sequence) {
            last_physical_stamp = Some(&transition.stamp);
        }
        if transition.transition_id.trim().is_empty()
            || transition.resulting_state_digest.trim().is_empty()
            || transition.thermal_model_digest.trim().is_empty()
            || transition.thermal_snapshot_digest.trim().is_empty()
        {
            push_finding(
                &mut findings,
                FindingSeverity::Unknown,
                "INCOMPLETE_PHYSICAL_TRANSITION",
                &transition.transition_id,
            );
        }
        if transition.predecessor_state_digest != expected_predecessor {
            push_finding(
                &mut findings,
                FindingSeverity::Invalid,
                "PHYSICAL_RESERVE_CHAIN_MISMATCH",
                &transition.transition_id,
            );
        }
        expected_predecessor = transition.resulting_state_digest.as_str();

        if !transition_commands.insert(transition.command_id.as_str()) {
            push_finding(
                &mut findings,
                FindingSeverity::Invalid,
                "DUPLICATE_PHYSICAL_COMMAND_TRANSITION",
                &transition.command_id,
            );
        }
        match application_by_command.get(transition.command_id.as_str()) {
            Some(application) => {
                if application.stamp.sequence >= transition.stamp.sequence {
                    push_finding(
                        &mut findings,
                        FindingSeverity::Invalid,
                        "PHYSICAL_TRANSITION_PRECEDES_APPLICATION",
                        &transition.transition_id,
                    );
                } else {
                    check_time_order(
                        &application.stamp,
                        &transition.stamp,
                        &transition.transition_id,
                        &mut findings,
                    );
                }
            }
            None => push_finding(
                &mut findings,
                FindingSeverity::Invalid,
                "PHYSICAL_TRANSITION_WITHOUT_APPLICATION",
                &transition.transition_id,
            ),
        }

        let actual_nodes: BTreeSet<&str> = transition
            .thermal_node_ids
            .iter()
            .map(String::as_str)
            .collect();
        if actual_nodes.len() != transition.thermal_node_ids.len()
            || actual_nodes != required_nodes
        {
            push_finding(
                &mut findings,
                FindingSeverity::Unknown,
                "THERMAL_NODE_MANIFEST_INCOMPLETE_OR_DUPLICATE",
                &transition.transition_id,
            );
        }
        if transition.topology == TopologyDisposition::Unknown {
            push_finding(
                &mut findings,
                FindingSeverity::Unknown,
                "RECOVERY_TOPOLOGY_UNKNOWN",
                &transition.transition_id,
            );
        }
    }

    for application in &trace.applications {
        if !transition_commands.contains(application.command_id.as_str()) {
            push_finding(
                &mut findings,
                FindingSeverity::Unknown,
                "APPLICATION_PHYSICAL_EFFECT_MISSING",
                &application.command_id,
            );
        }
    }

    let mut terminal_safe_stop_qualified = false;
    match &trace.terminal {
        Some(terminal) => {
            if terminal.stamp.sequence <= last_physical_sequence
                || terminal.reserve_state_digest != expected_predecessor
            {
                push_finding(
                    &mut findings,
                    FindingSeverity::Invalid,
                    "TERMINAL_STATE_OR_ORDER_MISMATCH",
                    &terminal.stamp.event_id,
                );
            } else if let Some(previous) = last_physical_stamp {
                check_time_order(
                    previous,
                    &terminal.stamp,
                    &terminal.stamp.event_id,
                    &mut findings,
                );
            }
            match terminal.disposition {
                TerminalDisposition::SafeStopQualified => {
                    terminal_safe_stop_qualified = true;
                }
                TerminalDisposition::Unknown => push_finding(
                    &mut findings,
                    FindingSeverity::Unknown,
                    "TERMINAL_SAFE_STOP_UNKNOWN",
                    &terminal.stamp.event_id,
                ),
                TerminalDisposition::NotQualified => push_finding(
                    &mut findings,
                    FindingSeverity::Invalid,
                    "TERMINAL_SAFE_STOP_NOT_QUALIFIED",
                    &terminal.stamp.event_id,
                ),
            }
        }
        None => push_finding(
            &mut findings,
            FindingSeverity::Unknown,
            "TERMINAL_SAFE_STOP_RECEIPT_MISSING",
            &trace.scope.mission_id,
        ),
    }

    let disposition = if findings.iter().any(|finding| finding.severity == FindingSeverity::Invalid) {
        ReplayDisposition::Rejected
    } else if findings.iter().any(|finding| finding.severity == FindingSeverity::Unknown) {
        ReplayDisposition::Unknown
    } else {
        ReplayDisposition::Closed
    };

    ReplayReport {
        disposition,
        findings,
        aoi_bounds,
        applications_examined: trace.applications.len(),
        applications_linked,
        terminal_safe_stop_qualified,
        actuation_authorized: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamp(id: &str, sequence: u64, tick: u64, time: u64) -> EventStamp {
        EventStamp {
            event_id: id.to_owned(),
            sequence,
            tick,
            robot_id: "robot-01".to_owned(),
            clock_id: "robot-clock".to_owned(),
            boot_epoch: 7,
            timestamp: TimestampInterval {
                lower_ns: time,
                upper_ns: time,
            },
        }
    }

    fn valid_trace() -> MissionTrace {
        MissionTrace {
            scope: MissionScope {
                mission_id: "mission-1".to_owned(),
                robot_id: "robot-01".to_owned(),
                clock_id: "robot-clock".to_owned(),
                boot_epoch: 7,
                scenario_manifest_digest: "scenario-sha256".to_owned(),
                initial_reserve_state_digest: "reserve-0".to_owned(),
                required_thermal_node_ids: vec!["motor".to_owned(), "driver".to_owned()],
            },
            observations: vec![SourceObservation {
                observation_id: "obs-1".to_owned(),
                stamp: stamp("event-1", 1, 1, 100),
                modality: "camera".to_owned(),
                source_sequence: 1,
                sensor_digest: "sensor-sha256".to_owned(),
            }],
            decisions: vec![DecisionRecord {
                decision_id: "decision-1".to_owned(),
                stamp: stamp("event-2", 2, 1, 110),
                selected_action_id: "walk-forward".to_owned(),
                source_observation_ids: vec!["obs-1".to_owned()],
                loss_model_digest: "loss-model-sha256".to_owned(),
                loss_scale_id: "mission-loss-v1".to_owned(),
                action_domain_digest: "action-domain-sha256".to_owned(),
                calibration_id: "calibration-1".to_owned(),
                calibration_digest: "calibration-receipt-sha256".to_owned(),
                calibration_disposition: CalibrationDisposition::Qualified,
                regret_upper_units: 2,
                epsilon_task_loss_units: 5,
                valid_through_tick: 10,
            }],
            remote_requests: Vec::new(),
            chunks: vec![ActionChunkRecord {
                proposal_id: "proposal-1".to_owned(),
                stamp: stamp("event-3", 3, 2, 120),
                decision_id: "decision-1".to_owned(),
                request_id: None,
                action_id: "walk-forward".to_owned(),
                model_digest: "loss-model-sha256".to_owned(),
                action_domain_digest: "action-domain-sha256".to_owned(),
                expected_executed_prefix_digest: "prefix-0".to_owned(),
                observed_executed_prefix_digest: "prefix-0".to_owned(),
            }],
            assurance: vec![AssuranceRecord {
                proposal_id: "proposal-1".to_owned(),
                stamp: stamp("event-4", 4, 2, 130),
                disposition: AssuranceDisposition::Admitted,
            }],
            applications: vec![ActuatorApplication {
                command_id: "command-1".to_owned(),
                stamp: stamp("event-5", 5, 3, 200),
                proposal_id: "proposal-1".to_owned(),
                decision_id: "decision-1".to_owned(),
                action_id: "walk-forward".to_owned(),
                source_observation_ids: vec!["obs-1".to_owned()],
                request_id: None,
                actuator_id: "left-knee".to_owned(),
                application_receipt_digest: "apply-sha256".to_owned(),
            }],
            physical_transitions: vec![PhysicalTransition {
                transition_id: "transition-1".to_owned(),
                stamp: stamp("event-6", 6, 3, 220),
                command_id: "command-1".to_owned(),
                predecessor_state_digest: "reserve-0".to_owned(),
                resulting_state_digest: "reserve-1".to_owned(),
                thermal_model_digest: "thermal-model-sha256".to_owned(),
                thermal_snapshot_digest: "thermal-snapshot-sha256".to_owned(),
                thermal_node_ids: vec!["motor".to_owned(), "driver".to_owned()],
                topology: TopologyDisposition::Known,
            }],
            terminal: Some(TerminalRecord {
                stamp: stamp("event-7", 7, 4, 300),
                disposition: TerminalDisposition::SafeStopQualified,
                reserve_state_digest: "reserve-1".to_owned(),
            }),
        }
    }

    fn has_finding(report: &ReplayReport, code: &str) -> bool {
        report.findings.iter().any(|finding| finding.code == code)
    }

    #[test]
    fn complete_same_clock_trace_closes_without_authorizing_actuation() {
        let report = replay_mission(&valid_trace());
        assert_eq!(report.disposition, ReplayDisposition::Closed);
        assert_eq!(report.applications_examined, 1);
        assert_eq!(report.applications_linked, 1);
        assert_eq!(report.aoi_bounds[0].minimum_age_ns, 100);
        assert_eq!(report.aoi_bounds[0].maximum_age_ns, 100);
        assert!(report.terminal_safe_stop_qualified);
        assert!(!report.actuation_authorized);
    }

    #[test]
    fn application_after_decision_horizon_is_rejected() {
        let mut trace = valid_trace();
        trace.applications[0].stamp.tick = 11;
        let report = replay_mission(&trace);
        assert_eq!(report.disposition, ReplayDisposition::Rejected);
        assert!(has_finding(&report, "APPLICATION_AFTER_DECISION_HORIZON"));
    }

    #[test]
    fn source_observation_after_decision_is_rejected_as_lookahead() {
        let mut trace = valid_trace();
        trace.observations[0].stamp.sequence = 3;
        let report = replay_mission(&trace);
        assert_eq!(report.disposition, ReplayDisposition::Rejected);
        assert!(has_finding(&report, "DECISION_LOOKAHEAD_SOURCE"));
    }

    #[test]
    fn application_must_use_exact_decision_source_vector() {
        let mut trace = valid_trace();
        trace.applications[0].source_observation_ids = vec!["other-observation".to_owned()];
        let report = replay_mission(&trace);
        assert_eq!(report.disposition, ReplayDisposition::Rejected);
        assert!(has_finding(&report, "APPLICATION_LINEAGE_MISMATCH"));
    }

    #[test]
    fn mismatched_boot_epoch_preserves_unknown() {
        let mut trace = valid_trace();
        trace.observations[0].stamp.boot_epoch = 8;
        let report = replay_mission(&trace);
        assert_eq!(report.disposition, ReplayDisposition::Unknown);
        assert!(has_finding(&report, "ROBOT_CLOCK_OR_BOOT_SCOPE_MISMATCH"));
        assert!(has_finding(&report, "AOI_CROSS_CLOCK_OR_BOOT"));
    }

    #[test]
    fn denied_remote_request_cannot_supply_applied_action() {
        let mut trace = valid_trace();
        trace.remote_requests.push(RemoteInferenceRecord {
            request_id: "request-1".to_owned(),
            request_stamp: stamp("event-r1", 3, 1, 112),
            admission: AdmissionDisposition::Rejected,
            response_stamp: Some(stamp("event-r2", 4, 2, 115)),
            response_model_digest: "loss-model-sha256".to_owned(),
            context_digest: "prefix-0".to_owned(),
            expiry_tick: 8,
        });
        trace.chunks[0].request_id = Some("request-1".to_owned());
        trace.chunks[0].stamp.sequence = 5;
        trace.chunks[0].stamp.event_id = "event-3b".to_owned();
        trace.assurance[0].stamp.sequence = 6;
        trace.assurance[0].stamp.event_id = "event-4b".to_owned();
        trace.applications[0].stamp.sequence = 7;
        trace.applications[0].stamp.event_id = "event-5b".to_owned();
        trace.applications[0].request_id = Some("request-1".to_owned());
        trace.physical_transitions[0].stamp.sequence = 8;
        trace.physical_transitions[0].stamp.event_id = "event-6b".to_owned();
        trace.terminal.as_mut().unwrap().stamp.sequence = 9;
        trace.terminal.as_mut().unwrap().stamp.event_id = "event-7b".to_owned();
        let report = replay_mission(&trace);
        assert_eq!(report.disposition, ReplayDisposition::Rejected);
        assert!(has_finding(&report, "REMOTE_REQUEST_NOT_ADMITTED"));
    }

    #[test]
    fn remote_response_must_precede_chunk_and_be_unexpired() {
        let mut trace = valid_trace();
        trace.remote_requests.push(RemoteInferenceRecord {
            request_id: "request-1".to_owned(),
            request_stamp: stamp("event-r1", 3, 1, 112),
            admission: AdmissionDisposition::Admitted,
            response_stamp: Some(stamp("event-r2", 5, 2, 140)),
            response_model_digest: "loss-model-sha256".to_owned(),
            context_digest: "prefix-0".to_owned(),
            expiry_tick: 2,
        });
        trace.chunks[0].request_id = Some("request-1".to_owned());
        trace.chunks[0].stamp.sequence = 4;
        trace.chunks[0].stamp.event_id = "event-3b".to_owned();
        trace.assurance[0].stamp.sequence = 6;
        trace.assurance[0].stamp.event_id = "event-4b".to_owned();
        trace.applications[0].stamp.sequence = 7;
        trace.applications[0].stamp.event_id = "event-5b".to_owned();
        trace.applications[0].request_id = Some("request-1".to_owned());
        trace.physical_transitions[0].stamp.sequence = 8;
        trace.physical_transitions[0].stamp.event_id = "event-6b".to_owned();
        trace.terminal.as_mut().unwrap().stamp.sequence = 9;
        trace.terminal.as_mut().unwrap().stamp.event_id = "event-7b".to_owned();
        let report = replay_mission(&trace);
        assert_eq!(report.disposition, ReplayDisposition::Rejected);
        assert!(has_finding(&report, "REMOTE_RESPONSE_CAUSAL_OR_EXPIRY_VIOLATION"));
    }

    #[test]
    fn remote_result_cannot_be_applied_after_expiry_even_if_it_arrived_on_time() {
        let mut trace = valid_trace();
        trace.remote_requests.push(RemoteInferenceRecord {
            request_id: "request-1".to_owned(),
            request_stamp: stamp("event-r1", 3, 1, 112),
            admission: AdmissionDisposition::Admitted,
            response_stamp: Some(stamp("event-r2", 4, 2, 140)),
            response_model_digest: "loss-model-sha256".to_owned(),
            context_digest: "prefix-0".to_owned(),
            expiry_tick: 2,
        });
        trace.chunks[0].request_id = Some("request-1".to_owned());
        trace.chunks[0].stamp.sequence = 5;
        trace.chunks[0].stamp.event_id = "event-3b".to_owned();
        trace.assurance[0].stamp.sequence = 6;
        trace.assurance[0].stamp.event_id = "event-4b".to_owned();
        trace.applications[0].stamp.sequence = 7;
        trace.applications[0].stamp.event_id = "event-5b".to_owned();
        trace.applications[0].stamp.tick = 3;
        trace.applications[0].request_id = Some("request-1".to_owned());
        trace.physical_transitions[0].stamp.sequence = 8;
        trace.physical_transitions[0].stamp.event_id = "event-6b".to_owned();
        trace.terminal.as_mut().unwrap().stamp.sequence = 9;
        trace.terminal.as_mut().unwrap().stamp.event_id = "event-7b".to_owned();
        let report = replay_mission(&trace);
        assert_eq!(report.disposition, ReplayDisposition::Rejected);
        assert!(has_finding(&report, "REMOTE_RESULT_APPLIED_AFTER_EXPIRY"));
    }

    #[test]
    fn actual_executed_prefix_must_match_proposal_anchor() {
        let mut trace = valid_trace();
        trace.chunks[0].observed_executed_prefix_digest = "prefix-other".to_owned();
        let report = replay_mission(&trace);
        assert_eq!(report.disposition, ReplayDisposition::Rejected);
        assert!(has_finding(&report, "EXECUTED_PREFIX_MISMATCH"));
    }

    #[test]
    fn unknown_assurance_disposition_prevents_trace_closure() {
        let mut trace = valid_trace();
        trace.assurance[0].disposition = AssuranceDisposition::Unknown;
        let report = replay_mission(&trace);
        assert_eq!(report.disposition, ReplayDisposition::Unknown);
        assert!(has_finding(&report, "APPLICATION_ASSURANCE_UNKNOWN"));
    }

    #[test]
    fn rejected_assurance_disposition_rejects_application() {
        let mut trace = valid_trace();
        trace.assurance[0].disposition = AssuranceDisposition::Rejected;
        let report = replay_mission(&trace);
        assert_eq!(report.disposition, ReplayDisposition::Rejected);
        assert!(has_finding(&report, "APPLICATION_AFTER_ASSURANCE_REJECTION"));
    }

    #[test]
    fn physical_reserve_predecessor_digest_must_chain() {
        let mut trace = valid_trace();
        trace.physical_transitions[0].predecessor_state_digest = "wrong-state".to_owned();
        let report = replay_mission(&trace);
        assert_eq!(report.disposition, ReplayDisposition::Rejected);
        assert!(has_finding(&report, "PHYSICAL_RESERVE_CHAIN_MISMATCH"));
    }

    #[test]
    fn incomplete_per_node_thermal_manifest_is_unknown() {
        let mut trace = valid_trace();
        trace.physical_transitions[0].thermal_node_ids.pop();
        let report = replay_mission(&trace);
        assert_eq!(report.disposition, ReplayDisposition::Unknown);
        assert!(has_finding(&report, "THERMAL_NODE_MANIFEST_INCOMPLETE_OR_DUPLICATE"));
    }

    #[test]
    fn unknown_recovery_topology_prevents_trace_closure() {
        let mut trace = valid_trace();
        trace.physical_transitions[0].topology = TopologyDisposition::Unknown;
        let report = replay_mission(&trace);
        assert_eq!(report.disposition, ReplayDisposition::Unknown);
        assert!(has_finding(&report, "RECOVERY_TOPOLOGY_UNKNOWN"));
    }

    #[test]
    fn missing_terminal_safe_stop_does_not_close_mission() {
        let mut trace = valid_trace();
        trace.terminal = None;
        let report = replay_mission(&trace);
        assert_eq!(report.disposition, ReplayDisposition::Unknown);
        assert!(has_finding(&report, "TERMINAL_SAFE_STOP_RECEIPT_MISSING"));
    }

    #[test]
    fn duplicate_event_sequence_is_rejected() {
        let mut trace = valid_trace();
        trace.assurance[0].stamp.sequence = trace.chunks[0].stamp.sequence;
        let report = replay_mission(&trace);
        assert_eq!(report.disposition, ReplayDisposition::Rejected);
        assert!(has_finding(&report, "DUPLICATE_EVENT_SEQUENCE"));
    }

    #[test]
    fn uncertain_same_clock_timestamp_order_does_not_invent_aoi() {
        let mut trace = valid_trace();
        trace.observations[0].stamp.timestamp = TimestampInterval {
            lower_ns: 90,
            upper_ns: 205,
        };
        let report = replay_mission(&trace);
        assert_eq!(report.disposition, ReplayDisposition::Unknown);
        assert!(has_finding(&report, "CAUSAL_TIMESTAMP_ORDER_UNCERTAIN"));
        assert!(report.aoi_bounds.is_empty());
    }

    #[test]
    fn invalid_decision_calibration_rejects_trace() {
        let mut trace = valid_trace();
        trace.decisions[0].calibration_disposition = CalibrationDisposition::Invalid;
        let report = replay_mission(&trace);
        assert_eq!(report.disposition, ReplayDisposition::Rejected);
        assert!(has_finding(&report, "DECISION_CALIBRATION_INVALID"));
    }

    #[test]
    fn terminal_state_digest_must_match_last_physical_state() {
        let mut trace = valid_trace();
        trace.terminal.as_mut().unwrap().reserve_state_digest = "wrong-terminal".to_owned();
        let report = replay_mission(&trace);
        assert_eq!(report.disposition, ReplayDisposition::Rejected);
        assert!(has_finding(&report, "TERMINAL_STATE_OR_ORDER_MISMATCH"));
    }
}
