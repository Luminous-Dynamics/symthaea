use super::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

const OBSERVATION_SUITE_ID: &str = "basalt-storage-observations";
const OBSERVATION_VERSION: u32 = 1;
const TRACE_SUITE_ID: &str = "basalt-storage-conformance-trace";
const TRACE_CLOCK_BASIS: &str = "monotonic_nanoseconds_per_runner";
const TRACE_ORDERING_RULE: &str = "per_runner_only; no cross-host wall-clock ordering";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BackendProvenance {
    pub backend_id: String,
    pub engine_name: String,
    pub engine_version: String,
    pub adapter_commit_sha: String,
    pub executable_sha256: String,
    pub configuration_sha256: String,
    pub durability_profile: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerClockWindow {
    pub runner_id: String,
    pub clock_basis: String,
    pub capture_start_ns: u64,
    pub capture_end_ns: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RawEvent {
    pub event_id: String,
    pub runner_id: String,
    pub operation_id: String,
    pub runner_sequence: u64,
    pub monotonic_ns: u64,
    pub phase: String,
    pub transaction_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sqlstate: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fault_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedOperation {
    pub op_id: String,
    pub kind: String,
    pub request: Request,
    pub happens_after: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_group_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transaction_attempt: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub read_snapshot_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decision_input_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<Scope>,
    pub runner_id: String,
    pub client_id: String,
    pub client_sequence: u64,
    pub invoked_at_ns: u64,
    pub completed_at_ns: Option<u64>,
    pub outcome: Option<String>,
    pub receipt_id: Option<String>,
    pub receipt_acceptance_revision: Option<String>,
    pub current_authority_revision: Option<String>,
    pub observed_store_revision: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedFault {
    pub fault_id: String,
    pub target_op_id: String,
    pub fault_class: String,
    pub action: String,
    pub applied: bool,
    pub observed_at_ns: u64,
    pub runner_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationEnvelope {
    pub suite_id: String,
    pub version: u32,
    pub trace_id: String,
    pub reference_trace_sha256: String,
    pub backend: BackendProvenance,
    pub raw_event_log_sha256: String,
    pub runner_clock_windows: Vec<RunnerClockWindow>,
    pub raw_events: Vec<RawEvent>,
    pub observed_operations: Vec<ObservedOperation>,
    pub observed_faults: Vec<ObservedFault>,
    pub witness_observations: Vec<WitnessObservation>,
    pub final_observation: FinalObservation,
    pub authority: Authority,
    pub initial_state: InitialState,
}

#[derive(Clone, Debug, Serialize)]
pub struct ObservationReport {
    pub report_version: u32,
    pub status: String,
    pub trace_id: String,
    pub reference_trace_sha256: String,
    pub raw_event_log_sha256: String,
    pub backend_id: Option<String>,
    pub backend_engine: Option<String>,
    pub backend_version: Option<String>,
    pub adapter_commit_sha: Option<String>,
    pub operations_observed: usize,
    pub events_observed: usize,
    pub runners_observed: usize,
    pub faults_observed: usize,
    pub retry_groups_validated: usize,
    pub pending_operation_ids: Vec<String>,
    pub incomplete_retry_groups: Vec<String>,
    pub counterexamples: Vec<String>,
    pub history_search: Option<HistorySearchReport>,
    pub claim_ceiling: String,
}

#[derive(Clone, Debug)]
struct ObservationValidation {
    computed_raw_event_log_sha256: String,
    runner_count: usize,
    pending_operation_ids: Vec<String>,
    incomplete_retry_groups: Vec<String>,
    retry_groups_validated: usize,
}

fn valid_sha256(value: &str) -> bool {
    value
        .strip_prefix("sha256:")
        .is_some_and(|hex| {
            hex.len() == 64
                && hex
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
}

fn sha256_prefixed_bytes(bytes: &[u8]) -> String {
    format!("sha256:{}", hex_lower(&Sha256::digest(bytes)))
}

fn valid_git_sha(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn event_is_transactional(phase: &str) -> bool {
    matches!(
        phase,
        "transaction_begin"
            | "read_state"
            | "decision"
            | "write_source"
            | "write_log"
            | "write_receipt"
            | "write_policy"
            | "commit"
            | "abort"
    )
}

fn valid_event_phase(phase: &str) -> bool {
    matches!(
        phase,
        "invoke"
            | "transaction_begin"
            | "read_state"
            | "decision"
            | "write_source"
            | "write_log"
            | "write_receipt"
            | "write_policy"
            | "commit"
            | "abort"
            | "response"
            | "receipt_lookup"
            | "witness_read"
            | "fault"
    )
}

fn valid_fault_class(class: &str) -> bool {
    matches!(
        class,
        "process_crash"
            | "power_loss_image"
            | "io_error"
            | "response_loss"
            | "network_partition"
            | "network_delay"
            | "serialization_failure"
            | "member_restart"
            | "snapshot_restore"
            | "witness_unavailable"
    )
}

/// Normative event-log serialization: sort by runner identity and runner-local
/// sequence, then serialize the typed event array as RFC 8785 JCS. Cross-runner
/// monotonic clocks are never compared or used to order events.
pub fn canonical_raw_event_log_bytes(events: &[RawEvent]) -> Result<Vec<u8>, String> {
    let mut canonical_order = events.to_vec();
    canonical_order.sort_by(|left, right| {
        left.runner_id
            .cmp(&right.runner_id)
            .then(left.runner_sequence.cmp(&right.runner_sequence))
            .then(left.event_id.cmp(&right.event_id))
    });
    serde_json_canonicalizer::to_vec(&canonical_order)
        .map_err(|error| format!("cannot canonicalize raw event log as RFC 8785 JCS: {error}"))
}

pub fn raw_event_log_digest(events: &[RawEvent]) -> Result<String, String> {
    Ok(sha256_prefixed_bytes(&canonical_raw_event_log_bytes(events)?))
}

fn invalid_observation_report(
    trace_id: String,
    reference_sha256: String,
    raw_event_sha256: String,
    operations: usize,
    events: usize,
    runners: usize,
    faults: usize,
    counterexamples: Vec<String>,
) -> ObservationReport {
    ObservationReport {
        report_version: 1,
        status: "invalid_evidence".to_string(),
        trace_id,
        reference_trace_sha256: reference_sha256,
        raw_event_log_sha256: raw_event_sha256,
        backend_id: None,
        backend_engine: None,
        backend_version: None,
        adapter_commit_sha: None,
        operations_observed: operations,
        events_observed: events,
        runners_observed: runners,
        faults_observed: faults,
        retry_groups_validated: 0,
        pending_operation_ids: Vec::new(),
        incomplete_retry_groups: Vec::new(),
        counterexamples,
        history_search: None,
        claim_ceiling: "Observation evidence failed strict envelope, provenance, digest, clock, event, fault or transaction validation. No consistency or freshness claim is made.".to_string(),
    }
}

fn validate_observation_envelope(
    observation: &ObservationEnvelope,
    expected_reference_trace_sha256: &str,
) -> Result<ObservationValidation, Vec<String>> {
    let mut errors = Vec::new();

    if observation.suite_id != OBSERVATION_SUITE_ID
        || observation.version != OBSERVATION_VERSION
    {
        errors.push("unsupported observation suite_id/version".to_string());
    }
    if observation.trace_id.is_empty()
        || observation.authority.authority_id.is_empty()
        || observation.authority.incarnation_id.is_empty()
    {
        errors.push("observation or authority identity is empty".to_string());
    }
    if !valid_sha256(&expected_reference_trace_sha256) {
        errors.push("caller-supplied reference-trace pin is malformed".to_string());
    }
    if observation.reference_trace_sha256 != expected_reference_trace_sha256 {
        errors.push("observation reference_trace_sha256 does not match the caller pin".to_string());
    }

    let backend = &observation.backend;
    if backend.backend_id.is_empty()
        || backend.engine_name.is_empty()
        || backend.engine_version.is_empty()
        || backend.durability_profile.is_empty()
        || !valid_git_sha(&backend.adapter_commit_sha)
        || !valid_sha256(&backend.executable_sha256)
        || !valid_sha256(&backend.configuration_sha256)
    {
        errors.push("backend provenance is incomplete or contains malformed digests".to_string());
    }

    let mut windows = BTreeMap::<String, &RunnerClockWindow>::new();
    for window in &observation.runner_clock_windows {
        if window.runner_id.is_empty()
            || window.clock_basis != "monotonic_nanoseconds"
            || window.capture_start_ns > window.capture_end_ns
        {
            errors.push(format!(
                "runner clock window {} has invalid identity, clock basis or bounds",
                window.runner_id
            ));
        }
        if windows.insert(window.runner_id.clone(), window).is_some() {
            errors.push(format!(
                "duplicate runner clock window {}",
                window.runner_id
            ));
        }
    }
    if windows.is_empty() {
        errors.push("observation has no runner clock windows".to_string());
    }

    let mut operations = HashMap::<String, &ObservedOperation>::new();
    let mut pending_operation_ids = Vec::new();
    for operation in &observation.observed_operations {
        if operation.op_id.is_empty()
            || operation.runner_id.is_empty()
            || operation.client_id.is_empty()
        {
            errors.push(format!(
                "operation {} has empty operation/runner/client identity",
                operation.op_id
            ));
        }
        if !matches!(
            operation.kind.as_str(),
            "advance" | "receipt_lookup" | "restore_verify" | "policy_publish"
        ) {
            errors.push(format!(
                "{} has unsupported observed operation kind {}",
                operation.op_id, operation.kind
            ));
        }
        if let Some(window) = windows.get(&operation.runner_id) {
            if operation.invoked_at_ns < window.capture_start_ns
                || operation.invoked_at_ns > window.capture_end_ns
            {
                errors.push(format!(
                    "{} invocation is outside runner {} capture window",
                    operation.op_id, operation.runner_id
                ));
            }
            if let Some(completed) = operation.completed_at_ns {
                if completed < operation.invoked_at_ns
                    || completed < window.capture_start_ns
                    || completed > window.capture_end_ns
                {
                    errors.push(format!(
                        "{} completion is outside its valid runner interval",
                        operation.op_id
                    ));
                }
            }
        } else {
            errors.push(format!(
                "{} references runner {} without a clock window",
                operation.op_id, operation.runner_id
            ));
        }

        match (&operation.completed_at_ns, &operation.outcome) {
            (None, None) => pending_operation_ids.push(operation.op_id.clone()),
            (Some(_), Some(_)) => {}
            _ => errors.push(format!(
                "{} must have both completed_at_ns and outcome, or neither",
                operation.op_id
            )),
        }

        if operation.kind == "advance" {
            if let Some(scope) = &operation.scope {
                if scope.source_id.is_empty() || scope.log_origin_id.is_empty() {
                    errors.push(format!("{} has an empty advance scope", operation.op_id));
                }
            } else {
                errors.push(format!(
                    "{} is missing required source/log scope",
                    operation.op_id
                ));
            }
        }
        if operations
            .insert(operation.op_id.clone(), operation)
            .is_some()
        {
            errors.push(format!("duplicate observed operation ID {}", operation.op_id));
        }
    }

    let mut event_ids = HashSet::<String>::new();
    let mut runner_sequences = HashSet::<(String, u64)>::new();
    let mut events_by_runner = BTreeMap::<String, Vec<&RawEvent>>::new();
    let mut events_by_operation = HashMap::<String, Vec<&RawEvent>>::new();
    let mut events_by_fault = HashMap::<String, Vec<&RawEvent>>::new();

    for event in &observation.raw_events {
        if event.event_id.is_empty() || !event_ids.insert(event.event_id.clone()) {
            errors.push(format!("raw event ID is empty or duplicated: {}", event.event_id));
        }
        if !valid_event_phase(&event.phase) {
            errors.push(format!(
                "raw event {} has unsupported phase {}",
                event.event_id, event.phase
            ));
        }
        if !runner_sequences.insert((event.runner_id.clone(), event.runner_sequence)) {
            errors.push(format!(
                "runner {} repeats sequence {}",
                event.runner_id, event.runner_sequence
            ));
        }
        if let Some(window) = windows.get(&event.runner_id) {
            if event.monotonic_ns < window.capture_start_ns
                || event.monotonic_ns > window.capture_end_ns
            {
                errors.push(format!(
                    "raw event {} is outside runner {} capture window",
                    event.event_id, event.runner_id
                ));
            }
        } else {
            errors.push(format!(
                "raw event {} references runner {} without a clock window",
                event.event_id, event.runner_id
            ));
        }
        if let Some(operation) = operations.get(&event.operation_id) {
            if operation.runner_id != event.runner_id {
                errors.push(format!(
                    "raw event {} runner does not match operation {}",
                    event.event_id, event.operation_id
                ));
            }
        } else {
            errors.push(format!(
                "raw event {} references unknown operation {}",
                event.event_id, event.operation_id
            ));
        }
        if event_is_transactional(&event.phase) && event.transaction_id.is_none() {
            errors.push(format!(
                "transaction phase {} in event {} has no transaction_id",
                event.phase, event.event_id
            ));
        }
        if event
            .payload_sha256
            .as_deref()
            .is_some_and(|digest| !valid_sha256(digest))
        {
            errors.push(format!("event {} has malformed payload_sha256", event.event_id));
        }
        if event
            .sqlstate
            .as_deref()
            .is_some_and(|sqlstate| {
                sqlstate.len() != 5
                    || !sqlstate
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'A'..=b'Z').contains(&byte))
            })
        {
            errors.push(format!("event {} has malformed SQLSTATE", event.event_id));
        }
        if event.phase == "fault" {
            if let Some(fault_id) = &event.fault_id {
                events_by_fault
                    .entry(fault_id.clone())
                    .or_default()
                    .push(event);
            } else {
                errors.push(format!("fault event {} has no fault_id", event.event_id));
            }
        } else if event.fault_id.is_some() {
            errors.push(format!(
                "non-fault event {} must not carry a fault_id",
                event.event_id
            ));
        }
        events_by_runner
            .entry(event.runner_id.clone())
            .or_default()
            .push(event);
        events_by_operation
            .entry(event.operation_id.clone())
            .or_default()
            .push(event);
    }

    // Sequences and clocks are validated only inside each runner's clock domain.
    for (runner_id, runner_events) in &mut events_by_runner {
        runner_events.sort_by_key(|event| event.runner_sequence);
        for pair in runner_events.windows(2) {
            if pair[1].monotonic_ns < pair[0].monotonic_ns {
                errors.push(format!(
                    "runner {} monotonic clock regressed between sequences {} and {}",
                    runner_id, pair[0].runner_sequence, pair[1].runner_sequence
                ));
            }
        }
    }

    for operation in &observation.observed_operations {
        let events = events_by_operation
            .get(&operation.op_id)
            .cloned()
            .unwrap_or_default();
        let invocations: Vec<_> = events
            .iter()
            .filter(|event| event.phase == "invoke")
            .collect();
        if invocations.len() != 1
            || invocations.first().is_none_or(|event| {
                event.monotonic_ns != operation.invoked_at_ns
            })
        {
            errors.push(format!(
                "{} must have exactly one invoke event at invoked_at_ns",
                operation.op_id
            ));
        }

        let responses: Vec<_> = events
            .iter()
            .filter(|event| event.phase == "response")
            .collect();
        if operation.outcome.is_none() {
            if !responses.is_empty() {
                errors.push(format!(
                    "{} is pending but already has a response event",
                    operation.op_id
                ));
            }
            // A pending operation may have a commit event but no observed response.
            // Its commit/no-commit result remains unknown and the caller must stop at
            // the inconclusive boundary rather than synthesizing an abort.
            continue;
        }

        let outcome = operation.outcome.as_deref().unwrap_or("");
        if ![
            "committed",
            "duplicate",
            "compare_conflict",
            "indeterminate",
            "retryable_abort",
            "receipt_found",
            "restore_verified",
            "rejected_anti_rollback_unknown",
            "rejected_anti_rollback",
        ]
        .contains(&outcome)
        {
            errors.push(format!("{} has an unsupported observed outcome", operation.op_id));
        }
        if outcome == "indeterminate" {
            if !responses.is_empty() {
                errors.push(format!(
                    "{} is indeterminate but has a definite response event",
                    operation.op_id
                ));
            }
            if !observation.observed_faults.iter().any(|fault| {
                fault.target_op_id == operation.op_id
                    && fault.applied
                    && fault.fault_class == "response_loss"
            }) {
                errors.push(format!(
                    "{} is indeterminate without an applied response_loss fault",
                    operation.op_id
                ));
            }
        } else if responses.len() != 1
            || responses.first().is_none_or(|event| {
                Some(event.monotonic_ns) != operation.completed_at_ns
            })
        {
            errors.push(format!(
                "{} completed outcome requires exactly one response event at completed_at_ns",
                operation.op_id
            ));
        }

        match operation.kind.as_str() {
            "advance" | "policy_publish" => {
                let transaction_events: Vec<_> = events
                    .iter()
                    .filter(|event| event_is_transactional(&event.phase))
                    .collect();
                let transaction_ids: BTreeSet<_> = transaction_events
                    .iter()
                    .filter_map(|event| event.transaction_id.as_deref())
                    .collect();
                if transaction_ids.len() != 1 {
                    errors.push(format!(
                        "{} must have exactly one transaction ID across transaction phases",
                        operation.op_id
                    ));
                }
                for required in ["transaction_begin", "read_state", "decision"] {
                    if !transaction_events
                        .iter()
                        .any(|event| event.phase == required)
                    {
                        errors.push(format!(
                            "{} is missing transaction phase {}",
                            operation.op_id, required
                        ));
                    }
                }
                let commits = transaction_events
                    .iter()
                    .filter(|event| event.phase == "commit")
                    .count();
                let aborts: Vec<_> = transaction_events
                    .iter()
                    .filter(|event| event.phase == "abort")
                    .collect();
                if commits > 1 || aborts.len() > 1 || (commits > 0 && !aborts.is_empty()) {
                    errors.push(format!(
                        "{} has conflicting or repeated commit/abort terminal events",
                        operation.op_id
                    ));
                }
                if outcome == "retryable_abort" {
                    if commits != 0
                        || aborts.len() != 1
                        || aborts[0].sqlstate.as_deref() != Some("40001")
                    {
                        errors.push(format!(
                            "{} retryable_abort must end with one SQLSTATE 40001 abort and no commit",
                            operation.op_id
                        ));
                    }
                }
                if outcome == "committed" || outcome == "indeterminate" {
                    if commits != 1 || !aborts.is_empty() {
                        errors.push(format!(
                            "{} committed/indeterminate result requires exactly one commit and no abort",
                            operation.op_id
                        ));
                    }
                    if operation.kind == "advance" {
                        for required in ["write_source", "write_log", "write_receipt"] {
                            if !transaction_events
                                .iter()
                                .any(|event| event.phase == required)
                            {
                                errors.push(format!(
                                    "{} is missing atomic write phase {}",
                                    operation.op_id, required
                                ));
                            }
                        }
                    } else if !transaction_events
                        .iter()
                        .any(|event| event.phase == "write_policy")
                    {
                        errors.push(format!(
                            "{} is missing write_policy event",
                            operation.op_id
                        ));
                    }
                }
            }
            "receipt_lookup" => {
                if !events.iter().any(|event| event.phase == "receipt_lookup") {
                    errors.push(format!("{} is missing receipt_lookup event", operation.op_id));
                }
            }
            "restore_verify" => {
                if !events.iter().any(|event| event.phase == "witness_read") {
                    errors.push(format!("{} is missing witness_read event", operation.op_id));
                }
            }
            _ => {}
        }
    }

    let mut fault_ids = HashSet::new();
    for fault in &observation.observed_faults {
        if fault.fault_id.is_empty()
            || !fault_ids.insert(fault.fault_id.as_str())
            || !valid_fault_class(&fault.fault_class)
            || fault.action.is_empty()
        {
            errors.push(format!("fault {} has invalid identity/class/action", fault.fault_id));
        }
        if !operations.contains_key(&fault.target_op_id) {
            errors.push(format!(
                "fault {} targets unknown operation {}",
                fault.fault_id, fault.target_op_id
            ));
        }
        let matching_events = events_by_fault
            .get(&fault.fault_id)
            .cloned()
            .unwrap_or_default();
        if fault.applied {
            if matching_events.len() != 1
                || matching_events.first().is_none_or(|event| {
                    event.operation_id != fault.target_op_id
                        || event.runner_id != fault.runner_id
                        || event.monotonic_ns != fault.observed_at_ns
                })
            {
                errors.push(format!(
                    "applied fault {} must have exactly one matching raw fault event",
                    fault.fault_id
                ));
            }
        } else if !matching_events.is_empty() {
            errors.push(format!(
                "unapplied fault {} appears in the raw event log",
                fault.fault_id
            ));
        }
    }
    for fault_id in events_by_fault.keys() {
        if !fault_ids.contains(fault_id.as_str()) {
            errors.push(format!("raw fault event references unknown fault ID {fault_id}"));
        }
    }

    let mut retry_groups = BTreeMap::<String, Vec<&ObservedOperation>>::new();
    for operation in &observation.observed_operations {
        if let Some(group_id) = &operation.retry_group_id {
            retry_groups
                .entry(group_id.clone())
                .or_default()
                .push(operation);
        }
    }
    let mut incomplete_retry_groups = Vec::new();
    let mut retry_groups_validated = 0usize;
    for (group_id, mut attempts) in retry_groups {
        let errors_before_group = errors.len();
        attempts.sort_by_key(|operation| operation.transaction_attempt.unwrap_or(u64::MAX));
        if attempts.len() < 2 {
            incomplete_retry_groups.push(group_id.clone());
            continue;
        }
        let first = attempts[0];
        let mut seen_attempts = BTreeSet::new();
        let mut seen_transaction_ids = BTreeSet::new();
        for (position, attempt) in attempts.iter().enumerate() {
            let Some(attempt_number) = attempt.transaction_attempt else {
                errors.push(format!(
                    "retry group {} operation {} has no transaction_attempt",
                    group_id, attempt.op_id
                ));
                continue;
            };
            if attempt_number != position as u64 + 1 || !seen_attempts.insert(attempt_number) {
                errors.push(format!(
                    "retry group {} attempt numbers must be unique and contiguous from one",
                    group_id
                ));
            }
            if attempt.request != first.request || attempt.scope != first.scope {
                errors.push(format!(
                    "retry group {} changed request intent or source/log scope",
                    group_id
                ));
            }
            let Some(read_snapshot) = attempt.read_snapshot_sha256.as_deref() else {
                errors.push(format!(
                    "retry group {} attempt {} has no read snapshot digest",
                    group_id, attempt_number
                ));
                continue;
            };
            let Some(decision_input) = attempt.decision_input_sha256.as_deref() else {
                errors.push(format!(
                    "retry group {} attempt {} has no decision input digest",
                    group_id, attempt_number
                ));
                continue;
            };
            if !valid_sha256(read_snapshot) || !valid_sha256(decision_input) {
                errors.push(format!(
                    "retry group {} attempt {} has malformed read/decision digest",
                    group_id, attempt_number
                ));
            }
            let events = events_by_operation
                .get(&attempt.op_id)
                .cloned()
                .unwrap_or_default();
            let read_events: Vec<_> = events
                .iter()
                .filter(|event| event.phase == "read_state")
                .collect();
            let decision_events: Vec<_> = events
                .iter()
                .filter(|event| event.phase == "decision")
                .collect();
            if !read_events
                .iter()
                .any(|event| event.payload_sha256.as_deref() == Some(read_snapshot))
                || !decision_events
                    .iter()
                    .any(|event| event.payload_sha256.as_deref() == Some(decision_input))
            {
                errors.push(format!(
                    "retry group {} attempt {} read/decision fingerprints do not match raw events",
                    group_id, attempt_number
                ));
            }
            let tx_ids: BTreeSet<_> = events
                .iter()
                .filter(|event| event_is_transactional(&event.phase))
                .filter_map(|event| event.transaction_id.as_deref())
                .collect();
            if tx_ids.len() != 1 {
                errors.push(format!(
                    "retry group {} attempt {} must use exactly one fresh transaction ID",
                    group_id, attempt_number
                ));
            } else if let Some(tx_id) = tx_ids.first() {
                if !seen_transaction_ids.insert(*tx_id) {
                    errors.push(format!(
                        "retry group {} reused transaction ID {} across attempts",
                        group_id, tx_id
                    ));
                }
            }
            if position + 1 < attempts.len()
                && attempt.outcome.as_deref() != Some("retryable_abort")
            {
                errors.push(format!(
                    "retry group {} non-final attempt {} must be retryable_abort",
                    group_id, attempt_number
                ));
            }
        }
        if errors.len() == errors_before_group {
            retry_groups_validated += 1;
        }
    }

    if observation.raw_events.is_empty() {
        errors.push("observation contains no raw events".to_string());
    }
    if !valid_sha256(&observation.raw_event_log_sha256) {
        errors.push("raw_event_log_sha256 is malformed".to_string());
    }

    let computed_digest = match raw_event_log_digest(&observation.raw_events) {
        Ok(digest) => digest,
        Err(error) => {
            errors.push(error);
            String::new()
        }
    };
    if computed_digest != observation.raw_event_log_sha256 {
        errors.push(format!(
            "raw_event_log_sha256 mismatch: declared {}, computed {}",
            observation.raw_event_log_sha256, computed_digest
        ));
    }

    if errors.is_empty() {
        Ok(ObservationValidation {
            computed_raw_event_log_sha256: computed_digest,
            runner_count: windows.len(),
            pending_operation_ids,
            incomplete_retry_groups,
            retry_groups_validated,
        })
    } else {
        Err(errors)
    }
}

fn observation_to_trace(observation: &ObservationEnvelope) -> Trace {
    let operations = observation
        .observed_operations
        .iter()
        .map(|observed| Operation {
            op_id: observed.op_id.clone(),
            runner_id: observed.runner_id.clone(),
            client_id: observed.client_id.clone(),
            client_sequence: observed.client_sequence,
            retry_group_id: observed.retry_group_id.clone(),
            transaction_attempt: observed.transaction_attempt,
            kind: observed.kind.clone(),
            scope: observed.scope.clone().unwrap_or_else(|| Scope {
                source_id: "scope:not-applicable".to_string(),
                log_origin_id: "scope:not-applicable".to_string(),
            }),
            invoked_at_ns: observed.invoked_at_ns,
            completed_at_ns: observed.completed_at_ns.unwrap_or(observed.invoked_at_ns),
            happens_after: observed.happens_after.clone(),
            request: observed.request.clone(),
            expected_result: ExpectedResult {
                outcome: observed.outcome.clone().unwrap_or_else(|| "compare_conflict".to_string()),
                receipt_id: observed.receipt_id.clone(),
                receipt_acceptance_revision: observed.receipt_acceptance_revision.clone(),
                current_authority_revision: observed.current_authority_revision.clone(),
                observed_store_revision: observed.observed_store_revision.clone(),
                reason: None,
            },
        })
        .collect();

    Trace {
        suite_id: TRACE_SUITE_ID.to_string(),
        version: 1,
        trace_id: observation.trace_id.clone(),
        claim_ceiling: "Observed backend operations checked against the BASALT transition model. Raw event integrity, backend provenance and runner-local clocks were validated; pending operations were excluded from pass.".to_string(),
        generator: Generator {
            name: "basalt-observation-ingest".to_string(),
            version: "1".to_string(),
            seed: observation.trace_id.clone(),
            clock_basis: TRACE_CLOCK_BASIS.to_string(),
            ordering_rule: TRACE_ORDERING_RULE.to_string(),
        },
        authority: observation.authority.clone(),
        initial_state: observation.initial_state.clone(),
        operations,
        reference_linearization_order: Vec::new(),
        faults: observation
            .observed_faults
            .iter()
            .filter(|fault| fault.applied)
            .map(|fault| Fault {
                fault_id: fault.fault_id.clone(),
                target_op_id: fault.target_op_id.clone(),
                runner_id: fault.runner_id.clone(),
                at_ns: fault.observed_at_ns,
                class: fault.fault_class.clone(),
                action: fault.action.clone(),
            })
            .collect(),
        witness_observations: observation.witness_observations.clone(),
        final_observation: observation.final_observation.clone(),
        invariant_expectations: Vec::new(),
        fixture_digest_catalog: Vec::new(),
        negative_controls: Vec::new(),
    }
}

/// Verifies a backend observation envelope against an externally supplied reference-trace
/// digest. Strict parsing, provenance, event digest, runner clocks, transaction phases,
/// fault application and retry attempts are checked before any history search is attempted.
pub fn verify_backend_observation(
    observation_bytes: &[u8],
    expected_reference_trace_sha256: &str,
    search_config: &HistorySearchConfig,
) -> ObservationReport {
    let mut raw_sha256 = String::new();
    let mut trace_id = String::new();
    let parsed: ObservationEnvelope = match serde_json::from_slice(observation_bytes) {
        Ok(value) => value,
        Err(error) => {
            return invalid_observation_report(
                trace_id,
                expected_reference_trace_sha256.to_string(),
                raw_sha256,
                0,
                0,
                0,
                0,
                vec![format!("strict observation envelope parse failed: {error}")],
            );
        }
    };
    trace_id = parsed.trace_id.clone();
    raw_sha256 = parsed.raw_event_log_sha256.clone();

    let validation = match validate_observation_envelope(
        &parsed,
        expected_reference_trace_sha256,
    ) {
        Ok(value) => value,
        Err(errors) => {
            return invalid_observation_report(
                trace_id,
                parsed.reference_trace_sha256.clone(),
                raw_sha256,
                parsed.observed_operations.len(),
                parsed.raw_events.len(),
                parsed.runner_clock_windows.len(),
                parsed.observed_faults.len(),
                errors,
            );
        }
    };

    let mut report = ObservationReport {
        report_version: 1,
        status: "inconclusive".to_string(),
        trace_id: parsed.trace_id.clone(),
        reference_trace_sha256: parsed.reference_trace_sha256.clone(),
        raw_event_log_sha256: validation.computed_raw_event_log_sha256,
        backend_id: Some(parsed.backend.backend_id.clone()),
        backend_engine: Some(parsed.backend.engine_name.clone()),
        backend_version: Some(parsed.backend.engine_version.clone()),
        adapter_commit_sha: Some(parsed.backend.adapter_commit_sha.clone()),
        operations_observed: parsed.observed_operations.len(),
        events_observed: parsed.raw_events.len(),
        runners_observed: validation.runner_count,
        faults_observed: parsed.observed_faults.len(),
        retry_groups_validated: validation.retry_groups_validated,
        pending_operation_ids: validation.pending_operation_ids.clone(),
        incomplete_retry_groups: validation.incomplete_retry_groups.clone(),
        counterexamples: Vec::new(),
        history_search: None,
        claim_ceiling: "A pass means the observed operation history admits a bounded legal sequential interpretation under the BASALT transition model. It does not prove physical durability, witness freshness, or correct behavior outside the captured operation/event window.".to_string(),
    };

    if !validation.pending_operation_ids.is_empty() {
        report.counterexamples.push(
            "one or more operations have no completion/response; commit versus abort remains unknown".to_string(),
        );
        report.status = "inconclusive".to_string();
        return report;
    }
    if !validation.incomplete_retry_groups.is_empty() {
        report.counterexamples.push(
            "one or more retry groups contain fewer than two captured attempts".to_string(),
        );
        report.status = "inconclusive".to_string();
        return report;
    }

    let trace = observation_to_trace(&parsed);
    let search = verify_observed_history(&trace, search_config);
    report.status = search.status.clone();
    report.counterexamples = search.counterexamples.clone();
    report.claim_ceiling = format!(
        "{} Backend={} {} {}; adapter commit {}. The observed event log is adapter-supplied evidence, not a hardware durability proof.",
        search.claim_ceiling,
        parsed.backend.backend_id,
        parsed.backend.engine_name,
        parsed.backend.engine_version,
        parsed.backend.adapter_commit_sha
    );
    report.history_search = Some(search);
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_observation() -> (ObservationEnvelope, Vec<u8>, String) {
        let raw_trace = include_bytes!("../../../benchmarks/basalt/storage-conformance-trace.v1.json");
        let trace: Trace = serde_json::from_slice(raw_trace).unwrap();
        let witness = trace.witness_observations.last().unwrap().clone();
        let mut operation = trace
            .operations
            .iter()
            .find(|operation| operation.kind == "restore_verify")
            .unwrap()
            .clone();
        operation.op_id = "obs-restore-check".to_string();
        operation.runner_id = "runner-observer".to_string();
        operation.client_id = "client-restore".to_string();
        operation.client_sequence = 0;
        operation.invoked_at_ns = 10;
        operation.completed_at_ns = 20;
        operation.happens_after.clear();
        operation.request.restored_incarnation_id = Some(witness.incarnation_id.clone());
        operation.request.restored_authority_revision = Some(witness.authority_revision.clone());
        operation.request.restored_state_digest = Some(witness.state_digest.clone());
        operation.request.witness_sequence_seen = Some(witness.sequence);
        operation.expected_result = ExpectedResult {
            outcome: "restore_verified".to_string(),
            receipt_id: None,
            receipt_acceptance_revision: None,
            current_authority_revision: Some(witness.authority_revision.clone()),
            observed_store_revision: Some(witness.authority_revision.clone()),
            reason: None,
        };

        let final_observation = FinalObservation {
            authority_revision: witness.authority_revision.clone(),
            state_digest: witness.state_digest.clone(),
            policy_digest: trace.final_observation.policy_digest.clone(),
            source_cursors: trace.final_observation.source_cursors.clone(),
            log_cursors: trace.final_observation.log_cursors.clone(),
            accepted_receipts: Vec::new(),
        };
        let initial_state = InitialState {
            authority_revision: final_observation.authority_revision.clone(),
            state_digest: final_observation.state_digest.clone(),
            policy_digest: final_observation.policy_digest.clone(),
            source_cursors: final_observation.source_cursors.clone(),
            log_cursors: final_observation.log_cursors.clone(),
        };
        let raw_events = vec![
            RawEvent {
                event_id: "event-invoke".to_string(),
                runner_id: "runner-observer".to_string(),
                operation_id: operation.op_id.clone(),
                runner_sequence: 0,
                monotonic_ns: 10,
                phase: "invoke".to_string(),
                transaction_id: None,
                payload_sha256: None,
                sqlstate: None,
                fault_id: None,
            },
            RawEvent {
                event_id: "event-witness-read".to_string(),
                runner_id: "runner-observer".to_string(),
                operation_id: operation.op_id.clone(),
                runner_sequence: 1,
                monotonic_ns: 15,
                phase: "witness_read".to_string(),
                transaction_id: None,
                payload_sha256: Some(witness.signed_record_digest.clone()),
                sqlstate: None,
                fault_id: None,
            },
            RawEvent {
                event_id: "event-response".to_string(),
                runner_id: "runner-observer".to_string(),
                operation_id: operation.op_id.clone(),
                runner_sequence: 2,
                monotonic_ns: 20,
                phase: "response".to_string(),
                transaction_id: None,
                payload_sha256: None,
                sqlstate: None,
                fault_id: None,
            },
        ];
        let raw_event_sha = raw_event_log_digest(&raw_events).unwrap();
        let reference_sha = sha256_prefixed_bytes(raw_trace);
        let observation = ObservationEnvelope {
            suite_id: OBSERVATION_SUITE_ID.to_string(),
            version: OBSERVATION_VERSION,
            trace_id: "observed-restore-check-v1".to_string(),
            reference_trace_sha256: reference_sha.clone(),
            backend: BackendProvenance {
                backend_id: "fixture-backend".to_string(),
                engine_name: "fixture".to_string(),
                engine_version: "1.0.0".to_string(),
                adapter_commit_sha: "0123456789abcdef0123456789abcdef01234567".to_string(),
                executable_sha256: format!("sha256:{}", "11".repeat(32)),
                configuration_sha256: format!("sha256:{}", "22".repeat(32)),
                durability_profile: "test-only".to_string(),
            },
            raw_event_log_sha256: raw_event_sha,
            runner_clock_windows: vec![RunnerClockWindow {
                runner_id: "runner-observer".to_string(),
                clock_basis: "monotonic_nanoseconds".to_string(),
                capture_start_ns: 0,
                capture_end_ns: 25,
            }],
            raw_events,
            observed_operations: vec![ObservedOperation {
                op_id: operation.op_id,
                kind: operation.kind,
                request: operation.request,
                happens_after: operation.happens_after,
                retry_group_id: None,
                transaction_attempt: None,
                read_snapshot_sha256: None,
                decision_input_sha256: None,
                scope: Some(Scope {
                    source_id: "scope:read".to_string(),
                    log_origin_id: "scope:read".to_string(),
                }),
                runner_id: operation.runner_id,
                client_id: operation.client_id,
                client_sequence: operation.client_sequence,
                invoked_at_ns: operation.invoked_at_ns,
                completed_at_ns: Some(operation.completed_at_ns),
                outcome: Some(operation.expected_result.outcome),
                receipt_id: None,
                receipt_acceptance_revision: None,
                current_authority_revision: operation.expected_result.current_authority_revision,
                observed_store_revision: operation.expected_result.observed_store_revision,
            }],
            observed_faults: Vec::new(),
            witness_observations: trace.witness_observations,
            final_observation,
            authority: trace.authority,
            initial_state,
        };
        (observation, raw_trace.to_vec(), reference_sha)
    }

    fn config() -> HistorySearchConfig {
        HistorySearchConfig {
            max_search_states: 10_000,
            max_operations: 32,
        }
    }

    fn retry_observation() -> (ObservationEnvelope, String) {
        let (mut observation, raw_trace, reference_pin) = valid_observation();
        let trace: Trace = serde_json::from_slice(&raw_trace).unwrap();
        let first = trace
            .operations
            .iter()
            .find(|operation| operation.op_id == "op-serializable-attempt-1")
            .unwrap();
        let second = trace
            .operations
            .iter()
            .find(|operation| operation.op_id == "op-serializable-attempt-2")
            .unwrap();
        assert_eq!(first.request, second.request);
        assert_eq!(first.scope, second.scope);

        let read_one = format!("sha256:{}", "aa".repeat(32));
        let decision_one = format!("sha256:{}", "bb".repeat(32));
        let read_two = format!("sha256:{}", "cc".repeat(32));
        let decision_two = format!("sha256:{}", "dd".repeat(32));

        let observed_attempt = |operation: &Operation,
                                attempt: u64,
                                read_digest: &str,
                                decision_digest: &str| ObservedOperation {
            op_id: operation.op_id.clone(),
            kind: operation.kind.clone(),
            request: operation.request.clone(),
            happens_after: if attempt == 1 {
                Vec::new()
            } else {
                vec![first.op_id.clone()]
            },
            retry_group_id: Some("retry-group-postgres-40001".to_string()),
            transaction_attempt: Some(attempt),
            read_snapshot_sha256: Some(read_digest.to_string()),
            decision_input_sha256: Some(decision_digest.to_string()),
            scope: Some(operation.scope.clone()),
            runner_id: "runner-postgres".to_string(),
            client_id: operation.client_id.clone(),
            client_sequence: operation.client_sequence,
            invoked_at_ns: operation.invoked_at_ns,
            completed_at_ns: Some(operation.completed_at_ns),
            outcome: Some(operation.expected_result.outcome.clone()),
            receipt_id: operation.expected_result.receipt_id.clone(),
            receipt_acceptance_revision: operation
                .expected_result
                .receipt_acceptance_revision
                .clone(),
            current_authority_revision: operation
                .expected_result
                .current_authority_revision
                .clone(),
            observed_store_revision: None,
        };

        let mut events = Vec::new();
        let mut sequence = 0_u64;
        let mut add = |op: &Operation,
                       phase: &str,
                       time: u64,
                       tx: Option<&str>,
                       payload: Option<&str>,
                       sqlstate: Option<&str>| {
            events.push(RawEvent {
                event_id: format!("event-{}", sequence),
                runner_id: "runner-postgres".to_string(),
                operation_id: op.op_id.clone(),
                runner_sequence: sequence,
                monotonic_ns: time,
                phase: phase.to_string(),
                transaction_id: tx.map(str::to_string),
                payload_sha256: payload.map(str::to_string),
                sqlstate: sqlstate.map(str::to_string),
                fault_id: None,
            });
            sequence += 1;
        };

        add(first, "invoke", first.invoked_at_ns, None, None, None);
        add(first, "transaction_begin", 151, Some("tx-attempt-1"), None, None);
        add(
            first,
            "read_state",
            152,
            Some("tx-attempt-1"),
            Some(&read_one),
            None,
        );
        add(
            first,
            "decision",
            153,
            Some("tx-attempt-1"),
            Some(&decision_one),
            None,
        );
        add(
            first,
            "abort",
            154,
            Some("tx-attempt-1"),
            None,
            Some("40001"),
        );
        add(first, "response", first.completed_at_ns, None, None, None);

        add(second, "invoke", second.invoked_at_ns, None, None, None);
        add(second, "transaction_begin", 163, Some("tx-attempt-2"), None, None);
        add(
            second,
            "read_state",
            164,
            Some("tx-attempt-2"),
            Some(&read_two),
            None,
        );
        add(
            second,
            "decision",
            165,
            Some("tx-attempt-2"),
            Some(&decision_two),
            None,
        );
        add(
            second,
            "write_source",
            166,
            Some("tx-attempt-2"),
            None,
            None,
        );
        add(
            second,
            "write_log",
            167,
            Some("tx-attempt-2"),
            None,
            None,
        );
        add(
            second,
            "write_receipt",
            168,
            Some("tx-attempt-2"),
            None,
            None,
        );
        add(second, "commit", 170, Some("tx-attempt-2"), None, None);
        add(second, "response", second.completed_at_ns, None, None, None);

        observation.backend.engine_name = "PostgreSQL".to_string();
        observation.backend.engine_version = "18".to_string();
        observation.runner_clock_windows = vec![RunnerClockWindow {
            runner_id: "runner-postgres".to_string(),
            clock_basis: "monotonic_nanoseconds".to_string(),
            capture_start_ns: 0,
            capture_end_ns: 200,
        }];
        observation.observed_operations = vec![
            observed_attempt(first, 1, &read_one, &decision_one),
            observed_attempt(second, 2, &read_two, &decision_two),
        ];
        observation.raw_events = events;
        observation.observed_faults.clear();
        observation.raw_event_log_sha256 = raw_event_log_digest(&observation.raw_events).unwrap();
        (observation, reference_pin)
    }

    #[test]
    fn sqlstate_40001_retry_requires_two_fresh_full_transactions() {
        let (observation, reference_pin) = retry_observation();
        let validated = validate_observation_envelope(&observation, &reference_pin).unwrap();
        assert_eq!(validated.retry_groups_validated, 1);
        assert!(validated.incomplete_retry_groups.is_empty());
    }

    #[test]
    fn serialization_retry_rejects_reused_transaction_id() {
        let (mut observation, reference_pin) = retry_observation();
        for event in &mut observation.raw_events {
            if event.operation_id == "op-serializable-attempt-2"
                && event_is_transactional(&event.phase)
            {
                event.transaction_id = Some("tx-attempt-1".to_string());
            }
        }
        observation.raw_event_log_sha256 = raw_event_log_digest(&observation.raw_events).unwrap();
        let errors = validate_observation_envelope(&observation, &reference_pin).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("reused transaction ID")));
    }

    #[test]
    fn serialization_retry_rejects_missing_second_read() {
        let (mut observation, reference_pin) = retry_observation();
        observation.raw_events.retain(|event| {
            !(event.operation_id == "op-serializable-attempt-2"
                && event.phase == "read_state")
        });
        observation.raw_event_log_sha256 = raw_event_log_digest(&observation.raw_events).unwrap();
        let errors = validate_observation_envelope(&observation, &reference_pin).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("missing transaction phase read_state")));
    }

    #[test]
    fn one_captured_attempt_of_a_declared_retry_group_is_inconclusive() {
        let (mut observation, _, reference_pin) = valid_observation();
        observation.observed_operations[0].retry_group_id =
            Some("incomplete-retry-group".to_string());
        observation.observed_operations[0].transaction_attempt = Some(1);
        let bytes = serde_json::to_vec(&observation).unwrap();
        let report = verify_backend_observation(&bytes, &reference_pin, &config());
        assert_eq!(report.status, "inconclusive");
        assert_eq!(
            report.incomplete_retry_groups,
            vec!["incomplete-retry-group"]
        );
    }

    #[test]
    fn strict_backend_observation_ingestion_reaches_history_search() {
        let (observation, raw_trace, reference_pin) = valid_observation();
        let bytes = serde_json::to_vec(&observation).unwrap();
        let report = verify_backend_observation(&bytes, &reference_pin, &config());
        assert_eq!(report.status, "pass", "{:#?}", report.counterexamples);
        assert_eq!(report.operations_observed, 1);
        assert_eq!(report.events_observed, 3);
        assert!(report.history_search.is_some());
        assert_eq!(report.reference_trace_sha256, sha256_prefixed_bytes(&raw_trace));
    }

    #[test]
    fn mismatched_external_reference_pin_is_invalid_evidence() {
        let (observation, _, _) = valid_observation();
        let bytes = serde_json::to_vec(&observation).unwrap();
        let report = verify_backend_observation(
            &bytes,
            &format!("sha256:{}", "ff".repeat(32)),
            &config(),
        );
        assert_eq!(report.status, "invalid_evidence");
        assert!(report.counterexamples.iter().any(|error| error.contains("caller pin")));
    }

    #[test]
    fn raw_event_log_mutation_is_detected() {
        let (mut observation, _, reference_pin) = valid_observation();
        observation.raw_events[1].monotonic_ns += 1;
        let bytes = serde_json::to_vec(&observation).unwrap();
        let report = verify_backend_observation(&bytes, &reference_pin, &config());
        assert_eq!(report.status, "invalid_evidence");
        assert!(report.counterexamples.iter().any(|error| error.contains("raw_event_log_sha256 mismatch")));
    }

    #[test]
    fn pending_operation_is_inconclusive_not_a_synthetic_abort() {
        let (mut observation, _, reference_pin) = valid_observation();
        observation.observed_operations[0].completed_at_ns = None;
        observation.observed_operations[0].outcome = None;
        observation.raw_events.retain(|event| event.phase != "response");
        observation.raw_event_log_sha256 = raw_event_log_digest(&observation.raw_events).unwrap();
        let bytes = serde_json::to_vec(&observation).unwrap();
        let report = verify_backend_observation(&bytes, &reference_pin, &config());
        assert_eq!(report.status, "inconclusive");
        assert_eq!(report.pending_operation_ids, vec!["obs-restore-check"]);
        assert!(report.history_search.is_none());
    }

    #[test]
    fn cross_runner_clock_values_are_not_compared() {
        let (mut observation, _, reference_pin) = valid_observation();
        observation.runner_clock_windows.push(RunnerClockWindow {
            runner_id: "runner-other".to_string(),
            clock_basis: "monotonic_nanoseconds".to_string(),
            capture_start_ns: 0,
            capture_end_ns: 1,
        });
        observation.raw_event_log_sha256 = raw_event_log_digest(&observation.raw_events).unwrap();
        let bytes = serde_json::to_vec(&observation).unwrap();
        let report = verify_backend_observation(&bytes, &reference_pin, &config());
        assert_eq!(report.status, "pass", "{:#?}", report.counterexamples);
    }

    #[test]
    fn monotonic_clock_regression_within_runner_is_invalid() {
        let (mut observation, _, reference_pin) = valid_observation();
        observation.raw_events[2].monotonic_ns = 14;
        observation.raw_event_log_sha256 = raw_event_log_digest(&observation.raw_events).unwrap();
        let bytes = serde_json::to_vec(&observation).unwrap();
        let report = verify_backend_observation(&bytes, &reference_pin, &config());
        assert_eq!(report.status, "invalid_evidence");
        assert!(report.counterexamples.iter().any(|error| error.contains("clock regressed")));
    }

    #[test]
    fn strict_deserialization_rejects_unknown_fields() {
        let (observation, _, reference_pin) = valid_observation();
        let mut value: serde_json::Value =
            serde_json::from_slice(&serde_json::to_vec(&observation).unwrap()).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("adapter_says_pass".to_string(), serde_json::Value::Bool(true));
        let report = verify_backend_observation(
            &serde_json::to_vec(&value).unwrap(),
            &reference_pin,
            &config(),
        );
        assert_eq!(report.status, "invalid_evidence");
    }
}
