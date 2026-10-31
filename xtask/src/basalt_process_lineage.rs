//! Reconciles independently supervised process lineage with an untrusted observer stream.
//!
//! Process identity is scoped by host boot ID, PID namespace, run/task/attempt,
//! PID and /proc start-time ticks. PID alone is never an identity. This auditor
//! checks a submitted oracle/receipt pair; it does not capture kernel events,
//! authenticate either author, or qualify an observer profile.

use anyhow::Context;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;

const MAX_INPUT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_PROCESSES: usize = 10_000;
const MAX_EVENTS: usize = 100_000;

#[derive(Debug, Serialize)]
pub struct ProcessLineageAuditReport {
    pub report_version: u32,
    pub oracle_digest: String,
    pub receipt_digest: String,
    pub run_id: Option<String>,
    pub task_id: Option<String>,
    pub attempt_id: Option<String>,
    pub oracle_well_formed: bool,
    pub receipt_well_formed: bool,
    pub identity_context_match: bool,
    pub process_graph_reconciled: bool,
    pub event_set_reconciled: bool,
    pub event_sequence_integral: bool,
    pub event_loss_free: bool,
    pub process_tree_closed: bool,
    pub lineage_audit_passed: bool,
    pub oracle_authenticity_established: bool,
    pub observer_receipt_authenticity_established: bool,
    pub qualification_eligible: bool,
    pub counts: ProcessLineageCounts,
    pub errors: Vec<LineageError>,
}

#[derive(Debug, Default, Serialize)]
pub struct ProcessLineageCounts {
    pub oracle_processes: usize,
    pub observed_processes: usize,
    pub oracle_events: usize,
    pub observed_events: usize,
    pub pid_values_reused: usize,
    pub missing_processes: usize,
    pub unexpected_processes: usize,
    pub missing_events: usize,
    pub unexpected_events: usize,
    pub duplicate_events: usize,
    pub computed_sequence_gaps: usize,
    pub reported_dropped_events: u64,
    pub reported_overflows: u64,
    pub reported_restarts: u64,
    pub reported_unmonitored_processes: u64,
}

#[derive(Debug, Serialize)]
pub struct LineageError {
    pub code: String,
    pub path: String,
    pub message: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct ProcessIdentity {
    pub pid: u32,
    pub start_time_ticks: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "relation", content = "identity", rename_all = "snake_case")]
enum ParentRelation {
    Root,
    Parent(ProcessIdentity),
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum ProcessEventKind {
    RootStart,
    Fork,
    Clone,
    Vfork,
    Exec,
    Exit,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProcessLineageOracle {
    format_version: u32,
    suite_id: String,
    run_id: String,
    task_id: String,
    attempt_id: String,
    host_boot_id_digest: String,
    pid_namespace_inode: u64,
    capture_start_monotonic_ns: u64,
    capture_end_monotonic_ns: u64,
    root_process: ProcessIdentity,
    expected_tree_closed: bool,
    harness_binary_digest: String,
    supervisor_binary_digest: String,
    processes: Vec<OracleProcess>,
    events: Vec<OracleEvent>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleProcess {
    identity: ProcessIdentity,
    parent: ParentRelation,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleEvent {
    kind: ProcessEventKind,
    process: ProcessIdentity,
    observed_at_monotonic_ns: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProcessLineageReceipt {
    format_version: u32,
    suite_id: String,
    run_id: String,
    task_id: String,
    attempt_id: String,
    host_boot_id_digest: String,
    pid_namespace_inode: u64,
    capture_start_monotonic_ns: u64,
    capture_end_monotonic_ns: u64,
    root_process: ProcessIdentity,
    observer_binary_digest: String,
    observer_configuration_digest: String,
    reported_tree_closed: bool,
    processes: Vec<ObservedProcess>,
    events: Vec<ObservedEvent>,
    health: ObserverHealth,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservedProcess {
    identity: ProcessIdentity,
    parent: ParentRelation,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservedEvent {
    source_id: String,
    sequence: u64,
    kind: ProcessEventKind,
    process: ProcessIdentity,
    observed_at_monotonic_ns: u64,
    evidence_digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObserverHealth {
    event_count: u64,
    dropped_event_count: u64,
    ring_buffer_overflow_count: u64,
    observer_restart_count: u64,
    unmonitored_process_count: u64,
    sequence_gap_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct EventKey {
    kind: ProcessEventKind,
    process: ProcessIdentity,
}

pub fn audit_files(oracle_path: &Path, receipt_path: &Path) -> anyhow::Result<ProcessLineageAuditReport> {
    let oracle_bytes = read_bounded(oracle_path)?;
    let receipt_bytes = read_bounded(receipt_path)?;
    Ok(audit_bytes(&oracle_bytes, &receipt_bytes))
}

pub fn audit_bytes(oracle_bytes: &[u8], receipt_bytes: &[u8]) -> ProcessLineageAuditReport {
    let mut report = ProcessLineageAuditReport {
        report_version: 1,
        oracle_digest: sha256_tagged(oracle_bytes),
        receipt_digest: sha256_tagged(receipt_bytes),
        run_id: None,
        task_id: None,
        attempt_id: None,
        oracle_well_formed: false,
        receipt_well_formed: false,
        identity_context_match: false,
        process_graph_reconciled: false,
        event_set_reconciled: false,
        event_sequence_integral: false,
        event_loss_free: false,
        process_tree_closed: false,
        lineage_audit_passed: false,
        oracle_authenticity_established: false,
        observer_receipt_authenticity_established: false,
        qualification_eligible: false,
        counts: ProcessLineageCounts::default(),
        errors: Vec::new(),
    };

    if oracle_bytes.len() as u64 > MAX_INPUT_BYTES || receipt_bytes.len() as u64 > MAX_INPUT_BYTES {
        error(&mut report, "E_INPUT_TOO_LARGE", "$", "oracle and receipt are each limited to 16 MiB");
        return report;
    }

    let oracle: ProcessLineageOracle = match serde_json::from_slice(oracle_bytes) {
        Ok(value) => value,
        Err(parse_error) => {
            error(&mut report, "E_ORACLE_JSON", "oracle", &format!("invalid or non-strict process-lineage oracle: {parse_error}"));
            return report;
        }
    };
    let receipt: ProcessLineageReceipt = match serde_json::from_slice(receipt_bytes) {
        Ok(value) => value,
        Err(parse_error) => {
            error(&mut report, "E_RECEIPT_JSON", "receipt", &format!("invalid or non-strict process-lineage receipt: {parse_error}"));
            return report;
        }
    };

    report.run_id = Some(oracle.run_id.clone());
    report.task_id = Some(oracle.task_id.clone());
    report.attempt_id = Some(oracle.attempt_id.clone());
    report.counts.oracle_processes = oracle.processes.len();
    report.counts.observed_processes = receipt.processes.len();
    report.counts.oracle_events = oracle.events.len();
    report.counts.observed_events = receipt.events.len();
    report.counts.reported_dropped_events = receipt.health.dropped_event_count;
    report.counts.reported_overflows = receipt.health.ring_buffer_overflow_count;
    report.counts.reported_restarts = receipt.health.observer_restart_count;
    report.counts.reported_unmonitored_processes = receipt.health.unmonitored_process_count;

    if oracle.format_version != 1 || oracle.suite_id != "basalt-process-lineage-oracle-v1" {
        error(&mut report, "E_ORACLE_VERSION", "oracle", "unsupported oracle format_version or suite_id");
    }
    if receipt.format_version != 1 || receipt.suite_id != "basalt-process-lineage-receipt-v1" {
        error(&mut report, "E_RECEIPT_VERSION", "receipt", "unsupported receipt format_version or suite_id");
    }
    if oracle.run_id.is_empty() || oracle.run_id.len() > 128 || oracle.task_id.is_empty() || oracle.task_id.len() > 128 || oracle.attempt_id.is_empty() || oracle.attempt_id.len() > 128 {
        error(&mut report, "E_ORACLE_SCOPE", "oracle", "run_id, task_id and attempt_id must be non-empty and at most 128 bytes");
    }
    if oracle.host_boot_id_digest != receipt.host_boot_id_digest
        || oracle.pid_namespace_inode != receipt.pid_namespace_inode
        || oracle.run_id != receipt.run_id
        || oracle.task_id != receipt.task_id
        || oracle.attempt_id != receipt.attempt_id
        || oracle.root_process != receipt.root_process
    {
        error(&mut report, "E_IDENTITY_CONTEXT_MISMATCH", "receipt", "boot identity, PID namespace, run/task/attempt or root process differs from the independent oracle");
    } else {
        report.identity_context_match = true;
    }
    if !valid_digest(&oracle.host_boot_id_digest)
        || !valid_digest(&oracle.harness_binary_digest)
        || !valid_digest(&oracle.supervisor_binary_digest)
        || !valid_digest(&receipt.host_boot_id_digest)
        || !valid_digest(&receipt.observer_binary_digest)
        || !valid_digest(&receipt.observer_configuration_digest)
    {
        error(&mut report, "E_DIGEST_FORMAT", "$", "identity and executable commitments must be lowercase sha256 digests");
    }
    if oracle.pid_namespace_inode == 0 || receipt.pid_namespace_inode == 0 {
        error(&mut report, "E_PID_NAMESPACE", "pid_namespace_inode", "PID namespace inode must be non-zero");
    }
    if oracle.capture_start_monotonic_ns >= oracle.capture_end_monotonic_ns
        || receipt.capture_start_monotonic_ns >= receipt.capture_end_monotonic_ns
    {
        error(&mut report, "E_CAPTURE_INTERVAL", "capture", "capture interval must be strictly increasing");
    }
    if receipt.capture_start_monotonic_ns > oracle.capture_start_monotonic_ns
        || receipt.capture_end_monotonic_ns < oracle.capture_end_monotonic_ns
    {
        error(&mut report, "E_CAPTURE_WINDOW_NOT_COVERED", "receipt.capture", "observer interval must cover the independent oracle interval");
    }
    if oracle.processes.is_empty() || oracle.processes.len() > MAX_PROCESSES
        || receipt.processes.len() > MAX_PROCESSES
    {
        error(&mut report, "E_PROCESS_COUNT", "processes", "process list is empty or exceeds the 10,000-process limit");
        return report;
    }
    if oracle.events.is_empty() || oracle.events.len() > MAX_EVENTS || receipt.events.len() > MAX_EVENTS {
        error(&mut report, "E_EVENT_COUNT", "events", "event list is empty or exceeds the 100,000-event limit");
        return report;
    }

    let before_oracle_validation = report.errors.len();
    let oracle_processes = validate_oracle_processes(&oracle, &mut report);
    validate_oracle_events(&oracle, &oracle_processes, &mut report);
    report.oracle_well_formed = report.errors.len() == before_oracle_validation;

    let before_receipt_validation = report.errors.len();
    let observed_processes = validate_observed_processes(&receipt, &mut report);
    let (observed_event_keys, sequence_integral, computed_gaps) = validate_observed_events(&receipt, &observed_processes, &mut report);
    report.counts.computed_sequence_gaps = computed_gaps;
    report.event_sequence_integral = sequence_integral;
    report.receipt_well_formed = report.errors.len() == before_receipt_validation;

    report.counts.pid_values_reused = count_reused_pids(oracle_processes.keys().map(|identity| identity.pid));
    let oracle_relations: BTreeMap<_, _> = oracle_processes.iter().map(|(identity, parent)| (identity.clone(), parent.clone())).collect();
    let observed_relations: BTreeMap<_, _> = observed_processes.iter().map(|(identity, parent)| (identity.clone(), parent.clone())).collect();

    for identity in oracle_relations.keys() {
        if !observed_relations.contains_key(identity) {
            report.counts.missing_processes += 1;
        }
    }
    for identity in observed_relations.keys() {
        if !oracle_relations.contains_key(identity) {
            report.counts.unexpected_processes += 1;
        }
    }
    report.process_graph_reconciled = oracle_relations == observed_relations
        && report.counts.missing_processes == 0
        && report.counts.unexpected_processes == 0;

    let oracle_event_keys: BTreeSet<EventKey> = oracle.events.iter().map(event_key_oracle).collect();
    let observed_event_set: BTreeSet<EventKey> = observed_event_keys.iter().cloned().collect();
    let observed_key_counts = count_event_keys(&observed_event_keys);

    for key in &oracle_event_keys {
        if !observed_key_counts.contains_key(key) {
            report.counts.missing_events += 1;
        }
    }
    for key in observed_key_counts.keys() {
        if !oracle_event_keys.contains(key) {
            report.counts.unexpected_events += 1;
        }
    }
    report.counts.duplicate_events = observed_key_counts.values().map(|count| count.saturating_sub(1)).sum();
    report.event_set_reconciled = oracle_event_keys == observed_event_set
        && report.counts.missing_events == 0
        && report.counts.unexpected_events == 0
        && report.counts.duplicate_events == 0;

    report.event_loss_free = receipt.health.event_count == receipt.events.len() as u64
        && receipt.health.dropped_event_count == 0
        && receipt.health.ring_buffer_overflow_count == 0
        && receipt.health.observer_restart_count == 0
        && receipt.health.unmonitored_process_count == 0
        && receipt.health.sequence_gap_count == 0
        && computed_gaps == 0;

    let required_closed = oracle.expected_tree_closed
        && receipt.reported_tree_closed
        && all_processes_have_exit_events(&oracle.processes, &oracle.events)
        && all_processes_have_exit_observed(&receipt.processes, &receipt.events);
    report.process_tree_closed = required_closed;

    report.lineage_audit_passed = report.oracle_well_formed
        && report.receipt_well_formed
        && report.identity_context_match
        && report.process_graph_reconciled
        && report.event_set_reconciled
        && report.event_sequence_integral
        && report.event_loss_free
        && report.process_tree_closed
        && report.errors.is_empty();
    report
}

fn validate_oracle_processes(oracle: &ProcessLineageOracle, report: &mut ProcessLineageAuditReport) -> BTreeMap<ProcessIdentity, Option<ProcessIdentity>> {
    let mut nodes = BTreeMap::new();
    let mut pid_start = BTreeMap::<u32, BTreeSet<u64>>::new();
    for (index, process) in oracle.processes.iter().enumerate() {
        let identity = &process.identity;
        if identity.pid == 0 || identity.start_time_ticks == 0 {
            error(report, "E_PROCESS_IDENTITY", &format!("oracle.processes[{index}]"), "PID and start_time_ticks must be non-zero");
        }
        pid_start.entry(identity.pid).or_default().insert(identity.start_time_ticks);
        if nodes.insert(identity.clone(), parent_identity(&process.parent)).is_some() {
            error(report, "E_PROCESS_DUPLICATE", &format!("oracle.processes[{index}]"), "process instance identity appears more than once");
        }
    }
    validate_process_graph(&nodes, &oracle.root_process, "oracle.processes", report);
    let _ = pid_start;
    nodes
}

fn validate_observed_processes(receipt: &ProcessLineageReceipt, report: &mut ProcessLineageAuditReport) -> BTreeMap<ProcessIdentity, Option<ProcessIdentity>> {
    let mut nodes = BTreeMap::new();
    for (index, process) in receipt.processes.iter().enumerate() {
        let identity = &process.identity;
        if identity.pid == 0 || identity.start_time_ticks == 0 {
            error(report, "E_PROCESS_IDENTITY", &format!("receipt.processes[{index}]"), "PID and start_time_ticks must be non-zero");
        }
        if nodes.insert(identity.clone(), parent_identity(&process.parent)).is_some() {
            error(report, "E_PROCESS_DUPLICATE", &format!("receipt.processes[{index}]"), "observed process instance identity appears more than once");
        }
    }
    validate_process_graph(&nodes, &receipt.root_process, "receipt.processes", report);
    nodes
}

fn validate_process_graph(
    nodes: &BTreeMap<ProcessIdentity, Option<ProcessIdentity>>,
    root: &ProcessIdentity,
    path: &str,
    report: &mut ProcessLineageAuditReport,
) {
    if !nodes.contains_key(root) {
        error(report, "E_ROOT_PROCESS_MISSING", path, "root process identity is absent from process inventory");
        return;
    }
    let roots = nodes.values().filter(|parent| parent.is_none()).count();
    if roots != 1 || nodes.get(root).is_some_and(Option::is_some) {
        error(report, "E_ROOT_COUNT", path, "process graph must contain exactly one root and it must be root_process");
    }
    for (identity, parent) in nodes {
        if let Some(parent_identity) = parent {
            if identity == parent_identity || !nodes.contains_key(parent_identity) {
                error(report, "E_PARENT_REFERENCE", path, "each non-root process must reference a distinct process in this same lineage");
            }
        }
    }
    // Parent chains are bounded by MAX_PROCESSES and checked without recursion.
    for identity in nodes.keys() {
        let mut seen = BTreeSet::new();
        let mut cursor = Some(identity);
        let mut steps = 0usize;
        while let Some(current) = cursor {
            if !seen.insert(current.clone()) {
                error(report, "E_PROCESS_CYCLE", path, "process parent graph contains a cycle");
                break;
            }
            steps += 1;
            if steps > nodes.len() {
                error(report, "E_PROCESS_DEPTH", path, "process parent chain exceeds process inventory size");
                break;
            }
            cursor = nodes.get(current).and_then(Option::as_ref);
        }
    }
}

fn validate_oracle_events(
    oracle: &ProcessLineageOracle,
    processes: &BTreeMap<ProcessIdentity, Option<ProcessIdentity>>,
    report: &mut ProcessLineageAuditReport,
) {
    let process_keys: BTreeSet<_> = processes.keys().cloned().collect();
    let mut counts = BTreeMap::<EventKey, usize>::new();
    let mut times = BTreeMap::<(ProcessIdentity, ProcessEventKind), Vec<u64>>::new();
    for (index, event) in oracle.events.iter().enumerate() {
        if !process_keys.contains(&event.process) {
            error(report, "E_EVENT_UNKNOWN_PROCESS", &format!("oracle.events[{index}]"), "oracle event references a process absent from its process graph");
        }
        if event.observed_at_monotonic_ns < oracle.capture_start_monotonic_ns || event.observed_at_monotonic_ns > oracle.capture_end_monotonic_ns {
            error(report, "E_EVENT_OUTSIDE_CAPTURE", &format!("oracle.events[{index}]"), "oracle event timestamp lies outside the declared capture interval");
        }
        *counts.entry(event_key_oracle(event)).or_default() += 1;
        times.entry((event.process.clone(), event.kind)).or_default().push(event.observed_at_monotonic_ns);
    }
    for (key, count) in &counts {
        if *count != 1 {
            error(report, "E_ORACLE_EVENT_DUPLICATE", "oracle.events", &format!("expected event {:?} for PID {} start {} appears {count} times", key.kind, key.process.pid, key.process.start_time_ticks));
        }
    }
    for identity in processes.keys() {
        let birth_kinds = if *identity == oracle.root_process {
            vec![ProcessEventKind::RootStart]
        } else {
            vec![ProcessEventKind::Fork, ProcessEventKind::Clone, ProcessEventKind::Vfork]
        };
        let birth_count = counts.iter().filter(|(key, count)| key.process == *identity && birth_kinds.contains(&key.kind) && **count == 1).count();
        if birth_count != 1 {
            error(report, "E_ORACLE_BIRTH_EVENT", "oracle.events", "each process must have exactly one root_start or child-creation event");
        }
        let exit_count = counts.get(&EventKey { kind: ProcessEventKind::Exit, process: identity.clone() }).copied().unwrap_or(0);
        if oracle.expected_tree_closed && exit_count != 1 {
            error(report, "E_ORACLE_EXIT_EVENT", "oracle.events", "a closed process tree requires exactly one exit event per process instance");
        }
        let mut lifecycle = Vec::<(u64, u8)>::new();
        for (key, event_count) in &counts {
            if key.process != *identity || *event_count != 1 { continue; }
            if let Some(values) = times.get(&(identity.clone(), key.kind)) {
                let rank = match key.kind {
                    ProcessEventKind::RootStart | ProcessEventKind::Fork | ProcessEventKind::Clone | ProcessEventKind::Vfork => 0,
                    ProcessEventKind::Exec => 1,
                    ProcessEventKind::Exit => 2,
                };
                lifecycle.extend(values.iter().map(|time| (*time, rank)));
            }
        }
        lifecycle.sort();
        if lifecycle.windows(2).any(|pair| pair[0].1 > pair[1].1) {
            error(report, "E_ORACLE_LIFECYCLE_ORDER", "oracle.events", "process lifecycle order must be creation, then exec, then exit");
        }
    }
    // Every child must be born after its parent and, for closed trees, exit before its parent.
    for (child, parent) in processes {
        if let Some(parent) = parent {
            let child_birth = earliest_birth_time(&oracle.events, child);
            let parent_birth = earliest_birth_time(&oracle.events, parent);
            if child_birth.zip(parent_birth).is_none_or(|(child_time,parent_time)| child_time <= parent_time) {
                error(report, "E_ORACLE_PARENT_BIRTH_ORDER", "oracle.events", "child creation must follow parent creation");
            }
            if oracle.expected_tree_closed {
                let child_exit = event_time(&oracle.events, child, ProcessEventKind::Exit);
                let parent_exit = event_time(&oracle.events, parent, ProcessEventKind::Exit);
                if child_exit.zip(parent_exit).is_none_or(|(child_time,parent_time)| child_time > parent_time) {
                    error(report, "E_ORACLE_CHILD_EXIT_ORDER", "oracle.events", "a closed child must exit before or at its parent");
                }
            }
        }
    }
}

fn validate_observed_events(
    receipt: &ProcessLineageReceipt,
    processes: &BTreeMap<ProcessIdentity, Option<ProcessIdentity>>,
    report: &mut ProcessLineageAuditReport,
) -> (Vec<EventKey>, bool, usize) {
    let process_keys: BTreeSet<_> = processes.keys().cloned().collect();
    let mut keys = Vec::new();
    let mut source_events = BTreeMap::<&str, Vec<(u64, u64)>>::new();
    for (index, event) in receipt.events.iter().enumerate() {
        if event.source_id.is_empty() || event.source_id.len() > 128 || event.sequence == 0 {
            error(report, "E_EVENT_SOURCE_ID", &format!("receipt.events[{index}]"), "source_id must be non-empty and sequence must start at 1");
        }
        if !valid_digest(&event.evidence_digest) {
            error(report, "E_EVENT_EVIDENCE_DIGEST", &format!("receipt.events[{index}].evidence_digest"), "event evidence digest must be lowercase sha256");
        }
        if !process_keys.contains(&event.process) {
            error(report, "E_EVENT_UNKNOWN_PROCESS", &format!("receipt.events[{index}]"), "observed event references a process absent from the observer process inventory");
        }
        if event.observed_at_monotonic_ns < receipt.capture_start_monotonic_ns || event.observed_at_monotonic_ns > receipt.capture_end_monotonic_ns {
            error(report, "E_EVENT_OUTSIDE_CAPTURE", &format!("receipt.events[{index}]"), "observer event timestamp lies outside its capture interval");
        }
        keys.push(EventKey { kind: event.kind, process: event.process.clone() });
        source_events.entry(event.source_id.as_str()).or_default().push((event.sequence,event.observed_at_monotonic_ns));
    }

    let mut gaps = 0usize;
    let mut integral = true;
    for (source, events) in &mut source_events {
        events.sort_by_key(|item| item.0);
        for (index, item) in events.iter().enumerate() {
            let expected = (index as u64) + 1;
            if item.0 != expected {
                integral = false;
                if item.0 > expected {
                    gaps = gaps.saturating_add((item.0 - expected) as usize);
                } else {
                    error(report, "E_EVENT_SEQUENCE_DUPLICATE", "receipt.events", &format!("event source {source} repeats sequence number {}", item.0));
                }
            }
            if index > 0 && item.1 < events[index - 1].1 {
                integral = false;
                error(report, "E_EVENT_TIME_REGRESSION", "receipt.events", &format!("event source {source} has non-monotonic timestamps"));
            }
        }
    }
    if receipt.health.sequence_gap_count as usize != gaps {
        integral = false;
        error(report, "E_SEQUENCE_GAP_COUNTER", "receipt.health.sequence_gap_count", "reported sequence-gap counter must equal independently computed sequence gaps");
    }
    if !integral {
        error(report, "E_EVENT_SEQUENCE_INTEGRITY", "receipt.events", "observer event stream is not contiguous and monotonic per source");
    }
    (keys, integral, gaps)
}

fn event_key_oracle(event: &OracleEvent) -> EventKey {
    EventKey { kind: event.kind, process: event.process.clone() }
}

fn count_event_keys(keys: &[EventKey]) -> BTreeMap<EventKey, usize> {
    let mut counts = BTreeMap::new();
    for key in keys {
        *counts.entry(key.clone()).or_default() += 1;
    }
    counts
}

fn parent_identity(parent: &ParentRelation) -> Option<ProcessIdentity> {
    match parent {
        ParentRelation::Root => None,
        ParentRelation::Parent(identity) => Some(identity.clone()),
    }
}

fn earliest_birth_time(events: &[OracleEvent], process: &ProcessIdentity) -> Option<u64> {
    events.iter().filter(|event| event.process == *process && matches!(
        event.kind,
        ProcessEventKind::RootStart | ProcessEventKind::Fork | ProcessEventKind::Clone | ProcessEventKind::Vfork
    )).map(|event| event.observed_at_monotonic_ns).min()
}

fn event_time(events: &[OracleEvent], process: &ProcessIdentity, kind: ProcessEventKind) -> Option<u64> {
    events.iter().find(|event| event.process == *process && event.kind == kind).map(|event| event.observed_at_monotonic_ns)
}

fn all_processes_have_exit_events(processes: &[OracleProcess], events: &[OracleEvent]) -> bool {
    processes.iter().all(|process| event_time(events, &process.identity, ProcessEventKind::Exit).is_some())
}

fn all_processes_have_exit_observed(processes: &[ObservedProcess], events: &[ObservedEvent]) -> bool {
    processes.iter().all(|process| events.iter().any(|event| event.process == process.identity && event.kind == ProcessEventKind::Exit))
}

fn count_reused_pids(pids: impl Iterator<Item = u32>) -> usize {
    let mut counts = BTreeMap::<u32, usize>::new();
    for pid in pids { *counts.entry(pid).or_default() += 1; }
    counts.values().filter(|count| **count > 1).count()
}

fn valid_digest(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    })
}

fn sha256_tagged(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    format!("sha256:{encoded}")
}

fn read_bounded(path: &Path) -> anyhow::Result<Vec<u8>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(MAX_INPUT_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_INPUT_BYTES {
        anyhow::bail!("{} exceeds the 16 MiB input limit", path.display());
    }
    Ok(bytes)
}

fn error(report: &mut ProcessLineageAuditReport, code: &str, path: &str, message: &str) {
    report.errors.push(LineageError {
        code: code.to_string(),
        path: path.to_string(),
        message: message.to_string(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn digest(seed: u8) -> String {
        format!("sha256:{}", format!("{:064x}", seed))
    }

    fn fixtures() -> (Value, Value) {
        let root = json!({"pid": 100, "start_time_ticks": 1000});
        let child = json!({"pid": 101, "start_time_ticks": 1010});
        let reused = json!({"pid": 101, "start_time_ticks": 1100});
        let oracle = json!({
            "format_version":1,"suite_id":"basalt-process-lineage-oracle-v1",
            "run_id":"run-1","task_id":"task-1","attempt_id":"attempt-1",
            "host_boot_id_digest":digest(1),"pid_namespace_inode":4026533001u64,
            "capture_start_monotonic_ns":100,"capture_end_monotonic_ns":1000,
            "root_process":root,"expected_tree_closed":true,
            "harness_binary_digest":digest(2),"supervisor_binary_digest":digest(3),
            "processes":[
                {"identity":root,"parent":{"relation":"root"}},
                {"identity":child,"parent":{"relation":"parent","identity":root}}
            ],
            "events":[
                {"kind":"root_start","process":root,"observed_at_monotonic_ns":110},
                {"kind":"exec","process":root,"observed_at_monotonic_ns":120},
                {"kind":"fork","process":child,"observed_at_monotonic_ns":130},
                {"kind":"exec","process":child,"observed_at_monotonic_ns":140},
                {"kind":"exit","process":child,"observed_at_monotonic_ns":900},
                {"kind":"exit","process":root,"observed_at_monotonic_ns":950}
            ]
        });
        let receipt = json!({
            "format_version":1,"suite_id":"basalt-process-lineage-receipt-v1",
            "run_id":"run-1","task_id":"task-1","attempt_id":"attempt-1",
            "host_boot_id_digest":digest(1),"pid_namespace_inode":4026533001u64,
            "capture_start_monotonic_ns":90,"capture_end_monotonic_ns":1010,
            "root_process":root,"observer_binary_digest":digest(4),
            "observer_configuration_digest":digest(5),"reported_tree_closed":true,
            "processes":[
                {"identity":root,"parent":{"relation":"root"}},
                {"identity":child,"parent":{"relation":"parent","identity":root}}
            ],
            "events":[
                {"source_id":"proc","sequence":1,"kind":"root_start","process":root,"observed_at_monotonic_ns":111,"evidence_digest":digest(6)},
                {"source_id":"proc","sequence":2,"kind":"exec","process":root,"observed_at_monotonic_ns":121,"evidence_digest":digest(7)},
                {"source_id":"proc","sequence":3,"kind":"fork","process":child,"observed_at_monotonic_ns":131,"evidence_digest":digest(8)},
                {"source_id":"proc","sequence":4,"kind":"exec","process":child,"observed_at_monotonic_ns":141,"evidence_digest":digest(9)},
                {"source_id":"proc","sequence":5,"kind":"exit","process":child,"observed_at_monotonic_ns":901,"evidence_digest":digest(10)},
                {"source_id":"proc","sequence":6,"kind":"exit","process":root,"observed_at_monotonic_ns":951,"evidence_digest":digest(11)}
            ],
            "health":{"event_count":6,"dropped_event_count":0,"ring_buffer_overflow_count":0,
              "observer_restart_count":0,"unmonitored_process_count":0,"sequence_gap_count":0}
        });
        let _ = reused;
        (oracle, receipt)
    }

    fn audit(oracle: &Value, receipt: &Value) -> ProcessLineageAuditReport {
        audit_bytes(oracle.to_string().as_bytes(), receipt.to_string().as_bytes())
    }

    #[test]
    fn exact_independent_lineage_and_event_stream_reconciles() {
        let (oracle, receipt) = fixtures();
        let report = audit(&oracle, &receipt);
        assert!(report.lineage_audit_passed, "{:?}", report.errors);
        assert!(report.process_graph_reconciled);
        assert!(report.event_set_reconciled);
        assert!(report.event_loss_free);
        assert!(report.process_tree_closed);
        assert!(!report.qualification_eligible);
        assert!(!report.oracle_authenticity_established);
    }

    #[test]
    fn pid_reuse_is_a_distinct_process_instance_not_a_pid_duplicate() {
        let (mut oracle, mut receipt) = fixtures();
        let reused = json!({"pid":101,"start_time_ticks":1100});
        oracle["processes"].as_array_mut().unwrap().push(json!({
            "identity":reused,"parent":{"relation":"parent","identity":{"pid":100,"start_time_ticks":1000}}
        }));
        oracle["events"].as_array_mut().unwrap().extend([
            json!({"kind":"fork","process":reused,"observed_at_monotonic_ns":160}),
            json!({"kind":"exit","process":reused,"observed_at_monotonic_ns":850})
        ]);
        receipt["processes"].as_array_mut().unwrap().push(json!({
            "identity":reused,"parent":{"relation":"parent","identity":{"pid":100,"start_time_ticks":1000}}
        }));
        let mut events=receipt["events"].as_array().unwrap().clone();
        events.push(json!({"source_id":"proc","sequence":7,"kind":"fork","process":reused,"observed_at_monotonic_ns":161,"evidence_digest":digest(12)}));
        events.push(json!({"source_id":"proc","sequence":8,"kind":"exit","process":reused,"observed_at_monotonic_ns":851,"evidence_digest":digest(13)}));
        receipt["events"]=Value::Array(events);
        receipt["health"]["event_count"]=json!(8);
        let report=audit(&oracle,&receipt);
        assert!(report.lineage_audit_passed,"{:?}",report.errors);
        assert_eq!(report.counts.pid_values_reused,1);
    }

    #[test]
    fn pid_only_identity_merging_is_caught_as_missing_and_unexpected() {
        let (oracle, mut receipt) = fixtures();
        receipt["processes"][1]["identity"]["start_time_ticks"]=json!(1000);
        receipt["events"][2]["process"]["start_time_ticks"]=json!(1000);
        receipt["events"][3]["process"]["start_time_ticks"]=json!(1000);
        receipt["events"][4]["process"]["start_time_ticks"]=json!(1000);
        let report=audit(&oracle,&receipt);
        assert!(!report.lineage_audit_passed);
        assert!(!report.process_graph_reconciled);
        assert!(report.counts.missing_processes>0);
        assert!(report.counts.unexpected_processes>0);
    }

    #[test]
    fn missing_child_exit_prevents_closed_tree_claim() {
        let (oracle, mut receipt) = fixtures();
        receipt["events"].as_array_mut().unwrap().remove(4);
        receipt["health"]["event_count"]=json!(5);
        receipt["health"]["dropped_event_count"]=json!(1);
        let report=audit(&oracle,&receipt);
        assert!(!report.lineage_audit_passed);
        assert!(!report.process_tree_closed);
        assert!(!report.event_loss_free);
        assert!(report.counts.missing_events>0);
    }

    #[test]
    fn reported_loss_is_fail_closed_even_when_expected_events_appear() {
        let (oracle, mut receipt)=fixtures();
        receipt["health"]["dropped_event_count"]=json!(1);
        let report=audit(&oracle,&receipt);
        assert!(!report.lineage_audit_passed);
        assert!(!report.event_loss_free);
    }

    #[test]
    fn sequence_gaps_must_match_reported_gap_counter_and_still_fail() {
        let (oracle, mut receipt)=fixtures();
        receipt["events"][2]["sequence"]=json!(4);
        receipt["health"]["sequence_gap_count"]=json!(1);
        let report=audit(&oracle,&receipt);
        assert!(!report.lineage_audit_passed);
        assert!(!report.event_sequence_integral);
        assert_eq!(report.counts.computed_sequence_gaps,1);
    }

    #[test]
    fn duplicate_event_sequence_is_rejected() {
        let (oracle, mut receipt)=fixtures();
        receipt["events"][2]["sequence"]=json!(2);
        let report=audit(&oracle,&receipt);
        assert!(!report.lineage_audit_passed);
        assert!(!report.event_sequence_integral);
    }

    #[test]
    fn parent_must_exist_and_process_graph_must_be_acyclic() {
        let (mut oracle, receipt)=fixtures();
        oracle["processes"][1]["parent"]["identity"]=json!({"pid":999,"start_time_ticks":9999});
        let report=audit(&oracle,&receipt);
        assert!(!report.oracle_well_formed);
        assert!(report.errors.iter().any(|e|e.code=="E_PARENT_REFERENCE"));
    }

    #[test]
    fn child_must_be_born_after_parent_and_exit_before_parent() {
        let (mut oracle, receipt)=fixtures();
        oracle["events"][2]["observed_at_monotonic_ns"]=json!(105);
        let report=audit(&oracle,&receipt);
        assert!(!report.oracle_well_formed);
        assert!(report.errors.iter().any(|e|e.code=="E_ORACLE_PARENT_BIRTH_ORDER"));
    }

    #[test]
    fn mismatched_pid_namespace_or_boot_id_rejects_cross_context_join() {
        let (oracle, mut receipt)=fixtures();
        receipt["pid_namespace_inode"]=json!(4026533002u64);
        let report=audit(&oracle,&receipt);
        assert!(!report.identity_context_match);
        assert!(!report.lineage_audit_passed);
    }

    #[test]
    fn observer_interval_must_cover_independent_oracle_interval() {
        let (oracle, mut receipt)=fixtures();
        receipt["capture_start_monotonic_ns"]=json!(200);
        let report=audit(&oracle,&receipt);
        assert!(report.errors.iter().any(|e|e.code=="E_CAPTURE_WINDOW_NOT_COVERED"));
    }

    #[test]
    fn out_of_scope_observer_process_is_unexpected() {
        let (oracle, mut receipt)=fixtures();
        receipt["processes"].as_array_mut().unwrap().push(json!({
            "identity":{"pid":999,"start_time_ticks":9999},
            "parent":{"relation":"parent","identity":{"pid":100,"start_time_ticks":1000}}
        }));
        let report=audit(&oracle,&receipt);
        assert!(!report.process_graph_reconciled);
        assert_eq!(report.counts.unexpected_processes,1);
    }

    #[test]
    fn malformed_event_digest_is_not_accepted() {
        let (oracle, mut receipt)=fixtures();
        receipt["events"][0]["evidence_digest"]=json!("sha256:not-a-digest");
        let report=audit(&oracle,&receipt);
        assert!(!report.receipt_well_formed);
        assert!(report.errors.iter().any(|e|e.code=="E_EVENT_EVIDENCE_DIGEST"));
    }

    #[test]
    fn unclosed_oracle_tree_cannot_be_promoted_to_closed() {
        let (mut oracle, receipt)=fixtures();
        oracle["expected_tree_closed"]=json!(false);
        let report=audit(&oracle,&receipt);
        assert!(!report.process_tree_closed);
        assert!(!report.lineage_audit_passed);
    }

    #[test]
    fn unknown_fields_are_rejected_by_strict_deserialization() {
        let (oracle, mut receipt)=fixtures();
        receipt["surprise"]=json!(true);
        let report=audit(&oracle,&receipt);
        assert!(!report.receipt_well_formed);
        assert!(report.errors.iter().any(|e|e.code=="E_RECEIPT_JSON"));
    }

    #[test]
    fn event_timestamp_must_be_inside_capture_window() {
        let (oracle, mut receipt)=fixtures();
        receipt["events"][0]["observed_at_monotonic_ns"]=json!(50);
        let report=audit(&oracle,&receipt);
        assert!(!report.receipt_well_formed);
        assert!(report.errors.iter().any(|e|e.code=="E_EVENT_OUTSIDE_CAPTURE"));
    }

    #[test]
    fn pid_reuse_count_is_scoped_to_same_pid_with_different_start_ticks() {
        let (mut oracle, receipt)=fixtures();
        let reused=json!({"pid":100,"start_time_ticks":1200});
        oracle["processes"].as_array_mut().unwrap().push(json!({"identity":reused,"parent":{"relation":"parent","identity":{"pid":100,"start_time_ticks":1000}}}));
        oracle["events"].as_array_mut().unwrap().extend([
          json!({"kind":"fork","process":reused,"observed_at_monotonic_ns":200}),
          json!({"kind":"exit","process":reused,"observed_at_monotonic_ns":800})
        ]);
        let mut receipt2=receipt;
        receipt2["processes"].as_array_mut().unwrap().push(json!({"identity":reused,"parent":{"relation":"parent","identity":{"pid":100,"start_time_ticks":1000}}}));
        receipt2["events"].as_array_mut().unwrap().extend([
          json!({"source_id":"proc","sequence":7,"kind":"fork","process":reused,"observed_at_monotonic_ns":201,"evidence_digest":digest(14)}),
          json!({"source_id":"proc","sequence":8,"kind":"exit","process":reused,"observed_at_monotonic_ns":801,"evidence_digest":digest(15)})
        ]);
        receipt2["health"]["event_count"]=json!(8);
        let report=audit(&oracle,&receipt2);
        assert!(report.lineage_audit_passed,"{:?}",report.errors);
        assert_eq!(report.counts.pid_values_reused,1);
    }
}
