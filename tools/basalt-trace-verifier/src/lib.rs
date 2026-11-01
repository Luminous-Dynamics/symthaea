use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Trace {
    pub suite_id: String,
    pub version: u32,
    pub trace_id: String,
    pub claim_ceiling: String,
    pub generator: Generator,
    pub authority: Authority,
    pub initial_state: InitialState,
    pub operations: Vec<Operation>,
    pub reference_linearization_order: Vec<String>,
    pub faults: Vec<Fault>,
    pub witness_observations: Vec<WitnessObservation>,
    pub final_observation: FinalObservation,
    pub invariant_expectations: Vec<InvariantExpectation>,
    pub fixture_digest_catalog: Vec<DigestEntry>,
    pub negative_controls: Vec<NegativeControl>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Generator {
    pub name: String,
    pub version: String,
    pub seed: String,
    pub clock_basis: String,
    pub ordering_rule: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Authority {
    pub authority_id: String,
    pub incarnation_id: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InitialState {
    pub authority_revision: String,
    pub state_digest: String,
    pub policy_digest: String,
    pub source_cursors: Vec<SourceCursor>,
    pub log_cursors: Vec<LogCursor>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SourceCursor {
    pub source_id: String,
    pub generation: u64,
    pub cursor: String,
    pub predecessor_digest: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LogCursor {
    pub log_origin_id: String,
    pub checkpoint: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Operation {
    pub op_id: String,
    pub runner_id: String,
    pub client_id: String,
    pub client_sequence: u64,
    pub retry_group_id: Option<String>,
    pub transaction_attempt: Option<u64>,
    pub kind: String,
    pub scope: Scope,
    pub invoked_at_ns: u64,
    pub completed_at_ns: u64,
    #[serde(default)]
    pub happens_after: Vec<String>,
    pub request: Request,
    pub expected_result: ExpectedResult,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Scope {
    pub source_id: String,
    pub log_origin_id: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub expected_source_generation: Option<u64>,
    pub expected_source_predecessor_digest: Option<String>,
    pub expected_log_checkpoint: Option<String>,
    pub expected_policy_digest: Option<String>,
    pub receipt_id: Option<String>,
    pub receipt_digest: Option<String>,
    pub next_source_cursor: Option<String>,
    pub next_source_predecessor_digest: Option<String>,
    pub next_log_checkpoint: Option<String>,
    pub restored_incarnation_id: Option<String>,
    pub restored_authority_revision: Option<String>,
    pub restored_state_digest: Option<String>,
    pub witness_sequence_seen: Option<u64>,
    pub next_policy_digest: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExpectedResult {
    pub outcome: String,
    pub receipt_id: Option<String>,
    pub receipt_acceptance_revision: Option<String>,
    pub current_authority_revision: Option<String>,
    pub observed_store_revision: Option<String>,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fault {
    pub fault_id: String,
    pub target_op_id: String,
    pub runner_id: String,
    pub at_ns: u64,
    pub class: String,
    pub action: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WitnessObservation {
    pub witness_id: String,
    pub sequence: u64,
    pub authority_id: String,
    pub incarnation_id: String,
    pub authority_revision: String,
    pub state_digest: String,
    pub predecessor_witness_digest: Option<String>,
    pub signed_record_digest: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FinalObservation {
    pub authority_revision: String,
    pub state_digest: String,
    pub policy_digest: String,
    pub source_cursors: Vec<SourceCursor>,
    pub log_cursors: Vec<LogCursor>,
    pub accepted_receipts: Vec<AcceptedReceipt>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AcceptedReceipt {
    pub receipt_id: String,
    pub receipt_digest: String,
    pub receipt_acceptance_revision: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InvariantExpectation {
    pub id: String,
    pub expected: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DigestEntry {
    pub label: String,
    pub preimage: String,
    pub digest: String,
    pub role: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NegativeControl {
    pub mutation_id: String,
    pub mutation: String,
    pub must_be_detected: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct VerificationReport {
    pub report_version: u32,
    pub verifier_version: String,
    pub schema_sha256: Option<String>,
    pub claim_ceiling: String,
    pub status: String,
    pub trace_id: String,
    pub trace_sha256: String,
    pub operations_replayed: usize,
    pub invariants_checked: usize,
    pub mutants_detected: usize,
    pub mutants_seeded: usize,
    pub counterexamples: Vec<String>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
struct ModelReceipt {
    receipt_digest: String,
    acceptance_revision: u64,
}

#[derive(Clone, Debug, Serialize)]
struct ModelState {
    revision: u64,
    policy_digest: String,
    sources: BTreeMap<String, SourceCursor>,
    logs: BTreeMap<String, String>,
    receipts: BTreeMap<String, ModelReceipt>,
    indeterminate_receipts: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ComputedResult {
    outcome: String,
    receipt_id: Option<String>,
    receipt_acceptance_revision: Option<String>,
    current_authority_revision: Option<String>,
    observed_store_revision: Option<String>,
}

impl ComputedResult {
    fn new(outcome: &str, current_revision: Option<u64>) -> Self {
        Self {
            outcome: outcome.to_string(),
            receipt_id: None,
            receipt_acceptance_revision: None,
            current_authority_revision: current_revision.map(revision_text),
            observed_store_revision: None,
        }
    }

    fn receipt(mut self, id: &str, digest_revision: u64) -> Self {
        self.receipt_id = Some(id.to_string());
        self.receipt_acceptance_revision = Some(revision_text(digest_revision));
        self
    }
}

fn revision_text(revision: u64) -> String {
    format!("rev-{revision}")
}

fn parse_revision(value: &str) -> Option<u64> {
    value.strip_prefix("rev-")?.parse().ok()
}

fn sha256_label(value: &str) -> String {
    let hash = Sha256::digest(value.as_bytes());
    format!("sha256:{}", hex_lower(&hash))
}

fn hex_lower(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn valid_id(value: &str) -> bool {
    !value.is_empty() && value.len() <= 128 && !value.chars().any(char::is_control)
}

fn valid_digest(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64
        && hex
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn push_error(errors: &mut Vec<String>, message: impl Into<String>) {
    errors.push(message.into());
}

fn validate_structure(trace: &Trace, errors: &mut Vec<String>) {
    if trace.suite_id != "basalt-storage-conformance-trace" || trace.version != 1 {
        push_error(errors, "unsupported suite_id/version");
    }
    if trace.trace_id.is_empty() || trace.claim_ceiling.is_empty() {
        push_error(errors, "trace identity or claim ceiling is empty");
    }
    if trace.generator.clock_basis != "monotonic_nanoseconds_per_runner" {
        push_error(errors, "unsupported clock basis");
    }
    if trace.generator.ordering_rule != "per_runner_only; no cross-host wall-clock ordering" {
        push_error(errors, "unsupported clock ordering rule");
    }
    if trace.operations.is_empty() {
        push_error(errors, "trace has no operations");
    }
    if parse_revision(&trace.initial_state.authority_revision).is_none() {
        push_error(errors, "initial authority revision is not canonical rev-N");
    }

    let mut initial_sources = BTreeSet::new();
    for cursor in &trace.initial_state.source_cursors {
        if !initial_sources.insert(&cursor.source_id) {
            push_error(
                errors,
                format!("duplicate initial source cursor {}", cursor.source_id),
            );
        }
        if !valid_digest(&cursor.predecessor_digest) {
            push_error(
                errors,
                format!(
                    "initial source {} has malformed predecessor digest",
                    cursor.source_id
                ),
            );
        }
    }
    let mut initial_logs = BTreeSet::new();
    for cursor in &trace.initial_state.log_cursors {
        if !initial_logs.insert(&cursor.log_origin_id) {
            push_error(
                errors,
                format!("duplicate initial log cursor {}", cursor.log_origin_id),
            );
        }
    }

    let mut operation_map = HashMap::<String, &Operation>::new();
    for operation in &trace.operations {
        if operation.op_id.is_empty()
            || operation.runner_id.is_empty()
            || operation.client_id.is_empty()
        {
            push_error(
                errors,
                format!(
                    "{} has empty operation/runner/client identity",
                    operation.op_id
                ),
            );
        }
        if operation.completed_at_ns < operation.invoked_at_ns {
            push_error(
                errors,
                format!("{} completes before invocation", operation.op_id),
            );
        }
        if !valid_id(&operation.scope.source_id) || !valid_id(&operation.scope.log_origin_id) {
            push_error(
                errors,
                format!("{} has malformed scope identity", operation.op_id),
            );
        }
        let request = &operation.request;
        let required = match operation.kind.as_str() {
            "advance" => vec![
                (
                    "expected_source_generation",
                    request.expected_source_generation.is_some(),
                ),
                (
                    "expected_source_predecessor_digest",
                    request.expected_source_predecessor_digest.is_some(),
                ),
                (
                    "expected_log_checkpoint",
                    request.expected_log_checkpoint.is_some(),
                ),
                (
                    "expected_policy_digest",
                    request.expected_policy_digest.is_some(),
                ),
                ("receipt_id", request.receipt_id.is_some()),
                ("receipt_digest", request.receipt_digest.is_some()),
                ("next_source_cursor", request.next_source_cursor.is_some()),
                (
                    "next_source_predecessor_digest",
                    request.next_source_predecessor_digest.is_some(),
                ),
                ("next_log_checkpoint", request.next_log_checkpoint.is_some()),
            ],
            "receipt_lookup" => vec![
                ("receipt_id", request.receipt_id.is_some()),
                ("receipt_digest", request.receipt_digest.is_some()),
            ],
            "restore_verify" => vec![
                (
                    "restored_incarnation_id",
                    request.restored_incarnation_id.is_some(),
                ),
                (
                    "restored_authority_revision",
                    request.restored_authority_revision.is_some(),
                ),
                (
                    "restored_state_digest",
                    request.restored_state_digest.is_some(),
                ),
                (
                    "witness_sequence_seen",
                    request.witness_sequence_seen.is_some(),
                ),
            ],
            "policy_publish" => vec![
                (
                    "expected_policy_digest",
                    request.expected_policy_digest.is_some(),
                ),
                ("next_policy_digest", request.next_policy_digest.is_some()),
            ],
            other => {
                push_error(
                    errors,
                    format!("{} has unsupported operation kind {other}", operation.op_id),
                );
                Vec::new()
            }
        };
        for (field, present) in required {
            if !present {
                push_error(
                    errors,
                    format!(
                        "{} is missing required request field {field}",
                        operation.op_id
                    ),
                );
            }
        }
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
        .contains(&operation.expected_result.outcome.as_str())
        {
            push_error(
                errors,
                format!("{} has unsupported expected outcome", operation.op_id),
            );
        }
        for revision in [
            operation
                .expected_result
                .receipt_acceptance_revision
                .as_deref(),
            operation
                .expected_result
                .current_authority_revision
                .as_deref(),
            operation.expected_result.observed_store_revision.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            let canonical = parse_revision(revision).map(revision_text);
            if canonical.as_deref() != Some(revision) {
                push_error(
                    errors,
                    format!(
                        "{} contains non-canonical result revision {revision}",
                        operation.op_id
                    ),
                );
            }
        }
        if operation_map
            .insert(operation.op_id.clone(), operation)
            .is_some()
        {
            push_error(
                errors,
                format!("duplicate operation ID {}", operation.op_id),
            );
        }
    }

    let mut order = HashMap::<String, usize>::new();
    for (index, id) in trace.reference_linearization_order.iter().enumerate() {
        if order.insert(id.clone(), index).is_some() {
            push_error(
                errors,
                format!("duplicate ID in reference linearization order: {id}"),
            );
        }
        if !operation_map.contains_key(id.as_str()) {
            push_error(
                errors,
                format!("linearization order references unknown operation {id}"),
            );
        }
    }
    if order.len() != operation_map.len() {
        push_error(
            errors,
            "reference linearization order must include every operation exactly once",
        );
    }

    for operation in &trace.operations {
        for dependency in &operation.happens_after {
            match (order.get(&operation.op_id), order.get(dependency)) {
                (Some(current), Some(previous)) if previous < current => {}
                (Some(_), Some(_)) => push_error(
                    errors,
                    format!(
                        "{} violates happens_after dependency {}",
                        operation.op_id, dependency
                    ),
                ),
                (_, None) => push_error(
                    errors,
                    format!(
                        "{} references missing dependency {}",
                        operation.op_id, dependency
                    ),
                ),
                _ => push_error(
                    errors,
                    format!("{} is missing from linearization order", operation.op_id),
                ),
            }
        }
    }

    // Program order is per client, even when that client moves between runners.
    let mut clients = BTreeMap::<&str, Vec<&Operation>>::new();
    for operation in &trace.operations {
        clients
            .entry(&operation.client_id)
            .or_default()
            .push(operation);
    }
    for (client, operations) in clients.iter_mut() {
        operations.sort_by_key(|operation| operation.client_sequence);
        for (expected_sequence, operation) in operations.iter().enumerate() {
            if operation.client_sequence != expected_sequence as u64 {
                push_error(
                    errors,
                    format!(
                        "client {client} has a sequence gap or duplicate at {}",
                        operation.op_id
                    ),
                );
            }
        }
        for pair in operations.windows(2) {
            if order.get(&pair[0].op_id) >= order.get(&pair[1].op_id) {
                push_error(
                    errors,
                    format!("client {} program order is violated", client),
                );
            }
        }
    }

    // Only compare monotonic clocks from the same runner.
    let operations: Vec<&Operation> = trace.operations.iter().collect();
    for left in &operations {
        for right in &operations {
            if left.runner_id == right.runner_id
                && left.op_id != right.op_id
                && left.completed_at_ns <= right.invoked_at_ns
                && order.get(&left.op_id) >= order.get(&right.op_id)
            {
                push_error(
                    errors,
                    format!(
                        "reference order violates same-runner real-time precedence: {} before {}",
                        left.op_id, right.op_id
                    ),
                );
            }
        }
    }

    let mut fault_ids = BTreeSet::new();
    for fault in &trace.faults {
        if !fault_ids.insert(&fault.fault_id) {
            push_error(errors, format!("duplicate fault ID {}", fault.fault_id));
        }
        if ![
            "process_crash",
            "power_loss_image",
            "io_error",
            "response_loss",
            "network_partition",
            "network_delay",
            "serialization_failure",
            "member_restart",
            "snapshot_restore",
            "witness_unavailable",
        ]
        .contains(&fault.class.as_str())
        {
            push_error(
                errors,
                format!(
                    "fault {} has unsupported class {}",
                    fault.fault_id, fault.class
                ),
            );
        }
        if fault.action.is_empty() {
            push_error(errors, format!("fault {} has empty action", fault.fault_id));
        }
        match operation_map.get(&fault.target_op_id) {
            Some(target) => {
                if target.runner_id != fault.runner_id {
                    push_error(
                        errors,
                        format!(
                            "fault {} runner differs from target operation",
                            fault.fault_id
                        ),
                    );
                }
                if fault.at_ns < target.invoked_at_ns || fault.at_ns > target.completed_at_ns {
                    push_error(
                        errors,
                        format!(
                            "fault {} falls outside target operation interval",
                            fault.fault_id
                        ),
                    );
                }
                let required_kind = match fault.class.as_str() {
                    "response_loss" | "serialization_failure" => "advance",
                    "snapshot_restore" => "restore_verify",
                    _ => target.kind.as_str(),
                };
                if target.kind != required_kind {
                    push_error(
                        errors,
                        format!(
                            "fault {} targets incompatible operation kind",
                            fault.fault_id
                        ),
                    );
                }
                if fault.class == "response_loss"
                    && target.expected_result.outcome != "indeterminate"
                {
                    push_error(
                        errors,
                        format!(
                            "response-loss fault {} is not recorded as indeterminate",
                            fault.fault_id
                        ),
                    );
                }
                if fault.class == "serialization_failure"
                    && target.expected_result.outcome != "retryable_abort"
                {
                    push_error(
                        errors,
                        format!(
                            "serialization fault {} is not recorded as retryable_abort",
                            fault.fault_id
                        ),
                    );
                }
            }
            None => push_error(
                errors,
                format!(
                    "fault {} targets missing operation {}",
                    fault.fault_id, fault.target_op_id
                ),
            ),
        }
    }

    let mut digest_labels = BTreeSet::new();
    let mut digest_values = BTreeSet::new();
    for entry in &trace.fixture_digest_catalog {
        if !digest_labels.insert(&entry.label) {
            push_error(errors, format!("duplicate digest label {}", entry.label));
        }
        if !digest_values.insert(&entry.digest) {
            push_error(errors, format!("duplicate digest value {}", entry.digest));
        }
        if entry.digest != sha256_label(&entry.preimage) {
            push_error(
                errors,
                format!("fixture digest does not match preimage for {}", entry.label),
            );
        }
        if !valid_digest(&entry.digest) {
            push_error(errors, format!("malformed digest for {}", entry.label));
        }
    }
    let catalog: HashSet<&str> = trace
        .fixture_digest_catalog
        .iter()
        .map(|x| x.digest.as_str())
        .collect();
    for digest in [
        &trace.initial_state.state_digest,
        &trace.initial_state.policy_digest,
        &trace.final_observation.state_digest,
        &trace.final_observation.policy_digest,
    ] {
        if !catalog.contains(digest.as_str()) {
            push_error(
                errors,
                format!("state/policy digest absent from fixture catalog: {digest}"),
            );
        }
    }
    for witness in &trace.witness_observations {
        for digest in [&witness.state_digest, &witness.signed_record_digest] {
            if !catalog.contains(digest.as_str()) {
                push_error(
                    errors,
                    format!(
                        "witness {} references digest absent from fixture catalog",
                        witness.sequence
                    ),
                );
            }
        }
        if let Some(previous) = witness.predecessor_witness_digest.as_deref()
            && !catalog.contains(previous)
        {
            push_error(
                errors,
                format!(
                    "witness {} predecessor digest absent from fixture catalog",
                    witness.sequence
                ),
            );
        }
    }
    for receipt in &trace.final_observation.accepted_receipts {
        if !catalog.contains(receipt.receipt_digest.as_str()) {
            push_error(
                errors,
                format!(
                    "final receipt {} digest absent from fixture catalog",
                    receipt.receipt_id
                ),
            );
        }
    }
    for operation in &trace.operations {
        let request_digests = [
            operation
                .request
                .expected_source_predecessor_digest
                .as_deref(),
            operation.request.expected_policy_digest.as_deref(),
            operation.request.receipt_digest.as_deref(),
            operation.request.next_source_predecessor_digest.as_deref(),
            operation.request.restored_state_digest.as_deref(),
            operation.request.next_policy_digest.as_deref(),
        ];
        for digest in request_digests.into_iter().flatten() {
            if !catalog.contains(digest) {
                push_error(
                    errors,
                    format!(
                        "{} references digest absent from fixture catalog",
                        operation.op_id
                    ),
                );
            }
        }
    }

    let mut invariant_ids = BTreeSet::new();
    for invariant in &trace.invariant_expectations {
        if !invariant_ids.insert(&invariant.id) {
            push_error(errors, format!("duplicate invariant ID {}", invariant.id));
        }
        if invariant.expected != "pass" {
            push_error(
                errors,
                format!("invariant {} has unexpected required state", invariant.id),
            );
        }
    }
    let mut mutant_ids = BTreeSet::new();
    for mutant in &trace.negative_controls {
        if !mutant_ids.insert(&mutant.mutation_id) || !mutant.must_be_detected {
            push_error(
                errors,
                format!(
                    "invalid or duplicate negative control {}",
                    mutant.mutation_id
                ),
            );
        }
    }
    if trace.witness_observations.is_empty() {
        push_error(errors, "external witness observations are missing");
    }
}

fn validate_retry_groups(trace: &Trace, errors: &mut Vec<String>) {
    let mut groups = BTreeMap::<&str, Vec<&Operation>>::new();
    for operation in &trace.operations {
        if let Some(group) = operation.retry_group_id.as_deref() {
            groups.entry(group).or_default().push(operation);
        } else if operation.transaction_attempt.is_some() {
            push_error(
                errors,
                format!(
                    "{} has transaction attempt without retry group",
                    operation.op_id
                ),
            );
        }
    }
    for (group, mut operations) in groups {
        operations.sort_by_key(|operation| operation.transaction_attempt.unwrap_or(0));
        if operations.len() < 2 {
            push_error(
                errors,
                format!("retry group {group} has fewer than two attempts"),
            );
            continue;
        }
        let first = operations[0];
        if first.transaction_attempt != Some(1) {
            push_error(
                errors,
                format!("retry group {group} does not start at attempt 1"),
            );
        }
        for (index, operation) in operations.iter().enumerate() {
            if operation.transaction_attempt != Some(index as u64 + 1) {
                push_error(
                    errors,
                    format!("retry group {group} has a non-contiguous attempt sequence"),
                );
            }
            if operation.client_id != first.client_id
                || operation.scope != first.scope
                || operation.kind != first.kind
                || operation.request != first.request
            {
                push_error(
                    errors,
                    format!("retry group {group} did not replay the whole transaction request"),
                );
            }
            if index > 0
                && !operation
                    .happens_after
                    .contains(&operations[index - 1].op_id)
            {
                push_error(
                    errors,
                    format!("retry group {group} attempt lacks happens_after edge"),
                );
            }
        }
        if fault_for(trace, first, "serialization_failure").is_none() {
            push_error(
                errors,
                format!("retry group {group} has no SQLSTATE/serialization-failure injection"),
            );
        }
        if first.expected_result.outcome != "retryable_abort" {
            push_error(
                errors,
                format!("retry group {group} first attempt is not a retryable abort"),
            );
        }
        if operations
            .last()
            .is_some_and(|last| last.expected_result.outcome != "committed")
        {
            push_error(
                errors,
                format!(
                    "retry group {group} does not end in a committed whole-transaction attempt"
                ),
            );
        }
    }
}

fn validate_witnesses(trace: &Trace, errors: &mut Vec<String>) {
    let Some(first) = trace.witness_observations.first() else {
        return;
    };
    let mut previous = None::<&WitnessObservation>;
    for witness in &trace.witness_observations {
        if witness.witness_id != first.witness_id
            || witness.authority_id != trace.authority.authority_id
            || witness.incarnation_id != trace.authority.incarnation_id
        {
            push_error(
                errors,
                format!("witness {} identity/incarnation mismatch", witness.sequence),
            );
        }
        if !valid_digest(&witness.state_digest) || !valid_digest(&witness.signed_record_digest) {
            push_error(
                errors,
                format!("witness {} has malformed digest", witness.sequence),
            );
        }
        match previous {
            None => {
                if witness.predecessor_witness_digest.is_some() {
                    push_error(errors, "first witness record must not claim a predecessor");
                }
            }
            Some(prev) => {
                if witness.sequence <= prev.sequence {
                    push_error(errors, "witness sequence is not strictly increasing");
                }
                if witness.predecessor_witness_digest.as_deref()
                    != Some(prev.signed_record_digest.as_str())
                {
                    push_error(
                        errors,
                        format!(
                            "witness {} predecessor digest does not link",
                            witness.sequence
                        ),
                    );
                }
                match (
                    parse_revision(&prev.authority_revision),
                    parse_revision(&witness.authority_revision),
                ) {
                    (Some(a), Some(b)) if b > a => {}
                    _ => push_error(
                        errors,
                        "witness authority revision is not strictly increasing",
                    ),
                }
            }
        }
        previous = Some(witness);
    }
    let head = trace
        .witness_observations
        .last()
        .expect("nonempty witnesses");
    if head.state_digest != trace.final_observation.state_digest
        || head.authority_revision != trace.final_observation.authority_revision
    {
        push_error(
            errors,
            "witness head does not bind final authority revision and state digest",
        );
    }
}

fn expected_fields_match(expected: &ExpectedResult, actual: &ComputedResult) -> bool {
    expected.outcome == actual.outcome
        && expected.receipt_id == actual.receipt_id
        && expected.receipt_acceptance_revision == actual.receipt_acceptance_revision
        && expected.current_authority_revision == actual.current_authority_revision
        && expected.observed_store_revision == actual.observed_store_revision
}

fn fault_for<'a>(trace: &'a Trace, operation: &Operation, class: &str) -> Option<&'a Fault> {
    trace
        .faults
        .iter()
        .find(|fault| fault.target_op_id == operation.op_id && fault.class == class)
}

fn advance(state: &mut ModelState, trace: &Trace, operation: &Operation) -> ComputedResult {
    let request = &operation.request;
    let Some(receipt_id) = request.receipt_id.as_deref() else {
        return ComputedResult::new("compare_conflict", Some(state.revision));
    };
    let Some(receipt_digest) = request.receipt_digest.as_deref() else {
        return ComputedResult::new("compare_conflict", Some(state.revision));
    };

    if let Some(existing) = state.receipts.get(receipt_id) {
        if existing.receipt_digest == receipt_digest {
            return ComputedResult::new("duplicate", Some(state.revision))
                .receipt(receipt_id, existing.acceptance_revision);
        }
        return ComputedResult::new("compare_conflict", Some(state.revision));
    }

    if fault_for(trace, operation, "serialization_failure").is_some() {
        return ComputedResult::new("retryable_abort", Some(state.revision));
    }

    let compare_ok = request.expected_policy_digest.as_deref()
        == Some(state.policy_digest.as_str())
        && state
            .sources
            .get(&operation.scope.source_id)
            .is_some_and(|cursor| {
                Some(cursor.generation) == request.expected_source_generation
                    && Some(cursor.predecessor_digest.as_str())
                        == request.expected_source_predecessor_digest.as_deref()
            })
        && state
            .logs
            .get(&operation.scope.log_origin_id)
            .map(String::as_str)
            == request.expected_log_checkpoint.as_deref();
    if !compare_ok {
        return ComputedResult::new("compare_conflict", Some(state.revision));
    }

    let Some(current_source) = state.sources.get(&operation.scope.source_id).cloned() else {
        return ComputedResult::new("compare_conflict", Some(state.revision));
    };
    let Some(next_generation) = current_source.generation.checked_add(1) else {
        return ComputedResult::new("compare_conflict", Some(state.revision));
    };
    let (Some(next_cursor), Some(next_predecessor), Some(next_log)) = (
        request.next_source_cursor.as_ref(),
        request.next_source_predecessor_digest.as_ref(),
        request.next_log_checkpoint.as_ref(),
    ) else {
        return ComputedResult::new("compare_conflict", Some(state.revision));
    };
    let Some(next_revision) = state.revision.checked_add(1) else {
        return ComputedResult::new("compare_conflict", Some(state.revision));
    };

    state.sources.insert(
        operation.scope.source_id.clone(),
        SourceCursor {
            source_id: operation.scope.source_id.clone(),
            generation: next_generation,
            cursor: next_cursor.clone(),
            predecessor_digest: next_predecessor.clone(),
        },
    );
    state
        .logs
        .insert(operation.scope.log_origin_id.clone(), next_log.clone());
    state.receipts.insert(
        receipt_id.to_string(),
        ModelReceipt {
            receipt_digest: receipt_digest.to_string(),
            acceptance_revision: next_revision,
        },
    );
    state.revision = next_revision;

    if fault_for(trace, operation, "response_loss").is_some() {
        state.indeterminate_receipts.insert(receipt_id.to_string());
        ComputedResult::new("indeterminate", None)
    } else {
        ComputedResult::new("committed", Some(state.revision)).receipt(receipt_id, next_revision)
    }
}

fn initial_model_state(trace: &Trace) -> ModelState {
    ModelState {
        revision: parse_revision(&trace.initial_state.authority_revision).unwrap_or(0),
        policy_digest: trace.initial_state.policy_digest.clone(),
        sources: trace
            .initial_state
            .source_cursors
            .iter()
            .cloned()
            .map(|cursor| (cursor.source_id.clone(), cursor))
            .collect(),
        logs: trace
            .initial_state
            .log_cursors
            .iter()
            .map(|cursor| (cursor.log_origin_id.clone(), cursor.checkpoint.clone()))
            .collect(),
        receipts: BTreeMap::new(),
        indeterminate_receipts: BTreeSet::new(),
    }
}

fn apply_operation(
    state: &mut ModelState,
    trace: &Trace,
    operation: &Operation,
    errors: &mut Vec<String>,
) -> ComputedResult {
    match operation.kind.as_str() {
            "advance" => advance(&mut state, trace, operation),
            "receipt_lookup" => {
                let receipt_id = operation.request.receipt_id.as_deref().unwrap_or("");
                let receipt_digest = operation.request.receipt_digest.as_deref().unwrap_or("");
                match state.receipts.get(receipt_id) {
                    Some(receipt) if receipt.receipt_digest == receipt_digest => {
                        let outcome = if state.indeterminate_receipts.contains(receipt_id) {
                            "receipt_found"
                        } else {
                            "duplicate"
                        };
                        ComputedResult::new(outcome, Some(state.revision))
                            .receipt(receipt_id, receipt.acceptance_revision)
                    }
                    _ => ComputedResult::new("compare_conflict", Some(state.revision)),
                }
            }
            "policy_publish" => {
                if operation.request.expected_policy_digest.as_deref()
                    == Some(state.policy_digest.as_str())
                {
                    if let Some(next_policy) = operation.request.next_policy_digest.as_ref() {
                        if let Some(next_revision) = state.revision.checked_add(1) {
                            state.policy_digest = next_policy.clone();
                            state.revision = next_revision;
                            ComputedResult::new("committed", Some(state.revision))
                        } else {
                            ComputedResult::new("compare_conflict", Some(state.revision))
                        }
                    } else {
                        ComputedResult::new("compare_conflict", Some(state.revision))
                    }
                } else {
                    ComputedResult::new("compare_conflict", Some(state.revision))
                }
            }
            "restore_verify" => {
                let observed_revision = operation.request.restored_authority_revision.clone();
                let observed_state = operation.request.restored_state_digest.as_deref();
                match trace.witness_observations.last() {
                    None => {
                        let mut result =
                            ComputedResult::new("rejected_anti_rollback_unknown", None);
                        result.observed_store_revision = observed_revision;
                        result
                    }
                    Some(head) => {
                        let fresh = operation.request.restored_incarnation_id.as_deref()
                            == Some(head.incarnation_id.as_str())
                            && observed_state == Some(head.state_digest.as_str())
                            && observed_revision.as_deref()
                                == Some(head.authority_revision.as_str())
                            && operation.request.witness_sequence_seen == Some(head.sequence);
                        let mut result = ComputedResult::new(
                            if fresh {
                                "restore_verified"
                            } else {
                                "rejected_anti_rollback"
                            },
                            parse_revision(&head.authority_revision),
                        );
                        result.observed_store_revision = observed_revision;
                        result
                    }
                }
            }
            other => {
                push_error(
                    errors,
                    format!("unsupported operation kind {other} in {}", operation.op_id),
                );
                ComputedResult::new("compare_conflict", Some(state.revision))
            }
        };
}

fn replay(trace: &Trace, errors: &mut Vec<String>) -> BTreeMap<String, ComputedResult> {
    let mut state =     let mut state = initial_model_state(trace);

    let operations: HashMap<String, &Operation> = trace
        .operations
        .iter()
        .map(|op| (op.op_id.clone(), op))
        .collect();
    let mut results = BTreeMap::new();

    for id in &trace.reference_linearization_order {
        let Some(operation) = operations.get(id).copied() else {
            continue;
        };
        let computed = apply_operation(&mut state, trace, operation, errors);)
            .is_some()
        {
            push_error(
                errors,
                format!("duplicate final source cursor {}", cursor.source_id),
            );
        }
    }
    let mut expected_logs = BTreeMap::<String, String>::new();
    for cursor in &trace.final_observation.log_cursors {
        if expected_logs
            .insert(cursor.log_origin_id.clone(), cursor.checkpoint.clone())
            .is_some()
        {
            push_error(
                errors,
                format!("duplicate final log cursor {}", cursor.log_origin_id),
            );
        }
    }
    let mut expected_receipts = BTreeMap::<String, ModelReceipt>::new();
    for receipt in &trace.final_observation.accepted_receipts {
        if expected_receipts
            .insert(
                receipt.receipt_id.clone(),
                ModelReceipt {
                    receipt_digest: receipt.receipt_digest.clone(),
                    acceptance_revision: parse_revision(&receipt.receipt_acceptance_revision)
                        .unwrap_or(u64::MAX),
                },
            )
            .is_some()
        {
            push_error(
                errors,
                format!("duplicate final receipt {}", receipt.receipt_id),
            );
        }
    }
    if state.revision
        != parse_revision(&trace.final_observation.authority_revision).unwrap_or(u64::MAX)
    {
        push_error(
            errors,
            "final authority revision differs from replayed revision",
        );
    }
    if state.policy_digest != trace.final_observation.policy_digest {
        push_error(
            errors,
            "final policy digest differs from replayed policy head",
        );
    }
    if state.sources != expected_sources {
        push_error(errors, "final source cursors differ from replayed state");
    }
    if state.logs != expected_logs {
        push_error(errors, "final log cursors differ from replayed state");
    }
    if state.receipts != expected_receipts {
        push_error(
            errors,
            "final accepted receipt set differs from replayed receipt history",
        );
    }
    results
}

pub fn verify_reference(trace: &Trace, trace_bytes: &[u8]) -> VerificationReport {
    let trace_hash = Sha256::digest(trace_bytes);
    let digest = format!("sha256:{}", hex_lower(&trace_hash));
    let mut evidence_errors = Vec::new();
    validate_structure(trace, &mut evidence_errors);
    validate_retry_groups(trace, &mut evidence_errors);
    validate_witnesses(trace, &mut evidence_errors);
    if !evidence_errors.is_empty() {
        return VerificationReport {
            report_version: 1,
            verifier_version: env!("CARGO_PKG_VERSION").to_string(),
            schema_sha256: None,
            claim_ceiling: "Reference trace semantics only; no backend run, cryptographic witness-signature verification, physical durability, or engine qualification.".to_string(),
            status: "invalid_evidence".to_string(),
            trace_id: trace.trace_id.clone(),
            trace_sha256: digest,
            operations_replayed: 0,
            invariants_checked: trace.invariant_expectations.len(),
            mutants_detected: 0,
            mutants_seeded: trace.negative_controls.len(),
            counterexamples: evidence_errors,
        };
    }
    let mut errors = Vec::new();
    let results = replay(trace, &mut errors);
    let mutant_results = verify_mutants(trace);
    let mut mutants_detected = 0usize;
    for control in &trace.negative_controls {
        match mutant_results
            .iter()
            .find(|(id, _)| id == &control.mutation_id)
        {
            Some((_, true)) => mutants_detected += 1,
            Some((_, false)) => push_error(
                &mut errors,
                format!("negative control {} was not detected", control.mutation_id),
            ),
            None => push_error(
                &mut errors,
                format!(
                    "negative control {} has no executable mutant",
                    control.mutation_id
                ),
            ),
        }
    }
    VerificationReport {
        report_version: 1,
        verifier_version: env!("CARGO_PKG_VERSION").to_string(),
        schema_sha256: None,
        claim_ceiling: "Reference trace semantics only; no backend run, cryptographic witness-signature verification, physical durability, or engine qualification.".to_string(),
        status: if errors.is_empty() { "pass" } else { "fail" }.to_string(),
        trace_id: trace.trace_id.clone(),
        trace_sha256: digest,
        operations_replayed: results.len(),
        invariants_checked: trace.invariant_expectations.len(),
        mutants_detected,
        mutants_seeded: trace.negative_controls.len(),
        counterexamples: errors,
    }
}

pub fn verify_mutants(trace: &Trace) -> Vec<(String, bool)> {
    let mut cases = Vec::<(String, Trace)>::new();

    let mut missing_log = trace.clone();
    if let Some(cursor) = missing_log
        .final_observation
        .log_cursors
        .iter_mut()
        .find(|x| x.log_origin_id == "log-A")
    {
        cursor.checkpoint = "la:1".to_string();
    }
    cases.push(("mutant.drop-log-cursor-write".to_string(), missing_log));

    let mut dual_winner = trace.clone();
    if let Some(op) = dual_winner
        .operations
        .iter_mut()
        .find(|x| x.op_id == "op-d-race-loser")
    {
        op.expected_result.outcome = "committed".to_string();
        op.expected_result.receipt_id = Some("receipt-D".to_string());
        op.expected_result.receipt_acceptance_revision = Some("rev-4".to_string());
    }
    cases.push(("mutant.dual-writer-accept".to_string(), dual_winner));

    let mut retry_revision = trace.clone();
    if let Some(op) = retry_revision
        .operations
        .iter_mut()
        .find(|x| x.op_id == "op-a-exact-retry")
    {
        op.expected_result.receipt_acceptance_revision = Some("rev-2".to_string());
    }
    cases.push((
        "mutant.retry-overwrites-acceptance-revision".to_string(),
        retry_revision,
    ));

    let mut stale_policy = trace.clone();
    if let Some(op) = stale_policy
        .operations
        .iter_mut()
        .find(|x| x.op_id == "op-stale-policy-rejected")
    {
        op.expected_result.outcome = "committed".to_string();
        op.expected_result.receipt_id = Some("receipt-F".to_string());
    }
    cases.push(("mutant.stale-policy-commit".to_string(), stale_policy));

    let mut partial_retry = trace.clone();
    if let Some(op) = partial_retry
        .operations
        .iter_mut()
        .find(|x| x.op_id == "op-serializable-attempt-2")
    {
        op.request.expected_log_checkpoint = Some("lb:2-mutated".to_string());
    }
    cases.push((
        "mutant.statement-only-serialization-retry".to_string(),
        partial_retry,
    ));

    let mut false_freshness = trace.clone();
    if let Some(op) = false_freshness
        .operations
        .iter_mut()
        .find(|x| x.op_id == "op-restore-old-snapshot")
    {
        op.expected_result.outcome = "committed".to_string();
    }
    cases.push((
        "mutant.restore-false-freshness".to_string(),
        false_freshness,
    ));

    cases
        .into_iter()
        .map(|(id, mutant)| {
            let mut errors = Vec::new();
            validate_structure(&mutant, &mut errors);
            validate_retry_groups(&mutant, &mut errors);
            validate_witnesses(&mutant, &mut errors);
            replay(&mutant, &mut errors);
            (id, !errors.is_empty())
        })
        .collect()
}

mod history;
pub use history::{HistorySearchConfig, HistorySearchReport, verify_observed_history};

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Trace, &'static [u8]) {
        let raw = include_bytes!("../../../benchmarks/basalt/storage-conformance-trace.v1.json");
        let trace: Trace =
            serde_json::from_slice(raw).expect("reference fixture must deserialize strictly");
        (trace, raw)
    }

    #[test]
    fn reference_trace_replays_to_expected_final_state() {
        let (trace, raw) = fixture();
        let report = verify_reference(&trace, raw);
        assert_eq!(report.status, "pass", "{:#?}", report.counterexamples);
        assert_eq!(report.operations_replayed, 12);
    }

    #[test]
    fn every_seeded_negative_control_is_detected() {
        let (trace, _) = fixture();
        let results = verify_mutants(&trace);
        assert_eq!(results.len(), trace.negative_controls.len());
        for (id, detected) in results {
            assert!(detected, "verifier failed to detect {id}");
        }
    }

    #[test]
    fn cross_runner_monotonic_clocks_are_never_compared() {
        let (mut trace, _) = fixture();
        let operation = trace
            .operations
            .iter_mut()
            .find(|x| x.op_id == "op-b-commit")
            .unwrap();
        operation.runner_id = "runner-2".to_string();
        operation.invoked_at_ns = 1;
        operation.completed_at_ns = 5;
        let report = verify_reference(&trace, b"cross-runner-clock-control");
        assert_eq!(report.status, "pass", "{:#?}", report.counterexamples);
    }

    #[test]
    fn missing_witness_is_not_freshness_evidence() {
        let (mut trace, _) = fixture();
        trace.witness_observations.clear();
        let report = verify_reference(&trace, b"missing-witness-control");
        assert_eq!(report.status, "invalid_evidence");
        assert!(report.counterexamples.iter().any(|x| x.contains("witness")));
    }

    #[test]
    fn malformed_advance_request_is_rejected_before_replay() {
        let (mut trace, _) = fixture();
        trace
            .operations
            .iter_mut()
            .find(|x| x.op_id == "op-a-commit")
            .unwrap()
            .request
            .next_log_checkpoint = None;
        let report = verify_reference(&trace, b"missing-next-log-control");
        assert_eq!(report.status, "invalid_evidence");
        assert!(
            report
                .counterexamples
                .iter()
                .any(|x| x.contains("next_log_checkpoint"))
        );
    }

    #[test]
    fn invalid_operation_interval_is_rejected() {
        let (mut trace, _) = fixture();
        let operation = trace
            .operations
            .iter_mut()
            .find(|x| x.op_id == "op-a-commit")
            .unwrap();
        operation.completed_at_ns = operation.invoked_at_ns - 1;
        let report = verify_reference(&trace, b"invalid-interval-control");
        assert_eq!(report.status, "invalid_evidence");
        assert!(
            report
                .counterexamples
                .iter()
                .any(|x| x.contains("completes before invocation"))
        );
    }

    #[test]
    fn unknown_fields_are_rejected_by_strict_deserialization() {
        let raw = include_bytes!("../../../benchmarks/basalt/storage-conformance-trace.v1.json");
        let mut value: serde_json::Value = serde_json::from_slice(raw).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("untrusted_extra".to_string(), serde_json::Value::Bool(true));
        assert!(serde_json::from_value::<Trace>(value).is_err());
    }

    #[test]
    fn broken_witness_predecessor_is_rejected() {
        let (mut trace, _) = fixture();
        trace.witness_observations[2].predecessor_witness_digest = None;
        let report = verify_reference(&trace, b"broken-witness-control");
        assert_eq!(report.status, "invalid_evidence");
    }

    #[test]
    fn well_formed_but_wrong_expected_transition_fails_semantic_replay() {
        let (mut trace, _) = fixture();
        trace
            .operations
            .iter_mut()
            .find(|x| x.op_id == "op-a-commit")
            .unwrap()
            .expected_result
            .outcome = "compare_conflict".to_string();
        let report = verify_reference(&trace, b"wrong-reference-outcome");
        assert_eq!(report.status, "fail");
        assert!(
            report
                .counterexamples
                .iter()
                .any(|x| x.contains("reducer computed"))
        );
    }

    #[test]
    fn same_runner_real_time_precedence_is_enforced() {
        let (mut trace, _) = fixture();
        // The B commit is ordered before the A commit but on the same runner,
        // while its completion precedes A's invocation: this is illegal.
        trace.reference_linearization_order.swap(0, 1);
        let report = verify_reference(&trace, b"same-runner-real-time-control");
        assert_eq!(report.status, "invalid_evidence");
        assert!(
            report
                .counterexamples
                .iter()
                .any(|x| x.contains("real-time"))
        );
    }
}
