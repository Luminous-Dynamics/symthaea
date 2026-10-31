//! Hash-chained evaluator-session custody receipts and Merkle checkpointing.
//!
//! The verifier checks internal event-chain consistency, exact plan/input binding,
//! serial lifecycle order and Merkle proofs. It does not authenticate the runner,
//! trust external checkpoints, prove wall-clock time, or establish actual execution.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const CUSTODY_PLAN_SCHEMA: &str =
    "symthaea.science_bench_refresh_execution_custody_plan.v1";
pub const CUSTODY_RECEIPT_SCHEMA: &str =
    "symthaea.science_bench_refresh_execution_custody_receipt.v1";
pub const CUSTODY_REPORT_SCHEMA: &str =
    "symthaea.science_bench_refresh_execution_custody_report.v1";

const PLAN_DOMAIN: &[u8] = b"symthaea.execution_custody.plan.v1\0";
const EVENT_DOMAIN: &[u8] = b"symthaea.execution_custody.event.v1\0";
const EVENT_ROOT_DOMAIN: &[u8] = b"symthaea.execution_custody.event_root.v1\0";
const BUNDLE_DOMAIN: &[u8] = b"symthaea.execution_custody.bundle.v1\0";
const MERKLE_LEAF_PREFIX: &[u8] = b"\0";
const MERKLE_NODE_PREFIX: &[u8] = b"\x01";
const REPORT_DOMAIN: &[u8] = b"symthaea.execution_custody.report.v1\0";
const MAX_CONDITIONS: usize = 100_000;
const MAX_EVENTS: usize = 1_000_000;
const MAX_TOOL_CALLS_PER_CONDITION: usize = 10_000;
const ZERO_DIGEST: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CustodyPurposeV1 {
    ExampleOnly,
    QualifiedRun,
}

impl CustodyPurposeV1 {
    fn qualified(self) -> bool {
        self == Self::QualifiedRun
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CustodyConditionPlanV1 {
    pub condition_id: String,
    pub planned_ordinal: u32,
    pub input_sha256: String,
    pub expected_input_bytes: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExecutionCustodyPlanFieldsV1 {
    pub schema: String,
    pub analyzer_source_revision: String,
    pub purpose: CustodyPurposeV1,
    pub run_id: String,
    pub run_nonce_sha256: String,
    pub evaluator_source_revision: String,
    pub evaluator_prompt_sha256: String,
    pub rubric_sha256: String,
    pub task_set_sha256: String,
    pub environment_sha256: String,
    pub max_parallelism: u32,
    pub conditions: Vec<CustodyConditionPlanV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExecutionCustodyPlanV1 {
    pub schema: String,
    pub analyzer_source_revision: String,
    pub purpose: CustodyPurposeV1,
    pub run_id: String,
    pub run_nonce_sha256: String,
    pub evaluator_source_revision: String,
    pub evaluator_prompt_sha256: String,
    pub rubric_sha256: String,
    pub task_set_sha256: String,
    pub environment_sha256: String,
    pub max_parallelism: u32,
    pub conditions: Vec<CustodyConditionPlanV1>,
    pub plan_commitment_sha256: String,
}

impl ExecutionCustodyPlanV1 {
    fn fields(&self) -> ExecutionCustodyPlanFieldsV1 {
        ExecutionCustodyPlanFieldsV1 {
            schema: self.schema.clone(),
            analyzer_source_revision: self.analyzer_source_revision.clone(),
            purpose: self.purpose,
            run_id: self.run_id.clone(),
            run_nonce_sha256: self.run_nonce_sha256.clone(),
            evaluator_source_revision: self.evaluator_source_revision.clone(),
            evaluator_prompt_sha256: self.evaluator_prompt_sha256.clone(),
            rubric_sha256: self.rubric_sha256.clone(),
            task_set_sha256: self.task_set_sha256.clone(),
            environment_sha256: self.environment_sha256.clone(),
            max_parallelism: self.max_parallelism,
            conditions: self.conditions.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CustodyTerminalStatusV1 {
    Completed,
    TimedOut,
    Blocked,
    InvalidReceipt,
    NotStarted,
    Failed,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "event_type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExecutionEventPayloadV1 {
    RunStarted {
        run_id: String,
        plan_sha256: String,
        runner_id_sha256: String,
        environment_sha256: String,
    },
    InputRendered {
        condition_id: String,
        planned_ordinal: u32,
        input_sha256: String,
        input_bytes: u64,
    },
    EvaluationStarted {
        condition_id: String,
        planned_ordinal: u32,
        evaluator_source_revision: String,
        evaluator_prompt_sha256: String,
    },
    ToolCall {
        condition_id: String,
        tool_name_sha256: String,
        request_sha256: String,
        request_bytes: u64,
        response_sha256: Option<String>,
        response_bytes: u64,
    },
    EvaluationFinished {
        condition_id: String,
        planned_ordinal: u32,
        status: CustodyTerminalStatusV1,
        output_sha256: Option<String>,
        output_bytes: u64,
        evaluator_call_count: u64,
        evaluator_token_count: u64,
    },
    RunFinished {
        run_id: String,
        planned_condition_count: u64,
        completed_condition_count: u64,
        terminal_condition_count: u64,
        run_status: String,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExecutionEventFieldsV1 {
    pub sequence: u64,
    pub timestamp_unix_milliseconds: u64,
    pub previous_event_sha256: String,
    pub payload: ExecutionEventPayloadV1,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExecutionEventV1 {
    pub sequence: u64,
    pub timestamp_unix_milliseconds: u64,
    pub previous_event_sha256: String,
    pub payload: ExecutionEventPayloadV1,
    pub event_sha256: String,
}

impl ExecutionEventV1 {
    fn fields(&self) -> ExecutionEventFieldsV1 {
        ExecutionEventFieldsV1 {
            sequence: self.sequence,
            timestamp_unix_milliseconds: self.timestamp_unix_milliseconds,
            previous_event_sha256: self.previous_event_sha256.clone(),
            payload: self.payload.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MerkleInclusionProofV1 {
    pub leaf_index: u64,
    pub tree_size: u64,
    pub sibling_sha256: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExternalCheckpointStatementV1 {
    pub provider: String,
    pub log_id: String,
    pub record_id: String,
    pub bundle_sha256: String,
    pub checkpoint_root_sha256: String,
    pub tree_size: u64,
    pub anchored_at_unix_milliseconds: u64,
    pub inclusion_proof: MerkleInclusionProofV1,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExecutionCustodyReceiptV1 {
    pub schema: String,
    pub analyzer_source_revision: String,
    pub purpose: CustodyPurposeV1,
    pub run_id: String,
    pub plan_sha256: String,
    pub plan_commitment_sha256: String,
    pub evaluator_source_revision: String,
    pub evaluator_prompt_sha256: String,
    pub rubric_sha256: String,
    pub environment_sha256: String,
    pub events: Vec<ExecutionEventV1>,
    pub external_checkpoint: Option<ExternalCheckpointStatementV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ConditionCustodySummaryV1 {
    pub condition_id: String,
    pub planned_ordinal: u32,
    pub input_rendered: bool,
    pub evaluation_started: bool,
    pub terminal_status: CustodyTerminalStatusV1,
    pub tool_call_count: u64,
    pub input_sha256: Option<String>,
    pub output_sha256: Option<String>,
    pub evaluator_call_count: u64,
    pub evaluator_token_count: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ExecutionCustodyReportV1 {
    pub fields: ExecutionCustodyReportFieldsV1,
    pub report_commitment_sha256: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest_domain(domain: &[u8], payload: &[u8]) -> Result<String, String> {
    let length = u64::try_from(payload.len())
        .map_err(|_| "domain-separated payload exceeds u64 length".to_owned())?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(length.to_be_bytes());
    hasher.update(payload);
    Ok(sha256_hex(&hasher.finalize()))
}

fn valid_digest(value: &str, allow_zero: bool) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && (allow_zero || value != ZERO_DIGEST)
}

fn valid_revision(value: &str) -> bool {
    value.len() == 40
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && value.bytes().any(|byte| byte != b'0')
}

fn nonempty(value: &str, max_bytes: usize) -> bool {
    !value.trim().is_empty() && value.len() <= max_bytes
}

fn plan_commitment(fields: &ExecutionCustodyPlanFieldsV1) -> Result<String, String> {
    let bytes = serde_json::to_vec(fields)
        .map_err(|error| format!("cannot serialize execution custody plan: {error}"))?;
    digest_domain(PLAN_DOMAIN, &bytes)
}

pub fn validate_execution_custody_plan_fields(
    fields: &ExecutionCustodyPlanFieldsV1,
) -> Result<(), Vec<String>> {
    let allow_zero = fields.purpose == CustodyPurposeV1::ExampleOnly;
    let mut issues = Vec::new();
    if fields.schema != CUSTODY_PLAN_SCHEMA {
        issues.push("unsupported execution custody plan schema".to_owned());
    }
    if !valid_revision(&fields.analyzer_source_revision)
        || fields.analyzer_source_revision != crate::BUILT_GENERATOR_SOURCE_REVISION
    {
        issues.push("analyzer source revision must match exact compiled validator revision".to_owned());
    }
    if fields.purpose.qualified() && crate::BUILT_GENERATOR_WORKTREE_CLEAN != "true" {
        issues.push("qualified custody plan requires clean analyzer worktree".to_owned());
    }
    if !nonempty(&fields.run_id, 128) {
        issues.push("run_id must be non-empty and at most 128 bytes".to_owned());
    }
    for (label, digest) in [
        ("run_nonce_sha256", &fields.run_nonce_sha256),
        ("evaluator_prompt_sha256", &fields.evaluator_prompt_sha256),
        ("rubric_sha256", &fields.rubric_sha256),
        ("task_set_sha256", &fields.task_set_sha256),
        ("environment_sha256", &fields.environment_sha256),
    ] {
        if !valid_digest(digest, allow_zero) {
            issues.push(format!("{label} must be a valid SHA-256 digest"));
        }
    }
    if !valid_revision(&fields.evaluator_source_revision) {
        issues.push("evaluator_source_revision must be a full nonzero lowercase Git revision".to_owned());
    }
    if fields.max_parallelism != 1 {
        issues.push("v1 custody plan requires serial max_parallelism=1".to_owned());
    }
    if fields.conditions.is_empty() || fields.conditions.len() > MAX_CONDITIONS {
        issues.push(format!("conditions must contain 1..={MAX_CONDITIONS} entries"));
    }
    let mut ids = BTreeSet::new();
    for (index, condition) in fields.conditions.iter().enumerate() {
        if !nonempty(&condition.condition_id, 128) || !ids.insert(condition.condition_id.as_str()) {
            issues.push("condition IDs must be non-empty and unique".to_owned());
        }
        if condition.planned_ordinal as usize != index {
            issues.push(format!("condition {} planned_ordinal must equal its schedule index", condition.condition_id));
        }
        if !valid_digest(&condition.input_sha256, allow_zero) {
            issues.push(format!("condition {} has invalid input_sha256", condition.condition_id));
        }
        if condition.expected_input_bytes == 0 {
            issues.push(format!("condition {} expected_input_bytes must be positive", condition.condition_id));
        }
    }
    if issues.is_empty() { Ok(()) } else { Err(issues) }
}

pub fn compute_execution_custody_plan_commitment(fields_bytes: &[u8]) -> Result<String, String> {
    let fields: ExecutionCustodyPlanFieldsV1 = serde_json::from_slice(fields_bytes)
        .map_err(|error| format!("execution custody plan fields invalid: {error}"))?;
    validate_execution_custody_plan_fields(&fields)
        .map_err(|issues| issues.join("; "))?;
    plan_commitment(&fields)
}

pub fn verify_execution_custody_plan(
    plan_bytes: &[u8],
) -> Result<(ExecutionCustodyPlanV1, String), String> {
    let plan: ExecutionCustodyPlanV1 = serde_json::from_slice(plan_bytes)
        .map_err(|error| format!("execution custody plan invalid: {error}"))?;
    let fields = plan.fields();
    validate_execution_custody_plan_fields(&fields)
        .map_err(|issues| issues.join("; "))?;
    let expected = plan_commitment(&fields)?;
    if !valid_digest(&plan.plan_commitment_sha256, !plan.purpose.qualified())
        || expected != plan.plan_commitment_sha256
    {
        return Err("execution custody plan commitment mismatch".to_owned());
    }
    Ok((plan, sha256_hex(plan_bytes)))
}

fn event_hash(fields: &ExecutionEventFieldsV1) -> Result<String, String> {
    let bytes = serde_json::to_vec(fields)
        .map_err(|error| format!("cannot serialize execution event: {error}"))?;
    digest_domain(EVENT_DOMAIN, &bytes)
}

fn merkle_leaf(event_sha256: &str) -> Result<String, String> {
    let mut bytes = Vec::with_capacity(1 + 32);
    bytes.extend_from_slice(MERKLE_LEAF_PREFIX);
    let raw = decode_digest(event_sha256)?;
    bytes.extend_from_slice(&raw);
    Ok(sha256_hex(&bytes))
}

fn merkle_node(left_sha256: &str, right_sha256: &str) -> Result<String, String> {
    let mut bytes = Vec::with_capacity(1 + 64);
    bytes.extend_from_slice(MERKLE_NODE_PREFIX);
    bytes.extend_from_slice(&decode_digest(left_sha256)?);
    bytes.extend_from_slice(&decode_digest(right_sha256)?);
    Ok(sha256_hex(&bytes))
}

fn decode_digest(value: &str) -> Result<Vec<u8>, String> {
    if !valid_digest(value, false) {
        return Err("invalid SHA-256 digest in Merkle operation".to_owned());
    }
    (0..64).step_by(2).map(|index| {
        u8::from_str_radix(&value[index..index+2], 16)
            .map_err(|_| "invalid hex digest".to_owned())
    }).collect()
}

fn merkle_levels(event_hashes: &[String]) -> Result<Vec<Vec<String>>, String> {
    if event_hashes.is_empty() {
        return Err("cannot build Merkle tree from zero events".to_owned());
    }
    let mut levels = vec![event_hashes.iter().map(|value| merkle_leaf(value)).collect::<Result<Vec<_>,_>>()?];
    while levels.last().unwrap().len() > 1 {
        let current = levels.last().unwrap();
        let mut next = Vec::with_capacity(current.len().div_ceil(2));
        for pair in current.chunks(2) {
            let right = pair.get(1).unwrap_or(&pair[0]);
            next.push(merkle_node(&pair[0], right)?);
        }
        levels.push(next);
    }
    Ok(levels)
}

pub fn compute_execution_merkle_root(event_hashes: &[String]) -> Result<String, String> {
    Ok(merkle_levels(event_hashes)?.last().unwrap()[0].clone())
}

pub fn compute_merkle_inclusion_proof(
    event_hashes: &[String],
    leaf_index: usize,
) -> Result<MerkleInclusionProofV1, String> {
    if leaf_index >= event_hashes.len() {
        return Err("Merkle inclusion index is out of range".to_owned());
    }
    let levels = merkle_levels(event_hashes)?;
    let mut index = leaf_index;
    let mut siblings = Vec::new();
    for level in levels.iter().take(levels.len().saturating_sub(1)) {
        let sibling_index = if index % 2 == 1 { index - 1 } else if index + 1 < level.len() { index + 1 } else { index };
        siblings.push(level[sibling_index].clone());
        index /= 2;
    }
    Ok(MerkleInclusionProofV1 {
        leaf_index: leaf_index as u64,
        tree_size: event_hashes.len() as u64,
        sibling_sha256: siblings,
    })
}

pub fn verify_merkle_inclusion_proof(
    leaf_sha256: &str,
    proof: &MerkleInclusionProofV1,
    expected_root_sha256: &str,
) -> Result<bool, String> {
    if !valid_digest(leaf_sha256, false)
        || !valid_digest(expected_root_sha256, false)
        || proof.tree_size == 0
        || proof.leaf_index >= proof.tree_size
    {
        return Err("invalid Merkle inclusion proof inputs".to_owned());
    }
    let mut current = merkle_leaf(leaf_sha256)?;
    let mut index = proof.leaf_index;
    let mut width = proof.tree_size;
    let mut proof_index = 0usize;
    while width > 1 {
        let sibling = proof.sibling_sha256.get(proof_index)
            .ok_or_else(|| "Merkle inclusion proof is missing a sibling".to_owned())?;
        if !valid_digest(sibling, false) {
            return Err("Merkle inclusion proof contains invalid sibling digest".to_owned());
        }
        current = if index % 2 == 1 {
            merkle_node(sibling, &current)?
        } else {
            merkle_node(&current, sibling)?
        };
        index /= 2;
        width = width.div_ceil(2);
        proof_index += 1;
    }
    if proof_index != proof.sibling_sha256.len() {
        return Err("Merkle inclusion proof has extra siblings".to_owned());
    }
    Ok(current == expected_root_sha256)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConditionStageV1 {
    Planned,
    InputRendered,
    Started,
    Terminal,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ExecutionCustodyReportFieldsV1 {
    pub schema: String,
    pub run_id: String,
    pub analyzer_source_revision: String,
    pub analyzer_worktree_clean: bool,
    pub plan_sha256: String,
    pub receipt_sha256: String,
    pub plan_commitment_sha256: String,
    pub event_count: u64,
    pub event_merkle_root_sha256: String,
    pub execution_bundle_sha256: String,
    pub planned_condition_count: u64,
    pub observed_condition_count: u64,
    pub completed_condition_count: u64,
    pub timed_out_condition_count: u64,
    pub blocked_condition_count: u64,
    pub invalid_receipt_condition_count: u64,
    pub not_started_condition_count: u64,
    pub failed_condition_count: u64,
    pub tool_call_count: u64,
    pub total_evaluator_call_count: u64,
    pub total_evaluator_token_count: u64,
    pub total_rendered_input_bytes: u64,
    pub total_output_bytes: u64,
    pub timestamps_monotonic: bool,
    pub event_sequence_contiguous: bool,
    pub event_chain_valid: bool,
    pub external_checkpoint_present: bool,
    pub external_checkpoint_matches_bundle: bool,
    pub external_inclusion_proof_valid: bool,
    pub external_checkpoint_authenticity_proven: bool,
    pub runner_identity_authenticated: bool,
    pub raw_prompt_custody_proven: bool,
    pub evaluator_execution_proven: bool,
    pub wall_clock_authenticity_proven: bool,
    pub external_anchor_required: bool,
    pub scoring_eligible_declared: bool,
    pub conditions: Vec<ConditionCustodySummaryV1>,
    pub claim_boundary: String,
}

pub fn analyze_execution_custody(
    plan_bytes: &[u8],
    receipt_bytes: &[u8],
) -> Result<ExecutionCustodyReportV1, String> {
    let (plan, plan_sha256) = verify_execution_custody_plan(plan_bytes)?;
    let receipt: ExecutionCustodyReceiptV1 = serde_json::from_slice(receipt_bytes)
        .map_err(|error| format!("execution custody receipt invalid: {error}"))?;
    if receipt.schema != CUSTODY_RECEIPT_SCHEMA {
        return Err("unsupported execution custody receipt schema".to_owned());
    }
    if receipt.analyzer_source_revision != plan.analyzer_source_revision
        || receipt.analyzer_source_revision != crate::BUILT_GENERATOR_SOURCE_REVISION
    {
        return Err("receipt analyzer revision differs from exact compiled validator revision".to_owned());
    }
    if receipt.purpose != plan.purpose
        || receipt.run_id != plan.run_id
        || receipt.plan_sha256 != plan_sha256
        || receipt.plan_commitment_sha256 != plan.plan_commitment_sha256
    {
        return Err("receipt does not bind exact plan bytes/run identity".to_owned());
    }
    if receipt.evaluator_source_revision != plan.evaluator_source_revision
        || receipt.evaluator_prompt_sha256 != plan.evaluator_prompt_sha256
        || receipt.rubric_sha256 != plan.rubric_sha256
        || receipt.environment_sha256 != plan.environment_sha256
    {
        return Err("receipt evaluator/prompt/rubric/environment differs from frozen plan".to_owned());
    }
    if receipt.events.len() < 2 || receipt.events.len() > MAX_EVENTS {
        return Err(format!("receipt events must contain 2..={MAX_EVENTS} records"));
    }

    let mut previous_event = ZERO_DIGEST.to_owned();
    let mut previous_timestamp = 0u64;
    let mut event_hashes = Vec::with_capacity(receipt.events.len());
    let mut sequence_valid = true;
    let mut chain_valid = true;
    let mut timestamps_monotonic = true;
    for (index, event) in receipt.events.iter().enumerate() {
        if event.sequence != index as u64 {
            sequence_valid = false;
            return Err(format!("event sequence gap or reorder at index {index}"));
        }
        if event.timestamp_unix_milliseconds < previous_timestamp {
            timestamps_monotonic = false;
            return Err(format!("event timestamps are not monotonic at sequence {index}"));
        }
        if event.previous_event_sha256 != previous_event {
            chain_valid = false;
            return Err(format!("event {} previous hash does not bind prior event", event.sequence));
        }
        let expected_hash = event_hash(&event.fields())?;
        if event.event_sha256 != expected_hash {
            chain_valid = false;
            return Err(format!("event {} event hash mismatch", event.sequence));
        }
        previous_event = event.event_sha256.clone();
        previous_timestamp = event.timestamp_unix_milliseconds;
        event_hashes.push(event.event_sha256.clone());
    }

    let first = receipt.events.first().unwrap();
    let last = receipt.events.last().unwrap();
    match &first.payload {
        ExecutionEventPayloadV1::RunStarted {run_id, plan_sha256: event_plan, runner_id_sha256, environment_sha256} => {
            if run_id != &plan.run_id || event_plan != &plan_sha256 || !valid_digest(runner_id_sha256, false) || environment_sha256 != &plan.environment_sha256 {
                return Err("run-start event does not bind the plan/run/runner/environment".to_owned());
            }
        }
        _ => return Err("first event must be run_started".to_owned()),
    }
    let run_finished_completed_count = match &last.payload {
        ExecutionEventPayloadV1::RunFinished {run_id, planned_condition_count, completed_condition_count, terminal_condition_count, run_status} => {
            if run_id != &plan.run_id
                || *planned_condition_count != plan.conditions.len() as u64
                || *terminal_condition_count != plan.conditions.len() as u64
                || !nonempty(run_status, 64)
            {
                return Err("run-finished event does not account for all planned conditions".to_owned());
            }
            *completed_condition_count
        }
        _ => return Err("last event must be run_finished".to_owned()),
    };
    if receipt.events.iter().enumerate().any(|(index,event)| matches!(&event.payload, ExecutionEventPayloadV1::RunStarted {..} if index != 0) || matches!(&event.payload, ExecutionEventPayloadV1::RunFinished {..} if index+1 != receipt.events.len())) {
        return Err("run start/finish event is duplicated or misplaced".to_owned());
    }

    let conditions_by_id: BTreeMap<&str, &CustodyConditionPlanV1> =
        plan.conditions.iter().map(|condition| (condition.condition_id.as_str(), condition)).collect();
    let mut stages: BTreeMap<String, ConditionStageV1> = plan.conditions.iter()
        .map(|condition| (condition.condition_id.clone(), ConditionStageV1::Planned)).collect();
    let mut summary: BTreeMap<String, ConditionCustodySummaryV1> = plan.conditions.iter().map(|condition| {
        (condition.condition_id.clone(), ConditionCustodySummaryV1 {
            condition_id: condition.condition_id.clone(),
            planned_ordinal: condition.planned_ordinal,
            input_rendered: false,
            evaluation_started: false,
            terminal_status: CustodyTerminalStatusV1::NotStarted,
            tool_call_count: 0,
            input_sha256: None,
            output_sha256: None,
            evaluator_call_count: 0,
            evaluator_token_count: 0,
        })
    }).collect();
    let mut current_ordinal = 0u32;
    let mut terminal_seen = BTreeSet::new();
    let mut total_tool_calls = 0u64;
    let mut total_evaluator_calls = 0u64;
    let mut total_evaluator_tokens = 0u64;
    let mut total_rendered_input_bytes = 0u64;
    let mut total_output_bytes = 0u64;
    for event in receipt.events.iter().skip(1).take(receipt.events.len().saturating_sub(2)) {
        match &event.payload {
            ExecutionEventPayloadV1::InputRendered {condition_id,planned_ordinal,input_sha256,input_bytes} => {
                let condition = conditions_by_id.get(condition_id.as_str()).ok_or_else(|| format!("input-rendered event references unknown condition {condition_id}"))?;
                if condition.planned_ordinal != *planned_ordinal || current_ordinal != *planned_ordinal {
                    return Err(format!("condition {condition_id} rendered outside its committed serial schedule"));
                }
                if stages.get(condition_id) != Some(&ConditionStageV1::Planned) {
                    return Err(format!("condition {condition_id} input rendered more than once"));
                }
                if input_sha256 != &condition.input_sha256 || *input_bytes != condition.expected_input_bytes {
                    return Err(format!("condition {condition_id} rendered input differs from plan commitment"));
                }
                stages.insert(condition_id.clone(), ConditionStageV1::InputRendered);
                total_rendered_input_bytes = total_rendered_input_bytes.checked_add(*input_bytes)
                    .ok_or_else(|| "rendered input byte count overflow".to_owned())?;
                let row=summary.get_mut(condition_id).unwrap();row.input_rendered=true;row.input_sha256=Some(input_sha256.clone());
            }
            ExecutionEventPayloadV1::EvaluationStarted {condition_id,planned_ordinal,evaluator_source_revision,evaluator_prompt_sha256} => {
                if conditions_by_id.get(condition_id.as_str()).is_none_or(|condition| condition.planned_ordinal != *planned_ordinal || *planned_ordinal != current_ordinal) {
                    return Err(format!("condition {condition_id} evaluation start violates committed schedule"));
                }
                if stages.get(condition_id) != Some(&ConditionStageV1::InputRendered) {
                    return Err(format!("condition {condition_id} started without exactly one rendered input"));
                }
                if evaluator_source_revision != &plan.evaluator_source_revision || evaluator_prompt_sha256 != &plan.evaluator_prompt_sha256 {
                    return Err(format!("condition {condition_id} evaluator revision or prompt differs from plan"));
                }
                stages.insert(condition_id.clone(), ConditionStageV1::Started);
                summary.get_mut(condition_id).unwrap().evaluation_started=true;
            }
            ExecutionEventPayloadV1::ToolCall {condition_id,tool_name_sha256,request_sha256,request_bytes,response_sha256,response_bytes} => {
                if stages.get(condition_id) != Some(&ConditionStageV1::Started) {
                    return Err(format!("tool call for {condition_id} occurred outside active evaluation"));
                }
                if !valid_digest(tool_name_sha256, false) || !valid_digest(request_sha256, false) || *request_bytes == 0 {
                    return Err(format!("tool call for {condition_id} has invalid request/tool commitment"));
                }
                if response_sha256.as_ref().is_some_and(|value| !valid_digest(value, false))
                    || (response_sha256.is_none() && *response_bytes > 0)
                    || (response_sha256.is_some() && *response_bytes == 0)
                {
                    return Err(format!("tool call for {condition_id} has inconsistent response commitment"));
                }
                let row=summary.get_mut(condition_id).unwrap();
                row.tool_call_count = row.tool_call_count.checked_add(1).ok_or_else(|| "tool-call count overflow".to_owned())?;
                total_tool_calls=total_tool_calls.checked_add(1).ok_or_else(|| "total tool-call count overflow".to_owned())?;
                if row.tool_call_count as usize > MAX_TOOL_CALLS_PER_CONDITION {
                    return Err(format!("condition {condition_id} exceeds tool call limit"));
                }
            }
            ExecutionEventPayloadV1::EvaluationFinished {condition_id,planned_ordinal,status,output_sha256,output_bytes,evaluator_call_count,evaluator_token_count} => {
                if terminal_seen.contains(condition_id) {
                    return Err(format!("condition {condition_id} has duplicate terminal event"));
                }
                let condition = conditions_by_id.get(condition_id.as_str()).ok_or_else(|| format!("terminal event references unknown condition {condition_id}"))?;
                if condition.planned_ordinal != *planned_ordinal || *planned_ordinal != current_ordinal {
                    return Err(format!("condition {condition_id} terminal event violates committed serial schedule"));
                }
                match status {
                    CustodyTerminalStatusV1::NotStarted | CustodyTerminalStatusV1::Blocked => {
                        if stages.get(condition_id) != Some(&ConditionStageV1::Planned) || output_sha256.is_some() || *output_bytes != 0 || *evaluator_call_count != 0 || *evaluator_token_count != 0 {
                            return Err(format!("condition {condition_id} blocked/not-started status carries execution outputs"));
                        }
                    }
                    CustodyTerminalStatusV1::Completed | CustodyTerminalStatusV1::TimedOut | CustodyTerminalStatusV1::InvalidReceipt | CustodyTerminalStatusV1::Failed => {
                        if stages.get(condition_id) != Some(&ConditionStageV1::Started) {
                            return Err(format!("condition {condition_id} terminal status requires input and evaluation start"));
                        }
                        if *status == CustodyTerminalStatusV1::Completed {
                            if !output_sha256.as_ref().is_some_and(|value| valid_digest(value, false)) || *output_bytes == 0 {
                                return Err(format!("completed condition {condition_id} requires output digest and positive byte length"));
                            }
                        } else if output_sha256.is_some() || *output_bytes != 0 {
                            return Err(format!("non-completed condition {condition_id} must not carry a completed output"));
                        }
                    }
                }
                if output_sha256.as_ref().is_some_and(|value| !valid_digest(value, false)) {
                    return Err(format!("condition {condition_id} has malformed output digest"));
                }
                total_output_bytes = total_output_bytes.checked_add(*output_bytes)
                    .ok_or_else(|| "output byte count overflow".to_owned())?;
                let row=summary.get_mut(condition_id).unwrap();
                row.terminal_status=*status;row.output_sha256=output_sha256.clone();
                row.evaluator_call_count=*evaluator_call_count;row.evaluator_token_count=*evaluator_token_count;
                total_evaluator_calls=total_evaluator_calls.checked_add(*evaluator_call_count).ok_or_else(|| "evaluator call count overflow".to_owned())?;
                total_evaluator_tokens=total_evaluator_tokens.checked_add(*evaluator_token_count).ok_or_else(|| "evaluator token count overflow".to_owned())?;
                stages.insert(condition_id.clone(),ConditionStageV1::Terminal);
                terminal_seen.insert(condition_id.clone());
                current_ordinal=current_ordinal.checked_add(1).ok_or_else(|| "condition ordinal overflow".to_owned())?;
            }
            ExecutionEventPayloadV1::RunStarted {..} | ExecutionEventPayloadV1::RunFinished {..} => {
                return Err("run boundary event appeared inside session body".to_owned());
            }
        }
    }
    if terminal_seen.len()!=plan.conditions.len() {
        return Err(format!("session has {} terminal conditions; expected {}",terminal_seen.len(),plan.conditions.len()));
    }
    let completed = summary.values().filter(|row| row.terminal_status==CustodyTerminalStatusV1::Completed).count() as u64;
    let timed_out = summary.values().filter(|row| row.terminal_status==CustodyTerminalStatusV1::TimedOut).count() as u64;
    let blocked = summary.values().filter(|row| row.terminal_status==CustodyTerminalStatusV1::Blocked).count() as u64;
    let invalid = summary.values().filter(|row| row.terminal_status==CustodyTerminalStatusV1::InvalidReceipt).count() as u64;
    let not_started = summary.values().filter(|row| row.terminal_status==CustodyTerminalStatusV1::NotStarted).count() as u64;
    let failed = summary.values().filter(|row| row.terminal_status==CustodyTerminalStatusV1::Failed).count() as u64;
    if completed+timed_out+blocked+invalid+not_started+failed != plan.conditions.len() as u64 {
        return Err("terminal status counts do not sum to planned conditions".to_owned());
    }
    if run_finished_completed_count != completed {
        return Err("run-finished completed count differs from terminal event accounting".to_owned());
    }
    let event_hashes:Vec<String>=receipt.events.iter().map(|event|event.event_sha256.clone()).collect();
    let event_root=compute_execution_merkle_root(&event_hashes)?;
    let bundle_payload=serde_json::to_vec(&(
        &plan_sha256,
        &plan.plan_commitment_sha256,
        &event_root,
        receipt.events.len() as u64,
        &plan.run_id,
    )).map_err(|error| format!("cannot serialize execution bundle identity: {error}"))?;
    let bundle_sha256=digest_domain(BUNDLE_DOMAIN,&bundle_payload)?;
    let checkpoint_present=receipt.external_checkpoint.is_some();
    let (checkpoint_matches,inclusion_valid)=if let Some(anchor)=&receipt.external_checkpoint {
        if !nonempty(&anchor.provider,128)||!nonempty(&anchor.log_id,256)||!nonempty(&anchor.record_id,256) {
            return Err("external checkpoint provider/log/record identifiers must be non-empty".to_owned());
        }
        if anchor.bundle_sha256!=bundle_sha256
            || anchor.tree_size==0
            || anchor.tree_size!=anchor.inclusion_proof.tree_size
            || anchor.anchored_at_unix_milliseconds==0
        {
            return Err("external checkpoint does not bind execution bundle/tree size/time".to_owned());
        }
        let proof_valid=verify_merkle_inclusion_proof(&anchor.bundle_sha256,&anchor.inclusion_proof,&anchor.checkpoint_root_sha256)?;
        if !proof_valid {
            return Err("external checkpoint inclusion proof does not match supplied root".to_owned());
        }
        (true,true)
    } else {(false,false)};
    let fields=ExecutionCustodyReportFieldsV1 {
        schema:CUSTODY_REPORT_SCHEMA.to_owned(),
        run_id:plan.run_id,
        analyzer_source_revision:plan.analyzer_source_revision,
        analyzer_worktree_clean:crate::BUILT_GENERATOR_WORKTREE_CLEAN=="true",
        plan_sha256,
        receipt_sha256:sha256_hex(receipt_bytes),
        plan_commitment_sha256:plan.plan_commitment_sha256,
        event_count:receipt.events.len() as u64,
        event_merkle_root_sha256:event_root,
        execution_bundle_sha256:bundle_sha256,
        planned_condition_count:plan.conditions.len() as u64,
        observed_condition_count:terminal_seen.len() as u64,
        completed_condition_count:completed,
        timed_out_condition_count:timed_out,
        blocked_condition_count:blocked,
        invalid_receipt_condition_count:invalid,
        not_started_condition_count:not_started,
        failed_condition_count:failed,
        tool_call_count:total_tool_calls,
        total_evaluator_call_count:total_evaluator_calls,
        total_evaluator_token_count:total_evaluator_tokens,
        total_rendered_input_bytes,
        total_output_bytes,
        timestamps_monotonic,
        event_sequence_contiguous:sequence_valid,
        event_chain_valid:chain_valid,
        external_checkpoint_present:checkpoint_present,
        external_checkpoint_matches_bundle:checkpoint_matches,
        external_inclusion_proof_valid:inclusion_valid,
        external_checkpoint_authenticity_proven:false,
        runner_identity_authenticated:false,
        raw_prompt_custody_proven:false,
        evaluator_execution_proven:false,
        wall_clock_authenticity_proven:false,
        external_anchor_required:true,
        scoring_eligible_declared:false,
        conditions:plan.conditions.iter().map(|condition|summary.get(&condition.condition_id).unwrap().clone()).collect(),
        claim_boundary:"The verifier validates a supplied ordered event chain, plan/input bindings and local Merkle inclusion proof. A self-supplied checkpoint root is not authenticated: runner identity, event execution, raw bytes custody, wall-clock authenticity and external checkpoint authenticity remain unproven. The report does not grant scoring eligibility.".to_owned(),
    };
    let payload=serde_json::to_vec(&fields).map_err(|error| format!("cannot serialize execution custody report: {error}"))?;
    let commitment=digest_domain(REPORT_DOMAIN,&payload)?;
    Ok(ExecutionCustodyReportV1 {fields,report_commitment_sha256:commitment})
}

pub fn verify_execution_custody_report(
    report_bytes: &[u8],
) -> Result<ExecutionCustodyReportV1, String> {
    let report: ExecutionCustodyReportV1 = serde_json::from_slice(report_bytes)
        .map_err(|error| format!("execution custody report invalid: {error}"))?;
    if !valid_digest(&report.report_commitment_sha256, false) {
        return Err("execution custody report commitment must be a nonzero SHA-256 digest".to_owned());
    }
    let payload = serde_json::to_vec(&report.fields)
        .map_err(|error| format!("cannot serialize execution custody report fields: {error}"))?;
    let expected = digest_domain(REPORT_DOMAIN, &payload)?;
    if expected != report.report_commitment_sha256 {
        return Err("execution custody report commitment mismatch".to_owned());
    }
    Ok(report)
}


#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String {
        sha256_hex(label.as_bytes())
    }

    fn push_event(
        events: &mut Vec<ExecutionEventV1>,
        timestamp: u64,
        payload: ExecutionEventPayloadV1,
    ) {
        let sequence = events.len() as u64;
        let previous = events.last().map(|event| event.event_sha256.clone())
            .unwrap_or_else(|| ZERO_DIGEST.to_owned());
        let fields = ExecutionEventFieldsV1 {
            sequence,
            timestamp_unix_milliseconds: timestamp,
            previous_event_sha256: previous.clone(),
            payload: payload.clone(),
        };
        let hash = event_hash(&fields).unwrap();
        events.push(ExecutionEventV1 {
            sequence,
            timestamp_unix_milliseconds: timestamp,
            previous_event_sha256: previous,
            payload,
            event_sha256: hash,
        });
    }

    fn make_plan() -> (ExecutionCustodyPlanV1, Vec<u8>) {
        let fields = ExecutionCustodyPlanFieldsV1 {
            schema: CUSTODY_PLAN_SCHEMA.to_owned(),
            analyzer_source_revision: crate::BUILT_GENERATOR_SOURCE_REVISION.to_owned(),
            purpose: CustodyPurposeV1::ExampleOnly,
            run_id: "custody-run-001".to_owned(),
            run_nonce_sha256: digest("run nonce"),
            evaluator_source_revision: "b".repeat(40),
            evaluator_prompt_sha256: digest("evaluator prompt"),
            rubric_sha256: digest("rubric"),
            task_set_sha256: digest("task set"),
            environment_sha256: digest("execution environment"),
            max_parallelism: 1,
            conditions: vec![CustodyConditionPlanV1 {
                condition_id: "condition-001".to_owned(),
                planned_ordinal: 0,
                input_sha256: digest("rendered input"),
                expected_input_bytes: 13,
            }],
        };
        let commitment = plan_commitment(&fields).unwrap();
        let plan = ExecutionCustodyPlanV1 {
            schema: fields.schema.clone(),
            analyzer_source_revision: fields.analyzer_source_revision.clone(),
            purpose: fields.purpose,
            run_id: fields.run_id.clone(),
            run_nonce_sha256: fields.run_nonce_sha256.clone(),
            evaluator_source_revision: fields.evaluator_source_revision.clone(),
            evaluator_prompt_sha256: fields.evaluator_prompt_sha256.clone(),
            rubric_sha256: fields.rubric_sha256.clone(),
            task_set_sha256: fields.task_set_sha256.clone(),
            environment_sha256: fields.environment_sha256.clone(),
            max_parallelism: fields.max_parallelism,
            conditions: fields.conditions.clone(),
            plan_commitment_sha256: commitment,
        };
        let bytes = serde_json::to_vec_pretty(&plan).unwrap();
        (plan, bytes)
    }

    fn rehash_events(events: &mut [ExecutionEventV1]) {
        let mut previous = ZERO_DIGEST.to_owned();
        for (index, event) in events.iter_mut().enumerate() {
            event.sequence = index as u64;
            event.previous_event_sha256 = previous.clone();
            event.event_sha256 = event_hash(&event.fields()).unwrap();
            previous = event.event_sha256.clone();
        }
    }

    fn make_receipt(with_anchor: bool) -> (Vec<u8>, Vec<u8>) {
        let (plan, plan_bytes) = make_plan();
        let plan_sha = sha256_hex(&plan_bytes);
        let condition = &plan.conditions[0];
        let mut events = Vec::new();
        push_event(
            &mut events,
            1_000,
            ExecutionEventPayloadV1::RunStarted {
                run_id: plan.run_id.clone(),
                plan_sha256: plan_sha.clone(),
                runner_id_sha256: digest("runner identity"),
                environment_sha256: plan.environment_sha256.clone(),
            },
        );
        push_event(
            &mut events,
            1_001,
            ExecutionEventPayloadV1::InputRendered {
                condition_id: condition.condition_id.clone(),
                planned_ordinal: condition.planned_ordinal,
                input_sha256: condition.input_sha256.clone(),
                input_bytes: condition.expected_input_bytes,
            },
        );
        push_event(
            &mut events,
            1_002,
            ExecutionEventPayloadV1::EvaluationStarted {
                condition_id: condition.condition_id.clone(),
                planned_ordinal: condition.planned_ordinal,
                evaluator_source_revision: plan.evaluator_source_revision.clone(),
                evaluator_prompt_sha256: plan.evaluator_prompt_sha256.clone(),
            },
        );
        push_event(
            &mut events,
            1_003,
            ExecutionEventPayloadV1::ToolCall {
                condition_id: condition.condition_id.clone(),
                tool_name_sha256: digest("tool"),
                request_sha256: digest("tool request"),
                request_bytes: 20,
                response_sha256: Some(digest("tool response")),
                response_bytes: 30,
            },
        );
        push_event(
            &mut events,
            1_004,
            ExecutionEventPayloadV1::EvaluationFinished {
                condition_id: condition.condition_id.clone(),
                planned_ordinal: condition.planned_ordinal,
                status: CustodyTerminalStatusV1::Completed,
                output_sha256: Some(digest("evaluator output")),
                output_bytes: 44,
                evaluator_call_count: 1,
                evaluator_token_count: 123,
            },
        );
        push_event(
            &mut events,
            1_005,
            ExecutionEventPayloadV1::RunFinished {
                run_id: plan.run_id.clone(),
                planned_condition_count: 1,
                completed_condition_count: 1,
                terminal_condition_count: 1,
                run_status: "completed".to_owned(),
            },
        );

        let event_hashes: Vec<String> = events.iter().map(|event| event.event_sha256.clone()).collect();
        let event_root = compute_execution_merkle_root(&event_hashes).unwrap();
        let bundle_payload = serde_json::to_vec(&(
            &plan_sha,
            &plan.plan_commitment_sha256,
            &event_root,
            events.len() as u64,
            &plan.run_id,
        )).unwrap();
        let bundle = digest_domain(BUNDLE_DOMAIN, &bundle_payload).unwrap();
        let anchor = if with_anchor {
            let proof = compute_merkle_inclusion_proof(&[bundle.clone()], 0).unwrap();
            let root = compute_execution_merkle_root(&[bundle.clone()]).unwrap();
            Some(ExternalCheckpointStatementV1 {
                provider: "synthetic-local-transparency-log".to_owned(),
                log_id: "synthetic-log-001".to_owned(),
                record_id: "entry-001".to_owned(),
                bundle_sha256: bundle,
                checkpoint_root_sha256: root,
                tree_size: 1,
                anchored_at_unix_milliseconds: 2_000,
                inclusion_proof: proof,
            })
        } else {
            None
        };
        let receipt = ExecutionCustodyReceiptV1 {
            schema: CUSTODY_RECEIPT_SCHEMA.to_owned(),
            analyzer_source_revision: plan.analyzer_source_revision.clone(),
            purpose: plan.purpose,
            run_id: plan.run_id.clone(),
            plan_sha256: plan_sha,
            plan_commitment_sha256: plan.plan_commitment_sha256.clone(),
            evaluator_source_revision: plan.evaluator_source_revision.clone(),
            evaluator_prompt_sha256: plan.evaluator_prompt_sha256.clone(),
            rubric_sha256: plan.rubric_sha256.clone(),
            environment_sha256: plan.environment_sha256.clone(),
            events,
            external_checkpoint: anchor,
        };
        let receipt_bytes = serde_json::to_vec_pretty(&receipt).unwrap();
        (plan_bytes, receipt_bytes)
    }

    #[test]
    fn verifies_complete_event_chain_and_local_merkle_anchor() {
        let (plan, receipt) = make_receipt(true);
        let report = analyze_execution_custody(&plan, &receipt).unwrap();
        assert_eq!(report.fields.event_count, 6);
        assert!(report.fields.event_chain_valid);
        assert!(report.fields.event_sequence_contiguous);
        assert!(report.fields.timestamps_monotonic);
        assert_eq!(report.fields.planned_condition_count, 1);
        assert_eq!(report.fields.observed_condition_count, 1);
        assert_eq!(report.fields.completed_condition_count, 1);
        assert_eq!(report.fields.total_tool_call_count, 1);
        assert_eq!(report.fields.total_evaluator_call_count, 1);
        assert_eq!(report.fields.total_evaluator_token_count, 123);
        assert_eq!(report.fields.total_rendered_input_bytes, 13);
        assert_eq!(report.fields.total_output_bytes, 44);
        assert!(report.fields.external_checkpoint_present);
        assert!(report.fields.external_checkpoint_matches_bundle);
        assert!(report.fields.external_inclusion_proof_valid);
        assert!(!report.fields.external_checkpoint_authenticity_proven);
        assert!(!report.fields.runner_identity_authenticated);
        assert!(!report.fields.raw_prompt_custody_proven);
        assert!(!report.fields.evaluator_execution_proven);
        assert!(!report.fields.wall_clock_authenticity_proven);
        assert!(report.fields.external_anchor_required);
        assert!(!report.fields.scoring_eligible_declared);
    }

    #[test]
    fn report_commitment_verifies_and_detects_tampering() {
        let (plan, receipt) = make_receipt(false);
        let report = analyze_execution_custody(&plan, &receipt).unwrap();
        let bytes = serde_json::to_vec(&report).unwrap();
        let verified = verify_execution_custody_report(&bytes).unwrap();
        assert_eq!(verified.report_commitment_sha256, report.report_commitment_sha256);
        let mut tampered: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        tampered["fields"]["completed_condition_count"] = serde_json::json!(999);
        let altered = serde_json::to_vec(&tampered).unwrap();
        assert!(verify_execution_custody_report(&altered).unwrap_err().contains("commitment mismatch"));
    }

    #[test]
    fn rejects_broken_previous_hash_and_event_hash() {
        let (plan, receipt) = make_receipt(false);
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["events"][2]["previous_event_sha256"] = serde_json::json!(digest("wrong predecessor"));
        assert!(analyze_execution_custody(&plan, &serde_json::to_vec(&value).unwrap())
            .unwrap_err().contains("previous hash"));
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["events"][2]["event_sha256"] = serde_json::json!(digest("tampered event"));
        assert!(analyze_execution_custody(&plan, &serde_json::to_vec(&value).unwrap())
            .unwrap_err().contains("event hash mismatch"));
    }

    #[test]
    fn rejects_sequence_gap_reorder_and_non_monotonic_time() {
        let (plan, receipt) = make_receipt(false);
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["events"][2]["sequence"] = serde_json::json!(20);
        assert!(analyze_execution_custody(&plan, &serde_json::to_vec(&value).unwrap())
            .unwrap_err().contains("sequence gap"));
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["events"][3]["timestamp_unix_milliseconds"] = serde_json::json!(999);
        assert!(analyze_execution_custody(&plan, &serde_json::to_vec(&value).unwrap())
            .unwrap_err().contains("timestamps are not monotonic"));
    }

    #[test]
    fn rejects_rendered_input_and_schedule_drift() {
        let (plan, receipt) = make_receipt(false);
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["events"][1]["payload"]["input_sha256"] = serde_json::json!(digest("different input"));
        let mut typed: ExecutionCustodyReceiptV1 = serde_json::from_value(value).unwrap();
        rehash_events(&mut typed.events);
        assert!(analyze_execution_custody(&plan, &serde_json::to_vec(&typed).unwrap())
            .unwrap_err().contains("rendered input differs"));
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["events"][1]["payload"]["planned_ordinal"] = serde_json::json!(8);
        let mut typed: ExecutionCustodyReceiptV1 = serde_json::from_value(value).unwrap();
        rehash_events(&mut typed.events);
        assert!(analyze_execution_custody(&plan, &serde_json::to_vec(&typed).unwrap())
            .unwrap_err().contains("committed serial schedule"));
    }

    #[test]
    fn rejects_tool_calls_outside_an_active_evaluation() {
        let (plan, receipt) = make_receipt(false);
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["events"][2]["payload"]["event_type"] = serde_json::json!("tool_call");
        let altered = serde_json::to_vec(&value).unwrap();
        assert!(analyze_execution_custody(&plan, &altered).is_err());
    }

    #[test]
    fn rejects_missing_terminal_event_and_run_finished_count_mismatch() {
        let (plan, receipt) = make_receipt(false);
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["events"].as_array_mut().unwrap().remove(4);
        assert!(analyze_execution_custody(&plan, &serde_json::to_vec(&value).unwrap()).is_err());
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["events"][5]["payload"]["completed_condition_count"] = serde_json::json!(0);
        let mut typed: ExecutionCustodyReceiptV1 = serde_json::from_value(value).unwrap();
        rehash_events(&mut typed.events);
        assert!(analyze_execution_custody(&plan, &serde_json::to_vec(&typed).unwrap())
            .unwrap_err().contains("completed count"));
    }

    #[test]
    fn rejects_external_checkpoint_root_and_tree_size_mismatch() {
        let (plan, receipt) = make_receipt(true);
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["external_checkpoint"]["checkpoint_root_sha256"] = serde_json::json!(digest("wrong root"));
        assert!(analyze_execution_custody(&plan, &serde_json::to_vec(&value).unwrap())
            .unwrap_err().contains("inclusion proof"));
        let mut value: serde_json::Value = serde_json::from_slice(&receipt).unwrap();
        value["external_checkpoint"]["tree_size"] = serde_json::json!(2);
        assert!(analyze_execution_custody(&plan, &serde_json::to_vec(&value).unwrap())
            .unwrap_err().contains("tree size"));
    }

    #[test]
    fn merkle_inclusion_proofs_work_for_odd_tree_sizes() {
        let leaves = vec![digest("a"), digest("b"), digest("c")];
        let root = compute_execution_merkle_root(&leaves).unwrap();
        for index in 0..leaves.len() {
            let proof = compute_merkle_inclusion_proof(&leaves, index).unwrap();
            assert!(verify_merkle_inclusion_proof(&leaves[index], &proof, &root).unwrap());
        }
        let mut proof = compute_merkle_inclusion_proof(&leaves, 2).unwrap();
        proof.sibling_sha256[0] = digest("wrong sibling");
        assert!(!verify_merkle_inclusion_proof(&leaves[2], &proof, &root).unwrap());
    }

    #[test]
    fn rejects_parallel_execution_and_empty_schedule() {
        let (plan, _) = make_plan();
        let mut fields = plan.fields();
        fields.max_parallelism = 2;
        assert!(validate_execution_custody_plan_fields(&fields).unwrap_err()
            .iter().any(|issue| issue.contains("serial max_parallelism")));
        fields.max_parallelism = 1;
        fields.conditions.clear();
        assert!(validate_execution_custody_plan_fields(&fields).unwrap_err()
            .iter().any(|issue| issue.contains("conditions must contain")));
    }

    #[test]
    fn plan_commitment_rejects_changed_environment_and_revision() {
        let (plan, _) = make_plan();
        let mut fields = plan.fields();
        fields.environment_sha256 = digest("changed env");
        assert!(validate_execution_custody_plan_fields(&fields).is_ok());
        fields.analyzer_source_revision = "c".repeat(40);
        assert!(validate_execution_custody_plan_fields(&fields).unwrap_err()
            .iter().any(|issue| issue.contains("exact compiled validator revision")));
    }
}
