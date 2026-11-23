// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bounded one-way IPC client for the standalone MACE candidate worker.
//!
//! A worker is scoped to one task block. It receives only candidate envelopes;
//! evaluator truth, intervention labels and scoring state never enter its
//! environment or request stream.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use broca_mace_hdc_retrieval::experience_view_digest;
use broca_mace_policy::{
    Action, DecisionNeedV1, Experience, IdentificationMethodV1, Objective, Regime, Scope,
};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};

const ENVELOPE_SCHEMA: &str = "mace_candidate_envelope_v1";
const RECEIPT_SCHEMA: &str = "mace_candidate_receipt_v1";
const READY_LINE: &str = "MACE_CANDIDATE_STREAM_READY_V1";
const MAX_REQUEST_BYTES: usize = 32 * 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = 1024 * 1024;
const MAX_STDERR_BYTES: usize = 64 * 1024;
const MAX_BINARY_BYTES: u64 = 512 * 1024 * 1024;
const STARTUP_TIMEOUT: Duration = Duration::from_secs(10);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug)]
pub struct CandidateProcessObservation {
    pub action: Option<Action>,
    pub support_event_ids: Vec<u32>,
    pub retrieved_event_ids: Vec<u32>,
    pub candidate_input_sha256: String,
    pub decision_need_sha256: String,
    pub candidate_evidence_sha256: String,
    pub candidate_observation_id: String,
    pub retrieval_profile_sha256: Option<String>,
    pub candidate_visible_event_count: u64,
    pub index_payload_bytes: u64,
    pub candidate_latency_ns: u64,
    pub request_build_ns: u64,
    pub process_wall_ns: u64,
    pub receipt_validation_ns: u64,
    pub end_to_end_ns: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateReceiptWire {
    schema_version: String,
    run_id: String,
    pair_block_id: String,
    observation_sequence: u64,
    task_nonce_sha256: String,
    method: String,
    request_sha256: String,
    decision_need_sha256: String,
    candidate_evidence_sha256: String,
    candidate_input_sha256: String,
    candidate_observation_id: String,
    decision: String,
    decision_support_event_ids: Vec<u32>,
    retrieved_event_ids: Vec<u32>,
    retrieval_profile_sha256: Option<String>,
    index_payload_bytes: u64,
    latency_ns: u64,
    probe_requested: bool,
}

pub struct CandidateProcessSession {
    child: Child,
    stdin: ChildStdin,
    responses: mpsc::Receiver<Result<String, String>>,
    stderr: Arc<Mutex<Vec<u8>>>,
    binary_sha256: String,
    startup_ns: u64,
    next_expected_sequence: Option<u64>,
}

impl CandidateProcessSession {
    pub fn from_environment() -> Result<Self, String> {
        let binary_path = env::var("MACE_CANDIDATE_BIN")
            .map_err(|_| "MACE_CANDIDATE_BIN must name the absolute candidate executable".to_string())?;
        let expected_sha256 = env::var("MACE_CANDIDATE_BIN_SHA256")
            .map_err(|_| "MACE_CANDIDATE_BIN_SHA256 must pin the candidate executable".to_string())?;
        if !valid_sha256(&expected_sha256) {
            return Err("MACE_CANDIDATE_BIN_SHA256 must be 64 lowercase hexadecimal characters".to_string());
        }

        let path = PathBuf::from(binary_path);
        if !path.is_absolute() {
            return Err("MACE_CANDIDATE_BIN must be an absolute path".to_string());
        }
        Self::spawn(path, expected_sha256)
    }

    fn spawn(path: PathBuf, expected_sha256: String) -> Result<Self, String> {
        let canonical_path = fs::canonicalize(&path)
            .map_err(|error| format!("canonicalize candidate binary {}: {error}", path.display()))?;
        let metadata = fs::metadata(&canonical_path)
            .map_err(|error| format!("stat candidate binary {}: {error}", canonical_path.display()))?;
        if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_BINARY_BYTES {
            return Err("candidate binary must be a non-empty regular file no larger than 512 MiB".into());
        }
        let binary = fs::read(&canonical_path)
            .map_err(|error| format!("read candidate binary {}: {error}", canonical_path.display()))?;
        let actual_sha256 = hex_digest(Sha256::digest(&binary).as_slice());
        if actual_sha256 != expected_sha256 {
            return Err(format!(
                "candidate binary digest mismatch: expected {expected_sha256}, got {actual_sha256}"
            ));
        }

        let started = Instant::now();
        let mut command = Command::new(&canonical_path);
        command
            .arg("--stream")
            .env_clear()
            .current_dir("/")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command
            .spawn()
            .map_err(|error| format!("spawn candidate worker: {error}"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "candidate worker stdin pipe unavailable".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "candidate worker stdout pipe unavailable".to_string())?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| "candidate worker stderr pipe unavailable".to_string())?;

        // Keep at most one complete response queued if a child emits unsolicited output.
        let (sender, responses) = mpsc::sync_channel(1);
        thread::Builder::new()
            .name("mace-candidate-stdout".into())
            .spawn(move || {
                let mut reader = BufReader::new(stdout);
                loop {
                    // Read in bounded chunks so a malformed child cannot force
                    // an unbounded allocation before the response-size check.
                    let mut bytes = Vec::with_capacity(4096);
                    let mut terminated = false;
                    loop {
                        let available = match reader.fill_buf() {
                            Ok([]) => break,
                            Ok(available) => available,
                            Err(error) => {
                                let _ = sender.send(Err(format!("read candidate stdout: {error}")));
                                return;
                            }
                        };
                        let newline = available.iter().position(|byte| *byte == b'\n');
                        let take = newline.unwrap_or(available.len());
                        if bytes.len().saturating_add(take) > MAX_RESPONSE_BYTES {
                            let _ = sender.send(Err(format!(
                                "candidate response exceeds {MAX_RESPONSE_BYTES} bytes"
                            )));
                            return;
                        }
                        bytes.extend_from_slice(&available[..take]);
                        let consumed = take + usize::from(newline.is_some());
                        reader.consume(consumed);
                        if newline.is_some() {
                            terminated = true;
                            break;
                        }
                    }

                    if bytes.is_empty() && !terminated {
                        let _ = sender.send(Err("candidate worker closed stdout".into()));
                        break;
                    }
                    if !terminated {
                        let _ = sender.send(Err("candidate worker closed stdout with an unterminated response".into()));
                        break;
                    }
                    if bytes.last() == Some(&b'\r') {
                        bytes.pop();
                    }
                    let line = match String::from_utf8(bytes) {
                        Ok(line) => line,
                        Err(error) => {
                            let _ = sender.send(Err(format!("candidate stdout is not UTF-8: {error}")));
                            break;
                        }
                    };
                    if sender.send(Ok(line)).is_err() {
                        break;
                    }
                }
            })
            .map_err(|error| format!("start candidate stdout reader: {error}"))?;

        let stderr_capture = Arc::new(Mutex::new(Vec::new()));
        let stderr_sink = Arc::clone(&stderr_capture);
        thread::Builder::new()
            .name("mace-candidate-stderr".into())
            .spawn(move || {
                let mut reader = stderr;
                let mut buffer = [0_u8; 8192];
                loop {
                    match reader.read(&mut buffer) {
                        Ok(0) | Err(_) => break,
                        Ok(count) => {
                            if let Ok(mut captured) = stderr_sink.lock() {
                                let remaining = MAX_STDERR_BYTES.saturating_sub(captured.len());
                                captured.extend_from_slice(&buffer[..count.min(remaining)]);
                            } else {
                                break;
                            }
                        }
                    }
                }
            })
            .map_err(|error| format!("start candidate stderr reader: {error}"))?;

        let ready = match responses.recv_timeout(STARTUP_TIMEOUT) {
            Ok(Ok(line)) => line,
            Ok(Err(error)) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("candidate worker failed during startup: {error}"));
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("candidate worker readiness timeout/failure: {error}"));
            }
        };
        if ready != READY_LINE {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("unexpected candidate worker readiness frame: {ready:?}"));
        }

        // Detect a path replacement between the first digest check and spawn.
        // This is a defensive consistency check, not an OS-level immutable-fd exec guarantee.
        let after_spawn = fs::read(&canonical_path)
            .map_err(|error| format!("re-read candidate binary after spawn: {error}"))?;
        if hex_digest(Sha256::digest(&after_spawn).as_slice()) != actual_sha256 {
            let _ = child.kill();
            let _ = child.wait();
            return Err("candidate binary changed during worker startup".into());
        }

        Ok(Self {
            child,
            stdin,
            responses,
            stderr: stderr_capture,
            binary_sha256: actual_sha256,
            startup_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            next_expected_sequence: None,
        })
    }

    pub fn binary_sha256(&self) -> &str {
        &self.binary_sha256
    }

    pub fn startup_ns(&self) -> u64 {
        self.startup_ns
    }

    pub fn invoke(
        &mut self,
        run_id: &str,
        pair_block_id: &str,
        observation_sequence: u64,
        method: &str,
        need: DecisionNeedV1,
        memory: &[Experience],
    ) -> Result<CandidateProcessObservation, String> {
        if !matches!(method, "exact_scope_scan" | "typed_content_hdc") {
            return Err(format!("unsupported detached candidate method {method}"));
        }
        if let Some(previous) = self.next_expected_sequence {
            if observation_sequence <= previous {
                return Err("candidate observation sequence must increase within a worker session".into());
            }
        }
        self.next_expected_sequence = Some(observation_sequence);

        let end_to_end_started = Instant::now();
        let request_build_started = Instant::now();
        let mut nonce_bytes = [0_u8; 24];
        fs::File::open("/dev/urandom")
            .and_then(|mut file| file.read_exact(&mut nonce_bytes))
            .map_err(|error| format!("obtain custodian task nonce from /dev/urandom: {error}"))?;
        let task_nonce = hex_bytes(&nonce_bytes);

        // Do not send unrelated regimes, objectives or action alternatives
        // to the candidate process. The candidate receives only the exact
        // scope/action projection declared by DecisionNeedV1.
        let mut ordered_memory: Vec<Experience> = memory
            .iter()
            .copied()
            .filter(|item| {
                item.scope == need.scope && need.action_alternatives.contains(&item.action)
            })
            .collect();
        ordered_memory.sort_by_key(|item| item.event_id);
        let request_value = candidate_envelope(
            run_id,
            pair_block_id,
            observation_sequence,
            &task_nonce,
            method,
            need,
            &ordered_memory,
        );
        let request_bytes = serde_json::to_vec(&request_value)
            .map_err(|error| format!("serialize candidate envelope: {error}"))?;
        if request_bytes.len() > MAX_REQUEST_BYTES {
            return Err(format!("candidate envelope exceeds {MAX_REQUEST_BYTES} bytes"));
        }

        let expected_decision_need_sha256 = need.canonical_digest().to_hex();
        let expected_candidate_evidence_sha256 = candidate_evidence_sha256(&ordered_memory);
        let expected_candidate_input_sha256 = candidate_input_sha256(
            &expected_decision_need_sha256,
            &expected_candidate_evidence_sha256,
        );
        let expected_observation_id = observation_id(
            run_id,
            pair_block_id,
            observation_sequence,
            method,
            &expected_candidate_input_sha256,
        );
        let expected_request_sha256 = hex_digest(Sha256::digest(&request_bytes).as_slice());
        let expected_nonce_sha256 = nonce_digest(&task_nonce);
        let request_build_ns =
            u64::try_from(request_build_started.elapsed().as_nanos()).unwrap_or(u64::MAX);

        let ipc_started = Instant::now();
        if let Err(error) = self.stdin
            .write_all(&request_bytes)
            .and_then(|_| self.stdin.write_all(b"\n"))
            .and_then(|_| self.stdin.flush())
        {
            let reason = format!("send candidate request: {error}");
            let _ = self.child.kill();
            let _ = self.child.wait();
            return Err(self.worker_error(reason));
        }

        let line = match self.responses.recv_timeout(REQUEST_TIMEOUT) {
            Ok(Ok(line)) => line,
            Ok(Err(error)) => return Err(self.worker_error(error)),
            Err(error) => {
                let _ = self.child.kill();
                let _ = self.child.wait();
                return Err(self.worker_error(format!("candidate response timeout/failure: {error}")));
            }
        };
        let process_wall_ns = u64::try_from(ipc_started.elapsed().as_nanos()).unwrap_or(u64::MAX);

        let validation_started = Instant::now();
        let receipt: CandidateReceiptWire = match serde_json::from_str(&line) {
            Ok(receipt) => receipt,
            Err(error) => {
                let reason = format!("invalid candidate receipt JSON: {error}");
                let _ = self.child.kill();
                let _ = self.child.wait();
                return Err(self.worker_error(reason));
            }
        };
        if let Err(error) = validate_receipt(
            &receipt,
            run_id,
            pair_block_id,
            observation_sequence,
            method,
            &expected_request_sha256,
            &expected_nonce_sha256,
            &expected_decision_need_sha256,
            &expected_candidate_evidence_sha256,
            &expected_candidate_input_sha256,
            &expected_observation_id,
            &ordered_memory,
            need.scope,
            need.action_alternatives,
        ) {
            let _ = self.child.kill();
            let _ = self.child.wait();
            return Err(self.worker_error(error));
        }
        let receipt_validation_ns =
            u64::try_from(validation_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let end_to_end_ns =
            u64::try_from(end_to_end_started.elapsed().as_nanos()).unwrap_or(u64::MAX);

        Ok(CandidateProcessObservation {
            action: match receipt.decision.as_str() {
                "action_a" => Some(Action::A),
                "action_b" => Some(Action::B),
                "abstain" => None,
                other => return Err(format!("candidate receipt has unsupported decision {other}")),
            },
            support_event_ids: receipt.decision_support_event_ids,
            retrieved_event_ids: receipt.retrieved_event_ids,
            candidate_input_sha256: receipt.candidate_input_sha256,
            decision_need_sha256: receipt.decision_need_sha256,
            candidate_evidence_sha256: receipt.candidate_evidence_sha256,
            candidate_observation_id: receipt.candidate_observation_id,
            retrieval_profile_sha256: receipt.retrieval_profile_sha256,
            candidate_visible_event_count: u64::try_from(ordered_memory.len()).unwrap_or(u64::MAX),
            index_payload_bytes: receipt.index_payload_bytes,
            candidate_latency_ns: receipt.latency_ns,
            request_build_ns,
            process_wall_ns,
            receipt_validation_ns,
            end_to_end_ns,
        })
    }

    fn worker_error(&self, error: String) -> String {
        let stderr = self
            .stderr
            .lock()
            .map(|value| String::from_utf8_lossy(&value).into_owned())
            .unwrap_or_else(|_| "<stderr unavailable>".into());
        format!("{error}; candidate stderr (capped): {stderr}")
    }
}

impl Drop for CandidateProcessSession {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn candidate_envelope(
    run_id: &str,
    pair_block_id: &str,
    observation_sequence: u64,
    task_nonce: &str,
    method: &str,
    need: DecisionNeedV1,
    memory: &[Experience],
) -> serde_json::Value {
    json!({
        "schema_version": ENVELOPE_SCHEMA,
        "run_id": run_id,
        "pair_block_id": pair_block_id,
        "observation_sequence": observation_sequence,
        "task_nonce": task_nonce,
        "method": method,
        "decision_need": {
            "schema_version": need.schema_version,
            "scope": scope_value(need.scope),
            "action_alternatives": need.action_alternatives.map(action_name),
            "semantic_estimand": semantic_estimand_value(need.semantic_estimand),
            "identification_method": identification_method_name(need.identification_method),
            "qualification_profile_sha256": need.qualification_profile_digest.to_hex()
        },
        "candidate_evidence": memory.iter().map(experience_value).collect::<Vec<_>>()
    })
}

fn scope_value(scope: Scope) -> serde_json::Value {
    json!({
        "regime": match scope.regime {
            Regime::Baseline => "baseline",
            Regime::Shifted => "shifted"
        },
        "objective": match scope.objective {
            Objective::Quality => "quality",
            Objective::Cost => "cost",
            Objective::Speed => "speed"
        }
    })
}

fn action_name(action: Action) -> &'static str {
    match action {
        Action::A => "a",
        Action::B => "b",
    }
}

fn identification_method_name(method: IdentificationMethodV1) -> &'static str {
    match method {
        IdentificationMethodV1::RandomizedControlledTrial => "randomized_controlled_trial",
        IdentificationMethodV1::QuasiExperimental => "quasi_experimental",
        IdentificationMethodV1::Observational => "observational",
        IdentificationMethodV1::Unknown => "unknown",
    }
}

fn semantic_estimand_value(value: broca_mace_policy::SemanticEstimandV1) -> serde_json::Value {
    json!({
        "schema_version": value.schema_version,
        "population_spec_sha256": value.population_spec_digest.to_hex(),
        "treatment_version_sha256": value.treatment_version_digest.to_hex(),
        "comparator_version_sha256": value.comparator_version_digest.to_hex(),
        "outcome_definition_sha256": value.outcome_definition_digest.to_hex(),
        "outcome_unit_sha256": value.outcome_unit_digest.to_hex(),
        "follow_up_horizon_ms": value.follow_up_horizon_ms,
        "censoring_strategy_sha256": value.censoring_strategy_digest.to_hex(),
        "intercurrent_event_strategy_sha256": value.intercurrent_event_strategy_digest.to_hex(),
        "target_weighting_sha256": value.target_weighting_digest.to_hex()
    })
}

fn experience_value(value: &Experience) -> serde_json::Value {
    let causal = value.causal_identification;
    json!({
        "event_id": value.event_id,
        "source_family": value.source_family,
        "source_root_id": value.source_root_id,
        "source_root_sha256": value.source_root_digest.to_hex(),
        "scope": scope_value(value.scope),
        "action": action_name(value.action),
        "observed_utility": value.observed_utility,
        "measurement_error_bound": value.measurement_error_bound,
        "causal_identification": {
            "method": identification_method_name(causal.method),
            "design_sha256": causal.design_digest.to_hex(),
            "estimand_sha256": causal.estimand_digest.to_hex(),
            "semantic_estimand": semantic_estimand_value(causal.semantic_estimand),
            "propensity_min_bps": causal.propensity_min_bps,
            "propensity_max_bps": causal.propensity_max_bps,
            "max_standardized_mean_difference_micros": causal.max_standardized_mean_difference_micros,
            "interference_controlled": causal.interference_controlled
        },
        "intervention": value.intervention,
        "prediction_pre_registered": value.prediction_pre_registered,
        "outcome_reconciled": value.outcome_reconciled,
        "source_trusted": value.source_trusted,
        "source_retracted": value.source_retracted,
        "confounders_controlled": value.confounders_controlled,
        "delayed_horizon_complete": value.delayed_horizon_complete
    })
}

fn candidate_evidence_sha256(memory: &[Experience]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:candidate-evidence-set:v1\0");
    for item in memory {
        hasher.update(item.event_id.to_be_bytes());
        hasher.update(experience_view_digest(item));
    }
    hex_digest(hasher.finalize().as_slice())
}

fn candidate_input_sha256(need_sha256: &str, evidence_sha256: &str) -> String {
    let payload = format!(
        "mace-candidate-input-v1|need={need_sha256}|evidence={evidence_sha256}"
    );
    hex_digest(Sha256::digest(payload.as_bytes()).as_slice())
}

fn observation_id(
    run_id: &str,
    pair_block_id: &str,
    sequence: u64,
    method: &str,
    input_sha256: &str,
) -> String {
    let identity = format!(
        "mace-candidate-observation-v1|run={run_id}|pair={pair_block_id}|sequence={sequence}|method={method}|input={input_sha256}"
    );
    format!("obs-{}", hex_digest(Sha256::digest(identity.as_bytes()).as_slice()))
}

fn nonce_digest(nonce: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"mace-candidate-task-nonce-v1\0");
    hasher.update(nonce.as_bytes());
    hex_digest(hasher.finalize().as_slice())
}

fn validate_receipt(
    receipt: &CandidateReceiptWire,
    run_id: &str,
    pair_block_id: &str,
    sequence: u64,
    method: &str,
    request_sha256: &str,
    nonce_sha256: &str,
    decision_need_sha256: &str,
    candidate_evidence_sha256: &str,
    candidate_input_sha256: &str,
    candidate_observation_id: &str,
    memory: &[Experience],
    scope: Scope,
    action_alternatives: [Action; 2],
) -> Result<(), String> {
    if receipt.schema_version != RECEIPT_SCHEMA
        || receipt.run_id != run_id
        || receipt.pair_block_id != pair_block_id
        || receipt.observation_sequence != sequence
        || receipt.method != method
        || receipt.request_sha256 != request_sha256
        || receipt.task_nonce_sha256 != nonce_sha256
        || receipt.decision_need_sha256 != decision_need_sha256
        || receipt.candidate_evidence_sha256 != candidate_evidence_sha256
        || receipt.candidate_input_sha256 != candidate_input_sha256
        || receipt.candidate_observation_id != candidate_observation_id
    {
        return Err("candidate receipt does not match the exact request/task/sequence/input".into());
    }
    if receipt.probe_requested {
        return Err("candidate requested an unsupported probe in this retrieval-only runner".into());
    }
    if !valid_sha256(&receipt.candidate_input_sha256)
        || !valid_sha256(&receipt.decision_need_sha256)
        || !valid_sha256(&receipt.candidate_evidence_sha256)
        || !valid_sha256(&receipt.request_sha256)
        || !valid_sha256(&receipt.task_nonce_sha256)
    {
        return Err("candidate receipt contains a malformed digest".into());
    }
    if let Some(profile) = &receipt.retrieval_profile_sha256 {
        if !valid_sha256(profile) {
            return Err("candidate receipt has a malformed retrieval profile digest".into());
        }
    }

    let memory_by_id: BTreeMap<u32, &Experience> =
        memory.iter().map(|item| (item.event_id, item)).collect();
    let retrieved: BTreeSet<u32> = receipt.retrieved_event_ids.iter().copied().collect();
    let support: BTreeSet<u32> = receipt.decision_support_event_ids.iter().copied().collect();
    if retrieved.len() != receipt.retrieved_event_ids.len()
        || support.len() != receipt.decision_support_event_ids.len()
    {
        return Err("candidate receipt contains duplicate event IDs".into());
    }
    if !support.is_subset(&retrieved) {
        return Err("candidate decision support IDs are not a subset of retrieved IDs".into());
    }
    for event_id in &retrieved {
        let item = memory_by_id
            .get(event_id)
            .ok_or_else(|| format!("candidate returned unknown event ID {event_id}"))?;
        if item.scope != scope || !action_alternatives.contains(&item.action) {
            return Err(format!("candidate returned out-of-contract event ID {event_id}"));
        }
    }
    if !matches!(receipt.decision.as_str(), "action_a" | "action_b" | "abstain") {
        return Err("candidate returned an unsupported decision".into());
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

fn hex_digest(bytes: &[u8]) -> String {
    hex_bytes(bytes)
}



#[cfg(test)]
mod tests {
    use super::*;

    fn valid_receipt_value() -> serde_json::Value {
        let run_id = "run-1";
        let pair_block_id = "world-1:task-1";
        let sequence = 7_u64;
        let method = "exact_scope_scan";
        let nonce = "nonce-0123456789abcdef";
        let request_sha256 = "e".repeat(64);
        let decision_need_sha256 = "a".repeat(64);
        let candidate_evidence_sha256 = "b".repeat(64);
        let candidate_input_sha256 =
            candidate_input_sha256(&decision_need_sha256, &candidate_evidence_sha256);
        let candidate_observation_id = observation_id(
            run_id,
            pair_block_id,
            sequence,
            method,
            &candidate_input_sha256,
        );
        json!({
            "schema_version": RECEIPT_SCHEMA,
            "run_id": run_id,
            "pair_block_id": pair_block_id,
            "observation_sequence": sequence,
            "task_nonce_sha256": nonce_digest(nonce),
            "method": method,
            "request_sha256": request_sha256,
            "decision_need_sha256": decision_need_sha256,
            "candidate_evidence_sha256": candidate_evidence_sha256,
            "candidate_input_sha256": candidate_input_sha256,
            "candidate_observation_id": candidate_observation_id,
            "decision": "abstain",
            "decision_support_event_ids": [],
            "retrieved_event_ids": [],
            "retrieval_profile_sha256": null,
            "index_payload_bytes": 0,
            "latency_ns": 10,
            "probe_requested": false
        })
    }

    fn validate_value(value: serde_json::Value) -> Result<(), String> {
        let receipt: CandidateReceiptWire = serde_json::from_value(value)
            .map_err(|error| format!("invalid candidate receipt JSON: {error}"))?;
        let run_id = "run-1";
        let pair_block_id = "world-1:task-1";
        let sequence = 7_u64;
        let method = "exact_scope_scan";
        let nonce = "nonce-0123456789abcdef";
        let request_sha256 = "e".repeat(64);
        let decision_need_sha256 = "a".repeat(64);
        let candidate_evidence_sha256 = "b".repeat(64);
        let candidate_input = candidate_input_sha256(
            &decision_need_sha256,
            &candidate_evidence_sha256,
        );
        let expected_observation = observation_id(
            run_id,
            pair_block_id,
            sequence,
            method,
            &candidate_input,
        );
        validate_receipt(
            &receipt,
            run_id,
            pair_block_id,
            sequence,
            method,
            &request_sha256,
            &nonce_digest(nonce),
            &decision_need_sha256,
            &candidate_evidence_sha256,
            &candidate_input,
            &expected_observation,
            &[],
            Scope {
                regime: Regime::Baseline,
                objective: Objective::Quality,
            },
            [Action::A, Action::B],
        )
    }

    #[test]
    fn candidate_receipt_accepts_exact_request_and_empty_support() {
        assert!(validate_value(valid_receipt_value()).is_ok());
    }

    #[test]
    fn candidate_receipt_rejects_unknown_evaluator_fields() {
        let mut value = valid_receipt_value();
        value["truth"] = json!({"optimal_action": "action_b"});
        assert!(validate_value(value).unwrap_err().contains("unknown field"));
    }

    #[test]
    fn candidate_receipt_rejects_duplicate_or_nonretrieved_support_ids() {
        let mut duplicate = valid_receipt_value();
        duplicate["retrieved_event_ids"] = json!([1, 1]);
        duplicate["decision_support_event_ids"] = json!([1]);
        assert!(validate_value(duplicate).unwrap_err().contains("duplicate event IDs"));

        let mut outside = valid_receipt_value();
        outside["retrieved_event_ids"] = json!([1]);
        outside["decision_support_event_ids"] = json!([2]);
        assert!(validate_value(outside)
            .unwrap_err()
            .contains("not a subset of retrieved IDs"));
    }

    #[test]
    fn candidate_receipt_rejects_unknown_retrieved_ids() {
        let mut value = valid_receipt_value();
        value["retrieved_event_ids"] = json!([42]);
        value["decision_support_event_ids"] = json!([42]);
        assert!(validate_value(value).unwrap_err().contains("unknown event ID 42"));
    }

    #[test]
    fn candidate_receipt_rejects_wrong_request_nonce_and_observation_identity() {
        let mut wrong_request = valid_receipt_value();
        wrong_request["request_sha256"] = json!("f".repeat(64));
        assert!(validate_value(wrong_request).unwrap_err().contains("exact request"));

        let mut wrong_nonce = valid_receipt_value();
        wrong_nonce["task_nonce_sha256"] = json!("f".repeat(64));
        assert!(validate_value(wrong_nonce).unwrap_err().contains("exact request"));

        let mut wrong_identity = valid_receipt_value();
        wrong_identity["candidate_observation_id"] = json!("obs-0000000000000000");
        assert!(validate_value(wrong_identity).unwrap_err().contains("exact request"));
    }

    #[test]
    fn candidate_artifact_digest_validation_is_canonical() {
        assert!(valid_sha256(&"a".repeat(64)));
        assert!(!valid_sha256(&"A".repeat(64)));
        assert!(!valid_sha256("deadbeef"));
        assert!(!valid_sha256(&"g".repeat(64)));
    }
}
