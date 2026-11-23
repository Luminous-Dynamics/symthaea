// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Standalone, oracle-blind MACE candidate process.
//!
//! The process accepts exactly one bounded candidate envelope on stdin and
//! emits exactly one candidate receipt on stdout. Evaluator-only labels are
//! not part of the wire types; serde rejects unknown fields at every level.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::time::Instant;

use broca_mace_hdc_retrieval::{
    experience_view_digest,
    typed_content::{SemanticExperienceQueryV1, TypedContentConfigV1, TypedContentIndexV1},
};
use broca_mace_policy::{
    decide, Action, CausalIdentificationV1, DecisionNeedV1, EvidenceDigest256, Experience,
    IdentificationMethodV1, Objective, Policy, Query, Regime, Scope, SemanticEstimandV1,
    SourceRootDigest, transfer_qualification_profile_digest,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const ENVELOPE_SCHEMA: &str = "mace_candidate_envelope_v1";
const RECEIPT_SCHEMA: &str = "mace_candidate_receipt_v1";
const MAX_REQUEST_BYTES: usize = 32 * 1024 * 1024;
const MAX_EVIDENCE_RECORDS: usize = 20_000;
const TOP_K_PER_ACTION: usize = 2;
const TYPED_HDC_DIMENSION: usize = 2_048;
const TYPED_HDC_SEED: u64 = 0x4D41_4345_4844_4301;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum ActionWire {
    A,
    B,
}

impl From<ActionWire> for Action {
    fn from(value: ActionWire) -> Self {
        match value {
            ActionWire::A => Self::A,
            ActionWire::B => Self::B,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum RegimeWire {
    Baseline,
    Shifted,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum ObjectiveWire {
    Quality,
    Cost,
    Speed,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ScopeWire {
    regime: RegimeWire,
    objective: ObjectiveWire,
}

impl From<ScopeWire> for Scope {
    fn from(value: ScopeWire) -> Self {
        Self {
            regime: match value.regime {
                RegimeWire::Baseline => Regime::Baseline,
                RegimeWire::Shifted => Regime::Shifted,
            },
            objective: match value.objective {
                ObjectiveWire::Quality => Objective::Quality,
                ObjectiveWire::Cost => Objective::Cost,
                ObjectiveWire::Speed => Objective::Speed,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum IdentificationMethodWire {
    RandomizedControlledTrial,
    QuasiExperimental,
    Observational,
    Unknown,
}

impl From<IdentificationMethodWire> for IdentificationMethodV1 {
    fn from(value: IdentificationMethodWire) -> Self {
        match value {
            IdentificationMethodWire::RandomizedControlledTrial => Self::RandomizedControlledTrial,
            IdentificationMethodWire::QuasiExperimental => Self::QuasiExperimental,
            IdentificationMethodWire::Observational => Self::Observational,
            IdentificationMethodWire::Unknown => Self::Unknown,
        }
    }
}

fn parse_digest(value: &str, field: &str) -> Result<EvidenceDigest256, String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{field} must be 64 lowercase hexadecimal characters"));
    }
    let mut bytes = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = decode_hex(pair[0]).ok_or_else(|| format!("invalid {field}"))?;
        let low = decode_hex(pair[1]).ok_or_else(|| format!("invalid {field}"))?;
        bytes[index] = (high << 4) | low;
    }
    Ok(EvidenceDigest256::from_bytes(bytes))
}

fn parse_root_digest(value: &str) -> Result<SourceRootDigest, String> {
    let digest = SourceRootDigest::from_hex(value)
        .map_err(|_| "source_root_sha256 must be exactly 64 hexadecimal characters".to_string())?;
    if digest.to_hex() != value {
        return Err("source_root_sha256 must use canonical lowercase hexadecimal".to_string());
    }
    Ok(digest)
}

fn decode_hex(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SemanticEstimandWire {
    schema_version: u16,
    population_spec_sha256: String,
    treatment_version_sha256: String,
    comparator_version_sha256: String,
    outcome_definition_sha256: String,
    outcome_unit_sha256: String,
    follow_up_horizon_ms: u64,
    censoring_strategy_sha256: String,
    intercurrent_event_strategy_sha256: String,
    target_weighting_sha256: String,
}

impl SemanticEstimandWire {
    fn to_domain(self) -> Result<SemanticEstimandV1, String> {
        Ok(SemanticEstimandV1::new(
            self.schema_version,
            parse_digest(&self.population_spec_sha256, "population_spec_sha256")?,
            parse_digest(&self.treatment_version_sha256, "treatment_version_sha256")?,
            parse_digest(&self.comparator_version_sha256, "comparator_version_sha256")?,
            parse_digest(&self.outcome_definition_sha256, "outcome_definition_sha256")?,
            parse_digest(&self.outcome_unit_sha256, "outcome_unit_sha256")?,
            self.follow_up_horizon_ms,
            parse_digest(&self.censoring_strategy_sha256, "censoring_strategy_sha256")?,
            parse_digest(
                &self.intercurrent_event_strategy_sha256,
                "intercurrent_event_strategy_sha256",
            )?,
            parse_digest(&self.target_weighting_sha256, "target_weighting_sha256")?,
        ))
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionNeedWire {
    schema_version: u16,
    scope: ScopeWire,
    action_alternatives: [ActionWire; 2],
    semantic_estimand: SemanticEstimandWire,
    identification_method: IdentificationMethodWire,
    qualification_profile_sha256: String,
}

impl DecisionNeedWire {
    fn to_domain(self) -> Result<DecisionNeedV1, String> {
        let need = DecisionNeedV1 {
            schema_version: self.schema_version,
            scope: self.scope.into(),
            action_alternatives: [
                self.action_alternatives[0].into(),
                self.action_alternatives[1].into(),
            ],
            semantic_estimand: self.semantic_estimand.to_domain()?,
            identification_method: self.identification_method.into(),
            qualification_profile_digest: parse_digest(
                &self.qualification_profile_sha256,
                "qualification_profile_sha256",
            )?,
        };
        if !need.is_valid() {
            return Err("decision_need is invalid or uses an unsupported qualification profile".into());
        }
        Ok(need)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CausalIdentificationWire {
    method: IdentificationMethodWire,
    design_sha256: String,
    estimand_sha256: String,
    semantic_estimand: SemanticEstimandWire,
    propensity_min_bps: u16,
    propensity_max_bps: u16,
    max_standardized_mean_difference_micros: u32,
    interference_controlled: bool,
}

impl CausalIdentificationWire {
    fn to_domain(self) -> Result<CausalIdentificationV1, String> {
        Ok(CausalIdentificationV1::new(
            self.method.into(),
            parse_digest(&self.design_sha256, "design_sha256")?,
            parse_digest(&self.estimand_sha256, "estimand_sha256")?,
            self.propensity_min_bps,
            self.propensity_max_bps,
            self.max_standardized_mean_difference_micros,
            self.interference_controlled,
        )
        .with_semantic_estimand(self.semantic_estimand.to_domain()?))
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExperienceWire {
    event_id: u32,
    source_family: u16,
    source_root_id: u32,
    source_root_sha256: String,
    scope: ScopeWire,
    action: ActionWire,
    observed_utility: f64,
    measurement_error_bound: f64,
    causal_identification: CausalIdentificationWire,
    intervention: bool,
    prediction_pre_registered: bool,
    outcome_reconciled: bool,
    source_trusted: bool,
    source_retracted: bool,
    confounders_controlled: bool,
    delayed_horizon_complete: bool,
}

impl ExperienceWire {
    fn to_domain(self) -> Result<Experience, String> {
        if !self.observed_utility.is_finite() || !(0.0..=1.0).contains(&self.observed_utility) {
            return Err(format!("event {} has out-of-range observed_utility", self.event_id));
        }
        if !self.measurement_error_bound.is_finite()
            || !(0.0..=1.0).contains(&self.measurement_error_bound)
        {
            return Err(format!("event {} has out-of-range measurement_error_bound", self.event_id));
        }
        if self.propensity_bounds_invalid() {
            return Err(format!("event {} has invalid propensity bounds", self.event_id));
        }
        Ok(Experience {
            event_id: self.event_id,
            source_family: self.source_family,
            source_root_id: self.source_root_id,
            source_root_digest: parse_root_digest(&self.source_root_sha256)?,
            scope: self.scope.into(),
            action: self.action.into(),
            observed_utility: self.observed_utility,
            measurement_error_bound: self.measurement_error_bound,
            causal_identification: self.causal_identification.to_domain()?,
            intervention: self.intervention,
            prediction_pre_registered: self.prediction_pre_registered,
            outcome_reconciled: self.outcome_reconciled,
            source_trusted: self.source_trusted,
            source_retracted: self.source_retracted,
            confounders_controlled: self.confounders_controlled,
            delayed_horizon_complete: self.delayed_horizon_complete,
        })
    }

    fn propensity_bounds_invalid(&self) -> bool {
        self.causal_identification.propensity_min_bps > 10_000
            || self.causal_identification.propensity_max_bps > 10_000
            || self.causal_identification.propensity_min_bps
                > self.causal_identification.propensity_max_bps
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum CandidateMethod {
    ExactScopeScan,
    TypedContentHdc,
}

impl CandidateMethod {
    fn name(self) -> &'static str {
        match self {
            Self::ExactScopeScan => "exact_scope_scan",
            Self::TypedContentHdc => "typed_content_hdc",
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateEnvelopeV1 {
    schema_version: String,
    run_id: String,
    pair_block_id: String,
    observation_sequence: u64,
    task_nonce: String,
    method: CandidateMethod,
    decision_need: DecisionNeedWire,
    candidate_evidence: Vec<ExperienceWire>,
}

#[derive(Debug, Serialize)]
struct CandidateReceiptV1 {
    schema_version: &'static str,
    run_id: String,
    pair_block_id: String,
    observation_sequence: u64,
    task_nonce_sha256: String,
    method: &'static str,
    request_sha256: String,
    decision_need_sha256: String,
    candidate_evidence_sha256: String,
    candidate_input_sha256: String,
    candidate_observation_id: String,
    decision: &'static str,
    decision_support_event_ids: Vec<u32>,
    retrieved_event_ids: Vec<u32>,
    retrieval_profile_sha256: Option<String>,
    index_payload_bytes: u64,
    latency_ns: u64,
    probe_requested: bool,
}

fn validate_identifier(value: &str, field: &str, max_len: usize) -> Result<(), String> {
    if value.is_empty()
        || value.len() > max_len
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':')
        })
    {
        return Err(format!("{field} is empty, too long, or contains unsupported characters"));
    }
    Ok(())
}

fn validate_envelope(envelope: CandidateEnvelopeV1) -> Result<(DecisionNeedV1, Vec<Experience>), String> {
    if envelope.schema_version != ENVELOPE_SCHEMA {
        return Err("unsupported candidate envelope schema_version".into());
    }
    validate_identifier(&envelope.run_id, "run_id", 128)?;
    validate_identifier(&envelope.pair_block_id, "pair_block_id", 256)?;
    if !(16..=128).contains(&envelope.task_nonce.len())
        || !envelope
            .task_nonce
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err("task_nonce must be 16..=128 ASCII alphanumeric, '-' or '_' characters".into());
    }
    if envelope.candidate_evidence.len() > MAX_EVIDENCE_RECORDS {
        return Err(format!("candidate_evidence exceeds {MAX_EVIDENCE_RECORDS} records"));
    }
    let need = envelope.decision_need.to_domain()?;
    let mut seen = BTreeSet::new();
    let mut memory = Vec::with_capacity(envelope.candidate_evidence.len());
    for wire in envelope.candidate_evidence {
        let event_id = wire.event_id;
        if !seen.insert(event_id) {
            return Err(format!("duplicate candidate event_id {event_id}"));
        }
        let item = wire.to_domain()?;
        if item.scope != need.scope || !need.action_alternatives.contains(&item.action) {
            return Err(format!(
                "candidate evidence event_id {event_id} is outside DecisionNeedV1 scope/action alternatives"
            ));
        }
        memory.push(item);
    }
    memory.sort_by_key(|item| item.event_id);
    Ok((need, memory))
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

fn evidence_digest(memory: &[Experience]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:candidate-evidence-set:v1\0");
    for item in memory {
        hasher.update(item.event_id.to_be_bytes());
        hasher.update(experience_view_digest(item));
    }
    hex_digest(hasher.finalize().as_slice())
}

fn candidate_input_digest(need: DecisionNeedV1, evidence_sha256: &str) -> String {
    let payload = format!(
        "mace-candidate-input-v1|need={}|evidence={evidence_sha256}",
        need.canonical_digest().to_hex()
    );
    hex_digest(Sha256::digest(payload.as_bytes()).as_slice())
}

fn candidate_observation_id(
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

fn run_candidate(
    method: CandidateMethod,
    need: DecisionNeedV1,
    memory: &[Experience],
) -> Result<(Vec<u32>, Option<String>, u64, broca_mace_policy::Decision), String> {
    let by_id: BTreeMap<u32, Experience> = memory.iter().map(|item| (item.event_id, *item)).collect();
    let mut retrieved: Vec<(u32, [u8; 32], usize)> = Vec::new();
    let mut profile_sha256 = None;
    let mut index_payload_bytes = 0_u64;

    match method {
        CandidateMethod::ExactScopeScan => {
            // The exact-scope arm is the candidate-visible ceiling: return all
            // records in the declared scope and action set, not a truncated top-k.
            let mut candidates: Vec<&Experience> = memory
                .iter()
                .filter(|item| {
                    item.scope == need.scope && need.action_alternatives.contains(&item.action)
                })
                .collect();
            candidates.sort_by_key(|item| item.event_id);
            for item in candidates {
                retrieved.push((item.event_id, experience_view_digest(item), 0));
            }
        }
        CandidateMethod::TypedContentHdc => {
            if !memory.is_empty() {
                let config = TypedContentConfigV1 {
                    dimension: TYPED_HDC_DIMENSION,
                    seed: TYPED_HDC_SEED,
                    ..TypedContentConfigV1::default()
                };
                let index = TypedContentIndexV1::build(memory, config)
                    .map_err(|error| format!("typed-content index build failed: {error:?}"))?;
                profile_sha256 = Some(index.profile_digest().to_string());
                index_payload_bytes = index.index_payload_bytes();
                for action in need.action_alternatives {
                    let typed_query = SemanticExperienceQueryV1::from_decision_need(need, action)
                        .map_err(|error| format!("typed-content query failed: {error:?}"))?;
                    let result = index
                        .retrieve(typed_query, TOP_K_PER_ACTION)
                        .map_err(|error| format!("typed-content retrieval failed: {error:?}"))?;
                    for hit in result.candidates {
                        retrieved.push((hit.event_id, hit.view_digest, hit.rank));
                    }
                }
            }
        }
    }

    retrieved.sort_by(|left, right| {
        left.2.cmp(&right.2).then_with(|| left.0.cmp(&right.0))
    });
    let mut seen = BTreeSet::new();
    let mut resolved = Vec::new();
    let mut retrieved_ids = Vec::new();
    for (event_id, indexed_digest, _) in retrieved {
        if !seen.insert(event_id) {
            continue;
        }
        let item = by_id
            .get(&event_id)
            .copied()
            .ok_or_else(|| format!("retrieval returned unresolved event_id {event_id}"))?;
        if experience_view_digest(&item) != indexed_digest {
            return Err(format!("canonical evidence view changed for event_id {event_id}"));
        }
        if item.scope != need.scope || !need.action_alternatives.contains(&item.action) {
            return Err(format!("retrieval returned out-of-contract event_id {event_id}"));
        }
        retrieved_ids.push(event_id);
        resolved.push(item);
    }

    let decision = decide(
        Policy::CausalQualified,
        &resolved,
        Query { scope: need.scope },
    );
    Ok((retrieved_ids, profile_sha256, index_payload_bytes, decision))
}

fn decision_name(action: Option<Action>) -> &'static str {
    match action {
        Some(Action::A) => "action_a",
        Some(Action::B) => "action_b",
        None => "abstain",
    }
}

fn execute_request(bytes: &[u8]) -> Result<CandidateReceiptV1, String> {
    if bytes.len() > MAX_REQUEST_BYTES {
        return Err(format!("request exceeds {MAX_REQUEST_BYTES} bytes"));
    }
    let envelope: CandidateEnvelopeV1 =
        serde_json::from_slice(bytes).map_err(|error| format!("invalid candidate envelope: {error}"))?;
    let run_id = envelope.run_id.clone();
    let pair_block_id = envelope.pair_block_id.clone();
    let sequence = envelope.observation_sequence;
    let nonce = envelope.task_nonce.clone();
    let method = envelope.method;
    let (need, memory) = validate_envelope(envelope)?;

    let request_sha256 = hex_digest(Sha256::digest(bytes).as_slice());
    let mut nonce_hasher = Sha256::new();
    nonce_hasher.update(b"mace-candidate-task-nonce-v1\0");
    nonce_hasher.update(nonce.as_bytes());
    let task_nonce_sha256 = hex_digest(nonce_hasher.finalize().as_slice());
    let decision_need_sha256 = need.canonical_digest().to_hex();
    let candidate_evidence_sha256 = evidence_digest(&memory);
    let candidate_input_sha256 = candidate_input_digest(need, &candidate_evidence_sha256);
    let candidate_observation_id = candidate_observation_id(
        &run_id,
        &pair_block_id,
        sequence,
        method.name(),
        &candidate_input_sha256,
    );

    let started = Instant::now();
    let (retrieved_event_ids, retrieval_profile_sha256, index_payload_bytes, decision) =
        run_candidate(method, need, &memory)?;
    let latency_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);

    Ok(CandidateReceiptV1 {
        schema_version: RECEIPT_SCHEMA,
        run_id,
        pair_block_id,
        observation_sequence: sequence,
        task_nonce_sha256,
        method: method.name(),
        request_sha256,
        decision_need_sha256,
        candidate_evidence_sha256,
        candidate_input_sha256,
        candidate_observation_id,
        decision: decision_name(decision.action),
        decision_support_event_ids: decision.evidence_ids,
        retrieved_event_ids,
        retrieval_profile_sha256,
        index_payload_bytes,
        latency_ns,
        probe_requested: decision.probe.is_some(),
    })
}

fn read_bounded_request_line<R: BufRead>(
    reader: &mut R,
    max_bytes: usize,
) -> Result<Option<Vec<u8>>, String> {
    let mut line = Vec::with_capacity(4096);
    let mut terminated = false;
    let mut reached_eof = false;
    loop {
        let available = match reader
            .fill_buf()
            .map_err(|error| format!("read candidate request line: {error}"))?
        {
            [] => {
                reached_eof = true;
                break;
            }
            available => available,
        };
        let newline = available.iter().position(|byte| *byte == b'\n');
        let take = newline.unwrap_or(available.len());
        if line.len().saturating_add(take) > max_bytes {
            return Err(format!("request line exceeds {max_bytes} bytes"));
        }
        line.extend_from_slice(&available[..take]);
        let consumed = take + usize::from(newline.is_some());
        reader.consume(consumed);
        if newline.is_some() {
            terminated = true;
            break;
        }
    }

    if line.is_empty() && reached_eof {
        return Ok(None);
    }
    if !terminated {
        return Err("candidate request ended without a newline delimiter".to_string());
    }
    if line.last() == Some(&b'\r') {
        line.pop();
    }
    if line.is_empty() {
        return Err("empty candidate request line".to_string());
    }
    Ok(Some(line))
}

fn run_stream() -> Result<(), String> {
    let stdin = io::stdin();
    let mut reader = BufReader::new(stdin.lock());
    let stdout = io::stdout();
    let mut writer = stdout.lock();

    // The parent validates this exact line before sending any candidate input.
    writer
        .write_all(b"MACE_CANDIDATE_STREAM_READY_V1\n")
        .map_err(|error| format!("write stream readiness: {error}"))?;
    writer
        .flush()
        .map_err(|error| format!("flush stream readiness: {error}"))?;

    while let Some(line) = read_bounded_request_line(&mut reader, MAX_REQUEST_BYTES)? {
        let receipt = execute_request(&line)?;
        serde_json::to_writer(&mut writer, &receipt)
            .map_err(|error| format!("serialize candidate receipt: {error}"))?;
        writer
            .write_all(b"\n")
            .map_err(|error| format!("terminate candidate receipt line: {error}"))?;
        writer
            .flush()
            .map_err(|error| format!("flush candidate receipt: {error}"))?;
    }
    Ok(())
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("--stream") {
        if let Err(error) = run_stream() {
            eprintln!("mace_candidate: {error}");
            std::process::exit(1);
        }
        return;
    }

    let mut input = Vec::new();
    let read_result = io::stdin()
        .take((MAX_REQUEST_BYTES + 1) as u64)
        .read_to_end(&mut input);
    if let Err(error) = read_result {
        eprintln!("mace_candidate: failed to read request: {error}");
        std::process::exit(2);
    }
    match execute_request(&input) {
        Ok(receipt) => {
            let stdout = io::stdout();
            let mut locked = stdout.lock();
            if let Err(error) = serde_json::to_writer(&mut locked, &receipt) {
                eprintln!("mace_candidate: failed to serialize receipt: {error}");
                std::process::exit(3);
            }
            if let Err(error) = locked.write_all(b"\n") {
                eprintln!("mace_candidate: failed to terminate receipt line: {error}");
                std::process::exit(3);
            }
        }
        Err(error) => {
            eprintln!("mace_candidate: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HASH_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const HASH_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn envelope_value() -> serde_json::Value {
        let qualification = hex_digest(transfer_qualification_profile_digest().as_bytes());
        serde_json::json!({
            "schema_version": ENVELOPE_SCHEMA,
            "run_id": "run-1",
            "pair_block_id": "world-1:task-a",
            "observation_sequence": 7,
            "task_nonce": "nonce-0123456789abcd",
            "method": "exact_scope_scan",
            "decision_need": {
                "schema_version": 1,
                "scope": {"regime": "baseline", "objective": "quality"},
                "action_alternatives": ["a", "b"],
                "semantic_estimand": {
                    "schema_version": 1,
                    "population_spec_sha256": HASH_A,
                    "treatment_version_sha256": HASH_A,
                    "comparator_version_sha256": HASH_A,
                    "outcome_definition_sha256": HASH_A,
                    "outcome_unit_sha256": HASH_A,
                    "follow_up_horizon_ms": 1000,
                    "censoring_strategy_sha256": HASH_A,
                    "intercurrent_event_strategy_sha256": HASH_A,
                    "target_weighting_sha256": HASH_A
                },
                "identification_method": "randomized_controlled_trial",
                "qualification_profile_sha256": qualification
            },
            "candidate_evidence": []
        })
    }

    fn valid_bytes() -> Vec<u8> {
        serde_json::to_vec(&envelope_value()).unwrap()
    }

    #[test]
    fn candidate_envelope_is_accepted_and_receipt_is_task_bound() {
        let bytes = valid_bytes();
        let receipt = execute_request(&bytes).unwrap();
        assert_eq!(receipt.schema_version, RECEIPT_SCHEMA);
        assert_eq!(receipt.decision, "abstain");
        assert_eq!(receipt.run_id, "run-1");
        assert_eq!(receipt.pair_block_id, "world-1:task-a");
        assert_eq!(receipt.observation_sequence, 7);
        assert_eq!(receipt.request_sha256, hex_digest(Sha256::digest(&bytes).as_slice()));
        assert!(receipt.candidate_observation_id.starts_with("obs-"));
    }

    #[test]
    fn unknown_evaluator_fields_are_rejected_at_the_envelope_boundary() {
        let mut value = envelope_value();
        value["truth"] = serde_json::json!({"optimal_action": "b", "utility_a": 1.0});
        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(execute_request(&bytes).unwrap_err().contains("unknown field"));
    }

    #[test]
    fn nested_intervention_and_score_labels_are_rejected() {
        let mut value = envelope_value();
        value["decision_need"]["intervention_kind"] = serde_json::json!("source_root");
        value["candidate_evidence"] = serde_json::json!([{
            "event_id": 1,
            "source_family": 1,
            "source_root_id": 1,
            "source_root_sha256": HASH_A,
            "scope": {"regime": "baseline", "objective": "quality"},
            "action": "a",
            "observed_utility": 0.5,
            "measurement_error_bound": 0.1,
            "causal_identification": {
                "method": "randomized_controlled_trial",
                "design_sha256": HASH_A,
                "estimand_sha256": HASH_A,
                "semantic_estimand": {
                    "schema_version": 1,
                    "population_spec_sha256": HASH_A,
                    "treatment_version_sha256": HASH_A,
                    "comparator_version_sha256": HASH_A,
                    "outcome_definition_sha256": HASH_A,
                    "outcome_unit_sha256": HASH_A,
                    "follow_up_horizon_ms": 1000,
                    "censoring_strategy_sha256": HASH_A,
                    "intercurrent_event_strategy_sha256": HASH_A,
                    "target_weighting_sha256": HASH_A
                },
                "propensity_min_bps": 500,
                "propensity_max_bps": 9500,
                "max_standardized_mean_difference_micros": 100000,
                "interference_controlled": true,
                "score": 0.99
            },
            "intervention": true,
            "prediction_pre_registered": true,
            "outcome_reconciled": true,
            "source_trusted": true,
            "source_retracted": false,
            "confounders_controlled": true,
            "delayed_horizon_complete": true,
            "removed_event_ids": [99]
        }]);
        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(execute_request(&bytes).unwrap_err().contains("unknown field"));
    }

    #[test]
    fn malformed_digest_and_task_nonce_fail_closed() {
        let mut value = envelope_value();
        value["decision_need"]["semantic_estimand"]["population_spec_sha256"] =
            serde_json::json!("deadbeef");
        assert!(execute_request(&serde_json::to_vec(&value).unwrap())
            .unwrap_err()
            .contains("population_spec_sha256"));

        let mut value = envelope_value();
        value["task_nonce"] = serde_json::json!("short");
        assert!(execute_request(&serde_json::to_vec(&value).unwrap())
            .unwrap_err()
            .contains("task_nonce"));
    }

    #[test]
    fn out_of_range_evidence_is_rejected_not_silently_clamped() {
        let mut value = envelope_value();
        let mut item = serde_json::json!({
            "event_id": 5,
            "source_family": 1,
            "source_root_id": 1,
            "source_root_sha256": HASH_A,
            "scope": {"regime": "baseline", "objective": "quality"},
            "action": "a",
            "observed_utility": 1.01,
            "measurement_error_bound": 0.1,
            "causal_identification": {
                "method": "randomized_controlled_trial",
                "design_sha256": HASH_A,
                "estimand_sha256": HASH_A,
                "semantic_estimand": {
                    "schema_version": 1,
                    "population_spec_sha256": HASH_A,
                    "treatment_version_sha256": HASH_A,
                    "comparator_version_sha256": HASH_A,
                    "outcome_definition_sha256": HASH_A,
                    "outcome_unit_sha256": HASH_A,
                    "follow_up_horizon_ms": 1000,
                    "censoring_strategy_sha256": HASH_A,
                    "intercurrent_event_strategy_sha256": HASH_A,
                    "target_weighting_sha256": HASH_A
                },
                "propensity_min_bps": 500,
                "propensity_max_bps": 9500,
                "max_standardized_mean_difference_micros": 100000,
                "interference_controlled": true
            },
            "intervention": true,
            "prediction_pre_registered": true,
            "outcome_reconciled": true,
            "source_trusted": true,
            "source_retracted": false,
            "confounders_controlled": true,
            "delayed_horizon_complete": true
        });
        value["candidate_evidence"] = serde_json::json!([item]);
        assert!(execute_request(&serde_json::to_vec(&value).unwrap())
            .unwrap_err()
            .contains("observed_utility"));

        item["observed_utility"] = serde_json::json!(0.5);
        item["scope"]["regime"] = serde_json::json!("shifted");
        let mut out_of_scope = envelope_value();
        out_of_scope["candidate_evidence"] = serde_json::json!([item]);
        assert!(execute_request(&serde_json::to_vec(&out_of_scope).unwrap())
            .unwrap_err()
            .contains("outside DecisionNeedV1 scope/action alternatives"));
    }

    #[test]
    fn duplicate_event_ids_and_record_cap_are_rejected() {
        let mut value = envelope_value();
        let sample = serde_json::json!({
            "event_id": 5,
            "source_family": 1,
            "source_root_id": 1,
            "source_root_sha256": HASH_A,
            "scope": {"regime": "baseline", "objective": "quality"},
            "action": "a",
            "observed_utility": 0.5,
            "measurement_error_bound": 0.1,
            "causal_identification": {
                "method": "randomized_controlled_trial",
                "design_sha256": HASH_A,
                "estimand_sha256": HASH_A,
                "semantic_estimand": {
                    "schema_version": 1,
                    "population_spec_sha256": HASH_A,
                    "treatment_version_sha256": HASH_A,
                    "comparator_version_sha256": HASH_A,
                    "outcome_definition_sha256": HASH_A,
                    "outcome_unit_sha256": HASH_A,
                    "follow_up_horizon_ms": 1000,
                    "censoring_strategy_sha256": HASH_A,
                    "intercurrent_event_strategy_sha256": HASH_A,
                    "target_weighting_sha256": HASH_A
                },
                "propensity_min_bps": 500,
                "propensity_max_bps": 9500,
                "max_standardized_mean_difference_micros": 100000,
                "interference_controlled": true
            },
            "intervention": true,
            "prediction_pre_registered": true,
            "outcome_reconciled": true,
            "source_trusted": true,
            "source_retracted": false,
            "confounders_controlled": true,
            "delayed_horizon_complete": true
        });
        value["candidate_evidence"] = serde_json::json!([sample.clone(), sample]);
        assert!(execute_request(&serde_json::to_vec(&value).unwrap())
            .unwrap_err()
            .contains("duplicate candidate event_id"));

        let mut value = envelope_value();
        let sample = serde_json::json!({
            "event_id": 6,
            "source_family": 1,
            "source_root_id": 1,
            "source_root_sha256": HASH_A,
            "scope": {"regime": "baseline", "objective": "quality"},
            "action": "a",
            "observed_utility": 0.5,
            "measurement_error_bound": 0.1,
            "causal_identification": {
                "method": "randomized_controlled_trial",
                "design_sha256": HASH_A,
                "estimand_sha256": HASH_A,
                "semantic_estimand": {
                    "schema_version": 1,
                    "population_spec_sha256": HASH_A,
                    "treatment_version_sha256": HASH_A,
                    "comparator_version_sha256": HASH_A,
                    "outcome_definition_sha256": HASH_A,
                    "outcome_unit_sha256": HASH_A,
                    "follow_up_horizon_ms": 1000,
                    "censoring_strategy_sha256": HASH_A,
                    "intercurrent_event_strategy_sha256": HASH_A,
                    "target_weighting_sha256": HASH_A
                },
                "propensity_min_bps": 500,
                "propensity_max_bps": 9500,
                "max_standardized_mean_difference_micros": 100000,
                "interference_controlled": true
            },
            "intervention": true,
            "prediction_pre_registered": true,
            "outcome_reconciled": true,
            "source_trusted": true,
            "source_retracted": false,
            "confounders_controlled": true,
            "delayed_horizon_complete": true
        });
        value["candidate_evidence"] =
            serde_json::json!(vec![sample; MAX_EVIDENCE_RECORDS + 1]);
        assert!(execute_request(&serde_json::to_vec(&value).unwrap())
            .unwrap_err()
            .contains("exceeds"));
    }

    #[test]
    fn request_byte_cap_is_enforced_before_json_parsing() {
        let bytes = vec![b' '; MAX_REQUEST_BYTES + 1];
        assert!(execute_request(&bytes).unwrap_err().contains("request exceeds"));
    }

    #[test]
    fn observation_identity_changes_with_sequence_and_evidence_changes_input() {
        let first = candidate_observation_id("run-1", "task-1", 1, "exact_scope_scan", HASH_A);
        let second = candidate_observation_id("run-1", "task-1", 2, "exact_scope_scan", HASH_A);
        assert_ne!(first, second);
        let need = DecisionNeedWire {
            schema_version: 1,
            scope: ScopeWire { regime: RegimeWire::Baseline, objective: ObjectiveWire::Quality },
            action_alternatives: [ActionWire::A, ActionWire::B],
            semantic_estimand: SemanticEstimandWire {
                schema_version: 1,
                population_spec_sha256: HASH_A.into(),
                treatment_version_sha256: HASH_A.into(),
                comparator_version_sha256: HASH_A.into(),
                outcome_definition_sha256: HASH_A.into(),
                outcome_unit_sha256: HASH_A.into(),
                follow_up_horizon_ms: 1000,
                censoring_strategy_sha256: HASH_A.into(),
                intercurrent_event_strategy_sha256: HASH_A.into(),
                target_weighting_sha256: HASH_A.into(),
            },
            identification_method: IdentificationMethodWire::RandomizedControlledTrial,
            qualification_profile_sha256: hex_digest(transfer_qualification_profile_digest().as_bytes()),
        }.to_domain().unwrap();
        assert_ne!(
            candidate_input_digest(need, HASH_A),
            candidate_input_digest(need, HASH_B)
        );
    }
    #[test]
    fn bounded_stream_reader_accepts_exact_size_and_clean_eof() {
        let mut exact = std::io::Cursor::new(b"12345\n");
        assert_eq!(
            read_bounded_request_line(&mut exact, 5).unwrap(),
            Some(b"12345".to_vec())
        );
        assert_eq!(read_bounded_request_line(&mut exact, 5).unwrap(), None);
    }

    #[test]
    fn bounded_stream_reader_rejects_oversize_and_unterminated_frames() {
        let mut oversize = std::io::Cursor::new(b"123456\n");
        assert!(read_bounded_request_line(&mut oversize, 5)
            .unwrap_err()
            .contains("exceeds 5 bytes"));

        let mut unterminated = std::io::Cursor::new(b"12345");
        assert!(read_bounded_request_line(&mut unterminated, 5)
            .unwrap_err()
            .contains("without a newline delimiter"));
    }

    #[test]
    fn bounded_stream_reader_rejects_empty_frames() {
        let mut empty = std::io::Cursor::new(b"\n");
        assert!(read_bounded_request_line(&mut empty, 5)
            .unwrap_err()
            .contains("empty candidate request line"));
    }

    #[test]
    fn typed_hdc_empty_evidence_abstains_without_building_an_index() {
        let mut value = envelope_value();
        value["method"] = serde_json::json!("typed_content_hdc");
        let receipt = execute_request(&serde_json::to_vec(&value).unwrap()).unwrap();
        assert_eq!(receipt.decision, "abstain");
        assert_eq!(receipt.retrieval_profile_sha256, None);
        assert_eq!(receipt.index_payload_bytes, 0);
    }

    #[test]
    fn reversed_action_order_is_rejected_as_a_noncanonical_task_contract() {
        let mut value = envelope_value();
        value["decision_need"]["action_alternatives"] = serde_json::json!(["b", "a"]);
        let error = execute_request(&serde_json::to_vec(&value).unwrap()).unwrap_err();
        assert!(error.contains("decision_need is invalid"));
    }

}
