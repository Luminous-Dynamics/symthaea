// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed JSONL bridge for the existing MACE DSM paired scoring core.
//! This binary scores candidate outputs; it does not execute the candidate or
//! establish private-label custody, process isolation, or benchmark validity.

use mace_dsm_evaluator::{score_paired, ActionLossV1, CandidateRunV1, EvaluatorCaseV1};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    io::{BufRead, BufReader},
};

const CANDIDATE_SCHEMA: &str = "mace_dsm_candidate_episode_v0_1";
const PRIVATE_SCHEMA: &str = "mace_dsm_evaluator_case_v0_1";
const RESPONSE_SCHEMA: &str = "mace_dsm_candidate_run_v0_1";
const MANIFEST_SCHEMA: &str = "mace_dsm_run_manifest_v0_1";
const CANDIDATE_DOMAIN: &[u8] = b"luminous:mace:dsm:candidate-record:v0.1\0";
const ARM_DOMAIN: &[u8] = b"luminous:mace:dsm:arm-response:v0.1\0";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateEpisode {
    schema_version: String,
    episode_id: String,
    query_time_ms: i64,
    query: String,
    scope: String,
    source_events: Vec<SourceEvent>,
    allowed_actions: Vec<String>,
    evidence_budget: usize,
    observation_budget: usize,
    observation_cost_units: f64,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceEvent {
    event_id: String,
    source_id: String,
    observed_at_ms: i64,
    valid_from_ms: i64,
    valid_until_ms: Option<i64>,
    authority_tier: AuthorityTier,
    uncertainty_ppm: u32,
    claim: String,
    relation_to_parent: Relation,
    parent_event_ids: Vec<String>,
}
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum AuthorityTier { Verified, Authenticated, UserAsserted, Untrusted }
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Relation { Asserts, Supersedes, Retracts, Contradicts, DerivedFrom }

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvaluatorCaseWire {
    schema_version: String,
    episode_id: String,
    candidate_sha256: String,
    action_losses: Vec<ActionLossWire>,
    no_memory_action: String,
    oracle_action: String,
    admissible_evidence_ids: Vec<String>,
    required_evidence_ids: Vec<String>,
    #[serde(default)]
    forbidden_evidence_ids: Vec<String>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ActionLossWire { action: String, loss_micros: u64 }

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateRunWire {
    schema_version: String,
    run_id: String,
    episode_id: String,
    candidate_sha256: String,
    arm_binding_sha256: String,
    system_id: String,
    downstream_policy_digest: String,
    decision_seed: u64,
    chosen_action: String,
    retrieved_evidence_ids: Vec<String>,
    #[serde(default)]
    selected_evidence_ids: Vec<String>,
    observation_requests: usize,
    #[serde(default)]
    pair_block_id: Option<String>,
    #[serde(default)]
    world_cluster_id: Option<String>,
    #[serde(default)]
    query_id: Option<String>,
    #[serde(default)]
    cost_receipt: Option<CostReceiptWire>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CostReceiptWire {
    #[serde(default)]
    latency_ns: Option<u64>,
    #[serde(default)]
    phase_timings_ns: Option<PhaseTimingsWire>,
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    model_calls: Option<u64>,
    #[serde(default)]
    tool_calls: Option<u64>,
    #[serde(default)]
    external_tool_calls: Option<u64>,
    #[serde(default)]
    index_bytes: Option<u64>,
    #[serde(default)]
    index_bytes_kind: Option<String>,
    #[serde(default)]
    evidence_sidecar_bytes: Option<u64>,
    #[serde(default)]
    evidence_sidecar_bytes_kind: Option<String>,
    #[serde(default)]
    unmeasured_cost_stages: Vec<String>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PhaseTimingsWire {
    #[serde(default)]
    memory_selection_ns: Option<u64>,
    #[serde(default)]
    index_build_ns: Option<u64>,
    #[serde(default)]
    retrieval_ns: Option<u64>,
    #[serde(default)]
    evidence_resolution_verification_ns: Option<u64>,
    #[serde(default)]
    policy_application_ns: Option<u64>,
    #[serde(default)]
    unattributed_coordinator_ns: Option<u64>,
    #[serde(default)]
    total_wall_ns: Option<u64>,
}

#[derive(Clone, Debug, Deserialize)]
struct RunManifestWire {
    schema_version: String,
    run_id: String,
    profile: String,
    downstream_policy: DownstreamPolicyWire,
    evaluation: EvaluationWire,
}
#[derive(Clone, Debug, Deserialize)]
struct DownstreamPolicyWire { policy_digest_sha256: String, reader_seed: u64 }
#[derive(Clone, Debug, Deserialize)]
struct EvaluationWire { metric_profile: String }

#[derive(Clone, Debug)]
struct CandidateRecord { episode: CandidateEpisode, digest: String }
#[derive(Clone, Debug)]
struct PairRow {
    episode_id: String,
    memory_action: String,
    no_memory_action: String,
    memory_loss: u64,
    no_memory_loss: u64,
    oracle_loss: u64,
    regret: u64,
    gain: i128,
    harm: bool,
    admissible_retrieved: u64,
    inadmissible_retrieved: u64,
    required_recall: Option<(u64, u64)>,
    admissible_precision: Option<(u64, u64)>,
    selected_count: usize,
    selected_invalid: usize,
    selected_required: usize,
    required_count: usize,
    conflicts: usize,
    sufficient: bool,
    memory_cost: Option<CostReceiptWire>,
    no_memory_cost: Option<CostReceiptWire>,
    pair_block_id: Option<String>,
    world_cluster_id: Option<String>,
    query_id: Option<String>,
}

fn fail(message: impl AsRef<str>) -> ! {
    eprintln!("error: {}", message.as_ref());
    std::process::exit(2);
}
fn hex(bytes: &[u8]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(TABLE[usize::from(byte >> 4)] as char);
        output.push(TABLE[usize::from(byte & 15)] as char);
    }
    output
}
fn sha256(bytes: &[u8]) -> String { hex(&Sha256::digest(bytes)) }
fn candidate_digest(raw_line: &str) -> String {
    let bytes = raw_line.as_bytes();
    let mut h = Sha256::new();
    h.update(CANDIDATE_DOMAIN);
    h.update((bytes.len() as u64).to_be_bytes());
    h.update(bytes);
    hex(&h.finalize())
}
fn arm_digest(candidate_sha256: &str, system_id: &str) -> String {
    let mut h = Sha256::new();
    h.update(ARM_DOMAIN);
    h.update(candidate_sha256.as_bytes());
    h.update([0]);
    h.update(system_id.as_bytes());
    hex(&h.finalize())
}
fn read_jsonl<T: DeserializeOwned>(path: &str) -> Result<Vec<(T, String)>, String> {
    let file = fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let mut rows = Vec::new();
    for (line_number, line) in BufReader::new(file).lines().enumerate() {
        let raw = line.map_err(|e| format!("read {path} line {}: {e}", line_number + 1))?;
        if raw.trim().is_empty() { return Err(format!("{path}: blank JSONL line {}", line_number + 1)); }
        let value = serde_json::from_str(&raw)
            .map_err(|e| format!("{path} line {}: {e}", line_number + 1))?;
        rows.push((value, raw));
    }
    if rows.is_empty() { return Err(format!("{path}: no JSONL rows")); }
    Ok(rows)
}
fn read_json<T: DeserializeOwned>(path: &str) -> Result<T, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("parse {path}: {e}"))
}

fn validate_ancestry(episode: &CandidateEpisode) -> Result<(), String> {
    let mut parents_by_id: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for event in &episode.source_events {
        if event.event_id.trim().is_empty() || event.source_id.trim().is_empty()
            || event.claim.trim().is_empty() || event.uncertainty_ppm > 1_000_000
            || event.valid_until_ms.is_some_and(|end| end < event.valid_from_ms) {
            return Err(format!("{}: malformed source event {}", episode.episode_id, event.event_id));
        }
        if parents_by_id.insert(event.event_id.clone(), event.parent_event_ids.clone()).is_some() {
            return Err(format!("{}: duplicate source event ID {}", episode.episode_id, event.event_id));
        }
    }
    for (event_id, parents) in &parents_by_id {
        let unique: BTreeSet<&str> = parents.iter().map(String::as_str).collect();
        if unique.len() != parents.len() || parents.iter().any(|p| p == event_id || !parents_by_id.contains_key(p)) {
            return Err(format!("{}: unresolved, self or duplicate parent on {event_id}", episode.episode_id));
        }
    }
    fn visit(
        id: &str,
        parents: &BTreeMap<String, Vec<String>>,
        active: &mut BTreeSet<String>,
        done: &mut BTreeSet<String>,
    ) -> Result<(), String> {
        if done.contains(id) { return Ok(()); }
        if !active.insert(id.to_owned()) { return Err(format!("cyclic ancestry at {id}")); }
        if let Some(items) = parents.get(id) {
            for parent in items { visit(parent, parents, active, done)?; }
        }
        active.remove(id);
        done.insert(id.to_owned());
        Ok(())
    }
    let mut active = BTreeSet::new();
    let mut done = BTreeSet::new();
    for id in parents_by_id.keys() { visit(id, &parents_by_id, &mut active, &mut done)
        .map_err(|e| format!("{}: {e}", episode.episode_id))?; }
    Ok(())
}
fn validate_episode(episode: &CandidateEpisode) -> Result<(), String> {
    if episode.schema_version != CANDIDATE_SCHEMA || episode.episode_id.trim().is_empty()
        || episode.query.trim().is_empty() || episode.scope.trim().is_empty()
        || episode.allowed_actions.is_empty() || episode.allowed_actions.iter().any(|a| a.trim().is_empty())
        || !episode.observation_cost_units.is_finite() || episode.observation_cost_units < 0.0
        || episode.evidence_budget > 1_000 || episode.observation_budget > 100 {
        return Err(format!("{}: malformed candidate episode", episode.episode_id));
    }
    let actions: BTreeSet<&str> = episode.allowed_actions.iter().map(String::as_str).collect();
    if actions.len() != episode.allowed_actions.len() {
        return Err(format!("{}: duplicate allowed action", episode.episode_id));
    }
    validate_ancestry(episode)
}
fn current_at(event: &SourceEvent, query_time_ms: i64) -> bool {
    event.observed_at_ms <= query_time_ms && event.valid_from_ms <= query_time_ms
        && event.valid_until_ms.is_none_or(|end| query_time_ms < end)
}
fn invalidated(id: &str, events: &[SourceEvent], at: i64) -> bool {
    events.iter().any(|event| event.authority_tier != AuthorityTier::Untrusted
        && event.observed_at_ms <= at && event.valid_from_ms <= at
        && matches!(event.relation_to_parent, Relation::Supersedes | Relation::Retracts)
        && event.parent_event_ids.iter().any(|parent| parent == id))
}
fn derived_admissible(episode: &CandidateEpisode) -> BTreeSet<String> {
    episode.source_events.iter().filter(|event| current_at(event, episode.query_time_ms)
        && !invalidated(&event.event_id, &episode.source_events, episode.query_time_ms)
        && event.authority_tier != AuthorityTier::Untrusted)
        .map(|event| event.event_id.clone()).collect()
}
fn contradiction_pairs(retrieved: &[String], episode: &CandidateEpisode) -> usize {
    let returned: BTreeSet<&str> = retrieved.iter().map(String::as_str).collect();
    episode.source_events.iter()
        .filter(|event| event.relation_to_parent == Relation::Contradicts
            && event.authority_tier != AuthorityTier::Untrusted
            && current_at(event, episode.query_time_ms)
            && returned.contains(event.event_id.as_str()))
        .map(|event| event.parent_event_ids.iter()
            .filter(|parent| returned.contains(parent.as_str())).count())
        .sum()
}
fn validate_cost(receipt: &CostReceiptWire) -> Result<&'static str, String> {
    let Some(t) = receipt.phase_timings_ns.as_ref() else { return Ok("not_reported"); };
    let components = [t.memory_selection_ns, t.index_build_ns, t.retrieval_ns,
        t.evidence_resolution_verification_ns, t.policy_application_ns, t.unattributed_coordinator_ns];
    match (components.iter().all(Option::is_some), t.total_wall_ns) {
        (true, Some(wall)) => {
            let sum: u128 = components.iter().map(|v| u128::from(v.unwrap_or_default())).sum();
            if sum != u128::from(wall) { return Err(format!("phase sum {sum} != wall time {wall}")); }
            Ok("reconciled")
        }
        _ => Ok("incomplete_missing_phase_or_wall"),
    }
}
fn ratio(value: Option<(u64, u64)>) -> Value {
    match value {
        Some((n, d)) => json!({"numerator":n,"denominator":d,"value":if d == 0 {Value::Null} else {json!(n as f64 / d as f64)}}),
        None => Value::Null,
    }
}

fn score(
    candidate_path: &str, private_path: &str, response_path: &str, manifest_path: &str,
    memory_id: &str, no_memory_id: &str, report_path: &str,
) -> Result<(), String> {
    if memory_id.trim().is_empty() || no_memory_id.trim().is_empty() || memory_id == no_memory_id {
        return Err("memory and no-memory system IDs must be nonempty and distinct".into());
    }
    let manifest: RunManifestWire = read_json(manifest_path)?;
    if manifest.schema_version != MANIFEST_SCHEMA || manifest.profile != "native-research"
        || manifest.evaluation.metric_profile != "dsm-v0.1"
        || manifest.downstream_policy.policy_digest_sha256.trim().is_empty() {
        return Err("run manifest is not a native DSM v0.1 manifest".into());
    }

    let mut candidates = BTreeMap::new();
    for (episode, raw) in read_jsonl::<CandidateEpisode>(candidate_path)? {
        validate_episode(&episode)?;
        let digest = candidate_digest(&raw);
        let id = episode.episode_id.clone();
        if candidates.insert(id.clone(), CandidateRecord {episode,digest}).is_some() {
            return Err(format!("duplicate candidate episode {id}"));
        }
    }

    let mut private_cases = BTreeMap::new();
    for (case, _) in read_jsonl::<EvaluatorCaseWire>(private_path)? {
        if case.schema_version != PRIVATE_SCHEMA { return Err(format!("{}: invalid private schema",case.episode_id)); }
        let id = case.episode_id.clone();
        if private_cases.insert(id.clone(),case).is_some() { return Err(format!("duplicate private episode {id}")); }
    }
    if candidates.len()!=private_cases.len() || candidates.keys().any(|id|!private_cases.contains_key(id)) {
        return Err("candidate and evaluator-private episode inventories differ".into());
    }

    let mut responses: BTreeMap<(String,String),CandidateRunWire> = BTreeMap::new();
    for (response,_) in read_jsonl::<CandidateRunWire>(response_path)? {
        let id=&response.episode_id;
        if response.schema_version!=RESPONSE_SCHEMA || response.run_id!=manifest.run_id
            || response.downstream_policy_digest!=manifest.downstream_policy.policy_digest_sha256
            || response.decision_seed!=manifest.downstream_policy.reader_seed {
            return Err(format!("{id}: response/run-manifest identity mismatch"));
        }
        let candidate=candidates.get(id).ok_or_else(||format!("unknown response episode {id}"))?;
        if response.candidate_sha256!=candidate.digest
            || response.arm_binding_sha256!=arm_digest(&candidate.digest,&response.system_id) {
            return Err(format!("{id}: candidate or system binding digest mismatch"));
        }
        if response.system_id!=memory_id && response.system_id!=no_memory_id {
            return Err(format!("{id}: undeclared system ID {}",response.system_id));
        }
        if response.retrieved_evidence_ids.len()>candidate.episode.evidence_budget
            || response.observation_requests>candidate.episode.observation_budget {
            return Err(format!("{id}: evidence or observation budget exceeded"));
        }
        let retrieved:BTreeSet<&str>=response.retrieved_evidence_ids.iter().map(String::as_str).collect();
        let selected:BTreeSet<&str>=response.selected_evidence_ids.iter().map(String::as_str).collect();
        if retrieved.len()!=response.retrieved_evidence_ids.len()
            || selected.len()!=response.selected_evidence_ids.len()
            || selected.iter().any(|item|!retrieved.contains(item)) {
            return Err(format!("{id}: evidence IDs duplicate or selected evidence is not retrieved"));
        }
        if let Some(receipt)=response.cost_receipt.as_ref(){validate_cost(receipt).map_err(|e|format!("{id}: {e}"))?;}
        let key=(response.episode_id.clone(),response.system_id.clone());
        if responses.insert(key,response).is_some(){return Err("duplicate episode/system response".into());}
    }
    let expected=candidates.len().checked_mul(2).ok_or("response matrix size overflow")?;
    if responses.len()!=expected{return Err(format!("incomplete paired response matrix: expected {expected}, found {}",responses.len()));}

    let mut rows=Vec::new();
    let mut gain_sum=0i128;
    let mut harms=0u64;
    let mut sufficient_count=0u64;
    for (id,record) in &candidates {
        let private=&private_cases[id];
        if private.candidate_sha256!=record.digest{return Err(format!("{id}: private truth not bound to candidate line"));}
        let derived=derived_admissible(&record.episode);
        let admissible:BTreeSet<String>=private.admissible_evidence_ids.iter().cloned().collect();
        if admissible.len()!=private.admissible_evidence_ids.len()||derived!=admissible {
            return Err(format!("{id}: evaluator admissibility labels disagree with candidate bitemporal/source rules"));
        }
        let required:BTreeSet<&str>=private.required_evidence_ids.iter().map(String::as_str).collect();
        let forbidden:BTreeSet<&str>=private.forbidden_evidence_ids.iter().map(String::as_str).collect();
        let event_ids:BTreeSet<&str>=record.episode.source_events.iter().map(|e|e.event_id.as_str()).collect();
        if required.len()!=private.required_evidence_ids.len()||forbidden.len()!=private.forbidden_evidence_ids.len()
            ||required.iter().any(|x|!admissible.contains(**x))
            ||forbidden.iter().any(|x|admissible.contains(**x))
            ||required.iter().chain(forbidden.iter()).any(|x|!event_ids.contains(x)) {
            return Err(format!("{id}: inconsistent private required/forbidden labels"));
        }
        let memory=responses.get(&(id.clone(),memory_id.to_owned())).ok_or_else(||format!("{id}: missing memory response"))?;
        let baseline=responses.get(&(id.clone(),no_memory_id.to_owned())).ok_or_else(||format!("{id}: missing no-memory response"))?;
        for (label,a,b) in [("pair_block_id",&memory.pair_block_id,&baseline.pair_block_id),
            ("world_cluster_id",&memory.world_cluster_id,&baseline.world_cluster_id),
            ("query_id",&memory.query_id,&baseline.query_id)] {
            if a!=b{return Err(format!("{id}: paired arms disagree on {label}"));}
        }
        if !baseline.retrieved_evidence_ids.is_empty()||!baseline.selected_evidence_ids.is_empty() {
            return Err(format!("{id}: no-memory arm returned or selected evidence"));
        }
        if memory.selected_evidence_ids.iter().chain(baseline.selected_evidence_ids.iter())
            .any(|e|forbidden.contains(e.as_str())) {
            return Err(format!("{id}: selected evidence contains evaluator-forbidden ID"));
        }

        let case=EvaluatorCaseV1 {
            episode_id:id.clone(),
            allowed_actions:record.episode.allowed_actions.clone(),
            action_losses:private.action_losses.iter().map(|loss|ActionLossV1 {
                action:loss.action.clone(),loss_micros:loss.loss_micros
            }).collect(),
            no_memory_action:private.no_memory_action.clone(),
            oracle_action:private.oracle_action.clone(),
            admissible_evidence_ids:private.admissible_evidence_ids.clone(),
            required_evidence_ids:private.required_evidence_ids.clone(),
        };
        let to_core=|run:&CandidateRunWire|CandidateRunV1 {
            episode_id:id.clone(),system_id:run.system_id.clone(),
            downstream_policy_digest:run.downstream_policy_digest.clone(),
            decision_seed:run.decision_seed,chosen_action:run.chosen_action.clone(),
            retrieved_evidence_ids:run.retrieved_evidence_ids.clone(),
        };
        let paired=score_paired(&case,&to_core(memory),&to_core(baseline))
            .map_err(|e|format!("{id}: paired scoring rejected response: {e:?}"))?;

        let selected:BTreeSet<&str>=memory.selected_evidence_ids.iter().map(String::as_str).collect();
        let selected_invalid=selected.iter().filter(|item|!admissible.contains(**item)).count();
        let selected_required=required.intersection(&selected).count();
        let conflicts=contradiction_pairs(&memory.retrieved_evidence_ids,&record.episode);
        let sufficient=paired.memory.decision_optimal
            &&paired.memory.inadmissible_returned_evidence_count==0
            &&selected_invalid==0&&selected_required==required.len()&&conflicts==0;
        let gain=paired.memory_gain_micros;
        gain_sum=gain_sum.checked_add(gain).ok_or("aggregate gain overflow")?;
        if paired.memory_induced_harm{harms+=1;}
        if sufficient{sufficient_count+=1;}
        rows.push(PairRow {
            episode_id:id.clone(),memory_action:paired.memory.chosen_action,
            no_memory_action:paired.no_memory.chosen_action,
            memory_loss:paired.memory.decision_loss_micros,no_memory_loss:paired.no_memory.decision_loss_micros,
            oracle_loss:paired.memory.oracle_loss_micros,regret:paired.memory.decision_regret_micros,
            gain,harm:paired.memory_induced_harm,
            admissible_retrieved:paired.memory.admissible_returned_evidence_count,
            inadmissible_retrieved:paired.memory.inadmissible_returned_evidence_count,
            required_recall:paired.memory.required_evidence_recall.map(|x|(x.numerator,x.denominator)),
            admissible_precision:paired.memory.admissible_evidence_precision.map(|x|(x.numerator,x.denominator)),
            selected_count:memory.selected_evidence_ids.len(),selected_invalid,
            selected_required,required_count:required.len(),conflicts,sufficient,
            memory_cost:memory.cost_receipt.clone(),no_memory_cost:baseline.cost_receipt.clone(),
            pair_block_id:memory.pair_block_id.clone(),world_cluster_id:memory.world_cluster_id.clone(),
            query_id:memory.query_id.clone()
        });
    }
    let n=rows.len() as u64;
    let output_rows:Vec<Value>=rows.iter().map(|row|json!({
        "episode_id":row.episode_id,"memory_action":row.memory_action,"no_memory_action":row.no_memory_action,
        "memory_loss_micros":row.memory_loss,"no_memory_loss_micros":row.no_memory_loss,
        "oracle_loss_micros":row.oracle_loss,"memory_regret_micros":row.regret,
        "memory_gain_micros":row.gain.to_string(),"memory_induced_harm":row.harm,
        "admissible_retrieved_evidence_count":row.admissible_retrieved,
        "inadmissible_retrieved_evidence_count":row.inadmissible_retrieved,
        "required_evidence_recall":ratio(row.required_recall),
        "admissible_evidence_precision":ratio(row.admissible_precision),
        "selected_evidence_count":row.selected_count,"selected_invalid_count":row.selected_invalid,
        "selected_required_count":row.selected_required,"required_evidence_count":row.required_count,
        "unresolved_contradiction_pairs":row.conflicts,"decision_sufficient":row.sufficient,
        "pair_block_id":row.pair_block_id,"world_cluster_id":row.world_cluster_id,"query_id":row.query_id,
        "memory_cost_receipt":row.memory_cost,"no_memory_cost_receipt":row.no_memory_cost
    })).collect();
    let report=json!({
        "schema_version":"mace_dsm_paired_report_v0_1",
        "qualification_status":"scorer_completed_unvalidated_rust_execution",
        "run_id":manifest.run_id,
        "run_manifest_sha256":sha256(&fs::read(manifest_path).map_err(|e|e.to_string())?),
        "candidate_jsonl_sha256":sha256(&fs::read(candidate_path).map_err(|e|e.to_string())?),
        "evaluator_private_jsonl_sha256":sha256(&fs::read(private_path).map_err(|e|e.to_string())?),
        "candidate_runs_jsonl_sha256":sha256(&fs::read(response_path).map_err(|e|e.to_string())?),
        "downstream_policy_digest":manifest.downstream_policy.policy_digest_sha256,
        "decision_seed":manifest.downstream_policy.reader_seed,
        "memory_system_id":memory_id,"no_memory_system_id":no_memory_id,
        "episode_count":n,
        "sum_memory_gain_micros":gain_sum.to_string(),
        "mean_memory_gain_micros":if n==0 {Value::Null}else{json!(gain_sum as f64/n as f64)},
        "memory_induced_harm_rate":if n==0 {Value::Null}else{json!(harms as f64/n as f64)},
        "decision_sufficiency_rate":if n==0 {Value::Null}else{json!(sufficient_count as f64/n as f64)},
        "cluster_metadata_coverage":{
            "pair_block_id_count":rows.iter().filter(|r|r.pair_block_id.is_some()).count(),
            "world_cluster_id_count":rows.iter().filter(|r|r.world_cluster_id.is_some()).count(),
            "query_id_count":rows.iter().filter(|r|r.query_id.is_some()).count(),
            "note":"This CLI does not perform cluster bootstrap; reuse the separately qualified paired analysis implementation."
        },
        "paired_scores":output_rows,
        "limitations":[
            "Evaluator-private labels must remain inaccessible to the candidate process.",
            "SHA-256 values bind exact candidate line bytes and system identity; they are not signatures or custody proofs.",
            "The public schema has episode-level scope only; per-event and cross-role authorization are not qualified.",
            "Decision sufficiency is a conservative diagnostic, not a substitute for independent outcome or authority verification.",
            "Cost receipts are candidate-reported; missing counters remain missing and unmeasured stages are not zero.",
            "The CLI scores receipts; it does not execute a memory backend, reader, observation probe, action environment or leave-one-out rerun."
        ]
    });
    fs::write(report_path,serde_json::to_vec_pretty(&report).map_err(|e|e.to_string())?)
        .map_err(|e|format!("write {report_path}: {e}"))
}

fn usage()->&'static str {
    "MACE DSM reference runner\n\nscore <candidate-episodes.jsonl> <evaluator-private.jsonl> <candidate-runs.jsonl> <run-manifest.json> <memory-system-id> <no-memory-system-id> <report.json>\n"
}
fn run()->Result<(),String>{
    let args:Vec<String>=env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("score") if args.len()==9=>score(&args[2],&args[3],&args[4],&args[5],&args[6],&args[7],&args[8]),
        _=>{eprintln!("{}",usage());Err("invalid command or argument count".into())}
    }
}
fn main(){if let Err(e)=run(){fail(e);}}

#[cfg(test)]
mod tests{
    use super::*;
    fn event(id:&str,observed:i64,from:i64,until:Option<i64>,authority:AuthorityTier,relation:Relation,parents:&[&str])->SourceEvent{
        SourceEvent{event_id:id.into(),source_id:"source".into(),observed_at_ms:observed,valid_from_ms:from,valid_until_ms:until,
            authority_tier:authority,uncertainty_ppm:0,claim:"claim".into(),relation_to_parent:relation,
            parent_event_ids:parents.iter().map(|x|(*x).into()).collect()}
    }
    #[test] fn candidate_digest_binds_exact_record_bytes(){
        assert_ne!(candidate_digest("{\"episode_id\":\"x\"}"),candidate_digest("{\"episode_id\": \"x\"}"));
    }
    #[test] fn arm_digest_binds_system_identity(){
        let d=candidate_digest("{\"episode_id\":\"x\"}");
        assert_ne!(arm_digest(&d,"memory-a"),arm_digest(&d,"memory-b"));
    }
    #[test] fn knowledge_time_and_valid_time_are_both_enforced(){
        let e=event("e",10,10,Some(20),AuthorityTier::Verified,Relation::Asserts,&[]);
        assert!(!current_at(&e,9));assert!(current_at(&e,10));assert!(current_at(&e,19));assert!(!current_at(&e,20));
    }
    #[test] fn untrusted_retraction_does_not_invalidate_parent(){
        let p=event("p",1,0,None,AuthorityTier::Verified,Relation::Asserts,&[]);
        let r=event("r",2,0,None,AuthorityTier::Untrusted,Relation::Retracts,&["p"]);
        assert!(!invalidated("p",&[p,r],2));
    }
    #[test] fn trusted_supersession_applies_only_after_observation_and_effective_time(){
        let p=event("p",1,0,None,AuthorityTier::Verified,Relation::Asserts,&[]);
        let s=event("s",10,10,None,AuthorityTier::Verified,Relation::Supersedes,&["p"]);
        assert!(!invalidated("p",&[p.clone(),s.clone()],9));assert!(invalidated("p",&[p,s],10));
    }
    #[test] fn retrieved_contradictions_are_detected(){
        let p=event("p",1,0,None,AuthorityTier::Verified,Relation::Asserts,&[]);
        let c=event("c",2,0,None,AuthorityTier::Verified,Relation::Contradicts,&["p"]);
        assert_eq!(contradiction_pairs(&["p".into(),"c".into()],&CandidateEpisode{
            schema_version:CANDIDATE_SCHEMA.into(),episode_id:"x".into(),query_time_ms:3,query:"q".into(),scope:"s".into(),
            source_events:vec![p,c],allowed_actions:vec!["hold".into()],evidence_budget:2,observation_budget:0,observation_cost_units:0.0
        }),1);
    }
    #[test] fn cyclic_ancestry_is_rejected(){
        let e=CandidateEpisode{schema_version:CANDIDATE_SCHEMA.into(),episode_id:"x".into(),query_time_ms:3,query:"q".into(),scope:"s".into(),
            source_events:vec![event("a",1,0,None,AuthorityTier::Verified,Relation::DerivedFrom,&["b"]),
                event("b",1,0,None,AuthorityTier::Verified,Relation::DerivedFrom,&["a"])],
            allowed_actions:vec!["hold".into()],evidence_budget:2,observation_budget:0,observation_cost_units:0.0};
        assert!(validate_ancestry(&e).is_err());
    }
}
