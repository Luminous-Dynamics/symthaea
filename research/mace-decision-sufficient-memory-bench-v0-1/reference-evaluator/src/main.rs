// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MACE DSM v0.1 deterministic reference evaluator.
//! Candidate hashes bind the prepared candidate bytes; they are not signatures.
//! This binary does not supply OS process isolation or evaluator custody.

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    io::{BufRead, BufReader, BufWriter, Write},
};

const EPISODE_SCHEMA: &str = "mace_dsm_episode_v0_1";
const CANDIDATE_SCHEMA: &str = "mace_dsm_candidate_input_v0_1";
const PRIVATE_SCHEMA: &str = "mace_dsm_evaluator_private_v0_1";
const RESPONSE_SCHEMA: &str = "mace_dsm_response_v0_1";
const REPORT_SCHEMA: &str = "mace_dsm_score_report_v0_1";
const CANDIDATE_DOMAIN: &[u8] = b"luminous:mace:dsm:candidate-input:v0.1\0";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Episode {
    schema_version: String,
    episode_id: String,
    seed: u64,
    family: String,
    candidate: Candidate,
    evaluator_private: EvaluatorPrivate,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Candidate {
    query_time_ms: i64,
    query: String,
    source_events: Vec<SourceEvent>,
    allowed_actions: Vec<String>,
    evidence_budget: usize,
    observation_budget: usize,
    observation_cost_units: f64,
    scope: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
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
    #[serde(skip_serializing_if = "Option::is_none")]
    relation_to_parent: Option<Relation>,
    parent_event_ids: Vec<String>,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum AuthorityTier { Verified, Authenticated, UserAsserted, Untrusted }
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Relation { Asserts, Supersedes, Retracts, Contradicts, DerivedFrom }
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct EvaluatorPrivate {
    latent_state: String,
    action_losses: Vec<ActionLoss>,
    required_evidence_ids: Vec<String>,
    forbidden_evidence_ids: Vec<String>,
    no_memory_action: String,
    oracle_action: String,
    attack_case: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(skip_serializing_if = "Option::is_none")]
    notes: Option<String>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ActionLoss { action: String, loss: f64 }

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CandidateRow {
    schema_version: String,
    episode_id: String,
    seed: u64,
    family: String,
    candidate_sha256: String,
    candidate: Candidate,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PrivateRow {
    schema_version: String,
    episode_id: String,
    seed: u64,
    family: String,
    candidate_sha256: String,
    evaluator_private: EvaluatorPrivate,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Response {
    schema_version: String,
    episode_id: String,
    arm_id: String,
    candidate_sha256: String,
    action: String,
    selected_evidence_ids: Vec<String>,
    observation_requests: usize,
    observation_cost_units: f64,
    cost: Cost,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Cost {
    ingest_operations: u64,
    query_operations: u64,
    maintenance_operations: u64,
    retained_bytes: u64,
    returned_bytes: u64,
    latency_us: Option<u64>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArmManifest { schema_version: String, expected_arm_ids: Vec<String> }

#[derive(Clone, Debug, Serialize)]
struct ScoreRow {
    episode_id: String,
    family: String,
    seed: u64,
    arm_id: String,
    action: String,
    declared_no_memory_reference_action: String,
    oracle_action: String,
    loss: f64,
    oracle_loss: f64,
    regret: f64,
    declared_no_memory_reference_loss: f64,
    gain_vs_declared_no_memory_reference: f64,
    worse_than_declared_no_memory_reference: bool,
    changed_action_vs_declared_no_memory_reference: bool,
    selected_evidence_count: usize,
    admissible_evidence_count: usize,
    invalid_evidence_count: usize,
    stale_or_superseded_evidence_count: usize,
    untrusted_evidence_count: usize,
    forbidden_evidence_count: usize,
    unresolved_contradiction_pairs: usize,
    valid_evidence_precision: Option<f64>,
    required_evidence_recall: f64,
    decision_sufficient: bool,
    attack_case: bool,
    observation_requests: usize,
    cost: Cost,
}

fn hex(bytes: &[u8]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes { out.push(H[(b >> 4) as usize] as char); out.push(H[(b & 15) as usize] as char); }
    out
}
fn digest(bytes: &[u8]) -> String { hex(&Sha256::digest(bytes)) }
#[derive(Serialize)]
struct CandidateBinding<'a> {
    episode_id: &'a str,
    seed: u64,
    family: &'a str,
    candidate: &'a Candidate,
}
fn candidate_digest(episode_id: &str, seed: u64, family: &str, candidate: &Candidate) -> Result<String, String> {
    let binding = CandidateBinding { episode_id, seed, family, candidate };
    let bytes = serde_json::to_vec(&binding).map_err(|e| e.to_string())?;
    let mut h = Sha256::new();
    h.update(CANDIDATE_DOMAIN);
    h.update((bytes.len() as u64).to_be_bytes());
    h.update(bytes);
    Ok(hex(&h.finalize()))
}
fn read_jsonl<T: DeserializeOwned>(path: &str) -> Result<Vec<T>, String> {
    let file = fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let mut rows = Vec::new();
    for (i, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("read {path} line {}: {e}", i + 1))?;
        if line.trim().is_empty() { return Err(format!("{path}: blank line {}", i + 1)); }
        rows.push(serde_json::from_str(&line).map_err(|e| format!("{path} line {}: {e}", i + 1))?);
    }
    if rows.is_empty() { return Err(format!("{path}: no JSONL records")); }
    Ok(rows)
}
fn write_jsonl<T: Serialize>(path: &str, rows: &[T]) -> Result<(), String> {
    let file = fs::File::create(path).map_err(|e| format!("create {path}: {e}"))?;
    let mut out = BufWriter::new(file);
    for row in rows {
        serde_json::to_writer(&mut out, row).map_err(|e| e.to_string())?;
        out.write_all(b"\n").map_err(|e| e.to_string())?;
    }
    out.flush().map_err(|e| e.to_string())
}
fn validate_episode(e: &Episode) -> Result<(), String> {
    let id = &e.episode_id;
    if e.schema_version != EPISODE_SCHEMA || id.trim().is_empty() ||
        !matches!(e.family.as_str(), "temporal_drift" | "contradiction" | "revocation" | "insufficient_evidence" | "poisoning" | "decision_sufficient_compression" | "streaming_update" | "forgetting_replay" | "scope_isolation") ||
        e.candidate.query.trim().is_empty() ||
        e.candidate.scope.trim().is_empty() { return Err(format!("{id}: invalid episode identity/schema")); }
    if !e.candidate.observation_cost_units.is_finite() || e.candidate.observation_cost_units < 0.0 {
        return Err(format!("{id}: invalid observation cost"));
    }
    let mut event_ids = BTreeSet::new();
    for v in &e.candidate.source_events {
        if v.event_id.trim().is_empty() || !event_ids.insert(v.event_id.as_str()) ||
            v.source_id.trim().is_empty() || v.claim.trim().is_empty() ||
            v.uncertainty_ppm > 1_000_000 ||
            v.valid_until_ms.is_some_and(|end| end < v.valid_from_ms) {
            return Err(format!("{id}: malformed/duplicate source event {}", v.event_id));
        }
    }
    for v in &e.candidate.source_events {
        if v.parent_event_ids.iter().any(|p| !event_ids.contains(p.as_str()) || p == &v.event_id) {
            return Err(format!("{id}: unresolved/self parent on {}", v.event_id));
        }
    }
    let allowed: BTreeSet<String> = e.candidate.allowed_actions.iter().cloned().collect();
    if allowed.is_empty() || allowed.len() != e.candidate.allowed_actions.len() {
        return Err(format!("{id}: empty or duplicate action set"));
    }
    let mut losses = BTreeMap::new();
    for x in &e.evaluator_private.action_losses {
        if !x.loss.is_finite() || !allowed.contains(&x.action) ||
            losses.insert(x.action.as_str(), x.loss).is_some() {
            return Err(format!("{id}: malformed/duplicate action loss"));
        }
    }
    if losses.len() != allowed.len() || !allowed.contains(&e.evaluator_private.no_memory_action) ||
        !allowed.contains(&e.evaluator_private.oracle_action) {
        return Err(format!("{id}: incomplete action loss or baseline set"));
    }
    let best = losses.values().copied().fold(f64::INFINITY, f64::min);
    if (losses[e.evaluator_private.oracle_action.as_str()] - best).abs() > 1e-9 {
        return Err(format!("{id}: oracle action is not minimum-loss"));
    }
    let required: BTreeSet<String> = e.evaluator_private.required_evidence_ids.iter().cloned().collect();
    let forbidden: BTreeSet<String> = e.evaluator_private.forbidden_evidence_ids.iter().cloned().collect();
    if required.len() != e.evaluator_private.required_evidence_ids.len() ||
        forbidden.len() != e.evaluator_private.forbidden_evidence_ids.len() ||
        !required.is_disjoint(&forbidden) ||
        required.iter().chain(forbidden.iter()).any(|x| !event_ids.contains(x.as_str())) {
        return Err(format!("{id}: malformed required/forbidden evidence set"));
    }
    Ok(())
}
fn prepare(source: &str, candidate_path: &str, private_path: &str) -> Result<(), String> {
    let episodes: Vec<Episode> = read_jsonl(source)?;
    let mut seen = BTreeSet::new();
    let mut candidates = Vec::new();
    let mut privates = Vec::new();
    for e in episodes {
        validate_episode(&e)?;
        if !seen.insert(e.episode_id.clone()) { return Err(format!("duplicate episode {}", e.episode_id)); }
        let binding = candidate_digest(&e.episode_id, e.seed, &e.family, &e.candidate)?;
        candidates.push(CandidateRow {
            schema_version: CANDIDATE_SCHEMA.into(), episode_id: e.episode_id.clone(),
            seed: e.seed, family: e.family.clone(), candidate_sha256: binding.clone(), candidate: e.candidate,
        });
        privates.push(PrivateRow {
            schema_version: PRIVATE_SCHEMA.into(), episode_id: e.episode_id,
            seed: e.seed, family: e.family, candidate_sha256: binding, evaluator_private: e.evaluator_private,
        });
    }
    write_jsonl(candidate_path, &candidates)?;
    write_jsonl(private_path, &privates)
}
fn current_at(e: &SourceEvent, at: i64) -> bool {
    e.observed_at_ms <= at && e.valid_from_ms <= at &&
        e.valid_until_ms.is_none_or(|end| at < end)
}
fn invalidated(id: &str, events: &[SourceEvent], at: i64) -> bool {
    events.iter().any(|event|
        event.observed_at_ms <= at && event.valid_from_ms <= at &&
        event.authority_tier != AuthorityTier::Untrusted &&
        matches!(event.relation_to_parent, Some(Relation::Supersedes | Relation::Retracts)) &&
        event.parent_event_ids.iter().any(|parent| parent == id))
}
fn admissible(e: &SourceEvent, c: &Candidate, forbidden: &BTreeSet<String>) -> bool {
    current_at(e, c.query_time_ms) && !invalidated(&e.event_id, &c.source_events, c.query_time_ms) &&
        e.authority_tier != AuthorityTier::Untrusted && !forbidden.contains(&e.event_id)
}
fn score_one(c: &CandidateRow, p: &PrivateRow, r: &Response) -> Result<ScoreRow, String> {
    let id = &c.episode_id;
    if r.schema_version != RESPONSE_SCHEMA || c.schema_version != CANDIDATE_SCHEMA ||
        p.schema_version != PRIVATE_SCHEMA || r.episode_id != *id ||
        p.episode_id != *id || p.seed != c.seed || p.family != c.family {
        return Err(format!("{id}: schema or candidate/private/response identity mismatch"));
    }
    let computed_candidate_sha256 = candidate_digest(&c.episode_id, c.seed, &c.family, &c.candidate)?;
    if computed_candidate_sha256 != c.candidate_sha256 ||
        computed_candidate_sha256 != p.candidate_sha256 ||
        computed_candidate_sha256 != r.candidate_sha256 {
        return Err(format!("{id}: candidate/response/evaluator digest binding mismatch"));
    }
    if r.arm_id.trim().is_empty() || !c.candidate.allowed_actions.contains(&r.action) {
        return Err(format!("{id}: empty arm or disallowed action"));
    }
    if r.selected_evidence_ids.len() > c.candidate.evidence_budget ||
        r.observation_requests > c.candidate.observation_budget {
        return Err(format!("{id}: evidence/observation budget exceeded"));
    }
    let event_map: BTreeMap<&str, &SourceEvent> = c.candidate.source_events.iter()
        .map(|e| (e.event_id.as_str(), e)).collect();
    let forbidden: BTreeSet<String> = p.evaluator_private.forbidden_evidence_ids.iter().cloned().collect();
    let required: BTreeSet<String> = p.evaluator_private.required_evidence_ids.iter().cloned().collect();
    let mut selected: BTreeSet<String> = BTreeSet::new();
    let (mut admissible_n, mut stale_n, mut untrusted_n, mut forbidden_n) = (0usize, 0usize, 0usize, 0usize);
    for event_id in &r.selected_evidence_ids {
        if !selected.insert(event_id.clone()) { return Err(format!("{id}: duplicate selected evidence {event_id}")); }
        let event = event_map.get(event_id.as_str()).ok_or_else(|| format!("{id}: unknown selected evidence {event_id}"))?;
        if !current_at(event, c.candidate.query_time_ms) || invalidated(event_id, &c.candidate.source_events, c.candidate.query_time_ms) { stale_n += 1; }
        if event.authority_tier == AuthorityTier::Untrusted { untrusted_n += 1; }
        if forbidden.contains(event_id) { forbidden_n += 1; }
        if admissible(event, &c.candidate, &forbidden) { admissible_n += 1; }
    }
    let contradiction_pairs = c.candidate.source_events.iter()
        .filter(|e| e.relation_to_parent == Some(Relation::Contradicts) && selected.contains(&e.event_id))
        .map(|e| e.parent_event_ids.iter().filter(|parent| selected.contains(*parent)).count())
        .sum::<usize>();
    let loss_map: BTreeMap<&str, f64> = p.evaluator_private.action_losses.iter().map(|x| (x.action.as_str(), x.loss)).collect();
    let loss = *loss_map.get(r.action.as_str()).ok_or_else(|| format!("{id}: action has no loss"))?;
    let oracle_loss = loss_map.values().copied().fold(f64::INFINITY, f64::min);
    let declared_no_memory_reference_loss = *loss_map.get(p.evaluator_private.no_memory_action.as_str())
        .ok_or_else(|| format!("{id}: no-memory action has no loss"))?;
    let required_found = required.iter().filter(|event_id|
        selected.contains(*event_id) &&
        event_map.get(event_id.as_str()).is_some_and(|e| admissible(e, &c.candidate, &forbidden))
    ).count();
    let recall = if required.is_empty() { 1.0 } else { required_found as f64 / required.len() as f64 };
    let invalid_n = r.selected_evidence_ids.len().saturating_sub(admissible_n);
    Ok(ScoreRow {
        episode_id: id.clone(), family: c.family.clone(), seed: c.seed,
        arm_id: r.arm_id.clone(), action: r.action.clone(),
        declared_no_memory_reference_action: p.evaluator_private.no_memory_action.clone(), 
        oracle_action: p.evaluator_private.oracle_action.clone(),
        loss, oracle_loss, regret: (loss - oracle_loss).max(0.0),
        declared_no_memory_reference_loss,
        gain_vs_declared_no_memory_reference: declared_no_memory_reference_loss - loss,
        worse_than_declared_no_memory_reference: loss > declared_no_memory_reference_loss + 1e-9,
        changed_action_vs_declared_no_memory_reference: r.action != p.evaluator_private.no_memory_action,
        selected_evidence_count: r.selected_evidence_ids.len(),
        admissible_evidence_count: admissible_n, invalid_evidence_count: invalid_n,
        stale_or_superseded_evidence_count: stale_n, untrusted_evidence_count: untrusted_n,
        forbidden_evidence_count: forbidden_n, unresolved_contradiction_pairs: contradiction_pairs,
        valid_evidence_precision: if r.selected_evidence_ids.is_empty() { None } else {
            Some(admissible_n as f64 / r.selected_evidence_ids.len() as f64)
        },
        required_evidence_recall: recall,
        decision_sufficient: (loss - oracle_loss).abs() <= 1e-9 && required_found == required.len() && invalid_n == 0,
        attack_case: p.evaluator_private.attack_case,
        observation_requests: r.observation_requests,
        observation_cost_units: c.candidate.observation_cost_units,
        cost: r.cost.clone(),
    })
}
fn average(values: impl Iterator<Item = f64>) -> Option<f64> {
    let values: Vec<f64> = values.collect();
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}
fn aggregate(key: &str, rows: &[ScoreRow]) -> serde_json::Value {
    let n = rows.len() as f64;
    let latencies: Vec<f64> = rows.iter().filter_map(|r| r.cost.latency_us.map(|x| x as f64)).collect();
    json!({
        "key": key, "episodes": rows.len(),
        "mean_loss": average(rows.iter().map(|r| r.loss)),
        "mean_regret": average(rows.iter().map(|r| r.regret)),
        "mean_gain_vs_declared_no_memory_reference": average(rows.iter().map(|r| r.gain_vs_declared_no_memory_reference)),
        "declared_no_memory_reference_comparison": "task-authored reference action; not an independently executed no-memory arm",
        "worse_than_declared_no_memory_reference_rate": rows.iter().filter(|r| r.worse_than_declared_no_memory_reference).count() as f64 / n,
        "decision_sufficiency_rate": rows.iter().filter(|r| r.decision_sufficient).count() as f64 / n,
        "decision_change_vs_declared_reference_rate": rows.iter().filter(|r| r.changed_action_vs_declared_no_memory_reference).count() as f64 / n,
        "mean_valid_evidence_precision": average(rows.iter().filter_map(|r| r.valid_evidence_precision)),
        "mean_required_evidence_recall": average(rows.iter().map(|r| r.required_evidence_recall)),
        "invalid_evidence_uses": rows.iter().map(|r| r.invalid_evidence_count).sum::<usize>(),
        "stale_or_superseded_uses": rows.iter().map(|r| r.stale_or_superseded_evidence_count).sum::<usize>(),
        "untrusted_evidence_uses": rows.iter().map(|r| r.untrusted_evidence_count).sum::<usize>(),
        "forbidden_evidence_uses": rows.iter().map(|r| r.forbidden_evidence_count).sum::<usize>(),
        "unresolved_contradiction_pairs": rows.iter().map(|r| r.unresolved_contradiction_pairs).sum::<usize>(),
        "observation_requests": rows.iter().map(|r| r.observation_requests).sum::<usize>(),
        "declared_observation_cost_units": rows.iter().map(|r| r.observation_cost_units).sum::<f64>(),
        "mean_declared_observation_cost_units": average(rows.iter().map(|r| r.observation_cost_units)),
        "ingest_operations": rows.iter().map(|r| r.cost.ingest_operations).sum::<u64>(),
        "query_operations": rows.iter().map(|r| r.cost.query_operations).sum::<u64>(),
        "maintenance_operations": rows.iter().map(|r| r.cost.maintenance_operations).sum::<u64>(),
        "retained_bytes": rows.iter().map(|r| r.cost.retained_bytes).sum::<u64>(),
        "returned_bytes": rows.iter().map(|r| r.cost.returned_bytes).sum::<u64>(),
        "latency_samples": latencies.len(),
        "mean_reported_latency_us": average(latencies.into_iter())
    })
}
fn score(candidate_path: &str, private_path: &str, response_path: &str, arm_path: &str, report_path: &str) -> Result<(), String> {
    let candidates: Vec<CandidateRow> = read_jsonl(candidate_path)?;
    let privates: Vec<PrivateRow> = read_jsonl(private_path)?;
    let responses: Vec<Response> = read_jsonl(response_path)?;
    let arm_bytes = fs::read(arm_path).map_err(|e| format!("read arm manifest: {e}"))?;
    let arm_manifest: ArmManifest = serde_json::from_slice(&arm_bytes).map_err(|e| format!("parse arm manifest: {e}"))?;
    if arm_manifest.schema_version != "mace_dsm_arm_manifest_v0_1" || arm_manifest.expected_arm_ids.is_empty() ||
        arm_manifest.expected_arm_ids.iter().any(|arm| arm.trim().is_empty()) {
        return Err("invalid or empty arm manifest".into());
    }
    let expected_arms: BTreeSet<String> = arm_manifest.expected_arm_ids.iter().cloned().collect();
    if expected_arms.len() != arm_manifest.expected_arm_ids.len() { return Err("duplicate arm in manifest".into()); }
    let mut cm = BTreeMap::new();
    let mut pm = BTreeMap::new();
    for c in &candidates {
        if cm.insert(c.episode_id.as_str(), c).is_some() { return Err(format!("duplicate candidate {}", c.episode_id)); }
    }
    for p in &privates {
        if pm.insert(p.episode_id.as_str(), p).is_some() { return Err(format!("duplicate private {}", p.episode_id)); }
    }
    if cm.len() != pm.len() || cm.keys().any(|id| !pm.contains_key(id)) {
        return Err("candidate/private episode inventories differ".into());
    }
    let mut pairs = BTreeMap::new();
    for r in &responses {
        if !cm.contains_key(r.episode_id.as_str()) { return Err(format!("unknown episode {}", r.episode_id)); }
        if !expected_arms.contains(&r.arm_id) { return Err(format!("response contains undeclared arm {}", r.arm_id)); }
        if pairs.insert((r.episode_id.as_str(), r.arm_id.as_str()), r).is_some() {
            return Err(format!("duplicate response {}/{}", r.episode_id, r.arm_id));
        }
    }
    let needed = cm.len().checked_mul(expected_arms.len()).ok_or("response matrix overflow")?;
    if pairs.len() != needed { return Err(format!("unbalanced response matrix: need {needed}, got {}", pairs.len())); }
    for id in cm.keys() { for arm in &expected_arms {
        if !pairs.contains_key(&(*id, arm.as_str())) { return Err(format!("missing response for {id}/{arm}")); }
    }}
    let mut scored = Vec::with_capacity(responses.len());
    for ((id, arm), r) in pairs {
        scored.push(score_one(cm[id], pm[id], r)?);
        if scored.last().is_some_and(|row| row.arm_id.as_str() != arm) { return Err("response key mismatch".into()); }
    }
    scored.sort_by(|a,b| (&a.episode_id,&a.arm_id).cmp(&(&b.episode_id,&b.arm_id)));
    let mut arm_groups: BTreeMap<String, Vec<ScoreRow>> = BTreeMap::new();
    let mut family_groups: BTreeMap<(String,String), Vec<ScoreRow>> = BTreeMap::new();
    for row in &scored {
        arm_groups.entry(row.arm_id.clone()).or_default().push(row.clone());
        family_groups.entry((row.family.clone(),row.arm_id.clone())).or_default().push(row.clone());
    }
    let per_arm: Vec<_> = arm_groups.iter().map(|(k,v)| aggregate(k,v)).collect();
    let per_family_arm: Vec<_> = family_groups.iter().map(|((f,a),v)| aggregate(&format!("{f}/{a}"),v)).collect();
    let report = json!({
        "schema_version": REPORT_SCHEMA,
        "qualification_status": "reference_evaluation_generated_not_independently_qualified",
        "candidate_input_sha256": digest(&fs::read(candidate_path).map_err(|e|e.to_string())?),
        "evaluator_private_input_sha256": digest(&fs::read(private_path).map_err(|e|e.to_string())?),
        "response_input_sha256": digest(&fs::read(response_path).map_err(|e|e.to_string())?),
        "episode_count": candidates.len(), "arm_ids": expected_arms,
        "arm_manifest_sha256": digest(&arm_bytes),
        "paired_response_count": scored.len(),
        "per_arm": per_arm, "per_family_arm": per_family_arm, "per_episode": scored,
        "interpretation": {
            "regret": "Candidate action loss minus minimum evaluator-private action loss.",
            "declared_no_memory_reference_gain": "Task-authored reference action loss minus this arm loss; this is not a separately executed no-memory agent arm.",
            "worse_than_declared_reference": "This arm loss exceeds the task-authored reference action loss; this is not a paired no-memory system comparison.",
            "admissible_evidence": "Current at query time and observed no later than the query, not superseded/retracted, not untrusted, and not evaluator-labelled forbidden.",
            "decision_sufficient": "Minimum loss, all required evidence selected, and no invalid evidence selected.",
            "cost": "Counters are candidate-reported. Latency is summarized only when supplied; this evaluator does not independently measure it."
        },
        "limitations": [
            "This binary does not isolate candidate and evaluator processes. Use separate OS identities/containers and mount only candidate_input.jsonl into the candidate.",
            "The candidate digest is a binding checksum, not a signature or proof of trusted custody.",
            "Authority tiers and required/forbidden evidence labels are synthetic benchmark conventions.",
            "uncertainty_ppm is retained but is not calibrated or interpreted as a probability.",
            "The evaluator scores response receipts; it does not execute memory operations, observations, or downstream actions."
        ]
    });
    fs::write(report_path, serde_json::to_vec_pretty(&report).map_err(|e|e.to_string())?)
        .map_err(|e| format!("write report: {e}"))
}

fn event(id: String, source: String, observed: i64, valid_from: i64, valid_until: Option<i64>,
    authority: AuthorityTier, uncertainty: u32, claim: &str, relation: Relation, parents: Vec<String>) -> SourceEvent {
    SourceEvent { event_id:id, source_id:source, observed_at_ms:observed, valid_from_ms:valid_from,
        valid_until_ms:valid_until, authority_tier:authority, uncertainty_ppm:uncertainty,
        claim:claim.into(), relation_to_parent:Some(relation), parent_event_ids:parents }
}
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z=x; z=(z^(z>>30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z=(z^(z>>27)).wrapping_mul(0x94D0_49BB_1331_11EB); z^(z>>31)
}
fn generated_episode(i: usize, seed: u64) -> Episode {
    let s=seed.wrapping_add(i as u64); let jitter=(mix(s)%97) as i64;
    let n=format!("{i:06}"); let old=format!("event-{n}-old"); let new=format!("event-{n}-new");
    let (family,query,events,actions,losses,required,forbidden,no_memory,oracle)=match i%4 {
        0 => ("temporal_drift","Which account is currently valid?",vec![
            event(old.clone(),format!("registry-{n}"),100,0,Some(1000),AuthorityTier::Verified,10000,"Former account ACCT-OLD.",Relation::Asserts,vec![]),
            event(new.clone(),format!("registry-{n}"),1000+jitter,1000,None,AuthorityTier::Verified,5000,"Current account ACCT-NEW.",Relation::Supersedes,vec![old.clone()])],
            vec!["use_old_account","use_current_account","abstain"],vec![("use_old_account",12.0),("use_current_account",0.0),("abstain",2.0)],
            vec![new.clone()],vec![old.clone()],"abstain","use_current_account"),
        1 => ("contradiction","Is the service cleared for deployment?",vec![
            event(old.clone(),format!("chat-{n}"),1000,1000,None,AuthorityTier::UserAsserted,150000,"A message says the service is healthy.",Relation::Asserts,vec![]),
            event(new.clone(),format!("signed-gate-{n}"),1500+jitter,1500,None,AuthorityTier::Verified,1000,"Signed release gate reports a blocking failure.",Relation::Contradicts,vec![old.clone()])],
            vec!["declare_ready","declare_blocked","request_independent_check"],vec![("declare_ready",20.0),("declare_blocked",0.0),("request_independent_check",1.0)],
            vec![new.clone()],vec![old.clone()],"request_independent_check","declare_blocked"),
        2 => ("revocation","Should the prior procedure still be applied?",vec![
            event(old.clone(),format!("office-{n}"),100,0,None,AuthorityTier::Verified,0,"Procedure OP-17 is approved.",Relation::Asserts,vec![]),
            event(new.clone(),format!("office-{n}"),1500+jitter,1500,None,AuthorityTier::Verified,0,"Approval for OP-17 is revoked.",Relation::Retracts,vec![old.clone()])],
            vec!["apply_old_procedure","honor_revocation","request_review"],vec![("apply_old_procedure",15.0),("honor_revocation",0.0),("request_review",1.0)],
            vec![new.clone()],vec![old.clone()],"request_review","honor_revocation"),
        _ => ("insufficient_evidence","What is the current remote-site temperature?",vec![
            event(old.clone(),format!("sensor-{n}"),1000,1000,Some(1200),AuthorityTier::Authenticated,10000,"Earlier sample was 21 degrees.",Relation::Asserts,vec![])],
            vec!["answer_from_old_sample","request_observation","abstain"],vec![("answer_from_old_sample",8.0),("request_observation",0.25),("abstain",1.0)],
            vec![],vec![old.clone()],"request_observation","request_observation")
    };
    Episode { schema_version:EPISODE_SCHEMA.into(), episode_id:format!("{family}/{n}"), seed:s, family:family.into(),
        candidate:Candidate { query_time_ms:2000+jitter, query:query.into(), source_events:events,
            allowed_actions:actions.into_iter().map(str::to_string).collect(), evidence_budget:2,
            observation_budget:if family=="insufficient_evidence" {1} else {0},
            observation_cost_units:if family=="insufficient_evidence" {0.25} else {0.0},
            scope:format!("generated:scope-{:02}",i%16) },
        evaluator_private:EvaluatorPrivate { latent_state:format!("Synthetic finite-world truth for {family}/{n}"),
            action_losses:losses.into_iter().map(|(action,loss)|ActionLoss{action:action.into(),loss}).collect(),
            required_evidence_ids:required, forbidden_evidence_ids:forbidden,
            no_memory_action:no_memory.into(),oracle_action:oracle.into(),attack_case:false,
            notes:Some("Generated deterministic control; not an externally validated scenario.".into()) } }
}
fn generate(seed:u64,count:usize,path:&str)->Result<(),String>{
    if count==0 || count>1_000_000 {return Err("count must be 1..=1,000,000".into())}
    let rows:Vec<Episode>=(0..count).map(|i|generated_episode(i,seed)).collect();
    for row in &rows {validate_episode(row)?;}
    write_jsonl(path,&rows)
}
fn run()->Result<(),String>{
    let a:Vec<String>=env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("generate") if a.len()==5 => generate(a[2].parse().map_err(|e|format!("seed: {e}"))?,
            a[3].parse().map_err(|e|format!("count: {e}"))?,&a[4]),
        Some("prepare") if a.len()==5 => prepare(&a[2],&a[3],&a[4]),
        Some("score") if a.len()==7 => score(&a[2],&a[3],&a[4],&a[5],&a[6]),
        _ => Err("usage: generate <seed> <count> <episodes.jsonl> | prepare <episodes.jsonl> <candidate.jsonl> <private.jsonl> | score <candidate.jsonl> <private.jsonl> <responses.jsonl> <arms.json> <report.json>".into())
    }
}
fn main(){if let Err(e)=run(){eprintln!("error: {e}");std::process::exit(2);}}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn valid_time_is_half_open_and_observation_time_prevents_future_leakage(){
        let e=event("e".into(),"s".into(),15,10,Some(20),AuthorityTier::Verified,0,"c",Relation::Asserts,vec![]);
        assert!(!current_at(&e,14)); assert!(current_at(&e,15)); assert!(current_at(&e,19)); assert!(!current_at(&e,20));
    }
    #[test] fn supersession_and_retraction_apply_only_after_they_are_observed(){
        let contra=event("c".into(),"s2".into(),2,0,None,AuthorityTier::Verified,0,"c",Relation::Contradicts,vec!["p".into()]);
        assert!(!invalidated("p",&[contra],10));
        let sup=event("s".into(),"s1".into(),2,0,None,AuthorityTier::Verified,0,"s",Relation::Supersedes,vec!["p".into()]);
        assert!(!invalidated("p",&[sup.clone()],1));
        assert!(invalidated("p",&[sup],2));
        let untrusted_revoke=event("u".into(),"attacker".into(),2,0,None,AuthorityTier::Untrusted,0,"fake revoke",Relation::Retracts,vec!["p".into()]);
        assert!(!invalidated("p",&[untrusted_revoke],2));
    }
    #[test] fn untrusted_evidence_is_never_admissible(){
        let e=event("e".into(),"s".into(),1,0,None,AuthorityTier::Untrusted,0,"x",Relation::Asserts,vec![]);
        let c=Candidate{query_time_ms:2,query:"q".into(),source_events:vec![e.clone()],allowed_actions:vec!["a".into()],evidence_budget:1,observation_budget:0,observation_cost_units:0.0,scope:"s".into()};
        assert!(!admissible(&e,&c,&BTreeSet::new()));
    }
    #[test] fn generator_is_deterministic_and_covers_four_confirmatory_families(){
        let a:Vec<_>=(0..40).map(|i|generated_episode(i,77)).collect();
        let b:Vec<_>=(0..40).map(|i|generated_episode(i,77)).collect();
        for (x,y) in a.iter().zip(b.iter()){validate_episode(x).unwrap();assert_eq!(serde_json::to_vec(x).unwrap(),serde_json::to_vec(y).unwrap());}
        let families:BTreeSet<_>=a.iter().map(|x|x.family.as_str()).collect();
        assert_eq!(families,["temporal_drift","contradiction","revocation","insufficient_evidence"].into_iter().collect());
    }
    #[test] fn bad_memory_is_measured_against_no_memory_separately_from_regret(){
        let e=generated_episode(0,77);
        let c=CandidateRow{schema_version:CANDIDATE_SCHEMA.into(),episode_id:e.episode_id.clone(),seed:e.seed,family:e.family.clone(),candidate_sha256:candidate_digest(&e.episode_id, e.seed, &e.family, &e.candidate).unwrap(),candidate:e.candidate.clone()};
        let p=PrivateRow{schema_version:PRIVATE_SCHEMA.into(),episode_id:e.episode_id.clone(),seed:e.seed,family:e.family.clone(),candidate_sha256:candidate_digest(&e.episode_id, e.seed, &e.family, &e.candidate).unwrap(),evaluator_private:e.evaluator_private.clone()};
        let r=Response{schema_version:RESPONSE_SCHEMA.into(),episode_id:e.episode_id,arm_id:"bad-memory".into(),candidate_sha256:c.candidate_sha256.clone(),action:"use_old_account".into(),selected_evidence_ids:vec!["event-000000-old".into()],observation_requests:0,cost:Cost{ingest_operations:1,query_operations:1,maintenance_operations:0,retained_bytes:10,returned_bytes:10,latency_us:None}};
        let s=score_one(&c,&p,&r).unwrap();
        let mut mismatched = r.clone(); mismatched.candidate_sha256 = "0".repeat(64);
        assert!(score_one(&c,&p,&mismatched).is_err());
        assert_eq!(s.regret,12.0);assert!(s.worse_than_declared_no_memory_reference);assert_eq!(s.gain_vs_declared_no_memory_reference,-10.0);assert_eq!(s.stale_or_superseded_evidence_count,1);
    }
}
