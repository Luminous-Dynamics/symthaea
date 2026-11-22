// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic paired memory-to-decision interventions over the existing
//! generated MACE worlds. The evaluator and candidate runner are co-located
//! and public: every output is diagnostic-only, never an external score.

#[path = "mace_generated_stress.rs"]
#[allow(dead_code)]
mod stress;

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::mem::size_of;
use std::time::Instant;

use broca_mace_hdc_retrieval::{
    experience_view_digest, CandidateIndex, CandidateHit, IndexConfig, RepresentationProfile,
};
use broca_mace_policy::{
    decide, is_qualified, Action, Decision, Experience, Policy, Query, Regime, Objective,
};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const SCHEMA_VERSION: &str = "mace_memory_decision_intervention_v0_1";
const HDC_DIMENSION: usize = 2_048;
const HDC_CODEBOOK_SEED: u64 = 0x4D41_4345_4844_4301;
const HDC_NEURON_SEED: u64 = 0x4D41_4345_4C54_4301;
const TOP_K: usize = 2;
const BOOTSTRAP_REPLICATES: usize = 2_000;
const SCALE: f64 = 1_000_000.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum Arm {
    NoMemory,
    ExactScopeScan,
    ConventionalRetrieval,
    StaticHdc,
    HdcLtc,
    SemanticPlacebo,
}

impl Arm {
    const ALL: [Self; 6] = [
        Self::NoMemory,
        Self::ExactScopeScan,
        Self::ConventionalRetrieval,
        Self::StaticHdc,
        Self::HdcLtc,
        Self::SemanticPlacebo,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::NoMemory => "no_memory",
            Self::ExactScopeScan => "exact_scope_scan",
            Self::ConventionalRetrieval => "conventional_retrieval",
            Self::StaticHdc => "static_hdc",
            Self::HdcLtc => "hdc_ltc",
            Self::SemanticPlacebo => "semantic_placebo",
        }
    }
}

#[derive(Clone, Debug)]
struct CandidateRun {
    decision: Decision,
    retrieved_ids: Vec<u32>,
    resolved: Vec<Experience>,
    latency_ns: u64,
    index_bytes: u64,
    sidecar_bytes: u64,
    memory_digest: String,
    abstention_reason: Option<String>,
}

#[derive(Clone, Debug)]
struct ScoredArm {
    arm: Arm,
    cluster: u64,
    family: String,
    pair_block_id: String,
    decision: Option<Action>,
    utility_micros: i64,
    regret_micros: i64,
    task_success: bool,
    latency_ns: u64,
    memory_bytes: i64,
    invalid_evidence_use_count: u64,
    stale_evidence_use_count: u64,
    wrong_scope_evidence_use_count: u64,
    false_causal_claim_count: u64,
    support_count: usize,
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn json_sha256<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    Ok(sha256(&bytes))
}

fn utility_micros(value: f64) -> i64 {
    (value * SCALE).round() as i64
}

fn source_memory_for_arm(
    arm: Arm,
    case: &stress::Case,
    all_cases: &[stress::Case],
) -> Vec<Experience> {
    match arm {
        Arm::NoMemory => Vec::new(),
        Arm::SemanticPlacebo => {
            // A real, candidate-visible sibling-task episode from the same
            // generated world cluster and same scope. This is a structural
            // placebo, not a text/embedding-level semantic placebo.
            let sibling = all_cases
                .iter()
                .filter(|candidate| {
                    candidate.cluster == case.cluster
                        && candidate.family != case.family
                        && candidate.query.scope.regime == case.query.scope.regime
                        && candidate.query.scope.objective == case.query.scope.objective
                        && !candidate.memory.is_empty()
                })
                .min_by_key(|candidate| {
                    candidate.memory.len().abs_diff(case.memory.len())
                });
            let mut memory = sibling
                .map(|candidate| candidate.memory.clone())
                .unwrap_or_default();
            memory.sort_by_key(|item| item.event_id);
            memory.truncate(case.memory.len());
            memory
        }
        _ => case.memory.clone(),
    }
}

fn resolve_ids(
    ids: &[u32],
    memory: &[Experience],
) -> Result<Vec<Experience>, String> {
    let mut canonical = BTreeMap::new();
    for item in memory {
        if canonical.insert(item.event_id, *item).is_some() {
            return Err(format!("duplicate canonical event ID {}", item.event_id));
        }
    }

    let mut seen = BTreeSet::new();
    let mut resolved = Vec::with_capacity(ids.len());
    for id in ids {
        if !seen.insert(*id) {
            return Err(format!("duplicate candidate event ID {id}"));
        }
        let item = canonical
            .get(id)
            .ok_or_else(|| format!("unresolved candidate event ID {id}"))?;
        resolved.push(*item);
    }
    Ok(resolved)
}

fn conventional_ids(memory: &[Experience], query: Query, top_k: usize) -> Vec<u32> {
    let mut ids = Vec::new();
    for action in [Action::A, Action::B] {
        let mut candidates: Vec<&Experience> = memory
            .iter()
            .filter(|item| item.scope == query.scope && item.action == action)
            .collect();
        candidates.sort_by_key(|item| item.event_id);
        ids.extend(
            candidates
                .into_iter()
                .take(top_k)
                .map(|item| item.event_id),
        );
    }
    ids
}

fn exact_scope_ids(memory: &[Experience], query: Query) -> Vec<u32> {
    let mut ids = Vec::new();
    for action in [Action::A, Action::B] {
        let mut candidates: Vec<&Experience> = memory
            .iter()
            .filter(|item| item.scope == query.scope && item.action == action)
            .collect();
        candidates.sort_by_key(|item| item.event_id);
        ids.extend(candidates.into_iter().map(|item| item.event_id));
    }
    ids
}

fn hdc_hits(
    memory: &[Experience],
    query: Query,
    profile: RepresentationProfile,
) -> Result<(Vec<CandidateHit>, u64), String> {
    let index = CandidateIndex::build(
        memory,
        IndexConfig {
            dimension: HDC_DIMENSION,
            codebook_seed: HDC_CODEBOOK_SEED,
            profile,
        },
    )
    .map_err(|error| error.to_string())?;

    let mut hits: Vec<CandidateHit> = Vec::new();
    for action in [Action::A, Action::B] {
        let result = index.retrieve(query.scope, action, TOP_K);
        hits.extend(result.candidates);
    }
    hits.sort_by(|left, right| {
        left.rank
            .cmp(&right.rank)
            .then_with(|| left.event_id.cmp(&right.event_id))
    });
    Ok((hits, index.index_payload_bytes()))
}

fn execute_arm(
    arm: Arm,
    case: &stress::Case,
    all_cases: &[stress::Case],
) -> CandidateRun {
    let started = Instant::now();
    let memory = source_memory_for_arm(arm, case, all_cases);
    let mut index_bytes = 0_u64;
    let mut retrieved_ids = Vec::new();
    let mut expected_view_digests = BTreeMap::new();
    let mut error = None;

    let ids_result: Result<Vec<u32>, String> = match arm {
        Arm::NoMemory => Ok(Vec::new()),
        Arm::ExactScopeScan => Ok(exact_scope_ids(&memory, case.query)),
        Arm::ConventionalRetrieval => {
            index_bytes = u64::try_from(memory.len())
                .unwrap_or(u64::MAX)
                .saturating_mul(size_of::<u32>() as u64);
            Ok(conventional_ids(&memory, case.query, TOP_K))
        }
        Arm::StaticHdc => hdc_hits(
            &memory,
            case.query,
            RepresentationProfile::StaticContinuousHv,
        )
        .map(|(hits, bytes)| {
            index_bytes = bytes;
            let mut ids = Vec::with_capacity(hits.len());
            for hit in hits {
                expected_view_digests.insert(hit.event_id, hit.view_digest);
                ids.push(hit.event_id);
            }
            ids
        }),
        Arm::HdcLtc => hdc_hits(
            &memory,
            case.query,
            RepresentationProfile::HdcLtcZeroInput {
                elapsed_seconds: 0.25,
                tau_base_seconds: 10.0,
                tau_min_seconds: 0.01,
                tau_max_seconds: 20.0,
                neuron_seed: HDC_NEURON_SEED,
            },
        )
        .map(|(hits, bytes)| {
            index_bytes = bytes;
            let mut ids = Vec::with_capacity(hits.len());
            for hit in hits {
                expected_view_digests.insert(hit.event_id, hit.view_digest);
                ids.push(hit.event_id);
            }
            ids
        }),
        Arm::SemanticPlacebo => {
            index_bytes = u64::try_from(memory.len())
                .unwrap_or(u64::MAX)
                .saturating_mul(size_of::<u32>() as u64);
            Ok(conventional_ids(&memory, case.query, TOP_K))
        }
    };

    match ids_result {
        Ok(ids) => retrieved_ids = ids,
        Err(message) => error = Some(message),
    }

    let resolved_result = if error.is_none() {
        resolve_ids(&retrieved_ids, &memory)
    } else {
        Err(error.clone().unwrap_or_else(|| "index failure".to_string()))
    };

    let resolved = match resolved_result {
        Ok(items) => items,
        Err(message) => {
            error = Some(message);
            Vec::new()
        }
    };

    // HDC candidate IDs are only hints. The canonical Experience view is
    // independently compared with the index-time view digest before policy use.
    if error.is_none() && matches!(arm, Arm::StaticHdc | Arm::HdcLtc) {
        let mut verified = Vec::with_capacity(retrieved_ids.len());
        for (id, item) in retrieved_ids.iter().zip(resolved.iter()) {
            let expected = experience_view_digest(item);
            let index_hit_is_current = expected_view_digests
                .get(id)
                .is_some_and(|indexed_digest| *indexed_digest == expected)
                && *id == item.event_id;
            if !index_hit_is_current {
                error = Some(format!("canonical candidate revalidation failed for {id}"));
                break;
            }
            verified.push(*item);
        }
        if error.is_none() && verified.len() != resolved.len() {
            error = Some("candidate/canonical resolution cardinality mismatch".to_string());
        }
    }

    let decision = if error.is_some() {
        Decision::abstain()
    } else if arm == Arm::NoMemory {
        decide(Policy::NoMemory, &[], case.query)
    } else {
        decide(Policy::CausalQualified, &resolved, case.query)
    };

    let sidecar_bytes = if arm == Arm::NoMemory {
        0
    } else {
        u64::try_from(memory.len())
            .unwrap_or(u64::MAX)
            .saturating_mul(size_of::<Experience>() as u64)
    };

    CandidateRun {
        decision,
        retrieved_ids: if error.is_some() {
            Vec::new()
        } else {
            retrieved_ids
        },
        memory_digest: experience_list_digest(&resolved),
        resolved,
        latency_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        index_bytes,
        sidecar_bytes,
        abstention_reason: error.or_else(|| {
            if decision.action.is_none() {
                Some("no admissible decision-support evidence".to_string())
            } else {
                None
            }
        }),
    }
}

fn experience_list_digest(items: &[Experience]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:experience-list:v1\0");
    for item in items {
        hasher.update(experience_view_digest(item));
    }
    hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
}

fn action_name(action: Option<Action>) -> &'static str {
    match action {
        Some(Action::A) => "action_a",
        Some(Action::B) => "action_b",
        None => "abstain",
    }
}

fn action_utility(truth: stress::Truth, action: Option<Action>) -> f64 {
    truth.utility(action.unwrap_or(Action::B))
}

fn candidate_input_digest(case: &stress::Case) -> String {
    sha256(
        format!(
            "mace-candidate-input-v1|cluster={}|family={}|regime={:?}|objective={:?}",
            case.cluster, case.family, case.query.scope.regime, case.query.scope.objective
        )
        .as_bytes(),
    )
}

fn initial_state_digest(case: &stress::Case) -> String {
    sha256(format!("mace-agent-state-v1|cluster={}", case.cluster).as_bytes())
}

fn candidate_policy_digest(arm: Arm) -> String {
    sha256(
        format!(
            "mace-candidate-policy-v1|arm={}|top_k={}|dimension={}|codebook_seed={:016x}|neuron_seed={:016x}",
            arm.name(), TOP_K, HDC_DIMENSION, HDC_CODEBOOK_SEED, HDC_NEURON_SEED
        )
        .as_bytes(),
    )
}

fn candidate_record(
    run_id: &str,
    case: &stress::Case,
    arm: Arm,
    result: &CandidateRun,
) -> Value {
    let pair_block_id = format!("cluster-{}:{}", case.cluster, case.family);
    let query_id = format!("query-{}", sha256(pair_block_id.as_bytes())[..16].to_string());
    let decision_support_ids = result.decision.evidence_ids.clone();
    let trace_digest = json_sha256(&json!({
        "pair_block_id": pair_block_id,
        "arm": arm.name(),
        "decision": action_name(result.decision.action),
        "retrieved_event_ids": result.retrieved_ids,
        "decision_support_event_ids": decision_support_ids,
        "memory_snapshot_sha256": result.memory_digest,
        "latency_ns": result.latency_ns
    }))
    .unwrap_or_else(|_| sha256(b"candidate-trace-serialization-error"));

    let mut record = json!({
        "schema_version": SCHEMA_VERSION,
        "record_type": "candidate_decision",
        "run_id": run_id,
        "pair_block_id": pair_block_id,
        "world_cluster_id": format!("world-{}", case.cluster),
        "query_id": query_id,
        "arm": arm.name(),
        "initial_agent_state_sha256": initial_state_digest(case),
        "candidate_input_sha256": candidate_input_digest(case),
        "candidate_policy_sha256": candidate_policy_digest(arm),
        "memory_snapshot_sha256": result.memory_digest,
        "decision": action_name(result.decision.action),
        "retrieved_event_ids": result.retrieved_ids,
        "decision_support_event_ids": decision_support_ids,
        "decision_trace_sha256": trace_digest,
        "latency_ns": result.latency_ns,
        "input_tokens": 0,
        "tool_calls": 0,
        "index_bytes": result.index_bytes,
        "evidence_sidecar_bytes": result.sidecar_bytes
    });
    if let Some(reason) = &result.abstention_reason {
        record["abstention_reason"] = json!(reason);
    }
    record
}

fn evaluate_arm(
    case: &stress::Case,
    arm: Arm,
    candidate: &CandidateRun,
    no_memory_action: Option<Action>,
) -> ScoredArm {
    let action = candidate.decision.action;
    let selected_utility = action_utility(case.truth, action);
    let best_utility = case.truth.optimal_utility();
    let utility = utility_micros(selected_utility);
    let regret = utility_micros((best_utility - selected_utility).max(0.0));
    let mut invalid = 0_u64;
    let mut stale = 0_u64;
    let mut wrong_scope = 0_u64;

    let support_by_id: BTreeMap<u32, &Experience> = candidate
        .resolved
        .iter()
        .map(|item| (item.event_id, item))
        .collect();
    for id in &candidate.decision.evidence_ids {
        match support_by_id.get(id) {
            None => invalid += 1,
            Some(item) => {
                if !is_qualified(item, case.query) {
                    invalid += 1;
                }
                if item.source_retracted {
                    stale += 1;
                }
                if item.scope != case.query.scope {
                    wrong_scope += 1;
                }
            }
        }
    }
    if candidate.abstention_reason.as_deref().is_some_and(|reason| {
        reason.contains("duplicate") || reason.contains("unresolved") || reason.contains("revalidation")
    }) {
        invalid += 1;
    }

    let false_causal_claim_count = u64::from(
        candidate.decision.causal_claim.is_some()
            && candidate.decision.causal_claim != case.truth.causal_action,
    );
    ScoredArm {
        arm,
        cluster: case.cluster,
        family: case.family.to_string(),
        pair_block_id: format!("cluster-{}:{}", case.cluster, case.family),
        decision: action,
        utility_micros: utility,
        regret_micros: regret,
        task_success: action == Some(case.truth.optimal_action()),
        latency_ns: candidate.latency_ns,
        memory_bytes: i64::try_from(candidate.index_bytes.saturating_add(candidate.sidecar_bytes))
            .unwrap_or(i64::MAX),
        invalid_evidence_use_count: invalid,
        stale_evidence_use_count: stale,
        wrong_scope_evidence_use_count: wrong_scope,
        false_causal_claim_count,
        support_count: candidate.decision.evidence_ids.len(),
    }
}

fn evaluator_record(
    run_id: &str,
    case: &stress::Case,
    score: &ScoredArm,
    baseline_action: Option<Action>,
    placebo_changed: Option<bool>,
) -> Value {
    let decision_changed = score.decision != baseline_action;
    let support_is_valid = score.invalid_evidence_use_count == 0
        && score.stale_evidence_use_count == 0
        && score.wrong_scope_evidence_use_count == 0;
    let justified = if decision_changed {
        Some(
            support_is_valid
                && score.support_count > 0
                && case.truth.causal_action.is_some()
                && score.decision == case.truth.causal_action
                && score.decision.is_some(),
        )
    } else {
        None
    };
    let harmful = score.utility_micros
        < utility_micros(action_utility(case.truth, baseline_action));
    let outcome_digest = json_sha256(&json!({
        "cluster": case.cluster,
        "family": case.family,
        "utility_a": case.truth.utility_a,
        "utility_b": case.truth.utility_b,
        "optimal_action": case.truth.optimal_action().name()
    }))
    .unwrap_or_else(|_| sha256(b"outcome-serialization-error"));

    let mut record = json!({
        "schema_version": SCHEMA_VERSION,
        "record_type": "evaluator_score",
        "run_id": run_id,
        "pair_block_id": score.pair_block_id,
        "world_cluster_id": format!("world-{}", score.cluster),
        "query_id": format!("query-{}", sha256(score.pair_block_id.as_bytes())[..16].to_string()),
        "arm": score.arm.name(),
        "net_utility_micros": score.utility_micros,
        "regret_micros": score.regret_micros,
        "task_success": score.task_success,
        "invalid_evidence_use_count": score.invalid_evidence_use_count,
        "stale_evidence_use_count": score.stale_evidence_use_count,
        "wrong_scope_evidence_use_count": score.wrong_scope_evidence_use_count,
        "false_causal_claim_count": score.false_causal_claim_count,
        "decision_changed_vs_no_memory": decision_changed,
        "decision_change_justified": justified,
        "harmful_memory_dependence": harmful,
        "correction_recovery_event_count": 0,
        "outcome_receipt_sha256": outcome_digest,
        "evaluation_validity": "diagnostic_only",
        "invalidity_reason": "public synthetic fixtures and co-located evaluator; not independent or externally qualified"
    });
    if let Some(changed) = placebo_changed {
        record["placebo_changed_decision"] = json!(changed);
    }
    record
}

#[derive(Clone, Debug)]
struct ContrastRows {
    utility_deltas_by_cluster: BTreeMap<u64, Vec<f64>>,
    regret_deltas: Vec<f64>,
    latency_deltas: Vec<i64>,
    memory_deltas: Vec<i64>,
    harmful_count: usize,
    pair_count: usize,
    invalid_use: u64,
    stale_use: u64,
    family_success: BTreeMap<String, (usize, usize, usize, usize)>,
}

impl Default for ContrastRows {
    fn default() -> Self {
        Self {
            utility_deltas_by_cluster: BTreeMap::new(),
            regret_deltas: Vec::new(),
            latency_deltas: Vec::new(),
            memory_deltas: Vec::new(),
            harmful_count: 0,
            pair_count: 0,
            invalid_use: 0,
            stale_use: 0,
            family_success: BTreeMap::new(),
        }
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * p.clamp(0.0, 1.0)).round() as usize;
    sorted[index]
}

fn bootstrap_interval(cluster_values: &[f64], seed: u64) -> (f64, f64) {
    if cluster_values.is_empty() {
        return (0.0, 0.0);
    }
    if cluster_values.len() == 1 {
        return (cluster_values[0], cluster_values[0]);
    }
    let mut state = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
    let mut samples = Vec::with_capacity(BOOTSTRAP_REPLICATES);
    for _ in 0..BOOTSTRAP_REPLICATES {
        let mut total = 0.0;
        for _ in 0..cluster_values.len() {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let draw = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            total += cluster_values[(draw as usize) % cluster_values.len()];
        }
        samples.push(total / cluster_values.len() as f64);
    }
    samples.sort_by(f64::total_cmp);
    (percentile(&samples, 0.025), percentile(&samples, 0.975))
}

fn contrast_record(
    contrast_id: &str,
    baseline: Arm,
    treatment: Arm,
    rows: &BTreeMap<(u64, String, Arm), ScoredArm>,
) -> Value {
    let mut contrast = ContrastRows::default();
    let mut keys = BTreeSet::new();
    for ((cluster, family, arm), _) in rows {
        if *arm == baseline {
            keys.insert((*cluster, family.clone()));
        }
    }

    for (cluster, family) in keys {
        let Some(base) = rows.get(&(cluster, family.clone(), baseline)) else {
            continue;
        };
        let Some(treat) = rows.get(&(cluster, family.clone(), treatment)) else {
            continue;
        };
        let delta = (treat.utility_micros - base.utility_micros) as f64;
        contrast.utility_deltas_by_cluster.entry(cluster).or_default().push(delta);
        contrast.regret_deltas.push((treat.regret_micros - base.regret_micros) as f64);
        contrast.latency_deltas.push(
            i64::try_from(treat.latency_ns).unwrap_or(i64::MAX)
                - i64::try_from(base.latency_ns).unwrap_or(i64::MAX),
        );
        contrast.memory_deltas.push(treat.memory_bytes - base.memory_bytes);
        contrast.harmful_count += usize::from(treat.utility_micros < base.utility_micros);
        contrast.pair_count += 1;
        contrast.invalid_use += treat.invalid_evidence_use_count;
        contrast.stale_use += treat.stale_evidence_use_count;

        let entry = contrast
            .family_success
            .entry(family)
            .or_insert((0, 0, 0, 0));
        entry.0 += usize::from(base.task_success);
        entry.1 += 1;
        entry.2 += usize::from(treat.task_success);
        entry.3 += 1;
    }

    let cluster_means: Vec<f64> = contrast
        .utility_deltas_by_cluster
        .values()
        .map(|deltas| deltas.iter().sum::<f64>() / deltas.len() as f64)
        .collect();
    let mean_delta = if contrast.pair_count == 0 {
        0.0
    } else {
        contrast.utility_deltas_by_cluster.values()
            .flat_map(|items| items.iter())
            .sum::<f64>() / contrast.pair_count as f64
    };
    let (ci_lower, ci_upper) = bootstrap_interval(
        &cluster_means,
        0x4D41_4345_0000_0000_u64 ^ (contrast_id.bytes().map(u64::from).sum::<u64>()),
    );
    let mean_regret_delta = if contrast.regret_deltas.is_empty() {
        0.0
    } else {
        contrast.regret_deltas.iter().sum::<f64>() / contrast.regret_deltas.len() as f64
    };
    let mean_memory_delta = if contrast.memory_deltas.is_empty() {
        0.0
    } else {
        contrast.memory_deltas.iter().sum::<i64>() as f64 / contrast.memory_deltas.len() as f64
    };
    let worst_family_delta = contrast
        .family_success
        .values()
        .filter(|(_, base_count, _, treat_count)| *base_count > 0 && *treat_count > 0)
        .map(|(base_success, base_count, treat_success, treat_count)| {
            100.0 * (*treat_success as f64 / *treat_count as f64
                - *base_success as f64 / *base_count as f64)
        })
        .fold(f64::INFINITY, f64::min);
    let worst_family_delta = if worst_family_delta.is_finite() {
        worst_family_delta
    } else {
        0.0
    };
    let mut latency_deltas = contrast.latency_deltas.clone();
    latency_deltas.sort_unstable();
    let p95_latency_delta = if latency_deltas.is_empty() {
        0
    } else {
        let index = ((latency_deltas.len() - 1) as f64 * 0.95).round() as usize;
        latency_deltas[index]
    };

    json!({
        "contrast_id": contrast_id,
        "baseline_arm": baseline.name(),
        "treatment_arm": treatment.name(),
        "mean_utility_delta_micros": mean_delta,
        "ci95_lower_micros": ci_lower,
        "ci95_upper_micros": ci_upper,
        "world_cluster_count": contrast.utility_deltas_by_cluster.len(),
        "pair_block_count": contrast.pair_count,
        "mean_regret_delta_micros": mean_regret_delta,
        "harmful_dependence_rate": if contrast.pair_count == 0 { 0.0 } else { contrast.harmful_count as f64 / contrast.pair_count as f64 },
        "invalid_evidence_use_count": contrast.invalid_use,
        "stale_evidence_use_count": contrast.stale_use,
        "worst_family_success_delta_pp": worst_family_delta,
        "p95_latency_delta_ns": p95_latency_delta,
        "all_in_memory_bytes_delta": mean_memory_delta.round() as i64
    })
}

fn write_jsonl(path: &str, records: &[Value]) -> Result<(), String> {
    let file = File::create(path).map_err(|error| format!("{path}: {error}"))?;
    let mut writer = BufWriter::new(file);
    for record in records {
        serde_json::to_writer(&mut writer, record).map_err(|error| error.to_string())?;
        writer.write_all(b"\n").map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())
}

fn parse_source_revision() -> Result<String, String> {
    let value = env::var("MACE_SOURCE_REVISION")
        .map_err(|_| "set MACE_SOURCE_REVISION to the exact 40- or 64-character source commit".to_string())?;
    let valid_length = value.len() == 40 || value.len() == 64;
    if !valid_length || !value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()) {
        return Err("MACE_SOURCE_REVISION must be lowercase hexadecimal SHA-1 or SHA-256".to_string());
    }
    Ok(value)
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let clusters = args
        .first()
        .map(|value| value.parse::<u64>().map_err(|error| error.to_string()))
        .transpose()?
        .unwrap_or(256);
    if clusters == 0 || clusters > 10_000 {
        return Err("cluster count must be in 1..=10000".to_string());
    }
    let candidate_path = args.get(1).map(String::as_str).unwrap_or("mace_candidate_decisions.jsonl");
    let evaluator_path = args.get(2).map(String::as_str).unwrap_or("mace_evaluator_scores.jsonl");
    let summary_path = args.get(3).map(String::as_str).unwrap_or("mace_paired_summary.jsonl");
    let source_revision = parse_source_revision()?;

    let cases = stress::generated_worlds(clusters);
    let pair_count = cases.len();
    let cluster_count = cases.iter().map(|case| case.cluster).collect::<BTreeSet<_>>().len();
    let public_fixture_digest = sha256(include_bytes!("mace_generated_stress.rs"));
    let split_digest = sha256(format!("mace-split-v1|clusters={clusters}|seed=cluster-xorshift-v1").as_bytes());
    let scenario_digest = sha256(
        format!(
            "mace-scenarios-v1|clusters={clusters}|families=21|generator={public_fixture_digest}"
        )
        .as_bytes(),
    );
    let policy_digest = sha256(
        Arm::ALL
            .iter()
            .map(|arm| format!("{}:{}", arm.name(), candidate_policy_digest(*arm)))
            .collect::<Vec<_>>()
            .join("|")
            .as_bytes(),
    );
    let run_id = format!("mace-paired-{}-{}", &source_revision[..12], clusters);
    let manifest = json!({
        "schema_version": SCHEMA_VERSION,
        "record_type": "run_manifest",
        "run_id": run_id,
        "source_revision": source_revision,
        "mace_revision": source_revision,
        "evaluator_revision": source_revision,
        "split_manifest_sha256": split_digest,
        "scenario_manifest_sha256": scenario_digest,
        "candidate_policy_sha256": policy_digest,
        // This schema field is a compatibility slot. The evaluator is public
        // and co-located; the digest is not a secret or independent custody.
        "hidden_evaluator_digest": public_fixture_digest,
        "independent_world_cluster_count": cluster_count,
        "pair_block_count": pair_count,
        "arms": Arm::ALL.iter().map(|arm| arm.name()).collect::<Vec<_>>(),
        "randomization": "deterministic_replay",
        "common_random_numbers": true,
        "model_revision": "deterministic-rust-policy-no-llm-v1"
    });

    let mut candidate_records = vec![manifest];
    let mut evaluator_records = Vec::new();
    let mut scores: BTreeMap<(u64, String, Arm), ScoredArm> = BTreeMap::new();

    for case in &cases {
        let mut arm_results = BTreeMap::new();
        for arm in Arm::ALL {
            let result = execute_arm(arm, case, &cases);
            candidate_records.push(candidate_record(&run_id, case, arm, &result));
            arm_results.insert(arm, result);
        }
        let baseline_action = arm_results
            .get(&Arm::NoMemory)
            .and_then(|result| result.decision.action);

        for arm in Arm::ALL {
            let result = arm_results
                .get(&arm)
                .ok_or_else(|| format!("missing arm result {}", arm.name()))?;
            let score = evaluate_arm(case, arm, result, baseline_action);
            let placebo_changed = if arm == Arm::SemanticPlacebo {
                Some(score.decision != baseline_action)
            } else {
                None
            };
            evaluator_records.push(evaluator_record(
                &run_id,
                case,
                &score,
                baseline_action,
                placebo_changed,
            ));
            scores.insert((case.cluster, case.family.to_string(), arm), score);
        }
    }

    let contrasts = vec![
        contrast_record("memory_vs_none", Arm::NoMemory, Arm::HdcLtc, &scores),
        contrast_record("content_vs_placebo", Arm::SemanticPlacebo, Arm::HdcLtc, &scores),
        contrast_record("hdc_ltc_vs_conventional", Arm::ConventionalRetrieval, Arm::HdcLtc, &scores),
        contrast_record("hdc_ltc_vs_static_hdc", Arm::StaticHdc, Arm::HdcLtc, &scores),
    ];
    let summary = json!({
        "schema_version": SCHEMA_VERSION,
        "record_type": "paired_summary",
        "run_id": run_id,
        "source_revision": source_revision,
        "split_manifest_sha256": split_digest,
        "primary_contrast_id": "memory_vs_none",
        "independent_world_cluster_count": cluster_count,
        "pair_block_count": pair_count,
        "contrasts": contrasts
    });

    write_jsonl(candidate_path, &candidate_records)?;
    write_jsonl(evaluator_path, &evaluator_records)?;
    write_jsonl(summary_path, &[summary])?;
    eprintln!(
        "diagnostic-only MACE paired runner completed: clusters={cluster_count}, pair_blocks={pair_count}, arms={}, candidate_records={}, evaluator_records={}",
        Arm::ALL.len(),
        candidate_records.len() - 1,
        evaluator_records.len()
    );
    eprintln!("candidate stream: {candidate_path}");
    eprintln!("evaluator stream: {evaluator_path}");
    eprintln!("summary stream: {summary_path}");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mace_paired_intervention: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn true_no_memory_arm_has_no_prior_memory_or_evidence() {
        let cases = stress::generated_worlds(1);
        let case = &cases[0];
        let result = execute_arm(Arm::NoMemory, case, &cases);
        assert!(result.retrieved_ids.is_empty());
        assert!(result.resolved.is_empty());
        assert!(result.decision.evidence_ids.is_empty());
        assert_eq!(result.decision.action, Some(Action::B));
    }

    #[test]
    fn candidate_input_digest_is_identical_across_memory_arms() {
        let cases = stress::generated_worlds(1);
        let case = &cases[0];
        let digest = candidate_input_digest(case);
        for arm in Arm::ALL {
            let result = execute_arm(arm, case, &cases);
            let record = candidate_record("test-run", case, arm, &result);
            assert_eq!(record["candidate_input_sha256"], digest);
            assert_eq!(record["initial_agent_state_sha256"], initial_state_digest(case));
        }
    }

    #[test]
    fn candidate_stream_never_contains_evaluator_truth_or_utility_fields() {
        let cases = stress::generated_worlds(1);
        let case = &cases[0];
        let result = execute_arm(Arm::HdcLtc, case, &cases);
        let record = candidate_record("test-run", case, Arm::HdcLtc, &result);
        let serialized = record.to_string();
        for forbidden in [
            "utility_a",
            "utility_b",
            "optimal_action",
            "causal_action",
            "net_utility_micros",
            "regret_micros",
            "task_success",
        ] {
            assert!(!serialized.contains(forbidden), "candidate leaked {forbidden}");
        }
    }

    #[test]
    fn placebo_source_never_crosses_world_cluster_and_is_not_same_task() {
        let cases = stress::generated_worlds(2);
        let case = &cases[0];
        let placebo = source_memory_for_arm(Arm::SemanticPlacebo, case, &cases);
        let target_ids: BTreeSet<u32> = case.memory.iter().map(|item| item.event_id).collect();
        assert!(placebo.iter().all(|item| !target_ids.contains(&item.event_id)));
        let sibling = cases
            .iter()
            .find(|candidate| {
                candidate.cluster == case.cluster
                    && candidate.family != case.family
                    && candidate.query.scope.regime == case.query.scope.regime
                    && candidate.query.scope.objective == case.query.scope.objective
            });
        assert!(sibling.is_some());
    }

    #[test]
    fn cluster_bootstrap_is_deterministic_and_uses_cluster_means() {
        let values = vec![-1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            bootstrap_interval(&values, 12345),
            bootstrap_interval(&values, 12345)
        );
        let (lower, upper) = bootstrap_interval(&values, 12345);
        assert!(lower <= upper);
    }
}
