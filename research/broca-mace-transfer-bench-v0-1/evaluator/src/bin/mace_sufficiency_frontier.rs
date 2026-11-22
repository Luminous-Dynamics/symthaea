// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Diagnostic decision-sufficiency frontier over the typed MACE generated corpus.
//!
//! Candidate selection functions receive only DecisionNeed, Query and Experience.
//! Evaluator truth is read only after a candidate decision has been frozen.
//! This runner deliberately does not synthesize text from Experience structs:
//! BM25/dense-text/raw-trajectory comparisons require real source-session adapters.
//!
//! No authenticated dependency sidecar is present in this generated-world lane.
//! Results are diagnostic only; no evidence-qualified sufficiency claim is emitted.

#[path = "mace_generated_stress.rs"]
mod mace_generated_stress;

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::mem::size_of;
use std::time::Instant;

use broca_mace_hdc_retrieval::{
    experience_view_digest,
    typed_content::{SemanticExperienceQueryV1, TypedContentConfigV1, TypedContentIndexV1},
    CandidateIndex, IndexConfig, RepresentationProfile,
};
use broca_mace_policy::{
    decide, Action, Decision, DecisionNeedV1, Experience, Policy, Query,
};
use serde_json::json;
use sha2::{Digest, Sha256};

const SCHEMA_VERSION: &str = "mace_sufficiency_frontier_v0_1";
const POLICY: Policy = Policy::CausalQualified;
const HDC_DIMENSION: usize = 2_048;
const HDC_SEED: u64 = 0x4D41_4345_4844_4301;
const DEFAULT_EPSILON: f64 = 0.01;
const PER_ACTION_BUDGETS: [usize; 4] = [1, 2, 4, 8];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RetrievalArm {
    NoMemory,
    FullScopeReference,
    EventIdOrderedControl,
    RandomControl,
    StaticHdc,
    TypedContentHdc,
}

impl RetrievalArm {
    const BUDGETED: [Self; 4] = [
        Self::EventIdOrderedControl,
        Self::RandomControl,
        Self::StaticHdc,
        Self::TypedContentHdc,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::NoMemory => "no_memory",
            Self::FullScopeReference => "full_scope_reference",
            Self::EventIdOrderedControl => "event_id_ordered_control",
            Self::RandomControl => "random_control",
            Self::StaticHdc => "static_hdc",
            Self::TypedContentHdc => "typed_content_hdc",
        }
    }
}

struct RetrievalIndexes {
    static_hdc: CandidateIndex,
    typed_content: TypedContentIndexV1,
    static_build_ns: u64,
    typed_build_ns: u64,
}

#[derive(Clone, Debug)]
struct CandidatePoint {
    method: &'static str,
    per_action_top_k: Option<usize>,
    selected: Vec<Experience>,
    decision: Decision,
    candidate_input_sha256: String,
    candidate_observation_id: String,
    index_payload_bytes: u64,
    index_build_ns: u64,
    candidate_selection_and_decision_ns: u64,
    retrieval_profile_sha256: Option<String>,
}

#[derive(Clone, Copy, Debug)]
struct CostVector {
    selected_experience_struct_bytes: u64,
    index_payload_bytes: u64,
    measured_latency_ns: u64,
    regret: f64,
}

#[derive(Clone, Debug)]
struct EvaluatedPoint {
    candidate: CandidatePoint,
    utility: f64,
    regret: f64,
    regret_gap_vs_full: f64,
    decision_agrees_with_full: bool,
    oracle_optimal: bool,
    pareto_frontier_member: bool,
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

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn valid_git_revision(value: &str) -> bool {
    matches!(value.len(), 40 | 64)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn action_name(action: Option<Action>) -> &'static str {
    match action {
        Some(Action::A) => "action_a",
        Some(Action::B) => "action_b",
        None => "abstain",
    }
}

fn decision_utility(
    truth: mace_generated_stress::Truth,
    action: Option<Action>,
) -> f64 {
    action.map(|selected| truth.utility(selected)).unwrap_or(0.0)
}

fn decision_regret(
    truth: mace_generated_stress::Truth,
    action: Option<Action>,
) -> f64 {
    (truth.optimal_utility() - decision_utility(truth, action)).max(0.0)
}

fn epsilon_sufficient(regret_gap: f64, epsilon: f64) -> bool {
    epsilon.is_finite() && epsilon >= 0.0 && regret_gap.is_finite() && regret_gap <= epsilon
}

fn dominates(left: CostVector, right: CostVector) -> bool {
    let no_worse = left.selected_experience_struct_bytes
        <= right.selected_experience_struct_bytes
        && left.index_payload_bytes <= right.index_payload_bytes
        && left.measured_latency_ns <= right.measured_latency_ns
        && left.regret <= right.regret;
    let strictly_better = left.selected_experience_struct_bytes
        < right.selected_experience_struct_bytes
        || left.index_payload_bytes < right.index_payload_bytes
        || left.measured_latency_ns < right.measured_latency_ns
        || left.regret < right.regret;
    no_worse && strictly_better
}

fn stable_pair_block_id(cluster: u64, need: DecisionNeedV1) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:sufficiency-pair-block:v1\0");
    hasher.update(cluster.to_be_bytes());
    hasher.update(need.canonical_digest().as_bytes());
    format!("world-{cluster}:need-{}", &hex_digest(hasher.finalize().as_slice())[..16])
}

fn candidate_identity(
    need: DecisionNeedV1,
    selected: &[Experience],
    method: &str,
    per_action_top_k: Option<usize>,
) -> (String, String) {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:sufficiency-candidate-input:v1\0");
    hasher.update(need.canonical_digest().as_bytes());
    hasher.update((method.len() as u64).to_be_bytes());
    hasher.update(method.as_bytes());
    let budget = per_action_top_k
        .and_then(|value| u64::try_from(value).ok())
        .unwrap_or(u64::MAX);
    hasher.update(budget.to_be_bytes());
    let mut ids: Vec<&Experience> = selected.iter().collect();
    ids.sort_by_key(|item| item.event_id);
    for item in ids {
        hasher.update(item.event_id.to_be_bytes());
        hasher.update(experience_view_digest(item));
    }
    let digest = hex_digest(hasher.finalize().as_slice());
    let observation = format!("obs-{}", &digest[..20]);
    (digest, observation)
}

fn resolve_ids(
    memory_by_id: &BTreeMap<u32, Experience>,
    ids: impl IntoIterator<Item = u32>,
) -> Result<Vec<Experience>, String> {
    let mut normalized = BTreeSet::new();
    for id in ids {
        normalized.insert(id);
    }
    normalized
        .into_iter()
        .map(|id| {
            memory_by_id
                .get(&id)
                .copied()
                .ok_or_else(|| format!("retrieval returned unresolved candidate event ID {id}"))
        })
        .collect()
}

fn build_indexes(memory: &[Experience]) -> Result<RetrievalIndexes, String> {
    let started = Instant::now();
    let static_hdc = CandidateIndex::build(
        memory,
        IndexConfig {
            dimension: HDC_DIMENSION,
            codebook_seed: HDC_SEED,
            profile: RepresentationProfile::StaticContinuousHv,
        },
    )
    .map_err(|error| format!("static HDC index build: {error}"))?;
    let static_build_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);

    let typed_started = Instant::now();
    let typed_content = TypedContentIndexV1::build(
        memory,
        TypedContentConfigV1 {
            dimension: HDC_DIMENSION,
            ..TypedContentConfigV1::default()
        },
    )
    .map_err(|error| format!("typed-content index build: {error}"))?;
    let typed_build_ns = u64::try_from(typed_started.elapsed().as_nanos()).unwrap_or(u64::MAX);

    Ok(RetrievalIndexes {
        static_hdc,
        typed_content,
        static_build_ns,
        typed_build_ns,
    })
}

/// Candidate-only selection. This function has no access to evaluator Truth.
fn select_candidate(
    arm: RetrievalArm,
    per_action_top_k: Option<usize>,
    memory: &[Experience],
    memory_by_id: &BTreeMap<u32, Experience>,
    query: Query,
    need: DecisionNeedV1,
    indexes: &RetrievalIndexes,
    public_seed: u64,
    cluster: u64,
) -> Result<CandidatePoint, String> {
    let started = Instant::now();
    let mut selected_ids = Vec::new();
    let mut profile_sha256 = None;
    let mut index_payload_bytes = 0_u64;
    let mut index_build_ns = 0_u64;

    match arm {
        RetrievalArm::NoMemory => {}
        RetrievalArm::FullScopeReference => {
            selected_ids.extend(
                memory
                    .iter()
                    .filter(|item| {
                        item.scope == query.scope && need.action_alternatives.contains(&item.action)
                    })
                    .map(|item| item.event_id),
            );
        }
        RetrievalArm::EventIdOrderedControl => {
            let k = per_action_top_k.ok_or("event-ID ordered arm requires a budget")?;
            for action in need.action_alternatives {
                let mut candidates: Vec<&Experience> = memory
                    .iter()
                    .filter(|item| item.scope == query.scope && item.action == action)
                    .collect();
                candidates.sort_by_key(|item| item.event_id);
                selected_ids.extend(candidates.into_iter().take(k).map(|item| item.event_id));
            }
        }
        RetrievalArm::RandomControl => {
            let k = per_action_top_k.ok_or("random-control arm requires a budget")?;
            for action in need.action_alternatives {
                let mut candidates: Vec<&Experience> = memory
                    .iter()
                    .filter(|item| item.scope == query.scope && item.action == action)
                    .collect();
                candidates.sort_by_key(|item| {
                    (
                        mix64(
                            public_seed
                                ^ cluster.rotate_left(17)
                                ^ u64::from(item.event_id)
                                ^ match action {
                                    Action::A => 0xA11C_E001,
                                    Action::B => 0xB11C_E002,
                                },
                        ),
                        item.event_id,
                    )
                });
                selected_ids.extend(candidates.into_iter().take(k).map(|item| item.event_id));
            }
        }
        RetrievalArm::StaticHdc => {
            let k = per_action_top_k.ok_or("static HDC arm requires a budget")?;
            index_payload_bytes = indexes.static_hdc.index_payload_bytes();
            index_build_ns = indexes.static_build_ns;
            profile_sha256 = Some(indexes.static_hdc.profile_digest().to_owned());
            for action in need.action_alternatives {
                let retrieval = indexes.static_hdc.retrieve(query.scope, action, k);
                for hit in retrieval.candidates {
                    let canonical = memory_by_id
                        .get(&hit.event_id)
                        .ok_or_else(|| format!("static HDC returned missing event {}", hit.event_id))?;
                    if experience_view_digest(canonical) != hit.view_digest {
                        return Err(format!("static HDC canonical view mismatch for event {}", hit.event_id));
                    }
                    selected_ids.push(hit.event_id);
                }
            }
        }
        RetrievalArm::TypedContentHdc => {
            let k = per_action_top_k.ok_or("typed-content HDC arm requires a budget")?;
            index_payload_bytes = indexes.typed_content.index_payload_bytes();
            index_build_ns = indexes.typed_build_ns;
            profile_sha256 = Some(indexes.typed_content.profile_digest().to_owned());
            for action in need.action_alternatives {
                let typed_query = SemanticExperienceQueryV1::from_decision_need(need, action)
                    .map_err(|error| format!("typed-content query: {error}"))?;
                let retrieval = indexes
                    .typed_content
                    .retrieve(typed_query, k)
                    .map_err(|error| format!("typed-content retrieval: {error}"))?;
                for hit in retrieval.candidates {
                    let canonical = memory_by_id
                        .get(&hit.event_id)
                        .ok_or_else(|| format!("typed-content returned missing event {}", hit.event_id))?;
                    if experience_view_digest(canonical) != hit.view_digest {
                        return Err(format!("typed-content canonical view mismatch for event {}", hit.event_id));
                    }
                    selected_ids.push(hit.event_id);
                }
            }
        }
    }

    let selected = resolve_ids(memory_by_id, selected_ids)?;
    let decision = decide(POLICY, &selected, query);
    let candidate_selection_and_decision_ns =
        u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let (candidate_input_sha256, candidate_observation_id) = candidate_identity(
        need,
        &selected,
        arm.name(),
        per_action_top_k,
    );

    Ok(CandidatePoint {
        method: arm.name(),
        per_action_top_k,
        selected,
        decision,
        candidate_input_sha256,
        candidate_observation_id,
        index_payload_bytes,
        index_build_ns,
        candidate_selection_and_decision_ns,
        retrieval_profile_sha256: profile_sha256,
    })
}

fn cost_vector(point: &EvaluatedPoint) -> CostVector {
    CostVector {
        selected_experience_struct_bytes: (point.candidate.selected.len() as u64)
            .saturating_mul(size_of::<Experience>() as u64),
        index_payload_bytes: point.candidate.index_payload_bytes,
        measured_latency_ns: point
            .candidate
            .index_build_ns
            .saturating_add(point.candidate.candidate_selection_and_decision_ns),
        regret: point.regret,
    }
}

fn assign_pareto_frontier(points: &mut [EvaluatedPoint]) {
    let costs: Vec<CostVector> = points.iter().map(cost_vector).collect();
    for index in 0..points.len() {
        points[index].pareto_frontier_member = !(0..points.len())
            .any(|other| other != index && dominates(costs[other], costs[index]));
    }
}

fn write_jsonl(path: &str, records: &[serde_json::Value]) -> Result<(), String> {
    let file = File::create(path).map_err(|error| format!("create {path}: {error}"))?;
    let mut writer = BufWriter::new(file);
    for record in records {
        serde_json::to_writer(&mut writer, record)
            .map_err(|error| format!("serialize {path}: {error}"))?;
        writer
            .write_all(b"\n")
            .map_err(|error| format!("write {path}: {error}"))?;
    }
    writer
        .flush()
        .map_err(|error| format!("flush {path}: {error}"))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let clusters = args
        .first()
        .map(|value| value.parse::<u64>().map_err(|error| error.to_string()))
        .transpose()?
        .unwrap_or(64);
    if clusters == 0 || clusters > 1_000 {
        return Err("cluster count must be in 1..=1000".to_string());
    }
    let candidate_path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("mace_sufficiency_candidate.jsonl");
    let evaluator_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("mace_sufficiency_evaluator.jsonl");
    let public_seed = args
        .get(3)
        .map(|value| value.parse::<u64>().map_err(|error| error.to_string()))
        .transpose()?
        .unwrap_or(0x4D41_4345_5355_4631);
    let epsilon = args
        .get(4)
        .map(|value| value.parse::<f64>().map_err(|error| error.to_string()))
        .transpose()?
        .unwrap_or(DEFAULT_EPSILON);
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err("epsilon must be finite and non-negative".to_string());
    }

    let cases = mace_generated_stress::generated_worlds(clusters);
    let source_revision = env::var("MACE_SOURCE_REVISION").ok();
    if source_revision
        .as_deref()
        .is_some_and(|revision| !valid_git_revision(revision))
    {
        return Err("MACE_SOURCE_REVISION must be a 40- or 64-character lowercase Git SHA".to_string());
    }

    let mut code_hasher = Sha256::new();
    code_hasher.update(include_bytes!("mace_sufficiency_frontier.rs"));
    code_hasher.update(include_bytes!("mace_generated_stress.rs"));
    code_hasher.update(include_bytes!("../../../policy/src/lib.rs"));
    code_hasher.update(include_bytes!("../../../hdc-retrieval/src/lib.rs"));
    code_hasher.update(include_bytes!("../../../hdc-retrieval/src/typed_content.rs"));
    let source_code_sha256 = hex_digest(code_hasher.finalize().as_slice());
    let fixture_sha256 = hex_digest(
        Sha256::digest(include_bytes!("mace_generated_stress.rs")).as_slice(),
    );
    let schema_sha256 = hex_digest(
        Sha256::digest(include_bytes!(
            "../../../../../data/schemas/mace_sufficiency_frontier_v0_1.schema.json"
        ))
        .as_slice(),
    );
    let source_revision_label = source_revision.as_deref().unwrap_or("unspecified");
    let run_identity = format!(
        "{clusters}:{public_seed}:{epsilon:.12}:{SCHEMA_VERSION}:{source_code_sha256}:{fixture_sha256}:{schema_sha256}:{source_revision_label}"
    );
    let run_id = format!(
        "mace-sufficiency-{}-{}",
        &hex_digest(Sha256::digest(run_identity.as_bytes()).as_slice())[..16],
        clusters
    );

    let mut candidate_records = Vec::new();
    let mut evaluator_records = Vec::new();

    for case in &cases {
        let mut memory_by_id = BTreeMap::new();
        for item in &case.memory {
            if memory_by_id.insert(item.event_id, *item).is_some() {
                return Err(format!("world {} has duplicate event ID {}", case.cluster, item.event_id));
            }
        }
        let indexes = build_indexes(&case.memory)?;
        let pair_block_id = stable_pair_block_id(case.cluster, case.decision_need);

        // Full-scope reference is candidate-visible; evaluator Truth is still
        // withheld until after its decision and every budgeted candidate exist.
        let full = select_candidate(
            RetrievalArm::FullScopeReference,
            None,
            &case.memory,
            &memory_by_id,
            case.query,
            case.decision_need,
            &indexes,
            public_seed,
            case.cluster,
        )?;
        let full_action = full.decision.action;

        let mut candidates = Vec::new();
        candidates.push(select_candidate(
            RetrievalArm::NoMemory,
            None,
            &case.memory,
            &memory_by_id,
            case.query,
            case.decision_need,
            &indexes,
            public_seed,
            case.cluster,
        )?);
        candidates.push(full);

        for arm in RetrievalArm::BUDGETED {
            for budget in PER_ACTION_BUDGETS {
                candidates.push(select_candidate(
                    arm,
                    Some(budget),
                    &case.memory,
                    &memory_by_id,
                    case.query,
                    case.decision_need,
                    &indexes,
                    public_seed,
                    case.cluster,
                )?);
            }
        }

        // Only after every candidate has been selected and decided do we
        // read the evaluator-only truth to score the frozen outputs.
        let full_regret = decision_regret(case.truth, full_action);
        let mut evaluated = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let utility = decision_utility(case.truth, candidate.decision.action);
            let regret = decision_regret(case.truth, candidate.decision.action);
            let regret_gap = regret - full_regret;
            evaluated.push(EvaluatedPoint {
                decision_agrees_with_full: candidate.decision.action == full_action,
                oracle_optimal: candidate.decision.action == Some(case.truth.optimal_action()),
                candidate,
                utility,
                regret,
                regret_gap_vs_full: regret_gap,
                pareto_frontier_member: false,
            });
        }
        assign_pareto_frontier(&mut evaluated);

        let available_struct_bytes = (case.memory.len() as u64)
            .saturating_mul(size_of::<Experience>() as u64);
        for point in &evaluated {
            let selected_struct_bytes = (point.candidate.selected.len() as u64)
                .saturating_mul(size_of::<Experience>() as u64);
            let candidate_json = json!({
                "schema_version": SCHEMA_VERSION,
                "record_type": "candidate_decision",
                "run_id": run_id,
                "pair_block_id": pair_block_id,
                "world_cluster": case.cluster,
                "public_seed": public_seed,
                "source_revision": source_revision,
                "source_code_sha256": source_code_sha256,
                "fixture_sha256": fixture_sha256,
                "schema_sha256": schema_sha256,
                "public_seed": public_seed,
                "source_revision": source_revision,
                "source_code_sha256": source_code_sha256,
                "fixture_sha256": fixture_sha256,
                "schema_sha256": schema_sha256,
                "candidate_observation_id": point.candidate.candidate_observation_id,
                "candidate_input_sha256": point.candidate.candidate_input_sha256,
                "decision_need_sha256": case.decision_need.canonical_digest().to_hex(),
                "method": point.candidate.method,
                "policy": POLICY.name(),
                "per_action_top_k": point.candidate.per_action_top_k,
                "selected_event_ids": point.candidate.selected.iter().map(|item| item.event_id).collect::<Vec<_>>(),
                "decision": action_name(point.candidate.decision.action),
                "decision_support_event_ids": point.candidate.decision.evidence_ids,
                "probe_requested": point.candidate.decision.probe.is_some(),
                "available_event_count": case.memory.len(),
                "selected_event_count": point.candidate.selected.len(),
                "available_experience_struct_bytes_lower_bound": available_struct_bytes,
                "selected_experience_struct_bytes_lower_bound": selected_struct_bytes,
                "index_payload_bytes_lower_bound": point.candidate.index_payload_bytes,
                "index_build_ns": point.candidate.index_build_ns,
                "candidate_selection_and_decision_ns": point.candidate.candidate_selection_and_decision_ns,
                "retrieval_profile_sha256": point.candidate.retrieval_profile_sha256,
                "dependency_sidecar_sha256": null,
                "dependency_sidecar_ingress_verified": null,
                "lineage_incomplete_event_count": case.memory.len(),
                "evidence_qualification": "diagnostic_only_missing_authenticated_lineage",
                "token_cost": null,
                "all_in_cost_accounting": "struct_and_index_payload_lower_bounds_only"
            });
            candidate_records.push(candidate_json);

            let evaluator_json = json!({
                "schema_version": SCHEMA_VERSION,
                "record_type": "evaluator_score",
                "run_id": run_id,
                "pair_block_id": pair_block_id,
                "candidate_observation_id": point.candidate.candidate_observation_id,
                "method": point.candidate.method,
                "per_action_top_k": point.candidate.per_action_top_k,
                "scenario_family": case.family,
                "decision": action_name(point.candidate.decision.action),
                "evaluator_optimal_action": action_name(Some(case.truth.optimal_action())),
                "utility": point.utility,
                "absolute_regret": point.regret,
                "full_scope_reference_regret": full_regret,
                "regret_gap_vs_full_scope_reference": point.regret_gap_vs_full,
                "epsilon": epsilon,
                "decision_sufficient_by_regret_only": epsilon_sufficient(point.regret_gap_vs_full, epsilon),
                "evidence_qualified_sufficient": false,
                "decision_agrees_with_full_scope_reference": point.decision_agrees_with_full,
                "oracle_optimal": point.oracle_optimal,
                "pareto_frontier_member": point.pareto_frontier_member,
                "selected_experience_struct_bytes_lower_bound": selected_struct_bytes,
                "index_payload_bytes_lower_bound": point.candidate.index_payload_bytes,
                "measured_index_and_selection_ns": point.candidate.index_build_ns.saturating_add(point.candidate.candidate_selection_and_decision_ns),
                "evidence_qualification": "diagnostic_only_missing_authenticated_lineage",
                "oracle_visibility": "evaluator_only"
            });
            evaluator_records.push(evaluator_json);
        }
    }

    write_jsonl(candidate_path, &candidate_records)?;
    write_jsonl(evaluator_path, &evaluator_records)?;
    eprintln!(
        "diagnostic-only MACE sufficiency frontier: worlds={}, candidate_records={}, evaluator_records={}, epsilon={epsilon}",
        cases.len(),
        candidate_records.len(),
        evaluator_records.len()
    );
    eprintln!("candidate stream: {candidate_path}");
    eprintln!("evaluator stream: {evaluator_path}");
    eprintln!("lineage: unauthenticated/absent; evidence-qualified sufficiency is never reported");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mace_sufficiency_frontier: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epsilon_sufficiency_rejects_invalid_margins() {
        assert!(epsilon_sufficient(0.005, 0.01));
        assert!(epsilon_sufficient(-0.01, 0.01));
        assert!(!epsilon_sufficient(0.02, 0.01));
        assert!(!epsilon_sufficient(0.0, f64::NAN));
        assert!(!epsilon_sufficient(f64::INFINITY, 0.01));
        assert!(!epsilon_sufficient(0.0, -0.01));
    }

    #[test]
    fn pareto_dominance_requires_no_worse_all_cost_dimensions() {
        let cheap = CostVector {
            selected_experience_struct_bytes: 64,
            index_payload_bytes: 0,
            measured_latency_ns: 10,
            regret: 0.2,
        };
        let dominated = CostVector {
            selected_experience_struct_bytes: 128,
            index_payload_bytes: 32,
            measured_latency_ns: 20,
            regret: 0.3,
        };
        let incomparable = CostVector {
            selected_experience_struct_bytes: 32,
            index_payload_bytes: 64,
            measured_latency_ns: 15,
            regret: 0.1,
        };
        assert!(dominates(cheap, dominated));
        assert!(!dominates(cheap, incomparable));
        assert!(!dominates(incomparable, cheap));
        assert!(!dominates(cheap, cheap));
    }

    #[test]
    fn random_control_order_is_seeded_and_event_order_invariant() {
        let first = mix64(9_u64 ^ 3_u64.rotate_left(17) ^ 41_u64 ^ 0xA11C_E001);
        let repeat = mix64(9_u64 ^ 3_u64.rotate_left(17) ^ 41_u64 ^ 0xA11C_E001);
        let different_seed = mix64(10_u64 ^ 3_u64.rotate_left(17) ^ 41_u64 ^ 0xA11C_E001);
        assert_eq!(first, repeat);
        assert_ne!(first, different_seed);
    }
}
