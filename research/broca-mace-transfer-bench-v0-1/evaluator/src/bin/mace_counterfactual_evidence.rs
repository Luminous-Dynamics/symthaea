// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Candidate-side counterfactual evidence interventions for MACE.
//!
//! The intervention planner accepts only candidate-visible Experience records,
//! a public seed and a coalition cap. It has no Truth/OracleTruth parameter.
//! Candidate and evaluator output streams are written separately.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::mem::size_of;
use std::time::Instant;

use broca_mace_hdc_retrieval::experience_view_digest;
use broca_mace_policy::{
    decide, Action, Decision, DecisionNeedV1, Experience, Objective, Policy, Query, Regime, SourceRootDigest,
};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

mod mace_generated_stress;

const SCHEMA_VERSION: &str = "mace_counterfactual_evidence_v0_1";
const ABSTENTION_UTILITY: f64 = 0.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum InterventionKind {
    Event,
    SourceRoot,
    SourceFamily,
    SourceRootPair,
    MatchedRandom,
    NoMemory,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct Intervention {
    intervention_id: String,
    kind: InterventionKind,
    target_key: String,
    removed_event_ids: Vec<u32>,
    matched_to: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct InterventionManifest {
    schema_version: &'static str,
    seed: u64,
    baseline_evidence_sha256: String,
    interventions: Vec<Intervention>,
    manifest_sha256: String,
}

#[derive(Clone, Debug)]
struct EvidenceDependencyIndex {
    records: BTreeMap<u32, Experience>,
    roots: BTreeMap<SourceRootDigest, BTreeSet<u32>>,
    families: BTreeMap<u16, BTreeSet<u32>>,
    unknown_root_ids: BTreeSet<u32>,
    unknown_family_ids: BTreeSet<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum IndexError {
    DuplicateEventId(u32),
}

impl EvidenceDependencyIndex {
    fn build(memory: &[Experience]) -> Result<Self, IndexError> {
        let mut records = BTreeMap::new();
        let mut roots: BTreeMap<SourceRootDigest, BTreeSet<u32>> = BTreeMap::new();
        let mut families: BTreeMap<u16, BTreeSet<u32>> = BTreeMap::new();
        let mut unknown_root_ids = BTreeSet::new();
        let mut unknown_family_ids = BTreeSet::new();

        for item in memory {
            if records.insert(item.event_id, *item).is_some() {
                return Err(IndexError::DuplicateEventId(item.event_id));
            }

            if item.source_root_digest.is_known() {
                roots
                    .entry(item.source_root_digest)
                    .or_default()
                    .insert(item.event_id);
            } else {
                unknown_root_ids.insert(item.event_id);
            }

            if item.source_family != 0 {
                families
                    .entry(item.source_family)
                    .or_default()
                    .insert(item.event_id);
            } else {
                unknown_family_ids.insert(item.event_id);
            }
        }

        Ok(Self {
            records,
            roots,
            families,
            unknown_root_ids,
            unknown_family_ids,
        })
    }

    fn event_count(&self) -> usize {
        self.records.len()
    }

    /// Only nonzero, canonical root digests count as known roots.
    fn known_root_count(&self) -> usize {
        self.roots.len()
    }

    /// Only nonzero source-family identities count as known families.
    fn known_family_count(&self) -> usize {
        self.families.len()
    }

    fn unknown_root_event_count(&self) -> usize {
        self.unknown_root_ids.len()
    }

    fn unknown_family_event_count(&self) -> usize {
        self.unknown_family_ids.len()
    }

    fn baseline_evidence_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"luminous:mace:candidate-evidence-set:v1\0");
        for (event_id, item) in &self.records {
            hasher.update(event_id.to_be_bytes());
            hasher.update(experience_view_digest(item));
        }
        hex_digest(hasher.finalize().as_slice())
    }

    fn retain_without(&self, removed_event_ids: &[u32]) -> Vec<Experience> {
        let removed: BTreeSet<u32> = removed_event_ids.iter().copied().collect();
        self.records
            .iter()
            .filter(|(event_id, _)| !removed.contains(event_id))
            .map(|(_, item)| *item)
            .collect()
    }

    fn support_closure_counts(&self, support_ids: &[u32]) -> (usize, usize, usize, usize) {
        let mut roots = BTreeSet::new();
        let mut families = BTreeSet::new();
        let mut unresolved = 0_usize;

        for event_id in support_ids {
            match self.records.get(event_id) {
                Some(item) => {
                    if item.source_root_digest.is_known() {
                        roots.insert(item.source_root_digest);
                    }
                    if item.source_family != 0 {
                        families.insert(item.source_family);
                    }
                }
                None => unresolved += 1,
            }
        }

        (support_ids.len(), roots.len(), families.len(), unresolved)
    }

    fn root_groups(&self) -> impl Iterator<Item = (String, Vec<u32>)> + '_ {
        self.roots.iter().map(|(root, ids)| {
            (format!("root:{}", root.to_hex()), ids.iter().copied().collect())
        })
    }

    fn family_groups(&self) -> impl Iterator<Item = (String, Vec<u32>)> + '_ {
        self.families.iter().map(|(family, ids)| {
            (format!("family:{family}"), ids.iter().copied().collect())
        })
    }
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

fn normalized_ids(mut ids: Vec<u32>) -> Vec<u32> {
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Select a disjoint dropout with the same per-action removal counts as the
/// target. A control is omitted if the candidate-visible pool cannot match it.
fn matched_random_ids(
    index: &EvidenceDependencyIndex,
    target_ids: &[u32],
    seed: u64,
) -> Option<Vec<u32>> {
    let target: BTreeSet<u32> = target_ids.iter().copied().collect();
    let mut required = BTreeMap::<(Regime, Objective, Action, String), usize>::new();
    for event_id in target_ids {
        let item = index.records.get(event_id)?;
        let stratum = (
            item.scope.regime,
            item.scope.objective,
            item.action,
            item.causal_identification.estimand_digest.to_hex(),
        );
        *required.entry(stratum).or_default() += 1;
    }

    let mut selected = Vec::new();
    for ((regime, objective, action, estimand), needed) in required {
        let mut eligible: Vec<&Experience> = index
            .records
            .values()
            .filter(|item| {
                item.scope.regime == regime
                    && item.scope.objective == objective
                    && item.action == action
                    && item.causal_identification.estimand_digest.to_hex() == estimand
                    && !target.contains(&item.event_id)
            })
            .collect();
        eligible.sort_by_key(|item| {
            mix64(seed ^ u64::from(item.event_id) ^ 0x4d41_5443_4845_4401)
        });
        if eligible.len() < needed {
            return None;
        }
        selected.extend(eligible.into_iter().take(needed).map(|item| item.event_id));
    }

    Some(normalized_ids(selected))
}

fn push_intervention(
    output: &mut Vec<Intervention>,
    kind: InterventionKind,
    target_key: String,
    removed_event_ids: Vec<u32>,
    matched_to: Option<String>,
) {
    let removed_event_ids = normalized_ids(removed_event_ids);
    if removed_event_ids.is_empty() {
        return;
    }
    let intervention_id = format!(
        "{:?}:{}:{}",
        kind,
        target_key,
        removed_event_ids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
    .to_ascii_lowercase();

    output.push(Intervention {
        intervention_id,
        kind,
        target_key,
        removed_event_ids,
        matched_to,
    });
}

/// The planner cannot access evaluator truth: its entire signature is
/// candidate-visible evidence + public seed + a deterministic coalition cap.
fn build_manifest(
    memory: &[Experience],
    public_seed: u64,
    max_root_pairs: usize,
) -> Result<InterventionManifest, IndexError> {
    let index = EvidenceDependencyIndex::build(memory)?;
    let baseline_evidence_sha256 = index.baseline_evidence_digest();
    let mut interventions = Vec::new();

    for event_id in index.records.keys() {
        push_intervention(
            &mut interventions,
            InterventionKind::Event,
            format!("event:{event_id}"),
            vec![*event_id],
            None,
        );
    }

    let root_groups: Vec<(String, Vec<u32>)> = index.root_groups().collect();
    for (key, ids) in &root_groups {
        push_intervention(
            &mut interventions,
            InterventionKind::SourceRoot,
            key.clone(),
            ids.clone(),
            None,
        );

        if let Some(random_ids) = matched_random_ids(
            &index,
            ids,
            public_seed ^ stable_key_seed(key),
        ) {
            push_intervention(
                &mut interventions,
                InterventionKind::MatchedRandom,
                format!("matched:{key}"),
                random_ids,
                Some(key.clone()),
            );
        }
    }

    for (key, ids) in index.family_groups() {
        push_intervention(
            &mut interventions,
            InterventionKind::SourceFamily,
            key.clone(),
            ids.clone(),
            None,
        );

        if let Some(random_ids) = matched_random_ids(
            &index,
            &ids,
            public_seed ^ stable_key_seed(&key),
        ) {
            push_intervention(
                &mut interventions,
                InterventionKind::MatchedRandom,
                format!("matched:{key}"),
                random_ids,
                Some(key),
            );
        }
    }

    let mut emitted_pairs = 0_usize;
    'pairs: for left in 0..root_groups.len() {
        for right in (left + 1)..root_groups.len() {
            if emitted_pairs >= max_root_pairs {
                break 'pairs;
            }
            let (left_key, left_ids) = &root_groups[left];
            let (right_key, right_ids) = &root_groups[right];
            let mut ids = left_ids.clone();
            ids.extend(right_ids.iter().copied());
            push_intervention(
                &mut interventions,
                InterventionKind::SourceRootPair,
                format!("{left_key}+{right_key}"),
                ids,
                None,
            );
            emitted_pairs += 1;
        }
    }

    push_intervention(
        &mut interventions,
        InterventionKind::NoMemory,
        "all_candidate_evidence".to_string(),
        index.records.keys().copied().collect(),
        None,
    );

    interventions.sort_by(|left, right| {
        left.kind
            .cmp(&right.kind)
            .then_with(|| left.target_key.cmp(&right.target_key))
            .then_with(|| left.removed_event_ids.cmp(&right.removed_event_ids))
    });

    let manifest_payload = serde_json::to_vec(&json!({
        "schema_version": SCHEMA_VERSION,
        "seed": public_seed,
        "baseline_evidence_sha256": baseline_evidence_sha256,
        "interventions": interventions
    }))
    .expect("candidate intervention manifest is serializable");
    let manifest_sha256 = hex_digest(Sha256::digest(manifest_payload).as_slice());

    Ok(InterventionManifest {
        schema_version: SCHEMA_VERSION,
        seed: public_seed,
        baseline_evidence_sha256,
        interventions,
        manifest_sha256,
    })
}

fn stable_key_seed(key: &str) -> u64 {
    let digest = Sha256::digest(key.as_bytes());
    u64::from_be_bytes(digest[..8].try_into().expect("SHA-256 prefix is eight bytes"))
}

fn action_name(action: Option<Action>) -> &'static str {
    match action {
        Some(Action::A) => "action_a",
        Some(Action::B) => "action_b",
        None => "abstain",
    }
}

fn candidate_input_digest(
    decision_need: DecisionNeedV1,
    evidence_sha256: &str,
) -> String {
    let payload = format!(
        "mace-candidate-input-v1|need={}|evidence={evidence_sha256}",
        decision_need.canonical_digest().to_hex()
    );
    hex_digest(Sha256::digest(payload.as_bytes()).as_slice())
}

#[derive(Clone, Debug)]
struct CandidateObservation {
    decision: Decision,
    input_sha256: String,
    evidence_sha256: String,
    evidence_event_count: usize,
    evidence_root_count: usize,
    evidence_family_count: usize,
    unknown_root_event_count: usize,
    unknown_family_event_count: usize,
    latency_ns: u64,
}

fn run_candidate(
    query: Query,
    decision_need: DecisionNeedV1,
    memory: &[Experience],
) -> Result<CandidateObservation, IndexError> {
    let started = Instant::now();
    let index = EvidenceDependencyIndex::build(memory)?;
    let evidence_sha256 = index.baseline_evidence_digest();
    let input_sha256 = candidate_input_digest(decision_need, &evidence_sha256);
    let decision = decide(Policy::CausalQualified, memory, query);

    Ok(CandidateObservation {
        decision,
        input_sha256,
        evidence_sha256,
        evidence_event_count: index.event_count(),
        evidence_root_count: index.known_root_count(),
        evidence_family_count: index.known_family_count(),
        unknown_root_event_count: index.unknown_root_event_count(),
        unknown_family_event_count: index.unknown_family_event_count(),
        latency_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
    })
}

fn truth_utility(truth: mace_generated_stress::Truth, action: Option<Action>) -> f64 {
    match action {
        Some(action) => truth.utility(action),
        None => ABSTENTION_UTILITY,
    }
}

fn regret(truth: mace_generated_stress::Truth, action: Option<Action>) -> f64 {
    (truth.optimal_utility() - truth_utility(truth, action)).max(0.0)
}

fn write_jsonl(path: &str, records: &[Value]) -> Result<(), String> {
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
    if clusters == 0 || clusters > 10_000 {
        return Err("cluster count must be in 1..=10000".to_string());
    }

    let candidate_path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("mace_counterfactual_candidate.jsonl");
    let evaluator_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("mace_counterfactual_evaluator.jsonl");
    let public_seed = args
        .get(3)
        .map(|value| value.parse::<u64>().map_err(|error| error.to_string()))
        .transpose()?
        .unwrap_or(0x4d41_4345_4346_3031);
    let max_root_pairs = args
        .get(4)
        .map(|value| value.parse::<usize>().map_err(|error| error.to_string()))
        .transpose()?
        .unwrap_or(32);

    let cases = mace_generated_stress::generated_worlds(clusters);
    let run_id = format!(
        "mace-counterfactual-{}-{}",
        &hex_digest(Sha256::digest(
            format!("{clusters}:{public_seed}:{SCHEMA_VERSION}").as_bytes()
        ).as_slice())[..16],
        clusters
    );
    let mut candidate_records = Vec::new();
    let mut evaluator_records = Vec::new();

    for case in &cases {
        let manifest = build_manifest(&case.memory, public_seed, max_root_pairs)
            .map_err(|error| format!("dependency index failed: {error:?}"))?;
        let baseline = run_candidate(case.query, case.decision_need, &case.memory)
            .map_err(|error| format!("baseline candidate failed: {error:?}"))?;
        let baseline_regret = regret(case.truth, baseline.decision.action);
        let pair_block_id = format!(
            "world-{}:{}:{:?}:{:?}",
            case.cluster, case.family, case.query.scope.regime, case.query.scope.objective
        );
        let manifest_record = json!({
            "schema_version": SCHEMA_VERSION,
            "record_type": "intervention_manifest",
            "run_id": run_id,
            "pair_block_id": pair_block_id,
            "manifest_sha256": manifest.manifest_sha256,
            "baseline_evidence_sha256": manifest.baseline_evidence_sha256,
            "public_seed": manifest.seed,
            "intervention_count": manifest.interventions.len(),
            "known_root_count": EvidenceDependencyIndex::build(&case.memory)
                .map_err(|error| format!("dependency index failed: {error:?}"))?
                .known_root_count(),
            "candidate_validity": "diagnostic_only"
        });
        evaluator_records.push(manifest_record);

        let baseline_support = {
            let index = EvidenceDependencyIndex::build(&case.memory)
                .map_err(|error| format!("dependency index failed: {error:?}"))?;
            index.support_closure_counts(&baseline.decision.evidence_ids)
        };
        candidate_records.push(json!({
            "schema_version": SCHEMA_VERSION,
            "record_type": "candidate_decision",
            "run_id": run_id,
            "pair_block_id": pair_block_id,
            "candidate_input_sha256": baseline.input_sha256,
            "decision_need_sha256": case.decision_need.canonical_digest().to_hex(),
            "candidate_evidence_sha256": baseline.evidence_sha256,
            "intervention_manifest_sha256": manifest.manifest_sha256,
            "intervention_id": "full_evidence_reference",
            "decision": action_name(baseline.decision.action),
            "decision_support_event_ids": baseline.decision.evidence_ids,
            "support_event_count": baseline_support.0,
            "support_known_root_count": baseline_support.1,
            "support_known_family_count": baseline_support.2,
            "support_unresolved_count": baseline_support.3,
            "available_event_count": baseline.evidence_event_count,
            "available_experience_struct_bytes_lower_bound": baseline.evidence_event_count.saturating_mul(size_of::<Experience>()),
            "available_known_root_count": baseline.evidence_root_count,
            "available_known_family_count": baseline.evidence_family_count,
            "unknown_root_event_count": baseline.unknown_root_event_count,
            "unknown_family_event_count": baseline.unknown_family_event_count,
            "latency_ns": baseline.latency_ns
        }));
        evaluator_records.push(json!({
            "schema_version": SCHEMA_VERSION,
            "record_type": "evaluator_score",
            "run_id": run_id,
            "pair_block_id": pair_block_id,
            "candidate_input_sha256": baseline.input_sha256,
            "intervention_id": "full_evidence_reference",
            "intervention_kind": "reference",
            "decision": action_name(baseline.decision.action),
            "utility": truth_utility(case.truth, baseline.decision.action),
            "regret": baseline_regret,
            "regret_gap_vs_full_evidence": 0.0,
            "decision_agrees_with_full": true,
            "decision_changed": false,
            "evaluation_validity": "diagnostic_only",
            "invalidity_reason": "public synthetic fixture and co-located evaluator"
        }));

        for intervention in &manifest.interventions {
            let index = EvidenceDependencyIndex::build(&case.memory)
                .map_err(|error| format!("dependency index failed: {error:?}"))?;
            let retained = index.retain_without(&intervention.removed_event_ids);
            let candidate = run_candidate(case.query, case.decision_need, &retained)
                .map_err(|error| format!("intervention candidate failed: {error:?}"))?;
            let support = EvidenceDependencyIndex::build(&retained)
                .map_err(|error| format!("retained evidence index failed: {error:?}"))?
                .support_closure_counts(&candidate.decision.evidence_ids);
            let candidate_observation_id = format!(
                "obs-{}",
                &candidate.input_sha256[..16]
            );

            candidate_records.push(json!({
                "schema_version": SCHEMA_VERSION,
                "record_type": "candidate_decision",
                "run_id": run_id,
                "pair_block_id": pair_block_id,
                "candidate_observation_id": candidate_observation_id,
                "candidate_input_sha256": candidate.input_sha256,
                "decision_need_sha256": case.decision_need.canonical_digest().to_hex(),
                "candidate_evidence_sha256": candidate.evidence_sha256,
                "intervention_manifest_sha256": manifest.manifest_sha256,
                "decision": action_name(candidate.decision.action),
                "decision_support_event_ids": candidate.decision.evidence_ids,
                "support_event_count": support.0,
                "support_known_root_count": support.1,
                "support_known_family_count": support.2,
                "support_unresolved_count": support.3,
                "available_event_count": candidate.evidence_event_count,
                "available_experience_struct_bytes_lower_bound": candidate.evidence_event_count.saturating_mul(size_of::<Experience>()),
                "available_known_root_count": candidate.evidence_root_count,
                "available_known_family_count": candidate.evidence_family_count,
                "unknown_root_event_count": candidate.unknown_root_event_count,
                "unknown_family_event_count": candidate.unknown_family_event_count,
                "latency_ns": candidate.latency_ns
            }));

            let current_regret = regret(case.truth, candidate.decision.action);
            let baseline_action = baseline.decision.action;
            evaluator_records.push(json!({
                "schema_version": SCHEMA_VERSION,
                "record_type": "evaluator_score",
                "run_id": run_id,
                "pair_block_id": pair_block_id,
                "candidate_observation_id": candidate_observation_id,
                "candidate_input_sha256": candidate.input_sha256,
                "intervention_id": intervention.intervention_id,
                "intervention_kind": intervention.kind,
                "target_key": intervention.target_key,
                "matched_to": intervention.matched_to,
                "removed_event_ids": intervention.removed_event_ids,
                "removed_event_count": intervention.removed_event_ids.len(),
                "removed_experience_struct_bytes_lower_bound": intervention.removed_event_ids.len().saturating_mul(size_of::<Experience>()),
                "decision": action_name(candidate.decision.action),
                "full_evidence_decision": action_name(baseline_action),
                "decision_agrees_with_full": candidate.decision.action == baseline_action,
                "decision_changed": candidate.decision.action != baseline_action,
                "utility": truth_utility(case.truth, candidate.decision.action),
                "regret": current_regret,
                "full_evidence_regret": baseline_regret,
                "regret_gap_vs_full_evidence": current_regret - baseline_regret,
                "decision_is_optimal": candidate.decision.action == Some(case.truth.optimal_action()),
                "abstained": candidate.decision.action.is_none(),
                "evaluation_validity": "diagnostic_only",
                "invalidity_reason": "public synthetic fixture and co-located evaluator"
            }));
        }
    }

    write_jsonl(candidate_path, &candidate_records)?;
    write_jsonl(evaluator_path, &evaluator_records)?;
    eprintln!(
        "diagnostic-only MACE counterfactual audit authored run: worlds={}, candidate_records={}, evaluator_records={}",
        cases.len(),
        candidate_records.len(),
        evaluator_records.len()
    );
    eprintln!("candidate stream: {candidate_path}");
    eprintln!("evaluator stream: {evaluator_path}");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mace_counterfactual_evidence: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mace_generated_stress::generated_worlds;

    fn first_memory() -> Vec<Experience> {
        generated_worlds(1)[0].memory.clone()
    }

    #[test]
    fn duplicate_event_ids_are_rejected_before_intervention_planning() {
        let mut memory = first_memory();
        memory.push(memory[0]);
        assert_eq!(
            EvidenceDependencyIndex::build(&memory).unwrap_err(),
            IndexError::DuplicateEventId(memory[0].event_id)
        );
    }

    #[test]
    fn verified_root_digest_not_legacy_root_id_defines_root_closure() {
        let mut memory = first_memory();
        let shared = SourceRootDigest::from_bytes([0xA5; 32]);
        memory[0].source_root_digest = shared;
        memory[1].source_root_digest = shared;
        memory[0].source_root_id = 100;
        memory[1].source_root_id = 999;

        let index = EvidenceDependencyIndex::build(&memory).unwrap();
        assert_eq!(index.known_root_count(), 3);
        let (_, closure) = index
            .root_groups()
            .find(|(key, _)| key == &format!("root:{}", shared.to_hex()))
            .unwrap();
        assert_eq!(closure, vec![memory[0].event_id, memory[1].event_id]);
    }

    #[test]
    fn unknown_roots_and_families_do_not_inflate_independence_counts() {
        let mut memory = first_memory();
        memory[0].source_root_digest = SourceRootDigest::UNKNOWN;
        memory[1].source_root_digest = SourceRootDigest::UNKNOWN;
        memory[0].source_family = 0;
        memory[1].source_family = 0;

        let index = EvidenceDependencyIndex::build(&memory).unwrap();
        assert_eq!(index.known_root_count(), 2);
        assert_eq!(index.known_family_count(), 2);
        assert_eq!(index.unknown_root_event_count(), 2);
        assert_eq!(index.unknown_family_event_count(), 2);
    }

    #[test]
    fn source_family_knockout_removes_every_event_in_that_family() {
        let memory = first_memory();
        let index = EvidenceDependencyIndex::build(&memory).unwrap();
        let family = memory[0].source_family;
        let group = index
            .family_groups()
            .find(|(key, _)| key == &format!("family:{family}"))
            .unwrap();
        assert!(group.1.contains(&memory[0].event_id));
        let retained = index.retain_without(&group.1);
        assert!(retained.iter().all(|item| item.source_family != family));
    }

    #[test]
    fn source_family_knockout_removes_multiple_alias_events() {
        let mut memory = first_memory();
        let shared_family = memory[0].source_family;
        memory[1].source_family = shared_family;

        let index = EvidenceDependencyIndex::build(&memory).unwrap();
        let group = index
            .family_groups()
            .find(|(key, _)| key == &format!("family:{shared_family}"))
            .unwrap();
        assert!(group.1.contains(&memory[0].event_id));
        assert!(group.1.contains(&memory[1].event_id));

        let retained = index.retain_without(&group.1);
        assert!(retained.iter().all(|item| item.source_family != shared_family));
    }

    #[test]
    fn intervention_manifest_is_deterministic_and_truth_blind_by_construction() {
        let memory = first_memory();
        let first = build_manifest(&memory, 1234, 8).unwrap();
        let second = build_manifest(&memory, 1234, 8).unwrap();
        assert_eq!(first, second);
        // There is no Truth input or generic callback in the manifest builder.
        assert!(first.interventions.iter().any(|item| item.kind == InterventionKind::Event));
        assert!(first.interventions.iter().any(|item| item.kind == InterventionKind::SourceRoot));
        assert!(first.interventions.iter().any(|item| item.kind == InterventionKind::SourceFamily));
    }

    #[test]
    fn matched_random_controls_preserve_per_action_dropout_counts() {
        let memory = first_memory();
        let index = EvidenceDependencyIndex::build(&memory).unwrap();
        let target = vec![memory[0].event_id];
        let random = matched_random_ids(&index, &target, 42).unwrap();
        assert_eq!(random.len(), 1);
        assert!(!random.iter().any(|id| target.contains(id)));
        let target_item = index.records[&target[0]];
        assert!(random.iter().all(|id| {
            let item = index.records[id];
            item.action == target_item.action
                && item.scope.regime == target_item.scope.regime
                && item.scope.objective == target_item.scope.objective
                && item.causal_identification.estimand_digest
                    == target_item.causal_identification.estimand_digest
        }));
    }

    #[test]
    fn evidence_snapshot_digest_changes_when_candidate_evidence_is_removed() {
        let memory = first_memory();
        let index = EvidenceDependencyIndex::build(&memory).unwrap();
        let before = index.baseline_evidence_digest();
        let retained = index.retain_without(&[memory[0].event_id]);
        let after = EvidenceDependencyIndex::build(&retained)
            .unwrap()
            .baseline_evidence_digest();
        assert_ne!(before, after);
    }

    #[test]
    fn candidate_input_digest_binds_need_and_post_intervention_evidence() {
        let case = generated_worlds(1).remove(0);
        let full = EvidenceDependencyIndex::build(&case.memory).unwrap();
        let full_digest = candidate_input_digest(case.decision_need, &full.baseline_evidence_digest());
        let reduced = full.retain_without(&[case.memory[0].event_id]);
        let reduced_index = EvidenceDependencyIndex::build(&reduced).unwrap();
        let reduced_digest = candidate_input_digest(case.decision_need, &reduced_index.baseline_evidence_digest());
        assert_ne!(full_digest, reduced_digest);
    }

    #[test]
    fn no_memory_outside_option_is_not_action_b_payoff() {
        let truth = mace_generated_stress::Truth {
            utility_a: 0.1,
            utility_b: 0.95,
            causal_action: None,
            invalid_a: false,
        };
        assert_eq!(truth_utility(truth, None), 0.0);
        assert_eq!(truth_utility(truth, Some(Action::B)), 0.95);
    }
}
