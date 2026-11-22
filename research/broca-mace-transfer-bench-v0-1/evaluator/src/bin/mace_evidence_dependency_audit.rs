// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Counterfactual evidence-dependency knockouts for MACE.
//!
//! The candidate receives only the post-intervention evidence and DecisionNeed.
//! Intervention labels, removed IDs, and evaluator outcomes are written to a
//! separate evaluator stream. All source roots in the generated worlds are
//! synthetic fixture identities; this binary is diagnostic-only.

#[path = "mace_generated_stress.rs"]
#[allow(dead_code)]
mod stress;

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::mem::size_of;
use std::time::Instant;

use broca_mace_hdc_retrieval::experience_view_digest;
use broca_mace_policy::{decide, is_qualified, Action, Decision, DecisionNeedV1, Experience, Policy, Query};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const SCHEMA_VERSION: &str = "mace_evidence_dependency_audit_v0_1";
const ABSTENTION_UTILITY: f64 = 0.0;
const MAX_PLANS_PER_CASE: usize = 512;
const MAX_ROOT_COALITIONS: usize = 32;

#[derive(Clone, Debug, Eq, PartialEq)]
struct EvidenceNode {
    event_id: u32,
    source_family: u16,
    source_root: [u8; 32],
    action: Action,
    derived_from: Vec<u32>,
    supersedes: Vec<u32>,
    payload_bytes: u64,
}

impl EvidenceNode {
    fn from_experience(item: &Experience) -> Self {
        Self {
            event_id: item.event_id,
            source_family: item.source_family,
            source_root: *item.source_root_digest.as_bytes(),
            action: item.action,
            derived_from: Vec::new(),
            supersedes: Vec::new(),
            // This is an in-memory representation lower bound, not serialized
            // storage accounting. Full receipt/sidecar costs are separate.
            payload_bytes: size_of::<Experience>() as u64,
        }
    }
}

#[derive(Clone, Debug)]
struct EvidenceDependencyIndex {
    nodes: BTreeMap<u32, EvidenceNode>,
    dependents: BTreeMap<u32, BTreeSet<u32>>,
    lifecycle_neighbors: BTreeMap<u32, BTreeSet<u32>>,
}

impl EvidenceDependencyIndex {
    fn from_experiences(memory: &[Experience]) -> Result<Self, String> {
        Self::new(memory.iter().map(EvidenceNode::from_experience).collect())
    }

    fn new(records: Vec<EvidenceNode>) -> Result<Self, String> {
        let mut nodes = BTreeMap::new();
        for record in records {
            let id = record.event_id;
            if nodes.insert(id, record).is_some() {
                return Err(format!("duplicate canonical event identity: {id}"));
            }
        }

        let mut dependents: BTreeMap<u32, BTreeSet<u32>> =
            nodes.keys().map(|id| (*id, BTreeSet::new())).collect();
        let mut lifecycle_neighbors: BTreeMap<u32, BTreeSet<u32>> =
            nodes.keys().map(|id| (*id, BTreeSet::new())).collect();
        let mut cycle_successors: BTreeMap<u32, BTreeSet<u32>> =
            nodes.keys().map(|id| (*id, BTreeSet::new())).collect();
        let mut indegree: BTreeMap<u32, usize> =
            nodes.keys().map(|id| (*id, 0_usize)).collect();

        for node in nodes.values() {
            let mut unique_dependencies = BTreeSet::new();

            // A derived view is invalidated when its source is knocked out.
            // Supersession is intentionally not folded into this edge type:
            // a newer correction can remain valid when an old version is
            // excluded from an ordinary event/source knockout.
            for predecessor in &node.derived_from {
                if *predecessor == node.event_id {
                    return Err(format!("event {} depends on itself", node.event_id));
                }
                if !nodes.contains_key(predecessor) {
                    return Err(format!(
                        "event {} references missing predecessor {predecessor}",
                        node.event_id
                    ));
                }
                if !unique_dependencies.insert(*predecessor) {
                    return Err(format!(
                        "event {} repeats dependency {predecessor}",
                        node.event_id
                    ));
                }
                dependents
                    .get_mut(predecessor)
                    .expect("all nodes have dependent sets")
                    .insert(node.event_id);
                cycle_successors
                    .get_mut(predecessor)
                    .expect("all nodes have cycle-successor sets")
                    .insert(node.event_id);
                *indegree
                    .get_mut(&node.event_id)
                    .expect("all nodes have indegree entries") += 1;
            }

            // Supersession participates in cycle validation and lifecycle
            // closure, but not ordinary derived-view dependency closure.
            for predecessor in &node.supersedes {
                if *predecessor == node.event_id {
                    return Err(format!("event {} supersedes itself", node.event_id));
                }
                if !nodes.contains_key(predecessor) {
                    return Err(format!(
                        "event {} supersedes missing predecessor {predecessor}",
                        node.event_id
                    ));
                }
                if !unique_dependencies.insert(*predecessor) {
                    return Err(format!(
                        "event {} repeats dependency {predecessor}",
                        node.event_id
                    ));
                }
                lifecycle_neighbors
                    .get_mut(&node.event_id)
                    .expect("all nodes have lifecycle sets")
                    .insert(*predecessor);
                lifecycle_neighbors
                    .get_mut(predecessor)
                    .expect("all nodes have lifecycle sets")
                    .insert(node.event_id);
                cycle_successors
                    .get_mut(predecessor)
                    .expect("all nodes have cycle-successor sets")
                    .insert(node.event_id);
                *indegree
                    .get_mut(&node.event_id)
                    .expect("all nodes have indegree entries") += 1;
            }
        }

        // Reject cycles across both derivation and supersession relations.
        // Their closure semantics remain distinct even though both need a
        // valid chronology/ancestry graph.
        let mut ready: BTreeSet<u32> = indegree
            .iter()
            .filter_map(|(id, degree)| (*degree == 0).then_some(*id))
            .collect();
        let mut visited = 0_usize;
        while let Some(id) = ready.pop_first() {
            visited += 1;
            if let Some(children) = cycle_successors.get(&id) {
                for child in children {
                    let degree = indegree
                        .get_mut(child)
                        .expect("all children have indegree entries");
                    *degree -= 1;
                    if *degree == 0 {
                        ready.insert(*child);
                    }
                }
            }
        }
        if visited != nodes.len() {
            return Err("dependency/supersession graph contains a cycle".to_string());
        }

        Ok(Self {
            nodes,
            dependents,
            lifecycle_neighbors,
        })
    }

    fn event_ids(&self) -> BTreeSet<u32> {
        self.nodes.keys().copied().collect()
    }

    fn unique_roots(&self) -> BTreeSet<[u8; 32]> {
        self.nodes
            .values()
            .filter(|node| node.source_root != [0; 32])
            .map(|node| node.source_root)
            .collect()
    }

    fn unique_families(&self) -> BTreeSet<u16> {
        self.nodes
            .values()
            .filter(|node| node.source_family != 0)
            .map(|node| node.source_family)
            .collect()
    }

    fn unknown_root_count(&self) -> usize {
        self.nodes
            .values()
            .filter(|node| node.source_root == [0; 32])
            .count()
    }

    fn dependency_link_count(&self) -> usize {
        self.nodes
            .values()
            .map(|node| node.derived_from.len() + node.supersedes.len())
            .sum()
    }

    fn close_dependents(&self, seeds: impl IntoIterator<Item = u32>) -> BTreeSet<u32> {
        let mut closure = BTreeSet::new();
        let mut queue = VecDeque::new();
        for id in seeds {
            if self.nodes.contains_key(&id) && closure.insert(id) {
                queue.push_back(id);
            }
        }
        while let Some(id) = queue.pop_front() {
            if let Some(children) = self.dependents.get(&id) {
                for child in children {
                    if closure.insert(*child) {
                        queue.push_back(*child);
                    }
                }
            }
        }
        closure
    }

    fn event_closure(&self, event_id: u32) -> Result<BTreeSet<u32>, String> {
        if !self.nodes.contains_key(&event_id) {
            return Err(format!("unknown event identity: {event_id}"));
        }
        Ok(self.close_dependents([event_id]))
    }

    fn source_root_closure(&self, root: [u8; 32]) -> Result<BTreeSet<u32>, String> {
        if root == [0; 32] {
            return Err("unknown/zero source root cannot define an independence group".to_string());
        }
        let seeds: Vec<u32> = self
            .nodes
            .values()
            .filter(|node| node.source_root == root)
            .map(|node| node.event_id)
            .collect();
        if seeds.is_empty() {
            return Err("source root is absent from the dependency index".to_string());
        }
        Ok(self.close_dependents(seeds))
    }

    fn source_family_closure(&self, family: u16) -> Result<BTreeSet<u32>, String> {
        if family == 0 {
            return Err("unknown/zero source family cannot define an independence group".to_string());
        }
        let seeds: Vec<u32> = self
            .nodes
            .values()
            .filter(|node| node.source_family == family)
            .map(|node| node.event_id)
            .collect();
        if seeds.is_empty() {
            return Err("source family is absent from the dependency index".to_string());
        }
        Ok(self.close_dependents(seeds))
    }

    fn lifecycle_closure(&self, event_id: u32) -> Result<BTreeSet<u32>, String> {
        if !self.nodes.contains_key(&event_id) {
            return Err(format!("unknown event identity: {event_id}"));
        }
        let mut lineage = BTreeSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(event_id);
        while let Some(id) = queue.pop_front() {
            if !lineage.insert(id) {
                continue;
            }
            if let Some(neighbors) = self.lifecycle_neighbors.get(&id) {
                queue.extend(neighbors.iter().copied());
            }
        }
        Ok(self.close_dependents(lineage))
    }

    fn coalition_root_closure(
        &self,
        roots: &[[u8; 32]],
    ) -> Result<BTreeSet<u32>, String> {
        if roots.len() < 2 {
            return Err("root coalition requires at least two roots".to_string());
        }
        let mut seeds = BTreeSet::new();
        for root in roots {
            seeds.extend(self.source_root_closure(*root)?);
        }
        Ok(self.close_dependents(seeds))
    }

    fn counts_for_ids(&self, ids: &BTreeSet<u32>) -> (usize, usize, usize) {
        let mut roots = BTreeSet::new();
        let mut families = BTreeSet::new();
        let mut unknown_roots = 0_usize;
        for id in ids {
            if let Some(node) = self.nodes.get(id) {
                if node.source_root == [0; 32] {
                    unknown_roots += 1;
                } else {
                    roots.insert(node.source_root);
                }
                if node.source_family != 0 {
                    families.insert(node.source_family);
                }
            }
        }
        (roots.len(), families.len(), unknown_roots)
    }
}

#[derive(Clone, Debug)]
struct InterventionPlan {
    plan_id: String,
    kind: String,
    target: String,
    removed_ids: BTreeSet<u32>,
    paired_plan_id: Option<String>,
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn root_hex(root: &[u8; 32]) -> String {
    hex(root)
}

fn action_name(action: Option<Action>) -> &'static str {
    match action {
        Some(Action::A) => "action_a",
        Some(Action::B) => "action_b",
        None => "abstain",
    }
}

fn action_key(action: Action) -> &'static str {
    match action {
        Action::A => "A",
        Action::B => "B",
    }
}

fn make_plan(kind: &str, target: &str, removed_ids: BTreeSet<u32>) -> InterventionPlan {
    let mut canonical = Vec::new();
    canonical.extend_from_slice(b"luminous:mace:dependency-knockout-plan:v1\0");
    canonical.extend_from_slice(kind.as_bytes());
    canonical.push(0);
    canonical.extend_from_slice(target.as_bytes());
    canonical.push(0);
    for id in &removed_ids {
        canonical.extend_from_slice(&id.to_be_bytes());
    }
    let plan_id = hex(&sha256(&canonical));
    InterventionPlan {
        plan_id,
        kind: kind.to_string(),
        target: target.to_string(),
        removed_ids,
        paired_plan_id: None,
    }
}

fn selection_key(seed: u64, plan_id: &str, event_id: u32) -> [u8; 32] {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"luminous:mace:matched-random-knockout:v1\0");
    bytes.extend_from_slice(&seed.to_be_bytes());
    bytes.extend_from_slice(plan_id.as_bytes());
    bytes.extend_from_slice(&event_id.to_be_bytes());
    sha256(&bytes)
}

/// Match a dependency knockout by the exact removed (action, in-memory-byte)
/// histogram. The matched control is selected from records outside the target
/// closure with a deterministic hash ordering. It is a count/byte/action
/// control, not a claim that the records are semantically exchangeable.
fn matched_random_control(
    index: &EvidenceDependencyIndex,
    removed_ids: &BTreeSet<u32>,
    seed: u64,
    plan_id: &str,
) -> Option<BTreeSet<u32>> {
    let mut target_histogram: BTreeMap<(&'static str, u64), usize> = BTreeMap::new();
    for id in removed_ids {
        let node = index.nodes.get(id)?;
        *target_histogram
            .entry((action_key(node.action), node.payload_bytes))
            .or_default() += 1;
    }
    let (target_root_count, target_family_count, target_unknown_root_count) =
        index.counts_for_ids(removed_ids);
    let excluded_roots: BTreeSet<[u8; 32]> = removed_ids
        .iter()
        .filter_map(|id| index.nodes.get(id).map(|node| node.source_root))
        .filter(|root| *root != [0; 32])
        .collect();
    let excluded_families: BTreeSet<u16> = removed_ids
        .iter()
        .filter_map(|id| index.nodes.get(id).map(|node| node.source_family))
        .filter(|family| *family != 0)
        .collect();
    let mut selected = BTreeSet::new();
    for ((action, byte_size), count) in target_histogram {
        let mut candidates: Vec<&EvidenceNode> = index
            .nodes
            .values()
            .filter(|node| {
                !removed_ids.contains(&node.event_id)
                    && (node.source_root == [0; 32] || !excluded_roots.contains(&node.source_root))
                    && (node.source_family == 0 || !excluded_families.contains(&node.source_family))
                    && action_key(node.action) == action
                    && node.payload_bytes == byte_size
            })
            .collect();
        candidates.sort_by_key(|node| selection_key(seed, plan_id, node.event_id));
        if candidates.len() < count {
            return None;
        }
        selected.extend(
            candidates
                .into_iter()
                .take(count)
                .map(|node| node.event_id),
        );
    }
    let (control_root_count, control_family_count, control_unknown_root_count) =
        index.counts_for_ids(&selected);
    if control_root_count != target_root_count
        || control_family_count != target_family_count
        || control_unknown_root_count != target_unknown_root_count
    {
        return None;
    }
    Some(selected)
}

fn build_plans(
    index: &EvidenceDependencyIndex,
    seed: u64,
) -> Result<Vec<InterventionPlan>, String> {
    let mut interventions = Vec::new();
    interventions.push(make_plan("full_evidence_reference", "all", BTreeSet::new()));

    for event_id in index.event_ids() {
        let closure = index.event_closure(event_id)?;
        interventions.push(make_plan("event_closure", &format!("event:{event_id}"), closure));
        let lifecycle = index.lifecycle_closure(event_id)?;
        if lifecycle.len() > index.event_closure(event_id)?.len() {
            interventions.push(make_plan(
                "lifecycle_closure",
                &format!("lifecycle:{event_id}"),
                lifecycle,
            ));
        }
    }

    for root in index.unique_roots() {
        interventions.push(make_plan(
            "source_root_closure",
            &format!("root:{}", root_hex(&root)),
            index.source_root_closure(root)?,
        ));
    }
    for family in index.unique_families() {
        interventions.push(make_plan(
            "source_family_closure",
            &format!("family:{family}"),
            index.source_family_closure(family)?,
        ));
    }

    let roots: Vec<[u8; 32]> = index.unique_roots().into_iter().collect();
    let mut coalition_count = 0_usize;
    'coalitions: for left in 0..roots.len() {
        for right in (left + 1)..roots.len() {
            if coalition_count >= MAX_ROOT_COALITIONS {
                break 'coalitions;
            }
            let pair = [roots[left], roots[right]];
            let removed = index.coalition_root_closure(&pair)?;
            interventions.push(make_plan(
                "two_root_coalition",
                &format!("roots:{}+{}", root_hex(&pair[0]), root_hex(&pair[1])),
                removed,
            ));
            coalition_count += 1;
        }
    }

    if interventions.len() > MAX_PLANS_PER_CASE {
        return Err(format!(
            "intervention plan bound exceeded: {} > {}",
            interventions.len(),
            MAX_PLANS_PER_CASE
        ));
    }

    let mut plans = interventions.clone();
    for intervention in interventions.iter().skip(1) {
        if let Some(control_ids) = matched_random_control(
            index,
            &intervention.removed_ids,
            seed,
            &intervention.plan_id,
        ) {
            let mut control = make_plan(
                "matched_random_control",
                &format!("matched:{}", intervention.plan_id),
                control_ids,
            );
            control.paired_plan_id = Some(intervention.plan_id.clone());
            plans.push(control);
        }
    }
    if plans.len() > MAX_PLANS_PER_CASE * 2 {
        return Err("matched-control plan bound exceeded".to_string());
    }
    Ok(plans)
}

fn candidate_evidence_digest(memory: &[Experience], need: DecisionNeedV1) -> String {
    let mut ordered = memory.to_vec();
    ordered.sort_by_key(|item| item.event_id);
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:dependency-audit-candidate-input:v1\0");
    hasher.update(need.canonical_digest().as_bytes());
    for item in &ordered {
        hasher.update(experience_view_digest(item));
    }
    hex(&hasher.finalize())
}

fn decision_trace_digest(
    input_digest: &str,
    decision: &Decision,
) -> String {
    let mut canonical = Vec::new();
    canonical.extend_from_slice(b"luminous:mace:dependency-audit-decision:v1\0");
    canonical.extend_from_slice(input_digest.as_bytes());
    canonical.push(0);
    canonical.extend_from_slice(action_name(decision.action).as_bytes());
    for event_id in &decision.evidence_ids {
        canonical.extend_from_slice(&event_id.to_be_bytes());
    }
    hex(&sha256(&canonical))
}

fn action_utility(truth: stress::Truth, action: Option<Action>) -> f64 {
    match action {
        Some(selected) => truth.utility(selected),
        None => ABSTENTION_UTILITY,
    }
}

#[derive(Clone, Debug)]
struct EvaluationMetrics {
    plan_id: String,
    kind: String,
    target: String,
    paired_plan_id: Option<String>,
    candidate_input_sha256: String,
    removed_count: usize,
    removed_roots: usize,
    removed_families: usize,
    decision: Option<Action>,
    utility: f64,
    regret: f64,
    added_regret_vs_full: f64,
    changed_vs_full: bool,
    invalid_evidence_use_count: usize,
    candidate_memory_count: usize,
    candidate_root_count: usize,
    candidate_family_count: usize,
    unknown_root_count: usize,
    lineage_edge_count: usize,
}

struct EvaluatedPlan {
    plan: InterventionPlan,
    candidate_record: Value,
    metrics: EvaluationMetrics,
}

fn evaluate_plan(
    case: &stress::Case,
    plan: &InterventionPlan,
    full_decision: &Decision,
    full_regret: f64,
) -> EvaluatedPlan {
    let candidate_memory: Vec<Experience> = case
        .memory
        .iter()
        .filter(|item| !plan.removed_ids.contains(&item.event_id))
        .copied()
        .collect();
    let input_digest = candidate_evidence_digest(&candidate_memory, case.decision_need);
    let started = Instant::now();
    let decision = decide(Policy::CausalQualified, &candidate_memory, case.query);
    let latency_ns = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    let trace_digest = decision_trace_digest(&input_digest, &decision);

    let candidate_ids: BTreeSet<u32> = candidate_memory.iter().map(|item| item.event_id).collect();
    let candidate_record = json!({
        "schema_version": SCHEMA_VERSION,
        "record_type": "candidate_decision",
        // Candidate-side identifiers bind only the visible input. They do not
        // reveal the evaluator's intervention kind/target or scenario-family label.
        "record_id": input_digest.clone(),
        "candidate_input_sha256": input_digest.clone(),
        "decision_need_sha256": case.decision_need.canonical_digest().to_hex(),
        "memory_snapshot_sha256": input_digest.clone(),
        "decision": action_name(decision.action),
        "decision_support_event_ids": decision.evidence_ids.clone(),
        "candidate_memory_event_count": candidate_ids.len(),
        "decision_trace_sha256": trace_digest,
        "latency_ns": latency_ns
    });

    let decision_utility = action_utility(case.truth, decision.action);
    let regret = (case.truth.optimal_utility() - decision_utility).max(0.0);
    let invalid_evidence_use_count = decision
        .evidence_ids
        .iter()
        .filter(|id| {
            candidate_memory
                .iter()
                .find(|item| item.event_id == **id)
                .is_none_or(|item| !is_qualified(item, case.query))
        })
        .count();

    let index = EvidenceDependencyIndex::from_experiences(&candidate_memory);
    let (candidate_root_count, candidate_family_count, unknown_root_count) = match index {
        Ok(index) => {
            let ids = index.event_ids();
            index.counts_for_ids(&ids)
        }
        Err(_) => (0, 0, candidate_memory.len()),
    };

    let original_index = EvidenceDependencyIndex::from_experiences(&case.memory)
        .expect("plans are only evaluated for a valid canonical event index");
    let (removed_roots, removed_families, _) =
        original_index.counts_for_ids(&plan.removed_ids);

    let metrics = EvaluationMetrics {
        plan_id: plan.plan_id.clone(),
        kind: plan.kind.clone(),
        target: plan.target.clone(),
        paired_plan_id: plan.paired_plan_id.clone(),
        candidate_input_sha256: input_digest.clone(),
        removed_count: plan.removed_ids.len(),
        removed_roots,
        removed_families,
        decision: decision.action,
        utility: decision_utility,
        regret,
        added_regret_vs_full: regret - full_regret,
        changed_vs_full: decision.action != full_decision.action,
        invalid_evidence_use_count,
        candidate_memory_count: candidate_ids.len(),
        candidate_root_count,
        candidate_family_count,
        unknown_root_count,
        lineage_edge_count: original_index.dependency_link_count(),
    };
    EvaluatedPlan {
        plan: plan.clone(),
        candidate_record,
        metrics,
    }
}

fn evaluator_record(
    case: &stress::Case,
    result: &EvaluatedPlan,
    matched: Option<&EvaluationMetrics>,
) -> Value {
    let metrics = &result.metrics;
    let mut value = json!({
        "schema_version": SCHEMA_VERSION,
        "record_type": "evaluator_score",
        "record_id": metrics.plan_id,
        "candidate_input_sha256": metrics.candidate_input_sha256,
        "world_cluster_id": case.cluster.to_string(),
        "scenario_family": case.family,
        "intervention_kind": metrics.kind,
        "intervention_target": metrics.target,
        "removed_event_ids": result.plan.removed_ids,
        "removed_event_count": metrics.removed_count,
        "removed_unique_root_count": metrics.removed_roots,
        "removed_unique_family_count": metrics.removed_families,
        "decision": action_name(metrics.decision),
        "utility": metrics.utility,
        "regret": metrics.regret,
        "added_regret_vs_full_evidence": metrics.added_regret_vs_full,
        "decision_changed_vs_full_evidence": metrics.changed_vs_full,
        "invalid_evidence_use_count": metrics.invalid_evidence_use_count,
        "remaining_event_count": metrics.candidate_memory_count,
        "remaining_unique_root_count": metrics.candidate_root_count,
        "remaining_unique_family_count": metrics.candidate_family_count,
        "remaining_unknown_root_event_count": metrics.unknown_root_count,
        "fixture_dependency_link_count": metrics.lineage_edge_count
    });
    if let Some(parent) = &metrics.paired_plan_id {
        value["matched_to_plan_id"] = json!(parent);
    }
    if let Some(control) = matched {
        value["matched_control"] = json!({
            "record_id": control.plan_id,
            "utility": control.utility,
            "regret": control.regret,
            "added_regret_vs_full_evidence": control.added_regret_vs_full,
            "decision_changed_vs_full_evidence": control.changed_vs_full,
            "excess_regret_over_matched_random": metrics.regret - control.regret
        });
    }
    value
}

fn write_jsonl(path: &str, rows: &[Value]) -> Result<(), String> {
    let file = File::create(path).map_err(|error| format!("create {path}: {error}"))?;
    let mut writer = BufWriter::new(file);
    for row in rows {
        serde_json::to_writer(&mut writer, row)
            .map_err(|error| format!("serialize JSONL row for {path}: {error}"))?;
        writer
            .write_all(b"\n")
            .map_err(|error| format!("write JSONL row for {path}: {error}"))?;
    }
    writer
        .flush()
        .map_err(|error| format!("flush {path}: {error}"))
}

fn source_revision() -> Result<String, String> {
    let revision = env::var("MACE_SOURCE_REVISION")
        .map_err(|_| "MACE_SOURCE_REVISION must be set to the exact 40- or 64-character Git SHA")?;
    if !matches!(revision.len(), 40 | 64)
        || !revision.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("MACE_SOURCE_REVISION is not a full hexadecimal Git SHA".to_string());
    }
    Ok(revision.to_ascii_lowercase())
}

#[derive(Default)]
struct Aggregate {
    rows: usize,
    decision_changes: usize,
    invalid_evidence_uses: usize,
    added_regret_sum: f64,
    matched_pairs: usize,
    excess_regret_sum: f64,
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
        .unwrap_or("mace_dependency_candidate_decisions.jsonl");
    let evaluator_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("mace_dependency_evaluator_scores.jsonl");
    let summary_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("mace_dependency_summary.jsonl");
    let revision = source_revision()?;
    let run_id = format!("mace-dependency-audit-{}-{clusters}", &revision[..12]);
    let cases = stress::generated_worlds(clusters);

    let mut candidate_records = Vec::new();
    let mut evaluator_records = Vec::new();
    let mut totals: BTreeMap<String, Aggregate> = BTreeMap::new();
    let mut rejected_blocks = 0_usize;
    let mut planned_rows = 0_usize;
    let mut matched_rows = 0_usize;

    candidate_records.push(json!({
        "schema_version": SCHEMA_VERSION,
        "record_type": "run_manifest",
        "run_id": run_id,
        "source_revision": revision,
        "cluster_count": clusters,
        "scenario_count": cases.len(),
        "candidate_policy": "causal_qualified_v1",
        "candidate_stream_contract": "no_intervention_or_scenario_family_labels; record_id is candidate-input digest",
        "dependency_lineage_contract": "synthetic Experience projection has root/family identity but no derived_from or supersedes edges",
        "candidate_stream_contains_evaluator_labels": false,
        "evaluator_custody": "public_co_located_diagnostic_only",
        "root_identity": "synthetic_fixture_digest_not_cryptographic_attestation",
        "payload_bytes": "in_memory_Experience_size_lower_bound"
    }));

    for case in &cases {
        let index = match EvidenceDependencyIndex::from_experiences(&case.memory) {
            Ok(index) => index,
            Err(reason) => {
                rejected_blocks += 1;
                evaluator_records.push(json!({
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "rejected_block",
                    "world_cluster_id": case.cluster.to_string(),
                    "scenario_family": case.family,
                    "rejection_reason": reason
                }));
                continue;
            }
        };
        let plans = build_plans(&index, case.cluster ^ 0x4D41_4345_4B4E_4F43)?;
        let full_decision = decide(Policy::CausalQualified, &case.memory, case.query);
        let full_utility = action_utility(case.truth, full_decision.action);
        let full_regret = (case.truth.optimal_utility() - full_utility).max(0.0);

        let mut evaluated = Vec::with_capacity(plans.len());
        for plan in &plans {
            evaluated.push(evaluate_plan(case, plan, &full_decision, full_regret));
        }

        let matched_by_parent: BTreeMap<String, &EvaluationMetrics> = evaluated
            .iter()
            .filter_map(|item| {
                item.metrics
                    .paired_plan_id
                    .as_ref()
                    .map(|parent| (parent.clone(), &item.metrics))
            })
            .collect();

        for item in &evaluated {
            planned_rows += 1;
            if item.metrics.kind == "matched_random_control" {
                matched_rows += 1;
            }
            candidate_records.push({
                let mut row = item.candidate_record.clone();
                row["run_id"] = json!(run_id);
                row["source_revision"] = json!(revision);
                row
            });

            let matched = matched_by_parent.get(&item.metrics.plan_id).copied();
            evaluator_records.push(evaluator_record(case, item, matched));

            let aggregate = totals.entry(item.metrics.kind.clone()).or_default();
            aggregate.rows += 1;
            aggregate.decision_changes += if item.metrics.changed_vs_full { 1 } else { 0 };
            aggregate.invalid_evidence_uses += item.metrics.invalid_evidence_use_count;
            aggregate.added_regret_sum += item.metrics.added_regret_vs_full;
            if let Some(control) = matched {
                aggregate.matched_pairs += 1;
                aggregate.excess_regret_sum += item.metrics.regret - control.regret;
            }
        }
    }

    let aggregate_json: Vec<Value> = totals
        .into_iter()
        .map(|(kind, aggregate)| {
            json!({
                "intervention_kind": kind,
                "rows": aggregate.rows,
                "decision_changes_vs_full": aggregate.decision_changes,
                "invalid_evidence_uses": aggregate.invalid_evidence_uses,
                "mean_added_regret_vs_full": if aggregate.rows == 0 { 0.0 } else { aggregate.added_regret_sum / aggregate.rows as f64 },
                "matched_pairs": aggregate.matched_pairs,
                "mean_excess_regret_over_matched_random": if aggregate.matched_pairs == 0 { Value::Null } else { json!(aggregate.excess_regret_sum / aggregate.matched_pairs as f64) }
            })
        })
        .collect();
    let summary = json!({
        "schema_version": SCHEMA_VERSION,
        "record_type": "run_summary",
        "run_id": run_id,
        "source_revision": revision,
        "scenario_count": cases.len(),
        "rejected_duplicate_or_invalid_identity_blocks": rejected_blocks,
        "candidate_decision_rows": planned_rows,
        "matched_random_control_rows": matched_rows,
        "aggregation": "descriptive_only_no_confirmatory_confidence_interval",
        "by_intervention_kind": aggregate_json,
        "claim_ceiling": [
            "synthetic dependency-closure and policy sensitivity diagnostic",
            "fixture root digests are not independently attested provenance",
            "the current Experience projection supplies no derived-view or supersession edges; graph closures are tested with explicit synthetic edge fixtures",
            "no production lifecycle or derived-view lineage is inferred from Experience fields",
            "public co-located evaluator is not independent custody",
            "no Rust execution result is implied by the presence of authored tests"
        ]
    });

    write_jsonl(candidate_path, &candidate_records)?;
    write_jsonl(evaluator_path, &evaluator_records)?;
    write_jsonl(summary_path, &[summary])?;
    eprintln!(
        "diagnostic-only MACE dependency audit authored run: scenarios={}, candidate_rows={}, matched_controls={}, rejected_blocks={}",
        cases.len(),
        planned_rows,
        matched_rows,
        rejected_blocks
    );
    eprintln!("candidate stream: {candidate_path}");
    eprintln!("evaluator stream: {evaluator_path}");
    eprintln!("summary stream: {summary_path}");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mace_evidence_dependency_audit: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root(seed: u8) -> [u8; 32] {
        let mut value = [0_u8; 32];
        value[0] = seed;
        value
    }

    fn node(
        event_id: u32,
        family: u16,
        root_digest: [u8; 32],
        action: Action,
    ) -> EvidenceNode {
        EvidenceNode {
            event_id,
            source_family: family,
            source_root: root_digest,
            action,
            derived_from: Vec::new(),
            supersedes: Vec::new(),
            payload_bytes: 64,
        }
    }

    #[test]
    fn event_knockout_removes_transitive_derived_views() {
        let mut child = node(2, 2, root(2), Action::A);
        child.derived_from.push(1);
        let mut grandchild = node(3, 3, root(3), Action::A);
        grandchild.derived_from.push(2);
        let index = EvidenceDependencyIndex::new(vec![
            node(1, 1, root(1), Action::A),
            child,
            grandchild,
        ])
        .unwrap();
        assert_eq!(index.event_closure(1).unwrap(), BTreeSet::from([1, 2, 3]));
        assert_eq!(index.event_closure(2).unwrap(), BTreeSet::from([2, 3]));
        assert_eq!(index.event_closure(3).unwrap(), BTreeSet::from([3]));
    }

    #[test]
    fn root_closure_unifies_different_family_aliases() {
        let shared = root(9);
        let index = EvidenceDependencyIndex::new(vec![
            node(1, 11, shared, Action::A),
            node(2, 12, shared, Action::A),
            node(3, 13, root(10), Action::B),
        ])
        .unwrap();
        assert_eq!(
            index.source_root_closure(shared).unwrap(),
            BTreeSet::from([1, 2])
        );
        assert_eq!(index.unique_roots().len(), 2);
    }

    #[test]
    fn family_closure_removes_all_family_members_and_descendants() {
        let mut dependent = node(3, 30, root(3), Action::B);
        dependent.derived_from.push(1);
        let index = EvidenceDependencyIndex::new(vec![
            node(1, 7, root(1), Action::A),
            node(2, 7, root(2), Action::A),
            dependent,
            node(4, 8, root(4), Action::B),
        ])
        .unwrap();
        assert_eq!(
            index.source_family_closure(7).unwrap(),
            BTreeSet::from([1, 2, 3])
        );
    }

    #[test]
    fn lifecycle_closure_follows_supersession_in_both_directions() {
        let mut corrected = node(2, 2, root(2), Action::A);
        corrected.supersedes.push(1);
        let mut derived = node(3, 3, root(3), Action::A);
        derived.derived_from.push(2);
        let index = EvidenceDependencyIndex::new(vec![
            node(1, 1, root(1), Action::A),
            corrected,
            derived,
            node(4, 4, root(4), Action::B),
        ])
        .unwrap();
        assert_eq!(index.event_closure(1).unwrap(), BTreeSet::from([1]));
        assert_eq!(
            index.lifecycle_closure(1).unwrap(),
            BTreeSet::from([1, 2, 3])
        );
        assert_eq!(
            index.lifecycle_closure(2).unwrap(),
            BTreeSet::from([1, 2, 3])
        );
    }

    #[test]
    fn event_knockout_preserves_superseding_replacement() {
        let mut replacement = node(2, 2, root(2), Action::A);
        replacement.supersedes.push(1);
        let index = EvidenceDependencyIndex::new(vec![
            node(1, 1, root(1), Action::A),
            replacement,
        ])
        .unwrap();
        assert_eq!(index.event_closure(1).unwrap(), BTreeSet::from([1]));
        assert_eq!(
            index.lifecycle_closure(1).unwrap(),
            BTreeSet::from([1, 2])
        );
    }

    #[test]
    fn coalition_closure_unions_root_groups_and_dependents() {
        let mut child = node(3, 3, root(3), Action::B);
        child.derived_from.push(1);
        let index = EvidenceDependencyIndex::new(vec![
            node(1, 1, root(1), Action::A),
            node(2, 2, root(2), Action::B),
            child,
        ])
        .unwrap();
        assert_eq!(
            index.coalition_root_closure(&[root(1), root(2)]).unwrap(),
            BTreeSet::from([1, 2, 3])
        );
        assert!(index.coalition_root_closure(&[root(1)]).is_err());
    }

    #[test]
    fn unknown_root_cannot_be_used_as_an_independence_group() {
        let index = EvidenceDependencyIndex::new(vec![
            node(1, 1, [0; 32], Action::A),
            node(2, 2, root(2), Action::B),
        ])
        .unwrap();
        assert!(index.source_root_closure([0; 32]).is_err());
        assert_eq!(index.unknown_root_count(), 1);
        assert_eq!(index.unique_roots().len(), 1);
    }

    #[test]
    fn experience_projection_does_not_invent_dependency_lineage() {
        let cases = stress::generated_worlds(1);
        let case = cases.iter().find(|case| !case.memory.is_empty()).unwrap();
        let index = EvidenceDependencyIndex::from_experiences(&case.memory).unwrap();
        assert_eq!(index.dependency_link_count(), 0);
        assert_eq!(index.event_closure(case.memory[0].event_id).unwrap(),
                   BTreeSet::from([case.memory[0].event_id]));
    }

    #[test]
    fn duplicate_event_ids_are_rejected_before_planning() {
        let duplicate = vec![
            node(1, 1, root(1), Action::A),
            node(1, 2, root(2), Action::A),
        ];
        assert!(EvidenceDependencyIndex::new(duplicate)
            .unwrap_err()
            .contains("duplicate"));
    }

    #[test]
    fn missing_predecessor_is_rejected() {
        let mut child = node(2, 2, root(2), Action::A);
        child.derived_from.push(99);
        assert!(EvidenceDependencyIndex::new(vec![child])
            .unwrap_err()
            .contains("missing predecessor"));
    }

    #[test]
    fn dependency_cycles_are_rejected() {
        let mut first = node(1, 1, root(1), Action::A);
        first.derived_from.push(2);
        let mut second = node(2, 2, root(2), Action::B);
        second.derived_from.push(1);
        assert!(EvidenceDependencyIndex::new(vec![first, second])
            .unwrap_err()
            .contains("cycle"));
    }

    #[test]
    fn matched_random_control_preserves_action_and_byte_histogram() {
        let index = EvidenceDependencyIndex::new(vec![
            node(1, 1, root(1), Action::A),
            node(2, 2, root(2), Action::A),
            node(3, 3, root(3), Action::A),
            node(4, 4, root(4), Action::B),
            node(5, 5, root(5), Action::B),
            node(6, 6, root(6), Action::B),
        ])
        .unwrap();
        let removed = BTreeSet::from([1, 4]);
        let first = matched_random_control(&index, &removed, 44, "plan");
        let second = matched_random_control(&index, &removed, 44, "plan");
        assert_eq!(first, second);
        let control = first.unwrap();
        assert_eq!(control.len(), 2);
        assert!(!control.contains(&1));
        assert!(!control.contains(&4));
        let actions: Vec<Action> = control
            .iter()
            .map(|id| index.nodes.get(id).unwrap().action)
            .collect();
        assert_eq!(actions.iter().filter(|a| **a == Action::A).count(), 1);
        assert_eq!(actions.iter().filter(|a| **a == Action::B).count(), 1);
        let (target_roots, target_families, _) = index.counts_for_ids(&removed);
        let (control_roots, control_families, _) = index.counts_for_ids(&control);
        assert_eq!(control_roots, target_roots);
        assert_eq!(control_families, target_families);
        assert_eq!(index.counts_for_ids(&control).2, index.counts_for_ids(&removed).2);
        for id in &control {
            assert_ne!(index.nodes.get(id).unwrap().source_root, root(1));
            assert_ne!(index.nodes.get(id).unwrap().source_root, root(4));
        }
    }

    #[test]
    fn matched_random_control_fails_closed_when_no_equivalent_records_exist() {
        let index = EvidenceDependencyIndex::new(vec![
            node(1, 1, root(1), Action::A),
            node(2, 2, root(2), Action::B),
        ])
        .unwrap();
        assert!(matched_random_control(
            &index,
            &BTreeSet::from([1]),
            1,
            "unmatchable"
        )
        .is_none());
    }

    #[test]
    fn candidate_input_digest_changes_when_visible_evidence_is_removed() {
        let cases = stress::generated_worlds(1);
        let case = cases
            .iter()
            .find(|case| case.memory.len() >= 2)
            .unwrap();
        let full = candidate_evidence_digest(&case.memory, case.decision_need);
        let reduced: Vec<Experience> = case.memory.iter().skip(1).copied().collect();
        let changed = candidate_evidence_digest(&reduced, case.decision_need);
        assert_ne!(full, changed);
    }

    #[test]
    fn hidden_outcome_mutation_does_not_change_candidate_input_digest() {
        let cases = stress::generated_worlds(1);
        let original = &cases[0];
        let mut changed = original.clone();
        changed.truth.utility_a = 0.001;
        changed.truth.utility_b = 0.999;
        changed.truth.causal_action = Some(Action::B);
        let before = candidate_evidence_digest(&original.memory, original.decision_need);
        changed.cluster = 99_999;
        changed.family = "hidden-family-label-mutated";
        let after = candidate_evidence_digest(&changed.memory, changed.decision_need);
        assert_eq!(before, after);
    }

    #[test]
    fn candidate_record_contains_no_evaluator_outcome_fields() {
        let cases = stress::generated_worlds(1);
        let case = cases
            .iter()
            .find(|case| {
                EvidenceDependencyIndex::from_experiences(&case.memory).is_ok()
            })
            .unwrap();
        let index = EvidenceDependencyIndex::from_experiences(&case.memory).unwrap();
        let plan = build_plans(&index, 7).unwrap().remove(0);
        let decision = decide(Policy::CausalQualified, &case.memory, case.query);
        let regret = (case.truth.optimal_utility() - action_utility(case.truth, decision.action))
            .max(0.0);
        let evaluated = evaluate_plan(case, &plan, &decision, regret);
        let serialized = evaluated.candidate_record.to_string();
        for forbidden in [
            "utility",
            "regret",
            "optimal_action",
            "causal_action",
            "invalid_a",
            "truth",
            "intervention_kind",
            "removed_event_ids",
        ] {
            assert!(
                !serialized.contains(forbidden),
                "candidate record leaked forbidden field marker: {forbidden}"
            );
        }
    }

    #[test]
    fn intervention_plan_ids_are_deterministic_and_bind_the_removed_set() {
        let first = make_plan("source_root_closure", "root-x", BTreeSet::from([1, 2]));
        let same = make_plan("source_root_closure", "root-x", BTreeSet::from([2, 1]));
        let changed = make_plan("source_root_closure", "root-x", BTreeSet::from([1, 3]));
        assert_eq!(first.plan_id, same.plan_id);
        assert_ne!(first.plan_id, changed.plan_id);
    }
}
