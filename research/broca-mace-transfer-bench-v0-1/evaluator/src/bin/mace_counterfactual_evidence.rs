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

use broca_mace_ingress::EvidenceDependencySidecarV1;
use broca_mace_ingress::EvidenceDependencySidecarV1;
use broca_mace_hdc_retrieval::{
    experience_view_digest,
    typed_content::{SemanticExperienceQueryV1, TypedContentConfigV1, TypedContentIndexV1},
    CandidateIndex, IndexConfig, RepresentationProfile,
};
use broca_mace_policy::{
    decide, Action, Decision, DecisionNeedV1, Experience, Objective, Policy, Query, Regime, Scope, SourceRootDigest,
};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

mod mace_generated_stress;
mod receipt_dependency;

use receipt_dependency::ReceiptDependencyProjection;
mod receipt_dependency;

use receipt_dependency::ReceiptDependencyProjection;

const SCHEMA_VERSION: &str = "mace_counterfactual_evidence_v0_1";
const ABSTENTION_UTILITY: f64 = 0.0;
const TYPED_HDC_DIMENSION: usize = 2_048;
const TYPED_HDC_SEED: u64 = 0x4D41_4345_4844_4301;
const TYPED_HDC_TOP_K: usize = 2;
const EVENT_ID_HDC_DIMENSION: usize = 2_048;
const EVENT_ID_HDC_SEED: u64 = 0x4D41_4345_4844_4301;
const EVENT_ID_LTC_SEED: u64 = 0x4D41_4345_4C54_4301;
const BASELINE_TOP_K: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum InterventionKind {
    Event,
    SourceRoot,
    SourceFamily,
    UnknownRoot,
    UnknownFamily,
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
    max_root_pairs: usize,
    root_pair_count: usize,
    matched_random_count: usize,
    unmatched_random_target_count: usize,
    dependency_sidecar_sha256: Option<String>,
    lineage_incomplete_event_count: usize,
    unresolved_parent_reference_count: usize,
    manifest_sha256: String,
}

#[derive(Clone, Debug)]
struct EvidenceDependencyIndex {
    records: BTreeMap<u32, Experience>,
    roots: BTreeMap<SourceRootDigest, BTreeSet<u32>>,
    families: BTreeMap<u16, BTreeSet<u32>>,
    unknown_root_ids: BTreeSet<u32>,
    unknown_family_ids: BTreeSet<u32>,
    receipt_dependencies: Option<ReceiptDependencyProjection>,
    lineage_incomplete_event_count: usize,
    unresolved_parent_reference_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum IndexError {
    DuplicateEventId(u32),
    InvalidDecisionNeed,
    TypedIndexFailure(String),
    UnresolvedCandidateId(u32),
    CanonicalViewMismatch(u32),
    InvalidDependencySidecar(String),
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

        let lineage_incomplete_event_count = records.len();
        Ok(Self {
            records,
            roots,
            families,
            unknown_root_ids,
            unknown_family_ids,
            receipt_dependencies: None,
            lineage_incomplete_event_count,
            unresolved_parent_reference_count: 0,
        })
    }

    fn build_with_sidecar(
        memory: &[Experience],
        sidecar: &EvidenceDependencySidecarV1,
    ) -> Result<Self, IndexError> {
        let mut index = Self::build(memory)?;
        let projection = ReceiptDependencyProjection::build(memory, sidecar)
            .map_err(|error| IndexError::InvalidDependencySidecar(format!("{error:?}")))?;
        index.lineage_incomplete_event_count = projection.lineage_incomplete_event_count();
        index.unresolved_parent_reference_count =
            projection.unresolved_parent_reference_count();
        index.receipt_dependencies = Some(projection);
        Ok(index)
    }

    fn dependent_closure(&self, seeds: &[u32]) -> Vec<u32> {
        match &self.receipt_dependencies {
            Some(projection) => projection.dependent_closure(seeds.iter().copied()),
            None => normalized_ids(seeds.to_vec()),
        }
    }

    fn dependency_sidecar_sha256(&self) -> Option<&str> {
        self.receipt_dependencies
            .as_ref()
            .map(ReceiptDependencyProjection::sidecar_sha256)
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
        match self.dependency_sidecar_sha256() {
            Some(digest) => {
                hasher.update(b"receipt-sidecar:verified:v1\\0");
                hasher.update(digest.as_bytes());
            }
            None => {
                // No sidecar means ancestry is unknown, never independent.
                hasher.update(b"receipt-sidecar:unavailable:lineage-unknown:v1\\0");
                hasher.update(
                    (self.lineage_incomplete_event_count as u64).to_be_bytes(),
                );
            }
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
fn evidence_strata(
    index: &EvidenceDependencyIndex,
    ids: &[u32],
) -> Option<BTreeMap<(Regime, Objective, Action, String), usize>> {
    let mut counts = BTreeMap::new();
    for event_id in ids {
        let item = index.records.get(event_id)?;
        let key = (
            item.scope.regime,
            item.scope.objective,
            item.action,
            item.causal_identification.estimand_digest.to_hex(),
        );
        *counts.entry(key).or_insert(0) += 1;
    }
    Some(counts)
}

fn provenance_group_counts(
    index: &EvidenceDependencyIndex,
    ids: &[u32],
) -> Option<(usize, usize, usize, usize)> {
    let mut roots = BTreeSet::new();
    let mut families = BTreeSet::new();
    let mut unknown_roots = 0_usize;
    let mut unknown_families = 0_usize;
    for event_id in ids {
        let item = index.records.get(event_id)?;
        if item.source_root_digest.is_known() {
            roots.insert(item.source_root_digest);
        } else {
            unknown_roots += 1;
        }
        if item.source_family != 0 {
            families.insert(item.source_family);
        } else {
            unknown_families += 1;
        }
    }
    Some((
        roots.len(),
        families.len(),
        unknown_roots,
        unknown_families,
    ))
}

/// Select a disjoint dropout matched on scope, action and estimand. After
/// dependency closure, the intervention must still match the target's
/// statistical strata and provenance-group counts or it is omitted.
fn matched_random_ids(
    index: &EvidenceDependencyIndex,
    target_ids: &[u32],
    seed: u64,
) -> Option<Vec<u32>> {
    let target = normalized_ids(target_ids.to_vec());
    let required = evidence_strata(index, &target)?;
    let target_counts = provenance_group_counts(index, &target)?;
    let target_roots: BTreeSet<SourceRootDigest> = target
        .iter()
        .filter_map(|id| index.records.get(id).map(|item| item.source_root_digest))
        .filter(|root| root.is_known())
        .collect();
    let target_families: BTreeSet<u16> = target
        .iter()
        .filter_map(|id| index.records.get(id).map(|item| item.source_family))
        .filter(|family| *family != 0)
        .collect();
    let target_set: BTreeSet<u32> = target.iter().copied().collect();

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
                    && !target_set.contains(&item.event_id)
                    && (!item.source_root_digest.is_known()
                        || !target_roots.contains(&item.source_root_digest))
                    && (item.source_family == 0 || !target_families.contains(&item.source_family))
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

    let selected = normalized_ids(selected);
    let expanded = index.dependent_closure(&selected);
    if expanded.iter().any(|id| target_set.contains(id)) {
        return None;
    }
    if evidence_strata(index, &expanded)? != evidence_strata(index, &target)? {
        return None;
    }
    if provenance_group_counts(index, &expanded)? != target_counts {
        return None;
    }
    Some(expanded)
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
    build_manifest_with_sidecar(memory, None, public_seed, max_root_pairs)
}

fn build_manifest_with_sidecar(
    memory: &[Experience],
    sidecar: Option<&EvidenceDependencySidecarV1>,
    public_seed: u64,
    max_root_pairs: usize,
) -> Result<InterventionManifest, IndexError> {
    let index = match sidecar {
        Some(sidecar) => EvidenceDependencyIndex::build_with_sidecar(memory, sidecar)?,
        None => EvidenceDependencyIndex::build(memory)?,
    };
    let baseline_evidence_sha256 = index.baseline_evidence_digest();
    let mut interventions = Vec::new();

    for event_id in index.records.keys() {
        let closure = index.dependent_closure(&[*event_id]);
        push_intervention(
            &mut interventions,
            InterventionKind::Event,
            format!("event:{event_id}"),
            closure,
            None,
        );
    }

    let root_groups: Vec<(String, Vec<u32>)> = index
        .root_groups()
        .map(|(key, ids)| (key, index.dependent_closure(&ids)))
        .collect();
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

    let family_groups: Vec<(String, Vec<u32>)> = index
        .family_groups()
        .map(|(key, ids)| (key, index.dependent_closure(&ids)))
        .collect();
    for (key, ids) in &family_groups {
        push_intervention(
            &mut interventions,
            InterventionKind::SourceFamily,
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

    if !index.unknown_root_ids.is_empty() {
        push_intervention(
            &mut interventions,
            InterventionKind::UnknownRoot,
            "unknown_source_roots".to_string(),
            index.dependent_closure(&index.unknown_root_ids.iter().copied().collect::<Vec<_>>()),
            None,
        );
    }
    if !index.unknown_family_ids.is_empty() {
        push_intervention(
            &mut interventions,
            InterventionKind::UnknownFamily,
            "unknown_source_families".to_string(),
            index.dependent_closure(&index.unknown_family_ids.iter().copied().collect::<Vec<_>>()),
            None,
        );
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

    let root_pair_count = interventions
        .iter()
        .filter(|item| item.kind == InterventionKind::SourceRootPair)
        .count();
    let matched_random_count = interventions
        .iter()
        .filter(|item| item.kind == InterventionKind::MatchedRandom)
        .count();
    let matched_random_target_count = root_groups.len() + family_groups.len();
    let unmatched_random_target_count =
        matched_random_target_count.saturating_sub(matched_random_count);

    let manifest_payload = serde_json::to_vec(&json!({
        "schema_version": SCHEMA_VERSION,
        "seed": public_seed,
        "baseline_evidence_sha256": baseline_evidence_sha256,
        "max_root_pairs": max_root_pairs,
        "root_pair_count": root_pair_count,
        "matched_random_count": matched_random_count,
        "unmatched_random_target_count": unmatched_random_target_count,
        "dependency_sidecar_sha256": index.dependency_sidecar_sha256(),
        "lineage_incomplete_event_count": index.lineage_incomplete_event_count,
        "unresolved_parent_reference_count": index.unresolved_parent_reference_count,
        "interventions": &interventions
    }))
    .expect("candidate intervention manifest is serializable");
    let manifest_sha256 = hex_digest(Sha256::digest(manifest_payload).as_slice());

    Ok(InterventionManifest {
        schema_version: SCHEMA_VERSION,
        seed: public_seed,
        baseline_evidence_sha256,
        interventions,
        max_root_pairs,
        root_pair_count,
        matched_random_count,
        unmatched_random_target_count,
        dependency_sidecar_sha256: index.dependency_sidecar_sha256().map(str::to_owned),
        lineage_incomplete_event_count: index.lineage_incomplete_event_count,
        unresolved_parent_reference_count: index.unresolved_parent_reference_count,
        manifest_sha256,
    })
}

fn stable_key_seed(key: &str) -> u64 {
    let digest = Sha256::digest(key.as_bytes());
    u64::from_be_bytes(digest[..8].try_into().expect("SHA-256 prefix is eight bytes"))
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

fn candidate_pair_block_id(
    world_cluster: u64,
    decision_need: DecisionNeedV1,
    evidence_sha256: &str,
) -> String {
    let input_digest = candidate_input_digest(decision_need, evidence_sha256);
    format!("world-{world_cluster}:task-{}", &input_digest[..16])
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum CandidateMode {
    ExactScopeScan,
    EventIdKeyValueTopK,
    StaticHdc,
    HdcLtcZeroInput,
    TypedContentHdc,
}

impl CandidateMode {
    const ALL: [Self; 5] = [
        Self::ExactScopeScan,
        Self::EventIdKeyValueTopK,
        Self::StaticHdc,
        Self::HdcLtcZeroInput,
        Self::TypedContentHdc,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::ExactScopeScan => "exact_scope_scan",
            Self::EventIdKeyValueTopK => "event_id_key_value_topk",
            Self::StaticHdc => "static_hdc",
            Self::HdcLtcZeroInput => "hdc_ltc_zero_input",
            Self::TypedContentHdc => "typed_content_hdc",
        }
    }
}

#[derive(Clone, Debug)]
struct CandidateObservation {
    mode: CandidateMode,
    decision: Decision,
    input_sha256: String,
    evidence_sha256: String,
    retrieved_event_ids: Vec<u32>,
    retrieval_profile_sha256: Option<String>,
    index_bytes: u64,
    evidence_event_count: usize,
    evidence_root_count: usize,
    evidence_family_count: usize,
    unknown_root_event_count: usize,
    unknown_family_event_count: usize,
    dependency_sidecar_sha256: Option<String>,
    lineage_incomplete_event_count: usize,
    unresolved_parent_reference_count: usize,
    latency_ns: u64,
}

fn run_candidate(
    mode: CandidateMode,
    query: Query,
    decision_need: DecisionNeedV1,
    memory: &[Experience],
) -> Result<CandidateObservation, IndexError> {
    run_candidate_with_sidecar(mode, query, decision_need, memory, None)
}

fn run_candidate_with_sidecar(
    mode: CandidateMode,
    query: Query,
    decision_need: DecisionNeedV1,
    memory: &[Experience],
    sidecar: Option<&EvidenceDependencySidecarV1>,
) -> Result<CandidateObservation, IndexError> {
    let started = Instant::now();
    if !decision_need.is_valid() || decision_need.scope != query.scope {
        return Err(IndexError::InvalidDecisionNeed);
    }

    let evidence_index = match sidecar {
        Some(sidecar) => EvidenceDependencyIndex::build_with_sidecar(memory, sidecar)?,
        None => EvidenceDependencyIndex::build(memory)?,
    };
    let evidence_sha256 = evidence_index.baseline_evidence_digest();
    let input_sha256 = candidate_input_digest(decision_need, &evidence_sha256);

    let (policy_memory, retrieved_event_ids, retrieval_profile_sha256, index_bytes) =
        match mode {
            CandidateMode::ExactScopeScan => {
                let eligible: Vec<Experience> = evidence_index
                    .records
                    .values()
                    .filter(|item| {
                        item.scope == query.scope
                            && decision_need.action_alternatives.contains(&item.action)
                    })
                    .copied()
                    .collect();
                let ids = eligible.iter().map(|item| item.event_id).collect();
                (eligible, ids, None, 0)
            }
            CandidateMode::EventIdKeyValueTopK => {
                let mut resolved = Vec::new();
                let mut ids = Vec::new();
                for action in decision_need.action_alternatives {
                    let mut candidates: Vec<&Experience> = evidence_index
                        .records
                        .values()
                        .filter(|item| item.scope == query.scope && item.action == action)
                        .collect();
                    candidates.sort_by_key(|item| item.event_id);
                    for item in candidates.into_iter().take(BASELINE_TOP_K) {
                        ids.push(item.event_id);
                        resolved.push(*item);
                    }
                }
                // This arm uses the shared canonical event map and an
                // event-ID ordering rule, not a separately retained index.
                (resolved, ids, None, 0)
            }
            CandidateMode::StaticHdc | CandidateMode::HdcLtcZeroInput => {
                if memory.is_empty() {
                    (Vec::new(), Vec::new(), None, 0)
                } else {
                    let profile = match mode {
                        CandidateMode::StaticHdc => RepresentationProfile::StaticContinuousHv,
                        CandidateMode::HdcLtcZeroInput => RepresentationProfile::HdcLtcZeroInput {
                            elapsed_seconds: 0.25,
                            tau_base_seconds: 10.0,
                            tau_min_seconds: 0.01,
                            tau_max_seconds: 20.0,
                            neuron_seed: EVENT_ID_LTC_SEED,
                        },
                        _ => unreachable!("matched HDC modes only"),
                    };
                    let index = CandidateIndex::build(
                        memory,
                        IndexConfig {
                            dimension: EVENT_ID_HDC_DIMENSION,
                            codebook_seed: EVENT_ID_HDC_SEED,
                            profile,
                        },
                    )
                    .map_err(|error| IndexError::TypedIndexFailure(error.to_string()))?;

                    let mut hits = Vec::new();
                    let mut profile_sha256 = None;
                    for action in decision_need.action_alternatives {
                        let retrieval = index.retrieve(query.scope, action, BASELINE_TOP_K);
                        profile_sha256 = Some(retrieval.profile_digest.clone());
                        for hit in retrieval.candidates {
                            hits.push((hit.event_id, hit.view_digest, hit.rank));
                        }
                    }
                    hits.sort_by(|left, right| {
                        left.2.cmp(&right.2).then_with(|| left.0.cmp(&right.0))
                    });
                    let mut resolved = Vec::new();
                    let mut ids = Vec::new();
                    let mut seen = BTreeSet::new();
                    for (event_id, indexed_digest, _) in hits {
                        if !seen.insert(event_id) {
                            continue;
                        }
                        let item = evidence_index.records.get(&event_id).copied()
                            .ok_or(IndexError::UnresolvedCandidateId(event_id))?;
                        if experience_view_digest(&item) != indexed_digest || item.event_id != event_id {
                            return Err(IndexError::CanonicalViewMismatch(event_id));
                        }
                        ids.push(event_id);
                        resolved.push(item);
                    }
                    (resolved, ids, profile_sha256, index.index_payload_bytes())
                }
            }
            CandidateMode::TypedContentHdc => {
                if memory.is_empty() {
                    (Vec::new(), Vec::new(), None, 0)
                } else {
                    let config = TypedContentConfigV1 {
                        dimension: TYPED_HDC_DIMENSION,
                        seed: TYPED_HDC_SEED,
                        ..TypedContentConfigV1::default()
                    };
                    let typed_index = TypedContentIndexV1::build(memory, config)
                        .map_err(|error| IndexError::TypedIndexFailure(error.to_string()))?;
                    let profile_sha256 = typed_index.profile_digest().to_string();

                    let mut hits: Vec<(u32, [u8; 32], usize)> = Vec::new();
                    for action in decision_need.action_alternatives {
                        let typed_query =
                            SemanticExperienceQueryV1::from_decision_need(decision_need, action)
                                .map_err(|error| IndexError::TypedIndexFailure(error.to_string()))?;
                        let retrieval = typed_index
                            .retrieve(typed_query, TYPED_HDC_TOP_K)
                            .map_err(|error| IndexError::TypedIndexFailure(error.to_string()))?;
                        for hit in retrieval.candidates {
                            hits.push((hit.event_id, hit.view_digest, hit.rank));
                        }
                    }
                    hits.sort_by(|left, right| {
                        left.2
                            .cmp(&right.2)
                            .then_with(|| left.0.cmp(&right.0))
                    });

                    let mut resolved = Vec::with_capacity(hits.len());
                    let mut retrieved_ids = Vec::with_capacity(hits.len());
                    let mut seen = BTreeSet::new();
                    for (event_id, indexed_view_digest, _) in hits {
                        if !seen.insert(event_id) {
                            continue;
                        }
                        let item = evidence_index
                            .records
                            .get(&event_id)
                            .copied()
                            .ok_or(IndexError::UnresolvedCandidateId(event_id))?;
                        if experience_view_digest(&item) != indexed_view_digest
                            || item.event_id != event_id
                        {
                            return Err(IndexError::CanonicalViewMismatch(event_id));
                        }
                        retrieved_ids.push(event_id);
                        resolved.push(item);
                    }
                    (
                        resolved,
                        retrieved_ids,
                        Some(profile_sha256),
                        typed_index.index_payload_bytes(),
                    )
                }
            }
        };

    let decision = decide(Policy::CausalQualified, &policy_memory, query);

    Ok(CandidateObservation {
        mode,
        decision,
        input_sha256,
        evidence_sha256,
        retrieved_event_ids,
        retrieval_profile_sha256,
        index_bytes,
        evidence_event_count: evidence_index.event_count(),
        evidence_root_count: evidence_index.known_root_count(),
        evidence_family_count: evidence_index.known_family_count(),
        unknown_root_event_count: evidence_index.unknown_root_event_count(),
        unknown_family_event_count: evidence_index.unknown_family_event_count(),
        dependency_sidecar_sha256: evidence_index.dependency_sidecar_sha256().map(str::to_owned),
        lineage_incomplete_event_count: evidence_index.lineage_incomplete_event_count,
        unresolved_parent_reference_count: evidence_index.unresolved_parent_reference_count,
        latency_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
    })
}

fn candidate_observation_id(candidate: &CandidateObservation) -> String {
    let identity = format!("{}:{}", candidate.mode.name(), candidate.input_sha256);
    format!("obs-{}", &hex_digest(Sha256::digest(identity.as_bytes()).as_slice())[..16])
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
    let source_code_sha256 =
        hex_digest(Sha256::digest(include_bytes!("mace_counterfactual_evidence.rs")).as_slice());
    let fixture_sha256 =
        hex_digest(Sha256::digest(include_bytes!("mace_generated_stress.rs")).as_slice());
    let schema_sha256 = hex_digest(
        Sha256::digest(include_bytes!(
            "../../../../../data/schemas/mace_counterfactual_evidence_v0_1.schema.json"
        ))
        .as_slice(),
    );
    let source_revision = env::var("MACE_SOURCE_REVISION").ok();
    if source_revision
        .as_deref()
        .is_some_and(|revision| !valid_git_revision(revision))
    {
        return Err("MACE_SOURCE_REVISION must be a 40- or 64-character lowercase Git SHA".to_string());
    }
    let run_identity = format!(
        "{clusters}:{public_seed}:{SCHEMA_VERSION}:{source_code_sha256}:{fixture_sha256}:{schema_sha256}:{}",
        source_revision.as_deref().unwrap_or("unspecified")
    );
    let run_id = format!(
        "mace-counterfactual-{}-{}",
        &hex_digest(Sha256::digest(run_identity.as_bytes()).as_slice())[..16],
        clusters
    );
    let mut candidate_records = Vec::new();
    let mut evaluator_records = Vec::new();

    for case in &cases {
        let manifest = build_manifest(&case.memory, public_seed, max_root_pairs)
            .map_err(|error| format!("dependency index failed: {error:?}"))?;
        let baseline_observations: Vec<CandidateObservation> = CandidateMode::ALL
            .iter()
            .map(|mode| {
                run_candidate(*mode, case.query, case.decision_need, &case.memory)
                    .map_err(|error| format!("{} baseline failed: {error:?}", mode.name()))
            })
            .collect::<Result<_, _>>()?;
        let baseline_regrets: BTreeMap<CandidateMode, f64> = baseline_observations
            .iter()
            .map(|observation| {
                (
                    observation.mode,
                    regret(case.truth, observation.decision.action),
                )
            })
            .collect();
        let baseline_exact = baseline_observations
            .iter()
            .find(|observation| observation.mode == CandidateMode::ExactScopeScan)
            .ok_or_else(|| "exact-scope baseline missing".to_string())?;
        let baseline_exact_regret = baseline_regrets[&CandidateMode::ExactScopeScan];

        // Opaque pair identity: do not expose generated evaluator scenario
        // family labels (for example, retraction or common-cause attack).
        let pair_block_id = candidate_pair_block_id(
            case.cluster,
            case.decision_need,
            &manifest.baseline_evidence_sha256,
        );
        evaluator_records.push(json!({
            "schema_version": SCHEMA_VERSION,
            "record_type": "intervention_manifest",
            "run_id": run_id,
            "pair_block_id": pair_block_id,
            "manifest_sha256": manifest.manifest_sha256,
            "baseline_evidence_sha256": manifest.baseline_evidence_sha256,
            "public_seed": manifest.seed,
            "intervention_count": manifest.interventions.len(),
            "interventions": &manifest.interventions,
            "max_root_pairs": manifest.max_root_pairs,
            "root_pair_count": manifest.root_pair_count,
            "matched_random_count": manifest.matched_random_count,
            "unmatched_random_target_count": manifest.unmatched_random_target_count,
            "dependency_sidecar_sha256": manifest.dependency_sidecar_sha256,
            "lineage_incomplete_event_count": manifest.lineage_incomplete_event_count,
            "unresolved_parent_reference_count": manifest.unresolved_parent_reference_count,
            "source_code_sha256": source_code_sha256,
            "fixture_sha256": fixture_sha256,
            "schema_sha256": schema_sha256,
            "source_revision": source_revision,
            "known_root_count": EvidenceDependencyIndex::build(&case.memory)
                .map_err(|error| format!("dependency index failed: {error:?}"))?
                .known_root_count(),
            "candidate_validity": "diagnostic_only"
        }));

        for baseline in &baseline_observations {
            let support = EvidenceDependencyIndex::build(&case.memory)
                .map_err(|error| format!("dependency index failed: {error:?}"))?
                .support_closure_counts(&baseline.decision.evidence_ids);
            let observation_id = candidate_observation_id(baseline);
            let candidate_record = json!({
                "schema_version": SCHEMA_VERSION,
                "record_type": "candidate_decision",
                "run_id": run_id,
                "pair_block_id": pair_block_id,
                "candidate_observation_id": observation_id,
                "method": baseline.mode.name(),
                "candidate_input_sha256": baseline.input_sha256,
                "decision_need_sha256": case.decision_need.canonical_digest().to_hex(),
                "candidate_evidence_sha256": baseline.evidence_sha256,
                "retrieved_event_ids": baseline.retrieved_event_ids,
                "decision": action_name(baseline.decision.action),
                "decision_support_event_ids": baseline.decision.evidence_ids,
                "support_event_count": support.0,
                "support_known_root_count": support.1,
                "support_known_family_count": support.2,
                "support_unresolved_count": support.3,
                "available_event_count": baseline.evidence_event_count,
                "available_experience_struct_bytes": baseline.evidence_event_count.saturating_mul(size_of::<Experience>()),
                "available_known_root_count": baseline.evidence_root_count,
                "available_known_family_count": baseline.evidence_family_count,
                "unknown_root_event_count": baseline.unknown_root_event_count,
                "unknown_family_event_count": baseline.unknown_family_event_count,
                "dependency_sidecar_sha256": baseline.dependency_sidecar_sha256,
                "lineage_incomplete_event_count": baseline.lineage_incomplete_event_count,
                "unresolved_parent_reference_count": baseline.unresolved_parent_reference_count,
                "index_bytes": baseline.index_bytes,
                "latency_ns": baseline.latency_ns
            });
            let mut candidate_record = candidate_record;
            if let Some(profile) = &baseline.retrieval_profile_sha256 {
                candidate_record["retrieval_profile_sha256"] = json!(profile);
            }
            candidate_records.push(candidate_record);

            let current_regret = regret(case.truth, baseline.decision.action);
            evaluator_records.push(json!({
                "schema_version": SCHEMA_VERSION,
                "record_type": "evaluator_score",
                "run_id": run_id,
                "pair_block_id": pair_block_id,
                "candidate_observation_id": observation_id,
                "candidate_input_sha256": baseline.input_sha256,
                "method": baseline.mode.name(),
                "intervention_id": format!("full_evidence_{}", baseline.mode.name()),
                "intervention_kind": "reference",
                "target_key": baseline.mode.name(),
                "retrieved_event_ids": baseline.retrieved_event_ids,
                "dependency_sidecar_sha256": baseline.dependency_sidecar_sha256,
                "lineage_incomplete_event_count": baseline.lineage_incomplete_event_count,
                "unresolved_parent_reference_count": baseline.unresolved_parent_reference_count,
                "index_bytes": baseline.index_bytes,
                "removed_event_ids": [],
                "removed_event_count": 0,
                "removed_experience_struct_bytes": 0,
                "decision": action_name(baseline.decision.action),
                "full_evidence_decision": action_name(baseline_exact.decision.action),
                "decision_agrees_with_full": baseline.decision.action == baseline_exact.decision.action,
                "decision_changed": baseline.decision.action != baseline_exact.decision.action,
                "utility": truth_utility(case.truth, baseline.decision.action),
                "regret": current_regret,
                "full_evidence_regret": baseline_exact_regret,
                "regret_gap_vs_full_evidence": current_regret - baseline_exact_regret,
                "regret_gap_vs_full_method": 0.0,
                "decision_is_optimal": baseline.decision.action == Some(case.truth.optimal_action()),
                "abstained": baseline.decision.action.is_none(),
                "evaluation_validity": "diagnostic_only",
                "invalidity_reason": "public synthetic fixture and co-located evaluator"
            }));
            if let Some(profile) = &baseline.retrieval_profile_sha256 {
                if let Some(last) = evaluator_records.last_mut() {
                    last["retrieval_profile_sha256"] = json!(profile);
                }
            }
        }

        for intervention in &manifest.interventions {
            let index = EvidenceDependencyIndex::build(&case.memory)
                .map_err(|error| format!("dependency index failed: {error:?}"))?;
            let retained = index.retain_without(&intervention.removed_event_ids);

            for mode in CandidateMode::ALL {
                let candidate = run_candidate(
                    mode,
                    case.query,
                    case.decision_need,
                    &retained,
                )
                .map_err(|error| format!("intervention candidate failed: {error:?}"))?;
                let support = EvidenceDependencyIndex::build(&retained)
                    .map_err(|error| format!("retained evidence index failed: {error:?}"))?
                    .support_closure_counts(&candidate.decision.evidence_ids);
                let observation_id = candidate_observation_id(&candidate);

                let candidate_record = json!({
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "candidate_decision",
                    "run_id": run_id,
                    "pair_block_id": pair_block_id,
                    "candidate_observation_id": observation_id,
                    "method": candidate.mode.name(),
                    "candidate_input_sha256": candidate.input_sha256,
                    "decision_need_sha256": case.decision_need.canonical_digest().to_hex(),
                    "candidate_evidence_sha256": candidate.evidence_sha256,
                    "retrieved_event_ids": candidate.retrieved_event_ids,
                    "decision": action_name(candidate.decision.action),
                    "decision_support_event_ids": candidate.decision.evidence_ids,
                    "support_event_count": support.0,
                    "support_known_root_count": support.1,
                    "support_known_family_count": support.2,
                    "support_unresolved_count": support.3,
                    "available_event_count": candidate.evidence_event_count,
                    "available_experience_struct_bytes": candidate.evidence_event_count.saturating_mul(size_of::<Experience>()),
                    "available_known_root_count": candidate.evidence_root_count,
                    "available_known_family_count": candidate.evidence_family_count,
                    "unknown_root_event_count": candidate.unknown_root_event_count,
                    "unknown_family_event_count": candidate.unknown_family_event_count,
                    "dependency_sidecar_sha256": candidate.dependency_sidecar_sha256,
                    "lineage_incomplete_event_count": candidate.lineage_incomplete_event_count,
                    "unresolved_parent_reference_count": candidate.unresolved_parent_reference_count,
                    "index_bytes": candidate.index_bytes,
                    "latency_ns": candidate.latency_ns
                });
                let mut candidate_record = candidate_record;
                if let Some(profile) = &candidate.retrieval_profile_sha256 {
                    candidate_record["retrieval_profile_sha256"] = json!(profile);
                }
                candidate_records.push(candidate_record);

                let current_regret = regret(case.truth, candidate.decision.action);
                let same_method_baseline_regret = baseline_regrets[&mode];
                let evaluator_record = json!({
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "evaluator_score",
                    "run_id": run_id,
                    "pair_block_id": pair_block_id,
                    "candidate_observation_id": observation_id,
                    "candidate_input_sha256": candidate.input_sha256,
                    "method": candidate.mode.name(),
                    "intervention_id": intervention.intervention_id,
                    "intervention_kind": intervention.kind,
                    "target_key": intervention.target_key,
                    "matched_to": intervention.matched_to,
                    "retrieved_event_ids": candidate.retrieved_event_ids,
                    "dependency_sidecar_sha256": candidate.dependency_sidecar_sha256,
                    "lineage_incomplete_event_count": candidate.lineage_incomplete_event_count,
                    "unresolved_parent_reference_count": candidate.unresolved_parent_reference_count,
                    "index_bytes": candidate.index_bytes,
                    "removed_event_ids": intervention.removed_event_ids,
                    "removed_event_count": intervention.removed_event_ids.len(),
                    "removed_experience_struct_bytes": intervention.removed_event_ids.len().saturating_mul(size_of::<Experience>()),
                    "decision": action_name(candidate.decision.action),
                    "full_evidence_decision": action_name(baseline_exact.decision.action),
                    "decision_agrees_with_full": candidate.decision.action == baseline_exact.decision.action,
                    "decision_changed": candidate.decision.action != baseline_exact.decision.action,
                    "utility": truth_utility(case.truth, candidate.decision.action),
                    "regret": current_regret,
                    "full_evidence_regret": baseline_exact_regret,
                    "regret_gap_vs_full_evidence": current_regret - baseline_exact_regret,
                    "regret_gap_vs_full_method": current_regret - same_method_baseline_regret,
                    "decision_is_optimal": candidate.decision.action == Some(case.truth.optimal_action()),
                    "abstained": candidate.decision.action.is_none(),
                    "evaluation_validity": "diagnostic_only",
                    "invalidity_reason": "public synthetic fixture and co-located evaluator"
                });
                let mut evaluator_record = evaluator_record;
                if let Some(profile) = &candidate.retrieval_profile_sha256 {
                    evaluator_record["retrieval_profile_sha256"] = json!(profile);
                }
                evaluator_records.push(evaluator_record);
            }
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

    fn sidecar_scope() -> broca_mace_policy::Scope {
        broca_mace_policy::Scope {
            regime: broca_mace_policy::Regime::Baseline,
            objective: broca_mace_policy::Objective::Quality,
        }
    }

    fn sidecar_semantic_estimand() -> broca_mace_policy::SemanticEstimandV1 {
        use broca_mace_policy::{EvidenceDigest256, SemanticEstimandV1};
        SemanticEstimandV1::new(
            1,
            EvidenceDigest256::from_bytes([20; 32]),
            EvidenceDigest256::from_bytes([21; 32]),
            EvidenceDigest256::from_bytes([22; 32]),
            EvidenceDigest256::from_bytes([23; 32]),
            EvidenceDigest256::from_bytes([24; 32]),
            30_000,
            EvidenceDigest256::from_bytes([25; 32]),
            EvidenceDigest256::from_bytes([26; 32]),
            EvidenceDigest256::from_bytes([27; 32]),
        )
    }

    fn sidecar_causal_identification() -> broca_mace_policy::CausalIdentificationV1 {
        use broca_mace_policy::{CausalIdentificationV1, EvidenceDigest256, IdentificationMethodV1};
        let semantic = sidecar_semantic_estimand();
        CausalIdentificationV1::new(
            IdentificationMethodV1::RandomizedControlledTrial,
            EvidenceDigest256::from_bytes([1; 32]),
            semantic.canonical_digest(),
            5_000,
            5_000,
            50_000,
            true,
        )
        .with_semantic_estimand(semantic)
    }

    fn sidecar_raw(event_id: u32) -> broca_mace_ingress::RawInteractionV1 {
        use broca_mace_ingress::RawInteractionV1;
        RawInteractionV1 {
            event_id,
            session_id: format!("session-{event_id}"),
            task_id: format!("task-{event_id}"),
            payload_sha256: format!("{event_id:064x}"),
            raw_event_receipt_id: format!("raw-{event_id}"),
            scope: sidecar_scope(),
            action: Action::A,
            observed_utility: 0.87,
            utility_unit: "normalized_utility".into(),
            valid_from_ms: 100,
            valid_until_ms: Some(300),
            occurred_at_ms: 170,
            prediction_registered_at_ms: Some(150),
            action_dispatched_at_ms: Some(170),
            outcome_observed_at_ms: 200,
            lineage_receipt_id: format!("lineage-{event_id}"),
            calibration_receipt_id: format!("calibration-{event_id}"),
            causal_assessment_receipt_id: format!("causal-{event_id}"),
            outcome_receipt_id: format!("outcome-{event_id}"),
        }
    }

    struct SidecarTestVerifier;

    impl broca_mace_ingress::EvidenceVerifier for SidecarTestVerifier {
        fn verify_raw_event(
            &self,
            _receipt_id: &str,
            raw: &broca_mace_ingress::RawInteractionV1,
            _known_at_ms: i64,
        ) -> Option<broca_mace_ingress::VerifiedRawEvent> {
            use broca_mace_ingress::{AuthorityStatus, VerifiedRawEvent};
            Some(VerifiedRawEvent {
                event_id: raw.event_id,
                session_id: raw.session_id.clone(),
                task_id: raw.task_id.clone(),
                payload_sha256: raw.payload_sha256.clone(),
                canonical_event_hash: "c".repeat(64),
                authority_status: AuthorityStatus::ReceiptAttested,
                authority_receipt_digest: "b".repeat(64),
                valid_from_ms: raw.valid_from_ms,
                valid_until_ms: raw.valid_until_ms,
                occurred_at_ms: raw.occurred_at_ms,
                received_at_ms: 220,
                receipt_digest: format!("{:064x}", u64::from(raw.event_id) + 100),
            })
        }

        fn verify_lineage(
            &self,
            _receipt_id: &str,
            event_id: u32,
            _scope: broca_mace_policy::Scope,
            _known_at_ms: i64,
        ) -> Option<broca_mace_ingress::VerifiedLineage> {
            use broca_mace_ingress::{
                DependencyLineageStatusV1, SourceTrustStatus, VerifiedLineage,
            };
            let parents = if event_id == 43 { vec![42] } else { Vec::new() };
            Some(VerifiedLineage {
                source_family: u16::try_from(event_id).ok()?,
                source_root_id: event_id,
                source_root_digest: {
                    let mut bytes = [0_u8; 32];
                    bytes[..4].copy_from_slice(&event_id.to_be_bytes());
                    broca_mace_policy::SourceRootDigest::from_bytes(bytes)
                },
                trust_status: SourceTrustStatus::TrustedForScope,
                trust_policy_digest: "8".repeat(64),
                acquisition_mode: "api".into(),
                upstream_source_id: format!("synthetic-upstream-{event_id}"),
                source_artifact_digest: format!("{:064x}", u64::from(event_id) + 10),
                transformation_digest: format!("{:064x}", u64::from(event_id) + 20),
                derived_from_event_ids: parents,
                dependency_lineage_status: DependencyLineageStatusV1::Complete,
                dependency_manifest_digest: Some("9".repeat(64)),
                issuer_id: "did:key:test-source-ledger".into(),
                receipt_digest: format!("{:064x}", u64::from(event_id) + 30),
            })
        }

        fn verify_measurement(
            &self,
            _receipt_id: &str,
            expected_unit: &str,
            _scope: broca_mace_policy::Scope,
            _observed_at_ms: i64,
            _known_at_ms: i64,
        ) -> Option<broca_mace_ingress::VerifiedMeasurement> {
            Some(broca_mace_ingress::VerifiedMeasurement {
                absolute_error_bound: 0.02,
                unit: expected_unit.into(),
                instrument_id: "instrument-1".into(),
                method_id: "method-1".into(),
                calibration_revision: "cal-rev-1".into(),
                valid_from_ms: 100,
                valid_until_ms: Some(300),
                calibration_digest: "1".repeat(64),
            })
        }

        fn verify_causal_assessment(
            &self,
            _receipt_id: &str,
            _event_id: u32,
            _scope: broca_mace_policy::Scope,
            _known_at_ms: i64,
        ) -> Option<broca_mace_ingress::VerifiedCausalAssessment> {
            Some(broca_mace_ingress::VerifiedCausalAssessment {
                intervention_confirmed: true,
                prediction_pre_registered: true,
                confounders_controlled: true,
                delayed_horizon_complete: true,
                identification: sidecar_causal_identification(),
                assessment_digest: "2".repeat(64),
            })
        }

        fn verify_outcome(
            &self,
            _receipt_id: &str,
            _event_id: u32,
            _observed_at_ms: i64,
            _known_at_ms: i64,
        ) -> Option<broca_mace_ingress::VerifiedOutcome> {
            Some(broca_mace_ingress::VerifiedOutcome {
                reconciled: true,
                success: true,
                observed_utility: 0.87,
                utility_unit: "normalized_utility".into(),
                outcome_digest: "3".repeat(64),
            })
        }

        fn lifecycle_status_as_of(
            &self,
            _event_id: u32,
            _known_at_ms: i64,
        ) -> Option<broca_mace_ingress::SourceLifecycleStatus> {
            Some(broca_mace_ingress::SourceLifecycleStatus::Active)
        }
    }

    #[test]
    fn receipt_sidecar_expands_knockouts_and_preserves_unresolved_parent_references() {
        use broca_mace_ingress::{
            build_consolidation_evidence_pack, ingest_batch_as_known, AsKnownQuery,
        };

        let as_known = AsKnownQuery {
            valid_at_ms: 180,
            known_at_ms: 250,
        };
        let batch = ingest_batch_as_known(
            &[sidecar_raw(42), sidecar_raw(43)],
            as_known,
            &SidecarTestVerifier,
        )
        .expect("receipt-backed two-event session should ingest");
        let pack = build_consolidation_evidence_pack(&batch, &[42, 43])
            .expect("both source events should enter the evidence pack");
        let sidecar = pack.dependency_sidecar();
        assert!(sidecar.verify_integrity());

        let memory: Vec<Experience> = pack
            .events()
            .iter()
            .map(|event| event.experience())
            .collect();
        let manifest = build_manifest_with_sidecar(&memory, Some(&sidecar), 1234, 8)
            .expect("canonical MACE manifest should consume verified lineage");
        assert_eq!(manifest.lineage_incomplete_event_count, 0);
        assert_eq!(manifest.unresolved_parent_reference_count, 0);
        assert_eq!(manifest.dependency_sidecar_sha256.as_deref(), Some(sidecar.canonical_digest()));

        let parent_knockout = manifest
            .interventions
            .iter()
            .find(|item| {
                item.kind == InterventionKind::Event && item.target_key == "event:42"
            })
            .expect("parent event knockout should exist");
        assert_eq!(parent_knockout.removed_event_ids, vec![42, 43]);

        let need = DecisionNeedV1::new(
            sidecar_scope(),
            sidecar_semantic_estimand(),
        )
        .expect("fixture estimand should define a valid candidate need");
        let without_sidecar = run_candidate(
            CandidateMode::ExactScopeScan,
            Query { scope: sidecar_scope() },
            need,
            &memory,
        )
        .unwrap();
        let with_sidecar = run_candidate_with_sidecar(
            CandidateMode::ExactScopeScan,
            Query { scope: sidecar_scope() },
            need,
            &memory,
            Some(&sidecar),
        )
        .unwrap();
        assert_ne!(without_sidecar.input_sha256, with_sidecar.input_sha256);
        assert_eq!(with_sidecar.lineage_incomplete_event_count, 0);

        // Keep the child but remove its declared parent from both memory and
        // sidecar. The surviving external parent reference remains unresolved.
        let reduced_sidecar = sidecar.retain_without_event_ids(&[42]);
        let reduced_memory: Vec<Experience> = pack
            .events()
            .iter()
            .filter(|event| event.event_id() == 43)
            .map(|event| event.experience())
            .collect();
        let reduced = run_candidate_with_sidecar(
            CandidateMode::ExactScopeScan,
            Query { scope: sidecar_scope() },
            need,
            &reduced_memory,
            Some(&reduced_sidecar),
        )
        .unwrap();
        assert_eq!(reduced.lineage_incomplete_event_count, 1);
        assert_eq!(reduced.unresolved_parent_reference_count, 1);
    }

    fn sidecar_scope() -> broca_mace_policy::Scope {
        broca_mace_policy::Scope {
            regime: broca_mace_policy::Regime::Baseline,
            objective: broca_mace_policy::Objective::Quality,
        }
    }

    fn sidecar_semantic_estimand() -> broca_mace_policy::SemanticEstimandV1 {
        use broca_mace_policy::{EvidenceDigest256, SemanticEstimandV1};
        SemanticEstimandV1::new(
            1,
            EvidenceDigest256::from_bytes([20; 32]),
            EvidenceDigest256::from_bytes([21; 32]),
            EvidenceDigest256::from_bytes([22; 32]),
            EvidenceDigest256::from_bytes([23; 32]),
            EvidenceDigest256::from_bytes([24; 32]),
            30_000,
            EvidenceDigest256::from_bytes([25; 32]),
            EvidenceDigest256::from_bytes([26; 32]),
            EvidenceDigest256::from_bytes([27; 32]),
        )
    }

    fn sidecar_causal_identification() -> broca_mace_policy::CausalIdentificationV1 {
        use broca_mace_policy::{CausalIdentificationV1, EvidenceDigest256, IdentificationMethodV1};
        let semantic = sidecar_semantic_estimand();
        CausalIdentificationV1::new(
            IdentificationMethodV1::RandomizedControlledTrial,
            EvidenceDigest256::from_bytes([1; 32]),
            semantic.canonical_digest(),
            5_000,
            5_000,
            50_000,
            true,
        )
        .with_semantic_estimand(semantic)
    }

    fn sidecar_raw(event_id: u32) -> broca_mace_ingress::RawInteractionV1 {
        use broca_mace_ingress::RawInteractionV1;
        RawInteractionV1 {
            event_id,
            session_id: format!("session-{event_id}"),
            task_id: format!("task-{event_id}"),
            payload_sha256: format!("{event_id:064x}"),
            raw_event_receipt_id: format!("raw-{event_id}"),
            scope: sidecar_scope(),
            action: Action::A,
            observed_utility: 0.87,
            utility_unit: "normalized_utility".into(),
            valid_from_ms: 100,
            valid_until_ms: Some(300),
            occurred_at_ms: 170,
            prediction_registered_at_ms: Some(150),
            action_dispatched_at_ms: Some(170),
            outcome_observed_at_ms: 200,
            lineage_receipt_id: format!("lineage-{event_id}"),
            calibration_receipt_id: format!("calibration-{event_id}"),
            causal_assessment_receipt_id: format!("causal-{event_id}"),
            outcome_receipt_id: format!("outcome-{event_id}"),
        }
    }

    struct SidecarTestVerifier;

    impl broca_mace_ingress::EvidenceVerifier for SidecarTestVerifier {
        fn verify_raw_event(
            &self,
            _receipt_id: &str,
            raw: &broca_mace_ingress::RawInteractionV1,
            _known_at_ms: i64,
        ) -> Option<broca_mace_ingress::VerifiedRawEvent> {
            use broca_mace_ingress::{AuthorityStatus, VerifiedRawEvent};
            Some(VerifiedRawEvent {
                event_id: raw.event_id,
                session_id: raw.session_id.clone(),
                task_id: raw.task_id.clone(),
                payload_sha256: raw.payload_sha256.clone(),
                canonical_event_hash: "c".repeat(64),
                authority_status: AuthorityStatus::ReceiptAttested,
                authority_receipt_digest: "b".repeat(64),
                valid_from_ms: raw.valid_from_ms,
                valid_until_ms: raw.valid_until_ms,
                occurred_at_ms: raw.occurred_at_ms,
                received_at_ms: 220,
                receipt_digest: format!("{:064x}", u64::from(raw.event_id) + 100),
            })
        }

        fn verify_lineage(
            &self,
            _receipt_id: &str,
            event_id: u32,
            _scope: broca_mace_policy::Scope,
            _known_at_ms: i64,
        ) -> Option<broca_mace_ingress::VerifiedLineage> {
            use broca_mace_ingress::{
                DependencyLineageStatusV1, SourceTrustStatus, VerifiedLineage,
            };
            let parents = if event_id == 43 { vec![42] } else { Vec::new() };
            Some(VerifiedLineage {
                source_family: u16::try_from(event_id).ok()?,
                source_root_id: event_id,
                source_root_digest: {
                    let mut bytes = [0_u8; 32];
                    bytes[..4].copy_from_slice(&event_id.to_be_bytes());
                    broca_mace_policy::SourceRootDigest::from_bytes(bytes)
                },
                trust_status: SourceTrustStatus::TrustedForScope,
                trust_policy_digest: "8".repeat(64),
                acquisition_mode: "api".into(),
                upstream_source_id: format!("synthetic-upstream-{event_id}"),
                source_artifact_digest: format!("{:064x}", u64::from(event_id) + 10),
                transformation_digest: format!("{:064x}", u64::from(event_id) + 20),
                derived_from_event_ids: parents,
                dependency_lineage_status: DependencyLineageStatusV1::Complete,
                dependency_manifest_digest: Some("9".repeat(64)),
                issuer_id: "did:key:test-source-ledger".into(),
                receipt_digest: format!("{:064x}", u64::from(event_id) + 30),
            })
        }

        fn verify_measurement(
            &self,
            _receipt_id: &str,
            expected_unit: &str,
            _scope: broca_mace_policy::Scope,
            _observed_at_ms: i64,
            _known_at_ms: i64,
        ) -> Option<broca_mace_ingress::VerifiedMeasurement> {
            Some(broca_mace_ingress::VerifiedMeasurement {
                absolute_error_bound: 0.02,
                unit: expected_unit.into(),
                instrument_id: "instrument-1".into(),
                method_id: "method-1".into(),
                calibration_revision: "cal-rev-1".into(),
                valid_from_ms: 100,
                valid_until_ms: Some(300),
                calibration_digest: "1".repeat(64),
            })
        }

        fn verify_causal_assessment(
            &self,
            _receipt_id: &str,
            _event_id: u32,
            _scope: broca_mace_policy::Scope,
            _known_at_ms: i64,
        ) -> Option<broca_mace_ingress::VerifiedCausalAssessment> {
            Some(broca_mace_ingress::VerifiedCausalAssessment {
                intervention_confirmed: true,
                prediction_pre_registered: true,
                confounders_controlled: true,
                delayed_horizon_complete: true,
                identification: sidecar_causal_identification(),
                assessment_digest: "2".repeat(64),
            })
        }

        fn verify_outcome(
            &self,
            _receipt_id: &str,
            _event_id: u32,
            _observed_at_ms: i64,
            _known_at_ms: i64,
        ) -> Option<broca_mace_ingress::VerifiedOutcome> {
            Some(broca_mace_ingress::VerifiedOutcome {
                reconciled: true,
                success: true,
                observed_utility: 0.87,
                utility_unit: "normalized_utility".into(),
                outcome_digest: "3".repeat(64),
            })
        }

        fn lifecycle_status_as_of(
            &self,
            _event_id: u32,
            _known_at_ms: i64,
        ) -> Option<broca_mace_ingress::SourceLifecycleStatus> {
            Some(broca_mace_ingress::SourceLifecycleStatus::Active)
        }
    }

    #[test]
    fn receipt_lineage_drives_canonical_mace_dependency_knockouts() {
        use broca_mace_ingress::{
            build_consolidation_evidence_pack, ingest_batch_as_known, AsKnownQuery,
        };

        let as_known = AsKnownQuery {
            valid_at_ms: 180,
            known_at_ms: 250,
        };
        let batch = ingest_batch_as_known(
            &[sidecar_raw(42), sidecar_raw(43)],
            as_known,
            &SidecarTestVerifier,
        )
        .expect("receipt-backed two-event session should ingest");
        let pack = build_consolidation_evidence_pack(&batch, &[42, 43])
            .expect("both source events should enter the evidence pack");
        let sidecar = pack.dependency_sidecar();
        assert!(sidecar.verify_integrity());

        let memory: Vec<Experience> = pack
            .events()
            .iter()
            .map(|event| event.experience())
            .collect();
        let unknown_index = EvidenceDependencyIndex::build(&memory).unwrap();
        assert_eq!(unknown_index.lineage_incomplete_event_count, 2);

        let index = EvidenceDependencyIndex::build_with_sidecar(&memory, &sidecar)
            .expect("canonical evaluator should consume verifier-issued sidecar");
        assert_eq!(index.lineage_incomplete_event_count, 0);
        assert_eq!(index.unresolved_parent_reference_count, 0);
        assert_eq!(index.dependent_closure(&[42]), vec![42, 43]);

        let manifest = build_manifest_with_sidecar(&memory, Some(&sidecar), 1234, 8)
            .expect("manifest should use receipt-backed dependency closure");
        assert_eq!(manifest.lineage_incomplete_event_count, 0);
        assert_eq!(manifest.unresolved_parent_reference_count, 0);
        assert_eq!(
            manifest.dependency_sidecar_sha256.as_deref(),
            Some(sidecar.canonical_digest())
        );
        let parent_knockout = manifest
            .interventions
            .iter()
            .find(|item| {
                item.kind == InterventionKind::Event && item.target_key == "event:42"
            })
            .expect("parent event knockout should be present");
        assert_eq!(parent_knockout.removed_event_ids, vec![42, 43]);

        let need = DecisionNeedV1::new(
            sidecar_scope(),
            sidecar_semantic_estimand(),
        )
        .expect("fixture estimand should define a valid task contract");
        let without_sidecar = run_candidate(
            CandidateMode::ExactScopeScan,
            Query { scope: sidecar_scope() },
            need,
            &memory,
        )
        .unwrap();
        let with_sidecar = run_candidate_with_sidecar(
            CandidateMode::ExactScopeScan,
            Query { scope: sidecar_scope() },
            need,
            &memory,
            Some(&sidecar),
        )
        .unwrap();
        assert_ne!(without_sidecar.input_sha256, with_sidecar.input_sha256);
        assert_eq!(with_sidecar.lineage_incomplete_event_count, 0);

        // Remove only the parent. The child's receipt still declares that
        // parent, so the reduced candidate pool must mark the lineage unknown.
        let reduced_sidecar = sidecar.retain_without_event_ids(&[42]);
        let reduced_memory: Vec<Experience> = pack
            .events()
            .iter()
            .filter(|event| event.event_id() == 43)
            .map(|event| event.experience())
            .collect();
        let reduced = run_candidate_with_sidecar(
            CandidateMode::ExactScopeScan,
            Query { scope: sidecar_scope() },
            need,
            &reduced_memory,
            Some(&reduced_sidecar),
        )
        .unwrap();
        assert_eq!(reduced.lineage_incomplete_event_count, 1);
        assert_eq!(reduced.unresolved_parent_reference_count, 1);
    }

    #[test]
    fn source_revision_accepts_only_canonical_lowercase_git_shas() {
        assert!(valid_git_revision(&"a".repeat(40)));
        assert!(valid_git_revision(&"b".repeat(64)));
        assert!(!valid_git_revision("deadbeef"));
        assert!(!valid_git_revision(&"A".repeat(40)));
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
    fn root_knockout_removes_every_event_sharing_digest_not_legacy_id() {
        let mut memory = first_memory();
        let shared = SourceRootDigest::from_bytes([0xB6; 32]);
        memory[0].source_root_digest = shared;
        memory[1].source_root_digest = shared;
        memory[0].source_root_id = 17;
        memory[1].source_root_id = 901;

        let index = EvidenceDependencyIndex::build(&memory).unwrap();
        let (_, root_ids) = index
            .root_groups()
            .find(|(key, _)| key == &format!("root:{}", shared.to_hex()))
            .unwrap();
        let retained = index.retain_without(&root_ids);
        assert!(retained.iter().all(|item| item.source_root_digest != shared));
        assert_eq!(retained.len(), memory.len() - 2);
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

        let manifest = build_manifest(&memory, 9, 4).unwrap();
        assert!(manifest.interventions.iter().any(|item| {
            item.kind == InterventionKind::UnknownRoot
                && item.removed_event_ids == vec![memory[0].event_id, memory[1].event_id]
        }));
        assert!(manifest.interventions.iter().any(|item| {
            item.kind == InterventionKind::UnknownFamily
                && item.removed_event_ids == vec![memory[0].event_id, memory[1].event_id]
        }));
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
    fn root_pair_interventions_respect_the_declared_coalition_cap() {
        let memory = first_memory();
        let capped = build_manifest(&memory, 77, 1).unwrap();
        let unlimited = build_manifest(&memory, 77, usize::MAX).unwrap();
        assert_eq!(capped.root_pair_count, 1);
        assert!(unlimited.root_pair_count >= capped.root_pair_count);
        assert_eq!(capped.max_root_pairs, 1);
        assert_eq!(unlimited.max_root_pairs, usize::MAX);
    }

    #[test]
    fn hidden_outcome_mutation_does_not_change_candidate_only_manifest() {
        let original = generated_worlds(1).remove(0);
        let mut changed_hidden = original.clone();
        changed_hidden.truth.utility_a = 0.001;
        changed_hidden.truth.utility_b = 0.999;
        changed_hidden.truth.causal_action = Some(Action::B);
        changed_hidden.truth.invalid_a = !changed_hidden.truth.invalid_a;

        assert_eq!(
            build_manifest(&original.memory, 0x1234, 8).unwrap(),
            build_manifest(&changed_hidden.memory, 0x1234, 8).unwrap()
        );
        for mode in CandidateMode::ALL {
            let original_candidate = run_candidate(
                mode,
                original.query,
                original.decision_need,
                &original.memory,
            )
            .unwrap();
            let changed_candidate = run_candidate(
                mode,
                changed_hidden.query,
                changed_hidden.decision_need,
                &changed_hidden.memory,
            )
            .unwrap();
            assert_eq!(
                original_candidate.input_sha256,
                changed_candidate.input_sha256
            );
            assert_eq!(
                original_candidate.retrieved_event_ids,
                changed_candidate.retrieved_event_ids
            );
            assert_eq!(
                original_candidate.decision.action,
                changed_candidate.decision.action
            );
        }
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
    fn candidate_rejects_invalid_or_scope_mismatched_decision_need() {
        let case = generated_worlds(1).remove(0);
        let mut invalid_need = case.decision_need;
        invalid_need.schema_version = 99;
        assert_eq!(
            run_candidate(CandidateMode::ExactScopeScan, case.query, invalid_need, &case.memory).unwrap_err(),
            IndexError::InvalidDecisionNeed
        );

        let wrong_scope = Query {
            scope: Scope {
                regime: Regime::Shifted,
                objective: case.query.scope.objective,
            },
        };
        assert_eq!(
            run_candidate(CandidateMode::ExactScopeScan, wrong_scope, case.decision_need, &case.memory).unwrap_err(),
            IndexError::InvalidDecisionNeed
        );
    }

    #[test]
    fn pair_block_identity_does_not_expose_evaluator_scenario_family() {
        let case = generated_worlds(1).remove(0);
        let index = EvidenceDependencyIndex::build(&case.memory).unwrap();
        let pair_id = candidate_pair_block_id(
            case.cluster,
            case.decision_need,
            &index.baseline_evidence_digest(),
        );
        assert!(pair_id.starts_with(&format!("world-{}:task-", case.cluster)));
        assert!(!pair_id.contains(case.family));
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
    fn all_retrieval_modes_share_candidate_inputs_and_resolve_in_scope_evidence() {
        let case = generated_worlds(1).remove(0);
        let exact = run_candidate(
            CandidateMode::ExactScopeScan,
            case.query,
            case.decision_need,
            &case.memory,
        )
        .unwrap();

        for mode in CandidateMode::ALL {
            let candidate = run_candidate(
                mode,
                case.query,
                case.decision_need,
                &case.memory,
            )
            .unwrap();
            assert_eq!(candidate.input_sha256, exact.input_sha256, "{}", mode.name());
            assert!(candidate.retrieved_event_ids.iter().all(|event_id| {
                case.memory.iter().any(|item| {
                    item.event_id == *event_id
                        && item.scope == case.query.scope
                        && case.decision_need.action_alternatives.contains(&item.action)
                })
            }), "{}", mode.name());
            assert!(candidate.decision.evidence_ids.iter().all(|event_id| {
                candidate.retrieved_event_ids.contains(event_id)
            }), "{}", mode.name());
        }
    }

    #[test]
    fn event_id_key_value_baseline_is_bounded_per_declared_action() {
        let case = generated_worlds(1).remove(0);
        let kv = run_candidate(
            CandidateMode::EventIdKeyValueTopK,
            case.query,
            case.decision_need,
            &case.memory,
        )
        .unwrap();
        assert!(kv.retrieved_event_ids.len() <= BASELINE_TOP_K * 2);
        assert!(kv.index_bytes > 0);
        assert!(kv.retrieval_profile_sha256.is_none());
    }

    #[test]
    fn static_hdc_and_zero_input_ltc_revalidate_canonical_views() {
        let case = generated_worlds(1).remove(0);
        for mode in [CandidateMode::StaticHdc, CandidateMode::HdcLtcZeroInput] {
            let candidate = run_candidate(
                mode,
                case.query,
                case.decision_need,
                &case.memory,
            )
            .unwrap();
            assert!(candidate.retrieval_profile_sha256.is_some(), "{}", mode.name());
            assert!(candidate.index_bytes > 0, "{}", mode.name());
            assert!(candidate.retrieved_event_ids.iter().all(|event_id| {
                case.memory.iter().any(|item| item.event_id == *event_id)
            }), "{}", mode.name());
        }
    }

    #[test]
    fn typed_content_hdc_resolves_only_canonical_in_scope_hits() {
        let case = generated_worlds(1).remove(0);
        let exact = run_candidate(
            CandidateMode::ExactScopeScan,
            case.query,
            case.decision_need,
            &case.memory,
        )
        .unwrap();
        let typed = run_candidate(
            CandidateMode::TypedContentHdc,
            case.query,
            case.decision_need,
            &case.memory,
        )
        .unwrap();

        assert_eq!(typed.input_sha256, exact.input_sha256);
        assert!(typed.retrieved_event_ids.len() <= exact.retrieved_event_ids.len());
        assert!(typed.retrieval_profile_sha256.is_some());
        assert!(typed.index_bytes > 0);
        assert!(typed.retrieved_event_ids.len() <= TYPED_HDC_TOP_K * 2);
        assert!(typed.retrieved_event_ids.iter().all(|event_id| {
            case.memory.iter().any(|item| {
                item.event_id == *event_id
                    && item.scope == case.query.scope
                    && case.decision_need.action_alternatives.contains(&item.action)
            })
        }));
        assert!(typed.decision.evidence_ids.iter().all(|event_id| {
            typed.retrieved_event_ids.contains(event_id)
        }));
    }

    #[test]
    fn typed_content_hdc_retrieval_is_recomputed_after_root_knockout() {
        let case = generated_worlds(1).remove(0);
        let full = run_candidate(
            CandidateMode::TypedContentHdc,
            case.query,
            case.decision_need,
            &case.memory,
        )
        .unwrap();
        let index = EvidenceDependencyIndex::build(&case.memory).unwrap();
        let first_root = index.root_groups().next().unwrap().1;
        let retained = index.retain_without(&first_root);
        let reduced = run_candidate(
            CandidateMode::TypedContentHdc,
            case.query,
            case.decision_need,
            &retained,
        )
        .unwrap();

        assert_ne!(full.evidence_sha256, reduced.evidence_sha256);
        assert!(reduced.retrieved_event_ids.iter().all(|event_id| {
            retained.iter().any(|item| item.event_id == *event_id)
        }));
        assert!(reduced.retrieved_event_ids.iter().all(|event_id| {
            !first_root.contains(event_id)
        }));
    }

    #[test]
    fn all_hdc_representations_rebuild_after_source_root_knockout() {
        let case = generated_worlds(1).remove(0);
        let dependency_index = EvidenceDependencyIndex::build(&case.memory).unwrap();
        let root_group = dependency_index.root_groups().next().unwrap().1;
        let retained = dependency_index.retain_without(&root_group);

        for mode in [
            CandidateMode::StaticHdc,
            CandidateMode::HdcLtcZeroInput,
            CandidateMode::TypedContentHdc,
        ] {
            let full = run_candidate(
                mode,
                case.query,
                case.decision_need,
                &case.memory,
            )
            .unwrap();
            let reduced = run_candidate(
                mode,
                case.query,
                case.decision_need,
                &retained,
            )
            .unwrap();

            assert_ne!(full.evidence_sha256, reduced.evidence_sha256, "{}", mode.name());
            assert!(reduced.retrieved_event_ids.iter().all(|event_id| {
                retained.iter().any(|item| item.event_id == *event_id)
            }), "{}", mode.name());
            assert!(reduced.retrieved_event_ids.iter().all(|event_id| {
                !root_group.contains(event_id)
            }), "{}", mode.name());
        }
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
