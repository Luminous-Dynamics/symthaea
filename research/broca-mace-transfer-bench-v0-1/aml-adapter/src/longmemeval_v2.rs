// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Text-only raw-trajectory BM25 baseline for the LongMemEval-V2 trajectory shape.
//! Screenshot references are retained, but image pixels are not decoded or embedded.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::raw_session::Bm25ProfileV1;

const TOKENIZER_VERSION: &str = "unicode_alphanumeric_v1";
const MAX_NEIGHBOR_RADIUS: usize = 32;
const MAX_TOP_K: usize = 100;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum V2TrajectoryReject {
    InvalidJson,
    TopLevelMustBeObject,
    MissingField(&'static str),
    InvalidField(&'static str),
    EmptyStates,
    InvalidState { index: usize, field: &'static str },
    DuplicateTrajectoryId,
    WrongDomain,
    DuplicateChunkId,
    DuplicateStateWindow,
    ChunkIntegrityMismatch,
    EmptyIndex,
    EmptyQuery,
    InvalidTopK,
    InvalidNeighborRadius,
    InvalidBm25Profile,
    DatasetRootUnavailable,
    ScreenshotPathInvalid,
    ScreenshotMissing,
    ScreenshotOutsideDatasetRoot,
    ScreenshotNotRegularFile,
    ScreenshotDigestMismatch,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LongMemEvalV2StateV1 {
    pub source_ordinal: usize,
    pub source_state_index: i64,
    pub step: Option<i64>,
    pub url: String,
    /// In this dataset the action is attached to the destination state.
    pub action_on_arrival: Option<String>,
    pub thoughts: Option<String>,
    pub accessibility_tree: String,
    /// Source path/reference only. Image pixels are not loaded by this module.
    pub screenshot_reference: String,
    /// Optional digest supplied by the controlled image-ingress adapter.
    pub screenshot_sha256: Option<String>,
    /// Canonical JSON serialization of the original state object, retaining unknown keys.
    pub raw_state_json: String,
    pub raw_state_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LongMemEvalV2TrajectoryV1 {
    pub id: String,
    pub domain: String,
    pub goal: String,
    pub start_url: String,
    pub outcome: Option<String>,
    pub states: Vec<LongMemEvalV2StateV1>,
    /// Exact JSONL record text retains unmodeled top-level metadata and original bytes.
    pub raw_source_json: String,
    pub raw_source_sha256: String,
}

impl LongMemEvalV2TrajectoryV1 {
    /// Parse one record from the official trajectories.jsonl file.
    /// Unknown source metadata remains available through raw_source_json/raw_state_json.
    pub fn parse_jsonl_record(line: &str) -> Result<Self, V2TrajectoryReject> {
        let value: Value = serde_json::from_str(line).map_err(|_| V2TrajectoryReject::InvalidJson)?;
        let object = value.as_object().ok_or(V2TrajectoryReject::TopLevelMustBeObject)?;
        let required = |field: &'static str| -> Result<String, V2TrajectoryReject> {
            object.get(field)
                .and_then(Value::as_str)
                .map(str::to_owned)
                .filter(|item| !item.trim().is_empty())
                .ok_or(V2TrajectoryReject::MissingField(field))
        };
        let id = required("id")?;
        let domain = required("domain")?;
        let goal = required("goal")?;
        let start_url = required("start_url")?;
        let outcome = match object.get("outcome") {
            None | Some(Value::Null) => None,
            Some(Value::String(value)) => Some(value.clone()),
            _ => return Err(V2TrajectoryReject::InvalidField("outcome")),
        };
        let public_states = object.get("states").and_then(Value::as_array)
            .ok_or(V2TrajectoryReject::MissingField("states"))?;
        if public_states.is_empty() {
            return Err(V2TrajectoryReject::EmptyStates);
        }

        let mut states = Vec::with_capacity(public_states.len());
        for (ordinal, value) in public_states.iter().enumerate() {
            let state = value.as_object().ok_or(V2TrajectoryReject::InvalidState {
                index: ordinal, field: "state",
            })?;
            let required_state = |field: &'static str| -> Result<String, V2TrajectoryReject> {
                state.get(field).and_then(Value::as_str).map(str::to_owned)
                    .filter(|item| !item.trim().is_empty())
                    .ok_or(V2TrajectoryReject::InvalidState { index: ordinal, field })
            };
            let url = required_state("url")?;
            let accessibility_tree = state.get("accessibility_tree").or_else(|| state.get("text"))
                .and_then(Value::as_str)
                .ok_or(V2TrajectoryReject::InvalidState { index: ordinal, field: "accessibility_tree/text" })?
                .to_owned();
            let screenshot_reference = required_state("screenshot")?;
            let screenshot_sha256 = match state.get("mace_screenshot_sha256") {
                None | Some(Value::Null) => None,
                Some(Value::String(value))
                    if value.len() == 64
                        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) =>
                {
                    Some(value.clone())
                }
                _ => return Err(V2TrajectoryReject::InvalidState {
                    index: ordinal, field: "mace_screenshot_sha256",
                }),
            };
            let action_on_arrival = optional_string(state.get("action"))
                .map_err(|_| V2TrajectoryReject::InvalidState { index: ordinal, field: "action" })?;
            let thoughts = optional_string(state.get("thought").or_else(|| state.get("thoughts")))
                .map_err(|_| V2TrajectoryReject::InvalidState { index: ordinal, field: "thought(s)" })?;
            let source_state_index = match state.get("state_index") {
                None | Some(Value::Null) => ordinal as i64,
                Some(value) => value.as_i64().ok_or(V2TrajectoryReject::InvalidState {
                    index: ordinal, field: "state_index",
                })?,
            };
            let step = match state.get("step") {
                None | Some(Value::Null) => None,
                Some(value) => Some(value.as_i64().ok_or(V2TrajectoryReject::InvalidState {
                    index: ordinal, field: "step",
                })?),
            };
            let raw_state_json = value.to_string();
            let raw_state_sha256 = sha256_hex(raw_state_json.as_bytes());
            states.push(LongMemEvalV2StateV1 {
                source_ordinal: ordinal, source_state_index, step, url,
                action_on_arrival, thoughts, accessibility_tree,
                screenshot_reference, screenshot_sha256, raw_state_json, raw_state_sha256,
            });
        }

        Ok(Self {
            id, domain, goal, start_url, outcome, states,
            raw_source_json: line.to_owned(),
            raw_source_sha256: sha256_hex(line.as_bytes()),
        })
    }

    pub fn evidence_chunks(
        &self,
        slice_radius: usize,
    ) -> Result<Vec<LongMemEvalV2EvidenceChunkV1>, V2TrajectoryReject> {
        if slice_radius > MAX_NEIGHBOR_RADIUS {
            return Err(V2TrajectoryReject::InvalidNeighborRadius);
        }
        let mut chunks = Vec::with_capacity(self.states.len());
        for center in 0..self.states.len() {
            let start = center.saturating_sub(slice_radius);
            let end = (center + slice_radius + 1).min(self.states.len());
            let state_window = self.states[start..end].to_vec();
            let evidence_text = render_evidence_text(
                &self.id, &self.goal, &self.start_url, start, &state_window,
            );
            let state_digests = state_window.iter()
                .map(|state| state.raw_state_sha256.clone()).collect::<Vec<_>>();
            let chunk_sha256 = chunk_digest(
                &self.id, &self.domain, &self.raw_source_sha256, center, start,
                &state_digests, &evidence_text,
            );
            let screenshot_references = state_window.iter()
                .map(|state| state.screenshot_reference.clone()).collect();
            chunks.push(LongMemEvalV2EvidenceChunkV1 {
                chunk_id: chunk_sha256.clone(),
                trajectory_id: self.id.clone(),
                domain: self.domain.clone(),
                goal: self.goal.clone(),
                start_url: self.start_url.clone(),
                center_state_ordinal: center,
                window_start_ordinal: start,
                window_end_ordinal_exclusive: end,
                source_trajectory_sha256: self.raw_source_sha256.clone(),
                state_window,
                screenshot_references,
                chunk_sha256,
                evidence_text,
            });
        }
        Ok(chunks)
    }
}

fn optional_string(value: Option<&Value>) -> Result<Option<String>, ()> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        _ => Err(()),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LongMemEvalV2EvidenceChunkV1 {
    pub chunk_id: String,
    pub trajectory_id: String,
    pub domain: String,
    pub goal: String,
    pub start_url: String,
    pub center_state_ordinal: usize,
    pub window_start_ordinal: usize,
    pub window_end_ordinal_exclusive: usize,
    pub source_trajectory_sha256: String,
    pub state_window: Vec<LongMemEvalV2StateV1>,
    pub screenshot_references: Vec<String>,
    pub chunk_sha256: String,
    evidence_text: String,
}

impl LongMemEvalV2EvidenceChunkV1 {
    pub fn evidence_text(&self) -> &str { &self.evidence_text }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LongMemEvalV2ResourceReceiptV1 {
    pub domain: String,
    pub corpus_sha256: String,
    pub trajectory_count: usize,
    pub state_chunk_count: usize,
    pub token_count: usize,
    pub vocabulary_size: usize,
    pub source_trajectory_json_bytes: u64,
    pub source_state_json_bytes: u64,
    pub source_accessibility_text_bytes: u64,
    pub screenshot_reference_bytes: u64,
    pub derived_evidence_text_bytes: u64,
    pub lexical_posting_payload_bytes_lower_bound: u64,
    pub vocabulary_payload_bytes_lower_bound: u64,
    pub total_payload_bytes_lower_bound: u64,
    pub index_build_latency_ns: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LongMemEvalV2ContextKind {
    Text,
    Image,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LongMemEvalV2ContextItemV1 {
    pub kind: LongMemEvalV2ContextKind,
    pub value: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LongMemEvalV2HitV1 {
    pub chunk_id: String,
    pub trajectory_id: String,
    pub center_state_ordinal: usize,
    pub score: f64,
    pub rank: usize,
    pub evidence_text: String,
    pub screenshot_references: Vec<String>,
    pub screenshot_sha256s: Vec<Option<String>>,
}

impl LongMemEvalV2HitV1 {
    /// Projection to the official V2 text/image context-item concept.
    /// Image items contain source paths; this module does not load image pixels.
    pub fn context_items(&self) -> Vec<LongMemEvalV2ContextItemV1> {
        let mut items = vec![LongMemEvalV2ContextItemV1 {
            kind: LongMemEvalV2ContextKind::Text,
            value: self.evidence_text.clone(),
        }];
        items.extend(self.screenshot_references.iter().map(|reference| {
            LongMemEvalV2ContextItemV1 {
                kind: LongMemEvalV2ContextKind::Image,
                value: reference.clone(),
            }
        }));
        items
    }

    /// Produce official-reader-compatible image paths after validating every
    /// reference against the prepared dataset root. Missing or escaping paths
    /// fail closed; a caller must not silently drop unavailable evidence.
    pub fn resolved_context_items(
        &self,
        dataset_root: &Path,
    ) -> Result<Vec<LongMemEvalV2ContextItemV1>, V2TrajectoryReject> {
        let mut items = vec![LongMemEvalV2ContextItemV1 {
            kind: LongMemEvalV2ContextKind::Text,
            value: self.evidence_text.clone(),
        }];
        if self.screenshot_references.len() != self.screenshot_sha256s.len() {
            return Err(V2TrajectoryReject::ChunkIntegrityMismatch);
        }
        for (reference, expected_digest) in self
            .screenshot_references
            .iter()
            .zip(&self.screenshot_sha256s)
        {
            let resolved = resolve_screenshot_reference(dataset_root, reference)?;
            if let Some(expected) = expected_digest {
                let image_bytes = fs::read(&resolved)
                    .map_err(|_| V2TrajectoryReject::ScreenshotMissing)?;
                if sha256_hex(&image_bytes) != *expected {
                    return Err(V2TrajectoryReject::ScreenshotDigestMismatch);
                }
            }
            items.push(LongMemEvalV2ContextItemV1 {
                kind: LongMemEvalV2ContextKind::Image,
                value: resolved.to_string_lossy().into_owned(),
            });
        }
        Ok(items)
    }
}

/// Resolve a dataset-relative screenshot reference to an existing canonical
/// file beneath the supplied dataset root. Reject absolute paths, traversal,
/// and symlinks that escape the dataset tree. Returned paths are absolute so
/// the official LongMemEval-V2 reader can open them without guessing the CWD.
pub fn resolve_screenshot_reference(
    dataset_root: &Path,
    reference: &str,
) -> Result<PathBuf, V2TrajectoryReject> {
    if reference.is_empty()
        || reference.trim() != reference
        || reference.contains('\\')
    {
        return Err(V2TrajectoryReject::ScreenshotPathInvalid);
    }
    let relative = Path::new(reference);
    if relative.is_absolute()
        || !relative
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(V2TrajectoryReject::ScreenshotPathInvalid);
    }

    let canonical_root = fs::canonicalize(dataset_root)
        .map_err(|_| V2TrajectoryReject::DatasetRootUnavailable)?;
    if !canonical_root.is_dir() {
        return Err(V2TrajectoryReject::DatasetRootUnavailable);
    }

    let resolved = fs::canonicalize(canonical_root.join(relative))
        .map_err(|_| V2TrajectoryReject::ScreenshotMissing)?;
    if !resolved.starts_with(&canonical_root) {
        return Err(V2TrajectoryReject::ScreenshotOutsideDatasetRoot);
    }
    if !resolved.is_file() {
        return Err(V2TrajectoryReject::ScreenshotNotRegularFile);
    }
    Ok(resolved)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LongMemEvalV2NeighborV1 {
    pub chunk: LongMemEvalV2EvidenceChunkV1,
    pub expanded_from_hit_rank: usize,
    pub relative_state_offset: i32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LongMemEvalV2SearchResultV1 {
    pub hits: Vec<LongMemEvalV2HitV1>,
    pub neighbor_context: Vec<LongMemEvalV2NeighborV1>,
    pub search_latency_ns: u64,
}

#[derive(Clone, Debug)]
struct IndexedChunk {
    chunk: LongMemEvalV2EvidenceChunkV1,
    terms: BTreeMap<String, usize>,
    token_count: usize,
}

#[derive(Clone, Debug)]
pub struct LongMemEvalV2Bm25IndexV1 {
    domain: String,
    profile: Bm25ProfileV1,
    chunks: Vec<IndexedChunk>,
    positions: BTreeMap<(String, usize), usize>,
    document_frequency: BTreeMap<String, usize>,
    average_document_length: f64,
    receipt: LongMemEvalV2ResourceReceiptV1,
}

impl LongMemEvalV2Bm25IndexV1 {
    /// Build one domain-specific corpus. Web and enterprise data cannot mix.
    pub fn build(
        domain: impl Into<String>,
        trajectories: &[LongMemEvalV2TrajectoryV1],
        slice_radius: usize,
        profile: Bm25ProfileV1,
    ) -> Result<Self, V2TrajectoryReject> {
        let started = Instant::now();
        let domain = domain.into();
        if domain.trim().is_empty() || trajectories.is_empty() {
            return Err(V2TrajectoryReject::EmptyIndex);
        }
        if profile.tokenizer_version != TOKENIZER_VERSION
            || !profile.k1.is_finite() || profile.k1 <= 0.0
            || !profile.b.is_finite() || !(0.0..=1.0).contains(&profile.b)
        {
            return Err(V2TrajectoryReject::InvalidBm25Profile);
        }
        if slice_radius > MAX_NEIGHBOR_RADIUS {
            return Err(V2TrajectoryReject::InvalidNeighborRadius);
        }

        let mut ids = BTreeSet::new();
        let mut chunks = Vec::new();
        for trajectory in trajectories {
            if trajectory.domain != domain {
                return Err(V2TrajectoryReject::WrongDomain);
            }
            if !ids.insert(trajectory.id.as_str()) {
                return Err(V2TrajectoryReject::DuplicateTrajectoryId);
            }
            chunks.extend(trajectory.evidence_chunks(slice_radius)?);
        }
        chunks.sort_by(|a, b| a.trajectory_id.cmp(&b.trajectory_id)
            .then_with(|| a.center_state_ordinal.cmp(&b.center_state_ordinal))
            .then_with(|| a.chunk_id.cmp(&b.chunk_id)));

        let source_trajectory_json_bytes = trajectories.iter().fold(0u64, |total, item| {
            total.saturating_add(item.raw_source_json.len() as u64)
        });
        let mut source_state_json_bytes = 0u64;
        let mut source_accessibility_text_bytes = 0u64;
        let mut screenshot_reference_bytes = 0u64;
        let mut derived_evidence_text_bytes = 0u64;
        let mut posting_bytes = 0u64;
        let mut total_tokens = 0usize;
        let mut df = BTreeMap::<String, usize>::new();
        let mut positions = BTreeMap::new();
        let mut seen = BTreeSet::new();
        let mut index = Vec::with_capacity(chunks.len());
        let mut hasher = Sha256::new();
        hasher.update(b"luminous:mace:longmemeval-v2-bm25:v1\0");
        hash_field(&mut hasher, &domain);
        hash_field(&mut hasher, profile.tokenizer_version);
        hasher.update(profile.k1.to_bits().to_be_bytes());
        hasher.update(profile.b.to_bits().to_be_bytes());
        hasher.update((slice_radius as u64).to_be_bytes());

        for chunk in chunks {
            validate_chunk(&chunk)?;
            if !seen.insert(chunk.chunk_id.clone()) {
                return Err(V2TrajectoryReject::DuplicateChunkId);
            }
            if positions.insert((chunk.trajectory_id.clone(), chunk.center_state_ordinal), index.len()).is_some() {
                return Err(V2TrajectoryReject::DuplicateStateWindow);
            }
            source_state_json_bytes = source_state_json_bytes.saturating_add(
                chunk.state_window.iter().fold(0u64, |total, state| {
                    total.saturating_add(state.raw_state_json.len() as u64)
                }),
            );
            source_accessibility_text_bytes = source_accessibility_text_bytes.saturating_add(
                chunk.state_window.iter().fold(0u64, |total, state| {
                    total.saturating_add(state.accessibility_tree.len() as u64)
                }),
            );
            screenshot_reference_bytes = screenshot_reference_bytes.saturating_add(
                chunk.screenshot_references.iter().fold(0u64, |total, path| {
                    total.saturating_add(path.len() as u64)
                }),
            );
            derived_evidence_text_bytes =
                derived_evidence_text_bytes.saturating_add(chunk.evidence_text.len() as u64);

            let tokens = tokenize(&chunk.evidence_text);
            let token_count = tokens.len();
            total_tokens = total_tokens.saturating_add(token_count);
            let mut terms = BTreeMap::<String, usize>::new();
            for token in tokens { *terms.entry(token).or_default() += 1; }
            for term in terms.keys() {
                *df.entry(term.clone()).or_default() += 1;
                posting_bytes = posting_bytes.saturating_add(term.len() as u64).saturating_add(16);
            }
            hash_field(&mut hasher, &chunk.chunk_id);
            hash_field(&mut hasher, &chunk.chunk_sha256);
            index.push(IndexedChunk {
                chunk,
                terms,
                token_count,
            });
        }

        let average_document_length = if index.is_empty() { 0.0 } else {
            total_tokens as f64 / index.len() as f64
        };
        let vocabulary_size = df.len();
        let vocabulary_bytes = df.keys().fold(0u64, |total, term| {
            total.saturating_add(term.len() as u64).saturating_add(8)
        });
        let corpus_sha256 = hex_digest(hasher.finalize().as_slice());
        let total_payload_bytes_lower_bound = source_trajectory_json_bytes
            .saturating_add(source_state_json_bytes)
            .saturating_add(source_accessibility_text_bytes)
            .saturating_add(screenshot_reference_bytes)
            .saturating_add(derived_evidence_text_bytes)
            .saturating_add(posting_bytes)
            .saturating_add(vocabulary_bytes);
        let receipt = LongMemEvalV2ResourceReceiptV1 {
            domain: domain.clone(),
            corpus_sha256,
            trajectory_count: ids.len(),
            state_chunk_count: index.len(),
            token_count: total_tokens,
            vocabulary_size,
            source_trajectory_json_bytes,
            source_state_json_bytes,
            source_accessibility_text_bytes,
            screenshot_reference_bytes,
            derived_evidence_text_bytes,
            lexical_posting_payload_bytes_lower_bound: posting_bytes,
            vocabulary_payload_bytes_lower_bound: vocabulary_bytes,
            total_payload_bytes_lower_bound,
            index_build_latency_ns: elapsed_ns(started),
        };
        Ok(Self {
            domain, profile, chunks: index, positions, document_frequency: df,
            average_document_length, receipt,
        })
    }

    pub fn resource_receipt(&self) -> &LongMemEvalV2ResourceReceiptV1 { &self.receipt }

    pub fn search(
        &self,
        domain: &str,
        query: &str,
        top_k: usize,
        neighbor_radius: usize,
    ) -> Result<LongMemEvalV2SearchResultV1, V2TrajectoryReject> {
        let started = Instant::now();
        if domain != self.domain { return Err(V2TrajectoryReject::WrongDomain); }
        if top_k == 0 || top_k > MAX_TOP_K { return Err(V2TrajectoryReject::InvalidTopK); }
        if neighbor_radius > MAX_NEIGHBOR_RADIUS { return Err(V2TrajectoryReject::InvalidNeighborRadius); }
        let query_terms: BTreeSet<String> = tokenize(query).into_iter().collect();
        if query_terms.is_empty() { return Err(V2TrajectoryReject::EmptyQuery); }
        let n = self.chunks.len() as f64;
        let mut hits = Vec::new();
        for item in &self.chunks {
            if item.token_count == 0 || self.average_document_length == 0.0 { continue; }
            let mut score = 0.0f64;
            for term in &query_terms {
                let Some(tf_value) = item.terms.get(term) else { continue; };
                let df = *self.document_frequency.get(term).unwrap_or(&0) as f64;
                let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
                let tf = *tf_value as f64;
                let len = item.token_count as f64;
                let denom = tf + self.profile.k1 *
                    (1.0 - self.profile.b + self.profile.b * len / self.average_document_length);
                score += idf * (tf * (self.profile.k1 + 1.0)) / denom;
            }
            if score > 0.0 {
                hits.push(LongMemEvalV2HitV1 {
                    chunk_id: item.chunk.chunk_id.clone(),
                    trajectory_id: item.chunk.trajectory_id.clone(),
                    center_state_ordinal: item.chunk.center_state_ordinal,
                    score, rank: 0,
                    evidence_text: item.chunk.evidence_text.clone(),
                    screenshot_references: item.chunk.screenshot_references.clone(),
                    screenshot_sha256s: item.chunk.state_window.iter()
                        .map(|state| state.screenshot_sha256.clone()).collect(),
                });
            }
        }
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.chunk_id.cmp(&b.chunk_id)));
        hits.truncate(top_k);
        for (i, hit) in hits.iter_mut().enumerate() { hit.rank = i + 1; }

        let mut neighbors = BTreeMap::<String, LongMemEvalV2NeighborV1>::new();
        for hit in &hits {
            for offset in -(neighbor_radius as i64)..=(neighbor_radius as i64) {
                if offset == 0 { continue; }
                let ordinal = if offset < 0 {
                    hit.center_state_ordinal.checked_sub(offset.unsigned_abs() as usize)
                } else {
                    hit.center_state_ordinal.checked_add(offset as usize)
                };
                let Some(ordinal) = ordinal else { continue; };
                let Some(pos) = self.positions.get(&(hit.trajectory_id.clone(), ordinal)) else { continue; };
                let chunk = &self.chunks[*pos].chunk;
                if hits.iter().any(|candidate| candidate.chunk_id == chunk.chunk_id) { continue; }
                neighbors.entry(chunk.chunk_id.clone()).or_insert_with(|| LongMemEvalV2NeighborV1 {
                    chunk: chunk.clone(),
                    expanded_from_hit_rank: hit.rank,
                    relative_state_offset: offset as i32,
                });
            }
        }
        let mut neighbor_context: Vec<_> = neighbors.into_values().collect();
        neighbor_context.sort_by(|a, b| a.expanded_from_hit_rank.cmp(&b.expanded_from_hit_rank)
            .then_with(|| a.chunk.trajectory_id.cmp(&b.chunk.trajectory_id))
            .then_with(|| a.chunk.center_state_ordinal.cmp(&b.chunk.center_state_ordinal)));
        Ok(LongMemEvalV2SearchResultV1 {
            hits, neighbor_context, search_latency_ns: elapsed_ns(started),
        })
    }
}

fn validate_chunk(chunk: &LongMemEvalV2EvidenceChunkV1) -> Result<(), V2TrajectoryReject> {
    if chunk.trajectory_id.trim().is_empty()
        || chunk.domain.trim().is_empty()
        || chunk.goal.trim().is_empty()
        || chunk.start_url.trim().is_empty()
        || chunk.source_trajectory_sha256.len() != 64
        || chunk.state_window.is_empty()
        || chunk.window_start_ordinal > chunk.center_state_ordinal
        || chunk.window_end_ordinal_exclusive <= chunk.center_state_ordinal
        || chunk.window_end_ordinal_exclusive - chunk.window_start_ordinal != chunk.state_window.len()
        || chunk.screenshot_references.len() != chunk.state_window.len()
    { return Err(V2TrajectoryReject::ChunkIntegrityMismatch); }
    for (i, state) in chunk.state_window.iter().enumerate() {
        if state.source_ordinal != chunk.window_start_ordinal + i
            || state.raw_state_sha256 != sha256_hex(state.raw_state_json.as_bytes())
            || state.url.trim().is_empty()
            || state.screenshot_reference.trim().is_empty()
            || chunk.screenshot_references[i] != state.screenshot_reference
        { return Err(V2TrajectoryReject::ChunkIntegrityMismatch); }
        validate_state_projection(state)?;
    }
    let rendered = render_evidence_text(&chunk.trajectory_id, &chunk.goal, &chunk.start_url,
        chunk.window_start_ordinal, &chunk.state_window);
    if rendered != chunk.evidence_text { return Err(V2TrajectoryReject::ChunkIntegrityMismatch); }
    let state_hashes = chunk.state_window.iter().map(|state| state.raw_state_sha256.clone()).collect::<Vec<_>>();
    let digest = chunk_digest(&chunk.trajectory_id, &chunk.domain, &chunk.source_trajectory_sha256,
        chunk.center_state_ordinal, chunk.window_start_ordinal, &state_hashes, &chunk.evidence_text);
    if digest != chunk.chunk_sha256 || digest != chunk.chunk_id {
        return Err(V2TrajectoryReject::ChunkIntegrityMismatch);
    }
    Ok(())
}

fn validate_state_projection(state: &LongMemEvalV2StateV1) -> Result<(), V2TrajectoryReject> {
    let value: Value = serde_json::from_str(&state.raw_state_json)
        .map_err(|_| V2TrajectoryReject::ChunkIntegrityMismatch)?;
    let object = value.as_object().ok_or(V2TrajectoryReject::ChunkIntegrityMismatch)?;
    if object.get("url").and_then(Value::as_str) != Some(state.url.as_str())
        || object.get("screenshot").and_then(Value::as_str) != Some(state.screenshot_reference.as_str())
        || object.get("accessibility_tree").or_else(|| object.get("text")).and_then(Value::as_str)
            != Some(state.accessibility_tree.as_str())
    { return Err(V2TrajectoryReject::ChunkIntegrityMismatch); }
    let action = optional_string(object.get("action")).map_err(|_| V2TrajectoryReject::ChunkIntegrityMismatch)?;
    let thought = optional_string(object.get("thought").or_else(|| object.get("thoughts")))
        .map_err(|_| V2TrajectoryReject::ChunkIntegrityMismatch)?;
    let source_index = match object.get("state_index") {
        None | Some(Value::Null) => state.source_ordinal as i64,
        Some(value) => value.as_i64().ok_or(V2TrajectoryReject::ChunkIntegrityMismatch)?,
    };
    let step = match object.get("step") {
        None | Some(Value::Null) => None,
        Some(value) => Some(value.as_i64().ok_or(V2TrajectoryReject::ChunkIntegrityMismatch)?),
    };
    let screenshot_sha256 = match object.get("mace_screenshot_sha256") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) if value.len() == 64
            && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) =>
        {
            Some(value.clone())
        }
        _ => return Err(V2TrajectoryReject::ChunkIntegrityMismatch),
    };
    if action != state.action_on_arrival || thought != state.thoughts
        || source_index != state.source_state_index || step != state.step
        || screenshot_sha256 != state.screenshot_sha256
    { return Err(V2TrajectoryReject::ChunkIntegrityMismatch); }
    Ok(())
}

fn render_evidence_text(
    trajectory_id: &str,
    goal: &str,
    start_url: &str,
    window_start: usize,
    states: &[LongMemEvalV2StateV1],
) -> String {
    let mut lines = vec![
        format!("Trajectory: {trajectory_id}"),
        format!("Goal: {goal}"),
        format!("Start URL: {start_url}"),
        format!("State window starts at: {window_start}"),
    ];
    for state in states {
        lines.push(format!("State ordinal: {}", state.source_ordinal));
        lines.push(format!("Source state index: {}", state.source_state_index));
        if let Some(step) = state.step { lines.push(format!("Step: {step}")); }
        lines.push(format!("URL: {}", state.url));
        if state.source_ordinal > 0 {
            lines.push(format!(
                "Action attached to destination state (transition from prior state): {}",
                state.action_on_arrival.as_deref().unwrap_or("<none recorded>")
            ));
        }
        if let Some(thoughts) = &state.thoughts {
            if !thoughts.trim().is_empty() { lines.push(format!("Thought: {thoughts}")); }
        }
        lines.push("Accessibility tree:".to_owned());
        lines.push(state.accessibility_tree.clone());
    }
    lines.join("\n")
}

fn chunk_digest(
    trajectory_id: &str,
    domain: &str,
    source_sha: &str,
    center: usize,
    start: usize,
    state_hashes: &[String],
    evidence: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:lmev2-evidence-chunk:v1\0");
    hash_field(&mut hasher, trajectory_id);
    hash_field(&mut hasher, domain);
    hash_field(&mut hasher, source_sha);
    hasher.update((center as u64).to_be_bytes());
    hasher.update((start as u64).to_be_bytes());
    for hash in state_hashes { hash_field(&mut hasher, hash); }
    hash_field(&mut hasher, evidence);
    hex_digest(hasher.finalize().as_slice())
}

fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() { current.extend(ch.to_lowercase()); }
        else if !current.is_empty() { tokens.push(std::mem::take(&mut current)); }
    }
    if !current.is_empty() { tokens.push(current); }
    tokens
}

fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_digest(Sha256::digest(bytes).as_slice())
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[usize::from(byte >> 4)] as char);
        out.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(id: &str, domain: &str) -> String {
        serde_json::json!({
            "id": id, "domain": domain, "goal": "Create a new customer order",
            "start_url": "https://shop.example/", "outcome": "success",
            "opaque_metadata": {"keep": true},
            "states": [
                {"url":"https://shop.example/orders","action":null,"thoughts":"Open order page",
                 "accessibility_tree":"Orders page. New order button.","screenshot":"screenshots/traj/000.png",
                 "state_index":10,"step":0,"opaque_state_field":"kept"},
                {"url":"https://shop.example/orders/new","action":"click New order","thought":"Open creation form",
                 "text":"Create order form. Customer field. Submit button.",
                 "screenshot":"screenshots/traj/001.png","state_index":20,"step":1},
                {"url":"https://shop.example/orders/42","action":"fill customer and submit","thoughts":"Order submitted",
                 "accessibility_tree":"Order details. Customer: Ada. Status: pending.",
                 "screenshot":"screenshots/traj/002.png","state_index":30,"step":2}
            ]
        }).to_string()
    }

    fn parse(id: &str) -> LongMemEvalV2TrajectoryV1 {
        LongMemEvalV2TrajectoryV1::parse_jsonl_record(&sample(id, "web")).expect("valid")
    }

    fn profile() -> Bm25ProfileV1 {
        Bm25ProfileV1 { k1: 1.2, b: 0.75, tokenizer_version: TOKENIZER_VERSION }
    }

    #[test]
    fn parser_preserves_raw_metadata_and_screenshot_references() {
        let raw = sample("t1", "web");
        let parsed = LongMemEvalV2TrajectoryV1::parse_jsonl_record(&raw).expect("parsed");
        assert_eq!(parsed.raw_source_json, raw);
        assert!(parsed.raw_source_json.contains("opaque_metadata"));
        assert!(parsed.states[0].raw_state_json.contains("opaque_state_field"));
        assert_eq!(parsed.states[1].accessibility_tree, "Create order form. Customer field. Submit button.");
        assert_eq!(parsed.states[2].screenshot_reference, "screenshots/traj/002.png");
    }

    #[test]
    fn action_attached_to_destination_state_is_not_shifted_backward() {
        let chunks = parse("t1").evidence_chunks(0).expect("chunks");
        assert!(!chunks[0].evidence_text.contains("click New order"));
        assert!(chunks[1].evidence_text.contains("Action attached to destination state"));
        assert!(chunks[1].evidence_text.contains("click New order"));
    }

    #[test]
    fn windows_preserve_state_and_screenshot_boundaries() {
        let chunks = parse("t1").evidence_chunks(1).expect("chunks");
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].state_window.len(), 2);
        assert_eq!(chunks[1].state_window.len(), 3);
        assert_eq!(chunks[2].state_window.len(), 2);
        assert_eq!(chunks[1].screenshot_references.len(), 3);
        assert_eq!(chunks[1].state_window[1].source_state_index, 20);
    }

    #[test]
    fn bm25_searches_raw_accessibility_evidence_and_emits_costs() {
        let trajectories = vec![parse("t1")];
        let index = LongMemEvalV2Bm25IndexV1::build("web", &trajectories, 0, profile()).expect("index");
        let result = index.search("web", "customer Ada pending order", 5, 0).expect("search");
        assert!(!result.hits.is_empty());
        assert!(result.hits[0].evidence_text.contains("Customer: Ada"));
        let context = result.hits[0].context_items();
        assert_eq!(context[0].kind, LongMemEvalV2ContextKind::Text);
        assert!(context.iter().any(|item| item.kind == LongMemEvalV2ContextKind::Image));
        assert!(result.search_latency_ns > 0);
        assert!(index.resource_receipt().total_payload_bytes_lower_bound > 0);
    }

    #[test]
    fn web_and_enterprise_trajectories_cannot_share_an_index() {
        let enterprise_raw = sample("t2", "enterprise");
        let enterprise = LongMemEvalV2TrajectoryV1::parse_jsonl_record(&enterprise_raw).expect("parsed");
        assert_eq!(
            LongMemEvalV2Bm25IndexV1::build("web", &[parse("t1"), enterprise], 0, profile()).map(|_| ()),
            Err(V2TrajectoryReject::WrongDomain)
        );
    }

    #[test]
    fn neighbor_expansion_never_crosses_trajectory_boundaries() {
        let trajectories = vec![parse("t1"), parse("t2")];
        let index = LongMemEvalV2Bm25IndexV1::build("web", &trajectories, 0, profile()).expect("index");
        let result = index.search("web", "customer Ada pending order", 1, 1).expect("search");
        assert_eq!(result.hits.len(), 1);
        assert!(result.neighbor_context.iter().all(|neighbor|
            neighbor.chunk.trajectory_id == result.hits[0].trajectory_id));
    }

    #[test]
    fn source_mutation_changes_trajectory_and_chunk_digest() {
        let original = parse("t1");
        let altered_json = sample("t1", "web").replace("Customer: Ada", "Customer: Grace");
        let altered = LongMemEvalV2TrajectoryV1::parse_jsonl_record(&altered_json).expect("altered");
        assert_ne!(original.raw_source_sha256, altered.raw_source_sha256);
        assert_ne!(
            original.evidence_chunks(0).expect("chunks")[2].chunk_id,
            altered.evidence_chunks(0).expect("chunks")[2].chunk_id
        );
    }

    #[test]
    fn missing_screenshot_and_empty_state_list_fail_closed() {
        let raw = sample("t1", "web").replace(
            "\"screenshot\":\"screenshots/traj/000.png\"",
            "\"screenshot\":null",
        );
        assert!(matches!(
            LongMemEvalV2TrajectoryV1::parse_jsonl_record(&raw),
            Err(V2TrajectoryReject::InvalidState { field: "screenshot", .. })
        ));
        let empty = serde_json::json!({
            "id":"empty","domain":"web","goal":"goal","start_url":"https://example.com","states":[]
        }).to_string();
        assert_eq!(
            LongMemEvalV2TrajectoryV1::parse_jsonl_record(&empty),
            Err(V2TrajectoryReject::EmptyStates)
        );
    }

    #[test]
    fn tampered_chunk_metadata_is_rejected() {
        let mut chunk = parse("t1").evidence_chunks(0).expect("chunks").remove(0);
        chunk.screenshot_references[0] = "forged.png".into();
        assert_eq!(validate_chunk(&chunk), Err(V2TrajectoryReject::ChunkIntegrityMismatch));
        let mut changed_state = parse("t1").evidence_chunks(0).expect("chunks").remove(0);
        changed_state.state_window[0].url = "https://attacker.invalid/".into();
        assert_eq!(validate_chunk(&changed_state), Err(V2TrajectoryReject::ChunkIntegrityMismatch));
    }

    #[test]
    fn duplicate_trajectory_ids_and_unimplemented_tokenizers_are_rejected() {
        let duplicate = vec![parse("same"), parse("same")];
        assert_eq!(
            LongMemEvalV2Bm25IndexV1::build("web", &duplicate, 0, profile()).map(|_| ()),
            Err(V2TrajectoryReject::DuplicateTrajectoryId)
        );
        let invalid = Bm25ProfileV1 { k1: 1.2, b: 0.75, tokenizer_version: "not-implemented" };
        assert_eq!(
            LongMemEvalV2Bm25IndexV1::build("web", &[parse("one")], 0, invalid).map(|_| ()),
            Err(V2TrajectoryReject::InvalidBm25Profile)
        );
    }

    fn temporary_dataset_root(label: &str) -> PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "mace-lmev2-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("temporary dataset root");
        path
    }

    #[test]
    fn screenshot_resolution_returns_canonical_existing_file() {
        let root = temporary_dataset_root("resolved");
        let relative = Path::new("screenshots/traj/000.png");
        let file = root.join(relative);
        fs::create_dir_all(file.parent().expect("parent")).expect("screenshots directory");
        fs::write(&file, b"fixture screenshot bytes").expect("screenshot fixture");

        let resolved = resolve_screenshot_reference(&root, "screenshots/traj/000.png")
            .expect("resolved screenshot");
        assert_eq!(resolved, fs::canonicalize(file).expect("canonical fixture"));
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn screenshot_resolution_rejects_missing_absolute_and_parent_paths() {
        let root = temporary_dataset_root("reject-paths");
        assert_eq!(
            resolve_screenshot_reference(&root, "screenshots/missing.png"),
            Err(V2TrajectoryReject::ScreenshotMissing)
        );
        assert_eq!(
            resolve_screenshot_reference(&root, "../outside.png"),
            Err(V2TrajectoryReject::ScreenshotPathInvalid)
        );
        assert_eq!(
            resolve_screenshot_reference(&root, "/tmp/outside.png"),
            Err(V2TrajectoryReject::ScreenshotPathInvalid)
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[cfg(unix)]
    #[test]
    fn screenshot_resolution_rejects_symlinks_that_escape_dataset_root() {
        use std::os::unix::fs::symlink;
        let root = temporary_dataset_root("symlink-root");
        let outside_root = temporary_dataset_root("symlink-outside");
        let outside = outside_root.join("outside.png");
        fs::write(&outside, b"outside fixture").expect("outside file");
        let inside_dir = root.join("screenshots");
        fs::create_dir_all(&inside_dir).expect("inside directory");
        symlink(&outside, inside_dir.join("escape.png")).expect("symlink fixture");

        assert_eq!(
            resolve_screenshot_reference(&root, "screenshots/escape.png"),
            Err(V2TrajectoryReject::ScreenshotOutsideDatasetRoot)
        );
        fs::remove_dir_all(root).expect("cleanup root");
        fs::remove_dir_all(outside_root).expect("cleanup outside");
    }

    #[test]
    fn resolved_context_items_keep_text_and_return_absolute_image_paths() {
        let root = temporary_dataset_root("context");
        let image = root.join("screenshots/traj/002.png");
        fs::create_dir_all(image.parent().expect("image parent")).expect("image directory");
        fs::write(&image, b"fixture screenshot bytes").expect("image fixture");

        let index = LongMemEvalV2Bm25IndexV1::build("web", &[parse("t1")], 0, profile())
            .expect("index");
        let result = index
            .search("web", "customer Ada pending order", 1, 0)
            .expect("search");
        let items = result.hits[0]
            .resolved_context_items(&root)
            .expect("resolved context");
        assert_eq!(items[0].kind, LongMemEvalV2ContextKind::Text);
        assert_eq!(items[0].value, result.hits[0].evidence_text);
        assert!(items.iter().skip(1).all(|item| {
            item.kind == LongMemEvalV2ContextKind::Image
                && Path::new(&item.value).is_absolute()
                && Path::new(&item.value).is_file()
        }));
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn resolved_context_items_verify_optional_screenshot_pixel_digest() {
        let root = temporary_dataset_root("context-digest");
        let image = root.join("screenshots/traj/000.png");
        fs::create_dir_all(image.parent().expect("image parent")).expect("image directory");
        let image_bytes = b"bound screenshot bytes";
        fs::write(&image, image_bytes).expect("image fixture");

        let mut value: Value = serde_json::from_str(&sample("t1", "web")).expect("sample JSON");
        value["states"][0]["mace_screenshot_sha256"] =
            serde_json::json!(sha256_hex(image_bytes));
        let trajectory = LongMemEvalV2TrajectoryV1::parse_jsonl_record(
            &value.to_string()
        ).expect("trajectory with screenshot digest");
        let index = LongMemEvalV2Bm25IndexV1::build("web", &[trajectory], 0, profile())
            .expect("index");
        let result = index.search("web", "Orders page New order button", 1, 0)
            .expect("search");
        assert_eq!(
            result.hits[0].resolved_context_items(&root).map(|_| ()),
            Ok(())
        );

        fs::write(&image, b"replaced screenshot bytes").expect("replace fixture");
        assert_eq!(
            result.hits[0].resolved_context_items(&root),
            Err(V2TrajectoryReject::ScreenshotDigestMismatch)
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn resolved_context_items_fail_closed_when_any_screenshot_is_missing() {
        let root = temporary_dataset_root("context-missing");
        let index = LongMemEvalV2Bm25IndexV1::build("web", &[parse("t1")], 0, profile())
            .expect("index");
        let result = index
            .search("web", "customer Ada pending order", 1, 0)
            .expect("search");
        assert_eq!(
            result.hits[0].resolved_context_items(&root),
            Err(V2TrajectoryReject::ScreenshotMissing)
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn retrieval_rejects_empty_query_wrong_domain_and_invalid_budgets() {
        let index = LongMemEvalV2Bm25IndexV1::build("web", &[parse("one")], 0, profile()).expect("index");
        assert_eq!(index.search("enterprise", "anything", 5, 0), Err(V2TrajectoryReject::WrongDomain));
        assert_eq!(index.search("web", "!!!", 5, 0), Err(V2TrajectoryReject::EmptyQuery));
        assert_eq!(index.search("web", "order", 0, 0), Err(V2TrajectoryReject::InvalidTopK));
        assert_eq!(index.search("web", "order", 5, 99), Err(V2TrajectoryReject::InvalidNeighborRadius));
    }
}
