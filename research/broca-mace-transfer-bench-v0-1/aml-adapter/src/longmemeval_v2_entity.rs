// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exact structured-literal retrieval for LongMemEval-V2 trajectories.
//!
//! This channel complements BM25 rather than replacing it. It indexes stable
//! identifiers, URLs, numeric state/step values and other structured fields.
//! It intentionally ignores ordinary prose tokens: generic semantic recall
//! remains the responsibility of lexical/dense/other channels.
//!
//! The index emits source-backed trajectory chunks with deterministic ranks.
//! It does not qualify evidence, infer semantic truth or authorize actions.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use sha2::{Digest, Sha256};

use crate::longmemeval_v2::{
    LongMemEvalV2EvidenceChunkV1, LongMemEvalV2HitV1, LongMemEvalV2TrajectoryV1,
    V2TrajectoryReject,
};

const MAX_TOP_K: usize = 100;
const INDEX_DOMAIN: &[u8] = b"luminous:mace:lmev2-entity-literal-index:v1\0";

#[derive(Clone, Debug)]
struct IndexedEntityChunkV1 {
    chunk: LongMemEvalV2EvidenceChunkV1,
    literals: BTreeMap<String, u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LongMemEvalV2EntitySearchResultV1 {
    pub hits: Vec<LongMemEvalV2HitV1>,
    pub search_latency_ns: u64,
}

#[derive(Clone, Debug)]
pub struct LongMemEvalV2EntityLiteralIndexV1 {
    domain: String,
    chunks: Vec<IndexedEntityChunkV1>,
    postings: BTreeMap<String, Vec<(usize, u32)>>,
    corpus_sha256: String,
}

impl LongMemEvalV2EntityLiteralIndexV1 {
    /// Build a domain-isolated index over structured trajectory fields.
    ///
    /// Slice radius must match the BM25 index so both channels identify the
    /// same source windows and can be fused by immutable chunk ID.
    pub fn build(
        domain: impl Into<String>,
        trajectories: &[LongMemEvalV2TrajectoryV1],
        slice_radius: usize,
    ) -> Result<Self, V2TrajectoryReject> {
        let domain = domain.into();
        if domain.trim().is_empty() || trajectories.is_empty() {
            return Err(V2TrajectoryReject::EmptyIndex);
        }

        let mut seen_trajectory_ids = BTreeSet::new();
        let mut chunks = Vec::new();
        let mut hasher = Sha256::new();
        hasher.update(INDEX_DOMAIN);
        hash_field(&mut hasher, &domain);
        hasher.update((slice_radius as u64).to_be_bytes());

        for trajectory in trajectories {
            if trajectory.domain != domain {
                return Err(V2TrajectoryReject::WrongDomain);
            }
            if !seen_trajectory_ids.insert(trajectory.id.as_str()) {
                return Err(V2TrajectoryReject::DuplicateTrajectoryId);
            }

            for chunk in trajectory.evidence_chunks(slice_radius)? {
                let literals = structured_literals(&chunk);
                hash_field(&mut hasher, &chunk.chunk_id);
                hasher.update((literals.len() as u64).to_be_bytes());
                for (literal, weight) in &literals {
                    hash_field(&mut hasher, literal);
                    hasher.update(weight.to_be_bytes());
                }
                chunks.push(IndexedEntityChunkV1 { chunk, literals });
            }
        }

        chunks.sort_by(|left, right| left.chunk.chunk_id.cmp(&right.chunk.chunk_id));
        let mut postings = BTreeMap::<String, Vec<(usize, u32)>>::new();
        for (document_index, document) in chunks.iter().enumerate() {
            for (literal, weight) in &document.literals {
                postings.entry(literal.clone())
                    .or_default()
                    .push((document_index, *weight));
            }
        }
        let corpus_sha256 = hex_digest(hasher.finalize().as_slice());

        Ok(Self { domain, chunks, postings, corpus_sha256 })
    }

    pub fn corpus_sha256(&self) -> &str {
        &self.corpus_sha256
    }

    pub fn document_count(&self) -> usize {
        self.chunks.len()
    }

    /// Search exact structured literals. Generic words intentionally return
    /// no results here so this channel does not merely duplicate BM25.
    pub fn search(
        &self,
        domain: &str,
        query: &str,
        top_k: usize,
    ) -> Result<LongMemEvalV2EntitySearchResultV1, V2TrajectoryReject> {
        let started = Instant::now();
        if domain != self.domain {
            return Err(V2TrajectoryReject::WrongDomain);
        }
        if query.trim().is_empty() {
            return Err(V2TrajectoryReject::EmptyQuery);
        }
        if top_k == 0 || top_k > MAX_TOP_K {
            return Err(V2TrajectoryReject::InvalidTopK);
        }

        let query_literals = query_literals(query);
        let mut scores = BTreeMap::<usize, u64>::new();
        for literal in query_literals {
            if let Some(matches) = self.postings.get(&literal) {
                for (document_index, field_weight) in matches {
                    let score = scores.entry(*document_index).or_default();
                    *score = score.saturating_add(u64::from(*field_weight));
                }
            }
        }

        let mut ranked = scores.into_iter()
            .filter(|(_, score)| *score > 0)
            .collect::<Vec<_>>();
        ranked.sort_by(|(left_index, left_score), (right_index, right_score)| {
            right_score.cmp(left_score)
                .then_with(|| {
                    self.chunks[*left_index].chunk.chunk_id
                        .cmp(&self.chunks[*right_index].chunk.chunk_id)
                })
        });
        ranked.truncate(top_k);

        let hits = ranked.into_iter().enumerate().map(
            |(rank_index, (document_index, score))| {
                let chunk = &self.chunks[document_index].chunk;
                chunk_to_hit(chunk, score as f64, rank_index + 1)
            }
        ).collect();

        Ok(LongMemEvalV2EntitySearchResultV1 {
            hits,
            search_latency_ns: u64::try_from(started.elapsed().as_nanos())
                .unwrap_or(u64::MAX),
        })
    }
}

fn structured_literals(chunk: &LongMemEvalV2EvidenceChunkV1) -> BTreeMap<String, u32> {
    let mut literals = BTreeMap::new();
    add_field_literals(&mut literals, &chunk.trajectory_id, 5);
    add_field_literals(&mut literals, &chunk.goal, 2);
    add_field_literals(&mut literals, &chunk.start_url, 4);

    for state in &chunk.state_window {
        add_field_literals(&mut literals, &state.url, 4);
        if let Some(action) = &state.action_on_arrival {
            add_field_literals(&mut literals, action, 3);
        }
        add_field_literals(&mut literals, &state.source_state_index.to_string(), 3);
        if let Some(step) = state.step {
            add_field_literals(&mut literals, &step.to_string(), 3);
        }
    }
    literals
}

fn add_field_literals(literals: &mut BTreeMap<String, u32>, value: &str, weight: u32) {
    for literal in query_literals(value) {
        literals.entry(literal)
            .and_modify(|existing| *existing = (*existing).max(weight))
            .or_insert(weight);
    }
}

/// Keep exact atoms and their structured components. A term is considered a
/// literal only when it contains a digit or a structural delimiter; ordinary
/// prose tokens are left to lexical/semantic retrieval channels.
fn query_literals(value: &str) -> BTreeSet<String> {
    let mut output = BTreeSet::new();
    for raw in value.split_whitespace() {
        let atom = raw.trim_matches(|character: char| {
            !character.is_alphanumeric() && !"-_./:@#".contains(character)
        });
        if atom.is_empty() {
            continue;
        }
        let normalized = atom.to_lowercase();
        if is_literal_atom(&normalized) {
            output.insert(normalized.clone());
        }

        for part in normalized.split(|character: char| {
            !character.is_alphanumeric() && character != '.' && character != '_' && character != '-'
        }) {
            if !part.is_empty() && is_literal_atom(part) {
                output.insert(part.to_owned());
            }
        }
    }
    output
}

fn is_literal_atom(value: &str) -> bool {
    value.chars().any(char::is_numeric)
        || value.chars().any(|character| "-_./:@#".contains(character))
}

fn chunk_to_hit(
    chunk: &LongMemEvalV2EvidenceChunkV1,
    score: f64,
    rank: usize,
) -> LongMemEvalV2HitV1 {
    LongMemEvalV2HitV1 {
        chunk_id: chunk.chunk_id.clone(),
        trajectory_id: chunk.trajectory_id.clone(),
        domain: chunk.domain.clone(),
        source_trajectory_sha256: chunk.source_trajectory_sha256.clone(),
        center_state_ordinal: chunk.center_state_ordinal,
        score,
        rank,
        evidence_text: chunk.evidence_text.clone(),
        screenshot_references: chunk.screenshot_references.clone(),
        screenshot_sha256s: chunk.state_window.iter()
            .map(|state| state.screenshot_sha256.clone())
            .collect(),
        state_window: chunk.state_window.clone(),
    }
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
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

#[cfg(test)]
mod tests {
    use super::*;

    fn trajectory(id: &str, url: &str, state_index: i64) -> LongMemEvalV2TrajectoryV1 {
        let value = serde_json::json!({
            "id": id,
            "domain": "web",
            "goal": "Create a customer order",
            "start_url": "https://shop.example/",
            "outcome": "success",
            "states": [{
                "url": url,
                "action": "submit order",
                "thoughts": "order details",
                "accessibility_tree": "Order confirmation",
                "screenshot": "screenshots/state.png",
                "state_index": state_index,
                "step": 7
            }]
        });
        LongMemEvalV2TrajectoryV1::parse_jsonl_record(&value.to_string())
            .expect("valid source trajectory")
    }

    #[test]
    fn exact_trajectory_identifier_retrieves_matching_source_chunk() {
        let input = trajectory(
            "web-run-2026-042",
            "https://shop.example/orders/42",
            42,
        );
        let index = LongMemEvalV2EntityLiteralIndexV1::build("web", &[input], 0)
            .expect("entity index");
        let result = index.search("web", "find web-run-2026-042", 5)
            .expect("literal search");

        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].trajectory_id, "web-run-2026-042");
        assert_eq!(result.hits[0].rank, 1);
        assert!(result.hits[0].score > 0.0);
    }

    #[test]
    fn exact_url_and_numeric_state_literals_are_searchable() {
        let input = trajectory(
            "web-run-alpha",
            "https://shop.example/orders/7319",
            7319,
        );
        let index = LongMemEvalV2EntityLiteralIndexV1::build("web", &[input], 0)
            .expect("entity index");

        let url_result = index.search("web", "https://shop.example/orders/7319", 5)
            .expect("exact URL search");
        let number_result = index.search("web", "state index 7319", 5)
            .expect("numeric literal search");

        assert_eq!(url_result.hits.len(), 1);
        assert_eq!(number_result.hits.len(), 1);
        assert_eq!(url_result.hits[0].chunk_id, number_result.hits[0].chunk_id);
    }

    #[test]
    fn generic_prose_does_not_duplicate_bm25_retrieval() {
        let input = trajectory(
            "web-run-alpha",
            "https://shop.example/orders/42",
            42,
        );
        let index = LongMemEvalV2EntityLiteralIndexV1::build("web", &[input], 0)
            .expect("entity index");
        let result = index.search("web", "customer requested an order", 5)
            .expect("generic query");

        assert!(result.hits.is_empty());
    }

    #[test]
    fn exact_literal_ranking_is_deterministic_and_top_k_bounded() {
        let first = trajectory("trajectory-42", "https://shop.example/orders/42", 42);
        let second = trajectory("trajectory-73", "https://shop.example/orders/73", 73);
        let index = LongMemEvalV2EntityLiteralIndexV1::build("web", &[first, second], 0)
            .expect("entity index");
        let result = index.search("web", "trajectory-73", 1)
            .expect("literal search");

        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].trajectory_id, "trajectory-73");
    }

    #[test]
    fn domain_and_top_k_are_enforced() {
        let input = trajectory("trajectory-42", "https://shop.example/orders/42", 42);
        let index = LongMemEvalV2EntityLiteralIndexV1::build("web", &[input], 0)
            .expect("entity index");

        assert_eq!(
            index.search("enterprise", "trajectory-42", 5).map(|_| ()),
            Err(V2TrajectoryReject::WrongDomain)
        );
        assert_eq!(
            index.search("web", "trajectory-42", 0).map(|_| ()),
            Err(V2TrajectoryReject::InvalidTopK)
        );
        assert_eq!(
            index.search("web", "   ", 5).map(|_| ()),
            Err(V2TrajectoryReject::EmptyQuery)
        );
    }

    #[test]
    fn corpus_commitment_is_stable_under_input_order() {
        let first = trajectory("trajectory-42", "https://shop.example/orders/42", 42);
        let second = trajectory("trajectory-73", "https://shop.example/orders/73", 73);
        let forward = LongMemEvalV2EntityLiteralIndexV1::build(
            "web", &[first.clone(), second.clone()], 0
        ).expect("forward index");
        let reverse = LongMemEvalV2EntityLiteralIndexV1::build(
            "web", &[second, first], 0
        ).expect("reverse index");

        assert_eq!(forward.corpus_sha256(), reverse.corpus_sha256());
    }
}
