// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Raw-session preservation, deterministic textual chunking and BM25 reference.
//!
//! This is a local lexical baseline, not a dense embedding model or hosted AML
//! implementation. It preserves original messages and metadata; normalization
//! is used only for the derived BM25 token index.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{
    add_payload_digest, chunk_textual_messages, AddRequest, FrozenAdapterWordCounter, Message,
    MessageContent, Role, TextualChunkPlan, Track,
};

const RAW_SESSION_SCHEMA_VERSION: u64 = 1;
const DEFAULT_BM25_K1: f64 = 1.2;
const DEFAULT_BM25_B: f64 = 0.75;
const DEFAULT_TOKENIZER_VERSION: &str = "unicode_alphanumeric_v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RawSessionReject {
    InvalidJson { line: usize },
    EventMustBeObject { line: usize },
    UnknownField { line: usize, field: String },
    MissingField { line: usize, field: &'static str },
    UnsupportedSchemaVersion { line: usize, version: u64 },
    InvalidField { line: usize, field: &'static str },
    EmptySession,
    MixedUsers,
    MixedSessions,
    DuplicateTurnId { turn_id: String },
    InvalidChunkPlan,
    EmptyIndex,
    DuplicateChunkId { chunk_id: String },
    DuplicateSessionChunkOrder,
    ChunkIntegrityMismatch,
    WrongUserScope,
    EmptyQuery,
    InvalidTopK,
    InvalidNeighborRadius,
    InvalidBm25Profile,
    InvalidRrfRankConstant,
    RankingCountMismatch,
    InvalidRrfWeight,
    DuplicateIdInRanking { ranking: usize, id: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawSessionMessageV1 {
    pub turn_id: String,
    pub role: Role,
    pub timestamp_ms: i64,
    /// Exact original message text. It is never rewritten by the indexer.
    pub text: String,
    pub raw_line_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawSessionV1 {
    pub schema_version: u64,
    pub user_id: String,
    pub session_id: String,
    pub messages: Vec<RawSessionMessageV1>,
    /// Digest of the complete original JSONL byte stream, including line endings.
    pub raw_session_sha256: String,
}

impl RawSessionV1 {
    /// Strictly parse one message per JSONL line.
    ///
    /// Expected fields: schema_version, user_id, session_id, turn_id, role,
    /// timestamp_ms and text. Unknown fields are rejected so a future adapter
    /// cannot silently drop metadata while claiming to preserve it.
    pub fn parse_jsonl(input: &str) -> Result<Self, RawSessionReject> {
        let mut messages = Vec::new();
        let mut seen_turn_ids = BTreeSet::new();
        let mut user_id: Option<String> = None;
        let mut session_id: Option<String> = None;

        for (line_index, line) in input.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let line_number = line_index + 1;
            let value: Value = serde_json::from_str(line)
                .map_err(|_| RawSessionReject::InvalidJson { line: line_number })?;
            let object = value
                .as_object()
                .ok_or(RawSessionReject::EventMustBeObject { line: line_number })?;

            for key in object.keys() {
                if !matches!(
                    key.as_str(),
                    "schema_version"
                        | "user_id"
                        | "session_id"
                        | "turn_id"
                        | "role"
                        | "timestamp_ms"
                        | "text"
                ) {
                    return Err(RawSessionReject::UnknownField {
                        line: line_number,
                        field: key.clone(),
                    });
                }
            }

            let required_string = |field: &'static str| {
                object
                    .get(field)
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .filter(|value| !value.trim().is_empty())
                    .ok_or(RawSessionReject::MissingField {
                        line: line_number,
                        field,
                    })
            };

            let schema_version = object
                .get("schema_version")
                .and_then(Value::as_u64)
                .ok_or(RawSessionReject::InvalidField {
                    line: line_number,
                    field: "schema_version",
                })?;
            if schema_version != RAW_SESSION_SCHEMA_VERSION {
                return Err(RawSessionReject::UnsupportedSchemaVersion {
                    line: line_number,
                    version: schema_version,
                });
            }

            let current_user = required_string("user_id")?;
            let current_session = required_string("session_id")?;
            let turn_id = required_string("turn_id")?;
            if user_id.as_ref().is_some_and(|value| value != &current_user) {
                return Err(RawSessionReject::MixedUsers);
            }
            if session_id.as_ref().is_some_and(|value| value != &current_session) {
                return Err(RawSessionReject::MixedSessions);
            }
            user_id.get_or_insert(current_user);
            session_id.get_or_insert(current_session);

            if !seen_turn_ids.insert(turn_id.clone()) {
                return Err(RawSessionReject::DuplicateTurnId { turn_id });
            }
            let role = match required_string("role")?.as_str() {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                _ => {
                    return Err(RawSessionReject::InvalidField {
                        line: line_number,
                        field: "role",
                    });
                }
            };
            let timestamp_ms = object
                .get("timestamp_ms")
                .and_then(Value::as_i64)
                .filter(|value| *value >= 0)
                .ok_or(RawSessionReject::InvalidField {
                    line: line_number,
                    field: "timestamp_ms",
                })?;
            let text = required_string("text")?;
            messages.push(RawSessionMessageV1 {
                turn_id,
                role,
                timestamp_ms,
                text,
                raw_line_sha256: sha256_hex(line.as_bytes()),
            });
        }

        if messages.is_empty() {
            return Err(RawSessionReject::EmptySession);
        }

        Ok(Self {
            schema_version: RAW_SESSION_SCHEMA_VERSION,
            user_id: user_id.ok_or(RawSessionReject::EmptySession)?,
            session_id: session_id.ok_or(RawSessionReject::EmptySession)?,
            messages,
            raw_session_sha256: sha256_hex(input.as_bytes()),
        })
    }

    pub fn to_add_request(&self, request_id: impl Into<String>) -> AddRequest {
        AddRequest {
            request_id: request_id.into(),
            user_id: self.user_id.clone(),
            session_id: self.session_id.clone(),
            messages: self
                .messages
                .iter()
                .map(|message| Message {
                    role: message.role,
                    timestamp_ms: Some(message.timestamp_ms),
                    content: MessageContent::Text(message.text.clone()),
                })
                .collect(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawSessionChunkV1 {
    pub chunk_id: String,
    pub user_id: String,
    pub session_id: String,
    pub ordinal: usize,
    pub request_id: String,
    pub turn_ids: Vec<String>,
    pub messages: Vec<Message>,
    pub first_timestamp_ms: i64,
    pub last_timestamp_ms: i64,
    pub source_session_sha256: String,
    pub source_payload_sha256: String,
    pub chunk_sha256: String,
    /// Exact text extracted from original messages, used only for lexical indexing.
    index_text: String,
}

impl RawSessionChunkV1 {
    pub fn index_text(&self) -> &str {
        &self.index_text
    }

    pub fn message_count(&self) -> usize {
        self.messages.len()
    }
}

/// Apply the same deterministic Textual limits used by the provider-neutral
/// Add adapter (20 messages / 2,000 words), while keeping original turn IDs,
/// message text, roles and timestamps attached to each resulting chunk.
pub fn chunk_raw_session(
    session: &RawSessionV1,
    request_id: impl Into<String>,
    counter: &impl FrozenAdapterWordCounter,
) -> Result<Vec<RawSessionChunkV1>, RawSessionReject> {
    let plan = chunk_textual_messages(&session.to_add_request(request_id), counter)
        .map_err(|_| RawSessionReject::InvalidChunkPlan)?;
    chunks_from_plan(session, plan)
}

fn chunks_from_plan(
    session: &RawSessionV1,
    plan: TextualChunkPlan,
) -> Result<Vec<RawSessionChunkV1>, RawSessionReject> {
    let mut source_offset = 0usize;
    let mut chunks = Vec::with_capacity(plan.chunks.len());

    for (ordinal, request) in plan.chunks.into_iter().enumerate() {
        let end = source_offset
            .checked_add(request.messages.len())
            .ok_or(RawSessionReject::InvalidChunkPlan)?;
        let source_messages = session
            .messages
            .get(source_offset..end)
            .ok_or(RawSessionReject::InvalidChunkPlan)?;
        if source_messages.len() != request.messages.len() {
            return Err(RawSessionReject::InvalidChunkPlan);
        }
        for (source, projected) in source_messages.iter().zip(&request.messages) {
            if source.role != projected.role
                || projected.timestamp_ms != Some(source.timestamp_ms)
                || projected.content != MessageContent::Text(source.text.clone())
            {
                return Err(RawSessionReject::InvalidChunkPlan);
            }
        }

        let turn_ids: Vec<String> = source_messages
            .iter()
            .map(|message| message.turn_id.clone())
            .collect();
        let first_timestamp_ms = source_messages
            .first()
            .ok_or(RawSessionReject::InvalidChunkPlan)?
            .timestamp_ms;
        let last_timestamp_ms = source_messages
            .last()
            .ok_or(RawSessionReject::InvalidChunkPlan)?
            .timestamp_ms;
        let index_text = request
            .messages
            .iter()
            .map(|message| match &message.content {
                MessageContent::Text(text) => Ok(text.as_str()),
                MessageContent::Parts(_) => Err(RawSessionReject::InvalidChunkPlan),
            })
            .collect::<Result<Vec<_>, _>>()?
            .join("\n");

        let source_payload_sha256 = add_payload_digest(Track::Textual, &request);
        let chunk_sha256 = chunk_digest(
            &session.user_id,
            &session.session_id,
            ordinal,
            &session.raw_session_sha256,
            &source_payload_sha256,
            &turn_ids,
        );
        source_offset = end;
        chunks.push(RawSessionChunkV1 {
            chunk_id: chunk_sha256.clone(),
            user_id: session.user_id.clone(),
            session_id: session.session_id.clone(),
            ordinal,
            request_id: request.request_id.clone(),
            turn_ids,
            messages: request.messages,
            first_timestamp_ms,
            last_timestamp_ms,
            source_session_sha256: session.raw_session_sha256.clone(),
            source_payload_sha256,
            chunk_sha256,
            index_text,
        });
    }

    if source_offset != session.messages.len() {
        return Err(RawSessionReject::InvalidChunkPlan);
    }
    Ok(chunks)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bm25ProfileV1 {
    pub k1: f64,
    pub b: f64,
    pub tokenizer_version: &'static str,
}

impl Default for Bm25ProfileV1 {
    fn default() -> Self {
        Self {
            k1: DEFAULT_BM25_K1,
            b: DEFAULT_BM25_B,
            tokenizer_version: DEFAULT_TOKENIZER_VERSION,
        }
    }
}

impl Bm25ProfileV1 {
    fn validate(self) -> Result<(), RawSessionReject> {
        if !self.k1.is_finite()
            || self.k1 <= 0.0
            || !self.b.is_finite()
            || !(0.0..=1.0).contains(&self.b)
            || self.tokenizer_version.trim().is_empty()
        {
            return Err(RawSessionReject::InvalidBm25Profile);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Bm25HitV1 {
    pub chunk_id: String,
    pub score: f64,
    pub rank: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NeighborContextV1 {
    pub chunk: RawSessionChunkV1,
    pub expanded_from_hit_rank: usize,
    /// Negative means an earlier source chunk; positive means a later one.
    pub relative_chunk_offset: i32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RawSessionSearchResultV1 {
    pub hits: Vec<Bm25HitV1>,
    /// Neighbor chunks are separate from scored hits; no synthetic relevance score is assigned.
    pub neighbor_context: Vec<NeighborContextV1>,
}

#[derive(Clone, Debug)]
struct IndexedChunk {
    chunk: RawSessionChunkV1,
    terms: BTreeMap<String, usize>,
    token_count: usize,
}

#[derive(Clone, Debug)]
pub struct RawSessionBm25IndexV1 {
    user_id: String,
    profile: Bm25ProfileV1,
    chunks: Vec<IndexedChunk>,
    chunk_positions: BTreeMap<(String, usize), usize>,
    document_frequency: BTreeMap<String, usize>,
    average_document_length: f64,
    corpus_sha256: String,
}

impl RawSessionBm25IndexV1 {
    pub fn build(
        user_id: impl Into<String>,
        mut chunks: Vec<RawSessionChunkV1>,
        profile: Bm25ProfileV1,
    ) -> Result<Self, RawSessionReject> {
        profile.validate()?;
        let user_id = user_id.into();
        if user_id.trim().is_empty() || chunks.is_empty() {
            return Err(RawSessionReject::EmptyIndex);
        }

        if chunks.iter().any(|chunk| chunk.user_id != user_id) {
            return Err(RawSessionReject::WrongUserScope);
        }
        for chunk in &chunks {
            validate_chunk_integrity(chunk)?;
        }
        chunks.sort_by(|left, right| {
            left.session_id
                .cmp(&right.session_id)
                .then_with(|| left.ordinal.cmp(&right.ordinal))
                .then_with(|| left.chunk_id.cmp(&right.chunk_id))
        });

        let mut seen = BTreeSet::new();
        let mut indexed = Vec::with_capacity(chunks.len());
        let mut positions = BTreeMap::new();
        let mut document_frequency = BTreeMap::<String, usize>::new();
        let mut total_tokens = 0usize;
        let mut corpus_hasher = Sha256::new();
        corpus_hasher.update(b"luminous:mace:raw-session-bm25-corpus:v1\0");
        hash_field(&mut corpus_hasher, &user_id);
        hash_field(&mut corpus_hasher, profile.tokenizer_version);
        corpus_hasher.update(profile.k1.to_bits().to_be_bytes());
        corpus_hasher.update(profile.b.to_bits().to_be_bytes());

        for (index, chunk) in chunks.into_iter().enumerate() {
            if !seen.insert(chunk.chunk_id.clone()) {
                return Err(RawSessionReject::DuplicateChunkId {
                    chunk_id: chunk.chunk_id,
                });
            }
            let tokens = tokenize(&chunk.index_text);
            let token_count = tokens.len();
            let mut terms = BTreeMap::<String, usize>::new();
            for token in tokens {
                *terms.entry(token).or_default() += 1;
            }
            for term in terms.keys() {
                *document_frequency.entry(term.clone()).or_default() += 1;
            }
            total_tokens = total_tokens.saturating_add(token_count);
            if positions.insert((chunk.session_id.clone(), chunk.ordinal), index).is_some() {
                return Err(RawSessionReject::DuplicateSessionChunkOrder);
            }
            hash_field(&mut corpus_hasher, &chunk.chunk_id);
            hash_field(&mut corpus_hasher, &chunk.source_payload_sha256);
            indexed.push(IndexedChunk {
                chunk,
                terms,
                token_count,
            });
        }

        let average_document_length = if indexed.is_empty() {
            0.0
        } else {
            total_tokens as f64 / indexed.len() as f64
        };
        let corpus_sha256 = hex_digest(corpus_hasher.finalize().as_slice());

        Ok(Self {
            user_id,
            profile,
            chunks: indexed,
            chunk_positions: positions,
            document_frequency,
            average_document_length,
            corpus_sha256,
        })
    }

    pub fn corpus_sha256(&self) -> &str {
        &self.corpus_sha256
    }

    pub fn document_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn profile(&self) -> Bm25ProfileV1 {
        self.profile
    }

    pub fn search(
        &self,
        user_id: &str,
        query: &str,
        top_k: usize,
        neighbor_radius: usize,
    ) -> Result<RawSessionSearchResultV1, RawSessionReject> {
        if user_id != self.user_id {
            return Err(RawSessionReject::WrongUserScope);
        }
        if top_k == 0 {
            return Err(RawSessionReject::InvalidTopK);
        }
        if neighbor_radius > 64 {
            return Err(RawSessionReject::InvalidNeighborRadius);
        }
        let query_terms: BTreeSet<String> = tokenize(query).into_iter().collect();
        if query_terms.is_empty() {
            return Err(RawSessionReject::EmptyQuery);
        }

        let document_count = self.chunks.len();
        let mut hits = Vec::new();
        for indexed in &self.chunks {
            if indexed.token_count == 0 || self.average_document_length == 0.0 {
                continue;
            }
            let mut score = 0.0_f64;
            for term in &query_terms {
                let Some(term_frequency) = indexed.terms.get(term) else {
                    continue;
                };
                let document_frequency = *self
                    .document_frequency
                    .get(term)
                    .unwrap_or(&0) as f64;
                let n = document_count as f64;
                let idf = (1.0 + (n - document_frequency + 0.5) / (document_frequency + 0.5)).ln();
                let tf = *term_frequency as f64;
                let length = indexed.token_count as f64;
                let denominator = tf
                    + self.profile.k1
                        * (1.0 - self.profile.b
                            + self.profile.b * length / self.average_document_length);
                score += idf * (tf * (self.profile.k1 + 1.0)) / denominator;
            }
            if score > 0.0 {
                hits.push(Bm25HitV1 {
                    chunk_id: indexed.chunk.chunk_id.clone(),
                    score,
                    rank: 0,
                });
            }
        }

        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.chunk_id.cmp(&right.chunk_id))
        });
        hits.truncate(top_k.min(FORMAL_TOP_K));
        for (rank, hit) in hits.iter_mut().enumerate() {
            hit.rank = rank + 1;
        }

        let mut neighbor_by_id = BTreeMap::<String, NeighborContextV1>::new();
        if neighbor_radius > 0 {
            for hit in &hits {
                let Some(indexed) = self.chunks.iter().find(|doc| doc.chunk.chunk_id == hit.chunk_id) else {
                    continue;
                };
                let session_id = indexed.chunk.session_id.clone();
                let ordinal = indexed.chunk.ordinal;
                for offset in -(neighbor_radius as i64)..=(neighbor_radius as i64) {
                    if offset == 0 {
                        continue;
                    }
                    let neighbor_ordinal = if offset < 0 {
                        ordinal.checked_sub(offset.unsigned_abs() as usize)
                    } else {
                        ordinal.checked_add(offset as usize)
                    };
                    let Some(neighbor_ordinal) = neighbor_ordinal else {
                        continue;
                    };
                    let Some(position) = self.chunk_positions.get(&(session_id.clone(), neighbor_ordinal)) else {
                        continue;
                    };
                    let neighbor = &self.chunks[*position].chunk;
                    if hits.iter().any(|known| known.chunk_id == neighbor.chunk_id) {
                        continue;
                    }
                    neighbor_by_id.entry(neighbor.chunk_id.clone()).or_insert_with(|| {
                        NeighborContextV1 {
                            chunk: neighbor.clone(),
                            expanded_from_hit_rank: hit.rank,
                            relative_chunk_offset: offset as i32,
                        }
                    });
                }
            }
        }

        let mut neighbor_context: Vec<_> = neighbor_by_id.into_values().collect();
        neighbor_context.sort_by(|left, right| {
            left.expanded_from_hit_rank
                .cmp(&right.expanded_from_hit_rank)
                .then_with(|| left.chunk.session_id.cmp(&right.chunk.session_id))
                .then_with(|| left.chunk.ordinal.cmp(&right.chunk.ordinal))
        });

        Ok(RawSessionSearchResultV1 {
            hits,
            neighbor_context,
        })
    }
}

/// Reciprocal-rank fusion over already ranked, user/scope-filtered result lists.
/// Dense retrieval is an injected ranking, not implemented or claimed by this
/// lexical baseline. The same function supports BM25+dense matched comparisons
/// once a pinned embedding backend supplies its ranked IDs.
pub fn reciprocal_rank_fusion(
    rankings: &[Vec<String>],
    weights: &[f64],
    rrf_k: usize,
) -> Result<Vec<RrfHitV1>, RawSessionReject> {
    if rankings.is_empty() || rankings.len() != weights.len() {
        return Err(RawSessionReject::RankingCountMismatch);
    }
    if rrf_k == 0 {
        return Err(RawSessionReject::InvalidRrfRankConstant);
    }
    if weights.iter().any(|weight| !weight.is_finite() || *weight <= 0.0) {
        return Err(RawSessionReject::InvalidRrfWeight);
    }

    let mut scores = BTreeMap::<String, f64>::new();
    for (ranking_index, (ranking, weight)) in rankings.iter().zip(weights).enumerate() {
        let mut seen = BTreeSet::new();
        for (rank, id) in ranking.iter().enumerate() {
            if id.trim().is_empty() {
                return Err(RawSessionReject::DuplicateIdInRanking {
                    ranking: ranking_index,
                    id: id.clone(),
                });
            }
            if !seen.insert(id.as_str()) {
                return Err(RawSessionReject::DuplicateIdInRanking {
                    ranking: ranking_index,
                    id: id.clone(),
                });
            }
            let denominator = rrf_k
                .checked_add(rank)
                .and_then(|value| value.checked_add(1))
                .ok_or(RawSessionReject::InvalidRrfRankConstant)?;
            *scores.entry(id.clone()).or_default() += *weight / denominator as f64;
        }
    }

    let mut fused: Vec<_> = scores
        .into_iter()
        .map(|(document_id, score)| RrfHitV1 {
            document_id,
            score,
            rank: 0,
        })
        .collect();
    fused.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.document_id.cmp(&right.document_id))
    });
    for (rank, hit) in fused.iter_mut().enumerate() {
        hit.rank = rank + 1;
    }
    Ok(fused)
}

#[derive(Clone, Debug, PartialEq)]
pub struct RrfHitV1 {
    pub document_id: String,
    pub score: f64,
    pub rank: usize,
}

fn validate_chunk_integrity(chunk: &RawSessionChunkV1) -> Result<(), RawSessionReject> {
    if chunk.user_id.trim().is_empty()
        || chunk.session_id.trim().is_empty()
        || chunk.request_id.trim().is_empty()
        || chunk.messages.is_empty()
        || chunk.messages.len() != chunk.turn_ids.len()
        || chunk.turn_ids.iter().any(|turn_id| turn_id.trim().is_empty())
        || chunk.source_session_sha256.len() != 64
        || chunk.source_payload_sha256.len() != 64
    {
        return Err(RawSessionReject::ChunkIntegrityMismatch);
    }
    let mut turn_ids = BTreeSet::new();
    if chunk.turn_ids.iter().any(|turn_id| !turn_ids.insert(turn_id.as_str())) {
        return Err(RawSessionReject::ChunkIntegrityMismatch);
    }
    let mut index_parts = Vec::with_capacity(chunk.messages.len());
    for message in &chunk.messages {
        let MessageContent::Text(text) = &message.content else {
            return Err(RawSessionReject::ChunkIntegrityMismatch);
        };
        index_parts.push(text.as_str());
    }
    let expected_index_text = index_parts.join("\n");
    if expected_index_text != chunk.index_text {
        return Err(RawSessionReject::ChunkIntegrityMismatch);
    }
    let request = AddRequest {
        request_id: chunk.request_id.clone(),
        user_id: chunk.user_id.clone(),
        session_id: chunk.session_id.clone(),
        messages: chunk.messages.clone(),
    };
    if add_payload_digest(Track::Textual, &request) != chunk.source_payload_sha256 {
        return Err(RawSessionReject::ChunkIntegrityMismatch);
    }
    let expected_digest = chunk_digest(
        &chunk.user_id,
        &chunk.session_id,
        chunk.ordinal,
        &chunk.source_session_sha256,
        &chunk.source_payload_sha256,
        &chunk.turn_ids,
    );
    if expected_digest != chunk.chunk_sha256 || chunk.chunk_id != chunk.chunk_sha256 {
        return Err(RawSessionReject::ChunkIntegrityMismatch);
    }
    let first = chunk.messages.first().and_then(|message| message.timestamp_ms);
    let last = chunk.messages.last().and_then(|message| message.timestamp_ms);
    if first != Some(chunk.first_timestamp_ms) || last != Some(chunk.last_timestamp_ms) {
        return Err(RawSessionReject::ChunkIntegrityMismatch);
    }
    Ok(())
}

fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for character in text.chars() {
        if character.is_alphanumeric() {
            current.extend(character.to_lowercase());
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn chunk_digest(
    user_id: &str,
    session_id: &str,
    ordinal: usize,
    source_session_sha256: &str,
    source_payload_sha256: &str,
    turn_ids: &[String],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:raw-session-chunk:v1\0");
    hash_field(&mut hasher, user_id);
    hash_field(&mut hasher, session_id);
    hasher.update((ordinal as u64).to_be_bytes());
    hash_field(&mut hasher, source_session_sha256);
    hash_field(&mut hasher, source_payload_sha256);
    hasher.update((turn_ids.len() as u64).to_be_bytes());
    for turn_id in turn_ids {
        hash_field(&mut hasher, turn_id);
    }
    hex_digest(hasher.finalize().as_slice())
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

    struct WhitespaceCounter;

    impl FrozenAdapterWordCounter for WhitespaceCounter {
        fn adapter_version(&self) -> &str {
            "fixture-whitespace-v1"
        }

        fn count_words(&self, messages: &[Message]) -> usize {
            messages
                .iter()
                .map(|message| match &message.content {
                    MessageContent::Text(text) => text.split_whitespace().count(),
                    MessageContent::Parts(_) => 0,
                })
                .sum()
        }
    }

    fn session_json(
        user: &str,
        session: &str,
        turns: &[(&str, &str, &str, i64)],
    ) -> String {
        turns
            .iter()
            .map(|(turn, role, text, timestamp)| {
                serde_json::json!({
                    "schema_version": 1,
                    "user_id": user,
                    "session_id": session,
                    "turn_id": turn,
                    "role": role,
                    "timestamp_ms": timestamp,
                    "text": text
                })
                .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn chunks(
        user: &str,
        session: &str,
        turns: &[(&str, &str, &str, i64)],
    ) -> Vec<RawSessionChunkV1> {
        let source = RawSessionV1::parse_jsonl(&session_json(user, session, turns))
            .expect("valid raw session");
        chunk_raw_session(&source, "request-v1", &WhitespaceCounter).expect("chunked")
    }

    #[test]
    fn raw_session_parser_preserves_verbatim_text_and_source_digests() {
        let input = session_json(
            "user-a",
            "session-1",
            &[
                ("t1", "user", "Keep  exact spacing and punctuation!", 100),
                ("t2", "assistant", "A follow-up answer.", 101),
            ],
        );
        let session = RawSessionV1::parse_jsonl(&input).expect("parsed");
        assert_eq!(session.messages[0].text, "Keep  exact spacing and punctuation!");
        assert_eq!(session.messages[0].turn_id, "t1");
        assert_eq!(session.messages[0].timestamp_ms, 100);
        assert_eq!(session.raw_session_sha256, sha256_hex(input.as_bytes()));
        assert_ne!(session.messages[0].raw_line_sha256, session.messages[1].raw_line_sha256);
    }

    #[test]
    fn raw_session_parser_rejects_unknown_fields_and_duplicate_turns() {
        let unknown = serde_json::json!({
            "schema_version": 1, "user_id": "u", "session_id": "s",
            "turn_id": "t1", "role": "user", "timestamp_ms": 1,
            "text": "hello", "untracked_metadata": "must not be dropped"
        }).to_string();
        assert!(matches!(
            RawSessionV1::parse_jsonl(&unknown),
            Err(RawSessionReject::UnknownField { .. })
        ));
        let duplicate = session_json(
            "u",
            "s",
            &[("t1", "user", "first", 1), ("t1", "assistant", "second", 2)],
        );
        assert_eq!(
            RawSessionV1::parse_jsonl(&duplicate),
            Err(RawSessionReject::DuplicateTurnId { turn_id: "t1".into() })
        );
    }

    #[test]
    fn parser_rejects_cross_user_or_cross_session_batches() {
        let mixed_user = format!(
            "{}\n{}",
            session_json("u1", "s", &[("t1", "user", "one", 1)]),
            session_json("u2", "s", &[("t2", "user", "two", 2)])
        );
        assert_eq!(RawSessionV1::parse_jsonl(&mixed_user), Err(RawSessionReject::MixedUsers));
        let mixed_session = format!(
            "{}\n{}",
            session_json("u", "s1", &[("t1", "user", "one", 1)]),
            session_json("u", "s2", &[("t2", "user", "two", 2)])
        );
        assert_eq!(RawSessionV1::parse_jsonl(&mixed_session), Err(RawSessionReject::MixedSessions));
    }

    #[test]
    fn chunking_is_deterministic_and_keeps_original_turn_boundaries() {
        let ids: Vec<String> = (0..25).map(|i| format!("t{i}")).collect();
        let texts: Vec<String> = (0..25).map(|i| format!("message {i}")).collect();
        let turns: Vec<_> = (0..25)
            .map(|i| (ids[i].as_str(), "user", texts[i].as_str(), i as i64))
            .collect();
        let input = session_json("u", "s", &turns);
        let session = RawSessionV1::parse_jsonl(&input).expect("session");
        let first = chunk_raw_session(&session, "r", &WhitespaceCounter).expect("chunks");
        let second = chunk_raw_session(&session, "r", &WhitespaceCounter).expect("chunks");
        assert_eq!(first, second);
        assert_eq!(first.len(), 2);
        assert_eq!(first[0].turn_ids.len(), 20);
        assert_eq!(first[1].turn_ids, (20..25).map(|i| format!("t{i}")).collect::<Vec<_>>());
        assert_eq!(first[0].messages[0].content, MessageContent::Text("message 0".into()));
    }

    #[test]
    fn source_edit_changes_session_and_chunk_identity() {
        let original = chunks("u", "s", &[("t1", "user", "budget is 10", 1)]);
        let edited = chunks("u", "s", &[("t1", "user", "budget is 11", 1)]);
        assert_ne!(original[0].source_session_sha256, edited[0].source_session_sha256);
        assert_ne!(original[0].chunk_id, edited[0].chunk_id);
    }

    #[test]
    fn bm25_retrieves_exact_lexical_evidence_with_stable_ties() {
        let corpus = vec![
            chunks("u", "s1", &[("t1", "user", "battery replacement warranty", 1)])[0].clone(),
            chunks("u", "s2", &[("t2", "user", "garden irrigation schedule", 2)])[0].clone(),
        ];
        let index = RawSessionBm25IndexV1::build("u", corpus, Bm25ProfileV1::default()).expect("index");
        let result = index.search("u", "battery warranty", 10, 0).expect("search");
        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].rank, 1);
    }

    #[test]
    fn index_rejects_cross_user_and_search_rejects_wrong_user() {
        let mut corpus = chunks("u1", "s", &[("t1", "user", "private preference", 1)]);
        corpus.extend(chunks("u2", "s", &[("t2", "user", "other private data", 2)]));
        assert_eq!(
            RawSessionBm25IndexV1::build("u1", corpus, Bm25ProfileV1::default()).map(|_| ()),
            Err(RawSessionReject::WrongUserScope)
        );
        let index = RawSessionBm25IndexV1::build(
            "u1",
            chunks("u1", "s", &[("t1", "user", "private preference", 1)]),
            Bm25ProfileV1::default(),
        ).expect("index");
        assert_eq!(
            index.search("u2", "private", 5, 0),
            Err(RawSessionReject::WrongUserScope)
        );
    }

    #[test]
    fn neighbor_expansion_stays_within_the_same_session() {
        let ids: Vec<String> = (0..42).map(|i| format!("t{i}")).collect();
        let turns: Vec<_> = (0..42)
            .map(|i| {
                let text = if i == 20 { "rare target marker" } else { "ordinary conversation" };
                (ids[i].as_str(), "user", text, i as i64)
            })
            .collect();
        let source = RawSessionV1::parse_jsonl(&session_json("u", "s", &turns)).expect("session");
        let same_session = chunk_raw_session(&source, "request", &WhitespaceCounter).expect("chunks");
        let other_session = chunks("u", "s-other", &[("other-1", "user", "rare target marker elsewhere", 500)]);
        let mut combined = same_session.clone();
        combined.extend(other_session);
        let index = RawSessionBm25IndexV1::build("u", combined, Bm25ProfileV1::default()).expect("index");
        let result = index.search("u", "rare target marker", 1, 1).expect("search");
        assert_eq!(result.hits.len(), 1);
        assert!(result.neighbor_context.iter().all(|item| item.chunk.session_id == "s"));
    }

    #[test]
    fn empty_queries_and_invalid_profiles_fail_closed() {
        let index = RawSessionBm25IndexV1::build(
            "u",
            chunks("u", "s", &[("t1", "user", "test content", 1)]),
            Bm25ProfileV1::default(),
        ).expect("index");
        assert_eq!(index.search("u", "   !!! ", 5, 0), Err(RawSessionReject::EmptyQuery));
        let invalid = Bm25ProfileV1 { k1: 1.2, b: 1.1, tokenizer_version: "invalid" };
        assert_eq!(
            RawSessionBm25IndexV1::build("u", chunks("u", "s", &[("t1", "user", "test", 1)]), invalid).map(|_| ()),
            Err(RawSessionReject::InvalidBm25Profile)
        );
    }

    #[test]
    fn rrf_combines_complementary_rankings_deterministically() {
        let first = vec!["a".into(), "b".into()];
        let second = vec!["b".into(), "c".into()];
        let result = reciprocal_rank_fusion(&[first.clone(), second.clone()], &[1.0, 1.0], 60)
            .expect("fusion");
        assert_eq!(result[0].document_id, "b");
        assert_eq!(
            result,
            reciprocal_rank_fusion(&[first, second], &[1.0, 1.0], 60).expect("same fusion")
        );
    }

    #[test]
    fn index_detects_chunk_metadata_mutation_after_chunking() {
        let mut corpus = chunks("u", "s", &[("t1", "user", "untouched source", 1)]);
        corpus[0].turn_ids[0] = "rewritten-turn".into();
        assert_eq!(
            RawSessionBm25IndexV1::build("u", corpus, Bm25ProfileV1::default()).map(|_| ()),
            Err(RawSessionReject::ChunkIntegrityMismatch)
        );
    }

    #[test]
    fn index_rejects_duplicate_session_ordinals() {
        let mut corpus = chunks("u", "s", &[("t1", "user", "one", 1)]);
        let mut duplicate = corpus[0].clone();
        duplicate.chunk_id = "another-chunk-id".into();
        duplicate.chunk_sha256 = "another-chunk-id".into();
        corpus.push(duplicate);
        assert_eq!(
            RawSessionBm25IndexV1::build("u", corpus, Bm25ProfileV1::default()).map(|_| ()),
            Err(RawSessionReject::DuplicateSessionChunkOrder)
        );
    }

    #[test]
    fn rrf_rejects_duplicate_ids_and_bad_weights() {
        assert_eq!(
            reciprocal_rank_fusion(&[vec!["a".into(), "a".into()]], &[1.0], 60),
            Err(RawSessionReject::DuplicateIdInRanking { ranking: 0, id: "a".into() })
        );
        assert_eq!(
            reciprocal_rank_fusion(&[vec!["a".into()]], &[f64::NAN], 60),
            Err(RawSessionReject::InvalidRrfWeight)
        );
    }
}
