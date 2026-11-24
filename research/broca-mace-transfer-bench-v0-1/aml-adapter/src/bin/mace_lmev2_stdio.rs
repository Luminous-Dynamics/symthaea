// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Persistent JSONL sidecar for the LongMemEval-V2 official Memory API shim.
//!
//! The current retrieval portfolio combines BM25 prose retrieval with a
//! structured exact-literal channel using reciprocal-rank fusion. query_image
//! remains accepted for API compatibility but does not affect ranking;
//! retrieved windows may still return original screenshot evidence.

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::io::{self, BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use broca_aml_adapter::longmemeval_v2::{
    LongMemEvalV2Bm25IndexV1, LongMemEvalV2ContextKind,
    LongMemEvalV2ContextItemV1, LongMemEvalV2HitV1,
    LongMemEvalV2TrajectoryV1, LongMemEvalV2SearchResultV1,
    resolve_screenshot_reference,
};
use broca_aml_adapter::raw_session::Bm25ProfileV1;
use broca_aml_adapter::longmemeval_v2_candidate_adapter::{
    LongMemEvalV2CandidateIdentityV1,
    encode_longmemeval_v2_search_candidate_evidence_for_channel_v1,
};
use broca_aml_adapter::longmemeval_v2_entity::LongMemEvalV2EntityLiteralIndexV1;
use broca_aml_adapter::retrieval_policy_memory::{
    RetrievalOperationOutcomeV1, RetrievalOperationV1, RetrievalPolicyMemoryV1,
    RetrievalStopReasonV1, RetrievalTraceReceiptV1,
};
use broca_aml_adapter::retrieval_policy_store::RetrievalPolicyTraceFileStoreV1;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RetrievalModeV1 {
    Bm25Only,
    EntityLiteralOnly,
    HybridRrf,
}

impl RetrievalModeV1 {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "bm25_only" => Ok(Self::Bm25Only),
            "entity_literal_only" => Ok(Self::EntityLiteralOnly),
            "hybrid_rrf" => Ok(Self::HybridRrf),
            _ => Err("MACE_LMEV2_RETRIEVAL_MODE must be bm25_only, entity_literal_only, or hybrid_rrf".to_owned()),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Bm25Only => "bm25_only_with_original_image_evidence",
            Self::EntityLiteralOnly => "entity_literal_only_with_original_image_evidence",
            Self::HybridRrf => "hybrid_bm25_entity_literal_with_original_image_evidence",
        }
    }

    const fn backend_id(self) -> &'static str {
        match self {
            Self::Bm25Only => "mace_lmev2_bm25_only",
            Self::EntityLiteralOnly => "mace_lmev2_entity_literal_only",
            Self::HybridRrf => "mace_lmev2_hybrid_rrf_bm25_entity",
        }
    }

    const fn evidence_channel(self) -> &'static str {
        match self {
            Self::Bm25Only => "trajectory_bm25_text",
            Self::EntityLiteralOnly => "trajectory_entity_literal",
            Self::HybridRrf => "hybrid_rrf_bm25_entity",
        }
    }

    const fn uses_bm25(self) -> bool {
        matches!(self, Self::Bm25Only | Self::HybridRrf)
    }

    const fn uses_entity_literal(self) -> bool {
        matches!(self, Self::EntityLiteralOnly | Self::HybridRrf)
    }

    const fn fusion_name(self) -> &'static str {
        match self {
            Self::Bm25Only | Self::EntityLiteralOnly => "none",
            Self::HybridRrf => "reciprocal_rank_fusion_k60",
        }
    }
}

struct RetrievalIndexesV1 {
    bm25: Option<LongMemEvalV2Bm25IndexV1>,
    entity_literal: Option<LongMemEvalV2EntityLiteralIndexV1>,
}

struct Config {
    domain: String,
    run_id: String,
    environment_revision_sha256: String,
    backend_revision_sha256: String,
    retrieval_mode: RetrievalModeV1,
    policy_trace_dir: PathBuf,
    dataset_root: PathBuf,
    top_k: usize,
    slice_radius: usize,
    neighbor_radius: usize,
    profile: Bm25ProfileV1,
}

impl Config {
    fn from_env() -> Result<Self, String> {
        let domain = env::var("MACE_LMEV2_DOMAIN")
            .map_err(|_| "MACE_LMEV2_DOMAIN is required".to_string())?;
        if domain != "web" && domain != "enterprise" {
            return Err("MACE_LMEV2_DOMAIN must be web or enterprise".into());
        }
        let run_id = env::var("MACE_LMEV2_RUN_ID")
            .map_err(|_| "MACE_LMEV2_RUN_ID is required".to_string())?;
        if run_id.trim().is_empty() || run_id.chars().any(char::is_control) {
            return Err("MACE_LMEV2_RUN_ID must be a non-empty printable identifier".into());
        }
        let environment_revision_sha256 = env::var("MACE_LMEV2_ENVIRONMENT_REVISION_SHA256")
            .map_err(|_| "MACE_LMEV2_ENVIRONMENT_REVISION_SHA256 is required".to_string())?;
        if !valid_sha256(&environment_revision_sha256) {
            return Err("MACE_LMEV2_ENVIRONMENT_REVISION_SHA256 must be lowercase SHA-256".into());
        }
        let policy_trace_dir = env::var("MACE_LMEV2_POLICY_TRACE_DIR")
            .map_err(|_| "MACE_LMEV2_POLICY_TRACE_DIR is required".to_string())?;
        let policy_trace_dir = PathBuf::from(policy_trace_dir);
        fs::create_dir_all(&policy_trace_dir)
            .map_err(|error| format!("create policy trace directory: {error}"))?;
        let policy_trace_dir = fs::canonicalize(policy_trace_dir)
            .map_err(|error| format!("canonicalize policy trace directory: {error}"))?;
        let backend_revision_sha256 = env::var("MACE_LMEV2_BACKEND_REVISION_SHA256")
            .map_err(|_| "MACE_LMEV2_BACKEND_REVISION_SHA256 is required".to_string())?;
        if !valid_sha256(&backend_revision_sha256) {
            return Err("MACE_LMEV2_BACKEND_REVISION_SHA256 must be lowercase SHA-256".into());
        }
        let retrieval_mode_value = env::var("MACE_LMEV2_RETRIEVAL_MODE")
            .unwrap_or_else(|_| "hybrid_rrf".to_owned());
        let retrieval_mode = RetrievalModeV1::parse(&retrieval_mode_value)?;
        let root = env::var("MACE_LMEV2_DATASET_ROOT")
            .map_err(|_| "MACE_LMEV2_DATASET_ROOT is required".to_string())?;
        let dataset_root = fs::canonicalize(root)
            .map_err(|error| format!("canonicalize dataset root: {error}"))?;
        if !dataset_root.is_dir() {
            return Err("MACE_LMEV2_DATASET_ROOT must be a directory".into());
        }
        let parse_usize = |name: &str, default: usize| -> Result<usize, String> {
            env::var(name)
                .ok()
                .map(|value| value.parse::<usize>().map_err(|error| format!("{name}: {error}")))
                .transpose()
                .map(|value| value.unwrap_or(default))
        };
        let parse_f64 = |name: &str, default: f64| -> Result<f64, String> {
            env::var(name)
                .ok()
                .map(|value| value.parse::<f64>().map_err(|error| format!("{name}: {error}")))
                .transpose()
                .map(|value| value.unwrap_or(default))
        };
        let top_k = parse_usize("MACE_LMEV2_TOP_K", 8)?;
        let slice_radius = parse_usize("MACE_LMEV2_SLICE_RADIUS", 1)?;
        let neighbor_radius = parse_usize("MACE_LMEV2_NEIGHBOR_RADIUS", 1)?;
        if top_k == 0 || top_k > 100 {
            return Err("MACE_LMEV2_TOP_K must be in 1..=100".into());
        }
        if slice_radius > 32 || neighbor_radius > 32 {
            return Err("MACE_LMEV2 slice and neighbor radii must be in 0..=32".into());
        }
        let profile = Bm25ProfileV1 {
            k1: parse_f64("MACE_LMEV2_BM25_K1", 1.2)?,
            b: parse_f64("MACE_LMEV2_BM25_B", 0.75)?,
            tokenizer_version: "unicode_alphanumeric_v1",
        };
        if !profile.k1.is_finite() || profile.k1 <= 0.0
            || !profile.b.is_finite() || !(0.0..=1.0).contains(&profile.b)
        {
            return Err("invalid BM25 k1/b profile".into());
        }
        Ok(Self {
            domain,
            run_id,
            environment_revision_sha256,
            backend_revision_sha256,
            retrieval_mode,
            policy_trace_dir,
            dataset_root,
            top_k,
            slice_radius,
            neighbor_radius,
            profile,
        })
    }
}


fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn bm25_action_template_sha256(config: &Config) -> String {
    let template = format!(
        "mace-lmev2-bm25-search:v1|domain={}|top_k={}|slice_radius={}|neighbor_radius={}|k1={:.17}|b={:.17}|tokenizer={}",
        config.domain,
        config.top_k,
        config.slice_radius,
        config.neighbor_radius,
        config.profile.k1,
        config.profile.b,
        config.profile.tokenizer_version
    );
    sha256_hex(template.as_bytes())
}

fn entity_literal_action_template_sha256(config: &Config) -> String {
    let template = format!(
        "mace-lmev2-entity-literal-search:v1|domain={}|top_k={}|slice_radius={}",
        config.domain,
        config.top_k,
        config.slice_radius
    );
    sha256_hex(template.as_bytes())
}

fn trace_evidence_ids(hits: &[LongMemEvalV2HitV1]) -> Vec<String> {
    let mut identifiers = hits.iter()
        .map(|hit| format!("lmev2:text:{}:{}", hit.trajectory_id, hit.chunk_id))
        .collect::<Vec<_>>();
    identifiers.sort();
    identifiers.dedup();
    identifiers
}

fn hybrid_rrf_result(
    bm25_hits: &[LongMemEvalV2HitV1],
    entity_hits: &[LongMemEvalV2HitV1],
    neighbor_context: Vec<broca_aml_adapter::longmemeval_v2::LongMemEvalV2NeighborV1>,
    top_k: usize,
    search_latency_ns: u64,
) -> LongMemEvalV2SearchResultV1 {
    const RRF_K: f64 = 60.0;
    let mut fused = std::collections::BTreeMap::<
        String,
        (LongMemEvalV2HitV1, f64),
    >::new();

    for (channel_weight, hits) in [(1.0_f64, bm25_hits), (1.0_f64, entity_hits)] {
        for hit in hits {
            let contribution = channel_weight / (RRF_K + hit.rank as f64);
            let entry = fused.entry(hit.chunk_id.clone())
                .or_insert_with(|| (hit.clone(), 0.0));
            entry.1 += contribution;
        }
    }

    let mut ranked = fused.into_values().collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right.1.total_cmp(&left.1)
            .then_with(|| left.0.chunk_id.cmp(&right.0.chunk_id))
    });
    ranked.truncate(top_k);

    let hits = ranked.into_iter().enumerate().map(|(index, (mut hit, score))| {
        hit.score = score;
        hit.rank = index + 1;
        hit
    }).collect();

    LongMemEvalV2SearchResultV1 {
        hits,
        neighbor_context,
        search_latency_ns,
    }
}

fn unix_now_seconds() -> Result<u64, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|error| format!("system clock precedes Unix epoch: {error}"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

fn context_json(item: LongMemEvalV2ContextItemV1) -> Value {
    let kind = match item.kind {
        LongMemEvalV2ContextKind::Text => "text",
        LongMemEvalV2ContextKind::Image => "image",
    };
    json!({"type": kind, "value": item.value})
}

fn append_hit(
    items: &mut Vec<Value>,
    seen_images: &mut BTreeSet<String>,
    hit: &LongMemEvalV2HitV1,
    root: &Path,
) -> Result<(), String> {
    let resolved = hit
        .resolved_context_items(root)
        .map_err(|error| format!("resolve hit {} screenshot context: {error:?}", hit.chunk_id))?;
    for item in resolved {
        if item.kind == LongMemEvalV2ContextKind::Image
            && !seen_images.insert(item.value.clone())
        {
            continue;
        }
        items.push(context_json(item));
    }
    Ok(())
}

fn handle_request(
    request: Value,
    config: &Config,
    trajectories: &mut Vec<LongMemEvalV2TrajectoryV1>,
    index: &mut Option<RetrievalIndexesV1>,
    policy_memory: &mut RetrievalPolicyMemoryV1,
    policy_trace_store: &RetrievalPolicyTraceFileStoreV1,
) -> Result<Value, String> {
    let operation = request
        .get("op")
        .and_then(Value::as_str)
        .ok_or_else(|| "request is missing string op".to_string())?;

    match operation {
        "insert" => {
            let source = request
                .get("trajectory")
                .ok_or_else(|| "insert request is missing trajectory".to_string())?;
            let trajectory_json = serde_json::to_string(source)
                .map_err(|error| format!("serialize trajectory: {error}"))?;
            let trajectory = LongMemEvalV2TrajectoryV1::parse_jsonl_record(&trajectory_json)
                .map_err(|error| format!("parse trajectory: {error:?}"))?;
            if trajectory.domain != config.domain {
                return Err(format!(
                    "trajectory domain {:?} does not match configured domain {:?}",
                    trajectory.domain, config.domain
                ));
            }
            if trajectories.iter().any(|item| item.id == trajectory.id) {
                return Err(format!("duplicate trajectory insert: {}", trajectory.id));
            }
            trajectories.push(trajectory);
            *index = None;
            Ok(json!({"ok": true, "inserted": 1}))
        }
        "query" => {
            let query = request
                .get("query")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| "query request requires non-empty query".to_string())?;
            let candidate_id = request
                .get("candidate_id")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .ok_or_else(|| "query request requires opaque candidate_id".to_string())?;
            // Accepted for API compatibility. The lexical baseline does not
            // inspect or rank the question image; this is not multimodal search.
            let _query_image = request.get("query_image").and_then(Value::as_str);
            let query_started = Instant::now();

            let build_started = Instant::now();
            let needs_build = index.is_none();
            if needs_build {
                let bm25 = if config.retrieval_mode.uses_bm25() {
                    Some(LongMemEvalV2Bm25IndexV1::build(
                        config.domain.clone(),
                        trajectories,
                        config.slice_radius,
                        config.profile,
                    ).map_err(|error| format!("build trajectory BM25 index: {error:?}"))?)
                } else {
                    None
                };
                let entity_literal = if config.retrieval_mode.uses_entity_literal() {
                    Some(LongMemEvalV2EntityLiteralIndexV1::build(
                        config.domain.clone(),
                        trajectories,
                        config.slice_radius,
                    ).map_err(|error| format!("build trajectory entity-literal index: {error:?}"))?)
                } else {
                    None
                };
                *index = Some(RetrievalIndexesV1 { bm25, entity_literal });
            }
            let index_build_ns = if needs_build {
                u64::try_from(build_started.elapsed().as_nanos()).unwrap_or(u64::MAX)
            } else {
                0
            };
            let index_ref = index.as_ref().ok_or("indexes unavailable after build")?;
            let mut operations = Vec::<RetrievalOperationV1>::with_capacity(2);
            let mut bm25_hit_count = 0_usize;
            let mut entity_literal_hit_count = 0_usize;

            let result = match config.retrieval_mode {
                RetrievalModeV1::Bm25Only => {
                    let retrieved = index_ref.bm25.as_ref()
                        .ok_or("BM25 index unavailable for BM25-only mode")?
                        .search(&config.domain, query, config.top_k, config.neighbor_radius)
                        .map_err(|error| format!("trajectory BM25 query: {error:?}"))?;
                    bm25_hit_count = retrieved.hits.len();
                    operations.push(RetrievalOperationV1 {
                        tool_id: "trajectory.bm25.search".to_owned(),
                        tool_revision_sha256: config.backend_revision_sha256.clone(),
                        action_template_sha256: bm25_action_template_sha256(config),
                        observed_evidence_ids: trace_evidence_ids(&retrieved.hits),
                        outcome: if retrieved.hits.is_empty() {
                            RetrievalOperationOutcomeV1::NoEvidence
                        } else {
                            RetrievalOperationOutcomeV1::EvidenceObserved
                        },
                        elapsed_ns: retrieved.search_latency_ns,
                        inspected_bytes: 0,
                        inspected_bytes_measured: false,
                    });
                    LongMemEvalV2SearchResultV1 {
                        hits: retrieved.hits,
                        neighbor_context: retrieved.neighbor_context,
                        search_latency_ns: retrieved.search_latency_ns,
                    }
                }
                RetrievalModeV1::EntityLiteralOnly => {
                    let retrieved = index_ref.entity_literal.as_ref()
                        .ok_or("entity-literal index unavailable for entity-only mode")?
                        .search(&config.domain, query, config.top_k)
                        .map_err(|error| format!("trajectory entity-literal query: {error:?}"))?;
                    entity_literal_hit_count = retrieved.hits.len();
                    operations.push(RetrievalOperationV1 {
                        tool_id: "trajectory.entity_literal.search".to_owned(),
                        tool_revision_sha256: config.backend_revision_sha256.clone(),
                        action_template_sha256: entity_literal_action_template_sha256(config),
                        observed_evidence_ids: trace_evidence_ids(&retrieved.hits),
                        outcome: if retrieved.hits.is_empty() {
                            RetrievalOperationOutcomeV1::NoEvidence
                        } else {
                            RetrievalOperationOutcomeV1::EvidenceObserved
                        },
                        elapsed_ns: retrieved.search_latency_ns,
                        inspected_bytes: 0,
                        inspected_bytes_measured: false,
                    });
                    LongMemEvalV2SearchResultV1 {
                        hits: retrieved.hits,
                        neighbor_context: Vec::new(),
                        search_latency_ns: retrieved.search_latency_ns,
                    }
                }
                RetrievalModeV1::HybridRrf => {
                    let bm25 = index_ref.bm25.as_ref()
                        .ok_or("BM25 index unavailable for hybrid mode")?
                        .search(&config.domain, query, config.top_k, config.neighbor_radius)
                        .map_err(|error| format!("trajectory BM25 query: {error:?}"))?;
                    let entity = index_ref.entity_literal.as_ref()
                        .ok_or("entity-literal index unavailable for hybrid mode")?
                        .search(&config.domain, query, config.top_k)
                        .map_err(|error| format!("trajectory entity-literal query: {error:?}"))?;
                    bm25_hit_count = bm25.hits.len();
                    entity_literal_hit_count = entity.hits.len();
                    operations.push(RetrievalOperationV1 {
                        tool_id: "trajectory.bm25.search".to_owned(),
                        tool_revision_sha256: config.backend_revision_sha256.clone(),
                        action_template_sha256: bm25_action_template_sha256(config),
                        observed_evidence_ids: trace_evidence_ids(&bm25.hits),
                        outcome: if bm25.hits.is_empty() {
                            RetrievalOperationOutcomeV1::NoEvidence
                        } else {
                            RetrievalOperationOutcomeV1::EvidenceObserved
                        },
                        elapsed_ns: bm25.search_latency_ns,
                        inspected_bytes: 0,
                        inspected_bytes_measured: false,
                    });
                    operations.push(RetrievalOperationV1 {
                        tool_id: "trajectory.entity_literal.search".to_owned(),
                        tool_revision_sha256: config.backend_revision_sha256.clone(),
                        action_template_sha256: entity_literal_action_template_sha256(config),
                        observed_evidence_ids: trace_evidence_ids(&entity.hits),
                        outcome: if entity.hits.is_empty() {
                            RetrievalOperationOutcomeV1::NoEvidence
                        } else {
                            RetrievalOperationOutcomeV1::EvidenceObserved
                        },
                        elapsed_ns: entity.search_latency_ns,
                        inspected_bytes: 0,
                        inspected_bytes_measured: false,
                    });
                    hybrid_rrf_result(
                        &bm25.hits,
                        &entity.hits,
                        bm25.neighbor_context,
                        config.top_k,
                        bm25.search_latency_ns.saturating_add(entity.search_latency_ns),
                    )
                }
            };


            let mut items = Vec::new();
            let mut seen_images = BTreeSet::new();
            for hit in &result.hits {
                append_hit(&mut items, &mut seen_images, hit, &config.dataset_root)?;
            }
            for neighbor in &result.neighbor_context {
                items.push(json!({
                    "type": "text",
                    "value": neighbor.chunk.evidence_text()
                }));
                for state in &neighbor.chunk.state_window {
                    let resolved = resolve_screenshot_reference(
                        &config.dataset_root,
                        &state.screenshot_reference,
                    )
                    .map_err(|error| format!("resolve neighbor screenshot: {error:?}"))?;
                    if let Some(expected_digest) = &state.screenshot_sha256 {
                        let bytes = fs::read(&resolved)
                            .map_err(|_| "neighbor screenshot is missing after resolution")?;
                        let actual_digest = sha256_hex(&bytes);
                        if actual_digest != *expected_digest {
                            return Err("neighbor screenshot digest mismatch".into());
                        }
                    }
                    let value = resolved.to_string_lossy().into_owned();
                    if seen_images.insert(value.clone()) {
                        items.push(json!({"type":"image","value":value}));
                    }
                }
            }
            let candidate_evidence = if result.hits.is_empty() {
                None
            } else {
                let identity = LongMemEvalV2CandidateIdentityV1 {
                    run_id: config.run_id.clone(),
                    candidate_id: candidate_id.to_owned(),
                    backend_id: "mace_lmev2_hybrid_rrf_bm25_entity".to_owned(),
                    backend_revision_sha256: config.backend_revision_sha256.clone(),
                };
                Some(
                    encode_longmemeval_v2_search_candidate_evidence_for_channel_v1(
                        &identity,
                        query,
                        &config.domain,
                        &result,
                        "hybrid_rrf_bm25_entity",
                    )
                    .map_err(|error| format!("candidate evidence projection: {error:?}"))?
                )
            };
            let candidate_evidence_jsonl = match &candidate_evidence {
                Some(batch) => batch
                    .to_jsonl()
                    .map_err(|error| format!("candidate evidence serialization: {error:?}"))?,
                None => String::new(),
            };
            let candidate_set_sha256 = candidate_evidence
                .as_ref()
                .map(|batch| batch.candidate_set_sha256.clone());
            let candidate_record_count = candidate_evidence
                .as_ref()
                .map_or(0, |batch| batch.evidence.len());

            let observed_evidence_ids = candidate_evidence
                .as_ref()
                .map(|batch| batch.evidence.iter()
                    .map(|item| item.evidence_id.clone())
                    .collect::<Vec<_>>())
                .unwrap_or_default();
            let bm25_outcome = if bm25_evidence_ids.is_empty() {
                RetrievalOperationOutcomeV1::NoEvidence
            } else {
                RetrievalOperationOutcomeV1::EvidenceObserved
            };
            let entity_outcome = if entity_evidence_ids.is_empty() {
                RetrievalOperationOutcomeV1::NoEvidence
            } else {
                RetrievalOperationOutcomeV1::EvidenceObserved
            };
            let stop_reason = if observed_evidence_ids.is_empty() {
                RetrievalStopReasonV1::NoCandidateEvidence
            } else {
                RetrievalStopReasonV1::EvidenceSufficient
            };
            let query_elapsed_ns = u64::try_from(query_started.elapsed().as_nanos())
                .unwrap_or(u64::MAX);
            let trace_elapsed_ns = bm25_result.search_latency_ns
                .saturating_add(entity_literal_result.search_latency_ns);
            let created_at_unix = unix_now_seconds()?;
            let trace = RetrievalTraceReceiptV1 {
                run_id: config.run_id.clone(),
                query_invocation_id: candidate_id.to_owned(),
                domain_id: config.domain.clone(),
                environment_revision_sha256: config.environment_revision_sha256.clone(),
                tool_profile_sha256: config.backend_revision_sha256.clone(),
                query_sha256: sha256_hex(query.as_bytes()),
                candidate_set_sha256: candidate_set_sha256.clone(),
                operations: vec![
                    RetrievalOperationV1 {
                        tool_id: "trajectory.bm25.search".to_owned(),
                        tool_revision_sha256: config.backend_revision_sha256.clone(),
                        action_template_sha256: bm25_action_template_sha256(config),
                        observed_evidence_ids: bm25_evidence_ids,
                        outcome: bm25_outcome,
                        elapsed_ns: bm25_result.search_latency_ns,
                        // Index/posting reads are not fully instrumented yet.
                        inspected_bytes: 0,
                        inspected_bytes_measured: false,
                    },
                    RetrievalOperationV1 {
                        tool_id: "trajectory.entity_literal.search".to_owned(),
                        tool_revision_sha256: config.backend_revision_sha256.clone(),
                        action_template_sha256: entity_literal_action_template_sha256(config),
                        observed_evidence_ids: entity_evidence_ids,
                        outcome: entity_outcome,
                        elapsed_ns: entity_literal_result.search_latency_ns,
                        // Structured-literal posting reads are not fully instrumented yet.
                        inspected_bytes: 0,
                        inspected_bytes_measured: false,
                    },
                ],
                stop_reason,
                tool_call_budget: 2,
                tool_calls_used: 2,
                elapsed_ns: trace_elapsed_ns,
                inspected_bytes: 0,
                inspected_bytes_complete: false,
                created_at_unix,
            };
            let retrieval_trace_sha256 = policy_trace_store
                .record_trace(policy_memory, &trace)
                .map_err(|error| format!("persist retrieval policy trace: {error}"))?;
            let retrieval_trace_json = trace.to_json_value();

            Ok(json!({
                "ok": true,
                "items": items,
                "candidate_evidence_jsonl": candidate_evidence_jsonl,
                "candidate_set_sha256": candidate_set_sha256,
                "candidate_record_count": candidate_record_count,
                "retrieval_trace": retrieval_trace_json,
                "retrieval_trace_sha256": retrieval_trace_sha256,
                "policy_trace_count": policy_memory.trace_count(),
                "run_id": config.run_id,
                "candidate_id": candidate_id,
                "hit_count": result.hits.len(),
                "bm25_hit_count": bm25_result.hits.len(),
                "entity_literal_hit_count": entity_literal_result.hits.len(),
                "neighbor_count": result.neighbor_context.len(),
                "index_build_ns": index_build_ns,
                "search_latency_ns": result.search_latency_ns,
                "query_total_ns": query_elapsed_ns,
                "retrieval_fusion": "reciprocal_rank_fusion_k60",
                "query_image_used": false,
                "retrieval_mode": "hybrid_bm25_entity_literal_with_original_image_evidence"
            }))
        }
        other => Err(format!("unsupported sidecar operation: {other}")),
    }
}

fn main() {
    let config = Config::from_env().unwrap_or_else(|error| {
        eprintln!("mace_lmev2_stdio configuration error: {error}");
        std::process::exit(2);
    });
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());
    let policy_trace_store = RetrievalPolicyTraceFileStoreV1::open(&config.policy_trace_dir)
        .unwrap_or_else(|error| {
            eprintln!("mace_lmev2_stdio policy trace store error: {error}");
            std::process::exit(2);
        });
    let mut policy_memory = RetrievalPolicyMemoryV1::default();
    policy_trace_store.load_into(&mut policy_memory).unwrap_or_else(|error| {
        eprintln!("mace_lmev2_stdio policy trace replay error: {error}");
        std::process::exit(2);
    });
    let mut trajectories = Vec::<LongMemEvalV2TrajectoryV1>::new();
    let mut index = None;

    for line in stdin.lock().lines() {
        let response = match line {
            Ok(line) => match serde_json::from_str::<Value>(&line) {
                Ok(request) => handle_request(
                    request,
                    &config,
                    &mut trajectories,
                    &mut index,
                    &mut policy_memory,
                    &policy_trace_store,
                )
                    .unwrap_or_else(|error| json!({"ok": false, "error": error})),
                Err(error) => json!({"ok": false, "error": format!("invalid request JSON: {error}")}),
            },
            Err(error) => json!({"ok": false, "error": format!("read request: {error}")}),
        };
        if serde_json::to_writer(&mut writer, &response).is_err()
            || writer.write_all(b"\n").is_err()
            || writer.flush().is_err()
        {
            eprintln!("mace_lmev2_stdio: failed writing response");
            std::process::exit(1);
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = env::temp_dir().join(format!(
            "mace-lmev2-sidecar-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).expect("fixture root");
        root
    }

    fn fixture_config(root: PathBuf) -> Config {
        Config {
            domain: "web".to_owned(),
            run_id: "fixture-run".to_owned(),
            backend_revision_sha256: "1111111111111111111111111111111111111111111111111111111111111111".to_owned(),
            dataset_root: root.clone(),
            top_k: 4,
            slice_radius: 0,
            neighbor_radius: 0,
            environment_revision_sha256: "2222222222222222222222222222222222222222222222222222222222222222".to_owned(),
            policy_trace_dir: root.join(".policy-traces"),
            profile: Bm25ProfileV1 {
                k1: 1.2,
                b: 0.75,
                tokenizer_version: "unicode_alphanumeric_v1",
            },
        }
    }

    fn fixture_trajectory(root: &Path, image_bytes: &[u8]) -> Value {
        let trajectory_dir = root.join("trajectory-1");
        fs::create_dir_all(&trajectory_dir).expect("trajectory fixture directory");
        fs::write(trajectory_dir.join("state.png"), image_bytes).expect("image fixture");
        json!({
            "id": "trajectory-1",
            "domain": "web",
            "goal": "Create an order for Ada",
            "start_url": "https://shop.example/",
            "outcome": "success",
            "states": [{
                "url": "https://shop.example/orders/42",
                "action": "submit order",
                "thoughts": "Order submitted",
                "accessibility_tree": "Order details. Customer Ada. Status pending.",
                "screenshot": "trajectory-1/state.png",
                "state_index": 42,
                "step": 1,
                "mace_screenshot_sha256": sha256_hex(image_bytes)
            }]
        })
    }

    #[test]
    fn reciprocal_rank_fusion_deduplicates_cross_channel_hits() {
        let root = fixture_root("rrf-dedup");
        let trajectory = fixture_trajectory(&root, b"rrf fixture image");
        let config = fixture_config(root.clone());
        let parsed = LongMemEvalV2TrajectoryV1::parse_jsonl_record(
            &serde_json::to_string(&trajectory).expect("trajectory JSON")
        ).expect("trajectory parse");
        let bm25 = LongMemEvalV2Bm25IndexV1::build(
            "web", &[parsed.clone()], 0, config.profile
        ).expect("BM25 index");
        let entity = LongMemEvalV2EntityLiteralIndexV1::build(
            "web", &[parsed], 0
        ).expect("entity-literal index");
        let bm25_result = bm25.search("web", "42", 4, 0).expect("BM25 literal");
        let entity_result = entity.search("web", "42", 4).expect("entity literal");
        assert_eq!(bm25_result.hits.len(), 1);
        assert_eq!(entity_result.hits.len(), 1);

        let fused = hybrid_rrf_result(
            &bm25_result.hits,
            &entity_result.hits,
            Vec::new(),
            4,
            10,
        );
        assert_eq!(fused.hits.len(), 1);
        assert_eq!(fused.hits[0].rank, 1);
        assert!((fused.hits[0].score - (2.0 / 61.0)).abs() < 1e-12);
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn official_memory_api_request_returns_text_and_verified_image_context_items() {
        let root = fixture_root("context");
        let image_bytes = b"verified screenshot bytes";
        let trajectory = fixture_trajectory(&root, image_bytes);
        let expected_image = fs::canonicalize(root.join("trajectory-1/state.png"))
            .expect("canonical fixture image");
        let config = fixture_config(root.clone());
        let policy_trace_store = RetrievalPolicyTraceFileStoreV1::open(&config.policy_trace_dir)
            .expect("open policy trace store");
        let mut trajectories = Vec::new();
        let mut index = None;
        let mut policy_memory = RetrievalPolicyMemoryV1::default();

        let inserted = handle_request(
            json!({"op": "insert", "trajectory": trajectory}),
            &config,
            &mut trajectories,
            &mut index,
            &mut policy_memory,
            &policy_trace_store,
        ).expect("insert");
        assert_eq!(inserted["ok"], true);
        assert_eq!(inserted["inserted"], 1);

        let response = handle_request(
            json!({
                "op": "query",
                "query": "customer Ada pending order",
                "query_image": "/private/query-image.png",
                "candidate_id": "query-invocation-1"
            }),
            &config,
            &mut trajectories,
            &mut index,
            &mut policy_memory,
            &policy_trace_store,
        ).expect("query");
        assert_eq!(response["ok"], true);
        assert_eq!(response["query_image_used"], false);
        assert_eq!(
            response["retrieval_mode"],
            "hybrid_bm25_entity_literal_with_original_image_evidence"
        );
        assert_eq!(response["retrieval_fusion"], "reciprocal_rank_fusion_k60");
        let operations = response["retrieval_trace"]["operations"]
            .as_array().expect("two retrieval action receipts");
        assert_eq!(operations.len(), 2);
        assert_eq!(operations[0]["tool_id"], "trajectory.bm25.search");
        assert_eq!(operations[0]["outcome"], "evidence_observed");
        assert_eq!(operations[1]["tool_id"], "trajectory.entity_literal.search");
        assert_eq!(operations[1]["outcome"], "no_evidence");
        assert!(response["candidate_evidence_jsonl"].as_str().unwrap()
            .contains("hybrid_rrf_bm25_entity"));
        assert_eq!(response["run_id"], "fixture-run");
        assert_eq!(response["candidate_id"], "query-invocation-1");
        assert_eq!(response["candidate_record_count"], 3);
        assert!(response["candidate_set_sha256"].as_str().is_some());
        assert!(response["candidate_evidence_jsonl"].as_str().unwrap().contains("lmev2:text:"));
        assert_eq!(response["retrieval_trace"]["run_id"], "fixture-run");
        assert_eq!(response["retrieval_trace"]["query_invocation_id"], "query-invocation-1");
        assert_eq!(response["retrieval_trace"]["inspected_bytes_complete"], false);
        assert!(response["retrieval_trace_sha256"].as_str().is_some());
        assert_eq!(response["policy_trace_count"], 1);
        assert_eq!(
            fs::read_dir(&config.policy_trace_dir).expect("trace store directory").count(),
            1
        );
        let reopened_store = RetrievalPolicyTraceFileStoreV1::open(&config.policy_trace_dir)
            .expect("reopen durable policy trace store");
        let mut restored_policy_memory = RetrievalPolicyMemoryV1::default();
        assert_eq!(
            reopened_store.load_into(&mut restored_policy_memory).expect("restore policy traces"),
            1
        );
        assert_eq!(restored_policy_memory.trace_count(), 1);
        assert_eq!(
            restored_policy_memory.trace_receipt_sha256("fixture-run", "query-invocation-1"),
            response["retrieval_trace_sha256"].as_str()
        );

        let literal_response = handle_request(
            json!({
                "op": "query",
                "query": "state 42",
                "candidate_id": "query-invocation-2"
            }),
            &config,
            &mut trajectories,
            &mut index,
            &mut policy_memory,
            &policy_trace_store,
        ).expect("literal query");
        assert_eq!(literal_response["ok"], true);
        let literal_operations = literal_response["retrieval_trace"]["operations"]
            .as_array().expect("two action receipts");
        assert_eq!(literal_operations.len(), 2);
        assert_eq!(literal_operations[1]["outcome"], "evidence_observed");
        assert!(literal_response["candidate_evidence_jsonl"].as_str().unwrap()
            .contains("hybrid_rrf_bm25_entity"));
        let items = response["items"].as_array().expect("context items");
        assert!(items.iter().any(|item| {
            item["type"] == "text" && item["value"].as_str()
                .is_some_and(|value| value.contains("Customer Ada"))
        }));
        assert!(items.iter().any(|item| {
            item["type"] == "image" && item["value"].as_str()
                == expected_image.to_str()
        }));
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn duplicate_insert_is_rejected_before_reindexing() {
        let root = fixture_root("duplicate");
        let trajectory = fixture_trajectory(&root, b"image bytes");
        let config = fixture_config(root.clone());
        let policy_trace_store = RetrievalPolicyTraceFileStoreV1::open(&config.policy_trace_dir)
            .expect("open policy trace store");
        let mut trajectories = Vec::new();
        let mut index = None;
        let mut policy_memory = RetrievalPolicyMemoryV1::default();

        handle_request(
            json!({"op": "insert", "trajectory": trajectory.clone()}),
            &config,
            &mut trajectories,
            &mut index,
            &mut policy_memory,
            &policy_trace_store,
        ).expect("first insert");
        let result = handle_request(
            json!({"op": "insert", "trajectory": trajectory}),
            &config,
            &mut trajectories,
            &mut index,
            &mut policy_memory,
            &policy_trace_store,
        );
        assert!(result.is_err());
        assert_eq!(trajectories.len(), 1);
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn screenshot_byte_replacement_fails_before_image_context_is_returned() {
        let root = fixture_root("digest-mismatch");
        let trajectory = fixture_trajectory(&root, b"original bytes");
        let config = fixture_config(root.clone());
        let policy_trace_store = RetrievalPolicyTraceFileStoreV1::open(&config.policy_trace_dir)
            .expect("open policy trace store");
        let image_path = root.join("trajectory-1/state.png");
        let mut trajectories = Vec::new();
        let mut index = None;
        let mut policy_memory = RetrievalPolicyMemoryV1::default();

        handle_request(
            json!({"op": "insert", "trajectory": trajectory}),
            &config,
            &mut trajectories,
            &mut index,
            &mut policy_memory,
            &policy_trace_store,
        ).expect("insert");
        fs::write(&image_path, b"replaced bytes").expect("tamper image bytes");

        let result = handle_request(
            json!({"op": "query", "query": "customer Ada pending order"}),
            &config,
            &mut trajectories,
            &mut index,
            &mut policy_memory,
            &policy_trace_store,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("ScreenshotDigestMismatch"));
        fs::remove_dir_all(root).expect("cleanup");
    }
}
