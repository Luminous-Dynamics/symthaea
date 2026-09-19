// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! MATH-RET-RUNTIME-001A — owner-independent qualified retrieval execution seam.
//!
//! This package intentionally has no dependency on the cognitive loop, HDC,
//! mathematical memory, Phi, theorem authority, or a similarity threshold.
//! It isolates deterministic retrieval execution and evidence emission before
//! any legacy runtime path is allowed to participate.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sha256Digest(String);

impl Sha256Digest {
    pub fn parse(value: impl Into<String>) -> Result<Self, DigestError> {
        let value = value.into();
        let body = value
            .strip_prefix("sha256:")
            .ok_or_else(|| DigestError(value.clone()))?;
        if body.len() != 64
            || !body
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(DigestError(value));
        }
        Ok(Self(format!("sha256:{body}")))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DigestError(String);

impl fmt::Display for DigestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid SHA-256 identity: {}", self.0)
    }
}

impl Error for DigestError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphIdentity {
    pub bundle_sha256: Sha256Digest,
    pub graph_report_sha256: Sha256Digest,
    pub experiment_sha256: Sha256Digest,
    pub retrieval_binding_sha256: Sha256Digest,
    pub candidate_set_sha256: Sha256Digest,
    pub candidate_count: usize,
    pub context_packer_sha256: Sha256Digest,
    pub source_object_contract_sha256: Sha256Digest,
    pub source_fetch_policy_sha256: Sha256Digest,
    pub payload_serialization_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetrievalBudget {
    pub max_output_items: usize,
    pub max_output_bytes: usize,
    pub max_output_item_bytes: usize,
    pub max_retrieval_queries: u32,
    /// Fixed-point compute accounting. One unit = 1e-6 normalized compute units.
    pub max_normalized_compute_microunits: u64,
    pub max_wall_time_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualifiedRetrievalRequest {
    pub trace_id: String,
    pub audit_id: String,
    pub experiment_id: String,
    pub arm_id: String,
    pub experiment_seed: u64,
    pub query_id: String,
    pub query_source_object_sha256: Sha256Digest,
    pub graph: GraphIdentity,
    pub budget: RetrievalBudget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RetrievalChannel {
    Syntax,
    ExactNormalForm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlTransform {
    None,
    RandomRetrieval,
    ShuffledHdcVectors,
    PermutedChallengeAssociations,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlBinding {
    pub index_manifest_sha256: Sha256Digest,
    pub control_transform: ControlTransform,
    pub control_seed: Option<u64>,
    pub control_artifact_sha256: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SingleIndexExecution {
    pub index_manifest_sha256: Sha256Digest,
    pub index_artifact_sha256: Sha256Digest,
    pub requested_k: u32,
    pub ranked_source_object_digests: Vec<Sha256Digest>,
    pub control: ControlBinding,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelExecution {
    pub channel: RetrievalChannel,
    pub index_manifest_sha256: Sha256Digest,
    pub index_artifact_sha256: Sha256Digest,
    pub requested_k: u32,
    pub input_bytes_used: usize,
    pub ranked_source_object_digests: Vec<Sha256Digest>,
    pub control: ControlBinding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionMethod {
    ReciprocalRankFusion { rrf_k: u32 },
    DeterministicInterleave { start_channel: RetrievalChannel },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionExecution {
    pub fusion_policy_sha256: Sha256Digest,
    pub method: FusionMethod,
    pub syntax: ChannelExecution,
    pub normal_form: ChannelExecution,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendRetrieval {
    SingleIndex(SingleIndexExecution),
    Fusion(FusionExecution),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendExecution {
    pub retrieval: BackendRetrieval,
    pub retrieval_queries_used: u32,
    pub wall_time_ms_used: u64,
    pub normalized_compute_microunits_used: u64,
}

/// Representation-specific retrieval. It returns source identities only.
/// Canonical payload bytes are resolved later by one shared materializer.
pub trait QualifiedRetrievalBackend {
    fn execute(
        &mut self,
        request: &QualifiedRetrievalRequest,
    ) -> Result<BackendExecution, RetrievalError>;
}

/// Shared source-object -> canonical-payload boundary.
///
/// A production implementation must correspond to the request's frozen
/// source-object contract, source-fetch policy, and payload serialization.
pub trait CanonicalSourceMaterializer {
    fn materialize(
        &mut self,
        source_object_sha256: &Sha256Digest,
    ) -> Result<Vec<u8>, RetrievalError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelTrace {
    pub channel: RetrievalChannel,
    pub index_manifest_sha256: Sha256Digest,
    pub index_artifact_sha256: Sha256Digest,
    pub requested_k: u32,
    pub input_bytes_used: usize,
    pub ranked_source_object_digests: Vec<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceRetrieval {
    SingleIndex {
        index_manifest_sha256: Sha256Digest,
        index_artifact_sha256: Sha256Digest,
        requested_k: u32,
        ranked_source_object_digests: Vec<Sha256Digest>,
    },
    Fusion {
        fusion_policy_sha256: Sha256Digest,
        channels: [ChannelTrace; 2],
        fused_ranked_source_object_digests: Vec<Sha256Digest>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedItem {
    pub rank: usize,
    pub source_object_sha256: Sha256Digest,
    pub canonical_payload_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Empty,
    RankedCandidatesExhausted,
    ItemLimitReached,
    FirstNonFittingItem,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackingTrace {
    pub context_packer_sha256: Sha256Digest,
    pub input_ranked_source_object_digests: Vec<Sha256Digest>,
    pub output: Vec<PackedItem>,
    pub output_items_used: usize,
    pub output_bytes_used: usize,
    pub stop_reason: StopReason,
    pub first_nonfitting: Option<PackedItem>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceUsage {
    pub retrieval_queries_used: u32,
    pub wall_time_ms_used: u64,
    pub normalized_compute_microunits_used: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetrievalTraceReceipt {
    pub version: &'static str,
    pub authority: &'static str,
    pub trace_id: String,
    pub experiment_id: String,
    pub arm_id: String,
    pub experiment_seed: u64,
    pub query_id: String,
    pub query_source_object_sha256: Sha256Digest,
    pub graph: GraphIdentity,
    pub retrieval: TraceRetrieval,
    pub packing: PackingTrace,
    pub resources: ResourceUsage,
    pub control_bindings: Vec<ControlBinding>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadRole {
    Delivered,
    FirstNonFitting,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadAuditTarget {
    pub role: PayloadRole,
    pub rank: usize,
    pub source_object_sha256: Sha256Digest,
    pub canonical_payload_utf8: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadAuditBatch {
    pub version: &'static str,
    pub authority: &'static str,
    pub audit_id: String,
    pub trace_id: String,
    pub graph_report_sha256: Sha256Digest,
    pub context_packer_sha256: Sha256Digest,
    pub source_object_contract_sha256: Sha256Digest,
    pub source_fetch_policy_sha256: Sha256Digest,
    pub payload_serialization_sha256: Sha256Digest,
    pub entries: Vec<PayloadAuditTarget>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceBundle {
    pub trace: RetrievalTraceReceipt,
    pub payload_audit: PayloadAuditBatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SinkError(pub String);

impl fmt::Display for SinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for SinkError {}

/// Stage the runtime trace generated from the immutable execution result.
pub trait RetrievalTraceSink {
    fn stage_trace(&mut self, trace: &RetrievalTraceReceipt) -> Result<(), SinkError>;
}

/// Stage materialization targets derived from the exact same packing decision.
pub trait PayloadAuditSink {
    fn stage_payload_audit(&mut self, audit: &PayloadAuditBatch) -> Result<(), SinkError>;
}

/// Transactional evidence sink.
///
/// Implementations MUST make `commit` all-or-nothing. On `Err`, no staged
/// evidence may become durably visible. `abort` must clear all staged state.
pub trait AtomicEvidenceSink: RetrievalTraceSink + PayloadAuditSink {
    fn commit(&mut self) -> Result<(), SinkError>;
    fn abort(&mut self);
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualifiedRetrievalOutcome {
    pub trace: RetrievalTraceReceipt,
    pub payload_audit: PayloadAuditBatch,
    pub evidence_committed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetrievalError {
    InvalidRequest(String),
    Candidate(String),
    Budget(String),
    Fusion(String),
    Backend(String),
    Materialization(String),
    Evidence(String),
}

impl fmt::Display for RetrievalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(s) => write!(f, "invalid request: {s}"),
            Self::Candidate(s) => write!(f, "candidate violation: {s}"),
            Self::Budget(s) => write!(f, "budget violation: {s}"),
            Self::Fusion(s) => write!(f, "fusion violation: {s}"),
            Self::Backend(s) => write!(f, "backend failure: {s}"),
            Self::Materialization(s) => write!(f, "materialization failure: {s}"),
            Self::Evidence(s) => write!(f, "evidence emission failure: {s}"),
        }
    }
}

impl Error for RetrievalError {}

pub struct RetrievalExecutor;

impl RetrievalExecutor {
    pub fn execute<B, M, S>(
        backend: &mut B,
        materializer: &mut M,
        sink: &mut S,
        request: QualifiedRetrievalRequest,
    ) -> Result<QualifiedRetrievalOutcome, RetrievalError>
    where
        B: QualifiedRetrievalBackend,
        M: CanonicalSourceMaterializer,
        S: AtomicEvidenceSink,
    {
        validate_request(&request)?;
        let backend_execution = backend.execute(&request)?;
        validate_resources(&request.budget, &backend_execution)?;

        let ProcessedRetrieval {
            trace_retrieval,
            final_ranked,
            controls,
            expected_retrieval_queries,
        } = process_backend_retrieval(&request, backend_execution.retrieval)?;

        if backend_execution.retrieval_queries_used != expected_retrieval_queries {
            return Err(RetrievalError::Budget(format!(
                "reported retrieval_queries_used={} but mode requires {}",
                backend_execution.retrieval_queries_used, expected_retrieval_queries
            )));
        }

        let packing = pack_ranked(
            &final_ranked,
            materializer,
            &request.budget,
            &request.graph.context_packer_sha256,
        )?;
        let resources = ResourceUsage {
            retrieval_queries_used: backend_execution.retrieval_queries_used,
            wall_time_ms_used: backend_execution.wall_time_ms_used,
            normalized_compute_microunits_used: backend_execution.normalized_compute_microunits_used,
        };

        let trace = RetrievalTraceReceipt {
            version: "math-retrieval-trace-v1",
            authority: "MeasurementOnly",
            trace_id: request.trace_id.clone(),
            experiment_id: request.experiment_id.clone(),
            arm_id: request.arm_id.clone(),
            experiment_seed: request.experiment_seed,
            query_id: request.query_id.clone(),
            query_source_object_sha256: request.query_source_object_sha256.clone(),
            graph: request.graph.clone(),
            retrieval: trace_retrieval,
            packing: packing.trace.clone(),
            resources,
            control_bindings: controls,
        };

        let payload_audit = PayloadAuditBatch {
            version: "math-retrieval-payload-audit-v1",
            authority: "MeasurementOnly",
            audit_id: request.audit_id,
            trace_id: trace.trace_id.clone(),
            graph_report_sha256: request.graph.graph_report_sha256,
            context_packer_sha256: request.graph.context_packer_sha256,
            source_object_contract_sha256: request.graph.source_object_contract_sha256,
            source_fetch_policy_sha256: request.graph.source_fetch_policy_sha256,
            payload_serialization_sha256: request.graph.payload_serialization_sha256,
            entries: packing.audit_targets,
        };

        if let Err(err) = sink.stage_trace(&trace) {
            sink.abort();
            return Err(RetrievalError::Evidence(err.to_string()));
        }
        if let Err(err) = sink.stage_payload_audit(&payload_audit) {
            sink.abort();
            return Err(RetrievalError::Evidence(err.to_string()));
        }
        if let Err(err) = sink.commit() {
            sink.abort();
            return Err(RetrievalError::Evidence(err.to_string()));
        }

        Ok(QualifiedRetrievalOutcome {
            trace,
            payload_audit,
            evidence_committed: true,
        })
    }
}

fn validate_request(request: &QualifiedRetrievalRequest) -> Result<(), RetrievalError> {
    for (name, value) in [
        ("trace_id", request.trace_id.as_str()),
        ("audit_id", request.audit_id.as_str()),
        ("experiment_id", request.experiment_id.as_str()),
        ("arm_id", request.arm_id.as_str()),
        ("query_id", request.query_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(RetrievalError::InvalidRequest(format!(
                "{name} must be non-empty"
            )));
        }
    }
    if request.graph.candidate_count == 0 {
        return Err(RetrievalError::InvalidRequest(
            "candidate_count must be positive".into(),
        ));
    }
    let b = &request.budget;
    if b.max_output_items == 0
        || b.max_output_bytes == 0
        || b.max_output_item_bytes == 0
        || b.max_retrieval_queries == 0
        || b.max_normalized_compute_microunits == 0
        || b.max_wall_time_ms == 0
    {
        return Err(RetrievalError::InvalidRequest(
            "all runtime budget ceilings must be positive".into(),
        ));
    }
    if b.max_output_item_bytes > b.max_output_bytes {
        return Err(RetrievalError::InvalidRequest(
            "per-item bytes cannot exceed total context bytes".into(),
        ));
    }
    Ok(())
}

fn validate_resources(
    budget: &RetrievalBudget,
    execution: &BackendExecution,
) -> Result<(), RetrievalError> {
    if execution.retrieval_queries_used == 0
        || execution.retrieval_queries_used > budget.max_retrieval_queries
    {
        return Err(RetrievalError::Budget(
            "retrieval query usage outside frozen ceiling".into(),
        ));
    }
    if execution.wall_time_ms_used > budget.max_wall_time_ms {
        return Err(RetrievalError::Budget("wall-time ceiling exceeded".into()));
    }
    if execution.normalized_compute_microunits_used > budget.max_normalized_compute_microunits {
        return Err(RetrievalError::Budget(
            "normalized-compute ceiling exceeded".into(),
        ));
    }
    Ok(())
}

struct ProcessedRetrieval {
    trace_retrieval: TraceRetrieval,
    final_ranked: Vec<Sha256Digest>,
    controls: Vec<ControlBinding>,
    expected_retrieval_queries: u32,
}

fn process_backend_retrieval(
    request: &QualifiedRetrievalRequest,
    retrieval: BackendRetrieval,
) -> Result<ProcessedRetrieval, RetrievalError> {
    match retrieval {
        BackendRetrieval::SingleIndex(single) => process_single(request, single),
        BackendRetrieval::Fusion(fusion) => process_fusion(request, fusion),
    }
}

fn process_single(
    request: &QualifiedRetrievalRequest,
    single: SingleIndexExecution,
) -> Result<ProcessedRetrieval, RetrievalError> {
    if single.requested_k == 0 || single.requested_k as usize > request.budget.max_output_items {
        return Err(RetrievalError::Budget(
            "single-index requested_k exceeds frozen item ceiling".into(),
        ));
    }
    if single.ranked_source_object_digests.len() > single.requested_k as usize {
        return Err(RetrievalError::Candidate(
            "single-index returned more candidates than requested".into(),
        ));
    }
    validate_ranked(request, &single.ranked_source_object_digests)?;
    validate_control(&single.index_manifest_sha256, &single.control)?;

    let trace_retrieval = TraceRetrieval::SingleIndex {
        index_manifest_sha256: single.index_manifest_sha256,
        index_artifact_sha256: single.index_artifact_sha256,
        requested_k: single.requested_k,
        ranked_source_object_digests: single.ranked_source_object_digests.clone(),
    };
    Ok(ProcessedRetrieval {
        trace_retrieval,
        final_ranked: single.ranked_source_object_digests,
        controls: vec![single.control],
        expected_retrieval_queries: 1,
    })
}

fn process_fusion(
    request: &QualifiedRetrievalRequest,
    fusion: FusionExecution,
) -> Result<ProcessedRetrieval, RetrievalError> {
    if fusion.syntax.channel != RetrievalChannel::Syntax
        || fusion.normal_form.channel != RetrievalChannel::ExactNormalForm
    {
        return Err(RetrievalError::Fusion(
            "fusion must bind Syntax and ExactNormalForm channels exactly".into(),
        ));
    }
    for channel in [&fusion.syntax, &fusion.normal_form] {
        if channel.requested_k == 0 {
            return Err(RetrievalError::Budget(
                "fusion channel requested_k must be positive".into(),
            ));
        }
        if channel.ranked_source_object_digests.len() > channel.requested_k as usize {
            return Err(RetrievalError::Candidate(
                "fusion channel returned more candidates than requested".into(),
            ));
        }
        validate_ranked(request, &channel.ranked_source_object_digests)?;
        validate_control(&channel.index_manifest_sha256, &channel.control)?;
    }
    let input_item_opportunity =
        fusion.syntax.requested_k as usize + fusion.normal_form.requested_k as usize;
    if input_item_opportunity > request.budget.max_output_items {
        return Err(RetrievalError::Budget(
            "fusion input opportunity exceeds shared item ceiling".into(),
        ));
    }
    let input_bytes = fusion
        .syntax
        .input_bytes_used
        .checked_add(fusion.normal_form.input_bytes_used)
        .ok_or_else(|| RetrievalError::Budget("fusion input byte accounting overflow".into()))?;
    if input_bytes > request.budget.max_output_bytes {
        return Err(RetrievalError::Budget(
            "fusion input opportunity exceeds shared byte ceiling".into(),
        ));
    }

    let final_ranked = match fusion.method {
        FusionMethod::ReciprocalRankFusion { rrf_k } => {
            if rrf_k == 0 {
                return Err(RetrievalError::Fusion("rrf_k must be positive".into()));
            }
            replay_rrf(
                rrf_k,
                &fusion.syntax.ranked_source_object_digests,
                &fusion.normal_form.ranked_source_object_digests,
            )?
        }
        FusionMethod::DeterministicInterleave { start_channel } => replay_interleave(
            start_channel,
            &fusion.syntax.ranked_source_object_digests,
            &fusion.normal_form.ranked_source_object_digests,
        ),
    };

    let syntax_trace = channel_trace(&fusion.syntax);
    let normal_trace = channel_trace(&fusion.normal_form);
    let trace_retrieval = TraceRetrieval::Fusion {
        fusion_policy_sha256: fusion.fusion_policy_sha256,
        channels: [syntax_trace, normal_trace],
        fused_ranked_source_object_digests: final_ranked.clone(),
    };
    Ok(ProcessedRetrieval {
        trace_retrieval,
        final_ranked,
        controls: vec![fusion.syntax.control, fusion.normal_form.control],
        expected_retrieval_queries: 2,
    })
}

fn channel_trace(channel: &ChannelExecution) -> ChannelTrace {
    ChannelTrace {
        channel: channel.channel,
        index_manifest_sha256: channel.index_manifest_sha256.clone(),
        index_artifact_sha256: channel.index_artifact_sha256.clone(),
        requested_k: channel.requested_k,
        input_bytes_used: channel.input_bytes_used,
        ranked_source_object_digests: channel.ranked_source_object_digests.clone(),
    }
}

fn validate_ranked(
    request: &QualifiedRetrievalRequest,
    ranked: &[Sha256Digest],
) -> Result<(), RetrievalError> {
    if ranked.len() > request.graph.candidate_count {
        return Err(RetrievalError::Candidate(
            "returned ranking exceeds frozen candidate universe".into(),
        ));
    }
    let mut seen = BTreeSet::new();
    for source in ranked {
        if source == &request.query_source_object_sha256 {
            return Err(RetrievalError::Candidate(
                "query source object leaked into retrieval results".into(),
            ));
        }
        if !seen.insert(source.clone()) {
            return Err(RetrievalError::Candidate(
                "duplicate source object within one ranked channel".into(),
            ));
        }
    }
    Ok(())
}

fn validate_control(
    index_manifest_sha256: &Sha256Digest,
    control: &ControlBinding,
) -> Result<(), RetrievalError> {
    if &control.index_manifest_sha256 != index_manifest_sha256 {
        return Err(RetrievalError::Candidate(
            "control binding references a different index manifest".into(),
        ));
    }
    match control.control_transform {
        ControlTransform::None => {
            if control.control_seed.is_some() || control.control_artifact_sha256.is_some() {
                return Err(RetrievalError::Candidate(
                    "non-control index must not carry control seed/artifact".into(),
                ));
            }
        }
        _ => {
            if control.control_seed.is_none() || control.control_artifact_sha256.is_none() {
                return Err(RetrievalError::Candidate(
                    "control index requires exact seed and control artifact".into(),
                ));
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RationalScore {
    numerator: u128,
    denominator: u128,
}

impl RationalScore {
    fn zero() -> Self {
        Self {
            numerator: 0,
            denominator: 1,
        }
    }

    fn add_reciprocal(&mut self, denominator: u32) {
        let d = denominator as u128;
        self.numerator = self.numerator * d + self.denominator;
        self.denominator *= d;
    }
}

fn replay_rrf(
    rrf_k: u32,
    syntax: &[Sha256Digest],
    normal: &[Sha256Digest],
) -> Result<Vec<Sha256Digest>, RetrievalError> {
    let mut scores: BTreeMap<Sha256Digest, RationalScore> = BTreeMap::new();
    for ranked in [syntax, normal] {
        for (index, source) in ranked.iter().enumerate() {
            let rank = u32::try_from(index + 1)
                .map_err(|_| RetrievalError::Fusion("rank exceeds u32".into()))?;
            let denominator = rrf_k
                .checked_add(rank)
                .ok_or_else(|| RetrievalError::Fusion("RRF denominator overflow".into()))?;
            scores
                .entry(source.clone())
                .or_insert_with(RationalScore::zero)
                .add_reciprocal(denominator);
        }
    }

    let mut ordered: Vec<(Sha256Digest, RationalScore)> = scores.into_iter().collect();
    ordered.sort_by(|(digest_a, score_a), (digest_b, score_b)| {
        let left = score_a.numerator * score_b.denominator;
        let right = score_b.numerator * score_a.denominator;
        right.cmp(&left).then_with(|| digest_a.cmp(digest_b))
    });
    Ok(ordered.into_iter().map(|(digest, _)| digest).collect())
}

fn replay_interleave(
    start_channel: RetrievalChannel,
    syntax: &[Sha256Digest],
    normal: &[Sha256Digest],
) -> Vec<Sha256Digest> {
    let mut syntax_index = 0usize;
    let mut normal_index = 0usize;
    let mut turn = start_channel;
    let mut seen = BTreeSet::new();
    let mut output = Vec::new();

    while syntax_index < syntax.len() || normal_index < normal.len() {
        if syntax_index >= syntax.len() {
            while normal_index < normal.len() {
                let source = normal[normal_index].clone();
                normal_index += 1;
                if seen.insert(source.clone()) {
                    output.push(source);
                }
            }
            break;
        }
        if normal_index >= normal.len() {
            while syntax_index < syntax.len() {
                let source = syntax[syntax_index].clone();
                syntax_index += 1;
                if seen.insert(source.clone()) {
                    output.push(source);
                }
            }
            break;
        }

        let source = match turn {
            RetrievalChannel::Syntax => {
                let source = syntax[syntax_index].clone();
                syntax_index += 1;
                turn = RetrievalChannel::ExactNormalForm;
                source
            }
            RetrievalChannel::ExactNormalForm => {
                let source = normal[normal_index].clone();
                normal_index += 1;
                turn = RetrievalChannel::Syntax;
                source
            }
        };
        if seen.insert(source.clone()) {
            output.push(source);
        }
    }
    output
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PackingResult {
    trace: PackingTrace,
    audit_targets: Vec<PayloadAuditTarget>,
}

fn pack_ranked<M: CanonicalSourceMaterializer>(
    ranked: &[Sha256Digest],
    materializer: &mut M,
    budget: &RetrievalBudget,
    context_packer_sha256: &Sha256Digest,
) -> Result<PackingResult, RetrievalError> {
    let mut output = Vec::new();
    let mut audit_targets = Vec::new();
    let mut bytes_used = 0usize;
    let mut first_nonfitting = None;
    let mut stop_reason = if ranked.is_empty() {
        StopReason::Empty
    } else {
        StopReason::RankedCandidatesExhausted
    };

    for (index, source) in ranked.iter().enumerate() {
        let rank = index + 1;
        if output.len() >= budget.max_output_items {
            stop_reason = StopReason::ItemLimitReached;
            break;
        }

        let payload = materializer.materialize(source)?;
        if payload.is_empty() {
            return Err(RetrievalError::Materialization(
                "canonical payload must be non-empty".into(),
            ));
        }
        std::str::from_utf8(&payload).map_err(|_| {
            RetrievalError::Materialization("canonical payload must be valid UTF-8".into())
        })?;
        let payload_bytes = payload.len();
        let exceeds_item = payload_bytes > budget.max_output_item_bytes;
        let exceeds_total = bytes_used
            .checked_add(payload_bytes)
            .map(|n| n > budget.max_output_bytes)
            .unwrap_or(true);

        if exceeds_item || exceeds_total {
            stop_reason = StopReason::FirstNonFittingItem;
            let item = PackedItem {
                rank,
                source_object_sha256: source.clone(),
                canonical_payload_bytes: payload_bytes,
            };
            first_nonfitting = Some(item);
            audit_targets.push(PayloadAuditTarget {
                role: PayloadRole::FirstNonFitting,
                rank,
                source_object_sha256: source.clone(),
                canonical_payload_utf8: payload,
            });
            break;
        }

        bytes_used += payload_bytes;
        output.push(PackedItem {
            rank,
            source_object_sha256: source.clone(),
            canonical_payload_bytes: payload_bytes,
        });
        audit_targets.push(PayloadAuditTarget {
            role: PayloadRole::Delivered,
            rank,
            source_object_sha256: source.clone(),
            canonical_payload_utf8: payload,
        });
    }

    Ok(PackingResult {
        trace: PackingTrace {
            context_packer_sha256: context_packer_sha256.clone(),
            input_ranked_source_object_digests: ranked.to_vec(),
            output_items_used: output.len(),
            output_bytes_used: bytes_used,
            output,
            stop_reason,
            first_nonfitting,
        },
        audit_targets,
    })
}

#[derive(Debug, Default)]
pub struct InMemoryEvidenceSink {
    staged_trace: Option<RetrievalTraceReceipt>,
    staged_payload_audit: Option<PayloadAuditBatch>,
    committed: Vec<EvidenceBundle>,
    pub fail_trace_stage: bool,
    pub fail_payload_stage: bool,
    pub fail_commit: bool,
}

impl InMemoryEvidenceSink {
    pub fn committed(&self) -> &[EvidenceBundle] {
        &self.committed
    }

    pub fn has_staged_state(&self) -> bool {
        self.staged_trace.is_some() || self.staged_payload_audit.is_some()
    }
}

impl RetrievalTraceSink for InMemoryEvidenceSink {
    fn stage_trace(&mut self, trace: &RetrievalTraceReceipt) -> Result<(), SinkError> {
        if self.fail_trace_stage {
            return Err(SinkError("injected trace-stage failure".into()));
        }
        if self.staged_trace.is_some() {
            return Err(SinkError("trace already staged".into()));
        }
        self.staged_trace = Some(trace.clone());
        Ok(())
    }
}

impl PayloadAuditSink for InMemoryEvidenceSink {
    fn stage_payload_audit(&mut self, audit: &PayloadAuditBatch) -> Result<(), SinkError> {
        if self.fail_payload_stage {
            return Err(SinkError("injected payload-stage failure".into()));
        }
        if self.staged_payload_audit.is_some() {
            return Err(SinkError("payload audit already staged".into()));
        }
        self.staged_payload_audit = Some(audit.clone());
        Ok(())
    }
}

impl AtomicEvidenceSink for InMemoryEvidenceSink {
    fn commit(&mut self) -> Result<(), SinkError> {
        if self.fail_commit {
            return Err(SinkError("injected commit failure".into()));
        }
        let trace = self
            .staged_trace
            .take()
            .ok_or_else(|| SinkError("commit without staged trace".into()))?;
        let payload_audit = self
            .staged_payload_audit
            .take()
            .ok_or_else(|| SinkError("commit without staged payload audit".into()))?;
        self.committed.push(EvidenceBundle {
            trace,
            payload_audit,
        });
        Ok(())
    }

    fn abort(&mut self) {
        self.staged_trace = None;
        self.staged_payload_audit = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(n: u8) -> Sha256Digest {
        Sha256Digest::parse(format!("sha256:{:064x}", n)).unwrap()
    }

    fn request() -> QualifiedRetrievalRequest {
        QualifiedRetrievalRequest {
            trace_id: "fixture-trace".into(),
            audit_id: "fixture-audit".into(),
            experiment_id: "fixture-experiment".into(),
            arm_id: "S".into(),
            experiment_seed: 7,
            query_id: "q-1".into(),
            query_source_object_sha256: digest(250),
            graph: GraphIdentity {
                bundle_sha256: digest(1),
                graph_report_sha256: digest(2),
                experiment_sha256: digest(3),
                retrieval_binding_sha256: digest(4),
                candidate_set_sha256: digest(5),
                candidate_count: 32,
                context_packer_sha256: digest(6),
                source_object_contract_sha256: digest(7),
                source_fetch_policy_sha256: digest(8),
                payload_serialization_sha256: digest(9),
            },
            budget: RetrievalBudget {
                max_output_items: 4,
                max_output_bytes: 8,
                max_output_item_bytes: 6,
                max_retrieval_queries: 2,
                max_normalized_compute_microunits: 1_000_000,
                max_wall_time_ms: 1_000,
            },
        }
    }

    fn control(index: Sha256Digest) -> ControlBinding {
        ControlBinding {
            index_manifest_sha256: index,
            control_transform: ControlTransform::None,
            control_seed: None,
            control_artifact_sha256: None,
        }
    }

    #[derive(Clone)]
    struct FakeBackend {
        execution: BackendExecution,
        calls: usize,
    }

    impl QualifiedRetrievalBackend for FakeBackend {
        fn execute(
            &mut self,
            _request: &QualifiedRetrievalRequest,
        ) -> Result<BackendExecution, RetrievalError> {
            self.calls += 1;
            Ok(self.execution.clone())
        }
    }

    #[derive(Default)]
    struct FakeMaterializer {
        payloads: BTreeMap<Sha256Digest, Vec<u8>>,
        calls: Vec<Sha256Digest>,
    }

    impl FakeMaterializer {
        fn with(mut self, source: Sha256Digest, payload: &str) -> Self {
            self.payloads.insert(source, payload.as_bytes().to_vec());
            self
        }
    }

    impl CanonicalSourceMaterializer for FakeMaterializer {
        fn materialize(
            &mut self,
            source_object_sha256: &Sha256Digest,
        ) -> Result<Vec<u8>, RetrievalError> {
            self.calls.push(source_object_sha256.clone());
            self.payloads.get(source_object_sha256).cloned().ok_or_else(|| {
                RetrievalError::Materialization(format!(
                    "fixture has no canonical payload for {source_object_sha256}"
                ))
            })
        }
    }

    #[test]
    fn single_index_uses_one_decision_point_and_shared_materializer_for_packing_and_evidence() {
        let index = digest(10);
        let mut backend = FakeBackend {
            execution: BackendExecution {
                retrieval: BackendRetrieval::SingleIndex(SingleIndexExecution {
                    index_manifest_sha256: index.clone(),
                    index_artifact_sha256: digest(11),
                    requested_k: 3,
                    ranked_source_object_digests: vec![digest(20), digest(21), digest(22)],
                    control: control(index),
                }),
                retrieval_queries_used: 1,
                wall_time_ms_used: 10,
                normalized_compute_microunits_used: 50,
            },
            calls: 0,
        };
        let mut materializer = FakeMaterializer::default()
            .with(digest(20), "abc")
            .with(digest(21), "defg")
            .with(digest(22), "12345");
        let mut sink = InMemoryEvidenceSink::default();
        let outcome = RetrievalExecutor::execute(
            &mut backend,
            &mut materializer,
            &mut sink,
            request(),
        )
        .unwrap();

        assert_eq!(backend.calls, 1);
        assert_eq!(materializer.calls, vec![digest(20), digest(21), digest(22)]);
        assert!(outcome.evidence_committed);
        assert_eq!(sink.committed().len(), 1);
        assert_eq!(outcome.trace.packing.output_items_used, 2);
        assert_eq!(outcome.trace.packing.output_bytes_used, 7);
        assert_eq!(outcome.trace.packing.stop_reason, StopReason::FirstNonFittingItem);
        assert_eq!(
            outcome
                .trace
                .packing
                .first_nonfitting
                .as_ref()
                .unwrap()
                .source_object_sha256,
            digest(22)
        );
        assert_eq!(outcome.payload_audit.entries.len(), 3);
        assert_eq!(outcome.payload_audit.entries[0].role, PayloadRole::Delivered);
        assert_eq!(outcome.payload_audit.entries[2].role, PayloadRole::FirstNonFitting);
        assert!(!sink.has_staged_state());
    }

    #[test]
    fn query_self_leak_is_rejected_before_materialization_or_evidence_commit() {
        let mut req = request();
        req.query_source_object_sha256 = digest(20);
        let index = digest(10);
        let mut backend = FakeBackend {
            execution: BackendExecution {
                retrieval: BackendRetrieval::SingleIndex(SingleIndexExecution {
                    index_manifest_sha256: index.clone(),
                    index_artifact_sha256: digest(11),
                    requested_k: 1,
                    ranked_source_object_digests: vec![digest(20)],
                    control: control(index),
                }),
                retrieval_queries_used: 1,
                wall_time_ms_used: 1,
                normalized_compute_microunits_used: 1,
            },
            calls: 0,
        };
        let mut materializer = FakeMaterializer::default().with(digest(20), "abc");
        let mut sink = InMemoryEvidenceSink::default();
        assert!(RetrievalExecutor::execute(&mut backend, &mut materializer, &mut sink, req).is_err());
        assert!(materializer.calls.is_empty());
        assert!(sink.committed().is_empty());
        assert!(!sink.has_staged_state());
    }

    #[test]
    fn evidence_failure_aborts_instead_of_succeeding_without_provenance() {
        let index = digest(10);
        let mut backend = FakeBackend {
            execution: BackendExecution {
                retrieval: BackendRetrieval::SingleIndex(SingleIndexExecution {
                    index_manifest_sha256: index.clone(),
                    index_artifact_sha256: digest(11),
                    requested_k: 1,
                    ranked_source_object_digests: vec![digest(20)],
                    control: control(index),
                }),
                retrieval_queries_used: 1,
                wall_time_ms_used: 1,
                normalized_compute_microunits_used: 1,
            },
            calls: 0,
        };
        let mut materializer = FakeMaterializer::default().with(digest(20), "abc");
        let mut sink = InMemoryEvidenceSink {
            fail_payload_stage: true,
            ..Default::default()
        };
        let error = RetrievalExecutor::execute(
            &mut backend,
            &mut materializer,
            &mut sink,
            request(),
        )
        .unwrap_err();
        assert!(matches!(error, RetrievalError::Evidence(_)));
        assert!(sink.committed().is_empty());
        assert!(!sink.has_staged_state());
    }

    #[test]
    fn deterministic_interleave_consumes_duplicate_turn_without_backfill() {
        let syntax_index = digest(30);
        let normal_index = digest(31);
        let mut backend = FakeBackend {
            execution: BackendExecution {
                retrieval: BackendRetrieval::Fusion(FusionExecution {
                    fusion_policy_sha256: digest(32),
                    method: FusionMethod::DeterministicInterleave {
                        start_channel: RetrievalChannel::Syntax,
                    },
                    syntax: ChannelExecution {
                        channel: RetrievalChannel::Syntax,
                        index_manifest_sha256: syntax_index.clone(),
                        index_artifact_sha256: digest(33),
                        requested_k: 2,
                        input_bytes_used: 2,
                        ranked_source_object_digests: vec![digest(40), digest(41)],
                        control: control(syntax_index),
                    },
                    normal_form: ChannelExecution {
                        channel: RetrievalChannel::ExactNormalForm,
                        index_manifest_sha256: normal_index.clone(),
                        index_artifact_sha256: digest(34),
                        requested_k: 2,
                        input_bytes_used: 2,
                        ranked_source_object_digests: vec![digest(41), digest(42)],
                        control: control(normal_index),
                    },
                }),
                retrieval_queries_used: 2,
                wall_time_ms_used: 2,
                normalized_compute_microunits_used: 2,
            },
            calls: 0,
        };
        let mut req = request();
        req.budget.max_output_bytes = 16;
        req.budget.max_output_item_bytes = 8;
        let mut materializer = FakeMaterializer::default()
            .with(digest(40), "a")
            .with(digest(41), "b")
            .with(digest(42), "c");
        let mut sink = InMemoryEvidenceSink::default();
        let outcome = RetrievalExecutor::execute(
            &mut backend,
            &mut materializer,
            &mut sink,
            req,
        )
        .unwrap();
        let TraceRetrieval::Fusion {
            fused_ranked_source_object_digests,
            ..
        } = &outcome.trace.retrieval
        else {
            panic!("expected fusion trace");
        };
        assert_eq!(
            fused_ranked_source_object_digests,
            &vec![digest(40), digest(41), digest(42)]
        );
    }

    #[test]
    fn exact_rrf_favors_candidate_present_in_both_channels() {
        let syntax_index = digest(50);
        let normal_index = digest(51);
        let mut backend = FakeBackend {
            execution: BackendExecution {
                retrieval: BackendRetrieval::Fusion(FusionExecution {
                    fusion_policy_sha256: digest(52),
                    method: FusionMethod::ReciprocalRankFusion { rrf_k: 60 },
                    syntax: ChannelExecution {
                        channel: RetrievalChannel::Syntax,
                        index_manifest_sha256: syntax_index.clone(),
                        index_artifact_sha256: digest(53),
                        requested_k: 2,
                        input_bytes_used: 2,
                        ranked_source_object_digests: vec![digest(60), digest(61)],
                        control: control(syntax_index),
                    },
                    normal_form: ChannelExecution {
                        channel: RetrievalChannel::ExactNormalForm,
                        index_manifest_sha256: normal_index.clone(),
                        index_artifact_sha256: digest(54),
                        requested_k: 2,
                        input_bytes_used: 2,
                        ranked_source_object_digests: vec![digest(61), digest(62)],
                        control: control(normal_index),
                    },
                }),
                retrieval_queries_used: 2,
                wall_time_ms_used: 2,
                normalized_compute_microunits_used: 2,
            },
            calls: 0,
        };
        let mut req = request();
        req.budget.max_output_bytes = 16;
        req.budget.max_output_item_bytes = 8;
        let mut materializer = FakeMaterializer::default()
            .with(digest(60), "a")
            .with(digest(61), "b")
            .with(digest(62), "c");
        let mut sink = InMemoryEvidenceSink::default();
        let outcome = RetrievalExecutor::execute(
            &mut backend,
            &mut materializer,
            &mut sink,
            req,
        )
        .unwrap();
        let TraceRetrieval::Fusion {
            fused_ranked_source_object_digests,
            ..
        } = &outcome.trace.retrieval
        else {
            panic!("expected fusion trace");
        };
        assert_eq!(fused_ranked_source_object_digests[0], digest(61));
    }

    #[test]
    fn wrong_query_count_is_rejected_as_nonqualifying() {
        let index = digest(90);
        let mut backend = FakeBackend {
            execution: BackendExecution {
                retrieval: BackendRetrieval::SingleIndex(SingleIndexExecution {
                    index_manifest_sha256: index.clone(),
                    index_artifact_sha256: digest(91),
                    requested_k: 1,
                    ranked_source_object_digests: vec![digest(92)],
                    control: control(index),
                }),
                retrieval_queries_used: 2,
                wall_time_ms_used: 1,
                normalized_compute_microunits_used: 1,
            },
            calls: 0,
        };
        let mut materializer = FakeMaterializer::default().with(digest(92), "x");
        let mut sink = InMemoryEvidenceSink::default();
        let error = RetrievalExecutor::execute(
            &mut backend,
            &mut materializer,
            &mut sink,
            request(),
        )
        .unwrap_err();
        assert!(matches!(error, RetrievalError::Budget(_)));
        assert!(materializer.calls.is_empty());
        assert!(sink.committed().is_empty());
    }

    #[test]
    fn representation_backend_cannot_supply_or_mutate_downstream_payload_bytes() {
        let index = digest(100);
        let mut backend = FakeBackend {
            execution: BackendExecution {
                retrieval: BackendRetrieval::SingleIndex(SingleIndexExecution {
                    index_manifest_sha256: index.clone(),
                    index_artifact_sha256: digest(101),
                    requested_k: 1,
                    ranked_source_object_digests: vec![digest(102)],
                    control: control(index),
                }),
                retrieval_queries_used: 1,
                wall_time_ms_used: 1,
                normalized_compute_microunits_used: 1,
            },
            calls: 0,
        };
        let mut materializer = FakeMaterializer::default().with(digest(102), "canonical-source");
        let mut req = request();
        req.budget.max_output_bytes = 64;
        req.budget.max_output_item_bytes = 64;
        let mut sink = InMemoryEvidenceSink::default();
        let outcome = RetrievalExecutor::execute(
            &mut backend,
            &mut materializer,
            &mut sink,
            req,
        )
        .unwrap();
        assert_eq!(
            outcome.payload_audit.entries[0].canonical_payload_utf8,
            b"canonical-source"
        );
    }
}
