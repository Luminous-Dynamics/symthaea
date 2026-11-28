// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-aware adapter contract for the official LongMemEval-V2 Memory API.
//!
//! The official backend interface exposes only question text and an optional
//! image to query(), plus a run-local opaque query invocation ID. The harness
//! calls post_query_hook() before truncating context. This crate therefore
//! requires a second, outer-runner receipt after truncation and after the exact
//! reader request/response are available.
//!
//! It never accepts question IDs, question types, gold answers or grader labels
//! as memory-query inputs. All content is represented by digests in receipts.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const ADAPTER_SCHEMA_V1: &str = "mace_longmemeval_v2_adapter_receipt_v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LongMemEvalTierV1 { Small, Medium }

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LongMemEvalDomainV1 { Web, Enterprise }

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextItemKindV1 { Text, Image }

/// Frozen identity for one official LongMemEval-V2 operating point.
/// Every insert, query, hook and final-reader receipt binds to this digest.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LongMemEvalRunBindingV1 {
    pub schema_version: String,
    pub run_id: String,
    pub benchmark_id: String,
    pub benchmark_release: String,
    pub harness_commit_sha: String,
    pub dataset_snapshot_sha256: String,
    pub dataset_split_sha256: String,
    pub tier: LongMemEvalTierV1,
    pub domain: LongMemEvalDomainV1,
    pub memory_context_budget_tokens: u32,
    pub reader_model_sha256: String,
    pub reader_generation_config_sha256: String,
    pub memory_backend_config_sha256: String,
    pub task_manifest_sha256: String,
    pub cluster_unit: String,
    pub binding_sha256: String,
}

fn run_binding_digest(binding: &LongMemEvalRunBindingV1) -> Result<String, AdapterRejectV1> {
    let mut unsigned = binding.clone();
    unsigned.binding_sha256.clear();
    digest_serialized(&unsigned)
}

impl LongMemEvalRunBindingV1 {
    pub fn seal(mut self) -> Result<Self, AdapterRejectV1> {
        if !nonempty(&self.run_id) || self.benchmark_id != "longmemeval_v2"
            || !nonempty(&self.benchmark_release) || !valid_git_sha(&self.harness_commit_sha)
            || !nonempty(&self.cluster_unit) || self.memory_context_budget_tokens == 0 {
            return Err(AdapterRejectV1::InvalidIdentity);
        }
        for value in [
            self.dataset_snapshot_sha256.as_str(), self.dataset_split_sha256.as_str(),
            self.reader_model_sha256.as_str(), self.reader_generation_config_sha256.as_str(),
            self.memory_backend_config_sha256.as_str(), self.task_manifest_sha256.as_str(),
        ] {
            if !valid_digest(value) { return Err(AdapterRejectV1::InvalidDigest); }
        }
        self.schema_version = ADAPTER_SCHEMA_V1.into();
        self.binding_sha256.clear();
        self.binding_sha256 = run_binding_digest(&self)?;
        Ok(self)
    }

    pub fn verify_integrity(&self) -> bool {
        self.schema_version == ADAPTER_SCHEMA_V1
            && run_binding_digest(self).is_ok_and(|value| value == self.binding_sha256)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrajectoryInsertReceiptV1 {
    pub schema_version: String,
    pub run_id: String,
    pub run_binding_sha256: String,
    pub tier: LongMemEvalTierV1,
    pub domain: LongMemEvalDomainV1,
    /// Monotonically increasing order supplied by the official harness.
    pub insert_order: u64,
    pub trajectory_sha256: String,
    pub payload_sha256: String,
    pub source_snapshot_sha256: String,
    pub inserted_at_monotonic_ns: u64,
    pub receipt_sha256: String,
}

/// Privacy-minimized input contract for the official Memory.query API.
/// Only query text and an optional question image are represented. No dataset
/// question ID, question type, gold answer, evaluator function or grader label
/// is provided to the memory backend.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryQueryInputV1 {
    pub schema_version: String,
    pub run_id: String,
    pub run_binding_sha256: String,
    /// The official harness uses uuid.uuid4().hex; this is not a dataset question ID.
    pub query_invocation_id: String,
    pub question_text_sha256: String,
    pub question_image_sha256: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreQueryContextItemV1 {
    pub ordinal: u32,
    pub kind: ContextItemKindV1,
    /// SHA-256 of UTF-8 text bytes or image file bytes, not an image path.
    pub context_item_sha256: String,
    pub evidence_use_receipt_sha256: String,
    /// Canonical #13973 DecisionEvidenceUseLinkV1 digest for this selected item.
    pub decision_evidence_use_link_sha256: String,
    pub source_lineage_receipt_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreQuerySelectionReceiptV1 {
    pub schema_version: String,
    pub run_id: String,
    pub run_binding_sha256: String,
    pub query_invocation_id: String,
    pub query_input_sha256: String,
    pub selection_policy_sha256: String,
    /// Exact ordered evidence-item list returned by Memory.query before token truncation.
    pub selected_items: Vec<PreQueryContextItemV1>,
    pub pre_truncation_items_sha256: String,
    pub receipt_sha256: String,
}

/// Safe metadata returned by the official backend's post_query_hook().
/// LongMemEval-V2 persists this as memory_post_query_metadata in per_question.jsonl.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreQueryHookMetadataV1 {
    pub schema_version: String,
    pub run_binding_sha256: String,
    pub query_invocation_id: String,
    pub query_input_sha256: String,
    pub pre_query_selection_receipt_sha256: String,
    pub selected_item_count: u32,
    pub metadata_sha256: String,
}

impl PreQuerySelectionReceiptV1 {
    /// Produce the opaque metadata map that the official post_query_hook can return.
    /// It contains no dataset question ID, answer, type or grader information.
    pub fn to_post_query_metadata(&self) -> Result<PreQueryHookMetadataV1, AdapterRejectV1> {
        if !self.verify_integrity() {
            return Err(AdapterRejectV1::PreQueryReceiptMismatch);
        }
        let mut metadata = PreQueryHookMetadataV1 {
            schema_version: ADAPTER_SCHEMA_V1.into(),
            run_binding_sha256: self.run_binding_sha256.clone(),
            query_invocation_id: self.query_invocation_id.clone(),
            query_input_sha256: self.query_input_sha256.clone(),
            pre_query_selection_receipt_sha256: self.receipt_sha256.clone(),
            selected_item_count: self.selected_items.len() as u32,
            metadata_sha256: String::new(),
        };
        let mut unsigned = metadata.clone();
        unsigned.metadata_sha256.clear();
        metadata.metadata_sha256 = digest_serialized(&unsigned)?;
        Ok(metadata)
    }
}

impl PreQueryHookMetadataV1 {
    pub fn verify_integrity(&self) -> bool {
        let mut unsigned = self.clone();
        unsigned.metadata_sha256.clear();
        digest_serialized(&unsigned)
            .is_ok_and(|digest| digest == self.metadata_sha256)
            && valid_digest(&self.run_binding_sha256)
            && valid_query_invocation_id(&self.query_invocation_id)
            && valid_digest(&self.query_input_sha256)
            && valid_digest(&self.pre_query_selection_receipt_sha256)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FinalContextItemV1 {
    pub ordinal: u32,
    pub kind: ContextItemKindV1,
    /// Hash of the final text/image bytes supplied to the fixed reader.
    pub context_item_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExposedEvidenceItemV1 {
    pub ordinal: u32,
    pub kind: ContextItemKindV1,
    pub context_item_sha256: String,
    pub evidence_use_receipt_sha256: String,
    pub decision_evidence_use_link_sha256: String,
    pub source_lineage_receipt_sha256: String,
    /// Digest of the exact post-truncation reader request containing this item.
    pub reader_request_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReaderExposureReceiptV1 {
    pub schema_version: String,
    pub run_id: String,
    pub run_binding_sha256: String,
    pub query_invocation_id: String,
    pub pre_query_selection_receipt_sha256: String,
    pub final_context_items_sha256: String,
    pub reader_request_sha256: String,
    pub reader_response_sha256: String,
    pub reader_model_sha256: String,
    pub reader_generation_config_sha256: String,
    pub selected_item_count: u32,
    pub exposed_item_count: u32,
    pub context_budget_tokens: u32,
    pub original_context_tokens: u32,
    pub final_context_tokens: u32,
    pub context_was_truncated: bool,
    pub memory_query_latency_ns: u64,
    pub post_query_hook_latency_ns: u64,
    pub exposed_evidence: Vec<ExposedEvidenceItemV1>,
    pub receipt_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdapterRejectV1 {
    InvalidIdentity,
    InvalidDigest,
    InvalidInsertOrder,
    DuplicateTrajectory,
    RunOrDomainMismatch,
    InvalidQueryInvocationId,
    QueryInputMismatch,
    InvalidContextOrdinal,
    DuplicateContextItem,
    PreQueryReceiptMismatch,
    FinalContextNotPrefix,
    FinalContextDigestMismatch,
    InvalidTokenBudget,
    InvalidTruncationFlag,
    MissingReaderRequest,
    MissingReaderResponse,
    IntegrityMismatch,
    SerializationFailure,
}

fn digest_bytes(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn digest_serialized<T: Serialize + ?Sized>(value: &T) -> Result<String, AdapterRejectV1> {
    serde_json::to_vec(value)
        .map(|bytes| digest_bytes(&bytes))
        .map_err(|_| AdapterRejectV1::SerializationFailure)
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Validate a full 40-hex-character Git commit object ID.
fn valid_git_sha(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty() && value.len() <= 512
}

fn valid_query_invocation_id(value: &str) -> bool {
    value.len() == 32 && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn trajectory_receipt_digest(value: &TrajectoryInsertReceiptV1) -> Result<String, AdapterRejectV1> {
    let mut unsigned = value.clone();
    unsigned.receipt_sha256.clear();
    digest_serialized(&unsigned)
}

impl TrajectoryInsertReceiptV1 {
    pub fn verify_integrity(&self) -> bool {
        trajectory_receipt_digest(self).is_ok_and(|value| value == self.receipt_sha256)
    }
}

pub fn seal_trajectory_insert_v1(
    mut value: TrajectoryInsertReceiptV1,
) -> Result<TrajectoryInsertReceiptV1, AdapterRejectV1> {
    value.schema_version = ADAPTER_SCHEMA_V1.into();
    for hash in [
        value.trajectory_sha256.as_str(),
        value.payload_sha256.as_str(),
        value.source_snapshot_sha256.as_str(),
        value.run_binding_sha256.as_str(),
    ] {
        if !valid_digest(hash) { return Err(AdapterRejectV1::InvalidDigest); }
    }
    if !nonempty(&value.run_id) { return Err(AdapterRejectV1::InvalidIdentity); }
    value.receipt_sha256.clear();
    value.receipt_sha256 = trajectory_receipt_digest(&value)?;
    Ok(value)
}

/// Check that insert receipts follow the harness' supplied order and describe
/// one run, tier, domain and source snapshot. It does not reorder trajectories.
pub fn validate_trajectory_insert_sequence_v1(
    receipts: &[TrajectoryInsertReceiptV1],
) -> Result<(), AdapterRejectV1> {
    if receipts.is_empty() { return Ok(()); }
    let first = &receipts[0];
    let mut previous_order = None;
    for receipt in receipts {
        if !receipt.verify_integrity() || receipt.schema_version != ADAPTER_SCHEMA_V1 {
            return Err(AdapterRejectV1::IntegrityMismatch);
        }
        if receipt.run_id != first.run_id
            || receipt.run_binding_sha256 != first.run_binding_sha256
            || receipt.tier != first.tier
            || receipt.domain != first.domain
            || receipt.source_snapshot_sha256 != first.source_snapshot_sha256 {
            return Err(AdapterRejectV1::RunOrDomainMismatch);
        }
        if previous_order.is_some_and(|order| receipt.insert_order <= order) {
            return Err(AdapterRejectV1::InvalidInsertOrder);
        }
        previous_order = Some(receipt.insert_order);
    }
    Ok(())
}

impl MemoryQueryInputV1 {
    pub fn verify_contract(&self) -> Result<(), AdapterRejectV1> {
        if self.schema_version != ADAPTER_SCHEMA_V1 || !nonempty(&self.run_id) {
            return Err(AdapterRejectV1::InvalidIdentity);
        }
        if !valid_query_invocation_id(&self.query_invocation_id) {
            return Err(AdapterRejectV1::InvalidQueryInvocationId);
        }
        if !valid_digest(&self.run_binding_sha256)
            || !valid_digest(&self.question_text_sha256)
            || self.question_image_sha256.as_deref().is_some_and(|value| !valid_digest(value)) {
            return Err(AdapterRejectV1::InvalidDigest);
        }
        Ok(())
    }
}

fn pre_query_digest(value: &PreQuerySelectionReceiptV1) -> Result<String, AdapterRejectV1> {
    let mut unsigned = value.clone();
    unsigned.receipt_sha256.clear();
    digest_serialized(&unsigned)
}

impl PreQuerySelectionReceiptV1 {
    pub fn verify_integrity(&self) -> bool {
        pre_query_digest(self).is_ok_and(|value| value == self.receipt_sha256)
    }
}

pub fn seal_pre_query_selection_v1(
    query: &MemoryQueryInputV1,
    selection_policy_sha256: String,
    selected_items: Vec<PreQueryContextItemV1>,
) -> Result<PreQuerySelectionReceiptV1, AdapterRejectV1> {
    query.verify_contract()?;
    if !valid_digest(&selection_policy_sha256) { return Err(AdapterRejectV1::InvalidDigest); }
    let mut seen = BTreeSet::new();
    for (index, item) in selected_items.iter().enumerate() {
        if item.ordinal != index as u32 { return Err(AdapterRejectV1::InvalidContextOrdinal); }
        if !valid_digest(&item.context_item_sha256)
            || !valid_digest(&item.evidence_use_receipt_sha256)
            || !valid_digest(&item.decision_evidence_use_link_sha256)
            || !valid_digest(&item.source_lineage_receipt_sha256) {
            return Err(AdapterRejectV1::InvalidDigest);
        }
        if !seen.insert((
            item.kind,
            item.context_item_sha256.as_str(),
            item.evidence_use_receipt_sha256.as_str(),
            item.decision_evidence_use_link_sha256.as_str(),
            item.source_lineage_receipt_sha256.as_str(),
        )) {
            return Err(AdapterRejectV1::DuplicateContextItem);
        }
    }
    let mut receipt = PreQuerySelectionReceiptV1 {
        schema_version: ADAPTER_SCHEMA_V1.into(),
        run_id: query.run_id.clone(),
        run_binding_sha256: query.run_binding_sha256.clone(),
        query_invocation_id: query.query_invocation_id.clone(),
        query_input_sha256: digest_serialized(query)?,
        selection_policy_sha256,
        pre_truncation_items_sha256: digest_serialized(&selected_items)?,
        selected_items,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = pre_query_digest(&receipt)?;
    Ok(receipt)
}

fn reader_exposure_digest(value: &ReaderExposureReceiptV1) -> Result<String, AdapterRejectV1> {
    let mut unsigned = value.clone();
    unsigned.receipt_sha256.clear();
    digest_serialized(&unsigned)
}

impl ReaderExposureReceiptV1 {
    pub fn verify_integrity(&self) -> bool {
        reader_exposure_digest(self).is_ok_and(|value| value == self.receipt_sha256)
    }
}

/// Finalize exposure only after the official harness has truncated the
/// pre-query context, built the exact reader messages, and received the reader
/// response. Since the official harness truncates by ordered prefix, every
/// exposed item must equal the corresponding pre-query item at the same ordinal.
#[allow(clippy::too_many_arguments)]
pub fn finalize_reader_exposure_v1(
    query: &MemoryQueryInputV1,
    pre_query: &PreQuerySelectionReceiptV1,
    final_items: &[FinalContextItemV1],
    context_budget_tokens: u32,
    original_context_tokens: u32,
    final_context_tokens: u32,
    context_was_truncated: bool,
    reader_request_sha256: String,
    reader_response_sha256: String,
    reader_model_sha256: String,
    reader_generation_config_sha256: String,
    memory_query_latency_ns: u64,
    post_query_hook_latency_ns: u64,
) -> Result<ReaderExposureReceiptV1, AdapterRejectV1> {
    query.verify_contract()?;
    if !pre_query.verify_integrity() {
        return Err(AdapterRejectV1::PreQueryReceiptMismatch);
    }
    if pre_query.run_id != query.run_id
        || pre_query.run_binding_sha256 != query.run_binding_sha256
        || pre_query.query_invocation_id != query.query_invocation_id
        || pre_query.query_input_sha256 != digest_serialized(query)? {
        return Err(AdapterRejectV1::QueryInputMismatch);
    }
    if !valid_digest(&reader_request_sha256) {
        return Err(AdapterRejectV1::MissingReaderRequest);
    }
    if !valid_digest(&reader_response_sha256) {
        return Err(AdapterRejectV1::MissingReaderResponse);
    }
    if !valid_digest(&reader_model_sha256) || !valid_digest(&reader_generation_config_sha256) {
        return Err(AdapterRejectV1::InvalidDigest);
    }
    if context_budget_tokens == 0 || final_context_tokens > context_budget_tokens
        || final_context_tokens > original_context_tokens {
        return Err(AdapterRejectV1::InvalidTokenBudget);
    }
    let actual_context_truncation = original_context_tokens > final_context_tokens
        || final_items.len() < pre_query.selected_items.len();
    if context_was_truncated != actual_context_truncation {
        return Err(AdapterRejectV1::InvalidTruncationFlag);
    }
    if final_items.len() > pre_query.selected_items.len() {
        return Err(AdapterRejectV1::FinalContextNotPrefix);
    }

    let mut seen_final = BTreeSet::new();
    let mut exposed_evidence = Vec::with_capacity(final_items.len());
    for (index, item) in final_items.iter().enumerate() {
        let selected = pre_query.selected_items.get(index).ok_or(AdapterRejectV1::FinalContextNotPrefix)?;
        if item.ordinal != index as u32 || selected.ordinal != index as u32 {
            return Err(AdapterRejectV1::InvalidContextOrdinal);
        }
        if item.kind != selected.kind || item.context_item_sha256 != selected.context_item_sha256 {
            return Err(AdapterRejectV1::FinalContextNotPrefix);
        }
        if !valid_digest(&item.context_item_sha256) {
            return Err(AdapterRejectV1::InvalidDigest);
        }
        if !seen_final.insert((
            item.kind,
            item.context_item_sha256.as_str(),
            selected.evidence_use_receipt_sha256.as_str(),
            selected.source_lineage_receipt_sha256.as_str(),
        )) {
            return Err(AdapterRejectV1::DuplicateContextItem);
        }
        exposed_evidence.push(ExposedEvidenceItemV1 {
            ordinal: item.ordinal,
            kind: item.kind,
            context_item_sha256: item.context_item_sha256.clone(),
            evidence_use_receipt_sha256: selected.evidence_use_receipt_sha256.clone(),
            decision_evidence_use_link_sha256: selected.decision_evidence_use_link_sha256.clone(),
            source_lineage_receipt_sha256: selected.source_lineage_receipt_sha256.clone(),
            reader_request_sha256: reader_request_sha256.clone(),
        });
    }

    let mut receipt = ReaderExposureReceiptV1 {
        schema_version: ADAPTER_SCHEMA_V1.into(),
        run_id: query.run_id.clone(),
        run_binding_sha256: query.run_binding_sha256.clone(),
        query_invocation_id: query.query_invocation_id.clone(),
        pre_query_selection_receipt_sha256: pre_query.receipt_sha256.clone(),
        final_context_items_sha256: digest_serialized(final_items)?,
        reader_request_sha256,
        reader_response_sha256,
        reader_model_sha256,
        reader_generation_config_sha256,
        selected_item_count: pre_query.selected_items.len() as u32,
        exposed_item_count: final_items.len() as u32,
        context_budget_tokens,
        original_context_tokens,
        final_context_tokens,
        context_was_truncated,
        memory_query_latency_ns,
        post_query_hook_latency_ns,
        exposed_evidence,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = reader_exposure_digest(&receipt)?;
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: &str) -> String { digest_bytes(value.as_bytes()) }

    fn run_binding() -> LongMemEvalRunBindingV1 {
        LongMemEvalRunBindingV1 {
            schema_version: ADAPTER_SCHEMA_V1.into(),
            run_id: "lme-small-web".into(),
            benchmark_id: "longmemeval_v2".into(),
            benchmark_release: "v2".into(),
            harness_commit_sha: "0123456789abcdef0123456789abcdef01234567".into(),
            dataset_snapshot_sha256: digest("dataset-snapshot"),
            dataset_split_sha256: digest("dataset-split"),
            tier: LongMemEvalTierV1::Small,
            domain: LongMemEvalDomainV1::Web,
            memory_context_budget_tokens: 4096,
            reader_model_sha256: digest("reader-model"),
            reader_generation_config_sha256: digest("reader-generation"),
            memory_backend_config_sha256: digest("memory-backend"),
            task_manifest_sha256: digest("task-manifest"),
            cluster_unit: "conversation_family".into(),
            binding_sha256: String::new(),
        }.seal().unwrap()
    }

    fn query() -> MemoryQueryInputV1 {
        MemoryQueryInputV1 {
            schema_version: ADAPTER_SCHEMA_V1.into(),
            run_id: "lme-small-web".into(),
            run_binding_sha256: run_binding().binding_sha256,
            query_invocation_id: "0123456789abcdef0123456789abcdef".into(),
            question_text_sha256: digest("how to complete workflow"),
            question_image_sha256: Some(digest("question-image-bytes")),
        }
    }

    fn item(ordinal: u32, label: &str) -> PreQueryContextItemV1 {
        PreQueryContextItemV1 {
            ordinal,
            kind: ContextItemKindV1::Text,
            context_item_sha256: digest(&format!("context-{label}")),
            evidence_use_receipt_sha256: digest(&format!("use-{label}")),
            decision_evidence_use_link_sha256: digest(&format!("link-{label}")),
            source_lineage_receipt_sha256: digest(&format!("lineage-{label}")),
        }
    }

    fn selection() -> PreQuerySelectionReceiptV1 {
        seal_pre_query_selection_v1(&query(), digest("selection-policy"), vec![item(0,"a"),item(1,"b"),item(2,"c")]).unwrap()
    }

    fn finalize(items: Vec<FinalContextItemV1>, budget: u32, original: u32, final_tokens: u32, truncated: bool) -> Result<ReaderExposureReceiptV1, AdapterRejectV1> {
        finalize_reader_exposure_v1(
            &query(), &selection(),
            &items, budget, original, final_tokens, truncated,
            digest("exact-reader-request"), digest("reader-response"),
            digest("reader-model"), digest("reader-generation-config"),
            120_000_000, 1_000_000,
        )
    }

    #[test]
    fn only_the_final_context_prefix_is_marked_exposed() {
        let selected = selection();
        let value = finalize(
            vec![
                FinalContextItemV1 { ordinal: 0, kind: ContextItemKindV1::Text, context_item_sha256: selected.selected_items[0].context_item_sha256.clone() },
                FinalContextItemV1 { ordinal: 1, kind: ContextItemKindV1::Text, context_item_sha256: selected.selected_items[1].context_item_sha256.clone() },
            ],
            1000, 1400, 900, true,
        ).unwrap();
        assert_eq!(value.selected_item_count, 3);
        assert_eq!(value.exposed_item_count, 2);
        assert_eq!(value.exposed_evidence[0].evidence_use_receipt_sha256, selected.selected_items[0].evidence_use_receipt_sha256);
        assert_eq!(value.exposed_evidence[1].evidence_use_receipt_sha256, selected.selected_items[1].evidence_use_receipt_sha256);
        assert!(value.verify_integrity());
    }

    #[test]
    fn no_truncation_exposes_the_full_selected_sequence() {
        let selected = selection();
        let items = selected.selected_items.iter().map(|value| FinalContextItemV1 {
            ordinal: value.ordinal, kind: value.kind, context_item_sha256: value.context_item_sha256.clone()
        }).collect();
        let receipt = finalize(items, 2000, 1200, 1200, false).unwrap();
        assert_eq!(receipt.exposed_item_count, 3);
        assert!(!receipt.context_was_truncated);
    }

    #[test]
    fn reordered_or_substituted_context_is_rejected() {
        let selected = selection();
        let reordered = vec![
            FinalContextItemV1 { ordinal: 0, kind: ContextItemKindV1::Text, context_item_sha256: selected.selected_items[1].context_item_sha256.clone() }
        ];
        assert_eq!(finalize(reordered, 100, 80, 70, true), Err(AdapterRejectV1::FinalContextNotPrefix));
    }

    #[test]
    fn omitted_tail_items_are_not_claimed_as_exposed() {
        let selected = selection();
        let first = FinalContextItemV1 {
            ordinal: 0, kind: selected.selected_items[0].kind,
            context_item_sha256: selected.selected_items[0].context_item_sha256.clone()
        };
        let receipt = finalize(vec![first], 100, 500, 80, true).unwrap();
        assert_eq!(receipt.exposed_evidence.len(), 1);
        assert_eq!(receipt.exposed_evidence[0].ordinal, 0);
    }

    #[test]
    fn token_budget_and_truncation_flags_must_match_observed_counts() {
        let items = vec![];
        assert_eq!(finalize(items.clone(), 10, 100, 20, true), Err(AdapterRejectV1::InvalidTokenBudget));
        assert_eq!(finalize(items.clone(), 100, 100, 100, false), Err(AdapterRejectV1::InvalidTruncationFlag));
        assert!(finalize(items, 100, 100, 100, true).is_ok());
        let selected_empty = seal_pre_query_selection_v1(&query(), digest("policy"), vec![]).unwrap();
        let result = finalize_reader_exposure_v1(
            &query(), &selected_empty, &[], 100, 100, 100, false,
            digest("request"), digest("response"), digest("reader"), digest("generation"), 1, 1
        ).unwrap();
        assert_eq!(result.exposed_item_count, 0);
        assert!(result.verify_integrity());
    }

    #[test]
    fn query_contract_contains_no_question_id_or_evaluator_labels() {
        let encoded = serde_json::to_value(query()).unwrap();
        let object = encoded.as_object().unwrap();
        assert!(object.contains_key("query_invocation_id"));
        assert!(object.contains_key("question_text_sha256"));
        for forbidden in ["question_id","question_type","gold_answer","answer_gold","eval_function","grader_label"] {
            assert!(!object.contains_key(forbidden));
        }
    }

    #[test]
    fn query_invocation_id_must_be_opaque_uuid_hex() {
        let mut invalid = query();
        invalid.query_invocation_id = "secret-question-id".into();
        assert_eq!(invalid.verify_contract(), Err(AdapterRejectV1::InvalidQueryInvocationId));
    }

    #[test]
    fn prequery_metadata_is_bound_to_the_exact_query_invocation() {
        let selected = selection();
        let mut changed_query = query();
        changed_query.query_invocation_id = "fedcba9876543210fedcba9876543210".into();
        assert_eq!(
            finalize_reader_exposure_v1(
                &changed_query, &selected, &[], 10, 0, 0, false,
                digest("request"), digest("response"), digest("reader"), digest("generation"), 0, 0
            ),
            Err(AdapterRejectV1::QueryInputMismatch)
        );
    }

    #[test]
    fn insertion_receipts_preserve_order_and_allow_repeated_identical_payloads() {
        let receipt = |order, id: &str| seal_trajectory_insert_v1(TrajectoryInsertReceiptV1 {
            schema_version: ADAPTER_SCHEMA_V1.into(),
            run_id: "run-1".into(), run_binding_sha256: run_binding().binding_sha256,
            tier: LongMemEvalTierV1::Small, domain: LongMemEvalDomainV1::Web,
            insert_order: order, trajectory_sha256: digest(id), payload_sha256: digest(&format!("payload-{id}")),
            source_snapshot_sha256: digest("snapshot"), inserted_at_monotonic_ns: order * 10,
            receipt_sha256: String::new(),
        }).unwrap();
        let first = receipt(0,"a");
        let second = receipt(1,"b");
        assert!(validate_trajectory_insert_sequence_v1(&[first.clone(),second.clone()]).is_ok());
        assert_eq!(validate_trajectory_insert_sequence_v1(&[second.clone(),first.clone()]),Err(AdapterRejectV1::InvalidInsertOrder));
        let repeated_payload = receipt(2,"a");
        assert!(validate_trajectory_insert_sequence_v1(&[first,repeated_payload]).is_ok());
    }

    #[test]
    fn cross_domain_or_snapshot_insert_receipts_are_rejected() {
        let base = seal_trajectory_insert_v1(TrajectoryInsertReceiptV1 {
            schema_version: ADAPTER_SCHEMA_V1.into(), run_id: "run-1".into(),
            run_binding_sha256: run_binding().binding_sha256,
            tier: LongMemEvalTierV1::Small, domain: LongMemEvalDomainV1::Web,
            insert_order: 0, trajectory_sha256: digest("a"), payload_sha256: digest("payload-a"),
            source_snapshot_sha256: digest("snapshot-a"), inserted_at_monotonic_ns: 10,
            receipt_sha256: String::new(),
        }).unwrap();
        let mut second = base.clone();
        second.insert_order = 1;
        second.domain = LongMemEvalDomainV1::Enterprise;
        second.trajectory_sha256 = digest("b");
        second.payload_sha256 = digest("payload-b");
        second.receipt_sha256.clear();
        let second = seal_trajectory_insert_v1(second).unwrap();
        assert_eq!(validate_trajectory_insert_sequence_v1(&[base,second]),Err(AdapterRejectV1::RunOrDomainMismatch));
    }

    #[test]
    fn missing_reader_request_or_response_is_rejected() {
        let selected = selection();
        let items = vec![FinalContextItemV1 {
            ordinal: 0, kind: selected.selected_items[0].kind,
            context_item_sha256: selected.selected_items[0].context_item_sha256.clone()
        }];
        assert_eq!(
            finalize_reader_exposure_v1(&query(), &selected, &items, 100, 50, 40, true, String::new(), digest("response"), digest("reader"), digest("generation"), 1, 1),
            Err(AdapterRejectV1::MissingReaderRequest)
        );
        assert_eq!(
            finalize_reader_exposure_v1(&query(), &selected, &items, 100, 50, 40, true, digest("request"), String::new(), digest("reader"), digest("generation"), 1, 1),
            Err(AdapterRejectV1::MissingReaderResponse)
        );
    }

    #[test]
    fn final_receipt_digest_detects_mutation() {
        let selected = selection();
        let first = FinalContextItemV1 {
            ordinal: 0, kind: selected.selected_items[0].kind,
            context_item_sha256: selected.selected_items[0].context_item_sha256.clone()
        };
        let mut receipt = finalize(vec![first], 100, 80, 70, true).unwrap();
        receipt.reader_request_sha256 = digest("different-request");
        assert!(!receipt.verify_integrity());
    }

    #[test]
    fn identical_context_bytes_from_distinct_evidence_sources_remain_distinguishable() {
        let query = query();
        let same_content = digest("identical-evidence-text");
        let selected = seal_pre_query_selection_v1(
            &query,
            digest("selection-policy"),
            vec![
                PreQueryContextItemV1 {
                    ordinal: 0,
                    kind: ContextItemKindV1::Text,
                    context_item_sha256: same_content.clone(),
                    evidence_use_receipt_sha256: digest("use-source-a"),
                    decision_evidence_use_link_sha256: digest("link-source-a"),
                    source_lineage_receipt_sha256: digest("lineage-source-a"),
                },
                PreQueryContextItemV1 {
                    ordinal: 1,
                    kind: ContextItemKindV1::Text,
                    context_item_sha256: same_content.clone(),
                    evidence_use_receipt_sha256: digest("use-source-b"),
                    decision_evidence_use_link_sha256: digest("link-source-b"),
                    source_lineage_receipt_sha256: digest("lineage-source-b"),
                },
            ],
        ).unwrap();
        let final_items = vec![
            FinalContextItemV1 { ordinal: 0, kind: ContextItemKindV1::Text, context_item_sha256: same_content.clone() },
            FinalContextItemV1 { ordinal: 1, kind: ContextItemKindV1::Text, context_item_sha256: same_content },
        ];
        let receipt = finalize_reader_exposure_v1(
            &query, &selected, &final_items,
            1000, 200, 200, false,
            digest("reader-request"), digest("reader-response"),
            digest("reader-model"), digest("reader-generation"),
            10, 2,
        ).unwrap();
        assert_eq!(receipt.exposed_evidence.len(), 2);
        assert_ne!(
            receipt.exposed_evidence[0].source_lineage_receipt_sha256,
            receipt.exposed_evidence[1].source_lineage_receipt_sha256
        );
        assert_ne!(
            receipt.exposed_evidence[0].decision_evidence_use_link_sha256,
            receipt.exposed_evidence[1].decision_evidence_use_link_sha256
        );
    }


    #[test]
    fn post_query_hook_metadata_links_per_question_row_without_evaluator_labels() {
        let selection = selection();
        let metadata = selection.to_post_query_metadata().unwrap();
        assert!(metadata.verify_integrity());
        assert_eq!(metadata.query_invocation_id, query().query_invocation_id);
        assert_eq!(metadata.pre_query_selection_receipt_sha256, selection.receipt_sha256);
        let encoded = serde_json::to_value(&metadata).unwrap();
        let object = encoded.as_object().unwrap();
        for forbidden in ["question_id","question_type","answer_gold","eval_function","score"] {
            assert!(!object.contains_key(forbidden));
        }
    }


    #[test]
    fn run_binding_pins_harness_dataset_tier_domain_and_reader_configuration() {
        let binding = run_binding();
        assert!(binding.verify_integrity());
        let mut tampered = binding.clone();
        tampered.harness_commit_sha = "fedcba9876543210fedcba9876543210fedcba98".into();
        assert!(!tampered.verify_integrity());
        let encoded = serde_json::to_value(&binding).unwrap();
        assert_eq!(encoded["benchmark_id"], "longmemeval_v2");
        assert_eq!(encoded["tier"], "small");
        assert_eq!(encoded["domain"], "web");
    }

    #[test]
    fn query_receipts_cannot_be_joined_across_run_bindings() {
        let mut invalid = selection();
        invalid.run_binding_sha256 = digest("other-run-binding");
        let mut unsigned = invalid.clone();
        unsigned.receipt_sha256.clear();
        invalid.receipt_sha256 = digest_serialized(&unsigned).unwrap();
        assert_eq!(
            finalize_reader_exposure_v1(
                &query(), &invalid, &[], 100, 100, 100, false,
                digest("request"), digest("response"), digest("reader"), digest("generation"), 1, 1
            ),
            Err(AdapterRejectV1::QueryInputMismatch)
        );
    }


    #[test]
    fn harness_commit_must_be_exact_lowercase_git_object_id() {
        let mut binding=run_binding();
        binding.harness_commit_sha="not-a-commit".into();
        assert_eq!(binding.seal(),Err(AdapterRejectV1::InvalidIdentity));
        let mut binding=run_binding();
        binding.harness_commit_sha="A".repeat(40);
        assert_eq!(binding.seal(),Err(AdapterRejectV1::InvalidIdentity));
        let mut binding=run_binding();
        binding.harness_commit_sha="a".repeat(40);
        assert!(binding.seal().is_ok());
    }


    #[test]
    fn prequery_selection_requires_a_canonical_decision_exposure_link() {
        let query=query();
        let mut selected=item(0,"bad-link");
        selected.decision_evidence_use_link_sha256="not-a-sha256".into();
        assert_eq!(
            seal_pre_query_selection_v1(&query,digest("policy"),vec![selected]),
            Err(AdapterRejectV1::InvalidDigest)
        );
    }

}
