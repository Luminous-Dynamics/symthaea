// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent verification of source bytes and ancestry metadata.
//!
//! This is a consumer-side verifier for a fourth DSSE authority. It checks
//! actual source/payload bytes supplied by an independent fetch boundary,
//! binds every graph node (including ancestors) to a signed origin receipt,
//! and revalidates the receipt at use time. It does not itself fetch URLs,
//! decide whether a publisher is authoritative, or prove that an upstream
//! publisher disclosed every dependency.

use std::collections::BTreeMap;

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::qualification_dsse::{
    parse_dsse_envelope_v1, parse_strict_json_bounded, DSSE_PAYLOAD_TYPE_V1,
    IN_TOTO_STATEMENT_TYPE_V1,
};
use crate::qualification_trust::PinnedEd25519QualificationVerifierV1;
use crate::source_lineage_graph::{
    VerifiedSourceLineageGraphV1, VerifiedSourceLineageNodeBindingV1,
};

pub const SOURCE_ORIGIN_VERIFIER_ID_V1: &str =
    "mace-independent-source-origin-verifier";
pub const MACE_SOURCE_ORIGIN_PREDICATE_TYPE_V1: &str =
    "https://luminousdynamics.io/attestations/mace/source-origin/v1";

const MAX_ORIGIN_ENVELOPES_V1: usize = 100_000;
const MAX_ORIGIN_ENVELOPE_BYTES_V1: usize = 2 * 1024 * 1024;
const MAX_ORIGIN_PAYLOAD_BYTES_V1: usize = 1024 * 1024;
const MAX_SOURCE_BYTES_V1: usize = 64 * 1024 * 1024;
const MAX_EVIDENCE_PAYLOAD_BYTES_V1: usize = 16 * 1024 * 1024;
const MAX_DERIVATION_RECEIPT_BYTES_V1: usize = 8 * 1024 * 1024;
const MAX_TOTAL_ARTIFACT_BYTES_V1: u64 = 512 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceOriginRejectV1 {
    InvalidEnvelope,
    InvalidJson,
    InvalidReceipt,
    InvalidDigest,
    WrongPredicateType,
    SubjectMismatch,
    BindingMismatch,
    ProfileMismatch,
    WrongVerifierRole,
    SharedAuthority,
    InvalidSignature,
    UnknownOrRevokedSigner,
    NotYetValid,
    Expired,
    MissingArtifact,
    UnexpectedArtifact,
    DuplicateArtifact,
    MissingReceipt,
    UnexpectedReceipt,
    DuplicateReceipt,
    SourceBytesMismatch,
    PayloadBytesMismatch,
    DerivationReceiptMismatch,
    ArtifactTooLarge,
    AggregateBudgetExceeded,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceOriginArtifactInputV1<'a> {
    /// Exact graph-node identifier. Inputs must cover every node, not only
    /// candidate nodes that satisfy a DecisionNeed requirement.
    pub evidence_id: &'a str,
    /// Raw bytes acquired through the independent source-fetch boundary.
    pub source_bytes: &'a [u8],
    /// Exact bytes represented by the graph node's payload digest.
    pub payload_bytes: &'a [u8],
    /// Exact derivation-receipt bytes for non-root nodes. Root nodes must omit it.
    pub derivation_receipt_bytes: Option<&'a [u8]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceOriginEnvelopeInputV1<'a> {
    pub evidence_id: &'a str,
    /// Exact UTF-8 DSSE envelope returned by the independent origin verifier.
    pub envelope_json: &'a str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SourceOriginReceiptV1 {
    run_id: String,
    query_invocation_id: String,
    graph_sha256: String,
    evidence_id: String,
    source_id: String,
    source_sha256: String,
    payload_sha256: String,
    evidence_class_sha256: String,
    parent_evidence_ids: Vec<String>,
    derivation_receipt_sha256: Option<String>,
    common_cause_group_sha256: Vec<String>,
    source_bytes_len: u64,
    payload_bytes_len: u64,
    derivation_receipt_bytes_len: u64,
    status: String,
    issued_at_unix: u64,
    expires_at_unix: u64,
    issuer_profile_sha256: String,
    key_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct VerifiedSourceOriginReceiptV1 {
    receipt: SourceOriginReceiptV1,
    envelope_sha256: String,
    pae: Vec<u8>,
    signature: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ArtifactLengthsV1 {
    source_bytes_len: u64,
    payload_bytes_len: u64,
    derivation_receipt_bytes_len: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedSourceOriginSetV1 {
    graph_sha256: String,
    origin_profile_sha256: String,
    receipts: Vec<VerifiedSourceOriginReceiptV1>,
}

impl VerifiedSourceOriginSetV1 {
    pub fn len(&self) -> usize { self.receipts.len() }
    pub fn is_empty(&self) -> bool { self.receipts.is_empty() }
    pub fn graph_sha256(&self) -> &str { &self.graph_sha256 }
    pub fn origin_profile_sha256(&self) -> &str { &self.origin_profile_sha256 }

    /// Signer identities retained from the independently verified per-node origin receipts.
    pub(crate) fn signer_key_ids(&self) -> Vec<String> {
        self.receipts.iter().map(|item| item.receipt.key_id.clone()).collect()
    }

    /// Exact envelope digests for the complete origin bundle, in evidence-id order.
    pub(crate) fn envelope_sha256s(&self) -> Vec<String> {
        let mut digests = self.receipts.iter()
            .map(|item| item.envelope_sha256.clone())
            .collect::<Vec<_>>();
        digests.sort();
        digests
    }

    /// Exact receipt-envelope digest for one graph node, used to bind separate
    /// acquisition witnesses to the four-authority source-origin result.
    pub(crate) fn envelope_sha256_for(&self, evidence_id: &str) -> Option<&str> {
        self.receipts.iter()
            .find(|item| item.receipt.evidence_id == evidence_id)
            .map(|item| item.envelope_sha256.as_str())
    }

    /// Recheck exact source/payload/derivation bytes and every DSSE signature
    /// against the current pinned source-origin profile. Call this after
    /// replaying the graph and revalidating the other three authorities at the
    /// same trusted current time.
    pub fn validate_at_use(
        &self,
        graph: &VerifiedSourceLineageGraphV1,
        artifacts: &[SourceOriginArtifactInputV1<'_>],
        current_profile: &PinnedEd25519QualificationVerifierV1,
        now_unix: u64,
    ) -> Result<(), SourceOriginRejectV1> {
        validate_origin_profile(graph, current_profile, &self.origin_profile_sha256)?;
        if graph.graph_sha256() != self.graph_sha256 {
            return Err(SourceOriginRejectV1::BindingMismatch);
        }
        let lengths = validate_artifacts(graph, artifacts)?;
        if self.receipts.len() != graph.node_bindings().len() {
            return Err(SourceOriginRejectV1::MissingReceipt);
        }
        for item in &self.receipts {
            if graph.authority_key_ids().iter()
                .any(|key| key == &item.receipt.key_id)
            {
                return Err(SourceOriginRejectV1::SharedAuthority);
            }
            let node = graph.node_bindings().iter()
                .find(|node| node.evidence_id == item.receipt.evidence_id)
                .ok_or(SourceOriginRejectV1::UnexpectedReceipt)?;
            let measured = lengths.get(&node.evidence_id)
                .ok_or(SourceOriginRejectV1::MissingArtifact)?;
            validate_receipt_binding(
                &item.receipt, graph, node, measured, &self.origin_profile_sha256,
            )?;
            validate_time_and_signature(
                &item.receipt, &item.pae, &item.signature, current_profile, now_unix,
            )?;
        }
        Ok(())
    }
}

/// Verify one independent origin attestation for every node in a replayed
/// source-lineage graph, including all ancestor nodes.
///
/// `artifacts` must come from an independently controlled acquisition path.
/// The function hashes the supplied raw bytes itself; it never trusts a
/// resolver-provided digest without recomputing it.
pub fn verify_source_origin_bundle_v1(
    graph: &VerifiedSourceLineageGraphV1,
    artifacts: &[SourceOriginArtifactInputV1<'_>],
    envelopes: &[SourceOriginEnvelopeInputV1<'_>],
    expected_origin_profile_sha256: &str,
    origin_profile: &PinnedEd25519QualificationVerifierV1,
    now_unix: u64,
) -> Result<VerifiedSourceOriginSetV1, SourceOriginRejectV1> {
    validate_origin_profile(graph, origin_profile, expected_origin_profile_sha256)?;
    let lengths = validate_artifacts(graph, artifacts)?;
    if envelopes.len() != graph.node_bindings().len()
        || envelopes.len() > MAX_ORIGIN_ENVELOPES_V1
    {
        return Err(SourceOriginRejectV1::MissingReceipt);
    }

    let mut by_id = BTreeMap::new();
    for input in envelopes {
        if input.evidence_id.trim().is_empty()
            || by_id.insert(input.evidence_id, input.envelope_json).is_some()
        {
            return Err(SourceOriginRejectV1::DuplicateReceipt);
        }
    }
    if by_id.len() != graph.node_bindings().len() {
        return Err(SourceOriginRejectV1::MissingReceipt);
    }

    let mut verified = Vec::with_capacity(graph.node_bindings().len());
    for node in graph.node_bindings() {
        let envelope = by_id.get(node.evidence_id.as_str())
            .ok_or(SourceOriginRejectV1::MissingReceipt)?;
        let measured = lengths.get(&node.evidence_id)
            .ok_or(SourceOriginRejectV1::MissingArtifact)?;
        let binding = ExpectedOriginBindingV1 {
            graph,
            node,
            measured,
            expected_profile_sha256: expected_origin_profile_sha256,
        };
        let item = verify_one_source_origin_receipt_v1(
            envelope, &binding, origin_profile, now_unix,
        )?;
        if graph.authority_key_ids().iter().any(|key| key == &item.receipt.key_id) {
            return Err(SourceOriginRejectV1::SharedAuthority);
        }
        verified.push(item);
    }

    if by_id.keys().any(|id| !graph.node_bindings().iter().any(|node| node.evidence_id.as_str() == *id)) {
        return Err(SourceOriginRejectV1::UnexpectedReceipt);
    }
    verified.sort_by(|a, b| a.receipt.evidence_id.cmp(&b.receipt.evidence_id));

    Ok(VerifiedSourceOriginSetV1 {
        graph_sha256: graph.graph_sha256().to_owned(),
        origin_profile_sha256: expected_origin_profile_sha256.to_owned(),
        receipts: verified,
    })
}

struct ExpectedOriginBindingV1<'a> {
    graph: &'a VerifiedSourceLineageGraphV1,
    node: &'a VerifiedSourceLineageNodeBindingV1,
    measured: &'a ArtifactLengthsV1,
    expected_profile_sha256: &'a str,
}

fn validate_origin_profile(
    graph: &VerifiedSourceLineageGraphV1,
    profile: &PinnedEd25519QualificationVerifierV1,
    expected_profile_sha256: &str,
) -> Result<(), SourceOriginRejectV1> {
    if profile.verifier_id() != SOURCE_ORIGIN_VERIFIER_ID_V1 {
        return Err(SourceOriginRejectV1::WrongVerifierRole);
    }
    if !valid_sha256(expected_profile_sha256)
        || profile.profile_sha256() != expected_profile_sha256
    {
        return Err(SourceOriginRejectV1::ProfileMismatch);
    }
    if graph.authority_profile_sha256().iter()
        .any(|authority| authority == expected_profile_sha256)
    {
        return Err(SourceOriginRejectV1::SharedAuthority);
    }
    Ok(())
}

fn validate_artifacts(
    graph: &VerifiedSourceLineageGraphV1,
    artifacts: &[SourceOriginArtifactInputV1<'_>],
) -> Result<BTreeMap<String, ArtifactLengthsV1>, SourceOriginRejectV1> {
    if artifacts.len() != graph.node_bindings().len() {
        return Err(if artifacts.len() < graph.node_bindings().len() {
            SourceOriginRejectV1::MissingArtifact
        } else {
            SourceOriginRejectV1::UnexpectedArtifact
        });
    }
    let mut by_id = BTreeMap::new();
    let mut total = 0u64;
    for input in artifacts {
        if input.evidence_id.trim().is_empty()
            || by_id.contains_key(input.evidence_id)
        {
            return Err(SourceOriginRejectV1::DuplicateArtifact);
        }
        if input.source_bytes.is_empty() || input.payload_bytes.is_empty()
            || input.source_bytes.len() > MAX_SOURCE_BYTES_V1
            || input.payload_bytes.len() > MAX_EVIDENCE_PAYLOAD_BYTES_V1
        {
            return Err(SourceOriginRejectV1::ArtifactTooLarge);
        }
        let source_len = input.source_bytes.len() as u64;
        let payload_len = input.payload_bytes.len() as u64;
        let derivation_len = match input.derivation_receipt_bytes {
            Some(bytes) if !bytes.is_empty() && bytes.len() <= MAX_DERIVATION_RECEIPT_BYTES_V1 =>
                bytes.len() as u64,
            Some(_) => return Err(SourceOriginRejectV1::ArtifactTooLarge),
            None => 0,
        };
        total = total.checked_add(source_len)
            .and_then(|n| n.checked_add(payload_len))
            .and_then(|n| n.checked_add(derivation_len))
            .ok_or(SourceOriginRejectV1::AggregateBudgetExceeded)?;
        if total > MAX_TOTAL_ARTIFACT_BYTES_V1 {
            return Err(SourceOriginRejectV1::AggregateBudgetExceeded);
        }
        by_id.insert(input.evidence_id.to_owned(), (input, ArtifactLengthsV1 {
            source_bytes_len: source_len,
            payload_bytes_len: payload_len,
            derivation_receipt_bytes_len: derivation_len,
        }));
    }

    let mut lengths = BTreeMap::new();
    for node in graph.node_bindings() {
        let (input, measured) = by_id.remove(&node.evidence_id)
            .ok_or(SourceOriginRejectV1::MissingArtifact)?;
        validate_node_artifact_bytes(node, input, &measured)?;
        lengths.insert(node.evidence_id.clone(), measured);
    }
    if !by_id.is_empty() {
        return Err(SourceOriginRejectV1::UnexpectedArtifact);
    }
    Ok(lengths)
}

fn validate_node_artifact_bytes(
    node: &VerifiedSourceLineageNodeBindingV1,
    input: &SourceOriginArtifactInputV1<'_>,
    measured: &ArtifactLengthsV1,
) -> Result<(), SourceOriginRejectV1> {
    if input.evidence_id != node.evidence_id {
        return Err(SourceOriginRejectV1::BindingMismatch);
    }
    if measured.source_bytes_len != input.source_bytes.len() as u64
        || measured.payload_bytes_len != input.payload_bytes.len() as u64
        || measured.derivation_receipt_bytes_len
            != input.derivation_receipt_bytes.map_or(0, |bytes| bytes.len() as u64)
    {
        return Err(SourceOriginRejectV1::BindingMismatch);
    }
    if sha256_hex(input.source_bytes) != node.source_sha256 {
        return Err(SourceOriginRejectV1::SourceBytesMismatch);
    }
    if sha256_hex(input.payload_bytes) != node.payload_sha256 {
        return Err(SourceOriginRejectV1::PayloadBytesMismatch);
    }
    match (&node.derivation_receipt_sha256, input.derivation_receipt_bytes) {
        (None, None) if node.parent_evidence_ids.is_empty() => {}
        (Some(expected), Some(bytes)) if !node.parent_evidence_ids.is_empty() => {
            if sha256_hex(bytes) != *expected {
                return Err(SourceOriginRejectV1::DerivationReceiptMismatch);
            }
        }
        _ => return Err(SourceOriginRejectV1::DerivationReceiptMismatch),
    }
    Ok(())
}

fn validate_receipt_binding(
    receipt: &SourceOriginReceiptV1,
    graph: &VerifiedSourceLineageGraphV1,
    node: &VerifiedSourceLineageNodeBindingV1,
    measured: &ArtifactLengthsV1,
    expected_profile_sha256: &str,
) -> Result<(), SourceOriginRejectV1> {
    if receipt.run_id != graph.run_id()
        || receipt.query_invocation_id != graph.query_invocation_id()
        || receipt.graph_sha256 != graph.graph_sha256()
        || receipt.evidence_id != node.evidence_id
        || receipt.source_id != node.source_id
        || receipt.source_sha256 != node.source_sha256
        || receipt.payload_sha256 != node.payload_sha256
        || receipt.evidence_class_sha256 != node.evidence_class_sha256
        || receipt.parent_evidence_ids != node.parent_evidence_ids
        || receipt.derivation_receipt_sha256 != node.derivation_receipt_sha256
        || receipt.common_cause_group_sha256 != node.common_cause_group_sha256
        || receipt.source_bytes_len != measured.source_bytes_len
        || receipt.payload_bytes_len != measured.payload_bytes_len
        || receipt.derivation_receipt_bytes_len != measured.derivation_receipt_bytes_len
        || receipt.issuer_profile_sha256 != expected_profile_sha256
        || receipt.status != "origin_verified"
    {
        return Err(SourceOriginRejectV1::BindingMismatch);
    }
    Ok(())
}

fn verify_one_source_origin_receipt_v1(
    envelope_json: &str,
    binding: &ExpectedOriginBindingV1<'_>,
    profile: &PinnedEd25519QualificationVerifierV1,
    now_unix: u64,
) -> Result<VerifiedSourceOriginReceiptV1, SourceOriginRejectV1> {
    if envelope_json.len() > MAX_ORIGIN_ENVELOPE_BYTES_V1 {
        return Err(SourceOriginRejectV1::InvalidEnvelope);
    }
    let envelope = parse_dsse_envelope_v1(envelope_json, DSSE_PAYLOAD_TYPE_V1)
        .map_err(|_| SourceOriginRejectV1::InvalidEnvelope)?;
    if envelope.payload.len() > MAX_ORIGIN_PAYLOAD_BYTES_V1 {
        return Err(SourceOriginRejectV1::InvalidEnvelope);
    }
    let statement_value = parse_strict_json_bounded(
        &envelope.payload, MAX_ORIGIN_PAYLOAD_BYTES_V1,
    ).map_err(|_| SourceOriginRejectV1::InvalidJson)?;
    let statement = statement_value.as_object()
        .ok_or(SourceOriginRejectV1::InvalidJson)?;
    const STATEMENT_FIELDS: &[&str] = &["_type", "subject", "predicateType", "predicate"];
    if statement.len() != STATEMENT_FIELDS.len()
        || statement.keys().any(|field| !STATEMENT_FIELDS.contains(&field.as_str()))
    {
        return Err(SourceOriginRejectV1::InvalidJson);
    }
    if statement.get("_type").and_then(Value::as_str) != Some(IN_TOTO_STATEMENT_TYPE_V1)
        || statement.get("predicateType").and_then(Value::as_str)
            != Some(MACE_SOURCE_ORIGIN_PREDICATE_TYPE_V1)
    {
        return Err(SourceOriginRejectV1::WrongPredicateType);
    }
    let subjects = statement.get("subject").and_then(Value::as_array)
        .ok_or(SourceOriginRejectV1::InvalidJson)?;
    if subjects.len() != 1 {
        return Err(SourceOriginRejectV1::SubjectMismatch);
    }
    let subject = subjects[0].as_object().ok_or(SourceOriginRejectV1::InvalidJson)?;
    let subject_digest = subject.get("digest").and_then(Value::as_object)
        .ok_or(SourceOriginRejectV1::SubjectMismatch)?;
    let expected_subject_name = format!(
        "urn:luminous:mace:source-origin-node:v1:{}",
        binding.node.evidence_id,
    );
    if subject.len() != 2
        || subject_digest.len() != 1
        || subject.get("name").and_then(Value::as_str) != Some(expected_subject_name.as_str())
        || subject_digest.get("sha256").and_then(Value::as_str)
            != Some(binding.node.source_sha256.as_str())
    {
        return Err(SourceOriginRejectV1::SubjectMismatch);
    }

    let predicate = statement.get("predicate").ok_or(SourceOriginRejectV1::InvalidJson)?;
    let receipt = parse_source_origin_predicate(predicate)?;
    validate_receipt_binding(
        &receipt, binding.graph, binding.node, binding.measured,
        binding.expected_profile_sha256,
    )?;
    if envelope.key_id_hint.as_deref() != Some(receipt.key_id.as_str()) {
        return Err(SourceOriginRejectV1::BindingMismatch);
    }
    validate_time_and_signature(&receipt, &envelope.pae, &envelope.signature, profile, now_unix)?;

    Ok(VerifiedSourceOriginReceiptV1 {
        envelope_sha256: sha256_hex(envelope_json.as_bytes()),
        receipt,
        pae: envelope.pae,
        signature: envelope.signature,
    })
}

fn validate_time_and_signature(
    receipt: &SourceOriginReceiptV1,
    pae: &[u8],
    signature: &[u8],
    profile: &PinnedEd25519QualificationVerifierV1,
    now_unix: u64,
) -> Result<(), SourceOriginRejectV1> {
    if now_unix < receipt.issued_at_unix {
        return Err(SourceOriginRejectV1::NotYetValid);
    }
    if now_unix >= receipt.expires_at_unix {
        return Err(SourceOriginRejectV1::Expired);
    }
    if receipt.key_id.is_empty() || receipt.key_id.len() > 256 {
        return Err(SourceOriginRejectV1::InvalidReceipt);
    }
    profile.verify_message(
        &receipt.issuer_profile_sha256,
        &receipt.key_id,
        pae,
        signature,
        receipt.issued_at_unix,
        now_unix,
    ).map_err(|error| match error {
        crate::evidence_qualification_gate::EvidenceQualificationRejectV1::UnknownKey
        | crate::evidence_qualification_gate::EvidenceQualificationRejectV1::RevokedKey =>
            SourceOriginRejectV1::UnknownOrRevokedSigner,
        crate::evidence_qualification_gate::EvidenceQualificationRejectV1::KeyNotYetValid =>
            SourceOriginRejectV1::NotYetValid,
        crate::evidence_qualification_gate::EvidenceQualificationRejectV1::KeyExpired =>
            SourceOriginRejectV1::Expired,
        _ => SourceOriginRejectV1::InvalidSignature,
    })
}

fn parse_source_origin_predicate(value: &Value) -> Result<SourceOriginReceiptV1, SourceOriginRejectV1> {
    let object = value.as_object().ok_or(SourceOriginRejectV1::InvalidJson)?;
    const FIELDS: &[&str] = &[
        "schema_version", "run_id", "query_invocation_id", "graph_sha256",
        "evidence_id", "source_id", "source_sha256", "payload_sha256",
        "evidence_class_sha256", "parent_evidence_ids", "derivation_receipt_sha256",
        "common_cause_group_sha256", "source_bytes_len", "payload_bytes_len",
        "derivation_receipt_bytes_len", "status", "issued_at_unix", "expires_at_unix",
        "issuer_profile_sha256", "key_id",
    ];
    if object.len() != FIELDS.len()
        || object.keys().any(|field| !FIELDS.contains(&field.as_str()))
        || object.get("schema_version").and_then(Value::as_str)
            != Some("mace_source_origin_receipt_v1")
    {
        return Err(SourceOriginRejectV1::InvalidJson);
    }
    let string = |field: &str| -> Result<String, SourceOriginRejectV1> {
        object.get(field).and_then(Value::as_str).map(str::to_owned)
            .ok_or(SourceOriginRejectV1::InvalidJson)
    };
    let integer = |field: &str| -> Result<u64, SourceOriginRejectV1> {
        object.get(field).and_then(Value::as_u64).ok_or(SourceOriginRejectV1::InvalidJson)
    };
    let optional_digest = match object.get("derivation_receipt_sha256") {
        Some(Value::Null) => None,
        Some(Value::String(value)) if valid_sha256(value) => Some(value.clone()),
        _ => return Err(SourceOriginRejectV1::InvalidDigest),
    };
    let receipt = SourceOriginReceiptV1 {
        run_id: string("run_id")?,
        query_invocation_id: string("query_invocation_id")?,
        graph_sha256: string("graph_sha256")?,
        evidence_id: string("evidence_id")?,
        source_id: string("source_id")?,
        source_sha256: string("source_sha256")?,
        payload_sha256: string("payload_sha256")?,
        evidence_class_sha256: string("evidence_class_sha256")?,
        parent_evidence_ids: string_array(object, "parent_evidence_ids", 1_024, false)?,
        derivation_receipt_sha256: optional_digest,
        common_cause_group_sha256: string_array(object, "common_cause_group_sha256", 1_024, true)?,
        source_bytes_len: integer("source_bytes_len")?,
        payload_bytes_len: integer("payload_bytes_len")?,
        derivation_receipt_bytes_len: integer("derivation_receipt_bytes_len")?,
        status: string("status")?,
        issued_at_unix: integer("issued_at_unix")?,
        expires_at_unix: integer("expires_at_unix")?,
        issuer_profile_sha256: string("issuer_profile_sha256")?,
        key_id: string("key_id")?,
    };
    for digest in [
        receipt.graph_sha256.as_str(),
        receipt.source_sha256.as_str(),
        receipt.payload_sha256.as_str(),
        receipt.evidence_class_sha256.as_str(),
        receipt.issuer_profile_sha256.as_str(),
    ] {
        if !valid_sha256(digest) { return Err(SourceOriginRejectV1::InvalidDigest); }
    }
    for atom in [
        receipt.run_id.as_str(), receipt.query_invocation_id.as_str(),
        receipt.evidence_id.as_str(), receipt.source_id.as_str(), receipt.key_id.as_str(),
    ] {
        if atom.trim().is_empty() || atom != atom.trim()
            || atom.chars().any(char::is_control) || atom.len() > 1_024
        {
            return Err(SourceOriginRejectV1::InvalidReceipt);
        }
    }
    if receipt.issued_at_unix >= receipt.expires_at_unix
        || receipt.expires_at_unix - receipt.issued_at_unix > 31 * 24 * 60 * 60
        || receipt.source_bytes_len == 0
        || receipt.payload_bytes_len == 0
        || receipt.source_bytes_len > MAX_SOURCE_BYTES_V1 as u64
        || receipt.payload_bytes_len > MAX_EVIDENCE_PAYLOAD_BYTES_V1 as u64
        || receipt.derivation_receipt_bytes_len > MAX_DERIVATION_RECEIPT_BYTES_V1 as u64
        || receipt.status != "origin_verified"
        || (receipt.parent_evidence_ids.is_empty()
            != (receipt.derivation_receipt_sha256.is_none()
                && receipt.derivation_receipt_bytes_len == 0))
        || (receipt.derivation_receipt_sha256.is_some()
            != (receipt.derivation_receipt_bytes_len > 0))
    {
        return Err(SourceOriginRejectV1::InvalidReceipt);
    }
    Ok(receipt)
}

fn string_array(
    object: &serde_json::Map<String, Value>,
    field: &str,
    maximum: usize,
    digest: bool,
) -> Result<Vec<String>, SourceOriginRejectV1> {
    let values = object.get(field).and_then(Value::as_array)
        .ok_or(SourceOriginRejectV1::InvalidJson)?;
    if values.len() > maximum { return Err(SourceOriginRejectV1::InvalidReceipt); }
    let parsed = values.iter().map(|value| {
        let value = value.as_str().ok_or(SourceOriginRejectV1::InvalidJson)?;
        if value.trim().is_empty() || value != value.trim()
            || value.chars().any(char::is_control)
            || digest && !valid_sha256(value)
        {
            return Err(if digest { SourceOriginRejectV1::InvalidDigest }
                else { SourceOriginRejectV1::InvalidReceipt });
        }
        Ok(value.to_owned())
    }).collect::<Result<Vec<_>, _>>()?;
    if parsed.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(SourceOriginRejectV1::InvalidReceipt);
    }
    Ok(parsed)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use crate::qualification_trust::{
        qualification_key_id_v1, QualificationKeyStatusV1, QualificationTrustKeyV1,
        QualificationTrustProfileV1,
    };

    fn signed_origin_fixture() -> (String, PinnedEd25519QualificationVerifierV1) {
        let signing_key = SigningKey::from_bytes(&[7u8; 32]);
        let public_key = signing_key.verifying_key().to_bytes();
        let key_id = qualification_key_id_v1(&public_key);
        let profile = QualificationTrustProfileV1 {
            schema_version: 1,
            verifier_id: SOURCE_ORIGIN_VERIFIER_ID_V1.to_owned(),
            profile_version: 1,
            keys: vec![QualificationTrustKeyV1 {
                key_id: key_id.clone(),
                public_key,
                status: QualificationKeyStatusV1::Active,
                not_before_unix: 1,
                not_after_unix: 1_000,
            }],
        };
        let profile_sha256 = profile.profile_sha256().expect("valid origin profile");
        let verifier = PinnedEd25519QualificationVerifierV1::from_pinned_profile(
            &profile, &profile_sha256, 1,
        ).expect("pinned origin profile");

        let source_bytes = b"source fixture";
        let payload_bytes = b"payload fixture";
        let predicate = serde_json::json!({
            "schema_version": "mace_source_origin_receipt_v1",
            "run_id": "run-1",
            "query_invocation_id": "invocation-1",
            "graph_sha256": "a".repeat(64),
            "evidence_id": "evidence-1",
            "source_id": "source-1",
            "source_sha256": sha256_hex(source_bytes),
            "payload_sha256": sha256_hex(payload_bytes),
            "evidence_class_sha256": "b".repeat(64),
            "parent_evidence_ids": [],
            "derivation_receipt_sha256": null,
            "common_cause_group_sha256": [],
            "source_bytes_len": source_bytes.len(),
            "payload_bytes_len": payload_bytes.len(),
            "derivation_receipt_bytes_len": 0,
            "status": "origin_verified",
            "issued_at_unix": 10,
            "expires_at_unix": 50,
            "issuer_profile_sha256": profile_sha256.clone(),
            "key_id": key_id.clone()
        });
        let statement = serde_json::json!({
            "_type": IN_TOTO_STATEMENT_TYPE_V1,
            "subject": [{
                "name": "urn:luminous:mace:source-origin-node:v1:evidence-1",
                "digest": { "sha256": sha256_hex(source_bytes) }
            }],
            "predicateType": MACE_SOURCE_ORIGIN_PREDICATE_TYPE_V1,
            "predicate": predicate
        });
        let payload = serde_json::to_vec(&statement).expect("statement JSON");
        let pae = crate::qualification_dsse::dsse_pae_v1(DSSE_PAYLOAD_TYPE_V1, &payload);
        let signature = signing_key.sign(&pae).to_bytes();
        let envelope = crate::qualification_dsse::encode_dsse_envelope_v1(
            &payload, &key_id, &signature,
        ).expect("DSSE envelope");
        (envelope, verifier)
    }

    #[test]
    fn signed_origin_dsse_verifies_and_expires_at_the_exact_boundary() {
        let (envelope_json, verifier) = signed_origin_fixture();
        let envelope = parse_dsse_envelope_v1(&envelope_json, DSSE_PAYLOAD_TYPE_V1)
            .expect("DSSE envelope");
        let statement = parse_strict_json_bounded(&envelope.payload, MAX_ORIGIN_PAYLOAD_BYTES_V1)
            .expect("strict statement JSON");
        let predicate = statement.get("predicate").expect("origin predicate");
        let receipt = parse_source_origin_predicate(predicate).expect("typed origin receipt");
        assert_eq!(
            validate_time_and_signature(&receipt, &envelope.pae, &envelope.signature, &verifier, 20),
            Ok(())
        );
        assert_eq!(
            validate_time_and_signature(&receipt, &envelope.pae, &envelope.signature, &verifier, 50),
            Err(SourceOriginRejectV1::Expired)
        );
    }

    #[test]
    fn modified_signed_payload_is_rejected_by_ed25519_verification() {
        let (envelope_json, verifier) = signed_origin_fixture();
        let envelope = parse_dsse_envelope_v1(&envelope_json, DSSE_PAYLOAD_TYPE_V1)
            .expect("DSSE envelope");
        let mut statement = parse_strict_json_bounded(&envelope.payload, MAX_ORIGIN_PAYLOAD_BYTES_V1)
            .expect("strict statement JSON");
        let predicate = statement.get_mut("predicate").and_then(Value::as_object_mut)
            .expect("origin predicate object");
        predicate.insert("graph_sha256".to_owned(), Value::String("c".repeat(64)));
        let receipt = parse_source_origin_predicate(
            statement.get("predicate").expect("altered origin predicate"),
        ).expect("validly shaped altered receipt");
        let altered_payload = serde_json::to_vec(&statement).expect("altered statement JSON");
        let altered_pae = crate::qualification_dsse::dsse_pae_v1(
            DSSE_PAYLOAD_TYPE_V1, &altered_payload,
        );
        assert_eq!(
            validate_time_and_signature(
                &receipt, &altered_pae, &envelope.signature, &verifier, 20,
            ),
            Err(SourceOriginRejectV1::InvalidSignature)
        );
    }

    #[test]
    fn exact_source_and_payload_bytes_are_domain_independent_hashes() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_ne!(sha256_hex(b"source"), sha256_hex(b"payload"));
    }

    fn test_node(
        evidence_id: &str,
        source_bytes: &[u8],
        payload_bytes: &[u8],
        parent_evidence_ids: Vec<String>,
        derivation_receipt_bytes: Option<&[u8]>,
    ) -> VerifiedSourceLineageNodeBindingV1 {
        VerifiedSourceLineageNodeBindingV1 {
            evidence_id: evidence_id.to_owned(),
            source_id: "source:test".to_owned(),
            source_sha256: sha256_hex(source_bytes),
            payload_sha256: sha256_hex(payload_bytes),
            evidence_class_sha256: "a".repeat(64),
            parent_evidence_ids,
            derivation_receipt_sha256: derivation_receipt_bytes.map(sha256_hex),
            common_cause_group_sha256: Vec::new(),
        }
    }

    #[test]
    fn exact_artifact_bytes_match_the_verified_node_commitments() {
        let source = b"immutable source bytes";
        let payload = b"derived evidence payload";
        let derivation = b"transformation receipt";
        let node = test_node(
            "derived-1", source, payload, vec!["root-1".to_owned()], Some(derivation),
        );
        let input = SourceOriginArtifactInputV1 {
            evidence_id: "derived-1",
            source_bytes: source,
            payload_bytes: payload,
            derivation_receipt_bytes: Some(derivation),
        };
        let measured = ArtifactLengthsV1 {
            source_bytes_len: source.len() as u64,
            payload_bytes_len: payload.len() as u64,
            derivation_receipt_bytes_len: derivation.len() as u64,
        };
        assert_eq!(validate_node_artifact_bytes(&node, &input, &measured), Ok(()));
    }

    #[test]
    fn altered_source_payload_and_derivation_bytes_are_rejected() {
        let source = b"immutable source bytes";
        let payload = b"derived evidence payload";
        let derivation = b"transformation receipt";
        let node = test_node(
            "derived-1", source, payload, vec!["root-1".to_owned()], Some(derivation),
        );
        let measured = ArtifactLengthsV1 {
            source_bytes_len: source.len() as u64,
            payload_bytes_len: payload.len() as u64,
            derivation_receipt_bytes_len: derivation.len() as u64,
        };
        let wrong_source = SourceOriginArtifactInputV1 {
            evidence_id: "derived-1",
            source_bytes: b"immutablf source bytes",
            payload_bytes: payload,
            derivation_receipt_bytes: Some(derivation),
        };
        assert_eq!(
            validate_node_artifact_bytes(&node, &wrong_source, &measured),
            Err(SourceOriginRejectV1::SourceBytesMismatch)
        );
        let wrong_payload = SourceOriginArtifactInputV1 {
            source_bytes: source,
            payload_bytes: b"derived evidence payloae",
            ..wrong_source
        };
        assert_eq!(
            validate_node_artifact_bytes(&node, &wrong_payload, &measured),
            Err(SourceOriginRejectV1::PayloadBytesMismatch)
        );
        let wrong_derivation = SourceOriginArtifactInputV1 {
            source_bytes: source,
            payload_bytes: payload,
            derivation_receipt_bytes: Some(b"transformation receipx"),
            ..wrong_source
        };
        assert_eq!(
            validate_node_artifact_bytes(&node, &wrong_derivation, &measured),
            Err(SourceOriginRejectV1::DerivationReceiptMismatch)
        );
    }

    #[test]
    fn roots_reject_derivation_artifacts_and_derived_nodes_require_them() {
        let source = b"root source";
        let payload = b"root payload";
        let node = test_node("root-1", source, payload, Vec::new(), None);
        let root_input = SourceOriginArtifactInputV1 {
            evidence_id: "root-1",
            source_bytes: source,
            payload_bytes: payload,
            derivation_receipt_bytes: None,
        };
        let measured = ArtifactLengthsV1 {
            source_bytes_len: source.len() as u64,
            payload_bytes_len: payload.len() as u64,
            derivation_receipt_bytes_len: 0,
        };
        assert_eq!(validate_node_artifact_bytes(&node, &root_input, &measured), Ok(()));
        let unexpected = SourceOriginArtifactInputV1 {
            derivation_receipt_bytes: Some(b"unexpected"),
            ..root_input
        };
        assert_eq!(
            validate_node_artifact_bytes(&node, &unexpected, &measured),
            Err(SourceOriginRejectV1::BindingMismatch)
        );
    }

    #[test]
    fn source_origin_role_is_not_the_lineage_resolver_role() {
        assert_ne!(SOURCE_ORIGIN_VERIFIER_ID_V1, "mace-independent-source-lineage-resolver");
    }

    #[test]
    fn parent_receipts_must_be_sorted_and_unique() {
        let object = serde_json::json!({
            "parent_evidence_ids": ["parent-b", "parent-a"]
        });
        assert_eq!(
            string_array(object.as_object().unwrap(), "parent_evidence_ids", 10, false),
            Err(SourceOriginRejectV1::InvalidReceipt)
        );
    }
}
