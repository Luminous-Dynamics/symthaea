// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent source-lineage receipt verification for MACE DecisionSufficiency.
//!
//! The source-lineage resolver is a third authority, separate from evidence
//! qualification and decision-sufficiency evaluation. This module verifies
//! exact DSSE attestations and their binding to one qualified evidence set and
//! one consumer-owned DecisionNeed. It does not itself retrieve source bytes or
//! recompute the resolver's provenance graph.

use std::collections::BTreeSet;

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::decision_sufficiency::{
    DecisionNeedV1, DecisionSufficiencyReceiptV1, DecisionSufficiencyRejectV1,
    DecisionSufficiencyUseBindingV1, VerifiedDecisionSufficiencyV1,
};
use crate::evidence_qualification_gate::{
    EvidenceQualificationBindingV1, EvidenceQualificationSignatureVerifierV1,
};
use crate::qualification_dsse::{
    DSSE_PAYLOAD_TYPE_V1, IN_TOTO_STATEMENT_TYPE_V1,
    parse_dsse_envelope_v1, parse_strict_json_bounded,
};
use crate::qualification_trust::PinnedEd25519QualificationVerifierV1;
use crate::qualified_candidate_evidence::QualifiedCandidateEvidenceBatchV1;

pub const SOURCE_LINEAGE_VERIFIER_ID_V1: &str =
    "mace-independent-source-lineage-resolver";
pub const MACE_SOURCE_LINEAGE_PREDICATE_TYPE_V1: &str =
    "https://luminousdynamics.io/attestations/mace/source-lineage/v1";

const MAX_LINEAGE_ENVELOPES_V1: usize = 10_000;
const MAX_LINEAGE_ENVELOPE_BYTES_V1: usize = 12 * 1024 * 1024;
const MAX_LINEAGE_PAYLOAD_BYTES_V1: usize = 8 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SourceLineageRejectV1 {
    InvalidJson,
    InvalidEnvelope,
    InvalidIdentifier,
    InvalidDigest,
    InvalidReceipt,
    WrongPredicateType,
    SubjectMismatch,
    PayloadBindingMismatch,
    ProfileMismatch,
    WrongVerifierRole,
    SharedAuthority,
    NotIndependent,
    UnresolvedAncestors,
    MissingReceipt,
    UnexpectedReceipt,
    DuplicateReceipt,
    BindingMismatch,
    NotYetValid,
    Expired,
    InvalidSignature,
    UnknownOrRevokedSigner,
    Sufficiency(DecisionSufficiencyRejectV1),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceLineageEnvelopeInputV1<'a> {
    /// DecisionNeed requirement to which this proof belongs.
    pub requirement_id: &'a str,
    /// Root family whose independent lineage is being attested.
    pub root_family_sha256: &'a str,
    /// Exact UTF-8 DSSE JSON envelope. Its SHA-256 must match the sufficiency
    /// receipt's provenance_receipt_sha256 commitment byte-for-byte.
    pub envelope_json: &'a str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SourceLineageReceiptV1 {
    run_id: String,
    query_invocation_id: String,
    qualified_evidence_set_sha256: String,
    decision_need_sha256: String,
    requirement_id: String,
    evidence_class_sha256: String,
    root_family_sha256: String,
    source_graph_sha256: String,
    lineage_closure_sha256: String,
    source_node_count: u64,
    dependency_edge_count: u64,
    unresolved_ancestor_count: u64,
    status: String,
    issued_at_unix: u64,
    expires_at_unix: u64,
    issuer_profile_sha256: String,
    key_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct VerifiedSourceLineageReceiptV1 {
    receipt: SourceLineageReceiptV1,
    envelope_sha256: String,
    pae: Vec<u8>,
    signature: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedSourceLineageSetV1 {
    receipts: Vec<VerifiedSourceLineageReceiptV1>,
    sufficiency_receipt_sha256: String,
    qualified_evidence_set_sha256: String,
    decision_need_sha256: String,
    lineage_profile_sha256: String,
}

impl VerifiedSourceLineageSetV1 {
    pub fn len(&self) -> usize {
        self.receipts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.receipts.is_empty()
    }

    pub fn lineage_profile_sha256(&self) -> &str {
        &self.lineage_profile_sha256
    }

    /// Recheck every lineage receipt using the currently pinned resolver profile.
    /// Call this at the same trusted current time as qualification and
    /// sufficiency revalidation; a stale lineage proof cannot outlive either.
    pub fn validate_at_use(
        &self,
        qualified: &QualifiedCandidateEvidenceBatchV1,
        qualification_binding: &EvidenceQualificationBindingV1<'_>,
        current_qualification_verifier: &impl EvidenceQualificationSignatureVerifierV1,
        sufficiency: &VerifiedDecisionSufficiencyV1,
        sufficiency_binding: &DecisionSufficiencyUseBindingV1<'_>,
        current_evaluator_profile: &PinnedEd25519QualificationVerifierV1,
        decision_need: &DecisionNeedV1,
        current_lineage_profile: &PinnedEd25519QualificationVerifierV1,
    ) -> Result<(), SourceLineageRejectV1> {
        if qualification_binding.now_unix != sufficiency_binding.now_unix {
            return Err(SourceLineageRejectV1::BindingMismatch);
        }
        sufficiency.validate_at_use(
            qualified,
            qualification_binding,
            current_qualification_verifier,
            sufficiency_binding,
            current_evaluator_profile,
        ).map_err(SourceLineageRejectV1::Sufficiency)?;
        let now_unix = sufficiency_binding.now_unix;
        let sufficiency_receipt = sufficiency.receipt();
        let need_digest = decision_need.digest()
            .map_err(SourceLineageRejectV1::Sufficiency)?;
        if current_lineage_profile.verifier_id() != SOURCE_LINEAGE_VERIFIER_ID_V1 {
            return Err(SourceLineageRejectV1::WrongVerifierRole);
        }
        if current_lineage_profile.profile_sha256() != self.lineage_profile_sha256.as_str() {
            return Err(SourceLineageRejectV1::ProfileMismatch);
        }
        if current_lineage_profile.profile_sha256()
            == qualified.qualification_verifier_profile_sha256()
            || current_lineage_profile.profile_sha256()
                == sufficiency_receipt_profile(sufficiency)
        {
            return Err(SourceLineageRejectV1::SharedAuthority);
        }
        if sufficiency.receipt_sha256() != self.sufficiency_receipt_sha256
            || qualified.qualified_candidate_set_sha256() != self.qualified_evidence_set_sha256.as_str()
            || sufficiency_receipt.qualified_evidence_set_sha256 != self.qualified_evidence_set_sha256
            || need_digest != self.decision_need_sha256
            || sufficiency_receipt.decision_need_sha256 != self.decision_need_sha256
        {
            return Err(SourceLineageRejectV1::BindingMismatch);
        }
        let expected = expected_lineage_pairs(sufficiency_receipt, decision_need)?;
        let actual = self.receipts.iter().map(|item| {
            (
                item.receipt.requirement_id.as_str(),
                item.receipt.root_family_sha256.as_str(),
                item.envelope_sha256.as_str(),
            )
        }).collect::<BTreeSet<_>>();
        let expected_set = expected.iter().map(|(requirement_id, root, digest)| {
            (requirement_id.as_str(), root.as_str(), digest.as_str())
        }).collect::<BTreeSet<_>>();
        if actual != expected_set || actual.len() != self.receipts.len() {
            return Err(SourceLineageRejectV1::BindingMismatch);
        }

        for item in &self.receipts {
            validate_lineage_binding_at_use(
                &item.receipt,
                qualified,
                sufficiency_receipt,
                decision_need,
            )?;
            if item.receipt.key_id == qualified.qualification_signer_key_id()
                || item.receipt.key_id == sufficiency_receipt.key_id
            {
                return Err(SourceLineageRejectV1::SharedAuthority);
            }
            if now_unix < item.receipt.issued_at_unix {
                return Err(SourceLineageRejectV1::NotYetValid);
            }
            if now_unix >= item.receipt.expires_at_unix {
                return Err(SourceLineageRejectV1::Expired);
            }
            current_lineage_profile.verify_message(
                &item.receipt.issuer_profile_sha256,
                &item.receipt.key_id,
                &item.pae,
                &item.signature,
                item.receipt.issued_at_unix,
                now_unix,
            ).map_err(map_signature_error)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct ExpectedLineageBindingV1<'a> {
    run_id: &'a str,
    query_invocation_id: &'a str,
    qualified_evidence_set_sha256: &'a str,
    decision_need_sha256: &'a str,
    requirement_id: &'a str,
    evidence_class_sha256: &'a str,
    root_family_sha256: &'a str,
    expected_profile_sha256: &'a str,
}

/// Verify all source-lineage envelopes committed by a previously verified
/// DecisionSufficiency receipt. Every referenced envelope must be supplied,
/// exact-byte-hash matched, signed by the independent source-lineage role and
/// bound to the same qualified set and DecisionNeed.
pub fn verify_source_lineage_receipts_v1(
    qualified: &QualifiedCandidateEvidenceBatchV1,
    qualification_binding: &EvidenceQualificationBindingV1<'_>,
    current_qualification_verifier: &impl EvidenceQualificationSignatureVerifierV1,
    sufficiency: &VerifiedDecisionSufficiencyV1,
    sufficiency_binding: &DecisionSufficiencyUseBindingV1<'_>,
    current_evaluator_profile: &PinnedEd25519QualificationVerifierV1,
    decision_need: &DecisionNeedV1,
    envelopes: &[SourceLineageEnvelopeInputV1<'_>],
    expected_lineage_profile_sha256: &str,
    lineage_profile: &PinnedEd25519QualificationVerifierV1,
) -> Result<VerifiedSourceLineageSetV1, SourceLineageRejectV1> {
    if qualification_binding.now_unix != sufficiency_binding.now_unix {
        return Err(SourceLineageRejectV1::BindingMismatch);
    }
    sufficiency.validate_at_use(
        qualified,
        qualification_binding,
        current_qualification_verifier,
        sufficiency_binding,
        current_evaluator_profile,
    ).map_err(SourceLineageRejectV1::Sufficiency)?;
    let now_unix = sufficiency_binding.now_unix;
    if sufficiency_binding.decision_need.digest()
        .map_err(SourceLineageRejectV1::Sufficiency)? != decision_need.digest()
            .map_err(SourceLineageRejectV1::Sufficiency)?
    {
        return Err(SourceLineageRejectV1::BindingMismatch);
    }
    let receipt = sufficiency.receipt();
    let need_digest = decision_need.digest()
        .map_err(SourceLineageRejectV1::Sufficiency)?;
    if lineage_profile.verifier_id() != SOURCE_LINEAGE_VERIFIER_ID_V1 {
        return Err(SourceLineageRejectV1::WrongVerifierRole);
    }
    if !valid_sha256(expected_lineage_profile_sha256)
        || lineage_profile.profile_sha256() != expected_lineage_profile_sha256
    {
        return Err(SourceLineageRejectV1::ProfileMismatch);
    }
    if lineage_profile.profile_sha256() == qualified.qualification_verifier_profile_sha256()
        || lineage_profile.profile_sha256() == receipt.issuer_profile_sha256.as_str()
    {
        return Err(SourceLineageRejectV1::SharedAuthority);
    }
    if qualified.qualified_candidate_set_sha256() != receipt.qualified_evidence_set_sha256.as_str()
        || qualified.run_id() != receipt.run_id.as_str()
        || qualified.query_invocation_id() != receipt.query_invocation_id.as_str()
        || need_digest != receipt.decision_need_sha256
        || qualified.decision_need_sha256() != need_digest.as_str()
    {
        return Err(SourceLineageRejectV1::BindingMismatch);
    }

    let expected = expected_lineage_pairs(receipt, decision_need)?;
    if expected.len() > MAX_LINEAGE_ENVELOPES_V1
        || envelopes.len() > MAX_LINEAGE_ENVELOPES_V1
    {
        return Err(SourceLineageRejectV1::InvalidReceipt);
    }
    if expected.len() != envelopes.len() {
        return Err(SourceLineageRejectV1::MissingReceipt);
    }

    let mut verified = Vec::with_capacity(envelopes.len());
    let mut seen_pairs = BTreeSet::new();
    for input in envelopes {
        if !seen_pairs.insert((input.requirement_id, input.root_family_sha256)) {
            return Err(SourceLineageRejectV1::DuplicateReceipt);
        }
        let expected_item = expected.iter().find(|(requirement_id, root, _)| {
            requirement_id.as_str() == input.requirement_id
                && root.as_str() == input.root_family_sha256
        }).ok_or(SourceLineageRejectV1::UnexpectedReceipt)?;
        if sha256_hex(input.envelope_json.as_bytes()) != expected_item.2 {
            return Err(SourceLineageRejectV1::PayloadBindingMismatch);
        }
        let need_requirement = decision_need.requirements.iter()
            .find(|item| item.requirement_id == input.requirement_id)
            .ok_or(SourceLineageRejectV1::BindingMismatch)?;
        let binding = ExpectedLineageBindingV1 {
            run_id: &receipt.run_id,
            query_invocation_id: &receipt.query_invocation_id,
            qualified_evidence_set_sha256: &receipt.qualified_evidence_set_sha256,
            decision_need_sha256: &need_digest,
            requirement_id: &need_requirement.requirement_id,
            evidence_class_sha256: &need_requirement.evidence_class_sha256,
            root_family_sha256: input.root_family_sha256,
            expected_profile_sha256: expected_lineage_profile_sha256,
        };
        let item = verify_one_source_lineage_receipt_v1(
            input.envelope_json,
            &binding,
            lineage_profile,
            now_unix,
        )?;
        if item.receipt.key_id == qualified.qualification_signer_key_id()
            || item.receipt.key_id == receipt.key_id
        {
            return Err(SourceLineageRejectV1::SharedAuthority);
        }
        verified.push(item);
    }
    verified.sort_by(|left, right| {
        (&left.receipt.requirement_id, &left.receipt.root_family_sha256)
            .cmp(&(&right.receipt.requirement_id, &right.receipt.root_family_sha256))
    });

    Ok(VerifiedSourceLineageSetV1 {
        receipts: verified,
        sufficiency_receipt_sha256: sufficiency.receipt_sha256(),
        qualified_evidence_set_sha256: receipt.qualified_evidence_set_sha256.clone(),
        decision_need_sha256: need_digest,
        lineage_profile_sha256: expected_lineage_profile_sha256.to_owned(),
    })
}

fn sufficiency_receipt_profile(sufficiency: &VerifiedDecisionSufficiencyV1) -> &str {
    sufficiency.receipt().issuer_profile_sha256.as_str()
}

fn expected_lineage_pairs(
    sufficiency: &DecisionSufficiencyReceiptV1,
    decision_need: &DecisionNeedV1,
) -> Result<Vec<(String, String, String)>, SourceLineageRejectV1> {
    decision_need.validate().map_err(SourceLineageRejectV1::Sufficiency)?;
    let mut expected = Vec::new();
    if sufficiency.requirements.len() != decision_need.requirements.len() {
        return Err(SourceLineageRejectV1::BindingMismatch);
    }
    for (claim, need) in sufficiency.requirements.iter().zip(decision_need.requirements.iter()) {
        if claim.requirement_id != need.requirement_id
            || claim.evidence_class_sha256 != need.evidence_class_sha256
        {
            return Err(SourceLineageRejectV1::BindingMismatch);
        }
        if claim.status == crate::decision_sufficiency::DecisionRequirementStatusV1::Met
            && claim.independent_root_families.len()
                < need.minimum_independent_root_families as usize
        {
            return Err(SourceLineageRejectV1::Sufficiency(
                DecisionSufficiencyRejectV1::InsufficientIndependentRootFamilies,
            ));
        }
        for family in &claim.independent_root_families {
            expected.push((
                claim.requirement_id.clone(),
                family.root_family_sha256.clone(),
                family.provenance_receipt_sha256.clone(),
            ));
        }
    }
    expected.sort();
    if expected.windows(2).any(|pair| pair[0].0 == pair[1].0 && pair[0].1 == pair[1].1) {
        return Err(SourceLineageRejectV1::DuplicateReceipt);
    }
    let digest_set = expected.iter().map(|item| item.2.as_str()).collect::<BTreeSet<_>>();
    if digest_set.len() != expected.len() {
        return Err(SourceLineageRejectV1::DuplicateReceipt);
    }
    Ok(expected)
}

fn verify_one_source_lineage_receipt_v1(
    envelope_json: &str,
    binding: &ExpectedLineageBindingV1<'_>,
    profile: &PinnedEd25519QualificationVerifierV1,
    now_unix: u64,
) -> Result<VerifiedSourceLineageReceiptV1, SourceLineageRejectV1> {
    if profile.verifier_id() != SOURCE_LINEAGE_VERIFIER_ID_V1 {
        return Err(SourceLineageRejectV1::WrongVerifierRole);
    }
    if profile.profile_sha256() != binding.expected_profile_sha256 {
        return Err(SourceLineageRejectV1::ProfileMismatch);
    }
    if envelope_json.len() > MAX_LINEAGE_ENVELOPE_BYTES_V1 {
        return Err(SourceLineageRejectV1::InvalidEnvelope);
    }
    let envelope = parse_dsse_envelope_v1(envelope_json, DSSE_PAYLOAD_TYPE_V1)
        .map_err(|_| SourceLineageRejectV1::InvalidEnvelope)?;
    if envelope.payload.len() > MAX_LINEAGE_PAYLOAD_BYTES_V1 {
        return Err(SourceLineageRejectV1::InvalidEnvelope);
    }
    let statement_value = parse_strict_json_bounded(&envelope.payload, MAX_LINEAGE_PAYLOAD_BYTES_V1)
        .map_err(|_| SourceLineageRejectV1::InvalidJson)?;
    let statement = statement_value.as_object().ok_or(SourceLineageRejectV1::InvalidJson)?;
    if statement.get("_type").and_then(Value::as_str) != Some(IN_TOTO_STATEMENT_TYPE_V1)
        || statement.get("predicateType").and_then(Value::as_str)
            != Some(MACE_SOURCE_LINEAGE_PREDICATE_TYPE_V1)
    {
        return Err(SourceLineageRejectV1::WrongPredicateType);
    }
    let subject_list = statement.get("subject").and_then(Value::as_array)
        .ok_or(SourceLineageRejectV1::InvalidJson)?;
    if subject_list.len() != 1 {
        return Err(SourceLineageRejectV1::SubjectMismatch);
    }
    let subject = subject_list[0].as_object().ok_or(SourceLineageRejectV1::InvalidJson)?;
    let expected_subject_name = format!(
        "urn:luminous:mace:source-root-family:v1:{}",
        binding.root_family_sha256
    );
    if subject.get("name").and_then(Value::as_str) != Some(expected_subject_name.as_str())
        || subject.get("digest").and_then(Value::as_object)
            .and_then(|digest| digest.get("sha256"))
            .and_then(Value::as_str) != Some(binding.root_family_sha256)
    {
        return Err(SourceLineageRejectV1::SubjectMismatch);
    }

    let predicate = statement.get("predicate").ok_or(SourceLineageRejectV1::InvalidJson)?;
    let receipt = parse_lineage_predicate(predicate)?;
    validate_lineage_binding_expected(&receipt, binding)?;
    if envelope.key_id_hint.as_ref().is_some_and(|hint| hint != &receipt.key_id) {
        return Err(SourceLineageRejectV1::PayloadBindingMismatch);
    }
    if now_unix < receipt.issued_at_unix {
        return Err(SourceLineageRejectV1::NotYetValid);
    }
    if now_unix >= receipt.expires_at_unix {
        return Err(SourceLineageRejectV1::Expired);
    }
    profile.verify_message(
        &receipt.issuer_profile_sha256,
        &receipt.key_id,
        &envelope.pae,
        &envelope.signature,
        receipt.issued_at_unix,
        now_unix,
    ).map_err(map_signature_error)?;

    Ok(VerifiedSourceLineageReceiptV1 {
        envelope_sha256: sha256_hex(envelope_json.as_bytes()),
        receipt,
        pae: envelope.pae,
        signature: envelope.signature,
    })
}

fn parse_lineage_predicate(value: &Value) -> Result<SourceLineageReceiptV1, SourceLineageRejectV1> {
    let object = value.as_object().ok_or(SourceLineageRejectV1::InvalidJson)?;
    const FIELDS: &[&str] = &[
        "schema_version", "run_id", "query_invocation_id",
        "qualified_evidence_set_sha256", "decision_need_sha256",
        "requirement_id", "evidence_class_sha256", "root_family_sha256",
        "source_graph_sha256", "lineage_closure_sha256", "source_node_count",
        "dependency_edge_count", "unresolved_ancestor_count", "status",
        "issued_at_unix", "expires_at_unix", "issuer_profile_sha256", "key_id",
    ];
    if object.len() != FIELDS.len()
        || object.keys().any(|key| !FIELDS.contains(&key.as_str()))
        || object.get("schema_version").and_then(Value::as_str)
            != Some("mace_source_lineage_receipt_v1")
    {
        return Err(SourceLineageRejectV1::InvalidJson);
    }
    let string = |field: &str| -> Result<String, SourceLineageRejectV1> {
        object.get(field).and_then(Value::as_str).map(str::to_owned)
            .ok_or(SourceLineageRejectV1::InvalidJson)
    };
    let integer = |field: &str| -> Result<u64, SourceLineageRejectV1> {
        object.get(field).and_then(Value::as_u64)
            .ok_or(SourceLineageRejectV1::InvalidJson)
    };
    let receipt = SourceLineageReceiptV1 {
        run_id: string("run_id")?,
        query_invocation_id: string("query_invocation_id")?,
        qualified_evidence_set_sha256: string("qualified_evidence_set_sha256")?,
        decision_need_sha256: string("decision_need_sha256")?,
        requirement_id: string("requirement_id")?,
        evidence_class_sha256: string("evidence_class_sha256")?,
        root_family_sha256: string("root_family_sha256")?,
        source_graph_sha256: string("source_graph_sha256")?,
        lineage_closure_sha256: string("lineage_closure_sha256")?,
        source_node_count: integer("source_node_count")?,
        dependency_edge_count: integer("dependency_edge_count")?,
        unresolved_ancestor_count: integer("unresolved_ancestor_count")?,
        status: string("status")?,
        issued_at_unix: integer("issued_at_unix")?,
        expires_at_unix: integer("expires_at_unix")?,
        issuer_profile_sha256: string("issuer_profile_sha256")?,
        key_id: string("key_id")?,
    };
    if !valid_atom(&receipt.run_id)
        || receipt.run_id.len() > 256
        || !valid_atom(&receipt.query_invocation_id)
        || receipt.query_invocation_id.len() > 256
        || !valid_atom(&receipt.requirement_id)
        || receipt.requirement_id.len() > 256
        || receipt.issued_at_unix >= receipt.expires_at_unix
        || receipt.source_node_count == 0
        || receipt.source_node_count > 10_000_000
        || receipt.dependency_edge_count > 100_000_000
        || receipt.unresolved_ancestor_count != 0
    {
        return Err(SourceLineageRejectV1::InvalidReceipt);
    }
    for digest in [
        receipt.qualified_evidence_set_sha256.as_str(),
        receipt.decision_need_sha256.as_str(),
        receipt.evidence_class_sha256.as_str(),
        receipt.root_family_sha256.as_str(),
        receipt.source_graph_sha256.as_str(),
        receipt.lineage_closure_sha256.as_str(),
        receipt.issuer_profile_sha256.as_str(),
        receipt.key_id.as_str(),
    ] {
        if !valid_sha256(digest) {
            return Err(SourceLineageRejectV1::InvalidDigest);
        }
    }
    if receipt.status != "independent" {
        return Err(SourceLineageRejectV1::NotIndependent);
    }
    Ok(receipt)
}

fn validate_lineage_binding_expected(
    receipt: &SourceLineageReceiptV1,
    binding: &ExpectedLineageBindingV1<'_>,
) -> Result<(), SourceLineageRejectV1> {
    if receipt.run_id != binding.run_id
        || receipt.query_invocation_id != binding.query_invocation_id
        || receipt.qualified_evidence_set_sha256 != binding.qualified_evidence_set_sha256
        || receipt.decision_need_sha256 != binding.decision_need_sha256
        || receipt.requirement_id != binding.requirement_id
        || receipt.evidence_class_sha256 != binding.evidence_class_sha256
        || receipt.root_family_sha256 != binding.root_family_sha256
        || receipt.issuer_profile_sha256 != binding.expected_profile_sha256
    {
        return Err(SourceLineageRejectV1::BindingMismatch);
    }
    Ok(())
}

fn validate_lineage_binding_at_use(
    receipt: &SourceLineageReceiptV1,
    qualified: &QualifiedCandidateEvidenceBatchV1,
    sufficiency: &DecisionSufficiencyReceiptV1,
    decision_need: &DecisionNeedV1,
) -> Result<(), SourceLineageRejectV1> {
    let need_digest = decision_need.digest().map_err(SourceLineageRejectV1::Sufficiency)?;
    let requirement = decision_need.requirements.iter()
        .find(|item| item.requirement_id == receipt.requirement_id)
        .ok_or(SourceLineageRejectV1::BindingMismatch)?;
    if receipt.run_id != qualified.run_id()
        || receipt.query_invocation_id != qualified.query_invocation_id()
        || receipt.qualified_evidence_set_sha256.as_str() != qualified.qualified_candidate_set_sha256()
        || receipt.qualified_evidence_set_sha256 != sufficiency.qualified_evidence_set_sha256
        || receipt.decision_need_sha256.as_str() != need_digest.as_str()
        || receipt.decision_need_sha256 != sufficiency.decision_need_sha256
        || receipt.evidence_class_sha256 != requirement.evidence_class_sha256
        || !sufficiency.requirements.iter().any(|claim| {
            claim.requirement_id == receipt.requirement_id
                && claim.evidence_class_sha256 == receipt.evidence_class_sha256
                && claim.independent_root_families.iter().any(|family| {
                    family.root_family_sha256 == receipt.root_family_sha256
                })
        })
    {
        return Err(SourceLineageRejectV1::BindingMismatch);
    }
    Ok(())
}

fn map_signature_error(
    error: crate::evidence_qualification_gate::EvidenceQualificationRejectV1,
) -> SourceLineageRejectV1 {
    use crate::evidence_qualification_gate::EvidenceQualificationRejectV1 as E;
    match error {
        E::RevokedKey | E::UnknownKey => SourceLineageRejectV1::UnknownOrRevokedSigner,
        E::TrustProfileDigestMismatch => SourceLineageRejectV1::ProfileMismatch,
        E::KeyNotYetValid => SourceLineageRejectV1::NotYetValid,
        E::KeyExpired => SourceLineageRejectV1::Expired,
        _ => SourceLineageRejectV1::InvalidSignature,
    }
}

fn valid_atom(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && !value.chars().any(char::is_control)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
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

    use crate::qualification_dsse::dsse_pae_v1;
    use crate::qualification_trust::{
        QualificationKeyStatusV1, QualificationTrustKeyV1, QualificationTrustProfileV1,
        qualification_key_id_v1,
    };

    fn resolver_profile(signing_key: &SigningKey, verifier_id: &str)
        -> (PinnedEd25519QualificationVerifierV1, String)
    {
        let public_key = signing_key.verifying_key().to_bytes();
        let key = QualificationTrustKeyV1 {
            key_id: qualification_key_id_v1(&public_key),
            public_key,
            status: QualificationKeyStatusV1::Active,
            not_before_unix: 10,
            not_after_unix: 1_000,
        };
        let profile = QualificationTrustProfileV1 {
            schema_version: 1,
            verifier_id: verifier_id.to_owned(),
            profile_version: 1,
            keys: vec![key],
        };
        let digest = profile.profile_sha256().unwrap();
        let verifier = PinnedEd25519QualificationVerifierV1::from_pinned_profile(
            &profile, &digest, 1,
        ).unwrap();
        (verifier, digest)
    }

    fn signed_lineage_envelope(
        binding: &ExpectedLineageBindingV1<'_>,
        signing_key: &SigningKey,
    ) -> String {
        let key_id = qualification_key_id_v1(&signing_key.verifying_key().to_bytes());
        let predicate = serde_json::json!({
            "schema_version": "mace_source_lineage_receipt_v1",
            "run_id": binding.run_id,
            "query_invocation_id": binding.query_invocation_id,
            "qualified_evidence_set_sha256": binding.qualified_evidence_set_sha256,
            "decision_need_sha256": binding.decision_need_sha256,
            "requirement_id": binding.requirement_id,
            "evidence_class_sha256": binding.evidence_class_sha256,
            "root_family_sha256": binding.root_family_sha256,
            "source_graph_sha256": "a".repeat(64),
            "lineage_closure_sha256": "b".repeat(64),
            "source_node_count": 2,
            "dependency_edge_count": 1,
            "unresolved_ancestor_count": 0,
            "status": "independent",
            "issued_at_unix": 100,
            "expires_at_unix": 500,
            "issuer_profile_sha256": binding.expected_profile_sha256,
            "key_id": key_id,
        });
        let subject_name = format!(
            "urn:luminous:mace:source-root-family:v1:{}",
            binding.root_family_sha256
        );
        let statement = serde_json::json!({
            "_type": IN_TOTO_STATEMENT_TYPE_V1,
            "subject": [{
                "name": subject_name,
                "digest": {"sha256": binding.root_family_sha256}
            }],
            "predicateType": MACE_SOURCE_LINEAGE_PREDICATE_TYPE_V1,
            "predicate": predicate
        });
        let payload = serde_json::to_vec(&statement).unwrap();
        let pae = dsse_pae_v1(DSSE_PAYLOAD_TYPE_V1, &payload);
        let signature = signing_key.sign(&pae).to_bytes();
        let envelope = serde_json::json!({
            "payloadType": DSSE_PAYLOAD_TYPE_V1,
            "payload": base64::Engine::encode(&base64::engine::general_purpose::STANDARD, payload),
            "signatures": [{
                "keyid": key_id,
                "sig": base64::Engine::encode(&base64::engine::general_purpose::STANDARD, signature)
            }]
        });
        serde_json::to_string(&envelope).unwrap()
    }

    fn binding<'a>(profile_digest: &'a str) -> ExpectedLineageBindingV1<'a> {
        ExpectedLineageBindingV1 {
            run_id: "run-1",
            query_invocation_id: "invocation-1",
            qualified_evidence_set_sha256:
                "1111111111111111111111111111111111111111111111111111111111111111",
            decision_need_sha256:
                "2222222222222222222222222222222222222222222222222222222222222222",
            requirement_id: "minimum-independent-evidence",
            evidence_class_sha256:
                "3333333333333333333333333333333333333333333333333333333333333333",
            root_family_sha256:
                "4444444444444444444444444444444444444444444444444444444444444444",
            expected_profile_sha256: profile_digest,
        }
    }

    #[test]
    fn resolver_role_and_dsse_pae_are_verified() {
        let signing_key = SigningKey::from_bytes(&[99; 32]);
        let (profile, digest) = resolver_profile(
            &signing_key, SOURCE_LINEAGE_VERIFIER_ID_V1,
        );
        let expected = binding(&digest);
        let envelope = signed_lineage_envelope(&expected, &signing_key);
        let verified = verify_one_source_lineage_receipt_v1(
            &envelope, &expected, &profile, 200,
        ).expect("independent lineage receipt");
        assert_eq!(verified.receipt.root_family_sha256, expected.root_family_sha256);
        assert_eq!(verified.envelope_sha256, sha256_hex(envelope.as_bytes()));
    }

    #[test]
    fn wrong_resolver_role_is_rejected() {
        let signing_key = SigningKey::from_bytes(&[100; 32]);
        let (profile, digest) = resolver_profile(&signing_key, "mace-other-role");
        let expected = binding(&digest);
        let envelope = signed_lineage_envelope(&expected, &signing_key);
        assert_eq!(
            verify_one_source_lineage_receipt_v1(
                &envelope, &expected, &profile, 200,
            ),
            Err(SourceLineageRejectV1::WrongVerifierRole)
        );
    }

    #[test]
    fn changed_envelope_bytes_change_the_provenance_commitment() {
        let signing_key = SigningKey::from_bytes(&[101; 32]);
        let (profile, digest) = resolver_profile(
            &signing_key, SOURCE_LINEAGE_VERIFIER_ID_V1,
        );
        let expected = binding(&digest);
        let envelope = signed_lineage_envelope(&expected, &signing_key);
        let mut altered = envelope.clone();
        altered.push(' ');
        assert_ne!(sha256_hex(envelope.as_bytes()), sha256_hex(altered.as_bytes()));
        assert!(verify_one_source_lineage_receipt_v1(
            &altered, &expected, &profile, 200
        ).is_ok());
    }
}
