// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Consumer-side exchange boundary for independently issued evidence
//! qualification receipts.
//!
//! This module does not implement the qualifier. It packages the exact
//! candidate batch and use binding for an injected issuer transport, then
//! verifies the returned receipt against an independently pinned verifier
//! profile and projects only the qualified subset. Issuer failure, timeout,
//! malformed output, signature failure or binding mismatch returns an error;
//! callers must not fall back to the original candidate batch.

use std::sync::Mutex;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::candidate_evidence::{
    CandidateEvidenceBatchV1, CandidateEvidenceReject, encode_candidate_evidence_jsonl,
    parse_candidate_evidence_jsonl,
};
use crate::evidence_qualification_gate::{
    EvidenceQualificationBindingV1, EvidenceQualificationRejectV1,
    VerifiedEvidenceQualificationV1,
};
use crate::qualification_dsse::{
    DsseQualificationSignatureVerifierV1,
    encode_dsse_envelope_v1,
    dsse_pae_v1,
    in_toto_qualification_payload_v1,
};
use crate::qualification_trust::{
    PinnedEd25519QualificationVerifierV1,
};
use crate::qualified_candidate_evidence::{
    CandidateEvidenceBatchExpectationV1, CandidateEvidenceProjectionRejectV1,
    CandidateEvidenceProjectionV1, project_qualified_candidate_evidence_v1,
};

pub const QUALIFICATION_REQUEST_SCHEMA_V1: &str =
    "mace_evidence_qualification_request_v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QualificationUseContextV1 {
    /// Opaque invocation identity supplied by the benchmark/consumer.
    pub query_invocation_id: String,
    pub domain_id: String,
    /// Digest of the effective tenant/consumer authorization scope.
    pub subject_scope_sha256: String,
    pub environment_revision_sha256: String,
    /// Digest of the exact consumer tool profile; separate from the candidate
    /// backend revision in the candidate batch.
    pub tool_profile_sha256: String,
    pub decision_need_sha256: String,
    pub purpose_sha256: String,
    pub qualification_policy_sha256: String,
    /// Digest of the authoritative lifecycle snapshot used by the qualifier.
    pub lifecycle_snapshot_sha256: String,
    /// Must equal the digest of the consumer's separately pinned trust profile.
    pub verifier_profile_sha256: String,
}

/// Owned, transport-neutral qualification request. The canonical candidate
/// JSONL is included so the issuer can independently recompute the batch
/// commitment and resolve source payloads; opaque payload_ref values are not
/// interpreted as local paths by this module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceQualificationRequestBindingV1 {
    pub run_id: String,
    pub query_invocation_id: String,
    pub domain_id: String,
    pub subject_scope_sha256: String,
    pub environment_revision_sha256: String,
    pub tool_profile_sha256: String,
    pub query_sha256: String,
    pub candidate_set_sha256: String,
    /// Sorted, complete candidate IDs, including records later rejected or
    /// marked indeterminate by the independent issuer.
    pub candidate_evidence_ids: Vec<String>,
    pub decision_need_sha256: String,
    pub purpose_sha256: String,
    pub qualification_policy_sha256: String,
    pub lifecycle_snapshot_sha256: String,
    pub verifier_profile_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceQualificationRequestV1 {
    pub schema_version: String,
    /// Canonical JSONL for the complete unqualified candidate batch.
    pub candidate_batch_jsonl: String,
    pub binding: EvidenceQualificationRequestBindingV1,
    /// Domain-separated commitment over the schema, canonical candidate bytes,
    /// and every binding field. This is a request identity, not an authority
    /// signature and not a substitute for issuer-side recomputation.
    pub request_sha256: String,
}

impl EvidenceQualificationRequestV1 {
    /// Strictly parse untrusted wire JSON without allowing duplicate object
    /// keys to be collapsed by serde_json::Value before request validation.
    pub fn from_json_str(input: &str) -> Result<Self, IndependentQualificationExchangeRejectV1> {
        let value = crate::qualification_dsse::parse_strict_json_bounded(
            input.as_bytes(),
            18 * 1024 * 1024,
        )
        .map_err(|_| IndependentQualificationExchangeRejectV1::InvalidRequestJson)?;
        Self::from_json_value(&value)
    }

    pub fn to_json_value(&self) -> Value {
        json!({
            "schema_version": self.schema_version,
            "candidate_batch_jsonl": self.candidate_batch_jsonl,
            "binding": {
                "run_id": self.binding.run_id,
                "query_invocation_id": self.binding.query_invocation_id,
                "domain_id": self.binding.domain_id,
                "subject_scope_sha256": self.binding.subject_scope_sha256,
                "environment_revision_sha256": self.binding.environment_revision_sha256,
                "tool_profile_sha256": self.binding.tool_profile_sha256,
                "query_sha256": self.binding.query_sha256,
                "candidate_set_sha256": self.binding.candidate_set_sha256,
                "candidate_evidence_ids": self.binding.candidate_evidence_ids,
                "decision_need_sha256": self.binding.decision_need_sha256,
                "purpose_sha256": self.binding.purpose_sha256,
                "qualification_policy_sha256": self.binding.qualification_policy_sha256,
                "lifecycle_snapshot_sha256": self.binding.lifecycle_snapshot_sha256,
                "verifier_profile_sha256": self.binding.verifier_profile_sha256,
            },
            "request_sha256": self.request_sha256,
        })
    }

    /// Strictly decode a request at the independent issuer boundary.
    ///
    /// This validates the complete JSON shape, canonical candidate JSONL,
    /// exact candidate/binding correspondence, sorted complete ID set, and
    /// request digest. It does not authenticate the caller or establish that
    /// the candidate sources are genuine; the issuer must independently
    /// resolve and verify those sources before signing a receipt.
    pub fn from_json_value(
        value: &Value,
    ) -> Result<Self, IndependentQualificationExchangeRejectV1> {
        let object = value.as_object()
            .ok_or(IndependentQualificationExchangeRejectV1::InvalidRequestJson)?;
        if object.len() != 4
            || object.get("schema_version").and_then(Value::as_str)
                != Some(QUALIFICATION_REQUEST_SCHEMA_V1)
        {
            return Err(IndependentQualificationExchangeRejectV1::InvalidRequestJson);
        }
        let required_string = |field: &str| -> Result<String, IndependentQualificationExchangeRejectV1> {
            object.get(field).and_then(Value::as_str).map(str::to_owned)
                .ok_or(IndependentQualificationExchangeRejectV1::InvalidRequestJson)
        };
        let candidate_batch_jsonl = required_string("candidate_batch_jsonl")?;
        let request_sha256 = required_string("request_sha256")?;
        if !valid_sha256(&request_sha256) {
            return Err(IndependentQualificationExchangeRejectV1::InvalidRequestJson);
        }

        let binding_object = object.get("binding").and_then(Value::as_object)
            .ok_or(IndependentQualificationExchangeRejectV1::InvalidRequestJson)?;
        if binding_object.len() != 14 {
            return Err(IndependentQualificationExchangeRejectV1::InvalidRequestJson);
        }
        let binding_string = |field: &str| -> Result<String, IndependentQualificationExchangeRejectV1> {
            binding_object.get(field).and_then(Value::as_str).map(str::to_owned)
                .ok_or(IndependentQualificationExchangeRejectV1::InvalidRequestJson)
        };
        let candidate_evidence_ids = binding_object
            .get("candidate_evidence_ids")
            .and_then(Value::as_array)
            .ok_or(IndependentQualificationExchangeRejectV1::InvalidRequestJson)?
            .iter()
            .map(|value| value.as_str().map(str::to_owned)
                .ok_or(IndependentQualificationExchangeRejectV1::InvalidRequestJson))
            .collect::<Result<Vec<_>, _>>()?;
        let binding = EvidenceQualificationRequestBindingV1 {
            run_id: binding_string("run_id")?,
            query_invocation_id: binding_string("query_invocation_id")?,
            domain_id: binding_string("domain_id")?,
            subject_scope_sha256: binding_string("subject_scope_sha256")?,
            environment_revision_sha256: binding_string("environment_revision_sha256")?,
            tool_profile_sha256: binding_string("tool_profile_sha256")?,
            query_sha256: binding_string("query_sha256")?,
            candidate_set_sha256: binding_string("candidate_set_sha256")?,
            candidate_evidence_ids,
            decision_need_sha256: binding_string("decision_need_sha256")?,
            purpose_sha256: binding_string("purpose_sha256")?,
            qualification_policy_sha256: binding_string("qualification_policy_sha256")?,
            lifecycle_snapshot_sha256: binding_string("lifecycle_snapshot_sha256")?,
            verifier_profile_sha256: binding_string("verifier_profile_sha256")?,
        };

        let use_context = QualificationUseContextV1 {
            query_invocation_id: binding.query_invocation_id.clone(),
            domain_id: binding.domain_id.clone(),
            subject_scope_sha256: binding.subject_scope_sha256.clone(),
            environment_revision_sha256: binding.environment_revision_sha256.clone(),
            tool_profile_sha256: binding.tool_profile_sha256.clone(),
            decision_need_sha256: binding.decision_need_sha256.clone(),
            purpose_sha256: binding.purpose_sha256.clone(),
            qualification_policy_sha256: binding.qualification_policy_sha256.clone(),
            lifecycle_snapshot_sha256: binding.lifecycle_snapshot_sha256.clone(),
            verifier_profile_sha256: binding.verifier_profile_sha256.clone(),
        };
        validate_use_context(&use_context)?;
        if !valid_atom(&binding.run_id, 256)
            || !valid_sha256(&binding.query_sha256)
            || !valid_sha256(&binding.candidate_set_sha256)
            || binding.candidate_evidence_ids.is_empty()
            || binding.candidate_evidence_ids.len() > 10_000
            || binding.candidate_evidence_ids.iter().any(|id| !valid_atom(id, 256))
            || binding.candidate_evidence_ids.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(IndependentQualificationExchangeRejectV1::CandidateIdsNotCanonical);
        }

        let candidate_batch = parse_candidate_evidence_jsonl(&candidate_batch_jsonl)
            .map_err(IndependentQualificationExchangeRejectV1::CandidateBatch)?;
        if candidate_batch.to_jsonl()
            .map_err(IndependentQualificationExchangeRejectV1::CandidateBatch)?
            != candidate_batch_jsonl
        {
            return Err(IndependentQualificationExchangeRejectV1::CandidateBatch(
                CandidateEvidenceReject::InvalidDigest,
            ));
        }
        let mut actual_ids = candidate_batch.evidence.iter()
            .map(|item| item.evidence_id.clone())
            .collect::<Vec<_>>();
        actual_ids.sort();
        if candidate_batch.run_id != binding.run_id
            || candidate_batch.query_sha256 != binding.query_sha256
            || candidate_batch.candidate_set_sha256 != binding.candidate_set_sha256
            || actual_ids != binding.candidate_evidence_ids
        {
            return Err(IndependentQualificationExchangeRejectV1::CandidateBatchBindingMismatch);
        }

        let calculated = request_digest(&candidate_batch_jsonl, &binding);
        if calculated != request_sha256 {
            return Err(IndependentQualificationExchangeRejectV1::RequestDigestMismatch);
        }

        Ok(Self {
            schema_version: QUALIFICATION_REQUEST_SCHEMA_V1.to_owned(),
            candidate_batch_jsonl,
            binding,
            request_sha256,
        })
    }
}

/// A transport implementation should call a separately governed qualification
/// service. It must not sign locally from the candidate backend's own store or
/// credentials. Timeouts and ambiguous outcomes must map to Unavailable.
pub trait IndependentEvidenceQualificationIssuerV1 {
    /// Return the complete DSSE JSON envelope from an independently governed
    /// issuer. The consumer validates its in-toto Statement, signature and
    /// complete MACE use binding; transport success is not qualification.
    fn issue(
        &self,
        request: &EvidenceQualificationRequestV1,
    ) -> Result<String, QualificationIssuerErrorV1>;
}

/// Result of the consumer-side exchange. It contains only a qualified
/// projection or explicit abstention, never the original raw candidate batch.
/// The exact signed envelope is retained so later reads can reconstruct a
/// verifier from the current pinned trust profile.
#[derive(Clone, Debug, PartialEq)]
pub struct IndependentQualificationExchangeResultV1 {
    projection: CandidateEvidenceProjectionV1,
    receipt_envelope_json: String,
}

impl IndependentQualificationExchangeResultV1 {
    pub fn projection(&self) -> &CandidateEvidenceProjectionV1 {
        &self.projection
    }

    pub fn receipt_envelope_json(&self) -> &str {
        &self.receipt_envelope_json
    }

    /// Rebuild a verifier using the current pinned trust profile before each
    /// subsequent use. A cached verifier can miss a later key/profile revocation.
    pub fn verifier_for_use<'a>(
        &self,
        pinned: &'a PinnedEd25519QualificationVerifierV1,
    ) -> Result<DsseQualificationSignatureVerifierV1<'a>, EvidenceQualificationRejectV1> {
        DsseQualificationSignatureVerifierV1::from_envelope_json(
            &self.receipt_envelope_json,
            pinned,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QualificationIssuerErrorV1 {
    Unavailable,
    Rejected,
    MalformedResponse,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QualificationClockErrorV1 {
    Unavailable,
    Untrusted,
}

/// Trusted consumer clock. Implementations must use a protected, monotonic
/// time source; wall-clock values supplied by requests or candidate storage
/// are not an acceptable substitute.
pub trait QualificationTrustedClockV1 {
    fn now_unix(&self) -> Result<u64, QualificationClockErrorV1>;
}

/// Process-local rollback guard for a trusted clock source.
///
/// This enforces non-decreasing readings during one process lifetime. It does
/// not survive process restart or protect against rollback of the underlying
/// clock source; production must still use an externally protected monotonic
/// time authority or persistent anti-rollback witness across restarts.
pub struct MonotonicQualificationClockGuardV1<C> {
    source: C,
    last_unix: Mutex<Option<u64>>,
}

impl<C> MonotonicQualificationClockGuardV1<C> {
    pub fn new(source: C) -> Self {
        Self {
            source,
            last_unix: Mutex::new(None),
        }
    }
}

impl<C: QualificationTrustedClockV1> QualificationTrustedClockV1
    for MonotonicQualificationClockGuardV1<C>
{
    fn now_unix(&self) -> Result<u64, QualificationClockErrorV1> {
        let now = self.source.now_unix()?;
        let mut last = self.last_unix
            .lock()
            .map_err(|_| QualificationClockErrorV1::Untrusted)?;
        if last.is_some_and(|previous| now < previous) {
            return Err(QualificationClockErrorV1::Untrusted);
        }
        *last = Some(now);
        Ok(now)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndependentQualificationExchangeRejectV1 {
    InvalidUseContext,
    InvalidRequestJson,
    RequestDigestMismatch,
    CandidateBatch(CandidateEvidenceReject),
    CandidateBatchBindingMismatch,
    CandidateIdentityMismatch,
    CandidateIdsNotCanonical,
    IssuerUnavailable,
    IssuerRejected,
    IssuerMalformedResponse,
    TrustedClockUnavailable,
    TrustedClockUntrusted,
    Qualification(EvidenceQualificationRejectV1),
    Projection(CandidateEvidenceProjectionRejectV1),
}

fn valid_atom(value: &str, maximum_bytes: usize) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= maximum_bytes
        && !value.chars().any(char::is_control)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn request_digest(
    candidate_batch_jsonl: &str,
    binding: &EvidenceQualificationRequestBindingV1,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:evidence-qualification-request:v1\0");
    hash_field(&mut hasher, QUALIFICATION_REQUEST_SCHEMA_V1);
    hash_field(&mut hasher, candidate_batch_jsonl);
    for value in [
        binding.run_id.as_str(),
        binding.query_invocation_id.as_str(),
        binding.domain_id.as_str(),
        binding.subject_scope_sha256.as_str(),
        binding.environment_revision_sha256.as_str(),
        binding.tool_profile_sha256.as_str(),
        binding.query_sha256.as_str(),
        binding.candidate_set_sha256.as_str(),
    ] {
        hash_field(&mut hasher, value);
    }
    hasher.update((binding.candidate_evidence_ids.len() as u64).to_be_bytes());
    for evidence_id in &binding.candidate_evidence_ids {
        hash_field(&mut hasher, evidence_id);
    }
    for value in [
        binding.decision_need_sha256.as_str(),
        binding.purpose_sha256.as_str(),
        binding.qualification_policy_sha256.as_str(),
        binding.lifecycle_snapshot_sha256.as_str(),
        binding.verifier_profile_sha256.as_str(),
    ] {
        hash_field(&mut hasher, value);
    }
    let digest = hasher.finalize();
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

fn validate_use_context(
    context: &QualificationUseContextV1,
) -> Result<(), IndependentQualificationExchangeRejectV1> {
    if !valid_atom(&context.query_invocation_id, 256)
        || !valid_atom(&context.domain_id, 512)
        || [
            context.subject_scope_sha256.as_str(),
            context.environment_revision_sha256.as_str(),
            context.tool_profile_sha256.as_str(),
            context.decision_need_sha256.as_str(),
            context.purpose_sha256.as_str(),
            context.qualification_policy_sha256.as_str(),
            context.lifecycle_snapshot_sha256.as_str(),
            context.verifier_profile_sha256.as_str(),
        ]
        .iter()
        .any(|digest| !valid_sha256(digest))
    {
        return Err(IndependentQualificationExchangeRejectV1::InvalidUseContext);
    }
    Ok(())
}

fn build_request(
    batch: &CandidateEvidenceBatchV1,
    context: &QualificationUseContextV1,
    expected: &CandidateEvidenceBatchExpectationV1<'_>,
) -> Result<EvidenceQualificationRequestV1, IndependentQualificationExchangeRejectV1> {
    validate_use_context(context)?;
    let rebuilt = encode_candidate_evidence_jsonl(&batch.evidence)
        .map_err(IndependentQualificationExchangeRejectV1::CandidateBatch)?;
    if &rebuilt != batch {
        return Err(IndependentQualificationExchangeRejectV1::CandidateBatch(
            CandidateEvidenceReject::InvalidDigest,
        ));
    }
    if batch.candidate_id != expected.candidate_id
        || batch.backend_id != expected.backend_id
        || batch.backend_revision_sha256 != expected.backend_revision_sha256
        || batch.query_scope != expected.query_scope
        || batch.query_purpose != expected.query_purpose
    {
        return Err(IndependentQualificationExchangeRejectV1::CandidateIdentityMismatch);
    }

    let mut candidate_evidence_ids = batch.evidence
        .iter()
        .map(|item| item.evidence_id.clone())
        .collect::<Vec<_>>();
    candidate_evidence_ids.sort();
    if candidate_evidence_ids.is_empty()
        || candidate_evidence_ids.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(IndependentQualificationExchangeRejectV1::CandidateIdsNotCanonical);
    }

    let binding = EvidenceQualificationRequestBindingV1 {
        run_id: batch.run_id.clone(),
        query_invocation_id: context.query_invocation_id.clone(),
        domain_id: context.domain_id.clone(),
        subject_scope_sha256: context.subject_scope_sha256.clone(),
        environment_revision_sha256: context.environment_revision_sha256.clone(),
        tool_profile_sha256: context.tool_profile_sha256.clone(),
        query_sha256: batch.query_sha256.clone(),
        candidate_set_sha256: batch.candidate_set_sha256.clone(),
        candidate_evidence_ids,
        decision_need_sha256: context.decision_need_sha256.clone(),
        purpose_sha256: context.purpose_sha256.clone(),
        qualification_policy_sha256: context.qualification_policy_sha256.clone(),
        lifecycle_snapshot_sha256: context.lifecycle_snapshot_sha256.clone(),
        verifier_profile_sha256: context.verifier_profile_sha256.clone(),
    };
    let candidate_batch_jsonl = batch.to_jsonl()
        .map_err(IndependentQualificationExchangeRejectV1::CandidateBatch)?;
    let request_sha256 = request_digest(&candidate_batch_jsonl, &binding);

    Ok(EvidenceQualificationRequestV1 {
        schema_version: QUALIFICATION_REQUEST_SCHEMA_V1.to_owned(),
        candidate_batch_jsonl,
        binding,
        request_sha256,
    })
}

/// Request qualification from an injected external issuer, verify its signed
/// response using the consumer's pinned verifier, and return only an exact
/// qualified projection or explicit projection abstention.
///
/// This function never returns the original candidate batch. On every issuer,
/// transport, signature, binding, or projection error the caller receives an
/// error and must withhold all candidate payloads from the reader.
pub fn independently_qualify_candidate_batch_v1(
    batch: &CandidateEvidenceBatchV1,
    context: &QualificationUseContextV1,
    expected: &CandidateEvidenceBatchExpectationV1<'_>,
    issuer: &impl IndependentEvidenceQualificationIssuerV1,
    pinned: &PinnedEd25519QualificationVerifierV1,
    clock: &impl QualificationTrustedClockV1,
) -> Result<IndependentQualificationExchangeResultV1, IndependentQualificationExchangeRejectV1> {
    let request = build_request(batch, context, expected)?;
    let envelope_json = issuer.issue(&request).map_err(|error| match error {
        QualificationIssuerErrorV1::Unavailable =>
            IndependentQualificationExchangeRejectV1::IssuerUnavailable,
        QualificationIssuerErrorV1::Rejected =>
            IndependentQualificationExchangeRejectV1::IssuerRejected,
        QualificationIssuerErrorV1::MalformedResponse =>
            IndependentQualificationExchangeRejectV1::IssuerMalformedResponse,
    })?;

    // Obtain current time after the issuer response; never trust a timestamp
    // supplied by the candidate backend or echoed in the request.
    let trusted_now_unix = clock.now_unix().map_err(|error| match error {
        QualificationClockErrorV1::Unavailable =>
            IndependentQualificationExchangeRejectV1::TrustedClockUnavailable,
        QualificationClockErrorV1::Untrusted =>
            IndependentQualificationExchangeRejectV1::TrustedClockUntrusted,
    })?;
    let evidence_qualification_binding = EvidenceQualificationBindingV1 {
        run_id: &request.binding.run_id,
        query_invocation_id: &request.binding.query_invocation_id,
        domain_id: &request.binding.domain_id,
        subject_scope_sha256: &request.binding.subject_scope_sha256,
        environment_revision_sha256: &request.binding.environment_revision_sha256,
        tool_profile_sha256: &request.binding.tool_profile_sha256,
        query_sha256: &request.binding.query_sha256,
        candidate_set_sha256: &request.binding.candidate_set_sha256,
        candidate_evidence_ids: &request.binding.candidate_evidence_ids,
        decision_need_sha256: &request.binding.decision_need_sha256,
        purpose_sha256: &request.binding.purpose_sha256,
        qualification_policy_sha256: &request.binding.qualification_policy_sha256,
        lifecycle_snapshot_sha256: &request.binding.lifecycle_snapshot_sha256,
        verifier_profile_sha256: &request.binding.verifier_profile_sha256,
        now_unix: trusted_now_unix,
    };

    // Parse and verify the DSSE/in-toto envelope with the independently pinned
    // profile. The typed receipt is not accepted unless its signed predicate
    // matches every consumer-owned request binding field.
    let dsse_verifier = DsseQualificationSignatureVerifierV1::from_envelope_json(
        &envelope_json,
        pinned,
    )
    .map_err(IndependentQualificationExchangeRejectV1::Qualification)?;
    let verified = VerifiedEvidenceQualificationV1::verify_and_bind(
        dsse_verifier.receipt(),
        &evidence_qualification_binding,
        &dsse_verifier,
    )
    .map_err(IndependentQualificationExchangeRejectV1::Qualification)?;

    let projection = project_qualified_candidate_evidence_v1(
        batch,
        &verified,
        &evidence_qualification_binding,
        expected,
        &dsse_verifier,
    )
    .map_err(IndependentQualificationExchangeRejectV1::Projection)?;

    Ok(IndependentQualificationExchangeResultV1 {
        projection,
        receipt_envelope_json: envelope_json,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate_evidence::{
        CandidateEvidenceV1, EvidenceModalityV1, encode_candidate_evidence_jsonl,
    };
    use crate::evidence_qualification_gate::{
        EvidenceDispositionV1, EvidenceQualificationEntryV1, EvidenceQualificationReceiptV1,
    };
    use crate::qualification_dsse::{
        DSSE_PAYLOAD_TYPE_V1,
    };
    use crate::qualification_trust::{
        PinnedEd25519QualificationVerifierV1, QualificationKeyStatusV1,
        QualificationTrustKeyV1, QualificationTrustProfileV1, qualification_key_id_v1,
    };
    use ed25519_dalek::{Signer, SigningKey};

    const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    struct TestIssuer {
        signing_key: SigningKey,
        profile_sha256: String,
        disposition: EvidenceDispositionV1,
        failure: Option<QualificationIssuerErrorV1>,
        rebind_purpose: bool,
        issued_at_unix: u64,
    }

    impl IndependentEvidenceQualificationIssuerV1 for TestIssuer {
        fn issue(
            &self,
            request: &EvidenceQualificationRequestV1,
        ) -> Result<String, QualificationIssuerErrorV1> {
            if let Some(error) = self.failure {
                return Err(error);
            }
            let binding = &request.binding;
            let entries = binding.candidate_evidence_ids.iter().map(|evidence_id| {
                EvidenceQualificationEntryV1 {
                    evidence_id: evidence_id.clone(),
                    disposition: self.disposition,
                    basis_receipt_sha256: F.to_owned(),
                }
            }).collect();
            let mut receipt = EvidenceQualificationReceiptV1 {
                run_id: binding.run_id.clone(),
                query_invocation_id: binding.query_invocation_id.clone(),
                domain_id: binding.domain_id.clone(),
                subject_scope_sha256: binding.subject_scope_sha256.clone(),
                environment_revision_sha256: binding.environment_revision_sha256.clone(),
                tool_profile_sha256: binding.tool_profile_sha256.clone(),
                query_sha256: binding.query_sha256.clone(),
                candidate_set_sha256: binding.candidate_set_sha256.clone(),
                candidate_evidence_ids_sha256: {
                    // The digest algorithm is part of the typed gate contract.
                    let mut hasher = Sha256::new();
                    hasher.update(b"luminous:mace:candidate-evidence-ids:v1\0");
                    hasher.update((binding.candidate_evidence_ids.len() as u64).to_be_bytes());
                    for id in &binding.candidate_evidence_ids {
                        hash_field(&mut hasher, id);
                    }
                    let digest = hasher.finalize();
                    digest.iter().map(|byte| format!("{byte:02x}")).collect()
                },
                decision_need_sha256: binding.decision_need_sha256.clone(),
                purpose_sha256: if self.rebind_purpose { A.to_owned() } else { binding.purpose_sha256.clone() },
                qualification_policy_sha256: binding.qualification_policy_sha256.clone(),
                lifecycle_snapshot_sha256: binding.lifecycle_snapshot_sha256.clone(),
                verifier_profile_sha256: self.profile_sha256.clone(),
                key_id: qualification_key_id_v1(&self.signing_key.verifying_key().to_bytes()),
                issued_at_unix: self.issued_at_unix,
                expires_at_unix: self.issued_at_unix + 300,
                entries,
                signature: Vec::new(),
            };
            let payload = in_toto_qualification_payload_v1(&receipt)
                .map_err(|_| QualificationIssuerErrorV1::MalformedResponse)?;
            let pae = dsse_pae_v1(crate::qualification_dsse::DSSE_PAYLOAD_TYPE_V1, &payload);
            let signature = self.signing_key.sign(&pae).to_bytes();
            encode_dsse_envelope_v1(&payload, &receipt.key_id, &signature)
                .map_err(|_| QualificationIssuerErrorV1::MalformedResponse)
        }
    }

    struct TestClock {
        now_unix: Result<u64, QualificationClockErrorV1>,
    }

    impl QualificationTrustedClockV1 for TestClock {
        fn now_unix(&self) -> Result<u64, QualificationClockErrorV1> {
            self.now_unix
        }
    }

    fn test_clock() -> TestClock {
        TestClock { now_unix: Ok(100) }
    }

    struct SequenceClock {
        readings: std::sync::Mutex<std::collections::VecDeque<u64>>,
    }

    impl QualificationTrustedClockV1 for SequenceClock {
        fn now_unix(&self) -> Result<u64, QualificationClockErrorV1> {
            self.readings.lock()
                .map_err(|_| QualificationClockErrorV1::Untrusted)?
                .pop_front()
                .ok_or(QualificationClockErrorV1::Unavailable)
        }
    }

    #[test]
    fn monotonic_clock_guard_rejects_rollback_and_allows_equal_time() {
        let source = SequenceClock {
            readings: std::sync::Mutex::new(
                [100, 101, 101, 99].into_iter().collect(),
            ),
        };
        let guarded = MonotonicQualificationClockGuardV1::new(source);
        assert_eq!(guarded.now_unix(), Ok(100));
        assert_eq!(guarded.now_unix(), Ok(101));
        assert_eq!(guarded.now_unix(), Ok(101));
        assert_eq!(guarded.now_unix(), Err(QualificationClockErrorV1::Untrusted));
    }

    fn batch() -> CandidateEvidenceBatchV1 {
        let evidence = CandidateEvidenceV1 {
            run_id: "run-1".into(),
            candidate_id: "candidate-1".into(),
            backend_id: "bm25-test".into(),
            backend_revision_sha256: A.into(),
            query_scope: "tenant:demo".into(),
            query_purpose: "benchmark:memory-context".into(),
            query_sha256: B.into(),
            evidence_id: "evidence:1".into(),
            source_id: "source:1".into(),
            source_sha256: C.into(),
            payload_sha256: D.into(),
            payload_ref: "cas://source:1/payload:1".into(),
            modality: EvidenceModalityV1::Text,
            retrieval_channel: "bm25".into(),
            retrieval_rank: 1,
            retrieval_score: Some(0.75),
            parent_evidence_ids: Vec::new(),
            derivation_receipt_sha256: None,
        };
        encode_candidate_evidence_jsonl(&[evidence]).expect("valid candidate batch")
    }

    fn profile(signing_key: &SigningKey) -> QualificationTrustProfileV1 {
        let public_key = signing_key.verifying_key().to_bytes();
        QualificationTrustProfileV1 {
            schema_version: 1,
            verifier_id: "independent-test-qualifier".into(),
            profile_version: 1,
            keys: vec![QualificationTrustKeyV1 {
                key_id: qualification_key_id_v1(&public_key),
                public_key,
                status: QualificationKeyStatusV1::Active,
                not_before_unix: 1,
                not_after_unix: 10_000,
            }],
        }
    }

    fn context(profile_sha256: String) -> QualificationUseContextV1 {
        QualificationUseContextV1 {
            query_invocation_id: "invocation-1".into(),
            domain_id: "web".into(),
            subject_scope_sha256: B.into(),
            environment_revision_sha256: C.into(),
            tool_profile_sha256: D.into(),
            decision_need_sha256: E.into(),
            purpose_sha256: F.into(),
            qualification_policy_sha256: A.into(),
            lifecycle_snapshot_sha256: B.into(),
            verifier_profile_sha256: profile_sha256,
        }
    }

    fn use_binding<'a>(
        batch: &'a CandidateEvidenceBatchV1,
        context: &'a QualificationUseContextV1,
        now_unix: u64,
    ) -> EvidenceQualificationBindingV1<'a> {
        let mut ids = batch.evidence.iter()
            .map(|item| item.evidence_id.clone())
            .collect::<Vec<_>>();
        ids.sort();
        EvidenceQualificationBindingV1 {
            run_id: &batch.run_id,
            query_invocation_id: &context.query_invocation_id,
            domain_id: &context.domain_id,
            subject_scope_sha256: &context.subject_scope_sha256,
            environment_revision_sha256: &context.environment_revision_sha256,
            tool_profile_sha256: &context.tool_profile_sha256,
            query_sha256: &batch.query_sha256,
            candidate_set_sha256: &batch.candidate_set_sha256,
            candidate_evidence_ids: Box::leak(ids.into_boxed_slice()),
            decision_need_sha256: &context.decision_need_sha256,
            purpose_sha256: &context.purpose_sha256,
            qualification_policy_sha256: &context.qualification_policy_sha256,
            lifecycle_snapshot_sha256: &context.lifecycle_snapshot_sha256,
            verifier_profile_sha256: &context.verifier_profile_sha256,
            now_unix,
        }
    }

    fn expected<'a>(candidate: &'a CandidateEvidenceBatchV1)
        -> CandidateEvidenceBatchExpectationV1<'a> {
        CandidateEvidenceBatchExpectationV1 {
            candidate_id: &candidate.candidate_id,
            backend_id: &candidate.backend_id,
            backend_revision_sha256: &candidate.backend_revision_sha256,
            query_scope: &candidate.query_scope,
            query_purpose: &candidate.query_purpose,
        }
    }

    fn setup(disposition: EvidenceDispositionV1, rebind_purpose: bool)
        -> (TestIssuer, PinnedEd25519QualificationVerifierV1, QualificationUseContextV1) {
        let signing_key = SigningKey::from_bytes(&[7; 32]);
        let profile = profile(&signing_key);
        let profile_sha256 = profile.profile_sha256().expect("valid profile");
        let verifier = PinnedEd25519QualificationVerifierV1::from_pinned_profile(
            &profile, &profile_sha256, profile.profile_version,
        ).expect("pinned test trust");
        let issuer = TestIssuer {
            signing_key,
            profile_sha256: profile_sha256.clone(),
            disposition,
            failure: None,
            rebind_purpose,
            issued_at_unix: 90,
        };
        (issuer, verifier, context(profile_sha256))
    }

    #[test]
    fn independently_signed_receipt_returns_only_qualified_projection() {
        let batch = batch();
        let (issuer, pinned_profile, context) = setup(EvidenceDispositionV1::Admissible, false);
        let result = independently_qualify_candidate_batch_v1(
            &batch, &context, &expected(&batch), &issuer, &pinned_profile, &test_clock(),
        ).expect("qualification exchange");
        match result.projection() {
            CandidateEvidenceProjectionV1::Qualified(qualified) => {
                assert_eq!(qualified.evidence().len(), 1);
                assert_eq!(qualified.source_candidate_set_sha256(), batch.candidate_set_sha256);
                assert_eq!(qualified.decision_need_sha256(), context.decision_need_sha256);
                let binding = use_binding(&batch, &context, 101);
                let verifier = result.verifier_for_use(&pinned_profile).expect("fresh DSSE verifier");
                qualified.validate_at_use(&binding, &verifier).expect("same context remains valid");
            }
            CandidateEvidenceProjectionV1::Abstained(_) => panic!("unexpected abstention"),
        }
    }

    #[test]
    fn rejected_or_indeterminate_candidate_is_never_returned_to_reader() {
        let batch = batch();
        let (issuer, verifier, context) = setup(EvidenceDispositionV1::Indeterminate, false);
        let result = independently_qualify_candidate_batch_v1(
            &batch, &context, &expected(&batch), &issuer, &verifier, &test_clock(),
        ).expect("valid signed receipt");
        assert!(matches!(result.projection(), CandidateEvidenceProjectionV1::Abstained(_)));
    }

    #[test]
    fn issuer_unavailable_is_an_error_and_never_falls_back_to_raw_candidates() {
        let batch = batch();
        let (mut issuer, verifier, context) = setup(EvidenceDispositionV1::Admissible, false);
        issuer.failure = Some(QualificationIssuerErrorV1::Unavailable);
        assert_eq!(
            independently_qualify_candidate_batch_v1(
                &batch, &context, &expected(&batch), &issuer, &verifier, &test_clock(),
            ),
            Err(IndependentQualificationExchangeRejectV1::IssuerUnavailable),
        );
    }

    #[test]
    fn purpose_rebinding_in_a_signed_response_is_rejected() {
        let batch = batch();
        let (issuer, verifier, context) = setup(EvidenceDispositionV1::Admissible, true);
        assert_eq!(
            independently_qualify_candidate_batch_v1(
                &batch, &context, &expected(&batch), &issuer, &verifier, &test_clock(),
            ),
            Err(IndependentQualificationExchangeRejectV1::Qualification(
                EvidenceQualificationRejectV1::BindingMismatch,
            )),
        );
    }

    #[test]
    fn candidate_identity_mismatch_is_rejected_before_calling_issuer() {
        struct MustNotBeCalled;
        impl IndependentEvidenceQualificationIssuerV1 for MustNotBeCalled {
            fn issue(&self, _: &EvidenceQualificationRequestV1)
                -> Result<String, QualificationIssuerErrorV1> {
                panic!("issuer must not receive a mismatched batch");
            }
        }
        let batch = batch();
        let (issuer, verifier, context) = setup(EvidenceDispositionV1::Admissible, false);
        let _ = issuer;
        let wrong = CandidateEvidenceBatchExpectationV1 {
            candidate_id: "different-candidate",
            backend_id: &batch.backend_id,
            backend_revision_sha256: &batch.backend_revision_sha256,
            query_scope: &batch.query_scope,
            query_purpose: &batch.query_purpose,
        };
        assert_eq!(
            independently_qualify_candidate_batch_v1(
                &batch, &context, &wrong, &MustNotBeCalled, &verifier, &test_clock(),
            ),
            Err(IndependentQualificationExchangeRejectV1::CandidateIdentityMismatch),
        );
    }


    #[test]
    fn unavailable_or_untrusted_clock_never_projects_qualified_context() {
        let batch = batch();
        let (issuer, verifier, context) = setup(EvidenceDispositionV1::Admissible, false);
        let expected_batch = expected(&batch);
        for (clock, expected_error) in [
            (
                TestClock { now_unix: Err(QualificationClockErrorV1::Unavailable) },
                IndependentQualificationExchangeRejectV1::TrustedClockUnavailable,
            ),
            (
                TestClock { now_unix: Err(QualificationClockErrorV1::Untrusted) },
                IndependentQualificationExchangeRejectV1::TrustedClockUntrusted,
            ),
        ] {
            assert_eq!(
                independently_qualify_candidate_batch_v1(
                    &batch, &context, &expected_batch, &issuer, &verifier, &clock,
                ),
                Err(expected_error),
            );
        }
    }

    #[test]
    fn trusted_clock_rejects_expired_issuer_receipt() {
        let batch = batch();
        let (issuer, verifier, context) = setup(EvidenceDispositionV1::Admissible, false);
        let clock = TestClock { now_unix: Ok(issuer.issued_at_unix + 300) };
        assert_eq!(
            independently_qualify_candidate_batch_v1(
                &batch, &context, &expected(&batch), &issuer, &verifier, &clock,
            ),
            Err(IndependentQualificationExchangeRejectV1::Qualification(
                EvidenceQualificationRejectV1::ReceiptExpired,
            )),
        );
    }

    #[test]
    fn request_sorts_complete_candidate_ids_independently_of_retrieval_order() {
        let original = batch();
        let mut second = original.evidence[0].clone();
        second.evidence_id = "evidence:0".into();
        second.retrieval_rank = 2;
        second.source_id = "source:0".into();
        second.payload_ref = "cas://source:0/payload:0".into();
        let multiple = encode_candidate_evidence_jsonl(&[
            original.evidence[0].clone(),
            second,
        ]).expect("two candidates");
        // The canonical batch is in channel/rank order, which is not the
        // lexical order required by the qualification receipt.
        assert_eq!(multiple.evidence[0].evidence_id, "evidence:1");
        let (issuer, verifier, context) = setup(EvidenceDispositionV1::Admissible, false);
        let request = build_request(&multiple, &context, &expected(&multiple))
            .expect("complete request binding");
        assert_eq!(
            request.binding.candidate_evidence_ids,
            vec!["evidence:0".to_owned(), "evidence:1".to_owned()]
        );
        let decoded = EvidenceQualificationRequestV1::from_json_value(
            &request.to_json_value(),
        ).expect("issuer-side strict decode");
        assert_eq!(decoded, request);
        let _ = (issuer, verifier);
    }

    #[test]
    fn request_digest_commits_to_context_and_exact_candidate_jsonl() {
        let batch = batch();
        let (issuer, verifier, context) = setup(EvidenceDispositionV1::Admissible, false);
        let request = build_request(&batch, &context, &expected(&batch)).expect("request");
        assert_eq!(request.request_sha256, request_digest(
            &request.candidate_batch_jsonl, &request.binding,
        ));
        let decoded = EvidenceQualificationRequestV1::from_json_value(
            &request.to_json_value(),
        ).expect("strict request decode");
        assert_eq!(decoded, request);

        let mut tampered = request.to_json_value();
        tampered["binding"]["purpose_sha256"] = Value::String(A.to_owned());
        assert_eq!(
            EvidenceQualificationRequestV1::from_json_value(&tampered),
            Err(IndependentQualificationExchangeRejectV1::RequestDigestMismatch),
        );
        assert_eq!(
            request.to_json_value()["schema_version"],
            QUALIFICATION_REQUEST_SCHEMA_V1,
        );
        let _ = (issuer, verifier);
        let mut changed = request.binding.clone();
        changed.purpose_sha256 = A.into();
        assert_ne!(
            request.request_sha256,
            request_digest(&request.candidate_batch_jsonl, &changed),
        );

        let vector_binding = EvidenceQualificationRequestBindingV1 {
            run_id: "run-1".into(),
            query_invocation_id: "invocation-1".into(),
            domain_id: "web".into(),
            subject_scope_sha256: B.into(),
            environment_revision_sha256: C.into(),
            tool_profile_sha256: D.into(),
            query_sha256: E.into(),
            candidate_set_sha256: F.into(),
            candidate_evidence_ids: vec!["evidence:1".into(), "evidence:2".into()],
            decision_need_sha256: "1".repeat(64),
            purpose_sha256: "2".repeat(64),
            qualification_policy_sha256: "3".repeat(64),
            lifecycle_snapshot_sha256: "4".repeat(64),
            verifier_profile_sha256: "5".repeat(64),
        };
        assert_eq!(
            request_digest("canonical-candidate-batch-jsonl-v1", &vector_binding),
            "495c022b39b611b578eeb7924c36c7eec7d8436df8034e444c4b7eda67ba4384",
        );
    }
}
