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

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::candidate_evidence::{
    CandidateEvidenceBatchV1, CandidateEvidenceReject, encode_candidate_evidence_jsonl,
};
use crate::evidence_qualification_gate::{
    EvidenceQualificationBindingV1, EvidenceQualificationReceiptV1,
    EvidenceQualificationRejectV1, EvidenceQualificationSignatureVerifierV1,
    VerifiedEvidenceQualificationV1,
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
    pub now_unix: u64,
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
    pub now_unix: u64,
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
                "now_unix": self.binding.now_unix,
            },
            "request_sha256": self.request_sha256,
        })
    }
}

/// A transport implementation should call a separately governed qualification
/// service. It must not sign locally from the candidate backend's own store or
/// credentials. Timeouts and ambiguous outcomes must map to Unavailable.
pub trait IndependentEvidenceQualificationIssuerV1 {
    fn issue(
        &self,
        request: &EvidenceQualificationRequestV1,
    ) -> Result<EvidenceQualificationReceiptV1, QualificationIssuerErrorV1>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QualificationIssuerErrorV1 {
    Unavailable,
    Rejected,
    MalformedResponse,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndependentQualificationExchangeRejectV1 {
    InvalidUseContext,
    CandidateBatch(CandidateEvidenceReject),
    CandidateIdentityMismatch,
    CandidateIdsNotCanonical,
    IssuerUnavailable,
    IssuerRejected,
    IssuerMalformedResponse,
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
    hasher.update(binding.now_unix.to_be_bytes());
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
        now_unix: context.now_unix,
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
    verifier: &impl EvidenceQualificationSignatureVerifierV1,
) -> Result<CandidateEvidenceProjectionV1, IndependentQualificationExchangeRejectV1> {
    let request = build_request(batch, context, expected)?;
    let receipt = issuer.issue(&request).map_err(|error| match error {
        QualificationIssuerErrorV1::Unavailable =>
            IndependentQualificationExchangeRejectV1::IssuerUnavailable,
        QualificationIssuerErrorV1::Rejected =>
            IndependentQualificationExchangeRejectV1::IssuerRejected,
        QualificationIssuerErrorV1::MalformedResponse =>
            IndependentQualificationExchangeRejectV1::IssuerMalformedResponse,
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
        now_unix: request.binding.now_unix,
    };
    let verified = VerifiedEvidenceQualificationV1::verify_and_bind(
        &receipt,
        &evidence_qualification_binding,
        verifier,
    )
    .map_err(IndependentQualificationExchangeRejectV1::Qualification)?;

    project_qualified_candidate_evidence_v1(
        batch,
        &verified,
        &evidence_qualification_binding,
        expected,
    )
    .map_err(IndependentQualificationExchangeRejectV1::Projection)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate_evidence::{
        CandidateEvidenceV1, EvidenceModalityV1, encode_candidate_evidence_jsonl,
    };
    use crate::evidence_qualification_gate::{
        EvidenceDispositionV1, EvidenceQualificationEntryV1,
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
    }

    impl IndependentEvidenceQualificationIssuerV1 for TestIssuer {
        fn issue(
            &self,
            request: &EvidenceQualificationRequestV1,
        ) -> Result<EvidenceQualificationReceiptV1, QualificationIssuerErrorV1> {
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
                issued_at_unix: binding.now_unix,
                expires_at_unix: binding.now_unix + 300,
                entries,
                signature: Vec::new(),
            };
            let signature = self.signing_key.sign(receipt.signing_payload_sha256().as_bytes());
            receipt.signature = signature.to_bytes().to_vec();
            Ok(receipt)
        }
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
            now_unix: 100,
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
        };
        (issuer, verifier, context(profile_sha256))
    }

    #[test]
    fn independently_signed_receipt_returns_only_qualified_projection() {
        let batch = batch();
        let (issuer, verifier, context) = setup(EvidenceDispositionV1::Admissible, false);
        let result = independently_qualify_candidate_batch_v1(
            &batch, &context, &expected(&batch), &issuer, &verifier,
        ).expect("qualification exchange");
        match result {
            CandidateEvidenceProjectionV1::Qualified(qualified) => {
                assert_eq!(qualified.evidence().len(), 1);
                assert_eq!(qualified.source_candidate_set_sha256(), batch.candidate_set_sha256);
                assert_eq!(qualified.decision_need_sha256(), context.decision_need_sha256);
            }
            CandidateEvidenceProjectionV1::Abstained(_) => panic!("unexpected abstention"),
        }
    }

    #[test]
    fn rejected_or_indeterminate_candidate_is_never_returned_to_reader() {
        let batch = batch();
        let (issuer, verifier, context) = setup(EvidenceDispositionV1::Indeterminate, false);
        let result = independently_qualify_candidate_batch_v1(
            &batch, &context, &expected(&batch), &issuer, &verifier,
        ).expect("valid signed receipt");
        assert!(matches!(result, CandidateEvidenceProjectionV1::Abstained(_)));
    }

    #[test]
    fn issuer_unavailable_is_an_error_and_never_falls_back_to_raw_candidates() {
        let batch = batch();
        let (mut issuer, verifier, context) = setup(EvidenceDispositionV1::Admissible, false);
        issuer.failure = Some(QualificationIssuerErrorV1::Unavailable);
        assert_eq!(
            independently_qualify_candidate_batch_v1(
                &batch, &context, &expected(&batch), &issuer, &verifier,
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
                &batch, &context, &expected(&batch), &issuer, &verifier,
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
                -> Result<EvidenceQualificationReceiptV1, QualificationIssuerErrorV1> {
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
                &batch, &context, &wrong, &MustNotBeCalled, &verifier,
            ),
            Err(IndependentQualificationExchangeRejectV1::CandidateIdentityMismatch),
        );
    }

    #[test]
    fn request_digest_commits_to_context_and_exact_candidate_jsonl() {
        let batch = batch();
        let (issuer, verifier, context) = setup(EvidenceDispositionV1::Admissible, false);
        let request = build_request(&batch, &context, &expected(&batch)).expect("request");
        assert_eq!(request.request_sha256, request_digest(
            &request.candidate_batch_jsonl, &request.binding,
        ));
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
    }
}
