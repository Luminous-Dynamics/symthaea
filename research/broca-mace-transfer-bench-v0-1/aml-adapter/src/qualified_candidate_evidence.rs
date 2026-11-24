// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed projection from a verified qualification to its exact candidate
//! batch. This filters evidence; it does not authorize actions or prove truth.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::candidate_evidence::{
    CandidateEvidenceBatchV1, CandidateEvidenceReject, CandidateEvidenceV1,
    encode_candidate_evidence_jsonl,
};
use crate::evidence_qualification_gate::{
    EvidenceQualificationBindingV1, EvidenceQualificationRejectV1,
    VerifiedEvidenceQualificationV1,
};

/// Independently expected batch identity, supplied by the query/runner context.
pub struct CandidateEvidenceBatchExpectationV1<'a> {
    pub candidate_id: &'a str,
    pub backend_id: &'a str,
    pub backend_revision_sha256: &'a str,
    pub query_scope: &'a str,
    pub query_purpose: &'a str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateEvidenceProjectionRejectV1 {
    Candidate(CandidateEvidenceReject),
    BatchNotCanonical,
    BatchIdentityMismatch,
    Qualification(EvidenceQualificationRejectV1),
    DependencyCycle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateEvidenceAbstentionReasonV1 {
    NoIndependentlyAdmissibleEvidence,
    DependencyClosureRemovedAllEvidence,
}

/// A purpose-, DecisionNeed- and lifecycle-scoped evidence projection.
/// This is not an action authorization capability.
#[derive(Clone, Debug, PartialEq)]
struct QualificationUseSnapshotV1 {
    run_id: String,
    query_invocation_id: String,
    domain_id: String,
    subject_scope_sha256: String,
    environment_revision_sha256: String,
    tool_profile_sha256: String,
    query_sha256: String,
    candidate_set_sha256: String,
    candidate_evidence_ids: Vec<String>,
    decision_need_sha256: String,
    purpose_sha256: String,
    qualification_policy_sha256: String,
    lifecycle_snapshot_sha256: String,
    verifier_profile_sha256: String,
    verified_at_unix: u64,
    expires_at_unix: u64,
}

impl QualificationUseSnapshotV1 {
    fn from_binding(binding: &EvidenceQualificationBindingV1<'_>, expires_at_unix: u64) -> Self {
        Self {
            run_id: binding.run_id.to_owned(),
            query_invocation_id: binding.query_invocation_id.to_owned(),
            domain_id: binding.domain_id.to_owned(),
            subject_scope_sha256: binding.subject_scope_sha256.to_owned(),
            environment_revision_sha256: binding.environment_revision_sha256.to_owned(),
            tool_profile_sha256: binding.tool_profile_sha256.to_owned(),
            query_sha256: binding.query_sha256.to_owned(),
            candidate_set_sha256: binding.candidate_set_sha256.to_owned(),
            candidate_evidence_ids: binding.candidate_evidence_ids.to_vec(),
            decision_need_sha256: binding.decision_need_sha256.to_owned(),
            purpose_sha256: binding.purpose_sha256.to_owned(),
            qualification_policy_sha256: binding.qualification_policy_sha256.to_owned(),
            lifecycle_snapshot_sha256: binding.lifecycle_snapshot_sha256.to_owned(),
            verifier_profile_sha256: binding.verifier_profile_sha256.to_owned(),
            verified_at_unix: binding.now_unix,
            expires_at_unix,
        }
    }

    fn validate_at_use(
        &self,
        binding: &EvidenceQualificationBindingV1<'_>,
    ) -> Result<(), EvidenceQualificationRejectV1> {
        for (actual, expected) in [
            (self.run_id.as_str(), binding.run_id),
            (self.query_invocation_id.as_str(), binding.query_invocation_id),
            (self.domain_id.as_str(), binding.domain_id),
            (self.subject_scope_sha256.as_str(), binding.subject_scope_sha256),
            (self.environment_revision_sha256.as_str(), binding.environment_revision_sha256),
            (self.tool_profile_sha256.as_str(), binding.tool_profile_sha256),
            (self.query_sha256.as_str(), binding.query_sha256),
            (self.candidate_set_sha256.as_str(), binding.candidate_set_sha256),
            (self.decision_need_sha256.as_str(), binding.decision_need_sha256),
            (self.purpose_sha256.as_str(), binding.purpose_sha256),
            (self.qualification_policy_sha256.as_str(), binding.qualification_policy_sha256),
            (self.lifecycle_snapshot_sha256.as_str(), binding.lifecycle_snapshot_sha256),
            (self.verifier_profile_sha256.as_str(), binding.verifier_profile_sha256),
        ] {
            if actual != expected {
                return Err(EvidenceQualificationRejectV1::BindingMismatch);
            }
        }
        if self.candidate_evidence_ids != binding.candidate_evidence_ids {
            return Err(EvidenceQualificationRejectV1::BindingMismatch);
        }
        // A caller cannot move the consuming clock backward past the time at
        // which the signed qualification was first verified for use.
        if binding.now_unix < self.verified_at_unix {
            return Err(EvidenceQualificationRejectV1::ReceiptNotYetValid);
        }
        if binding.now_unix >= self.expires_at_unix {
            return Err(EvidenceQualificationRejectV1::ReceiptExpired);
        }
        Ok(())
    }
}

/// A purpose-, DecisionNeed- and lifecycle-scoped evidence projection.
/// This is not an action authorization capability. It is valid only for the
/// exact use binding and trusted clock interval recorded at projection time.
#[derive(Clone, Debug, PartialEq)]
pub struct QualifiedCandidateEvidenceBatchV1 {
    batch: CandidateEvidenceBatchV1,
    source_candidate_set_sha256: String,
    qualification_receipt_sha256: String,
    use_snapshot: QualificationUseSnapshotV1,
}

impl QualifiedCandidateEvidenceBatchV1 {
    /// Revalidate the exact context and time interval before every consumption.
    /// Live authorization and lifecycle currentness remain separate mandatory
    /// checks at the consequential read/action boundary.
    pub fn validate_at_use(
        &self,
        binding: &EvidenceQualificationBindingV1<'_>,
    ) -> Result<(), EvidenceQualificationRejectV1> {
        self.use_snapshot.validate_at_use(binding)
    }

    pub fn batch(&self) -> &CandidateEvidenceBatchV1 { &self.batch }
    pub fn evidence(&self) -> &[CandidateEvidenceV1] { &self.batch.evidence }
    pub fn source_candidate_set_sha256(&self) -> &str { &self.source_candidate_set_sha256 }
    pub fn qualified_candidate_set_sha256(&self) -> &str { &self.batch.candidate_set_sha256 }
    pub fn qualification_receipt_sha256(&self) -> &str { &self.qualification_receipt_sha256 }
    pub fn decision_need_sha256(&self) -> &str { &self.use_snapshot.decision_need_sha256 }
    pub fn purpose_sha256(&self) -> &str { &self.use_snapshot.purpose_sha256 }
    pub fn lifecycle_snapshot_sha256(&self) -> &str { &self.use_snapshot.lifecycle_snapshot_sha256 }
    pub fn qualification_expires_at_unix(&self) -> u64 { self.use_snapshot.expires_at_unix }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateEvidenceAbstentionV1 {
    source_candidate_set_sha256: String,
    qualification_receipt_sha256: String,
    reason: CandidateEvidenceAbstentionReasonV1,
    source_evidence_count: usize,
    independently_admissible_count: usize,
}

impl CandidateEvidenceAbstentionV1 {
    pub fn source_candidate_set_sha256(&self) -> &str { &self.source_candidate_set_sha256 }
    pub fn qualification_receipt_sha256(&self) -> &str { &self.qualification_receipt_sha256 }
    pub fn reason(&self) -> CandidateEvidenceAbstentionReasonV1 { self.reason }
    pub fn source_evidence_count(&self) -> usize { self.source_evidence_count }
    pub fn independently_admissible_count(&self) -> usize { self.independently_admissible_count }
}

/// An empty result is explicit abstention, never a successful empty batch.
#[derive(Clone, Debug, PartialEq)]
pub enum CandidateEvidenceProjectionV1 {
    Qualified(QualifiedCandidateEvidenceBatchV1),
    Abstained(CandidateEvidenceAbstentionV1),
}

/// Apply a signed qualification to one exact batch at consumption time.
///
/// Rebuilds the canonical batch; matches its request identity; rebinds the
/// complete qualification context; checks exact evidence-ID set equality;
/// removes rejected/indeterminate evidence; closes over all required parents;
/// rejects dependency cycles; and re-encodes the surviving subset.
pub fn project_qualified_candidate_evidence_v1(
    batch: &CandidateEvidenceBatchV1,
    qualification: &VerifiedEvidenceQualificationV1,
    binding: &EvidenceQualificationBindingV1<'_>,
    expected: &CandidateEvidenceBatchExpectationV1<'_>,
) -> Result<CandidateEvidenceProjectionV1, CandidateEvidenceProjectionRejectV1> {
    let rebuilt = encode_candidate_evidence_jsonl(&batch.evidence)
        .map_err(CandidateEvidenceProjectionRejectV1::Candidate)?;
    if &rebuilt != batch {
        return Err(CandidateEvidenceProjectionRejectV1::BatchNotCanonical);
    }

    if batch.run_id != binding.run_id
        || batch.query_sha256 != binding.query_sha256
        || batch.candidate_set_sha256 != binding.candidate_set_sha256
        || batch.candidate_id != expected.candidate_id
        || batch.backend_id != expected.backend_id
        || batch.backend_revision_sha256 != expected.backend_revision_sha256
        || batch.query_scope != expected.query_scope
        || batch.query_purpose != expected.query_purpose
    {
        return Err(CandidateEvidenceProjectionRejectV1::BatchIdentityMismatch);
    }

    let mut batch_ids = batch.evidence.iter()
        .map(|item| item.evidence_id.clone()).collect::<Vec<_>>();
    batch_ids.sort();
    if binding.candidate_evidence_ids != batch_ids.as_slice() {
        return Err(CandidateEvidenceProjectionRejectV1::Qualification(
            EvidenceQualificationRejectV1::BindingMismatch));
    }

    let admissible = qualification.admissible_evidence_ids_for_use(binding)
        .map_err(CandidateEvidenceProjectionRejectV1::Qualification)?;
    let independently_admissible_count = admissible.len();
    let mut keep = admissible.iter().cloned().collect::<BTreeSet<_>>();
    let source_evidence_count = batch.evidence.len();

    if keep.is_empty() {
        return Ok(CandidateEvidenceProjectionV1::Abstained(CandidateEvidenceAbstentionV1 {
            source_candidate_set_sha256: batch.candidate_set_sha256.clone(),
            qualification_receipt_sha256: qualification.receipt_sha256().to_owned(),
            reason: CandidateEvidenceAbstentionReasonV1::NoIndependentlyAdmissibleEvidence,
            source_evidence_count,
            independently_admissible_count: 0,
        }));
    }

    let by_id = batch.evidence.iter()
        .map(|item| (item.evidence_id.as_str(), item))
        .collect::<BTreeMap<_, _>>();

    // Greatest fixed point: every retained record must have every required
    // parent in this exact batch and in the retained admissible set.
    loop {
        let remove = keep.iter().filter(|id| {
            let Some(item) = by_id.get(id.as_str()) else { return true; };
            item.parent_evidence_ids.iter()
                .any(|parent| !keep.contains(parent) || !by_id.contains_key(parent.as_str()))
        }).cloned().collect::<Vec<_>>();
        if remove.is_empty() { break; }
        for id in remove { keep.remove(&id); }
    }

    if keep.is_empty() {
        return Ok(CandidateEvidenceProjectionV1::Abstained(CandidateEvidenceAbstentionV1 {
            source_candidate_set_sha256: batch.candidate_set_sha256.clone(),
            qualification_receipt_sha256: qualification.receipt_sha256().to_owned(),
            reason: CandidateEvidenceAbstentionReasonV1::DependencyClosureRemovedAllEvidence,
            source_evidence_count,
            independently_admissible_count,
        }));
    }

    ensure_acyclic(&keep, &by_id)?;
    let records = batch.evidence.iter()
        .filter(|item| keep.contains(&item.evidence_id)).cloned().collect::<Vec<_>>();
    let qualified_batch = encode_candidate_evidence_jsonl(&records)
        .map_err(CandidateEvidenceProjectionRejectV1::Candidate)?;

    Ok(CandidateEvidenceProjectionV1::Qualified(QualifiedCandidateEvidenceBatchV1 {
        batch: qualified_batch,
        source_candidate_set_sha256: batch.candidate_set_sha256.clone(),
        qualification_receipt_sha256: qualification.receipt_sha256().to_owned(),
        use_snapshot: QualificationUseSnapshotV1::from_binding(
            binding,
            qualification.expires_at_unix(),
        ),
    }))
}

/// Iterative Kahn cycle check; no attacker-controlled recursive call depth.
fn ensure_acyclic(
    keep: &BTreeSet<String>,
    by_id: &BTreeMap<&str, &CandidateEvidenceV1>,
) -> Result<(), CandidateEvidenceProjectionRejectV1> {
    let mut indegree = keep.iter().map(|id| (id.clone(), 0_usize))
        .collect::<BTreeMap<_, _>>();
    let mut children = BTreeMap::<String, Vec<String>>::new();

    for id in keep {
        let item = by_id.get(id.as_str())
            .ok_or(CandidateEvidenceProjectionRejectV1::BatchNotCanonical)?;
        for parent in &item.parent_evidence_ids {
            if !keep.contains(parent) {
                return Err(CandidateEvidenceProjectionRejectV1::BatchNotCanonical);
            }
            let degree = indegree.get_mut(id)
                .ok_or(CandidateEvidenceProjectionRejectV1::BatchNotCanonical)?;
            *degree = degree.checked_add(1)
                .ok_or(CandidateEvidenceProjectionRejectV1::DependencyCycle)?;
            children.entry(parent.clone()).or_default().push(id.clone());
        }
    }

    let mut ready = indegree.iter()
        .filter_map(|(id, degree)| (*degree == 0).then_some(id.clone()))
        .collect::<VecDeque<_>>();
    let mut visited = 0_usize;
    while let Some(parent) = ready.pop_front() {
        visited += 1;
        if let Some(dependents) = children.get(&parent) {
            for child in dependents {
                let degree = indegree.get_mut(child)
                    .ok_or(CandidateEvidenceProjectionRejectV1::BatchNotCanonical)?;
                *degree = degree.checked_sub(1)
                    .ok_or(CandidateEvidenceProjectionRejectV1::BatchNotCanonical)?;
                if *degree == 0 { ready.push_back(child.clone()); }
            }
        }
    }
    if visited != keep.len() {
        return Err(CandidateEvidenceProjectionRejectV1::DependencyCycle);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate_evidence::{EvidenceModalityV1, encode_candidate_evidence_jsonl};
    use crate::evidence_qualification_gate::{
        EvidenceDispositionV1, EvidenceQualificationEntryV1, EvidenceQualificationReceiptV1,
        EvidenceQualificationSignatureVerifierV1,
    };
    use sha2::{Digest, Sha256};

    const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    struct TestVerifier;
    impl EvidenceQualificationSignatureVerifierV1 for TestVerifier {
        fn verify_signature(&self, profile: &str, key: &str, payload: &str,
            signature: &[u8], _issued: u64, _now: u64)
            -> Result<(), EvidenceQualificationRejectV1> {
            if profile == F && key == "projection-test-key" && signature == payload.as_bytes() {
                Ok(())
            } else { Err(EvidenceQualificationRejectV1::SignatureRejected) }
        }
    }

    fn record(id: &str, rank: u32, parents: Vec<String>) -> CandidateEvidenceV1 {
        CandidateEvidenceV1 {
            run_id: "run-projection".into(), candidate_id: "candidate-projection".into(),
            backend_id: "bm25-test".into(), backend_revision_sha256: A.into(),
            query_scope: "tenant:test-scope".into(), query_purpose: "answer:workflow".into(),
            query_sha256: B.into(), evidence_id: id.into(), source_id: format!("source-{id}"),
            source_sha256: C.into(), payload_sha256: D.into(),
            payload_ref: format!("cas://{D}/{id}"), modality: EvidenceModalityV1::Text,
            retrieval_channel: "bm25".into(), retrieval_rank: rank, retrieval_score: Some(0.5),
            derivation_receipt_sha256: (!parents.is_empty()).then(|| E.to_owned()),
            parent_evidence_ids: parents,
        }
    }

    fn make_batch(records: &[CandidateEvidenceV1]) -> CandidateEvidenceBatchV1 {
        encode_candidate_evidence_jsonl(records).expect("valid candidate batch")
    }

    fn ids(batch: &CandidateEvidenceBatchV1) -> Vec<String> {
        let mut result = batch.evidence.iter().map(|item| item.evidence_id.clone()).collect::<Vec<_>>();
        result.sort();
        result
    }

    fn hash_field(hasher: &mut Sha256, value: &str) {
        hasher.update((value.len() as u64).to_be_bytes());
        hasher.update(value.as_bytes());
    }

    fn ids_digest(ids: &[String]) -> String {
        let mut h = Sha256::new();
        h.update(b"luminous:mace:candidate-evidence-ids:v1\0");
        h.update((ids.len() as u64).to_be_bytes());
        for id in ids { hash_field(&mut h, id); }
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }

    fn binding<'a>(batch: &'a CandidateEvidenceBatchV1, ids: &'a [String])
        -> EvidenceQualificationBindingV1<'a> {
        EvidenceQualificationBindingV1 {
            run_id: &batch.run_id, query_invocation_id: "invocation-projection",
            domain_id: "web", subject_scope_sha256: C, environment_revision_sha256: D,
            tool_profile_sha256: E, query_sha256: &batch.query_sha256,
            candidate_set_sha256: &batch.candidate_set_sha256, candidate_evidence_ids: ids,
            decision_need_sha256: A, purpose_sha256: B, qualification_policy_sha256: C,
            lifecycle_snapshot_sha256: D, verifier_profile_sha256: F, now_unix: 100,
        }
    }

    fn token(binding: &EvidenceQualificationBindingV1<'_>, dispositions: &[EvidenceDispositionV1])
        -> VerifiedEvidenceQualificationV1 {
        assert_eq!(binding.candidate_evidence_ids.len(), dispositions.len());
        let mut receipt = EvidenceQualificationReceiptV1 {
            run_id: binding.run_id.into(), query_invocation_id: binding.query_invocation_id.into(),
            domain_id: binding.domain_id.into(), subject_scope_sha256: binding.subject_scope_sha256.into(),
            environment_revision_sha256: binding.environment_revision_sha256.into(),
            tool_profile_sha256: binding.tool_profile_sha256.into(), query_sha256: binding.query_sha256.into(),
            candidate_set_sha256: binding.candidate_set_sha256.into(),
            candidate_evidence_ids_sha256: ids_digest(binding.candidate_evidence_ids),
            decision_need_sha256: binding.decision_need_sha256.into(), purpose_sha256: binding.purpose_sha256.into(),
            qualification_policy_sha256: binding.qualification_policy_sha256.into(),
            lifecycle_snapshot_sha256: binding.lifecycle_snapshot_sha256.into(),
            verifier_profile_sha256: binding.verifier_profile_sha256.into(),
            key_id: "projection-test-key".into(), issued_at_unix: 10, expires_at_unix: 200,
            entries: binding.candidate_evidence_ids.iter().zip(dispositions)
                .map(|(id, disposition)| EvidenceQualificationEntryV1 {
                    evidence_id: id.clone(), disposition: *disposition, basis_receipt_sha256: E.into(),
                }).collect(),
            signature: Vec::new(),
        };
        receipt.signature = receipt.signing_payload_sha256().into_bytes();
        VerifiedEvidenceQualificationV1::verify_and_bind(&receipt, binding, &TestVerifier)
            .expect("test qualification verifies")
    }

    fn expectation<'a>(batch: &'a CandidateEvidenceBatchV1) -> CandidateEvidenceBatchExpectationV1<'a> {
        CandidateEvidenceBatchExpectationV1 {
            candidate_id: &batch.candidate_id, backend_id: &batch.backend_id,
            backend_revision_sha256: &batch.backend_revision_sha256,
            query_scope: &batch.query_scope, query_purpose: &batch.query_purpose,
        }
    }

    #[test]
    fn only_independently_admissible_items_are_projected() {
        let batch = make_batch(&[record("a", 1, vec![]), record("b", 2, vec![]), record("c", 3, vec![])]);
        let ids = ids(&batch); let binding = binding(&batch, &ids);
        let token = token(&binding, &[EvidenceDispositionV1::Admissible,
            EvidenceDispositionV1::Rejected, EvidenceDispositionV1::Indeterminate]);
        let result = project_qualified_candidate_evidence_v1(&batch, &token, &binding, &expectation(&batch)).unwrap();
        let CandidateEvidenceProjectionV1::Qualified(projected) = result else { panic!("expected qualified subset"); };
        assert_eq!(projected.evidence().len(), 1);
        assert_eq!(projected.evidence()[0].evidence_id, "a");
        assert_ne!(projected.qualified_candidate_set_sha256(), projected.source_candidate_set_sha256());
    }

    #[test]
    fn query_scope_mismatch_is_rejected() {
        let batch = make_batch(&[record("a", 1, vec![])]);
        let ids = ids(&batch); let binding = binding(&batch, &ids);
        let token = token(&binding, &[EvidenceDispositionV1::Admissible]);
        let mut expected = expectation(&batch); expected.query_scope = "tenant:other";
        assert_eq!(project_qualified_candidate_evidence_v1(&batch, &token, &binding, &expected),
            Err(CandidateEvidenceProjectionRejectV1::BatchIdentityMismatch));
    }

    #[test]
    fn stale_batch_digest_cannot_survive_public_field_mutation() {
        let mut batch = make_batch(&[record("a", 1, vec![])]);
        batch.query_purpose = "answer:other".into();
        let ids = ids(&batch); let binding = binding(&batch, &ids);
        let token = token(&binding, &[EvidenceDispositionV1::Admissible]);
        assert_eq!(project_qualified_candidate_evidence_v1(&batch, &token, &binding, &expectation(&batch)),
            Err(CandidateEvidenceProjectionRejectV1::BatchNotCanonical));
    }

    #[test]
    fn binding_id_set_must_equal_the_actual_batch_id_set() {
        let batch = make_batch(&[record("a", 1, vec![])]);
        let wrong_ids = vec!["other".to_owned()]; let binding = binding(&batch, &wrong_ids);
        let token = token(&binding, &[EvidenceDispositionV1::Admissible]);
        assert_eq!(project_qualified_candidate_evidence_v1(&batch, &token, &binding, &expectation(&batch)),
            Err(CandidateEvidenceProjectionRejectV1::Qualification(EvidenceQualificationRejectV1::BindingMismatch)));
    }

    #[test]
    fn descendants_of_a_disqualified_parent_are_removed() {
        let batch = make_batch(&[record("root", 1, vec![]),
            record("child", 2, vec!["root".into()]),
            record("grandchild", 3, vec!["child".into()])]);
        let ids = ids(&batch); let binding = binding(&batch, &ids);
        let token = token(&binding, &[EvidenceDispositionV1::Admissible,
            EvidenceDispositionV1::Rejected, EvidenceDispositionV1::Admissible]);
        let result = project_qualified_candidate_evidence_v1(&batch, &token, &binding, &expectation(&batch)).unwrap();
        let CandidateEvidenceProjectionV1::Qualified(projected) = result else { panic!("root should remain"); };
        assert_eq!(projected.evidence().iter().map(|item| item.evidence_id.as_str()).collect::<Vec<_>>(), vec!["root"]);
    }

    #[test]
    fn missing_parent_causes_explicit_dependency_abstention() {
        let batch = make_batch(&[record("derived", 1, vec!["missing".into()])]);
        let ids = ids(&batch); let binding = binding(&batch, &ids);
        let token = token(&binding, &[EvidenceDispositionV1::Admissible]);
        let result = project_qualified_candidate_evidence_v1(&batch, &token, &binding, &expectation(&batch)).unwrap();
        let CandidateEvidenceProjectionV1::Abstained(abstention) = result else { panic!("must abstain"); };
        assert_eq!(abstention.reason(), CandidateEvidenceAbstentionReasonV1::DependencyClosureRemovedAllEvidence);
    }

    #[test]
    fn admissible_dependency_cycles_are_rejected() {
        let batch = make_batch(&[record("a", 1, vec!["b".into()]), record("b", 2, vec!["a".into()])]);
        let ids = ids(&batch); let binding = binding(&batch, &ids);
        let token = token(&binding, &[EvidenceDispositionV1::Admissible, EvidenceDispositionV1::Admissible]);
        assert_eq!(project_qualified_candidate_evidence_v1(&batch, &token, &binding, &expectation(&batch)),
            Err(CandidateEvidenceProjectionRejectV1::DependencyCycle));
    }

    #[test]
    fn projected_evidence_expires_at_later_consumption() {
        let batch = make_batch(&[record("a", 1, vec![])]);
        let ids = ids(&batch);
        let initial_binding = binding(&batch, &ids);
        let token = token(&initial_binding, &[EvidenceDispositionV1::Admissible]);
        let result = project_qualified_candidate_evidence_v1(
            &batch, &token, &initial_binding, &expectation(&batch),
        ).unwrap();
        let CandidateEvidenceProjectionV1::Qualified(projected) = result else {
            panic!("expected qualified projection");
        };
        let mut expired_binding = binding(&batch, &ids);
        expired_binding.now_unix = 200;
        assert_eq!(
            projected.validate_at_use(&expired_binding),
            Err(EvidenceQualificationRejectV1::ReceiptExpired)
        );
    }

    #[test]
    fn projected_evidence_cannot_be_reused_for_a_different_decision_context() {
        let batch = make_batch(&[record("a", 1, vec![])]);
        let ids = ids(&batch);
        let initial_binding = binding(&batch, &ids);
        let token = token(&initial_binding, &[EvidenceDispositionV1::Admissible]);
        let result = project_qualified_candidate_evidence_v1(
            &batch, &token, &initial_binding, &expectation(&batch),
        ).unwrap();
        let CandidateEvidenceProjectionV1::Qualified(projected) = result else {
            panic!("expected qualified projection");
        };
        let mut changed_binding = binding(&batch, &ids);
        changed_binding.purpose_sha256 = A;
        assert_eq!(
            projected.validate_at_use(&changed_binding),
            Err(EvidenceQualificationRejectV1::BindingMismatch)
        );
    }

    #[test]
    fn projected_evidence_rejects_consuming_clock_rollback() {
        let batch = make_batch(&[record("a", 1, vec![])]);
        let ids = ids(&batch);
        let initial_binding = binding(&batch, &ids);
        let token = token(&initial_binding, &[EvidenceDispositionV1::Admissible]);
        let result = project_qualified_candidate_evidence_v1(
            &batch, &token, &initial_binding, &expectation(&batch),
        ).unwrap();
        let CandidateEvidenceProjectionV1::Qualified(projected) = result else {
            panic!("expected qualified projection");
        };
        let mut rolled_back_binding = binding(&batch, &ids);
        rolled_back_binding.now_unix = 99;
        assert_eq!(
            projected.validate_at_use(&rolled_back_binding),
            Err(EvidenceQualificationRejectV1::ReceiptNotYetValid)
        );
    }

    #[test]
    fn no_admissible_evidence_is_never_reported_as_success() {
        let batch = make_batch(&[record("a", 1, vec![]), record("b", 2, vec![])]);
        let ids = ids(&batch); let binding = binding(&batch, &ids);
        let token = token(&binding, &[EvidenceDispositionV1::Rejected, EvidenceDispositionV1::Indeterminate]);
        let result = project_qualified_candidate_evidence_v1(&batch, &token, &binding, &expectation(&batch)).unwrap();
        let CandidateEvidenceProjectionV1::Abstained(abstention) = result else { panic!("must abstain"); };
        assert_eq!(abstention.reason(), CandidateEvidenceAbstentionReasonV1::NoIndependentlyAdmissibleEvidence);
        assert_eq!(abstention.source_evidence_count(), 2);
    }
}
