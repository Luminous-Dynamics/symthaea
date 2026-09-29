//! Frontier-aware retrieval primitives. Retrieval ranks representations; it never
//! upgrades epistemic state or creates canonical evidence.
use crate::{sha256_hex, MemoryKind, MemoryProjectionRef, MemoryProvenance};

const RETRIEVAL_RECEIPT_DOMAIN: &[u8] = b"epistemic-retrieval-receipt:v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalRequestError { MissingHistoricalFrontier, EmptyHistoricalFrontier, WhitespaceOnlyHistoricalFrontier }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalMode { Historical, Live }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRetrievalRequest {
    pub mode: RetrievalMode, pub frontier_ref: Option<String>, pub query: String, pub max_results: usize,
}
impl MemoryRetrievalRequest {
    pub fn historical(frontier_ref: impl Into<String>, query: impl Into<String>, max_results: usize) -> Self {
        Self { mode: RetrievalMode::Historical, frontier_ref: Some(frontier_ref.into()), query: query.into(), max_results }
    }
    pub fn live(query: impl Into<String>, max_results: usize) -> Self {
        Self { mode: RetrievalMode::Live, frontier_ref: None, query: query.into(), max_results }
    }
    pub fn validate(&self) -> Result<(), RetrievalRequestError> {
        if self.mode == RetrievalMode::Historical {
            match self.frontier_ref.as_deref() {
                None => return Err(RetrievalRequestError::MissingHistoricalFrontier),
                Some("") => return Err(RetrievalRequestError::EmptyHistoricalFrontier),
                Some(frontier) if frontier.trim().is_empty() => {
                    return Err(RetrievalRequestError::WhitespaceOnlyHistoricalFrontier)
                }
                Some(_) => {}
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrontierEligibility { Eligible, Ineligible, Unknown }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalExclusion { PostFrontier, FrontierUnknown, MissingHistoricalFrontier }

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRetrievalCandidate {
    pub projection: MemoryProjectionRef, pub provenance: MemoryProvenance,
    pub retrieval_score: f64, pub freshness: f64, pub frontier_eligibility: FrontierEligibility,
}
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievedMemory {
    pub canonical_identity: String, pub representations: Vec<MemoryRetrievalCandidate>,
    pub provenance_families: Vec<String>, pub best_retrieval_score: f64,
}
#[derive(Debug, Clone, PartialEq)]
pub struct ExcludedMemory { pub canonical_identity: String, pub reason: RetrievalExclusion }
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRetrievalReceipt {
    pub mode: RetrievalMode, pub frontier_ref: Option<String>, pub query: String, pub max_results: usize,
    pub selected: Vec<String>, pub selected_representation_digests: Vec<(String, String)>,
    pub excluded: Vec<ExcludedMemory>, pub provenance_families: Vec<String>,
    pub retrieval_profile_versions: Vec<String>,
    pub receipt_digest: String,
}


/// Receipt integrity validation failures. Integrity is not semantic truth or authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiptVerificationError {
    Canonicalization(ReceiptCanonicalizationError),
    DigestMismatch,
    DuplicateSelectedIdentity,
    DuplicateRepresentationBinding,
    DuplicateExclusion,
    DuplicateRetrievalProfileVersion,
    EmptyRepresentationDigest,
    EmptyRetrievalProfileVersion,
    WhitespaceOnlyRetrievalProfileVersion,
    UnselectedRepresentationIdentity,
}

/// A receipt whose canonical digest and internal selection bindings were checked.
/// This is an integrity marker only; it does not certify source authenticity or claim truth.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedRetrievalReceipt(MemoryRetrievalReceipt);

impl VerifiedRetrievalReceipt {
    pub fn receipt(&self) -> &MemoryRetrievalReceipt { &self.0 }
}

impl MemoryRetrievalReceipt {
    /// Verify the receipt's canonical digest and basic referential integrity.
    pub fn verify(&self) -> Result<VerifiedRetrievalReceipt, ReceiptVerificationError> {
        let expected_digest = self
            .canonical_digest()
            .map_err(ReceiptVerificationError::Canonicalization)?;
        if self.receipt_digest.is_empty() || self.receipt_digest != expected_digest {
            return Err(ReceiptVerificationError::DigestMismatch);
        }
        let mut selected = self.selected.clone();
        selected.sort();
        selected.dedup();
        if selected.len() != self.selected.len() {
            return Err(ReceiptVerificationError::DuplicateSelectedIdentity);
        }
        let mut bindings = self.selected_representation_digests.clone();
        bindings.sort();
        for pair in &bindings {
            if pair.1.is_empty() {
                return Err(ReceiptVerificationError::EmptyRepresentationDigest);
            }
        }
        if bindings.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ReceiptVerificationError::DuplicateRepresentationBinding);
        }
        if bindings.iter().any(|(identity, _)| !selected.contains(identity)) {
            return Err(ReceiptVerificationError::UnselectedRepresentationIdentity);
        }
        let mut exclusions = self.excluded.clone();
        exclusions.sort_by(|a, b| {
            a.canonical_identity
                .cmp(&b.canonical_identity)
                .then_with(|| (a.reason as u8).cmp(&(b.reason as u8)))
        });
        if exclusions.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ReceiptVerificationError::DuplicateExclusion);
        }
        let mut profiles = self.retrieval_profile_versions.clone();
        if profiles.iter().any(|profile| profile.is_empty()) {
            return Err(ReceiptVerificationError::EmptyRetrievalProfileVersion);
        }
        if profiles.iter().any(|profile| profile.trim().is_empty()) {
            return Err(ReceiptVerificationError::WhitespaceOnlyRetrievalProfileVersion);
        }
        profiles.sort();
        if profiles.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ReceiptVerificationError::DuplicateRetrievalProfileVersion);
        }
        Ok(VerifiedRetrievalReceipt(self.clone()))
    }

    /// Canonical, deterministic encoding of the receipt's retrieval contract and selection.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ReceiptCanonicalizationError> {
        let mut out = RETRIEVAL_RECEIPT_DOMAIN.to_vec();
        out.push(match self.mode { RetrievalMode::Historical => 0, RetrievalMode::Live => 1 });
        put_optional(&mut out, self.frontier_ref.as_deref())?;
        put_string(&mut out, &self.query)?;
        put_u32(&mut out, self.max_results)?;
        let mut selected = self.selected.clone();
        selected.sort();
        selected.dedup();
        put_strings(&mut out, &selected)?;
        let mut bindings = self.selected_representation_digests.clone();
        bindings.sort();
        put_u32(&mut out, bindings.len())?;
        for (identity, digest) in bindings {
            put_string(&mut out, &identity)?;
            put_string(&mut out, &digest)?;
        }
        let mut excluded = self.excluded.clone();
        excluded.sort_by(|a,b| a.canonical_identity.cmp(&b.canonical_identity).then_with(|| (a.reason as u8).cmp(&(b.reason as u8))));
        put_u32(&mut out, excluded.len())?;
        for item in excluded {
            put_string(&mut out, &item.canonical_identity)?;
            out.push(match item.reason { RetrievalExclusion::PostFrontier=>0, RetrievalExclusion::FrontierUnknown=>1, RetrievalExclusion::MissingHistoricalFrontier=>2 });
        }
        let mut families=self.provenance_families.clone(); families.sort(); families.dedup(); put_strings(&mut out,&families)?;
        let mut profiles=self.retrieval_profile_versions.clone(); profiles.sort(); profiles.dedup(); put_strings(&mut out,&profiles)?;
        Ok(out)
    }
    pub fn canonical_digest(&self) -> Result<String, ReceiptCanonicalizationError> { Ok(sha256_hex(&self.canonical_bytes()?)) }
    /// Compatibility-only receipt enrichment. Prefer `RetrievalEngine::execute`, which owns
    /// the execution profile and returns a verified receipt. This method remains public
    /// temporarily for migration of lower-level callers and must not be used as a
    /// reasoning authorization step.
    /// Internal sealing operation used by the retrieval execution boundary.
    ///
    /// Keeping this fallible and crate-visible prevents the execution engine from
    /// relying on a public compatibility API while preserving the legacy method
    /// for lower-level migration callers.
    pub(crate) fn seal_with_retrieval_profile_versions(
        mut self,
        versions: impl IntoIterator<Item = String>,
    ) -> Result<Self, ReceiptCanonicalizationError> {
        self.retrieval_profile_versions = versions.into_iter().collect();
        self.retrieval_profile_versions.sort();
        self.retrieval_profile_versions.dedup();
        self.receipt_digest = self.canonical_digest()?;
        Ok(self)
    }

    /// Compatibility-only receipt enrichment. Prefer `RetrievalEngine::execute`, which owns
    /// the execution profile and returns a verified receipt. This method remains public
    /// temporarily for migration of lower-level callers and must not be used as a
    /// reasoning authorization step.
    #[deprecated(note = "prefer RetrievalEngine::execute so retrieval provenance is owned by the execution boundary")]
    pub fn with_retrieval_profile_versions(self, versions: impl IntoIterator<Item = String>) -> Self {
        self.seal_with_retrieval_profile_versions(versions)
            .expect("receipt canonicalization overflow")
    }
    pub fn is_self_consistent(&self) -> bool { !self.receipt_digest.is_empty() && self.canonical_digest().map(|d| self.receipt_digest == d).unwrap_or(false) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiptCanonicalizationError {
    LengthOverflow,
}

fn put_u32(out: &mut Vec<u8>, n: usize) -> Result<(), ReceiptCanonicalizationError> {
    let value = u32::try_from(n).map_err(|_| ReceiptCanonicalizationError::LengthOverflow)?;
    out.extend_from_slice(&value.to_be_bytes());
    Ok(())
}
fn put_string(out: &mut Vec<u8>, value: &str) -> Result<(), ReceiptCanonicalizationError> {
    put_u32(out, value.len())?;
    out.extend_from_slice(value.as_bytes());
    Ok(())
}
fn put_optional(out: &mut Vec<u8>, value: Option<&str>) -> Result<(), ReceiptCanonicalizationError> {
    match value { Some(v)=>{out.push(1);put_string(out,v)},None=>{out.push(0);Ok(())} }
}
fn put_strings(out: &mut Vec<u8>, values: &[String]) -> Result<(), ReceiptCanonicalizationError> {
    put_u32(out, values.len())?;
    for value in values { put_string(out, value)?; }
    Ok(())
}

/// Fallible entry point for untrusted or externally constructed requests.
pub fn try_retrieve(
    request: &MemoryRetrievalRequest,
    candidates: impl IntoIterator<Item = MemoryRetrievalCandidate>,
) -> Result<(Vec<RetrievedMemory>, MemoryRetrievalReceipt), RetrievalRequestError> {
    request.validate()?;
    Ok(retrieve_validated(request, candidates))
}

/// Convenience entry point for requests built and validated by the caller.
/// Use try_retrieve at trust boundaries; invalid historical requests return an error.
pub fn retrieve(
    request: &MemoryRetrievalRequest,
    candidates: impl IntoIterator<Item = MemoryRetrievalCandidate>,
) -> (Vec<RetrievedMemory>, MemoryRetrievalReceipt) {
    try_retrieve(request, candidates).expect("invalid retrieval request; use try_retrieve for fallible input")
}

fn retrieve_validated(
    request: &MemoryRetrievalRequest,
    candidates: impl IntoIterator<Item = MemoryRetrievalCandidate>,
) -> (Vec<RetrievedMemory>, MemoryRetrievalReceipt) {
    let mut eligible = Vec::new();
    let mut excluded = Vec::new();
    for candidate in candidates {
        match request.mode {
            RetrievalMode::Live => eligible.push(candidate),
            RetrievalMode::Historical => match candidate.frontier_eligibility {
                FrontierEligibility::Eligible => eligible.push(candidate),
                FrontierEligibility::Ineligible => excluded.push(ExcludedMemory { canonical_identity: candidate.projection.canonical_identity.clone(), reason: RetrievalExclusion::PostFrontier }),
                FrontierEligibility::Unknown => excluded.push(ExcludedMemory { canonical_identity: candidate.projection.canonical_identity.clone(), reason: RetrievalExclusion::FrontierUnknown }),
            }
        }
    }
    eligible.sort_by(|a,b| b.retrieval_score.total_cmp(&a.retrieval_score)
        .then_with(|| a.projection.projection_identity_digest().cmp(&b.projection.projection_identity_digest())));
    let mut groups: Vec<RetrievedMemory> = Vec::new();
    for candidate in eligible {
        if let Some(group) = groups.iter_mut().find(|g| g.canonical_identity == candidate.projection.canonical_identity) {
            group.best_retrieval_score = group.best_retrieval_score.max(candidate.retrieval_score);
            if let Some(family) = candidate.provenance.provenance_identity() {
                if !group.provenance_families.iter().any(|f| f == family) {
                    group.provenance_families.push(family.to_owned()); group.provenance_families.sort();
                }
            }
            group.representations.push(candidate);
        } else {
            let family = candidate.provenance.provenance_identity().map(str::to_owned);
            groups.push(RetrievedMemory { canonical_identity: candidate.projection.canonical_identity.clone(),
                best_retrieval_score: candidate.retrieval_score, provenance_families: family.into_iter().collect(), representations: vec![candidate] });
        }
    }
    groups.sort_by(|a,b| b.best_retrieval_score.total_cmp(&a.best_retrieval_score).then_with(|| a.canonical_identity.cmp(&b.canonical_identity)));
    groups.truncate(request.max_results);
    let mut selected = groups.iter().map(|g| g.canonical_identity.clone()).collect::<Vec<_>>(); selected.sort();
    let mut families = groups.iter().flat_map(|g| g.provenance_families.iter().cloned()).collect::<Vec<_>>(); families.sort(); families.dedup();
    excluded.sort_by(|a,b| a.canonical_identity.cmp(&b.canonical_identity).then_with(|| (a.reason as u8).cmp(&(b.reason as u8))));
    let mut selected_representation_digests = groups.iter()
        .flat_map(|g| g.representations.iter().map(|candidate| {
            (g.canonical_identity.clone(), candidate.projection.representation_digest.clone())
        }))
        .collect::<Vec<_>>();
    selected_representation_digests.sort();
    let mut receipt = MemoryRetrievalReceipt {
        mode: request.mode,
        frontier_ref: request.frontier_ref.clone(),
        query: request.query.clone(),
        max_results: request.max_results,
        selected,
        selected_representation_digests,
        excluded,
        provenance_families: families,
        retrieval_profile_versions: Vec::new(),
        receipt_digest: String::new(),
    };
    receipt.receipt_digest = receipt.canonical_digest().expect("receipt canonicalization overflow");
    (groups, receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn candidate(id:&str, kind:MemoryKind, family:&str, score:f64, eligibility:FrontierEligibility)->MemoryRetrievalCandidate {
        MemoryRetrievalCandidate {
            projection:MemoryProjectionRef::new(id,kind,"test-v1",id.as_bytes(),Some("frontier:1".into())),
            provenance:MemoryProvenance { memory_id:format!("memory:{id}:{family}:{score}"),memory_kind:kind,created_at:"2026-09-29T00:00:00Z".into(),
                source_event:None,canonical_artifact_ref:Some(id.into()),statement_ref:None,provenance_family:Some(family.into()),epistemic_state:Some("Observed".into()),
                claim_ceiling:Some("source-scoped".into()),frontier_ref:Some("frontier:1".into()),derivation_ref:None,model_ref:None,retrieval_index_ref:None },
            retrieval_score:score,freshness:score,frontier_eligibility:eligibility
        }
    }
    #[test] fn seven_representations_are_one_semantic_object() {
        let cs=[MemoryKind::Working,MemoryKind::Episodic,MemoryKind::Semantic,MemoryKind::Procedural,MemoryKind::KnowledgeGraph,MemoryKind::Vector,MemoryKind::Hdc].into_iter().map(|k|candidate("claim:a",k,"family:a",0.5,FrontierEligibility::Eligible)).collect::<Vec<_>>();
        let (g,_)=retrieve(&MemoryRetrievalRequest::historical("frontier:1","claim:a",10),cs); assert_eq!(g.len(),1); assert_eq!(g[0].representations.len(),7); assert_eq!(g[0].provenance_families,vec!["family:a"]);
    }
    #[test] fn historical_frontier_filters_unknown_and_later_items() {
        let cs=vec![candidate("old",MemoryKind::Semantic,"a",0.9,FrontierEligibility::Eligible),candidate("new",MemoryKind::Semantic,"b",1.0,FrontierEligibility::Ineligible),candidate("unknown",MemoryKind::Semantic,"c",1.1,FrontierEligibility::Unknown)];
        let (g,r)=retrieve(&MemoryRetrievalRequest::historical("f:1","x",10),cs); assert_eq!(g.len(),1); assert_eq!(r.excluded.len(),2);
    }
    #[test] fn provenance_families_remain_distinct() {
        let cs=vec![candidate("x",MemoryKind::Semantic,"a",0.8,FrontierEligibility::Eligible),candidate("x",MemoryKind::Vector,"b",0.7,FrontierEligibility::Eligible)];
        let (g,_)=retrieve(&MemoryRetrievalRequest::historical("f:1","x",10),cs); assert_eq!(g[0].provenance_families,vec!["a","b"]);
    }
    #[test] fn score_changes_ranking_not_identity() {
        let (g,_)=retrieve(&MemoryRetrievalRequest::historical("f:1","x",10),vec![candidate("a",MemoryKind::Semantic,"a",0.1,FrontierEligibility::Eligible),candidate("b",MemoryKind::Semantic,"b",0.9,FrontierEligibility::Eligible)]);
        assert_eq!(g[0].canonical_identity,"b"); assert_eq!(g[1].canonical_identity,"a");
    }
    #[test] fn invalid_historical_request_is_rejected() {
        let r=MemoryRetrievalRequest{mode:RetrievalMode::Historical,frontier_ref:None,query:"x".into(),max_results:1};
        assert_eq!(try_retrieve(&r,Vec::new()),Err(RetrievalRequestError::MissingHistoricalFrontier));
        assert_eq!(MemoryRetrievalRequest::historical("","x",1).validate(),Err(RetrievalRequestError::EmptyHistoricalFrontier));
    }
    #[test] fn receipt_is_self_consistent_and_binds_selection_metadata() {
        let (_g, receipt)=retrieve(&MemoryRetrievalRequest::historical("f:1","x",10), vec![candidate("x",MemoryKind::Semantic,"a",0.8,FrontierEligibility::Eligible)]);
        assert!(receipt.is_self_consistent());
        assert!(receipt.verify().is_ok());
        assert!(!receipt.selected_representation_digests.is_empty());
        let mut tampered=receipt.clone();
        tampered.query="tampered".into();
        assert!(!tampered.is_self_consistent());
        assert_eq!(tampered.verify(), Err(ReceiptVerificationError::DigestMismatch));
        let profiled = receipt.clone().with_retrieval_profile_versions(vec!["retrieval-profile:v1".into()]);
        assert!(profiled.is_self_consistent());
        assert_ne!(profiled.receipt_digest, receipt.receipt_digest);
    }
    #[test]
    fn duplicate_profile_versions_are_rejected_even_if_digest_is_recomputed() {
        let (_groups, receipt)=retrieve(&MemoryRetrievalRequest::historical("f:1","x",10), vec![candidate("x",MemoryKind::Semantic,"a",0.8,FrontierEligibility::Eligible)]);
        let mut duplicated=receipt;
        duplicated.retrieval_profile_versions = vec!["algorithm:v1".into(), "algorithm:v1".into()];
        duplicated.receipt_digest=duplicated.canonical_digest().unwrap();
        assert_eq!(duplicated.verify(), Err(ReceiptVerificationError::DuplicateRetrievalProfileVersion));
    }

    #[test]
    fn empty_profile_version_is_rejected() {
        let (_groups, receipt)=retrieve(&MemoryRetrievalRequest::historical("f:1","x",10), vec![candidate("x",MemoryKind::Semantic,"a",0.8,FrontierEligibility::Eligible)]);
        let mut malformed=receipt;
        malformed.retrieval_profile_versions = vec!["".into()];
        malformed.receipt_digest=malformed.canonical_digest().unwrap();
        assert_eq!(malformed.verify(), Err(ReceiptVerificationError::EmptyRetrievalProfileVersion));
    }

    #[test] fn verification_digest_is_not_producer_authentication() {
        let (_groups, receipt)=retrieve(&MemoryRetrievalRequest::historical("f:1","x",10), vec![candidate("x",MemoryKind::Semantic,"a",0.8,FrontierEligibility::Eligible)]);
        // A party able to rewrite the receipt can recompute an unkeyed digest.
        // Verification therefore means internal consistency, not origin authentication.
        let mut rewritten=receipt;
        rewritten.query="different query".into();
        rewritten.receipt_digest=rewritten.canonical_digest().unwrap();
        assert!(rewritten.verify().is_ok());
    }

    #[test] fn duplicate_selected_identity_is_rejected_even_if_digest_is_recomputed() {
        let (_groups, receipt)=retrieve(&MemoryRetrievalRequest::historical("f:1","x",10), vec![candidate("x",MemoryKind::Semantic,"a",0.8,FrontierEligibility::Eligible)]);
        let mut duplicated=receipt;
        duplicated.selected.push("x".into());
        duplicated.receipt_digest=duplicated.canonical_digest().unwrap();
        assert_eq!(duplicated.verify(), Err(ReceiptVerificationError::DuplicateSelectedIdentity));
    }

    #[test] fn duplicate_representation_binding_is_rejected_even_if_digest_is_recomputed() {
        let (_groups, receipt)=retrieve(&MemoryRetrievalRequest::historical("f:1","x",10), vec![candidate("x",MemoryKind::Semantic,"a",0.8,FrontierEligibility::Eligible)]);
        let mut duplicated=receipt;
        duplicated.selected_representation_digests.push(duplicated.selected_representation_digests[0].clone());
        duplicated.receipt_digest=duplicated.canonical_digest().unwrap();
        assert_eq!(duplicated.verify(), Err(ReceiptVerificationError::DuplicateRepresentationBinding));
    }

    #[test] fn duplicate_exclusion_is_rejected_even_if_digest_is_recomputed() {
        let (_groups, receipt)=retrieve(&MemoryRetrievalRequest::historical("f:1","x",10), vec![
            candidate("x",MemoryKind::Semantic,"a",0.8,FrontierEligibility::Eligible),
            candidate("new",MemoryKind::Semantic,"b",0.7,FrontierEligibility::Ineligible),
        ]);
        let mut duplicated=receipt;
        duplicated.excluded.push(duplicated.excluded[0].clone());
        duplicated.receipt_digest=duplicated.canonical_digest().unwrap();
        assert_eq!(duplicated.verify(), Err(ReceiptVerificationError::DuplicateExclusion));
    }

    #[test] fn empty_representation_digest_is_rejected() {
        let (_groups, receipt)=retrieve(&MemoryRetrievalRequest::historical("f:1","x",10), vec![candidate("x",MemoryKind::Semantic,"a",0.8,FrontierEligibility::Eligible)]);
        let mut empty=receipt;
        empty.selected_representation_digests[0].1.clear();
        empty.receipt_digest=empty.canonical_digest().unwrap();
        assert_eq!(empty.verify(), Err(ReceiptVerificationError::EmptyRepresentationDigest));
    }

    #[test] fn live_mode_needs_no_frontier {
        let (g,r)=retrieve(&MemoryRetrievalRequest::live("x",10),vec![candidate("live",MemoryKind::Vector,"f",0.8,FrontierEligibility::Unknown)]);
        assert_eq!(g.len(),1); assert!(r.excluded.is_empty());
    }
}