//! A normalized, provenance-preserving boundary between retrieval and reasoning.
//!
//! EvidenceView is a read model, not an evidence assessment or authority object.
//! It preserves backend representations and their lineage while making it explicit
//! that retrieval metadata cannot upgrade epistemic status.

use crate::{
    sha256_hex, ExcludedMemory, MemoryRetrievalReceipt, RetrievedMemory,
    RetrievalMode,
};

const EVIDENCE_VIEW_DOMAIN: &[u8] = b"epistemic-evidence-view:v1\0";

#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceViewRepresentation {
    pub projection_identity_digest: String,
    pub representation_digest: String,
    pub provenance_family: Option<String>,
    pub epistemic_state: Option<String>,
    pub claim_ceiling: Option<String>,
    pub retrieval_score: f64,
    pub freshness: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceViewItem {
    pub canonical_identity: String,
    pub representations: Vec<EvidenceViewRepresentation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceView {
    pub mode: RetrievalMode,
    pub frontier_ref: Option<String>,
    pub query: String,
    pub items: Vec<EvidenceViewItem>,
    pub excluded: Vec<ExcludedMemory>,
    /// Digest binds this normalized view; it does not certify truth or support.
    pub view_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceViewError {
    NonFiniteRetrievalMetadata,
    ReceiptMismatch,
    StringTooLong,
}

fn put_bytes(out: &mut Vec<u8>, value: &[u8]) -> Result<(), EvidenceViewError> {
    let len = u32::try_from(value.len()).map_err(|_| EvidenceViewError::StringTooLong)?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(value);
    Ok(())
}

fn put_string(out: &mut Vec<u8>, value: &str) -> Result<(), EvidenceViewError> {
    put_bytes(out, value.as_bytes())
}

fn put_optional(out: &mut Vec<u8>, value: Option<&str>) -> Result<(), EvidenceViewError> {
    match value {
        Some(value) => { out.push(1); put_string(out, value) }
        None => { out.push(0); Ok(()) }
    }
}

fn put_float(out: &mut Vec<u8>, value: f64) -> Result<(), EvidenceViewError> {
    if !value.is_finite() { return Err(EvidenceViewError::NonFiniteRetrievalMetadata); }
    // Normalize negative zero so equivalent ranking metadata has one encoding.
    let normalized = if value == 0.0 { 0.0 } else { value };
    out.extend_from_slice(&normalized.to_bits().to_be_bytes());
    Ok(())
}

impl EvidenceView {
    pub fn from_retrieval(
        groups: &[RetrievedMemory],
        receipt: &MemoryRetrievalReceipt,
    ) -> Result<Self, EvidenceViewError> {
        let mut items = groups.iter().map(|group| {
            let mut representations = group.representations.iter().map(|candidate| {
                EvidenceViewRepresentation {
                    projection_identity_digest: candidate.projection.projection_identity_digest(),
                    representation_digest: candidate.projection.representation_digest.clone(),
                    provenance_family: candidate.provenance.provenance_identity().map(str::to_owned),
                    epistemic_state: candidate.provenance.epistemic_state.clone(),
                    claim_ceiling: candidate.provenance.claim_ceiling.clone(),
                    retrieval_score: candidate.retrieval_score,
                    freshness: candidate.freshness,
                }
            }).collect::<Vec<_>>();
            representations.sort_by(|a, b| {
                a.projection_identity_digest.cmp(&b.projection_identity_digest)
                    .then_with(|| a.representation_digest.cmp(&b.representation_digest))
                    .then_with(|| a.provenance_family.cmp(&b.provenance_family))
            });
            EvidenceViewItem { canonical_identity: group.canonical_identity.clone(), representations }
        }).collect::<Vec<_>>();
        items.sort_by(|a, b| a.canonical_identity.cmp(&b.canonical_identity));

        // A view must describe exactly the selection attested by its retrieval receipt.
        // Do not silently accept a stale/mismatched receipt as metadata.
        let mut item_ids = items.iter().map(|item| item.canonical_identity.clone()).collect::<Vec<_>>();
        item_ids.sort();
        item_ids.dedup();
        let mut receipt_ids = receipt.selected.clone();
        receipt_ids.sort();
        receipt_ids.dedup();
        if item_ids != receipt_ids {
            return Err(EvidenceViewError::ReceiptMismatch);
        }
        if !receipt.is_self_consistent() {
            return Err(EvidenceViewError::ReceiptMismatch);
        }
        let mut representation_digests = groups.iter()
            .flat_map(|group| group.representations.iter().map(|candidate| candidate.projection.representation_digest.clone()))
            .collect::<Vec<_>>();
        representation_digests.sort();
        let mut receipt_representation_digests = receipt.selected_representation_digests.clone();
        receipt_representation_digests.sort();
        if representation_digests != receipt_representation_digests {
            return Err(EvidenceViewError::ReceiptMismatch);
        }
        let mut view = Self {
            mode: receipt.mode,
            frontier_ref: receipt.frontier_ref.clone(),
            query: receipt.query.clone(),
            items,
            excluded: receipt.excluded.clone(),
            view_digest: String::new(),
        };
        view.excluded.sort_by(|a, b| a.canonical_identity.cmp(&b.canonical_identity)
            .then_with(|| (a.reason as u8).cmp(&(b.reason as u8))));
        view.view_digest = sha256_hex(&view.canonical_bytes()?);
        Ok(view)
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, EvidenceViewError> {
        let mut out = EVIDENCE_VIEW_DOMAIN.to_vec();
        out.push(match self.mode { RetrievalMode::Historical => 0, RetrievalMode::Live => 1 });
        put_optional(&mut out, self.frontier_ref.as_deref())?;
        put_string(&mut out, &self.query)?;
        out.extend_from_slice(&(u32::try_from(self.items.len()).map_err(|_| EvidenceViewError::StringTooLong)?).to_be_bytes());
        for item in &self.items {
            put_string(&mut out, &item.canonical_identity)?;
            out.extend_from_slice(&(u32::try_from(item.representations.len()).map_err(|_| EvidenceViewError::StringTooLong)?).to_be_bytes());
            for rep in &item.representations {
                put_string(&mut out, &rep.projection_identity_digest)?;
                put_string(&mut out, &rep.representation_digest)?;
                put_optional(&mut out, rep.provenance_family.as_deref())?;
                put_optional(&mut out, rep.epistemic_state.as_deref())?;
                put_optional(&mut out, rep.claim_ceiling.as_deref())?;
                put_float(&mut out, rep.retrieval_score)?;
                put_float(&mut out, rep.freshness)?;
            }
        }
        out.extend_from_slice(&(u32::try_from(self.excluded.len()).map_err(|_| EvidenceViewError::StringTooLong)?).to_be_bytes());
        for excluded in &self.excluded {
            put_string(&mut out, &excluded.canonical_identity)?;
            out.push(match excluded.reason {
                crate::RetrievalExclusion::PostFrontier => 0,
                crate::RetrievalExclusion::FrontierUnknown => 1,
                crate::RetrievalExclusion::MissingHistoricalFrontier => 2,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{retrieve, FrontierEligibility, MemoryKind, MemoryProjectionRef, MemoryProvenance, MemoryRetrievalCandidate, MemoryRetrievalRequest};

    fn candidate(id: &str, score: f64, family: &str) -> MemoryRetrievalCandidate {
        MemoryRetrievalCandidate {
            projection: MemoryProjectionRef::new(id, MemoryKind::Semantic, "semantic-v1", id.as_bytes(), Some("f:1".into())),
            provenance: MemoryProvenance {
                memory_id: format!("m:{id}"), memory_kind: MemoryKind::Semantic,
                created_at: "2026-09-29T00:00:00Z".into(), source_event: None,
                canonical_artifact_ref: Some(id.into()), statement_ref: None,
                provenance_family: Some(family.into()), epistemic_state: Some("Observed".into()),
                claim_ceiling: Some("source-scoped".into()), frontier_ref: Some("f:1".into()),
                derivation_ref: None, model_ref: None, retrieval_index_ref: None,
            },
            retrieval_score: score, freshness: 0.5, frontier_eligibility: FrontierEligibility::Eligible,
        }
    }

    #[test]
    fn view_preserves_independent_families_without_assessing_them() {
        let (groups, receipt) = retrieve(&MemoryRetrievalRequest::historical("f:1", "q", 5),
            vec![candidate("claim:x", 0.7, "family:a"), candidate("claim:x", 0.6, "family:b")]);
        let view = EvidenceView::from_retrieval(&groups, &receipt).unwrap();
        assert_eq!(view.items.len(), 1);
        assert_eq!(view.items[0].representations.len(), 2);
        assert_eq!(view.items[0].representations.iter().filter_map(|r| r.provenance_family.as_deref()).collect::<Vec<_>>(), vec!["family:a", "family:b"]);
        assert!(!view.view_digest.is_empty());
    }

    #[test]
    fn view_digest_is_order_independent_for_equivalent_candidate_sets() {
        let request = MemoryRetrievalRequest::historical("f:1", "q", 5);
        let (a, ar) = retrieve(&request, vec![candidate("claim:a", 0.8, "family:a"), candidate("claim:b", 0.6, "family:b")]);
        let (b, br) = retrieve(&request, vec![candidate("claim:b", 0.6, "family:b"), candidate("claim:a", 0.8, "family:a")]);
        assert_eq!(EvidenceView::from_retrieval(&a, &ar).unwrap().view_digest, EvidenceView::from_retrieval(&b, &br).unwrap().view_digest);
    }

    #[test]
    fn non_finite_retrieval_metadata_is_rejected() {
        let (groups, receipt) = retrieve(&MemoryRetrievalRequest::live("q", 5), vec![candidate("claim:x", f64::NAN, "family:a")]);
        assert_eq!(EvidenceView::from_retrieval(&groups, &receipt), Err(EvidenceViewError::NonFiniteRetrievalMetadata));
    }

    #[test]
    fn mismatched_receipt_selection_is_rejected() {
        let (groups, mut receipt) = retrieve(
            &MemoryRetrievalRequest::historical("f:1", "q", 5),
            vec![candidate("claim:x", 0.7, "family:a")],
        );
        receipt.selected = vec!["claim:other".into()];
        assert_eq!(
            EvidenceView::from_retrieval(&groups, &receipt),
            Err(EvidenceViewError::ReceiptMismatch)
        );
    }
}
