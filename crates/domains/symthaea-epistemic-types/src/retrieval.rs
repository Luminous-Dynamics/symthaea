//! Frontier-aware retrieval primitives.
//!
//! Retrieval is deliberately modeled as a projection operation. It can rank and
//! filter representations, but it cannot upgrade epistemic state or create
//! canonical evidence.

use crate::{MemoryKind, MemoryProjectionRef, MemoryProvenance};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalMode {
    Historical,
    Live,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRetrievalRequest {
    pub mode: RetrievalMode,
    pub frontier_ref: Option<String>,
    pub query: String,
    pub max_results: usize,
}

impl MemoryRetrievalRequest {
    pub fn historical(frontier_ref: impl Into<String>, query: impl Into<String>, max_results: usize) -> Self {
        Self {
            mode: RetrievalMode::Historical,
            frontier_ref: Some(frontier_ref.into()),
            query: query.into(),
            max_results,
        }
    }

    pub fn live(query: impl Into<String>, max_results: usize) -> Self {
        Self { mode: RetrievalMode::Live, frontier_ref: None, query: query.into(), max_results }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrontierEligibility {
    Eligible,
    Ineligible,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalExclusion {
    PostFrontier,
    FrontierUnknown,
    MissingHistoricalFrontier,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRetrievalCandidate {
    pub projection: MemoryProjectionRef,
    pub provenance: MemoryProvenance,
    pub retrieval_score: f64,
    pub freshness: f64,
    pub frontier_eligibility: FrontierEligibility,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RetrievedMemory {
    pub canonical_identity: String,
    pub representations: Vec<MemoryRetrievalCandidate>,
    pub provenance_families: Vec<String>,
    pub best_retrieval_score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExcludedMemory {
    pub canonical_identity: String,
    pub reason: RetrievalExclusion,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRetrievalReceipt {
    pub mode: RetrievalMode,
    pub frontier_ref: Option<String>,
    pub query: String,
    pub selected: Vec<String>,
    pub excluded: Vec<ExcludedMemory>,
    pub provenance_families: Vec<String>,
}

pub fn retrieve(
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
                FrontierEligibility::Ineligible => excluded.push(ExcludedMemory {
                    canonical_identity: candidate.projection.canonical_identity.clone(),
                    reason: RetrievalExclusion::PostFrontier,
                }),
                FrontierEligibility::Unknown => excluded.push(ExcludedMemory {
                    canonical_identity: candidate.projection.canonical_identity.clone(),
                    reason: RetrievalExclusion::FrontierUnknown,
                }),
            },
        }
    }

    eligible.sort_by(|a, b| {
        b.retrieval_score
            .total_cmp(&a.retrieval_score)
            .then_with(|| a.projection.projection_identity_digest().cmp(&b.projection.projection_identity_digest()))
    });

    let mut groups: Vec<RetrievedMemory> = Vec::new();
    for candidate in eligible {
        if let Some(group) = groups.iter_mut().find(|g| g.canonical_identity == candidate.projection.canonical_identity) {
            group.best_retrieval_score = group.best_retrieval_score.max(candidate.retrieval_score);
            if let Some(family) = candidate.provenance.provenance_identity() {
                if !group.provenance_families.iter().any(|f| f == family) {
                    group.provenance_families.push(family.to_owned());
                    group.provenance_families.sort();
                }
            }
            group.representations.push(candidate);
        } else {
            let family = candidate.provenance.provenance_identity().map(str::to_owned);
            groups.push(RetrievedMemory {
                canonical_identity: candidate.projection.canonical_identity.clone(),
                best_retrieval_score: candidate.retrieval_score,
                provenance_families: family.into_iter().collect(),
                representations: vec![candidate],
            });
        }
    }

    groups.sort_by(|a, b| {
        b.best_retrieval_score
            .total_cmp(&a.best_retrieval_score)
            .then_with(|| a.canonical_identity.cmp(&b.canonical_identity))
    });
    groups.truncate(request.max_results);

    let mut selected = groups.iter().map(|g| g.canonical_identity.clone()).collect::<Vec<_>>();
    selected.sort();

    let mut families = groups.iter().flat_map(|g| g.provenance_families.iter().cloned()).collect::<Vec<_>>();
    families.sort();
    families.dedup();

    excluded.sort_by(|a, b| a.canonical_identity.cmp(&b.canonical_identity));

    let receipt = MemoryRetrievalReceipt {
        mode: request.mode,
        frontier_ref: request.frontier_ref.clone(),
        query: request.query.clone(),
        selected,
        excluded,
        provenance_families: families,
    };

    (groups, receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryProjectionRef;

    fn candidate(id: &str, kind: MemoryKind, family: &str, score: f64, eligibility: FrontierEligibility) -> MemoryRetrievalCandidate {
        MemoryRetrievalCandidate {
            projection: MemoryProjectionRef::new(id, kind, "test-v1", id.as_bytes(), Some("frontier:1".into())),
            provenance: MemoryProvenance {
                memory_id: format!("memory:{id}:{family}:{score}"),
                memory_kind: kind,
                created_at: "2026-09-29T00:00:00Z".into(),
                source_event: None,
                canonical_artifact_ref: Some(id.into()),
                statement_ref: None,
                provenance_family: Some(family.into()),
                epistemic_state: Some("Observed".into()),
                claim_ceiling: Some("source-scoped".into()),
                frontier_ref: Some("frontier:1".into()),
                derivation_ref: None,
                model_ref: None,
                retrieval_index_ref: None,
            },
            retrieval_score: score,
            freshness: score,
            frontier_eligibility: eligibility,
        }
    }

    #[test]
    fn seven_representations_are_one_semantic_object() {
        let mut cs = Vec::new();
        for kind in [MemoryKind::Working, MemoryKind::Episodic, MemoryKind::Semantic, MemoryKind::Procedural, MemoryKind::KnowledgeGraph, MemoryKind::Vector, MemoryKind::Hdc] {
            cs.push(candidate("claim:a", kind, "family:a", 0.5, FrontierEligibility::Eligible));
        }
        let (groups, _) = retrieve(&MemoryRetrievalRequest::historical("frontier:1", "claim:a", 10), cs);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].representations.len(), 7);
        assert_eq!(groups[0].provenance_families, vec!["family:a"]);
    }

    #[test]
    fn post_frontier_candidates_are_fail_closed() {
        let cs = vec![
            candidate("claim:old", MemoryKind::Semantic, "family:a", 0.9, FrontierEligibility::Eligible),
            candidate("claim:new", MemoryKind::Semantic, "family:b", 1.0, FrontierEligibility::Ineligible),
            candidate("claim:unknown", MemoryKind::Semantic, "family:c", 1.1, FrontierEligibility::Unknown),
        ];
        let (groups, receipt) = retrieve(&MemoryRetrievalRequest::historical("frontier:1", "x", 10), cs);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].canonical_identity, "claim:old");
        assert_eq!(receipt.excluded.len(), 2);
    }

    #[test]
    fn independent_provenance_families_remain_distinct() {
        let cs = vec![
            candidate("claim:a", MemoryKind::Semantic, "family:a", 0.8, FrontierEligibility::Eligible),
            candidate("claim:a", MemoryKind::Vector, "family:b", 0.7, FrontierEligibility::Eligible),
        ];
        let (groups, _) = retrieve(&MemoryRetrievalRequest::historical("frontier:1", "x", 10), cs);
        assert_eq!(groups[0].provenance_families, vec!["family:a", "family:b"]);
    }

    #[test]
    fn retrieval_score_only_changes_ranking() {
        let low = candidate("claim:a", MemoryKind::Semantic, "family:a", 0.1, FrontierEligibility::Eligible);
        let high = candidate("claim:b", MemoryKind::Semantic, "family:b", 0.9, FrontierEligibility::Eligible);
        let (groups, _) = retrieve(&MemoryRetrievalRequest::historical("frontier:1", "x", 10), vec![low, high]);
        assert_eq!(groups[0].canonical_identity, "claim:b");
        assert_eq!(groups[1].canonical_identity, "claim:a");
    }

    #[test]
    fn live_mode_does_not_require_frontier() {
        let c = candidate("claim:live", MemoryKind::Vector, "family:live", 0.8, FrontierEligibility::Unknown);
        let (groups, receipt) = retrieve(&MemoryRetrievalRequest::live("x", 10), vec![c]);
        assert_eq!(groups.len(), 1);
        assert!(receipt.excluded.is_empty());
    }
}
