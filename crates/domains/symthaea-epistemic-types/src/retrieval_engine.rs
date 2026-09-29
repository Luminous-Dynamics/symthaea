//! Retrieval execution boundary that owns retrieval provenance.
//!
//! The compatibility-level `retrieve` API remains available for lower-level
//! callers, but reasoning-facing integrations should use `RetrievalEngine`.
//! The engine owns the execution profile and seals it into the receipt before
//! returning a `VerifiedRetrievalReceipt`.

use crate::{
    try_retrieve, MemoryRetrievalCandidate, MemoryRetrievalRequest, MemoryRetrievalReceipt,
    ReceiptVerificationError, RetrievalRequestError, RetrievedMemory, VerifiedRetrievalReceipt,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetrievalExecutionProfile {
    pub algorithm_version: String,
    pub ranking_profile_version: String,
    pub frontier_semantics_version: String,
    pub normalization_version: String,
    /// Optional immutable source/index snapshot identifier. Its presence identifies
    /// a replay target; it does not by itself prove byte-for-byte reproducibility.
    /// The value is stored without a `snapshot:` prefix; the receipt encoding adds
    /// that namespace exactly once.
    pub snapshot_ref: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalProfileError {
    EmptyAlgorithmVersion,
    EmptyRankingProfileVersion,
    EmptyFrontierSemanticsVersion,
    EmptyNormalizationVersion,
    EmptySnapshotRef,
}

impl RetrievalExecutionProfile {
    pub fn new(
        algorithm_version: impl Into<String>,
        ranking_profile_version: impl Into<String>,
        frontier_semantics_version: impl Into<String>,
        normalization_version: impl Into<String>,
    ) -> Result<Self, RetrievalProfileError> {
        let profile = Self {
            algorithm_version: algorithm_version.into(),
            ranking_profile_version: ranking_profile_version.into(),
            frontier_semantics_version: frontier_semantics_version.into(),
            normalization_version: normalization_version.into(),
            snapshot_ref: None,
        };
        if profile.algorithm_version.trim().is_empty() {
            return Err(RetrievalProfileError::EmptyAlgorithmVersion);
        }
        if profile.ranking_profile_version.trim().is_empty() {
            return Err(RetrievalProfileError::EmptyRankingProfileVersion);
        }
        if profile.frontier_semantics_version.trim().is_empty() {
            return Err(RetrievalProfileError::EmptyFrontierSemanticsVersion);
        }
        if profile.normalization_version.trim().is_empty() {
            return Err(RetrievalProfileError::EmptyNormalizationVersion);
        }
        Ok(profile)
    }

    /// Attach a retrieval-source/index snapshot identifier.
    ///
    /// This is deliberately separate from the algorithm/ranking versions: a stable
    /// algorithm can still produce different results against different snapshots.
    /// This identifies the replay target only; callers must not interpret it as a
    /// content hash or as proof that the same bytes can be reconstructed.
    pub fn with_snapshot_ref(mut self, snapshot_ref: impl Into<String>) -> Result<Self, RetrievalProfileError> {
        let snapshot_ref = snapshot_ref.into();
        if snapshot_ref.trim().is_empty() {
            return Err(RetrievalProfileError::EmptySnapshotRef);
        }
        self.snapshot_ref = Some(snapshot_ref);
        Ok(self)
    }

    fn receipt_versions(&self) -> Vec<String> {
        let mut versions = vec![
            format!("algorithm:{}", self.algorithm_version),
            format!("ranking:{}", self.ranking_profile_version),
            format!("frontier-semantics:{}", self.frontier_semantics_version),
            format!("normalization:{}", self.normalization_version),
        ];
        if let Some(snapshot_ref) = &self.snapshot_ref {
            versions.push(format!("snapshot:{}", snapshot_ref));
        }
        versions
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalExecutionError {
    Request(RetrievalRequestError),
    ReceiptIntegrity(ReceiptVerificationError),
}

impl From<RetrievalRequestError> for RetrievalExecutionError {
    fn from(value: RetrievalRequestError) -> Self {
        Self::Request(value)
    }
}

/// The complete output of one retrieval execution.
///
/// The receipt is already verified when returned. The execution profile is
/// retained alongside it so callers do not have to reconstruct provenance from
/// free-form metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedRetrievalExecution {
    pub groups: Vec<RetrievedMemory>,
    pub receipt: VerifiedRetrievalReceipt,
    pub profile: RetrievalExecutionProfile,
}

impl VerifiedRetrievalExecution {
    pub fn receipt(&self) -> &MemoryRetrievalReceipt {
        self.receipt.receipt()
    }
}

/// Retrieval authority: execution metadata is owned by the engine, not callers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetrievalEngine {
    profile: RetrievalExecutionProfile,
}

impl RetrievalEngine {
    pub fn new(profile: RetrievalExecutionProfile) -> Self {
        Self { profile }
    }

    pub fn profile(&self) -> &RetrievalExecutionProfile {
        &self.profile
    }

    /// Execute retrieval and seal the engine's provenance into the receipt.
    ///
    /// The returned receipt has already crossed the integrity boundary.
    pub fn execute(
        &self,
        request: &MemoryRetrievalRequest,
        candidates: impl IntoIterator<Item = MemoryRetrievalCandidate>,
    ) -> Result<VerifiedRetrievalExecution, RetrievalExecutionError> {
        let (groups, receipt) = try_retrieve(request, candidates)?;
        let receipt = receipt
            .seal_with_retrieval_profile_versions(self.profile.receipt_versions())
            .map_err(|error| RetrievalExecutionError::ReceiptIntegrity(ReceiptVerificationError::Canonicalization(error)))?;
        let receipt = receipt.verify().map_err(RetrievalExecutionError::ReceiptIntegrity)?;
        Ok(VerifiedRetrievalExecution {
            groups,
            receipt,
            profile: self.profile.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FrontierEligibility, MemoryKind, MemoryProjectionRef, MemoryProvenance};

    fn candidate(id: &str) -> MemoryRetrievalCandidate {
        MemoryRetrievalCandidate {
            projection: MemoryProjectionRef::new(
                id, MemoryKind::Semantic, "semantic-v1", id.as_bytes(), Some("frontier:1".into()),
            ),
            provenance: MemoryProvenance {
                memory_id: format!("memory:{id}"),
                memory_kind: MemoryKind::Semantic,
                created_at: "2026-09-29T00:00:00Z".into(),
                source_event: None,
                canonical_artifact_ref: Some(id.into()),
                statement_ref: None,
                provenance_family: Some("family:a".into()),
                epistemic_state: Some("Observed".into()),
                claim_ceiling: Some("source-scoped".into()),
                frontier_ref: Some("frontier:1".into()),
                derivation_ref: None, model_ref: None, retrieval_index_ref: None,
            },
            retrieval_score: 0.8, freshness: 0.8, frontier_eligibility: FrontierEligibility::Eligible,
        }
    }

    #[test]
    fn engine_owns_all_four_provenance_dimensions() {
        let profile = RetrievalExecutionProfile::new("retrieval:v3", "cosine-v2", "frontier:v2", "utf8-v1").unwrap();
        let execution = RetrievalEngine::new(profile.clone())
            .execute(&MemoryRetrievalRequest::historical("frontier:1", "claim:x", 5), vec![candidate("claim:x")])
            .unwrap();
        assert_eq!(execution.profile, profile);
        assert!(execution.receipt().retrieval_profile_versions.contains(&"algorithm:retrieval:v3".to_string()));
        assert!(execution.receipt().retrieval_profile_versions.contains(&"ranking:cosine-v2".to_string()));
        assert!(execution.receipt().retrieval_profile_versions.contains(&"frontier-semantics:frontier:v2".to_string()));
        assert!(execution.receipt().retrieval_profile_versions.contains(&"normalization:utf8-v1".to_string()));
        assert!(execution.receipt.verify().is_ok());
    }

    #[test]
    fn snapshot_ref_is_bound_when_present() {
        let profile = RetrievalExecutionProfile::new("a", "b", "c", "d").unwrap()
            .with_snapshot_ref("2026-09-29T00:00:00Z").unwrap();
        let execution = RetrievalEngine::new(profile)
            .execute(&MemoryRetrievalRequest::historical("frontier:1", "claim:x", 5), vec![candidate("claim:x")])
            .unwrap();
        assert!(execution.receipt().retrieval_profile_versions.contains(&"snapshot:2026-09-29T00:00:00Z".to_string()));
    }

    #[test]
    fn whitespace_only_profile_components_are_rejected() {
        assert_eq!(
            RetrievalExecutionProfile::new("  ", "ranking:v1", "frontier:v1", "normalization:v1"),
            Err(RetrievalProfileError::EmptyAlgorithmVersion)
        );
        assert_eq!(
            RetrievalExecutionProfile::new("algorithm:v1", "  ", "frontier:v1", "normalization:v1"),
            Err(RetrievalProfileError::EmptyRankingProfileVersion)
        );
        assert_eq!(
            RetrievalExecutionProfile::new("algorithm:v1", "ranking:v1", "\t", "normalization:v1"),
            Err(RetrievalProfileError::EmptyFrontierSemanticsVersion)
        );
        assert_eq!(
            RetrievalExecutionProfile::new("algorithm:v1", "ranking:v1", "frontier:v1", "\n"),
            Err(RetrievalProfileError::EmptyNormalizationVersion)
        );
    }

    #[test]
    fn whitespace_only_snapshot_ref_is_rejected() {
        let profile = RetrievalExecutionProfile::new("a", "b", "c", "d").unwrap();
        assert_eq!(profile.with_snapshot_ref("   "), Err(RetrievalProfileError::EmptySnapshotRef));
    }

    #[test]
    fn empty_snapshot_ref_is_rejected() {
        let profile = RetrievalExecutionProfile::new("a", "b", "c", "d").unwrap();
        assert_eq!(profile.with_snapshot_ref(""), Err(RetrievalProfileError::EmptySnapshotRef));
    }

    #[test]
    fn profile_versions_change_receipt_digest() {
        let base = RetrievalExecutionProfile::new("a", "b", "c", "d").unwrap();
        let changed = RetrievalExecutionProfile::new("a", "b2", "c", "d").unwrap();
        let request = MemoryRetrievalRequest::historical("frontier:1", "claim:x", 5);
        let first = RetrievalEngine::new(base).execute(&request, vec![candidate("claim:x")]).unwrap();
        let second = RetrievalEngine::new(changed).execute(&request, vec![candidate("claim:x")]).unwrap();
        assert_ne!(first.receipt().receipt_digest, second.receipt().receipt_digest);
    }

    #[test]
    fn invalid_request_is_rejected_without_panicking() {
        let profile = RetrievalExecutionProfile::new("a", "b", "c", "d").unwrap();
        let request = MemoryRetrievalRequest {
            mode: crate::RetrievalMode::Historical,
            frontier_ref: None,
            query: "claim:x".into(),
            max_results: 5,
        };
        let result = RetrievalEngine::new(profile).execute(&request, Vec::<MemoryRetrievalCandidate>::new());
        assert_eq!(result, Err(RetrievalExecutionError::Request(RetrievalRequestError::MissingHistoricalFrontier)));
    }

    #[test]
    fn verified_execution_feeds_reasoning_boundary() {
        let profile = RetrievalExecutionProfile::new("a", "b", "c", "d").unwrap();
        let execution = RetrievalEngine::new(profile)
            .execute(
                &MemoryRetrievalRequest::historical("frontier:1", "claim:x", 5),
                vec![candidate("claim:x")],
            )
            .unwrap();
        let view = crate::EvidenceView::from_verified_retrieval(&execution.groups, &execution.receipt).unwrap();
        assert_eq!(view.items.len(), 1);
        assert_eq!(view.items[0].canonical_identity, "claim:x");
    }

    #[test]
    fn empty_profile_component_is_rejected() {
        assert_eq!(
            RetrievalExecutionProfile::new("", "ranking:v1", "frontier:v1", "normalization:v1"),
            Err(RetrievalProfileError::EmptyAlgorithmVersion)
        );
    }
}
