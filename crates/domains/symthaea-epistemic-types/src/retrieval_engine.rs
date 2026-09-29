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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalProfileError {
    EmptyAlgorithmVersion,
    EmptyRankingProfileVersion,
    EmptyFrontierSemanticsVersion,
    EmptyNormalizationVersion,
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
        };
        if profile.algorithm_version.is_empty() {
            return Err(RetrievalProfileError::EmptyAlgorithmVersion);
        }
        if profile.ranking_profile_version.is_empty() {
            return Err(RetrievalProfileError::EmptyRankingProfileVersion);
        }
        if profile.frontier_semantics_version.is_empty() {
            return Err(RetrievalProfileError::EmptyFrontierSemanticsVersion);
        }
        if profile.normalization_version.is_empty() {
            return Err(RetrievalProfileError::EmptyNormalizationVersion);
        }
        Ok(profile)
    }

    fn receipt_versions(&self) -> Vec<String> {
        vec![
            format!("algorithm:{}", self.algorithm_version),
            format!("ranking:{}", self.ranking_profile_version),
            format!("frontier-semantics:{}", self.frontier_semantics_version),
            format!("normalization:{}", self.normalization_version),
        ]
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
        let receipt = receipt.with_retrieval_profile_versions(self.profile.receipt_versions());
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
    fn profile_versions_change_receipt_digest() {
        let base = RetrievalExecutionProfile::new("a", "b", "c", "d").unwrap();
        let changed = RetrievalExecutionProfile::new("a", "b2", "c", "d").unwrap();
        let request = MemoryRetrievalRequest::historical("frontier:1", "claim:x", 5);
        let first = RetrievalEngine::new(base).execute(&request, vec![candidate("claim:x")]).unwrap();
        let second = RetrievalEngine::new(changed).execute(&request, vec![candidate("claim:x")]).unwrap();
        assert_ne!(first.receipt().receipt_digest, second.receipt().receipt_digest);
    }

    #[test]
    fn empty_profile_component_is_rejected() {
        assert_eq!(
            RetrievalExecutionProfile::new("", "ranking:v1", "frontier:v1", "normalization:v1"),
            Err(RetrievalProfileError::EmptyAlgorithmVersion)
        );
    }
}
