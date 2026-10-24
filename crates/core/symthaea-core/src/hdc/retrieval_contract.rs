// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-aware retrieval decisions.
//!
//! Search scores rank candidates; they do not independently prove identity,
//! freshness, provenance, or semantic relevance. This module keeps those
//! contracts distinct and uses explicit abstention when required evidence is
//! absent. Evidence statuses are assertions supplied by upstream verifiers;
//! this module does not implement cryptography or verify provenance itself.

/// Result status reported by an independent evidence verifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceStatus {
    /// The corresponding verifier ran and its check passed.
    Verified,
    /// The check was not available or did not run.
    Missing,
    /// The verifier ran and the check failed.
    Failed,
}

/// The operation requested by the caller.
#[derive(Debug, Clone, PartialEq)]
pub enum RetrievalMode {
    /// Resolve a requested record identity, optionally at an exact version.
    ExactIdentity {
        /// Stable record ID requested by the caller.
        requested_record_id: String,
        /// Required version; None means any current verified version.
        requested_version: Option<String>,
    },
    /// Return a semantically relevant result only with independent verification.
    Semantic {
        /// Minimum accepted similarity, in the inclusive range [0, 1].
        minimum_score: f32,
        /// Minimum top-candidate margin, in the inclusive range [0, 1].
        minimum_margin: f32,
    },
    /// Reject low-scoring queries; abstain on ambiguous high-scoring queries.
    OpenSet {
        /// Minimum similarity to remain a plausible in-set candidate.
        minimum_score: f32,
        /// Minimum top-candidate margin required to resolve ambiguity.
        minimum_margin: f32,
    },
}

/// Evidence accompanying one proposed candidate.
///
/// Score and margin must use a documented, fixed convention. The margin is
/// normally top-1 minus top-2 similarity and must be in [0, 1]. Exact identity
/// decisions intentionally do not use either score. EvidenceStatus fields
/// must be populated by actual verifier results, not inferred from similarity.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateEvidence {
    /// Stable ID attached to the candidate record.
    pub record_id: String,
    /// Version attached to the candidate, if available.
    pub version: Option<String>,
    /// Candidate similarity in [0, 1], if measured.
    pub score: Option<f32>,
    /// Top-1 minus top-2 similarity in [0, 1], if measured.
    pub margin: Option<f32>,
    /// Whether the record's provenance chain was verified.
    pub provenance: EvidenceStatus,
    /// Whether the candidate's stored content/digest was verified.
    pub content_integrity: EvidenceStatus,
    /// Whether the record's validity/freshness was verified.
    pub freshness: EvidenceStatus,
    /// Whether an independent semantic relevance verifier accepted it.
    pub semantic_relevance: EvidenceStatus,
}

/// Machine-readable reasons for not making a confident acceptance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionReason {
    /// No candidate was returned by search.
    NoCandidate,
    /// Thresholds are non-finite or outside [0, 1].
    InvalidPolicy,
    /// Similarity was absent, non-finite, or outside [0, 1].
    InvalidScore,
    /// Margin was absent, non-finite, or outside [0, 1].
    InvalidMargin,
    /// The candidate ID differs from the requested ID.
    IdentityMismatch,
    /// Candidate version differs from the requested version.
    VersionMismatch,
    /// A requested version could not be checked.
    MissingVersion,
    /// Provenance verification did not run.
    MissingProvenance,
    /// Provenance verification failed.
    FailedProvenance,
    /// Content integrity verification did not run.
    MissingIntegrity,
    /// Content integrity verification failed.
    FailedIntegrity,
    /// Freshness/validity verification did not run.
    MissingFreshness,
    /// Freshness/validity verification failed.
    FailedFreshness,
    /// Candidate similarity is below the policy floor.
    BelowMinimumScore,
    /// Candidate margin is too small to resolve ambiguity.
    InsufficientMargin,
    /// Independent semantic relevance verification did not run.
    MissingSemanticVerification,
    /// Independent semantic relevance verification rejected the candidate.
    FailedSemanticVerification,
}

/// Explicit outcome of a retrieval decision.
#[derive(Debug, Clone, PartialEq)]
pub enum RetrievalDecision {
    /// Requested ID/version and all identity evidence checks passed.
    ExactIdentityVerified {
        /// Verified requested record ID.
        record_id: String,
        /// Verified candidate version.
        version: Option<String>,
    },
    /// Independent semantic verification and the policy gates passed.
    AcceptedSemantic {
        /// Candidate record ID.
        record_id: String,
        /// Measured similarity.
        score: f32,
        /// Measured score margin.
        margin: f32,
    },
    /// A plausible candidate exists, but evidence does not justify acceptance.
    CandidateOnly {
        /// Candidate record ID.
        record_id: String,
        /// Candidate score if it was valid and available.
        score: Option<f32>,
        /// Why the candidate remains tentative.
        reason: DecisionReason,
    },
    /// The requested identity or open-set policy found no acceptable match.
    NoMatch {
        /// Why no result was accepted.
        reason: DecisionReason,
    },
    /// Evidence is missing, contradictory, or insufficient to safely decide.
    Abstain {
        /// Why the decision was deferred.
        reason: DecisionReason,
    },
}

/// Apply a caller-selected retrieval contract to a candidate.
///
/// This function does not search the corpus and does not turn evidence labels
/// into cryptographic proofs. It enforces the boundary between candidate
/// ranking and acceptance. Upstream verifiers must produce the evidence
/// statuses; absent or failed mandatory checks never become implicit passes.
pub fn decide(
    mode: &RetrievalMode,
    candidate: Option<&CandidateEvidence>,
) -> RetrievalDecision {
    let Some(candidate) = candidate else {
        return RetrievalDecision::NoMatch {
            reason: DecisionReason::NoCandidate,
        };
    };

    match mode {
        RetrievalMode::ExactIdentity {
            requested_record_id,
            requested_version,
        } => decide_exact_identity(candidate, requested_record_id, requested_version.as_deref()),
        RetrievalMode::Semantic {
            minimum_score,
            minimum_margin,
        } => decide_semantic(candidate, *minimum_score, *minimum_margin, false),
        RetrievalMode::OpenSet {
            minimum_score,
            minimum_margin,
        } => decide_semantic(candidate, *minimum_score, *minimum_margin, true),
    }
}

fn valid_unit_interval(value: f32) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn decide_exact_identity(
    candidate: &CandidateEvidence,
    requested_id: &str,
    requested_version: Option<&str>,
) -> RetrievalDecision {
    if candidate.record_id != requested_id {
        return RetrievalDecision::NoMatch {
            reason: DecisionReason::IdentityMismatch,
        };
    }

    if let Some(requested) = requested_version {
        match candidate.version.as_deref() {
            Some(actual) if actual == requested => {}
            Some(_) => {
                return RetrievalDecision::NoMatch {
                    reason: DecisionReason::VersionMismatch,
                };
            }
            None => {
                return RetrievalDecision::Abstain {
                    reason: DecisionReason::MissingVersion,
                };
            }
        }
    }

    for (status, missing, failed) in [
        (
            candidate.content_integrity,
            DecisionReason::MissingIntegrity,
            DecisionReason::FailedIntegrity,
        ),
        (
            candidate.provenance,
            DecisionReason::MissingProvenance,
            DecisionReason::FailedProvenance,
        ),
        (
            candidate.freshness,
            DecisionReason::MissingFreshness,
            DecisionReason::FailedFreshness,
        ),
    ] {
        match status {
            EvidenceStatus::Verified => {}
            EvidenceStatus::Missing => return RetrievalDecision::Abstain { reason: missing },
            EvidenceStatus::Failed => return RetrievalDecision::Abstain { reason: failed },
        }
    }

    RetrievalDecision::ExactIdentityVerified {
        record_id: candidate.record_id.clone(),
        version: candidate.version.clone(),
    }
}

fn decide_semantic(
    candidate: &CandidateEvidence,
    minimum_score: f32,
    minimum_margin: f32,
    open_set: bool,
) -> RetrievalDecision {
    if !valid_unit_interval(minimum_score) || !valid_unit_interval(minimum_margin) {
        return RetrievalDecision::Abstain {
            reason: DecisionReason::InvalidPolicy,
        };
    }

    let score = match candidate.score {
        Some(score) if valid_unit_interval(score) => score,
        _ => {
            return RetrievalDecision::Abstain {
                reason: DecisionReason::InvalidScore,
            };
        }
    };
    let margin = match candidate.margin {
        Some(margin) if valid_unit_interval(margin) => margin,
        _ => {
            return RetrievalDecision::Abstain {
                reason: DecisionReason::InvalidMargin,
            };
        }
    };

    if score < minimum_score {
        return if open_set {
            RetrievalDecision::NoMatch {
                reason: DecisionReason::BelowMinimumScore,
            }
        } else {
            RetrievalDecision::CandidateOnly {
                record_id: candidate.record_id.clone(),
                score: Some(score),
                reason: DecisionReason::BelowMinimumScore,
            }
        };
    }

    if margin < minimum_margin {
        return if open_set {
            RetrievalDecision::Abstain {
                reason: DecisionReason::InsufficientMargin,
            }
        } else {
            RetrievalDecision::CandidateOnly {
                record_id: candidate.record_id.clone(),
                score: Some(score),
                reason: DecisionReason::InsufficientMargin,
            }
        };
    }

    match candidate.content_integrity {
        EvidenceStatus::Verified => {}
        EvidenceStatus::Missing => {
            return RetrievalDecision::Abstain {
                reason: DecisionReason::MissingIntegrity,
            };
        }
        EvidenceStatus::Failed => {
            return RetrievalDecision::Abstain {
                reason: DecisionReason::FailedIntegrity,
            };
        }
    }

    match candidate.semantic_relevance {
        EvidenceStatus::Verified => RetrievalDecision::AcceptedSemantic {
            record_id: candidate.record_id.clone(),
            score,
            margin,
        },
        EvidenceStatus::Missing => RetrievalDecision::CandidateOnly {
            record_id: candidate.record_id.clone(),
            score: Some(score),
            reason: DecisionReason::MissingSemanticVerification,
        },
        EvidenceStatus::Failed if open_set => RetrievalDecision::NoMatch {
            reason: DecisionReason::FailedSemanticVerification,
        },
        EvidenceStatus::Failed => RetrievalDecision::CandidateOnly {
            record_id: candidate.record_id.clone(),
            score: Some(score),
            reason: DecisionReason::FailedSemanticVerification,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate() -> CandidateEvidence {
        CandidateEvidence {
            record_id: "record-7".to_owned(),
            version: Some("v2".to_owned()),
            score: Some(0.99),
            margin: Some(0.25),
            provenance: EvidenceStatus::Verified,
            content_integrity: EvidenceStatus::Verified,
            freshness: EvidenceStatus::Verified,
            semantic_relevance: EvidenceStatus::Verified,
        }
    }

    #[test]
    fn exact_identity_accepts_verified_id_version_and_evidence() {
        let result = decide(
            &RetrievalMode::ExactIdentity {
                requested_record_id: "record-7".to_owned(),
                requested_version: Some("v2".to_owned()),
            },
            Some(&candidate()),
        );
        assert_eq!(
            result,
            RetrievalDecision::ExactIdentityVerified {
                record_id: "record-7".to_owned(),
                version: Some("v2".to_owned()),
            }
        );
    }

    #[test]
    fn perfect_similarity_cannot_override_wrong_identity() {
        let mut item = candidate();
        item.score = Some(1.0);
        item.margin = Some(1.0);
        let result = decide(
            &RetrievalMode::ExactIdentity {
                requested_record_id: "different-record".to_owned(),
                requested_version: Some("v2".to_owned()),
            },
            Some(&item),
        );
        assert_eq!(
            result,
            RetrievalDecision::NoMatch {
                reason: DecisionReason::IdentityMismatch,
            }
        );
    }

    #[test]
    fn exact_identity_abstains_when_version_is_missing_or_wrong() {
        let mut item = candidate();
        item.version = None;
        assert_eq!(
            decide(
                &RetrievalMode::ExactIdentity {
                    requested_record_id: "record-7".to_owned(),
                    requested_version: Some("v2".to_owned()),
                },
                Some(&item),
            ),
            RetrievalDecision::Abstain {
                reason: DecisionReason::MissingVersion,
            }
        );

        let mut item = candidate();
        item.version = Some("v1".to_owned());
        assert_eq!(
            decide(
                &RetrievalMode::ExactIdentity {
                    requested_record_id: "record-7".to_owned(),
                    requested_version: Some("v2".to_owned()),
                },
                Some(&item),
            ),
            RetrievalDecision::NoMatch {
                reason: DecisionReason::VersionMismatch,
            }
        );
    }

    #[test]
    fn exact_identity_abstains_on_missing_or_failed_integrity() {
        for status in [EvidenceStatus::Missing, EvidenceStatus::Failed] {
            let mut item = candidate();
            item.content_integrity = status;
            let expected = match status {
                EvidenceStatus::Missing => DecisionReason::MissingIntegrity,
                EvidenceStatus::Failed => DecisionReason::FailedIntegrity,
                EvidenceStatus::Verified => unreachable!(),
            };
            assert_eq!(
                decide(
                    &RetrievalMode::ExactIdentity {
                        requested_record_id: "record-7".to_owned(),
                        requested_version: None,
                    },
                    Some(&item),
                ),
                RetrievalDecision::Abstain { reason: expected }
            );
        }
    }

    #[test]
    fn semantic_acceptance_requires_independent_verification() {
        let mut item = candidate();
        item.semantic_relevance = EvidenceStatus::Missing;
        let result = decide(
            &RetrievalMode::Semantic {
                minimum_score: 0.8,
                minimum_margin: 0.1,
            },
            Some(&item),
        );
        assert_eq!(
            result,
            RetrievalDecision::CandidateOnly {
                record_id: "record-7".to_owned(),
                score: Some(0.99),
                reason: DecisionReason::MissingSemanticVerification,
            }
        );
    }

    #[test]
    fn semantic_acceptance_passes_only_when_gates_and_verifier_pass() {
        assert_eq!(
            decide(
                &RetrievalMode::Semantic {
                    minimum_score: 0.8,
                    minimum_margin: 0.1,
                },
                Some(&candidate()),
            ),
            RetrievalDecision::AcceptedSemantic {
                record_id: "record-7".to_owned(),
                score: 0.99,
                margin: 0.25,
            }
        );
    }

    #[test]
    fn open_set_rejects_low_score_but_abstains_on_near_tie() {
        let mut item = candidate();
        item.score = Some(0.2);
        assert_eq!(
            decide(
                &RetrievalMode::OpenSet {
                    minimum_score: 0.7,
                    minimum_margin: 0.1,
                },
                Some(&item),
            ),
            RetrievalDecision::NoMatch {
                reason: DecisionReason::BelowMinimumScore,
            }
        );

        item.score = Some(0.95);
        item.margin = Some(0.01);
        assert_eq!(
            decide(
                &RetrievalMode::OpenSet {
                    minimum_score: 0.7,
                    minimum_margin: 0.1,
                },
                Some(&item),
            ),
            RetrievalDecision::Abstain {
                reason: DecisionReason::InsufficientMargin,
            }
        );
    }

    #[test]
    fn invalid_nan_scores_and_invalid_policies_abstain() {
        let mut item = candidate();
        item.score = Some(f32::NAN);
        assert_eq!(
            decide(
                &RetrievalMode::Semantic {
                    minimum_score: 0.8,
                    minimum_margin: 0.1,
                },
                Some(&item),
            ),
            RetrievalDecision::Abstain {
                reason: DecisionReason::InvalidScore,
            }
        );

        item.score = Some(0.99);
        assert_eq!(
            decide(
                &RetrievalMode::Semantic {
                    minimum_score: f32::NAN,
                    minimum_margin: 0.1,
                },
                Some(&item),
            ),
            RetrievalDecision::Abstain {
                reason: DecisionReason::InvalidPolicy,
            }
        );
    }

    #[test]
    fn missing_candidate_returns_no_match() {
        assert_eq!(
            decide(
                &RetrievalMode::OpenSet {
                    minimum_score: 0.7,
                    minimum_margin: 0.1,
                },
                None,
            ),
            RetrievalDecision::NoMatch {
                reason: DecisionReason::NoCandidate,
            }
        );
    }
}
