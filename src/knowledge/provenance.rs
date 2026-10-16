// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence provenance and revocation policy primitives.
//!
//! This module deliberately separates source identity, byte integrity, source
//! lifecycle, claim-validity time, and semantic truth. It does not fetch sources,
//! verify signatures, establish that an issuer is authoritative, or prove a claim.
//! An external verifier must produce the integrity/status observations consumed
//! here. "EligibleForReview" means only that this record meets a configured
//! provenance policy; it is never a truth or causal-validity verdict.
//!
//! The types are not wired into the production retrieval path yet. That integration
//! should follow only after ownership of canonical provenance/status records and
//! persistence migrations are agreed, rather than duplicating a trust ledger here.

/// Unix timestamp in milliseconds. Using u64 avoids a new time dependency in the
/// low-level knowledge module; adapters are responsible for epoch conversion.
pub type ProvenanceTimestampMillis = u64;

/// Content digest for an immutable source revision.
///
/// Constructing this value does not hash any bytes. Policy evaluation validates
/// its supported algorithm and shape only; a separate verifier must compute and
/// compare the digest against the retrieved bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentDigest {
    pub algorithm: String,
    pub value: String,
}

impl ContentDigest {
    /// Describe a SHA-256 digest using its hexadecimal representation.
    pub fn sha256(hex: impl Into<String>) -> Self {
        Self {
            algorithm: "sha256".to_owned(),
            value: hex.into(),
        }
    }

    fn validation_error(&self) -> Option<ProvenanceAbstentionReason> {
        if !self.algorithm.eq_ignore_ascii_case("sha256") {
            return Some(ProvenanceAbstentionReason::UnsupportedDigestAlgorithm);
        }
        let is_sha256_hex = self.value.len() == 64
            && self.value.bytes().all(|byte| byte.is_ascii_hexdigit());
        if !is_sha256_hex {
            return Some(ProvenanceAbstentionReason::MalformedContentDigest);
        }
        None
    }
}

/// A reference to the exact source material used to derive an observation.
///
/// source_id identifies the source record, not its truth. revision_id and
/// content_digest are separate on purpose: a URL or revision label alone does
/// not prove which bytes were observed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceSourceReference {
    /// Stable, non-empty identifier for this source record.
    pub source_id: String,
    /// Stable identity of the publisher/issuer, when known.
    pub publisher_id: Option<String>,
    /// Source-specific immutable revision/version identifier, when available.
    pub revision_id: Option<String>,
    /// Digest of the exact source bytes or canonical source representation.
    pub content_digest: Option<ContentDigest>,
    /// Passage, page, JSON pointer, byte range, or other source-local locator.
    pub locator: Option<String>,
    /// When this exact source representation was retrieved/observed.
    pub retrieved_at_ms: Option<ProvenanceTimestampMillis>,
    /// Publisher-declared time; not a substitute for retrieval or valid time.
    pub published_at_ms: Option<ProvenanceTimestampMillis>,
    /// Reference to licence/rights metadata. The referenced terms are not
    /// interpreted by this module.
    pub rights_reference: Option<String>,
}

/// Result of an external integrity/authenticity check.
///
/// Verified records that a named verifier ran; it does not say that the
/// publisher is epistemically trustworthy or that the contained claim is true.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegrityStatus {
    /// No check is recorded.
    Unknown,
    /// An upstream verifier reports successful verification.
    Verified {
        method: String,
        checked_at_ms: ProvenanceTimestampMillis,
    },
    /// The observed bytes did not match the expected digest/signature.
    Mismatch {
        checked_at_ms: ProvenanceTimestampMillis,
        reason: String,
    },
    /// The verification attestation or key/credential was revoked.
    Revoked {
        checked_at_ms: ProvenanceTimestampMillis,
        reason: String,
    },
}

/// Current lifecycle observation for a source record or issuing credential.
///
/// Withdrawal/revocation is a status of the source/attestation. It does not erase
/// historical observations and does not, by itself, prove the claim is false.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceStatus {
    /// Current status has not been checked or is not known.
    Unknown,
    /// The latest source-status check says the record is active.
    Active {
        checked_at_ms: ProvenanceTimestampMillis,
    },
    /// The source has been withdrawn from current use.
    Withdrawn {
        checked_at_ms: ProvenanceTimestampMillis,
        reason: String,
    },
    /// The source/credential has been explicitly revoked.
    Revoked {
        checked_at_ms: ProvenanceTimestampMillis,
        reason: String,
    },
    /// A status check was attempted but could not establish current status.
    Unavailable {
        checked_at_ms: ProvenanceTimestampMillis,
        reason: String,
    },
}

/// Claim-level validity interval, distinct from when the source was published or
/// retrieved. Bounds are inclusive at valid_from_ms and exclusive at valid_until_ms.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ClaimValidity {
    pub valid_from_ms: Option<ProvenanceTimestampMillis>,
    pub valid_until_ms: Option<ProvenanceTimestampMillis>,
}

/// Provenance attached to one evidence item. The source record is a reference;
/// canonical issuer trust and revocation feeds should be owned by the appropriate
/// shared trust/provenance service, not copied as independent authority here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceProvenance {
    pub source: EvidenceSourceReference,
    pub integrity: IntegrityStatus,
    pub source_status: SourceStatus,
    pub claim_validity: ClaimValidity,
}

/// Fail-closed policy applied at a specific evaluation time.
///
/// This policy assesses provenance completeness and freshness only. It does not
/// compute semantic relevance, source credibility rankings, or claim truth.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenancePolicy {
    pub require_publisher_id: bool,
    pub require_revision_id: bool,
    pub require_content_digest: bool,
    pub require_locator: bool,
    pub require_retrieved_at: bool,
    pub require_rights_reference: bool,
    pub require_verified_integrity: bool,
    pub require_active_source: bool,
    /// Maximum age of the retrieval observation. None disables this age gate.
    pub max_retrieval_age_ms: Option<u64>,
    /// Maximum age of source-status and integrity checks. None disables age
    /// gates, but never bypasses explicit mismatch, withdrawal, or revocation.
    pub max_status_age_ms: Option<u64>,
}

impl ProvenancePolicy {
    /// Conservative baseline for evidence that may advance to review.
    ///
    /// The 30-day retrieval and 24-hour verification/status limits are explicit
    /// policy defaults, not empirically tuned constants. Benchmark them against
    /// the target domain before using this policy in a production decision path.
    pub fn strict() -> Self {
        Self {
            require_publisher_id: true,
            require_revision_id: true,
            require_content_digest: true,
            require_locator: true,
            require_retrieved_at: true,
            require_rights_reference: false,
            require_verified_integrity: true,
            require_active_source: true,
            max_retrieval_age_ms: Some(30 * 24 * 60 * 60 * 1_000),
            max_status_age_ms: Some(24 * 60 * 60 * 1_000),
        }
    }
}

impl Default for ProvenancePolicy {
    fn default() -> Self {
        Self::strict()
    }
}

/// Stable reason codes for a provenance-policy abstention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProvenanceAbstentionReason {
    MissingSourceId,
    MissingPublisherId,
    MissingRevisionId,
    MissingContentDigest,
    UnsupportedDigestAlgorithm,
    MalformedContentDigest,
    MissingLocator,
    MissingRetrievedAt,
    MissingRightsReference,
    PublicationTimeInFuture,
    RetrievalTimeInFuture,
    RetrievalTooOld,
    EmptyVerificationMethod,
    UnknownIntegrity,
    VerificationTimeInFuture,
    VerificationStale,
    IntegrityMismatch,
    IntegrityRevoked,
    UnknownSourceStatus,
    SourceStatusUnavailable,
    SourceStatusTimeInFuture,
    SourceStatusStale,
    SourceWithdrawn,
    SourceRevoked,
    InvalidClaimValidityInterval,
    ClaimNotYetValid,
    ClaimExpired,
}

/// Provenance gate outcome. Eligibility is deliberately phrased as "for review";
/// only a separate reasoning/evaluation step can assess support or truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProvenanceDecision {
    EligibleForReview,
    Abstain,
}

/// Result with deterministic, machine-readable reasons for an abstention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceAssessment {
    pub decision: ProvenanceDecision,
    pub reasons: Vec<ProvenanceAbstentionReason>,
}

impl ProvenanceAssessment {
    pub fn is_eligible_for_review(&self) -> bool {
        self.decision == ProvenanceDecision::EligibleForReview
    }
}

impl EvidenceProvenance {
    /// Evaluate this provenance record at now_ms.
    ///
    /// The function is deterministic and side-effect free. It never performs
    /// I/O, trusts a locator as proof, or silently converts unknown status into
    /// an active/verified state.
    pub fn assess(
        &self,
        policy: &ProvenancePolicy,
        now_ms: ProvenanceTimestampMillis,
    ) -> ProvenanceAssessment {
        use ProvenanceAbstentionReason as R;

        let mut reasons = Vec::new();
        let source = &self.source;

        if source.source_id.trim().is_empty() {
            reasons.push(R::MissingSourceId);
        }
        if policy.require_publisher_id && !has_text(source.publisher_id.as_deref()) {
            reasons.push(R::MissingPublisherId);
        }
        if policy.require_revision_id && !has_text(source.revision_id.as_deref()) {
            reasons.push(R::MissingRevisionId);
        }
        match &source.content_digest {
            None if policy.require_content_digest => reasons.push(R::MissingContentDigest),
            Some(digest) => {
                if let Some(reason) = digest.validation_error() {
                    reasons.push(reason);
                }
            }
            None => {}
        }
        if policy.require_locator && !has_text(source.locator.as_deref()) {
            reasons.push(R::MissingLocator);
        }
        if policy.require_rights_reference && !has_text(source.rights_reference.as_deref()) {
            reasons.push(R::MissingRightsReference);
        }

        match source.retrieved_at_ms {
            None if policy.require_retrieved_at => reasons.push(R::MissingRetrievedAt),
            Some(retrieved_at) => {
                if retrieved_at > now_ms {
                    reasons.push(R::RetrievalTimeInFuture);
                } else if policy
                    .max_retrieval_age_ms
                    .is_some_and(|max_age| now_ms - retrieved_at > max_age)
                {
                    reasons.push(R::RetrievalTooOld);
                }
            }
            None => {}
        }
        if source.published_at_ms.is_some_and(|published_at| published_at > now_ms) {
            reasons.push(R::PublicationTimeInFuture);
        }

        match &self.integrity {
            IntegrityStatus::Unknown if policy.require_verified_integrity => {
                reasons.push(R::UnknownIntegrity);
            }
            IntegrityStatus::Unknown => {}
            IntegrityStatus::Verified { method, checked_at_ms } => {
                if method.trim().is_empty() {
                    reasons.push(R::EmptyVerificationMethod);
                }
                if *checked_at_ms > now_ms {
                    reasons.push(R::VerificationTimeInFuture);
                } else if policy
                    .max_status_age_ms
                    .is_some_and(|max_age| now_ms - *checked_at_ms > max_age)
                {
                    reasons.push(R::VerificationStale);
                }
            }
            IntegrityStatus::Mismatch { .. } => reasons.push(R::IntegrityMismatch),
            IntegrityStatus::Revoked { .. } => reasons.push(R::IntegrityRevoked),
        }

        match &self.source_status {
            SourceStatus::Unknown if policy.require_active_source => {
                reasons.push(R::UnknownSourceStatus);
            }
            SourceStatus::Unknown => {}
            SourceStatus::Active { checked_at_ms } => {
                if *checked_at_ms > now_ms {
                    reasons.push(R::SourceStatusTimeInFuture);
                } else if policy
                    .max_status_age_ms
                    .is_some_and(|max_age| now_ms - *checked_at_ms > max_age)
                {
                    reasons.push(R::SourceStatusStale);
                }
            }
            SourceStatus::Withdrawn { .. } => reasons.push(R::SourceWithdrawn),
            SourceStatus::Revoked { .. } => reasons.push(R::SourceRevoked),
            SourceStatus::Unavailable { .. } if policy.require_active_source => {
                reasons.push(R::SourceStatusUnavailable);
            }
            SourceStatus::Unavailable { .. } => {}
        }

        match (
            self.claim_validity.valid_from_ms,
            self.claim_validity.valid_until_ms,
        ) {
            (Some(from), Some(until)) if until <= from => {
                reasons.push(R::InvalidClaimValidityInterval);
            }
            (Some(from), _) if now_ms < from => reasons.push(R::ClaimNotYetValid),
            (_, Some(until)) if now_ms >= until => reasons.push(R::ClaimExpired),
            _ => {}
        }

        let decision = if reasons.is_empty() {
            ProvenanceDecision::EligibleForReview
        } else {
            ProvenanceDecision::Abstain
        };
        ProvenanceAssessment { decision, reasons }
    }
}

fn has_text(value: Option<&str>) -> bool {
    value.is_some_and(|text| !text.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_000_000_000;

    fn complete_provenance() -> EvidenceProvenance {
        EvidenceProvenance {
            source: EvidenceSourceReference {
                source_id: "urn:example:source:report-7".to_owned(),
                publisher_id: Some("urn:example:publisher:lab-a".to_owned()),
                revision_id: Some("revision-17".to_owned()),
                // Shape-only fixture: tests do not assert this digest matches real bytes.
                content_digest: Some(ContentDigest::sha256("a".repeat(64))),
                locator: Some("https://example.org/report#page=7".to_owned()),
                retrieved_at_ms: Some(NOW - 1_000),
                published_at_ms: Some(NOW - 10_000),
                rights_reference: Some("https://example.org/licence".to_owned()),
            },
            integrity: IntegrityStatus::Verified {
                method: "fixture-verifier".to_owned(),
                checked_at_ms: NOW - 1_000,
            },
            source_status: SourceStatus::Active {
                checked_at_ms: NOW - 1_000,
            },
            claim_validity: ClaimValidity::default(),
        }
    }

    #[test]
    fn complete_fresh_provenance_is_eligible_only_for_review() {
        let assessment = complete_provenance().assess(&ProvenancePolicy::strict(), NOW);
        assert_eq!(assessment.decision, ProvenanceDecision::EligibleForReview);
        assert!(assessment.reasons.is_empty());
        assert!(assessment.is_eligible_for_review());
    }

    #[test]
    fn missing_anchors_and_malformed_digest_fail_closed() {
        let mut provenance = complete_provenance();
        provenance.source.revision_id = None;
        provenance.source.content_digest = Some(ContentDigest::sha256("not-a-digest"));
        provenance.source.locator = Some("   ".to_owned());
        provenance.source.publisher_id = None;

        let assessment = provenance.assess(&ProvenancePolicy::strict(), NOW);
        assert_eq!(assessment.decision, ProvenanceDecision::Abstain);
        assert!(assessment.reasons.contains(&ProvenanceAbstentionReason::MissingRevisionId));
        assert!(assessment.reasons.contains(&ProvenanceAbstentionReason::MalformedContentDigest));
        assert!(assessment.reasons.contains(&ProvenanceAbstentionReason::MissingLocator));
        assert!(assessment.reasons.contains(&ProvenanceAbstentionReason::MissingPublisherId));
    }

    #[test]
    fn unknown_integrity_or_source_status_abstains() {
        let mut provenance = complete_provenance();
        provenance.integrity = IntegrityStatus::Unknown;
        provenance.source_status = SourceStatus::Unknown;

        let assessment = provenance.assess(&ProvenancePolicy::strict(), NOW);
        assert_eq!(assessment.decision, ProvenanceDecision::Abstain);
        assert!(assessment.reasons.contains(&ProvenanceAbstentionReason::UnknownIntegrity));
        assert!(assessment.reasons.contains(&ProvenanceAbstentionReason::UnknownSourceStatus));
    }

    #[test]
    fn revocation_and_integrity_mismatch_cannot_be_bypassed_by_permissive_policy() {
        let mut provenance = complete_provenance();
        provenance.integrity = IntegrityStatus::Mismatch {
            checked_at_ms: NOW,
            reason: "fixture digest mismatch".to_owned(),
        };
        provenance.source_status = SourceStatus::Revoked {
            checked_at_ms: NOW,
            reason: "fixture revocation".to_owned(),
        };
        let permissive = ProvenancePolicy {
            require_publisher_id: false,
            require_revision_id: false,
            require_content_digest: false,
            require_locator: false,
            require_retrieved_at: false,
            require_rights_reference: false,
            require_verified_integrity: false,
            require_active_source: false,
            max_retrieval_age_ms: None,
            max_status_age_ms: None,
        };

        let assessment = provenance.assess(&permissive, NOW);
        assert_eq!(assessment.decision, ProvenanceDecision::Abstain);
        assert!(assessment.reasons.contains(&ProvenanceAbstentionReason::IntegrityMismatch));
        assert!(assessment.reasons.contains(&ProvenanceAbstentionReason::SourceRevoked));
    }

    #[test]
    fn stale_observation_and_stale_status_abstain_independently() {
        let mut provenance = complete_provenance();
        provenance.source.retrieved_at_ms = Some(NOW - 31 * 24 * 60 * 60 * 1_000);
        provenance.integrity = IntegrityStatus::Verified {
            method: "fixture-verifier".to_owned(),
            checked_at_ms: NOW - 25 * 60 * 60 * 1_000,
        };
        provenance.source_status = SourceStatus::Active {
            checked_at_ms: NOW - 25 * 60 * 60 * 1_000,
        };

        let assessment = provenance.assess(&ProvenancePolicy::strict(), NOW);
        assert_eq!(assessment.decision, ProvenanceDecision::Abstain);
        assert!(assessment.reasons.contains(&ProvenanceAbstentionReason::RetrievalTooOld));
        assert!(assessment.reasons.contains(&ProvenanceAbstentionReason::VerificationStale));
        assert!(assessment.reasons.contains(&ProvenanceAbstentionReason::SourceStatusStale));
    }

    #[test]
    fn future_timestamps_and_invalid_claim_intervals_abstain() {
        let mut provenance = complete_provenance();
        provenance.source.retrieved_at_ms = Some(NOW + 1);
        provenance.source.published_at_ms = Some(NOW + 1);
        provenance.integrity = IntegrityStatus::Verified {
            method: "fixture-verifier".to_owned(),
            checked_at_ms: NOW + 1,
        };
        provenance.source_status = SourceStatus::Active {
            checked_at_ms: NOW + 1,
        };
        provenance.claim_validity = ClaimValidity {
            valid_from_ms: Some(NOW + 100),
            valid_until_ms: Some(NOW + 50),
        };

        let assessment = provenance.assess(&ProvenancePolicy::strict(), NOW);
        assert_eq!(assessment.decision, ProvenanceDecision::Abstain);
        assert!(assessment.reasons.contains(&ProvenanceAbstentionReason::RetrievalTimeInFuture));
        assert!(assessment.reasons.contains(&ProvenanceAbstentionReason::PublicationTimeInFuture));
        assert!(assessment.reasons.contains(&ProvenanceAbstentionReason::VerificationTimeInFuture));
        assert!(assessment.reasons.contains(&ProvenanceAbstentionReason::SourceStatusTimeInFuture));
        assert!(assessment
            .reasons
            .contains(&ProvenanceAbstentionReason::InvalidClaimValidityInterval));
    }

    #[test]
    fn claim_validity_uses_half_open_interval() {
        let mut provenance = complete_provenance();
        provenance.claim_validity = ClaimValidity {
            valid_from_ms: Some(NOW),
            valid_until_ms: Some(NOW + 10),
        };
        assert_eq!(
            provenance.assess(&ProvenancePolicy::strict(), NOW).decision,
            ProvenanceDecision::EligibleForReview
        );
        assert!(provenance
            .assess(&ProvenancePolicy::strict(), NOW + 10)
            .reasons
            .contains(&ProvenanceAbstentionReason::ClaimExpired));
    }

    #[test]
    fn unavailable_source_status_is_not_silently_treated_as_active() {
        let mut provenance = complete_provenance();
        provenance.source_status = SourceStatus::Unavailable {
            checked_at_ms: NOW,
            reason: "fixture network error".to_owned(),
        };
        let assessment = provenance.assess(&ProvenancePolicy::strict(), NOW);
        assert_eq!(assessment.decision, ProvenanceDecision::Abstain);
        assert!(assessment.reasons.contains(&ProvenanceAbstentionReason::SourceStatusUnavailable));
    }

    #[test]
    fn withdrawn_sources_remain_historically_representable_but_do_not_advance() {
        let mut provenance = complete_provenance();
        provenance.source_status = SourceStatus::Withdrawn {
            checked_at_ms: NOW,
            reason: "publisher withdrew source".to_owned(),
        };
        let assessment = provenance.assess(&ProvenancePolicy::strict(), NOW);
        assert_eq!(assessment.decision, ProvenanceDecision::Abstain);
        assert!(assessment.reasons.contains(&ProvenanceAbstentionReason::SourceWithdrawn));
    }


    #[test]
    fn required_rights_and_unsupported_digest_algorithm_are_explicit() {
        let mut provenance = complete_provenance();
        provenance.source.rights_reference = None;
        provenance.source.content_digest = Some(ContentDigest {
            algorithm: "sha512".to_owned(),
            value: "a".repeat(128),
        });
        let mut policy = ProvenancePolicy::strict();
        policy.require_rights_reference = true;

        let assessment = provenance.assess(&policy, NOW);
        assert_eq!(assessment.decision, ProvenanceDecision::Abstain);
        assert!(assessment.reasons.contains(&ProvenanceAbstentionReason::MissingRightsReference));
        assert!(assessment
            .reasons
            .contains(&ProvenanceAbstentionReason::UnsupportedDigestAlgorithm));
    }

    #[test]
    fn integrity_attestation_revocation_is_not_source_status_revocation() {
        let mut provenance = complete_provenance();
        provenance.integrity = IntegrityStatus::Revoked {
            checked_at_ms: NOW,
            reason: "fixture verifier credential revoked".to_owned(),
        };
        provenance.source_status = SourceStatus::Active {
            checked_at_ms: NOW,
        };
        let permissive = ProvenancePolicy {
            require_publisher_id: false,
            require_revision_id: false,
            require_content_digest: false,
            require_locator: false,
            require_retrieved_at: false,
            require_rights_reference: false,
            require_verified_integrity: false,
            require_active_source: false,
            max_retrieval_age_ms: None,
            max_status_age_ms: None,
        };

        let assessment = provenance.assess(&permissive, NOW);
        assert_eq!(assessment.decision, ProvenanceDecision::Abstain);
        assert!(assessment.reasons.contains(&ProvenanceAbstentionReason::IntegrityRevoked));
        assert!(!assessment.reasons.contains(&ProvenanceAbstentionReason::SourceRevoked));
    }

    #[test]
    fn source_identity_is_required_even_when_other_gates_are_relaxed() {
        let mut provenance = complete_provenance();
        provenance.source.source_id = "  ".to_owned();
        let permissive = ProvenancePolicy {
            require_publisher_id: false,
            require_revision_id: false,
            require_content_digest: false,
            require_locator: false,
            require_retrieved_at: false,
            require_rights_reference: false,
            require_verified_integrity: false,
            require_active_source: false,
            max_retrieval_age_ms: None,
            max_status_age_ms: None,
        };

        let assessment = provenance.assess(&permissive, NOW);
        assert_eq!(assessment.decision, ProvenanceDecision::Abstain);
        assert!(assessment.reasons.contains(&ProvenanceAbstentionReason::MissingSourceId));
    }

    #[test]
    fn claim_cannot_be_used_before_its_valid_from_boundary() {
        let mut provenance = complete_provenance();
        provenance.claim_validity = ClaimValidity {
            valid_from_ms: Some(NOW + 1),
            valid_until_ms: Some(NOW + 100),
        };

        let assessment = provenance.assess(&ProvenancePolicy::strict(), NOW);
        assert_eq!(assessment.decision, ProvenanceDecision::Abstain);
        assert!(assessment.reasons.contains(&ProvenanceAbstentionReason::ClaimNotYetValid));
    }


    /// Pure-function microbenchmark only; it excludes storage, hashing, source
    /// status I/O, signature verification, and graph retrieval.
    #[test]
    #[ignore = "manual release-mode microbenchmark; not a CI correctness gate"]
    fn benchmark_provenance_gate_throughput() {
        use std::time::Instant;

        const REPEATS: usize = 21;
        let fixture = complete_provenance();
        let policy = ProvenancePolicy::strict();
        assert_eq!(
            fixture.assess(&policy, NOW).decision,
            ProvenanceDecision::EligibleForReview
        );

        for batch_size in [1_000usize, 10_000, 100_000] {
            let mut batch_ns = Vec::with_capacity(REPEATS);
            for _ in 0..REPEATS {
                let start = Instant::now();
                for _ in 0..batch_size {
                    let assessment = fixture.assess(&policy, NOW);
                    std::hint::black_box((assessment.decision, assessment.reasons.len()));
                }
                batch_ns.push(start.elapsed().as_nanos());
            }
            batch_ns.sort_unstable();
            let p50_ns = batch_ns[REPEATS / 2];
            let p95_index = (REPEATS * 95 + 99) / 100 - 1;
            let p95_ns = batch_ns[p95_index];
            println!(
                "n={batch_size} rep={REPEATS} p50={p50_ns} p95={p95_ns} med_ns/item={}",
                p50_ns / batch_size as u128
            );
        }
    }

}
