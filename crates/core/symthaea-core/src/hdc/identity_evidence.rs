// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-gated identity resolution for HDC similarity search.
//!
//! Similarity ranks candidates; it does not establish semantic or physical identity.
//! This module keeps those claims separate in the type-level API. Its registration
//! receipt is an in-process capability for experiments, not a cryptographic
//! attestation. Production callers must authenticate identity evidence before
//! passing a receipt across a trust boundary.

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use super::pattern_interaction::{PatternSpaceError, PatternVector};

/// Fail-closed search thresholds. A large nearest-neighbour score is not identity
/// proof; a receipt is separately required for a Verified identity result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolutionPolicy {
    pub min_similarity: f64,
    pub min_top_two_margin: f64,
}

impl ResolutionPolicy {
    pub fn new(
        min_similarity: f64,
        min_top_two_margin: f64,
    ) -> Result<Self, IdentityEvidenceError> {
        if !min_similarity.is_finite() || !(-1.0..=1.0).contains(&min_similarity) {
            return Err(IdentityEvidenceError::InvalidPolicy(
                "minimum similarity must be finite and in [-1, 1]",
            ));
        }
        if !min_top_two_margin.is_finite() || !(0.0..=2.0).contains(&min_top_two_margin) {
            return Err(IdentityEvidenceError::InvalidPolicy(
                "minimum top-two margin must be finite and in [0, 2]",
            ));
        }
        Ok(Self {
            min_similarity,
            min_top_two_margin,
        })
    }
}

impl Default for ResolutionPolicy {
    fn default() -> Self {
        Self {
            min_similarity: 0.50,
            min_top_two_margin: 0.04,
        }
    }
}

/// Issued by PatternRegistry::register. This is an in-process capability,
/// not a signature or remotely verifiable credential.
#[derive(Debug)]
struct RegistryInstanceToken {
    _marker: u8,
}

#[derive(Debug)]
struct EnrollmentToken {
    _marker: u8,
}

#[derive(Debug, Clone)]
pub struct RegistrationReceipt {
    registry_id: String,
    registry_instance: Arc<RegistryInstanceToken>,
    enrollment_token: Arc<EnrollmentToken>,
    identity_id: String,
    enrollment_revision: u64,
}

impl PartialEq for RegistrationReceipt {
    fn eq(&self, other: &Self) -> bool {
        self.registry_id == other.registry_id
            && Arc::ptr_eq(&self.registry_instance, &other.registry_instance)
            && Arc::ptr_eq(&self.enrollment_token, &other.enrollment_token)
            && self.identity_id == other.identity_id
            && self.enrollment_revision == other.enrollment_revision
    }
}

impl Eq for RegistrationReceipt {}

/// Observation-bound identity assertion supplied by an independently trusted source.
///
/// The digest binds the assertion to the exact observed vector, while the enrollment
/// tokens bind it to one registry enrollment. This type does not authenticate its
/// source: callers may create it only after authenticating the source/claim at their
/// own trust boundary. Production systems should construct it only after verifying a
/// signed or otherwise authenticated observation-to-identity assertion.
#[derive(Debug, Clone)]
pub struct ObservationIdentityEvidence {
    registry_id: String,
    registry_instance: Arc<RegistryInstanceToken>,
    enrollment_token: Arc<EnrollmentToken>,
    identity_id: String,
    observation_id: String,
    observation_digest: [u8; 32],
    source_id: String,
}

impl ObservationIdentityEvidence {
    /// Bind an upstream-authenticated identity claim to one exact observation.
    ///
    /// This constructor records the caller's authentication decision; it does not
    /// perform cryptographic authentication itself.
    pub fn from_authenticated_source(
        receipt: &RegistrationReceipt,
        observation_id: impl Into<String>,
        observed: &PatternVector,
        source_id: impl Into<String>,
    ) -> Result<Self, IdentityEvidenceError> {
        let observation_id = observation_id.into();
        if observation_id.trim().is_empty() {
            return Err(IdentityEvidenceError::EmptyObservationId);
        }
        let source_id = source_id.into();
        if source_id.trim().is_empty() {
            return Err(IdentityEvidenceError::EmptyEvidenceSource);
        }
        Ok(Self {
            registry_id: receipt.registry_id.clone(),
            registry_instance: Arc::clone(&receipt.registry_instance),
            enrollment_token: Arc::clone(&receipt.enrollment_token),
            identity_id: receipt.identity_id.clone(),
            observation_id,
            observation_digest: observation_digest(observed),
            source_id,
        })
    }

    pub fn observation_id(&self) -> &str {
        &self.observation_id
    }

    pub fn source_id(&self) -> &str {
        &self.source_id
    }
}

impl RegistrationReceipt {
    pub fn registry_id(&self) -> &str {
        &self.registry_id
    }

    pub fn identity_id(&self) -> &str {
        &self.identity_id
    }

    pub fn enrollment_revision(&self) -> u64 {
        self.enrollment_revision
    }
}

/// Geometric candidate. The type intentionally contains no identity-verification claim.
#[derive(Debug, Clone, PartialEq)]
pub struct IdentityCandidate {
    pub identity_id: String,
    pub similarity: f64,
}

/// Pure similarity-search outcome. CandidateOnly means that the vector supports a
/// candidate ranking; it must not be promoted to verified identity by a threshold.
#[derive(Debug, Clone, PartialEq)]
pub enum SimilarityResolution {
    NoCandidate {
        nearest: IdentityCandidate,
    },
    Ambiguous {
        candidates: Vec<IdentityCandidate>,
        margin: f64,
    },
    CandidateOnly {
        candidate: IdentityCandidate,
        registry_revision: u64,
    },
}

/// Identity outcome after checking a separately supplied registration capability.
#[derive(Debug, Clone, PartialEq)]
pub enum IdentityResolution {
    /// Enrollment and vector geometry are consistent, but no independent assertion
    /// binds this observation to the claimed semantic identity.
    EvidenceRequired {
        claimed_identity: String,
        nearest_identity: String,
        nearest_similarity: f64,
        claimed_similarity: f64,
        top_two_margin: f64,
        registry_revision: u64,
        enrollment_revision: u64,
    },
    /// Registration evidence exists and the observation is geometrically consistent
    /// with the claimed registry entry under the requested policy.
    Verified {
        identity_id: String,
        similarity: f64,
        top_two_margin: f64,
        registry_revision: u64,
        enrollment_revision: u64,
    },
    /// The capability was valid, but vector evidence contradicts the claim or does
    /// not meet the configured similarity/margin requirements.
    EvidenceConflict {
        claimed_identity: String,
        nearest_identity: String,
        nearest_similarity: f64,
        top_two_margin: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdentityEvidenceError {
    EmptyRegistryId,
    EmptyIdentityId,
    DuplicateIdentity(String),
    EmptyRegistry,
    InvalidReceipt,
    InvalidObservationEvidence,
    EmptyObservationId,
    EmptyEvidenceSource,
    RevisionOverflow,
    InvalidPolicy(&'static str),
    PatternSpace(PatternSpaceError),
}

impl fmt::Display for IdentityEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRegistryId => write!(f, "registry id must not be empty"),
            Self::EmptyIdentityId => write!(f, "identity id must not be empty"),
            Self::DuplicateIdentity(id) => write!(f, "identity already registered: {id}"),
            Self::EmptyRegistry => write!(f, "cannot resolve against an empty registry"),
            Self::InvalidReceipt => write!(f, "registration receipt is invalid for this registry"),
            Self::InvalidObservationEvidence => {
                write!(
                    f,
                    "observation identity evidence is invalid for this query or enrollment"
                )
            }
            Self::EmptyObservationId => write!(f, "observation id must not be empty"),
            Self::EmptyEvidenceSource => write!(f, "evidence source id must not be empty"),
            Self::RevisionOverflow => write!(f, "registry revision overflow"),
            Self::InvalidPolicy(message) => write!(f, "invalid resolution policy: {message}"),
            Self::PatternSpace(error) => write!(f, "pattern operation failed: {error}"),
        }
    }
}

impl Error for IdentityEvidenceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::PatternSpace(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PatternSpaceError> for IdentityEvidenceError {
    fn from(value: PatternSpaceError) -> Self {
        Self::PatternSpace(value)
    }
}

#[derive(Debug, Clone)]
struct RegisteredPattern {
    identity_id: String,
    pattern: PatternVector,
    enrollment_revision: u64,
    enrollment_token: Arc<EnrollmentToken>,
}

/// Append-only registry for the identity-evidence boundary experiment.
///
/// The registry assigns an explicit semantic ID to a prototype. Similarity-only
/// queries return candidates only. A RegistrationReceipt can authorize a verified
/// identity result only when it belongs to this registry and the observed vector is
/// consistent with the registered prototype. Trust in how a receipt was obtained
/// remains the caller's responsibility.
#[derive(Debug, Clone)]
pub struct PatternRegistry {
    registry_id: String,
    registry_instance: Arc<RegistryInstanceToken>,
    revision: u64,
    entries: Vec<RegisteredPattern>,
}

impl PatternRegistry {
    pub fn new(registry_id: impl Into<String>) -> Result<Self, IdentityEvidenceError> {
        let registry_id = registry_id.into();
        if registry_id.trim().is_empty() {
            return Err(IdentityEvidenceError::EmptyRegistryId);
        }
        Ok(Self {
            registry_id,
            registry_instance: Arc::new(RegistryInstanceToken { _marker: 0 }),
            revision: 0,
            entries: Vec::new(),
        })
    }

    pub fn registry_id(&self) -> &str {
        &self.registry_id
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn register(
        &mut self,
        identity_id: impl Into<String>,
        pattern: PatternVector,
    ) -> Result<RegistrationReceipt, IdentityEvidenceError> {
        let identity_id = identity_id.into();
        if identity_id.trim().is_empty() {
            return Err(IdentityEvidenceError::EmptyIdentityId);
        }
        if self
            .entries
            .iter()
            .any(|entry| entry.identity_id == identity_id)
        {
            return Err(IdentityEvidenceError::DuplicateIdentity(identity_id));
        }
        if let Some(first) = self.entries.first() {
            if first.pattern.dimension() != pattern.dimension() {
                return Err(IdentityEvidenceError::PatternSpace(
                    PatternSpaceError::DimensionMismatch {
                        expected: first.pattern.dimension(),
                        actual: pattern.dimension(),
                    },
                ));
            }
        }

        self.revision = self
            .revision
            .checked_add(1)
            .ok_or(IdentityEvidenceError::RevisionOverflow)?;
        let enrollment_revision = self.revision;
        let enrollment_token = Arc::new(EnrollmentToken { _marker: 0 });
        self.entries.push(RegisteredPattern {
            identity_id: identity_id.clone(),
            pattern,
            enrollment_revision,
            enrollment_token: Arc::clone(&enrollment_token),
        });
        Ok(RegistrationReceipt {
            registry_id: self.registry_id.clone(),
            registry_instance: Arc::clone(&self.registry_instance),
            enrollment_token,
            identity_id,
            enrollment_revision,
        })
    }

    /// Rank candidates from vector geometry only. The result never asserts identity.
    pub fn resolve_similarity_only(
        &self,
        query: &PatternVector,
        policy: ResolutionPolicy,
    ) -> Result<SimilarityResolution, IdentityEvidenceError> {
        let (nearest, runner_up, margin) = self.rank(query)?;
        if nearest.similarity < policy.min_similarity {
            return Ok(SimilarityResolution::NoCandidate { nearest });
        }
        if let Some(second) = runner_up {
            if margin < policy.min_top_two_margin {
                return Ok(SimilarityResolution::Ambiguous {
                    candidates: vec![nearest, second],
                    margin,
                });
            }
        }
        Ok(SimilarityResolution::CandidateOnly {
            candidate: nearest,
            registry_revision: self.revision,
        })
    }

    /// Check that a registration receipt's enrolled prototype is geometrically
    /// consistent with the query. Registration alone is not proof that this query
    /// belongs to that identity, so a consistent result is EvidenceRequired, never
    /// Verified. A receipt never suppresses score, nearest-neighbour, or margin conflicts.
    pub fn resolve_with_receipt(
        &self,
        query: &PatternVector,
        receipt: &RegistrationReceipt,
        policy: ResolutionPolicy,
    ) -> Result<IdentityResolution, IdentityEvidenceError> {
        if receipt.registry_id != self.registry_id
            || !Arc::ptr_eq(&receipt.registry_instance, &self.registry_instance)
        {
            return Err(IdentityEvidenceError::InvalidReceipt);
        }
        let entry = self
            .entries
            .iter()
            .find(|entry| {
                entry.identity_id == receipt.identity_id
                    && entry.enrollment_revision == receipt.enrollment_revision
                    && Arc::ptr_eq(&entry.enrollment_token, &receipt.enrollment_token)
            })
            .ok_or(IdentityEvidenceError::InvalidReceipt)?;

        let (nearest, runner_up, margin) = self.rank(query)?;
        let claimed_similarity = query.similarity(&entry.pattern)?;
        let claimed_is_nearest = nearest.identity_id == receipt.identity_id;
        let has_sufficient_similarity = claimed_similarity >= policy.min_similarity;
        let has_sufficient_margin = runner_up.is_none() || margin >= policy.min_top_two_margin;

        if !claimed_is_nearest || !has_sufficient_similarity || !has_sufficient_margin {
            return Ok(IdentityResolution::EvidenceConflict {
                claimed_identity: receipt.identity_id.clone(),
                nearest_identity: nearest.identity_id,
                nearest_similarity: nearest.similarity,
                top_two_margin: margin,
            });
        }

        Ok(IdentityResolution::EvidenceRequired {
            claimed_identity: receipt.identity_id.clone(),
            nearest_identity: nearest.identity_id,
            nearest_similarity: nearest.similarity,
            claimed_similarity,
            top_two_margin: margin,
            registry_revision: self.revision,
            enrollment_revision: receipt.enrollment_revision,
        })
    }

    /// Verify a query only when an independently authenticated source has bound
    /// this exact observed vector to the enrolled identity. The evidence is
    /// observation-specific: reusing it for any other vector fails closed.
    pub fn resolve_with_observation_evidence(
        &self,
        query: &PatternVector,
        observation_id: &str,
        receipt: &RegistrationReceipt,
        evidence: &ObservationIdentityEvidence,
        policy: ResolutionPolicy,
    ) -> Result<IdentityResolution, IdentityEvidenceError> {
        if observation_id.trim().is_empty()
            || evidence.observation_id != observation_id
            || evidence.registry_id != self.registry_id
            || !Arc::ptr_eq(&evidence.registry_instance, &self.registry_instance)
            || evidence.identity_id != receipt.identity_id
            || evidence.registry_id != receipt.registry_id
            || !Arc::ptr_eq(&evidence.registry_instance, &receipt.registry_instance)
            || !Arc::ptr_eq(&evidence.enrollment_token, &receipt.enrollment_token)
            || evidence.observation_digest != observation_digest(query)
            || evidence.source_id.trim().is_empty()
        {
            return Err(IdentityEvidenceError::InvalidObservationEvidence);
        }

        match self.resolve_with_receipt(query, receipt, policy)? {
            IdentityResolution::EvidenceRequired {
                claimed_identity,
                claimed_similarity,
                top_two_margin,
                registry_revision,
                enrollment_revision,
                ..
            } => Ok(IdentityResolution::Verified {
                identity_id: claimed_identity,
                similarity: claimed_similarity,
                top_two_margin,
                registry_revision,
                enrollment_revision,
            }),
            other => Ok(other),
        }
    }

    fn rank(
        &self,
        query: &PatternVector,
    ) -> Result<(IdentityCandidate, Option<IdentityCandidate>, f64), IdentityEvidenceError> {
        let mut candidates = self
            .entries
            .iter()
            .map(|entry| {
                Ok(IdentityCandidate {
                    identity_id: entry.identity_id.clone(),
                    similarity: query.similarity(&entry.pattern)?,
                })
            })
            .collect::<Result<Vec<_>, PatternSpaceError>>()?;
        if candidates.is_empty() {
            return Err(IdentityEvidenceError::EmptyRegistry);
        }
        candidates.sort_by(|left, right| {
            right
                .similarity
                .total_cmp(&left.similarity)
                .then_with(|| left.identity_id.cmp(&right.identity_id))
        });
        let nearest = candidates[0].clone();
        let runner_up = candidates.get(1).cloned();
        let margin = runner_up
            .as_ref()
            .map(|candidate| nearest.similarity - candidate.similarity)
            .unwrap_or(2.0);
        Ok((nearest, runner_up, margin))
    }
}

fn observation_digest(query: &PatternVector) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-observation-identity-evidence-v1");
    hasher.update(&(query.dimension() as u64).to_le_bytes());
    for value in query.values() {
        hasher.update(&[*value as u8]);
    }
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> ResolutionPolicy {
        ResolutionPolicy::new(0.40, 0.02).expect("valid test policy")
    }

    #[test]
    fn high_similarity_without_registration_evidence_never_verifies_identity() {
        let pattern = PatternVector::seeded_random(2_048, 11).expect("pattern");
        let mut registry = PatternRegistry::new("registry-a").expect("registry");
        registry
            .register("object-17", pattern.clone())
            .expect("enrollment");

        let result = registry
            .resolve_similarity_only(&pattern, policy())
            .expect("similarity resolution");
        assert!(matches!(
            result,
            SimilarityResolution::CandidateOnly { ref candidate, .. }
                if candidate.identity_id == "object-17" && candidate.similarity == 1.0
        ));
    }

    #[test]
    fn registration_receipt_requires_separate_observation_bound_identity_evidence() {
        let pattern = PatternVector::seeded_random(2_048, 29).expect("pattern");
        let mut registry = PatternRegistry::new("registry-a").expect("registry");
        let receipt = registry
            .register("object-29", pattern.clone())
            .expect("enrollment");

        let vector_only = registry
            .resolve_similarity_only(&pattern, policy())
            .expect("candidate");
        let registration_only = registry
            .resolve_with_receipt(&pattern, &receipt, policy())
            .expect("registration consistency");
        let observation_evidence = ObservationIdentityEvidence::from_authenticated_source(
            &receipt,
            "capture-object-29-0",
            &pattern,
            "test-identity-authority",
        )
        .expect("source assertion");
        let with_observation_evidence = registry
            .resolve_with_observation_evidence(
                &pattern,
                "capture-object-29-0",
                &receipt,
                &observation_evidence,
                policy(),
            )
            .expect("observation-bound evidence");

        assert!(matches!(
            vector_only,
            SimilarityResolution::CandidateOnly { .. }
        ));
        assert!(matches!(
            registration_only,
            IdentityResolution::EvidenceRequired {
                ref claimed_identity,
                claimed_similarity,
                ..
            } if claimed_identity == "object-29" && claimed_similarity == 1.0
        ));
        assert!(matches!(
            with_observation_evidence,
            IdentityResolution::Verified { ref identity_id, similarity, .. }
                if identity_id == "object-29" && similarity == 1.0
        ));
    }

    #[test]
    fn observation_identity_evidence_cannot_be_replayed_for_another_vector() {
        let pattern = PatternVector::seeded_random(2_048, 31).expect("pattern");
        let other_query = pattern.with_noise(0.10, 37).expect("different observation");
        let mut registry = PatternRegistry::new("registry-a").expect("registry");
        let receipt = registry
            .register("object-31", pattern.clone())
            .expect("enrollment");
        let evidence = ObservationIdentityEvidence::from_authenticated_source(
            &receipt,
            "capture-original",
            &pattern,
            "test-identity-authority",
        )
        .expect("source assertion");

        assert_eq!(
            registry.resolve_with_observation_evidence(
                &other_query,
                "capture-original",
                &receipt,
                &evidence,
                policy(),
            ),
            Err(IdentityEvidenceError::InvalidObservationEvidence)
        );
    }

    #[test]
    fn observation_identity_evidence_cannot_replay_across_identical_vectors_in_distinct_captures() {
        let pattern = PatternVector::seeded_random(1_024, 47).expect("pattern");
        let mut registry = PatternRegistry::new("registry-a").expect("registry");
        let receipt = registry
            .register("object-47", pattern.clone())
            .expect("enrollment");
        let evidence = ObservationIdentityEvidence::from_authenticated_source(
            &receipt,
            "capture-instance-A",
            &pattern,
            "test-identity-authority",
        )
        .expect("source assertion");

        // The pattern bytes are identical; only the unique capture identity differs.
        assert_eq!(
            registry.resolve_with_observation_evidence(
                &pattern,
                "capture-instance-B",
                &receipt,
                &evidence,
                policy(),
            ),
            Err(IdentityEvidenceError::InvalidObservationEvidence)
        );
    }

    #[test]
    fn observation_identity_evidence_rejects_empty_source_ids() {
        let pattern = PatternVector::seeded_random(1_024, 43).expect("pattern");
        let mut registry = PatternRegistry::new("registry-a").expect("registry");
        let receipt = registry
            .register("object-43", pattern.clone())
            .expect("enrollment");
        assert_eq!(
            ObservationIdentityEvidence::from_authenticated_source(
                &receipt,
                "capture-43",
                &pattern,
                "  ",
            )
            .err(),
            Some(IdentityEvidenceError::EmptyEvidenceSource)
        );
    }

    #[test]
    fn a_valid_receipt_does_not_override_contradictory_vector_evidence() {
        let target = PatternVector::seeded_random(2_048, 37).expect("target");
        let distractor = PatternVector::seeded_random(2_048, 41).expect("distractor");
        let mut registry = PatternRegistry::new("registry-a").expect("registry");
        let target_receipt = registry
            .register("target", target)
            .expect("target enrollment");
        registry
            .register("distractor", distractor.clone())
            .expect("distractor enrollment");

        let result = registry
            .resolve_with_receipt(&distractor, &target_receipt, policy())
            .expect("conflict is a result, not an error");
        assert!(matches!(
            result,
            IdentityResolution::EvidenceConflict {
                ref claimed_identity,
                ref nearest_identity,
                ..
            } if claimed_identity == "target" && nearest_identity == "distractor"
        ));
    }

    #[test]
    fn appending_a_new_identity_does_not_revoke_existing_enrollment_receipt() {
        let first_pattern = PatternVector::seeded_random(2_048, 47).expect("first pattern");
        let second_pattern = PatternVector::seeded_random(2_048, 49).expect("second pattern");
        let mut registry = PatternRegistry::new("registry-a").expect("registry");
        let first_receipt = registry
            .register("first", first_pattern.clone())
            .expect("first enrollment");

        registry
            .register("second", second_pattern)
            .expect("second enrollment");
        let result = registry
            .resolve_with_receipt(&first_pattern, &first_receipt, policy())
            .expect("previous enrollment remains present");

        assert!(matches!(
            result,
            IdentityResolution::EvidenceRequired {
                ref claimed_identity,
                registry_revision: 2,
                enrollment_revision: 1,
                ..
            } if claimed_identity == "first"
        ));
    }

    #[test]
    fn receipt_from_another_registry_is_rejected() {
        let pattern = PatternVector::seeded_random(1_024, 53).expect("pattern");
        let mut first = PatternRegistry::new("registry-one").expect("first");
        let receipt = first.register("subject", pattern.clone()).expect("enroll");
        let second = PatternRegistry::new("registry-two").expect("second");

        assert_eq!(
            second.resolve_with_receipt(&pattern, &receipt, policy()),
            Err(IdentityEvidenceError::InvalidReceipt)
        );
    }

    #[test]
    fn receipt_cannot_cross_registry_instances_with_the_same_public_id() {
        let first_pattern = PatternVector::seeded_random(1_024, 59).expect("first pattern");
        let second_pattern = PatternVector::seeded_random(1_024, 61).expect("second pattern");
        let mut first = PatternRegistry::new("shared-registry-label").expect("first registry");
        let mut second = PatternRegistry::new("shared-registry-label").expect("second registry");
        let receipt = first
            .register("same-subject", first_pattern)
            .expect("first enrollment");
        second
            .register("same-subject", second_pattern.clone())
            .expect("second enrollment");

        assert_eq!(
            second.resolve_with_receipt(&second_pattern, &receipt, policy()),
            Err(IdentityEvidenceError::InvalidReceipt)
        );
    }

    #[test]
    fn receipt_cannot_cross_divergent_clones_that_reuse_identity_and_revision() {
        let anchor_pattern = PatternVector::seeded_random(1_024, 67).expect("anchor pattern");
        let first_fork_pattern =
            PatternVector::seeded_random(1_024, 83).expect("first fork pattern");
        let second_fork_pattern =
            PatternVector::seeded_random(1_024, 89).expect("second fork pattern");
        let mut original = PatternRegistry::new("shared-registry-label").expect("registry");
        original
            .register("anchor", anchor_pattern)
            .expect("anchor enrollment");
        let mut fork = original.clone();

        let original_receipt = original
            .register("forked-subject", first_fork_pattern)
            .expect("original fork enrollment");
        fork.register("forked-subject", second_fork_pattern.clone())
            .expect("independent fork enrollment");

        assert_eq!(
            fork.resolve_with_receipt(&second_fork_pattern, &original_receipt, policy()),
            Err(IdentityEvidenceError::InvalidReceipt)
        );
    }

    #[test]
    fn ambiguous_or_low_similarity_queries_fail_closed() {
        let a = PatternVector::seeded_random(4_096, 61).expect("pattern a");
        let b = a.clone();
        let mut registry = PatternRegistry::new("registry-a").expect("registry");
        registry.register("a", a.clone()).expect("enrollment a");
        registry.register("b", b).expect("enrollment b");
        let permissive = ResolutionPolicy::new(-1.0, 0.01).expect("policy");
        assert!(matches!(
            registry.resolve_similarity_only(&a, permissive).expect("resolution"),
            SimilarityResolution::Ambiguous { margin, .. } if margin == 0.0
        ));

        let strict = ResolutionPolicy::new(1.0, 0.01).expect("strict policy");
        let noisy = a.with_noise(0.10, 99).expect("noise");
        assert!(matches!(
            registry
                .resolve_similarity_only(&noisy, strict)
                .expect("low similarity"),
            SimilarityResolution::NoCandidate { .. } | SimilarityResolution::Ambiguous { .. }
        ));
    }

    #[test]
    fn invalid_registry_ids_and_policies_are_rejected() {
        assert!(matches!(
            PatternRegistry::new("  "),
            Err(IdentityEvidenceError::EmptyRegistryId)
        ));
        assert!(ResolutionPolicy::new(f64::NAN, 0.1).is_err());
        assert!(ResolutionPolicy::new(0.2, f64::INFINITY).is_err());

        let mut registry = PatternRegistry::new("registry-a").expect("registry");
        registry
            .register(
                "duplicate",
                PatternVector::seeded_random(1_024, 71).expect("first prototype"),
            )
            .expect("first enrollment");
        assert!(matches!(
            registry.register(
                "duplicate",
                PatternVector::seeded_random(1_024, 73).expect("second prototype"),
            ),
            Err(IdentityEvidenceError::DuplicateIdentity(id)) if id == "duplicate"
        ));
        assert!(matches!(
            registry.register(
                "wrong-dimension",
                PatternVector::seeded_random(2_048, 79).expect("different dimension"),
            ),
            Err(IdentityEvidenceError::PatternSpace(
                PatternSpaceError::DimensionMismatch {
                    expected: 1_024,
                    actual: 2_048
                }
            ))
        ));
    }
}
