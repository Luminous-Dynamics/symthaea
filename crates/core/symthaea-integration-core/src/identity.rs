// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bearing identity claims and conservative entity-resolution proposals.
//!
//! Integrations frequently describe the same real resource under incompatible
//! identifiers. This module records those identifiers as claims and produces
//! deterministic *proposals* about entity equivalence. It never rewrites an
//! [`EntityRef`] or silently merges world-model entities.

use crate::{EntityRef, ObservationId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// How uniquely an identifier is expected to name an entity.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum IdentifierUniqueness {
    /// Hostnames, service names, labels, or other identifiers that may repeat.
    Ambiguous,
    /// Unique only inside an explicit authority/scope such as a tenant, VPC, or cluster.
    Scoped,
    /// Intended to be globally unique in its identifier domain.
    Global,
}

/// Expected lifetime/stability of an identifier.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum IdentifierStability {
    Ephemeral,
    Session,
    Persistent,
}

/// Evidence strength assigned by the adapter that understands the identifier's
/// semantics. This is categorical on purpose; it is not a fabricated posterior
/// probability.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum IdentityStrength {
    Weak,
    Moderate,
    Strong,
    Authoritative,
}

/// External identifier carried by an integration source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalIdentifier {
    /// Semantic identifier scheme, e.g. `otel.service.instance.id`,
    /// `aws.ec2.instance_id`, `k8s.pod.uid`, `dmi.product_uuid`, `mac`.
    pub scheme: String,
    pub value: String,
    /// Administrative namespace required for Scoped identifiers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    pub uniqueness: IdentifierUniqueness,
    pub stability: IdentifierStability,
    /// False only when the identifier's specification explicitly defines
    /// case-insensitive comparison. Adapters must not guess normalization.
    pub case_sensitive: bool,
}

impl ExternalIdentifier {
    pub fn canonical_key(&self) -> Result<String, IdentityValidationError> {
        self.validate()?;
        let value = if self.case_sensitive {
            self.value.clone()
        } else {
            self.value.to_ascii_lowercase()
        };
        Ok(format!(
            "{}|{}|{}",
            self.scheme,
            self.scope.as_deref().unwrap_or(""),
            value
        ))
    }

    pub fn validate(&self) -> Result<(), IdentityValidationError> {
        require_non_empty("identifier.scheme", &self.scheme)?;
        require_non_empty("identifier.value", &self.value)?;
        if self.uniqueness == IdentifierUniqueness::Scoped
            && self.scope.as_deref().is_none_or(|scope| scope.trim().is_empty())
        {
            return Err(IdentityValidationError::ScopedIdentifierMissingScope);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityClaimSource {
    pub integration_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collector_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
}

/// One source's assertion that `subject` possesses `identifier`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IdentityClaim {
    pub claim_id: String,
    pub subject: EntityRef,
    pub identifier: ExternalIdentifier,
    pub strength: IdentityStrength,
    /// Confidence in source extraction/mapping only, [0,1]. It is not the
    /// probability that two entities are identical.
    pub source_confidence: f32,
    pub source: IdentityClaimSource,
    pub observed_at_unix_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_from_unix_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_until_unix_ms: Option<u64>,
    /// Runtime observations supporting this claim, when available.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_observation_ids: Vec<ObservationId>,
}

impl IdentityClaim {
    pub fn validate(&self) -> Result<(), IdentityValidationError> {
        require_non_empty("claim_id", &self.claim_id)?;
        validate_entity(&self.subject)?;
        self.identifier.validate()?;
        require_non_empty("source.integration_id", &self.source.integration_id)?;
        validate_probability("source_confidence", self.source_confidence)?;
        validate_interval(self.valid_from_unix_ms, self.valid_until_unix_ms)?;
        validate_evidence_ids(&self.evidence_observation_ids)?;
        Ok(())
    }

    pub fn is_active_at(&self, at_unix_ms: u64) -> bool {
        self.valid_from_unix_ms.is_none_or(|from| at_unix_ms >= from)
            && self.valid_until_unix_ms.is_none_or(|until| at_unix_ms <= until)
    }
}

/// Explicit evidence that two source-local references are different entities.
/// Examples include a CMDB assertion, distinct immutable hardware IDs observed
/// simultaneously, or an operator-reviewed disambiguation. This prevents a
/// shared weak alias such as a hostname from forcing an equivalence proposal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeparationClaim {
    pub claim_id: String,
    pub left: EntityRef,
    pub right: EntityRef,
    pub strength: IdentityStrength,
    pub source_confidence: f32,
    pub source: IdentityClaimSource,
    pub observed_at_unix_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_from_unix_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_until_unix_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence_observation_ids: Vec<ObservationId>,
}

impl SeparationClaim {
    pub fn validate(&self) -> Result<(), IdentityValidationError> {
        require_non_empty("claim_id", &self.claim_id)?;
        validate_entity(&self.left)?;
        validate_entity(&self.right)?;
        if self.left == self.right {
            return Err(IdentityValidationError::SelfSeparation);
        }
        require_non_empty("source.integration_id", &self.source.integration_id)?;
        validate_probability("source_confidence", self.source_confidence)?;
        validate_interval(self.valid_from_unix_ms, self.valid_until_unix_ms)?;
        validate_evidence_ids(&self.evidence_observation_ids)?;
        Ok(())
    }

    pub fn is_active_at(&self, at_unix_ms: u64) -> bool {
        self.valid_from_unix_ms.is_none_or(|from| at_unix_ms >= from)
            && self.valid_until_unix_ms.is_none_or(|until| at_unix_ms <= until)
    }

    fn applies_to(&self, left: &EntityRef, right: &EntityRef) -> bool {
        (&self.left == left && &self.right == right)
            || (&self.left == right && &self.right == left)
    }
}

/// Deterministically ordered pair used by the candidate index.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EntityPair {
    pub left: EntityRef,
    pub right: EntityRef,
}

impl EntityPair {
    pub fn new(left: EntityRef, right: EntityRef) -> Option<Self> {
        if left == right {
            return None;
        }
        if left <= right {
            Some(Self { left, right })
        } else {
            Some(Self {
                left: right,
                right: left,
            })
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IdentifierMatchEvidence {
    pub canonical_identifier: String,
    pub left_claim_ids: Vec<String>,
    pub right_claim_ids: Vec<String>,
    /// Conservative metadata across both sides of the match.
    pub effective_strength: IdentityStrength,
    pub uniqueness: IdentifierUniqueness,
    pub stability: IdentifierStability,
    /// Minimum extraction confidence among the strongest claim on each side.
    /// This qualifies source mapping reliability, not identity probability.
    pub min_source_confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolutionStatus {
    SameReference,
    /// At least one identifier matches, but only weak/ambiguous evidence exists.
    CandidateSame,
    /// A strong, non-ambiguous identifier match exists without strong separation evidence.
    StrongCandidateSame,
    /// Explicit strong separation evidence exists and no strong identity match overrides it.
    ExplicitlyDistinct,
    /// Strong identity and strong separation evidence disagree.
    ConflictingEvidence,
    Indeterminate,
}

/// Evidence-backed proposal. The caller decides whether and how a canonical
/// world model consumes it; this type performs no merge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityResolutionProposal {
    pub left: EntityRef,
    pub right: EntityRef,
    pub status: ResolutionStatus,
    pub identifier_matches: Vec<IdentifierMatchEvidence>,
    pub separation_claim_ids: Vec<String>,
    pub unresolved_identity_claim_ids: Vec<String>,
}

/// Index source-local identity claims and discover possible equivalence pairs.
#[derive(Debug, Clone, Default)]
pub struct IdentityClaimIndex {
    claims: Vec<IdentityClaim>,
    by_identifier: BTreeMap<String, Vec<usize>>,
}

impl IdentityClaimIndex {
    pub fn build(claims: Vec<IdentityClaim>) -> Result<Self, IdentityValidationError> {
        let mut by_identifier: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        let mut ids = BTreeSet::new();
        for (index, claim) in claims.iter().enumerate() {
            claim.validate()?;
            if !ids.insert(claim.claim_id.clone()) {
                return Err(IdentityValidationError::DuplicateClaimId(
                    claim.claim_id.clone(),
                ));
            }
            by_identifier
                .entry(claim.identifier.canonical_key()?)
                .or_default()
                .push(index);
        }
        Ok(Self {
            claims,
            by_identifier,
        })
    }

    pub fn claims(&self) -> &[IdentityClaim] {
        &self.claims
    }

    /// Candidate pairs that share at least one active external identifier.
    /// Weak identifiers intentionally still produce candidates; classification
    /// happens separately and never upgrades them to strong equivalence.
    pub fn candidate_pairs_at(&self, at_unix_ms: u64) -> BTreeSet<EntityPair> {
        let mut pairs = BTreeSet::new();
        for indices in self.by_identifier.values() {
            let active: Vec<&IdentityClaim> = indices
                .iter()
                .map(|index| &self.claims[*index])
                .filter(|claim| claim.is_active_at(at_unix_ms))
                .collect();
            for left in 0..active.len() {
                for right in (left + 1)..active.len() {
                    if let Some(pair) = EntityPair::new(
                        active[left].subject.clone(),
                        active[right].subject.clone(),
                    ) {
                        pairs.insert(pair);
                    }
                }
            }
        }
        pairs
    }
}

pub fn assess_entity_pair(
    left: &EntityRef,
    right: &EntityRef,
    identity_claims: &[IdentityClaim],
    separation_claims: &[SeparationClaim],
    at_unix_ms: u64,
) -> Result<EntityResolutionProposal, IdentityValidationError> {
    validate_entity(left)?;
    validate_entity(right)?;

    for claim in identity_claims {
        claim.validate()?;
    }
    for claim in separation_claims {
        claim.validate()?;
    }

    if left == right {
        return Ok(EntityResolutionProposal {
            left: left.clone(),
            right: right.clone(),
            status: ResolutionStatus::SameReference,
            identifier_matches: vec![],
            separation_claim_ids: vec![],
            unresolved_identity_claim_ids: vec![],
        });
    }

    let left_claims: Vec<&IdentityClaim> = identity_claims
        .iter()
        .filter(|claim| &claim.subject == left && claim.is_active_at(at_unix_ms))
        .collect();
    let right_claims: Vec<&IdentityClaim> = identity_claims
        .iter()
        .filter(|claim| &claim.subject == right && claim.is_active_at(at_unix_ms))
        .collect();

    let mut left_by_identifier: BTreeMap<String, Vec<&IdentityClaim>> = BTreeMap::new();
    let mut right_by_identifier: BTreeMap<String, Vec<&IdentityClaim>> = BTreeMap::new();
    for claim in &left_claims {
        left_by_identifier
            .entry(claim.identifier.canonical_key()?)
            .or_default()
            .push(*claim);
    }
    for claim in &right_claims {
        right_by_identifier
            .entry(claim.identifier.canonical_key()?)
            .or_default()
            .push(*claim);
    }

    let mut identifier_matches = Vec::new();
    let mut matched_claim_ids = BTreeSet::new();
    for (identifier, left_group) in &left_by_identifier {
        let Some(right_group) = right_by_identifier.get(identifier) else {
            continue;
        };

        let left_strength = left_group
            .iter()
            .map(|claim| claim.strength)
            .max()
            .unwrap_or(IdentityStrength::Weak);
        let right_strength = right_group
            .iter()
            .map(|claim| claim.strength)
            .max()
            .unwrap_or(IdentityStrength::Weak);
        let left_source_confidence = left_group
            .iter()
            .filter(|claim| claim.strength == left_strength)
            .map(|claim| claim.source_confidence)
            .max_by(f32::total_cmp)
            .unwrap_or(0.0);
        let right_source_confidence = right_group
            .iter()
            .filter(|claim| claim.strength == right_strength)
            .map(|claim| claim.source_confidence)
            .max_by(f32::total_cmp)
            .unwrap_or(0.0);
        let uniqueness = left_group
            .iter()
            .chain(right_group.iter())
            .map(|claim| claim.identifier.uniqueness)
            .min()
            .unwrap_or(IdentifierUniqueness::Ambiguous);
        let stability = left_group
            .iter()
            .chain(right_group.iter())
            .map(|claim| claim.identifier.stability)
            .min()
            .unwrap_or(IdentifierStability::Ephemeral);

        let mut left_claim_ids: Vec<String> = left_group
            .iter()
            .map(|claim| claim.claim_id.clone())
            .collect();
        let mut right_claim_ids: Vec<String> = right_group
            .iter()
            .map(|claim| claim.claim_id.clone())
            .collect();
        left_claim_ids.sort();
        right_claim_ids.sort();
        matched_claim_ids.extend(left_claim_ids.iter().cloned());
        matched_claim_ids.extend(right_claim_ids.iter().cloned());

        identifier_matches.push(IdentifierMatchEvidence {
            canonical_identifier: identifier.clone(),
            left_claim_ids,
            right_claim_ids,
            effective_strength: left_strength.min(right_strength),
            uniqueness,
            stability,
            min_source_confidence: left_source_confidence.min(right_source_confidence),
        });
    }

    let mut separation_claim_ids: Vec<String> = separation_claims
        .iter()
        .filter(|claim| claim.is_active_at(at_unix_ms) && claim.applies_to(left, right))
        .map(|claim| claim.claim_id.clone())
        .collect();
    separation_claim_ids.sort();

    let strong_identity = identifier_matches.iter().any(|evidence| {
        evidence.effective_strength >= IdentityStrength::Strong
            && evidence.uniqueness >= IdentifierUniqueness::Scoped
            && evidence.stability >= IdentifierStability::Session
            && evidence.min_source_confidence >= 0.9
    });
    let any_identity = !identifier_matches.is_empty();
    let strong_separation = separation_claims.iter().any(|claim| {
        claim.is_active_at(at_unix_ms)
            && claim.applies_to(left, right)
            && claim.strength >= IdentityStrength::Strong
            && claim.source_confidence >= 0.9
    });

    let status = if strong_identity && strong_separation {
        ResolutionStatus::ConflictingEvidence
    } else if strong_identity {
        ResolutionStatus::StrongCandidateSame
    } else if strong_separation {
        ResolutionStatus::ExplicitlyDistinct
    } else if any_identity {
        ResolutionStatus::CandidateSame
    } else {
        ResolutionStatus::Indeterminate
    };

    let mut unresolved_identity_claim_ids: Vec<String> = left_claims
        .iter()
        .chain(right_claims.iter())
        .map(|claim| claim.claim_id.clone())
        .filter(|claim_id| !matched_claim_ids.contains(claim_id))
        .collect();
    unresolved_identity_claim_ids.sort();

    Ok(EntityResolutionProposal {
        left: left.clone(),
        right: right.clone(),
        status,
        identifier_matches,
        separation_claim_ids,
        unresolved_identity_claim_ids,
    })
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum IdentityValidationError {
    #[error("required identity field `{0}` is empty")]
    EmptyField(&'static str),
    #[error("scoped identifier is missing an authority scope")]
    ScopedIdentifierMissingScope,
    #[error("identity confidence `{field}` must be finite and within [0,1], got {value}")]
    ConfidenceOutOfRange { field: &'static str, value: f32 },
    #[error("identity validity range is inverted: from {from} > until {until}")]
    InvertedValidityRange { from: u64, until: u64 },
    #[error("duplicate evidence observation id `{0}`")]
    DuplicateEvidenceObservationId(ObservationId),
    #[error("duplicate identity claim id `{0}`")]
    DuplicateClaimId(String),
    #[error("separation claim cannot distinguish an entity from itself")]
    SelfSeparation,
}

fn validate_entity(entity: &EntityRef) -> Result<(), IdentityValidationError> {
    require_non_empty("entity.namespace", &entity.namespace)?;
    require_non_empty("entity.kind", &entity.kind)?;
    require_non_empty("entity.id", &entity.id)?;
    Ok(())
}

fn require_non_empty(field: &'static str, value: &str) -> Result<(), IdentityValidationError> {
    if value.trim().is_empty() {
        Err(IdentityValidationError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_probability(
    field: &'static str,
    value: f32,
) -> Result<(), IdentityValidationError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(IdentityValidationError::ConfidenceOutOfRange { field, value })
    }
}

fn validate_interval(
    from: Option<u64>,
    until: Option<u64>,
) -> Result<(), IdentityValidationError> {
    if let (Some(from), Some(until)) = (from, until) {
        if from > until {
            return Err(IdentityValidationError::InvertedValidityRange { from, until });
        }
    }
    Ok(())
}

fn validate_evidence_ids(ids: &[ObservationId]) -> Result<(), IdentityValidationError> {
    let mut seen = BTreeSet::new();
    for id in ids {
        require_non_empty("evidence_observation_id", id.as_str())?;
        if !seen.insert(id.clone()) {
            return Err(IdentityValidationError::DuplicateEvidenceObservationId(
                id.clone(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entity(namespace: &str, id: &str) -> EntityRef {
        EntityRef::new(namespace, "host", id)
    }

    fn source(integration: &str) -> IdentityClaimSource {
        IdentityClaimSource {
            integration_id: integration.into(),
            collector_id: None,
            tenant: None,
        }
    }

    fn claim(
        claim_id: &str,
        subject: EntityRef,
        scheme: &str,
        value: &str,
        uniqueness: IdentifierUniqueness,
        stability: IdentifierStability,
        strength: IdentityStrength,
    ) -> IdentityClaim {
        IdentityClaim {
            claim_id: claim_id.into(),
            subject,
            identifier: ExternalIdentifier {
                scheme: scheme.into(),
                value: value.into(),
                scope: if uniqueness == IdentifierUniqueness::Scoped {
                    Some("tenant-a".into())
                } else {
                    None
                },
                uniqueness,
                stability,
                case_sensitive: true,
            },
            strength,
            source_confidence: 1.0,
            source: source("fixture"),
            observed_at_unix_ms: 100,
            valid_from_unix_ms: None,
            valid_until_unix_ms: None,
            evidence_observation_ids: vec![],
        }
    }

    #[test]
    fn shared_persistent_unique_identifier_is_strong_candidate_not_auto_merge() {
        let left = entity("otel", "service-a");
        let right = entity("aws", "i-123");
        let claims = vec![
            claim(
                "left-id",
                left.clone(),
                "cloud.resource_id",
                "arn:example:123",
                IdentifierUniqueness::Global,
                IdentifierStability::Persistent,
                IdentityStrength::Authoritative,
            ),
            claim(
                "right-id",
                right.clone(),
                "cloud.resource_id",
                "arn:example:123",
                IdentifierUniqueness::Global,
                IdentifierStability::Persistent,
                IdentityStrength::Strong,
            ),
        ];
        let proposal = assess_entity_pair(&left, &right, &claims, &[], 100).unwrap();
        assert_eq!(proposal.status, ResolutionStatus::StrongCandidateSame);
        assert_eq!(proposal.identifier_matches.len(), 1);
        assert_ne!(proposal.left, proposal.right);
    }

    #[test]
    fn shared_ambiguous_service_name_never_becomes_strong_identity() {
        let left = entity("otel", "payments");
        let right = entity("cmdb", "payments");
        let claims = vec![
            claim(
                "left-name",
                left.clone(),
                "service.name",
                "payments",
                IdentifierUniqueness::Ambiguous,
                IdentifierStability::Persistent,
                IdentityStrength::Weak,
            ),
            claim(
                "right-name",
                right.clone(),
                "service.name",
                "payments",
                IdentifierUniqueness::Ambiguous,
                IdentifierStability::Persistent,
                IdentityStrength::Weak,
            ),
        ];
        let proposal = assess_entity_pair(&left, &right, &claims, &[], 100).unwrap();
        assert_eq!(proposal.status, ResolutionStatus::CandidateSame);
    }

    #[test]
    fn strong_separation_and_strong_identity_surface_conflict() {
        let left = entity("otel", "node-a");
        let right = entity("cmdb", "ci-7");
        let claims = vec![
            claim(
                "left-host-id",
                left.clone(),
                "host.id",
                "uuid-7",
                IdentifierUniqueness::Global,
                IdentifierStability::Persistent,
                IdentityStrength::Strong,
            ),
            claim(
                "right-host-id",
                right.clone(),
                "host.id",
                "uuid-7",
                IdentifierUniqueness::Global,
                IdentifierStability::Persistent,
                IdentityStrength::Strong,
            ),
        ];
        let separation = SeparationClaim {
            claim_id: "operator-distinct".into(),
            left: left.clone(),
            right: right.clone(),
            strength: IdentityStrength::Authoritative,
            source_confidence: 1.0,
            source: source("operator-review"),
            observed_at_unix_ms: 100,
            valid_from_unix_ms: None,
            valid_until_unix_ms: None,
            evidence_observation_ids: vec![],
        };
        let proposal =
            assess_entity_pair(&left, &right, &claims, &[separation], 100).unwrap();
        assert_eq!(proposal.status, ResolutionStatus::ConflictingEvidence);
    }

    #[test]
    fn strong_separation_beats_only_weak_shared_alias() {
        let left = entity("otel", "payments-a");
        let right = entity("cmdb", "payments-b");
        let claims = vec![
            claim(
                "left-name",
                left.clone(),
                "service.name",
                "payments",
                IdentifierUniqueness::Ambiguous,
                IdentifierStability::Persistent,
                IdentityStrength::Weak,
            ),
            claim(
                "right-name",
                right.clone(),
                "service.name",
                "payments",
                IdentifierUniqueness::Ambiguous,
                IdentifierStability::Persistent,
                IdentityStrength::Weak,
            ),
        ];
        let separation = SeparationClaim {
            claim_id: "reviewed-distinct".into(),
            left: left.clone(),
            right: right.clone(),
            strength: IdentityStrength::Strong,
            source_confidence: 1.0,
            source: source("cmdb"),
            observed_at_unix_ms: 100,
            valid_from_unix_ms: None,
            valid_until_unix_ms: None,
            evidence_observation_ids: vec![],
        };
        let proposal =
            assess_entity_pair(&left, &right, &claims, &[separation], 100).unwrap();
        assert_eq!(proposal.status, ResolutionStatus::ExplicitlyDistinct);
        assert_eq!(proposal.identifier_matches.len(), 1);
    }

    #[test]
    fn low_extraction_confidence_cannot_create_strong_identity() {
        let left = entity("otel", "a");
        let right = entity("cmdb", "b");
        let mut left_claim = claim(
            "left-host",
            left.clone(),
            "host.id",
            "uuid-x",
            IdentifierUniqueness::Global,
            IdentifierStability::Persistent,
            IdentityStrength::Strong,
        );
        left_claim.source_confidence = 0.5;
        let right_claim = claim(
            "right-host",
            right.clone(),
            "host.id",
            "uuid-x",
            IdentifierUniqueness::Global,
            IdentifierStability::Persistent,
            IdentityStrength::Strong,
        );
        let proposal =
            assess_entity_pair(&left, &right, &[left_claim, right_claim], &[], 100).unwrap();
        assert_eq!(proposal.status, ResolutionStatus::CandidateSame);
        assert_eq!(proposal.identifier_matches[0].min_source_confidence, 0.5);
    }

    #[test]
    fn expired_claim_does_not_generate_candidate_pair() {
        let left = entity("otel", "a");
        let right = entity("prom", "b");
        let mut left_claim = claim(
            "old-a",
            left,
            "host.id",
            "uuid-x",
            IdentifierUniqueness::Global,
            IdentifierStability::Persistent,
            IdentityStrength::Strong,
        );
        left_claim.valid_until_unix_ms = Some(50);
        let right_claim = claim(
            "new-b",
            right,
            "host.id",
            "uuid-x",
            IdentifierUniqueness::Global,
            IdentifierStability::Persistent,
            IdentityStrength::Strong,
        );
        let index = IdentityClaimIndex::build(vec![left_claim, right_claim]).unwrap();
        assert!(index.candidate_pairs_at(100).is_empty());
    }

    #[test]
    fn candidate_index_deduplicates_pairs_across_multiple_shared_identifiers() {
        let left = entity("otel", "a");
        let right = entity("prom", "b");
        let claims = vec![
            claim(
                "a-host",
                left.clone(),
                "host.id",
                "uuid-x",
                IdentifierUniqueness::Global,
                IdentifierStability::Persistent,
                IdentityStrength::Strong,
            ),
            claim(
                "b-host",
                right.clone(),
                "host.id",
                "uuid-x",
                IdentifierUniqueness::Global,
                IdentifierStability::Persistent,
                IdentityStrength::Strong,
            ),
            claim(
                "a-mac",
                left,
                "mac",
                "00:11:22:33:44:55",
                IdentifierUniqueness::Scoped,
                IdentifierStability::Persistent,
                IdentityStrength::Moderate,
            ),
            claim(
                "b-mac",
                right,
                "mac",
                "00:11:22:33:44:55",
                IdentifierUniqueness::Scoped,
                IdentifierStability::Persistent,
                IdentityStrength::Moderate,
            ),
        ];
        let index = IdentityClaimIndex::build(claims).unwrap();
        assert_eq!(index.candidate_pairs_at(100).len(), 1);
    }

    #[test]
    fn scoped_identifier_requires_explicit_scope() {
        let identifier = ExternalIdentifier {
            scheme: "ip".into(),
            value: "10.0.0.1".into(),
            scope: None,
            uniqueness: IdentifierUniqueness::Scoped,
            stability: IdentifierStability::Session,
            case_sensitive: true,
        };
        assert_eq!(
            identifier.validate(),
            Err(IdentityValidationError::ScopedIdentifierMissingScope)
        );
    }
}
