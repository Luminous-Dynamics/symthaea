// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic batch evaluation for evidence-bearing entity identity claims.
//!
//! This layer discovers the pairs worth comparing and produces proposals. It
//! deliberately does not collapse, rename, or mutate entities in a world model.
//! Candidate generation and pair assessment are explicitly budgeted so a weak
//! shared alias cannot turn a bounded identity snapshot into quadratic work.

use crate::{
    EntityPair, EntityResolutionProposal, IdentifierUniqueness, IdentityClaim,
    IdentityStrength, IdentityValidationError, ResolutionStatus, SeparationClaim,
    assess_entity_pair,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Central work budget for one identity-resolution pass.
///
/// Limits are intentionally conservative. Exceeding a budget fails the whole
/// resolution pass rather than returning a truncated result that could be
/// mistaken for complete identity knowledge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolutionLimits {
    /// Maximum distinct entities allowed to share one identifier before pair
    /// generation is rejected, even for strong identifiers.
    pub max_subjects_per_identifier: usize,
    /// Tighter fan-out bound for ambiguous or otherwise weak identifiers such
    /// as hostnames, service names, labels, and private addresses.
    pub max_weak_alias_subjects_per_identifier: usize,
    /// Maximum unique entity pairs considered in one resolution pass.
    pub max_candidate_pairs: usize,
    /// Conservative upper bound on pair × claim scans performed by the current
    /// pair assessor. This protects CPU even when pair cardinality itself looks
    /// reasonable. A future indexed assessor may safely raise this budget.
    pub max_pair_claim_scans: u64,
}

impl Default for ResolutionLimits {
    fn default() -> Self {
        Self {
            max_subjects_per_identifier: 2_048,
            max_weak_alias_subjects_per_identifier: 64,
            max_candidate_pairs: 25_000,
            max_pair_claim_scans: 5_000_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityResolutionBatch {
    pub evaluated_at_unix_ms: u64,
    pub proposals: Vec<EntityResolutionProposal>,
}

impl EntityResolutionBatch {
    pub fn strong_candidate_count(&self) -> usize {
        self.count_status(ResolutionStatus::StrongCandidateSame)
    }

    pub fn candidate_count(&self) -> usize {
        self.count_status(ResolutionStatus::CandidateSame)
    }

    pub fn explicit_distinct_count(&self) -> usize {
        self.count_status(ResolutionStatus::ExplicitlyDistinct)
    }

    pub fn conflict_count(&self) -> usize {
        self.count_status(ResolutionStatus::ConflictingEvidence)
    }

    pub fn indeterminate_count(&self) -> usize {
        self.count_status(ResolutionStatus::Indeterminate)
    }

    pub fn has_conflicts(&self) -> bool {
        self.conflict_count() > 0
    }

    pub fn evaluated_pair_count(&self) -> usize {
        self.proposals.len()
    }

    fn count_status(&self, status: ResolutionStatus) -> usize {
        self.proposals
            .iter()
            .filter(|proposal| proposal.status == status)
            .count()
    }
}

/// Evaluate every pair that either shares an active external identifier or has
/// an active explicit separation claim using the default bounded work profile.
pub fn resolve_identity_claims(
    identity_claims: &[IdentityClaim],
    separation_claims: &[SeparationClaim],
    at_unix_ms: u64,
) -> Result<EntityResolutionBatch, ResolutionError> {
    resolve_identity_claims_with_limits(
        identity_claims,
        separation_claims,
        at_unix_ms,
        &ResolutionLimits::default(),
    )
}

/// Bounded deterministic identity resolution.
///
/// Pair discovery uses ordered maps/sets so replaying the same claims at the
/// same evaluation time yields proposals in stable order. No limit breach is
/// silently truncated: callers receive an error and no partial batch.
pub fn resolve_identity_claims_with_limits(
    identity_claims: &[IdentityClaim],
    separation_claims: &[SeparationClaim],
    at_unix_ms: u64,
    limits: &ResolutionLimits,
) -> Result<EntityResolutionBatch, ResolutionError> {
    let mut by_identifier: BTreeMap<String, Vec<&IdentityClaim>> = BTreeMap::new();
    let mut claim_ids = BTreeSet::new();

    for claim in identity_claims {
        claim.validate()?;
        if !claim_ids.insert(claim.claim_id.clone()) {
            return Err(ResolutionError::Identity(
                IdentityValidationError::DuplicateClaimId(claim.claim_id.clone()),
            ));
        }
        if claim.is_active_at(at_unix_ms) {
            by_identifier
                .entry(claim.identifier.canonical_key()?)
                .or_default()
                .push(claim);
        }
    }

    for claim in separation_claims {
        claim.validate()?;
        if !claim_ids.insert(claim.claim_id.clone()) {
            return Err(ResolutionError::Identity(
                IdentityValidationError::DuplicateClaimId(claim.claim_id.clone()),
            ));
        }
    }

    let mut pairs = BTreeSet::new();
    for (identifier, claims) in by_identifier {
        let subjects: BTreeSet<_> = claims.iter().map(|claim| claim.subject.clone()).collect();
        let subject_count = subjects.len();
        if subject_count > limits.max_subjects_per_identifier {
            return Err(ResolutionError::IdentifierFanoutExceeded {
                canonical_identifier: identifier,
                subjects: subject_count,
                limit: limits.max_subjects_per_identifier,
                weak_alias: false,
            });
        }

        let weak_alias = claims.iter().any(|claim| {
            claim.identifier.uniqueness == IdentifierUniqueness::Ambiguous
                || claim.strength < IdentityStrength::Strong
                || claim.source_confidence < 0.9
        });
        if weak_alias && subject_count > limits.max_weak_alias_subjects_per_identifier {
            return Err(ResolutionError::IdentifierFanoutExceeded {
                canonical_identifier: identifier,
                subjects: subject_count,
                limit: limits.max_weak_alias_subjects_per_identifier,
                weak_alias: true,
            });
        }

        let subjects: Vec<_> = subjects.into_iter().collect();
        for left in 0..subjects.len() {
            for right in (left + 1)..subjects.len() {
                if let Some(pair) =
                    EntityPair::new(subjects[left].clone(), subjects[right].clone())
                {
                    if pairs.insert(pair) && pairs.len() > limits.max_candidate_pairs {
                        return Err(ResolutionError::CandidatePairBudgetExceeded {
                            actual_at_least: pairs.len(),
                            limit: limits.max_candidate_pairs,
                        });
                    }
                }
            }
        }
    }

    for claim in separation_claims {
        if claim.is_active_at(at_unix_ms) {
            if let Some(pair) = EntityPair::new(claim.left.clone(), claim.right.clone()) {
                if pairs.insert(pair) && pairs.len() > limits.max_candidate_pairs {
                    return Err(ResolutionError::CandidatePairBudgetExceeded {
                        actual_at_least: pairs.len(),
                        limit: limits.max_candidate_pairs,
                    });
                }
            }
        }
    }

    let claims_per_pair = identity_claims
        .len()
        .saturating_add(separation_claims.len()) as u64;
    let estimated_pair_claim_scans = (pairs.len() as u64).saturating_mul(claims_per_pair);
    if estimated_pair_claim_scans > limits.max_pair_claim_scans {
        return Err(ResolutionError::PairClaimScanBudgetExceeded {
            estimated: estimated_pair_claim_scans,
            limit: limits.max_pair_claim_scans,
            pairs: pairs.len(),
            claims_per_pair: claims_per_pair as usize,
        });
    }

    let mut proposals = Vec::with_capacity(pairs.len());
    for pair in pairs {
        proposals.push(assess_entity_pair(
            &pair.left,
            &pair.right,
            identity_claims,
            separation_claims,
            at_unix_ms,
        )?);
    }

    Ok(EntityResolutionBatch {
        evaluated_at_unix_ms: at_unix_ms,
        proposals,
    })
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ResolutionError {
    #[error("identity claim validation failed: {0}")]
    Identity(#[from] IdentityValidationError),
    #[error(
        "identifier `{canonical_identifier}` fans out to {subjects} entities; limit is {limit} (weak_alias={weak_alias})"
    )]
    IdentifierFanoutExceeded {
        canonical_identifier: String,
        subjects: usize,
        limit: usize,
        weak_alias: bool,
    },
    #[error("identity resolution generated at least {actual_at_least} candidate pairs; limit is {limit}")]
    CandidatePairBudgetExceeded {
        actual_at_least: usize,
        limit: usize,
    },
    #[error(
        "identity resolution would scan approximately {estimated} pair/claim combinations; limit is {limit} ({pairs} pairs × {claims_per_pair} claims)"
    )]
    PairClaimScanBudgetExceeded {
        estimated: u64,
        limit: u64,
        pairs: usize,
        claims_per_pair: usize,
    },
}

/// Return only proposals requiring operator/model attention: contradictory
/// evidence or explicit distinctness. This is a presentation helper, not a
/// policy decision and never mutates the underlying batch.
pub fn attention_required(batch: &EntityResolutionBatch) -> Vec<&EntityResolutionProposal> {
    batch
        .proposals
        .iter()
        .filter(|proposal| {
            matches!(
                proposal.status,
                ResolutionStatus::ConflictingEvidence | ResolutionStatus::ExplicitlyDistinct
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EntityRef, ExternalIdentifier, IdentifierStability, IdentityClaimSource,
    };

    fn entity(namespace: &str, id: &str) -> EntityRef {
        EntityRef::new(namespace, "host", id)
    }

    fn source() -> IdentityClaimSource {
        IdentityClaimSource {
            integration_id: "fixture".into(),
            collector_id: None,
            tenant: None,
        }
    }

    fn claim(claim_id: &str, subject: EntityRef, value: &str) -> IdentityClaim {
        IdentityClaim {
            claim_id: claim_id.into(),
            subject,
            identifier: ExternalIdentifier {
                scheme: "host.id".into(),
                value: value.into(),
                scope: None,
                uniqueness: IdentifierUniqueness::Global,
                stability: IdentifierStability::Persistent,
                case_sensitive: true,
            },
            strength: IdentityStrength::Strong,
            source_confidence: 1.0,
            source: source(),
            observed_at_unix_ms: 100,
            valid_from_unix_ms: None,
            valid_until_unix_ms: None,
            evidence_observation_ids: vec![],
        }
    }

    fn weak_claim(claim_id: &str, subject: EntityRef, value: &str) -> IdentityClaim {
        let mut claim = claim(claim_id, subject, value);
        claim.identifier.scheme = "hostname".into();
        claim.identifier.uniqueness = IdentifierUniqueness::Ambiguous;
        claim.strength = IdentityStrength::Weak;
        claim
    }

    #[test]
    fn shared_identifier_candidates_are_discovered_automatically() {
        let a = entity("otel", "a");
        let b = entity("cmdb", "b");
        let batch = resolve_identity_claims(
            &[
                claim("a-id", a, "uuid-1"),
                claim("b-id", b, "uuid-1"),
            ],
            &[],
            100,
        )
        .unwrap();
        assert_eq!(batch.proposals.len(), 1);
        assert_eq!(batch.strong_candidate_count(), 1);
    }

    #[test]
    fn separation_only_pair_is_not_lost() {
        let a = entity("otel", "a");
        let b = entity("cmdb", "b");
        let separation = SeparationClaim {
            claim_id: "different".into(),
            left: a,
            right: b,
            strength: IdentityStrength::Strong,
            source_confidence: 1.0,
            source: source(),
            observed_at_unix_ms: 100,
            valid_from_unix_ms: None,
            valid_until_unix_ms: None,
            evidence_observation_ids: vec![],
        };
        let batch = resolve_identity_claims(&[], &[separation], 100).unwrap();
        assert_eq!(batch.proposals.len(), 1);
        assert_eq!(batch.explicit_distinct_count(), 1);
        assert_eq!(attention_required(&batch).len(), 1);
    }

    #[test]
    fn conflicting_identity_and_separation_are_counted_not_hidden() {
        let a = entity("otel", "a");
        let b = entity("cmdb", "b");
        let identity = vec![
            claim("a-id", a.clone(), "uuid-1"),
            claim("b-id", b.clone(), "uuid-1"),
        ];
        let separation = SeparationClaim {
            claim_id: "review-disagrees".into(),
            left: a,
            right: b,
            strength: IdentityStrength::Authoritative,
            source_confidence: 1.0,
            source: source(),
            observed_at_unix_ms: 100,
            valid_from_unix_ms: None,
            valid_until_unix_ms: None,
            evidence_observation_ids: vec![],
        };
        let batch = resolve_identity_claims(&identity, &[separation], 100).unwrap();
        assert!(batch.has_conflicts());
        assert_eq!(batch.conflict_count(), 1);
    }

    #[test]
    fn repeated_shared_identifiers_do_not_duplicate_pair_proposals() {
        let a = entity("otel", "a");
        let b = entity("cmdb", "b");
        let claims = vec![
            claim("a-1", a.clone(), "uuid-1"),
            claim("b-1", b.clone(), "uuid-1"),
            claim("a-2", a, "uuid-1"),
            claim("b-2", b, "uuid-1"),
        ];
        let batch = resolve_identity_claims(&claims, &[], 100).unwrap();
        assert_eq!(batch.proposals.len(), 1);
    }

    #[test]
    fn weak_alias_fanout_fails_before_quadratic_pair_generation() {
        let claims: Vec<_> = (0..5)
            .map(|index| {
                weak_claim(
                    &format!("weak-{index}"),
                    entity("source", &format!("host-{index}")),
                    "localhost",
                )
            })
            .collect();
        let limits = ResolutionLimits {
            max_weak_alias_subjects_per_identifier: 4,
            ..ResolutionLimits::default()
        };
        let result = resolve_identity_claims_with_limits(&claims, &[], 100, &limits);
        assert!(matches!(
            result,
            Err(ResolutionError::IdentifierFanoutExceeded {
                weak_alias: true,
                subjects: 5,
                limit: 4,
                ..
            })
        ));
    }

    #[test]
    fn candidate_pair_budget_fails_closed_without_truncation() {
        let claims = vec![
            claim("a", entity("src", "a"), "shared"),
            claim("b", entity("src", "b"), "shared"),
            claim("c", entity("src", "c"), "shared"),
        ];
        let limits = ResolutionLimits {
            max_candidate_pairs: 2,
            ..ResolutionLimits::default()
        };
        let result = resolve_identity_claims_with_limits(&claims, &[], 100, &limits);
        assert!(matches!(
            result,
            Err(ResolutionError::CandidatePairBudgetExceeded {
                actual_at_least: 3,
                limit: 2
            })
        ));
    }

    #[test]
    fn pair_claim_scan_budget_bounds_current_linear_pair_assessor() {
        let claims = vec![
            claim("a", entity("src", "a"), "shared"),
            claim("b", entity("src", "b"), "shared"),
            claim("c", entity("src", "c"), "shared"),
        ];
        let limits = ResolutionLimits {
            max_pair_claim_scans: 8,
            ..ResolutionLimits::default()
        };
        let result = resolve_identity_claims_with_limits(&claims, &[], 100, &limits);
        assert!(matches!(
            result,
            Err(ResolutionError::PairClaimScanBudgetExceeded {
                estimated: 9,
                limit: 8,
                pairs: 3,
                claims_per_pair: 3,
            })
        ));
    }

    #[test]
    fn separation_pairs_also_count_toward_pair_budget() {
        let separations = vec![
            SeparationClaim {
                claim_id: "s1".into(),
                left: entity("src", "a"),
                right: entity("src", "b"),
                strength: IdentityStrength::Strong,
                source_confidence: 1.0,
                source: source(),
                observed_at_unix_ms: 100,
                valid_from_unix_ms: None,
                valid_until_unix_ms: None,
                evidence_observation_ids: vec![],
            },
            SeparationClaim {
                claim_id: "s2".into(),
                left: entity("src", "a"),
                right: entity("src", "c"),
                strength: IdentityStrength::Strong,
                source_confidence: 1.0,
                source: source(),
                observed_at_unix_ms: 100,
                valid_from_unix_ms: None,
                valid_until_unix_ms: None,
                evidence_observation_ids: vec![],
            },
        ];
        let limits = ResolutionLimits {
            max_candidate_pairs: 1,
            ..ResolutionLimits::default()
        };
        let result = resolve_identity_claims_with_limits(&[], &separations, 100, &limits);
        assert!(matches!(
            result,
            Err(ResolutionError::CandidatePairBudgetExceeded {
                actual_at_least: 2,
                limit: 1
            })
        ));
    }
}
