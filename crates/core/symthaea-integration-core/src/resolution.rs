// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic batch evaluation for evidence-bearing entity identity claims.
//!
//! This layer discovers the pairs worth comparing and produces proposals. It
//! deliberately does not collapse, rename, or mutate entities in a world model.

use crate::{
    EntityPair, EntityResolutionProposal, IdentityClaim, IdentityClaimIndex,
    IdentityValidationError, ResolutionStatus, SeparationClaim, assess_entity_pair,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

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

    fn count_status(&self, status: ResolutionStatus) -> usize {
        self.proposals
            .iter()
            .filter(|proposal| proposal.status == status)
            .count()
    }
}

/// Evaluate every pair that either shares an active external identifier or has
/// an active explicit separation claim.
///
/// Pair discovery is deterministic (`BTreeSet` ordering) so replaying the same
/// claims at the same evaluation time yields proposals in stable order.
pub fn resolve_identity_claims(
    identity_claims: &[IdentityClaim],
    separation_claims: &[SeparationClaim],
    at_unix_ms: u64,
) -> Result<EntityResolutionBatch, IdentityValidationError> {
    let index = IdentityClaimIndex::build(identity_claims.to_vec())?;
    let mut pairs = index.candidate_pairs_at(at_unix_ms);

    for claim in separation_claims {
        claim.validate()?;
        if claim.is_active_at(at_unix_ms) {
            if let Some(pair) = EntityPair::new(claim.left.clone(), claim.right.clone()) {
                pairs.insert(pair);
            }
        }
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
        EntityRef, ExternalIdentifier, IdentifierStability, IdentifierUniqueness,
        IdentityClaimSource, IdentityStrength,
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
}
