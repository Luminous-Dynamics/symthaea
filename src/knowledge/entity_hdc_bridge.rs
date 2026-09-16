// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shadow-mode bridge between persistent entity identity and HDC representations.
//!
//! This module is intentionally observational. It can encode a persistent entity
//! into an HDC semantic signature and rank explicitly supplied entity candidates
//! against an HDC observation, but it cannot mutate entity identity, knowledge
//! confidence, the causal model, the CfC/JEPA world model, or action selection.
//!
//! The key boundary is:
//!
//! ```text
//! HDC similarity => association candidate
//! HDC similarity != identity assertion
//! ```

use super::claim_evidence::{ClaimId, EpistemicLedger};
use super::encoding::KnowledgeEncoder;
use super::entity_event::{EntityEventStore, EntityId};
use std::cmp::Ordering;
use std::error::Error;
use std::fmt;
use symthaea_core::hdc::unified_hv::BinaryHV;

/// Deterministic HDC representation assembled from a persistent entity record.
#[derive(Debug, Clone)]
pub struct EntitySemanticSignature {
    pub entity_id: EntityId,
    pub vector: BinaryHV,
    /// Number of role-bound semantic components bundled into `vector`.
    pub component_count: usize,
}

/// One shadow-mode association candidate for an HDC observation.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityAssociationCandidate {
    pub entity_id: EntityId,
    pub similarity: f32,
    pub component_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntityHdcBridgeError {
    UnknownEntity(EntityId),
    /// Entity claim attachments are not silently ignored if the ledger cannot
    /// resolve them. A dangling attachment is an integrity failure.
    UnknownClaim {
        entity_id: EntityId,
        claim_id: ClaimId,
    },
}

impl fmt::Display for EntityHdcBridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownEntity(id) => write!(f, "unknown entity id {}", id.0),
            Self::UnknownClaim {
                entity_id,
                claim_id,
            } => write!(
                f,
                "entity {} references unknown claim {}",
                entity_id.0, claim_id.0
            ),
        }
    }
}

impl Error for EntityHdcBridgeError {}

/// Read-only semantic bridge for evaluating entity/world-representation alignment.
///
/// The only mutable state is the encoder's token cache. Persistent stores and
/// cognitive/world-model state are accepted through shared references only.
pub struct ShadowEntityHdcBridge {
    encoder: KnowledgeEncoder,
}

impl Default for ShadowEntityHdcBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl ShadowEntityHdcBridge {
    pub fn new() -> Self {
        Self {
            encoder: KnowledgeEncoder::new(),
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self {
            encoder: KnowledgeEncoder::with_seed(seed),
        }
    }

    /// Encode one persistent entity into a deterministic role-bound signature.
    ///
    /// Components include canonical identity, entity type, explicit aliases, and
    /// attached claim text. Claim evidence is deliberately *not* folded into the
    /// signature: evidence quality belongs to epistemic qualification, not entity
    /// identity representation.
    pub fn encode_entity_signature(
        &mut self,
        store: &EntityEventStore,
        ledger: &EpistemicLedger,
        entity_id: EntityId,
    ) -> Result<EntitySemanticSignature, EntityHdcBridgeError> {
        let entity = store
            .entity(entity_id)
            .ok_or(EntityHdcBridgeError::UnknownEntity(entity_id))?;

        let mut components = Vec::new();
        components.push(self.bind_component("entity:canonical", &entity.canonical_name));
        components.push(self.bind_component(
            "entity:type",
            &format!("{:?}", entity.entity_type),
        ));

        for alias in &entity.aliases {
            components.push(self.bind_component("entity:alias", alias));
        }

        let mut claim_ids = entity.claim_ids.clone();
        claim_ids.sort_unstable();
        claim_ids.dedup();
        for claim_id in claim_ids {
            let claim = ledger
                .claim(claim_id)
                .ok_or(EntityHdcBridgeError::UnknownClaim {
                    entity_id,
                    claim_id,
                })?;
            components.push(self.bind_component("entity:claim", &claim.statement));
            components.push(self.bind_component(
                "entity:claim-kind",
                &format!("{:?}", claim.kind),
            ));
            if let Some(domain) = claim.domain.as_deref() {
                components.push(self.bind_component("entity:claim-domain", domain));
            }
            if let Some(scope) = claim.scope.as_deref() {
                components.push(self.bind_component("entity:claim-scope", scope));
            }
        }

        let component_count = components.len();
        let vector = BinaryHV::bundle(&components);

        Ok(EntitySemanticSignature {
            entity_id,
            vector,
            component_count,
        })
    }

    /// Rank an explicit candidate set against an HDC observation.
    ///
    /// Candidate IDs are supplied by the caller on purpose. This bridge does not
    /// enumerate, create, merge, or otherwise alter persistent identities. Scores
    /// are association diagnostics only and carry no admission threshold here.
    pub fn rank_candidates(
        &mut self,
        observation: &BinaryHV,
        store: &EntityEventStore,
        ledger: &EpistemicLedger,
        candidate_ids: &[EntityId],
        max_results: usize,
    ) -> Result<Vec<EntityAssociationCandidate>, EntityHdcBridgeError> {
        let mut candidates = Vec::with_capacity(candidate_ids.len());
        for entity_id in candidate_ids.iter().copied() {
            let signature = self.encode_entity_signature(store, ledger, entity_id)?;
            candidates.push(EntityAssociationCandidate {
                entity_id,
                similarity: signature.vector.similarity(observation),
                component_count: signature.component_count,
            });
        }

        candidates.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.entity_id.cmp(&b.entity_id))
        });
        candidates.truncate(max_results.min(candidates.len()));
        Ok(candidates)
    }

    fn bind_component(&mut self, role: &str, value: &str) -> BinaryHV {
        let role_hv = self.encoder.encode_token(role);
        let value_hv = self.encoder.encode_token(value);
        role_hv.bind(&value_hv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{ClaimKind, EntityType};

    #[test]
    fn exact_signature_is_top_shadow_candidate_without_identity_mutation() {
        let mut store = EntityEventStore::new();
        let mut ledger = EpistemicLedger::new();
        let alpha = store
            .create_entity("Alpha Laboratory", EntityType::Organization, 1)
            .unwrap();
        let beta = store
            .create_entity("Beta Laboratory", EntityType::Organization, 1)
            .unwrap();
        store.add_alias(alpha, "Alpha Lab").unwrap();
        let claim = ledger.add_claim(
            "Alpha Laboratory studies photonics",
            ClaimKind::Descriptive,
            Some("science".into()),
            None,
            2,
        );
        store.attach_entity_claim(alpha, claim).unwrap();

        let entity_count_before = store.entity_count();
        let claim_count_before = ledger.claim_count();
        let alpha_alias_before = store.resolve_name("Alpha Lab");

        let mut bridge = ShadowEntityHdcBridge::with_seed(7);
        let observation = bridge
            .encode_entity_signature(&store, &ledger, alpha)
            .unwrap()
            .vector;
        let ranked = bridge
            .rank_candidates(&observation, &store, &ledger, &[beta, alpha], 2)
            .unwrap();

        assert_eq!(ranked[0].entity_id, alpha);
        assert!((ranked[0].similarity - 1.0).abs() < f32::EPSILON);
        assert_eq!(store.entity_count(), entity_count_before);
        assert_eq!(ledger.claim_count(), claim_count_before);
        assert_eq!(store.resolve_name("Alpha Lab"), alpha_alias_before);
    }

    #[test]
    fn high_similarity_does_not_create_alias_or_merge_identity() {
        let mut store = EntityEventStore::new();
        let ledger = EpistemicLedger::new();
        let mercury_planet = store
            .create_entity("Mercury the planet", EntityType::Place, 1)
            .unwrap();
        let mercury_element = store
            .create_entity("Mercury the element", EntityType::Artifact, 1)
            .unwrap();

        let mut bridge = ShadowEntityHdcBridge::with_seed(11);
        let observation = bridge
            .encode_entity_signature(&store, &ledger, mercury_planet)
            .unwrap()
            .vector;
        let _ = bridge
            .rank_candidates(
                &observation,
                &store,
                &ledger,
                &[mercury_planet, mercury_element],
                2,
            )
            .unwrap();

        assert_eq!(store.resolve_name("Mercury"), None);
        assert_ne!(mercury_planet, mercury_element);
    }

    #[test]
    fn dangling_entity_claim_fails_closed() {
        let mut store = EntityEventStore::new();
        let ledger = EpistemicLedger::new();
        let entity = store
            .create_entity("Test Entity", EntityType::Concept, 1)
            .unwrap();
        store
            .attach_entity_claim(entity, ClaimId(999))
            .expect("entity store does not own ledger validation");

        let mut bridge = ShadowEntityHdcBridge::with_seed(13);
        assert_eq!(
            bridge
                .encode_entity_signature(&store, &ledger, entity)
                .unwrap_err(),
            EntityHdcBridgeError::UnknownClaim {
                entity_id: entity,
                claim_id: ClaimId(999),
            }
        );
    }

    #[test]
    fn explicit_candidate_set_is_respected() {
        let mut store = EntityEventStore::new();
        let ledger = EpistemicLedger::new();
        let alpha = store
            .create_entity("Alpha", EntityType::Concept, 1)
            .unwrap();
        let beta = store
            .create_entity("Beta", EntityType::Concept, 1)
            .unwrap();

        let mut bridge = ShadowEntityHdcBridge::with_seed(17);
        let observation = bridge
            .encode_entity_signature(&store, &ledger, alpha)
            .unwrap()
            .vector;
        let ranked = bridge
            .rank_candidates(&observation, &store, &ledger, &[beta], 10)
            .unwrap();

        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].entity_id, beta);
        assert!(ranked.iter().all(|candidate| candidate.entity_id != alpha));
    }
}