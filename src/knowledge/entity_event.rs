// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Persistent entity / event / relation identity for the knowledge layer.
//!
//! HDC similarity is useful for association, but similarity is not identity.
//! This store therefore resolves only canonical names and aliases that have been
//! explicitly registered. Fuzzy or embedding-based matches may propose aliases
//! in higher layers, but they must not silently merge persistent entities here.

use super::claim_evidence::ClaimId;
use super::extraction::EntityType;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EntityId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EventId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RelationId(pub u64);

/// Persistent identity and claim attachments for a real or conceptual entity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistentEntity {
    pub id: EntityId,
    pub canonical_name: String,
    pub entity_type: EntityType,
    pub aliases: BTreeSet<String>,
    pub created_at_cycle: u64,
    /// When present, the entity is no longer treated as active after this cycle.
    /// Historical identity remains addressable.
    pub retired_at_cycle: Option<u64>,
    /// Claims describing properties/state/history of this entity.
    pub claim_ids: Vec<ClaimId>,
}

/// An event with stable identity and explicitly role-labelled participants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistentEvent {
    pub id: EventId,
    pub event_type: String,
    /// Role label -> persistent entity. BTreeMap gives deterministic ordering.
    pub participants: BTreeMap<String, EntityId>,
    pub start_cycle: Option<u64>,
    pub end_cycle: Option<u64>,
    pub claim_ids: Vec<ClaimId>,
}

/// A temporally scoped typed relation between two persistent entities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntityRelation {
    pub id: RelationId,
    pub subject: EntityId,
    pub predicate: String,
    pub object: EntityId,
    pub valid_from_cycle: u64,
    /// Inclusive final cycle when known; `None` means open-ended/unknown.
    pub valid_until_cycle: Option<u64>,
    /// Optional epistemic claim that asserts/supports this relation.
    pub claim_id: Option<ClaimId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntityEventError {
    EmptyName,
    EmptyAlias,
    EmptyRole,
    EmptyPredicate,
    UnknownEntity(EntityId),
    UnknownEvent(EventId),
    AliasAlreadyBound {
        alias: String,
        existing: EntityId,
        requested: EntityId,
    },
    InvalidInterval {
        start: u64,
        end: u64,
    },
}

impl fmt::Display for EntityEventError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyName => write!(f, "entity canonical name cannot be empty"),
            Self::EmptyAlias => write!(f, "entity alias cannot be empty"),
            Self::EmptyRole => write!(f, "event participant role cannot be empty"),
            Self::EmptyPredicate => write!(f, "entity relation predicate cannot be empty"),
            Self::UnknownEntity(id) => write!(f, "unknown entity id {}", id.0),
            Self::UnknownEvent(id) => write!(f, "unknown event id {}", id.0),
            Self::AliasAlreadyBound {
                alias,
                existing,
                requested,
            } => write!(
                f,
                "alias '{alias}' is already bound to entity {}, not entity {}",
                existing.0, requested.0
            ),
            Self::InvalidInterval { start, end } => {
                write!(f, "invalid temporal interval: end {end} precedes start {start}")
            }
        }
    }
}

impl Error for EntityEventError {}

/// Canonical entity-event spine for the knowledge layer.
#[derive(Debug, Clone)]
pub struct EntityEventStore {
    entities: HashMap<EntityId, PersistentEntity>,
    events: HashMap<EventId, PersistentEvent>,
    relations: HashMap<RelationId, EntityRelation>,
    /// Normalized canonical name or explicit alias -> stable entity ID.
    name_index: HashMap<String, EntityId>,
    next_entity_id: u64,
    next_event_id: u64,
    next_relation_id: u64,
}

impl Default for EntityEventStore {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityEventStore {
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            events: HashMap::new(),
            relations: HashMap::new(),
            name_index: HashMap::new(),
            next_entity_id: 1,
            next_event_id: 1,
            next_relation_id: 1,
        }
    }

    /// Create an entity with a stable identity.
    ///
    /// An already registered canonical name resolves to the existing entity only
    /// when its type also matches. Otherwise the name is treated as an occupied
    /// alias and the operation fails closed rather than merging identities.
    pub fn create_entity(
        &mut self,
        canonical_name: impl Into<String>,
        entity_type: EntityType,
        created_at_cycle: u64,
    ) -> Result<EntityId, EntityEventError> {
        let canonical_name = canonical_name.into();
        let key = normalize_name(&canonical_name).ok_or(EntityEventError::EmptyName)?;

        if let Some(existing_id) = self.name_index.get(&key).copied() {
            if self
                .entities
                .get(&existing_id)
                .is_some_and(|entity| entity.entity_type == entity_type)
            {
                return Ok(existing_id);
            }
            return Err(EntityEventError::AliasAlreadyBound {
                alias: canonical_name,
                existing: existing_id,
                requested: EntityId(self.next_entity_id),
            });
        }

        let id = EntityId(self.next_entity_id);
        self.next_entity_id += 1;
        self.entities.insert(
            id,
            PersistentEntity {
                id,
                canonical_name: canonical_name.clone(),
                entity_type,
                aliases: BTreeSet::new(),
                created_at_cycle,
                retired_at_cycle: None,
                claim_ids: Vec::new(),
            },
        );
        self.name_index.insert(key, id);
        Ok(id)
    }

    /// Explicitly bind an alias to an existing entity.
    pub fn add_alias(
        &mut self,
        entity_id: EntityId,
        alias: impl Into<String>,
    ) -> Result<(), EntityEventError> {
        if !self.entities.contains_key(&entity_id) {
            return Err(EntityEventError::UnknownEntity(entity_id));
        }

        let alias = alias.into();
        let key = normalize_name(&alias).ok_or(EntityEventError::EmptyAlias)?;
        if let Some(existing_id) = self.name_index.get(&key).copied() {
            if existing_id == entity_id {
                return Ok(());
            }
            return Err(EntityEventError::AliasAlreadyBound {
                alias,
                existing: existing_id,
                requested: entity_id,
            });
        }

        self.name_index.insert(key, entity_id);
        self.entities
            .get_mut(&entity_id)
            .expect("entity existence checked above")
            .aliases
            .insert(alias);
        Ok(())
    }

    /// Resolve only explicit canonical names/aliases; no fuzzy identity merge.
    pub fn resolve_name(&self, name: &str) -> Option<EntityId> {
        normalize_name(name).and_then(|key| self.name_index.get(&key).copied())
    }

    pub fn entity(&self, id: EntityId) -> Option<&PersistentEntity> {
        self.entities.get(&id)
    }

    pub fn event(&self, id: EventId) -> Option<&PersistentEvent> {
        self.events.get(&id)
    }

    pub fn relation(&self, id: RelationId) -> Option<&EntityRelation> {
        self.relations.get(&id)
    }

    pub fn attach_entity_claim(
        &mut self,
        entity_id: EntityId,
        claim_id: ClaimId,
    ) -> Result<(), EntityEventError> {
        let entity = self
            .entities
            .get_mut(&entity_id)
            .ok_or(EntityEventError::UnknownEntity(entity_id))?;
        if !entity.claim_ids.contains(&claim_id) {
            entity.claim_ids.push(claim_id);
        }
        Ok(())
    }

    pub fn retire_entity(
        &mut self,
        entity_id: EntityId,
        retired_at_cycle: u64,
    ) -> Result<(), EntityEventError> {
        let entity = self
            .entities
            .get_mut(&entity_id)
            .ok_or(EntityEventError::UnknownEntity(entity_id))?;
        if retired_at_cycle < entity.created_at_cycle {
            return Err(EntityEventError::InvalidInterval {
                start: entity.created_at_cycle,
                end: retired_at_cycle,
            });
        }
        entity.retired_at_cycle = Some(retired_at_cycle);
        Ok(())
    }

    pub fn create_event(
        &mut self,
        event_type: impl Into<String>,
        participants: BTreeMap<String, EntityId>,
        start_cycle: Option<u64>,
        end_cycle: Option<u64>,
        claim_ids: Vec<ClaimId>,
    ) -> Result<EventId, EntityEventError> {
        if let (Some(start), Some(end)) = (start_cycle, end_cycle) {
            if end < start {
                return Err(EntityEventError::InvalidInterval { start, end });
            }
        }

        for (role, entity_id) in &participants {
            if role.trim().is_empty() {
                return Err(EntityEventError::EmptyRole);
            }
            if !self.entities.contains_key(entity_id) {
                return Err(EntityEventError::UnknownEntity(*entity_id));
            }
        }

        let id = EventId(self.next_event_id);
        self.next_event_id += 1;
        self.events.insert(
            id,
            PersistentEvent {
                id,
                event_type: event_type.into(),
                participants,
                start_cycle,
                end_cycle,
                claim_ids,
            },
        );
        Ok(id)
    }

    pub fn attach_event_claim(
        &mut self,
        event_id: EventId,
        claim_id: ClaimId,
    ) -> Result<(), EntityEventError> {
        let event = self
            .events
            .get_mut(&event_id)
            .ok_or(EntityEventError::UnknownEvent(event_id))?;
        if !event.claim_ids.contains(&claim_id) {
            event.claim_ids.push(claim_id);
        }
        Ok(())
    }

    pub fn add_relation(
        &mut self,
        subject: EntityId,
        predicate: impl Into<String>,
        object: EntityId,
        valid_from_cycle: u64,
        valid_until_cycle: Option<u64>,
        claim_id: Option<ClaimId>,
    ) -> Result<RelationId, EntityEventError> {
        if !self.entities.contains_key(&subject) {
            return Err(EntityEventError::UnknownEntity(subject));
        }
        if !self.entities.contains_key(&object) {
            return Err(EntityEventError::UnknownEntity(object));
        }

        let predicate = predicate.into();
        if predicate.trim().is_empty() {
            return Err(EntityEventError::EmptyPredicate);
        }
        if let Some(end) = valid_until_cycle {
            if end < valid_from_cycle {
                return Err(EntityEventError::InvalidInterval {
                    start: valid_from_cycle,
                    end,
                });
            }
        }

        let id = RelationId(self.next_relation_id);
        self.next_relation_id += 1;
        self.relations.insert(
            id,
            EntityRelation {
                id,
                subject,
                predicate,
                object,
                valid_from_cycle,
                valid_until_cycle,
                claim_id,
            },
        );
        Ok(id)
    }

    /// All relations whose temporal interval contains `cycle`.
    pub fn relations_at(&self, cycle: u64) -> Vec<&EntityRelation> {
        self.relations
            .values()
            .filter(|relation| {
                relation.valid_from_cycle <= cycle
                    && relation
                        .valid_until_cycle
                        .is_none_or(|end| cycle <= end)
            })
            .collect()
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    pub fn relation_count(&self) -> usize {
        self.relations.len()
    }
}

/// Minimal name normalization for explicit identity lookup.
///
/// Deliberately does not stem, fuzzy-match, transliterate, or apply semantic
/// similarity: those operations are useful for *candidate generation*, not for
/// canonical identity assertion.
fn normalize_name(name: &str) -> Option<String> {
    let normalized = name.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
    (!normalized.is_empty()).then_some(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_alias_preserves_stable_identity() {
        let mut store = EntityEventStore::new();
        let entity = store
            .create_entity("International Business Machines", EntityType::Organization, 1)
            .unwrap();
        store.add_alias(entity, "IBM").unwrap();

        assert_eq!(store.resolve_name("ibm"), Some(entity));
        assert_eq!(
            store.resolve_name("  International   Business Machines  "),
            Some(entity)
        );
        assert_eq!(store.entity(entity).unwrap().canonical_name, "International Business Machines");
    }

    #[test]
    fn alias_collision_fails_closed_instead_of_merging() {
        let mut store = EntityEventStore::new();
        let first = store
            .create_entity("Alpha Research", EntityType::Organization, 1)
            .unwrap();
        let second = store
            .create_entity("Beta Research", EntityType::Organization, 1)
            .unwrap();
        store.add_alias(first, "AR").unwrap();

        let error = store.add_alias(second, "ar").unwrap_err();
        assert_eq!(
            error,
            EntityEventError::AliasAlreadyBound {
                alias: "ar".into(),
                existing: first,
                requested: second,
            }
        );
        assert_eq!(store.resolve_name("AR"), Some(first));
    }

    #[test]
    fn fuzzy_similarity_does_not_create_identity() {
        let mut store = EntityEventStore::new();
        store
            .create_entity("Mercury the planet", EntityType::Place, 1)
            .unwrap();

        assert_eq!(store.resolve_name("Mercury"), None);
        assert_eq!(store.resolve_name("planet Mercury"), None);
    }

    #[test]
    fn event_participants_must_be_known_entities() {
        let mut store = EntityEventStore::new();
        let alice = store
            .create_entity("Alice", EntityType::Person, 1)
            .unwrap();
        let mut participants = BTreeMap::new();
        participants.insert("agent".into(), alice);
        participants.insert("patient".into(), EntityId(999));

        assert_eq!(
            store.create_event("transfer", participants, Some(2), Some(3), vec![]),
            Err(EntityEventError::UnknownEntity(EntityId(999)))
        );
    }

    #[test]
    fn temporal_relations_preserve_history_instead_of_overwriting() {
        let mut store = EntityEventStore::new();
        let alice = store
            .create_entity("Alice", EntityType::Person, 1)
            .unwrap();
        let org_a = store
            .create_entity("Org A", EntityType::Organization, 1)
            .unwrap();
        let org_b = store
            .create_entity("Org B", EntityType::Organization, 1)
            .unwrap();

        store
            .add_relation(alice, "member_of", org_a, 1, Some(9), None)
            .unwrap();
        store
            .add_relation(alice, "member_of", org_b, 10, None, None)
            .unwrap();

        let at_5 = store.relations_at(5);
        let at_10 = store.relations_at(10);
        assert_eq!(at_5.len(), 1);
        assert_eq!(at_5[0].object, org_a);
        assert_eq!(at_10.len(), 1);
        assert_eq!(at_10[0].object, org_b);
        assert_eq!(store.relation_count(), 2);
    }

    #[test]
    fn invalid_temporal_intervals_fail_closed() {
        let mut store = EntityEventStore::new();
        let a = store.create_entity("A", EntityType::Concept, 5).unwrap();
        let b = store.create_entity("B", EntityType::Concept, 5).unwrap();

        assert_eq!(
            store.add_relation(a, "related_to", b, 10, Some(9), None),
            Err(EntityEventError::InvalidInterval { start: 10, end: 9 })
        );
        assert_eq!(
            store.retire_entity(a, 4),
            Err(EntityEventError::InvalidInterval { start: 5, end: 4 })
        );
    }
}
