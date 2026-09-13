// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic HDC codec for the state-tracking benchmark.
//!
//! In the associative-learning lineage, [`encode_event`](Self::encode_event)
//! writes one-hop relation assignments directly as `relation_key ⊙ value`.
//! Entity/object/location identity symbols are themselves bipolar unitary
//! vectors, so retrieved identities can immediately participate as keys in a
//! subsequent HDC relation lookup. This is standard VSA-style compositional
//! plumbing, not a learned decoder transformation.
//!
//! The earlier architecture-neutral event vector remains available through
//! [`encode_diagnostic_event`](Self::encode_diagnostic_event).

use crate::continuous_hv::{ContinuousHV, UnitaryRole};
use crate::state_tracking_benchmark::{
    EntityId, LocationId, ObjectId, StateTrackingBenchmarkConfig, TrackingAnswer, TrackingEvent,
    TrackingEventKind, TrackingQuery, TrackingQueryKind,
};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum TrackingCodecError {
    ZeroDimension,
    EmptyDomain(&'static str),
    EntityOutOfRange(EntityId),
    ObjectOutOfRange(ObjectId),
    LocationOutOfRange(LocationId),
    InvalidElapsedTime(f64),
    InvalidQueryTime { asked_time: f64, as_of_time: f64 },
}

impl fmt::Display for TrackingCodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "tracking codec dimension must be non-zero"),
            Self::EmptyDomain(name) => write!(f, "tracking codec domain {name} must be non-empty"),
            Self::EntityOutOfRange(id) => write!(f, "entity id {id} is outside the codec codebook"),
            Self::ObjectOutOfRange(id) => write!(f, "object id {id} is outside the codec codebook"),
            Self::LocationOutOfRange(id) => write!(f, "location id {id} is outside the codec codebook"),
            Self::InvalidElapsedTime(dt) => write!(f, "elapsed event time must be finite and positive, got {dt}"),
            Self::InvalidQueryTime { asked_time, as_of_time } => write!(
                f,
                "query times must be finite with asked_time >= as_of_time, got asked={asked_time} as_of={as_of_time}"
            ),
        }
    }
}

impl std::error::Error for TrackingCodecError {}

#[derive(Debug, Clone)]
pub struct StateTrackingCodec {
    dim: usize,
    entity_symbols: Vec<ContinuousHV>,
    object_symbols: Vec<ContinuousHV>,
    location_symbols: Vec<ContinuousHV>,
    entity_keys: Vec<UnitaryRole>,
    object_keys: Vec<UnitaryRole>,
    location_keys: Vec<UnitaryRole>,
    entity_role: UnitaryRole,
    object_role: UnitaryRole,
    location_role: UnitaryRole,
    time_role: UnitaryRole,
    query_entity_location_role: UnitaryRole,
    query_object_owner_role: UnitaryRole,
    query_object_location_role: UnitaryRole,
    query_time_role: UnitaryRole,
    move_marker: ContinuousHV,
    transfer_marker: ContinuousHV,
    query_entity_location_marker: ContinuousHV,
    query_object_owner_marker: ContinuousHV,
    query_object_location_marker: ContinuousHV,
}

impl StateTrackingCodec {
    pub fn from_benchmark_config(
        dim: usize,
        config: &StateTrackingBenchmarkConfig,
        seed: u64,
    ) -> Result<Self, TrackingCodecError> {
        if dim == 0 {
            return Err(TrackingCodecError::ZeroDimension);
        }
        for (name, count) in [
            ("entities", config.entities),
            ("objects", config.objects),
            ("locations", config.locations),
        ] {
            if count == 0 {
                return Err(TrackingCodecError::EmptyDomain(name));
            }
        }

        let entity_keys = make_role_codebook(dim, config.entities, seed.wrapping_add(4_000_000));
        let object_keys = make_role_codebook(dim, config.objects, seed.wrapping_add(5_000_000));
        let location_keys = make_role_codebook(dim, config.locations, seed.wrapping_add(6_000_000));
        let entity_symbols = symbols_from_roles(&entity_keys);
        let object_symbols = symbols_from_roles(&object_keys);
        let location_symbols = symbols_from_roles(&location_keys);

        Ok(Self {
            dim,
            entity_symbols,
            object_symbols,
            location_symbols,
            entity_keys,
            object_keys,
            location_keys,
            entity_role: UnitaryRole::new(dim, seed.wrapping_add(10)),
            object_role: UnitaryRole::new(dim, seed.wrapping_add(11)),
            location_role: UnitaryRole::new(dim, seed.wrapping_add(12)),
            time_role: UnitaryRole::new(dim, seed.wrapping_add(13)),
            query_entity_location_role: UnitaryRole::new(dim, seed.wrapping_add(20)),
            query_object_owner_role: UnitaryRole::new(dim, seed.wrapping_add(21)),
            query_object_location_role: UnitaryRole::new(dim, seed.wrapping_add(22)),
            query_time_role: UnitaryRole::new(dim, seed.wrapping_add(23)),
            move_marker: ContinuousHV::new_random(dim, seed.wrapping_add(100)),
            transfer_marker: ContinuousHV::new_random(dim, seed.wrapping_add(101)),
            query_entity_location_marker: ContinuousHV::new_random(dim, seed.wrapping_add(200)),
            query_object_owner_marker: ContinuousHV::new_random(dim, seed.wrapping_add(201)),
            query_object_location_marker: ContinuousHV::new_random(dim, seed.wrapping_add(202)),
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Associative-learning event representation.
    pub fn encode_event(
        &self,
        event: &TrackingEvent,
        elapsed_since_previous: f64,
    ) -> Result<ContinuousHV, TrackingCodecError> {
        validate_elapsed(elapsed_since_previous)?;
        match event.kind {
            TrackingEventKind::MoveEntity { entity, to } => {
                Ok(self.entity_location_key(entity)?.bind(self.location_symbol(to)?))
            }
            TrackingEventKind::TransferObject { object, to } => {
                Ok(self.object_owner_key(object)?.bind(self.entity_symbol(to)?))
            }
        }
    }

    pub fn encode_associative_event(
        &self,
        event: &TrackingEvent,
        elapsed_since_previous: f64,
    ) -> Result<ContinuousHV, TrackingCodecError> {
        self.encode_event(event, elapsed_since_previous)
    }

    /// Architecture-neutral diagnostic event representation retained for codec
    /// ablations. The earlier #2641 branch is unchanged by this lineage.
    pub fn encode_diagnostic_event(
        &self,
        event: &TrackingEvent,
        elapsed_since_previous: f64,
    ) -> Result<ContinuousHV, TrackingCodecError> {
        validate_elapsed(elapsed_since_previous)?;
        let time = self.time_channel(encode_event_dt(elapsed_since_previous));
        Ok(match event.kind {
            TrackingEventKind::MoveEntity { entity, to } => {
                let entity = self.entity_role.bind(self.entity_symbol(entity)?);
                let location = self.location_role.bind(self.location_symbol(to)?);
                ContinuousHV::bundle(&[&self.move_marker, &entity, &location, &time])
            }
            TrackingEventKind::TransferObject { object, to } => {
                let object = self.object_role.bind(self.object_symbol(object)?);
                let entity = self.entity_role.bind(self.entity_symbol(to)?);
                ContinuousHV::bundle(&[&self.transfer_marker, &object, &entity, &time])
            }
        })
    }

    pub fn encode_query(
        &self,
        query: &TrackingQuery,
        asked_time: f64,
    ) -> Result<ContinuousHV, TrackingCodecError> {
        validate_query_time(query, asked_time)?;
        let time = self.time_channel(encode_query_lag(asked_time - query.as_of_time));
        let (marker, target) = match query.kind {
            TrackingQueryKind::EntityLocation { entity } => (
                &self.query_entity_location_marker,
                self.entity_role.bind(self.entity_symbol(entity)?),
            ),
            TrackingQueryKind::ObjectOwner { object } => (
                &self.query_object_owner_marker,
                self.object_role.bind(self.object_symbol(object)?),
            ),
            TrackingQueryKind::ObjectLocation { object } => (
                &self.query_object_location_marker,
                self.object_role.bind(self.object_symbol(object)?),
            ),
        };
        Ok(ContinuousHV::bundle(&[marker, &target, &time]))
    }

    /// Deterministic unitary key for direct/current or stress-test historical
    /// associative unbinding.
    pub fn query_key(
        &self,
        query: &TrackingQuery,
        asked_time: f64,
    ) -> Result<UnitaryRole, TrackingCodecError> {
        validate_query_time(query, asked_time)?;
        let base = self.current_query_key(query.kind)?;
        if !query.is_historical() {
            return Ok(base);
        }
        let lag = asked_time - query.as_of_time;
        let time_key = self.query_time_role.permute(lag_bucket(lag, self.dim));
        Ok(base.compose(&time_key))
    }

    /// Current object-owner lookup key, exposed for fixed HDC composition.
    pub fn object_owner_key(&self, object: ObjectId) -> Result<UnitaryRole, TrackingCodecError> {
        Ok(self.query_object_owner_role.compose(self.object_key(object)?))
    }

    /// Current entity-location lookup key, exposed for direct checks.
    pub fn entity_location_key(&self, entity: EntityId) -> Result<UnitaryRole, TrackingCodecError> {
        Ok(self.query_entity_location_role.compose(self.entity_key(entity)?))
    }

    /// Relation-only role used to turn a retrieved entity identity into its
    /// entity-location lookup key during two-hop composition.
    pub fn entity_location_relation_role(&self) -> &UnitaryRole {
        &self.query_entity_location_role
    }

    pub fn answer_symbol(&self, answer: TrackingAnswer) -> Result<&ContinuousHV, TrackingCodecError> {
        match answer {
            TrackingAnswer::Entity(id) => self.entity_symbol(id),
            TrackingAnswer::Location(id) => self.location_symbol(id),
        }
    }

    pub fn decode_answer_symbol(&self, vector: &ContinuousHV, query_kind: TrackingQueryKind) -> TrackingAnswer {
        assert_eq!(vector.dim(), self.dim, "answer vector dimension mismatch");
        match query_kind {
            TrackingQueryKind::ObjectOwner { .. } => {
                TrackingAnswer::Entity(nearest(vector, &self.entity_symbols) as EntityId)
            }
            TrackingQueryKind::EntityLocation { .. } | TrackingQueryKind::ObjectLocation { .. } => {
                TrackingAnswer::Location(nearest(vector, &self.location_symbols) as LocationId)
            }
        }
    }

    fn current_query_key(&self, kind: TrackingQueryKind) -> Result<UnitaryRole, TrackingCodecError> {
        match kind {
            TrackingQueryKind::EntityLocation { entity } => self.entity_location_key(entity),
            TrackingQueryKind::ObjectOwner { object } => self.object_owner_key(object),
            TrackingQueryKind::ObjectLocation { object } => {
                Ok(self.query_object_location_role.compose(self.object_key(object)?))
            }
        }
    }

    fn entity_symbol(&self, id: EntityId) -> Result<&ContinuousHV, TrackingCodecError> {
        self.entity_symbols.get(id as usize).ok_or(TrackingCodecError::EntityOutOfRange(id))
    }
    fn object_symbol(&self, id: ObjectId) -> Result<&ContinuousHV, TrackingCodecError> {
        self.object_symbols.get(id as usize).ok_or(TrackingCodecError::ObjectOutOfRange(id))
    }
    fn location_symbol(&self, id: LocationId) -> Result<&ContinuousHV, TrackingCodecError> {
        self.location_symbols.get(id as usize).ok_or(TrackingCodecError::LocationOutOfRange(id))
    }
    fn entity_key(&self, id: EntityId) -> Result<&UnitaryRole, TrackingCodecError> {
        self.entity_keys.get(id as usize).ok_or(TrackingCodecError::EntityOutOfRange(id))
    }
    fn object_key(&self, id: ObjectId) -> Result<&UnitaryRole, TrackingCodecError> {
        self.object_keys.get(id as usize).ok_or(TrackingCodecError::ObjectOutOfRange(id))
    }
    #[allow(dead_code)]
    fn location_key(&self, id: LocationId) -> Result<&UnitaryRole, TrackingCodecError> {
        self.location_keys.get(id as usize).ok_or(TrackingCodecError::LocationOutOfRange(id))
    }
    fn time_channel(&self, scalar: f32) -> ContinuousHV {
        ContinuousHV::from_values(self.time_role.as_slice().iter().map(|role| role * scalar).collect())
    }
}

fn validate_elapsed(dt: f64) -> Result<(), TrackingCodecError> {
    if !dt.is_finite() || dt <= 0.0 {
        Err(TrackingCodecError::InvalidElapsedTime(dt))
    } else {
        Ok(())
    }
}
fn validate_query_time(query: &TrackingQuery, asked_time: f64) -> Result<(), TrackingCodecError> {
    if !asked_time.is_finite() || !query.as_of_time.is_finite() || asked_time < query.as_of_time {
        Err(TrackingCodecError::InvalidQueryTime { asked_time, as_of_time: query.as_of_time })
    } else {
        Ok(())
    }
}
fn make_role_codebook(dim: usize, count: usize, seed: u64) -> Vec<UnitaryRole> {
    (0..count).map(|index| UnitaryRole::new(dim, seed.wrapping_add(index as u64))).collect()
}
fn symbols_from_roles(roles: &[UnitaryRole]) -> Vec<ContinuousHV> {
    roles.iter().map(|role| ContinuousHV::from_values(role.as_slice().to_vec())).collect()
}
fn encode_event_dt(dt: f64) -> f32 { (dt.ln() / 12.0).clamp(-1.0, 1.0) as f32 }
fn encode_query_lag(lag: f64) -> f32 { (lag.ln_1p() / 12.0).clamp(0.0, 1.0) as f32 }
fn lag_bucket(lag: f64, dim: usize) -> usize {
    if dim <= 1 { return 0; }
    let scaled = encode_query_lag(lag) as f64 * (dim - 1) as f64;
    scaled.round().clamp(0.0, (dim - 1) as f64) as usize
}
fn nearest(query: &ContinuousHV, codebook: &[ContinuousHV]) -> usize {
    codebook.iter().enumerate().map(|(i, c)| (i, query.similarity(c)))
        .max_by(|a,b| a.1.total_cmp(&b.1)).map(|(i,_)| i).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_tracking_benchmark::StateTrackingBenchmark;

    fn fixture(dim: usize) -> (StateTrackingBenchmark, StateTrackingCodec) {
        let benchmark = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig {
            entities: 8, objects: 16, locations: 4, events: 64, query_every: 4,
            historical_query_rate: 1.0, seed: 7, ..Default::default()
        }).unwrap();
        let codec = StateTrackingCodec::from_benchmark_config(dim, &benchmark.config, 99).unwrap();
        (benchmark, codec)
    }

    #[test]
    fn current_move_write_unbinds_exact_location() {
        let (_, codec) = fixture(512);
        let event = TrackingEvent { time: 1.0, kind: TrackingEventKind::MoveEntity { entity: 3, to: 2 } };
        let memory = codec.encode_event(&event, 1.0).unwrap();
        let key = codec.entity_location_key(3).unwrap();
        assert_eq!(key.unbind(&memory), codec.answer_symbol(TrackingAnswer::Location(2)).unwrap().clone());
    }

    #[test]
    fn current_transfer_write_unbinds_exact_self_keying_owner() {
        let (_, codec) = fixture(512);
        let event = TrackingEvent { time: 1.0, kind: TrackingEventKind::TransferObject { object: 5, to: 4 } };
        let memory = codec.encode_event(&event, 1.0).unwrap();
        let owner = codec.object_owner_key(5).unwrap().unbind(&memory);
        assert_eq!(owner, codec.answer_symbol(TrackingAnswer::Entity(4)).unwrap().clone());
        let location_key = codec.entity_location_relation_role().bind(&owner);
        assert!(location_key.values.iter().all(|value| *value == -1.0 || *value == 1.0));
    }

    #[test]
    fn two_hop_hdc_composition_recovers_location_from_clean_superposition() {
        let (_, codec) = fixture(8_192);
        let owner_event = TrackingEvent { time: 1.0, kind: TrackingEventKind::TransferObject { object: 5, to: 4 } };
        let location_event = TrackingEvent { time: 2.0, kind: TrackingEventKind::MoveEntity { entity: 4, to: 2 } };
        let owner_memory = codec.encode_event(&owner_event, 1.0).unwrap();
        let location_memory = codec.encode_event(&location_event, 1.0).unwrap();
        let memory = owner_memory.add(&location_memory);

        let owner = codec.object_owner_key(5).unwrap().bind(&memory);
        let soft_location_key = codec.entity_location_relation_role().bind(&owner);
        let location = soft_location_key.bind(&memory);
        assert_eq!(
            codec.decode_answer_symbol(&location, TrackingQueryKind::ObjectLocation { object: 5 }),
            TrackingAnswer::Location(2)
        );
    }

    #[test]
    fn historical_key_is_distinct_from_current_relation_key() {
        let (benchmark, codec) = fixture(512);
        let historical = benchmark.queries.iter().find(|q| q.is_historical()).unwrap();
        let asked = benchmark.events[historical.asked_after_event].time;
        let historical_key = codec.query_key(historical, asked).unwrap();
        let mut current = historical.clone();
        current.as_of_event = current.asked_after_event;
        current.as_of_time = asked;
        assert_ne!(historical_key, codec.query_key(&current, asked).unwrap());
    }

    #[test]
    fn diagnostic_and_associative_surfaces_are_distinct() {
        let (benchmark, codec) = fixture(512);
        let event = &benchmark.events[0];
        assert_ne!(
            codec.encode_event(event, event.time).unwrap(),
            codec.encode_diagnostic_event(event, event.time).unwrap()
        );
    }
}
