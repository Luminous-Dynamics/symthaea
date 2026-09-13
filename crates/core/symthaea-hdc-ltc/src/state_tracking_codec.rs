// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Shared deterministic HDC codec for the state-tracking benchmark.
//!
//! The codec is deliberately independent of any recurrent architecture. Every
//! HLS/HDC-LTC ablation can therefore receive byte-for-byte identical event and
//! query hypervectors.

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
            Self::LocationOutOfRange(id) => {
                write!(f, "location id {id} is outside the codec codebook")
            }
            Self::InvalidElapsedTime(dt) => {
                write!(f, "elapsed event time must be finite and positive, got {dt}")
            }
            Self::InvalidQueryTime {
                asked_time,
                as_of_time,
            } => write!(
                f,
                "query times must be finite with asked_time >= as_of_time, got asked={asked_time} as_of={as_of_time}"
            ),
        }
    }
}

impl std::error::Error for TrackingCodecError {}

/// Deterministic vector-symbolic encoding shared by all benchmark adapters.
#[derive(Debug, Clone)]
pub struct StateTrackingCodec {
    dim: usize,
    entity_symbols: Vec<ContinuousHV>,
    object_symbols: Vec<ContinuousHV>,
    location_symbols: Vec<ContinuousHV>,
    entity_role: UnitaryRole,
    object_role: UnitaryRole,
    location_role: UnitaryRole,
    time_role: UnitaryRole,
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

        Ok(Self {
            dim,
            entity_symbols: make_codebook(dim, config.entities, seed.wrapping_add(1_000_000)),
            object_symbols: make_codebook(dim, config.objects, seed.wrapping_add(2_000_000)),
            location_symbols: make_codebook(dim, config.locations, seed.wrapping_add(3_000_000)),
            entity_role: UnitaryRole::new(dim, seed.wrapping_add(10)),
            object_role: UnitaryRole::new(dim, seed.wrapping_add(11)),
            location_role: UnitaryRole::new(dim, seed.wrapping_add(12)),
            time_role: UnitaryRole::new(dim, seed.wrapping_add(13)),
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

    pub fn encode_event(
        &self,
        event: &TrackingEvent,
        elapsed_since_previous: f64,
    ) -> Result<ContinuousHV, TrackingCodecError> {
        if !elapsed_since_previous.is_finite() || elapsed_since_previous <= 0.0 {
            return Err(TrackingCodecError::InvalidElapsedTime(elapsed_since_previous));
        }

        let time = self.time_channel(encode_event_dt(elapsed_since_previous));
        let encoded = match event.kind {
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
        };
        Ok(encoded)
    }

    /// Encode a query using its target identity plus a monotone log-lag channel.
    pub fn encode_query(
        &self,
        query: &TrackingQuery,
        asked_time: f64,
    ) -> Result<ContinuousHV, TrackingCodecError> {
        if !asked_time.is_finite()
            || !query.as_of_time.is_finite()
            || asked_time < query.as_of_time
        {
            return Err(TrackingCodecError::InvalidQueryTime {
                asked_time,
                as_of_time: query.as_of_time,
            });
        }

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

    pub fn answer_symbol(
        &self,
        answer: TrackingAnswer,
    ) -> Result<&ContinuousHV, TrackingCodecError> {
        match answer {
            TrackingAnswer::Entity(id) => self.entity_symbol(id),
            TrackingAnswer::Location(id) => self.location_symbol(id),
        }
    }

    pub fn decode_answer_symbol(
        &self,
        vector: &ContinuousHV,
        query_kind: TrackingQueryKind,
    ) -> TrackingAnswer {
        assert_eq!(vector.dim(), self.dim, "answer vector dimension mismatch");
        match query_kind {
            TrackingQueryKind::ObjectOwner { .. } => {
                TrackingAnswer::Entity(nearest(vector, &self.entity_symbols) as EntityId)
            }
            TrackingQueryKind::EntityLocation { .. }
            | TrackingQueryKind::ObjectLocation { .. } => {
                TrackingAnswer::Location(nearest(vector, &self.location_symbols) as LocationId)
            }
        }
    }

    fn entity_symbol(&self, id: EntityId) -> Result<&ContinuousHV, TrackingCodecError> {
        self.entity_symbols
            .get(id as usize)
            .ok_or(TrackingCodecError::EntityOutOfRange(id))
    }

    fn object_symbol(&self, id: ObjectId) -> Result<&ContinuousHV, TrackingCodecError> {
        self.object_symbols
            .get(id as usize)
            .ok_or(TrackingCodecError::ObjectOutOfRange(id))
    }

    fn location_symbol(&self, id: LocationId) -> Result<&ContinuousHV, TrackingCodecError> {
        self.location_symbols
            .get(id as usize)
            .ok_or(TrackingCodecError::LocationOutOfRange(id))
    }

    fn time_channel(&self, scalar: f32) -> ContinuousHV {
        ContinuousHV::from_values(
            self.time_role
                .as_slice()
                .iter()
                .map(|role| role * scalar)
                .collect(),
        )
    }
}

fn make_codebook(dim: usize, count: usize, seed: u64) -> Vec<ContinuousHV> {
    (0..count)
        .map(|index| ContinuousHV::new_random(dim, seed.wrapping_add(index as u64)))
        .collect()
}

fn encode_event_dt(dt: f64) -> f32 {
    (dt.ln() / 12.0).clamp(-1.0, 1.0) as f32
}

fn encode_query_lag(lag: f64) -> f32 {
    (lag.ln_1p() / 12.0).clamp(0.0, 1.0) as f32
}

fn nearest(query: &ContinuousHV, codebook: &[ContinuousHV]) -> usize {
    codebook
        .iter()
        .enumerate()
        .map(|(index, candidate)| (index, query.similarity(candidate)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_tracking_benchmark::StateTrackingBenchmark;

    fn fixture() -> (StateTrackingBenchmark, StateTrackingCodec) {
        let benchmark = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig {
            entities: 8,
            objects: 16,
            locations: 4,
            events: 64,
            query_every: 4,
            historical_query_rate: 1.0,
            seed: 7,
            ..StateTrackingBenchmarkConfig::default()
        })
        .unwrap();
        let codec = StateTrackingCodec::from_benchmark_config(512, &benchmark.config, 99).unwrap();
        (benchmark, codec)
    }

    #[test]
    fn encoding_is_deterministic() {
        let (benchmark, a) = fixture();
        let b = StateTrackingCodec::from_benchmark_config(512, &benchmark.config, 99).unwrap();
        assert_eq!(
            a.encode_event(&benchmark.events[0], benchmark.events[0].time)
                .unwrap(),
            b.encode_event(&benchmark.events[0], benchmark.events[0].time)
                .unwrap()
        );
    }

    #[test]
    fn answer_symbols_roundtrip() {
        let (benchmark, codec) = fixture();
        for query in &benchmark.queries {
            let symbol = codec.answer_symbol(query.expected).unwrap();
            assert_eq!(codec.decode_answer_symbol(symbol, query.kind), query.expected);
        }
    }

    #[test]
    fn historical_query_encoding_differs_from_current_query() {
        let (benchmark, codec) = fixture();
        let historical = benchmark
            .queries
            .iter()
            .find(|query| query.is_historical())
            .unwrap();
        let asked_time = benchmark.events[historical.asked_after_event].time;
        let historical_vector = codec.encode_query(historical, asked_time).unwrap();

        let mut current = historical.clone();
        current.as_of_event = current.asked_after_event;
        current.as_of_time = asked_time;
        assert_ne!(
            historical_vector,
            codec.encode_query(&current, asked_time).unwrap()
        );
    }
}
