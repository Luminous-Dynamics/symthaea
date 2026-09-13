// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic irregular-time compositional state-tracking benchmark.
//!
//! This module is intentionally model-agnostic. It generates an exact symbolic
//! world, event stream, query stream, and oracle so HLS, legacy HDC-LTC, CfC,
//! recurrent, SSM, and attention baselines can be evaluated on the same task.
//!
//! World structure:
//!
//! ```text
//! object --owned_by--> entity --located_at--> location
//! ```
//!
//! Every generated episode begins with an **observable initialization prefix**:
//! one `MoveEntity` fact for every entity and one `TransferObject` fact for every
//! object, deterministically shuffled. Only after that complete world snapshot
//! has been presented do ordinary mutations and scored queries begin. This keeps
//! train/test initial worlds seed-randomized without making any scored fact
//! unknowable to the model.
//!
//! Ownership and location then mutate independently. `ObjectLocation` requires
//! composition across two mutable relations. Historical queries ask about an
//! earlier fully-observed world state after later events have been seen.

use serde::{Deserialize, Serialize};
use std::fmt;

pub type EntityId = u16;
pub type ObjectId = u16;
pub type LocationId = u16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackingEventKind {
    MoveEntity { entity: EntityId, to: LocationId },
    TransferObject { object: ObjectId, to: EntityId },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackingEvent {
    pub time: f64,
    pub kind: TrackingEventKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackingQueryKind {
    EntityLocation { entity: EntityId },
    ObjectOwner { object: ObjectId },
    ObjectLocation { object: ObjectId },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackingAnswer {
    Entity(EntityId),
    Location(LocationId),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackingQuery {
    pub asked_after_event: usize,
    pub as_of_event: usize,
    pub as_of_time: f64,
    pub kind: TrackingQueryKind,
    pub expected: TrackingAnswer,
}

impl TrackingQuery {
    pub fn is_historical(&self) -> bool {
        self.as_of_event < self.asked_after_event
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTrackingBenchmarkConfig {
    pub entities: usize,
    pub objects: usize,
    pub locations: usize,
    /// Number of mutation events after the initialization prefix.
    pub events: usize,
    pub query_every: usize,
    pub min_dt: f64,
    pub max_dt: f64,
    pub historical_query_rate: f64,
    pub seed: u64,
}

impl Default for StateTrackingBenchmarkConfig {
    fn default() -> Self {
        Self {
            entities: 32,
            objects: 64,
            locations: 16,
            events: 10_000,
            query_every: 10,
            min_dt: 1e-3,
            max_dt: 1e2,
            historical_query_rate: 0.5,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateTrackingBenchmarkError {
    EmptyDomain(&'static str),
    TooManyIds(&'static str),
    InvalidQueryCadence,
    InvalidTimeRange,
    InvalidHistoricalRate,
    EventIndexOutOfRange { index: usize, len: usize },
    PredictionCountMismatch { expected: usize, actual: usize },
}

impl fmt::Display for StateTrackingBenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDomain(name) => write!(f, "benchmark domain {name} must be non-empty"),
            Self::TooManyIds(name) => write!(f, "benchmark domain {name} exceeds u16 id capacity"),
            Self::InvalidQueryCadence => write!(f, "query_every must be non-zero"),
            Self::InvalidTimeRange => write!(f, "dt range must satisfy 0 < min_dt <= max_dt"),
            Self::InvalidHistoricalRate => write!(f, "historical_query_rate must be finite and in [0, 1]"),
            Self::EventIndexOutOfRange { index, len } => write!(f, "event index {index} out of range for event length {len}"),
            Self::PredictionCountMismatch { expected, actual } => write!(f, "prediction count mismatch: expected {expected}, got {actual}"),
        }
    }
}

impl std::error::Error for StateTrackingBenchmarkError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackingScore {
    pub correct: usize,
    pub total: usize,
    pub current_correct: usize,
    pub current_total: usize,
    pub historical_correct: usize,
    pub historical_total: usize,
    pub entity_location_correct: usize,
    pub entity_location_total: usize,
    pub object_owner_correct: usize,
    pub object_owner_total: usize,
    pub object_location_correct: usize,
    pub object_location_total: usize,
}

impl TrackingScore {
    pub fn accuracy(&self) -> f64 { ratio(self.correct, self.total) }
    pub fn current_accuracy(&self) -> f64 { ratio(self.current_correct, self.current_total) }
    pub fn historical_accuracy(&self) -> f64 { ratio(self.historical_correct, self.historical_total) }
    pub fn compositional_accuracy(&self) -> f64 { ratio(self.object_location_correct, self.object_location_total) }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTrackingBenchmark {
    pub config: StateTrackingBenchmarkConfig,
    pub events: Vec<TrackingEvent>,
    pub queries: Vec<TrackingQuery>,
    #[serde(default)]
    pub initialization_events: usize,
    initial_entity_locations: Vec<LocationId>,
    initial_object_owners: Vec<EntityId>,
}

impl StateTrackingBenchmark {
    pub fn generate(config: StateTrackingBenchmarkConfig) -> Result<Self, StateTrackingBenchmarkError> {
        validate_config(&config)?;
        let mut rng = XorShift64::new(config.seed);
        let initial_entity_locations = (0..config.entities)
            .map(|_| rng.index(config.locations) as LocationId)
            .collect::<Vec<_>>();
        let initial_object_owners = (0..config.objects)
            .map(|_| rng.index(config.entities) as EntityId)
            .collect::<Vec<_>>();

        let mut bootstrap = Vec::with_capacity(config.entities + config.objects);
        for (entity, &to) in initial_entity_locations.iter().enumerate() {
            bootstrap.push(TrackingEventKind::MoveEntity { entity: entity as EntityId, to });
        }
        for (object, &to) in initial_object_owners.iter().enumerate() {
            bootstrap.push(TrackingEventKind::TransferObject { object: object as ObjectId, to });
        }
        rng.shuffle(&mut bootstrap);
        let initialization_events = bootstrap.len();

        let mut events = Vec::with_capacity(initialization_events + config.events);
        let mut time = 0.0;
        for kind in bootstrap {
            time += sample_log_uniform(&mut rng, config.min_dt, config.max_dt);
            events.push(TrackingEvent { time, kind });
        }
        for _ in 0..config.events {
            time += sample_log_uniform(&mut rng, config.min_dt, config.max_dt);
            let kind = if rng.next_bool() {
                TrackingEventKind::MoveEntity {
                    entity: rng.index(config.entities) as EntityId,
                    to: rng.index(config.locations) as LocationId,
                }
            } else {
                TrackingEventKind::TransferObject {
                    object: rng.index(config.objects) as ObjectId,
                    to: rng.index(config.entities) as EntityId,
                }
            };
            events.push(TrackingEvent { time, kind });
        }

        let mut benchmark = Self {
            config,
            events,
            queries: Vec::new(),
            initialization_events,
            initial_entity_locations,
            initial_object_owners,
        };
        benchmark.queries = benchmark.generate_queries(&mut rng)?;
        Ok(benchmark)
    }

    pub fn first_fully_observed_event(&self) -> Option<usize> {
        self.initialization_events.checked_sub(1)
    }

    pub fn oracle_answer(&self, as_of_event: usize, kind: &TrackingQueryKind) -> Result<TrackingAnswer, StateTrackingBenchmarkError> {
        if as_of_event >= self.events.len() {
            return Err(StateTrackingBenchmarkError::EventIndexOutOfRange { index: as_of_event, len: self.events.len() });
        }
        Ok(answer(&self.replay_through(as_of_event), kind))
    }

    pub fn score(&self, predictions: &[TrackingAnswer]) -> Result<TrackingScore, StateTrackingBenchmarkError> {
        if predictions.len() != self.queries.len() {
            return Err(StateTrackingBenchmarkError::PredictionCountMismatch { expected: self.queries.len(), actual: predictions.len() });
        }
        let mut score = TrackingScore {
            correct: 0, total: self.queries.len(), current_correct: 0, current_total: 0,
            historical_correct: 0, historical_total: 0, entity_location_correct: 0,
            entity_location_total: 0, object_owner_correct: 0, object_owner_total: 0,
            object_location_correct: 0, object_location_total: 0,
        };
        for (prediction, query) in predictions.iter().zip(&self.queries) {
            let hit = (*prediction == query.expected) as usize;
            score.correct += hit;
            if query.is_historical() { score.historical_total += 1; score.historical_correct += hit; }
            else { score.current_total += 1; score.current_correct += hit; }
            match query.kind {
                TrackingQueryKind::EntityLocation { .. } => { score.entity_location_total += 1; score.entity_location_correct += hit; }
                TrackingQueryKind::ObjectOwner { .. } => { score.object_owner_total += 1; score.object_owner_correct += hit; }
                TrackingQueryKind::ObjectLocation { .. } => { score.object_location_total += 1; score.object_location_correct += hit; }
            }
        }
        Ok(score)
    }

    fn generate_queries(&self, rng: &mut XorShift64) -> Result<Vec<TrackingQuery>, StateTrackingBenchmarkError> {
        let mut queries = Vec::new();
        let Some(first_observed) = self.first_fully_observed_event() else { return Ok(queries); };
        let first_query = self.initialization_events + self.config.query_every - 1;
        if first_query >= self.events.len() { return Ok(queries); }

        for asked_after in (first_query..self.events.len()).step_by(self.config.query_every) {
            let entity = rng.index(self.config.entities) as EntityId;
            let object = rng.index(self.config.objects) as ObjectId;
            self.push_query(&mut queries, asked_after, asked_after, TrackingQueryKind::EntityLocation { entity })?;
            self.push_query(&mut queries, asked_after, asked_after, TrackingQueryKind::ObjectOwner { object })?;
            self.push_query(&mut queries, asked_after, asked_after, TrackingQueryKind::ObjectLocation { object })?;
            if asked_after > first_observed && rng.next_f64() < self.config.historical_query_rate {
                let as_of = first_observed + rng.index(asked_after - first_observed);
                let object = rng.index(self.config.objects) as ObjectId;
                self.push_query(&mut queries, asked_after, as_of, TrackingQueryKind::ObjectLocation { object })?;
            }
        }
        Ok(queries)
    }

    fn push_query(&self, queries: &mut Vec<TrackingQuery>, asked_after_event: usize, as_of_event: usize, kind: TrackingQueryKind) -> Result<(), StateTrackingBenchmarkError> {
        let expected = self.oracle_answer(as_of_event, &kind)?;
        queries.push(TrackingQuery { asked_after_event, as_of_event, as_of_time: self.events[as_of_event].time, kind, expected });
        Ok(())
    }

    fn replay_through(&self, as_of_event: usize) -> WorldState {
        let mut state = WorldState { entity_locations: self.initial_entity_locations.clone(), object_owners: self.initial_object_owners.clone() };
        for event in self.events.iter().take(as_of_event + 1) { apply_event(&mut state, &event.kind); }
        state
    }
}

#[derive(Debug, Clone)]
struct WorldState { entity_locations: Vec<LocationId>, object_owners: Vec<EntityId> }

fn apply_event(state: &mut WorldState, event: &TrackingEventKind) {
    match *event {
        TrackingEventKind::MoveEntity { entity, to } => state.entity_locations[entity as usize] = to,
        TrackingEventKind::TransferObject { object, to } => state.object_owners[object as usize] = to,
    }
}

fn answer(state: &WorldState, kind: &TrackingQueryKind) -> TrackingAnswer {
    match *kind {
        TrackingQueryKind::EntityLocation { entity } => TrackingAnswer::Location(state.entity_locations[entity as usize]),
        TrackingQueryKind::ObjectOwner { object } => TrackingAnswer::Entity(state.object_owners[object as usize]),
        TrackingQueryKind::ObjectLocation { object } => {
            let owner = state.object_owners[object as usize];
            TrackingAnswer::Location(state.entity_locations[owner as usize])
        }
    }
}

fn validate_config(config: &StateTrackingBenchmarkConfig) -> Result<(), StateTrackingBenchmarkError> {
    for (name, count) in [("entities", config.entities), ("objects", config.objects), ("locations", config.locations), ("events", config.events)] {
        if count == 0 { return Err(StateTrackingBenchmarkError::EmptyDomain(name)); }
    }
    for (name, count) in [("entities", config.entities), ("objects", config.objects), ("locations", config.locations)] {
        if count > u16::MAX as usize + 1 { return Err(StateTrackingBenchmarkError::TooManyIds(name)); }
    }
    if config.query_every == 0 { return Err(StateTrackingBenchmarkError::InvalidQueryCadence); }
    if !config.min_dt.is_finite() || !config.max_dt.is_finite() || config.min_dt <= 0.0 || config.max_dt < config.min_dt {
        return Err(StateTrackingBenchmarkError::InvalidTimeRange);
    }
    if !config.historical_query_rate.is_finite() || !(0.0..=1.0).contains(&config.historical_query_rate) {
        return Err(StateTrackingBenchmarkError::InvalidHistoricalRate);
    }
    Ok(())
}

fn sample_log_uniform(rng: &mut XorShift64, min: f64, max: f64) -> f64 {
    if min == max { return min; }
    (min.ln() + rng.next_f64() * (max.ln() - min.ln())).exp()
}

fn ratio(correct: usize, total: usize) -> f64 { if total == 0 { 0.0 } else { correct as f64 / total as f64 } }

#[derive(Debug, Clone)]
struct XorShift64 { state: u64 }
impl XorShift64 {
    fn new(seed: u64) -> Self {
        let mixed = seed ^ 0x9E3779B97F4A7C15;
        Self { state: if mixed == 0 { 0xD1B54A32D192ED03 } else { mixed } }
    }
    fn next_u64(&mut self) -> u64 { self.state ^= self.state << 13; self.state ^= self.state >> 7; self.state ^= self.state << 17; self.state }
    fn next_bool(&mut self) -> bool { self.next_u64() & 1 == 1 }
    fn next_f64(&mut self) -> f64 { let bits = self.next_u64() >> 11; bits as f64 * (1.0 / ((1_u64 << 53) as f64)) }
    fn index(&mut self, len: usize) -> usize { debug_assert!(len > 0); (self.next_u64() as usize) % len }
    fn shuffle<T>(&mut self, values: &mut [T]) { for i in (1..values.len()).rev() { let j = self.index(i + 1); values.swap(i, j); } }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn generation_is_deterministic() {
        let config = StateTrackingBenchmarkConfig { events: 200, ..Default::default() };
        let a = StateTrackingBenchmark::generate(config.clone()).unwrap();
        let b = StateTrackingBenchmark::generate(config).unwrap();
        assert_eq!(a.events, b.events); assert_eq!(a.queries, b.queries); assert_eq!(a.initialization_events, b.initialization_events);
    }

    #[test]
    fn initialization_prefix_discloses_every_randomized_fact_exactly_once() {
        let benchmark = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig { entities: 7, objects: 11, locations: 5, events: 32, seed: 9, ..Default::default() }).unwrap();
        assert_eq!(benchmark.initialization_events, 18); assert_eq!(benchmark.events.len(), 50);
        let mut entities = HashSet::new(); let mut objects = HashSet::new();
        for event in benchmark.events.iter().take(benchmark.initialization_events) {
            match event.kind {
                TrackingEventKind::MoveEntity { entity, to } => { assert!(entities.insert(entity)); assert_eq!(to, benchmark.initial_entity_locations[entity as usize]); }
                TrackingEventKind::TransferObject { object, to } => { assert!(objects.insert(object)); assert_eq!(to, benchmark.initial_object_owners[object as usize]); }
            }
        }
        assert_eq!(entities.len(), 7); assert_eq!(objects.len(), 11);
    }

    #[test]
    fn no_query_uses_partially_initialized_world() {
        let benchmark = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig { entities: 8, objects: 12, events: 100, query_every: 1, historical_query_rate: 1.0, ..Default::default() }).unwrap();
        let first = benchmark.first_fully_observed_event().unwrap();
        assert!(benchmark.queries.iter().all(|q| q.asked_after_event >= benchmark.initialization_events));
        assert!(benchmark.queries.iter().all(|q| q.as_of_event >= first));
    }

    #[test]
    fn disjoint_seeds_randomize_initial_world() {
        let a = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig { seed: 1, events: 32, ..Default::default() }).unwrap();
        let b = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig { seed: 2, events: 32, ..Default::default() }).unwrap();
        assert_ne!(a.initial_entity_locations, b.initial_entity_locations); assert_ne!(a.initial_object_owners, b.initial_object_owners);
    }

    #[test]
    fn pathological_zero_xorshift_seed_remains_live() {
        let b = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig { seed: 0x9E3779B97F4A7C15, events: 32, ..Default::default() }).unwrap();
        assert!(b.events.windows(2).all(|pair| pair[1].time > pair[0].time));
    }

    #[test]
    fn timestamps_are_strictly_increasing_and_irregular() {
        let b = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig { events: 1000, min_dt: 1e-3, max_dt: 1e2, ..Default::default() }).unwrap();
        let dts = b.events.windows(2).map(|p| p[1].time - p[0].time).collect::<Vec<_>>();
        let min = dts.iter().copied().fold(f64::INFINITY, f64::min); let max = dts.iter().copied().fold(0.0_f64, f64::max);
        assert!(dts.iter().all(|dt| *dt > 0.0)); assert!(max / min > 1_000.0);
    }

    #[test]
    fn oracle_matches_stored_answers() {
        let b = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig { events: 250, query_every: 5, ..Default::default() }).unwrap();
        for q in &b.queries { assert_eq!(b.oracle_answer(q.as_of_event, &q.kind).unwrap(), q.expected); }
    }

    #[test]
    fn requested_historical_queries_are_strict_and_fully_observed() {
        let b = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig { events: 500, query_every: 5, historical_query_rate: 1.0, ..Default::default() }).unwrap();
        let first = b.first_fully_observed_event().unwrap();
        let hist = b.queries.iter().filter(|q| q.is_historical()).collect::<Vec<_>>();
        assert!(!hist.is_empty()); assert!(hist.iter().all(|q| q.as_of_event >= first && q.as_of_event < q.asked_after_event));
    }

    #[test]
    fn perfect_oracle_predictions_score_one() {
        let b = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig { events: 100, historical_query_rate: 1.0, ..Default::default() }).unwrap();
        let p = b.queries.iter().map(|q| q.expected).collect::<Vec<_>>(); let s = b.score(&p).unwrap();
        assert_eq!(s.accuracy(), 1.0); assert_eq!(s.current_accuracy(), 1.0); assert_eq!(s.historical_accuracy(), 1.0); assert_eq!(s.compositional_accuracy(), 1.0);
    }
}
