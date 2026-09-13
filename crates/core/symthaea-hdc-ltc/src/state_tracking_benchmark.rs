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
//! Ownership and location mutate independently. `ObjectLocation` therefore
//! requires composition across two mutable relations. Historical queries ask
//! about an earlier world state after later events have already been observed.

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
    /// Strictly increasing event time in arbitrary continuous-time units.
    pub time: f64,
    pub kind: TrackingEventKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackingQueryKind {
    EntityLocation { entity: EntityId },
    ObjectOwner { object: ObjectId },
    /// Two-hop mutable query: object -> owner -> owner's location.
    ObjectLocation { object: ObjectId },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackingAnswer {
    Entity(EntityId),
    Location(LocationId),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackingQuery {
    /// Query is presented after this event has been observed.
    pub asked_after_event: usize,
    /// State index to answer about. Historical iff this is strictly smaller
    /// than `asked_after_event`.
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
    pub events: usize,
    /// Emit a current-state query bundle after every N events.
    pub query_every: usize,
    /// Minimum positive inter-event interval.
    pub min_dt: f64,
    /// Maximum positive inter-event interval.
    pub max_dt: f64,
    /// Probability that a query bundle also receives one strictly historical
    /// compositional query.
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
            Self::InvalidHistoricalRate => {
                write!(f, "historical_query_rate must be finite and in [0, 1]")
            }
            Self::EventIndexOutOfRange { index, len } => {
                write!(f, "event index {index} out of range for event length {len}")
            }
            Self::PredictionCountMismatch { expected, actual } => write!(
                f,
                "prediction count mismatch: expected {expected}, got {actual}"
            ),
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
    pub fn accuracy(&self) -> f64 {
        ratio(self.correct, self.total)
    }

    pub fn current_accuracy(&self) -> f64 {
        ratio(self.current_correct, self.current_total)
    }

    pub fn historical_accuracy(&self) -> f64 {
        ratio(self.historical_correct, self.historical_total)
    }

    pub fn compositional_accuracy(&self) -> f64 {
        ratio(self.object_location_correct, self.object_location_total)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTrackingBenchmark {
    pub config: StateTrackingBenchmarkConfig,
    pub events: Vec<TrackingEvent>,
    pub queries: Vec<TrackingQuery>,
    initial_entity_locations: Vec<LocationId>,
    initial_object_owners: Vec<EntityId>,
}

impl StateTrackingBenchmark {
    pub fn generate(
        config: StateTrackingBenchmarkConfig,
    ) -> Result<Self, StateTrackingBenchmarkError> {
        validate_config(&config)?;
        let mut rng = XorShift64::new(config.seed);

        let initial_entity_locations = (0..config.entities)
            .map(|entity| (entity % config.locations) as LocationId)
            .collect::<Vec<_>>();
        let initial_object_owners = (0..config.objects)
            .map(|object| (object % config.entities) as EntityId)
            .collect::<Vec<_>>();

        let mut events = Vec::with_capacity(config.events);
        let mut time = 0.0_f64;
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
            initial_entity_locations,
            initial_object_owners,
        };
        benchmark.queries = benchmark.generate_queries(&mut rng)?;
        Ok(benchmark)
    }

    /// Exact symbolic answer at the state immediately after `as_of_event`.
    pub fn oracle_answer(
        &self,
        as_of_event: usize,
        kind: &TrackingQueryKind,
    ) -> Result<TrackingAnswer, StateTrackingBenchmarkError> {
        if as_of_event >= self.events.len() {
            return Err(StateTrackingBenchmarkError::EventIndexOutOfRange {
                index: as_of_event,
                len: self.events.len(),
            });
        }
        Ok(answer(&self.replay_through(as_of_event), kind))
    }

    /// Score predictions in query order.
    pub fn score(
        &self,
        predictions: &[TrackingAnswer],
    ) -> Result<TrackingScore, StateTrackingBenchmarkError> {
        if predictions.len() != self.queries.len() {
            return Err(StateTrackingBenchmarkError::PredictionCountMismatch {
                expected: self.queries.len(),
                actual: predictions.len(),
            });
        }

        let mut score = TrackingScore {
            correct: 0,
            total: self.queries.len(),
            current_correct: 0,
            current_total: 0,
            historical_correct: 0,
            historical_total: 0,
            entity_location_correct: 0,
            entity_location_total: 0,
            object_owner_correct: 0,
            object_owner_total: 0,
            object_location_correct: 0,
            object_location_total: 0,
        };

        for (prediction, query) in predictions.iter().zip(self.queries.iter()) {
            let correct = *prediction == query.expected;
            let hit = correct as usize;
            score.correct += hit;

            if query.is_historical() {
                score.historical_total += 1;
                score.historical_correct += hit;
            } else {
                score.current_total += 1;
                score.current_correct += hit;
            }

            match query.kind {
                TrackingQueryKind::EntityLocation { .. } => {
                    score.entity_location_total += 1;
                    score.entity_location_correct += hit;
                }
                TrackingQueryKind::ObjectOwner { .. } => {
                    score.object_owner_total += 1;
                    score.object_owner_correct += hit;
                }
                TrackingQueryKind::ObjectLocation { .. } => {
                    score.object_location_total += 1;
                    score.object_location_correct += hit;
                }
            }
        }
        Ok(score)
    }

    fn generate_queries(
        &self,
        rng: &mut XorShift64,
    ) -> Result<Vec<TrackingQuery>, StateTrackingBenchmarkError> {
        let mut queries = Vec::new();

        for asked_after in (self.config.query_every - 1..self.events.len())
            .step_by(self.config.query_every)
        {
            let current_entity = rng.index(self.config.entities) as EntityId;
            let current_object = rng.index(self.config.objects) as ObjectId;

            self.push_query(
                &mut queries,
                asked_after,
                asked_after,
                TrackingQueryKind::EntityLocation {
                    entity: current_entity,
                },
            )?;
            self.push_query(
                &mut queries,
                asked_after,
                asked_after,
                TrackingQueryKind::ObjectOwner {
                    object: current_object,
                },
            )?;
            self.push_query(
                &mut queries,
                asked_after,
                asked_after,
                TrackingQueryKind::ObjectLocation {
                    object: current_object,
                },
            )?;

            if asked_after > 0 && rng.next_f64() < self.config.historical_query_rate {
                // Strictly retrospective: index is always in [0, asked_after).
                let as_of = rng.index(asked_after);
                let historical_object = rng.index(self.config.objects) as ObjectId;
                self.push_query(
                    &mut queries,
                    asked_after,
                    as_of,
                    TrackingQueryKind::ObjectLocation {
                        object: historical_object,
                    },
                )?;
            }
        }
        Ok(queries)
    }

    fn push_query(
        &self,
        queries: &mut Vec<TrackingQuery>,
        asked_after_event: usize,
        as_of_event: usize,
        kind: TrackingQueryKind,
    ) -> Result<(), StateTrackingBenchmarkError> {
        let expected = self.oracle_answer(as_of_event, &kind)?;
        queries.push(TrackingQuery {
            asked_after_event,
            as_of_event,
            as_of_time: self.events[as_of_event].time,
            kind,
            expected,
        });
        Ok(())
    }

    fn replay_through(&self, as_of_event: usize) -> WorldState {
        let mut state = WorldState {
            entity_locations: self.initial_entity_locations.clone(),
            object_owners: self.initial_object_owners.clone(),
        };
        for event in self.events.iter().take(as_of_event + 1) {
            apply_event(&mut state, &event.kind);
        }
        state
    }
}

#[derive(Debug, Clone)]
struct WorldState {
    entity_locations: Vec<LocationId>,
    object_owners: Vec<EntityId>,
}

fn apply_event(state: &mut WorldState, event: &TrackingEventKind) {
    match *event {
        TrackingEventKind::MoveEntity { entity, to } => {
            state.entity_locations[entity as usize] = to;
        }
        TrackingEventKind::TransferObject { object, to } => {
            state.object_owners[object as usize] = to;
        }
    }
}

fn answer(state: &WorldState, kind: &TrackingQueryKind) -> TrackingAnswer {
    match *kind {
        TrackingQueryKind::EntityLocation { entity } => {
            TrackingAnswer::Location(state.entity_locations[entity as usize])
        }
        TrackingQueryKind::ObjectOwner { object } => {
            TrackingAnswer::Entity(state.object_owners[object as usize])
        }
        TrackingQueryKind::ObjectLocation { object } => {
            let owner = state.object_owners[object as usize];
            TrackingAnswer::Location(state.entity_locations[owner as usize])
        }
    }
}

fn validate_config(
    config: &StateTrackingBenchmarkConfig,
) -> Result<(), StateTrackingBenchmarkError> {
    for (name, count) in [
        ("entities", config.entities),
        ("objects", config.objects),
        ("locations", config.locations),
        ("events", config.events),
    ] {
        if count == 0 {
            return Err(StateTrackingBenchmarkError::EmptyDomain(name));
        }
    }
    for (name, count) in [
        ("entities", config.entities),
        ("objects", config.objects),
        ("locations", config.locations),
    ] {
        // A u16 id can represent 65,536 distinct values: 0..=65,535.
        if count > u16::MAX as usize + 1 {
            return Err(StateTrackingBenchmarkError::TooManyIds(name));
        }
    }
    if config.query_every == 0 {
        return Err(StateTrackingBenchmarkError::InvalidQueryCadence);
    }
    if !config.min_dt.is_finite()
        || !config.max_dt.is_finite()
        || config.min_dt <= 0.0
        || config.max_dt < config.min_dt
    {
        return Err(StateTrackingBenchmarkError::InvalidTimeRange);
    }
    if !config.historical_query_rate.is_finite()
        || !(0.0..=1.0).contains(&config.historical_query_rate)
    {
        return Err(StateTrackingBenchmarkError::InvalidHistoricalRate);
    }
    Ok(())
}

fn sample_log_uniform(rng: &mut XorShift64, min: f64, max: f64) -> f64 {
    if min == max {
        return min;
    }
    let log_min = min.ln();
    let log_max = max.ln();
    (log_min + rng.next_f64() * (log_max - log_min)).exp()
}

fn ratio(correct: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        correct as f64 / total as f64
    }
}

#[derive(Debug, Clone)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let mixed = seed ^ 0x9E3779B97F4A7C15;
        Self {
            // Xorshift has an absorbing all-zero state. Remap that one seed to a
            // fixed non-zero state so every u64 seed remains usable.
            state: if mixed == 0 {
                0xD1B54A32D192ED03
            } else {
                mixed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        bits as f64 * (1.0 / ((1_u64 << 53) as f64))
    }

    fn index(&mut self, len: usize) -> usize {
        debug_assert!(len > 0);
        (self.next_u64() as usize) % len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_deterministic() {
        let config = StateTrackingBenchmarkConfig {
            events: 200,
            ..StateTrackingBenchmarkConfig::default()
        };
        let a = StateTrackingBenchmark::generate(config.clone()).unwrap();
        let b = StateTrackingBenchmark::generate(config).unwrap();
        assert_eq!(a.events, b.events);
        assert_eq!(a.queries, b.queries);
    }

    #[test]
    fn pathological_zero_xorshift_seed_remains_live() {
        let seed = 0x9E3779B97F4A7C15;
        let benchmark = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig {
            seed,
            events: 32,
            ..StateTrackingBenchmarkConfig::default()
        })
        .unwrap();
        assert!(benchmark.events.windows(2).all(|pair| pair[1].time > pair[0].time));
    }

    #[test]
    fn timestamps_are_strictly_increasing_and_irregular() {
        let benchmark = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig {
            events: 1000,
            min_dt: 1e-3,
            max_dt: 1e2,
            ..StateTrackingBenchmarkConfig::default()
        })
        .unwrap();
        let dts = benchmark
            .events
            .windows(2)
            .map(|pair| pair[1].time - pair[0].time)
            .collect::<Vec<_>>();
        assert!(dts.iter().all(|dt| *dt > 0.0));
        let min = dts.iter().copied().fold(f64::INFINITY, f64::min);
        let max = dts.iter().copied().fold(0.0_f64, f64::max);
        assert!(max / min > 1_000.0, "insufficient irregularity: min={min} max={max}");
    }

    #[test]
    fn oracle_matches_stored_answers() {
        let benchmark = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig {
            events: 250,
            query_every: 5,
            ..StateTrackingBenchmarkConfig::default()
        })
        .unwrap();
        for query in &benchmark.queries {
            assert_eq!(
                benchmark
                    .oracle_answer(query.as_of_event, &query.kind)
                    .unwrap(),
                query.expected
            );
        }
    }

    #[test]
    fn requested_historical_queries_are_strictly_retrospective() {
        let benchmark = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig {
            events: 500,
            query_every: 5,
            historical_query_rate: 1.0,
            ..StateTrackingBenchmarkConfig::default()
        })
        .unwrap();
        let historical = benchmark
            .queries
            .iter()
            .filter(|query| query.is_historical())
            .collect::<Vec<_>>();
        assert!(!historical.is_empty());
        assert!(historical
            .iter()
            .all(|query| query.as_of_event < query.asked_after_event));
    }

    #[test]
    fn benchmark_contains_compositional_queries() {
        let benchmark = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig {
            events: 100,
            query_every: 5,
            ..StateTrackingBenchmarkConfig::default()
        })
        .unwrap();
        assert!(benchmark
            .queries
            .iter()
            .any(|query| matches!(query.kind, TrackingQueryKind::ObjectLocation { .. })));
    }

    #[test]
    fn perfect_oracle_predictions_score_one() {
        let benchmark = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig {
            events: 100,
            historical_query_rate: 1.0,
            ..StateTrackingBenchmarkConfig::default()
        })
        .unwrap();
        let predictions = benchmark
            .queries
            .iter()
            .map(|query| query.expected)
            .collect::<Vec<_>>();
        let score = benchmark.score(&predictions).unwrap();
        assert_eq!(score.accuracy(), 1.0);
        assert_eq!(score.current_accuracy(), 1.0);
        assert_eq!(score.historical_accuracy(), 1.0);
        assert_eq!(score.compositional_accuracy(), 1.0);
    }
}
