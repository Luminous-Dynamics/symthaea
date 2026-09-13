// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Offline historical archive for the state-tracking benchmark.
//!
//! This module is deliberately separate from HLS recurrence. It asks whether the
//! checkpoint-validity algebra can represent the benchmark's exact as-of semantics
//! before historical queries are allowed back into a learned HLS experiment.

use crate::continuous_hv::UnitaryRole;
use crate::state_tracking_benchmark::{
    EntityId, LocationId, ObjectId, StateTrackingBenchmark, TrackingAnswer, TrackingEventKind,
    TrackingQuery, TrackingQueryKind,
};
use crate::state_tracking_codec::{StateTrackingCodec, TrackingCodecError};
use crate::temporal_phasor::{TemporalAlgebraError, TemporalAxis};
use crate::validity_interval_memory::{
    ValidityCleanupResult, ValidityIntervalMemory, ValidityMemoryError,
};
use std::fmt;

#[derive(Debug)]
pub enum StateTrackingValidityArchiveError {
    EmptyEventStream,
    IncompleteInitialization(&'static str, usize),
    Codec(TrackingCodecError),
    Temporal(TemporalAlgebraError),
    Memory(ValidityMemoryError),
}

impl fmt::Display for StateTrackingValidityArchiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyEventStream => write!(f, "state-tracking validity archive requires events"),
            Self::IncompleteInitialization(kind, index) => write!(
                f,
                "state-tracking validity archive never observed initial {kind} assignment for index {index}"
            ),
            Self::Codec(error) => write!(f, "state-tracking validity archive codec error: {error}"),
            Self::Temporal(error) => write!(f, "state-tracking validity archive temporal error: {error}"),
            Self::Memory(error) => write!(f, "state-tracking validity archive memory error: {error}"),
        }
    }
}

impl std::error::Error for StateTrackingValidityArchiveError {}

impl From<TrackingCodecError> for StateTrackingValidityArchiveError {
    fn from(value: TrackingCodecError) -> Self {
        Self::Codec(value)
    }
}
impl From<TemporalAlgebraError> for StateTrackingValidityArchiveError {
    fn from(value: TemporalAlgebraError) -> Self {
        Self::Temporal(value)
    }
}
impl From<ValidityMemoryError> for StateTrackingValidityArchiveError {
    fn from(value: ValidityMemoryError) -> Self {
        Self::Memory(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HistoricalArchiveAnswer {
    pub answer: TrackingAnswer,
    /// Minimum cleanup margin across all hops used by this answer.
    pub min_margin: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HistoricalArchiveEvaluation {
    pub correct: usize,
    pub total: usize,
    pub object_location_correct: usize,
    pub object_location_total: usize,
    pub mean_min_margin: f64,
    pub smallest_margin: f64,
}

impl HistoricalArchiveEvaluation {
    pub fn accuracy(&self) -> f64 {
        ratio(self.correct, self.total)
    }

    pub fn compositional_accuracy(&self) -> f64 {
        ratio(self.object_location_correct, self.object_location_total)
    }
}

#[derive(Debug, Clone)]
pub struct StateTrackingValidityArchive {
    axis: TemporalAxis,
    memory: ValidityIntervalMemory,
}

impl StateTrackingValidityArchive {
    pub fn build(
        benchmark: &StateTrackingBenchmark,
        codec: &StateTrackingCodec,
        temporal_seed: u64,
    ) -> Result<Self, StateTrackingValidityArchiveError> {
        if benchmark.events.is_empty() {
            return Err(StateTrackingValidityArchiveError::EmptyEventStream);
        }
        let axis = TemporalAxis::new(codec.dim(), temporal_seed)?;
        let mut memory = ValidityIntervalMemory::new(codec.dim())?;
        let mut entity_open = vec![None::<(u64, LocationId)>; benchmark.config.entities];
        let mut object_open = vec![None::<(u64, EntityId)>; benchmark.config.objects];

        for (event_index, event) in benchmark.events.iter().enumerate() {
            let checkpoint = event_index as u64;
            match event.kind {
                TrackingEventKind::MoveEntity { entity, to } => {
                    let slot = &mut entity_open[entity as usize];
                    if let Some((start, previous)) = *slot {
                        write_entity_location_span(
                            &mut memory,
                            &axis,
                            codec,
                            entity,
                            previous,
                            start,
                            checkpoint,
                        )?;
                    }
                    *slot = Some((checkpoint, to));
                }
                TrackingEventKind::TransferObject { object, to } => {
                    let slot = &mut object_open[object as usize];
                    if let Some((start, previous)) = *slot {
                        write_object_owner_span(
                            &mut memory,
                            &axis,
                            codec,
                            object,
                            previous,
                            start,
                            checkpoint,
                        )?;
                    }
                    *slot = Some((checkpoint, to));
                }
            }
        }

        let end = benchmark.events.len() as u64;
        for (entity, open) in entity_open.into_iter().enumerate() {
            let Some((start, value)) = open else {
                return Err(StateTrackingValidityArchiveError::IncompleteInitialization(
                    "entity-location",
                    entity,
                ));
            };
            write_entity_location_span(
                &mut memory,
                &axis,
                codec,
                entity as EntityId,
                value,
                start,
                end,
            )?;
        }
        for (object, open) in object_open.into_iter().enumerate() {
            let Some((start, value)) = open else {
                return Err(StateTrackingValidityArchiveError::IncompleteInitialization(
                    "object-owner",
                    object,
                ));
            };
            write_object_owner_span(
                &mut memory,
                &axis,
                codec,
                object as ObjectId,
                value,
                start,
                end,
            )?;
        }

        Ok(Self { axis, memory })
    }

    pub fn spans_written(&self) -> usize {
        self.memory.spans_written()
    }

    pub fn answer(
        &self,
        benchmark: &StateTrackingBenchmark,
        codec: &StateTrackingCodec,
        query: &TrackingQuery,
    ) -> Result<HistoricalArchiveAnswer, StateTrackingValidityArchiveError> {
        let checkpoint = query.as_of_event as u64;
        Ok(match query.kind {
            TrackingQueryKind::EntityLocation { entity } => {
                let locations = location_roles(codec, benchmark.config.locations)?;
                let cleanup = self.memory.cleanup(
                    &self.axis,
                    &codec.entity_location_key(entity)?,
                    &locations,
                    checkpoint,
                )?;
                HistoricalArchiveAnswer {
                    answer: TrackingAnswer::Location(cleanup.best_index as LocationId),
                    min_margin: cleanup.margin,
                }
            }
            TrackingQueryKind::ObjectOwner { object } => {
                let entities = entity_roles(codec, benchmark.config.entities)?;
                let cleanup = self.memory.cleanup(
                    &self.axis,
                    &codec.object_owner_key(object)?,
                    &entities,
                    checkpoint,
                )?;
                HistoricalArchiveAnswer {
                    answer: TrackingAnswer::Entity(cleanup.best_index as EntityId),
                    min_margin: cleanup.margin,
                }
            }
            TrackingQueryKind::ObjectLocation { object } => {
                let entities = entity_roles(codec, benchmark.config.entities)?;
                let owner = self.memory.cleanup(
                    &self.axis,
                    &codec.object_owner_key(object)?,
                    &entities,
                    checkpoint,
                )?;
                let owner_id = owner.best_index as EntityId;
                let locations = location_roles(codec, benchmark.config.locations)?;
                let location = self.memory.cleanup(
                    &self.axis,
                    &codec.entity_location_key(owner_id)?,
                    &locations,
                    checkpoint,
                )?;
                HistoricalArchiveAnswer {
                    answer: TrackingAnswer::Location(location.best_index as LocationId),
                    min_margin: owner.margin.min(location.margin),
                }
            }
        })
    }

    /// Evaluate only historical benchmark queries. Current-state performance stays
    /// owned by the independently preregistered #2662 experiment.
    pub fn evaluate_historical(
        &self,
        benchmark: &StateTrackingBenchmark,
        codec: &StateTrackingCodec,
    ) -> Result<HistoricalArchiveEvaluation, StateTrackingValidityArchiveError> {
        let mut correct = 0usize;
        let mut total = 0usize;
        let mut object_location_correct = 0usize;
        let mut object_location_total = 0usize;
        let mut margin_sum = 0.0_f64;
        let mut smallest_margin = f64::INFINITY;

        for query in benchmark.queries.iter().filter(|query| query.is_historical()) {
            let result = self.answer(benchmark, codec, query)?;
            let hit = result.answer == query.expected;
            correct += hit as usize;
            total += 1;
            margin_sum += result.min_margin;
            smallest_margin = smallest_margin.min(result.min_margin);
            if matches!(query.kind, TrackingQueryKind::ObjectLocation { .. }) {
                object_location_total += 1;
                object_location_correct += hit as usize;
            }
        }

        Ok(HistoricalArchiveEvaluation {
            correct,
            total,
            object_location_correct,
            object_location_total,
            mean_min_margin: if total == 0 { 0.0 } else { margin_sum / total as f64 },
            smallest_margin: if total == 0 { 0.0 } else { smallest_margin },
        })
    }
}

fn write_entity_location_span(
    memory: &mut ValidityIntervalMemory,
    axis: &TemporalAxis,
    codec: &StateTrackingCodec,
    entity: EntityId,
    location: LocationId,
    start: u64,
    end: u64,
) -> Result<(), StateTrackingValidityArchiveError> {
    let value = answer_role(codec, TrackingAnswer::Location(location))?;
    memory.write_span(axis, &codec.entity_location_key(entity)?, &value, start, end)?;
    Ok(())
}

fn write_object_owner_span(
    memory: &mut ValidityIntervalMemory,
    axis: &TemporalAxis,
    codec: &StateTrackingCodec,
    object: ObjectId,
    entity: EntityId,
    start: u64,
    end: u64,
) -> Result<(), StateTrackingValidityArchiveError> {
    let value = answer_role(codec, TrackingAnswer::Entity(entity))?;
    memory.write_span(axis, &codec.object_owner_key(object)?, &value, start, end)?;
    Ok(())
}

fn entity_roles(
    codec: &StateTrackingCodec,
    count: usize,
) -> Result<Vec<UnitaryRole>, StateTrackingValidityArchiveError> {
    (0..count)
        .map(|id| answer_role(codec, TrackingAnswer::Entity(id as EntityId)))
        .collect()
}

fn location_roles(
    codec: &StateTrackingCodec,
    count: usize,
) -> Result<Vec<UnitaryRole>, StateTrackingValidityArchiveError> {
    (0..count)
        .map(|id| answer_role(codec, TrackingAnswer::Location(id as LocationId)))
        .collect()
}

fn answer_role(
    codec: &StateTrackingCodec,
    answer: TrackingAnswer,
) -> Result<UnitaryRole, StateTrackingValidityArchiveError> {
    let symbol = codec.answer_symbol(answer)?;
    // StateTrackingCodec deliberately makes answer identities bipolar/self-keying.
    UnitaryRole::try_from_values(symbol.values.clone())
        .map_err(|_| StateTrackingValidityArchiveError::Codec(match answer {
            TrackingAnswer::Entity(id) => TrackingCodecError::EntityOutOfRange(id),
            TrackingAnswer::Location(id) => TrackingCodecError::LocationOutOfRange(id),
        }))
}

fn ratio(correct: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        correct as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_tracking_benchmark::StateTrackingBenchmarkConfig;

    #[test]
    fn archive_has_one_closed_span_per_observed_event() {
        let benchmark = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig {
            entities: 6,
            objects: 8,
            locations: 4,
            events: 40,
            query_every: 4,
            historical_query_rate: 1.0,
            seed: 700,
            ..Default::default()
        })
        .unwrap();
        let codec = StateTrackingCodec::from_benchmark_config(4096, &benchmark.config, 701).unwrap();
        let archive = StateTrackingValidityArchive::build(&benchmark, &codec, 702).unwrap();
        assert_eq!(archive.spans_written(), benchmark.events.len());
    }

    #[test]
    fn small_historical_world_recovers_oracle_including_two_hop_queries() {
        let benchmark = StateTrackingBenchmark::generate(StateTrackingBenchmarkConfig {
            entities: 4,
            objects: 6,
            locations: 3,
            events: 48,
            query_every: 3,
            historical_query_rate: 1.0,
            seed: 710,
            ..Default::default()
        })
        .unwrap();
        let codec = StateTrackingCodec::from_benchmark_config(16_384, &benchmark.config, 711).unwrap();
        let archive = StateTrackingValidityArchive::build(&benchmark, &codec, 712).unwrap();
        let evaluation = archive.evaluate_historical(&benchmark, &codec).unwrap();
        assert!(evaluation.total > 0);
        assert!(evaluation.object_location_total > 0);
        assert_eq!(evaluation.correct, evaluation.total);
        assert_eq!(evaluation.object_location_correct, evaluation.object_location_total);
        assert!(evaluation.smallest_margin > 0.0, "evaluation={evaluation:?}");
    }
}
