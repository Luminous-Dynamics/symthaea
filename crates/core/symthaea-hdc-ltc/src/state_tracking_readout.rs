// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Minimal shared diagnostic readout for state-tracking reservoir experiments.
//!
//! This dependency-free nearest-centroid classifier is intentionally simple so
//! the same head can be held fixed across recurrent ablations. Stronger ridge or
//! gradient-trained heads should be evaluated separately.

use crate::continuous_hv::ContinuousHV;
use crate::state_tracking_benchmark::{
    EntityId, LocationId, StateTrackingBenchmarkConfig, TrackingAnswer, TrackingQueryKind,
};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrackingReadoutError {
    ZeroDimension,
    DimensionMismatch { expected: usize, actual: usize },
    EntityOutOfRange(EntityId),
    LocationOutOfRange(LocationId),
    UntrainedDomain(&'static str),
}

impl fmt::Display for TrackingReadoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "readout dimension must be non-zero"),
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "readout dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::EntityOutOfRange(id) => write!(f, "entity answer {id} is outside readout range"),
            Self::LocationOutOfRange(id) => {
                write!(f, "location answer {id} is outside readout range")
            }
            Self::UntrainedDomain(name) => write!(f, "readout domain {name} has no observations"),
        }
    }
}

impl std::error::Error for TrackingReadoutError {}

#[derive(Debug, Clone)]
struct PrototypeBank {
    sums: Vec<ContinuousHV>,
    counts: Vec<u64>,
}

impl PrototypeBank {
    fn new(classes: usize, dim: usize) -> Self {
        Self {
            sums: (0..classes).map(|_| ContinuousHV::new(dim)).collect(),
            counts: vec![0; classes],
        }
    }

    fn observe(&mut self, class: usize, feature: &ContinuousHV) {
        self.sums[class].add_scaled(feature, 1.0);
        self.counts[class] += 1;
    }

    fn predict(&self, feature: &ContinuousHV) -> Option<usize> {
        self.sums
            .iter()
            .zip(self.counts.iter())
            .enumerate()
            .filter(|(_, (_, count))| **count > 0)
            .map(|(index, (sum, _))| (index, feature.similarity(sum)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(index, _)| index)
    }

    fn observed_classes(&self) -> usize {
        self.counts.iter().filter(|count| **count > 0).count()
    }
}

#[derive(Debug, Clone)]
pub struct TrackingPrototypeReadout {
    dim: usize,
    entities: PrototypeBank,
    locations: PrototypeBank,
}

impl TrackingPrototypeReadout {
    pub fn from_benchmark_config(
        dim: usize,
        config: &StateTrackingBenchmarkConfig,
    ) -> Result<Self, TrackingReadoutError> {
        if dim == 0 {
            return Err(TrackingReadoutError::ZeroDimension);
        }
        Ok(Self {
            dim,
            entities: PrototypeBank::new(config.entities, dim),
            locations: PrototypeBank::new(config.locations, dim),
        })
    }

    pub fn observe(
        &mut self,
        feature: &ContinuousHV,
        answer: TrackingAnswer,
    ) -> Result<(), TrackingReadoutError> {
        self.check_dim(feature.dim())?;
        let normalized = feature.normalize();
        match answer {
            TrackingAnswer::Entity(id) => {
                if id as usize >= self.entities.sums.len() {
                    return Err(TrackingReadoutError::EntityOutOfRange(id));
                }
                self.entities.observe(id as usize, &normalized);
            }
            TrackingAnswer::Location(id) => {
                if id as usize >= self.locations.sums.len() {
                    return Err(TrackingReadoutError::LocationOutOfRange(id));
                }
                self.locations.observe(id as usize, &normalized);
            }
        }
        Ok(())
    }

    pub fn predict(
        &self,
        feature: &ContinuousHV,
        query_kind: TrackingQueryKind,
    ) -> Result<TrackingAnswer, TrackingReadoutError> {
        self.check_dim(feature.dim())?;
        match query_kind {
            TrackingQueryKind::ObjectOwner { .. } => self
                .entities
                .predict(feature)
                .map(|index| TrackingAnswer::Entity(index as EntityId))
                .ok_or(TrackingReadoutError::UntrainedDomain("entity")),
            TrackingQueryKind::EntityLocation { .. }
            | TrackingQueryKind::ObjectLocation { .. } => self
                .locations
                .predict(feature)
                .map(|index| TrackingAnswer::Location(index as LocationId))
                .ok_or(TrackingReadoutError::UntrainedDomain("location")),
        }
    }

    pub fn observed_entity_classes(&self) -> usize {
        self.entities.observed_classes()
    }

    pub fn observed_location_classes(&self) -> usize {
        self.locations.observed_classes()
    }

    fn check_dim(&self, actual: usize) -> Result<(), TrackingReadoutError> {
        if actual == self.dim {
            Ok(())
        } else {
            Err(TrackingReadoutError::DimensionMismatch {
                expected: self.dim,
                actual,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_feature_retrieves_observed_class() {
        let config = StateTrackingBenchmarkConfig {
            entities: 4,
            locations: 3,
            ..StateTrackingBenchmarkConfig::default()
        };
        let mut readout = TrackingPrototypeReadout::from_benchmark_config(128, &config).unwrap();
        let entity_feature = ContinuousHV::new_random(128, 1);
        let location_feature = ContinuousHV::new_random(128, 2);
        readout
            .observe(&entity_feature, TrackingAnswer::Entity(2))
            .unwrap();
        readout
            .observe(&location_feature, TrackingAnswer::Location(1))
            .unwrap();

        assert_eq!(
            readout
                .predict(&entity_feature, TrackingQueryKind::ObjectOwner { object: 0 })
                .unwrap(),
            TrackingAnswer::Entity(2)
        );
        assert_eq!(
            readout
                .predict(
                    &location_feature,
                    TrackingQueryKind::EntityLocation { entity: 0 }
                )
                .unwrap(),
            TrackingAnswer::Location(1)
        );
    }
}
