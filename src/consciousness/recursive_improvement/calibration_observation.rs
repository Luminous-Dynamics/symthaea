// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Keyed calibration observations and exact-population cohorts.
//!
//! CAL-004C closes the evidence-identity loop between `CalibrationKeyV1` and
//! resolved binary probability observations. A cohort may contain observations
//! for exactly one key; cross-key pooling, duplicate observation identities, and
//! silent capacity overflow are rejected.

use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;
use std::num::{NonZeroU64, NonZeroUsize};

use super::calibration_key::{CalibrationIdentity, CalibrationKeyV1};

/// Validation error for a probability value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbabilityError {
    pub value: f64,
}

impl fmt::Display for ProbabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "probability must be finite and within [0, 1], got {}", self.value)
    }
}

impl std::error::Error for ProbabilityError {}

/// Valid finite unit-interval probability.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct UnitProbability(f64);

impl UnitProbability {
    pub fn new(value: f64) -> Result<Self, ProbabilityError> {
        if value.is_finite() && (0.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(ProbabilityError { value })
        }
    }

    pub const fn get(self) -> f64 {
        self.0
    }
}

impl<'de> Deserialize<'de> for UnitProbability {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = f64::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// Construction error for one resolved calibration observation.
#[derive(Debug, Clone, PartialEq)]
pub enum CalibrationObservationError {
    InvalidProbability(ProbabilityError),
    ZeroResolvedAt,
}

impl fmt::Display for CalibrationObservationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProbability(err) => fmt::Display::fmt(err, f),
            Self::ZeroResolvedAt => write!(f, "resolved_at_unix_ms must be non-zero"),
        }
    }
}

impl std::error::Error for CalibrationObservationError {}

impl From<ProbabilityError> for CalibrationObservationError {
    fn from(value: ProbabilityError) -> Self {
        Self::InvalidProbability(value)
    }
}

/// One externally resolved binary probability observation bound to an exact
/// calibration population identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryCalibrationObservationV1 {
    observation_id: CalibrationIdentity,
    key: CalibrationKeyV1,
    forecast: UnitProbability,
    outcome: bool,
    resolved_at_unix_ms: NonZeroU64,
}

impl BinaryCalibrationObservationV1 {
    pub fn new(
        observation_id: CalibrationIdentity,
        key: CalibrationKeyV1,
        forecast: f64,
        outcome: bool,
        resolved_at_unix_ms: u64,
    ) -> Result<Self, CalibrationObservationError> {
        Ok(Self {
            observation_id,
            key,
            forecast: UnitProbability::new(forecast)?,
            outcome,
            resolved_at_unix_ms: NonZeroU64::new(resolved_at_unix_ms)
                .ok_or(CalibrationObservationError::ZeroResolvedAt)?,
        })
    }

    pub fn observation_id(&self) -> &CalibrationIdentity {
        &self.observation_id
    }

    pub fn key(&self) -> &CalibrationKeyV1 {
        &self.key
    }

    pub const fn forecast(&self) -> UnitProbability {
        self.forecast
    }

    pub const fn outcome(&self) -> bool {
        self.outcome
    }

    pub const fn resolved_at_unix_ms(&self) -> NonZeroU64 {
        self.resolved_at_unix_ms
    }

    pub fn brier_component(&self) -> f64 {
        let observed = if self.outcome { 1.0 } else { 0.0 };
        (self.forecast.get() - observed).powi(2)
    }
}

/// Failure to build or extend an exact-key calibration cohort.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationCohortError {
    ZeroCapacity,
    KeyMismatch,
    DuplicateObservation { observation_id: CalibrationIdentity },
    CapacityExceeded { capacity: usize },
}

impl fmt::Display for CalibrationCohortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCapacity => write!(f, "calibration cohort capacity must be non-zero"),
            Self::KeyMismatch => write!(
                f,
                "observation calibration key does not match cohort calibration key"
            ),
            Self::DuplicateObservation { observation_id } => {
                write!(f, "duplicate calibration observation {observation_id}")
            }
            Self::CapacityExceeded { capacity } => write!(
                f,
                "calibration cohort capacity {capacity} exceeded; evidence is not silently evicted"
            ),
        }
    }
}

impl std::error::Error for CalibrationCohortError {}

/// Exact-key bounded calibration cohort.
///
/// Capacity is a storage safety bound, not a rolling-window policy. Once full,
/// insertion fails explicitly instead of silently dropping historical evidence.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CalibrationCohortV1 {
    key: CalibrationKeyV1,
    capacity: NonZeroUsize,
    observations: Vec<BinaryCalibrationObservationV1>,
}

impl CalibrationCohortV1 {
    pub fn new(key: CalibrationKeyV1, capacity: usize) -> Result<Self, CalibrationCohortError> {
        let capacity = NonZeroUsize::new(capacity).ok_or(CalibrationCohortError::ZeroCapacity)?;
        Ok(Self {
            key,
            capacity,
            observations: Vec::new(),
        })
    }

    fn from_parts(
        key: CalibrationKeyV1,
        capacity: usize,
        observations: Vec<BinaryCalibrationObservationV1>,
    ) -> Result<Self, CalibrationCohortError> {
        let mut cohort = Self::new(key, capacity)?;
        for observation in observations {
            cohort.insert(observation)?;
        }
        Ok(cohort)
    }

    pub fn key(&self) -> &CalibrationKeyV1 {
        &self.key
    }

    pub const fn capacity(&self) -> usize {
        self.capacity.get()
    }

    pub fn sample_count(&self) -> usize {
        self.observations.len()
    }

    pub fn observations(&self) -> &[BinaryCalibrationObservationV1] {
        &self.observations
    }

    pub fn insert(
        &mut self,
        observation: BinaryCalibrationObservationV1,
    ) -> Result<(), CalibrationCohortError> {
        if observation.key() != &self.key {
            return Err(CalibrationCohortError::KeyMismatch);
        }
        if self
            .observations
            .iter()
            .any(|existing| existing.observation_id() == observation.observation_id())
        {
            return Err(CalibrationCohortError::DuplicateObservation {
                observation_id: observation.observation_id().clone(),
            });
        }
        if self.observations.len() >= self.capacity.get() {
            return Err(CalibrationCohortError::CapacityExceeded {
                capacity: self.capacity.get(),
            });
        }
        self.observations.push(observation);
        Ok(())
    }

    /// Mean Brier score of the exact-key cohort. `None` means no observations,
    /// not perfect calibration.
    pub fn mean_brier(&self) -> Option<f64> {
        if self.observations.is_empty() {
            return None;
        }
        Some(
            self.observations
                .iter()
                .map(BinaryCalibrationObservationV1::brier_component)
                .sum::<f64>()
                / self.observations.len() as f64,
        )
    }
}

#[derive(Deserialize)]
struct CalibrationCohortWire {
    key: CalibrationKeyV1,
    capacity: usize,
    observations: Vec<BinaryCalibrationObservationV1>,
}

impl<'de> Deserialize<'de> for CalibrationCohortV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = CalibrationCohortWire::deserialize(deserializer)?;
        Self::from_parts(wire.key, wire.capacity, wire.observations)
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::recursive_improvement::{
        CalibrationIdentity, CalibrationKeyV1, PredictionDomain, PredictionHorizon,
    };

    fn id(value: &str) -> CalibrationIdentity {
        CalibrationIdentity::new(value).unwrap()
    }

    fn key(population: &str) -> CalibrationKeyV1 {
        CalibrationKeyV1::new(
            PredictionDomain::ToolUse,
            id("producer:magi@v1"),
            id("calibrator:brier@v1"),
            id("outcome:binary-success@v1"),
            id("resolver:exit-code@v1"),
            id("context:tool@v1"),
            PredictionHorizon::bounded_millis(1_000).unwrap(),
            id(population),
        )
    }

    fn observation(
        observation_id: &str,
        calibration_key: CalibrationKeyV1,
        forecast: f64,
        outcome: bool,
    ) -> BinaryCalibrationObservationV1 {
        BinaryCalibrationObservationV1::new(
            id(observation_id),
            calibration_key,
            forecast,
            outcome,
            1,
        )
        .unwrap()
    }

    #[test]
    fn probability_rejects_nonfinite_and_out_of_range_values() {
        assert!(UnitProbability::new(f64::NAN).is_err());
        assert!(UnitProbability::new(f64::INFINITY).is_err());
        assert!(UnitProbability::new(-0.01).is_err());
        assert!(UnitProbability::new(1.01).is_err());
        assert_eq!(UnitProbability::new(0.0).unwrap().get(), 0.0);
        assert_eq!(UnitProbability::new(1.0).unwrap().get(), 1.0);
    }

    #[test]
    fn probability_deserialization_cannot_bypass_validation() {
        assert!(serde_json::from_str::<UnitProbability>("1.5").is_err());
    }

    #[test]
    fn cohort_rejects_cross_key_pooling_even_with_same_coarse_domain() {
        let local = key("population:local@v1");
        let remote = key("population:remote@v1");
        let mut cohort = CalibrationCohortV1::new(local, 4).unwrap();

        let err = cohort
            .insert(observation("obs:1", remote, 0.8, true))
            .unwrap_err();
        assert_eq!(err, CalibrationCohortError::KeyMismatch);
        assert_eq!(cohort.sample_count(), 0);
    }

    #[test]
    fn duplicate_observation_identity_is_rejected() {
        let calibration_key = key("population:local@v1");
        let mut cohort = CalibrationCohortV1::new(calibration_key.clone(), 4).unwrap();
        cohort
            .insert(observation("obs:1", calibration_key.clone(), 0.8, true))
            .unwrap();

        let err = cohort
            .insert(observation("obs:1", calibration_key, 0.7, false))
            .unwrap_err();
        assert!(matches!(
            err,
            CalibrationCohortError::DuplicateObservation { .. }
        ));
        assert_eq!(cohort.sample_count(), 1);
    }

    #[test]
    fn capacity_is_explicit_failure_not_silent_eviction() {
        let calibration_key = key("population:local@v1");
        let mut cohort = CalibrationCohortV1::new(calibration_key.clone(), 1).unwrap();
        cohort
            .insert(observation("obs:1", calibration_key.clone(), 0.8, true))
            .unwrap();

        assert_eq!(
            cohort
                .insert(observation("obs:2", calibration_key, 0.8, true))
                .unwrap_err(),
            CalibrationCohortError::CapacityExceeded { capacity: 1 }
        );
        assert_eq!(cohort.sample_count(), 1);
    }

    #[test]
    fn mean_brier_is_exact_key_measurement_and_empty_is_unmeasured() {
        let calibration_key = key("population:local@v1");
        let mut cohort = CalibrationCohortV1::new(calibration_key.clone(), 4).unwrap();
        assert_eq!(cohort.mean_brier(), None);

        cohort
            .insert(observation("obs:1", calibration_key.clone(), 0.8, true))
            .unwrap();
        cohort
            .insert(observation("obs:2", calibration_key, 0.8, false))
            .unwrap();

        // ((0.8 - 1)^2 + (0.8 - 0)^2) / 2 = (0.04 + 0.64) / 2
        assert!((cohort.mean_brier().unwrap() - 0.34).abs() < 1e-12);
    }

    #[test]
    fn cohort_deserialization_revalidates_cross_key_and_duplicates() {
        let local = key("population:local@v1");
        let remote = key("population:remote@v1");
        let foreign = observation("obs:1", remote, 0.8, true);

        let cross_key_json = serde_json::json!({
            "key": local,
            "capacity": 4,
            "observations": [foreign]
        });
        assert!(serde_json::from_value::<CalibrationCohortV1>(cross_key_json).is_err());

        let local = key("population:local@v1");
        let first = observation("obs:1", local.clone(), 0.8, true);
        let duplicate = observation("obs:1", local.clone(), 0.7, false);
        let duplicate_json = serde_json::json!({
            "key": local,
            "capacity": 4,
            "observations": [first, duplicate]
        });
        assert!(serde_json::from_value::<CalibrationCohortV1>(duplicate_json).is_err());
    }
}
