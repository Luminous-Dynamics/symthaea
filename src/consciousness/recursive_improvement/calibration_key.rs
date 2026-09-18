// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Stable identity for calibration populations.
//!
//! CAL-004B prevents a future calibrator or decision learner from pooling
//! evidence merely because observations share a coarse [`PredictionDomain`].
//! A calibration population is only the same population when its producer,
//! calibrator, outcome semantics, resolver, context semantics, prediction
//! horizon, and population identity are all the same.

use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;
use std::num::NonZeroU64;

use super::world_prediction::PredictionDomain;

const MAX_IDENTITY_LEN: usize = 160;

/// Validation error for calibration identity/key construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationKeyError {
    EmptyIdentity,
    IdentityTooLong { len: usize, max: usize },
    InvalidIdentityCharacter { character: char },
    ZeroHorizon,
}

impl fmt::Display for CalibrationKeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentity => write!(f, "calibration identity must not be empty"),
            Self::IdentityTooLong { len, max } => {
                write!(f, "calibration identity length {len} exceeds maximum {max}")
            }
            Self::InvalidIdentityCharacter { character } => write!(
                f,
                "calibration identity contains unsupported character {character:?}"
            ),
            Self::ZeroHorizon => write!(f, "bounded prediction horizon must be non-zero"),
        }
    }
}

impl std::error::Error for CalibrationKeyError {}

/// Stable, validated identifier used inside a calibration key.
///
/// This is intentionally not arbitrary display text. The restricted ASCII form
/// keeps identities deterministic across JSON, hashing, receipts, and future
/// cross-language implementations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct CalibrationIdentity(String);

impl CalibrationIdentity {
    pub fn new(value: impl Into<String>) -> Result<Self, CalibrationKeyError> {
        let value = value.into();
        if value.is_empty() {
            return Err(CalibrationKeyError::EmptyIdentity);
        }
        if value.len() > MAX_IDENTITY_LEN {
            return Err(CalibrationKeyError::IdentityTooLong {
                len: value.len(),
                max: MAX_IDENTITY_LEN,
            });
        }

        for character in value.chars() {
            let allowed = character.is_ascii_alphanumeric()
                || matches!(character, '-' | '_' | '.' | ':' | '/' | '@' | '+');
            if !allowed {
                return Err(CalibrationKeyError::InvalidIdentityCharacter { character });
            }
        }

        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for CalibrationIdentity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for CalibrationIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Prediction horizon whose semantics participate in calibration identity.
///
/// A 10 ms forecast and a one-hour forecast are different calibration
/// populations even if every other key field is identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PredictionHorizon {
    /// Outcome is resolved without an intentional temporal forecast horizon.
    Immediate,
    /// Outcome is expected within a concrete, non-zero duration.
    BoundedMillis { millis: NonZeroU64 },
    /// Resolution time is controlled by an external contract/authority rather
    /// than a fixed forecasting horizon.
    ExternalResolution,
}

impl PredictionHorizon {
    pub fn bounded_millis(millis: u64) -> Result<Self, CalibrationKeyError> {
        let millis = NonZeroU64::new(millis).ok_or(CalibrationKeyError::ZeroHorizon)?;
        Ok(Self::BoundedMillis { millis })
    }
}

/// Versioned identity of one calibration population.
///
/// Equality means evidence is eligible to be considered part of the same
/// calibration population. It does *not* mean the evidence is sufficient,
/// independent, current, or authority-bearing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CalibrationKeyV1 {
    domain: PredictionDomain,
    producer: CalibrationIdentity,
    calibrator: CalibrationIdentity,
    outcome_schema: CalibrationIdentity,
    resolver: CalibrationIdentity,
    context_schema: CalibrationIdentity,
    horizon: PredictionHorizon,
    population: CalibrationIdentity,
}

impl CalibrationKeyV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        domain: PredictionDomain,
        producer: CalibrationIdentity,
        calibrator: CalibrationIdentity,
        outcome_schema: CalibrationIdentity,
        resolver: CalibrationIdentity,
        context_schema: CalibrationIdentity,
        horizon: PredictionHorizon,
        population: CalibrationIdentity,
    ) -> Self {
        Self {
            domain,
            producer,
            calibrator,
            outcome_schema,
            resolver,
            context_schema,
            horizon,
            population,
        }
    }

    pub const fn domain(&self) -> PredictionDomain {
        self.domain
    }

    pub fn producer(&self) -> &CalibrationIdentity {
        &self.producer
    }

    pub fn calibrator(&self) -> &CalibrationIdentity {
        &self.calibrator
    }

    pub fn outcome_schema(&self) -> &CalibrationIdentity {
        &self.outcome_schema
    }

    pub fn resolver(&self) -> &CalibrationIdentity {
        &self.resolver
    }

    pub fn context_schema(&self) -> &CalibrationIdentity {
        &self.context_schema
    }

    pub const fn horizon(&self) -> PredictionHorizon {
        self.horizon
    }

    pub fn population(&self) -> &CalibrationIdentity {
        &self.population
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn id(value: &str) -> CalibrationIdentity {
        CalibrationIdentity::new(value).unwrap()
    }

    fn baseline() -> CalibrationKeyV1 {
        CalibrationKeyV1::new(
            PredictionDomain::ToolUse,
            id("producer:symthaea-magi@v1"),
            id("calibrator:brier-ece@v1"),
            id("outcome:shell-exit@v1"),
            id("resolver:exit-code@v1"),
            id("context:tool-action@v1"),
            PredictionHorizon::bounded_millis(30_000).unwrap(),
            id("population:local-shell@v1"),
        )
    }

    #[test]
    fn identity_rejects_whitespace_and_unstable_text() {
        assert!(matches!(
            CalibrationIdentity::new("producer with spaces"),
            Err(CalibrationKeyError::InvalidIdentityCharacter { .. })
        ));
        assert!(matches!(
            CalibrationIdentity::new(""),
            Err(CalibrationKeyError::EmptyIdentity)
        ));
    }

    #[test]
    fn deserialization_cannot_bypass_identity_validation() {
        let invalid = r#""producer with spaces""#;
        assert!(serde_json::from_str::<CalibrationIdentity>(invalid).is_err());
    }

    #[test]
    fn bounded_horizon_rejects_zero() {
        assert_eq!(
            PredictionHorizon::bounded_millis(0),
            Err(CalibrationKeyError::ZeroHorizon)
        );
    }

    #[test]
    fn same_coarse_domain_does_not_imply_same_calibration_population() {
        let base = baseline();
        let shifted_population = CalibrationKeyV1::new(
            base.domain(),
            base.producer().clone(),
            base.calibrator().clone(),
            base.outcome_schema().clone(),
            base.resolver().clone(),
            base.context_schema().clone(),
            base.horizon(),
            id("population:remote-shell@v1"),
        );
        let shifted_producer = CalibrationKeyV1::new(
            base.domain(),
            id("producer:other-agent@v1"),
            base.calibrator().clone(),
            base.outcome_schema().clone(),
            base.resolver().clone(),
            base.context_schema().clone(),
            base.horizon(),
            base.population().clone(),
        );

        assert_ne!(base, shifted_population);
        assert_ne!(base, shifted_producer);
    }

    #[test]
    fn horizon_and_resolver_are_identity_bearing() {
        let base = baseline();
        let different_horizon = CalibrationKeyV1::new(
            base.domain(),
            base.producer().clone(),
            base.calibrator().clone(),
            base.outcome_schema().clone(),
            base.resolver().clone(),
            base.context_schema().clone(),
            PredictionHorizon::bounded_millis(60_000).unwrap(),
            base.population().clone(),
        );
        let different_resolver = CalibrationKeyV1::new(
            base.domain(),
            base.producer().clone(),
            base.calibrator().clone(),
            base.outcome_schema().clone(),
            id("resolver:human-confirmation@v1"),
            base.context_schema().clone(),
            base.horizon(),
            base.population().clone(),
        );

        assert_ne!(base, different_horizon);
        assert_ne!(base, different_resolver);
    }

    #[test]
    fn exact_key_is_hash_stable_within_the_type_contract() {
        let base = baseline();
        let same = base.clone();
        let mut set = HashSet::new();
        set.insert(base);
        assert!(set.contains(&same));
    }

    #[test]
    fn json_round_trip_preserves_exact_identity() {
        let key = baseline();
        let json = serde_json::to_string(&key).unwrap();
        let restored: CalibrationKeyV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, key);
    }
}
