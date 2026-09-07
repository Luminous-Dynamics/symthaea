// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Quality-aware compatibility for typed multiscale resources.
//!
//! `symthaea-resource-model` answers "how much of what resource crosses this
//! boundary?" This crate answers the orthogonal question "is that resource of a
//! usable grade for this consumer?" It keeps amount/conservation accounting
//! separate from thermal, purity, latency, provenance, and categorical quality.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use symthaea_resource_model::ResourceKey;
use thiserror::Error;

/// Numeric quality dimensions that can constrain resource compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum QualityMetric {
    TemperatureCelsius,
    PressurePascal,
    PurityFraction,
    CarbonIntensityKgPerJoule,
    LatencySeconds,
    AvailabilityFraction,
}

impl QualityMetric {
    fn validate_value(self, value: f64) -> Result<(), QualityError> {
        if !value.is_finite() {
            return Err(QualityError::NonFiniteMetric { metric: self, value });
        }
        match self {
            Self::TemperatureCelsius => Ok(()),
            Self::PressurePascal | Self::CarbonIntensityKgPerJoule | Self::LatencySeconds => {
                if value < 0.0 {
                    Err(QualityError::NegativeMetric { metric: self, value })
                } else {
                    Ok(())
                }
            }
            Self::PurityFraction | Self::AvailabilityFraction => {
                if (0.0..=1.0).contains(&value) {
                    Ok(())
                } else {
                    Err(QualityError::FractionOutOfRange { metric: self, value })
                }
            }
        }
    }
}

/// Quality metadata for one typed resource dimension.
///
/// Numeric metrics are strongly enumerated. Open categorical properties use
/// namespaced string tags so domain adapters can express hardware/provenance classes
/// without forcing those catalogs into this foundational crate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceQualityProfile {
    pub key: ResourceKey,
    numeric: BTreeMap<QualityMetric, f64>,
    tags: BTreeMap<String, String>,
}

impl ResourceQualityProfile {
    pub fn new(key: ResourceKey) -> Self {
        Self {
            key,
            numeric: BTreeMap::new(),
            tags: BTreeMap::new(),
        }
    }

    pub fn set_numeric(
        &mut self,
        metric: QualityMetric,
        value: f64,
    ) -> Result<(), QualityError> {
        metric.validate_value(value)?;
        self.numeric.insert(metric, value);
        Ok(())
    }

    pub fn numeric(&self, metric: QualityMetric) -> Option<f64> {
        self.numeric.get(&metric).copied()
    }

    pub fn set_tag(
        &mut self,
        namespace: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<(), QualityError> {
        let namespace = namespace.into();
        let value = value.into();
        if namespace.trim().is_empty() {
            return Err(QualityError::EmptyTagNamespace);
        }
        if value.trim().is_empty() {
            return Err(QualityError::EmptyTagValue(namespace));
        }
        self.tags.insert(namespace, value);
        Ok(())
    }

    pub fn tag(&self, namespace: &str) -> Option<&str> {
        self.tags.get(namespace).map(String::as_str)
    }
}

/// Numeric interval requirement. Either bound may be omitted, but not both.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NumericQualityConstraint {
    pub metric: QualityMetric,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
}

impl NumericQualityConstraint {
    pub fn validate(&self) -> Result<(), QualityError> {
        if self.minimum.is_none() && self.maximum.is_none() {
            return Err(QualityError::UnboundedConstraint(self.metric));
        }
        if let Some(minimum) = self.minimum {
            self.metric.validate_value(minimum)?;
        }
        if let Some(maximum) = self.maximum {
            self.metric.validate_value(maximum)?;
        }
        if let (Some(minimum), Some(maximum)) = (self.minimum, self.maximum)
            && minimum > maximum
        {
            return Err(QualityError::InvertedConstraint {
                metric: self.metric,
                minimum,
                maximum,
            });
        }
        Ok(())
    }
}

/// Required resource grade for a consumer or transfer boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceQualityRequirement {
    pub key: ResourceKey,
    pub numeric: Vec<NumericQualityConstraint>,
    pub required_tags: BTreeMap<String, String>,
}

impl ResourceQualityRequirement {
    pub fn new(key: ResourceKey) -> Self {
        Self {
            key,
            numeric: Vec::new(),
            required_tags: BTreeMap::new(),
        }
    }

    pub fn require_numeric(
        &mut self,
        constraint: NumericQualityConstraint,
    ) -> Result<(), QualityError> {
        constraint.validate()?;
        self.numeric.push(constraint);
        Ok(())
    }

    pub fn require_tag(
        &mut self,
        namespace: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<(), QualityError> {
        let namespace = namespace.into();
        let value = value.into();
        if namespace.trim().is_empty() {
            return Err(QualityError::EmptyTagNamespace);
        }
        if value.trim().is_empty() {
            return Err(QualityError::EmptyTagValue(namespace));
        }
        self.required_tags.insert(namespace, value);
        Ok(())
    }

    /// Evaluate a producer profile against all required grade constraints.
    pub fn evaluate(
        &self,
        profile: &ResourceQualityProfile,
    ) -> Result<QualityCompatibility, QualityError> {
        if profile.key != self.key {
            return Ok(QualityCompatibility {
                compatible: false,
                violations: vec![QualityViolation::ResourceKeyMismatch {
                    required: self.key,
                    provided: profile.key,
                }],
            });
        }

        let mut violations = Vec::new();
        for constraint in &self.numeric {
            constraint.validate()?;
            let Some(value) = profile.numeric(constraint.metric) else {
                violations.push(QualityViolation::MissingMetric(constraint.metric));
                continue;
            };
            constraint.metric.validate_value(value)?;
            if let Some(minimum) = constraint.minimum
                && value < minimum
            {
                violations.push(QualityViolation::BelowMinimum {
                    metric: constraint.metric,
                    observed: value,
                    minimum,
                });
            }
            if let Some(maximum) = constraint.maximum
                && value > maximum
            {
                violations.push(QualityViolation::AboveMaximum {
                    metric: constraint.metric,
                    observed: value,
                    maximum,
                });
            }
        }

        for (namespace, expected) in &self.required_tags {
            match profile.tag(namespace) {
                None => violations.push(QualityViolation::MissingTag(namespace.clone())),
                Some(observed) if observed != expected => {
                    violations.push(QualityViolation::TagMismatch {
                        namespace: namespace.clone(),
                        expected: expected.clone(),
                        observed: observed.to_owned(),
                    });
                }
                Some(_) => {}
            }
        }

        Ok(QualityCompatibility {
            compatible: violations.is_empty(),
            violations,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityCompatibility {
    pub compatible: bool,
    pub violations: Vec<QualityViolation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QualityViolation {
    ResourceKeyMismatch {
        required: ResourceKey,
        provided: ResourceKey,
    },
    MissingMetric(QualityMetric),
    BelowMinimum {
        metric: QualityMetric,
        observed: f64,
        minimum: f64,
    },
    AboveMaximum {
        metric: QualityMetric,
        observed: f64,
        maximum: f64,
    },
    MissingTag(String),
    TagMismatch {
        namespace: String,
        expected: String,
        observed: String,
    },
}

/// Ideal exergy fraction of heat available at a uniform source temperature
/// relative to an ambient reference, `1 - T_ambient / T_source` in kelvin.
///
/// Returns zero when the source is not hotter than ambient. Temperatures below
/// absolute zero are rejected.
pub fn ideal_heat_exergy_fraction(
    source_temperature_c: f64,
    ambient_temperature_c: f64,
) -> Result<f64, QualityError> {
    if !source_temperature_c.is_finite() || !ambient_temperature_c.is_finite() {
        return Err(QualityError::InvalidTemperature);
    }
    const KELVIN_OFFSET: f64 = 273.15;
    let source_k = source_temperature_c + KELVIN_OFFSET;
    let ambient_k = ambient_temperature_c + KELVIN_OFFSET;
    if source_k <= 0.0 || ambient_k <= 0.0 {
        return Err(QualityError::InvalidTemperature);
    }
    if source_k <= ambient_k {
        return Ok(0.0);
    }
    Ok(1.0 - ambient_k / source_k)
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum QualityError {
    #[error("non-finite value {value} for quality metric {metric:?}")]
    NonFiniteMetric { metric: QualityMetric, value: f64 },
    #[error("negative value {value} for non-negative quality metric {metric:?}")]
    NegativeMetric { metric: QualityMetric, value: f64 },
    #[error("fraction value {value} outside [0,1] for quality metric {metric:?}")]
    FractionOutOfRange { metric: QualityMetric, value: f64 },
    #[error("quality constraint for {0:?} has neither minimum nor maximum")]
    UnboundedConstraint(QualityMetric),
    #[error("quality constraint for {metric:?} has minimum {minimum} > maximum {maximum}")]
    InvertedConstraint {
        metric: QualityMetric,
        minimum: f64,
        maximum: f64,
    },
    #[error("quality tag namespace must not be empty")]
    EmptyTagNamespace,
    #[error("quality tag {0} must not have an empty value")]
    EmptyTagValue(String),
    #[error("temperature must be finite and above absolute zero")]
    InvalidTemperature,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_resource_model::{ResourceKind, ResourceUnit};

    fn thermal_key() -> ResourceKey {
        ResourceKey::new(ResourceKind::ThermalEnergy, ResourceUnit::Watt).unwrap()
    }

    fn water_key() -> ResourceKey {
        ResourceKey::new(ResourceKind::Water, ResourceUnit::CubicMeterPerSecond).unwrap()
    }

    #[test]
    fn low_grade_heat_does_not_satisfy_high_temperature_consumer() {
        let mut profile = ResourceQualityProfile::new(thermal_key());
        profile
            .set_numeric(QualityMetric::TemperatureCelsius, 35.0)
            .unwrap();
        let mut requirement = ResourceQualityRequirement::new(thermal_key());
        requirement
            .require_numeric(NumericQualityConstraint {
                metric: QualityMetric::TemperatureCelsius,
                minimum: Some(60.0),
                maximum: None,
            })
            .unwrap();
        let result = requirement.evaluate(&profile).unwrap();
        assert!(!result.compatible);
        assert!(matches!(
            result.violations.as_slice(),
            [QualityViolation::BelowMinimum { observed, minimum, .. }]
                if *observed == 35.0 && *minimum == 60.0
        ));
    }

    #[test]
    fn suitable_heat_grade_matches_consumer() {
        let mut profile = ResourceQualityProfile::new(thermal_key());
        profile
            .set_numeric(QualityMetric::TemperatureCelsius, 72.0)
            .unwrap();
        let mut requirement = ResourceQualityRequirement::new(thermal_key());
        requirement
            .require_numeric(NumericQualityConstraint {
                metric: QualityMetric::TemperatureCelsius,
                minimum: Some(60.0),
                maximum: Some(90.0),
            })
            .unwrap();
        assert!(requirement.evaluate(&profile).unwrap().compatible);
    }

    #[test]
    fn purity_and_categorical_provenance_can_be_required_together() {
        let mut profile = ResourceQualityProfile::new(water_key());
        profile
            .set_numeric(QualityMetric::PurityFraction, 0.99)
            .unwrap();
        profile.set_tag("water.class", "potable").unwrap();

        let mut requirement = ResourceQualityRequirement::new(water_key());
        requirement
            .require_numeric(NumericQualityConstraint {
                metric: QualityMetric::PurityFraction,
                minimum: Some(0.95),
                maximum: None,
            })
            .unwrap();
        requirement.require_tag("water.class", "potable").unwrap();
        assert!(requirement.evaluate(&profile).unwrap().compatible);
    }

    #[test]
    fn categorical_mismatch_is_visible() {
        let mut profile = ResourceQualityProfile::new(water_key());
        profile.set_tag("water.class", "greywater").unwrap();
        let mut requirement = ResourceQualityRequirement::new(water_key());
        requirement.require_tag("water.class", "potable").unwrap();
        let result = requirement.evaluate(&profile).unwrap();
        assert!(matches!(
            result.violations.as_slice(),
            [QualityViolation::TagMismatch { namespace, expected, observed }]
                if namespace == "water.class" && expected == "potable" && observed == "greywater"
        ));
    }

    #[test]
    fn fractions_outside_unit_interval_are_rejected() {
        let mut profile = ResourceQualityProfile::new(water_key());
        assert!(matches!(
            profile.set_numeric(QualityMetric::PurityFraction, 1.1),
            Err(QualityError::FractionOutOfRange { .. })
        ));
    }

    #[test]
    fn wrong_resource_dimension_never_matches() {
        let profile = ResourceQualityProfile::new(water_key());
        let requirement = ResourceQualityRequirement::new(thermal_key());
        let result = requirement.evaluate(&profile).unwrap();
        assert!(!result.compatible);
        assert!(matches!(
            result.violations.as_slice(),
            [QualityViolation::ResourceKeyMismatch { .. }]
        ));
    }

    #[test]
    fn higher_temperature_heat_has_more_ideal_exergy() {
        let low = ideal_heat_exergy_fraction(35.0, 20.0).unwrap();
        let high = ideal_heat_exergy_fraction(80.0, 20.0).unwrap();
        assert!(low > 0.0);
        assert!(high > low);
        assert!(high < 1.0);
    }

    #[test]
    fn heat_at_or_below_ambient_has_zero_ideal_exergy() {
        assert_eq!(ideal_heat_exergy_fraction(20.0, 20.0).unwrap(), 0.0);
        assert_eq!(ideal_heat_exergy_fraction(10.0, 20.0).unwrap(), 0.0);
    }
}
