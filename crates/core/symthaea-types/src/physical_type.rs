// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical gradual physical semantics shared across Symthaea.
//!
//! This module intentionally does not depend on physics solvers or ASTs.
//! Physical meaning, evidence maturity, and executable expression inference are
//! separate concerns. Consumers can therefore share one vocabulary without
//! creating dependency cycles.

use serde::{Deserialize, Serialize};

/// Seven-base-dimension SI signature: [M, L, T, I, Θ, N, J].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PhysicalDimension {
    pub mass: i8,
    pub length: i8,
    pub time: i8,
    pub current: i8,
    pub temperature: i8,
    pub amount: i8,
    pub luminous: i8,
}

impl PhysicalDimension {
    pub const DIMENSIONLESS: Self = Self {
        mass: 0, length: 0, time: 0, current: 0,
        temperature: 0, amount: 0, luminous: 0,
    };

    pub const MASS: Self = Self { mass: 1, ..Self::DIMENSIONLESS };
    pub const LENGTH: Self = Self { length: 1, ..Self::DIMENSIONLESS };
    pub const TIME: Self = Self { time: 1, ..Self::DIMENSIONLESS };
    pub const VELOCITY: Self = Self { length: 1, time: -1, ..Self::DIMENSIONLESS };
    pub const ACCELERATION: Self = Self { length: 1, time: -2, ..Self::DIMENSIONLESS };
    pub const MOMENTUM: Self = Self { mass: 1, length: 1, time: -1, ..Self::DIMENSIONLESS };
    pub const FORCE: Self = Self { mass: 1, length: 1, time: -2, ..Self::DIMENSIONLESS };
    pub const ENERGY: Self = Self { mass: 1, length: 2, time: -2, ..Self::DIMENSIONLESS };
    pub const POWER: Self = Self { mass: 1, length: 2, time: -3, ..Self::DIMENSIONLESS };
    pub const PRESSURE: Self = Self { mass: 1, length: -1, time: -2, ..Self::DIMENSIONLESS };
    pub const FREQUENCY: Self = Self { time: -1, ..Self::DIMENSIONLESS };
    pub const TEMPERATURE: Self = Self { temperature: 1, ..Self::DIMENSIONLESS };
    pub const CHARGE: Self = Self { time: 1, current: 1, ..Self::DIMENSIONLESS };

    pub const fn from_array(e: [i8; 7]) -> Self {
        Self {
            mass: e[0], length: e[1], time: e[2], current: e[3],
            temperature: e[4], amount: e[5], luminous: e[6],
        }
    }

    pub const fn as_array(self) -> [i8; 7] {
        [self.mass, self.length, self.time, self.current,
         self.temperature, self.amount, self.luminous]
    }

    pub const fn is_dimensionless(self) -> bool {
        self.mass == 0 && self.length == 0 && self.time == 0 &&
        self.current == 0 && self.temperature == 0 &&
        self.amount == 0 && self.luminous == 0
    }

    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        let a = self.as_array();
        let b = rhs.as_array();
        let mut out = [0i8; 7];
        for i in 0..7 {
            out[i] = a[i].checked_add(b[i])?;
        }
        Some(Self::from_array(out))
    }

    pub fn checked_sub(self, rhs: Self) -> Option<Self> {
        let a = self.as_array();
        let b = rhs.as_array();
        let mut out = [0i8; 7];
        for i in 0..7 {
            out[i] = a[i].checked_sub(b[i])?;
        }
        Some(Self::from_array(out))
    }

    /// Convenience arithmetic for already-bounded exponents.
    /// New fail-closed inference code should prefer checked_add/checked_sub.
    pub fn add(self, rhs: Self) -> Self {
        self.checked_add(rhs).expect("physical dimension exponent overflow")
    }

    pub fn sub(self, rhs: Self) -> Self {
        self.checked_sub(rhs).expect("physical dimension exponent overflow")
    }

    pub fn scale(self, factor: i8) -> Option<Self> {
        let mut out = [0i8; 7];
        for (i, exponent) in self.as_array().into_iter().enumerate() {
            out[i] = exponent.checked_mul(factor)?;
        }
        Some(Self::from_array(out))
    }
}

/// Exact rational coefficient used by UnitTransform for scale or affine offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RationalScale {
    pub numerator: i64,
    pub denominator: i64,
}

impl RationalScale {
    pub const ONE: Self = Self { numerator: 1, denominator: 1 };

    pub const fn new(numerator: i64, denominator: i64) -> Option<Self> {
        if denominator == 0 || denominator < 0 {
            None
        } else {
            Some(Self { numerator, denominator })
        }
    }

    pub const fn is_valid(self) -> bool {
        self.denominator > 0
    }

    pub fn as_f64(self) -> Result<f64, PhysicalTypeError> {
        if !self.is_valid() {
            return Err(PhysicalTypeError {
                operation: "rational_scale".into(),
                reason: "rational denominator must be positive".into(),
            });
        }
        let value = self.numerator as f64 / self.denominator as f64;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(PhysicalTypeError {
                operation: "rational_scale".into(),
                reason: "rational value is not finite in f64".into(),
            })
        }
    }
}

/// Exact affine conversion to SI: si = value * scale + offset.
/// Affine offsets are required for absolute temperature-like units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UnitTransform {
    pub scale: RationalScale,
    pub offset: RationalScale,
}

impl UnitTransform {
    pub const IDENTITY: Self = Self {
        scale: RationalScale::ONE,
        offset: RationalScale { numerator: 0, denominator: 1 },
    };

    pub const fn new(scale: RationalScale, offset: RationalScale) -> Self {
        Self { scale, offset }
    }

    pub fn validate(self) -> Result<(), PhysicalTypeError> {
        if !self.scale.is_valid() || self.scale.numerator <= 0 {
            return Err(PhysicalTypeError {
                operation: "unit_transform".into(),
                reason: "unit scale must have a positive denominator and strictly positive numerator".into(),
            });
        }
        if !self.offset.is_valid() {
            return Err(PhysicalTypeError {
                operation: "unit_transform".into(),
                reason: "unit offset denominator must be positive".into(),
            });
        }
        Ok(())
    }

    pub fn to_si_value(self, value: f64) -> Result<f64, PhysicalTypeError> {
        if !value.is_finite() {
            return Err(PhysicalTypeError {
                operation: "unit_conversion".into(),
                reason: "value must be finite".into(),
            });
        }
        self.validate()?;
        let scale = self.scale.as_f64()?;
        let offset = self.offset.as_f64()?;
        let si = value * scale + offset;
        if si.is_finite() {
            Ok(si)
        } else {
            Err(PhysicalTypeError {
                operation: "unit_conversion".into(),
                reason: "SI conversion produced a non-finite value".into(),
            })
        }
    }

    pub fn from_si_value(self, si_value: f64) -> Result<f64, PhysicalTypeError> {
        if !si_value.is_finite() {
            return Err(PhysicalTypeError {
                operation: "unit_conversion".into(),
                reason: "SI value must be finite".into(),
            });
        }
        self.validate()?;
        let scale = self.scale.as_f64()?;
        let offset = self.offset.as_f64()?;
        let value = (si_value - offset) / scale;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(PhysicalTypeError {
                operation: "unit_conversion".into(),
                reason: "conversion from SI produced a non-finite value".into(),
            })
        }
    }
}

/// Backward-compatible name for multiplicative units.
pub type UnitScale = RationalScale;

/// Stable external semantic identifier. The core does not interpret the
/// namespace; adapters may map it to QUDT, SysML, domain catalogs, etc.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SemanticIdentifier {
    pub namespace: String,
    pub identifier: String,
}

impl SemanticIdentifier {
    pub fn new(namespace: impl Into<String>, identifier: impl Into<String>) -> Option<Self> {
        let namespace = namespace.into();
        let identifier = identifier.into();
        if namespace.trim().is_empty() || identifier.trim().is_empty() {
            None
        } else {
            Some(Self { namespace, identifier })
        }
    }

    fn validate(&self) -> Result<(), PhysicalTypeError> {
        if self.namespace.trim().is_empty() || self.identifier.trim().is_empty() {
            return Err(PhysicalTypeError {
                operation: "semantic_identifier".into(),
                reason: "semantic identifier namespace and identifier must be non-empty".into(),
            });
        }
        Ok(())
    }
}

/// Named unit identity plus exact conversion to canonical SI semantics.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UnitRef {
    pub symbol: String,
    pub transform_to_si: UnitTransform,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_id: Option<SemanticIdentifier>,
}

impl UnitRef {
    pub fn validate(&self) -> Result<(), PhysicalTypeError> {
        if self.symbol.trim().is_empty() {
            return Err(PhysicalTypeError {
                operation: "unit".into(),
                reason: "unit symbol cannot be empty".into(),
            });
        }
        self.transform_to_si.validate()?;
        if let Some(semantic_id) = &self.semantic_id {
            semantic_id.validate()?;
        }
        Ok(())
    }
}

/// Physical quantity meaning, distinct from evidence about that meaning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicalType {
    pub kind: QuantityKind,
    /// Optional external quantity-kind identity for ontology interoperability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_id: Option<SemanticIdentifier>,
    /// None means genuinely unknown; it is never encoded as dimensionless.
    pub dimension: Option<PhysicalDimension>,
    pub unit: Option<UnitRef>,
    pub scalar: ScalarDomain,
    pub refinements: Vec<Refinement>,
}

impl PhysicalType {
    pub fn unknown() -> Self {
        Self {
            kind: QuantityKind::Unknown,
            semantic_id: None,
            dimension: None,
            unit: None,
            scalar: ScalarDomain::Unknown,
            refinements: Vec::new(),
        }
    }

    pub fn dimensionless() -> Self {
        Self {
            kind: QuantityKind::Dimensionless,
            semantic_id: None,
            dimension: Some(PhysicalDimension::DIMENSIONLESS),
            unit: None,
            scalar: ScalarDomain::Real,
            refinements: Vec::new(),
        }
    }

    pub fn with_kind(kind: QuantityKind, dimension: PhysicalDimension) -> Self {
        Self {
            kind,
            semantic_id: None,
            dimension: Some(dimension),
            unit: None,
            scalar: ScalarDomain::Real,
            refinements: Vec::new(),
        }
    }

    pub fn with_unit(mut self, unit: UnitRef) -> Self {
        self.unit = Some(unit);
        self
    }

    pub fn with_semantic_id(mut self, semantic_id: SemanticIdentifier) -> Self {
        self.semantic_id = Some(semantic_id);
        self
    }

    pub fn has_refinement(&self, refinement: Refinement) -> bool {
        self.refinements.contains(&refinement)
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("PhysicalType serialization must be infallible")
    }

    pub fn digest(&self) -> [u8; 32] {
        *blake3::hash(&self.canonical_bytes()).as_bytes()
    }

    pub fn digest_hex(&self) -> String {
        blake3::hash(&self.canonical_bytes()).to_hex().to_string()
    }

    /// Fail-closed compatibility judgment for a typed signal/port boundary.
    /// Infer the type of the first derivative with respect to an independent
    /// physical quantity. Higher-order derivatives may be represented by
    /// repeatedly applying this operation.
    pub fn derivative_with(&self, independent: &Self) -> TypeJudgement<Self> {
        let (Some(value_dim), Some(independent_dim)) = (self.dimension, independent.dimension) else {
            return TypeJudgement::Unknown(
                "derivative requires known dimensions for value and independent variable".into(),
            );
        };
        let Some(dimension) = value_dim.checked_sub(independent_dim) else {
            return TypeJudgement::Invalid(PhysicalTypeError {
                operation: "derivative".into(),
                reason: "physical dimension exponent overflow".into(),
            });
        };
        let kind = match (self.kind, independent.kind) {
            (QuantityKind::Length, QuantityKind::Time) => QuantityKind::Velocity,
            (QuantityKind::Velocity, QuantityKind::Time) => QuantityKind::Acceleration,
            (QuantityKind::Acceleration, QuantityKind::Time) => QuantityKind::Custom,
            _ => QuantityKind::Unknown,
        };
        TypeJudgement::Valid(Self::with_kind(kind, dimension))
    }

    pub fn validate(&self) -> Result<(), PhysicalTypeError> {
        if let Some(expected) = self.kind.expected_dimension() {
            match self.dimension {
                Some(actual) if actual == expected => {}
                Some(_) => {
                    return Err(PhysicalTypeError {
                        operation: "physical_type".into(),
                        reason: format!(
                            "quantity kind {:?} has an incompatible physical dimension",
                            self.kind
                        ),
                    });
                }
                None => {
                    return Err(PhysicalTypeError {
                        operation: "physical_type".into(),
                        reason: format!(
                            "quantity kind {:?} requires a known physical dimension",
                            self.kind
                        ),
                    });
                }
            }
        }
        if let Some(unit) = &self.unit {
            unit.validate()?;
            if self.kind == QuantityKind::TemperatureDifference
                && unit.transform_to_si.offset.numerator != 0
            {
                return Err(PhysicalTypeError {
                    operation: "physical_type".into(),
                    reason:
                        "temperature differences cannot use affine unit offsets; use a delta unit"
                            .into(),
                });
            }
        }
        if let Some(semantic_id) = &self.semantic_id {
            semantic_id.validate()?;
        }
        Ok(())
    }

    pub fn judge_compatibility(&self, other: &Self) -> TypeJudgement<()> {
        if let Err(error) = self.validate() {
            return TypeJudgement::Invalid(error);
        }
        if let Err(error) = other.validate() {
            return TypeJudgement::Invalid(error);
        }
        let (Some(a), Some(b)) = (self.dimension, other.dimension) else {
            return TypeJudgement::Unknown(
                "physical dimension is unknown on one or both sides of the boundary".into(),
            );
        };
        if a != b {
            return TypeJudgement::Invalid(PhysicalTypeError {
                operation: "compatibility".into(),
                reason: "physical dimensions differ".into(),
            });
        }
        if self.kind != QuantityKind::Unknown
            && other.kind != QuantityKind::Unknown
            && self.kind != other.kind
        {
            return TypeJudgement::Invalid(PhysicalTypeError {
                operation: "compatibility".into(),
                reason: "physical quantity kinds differ".into(),
            });
        }
        TypeJudgement::Valid(())
    }

    /// Compatibility required when an edge carries actual numeric values.
    ///
    /// Ordinary semantic compatibility does not require unit metadata on both
    /// sides, because a symbolic quantity may intentionally omit display-unit
    /// information. Executable numeric transport is stricter: either both
    /// sides have explicit units, or neither does.
    pub fn judge_numeric_compatibility(&self, other: &Self) -> TypeJudgement<()> {
        match self.judge_compatibility(other) {
            TypeJudgement::Invalid(error) => TypeJudgement::Invalid(error),
            TypeJudgement::Unknown(reason) => TypeJudgement::Unknown(reason),
            TypeJudgement::Valid(()) => {
                match (&self.unit, &other.unit) {
                    (Some(_), Some(_)) | (None, None) => {}
                    _ => {
                        return TypeJudgement::Unknown(
                            "numeric transport requires explicit units on both sides or neither side"
                                .into(),
                        );
                    }
                }

                match (self.scalar, other.scalar) {
                    (ScalarDomain::Unknown, _) | (_, ScalarDomain::Unknown) => {
                        TypeJudgement::Unknown(
                            "numeric transport requires known scalar domains on both sides"
                                .into(),
                        )
                    }
                    (left, right) if left == right => TypeJudgement::Valid(()),
                    _ => TypeJudgement::Invalid(PhysicalTypeError {
                        operation: "numeric_compatibility".into(),
                        reason: "scalar domains differ; an explicit numeric cast is required"
                            .into(),
                    }),
                }
            },
        }
    }

    pub fn convert_value_to(
        &self,
        value: f64,
        target: &Self,
    ) -> TypeJudgement<f64> {
        if let Err(error) = self.validate() {
            return TypeJudgement::Invalid(error);
        }
        if let Err(error) = target.validate() {
            return TypeJudgement::Invalid(error);
        }

        match self.judge_compatibility(target) {
            TypeJudgement::Invalid(error) => return TypeJudgement::Invalid(error),
            TypeJudgement::Unknown(reason) => return TypeJudgement::Unknown(reason),
            TypeJudgement::Valid(()) => {}
        }

        if !value.is_finite() {
            return TypeJudgement::Invalid(PhysicalTypeError {
                operation: "unit_conversion".into(),
                reason: "value must be finite".into(),
            });
        }

        match (&self.unit, &target.unit) {
            (None, None) => TypeJudgement::Valid(value),
            (Some(source), Some(destination)) => {
                let si = match source.transform_to_si.to_si_value(value) {
                    Ok(value) => value,
                    Err(error) => return TypeJudgement::Invalid(error),
                };
                match destination.transform_to_si.from_si_value(si) {
                    Ok(value) => TypeJudgement::Valid(value),
                    Err(error) => TypeJudgement::Invalid(error),
                }
            }
            _ => TypeJudgement::Unknown(
                "numeric conversion requires explicit units on both sides".into(),
            ),
        }
    }
}

/// Physical quantity kinds are stricter than dimensions. Energy and Torque,
/// for example, may share dimensions while remaining different semantic kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantityKind {
    Unknown,
    Dimensionless,
    Angle,
    Length,
    Mass,
    Time,
    Velocity,
    Acceleration,
    Momentum,
    Force,
    Energy,
    Power,
    Torque,
    Charge,
    Temperature,
    /// Difference/interval of thermodynamic temperature; unlike absolute
    /// temperature, its unit transform must not contain an affine offset.
    TemperatureDifference,
    Pressure,
    Stress,
    Strain,
    Frequency,
    Custom,
}

impl QuantityKind {
    /// Canonical SI dimension for quantity kinds with a fixed physical meaning.
    /// Unknown and Custom remain gradual and therefore do not constrain the
    /// dimension to a single built-in interpretation.
    pub const fn expected_dimension(self) -> Option<PhysicalDimension> {
        match self {
            Self::Unknown | Self::Custom => None,
            Self::Dimensionless | Self::Angle | Self::Strain => {
                Some(PhysicalDimension::DIMENSIONLESS)
            }
            Self::Length => Some(PhysicalDimension::LENGTH),
            Self::Mass => Some(PhysicalDimension::MASS),
            Self::Time => Some(PhysicalDimension::TIME),
            Self::Velocity => Some(PhysicalDimension::VELOCITY),
            Self::Acceleration => Some(PhysicalDimension::ACCELERATION),
            Self::Momentum => Some(PhysicalDimension::MOMENTUM),
            Self::Force => Some(PhysicalDimension::FORCE),
            Self::Energy | Self::Torque => Some(PhysicalDimension::ENERGY),
            Self::Power => Some(PhysicalDimension::POWER),
            Self::Charge => Some(PhysicalDimension::CHARGE),
            Self::Temperature | Self::TemperatureDifference => {
                Some(PhysicalDimension::TEMPERATURE)
            }
            Self::Pressure | Self::Stress => Some(PhysicalDimension::PRESSURE),
            Self::Frequency => Some(PhysicalDimension::FREQUENCY),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ScalarDomain {
    Unknown,
    Integer,
    Rational,
    Real,
    Complex,
    ApproximateReal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Refinement {
    Positive,
    NonNegative,
    NonZero,
    Bounded,
    Periodic,
}

/// Evidence/model maturity is intentionally independent from physical meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelMaturity {
    TextbookAnalytic,
    ValidatedNumerical,
    CalibratedEmpirical,
    ResearchPrototype,
    FrontierHypothesis,
    SyntheticInstrumental,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeJudgement<T> {
    Valid(T),
    Invalid(PhysicalTypeError),
    Unknown(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalTypeError {
    pub operation: String,
    pub reason: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_is_not_dimensionless() {
        let unknown = PhysicalType::unknown();
        assert_eq!(unknown.kind, QuantityKind::Unknown);
        assert_eq!(unknown.dimension, None);
    }

    #[test]
    fn derivative_preserves_physical_meaning() {
        let length = PhysicalType::with_kind(QuantityKind::Length, PhysicalDimension::LENGTH);
        let time = PhysicalType::with_kind(QuantityKind::Time, PhysicalDimension::TIME);
        match length.derivative_with(&time) {
            TypeJudgement::Valid(v) => {
                assert_eq!(v.kind, QuantityKind::Velocity);
                assert_eq!(v.dimension, Some(PhysicalDimension::VELOCITY));
            }
            other => panic!("unexpected judgment: {other:?}"),
        }
    }

    #[test]
    fn dimension_arithmetic_is_deterministic() {
        let force = PhysicalDimension::MASS.add(PhysicalDimension::ACCELERATION);
        assert_eq!(force, PhysicalDimension::FORCE);
        assert_eq!(
            PhysicalDimension::FORCE.add(PhysicalDimension::LENGTH),
            PhysicalDimension::ENERGY
        );
    }

    #[test]
    fn digest_is_stable_for_equal_types() {
        let a = PhysicalType::with_kind(QuantityKind::Energy, PhysicalDimension::ENERGY);
        let b = a.clone();
        assert_eq!(a.digest(), b.digest());
        assert_eq!(a.digest_hex(), b.digest_hex());
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn unit_scale_rejects_zero_denominator() {
        assert!(RationalScale::new(1, 0).is_none());
        assert!(RationalScale::new(1, -1).is_none());
    }

    #[test]
    fn affine_unit_transform_preserves_offset_semantics() {
        let celsius = UnitRef {
            symbol: "degC".into(),
            transform_to_si: UnitTransform::new(
                RationalScale::ONE,
                RationalScale { numerator: 27315, denominator: 100 },
            ),
            semantic_id: None,
        };
        assert_eq!(celsius.transform_to_si.offset.numerator, 27315);
    }

    #[test]
    fn unit_conversion_is_explicit_and_affine() {
        let length_m = PhysicalType::with_kind(QuantityKind::Length, PhysicalDimension::LENGTH)
            .with_unit(UnitRef {
                symbol: "m".into(),
                transform_to_si: UnitTransform::IDENTITY,
                semantic_id: None,
            });
        let length_ft = PhysicalType::with_kind(QuantityKind::Length, PhysicalDimension::LENGTH)
            .with_unit(UnitRef {
                symbol: "ft".into(),
                transform_to_si: UnitTransform::new(
                    RationalScale { numerator: 3048, denominator: 10000 },
                    RationalScale { numerator: 0, denominator: 1 },
                ),
                semantic_id: None,
            });

        match length_ft.convert_value_to(1.0, &length_m) {
            TypeJudgement::Valid(value) => assert!((value - 0.3048).abs() < 1e-12),
            other => panic!("unexpected conversion judgment: {other:?}"),
        }

        let celsius = PhysicalType::with_kind(
            QuantityKind::Temperature,
            PhysicalDimension::TEMPERATURE,
        )
        .with_unit(UnitRef {
            symbol: "degC".into(),
            transform_to_si: UnitTransform::new(
                RationalScale::ONE,
                RationalScale { numerator: 27315, denominator: 100 },
            ),
            semantic_id: None,
        });
        let kelvin = PhysicalType::with_kind(
            QuantityKind::Temperature,
            PhysicalDimension::TEMPERATURE,
        )
        .with_unit(UnitRef {
            symbol: "K".into(),
            transform_to_si: UnitTransform::IDENTITY,
            semantic_id: None,
        });

        match celsius.convert_value_to(0.0, &kelvin) {
            TypeJudgement::Valid(value) => assert!((value - 273.15).abs() < 1e-12),
            other => panic!("unexpected affine conversion judgment: {other:?}"),
        }
    }

    #[test]
    fn temperature_difference_fahrenheit_uses_scale_without_offset() {
        let delta_f = PhysicalType::with_kind(
            QuantityKind::TemperatureDifference,
            PhysicalDimension::TEMPERATURE,
        )
        .with_unit(UnitRef {
            symbol: "delta_degF".into(),
            transform_to_si: UnitTransform::new(
                RationalScale { numerator: 5, denominator: 9 },
                RationalScale { numerator: 0, denominator: 1 },
            ),
            semantic_id: None,
        });
        let delta_k = PhysicalType::with_kind(
            QuantityKind::TemperatureDifference,
            PhysicalDimension::TEMPERATURE,
        )
        .with_unit(UnitRef {
            symbol: "delta_K".into(),
            transform_to_si: UnitTransform::IDENTITY,
            semantic_id: None,
        });

        match delta_f.convert_value_to(18.0, &delta_k) {
            TypeJudgement::Valid(value) => assert!((value - 10.0).abs() < 1e-12),
            other => panic!("unexpected temperature-difference conversion: {other:?}"),
        }
    }

    #[test]
    fn unit_conversion_is_unknown_when_units_are_missing() {
        let metres = PhysicalType::with_kind(QuantityKind::Length, PhysicalDimension::LENGTH);
        let feet = metres.clone().with_unit(UnitRef {
            symbol: "ft".into(),
            transform_to_si: UnitTransform::new(
                RationalScale { numerator: 3048, denominator: 10000 },
                RationalScale { numerator: 0, denominator: 1 },
            ),
            semantic_id: None,
        });

        assert!(matches!(
            metres.convert_value_to(1.0, &feet),
            TypeJudgement::Unknown(_)
        ));
    }

    #[test]
    fn temperature_difference_rejects_affine_units() {
        let delta_c = PhysicalType::with_kind(
            QuantityKind::TemperatureDifference,
            PhysicalDimension::TEMPERATURE,
        )
        .with_unit(UnitRef {
            symbol: "degC".into(),
            transform_to_si: UnitTransform::new(
                RationalScale::ONE,
                RationalScale { numerator: 27315, denominator: 100 },
            ),
            semantic_id: None,
        });
        let delta_k = PhysicalType::with_kind(
            QuantityKind::TemperatureDifference,
            PhysicalDimension::TEMPERATURE,
        )
        .with_unit(UnitRef {
            symbol: "delta_K".into(),
            transform_to_si: UnitTransform::IDENTITY,
            semantic_id: None,
        });

        assert!(delta_c.validate().is_err());
        assert!(matches!(
            delta_c.judge_compatibility(&delta_k),
            TypeJudgement::Invalid(_)
        ));
    }

    #[test]
    fn numeric_compatibility_rejects_missing_unit_on_one_side() {
        let metres = PhysicalType::with_kind(QuantityKind::Length, PhysicalDimension::LENGTH);
        let metres_explicit = metres.clone().with_unit(UnitRef {
            symbol: "m".into(),
            transform_to_si: UnitTransform::IDENTITY,
            semantic_id: None,
        });

        assert!(matches!(
            metres.judge_compatibility(&metres_explicit),
            TypeJudgement::Valid(())
        ));
        assert!(matches!(
            metres.judge_numeric_compatibility(&metres_explicit),
            TypeJudgement::Unknown(_)
        ));
    }

    #[test]
    fn numeric_compatibility_rejects_scalar_domain_mismatch() {
        let real = PhysicalType::with_kind(QuantityKind::Length, PhysicalDimension::LENGTH);
        let mut complex = real.clone();
        complex.scalar = ScalarDomain::Complex;

        assert!(matches!(
            real.judge_numeric_compatibility(&complex),
            TypeJudgement::Invalid(_)
        ));
    }

    #[test]
    fn numeric_compatibility_stays_unknown_for_unknown_scalar_domain() {
        let real = PhysicalType::with_kind(QuantityKind::Length, PhysicalDimension::LENGTH);
        let mut unknown_scalar = real.clone();
        unknown_scalar.scalar = ScalarDomain::Unknown;

        assert!(matches!(
            real.judge_numeric_compatibility(&unknown_scalar),
            TypeJudgement::Unknown(_)
        ));
    }

    #[test]
    fn numeric_compatibility_accepts_explicit_convertible_units() {
        let metres = PhysicalType::with_kind(QuantityKind::Length, PhysicalDimension::LENGTH)
            .with_unit(UnitRef {
                symbol: "m".into(),
                transform_to_si: UnitTransform::IDENTITY,
                semantic_id: None,
            });
        let feet = PhysicalType::with_kind(QuantityKind::Length, PhysicalDimension::LENGTH)
            .with_unit(UnitRef {
                symbol: "ft".into(),
                transform_to_si: UnitTransform::new(
                    RationalScale { numerator: 3048, denominator: 10000 },
                    RationalScale { numerator: 0, denominator: 1 },
                ),
                semantic_id: None,
            });

        assert!(matches!(
            metres.judge_numeric_compatibility(&feet),
            TypeJudgement::Valid(())
        ));
    }

    #[test]
    fn known_quantity_kind_cannot_carry_arbitrary_dimension() {
        let malformed =
            PhysicalType::with_kind(QuantityKind::Energy, PhysicalDimension::LENGTH);
        assert!(malformed.validate().is_err());
        assert!(matches!(
            malformed.judge_compatibility(&PhysicalType::with_kind(
                QuantityKind::Energy,
                PhysicalDimension::LENGTH,
            )),
            TypeJudgement::Invalid(_)
        ));
    }

    #[test]
    fn negative_unit_scale_is_rejected() {
        let unit = UnitRef {
            symbol: "reverse_m".into(),
            transform_to_si: UnitTransform::new(
                RationalScale { numerator: -1, denominator: 1 },
                RationalScale { numerator: 0, denominator: 1 },
            ),
            semantic_id: None,
        };
        let typed =
            PhysicalType::with_kind(QuantityKind::Length, PhysicalDimension::LENGTH)
                .with_unit(unit);
        assert!(typed.validate().is_err());
    }

    #[test]
    fn malformed_unit_transform_is_rejected() {
        let unit = UnitRef {
            symbol: "broken".into(),
            transform_to_si: UnitTransform::new(
                RationalScale { numerator: 0, denominator: 1 },
                RationalScale { numerator: 0, denominator: 1 },
            ),
            semantic_id: None,
        };
        let typed =
            PhysicalType::with_kind(QuantityKind::Length, PhysicalDimension::LENGTH)
                .with_unit(unit);
        assert!(typed.validate().is_err());
    }

    #[test]
    fn semantic_identifiers_are_optional_but_nonempty() {
        assert!(SemanticIdentifier::new("", "Length").is_none());
        assert!(SemanticIdentifier::new("qudt", "").is_none());
        assert_eq!(
            SemanticIdentifier::new("qudt", "Length").unwrap().identifier,
            "Length"
        );
    }

    #[test]
    fn energy_and_torque_can_share_dimension_without_being_equal() {
        let energy = PhysicalType::with_kind(QuantityKind::Energy, PhysicalDimension::ENERGY);
        let torque = PhysicalType::with_kind(QuantityKind::Torque, PhysicalDimension::ENERGY);
        assert_ne!(energy.kind, torque.kind);
        assert_eq!(energy.dimension, torque.dimension);
    }
}
