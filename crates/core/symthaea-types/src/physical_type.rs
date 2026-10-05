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
    pub const PRESSURE: Self = Self { mass: 1, length: -1, time: -2, ..Self::DIMENSIONLESS };
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

/// Exact rational scale to SI base units. Affine offsets are deliberately not
/// modeled yet; absolute-vs-delta temperature semantics remain an adapter concern.
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
}

/// Named unit identity plus exact conversion to canonical SI semantics.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UnitRef {
    pub symbol: String,
    pub transform_to_si: UnitTransform,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_id: Option<SemanticIdentifier>,
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

    pub fn judge_compatibility(&self, other: &Self) -> TypeJudgement<()> {
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
    Pressure,
    Stress,
    Strain,
    Frequency,
    Custom,
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
        };
        assert_eq!(celsius.transform_to_si.offset.numerator, 27315);
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
