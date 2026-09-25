// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canonical unit identity for engineering evidence and cross-crate protocols.
//!
//! This crate intentionally implements a small **closed** unit vocabulary rather
//! than accepting arbitrary strings or claiming to be a complete UCUM parser.
//! Each admitted unit has a stable UCUM-compatible code, an explicit quantity
//! kind, and a deterministic conversion to one canonical SI representation.
//!
//! Human display labels are not evidence identities. `UnitCode` serializes as
//! its canonical machine code (for example `"Ohm"`, `"m3"`, `"Cel"`), not as
//! a Rust enum variant name.

#![deny(unsafe_code)]

use serde::{de::Error as _, Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

/// Semantic quantity kind. Equal physical dimensions do not automatically imply
/// semantic interchangeability; e.g. force factor remains its own engineering
/// quantity rather than being treated as any arbitrary `T.m` value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantityKind {
    ElectricalResistance,
    Inductance,
    ElectricPotential,
    ElectricCurrent,
    Power,
    ThermodynamicTemperature,
    ReciprocalTemperature,
    Length,
    Area,
    Volume,
    Mass,
    MechanicalCompliance,
    Stiffness,
    MechanicalResistance,
    Velocity,
    MassDensity,
    ForceFactor,
    Frequency,
    Time,
    Pressure,
}

/// Closed unit set admitted by ENG-UNITS-001 v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnitCode {
    Ohm,
    Milliohm,
    Henry,
    Millihenry,
    Microhenry,
    Volt,
    Millivolt,
    Ampere,
    Milliampere,
    Watt,
    Milliwatt,
    Kilowatt,
    Kelvin,
    Celsius,
    PerKelvin,
    Meter,
    Centimeter,
    Millimeter,
    SquareMeter,
    SquareCentimeter,
    SquareMillimeter,
    CubicMeter,
    Liter,
    Milliliter,
    Kilogram,
    Gram,
    MeterPerNewton,
    NewtonPerMeter,
    NewtonSecondPerMeter,
    MeterPerSecond,
    KilogramPerCubicMeter,
    TeslaMeter,
    Hertz,
    Kilohertz,
    Second,
    Millisecond,
    Pascal,
    Kilopascal,
}

impl UnitCode {
    pub const fn quantity_kind(self) -> QuantityKind {
        match self {
            Self::Ohm | Self::Milliohm => QuantityKind::ElectricalResistance,
            Self::Henry | Self::Millihenry | Self::Microhenry => QuantityKind::Inductance,
            Self::Volt | Self::Millivolt => QuantityKind::ElectricPotential,
            Self::Ampere | Self::Milliampere => QuantityKind::ElectricCurrent,
            Self::Watt | Self::Milliwatt | Self::Kilowatt => QuantityKind::Power,
            Self::Kelvin | Self::Celsius => QuantityKind::ThermodynamicTemperature,
            Self::PerKelvin => QuantityKind::ReciprocalTemperature,
            Self::Meter | Self::Centimeter | Self::Millimeter => QuantityKind::Length,
            Self::SquareMeter | Self::SquareCentimeter | Self::SquareMillimeter => {
                QuantityKind::Area
            }
            Self::CubicMeter | Self::Liter | Self::Milliliter => QuantityKind::Volume,
            Self::Kilogram | Self::Gram => QuantityKind::Mass,
            Self::MeterPerNewton => QuantityKind::MechanicalCompliance,
            Self::NewtonPerMeter => QuantityKind::Stiffness,
            Self::NewtonSecondPerMeter => QuantityKind::MechanicalResistance,
            Self::MeterPerSecond => QuantityKind::Velocity,
            Self::KilogramPerCubicMeter => QuantityKind::MassDensity,
            Self::TeslaMeter => QuantityKind::ForceFactor,
            Self::Hertz | Self::Kilohertz => QuantityKind::Frequency,
            Self::Second | Self::Millisecond => QuantityKind::Time,
            Self::Pascal | Self::Kilopascal => QuantityKind::Pressure,
        }
    }

    /// Canonical case-sensitive UCUM-compatible machine code.
    pub const fn ucum_code(self) -> &'static str {
        match self {
            Self::Ohm => "Ohm",
            Self::Milliohm => "mOhm",
            Self::Henry => "H",
            Self::Millihenry => "mH",
            Self::Microhenry => "uH",
            Self::Volt => "V",
            Self::Millivolt => "mV",
            Self::Ampere => "A",
            Self::Milliampere => "mA",
            Self::Watt => "W",
            Self::Milliwatt => "mW",
            Self::Kilowatt => "kW",
            Self::Kelvin => "K",
            Self::Celsius => "Cel",
            Self::PerKelvin => "K-1",
            Self::Meter => "m",
            Self::Centimeter => "cm",
            Self::Millimeter => "mm",
            Self::SquareMeter => "m2",
            Self::SquareCentimeter => "cm2",
            Self::SquareMillimeter => "mm2",
            Self::CubicMeter => "m3",
            Self::Liter => "L",
            Self::Milliliter => "mL",
            Self::Kilogram => "kg",
            Self::Gram => "g",
            Self::MeterPerNewton => "m/N",
            Self::NewtonPerMeter => "N/m",
            Self::NewtonSecondPerMeter => "N.s/m",
            Self::MeterPerSecond => "m/s",
            Self::KilogramPerCubicMeter => "kg/m3",
            Self::TeslaMeter => "T.m",
            Self::Hertz => "Hz",
            Self::Kilohertz => "kHz",
            Self::Second => "s",
            Self::Millisecond => "ms",
            Self::Pascal => "Pa",
            Self::Kilopascal => "kPa",
        }
    }

    /// Parse only the exact case-sensitive codes admitted by this contract.
    /// This is not a general UCUM parser and intentionally accepts no aliases.
    pub fn from_ucum_code(code: &str) -> Option<Self> {
        Some(match code {
            "Ohm" => Self::Ohm,
            "mOhm" => Self::Milliohm,
            "H" => Self::Henry,
            "mH" => Self::Millihenry,
            "uH" => Self::Microhenry,
            "V" => Self::Volt,
            "mV" => Self::Millivolt,
            "A" => Self::Ampere,
            "mA" => Self::Milliampere,
            "W" => Self::Watt,
            "mW" => Self::Milliwatt,
            "kW" => Self::Kilowatt,
            "K" => Self::Kelvin,
            "Cel" => Self::Celsius,
            "K-1" => Self::PerKelvin,
            "m" => Self::Meter,
            "cm" => Self::Centimeter,
            "mm" => Self::Millimeter,
            "m2" => Self::SquareMeter,
            "cm2" => Self::SquareCentimeter,
            "mm2" => Self::SquareMillimeter,
            "m3" => Self::CubicMeter,
            "L" => Self::Liter,
            "mL" => Self::Milliliter,
            "kg" => Self::Kilogram,
            "g" => Self::Gram,
            "m/N" => Self::MeterPerNewton,
            "N/m" => Self::NewtonPerMeter,
            "N.s/m" => Self::NewtonSecondPerMeter,
            "m/s" => Self::MeterPerSecond,
            "kg/m3" => Self::KilogramPerCubicMeter,
            "T.m" => Self::TeslaMeter,
            "Hz" => Self::Hertz,
            "kHz" => Self::Kilohertz,
            "s" => Self::Second,
            "ms" => Self::Millisecond,
            "Pa" => Self::Pascal,
            "kPa" => Self::Kilopascal,
            _ => return None,
        })
    }

    pub const fn canonical_unit(self) -> UnitCode {
        match self.quantity_kind() {
            QuantityKind::ElectricalResistance => Self::Ohm,
            QuantityKind::Inductance => Self::Henry,
            QuantityKind::ElectricPotential => Self::Volt,
            QuantityKind::ElectricCurrent => Self::Ampere,
            QuantityKind::Power => Self::Watt,
            QuantityKind::ThermodynamicTemperature => Self::Kelvin,
            QuantityKind::ReciprocalTemperature => Self::PerKelvin,
            QuantityKind::Length => Self::Meter,
            QuantityKind::Area => Self::SquareMeter,
            QuantityKind::Volume => Self::CubicMeter,
            QuantityKind::Mass => Self::Kilogram,
            QuantityKind::MechanicalCompliance => Self::MeterPerNewton,
            QuantityKind::Stiffness => Self::NewtonPerMeter,
            QuantityKind::MechanicalResistance => Self::NewtonSecondPerMeter,
            QuantityKind::Velocity => Self::MeterPerSecond,
            QuantityKind::MassDensity => Self::KilogramPerCubicMeter,
            QuantityKind::ForceFactor => Self::TeslaMeter,
            QuantityKind::Frequency => Self::Hertz,
            QuantityKind::Time => Self::Second,
            QuantityKind::Pressure => Self::Pascal,
        }
    }

    const fn scale_to_canonical(self) -> f64 {
        match self {
            Self::Milliohm
            | Self::Millihenry
            | Self::Millivolt
            | Self::Milliampere
            | Self::Milliwatt
            | Self::Millimeter
            | Self::Millisecond => 1.0e-3,
            Self::Microhenry => 1.0e-6,
            Self::Kilowatt | Self::Kilohertz | Self::Kilopascal => 1.0e3,
            Self::Centimeter => 1.0e-2,
            Self::SquareCentimeter => 1.0e-4,
            Self::SquareMillimeter => 1.0e-6,
            Self::Liter => 1.0e-3,
            Self::Milliliter => 1.0e-6,
            Self::Gram => 1.0e-3,
            _ => 1.0,
        }
    }

    const fn offset_to_canonical(self) -> f64 {
        match self {
            Self::Celsius => 273.15,
            _ => 0.0,
        }
    }

    pub fn to_canonical_value(self, value: f64) -> Result<f64, UnitError> {
        if !value.is_finite() {
            return Err(UnitError::NonFiniteValue(value));
        }
        let canonical = value * self.scale_to_canonical() + self.offset_to_canonical();
        if !canonical.is_finite() {
            return Err(UnitError::NonFiniteValue(canonical));
        }
        if self.quantity_kind() == QuantityKind::ThermodynamicTemperature && canonical < 0.0 {
            return Err(UnitError::BelowAbsoluteZero { value, unit: self });
        }
        Ok(canonical)
    }

    pub fn from_canonical_value(self, canonical: f64) -> Result<f64, UnitError> {
        if !canonical.is_finite() {
            return Err(UnitError::NonFiniteValue(canonical));
        }
        if self.quantity_kind() == QuantityKind::ThermodynamicTemperature && canonical < 0.0 {
            return Err(UnitError::BelowAbsoluteZero {
                value: canonical,
                unit: self.canonical_unit(),
            });
        }
        Ok((canonical - self.offset_to_canonical()) / self.scale_to_canonical())
    }
}

impl Serialize for UnitCode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.ucum_code())
    }
}

impl<'de> Deserialize<'de> for UnitCode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let code = String::deserialize(deserializer)?;
        Self::from_ucum_code(&code)
            .ok_or_else(|| D::Error::custom(format!("unsupported canonical engineering unit {code:?}")))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuantityValue {
    pub value: f64,
    pub unit: UnitCode,
}

impl QuantityValue {
    pub fn new(value: f64, unit: UnitCode) -> Result<Self, UnitError> {
        unit.to_canonical_value(value)?;
        Ok(Self { value, unit })
    }

    pub fn quantity_kind(self) -> QuantityKind {
        self.unit.quantity_kind()
    }

    pub fn canonicalize(self) -> Result<CanonicalQuantity, UnitError> {
        Ok(CanonicalQuantity {
            value: self.unit.to_canonical_value(self.value)?,
            unit: self.unit.canonical_unit(),
        })
    }

    pub fn convert_to(self, target: UnitCode) -> Result<Self, UnitError> {
        if self.quantity_kind() != target.quantity_kind() {
            return Err(UnitError::IncompatibleQuantityKinds {
                source: self.quantity_kind(),
                target: target.quantity_kind(),
            });
        }
        let canonical = self.unit.to_canonical_value(self.value)?;
        Self::new(target.from_canonical_value(canonical)?, target)
    }
}

/// Quantity normalized to this contract's canonical SI representation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CanonicalQuantity {
    pub value: f64,
    pub unit: UnitCode,
}

impl CanonicalQuantity {
    pub fn validate(self) -> Result<(), UnitError> {
        if self.unit != self.unit.canonical_unit() {
            return Err(UnitError::NonCanonicalUnit(self.unit));
        }
        self.unit.to_canonical_value(self.value)?;
        Ok(())
    }

    pub fn quantity_kind(self) -> QuantityKind {
        self.unit.quantity_kind()
    }
}

#[derive(Debug, Error, Clone, Copy, PartialEq)]
pub enum UnitError {
    #[error("quantity value must be finite, got {0}")]
    NonFiniteValue(f64),
    #[error("temperature {value} {unit:?} is below absolute zero")]
    BelowAbsoluteZero { value: f64, unit: UnitCode },
    #[error("incompatible quantity kinds: {source:?} vs {target:?}")]
    IncompatibleQuantityKinds {
        source: QuantityKind,
        target: QuantityKind,
    },
    #[error("unit {0:?} is not canonical for its quantity kind")]
    NonCanonicalUnit(UnitCode),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() <= 1.0e-12 * a.abs().max(b.abs()).max(1.0));
    }

    #[test]
    fn canonical_machine_codes_are_stable() {
        assert_eq!(UnitCode::Ohm.ucum_code(), "Ohm");
        assert_eq!(UnitCode::Hertz.ucum_code(), "Hz");
        assert_eq!(UnitCode::Pascal.ucum_code(), "Pa");
        assert_eq!(UnitCode::Celsius.ucum_code(), "Cel");
        assert_eq!(UnitCode::CubicMeter.ucum_code(), "m3");
        assert_eq!(UnitCode::TeslaMeter.ucum_code(), "T.m");
    }

    #[test]
    fn wire_serialization_uses_canonical_machine_code() {
        assert_eq!(serde_json::to_string(&UnitCode::Ohm).unwrap(), "\"Ohm\"");
        assert_eq!(serde_json::to_string(&UnitCode::CubicMeter).unwrap(), "\"m3\"");
        assert_eq!(
            serde_json::from_str::<UnitCode>("\"Cel\"").unwrap(),
            UnitCode::Celsius
        );
        assert!(serde_json::from_str::<UnitCode>("\"ohm\"").is_err());
        assert!(serde_json::from_str::<UnitCode>("\"cubic_meter\"").is_err());
    }

    #[test]
    fn prefixes_convert_to_canonical_si() {
        close(
            QuantityValue::new(250.0, UnitCode::Millihenry)
                .unwrap()
                .canonicalize()
                .unwrap()
                .value,
            0.25,
        );
        close(
            QuantityValue::new(12.0, UnitCode::Liter)
                .unwrap()
                .canonicalize()
                .unwrap()
                .value,
            0.012,
        );
        close(
            QuantityValue::new(2500.0, UnitCode::SquareMillimeter)
                .unwrap()
                .canonicalize()
                .unwrap()
                .value,
            0.0025,
        );
    }

    #[test]
    fn celsius_uses_offset_not_scale_only() {
        let zero_c = QuantityValue::new(0.0, UnitCode::Celsius).unwrap();
        let kelvin = zero_c.convert_to(UnitCode::Kelvin).unwrap();
        close(kelvin.value, 273.15);
        close(
            QuantityValue::new(293.15, UnitCode::Kelvin)
                .unwrap()
                .convert_to(UnitCode::Celsius)
                .unwrap()
                .value,
            20.0,
        );
    }

    #[test]
    fn below_absolute_zero_fails_closed() {
        assert!(matches!(
            QuantityValue::new(-274.0, UnitCode::Celsius),
            Err(UnitError::BelowAbsoluteZero { .. })
        ));
        assert!(matches!(
            QuantityValue::new(-1.0, UnitCode::Kelvin),
            Err(UnitError::BelowAbsoluteZero { .. })
        ));
    }

    #[test]
    fn dimensionally_incompatible_conversion_is_rejected() {
        assert_eq!(
            QuantityValue::new(1.0, UnitCode::Meter)
                .unwrap()
                .convert_to(UnitCode::Second),
            Err(UnitError::IncompatibleQuantityKinds {
                source: QuantityKind::Length,
                target: QuantityKind::Time,
            })
        );
    }

    #[test]
    fn semantic_force_factor_kind_is_preserved() {
        let force_factor = QuantityValue::new(5.0, UnitCode::TeslaMeter).unwrap();
        assert_eq!(force_factor.quantity_kind(), QuantityKind::ForceFactor);
        assert_eq!(force_factor.unit.ucum_code(), "T.m");
    }

    #[test]
    fn canonical_quantity_rejects_prefixed_storage_unit() {
        assert_eq!(
            CanonicalQuantity {
                value: 1.0,
                unit: UnitCode::Millimeter,
            }
            .validate(),
            Err(UnitError::NonCanonicalUnit(UnitCode::Millimeter))
        );
    }

    #[test]
    fn round_trip_conversion_is_stable() {
        let source = QuantityValue::new(1250.0, UnitCode::Millivolt).unwrap();
        let volts = source.convert_to(UnitCode::Volt).unwrap();
        let restored = volts.convert_to(UnitCode::Millivolt).unwrap();
        close(restored.value, source.value);
    }
}
