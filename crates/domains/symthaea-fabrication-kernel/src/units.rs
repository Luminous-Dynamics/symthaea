// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit physical units used at fabrication trust boundaries.
//!
//! Geometry inside this crate is expressed in millimetres. Analytical mechanics
//! uses SI units. These wrappers make conversion sites visible and prevent a raw
//! mesh extent from being silently interpreted as metres.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Canonical unit for mesh vertices, slice coordinates, and machine motion.
pub const CANONICAL_GEOMETRY_UNIT: &str = "millimetre";

/// Invalid scalar supplied to a physical-unit wrapper.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnitError {
    /// NaN or infinity is never a valid physical quantity.
    NonFinite,
    /// A quantity that must be strictly positive was zero or negative.
    NotPositive,
}

impl fmt::Display for UnitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => write!(f, "physical quantity must be finite"),
            Self::NotPositive => write!(f, "physical quantity must be positive"),
        }
    }
}

impl std::error::Error for UnitError {}

macro_rules! finite_quantity {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, PartialOrd)]
        #[serde(transparent)]
        pub struct $name(f64);

        impl $name {
            /// Construct a finite quantity.
            pub fn new(value: f64) -> Result<Self, UnitError> {
                if value.is_finite() {
                    Ok(Self(value))
                } else {
                    Err(UnitError::NonFinite)
                }
            }

            /// Construct a finite, strictly positive quantity.
            pub fn positive(value: f64) -> Result<Self, UnitError> {
                let quantity = Self::new(value)?;
                if quantity.0 > 0.0 {
                    Ok(quantity)
                } else {
                    Err(UnitError::NotPositive)
                }
            }

            /// Return the scalar in this type's documented unit.
            pub const fn get(self) -> f64 {
                self.0
            }
        }
    };
}

finite_quantity!(Millimeters, "A length measured in millimetres.");
finite_quantity!(Meters, "A length measured in metres.");
finite_quantity!(Newtons, "A force measured in newtons.");
finite_quantity!(Pascals, "A pressure or stress measured in pascals.");

impl Millimeters {
    /// Convert millimetres to metres for SI analytical calculations.
    pub fn to_meters(self) -> Meters {
        // A finite value remains finite under this scale factor.
        Meters(self.0 * 1.0e-3)
    }
}

impl Meters {
    /// Convert metres to millimetres for geometry and machine coordinates.
    pub fn to_millimeters(self) -> Millimeters {
        Millimeters(self.0 * 1.0e3)
    }
}

impl From<Millimeters> for Meters {
    fn from(value: Millimeters) -> Self {
        value.to_meters()
    }
}

impl From<Meters> for Millimeters {
    fn from(value: Meters) -> Self {
        value.to_millimeters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn millimetres_convert_to_metres() {
        let length = Millimeters::positive(100.0).unwrap();
        assert!((length.to_meters().get() - 0.1).abs() < 1.0e-12);
    }

    #[test]
    fn metres_round_trip_to_millimetres() {
        let length = Meters::positive(0.025).unwrap();
        assert!((length.to_millimeters().get() - 25.0).abs() < 1.0e-12);
    }

    #[test]
    fn wrappers_reject_non_finite_values() {
        assert_eq!(Millimeters::new(f64::NAN), Err(UnitError::NonFinite));
        assert_eq!(Newtons::new(f64::INFINITY), Err(UnitError::NonFinite));
    }

    #[test]
    fn positive_constructor_rejects_zero() {
        assert_eq!(Meters::positive(0.0), Err(UnitError::NotPositive));
    }
}
