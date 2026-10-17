// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Materials design, aging prediction, and HDC-based similarity search.
//!
//! Genesis Mission Challenge 9: Materials by Design.
//! Encodes material properties into 16,384-dimensional hypervectors,
//! predicts aging via O(1) CfC closed-form temporal jumps, and
//! provides constraint-filtered HDC similarity search.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod aging;
pub mod capacitor_screening;
pub mod compound_stability;
pub mod database;
pub mod dielectric;
pub mod encoder;
pub mod haptic_prober;
pub mod mining;
pub mod properties;
pub mod robust_capacitor;
pub mod strategic;

pub use aging::{AGING_HORIZON_LABELS, AGING_HORIZONS, AgingPrediction, MaterialAgingModel};
pub use capacitor_screening::{
    CapacitorConstraintViolation, CapacitorScreeningDisposition, CapacitorScreeningError,
    CapacitorScreeningReport, CapacitorScreeningRequirements, ScreeningLimitation,
    screen_capacitor_candidate,
};
pub use database::{MaterialDatabase, MaterialSearchResult};
pub use dielectric::{
    CapacitorGeometry, DielectricEvidenceKind, DielectricPropertyRecord, DielectricValidationError,
    BreakdownTestKind, IdealCapacitorMetrics, PropertyUncertainty, PropertyUncertaintySemantics,
    VACUUM_PERMITTIVITY_F_PER_M,
    calculate_ideal_capacitor_metrics,
};
pub use encoder::MaterialHdcEncoder;
pub use mining::{
    MINING_HORIZON_LABELS, MINING_HORIZONS, MiningFepAction, MiningFepAgent, MiningHdcEncoder,
    MiningPredictor, MiningReading,
};
pub use robust_capacitor::{
    BoundedCapacitorMetrics, RobustCapacitorConstraintViolation,
    RobustCapacitorScreeningDisposition, RobustCapacitorScreeningReport,
    RobustMetricError, RobustMetricLimitation, RobustScreeningError,
    RobustScreeningLimitation, ScalarBounds, calculate_bounded_capacitor_metrics,
    screen_capacitor_candidate_robust,
};
pub use properties::{MaterialCategory, MaterialProperty};
pub use strategic::{
    STRATEGIC_HORIZON_LABELS, STRATEGIC_HORIZONS, StrategicFepAction, StrategicFepAgent,
    StrategicHdcEncoder, StrategicPredictor, StrategicReading,
};
