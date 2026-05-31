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
pub mod compound_stability;
pub mod database;
pub mod encoder;
pub mod haptic_prober;
pub mod mining;
pub mod properties;
pub mod strategic;

pub use aging::{AGING_HORIZON_LABELS, AGING_HORIZONS, AgingPrediction, MaterialAgingModel};
pub use database::{MaterialDatabase, MaterialSearchResult};
pub use encoder::MaterialHdcEncoder;
pub use mining::{
    MINING_HORIZON_LABELS, MINING_HORIZONS, MiningFepAction, MiningFepAgent, MiningHdcEncoder,
    MiningPredictor, MiningReading,
};
pub use properties::{MaterialCategory, MaterialProperty};
pub use strategic::{
    STRATEGIC_HORIZON_LABELS, STRATEGIC_HORIZONS, StrategicFepAction, StrategicFepAgent,
    StrategicHdcEncoder, StrategicPredictor, StrategicReading,
};
