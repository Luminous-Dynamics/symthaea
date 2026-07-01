// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Waste classification and decomposition prediction via HDC.
//!
//! Encodes waste material properties into 16,384-dimensional hypervectors,
//! predicts decomposition timelines via sigmoid-based decay models, detects
//! contamination through HDC outlier detection, and provides FEP-based
//! action selection for circular economy automation.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod circular_fep;
pub mod contamination_detector;
pub mod decomposition_predictor;
pub mod waste_encoder;

pub use circular_fep::{CircularAction, CircularFepAgent, CircularObservation, SafetyLevel};
pub use contamination_detector::{ContaminationDetector, ContaminationResult};
pub use decomposition_predictor::{
    DecompositionConditions, DecompositionPrediction, DecompositionPredictor,
};
pub use waste_encoder::{WasteCategory, WasteDatabase, WasteHdcEncoder, WasteProperties};
