// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-chemosensation
//!
//! Shared foundations for artificial olfaction and gustation.
//!
//! This crate deliberately separates **physical observations** from learned
//! percepts. Hardware produces [`ChemicalObservation`] values with calibration,
//! environment, health, and provenance metadata. Higher layers may then derive
//! odor, taste, flavor, novelty, and semantic hypotheses without overwriting the
//! underlying measurement.
//!
//! The first tranche is hardware-independent and provides:
//! - typed gas/liquid observations;
//! - calibration provenance and sensor-health metadata;
//! - locality-preserving scalar HDC encoding for continuous chemistry values.

#![deny(unsafe_code)]

pub mod calibration;
pub mod encoding;
pub mod observation;

pub use calibration::{CalibrationId, CalibrationState, SensorHealth};
pub use encoding::ScalarHdcEncoder;
pub use observation::{
    ChemicalChannel, ChemicalModality, ChemicalObservation, EnvironmentReading, MeasurementUnit,
};
