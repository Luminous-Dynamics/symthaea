// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Measurement types for multi-modal positioning.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasurementModality {
    Rssi,
    LoraToA,
    UwbToF,
    WifiRtt,
    Gps,
    Barometric,
    Imu,
    Visual,
    Acoustic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasurementProvenance {
    Local,
    Peer,
    Infrastructure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferenceFrame {
    Ecef,
    Enu,
    Body,
    Local,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeasurementValue {
    Range { distance_m: f64, sigma_m: f64 },
    Position { xyz: [f64; 3], sigma: [f64; 3] },
    Velocity { vxyz: [f64; 3], sigma: [f64; 3] },
    Bearing { azimuth_rad: f64, sigma_rad: f64 },
    Altitude { height_m: f64, sigma_m: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurement {
    pub modality: MeasurementModality,
    pub provenance: MeasurementProvenance,
    pub frame: ReferenceFrame,
    pub value: MeasurementValue,
    pub timestamp_us: u64,
    pub source_id: String,
}
