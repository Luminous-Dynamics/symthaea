// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Decentralized cooperative positioning library.
//!
//! Pure Rust implementation of positioning algorithms for peer-to-peer
//! localization without GPS/GNSS infrastructure. Body-agnostic: works
//! on Earth (WGS-84), Moon (IAU sphere), and Mars (IAU ellipsoid).
//!
//! # Architecture
//!
//! This library provides the mathematical primitives. It has zero
//! Holochain dependencies — the trust fabric and DHT storage live
//! in the zome layer above.
//!
//! # Modules
//!
//! - [`bodies`] — Celestial body trait + Earth/Moon/Mars implementations
//! - [`ranging`] — Sensor models: RSSI, LoRa ToA, UWB ToF, WiFi RTT
//! - [`trilateration`] — Core positioning: N ranges → position estimate
//! - [`coverage`] — GDOP/PDOP geometric dilution of precision
//! - [`kalman`] — Extended Kalman Filter for continuous tracking

pub mod bodies;
pub mod ranging;
pub mod trilateration;
pub mod coverage;
pub mod kalman;

pub use bodies::{CelestialBody, Earth, Moon, Mars};
pub use trilateration::{PositionEstimate, TrilaterationError, trilaterate_3d, trilaterate_2d};
pub use ranging::{RangeEstimate, RangingMethod};
pub use coverage::{gdop, pdop, CoveragePoint};
pub use kalman::{PositionFilter, FilterState};
