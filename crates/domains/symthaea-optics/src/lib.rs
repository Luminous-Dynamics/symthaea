// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-optics
//!
//! Geometric optics for Symthaea, completing the applied classical-physics layer
//! (the core crates have Maxwell/FDTD, but no ray optics). Angles in radians.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Checked vs textbook.
//!
//! ## Scope
//!
//! - Thin-lens/mirror imaging (image distance, magnification).
//! - Snell's law refraction + total internal reflection (critical angle).
//! - Diffraction-grating orders.
//!
//! ## Example
//!
//! ```
//! use symthaea_optics::geometric::{image_distance, refraction_angle};
//! assert!((image_distance(10.0, 30.0) - 15.0).abs() < 1e-9);
//! let t2 = refraction_angle(1.0, 30f64.to_radians(), 1.33).unwrap();
//! assert!((t2.to_degrees() - 22.08).abs() < 0.02);
//! ```

pub mod geometric;

pub use geometric::{
    critical_angle, grating_angle, image_distance, magnification, refraction_angle,
};
