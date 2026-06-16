// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Minimal stub for `navigation_estimator`.
//!
//! `pub mod navigation_estimator;` + `pub use navigation_estimator::{…}` in
//! `lib.rs` were added in commit `602ad76c417` (2026-04-11) without the
//! corresponding source file, breaking every `cargo check/test -p
//! symthaea-helicopter` invocation since. This stub provides the two
//! re-exported symbols so the crate builds; the actual estimator logic
//! can replace these types when the in-flight design lands.

/// Navigation state estimate (position + velocity + covariance).
///
/// Placeholder — expand when the estimator implementation lands.
#[derive(Debug, Clone, Default)]
pub struct HelicopterNavigationEstimate {
    /// Position in world frame (x, y, z), meters.
    pub position: [f64; 3],
    /// Velocity in world frame (vx, vy, vz), m/s.
    pub velocity: [f64; 3],
    /// Position variance (trace of 3×3 covariance), m².
    pub position_variance: f64,
}

/// Placeholder estimator.
#[derive(Debug, Clone, Default)]
pub struct HelicopterNavigationEstimator;

impl HelicopterNavigationEstimator {
    pub fn new() -> Self {
        Self
    }

    pub fn estimate(&self) -> HelicopterNavigationEstimate {
        HelicopterNavigationEstimate::default()
    }
}
