// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-geodesy
//!
//! Geodesy & navigation on a spherical Earth: great-circle distance and
//! bearing. A practical geospatial layer the workspace lacked.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Checked vs known
//! distances.
//!
//! ## Example
//!
//! ```
//! use symthaea_geodesy::sphere::haversine_distance;
//! // London → Paris ≈ 343 km.
//! let d = haversine_distance(51.5074, -0.1278, 48.8566, 2.3522);
//! assert!((d - 343.5).abs() < 2.0);
//! ```

pub mod sphere;

pub use sphere::{EARTH_RADIUS_KM, haversine_distance, initial_bearing};
