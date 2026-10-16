// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-geodesy
//!
//! Geodesy & navigation on a spherical Earth: great-circle distance and
//! bearing. A practical geospatial layer the workspace lacked.
//!
//! Pure std, zero dependencies, no symthaea-core link. Checked vs known
//! distances.
//!
//! For new callers that handle untrusted coordinate input, prefer GeoPoint,
//! which validates finite latitude/longitude values and their conventional
//! degree ranges at construction. The raw-float functions in sphere remain
//! available for compatibility and do not themselves validate input ranges.
//!
//! ## Example
//!
//! ~~~
//! use symthaea_geodesy::GeoPoint;
//! let london = GeoPoint::new(51.5074, -0.1278)?;
//! let paris = GeoPoint::new(48.8566, 2.3522)?;
//! assert!((london.distance_to(paris) - 343.5).abs() < 2.0);
//! # Ok::<(), symthaea_geodesy::GeoPointError>(())
//! ~~~

pub mod point;
pub mod sphere;

pub use point::{GeoPoint, GeoPointError};
pub use sphere::{EARTH_RADIUS_KM, haversine_distance, initial_bearing};
