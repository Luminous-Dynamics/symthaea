// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::{Deserialize, Serialize};

/// Area of Interest for Earth observation.
///
/// Uses lat/lon coordinates to define a bounding box or polygon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Aoi {
    pub name: String,
    pub center_lat: f64,
    pub center_lon: f64,
    /// Bounding box: [min_lat, min_lon, max_lat, max_lon]
    pub bbox: [f64; 4],
    /// Optional GeoJSON-style polygon (sequence of [lat, lon])
    pub polygon: Option<Vec<[f64; 2]>>,
}

impl Aoi {
    /// Calculate a simple hash/checksum of the AOI geometry for provenance.
    pub fn geometry_hash(&self) -> String {
        // Simple implementation for v0
        format!("{:?}-{:?}", self.bbox, self.polygon)
    }
}
