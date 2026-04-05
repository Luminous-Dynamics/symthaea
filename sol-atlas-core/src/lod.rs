// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Spherical Level of Detail (LOD) clustering for globe markers.
//!
//! Groups markers into spatial cells using a simple latitude/longitude grid.
//! At far zoom, cells render as single heat blobs. At close zoom, individual
//! markers are revealed.

/// LOD level based on camera distance from globe center.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LodLevel {
    /// Fully zoomed out (distance > 6.0) — show heat blobs only.
    Orbit,
    /// Mid distance (3.0-6.0) — show top markers per cell + routing lines.
    Atmosphere,
    /// Close (< 3.0) — show all markers in view.
    Surface,
}

impl LodLevel {
    pub fn from_camera_distance(distance: f32) -> Self {
        if distance > 3.8 {
            Self::Orbit      // far zoom — heat blobs only
        } else if distance > 2.8 {
            Self::Atmosphere  // mid — nothing (clean transition gap)
        } else {
            Self::Surface     // close — individual markers only
        }
    }
}

/// A spatial cell on the globe surface, grouping nearby markers.
#[derive(Debug, Clone)]
pub struct SpatialCell {
    /// Center latitude of this cell.
    pub center_lat: f64,
    /// Center longitude of this cell.
    pub center_lon: f64,
    /// Number of markers in this cell.
    pub count: usize,
    /// Sum of marker "importance" values (e.g., capacity, reserves).
    pub total_importance: f64,
    /// Average color RGB of markers in this cell.
    pub avg_color: [f32; 3],
}

/// Cluster a set of (lat, lon, importance, color) markers into spatial cells.
/// Uses a simple lat/lon grid with `lat_cells` x `lon_cells` divisions.
pub fn cluster_markers(
    markers: &[(f64, f64, f64, [f32; 3])],
    lat_cells: usize,
    lon_cells: usize,
) -> Vec<SpatialCell> {
    let lat_step = 180.0 / lat_cells as f64;
    let lon_step = 360.0 / lon_cells as f64;

    let mut cells: Vec<Vec<(f64, f64, f64, [f32; 3])>> =
        vec![Vec::new(); lat_cells * lon_cells];

    for &(lat, lon, importance, color) in markers {
        let lat_idx = ((lat + 90.0) / lat_step).floor() as usize;
        let lon_idx = ((lon + 180.0) / lon_step).floor() as usize;
        let lat_idx = lat_idx.min(lat_cells - 1);
        let lon_idx = lon_idx.min(lon_cells - 1);
        let cell_idx = lat_idx * lon_cells + lon_idx;
        cells[cell_idx].push((lat, lon, importance, color));
    }

    cells
        .into_iter()
        .enumerate()
        .filter(|(_, cell)| !cell.is_empty())
        .map(|(idx, cell)| {
            let lat_idx = idx / lon_cells;
            let lon_idx = idx % lon_cells;
            let center_lat = -90.0 + (lat_idx as f64 + 0.5) * lat_step;
            let center_lon = -180.0 + (lon_idx as f64 + 0.5) * lon_step;
            let count = cell.len();
            let total_importance: f64 = cell.iter().map(|m| m.2).sum();
            let avg_color = [
                cell.iter().map(|m| m.3[0]).sum::<f32>() / count as f32,
                cell.iter().map(|m| m.3[1]).sum::<f32>() / count as f32,
                cell.iter().map(|m| m.3[2]).sum::<f32>() / count as f32,
            ];
            SpatialCell {
                center_lat,
                center_lon,
                count,
                total_importance,
                avg_color,
            }
        })
        .collect()
}

/// Size of a heat blob based on marker count in the cell.
pub fn heat_blob_size(count: usize) -> f32 {
    let size = (count as f32).sqrt() * 0.008;
    size.clamp(0.008, 0.030) // smaller — blobs are background, data markers are the hero
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lod_levels() {
        assert_eq!(LodLevel::from_camera_distance(4.2), LodLevel::Orbit); // default zoom
        assert_eq!(LodLevel::from_camera_distance(3.5), LodLevel::Atmosphere); // transition gap
        assert_eq!(LodLevel::from_camera_distance(2.5), LodLevel::Surface);
    }

    #[test]
    fn test_cluster_basic() {
        let markers = vec![
            (51.5, -0.1, 100.0, [1.0, 0.0, 0.0]),  // London
            (48.9, 2.3, 200.0, [0.0, 1.0, 0.0]),    // Paris
            (-33.9, 18.4, 50.0, [0.0, 0.0, 1.0]),   // Cape Town
        ];
        let cells = cluster_markers(&markers, 6, 12);
        // London and Paris should be in the same cell (close together in a coarse grid)
        assert!(cells.len() >= 2); // at least 2 cells (Europe + SA)
    }

    #[test]
    fn test_cluster_empty() {
        let cells = cluster_markers(&[], 6, 12);
        assert!(cells.is_empty());
    }

    #[test]
    fn test_heat_blob_size_scales() {
        assert!(heat_blob_size(10) > heat_blob_size(1));
        assert!(heat_blob_size(100) <= 0.030);
    }
}
