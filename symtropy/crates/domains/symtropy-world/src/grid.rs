// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stable body-cell identifiers for Earth, Solar System bodies, and procedural worlds.
//!
//! The grid layer is an identity and streaming contract. Earth can use H3 cells,
//! mapped from source datasets at ingest time. Other bodies use body-local cells
//! so Mars, Luna, Ceres, Europa, and extrasolar planets do not inherit Earth-only
//! geodesy assumptions.

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BodyId(String);

impl BodyId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn earth() -> Self {
        Self::new("sol:earth")
    }

    pub fn luna() -> Self {
        Self::new("sol:luna")
    }

    pub fn mars() -> Self {
        Self::new("sol:mars")
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for BodyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GridSystem {
    EarthH3,
    BodyIcosahedral,
    ProceduralIcosphere,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HexCellId {
    pub body: BodyId,
    pub grid_system: GridSystem,
    pub resolution: u8,
    pub index: String,
}

impl HexCellId {
    pub fn new(
        body: BodyId,
        grid_system: GridSystem,
        resolution: u8,
        index: impl Into<String>,
    ) -> Self {
        Self {
            body,
            grid_system,
            resolution,
            index: index.into(),
        }
    }

    pub fn earth_h3(resolution: u8, h3_index: impl Into<String>) -> Self {
        Self::new(BodyId::earth(), GridSystem::EarthH3, resolution, h3_index)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BiomeKind {
    Unknown,
    Urban,
    Wetland,
    Forest,
    Grassland,
    Desert,
    Ice,
    Ocean,
    LunarRegolith,
    MartianRegolith,
    Procedural,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HydrologyState {
    pub surface_water_m: f32,
    pub groundwater_m: f32,
    pub flow_accumulation: f32,
    pub salinity: f32,
}

impl Default for HydrologyState {
    fn default() -> Self {
        Self {
            surface_water_m: 0.0,
            groundwater_m: 0.0,
            flow_accumulation: 0.0,
            salinity: 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlanetCell {
    pub id: HexCellId,
    pub center_lat_deg: f64,
    pub center_lon_deg: f64,
    pub area_m2: f64,
    pub elevation_m: f32,
    pub slope: f32,
    pub biome: BiomeKind,
    pub hydrology: HydrologyState,
    pub temperature_k: f32,
    pub atmosphere_pressure_pa: f32,
    pub resource_tags: Vec<String>,
    pub history_tags: Vec<String>,
}

impl PlanetCell {
    pub fn new(id: HexCellId, center_lat_deg: f64, center_lon_deg: f64) -> Self {
        Self {
            id,
            center_lat_deg,
            center_lon_deg,
            area_m2: 0.0,
            elevation_m: 0.0,
            slope: 0.0,
            biome: BiomeKind::Unknown,
            hydrology: HydrologyState::default(),
            temperature_k: 288.15,
            atmosphere_pressure_pa: 101_325.0,
            resource_tags: Vec::new(),
            history_tags: Vec::new(),
        }
    }
}

pub trait BodyHexGrid {
    fn cell_at_lat_lon(&self, lat_deg: f64, lon_deg: f64, resolution: u8) -> HexCellId;
    fn parent(&self, cell: &HexCellId, parent_resolution: u8) -> Option<HexCellId>;
    fn neighbors(&self, cell: &HexCellId) -> Vec<HexCellId>;
}

#[derive(Clone, Debug)]
pub struct EarthH3CellRef {
    pub id: HexCellId,
}

impl EarthH3CellRef {
    pub fn from_h3_index(resolution: u8, h3_index: impl Into<String>) -> Self {
        Self {
            id: HexCellId::earth_h3(resolution, h3_index),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ProceduralBodyGrid {
    pub body: BodyId,
    pub grid_system: GridSystem,
    pub seed: u64,
}

impl ProceduralBodyGrid {
    pub fn new(body: BodyId, seed: u64) -> Self {
        Self {
            body,
            grid_system: GridSystem::ProceduralIcosphere,
            seed,
        }
    }

    pub fn body_icosahedral(body: BodyId, seed: u64) -> Self {
        Self {
            body,
            grid_system: GridSystem::BodyIcosahedral,
            seed,
        }
    }

    fn quantized_index(&self, lat_deg: f64, lon_deg: f64, resolution: u8) -> String {
        let scale = 1_u32 << resolution.min(20);
        let lat_bin = (((lat_deg.clamp(-90.0, 90.0) + 90.0) / 180.0) * scale as f64).floor();
        let lon = normalize_lon_deg(lon_deg);
        let lon_bin = (((lon + 180.0) / 360.0) * (scale as f64 * 2.0)).floor();
        format!(
            "seed{:016x}:r{resolution}:y{:x}:x{:x}",
            self.seed, lat_bin as u64, lon_bin as u64
        )
    }
}

impl BodyHexGrid for ProceduralBodyGrid {
    fn cell_at_lat_lon(&self, lat_deg: f64, lon_deg: f64, resolution: u8) -> HexCellId {
        HexCellId::new(
            self.body.clone(),
            self.grid_system,
            resolution,
            self.quantized_index(lat_deg, lon_deg, resolution),
        )
    }

    fn parent(&self, cell: &HexCellId, parent_resolution: u8) -> Option<HexCellId> {
        if cell.body != self.body || parent_resolution >= cell.resolution {
            return None;
        }
        let mut parts = cell.index.split(':');
        let seed = parts.next()?;
        let _resolution = parts.next()?;
        let y_hex = parts.next()?.strip_prefix('y')?;
        let x_hex = parts.next()?.strip_prefix('x')?;
        let y = u64::from_str_radix(y_hex, 16).ok()? >> (cell.resolution - parent_resolution);
        let x = u64::from_str_radix(x_hex, 16).ok()? >> (cell.resolution - parent_resolution);
        Some(HexCellId::new(
            self.body.clone(),
            self.grid_system,
            parent_resolution,
            format!("{seed}:r{parent_resolution}:y{y:x}:x{x:x}"),
        ))
    }

    fn neighbors(&self, cell: &HexCellId) -> Vec<HexCellId> {
        if cell.body != self.body {
            return Vec::new();
        }
        let mut parts = cell.index.split(':');
        let Some(seed) = parts.next() else {
            return Vec::new();
        };
        let _resolution = parts.next();
        let Some(y_hex) = parts.next().and_then(|part| part.strip_prefix('y')) else {
            return Vec::new();
        };
        let Some(x_hex) = parts.next().and_then(|part| part.strip_prefix('x')) else {
            return Vec::new();
        };
        let Some(y) = u64::from_str_radix(y_hex, 16).ok() else {
            return Vec::new();
        };
        let Some(x) = u64::from_str_radix(x_hex, 16).ok() else {
            return Vec::new();
        };

        [(0_i64, -1_i64), (0, 1), (-1, 0), (1, 0), (-1, 1), (1, -1)]
            .into_iter()
            .filter_map(|(dy, dx)| {
                let ny = y.checked_add_signed(dy)?;
                let nx = x.checked_add_signed(dx)?;
                Some(HexCellId::new(
                    self.body.clone(),
                    self.grid_system,
                    cell.resolution,
                    format!("{seed}:r{}:y{ny:x}:x{nx:x}", cell.resolution),
                ))
            })
            .collect()
    }
}

pub fn normalize_lon_deg(lon_deg: f64) -> f64 {
    let wrapped = (lon_deg + 180.0).rem_euclid(360.0) - 180.0;
    if wrapped == -180.0 { 180.0 } else { wrapped }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn earth_h3_cell_keeps_external_index_verbatim() {
        let cell = HexCellId::earth_h3(7, "872830828ffffff");
        assert_eq!(cell.body, BodyId::earth());
        assert_eq!(cell.grid_system, GridSystem::EarthH3);
        assert_eq!(cell.resolution, 7);
        assert_eq!(cell.index, "872830828ffffff");
    }

    #[test]
    fn procedural_cells_are_deterministic_and_body_scoped() {
        let grid = ProceduralBodyGrid::new(BodyId::new("fictional:aster-vale"), 42);
        let a = grid.cell_at_lat_lon(-33.9249, 18.4241, 8);
        let b = grid.cell_at_lat_lon(-33.9249, 18.4241, 8);
        assert_eq!(a, b);
        assert_eq!(a.body.as_str(), "fictional:aster-vale");
    }

    #[test]
    fn procedural_parent_reduces_resolution() {
        let grid = ProceduralBodyGrid::body_icosahedral(BodyId::mars(), 7);
        let cell = grid.cell_at_lat_lon(18.65, 226.2, 9);
        let parent = grid.parent(&cell, 5).expect("parent cell");
        assert_eq!(parent.body, BodyId::mars());
        assert_eq!(parent.grid_system, GridSystem::BodyIcosahedral);
        assert_eq!(parent.resolution, 5);
    }

    #[test]
    fn procedural_neighbors_return_six_body_scoped_cells() {
        let grid = ProceduralBodyGrid::new(BodyId::luna(), 99);
        let cell = grid.cell_at_lat_lon(0.0, 0.0, 4);
        let neighbors = grid.neighbors(&cell);
        assert_eq!(neighbors.len(), 6);
        assert!(
            neighbors
                .iter()
                .all(|neighbor| neighbor.body == BodyId::luna())
        );
    }

    #[test]
    fn longitude_normalization_is_stable() {
        assert_eq!(normalize_lon_deg(181.0), -179.0);
        assert_eq!(normalize_lon_deg(-181.0), 179.0);
        assert_eq!(normalize_lon_deg(540.0), 180.0);
    }
}
