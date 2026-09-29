// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Versioned MOLA MEGDR raster adapter.
//!
//! This adapter validates product metadata before any raster value can enter
//! the Mars tether geometry layer. It intentionally supports the simple
//! cylindrical, planetocentric, east-positive MEGDR grid only.
//!
//! The adapter is not a geodetic authenticator: a syntactically valid label
//! can still describe the wrong bytes. Callers should pin the product files
//! by content hash and retain the validated label/header bytes as provenance.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use crate::mars_tether::{
    TerrainProvenance, TerrainQuality, TerrainSample, TerrainVerticalDatum,
};

#[derive(Debug)]
pub enum MolaError {
    Io(io::Error),
    InvalidMetadata(String),
    Unsupported(String),
    OutOfBounds,
    MissingValue,
}

impl From<io::Error> for MolaError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MolaMegdrMetadata {
    pub product_id: String,
    pub product_version: String,
    pub resolution_pixels_per_degree: u32,
    pub lines: u32,
    pub samples: u32,
    pub record_bytes: u64,
    pub sample_bits: u16,
    pub sample_type: String,
    pub map_projection: String,
    pub latitude_type: String,
    pub longitude_direction: String,
    pub longitude_min_deg: f64,
    pub longitude_max_deg: f64,
    pub latitude_min_deg: f64,
    pub latitude_max_deg: f64,
    pub line_offset: u32,
    pub sample_offset: u32,
    pub line_projection_offset: f64,
    pub sample_projection_offset: f64,
    pub pixel_scale: f64,
    pub pixel_offset: f64,
    pub missing_value: Option<f64>,
    pub map_kind: char,
    pub tile_origin_lat_deg: f64,
    pub tile_origin_lon_deg: f64,
}

#[derive(Debug, Clone)]
pub struct MolaMegdrProduct {
    pub metadata: MolaMegdrMetadata,
    pub provenance: TerrainProvenance,
    img_path: std::path::PathBuf,
}

impl MolaMegdrProduct {
    /// Open and validate a MOLA MEGDR PDS label against the raster.
    ///
    /// The label is authoritative for the file layout only after these
    /// invariants are checked. The image itself is never modified.
    pub fn open(
        label_path: impl AsRef<Path>,
        img_path: impl AsRef<Path>,
        expected_product_id: &str,
        expected_revision: &str,
    ) -> Result<Self, MolaError> {
        let label_text = std::fs::read_to_string(label_path)?;
        let kv = parse_label(&label_text);
        let metadata = MolaMegdrMetadata::from_label(&kv, expected_product_id)?;
        if expected_revision.trim().is_empty() {
            return Err(MolaError::InvalidMetadata(
                "expected revision must be non-empty".into(),
            ));
        }
        validate_img_size(&metadata, img_path.as_ref())?;
        let provenance = TerrainProvenance {
            source_id: expected_product_id.to_string(),
            source_revision: expected_revision.to_string(),
            coordinate_reference: "IAU-2000 planetocentric latitude, east-positive longitude".into(),
        };
        Ok(Self {
            metadata,
            provenance,
            img_path: img_path.as_ref().to_path_buf(),
        })
    }

    /// Sample the nearest topography cell only when its companion counts map
    /// proves that the cell has at least one observation. No interpolation is
    /// performed and uncertainty must be supplied explicitly by the caller.
    pub fn sample_nearest_with_count(
        &self,
        counts: &Self,
        latitude_deg: f64,
        longitude_deg: f64,
        elevation_uncertainty_m: f64,
    ) -> Result<TerrainSample, MolaError> {
        self.validate_companion(counts)?;
        if !elevation_uncertainty_m.is_finite() || elevation_uncertainty_m < 0.0 {
            return Err(MolaError::InvalidMetadata(
                "elevation uncertainty must be finite and non-negative".into(),
            ));
        }
        if self.metadata.map_kind != 'T' {
            return Err(MolaError::InvalidMetadata(
                "sample source must be a topography map".into(),
            ));
        }
        let (line, sample) = self.metadata.cell_for(latitude_deg, longitude_deg)?;
        let count = counts.read_count(line, sample)?;
        if count == 0 {
            return Ok(self.missing_sample(latitude_deg, longitude_deg));
        }
        let value = self.read_i16(line, sample)? as f64;
        if self.metadata.missing_value.is_some_and(|m| value == m) {
            return Ok(self.missing_sample(latitude_deg, longitude_deg));
        }
        let elevation = value * self.metadata.pixel_scale + self.metadata.pixel_offset;
        if !elevation.is_finite() {
            return Err(MolaError::InvalidMetadata("decoded elevation is non-finite".into()));
        }
        Ok(TerrainSample {
            latitude_rad: latitude_deg.to_radians(),
            longitude_rad: normalize_lon(longitude_deg).to_radians(),
            elevation_m: Some(elevation),
            elevation_uncertainty_m: Some(elevation_uncertainty_m),
            vertical_datum: TerrainVerticalDatum::AreoidRelative,
            slope_rad: None,
            roughness_m: None,
            quality: TerrainQuality::Measured,
            provenance: self.provenance.clone(),
        })
    }

    fn missing_sample(&self, latitude_deg: f64, longitude_deg: f64) -> TerrainSample {
        TerrainSample {
            latitude_rad: latitude_deg.to_radians(),
            longitude_rad: normalize_lon(longitude_deg).to_radians(),
            elevation_m: None,
            elevation_uncertainty_m: None,
            vertical_datum: TerrainVerticalDatum::AreoidRelative,
            slope_rad: None,
            roughness_m: None,
            quality: TerrainQuality::Missing,
            provenance: self.provenance.clone(),
        }
    }

    fn validate_companion(&self, counts: &Self) -> Result<(), MolaError> {
        let a = &self.metadata;
        let b = &counts.metadata;
        if b.map_kind != 'C' {
            return Err(MolaError::InvalidMetadata(
                "companion product must be a counts map".into(),
            ));
        }
        if a.product_version != b.product_version
            || a.resolution_pixels_per_degree != b.resolution_pixels_per_degree
            || a.lines != b.lines
            || a.samples != b.samples
            || a.latitude_min_deg != b.latitude_min_deg
            || a.latitude_max_deg != b.latitude_max_deg
            || a.longitude_min_deg != b.longitude_min_deg
            || a.longitude_max_deg != b.longitude_max_deg
            || a.tile_origin_lat_deg != b.tile_origin_lat_deg
            || a.tile_origin_lon_deg != b.tile_origin_lon_deg
        {
            return Err(MolaError::InvalidMetadata(
                "topography/counts grids are not registration-identical".into(),
            ));
        }
        Ok(())
    }

    fn read_count(&self, line: u32, sample: u32) -> Result<u32, MolaError> {
        if line >= self.metadata.lines || sample >= self.metadata.samples {
            return Err(MolaError::OutOfBounds);
        }
        let byte_offset = u64::from(self.metadata.record_bytes)
            * (u64::from(line) + u64::from(self.metadata.line_offset))
            + u64::from(sample) * u64::from(self.metadata.sample_bits / 8)
            + u64::from(self.metadata.sample_offset);
        let mut file = File::open(&self.img_path)?;
        file.seek(SeekFrom::Start(byte_offset))?;
        match self.metadata.sample_bits {
            8 => {
                let mut b = [0u8; 1];
                file.read_exact(&mut b)?;
                Ok(u32::from(b[0]))
            }
            16 => {
                let mut b = [0u8; 2];
                file.read_exact(&mut b)?;
                Ok(u32::from(u16::from_be_bytes(b)))
            }
            _ => Err(MolaError::Unsupported(
                "counts must be an 8-bit or 16-bit integer".into(),
            )),
        }
    }

    fn read_i16(&self, line: u32, sample: u32) -> Result<i16, MolaError> {
        if line >= self.metadata.lines || sample >= self.metadata.samples {
            return Err(MolaError::OutOfBounds);
        }
        let byte_offset = u64::from(self.metadata.record_bytes)
            * (u64::from(line) + u64::from(self.metadata.line_offset))
            + u64::from(sample) * 2
            + u64::from(self.metadata.sample_offset);
        let mut file = File::open(&self.img_path)?;
        file.seek(SeekFrom::Start(byte_offset))?;
        let mut bytes = [0u8; 2];
        file.read_exact(&mut bytes)?;
        match self.metadata.sample_type.as_str() {
            "MSB_INTEGER" => Ok(i16::from_be_bytes(bytes)),
            "LSB_INTEGER" => Ok(i16::from_le_bytes(bytes)),
            other => Err(MolaError::Unsupported(format!(
                "sample type {other} is not a signed 16-bit integer"
            ))),
        }
    }
}

impl MolaMegdrMetadata {
    fn from_label(
        kv: &std::collections::BTreeMap<String, String>,
        expected_product_id: &str,
    ) -> Result<Self, MolaError> {
        let product_id = required(kv, "PRODUCT_ID")?;
        let product_version = required(kv, "PRODUCT_VERSION_ID")?;
        let resolution = parse_u32(kv, "MAP_RESOLUTION")?;
        let lines = parse_u32(kv, "LINES")?;
        let samples = parse_u32(kv, "LINE_SAMPLES")?;
        let record_bytes = parse_u64(kv, "RECORD_BYTES")?;
        let sample_bits = parse_u16(kv, "SAMPLE_BITS")?;
        let sample_type = required(kv, "SAMPLE_TYPE")?;
        let map_projection = required(kv, "MAP_PROJECTION_TYPE")?;
        let latitude_type = required(kv, "COORDINATE_SYSTEM_NAME")?;
        let longitude_direction = required(kv, "POSITIVE_LONGITUDE_DIRECTION")?;
        let longitude_min_deg = parse_f64(kv, "WESTERNMOST_LONGITUDE")?;
        let longitude_max_deg = parse_f64(kv, "EASTERNMOST_LONGITUDE")?;
        let latitude_min_deg = parse_f64(kv, "MINIMUM_LATITUDE")?;
        let latitude_max_deg = parse_f64(kv, "MAXIMUM_LATITUDE")?;
        let line_offset = parse_u32_default(kv, "LABEL_RECORDS", 0)?;
        let sample_offset = 0;
        let line_projection_offset =
            parse_f64_default(kv, "LINE_PROJECTION_OFFSET", latitude_max_deg * resolution as f64 + 0.5)?;
        let sample_projection_offset =
            parse_f64_default(kv, "SAMPLE_PROJECTION_OFFSET", longitude_min_deg * resolution as f64 + 0.5)?;
        let pixel_scale = parse_f64_default(kv, "SCALING_FACTOR", 1.0)?;
        let pixel_offset = parse_f64_default(kv, "OFFSET", 0.0)?;
        let missing_value = kv.get("MISSING_CONSTANT").and_then(|v| parse_number(v).ok());
        let map_kind = required(kv, "MAP_TYPE")?.chars().next().ok_or_else(|| {
            MolaError::InvalidMetadata("MAP_TYPE is empty".into())
        })?;
        if !matches!(map_kind, 'T' | 'C' | 'R' | 'A') {
            return Err(MolaError::InvalidMetadata("unsupported MEGDR map type".into()));
        }
        let tile_origin_lat_deg = parse_f64_default(kv, "TILE_ORIGIN_LATITUDE", latitude_max_deg)?;
        let tile_origin_lon_deg = parse_f64_default(kv, "TILE_ORIGIN_LONGITUDE", longitude_min_deg)?;

        if product_id != expected_product_id {
            return Err(MolaError::InvalidMetadata(format!(
                "product id {product_id} does not match pinned id {expected_product_id}"
            )));
        }
        if product_version != "2.0" {
            return Err(MolaError::InvalidMetadata(
                "only final MEGDR PRODUCT_VERSION_ID=2.0 is accepted".into(),
            ));
        }
        if !matches!(resolution, 4 | 16 | 32 | 64 | 128) {
            return Err(MolaError::Unsupported(format!(
                "unsupported MEGDR resolution: {resolution} pixels/degree"
            )));
        }
        if lines == 0 || samples == 0 || record_bytes < 2 {
            return Err(MolaError::InvalidMetadata("invalid raster dimensions".into()));
        }
        if !matches!(sample_type.as_str(), "MSB_INTEGER" | "LSB_INTEGER") {
            return Err(MolaError::Unsupported("unsupported MEGDR integer sample type".into()));
        }
        if map_kind == 'C' && resolution >= 64 {
            if sample_bits != 8 {
                return Err(MolaError::Unsupported(
                    "64/128 ppd MEGDR counts must be 8-bit unsigned integers".into(),
                ));
            }
        } else if sample_bits != 16 {
            return Err(MolaError::Unsupported(
                "topography/radius and low-resolution counts must be 16-bit integers".into(),
            ));
        }
        if !map_projection.eq_ignore_ascii_case("SIMPLE CYLINDRICAL") {
            return Err(MolaError::Unsupported(
                "only simple cylindrical MEGDR grids are supported".into(),
            ));
        }
        if !latitude_type.to_ascii_lowercase().contains("planetocentric") {
            return Err(MolaError::InvalidMetadata(
                "latitude coordinate system must be planetocentric".into(),
            ));
        }
        if !longitude_direction.eq_ignore_ascii_case("EAST") {
            return Err(MolaError::InvalidMetadata(
                "longitude direction must be positive east".into(),
            ));
        }
        if !latitude_min_deg.is_finite()
            || !latitude_max_deg.is_finite()
            || !longitude_min_deg.is_finite()
            || !longitude_max_deg.is_finite()
            || !pixel_scale.is_finite()
            || !pixel_offset.is_finite()
            || !line_projection_offset.is_finite()
            || !sample_projection_offset.is_finite()
            || latitude_min_deg < -90.0
            || latitude_max_deg > 90.0
            || latitude_min_deg >= latitude_max_deg
            || longitude_min_deg < 0.0
            || longitude_max_deg > 360.0
            || longitude_min_deg >= longitude_max_deg
        {
            return Err(MolaError::InvalidMetadata(
                "invalid or non-finite MEGDR geographic/scaling metadata".into(),
            ));
        }
        if resolution >= 64 {
            let coverage_limit = if resolution == 128 { 88.0 } else { 90.0 };
            if latitude_min_deg < -coverage_limit || latitude_max_deg > coverage_limit {
                return Err(MolaError::Unsupported(format!(
                    "{resolution} ppd cylindrical MEGDR coverage cannot extend beyond ±{coverage_limit}°; polar products use a different projection"
                )));
            }
        }
        let expected_lat_rows =
            (latitude_max_deg - latitude_min_deg) * resolution as f64;
        let expected_lon_samples =
            (longitude_max_deg - longitude_min_deg) * resolution as f64;
        let row_tolerance = 1.0;
        if (expected_lat_rows - lines as f64).abs() > row_tolerance
            || (expected_lon_samples - samples as f64).abs() > row_tolerance
        {
            return Err(MolaError::InvalidMetadata(format!(
                "grid dimensions do not match geographic extent at {resolution} ppd"
            )));
        }
        Ok(Self {
            product_id,
            product_version,
            resolution_pixels_per_degree: resolution,
            lines,
            samples,
            record_bytes,
            sample_bits,
            sample_type,
            map_projection,
            latitude_type,
            longitude_direction,
            longitude_min_deg,
            longitude_max_deg,
            latitude_min_deg,
            latitude_max_deg,
            line_offset,
            sample_offset,
            line_projection_offset,
            sample_projection_offset,
            pixel_scale,
            pixel_offset,
            missing_value,
            map_kind,
            tile_origin_lat_deg,
            tile_origin_lon_deg,
        })
    }

    fn cell_for(&self, latitude_deg: f64, longitude_deg: f64) -> Result<(u32, u32), MolaError> {
        if !latitude_deg.is_finite() || !longitude_deg.is_finite() {
            return Err(MolaError::InvalidMetadata(
                "query coordinates must be finite".into(),
            ));
        }
        let coverage_limit = if self.resolution_pixels_per_degree == 128 {
            88.0
        } else {
            90.0
        };
        if !(-coverage_limit..=coverage_limit).contains(&latitude_deg) {
            return Err(MolaError::OutOfBounds);
        }
        let lon = normalize_lon(longitude_deg);
        if !(0.0..360.0).contains(&lon) && lon != 0.0 {
            return Err(MolaError::InvalidMetadata(
                "normalized longitude outside [0, 360)".into(),
            ));
        }
        // PDS simple-cylindrical projection coordinates are 1-based pixel
        // coordinates whose centers are represented by the .5 projection
        // offsets (e.g. 360.5/720.5 in the official 4 ppd label). Convert the
        // center coordinate to a zero-based cell without assuming the tile's
        // geographic bounds are themselves pixel-center coordinates.
        let sample_coord = self.sample_projection_offset
            + lon * self.resolution_pixels_per_degree as f64;
        let line_coord = self.line_projection_offset
            - latitude_deg * self.resolution_pixels_per_degree as f64;
        let x = (sample_coord - 1.0).floor() as i64;
        let y = (line_coord - 1.0).floor() as i64;
        if x < 0 || y < 0 || x as u32 >= self.samples || y as u32 >= self.lines {
            return Err(MolaError::OutOfBounds);
        }
        Ok((y as u32, x as u32))
    }
}

fn parse_label(text: &str) -> std::collections::BTreeMap<String, String> {
    let mut out = std::collections::BTreeMap::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("/*") || line.starts_with("END") {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            let key = key.trim().to_ascii_uppercase();
            let value = value.trim().trim_matches('"').trim().to_string();
            let value = value.split("/*").next().unwrap_or(&value).trim().to_string();
            out.insert(key, value);
        }
    }
    out
}

fn required(
    kv: &std::collections::BTreeMap<String, String>,
    key: &str,
) -> Result<String, MolaError> {
    kv.get(key)
        .filter(|v| !v.trim().is_empty())
        .cloned()
        .ok_or_else(|| MolaError::InvalidMetadata(format!("missing required label key {key}")))
}

fn parse_number(value: &str) -> Result<f64, MolaError> {
    value
        .split_whitespace()
        .next()
        .unwrap_or("")
        .parse::<f64>()
        .map_err(|_| MolaError::InvalidMetadata(format!("invalid numeric value {value}")))
}

fn parse_f64(
    kv: &std::collections::BTreeMap<String, String>,
    key: &str,
) -> Result<f64, MolaError> {
    parse_number(&required(kv, key)?)
}

fn parse_u32(
    kv: &std::collections::BTreeMap<String, String>,
    key: &str,
) -> Result<u32, MolaError> {
    let value = parse_number(&required(kv, key)?)?;
    if value < 0.0 || !value.is_finite() || value.fract() != 0.0 || value > u32::MAX as f64 {
        return Err(MolaError::InvalidMetadata(format!("invalid integer key {key}")));
    }
    Ok(value as u32)
}

fn parse_u64(
    kv: &std::collections::BTreeMap<String, String>,
    key: &str,
) -> Result<u64, MolaError> {
    let value = parse_number(&required(kv, key)?)?;
    if value < 0.0 || !value.is_finite() || value.fract() != 0.0 || value > u64::MAX as f64 {
        return Err(MolaError::InvalidMetadata(format!("invalid integer key {key}")));
    }
    Ok(value as u64)
}

fn parse_u16(
    kv: &std::collections::BTreeMap<String, String>,
    key: &str,
) -> Result<u16, MolaError> {
    let value = parse_number(&required(kv, key)?)?;
    if value < 0.0 || !value.is_finite() || value.fract() != 0.0 || value > u16::MAX as f64 {
        return Err(MolaError::InvalidMetadata(format!("invalid integer key {key}")));
    }
    Ok(value as u16)
}

fn parse_u32_default(
    kv: &std::collections::BTreeMap<String, String>,
    key: &str,
    default: u32,
) -> Result<u32, MolaError> {
    kv.get(key)
        .map(|v| v.parse::<u32>().map_err(|_| MolaError::InvalidMetadata(format!("invalid integer key {key}"))))
        .unwrap_or(Ok(default))
}

fn parse_f64_default(
    kv: &std::collections::BTreeMap<String, String>,
    key: &str,
    default: f64,
) -> Result<f64, MolaError> {
    kv.get(key)
        .map(|v| parse_number(v))
        .unwrap_or(Ok(default))
}

fn normalize_lon(lon_deg: f64) -> f64 {
    lon_deg.rem_euclid(360.0)
}

fn validate_img_size(metadata: &MolaMegdrMetadata, img_path: &Path) -> Result<(), MolaError> {
    let len = std::fs::metadata(img_path)?.len();
    let bytes_per_sample = u64::from(metadata.sample_bits / 8);
    let row_payload = u64::from(metadata.samples) * bytes_per_sample;
    let required = metadata.record_bytes * u64::from(metadata.line_offset)
        + u64::from(metadata.lines.saturating_sub(1)) * metadata.record_bytes
        + row_payload;
    if len < required {
        return Err(MolaError::InvalidMetadata(format!(
            "IMG is too small: {len} bytes, expected at least {required}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn label() -> String {
        [
            "PRODUCT_ID = MEGT00N000HB",
            "PRODUCT_VERSION_ID = 2.0",
            "MAP_RESOLUTION = 128",
            "LINES = 4",
            "LINE_SAMPLES = 8",
            "RECORD_BYTES = 16",
            "SAMPLE_TYPE = MSB_INTEGER",
            "SAMPLE_BITS = 16",
            "MAP_PROJECTION_TYPE = SIMPLE CYLINDRICAL",
            "COORDINATE_SYSTEM_NAME = PLANETOCENTRIC",
            "POSITIVE_LONGITUDE_DIRECTION = EAST",
            "WESTERNMOST_LONGITUDE = 0.0",
            "EASTERNMOST_LONGITUDE = 0.0625",
            "MINIMUM_LATITUDE = -0.015625",
            "MAXIMUM_LATITUDE = 0.015625",
            "MAP_TYPE = T",
            "SCALING_FACTOR = 1.0",
            "OFFSET = 0.0",
        ]
        .join("\n")
    }

    #[test]
    fn validates_pinned_final_product_metadata() {
        let mut path = std::env::temp_dir();
        path.push(format!("mola_adapter_{}_label.lbl", std::process::id()));
        let mut img = path.clone();
        img.set_extension("img");
        std::fs::write(&path, label()).unwrap();
        std::fs::write(&img, vec![0u8; 64]).unwrap();
        let product = MolaMegdrProduct::open(&path, &img, "MEGT00N000HB", "pds4-v1").unwrap();
        assert_eq!(product.metadata.resolution_pixels_per_degree, 128);
        assert_eq!(product.metadata.map_kind, 'T');
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(img);
    }

    #[test]
    fn rejects_wrong_product_version_and_grid_registration() {
        let mut text = label().replace("PRODUCT_VERSION_ID = 2.0", "PRODUCT_VERSION_ID = 1.0");
        let mut path = std::env::temp_dir();
        path.push(format!("mola_adapter_bad_{}_label.lbl", std::process::id()));
        let mut img = path.clone();
        img.set_extension("img");
        std::fs::write(&path, text).unwrap();
        std::fs::write(&img, vec![0u8; 64]).unwrap();
        assert!(MolaMegdrProduct::open(&path, &img, "MEGT00N000HB", "pds4-v1").is_err());
        text = label().replace("MAP_PROJECTION_TYPE = SIMPLE CYLINDRICAL", "MAP_PROJECTION_TYPE = POLAR");
        std::fs::write(&path, text).unwrap();
        assert!(MolaMegdrProduct::open(&path, &img, "MEGT00N000HB", "pds4-v1").is_err());
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(img);
    }

    #[test]
    fn rejects_128_ppd_queries_in_polar_coverage() {
        let text = label()
            .replace("LINES = 4", "LINES = 5632")
            .replace("LINE_SAMPLES = 8", "LINE_SAMPLES = 11520")
            .replace("EASTERNMOST_LONGITUDE = 0.0625", "EASTERNMOST_LONGITUDE = 90.0")
            .replace("MINIMUM_LATITUDE = -0.015625", "MINIMUM_LATITUDE = 44.0")
            .replace("MAXIMUM_LATITUDE = 0.015625", "MAXIMUM_LATITUDE = 88.0");
        let metadata = MolaMegdrMetadata::from_label(
            &parse_label(&text),
            "MEGT00N000HB",
        )
        .unwrap();
        assert_eq!(metadata.cell_for(89.0, 0.0), Err(MolaError::OutOfBounds));
    }

    #[test]
    fn rejects_128_ppd_global_latitude_claim() {
        let text = label()
            .replace("MINIMUM_LATITUDE = -0.015625", "MINIMUM_LATITUDE = -89.0")
            .replace("MAXIMUM_LATITUDE = 0.015625", "MAXIMUM_LATITUDE = 89.0")
            .replace("LINES = 4", "LINES = 22784");
        let error = MolaMegdrMetadata::from_label(
            &parse_label(&text),
            "MEGT00N000HB",
        )
        .unwrap_err();
        assert!(matches!(error, MolaError::Unsupported(_)));
    }

    #[test]
    fn uses_projection_offsets_for_pixel_centers() {
        let text = label()
            .replace("MAP_RESOLUTION = 128", "MAP_RESOLUTION = 4")
            .replace("LINES = 4", "LINES = 720")
            .replace("LINE_SAMPLES = 8", "LINE_SAMPLES = 1440")
            .replace("MINIMUM_LATITUDE = -0.015625", "MINIMUM_LATITUDE = -90.0")
            .replace("MAXIMUM_LATITUDE = 0.015625", "MAXIMUM_LATITUDE = 90.0")
            .replace("EASTERNMOST_LONGITUDE = 0.0625", "EASTERNMOST_LONGITUDE = 360.0")
            .replace("LINE_PROJECTION_OFFSET = 2.5", "LINE_PROJECTION_OFFSET = 360.5")
            .replace("SAMPLE_PROJECTION_OFFSET = 4.5", "SAMPLE_PROJECTION_OFFSET = 720.5");
        let metadata = MolaMegdrMetadata::from_label(
            &parse_label(&text),
            "MEGT00N000HB",
        )
        .unwrap();
        assert_eq!(metadata.cell_for(89.875, 0.125), Ok((0, 0)));
        assert_eq!(metadata.cell_for(-89.875, 359.875), Ok((719, 1439)));
    }

    #[test]
    fn rejects_non_integral_integer_metadata() {
        let text = label().replace("LINES = 4", "LINES = 4.5");
        let error = MolaMegdrMetadata::from_label(
            &parse_label(&text),
            "MEGT00N000HB",
        )
        .unwrap_err();
        assert!(matches!(error, MolaError::InvalidMetadata(_)));
    }

    #[test]
    fn zero_observation_count_cannot_become_terrain() {
        let mut path = std::env::temp_dir();
        path.push(format!("mola_adapter_zero_{}_label.lbl", std::process::id()));
        let mut img = path.clone();
        img.set_extension("img");
        std::fs::write(&path, label()).unwrap();
        std::fs::write(&img, vec![0u8; 64]).unwrap();

        let product = MolaMegdrProduct::open(&path, &img, "MEGT00N000HB", "pds4-v1").unwrap();

        let mut count_path = path.clone();
        count_path.set_file_name(format!("mola_adapter_zero_count_{}_label.lbl", std::process::id()));
        let mut count_img = count_path.clone();
        count_img.set_extension("img");
        let count_label = label()
            .replace("MEGT00N000HB", "MEGC00N000HB")
            .replace("MAP_TYPE = T", "MAP_TYPE = C")
            .replace("SAMPLE_BITS = 16", "SAMPLE_BITS = 8");
        std::fs::write(&count_path, count_label).unwrap();
        std::fs::write(&count_img, vec![0u8; 64]).unwrap();

        let counts = MolaMegdrProduct::open(&count_path, &count_img, "MEGC00N000HB", "pds4-v1").unwrap();
        let sample = product.sample_nearest_with_count(&counts, 0.0, 0.007, 3.0).unwrap();

        assert_eq!(sample.quality, TerrainQuality::Missing);
        assert!(sample.elevation_m.is_none());
        assert!(!sample.is_usable());

        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(img);
        let _ = std::fs::remove_file(count_path);
        let _ = std::fs::remove_file(count_img);
    }

    #[test]
    fn decodes_big_endian_sample_without_interpolation() {
        let mut path = std::env::temp_dir();
        path.push(format!("mola_adapter_sample_{}_label.lbl", std::process::id()));
        let mut img = path.clone();
        img.set_extension("img");
        std::fs::write(&path, label()).unwrap();
        let mut bytes = vec![0u8; 64];
        bytes[2 * 1] = 0x03;
        bytes[2 * 1 + 1] = 0xE8;
        std::fs::write(&img, bytes).unwrap();
        let product = MolaMegdrProduct::open(&path, &img, "MEGT00N000HB", "pds4-v1").unwrap();
        let mut count_path = path.clone();
        count_path.set_file_name(format!("mola_adapter_count_{}_label.lbl", std::process::id()));
        let mut count_img = count_path.clone();
        count_img.set_extension("img");
        let count_label = label()
            .replace("MEGT00N000HB", "MEGC00N000HB")
            .replace("MAP_TYPE = T", "MAP_TYPE = C")
            .replace("SAMPLE_BITS = 16", "SAMPLE_BITS = 8");
        std::fs::write(&count_path, count_label).unwrap();
        let mut count_bytes = vec![0u8; 64];
        count_bytes[32] = 1;
        std::fs::write(&count_img, count_bytes).unwrap();
        let counts = MolaMegdrProduct::open(&count_path, &count_img, "MEGC00N000HB", "pds4-v1").unwrap();
        let sample = product.sample_nearest_with_count(&counts, 0.0, 0.007, 3.0).unwrap();
        assert_eq!(sample.elevation_m, Some(1000.0));
        assert!(sample.is_usable());
        let _ = std::fs::remove_file(count_path);
        let _ = std::fs::remove_file(count_img);
        assert_eq!(sample.quality, TerrainQuality::Measured);
        assert_eq!(sample.vertical_datum, TerrainVerticalDatum::AreoidRelative);
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(img);
    }
}
