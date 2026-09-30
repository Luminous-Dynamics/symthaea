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

use sha2::{Digest, Sha256};

use crate::mars_tether::{
    TerrainArtifactComposition, TerrainProvenance, TerrainQuality, TerrainSample,
    TerrainSamplingMethod,
    TerrainVerticalDatum,
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
    pub product_creation_time: String,
    pub resolution_pixels_per_degree: u32,
    pub lines: u32,
    pub samples: u32,
    pub record_bytes: u64,
    pub sample_bits: u16,
    pub sample_type: String,
    pub map_projection: String,
    pub coordinate_system_type: String,
    pub latitude_type: String,
    pub longitude_direction: String,
    pub center_latitude_deg: f64,
    pub center_longitude_deg: f64,
    pub projection_rotation_deg: f64,
    pub longitude_min_deg: f64,
    pub longitude_max_deg: f64,
    pub latitude_min_deg: f64,
    pub latitude_max_deg: f64,
    pub line_offset: u32,
    pub image_byte_offset: u64,
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
    /// SHA-256 of the exact detached label bytes consumed by this adapter.
    pub label_sha256: String,
    /// SHA-256 of the exact raster bytes sampled by this adapter.
    pub image_sha256: String,
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
        let label_bytes = std::fs::read(label_path)?;
        let label_text = std::str::from_utf8(&label_bytes).map_err(|_| {
            MolaError::InvalidMetadata("PDS label is not valid UTF-8".into())
        })?;
        let kv = parse_label(label_text);
        let metadata = MolaMegdrMetadata::from_label(&kv, expected_product_id)?;
        validate_image_pointer_filename(&kv, img_path.as_ref())?;
        let label_sha256 = sha256_hex(&label_bytes);
        let image_sha256 = sha256_file(img_path.as_ref())?;
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
            content_digests: vec![
                ("detached-label".into(), "SHA-256".into(), label_sha256.clone()),
                ("raster-image".into(), "SHA-256".into(), image_sha256.clone()),
            ],
        };
        Ok(Self {
            metadata,
            provenance,
            label_sha256,
            image_sha256,
            img_path: img_path.as_ref().to_path_buf(),
        })
    }

    /// Open a product and require exact SHA-256 identities for both the
    /// detached label and raster. This is the cryptographic provenance gate
    /// intended for reproducible engineering runs.
    /// Return the provenance digest for a logical source file.
    pub fn content_digest(&self, logical_file: &str) -> Option<&str> {
        self.provenance
            .content_digests
            .iter()
            .find(|(name, _, _)| name == logical_file)
            .map(|(_, _, digest)| digest.as_str())
    }

    /// Return the independently authenticated artifacts belonging to this
    /// product. Unlike the legacy merged terrain provenance, this projection
    /// retains this product's own source/revision envelope for both its
    /// detached label and raster bytes.
    ///
    /// This is intentionally product-local: composing a topography product
    /// with its observation-count companion must happen at the observation
    /// boundary, where both independent source identities can be retained.
    pub fn artifact_identities(&self) -> Result<TerrainArtifactComposition, MolaError> {
        self.provenance
            .artifact_identities()
            .map_err(MolaError::InvalidMetadata)
    }

    pub fn open_verified(
        label_path: impl AsRef<Path>,
        img_path: impl AsRef<Path>,
        expected_product_id: &str,
        expected_revision: &str,
        expected_label_sha256: &str,
        expected_image_sha256: &str,
    ) -> Result<Self, MolaError> {
        let product = Self::open(
            label_path,
            img_path,
            expected_product_id,
            expected_revision,
        )?;
        verify_sha256("label", &product.label_sha256, expected_label_sha256)?;
        verify_sha256("image", &product.image_sha256, expected_image_sha256)?;
        Ok(product)
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
        let provenance = self.provenance_with_counts(counts)?;
        if count == 0 {
            return Ok(self.missing_sample(latitude_deg, longitude_deg, line, sample, provenance));
        }
        let value = self.read_i16(line, sample)? as f64;
        if self.metadata.missing_value.is_some_and(|m| value == m) {
            return Ok(self.missing_sample(latitude_deg, longitude_deg, line, sample, provenance));
        }
        let elevation = value * self.metadata.pixel_scale + self.metadata.pixel_offset;
        if !elevation.is_finite() {
            return Err(MolaError::InvalidMetadata("decoded elevation is non-finite".into()));
        }
        Ok(TerrainSample {
            latitude_rad: latitude_deg.to_radians(),
            longitude_rad: normalize_lon(longitude_deg).to_radians(),
            sampling_method: TerrainSamplingMethod::NearestCellWithObservationCount,
            source_grid_cell: Some((line, sample)),
            elevation_m: Some(elevation),
            elevation_uncertainty_m: Some(elevation_uncertainty_m),
            vertical_datum: TerrainVerticalDatum::AreoidRelative,
            slope_rad: None,
            roughness_m: None,
            quality: TerrainQuality::Measured,
            provenance,
        })
    }

    fn provenance_with_counts(&self, counts: &Self) -> Result<TerrainProvenance, MolaError> {
        self.provenance
            .merged_with(&counts.provenance)
            .map_err(MolaError::InvalidMetadata)
    }

    fn missing_sample(
        &self,
        latitude_deg: f64,
        longitude_deg: f64,
        line: u32,
        sample: u32,
        provenance: TerrainProvenance,
    ) -> TerrainSample {
        TerrainSample {
            latitude_rad: latitude_deg.to_radians(),
            longitude_rad: normalize_lon(longitude_deg).to_radians(),
            sampling_method: TerrainSamplingMethod::NearestCellWithObservationCount,
            source_grid_cell: Some((line, sample)),
            elevation_m: None,
            elevation_uncertainty_m: None,
            vertical_datum: TerrainVerticalDatum::AreoidRelative,
            slope_rad: None,
            roughness_m: None,
            quality: TerrainQuality::Missing,
            provenance,
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
            || a.product_creation_time != b.product_creation_time
            || a.map_projection != b.map_projection
            || a.coordinate_system_type != b.coordinate_system_type
            || a.latitude_type != b.latitude_type
            || a.longitude_direction != b.longitude_direction
            || a.projection_rotation_deg != b.projection_rotation_deg
            || a.resolution_pixels_per_degree != b.resolution_pixels_per_degree
            || a.lines != b.lines
            || a.samples != b.samples
            || a.latitude_min_deg != b.latitude_min_deg
            || a.latitude_max_deg != b.latitude_max_deg
            || a.longitude_min_deg != b.longitude_min_deg
            || a.longitude_max_deg != b.longitude_max_deg
            || a.center_latitude_deg != b.center_latitude_deg
            || a.center_longitude_deg != b.center_longitude_deg
            || a.line_projection_offset != b.line_projection_offset
            || a.sample_projection_offset != b.sample_projection_offset
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
        let bytes_per_sample = u64::from(self.metadata.sample_bits / 8);
        let byte_offset = self.raster_byte_offset(line, sample, bytes_per_sample)?;
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
                let value = match self.metadata.sample_type.as_str() {
                    "MSB_INTEGER" => u16::from_be_bytes(b),
                    "LSB_INTEGER" => u16::from_le_bytes(b),
                    other => return Err(MolaError::Unsupported(format!(
                        "sample type {other} is not a supported count encoding"
                    ))),
                };
                Ok(u32::from(value))
            }
            _ => Err(MolaError::Unsupported(
                "counts must be an 8-bit or 16-bit integer".into(),
            )),
        }
    }

    fn raster_byte_offset(
        &self,
        line: u32,
        sample: u32,
        bytes_per_sample: u64,
    ) -> Result<u64, MolaError> {
        if line >= self.metadata.lines || sample >= self.metadata.samples {
            return Err(MolaError::OutOfBounds);
        }
        if bytes_per_sample == 0 {
            return Err(MolaError::InvalidMetadata(
                "raster sample width must be non-zero".into(),
            ));
        }
        let row = u64::from(line)
            .checked_add(u64::from(self.metadata.line_offset))
            .ok_or_else(|| MolaError::InvalidMetadata("raster row offset overflow".into()))?;
        let row_offset = u64::from(self.metadata.record_bytes)
            .checked_mul(row)
            .ok_or_else(|| MolaError::InvalidMetadata("raster byte offset overflow".into()))?;
        let sample_offset = u64::from(sample)
            .checked_mul(bytes_per_sample)
            .and_then(|offset| offset.checked_add(u64::from(self.metadata.sample_offset)))
            .ok_or_else(|| MolaError::InvalidMetadata("raster byte offset overflow".into()))?;
        self.metadata
            .image_byte_offset
            .checked_add(row_offset)
            .and_then(|offset| offset.checked_add(sample_offset))
            .ok_or_else(|| MolaError::InvalidMetadata("raster byte offset overflow".into()))
    }

    fn read_i16(&self, line: u32, sample: u32) -> Result<i16, MolaError> {
        if line >= self.metadata.lines || sample >= self.metadata.samples {
            return Err(MolaError::OutOfBounds);
        }
        let byte_offset = self.raster_byte_offset(line, sample, 2)?;
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
        let product_creation_time = required(kv, "PRODUCT_CREATION_TIME")?;
        validate_creation_time(&product_creation_time)?;
        let resolution = parse_u32(kv, "MAP_RESOLUTION")?;
        let lines = parse_u32(kv, "LINES")?;
        let samples = parse_u32(kv, "LINE_SAMPLES")?;
        let record_bytes = parse_u64(kv, "RECORD_BYTES")?;
        let sample_bits = parse_u16(kv, "SAMPLE_BITS")?;
        let sample_type = required(kv, "SAMPLE_TYPE")?;
        let map_projection = required(kv, "MAP_PROJECTION_TYPE")?;
        let coordinate_system_type = required(kv, "COORDINATE_SYSTEM_TYPE")?;
        let latitude_type = required(kv, "COORDINATE_SYSTEM_NAME")?;
        let longitude_direction = required(kv, "POSITIVE_LONGITUDE_DIRECTION")?;
        let center_latitude_deg = parse_f64(kv, "CENTER_LATITUDE")?;
        let center_longitude_deg = parse_f64(kv, "CENTER_LONGITUDE")?;
        let projection_rotation_deg = parse_f64(kv, "MAP_PROJECTION_ROTATION")?;
        let longitude_min_deg = parse_f64(kv, "WESTERNMOST_LONGITUDE")?;
        let longitude_max_deg = parse_f64(kv, "EASTERNMOST_LONGITUDE")?;
        let latitude_min_deg = parse_f64(kv, "MINIMUM_LATITUDE")?;
        let latitude_max_deg = parse_f64(kv, "MAXIMUM_LATITUDE")?;
        let (line_offset, image_byte_offset) = image_data_pointer(kv)?;
        let sample_offset = 0;
        // Projection offsets are georeferencing authority, not safe-to-guess
        // defaults. A missing offset can silently shift every sampled cell.
        let line_projection_offset = parse_f64(kv, "LINE_PROJECTION_OFFSET")?;
        let sample_projection_offset = parse_f64(kv, "SAMPLE_PROJECTION_OFFSET")?;
        let pixel_scale = parse_f64_default(kv, "SCALING_FACTOR", 1.0)?;
        let pixel_offset = parse_f64_default(kv, "OFFSET", 0.0)?;
        let missing_value = match kv.get("MISSING_CONSTANT") {
            Some(value) => Some(parse_number(value)?),
            None => None,
        };
        let map_kind = required(kv, "MAP_TYPE")?.chars().next().ok_or_else(|| {
            MolaError::InvalidMetadata("MAP_TYPE is empty".into())
        })?;
        if !matches!(map_kind, 'T' | 'C' | 'R' | 'A') {
            return Err(MolaError::InvalidMetadata("unsupported MEGDR map type".into()));
        }
        let product_kind = product_id.chars().nth(3);
        if product_kind != Some(map_kind) {
            return Err(MolaError::InvalidMetadata(
                "PRODUCT_ID map-kind prefix does not match MAP_TYPE".into(),
            ));
        }
        validate_optional_raster_layout(kv, lines, samples, record_bytes)?;
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
        if product_creation_time < "2003-03-21T00:00:00" {
            return Err(MolaError::InvalidMetadata(
                "MEGDR product creation time predates the final 2.0 release".into(),
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
        if !coordinate_system_type.eq_ignore_ascii_case("BODY-FIXED ROTATING") {
            return Err(MolaError::InvalidMetadata(
                "coordinate system type must be BODY-FIXED ROTATING".into(),
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
        if !projection_rotation_deg.is_finite() || projection_rotation_deg != 0.0 {
            return Err(MolaError::Unsupported(
                "only zero-rotation MOLA simple-cylindrical grids are supported".into(),
            ));
        }
        if !center_latitude_deg.is_finite()
            || center_latitude_deg != 0.0
            || !center_longitude_deg.is_finite()
            || center_longitude_deg < 0.0
            || center_longitude_deg >= 360.0
            || !latitude_min_deg.is_finite()
            || !latitude_max_deg.is_finite()
            || !longitude_min_deg.is_finite()
            || !longitude_max_deg.is_finite()
            || !pixel_scale.is_finite()
            || !pixel_offset.is_finite()
            || !line_projection_offset.is_finite()
            || !sample_projection_offset.is_finite()
            || missing_value.is_some_and(|value| !value.is_finite())
            || !tile_origin_lat_deg.is_finite()
            || !tile_origin_lon_deg.is_finite()
            || latitude_min_deg < -90.0
            || latitude_max_deg > 90.0
            || latitude_min_deg >= latitude_max_deg
            || longitude_min_deg < 0.0
            || longitude_max_deg > 360.0
            || longitude_min_deg >= longitude_max_deg
            || tile_origin_lat_deg < latitude_min_deg
            || tile_origin_lat_deg > latitude_max_deg
            || tile_origin_lon_deg < longitude_min_deg
            || tile_origin_lon_deg > longitude_max_deg
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
            product_creation_time,
            resolution_pixels_per_degree: resolution,
            lines,
            samples,
            record_bytes,
            sample_bits,
            sample_type,
            map_projection,
            coordinate_system_type,
            latitude_type,
            longitude_direction,
            center_latitude_deg,
            center_longitude_deg,
            projection_rotation_deg,
            longitude_min_deg,
            longitude_max_deg,
            latitude_min_deg,
            latitude_max_deg,
            line_offset,
            image_byte_offset,
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

        // A tiled MEGDR raster is not a global longitude lookup table. Reject
        // coordinates outside this label's declared geographic footprint before
        // applying projection arithmetic; otherwise a valid-looking coordinate
        // from another tile can wrap into an unrelated cell.
        let longitude_is_global =
            self.longitude_min_deg == 0.0 && self.longitude_max_deg == 360.0;
        if latitude_deg < self.latitude_min_deg || latitude_deg > self.latitude_max_deg {
            return Err(MolaError::OutOfBounds);
        }
        if !longitude_is_global
            && (lon < self.longitude_min_deg || lon >= self.longitude_max_deg)
        {
            return Err(MolaError::OutOfBounds);
        }

        // PDS simple-cylindrical projection coordinates are 1-based pixel
        // coordinates whose centers are represented by the .5 projection
        // offsets (e.g. 360.5/720.5 in the official 4 ppd label). Convert the
        // center coordinate to a zero-based cell without assuming the tile's
        // geographic bounds are themselves pixel-center coordinates.
        let sample_coord = self.sample_projection_offset
            + shortest_lon_delta(lon, self.center_longitude_deg)
                * self.resolution_pixels_per_degree as f64;
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
        if line.is_empty() || line.starts_with("/*") || line.eq_ignore_ascii_case("END") {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            let key = key.trim().to_ascii_uppercase();
            let value = value.split("/*").next().unwrap_or(value).trim();
            let value = value.trim_matches('"').trim().to_string();
            out.insert(key, value);
        }
    }
    out
}

fn image_data_pointer(
    kv: &std::collections::BTreeMap<String, String>,
) -> Result<(u32, u64), MolaError> {
    let pointer = match kv.get("^IMAGE") {
        None => return Ok((0, 0)),
        Some(pointer) => pointer.trim(),
    };

    // Detached PDS3 pointers have a filename, optionally followed by a
    // record or byte offset. Bare numeric pointers belong to attached labels
    // and are rejected by this detached-label API.
    if pointer.parse::<f64>().is_ok() {
        return Err(MolaError::InvalidMetadata(
            "bare numeric ^IMAGE pointers are invalid for detached labels".into(),
        ));
    }

    if pointer.starts_with('"') && pointer.ends_with('"') {
        return Ok((0, 0));
    }

    if pointer.starts_with('(') && pointer.ends_with(')') {
        let inner = pointer[1..pointer.len() - 1].trim();
        let (filename, offset) = inner.split_once(',').ok_or_else(|| {
            MolaError::InvalidMetadata(
                "detached ^IMAGE tuple must contain filename and offset".into(),
            )
        })?;
        let filename = filename.trim();
        if filename.len() < 2 || !filename.starts_with('"') || !filename.ends_with('"') {
            return Err(MolaError::InvalidMetadata(
                "detached ^IMAGE tuple must begin with a quoted filename".into(),
            ));
        }
        let offset = offset.trim();
        if let Some(byte_value) = offset.strip_suffix("<BYTES>") {
            let byte_value = parse_positive_pointer(byte_value, "^IMAGE byte offset")?;
            return Ok((0, byte_value - 1));
        }
        let record = parse_positive_pointer(offset, "^IMAGE record offset")?;
        let record = u32::try_from(record - 1).map_err(|_| {
            MolaError::InvalidMetadata("^IMAGE record offset exceeds u32 range".into())
        })?;
        return Ok((record, 0));
    }

    if pointer.to_ascii_lowercase().ends_with(".img") {
        return Ok((0, 0));
    }

    Err(MolaError::InvalidMetadata("invalid detached ^IMAGE pointer".into()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn verify_sha256(
    kind: &str,
    actual: &str,
    expected: &str,
) -> Result<(), MolaError> {
    let expected = expected.trim().to_ascii_lowercase();
    if expected.len() != 64 || !expected.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(MolaError::InvalidMetadata(format!(
            "expected {kind} SHA-256 must be 64 hexadecimal characters"
        )));
    }
    if actual != expected {
        return Err(MolaError::InvalidMetadata(format!(
            "{kind} SHA-256 mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, MolaError> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(sha256_hex(&hasher.finalize()))
}

fn validate_image_pointer_filename(
    kv: &std::collections::BTreeMap<String, String>,
    img_path: &Path,
) -> Result<(), MolaError> {
    let pointer = match kv.get("^IMAGE") {
        None => return Ok(()),
        Some(pointer) => pointer.trim(),
    };

    let filename = if pointer.starts_with('(') && pointer.ends_with(')') {
        let inner = pointer[1..pointer.len() - 1].trim();
        let (filename, _) = inner.split_once(',').ok_or_else(|| {
            MolaError::InvalidMetadata(
                "detached ^IMAGE tuple must contain filename and offset".into(),
            )
        })?;
        filename.trim()
    } else if pointer.starts_with('"') && pointer.ends_with('"') {
        pointer
    } else if pointer.to_ascii_lowercase().ends_with(".img") {
        pointer
    } else {
        return Ok(());
    };

    let filename = filename
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .unwrap_or(filename)
        .trim();

    // MOLA's detached-label SIS permits an optional [dirlist] prefix. The
    // caller supplies the actual raster path, so only the referenced terminal
    // filename is authoritative here; the directory list must not redirect
    // filesystem access outside the caller-selected path.
    let terminal = filename
        .rsplit_once(']')
        .map(|(_, name)| name)
        .unwrap_or(filename);

    if terminal.is_empty()
        || terminal.contains('/')
        || terminal.contains('\\')
        || Path::new(terminal).file_name().and_then(|name| name.to_str()) != Some(terminal)
    {
        return Err(MolaError::InvalidMetadata(
            "detached ^IMAGE filename is not a valid terminal filename".into(),
        ));
    }

    let actual = img_path.file_name().and_then(|name| name.to_str()).ok_or_else(|| {
        MolaError::InvalidMetadata("IMG path has no valid filename".into())
    })?;

    if !terminal.eq_ignore_ascii_case(actual) {
        return Err(MolaError::InvalidMetadata(format!(
            "detached ^IMAGE references {terminal}, but caller supplied {actual}"
        )));
    }

    Ok(())
}

fn parse_positive_pointer(value: &str, field: &str) -> Result<u64, MolaError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(MolaError::InvalidMetadata(format!("{field} is empty")));
    }
    let parsed = value.parse::<u64>().map_err(|_| {
        MolaError::InvalidMetadata(format!("{field} must be a positive integer"))
    })?;
    if parsed == 0 {
        return Err(MolaError::InvalidMetadata(format!(
            "{field} must be positive"
        )));
    }
    Ok(parsed)
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

fn validate_creation_time(value: &str) -> Result<(), MolaError> {
    let b = value.as_bytes();
    if b.len() < 19
        || b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'T'
        || b[13] != b':'
        || b[16] != b':'
        || !b[..19].iter().enumerate().all(|(i, c)| {
            matches!(i, 4 | 7) && *c == b'-'
                || i == 10 && *c == b'T'
                || matches!(i, 13 | 16) && *c == b':'
                || matches!(i, 0..=3 | 5..=6 | 8..=9 | 11..=12 | 14..=15 | 17..=18)
                    && c.is_ascii_digit()
        })
    {
        return Err(MolaError::InvalidMetadata(
            "PRODUCT_CREATION_TIME must use YYYY-MM-DDThh:mm:ss format".into(),
        ));
    }

    // The release gate below compares this field lexicographically, so the
    // structural validation must reject impossible calendar/time components
    // rather than merely checking separators and digit classes. MEGDR labels
    // may carry optional fractional seconds, but no other suffix is accepted.
    if b.len() > 19 && (b[19] != b'.' || !b[20..].iter().all(u8::is_ascii_digit)) {
        return Err(MolaError::InvalidMetadata(
            "PRODUCT_CREATION_TIME may only add fractional seconds after hh:mm:ss".into(),
        ));
    }

    let year = value[0..4]
        .parse::<u32>()
        .map_err(|_| MolaError::InvalidMetadata("invalid creation year".into()))?;
    let month = value[5..7]
        .parse::<u32>()
        .map_err(|_| MolaError::InvalidMetadata("invalid creation month".into()))?;
    let day = value[8..10]
        .parse::<u32>()
        .map_err(|_| MolaError::InvalidMetadata("invalid creation day".into()))?;
    let hour = value[11..13]
        .parse::<u32>()
        .map_err(|_| MolaError::InvalidMetadata("invalid creation hour".into()))?;
    let minute = value[14..16]
        .parse::<u32>()
        .map_err(|_| MolaError::InvalidMetadata("invalid creation minute".into()))?;
    let second = value[17..19]
        .parse::<u32>()
        .map_err(|_| MolaError::InvalidMetadata("invalid creation second".into()))?;

    let leap_year = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap_year => 29,
        2 => 28,
        _ => 0,
    };
    if days_in_month == 0 || day == 0 || day > days_in_month || hour > 23 || minute > 59 || second > 59 {
        return Err(MolaError::InvalidMetadata(
            "PRODUCT_CREATION_TIME contains an impossible calendar/time value".into(),
        ));
    }
    Ok(())
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

fn parse_f64_default(
    kv: &std::collections::BTreeMap<String, String>,
    key: &str,
    default: f64,
) -> Result<f64, MolaError> {
    kv.get(key)
        .map(|v| parse_number(v))
        .unwrap_or(Ok(default))
}

fn validate_optional_raster_layout(
    kv: &std::collections::BTreeMap<String, String>,
    lines: u32,
    samples: u32,
    record_bytes: u64,
) -> Result<(), MolaError> {
    if let Some(record_type) = kv.get("RECORD_TYPE") {
        if !record_type.eq_ignore_ascii_case("FIXED_LENGTH") {
            return Err(MolaError::Unsupported(
                "MEGDR raster records must be FIXED_LENGTH".into(),
            ));
        }
    }
    if let Some(pds_version) = kv.get("PDS_VERSION_ID") {
        if !pds_version.eq_ignore_ascii_case("PDS3") {
            return Err(MolaError::Unsupported(
                "this adapter expects PDS3 MEGDR labels".into(),
            ));
        }
    }
    if let Some(file_records) = kv.get("FILE_RECORDS") {
        let value = parse_number(file_records)?;
        if !value.is_finite() || value.fract() != 0.0 || value < 0.0 || value > u32::MAX as f64 {
            return Err(MolaError::InvalidMetadata(
                "FILE_RECORDS must be a non-negative integer".into(),
            ));
        }
        if value as u32 != lines {
            return Err(MolaError::InvalidMetadata(
                "FILE_RECORDS must match LINES for fixed-length MEGDR images".into(),
            ));
        }
    }
    if let Some(first) = kv.get("LINE_FIRST_PIXEL") {
        if parse_number(first)? != 1.0 {
            return Err(MolaError::InvalidMetadata(
                "LINE_FIRST_PIXEL must be 1".into(),
            ));
        }
    }
    if let Some(first) = kv.get("SAMPLE_FIRST_PIXEL") {
        if parse_number(first)? != 1.0 {
            return Err(MolaError::InvalidMetadata(
                "SAMPLE_FIRST_PIXEL must be 1".into(),
            ));
        }
    }
    if let Some(last) = kv.get("LINE_LAST_PIXEL") {
        if parse_number(last)? != lines as f64 {
            return Err(MolaError::InvalidMetadata(
                "LINE_LAST_PIXEL must match LINES".into(),
            ));
        }
    }
    if let Some(last) = kv.get("SAMPLE_LAST_PIXEL") {
        if parse_number(last)? != samples as f64 {
            return Err(MolaError::InvalidMetadata(
                "SAMPLE_LAST_PIXEL must match LINE_SAMPLES".into(),
            ));
        }
    }
    if record_bytes == 0 {
        return Err(MolaError::InvalidMetadata(
            "RECORD_BYTES must be non-zero".into(),
        ));
    }
    Ok(())
}

fn normalize_lon(lon_deg: f64) -> f64 {
    lon_deg.rem_euclid(360.0)
}

fn shortest_lon_delta(lon_deg: f64, center_deg: f64) -> f64 {
    (lon_deg - center_deg + 180.0).rem_euclid(360.0) - 180.0
}

fn validate_img_size(metadata: &MolaMegdrMetadata, img_path: &Path) -> Result<(), MolaError> {
    let len = std::fs::metadata(img_path)?.len();
    let bytes_per_sample = u64::from(metadata.sample_bits / 8);
    let row_payload = u64::from(metadata.samples)
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| MolaError::InvalidMetadata("IMG row size calculation overflow".into()))?;
    let row_end = u64::from(metadata.sample_offset)
        .checked_add(row_payload)
        .ok_or_else(|| MolaError::InvalidMetadata("IMG row size calculation overflow".into()))?;
    if row_end > metadata.record_bytes {
        return Err(MolaError::InvalidMetadata(
            "sample payload exceeds the declared record size".into(),
        ));
    }
    let first_row_offset = metadata
        .record_bytes
        .checked_mul(u64::from(metadata.line_offset))
        .ok_or_else(|| MolaError::InvalidMetadata("IMG size calculation overflow".into()))?;
    let preceding_rows = metadata
        .record_bytes
        .checked_mul(u64::from(metadata.lines.saturating_sub(1)))
        .ok_or_else(|| MolaError::InvalidMetadata("IMG size calculation overflow".into()))?;
    let required = metadata
        .image_byte_offset
        .checked_add(first_row_offset)
        .and_then(|offset| offset.checked_add(preceding_rows))
        .and_then(|offset| offset.checked_add(row_end))
        .ok_or_else(|| MolaError::InvalidMetadata("IMG size calculation overflow".into()))?;
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
            "PRODUCT_CREATION_TIME = 2003-03-21T00:00:00",
            "MAP_RESOLUTION = 128",
            "LINES = 4",
            "LINE_SAMPLES = 8",
            "RECORD_BYTES = 16",
            "SAMPLE_TYPE = MSB_INTEGER",
            "SAMPLE_BITS = 16",
            "MAP_PROJECTION_TYPE = SIMPLE CYLINDRICAL",
            "COORDINATE_SYSTEM_TYPE = BODY-FIXED ROTATING",
            "COORDINATE_SYSTEM_NAME = PLANETOCENTRIC",
            "POSITIVE_LONGITUDE_DIRECTION = EAST",
            "CENTER_LATITUDE = 0.0",
            "CENTER_LONGITUDE = 180.0",
            "MAP_PROJECTION_ROTATION = 0.0",
            "LINE_PROJECTION_OFFSET = 2.5",
            "SAMPLE_PROJECTION_OFFSET = 4.5",
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
    fn tiled_grid_rejects_coordinates_outside_declared_footprint() {
        let text = label()
            .replace("WESTERNMOST_LONGITUDE = 0.0", "WESTERNMOST_LONGITUDE = 270.0")
            .replace("EASTERNMOST_LONGITUDE = 0.0625", "EASTERNMOST_LONGITUDE = 360.0")
            .replace("MINIMUM_LATITUDE = -0.015625", "MINIMUM_LATITUDE = -44.0")
            .replace("MAXIMUM_LATITUDE = 0.015625", "MAXIMUM_LATITUDE = 0.0")
            .replace("LINES = 4", "LINES = 5632")
            .replace("LINE_SAMPLES = 8", "LINE_SAMPLES = 11520")
            .replace("RECORD_BYTES = 16", "RECORD_BYTES = 23040")
            .replace("LINE_PROJECTION_OFFSET = 2.5", "LINE_PROJECTION_OFFSET = 0.5")
            .replace("SAMPLE_PROJECTION_OFFSET = 4.5", "SAMPLE_PROJECTION_OFFSET = -11519.5");
        let metadata =
            MolaMegdrMetadata::from_label(&parse_label(&text), "MEGT00N000HB").unwrap();

        assert!(matches!(
            metadata.cell_for(-10.0, 269.999),
            Err(MolaError::OutOfBounds)
        ));
        assert!(matches!(
            metadata.cell_for(-10.0, 360.0),
            Err(MolaError::OutOfBounds)
        ));
        assert!(matches!(
            metadata.cell_for(-44.001, 300.0),
            Err(MolaError::OutOfBounds)
        ));
        assert!(matches!(
            metadata.cell_for(0.001, 300.0),
            Err(MolaError::OutOfBounds)
        ));
    }

    #[test]
    fn valles_128ppd_geometry_matches_published_archive_sizes() {
        let topography = label()
            .replace("MEGT00N000HB", "MEGT00N270HB")
            .replace("LINES = 4", "LINES = 5632")
            .replace("LINE_SAMPLES = 8", "LINE_SAMPLES = 11520")
            .replace("RECORD_BYTES = 16", "RECORD_BYTES = 23040")
            .replace("LINE_PROJECTION_OFFSET = 2.5", "LINE_PROJECTION_OFFSET = 0.5")
            .replace("SAMPLE_PROJECTION_OFFSET = 4.5", "SAMPLE_PROJECTION_OFFSET = -11519.5")
            .replace("WESTERNMOST_LONGITUDE = 0.0", "WESTERNMOST_LONGITUDE = 270.0")
            .replace("EASTERNMOST_LONGITUDE = 0.0625", "EASTERNMOST_LONGITUDE = 360.0")
            .replace("MINIMUM_LATITUDE = -0.015625", "MINIMUM_LATITUDE = -44.0")
            .replace("MAXIMUM_LATITUDE = 0.015625", "MAXIMUM_LATITUDE = 0.0");
        let topography =
            MolaMegdrMetadata::from_label(&parse_label(&topography), "MEGT00N270HB").unwrap();

        let counts = label()
            .replace("MEGT00N000HB", "MEGC00N270HB")
            .replace("MAP_TYPE = T", "MAP_TYPE = C")
            .replace("SAMPLE_BITS = 16", "SAMPLE_BITS = 8")
            .replace("LINES = 4", "LINES = 5632")
            .replace("LINE_SAMPLES = 8", "LINE_SAMPLES = 11520")
            .replace("RECORD_BYTES = 16", "RECORD_BYTES = 11520")
            .replace("LINE_PROJECTION_OFFSET = 2.5", "LINE_PROJECTION_OFFSET = 0.5")
            .replace("SAMPLE_PROJECTION_OFFSET = 4.5", "SAMPLE_PROJECTION_OFFSET = -11519.5")
            .replace("WESTERNMOST_LONGITUDE = 0.0", "WESTERNMOST_LONGITUDE = 270.0")
            .replace("EASTERNMOST_LONGITUDE = 0.0625", "EASTERNMOST_LONGITUDE = 360.0")
            .replace("MINIMUM_LATITUDE = -0.015625", "MINIMUM_LATITUDE = -44.0")
            .replace("MAXIMUM_LATITUDE = 0.015625", "MAXIMUM_LATITUDE = 0.0");
        let counts =
            MolaMegdrMetadata::from_label(&parse_label(&counts), "MEGC00N270HB").unwrap();

        let topography_bytes =
            u64::from(topography.lines) * topography.record_bytes;
        let count_bytes = u64::from(counts.lines) * counts.record_bytes;

        // These are the exact byte sizes listed by the maintained legacy PDS3 archive.
        assert_eq!(topography_bytes, 129_761_280);
        assert_eq!(count_bytes, 64_880_640);
        assert_eq!(
            u64::from(topography.lines) * u64::from(topography.samples) * 2,
            topography_bytes
        );
        assert_eq!(
            u64::from(counts.lines) * u64::from(counts.samples),
            count_bytes
        );
    }

    #[test]
    fn pins_valles_128ppd_projection_offsets_and_cell_centers() {
        // The published MEGT00N270HB label reproduces the 128 ppd 270E-360E
        // tile as 5632 x 11520 pixels with LINE_PROJECTION_OFFSET=0.5 and
        // SAMPLE_PROJECTION_OFFSET=-11519.5. Those values are the evidence
        // needed to exercise the tile-local longitude convention without
        // importing a generic planetary-map convention.
        let text = label()
            .replace("MEGT00N000HB", "MEGT00N270HB")
            .replace("LINES = 4", "LINES = 5632")
            .replace("LINE_SAMPLES = 8", "LINE_SAMPLES = 11520")
            .replace("RECORD_BYTES = 16", "RECORD_BYTES = 23040")
            .replace("LINE_PROJECTION_OFFSET = 2.5", "LINE_PROJECTION_OFFSET = 0.5")
            .replace("SAMPLE_PROJECTION_OFFSET = 4.5", "SAMPLE_PROJECTION_OFFSET = -11519.5")
            .replace("WESTERNMOST_LONGITUDE = 0.0", "WESTERNMOST_LONGITUDE = 270.0")
            .replace("EASTERNMOST_LONGITUDE = 0.0625", "EASTERNMOST_LONGITUDE = 360.0")
            .replace("MINIMUM_LATITUDE = -0.015625", "MINIMUM_LATITUDE = -44.0")
            .replace("MAXIMUM_LATITUDE = 0.015625", "MAXIMUM_LATITUDE = 0.0");

        let metadata =
            MolaMegdrMetadata::from_label(&parse_label(&text), "MEGT00N270HB").unwrap();

        assert_eq!(metadata.lines, 5632);
        assert_eq!(metadata.samples, 11520);
        assert_eq!(metadata.line_projection_offset, 0.5);
        assert_eq!(metadata.sample_projection_offset, -11519.5);

        // Pixel centers of the published tile footprint map to its first and
        // last raster cells. The exact center coordinates avoid boundary
        // ambiguity at 0/360 and -44/0.
        assert_eq!(
            metadata.cell_for(-0.00390625, 270.00390625).unwrap(),
            (0, 0)
        );
        assert_eq!(
            metadata.cell_for(-43.99609375, 359.99609375).unwrap(),
            (5631, 11519)
        );
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
    fn detached_image_pointer_does_not_apply_label_records() {
        let text = format!(
            "{}\nLABEL_RECORDS = 99\n^IMAGE = \"MEGT00N270HB.IMG\"",
            label()
        );
        let metadata = MolaMegdrMetadata::from_label(
            &parse_label(&text),
            "MEGT00N000HB",
        )
        .unwrap();
        assert_eq!(metadata.line_offset, 0);
    }

    #[test]
    fn rejects_malformed_numeric_image_record_pointer() {
        for value in ["1.5", "-1", "0", "NaN"] {
            let text = format!("{}\n^IMAGE = {}", label(), value);
            let error = MolaMegdrMetadata::from_label(
                &parse_label(&text),
                "MEGT00N000HB",
            )
            .unwrap_err();
            assert!(matches!(error, MolaError::InvalidMetadata(_)), "^IMAGE={value}");
        }
    }

    #[test]
    fn rejects_bare_numeric_image_pointer_for_detached_label() {
        for value in ["1", "7", "1.5", "-1", "0", "NaN"] {
            let text = format!("{}\n^IMAGE = {}", label(), value);
            let error = MolaMegdrMetadata::from_label(
                &parse_label(&text),
                "MEGT00N000HB",
            )
            .unwrap_err();
            assert!(matches!(error, MolaError::InvalidMetadata(_)), "^IMAGE={value}");
        }
    }

    #[test]
    fn parses_detached_image_record_and_byte_pointers() {
        for (pointer, expected_record, expected_byte) in [
            (r#"("MEGT00N000HB.IMG",2)"#, 1, 0),
            (r#"("MEGT00N000HB.IMG",700 <BYTES>)"#, 0, 699),
        ] {
            let text = format!("{}\n^IMAGE = {}", label(), pointer);
            let metadata = MolaMegdrMetadata::from_label(
                &parse_label(&text),
                "MEGT00N000HB",
            )
            .unwrap();
            assert_eq!(metadata.line_offset, expected_record);
            assert_eq!(metadata.image_byte_offset, expected_byte);
        }
    }

    #[test]
    fn count_gated_sample_carries_topography_and_count_digests() {
        let mut label_path = std::env::temp_dir();
        label_path.push(format!("mola_provenance_{}_label.lbl", std::process::id()));
        let mut topography_path = label_path.clone();
        topography_path.set_file_name("MEGT00N000HB.IMG");
        let mut counts_label_path = label_path.clone();
        counts_label_path.set_file_name("mola_counts.lbl");
        let mut counts_path = label_path.clone();
        counts_path.set_file_name("MEGC00N000HB.IMG");

        let mut topography_bytes = vec![0u8; 64];
        topography_bytes[40..42].copy_from_slice(&1000i16.to_be_bytes());
        let mut count_bytes = vec![0u8; 32];
        count_bytes[20] = 1;
        std::fs::write(&label_path, label()).unwrap();
        std::fs::write(&topography_path, topography_bytes).unwrap();
        let counts_label = label()
            .replace("MEGT00N000HB", "MEGC00N000HB")
            .replace("MAP_TYPE = T", "MAP_TYPE = C")
            .replace("SAMPLE_BITS = 16", "SAMPLE_BITS = 8")
            .replace("RECORD_BYTES = 16", "RECORD_BYTES = 8");
        std::fs::write(&counts_label_path, counts_label).unwrap();
        std::fs::write(&counts_path, count_bytes).unwrap();

        let topo = MolaMegdrProduct::open(
            &label_path,
            &topography_path,
            "MEGT00N000HB",
            "fixture-v1",
        )
        .unwrap();
        let counts = MolaMegdrProduct::open(
            &counts_label_path,
            &counts_path,
            "MEGC00N000HB",
            "fixture-v1",
        )
        .unwrap();
        let sample = topo.sample_nearest_with_count(&counts, 0.0, 180.0, 2.0).unwrap();

        assert_eq!(sample.quality, TerrainQuality::Measured);
        assert_eq!(sample.elevation_m, Some(1000.0));
        assert_eq!(sample.provenance.content_digests.len(), 4);
        assert_eq!(sample.provenance.content_digests[0].0, "detached-label");
        assert_eq!(sample.provenance.content_digests[1].0, "raster-image");
        assert_eq!(sample.provenance.content_digests[2].0, "counts-label");
        assert_eq!(sample.provenance.content_digests[3].0, "counts-raster");
        assert_eq!(sample.provenance.content_digests[2].2, counts.label_sha256);
        assert_eq!(sample.provenance.content_digests[3].2, counts.image_sha256);

        let _ = std::fs::remove_file(label_path);
        let _ = std::fs::remove_file(topography_path);
        let _ = std::fs::remove_file(counts_label_path);
        let _ = std::fs::remove_file(counts_path);
    }

    #[test]
    fn sha256_verifier_is_strict_and_case_insensitive() {
        let bytes = b"mola-provenance";
        let actual = sha256_hex(bytes);
        assert!(verify_sha256("fixture", &actual, &actual.to_ascii_uppercase()).is_ok());
        assert!(matches!(
            verify_sha256("fixture", &actual, "00"),
            Err(MolaError::InvalidMetadata(_))
        ));
        assert!(matches!(
            verify_sha256("fixture", &actual, &"0".repeat(64)),
            Err(MolaError::InvalidMetadata(_))
        ));
    }

    #[test]
    fn detached_image_pointer_must_match_supplied_raster_filename() {
        let matching = label().replace(
            "MAP_TYPE = T",
            "MAP_TYPE = T\n^IMAGE = (\"MEGT00N000HB.IMG\",1)",
        );
        let kv = parse_label(&matching);
        assert!(validate_image_pointer_filename(
            &kv,
            Path::new("/tmp/MEGT00N000HB.IMG"),
        )
        .is_ok());

        let mismatched = matching.replace("MEGT00N000HB.IMG", "MEGT00N270HB.IMG");
        let kv = parse_label(&mismatched);
        assert!(matches!(
            validate_image_pointer_filename(&kv, Path::new("/tmp/MEGT00N000HB.IMG")),
            Err(MolaError::InvalidMetadata(_))
        ));
    }

    #[test]
    fn detached_image_pointer_dirlist_cannot_redirect_supplied_path() {
        let text = label().replace(
            "MAP_TYPE = T",
            "MAP_TYPE = T\n^IMAGE = (\"[DATA]MEGT00N000HB.IMG\",1)",
        );
        let kv = parse_label(&text);
        assert!(validate_image_pointer_filename(
            &kv,
            Path::new("/archive/MEGT00N000HB.IMG"),
        )
        .is_ok());

        let traversal = label().replace(
            "MAP_TYPE = T",
            "MAP_TYPE = T\n^IMAGE = (\"../MEGT00N000HB.IMG\",1)",
        );
        let kv = parse_label(&traversal);
        assert!(matches!(
            validate_image_pointer_filename(&kv, Path::new("/archive/MEGT00N000HB.IMG")),
            Err(MolaError::InvalidMetadata(_))
        ));
    }

    #[test]
    fn rejects_malformed_detached_image_pointers() {
        for pointer in [
            r#"("MEGT00N000HB.IMG",0)"#,
            r#"("MEGT00N000HB.IMG",-1)"#,
            r#"("MEGT00N000HB.IMG",1.5)"#,
            r#"("MEGT00N000HB.IMG",NaN)"#,
            r#"("MEGT00N000HB.IMG")"#,
            r#"("MEGT00N000HB.IMG",1 <WORDS>)"#,
        ] {
            let text = format!("{}\n^IMAGE = {}", label(), pointer);
            let error = MolaMegdrMetadata::from_label(
                &parse_label(&text),
                "MEGT00N000HB",
            )
            .unwrap_err();
            assert!(matches!(error, MolaError::InvalidMetadata(_)), "^IMAGE={pointer}");
        }
    }

    #[test]
    fn detached_image_filename_pointer_is_zero_based_byte_origin() {
        for pointer in [
            r#""MEGT00N000HB.IMG""#,
            "MEGT00N000HB.IMG",
        ] {
            let text = format!("{}\n^IMAGE = {}", label(), pointer);
            let metadata = MolaMegdrMetadata::from_label(
                &parse_label(&text),
                "MEGT00N000HB",
            )
            .unwrap();
            assert_eq!(metadata.line_offset, 0, "pointer={pointer}");
        }
    }

    #[test]
    fn rejects_malformed_creation_time() {
        let text = label().replace(
            "PRODUCT_CREATION_TIME = 2003-03-21T00:00:00",
            "PRODUCT_CREATION_TIME = 2003/03/21 01:00:00",
        );
        let error = MolaMegdrMetadata::from_label(&parse_label(&text), "MEGT00N000HB").unwrap_err();
        assert!(matches!(error, MolaError::InvalidMetadata(_)));
    }

    #[test]
    fn rejects_impossible_creation_time_components() {
        for replacement in [
            ("2003-13-21T00:00:00", "invalid month"),
            ("2003-02-30T00:00:00", "invalid day"),
            ("2003-03-21T24:00:00", "invalid hour"),
            ("2003-03-21T00:60:00", "invalid minute"),
            ("2003-03-21T00:00:60", "invalid second"),
            ("2003-03-21T00:00:00Z", "invalid suffix"),
        ] {
            let text = label().replace(
                "PRODUCT_CREATION_TIME = 2003-03-21T00:00:00",
                &format!("PRODUCT_CREATION_TIME = {}", replacement.0),
            );
            let error =
                MolaMegdrMetadata::from_label(&parse_label(&text), "MEGT00N000HB").unwrap_err();
            assert!(matches!(error, MolaError::InvalidMetadata(_)), "{}", replacement.1);
        }
    }

    #[test]
    fn rejects_non_finite_optional_georeferencing_values() {
        for (key, value) in [
            ("MISSING_CONSTANT", "NaN"),
            ("MISSING_CONSTANT", "INF"),
            ("TILE_ORIGIN_LATITUDE", "NaN"),
            ("TILE_ORIGIN_LATITUDE", "INF"),
            ("TILE_ORIGIN_LONGITUDE", "NaN"),
            ("TILE_ORIGIN_LONGITUDE", "INF"),
        ] {
            let text = format!("{}\n{} = {}", label(), key, value);
            let error =
                MolaMegdrMetadata::from_label(&parse_label(&text), "MEGT00N000HB").unwrap_err();
            assert!(matches!(error, MolaError::InvalidMetadata(_)), "{key}={value}");
        }
    }

    #[test]
    fn rejects_tile_origin_outside_declared_footprint() {
        for (key, value) in [
            ("TILE_ORIGIN_LATITUDE", "1.0"),
            ("TILE_ORIGIN_LONGITUDE", "1.0"),
        ] {
            let text = format!("{}\n{} = {}", label(), key, value);
            let error =
                MolaMegdrMetadata::from_label(&parse_label(&text), "MEGT00N000HB").unwrap_err();
            assert!(matches!(error, MolaError::InvalidMetadata(_)), "{key}={value}");
        }
    }

    #[test]
    fn rejects_malformed_missing_constant_instead_of_dropping_it() {
        let text = label().replace("OFFSET = 0.0", "OFFSET = 0.0\nMISSING_CONSTANT = not-a-number");
        let error =
            MolaMegdrMetadata::from_label(&parse_label(&text), "MEGT00N000HB").unwrap_err();
        assert!(matches!(error, MolaError::InvalidMetadata(_)));
    }

    #[test]
    fn rejects_pre_final_creation_time() {
        let text = label().replace(
            "PRODUCT_CREATION_TIME = 2003-03-21T00:00:00",
            "PRODUCT_CREATION_TIME = 2003-03-20T23:59:59",
        );
        let error = MolaMegdrMetadata::from_label(&parse_label(&text), "MEGT00N000HB").unwrap_err();
        assert!(matches!(error, MolaError::InvalidMetadata(_)));
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
    fn rejects_mismatched_coordinate_registration() {
        let topography = MolaMegdrMetadata::from_label(
            &parse_label(&label()),
            "MEGT00N000HB",
        )
        .unwrap();

        for (field, replacement) in [
            (
                "COORDINATE_SYSTEM_TYPE = BODY-FIXED ROTATING",
                "COORDINATE_SYSTEM_TYPE = Body-fixed rotating",
            ),
            (
                "COORDINATE_SYSTEM_NAME = PLANETOCENTRIC",
                "COORDINATE_SYSTEM_NAME = Planetocentric",
            ),
            (
                "POSITIVE_LONGITUDE_DIRECTION = EAST",
                "POSITIVE_LONGITUDE_DIRECTION = east",
            ),
            (
                "MAP_PROJECTION_TYPE = SIMPLE CYLINDRICAL",
                "MAP_PROJECTION_TYPE = simple cylindrical",
            ),
            (
                "MAP_PROJECTION_ROTATION = 0.0",
                "MAP_PROJECTION_ROTATION = 0.0",
            ),
        ] {
            if field == replacement {
                continue;
            }
            let text = label()
                .replace("MEGT00N000HB", "MEGC00N000HB")
                .replace("MAP_TYPE = T", "MAP_TYPE = C")
                .replace("SAMPLE_BITS = 16", "SAMPLE_BITS = 8")
                .replace(field, replacement);
            let counts = MolaMegdrMetadata::from_label(
                &parse_label(&text),
                "MEGC00N000HB",
            )
            .unwrap();
            assert!(topography.validate_companion(&MolaMegdrProduct {
                metadata: counts,
                provenance: TerrainProvenance {
                    source_id: "test".into(),
                    source_revision: "test".into(),
                    coordinate_reference: "test".into(),
                },
                img_path: std::path::PathBuf::new(),
            }).is_err());
        }
    }

    #[test]
    fn rejects_companion_projection_rotation_mismatch() {
        let topography = MolaMegdrMetadata::from_label(
            &parse_label(&label()),
            "MEGT00N000HB",
        ).unwrap();
        let mut counts = MolaMegdrMetadata::from_label(
            &parse_label(
                &label()
                    .replace("MEGT00N000HB", "MEGC00N000HB")
                    .replace("MAP_TYPE = T", "MAP_TYPE = C")
                    .replace("SAMPLE_BITS = 16", "SAMPLE_BITS = 8"),
            ),
            "MEGC00N000HB",
        ).unwrap();
        counts.projection_rotation_deg = 0.001;
        assert!(topography.validate_companion(&MolaMegdrProduct {
            metadata: counts,
            provenance: TerrainProvenance {
                source_id: "test".into(),
                source_revision: "test".into(),
                coordinate_reference: "test".into(),
            },
            img_path: std::path::PathBuf::new(),
        }).is_err());
    }

    #[test]
    fn rejects_mismatched_projection_registration() {
        let topography = MolaMegdrMetadata::from_label(&parse_label(&label()), "MEGT00N000HB").unwrap();
        let count_label = label()
            .replace("MEGT00N000HB", "MEGC00N000HB")
            .replace("MAP_TYPE = T", "MAP_TYPE = C")
            .replace("SAMPLE_BITS = 16", "SAMPLE_BITS = 8")
            .replace("PRODUCT_CREATION_TIME = 2003-03-21T00:00:00", "PRODUCT_CREATION_TIME = 2003-03-21T00:00:01")
            .replace("MAXIMUM_LATITUDE = 0.015625", "MAXIMUM_LATITUDE = 0.015626")
            .replace("CENTER_LONGITUDE = 180.0", "CENTER_LONGITUDE = 179.999");
        let counts = MolaMegdrMetadata::from_label(&parse_label(&count_label), "MEGC00N000HB").unwrap();
        let product = MolaMegdrProduct {
            metadata: topography,
            provenance: TerrainProvenance {
                source_id: "MEGT00N000HB".into(),
                source_revision: "pds4-v1".into(),
                coordinate_reference: "IAU-2000 planetocentric latitude, east-positive longitude".into(),
            },
            img_path: std::path::PathBuf::from("unused"),
        };
        let count_product = MolaMegdrProduct {
            metadata: counts,
            provenance: product.provenance.clone(),
            img_path: std::path::PathBuf::from("unused"),
        };
        assert!(product.validate_companion(&count_product).is_err());
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
            .replace("SAMPLE_PROJECTION_OFFSET = 4.5", "SAMPLE_PROJECTION_OFFSET = 720.5")
            .replace("LINE_PROJECTION_OFFSET = 2.5", "LINE_PROJECTION_OFFSET = 360.5");
        let metadata = MolaMegdrMetadata::from_label(
            &parse_label(&text),
            "MEGT00N000HB",
        )
        .unwrap();
        assert_eq!(metadata.cell_for(89.875, 0.125), Ok((0, 0)));
        assert_eq!(metadata.cell_for(-89.875, 359.875), Ok((719, 1439)));
        assert_eq!(metadata.center_latitude_deg, 0.0);
    }

    #[test]
    fn rejects_nonzero_projection_rotation() {
        let text = label().replace(
            "MAP_PROJECTION_ROTATION = 0.0",
            "MAP_PROJECTION_ROTATION = 1.0",
        );
        let error = MolaMegdrMetadata::from_label(
            &parse_label(&text),
            "MEGT00N000HB",
        )
        .unwrap_err();
        assert!(matches!(error, MolaError::Unsupported(_)));
    }

    #[test]
    fn rejects_wrong_coordinate_system_type() {
        let text = label().replace(
            "COORDINATE_SYSTEM_TYPE = BODY-FIXED ROTATING",
            "COORDINATE_SYSTEM_TYPE = INERTIAL",
        );
        let error = MolaMegdrMetadata::from_label(
            &parse_label(&text),
            "MEGT00N000HB",
        )
        .unwrap_err();
        assert!(matches!(error, MolaError::InvalidMetadata(_)));
    }

    #[test]
    fn rejects_missing_projection_offsets() {
        let text = label().replace("LINE_PROJECTION_OFFSET = 2.5", "");
        let error = MolaMegdrMetadata::from_label(
            &parse_label(&text),
            "MEGT00N000HB",
        )
        .unwrap_err();
        assert!(matches!(error, MolaError::InvalidMetadata(_)));
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
    fn artifact_identities_retain_product_local_source_identity() {
        let topography = TerrainProvenance {
            source_id: "MEGT00N000HB".into(),
            source_revision: "pds4-v1".into(),
            coordinate_reference: "IAU-2000 planetocentric latitude, east-positive longitude".into(),
            content_digests: vec![
                ("detached-label".into(), "SHA-256".into(), "a".repeat(64)),
                ("raster-image".into(), "SHA-256".into(), "b".repeat(64)),
            ],
        };
        let counts = TerrainProvenance {
            source_id: "MEGC00N000HB".into(),
            source_revision: "pds4-v1".into(),
            coordinate_reference: "IAU-2000 planetocentric latitude, east-positive longitude".into(),
            content_digests: vec![
                ("detached-label".into(), "SHA-256".into(), "c".repeat(64)),
                ("raster-image".into(), "SHA-256".into(), "d".repeat(64)),
            ],
        };
        let topography = MolaMegdrProduct {
            metadata: MolaMegdrMetadata::from_label(&parse_label(&label()), "MEGT00N000HB").unwrap(),
            provenance: topography,
            label_sha256: "a".repeat(64),
            image_sha256: "b".repeat(64),
            img_path: std::path::PathBuf::new(),
        };
        let counts = MolaMegdrProduct {
            metadata: MolaMegdrMetadata::from_label(
                &parse_label(
                    &label()
                        .replace("MEGT00N000HB", "MEGC00N000HB")
                        .replace("MAP_TYPE = T", "MAP_TYPE = C")
                        .replace("SAMPLE_BITS = 16", "SAMPLE_BITS = 8"),
                ),
                "MEGC00N000HB",
            ).unwrap(),
            provenance: counts,
            label_sha256: "c".repeat(64),
            image_sha256: "d".repeat(64),
            img_path: std::path::PathBuf::new(),
        };

        let topography_artifacts = topography.artifact_identities().unwrap();
        let counts_artifacts = counts.artifact_identities().unwrap();
        assert_eq!(topography_artifacts.artifacts[0].source_id, "MEGT00N000HB");
        assert_eq!(counts_artifacts.artifacts[0].source_id, "MEGC00N000HB");
        assert_ne!(topography_artifacts, counts_artifacts);
    }

    #[test]
    fn zero_observation_count_cannot_become_terrain() {
        let mut path = std::env::temp_dir();
        path.push(format!("mola_adapter_zero_{}_label.lbl", std::process::id()));
        let mut img = path.clone();
        img.set_extension("img");
        let sample_label = label()
            .replace("WESTERNMOST_LONGITUDE = 0.0", "WESTERNMOST_LONGITUDE = 179.9")
            .replace("EASTERNMOST_LONGITUDE = 0.0625", "EASTERNMOST_LONGITUDE = 180.1");
        std::fs::write(&path, sample_label).unwrap();
        std::fs::write(&img, vec![0u8; 64]).unwrap();

        let product = MolaMegdrProduct::open(&path, &img, "MEGT00N000HB", "pds4-v1").unwrap();

        let mut count_path = path.clone();
        count_path.set_file_name(format!("mola_adapter_zero_count_{}_label.lbl", std::process::id()));
        let mut count_img = count_path.clone();
        count_img.set_extension("img");
        let count_label = label()
            .replace("WESTERNMOST_LONGITUDE = 0.0", "WESTERNMOST_LONGITUDE = 179.9")
            .replace("EASTERNMOST_LONGITUDE = 0.0625", "EASTERNMOST_LONGITUDE = 180.1")
            .replace("MEGT00N000HB", "MEGC00N000HB")
            .replace("MAP_TYPE = T", "MAP_TYPE = C")
            .replace("SAMPLE_BITS = 16", "SAMPLE_BITS = 8");
        std::fs::write(&count_path, count_label).unwrap();
        std::fs::write(&count_img, vec![0u8; 64]).unwrap();

        let counts = MolaMegdrProduct::open(&count_path, &count_img, "MEGC00N000HB", "pds4-v1").unwrap();
        let sample = product.sample_nearest_with_count(&counts, 0.0, 179.984375, 3.0).unwrap();

        assert_eq!(sample.quality, TerrainQuality::Missing);
        assert_eq!(sample.source_grid_cell, Some((1, 1)));
        assert_eq!(
            sample.sampling_method,
            TerrainSamplingMethod::NearestCellWithObservationCount
        );
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
        let sample_label = label()
            .replace("WESTERNMOST_LONGITUDE = 0.0", "WESTERNMOST_LONGITUDE = 179.9")
            .replace("EASTERNMOST_LONGITUDE = 0.0625", "EASTERNMOST_LONGITUDE = 180.1")
            .replace(
                "MAP_TYPE = T",
                "MAP_TYPE = T\n^IMAGE = (\"MEGT00N000HB.IMG\", 33 <BYTES>)",
            );
        std::fs::write(&path, sample_label).unwrap();
        let mut bytes = vec![0u8; 96];
        // The PDS byte pointer is 1-based: byte 33 => zero-based origin 32.
        // The first sampled cell remains at the old row/sample offset 18,
        // therefore its terrain bytes now begin at 32 + 18 = 50.
        bytes[50] = 0x03;
        bytes[51] = 0xE8;
        std::fs::write(&img, bytes).unwrap();
        let product = MolaMegdrProduct::open(&path, &img, "MEGT00N000HB", "pds4-v1").unwrap();
        let mut count_path = path.clone();
        count_path.set_file_name(format!("mola_adapter_count_{}_label.lbl", std::process::id()));
        let mut count_img = count_path.clone();
        count_img.set_extension("img");
        let count_label = label()
            .replace("WESTERNMOST_LONGITUDE = 0.0", "WESTERNMOST_LONGITUDE = 179.9")
            .replace("EASTERNMOST_LONGITUDE = 0.0625", "EASTERNMOST_LONGITUDE = 180.1")
            .replace("MEGT00N000HB", "MEGC00N000HB")
            .replace("MAP_TYPE = T", "MAP_TYPE = C")
            .replace("SAMPLE_BITS = 16", "SAMPLE_BITS = 8")
            .replace(
                "MAP_TYPE = C",
                "MAP_TYPE = C\n^IMAGE = (\"MEGC00N000HB.IMG\", 33 <BYTES>)",
            );
        std::fs::write(&count_path, count_label).unwrap();
        let mut count_bytes = vec![0u8; 96];
        // The count byte uses the same detached byte origin: 32 + 17 = 49.
        count_bytes[49] = 1;
        std::fs::write(&count_img, count_bytes).unwrap();
        let counts = MolaMegdrProduct::open(&count_path, &count_img, "MEGC00N000HB", "pds4-v1").unwrap();
        let sample = product.sample_nearest_with_count(&counts, 0.0, 179.984375, 3.0).unwrap();
        assert_eq!(sample.elevation_m, Some(1000.0));
        assert_eq!(sample.source_grid_cell, Some((1, 1)));
        assert_eq!(
            sample.sampling_method,
            TerrainSamplingMethod::NearestCellWithObservationCount
        );
        assert!(sample.is_usable());
        let _ = std::fs::remove_file(count_path);
        let _ = std::fs::remove_file(count_img);
        assert_eq!(sample.quality, TerrainQuality::Measured);
        assert_eq!(sample.vertical_datum, TerrainVerticalDatum::AreoidRelative);
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(img);
    }
}