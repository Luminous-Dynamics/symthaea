// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Synthetic-only validation of global-field to local REBCO tape-frame transforms.
//! This binary does not fit an Ic surface or qualify a magnet/coil.

use jsonschema::validator_for;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

const DIALECT: &str = "https://json-schema.org/draft/2020-12/schema";
const SCHEMA_VERSION: &str = "sc-rebco-local-tape-frame.v0.1";
const CLAIM_CEILING: &str = "local-frame-transform-and-vector-geometry-semantics-only";
const DEFAULT_SCHEMA: &str = "data/schemas/sc_rebco_local_tape_frame_v0_1.schema.json";
const DEFAULT_FIXTURE: &str =
    "tools/superconductivity-discovery-harness/fixtures/rebco_local_tape_frame_smoke_v0_1.json";
const FRAME_TOLERANCE: f64 = 1.0e-6;
const VECTOR_TOLERANCE: f64 = 1.0e-9;
const ZERO_FIELD_EPSILON_T: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    fn to_value(self) -> Value {
        json!([self.x, self.y, self.z])
    }
}

#[derive(Debug, Clone, Copy)]
struct Mat3 {
    rows: [[f64; 3]; 3],
}

impl Mat3 {
    fn determinant(self) -> f64 {
        let m = self.rows;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    fn column(self, index: usize) -> Vec3 {
        Vec3 {
            x: self.rows[0][index],
            y: self.rows[1][index],
            z: self.rows[2][index],
        }
    }

    fn orthogonality_error(self) -> f64 {
        let columns = [self.column(0), self.column(1), self.column(2)];
        let mut maximum = 0.0_f64;
        for (index, column) in columns.iter().enumerate() {
            maximum = maximum.max((column.norm() - 1.0).abs());
            for other in columns.iter().skip(index + 1) {
                maximum = maximum.max(column.dot(*other).abs());
            }
        }
        maximum
    }

    // Matrix columns are local basis axes expressed in global coordinates.
    // Therefore B_local = R^T * B_global.
    fn global_to_local(self, global: Vec3) -> Vec3 {
        Vec3 {
            x: self.column(0).dot(global),
            y: self.column(1).dot(global),
            z: self.column(2).dot(global),
        }
    }

    fn local_to_global(self, local: Vec3) -> Vec3 {
        Vec3 {
            x: self.rows[0][0] * local.x
                + self.rows[0][1] * local.y
                + self.rows[0][2] * local.z,
            y: self.rows[1][0] * local.x
                + self.rows[1][1] * local.y
                + self.rows[1][2] * local.z,
            z: self.rows[2][0] * local.x
                + self.rows[2][1] * local.y
                + self.rows[2][2] * local.z,
        }
    }
}

fn fail(path: &str, message: impl AsRef<str>) -> String {
    format!("{path}: {}", message.as_ref())
}

fn object<'a>(value: &'a Value, path: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| fail(path, "must be an object"))
}

fn field<'a>(value: &'a Value, key: &str, path: &str) -> Result<&'a Value, String> {
    object(value, path)?
        .get(key)
        .ok_or_else(|| fail(path, format!("missing property {key}")))
}

fn text_field<'a>(value: &'a Value, key: &str, path: &str) -> Result<&'a str, String> {
    field(value, key, path)?
        .as_str()
        .filter(|item| !item.trim().is_empty())
        .ok_or_else(|| fail(path, format!("{key} must be a non-empty string")))
}

fn finite_number(value: &Value, path: &str) -> Result<f64, String> {
    let number = value
        .as_f64()
        .ok_or_else(|| fail(path, "must be a finite number"))?;
    if !number.is_finite() {
        return Err(fail(path, "must be finite"));
    }
    Ok(number)
}

fn vector3(value: &Value, path: &str) -> Result<Vec3, String> {
    let values = value
        .as_array()
        .ok_or_else(|| fail(path, "must be an array of three finite numbers"))?;
    if values.len() != 3 {
        return Err(fail(path, "must contain exactly three finite numbers"));
    }
    Ok(Vec3 {
        x: finite_number(&values[0], &format!("{path}[0]"))?,
        y: finite_number(&values[1], &format!("{path}[1]"))?,
        z: finite_number(&values[2], &format!("{path}[2]"))?,
    })
}

fn matrix3(value: &Value, path: &str) -> Result<Mat3, String> {
    let rows = value
        .as_array()
        .ok_or_else(|| fail(path, "must be a 3x3 matrix"))?;
    if rows.len() != 3 {
        return Err(fail(path, "must have exactly three rows"));
    }
    let mut matrix = [[0.0; 3]; 3];
    for (row_index, row) in rows.iter().enumerate() {
        let entries = row
            .as_array()
            .ok_or_else(|| fail(path, "each matrix row must contain three numbers"))?;
        if entries.len() != 3 {
            return Err(fail(path, "each matrix row must contain exactly three numbers"));
        }
        for (column_index, entry) in entries.iter().enumerate() {
            matrix[row_index][column_index] =
                finite_number(entry, &format!("{path}[{row_index}][{column_index}]"))?;
        }
    }
    Ok(Mat3 { rows: matrix })
}

fn validate_digest(value: &str, path: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(fail(path, "must be lowercase 64-character SHA-256 hex"));
    }
    Ok(())
}

fn transform_station(station: &Value, index: usize) -> Result<Value, String> {
    let path = format!("stations[{index}]");
    let rotation = matrix3(
        field(station, "local_to_global_rotation_matrix", &path)?,
        &format!("{path}.local_to_global_rotation_matrix"),
    )?;
    let determinant = rotation.determinant();
    let orthogonality_error = rotation.orthogonality_error();
    if orthogonality_error > FRAME_TOLERANCE {
        return Err(fail(
            &path,
            format!("local tape frame is not orthonormal: maximum error {orthogonality_error}"),
        ));
    }
    if (determinant - 1.0).abs() > FRAME_TOLERANCE {
        return Err(fail(
            &path,
            format!(
                "local tape frame must be right-handed with determinant +1; found {determinant}"
            ),
        ));
    }

    let global = vector3(
        field(station, "global_field_vector_t", &path)?,
        &format!("{path}.global_field_vector_t"),
    )?;
    let local = rotation.global_to_local(global);
    let reconstructed = rotation.local_to_global(local);
    let residual = Vec3 {
        x: reconstructed.x - global.x,
        y: reconstructed.y - global.y,
        z: reconstructed.z - global.z,
    }
    .norm();
    if residual > VECTOR_TOLERANCE * global.norm().max(1.0) {
        return Err(fail(&path, format!("inverse transform residual too large: {residual}")));
    }

    let magnitude_t = global.norm();
    let orientation_defined = magnitude_t > ZERO_FIELD_EPSILON_T;
    let in_plane_magnitude = local.x.hypot(local.y);
    let theta = orientation_defined.then(|| in_plane_magnitude.atan2(local.z));
    let phi = (orientation_defined && in_plane_magnitude > ZERO_FIELD_EPSILON_T)
        .then(|| local.y.atan2(local.x));

    Ok(json!({
        "station_id": text_field(station, "station_id", &path)?,
        "reel_id": text_field(station, "reel_id", &path)?,
        "segment_id": text_field(station, "segment_id", &path)?,
        "coordinate_frame_id": text_field(station, "coordinate_frame_id", &path)?,
        "longitudinal_position_m": finite_number(
            field(station, "longitudinal_position_m", &path)?,
            &format!("{path}.longitudinal_position_m")
        )?,
        "global_field_vector_t": global.to_value(),
        "local_field_vector_t": local.to_value(),
        "field_magnitude_t": magnitude_t,
        "theta_from_local_normal_rad": theta,
        "phi_from_longitudinal_toward_width_rad": phi,
        "orientation_defined": orientation_defined,
        "transform_uncertainty_rad": finite_number(
            field(station, "transform_uncertainty_rad", &path)?,
            &format!("{path}.transform_uncertainty_rad")
        )?,
        "frame_determinant": determinant,
        "frame_orthogonality_error": orthogonality_error,
        "inverse_transform_residual_t": residual,
        "source_artifact_sha256": text_field(station, "source_artifact_sha256", &path)?
    }))
}

fn audit(schema_bytes: &[u8], fixture_bytes: &[u8]) -> Result<Value, String> {
    let schema: Value = serde_json::from_slice(schema_bytes)
        .map_err(|error| format!("schema JSON parse failed: {error}"))?;
    let fixture: Value = serde_json::from_slice(fixture_bytes)
        .map_err(|error| format!("fixture JSON parse failed: {error}"))?;
    if schema.get("$schema").and_then(Value::as_str) != Some(DIALECT) {
        return Err(format!("schema must declare {DIALECT}"));
    }
    let validator = validator_for(&schema)
        .map_err(|error| format!("schema compilation failed: {error}"))?;
    if let Some(error) = validator.iter_errors(&fixture).next() {
        return Err(format!(
            "fixture schema validation failed: {} at {}",
            error,
            error.instance_path()
        ));
    }
    if text_field(&fixture, "schema_version", "$")? != SCHEMA_VERSION
        || text_field(&fixture, "evidence_class", "$")? != "synthetic"
        || text_field(&fixture, "claim_ceiling", "$")? != CLAIM_CEILING
    {
        return Err("fixture version, evidence class, or claim ceiling does not match v0.1".to_owned());
    }

    let convention = field(&fixture, "frame_convention", "$")?;
    let expected_convention = json!({
        "matrix_semantics": "columns_are_local_axes_expressed_in_global_coordinates",
        "basis_axis_order": ["longitudinal", "width", "normal"],
        "handedness": "right_handed",
        "theta_convention": "angle_from_local_tape_normal",
        "phi_convention": "azimuth_in_tape_plane_from_longitudinal_toward_width",
        "angle_unit": "radian"
    });
    if convention != &expected_convention {
        return Err(fail(
            "$.frame_convention",
            "must exactly match the declared v0.1 local tape-frame convention",
        ));
    }

    let stations = field(&fixture, "stations", "$")?
        .as_array()
        .ok_or_else(|| fail("$.stations", "must be an array"))?;
    if stations.is_empty() {
        return Err(fail("$.stations", "must contain at least one station"));
    }
    let mut ids = BTreeSet::new();
    let mut transformed = Vec::with_capacity(stations.len());
    for (index, station) in stations.iter().enumerate() {
        let path = format!("stations[{index}]");
        let id = text_field(station, "station_id", &path)?;
        if !ids.insert(id.to_owned()) {
            return Err(fail(&path, "duplicate station_id"));
        }
        let position = finite_number(
            field(station, "longitudinal_position_m", &path)?,
            &format!("{path}.longitudinal_position_m"),
        )?;
        if position < 0.0 {
            return Err(fail(&path, "longitudinal_position_m must be non-negative"));
        }
        let uncertainty = finite_number(
            field(station, "transform_uncertainty_rad", &path)?,
            &format!("{path}.transform_uncertainty_rad"),
        )?;
        if uncertainty < 0.0 {
            return Err(fail(&path, "transform_uncertainty_rad must be non-negative"));
        }
        validate_digest(
            text_field(station, "source_artifact_sha256", &path)?,
            &format!("{path}.source_artifact_sha256"),
        )?;
        transformed.push(transform_station(station, index)?);
    }

    Ok(json!({
        "schema_version": "sc-rebco-local-tape-frame-audit-receipt.v0.1",
        "schema_sha256": format!("{:x}", Sha256::digest(schema_bytes)),
        "fixture_sha256": format!("{:x}", Sha256::digest(fixture_bytes)),
        "station_count": transformed.len(),
        "stations": transformed,
        "claim_ceiling": CLAIM_CEILING,
        "physical_measurement_or_material_model": "NOT_PERFORMED",
        "product_or_coil_qualification": "NOT_PROVIDED"
    }))
}

fn parse_args() -> Result<(PathBuf, PathBuf), String> {
    let mut schema = PathBuf::from(DEFAULT_SCHEMA);
    let mut fixture = PathBuf::from(DEFAULT_FIXTURE);
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {argument}"))?;
        match argument.as_str() {
            "--schema" => schema = PathBuf::from(value),
            "--fixture" => fixture = PathBuf::from(value),
            other => return Err(format!("unsupported argument {other}")),
        }
    }
    Ok((schema, fixture))
}

fn run() -> Result<(), String> {
    let (schema_path, fixture_path) = parse_args()?;
    let schema_bytes = fs::read(&schema_path)
        .map_err(|error| format!("cannot read schema {}: {error}", schema_path.display()))?;
    let fixture_bytes = fs::read(&fixture_path)
        .map_err(|error| format!("cannot read fixture {}: {error}", fixture_path.display()))?;
    let receipt = audit(&schema_bytes, &fixture_bytes)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&receipt)
            .map_err(|error| format!("receipt serialization failed: {error}"))?
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("sc_rebco_local_tape_frame_audit: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        let schema = include_bytes!("../../../../data/schemas/sc_rebco_local_tape_frame_v0_1.schema.json");
        let bytes = include_bytes!("../../fixtures/rebco_local_tape_frame_smoke_v0_1.json");
        serde_json::from_slice(bytes).expect("local tape-frame fixture must parse")
    }

    fn schema_bytes() -> &'static [u8] {
        include_bytes!("../../../../data/schemas/sc_rebco_local_tape_frame_v0_1.schema.json")
    }

    fn fixture_bytes() -> &'static [u8] {
        include_bytes!("../../fixtures/rebco_local_tape_frame_smoke_v0_1.json")
    }

    fn audit_fixture(value: &Value) -> Result<Value, String> {
        let fixture = serde_json::to_vec(value).expect("fixture serializes");
        audit(schema_bytes(), &fixture)
    }

    #[test]
    fn identity_frame_maps_global_axes_to_same_local_axes() {
        let receipt = audit(schema_bytes(), fixture_bytes()).expect("fixture audits");
        let station = &receipt["stations"][0];
        assert_eq!(station["local_field_vector_t"], json!([5.0, 0.0, 0.0]));
        let theta = station["theta_from_local_normal_rad"].as_f64().unwrap();
        assert!((theta - std::f64::consts::FRAC_PI_2).abs() < 1.0e-12);
        assert_eq!(station["phi_from_longitudinal_toward_width_rad"], json!(0.0));
    }

    #[test]
    fn curved_station_rotates_same_global_field_to_local_normal() {
        let receipt = audit(schema_bytes(), fixture_bytes()).expect("fixture audits");
        let station = &receipt["stations"][1];
        assert_eq!(station["local_field_vector_t"], json!([0.0, 0.0, 5.0]));
        let theta = station["theta_from_local_normal_rad"].as_f64().unwrap();
        assert!(theta.abs() < 1.0e-12);
        assert_eq!(station["phi_from_longitudinal_toward_width_rad"], Value::Null);
    }

    #[test]
    fn zero_field_has_no_defined_orientation() {
        let receipt = audit(schema_bytes(), fixture_bytes()).expect("fixture audits");
        let station = &receipt["stations"][2];
        assert_eq!(station["field_magnitude_t"], 0.0);
        assert_eq!(station["orientation_defined"], false);
        assert_eq!(station["theta_from_local_normal_rad"], Value::Null);
        assert_eq!(station["phi_from_longitudinal_toward_width_rad"], Value::Null);
    }

    #[test]
    fn same_global_field_has_different_local_angles_along_curved_tape() {
        let receipt = audit(schema_bytes(), fixture_bytes()).expect("fixture audits");
        let straight = receipt["stations"][0]["theta_from_local_normal_rad"].as_f64().unwrap();
        let curved = receipt["stations"][1]["theta_from_local_normal_rad"].as_f64().unwrap();
        assert_ne!(straight, curved);
    }

    #[test]
    fn rejects_left_handed_reflection_frame() {
        let mut value = fixture();
        value["stations"][0]["local_to_global_rotation_matrix"] =
            json!([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        assert!(audit_fixture(&value).unwrap_err().contains("determinant +1"));
    }

    #[test]
    fn rejects_non_orthogonal_frame() {
        let mut value = fixture();
        value["stations"][0]["local_to_global_rotation_matrix"] =
            json!([[1, 0.1, 0], [0, 1, 0], [0, 0, 1]]);
        assert!(audit_fixture(&value).unwrap_err().contains("not orthonormal"));
    }

    #[test]
    fn rejects_wrong_matrix_shape() {
        let mut value = fixture();
        value["stations"][0]["local_to_global_rotation_matrix"] = json!([[1, 0], [0, 1]]);
        assert!(audit_fixture(&value).unwrap_err().contains("exactly three rows"));
    }

    #[test]
    fn rejects_duplicate_station_ids() {
        let mut value = fixture();
        let first_id = value["stations"][0]["station_id"].clone();
        value["stations"][1]["station_id"] = first_id;
        assert!(audit_fixture(&value).unwrap_err().contains("duplicate station_id"));
    }

    #[test]
    fn rejects_negative_frame_uncertainty() {
        let mut value = fixture();
        value["stations"][0]["transform_uncertainty_rad"] = json!(-0.1);
        assert!(audit_fixture(&value).unwrap_err().contains("must be non-negative"));
    }

    #[test]
    fn rejects_malformed_source_digest() {
        let mut value = fixture();
        value["stations"][0]["source_artifact_sha256"] = json!("ABC");
        assert!(audit_fixture(&value).unwrap_err().contains("lowercase 64-character SHA-256"));
    }

    #[test]
    fn same_input_receipt_is_deterministic() {
        let first = audit(schema_bytes(), fixture_bytes()).expect("first audit");
        let second = audit(schema_bytes(), fixture_bytes()).expect("second audit");
        assert_eq!(first, second);
    }
}
