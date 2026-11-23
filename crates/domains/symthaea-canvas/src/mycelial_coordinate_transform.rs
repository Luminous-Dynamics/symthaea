// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Version-aware binding from OME-NGFF spatial axes to a mycelial graph frame.
//!
//! OME-Zarr v0.5 is treated as the published profile (scale / translation).
//! RFC-5 v0.6 implementation-profile transforms add affine, rotation and axis
//! permutation operations. Nonlinear fields are retained as provenance-bearing
//! references, but are not evaluated until their samples are independently resolved.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::mycelial_graph_interchange::MycelialGraphInterchangeSidecar;
use crate::mycelial_reconstruction_metrics::{
    AnnotatedMycelialSkeleton, ReconstructionDimensionality, ReconstructionLengthUnit,
    ReconstructionPoint3,
};

pub const MYCELIAL_COORDINATE_TRANSFORM_SCHEMA_VERSION: u16 = 1;
pub const MAX_MYCELIAL_COORDINATE_SYSTEMS: usize = 16;
pub const MAX_MYCELIAL_COORDINATE_TRANSFORMS: usize = 64;
pub const MAX_MYCELIAL_COORDINATE_ID_BYTES: usize = 128;
pub const MAX_MYCELIAL_COORDINATE_FIELD_PATH_BYTES: usize = 2_048;
const DETERMINANT_EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialOmeNgffTransformProfile {
    /// Published OME-Zarr 0.5 profile: identity, scale and translation only.
    OmeZarrV05Published,
    /// RFC-5 / OME-Zarr v0.6 implementation profile; not a claim of final conformance.
    Rfc5V06Implementation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialSpatialAxisRole {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialSpatialAxis {
    pub name: String,
    pub role: MycelialSpatialAxisRole,
    /// Unit declared for this spatial axis in OME-NGFF metadata.
    pub unit: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialSpatialCoordinateSystem {
    pub system_id: String,
    pub frame_id: String,
    /// Spatial axes in the exact vector order used by the coordinate system.
    pub axes: Vec<MycelialSpatialAxis>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum MycelialSpatialTransformOperation {
    Identity,
    Scale { factors: Vec<f64> },
    Translation { offsets: Vec<f64> },
    /// Matrix has one row per output axis and one column per input axis plus offset.
    Affine { matrix: Vec<Vec<f64>> },
    Rotation { matrix: Vec<Vec<f64>> },
    /// source_axis_for_output[i] selects the input axis used for output axis i.
    AxisPermutation { source_axis_for_output: Vec<usize> },
    /// Field references are validated and digested but not sampled by this evaluator.
    Displacements {
        field_path: String,
        field_digest: String,
        interpolation: String,
    },
    Coordinates {
        field_path: String,
        field_digest: String,
        interpolation: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialSpatialTransformStep {
    pub transform_id: String,
    pub input_system_id: String,
    pub output_system_id: String,
    pub operation: MycelialSpatialTransformOperation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialCoordinateTransformManifest {
    pub schema_version: u16,
    pub profile: MycelialOmeNgffTransformProfile,
    pub source_content_digest: String,
    /// Digest of full source axes metadata, including non-spatial axes.
    pub axes_metadata_digest: String,
    /// Digest of declared transformations and referenced affine / field arrays.
    pub declared_transform_digest: String,
    pub source_system_id: String,
    pub graph_system_id: String,
    pub coordinate_systems: Vec<MycelialSpatialCoordinateSystem>,
    /// Ordered transformation chain from source coordinates to graph coordinates.
    pub transformations: Vec<MycelialSpatialTransformStep>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialTransformedPoint {
    pub position: ReconstructionPoint3,
    pub coordinate_frame_id: String,
    pub length_unit: ReconstructionLengthUnit,
    pub coordinate_transform_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MycelialCoordinateTransformError {
    InvalidSchemaVersion(u16),
    InvalidMetadata,
    InvalidCoordinateSystem(String),
    DuplicateCoordinateSystem(String),
    InvalidTransform(String),
    DisconnectedTransformChain(String),
    UnsupportedOperationForProfile(String),
    NonInvertibleTransform(String),
    UnsupportedUnresolvedField(String),
    UnknownLengthUnit(String),
    IncompatibleGraphDimensionality,
    GraphFrameMismatch,
    SourceDigestMismatch,
    AxesDigestMismatch,
    TransformDigestMismatch,
    PointDimensionMismatch,
    NonFinitePoint,
}

impl fmt::Display for MycelialCoordinateTransformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchemaVersion(version) => {
                write!(f, "unsupported mycelial coordinate-transform schema version {version}")
            }
            Self::InvalidMetadata => write!(f, "invalid coordinate-transform manifest metadata"),
            Self::InvalidCoordinateSystem(id) => {
                write!(f, "invalid spatial coordinate system {id}")
            }
            Self::DuplicateCoordinateSystem(id) => {
                write!(f, "duplicate spatial coordinate system {id}")
            }
            Self::InvalidTransform(id) => write!(f, "invalid spatial transform {id}"),
            Self::DisconnectedTransformChain(id) => {
                write!(f, "coordinate-transform chain is disconnected at {id}")
            }
            Self::UnsupportedOperationForProfile(id) => {
                write!(f, "transform {id} is not supported by the declared OME-NGFF profile")
            }
            Self::NonInvertibleTransform(id) => {
                write!(f, "spatial transform {id} is singular or non-invertible")
            }
            Self::UnsupportedUnresolvedField(id) => {
                write!(f, "nonlinear field transform {id} has no resolved field samples")
            }
            Self::UnknownLengthUnit(unit) => {
                write!(f, "unsupported or incompatible spatial unit {unit}")
            }
            Self::IncompatibleGraphDimensionality => {
                write!(f, "source spatial dimensionality does not match graph dimensionality")
            }
            Self::GraphFrameMismatch => {
                write!(f, "coordinate-transform target frame does not match graph coordinate frame")
            }
            Self::SourceDigestMismatch => {
                write!(f, "coordinate-transform source digest does not match sidecar source data")
            }
            Self::AxesDigestMismatch => {
                write!(f, "coordinate-transform axes digest does not match sidecar axes metadata")
            }
            Self::TransformDigestMismatch => {
                write!(f, "coordinate-transform digest does not match sidecar transform evidence")
            }
            Self::PointDimensionMismatch => {
                write!(f, "input point does not match the source spatial-axis count")
            }
            Self::NonFinitePoint => write!(f, "input point contains a non-finite coordinate"),
        }
    }
}

impl Error for MycelialCoordinateTransformError {}

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value.len() <= MAX_MYCELIAL_COORDINATE_ID_BYTES
        && value.bytes().all(|byte| !byte.is_ascii_control())
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_field_path(value: &str) -> bool {
    !value.trim().is_empty()
        && value.len() <= MAX_MYCELIAL_COORDINATE_FIELD_PATH_BYTES
        && !value
            .bytes()
            .any(|byte| byte.is_ascii_control() || byte.is_ascii_whitespace())
        && !value.split(['/', '\\']).any(|component| component == "..")
        && !value.starts_with('/')
        && !value.starts_with('\\')
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    use fmt::Write as _;
    for byte in bytes {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn determinant(matrix: &[Vec<f64>]) -> Option<f64> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    let mut values = matrix.to_vec();
    let mut sign = 1.0_f64;
    let mut result = 1.0_f64;
    for pivot in 0..n {
        let pivot_row = (pivot..n).max_by(|left, right| {
            values[*left][pivot]
                .abs()
                .total_cmp(&values[*right][pivot].abs())
        })?;
        if values[pivot_row][pivot].abs() <= DETERMINANT_EPSILON {
            return Some(0.0);
        }
        if pivot_row != pivot {
            values.swap(pivot_row, pivot);
            sign = -sign;
        }
        let pivot_value = values[pivot][pivot];
        result *= pivot_value;
        for row in (pivot + 1)..n {
            let factor = values[row][pivot] / pivot_value;
            for col in (pivot + 1)..n {
                values[row][col] -= factor * values[pivot][col];
            }
        }
    }
    Some(result * sign)
}

fn validate_axes(system: &MycelialSpatialCoordinateSystem) -> bool {
    if !valid_id(&system.system_id)
        || !valid_id(&system.frame_id)
        || !(system.axes.len() == 2 || system.axes.len() == 3)
    {
        return false;
    }
    let mut names = BTreeSet::new();
    let mut roles = BTreeSet::new();
    for axis in &system.axes {
        if !valid_id(&axis.name)
            || !valid_id(&axis.unit)
            || !names.insert(axis.name.as_str())
            || !roles.insert(axis.role)
        {
            return false;
        }
    }
    let expected: BTreeSet<_> = if system.axes.len() == 2 {
        [MycelialSpatialAxisRole::X, MycelialSpatialAxisRole::Y]
            .into_iter()
            .collect()
    } else {
        [
            MycelialSpatialAxisRole::X,
            MycelialSpatialAxisRole::Y,
            MycelialSpatialAxisRole::Z,
        ]
        .into_iter()
        .collect()
    };
    roles == expected
}

fn transform_supported(
    profile: MycelialOmeNgffTransformProfile,
    operation: &MycelialSpatialTransformOperation,
) -> bool {
    match profile {
        MycelialOmeNgffTransformProfile::OmeZarrV05Published => matches!(
            operation,
            MycelialSpatialTransformOperation::Identity
                | MycelialSpatialTransformOperation::Scale { .. }
                | MycelialSpatialTransformOperation::Translation { .. }
        ),
        MycelialOmeNgffTransformProfile::Rfc5V06Implementation => true,
    }
}

fn validate_step(
    step: &MycelialSpatialTransformStep,
    input: &MycelialSpatialCoordinateSystem,
    output: &MycelialSpatialCoordinateSystem,
    profile: MycelialOmeNgffTransformProfile,
) -> Result<(), MycelialCoordinateTransformError> {
    if !valid_id(&step.transform_id)
        || step.input_system_id != input.system_id
        || step.output_system_id != output.system_id
    {
        return Err(MycelialCoordinateTransformError::InvalidTransform(
            step.transform_id.clone(),
        ));
    }
    if !transform_supported(profile, &step.operation) {
        return Err(
            MycelialCoordinateTransformError::UnsupportedOperationForProfile(
                step.transform_id.clone(),
            ),
        );
    }
    let input_count = input.axes.len();
    let output_count = output.axes.len();
    match &step.operation {
        MycelialSpatialTransformOperation::Identity => {
            if input_count != output_count
                || input.axes.iter().map(|axis| axis.role).collect::<Vec<_>>()
                    != output.axes.iter().map(|axis| axis.role).collect::<Vec<_>>()
            {
                return Err(MycelialCoordinateTransformError::InvalidTransform(
                    step.transform_id.clone(),
                ));
            }
        }
        MycelialSpatialTransformOperation::Scale { factors } => {
            if input_count != output_count
                || input.axes.iter().map(|axis| axis.role).collect::<Vec<_>>()
                    != output.axes.iter().map(|axis| axis.role).collect::<Vec<_>>()
                || factors.len() != input_count
                || factors.iter().any(|value| !value.is_finite() || *value <= 0.0)
            {
                return Err(MycelialCoordinateTransformError::InvalidTransform(
                    step.transform_id.clone(),
                ));
            }
        }
        MycelialSpatialTransformOperation::Translation { offsets } => {
            if input_count != output_count
                || input.axes.iter().map(|axis| axis.role).collect::<Vec<_>>()
                    != output.axes.iter().map(|axis| axis.role).collect::<Vec<_>>()
                || offsets.len() != output_count
                || offsets.iter().any(|value| !value.is_finite())
            {
                return Err(MycelialCoordinateTransformError::InvalidTransform(
                    step.transform_id.clone(),
                ));
            }
        }
        MycelialSpatialTransformOperation::Affine { matrix } => {
            if input_count != output_count
                || matrix.len() != output_count
                || matrix.iter().any(|row| {
                    row.len() != input_count + 1 || row.iter().any(|value| !value.is_finite())
                })
            {
                return Err(MycelialCoordinateTransformError::InvalidTransform(
                    step.transform_id.clone(),
                ));
            }
            let linear: Vec<Vec<f64>> = matrix
                .iter()
                .map(|row| row[..input_count].to_vec())
                .collect();
            if determinant(&linear).is_none_or(|value| value.abs() <= DETERMINANT_EPSILON) {
                return Err(MycelialCoordinateTransformError::NonInvertibleTransform(
                    step.transform_id.clone(),
                ));
            }
        }
        MycelialSpatialTransformOperation::Rotation { matrix } => {
            if input_count != output_count
                || matrix.len() != output_count
                || matrix.iter().any(|row| {
                    row.len() != input_count || row.iter().any(|value| !value.is_finite())
                })
            {
                return Err(MycelialCoordinateTransformError::InvalidTransform(
                    step.transform_id.clone(),
                ));
            }
            if determinant(matrix).is_none_or(|value| value.abs() <= DETERMINANT_EPSILON) {
                return Err(MycelialCoordinateTransformError::NonInvertibleTransform(
                    step.transform_id.clone(),
                ));
            }
        }
        MycelialSpatialTransformOperation::AxisPermutation {
            source_axis_for_output,
        } => {
            let mut seen = BTreeSet::new();
            if input_count != output_count
                || source_axis_for_output.len() != output_count
                || source_axis_for_output
                    .iter()
                    .any(|index| *index >= input_count || !seen.insert(*index))
                || source_axis_for_output.iter().enumerate().any(
                    |(output_index, input_index)| {
                        output.axes[output_index].role != input.axes[*input_index].role
                    },
                )
            {
                return Err(MycelialCoordinateTransformError::InvalidTransform(
                    step.transform_id.clone(),
                ));
            }
        }
        MycelialSpatialTransformOperation::Displacements {
            field_path,
            field_digest,
            interpolation,
        }
        | MycelialSpatialTransformOperation::Coordinates {
            field_path,
            field_digest,
            interpolation,
        } => {
            if !valid_field_path(field_path)
                || !valid_digest(field_digest)
                || !matches!(interpolation.as_str(), "nearest" | "linear" | "cubic")
            {
                return Err(MycelialCoordinateTransformError::InvalidTransform(
                    step.transform_id.clone(),
                ));
            }
        }
    }
    Ok(())
}

impl MycelialCoordinateTransformManifest {
    pub fn validate(&self) -> Result<(), MycelialCoordinateTransformError> {
        if self.schema_version != MYCELIAL_COORDINATE_TRANSFORM_SCHEMA_VERSION
            || !valid_digest(&self.source_content_digest)
            || !valid_digest(&self.axes_metadata_digest)
            || !valid_digest(&self.declared_transform_digest)
            || !valid_id(&self.source_system_id)
            || !valid_id(&self.graph_system_id)
            || self.coordinate_systems.is_empty()
            || self.coordinate_systems.len() > MAX_MYCELIAL_COORDINATE_SYSTEMS
            || self.transformations.is_empty()
            || self.transformations.len() > MAX_MYCELIAL_COORDINATE_TRANSFORMS
        {
            return Err(MycelialCoordinateTransformError::InvalidMetadata);
        }
        let mut systems = BTreeMap::new();
        for system in &self.coordinate_systems {
            if !validate_axes(system) {
                return Err(MycelialCoordinateTransformError::InvalidCoordinateSystem(
                    system.system_id.clone(),
                ));
            }
            if systems.insert(system.system_id.as_str(), system).is_some() {
                return Err(MycelialCoordinateTransformError::DuplicateCoordinateSystem(
                    system.system_id.clone(),
                ));
            }
        }
        let source = systems.get(self.source_system_id.as_str()).ok_or_else(|| {
            MycelialCoordinateTransformError::InvalidCoordinateSystem(
                self.source_system_id.clone(),
            )
        })?;
        let target = systems.get(self.graph_system_id.as_str()).ok_or_else(|| {
            MycelialCoordinateTransformError::InvalidCoordinateSystem(
                self.graph_system_id.clone(),
            )
        })?;
        if source.axes.len() != target.axes.len() {
            return Err(MycelialCoordinateTransformError::IncompatibleGraphDimensionality);
        }

        let mut current_id = self.source_system_id.as_str();
        let mut visited = BTreeSet::new();
        visited.insert(current_id);
        for step in &self.transformations {
            if step.input_system_id != current_id {
                return Err(MycelialCoordinateTransformError::DisconnectedTransformChain(
                    step.transform_id.clone(),
                ));
            }
            let input = systems.get(step.input_system_id.as_str()).ok_or_else(|| {
                MycelialCoordinateTransformError::InvalidCoordinateSystem(
                    step.input_system_id.clone(),
                )
            })?;
            let output = systems.get(step.output_system_id.as_str()).ok_or_else(|| {
                MycelialCoordinateTransformError::InvalidCoordinateSystem(
                    step.output_system_id.clone(),
                )
            })?;
            validate_step(step, input, output, self.profile)?;
            current_id = step.output_system_id.as_str();
            if step.input_system_id != step.output_system_id && !visited.insert(current_id) {
                return Err(MycelialCoordinateTransformError::DisconnectedTransformChain(
                    step.transform_id.clone(),
                ));
            }
        }
        if current_id != self.graph_system_id {
            return Err(MycelialCoordinateTransformError::DisconnectedTransformChain(
                self.graph_system_id.clone(),
            ));
        }
        Ok(())
    }

    pub fn canonical_digest_hex(
        &self,
    ) -> Result<String, MycelialCoordinateTransformError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-mycelial-coordinate-transform-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        hasher.update(&[match self.profile {
            MycelialOmeNgffTransformProfile::OmeZarrV05Published => 0,
            MycelialOmeNgffTransformProfile::Rfc5V06Implementation => 1,
        }]);
        for value in [
            &self.source_content_digest,
            &self.axes_metadata_digest,
            &self.declared_transform_digest,
            &self.source_system_id,
            &self.graph_system_id,
        ] {
            hash_string(&mut hasher, value);
        }
        hasher.update(&(self.coordinate_systems.len() as u64).to_le_bytes());
        for system in &self.coordinate_systems {
            hash_string(&mut hasher, &system.system_id);
            hash_string(&mut hasher, &system.frame_id);
            hasher.update(&(system.axes.len() as u64).to_le_bytes());
            for axis in &system.axes {
                hash_string(&mut hasher, &axis.name);
                hasher.update(&[match axis.role {
                    MycelialSpatialAxisRole::X => 0,
                    MycelialSpatialAxisRole::Y => 1,
                    MycelialSpatialAxisRole::Z => 2,
                }]);
                hash_string(&mut hasher, &axis.unit);
            }
        }
        hasher.update(&(self.transformations.len() as u64).to_le_bytes());
        for step in &self.transformations {
            hash_string(&mut hasher, &step.transform_id);
            hash_string(&mut hasher, &step.input_system_id);
            hash_string(&mut hasher, &step.output_system_id);
            match &step.operation {
                MycelialSpatialTransformOperation::Identity => {
                    hasher.update(&[0]);
                }
                MycelialSpatialTransformOperation::Scale { factors } => {
                    hasher.update(&[1]);
                    for value in factors {
                        hasher.update(&value.to_bits().to_le_bytes());
                    }
                }
                MycelialSpatialTransformOperation::Translation { offsets } => {
                    hasher.update(&[2]);
                    for value in offsets {
                        hasher.update(&value.to_bits().to_le_bytes());
                    }
                }
                MycelialSpatialTransformOperation::Affine { matrix } => {
                    hasher.update(&[3]);
                    for row in matrix {
                        for value in row {
                            hasher.update(&value.to_bits().to_le_bytes());
                        }
                    }
                }
                MycelialSpatialTransformOperation::Rotation { matrix } => {
                    hasher.update(&[4]);
                    for row in matrix {
                        for value in row {
                            hasher.update(&value.to_bits().to_le_bytes());
                        }
                    }
                }
                MycelialSpatialTransformOperation::AxisPermutation {
                    source_axis_for_output,
                } => {
                    hasher.update(&[5]);
                    for index in source_axis_for_output {
                        hasher.update(&(*index as u64).to_le_bytes());
                    }
                }
                MycelialSpatialTransformOperation::Displacements {
                    field_path,
                    field_digest,
                    interpolation,
                } => {
                    hasher.update(&[6]);
                    hash_string(&mut hasher, field_path);
                    hash_string(&mut hasher, field_digest);
                    hash_string(&mut hasher, interpolation);
                }
                MycelialSpatialTransformOperation::Coordinates {
                    field_path,
                    field_digest,
                    interpolation,
                } => {
                    hasher.update(&[7]);
                    hash_string(&mut hasher, field_path);
                    hash_string(&mut hasher, field_digest);
                    hash_string(&mut hasher, interpolation);
                }
            }
        }
        Ok(hex_digest(hasher.finalize().as_bytes()))
    }

    pub fn validate_against_sidecar(
        &self,
        sidecar: &MycelialGraphInterchangeSidecar,
    ) -> Result<(), MycelialCoordinateTransformError> {
        self.validate()?;
        sidecar
            .validate()
            .map_err(|_| MycelialCoordinateTransformError::GraphFrameMismatch)?;
        if self.source_content_digest != sidecar.source.source_content_digest {
            return Err(MycelialCoordinateTransformError::SourceDigestMismatch);
        }
        if self.axes_metadata_digest != sidecar.source.axes_metadata_digest {
            return Err(MycelialCoordinateTransformError::AxesDigestMismatch);
        }
        if self.declared_transform_digest != sidecar.source.spatial_transform_digest {
            return Err(MycelialCoordinateTransformError::TransformDigestMismatch);
        }
        let target = self
            .coordinate_systems
            .iter()
            .find(|system| system.system_id == self.graph_system_id)
            .ok_or_else(|| {
                MycelialCoordinateTransformError::InvalidCoordinateSystem(
                    self.graph_system_id.clone(),
                )
            })?;
        let graph_dimensions = match sidecar.graph.dimensionality {
            ReconstructionDimensionality::TwoD => 2,
            ReconstructionDimensionality::ThreeD => 3,
        };
        if target.frame_id != sidecar.graph.coordinate_frame_id
            || target.axes.len() != graph_dimensions
        {
            return Err(MycelialCoordinateTransformError::GraphFrameMismatch);
        }
        Ok(())
    }

    pub fn transform_spatial_point(
        &self,
        input_coordinates: &[f64],
        graph: &AnnotatedMycelialSkeleton,
    ) -> Result<MycelialTransformedPoint, MycelialCoordinateTransformError> {
        self.validate()?;
        let source = self
            .coordinate_systems
            .iter()
            .find(|system| system.system_id == self.source_system_id)
            .ok_or_else(|| {
                MycelialCoordinateTransformError::InvalidCoordinateSystem(
                    self.source_system_id.clone(),
                )
            })?;
        let target = self
            .coordinate_systems
            .iter()
            .find(|system| system.system_id == self.graph_system_id)
            .ok_or_else(|| {
                MycelialCoordinateTransformError::InvalidCoordinateSystem(
                    self.graph_system_id.clone(),
                )
            })?;
        if input_coordinates.len() != source.axes.len() {
            return Err(MycelialCoordinateTransformError::PointDimensionMismatch);
        }
        if input_coordinates.iter().any(|value| !value.is_finite()) {
            return Err(MycelialCoordinateTransformError::NonFinitePoint);
        }
        let graph_dimensions = match graph.dimensionality {
            ReconstructionDimensionality::TwoD => 2,
            ReconstructionDimensionality::ThreeD => 3,
        };
        if target.frame_id != graph.coordinate_frame_id || target.axes.len() != graph_dimensions {
            return Err(MycelialCoordinateTransformError::GraphFrameMismatch);
        }

        let systems: BTreeMap<_, _> = self
            .coordinate_systems
            .iter()
            .map(|system| (system.system_id.as_str(), system))
            .collect();
        let mut current = input_coordinates.to_vec();
        for step in &self.transformations {
            let input = systems[step.input_system_id.as_str()];
            let output = systems[step.output_system_id.as_str()];
            current = apply_operation(
                &step.operation,
                &current,
                input,
                output,
                &step.transform_id,
            )?;
        }
        if current.len() != target.axes.len() {
            return Err(MycelialCoordinateTransformError::PointDimensionMismatch);
        }

        let mut xyz = [0.0_f64; 3];
        for (index, axis) in target.axes.iter().enumerate() {
            let factor = unit_conversion_factor(&axis.unit, graph.length_unit)?;
            let value = current[index] * factor;
            if !value.is_finite() {
                return Err(MycelialCoordinateTransformError::NonFinitePoint);
            }
            match axis.role {
                MycelialSpatialAxisRole::X => xyz[0] = value,
                MycelialSpatialAxisRole::Y => xyz[1] = value,
                MycelialSpatialAxisRole::Z => xyz[2] = value,
            }
        }
        if graph.dimensionality == ReconstructionDimensionality::TwoD {
            xyz[2] = 0.0;
        }
        Ok(MycelialTransformedPoint {
            position: ReconstructionPoint3 {
                x: xyz[0],
                y: xyz[1],
                z: xyz[2],
            },
            coordinate_frame_id: graph.coordinate_frame_id.clone(),
            length_unit: graph.length_unit,
            coordinate_transform_digest: self.canonical_digest_hex()?,
        })
    }
}

fn unit_conversion_factor(
    input_unit: &str,
    graph_unit: ReconstructionLengthUnit,
) -> Result<f64, MycelialCoordinateTransformError> {
    let normalized = input_unit.to_ascii_lowercase();
    let micrometers = match normalized.as_str() {
        "micrometer" | "micrometers" | "micrometre" | "micrometres" | "um" | "µm" | "micron"
        | "microns" => Some(1.0),
        "nanometer" | "nanometers" | "nanometre" | "nanometres" | "nm" => Some(0.001),
        "millimeter" | "millimeters" | "millimetre" | "millimetres" | "mm" => Some(1_000.0),
        "meter" | "meters" | "metre" | "metres" | "m" => Some(1_000_000.0),
        _ => None,
    };
    match graph_unit {
        ReconstructionLengthUnit::Pixel => {
            if matches!(normalized.as_str(), "pixel" | "pixels" | "px") {
                Ok(1.0)
            } else {
                Err(MycelialCoordinateTransformError::UnknownLengthUnit(
                    input_unit.to_string(),
                ))
            }
        }
        ReconstructionLengthUnit::Micrometer => micrometers.ok_or_else(|| {
            MycelialCoordinateTransformError::UnknownLengthUnit(input_unit.to_string())
        }),
        ReconstructionLengthUnit::Millimeter => micrometers
            .map(|factor| factor / 1_000.0)
            .ok_or_else(|| {
                MycelialCoordinateTransformError::UnknownLengthUnit(input_unit.to_string())
            }),
        ReconstructionLengthUnit::OrganismLocalModelUnit => {
            if normalized == "organism_local_model_unit" || normalized == "model_unit" {
                Ok(1.0)
            } else {
                Err(MycelialCoordinateTransformError::UnknownLengthUnit(
                    input_unit.to_string(),
                ))
            }
        }
    }
}

fn apply_operation(
    operation: &MycelialSpatialTransformOperation,
    input_values: &[f64],
    input_system: &MycelialSpatialCoordinateSystem,
    output_system: &MycelialSpatialCoordinateSystem,
    transform_id: &str,
) -> Result<Vec<f64>, MycelialCoordinateTransformError> {
    let input_count = input_system.axes.len();
    let output_count = output_system.axes.len();
    if input_values.len() != input_count {
        return Err(MycelialCoordinateTransformError::PointDimensionMismatch);
    }
    let result = match operation {
        MycelialSpatialTransformOperation::Identity => input_values.to_vec(),
        MycelialSpatialTransformOperation::Scale { factors } => input_values
            .iter()
            .zip(factors)
            .map(|(value, factor)| value * factor)
            .collect(),
        MycelialSpatialTransformOperation::Translation { offsets } => input_values
            .iter()
            .zip(offsets)
            .map(|(value, offset)| value + offset)
            .collect(),
        MycelialSpatialTransformOperation::Affine { matrix } => matrix
            .iter()
            .map(|row| {
                let linear: f64 = row[..input_count]
                    .iter()
                    .zip(input_values)
                    .map(|(coefficient, value)| coefficient * value)
                    .sum();
                linear + row[input_count]
            })
            .collect(),
        MycelialSpatialTransformOperation::Rotation { matrix } => matrix
            .iter()
            .map(|row| {
                row.iter()
                    .zip(input_values)
                    .map(|(coefficient, value)| coefficient * value)
                    .sum()
            })
            .collect(),
        MycelialSpatialTransformOperation::AxisPermutation {
            source_axis_for_output,
        } => source_axis_for_output
            .iter()
            .map(|input_index| input_values[*input_index])
            .collect(),
        MycelialSpatialTransformOperation::Displacements { .. }
        | MycelialSpatialTransformOperation::Coordinates { .. } => {
            return Err(MycelialCoordinateTransformError::UnsupportedUnresolvedField(
                transform_id.to_string(),
            ));
        }
    };
    if result.len() != output_count || result.iter().any(|value| !value.is_finite()) {
        return Err(MycelialCoordinateTransformError::NonFinitePoint);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mycelial_graph_interchange::{
        MycelialGraphInterchangeSidecar, MycelialRightsStatus, MycelialSourceDatasetReference,
        MycelialSourceFormat,
    };
    use crate::mycelial_reconstruction_metrics::{
        AnnotatedHyphaSegment, AnnotatedJunction, AnnotatedJunctionKind,
        MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION,
    };

    fn digest(character: char) -> String {
        let nibble = character.to_digit(16).unwrap_or((character as u32) % 16);
        std::iter::repeat_n(char::from_digit(nibble, 16).unwrap(), 64).collect()
    }

    fn graph() -> AnnotatedMycelialSkeleton {
        AnnotatedMycelialSkeleton {
            schema_version: MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION,
            frame_id: "frame-coordinate-transform".to_string(),
            coordinate_frame_id: "stage-xyz".to_string(),
            calibration_digest: digest('b'),
            source_observation_digest: digest('a'),
            dimensionality: ReconstructionDimensionality::ThreeD,
            length_unit: ReconstructionLengthUnit::Micrometer,
            producer_id: "human-annotation".to_string(),
            producer_version: "v1".to_string(),
            junctions: vec![
                AnnotatedJunction {
                    id: "A".to_string(),
                    kind: AnnotatedJunctionKind::Origin,
                    position: ReconstructionPoint3 { x: 12.0, y: 21.5, z: 34.0 },
                },
                AnnotatedJunction {
                    id: "B".to_string(),
                    kind: AnnotatedJunctionKind::Tip,
                    position: ReconstructionPoint3 { x: 13.0, y: 21.5, z: 34.0 },
                },
            ],
            segments: vec![AnnotatedHyphaSegment {
                id: "ab".to_string(),
                start_junction_id: "A".to_string(),
                end_junction_id: "B".to_string(),
                length: 1.0,
            }],
        }
    }

    fn axes_zyx(unit: &str) -> Vec<MycelialSpatialAxis> {
        vec![
            MycelialSpatialAxis { name: "z".to_string(), role: MycelialSpatialAxisRole::Z, unit: unit.to_string() },
            MycelialSpatialAxis { name: "y".to_string(), role: MycelialSpatialAxisRole::Y, unit: unit.to_string() },
            MycelialSpatialAxis { name: "x".to_string(), role: MycelialSpatialAxisRole::X, unit: unit.to_string() },
        ]
    }

    fn axes_xyz(unit: &str) -> Vec<MycelialSpatialAxis> {
        vec![
            MycelialSpatialAxis { name: "x".to_string(), role: MycelialSpatialAxisRole::X, unit: unit.to_string() },
            MycelialSpatialAxis { name: "y".to_string(), role: MycelialSpatialAxisRole::Y, unit: unit.to_string() },
            MycelialSpatialAxis { name: "z".to_string(), role: MycelialSpatialAxisRole::Z, unit: unit.to_string() },
        ]
    }

    fn sidecar(graph: AnnotatedMycelialSkeleton) -> MycelialGraphInterchangeSidecar {
        MycelialGraphInterchangeSidecar::new(
            MycelialSourceDatasetReference {
                source_uri: "https://example.org/sample.ome.zarr".to_string(),
                source_format: MycelialSourceFormat::OmeZarr,
                dataset_id: "dataset".to_string(),
                acquisition_session_id: "session".to_string(),
                source_content_digest: digest('a'),
                axes_metadata_digest: digest('c'),
                spatial_transform_digest: digest('d'),
                rights_status: MycelialRightsStatus::VerifiedOpen,
                rights_evidence_digest: digest('e'),
            },
            graph,
            Vec::new(),
        )
        .unwrap()
    }

    fn manifest(
        profile: MycelialOmeNgffTransformProfile,
        coordinate_systems: Vec<MycelialSpatialCoordinateSystem>,
        source_system_id: &str,
        graph_system_id: &str,
        transformations: Vec<MycelialSpatialTransformStep>,
    ) -> MycelialCoordinateTransformManifest {
        MycelialCoordinateTransformManifest {
            schema_version: MYCELIAL_COORDINATE_TRANSFORM_SCHEMA_VERSION,
            profile,
            source_content_digest: digest('a'),
            axes_metadata_digest: digest('c'),
            declared_transform_digest: digest('d'),
            source_system_id: source_system_id.to_string(),
            graph_system_id: graph_system_id.to_string(),
            coordinate_systems,
            transformations,
        }
    }

    #[test]
    fn v05_scale_translation_maps_anisotropic_pixels_to_graph_micrometers() {
        let systems = vec![
            MycelialSpatialCoordinateSystem {
                system_id: "pixel-zyx".to_string(),
                frame_id: "pixel-frame".to_string(),
                axes: axes_zyx("pixel"),
            },
            MycelialSpatialCoordinateSystem {
                system_id: "scaled-zyx".to_string(),
                frame_id: "scaled-frame".to_string(),
                axes: axes_zyx("micrometer"),
            },
        ];
        let transform = manifest(
            MycelialOmeNgffTransformProfile::OmeZarrV05Published,
            systems,
            "pixel-zyx",
            "scaled-zyx",
            vec![
                MycelialSpatialTransformStep {
                    transform_id: "scale".to_string(),
                    input_system_id: "pixel-zyx".to_string(),
                    output_system_id: "scaled-zyx".to_string(),
                    operation: MycelialSpatialTransformOperation::Scale {
                        factors: vec![4.0, 2.0, 1.0],
                    },
                },
                MycelialSpatialTransformStep {
                    transform_id: "translation".to_string(),
                    input_system_id: "scaled-zyx".to_string(),
                    output_system_id: "scaled-zyx".to_string(),
                    operation: MycelialSpatialTransformOperation::Translation {
                        offsets: vec![30.0, 20.0, 10.0],
                    },
                },
            ],
        );
        let result = transform.transform_spatial_point(&[2.0, 3.0, 4.0], &graph()).unwrap();
        assert_eq!(
            result.position,
            ReconstructionPoint3 { x: 14.0, y: 26.0, z: 38.0 }
        );
    }

    #[test]
    fn rfc5_axis_permutation_scale_translation_chain_maps_zyx_to_xyz() {
        let systems = vec![
            MycelialSpatialCoordinateSystem {
                system_id: "pixel-zyx".to_string(),
                frame_id: "pixel-frame".to_string(),
                axes: axes_zyx("pixel"),
            },
            MycelialSpatialCoordinateSystem {
                system_id: "pixel-xyz".to_string(),
                frame_id: "pixel-reordered".to_string(),
                axes: axes_xyz("pixel"),
            },
            MycelialSpatialCoordinateSystem {
                system_id: "scaled-xyz".to_string(),
                frame_id: "scaled-xyz".to_string(),
                axes: axes_xyz("micrometer"),
            },
            MycelialSpatialCoordinateSystem {
                system_id: "stage-xyz".to_string(),
                frame_id: "stage-xyz".to_string(),
                axes: axes_xyz("micrometer"),
            },
        ];
        let transform = manifest(
            MycelialOmeNgffTransformProfile::Rfc5V06Implementation,
            systems,
            "pixel-zyx",
            "stage-xyz",
            vec![
                MycelialSpatialTransformStep {
                    transform_id: "reorder-zyx-to-xyz".to_string(),
                    input_system_id: "pixel-zyx".to_string(),
                    output_system_id: "pixel-xyz".to_string(),
                    operation: MycelialSpatialTransformOperation::AxisPermutation {
                        source_axis_for_output: vec![2, 1, 0],
                    },
                },
                MycelialSpatialTransformStep {
                    transform_id: "scale".to_string(),
                    input_system_id: "pixel-xyz".to_string(),
                    output_system_id: "scaled-xyz".to_string(),
                    operation: MycelialSpatialTransformOperation::Scale {
                        factors: vec![0.5, 0.5, 2.0],
                    },
                },
                MycelialSpatialTransformStep {
                    transform_id: "translation".to_string(),
                    input_system_id: "scaled-xyz".to_string(),
                    output_system_id: "stage-xyz".to_string(),
                    operation: MycelialSpatialTransformOperation::Translation {
                        offsets: vec![10.0, 20.0, 30.0],
                    },
                },
            ],
        );
        let result = transform.transform_spatial_point(&[2.0, 3.0, 4.0], &graph()).unwrap();
        assert_eq!(
            result.position,
            ReconstructionPoint3 { x: 12.0, y: 21.5, z: 34.0 }
        );
        assert_eq!(result.coordinate_frame_id, "stage-xyz");
        assert_eq!(result.length_unit, ReconstructionLengthUnit::Micrometer);
    }

    #[test]
    fn affine_transform_applies_rotation_and_translation() {
        let systems = vec![
            MycelialSpatialCoordinateSystem {
                system_id: "pixel-zyx".to_string(),
                frame_id: "pixel-frame".to_string(),
                axes: axes_zyx("pixel"),
            },
            MycelialSpatialCoordinateSystem {
                system_id: "stage-xyz".to_string(),
                frame_id: "stage-xyz".to_string(),
                axes: axes_xyz("micrometer"),
            },
        ];
        let transform = manifest(
            MycelialOmeNgffTransformProfile::Rfc5V06Implementation,
            systems,
            "pixel-zyx",
            "stage-xyz",
            vec![MycelialSpatialTransformStep {
                transform_id: "affine".to_string(),
                input_system_id: "pixel-zyx".to_string(),
                output_system_id: "stage-xyz".to_string(),
                operation: MycelialSpatialTransformOperation::Affine {
                    matrix: vec![
                        vec![0.0, -1.0, 0.0, 10.0],
                        vec![1.0, 0.0, 0.0, 20.0],
                        vec![0.0, 0.0, 1.0, 30.0],
                    ],
                },
            }],
        );
        let result = transform.transform_spatial_point(&[2.0, 3.0, 4.0], &graph()).unwrap();
        assert_eq!(
            result.position,
            ReconstructionPoint3 { x: 7.0, y: 22.0, z: 34.0 }
        );
    }

    #[test]
    fn profile_and_singularity_constraints_fail_closed() {
        let systems = vec![
            MycelialSpatialCoordinateSystem {
                system_id: "pixel-zyx".to_string(),
                frame_id: "pixel-frame".to_string(),
                axes: axes_zyx("pixel"),
            },
            MycelialSpatialCoordinateSystem {
                system_id: "stage-xyz".to_string(),
                frame_id: "stage-xyz".to_string(),
                axes: axes_xyz("micrometer"),
            },
        ];
        let affine_step = MycelialSpatialTransformStep {
            transform_id: "affine".to_string(),
            input_system_id: "pixel-zyx".to_string(),
            output_system_id: "stage-xyz".to_string(),
            operation: MycelialSpatialTransformOperation::Affine {
                matrix: vec![
                    vec![1.0, 0.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0, 0.0],
                    vec![0.0, 0.0, 1.0, 0.0],
                ],
            },
        };
        let v05 = manifest(
            MycelialOmeNgffTransformProfile::OmeZarrV05Published,
            systems.clone(),
            "pixel-zyx",
            "stage-xyz",
            vec![affine_step.clone()],
        );
        assert!(matches!(
            v05.validate(),
            Err(MycelialCoordinateTransformError::UnsupportedOperationForProfile(_))
        ));

        let mut singular = affine_step;
        singular.operation = MycelialSpatialTransformOperation::Affine {
            matrix: vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
            ],
        };
        let rfc5 = manifest(
            MycelialOmeNgffTransformProfile::Rfc5V06Implementation,
            systems,
            "pixel-zyx",
            "stage-xyz",
            vec![singular],
        );
        assert!(matches!(
            rfc5.validate(),
            Err(MycelialCoordinateTransformError::NonInvertibleTransform(_))
        ));
    }

    #[test]
    fn nonlinear_field_reference_is_bound_but_not_silently_approximated() {
        let systems = vec![
            MycelialSpatialCoordinateSystem {
                system_id: "pixel-zyx".to_string(),
                frame_id: "pixel-frame".to_string(),
                axes: axes_zyx("pixel"),
            },
            MycelialSpatialCoordinateSystem {
                system_id: "stage-xyz".to_string(),
                frame_id: "stage-xyz".to_string(),
                axes: axes_xyz("micrometer"),
            },
        ];
        let transform = manifest(
            MycelialOmeNgffTransformProfile::Rfc5V06Implementation,
            systems,
            "pixel-zyx",
            "stage-xyz",
            vec![MycelialSpatialTransformStep {
                transform_id: "nonlinear-registration".to_string(),
                input_system_id: "pixel-zyx".to_string(),
                output_system_id: "stage-xyz".to_string(),
                operation: MycelialSpatialTransformOperation::Displacements {
                    field_path: "registration/displacement-field".to_string(),
                    field_digest: digest('f'),
                    interpolation: "linear".to_string(),
                },
            }],
        );
        assert!(transform.validate().is_ok());
        assert!(matches!(
            transform.transform_spatial_point(&[2.0, 3.0, 4.0], &graph()),
            Err(MycelialCoordinateTransformError::UnsupportedUnresolvedField(_))
        ));
    }

    #[test]
    fn transform_manifest_is_bound_to_exact_sidecar_digests_and_graph_frame() {
        let systems = vec![
            MycelialSpatialCoordinateSystem {
                system_id: "pixel-zyx".to_string(),
                frame_id: "pixel-frame".to_string(),
                axes: axes_zyx("pixel"),
            },
            MycelialSpatialCoordinateSystem {
                system_id: "stage-xyz".to_string(),
                frame_id: "stage-xyz".to_string(),
                axes: axes_xyz("micrometer"),
            },
        ];
        let transform = manifest(
            MycelialOmeNgffTransformProfile::Rfc5V06Implementation,
            systems,
            "pixel-zyx",
            "stage-xyz",
            vec![MycelialSpatialTransformStep {
                transform_id: "affine".to_string(),
                input_system_id: "pixel-zyx".to_string(),
                output_system_id: "stage-xyz".to_string(),
                operation: MycelialSpatialTransformOperation::Affine {
                    matrix: vec![
                        vec![1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0],
                        vec![0.0, 0.0, 1.0, 0.0],
                    ],
                },
            }],
        );
        let sidecar = sidecar(graph());
        transform.validate_against_sidecar(&sidecar).unwrap();
        assert_eq!(
            transform.canonical_digest_hex().unwrap(),
            transform.canonical_digest_hex().unwrap()
        );
        let mut tampered = transform.clone();
        tampered.declared_transform_digest = digest('9');
        assert_eq!(
            tampered.validate_against_sidecar(&sidecar).unwrap_err(),
            MycelialCoordinateTransformError::TransformDigestMismatch
        );
    }
}
