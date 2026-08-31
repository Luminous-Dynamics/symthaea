//! Content-addressed evidence binding for exact, non-resampled raster windows.
//!
//! PP-06 defines geometry-only raster/window semantics in the provider-neutral
//! Earth-observation domain. This module binds one such plan to a concrete
//! fixture source and exact output bytes without mislabelling the operation as
//! resampling.

use serde::{Deserialize, Serialize};
use symthaea_earth_observation::{
    AffineGridTransform, GridAnchor, RasterWindowPlan,
};
use thiserror::Error;

use crate::{
    FixtureArtifactKind, FixtureSourceRef, FrozenDigest, FrozenProcessingStep,
    SentinelFixtureArtifact,
};

const EXACT_WINDOW_SCHEMA: &str = "symthaea-sentinel-exact-pixel-window/v1";

pub type ExactWindowResult<T> = std::result::Result<T, ExactWindowError>;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExactWindowError {
    #[error("{0} must not be empty")]
    EmptyField(&'static str),
    #[error("{0} must be canonical lowercase hexadecimal")]
    NonCanonicalDigest(&'static str),
    #[error("exact-window evidence digest does not match recomputation")]
    EvidenceDigestMismatch,
    #[error("exact-window serialization failed: {0}")]
    Serialization(String),
    #[error("fixture artifact construction failed: {0}")]
    Fixture(String),
}

fn non_empty(value: &str, field: &'static str) -> ExactWindowResult<()> {
    if value.trim().is_empty() {
        Err(ExactWindowError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn canonical_hex(value: &str, field: &'static str) -> ExactWindowResult<()> {
    if value.is_empty()
        || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
        || value.bytes().any(|byte| byte.is_ascii_uppercase())
    {
        Err(ExactWindowError::NonCanonicalDigest(field))
    } else {
        Ok(())
    }
}

fn validate_digest(value: &FrozenDigest, field: &'static str) -> ExactWindowResult<()> {
    non_empty(&value.algorithm, "digest algorithm")?;
    canonical_hex(&value.hex, field)
}

fn blake3_json<T: Serialize>(value: &T) -> ExactWindowResult<String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| ExactWindowError::Serialization(error.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FrozenGridAnchor {
    PixelCorner,
    PixelCenter,
}

impl From<GridAnchor> for FrozenGridAnchor {
    fn from(value: GridAnchor) -> Self {
        match value {
            GridAnchor::PixelCorner => Self::PixelCorner,
            GridAnchor::PixelCenter => Self::PixelCenter,
        }
    }
}

/// Exact IEEE-754 representation of an affine raster transform.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenAffineGrid {
    pub origin_x_bits: u64,
    pub origin_y_bits: u64,
    pub col_step_x_bits: u64,
    pub col_step_y_bits: u64,
    pub row_step_x_bits: u64,
    pub row_step_y_bits: u64,
    pub anchor: FrozenGridAnchor,
}

impl From<AffineGridTransform> for FrozenAffineGrid {
    fn from(value: AffineGridTransform) -> Self {
        Self {
            origin_x_bits: value.origin_x.to_bits(),
            origin_y_bits: value.origin_y.to_bits(),
            col_step_x_bits: value.col_step_x.to_bits(),
            col_step_y_bits: value.col_step_y.to_bits(),
            row_step_x_bits: value.row_step_x.to_bits(),
            row_step_y_bits: value.row_step_y.to_bits(),
            anchor: value.anchor.into(),
        }
    }
}

/// Geometry-only identity of one exact pixel-window extraction.
///
/// Both the immutable root transform and the exact integer support are frozen.
/// The local effective transform is intentionally not a second source of truth:
/// it is deterministically derivable from these fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenExactWindowGeometry {
    pub crs_id: String,
    pub root_affine: FrozenAffineGrid,
    pub source_rows: u32,
    pub source_cols: u32,
    pub source_root_row_offset: u32,
    pub source_root_col_offset: u32,
    pub window_row_offset: u32,
    pub window_col_offset: u32,
    pub window_rows: u32,
    pub window_cols: u32,
    pub output_root_row_offset: u32,
    pub output_root_col_offset: u32,
    pub output_rows: u32,
    pub output_cols: u32,
    pub geometry_digest: String,
}

#[derive(Serialize)]
struct GeometryDigestView<'a> {
    schema: &'static str,
    crs_id: &'a str,
    root_affine: &'a FrozenAffineGrid,
    source_rows: u32,
    source_cols: u32,
    source_root_row_offset: u32,
    source_root_col_offset: u32,
    window_row_offset: u32,
    window_col_offset: u32,
    window_rows: u32,
    window_cols: u32,
    output_root_row_offset: u32,
    output_root_col_offset: u32,
    output_rows: u32,
    output_cols: u32,
}

impl FrozenExactWindowGeometry {
    pub fn from_plan(plan: &RasterWindowPlan) -> ExactWindowResult<Self> {
        let mut geometry = Self {
            crs_id: plan.output.crs().as_str().to_string(),
            root_affine: plan.output.reference().transform.into(),
            source_rows: plan.source_shape.rows(),
            source_cols: plan.source_shape.cols(),
            source_root_row_offset: plan.source_root_row_offset,
            source_root_col_offset: plan.source_root_col_offset,
            window_row_offset: plan.window.row_offset(),
            window_col_offset: plan.window.col_offset(),
            window_rows: plan.window.rows(),
            window_cols: plan.window.cols(),
            output_root_row_offset: plan.output.root_row_offset(),
            output_root_col_offset: plan.output.root_col_offset(),
            output_rows: plan.output.shape().rows(),
            output_cols: plan.output.shape().cols(),
            geometry_digest: String::new(),
        };
        geometry.validate_payload()?;
        geometry.geometry_digest = geometry.compute_digest()?;
        Ok(geometry)
    }

    fn digest_view(&self) -> GeometryDigestView<'_> {
        GeometryDigestView {
            schema: EXACT_WINDOW_SCHEMA,
            crs_id: &self.crs_id,
            root_affine: &self.root_affine,
            source_rows: self.source_rows,
            source_cols: self.source_cols,
            source_root_row_offset: self.source_root_row_offset,
            source_root_col_offset: self.source_root_col_offset,
            window_row_offset: self.window_row_offset,
            window_col_offset: self.window_col_offset,
            window_rows: self.window_rows,
            window_cols: self.window_cols,
            output_root_row_offset: self.output_root_row_offset,
            output_root_col_offset: self.output_root_col_offset,
            output_rows: self.output_rows,
            output_cols: self.output_cols,
        }
    }

    pub fn compute_digest(&self) -> ExactWindowResult<String> {
        blake3_json(&self.digest_view())
    }

    pub fn verify_digest(&self) -> ExactWindowResult<()> {
        self.validate_payload()?;
        canonical_hex(&self.geometry_digest, "geometry digest")?;
        if self.compute_digest()? != self.geometry_digest {
            return Err(ExactWindowError::EvidenceDigestMismatch);
        }
        Ok(())
    }

    fn validate_payload(&self) -> ExactWindowResult<()> {
        non_empty(&self.crs_id, "CRS id")?;
        if self.source_rows == 0 || self.source_cols == 0 {
            return Err(ExactWindowError::EmptyField("source raster shape"));
        }
        if self.window_rows == 0 || self.window_cols == 0 {
            return Err(ExactWindowError::EmptyField("window shape"));
        }
        if self.output_rows != self.window_rows || self.output_cols != self.window_cols {
            return Err(ExactWindowError::EmptyField(
                "exact-window output shape must equal requested window shape",
            ));
        }
        if self.output_root_row_offset
            != self.source_root_row_offset.saturating_add(self.window_row_offset)
            || self.output_root_col_offset
                != self.source_root_col_offset.saturating_add(self.window_col_offset)
        {
            return Err(ExactWindowError::EmptyField(
                "exact-window root-relative offsets must compose",
            ));
        }
        Ok(())
    }
}

/// Evidence that exact output bytes were materialized from a declared fixture
/// source under one frozen PP-06 window geometry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExactPixelWindowEvidence {
    pub source: FixtureSourceRef,
    pub geometry: FrozenExactWindowGeometry,
    pub output_content_digest: FrozenDigest,
    pub byte_len: Option<u64>,
    pub extractor_software: String,
    pub extractor_version: String,
    pub evidence_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct ExactPixelWindowEvidenceRepr {
    source: FixtureSourceRef,
    geometry: FrozenExactWindowGeometry,
    output_content_digest: FrozenDigest,
    byte_len: Option<u64>,
    extractor_software: String,
    extractor_version: String,
    evidence_digest: String,
}

#[derive(Serialize)]
struct EvidenceDigestView<'a> {
    schema: &'static str,
    source: &'a FixtureSourceRef,
    geometry_digest: &'a str,
    output_content_digest: &'a FrozenDigest,
    byte_len: Option<u64>,
    extractor_software: &'a str,
    extractor_version: &'a str,
}

impl ExactPixelWindowEvidence {
    pub fn new(
        source: FixtureSourceRef,
        plan: &RasterWindowPlan,
        output_content_digest: FrozenDigest,
        byte_len: Option<u64>,
        extractor_software: impl Into<String>,
        extractor_version: impl Into<String>,
    ) -> ExactWindowResult<Self> {
        let mut evidence = Self {
            source,
            geometry: FrozenExactWindowGeometry::from_plan(plan)?,
            output_content_digest,
            byte_len,
            extractor_software: extractor_software.into(),
            extractor_version: extractor_version.into(),
            evidence_digest: String::new(),
        };
        evidence.validate_payload()?;
        evidence.evidence_digest = evidence.compute_digest()?;
        Ok(evidence)
    }

    fn digest_view(&self) -> EvidenceDigestView<'_> {
        EvidenceDigestView {
            schema: EXACT_WINDOW_SCHEMA,
            source: &self.source,
            geometry_digest: &self.geometry.geometry_digest,
            output_content_digest: &self.output_content_digest,
            byte_len: self.byte_len,
            extractor_software: &self.extractor_software,
            extractor_version: &self.extractor_version,
        }
    }

    pub fn compute_digest(&self) -> ExactWindowResult<String> {
        blake3_json(&self.digest_view())
    }

    pub fn verify_digest(&self) -> ExactWindowResult<()> {
        self.validate_payload()?;
        canonical_hex(&self.evidence_digest, "evidence digest")?;
        if self.compute_digest()? != self.evidence_digest {
            return Err(ExactWindowError::EvidenceDigestMismatch);
        }
        Ok(())
    }

    fn validate_payload(&self) -> ExactWindowResult<()> {
        match &self.source {
            FixtureSourceRef::Product { product_id } => non_empty(product_id, "source product id")?,
            FixtureSourceRef::Artifact { artifact_id } => non_empty(artifact_id, "source artifact id")?,
        }
        self.geometry.verify_digest()?;
        validate_digest(&self.output_content_digest, "output content digest")?;
        non_empty(&self.extractor_software, "extractor software")?;
        non_empty(&self.extractor_version, "extractor version")?;
        Ok(())
    }

    /// Materialize the generic PP-05 artifact node without ever calling the
    /// operation a resample. The exact semantics remain bound by this sidecar
    /// and its geometry digest.
    pub fn to_fixture_artifact(
        &self,
        artifact_id: impl Into<String>,
    ) -> ExactWindowResult<SentinelFixtureArtifact> {
        self.verify_digest()?;
        SentinelFixtureArtifact::new(
            artifact_id,
            FixtureArtifactKind::Other,
            self.output_content_digest.clone(),
            self.byte_len,
            vec![self.source.clone()],
            vec![FrozenProcessingStep {
                name: "exact-pixel-window".to_string(),
                software: self.extractor_software.clone(),
                version: self.extractor_version.clone(),
                parameters_digest: Some(FrozenDigest {
                    algorithm: "blake3".to_string(),
                    hex: self.geometry.geometry_digest.clone(),
                }),
            }],
        )
        .map_err(|error| ExactWindowError::Fixture(error.to_string()))
    }
}

impl TryFrom<ExactPixelWindowEvidenceRepr> for ExactPixelWindowEvidence {
    type Error = ExactWindowError;

    fn try_from(value: ExactPixelWindowEvidenceRepr) -> ExactWindowResult<Self> {
        let evidence = Self {
            source: value.source,
            geometry: value.geometry,
            output_content_digest: value.output_content_digest,
            byte_len: value.byte_len,
            extractor_software: value.extractor_software,
            extractor_version: value.extractor_version,
            evidence_digest: value.evidence_digest,
        };
        evidence.verify_digest()?;
        Ok(evidence)
    }
}

impl<'de> Deserialize<'de> for ExactPixelWindowEvidence {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let repr = ExactPixelWindowEvidenceRepr::deserialize(deserializer)?;
        Self::try_from(repr).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_earth_observation::{
        AffineGridTransform, CrsId, GridAnchor, PixelWindow, RasterGrid,
        RasterReference, RasterShape,
    };

    fn plan(anchor: GridAnchor) -> RasterWindowPlan {
        let reference = RasterReference::new(
            CrsId::new("EPSG:32635").unwrap(),
            AffineGridTransform::new(
                500_000.0,
                7_200_000.0,
                10.0,
                0.0,
                0.0,
                -10.0,
                anchor,
            )
            .unwrap(),
        );
        RasterGrid::new(RasterShape::new(100, 200).unwrap(), reference)
            .window(PixelWindow::new(7, 11, 20, 30).unwrap())
            .unwrap()
    }

    fn digest(hex: &str) -> FrozenDigest {
        FrozenDigest {
            algorithm: "sha256".to_string(),
            hex: hex.to_string(),
        }
    }

    #[test]
    fn exact_window_binds_integer_support_and_affine_bits() {
        let evidence = ExactPixelWindowEvidence::new(
            FixtureSourceRef::Product {
                product_id: "S2-L2A-001".to_string(),
            },
            &plan(GridAnchor::PixelCorner),
            digest("00aa"),
            Some(600),
            "symthaea-window",
            "0.1.0",
        )
        .unwrap();

        assert_eq!(evidence.geometry.source_rows, 100);
        assert_eq!(evidence.geometry.window_row_offset, 7);
        assert_eq!(evidence.geometry.output_root_row_offset, 7);
        assert_eq!(evidence.geometry.output_rows, 20);
        evidence.verify_digest().unwrap();
    }

    #[test]
    fn exact_window_is_not_encoded_as_resampling() {
        let evidence = ExactPixelWindowEvidence::new(
            FixtureSourceRef::Artifact {
                artifact_id: "masked-s2".to_string(),
            },
            &plan(GridAnchor::PixelCorner),
            digest("00aa"),
            None,
            "symthaea-window",
            "0.1.0",
        )
        .unwrap();
        let artifact = evidence.to_fixture_artifact("roi-1").unwrap();

        assert_eq!(artifact.kind, FixtureArtifactKind::Other);
        assert_eq!(artifact.processing_steps[0].name, "exact-pixel-window");
        assert_eq!(
            artifact.processing_steps[0]
                .parameters_digest
                .as_ref()
                .unwrap()
                .hex,
            evidence.geometry.geometry_digest
        );
        assert_ne!(artifact.kind, FixtureArtifactKind::ResampledWindow);
    }

    #[test]
    fn anchor_convention_changes_geometry_identity() {
        let corner = FrozenExactWindowGeometry::from_plan(&plan(GridAnchor::PixelCorner)).unwrap();
        let center = FrozenExactWindowGeometry::from_plan(&plan(GridAnchor::PixelCenter)).unwrap();
        assert_ne!(corner.geometry_digest, center.geometry_digest);
    }

    #[test]
    fn output_bytes_are_part_of_evidence_identity() {
        let source = FixtureSourceRef::Product {
            product_id: "S2-L2A-001".to_string(),
        };
        let first = ExactPixelWindowEvidence::new(
            source.clone(),
            &plan(GridAnchor::PixelCorner),
            digest("00aa"),
            None,
            "symthaea-window",
            "0.1.0",
        )
        .unwrap();
        let second = ExactPixelWindowEvidence::new(
            source,
            &plan(GridAnchor::PixelCorner),
            digest("00ab"),
            None,
            "symthaea-window",
            "0.1.0",
        )
        .unwrap();
        assert_ne!(first.evidence_digest, second.evidence_digest);
    }

    #[test]
    fn uppercase_digest_spelling_is_rejected() {
        assert!(matches!(
            ExactPixelWindowEvidence::new(
                FixtureSourceRef::Product {
                    product_id: "S2-L2A-001".to_string(),
                },
                &plan(GridAnchor::PixelCorner),
                digest("00AA"),
                None,
                "symthaea-window",
                "0.1.0",
            ),
            Err(ExactWindowError::NonCanonicalDigest("output content digest"))
        ));
    }

    #[test]
    fn persisted_tampering_is_rejected_even_with_valid_json() {
        let evidence = ExactPixelWindowEvidence::new(
            FixtureSourceRef::Product {
                product_id: "S2-L2A-001".to_string(),
            },
            &plan(GridAnchor::PixelCorner),
            digest("00aa"),
            None,
            "symthaea-window",
            "0.1.0",
        )
        .unwrap();
        let mut value = serde_json::to_value(&evidence).unwrap();
        value["geometry"]["window_rows"] = serde_json::json!(21);

        assert!(serde_json::from_value::<ExactPixelWindowEvidence>(value).is_err());
    }
}
