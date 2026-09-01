//! Content-addressed evidence binding for exact, non-resampled raster windows.
//!
//! PP-06 defines geometry-only raster/window semantics in the provider-neutral
//! Earth-observation domain. This module binds one such plan to a concrete
//! frozen fixture universe, source node, and exact output bytes without
//! mislabelling the operation as resampling.

use serde::{Deserialize, Serialize};
use symthaea_earth_observation::{AffineGridTransform, GridAnchor, RasterWindowPlan};
use thiserror::Error;

use crate::{
    FixtureArtifactKind, FixtureSourceRef, FrozenDigest, FrozenProcessingStep,
    SentinelFixtureArtifact,
};

const EXACT_WINDOW_GEOMETRY_SCHEMA: &str = "symthaea-sentinel-exact-window-geometry/v1";
const EXACT_WINDOW_EVIDENCE_SCHEMA: &str = "symthaea-sentinel-exact-window-evidence/v1";
const HEX_256_LEN: usize = 64;

pub type ExactWindowResult<T> = std::result::Result<T, ExactWindowError>;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExactWindowError {
    #[error("{0} must not be empty")]
    EmptyField(&'static str),
    #[error("{0} must be canonical lowercase hexadecimal")]
    NonCanonicalDigest(&'static str),
    #[error("{field} must contain exactly {expected} hexadecimal characters, got {actual}")]
    InvalidDigestLength {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("unsupported frozen digest algorithm: {0}")]
    UnsupportedDigestAlgorithm(String),
    #[error("arithmetic overflow while validating {0}")]
    ArithmeticOverflow(&'static str),
    #[error("invalid exact-window geometry: {0}")]
    Geometry(String),
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

fn canonical_hex_256(value: &str, field: &'static str) -> ExactWindowResult<()> {
    canonical_hex(value, field)?;
    if value.len() != HEX_256_LEN {
        return Err(ExactWindowError::InvalidDigestLength {
            field,
            expected: HEX_256_LEN,
            actual: value.len(),
        });
    }
    Ok(())
}

fn validate_digest(value: &FrozenDigest, field: &'static str) -> ExactWindowResult<()> {
    match value.algorithm.as_str() {
        "sha256" | "blake3" => canonical_hex_256(&value.hex, field),
        "other" => canonical_hex(&value.hex, field),
        other => Err(ExactWindowError::UnsupportedDigestAlgorithm(
            other.to_string(),
        )),
    }
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

impl From<FrozenGridAnchor> for GridAnchor {
    fn from(value: FrozenGridAnchor) -> Self {
        match value {
            FrozenGridAnchor::PixelCorner => Self::PixelCorner,
            FrozenGridAnchor::PixelCenter => Self::PixelCenter,
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

impl FrozenAffineGrid {
    fn validate(&self) -> ExactWindowResult<()> {
        AffineGridTransform::new(
            f64::from_bits(self.origin_x_bits),
            f64::from_bits(self.origin_y_bits),
            f64::from_bits(self.col_step_x_bits),
            f64::from_bits(self.col_step_y_bits),
            f64::from_bits(self.row_step_x_bits),
            f64::from_bits(self.row_step_y_bits),
            self.anchor.into(),
        )
        .map(|_| ())
        .map_err(|error| ExactWindowError::Geometry(error.to_string()))
    }
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
/// The local effective transform is intentionally not stored as a second source
/// of truth: it is deterministically derivable from these fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
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

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct FrozenExactWindowGeometryRepr {
    crs_id: String,
    root_affine: FrozenAffineGrid,
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
    geometry_digest: String,
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
            schema: EXACT_WINDOW_GEOMETRY_SCHEMA,
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
        canonical_hex_256(&self.geometry_digest, "geometry digest")?;
        if self.compute_digest()? != self.geometry_digest {
            return Err(ExactWindowError::EvidenceDigestMismatch);
        }
        Ok(())
    }

    fn validate_payload(&self) -> ExactWindowResult<()> {
        non_empty(&self.crs_id, "CRS id")?;
        self.root_affine.validate()?;

        if self.source_rows == 0 || self.source_cols == 0 {
            return Err(ExactWindowError::Geometry(
                "source raster shape must be nonzero".to_string(),
            ));
        }
        if self.window_rows == 0 || self.window_cols == 0 {
            return Err(ExactWindowError::Geometry(
                "window shape must be nonzero".to_string(),
            ));
        }

        self.source_root_row_offset
            .checked_add(self.source_rows)
            .ok_or(ExactWindowError::ArithmeticOverflow(
                "source root row support end",
            ))?;
        self.source_root_col_offset
            .checked_add(self.source_cols)
            .ok_or(ExactWindowError::ArithmeticOverflow(
                "source root column support end",
            ))?;

        let row_end = self
            .window_row_offset
            .checked_add(self.window_rows)
            .ok_or(ExactWindowError::ArithmeticOverflow("window row end"))?;
        let col_end = self
            .window_col_offset
            .checked_add(self.window_cols)
            .ok_or(ExactWindowError::ArithmeticOverflow("window column end"))?;
        if row_end > self.source_rows || col_end > self.source_cols {
            return Err(ExactWindowError::Geometry(
                "window exceeds declared source support".to_string(),
            ));
        }

        if self.output_rows != self.window_rows || self.output_cols != self.window_cols {
            return Err(ExactWindowError::Geometry(
                "exact-window output shape must equal requested window shape".to_string(),
            ));
        }

        let expected_output_row = self
            .source_root_row_offset
            .checked_add(self.window_row_offset)
            .ok_or(ExactWindowError::ArithmeticOverflow(
                "output root row offset",
            ))?;
        let expected_output_col = self
            .source_root_col_offset
            .checked_add(self.window_col_offset)
            .ok_or(ExactWindowError::ArithmeticOverflow(
                "output root column offset",
            ))?;
        if self.output_root_row_offset != expected_output_row
            || self.output_root_col_offset != expected_output_col
        {
            return Err(ExactWindowError::Geometry(
                "exact-window root-relative offsets must compose".to_string(),
            ));
        }
        Ok(())
    }
}

impl TryFrom<FrozenExactWindowGeometryRepr> for FrozenExactWindowGeometry {
    type Error = ExactWindowError;

    fn try_from(value: FrozenExactWindowGeometryRepr) -> ExactWindowResult<Self> {
        let geometry = Self {
            crs_id: value.crs_id,
            root_affine: value.root_affine,
            source_rows: value.source_rows,
            source_cols: value.source_cols,
            source_root_row_offset: value.source_root_row_offset,
            source_root_col_offset: value.source_root_col_offset,
            window_row_offset: value.window_row_offset,
            window_col_offset: value.window_col_offset,
            window_rows: value.window_rows,
            window_cols: value.window_cols,
            output_root_row_offset: value.output_root_row_offset,
            output_root_col_offset: value.output_root_col_offset,
            output_rows: value.output_rows,
            output_cols: value.output_cols,
            geometry_digest: value.geometry_digest,
        };
        geometry.verify_digest()?;
        Ok(geometry)
    }
}

impl<'de> Deserialize<'de> for FrozenExactWindowGeometry {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let repr = FrozenExactWindowGeometryRepr::deserialize(deserializer)?;
        Self::try_from(repr).map_err(serde::de::Error::custom)
    }
}

/// Evidence that exact output bytes were materialized from one declared source
/// node in one exact frozen fixture universe under one frozen PP-06 geometry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExactPixelWindowEvidence {
    /// BLAKE3 identity of the PP-05 `FrozenSentinelFixtureManifest` used to
    /// resolve `source`. The source label alone is not sufficient provenance.
    pub fixture_manifest_digest: String,
    /// BLAKE3 metadata/identity digest of the referenced frozen product or
    /// derived artifact node.
    pub source_identity_digest: String,
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
    fixture_manifest_digest: String,
    source_identity_digest: String,
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
    fixture_manifest_digest: &'a str,
    source_identity_digest: &'a str,
    source: &'a FixtureSourceRef,
    geometry_digest: &'a str,
    output_content_digest: &'a FrozenDigest,
    byte_len: Option<u64>,
    extractor_software: &'a str,
    extractor_version: &'a str,
}

impl ExactPixelWindowEvidence {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        fixture_manifest_digest: impl Into<String>,
        source_identity_digest: impl Into<String>,
        source: FixtureSourceRef,
        plan: &RasterWindowPlan,
        output_content_digest: FrozenDigest,
        byte_len: Option<u64>,
        extractor_software: impl Into<String>,
        extractor_version: impl Into<String>,
    ) -> ExactWindowResult<Self> {
        let mut evidence = Self {
            fixture_manifest_digest: fixture_manifest_digest.into(),
            source_identity_digest: source_identity_digest.into(),
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
            schema: EXACT_WINDOW_EVIDENCE_SCHEMA,
            fixture_manifest_digest: &self.fixture_manifest_digest,
            source_identity_digest: &self.source_identity_digest,
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
        canonical_hex_256(&self.evidence_digest, "evidence digest")?;
        if self.compute_digest()? != self.evidence_digest {
            return Err(ExactWindowError::EvidenceDigestMismatch);
        }
        Ok(())
    }

    fn validate_payload(&self) -> ExactWindowResult<()> {
        canonical_hex_256(&self.fixture_manifest_digest, "fixture manifest digest")?;
        canonical_hex_256(&self.source_identity_digest, "source identity digest")?;
        match &self.source {
            FixtureSourceRef::Product { product_id } => {
                non_empty(product_id, "source product id")?
            }
            FixtureSourceRef::Artifact { artifact_id } => {
                non_empty(artifact_id, "source artifact id")?
            }
        }
        self.geometry.verify_digest()?;
        validate_digest(&self.output_content_digest, "output content digest")?;
        non_empty(&self.extractor_software, "extractor software")?;
        non_empty(&self.extractor_version, "extractor version")?;
        Ok(())
    }

    /// Materialize the generic PP-05 v1 artifact node without ever calling the
    /// operation a resample. The processing-parameters digest points to this
    /// full exact-window sidecar, which binds fixture universe, source identity,
    /// geometry, and output bytes.
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
                    hex: self.evidence_digest.clone(),
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
            fixture_manifest_digest: value.fixture_manifest_digest,
            source_identity_digest: value.source_identity_digest,
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
        AffineGridTransform, CrsId, GridAnchor, PixelWindow, RasterGrid, RasterReference,
        RasterShape,
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

    fn hex256(ch: char) -> String {
        std::iter::repeat_n(ch, HEX_256_LEN).collect()
    }

    fn digest(ch: char) -> FrozenDigest {
        FrozenDigest {
            algorithm: "sha256".to_string(),
            hex: hex256(ch),
        }
    }

    fn evidence(output_digest_char: char) -> ExactPixelWindowEvidence {
        ExactPixelWindowEvidence::new(
            hex256('a'),
            hex256('b'),
            FixtureSourceRef::Product {
                product_id: "S2-L2A-001".to_string(),
            },
            &plan(GridAnchor::PixelCorner),
            digest(output_digest_char),
            Some(600),
            "symthaea-window",
            "0.1.0",
        )
        .unwrap()
    }

    #[test]
    fn exact_window_binds_integer_support_and_source_universe() {
        let evidence = evidence('c');

        assert_eq!(evidence.geometry.source_rows, 100);
        assert_eq!(evidence.geometry.window_row_offset, 7);
        assert_eq!(evidence.geometry.output_root_row_offset, 7);
        assert_eq!(evidence.geometry.output_rows, 20);
        assert_eq!(evidence.fixture_manifest_digest, hex256('a'));
        assert_eq!(evidence.source_identity_digest, hex256('b'));
        evidence.verify_digest().unwrap();
    }

    #[test]
    fn exact_window_is_not_encoded_as_resampling() {
        let evidence = evidence('c');
        let artifact = evidence.to_fixture_artifact("roi-1").unwrap();

        assert_eq!(artifact.kind, FixtureArtifactKind::Other);
        assert_eq!(artifact.processing_steps[0].name, "exact-pixel-window");
        assert_eq!(
            artifact.processing_steps[0]
                .parameters_digest
                .as_ref()
                .unwrap()
                .hex,
            evidence.evidence_digest
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
        let first = evidence('c');
        let second = evidence('d');
        assert_ne!(first.evidence_digest, second.evidence_digest);
    }

    #[test]
    fn source_identity_is_not_just_a_label() {
        let first = evidence('c');
        let mut second = ExactPixelWindowEvidence::new(
            hex256('a'),
            hex256('d'),
            first.source.clone(),
            &plan(GridAnchor::PixelCorner),
            first.output_content_digest.clone(),
            first.byte_len,
            "symthaea-window",
            "0.1.0",
        )
        .unwrap();
        assert_ne!(first.evidence_digest, second.evidence_digest);

        second.source_identity_digest = first.source_identity_digest.clone();
        assert!(second.verify_digest().is_err());
    }

    #[test]
    fn fixture_universe_is_identity_significant() {
        let first = evidence('c');
        let second = ExactPixelWindowEvidence::new(
            hex256('d'),
            first.source_identity_digest.clone(),
            first.source.clone(),
            &plan(GridAnchor::PixelCorner),
            first.output_content_digest.clone(),
            first.byte_len,
            "symthaea-window",
            "0.1.0",
        )
        .unwrap();
        assert_ne!(first.evidence_digest, second.evidence_digest);
    }

    #[test]
    fn uppercase_digest_spelling_is_rejected() {
        let result = ExactPixelWindowEvidence::new(
            hex256('a'),
            hex256('b'),
            FixtureSourceRef::Product {
                product_id: "S2-L2A-001".to_string(),
            },
            &plan(GridAnchor::PixelCorner),
            FrozenDigest {
                algorithm: "sha256".to_string(),
                hex: hex256('A'),
            },
            None,
            "symthaea-window",
            "0.1.0",
        );
        assert!(matches!(
            result,
            Err(ExactWindowError::NonCanonicalDigest("output content digest"))
        ));
    }

    #[test]
    fn truncated_blake3_identity_is_rejected() {
        let result = ExactPixelWindowEvidence::new(
            "abc",
            hex256('b'),
            FixtureSourceRef::Product {
                product_id: "S2-L2A-001".to_string(),
            },
            &plan(GridAnchor::PixelCorner),
            digest('c'),
            None,
            "symthaea-window",
            "0.1.0",
        );
        assert!(matches!(
            result,
            Err(ExactWindowError::InvalidDigestLength {
                field: "fixture manifest digest",
                ..
            })
        ));
    }

    #[test]
    fn direct_geometry_deserialization_revalidates_affine_bits() {
        let geometry = FrozenExactWindowGeometry::from_plan(&plan(GridAnchor::PixelCorner)).unwrap();
        let mut value = serde_json::to_value(&geometry).unwrap();
        value["root_affine"]["origin_x_bits"] = serde_json::json!(f64::NAN.to_bits());
        assert!(serde_json::from_value::<FrozenExactWindowGeometry>(value).is_err());
    }

    #[test]
    fn overflowed_root_support_is_rejected_on_persistence() {
        let geometry = FrozenExactWindowGeometry::from_plan(&plan(GridAnchor::PixelCorner)).unwrap();
        let mut value = serde_json::to_value(&geometry).unwrap();
        value["source_root_row_offset"] = serde_json::json!(u32::MAX);
        assert!(serde_json::from_value::<FrozenExactWindowGeometry>(value).is_err());
    }

    #[test]
    fn persisted_tampering_is_rejected_even_with_valid_json() {
        let evidence = evidence('c');
        let mut value = serde_json::to_value(&evidence).unwrap();
        value["geometry"]["window_rows"] = serde_json::json!(21);

        assert!(serde_json::from_value::<ExactPixelWindowEvidence>(value).is_err());
    }
}
