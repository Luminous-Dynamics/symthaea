//! Authoritative fixture verification for exact pixel-window evidence.
//!
//! `ExactPixelWindowEvidence` is self-authenticating with respect to its own
//! fields, but self-consistency is not enough: the referenced source id must
//! actually resolve to the claimed immutable node inside the claimed PP-05
//! fixture universe. This module closes that loop without weakening the
//! standalone persistence checks.

use crate::{
    ExactPixelWindowEvidence, ExactWindowError, ExactWindowResult, FixtureSourceRef,
    FrozenDigest, FrozenSentinelFixtureManifest,
};
use symthaea_earth_observation::RasterWindowPlan;

fn fixture_error(message: impl Into<String>) -> ExactWindowError {
    ExactWindowError::Fixture(message.into())
}

fn source_identity<'a>(
    fixture: &'a FrozenSentinelFixtureManifest,
    source: &FixtureSourceRef,
) -> ExactWindowResult<&'a str> {
    match source {
        FixtureSourceRef::Product { product_id } => fixture
            .products
            .iter()
            .find(|product| product.product_id == *product_id)
            .map(|product| product.metadata_digest.as_str())
            .ok_or_else(|| fixture_error(format!("fixture does not contain product {product_id}"))),
        FixtureSourceRef::Artifact { artifact_id } => fixture
            .artifacts
            .iter()
            .find(|artifact| artifact.artifact_id == *artifact_id)
            .map(|artifact| artifact.identity_digest.as_str())
            .ok_or_else(|| fixture_error(format!("fixture does not contain artifact {artifact_id}"))),
    }
}

impl ExactPixelWindowEvidence {
    /// Construct evidence from an authoritative frozen fixture rather than
    /// accepting caller-supplied fixture/source identity strings.
    #[allow(clippy::too_many_arguments)]
    pub fn new_for_fixture(
        fixture: &FrozenSentinelFixtureManifest,
        source: FixtureSourceRef,
        plan: &RasterWindowPlan,
        output_content_digest: FrozenDigest,
        byte_len: Option<u64>,
        extractor_software: impl Into<String>,
        extractor_version: impl Into<String>,
    ) -> ExactWindowResult<Self> {
        fixture
            .verify_digest()
            .map_err(|error| fixture_error(format!("fixture verification failed: {error}")))?;
        let source_identity_digest = source_identity(fixture, &source)?.to_string();
        let evidence = Self::new(
            fixture.manifest_digest.clone(),
            source_identity_digest,
            source,
            plan,
            output_content_digest,
            byte_len,
            extractor_software,
            extractor_version,
        )?;
        evidence.verify_against_fixture(fixture)?;
        Ok(evidence)
    }

    /// Revalidate both the receipt itself and its claimed source against the
    /// authoritative PP-05 fixture manifest.
    pub fn verify_against_fixture(
        &self,
        fixture: &FrozenSentinelFixtureManifest,
    ) -> ExactWindowResult<()> {
        self.verify_digest()?;
        fixture
            .verify_digest()
            .map_err(|error| fixture_error(format!("fixture verification failed: {error}")))?;

        if self.fixture_manifest_digest != fixture.manifest_digest {
            return Err(fixture_error(format!(
                "fixture manifest digest mismatch: evidence={} authoritative={}",
                self.fixture_manifest_digest, fixture.manifest_digest
            )));
        }

        let authoritative_source_identity = source_identity(fixture, &self.source)?;
        if self.source_identity_digest != authoritative_source_identity {
            return Err(fixture_error(format!(
                "source identity digest mismatch for {}: evidence={} authoritative={}",
                match &self.source {
                    FixtureSourceRef::Product { product_id } => product_id.as_str(),
                    FixtureSourceRef::Artifact { artifact_id } => artifact_id.as_str(),
                },
                self.source_identity_digest, authoritative_source_identity
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SentinelProductKind, SentinelProductMetadata};
    use symthaea_earth_observation::{
        AffineGridTransform, ContentDigest, CrsId, DigestAlgorithm, GeoFootprint, GeoPoint,
        GridAnchor, ObservationUncertainty, PixelWindow, ProcessingLineage, RasterGrid,
        RasterReference, RasterShape, SensorModality,
    };

    fn hex256(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn product(acquired_at_unix_ms: i64) -> SentinelProductMetadata {
        SentinelProductMetadata {
            observation_id: "obs-S2-L2A-001".to_string(),
            mission_id: "Sentinel-2".to_string(),
            instrument_id: "MSI".to_string(),
            product_id: "S2-L2A-001".to_string(),
            product_kind: SentinelProductKind::Sentinel2L2A,
            acquired_at_unix_ms,
            footprint: GeoFootprint::new(vec![
                GeoPoint::new(-25.0, 28.0).unwrap(),
                GeoPoint::new(-25.1, 28.0).unwrap(),
                GeoPoint::new(-25.0, 28.1).unwrap(),
            ])
            .unwrap(),
            modality: SensorModality::Multispectral,
            bands: vec![],
            uncertainty: ObservationUncertainty::new(None, None, None).unwrap(),
            source_digest: ContentDigest::new(DigestAlgorithm::Sha256, hex256('1')).unwrap(),
            lineage: ProcessingLineage::default(),
        }
    }

    fn fixture(acquired_at_unix_ms: i64) -> FrozenSentinelFixtureManifest {
        FrozenSentinelFixtureManifest::new(
            "fixture-wetland-001",
            vec![product(acquired_at_unix_ms)],
            vec![],
        )
        .unwrap()
    }

    fn plan() -> RasterWindowPlan {
        let reference = RasterReference::new(
            CrsId::new("EPSG:32635").unwrap(),
            AffineGridTransform::new(
                500_000.0,
                7_200_000.0,
                10.0,
                0.0,
                0.0,
                -10.0,
                GridAnchor::PixelCorner,
            )
            .unwrap(),
        );
        RasterGrid::new(RasterShape::new(100, 200).unwrap(), reference)
            .window(PixelWindow::new(7, 11, 20, 30).unwrap())
            .unwrap()
    }

    fn output_digest() -> FrozenDigest {
        FrozenDigest {
            algorithm: "sha256".to_string(),
            hex: hex256('2'),
        }
    }

    #[test]
    fn authoritative_constructor_derives_fixture_and_source_identity() {
        let fixture = fixture(100);
        let evidence = ExactPixelWindowEvidence::new_for_fixture(
            &fixture,
            FixtureSourceRef::Product {
                product_id: "S2-L2A-001".to_string(),
            },
            &plan(),
            output_digest(),
            None,
            "symthaea-window",
            "0.1.0",
        )
        .unwrap();

        assert_eq!(evidence.fixture_manifest_digest, fixture.manifest_digest);
        assert_eq!(
            evidence.source_identity_digest,
            fixture.products[0].metadata_digest
        );
        evidence.verify_against_fixture(&fixture).unwrap();
    }

    #[test]
    fn self_consistent_forged_source_identity_fails_authoritative_verification() {
        let fixture = fixture(100);
        let mut evidence = ExactPixelWindowEvidence::new_for_fixture(
            &fixture,
            FixtureSourceRef::Product {
                product_id: "S2-L2A-001".to_string(),
            },
            &plan(),
            output_digest(),
            None,
            "symthaea-window",
            "0.1.0",
        )
        .unwrap();

        evidence.source_identity_digest = hex256('f');
        evidence.evidence_digest = evidence.compute_digest().unwrap();
        evidence.verify_digest().unwrap();
        assert!(evidence.verify_against_fixture(&fixture).is_err());
    }

    #[test]
    fn same_source_label_in_different_fixture_universe_is_rejected() {
        let first_fixture = fixture(100);
        let second_fixture = fixture(200);
        let evidence = ExactPixelWindowEvidence::new_for_fixture(
            &first_fixture,
            FixtureSourceRef::Product {
                product_id: "S2-L2A-001".to_string(),
            },
            &plan(),
            output_digest(),
            None,
            "symthaea-window",
            "0.1.0",
        )
        .unwrap();

        assert!(evidence.verify_against_fixture(&second_fixture).is_err());
    }

    #[test]
    fn missing_source_label_is_rejected_before_evidence_construction() {
        let fixture = fixture(100);
        let result = ExactPixelWindowEvidence::new_for_fixture(
            &fixture,
            FixtureSourceRef::Product {
                product_id: "missing".to_string(),
            },
            &plan(),
            output_digest(),
            None,
            "symthaea-window",
            "0.1.0",
        );
        assert!(result.is_err());
    }
}
