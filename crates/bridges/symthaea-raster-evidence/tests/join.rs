use symthaea_earth_observation::{
    payload::{
        BandInterleave, RasterBandSemantics, RasterByteOrder, RasterPayloadDescriptor,
        RasterSampleType, SampleTransform, ValidityMaskSemantics,
    },
    AffineGridTransform, ContentDigest, CrsId, DigestAlgorithm, GeoFootprint, GeoPoint,
    GridAnchor, ObservationUncertainty, PixelWindow, ProcessingLineage, RasterGrid,
    RasterReference, RasterShape, SensorModality,
};
use symthaea_raster_evidence::{CanonicalRasterEvidence, RasterEvidenceError};
use symthaea_sentinel_eo::{
    ExactPixelWindowEvidence, FixtureSourceRef, FrozenDigest, FrozenSentinelFixtureManifest,
    SentinelProductKind, SentinelProductMetadata,
};

fn hex256(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn fixture() -> FrozenSentinelFixtureManifest {
    let product = SentinelProductMetadata {
        observation_id: "obs-S2-L2A-001".into(),
        mission_id: "Sentinel-2".into(),
        instrument_id: "MSI".into(),
        product_id: "S2-L2A-001".into(),
        product_kind: SentinelProductKind::Sentinel2L2A,
        acquired_at_unix_ms: 100,
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
    };
    FrozenSentinelFixtureManifest::new("fixture", vec![product], vec![]).unwrap()
}

fn plan() -> symthaea_earth_observation::RasterWindowPlan {
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
    RasterGrid::new(RasterShape::new(4, 5).unwrap(), reference)
        .window(PixelWindow::new(1, 1, 2, 3).unwrap())
        .unwrap()
}

fn window(fixture: &FrozenSentinelFixtureManifest) -> ExactPixelWindowEvidence {
    ExactPixelWindowEvidence::new_for_fixture(
        fixture,
        FixtureSourceRef::Product {
            product_id: "S2-L2A-001".into(),
        },
        &plan(),
        FrozenDigest {
            algorithm: "sha256".into(),
            hex: hex256('2'),
        },
        Some(24),
        "fixture-extractor",
        "1",
    )
    .unwrap()
}

fn payload(byte_order: RasterByteOrder, digest_char: char) -> RasterPayloadDescriptor {
    RasterPayloadDescriptor::new(
        RasterShape::new(2, 3).unwrap(),
        RasterSampleType::U16,
        byte_order,
        BandInterleave::BandInterleavedByPixel,
        vec![
            RasterBandSemantics::new(
                "red",
                SampleTransform::new(0.0001, 0.0).unwrap(),
                None,
            )
            .unwrap(),
            RasterBandSemantics::new(
                "nir",
                SampleTransform::new(0.0001, 0.0).unwrap(),
                None,
            )
            .unwrap(),
        ],
        ValidityMaskSemantics::None,
        24,
        ContentDigest::new(DigestAlgorithm::Sha256, hex256(digest_char)).unwrap(),
    )
    .unwrap()
}

#[test]
fn authoritative_join_accepts_same_bytes_and_shape() {
    let fixture = fixture();
    let window = window(&fixture);
    let payload = payload(RasterByteOrder::LittleEndian, '2');
    let receipt = CanonicalRasterEvidence::new_for_fixture(&fixture, &window, &payload).unwrap();
    receipt.verify_against(&fixture, &window, &payload).unwrap();
}

#[test]
fn different_bytes_fail_even_when_shape_matches() {
    let fixture = fixture();
    let window = window(&fixture);
    let payload = payload(RasterByteOrder::LittleEndian, '3');
    assert_eq!(
        CanonicalRasterEvidence::new_for_fixture(&fixture, &window, &payload).unwrap_err(),
        RasterEvidenceError::ContentDigestMismatch
    );
}

#[test]
fn same_bytes_with_different_endian_get_different_semantics() {
    let fixture = fixture();
    let window = window(&fixture);
    let little = CanonicalRasterEvidence::new_for_fixture(
        &fixture,
        &window,
        &payload(RasterByteOrder::LittleEndian, '2'),
    )
    .unwrap();
    let big = CanonicalRasterEvidence::new_for_fixture(
        &fixture,
        &window,
        &payload(RasterByteOrder::BigEndian, '2'),
    )
    .unwrap();
    assert_ne!(little.payload_semantics_digest(), big.payload_semantics_digest());
    assert_ne!(little.receipt_digest(), big.receipt_digest());
}

#[test]
fn different_shape_fails_even_when_total_byte_count_matches() {
    let fixture = fixture();
    let window = window(&fixture);
    let payload = RasterPayloadDescriptor::new(
        RasterShape::new(1, 6).unwrap(),
        RasterSampleType::U16,
        RasterByteOrder::LittleEndian,
        BandInterleave::BandInterleavedByPixel,
        vec![
            RasterBandSemantics::new("red", SampleTransform::IDENTITY, None).unwrap(),
            RasterBandSemantics::new("nir", SampleTransform::IDENTITY, None).unwrap(),
        ],
        ValidityMaskSemantics::None,
        24,
        ContentDigest::new(DigestAlgorithm::Sha256, hex256('2')).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        CanonicalRasterEvidence::new_for_fixture(&fixture, &window, &payload),
        Err(RasterEvidenceError::ShapeMismatch { .. })
    ));
}

#[test]
fn persisted_receipt_still_requires_authoritative_revalidation() {
    let fixture = fixture();
    let window = window(&fixture);
    let payload = payload(RasterByteOrder::LittleEndian, '2');
    let receipt = CanonicalRasterEvidence::new_for_fixture(&fixture, &window, &payload).unwrap();
    let encoded = serde_json::to_vec(&receipt).unwrap();
    let loaded: CanonicalRasterEvidence = serde_json::from_slice(&encoded).unwrap();
    loaded.verify_against(&fixture, &window, &payload).unwrap();
}
