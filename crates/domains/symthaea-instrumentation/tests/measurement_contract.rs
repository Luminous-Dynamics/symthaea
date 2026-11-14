// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use ed25519_dalek::{Signer, SigningKey};
use std::io::{Cursor, Read};
use std::sync::Mutex;
use symthaea_instrumentation::{
    sha256_hex, ArtifactReference, AssessmentFailure, CalibrationCertificate,
    CalibrationEvidenceResolver, CalibrationReference, CalibrationReviewPayload,
    ClockDomainId, ContractError,
    ImmutableArtifactReader, InstrumentIdentity, MeasurementEnvelope, MeasurementInput,
    MeasurementPolicy,
    MeasurementStreamGuard, Quantity, QualityFlag, RawDataEvidenceResolver, RawDataReference,
    ResolvedCalibration, ResolvedRawData, ReviewTrustKey, ReviewTrustStore,
    ReviewAuthorizationPolicy, ReviewTrustStoreGuard, ReviewVerificationError,
    SignedCalibrationReviewReceipt,
    StreamOrderFailure, Unit, VerifiedArtifactBytes, VerifiedArtifactDigest,
    VerifiedCalibrationEvidence,
    VerifiedCalibrationReviewReceipt,
};

const SAMPLE_TIME_NS: u64 = 1_000_000_000;
const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const RAW_FIXTURE_LENGTH: usize = 4_096;

fn raw_fixture_bytes() -> Vec<u8> {
    (0..RAW_FIXTURE_LENGTH)
        .map(|index| (index as u8).wrapping_mul(31).wrapping_add(17))
        .collect()
}

fn raw_digest_hex() -> String {
    sha256_hex(&raw_fixture_bytes())
}

fn raw_artifact() -> ArtifactReference {
    artifact("raw-acquisition-1", &raw_digest_hex())
}

fn raw_reference() -> RawDataReference {
    RawDataReference::new(raw_artifact())
}

fn primary_reference_standard_fixture_bytes() -> Vec<u8> {
    b"primary frequency reference standard fixture v1".to_vec()
}

fn reference_standard_certificate() -> CalibrationCertificate {
    CalibrationCertificate::new(
        "reference-standard-01",
        InstrumentIdentity::new("frequency-standard-lab-01", "reference-output-a").unwrap(),
        Quantity::Frequency,
        Unit::Hertz,
        4_000_000.0,
        6_000_000.0,
        SAMPLE_TIME_NS - 10_000,
        SAMPLE_TIME_NS + 2_000_000_000,
        100.0,
        "primary-metrology-lab-01",
        "frequency-reference-procedure-v1",
        artifact(
            "primary-reference-standard-01",
            &sha256_hex(&primary_reference_standard_fixture_bytes()),
        ),
    )
    .unwrap()
}

fn reference_standard_fixture_bytes() -> Vec<u8> {
    reference_standard_certificate().to_canonical_bytes()
}

fn calibration_certificate() -> CalibrationCertificate {
    CalibrationCertificate::new(
        "calibration-17",
        instrument_identity(),
        Quantity::Frequency,
        Unit::Hertz,
        4_000_000.0,
        6_000_000.0,
        SAMPLE_TIME_NS - 1,
        SAMPLE_TIME_NS + 2_000_000_000,
        900.0,
        "calibration-lab-01",
        "ultrasound-frequency-procedure-v1",
        artifact(
            "reference-standard-01",
            &sha256_hex(&reference_standard_fixture_bytes()),
        ),
    )
    .unwrap()
}

fn calibration_fixture_bytes() -> Vec<u8> {
    calibration_certificate().to_canonical_bytes()
}

fn verified_reference_standard() -> VerifiedArtifactBytes {
    let reference = calibration_certificate().reference_standard().clone();
    VerifiedArtifactBytes::verify(reference, reference_standard_fixture_bytes()).unwrap()
}

fn verified_calibration_with_reference_standard_scope(
    standard_min: f64,
    standard_max: f64,
    standard_valid_from: u64,
    standard_valid_until: u64,
) -> Result<VerifiedCalibrationEvidence, ContractError> {
    let standard_certificate = CalibrationCertificate::new(
        "reference-standard-01",
        InstrumentIdentity::new(
            "frequency-standard-lab-01",
            "reference-output-a",
        )
        .unwrap(),
        Quantity::Frequency,
        Unit::Hertz,
        standard_min,
        standard_max,
        standard_valid_from,
        standard_valid_until,
        100.0,
        "primary-metrology-lab-01",
        "frequency-reference-procedure-v1",
        artifact(
            "primary-reference-standard-01",
            &sha256_hex(&primary_reference_standard_fixture_bytes()),
        ),
    )
    .unwrap();
    let standard_bytes = standard_certificate.to_canonical_bytes();
    let standard_reference =
        artifact("reference-standard-01", &sha256_hex(&standard_bytes));
    let certificate = CalibrationCertificate::new(
        "calibration-17",
        instrument_identity(),
        Quantity::Frequency,
        Unit::Hertz,
        4_000_000.0,
        6_000_000.0,
        SAMPLE_TIME_NS - 1,
        SAMPLE_TIME_NS + 2_000_000_000,
        900.0,
        "calibration-lab-01",
        "ultrasound-frequency-procedure-v1",
        standard_reference.clone(),
    )
    .unwrap();
    let certificate_bytes = certificate.to_canonical_bytes();
    let certificate_reference = artifact(
        "calibration-certificate",
        &sha256_hex(&certificate_bytes),
    );
    let review_bytes =
        signed_review_fixture_bytes(&certificate_reference, "calibration-17");
    let review = verify_review_bytes(
        review_bytes,
        &certificate_reference,
        "calibration-17",
        "independent-review-receipt",
    )
    .unwrap();
    let resolved = ResolvedCalibration::new(
        "calibration-17",
        instrument_identity(),
        certificate_reference.clone(),
        review.artifact().reference().clone(),
        Quantity::Frequency,
        Unit::Hertz,
        4_000_000.0,
        6_000_000.0,
        SAMPLE_TIME_NS - 1,
        SAMPLE_TIME_NS + 2_000_000_000,
        900.0,
    )
    .unwrap();

    VerifiedCalibrationEvidence::new(
        resolved,
        VerifiedArtifactBytes::verify(certificate_reference, certificate_bytes).unwrap(),
        VerifiedArtifactBytes::verify(standard_reference, standard_bytes).unwrap(),
        review,
    )
}

const REVIEW_NOW_UNIX_S: u64 = 1_750_000_000;

fn review_signing_key() -> SigningKey {
    SigningKey::from_bytes(&[0x42; 32])
}

fn calibration_digest_hex() -> String {
    sha256_hex(&calibration_fixture_bytes())
}

fn calibration_artifact() -> ArtifactReference {
    artifact("calibration-certificate", &calibration_digest_hex())
}

fn signed_review_fixture_bytes(
    certificate: &ArtifactReference,
    record_id: &str,
) -> Vec<u8> {
    let payload = CalibrationReviewPayload::new(
        "fixture-reviewer-01",
        record_id,
        certificate,
        REVIEW_NOW_UNIX_S - 1,
        REVIEW_NOW_UNIX_S + 50,
    )
    .unwrap();
    let signature = review_signing_key()
        .sign(&payload.signing_bytes())
        .to_bytes();
    SignedCalibrationReviewReceipt::new(payload, signature).to_canonical_bytes()
}

fn review_fixture_bytes_for_certificate_digest(certificate_digest: &str) -> Vec<u8> {
    let certificate = artifact("calibration-certificate", certificate_digest);
    signed_review_fixture_bytes(&certificate, "calibration-17")
}

fn review_fixture_bytes() -> Vec<u8> {
    signed_review_fixture_bytes(&calibration_artifact(), "calibration-17")
}

fn review_digest_hex() -> String {
    sha256_hex(&review_fixture_bytes())
}

fn review_artifact() -> ArtifactReference {
    artifact("independent-review-receipt", &review_digest_hex())
}

fn review_artifact_for_bytes(bytes: &[u8]) -> ArtifactReference {
    artifact("independent-review-receipt", &sha256_hex(bytes))
}

fn review_trust_authority_key() -> SigningKey {
    SigningKey::from_bytes(&[0x24; 32])
}

fn review_trust_store() -> VerifiedReviewTrustStore {
    let store = ReviewTrustStore::new(
        4,
        REVIEW_NOW_UNIX_S - 10,
        REVIEW_NOW_UNIX_S + 100,
        vec![
            ReviewTrustKey::new(
                "fixture-reviewer-01",
                review_signing_key().verifying_key().to_bytes(),
                REVIEW_NOW_UNIX_S - 100,
                Some(REVIEW_NOW_UNIX_S + 500),
                None,
            )
            .unwrap()
            .with_reviewer_authorization(
                "fixture-reviewer-person-01",
                "independent-calibration-lab-01",
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let authority = review_trust_authority_key();
    let signature = authority
        .sign(&SignedReviewTrustStoreSnapshot::signing_message(&store))
        .to_bytes();
    SignedReviewTrustStoreSnapshot::new(store, signature)
        .verify(&authority.verifying_key())
        .unwrap()
}

fn verify_review_artifact(
    reference: ArtifactReference,
    bytes: Vec<u8>,
    certificate: &ArtifactReference,
    record_id: &str,
    trust_guard: &mut ReviewTrustStoreGuard,
) -> Result<VerifiedCalibrationReviewReceipt, String> {
    let verified = VerifiedArtifactBytes::verify(reference, bytes)
        .map_err(|_| "calibration review receipt SHA-256 mismatch".to_owned())?;
    let authorization_policy =
        ReviewAuthorizationPolicy::new("calibration-lab-01", Vec::new()).unwrap();
    VerifiedCalibrationReviewReceipt::verify_for_calibration(
        verified,
        &review_trust_store(),
        trust_guard,
        record_id,
        certificate,
        &authorization_policy,
        REVIEW_NOW_UNIX_S,
    )
    .map_err(|error| match error {
        ReviewVerificationError::CertificateReferenceMismatch => {
            "calibration review receipt does not bind the certificate SHA-256".to_owned()
        }
        ReviewVerificationError::InvalidSignature => {
            "calibration review receipt signature is invalid".to_owned()
        }
        other => format!("calibration review verification failed: {other}"),
    })
}

fn verify_review_bytes(
    bytes: Vec<u8>,
    certificate: &ArtifactReference,
    record_id: &str,
    artifact_id: &str,
) -> Result<VerifiedCalibrationReviewReceipt, String> {
    let reference = artifact(artifact_id, &sha256_hex(&bytes));
    let mut trust_guard = ReviewTrustStoreGuard::default();
    verify_review_artifact(reference, bytes, certificate, record_id, &mut trust_guard)
}

fn clock_domain() -> ClockDomainId {
    ClockDomainId::new("ultrasound-rig-boot-epoch-01").unwrap()
}

fn instrument_identity() -> InstrumentIdentity {
    InstrumentIdentity::new("ultrasound-research-rig-01", "rf-channel-0").unwrap()
}

fn artifact(id: &str, sha: &str) -> ArtifactReference {
    ArtifactReference::new(id, sha).unwrap()
}

fn calibration_reference() -> CalibrationReference {
    CalibrationReference::new("calibration-17", calibration_artifact()).unwrap()
}

fn resolved_calibration() -> ResolvedCalibration {
    ResolvedCalibration::new(
        "calibration-17",
        instrument_identity(),
        calibration_artifact(),
        review_artifact(),
        Quantity::Frequency,
        Unit::Hertz,
        4_000_000.0,
        6_000_000.0,
        SAMPLE_TIME_NS - 1,
        SAMPLE_TIME_NS + 2_000_000_000,
        900.0,
    )
    .unwrap()
}

fn calibration_fixture_for(
    record_id: &str,
    instrument: InstrumentIdentity,
    quantity: Quantity,
    unit: Unit,
    range_min_value: f64,
    range_max_value: f64,
    valid_from_ns: u64,
    valid_until_ns: u64,
    standard_uncertainty: f64,
) -> (ResolvedCalibration, Vec<u8>, Vec<u8>) {
    let certificate = CalibrationCertificate::new(
        record_id,
        instrument.clone(),
        quantity,
        unit,
        range_min_value,
        range_max_value,
        valid_from_ns,
        valid_until_ns,
        standard_uncertainty,
        "calibration-lab-01",
        "ultrasound-frequency-procedure-v1",
        artifact(
            "reference-standard-01",
            &sha256_hex(b"traceable frequency reference standard fixture v1"),
        ),
    )
    .unwrap();
    let certificate_bytes = certificate.to_canonical_bytes();
    let certificate_reference =
        artifact("calibration-certificate", &sha256_hex(&certificate_bytes));
    let review_bytes = signed_review_fixture_bytes(&certificate_reference, record_id);
    let review_reference = review_artifact_for_bytes(&review_bytes);
    let resolved = ResolvedCalibration::new(
        record_id,
        instrument,
        certificate_reference,
        review_reference,
        quantity,
        unit,
        range_min_value,
        range_max_value,
        valid_from_ns,
        valid_until_ns,
        standard_uncertainty,
    )
    .unwrap();
    (resolved, certificate_bytes, review_bytes)
}

fn measurement(sequence: u64, captured_at_ns: u64) -> MeasurementEnvelope {
    measurement_with_calibration(
        sequence,
        captured_at_ns,
        calibration_reference(),
    )
}

fn measurement_with_calibration(
    sequence: u64,
    captured_at_ns: u64,
    calibration: CalibrationReference,
) -> MeasurementEnvelope {
    measurement_with_calibration_in_clock_domain(
        sequence,
        captured_at_ns,
        clock_domain(),
        calibration,
    )
}

fn measurement_in_clock_domain(
    sequence: u64,
    captured_at_ns: u64,
    clock_domain: ClockDomainId,
) -> MeasurementEnvelope {
    measurement_with_calibration_in_clock_domain(
        sequence,
        captured_at_ns,
        clock_domain,
        calibration_reference(),
    )
}

fn measurement_with_calibration_in_clock_domain(
    sequence: u64,
    captured_at_ns: u64,
    clock_domain: ClockDomainId,
    calibration: CalibrationReference,
) -> MeasurementEnvelope {
    MeasurementEnvelope::new(MeasurementInput {
        identity: instrument_identity(),
        sequence,
        captured_at_ns,
        clock_domain,
        quantity: Quantity::Frequency,
        unit: Unit::Hertz,
        value: 5_000_000.0,
        standard_uncertainty: 1_000.0,
        calibration: Some(calibration),
        raw_data: Some(raw_reference()),
        processing_chain_version: "sha256:reconstruction-pipeline-v1".to_owned(),
        quality_flags: vec![],
    })
    .unwrap()
}

fn policy() -> MeasurementPolicy {
    MeasurementPolicy::new(500_000_000, Unit::Hertz, 2_000.0, 1_000.0).unwrap()
}

struct ImmutableFixtureReader {
    inner: Cursor<Vec<u8>>,
    artifact_id: String,
    version_id: String,
}

impl ImmutableFixtureReader {
    fn new(
        bytes: Vec<u8>,
        artifact_id: impl Into<String>,
        version_id: impl Into<String>,
    ) -> Self {
        Self {
            inner: Cursor::new(bytes),
            artifact_id: artifact_id.into(),
            version_id: version_id.into(),
        }
    }
}

impl Read for ImmutableFixtureReader {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(buffer)
    }
}

impl ImmutableArtifactReader for ImmutableFixtureReader {
    fn artifact_id(&self) -> &str {
        &self.artifact_id
    }

    fn immutable_version_id(&self) -> &str {
        &self.version_id
    }
}

struct MockResolver {
    resolved: Result<ResolvedCalibration, String>,
    raw_bytes: Result<Vec<u8>, String>,
    raw_reference_override: Option<ArtifactReference>,
    calibration_bytes_override: Option<Vec<u8>>,
    review_bytes_override: Option<Vec<u8>>,
    reference_standard_bytes_override: Option<Vec<u8>>,
    review_trust_guard: Mutex<ReviewTrustStoreGuard>,
}

impl CalibrationEvidenceResolver for MockResolver {
    fn resolve_calibration(
        &self,
        _reference: &CalibrationReference,
    ) -> Result<VerifiedCalibrationEvidence, String> {
        let resolved = self.resolved.clone()?;
        let certificate_bytes = self
            .calibration_bytes_override
            .clone()
            .unwrap_or_else(calibration_fixture_bytes);
        let verified_certificate =
            VerifiedArtifactBytes::verify(resolved.evidence().clone(), certificate_bytes)
                .map_err(|_| "calibration certificate SHA-256 mismatch".to_owned())?;

        let parsed_certificate = CalibrationCertificate::from_canonical_bytes(
            verified_certificate.bytes(),
        )
        .map_err(|error| format!("calibration certificate parse failed: {error}"))?;
        let standard_bytes = self
            .reference_standard_bytes_override
            .clone()
            .unwrap_or_else(reference_standard_fixture_bytes);
        let verified_reference_standard = VerifiedArtifactBytes::verify(
            parsed_certificate.reference_standard().clone(),
            standard_bytes,
        )
        .map_err(|_| "reference standard SHA-256 mismatch".to_owned())?;

        let review_bytes = self
            .review_bytes_override
            .clone()
            .unwrap_or_else(|| signed_review_fixture_bytes(resolved.evidence(), "calibration-17"));
        let mut trust_guard = self
            .review_trust_guard
            .lock()
            .map_err(|_| "review trust guard mutex poisoned".to_owned())?;
        let verified_review = verify_review_artifact(
            resolved.review_receipt().clone(),
            review_bytes,
            resolved.evidence(),
            "calibration-17",
            &mut trust_guard,
        )?;

        VerifiedCalibrationEvidence::new(
            resolved,
            verified_certificate,
            verified_reference_standard,
            verified_review,
        )
        .map_err(|error| error.to_string())
    }
}

impl RawDataEvidenceResolver for MockResolver {
    fn resolve_raw_data(
        &self,
        reference: &RawDataReference,
    ) -> Result<ResolvedRawData, String> {
        let bytes = self.raw_bytes.clone()?;
        let reference_to_verify = self
            .raw_reference_override
            .clone()
            .unwrap_or_else(|| reference.artifact().clone());
        let source_artifact_id = reference_to_verify.artifact_id().to_owned();
        let version_id = format!("fixture-object-version:{source_artifact_id}:v1");
        let verified = VerifiedArtifactDigest::verify_immutable_reader(
            reference_to_verify,
            ImmutableFixtureReader::new(bytes, source_artifact_id, version_id),
        )
        .map_err(|_| "raw acquisition digest or source version is invalid".to_owned())?;
        ResolvedRawData::from_verified_digest(verified, "application/octet-stream")
            .map_err(|error| error.to_string())
    }
}

fn resolver() -> MockResolver {
    MockResolver {
        resolved: Ok(resolved_calibration()),
        raw_bytes: Ok(raw_fixture_bytes()),
        raw_reference_override: None,
        calibration_bytes_override: None,
        review_bytes_override: None,
        reference_standard_bytes_override: None,
        review_trust_guard: Mutex::new(ReviewTrustStoreGuard::default()),
    }
}

#[test]
fn signature_only_review_receipt_cannot_construct_verified_calibration_evidence() {
    let resolved = resolved_calibration();
    let certificate = VerifiedArtifactBytes::verify(
        resolved.evidence().clone(),
        calibration_fixture_bytes(),
    )
    .unwrap();
    let standard = VerifiedArtifactBytes::verify(
        calibration_certificate().reference_standard().clone(),
        reference_standard_fixture_bytes(),
    )
    .unwrap();
    let review = VerifiedArtifactBytes::verify(
        resolved.review_receipt().clone(),
        review_fixture_bytes(),
    )
    .unwrap();
    let mut guard = ReviewTrustStoreGuard::default();
    let signature_verified_only = VerifiedCalibrationReviewReceipt::verify(
        review,
        &review_trust_store(),
        &mut guard,
        resolved.record_id(),
        resolved.evidence(),
        REVIEW_NOW_UNIX_S,
    )
    .unwrap();

    assert_eq!(
        VerifiedCalibrationEvidence::new(
            resolved,
            certificate,
            standard,
            signature_verified_only,
        )
        .unwrap_err(),
        ContractError::CalibrationReviewAuthorizationMissing
    );
}

#[test]
fn creates_a_unit_consistent_measurement_envelope() {
    let m = measurement(1, SAMPLE_TIME_NS);
    assert_eq!(m.quantity(), Quantity::Frequency);
    assert_eq!(m.unit(), Unit::Hertz);
    assert_eq!(m.value(), 5_000_000.0);
    assert_eq!(m.clock_domain().as_str(), "ultrasound-rig-boot-epoch-01");
    assert_eq!(m.identity().instrument_id(), "ultrasound-research-rig-01");
    assert_eq!(m.processing_chain_version(), "sha256:reconstruction-pipeline-v1");
}

#[test]
fn rejects_mismatched_quantity_and_unit() {
    let result = MeasurementEnvelope::new(MeasurementInput {
        identity: InstrumentIdentity::new("instrument", "channel").unwrap(),
        sequence: 1,
        captured_at_ns: SAMPLE_TIME_NS,
        clock_domain: clock_domain(),
        quantity: Quantity::Frequency,
        unit: Unit::Pascal,
        value: 1.0,
        standard_uncertainty: 0.1,
        calibration: None,
        raw_data: None,
        processing_chain_version: "capture-v1".into(),
        quality_flags: vec![],
    });
    assert!(matches!(
        result,
        Err(ContractError::QuantityUnitMismatch {
            quantity: Quantity::Frequency,
            unit: Unit::Pascal
        })
    ));
}

#[test]
fn rejects_non_finite_values_and_uncertainties() {
    let mut input = MeasurementInput {
        identity: InstrumentIdentity::new("instrument", "channel").unwrap(),
        sequence: 1,
        captured_at_ns: SAMPLE_TIME_NS,
        clock_domain: clock_domain(),
        quantity: Quantity::Frequency,
        unit: Unit::Hertz,
        value: 1.0,
        standard_uncertainty: 0.1,
        calibration: None,
        raw_data: None,
        processing_chain_version: "capture-v1".into(),
        quality_flags: vec![],
    };
    input.value = f64::NAN;
    assert_eq!(
        MeasurementEnvelope::new(input.clone()).unwrap_err(),
        ContractError::NonFiniteValue("value")
    );
    input.value = 1.0;
    input.standard_uncertainty = f64::INFINITY;
    assert_eq!(
        MeasurementEnvelope::new(input.clone()).unwrap_err(),
        ContractError::NonFiniteValue("standard_uncertainty")
    );
    input.standard_uncertainty = -0.1;
    assert_eq!(
        MeasurementEnvelope::new(input).unwrap_err(),
        ContractError::NegativeUncertainty("standard_uncertainty")
    );
}

#[test]
fn rejects_empty_ids_processing_version_and_invalid_digests() {
    assert_eq!(
        InstrumentIdentity::new(" ", "channel").unwrap_err(),
        ContractError::EmptyIdentifier("instrument_id")
    );
    assert_eq!(
        InstrumentIdentity::new("instrument", "\t").unwrap_err(),
        ContractError::EmptyIdentifier("channel_id")
    );
    assert_eq!(
        ArtifactReference::new("calibration", "not-a-digest").unwrap_err(),
        ContractError::InvalidSha256
    );
    assert_eq!(
        ClockDomainId::new(" ").unwrap_err(),
        ContractError::EmptyIdentifier("clock_domain_id")
    );
    let result = MeasurementEnvelope::new(MeasurementInput {
        identity: InstrumentIdentity::new("instrument", "channel").unwrap(),
        sequence: 1,
        captured_at_ns: SAMPLE_TIME_NS,
        clock_domain: clock_domain(),
        quantity: Quantity::Frequency,
        unit: Unit::Hertz,
        value: 1.0,
        standard_uncertainty: 0.0,
        calibration: None,
        raw_data: None,
        processing_chain_version: "   ".into(),
        quality_flags: vec![],
    });
    assert_eq!(result.unwrap_err(), ContractError::EmptyProcessingVersion);
}

#[test]
fn calibration_intervals_must_be_non_empty() {
    assert_eq!(
        ResolvedCalibration::new(
            "calibration",
            instrument_identity(),
            artifact("certificate", SHA_A),
            review_artifact(),
            Quantity::Frequency,
            Unit::Hertz,
            4_000_000.0,
            6_000_000.0,
            42,
            42,
            0.0,
        )
        .unwrap_err(),
        ContractError::InvalidValidityInterval
    );
}

#[test]
fn calibration_evidence_and_review_receipt_must_be_distinct_artifacts() {
    let cases = [
        // Exact duplicate artifact.
        (artifact("same-artifact", SHA_A), artifact("same-artifact", SHA_A)),
        // Different aliases for identical content are not independent evidence.
        (artifact("certificate", SHA_A), artifact("review", SHA_A)),
        // Same alias pointing at different content is also suspicious.
        (artifact("same-id", SHA_A), artifact("same-id", SHA_B)),
    ];

    for (evidence, receipt) in cases {
        assert_eq!(
            ResolvedCalibration::new(
                "calibration-17",
                instrument_identity(),
                evidence,
                receipt,
                Quantity::Frequency,
                Unit::Hertz,
                4_000_000.0,
                6_000_000.0,
                SAMPLE_TIME_NS - 1,
                SAMPLE_TIME_NS + 2_000_000_000,
                900.0,
            )
            .unwrap_err(),
            ContractError::CalibrationReviewMustBeDistinct
        );
    }
}

#[test]
fn calibration_range_constructor_rejects_inverted_bounds() {
    assert_eq!(
        ResolvedCalibration::new(
            "bad-range-calibration",
            instrument_identity(),
            artifact("certificate", SHA_A),
            review_artifact(),
            Quantity::Frequency,
            Unit::Hertz,
            6_000_000.0,
            4_000_000.0,
            SAMPLE_TIME_NS - 1,
            SAMPLE_TIME_NS + 2_000_000_000,
            900.0,
        )
        .unwrap_err(),
        ContractError::InvalidCalibrationRange
    );
}

#[test]
fn quantitative_gate_rejects_calibration_for_another_instrument_channel() {
    let wrong_identity =
        InstrumentIdentity::new("another-ultrasound-rig", "rf-channel-0").unwrap();
    let (resolved, certificate_bytes, review_bytes) = calibration_fixture_for(
        "calibration-17",
        wrong_identity,
        Quantity::Frequency,
        Unit::Hertz,
        4_000_000.0,
        6_000_000.0,
        SAMPLE_TIME_NS - 1,
        SAMPLE_TIME_NS + 2_000_000_000,
        900.0,
    );
    let calibration_reference =
        CalibrationReference::new("calibration-17", resolved.evidence().clone()).unwrap();
    let wrong_instrument = MockResolver {
        resolved: Ok(resolved),
        raw_bytes: Ok(raw_fixture_bytes()),
        raw_reference_override: None,
        calibration_bytes_override: Some(certificate_bytes),
        review_bytes_override: Some(review_bytes),
        reference_standard_bytes_override: None,
        review_trust_guard: Mutex::new(ReviewTrustStoreGuard::default()),
    };

    assert_eq!(
        measurement_with_calibration(1, SAMPLE_TIME_NS, calibration_reference)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &policy(),
                &wrong_instrument,
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::CalibrationInstrumentMismatch
    );
}
#[test]
fn quantitative_gate_rejects_values_outside_the_calibrated_range() {
    let (resolved, certificate_bytes, review_bytes) = calibration_fixture_for(
        "calibration-17",
        instrument_identity(),
        Quantity::Frequency,
        Unit::Hertz,
        4_000_000.0,
        4_900_000.0,
        SAMPLE_TIME_NS - 1,
        SAMPLE_TIME_NS + 2_000_000_000,
        900.0,
    );
    let calibration_reference =
        CalibrationReference::new("calibration-17", resolved.evidence().clone()).unwrap();
    let narrow_range = MockResolver {
        resolved: Ok(resolved),
        raw_bytes: Ok(raw_fixture_bytes()),
        raw_reference_override: None,
        calibration_bytes_override: Some(certificate_bytes),
        review_bytes_override: Some(review_bytes),
        reference_standard_bytes_override: None,
        review_trust_guard: Mutex::new(ReviewTrustStoreGuard::default()),
    };

    assert_eq!(
        measurement_with_calibration(1, SAMPLE_TIME_NS, calibration_reference)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &policy(),
                &narrow_range,
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::CalibrationRangeExceeded {
            value: 5_000_000.0,
            minimum: 4_000_000.0,
            maximum: 4_900_000.0,
        }
    );
}
#[test]
fn assessment_passes_only_with_resolved_applicable_calibration_and_complete_provenance() {
    let assessment = measurement(1, SAMPLE_TIME_NS)
        .assess_for_quantitative_use(
            SAMPLE_TIME_NS + 10,
            &clock_domain(),
            &policy(),
            &resolver(),
            &mut MeasurementStreamGuard::default(),
        )
        .unwrap();
    assert_eq!(assessment.instrument().instrument_id(), "ultrasound-research-rig-01");
    assert_eq!(assessment.sequence(), 1);
    assert_eq!(assessment.captured_at_ns(), SAMPLE_TIME_NS);
    assert_eq!(assessment.clock_domain().as_str(), "ultrasound-rig-boot-epoch-01");
    assert_eq!(assessment.quantity(), Quantity::Frequency);
    assert_eq!(assessment.unit(), Unit::Hertz);
    assert_eq!(assessment.value(), 5_000_000.0);
    assert_eq!(assessment.age_ns(), 10);
    assert_eq!(assessment.measurement_standard_uncertainty(), 1_000.0);
    assert_eq!(assessment.calibration_standard_uncertainty(), 900.0);
    assert_eq!(assessment.calibration_record_id(), "calibration-17");
    assert_eq!(
        assessment.calibration_evidence().artifact_id(),
        "calibration-certificate"
    );
    assert_eq!(
        assessment.calibration_range(),
        (4_000_000.0, 6_000_000.0)
    );
    assert_eq!(assessment.calibration_review_receipt().artifact_id(), "independent-review-receipt");
    assert_eq!(assessment.calibration_provider_id(), "calibration-lab-01");
    assert_eq!(
        assessment.calibration_procedure_id(),
        "ultrasound-frequency-procedure-v1"
    );
    assert_eq!(
        assessment.calibration_reference_standard().artifact_id(),
        "reference-standard-01"
    );
    let review_authorization = assessment.calibration_review_authorization();
    assert_eq!(review_authorization.reviewer_id(), "fixture-reviewer-person-01");
    assert_eq!(
        review_authorization.reviewer_organization_id(),
        "independent-calibration-lab-01"
    );
    assert_eq!(
        review_authorization.provider_organization_id(),
        assessment.calibration_provider_id()
    );
    assert_eq!(review_authorization.policy_sha256().len(), 64);
    assert_eq!(review_authorization.trust_store_sequence(), 4);
    assert_eq!(review_authorization.trust_store_sha256().len(), 64);
    assert_eq!(review_authorization.trust_authority_key_sha256().len(), 64);
    assert_eq!(review_authorization.verified_at_unix_s(), REVIEW_NOW_UNIX_S);
    assert_eq!(assessment.raw_data().artifact().artifact_id(), "raw-acquisition-1");
    assert_eq!(assessment.raw_data().byte_length(), 4_096);
    assert_eq!(assessment.raw_data().media_type(), "application/octet-stream");
    assert_eq!(
        assessment.raw_data().source_version_id(),
        "fixture-object-version:raw-acquisition-1:v1"
    );
}

#[test]
fn rejects_measurements_from_a_future_clock_or_beyond_freshness_limit() {
    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS - 1,
                &clock_domain(),
                &policy(),
                &resolver(),
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::ClockInFuture
    );
    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS + 500_000_001,
                &clock_domain(),
                &policy(),
                &resolver(),
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::Stale {
            age_ns: 500_000_001,
            max_age_ns: 500_000_000
        }
    );
}

#[test]
fn any_quality_flag_blocks_strict_quantitative_use() {
    let mut input = MeasurementInput {
        identity: InstrumentIdentity::new("instrument", "channel").unwrap(),
        sequence: 1,
        captured_at_ns: SAMPLE_TIME_NS,
        clock_domain: clock_domain(),
        quantity: Quantity::Frequency,
        unit: Unit::Hertz,
        value: 1.0,
        standard_uncertainty: 0.1,
        calibration: Some(calibration_reference()),
        raw_data: Some(RawDataReference::new(artifact("raw", SHA_B))),
        processing_chain_version: "capture-v1".into(),
        quality_flags: vec![QualityFlag::Saturated, QualityFlag::MotionArtifact],
    };
    input.quality_flags.dedup();
    let m = MeasurementEnvelope::new(input).unwrap();
    assert_eq!(
        m.assess_for_quantitative_use(
            SAMPLE_TIME_NS,
            &clock_domain(),
            &policy(),
            &resolver(),
            &mut MeasurementStreamGuard::default(),
        )
        .unwrap_err(),
        AssessmentFailure::QualityFlagsPresent(vec![
            QualityFlag::Saturated,
            QualityFlag::MotionArtifact
        ])
    );
}

#[test]
fn missing_raw_data_or_calibration_reference_fails_closed() {
    let mut input = MeasurementInput {
        identity: InstrumentIdentity::new("instrument", "channel").unwrap(),
        sequence: 1,
        captured_at_ns: SAMPLE_TIME_NS,
        clock_domain: clock_domain(),
        quantity: Quantity::Frequency,
        unit: Unit::Hertz,
        value: 1.0,
        standard_uncertainty: 0.1,
        calibration: Some(calibration_reference()),
        raw_data: None,
        processing_chain_version: "capture-v1".into(),
        quality_flags: vec![],
    };
    let no_raw = MeasurementEnvelope::new(input.clone()).unwrap();
    assert_eq!(
        no_raw.assess_for_quantitative_use(
            SAMPLE_TIME_NS,
            &clock_domain(),
            &policy(),
            &resolver(),
            &mut MeasurementStreamGuard::default(),
        )
        .unwrap_err(),
        AssessmentFailure::MissingRawDataReference
    );
    input.raw_data = Some(RawDataReference::new(artifact("raw", SHA_B)));
    input.calibration = None;
    let no_cal = MeasurementEnvelope::new(input).unwrap();
    assert_eq!(
        no_cal.assess_for_quantitative_use(
            SAMPLE_TIME_NS,
            &clock_domain(),
            &policy(),
            &resolver(),
            &mut MeasurementStreamGuard::default(),
        )
        .unwrap_err(),
        AssessmentFailure::MissingCalibrationReference
    );
}

#[test]
fn rejects_empty_or_malformed_resolved_raw_data_metadata() {
    let empty_digest = sha256_hex(b"");
    let empty_verified = VerifiedArtifactDigest::verify_reader(
        artifact("empty-raw", &empty_digest),
        Cursor::new(Vec::new()),
    )
    .unwrap();
    assert_eq!(
        ResolvedRawData::from_verified_digest(empty_verified, "application/octet-stream")
            .unwrap_err(),
        ContractError::EmptyRawDataArtifact
    );

    let unversioned = VerifiedArtifactDigest::verify_reader(
        raw_artifact(),
        Cursor::new(raw_fixture_bytes()),
    )
    .unwrap();
    assert_eq!(
        ResolvedRawData::from_verified_digest(
            unversioned,
            "application/octet-stream",
        )
        .unwrap_err(),
        ContractError::ImmutableSourceVersionRequired
    );

    let versioned_reference = raw_artifact();
    let versioned = VerifiedArtifactDigest::verify_immutable_reader(
        versioned_reference,
        ImmutableFixtureReader::new(
            raw_fixture_bytes(),
            "raw-acquisition-1",
            "fixture-object-version:raw-acquisition-1:v1",
        ),
    )
    .unwrap();
    assert_eq!(
        ResolvedRawData::from_verified_digest(versioned, " ").unwrap_err(),
        ContractError::EmptyIdentifier("raw_data_media_type")
    );
}

#[test]
fn unresolved_raw_acquisition_evidence_fails_closed() {
    let missing_raw = MockResolver {
        resolved: Ok(resolved_calibration()),
        raw_bytes: Err("raw artifact bytes not found".into()),
        raw_reference_override: None,
        calibration_bytes_override: None,
        review_bytes_override: None,
        reference_standard_bytes_override: None,
        review_trust_guard: Mutex::new(ReviewTrustStoreGuard::default()),
    };

    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &policy(),
                &missing_raw,
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::RawDataEvidenceUnresolved("raw artifact bytes not found".into())
    );
}

#[test]
fn raw_resolver_rejects_bytes_whose_sha256_does_not_match_the_reference() {
    let tampered = MockResolver {
        resolved: Ok(resolved_calibration()),
        raw_bytes: Ok(b"tampered acquisition bytes".to_vec()),
        raw_reference_override: None,
        calibration_bytes_override: None,
        review_bytes_override: None,
        reference_standard_bytes_override: None,
        review_trust_guard: Mutex::new(ReviewTrustStoreGuard::default()),
    };

    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &policy(),
                &tampered,
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::RawDataEvidenceUnresolved("raw acquisition SHA-256 mismatch".into())
    );
}

#[test]
fn resolved_raw_data_must_match_the_envelope_digest_and_artifact_id() {
    let mismatched_raw = MockResolver {
        resolved: Ok(resolved_calibration()),
        raw_bytes: Ok(raw_fixture_bytes()),
        raw_reference_override: Some(artifact("other-acquisition", &raw_digest_hex())),
        calibration_bytes_override: None,
        review_bytes_override: None,
        reference_standard_bytes_override: None,
        review_trust_guard: Mutex::new(ReviewTrustStoreGuard::default()),
    };

    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &policy(),
                &mismatched_raw,
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::RawDataReferenceMismatch
    );
}

#[test]
fn unresolved_calibration_evidence_fails_closed() {
    let missing = MockResolver {
        resolved: Err("calibration artifact not found".into()),
        raw_bytes: Ok(raw_fixture_bytes()),
        raw_reference_override: None,
        calibration_bytes_override: None,
        review_bytes_override: None,
        reference_standard_bytes_override: None,
        review_trust_guard: Mutex::new(ReviewTrustStoreGuard::default()),
    };
    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &policy(),
                &missing,
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::CalibrationEvidenceUnresolved("calibration artifact not found".into())
    );
}

#[test]
fn calibration_resolver_rejects_certificate_bytes_with_a_wrong_digest() {
    let tampered = MockResolver {
        resolved: Ok(resolved_calibration()),
        raw_bytes: Ok(raw_fixture_bytes()),
        raw_reference_override: None,
        calibration_bytes_override: Some(b"tampered calibration certificate".to_vec()),
        review_bytes_override: None,
        reference_standard_bytes_override: None,
        review_trust_guard: Mutex::new(ReviewTrustStoreGuard::default()),
    };
    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &policy(),
                &tampered,
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::CalibrationEvidenceUnresolved(
            "calibration certificate SHA-256 mismatch".into()
        )
    );
}

#[test]
fn verified_calibration_rejects_reference_standard_with_wrong_quantity() {
    let incompatible_standard = CalibrationCertificate::new(
        "reference-standard-01",
        InstrumentIdentity::new("acoustic-standard-lab-01", "pressure-output-a").unwrap(),
        Quantity::AcousticPressure,
        Unit::Pascal,
        0.0,
        10.0,
        SAMPLE_TIME_NS - 10_000,
        SAMPLE_TIME_NS + 2_000_000_000,
        0.1,
        "primary-metrology-lab-01",
        "pressure-reference-procedure-v1",
        artifact(
            "primary-reference-standard-01",
            &sha256_hex(&primary_reference_standard_fixture_bytes()),
        ),
    )
    .unwrap();
    let standard_bytes = incompatible_standard.to_canonical_bytes();
    let standard_reference = artifact(
        "reference-standard-01",
        &sha256_hex(&standard_bytes),
    );

    let certificate = CalibrationCertificate::new(
        "calibration-17",
        instrument_identity(),
        Quantity::Frequency,
        Unit::Hertz,
        4_000_000.0,
        6_000_000.0,
        SAMPLE_TIME_NS - 1,
        SAMPLE_TIME_NS + 2_000_000_000,
        900.0,
        "calibration-lab-01",
        "ultrasound-frequency-procedure-v1",
        standard_reference.clone(),
    )
    .unwrap();
    let certificate_bytes = certificate.to_canonical_bytes();
    let certificate_reference = artifact(
        "calibration-certificate",
        &sha256_hex(&certificate_bytes),
    );
    let review_bytes = signed_review_fixture_bytes(&certificate_reference, "calibration-17");
    let review = verify_review_bytes(
        review_bytes,
        &certificate_reference,
        "calibration-17",
        "independent-review-receipt",
    )
    .unwrap();
    let resolved = ResolvedCalibration::new(
        "calibration-17",
        instrument_identity(),
        certificate_reference.clone(),
        review.artifact().reference().clone(),
        Quantity::Frequency,
        Unit::Hertz,
        4_000_000.0,
        6_000_000.0,
        SAMPLE_TIME_NS - 1,
        SAMPLE_TIME_NS + 2_000_000_000,
        900.0,
    )
    .unwrap();

    assert_eq!(
        VerifiedCalibrationEvidence::new(
            resolved,
            VerifiedArtifactBytes::verify(certificate_reference, certificate_bytes).unwrap(),
            VerifiedArtifactBytes::verify(standard_reference, standard_bytes).unwrap(),
            review,
        )
        .unwrap_err(),
        ContractError::ReferenceStandardClaimMismatch("quantity")
    );
}

#[test]
fn verified_calibration_rejects_reference_standard_range_that_does_not_cover_calibration() {
    assert_eq!(
        verified_calibration_with_reference_standard_scope(
            4_100_000.0,
            6_000_000.0,
            SAMPLE_TIME_NS - 10_000,
            SAMPLE_TIME_NS + 2_000_000_000,
        )
        .unwrap_err(),
        ContractError::ReferenceStandardClaimMismatch("calibration_range_not_covered")
    );
}

#[test]
fn verified_calibration_rejects_reference_standard_validity_that_does_not_cover_calibration() {
    assert_eq!(
        verified_calibration_with_reference_standard_scope(
            4_000_000.0,
            6_000_000.0,
            SAMPLE_TIME_NS,
            SAMPLE_TIME_NS + 2_000_000_000,
        )
        .unwrap_err(),
        ContractError::ReferenceStandardClaimMismatch("validity_interval_not_covered")
    );
}

#[test]
fn calibration_resolver_rejects_reference_standard_bytes_with_a_wrong_digest() {
    let mut tampered = resolver();
    tampered.reference_standard_bytes_override =
        Some(b"substituted reference standard".to_vec());

    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &policy(),
                &tampered,
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::CalibrationEvidenceUnresolved(
            "reference standard SHA-256 mismatch".into()
        )
    );
}

#[test]
fn calibration_resolver_rejects_review_receipts_with_a_wrong_digest() {
    let tampered = MockResolver {
        resolved: Ok(resolved_calibration()),
        raw_bytes: Ok(raw_fixture_bytes()),
        raw_reference_override: None,
        calibration_bytes_override: None,
        review_bytes_override: Some(b"tampered independent review receipt".to_vec()),
        reference_standard_bytes_override: None,
        review_trust_guard: Mutex::new(ReviewTrustStoreGuard::default()),
    };
    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &policy(),
                &tampered,
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::CalibrationEvidenceUnresolved(
            "calibration review receipt SHA-256 mismatch".into()
        )
    );
}

#[test]
fn verified_calibration_rejects_empty_certificate_or_review_bytes() {
    let empty_digest = sha256_hex(b"");
    let empty_certificate_reference = artifact("calibration-certificate", &empty_digest);
    let empty_certificate = VerifiedArtifactBytes::verify(
        empty_certificate_reference.clone(),
        Vec::new(),
    )
    .unwrap();
    let empty_cert_review_bytes =
        signed_review_fixture_bytes(&empty_certificate_reference, "calibration-17");
    let valid_review = verify_review_bytes(
        empty_cert_review_bytes,
        &empty_certificate_reference,
        "calibration-17",
        "independent-review-receipt",
    )
    .unwrap();
    let record_with_empty_certificate = ResolvedCalibration::new(
        "calibration-17",
        instrument_identity(),
        empty_certificate_reference,
        valid_review.artifact().reference().clone(),
        Quantity::Frequency,
        Unit::Hertz,
        4_000_000.0,
        6_000_000.0,
        SAMPLE_TIME_NS - 1,
        SAMPLE_TIME_NS + 2_000_000_000,
        900.0,
    )
    .unwrap();
    assert_eq!(
        VerifiedCalibrationEvidence::new(
            record_with_empty_certificate,
            empty_certificate,
            verified_reference_standard(),
            valid_review,
        )
        .unwrap_err(),
        ContractError::EmptyEvidenceArtifact("calibration_certificate")
    );

    let empty_review_reference = artifact("independent-review-receipt", &empty_digest);
    let empty_review =
        VerifiedArtifactBytes::verify(empty_review_reference, Vec::new()).unwrap();
    assert_eq!(
        VerifiedCalibrationReviewReceipt::verify(
            empty_review,
            &review_trust_store(),
            &mut ReviewTrustStoreGuard::default(),
            "calibration-17",
            &calibration_artifact(),
            REVIEW_NOW_UNIX_S,
        )
        .unwrap_err(),
        ReviewVerificationError::EmptyReceipt
    );
}

#[test]
fn verified_calibration_requires_exact_certificate_and_review_artifact_references() {
    let calibration = resolved_calibration();
    let wrong_certificate = VerifiedArtifactBytes::verify(
        artifact("another-certificate-id", &calibration_digest_hex()),
        calibration_fixture_bytes(),
    )
    .unwrap();
    let verified_review = verify_review_bytes(
        review_fixture_bytes(),
        &calibration_artifact(),
        "calibration-17",
        "independent-review-receipt",
    )
    .unwrap();
    assert_eq!(
        VerifiedCalibrationEvidence::new(
            calibration.clone(),
            wrong_certificate,
            verified_reference_standard(),
            verified_review,
        )
        .unwrap_err(),
        ContractError::VerifiedArtifactReferenceMismatch("calibration_certificate")
    );

    let verified_certificate =
        VerifiedArtifactBytes::verify(calibration_artifact(), calibration_fixture_bytes())
            .unwrap();
    let verified_review = verify_review_bytes(
        review_fixture_bytes(),
        &calibration_artifact(),
        "calibration-17",
        "independent-review-receipt",
    )
    .unwrap();
    let wrong_standard = VerifiedArtifactBytes::verify(
        artifact(
            "reference-standard-alias",
            &sha256_hex(&reference_standard_fixture_bytes()),
        ),
        reference_standard_fixture_bytes(),
    )
    .unwrap();
    assert_eq!(
        VerifiedCalibrationEvidence::new(
            calibration.clone(),
            verified_certificate,
            wrong_standard,
            verified_review,
        )
        .unwrap_err(),
        ContractError::VerifiedArtifactReferenceMismatch("reference_standard")
    );

    let verified_certificate =
        VerifiedArtifactBytes::verify(calibration_artifact(), calibration_fixture_bytes())
            .unwrap();
    let wrong_review = verify_review_bytes(
        review_fixture_bytes(),
        &calibration_artifact(),
        "calibration-17",
        "another-review-id",
    )
    .unwrap();
    assert_eq!(
        VerifiedCalibrationEvidence::new(
            calibration.clone(),
            verified_certificate,
            verified_reference_standard(),
            wrong_review,
        )
            .unwrap_err(),
        ContractError::VerifiedArtifactReferenceMismatch("calibration_review_receipt")
    );
}

#[test]
fn verified_calibration_rejects_metadata_that_disagrees_with_certificate_claims() {
    let calibration = ResolvedCalibration::new(
        "calibration-17",
        instrument_identity(),
        calibration_artifact(),
        review_artifact(),
        Quantity::Frequency,
        Unit::Hertz,
        4_000_000.0,
        7_000_000.0,
        SAMPLE_TIME_NS - 1,
        SAMPLE_TIME_NS + 2_000_000_000,
        900.0,
    )
    .unwrap();
    let certificate =
        VerifiedArtifactBytes::verify(calibration_artifact(), calibration_fixture_bytes())
            .unwrap();
    let review = verify_review_bytes(
        review_fixture_bytes(),
        &calibration_artifact(),
        "calibration-17",
        "independent-review-receipt",
    )
    .unwrap();

    assert_eq!(
        VerifiedCalibrationEvidence::new(
            calibration,
            certificate,
            verified_reference_standard(),
            review,
        )
        .unwrap_err(),
        ContractError::CalibrationCertificateClaimMismatch("range_max_value")
    );
}

#[test]
fn calibration_resolver_rejects_a_review_bound_to_another_certificate_digest() {
    let wrong_review_bytes = review_fixture_bytes_for_certificate_digest(SHA_A);
    let wrong_review = review_artifact_for_bytes(&wrong_review_bytes);
    let resolved = ResolvedCalibration::new(
        "calibration-17",
        instrument_identity(),
        calibration_artifact(),
        wrong_review,
        Quantity::Frequency,
        Unit::Hertz,
        4_000_000.0,
        6_000_000.0,
        SAMPLE_TIME_NS - 1,
        SAMPLE_TIME_NS + 2_000_000_000,
        900.0,
    )
    .unwrap();
    let wrong_binding = MockResolver {
        resolved: Ok(resolved),
        raw_bytes: Ok(raw_fixture_bytes()),
        raw_reference_override: None,
        calibration_bytes_override: None,
        review_bytes_override: Some(wrong_review_bytes),
        reference_standard_bytes_override: None,
        review_trust_guard: Mutex::new(ReviewTrustStoreGuard::default()),
    };

    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &policy(),
                &wrong_binding,
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::CalibrationEvidenceUnresolved(
            "calibration review receipt does not bind the certificate SHA-256".into()
        )
    );
}

#[test]
fn rejects_a_resolver_result_for_a_different_calibration_artifact() {
    let other_certificate = artifact("different-certificate", &calibration_digest_hex());
    let other_review_bytes =
        signed_review_fixture_bytes(&other_certificate, "calibration-17");
    let other_review = review_artifact_for_bytes(&other_review_bytes);
    let wrong_artifact = MockResolver {
        resolved: Ok(ResolvedCalibration::new(
            "calibration-17",
            instrument_identity(),
            other_certificate,
            other_review,
            Quantity::Frequency,
            Unit::Hertz,
            4_000_000.0,
            6_000_000.0,
            SAMPLE_TIME_NS - 1,
            SAMPLE_TIME_NS + 2_000_000_000,
            900.0,
        )
        .unwrap()),
        raw_bytes: Ok(raw_fixture_bytes()),
        raw_reference_override: None,
        calibration_bytes_override: None,
        review_bytes_override: Some(other_review_bytes),
        reference_standard_bytes_override: None,
        review_trust_guard: Mutex::new(ReviewTrustStoreGuard::default()),
    };
    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &policy(),
                &wrong_artifact,
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::CalibrationReferenceMismatch
    );
}

#[test]
fn rejects_wrong_calibration_unit_or_out_of_validity_time() {
    let (wrong_resolved, wrong_certificate, wrong_review) = calibration_fixture_for(
        "calibration-17",
        instrument_identity(),
        Quantity::AcousticPressure,
        Unit::Pascal,
        0.0,
        10_000_000.0,
        SAMPLE_TIME_NS - 1,
        SAMPLE_TIME_NS + 2_000_000_000,
        900.0,
    );
    let wrong_reference =
        CalibrationReference::new("calibration-17", wrong_resolved.evidence().clone()).unwrap();
    let wrong_unit = MockResolver {
        resolved: Ok(wrong_resolved),
        raw_bytes: Ok(raw_fixture_bytes()),
        raw_reference_override: None,
        calibration_bytes_override: Some(wrong_certificate),
        review_bytes_override: Some(wrong_review),
        reference_standard_bytes_override: None,
        review_trust_guard: Mutex::new(ReviewTrustStoreGuard::default()),
    };
    assert_eq!(
        measurement_with_calibration(1, SAMPLE_TIME_NS, wrong_reference)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &policy(),
                &wrong_unit,
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::CalibrationUnitMismatch
    );

    let (outdated_resolved, outdated_certificate, outdated_review) = calibration_fixture_for(
        "calibration-17",
        instrument_identity(),
        Quantity::Frequency,
        Unit::Hertz,
        4_000_000.0,
        6_000_000.0,
        SAMPLE_TIME_NS + 1,
        SAMPLE_TIME_NS + 2_000_000_000,
        900.0,
    );
    let outdated_reference =
        CalibrationReference::new("calibration-17", outdated_resolved.evidence().clone()).unwrap();
    let out_of_date = MockResolver {
        resolved: Ok(outdated_resolved),
        raw_bytes: Ok(raw_fixture_bytes()),
        raw_reference_override: None,
        calibration_bytes_override: Some(outdated_certificate),
        review_bytes_override: Some(outdated_review),
        reference_standard_bytes_override: None,
        review_trust_guard: Mutex::new(ReviewTrustStoreGuard::default()),
    };
    assert_eq!(
        measurement_with_calibration(1, SAMPLE_TIME_NS, outdated_reference)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &policy(),
                &out_of_date,
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::CalibrationNotValidAtCapture
    );
}
#[test]
fn uncertainty_policy_must_use_the_exact_measurement_unit() {
    let megahertz_policy =
        MeasurementPolicy::new(500_000_000, Unit::Megahertz, 0.002, 0.001).unwrap();
    let mut guard = MeasurementStreamGuard::default();

    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &megahertz_policy,
                &resolver(),
                &mut guard,
            )
            .unwrap_err(),
        AssessmentFailure::PolicyUnitMismatch {
            policy_unit: Unit::Megahertz,
            measurement_unit: Unit::Hertz,
        }
    );

    // Rejecting a policy/measurement unit mismatch does not consume a sensor
    // sequence; with the correctly typed policy, the same sample is eligible.
    measurement(1, SAMPLE_TIME_NS)
        .assess_for_quantitative_use(
            SAMPLE_TIME_NS,
            &clock_domain(),
            &policy(),
            &resolver(),
            &mut guard,
        )
        .unwrap();
}

#[test]
fn measurement_policy_rejects_invalid_uncertainty_limits() {
    assert_eq!(
        MeasurementPolicy::new(1_000, Unit::Hertz, f64::NAN, 1.0).unwrap_err(),
        ContractError::NonFiniteValue("max_measurement_standard_uncertainty")
    );
    assert_eq!(
        MeasurementPolicy::new(1_000, Unit::Hertz, 1.0, -0.1).unwrap_err(),
        ContractError::NegativeUncertainty("max_calibration_standard_uncertainty")
    );
}

#[test]
fn excessive_measurement_or_calibration_uncertainty_fails_closed() {
    let tight_measurement_policy =
        MeasurementPolicy::new(1_000, Unit::Hertz, 999.0, 1_000.0).unwrap();
    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &tight_measurement_policy,
                &resolver(),
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::MeasurementUncertaintyExceeded
    );

    let tight_calibration_policy =
        MeasurementPolicy::new(1_000, Unit::Hertz, 2_000.0, 899.0).unwrap();
    assert_eq!(
        measurement(1, SAMPLE_TIME_NS)
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS,
                &clock_domain(),
                &tight_calibration_policy,
                &resolver(),
                &mut MeasurementStreamGuard::default(),
            )
            .unwrap_err(),
        AssessmentFailure::CalibrationUncertaintyExceeded
    );
}

#[test]
fn stream_guard_consumes_new_sequence_even_when_timestamp_regresses() {
    let first = measurement(7, SAMPLE_TIME_NS);
    let duplicate = measurement(7, SAMPLE_TIME_NS + 1);
    let backward_time = measurement(8, SAMPLE_TIME_NS - 1);
    let retry_same_sequence = measurement(8, SAMPLE_TIME_NS + 1);
    let next = measurement(9, SAMPLE_TIME_NS + 1);

    let mut guard = MeasurementStreamGuard::default();
    guard.observe(&first).unwrap();
    assert_eq!(
        guard.observe(&duplicate).unwrap_err(),
        StreamOrderFailure::SequenceNotIncreasing { previous: 7, received: 7 }
    );
    assert_eq!(
        guard.observe(&backward_time).unwrap_err(),
        StreamOrderFailure::TimestampMovedBackward {
            previous_ns: SAMPLE_TIME_NS,
            received_ns: SAMPLE_TIME_NS - 1
        }
    );
    assert_eq!(
        guard.observe(&retry_same_sequence).unwrap_err(),
        StreamOrderFailure::SequenceNotIncreasing { previous: 8, received: 8 }
    );
    guard.observe(&next).unwrap();
}

#[test]
fn quantitative_gate_itself_rejects_a_replayed_measurement() {
    let m = measurement(9, SAMPLE_TIME_NS);
    let mut guard = MeasurementStreamGuard::default();
    m.assess_for_quantitative_use(
        SAMPLE_TIME_NS,
        &clock_domain(),
        &policy(),
        &resolver(),
        &mut guard,
    )
    .unwrap();
    assert_eq!(
        m.assess_for_quantitative_use(
            SAMPLE_TIME_NS,
            &clock_domain(),
            &policy(),
            &resolver(),
            &mut guard,
        )
        .unwrap_err(),
        AssessmentFailure::StreamOrder(StreamOrderFailure::SequenceNotIncreasing {
            previous: 9,
            received: 9,
        })
    );
}

#[test]
fn clock_domain_mismatch_fails_before_consuming_the_measurement_sequence() {
    let m = measurement(1, SAMPLE_TIME_NS);
    let wrong_domain = ClockDomainId::new("other-rig-boot-epoch").unwrap();
    let mut guard = MeasurementStreamGuard::default();

    assert_eq!(
        m.assess_for_quantitative_use(
            SAMPLE_TIME_NS,
            &wrong_domain,
            &policy(),
            &resolver(),
            &mut guard,
        )
        .unwrap_err(),
        AssessmentFailure::ClockDomainMismatch
    );

    // A caller using the measurement's actual clock domain may retry because the
    // prior failure was a clock-context mismatch, not a consumed sensor sample.
    m.assess_for_quantitative_use(
        SAMPLE_TIME_NS,
        &clock_domain(),
        &policy(),
        &resolver(),
        &mut guard,
    )
    .unwrap();
}

#[test]
fn clock_epoch_change_is_rejected_by_an_existing_stream_guard() {
    let first = measurement(1, SAMPLE_TIME_NS);
    let next_epoch_id = ClockDomainId::new("ultrasound-rig-boot-epoch-02").unwrap();
    let second_epoch = measurement_in_clock_domain(
        2,
        SAMPLE_TIME_NS + 1,
        next_epoch_id.clone(),
    );
    let mut guard = MeasurementStreamGuard::default();

    first
        .assess_for_quantitative_use(
            SAMPLE_TIME_NS,
            &clock_domain(),
            &policy(),
            &resolver(),
            &mut guard,
        )
        .unwrap();

    assert_eq!(
        second_epoch
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS + 2,
                &next_epoch_id,
                &policy(),
                &resolver(),
                &mut guard,
            )
            .unwrap_err(),
        AssessmentFailure::StreamOrder(StreamOrderFailure::ClockDomainChanged {
            previous: clock_domain(),
            received: next_epoch_id,
        })
    );
}

#[test]
fn future_timestamp_consumes_sequence_without_poisoning_timestamp_high_water_mark() {
    let first = measurement(7, SAMPLE_TIME_NS);
    let future = measurement(8, SAMPLE_TIME_NS + 2_000_000_000);
    let next_valid = measurement(9, SAMPLE_TIME_NS + 10);
    let retry_future_sequence = measurement(8, SAMPLE_TIME_NS + 20);

    let mut guard = MeasurementStreamGuard::default();
    first
        .assess_for_quantitative_use(
            SAMPLE_TIME_NS,
            &clock_domain(),
            &policy(),
            &resolver(),
            &mut guard,
        )
        .unwrap();

    assert_eq!(
        future
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS + 1_000_000_000,
                &clock_domain(),
                &policy(),
                &resolver(),
                &mut guard,
            )
            .unwrap_err(),
        AssessmentFailure::ClockInFuture
    );

    // A later sequence with a timestamp beyond the last accepted sample remains
    // usable: the future-dated timestamp did not poison the timestamp high-water mark.
    next_valid
        .assess_for_quantitative_use(
            SAMPLE_TIME_NS + 20,
            &clock_domain(),
            &policy(),
            &resolver(),
            &mut guard,
        )
        .unwrap();

    // The rejected future sample's sequence was still consumed and cannot be replayed.
    assert_eq!(
        retry_future_sequence
            .assess_for_quantitative_use(
                SAMPLE_TIME_NS + 30,
                &clock_domain(),
                &policy(),
                &resolver(),
                &mut guard,
            )
            .unwrap_err(),
        AssessmentFailure::StreamOrder(StreamOrderFailure::SequenceNotIncreasing {
            previous: 9,
            received: 8,
        })
    );
}

#[test]
fn units_map_to_the_declared_quantity_without_implicit_conversion() {
    assert_eq!(Unit::Megahertz.quantity(), Quantity::Frequency);
    assert_eq!(Unit::Microvolt.quantity(), Quantity::ElectricalPotential);
    assert_eq!(Unit::Percent.quantity(), Quantity::Dimensionless);
    assert_eq!(
        Unit::OxygenSaturationPercent.quantity(),
        Quantity::OxygenSaturation
    );
    assert_eq!(Unit::DecibelRe20Micropascal.quantity(), Quantity::SoundPressureLevel);
    // Same physical quantity, different units remain explicit; this crate does not convert.
    assert_ne!(Unit::Hertz, Unit::Megahertz);
}
