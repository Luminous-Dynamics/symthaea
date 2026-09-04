// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use symthaea_authority::{Digest32, ResourceRef};
use symthaea_iot_device_protocol::DeviceSemanticHead;
use symthaea_iot_final_gate::{
    FinalActuatorGateError, HardwareInterlockEvidenceVerifier,
    PHYSICAL_INTERLOCK_POLICY_SCHEMA_VERSION, PHYSICAL_INTERLOCK_REPORT_SCHEMA_VERSION,
    PhysicalInterlockPolicyV1, PhysicalInterlockReportV1, verify_physical_interlock,
};
use symthaea_iot_transport_receipt::TransportTrustHead;

const INTERLOCK_ED25519_MESSAGE_DOMAIN: &[u8] = b"symthaea-iot-interlock-ed25519-v1\0";

fn d(byte: u8) -> Digest32 {
    Digest32([byte; 32])
}

fn policy() -> PhysicalInterlockPolicyV1 {
    PhysicalInterlockPolicyV1 {
        schema_version: PHYSICAL_INTERLOCK_POLICY_SCHEMA_VERSION,
        device: ResourceRef("iot:valve:72".into()),
        allowed_controllers: BTreeSet::from(["safety-plc:field-a".into()]),
        required_interlocks: BTreeSet::from(["pressure-safe".into()]),
        max_report_lifetime_ms: 1_000,
    }
}

fn unsigned_report() -> PhysicalInterlockReportV1 {
    PhysicalInterlockReportV1 {
        schema_version: PHYSICAL_INTERLOCK_REPORT_SCHEMA_VERSION,
        controller_id: "safety-plc:field-a".into(),
        device: ResourceRef("iot:valve:72".into()),
        envelope_digest: d(0x31),
        device_head: DeviceSemanticHead {
            generation: 9,
            digest: d(0x32),
        },
        transport_trust_head: TransportTrustHead {
            sequence: 7,
            digest: d(0x33),
        },
        asserted_interlocks: BTreeSet::from(["pressure-safe".into()]),
        checked_at_unix_ms: 100_000,
        expires_at_unix_ms: 100_800,
        // A nonzero placeholder is sufficient for structural validation. The
        // controller-signing digest must not depend on this field.
        evidence_digest: d(0xEE),
    }
}

fn signing_message(report_digest: Digest32) -> Vec<u8> {
    let Digest32(digest) = report_digest;
    let mut message = Vec::with_capacity(INTERLOCK_ED25519_MESSAGE_DOMAIN.len() + digest.len());
    message.extend_from_slice(INTERLOCK_ED25519_MESSAGE_DOMAIN);
    message.extend_from_slice(&digest);
    message
}

struct RealEd25519Provider {
    public_key: [u8; 32],
}

impl HardwareInterlockEvidenceVerifier for RealEd25519Provider {
    fn verify_interlock_evidence(
        &self,
        controller_id: &str,
        report_digest: Digest32,
        raw_evidence: &[u8],
    ) -> bool {
        if controller_id != "safety-plc:field-a" || raw_evidence.len() != 64 {
            return false;
        }
        let Ok(key) = VerifyingKey::from_bytes(&self.public_key) else {
            return false;
        };
        let Ok(signature) = Signature::try_from(raw_evidence) else {
            return false;
        };
        key.verify(&signing_message(report_digest), &signature).is_ok()
    }
}

fn signed_report() -> (PhysicalInterlockReportV1, [u8; 64], RealEd25519Provider, Digest32) {
    let signing_key = SigningKey::from_bytes(&[0x31; 32]);
    let mut report = unsigned_report();

    let report_digest = report.digest().expect("constructible report-content digest");
    let signature = signing_key.sign(&signing_message(report_digest)).to_bytes();

    // Completing the report with H(signature) must not change the bytes the
    // controller had to sign. This is the regression for the former circular
    // signature = Sign(report_digest(H(signature))) contract.
    report.evidence_digest = Digest32(*blake3::hash(&signature).as_bytes());
    assert_eq!(
        report.digest().expect("completed report-content digest"),
        report_digest
    );

    let provider = RealEd25519Provider {
        public_key: signing_key.verifying_key().to_bytes(),
    };
    (report, signature, provider, report_digest)
}

#[test]
fn controller_can_construct_signature_then_complete_evidence_commitment() {
    let (report, signature, provider, report_digest) = signed_report();
    let evidence_digest = report.evidence_digest;

    let verified = verify_physical_interlock(
        &policy(),
        report,
        &signature,
        100_100,
        &provider,
    )
    .expect("real Ed25519 report/evidence pair should verify");

    assert_eq!(verified.report_digest(), report_digest);
    assert_eq!(verified.evidence_digest(), evidence_digest);
}

#[test]
fn signed_report_field_mutation_fails_even_when_signature_bytes_are_unchanged() {
    let (mut report, signature, provider, _) = signed_report();
    report.envelope_digest = d(0x91);

    assert!(matches!(
        verify_physical_interlock(&policy(), report, &signature, 100_100, &provider),
        Err(FinalActuatorGateError::InterlockEvidenceVerificationFailed)
    ));
}

#[test]
fn evidence_mutation_fails_before_signature_provider_can_accept_it() {
    let (report, mut signature, provider, _) = signed_report();
    signature[0] ^= 1;

    assert!(matches!(
        verify_physical_interlock(&policy(), report, &signature, 100_100, &provider),
        Err(FinalActuatorGateError::InterlockEvidenceDigestMismatch)
    ));
}

#[test]
fn evidence_commitment_is_independent_from_controller_signing_digest() {
    let report = unsigned_report();
    let digest = report.digest().unwrap();
    let mut altered = report;
    altered.evidence_digest = d(0xEF);

    assert_eq!(altered.digest().unwrap(), digest);
}
