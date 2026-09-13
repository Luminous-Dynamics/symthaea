// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::cell::Cell;
use symthaea_trust_kernel::{
    ClockBootstrapAuthorityError, ClockBootstrapAuthorityEvidenceV1,
    ClockBootstrapAuthorityVerifier, ClockBootstrapClaimV1, Sha256Digest,
    verify_clock_bootstrap_authority,
};

const TRUST: &str = "609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633";
const CLAIM: &str = "5bebb59ba5b95bd487977ffd296bf398e815a80eebca36813f80344e3b68439d";
const EVIDENCE: &str = "1bf502c3fe07d5dc7151b35eb93e6c12e49bb1f465ae55b74a60ca3aabc09d7d";
const VERIFIED: &str = "0298a2d298c7e1f1cfebee21fb91aac61c08f749f11de368910f4d3118e48760";

fn digest(hex: &str) -> Sha256Digest {
    Sha256Digest::from_hex(hex).unwrap()
}

struct StubVerifier {
    provider: &'static str,
    policy: Sha256Digest,
    decision: Result<bool, &'static str>,
    calls: Cell<usize>,
}

impl StubVerifier {
    fn accepting() -> Self {
        Self {
            provider: "platform-root-01",
            policy: digest(&"b".repeat(64)),
            decision: Ok(true),
            calls: Cell::new(0),
        }
    }
}

impl ClockBootstrapAuthorityVerifier for StubVerifier {
    fn provider_id(&self) -> &str {
        self.provider
    }

    fn authority_policy_digest(&self) -> Sha256Digest {
        self.policy
    }

    fn verify_clock_bootstrap_authority(
        &self,
        _canonical_claim_bytes: &[u8],
        external_evidence_digest: Sha256Digest,
    ) -> Result<bool, String> {
        self.calls.set(self.calls.get() + 1);
        assert_eq!(external_evidence_digest, digest(&"c".repeat(64)));
        assert_eq!(
            std::str::from_utf8(_canonical_claim_bytes).unwrap(),
            concat!(
                "{\"purpose\":\"ClockBootstrap\",",
                "\"schema\":\"symthaea.trust.clock-bootstrap-claim.v1\",",
                "\"trust_snapshot_digest\":\"", TRUST, "\",",
                "\"trusted_lower_unix_ms\":1499000,",
                "\"trusted_upper_unix_ms\":1499500}"
            )
        );
        self.decision.map_err(str::to_string)
    }
}

fn fixture() -> (ClockBootstrapClaimV1, ClockBootstrapAuthorityEvidenceV1) {
    let claim = ClockBootstrapClaimV1::new(digest(TRUST), 1_499_000, 1_499_500).unwrap();
    let evidence = ClockBootstrapAuthorityEvidenceV1::new(
        &claim,
        "platform-root-01",
        digest(&"b".repeat(64)),
        digest(&"c".repeat(64)),
    )
    .unwrap();
    (claim, evidence)
}

#[test]
fn production_binding_matches_independent_reference_vectors() {
    let (claim, evidence) = fixture();
    assert_eq!(claim.id().to_hex(), CLAIM);
    assert_eq!(evidence.id().to_hex(), EVIDENCE);

    let verifier = StubVerifier::accepting();
    let verified = verify_clock_bootstrap_authority(&claim, &evidence, &verifier).unwrap();
    assert_eq!(verified.id().to_hex(), VERIFIED);
    assert_eq!(verified.claim_id(), claim.id());
    assert_eq!(verified.authority_evidence_id(), evidence.id());
    assert_eq!(verified.provider_id(), "platform-root-01");
    assert_eq!(verified.authority_policy_digest(), digest(&"b".repeat(64)));
    assert_eq!(verifier.calls.get(), 1);
}

#[test]
fn binding_mismatches_fail_before_external_adapter_is_called() {
    let (claim, evidence) = fixture();

    let wrong_provider = StubVerifier {
        provider: "other-root",
        ..StubVerifier::accepting()
    };
    assert_eq!(
        verify_clock_bootstrap_authority(&claim, &evidence, &wrong_provider).unwrap_err(),
        ClockBootstrapAuthorityError::ProviderMismatch
    );
    assert_eq!(wrong_provider.calls.get(), 0);

    let wrong_policy = StubVerifier {
        policy: digest(&"e".repeat(64)),
        ..StubVerifier::accepting()
    };
    assert_eq!(
        verify_clock_bootstrap_authority(&claim, &evidence, &wrong_policy).unwrap_err(),
        ClockBootstrapAuthorityError::AuthorityPolicyMismatch
    );
    assert_eq!(wrong_policy.calls.get(), 0);

    let other_claim = ClockBootstrapClaimV1::new(digest(TRUST), 1_499_100, 1_499_500).unwrap();
    assert_eq!(
        verify_clock_bootstrap_authority(&other_claim, &evidence, &StubVerifier::accepting())
            .unwrap_err(),
        ClockBootstrapAuthorityError::EvidenceClaimMismatch
    );
}

#[test]
fn explicit_external_rejection_does_not_mint_capability() {
    let (claim, evidence) = fixture();
    let verifier = StubVerifier {
        decision: Ok(false),
        ..StubVerifier::accepting()
    };
    assert_eq!(
        verify_clock_bootstrap_authority(&claim, &evidence, &verifier).unwrap_err(),
        ClockBootstrapAuthorityError::ExternalAuthorityRejected
    );
    assert_eq!(verifier.calls.get(), 1);
}

#[test]
fn tampered_serializable_records_fail_self_validation() {
    let (claim, evidence) = fixture();

    let mut claim_json = serde_json::to_value(&claim).unwrap();
    claim_json["trusted_upper_unix_ms"] = serde_json::Value::from(1_499_501u64);
    let tampered_claim: ClockBootstrapClaimV1 = serde_json::from_value(claim_json).unwrap();
    assert_eq!(
        tampered_claim.validate().unwrap_err(),
        ClockBootstrapAuthorityError::ClaimIdentityMismatch
    );

    let mut evidence_json = serde_json::to_value(&evidence).unwrap();
    evidence_json["external_evidence_digest"] =
        serde_json::to_value(digest(&"d".repeat(64))).unwrap();
    let tampered_evidence: ClockBootstrapAuthorityEvidenceV1 =
        serde_json::from_value(evidence_json).unwrap();
    assert_eq!(
        tampered_evidence.validate().unwrap_err(),
        ClockBootstrapAuthorityError::EvidenceIdentityMismatch
    );
}

#[test]
fn production_source_has_no_builtin_permissive_adapter_or_unchecked_capability_constructor() {
    let source = include_str!("../src/clock_bootstrap_authority.rs");
    assert!(!source.contains("struct AcceptAll"));
    assert!(!source.contains("impl ClockBootstrapAuthorityVerifier for"));
    assert!(!source.contains("pub fn unchecked"));
    assert!(!source.contains("pub fn from_verified"));
    assert!(!source.contains("VerifiedClockWindow"));
    assert!(!source.contains("now_unix_s"));
    assert!(!source.contains("evaluation_time_unix"));
}
