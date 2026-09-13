// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::cell::Cell;
use symthaea_trust_kernel::{
    ClockBootstrapAuthorityError, ClockBootstrapAuthorityEvidenceV2,
    ClockBootstrapAuthorityVerifier, ClockBootstrapClaimV2, Sha256Digest,
    verify_clock_bootstrap_authority,
};

const TRUST: &str = "609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633";
const EVALUATION_POLICY: &str = "7c8d8a8f849999d8cb4b2dddc685d37b18c901801f6ced2c26ce4d7adf09c318";
const CLAIM: &str = "15d99bb8fd9111972c089882717481917aae06d911f759c512712bcf157e9877";
const EVIDENCE: &str = "62991edeba3a575b28801bcceaca99dceaec476a0bab699792df6fbe2ec0cee7";
const VERIFIED: &str = "b8f80e4e5368e2fa1a0230b85db984997b52d465e47a76d91302f49409a9bcc3";

fn digest(hex: &str) -> Sha256Digest { Sha256Digest::from_hex(hex).unwrap() }

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
    fn provider_id(&self) -> &str { self.provider }
    fn authority_policy_digest(&self) -> Sha256Digest { self.policy }
    fn verify_clock_bootstrap_authority(
        &self,
        canonical_claim_bytes: &[u8],
        external_evidence_digest: Sha256Digest,
    ) -> Result<bool, String> {
        self.calls.set(self.calls.get() + 1);
        assert_eq!(external_evidence_digest, digest(&"c".repeat(64)));
        assert_eq!(
            std::str::from_utf8(canonical_claim_bytes).unwrap(),
            concat!(
                "{\"clock_evaluation_policy_id\":\"", EVALUATION_POLICY, "\",",
                "\"purpose\":\"ClockBootstrap\",",
                "\"schema\":\"symthaea.trust.clock-bootstrap-claim.v2\",",
                "\"trust_snapshot_digest\":\"", TRUST, "\",",
                "\"trusted_lower_unix_ms\":1499000,",
                "\"trusted_upper_unix_ms\":1499500}"
            )
        );
        self.decision.map_err(str::to_string)
    }
}

fn fixture() -> (ClockBootstrapClaimV2, ClockBootstrapAuthorityEvidenceV2) {
    let claim = ClockBootstrapClaimV2::new(
        digest(TRUST),
        digest(EVALUATION_POLICY),
        1_499_000,
        1_499_500,
    ).unwrap();
    let evidence = ClockBootstrapAuthorityEvidenceV2::new(
        &claim,
        "platform-root-01",
        digest(&"b".repeat(64)),
        digest(&"c".repeat(64)),
    ).unwrap();
    (claim, evidence)
}

#[test]
fn production_v2_binding_matches_independent_reference_vectors() {
    let (claim, evidence) = fixture();
    assert_eq!(claim.id().to_hex(), CLAIM);
    assert_eq!(claim.clock_evaluation_policy_id().to_hex(), EVALUATION_POLICY);
    assert_eq!(evidence.id().to_hex(), EVIDENCE);

    let verifier = StubVerifier::accepting();
    let verified = verify_clock_bootstrap_authority(&claim, &evidence, &verifier).unwrap();
    assert_eq!(verified.id().to_hex(), VERIFIED);
    assert_eq!(verified.claim_id(), claim.id());
    assert_eq!(verified.authority_evidence_id(), evidence.id());
    assert_eq!(verified.provider_id(), "platform-root-01");
    assert_eq!(verified.authority_policy_digest(), digest(&"b".repeat(64)));
    assert_eq!(verified.clock_evaluation_policy_id(), digest(EVALUATION_POLICY));
    assert_eq!(verifier.calls.get(), 1);
}

#[test]
fn binding_mismatches_fail_before_external_adapter_is_called() {
    let (claim, evidence) = fixture();

    let wrong_provider = StubVerifier { provider: "other-root", ..StubVerifier::accepting() };
    assert_eq!(
        verify_clock_bootstrap_authority(&claim, &evidence, &wrong_provider).unwrap_err(),
        ClockBootstrapAuthorityError::ProviderMismatch
    );
    assert_eq!(wrong_provider.calls.get(), 0);

    let wrong_policy = StubVerifier { policy: digest(&"e".repeat(64)), ..StubVerifier::accepting() };
    assert_eq!(
        verify_clock_bootstrap_authority(&claim, &evidence, &wrong_policy).unwrap_err(),
        ClockBootstrapAuthorityError::AuthorityPolicyMismatch
    );
    assert_eq!(wrong_policy.calls.get(), 0);

    let other_claim = ClockBootstrapClaimV2::new(
        digest(TRUST), digest(EVALUATION_POLICY), 1_499_100, 1_499_500
    ).unwrap();
    assert_eq!(
        verify_clock_bootstrap_authority(&other_claim, &evidence, &StubVerifier::accepting()).unwrap_err(),
        ClockBootstrapAuthorityError::EvidenceClaimMismatch
    );

    let other_eval_policy_claim = ClockBootstrapClaimV2::new(
        digest(TRUST), digest(&"e".repeat(64)), 1_499_000, 1_499_500
    ).unwrap();
    assert_eq!(
        verify_clock_bootstrap_authority(&other_eval_policy_claim, &evidence, &StubVerifier::accepting()).unwrap_err(),
        ClockBootstrapAuthorityError::EvidenceClaimMismatch
    );
}

#[test]
fn explicit_external_rejection_does_not_mint_capability() {
    let (claim, evidence) = fixture();
    let verifier = StubVerifier { decision: Ok(false), ..StubVerifier::accepting() };
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
    claim_json["clock_evaluation_policy_id"] = serde_json::to_value(digest(&"e".repeat(64))).unwrap();
    let tampered_claim: ClockBootstrapClaimV2 = serde_json::from_value(claim_json).unwrap();
    assert_eq!(tampered_claim.validate().unwrap_err(), ClockBootstrapAuthorityError::ClaimIdentityMismatch);

    let mut evidence_json = serde_json::to_value(&evidence).unwrap();
    evidence_json["external_evidence_digest"] = serde_json::to_value(digest(&"d".repeat(64))).unwrap();
    let tampered_evidence: ClockBootstrapAuthorityEvidenceV2 = serde_json::from_value(evidence_json).unwrap();
    assert_eq!(tampered_evidence.validate().unwrap_err(), ClockBootstrapAuthorityError::EvidenceIdentityMismatch);
}

#[test]
fn production_source_has_no_v1_or_permissive_or_circular_bootstrap_path() {
    let source = include_str!("../src/clock_bootstrap_authority.rs");
    assert!(!source.contains("ClockBootstrapClaimV1"));
    assert!(!source.contains("VerifiedClockBootstrapAuthorityV1"));
    assert!(!source.contains("struct AcceptAll"));
    assert!(!source.contains("impl ClockBootstrapAuthorityVerifier for"));
    assert!(!source.contains("pub fn unchecked"));
    assert!(!source.contains("pub fn from_verified"));
    assert!(!source.contains("VerifiedClockWindow"));
    assert!(!source.contains("now_unix_s"));
    assert!(!source.contains("evaluation_time_unix"));
}
