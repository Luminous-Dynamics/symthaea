// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::fs;
use std::path::{Path, PathBuf};

use symthaea_authority::Digest32;
use symthaea_platform_attestation::{
    LocalTpm2QuoteInputs, PLATFORM_ATTESTATION_SCHEMA_VERSION,
    PendingPlatformAttestationChallenge, PlatformAttestationError, PlatformAttestationPolicyV1,
    verify_local_tpm2_attestation_v1,
};

fn temp_file(label: &str, bytes: &[u8]) -> PathBuf {
    let mut nonce = [0u8; 8];
    getrandom::getrandom(&mut nonce).unwrap();
    let suffix = nonce.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let path = std::env::temp_dir().join(format!("symthaea-tpm2-{label}-{suffix}"));
    fs::write(&path, bytes).unwrap();
    path
}

fn digest_file(path: &Path) -> Digest32 {
    Digest32(*blake3::hash(&fs::read(path).unwrap()).as_bytes())
}

fn policy(quote: &Path, check: &Path, ak: &Path) -> PlatformAttestationPolicyV1 {
    PlatformAttestationPolicyV1 {
        schema_version: PLATFORM_ATTESTATION_SCHEMA_VERSION,
        policy_id: [1; 16],
        tpm2_quote_path: quote.to_string_lossy().into_owned(),
        tpm2_quote_digest: digest_file(quote),
        tpm2_checkquote_path: check.to_string_lossy().into_owned(),
        tpm2_checkquote_digest: digest_file(check),
        trusted_ak_public_digest: digest_file(ak),
        sha256_pcr_selection: vec![16],
        approved_pcr_profile_digests: vec![Digest32([9; 32])],
        require_nix_store_tools: false,
        maximum_challenge_age_ns: 5_000_000_000,
        maximum_post_verification_age_ns: 5_000_000_000,
    }
}

fn challenge(policy: &PlatformAttestationPolicyV1) -> PendingPlatformAttestationChallenge {
    PendingPlatformAttestationChallenge::new(
        policy,
        Digest32([2; 32]),
        Digest32([3; 32]),
        Digest32([4; 32]),
    )
    .unwrap()
}

#[test]
fn replaced_quote_tool_is_rejected_before_execution() {
    let quote = temp_file("quote-tool", b"reviewed quote binary");
    let check = temp_file("check-tool", b"reviewed check binary");
    let ak = temp_file("ak", b"reviewed ak public");
    let p = policy(&quote, &check, &ak);
    let pending = challenge(&p);

    fs::write(&quote, b"replaced quote binary").unwrap();
    let error = verify_local_tpm2_attestation_v1(
        &p,
        pending,
        &LocalTpm2QuoteInputs {
            ak_context_path: PathBuf::from("never-used.ctx"),
            ak_public_path: ak.clone(),
        },
    )
    .unwrap_err();
    assert!(matches!(error, PlatformAttestationError::ToolDigestMismatch));

    let _ = fs::remove_file(quote);
    let _ = fs::remove_file(check);
    let _ = fs::remove_file(ak);
}

#[test]
fn replaced_ak_public_is_rejected_before_quote_execution() {
    let quote = temp_file("quote-tool", b"reviewed quote binary");
    let check = temp_file("check-tool", b"reviewed check binary");
    let ak = temp_file("ak", b"reviewed ak public");
    let p = policy(&quote, &check, &ak);
    let pending = challenge(&p);

    fs::write(&ak, b"different ak public").unwrap();
    let error = verify_local_tpm2_attestation_v1(
        &p,
        pending,
        &LocalTpm2QuoteInputs {
            ak_context_path: PathBuf::from("never-used.ctx"),
            ak_public_path: ak.clone(),
        },
    )
    .unwrap_err();
    assert!(matches!(error, PlatformAttestationError::AkPublicKeyMismatch));

    let _ = fs::remove_file(quote);
    let _ = fs::remove_file(check);
    let _ = fs::remove_file(ak);
}

#[test]
fn production_nix_tool_requirement_rejects_temp_tools() {
    let quote = temp_file("quote-tool", b"reviewed quote binary");
    let check = temp_file("check-tool", b"reviewed check binary");
    let ak = temp_file("ak", b"reviewed ak public");
    let mut p = policy(&quote, &check, &ak);
    p.require_nix_store_tools = true;
    let pending = challenge(&p);

    let error = verify_local_tpm2_attestation_v1(
        &p,
        pending,
        &LocalTpm2QuoteInputs {
            ak_context_path: PathBuf::from("never-used.ctx"),
            ak_public_path: ak.clone(),
        },
    )
    .unwrap_err();
    assert!(matches!(error, PlatformAttestationError::ToolOutsideNixStore));

    let _ = fs::remove_file(quote);
    let _ = fs::remove_file(check);
    let _ = fs::remove_file(ak);
}
