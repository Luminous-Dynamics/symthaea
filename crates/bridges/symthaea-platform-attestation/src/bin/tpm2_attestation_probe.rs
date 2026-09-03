// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Qualification-only CLI for the TPM2 adapter.
//!
//! The `verify-nix` path intentionally exercises the production verifier-tool
//! policy: both reviewed entry points must resolve under `/nix/store`. There is
//! no host-tool verification mode in this probe.

use std::fs;
use std::path::{Path, PathBuf};

use symthaea_authority::Digest32;
use symthaea_platform_attestation::{
    LocalTpm2QuoteInputs, PLATFORM_ATTESTATION_SCHEMA_VERSION,
    PendingPlatformAttestationChallenge, PlatformAttestationPolicyV1,
    verify_local_tpm2_attestation_v1,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("tpm2_attestation_probe_error={error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    match args.as_slice() {
        [_, command, path] if command == "profile-digest" => {
            let digest = digest_file(Path::new(path))?;
            println!("{}", hex(&digest.0));
            Ok(())
        }
        [_, command, quote_tool, check_tool, ak_context, ak_public, profile, pcrs]
            if command == "verify-nix" =>
        {
            let approved = Digest32(parse_hex_32(profile)?);
            let mut selection = pcrs
                .split(',')
                .map(str::parse::<u8>)
                .collect::<Result<Vec<_>, _>>()?;
            selection.sort_unstable();
            selection.dedup();

            let policy = PlatformAttestationPolicyV1 {
                schema_version: PLATFORM_ATTESTATION_SCHEMA_VERSION,
                policy_id: [0x71; 16],
                tpm2_quote_path: canonical_string(quote_tool)?,
                tpm2_quote_digest: digest_file(Path::new(quote_tool))?,
                tpm2_checkquote_path: canonical_string(check_tool)?,
                tpm2_checkquote_digest: digest_file(Path::new(check_tool))?,
                trusted_ak_public_digest: digest_file(Path::new(ak_public))?,
                sha256_pcr_selection: selection,
                approved_pcr_profile_digests: vec![approved],
                require_nix_store_tools: true,
                maximum_challenge_age_ns: 5_000_000_000,
                maximum_post_verification_age_ns: 5_000_000_000,
            };
            let challenge = PendingPlatformAttestationChallenge::new(
                &policy,
                Digest32([0x11; 32]),
                Digest32([0x22; 32]),
                Digest32([0x33; 32]),
            )?;
            let verified = verify_local_tpm2_attestation_v1(
                &policy,
                challenge,
                &LocalTpm2QuoteInputs {
                    ak_context_path: PathBuf::from(ak_context),
                    ak_public_path: PathBuf::from(ak_public),
                },
            )?;
            verified.ensure_fresh(
                &policy,
                Digest32([0x11; 32]),
                Digest32([0x22; 32]),
                Digest32([0x33; 32]),
            )?;
            println!("platform_attestation=verified");
            println!("policy_digest={}", hex(&verified.policy_digest().0));
            println!(
                "pcr_profile_digest={}",
                hex(&verified.pcr_profile_digest().0)
            );
            println!("ak_public_digest={}", hex(&verified.ak_public_digest().0));
            println!("challenge_digest={}", hex(&verified.challenge_digest().0));
            Ok(())
        }
        _ => Err("usage: tpm2_attestation_probe profile-digest <pcr-file> | verify-nix <nix-tpm2-quote-wrapper> <nix-tpm2-checkquote-wrapper> <ak.ctx> <ak-public> <approved-pcr-digest-hex> <comma-separated-pcrs>".into()),
    }
}

fn digest_file(path: &Path) -> Result<Digest32, Box<dyn std::error::Error>> {
    Ok(Digest32(*blake3::hash(&fs::read(path)?).as_bytes()))
}

fn canonical_string(path: &str) -> Result<String, Box<dyn std::error::Error>> {
    Ok(fs::canonicalize(path)?.to_string_lossy().into_owned())
}

fn parse_hex_32(value: &str) -> Result<[u8; 32], Box<dyn std::error::Error>> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("expected exactly 64 hexadecimal characters".into());
    }
    let mut out = [0u8; 32];
    for (index, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)?;
    }
    Ok(out)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
