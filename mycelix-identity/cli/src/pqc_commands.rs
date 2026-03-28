// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! PQC key generation, signing, and verification commands.
//!
//! Uses `mycelix-crypto` with the `native` feature for real PQC operations
//! (Ed25519, ML-DSA-65/87, SPHINCS+, Hybrid Ed25519+ML-DSA-65).

use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use chrono::Utc;
use mycelix_crypto::algorithm::AlgorithmId;
use mycelix_crypto::envelope::{TaggedPublicKey, TaggedSignature};
use mycelix_crypto::pqc::dilithium::{MlDsa65Signer, MlDsa65Verifier, MlDsa87Signer, MlDsa87Verifier};
use mycelix_crypto::pqc::ed25519_native::{Ed25519Signer, Ed25519Verifier};
use mycelix_crypto::pqc::hybrid::{HybridSigner, HybridVerifier};
use mycelix_crypto::pqc::sphincs::{SlhDsaSha2128sSigner, SlhDsaSha2128sVerifier};
use mycelix_crypto::traits::{Signer, Verifier};
use serde::{Deserialize, Serialize};
use std::path::Path;
use zeroize::{Zeroize, Zeroizing};

/// Serialized key file format.
#[derive(Debug, Serialize, Deserialize)]
pub struct KeyFile {
    pub algorithm: String,
    pub algorithm_id: u16,
    pub public_key: String,
    pub public_key_bytes: String,
    pub secret_key_bytes: String,
    pub created_at: String,
}

/// Signed credential output format.
#[derive(Debug, Serialize, Deserialize)]
pub struct SignedCredential {
    pub credential: serde_json::Value,
    pub proof: CredentialProof,
}

/// Credential proof structure.
#[derive(Debug, Serialize, Deserialize)]
pub struct CredentialProof {
    #[serde(rename = "type")]
    pub proof_type: String,
    pub cryptosuite: String,
    pub created: String,
    #[serde(rename = "proofPurpose")]
    pub proof_purpose: String,
    #[serde(rename = "proofValue")]
    pub proof_value: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub challenge: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
}

/// Parse algorithm name to AlgorithmId.
fn parse_algorithm(name: &str) -> Result<AlgorithmId> {
    match name.to_lowercase().replace('-', "_").as_str() {
        "ed25519" => Ok(AlgorithmId::Ed25519),
        "ml_dsa_65" | "mldsa65" | "ml_dsa65" => Ok(AlgorithmId::MlDsa65),
        "ml_dsa_87" | "mldsa87" | "ml_dsa87" => Ok(AlgorithmId::MlDsa87),
        "slh_dsa_sha2_128s" | "slhdsa_sha2_128s" | "sphincs" => Ok(AlgorithmId::SlhDsaSha2_128s),
        "hybrid_ed25519_ml_dsa_65" | "hybrid" | "hybrid_ed25519_mldsa65" => {
            Ok(AlgorithmId::HybridEd25519MlDsa65)
        }
        _ => bail!(
            "Unknown algorithm '{}'. Supported: ed25519, ml-dsa-65, ml-dsa-87, slh-dsa-sha2-128s, hybrid-ed25519-ml-dsa-65",
            name
        ),
    }
}

/// Generate a PQC keypair and write to a JSON file.
pub fn keygen(algorithm: &str, output: &Path) -> Result<()> {
    let alg = parse_algorithm(algorithm)?;

    let (pub_key, pub_key_bytes, secret_bytes): (TaggedPublicKey, Vec<u8>, Zeroizing<Vec<u8>>) = match alg {
        AlgorithmId::Ed25519 => {
            let s = Ed25519Signer::generate();
            let pk = s.public_key();
            let pk_bytes = pk.key_bytes.clone();
            let sk = s.secret_key_bytes();
            (pk, pk_bytes, sk)
        }
        AlgorithmId::MlDsa65 => {
            let s = MlDsa65Signer::generate();
            let pk = s.public_key();
            let pk_bytes = s.public_key_bytes();
            let sk = s.secret_key_bytes();
            (pk, pk_bytes, sk)
        }
        AlgorithmId::MlDsa87 => {
            let s = MlDsa87Signer::generate();
            let pk = s.public_key();
            let pk_bytes = s.public_key_bytes();
            let sk = s.secret_key_bytes();
            (pk, pk_bytes, sk)
        }
        AlgorithmId::SlhDsaSha2_128s => {
            let s = SlhDsaSha2128sSigner::generate();
            let pk = s.public_key();
            let pk_bytes = s.public_key_bytes();
            let sk = s.secret_key_bytes();
            (pk, pk_bytes, sk)
        }
        AlgorithmId::HybridEd25519MlDsa65 => {
            let s = HybridSigner::generate();
            let pk = s.public_key();
            let pk_bytes = pk.key_bytes.clone();
            let sk = s.secret_key_bytes();
            (pk, pk_bytes, sk)
        }
        _ => bail!("Algorithm {:?} is not a signature algorithm", alg),
    };

    let key_file = KeyFile {
        algorithm: format!("{:?}", alg),
        algorithm_id: alg.as_u16(),
        public_key: pub_key.to_multibase(),
        public_key_bytes: BASE64.encode(&pub_key_bytes),
        secret_key_bytes: BASE64.encode(secret_bytes.as_slice()),
        created_at: Utc::now().to_rfc3339(),
    };

    let json = Zeroizing::new(serde_json::to_string_pretty(&key_file)?);
    std::fs::write(output, json.as_bytes()).context("Failed to write key file")?;

    // Set key file permissions to owner-only (0600)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(output, std::fs::Permissions::from_mode(0o600))
            .context("Failed to set key file permissions to 0600")?;
    }

    println!("Generated {} keypair", key_file.algorithm);
    println!("  Public key: {}", key_file.public_key);
    println!("  Key size:   {} bytes (public), {} bytes (secret)", pub_key_bytes.len(), secret_bytes.len());
    println!("  Output:     {}", output.display());

    Ok(())
}

/// Load a signer from a key file.
fn load_signer(key_path: &Path) -> Result<Box<dyn Signer>> {
    let data = std::fs::read_to_string(key_path).context("Failed to read key file")?;
    let key_file: KeyFile = serde_json::from_str(&data).context("Failed to parse key file JSON")?;

    let alg = AlgorithmId::from_u16(key_file.algorithm_id)
        .with_context(|| format!("Unknown algorithm_id: {}", key_file.algorithm_id))?;

    let sk_bytes = Zeroizing::new(
        BASE64
            .decode(&key_file.secret_key_bytes)
            .context("Failed to decode secret_key_bytes")?,
    );
    let pk_bytes = BASE64
        .decode(&key_file.public_key_bytes)
        .context("Failed to decode public_key_bytes")?;

    match alg {
        AlgorithmId::Ed25519 => {
            let mut sk: [u8; 32] = sk_bytes
                .as_slice()
                .try_into()
                .context("Ed25519 secret key must be 32 bytes")?;
            let signer = Ed25519Signer::from_bytes(&sk);
            sk.zeroize();
            Ok(Box::new(signer))
        }
        AlgorithmId::MlDsa65 => {
            Ok(Box::new(MlDsa65Signer::from_bytes(&pk_bytes, &sk_bytes)?))
        }
        AlgorithmId::MlDsa87 => {
            Ok(Box::new(MlDsa87Signer::from_bytes(&pk_bytes, &sk_bytes)?))
        }
        AlgorithmId::SlhDsaSha2_128s => {
            Ok(Box::new(SlhDsaSha2128sSigner::from_bytes(&pk_bytes, &sk_bytes)?))
        }
        AlgorithmId::HybridEd25519MlDsa65 => {
            Ok(Box::new(HybridSigner::from_bytes(&pk_bytes, &sk_bytes)?))
        }
        _ => bail!("Algorithm {:?} is not a signature algorithm", alg),
    }
}

/// Sign a credential JSON file with a PQC key.
pub fn sign(key_path: &Path, credential_path: &Path, output: &Path) -> Result<()> {
    let signer = load_signer(key_path)?;

    let cred_data = std::fs::read_to_string(credential_path).context("Failed to read credential")?;
    let credential: serde_json::Value =
        serde_json::from_str(&cred_data).context("Failed to parse credential JSON")?;

    let canonical = serde_json::to_vec(&credential)?;
    let signature = signer
        .sign(&canonical)
        .map_err(|e| anyhow::anyhow!("Signing failed: {:?}", e))?;

    let alg = signer.algorithm();
    let signed = SignedCredential {
        credential,
        proof: CredentialProof {
            proof_type: "DataIntegrityProof".into(),
            cryptosuite: alg.cryptosuite().into(),
            created: Utc::now().to_rfc3339(),
            proof_purpose: "assertionMethod".into(),
            proof_value: signature.to_multibase(),
            algorithm: Some(alg.as_u16()),
            challenge: None,
            domain: None,
        },
    };

    let json = serde_json::to_string_pretty(&signed)?;
    std::fs::write(output, &json).context("Failed to write signed credential")?;

    println!("Signed with {:?}", alg);
    println!("  Cryptosuite: {}", alg.cryptosuite());
    println!("  Signature:   {} bytes", signature.signature_bytes.len());
    println!("  Output:      {}", output.display());

    Ok(())
}

/// Dual-sign a credential with Ed25519 + ML-DSA-65 (hybrid).
pub fn hybrid_sign(
    ed25519_key_path: &Path,
    pqc_key_path: &Path,
    credential_path: &Path,
    output: &Path,
) -> Result<()> {
    let ed_signer = load_signer(ed25519_key_path)?;
    let pqc_signer = load_signer(pqc_key_path)?;

    if ed_signer.algorithm() != AlgorithmId::Ed25519 {
        bail!(
            "Expected Ed25519 key, got {:?}",
            ed_signer.algorithm()
        );
    }
    if pqc_signer.algorithm() != AlgorithmId::MlDsa65 {
        bail!(
            "Expected ML-DSA-65 key, got {:?}",
            pqc_signer.algorithm()
        );
    }

    let cred_data = std::fs::read_to_string(credential_path).context("Failed to read credential")?;
    let credential: serde_json::Value =
        serde_json::from_str(&cred_data).context("Failed to parse credential JSON")?;

    let canonical = serde_json::to_vec(&credential)?;
    let ed_sig = ed_signer
        .sign(&canonical)
        .map_err(|e| anyhow::anyhow!("Ed25519 signing failed: {:?}", e))?;
    let pqc_sig = pqc_signer
        .sign(&canonical)
        .map_err(|e| anyhow::anyhow!("ML-DSA-65 signing failed: {:?}", e))?;

    // Build hybrid signature: Ed25519 (64) || ML-DSA-65 (3309)
    let mut combined = Vec::with_capacity(64 + 3309);
    combined.extend_from_slice(&ed_sig.signature_bytes);
    combined.extend_from_slice(&pqc_sig.signature_bytes);
    let hybrid_sig = TaggedSignature::new(AlgorithmId::HybridEd25519MlDsa65, combined)
        .map_err(|e| anyhow::anyhow!("Failed to build hybrid signature: {:?}", e))?;

    let signed = SignedCredential {
        credential,
        proof: CredentialProof {
            proof_type: "DataIntegrityProof".into(),
            cryptosuite: AlgorithmId::HybridEd25519MlDsa65.cryptosuite().into(),
            created: Utc::now().to_rfc3339(),
            proof_purpose: "assertionMethod".into(),
            proof_value: hybrid_sig.to_multibase(),
            algorithm: Some(AlgorithmId::HybridEd25519MlDsa65.as_u16()),
            challenge: None,
            domain: None,
        },
    };

    let json = serde_json::to_string_pretty(&signed)?;
    std::fs::write(output, &json).context("Failed to write signed credential")?;

    println!("Hybrid-signed (Ed25519 + ML-DSA-65)");
    println!("  Cryptosuite: {}", AlgorithmId::HybridEd25519MlDsa65.cryptosuite());
    println!("  Signature:   {} bytes (64 Ed25519 + 3309 ML-DSA-65)", hybrid_sig.signature_bytes.len());
    println!("  Output:      {}", output.display());

    Ok(())
}

/// Load a public key from a key file for verification.
fn load_public_key(key_path: &Path) -> Result<TaggedPublicKey> {
    let data = std::fs::read_to_string(key_path).context("Failed to read key file")?;
    let key_file: KeyFile = serde_json::from_str(&data).context("Failed to parse key file JSON")?;
    TaggedPublicKey::from_multibase(&key_file.public_key)
        .map_err(|e| anyhow::anyhow!("Failed to decode public key from key file: {:?}", e))
}

/// Verify a PQC/hybrid signature on a signed credential.
///
/// Key modes:
/// - `key_path`: single key file (works for all algorithms including hybrid)
/// - `ed25519_key_path` + `pqc_key_path`: separate keys for dual-key hybrid
///   signatures produced by `hybrid-sign`
/// - No keys: structural validation only (with opportunistic issuer-key check)
pub fn pqc_verify(
    credential_path: &Path,
    key_path: Option<&Path>,
    ed25519_key_path: Option<&Path>,
    pqc_key_path: Option<&Path>,
    verbose: bool,
) -> Result<()> {
    let data = std::fs::read_to_string(credential_path).context("Failed to read signed credential")?;
    let signed: SignedCredential =
        serde_json::from_str(&data).context("Failed to parse signed credential JSON")?;

    let alg_id = signed
        .proof
        .algorithm
        .with_context(|| "No algorithm field in proof")?;
    let alg = AlgorithmId::from_u16(alg_id)
        .with_context(|| format!("Unknown algorithm_id: {}", alg_id))?;

    let signature = TaggedSignature::from_multibase(&signed.proof.proof_value)
        .map_err(|e| anyhow::anyhow!("Failed to decode proof_value: {:?}", e))?;

    println!("Verifying {:?} signature...", alg);
    println!("  Cryptosuite: {}", signed.proof.cryptosuite);
    println!("  Signature:   {} bytes", signature.signature_bytes.len());

    if verbose {
        println!("  Algorithm ID: 0x{:04X}", alg_id);
        println!("  Created:      {}", signed.proof.created);
        if let Some(ed_comp) = signature.ed25519_component() {
            println!("  Ed25519 component: {} bytes", ed_comp.len());
        }
        if let Some(pqc_comp) = signature.pqc_component() {
            println!("  PQC component:     {} bytes", pqc_comp.len());
        }
    }

    // Structural validation
    let valid_structure = match alg {
        AlgorithmId::Ed25519 => signature.signature_bytes.len() == 64,
        AlgorithmId::MlDsa65 => signature.signature_bytes.len() == 3309,
        AlgorithmId::MlDsa87 => signature.signature_bytes.len() == 4627,
        AlgorithmId::SlhDsaSha2_128s => signature.signature_bytes.len() == 7856,
        AlgorithmId::HybridEd25519MlDsa65 => {
            signature.signature_bytes.len() == 64 + 3309
                && signature.ed25519_component().is_some()
                && signature.pqc_component().is_some()
        }
        _ => false,
    };

    if valid_structure {
        println!("  Structure:   VALID (correct length for {:?})", alg);
    } else {
        println!("  Structure:   INVALID (unexpected length {} for {:?})", signature.signature_bytes.len(), alg);
    }

    // Full cryptographic verification when key file(s) are provided.
    let has_dual_keys = ed25519_key_path.is_some() && pqc_key_path.is_some();
    if let Some(kp) = key_path {
        let pub_key = load_public_key(kp)?;
        let canonical = serde_json::to_vec(&signed.credential)?;
        let result = verify_with_algorithm(&pub_key, &canonical, &signature, alg);
        match result {
            Ok(true) => println!("  Cryptographic verification: PASSED"),
            Ok(false) => {
                println!("  Cryptographic verification: FAILED");
                bail!("Cryptographic signature verification failed");
            }
            Err(e) => {
                println!("  Cryptographic verification: ERROR ({:?})", e);
                bail!("Cryptographic verification error: {:?}", e);
            }
        }
    } else if has_dual_keys && alg == AlgorithmId::HybridEd25519MlDsa65 {
        // Dual-key verification for hybrid-sign output: verify each component separately.
        let ed_pub = load_public_key(ed25519_key_path.unwrap())?;
        let pqc_pub = load_public_key(pqc_key_path.unwrap())?;
        let canonical = serde_json::to_vec(&signed.credential)?;

        // Split the hybrid signature into Ed25519 (64 bytes) + ML-DSA-65 (rest)
        let ed_sig_bytes = signature.ed25519_component()
            .with_context(|| "Could not extract Ed25519 component from hybrid signature")?;
        let pqc_sig_bytes = signature.pqc_component()
            .with_context(|| "Could not extract PQC component from hybrid signature")?;

        let ed_sig = TaggedSignature::new(AlgorithmId::Ed25519, ed_sig_bytes.to_vec())
            .map_err(|e| anyhow::anyhow!("Failed to build Ed25519 signature: {:?}", e))?;
        let pqc_sig = TaggedSignature::new(AlgorithmId::MlDsa65, pqc_sig_bytes.to_vec())
            .map_err(|e| anyhow::anyhow!("Failed to build ML-DSA-65 signature: {:?}", e))?;

        let ed_result = verify_with_algorithm(&ed_pub, &canonical, &ed_sig, AlgorithmId::Ed25519);
        let pqc_result = verify_with_algorithm(&pqc_pub, &canonical, &pqc_sig, AlgorithmId::MlDsa65);

        match (ed_result, pqc_result) {
            (Ok(true), Ok(true)) => println!("  Cryptographic verification: PASSED (both components)"),
            (Ok(ed_ok), Ok(pqc_ok)) => {
                println!("  Ed25519 component:  {}", if ed_ok { "PASSED" } else { "FAILED" });
                println!("  ML-DSA-65 component: {}", if pqc_ok { "PASSED" } else { "FAILED" });
                bail!("Hybrid signature verification failed");
            }
            (Err(e), _) => {
                println!("  Ed25519 verification ERROR: {:?}", e);
                bail!("Ed25519 component verification error: {:?}", e);
            }
            (_, Err(e)) => {
                println!("  ML-DSA-65 verification ERROR: {:?}", e);
                bail!("ML-DSA-65 component verification error: {:?}", e);
            }
        }
    } else if has_dual_keys {
        bail!("--ed25519-key and --pqc-key are only valid for hybrid signatures");
    } else if let Some(issuer) = signed.credential.get("issuer") {
        // Opportunistic: try embedded issuer public key if no key file given.
        if let Some(pk_str) = issuer.get("publicKeyMultibase").and_then(|v| v.as_str()) {
            match TaggedPublicKey::from_multibase(pk_str) {
                Ok(pub_key) => {
                    let canonical = serde_json::to_vec(&signed.credential)?;
                    let result = verify_with_algorithm(&pub_key, &canonical, &signature, alg);
                    match result {
                        Ok(true) => println!("  Cryptographic verification: PASSED"),
                        Ok(false) => println!("  Cryptographic verification: FAILED"),
                        Err(e) => println!("  Cryptographic verification: ERROR ({:?})", e),
                    }
                }
                Err(e) => {
                    if verbose {
                        println!("  Could not parse issuer public key: {:?}", e);
                    }
                }
            }
        }
    }

    println!();
    if valid_structure {
        println!("Signature structure verified for {:?}", alg);
    } else {
        bail!("Signature structure invalid");
    }

    Ok(())
}

fn verify_with_algorithm(
    pub_key: &TaggedPublicKey,
    message: &[u8],
    signature: &TaggedSignature,
    alg: AlgorithmId,
) -> Result<bool> {
    let result = match alg {
        AlgorithmId::Ed25519 => Ed25519Verifier.verify(pub_key, message, signature),
        AlgorithmId::MlDsa65 => MlDsa65Verifier.verify(pub_key, message, signature),
        AlgorithmId::MlDsa87 => MlDsa87Verifier.verify(pub_key, message, signature),
        AlgorithmId::SlhDsaSha2_128s => SlhDsaSha2128sVerifier.verify(pub_key, message, signature),
        AlgorithmId::HybridEd25519MlDsa65 => HybridVerifier.verify(pub_key, message, signature),
        _ => return Err(anyhow::anyhow!("Unsupported algorithm for verification: {:?}", alg)),
    };
    result.map_err(|e| anyhow::anyhow!("Verification error: {:?}", e))
}
