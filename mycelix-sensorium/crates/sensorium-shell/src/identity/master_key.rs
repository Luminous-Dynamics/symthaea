// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Master key management — one key to derive all domain-specific keys.
//!
//! The master secret is generated once during First Breath and never stored
//! unencrypted. Domain keys are derived via HKDF with unique context bytes
//! per domain, ensuring backward compatibility with existing health vault keys.

use sha2::{Digest, Sha256};

/// Check if a master key exists in localStorage.
pub fn has_stored_master() -> bool {
    let Some(window) = web_sys::window() else {
        return false;
    };
    let Ok(Some(storage)) = window.local_storage() else {
        return false;
    };
    storage
        .get_item("mycelix_master_vault")
        .ok()
        .flatten()
        .is_some()
}

/// Generate 32 bytes of cryptographic entropy.
pub fn generate_entropy() -> Result<[u8; 32], String> {
    let window = web_sys::window().ok_or("No window")?;
    let crypto = window.crypto().map_err(|_| "No Web Crypto API")?;
    let mut entropy = [0u8; 32];
    crypto
        .get_random_values_with_u8_array(&mut entropy)
        .map_err(|_| "getRandomValues failed")?;
    Ok(entropy)
}

/// Derive a domain-specific key from master entropy using HKDF-SHA256.
///
/// Each domain has a unique `context` (e.g., b"mycelix-health-v1-patient-encryption").
/// The same master entropy always produces the same domain key for a given context.
/// This means the health vault key derived here is identical to the one the
/// health portal derives, ensuring backward compatibility.
pub fn derive_domain_key(context: &[u8]) -> [u8; 32] {
    // This is a placeholder — in production, the actual master entropy
    // would be held in memory after vault unlock.
    // The HKDF derivation matches the health portal's derive_key() exactly.
    use hmac::{Hmac, Mac};
    type HmacSha256 = Hmac<Sha256>;

    let salt = b"mycelix-health-v1-patient-encryption"; // Universal salt
    let ikm = b"placeholder-master-key"; // Would be real master in production

    // HKDF-Extract
    let mut mac = HmacSha256::new_from_slice(salt).expect("valid key");
    mac.update(ikm);
    let prk = mac.finalize().into_bytes();

    // HKDF-Expand with domain context
    let mut mac = HmacSha256::new_from_slice(&prk).expect("valid key");
    mac.update(context);
    mac.update(&[0x01]);
    let okm = mac.finalize().into_bytes();

    let mut key = [0u8; 32];
    key.copy_from_slice(&okm);
    key
}

/// Store wrapped master key in localStorage.
pub fn store_master(entropy: &[u8; 32], passphrase: &str) -> Result<(), String> {
    let window = web_sys::window().ok_or("No window")?;
    let storage = window
        .local_storage()
        .map_err(|_| "No localStorage")?
        .ok_or("localStorage null")?;

    // Derive wrapping key via iterated SHA-256
    let pass_key = stretch_passphrase(passphrase);

    let mut wrapped = [0u8; 32];
    for i in 0..32 {
        wrapped[i] = entropy[i] ^ pass_key[i];
    }

    let hex: String = wrapped.iter().map(|b| format!("{:02x}", b)).collect();
    storage
        .set_item("mycelix_master_vault", &hex)
        .map_err(|_| "Failed to store")?;

    // Fingerprint for verification
    let fp = Sha256::digest(entropy);
    let fp_hex: String = fp[..8].iter().map(|b| format!("{:02x}", b)).collect();
    storage
        .set_item("mycelix_master_fp", &fp_hex)
        .map_err(|_| "Failed to store fingerprint")?;

    Ok(())
}

/// Unwrap master key with passphrase.
pub fn unwrap_master(passphrase: &str) -> Result<[u8; 32], String> {
    let window = web_sys::window().ok_or("No window")?;
    let storage = window
        .local_storage()
        .map_err(|_| "No localStorage")?
        .ok_or("localStorage null")?;

    let hex = storage
        .get_item("mycelix_master_vault")
        .map_err(|_| "Read failed")?
        .ok_or("No stored master key")?;

    let wrapped: Vec<u8> = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap_or(0))
        .collect();

    if wrapped.len() != 32 {
        return Err("Stored key has wrong length".into());
    }

    let pass_key = stretch_passphrase(passphrase);
    let mut entropy = [0u8; 32];
    for i in 0..32 {
        entropy[i] = wrapped[i] ^ pass_key[i];
    }

    // Verify fingerprint
    let fp = Sha256::digest(&entropy);
    let fp_hex: String = fp[..8].iter().map(|b| format!("{:02x}", b)).collect();
    let stored_fp = storage
        .get_item("mycelix_master_fp")
        .map_err(|_| "Read failed")?
        .ok_or("No fingerprint")?;

    if fp_hex != stored_fp {
        return Err("Wrong passphrase".into());
    }

    Ok(entropy)
}

/// 10,000-iteration SHA-256 stretching.
fn stretch_passphrase(passphrase: &str) -> [u8; 32] {
    // Keep the legacy salt stable so existing wrapped masters remain decryptable
    // across the Portal -> Sensorium package rename.
    let salt = b"mycelix-portal-master-wrap-v1";
    let mut key: [u8; 32] = Sha256::digest(passphrase.as_bytes()).into();
    for _ in 0..10_000 {
        let mut input = Vec::with_capacity(32 + salt.len());
        input.extend_from_slice(&key);
        input.extend_from_slice(salt);
        key = Sha256::digest(&input).into();
    }
    key
}
