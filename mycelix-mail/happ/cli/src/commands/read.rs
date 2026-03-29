// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use std::fs;

use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};
use chacha20poly1305::{
    aead::{Aead, KeyInit},
    ChaCha20Poly1305, Nonce,
};
use x25519_dalek::{PublicKey, StaticSecret};

use crate::client::MycellixClient;
use crate::config::Config;

/// Read and display a specific message
pub async fn handle_read(
    client: &MycellixClient,
    message_id: String,
    mark_read: bool,
) -> Result<()> {
    println!("📖 Reading message...");
    println!();

    // 1. Get message from client
    let message = client
        .get_message(&message_id)
        .await
        .context("Failed to fetch message")?;

    // 2. Decrypt subject
    let subject = decrypt_subject(&message.subject_encrypted);

    // 3. Fetch body from IPFS/DHT
    let body = fetch_body(&message.body_cid).await?;

    // 4. Display formatted message
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("                         MESSAGE DETAILS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("📬 From:    {}", message.from_did);
    println!("📭 To:      {}", message.to_did);
    println!("📅 Date:    {}", format_timestamp(message.timestamp));
    println!("🏷️  Tier:    {}", format_tier(&message.epistemic_tier));

    if let Some(ref thread) = message.thread_id {
        println!("🧵 Thread:  {}", thread);
    }

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Subject: {}", subject);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("{}", body);
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // 5. Mark as read if requested
    if mark_read {
        println!();
        println!("✅ Marking message as read...");

        client
            .mark_read(&message_id)
            .await
            .context("Failed to mark message as read")?;

        println!("   Message marked as read");
    } else {
        println!();
        println!("💡 Use --mark-read to mark this message as read");
    }

    Ok(())
}

/// Load private key from config directory
fn load_private_key() -> Result<StaticSecret> {
    let keys_dir = Config::keys_dir().context("Failed to get keys directory")?;
    let private_key_path = keys_dir.join("private.key");

    if !private_key_path.exists() {
        bail!(
            "Private key not found at {:?}. Run 'mycelix-mail init' first.",
            private_key_path
        );
    }

    let key_bytes = fs::read(&private_key_path).context("Failed to read private key")?;

    if key_bytes.len() != 32 {
        bail!(
            "Invalid private key length: expected 32 bytes, got {}",
            key_bytes.len()
        );
    }

    let mut key_array = [0u8; 32];
    key_array.copy_from_slice(&key_bytes);
    Ok(StaticSecret::from(key_array))
}

/// Decrypt subject using X25519 + ChaCha20-Poly1305
///
/// Decrypts hybrid-encrypted subject line:
/// Input format: ephemeral_pubkey (32) || nonce (12) || ciphertext
fn decrypt_subject(encrypted: &[u8]) -> String {
    // Check minimum length: 32 (pubkey) + 12 (nonce) + 16 (auth tag) = 60 bytes minimum
    if encrypted.len() < 60 {
        // Try legacy/plaintext format
        if let Ok(s) = String::from_utf8(encrypted.to_vec()) {
            if s.starts_with("ENC:") {
                return s[4..].to_string();
            }
            return s;
        }
        return "<encrypted - invalid format>".to_string();
    }

    // Try to load private key and decrypt
    match decrypt_subject_impl(encrypted) {
        Ok(plaintext) => plaintext,
        Err(e) => {
            // Fall back to legacy format
            if let Ok(s) = String::from_utf8(encrypted.to_vec()) {
                if s.starts_with("ENC:") {
                    return s[4..].to_string();
                }
                return s;
            }
            format!("<decryption failed: {}>", e)
        }
    }
}

/// Implementation of subject decryption
fn decrypt_subject_impl(encrypted: &[u8]) -> Result<String> {
    // Parse input: ephemeral_pubkey (32) || nonce (12) || ciphertext
    if encrypted.len() < 60 {
        bail!("Ciphertext too short");
    }

    let ephemeral_pubkey_bytes: [u8; 32] = encrypted[0..32]
        .try_into()
        .context("Invalid ephemeral public key")?;
    let nonce_bytes: [u8; 12] = encrypted[32..44].try_into().context("Invalid nonce")?;
    let ciphertext = &encrypted[44..];

    let ephemeral_pubkey = PublicKey::from(ephemeral_pubkey_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);

    // Load our private key
    let private_key = load_private_key()?;

    // Perform X25519 ECDH to get shared secret
    let shared_secret = private_key.diffie_hellman(&ephemeral_pubkey);

    // Derive symmetric key using Blake2b (must match send.rs)
    let mut hasher = Blake2b::<U32>::new();
    hasher.update(shared_secret.as_bytes());
    hasher.update(b"mycelix-mail-subject-encryption");
    let symmetric_key: [u8; 32] = hasher.finalize().into();

    // Decrypt with ChaCha20-Poly1305
    let cipher =
        ChaCha20Poly1305::new_from_slice(&symmetric_key).context("Invalid symmetric key")?;

    let plaintext = cipher
        .decrypt(nonce, ciphertext)
        .map_err(|e| anyhow::anyhow!("Decryption failed: {:?}", e))?;

    String::from_utf8(plaintext).context("Decrypted subject is not valid UTF-8")
}

/// Fetch body from IPFS/DHT using CID
///
/// Attempts to retrieve content from:
/// 1. Local IPFS gateway (localhost:8080)
/// 2. Public IPFS gateways as fallback
async fn fetch_body(cid: &str) -> Result<String> {
    // Validate CID format
    if !cid.starts_with("bafy") && !cid.starts_with("Qm") {
        bail!(
            "Invalid CID format: {}. Expected IPFS CID starting with 'bafy' or 'Qm'",
            cid
        );
    }

    // List of IPFS gateways to try (local first, then public)
    let gateways = [
        "http://localhost:8080/ipfs",
        "http://127.0.0.1:8080/ipfs",
        "http://localhost:5001/api/v0/cat?arg=", // IPFS API endpoint
        "https://ipfs.io/ipfs",
        "https://dweb.link/ipfs",
        "https://cloudflare-ipfs.com/ipfs",
        "https://gateway.pinata.cloud/ipfs",
    ];

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .context("Failed to create HTTP client")?;

    let mut last_error = None;

    for gateway in gateways {
        let url = if gateway.contains("?arg=") {
            // IPFS API format
            format!("{}{}", gateway, cid)
        } else {
            // Gateway format
            format!("{}/{}", gateway, cid)
        };

        match client.get(&url).send().await {
            Ok(response) if response.status().is_success() => {
                match response.text().await {
                    Ok(body) => {
                        // Check if body is encrypted (starts with our encryption marker)
                        if body.starts_with("ENCRYPTED:") {
                            // Decrypt body using same scheme as subject
                            match decrypt_body(&body[10..]) {
                                Ok(decrypted) => return Ok(decrypted),
                                Err(e) => {
                                    eprintln!("Warning: Failed to decrypt body: {}", e);
                                    return Ok(body); // Return encrypted body if decryption fails
                                }
                            }
                        }
                        return Ok(body);
                    }
                    Err(e) => {
                        last_error = Some(format!("Failed to read response body: {}", e));
                    }
                }
            }
            Ok(response) => {
                last_error = Some(format!(
                    "Gateway {} returned status {}",
                    gateway,
                    response.status()
                ));
            }
            Err(e) => {
                last_error = Some(format!("Failed to connect to {}: {}", gateway, e));
            }
        }
    }

    // If all gateways failed, return helpful error
    bail!(
        "Failed to fetch body from IPFS. CID: {}\n\
         Last error: {}\n\n\
         To fix this:\n\
         1. Ensure IPFS daemon is running: ipfs daemon\n\
         2. Or check internet connectivity for public gateways\n\
         3. The content may not be pinned or available yet",
        cid,
        last_error.unwrap_or_else(|| "Unknown error".to_string())
    )
}

/// Decrypt body content (if encrypted)
fn decrypt_body(encrypted_base64: &str) -> Result<String> {
    // Decode base64
    let encrypted = base64_decode(encrypted_base64)?;

    // Use same decryption as subject
    decrypt_subject_impl(&encrypted)
}

/// Simple base64 decoding helper
fn base64_decode(input: &str) -> Result<Vec<u8>> {
    // Use standard base64 alphabet
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let input = input.trim().replace("\n", "").replace("\r", "");
    let input = input.trim_end_matches('=');

    let mut output = Vec::with_capacity(input.len() * 3 / 4);
    let mut buffer = 0u32;
    let mut bits = 0;

    for c in input.bytes() {
        let value = ALPHABET
            .iter()
            .position(|&x| x == c)
            .ok_or_else(|| anyhow::anyhow!("Invalid base64 character: {}", c as char))?;

        buffer = (buffer << 6) | (value as u32);
        bits += 6;

        if bits >= 8 {
            bits -= 8;
            output.push((buffer >> bits) as u8);
            buffer &= (1 << bits) - 1;
        }
    }

    Ok(output)
}

/// Format timestamp as human-readable date/time
fn format_timestamp(ts: i64) -> String {
    let dt = DateTime::<Utc>::from_timestamp(ts, 0).unwrap_or_else(|| Utc::now());

    dt.format("%Y-%m-%d %H:%M:%S UTC").to_string()
}

/// Format epistemic tier as full string
fn format_tier(tier: &crate::types::EpistemicTier) -> String {
    use crate::types::EpistemicTier;
    match tier {
        EpistemicTier::Tier0Null => "Tier 0 (Null - Unverifiable belief)".to_string(),
        EpistemicTier::Tier1Testimonial => {
            "Tier 1 (Testimonial - Personal attestation)".to_string()
        }
        EpistemicTier::Tier2PrivatelyVerifiable => {
            "Tier 2 (Privately Verifiable - Audit guild)".to_string()
        }
        EpistemicTier::Tier3CryptographicallyProven => {
            "Tier 3 (Cryptographically Proven - ZKP)".to_string()
        }
        EpistemicTier::Tier4PubliclyReproducible => {
            "Tier 4 (Publicly Reproducible - Open data/code)".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decrypt_subject_legacy_format() {
        // Test with legacy "ENC:" prefix (fallback format)
        let encrypted = b"ENC:Test Subject";
        let decrypted = decrypt_subject(encrypted);
        assert_eq!(decrypted, "Test Subject");

        // Test with non-encrypted plaintext
        let plain = b"Plain Subject";
        let result = decrypt_subject(plain);
        assert_eq!(result, "Plain Subject");
    }

    #[test]
    fn test_decrypt_subject_too_short() {
        // Test with data too short for real encryption format
        let short = b"short";
        let result = decrypt_subject(short);
        // Should fall back to plaintext
        assert_eq!(result, "short");
    }

    #[tokio::test]
    async fn test_fetch_body_invalid_cid() {
        let cid = "invalid-cid-format";
        let result = fetch_body(cid).await;

        // Should fail for invalid CID
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Invalid CID format"));
    }

    #[tokio::test]
    async fn test_fetch_body_valid_cid_format() {
        // Test with valid CID format (will fail to fetch but validates format)
        let cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi";
        let result = fetch_body(cid).await;

        // Will likely fail (no IPFS running) but should not fail on format
        // The error should mention IPFS/gateway issues, not format
        if let Err(e) = result {
            let err_str = e.to_string();
            assert!(!err_str.contains("Invalid CID format"));
        }
    }

    #[test]
    fn test_base64_decode() {
        // Test base64 decoding
        let encoded = "SGVsbG8gV29ybGQ="; // "Hello World"
        let decoded = base64_decode(encoded).unwrap();
        assert_eq!(decoded, b"Hello World");

        // Test without padding
        let encoded_no_pad = "SGVsbG8gV29ybGQ";
        let decoded_no_pad = base64_decode(encoded_no_pad).unwrap();
        assert_eq!(decoded_no_pad, b"Hello World");
    }

    #[test]
    fn test_format_timestamp() {
        // Test with known timestamp
        let ts = 1609459200; // 2021-01-01 00:00:00 UTC
        let formatted = format_timestamp(ts);
        assert!(formatted.contains("2021"));
        assert!(formatted.contains("UTC"));
    }

    #[test]
    fn test_format_tier() {
        use crate::types::EpistemicTier;

        let tier0 = format_tier(&EpistemicTier::Tier0Null);
        assert!(tier0.contains("Tier 0"));
        assert!(tier0.contains("Null"));

        let tier2 = format_tier(&EpistemicTier::Tier2PrivatelyVerifiable);
        assert!(tier2.contains("Tier 2"));
        assert!(tier2.contains("Privately Verifiable"));

        let tier4 = format_tier(&EpistemicTier::Tier4PubliclyReproducible);
        assert!(tier4.contains("Tier 4"));
        assert!(tier4.contains("Publicly Reproducible"));
    }
}
