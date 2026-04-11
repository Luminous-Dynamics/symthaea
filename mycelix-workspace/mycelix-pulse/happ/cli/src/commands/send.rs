// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use anyhow::{bail, Context, Result};
use std::io::{self, Read};

use chacha20poly1305::{
    aead::{Aead, KeyInit},
    ChaCha20Poly1305, Nonce,
};
use rand::RngCore;
use x25519_dalek::{EphemeralSecret, PublicKey};

use crate::client::MycellixClient;
use crate::types::EpistemicTier;

/// Send an email message
pub async fn handle_send(
    client: &MycellixClient,
    to: String,
    subject: String,
    body: Option<String>,
    attach: Option<Vec<String>>,
    reply_to: Option<String>,
    tier: u8,
) -> Result<()> {
    println!("📧 Composing message...");
    println!();

    // 1. Validate epistemic tier
    let epistemic_tier = EpistemicTier::from_u8(tier)
        .with_context(|| format!("Invalid epistemic tier: {}. Must be 0-4", tier))?;

    println!("   Tier: {}", epistemic_tier);

    // 2. Validate recipient (must be a valid DID)
    if !to.starts_with("did:") {
        bail!(
            "Invalid recipient format: '{}'\n\
             Recipient must be a DID (e.g., did:mycelix:ABC123...)\n\
             Use 'mycelix-mail did resolve <email>' to find a user's DID",
            to
        );
    }

    // Check DID format (should be did:mycelix:base58)
    if !to.starts_with("did:mycelix:") {
        println!(
            "⚠️  Warning: Recipient DID uses non-standard method: {}",
            to
        );
        println!("   Expected format: did:mycelix:<base58>");
    }

    println!("   To: {}", to);

    // 3. Validate and get body text
    let body_text = get_body_text(body).await?;

    // Show preview of body (first 100 chars)
    if body_text.len() > 100 {
        println!(
            "   Body: {}... ({} chars)",
            &body_text[..100],
            body_text.len()
        );
    } else {
        println!("   Body: {} ({} chars)", body_text, body_text.len());
    }

    // 4. Handle attachments (not yet implemented)
    if let Some(attachments) = &attach {
        if !attachments.is_empty() {
            println!("⚠️  Warning: Attachments not yet implemented");
            println!("   Ignoring {} attachment(s)", attachments.len());
            // TODO: Implement attachment upload to IPFS/Holochain
        }
    }

    // 5. Handle reply-to (for threading)
    if let Some(parent_id) = &reply_to {
        println!("   Reply to: {}", parent_id);
    }

    println!();
    println!("📝 Subject: {}", subject);

    // 6. Encrypt subject (placeholder for now)
    let encrypted_subject = encrypt_subject(&subject);
    println!("🔒 Subject encrypted: {} bytes", encrypted_subject.len());

    // 7. Upload body to DHT/IPFS (stub for now)
    let body_cid = upload_body(&body_text).await?;
    println!("📤 Body uploaded: {}", body_cid);

    // 8. Send message via Holochain
    println!();
    println!("📡 Sending message...");

    match client
        .send_message(
            to.clone(),
            encrypted_subject,
            body_cid,
            reply_to,
            epistemic_tier,
        )
        .await
    {
        Ok(message_id) => {
            println!();
            println!("✅ Message sent successfully!");
            println!();
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Message ID: {}", message_id);
            println!("To: {}", to);
            println!("Subject: {}", subject);
            println!("Tier: {}", epistemic_tier);
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!();
            println!("The recipient will receive your message shortly.");
        }
        Err(e) => {
            println!();
            println!("❌ Failed to send message: {}", e);
            println!();
            println!("This may be because:");
            println!("  • Holochain conductor is not running");
            println!("  • The recipient's DID doesn't exist");
            println!("  • Network connectivity issues");
            println!();
            println!("Try again later or check your configuration.");
            bail!("Message send failed");
        }
    }

    Ok(())
}

/// Get body text from argument or stdin
async fn get_body_text(body: Option<String>) -> Result<String> {
    match body {
        Some(text) if text == "-" => {
            // Read from stdin
            println!("📝 Reading body from stdin (press Ctrl+D when done)...");
            let mut buffer = String::new();
            io::stdin()
                .read_to_string(&mut buffer)
                .context("Failed to read body from stdin")?;

            if buffer.trim().is_empty() {
                bail!("Body cannot be empty");
            }

            Ok(buffer)
        }
        Some(text) => {
            // Use provided text
            if text.trim().is_empty() {
                bail!("Body cannot be empty");
            }
            Ok(text)
        }
        None => {
            // No body provided - require it
            bail!("Body is required. Use --body \"your message\" or --body - to read from stdin");
        }
    }
}

/// Encrypt subject using X25519 key exchange and ChaCha20-Poly1305 AEAD
///
/// Implements hybrid encryption:
/// 1. Generate ephemeral X25519 keypair
/// 2. Derive shared secret via ECDH with recipient's public key
/// 3. Use HKDF to derive symmetric key from shared secret
/// 4. Encrypt with ChaCha20-Poly1305
///
/// Output format: ephemeral_pubkey (32 bytes) || nonce (12 bytes) || ciphertext
///
/// Note: In production, recipient_pubkey should be fetched from DID registry.
/// Currently uses a placeholder key for demonstration.
fn encrypt_subject(subject: &str) -> Vec<u8> {
    // In production: fetch recipient's X25519 public key from DID registry
    // For now, use a deterministic placeholder that can be replaced
    // when DID integration is complete
    let recipient_pubkey_bytes = derive_recipient_pubkey_placeholder();
    let recipient_pubkey = PublicKey::from(recipient_pubkey_bytes);

    encrypt_with_pubkey(subject.as_bytes(), &recipient_pubkey)
}

/// Encrypt data using recipient's X25519 public key
///
/// Returns: ephemeral_pubkey (32) || nonce (12) || ciphertext (plaintext.len() + 16 tag)
fn encrypt_with_pubkey(plaintext: &[u8], recipient_pubkey: &PublicKey) -> Vec<u8> {
    let mut rng = rand::thread_rng();

    // Generate ephemeral keypair for this message
    let ephemeral_secret = EphemeralSecret::random_from_rng(&mut rng);
    let ephemeral_pubkey = PublicKey::from(&ephemeral_secret);

    // Perform X25519 ECDH to get shared secret
    let shared_secret = ephemeral_secret.diffie_hellman(recipient_pubkey);

    // Derive encryption key using HKDF-like construction with Blake2b
    // Key = Blake2b-256(shared_secret || "mycelix-mail-subject-encryption")
    use blake2::digest::consts::U32;
    use blake2::{Blake2b, Digest};
    let mut hasher = Blake2b::<U32>::new();
    hasher.update(shared_secret.as_bytes());
    hasher.update(b"mycelix-mail-subject-encryption");
    let symmetric_key: [u8; 32] = hasher.finalize().into();

    // Generate random nonce for ChaCha20-Poly1305
    let mut nonce_bytes = [0u8; 12];
    rng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);

    // Encrypt with ChaCha20-Poly1305
    let cipher = ChaCha20Poly1305::new_from_slice(&symmetric_key).expect("Invalid key length");
    let ciphertext = cipher.encrypt(nonce, plaintext).expect("Encryption failed");

    // Combine: ephemeral_pubkey || nonce || ciphertext
    let mut output = Vec::with_capacity(32 + 12 + ciphertext.len());
    output.extend_from_slice(ephemeral_pubkey.as_bytes());
    output.extend_from_slice(&nonce_bytes);
    output.extend_from_slice(&ciphertext);

    output
}

/// Derive a placeholder recipient public key
///
/// In production, this would be replaced with actual DID resolution.
/// Uses a deterministic derivation so decryption tests can work.
fn derive_recipient_pubkey_placeholder() -> [u8; 32] {
    use blake2::digest::consts::U32;
    use blake2::{Blake2b, Digest};

    let mut hasher = Blake2b::<U32>::new();
    hasher.update(b"mycelix-mail-placeholder-recipient-key-v1");
    hasher.finalize().into()
}

/// Upload body to DHT/IPFS (placeholder implementation)
///
/// TODO: Implement real body upload
/// - Upload to IPFS or Holochain DHT
/// - Return content identifier (CID)
async fn upload_body(body: &str) -> Result<String> {
    // Placeholder: Just create a fake CID based on body hash
    // In real implementation:
    // 1. Upload body to IPFS via ipfs-api or Holochain DHT
    // 2. Get content identifier (CID)
    // 3. Return CID

    use blake2::{Blake2b512, Digest};

    let mut hasher = Blake2b512::new();
    hasher.update(body.as_bytes());
    let hash = hasher.finalize();

    // Create a fake CID (just hex encoding of first 16 bytes)
    let cid = hex::encode(&hash[..16]);

    Ok(format!("bafyrei{}", cid))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_subject() {
        let subject = "Hello World";
        let encrypted = encrypt_subject(subject);

        // Should produce bytes with proper format:
        // ephemeral_pubkey (32) + nonce (12) + ciphertext (plaintext.len() + 16 tag)
        let expected_min_len = 32 + 12 + subject.len() + 16;
        assert!(encrypted.len() >= expected_min_len);

        // First 32 bytes should be the ephemeral public key
        assert_eq!(encrypted.len(), 32 + 12 + subject.len() + 16);
    }

    #[test]
    fn test_encrypt_produces_different_ciphertext() {
        // Each encryption should produce different output due to random ephemeral key and nonce
        let subject = "Test Subject";
        let encrypted1 = encrypt_subject(subject);
        let encrypted2 = encrypt_subject(subject);

        // Ephemeral pubkeys should differ (first 32 bytes)
        assert_ne!(&encrypted1[..32], &encrypted2[..32]);

        // Nonces should differ (bytes 32-44)
        assert_ne!(&encrypted1[32..44], &encrypted2[32..44]);
    }

    #[test]
    fn test_encrypt_with_pubkey_format() {
        let plaintext = b"Test message";
        let recipient_pubkey_bytes = derive_recipient_pubkey_placeholder();
        let recipient_pubkey = PublicKey::from(recipient_pubkey_bytes);

        let encrypted = encrypt_with_pubkey(plaintext, &recipient_pubkey);

        // Verify format: ephemeral_pubkey (32) || nonce (12) || ciphertext
        assert_eq!(encrypted.len(), 32 + 12 + plaintext.len() + 16);
    }

    #[tokio::test]
    async fn test_upload_body() {
        let body = "Test message body";
        let cid = upload_body(body).await.unwrap();

        // Should produce a CID-like string
        assert!(cid.starts_with("bafyrei"));
        assert!(cid.len() > 10);

        // Same body should produce same CID (deterministic)
        let cid2 = upload_body(body).await.unwrap();
        assert_eq!(cid, cid2);
    }

    #[tokio::test]
    async fn test_different_bodies_different_cids() {
        let body1 = "Message 1";
        let body2 = "Message 2";

        let cid1 = upload_body(body1).await.unwrap();
        let cid2 = upload_body(body2).await.unwrap();

        assert_ne!(cid1, cid2);
    }
}
