//! Optional PSK encryption for mesh payloads.
//!
//! When MESH_ENCRYPTION_KEY is set (64 hex chars = 32 bytes),
//! payloads are encrypted with XChaCha20-Poly1305 before serialization
//! and decrypted after reassembly. Adds 40 bytes overhead (24 nonce + 16 tag).

use chacha20poly1305::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    XChaCha20Poly1305, XNonce,
};

/// Overhead per encrypted payload: 24-byte nonce + 16-byte auth tag.
pub const ENCRYPTION_OVERHEAD: usize = 40;

/// Load the PSK from MESH_ENCRYPTION_KEY env var.
/// Returns None if not set or invalid.
pub fn load_psk() -> Option<XChaCha20Poly1305> {
    let key_hex = std::env::var("MESH_ENCRYPTION_KEY").ok()?;
    let key_bytes = hex::decode(key_hex.trim()).ok()?;
    if key_bytes.len() != 32 {
        tracing::warn!(
            "MESH_ENCRYPTION_KEY must be 64 hex chars (32 bytes), got {} bytes — encryption disabled",
            key_bytes.len()
        );
        return None;
    }
    let key = chacha20poly1305::Key::from_slice(&key_bytes);
    Some(XChaCha20Poly1305::new(key))
}

/// Encrypt plaintext with the PSK. Returns nonce || ciphertext.
pub fn encrypt(cipher: &XChaCha20Poly1305, plaintext: &[u8]) -> Result<Vec<u8>, String> {
    let nonce = XChaCha20Poly1305::generate_nonce(&mut OsRng);
    let ciphertext = cipher
        .encrypt(&nonce, plaintext)
        .map_err(|e| format!("encrypt: {e}"))?;
    let mut out = Vec::with_capacity(24 + ciphertext.len());
    out.extend_from_slice(nonce.as_slice());
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

/// Decrypt nonce || ciphertext with the PSK.
pub fn decrypt(cipher: &XChaCha20Poly1305, data: &[u8]) -> Result<Vec<u8>, String> {
    if data.len() < 24 {
        return Err("ciphertext too short for nonce".into());
    }
    let (nonce_bytes, ciphertext) = data.split_at(24);
    let nonce = XNonce::from_slice(nonce_bytes);
    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|e| format!("decrypt: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chacha20poly1305::aead::KeyInit;

    fn test_cipher(key_bytes: &[u8; 32]) -> XChaCha20Poly1305 {
        let key = chacha20poly1305::Key::from_slice(key_bytes);
        XChaCha20Poly1305::new(key)
    }

    #[test]
    fn test_roundtrip() {
        let cipher = test_cipher(&[0xAB; 32]);
        let plaintext = b"Hello mesh network!";
        let encrypted = encrypt(&cipher, plaintext).unwrap();
        assert_ne!(&encrypted[24..], plaintext); // not plaintext
        let decrypted = decrypt(&cipher, &encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_bad_key_fails_decrypt() {
        let cipher_a = test_cipher(&[0xAA; 32]);
        let cipher_b = test_cipher(&[0xBB; 32]);
        let plaintext = b"secret data";
        let encrypted = encrypt(&cipher_a, plaintext).unwrap();
        let result = decrypt(&cipher_b, &encrypted);
        assert!(result.is_err(), "different key must fail decryption");
    }

    #[test]
    fn test_short_ciphertext_rejected() {
        let cipher = test_cipher(&[0xCC; 32]);
        // Fewer than 24 bytes (nonce length)
        let result = decrypt(&cipher, &[0u8; 10]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too short"));
    }

    #[test]
    fn test_load_psk_none_when_unset() {
        // MESH_ENCRYPTION_KEY should not be set in the test environment
        // (and if it is, this test would need to be adjusted).
        // We can't easily unset env vars in a thread-safe way, so we
        // verify load_psk returns None when given a bad-length key.
        std::env::remove_var("MESH_ENCRYPTION_KEY");
        assert!(load_psk().is_none());
    }

    #[test]
    fn test_encryption_overhead_fits_lora() {
        // A heartbeat payload (empty data) + RelayPayload bincode overhead + encryption
        // must fit in a single 255-byte LoRa frame (with 8-byte fragment header).
        use crate::serializer::{RelayPayload, RelayType};

        let heartbeat = RelayPayload::new(RelayType::Heartbeat, [1; 8], Vec::new());
        let bytes = heartbeat.to_bytes();
        let total_with_encryption = bytes.len() + ENCRYPTION_OVERHEAD;
        // LoRa max frame = 255, fragment header = 8, so usable = 247
        assert!(
            total_with_encryption <= 247,
            "heartbeat + encryption = {} bytes, exceeds LoRa single-frame capacity (247 usable)",
            total_with_encryption
        );
    }
}
