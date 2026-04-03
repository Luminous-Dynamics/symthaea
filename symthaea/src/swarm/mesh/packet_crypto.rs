// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cryptographic operations for mesh packets: MAC, compression, AEAD encryption.

use super::wisdom_packet::WISDOM_PACKET_SIZE;
use super::{COMPRESS_LZ4, COMPRESS_NONE};

// ============================================================================
// PACKET AUTHENTICATION (BLAKE3)
// ============================================================================

/// Compute a truncated 8-bit BLAKE3 keyed MAC over the packet bytes.
///
/// The MAC is computed with byte 22 (the auth_mac field itself) zeroed,
/// so the MAC doesn't include itself in the hash input.
pub fn compute_packet_mac(packet_bytes: &[u8; WISDOM_PACKET_SIZE], key: &[u8; 32]) -> u8 {
    let mut input = *packet_bytes;
    input[22] = 0; // Zero the auth_mac field before computing
    let hash = blake3::keyed_hash(key, &input);
    hash.as_bytes()[0]
}

/// Verify the MAC on a packet byte slice.
///
/// Returns `true` if the MAC matches, `false` otherwise.
/// Returns `false` if the slice is too short.
pub fn verify_packet_mac(packet_bytes: &[u8], key: &[u8; 32]) -> bool {
    if packet_bytes.len() < WISDOM_PACKET_SIZE {
        return false;
    }
    let mut input = [0u8; WISDOM_PACKET_SIZE];
    input.copy_from_slice(&packet_bytes[..WISDOM_PACKET_SIZE]);
    let stored_mac = input[22];
    input[22] = 0;
    let hash = blake3::keyed_hash(key, &input);
    hash.as_bytes()[0] == stored_mac
}

// ============================================================================
// PACKET COMPRESSION (LZ4)
// ============================================================================

/// Compress a raw WISDOM_PACKET_SIZE packet into an envelope.
///
/// Returns `[COMPRESS_NONE | raw]` if compression doesn't help (output >= input),
/// or `[COMPRESS_LZ4 | lz4_data]` if compression reduces size.
///
/// When `lz4_compression` feature is disabled, always returns uncompressed.
pub fn compress_packet(raw: &[u8; WISDOM_PACKET_SIZE]) -> Vec<u8> {
    #[cfg(feature = "lz4_compression")]
    {
        let compressed = lz4_flex::compress_prepend_size(raw);
        if compressed.len() + 1 < WISDOM_PACKET_SIZE + 1 {
            let mut envelope = Vec::with_capacity(1 + compressed.len());
            envelope.push(COMPRESS_LZ4);
            envelope.extend_from_slice(&compressed);
            return envelope;
        }
    }
    // Uncompressed fallback (or feature disabled)
    let mut envelope = Vec::with_capacity(1 + WISDOM_PACKET_SIZE);
    envelope.push(COMPRESS_NONE);
    envelope.extend_from_slice(raw);
    envelope
}

/// Decompress a packet envelope back to WISDOM_PACKET_SIZE bytes.
///
/// Returns `None` if the header byte is unknown or decompression fails.
/// Tolerates trailing bytes (FEC safety).
pub fn decompress_packet(data: &[u8]) -> Option<Vec<u8>> {
    if data.is_empty() {
        return None;
    }
    match data[0] {
        COMPRESS_NONE => {
            if data.len() < 1 + WISDOM_PACKET_SIZE {
                return None;
            }
            Some(data[1..1 + WISDOM_PACKET_SIZE].to_vec())
        }
        COMPRESS_LZ4 => {
            #[cfg(feature = "lz4_compression")]
            {
                lz4_flex::decompress_size_prepended(&data[1..])
                    .ok()
                    .filter(|d| d.len() == WISDOM_PACKET_SIZE)
            }
            #[cfg(not(feature = "lz4_compression"))]
            {
                let _ = data;
                None
            }
        }
        _ => None,
    }
}

// ============================================================================
// ENCRYPTION (ChaCha20-Poly1305)
// ============================================================================

/// Nonce size for ChaCha20-Poly1305 (96 bits).
#[cfg(feature = "mesh-encryption")]
pub const AEAD_NONCE_SIZE: usize = 12;

/// Authentication tag size (128 bits).
#[cfg(feature = "mesh-encryption")]
pub const AEAD_TAG_SIZE: usize = 16;

/// Build a 12-byte ChaCha20-Poly1305 nonce with type separation and epoch.
///
/// Layout: `source_id[0..6] | payload_type | epoch | sequence[0..4]`
///
/// - **payload_type** prevents nonce collision across wisdom/heartbeat/affective/gradient
///   sequences (all start at 0 but use different type bytes).
/// - **epoch** is a random byte generated once at Mind construction, preventing
///   restart nonce reuse under the same key.
/// - **sequence** is per-type monotonic (wraps safely at 2^32 ≈ 2.7 years at 50Hz).
#[cfg(feature = "mesh-encryption")]
pub fn build_nonce(source_id: &[u8; 8], payload_type: u8, epoch: u8, sequence: u32) -> [u8; 12] {
    let mut nonce = [0u8; 12];
    nonce[..6].copy_from_slice(&source_id[..6]);
    nonce[6] = payload_type;
    nonce[7] = epoch;
    nonce[8..12].copy_from_slice(&sequence.to_le_bytes());
    nonce
}

/// Encrypt a compressed packet envelope using ChaCha20-Poly1305.
///
/// Returns `[nonce (12 bytes) | ciphertext+tag]`.
/// The nonce includes payload_type and epoch to prevent cross-type
/// and restart nonce reuse. See [`build_nonce`].
///
/// **`epoch`** must be a random byte generated once at node startup
/// (via `rand::thread_rng().gen::<u8>()`). This prevents nonce reuse
/// across node restarts when using the same key material.
#[cfg(feature = "mesh-encryption")]
pub fn encrypt_packet(
    envelope: &[u8],
    key: &[u8; 32],
    source_id: &[u8; 8],
    epoch: u8,
    sequence: u32,
) -> Vec<u8> {
    encrypt_packet_typed(envelope, key, source_id, 0, epoch, sequence)
}

/// Encrypt with explicit payload_type and epoch (nonce-safe variant).
///
/// Prefer this over [`encrypt_packet`] to prevent nonce reuse across
/// packet types and across node restarts.
#[cfg(feature = "mesh-encryption")]
pub fn encrypt_packet_typed(
    envelope: &[u8],
    key: &[u8; 32],
    source_id: &[u8; 8],
    payload_type: u8,
    epoch: u8,
    sequence: u32,
) -> Vec<u8> {
    match try_encrypt_packet_typed(envelope, key, source_id, payload_type, epoch, sequence) {
        Ok(out) => out,
        Err(e) => {
            // This should never happen with valid inputs, but if it does,
            // log and return empty (safer than panic in production).
            tracing::error!("ChaCha20-Poly1305 encryption failed: {e}");
            Vec::new()
        }
    }
}

/// Fallible variant of [`encrypt_packet_typed`] that returns `Result`.
///
/// Prefer this when the caller can handle encryption failures gracefully
/// (e.g., skip the packet rather than crash).
#[cfg(feature = "mesh-encryption")]
pub fn try_encrypt_packet_typed(
    envelope: &[u8],
    key: &[u8; 32],
    source_id: &[u8; 8],
    payload_type: u8,
    epoch: u8,
    sequence: u32,
) -> Result<Vec<u8>, crate::swarm::SwarmError> {
    use chacha20poly1305::{aead::Aead, ChaCha20Poly1305, KeyInit, Nonce};
    let cipher = ChaCha20Poly1305::new(key.into());
    let nonce_bytes = build_nonce(source_id, payload_type, epoch, sequence);
    let nonce = Nonce::from(nonce_bytes);
    let ciphertext = cipher.encrypt(&nonce, envelope).map_err(|e| {
        crate::swarm::SwarmError::EncryptionFailed {
            reason: format!("ChaCha20-Poly1305: {e}"),
        }
    })?;
    let mut out = Vec::with_capacity(12 + ciphertext.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

/// Encrypt with a 1-byte key version prefix for versioned decrypt.
///
/// Wire format: `[key_version (1) | nonce (12) | ciphertext+tag]`
/// The version byte lets the receiver select the correct key without
/// blind trial decryption during key rotation grace periods.
#[cfg(feature = "mesh-encryption")]
pub fn encrypt_packet_versioned(
    envelope: &[u8],
    key: &[u8; 32],
    key_version: u8,
    source_id: &[u8; 8],
    payload_type: u8,
    epoch: u8,
    sequence: u32,
) -> Vec<u8> {
    let inner = encrypt_packet_typed(envelope, key, source_id, payload_type, epoch, sequence);
    let mut out = Vec::with_capacity(1 + inner.len());
    out.push(key_version);
    out.extend_from_slice(&inner);
    out
}

/// Decrypt a packet encrypted with `encrypt_packet`.
///
/// Returns `None` if decryption/authentication fails.
#[cfg(feature = "mesh-encryption")]
pub fn decrypt_packet(data: &[u8], key: &[u8; 32]) -> Option<Vec<u8>> {
    use chacha20poly1305::{aead::Aead, ChaCha20Poly1305, KeyInit, Nonce};
    if data.len() < AEAD_NONCE_SIZE + AEAD_TAG_SIZE {
        return None;
    }
    let (nonce_bytes, ciphertext) = data.split_at(AEAD_NONCE_SIZE);
    let cipher = ChaCha20Poly1305::new(key.into());
    let nonce = Nonce::from_slice(nonce_bytes);
    cipher.decrypt(nonce, ciphertext).ok()
}

// ============================================================================
// XChaCha20-Poly1305 (NONCE-MISUSE RESISTANT)
// ============================================================================

/// Nonce size for XChaCha20-Poly1305 (192 bits).
#[cfg(feature = "mesh-encryption")]
pub const XCHACHA_NONCE_SIZE: usize = 24;

/// Encrypt using XChaCha20-Poly1305 with a random 24-byte nonce.
///
/// XChaCha20 uses a 192-bit nonce large enough for random generation
/// without collision risk (birthday bound ≈ 2^96). This eliminates
/// nonce-reuse concerns entirely at the cost of 12 extra nonce bytes per packet.
///
/// Returns `[nonce (24 bytes) | ciphertext+tag]`.
#[cfg(feature = "mesh-encryption")]
pub fn encrypt_packet_xchacha(envelope: &[u8], key: &[u8; 32]) -> Vec<u8> {
    use chacha20poly1305::{aead::Aead, KeyInit, XChaCha20Poly1305, XNonce};
    let cipher = XChaCha20Poly1305::new(key.into());
    let mut nonce_bytes = [0u8; 24];
    rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut nonce_bytes);
    let nonce = XNonce::from(nonce_bytes);
    let ciphertext = cipher
        .encrypt(&nonce, envelope)
        .expect("encryption never fails for valid inputs");
    let mut out = Vec::with_capacity(24 + ciphertext.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    out
}

/// Decrypt a packet encrypted with [`encrypt_packet_xchacha`].
///
/// Returns `None` if decryption/authentication fails.
#[cfg(feature = "mesh-encryption")]
pub fn decrypt_packet_xchacha(data: &[u8], key: &[u8; 32]) -> Option<Vec<u8>> {
    use chacha20poly1305::{aead::Aead, KeyInit, XChaCha20Poly1305, XNonce};
    if data.len() < XCHACHA_NONCE_SIZE + AEAD_TAG_SIZE {
        return None;
    }
    let (nonce_bytes, ciphertext) = data.split_at(XCHACHA_NONCE_SIZE);
    let cipher = XChaCha20Poly1305::new(key.into());
    let nonce = XNonce::from_slice(nonce_bytes);
    cipher.decrypt(nonce, ciphertext).ok()
}

// ============================================================================
// FRAGMENT-LEVEL AEAD ENCRYPTION
// ============================================================================

/// Encrypt a single LoRa fragment with ChaCha20-Poly1305.
///
/// Each fragment gets its own AEAD envelope so that:
/// 1. Individual fragments are authenticated (tamper-proof)
/// 2. A partial capture (< all fragments) cannot be decrypted
/// 3. Fragment reordering is detected
///
/// Nonce derivation: `source_id[0..8] | thought_id[2] | fragment_index[1] | 0[1]`
///
/// Returns `[nonce (12) | ciphertext + tag]`. Overhead: 28 bytes per fragment.
#[cfg(feature = "mesh-encryption")]
pub fn encrypt_fragment(
    payload: &[u8],
    key: &[u8; 32],
    source_id: &[u8; 8],
    thought_id: u16,
    fragment_index: u8,
) -> Vec<u8> {
    use chacha20poly1305::{aead::Aead, ChaCha20Poly1305, KeyInit, Nonce};
    let cipher = ChaCha20Poly1305::new(key.into());
    let mut nonce_bytes = [0u8; 12];
    nonce_bytes[..8].copy_from_slice(source_id);
    nonce_bytes[8..10].copy_from_slice(&thought_id.to_le_bytes());
    nonce_bytes[10] = fragment_index;
    nonce_bytes[11] = 0; // reserved
    let nonce = Nonce::from(nonce_bytes);
    let ciphertext = match cipher.encrypt(&nonce, payload) {
        Ok(ct) => ct,
        Err(e) => {
            tracing::error!("Fragment encryption failed: {e}");
            return Vec::new();
        }
    };
    let mut out = Vec::with_capacity(12 + ciphertext.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    out
}

/// Decrypt a fragment encrypted with [`encrypt_fragment`].
///
/// Returns `None` if decryption/authentication fails.
#[cfg(feature = "mesh-encryption")]
pub fn decrypt_fragment(data: &[u8], key: &[u8; 32]) -> Option<Vec<u8>> {
    use chacha20poly1305::{aead::Aead, ChaCha20Poly1305, KeyInit, Nonce};
    if data.len() < AEAD_NONCE_SIZE + AEAD_TAG_SIZE {
        return None;
    }
    let (nonce_bytes, ciphertext) = data.split_at(AEAD_NONCE_SIZE);
    let cipher = ChaCha20Poly1305::new(key.into());
    let nonce = Nonce::from_slice(nonce_bytes);
    cipher.decrypt(nonce, ciphertext).ok()
}
