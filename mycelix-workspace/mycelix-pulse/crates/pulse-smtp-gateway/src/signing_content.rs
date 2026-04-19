// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical byte layout for Pulse signing content.
//!
//! These helpers MUST stay in lock-step with the integrity zome at
//! `holochain/zomes/messages/integrity/src/lib.rs:22` (email) and `:51`
//! (delivery receipt). Any change to the canonical format there must be
//! mirrored here or the gateway's re-signed envelopes won't validate.
//!
//! Used by the gateway to sign `receive_external` entries under the
//! gateway's own Dilithium3 DID key, so recipients can distinguish
//! gateway-origin mail from peer-origin mail.

/// Canonical content for an `EncryptedEmail` signature.
///
/// Layout:
/// ```text
/// 0x01 || sender[39] || recipient[39]
///      || subj_len[4 LE] || subj_bytes
///      || body_len[4 LE] || body_bytes
///      || nonce[24]
///      || msgid_len[4 LE] || msgid_bytes
///      || timestamp_micros[8 LE]
/// ```
pub fn email_signing_content(
    sender_raw_39: &[u8],
    recipient_raw_39: &[u8],
    encrypted_subject: &[u8],
    encrypted_body: &[u8],
    nonce: &[u8; 24],
    message_id: &str,
    timestamp_micros: i64,
) -> Vec<u8> {
    assert_eq!(sender_raw_39.len(), 39, "sender pubkey must be 39 bytes");
    assert_eq!(
        recipient_raw_39.len(),
        39,
        "recipient pubkey must be 39 bytes"
    );

    let mut content = Vec::with_capacity(256);
    content.push(0x01);
    content.extend_from_slice(sender_raw_39);
    content.extend_from_slice(recipient_raw_39);
    content.extend_from_slice(&(encrypted_subject.len() as u32).to_le_bytes());
    content.extend_from_slice(encrypted_subject);
    content.extend_from_slice(&(encrypted_body.len() as u32).to_le_bytes());
    content.extend_from_slice(encrypted_body);
    content.extend_from_slice(nonce);
    content.extend_from_slice(&(message_id.len() as u32).to_le_bytes());
    content.extend_from_slice(message_id.as_bytes());
    content.extend_from_slice(&timestamp_micros.to_le_bytes());
    content
}

/// Canonical content for a `DeliveryReceipt` signature.
///
/// Layout:
/// ```text
/// 0x01 || email_hash[39] || recipient[39] || delivered_at_micros[8 LE]
/// ```
pub fn delivery_receipt_signing_content(
    email_hash_raw_39: &[u8],
    recipient_raw_39: &[u8],
    delivered_at_micros: i64,
) -> Vec<u8> {
    let mut content = Vec::with_capacity(128);
    content.push(0x01);
    content.extend_from_slice(email_hash_raw_39);
    content.extend_from_slice(recipient_raw_39);
    content.extend_from_slice(&delivered_at_micros.to_le_bytes());
    content
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn email_content_is_deterministic() {
        let sender = [0u8; 39];
        let recipient = [1u8; 39];
        let nonce = [2u8; 24];
        let a = email_signing_content(
            &sender,
            &recipient,
            b"subj",
            b"body",
            &nonce,
            "m001",
            1700000000_000_000,
        );
        let b = email_signing_content(
            &sender,
            &recipient,
            b"subj",
            b"body",
            &nonce,
            "m001",
            1700000000_000_000,
        );
        assert_eq!(a, b);
    }

    #[test]
    fn email_content_differs_when_timestamp_differs() {
        let sender = [0u8; 39];
        let recipient = [1u8; 39];
        let nonce = [2u8; 24];
        let a = email_signing_content(&sender, &recipient, b"", b"", &nonce, "x", 1000);
        let b = email_signing_content(&sender, &recipient, b"", b"", &nonce, "x", 1001);
        assert_ne!(a, b);
    }

    #[test]
    fn receipt_content_deterministic() {
        let email_hash = [3u8; 39];
        let recipient = [4u8; 39];
        let a = delivery_receipt_signing_content(&email_hash, &recipient, 42);
        let b = delivery_receipt_signing_content(&email_hash, &recipient, 42);
        assert_eq!(a, b);
    }
}
