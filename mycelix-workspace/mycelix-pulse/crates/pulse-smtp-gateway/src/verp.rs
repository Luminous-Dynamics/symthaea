// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Variable Envelope Return Path (VERP) encoding for bounce routing.
//!
//! Problem: when Alice sends `bob@gmail.com` and the message bounces, the
//! DSN goes to whatever `MAIL FROM:<>` we used. If we just use
//! `alice@mycelix.net`, Alice's inbox gets the raw DSN and we have no way
//! to correlate it back to her original send.
//!
//! Solution: use a unique envelope-from per outbound message, encoding the
//! Pulse `message_id` directly. When the DSN comes back, we decode the
//! msg_id from the envelope-to and look up the original `OutboxEntry`.
//!
//! Format: `bounce+<HMAC[8]><base32(msg_id)>@mycelix.net`
//!   - HMAC prefix (8 bytes of HMAC-SHA256 truncated) prevents forgery:
//!     an attacker cannot make the gateway `receive_bounce` on a
//!     message_id they fabricated.
//!   - base32 (RFC 4648 no-padding) keeps the local-part legal for SMTP.

use hmac::{Hmac, Mac};
use sha2::Sha256;

use crate::error::{GatewayError, GatewayResult};

type HmacSha256 = Hmac<Sha256>;

/// Number of bytes from the HMAC output to prepend. 8 bytes = 64 bits, which
/// is enough deterrent against forgery for VERP (attacker gets no oracle —
/// incorrect HMACs are dropped without feedback).
pub const HMAC_LEN: usize = 8;

pub struct VerpCodec {
    hmac_key: Vec<u8>,
    prefix: String,
    domain: String,
}

impl VerpCodec {
    pub fn new(hmac_key: Vec<u8>, prefix: String, domain: String) -> Self {
        Self {
            hmac_key,
            prefix,
            domain,
        }
    }

    /// Produce the envelope-from for an outbound message.
    /// Example: `bounce+A1B2C3D4E5F6G7H8MFRGG43FMFRA====@mycelix.net`
    pub fn encode(&self, msg_id: &[u8]) -> String {
        let mut mac =
            HmacSha256::new_from_slice(&self.hmac_key).expect("HMAC key length is flexible");
        mac.update(msg_id);
        let tag = mac.finalize().into_bytes();
        let tag_hex = hex_encode_lower(&tag[..HMAC_LEN]);

        let msg_b32 = base32::encode(base32::Alphabet::Rfc4648 { padding: false }, msg_id);

        format!("{}+{}{}@{}", self.prefix, tag_hex, msg_b32, self.domain)
    }

    /// Decode + verify a VERP envelope address back to a `msg_id`.
    ///
    /// Returns Err on anything unexpected — wrong domain, missing '+',
    /// bad HMAC, bad base32. The caller (bounce handler) should log +
    /// drop the DSN in that case.
    pub fn decode(&self, envelope_address: &str) -> GatewayResult<Vec<u8>> {
        let (local, domain) = envelope_address
            .rsplit_once('@')
            .ok_or_else(|| GatewayError::Verp("no @ in address".into()))?;
        if !domain.eq_ignore_ascii_case(&self.domain) {
            return Err(GatewayError::Verp(format!(
                "wrong domain: got {}, expected {}",
                domain, self.domain
            )));
        }
        let (prefix, tail) = local
            .split_once('+')
            .ok_or_else(|| GatewayError::Verp("no + in local part".into()))?;
        if prefix != self.prefix {
            return Err(GatewayError::Verp(format!(
                "wrong prefix: got {}, expected {}",
                prefix, self.prefix
            )));
        }

        let hmac_hex_len = HMAC_LEN * 2;
        if tail.len() < hmac_hex_len {
            return Err(GatewayError::Verp("tail too short for HMAC".into()));
        }
        let (tag_hex, msg_b32) = tail.split_at(hmac_hex_len);

        let msg_id = base32::decode(base32::Alphabet::Rfc4648 { padding: false }, msg_b32)
            .ok_or_else(|| GatewayError::Verp("base32 decode failed".into()))?;

        let mut mac =
            HmacSha256::new_from_slice(&self.hmac_key).expect("HMAC key length is flexible");
        mac.update(&msg_id);
        let expected_tag = mac.finalize().into_bytes();
        let expected_hex = hex_encode_lower(&expected_tag[..HMAC_LEN]);

        // Constant-time-ish compare. HMAC tags leak no info on mismatch,
        // but we check length equality first to avoid panicking.
        if tag_hex.len() != expected_hex.len()
            || !constant_time_eq(tag_hex.as_bytes(), expected_hex.as_bytes())
        {
            return Err(GatewayError::Verp(
                "HMAC mismatch (possible forgery)".into(),
            ));
        }

        Ok(msg_id)
    }
}

fn hex_encode_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut acc = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        acc |= x ^ y;
    }
    acc == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codec() -> VerpCodec {
        VerpCodec::new(
            b"test-secret-32-bytes-of-high-entropy".to_vec(),
            "bounce".into(),
            "mycelix.net".into(),
        )
    }

    #[test]
    fn roundtrip() {
        let codec = codec();
        let msg_id = b"phase5-msg-001";
        let addr = codec.encode(msg_id);
        assert!(addr.ends_with("@mycelix.net"));
        assert!(addr.starts_with("bounce+"));
        let decoded = codec.decode(&addr).unwrap();
        assert_eq!(decoded, msg_id);
    }

    #[test]
    fn forged_hmac_rejected() {
        let codec = codec();
        let addr = "bounce+deadbeefdeadbeefMFRGG43FMFRA@mycelix.net";
        assert!(codec.decode(addr).is_err());
    }

    #[test]
    fn wrong_domain_rejected() {
        let codec = codec();
        let msg_id = b"x";
        let addr = codec
            .encode(msg_id)
            .replace("mycelix.net", "attacker.example");
        assert!(codec.decode(&addr).is_err());
    }

    #[test]
    fn wrong_prefix_rejected() {
        let codec = codec();
        let msg_id = b"x";
        let addr = codec.encode(msg_id).replace("bounce+", "spoofed+");
        assert!(codec.decode(&addr).is_err());
    }

    #[test]
    fn malformed_addresses_rejected() {
        let codec = codec();
        assert!(codec.decode("no-at-sign").is_err());
        assert!(codec.decode("no-plus@mycelix.net").is_err());
        assert!(codec.decode("bounce+short@mycelix.net").is_err());
    }
}
