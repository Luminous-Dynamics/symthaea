//! Algorithm-tagged cryptographic envelopes.
//!
//! These types carry opaque key/signature bytes alongside their [`AlgorithmId`],
//! enabling WASM zomes to store, validate lengths, and round-trip crypto material
//! without performing actual cryptographic operations (which happen off-chain).

use crate::algorithm::AlgorithmId;
use crate::error::CryptoError;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// TaggedPublicKey
// ---------------------------------------------------------------------------

/// A public key tagged with its algorithm, supporting multibase encoding.
///
/// Multibase format: `z` prefix + base58btc( multicodec_prefix || raw_key_bytes )
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaggedPublicKey {
    pub algorithm: AlgorithmId,
    pub key_bytes: Vec<u8>,
}

impl TaggedPublicKey {
    /// Create a new tagged public key, validating length.
    pub fn new(algorithm: AlgorithmId, key_bytes: Vec<u8>) -> Result<Self, CryptoError> {
        let expected = algorithm.public_key_size();
        if expected > 0 && key_bytes.len() != expected {
            return Err(CryptoError::InvalidKeyLength {
                algorithm: algorithm.did_verification_method_type(),
                expected,
                actual: key_bytes.len(),
            });
        }
        Ok(Self {
            algorithm,
            key_bytes,
        })
    }

    /// Encode as multibase string: `z` + base58btc(multicodec_prefix || key_bytes).
    pub fn to_multibase(&self) -> String {
        let prefix = self.algorithm.multicodec_prefix();
        let mut buf = Vec::with_capacity(2 + self.key_bytes.len());
        buf.extend_from_slice(&prefix);
        buf.extend_from_slice(&self.key_bytes);
        let encoded = bs58::encode(&buf)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .into_string();
        format!("z{}", encoded)
    }

    /// Decode from a multibase string.
    ///
    /// Accepts:
    /// - 2-byte multicodec prefix + key bytes (algorithm detected from prefix)
    /// - Raw 32-byte Ed25519 key (legacy, no prefix — assumes Ed25519)
    pub fn from_multibase(s: &str) -> Result<Self, CryptoError> {
        if s.is_empty() {
            return Err(CryptoError::InvalidMultibase("empty string".into()));
        }
        if !s.starts_with('z') {
            return Err(CryptoError::InvalidMultibase(format!(
                "expected 'z' (base58btc) prefix, got '{}'",
                s.chars().next().unwrap_or('?')
            )));
        }
        let encoded = &s[1..];
        if encoded.is_empty() {
            return Err(CryptoError::InvalidMultibase("no data after prefix".into()));
        }
        let decoded = bs58::decode(encoded)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .into_vec()
            .map_err(|e| CryptoError::Base58Decode(e.to_string()))?;

        // Try to detect algorithm from multicodec prefix
        if decoded.len() >= 2 {
            let prefix = [decoded[0], decoded[1]];
            if let Some(algorithm) = AlgorithmId::from_multicodec_prefix(prefix) {
                let key_bytes = decoded[2..].to_vec();
                let expected = algorithm.public_key_size();
                if expected > 0 && key_bytes.len() != expected {
                    return Err(CryptoError::InvalidKeyLength {
                        algorithm: algorithm.did_verification_method_type(),
                        expected,
                        actual: key_bytes.len(),
                    });
                }
                return Ok(Self {
                    algorithm,
                    key_bytes,
                });
            }
        }

        // Legacy fallback: raw 32-byte Ed25519 key without multicodec prefix
        if decoded.len() == 32 {
            return Ok(Self {
                algorithm: AlgorithmId::Ed25519,
                key_bytes: decoded,
            });
        }

        Err(CryptoError::InvalidMultibase(format!(
            "unrecognized multicodec prefix or invalid length ({})",
            decoded.len()
        )))
    }
}

// ---------------------------------------------------------------------------
// TaggedSignature
// ---------------------------------------------------------------------------

/// A signature tagged with its algorithm, supporting multibase encoding.
///
/// For hybrid signatures the raw bytes are `ed25519_sig (64) || mldsa65_sig (3309)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaggedSignature {
    pub algorithm: AlgorithmId,
    pub signature_bytes: Vec<u8>,
}

impl TaggedSignature {
    /// Create a new tagged signature, validating length.
    pub fn new(algorithm: AlgorithmId, signature_bytes: Vec<u8>) -> Result<Self, CryptoError> {
        let expected = algorithm.signature_size();
        if expected > 0 && signature_bytes.len() != expected {
            return Err(CryptoError::InvalidSignatureLength {
                algorithm: algorithm.did_verification_method_type(),
                expected,
                actual: signature_bytes.len(),
            });
        }
        Ok(Self {
            algorithm,
            signature_bytes,
        })
    }

    /// Encode as multibase: `z` + base58btc(multicodec_prefix || signature_bytes).
    pub fn to_multibase(&self) -> String {
        let prefix = self.algorithm.multicodec_prefix();
        let mut buf = Vec::with_capacity(2 + self.signature_bytes.len());
        buf.extend_from_slice(&prefix);
        buf.extend_from_slice(&self.signature_bytes);
        let encoded = bs58::encode(&buf)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .into_string();
        format!("z{}", encoded)
    }

    /// Decode from multibase string.
    ///
    /// Accepts:
    /// - 2-byte multicodec prefix + signature bytes (algorithm from prefix)
    /// - Raw 64-byte Ed25519 signature (legacy, no prefix)
    pub fn from_multibase(s: &str) -> Result<Self, CryptoError> {
        if s.is_empty() {
            return Err(CryptoError::InvalidMultibase("empty string".into()));
        }
        if !s.starts_with('z') {
            return Err(CryptoError::InvalidMultibase(format!(
                "expected 'z' prefix, got '{}'",
                s.chars().next().unwrap_or('?')
            )));
        }
        let encoded = &s[1..];
        if encoded.is_empty() {
            return Err(CryptoError::InvalidMultibase("no data after prefix".into()));
        }
        let decoded = bs58::decode(encoded)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .into_vec()
            .map_err(|e| CryptoError::Base58Decode(e.to_string()))?;

        // Try multicodec prefix
        if decoded.len() >= 2 {
            let prefix = [decoded[0], decoded[1]];
            if let Some(algorithm) = AlgorithmId::from_multicodec_prefix(prefix) {
                let sig_bytes = decoded[2..].to_vec();
                let expected = algorithm.signature_size();
                if expected > 0 && sig_bytes.len() != expected {
                    return Err(CryptoError::InvalidSignatureLength {
                        algorithm: algorithm.did_verification_method_type(),
                        expected,
                        actual: sig_bytes.len(),
                    });
                }
                return Ok(Self {
                    algorithm,
                    signature_bytes: sig_bytes,
                });
            }
        }

        // Legacy fallback: raw 64-byte Ed25519 signature
        if decoded.len() == 64 {
            return Ok(Self {
                algorithm: AlgorithmId::Ed25519,
                signature_bytes: decoded,
            });
        }

        Err(CryptoError::InvalidMultibase(format!(
            "unrecognized prefix or invalid signature length ({})",
            decoded.len()
        )))
    }

    /// Extract the Ed25519 component from a hybrid or pure Ed25519 signature.
    ///
    /// Returns the first 64 bytes for hybrid, or the full bytes for Ed25519.
    /// Returns `None` for non-Ed25519 algorithms.
    pub fn ed25519_component(&self) -> Option<&[u8]> {
        match self.algorithm {
            AlgorithmId::Ed25519 => Some(&self.signature_bytes),
            AlgorithmId::HybridEd25519MlDsa65 => {
                if self.signature_bytes.len() >= 64 {
                    Some(&self.signature_bytes[..64])
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Extract the PQC component from a hybrid signature.
    ///
    /// Returns bytes after the Ed25519 component for hybrid signatures.
    /// Returns the full bytes for pure PQC signatures.
    /// Returns `None` for Ed25519.
    pub fn pqc_component(&self) -> Option<&[u8]> {
        match self.algorithm {
            AlgorithmId::HybridEd25519MlDsa65 => {
                if self.signature_bytes.len() > 64 {
                    Some(&self.signature_bytes[64..])
                } else {
                    None
                }
            }
            AlgorithmId::MlDsa65
            | AlgorithmId::MlDsa87
            | AlgorithmId::SlhDsaSha2_128s
            | AlgorithmId::SlhDsaShake128s => Some(&self.signature_bytes),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// EncryptedEnvelope
// ---------------------------------------------------------------------------

/// An encrypted payload using hybrid KEM + AEAD.
///
/// The sender encapsulates a shared secret via the recipient's KEM public key,
/// then encrypts the plaintext with XChaCha20-Poly1305 keyed by that shared secret.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncryptedEnvelope {
    /// KEM algorithm used to encapsulate the symmetric key.
    pub kem_algorithm: AlgorithmId,
    /// The KEM ciphertext (encapsulated key).
    pub encapsulated_key: Vec<u8>,
    /// 24-byte nonce for XChaCha20-Poly1305.
    pub nonce: Vec<u8>,
    /// AEAD ciphertext (plaintext + 16-byte Poly1305 tag).
    pub ciphertext: Vec<u8>,
    /// Key ID of the recipient's KEM public key (e.g. DID URL fragment).
    pub recipient_key_id: String,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tagged_key_multibase_roundtrip() {
        let key = TaggedPublicKey::new(AlgorithmId::Ed25519, vec![0x42; 32]).unwrap();
        let mb = key.to_multibase();
        assert!(mb.starts_with('z'));
        let decoded = TaggedPublicKey::from_multibase(&mb).unwrap();
        assert_eq!(decoded.algorithm, AlgorithmId::Ed25519);
        assert_eq!(decoded.key_bytes, vec![0x42; 32]);
    }

    #[test]
    fn tagged_key_legacy_raw_32_bytes() {
        // Simulate legacy encoding: z + base58(raw 32-byte key, no multicodec prefix)
        let raw = vec![0x42; 32];
        let encoded = bs58::encode(&raw)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .into_string();
        let multibase = format!("z{}", encoded);

        let key = TaggedPublicKey::from_multibase(&multibase).unwrap();
        assert_eq!(key.algorithm, AlgorithmId::Ed25519);
        assert_eq!(key.key_bytes, raw);
    }

    #[test]
    fn tagged_key_invalid_prefix() {
        assert!(TaggedPublicKey::from_multibase("").is_err());
        assert!(TaggedPublicKey::from_multibase("m123").is_err());
        assert!(TaggedPublicKey::from_multibase("z").is_err());
    }

    #[test]
    fn tagged_key_wrong_length_rejected() {
        // Ed25519 key with multicodec prefix but wrong key length (16 instead of 32)
        let mut buf = vec![0xed, 0x01];
        buf.extend(vec![0x42; 16]);
        let encoded = bs58::encode(&buf)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .into_string();
        let multibase = format!("z{}", encoded);

        let err = TaggedPublicKey::from_multibase(&multibase).unwrap_err();
        assert!(matches!(err, CryptoError::InvalidKeyLength { .. }));
    }

    #[test]
    fn tagged_signature_multibase_roundtrip() {
        let sig = TaggedSignature::new(AlgorithmId::Ed25519, vec![0xAB; 64]).unwrap();
        let mb = sig.to_multibase();
        let decoded = TaggedSignature::from_multibase(&mb).unwrap();
        assert_eq!(decoded.algorithm, AlgorithmId::Ed25519);
        assert_eq!(decoded.signature_bytes, vec![0xAB; 64]);
    }

    #[test]
    fn tagged_signature_legacy_raw_64_bytes() {
        let raw = vec![0xAB; 64];
        let encoded = bs58::encode(&raw)
            .with_alphabet(bs58::Alphabet::BITCOIN)
            .into_string();
        let multibase = format!("z{}", encoded);

        let sig = TaggedSignature::from_multibase(&multibase).unwrap();
        assert_eq!(sig.algorithm, AlgorithmId::Ed25519);
        assert_eq!(sig.signature_bytes, raw);
    }

    #[test]
    fn hybrid_signature_components() {
        let ed_part = vec![0xAA; 64];
        let pqc_part = vec![0xBB; 3309];
        let mut combined = ed_part.clone();
        combined.extend(&pqc_part);

        let sig = TaggedSignature::new(AlgorithmId::HybridEd25519MlDsa65, combined).unwrap();
        assert_eq!(sig.ed25519_component().unwrap(), &ed_part[..]);
        assert_eq!(sig.pqc_component().unwrap(), &pqc_part[..]);
    }

    #[test]
    fn ed25519_has_no_pqc_component() {
        let sig = TaggedSignature::new(AlgorithmId::Ed25519, vec![0xAB; 64]).unwrap();
        assert!(sig.ed25519_component().is_some());
        assert!(sig.pqc_component().is_none());
    }

    #[test]
    fn invalid_signature_length_rejected() {
        let err = TaggedSignature::new(AlgorithmId::Ed25519, vec![0; 32]).unwrap_err();
        assert!(matches!(err, CryptoError::InvalidSignatureLength { .. }));
    }

    #[test]
    fn pqc_key_roundtrip() {
        // ML-DSA-65 has a 1952-byte public key
        let key = TaggedPublicKey::new(AlgorithmId::MlDsa65, vec![0x33; 1952]).unwrap();
        let mb = key.to_multibase();
        let decoded = TaggedPublicKey::from_multibase(&mb).unwrap();
        assert_eq!(decoded.algorithm, AlgorithmId::MlDsa65);
        assert_eq!(decoded.key_bytes.len(), 1952);
    }

    #[test]
    fn encrypted_envelope_json_roundtrip() {
        let envelope = EncryptedEnvelope {
            kem_algorithm: AlgorithmId::MlKem768,
            encapsulated_key: vec![0xAA; 1088],
            nonce: vec![0xBB; 24],
            ciphertext: vec![0xCC; 256],
            recipient_key_id: "#kem-1".to_string(),
        };

        // Serialize to JSON
        let json = serde_json::to_string(&envelope).unwrap();

        // Verify field names are camelCase (serde default for snake_case fields)
        assert!(
            json.contains("kem_algorithm"),
            "JSON should contain kem_algorithm field"
        );
        assert!(
            json.contains("encapsulated_key"),
            "JSON should contain encapsulated_key field"
        );
        assert!(
            json.contains("recipient_key_id"),
            "JSON should contain recipient_key_id field"
        );

        // Deserialize back
        let decoded: EncryptedEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.kem_algorithm, AlgorithmId::MlKem768);
        assert_eq!(decoded.encapsulated_key.len(), 1088);
        assert_eq!(decoded.nonce.len(), 24);
        assert_eq!(decoded.ciphertext.len(), 256);
        assert_eq!(decoded.recipient_key_id, "#kem-1");
    }

    #[test]
    fn encrypted_envelope_field_stability() {
        // Simulate receiving a JSON envelope from an external source
        // This catches field name drift if serde attributes are added/changed
        let external_json = r##"{
            "kem_algorithm": "MlKem768",
            "encapsulated_key": [170, 170, 170],
            "nonce": [187, 187, 187, 187, 187, 187, 187, 187, 187, 187, 187, 187, 187, 187, 187, 187, 187, 187, 187, 187, 187, 187, 187, 187],
            "ciphertext": [204, 204],
            "recipient_key_id": "#kem-2"
        }"##;

        let decoded: EncryptedEnvelope = serde_json::from_str(external_json).unwrap();
        assert_eq!(decoded.kem_algorithm, AlgorithmId::MlKem768);
        assert_eq!(decoded.encapsulated_key, vec![0xAA; 3]);
        assert_eq!(decoded.nonce.len(), 24);
        assert_eq!(decoded.ciphertext, vec![0xCC; 2]);
        assert_eq!(decoded.recipient_key_id, "#kem-2");
    }
}
