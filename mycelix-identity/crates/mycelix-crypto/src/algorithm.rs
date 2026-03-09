//! Algorithm identifiers and metadata for crypto-agile operations.
//!
//! Each `AlgorithmId` variant carries methods that return the W3C DID verification
//! method type string, the Data Integrity cryptosuite name, expected key/signature
//! sizes, and the 2-byte multicodec prefix used in multibase-encoded keys.

use serde::{Deserialize, Serialize};

/// Identifies a cryptographic algorithm used for signing, verification, or key agreement.
///
/// The `u16` discriminant matches the multicodec code-point where one exists. For
/// algorithms without an assigned multicodec code we use a private-use range (0xF0xx).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u16)]
pub enum AlgorithmId {
    // -- Signatures --
    /// Ed25519 (RFC 8032). Multicodec 0xed.
    Ed25519 = 0xed01,
    /// FIPS 204 ML-DSA-65 (Dilithium3). Multicodec not yet assigned.
    MlDsa65 = 0xF001,
    /// FIPS 204 ML-DSA-87 (Dilithium5).
    MlDsa87 = 0xF002,
    /// FIPS 205 SLH-DSA-SHA2-128s (SPHINCS+).
    SlhDsaSha2_128s = 0xF003,
    /// FIPS 205 SLH-DSA-SHAKE-128s (SPHINCS+).
    SlhDsaShake128s = 0xF004,
    /// Hybrid: Ed25519 || ML-DSA-65 (concatenated signatures).
    HybridEd25519MlDsa65 = 0xF010,

    // -- Key Encapsulation --
    /// FIPS 203 ML-KEM-768.
    MlKem768 = 0xF020,
    /// FIPS 203 ML-KEM-1024.
    MlKem1024 = 0xF021,

    // -- Symmetric (for encrypted envelopes) --
    /// XChaCha20-Poly1305 AEAD.
    XChaCha20Poly1305 = 0xF030,
}

impl AlgorithmId {
    /// Two-byte multicodec varint prefix for multibase key encoding.
    ///
    /// Ed25519 uses the canonical `[0xed, 0x01]`; PQC algorithms use the
    /// private-use range until IANA assigns official code-points.
    pub const fn multicodec_prefix(&self) -> [u8; 2] {
        match self {
            Self::Ed25519 => [0xed, 0x01],
            Self::MlDsa65 => [0xF0, 0x01],
            Self::MlDsa87 => [0xF0, 0x02],
            Self::SlhDsaSha2_128s => [0xF0, 0x03],
            Self::SlhDsaShake128s => [0xF0, 0x04],
            Self::HybridEd25519MlDsa65 => [0xF0, 0x10],
            Self::MlKem768 => [0xF0, 0x20],
            Self::MlKem1024 => [0xF0, 0x21],
            Self::XChaCha20Poly1305 => [0xF0, 0x30],
        }
    }

    /// W3C DID Core `type` value for a verification method using this algorithm.
    ///
    /// Reference: <https://www.w3.org/TR/did-spec-registries/#verification-method-types>
    pub const fn did_verification_method_type(&self) -> &'static str {
        match self {
            Self::Ed25519 => "Ed25519VerificationKey2020",
            Self::MlDsa65 => "MlDsa65VerificationKey2024",
            Self::MlDsa87 => "MlDsa87VerificationKey2024",
            Self::SlhDsaSha2_128s => "SlhDsaSha2128sVerificationKey2024",
            Self::SlhDsaShake128s => "SlhDsaShake128sVerificationKey2024",
            Self::HybridEd25519MlDsa65 => "HybridEd25519MlDsa65VerificationKey2024",
            Self::MlKem768 => "MlKem768KeyAgreementKey2024",
            Self::MlKem1024 => "MlKem1024KeyAgreementKey2024",
            Self::XChaCha20Poly1305 => "XChaCha20Poly1305",
        }
    }

    /// W3C Data Integrity `cryptosuite` value.
    ///
    /// Reference: <https://www.w3.org/TR/vc-di-eddsa/>
    pub const fn cryptosuite(&self) -> &'static str {
        match self {
            Self::Ed25519 => "eddsa-rdfc-2022",
            Self::MlDsa65 => "mldsa65-rdfc-2024",
            Self::MlDsa87 => "mldsa87-rdfc-2024",
            Self::SlhDsaSha2_128s => "slhdsa-sha2-128s-rdfc-2024",
            Self::SlhDsaShake128s => "slhdsa-shake-128s-rdfc-2024",
            Self::HybridEd25519MlDsa65 => "hybrid-eddsa-mldsa65-rdfc-2024",
            Self::MlKem768 | Self::MlKem1024 | Self::XChaCha20Poly1305 => "",
        }
    }

    /// Expected raw public key size in bytes (without multicodec prefix).
    pub const fn public_key_size(&self) -> usize {
        match self {
            Self::Ed25519 => 32,
            Self::MlDsa65 => 1952,
            Self::MlDsa87 => 2592,
            Self::SlhDsaSha2_128s | Self::SlhDsaShake128s => 32,
            Self::HybridEd25519MlDsa65 => 32 + 1952, // Ed25519 pk || ML-DSA-65 pk
            Self::MlKem768 => 1184,
            Self::MlKem1024 => 1568,
            Self::XChaCha20Poly1305 => 0, // symmetric, no public key
        }
    }

    /// Expected raw signature size in bytes (per FIPS 204/205).
    pub const fn signature_size(&self) -> usize {
        match self {
            Self::Ed25519 => 64,
            Self::MlDsa65 => 3309, // FIPS 204 ML-DSA-65
            Self::MlDsa87 => 4627, // FIPS 204 ML-DSA-87
            Self::SlhDsaSha2_128s | Self::SlhDsaShake128s => 7856,
            Self::HybridEd25519MlDsa65 => 64 + 3309, // Ed25519 sig || ML-DSA-65 sig
            Self::MlKem768 | Self::MlKem1024 | Self::XChaCha20Poly1305 => 0,
        }
    }

    /// Whether this is a signature algorithm (vs key-agreement or symmetric).
    pub const fn is_signature_algorithm(&self) -> bool {
        matches!(
            self,
            Self::Ed25519
                | Self::MlDsa65
                | Self::MlDsa87
                | Self::SlhDsaSha2_128s
                | Self::SlhDsaShake128s
                | Self::HybridEd25519MlDsa65
        )
    }

    /// Try to identify algorithm from a 2-byte multicodec prefix.
    pub fn from_multicodec_prefix(prefix: [u8; 2]) -> Option<Self> {
        match prefix {
            [0xed, 0x01] => Some(Self::Ed25519),
            [0xF0, 0x01] => Some(Self::MlDsa65),
            [0xF0, 0x02] => Some(Self::MlDsa87),
            [0xF0, 0x03] => Some(Self::SlhDsaSha2_128s),
            [0xF0, 0x04] => Some(Self::SlhDsaShake128s),
            [0xF0, 0x10] => Some(Self::HybridEd25519MlDsa65),
            [0xF0, 0x20] => Some(Self::MlKem768),
            [0xF0, 0x21] => Some(Self::MlKem1024),
            [0xF0, 0x30] => Some(Self::XChaCha20Poly1305),
            _ => None,
        }
    }

    /// Numeric discriminant for serialization (e.g. in integrity entry fields).
    pub const fn as_u16(&self) -> u16 {
        *self as u16
    }

    /// Reconstruct from the `u16` discriminant. Returns `None` for unknown values.
    pub fn from_u16(v: u16) -> Option<Self> {
        match v {
            0xed01 => Some(Self::Ed25519),
            0xF001 => Some(Self::MlDsa65),
            0xF002 => Some(Self::MlDsa87),
            0xF003 => Some(Self::SlhDsaSha2_128s),
            0xF004 => Some(Self::SlhDsaShake128s),
            0xF010 => Some(Self::HybridEd25519MlDsa65),
            0xF020 => Some(Self::MlKem768),
            0xF021 => Some(Self::MlKem1024),
            0xF030 => Some(Self::XChaCha20Poly1305),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ed25519_constants() {
        let alg = AlgorithmId::Ed25519;
        assert_eq!(alg.multicodec_prefix(), [0xed, 0x01]);
        assert_eq!(
            alg.did_verification_method_type(),
            "Ed25519VerificationKey2020"
        );
        assert_eq!(alg.cryptosuite(), "eddsa-rdfc-2022");
        assert_eq!(alg.public_key_size(), 32);
        assert_eq!(alg.signature_size(), 64);
        assert!(alg.is_signature_algorithm());
    }

    #[test]
    fn hybrid_constants() {
        let alg = AlgorithmId::HybridEd25519MlDsa65;
        assert_eq!(alg.public_key_size(), 32 + 1952);
        assert_eq!(alg.signature_size(), 64 + 3309);
        assert!(alg.is_signature_algorithm());
    }

    #[test]
    fn multicodec_roundtrip() {
        for alg in [
            AlgorithmId::Ed25519,
            AlgorithmId::MlDsa65,
            AlgorithmId::MlDsa87,
            AlgorithmId::HybridEd25519MlDsa65,
            AlgorithmId::MlKem768,
        ] {
            let prefix = alg.multicodec_prefix();
            assert_eq!(AlgorithmId::from_multicodec_prefix(prefix), Some(alg));
        }
    }

    #[test]
    fn u16_roundtrip() {
        for alg in [
            AlgorithmId::Ed25519,
            AlgorithmId::MlDsa65,
            AlgorithmId::HybridEd25519MlDsa65,
            AlgorithmId::XChaCha20Poly1305,
        ] {
            assert_eq!(AlgorithmId::from_u16(alg.as_u16()), Some(alg));
        }
    }

    #[test]
    fn unknown_u16_returns_none() {
        assert_eq!(AlgorithmId::from_u16(0x0000), None);
        assert_eq!(AlgorithmId::from_u16(0xFFFF), None);
    }
}
