//! Post-quantum cryptography stubs.

pub mod ml_kem {
    use crate::traits::KeyEncapsulator;
    use crate::{AlgorithmId, CryptoError, TaggedPublicKey};

    /// Stub ML-KEM-768 keypair.
    ///
    /// In the real crate this wraps pqcrypto-kyber. Here it provides
    /// deterministic stub behaviour for compilation.
    #[derive(Debug, Clone)]
    pub struct MlKem768KeyPair {
        /// Stub public key bytes.
        pk: Vec<u8>,
        /// Stub secret key bytes.
        sk: Vec<u8>,
    }

    impl KeyEncapsulator for MlKem768KeyPair {
        fn generate() -> Self {
            Self {
                pk: vec![0u8; 1184], // ML-KEM-768 pk size
                sk: vec![0u8; 2400], // ML-KEM-768 sk size
            }
        }

        fn public_key(&self) -> TaggedPublicKey {
            TaggedPublicKey {
                algorithm: AlgorithmId::MlKem768,
                key_bytes: self.pk.clone(),
            }
        }

        fn encapsulate(&self, _pk: &TaggedPublicKey) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
            // Stub: return deterministic ciphertext + shared secret
            Ok((vec![0u8; 1088], vec![0u8; 32]))
        }

        fn decapsulate(&self, ciphertext: &[u8]) -> Result<Vec<u8>, CryptoError> {
            if ciphertext.len() != 1088 {
                return Err(CryptoError(format!(
                    "invalid ciphertext length: {} (expected 1088)",
                    ciphertext.len()
                )));
            }
            // Stub: return deterministic shared secret
            Ok(vec![0u8; 32])
        }
    }

    impl MlKem768KeyPair {
        /// Generate a new keypair (convenience, delegates to trait).
        pub fn generate() -> Self {
            <Self as KeyEncapsulator>::generate()
        }

        /// Get the public key (convenience, delegates to trait).
        pub fn public_key(&self) -> TaggedPublicKey {
            <Self as KeyEncapsulator>::public_key(self)
        }

        /// Encapsulate (convenience, delegates to trait).
        pub fn encapsulate(&self, pk: &TaggedPublicKey) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
            <Self as KeyEncapsulator>::encapsulate(self, pk)
        }

        /// Decapsulate (convenience, delegates to trait).
        pub fn decapsulate(&self, ciphertext: &[u8]) -> Result<Vec<u8>, CryptoError> {
            <Self as KeyEncapsulator>::decapsulate(self, ciphertext)
        }
    }
}
