//! Crypto trait stubs.

use crate::{CryptoError, TaggedPublicKey};

/// Key encapsulation mechanism trait.
pub trait KeyEncapsulator {
    /// Generate a new keypair.
    fn generate() -> Self;
    /// Get the public key.
    fn public_key(&self) -> TaggedPublicKey;
    /// Encapsulate a shared secret to the given public key.
    fn encapsulate(&self, pk: &TaggedPublicKey) -> Result<(Vec<u8>, Vec<u8>), CryptoError>;
    /// Decapsulate a ciphertext to recover the shared secret.
    fn decapsulate(&self, ciphertext: &[u8]) -> Result<Vec<u8>, CryptoError>;
}
