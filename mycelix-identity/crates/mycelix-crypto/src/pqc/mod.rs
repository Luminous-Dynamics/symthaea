//! Post-quantum cryptographic implementations (native feature only).
//!
//! Each sub-module implements the `Signer`/`Verifier` traits for a specific
//! PQC algorithm family. These are backed by the `pqcrypto-*` crate family which
//! wraps the reference C implementations from the NIST PQC standardization process.

pub mod dilithium;
pub mod ed25519_native;
pub mod encryption;
pub mod hybrid;
pub mod ml_kem;
pub mod sphincs;
