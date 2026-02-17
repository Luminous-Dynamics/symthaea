//! Privacy-preserving techniques for federated learning.
//!
//! This module provides:
//! - AES-GCM gradient encryption for confidentiality
//! - Differential privacy (Gaussian/Laplace mechanisms)
//! - Privacy budget accounting
//! - GDPR cryptographic erasure (right to be forgotten)
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use fl_aggregator::privacy::{encrypt_gradient, decrypt_gradient, generate_key};
//! use fl_aggregator::privacy::{DifferentialPrivacy, DPConfig};
//!
//! // Encryption
//! let key = generate_key();
//! let ciphertext = encrypt_gradient(&key, &gradient_bytes)?;
//! let plaintext = decrypt_gradient(&key, &ciphertext)?;
//!
//! // Differential Privacy
//! let config = DPConfig::default().with_epsilon(1.0).with_max_grad_norm(1.0);
//! let dp = DifferentialPrivacy::new(config);
//! let (private_gradient, metadata) = dp.privatize_gradient(&gradient)?;
//! ```
//!
//! ## GDPR Erasure (P2-05)
//!
//! ```rust,ignore
//! use fl_aggregator::privacy::erasure::ErasureKeyManager;
//!
//! let manager = ErasureKeyManager::new();
//!
//! // Encrypt user data with user-specific key
//! let encrypted = manager.encrypt_for_agent("user-123", b"data")?;
//!
//! // User requests erasure (GDPR Art. 17)
//! let receipt = manager.request_erasure("user-123")?;
//!
//! // Data is now permanently unrecoverable
//! ```

pub mod encryption;
pub mod differential;
pub mod erasure;

pub use encryption::{
    decrypt_gradient, encrypt_gradient, generate_key, EncryptedGradient, EncryptionError,
};
pub use differential::{
    clip_gradient, add_noise, compute_privacy_budget,
    DPConfig, DifferentialPrivacy, GaussianMechanism, LaplaceMechanism,
    NoiseType, PrivacyAccountant, PrivacyMetadata,
};
pub use erasure::{ErasureKeyManager, ErasureReceipt, ErasureResult, ErasureStats};
