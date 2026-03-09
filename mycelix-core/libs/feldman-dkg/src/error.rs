//! Error types for DKG operations

use thiserror::Error;

/// Result type for DKG operations
pub type DkgResult<T> = Result<T, DkgError>;

/// Errors that can occur during DKG
#[derive(Debug, Error)]
pub enum DkgError {
    /// Invalid threshold configuration
    #[error("Invalid threshold: t={threshold} must be > 0 and <= n={participants}")]
    InvalidThreshold {
        threshold: usize,
        participants: usize,
    },

    /// Invalid participant count
    #[error("Invalid participant count: must be >= 2, got {0}")]
    InvalidParticipantCount(usize),

    /// Invalid participant ID
    #[error("Invalid participant ID: {0} (must be 1..=n)")]
    InvalidParticipantId(u32),

    /// Participant already exists
    #[error("Participant {0} already registered")]
    ParticipantExists(u32),

    /// Participant not found
    #[error("Participant {0} not found")]
    ParticipantNotFound(u32),

    /// Share verification failed
    #[error("Share verification failed for participant {participant} from dealer {dealer}")]
    ShareVerificationFailed { participant: u32, dealer: u32 },

    /// Commitment verification failed
    #[error("Commitment verification failed")]
    CommitmentVerificationFailed,

    /// Not enough participants
    #[error("Not enough participants: need {required}, have {actual}")]
    NotEnoughParticipants { required: usize, actual: usize },

    /// Missing deal from participant
    #[error("Missing deal from participant {0}")]
    MissingDeal(u32),

    /// Ceremony in wrong phase
    #[error("Ceremony in wrong phase: expected {expected}, actual {actual}")]
    WrongPhase { expected: String, actual: String },

    /// Invalid share index
    #[error("Invalid share index: {0}")]
    InvalidShareIndex(usize),

    /// Duplicate share
    #[error("Duplicate share from dealer {0}")]
    DuplicateShare(u32),

    /// Lagrange interpolation error
    #[error("Lagrange interpolation failed: {0}")]
    InterpolationError(String),

    /// Cryptographic error
    #[error("Cryptographic error: {0}")]
    CryptoError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Encryption error
    #[error("Encryption error: {0}")]
    EncryptionError(String),

    /// Decryption error
    #[error("Decryption error: {0}")]
    DecryptionError(String),

    /// Hash commitment mismatch during reveal
    #[error("Hash commitment mismatch for dealer {dealer}: expected {expected}, got {actual}")]
    HashCommitmentMismatch {
        dealer: u32,
        expected: String,
        actual: String,
    },

    /// Missing commitment in commit-reveal protocol
    #[error("Missing hash commitment from dealer {0}")]
    MissingHashCommitment(u32),

    /// Refresh protocol error
    #[error("Refresh error: {0}")]
    RefreshError(String),

    /// Violation report
    #[error("Protocol violation by participant {participant}: {violation}")]
    ProtocolViolation { participant: u32, violation: String },
}
