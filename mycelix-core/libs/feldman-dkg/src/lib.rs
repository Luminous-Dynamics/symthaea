//! Feldman's Verifiable Secret Sharing for Distributed Key Generation
//!
//! This library implements Feldman's VSS scheme for threshold cryptography.
//! It enables a group of n participants to jointly generate a shared secret
//! key where any t participants can reconstruct it, but fewer than t cannot.
//!
//! ## Security Properties
//!
//! - **Verifiable**: Each participant can verify their share is consistent
//! - **Secure**: Requires t-of-n participants to reconstruct
//! - **Information-theoretic**: Security based on discrete log assumption
//!
//! ## Usage
//!
//! ```ignore
//! use feldman_dkg::{DkgCeremony, DkgConfig};
//!
//! // Configure a 3-of-5 threshold scheme
//! let config = DkgConfig::new(3, 5)?;
//!
//! // Create ceremony coordinator (now_secs = current unix time)
//! let mut ceremony = DkgCeremony::new(config, now_secs);
//!
//! // Each participant generates and distributes shares
//! // ...
//!
//! // Combine to get the shared public key
//! let public_key = ceremony.finalize()?;
//! ```

pub mod error;
pub mod scalar;
pub mod polynomial;
pub mod commitment;
pub mod share;
pub mod dealer;
pub mod participant;
pub mod ceremony;
pub mod encrypted_deal;
pub mod hash_commitment;
#[cfg(feature = "ml-kem-768")]
pub mod pq_kem;
#[cfg(feature = "ml-dsa-65")]
pub mod pq_sig;
pub mod refresh;
pub mod violation;

pub use error::{DkgError, DkgResult};
pub use scalar::Scalar;
pub use polynomial::Polynomial;
pub use commitment::{Commitment, CommitmentSet};
pub use share::{Share, ShareSet};
pub use dealer::Dealer;
pub use participant::{Participant, ParticipantId};
pub use ceremony::{DkgCeremony, DkgConfig, CeremonyPhase};
pub use encrypted_deal::{EncryptedDeal, EncryptedSharePayload, EncryptResult};
pub use hash_commitment::{HashCommitment, HashCommitmentSet, HashReveal, CommitmentSalt, CommitmentScheme};
pub use refresh::{RefreshDeal, RefreshRound, EpochShare};
pub use violation::{Violation, ViolationTracker, ViolationType, ViolationSeverity};

/// Re-export curve types for convenience
pub use k256::{ProjectivePoint, AffinePoint};

/// The generator point G for secp256k1
pub fn generator() -> ProjectivePoint {
    ProjectivePoint::GENERATOR
}

/// Default threshold for a 5-participant ceremony (3-of-5)
pub const DEFAULT_THRESHOLD: usize = 3;
/// Default number of participants
pub const DEFAULT_PARTICIPANTS: usize = 5;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn test_basic_dkg_ceremony() {
        // Create a 2-of-3 ceremony
        let config = DkgConfig::new(2, 3).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        // Add participants
        for i in 1..=3 {
            ceremony.add_participant(ParticipantId(i as u32), 0).unwrap();
        }

        // Each participant deals
        let deals: Vec<_> = (1..=3)
            .map(|i| {
                let dealer = Dealer::new(ParticipantId(i as u32), 2, 3, &mut OsRng).unwrap();
                dealer.generate_deal()
            })
            .collect();

        // Submit deals to ceremony
        for (i, deal) in deals.iter().enumerate() {
            ceremony.submit_deal(ParticipantId((i + 1) as u32), deal.clone(), 0).unwrap();
        }

        // Finalize
        let result = ceremony.finalize();
        assert!(result.is_ok());
    }
}
