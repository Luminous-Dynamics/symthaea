// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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

pub mod ceremony;
pub mod commitment;
pub mod dealer;
pub mod encrypted_deal;
pub mod error;
pub mod hash_commitment;
pub mod participant;
pub mod polynomial;
#[cfg(feature = "ml-kem-768")]
pub mod pq_kem;
#[cfg(feature = "ml-dsa-65")]
pub mod pq_sig;
pub mod refresh;
pub mod scalar;
pub mod share;
pub mod violation;

pub use ceremony::{CeremonyPhase, DkgCeremony, DkgConfig};
pub use commitment::{Commitment, CommitmentSet};
pub use dealer::Dealer;
pub use encrypted_deal::{EncryptResult, EncryptedDeal, EncryptedSharePayload};
pub use error::{DkgError, DkgResult};
pub use hash_commitment::{
    CommitmentSalt, CommitmentScheme, HashCommitment, HashCommitmentSet, HashReveal,
};
pub use participant::{Participant, ParticipantId};
pub use polynomial::Polynomial;
pub use refresh::{EpochShare, RefreshDeal, RefreshRound};
pub use scalar::Scalar;
pub use share::{Share, ShareSet};
pub use violation::{Violation, ViolationSeverity, ViolationTracker, ViolationType};

/// Re-export curve types for convenience
pub use k256::{AffinePoint, ProjectivePoint};

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
            ceremony
                .add_participant(ParticipantId(i as u32), 0)
                .unwrap();
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
            ceremony
                .submit_deal(ParticipantId((i + 1) as u32), deal.clone(), 0)
                .unwrap();
        }

        // Finalize
        let result = ceremony.finalize();
        assert!(result.is_ok());
    }
}
