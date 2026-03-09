//! Dealer for generating and distributing shares

use rand_core::CryptoRngCore;
use serde::{Deserialize, Serialize};

use crate::commitment::CommitmentSet;
use crate::error::{DkgError, DkgResult};
use crate::participant::ParticipantId;
use crate::polynomial::Polynomial;
use crate::scalar::Scalar;
use crate::share::Share;

/// A deal from a dealer containing shares and commitments
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Deal {
    /// The dealer's ID
    pub dealer: ParticipantId,
    /// Commitments to the polynomial coefficients
    pub commitments: CommitmentSet,
    /// Shares for each participant (encrypted in production)
    /// Map from participant index to their share
    pub shares: Vec<Share>,
}

impl Deal {
    /// Get the share for a specific participant
    pub fn get_share(&self, participant: u32) -> Option<&Share> {
        self.shares.iter().find(|s| s.index == participant)
    }

    /// Get the number of shares
    pub fn share_count(&self) -> usize {
        self.shares.len()
    }

    /// Verify that a share is consistent with the commitments
    pub fn verify_share(&self, participant: u32) -> DkgResult<bool> {
        let share = self
            .get_share(participant)
            .ok_or(DkgError::ParticipantNotFound(participant))?;
        Ok(self.commitments.verify_share(participant, share.value()))
    }

    /// Verify all shares
    pub fn verify_all_shares(&self) -> Vec<(u32, bool)> {
        self.shares
            .iter()
            .map(|s| (s.index, self.commitments.verify_share(s.index, s.value())))
            .collect()
    }
}

/// A dealer that generates secret shares
#[derive(Debug)]
pub struct Dealer {
    /// Dealer's participant ID
    id: ParticipantId,
    /// The secret polynomial
    polynomial: Polynomial,
    /// Number of participants
    num_participants: usize,
}

impl Dealer {
    /// Create a new dealer with a random secret
    pub fn new(
        id: ParticipantId,
        threshold: usize,
        num_participants: usize,
        rng: &mut impl CryptoRngCore,
    ) -> DkgResult<Self> {
        if threshold == 0 || threshold > num_participants {
            return Err(DkgError::InvalidThreshold {
                threshold,
                participants: num_participants,
            });
        }

        if num_participants < 2 {
            return Err(DkgError::InvalidParticipantCount(num_participants));
        }

        let polynomial = Polynomial::random(threshold, rng)?;

        Ok(Self {
            id,
            polynomial,
            num_participants,
        })
    }

    /// Create a dealer with a specific secret
    pub fn with_secret(
        id: ParticipantId,
        secret: Scalar,
        threshold: usize,
        num_participants: usize,
        rng: &mut impl CryptoRngCore,
    ) -> DkgResult<Self> {
        if threshold == 0 || threshold > num_participants {
            return Err(DkgError::InvalidThreshold {
                threshold,
                participants: num_participants,
            });
        }

        if num_participants < 2 {
            return Err(DkgError::InvalidParticipantCount(num_participants));
        }

        let polynomial = Polynomial::random_with_secret(secret, threshold, rng)?;

        Ok(Self {
            id,
            polynomial,
            num_participants,
        })
    }

    /// Get the dealer's ID
    pub fn id(&self) -> ParticipantId {
        self.id
    }

    /// Get the dealer's secret (constant term of polynomial)
    pub fn secret(&self) -> &Scalar {
        self.polynomial.secret()
    }

    /// Get the threshold
    pub fn threshold(&self) -> usize {
        self.polynomial.threshold()
    }

    /// Generate the deal (commitments and shares)
    pub fn generate_deal(&self) -> Deal {
        // Generate commitments
        let commitments = CommitmentSet::from_polynomial(&self.polynomial);

        // Generate shares for each participant (indices 1..=n)
        let shares: Vec<Share> = (1..=self.num_participants as u32)
            .map(|i| {
                let value = self.polynomial.evaluate_at_index(i);
                Share::new(i, self.id.0, value)
            })
            .collect();

        Deal {
            dealer: self.id,
            commitments,
            shares,
        }
    }

    /// Generate a share for a specific participant
    pub fn generate_share(&self, participant: u32) -> DkgResult<Share> {
        if participant == 0 || participant > self.num_participants as u32 {
            return Err(DkgError::InvalidParticipantId(participant));
        }

        let value = self.polynomial.evaluate_at_index(participant);
        Ok(Share::new(participant, self.id.0, value))
    }

    /// Get commitments without shares
    pub fn get_commitments(&self) -> CommitmentSet {
        CommitmentSet::from_polynomial(&self.polynomial)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn test_dealer_creation() {
        let dealer = Dealer::new(ParticipantId(1), 3, 5, &mut OsRng).unwrap();
        assert_eq!(dealer.id(), ParticipantId(1));
        assert_eq!(dealer.threshold(), 3);
    }

    #[test]
    fn test_invalid_threshold() {
        // Threshold > participants
        let result = Dealer::new(ParticipantId(1), 6, 5, &mut OsRng);
        assert!(matches!(result, Err(DkgError::InvalidThreshold { .. })));

        // Zero threshold
        let result = Dealer::new(ParticipantId(1), 0, 5, &mut OsRng);
        assert!(matches!(result, Err(DkgError::InvalidThreshold { .. })));
    }

    #[test]
    fn test_deal_generation() {
        let dealer = Dealer::new(ParticipantId(1), 3, 5, &mut OsRng).unwrap();
        let deal = dealer.generate_deal();

        // Should have 5 shares
        assert_eq!(deal.share_count(), 5);

        // Should have 3 commitments (threshold)
        assert_eq!(deal.commitments.len(), 3);

        // All shares should verify
        for (idx, valid) in deal.verify_all_shares() {
            assert!(valid, "Share {} failed verification", idx);
        }
    }

    #[test]
    fn test_dealer_with_secret() {
        let secret = Scalar::from_u64(42);
        let dealer =
            Dealer::with_secret(ParticipantId(1), secret.clone(), 3, 5, &mut OsRng).unwrap();

        assert_eq!(dealer.secret(), &secret);
    }

    #[test]
    fn test_share_retrieval() {
        let dealer = Dealer::new(ParticipantId(1), 2, 3, &mut OsRng).unwrap();
        let deal = dealer.generate_deal();

        // Get share for participant 2
        let share = deal.get_share(2);
        assert!(share.is_some());
        assert_eq!(share.unwrap().index, 2);

        // Non-existent participant
        let share = deal.get_share(10);
        assert!(share.is_none());
    }
}
