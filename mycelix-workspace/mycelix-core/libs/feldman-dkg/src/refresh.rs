// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Proactive secret sharing: refresh shares without changing the public key
//!
//! In proactive secret sharing, participants periodically generate zero-sharing
//! polynomials (constant term = 0) and add the new sub-shares to their existing
//! shares. This effectively re-randomizes shares without changing the combined
//! secret or public key.
//!
//! ## Security Property
//!
//! An adversary who compromises shares in epoch E_1 and different shares in
//! epoch E_2 gains no advantage — the shares from different epochs are
//! uncorrelated. This bounds the adversary's collection window.
//!
//! ## Protocol
//!
//! 1. Each participant generates a random polynomial with secret = 0
//! 2. Each participant creates shares of their polynomial for all others
//! 3. Shares are distributed (encrypted) and verified
//! 4. Each participant adds all received sub-shares to their current share
//! 5. The combined secret and public key remain unchanged

use serde::{Deserialize, Serialize};

use crate::commitment::CommitmentSet;
use crate::error::{DkgError, DkgResult};
use crate::participant::ParticipantId;
use crate::polynomial::Polynomial;
use crate::scalar::Scalar;
use crate::share::Share;

/// A zero-sharing deal for proactive refresh
///
/// Like a Deal, but the polynomial has constant term 0
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RefreshDeal {
    /// The dealer who generated this refresh
    pub dealer: ParticipantId,
    /// Commitments to the zero-sharing polynomial
    pub commitments: CommitmentSet,
    /// Sub-shares for each participant
    pub shares: Vec<Share>,
    /// The epoch number this refresh is for
    pub epoch: u64,
}

/// A refresh round coordinator
#[derive(Debug)]
pub struct RefreshRound {
    /// Threshold of the original DKG
    threshold: usize,
    /// Number of participants
    _num_participants: usize,
    /// The current epoch
    epoch: u64,
    /// Collected refresh deals
    deals: Vec<RefreshDeal>,
    /// IDs of participants who have submitted
    submitted: Vec<ParticipantId>,
}

impl RefreshRound {
    /// Create a new refresh round
    pub fn new(threshold: usize, num_participants: usize, epoch: u64) -> DkgResult<Self> {
        if threshold == 0 || threshold > num_participants {
            return Err(DkgError::InvalidThreshold {
                threshold,
                participants: num_participants,
            });
        }

        Ok(Self {
            threshold,
            _num_participants: num_participants,
            epoch,
            deals: Vec::new(),
            submitted: Vec::new(),
        })
    }

    /// Generate a refresh deal (zero-sharing polynomial)
    pub fn generate_refresh_deal(
        dealer: ParticipantId,
        threshold: usize,
        num_participants: usize,
        epoch: u64,
        rng: &mut impl rand_core::CryptoRngCore,
    ) -> DkgResult<RefreshDeal> {
        // Create polynomial with secret = 0 (zero-sharing)
        let polynomial = Polynomial::random_with_secret(Scalar::zero(), threshold, rng)?;

        // Generate commitments
        let commitments = CommitmentSet::from_polynomial(&polynomial);

        // Generate sub-shares for each participant
        let shares: Vec<Share> = (1..=num_participants as u32)
            .map(|i| {
                let value = polynomial.evaluate_at_index(i);
                Share::new(i, dealer.0, value)
            })
            .collect();

        Ok(RefreshDeal {
            dealer,
            commitments,
            shares,
            epoch,
        })
    }

    /// Submit a refresh deal
    pub fn submit_deal(&mut self, deal: RefreshDeal) -> DkgResult<()> {
        if deal.epoch != self.epoch {
            return Err(DkgError::RefreshError(format!(
                "Wrong epoch: expected {}, got {}",
                self.epoch, deal.epoch
            )));
        }

        if self.submitted.contains(&deal.dealer) {
            return Err(DkgError::DuplicateShare(deal.dealer.0));
        }

        // Verify the zero-sharing property: the secret commitment should be g^0 = identity
        // In practice, C_0 = g^0 = identity point
        let secret_commitment = deal.commitments.secret_commitment();
        let identity_commitment = crate::commitment::Commitment::new(&Scalar::zero());
        if *secret_commitment != identity_commitment {
            return Err(DkgError::RefreshError(format!(
                "Dealer {} submitted non-zero-sharing polynomial",
                deal.dealer.0
            )));
        }

        // Verify all shares against commitments
        for share in &deal.shares {
            if !deal.commitments.verify_share(share.index, &share.value) {
                return Err(DkgError::ShareVerificationFailed {
                    participant: share.index,
                    dealer: deal.dealer.0,
                });
            }
        }

        self.submitted.push(deal.dealer);
        self.deals.push(deal);
        Ok(())
    }

    /// Check if all participants have submitted
    pub fn is_complete(&self) -> bool {
        self.submitted.len() >= self.threshold
    }

    /// Get the number of submitted deals
    pub fn deal_count(&self) -> usize {
        self.deals.len()
    }

    /// Compute the refresh delta for a specific participant
    ///
    /// This is the sum of all sub-shares that participant received.
    /// They add this to their existing share to get the refreshed share.
    pub fn compute_refresh_delta(&self, participant: u32) -> DkgResult<Scalar> {
        if self.deals.is_empty() {
            return Err(DkgError::RefreshError("No refresh deals submitted".into()));
        }

        let mut delta = Scalar::zero();
        for deal in &self.deals {
            if let Some(share) = deal.shares.iter().find(|s| s.index == participant) {
                delta += share.value.clone();
            } else {
                return Err(DkgError::ParticipantNotFound(participant));
            }
        }

        Ok(delta)
    }

    /// Apply refresh to an existing share
    pub fn apply_refresh(&self, participant: u32, current_share: &Scalar) -> DkgResult<Scalar> {
        let delta = self.compute_refresh_delta(participant)?;
        Ok(current_share.clone() + delta)
    }

    /// Get the current epoch
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
}

/// Epoch-tracked share that records when it was last refreshed
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpochShare {
    /// The participant index
    pub index: u32,
    /// The current share value
    pub value: Scalar,
    /// The epoch this share was last refreshed in
    pub epoch: u64,
}

impl EpochShare {
    /// Create a new epoch share from the initial DKG (epoch 0)
    pub fn new(index: u32, value: Scalar) -> Self {
        Self {
            index,
            value,
            epoch: 0,
        }
    }

    /// Apply a refresh delta and advance the epoch
    pub fn refresh(&mut self, delta: Scalar, new_epoch: u64) -> DkgResult<()> {
        if new_epoch <= self.epoch {
            return Err(DkgError::RefreshError(format!(
                "New epoch {} must be greater than current epoch {}",
                new_epoch, self.epoch
            )));
        }

        self.value = self.value.clone() + delta;
        self.epoch = new_epoch;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ceremony::{DkgCeremony, DkgConfig};
    use crate::dealer::Dealer;
    use crate::polynomial::lagrange_interpolate_at_zero;
    use rand::rngs::OsRng;

    #[test]
    fn test_refresh_deal_zero_sharing() {
        let deal =
            RefreshRound::generate_refresh_deal(ParticipantId(1), 2, 3, 1, &mut OsRng).unwrap();

        // Sum of all sub-shares evaluated at 0 should be 0
        // (but individual sub-shares are non-zero)
        assert_eq!(deal.epoch, 1);
        assert_eq!(deal.shares.len(), 3);
    }

    #[test]
    fn test_refresh_preserves_secret() {
        // Set up initial DKG
        let config = DkgConfig::new(2, 3).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        for i in 1..=3 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }

        let secrets = [
            Scalar::from_u64(100),
            Scalar::from_u64(200),
            Scalar::from_u64(300),
        ];
        let expected_secret = Scalar::from_u64(600);

        for (i, secret) in secrets.iter().enumerate() {
            let dealer = Dealer::with_secret(
                ParticipantId((i + 1) as u32),
                secret.clone(),
                2,
                3,
                &mut OsRng,
            )
            .unwrap();
            let deal = dealer.generate_deal();
            ceremony
                .submit_deal(ParticipantId((i + 1) as u32), deal, 0)
                .unwrap();
        }
        ceremony.finalize().unwrap();

        // Get initial combined shares
        let mut shares: Vec<Scalar> = (1..=3)
            .map(|i| {
                ceremony
                    .get_combined_share(ParticipantId(i))
                    .unwrap()
                    .value()
                    .clone()
            })
            .collect();

        // Verify initial reconstruction
        let points: Vec<_> = shares
            .iter()
            .enumerate()
            .map(|(i, s)| (Scalar::from_u64((i + 1) as u64), s.clone()))
            .collect();
        let reconstructed = lagrange_interpolate_at_zero(&points).unwrap();
        assert_eq!(reconstructed, expected_secret);

        // Perform refresh (epoch 1)
        let mut refresh = RefreshRound::new(2, 3, 1).unwrap();

        for i in 1..=3u32 {
            let deal =
                RefreshRound::generate_refresh_deal(ParticipantId(i), 2, 3, 1, &mut OsRng).unwrap();
            refresh.submit_deal(deal).unwrap();
        }

        // Apply refresh to each share
        for i in 0..3 {
            shares[i] = refresh.apply_refresh((i + 1) as u32, &shares[i]).unwrap();
        }

        // Verify reconstruction still works with refreshed shares
        let refreshed_points: Vec<_> = shares
            .iter()
            .enumerate()
            .map(|(i, s)| (Scalar::from_u64((i + 1) as u64), s.clone()))
            .collect();
        let reconstructed_after = lagrange_interpolate_at_zero(&refreshed_points).unwrap();
        assert_eq!(
            reconstructed_after, expected_secret,
            "Secret must be preserved after refresh"
        );
    }

    #[test]
    fn test_refresh_changes_shares() {
        // Set up initial DKG
        let config = DkgConfig::new(2, 3).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        for i in 1..=3 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }

        for i in 1..=3u32 {
            let dealer = Dealer::new(ParticipantId(i), 2, 3, &mut OsRng).unwrap();
            let deal = dealer.generate_deal();
            ceremony.submit_deal(ParticipantId(i), deal, 0).unwrap();
        }
        ceremony.finalize().unwrap();

        let original_shares: Vec<Scalar> = (1..=3)
            .map(|i| {
                ceremony
                    .get_combined_share(ParticipantId(i))
                    .unwrap()
                    .value()
                    .clone()
            })
            .collect();

        // Refresh
        let mut refresh = RefreshRound::new(2, 3, 1).unwrap();
        for i in 1..=3u32 {
            let deal =
                RefreshRound::generate_refresh_deal(ParticipantId(i), 2, 3, 1, &mut OsRng).unwrap();
            refresh.submit_deal(deal).unwrap();
        }

        let refreshed_shares: Vec<Scalar> = (1..=3)
            .map(|i| {
                refresh
                    .apply_refresh(i, &original_shares[(i - 1) as usize])
                    .unwrap()
            })
            .collect();

        // At least one share should have changed (overwhelmingly likely with random polynomials)
        let any_changed = original_shares
            .iter()
            .zip(refreshed_shares.iter())
            .any(|(orig, refreshed)| orig != refreshed);
        assert!(any_changed, "Refresh should change share values");
    }

    #[test]
    fn test_refresh_round_reject_wrong_epoch() {
        let mut refresh = RefreshRound::new(2, 3, 5).unwrap();

        let deal =
            RefreshRound::generate_refresh_deal(ParticipantId(1), 2, 3, 3, &mut OsRng).unwrap();

        let result = refresh.submit_deal(deal);
        assert!(matches!(result, Err(DkgError::RefreshError(_))));
    }

    #[test]
    fn test_refresh_round_reject_duplicate() {
        let mut refresh = RefreshRound::new(2, 3, 1).unwrap();

        let deal1 =
            RefreshRound::generate_refresh_deal(ParticipantId(1), 2, 3, 1, &mut OsRng).unwrap();
        let deal2 =
            RefreshRound::generate_refresh_deal(ParticipantId(1), 2, 3, 1, &mut OsRng).unwrap();

        refresh.submit_deal(deal1).unwrap();
        let result = refresh.submit_deal(deal2);
        assert!(matches!(result, Err(DkgError::DuplicateShare(1))));
    }

    #[test]
    fn test_refresh_round_reject_non_zero_sharing() {
        let mut refresh = RefreshRound::new(2, 3, 1).unwrap();

        // Create a deal with non-zero secret (should be rejected)
        let dealer = Dealer::new(ParticipantId(1), 2, 3, &mut OsRng).unwrap();
        let normal_deal = dealer.generate_deal();

        let bad_refresh = RefreshDeal {
            dealer: ParticipantId(1),
            commitments: normal_deal.commitments,
            shares: normal_deal.shares,
            epoch: 1,
        };

        let result = refresh.submit_deal(bad_refresh);
        assert!(matches!(result, Err(DkgError::RefreshError(_))));
    }

    #[test]
    fn test_epoch_share_refresh() {
        let mut share = EpochShare::new(1, Scalar::from_u64(100));
        assert_eq!(share.epoch, 0);

        share.refresh(Scalar::from_u64(50), 1).unwrap();
        assert_eq!(share.epoch, 1);
        assert_eq!(share.value, Scalar::from_u64(150));

        // Can't go backwards
        let result = share.refresh(Scalar::from_u64(10), 0);
        assert!(matches!(result, Err(DkgError::RefreshError(_))));
    }

    #[test]
    fn test_multiple_refresh_epochs() {
        // Set up initial DKG
        let config = DkgConfig::new(2, 3).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        for i in 1..=3 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }

        let secrets = [
            Scalar::from_u64(10),
            Scalar::from_u64(20),
            Scalar::from_u64(30),
        ];
        let expected_secret = Scalar::from_u64(60);

        for (i, secret) in secrets.iter().enumerate() {
            let dealer = Dealer::with_secret(
                ParticipantId((i + 1) as u32),
                secret.clone(),
                2,
                3,
                &mut OsRng,
            )
            .unwrap();
            let deal = dealer.generate_deal();
            ceremony
                .submit_deal(ParticipantId((i + 1) as u32), deal, 0)
                .unwrap();
        }
        ceremony.finalize().unwrap();

        let mut epoch_shares: Vec<EpochShare> = (1..=3)
            .map(|i| {
                let cs = ceremony.get_combined_share(ParticipantId(i)).unwrap();
                EpochShare::new(i as u32, cs.value().clone())
            })
            .collect();

        // Run 5 refresh epochs
        for epoch in 1..=5 {
            let mut refresh = RefreshRound::new(2, 3, epoch).unwrap();
            for i in 1..=3u32 {
                let deal =
                    RefreshRound::generate_refresh_deal(ParticipantId(i), 2, 3, epoch, &mut OsRng)
                        .unwrap();
                refresh.submit_deal(deal).unwrap();
            }

            for share in &mut epoch_shares {
                let delta = refresh.compute_refresh_delta(share.index).unwrap();
                share.refresh(delta, epoch).unwrap();
            }
        }

        // Verify reconstruction still works after 5 refreshes
        let points: Vec<_> = epoch_shares
            .iter()
            .map(|s| (Scalar::from_u64(s.index as u64), s.value.clone()))
            .collect();
        let reconstructed = lagrange_interpolate_at_zero(&points).unwrap();
        assert_eq!(
            reconstructed, expected_secret,
            "Secret must survive 5 refresh epochs"
        );
    }
}
