// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Participant in DKG ceremony

use rand_core::CryptoRngCore;
use serde::{Deserialize, Serialize};

use crate::commitment::CommitmentSet;
use crate::dealer::{Deal, Dealer};
use crate::error::{DkgError, DkgResult};
use crate::scalar::Scalar;
use crate::share::{CombinedShare, ShareSet};

/// A participant identifier (1-indexed)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ParticipantId(pub u32);

impl ParticipantId {
    /// Create a new participant ID
    pub fn new(id: u32) -> DkgResult<Self> {
        if id == 0 {
            return Err(DkgError::InvalidParticipantId(id));
        }
        Ok(Self(id))
    }

    /// Get the inner value
    pub fn value(&self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for ParticipantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "P{}", self.0)
    }
}

/// State of a participant in the DKG ceremony
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParticipantState {
    /// Registered but not yet dealt
    Registered,
    /// Has generated and distributed their deal
    Dealt,
    /// Has received and verified all shares
    Verified,
    /// Ceremony complete, has combined share
    Complete,
    /// Participant has been disqualified
    Disqualified,
}

/// A participant in the DKG ceremony
#[derive(Debug)]
pub struct Participant {
    /// This participant's ID
    id: ParticipantId,
    /// Current state
    state: ParticipantState,
    /// Threshold for the ceremony
    threshold: usize,
    /// Total number of participants
    num_participants: usize,
    /// Received shares from other dealers
    received_shares: ShareSet,
    /// Received commitments from other dealers
    received_commitments: Vec<(ParticipantId, CommitmentSet)>,
    /// This participant's dealer (if they are dealing)
    dealer: Option<Dealer>,
    /// Final combined share (after ceremony completion)
    combined_share: Option<CombinedShare>,
}

impl Participant {
    /// Create a new participant
    pub fn new(id: ParticipantId, threshold: usize, num_participants: usize) -> DkgResult<Self> {
        if threshold == 0 || threshold > num_participants {
            return Err(DkgError::InvalidThreshold {
                threshold,
                participants: num_participants,
            });
        }

        if id.0 > num_participants as u32 {
            return Err(DkgError::InvalidParticipantId(id.0));
        }

        Ok(Self {
            id,
            state: ParticipantState::Registered,
            threshold,
            num_participants,
            received_shares: ShareSet::new(id.0),
            received_commitments: Vec::new(),
            dealer: None,
            combined_share: None,
        })
    }

    /// Get the participant ID
    pub fn id(&self) -> ParticipantId {
        self.id
    }

    /// Get current state
    pub fn state(&self) -> ParticipantState {
        self.state
    }

    /// Initialize this participant as a dealer
    pub fn init_dealer(&mut self, rng: &mut impl CryptoRngCore) -> DkgResult<()> {
        if self.dealer.is_some() {
            return Ok(()); // Already initialized
        }

        self.dealer = Some(Dealer::new(
            self.id,
            self.threshold,
            self.num_participants,
            rng,
        )?);
        Ok(())
    }

    /// Initialize this participant as a dealer with a specific secret
    pub fn init_dealer_with_secret(
        &mut self,
        secret: Scalar,
        rng: &mut impl CryptoRngCore,
    ) -> DkgResult<()> {
        self.dealer = Some(Dealer::with_secret(
            self.id,
            secret,
            self.threshold,
            self.num_participants,
            rng,
        )?);
        Ok(())
    }

    /// Generate this participant's deal
    pub fn generate_deal(&mut self, rng: &mut impl CryptoRngCore) -> DkgResult<Deal> {
        if self.state == ParticipantState::Disqualified {
            return Err(DkgError::WrongPhase {
                expected: "Registered".into(),
                actual: "Disqualified".into(),
            });
        }

        self.init_dealer(rng)?;
        let deal = self.dealer.as_ref().expect("dealer initialized by init_dealer above").generate_deal();
        self.state = ParticipantState::Dealt;
        Ok(deal)
    }

    /// Process a deal received from another participant
    pub fn process_deal(&mut self, deal: &Deal) -> DkgResult<bool> {
        if self.state == ParticipantState::Disqualified {
            return Err(DkgError::WrongPhase {
                expected: "Registered or Dealt".into(),
                actual: "Disqualified".into(),
            });
        }

        // Get our share from this deal
        let share = deal
            .get_share(self.id.0)
            .ok_or(DkgError::ParticipantNotFound(self.id.0))?;

        // Verify the share against commitments
        let valid = deal.commitments.verify_share(self.id.0, share.value());

        if valid {
            // Store the share and commitments
            self.received_shares.add_share(share.clone())?;
            self.received_commitments
                .push((deal.dealer, deal.commitments.clone()));
            Ok(true)
        } else {
            // Invalid share - would trigger complaint in real protocol
            Err(DkgError::ShareVerificationFailed {
                participant: self.id.0,
                dealer: deal.dealer.0,
            })
        }
    }

    /// Get the number of received shares
    pub fn received_share_count(&self) -> usize {
        self.received_shares.len()
    }

    /// Check if we have received shares from all participants
    pub fn has_all_shares(&self) -> bool {
        self.received_shares.len() == self.num_participants
    }

    /// Combine all received shares into the final share
    pub fn combine_shares(&mut self) -> DkgResult<()> {
        if !self.has_all_shares() {
            return Err(DkgError::NotEnoughParticipants {
                required: self.num_participants,
                actual: self.received_shares.len(),
            });
        }

        let combined = CombinedShare::from_share_set(&self.received_shares)?;
        self.combined_share = Some(combined);
        self.state = ParticipantState::Complete;
        Ok(())
    }

    /// Get the combined share (after ceremony completion)
    pub fn combined_share(&self) -> Option<&CombinedShare> {
        self.combined_share.as_ref()
    }

    /// Get the combined public key commitment
    ///
    /// This is the sum of all dealers' secret commitments (C_0 values)
    pub fn combined_public_key(&self) -> DkgResult<crate::commitment::Commitment> {
        if self.received_commitments.is_empty() {
            return Err(DkgError::CryptoError("No commitments received".into()));
        }

        let commitment_refs: Vec<_> = self.received_commitments.iter().map(|(_, c)| c).collect();
        CommitmentSet::combine(&commitment_refs)
    }

    /// Disqualify this participant
    pub fn disqualify(&mut self) {
        self.state = ParticipantState::Disqualified;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn test_participant_creation() {
        let p = Participant::new(ParticipantId(1), 2, 3).unwrap();
        assert_eq!(p.id(), ParticipantId(1));
        assert_eq!(p.state(), ParticipantState::Registered);
    }

    #[test]
    fn test_invalid_participant_id() {
        // ID 0 is invalid
        let result = ParticipantId::new(0);
        assert!(result.is_err());

        // ID > num_participants is invalid
        let result = Participant::new(ParticipantId(5), 2, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_deal_generation_and_processing() {
        let mut p1 = Participant::new(ParticipantId(1), 2, 3).unwrap();
        let mut p2 = Participant::new(ParticipantId(2), 2, 3).unwrap();

        // P1 generates deal
        let deal1 = p1.generate_deal(&mut OsRng).unwrap();
        assert_eq!(p1.state(), ParticipantState::Dealt);

        // P2 processes P1's deal
        let valid = p2.process_deal(&deal1).unwrap();
        assert!(valid);
        assert_eq!(p2.received_share_count(), 1);
    }

    #[test]
    fn test_full_participant_workflow() {
        // 2-of-3 scheme
        let mut participants: Vec<_> = (1..=3)
            .map(|i| Participant::new(ParticipantId(i), 2, 3).unwrap())
            .collect();

        // Each participant generates a deal
        let deals: Vec<_> = participants
            .iter_mut()
            .map(|p| p.generate_deal(&mut OsRng).unwrap())
            .collect();

        // Each participant processes all deals
        for p in &mut participants {
            for deal in &deals {
                p.process_deal(deal).unwrap();
            }
        }

        // Each participant combines shares
        for p in &mut participants {
            p.combine_shares().unwrap();
            assert_eq!(p.state(), ParticipantState::Complete);
            assert!(p.combined_share().is_some());
        }

        // All participants should have the same public key
        let pk1 = participants[0].combined_public_key().unwrap();
        let pk2 = participants[1].combined_public_key().unwrap();
        let pk3 = participants[2].combined_public_key().unwrap();
        assert_eq!(pk1, pk2);
        assert_eq!(pk2, pk3);
    }
}
