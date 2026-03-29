// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Secret share types for DKG

use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::error::{DkgError, DkgResult};
use crate::scalar::Scalar;

/// A secret share for a participant
///
/// Share s_i = f(i) where f is the dealer's polynomial
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop, Serialize, Deserialize)]
pub struct Share {
    /// The participant index this share is for
    pub index: u32,
    /// The dealer who created this share
    pub dealer: u32,
    /// The share value s_i = f(i)
    pub value: Scalar,
}

impl Share {
    /// Create a new share
    pub fn new(index: u32, dealer: u32, value: Scalar) -> Self {
        Self {
            index,
            dealer,
            value,
        }
    }

    /// Get the participant index
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Get the dealer index
    pub fn dealer(&self) -> u32 {
        self.dealer
    }

    /// Get the share value
    pub fn value(&self) -> &Scalar {
        &self.value
    }
}

/// A set of shares from multiple dealers for one participant
///
/// In DKG, each participant receives shares from all dealers and
/// combines them into their final share.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShareSet {
    /// The participant these shares are for
    participant: u32,
    /// Shares from each dealer
    shares: Vec<Share>,
}

impl ShareSet {
    /// Create a new share set for a participant
    pub fn new(participant: u32) -> Self {
        Self {
            participant,
            shares: Vec::new(),
        }
    }

    /// Add a share from a dealer
    pub fn add_share(&mut self, share: Share) -> DkgResult<()> {
        if share.index != self.participant {
            return Err(DkgError::InvalidShareIndex(share.index as usize));
        }

        // Check for duplicate
        if self.shares.iter().any(|s| s.dealer == share.dealer) {
            return Err(DkgError::DuplicateShare(share.dealer));
        }

        self.shares.push(share);
        Ok(())
    }

    /// Get the participant index
    pub fn participant(&self) -> u32 {
        self.participant
    }

    /// Get all shares
    pub fn shares(&self) -> &[Share] {
        &self.shares
    }

    /// Get number of shares received
    pub fn len(&self) -> usize {
        self.shares.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.shares.is_empty()
    }

    /// Combine all shares into a single share
    ///
    /// The combined share is s_i = Σ s_{i,j} for all dealers j
    pub fn combine(&self) -> DkgResult<Scalar> {
        if self.shares.is_empty() {
            return Err(DkgError::CryptoError("No shares to combine".into()));
        }

        let mut combined = Scalar::zero();
        for share in &self.shares {
            combined += share.value.clone();
        }
        Ok(combined)
    }

    /// Get the share from a specific dealer
    pub fn get_share_from(&self, dealer: u32) -> Option<&Share> {
        self.shares.iter().find(|s| s.dealer == dealer)
    }
}

/// Combined share after DKG completion
///
/// This is the participant's final share of the distributed secret.
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop, Serialize, Deserialize)]
pub struct CombinedShare {
    /// The participant index
    pub index: u32,
    /// The combined share value
    pub value: Scalar,
}

impl CombinedShare {
    /// Create from a share set
    pub fn from_share_set(shares: &ShareSet) -> DkgResult<Self> {
        Ok(Self {
            index: shares.participant(),
            value: shares.combine()?,
        })
    }

    /// Create directly
    pub fn new(index: u32, value: Scalar) -> Self {
        Self { index, value }
    }

    /// Get the participant index
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Get the share value
    pub fn value(&self) -> &Scalar {
        &self.value
    }

    /// Convert to a (x, y) point for interpolation
    pub fn to_point(&self) -> (Scalar, Scalar) {
        (Scalar::from_u64(self.index as u64), self.value.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_share_set_combination() {
        let mut shares = ShareSet::new(1);

        // Add shares from 3 dealers
        shares
            .add_share(Share::new(1, 1, Scalar::from_u64(10)))
            .unwrap();
        shares
            .add_share(Share::new(1, 2, Scalar::from_u64(20)))
            .unwrap();
        shares
            .add_share(Share::new(1, 3, Scalar::from_u64(30)))
            .unwrap();

        let combined = shares.combine().unwrap();
        assert_eq!(combined, Scalar::from_u64(60));
    }

    #[test]
    fn test_duplicate_share_rejected() {
        let mut shares = ShareSet::new(1);

        shares
            .add_share(Share::new(1, 1, Scalar::from_u64(10)))
            .unwrap();

        // Duplicate from same dealer should fail
        let result = shares.add_share(Share::new(1, 1, Scalar::from_u64(20)));
        assert!(matches!(result, Err(DkgError::DuplicateShare(1))));
    }

    #[test]
    fn test_wrong_index_rejected() {
        let mut shares = ShareSet::new(1);

        // Share for wrong participant should fail
        let result = shares.add_share(Share::new(2, 1, Scalar::from_u64(10)));
        assert!(matches!(result, Err(DkgError::InvalidShareIndex(2))));
    }

    #[test]
    fn test_combined_share() {
        let mut shares = ShareSet::new(5);
        shares
            .add_share(Share::new(5, 1, Scalar::from_u64(100)))
            .unwrap();
        shares
            .add_share(Share::new(5, 2, Scalar::from_u64(200)))
            .unwrap();

        let combined = CombinedShare::from_share_set(&shares).unwrap();
        assert_eq!(combined.index(), 5);
        assert_eq!(combined.value(), &Scalar::from_u64(300));
    }
}
