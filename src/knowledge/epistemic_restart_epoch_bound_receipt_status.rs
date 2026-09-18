// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Machine-readable authority status for the passive EKM-076 receipt DTO.
//!
//! Shape/digest verification is deliberately separated from authority verification.
//! A future tranche must prove epoch issuance and evaluation timing before this data
//! can participate in mutation preparation.

use crate::knowledge::epistemic_restart_continuity::epoch_bound_receipt_data::{
    EpochBoundRevisionReceiptDataError, EpochBoundRevisionReceiptDataV2,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpochBoundRevisionReceiptAuthorityStatusV2 {
    shape_and_digest_verified: bool,
    epoch_issuance_verified: bool,
    evaluation_after_epoch_activation_proven: bool,
    accepted_by_current_mutation_facade: bool,
    mutation_authority: bool,
    activation_authorized: bool,
}

impl EpochBoundRevisionReceiptAuthorityStatusV2 {
    pub fn evaluate(
        receipt: &EpochBoundRevisionReceiptDataV2,
    ) -> Result<Self, EpochBoundRevisionReceiptDataError> {
        receipt.verify()?;
        Ok(Self {
            shape_and_digest_verified: true,
            epoch_issuance_verified: false,
            evaluation_after_epoch_activation_proven: false,
            accepted_by_current_mutation_facade: false,
            mutation_authority: false,
            activation_authorized: false,
        })
    }

    pub fn shape_and_digest_verified(self) -> bool {
        self.shape_and_digest_verified
    }

    pub fn epoch_issuance_verified(self) -> bool {
        self.epoch_issuance_verified
    }

    pub fn evaluation_after_epoch_activation_proven(self) -> bool {
        self.evaluation_after_epoch_activation_proven
    }

    pub fn accepted_by_current_mutation_facade(self) -> bool {
        self.accepted_by_current_mutation_facade
    }

    pub fn mutation_authority(self) -> bool {
        self.mutation_authority
    }

    pub fn activation_authorized(self) -> bool {
        self.activation_authorized
    }
}
