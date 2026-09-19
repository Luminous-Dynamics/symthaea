// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Namespace bridge for EKM-082 activation-verifier current-head attestation.

use crate::knowledge::epistemic_restart_continuity::authority_epoch_contract::issuance_record::runtime_validation::trust_review::{
    ActivationVerifierTrustReviewDigestV1, ActivationVerifierTrustReviewReceiptV1,
    CanonicalVerifierTrustProfileDigestV1,
};

#[path = "epistemic_restart_activation_verifier_currentness.rs"]
mod implementation;

pub use implementation::*;
