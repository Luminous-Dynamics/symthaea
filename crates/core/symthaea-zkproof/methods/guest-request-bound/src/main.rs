#![no_main]
#![no_std]

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use risc0_zkvm::guest::env;
use symthaea_zkproof_request_bound_core::{
    BalanceProofRequestBoundInputV1, BalanceProofRequestBoundOutputV1,
};

risc0_zkvm::guest::entry!(main);

/// Prove the threshold relation and commit the exact request-context digest.
///
/// This separate guest package receives its own RISC Zero ImageID. The legacy
/// guest package and its ImageID remain unchanged.
fn main() {
    let input: BalanceProofRequestBoundInputV1 = env::read();

    // A malformed request cannot produce a valid request-bound receipt.
    assert!(input.request.validate().is_ok());

    let output = BalanceProofRequestBoundOutputV1 {
        sufficient: input.balance >= input.request.required_minimum,
        required_minimum: input.request.required_minimum,
        request_digest: input.request.canonical_digest(),
    };

    env::commit(&output);
}
