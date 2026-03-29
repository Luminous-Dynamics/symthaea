// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use loss_delta_guest::{self, LossDeltaInput};
use mycelix_core::ProofService;

#[test]
fn host_matches_guest_outputs() {
    let input = LossDeltaInput {
        round_id: 42,
        model_hash: "feedface".into(),
        baseline_loss: 0.55,
        new_loss: 0.45,
        tolerance: 0.2,
    };

    let guest_output = loss_delta_guest::compute_loss_delta(&input);
    let proof = ProofService::prove_loss_delta(&input).expect("prove");
    let host_output = ProofService::verify_loss_delta(&proof).expect("verify");

    assert_eq!(guest_output, host_output);
}
