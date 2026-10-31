// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! IIT 4.0 proper-subsystem background regression tests.
//!
//! The noisy-OR substrate and reference repertoire values are from PyPhi's
//! test/example_substrates.py and test/test_background_conditioning.py:
//! https://github.com/wmayner/pyphi/blob/main/test/example_substrates.py
//! https://github.com/wmayner/pyphi/blob/main/test/test_background_conditioning.py
//!
//! System S={A,B} is embedded in substrate U={A,B,C}. Eq. 3 fixes C at
//! its observed current state for effects. Eq. 4 weights C's past by the
//! posterior conditioned on the observed whole-substrate state for causes.

use symthaea_iit::{
    BinaryCausalModel, Iit4SystemPhiSearchConfig, RepertoireError,
};

fn noisy_or_background_model() -> BinaryCausalModel {
    // PyPhi's row order is C outer, B middle, A inner, matching the crate's
    // little-endian indexing with A as bit 0. Each row stores P(A'), P(B'), P(C').
    let mut tpm = Vec::with_capacity(24);
    for c in [false, true] {
        for b in [false, true] {
            for a in [false, true] {
                tpm.push(if b || c { 0.9 } else { 0.1 });
                tpm.push(if a { 1.0 } else { 0.0 });
                tpm.push(0.5);
            }
        }
    }

    // Directed edges: B -> A, C -> A, and A -> B.
    let connectivity = vec![
        false, true, false,
        true, false, false,
        true, false, false,
    ];
    BinaryCausalModel::new(3, tpm, connectivity).expect("valid PyPhi noisy-OR fixture")
}

#[test]
fn proper_subsystem_eq4_cause_repertoire_matches_pyphi_background_oracle() {
    let model = noisy_or_background_model();
    let current_state = [true, false, false]; // (A, B, C) = (1, 0, 0)

    let repertoire = model
        .cause_repertoire_iit4(&[0, 1], &[0], &current_state, &[1])
        .unwrap();

    assert!(
        (repertoire[0] - 0.40566037735849053).abs() <= 1e-12,
        "P(B_past=0 | A_now=1, B_now=0, C_now=0) = {:.17}",
        repertoire[0]
    );
    assert!(
        (repertoire[1] - 0.5943396226415094).abs() <= 1e-12,
        "P(B_past=1 | A_now=1, B_now=0, C_now=0) = {:.17}",
        repertoire[1]
    );
}

#[test]
fn proper_subsystem_eq3_effect_conditions_background_at_observed_state() {
    let model = noisy_or_background_model();
    let current_state = [true, false, false];

    let repertoire = model
        .effect_repertoire_iit4(&[0, 1], &[0], &current_state, &[1])
        .unwrap();

    // B' is a copy of A; external C is held at its observed state.
    assert_eq!(repertoire, vec![0.0, 1.0]);
}

#[test]
fn full_substrate_iit4_repertoires_reduce_to_existing_primitives() {
    let model = noisy_or_background_model();
    let current_state = [true, false, false];
    let full = [0, 1, 2];

    let expected_cause = model.cause_repertoire(&[0], &current_state, &[1]).unwrap();
    let actual_cause = model
        .cause_repertoire_iit4(&full, &[0], &current_state, &[1])
        .unwrap();
    assert_eq!(actual_cause, expected_cause);

    let expected_effect = model.effect_repertoire(&[0], &current_state, &[1]).unwrap();
    let actual_effect = model
        .effect_repertoire_iit4(&full, &[0], &current_state, &[1])
        .unwrap();
    assert_eq!(actual_effect, expected_effect);
}

#[test]
fn proper_subsystem_repertoire_rejects_external_mechanism_or_purview_units() {
    let model = noisy_or_background_model();
    let current_state = [true, false, false];

    assert!(matches!(
        model.cause_repertoire_iit4(&[0, 1], &[2], &current_state, &[1]),
        Err(RepertoireError::SubsetOutsideSystem { subset: "mechanism", unit: 2 })
    ));
    assert!(matches!(
        model.effect_repertoire_iit4(&[0, 1], &[0], &current_state, &[2]),
        Err(RepertoireError::SubsetOutsideSystem { subset: "purview", unit: 2 })
    ));
}

#[test]
fn proper_subsystem_iit4_2023_system_search_no_longer_rejects_embedded_systems() {
    let model = noisy_or_background_model();
    let current_state = [true, false, false];

    let result = model
        .analyze_iit4_2023_system_phi_s(
            &[0, 1],
            &current_state,
            Iit4SystemPhiSearchConfig {
                max_partition_combinations: 32,
                max_state_pair_evaluations: 16,
                ..Iit4SystemPhiSearchConfig::default()
            },
        )
        .unwrap();

    assert!(result.complete);
    assert_eq!(result.candidate_system, vec![0, 1]);
    assert_eq!(result.partition_score_evaluations, 3);
}
