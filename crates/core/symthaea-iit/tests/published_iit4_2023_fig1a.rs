// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Published-input regression fixture for IIT 4.0 (2023), Figure 1A.
//!
//! Source: PyPhi's `iit4_2023_fig1a_substrate` in
//! `pyphi/examples.py` (upstream reference:
//! https://github.com/wmayner/pyphi/blob/134a8f31b49db512322465ed3ba5da7194d362fd/pyphi/examples.py).
//! The published state is (A, B, C) = (0, 1, 1). PyPhi documents the 2023
//! `phi_s` values as approximately 0.04 for {A}, 0.17 for {A, B}, and
//! 0.13 for {A, B, C}; the 2026 formalism differs for {A, B}.
//!
//! The fixture pins the input model and the published IIT 4.0 (2023) system
//! `phi_s` anchors for candidate systems {A}, {A,B}, and {A,B,C}. The
//! single-node {A} candidate is a valid nonzero system because the published
//! TPM contains an A -> A self-loop and its TotalCut is evaluated.

use symthaea_iit::{
    enumerate_iit4_system_partitions, BinaryCausalModel, Iit4SystemPhiSearchConfig,
    Iit4SystemPhiStatus,
};

const FIG1A_TPM: [f64; 24] = [
    0.11920292202211757, 0.7685247834990178, 0.16798161486607552, // 000
    0.026596993576865863, 0.9988874639671398, 0.5, // 100
    0.973403006423134, 0.40131233988754816, 0.16798161486607552, // 010
    0.8807970779778823, 0.9945137011005495, 0.5, // 110
    0.11920292202211757, 0.0054862988994504045, 0.5, // 001
    0.026596993576865863, 0.5986876601124519, 0.8320183851339245, // 101
    0.973403006423134, 0.0011125360328603216, 0.5, // 011
    0.8807970779778823, 0.23147521650098227, 0.8320183851339245, // 111
];

const FIG1A_WEIGHTS: [f64; 9] = [
    -0.2, 0.7, 0.2, // inputs A -> targets A, B, C
    0.7, -0.2, 0.0, // inputs B -> targets A, B, C
    0.0, -0.8, 0.2, // inputs C -> targets A, B, C
];

const FIG1A_CONNECTIVITY: [bool; 9] = [
    true, true, true, // A -> A, B, C
    true, true, false, // B -> A, B, no B -> C
    false, true, true, // no C -> A, C -> B, C -> C
];

fn fig1a_model() -> BinaryCausalModel {
    BinaryCausalModel::new(3, FIG1A_TPM.to_vec(), FIG1A_CONNECTIVITY.to_vec())
        .expect("published Fig. 1A substrate must satisfy the binary causal-model contract")
}

#[test]
fn published_fig1a_tpm_matches_pinned_logistic_fixture() {
    let model = fig1a_model();
    assert_eq!(model.n_units(), 3);
    assert_eq!(model.n_states(), 8);
    assert_eq!(model.transition_probabilities().len(), FIG1A_TPM.len());
    // These identities come from the executed PyPhi 2.0.0rc4 / IIT_4_0_2023
    // reference artifact, not merely from the rounded published display values.
    assert_eq!(
        model.model_sha256(),
        "6d67ba617b5d2ccbba711cb44f4753286a45ae6ded16fc1a544a77dc8e931ac1"
    );
    assert_eq!(
        model.evaluation_sha256(&[false, true, true]).unwrap(),
        "1442d185bc5d2c23e3c6fc92e94f21d1c492cbe41a451cebd1d51fa36e862744"
    );

    for (index, (actual, expected)) in model
        .transition_probabilities()
        .iter()
        .zip(FIG1A_TPM)
        .enumerate()
    {
        assert!(
            (actual - expected).abs() <= 1e-14,
            "TPM cell {index}: actual={actual:.17}, expected={expected:.17}"
        );
    }
}

#[test]
fn published_fig1a_tpm_is_reproducible_from_weights_and_logistic_rule() {
    // The upstream example maps binary 0 -> spin -1 and 1 -> spin +1,
    // then computes sigmoid(weighted_input / temperature) with T = 1/4.
    // Recompute every table entry independently of the literal fixture.
    for state_index in 0..8 {
        for target in 0..3 {
            let mut weighted_input = 0.0;
            for source in 0..3 {
                let spin = if ((state_index >> source) & 1) == 1 {
                    1.0
                } else {
                    -1.0
                };
                weighted_input += FIG1A_WEIGHTS[source * 3 + target] * spin;
            }
            let generated = 1.0 / (1.0 + (-weighted_input / 0.25).exp());
            let index = state_index * 3 + target;
            assert!(
                (generated - FIG1A_TPM[index]).abs() <= 1e-14,
                "generated TPM cell {index}: generated={generated:.17}, pinned={:.17}",
                FIG1A_TPM[index]
            );
        }
    }
}

#[test]
fn published_fig1a_state_index_uses_pyphi_little_endian_order() {
    let model = fig1a_model();
    let specified_state = [false, true, true]; // (A, B, C) = (0, 1, 1)
    assert_eq!(model.state_index(&specified_state).unwrap(), 6);

    // Row 6 is state 011. Unit A's next-state probability is the first
    // cell of that row; the tuple state is not reversed when indexed.
    assert!(
        (model
            .probability_next_one(&specified_state, 0)
            .unwrap()
            - 0.973403006423134)
            .abs()
            <= 1e-14
    );
}

#[test]
fn published_fig1a_connectivity_matches_the_weight_matrix() {
    let model = fig1a_model();
    for source in 0..3 {
        for target in 0..3 {
            assert_eq!(
                model.has_edge(source, target).unwrap(),
                FIG1A_CONNECTIVITY[source * 3 + target],
                "connectivity mismatch at {source} -> {target}"
            );
        }
    }
}

#[test]
fn published_fig1a_singleton_cut_reuses_uncut_eq4_background_posterior() {
    let model = fig1a_model();
    let state = [false, true, true];
    let uncut = model
        .forward_cause_likelihood_iit4(&[0], &[0], &state, &[0])
        .unwrap();
    let total_cut = enumerate_iit4_system_partitions(&[0], 3, false, 1)
        .unwrap()
        .into_iter()
        .find(|partition| partition.is_total_cut())
        .expect("a one-unit system must include TotalCut");

    let partitioned = model
        .partitioned_forward_cause_likelihood_iit4(&total_cut, &state)
        .unwrap();
    let expected = (uncut[0] + uncut[1]) * 0.5;

    assert_eq!(uncut.len(), 2);
    assert_eq!(partitioned.len(), 2);
    for (index, actual) in partitioned.iter().copied().enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-14,
            "cut cause likelihood at state {index}: actual={actual:.16}, expected={expected:.16}"
        );
    }
}

#[test]
fn published_fig1a_a_unit_phi_s_matches_pinned_iit4_2023_anchor() {
    let model = fig1a_model();
    let result = model
        .analyze_iit4_2023_system_phi_s(
            &[0],
            &[false, true, true], // substrate state (A, B, C) = (0, 1, 1)
            Iit4SystemPhiSearchConfig::default(),
        )
        .expect("the published single-unit self-loop system must complete");

    assert_eq!(
        result.request_sha256,
        "9b98c27017572e39052d64f69c82369a1d682efa0ad7823359c256faf2cc53eb"
    );
    assert_eq!(result.formalism.id(), "IIT_4_0_2023");
    assert!(result.complete);
    assert!(result.partition_summary.total_cut_included);
    assert_eq!(result.partition_summary.unique_non_total_cuts, 0);
    assert_eq!(result.partition_score_evaluations, 1);

    let phi_s = result.phi_s.expect("qualified A system must expose phi_s");
    assert!(
        (phi_s - 0.039590997724974646).abs() <= 1e-9,
        "Fig. 1A {{A}} phi_s mismatch: computed={phi_s:.12}, PyPhi 2.0.0rc4=0.039590997724974646 ± 1e-9"
    );
}

#[test]
fn published_fig1a_ab_proper_subsystem_phi_s_matches_pinned_iit4_2023_anchor() {
    let model = fig1a_model();
    let result = model
        .analyze_iit4_2023_system_phi_s(
            &[0, 1],
            &[false, true, true], // substrate state (A, B, C) = (0, 1, 1)
            Iit4SystemPhiSearchConfig::default(),
        )
        .expect("the published embedded {A,B} IIT 4.0 (2023) search must complete");

    assert_eq!(
        result.request_sha256,
        "83e7343053f4747a1cea57749b9a94f46baa2617bf31add6f4a94b8578a2468f"
    );
    assert_eq!(result.formalism.id(), "IIT_4_0_2023");
    assert!(result.complete);
    assert_eq!(result.partition_summary.unique_non_total_cuts, 3);
    assert_eq!(result.partition_score_evaluations, 3);

    let phi_s = result.phi_s.expect("unique {A,B} state pair must expose phi_s");
    assert!(
        (phi_s - 0.1718628889918045).abs() <= 1e-9,
        "Fig. 1A {{A,B}} phi_s mismatch: computed={phi_s:.12}, PyPhi 2.0.0rc4=0.1718628889918045 ± 1e-9"
    );
}

#[test]
fn published_fig1a_full_system_phi_s_matches_pinned_iit4_2023_anchor() {
    let model = fig1a_model();
    let result = model
        .analyze_iit4_2023_system_phi_s(
            &[0, 1, 2],
            &[false, true, true], // substrate state (A, B, C) = (0, 1, 1)
            Iit4SystemPhiSearchConfig::default(),
        )
        .expect("the published full-system IIT 4.0 (2023) search must complete");

    assert_eq!(
        result.request_sha256,
        "ea88d7dc1612dc72bd51d999cf582199ad76e0fd8fb90dfcd65f4788aa16fbfa"
    );
    assert_eq!(result.formalism.id(), "IIT_4_0_2023");
    assert!(result.complete);
    assert_eq!(result.status, Iit4SystemPhiStatus::ExactUniqueStatePair);
    assert_eq!(result.state_pair_results.len(), 1);
    assert_eq!(result.selected_state_pair_index, Some(0));
    assert_eq!(result.partition_summary.unique_non_total_cuts, 22);
    assert_eq!(result.partition_score_evaluations, 22);

    let phi_s = result.phi_s.expect("unique state-pair result must expose phi_s");
    assert!(
        (phi_s - 0.13387310087987658).abs() <= 1e-9,
        "Fig. 1A full-system phi_s mismatch: computed={phi_s:.12}, PyPhi 2.0.0rc4=0.13387310087987658 ± 1e-9"
    );
}
