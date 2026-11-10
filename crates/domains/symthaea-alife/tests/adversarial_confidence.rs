// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::convert::Infallible;

use symthaea_alife::adversarial_confidence::{
    ConfidenceModel, ExactSequenceOracle, GroundTruthOracle, HillClimbConfig, HillClimbError,
    InfallibleConfidence, greedy_single_site_hill_climb,
};

struct NeverPositive;

impl GroundTruthOracle for NeverPositive {
    fn is_positive(&self, _candidate: &[u8]) -> bool {
        false
    }
}

struct InvalidConfidence(f64);

impl ConfidenceModel for InvalidConfidence {
    type Error = Infallible;

    fn confidence(&mut self, _candidate: &[u8]) -> Result<f64, Self::Error> {
        Ok(self.0)
    }
}

#[test]
fn confidence_ascent_can_find_a_confident_false_positive() {
    let mut model = InfallibleConfidence(|candidate: &[u8]| {
        let b_count = candidate.iter().filter(|&&symbol| symbol == b'b').count();
        b_count as f64 / candidate.len() as f64
    });
    let oracle = NeverPositive;

    let result = greedy_single_site_hill_climb(
        &mut model,
        &oracle,
        b"aaaaa",
        b"ab",
        HillClimbConfig {
            max_model_queries: 500,
            seed: 7,
        },
    )
    .expect("attack completes");

    assert_eq!(result.model_queries, 500);
    assert_eq!(result.endpoint.as_slice(), b"bbbbb");
    assert_eq!(result.endpoint_confidence, 1.0);
    assert!(!result.endpoint_is_positive);
    assert!(result.accepted_mutations > 0);
}

#[test]
fn identical_seed_and_inputs_produce_identical_attack_trace() {
    let run = || {
        let mut model = InfallibleConfidence(|candidate: &[u8]| {
            candidate.iter().filter(|&&symbol| symbol == b'c').count() as f64
                / candidate.len() as f64
        });
        greedy_single_site_hill_climb(
            &mut model,
            &NeverPositive,
            b"aaaaaa",
            b"abc",
            HillClimbConfig {
                max_model_queries: 100,
                seed: 0x0A11_CE55,
            },
        )
        .expect("attack completes")
    };

    assert_eq!(run(), run());
}

#[test]
fn flat_confidence_surface_never_accepts_mutations() {
    let mut model = InfallibleConfidence(|_candidate: &[u8]| 0.5);
    let result = greedy_single_site_hill_climb(
        &mut model,
        &NeverPositive,
        b"aaaa",
        b"ab",
        HillClimbConfig {
            max_model_queries: 40,
            seed: 13,
        },
    )
    .expect("attack completes");

    assert_eq!(result.accepted_mutations, 0);
    assert_eq!(result.endpoint.as_slice(), b"aaaa");
    assert_eq!(result.endpoint_confidence, 0.5);
}

#[test]
fn exact_sequence_oracle_is_separate_from_the_model() {
    let oracle = ExactSequenceOracle::new([b"abc".to_vec(), b"bca".to_vec()]);
    assert_eq!(oracle.len(), 2);
    assert!(oracle.is_positive(b"abc"));
    assert!(!oracle.is_positive(b"acb"));
}

#[test]
fn query_budget_includes_the_initial_model_evaluation() {
    let mut model = InfallibleConfidence(|_candidate: &[u8]| 0.7);
    let result = greedy_single_site_hill_climb(
        &mut model,
        &NeverPositive,
        b"aa",
        b"ab",
        HillClimbConfig {
            max_model_queries: 1,
            seed: 1,
        },
    )
    .expect("one initial evaluation");
    assert_eq!(result.model_queries, 1);
    assert_eq!(result.trace.len(), 1);
}

#[test]
fn invalid_confidence_fails_closed() {
    let mut model = InvalidConfidence(f64::NAN);
    let error = greedy_single_site_hill_climb(
        &mut model,
        &NeverPositive,
        b"aa",
        b"ab",
        HillClimbConfig {
            max_model_queries: 10,
            seed: 1,
        },
    )
    .expect_err("NaN confidence is invalid");

    assert!(matches!(
        error,
        HillClimbError::InvalidConfidence {
            query_index: 1,
            confidence
        } if confidence.is_nan()
    ));
}

#[test]
fn empty_start_and_out_of_alphabet_symbols_are_rejected() {
    let mut model = InfallibleConfidence(|_candidate: &[u8]| 0.5);

    let empty = greedy_single_site_hill_climb(
        &mut model,
        &NeverPositive,
        b"",
        b"ab",
        HillClimbConfig {
            max_model_queries: 10,
            seed: 1,
        },
    );
    assert!(matches!(empty, Err(HillClimbError::EmptyStart)));

    let invalid = greedy_single_site_hill_climb(
        &mut model,
        &NeverPositive,
        b"ac",
        b"ab",
        HillClimbConfig {
            max_model_queries: 10,
            seed: 1,
        },
    );
    assert!(matches!(
        invalid,
        Err(HillClimbError::StartSymbolOutsideAlphabet(b'c'))
    ));
}
