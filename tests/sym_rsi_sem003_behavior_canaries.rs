// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#![cfg(any(feature = "full_consciousness", feature = "magi_loop"))]

use symthaea::consciousness::recursive_improvement::{
    semantic_context::SemanticContextEncoder,
    sym_rsi_semantic_set_experiment::{
        Sem3Disposition, Sem3SetSizeBin, run_sym_rsi_sem_003,
    },
};
use symthaea_core::hdc::BinaryHV;

const PRIMARY_SUBJECT: &str = "dd533164f8aff636f06e49a161fa75b4e70359d1";

#[test]
fn sem3_hdc_primitives_match_independent_known_answers() {
    let random = BinaryHV::random(42);
    assert_eq!(
        &random.0[..16],
        &[
            0xfa, 0xe6, 0x24, 0xa6, 0xc2, 0xdc, 0xaa, 0x94, 0x6e, 0xc8, 0x1b, 0xbe, 0xe9, 0xd0,
            0xee, 0x5c,
        ]
    );
    assert_eq!(
        blake3::hash(&random.0).as_bytes(),
        &[
            0x64, 0x6e, 0xe4, 0xd3, 0x3e, 0x5f, 0xad, 0x0e, 0x9f, 0x58, 0xb2, 0x7e, 0xba, 0x6c,
            0xfc, 0x2f, 0x78, 0xac, 0xda, 0xd6, 0xa3, 0x4c, 0x1b, 0x94, 0x14, 0x22, 0x60, 0x51,
            0x92, 0x98, 0x7b, 0xfc,
        ]
    );

    let encoded = SemanticContextEncoder::default()
        .encode(&[0.2, -0.4, 0.8])
        .expect("known-answer semantic context must encode");
    assert_eq!(
        blake3::hash(&encoded.0).as_bytes(),
        &[
            0xac, 0x8e, 0x10, 0x0f, 0x2c, 0xba, 0x1f, 0xd6, 0x90, 0xcb, 0x6a, 0x07, 0xc3, 0x66,
            0xe4, 0xf5, 0x11, 0x01, 0x9c, 0x9e, 0xa7, 0x3b, 0x20, 0x86, 0xd5, 0x6f, 0x15, 0xab,
            0xa7, 0xe3, 0x1d, 0x90,
        ]
    );
}

#[test]
fn sem3_public_runner_matches_independent_behavior_canaries() {
    let receipt = run_sym_rsi_sem_003(PRIMARY_SUBJECT)
        .expect("replacement SEM-003 candidate should execute the frozen synthetic benchmark");

    let expected_thresholds = [
        (4usize, 0x3b50_0000u32),
        (8usize, 0x3b20_0000u32),
        (16usize, 0x3af0_0000u32),
    ];
    for (dimension, expected_bits) in expected_thresholds {
        let threshold = receipt
            .thresholds
            .iter()
            .find(|threshold| threshold.context_dimension == dimension)
            .expect("every frozen dimension must have a threshold");
        assert_eq!(threshold.calibration_count, 32);
        assert_eq!(threshold.nonconformity_threshold.to_bits(), expected_bits);
    }

    assert_eq!(receipt.clean_target_coverage, 46.0 / 96.0);
    assert_eq!(receipt.clean_mean_set_size, 46.0 / 96.0);
    assert_eq!(receipt.ambiguous_at_least_one_parent_inclusion, 0.0);
    assert_eq!(receipt.ambiguous_dual_parent_inclusion, 0.0);
    assert_eq!(receipt.ambiguous_singleton_forced_choice_rate, 0.0);
    assert_eq!(receipt.unrelated_nonempty_set_rate, 0.0);
    assert_eq!(receipt.ood_support_rejection_rate, 1.0);

    let expected_clean_coverage = [
        (4usize, 8.0 / 32.0),
        (8usize, 20.0 / 32.0),
        (16usize, 18.0 / 32.0),
    ];
    for (dimension, expected_coverage) in expected_clean_coverage {
        let metrics = receipt
            .dimension_metrics
            .iter()
            .find(|metrics| metrics.context_dimension == dimension)
            .expect("every frozen dimension must have metrics");
        assert_eq!(metrics.clean_target_coverage, expected_coverage);
        assert_eq!(metrics.clean_mean_set_size, expected_coverage);
        assert_eq!(metrics.ambiguous_at_least_one_parent_inclusion, 0.0);
        assert_eq!(metrics.ambiguous_dual_parent_inclusion, 0.0);
        assert_eq!(metrics.ambiguous_singleton_forced_choice_rate, 0.0);
        assert_eq!(metrics.ambiguous_mean_set_size, 0.0);
        assert_eq!(metrics.unrelated_nonempty_set_rate, 0.0);
        assert_eq!(metrics.ood_support_rejection_rate, 1.0);
    }

    assert_eq!(
        receipt.clean_set_size_histogram,
        vec![
            Sem3SetSizeBin {
                set_size: 0,
                count: 50,
            },
            Sem3SetSizeBin {
                set_size: 1,
                count: 46,
            },
        ]
    );
    assert_eq!(
        receipt.ambiguous_set_size_histogram,
        vec![Sem3SetSizeBin {
            set_size: 0,
            count: 72,
        }]
    );
    assert_eq!(
        receipt.unrelated_set_size_histogram,
        vec![Sem3SetSizeBin {
            set_size: 0,
            count: 72,
        }]
    );

    assert!(!receipt.clean_coverage_passed);
    assert!(!receipt.per_dimension_clean_coverage_passed);
    assert!(receipt.clean_set_size_guard_passed);
    assert!(!receipt.ambiguous_parent_guard_passed);
    assert!(!receipt.ambiguous_dual_parent_guard_passed);
    assert!(receipt.ambiguous_singleton_guard_passed);
    assert!(receipt.unrelated_guard_passed);
    assert!(receipt.ood_support_guard_passed);
    assert_eq!(receipt.disposition, Sem3Disposition::NotEstablished);
}
