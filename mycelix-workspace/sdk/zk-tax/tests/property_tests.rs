// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property-based tests using proptest.
//!
//! These tests verify invariants that should hold for all inputs.

use proptest::prelude::*;
use mycelix_zk_tax::{
    Jurisdiction, FilingStatus, TaxBracketProver, TaxBracketProof,
    brackets::{find_bracket, get_brackets},
    proof::{RangeProofBuilder, BatchProofBuilder, CompressedProof, CompressedProofType},
};

// =============================================================================
// Strategies
// =============================================================================

/// Strategy for generating valid incomes (0 to 10 million)
fn income_strategy() -> impl Strategy<Value = u64> {
    0u64..10_000_000
}

/// Strategy for generating high incomes
fn high_income_strategy() -> impl Strategy<Value = u64> {
    100_000u64..50_000_000
}

/// Strategy for generating valid tax years
fn tax_year_strategy() -> impl Strategy<Value = u32> {
    prop_oneof![
        Just(2020),
        Just(2021),
        Just(2022),
        Just(2023),
        Just(2024),
        Just(2025),
    ]
}

/// Strategy for generating jurisdictions
fn jurisdiction_strategy() -> impl Strategy<Value = Jurisdiction> {
    prop_oneof![
        Just(Jurisdiction::US),
        Just(Jurisdiction::UK),
        Just(Jurisdiction::DE),
        Just(Jurisdiction::CA),
        Just(Jurisdiction::AU),
        Just(Jurisdiction::JP),
        Just(Jurisdiction::FR),
    ]
}

/// Strategy for generating filing statuses
fn filing_status_strategy() -> impl Strategy<Value = FilingStatus> {
    prop_oneof![
        Just(FilingStatus::Single),
        Just(FilingStatus::MarriedFilingJointly),
        Just(FilingStatus::MarriedFilingSeparately),
        Just(FilingStatus::HeadOfHousehold),
    ]
}

// =============================================================================
// Bracket Property Tests
// =============================================================================

proptest! {
    /// Any valid income should find a bracket (no gaps).
    #[test]
    fn bracket_exists_for_any_income(income in income_strategy()) {
        let result = find_bracket(income, Jurisdiction::US, 2024, FilingStatus::Single);
        prop_assert!(result.is_ok(), "No bracket found for income: {}", income);
    }

    /// Bracket bounds should be consistent (lower < upper).
    #[test]
    fn bracket_bounds_are_valid(income in income_strategy()) {
        let bracket = find_bracket(income, Jurisdiction::US, 2024, FilingStatus::Single).unwrap();
        prop_assert!(bracket.lower <= bracket.upper, "Invalid bounds: {} > {}", bracket.lower, bracket.upper);
    }

    /// Income should actually be within its bracket.
    #[test]
    fn income_is_within_bracket(income in income_strategy()) {
        let bracket = find_bracket(income, Jurisdiction::US, 2024, FilingStatus::Single).unwrap();
        prop_assert!(income >= bracket.lower, "Income {} below lower bound {}", income, bracket.lower);
        prop_assert!(income < bracket.upper || bracket.upper == u64::MAX,
            "Income {} above upper bound {}", income, bracket.upper);
    }

    /// Brackets should be sorted by lower bound.
    #[test]
    fn brackets_are_sorted(year in tax_year_strategy()) {
        let brackets = get_brackets(Jurisdiction::US, year, FilingStatus::Single).unwrap();

        for window in brackets.windows(2) {
            prop_assert!(window[0].lower < window[1].lower,
                "Brackets not sorted: {} >= {}", window[0].lower, window[1].lower);
        }
    }

    /// Rate should increase or stay same for higher brackets (progressive taxation).
    #[test]
    fn rates_are_progressive(year in tax_year_strategy()) {
        let brackets = get_brackets(Jurisdiction::US, year, FilingStatus::Single).unwrap();

        for window in brackets.windows(2) {
            prop_assert!(window[0].rate_bps <= window[1].rate_bps,
                "Non-progressive rates: {}bps > {}bps", window[0].rate_bps, window[1].rate_bps);
        }
    }
}

// =============================================================================
// Proof Property Tests
// =============================================================================

proptest! {
    /// Any proof should verify successfully.
    #[test]
    fn proofs_always_verify(income in income_strategy(), year in tax_year_strategy()) {
        let prover = TaxBracketProver::dev_mode();
        let proof = prover.prove(income, Jurisdiction::US, FilingStatus::Single, year).unwrap();

        let result = proof.verify();
        prop_assert!(result.is_ok(), "Proof verification failed: {:?}", result.err());
    }

    /// Proof bracket index should be consistent.
    #[test]
    fn proof_bracket_matches_find_bracket(income in income_strategy()) {
        let prover = TaxBracketProver::dev_mode();
        let proof = prover.prove(income, Jurisdiction::US, FilingStatus::Single, 2024).unwrap();
        let bracket = find_bracket(income, Jurisdiction::US, 2024, FilingStatus::Single).unwrap();

        prop_assert_eq!(proof.bracket_index, bracket.index,
            "Bracket mismatch: proof={} vs find={}", proof.bracket_index, bracket.index);
    }

    /// Two proofs with same inputs should have same commitment.
    #[test]
    fn proof_commitment_is_deterministic(income in income_strategy()) {
        let prover = TaxBracketProver::dev_mode();
        let proof1 = prover.prove(income, Jurisdiction::US, FilingStatus::Single, 2024).unwrap();
        let proof2 = prover.prove(income, Jurisdiction::US, FilingStatus::Single, 2024).unwrap();

        prop_assert_eq!(proof1.commitment, proof2.commitment,
            "Non-deterministic commitment");
        prop_assert_eq!(proof1.bracket_index, proof2.bracket_index,
            "Non-deterministic bracket");
    }

    /// Different brackets should produce different commitments.
    #[test]
    fn different_brackets_different_commitments(
        // These incomes span different US tax brackets for 2024
        income1 in 10_000u64..20_000,  // 10% bracket
        income2 in 100_000u64..200_000  // 22% or 24% bracket
    ) {
        let prover = TaxBracketProver::dev_mode();

        let proof1 = prover.prove(income1, Jurisdiction::US, FilingStatus::Single, 2024);
        let proof2 = prover.prove(income2, Jurisdiction::US, FilingStatus::Single, 2024);

        if let (Ok(p1), Ok(p2)) = (proof1, proof2) {
            // Different brackets should produce different commitments
            prop_assert_ne!(p1.bracket_index, p2.bracket_index,
                "Incomes in same bracket - adjust test ranges");
            prop_assert_ne!(p1.commitment, p2.commitment,
                "Same commitment for different brackets");
        }
    }
}

// =============================================================================
// Compression Property Tests
// =============================================================================

proptest! {
    /// Compression and decompression should be lossless.
    #[test]
    fn compression_roundtrip(income in income_strategy()) {
        let prover = TaxBracketProver::dev_mode();
        let proof = prover.prove(income, Jurisdiction::US, FilingStatus::Single, 2024).unwrap();

        // Compress and decompress
        let compressed = CompressedProof::from_bracket_proof(&proof).unwrap();
        let decompressed = compressed.to_bracket_proof().unwrap();

        // Original should be recoverable
        prop_assert_eq!(proof.bracket_index, decompressed.bracket_index);
        prop_assert_eq!(proof.rate_bps, decompressed.rate_bps);
        prop_assert_eq!(proof.tax_year, decompressed.tax_year);
        prop_assert_eq!(proof.commitment, decompressed.commitment);
    }
}

// =============================================================================
// Range Proof Property Tests
// =============================================================================

proptest! {
    /// Range proofs should verify for incomes within range.
    #[test]
    fn range_proof_covers_income(
        income in 50_000u64..200_000,
        year in tax_year_strategy()
    ) {
        let proof = RangeProofBuilder::new(income, year)
            .prove_between(40_000, 250_000)
            .unwrap();

        // Verify income falls within the proven range
        prop_assert!(income >= proof.range_lower);
        prop_assert!(income <= proof.range_upper);
    }

    /// Range proof min should be <= max.
    #[test]
    fn range_proof_bounds_valid(
        min in 10_000u64..50_000,
        max in 100_000u64..500_000
    ) {
        let income = (min + max) / 2;
        let proof = RangeProofBuilder::new(income, 2024)
            .prove_between(min, max)
            .unwrap();

        prop_assert!(
            proof.range_lower <= proof.range_upper,
            "Invalid range: {} > {}", proof.range_lower, proof.range_upper
        );
    }
}

// =============================================================================
// Batch Proof Property Tests
// =============================================================================

proptest! {
    /// Batch proofs should contain the right number of years.
    #[test]
    fn batch_proof_year_count(
        income1 in income_strategy(),
        income2 in income_strategy(),
        income3 in income_strategy()
    ) {
        let proof = BatchProofBuilder::new(Jurisdiction::US, FilingStatus::Single)
            .add_year(2022, income1)
            .add_year(2023, income2)
            .add_year(2024, income3)
            .build_dev()
            .unwrap();

        prop_assert_eq!(proof.year_proofs.len(), 3, "Wrong number of year proofs");
    }

    /// Each year in batch should have valid bracket.
    #[test]
    fn batch_proof_brackets_valid(
        income1 in income_strategy(),
        income2 in income_strategy()
    ) {
        let proof = BatchProofBuilder::new(Jurisdiction::US, FilingStatus::Single)
            .add_year(2023, income1)
            .add_year(2024, income2)
            .build_dev()
            .unwrap();

        for (i, year_proof) in proof.year_proofs.iter().enumerate() {
            prop_assert!(year_proof.bracket_index < 10,
                "Invalid bracket {} at index {}", year_proof.bracket_index, i);
        }
    }
}

// =============================================================================
// Jurisdiction Property Tests
// =============================================================================

proptest! {
    /// All jurisdictions should have valid brackets.
    #[test]
    fn all_jurisdictions_have_brackets(
        jurisdiction in jurisdiction_strategy(),
        year in tax_year_strategy()
    ) {
        let result = get_brackets(jurisdiction, year, FilingStatus::Single);
        prop_assert!(result.is_ok(),
            "No brackets for {:?} in {}: {:?}", jurisdiction, year, result.err());

        let brackets = result.unwrap();
        prop_assert!(!brackets.is_empty(), "Empty brackets for {:?}", jurisdiction);
    }

    /// All jurisdictions should produce valid proofs.
    #[test]
    fn all_jurisdictions_produce_proofs(
        jurisdiction in jurisdiction_strategy(),
        income in income_strategy()
    ) {
        let prover = TaxBracketProver::dev_mode();
        let result = prover.prove(income, jurisdiction, FilingStatus::Single, 2024);

        prop_assert!(result.is_ok(),
            "Failed to prove for {:?}: {:?}", jurisdiction, result.err());
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

proptest! {
    /// Zero income should work.
    #[test]
    fn zero_income_works(year in tax_year_strategy()) {
        let prover = TaxBracketProver::dev_mode();
        let result = prover.prove(0, Jurisdiction::US, FilingStatus::Single, year);
        prop_assert!(result.is_ok());
    }

    /// Very high income should find the top bracket.
    #[test]
    fn very_high_income_finds_bracket(income in high_income_strategy()) {
        let prover = TaxBracketProver::dev_mode();
        let result = prover.prove(income, Jurisdiction::US, FilingStatus::Single, 2024);
        prop_assert!(result.is_ok());
    }
}

// =============================================================================
// Serialization Property Tests
// =============================================================================

proptest! {
    /// Proofs should serialize and deserialize correctly.
    #[test]
    fn proof_serialization_roundtrip(income in income_strategy()) {
        let prover = TaxBracketProver::dev_mode();
        let proof = prover.prove(income, Jurisdiction::US, FilingStatus::Single, 2024).unwrap();

        let json = serde_json::to_string(&proof).unwrap();
        let deserialized: TaxBracketProof = serde_json::from_str(&json).unwrap();

        prop_assert_eq!(proof.bracket_index, deserialized.bracket_index);
        prop_assert_eq!(proof.rate_bps, deserialized.rate_bps);
        prop_assert_eq!(proof.commitment, deserialized.commitment);
    }
}
