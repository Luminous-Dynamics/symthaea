// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic software-oracle corpus for NEURO-SILICON-001B (#5808).
//!
//! This tool does not implement HDC algebra. It calls the canonical
//! `symthaea_core::hdc::binary_hv::BinaryHV` implementation and serializes
//! exact inputs/outputs for future RTL differential qualification.
//!
//! Claim boundary:
//! - generated fixture == canonical software-oracle observation;
//! - fixture agreement != FPGA hardware observation;
//! - fixture agreement != performance or energy advantage;
//! - fixture agreement != ASIC/fabricated-silicon qualification.

use serde::Serialize;
use std::fmt::Write as _;
use symthaea_core::hdc::binary_hv::BinaryHV;

const SCHEMA_VERSION: &str = "neuro-silicon-hdc-rtl-corpus-v1";
const ORACLE_SOURCE_SHA: &str = "458c7b98d81c64b9361e252f85ef9d45132e6682";
const BENCHMARK_CONTRACT_SHA: &str = "68370c5431c8843553380c70d21952db5696cb50";

#[derive(Debug, Serialize)]
struct Corpus {
    schema_version: &'static str,
    oracle_source_sha: &'static str,
    benchmark_contract_sha: &'static str,
    dimension_bits: usize,
    vector_bytes: usize,
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Serialize)]
struct Fixture {
    id: &'static str,
    operation: &'static str,
    input_hex: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    shift: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_vector_hex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_hamming_distance: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_matching_bits: Option<u32>,
    hardware_required_v1: bool,
    note: &'static str,
}

fn hv_hex(hv: &BinaryHV) -> String {
    let mut out = String::with_capacity(BinaryHV::BYTES * 2);
    for byte in hv.0 {
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

fn vector_fixture(
    id: &'static str,
    operation: &'static str,
    inputs: &[BinaryHV],
    shift: Option<usize>,
    output: BinaryHV,
    hardware_required_v1: bool,
    note: &'static str,
) -> Fixture {
    Fixture {
        id,
        operation,
        input_hex: inputs.iter().map(hv_hex).collect(),
        shift,
        expected_vector_hex: Some(hv_hex(&output)),
        expected_hamming_distance: None,
        expected_matching_bits: None,
        hardware_required_v1,
        note,
    }
}

fn distance_fixture(
    id: &'static str,
    a: BinaryHV,
    b: BinaryHV,
    hardware_required_v1: bool,
    note: &'static str,
) -> Fixture {
    let distance = a.hamming_distance(&b);
    Fixture {
        id,
        operation: "hamming_distance",
        input_hex: vec![hv_hex(&a), hv_hex(&b)],
        shift: None,
        expected_vector_hex: None,
        expected_hamming_distance: Some(distance),
        expected_matching_bits: Some(BinaryHV::DIM as u32 - distance),
        hardware_required_v1,
        note,
    }
}

fn build_corpus() -> Corpus {
    let zero = BinaryHV::zero();
    let ones = BinaryHV::ones();
    let a = BinaryHV::random(1);
    let b = BinaryHV::random(2);
    let c = BinaryHV::random(3);
    let d = BinaryHV::random(4);
    let e = BinaryHV::random(5);
    let f = BinaryHV::random(6);
    let g = BinaryHV::random(7);
    let h = BinaryHV::random(8);
    let i = BinaryHV::random(9);

    let bound_ab = a.bind(&b);
    let temporal_ab = a.bind_temporal(&b);
    let temporal_ba = b.bind_temporal(&a);

    let bind_chain_inputs = [&a, &b, &c, &d, &e, &f, &g, &h, &i];
    let bind_chain = BinaryHV::bind_chain(&bind_chain_inputs);

    let bundle3 = BinaryHV::bundle(&[a, b, c]);
    let bundle5 = BinaryHV::bundle(&[a, b, c, d, e]);
    let bundle9 = BinaryHV::bundle(&[a, b, c, d, e, f, g, h, i]);
    let bundle2_tie_sensitive = BinaryHV::bundle(&[a, b]);

    let fixtures = vec![
        vector_fixture(
            "bind-zero-ones",
            "bind",
            &[zero, ones],
            None,
            zero.bind(&ones),
            true,
            "Identity-pattern edge fixture.",
        ),
        vector_fixture(
            "bind-self-zero",
            "bind",
            &[a, a],
            None,
            a.bind(&a),
            true,
            "Canonical XOR self-inverse fixture; expected vector is all zero.",
        ),
        vector_fixture(
            "bind-seed-1-2",
            "bind",
            &[a, b],
            None,
            bound_ab,
            true,
            "Deterministic random-vector bind fixture.",
        ),
        vector_fixture(
            "unbind-recovery",
            "bind",
            &[bound_ab, a],
            None,
            bound_ab.bind(&a),
            true,
            "Binding the bound result with the first operand must recover the second operand.",
        ),
        vector_fixture(
            "permute-seed-1-shift-1",
            "permute",
            &[a],
            Some(1),
            a.permute(1),
            true,
            "One-step canonical permutation.",
        ),
        vector_fixture(
            "permute-wraparound",
            "permute",
            &[a],
            Some(BinaryHV::DIM),
            a.permute(BinaryHV::DIM),
            true,
            "Full-dimension shift must follow the canonical software wraparound semantics.",
        ),
        vector_fixture(
            "temporal-bind-a-b",
            "bind_temporal",
            &[a, b],
            Some(1),
            temporal_ab,
            true,
            "Canonical temporal bind is permute(a, 1) XOR b.",
        ),
        vector_fixture(
            "temporal-bind-b-a",
            "bind_temporal",
            &[b, a],
            Some(1),
            temporal_ba,
            true,
            "Paired fixture preserves the expected non-commutativity of temporal bind.",
        ),
        distance_fixture(
            "hamming-identical",
            a,
            a,
            true,
            "Identical vectors must have distance 0 and matching_bits == DIM.",
        ),
        distance_fixture(
            "hamming-zero-ones",
            zero,
            ones,
            true,
            "Opposite bit patterns must have distance == DIM.",
        ),
        distance_fixture(
            "hamming-seed-1-2",
            a,
            b,
            true,
            "Deterministic near-half random-vector distance fixture.",
        ),
        vector_fixture(
            "bundle-3",
            "bundle",
            &[a, b, c],
            None,
            bundle3,
            true,
            "Odd-N majority bundle fixture.",
        ),
        vector_fixture(
            "bundle-5",
            "bundle",
            &[a, b, c, d, e],
            None,
            bundle5,
            true,
            "Odd-N majority bundle fixture.",
        ),
        vector_fixture(
            "bundle-9",
            "bundle",
            &[a, b, c, d, e, f, g, h, i],
            None,
            bundle9,
            true,
            "Larger odd-N majority bundle fixture.",
        ),
        vector_fixture(
            "bundle-2-tie-sensitive",
            "bundle",
            &[a, b],
            None,
            bundle2_tie_sensitive,
            false,
            "Software-oracle tie behavior is frozen here before even-N RTL support is admitted.",
        ),
        vector_fixture(
            "bind-chain-9",
            "bind_chain",
            &[a, b, c, d, e, f, g, h, i],
            None,
            bind_chain,
            true,
            "Long-chain fixture used to qualify chunked/streaming execution equivalence.",
        ),
    ];

    Corpus {
        schema_version: SCHEMA_VERSION,
        oracle_source_sha: ORACLE_SOURCE_SHA,
        benchmark_contract_sha: BENCHMARK_CONTRACT_SHA,
        dimension_bits: BinaryHV::DIM,
        vector_bytes: BinaryHV::BYTES,
        fixtures,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let corpus = build_corpus();
    println!("{}", serde_json::to_string_pretty(&corpus)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_has_unique_fixture_ids() {
        let corpus = build_corpus();
        let mut ids: Vec<_> = corpus.fixtures.iter().map(|fixture| fixture.id).collect();
        let original_len = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), original_len);
    }

    #[test]
    fn all_serialized_vectors_have_exact_binary_hv_width() {
        let corpus = build_corpus();
        let expected_hex_len = BinaryHV::BYTES * 2;
        for fixture in corpus.fixtures {
            for input in fixture.input_hex {
                assert_eq!(input.len(), expected_hex_len, "{} input width", fixture.id);
            }
            if let Some(output) = fixture.expected_vector_hex {
                assert_eq!(output.len(), expected_hex_len, "{} output width", fixture.id);
            }
        }
    }

    #[test]
    fn temporal_bind_pair_is_non_commutative_for_frozen_fixture() {
        let a = BinaryHV::random(1);
        let b = BinaryHV::random(2);
        assert!(a.bind_temporal(&b) != b.bind_temporal(&a));
    }

    #[test]
    fn bind_chain_matches_repeated_canonical_bind() {
        let vectors: Vec<_> = (1..=9).map(BinaryHV::random).collect();
        let refs: Vec<_> = vectors.iter().collect();
        let chain = BinaryHV::bind_chain(&refs);

        let mut repeated = vectors[0];
        for vector in &vectors[1..] {
            repeated = repeated.bind(vector);
        }

        assert!(chain == repeated);
    }
}