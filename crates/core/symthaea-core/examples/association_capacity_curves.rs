// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic association-capacity curve for HDC record bundles.
//!
//! This is a measurement example, not a unit test or a performance claim.
//! It measures role/filler retrieval as the number of superposed associations grows,
//! and emits unrelated-pair scores for later threshold calibration. The rows retain
//! each seed so downstream analysis can estimate uncertainty without losing replicates.
//!
//! Run:
//!   cargo run --release --example association_capacity_curves -p symthaea-core -- /tmp/association-capacity.csv
//!
//! BinaryHV has a fixed 16,384-bit representation. ContinuousHV's bipolar-f32 control
//! sweeps dimensions. It is deliberately initialized with {-1,+1} values so both arms
//! have reversible multiplication/XOR binding semantics; this does not characterize
//! native uniform[-1,1] ContinuousHV algebra or the dynamical LTC path.
//!
//! For a dimension/fan-in pair, one (role, filler) association is designated as the
//! query target. The record bundles all bound role/filler pairs. Unbinding by the target
//! role yields a noisy query; success requires the target filler to score strictly above
//! every distractor filler. A tie is a miss. Negative-pair scores are independent random
//! vectors and are not a calibrated tail guarantee at extreme index sizes.

use std::fmt::Write as _;
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::unified_hv::ContinuousHV;

const CONTINUOUS_DIMS: &[usize] = &[1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536];
const BUNDLE_FAN_INS: &[usize] = &[1, 3, 7, 15, 31, 63];
const SEEDS: u64 = 12;
const NEGATIVE_PAIRS_PER_SEED: usize = 32;

fn sub_seed(root: u64, dimension: usize, slot: usize, domain: u64) -> u64 {
    root.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((dimension as u64).wrapping_mul(0xD1B5_4A32_D192_ED03))
        .wrapping_add((slot as u64).wrapping_mul(0x94D0_49BB_1331_11EB))
        .wrapping_add(domain)
}

/// Deterministic bipolar float hypervector so binding is self-inverse for this control.
fn bipolar_continuous(dim: usize, seed: u64) -> ContinuousHV {
    let raw = ContinuousHV::random(dim, seed);
    let values = raw
        .values
        .into_iter()
        .map(|value| if value >= 0.0 { 1.0_f32 } else { -1.0_f32 })
        .collect();
    ContinuousHV::from_values(values)
}

fn append_continuous_bundle_row(out: &mut String, dim: usize, fan_in: usize, seed: u64) {
    let mut pairs = Vec::with_capacity(fan_in);
    for slot in 0..fan_in {
        let role = bipolar_continuous(
            dim,
            sub_seed(seed, dim, slot, 0xC011_0001),
        );
        let filler = bipolar_continuous(
            dim,
            sub_seed(seed, dim, slot, 0xC011_0002),
        );
        let bound = role.bind(&filler);
        pairs.push((role, filler, bound));
    }

    let bound_refs: Vec<&ContinuousHV> = pairs.iter().map(|pair| &pair.2).collect();
    let record = ContinuousHV::bundle(&bound_refs);
    let query = record.bind(&pairs[0].0);
    let target_score = query.similarity(&pairs[0].1) as f64;
    let best_distractor = pairs
        .iter()
        .skip(1)
        .map(|pair| query.similarity(&pair.1) as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let hit = fan_in == 1 || target_score > best_distractor;
    let best_distractor_field = if fan_in == 1 {
        String::new()
    } else {
        format!("{best_distractor:.8}")
    };
    let margin_field = if fan_in == 1 {
        String::new()
    } else {
        format!("{:.8}", target_score - best_distractor)
    };
    writeln!(
        out,
        "bundle_retrieval,continuous_f32_bipolar,{dim},{fan_in},{seed},0,{target_score:.8},{best_distractor_field},{margin_field},{hit},"
    )
    .unwrap();
}

fn append_binary_bundle_row(out: &mut String, fan_in: usize, seed: u64) {
    let dim = BinaryHV::DIM;
    let mut pairs = Vec::with_capacity(fan_in);
    for slot in 0..fan_in {
        let role = BinaryHV::random(sub_seed(seed, dim, slot, 0xB1A0_0001));
        let filler = BinaryHV::random(sub_seed(seed, dim, slot, 0xB1A0_0002));
        let bound = role.bind(&filler);
        pairs.push((role, filler, bound));
    }

    let bound_vectors: Vec<BinaryHV> = pairs.iter().map(|pair| pair.2).collect();
    let record = BinaryHV::bundle_safe(&bound_vectors);
    let query = record.bind(&pairs[0].0);
    let target_score = query.similarity(&pairs[0].1) as f64;
    let best_distractor = pairs
        .iter()
        .skip(1)
        .map(|pair| query.similarity(&pair.1) as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let hit = fan_in == 1 || target_score > best_distractor;
    let best_distractor_field = if fan_in == 1 {
        String::new()
    } else {
        format!("{best_distractor:.8}")
    };
    let margin_field = if fan_in == 1 {
        String::new()
    } else {
        format!("{:.8}", target_score - best_distractor)
    };
    writeln!(
        out,
        "bundle_retrieval,binary_bsc_xor_majority,{dim},{fan_in},{seed},0,{target_score:.8},{best_distractor_field},{margin_field},{hit},"
    )
    .unwrap();
}

fn append_continuous_negative_rows(out: &mut String, dim: usize, seed: u64) {
    for pair_index in 0..NEGATIVE_PAIRS_PER_SEED {
        let a = bipolar_continuous(
            dim,
            sub_seed(seed, dim, pair_index, 0xC011_1001),
        );
        let b = bipolar_continuous(
            dim,
            sub_seed(seed, dim, pair_index, 0xC011_1002),
        );
        let score = a.similarity(&b);
        writeln!(
            out,
            "unrelated_pair,continuous_f32_bipolar,{dim},,{seed},{pair_index},,,,,{score:.8}"
        )
        .unwrap();
    }
}

fn append_binary_negative_rows(out: &mut String, seed: u64) {
    let dim = BinaryHV::DIM;
    for pair_index in 0..NEGATIVE_PAIRS_PER_SEED {
        let a = BinaryHV::random(sub_seed(seed, dim, pair_index, 0xB1A0_1001));
        let b = BinaryHV::random(sub_seed(seed, dim, pair_index, 0xB1A0_1002));
        let score = a.cosine_similarity(&b);
        writeln!(
            out,
            "unrelated_pair,binary_bsc_xor_majority,{dim},,{seed},{pair_index},,,,,{score:.8}"
        )
        .unwrap();
    }
}

fn main() {
    let mut csv = String::from(
        "record_kind,representation,dimension,bundle_fan_in,seed,sample_index,target_similarity,best_distractor_similarity,margin,top1_hit,unrelated_pair_similarity\n",
    );

    for &dim in CONTINUOUS_DIMS {
        for seed in 0..SEEDS {
            for &fan_in in BUNDLE_FAN_INS {
                append_continuous_bundle_row(&mut csv, dim, fan_in, seed);
            }
            append_continuous_negative_rows(&mut csv, dim, seed);
        }
    }

    for seed in 0..SEEDS {
        for &fan_in in BUNDLE_FAN_INS {
            append_binary_bundle_row(&mut csv, fan_in, seed);
        }
        append_binary_negative_rows(&mut csv, seed);
    }

    let output_path = std::env::args().nth(1);
    if let Some(path) = output_path {
        std::fs::write(&path, csv).unwrap_or_else(|error| {
            panic!("failed to write CSV output to {path}: {error}");
        });
        eprintln!("Association-capacity raw rows written to {path}");
    } else {
        print!("{csv}");
    }
}
