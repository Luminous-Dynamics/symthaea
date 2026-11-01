// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded default macro-mapping-family enumeration sweep.
//!
//! This measures candidate-table generation and deduplication only. It is not
//! macro TPM construction, unithood evaluation, cross-grain exclusion, or phi_s.
//!
//! Run with:
//! cargo run --release --locked -p symthaea-iit --example macro_mapping_family_sweep

use std::time::{Duration, Instant};

use symthaea_iit::{
    MacroMappingSearchLimits, MacroUnitError, MacroUnitSpec,
};

#[derive(Clone, Copy)]
struct Fixture {
    constituents: usize,
    update_grain: usize,
}

const FIXTURES: [Fixture; 8] = [
    Fixture { constituents: 1, update_grain: 1 },
    Fixture { constituents: 2, update_grain: 1 },
    Fixture { constituents: 3, update_grain: 1 },
    Fixture { constituents: 4, update_grain: 1 },
    Fixture { constituents: 2, update_grain: 2 },
    Fixture { constituents: 3, update_grain: 2 },
    Fixture { constituents: 4, update_grain: 2 },
    Fixture { constituents: 4, update_grain: 3 },
];

const LIMITS: MacroMappingSearchLimits = MacroMappingSearchLimits {
    max_candidate_upper_bound: 128,
    max_total_mapping_entries: 65_536,
    max_mapping_entries: 4_096,
};
const REPEATS: usize = 5;

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn run() -> Result<(), String> {
    println!(
        "constituents,update_grain,mapping_entries,raw_coarse_candidates,raw_blackbox_candidates,candidate_upper_bound,unique_candidates,duplicates_removed,total_unique_mapping_entries,median_ns,min_ns,candidate_set_sha256,validated"
    );
    for fixture in FIXTURES {
        let constituents = (0..fixture.constituents).collect::<Vec<_>>();
        let warmup = MacroUnitSpec::enumerate_default_mapping_families(
            constituents.clone(),
            fixture.update_grain,
            Vec::new(),
            LIMITS,
        )
        .map_err(|error| format!("enumeration failed: {error}"))?;

        let mut samples = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let start = Instant::now();
            let current = MacroUnitSpec::enumerate_default_mapping_families(
                constituents.clone(),
                fixture.update_grain,
                Vec::new(),
                LIMITS,
            )
            .map_err(|error| format!("repeated enumeration failed: {error}"))?;
            samples.push(start.elapsed());
            if current != warmup {
                return Err(format!(
                    "candidate set changed between repetitions for n={} tau={}",
                    fixture.constituents, fixture.update_grain
                ));
            }
        }
        let elapsed = median(&mut samples);
        let min = samples.iter().copied().min().unwrap_or_default();
        println!(
            "{},{},{},{},{},{},{},{},{},{},{},{},true",
            fixture.constituents,
            fixture.update_grain,
            1usize << (fixture.constituents * fixture.update_grain),
            warmup.generated_coarse_grain,
            warmup.generated_blackbox,
            warmup.candidate_upper_bound,
            warmup.candidates.len(),
            warmup.duplicates_removed,
            warmup.total_mapping_entries,
            elapsed.as_nanos(),
            min.as_nanos(),
            warmup.candidate_set_sha256,
        );
    }

    let rejected = MacroUnitSpec::enumerate_default_mapping_families(
        (0..6).collect(),
        1,
        Vec::new(),
        MacroMappingSearchLimits {
            max_candidate_upper_bound: 128,
            max_total_mapping_entries: 65_536,
            max_mapping_entries: 4_096,
        },
    );
    match rejected {
        Err(MacroUnitError::CandidateBudgetExceeded {
            required: 189,
            limit: 128,
        }) => {
            println!(
                "6,1,64,126,63,189,,63,,0,0,,true"
            );
        }
        other => {
            return Err(format!(
                "candidate-budget negative control did not reject before allocation: {other:?}"
            ));
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("macro mapping-family sweep failed: {error}");
        std::process::exit(1);
    }
}
