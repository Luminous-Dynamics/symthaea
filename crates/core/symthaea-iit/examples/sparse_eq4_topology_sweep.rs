// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Topology-sensitive sparse IIT 4.0 Eq. 4 inference benchmark.
//!
//! Run:
//! cargo run --release --locked -p symthaea-iit --example sparse_eq4_topology_sweep
//! cargo run --release --locked -p symthaea-iit --example sparse_eq4_topology_sweep -- --repeats 9
//!
//! The ring and hub-spoke families isolate substrate-size effects at low
//! induced width. The open grid family exposes the growth in induced width
//! even when every local CPT has at most three parents. A fixed four-unit
//! candidate is evaluated in a growing substrate; substrate size is not
//! reported as candidate-system size.
//!
//! This is an exact sparse Eq. 4 primitive benchmark, not a complete IIT
//! cause-effect structure benchmark, not an external PyPhi comparison, and
//! not evidence of a larger integrated system. Every successful repetition
//! must reproduce the exact report receipt before its timing is emitted.

use std::time::{Duration, Instant};

use symthaea_iit::{
    FactorGraphError, SparseBinaryCausalModel, SparseCausalError, SparseMechanismCpt,
};

const FACTOR_ENTRY_BUDGET: usize = 1 << 16;
const FIXED_CANDIDATE_UNITS: usize = 4;

#[derive(Debug, Clone, Copy)]
enum Topology {
    Ring,
    HubSpoke,
    OpenGrid,
}

impl Topology {
    fn label(self) -> &'static str {
        match self {
            Self::Ring => "ring",
            Self::HubSpoke => "hub_spoke",
            Self::OpenGrid => "open_grid",
        }
    }
}

struct Fixture {
    topology: Topology,
    substrate_units: usize,
    candidate: Vec<usize>,
}

fn local_cpt(target: usize, mut parents: Vec<usize>) -> Result<SparseMechanismCpt, String> {
    parents.sort_unstable();
    parents.dedup();
    if !parents.contains(&target) {
        parents.push(target);
        parents.sort_unstable();
    }
    if parents.len() > 3 {
        return Err(format!("fixture generated {} parents for target {target}", parents.len()));
    }

    let rows = 1usize
        .checked_shl(parents.len() as u32)
        .ok_or_else(|| format!("local CPT size overflow at target {target}"))?;
    let mut probabilities = Vec::with_capacity(rows);
    for assignment in 0..rows {
        let mut probability = 0.12;
        for (slot, parent) in parents.iter().copied().enumerate() {
            if ((assignment >> slot) & 1) != 0 {
                probability += if parent == target { 0.28 } else { 0.12 };
            }
        }
        if !(0.0..=1.0).contains(&probability) {
            return Err(format!("invalid generated probability at target {target}: {probability}"));
        }
        probabilities.push(probability);
    }

    SparseMechanismCpt::new(target, parents, probabilities)
        .map_err(|error| error.to_string())
}

fn build_model(topology: Topology, n_units: usize) -> Result<(SparseBinaryCausalModel, Vec<usize>), String> {
    if n_units < FIXED_CANDIDATE_UNITS {
        return Err(format!("topology {} needs at least four substrate units", topology.label()));
    }

    let (mechanisms, candidate) = match topology {
        Topology::Ring => {
            let mechanisms = (0..n_units)
                .map(|target| {
                    let previous = (target + n_units - 1) % n_units;
                    local_cpt(target, vec![target, previous])
                })
                .collect::<Result<Vec<_>, _>>()?;
            (mechanisms, vec![0, 1, 2, 3])
        }
        Topology::HubSpoke => {
            let mechanisms = (0..n_units)
                .map(|target| {
                    if target == 0 {
                        local_cpt(target, vec![target])
                    } else {
                        local_cpt(target, vec![0, target])
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            (mechanisms, vec![0, 1, 2, 3])
        }
        Topology::OpenGrid => {
            let side = integer_square_root(n_units);
            if side * side != n_units || side < 2 {
                return Err(format!("open_grid substrate size must be a square >= 4; got {n_units}"));
            }
            let mut mechanisms = Vec::with_capacity(n_units);
            for row in 0..side {
                for column in 0..side {
                    let target = row * side + column;
                    let mut parents = vec![target];
                    if row > 0 {
                        parents.push((row - 1) * side + column);
                    }
                    if column > 0 {
                        parents.push(row * side + column - 1);
                    }
                    mechanisms.push(local_cpt(target, parents)?);
                }
            }
            (mechanisms, vec![0, 1, side, side + 1])
        }
    };

    let model = SparseBinaryCausalModel::new(n_units, mechanisms)
        .map_err(|error| error.to_string())?;
    Ok((model, candidate))
}

fn integer_square_root(value: usize) -> usize {
    let mut root = 0usize;
    while (root + 1).checked_mul(root + 1).is_some_and(|square| square <= value) {
        root += 1;
    }
    root
}

fn deterministic_state(n_units: usize) -> Vec<bool> {
    (0..n_units)
        .map(|unit| (unit.wrapping_mul(13).wrapping_add(7)) % 11 < 5)
        .collect()
}

fn median(samples: &[Duration]) -> Duration {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

fn percentile_95(samples: &[Duration]) -> Duration {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let rank = (sorted.len() * 95).div_ceil(100).max(1);
    sorted[rank - 1]
}

fn parse_repeats() -> Result<usize, String> {
    let mut repeats = 5usize;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--repeats" => {
                let value = args.next().ok_or("--repeats requires a positive integer")?;
                repeats = value.parse::<usize>()
                    .map_err(|_| format!("invalid repeat count: {value}"))?;
                if !(1..=100).contains(&repeats) {
                    return Err("--repeats must be in 1..=100".to_string());
                }
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(repeats)
}

fn fixture_set() -> Vec<Fixture> {
    let mut fixtures = Vec::new();
    for n_units in [64usize, 256, 512, 1_024, 2_048] {
        fixtures.push(Fixture {
            topology: Topology::Ring,
            substrate_units: n_units,
            candidate: vec![0, 1, 2, 3],
        });
        fixtures.push(Fixture {
            topology: Topology::HubSpoke,
            substrate_units: n_units,
            candidate: vec![0, 1, 2, 3],
        });
    }
    for side in [4usize, 8, 12, 16] {
        fixtures.push(Fixture {
            topology: Topology::OpenGrid,
            substrate_units: side * side,
            candidate: vec![0, 1, side, side + 1],
        });
    }
    fixtures
}

fn run_fixture(fixture: &Fixture, repeats: usize) -> Result<(), String> {
    let build_start = Instant::now();
    let (model, candidate) = build_model(fixture.topology, fixture.substrate_units)?;
    if candidate != fixture.candidate {
        return Err("fixture candidate differs from model builder candidate".to_string());
    }
    let model_build_ns = build_start.elapsed().as_nanos();

    let state = deterministic_state(fixture.substrate_units);
    let warmup = match model.forward_cause_likelihood_iit4(
        &candidate,
        &state,
        FACTOR_ENTRY_BUDGET,
    ) {
        Ok(report) => report,
        Err(SparseCausalError::FactorGraph(FactorGraphError::IntermediateTooLarge { entries, limit })) => {
            let evidence = format!("required_entries={entries};limit={limit}");
            println!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                fixture.topology.label(),
                fixture.substrate_units,
                candidate.len(),
                repeats,
                model_build_ns,
                FACTOR_ENTRY_BUDGET,
                "budget_rejected",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                evidence,
            );
            return Ok(());
        }
        Err(error) => return Err(format!(
            "{} n={}: sparse Eq. 4 failed: {error}",
            fixture.topology.label(),
            fixture.substrate_units
        )),
    };

    let receipt = warmup.receipt_sha256().map_err(|error| error.to_string())?;
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        let report = model.forward_cause_likelihood_iit4(
            &candidate,
            &state,
            FACTOR_ENTRY_BUDGET,
        ).map_err(|error| error.to_string())?;
        samples.push(start.elapsed());
        let current_receipt = report.receipt_sha256().map_err(|error| error.to_string())?;
        if current_receipt != receipt {
            return Err(format!(
                "nondeterministic Eq. 4 report for {} n={}",
                fixture.topology.label(),
                fixture.substrate_units
            ));
        }
    }

    let p50_ns = median(&samples).as_nanos();
    let p95_ns = percentile_95(&samples).as_nanos();
    let min_ns = samples.iter().map(Duration::as_nanos).min().unwrap_or_default();
    let forward_sum: f64 = warmup.forward_likelihood.iter().sum();
    let finite = warmup.forward_likelihood.iter().all(|value| value.is_finite());
    let evidence = format!("receipt={receipt};forward_sum={forward_sum:.17};finite={finite}");

    println!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        fixture.topology.label(),
        fixture.substrate_units,
        candidate.len(),
        repeats,
        model_build_ns,
        FACTOR_ENTRY_BUDGET,
        "exact",
        warmup.posterior_induced_width,
        warmup.maximum_target_induced_width,
        warmup.predicted_peak_factor_entries,
        warmup.peak_factor_entries,
        warmup.factor_multiplications,
        p50_ns,
        p95_ns,
        min_ns,
        evidence,
    );
    if !finite || !forward_sum.is_finite() {
        return Err(format!("non-finite Eq. 4 output for {} n={}", fixture.topology.label(), fixture.substrate_units));
    }
    if warmup.peak_factor_entries > FACTOR_ENTRY_BUDGET {
        return Err(format!("reported peak exceeds budget for {} n={}", fixture.topology.label(), fixture.substrate_units));
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let repeats = parse_repeats()?;
    println!(
        "topology,substrate_units,candidate_units,repeats,model_build_ns,max_factor_entries,status,posterior_induced_width,target_induced_width,predicted_peak_entries,observed_peak_entries,factor_multiplications,p50_ns,p95_ns,min_ns,evidence"
    );
    for fixture in fixture_set() {
        run_fixture(&fixture, repeats)?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("sparse Eq. 4 topology sweep failed: {error}");
        std::process::exit(1);
    }
}
