// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Same-process comparison of legacy full-rescan min-fill and optimized exact min-fill.
//!
//! The legacy planner is frozen from the pre-leaf-fast-path implementation.
//! Both planners receive identical factor scopes. A row is emitted only after
//! exact elimination-order and induced-width parity has been established.
//!
//! Run with:
//! cargo run --release --locked -p symthaea-iit --example min_fill_planner_comparison_sweep

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::time::{Duration, Instant};

use symthaea_iit::plan_min_fill_elimination;

#[derive(Clone, Copy)]
struct Fixture {
    topology: &'static str,
    variables: usize,
    width: usize,
    height: usize,
    repeats: usize,
}

const FIXTURES: [Fixture; 13] = [
    Fixture { topology: "star", variables: 64, width: 0, height: 0, repeats: 5 },
    Fixture { topology: "star", variables: 128, width: 0, height: 0, repeats: 5 },
    Fixture { topology: "star", variables: 256, width: 0, height: 0, repeats: 5 },
    Fixture { topology: "star", variables: 512, width: 0, height: 0, repeats: 3 },
    Fixture { topology: "star", variables: 1_024, width: 0, height: 0, repeats: 3 },
    Fixture { topology: "star", variables: 2_048, width: 0, height: 0, repeats: 3 },
    Fixture { topology: "chain", variables: 256, width: 0, height: 0, repeats: 5 },
    Fixture { topology: "chain", variables: 1_024, width: 0, height: 0, repeats: 3 },
    Fixture { topology: "ring", variables: 128, width: 0, height: 0, repeats: 5 },
    Fixture { topology: "ring", variables: 512, width: 0, height: 0, repeats: 3 },
    Fixture { topology: "grid", variables: 64, width: 8, height: 8, repeats: 5 },
    Fixture { topology: "grid", variables: 144, width: 12, height: 12, repeats: 3 },
    Fixture { topology: "grid", variables: 196, width: 14, height: 14, repeats: 1 },
];

// Planner-only comparison: no factor contraction is executed, so use the
// largest representable entry ceiling to obtain exact orders even for grids
// that production inference correctly rejects under its configured budget.
const PLANNER_ONLY_ENTRY_CEILING: usize = usize::MAX;

fn build_scopes(fixture: Fixture) -> Result<Vec<Vec<usize>>, String> {
    let mut scopes = Vec::new();
    let mut add_edge = |left: usize, right: usize| {
        scopes.push(if left < right {
            vec![left, right]
        } else {
            vec![right, left]
        });
    };
    match fixture.topology {
        "star" => {
            for leaf in 1..fixture.variables {
                add_edge(0, leaf);
            }
        }
        "chain" => {
            for variable in 0..fixture.variables.saturating_sub(1) {
                add_edge(variable, variable + 1);
            }
        }
        "ring" => {
            if fixture.variables < 3 {
                return Err("ring requires at least three variables".to_string());
            }
            for variable in 0..fixture.variables {
                add_edge(variable, (variable + 1) % fixture.variables);
            }
        }
        "grid" => {
            if fixture.width == 0
                || fixture.height == 0
                || fixture.width.checked_mul(fixture.height) != Some(fixture.variables)
            {
                return Err("grid dimensions do not match variable count".to_string());
            }
            for row in 0..fixture.height {
                for column in 0..fixture.width {
                    let here = row * fixture.width + column;
                    if column + 1 < fixture.width {
                        add_edge(here, here + 1);
                    }
                    if row + 1 < fixture.height {
                        add_edge(here, here + fixture.width);
                    }
                }
            }
        }
        other => return Err(format!("unknown topology: {other}")),
    }
    Ok(scopes)
}

/// Frozen pre-optimization planner. Keep this deliberately straightforward: it
/// is an independent performance baseline, not production code.
fn legacy_full_rescan_order(
    factor_scopes: &[Vec<usize>],
    eliminate: &[usize],
) -> Result<(Vec<usize>, usize), String> {
    let mut adjacency: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for variable in eliminate.iter().copied() {
        adjacency.entry(variable).or_default();
    }
    for scope in factor_scopes {
        for variable in scope.iter().copied() {
            adjacency.entry(variable).or_default();
        }
        for (left_index, left) in scope.iter().copied().enumerate() {
            for right in scope.iter().copied().skip(left_index + 1) {
                adjacency.entry(left).or_default().insert(right);
                adjacency.entry(right).or_default().insert(left);
            }
        }
    }

    let mut active: BTreeSet<usize> = eliminate.iter().copied().collect();
    let mut versions = BTreeMap::<usize, u64>::new();
    let mut queue = BinaryHeap::<Reverse<(usize, usize, usize, u64)>>::new();
    for variable in active.iter().copied() {
        versions.insert(variable, 0);
        let (fill, degree) = legacy_fill_score(variable, &adjacency);
        queue.push(Reverse((fill, degree, variable, 0)));
    }

    let mut order = Vec::with_capacity(active.len());
    let mut width = 0usize;
    while !active.is_empty() {
        let variable = loop {
            let Reverse((_, degree, variable, version)) =
                queue.pop().ok_or("legacy planner queue unexpectedly empty")?;
            if active.contains(&variable)
                && versions.get(&variable).copied() == Some(version)
            {
                width = width.max(degree);
                break variable;
            }
        };

        let neighbours = adjacency
            .get(&variable)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|neighbour| adjacency.contains_key(neighbour))
            .collect::<Vec<_>>();

        let mut affected = BTreeSet::new();
        let mut neighbour_hits = BTreeMap::<usize, usize>::new();
        for neighbour in &neighbours {
            if active.contains(neighbour) {
                affected.insert(*neighbour);
            }
            if let Some(edges) = adjacency.get(neighbour) {
                for candidate in edges.iter().copied() {
                    if candidate != variable && active.contains(&candidate) {
                        let hits = neighbour_hits.entry(candidate).or_default();
                        *hits += 1;
                    }
                }
            }
        }
        for (candidate, hits) in neighbour_hits {
            if hits >= 2 {
                affected.insert(candidate);
            }
        }

        for (left_index, left) in neighbours.iter().copied().enumerate() {
            for right in neighbours.iter().copied().skip(left_index + 1) {
                adjacency.entry(left).or_default().insert(right);
                adjacency.entry(right).or_default().insert(left);
            }
        }
        for neighbour in &neighbours {
            if let Some(edges) = adjacency.get_mut(neighbour) {
                edges.remove(&variable);
            }
        }
        adjacency.remove(&variable);
        active.remove(&variable);
        order.push(variable);

        for candidate in affected {
            if !active.contains(&candidate) {
                continue;
            }
            let version = versions
                .get(&candidate)
                .copied()
                .unwrap_or(0)
                .checked_add(1)
                .ok_or("legacy planner version overflow")?;
            versions.insert(candidate, version);
            let (fill, degree) = legacy_fill_score(candidate, &adjacency);
            queue.push(Reverse((fill, degree, candidate, version)));
        }
    }
    Ok((order, width))
}

fn legacy_fill_score(
    variable: usize,
    adjacency: &BTreeMap<usize, BTreeSet<usize>>,
) -> (usize, usize) {
    let neighbours = adjacency
        .get(&variable)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter(|neighbour| adjacency.contains_key(neighbour))
        .collect::<Vec<_>>();
    let mut fill = 0usize;
    for (left_index, left) in neighbours.iter().copied().enumerate() {
        for right in neighbours.iter().copied().skip(left_index + 1) {
            if !adjacency[&left].contains(&right) {
                fill += 1;
            }
        }
    }
    (fill, neighbours.len())
}

fn median_duration(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn run() -> Result<(), String> {
    println!(
        "topology,variables,edges,repeats,legacy_median_ns,optimized_median_ns,median_speedup,legacy_width,optimized_width,optimized_fill_score_evaluations,optimized_fill_adjacency_visits,optimized_leaf_updates,order_digest,validated"
    );

    for fixture in FIXTURES {
        let scopes = build_scopes(fixture)?;
        let eliminate = (0..fixture.variables).collect::<Vec<_>>();

        let legacy_warmup = legacy_full_rescan_order(&scopes, &eliminate)?;
        let optimized_warmup = plan_min_fill_elimination(
            &scopes,
            &eliminate,
            PLANNER_ONLY_ENTRY_CEILING,
        )
        .map_err(|error| format!("optimized preflight failed: {error}"))?;

        if legacy_warmup.0 != optimized_warmup.elimination_order
            || legacy_warmup.1 != optimized_warmup.induced_width
        {
            return Err(format!(
                "planner mismatch on {}-{}: legacy order/width != optimized order/width",
                fixture.topology, fixture.variables
            ));
        }

        // The largest legacy star is intentionally sampled once; the original
        // full-rescan implementation is cubic-like on this shape. Other rows
        // use the same repetition count for both implementations.
        let repeats = fixture.repeats;
        let mut legacy_samples = Vec::with_capacity(repeats);
        let mut optimized_samples = Vec::with_capacity(repeats);
        for sample in 0..repeats {
            if sample % 2 == 0 {
                let start = Instant::now();
                let legacy = legacy_full_rescan_order(&scopes, &eliminate)?;
                legacy_samples.push(start.elapsed());
                if legacy != legacy_warmup {
                    return Err("legacy plan changed between repetitions".to_string());
                }

                let start = Instant::now();
                let optimized = plan_min_fill_elimination(
                    &scopes,
                    &eliminate,
                    PLANNER_ONLY_ENTRY_CEILING,
                )
                .map_err(|error| error.to_string())?;
                optimized_samples.push(start.elapsed());
                if optimized.elimination_order != optimized_warmup.elimination_order
                    || optimized.induced_width != optimized_warmup.induced_width
                {
                    return Err("optimized plan changed between repetitions".to_string());
                }
            } else {
                let start = Instant::now();
                let optimized = plan_min_fill_elimination(
                    &scopes,
                    &eliminate,
                    PLANNER_ONLY_ENTRY_CEILING,
                )
                .map_err(|error| error.to_string())?;
                optimized_samples.push(start.elapsed());

                let start = Instant::now();
                let legacy = legacy_full_rescan_order(&scopes, &eliminate)?;
                legacy_samples.push(start.elapsed());
                if legacy != legacy_warmup {
                    return Err("legacy plan changed between repetitions".to_string());
                }
                if optimized.elimination_order != optimized_warmup.elimination_order
                    || optimized.induced_width != optimized_warmup.induced_width
                {
                    return Err("optimized plan changed between repetitions".to_string());
                }
            }
        }

        let legacy_median = median_duration(&mut legacy_samples);
        let optimized_median = median_duration(&mut optimized_samples);
        let speedup = legacy_median.as_nanos() as f64
            / optimized_median.as_nanos().max(1) as f64;
        let mut order_bytes = Vec::with_capacity(optimized_warmup.elimination_order.len() * 8);
        for variable in &optimized_warmup.elimination_order {
            order_bytes.extend_from_slice(&(*variable as u64).to_le_bytes());
        }
        let order_digest = {
            use sha2::{Digest, Sha256};
            let digest = Sha256::digest(&order_bytes);
            digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>()
        };

        println!(
            "{},{},{},{},{},{},{:.4},{},{},{},{},{},{},true",
            fixture.topology,
            fixture.variables,
            scopes.len(),
            repeats,
            legacy_median.as_nanos(),
            optimized_median.as_nanos(),
            speedup,
            legacy_warmup.1,
            optimized_warmup.induced_width,
            optimized_warmup.planner_work.fill_score_evaluations,
            optimized_warmup.planner_work.fill_score_adjacency_visits,
            optimized_warmup.planner_work.leaf_fast_path_updates,
            order_digest,
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("matched min-fill planner comparison failed: {error}");
        std::process::exit(1);
    }
}
