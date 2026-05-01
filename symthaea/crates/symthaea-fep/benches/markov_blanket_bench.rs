// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Performance benchmarks for the Markov Blanket operator.
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use symthaea_fep::markov_blanket::{
    identify_coalitions, MarkovBoundaryOperator, MarkovPartition, PermeabilityInputs,
    TopologyBoundaryInputs,
};
use symthaea_fep::{ActiveInferenceAgentConfig, EnhancedFEPBridge, HiddenState, Observation};

fn bench_compute_permeability(c: &mut Criterion) {
    let mut op = MarkovBoundaryOperator::new(MarkovPartition {
        internal_dim: 16384,
        sensory_dim: 4,
        active_dim: 8,
    });
    let inputs = PermeabilityInputs {
        serotonin: 0.7,
        oxytocin: 0.6,
        flow_state: 0.5,
        threat_level: 0.2,
        noradrenaline: 0.3,
        acetylcholine: 0.4,
        peer_trust: 0.7,
    };
    c.bench_function("compute_permeability", |b| {
        b.iter(|| {
            let permeability = op.compute_permeability(black_box(&inputs)).clone();
            black_box(permeability)
        })
    });
}

fn bench_gate_observation(c: &mut Criterion) {
    let mut op = MarkovBoundaryOperator::new(MarkovPartition {
        internal_dim: 16384,
        sensory_dim: 4,
        active_dim: 8,
    });
    let inputs = PermeabilityInputs::default();
    for _ in 0..50 {
        op.compute_permeability(&inputs);
    }
    let obs = Observation::from_consciousness_state(0.7, 0.6, 0.5, 0.4);
    let prior = HiddenState::new(4);
    c.bench_function("gate_observation", |b| {
        b.iter(|| op.gate_observation(black_box(&obs), black_box(&prior)))
    });
}

fn bench_topology_constraints(c: &mut Criterion) {
    let mut op = MarkovBoundaryOperator::new(MarkovPartition {
        internal_dim: 16384,
        sensory_dim: 4,
        active_dim: 8,
    });
    let inputs = PermeabilityInputs::default();
    for _ in 0..50 {
        op.compute_permeability(&inputs);
    }
    let topo = TopologyBoundaryInputs {
        boundary_thickness: 0.5,
        fiedler_value: 0.8,
        boundary_components: 3,
    };
    c.bench_function("apply_topology_constraints", |b| {
        b.iter(|| op.apply_topology_constraints(black_box(&topo)))
    });
}

fn bench_coalition_identification(c: &mut Criterion) {
    let mut group = c.benchmark_group("coalition_identification");
    for n in [10, 50, 100, 256] {
        let peers: Vec<(String, f64)> = (0..n)
            .map(|i| (format!("p{}", i), 0.5 + 0.3 * (i as f64 / n as f64)))
            .collect();
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n.min(i + 8) {
                let dist = (j - i) as f64 / 8.0;
                edges.push((i, j, (1.0 - dist).max(0.0)));
            }
        }
        group.bench_function(format!("{}_peers", n), |b| {
            b.iter(|| {
                let peers = black_box(peers.as_slice());
                let edges = black_box(edges.as_slice());
                identify_coalitions(peers, edges, 0.5)
            })
        });
    }
    group.finish();
}

fn bench_full_fep_cycle_with_blanket(c: &mut Criterion) {
    let config = ActiveInferenceAgentConfig::default();
    let mut bridge = EnhancedFEPBridge::new(config, 4);
    let inputs = PermeabilityInputs::default();
    for _ in 0..20 {
        bridge.update_blanket_permeability(&inputs);
        bridge.cycle(0.5, 0.5, 0.5, 0.5);
    }
    c.bench_function("full_fep_cycle_with_blanket", |b| {
        b.iter(|| {
            bridge.update_blanket_permeability(black_box(&inputs));
            bridge.cycle(
                black_box(0.5),
                black_box(0.5),
                black_box(0.5),
                black_box(0.5),
            )
        })
    });
}

criterion_group!(
    benches,
    bench_compute_permeability,
    bench_gate_observation,
    bench_topology_constraints,
    bench_coalition_identification,
    bench_full_fep_cycle_with_blanket,
);
criterion_main!(benches);
