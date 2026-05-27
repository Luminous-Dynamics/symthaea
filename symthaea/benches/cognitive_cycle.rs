// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cognitive Cycle Benchmark Suite
//!
//! End-to-end benchmarks for the cognitive loop cycle, measuring:
//! - Single cycle latency (input → CycleResult)
//! - Throughput (cycles/second)
//! - Warm-up vs steady-state performance
//! - Backend comparison (CfC vs HdcLtcUnified)
//! - Impact of optional subsystems (surprise exploration, prefrontal)
//!
//! ## Usage
//!
//! ```bash
//! cargo bench --bench cognitive_cycle
//! cargo bench --bench cognitive_cycle -- --save-baseline main
//! ```

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, TemporalBackend};

/// Warm up a service by running a few cycles
fn warm_up(service: &mut CognitiveLoopService, cycles: usize) {
    for _ in 0..cycles {
        let _ = service.cycle("warm up input");
    }
}

// =============================================================================
// SINGLE CYCLE LATENCY
// =============================================================================

fn bench_single_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("cognitive_cycle_single");
    group.sample_size(30);

    // Cold start: first cycle (includes one-time initialization costs)
    group.bench_function("cold_start", |b| {
        b.iter_with_setup(
            || {
                CognitiveLoopService::new(CognitiveLoopConfig {
                    async_training: false,
                    ..Default::default()
                })
                .unwrap()
            },
            |mut service| black_box(service.cycle("cause leads to effect")),
        )
    });

    // Warm: steady-state cycle after warm-up
    group.bench_function("warm_steady_state", |b| {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
            async_training: false,
            ..Default::default()
        })
        .unwrap();
        warm_up(&mut service, 10);

        b.iter(|| black_box(service.cycle("cause leads to effect")))
    });

    group.finish();
}

// =============================================================================
// BACKEND COMPARISON: CfC vs HdcLtcUnified
// =============================================================================

fn bench_backend_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("cognitive_cycle_backend");
    group.sample_size(20);

    let backends = [
        ("cfc", TemporalBackend::CfC),
        ("hdc_ltc", TemporalBackend::HdcLtcUnified),
    ];

    for (name, backend) in &backends {
        group.bench_with_input(BenchmarkId::new("cycle", name), backend, |b, backend| {
            let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
                temporal_backend: *backend,
                async_training: false,
                ..Default::default()
            })
            .unwrap();
            warm_up(&mut service, 5);

            b.iter(|| black_box(service.cycle("the cat sat on the mat")))
        });
    }

    group.finish();
}

// =============================================================================
// SUBSYSTEM IMPACT: measure overhead of optional subsystems
// =============================================================================

fn bench_subsystem_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("cognitive_cycle_subsystems");
    group.sample_size(20);

    let configs = [
        (
            "minimal",
            CognitiveLoopConfig {
                async_training: false,
                enable_surprise_exploration: false,
                enable_prefrontal: false,
                causal_enhancement: false,
                episodic_replay_training: false,
                ..Default::default()
            },
        ),
        (
            "surprise",
            CognitiveLoopConfig {
                async_training: false,
                enable_surprise_exploration: true,
                enable_prefrontal: false,
                ..Default::default()
            },
        ),
        (
            "prefrontal",
            CognitiveLoopConfig {
                async_training: false,
                enable_prefrontal: true,
                enable_surprise_exploration: false,
                ..Default::default()
            },
        ),
        (
            "full",
            CognitiveLoopConfig {
                async_training: false,
                enable_surprise_exploration: true,
                enable_prefrontal: true,
                causal_enhancement: true,
                episodic_replay_training: true,
                ..Default::default()
            },
        ),
    ];

    for (name, config) in &configs {
        group.bench_with_input(BenchmarkId::new("cycle", name), config, |b, config| {
            let mut service = CognitiveLoopService::new(config.clone()).unwrap();
            warm_up(&mut service, 5);

            b.iter(|| black_box(service.cycle("exploration of novel patterns")))
        });
    }

    group.finish();
}

// =============================================================================
// THROUGHPUT: cycles per second
// =============================================================================

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("cognitive_cycle_throughput");
    group.sample_size(10);

    // Measure 100 consecutive cycles
    group.bench_function("100_cycles", |b| {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
            async_training: false,
            ..Default::default()
        })
        .unwrap();

        let inputs = [
            "the sun rises in the east",
            "cause and effect are linked",
            "patterns emerge from chaos",
            "consciousness arises from integration",
            "temporal prediction drives learning",
        ];

        b.iter(|| {
            for input in &inputs {
                for _ in 0..20 {
                    black_box(service.cycle(input));
                }
            }
        })
    });

    group.finish();
}

// =============================================================================
// INPUT COMPLEXITY SCALING
// =============================================================================

fn bench_input_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("cognitive_cycle_input_scale");
    group.sample_size(20);

    let inputs = [
        ("short", "hello"),
        ("medium", "the quick brown fox jumps over the lazy dog"),
        (
            "long",
            "consciousness emerges from the integration of information across distributed \
             neural networks through predictive coding and free energy minimization principles",
        ),
    ];

    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        async_training: false,
        ..Default::default()
    })
    .unwrap();
    warm_up(&mut service, 5);

    for (name, input) in &inputs {
        group.bench_with_input(BenchmarkId::new("cycle", name), input, |b, input| {
            b.iter(|| black_box(service.cycle(input)))
        });
    }

    group.finish();
}

// =============================================================================
// SUSTAINED THROUGHPUT: validate 50Hz claim over fixed duration
// =============================================================================

fn bench_sustained_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("cognitive_cycle_sustained");
    // Use fewer samples but longer measurement time
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(10));

    let configs = [
        (
            "minimal_500cycles",
            CognitiveLoopConfig {
                async_training: false,
                enable_surprise_exploration: false,
                enable_prefrontal: false,
                enable_meta_cognition: false,
                causal_enhancement: false,
                episodic_replay_training: false,
                ..Default::default()
            },
        ),
        (
            "full_500cycles",
            CognitiveLoopConfig {
                async_training: false,
                enable_surprise_exploration: true,
                enable_prefrontal: true,
                enable_meta_cognition: true,
                causal_enhancement: true,
                episodic_replay_training: true,
                ..Default::default()
            },
        ),
    ];

    let inputs = [
        "the sun rises in the east",
        "cause and effect are linked",
        "patterns emerge from chaos",
        "consciousness arises from integration",
        "temporal prediction drives learning",
        "moral reasoning requires empathy",
        "exploration balances exploitation",
        "free energy minimization is universal",
        "hyperdimensional binding creates meaning",
        "precision weighting modulates attention",
    ];

    for (name, config) in &configs {
        group.bench_with_input(BenchmarkId::new("sustained", name), config, |b, config| {
            let mut service = CognitiveLoopService::new(config.clone()).unwrap();
            warm_up(&mut service, 10);

            b.iter(|| {
                // Run 500 consecutive cycles with varied input (sustained load)
                for i in 0..500 {
                    black_box(service.cycle(inputs[i % inputs.len()]));
                }
            })
        });
    }

    group.finish();
}

// =============================================================================
// URGENCY MODE DISTRIBUTION: measure natural urgency mode transitions
// =============================================================================

fn bench_urgency_natural(c: &mut Criterion) {
    let mut group = c.benchmark_group("cognitive_cycle_urgency");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(5));

    // Repeated same input drives error down → system should transition to Cruise
    group.bench_function("same_input_500cycles", |b| {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
            async_training: false,
            enable_surprise_exploration: true,
            enable_prefrontal: true,
            enable_meta_cognition: true,
            enable_predictive_processing: true,
            enable_cross_modal_binding: true,
            enable_affective_bridge: true,
            enable_embodied_cognition: true,
            enable_narrative_gwt: true,
            causal_enhancement: true,
            ..Default::default()
        })
        .unwrap();
        warm_up(&mut service, 50); // let it stabilize into Cruise

        b.iter(|| {
            for _ in 0..500 {
                black_box(service.cycle("stable predictable input"));
            }
        })
    });

    // Varied input keeps error high → system stays in Normal/Critical
    group.bench_function("varied_input_500cycles", |b| {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
            async_training: false,
            enable_surprise_exploration: true,
            enable_prefrontal: true,
            enable_meta_cognition: true,
            enable_predictive_processing: true,
            enable_cross_modal_binding: true,
            enable_affective_bridge: true,
            enable_embodied_cognition: true,
            enable_narrative_gwt: true,
            causal_enhancement: true,
            ..Default::default()
        })
        .unwrap();
        warm_up(&mut service, 10);

        let inputs = [
            "alpha cognition learning",
            "beta temporal dynamics",
            "gamma oscillation binding",
            "delta sleep consolidation",
            "theta memory encoding",
            "epsilon exploration novelty",
            "zeta moral reasoning",
            "eta causal inference",
            "iota consciousness integration",
            "kappa prediction error",
        ];

        b.iter(|| {
            for i in 0..500 {
                black_box(service.cycle(inputs[i % inputs.len()]));
            }
        })
    });

    group.finish();
}

// =============================================================================
// CYCLE_WITH_HV: direct HV injection (bypasses text encoder)
// =============================================================================

fn bench_cycle_with_hv(c: &mut Criterion) {
    use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

    let mut group = c.benchmark_group("cognitive_cycle_hv");
    group.sample_size(30);

    // Single cycle with pre-encoded HV
    group.bench_function("single_hv", |b| {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
            async_training: false,
            ..Default::default()
        })
        .unwrap();
        let hdv = ContinuousHV::random(HDC_DIMENSION, 42);
        warm_up(&mut service, 5);

        b.iter(|| black_box(service.cycle_with_hv(&hdv)))
    });

    // Compare text vs HV cycle (same service, isolates encoder cost)
    group.bench_function("100_text_cycles", |b| {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
            async_training: false,
            ..Default::default()
        })
        .unwrap();
        warm_up(&mut service, 5);

        b.iter(|| {
            for _ in 0..100 {
                black_box(service.cycle("benchmark comparison input"));
            }
        })
    });

    group.bench_function("100_hv_cycles", |b| {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
            async_training: false,
            ..Default::default()
        })
        .unwrap();
        let hdv = ContinuousHV::random(HDC_DIMENSION, 42);
        warm_up(&mut service, 5);

        b.iter(|| {
            for _ in 0..100 {
                black_box(service.cycle_with_hv(&hdv));
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_cycle,
    bench_backend_comparison,
    bench_subsystem_impact,
    bench_throughput,
    bench_input_scaling,
    bench_sustained_throughput,
    bench_urgency_natural,
    bench_cycle_with_hv,
);
criterion_main!(benches);
