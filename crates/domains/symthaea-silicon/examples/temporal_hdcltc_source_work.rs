// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Source-bound work and latency observations for the active HDC-LTC neuron API.
//!
//! This example calls the real HdcLtcUnifiedNeuron public evolution methods.
//! It reports three evidence classes separately:
//! 1. observed public API counters (evolution_clock before/after);
//! 2. source-derived internal work for one pinned source blob;
//! 3. direct software wall-time observations around the public method call.
//!
//! The measured boundary includes the method's internal state-bound and
//! update-statistics work. It excludes neuron construction, cloning, input
//! generation, serialization, and process startup. It does not measure
//! physical energy, DRAM traffic, allocation counts, or a complete service path.
//!
//! FOURIER_ENABLED is deliberately false: at this source revision,
//! compute_equilibrium can include the optional Fourier basis while the RK4
//! compute_derivative path does not. Do not describe Fourier-enabled RK4 as
//! a same-equation comparison until that reference mismatch is adjudicated.
//!
//! From the repository root:
//!   cargo fmt --check
//!   cargo test --locked -p symthaea-silicon --example temporal_hdcltc_source_work
//!   cargo run --release --locked -p symthaea-silicon --example temporal_hdcltc_source_work
//!
//! The timing output is descriptive for the machine/toolchain on which it is
//! executed. It is not a portable hardware ranking or a performance claim.

use serde::Serialize;
use std::error::Error;
use std::time::Instant;

use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::ContinuousHV;

const SCHEMA: &str = "symthaea.hdc-ltc-source-work";
const SCHEMA_VERSION: &str = "0.1.0";
const ADAPTER_VERSION: &str = "0.1.0";

// Git blob object ID of the inspected main-branch source file. This is an
// identity for the exact source file, not a SHA-256 digest or a build attestation.
const PINNED_SOURCE_BLOB_OID: &str = "40c3b0e75845a20b3ef2dbbe0ea3a0af0068f4b7";
const PINNED_SOURCE_COMMIT_SHA: &str = "36eb3adc1dc2d5ad1ae643863fee828d594472d4";
const PINNED_SOURCE_PATH: &str = "crates/core/symthaea-core/src/hdc/hdc_ltc_unified.rs";

const FOURIER_ENABLED: bool = false;
const WARMUP_SAMPLES: usize = 4;
const MEASURED_SAMPLES: usize = 32;

fn simd_runtime_path() -> &'static str {
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            return "x86_64_avx2_fma";
        }
        return "x86_64_scalar_fallback";
    }
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        return "aarch64_neon";
    }
    #[allow(unreachable_code)]
    "scalar_fallback"
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum EvolutionMethod {
    Euler,
    RungeKutta4,
    ClosedForm,
    FusedClosedForm,
    AnalyticalClosedForm,
    IterativeClosedForm,
}

impl EvolutionMethod {
    const ALL: [Self; 6] = [
        Self::Euler,
        Self::RungeKutta4,
        Self::ClosedForm,
        Self::FusedClosedForm,
        Self::AnalyticalClosedForm,
        Self::IterativeClosedForm,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Euler => "euler",
            Self::RungeKutta4 => "runge_kutta4",
            Self::ClosedForm => "closed_form",
            Self::FusedClosedForm => "fused_closed_form",
            Self::AnalyticalClosedForm => "analytical_closed_form",
            Self::IterativeClosedForm => "iterative_closed_form",
        }
    }

    fn evolve(self, neuron: &mut HdcLtcUnifiedNeuron, dt: f32, input: &ContinuousHV) {
        match self {
            Self::Euler => neuron.evolve(dt, input),
            Self::RungeKutta4 => neuron.evolve_rk4(dt, input),
            Self::ClosedForm => neuron.evolve_closed_form(dt, input),
            Self::FusedClosedForm => neuron.evolve_closed_form_fused(dt, input),
            Self::AnalyticalClosedForm => neuron.evolve_closed_form_exact(dt, input),
            Self::IterativeClosedForm => neuron.evolve_closed_form_iterative(dt, input),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
struct SourceDerivedWork {
    public_api_invocations: u64,
    derivative_rhs_evaluations: u64,
    equilibrium_recomputations: u64,
    tau_recomputations: u64,
    gating_evaluations: u64,
    logical_hdc_bind_ops: u64,
    bundle_ops: u64,
    activation_ops: u64,
    input_similarity_evaluations: u64,
    gating_vector_passes: u64,
    fused_vector_passes: u64,
    /// Direct calls to f32::exp in the pinned source; excludes tanh/libm internals.
    explicit_exp_calls: u64,
    state_norm_scans: u64,
    iterative_substeps: u64,
}

impl SourceDerivedWork {
    /// Derive source-level counts for one public method invocation.
    ///
    /// Counts follow the pinned hdc_ltc_unified.rs source structure. They are
    /// not primitive instruction counts and must not be converted into joules.
    fn for_invocation(
        method: EvolutionMethod,
        dt: f32,
        tau_base: f32,
    ) -> Result<Self, &'static str> {
        if !dt.is_finite() || dt <= 0.0 || !tau_base.is_finite() || tau_base <= 0.0 {
            return Err("source-work derivation requires finite positive dt and tau_base");
        }

        let mut work = Self {
            public_api_invocations: 1,
            derivative_rhs_evaluations: 0,
            equilibrium_recomputations: 0,
            tau_recomputations: 0,
            gating_evaluations: 0,
            logical_hdc_bind_ops: 0,
            bundle_ops: 0,
            activation_ops: 0,
            input_similarity_evaluations: 0,
            gating_vector_passes: 0,
            fused_vector_passes: 0,
            explicit_exp_calls: 0,
            state_norm_scans: 0,
            iterative_substeps: 0,
        };

        match method {
            EvolutionMethod::Euler => {
                work.derivative_rhs_evaluations = 1;
                work.equilibrium_recomputations = 1;
                work.tau_recomputations = 1;
                work.logical_hdc_bind_ops = 2;
                work.bundle_ops = 1;
                work.activation_ops = 1;
                work.input_similarity_evaluations = 1;
                // compute_tau + apply_state_bounds + update_stats.
                work.state_norm_scans = 3;
            }
            EvolutionMethod::RungeKutta4 => {
                work.derivative_rhs_evaluations = 4;
                work.equilibrium_recomputations = 4;
                work.tau_recomputations = 4;
                work.logical_hdc_bind_ops = 8;
                work.bundle_ops = 4;
                work.activation_ops = 4;
                work.input_similarity_evaluations = 4;
                // One state norm in each derivative stage, then bounds + stats.
                work.state_norm_scans = 6;
            }
            EvolutionMethod::ClosedForm => {
                work.equilibrium_recomputations = 1;
                work.tau_recomputations = 1; // compute_gating calls compute_tau.
                work.gating_evaluations = 1;
                work.logical_hdc_bind_ops = 2;
                work.bundle_ops = 1;
                work.activation_ops = 1;
                work.input_similarity_evaluations = 1;
                work.gating_vector_passes = 1;
                // compute_tau + apply_state_bounds + update_stats.
                work.state_norm_scans = 3;
                // One sigmoid exponential and one time-decay exponential.
                work.explicit_exp_calls = 2;
            }
            EvolutionMethod::FusedClosedForm => {
                work.tau_recomputations = 1;
                work.gating_evaluations = 1;
                // The fused dimension pass still implements two logical binds,
                // one bundle and one activation without materializing intermediates.
                work.logical_hdc_bind_ops = 2;
                work.bundle_ops = 1;
                work.activation_ops = 1;
                work.input_similarity_evaluations = 1;
                work.gating_vector_passes = 1;
                work.fused_vector_passes = 1;
                work.state_norm_scans = 3;
                work.explicit_exp_calls = 2;
            }
            EvolutionMethod::AnalyticalClosedForm => {
                work.equilibrium_recomputations = 1;
                work.tau_recomputations = 1;
                work.logical_hdc_bind_ops = 2;
                work.bundle_ops = 1;
                work.activation_ops = 1;
                work.input_similarity_evaluations = 1;
                work.state_norm_scans = 3;
                work.explicit_exp_calls = 1;
            }
            EvolutionMethod::IterativeClosedForm => {
                let max_substep = tau_base / 10.0;
                if !max_substep.is_finite() || max_substep <= 0.0 {
                    return Err("iterative max-substep is not finite and positive");
                }
                let raw_steps = (dt / max_substep).ceil();
                if !raw_steps.is_finite() || raw_steps < 1.0 || raw_steps >= u64::MAX as f32 {
                    return Err("iterative substep count is outside the auditable range");
                }
                let steps = raw_steps as u64;
                work.iterative_substeps = steps;
                work.equilibrium_recomputations = steps;
                work.tau_recomputations = steps;
                work.logical_hdc_bind_ops = steps.checked_mul(2)
                    .ok_or("logical HDC bind counter overflow")?;
                work.bundle_ops = steps;
                work.activation_ops = steps;
                work.input_similarity_evaluations = steps;
                work.explicit_exp_calls = steps;
                // One state norm per compute_tau substep, then bounds + stats.
                work.state_norm_scans = steps
                    .checked_add(2)
                    .ok_or("state norm scan counter overflow")?;
            }
        }
        Ok(work)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct LatencyObservation {
    p50_ns: u64,
    p95_ns: u64,
    /// Withheld until at least 1,000 measured samples are available.
    p99_ns: Option<u64>,
    p99_minimum_sample_gate: u64,
    minimum_ns: u64,
    maximum_ns: u64,
    sample_count: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct MethodObservation {
    method: String,
    source_derived_work: SourceDerivedWork,
    direct_wall_time: LatencyObservation,
    observed_api_update_delta_min: u64,
    observed_api_update_delta_max: u64,
    logical_time_delta_consistent: bool,
    deterministic_final_state_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct BenchmarkReceipt {
    schema: String,
    schema_version: String,
    adapter_version: String,
    source_path: String,
    source_commit_sha: String,
    source_blob_oid: String,
    evidence_boundary: String,
    architecture: String,
    operating_system: String,
    release_build: bool,
    simd_feature_enabled: bool,
    simd_runtime_path: String,
    activation: String,
    fourier_enabled: bool,
    dimension: usize,
    tau_base_seconds: f32,
    dt_seconds: f32,
    model_seed: u64,
    input_seed: u64,
    initial_state_seed: u64,
    warmup_samples: u64,
    measured_samples: u64,
    methods: Vec<MethodObservation>,
    physical_energy_joules: Option<f64>,
}

fn percentile(sorted: &[u64], numerator: usize, denominator: usize) -> Option<u64> {
    if sorted.is_empty() || denominator == 0 {
        return None;
    }
    // Nearest-rank percentile with 1-based rank and checked integer arithmetic.
    let rank = sorted
        .len()
        .checked_mul(numerator)?
        .checked_add(denominator - 1)?
        / denominator;
    let index = rank.max(1).min(sorted.len()) - 1;
    sorted.get(index).copied()
}

fn summarize_latency(mut samples: Vec<u64>) -> Result<LatencyObservation, &'static str> {
    if samples.is_empty() {
        return Err("latency summary requires at least one observation");
    }
    samples.sort_unstable();
    Ok(LatencyObservation {
        p50_ns: percentile(&samples, 50, 100).ok_or("p50 unavailable")?,
        p95_ns: percentile(&samples, 95, 100).ok_or("p95 unavailable")?,
        p99_ns: if samples.len() >= 1_000 {
            percentile(&samples, 99, 100)
        } else {
            None
        },
        p99_minimum_sample_gate: 1_000,
        minimum_ns: *samples.first().ok_or("minimum unavailable")?,
        maximum_ns: *samples.last().ok_or("maximum unavailable")?,
        sample_count: samples.len() as u64,
    })
}

fn state_fingerprint(state: &ContinuousHV) -> String {
    // FNV-1a provides a stable local determinism fingerprint, not integrity.
    let mut hash = 0xcbf29ce484222325_u64;
    for value in &state.values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    format!("fnv1a64:{hash:016x}")
}

fn run_method(
    method: EvolutionMethod,
    dt: f32,
    tau_base: f32,
    template: &HdcLtcUnifiedNeuron,
    input: &ContinuousHV,
) -> Result<MethodObservation, Box<dyn Error>> {
    let source_derived_work = SourceDerivedWork::for_invocation(method, dt, tau_base)?;
    let mut latencies_ns = Vec::with_capacity(MEASURED_SAMPLES);
    let mut update_deltas = Vec::with_capacity(MEASURED_SAMPLES);
    let mut time_consistent = true;
    let mut final_fingerprint: Option<String> = None;

    for sample_index in 0..(WARMUP_SAMPLES + MEASURED_SAMPLES) {
        // Clone the identical model state outside the timer so all methods
        // start from the same recurrent state and weights.
        let mut neuron = template.clone();
        let (before_time, before_updates) = neuron.evolution_clock();

        let started = Instant::now();
        method.evolve(&mut neuron, dt, input);
        let elapsed_ns = u64::try_from(started.elapsed().as_nanos())?;
        let (after_time, after_updates) = neuron.evolution_clock();

        let update_delta = after_updates
            .checked_sub(before_updates)
            .ok_or("public API update counter regressed")?;
        let time_delta = after_time - before_time;
        let expected_dt = dt as f64;
        if update_delta != 1 {
            return Err(format!(
                "{} expected one API update, observed {update_delta}",
                method.name()
            )
            .into());
        }
        if !time_delta.is_finite() || (time_delta - expected_dt).abs() > 1.0e-9 {
            time_consistent = false;
        }

        let fingerprint = state_fingerprint(neuron.state());
        if let Some(previous) = &final_fingerprint {
            if previous != &fingerprint {
                return Err(format!(
                    "{} produced non-deterministic final state for identical inputs",
                    method.name()
                )
                .into());
            }
        } else {
            final_fingerprint = Some(fingerprint);
        }

        if sample_index >= WARMUP_SAMPLES {
            latencies_ns.push(elapsed_ns);
            update_deltas.push(update_delta);
        }
    }

    let min_delta = *update_deltas.iter().min().ok_or("no update samples")?;
    let max_delta = *update_deltas.iter().max().ok_or("no update samples")?;

    Ok(MethodObservation {
        method: method.name().to_owned(),
        source_derived_work,
        direct_wall_time: summarize_latency(latencies_ns)?,
        observed_api_update_delta_min: min_delta,
        observed_api_update_delta_max: max_delta,
        logical_time_delta_consistent: time_consistent,
        deterministic_final_state_fingerprint: final_fingerprint.ok_or("no final state")?,
    })
}

fn run_case(dimension: usize, dt: f32) -> Result<BenchmarkReceipt, Box<dyn Error>> {
    const TAU_BASE: f32 = 0.1;
    const MODEL_SEED: u64 = 0x51A7_001;
    const INPUT_SEED: u64 = 0x51A7_002;
    const INITIAL_STATE_SEED: u64 = 0x51A7_003;

    if dimension == 0 || !dimension.is_power_of_two() {
        return Err("dimension must be a non-zero power of two".into());
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err("dt must be finite and positive".into());
    }
    if FOURIER_ENABLED {
        return Err("Fourier-enabled comparison is excluded by the source parity caveat".into());
    }

    let config = UnifiedConfig {
        dimension,
        tau_base: TAU_BASE,
        fourier_frequencies: Vec::new(),
        ..UnifiedConfig::default()
    };
    let mut template = HdcLtcUnifiedNeuron::new(config, MODEL_SEED);
    template.set_state(ContinuousHV::random(dimension, INITIAL_STATE_SEED));
    let input = ContinuousHV::random(dimension, INPUT_SEED);

    let mut methods = Vec::with_capacity(EvolutionMethod::ALL.len());
    for method in EvolutionMethod::ALL {
        methods.push(run_method(method, dt, TAU_BASE, &template, &input)?);
    }

    Ok(BenchmarkReceipt {
        schema: SCHEMA.to_owned(),
        schema_version: SCHEMA_VERSION.to_owned(),
        adapter_version: ADAPTER_VERSION.to_owned(),
        source_path: PINNED_SOURCE_PATH.to_owned(),
        source_commit_sha: PINNED_SOURCE_COMMIT_SHA.to_owned(),
        source_blob_oid: PINNED_SOURCE_BLOB_OID.to_owned(),
        evidence_boundary: "public neuron evolution call including state bounds and update_stats; excludes construction, cloning, input creation, serialization, process startup".to_owned(),
        architecture: std::env::consts::ARCH.to_owned(),
        operating_system: std::env::consts::OS.to_owned(),
        release_build: !cfg!(debug_assertions),
        simd_feature_enabled: cfg!(feature = "simd"),
        simd_runtime_path: simd_runtime_path().to_owned(),
        activation: "tanh".to_owned(),
        fourier_enabled: FOURIER_ENABLED,
        dimension,
        tau_base_seconds: TAU_BASE,
        dt_seconds: dt,
        model_seed: MODEL_SEED,
        input_seed: INPUT_SEED,
        initial_state_seed: INITIAL_STATE_SEED,
        warmup_samples: WARMUP_SAMPLES as u64,
        measured_samples: MEASURED_SAMPLES as u64,
        methods,
        physical_energy_joules: None,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    // Keep the first direct measurements bounded; expand only after the
    // exact-head build/test gate and machine/toolchain manifest are available.
    for dimension in [1_024, 16_384] {
        for dt in [0.01_f32, 0.1_f32, 0.5_f32] {
            let receipt = run_case(dimension, dt)?;
            println!("{}", serde_json::to_string(&receipt)?);
        }
    }
    println!(
        "scope: source-derived HDC-LTC work + direct software timing; no physical energy or SOTA claim"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p99_is_withheld_until_the_minimum_tail_sample_count() {
        let small = summarize_latency((0..32).map(|value| value as u64).collect()).unwrap();
        assert_eq!(small.sample_count, 32);
        assert_eq!(small.p99_ns, None);
        assert_eq!(small.p99_minimum_sample_gate, 1_000);

        let large = summarize_latency((0..1_000).map(|value| value as u64).collect()).unwrap();
        assert_eq!(large.sample_count, 1_000);
        assert!(large.p99_ns.is_some());
    }

    #[test]
    fn every_public_method_is_accounted_as_one_api_invocation() {
        for method in EvolutionMethod::ALL {
            let work = SourceDerivedWork::for_invocation(method, 0.1, 0.1).unwrap();
            assert_eq!(work.public_api_invocations, 1);
        }
    }

    #[test]
    fn rk4_exposes_four_internal_stages_but_one_public_invocation() {
        let work =
            SourceDerivedWork::for_invocation(EvolutionMethod::RungeKutta4, 0.1, 0.1).unwrap();
        assert_eq!(work.public_api_invocations, 1);
        assert_eq!(work.derivative_rhs_evaluations, 4);
        assert_eq!(work.equilibrium_recomputations, 4);
        assert_eq!(work.tau_recomputations, 4);
        assert_eq!(work.state_norm_scans, 6);
        assert_eq!(work.logical_hdc_bind_ops, 8);
        assert_eq!(work.bundle_ops, 4);
        assert_eq!(work.activation_ops, 4);
        assert_eq!(work.input_similarity_evaluations, 4);
    }

    #[test]
    fn iterative_work_scales_with_base_tau_and_logical_interval() {
        let work =
            SourceDerivedWork::for_invocation(EvolutionMethod::IterativeClosedForm, 0.5, 0.1)
                .unwrap();
        assert_eq!(work.public_api_invocations, 1);
        assert_eq!(work.iterative_substeps, 50);
        assert_eq!(work.equilibrium_recomputations, 50);
        assert_eq!(work.tau_recomputations, 50);
        assert_eq!(work.explicit_exp_calls, 50);
        assert_eq!(work.state_norm_scans, 52);
    }

    #[test]
    fn closed_form_variants_preserve_distinct_source_work_categories() {
        let ordinary =
            SourceDerivedWork::for_invocation(EvolutionMethod::ClosedForm, 0.1, 0.1).unwrap();
        let fused =
            SourceDerivedWork::for_invocation(EvolutionMethod::FusedClosedForm, 0.1, 0.1).unwrap();
        let analytical =
            SourceDerivedWork::for_invocation(EvolutionMethod::AnalyticalClosedForm, 0.1, 0.1)
                .unwrap();

        assert_eq!(ordinary.equilibrium_recomputations, 1);
        assert_eq!(ordinary.gating_evaluations, 1);
        assert_eq!(ordinary.logical_hdc_bind_ops, 2);
        assert_eq!(ordinary.bundle_ops, 1);
        assert_eq!(ordinary.activation_ops, 1);
        assert_eq!(ordinary.input_similarity_evaluations, 1);
        assert_eq!(ordinary.gating_vector_passes, 1);
        assert_eq!(ordinary.explicit_exp_calls, 2);
        assert_eq!(fused.equilibrium_recomputations, 0);
        assert_eq!(fused.fused_vector_passes, 1);
        assert_eq!(fused.logical_hdc_bind_ops, ordinary.logical_hdc_bind_ops);
        assert_eq!(fused.bundle_ops, ordinary.bundle_ops);
        assert_eq!(fused.activation_ops, ordinary.activation_ops);
        assert_eq!(fused.gating_evaluations, 1);
        assert_eq!(analytical.gating_evaluations, 0);
        assert_eq!(analytical.logical_hdc_bind_ops, 2);
        assert_eq!(analytical.bundle_ops, 1);
        assert_eq!(analytical.activation_ops, 1);
        assert_eq!(analytical.input_similarity_evaluations, 1);
        assert_eq!(analytical.explicit_exp_calls, 1);
    }

    #[test]
    fn rejects_invalid_logical_intervals_and_time_constants() {
        for dt in [0.0, -0.1, f32::NAN, f32::INFINITY] {
            assert!(SourceDerivedWork::for_invocation(EvolutionMethod::Euler, dt, 0.1).is_err());
        }
        for tau in [0.0, -0.1, f32::NAN, f32::INFINITY] {
            assert!(SourceDerivedWork::for_invocation(
                EvolutionMethod::IterativeClosedForm,
                0.1,
                tau
            )
            .is_err());
        }
    }

    #[test]
    fn observed_api_counters_and_logical_time_match_one_call() {
        let config = UnifiedConfig {
            dimension: 256,
            tau_base: 0.1,
            fourier_frequencies: Vec::new(),
            ..UnifiedConfig::default()
        };
        let mut neuron = HdcLtcUnifiedNeuron::new(config, 123);
        let input = ContinuousHV::random(256, 456);
        let before = neuron.evolution_clock();

        neuron.evolve_rk4(0.1, &input);

        let after = neuron.evolution_clock();
        assert_eq!(after.1 - before.1, 1);
        assert!((after.0 - before.0 - 0.1_f64).abs() < 1.0e-9);
    }

    #[test]
    fn repeated_identical_inputs_produce_identical_state_fingerprints() {
        let config = UnifiedConfig {
            dimension: 256,
            tau_base: 0.1,
            fourier_frequencies: Vec::new(),
            ..UnifiedConfig::default()
        };
        let mut template = HdcLtcUnifiedNeuron::new(config, 123);
        template.set_state(ContinuousHV::random(256, 789));
        let input = ContinuousHV::random(256, 456);

        for method in EvolutionMethod::ALL {
            let mut first = template.clone();
            let mut second = template.clone();
            method.evolve(&mut first, 0.1, &input);
            method.evolve(&mut second, 0.1, &input);
            assert_eq!(
                state_fingerprint(first.state()),
                state_fingerprint(second.state())
            );
        }
    }
}