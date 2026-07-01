// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Consciousness State-Space Explorer
//!
//! Systematic exploration of the 7-component consciousness equation parameter space.
//! Identifies phase transitions, minimum viable configurations, and interaction effects.
//!
//! ## Modes
//!
//! 1. **heatmap** — 2D sweep of two components (others fixed at 0.5)
//! 2. **lhs** — Latin Hypercube Sampling across all 7 dimensions
//! 3. **transition** — Phase transition detection via gradient analysis
//! 4. **substrate** — Sweep substrate types with varying component values
//!
//! ## Usage
//!
//! ```bash
//! # 2D heatmap: Integration vs Binding
//! cargo run --example consciousness_state_space -- --mode heatmap --axis-a integration --axis-b binding --resolution 20
//!
//! # Latin Hypercube Sampling (10,000 points)
//! cargo run --example consciousness_state_space -- --mode lhs --samples 10000 --seed 42
//!
//! # Phase transition detection
//! cargo run --example consciousness_state_space -- --mode transition --resolution 100
//!
//! # Substrate comparison
//! cargo run --example consciousness_state_space -- --mode substrate --resolution 10
//! ```

use symthaea::consciousness::consciousness_equation_v2::ConsciousnessEquationV2;
use symthaea::consciousness::{ConsciousnessStateV2, CoreComponent};

// ============================================================================
// CONFIGURATION
// ============================================================================

#[derive(Clone)]
struct Config {
    mode: Mode,
    resolution: usize,
    samples: usize,
    seed: u64,
    axis_a: CoreComponent,
    axis_b: CoreComponent,
    fixed_value: f64,
    substrate_values: Vec<f64>,
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Heatmap,
    Lhs,
    Transition,
    Substrate,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mode: Mode::Heatmap,
            resolution: 20,
            samples: 10_000,
            seed: 42,
            axis_a: CoreComponent::Integration,
            axis_b: CoreComponent::Binding,
            fixed_value: 0.5,
            substrate_values: vec![0.1, 0.3, 0.5, 0.7, 1.0],
        }
    }
}

fn parse_component(s: &str) -> Option<CoreComponent> {
    match s.to_lowercase().as_str() {
        "integration" | "phi" => Some(CoreComponent::Integration),
        "binding" => Some(CoreComponent::Binding),
        "workspace" => Some(CoreComponent::Workspace),
        "attention" => Some(CoreComponent::Attention),
        "recursion" => Some(CoreComponent::Recursion),
        "efficacy" => Some(CoreComponent::Efficacy),
        "knowledge" => Some(CoreComponent::Knowledge),
        _ => None,
    }
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut config = Config::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => {
                i += 1;
                config.mode = match args.get(i).map(|s| s.as_str()) {
                    Some("heatmap") => Mode::Heatmap,
                    Some("lhs") => Mode::Lhs,
                    Some("transition") => Mode::Transition,
                    Some("substrate") => Mode::Substrate,
                    _ => {
                        eprintln!("Unknown mode. Options: heatmap, lhs, transition, substrate");
                        std::process::exit(1);
                    }
                };
            }
            "--resolution" => {
                i += 1;
                config.resolution = args[i].parse().unwrap_or(20);
            }
            "--samples" => {
                i += 1;
                config.samples = args[i].parse().unwrap_or(10_000);
            }
            "--seed" => {
                i += 1;
                config.seed = args[i].parse().unwrap_or(42);
            }
            "--axis-a" => {
                i += 1;
                config.axis_a = parse_component(&args[i]).unwrap_or(CoreComponent::Integration);
            }
            "--axis-b" => {
                i += 1;
                config.axis_b = parse_component(&args[i]).unwrap_or(CoreComponent::Binding);
            }
            "--fixed" => {
                i += 1;
                config.fixed_value = args[i].parse().unwrap_or(0.5);
            }
            "--help" | "-h" => {
                eprintln!("Consciousness State-Space Explorer");
                eprintln!();
                eprintln!("USAGE:");
                eprintln!("  cargo run --example consciousness_state_space -- [OPTIONS]");
                eprintln!();
                eprintln!("OPTIONS:");
                eprintln!(
                    "  --mode <MODE>        heatmap|lhs|transition|substrate (default: heatmap)"
                );
                eprintln!("  --resolution <N>     Steps per dimension (default: 20)");
                eprintln!("  --samples <N>        LHS sample count (default: 10000)");
                eprintln!("  --seed <N>           Random seed (default: 42)");
                eprintln!("  --axis-a <COMP>      Heatmap X axis component (default: integration)");
                eprintln!("  --axis-b <COMP>      Heatmap Y axis component (default: binding)");
                eprintln!(
                    "  --fixed <F>          Fixed value for non-swept components (default: 0.5)"
                );
                eprintln!();
                eprintln!("COMPONENTS:");
                eprintln!(
                    "  integration, binding, workspace, attention, recursion, efficacy, knowledge"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }
    config
}

// ============================================================================
// MODE 1: 2D HEATMAP
// ============================================================================

fn run_heatmap(config: &Config) {
    eprintln!(
        "=== 2D Heatmap: {} vs {} (fixed={}) ===",
        config.axis_a.as_str(),
        config.axis_b.as_str(),
        config.fixed_value
    );
    eprintln!("Resolution: {}x{}", config.resolution, config.resolution);

    // CSV header
    println!(
        "{},{},consciousness,core_minimum,weighted_sum,limiting_factor,collapsed",
        config.axis_a.as_str(),
        config.axis_b.as_str()
    );

    let steps = config.resolution;
    for i in 0..=steps {
        let val_a = i as f64 / steps as f64;
        for j in 0..=steps {
            let val_b = j as f64 / steps as f64;

            let mut state = ConsciousnessStateV2::new();
            // Set all components to fixed value
            for comp in CoreComponent::all() {
                state.set_core(comp, config.fixed_value);
            }
            // Override the two sweep axes
            state.set_core(config.axis_a, val_a);
            state.set_core(config.axis_b, val_b);

            let mut eq = ConsciousnessEquationV2::new();
            let result = eq.compute(&state);

            let collapsed = result.consciousness < 0.01
                || CoreComponent::all()
                    .iter()
                    .any(|c| state.get_core(*c) < 0.01);

            println!(
                "{:.3},{:.3},{:.6},{:.6},{:.6},{},{}",
                val_a,
                val_b,
                result.consciousness,
                result.core_minimum,
                result.weighted_sum,
                result.limiting_factor.as_str(),
                collapsed
            );
        }
    }

    eprintln!("Done. {} points.", (steps + 1) * (steps + 1));
}

// ============================================================================
// MODE 2: LATIN HYPERCUBE SAMPLING
// ============================================================================

/// Simple LHS implementation using a permutation-based approach.
fn latin_hypercube_sample(n_samples: usize, n_dims: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = SimpleRng::new(seed);
    let mut result = vec![vec![0.0; n_dims]; n_samples];

    for dim in 0..n_dims {
        // Create permutation of [0..n_samples]
        let mut indices: Vec<usize> = (0..n_samples).collect();
        // Fisher-Yates shuffle
        for i in (1..n_samples).rev() {
            let j = rng.next_u64() as usize % (i + 1);
            indices.swap(i, j);
        }

        for (sample_idx, &perm_idx) in indices.iter().enumerate() {
            // Uniform within stratum
            let u = rng.next_f64();
            result[sample_idx][dim] = (perm_idx as f64 + u) / n_samples as f64;
        }
    }

    result
}

fn run_lhs(config: &Config) {
    let components = CoreComponent::all();
    let n_dims = components.len(); // 7

    eprintln!("=== Latin Hypercube Sampling ===");
    eprintln!(
        "Samples: {}, Dimensions: {}, Seed: {}",
        config.samples, n_dims, config.seed
    );

    // CSV header
    println!(
        "integration,binding,workspace,attention,recursion,efficacy,knowledge,\
         substrate,consciousness,core_minimum,weighted_sum,limiting_factor,collapsed"
    );

    let samples = latin_hypercube_sample(config.samples, n_dims, config.seed);

    let mut min_viable: Option<(f64, Vec<f64>)> = None; // (sum of components, values)
    let mut max_consciousness = 0.0f64;
    let mut collapsed_count = 0usize;

    for sample in &samples {
        let mut state = ConsciousnessStateV2::new();
        for (idx, &comp) in components.iter().enumerate() {
            state.set_core(comp, sample[idx]);
        }

        let mut eq = ConsciousnessEquationV2::new();
        let result = eq.compute(&state);

        let collapsed = result.consciousness < 0.01;
        if collapsed {
            collapsed_count += 1;
        }

        // Track minimum viable configuration
        if result.consciousness > 0.1 {
            let comp_sum: f64 = sample.iter().sum();
            if min_viable.is_none() || comp_sum < min_viable.as_ref().unwrap().0 {
                min_viable = Some((comp_sum, sample.clone()));
            }
        }

        max_consciousness = max_consciousness.max(result.consciousness);

        println!(
            "{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.2},{:.6},{:.6},{:.6},{},{}",
            sample[0],
            sample[1],
            sample[2],
            sample[3],
            sample[4],
            sample[5],
            sample[6],
            1.0, // default substrate
            result.consciousness,
            result.core_minimum,
            result.weighted_sum,
            result.limiting_factor.as_str(),
            collapsed
        );
    }

    eprintln!();
    eprintln!("--- Summary ---");
    eprintln!("Max consciousness: {:.6}", max_consciousness);
    eprintln!(
        "Collapsed: {}/{} ({:.1}%)",
        collapsed_count,
        config.samples,
        100.0 * collapsed_count as f64 / config.samples as f64
    );

    if let Some((sum, vals)) = min_viable {
        eprintln!("Minimum viable config (consciousness > 0.1):");
        eprintln!("  Component sum: {:.3}", sum);
        for (idx, comp) in components.iter().enumerate() {
            eprintln!("  {}: {:.3}", comp.as_str(), vals[idx]);
        }
    } else {
        eprintln!("No viable configuration found (all consciousness < 0.1)");
    }
}

// ============================================================================
// MODE 3: PHASE TRANSITION DETECTION
// ============================================================================

fn run_transition(config: &Config) {
    let components = CoreComponent::all();

    eprintln!("=== Phase Transition Detection ===");
    eprintln!(
        "Resolution: {}, Fixed: {}",
        config.resolution, config.fixed_value
    );

    // CSV header
    println!("component,value,consciousness,gradient,is_transition");

    let gradient_threshold = 2.0; // Sharp change indicator
    let steps = config.resolution;

    for comp in &components {
        let mut prev_c: Option<f64> = None;
        let mut transitions = Vec::new();

        for i in 0..=steps {
            let val = i as f64 / steps as f64;

            let mut state = ConsciousnessStateV2::new();
            for c in &components {
                state.set_core(*c, config.fixed_value);
            }
            state.set_core(*comp, val);

            let mut eq = ConsciousnessEquationV2::new();
            let result = eq.compute(&state);

            let gradient = if let Some(prev) = prev_c {
                let dv = 1.0 / steps as f64;
                (result.consciousness - prev) / dv
            } else {
                0.0
            };

            let is_transition = gradient.abs() > gradient_threshold;
            if is_transition {
                transitions.push((val, gradient));
            }

            println!(
                "{},{:.4},{:.6},{:.4},{}",
                comp.as_str(),
                val,
                result.consciousness,
                gradient,
                is_transition
            );

            prev_c = Some(result.consciousness);
        }

        if transitions.is_empty() {
            eprintln!("  {}: No sharp transitions detected", comp.as_str());
        } else {
            for (val, grad) in &transitions {
                eprintln!(
                    "  {}: TRANSITION at {:.3} (gradient={:.2})",
                    comp.as_str(),
                    val,
                    grad
                );
            }
        }
    }
}

// ============================================================================
// MODE 4: SUBSTRATE SWEEP
// ============================================================================

fn run_substrate(config: &Config) {
    let components = CoreComponent::all();

    eprintln!("=== Substrate Feasibility Sweep ===");
    eprintln!(
        "Substrates: {:?}, Resolution: {}",
        config.substrate_values, config.resolution
    );

    // CSV header
    println!("substrate_feasibility,component,value,consciousness,core_minimum,limiting_factor");

    let steps = config.resolution;

    for &substrate in &config.substrate_values {
        for comp in &components {
            for i in 0..=steps {
                let val = i as f64 / steps as f64;

                let mut state = ConsciousnessStateV2::new();
                for c in &components {
                    state.set_core(*c, config.fixed_value);
                }
                state.set_core(*comp, val);
                state.substrate_feasibility = substrate;

                let mut eq = ConsciousnessEquationV2::new();
                let result = eq.compute(&state);

                println!(
                    "{:.2},{},{:.3},{:.6},{:.6},{}",
                    substrate,
                    comp.as_str(),
                    val,
                    result.consciousness,
                    result.core_minimum,
                    result.limiting_factor.as_str()
                );
            }
        }
    }

    // Summary: consciousness at default (all 0.5) for each substrate level
    eprintln!();
    eprintln!(
        "--- Substrate Impact at Default Config (all components = {}) ---",
        config.fixed_value
    );
    for &substrate in &config.substrate_values {
        let mut state = ConsciousnessStateV2::new();
        for c in &components {
            state.set_core(*c, config.fixed_value);
        }
        state.substrate_feasibility = substrate;

        let mut eq = ConsciousnessEquationV2::new();
        let result = eq.compute(&state);
        eprintln!(
            "  Substrate {:.2}: consciousness = {:.6}",
            substrate, result.consciousness
        );
    }
}

// ============================================================================
// SIMPLE RNG (no external dep needed)
// ============================================================================

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        // SplitMix64
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    let config = parse_args();

    match config.mode {
        Mode::Heatmap => run_heatmap(&config),
        Mode::Lhs => run_lhs(&config),
        Mode::Transition => run_transition(&config),
        Mode::Substrate => run_substrate(&config),
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monotonic_individual_components() {
        // Each component should monotonically increase consciousness
        // when others are fixed at 0.5
        for comp in CoreComponent::all() {
            let mut prev = 0.0f64;
            for i in 0..=10 {
                let val = i as f64 / 10.0;
                let mut state = ConsciousnessStateV2::new(); // all 0.5
                state.set_core(comp, val);

                let mut eq = ConsciousnessEquationV2::new();
                let result = eq.compute(&state);

                assert!(
                    result.consciousness >= prev - 1e-6,
                    "{} at {:.1}: {:.6} < prev {:.6}",
                    comp.as_str(),
                    val,
                    result.consciousness,
                    prev
                );
                prev = result.consciousness;
            }
        }
    }

    #[test]
    fn test_collapsed_when_component_zero() {
        // Any component at 0 should produce very low consciousness
        for comp in CoreComponent::all() {
            let mut state = ConsciousnessStateV2::new(); // all 0.5
            state.set_core(comp, 0.0);

            let mut eq = ConsciousnessEquationV2::new();
            let result = eq.compute(&state);

            assert!(
                result.consciousness < 0.1,
                "{} at 0.0 produced consciousness {:.6} (expected < 0.1)",
                comp.as_str(),
                result.consciousness
            );
        }
    }

    #[test]
    fn test_substrate_scales_consciousness() {
        let mut state = ConsciousnessStateV2::new(); // all 0.5
        state.substrate_feasibility = 1.0;
        let mut eq = ConsciousnessEquationV2::new();
        let full = eq.compute(&state).consciousness;

        let mut state_half = ConsciousnessStateV2::new();
        state_half.substrate_feasibility = 0.5;
        let mut eq2 = ConsciousnessEquationV2::new();
        let half = eq2.compute(&state_half).consciousness;

        assert!(
            (half - full * 0.5).abs() < 0.01,
            "Expected substrate to scale linearly: full={:.4}, half={:.4}",
            full,
            half
        );
    }

    #[test]
    fn test_lhs_coverage() {
        // LHS should cover the space reasonably
        let samples = latin_hypercube_sample(100, 7, 42);
        assert_eq!(samples.len(), 100);
        assert_eq!(samples[0].len(), 7);

        // Each dimension should have values spread across [0,1]
        for dim in 0..7 {
            let mut vals: Vec<f64> = samples.iter().map(|s| s[dim]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // First value should be in bottom 10%, last in top 10%
            assert!(vals[0] < 0.1, "Dim {} min too high: {:.3}", dim, vals[0]);
            assert!(vals[99] > 0.9, "Dim {} max too low: {:.3}", dim, vals[99]);
        }
    }

    #[test]
    fn test_max_consciousness_at_all_ones() {
        let mut state = ConsciousnessStateV2::new();
        for comp in CoreComponent::all() {
            state.set_core(comp, 1.0);
        }

        let mut eq = ConsciousnessEquationV2::new();
        let result = eq.compute(&state);

        assert!(
            result.consciousness > 0.5,
            "All-ones should produce high consciousness, got {:.6}",
            result.consciousness
        );
    }
}