// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Consciousness Feature Evaluation Benchmark
//!
//! Evaluates the 7 consciousness theory upgrades across quality metrics:
//!
//! 1. **Sensitivity**: How well does C(t) respond to component changes?
//! 2. **Discrimination**: Can the equation distinguish qualitatively different states?
//! 3. **Stability**: Does C(t) converge or oscillate over repeated cycles?
//! 4. **Theoretical correctness**: Does the equation match theory predictions?
//! 5. **Dynamic range**: What fraction of [0, 1] does C(t) actually use?
//!
//! ## Features Evaluated
//!
//! - `structured_bottleneck`: Necessary {Phi,B,W} + Amplifiers {A,R,E,K}
//! - `ctc_wiring`: Binding-gated GWT entry (Fries 2015 CTC)
//! - `continuous_hot`: Sub-level HOT depth resolution
//! - `unified_precision`: Dynamic HFE precision (Parr & Friston 2019)
//! - `interoceptive_inference`: Seth (2021) beast machine
//! - `england_dissipation`: England (2013) dissipation-driven adaptation
//! - `iit4`: IIT 4.0 concept structures (Albantakis 2023)
//!
//! ## Usage
//!
//! ```bash
//! # Default features (baseline)
//! cargo run --example benchmark_consciousness_features
//!
//! # With structured bottleneck
//! cargo run --example benchmark_consciousness_features --features structured_bottleneck
//!
//! # With all features
//! cargo run --example benchmark_consciousness_features --features "structured_bottleneck,ctc_wiring,continuous_hot,unified_precision,interoceptive_inference,england_dissipation,iit4"
//! ```

use symthaea::consciousness::consciousness_equation_v2::{
    ConsciousnessEquationV2, ConsciousnessStateV2, CoreComponent, EquationConfig,
};

fn main() {
    println!("=== Consciousness Feature Evaluation Benchmark ===\n");

    // Report active features
    print_active_features();
    println!();

    // Run all evaluations
    let sensitivity = eval_sensitivity();
    let discrimination = eval_discrimination();
    let stability = eval_stability();
    let theoretical = eval_theoretical_correctness();
    let dynamic_range = eval_dynamic_range();

    // Summary
    println!("\n=== SUMMARY ===\n");
    println!("  Sensitivity score:            {:.3}", sensitivity);
    println!("  Discrimination score:         {:.3}", discrimination);
    println!("  Stability score:              {:.3}", stability);
    println!("  Theoretical correctness:      {:.3}", theoretical);
    println!("  Dynamic range:                {:.3}", dynamic_range);

    let composite = (sensitivity + discrimination + stability + theoretical + dynamic_range) / 5.0;
    println!("\n  COMPOSITE SCORE:              {:.3}", composite);
    println!();

    // Feature-specific evaluations
    eval_structured_bottleneck();
    eval_thermodynamic_features();
    eval_iit4_features();
}

fn print_active_features() {
    println!("Active features:");

    let features = [
        (
            "structured_bottleneck",
            cfg!(feature = "structured_bottleneck"),
        ),
        ("ctc_wiring", cfg!(feature = "ctc_wiring")),
        ("continuous_hot", cfg!(feature = "continuous_hot")),
        ("unified_precision", cfg!(feature = "unified_precision")),
        (
            "interoceptive_inference",
            cfg!(feature = "interoceptive_inference"),
        ),
        ("england_dissipation", cfg!(feature = "england_dissipation")),
        ("iit4", cfg!(feature = "iit4")),
    ];

    for (name, enabled) in features {
        let marker = if enabled { "[ON] " } else { "[off]" };
        println!("  {} {}", marker, name);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. SENSITIVITY: How well does C(t) respond to component changes?
// ═══════════════════════════════════════════════════════════════════════════════

fn eval_sensitivity() -> f64 {
    println!("--- 1. Sensitivity ---");

    let mut total_sensitivity = 0.0;
    let mut count = 0;

    // For each component, measure dC/d(component) at the midpoint
    for component in CoreComponent::all() {
        let mut eq = ConsciousnessEquationV2::new();
        let epsilon = 0.1;

        let mut state_lo = uniform_state(0.5);
        let mut state_hi = uniform_state(0.5);
        state_lo.set_core(component, 0.5 - epsilon);
        state_hi.set_core(component, 0.5 + epsilon);

        let c_lo = eq.compute(&state_lo).consciousness;
        let mut eq2 = ConsciousnessEquationV2::new();
        let c_hi = eq2.compute(&state_hi).consciousness;

        let sensitivity = (c_hi - c_lo).abs() / (2.0 * epsilon);
        println!(
            "  {:12} sensitivity: {:.4} (C: {:.3} -> {:.3})",
            component.as_str(),
            sensitivity,
            c_lo,
            c_hi
        );

        total_sensitivity += sensitivity;
        count += 1;
    }

    let avg = total_sensitivity / count as f64;
    println!("  Average sensitivity: {:.4}", avg);
    // Normalize: 0.5 sensitivity = perfect score
    (avg / 0.5).min(1.0)
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. DISCRIMINATION: Can the equation distinguish qualitatively different states?
// ═══════════════════════════════════════════════════════════════════════════════

fn eval_discrimination() -> f64 {
    println!("\n--- 2. Discrimination ---");

    // Define qualitatively different states
    let scenarios: Vec<(&str, fn() -> ConsciousnessStateV2, &str)> = vec![
        (
            "fully_conscious",
            fully_conscious_state,
            "High all components — should be near 1.0",
        ),
        (
            "unconscious",
            unconscious_state,
            "Low all components — should be near 0.0",
        ),
        (
            "binding_impaired",
            binding_impaired_state,
            "Low binding only — should reduce C",
        ),
        (
            "high_attention_low_phi",
            high_attention_low_phi_state,
            "High attention but low Phi",
        ),
        (
            "dreaming",
            dreaming_state,
            "High binding/workspace but low attention",
        ),
        (
            "zombie",
            zombie_state,
            "High efficacy but low integration — philosophical zombie",
        ),
        (
            "meditative",
            meditative_state,
            "High recursion/knowledge, moderate others",
        ),
    ];

    let mut results = Vec::new();

    for (name, state_fn, desc) in &scenarios {
        let mut eq = ConsciousnessEquationV2::new();
        let state = state_fn();
        let result = eq.compute(&state);
        println!(
            "  {:25} C={:.3}  lim={:12}  ({})",
            name,
            result.consciousness,
            result.limiting_factor.as_str(),
            desc
        );
        results.push((*name, result.consciousness));
    }

    // Score: how well are these ordered?
    let mut score = 0.0;
    let total_checks = 6.0;

    // fully_conscious > unconscious
    if results[0].1 > results[1].1 + 0.1 {
        score += 1.0;
    }
    // fully_conscious > binding_impaired
    if results[0].1 > results[2].1 {
        score += 1.0;
    }
    // binding_impaired < fully_conscious (binding matters)
    if results[2].1 < results[0].1 - 0.05 {
        score += 1.0;
    }
    // zombie should have lower consciousness than fully_conscious
    if results[5].1 < results[0].1 {
        score += 1.0;
    }
    // dreaming should have some consciousness (binding + workspace present)
    if results[4].1 > 0.1 {
        score += 1.0;
    }
    // Dynamic range: max - min > 0.5
    let max_c = results.iter().map(|r| r.1).fold(0.0f64, f64::max);
    let min_c = results.iter().map(|r| r.1).fold(1.0f64, f64::min);
    if max_c - min_c > 0.5 {
        score += 1.0;
    }

    let discrimination = score / total_checks;
    println!(
        "  Discrimination score: {:.3} ({}/{} checks passed)",
        discrimination, score as u32, total_checks as u32
    );
    discrimination
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. STABILITY: Does C(t) converge over repeated cycles?
// ═══════════════════════════════════════════════════════════════════════════════

fn eval_stability() -> f64 {
    println!("\n--- 3. Stability ---");

    let mut eq = ConsciousnessEquationV2::new();
    let state = uniform_state(0.6);

    let mut values = Vec::new();
    for _ in 0..100 {
        let result = eq.compute(&state);
        values.push(result.consciousness);
    }

    // Measure convergence: variance of last 20 values
    let last_20 = &values[80..];
    let mean: f64 = last_20.iter().sum::<f64>() / 20.0;
    let variance: f64 = last_20.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 20.0;
    let std_dev = variance.sqrt();

    // Measure drift: difference between first 10 and last 10 means
    let early_mean: f64 = values[..10].iter().sum::<f64>() / 10.0;
    let late_mean: f64 = values[90..].iter().sum::<f64>() / 10.0;
    let drift = (late_mean - early_mean).abs();

    println!("  Convergence std_dev:  {:.6}", std_dev);
    println!("  Drift (first→last):   {:.6}", drift);
    println!("  Final C(t):           {:.4}", values.last().unwrap());

    // Score: low variance + low drift = high stability
    let stability = 1.0 - (std_dev * 10.0 + drift).min(1.0);
    println!("  Stability score:      {:.3}", stability);
    stability
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. THEORETICAL CORRECTNESS: Does the equation match theory predictions?
// ═══════════════════════════════════════════════════════════════════════════════

fn eval_theoretical_correctness() -> f64 {
    println!("\n--- 4. Theoretical Correctness ---");

    let mut score = 0.0;
    let total = 8.0;

    // Test 1: IIT predicts high Phi → high consciousness
    {
        let mut eq = ConsciousnessEquationV2::new();
        let mut state = uniform_state(0.5);
        state.set_core(CoreComponent::Integration, 0.9);
        let c_high_phi = eq.compute(&state).consciousness;

        let mut eq2 = ConsciousnessEquationV2::new();
        let mut state2 = uniform_state(0.5);
        state2.set_core(CoreComponent::Integration, 0.1);
        let c_low_phi = eq2.compute(&state2).consciousness;

        let pass = c_high_phi > c_low_phi;
        if pass {
            score += 1.0;
        }
        println!(
            "  [{}] IIT: High Phi ({:.3}) > Low Phi ({:.3})",
            if pass { "PASS" } else { "FAIL" },
            c_high_phi,
            c_low_phi
        );
    }

    // Test 2: GWT predicts workspace access is necessary
    {
        let mut eq = ConsciousnessEquationV2::new();
        let mut state = uniform_state(0.8);
        state.set_core(CoreComponent::Workspace, 0.05);
        let c = eq.compute(&state).consciousness;
        let pass = c < 0.3;
        if pass {
            score += 1.0;
        }
        println!(
            "  [{}] GWT: No workspace → low C ({:.3} < 0.3)",
            if pass { "PASS" } else { "FAIL" },
            c
        );
    }

    // Test 3: Binding theory predicts synchrony enables consciousness
    {
        let mut eq = ConsciousnessEquationV2::new();
        let mut state = uniform_state(0.8);
        state.set_core(CoreComponent::Binding, 0.05);
        let c = eq.compute(&state).consciousness;
        let pass = c < 0.3;
        if pass {
            score += 1.0;
        }
        println!(
            "  [{}] Binding: No binding → low C ({:.3} < 0.3)",
            if pass { "PASS" } else { "FAIL" },
            c
        );
    }

    // Test 4: HOT theory predicts recursion enriches (with structured_bottleneck: amplifies but doesn't gate)
    {
        let mut eq1 = ConsciousnessEquationV2::new();
        let mut state1 = uniform_state(0.6);
        state1.set_core(CoreComponent::Recursion, 0.9);
        let c_high_r = eq1.compute(&state1).consciousness;

        let mut eq2 = ConsciousnessEquationV2::new();
        let mut state2 = uniform_state(0.6);
        state2.set_core(CoreComponent::Recursion, 0.1);
        let c_low_r = eq2.compute(&state2).consciousness;

        let pass = c_high_r > c_low_r;
        if pass {
            score += 1.0;
        }
        println!(
            "  [{}] HOT: High recursion ({:.3}) > Low recursion ({:.3})",
            if pass { "PASS" } else { "FAIL" },
            c_high_r,
            c_low_r
        );
    }

    // Test 5: First-order consciousness should be possible (high Phi/B/W, low R)
    // This is the key test for structured_bottleneck
    {
        let mut eq = ConsciousnessEquationV2::new();
        let mut state = ConsciousnessStateV2::new();
        state.set_core(CoreComponent::Integration, 0.8);
        state.set_core(CoreComponent::Binding, 0.8);
        state.set_core(CoreComponent::Workspace, 0.8);
        state.set_core(CoreComponent::Attention, 0.7);
        state.set_core(CoreComponent::Recursion, 0.05); // Very low HOT
        state.set_core(CoreComponent::Efficacy, 0.6);
        state.set_core(CoreComponent::Knowledge, 0.6);
        let c = eq.compute(&state).consciousness;

        // With structured_bottleneck: should still be conscious (>0.2)
        // Without: will collapse due to softmin
        let threshold = if cfg!(feature = "structured_bottleneck") {
            0.2
        } else {
            0.05
        };
        let pass = c > threshold;
        if pass {
            score += 1.0;
        }
        println!(
            "  [{}] First-order: Low HOT but high Phi/B/W → C={:.3} (threshold={:.2})",
            if pass { "PASS" } else { "FAIL" },
            c,
            threshold
        );
    }

    // Test 6: Attention Schema Theory: attention modulates but is not absolute gate
    {
        let mut eq = ConsciousnessEquationV2::new();
        let mut state = uniform_state(0.7);
        state.set_core(CoreComponent::Attention, 0.0);
        let c = eq.compute(&state).consciousness;

        // With structured_bottleneck: attention is amplifier, C > 0
        // Without: C ≈ 0 from softmin
        let threshold = if cfg!(feature = "structured_bottleneck") {
            0.1
        } else {
            0.0
        };
        let pass = c >= threshold;
        if pass {
            score += 1.0;
        }
        println!(
            "  [{}] AST: Zero attention → C={:.3} (>={:.2})",
            if pass { "PASS" } else { "FAIL" },
            c,
            threshold
        );
    }

    // Test 7: Temporal continuity — consciousness persists
    {
        let mut eq = ConsciousnessEquationV2::new();
        let high_state = uniform_state(0.8);
        // Build up temporal memory
        for _ in 0..20 {
            eq.compute(&high_state);
        }
        // Sudden drop
        let low_state = uniform_state(0.3);
        let result = eq.compute(&low_state);
        let c = result.consciousness;
        let rho = result.temporal_continuity;

        let pass = rho > 0.5; // temporal memory buffers the drop
        if pass {
            score += 1.0;
        }
        println!(
            "  [{}] Temporal: After sustained high, sudden drop → rho={:.3} (>0.5)",
            if pass { "PASS" } else { "FAIL" },
            rho
        );
    }

    // Test 8: Substrate feasibility gates consciousness
    {
        let mut eq = ConsciousnessEquationV2::new();
        let mut state = uniform_state(0.8);
        state.substrate_feasibility = 0.1;
        let c = eq.compute(&state).consciousness;
        let pass = c < 0.3;
        if pass {
            score += 1.0;
        }
        println!(
            "  [{}] Substrate: Low feasibility → C={:.3} (<0.3)",
            if pass { "PASS" } else { "FAIL" },
            c
        );
    }

    let correctness = score / total;
    println!(
        "  Theoretical correctness: {:.3} ({}/{} passed)",
        correctness, score as u32, total as u32
    );
    correctness
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. DYNAMIC RANGE: What fraction of [0, 1] does C(t) actually use?
// ═══════════════════════════════════════════════════════════════════════════════

fn eval_dynamic_range() -> f64 {
    println!("\n--- 5. Dynamic Range ---");

    let mut min_c = 1.0f64;
    let mut max_c = 0.0f64;
    let n = 1000;

    // Simple random-ish sweep using LCG
    let mut rng = 12345u64;
    for _ in 0..n {
        let mut eq = ConsciousnessEquationV2::new();
        let mut state = ConsciousnessStateV2::new();
        for component in CoreComponent::all() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = (rng >> 33) as f64 / (1u64 << 31) as f64;
            state.set_core(component, val);
        }
        let c = eq.compute(&state).consciousness;
        min_c = min_c.min(c);
        max_c = max_c.max(c);
    }

    let range = max_c - min_c;
    println!("  Min C(t): {:.4}", min_c);
    println!("  Max C(t): {:.4}", max_c);
    println!("  Range:    {:.4}", range);
    println!("  Dynamic range score: {:.3}", range);
    range
}

// ═══════════════════════════════════════════════════════════════════════════════
// FEATURE-SPECIFIC EVALUATIONS
// ═══════════════════════════════════════════════════════════════════════════════

fn eval_structured_bottleneck() {
    println!("\n--- Structured Bottleneck Analysis ---");

    // Key question: does consciousness survive when amplifiers are low
    // but necessary conditions are high?
    let scenarios = [
        ("necessary=0.8, amplifiers=0.8", 0.8, 0.8),
        ("necessary=0.8, amplifiers=0.5", 0.8, 0.5),
        ("necessary=0.8, amplifiers=0.2", 0.8, 0.2),
        ("necessary=0.8, amplifiers=0.0", 0.8, 0.0),
        ("necessary=0.5, amplifiers=0.8", 0.5, 0.8),
        ("necessary=0.2, amplifiers=0.8", 0.2, 0.8),
    ];

    for (label, nec, amp) in scenarios {
        let mut eq = ConsciousnessEquationV2::new();
        let mut state = ConsciousnessStateV2::new();
        state.set_core(CoreComponent::Integration, nec);
        state.set_core(CoreComponent::Binding, nec);
        state.set_core(CoreComponent::Workspace, nec);
        state.set_core(CoreComponent::Attention, amp);
        state.set_core(CoreComponent::Recursion, amp);
        state.set_core(CoreComponent::Efficacy, amp);
        state.set_core(CoreComponent::Knowledge, amp);

        let result = eq.compute(&state);
        println!(
            "  {:40} C={:.4}  necessary_min={:.4}  amplifier={:.4}",
            label, result.consciousness, result.necessary_minimum, result.amplifier_factor,
        );
    }

    // Critical test: does lowering amplifiers reduce C less than lowering necessary?
    let mut eq1 = ConsciousnessEquationV2::new();
    let mut s1 = uniform_state(0.8);
    s1.set_core(CoreComponent::Attention, 0.1);
    let c_low_amp = eq1.compute(&s1).consciousness;

    let mut eq2 = ConsciousnessEquationV2::new();
    let mut s2 = uniform_state(0.8);
    s2.set_core(CoreComponent::Integration, 0.1);
    let c_low_nec = eq2.compute(&s2).consciousness;

    println!("\n  Asymmetry test:");
    println!("    Low Attention (amplifier):   C={:.4}", c_low_amp);
    println!("    Low Integration (necessary): C={:.4}", c_low_nec);
    if cfg!(feature = "structured_bottleneck") {
        println!("    [structured_bottleneck ON] Amplifier should hurt less than necessary");
        if c_low_amp > c_low_nec * 1.5 {
            println!(
                "    PASS: Amplifier loss ({:.3}) < necessary loss ({:.3})",
                c_low_amp, c_low_nec
            );
        } else {
            println!(
                "    MARGINAL: Amplifier ({:.3}) vs necessary ({:.3}) — ratio: {:.2}x",
                c_low_amp,
                c_low_nec,
                c_low_amp / c_low_nec.max(0.001)
            );
        }
    } else {
        println!("    [structured_bottleneck OFF] Both should collapse similarly via softmin");
    }
}

fn eval_thermodynamic_features() {
    println!("\n--- Thermodynamic Features ---");

    if cfg!(feature = "england_dissipation") {
        println!("  [england_dissipation ON]");
        println!("  England signal fires when entropy + order growth + PE reduction align");
        println!("  Dissipation efficiency uses multiplicative (order * entropy) formula");
        println!("  This resolves the Prigogine-England tension at edge of chaos");
    } else {
        println!("  [england_dissipation OFF]");
        println!("  Using Prigogine divisive (order / entropy) efficiency");
        println!("  Recommendation: Enable for edge-of-chaos operation (lambda ~0.273)");
    }

    // Efficiency formula comparison (illustrative)
    println!("\n  Efficiency formula comparison (order=0.6, entropy_rate=0.4):");
    let order = 0.6;
    let entropy = 0.4;
    let prigogine_eff = order / entropy;
    let england_eff = order * entropy;
    println!(
        "    Prigogine (order/entropy): {:.3} — favors low entropy",
        prigogine_eff
    );
    println!(
        "    England  (order*entropy):  {:.3} — favors productive dissipation",
        england_eff
    );
    println!(
        "    Active mode: {}",
        if cfg!(feature = "england_dissipation") {
            "England"
        } else {
            "Prigogine"
        }
    );
}

fn eval_iit4_features() {
    println!("\n--- IIT 4.0 Features ---");

    // Note: IIT 4.0 concept structures are internal to symthaea-core and cannot
    // be benchmarked from an example binary. Use `cargo test -p symthaea-core --features iit4`
    // to run IIT 4.0 unit tests directly.

    #[cfg(not(feature = "iit4"))]
    {
        println!("  [iit4 OFF] Concept structures not available");
        println!("  Enable with --features iit4 to see concept-level analysis");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER: State Constructors
// ═══════════════════════════════════════════════════════════════════════════════

fn uniform_state(value: f64) -> ConsciousnessStateV2 {
    let mut state = ConsciousnessStateV2::new();
    for component in CoreComponent::all() {
        state.set_core(component, value);
    }
    state
}

fn fully_conscious_state() -> ConsciousnessStateV2 {
    let mut state = ConsciousnessStateV2::new();
    state.set_core(CoreComponent::Integration, 0.9);
    state.set_core(CoreComponent::Binding, 0.85);
    state.set_core(CoreComponent::Workspace, 0.88);
    state.set_core(CoreComponent::Attention, 0.87);
    state.set_core(CoreComponent::Recursion, 0.82);
    state.set_core(CoreComponent::Efficacy, 0.78);
    state.set_core(CoreComponent::Knowledge, 0.80);
    state
}

fn unconscious_state() -> ConsciousnessStateV2 {
    let mut state = ConsciousnessStateV2::new();
    for component in CoreComponent::all() {
        state.set_core(component, 0.05);
    }
    state
}

fn binding_impaired_state() -> ConsciousnessStateV2 {
    let mut state = fully_conscious_state();
    state.set_core(CoreComponent::Binding, 0.1);
    state
}

fn high_attention_low_phi_state() -> ConsciousnessStateV2 {
    let mut state = ConsciousnessStateV2::new();
    state.set_core(CoreComponent::Integration, 0.1);
    state.set_core(CoreComponent::Binding, 0.5);
    state.set_core(CoreComponent::Workspace, 0.5);
    state.set_core(CoreComponent::Attention, 0.95);
    state.set_core(CoreComponent::Recursion, 0.5);
    state.set_core(CoreComponent::Efficacy, 0.5);
    state.set_core(CoreComponent::Knowledge, 0.5);
    state
}

fn dreaming_state() -> ConsciousnessStateV2 {
    let mut state = ConsciousnessStateV2::new();
    state.set_core(CoreComponent::Integration, 0.4);
    state.set_core(CoreComponent::Binding, 0.7);
    state.set_core(CoreComponent::Workspace, 0.6);
    state.set_core(CoreComponent::Attention, 0.1); // Low external attention
    state.set_core(CoreComponent::Recursion, 0.5);
    state.set_core(CoreComponent::Efficacy, 0.2);
    state.set_core(CoreComponent::Knowledge, 0.3);
    state
}

fn zombie_state() -> ConsciousnessStateV2 {
    let mut state = ConsciousnessStateV2::new();
    state.set_core(CoreComponent::Integration, 0.1); // No integration
    state.set_core(CoreComponent::Binding, 0.3);
    state.set_core(CoreComponent::Workspace, 0.3);
    state.set_core(CoreComponent::Attention, 0.5);
    state.set_core(CoreComponent::Recursion, 0.1);
    state.set_core(CoreComponent::Efficacy, 0.9); // High efficacy — behaves but isn't conscious
    state.set_core(CoreComponent::Knowledge, 0.2);
    state
}

fn meditative_state() -> ConsciousnessStateV2 {
    let mut state = ConsciousnessStateV2::new();
    state.set_core(CoreComponent::Integration, 0.7);
    state.set_core(CoreComponent::Binding, 0.6);
    state.set_core(CoreComponent::Workspace, 0.5);
    state.set_core(CoreComponent::Attention, 0.4); // Defocused
    state.set_core(CoreComponent::Recursion, 0.9); // Deep self-reflection
    state.set_core(CoreComponent::Efficacy, 0.3);
    state.set_core(CoreComponent::Knowledge, 0.8);
    state
}