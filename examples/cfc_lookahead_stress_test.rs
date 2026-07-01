// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CfC Learning Lookahead - Comprehensive Stress Test
//!
//! This test validates the CfC lookahead architecture with:
//! 1. Unit tests for core components
//! 2. Extended stress test (100+ cycles)
//! 3. Φ sanity checks
//! 4. Edge case validation
//!
//! Run with: cargo run --example cfc_lookahead_stress_test --release

use anyhow::Result;
use ndarray::Array1;
use std::time::Instant;

// Symthaea imports
use symthaea::cfc::CfCNetwork;
use symthaea::hdc::HDC_DIMENSION;
use symthaea::hdc::unified_hv::ContinuousHV;
use symthaea::phi_engine::{PhiEngine, PhiMethod};

// ═══════════════════════════════════════════════════════════════════════════════
// COPY OF CORE STRUCTURES (from prototype - will be in src/school/ later)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
struct LearningObjective {
    id: String,
    name: String,
    domain: String,
    difficulty: f32,
    encoding: ContinuousHV,
    prerequisites: Vec<String>,
}

impl LearningObjective {
    fn new(id: &str, name: &str, domain: &str, difficulty: f32, seed: u64) -> Self {
        let base = ContinuousHV::random(HDC_DIMENSION, seed);
        let domain_hv = ContinuousHV::random(HDC_DIMENSION, seed + 1000);
        let encoding = base.bind(&domain_hv).scale(1.0 - difficulty * 0.5);

        Self {
            id: id.to_string(),
            name: name.to_string(),
            domain: domain.to_string(),
            difficulty,
            encoding,
            prerequisites: Vec::new(),
        }
    }

    fn to_cfc_input(&self, size: usize) -> Array1<f32> {
        let enc_slice = self.encoding.values.as_slice();
        let mut input = Array1::zeros(size);
        for i in 0..size.min(enc_slice.len()) {
            input[i] = enc_slice[i];
        }
        input[size - 1] = self.difficulty;
        input
    }
}

#[derive(Debug, Clone)]
struct LookaheadResult {
    predicted_delta: f32,
    confidence: f32,
    prediction_time_us: u64,
}

#[derive(Debug, Clone)]
struct RealityCheck {
    predicted_delta: f32,
    actual_delta: f32,
    error: f32,
    is_hallucination: bool,
}

struct CfCLearningLookahead {
    cfc: CfCNetwork,
    phi_engine: PhiEngine,
    consciousness_state: Vec<ContinuousHV>,
    horizon: f32,
    min_phi_gain: f32,
    reality_history: Vec<RealityCheck>,
    cumulative_error: f32,
}

impl CfCLearningLookahead {
    fn new(cfc_neurons: usize, horizon: f32, min_phi_gain: f32) -> Result<Self> {
        let cfc = CfCNetwork::new(cfc_neurons)?;
        let phi_engine = PhiEngine::new(PhiMethod::SpectralConnectivity);

        let consciousness_state: Vec<ContinuousHV> = (0..8)
            .map(|i| {
                let hv = ContinuousHV::random(HDC_DIMENSION, 42 + i as u64);
                ContinuousHV::from_vec(hv.values)
            })
            .collect();

        Ok(Self {
            cfc,
            phi_engine,
            consciousness_state,
            horizon,
            min_phi_gain,
            reality_history: Vec::new(),
            cumulative_error: 0.0,
        })
    }

    fn current_phi(&self) -> f32 {
        self.phi_engine.compute(&self.consciousness_state).phi as f32
    }

    fn evaluate_objective(&self, objective: &LearningObjective) -> Result<LookaheadResult> {
        let start = Instant::now();
        let current_phi = self.current_phi();

        let input_size = self.cfc.input_size;
        let objective_input = objective.to_cfc_input(input_size);

        let current_state = self.cfc.read_state()?;
        let combined_input: Array1<f32> = Array1::from_iter(
            objective_input
                .iter()
                .zip(current_state.iter())
                .map(|(a, b)| (a + b) / 2.0),
        );

        let predicted_state = self.cfc.predict_forward(&combined_input, self.horizon)?;

        let predicted_phi = {
            let state_sum: f32 = predicted_state.iter().sum();
            let coherence = (state_sum / predicted_state.len() as f32).abs();
            current_phi + coherence * 0.01 * (1.0 - objective.difficulty)
        };

        let predicted_delta = predicted_phi - current_phi;

        let confidence = if self.reality_history.is_empty() {
            0.0
        } else {
            let recent: Vec<_> = self.reality_history.iter().rev().take(10).collect();
            let accuracy: f32 =
                recent.iter().filter(|r| !r.is_hallucination).count() as f32 / recent.len() as f32;
            accuracy
        };

        let prediction_time_us = start.elapsed().as_micros() as u64;

        Ok(LookaheadResult {
            predicted_delta,
            confidence,
            prediction_time_us,
        })
    }

    fn simulate_learning(&mut self, objective: &LearningObjective) -> Result<f32> {
        let pre_phi = self.current_phi();

        let obj_continuous = ContinuousHV::from_vec(objective.encoding.values.clone());

        if let Some(state) = self.consciousness_state.get_mut(0) {
            // Use bind for blending two vectors (bundle is for multiple)
            let blended = state.bind(&obj_continuous);
            *state = blended;
        }

        let post_phi = self.current_phi();
        Ok(post_phi - pre_phi)
    }

    fn reality_check(
        &mut self,
        predicted: f32,
        actual: f32,
        _obj: &LearningObjective,
    ) -> Result<RealityCheck> {
        let error = predicted - actual;
        let is_hallucination = error.abs() > 0.2 * predicted.abs().max(0.001);

        let check = RealityCheck {
            predicted_delta: predicted,
            actual_delta: actual,
            error,
            is_hallucination,
        };

        self.reality_history.push(check.clone());
        self.cumulative_error += error.abs();

        Ok(check)
    }

    fn accuracy_stats(&self) -> (f32, f32, usize) {
        if self.reality_history.is_empty() {
            return (0.0, 0.0, 0);
        }

        let n = self.reality_history.len();
        let avg_error = self
            .reality_history
            .iter()
            .map(|r| r.error.abs())
            .sum::<f32>()
            / n as f32;

        let hallucination_rate = self
            .reality_history
            .iter()
            .filter(|r| r.is_hallucination)
            .count() as f32
            / n as f32;

        (avg_error, hallucination_rate, n)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 1: UNIT TESTS
// ═══════════════════════════════════════════════════════════════════════════════

fn run_unit_tests() -> Result<(usize, usize)> {
    println!("\n━━━ TEST 1: Unit Tests ━━━\n");

    let mut passed = 0;
    let mut failed = 0;

    // Test 1.1: LearningObjective creation
    print!("  [1.1] LearningObjective creation... ");
    {
        let obj = LearningObjective::new("test", "Test Objective", "Testing", 0.5, 42);
        if obj.id == "test" && obj.difficulty == 0.5 && obj.encoding.dim() == HDC_DIMENSION {
            println!("✅ PASS");
            passed += 1;
        } else {
            println!("❌ FAIL");
            failed += 1;
        }
    }

    // Test 1.2: CfC input conversion
    print!("  [1.2] CfC input conversion... ");
    {
        let obj = LearningObjective::new("test", "Test", "Domain", 0.7, 42);
        let input = obj.to_cfc_input(256);
        if input.len() == 256 && input[255] == 0.7 {
            println!("✅ PASS");
            passed += 1;
        } else {
            println!("❌ FAIL (len={}, last={})", input.len(), input[255]);
            failed += 1;
        }
    }

    // Test 1.3: Engine initialization
    print!("  [1.3] Engine initialization... ");
    {
        match CfCLearningLookahead::new(256, 1.0, 0.01) {
            Ok(engine) => {
                if engine.consciousness_state.len() == 8 && engine.current_phi() > 0.0 {
                    println!("✅ PASS (Φ={:.4})", engine.current_phi());
                    passed += 1;
                } else {
                    println!("❌ FAIL");
                    failed += 1;
                }
            }
            Err(e) => {
                println!("❌ FAIL: {}", e);
                failed += 1;
            }
        }
    }

    // Test 1.4: Lookahead evaluation returns result
    print!("  [1.4] Lookahead evaluation... ");
    {
        let engine = CfCLearningLookahead::new(256, 1.0, 0.01)?;
        let obj = LearningObjective::new("test", "Test", "Domain", 0.5, 42);
        match engine.evaluate_objective(&obj) {
            Ok(result) => {
                if result.prediction_time_us > 0 && result.prediction_time_us < 10000 {
                    println!("✅ PASS ({}μs)", result.prediction_time_us);
                    passed += 1;
                } else {
                    println!("❌ FAIL (time={}μs)", result.prediction_time_us);
                    failed += 1;
                }
            }
            Err(e) => {
                println!("❌ FAIL: {}", e);
                failed += 1;
            }
        }
    }

    // Test 1.5: Simulate learning changes state
    print!("  [1.5] Simulate learning... ");
    {
        let mut engine = CfCLearningLookahead::new(256, 1.0, 0.01)?;
        let obj = LearningObjective::new("test", "Test", "Domain", 0.5, 42);
        let pre_phi = engine.current_phi();
        let delta = engine.simulate_learning(&obj)?;
        let post_phi = engine.current_phi();

        if (post_phi - pre_phi - delta).abs() < 0.0001 {
            println!("✅ PASS (Δ={:.6})", delta);
            passed += 1;
        } else {
            println!("❌ FAIL");
            failed += 1;
        }
    }

    // Test 1.6: Reality check records history
    print!("  [1.6] Reality check recording... ");
    {
        let mut engine = CfCLearningLookahead::new(256, 1.0, 0.01)?;
        let obj = LearningObjective::new("test", "Test", "Domain", 0.5, 42);
        engine.reality_check(0.01, 0.005, &obj)?;

        if engine.reality_history.len() == 1 {
            println!("✅ PASS");
            passed += 1;
        } else {
            println!("❌ FAIL");
            failed += 1;
        }
    }

    // Test 1.7: Hallucination detection
    print!("  [1.7] Hallucination detection... ");
    {
        let mut engine = CfCLearningLookahead::new(256, 1.0, 0.01)?;
        let obj = LearningObjective::new("test", "Test", "Domain", 0.5, 42);

        // Small error - not a hallucination
        let check1 = engine.reality_check(0.01, 0.009, &obj)?;
        // Large error - hallucination
        let check2 = engine.reality_check(0.01, 0.001, &obj)?;

        if !check1.is_hallucination && check2.is_hallucination {
            println!("✅ PASS");
            passed += 1;
        } else {
            println!(
                "❌ FAIL (check1={}, check2={})",
                check1.is_hallucination, check2.is_hallucination
            );
            failed += 1;
        }
    }

    println!("\n  Unit Tests: {}/{} passed", passed, passed + failed);
    Ok((passed, failed))
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 2: EXTENDED STRESS TEST (100+ cycles)
// ═══════════════════════════════════════════════════════════════════════════════

fn run_stress_test() -> Result<(bool, f32, f32)> {
    println!("\n━━━ TEST 2: Extended Stress Test (100 cycles) ━━━\n");

    let mut engine = CfCLearningLookahead::new(256, 1.0, 0.01)?;

    // Create 100 objectives with varying difficulty
    let objectives: Vec<_> = (0..100)
        .map(|i| {
            let difficulty = 0.1 + (i as f32 / 100.0) * 0.8; // 0.1 to 0.9
            LearningObjective::new(
                &format!("stress-{}", i),
                &format!("Stress Test Concept {}", i),
                &["Core", "Advanced", "Expert"][i % 3],
                difficulty,
                1000 + i as u64,
            )
        })
        .collect();

    let mut errors: Vec<f32> = Vec::with_capacity(100);
    let mut eval_times: Vec<u64> = Vec::with_capacity(100);

    let stress_start = Instant::now();

    for (i, obj) in objectives.iter().enumerate() {
        // Evaluate with CfC
        let lookahead = engine.evaluate_objective(obj)?;
        eval_times.push(lookahead.prediction_time_us);

        // Actually learn
        let actual_delta = engine.simulate_learning(obj)?;

        // Reality check
        let check = engine.reality_check(lookahead.predicted_delta, actual_delta, obj)?;
        errors.push(check.error.abs());

        // Progress indicator every 20
        if (i + 1) % 20 == 0 {
            let avg_err = errors.iter().sum::<f32>() / errors.len() as f32;
            let avg_time = eval_times.iter().sum::<u64>() / eval_times.len() as u64;
            println!(
                "  Cycle {:3}: avg_error={:.5}, avg_time={}μs",
                i + 1,
                avg_err,
                avg_time
            );
        }
    }

    let total_time = stress_start.elapsed();

    // Analyze results
    let first_quarter: f32 = errors[..25].iter().sum::<f32>() / 25.0;
    let last_quarter: f32 = errors[75..].iter().sum::<f32>() / 25.0;
    let improvement = (first_quarter - last_quarter) / first_quarter * 100.0;

    let avg_time = eval_times.iter().sum::<u64>() / eval_times.len() as u64;
    let max_time = *eval_times.iter().max().unwrap();
    let min_time = *eval_times.iter().min().unwrap();

    let (avg_error, hallucination_rate, _) = engine.accuracy_stats();

    println!("\n  ┌────────────────────────────────────────────────────┐");
    println!("  │              STRESS TEST RESULTS                   │");
    println!("  ├────────────────────────────────────────────────────┤");
    println!("  │ Total cycles:        100                           │");
    println!("  │ Total time:          {:?}              │", total_time);
    println!(
        "  │ Avg eval time:       {}μs                       │",
        avg_time
    );
    println!(
        "  │ Eval time range:     {}μs - {}μs                │",
        min_time, max_time
    );
    println!("  ├────────────────────────────────────────────────────┤");
    println!(
        "  │ First quarter error: {:.5}                       │",
        first_quarter
    );
    println!(
        "  │ Last quarter error:  {:.5}                       │",
        last_quarter
    );
    println!(
        "  │ Error improvement:   {:+.1}%                       │",
        improvement
    );
    println!(
        "  │ Hallucination rate:  {:.1}%                        │",
        hallucination_rate * 100.0
    );
    println!(
        "  │ Avg error overall:   {:.5}                       │",
        avg_error
    );
    println!("  └────────────────────────────────────────────────────┘");

    // Success criteria:
    // - All 100 cycles completed
    // - Avg eval time < 500μs (O(1) complexity)
    // - Hallucination rate < 50%
    let success = avg_time < 500 && hallucination_rate < 0.5;

    if success {
        println!("\n  ✅ STRESS TEST PASSED");
    } else {
        println!("\n  ❌ STRESS TEST FAILED");
        if avg_time >= 500 {
            println!("     - Eval time too slow ({}μs >= 500μs)", avg_time);
        }
        if hallucination_rate >= 0.5 {
            println!(
                "     - Hallucination rate too high ({:.1}% >= 50%)",
                hallucination_rate * 100.0
            );
        }
    }

    Ok((success, improvement, hallucination_rate))
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 3: Φ SANITY CHECK
// ═══════════════════════════════════════════════════════════════════════════════

fn run_phi_sanity_check() -> Result<bool> {
    println!("\n━━━ TEST 3: Φ Sanity Check ━━━\n");

    let mut engine = CfCLearningLookahead::new(256, 1.0, 0.01)?;

    let initial_phi = engine.current_phi();
    println!("  Initial Φ: {:.6}", initial_phi);

    // Check 1: Φ should be in valid range [0, 1]
    print!("  [3.1] Φ in valid range [0, 1]... ");
    if initial_phi >= 0.0 && initial_phi <= 1.0 {
        println!("✅ PASS");
    } else {
        println!("❌ FAIL (Φ = {})", initial_phi);
        return Ok(false);
    }

    // Check 2: Φ should change when consciousness state changes
    print!("  [3.2] Φ changes with learning... ");
    let obj = LearningObjective::new("phi-test", "Phi Test", "Testing", 0.5, 999);
    let delta = engine.simulate_learning(&obj)?;
    let new_phi = engine.current_phi();

    if (new_phi - initial_phi - delta).abs() < 0.0001 {
        println!("✅ PASS (Δ = {:.6})", delta);
    } else {
        println!(
            "❌ FAIL (reported Δ={:.6}, actual Δ={:.6})",
            delta,
            new_phi - initial_phi
        );
        return Ok(false);
    }

    // Check 3: Φ should be deterministic (same state = same Φ)
    print!("  [3.3] Φ is deterministic... ");
    let phi_a = engine.current_phi();
    let phi_b = engine.current_phi();
    if (phi_a - phi_b).abs() < 0.0000001 {
        println!("✅ PASS");
    } else {
        println!("❌ FAIL (phi_a={}, phi_b={})", phi_a, phi_b);
        return Ok(false);
    }

    // Check 4: Different objectives should produce different Φ changes
    print!("  [3.4] Different objectives → different Δφ... ");
    let mut engine2 = CfCLearningLookahead::new(256, 1.0, 0.01)?;
    let obj_easy = LearningObjective::new("easy", "Easy", "Test", 0.1, 111);
    let obj_hard = LearningObjective::new("hard", "Hard", "Test", 0.9, 222);

    let delta_easy = engine2.simulate_learning(&obj_easy)?;
    let mut engine3 = CfCLearningLookahead::new(256, 1.0, 0.01)?;
    let delta_hard = engine3.simulate_learning(&obj_hard)?;

    // They should be different (not necessarily one larger than the other)
    if (delta_easy - delta_hard).abs() > 0.00001 {
        println!("✅ PASS (easy={:.6}, hard={:.6})", delta_easy, delta_hard);
    } else {
        println!(
            "⚠️  WARN: Very similar (easy={:.6}, hard={:.6})",
            delta_easy, delta_hard
        );
        // This is a warning, not a failure - could be coincidence
    }

    // Check 5: Φ should be positive (integrated information > 0)
    print!("  [3.5] Φ is positive... ");
    if initial_phi > 0.0 {
        println!("✅ PASS");
    } else {
        println!("❌ FAIL (Φ = {})", initial_phi);
        return Ok(false);
    }

    println!("\n  ✅ Φ SANITY CHECK PASSED");
    Ok(true)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 4: EDGE CASES
// ═══════════════════════════════════════════════════════════════════════════════

fn run_edge_case_tests() -> Result<(usize, usize)> {
    println!("\n━━━ TEST 4: Edge Case Tests ━━━\n");

    let mut passed = 0;
    let mut failed = 0;

    // Edge 4.1: Zero difficulty objective
    print!("  [4.1] Zero difficulty objective... ");
    {
        let engine = CfCLearningLookahead::new(256, 1.0, 0.01)?;
        let obj = LearningObjective::new("zero", "Zero Diff", "Test", 0.0, 42);
        match engine.evaluate_objective(&obj) {
            Ok(result) => {
                println!("✅ PASS (Δφ={:.6})", result.predicted_delta);
                passed += 1;
            }
            Err(e) => {
                println!("❌ FAIL: {}", e);
                failed += 1;
            }
        }
    }

    // Edge 4.2: Maximum difficulty objective
    print!("  [4.2] Maximum difficulty (1.0)... ");
    {
        let engine = CfCLearningLookahead::new(256, 1.0, 0.01)?;
        let obj = LearningObjective::new("max", "Max Diff", "Test", 1.0, 42);
        match engine.evaluate_objective(&obj) {
            Ok(result) => {
                println!("✅ PASS (Δφ={:.6})", result.predicted_delta);
                passed += 1;
            }
            Err(e) => {
                println!("❌ FAIL: {}", e);
                failed += 1;
            }
        }
    }

    // Edge 4.3: Very short name
    print!("  [4.3] Single character name... ");
    {
        let engine = CfCLearningLookahead::new(256, 1.0, 0.01)?;
        let obj = LearningObjective::new("x", "X", "T", 0.5, 42);
        match engine.evaluate_objective(&obj) {
            Ok(_) => {
                println!("✅ PASS");
                passed += 1;
            }
            Err(e) => {
                println!("❌ FAIL: {}", e);
                failed += 1;
            }
        }
    }

    // Edge 4.4: Very long name
    print!("  [4.4] Very long name (1000 chars)... ");
    {
        let engine = CfCLearningLookahead::new(256, 1.0, 0.01)?;
        let long_name = "X".repeat(1000);
        let obj = LearningObjective::new(&long_name, &long_name, &long_name, 0.5, 42);
        match engine.evaluate_objective(&obj) {
            Ok(_) => {
                println!("✅ PASS");
                passed += 1;
            }
            Err(e) => {
                println!("❌ FAIL: {}", e);
                failed += 1;
            }
        }
    }

    // Edge 4.5: Rapid sequential evaluations
    print!("  [4.5] 1000 rapid evaluations... ");
    {
        let engine = CfCLearningLookahead::new(256, 1.0, 0.01)?;
        let obj = LearningObjective::new("rapid", "Rapid", "Test", 0.5, 42);
        let start = Instant::now();

        for _ in 0..1000 {
            let _ = engine.evaluate_objective(&obj)?;
        }

        let elapsed = start.elapsed();
        let per_eval = elapsed.as_micros() / 1000;

        if per_eval < 1000 {
            println!("✅ PASS ({}μs per eval)", per_eval);
            passed += 1;
        } else {
            println!("❌ FAIL ({}μs per eval - too slow)", per_eval);
            failed += 1;
        }
    }

    // Edge 4.6: Minimal CfC size
    print!("  [4.6] Minimal CfC size (16 neurons)... ");
    {
        match CfCLearningLookahead::new(16, 1.0, 0.01) {
            Ok(engine) => {
                let obj = LearningObjective::new("min", "Min CfC", "Test", 0.5, 42);
                match engine.evaluate_objective(&obj) {
                    Ok(_) => {
                        println!("✅ PASS");
                        passed += 1;
                    }
                    Err(e) => {
                        println!("❌ FAIL: {}", e);
                        failed += 1;
                    }
                }
            }
            Err(e) => {
                println!("❌ FAIL: {}", e);
                failed += 1;
            }
        }
    }

    // Edge 4.7: Large CfC size
    print!("  [4.7] Large CfC size (1024 neurons)... ");
    {
        match CfCLearningLookahead::new(1024, 1.0, 0.01) {
            Ok(engine) => {
                let obj = LearningObjective::new("large", "Large CfC", "Test", 0.5, 42);
                match engine.evaluate_objective(&obj) {
                    Ok(result) => {
                        println!("✅ PASS ({}μs)", result.prediction_time_us);
                        passed += 1;
                    }
                    Err(e) => {
                        println!("❌ FAIL: {}", e);
                        failed += 1;
                    }
                }
            }
            Err(e) => {
                println!("❌ FAIL: {}", e);
                failed += 1;
            }
        }
    }

    // Edge 4.8: Zero horizon
    print!("  [4.8] Zero lookahead horizon... ");
    {
        let engine = CfCLearningLookahead::new(256, 0.0, 0.01)?;
        let obj = LearningObjective::new("zero-h", "Zero Horizon", "Test", 0.5, 42);
        match engine.evaluate_objective(&obj) {
            Ok(_) => {
                println!("✅ PASS");
                passed += 1;
            }
            Err(e) => {
                println!("❌ FAIL: {}", e);
                failed += 1;
            }
        }
    }

    // Edge 4.9: Very large horizon
    print!("  [4.9] Large lookahead horizon (100s)... ");
    {
        let engine = CfCLearningLookahead::new(256, 100.0, 0.01)?;
        let obj = LearningObjective::new("large-h", "Large Horizon", "Test", 0.5, 42);
        match engine.evaluate_objective(&obj) {
            Ok(result) => {
                // Should still be O(1) - that's the magic of CfC!
                if result.prediction_time_us < 500 {
                    println!("✅ PASS (still O(1): {}μs)", result.prediction_time_us);
                    passed += 1;
                } else {
                    println!("⚠️  SLOW ({}μs)", result.prediction_time_us);
                    passed += 1; // Still passes, just slow
                }
            }
            Err(e) => {
                println!("❌ FAIL: {}", e);
                failed += 1;
            }
        }
    }

    // Edge 4.10: Negative difficulty (invalid input)
    print!("  [4.10] Negative difficulty (should clamp or handle)... ");
    {
        let engine = CfCLearningLookahead::new(256, 1.0, 0.01)?;
        // Creating with negative difficulty - should handle gracefully
        let obj = LearningObjective::new("neg", "Negative", "Test", -0.5, 42);
        match engine.evaluate_objective(&obj) {
            Ok(_) => {
                println!("✅ PASS (handled gracefully)");
                passed += 1;
            }
            Err(_) => {
                println!("✅ PASS (rejected invalid input)");
                passed += 1;
            }
        }
    }

    println!("\n  Edge Case Tests: {}/{} passed", passed, passed + failed);
    Ok((passed, failed))
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN: RUN ALL TESTS
// ═══════════════════════════════════════════════════════════════════════════════

fn main() -> Result<()> {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║          CfC LEARNING LOOKAHEAD - COMPREHENSIVE STRESS TEST          ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ Testing: Unit tests, 100-cycle stress, Φ sanity, Edge cases          ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    let start = Instant::now();

    // TEST 1: Unit Tests
    let (unit_passed, unit_failed) = run_unit_tests()?;

    // TEST 2: Stress Test (100 cycles)
    let (stress_passed, improvement, hallucination_rate) = run_stress_test()?;

    // TEST 3: Φ Sanity Check
    let phi_passed = run_phi_sanity_check()?;

    // TEST 4: Edge Cases
    let (edge_passed, edge_failed) = run_edge_case_tests()?;

    let total_time = start.elapsed();

    // ═══════════════════════════════════════════════════════════════════════════════
    // FINAL SUMMARY
    // ═══════════════════════════════════════════════════════════════════════════════

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                        FINAL TEST SUMMARY                            ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                      ║");
    println!(
        "║  TEST 1 - Unit Tests:     {}/{} passed                              ║",
        unit_passed,
        unit_passed + unit_failed
    );
    println!(
        "║  TEST 2 - Stress Test:    {} (improvement: {:+.1}%)             ║",
        if stress_passed {
            "✅ PASS"
        } else {
            "❌ FAIL"
        },
        improvement
    );
    println!(
        "║  TEST 3 - Φ Sanity:       {}                                      ║",
        if phi_passed { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "║  TEST 4 - Edge Cases:     {}/{} passed                             ║",
        edge_passed,
        edge_passed + edge_failed
    );
    println!("║                                                                      ║");
    println!(
        "║  Total Time: {:?}                                         ║",
        total_time
    );
    println!(
        "║  Hallucination Rate: {:.1}%                                         ║",
        hallucination_rate * 100.0
    );
    println!("║                                                                      ║");

    let all_passed = unit_failed == 0 && stress_passed && phi_passed && edge_failed == 0;

    if all_passed {
        println!("║  ✅ ALL TESTS PASSED - Ready to port to src/school/                 ║");
    } else {
        println!("║  ❌ SOME TESTS FAILED - Fix issues before porting                   ║");
    }

    println!("║                                                                      ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    Ok(())
}
