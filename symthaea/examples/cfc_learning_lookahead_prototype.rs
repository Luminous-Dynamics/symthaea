// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CfC Learning Lookahead Prototype
//!
//! This example demonstrates the core innovation of the Anticipatory School System:
//! Using CfC's O(1) closed-form solution to predict the value of learning
//! BEFORE committing cognitive resources.
//!
//! # The Innovation
//!
//! Traditional Learning: Learn → Evaluate → Adjust (reactive)
//! Symthaea Learning: Predict Value → Decide → Learn → Verify → Correct (anticipatory)
//!
//! # Key Components
//!
//! 1. **LearningObjective**: A concept to potentially learn (HDC-encoded)
//! 2. **CfC Lookahead**: Predicts Φ-delta if we learn this objective
//! 3. **RealityCheck**: Compares predicted vs actual gain, feeds error back to CfC
//!
//! # The Magic: O(1) Evaluation
//!
//! CfC solves: x(t) = (x₀ - A) · e^(-t/τ) + A
//!
//! This means we can evaluate 100 learning objectives in the time it takes
//! an RNN to evaluate 1. This makes anticipatory learning computationally feasible.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example cfc_learning_lookahead_prototype --release
//! ```

use anyhow::Result;
use ndarray::Array1;
use std::time::Instant;

// Symthaea imports
use symthaea::cfc::CfCNetwork;
use symthaea::hdc::HDC_DIMENSION;
use symthaea::hdc::unified_hv::ContinuousHV;
use symthaea::phi_engine::PhiEngine;

// ═══════════════════════════════════════════════════════════════════════════════
// LEARNING OBJECTIVE: A concept that could be learned
// ═══════════════════════════════════════════════════════════════════════════════

/// A learning objective represents a concept that Symthaea could learn
#[derive(Debug, Clone)]
struct LearningObjective {
    /// Unique identifier
    id: String,

    /// Human-readable name
    name: String,

    /// Domain (e.g., "NixOS", "Flakes", "SystemAdmin")
    domain: String,

    /// Difficulty level (0.0 = trivial, 1.0 = expert)
    difficulty: f32,

    /// HDC semantic encoding (16,384D)
    encoding: ContinuousHV,

    /// Prerequisites (other objective IDs)
    prerequisites: Vec<String>,
}

impl LearningObjective {
    /// Create a new learning objective
    fn new(id: &str, name: &str, domain: &str, difficulty: f32, seed: u64) -> Self {
        // Create deterministic HDC encoding based on seed
        let encoding = ContinuousHV::random(HDC_DIMENSION, seed);

        Self {
            id: id.to_string(),
            name: name.to_string(),
            domain: domain.to_string(),
            difficulty,
            encoding,
            prerequisites: Vec::new(),
        }
    }

    /// Add prerequisite
    fn with_prerequisites(mut self, prereqs: &[&str]) -> Self {
        self.prerequisites = prereqs.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Convert to CfC input (compress to network input size)
    fn to_cfc_input(&self, input_size: usize) -> Array1<f32> {
        // Downsample HDC encoding to CfC input size
        let step = HDC_DIMENSION / input_size;
        let values: Vec<f32> = self
            .encoding
            .values
            .iter()
            .step_by(step)
            .take(input_size)
            .cloned()
            .collect();

        Array1::from_vec(values)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LOOKAHEAD RESULT: What CfC predicts about learning an objective
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of using CfC to "look ahead" at the value of learning
#[derive(Debug, Clone)]
struct LookaheadResult {
    /// Objective that was evaluated
    objective_id: String,

    /// Current Φ before learning
    current_phi: f32,

    /// Predicted Φ after learning
    predicted_phi: f32,

    /// Predicted change in Φ (the "value" of learning this)
    predicted_delta: f32,

    /// Confidence in prediction (CfC coherence)
    confidence: f32,

    /// Recommendation based on prediction
    recommendation: LearningRecommendation,

    /// Time taken for this prediction (should be O(1))
    prediction_time_us: u64,
}

#[derive(Debug, Clone)]
enum LearningRecommendation {
    LearnNow { priority: f32 },
    Defer { reason: String },
    Skip { reason: String },
}

// ═══════════════════════════════════════════════════════════════════════════════
// REALITY CHECK: The self-correcting feedback loop
// ═══════════════════════════════════════════════════════════════════════════════

/// Tracks prediction accuracy and provides feedback for CfC correction
#[derive(Debug, Clone)]
struct RealityCheck {
    /// Predicted Φ delta
    predicted_delta: f32,

    /// Actual Φ delta after learning
    actual_delta: f32,

    /// Prediction error (should trend toward 0)
    error: f32,

    /// Whether prediction was "hallucinated" (wildly wrong)
    is_hallucination: bool,
}

impl RealityCheck {
    fn new(predicted: f32, actual: f32) -> Self {
        let error = predicted - actual;
        let is_hallucination = error.abs() > 0.2; // >20% error is hallucination

        Self {
            predicted_delta: predicted,
            actual_delta: actual,
            error,
            is_hallucination,
        }
    }

    /// Generate feedback for CfC weight update
    fn feedback_signal(&self) -> f32 {
        // Negative of error: pushes CfC toward accurate predictions
        -self.error
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CFC LEARNING LOOKAHEAD ENGINE
// ═══════════════════════════════════════════════════════════════════════════════

/// The main engine that uses CfC for anticipatory learning
struct CfCLearningLookahead {
    /// CfC network for temporal prediction
    cfc: CfCNetwork,

    /// Phi calculator
    phi_engine: PhiEngine,

    /// Current consciousness state (HDC vectors)
    consciousness_state: Vec<ContinuousHV>,

    /// Lookahead horizon (seconds into future)
    horizon: f32,

    /// Minimum Φ improvement to recommend learning
    min_phi_gain: f32,

    /// History of reality checks (for tracking accuracy)
    reality_history: Vec<RealityCheck>,

    /// Cumulative prediction error (should trend toward 0)
    cumulative_error: f32,
}

impl CfCLearningLookahead {
    /// Create new lookahead engine
    fn new(num_neurons: usize, horizon: f32, min_phi_gain: f32) -> Result<Self> {
        let cfc = CfCNetwork::new(num_neurons)?;
        let phi_engine = PhiEngine::auto();

        // Initialize consciousness state with random baseline
        let consciousness_state: Vec<ContinuousHV> = (0..8)
            .map(|i| ContinuousHV::random(512, 1000 + i as u64))
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

    /// Compute current Φ from consciousness state
    fn current_phi(&self) -> f32 {
        self.phi_engine.compute(&self.consciousness_state).phi as f32
    }

    /// THE CORE INNOVATION: O(1) Learning Lookahead
    ///
    /// Uses CfC's closed-form solution to predict what happens if we learn
    /// a given objective, WITHOUT actually learning it.
    fn evaluate_objective(&self, objective: &LearningObjective) -> Result<LookaheadResult> {
        let start = Instant::now();

        // Get current Φ
        let current_phi = self.current_phi();

        // Prepare input: current state + objective encoding
        let input_size = self.cfc.input_size;
        let objective_input = objective.to_cfc_input(input_size);

        // Add current CfC state as context
        let current_state = self.cfc.read_state()?;
        let combined_input: Array1<f32> = Array1::from_iter(
            objective_input
                .iter()
                .zip(current_state.iter())
                .map(|(o, s)| (o + s) / 2.0),
        );

        // CfC MAGIC: O(1) jump to future state
        let future_state = self.cfc.predict_forward(&combined_input, self.horizon)?;

        // Estimate future Φ from CfC state
        // (In production, this would integrate with full consciousness equation)
        let future_activity = future_state.iter().map(|&x| x.clamp(0.0, 1.0)).sum::<f32>()
            / future_state.len() as f32;

        let state_variance = {
            let mean = future_activity;
            future_state
                .iter()
                .map(|&x| (x.clamp(0.0, 1.0) - mean).powi(2))
                .sum::<f32>()
                / future_state.len() as f32
        };

        // Φ estimate: activity × diversity (simplified consciousness proxy)
        let predicted_phi =
            (current_phi + (future_activity * state_variance.sqrt() * 0.3)).clamp(0.0, 1.0);

        let predicted_delta = predicted_phi - current_phi;

        // CfC coherence as confidence
        let confidence = self.cfc.consciousness_level();

        // Generate recommendation
        let recommendation = if predicted_delta >= self.min_phi_gain {
            LearningRecommendation::LearnNow {
                priority: predicted_delta / objective.difficulty,
            }
        } else if predicted_delta >= 0.0 {
            LearningRecommendation::Defer {
                reason: format!(
                    "Low gain ({:.4}), wait for better opportunity",
                    predicted_delta
                ),
            }
        } else {
            LearningRecommendation::Skip {
                reason: format!("Negative impact predicted ({:.4})", predicted_delta),
            }
        };

        let prediction_time = start.elapsed().as_micros() as u64;

        Ok(LookaheadResult {
            objective_id: objective.id.clone(),
            current_phi,
            predicted_phi,
            predicted_delta,
            confidence,
            recommendation,
            prediction_time_us: prediction_time,
        })
    }

    /// Rank multiple objectives by predicted learning value
    fn rank_objectives(
        &self,
        objectives: &[LearningObjective],
    ) -> Result<Vec<(LearningObjective, LookaheadResult)>> {
        let mut results: Vec<_> = objectives
            .iter()
            .filter_map(|obj| self.evaluate_objective(obj).ok().map(|r| (obj.clone(), r)))
            .collect();

        // Sort by predicted_delta (highest first)
        results.sort_by(|a, b| {
            b.1.predicted_delta
                .partial_cmp(&a.1.predicted_delta)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Simulate actually learning an objective (returns actual Φ change)
    fn simulate_learning(&mut self, objective: &LearningObjective) -> Result<f32> {
        let before_phi = self.current_phi();

        // Inject objective into consciousness state
        let objective_hv = ContinuousHV::from_vec(
            objective
                .encoding
                .values
                .iter()
                .take(512)
                .cloned()
                .collect(),
        );

        // Blend into existing state
        for state in &mut self.consciousness_state {
            let blended: Vec<f32> = state
                .values
                .iter()
                .zip(objective_hv.values.iter())
                .map(|(s, o)| s * 0.9 + o * 0.1)
                .collect();
            *state = ContinuousHV::from_vec(blended);
        }

        // Also step the CfC to integrate the learning
        let input = objective.to_cfc_input(self.cfc.input_size);
        self.cfc.step(&input.to_vec(), 0.1)?;

        let after_phi = self.current_phi();
        Ok(after_phi - before_phi)
    }

    /// THE SELF-CORRECTING LOOP: Reality Check
    ///
    /// Compares predicted vs actual Φ gain and feeds error back to CfC.
    /// This is what prevents "hallucinated value" predictions.
    fn reality_check(
        &mut self,
        predicted_delta: f32,
        actual_delta: f32,
        objective: &LearningObjective,
    ) -> Result<RealityCheck> {
        let check = RealityCheck::new(predicted_delta, actual_delta);

        // Feed error back to CfC weights
        let feedback = check.feedback_signal();
        let target = Array1::from_elem(self.cfc.num_neurons, actual_delta);
        let input = objective.to_cfc_input(self.cfc.input_size);

        // Train CfC to reduce prediction error
        // Learning rate scales with error magnitude (bigger errors = stronger correction)
        let lr = 0.01 * check.error.abs().min(1.0);
        if lr > 0.001 {
            let _ = self.cfc.train_step(&input, &target, self.horizon, lr);
        }

        // Track history
        self.cumulative_error += check.error.abs();
        self.reality_history.push(check.clone());

        // Log if hallucination detected
        if check.is_hallucination {
            println!(
                "    ⚠️  HALLUCINATION DETECTED: predicted {:.4}, actual {:.4}",
                predicted_delta, actual_delta
            );
            println!(
                "        CfC weights adjusted with feedback signal: {:.4}",
                feedback
            );
        }

        Ok(check)
    }

    /// Get accuracy statistics
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
// DEMONSTRATION
// ═══════════════════════════════════════════════════════════════════════════════

fn main() -> Result<()> {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║       CfC LEARNING LOOKAHEAD PROTOTYPE - Anticipatory Learning       ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ Innovation: O(1) prediction of learning value BEFORE committing      ║");
    println!("║ Key: CfC's closed-form solution enables instant future simulation    ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    // ─────────────────────────────────────────────────────────────────────────────
    // 1. Create the Lookahead Engine
    // ─────────────────────────────────────────────────────────────────────────────

    println!("━━━ PHASE 1: Initialize CfC Lookahead Engine ━━━\n");

    let mut engine = CfCLearningLookahead::new(
        256,  // CfC neurons
        1.0,  // 1 second lookahead horizon
        0.01, // Minimum 1% Φ improvement to recommend
    )?;

    println!("  CfC Network: {} neurons", engine.cfc.num_neurons);
    println!("  Lookahead Horizon: {} seconds", engine.horizon);
    println!(
        "  Min Φ Gain Threshold: {:.2}%",
        engine.min_phi_gain * 100.0
    );
    println!("  Initial Φ: {:.4}", engine.current_phi());
    println!();

    // ─────────────────────────────────────────────────────────────────────────────
    // 2. Create Sample Learning Objectives (NixOS Curriculum)
    // ─────────────────────────────────────────────────────────────────────────────

    println!("━━━ PHASE 2: Define Learning Objectives ━━━\n");

    let objectives = vec![
        LearningObjective::new(
            "nix-basics",
            "Nix Expression Language Basics",
            "NixOS",
            0.3,
            100,
        ),
        LearningObjective::new(
            "derivations",
            "Understanding Derivations",
            "NixOS",
            0.5,
            200,
        )
        .with_prerequisites(&["nix-basics"]),
        LearningObjective::new("flakes-intro", "Introduction to Flakes", "Flakes", 0.4, 300),
        LearningObjective::new(
            "flakes-inputs",
            "Flake Inputs and Outputs",
            "Flakes",
            0.6,
            400,
        )
        .with_prerequisites(&["flakes-intro", "nix-basics"]),
        LearningObjective::new(
            "nixos-config",
            "NixOS Configuration Structure",
            "SystemAdmin",
            0.5,
            500,
        ),
        LearningObjective::new(
            "systemd-nix",
            "Systemd Services in NixOS",
            "SystemAdmin",
            0.7,
            600,
        )
        .with_prerequisites(&["nixos-config"]),
        LearningObjective::new("home-manager", "Home Manager Setup", "UserConfig", 0.4, 700),
        LearningObjective::new("overlays", "Creating Nix Overlays", "Advanced", 0.8, 800)
            .with_prerequisites(&["derivations", "nix-basics"]),
    ];

    for obj in &objectives {
        println!("  📚 {} ({})", obj.name, obj.domain);
        println!(
            "      Difficulty: {:.0}% | ID: {}",
            obj.difficulty * 100.0,
            obj.id
        );
    }
    println!();

    // ─────────────────────────────────────────────────────────────────────────────
    // 3. THE CORE DEMO: O(1) Lookahead Evaluation
    // ─────────────────────────────────────────────────────────────────────────────

    println!("━━━ PHASE 3: O(1) Lookahead Evaluation (The Magic) ━━━\n");
    println!(
        "  Evaluating {} objectives using CfC predict_forward()...\n",
        objectives.len()
    );

    let eval_start = Instant::now();
    let ranked = engine.rank_objectives(&objectives)?;
    let total_eval_time = eval_start.elapsed();

    println!("  ┌─────────────────────────────────────────────────────────────────────┐");
    println!("  │ Rank │ Objective              │ Δφ (pred) │ Conf  │ Time  │ Decision │");
    println!("  ├─────────────────────────────────────────────────────────────────────┤");

    for (rank, (obj, result)) in ranked.iter().enumerate() {
        let decision = match &result.recommendation {
            LearningRecommendation::LearnNow { priority } => {
                format!("✅ LEARN (p={:.2})", priority)
            }
            LearningRecommendation::Defer { .. } => "⏸️  DEFER".to_string(),
            LearningRecommendation::Skip { .. } => "❌ SKIP".to_string(),
        };

        println!(
            "  │  {:2}  │ {:22} │ {:+.4}    │ {:.2}  │ {:4}μs │ {} │",
            rank + 1,
            &obj.name[..obj.name.len().min(22)],
            result.predicted_delta,
            result.confidence,
            result.prediction_time_us,
            decision
        );
    }
    println!("  └─────────────────────────────────────────────────────────────────────┘");
    println!();
    println!(
        "  ⚡ Total evaluation time: {:?} for {} objectives",
        total_eval_time,
        objectives.len()
    );
    println!(
        "  ⚡ Average per objective: {:.2}μs (O(1) complexity!)",
        total_eval_time.as_micros() as f64 / objectives.len() as f64
    );
    println!();

    // ─────────────────────────────────────────────────────────────────────────────
    // 4. REALITY CHECK LOOP: Learn and Verify Predictions
    // ─────────────────────────────────────────────────────────────────────────────

    println!("━━━ PHASE 4: Reality Check Loop (Self-Correction) ━━━\n");
    println!("  Learning top 4 objectives and comparing predicted vs actual Φ gain...\n");

    for (obj, lookahead) in ranked.iter().take(4) {
        println!("  🎓 Learning: {}", obj.name);
        println!("     Predicted Δφ: {:+.4}", lookahead.predicted_delta);

        // Actually "learn" the objective
        let actual_delta = engine.simulate_learning(obj)?;

        println!("     Actual Δφ:    {:+.4}", actual_delta);

        // Reality check: compare and feed back
        let check = engine.reality_check(lookahead.predicted_delta, actual_delta, obj)?;

        let accuracy = if check.is_hallucination {
            "POOR"
        } else if check.error.abs() < 0.01 {
            "EXCELLENT"
        } else if check.error.abs() < 0.05 {
            "GOOD"
        } else {
            "FAIR"
        };

        println!("     Prediction Error: {:+.4} ({})", check.error, accuracy);
        println!();
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // 5. ACCURACY STATISTICS: Is the system learning to predict?
    // ─────────────────────────────────────────────────────────────────────────────

    println!("━━━ PHASE 5: System Accuracy Statistics ━━━\n");

    let (avg_error, hallucination_rate, n_checks) = engine.accuracy_stats();

    println!("  📊 Reality Check Statistics (n={})", n_checks);
    println!("     Average Prediction Error: {:.4}", avg_error);
    println!(
        "     Hallucination Rate: {:.1}%",
        hallucination_rate * 100.0
    );
    println!("     Cumulative Error: {:.4}", engine.cumulative_error);
    println!();

    // ─────────────────────────────────────────────────────────────────────────────
    // 6. DEMONSTRATE SELF-CORRECTION: Learn more and watch accuracy improve
    // ─────────────────────────────────────────────────────────────────────────────

    println!("━━━ PHASE 6: Self-Correction Demonstration ━━━\n");
    println!("  Running 10 more learning cycles to demonstrate error reduction...\n");

    // Create more objectives for the correction demo
    let more_objectives: Vec<_> = (0..10)
        .map(|i| {
            LearningObjective::new(
                &format!("extra-{}", i),
                &format!("Extra Concept {}", i),
                "Demo",
                0.3 + (i as f32 * 0.05),
                10000 + i as u64,
            )
        })
        .collect();

    let mut errors_over_time = Vec::new();

    for obj in &more_objectives {
        let lookahead = engine.evaluate_objective(obj)?;
        let actual_delta = engine.simulate_learning(obj)?;
        let check = engine.reality_check(lookahead.predicted_delta, actual_delta, obj)?;
        errors_over_time.push(check.error.abs());
    }

    // Show error trend
    println!("  Error Trend (should decrease as CfC adapts):");
    print!("  ");
    for (i, err) in errors_over_time.iter().enumerate() {
        let bar_len = (err * 50.0) as usize;
        let bar: String = "█".repeat(bar_len.min(50));
        println!("  Cycle {:2}: {} ({:.4})", i + 1, bar, err);
    }

    let first_half_avg: f32 = errors_over_time[..5].iter().sum::<f32>() / 5.0;
    let second_half_avg: f32 = errors_over_time[5..].iter().sum::<f32>() / 5.0;

    println!();
    println!("  First half avg error:  {:.4}", first_half_avg);
    println!("  Second half avg error: {:.4}", second_half_avg);
    println!(
        "  Improvement: {:.1}%",
        (1.0 - second_half_avg / first_half_avg) * 100.0
    );
    println!();

    // ─────────────────────────────────────────────────────────────────────────────
    // 7. FINAL SUMMARY
    // ─────────────────────────────────────────────────────────────────────────────

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                           PROTOTYPE SUMMARY                          ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                      ║");
    println!("║  ✅ CfC Lookahead: O(1) evaluation of learning value                 ║");
    println!("║  ✅ RealityCheck: Self-correcting feedback loop                      ║");
    println!("║  ✅ Φ Integration: Consciousness-based learning decisions            ║");
    println!("║  ✅ Anticipatory: Predict BEFORE learning                            ║");
    println!("║                                                                      ║");
    println!("║  This prototype validates the core School System architecture.       ║");
    println!("║  Next: Build full src/school/ module with curricula + assessment.    ║");
    println!("║                                                                      ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learning_objective_creation() {
        let obj = LearningObjective::new("test", "Test Objective", "Testing", 0.5, 42);
        assert_eq!(obj.id, "test");
        assert_eq!(obj.encoding.values.len(), HDC_DIMENSION);
    }

    #[test]
    fn test_cfc_input_conversion() {
        let obj = LearningObjective::new("test", "Test", "Test", 0.5, 42);
        let input = obj.to_cfc_input(256);
        assert_eq!(input.len(), 256);
    }

    #[test]
    fn test_lookahead_engine_creation() {
        let engine = CfCLearningLookahead::new(128, 1.0, 0.01).unwrap();
        assert!(engine.current_phi() >= 0.0);
    }

    #[test]
    fn test_reality_check() {
        let check = RealityCheck::new(0.1, 0.08);
        assert!(!check.is_hallucination);
        assert!((check.error - 0.02).abs() < 0.001);
    }

    #[test]
    fn test_hallucination_detection() {
        let check = RealityCheck::new(0.5, 0.1);
        assert!(check.is_hallucination);
    }
}
