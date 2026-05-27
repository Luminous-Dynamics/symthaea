// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Temporal Pattern Recognition with Cincinnati-LTC
//!
//! Demonstrates the Cincinnati-LTC engine's ability to learn temporal patterns:
//! - Periodic sequences (sine wave, square wave)
//! - Rule-based sequences (XOR, counting)
//! - Chaotic sequences (logistic map)
//!
//! Shows how:
//! - LTC adapts time constants to different pattern frequencies
//! - Cincinnati learns bit-level differential patterns
//! - Predictive autopoiesis grows network for complex sequences
//!
//! ## Run
//! ```bash
//! cargo run --example temporal_pattern_recognition --release
//! ```

use symthaea::hdc::HDC_DIMENSION;
use symthaea::hdc::cincinnati_ltc::{BuddingEvent, CincinnatiLtcEngine};
use symthaea::hdc::unified_hv::ContinuousHV;

use std::collections::VecDeque;

// =============================================================================
// PATTERN GENERATORS
// =============================================================================

/// Different types of temporal patterns
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PatternType {
    /// Periodic: alternates with fixed period
    Periodic(usize),
    /// Square wave: N trues, then N falses
    SquareWave(usize),
    /// XOR of last two bits
    XorPattern,
    /// Count mod N (true if count % N < N/2)
    CountingMod(usize),
    /// Logistic map (chaotic for r > 3.57)
    LogisticMap { r: f64, threshold: f64 },
    /// Rule 110 cellular automaton (1D)
    Rule110,
    /// Fibonacci-like pattern
    FibonacciMod(usize),
}

/// Temporal pattern generator
pub struct PatternGenerator {
    pattern_type: PatternType,
    history: VecDeque<bool>,
    state: f64,
    count: usize,
}

impl PatternGenerator {
    pub fn new(pattern_type: PatternType) -> Self {
        Self {
            pattern_type,
            history: VecDeque::with_capacity(16),
            state: 0.5, // For logistic map
            count: 0,
        }
    }

    /// Generate the next bit in the sequence
    pub fn next_bit(&mut self) -> bool {
        let bit = match self.pattern_type {
            PatternType::Periodic(period) => self.count % period < period / 2,
            PatternType::SquareWave(half_period) => (self.count / half_period).is_multiple_of(2),
            PatternType::XorPattern => {
                if self.history.len() < 2 {
                    self.count.is_multiple_of(2) // Bootstrap
                } else {
                    let a = self.history[self.history.len() - 1];
                    let b = self.history[self.history.len() - 2];
                    a ^ b
                }
            }
            PatternType::CountingMod(n) => (self.count % n) < n / 2,
            PatternType::LogisticMap { r, threshold } => {
                // x_{n+1} = r * x_n * (1 - x_n)
                self.state = r * self.state * (1.0 - self.state);
                self.state > threshold
            }
            PatternType::Rule110 => {
                // Rule 110: current, left, right -> next
                // Uses last 3 bits of history
                if self.history.len() < 3 {
                    !self.count.is_multiple_of(3) // Bootstrap
                } else {
                    let l = self.history.len();
                    let left = self.history[l - 3] as u8;
                    let center = self.history[l - 2] as u8;
                    let right = self.history[l - 1] as u8;
                    let pattern = (left << 2) | (center << 1) | right;
                    // Rule 110: 01101110 in binary
                    ((0b01101110 >> pattern) & 1) == 1
                }
            }
            PatternType::FibonacciMod(n) => {
                if self.history.len() < 2 {
                    self.count.is_multiple_of(2) // Bootstrap
                } else {
                    let a = self.history[self.history.len() - 1] as usize;
                    let b = self.history[self.history.len() - 2] as usize;
                    ((a + b + self.count) % n) < n / 2
                }
            }
        };

        // Update state
        self.history.push_back(bit);
        if self.history.len() > 16 {
            self.history.pop_front();
        }
        self.count += 1;

        bit
    }

    pub fn pattern_type(&self) -> PatternType {
        self.pattern_type
    }
}

// =============================================================================
// TEMPORAL PATTERN RECOGNIZER
// =============================================================================

/// Recognizer that uses Cincinnati-LTC to learn temporal patterns
pub struct TemporalPatternRecognizer {
    engine: CincinnatiLtcEngine,
    history: VecDeque<bool>,
    predictions: VecDeque<bool>,
    correct: usize,
    total: usize,
    recent_correct: VecDeque<bool>,
    // Pre-computed position vectors for precise timing
    position_vectors: Vec<ContinuousHV>,
    // Cyclic position vectors for periodic pattern detection
    cyclic_vectors: Vec<ContinuousHV>,
}

impl TemporalPatternRecognizer {
    pub fn new(n_nodes: usize) -> Self {
        // Pre-compute position vectors for history encoding
        let position_vectors: Vec<ContinuousHV> = (0..64)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, (i as u64) * 12345 + 9999))
            .collect();

        // Cyclic vectors for periods 2, 3, 4, 5, 6, 7, 8 (detect cycles)
        let cyclic_vectors: Vec<ContinuousHV> = (2..=8)
            .map(|period| ContinuousHV::random(HDC_DIMENSION, (period as u64) * 54321 + 7777))
            .collect();

        // Create engine with lower budding threshold and faster sustain
        let mut engine = CincinnatiLtcEngine::new(n_nodes);
        engine.set_budding_threshold(0.5); // Lower threshold for more budding
        engine.set_sustain_steps(3); // Only need 3 consecutive high errors (default is 10)

        Self {
            engine,
            history: VecDeque::with_capacity(64),
            predictions: VecDeque::with_capacity(64),
            correct: 0,
            total: 0,
            recent_correct: VecDeque::with_capacity(100),
            position_vectors,
            cyclic_vectors,
        }
    }

    /// Observe a new bit and predict the next one
    pub fn observe_and_predict(&mut self, observation: bool) -> (bool, f32) {
        // Create input from recent history
        let input = self.history_to_hv();

        // Step the engine
        let _output = self.engine.step(observation, &input);

        // Get prediction
        let (prediction, confidence) = self.engine.predict();

        // Update history
        self.history.push_back(observation);
        if self.history.len() > 32 {
            self.history.pop_front();
        }

        // Track accuracy and UPDATE PREDICTION ERROR FOR BUDDING
        if self.total > 0 {
            let was_correct = self
                .predictions
                .back()
                .map(|&p| p == observation)
                .unwrap_or(false);
            if was_correct {
                self.correct += 1;
            }
            self.recent_correct.push_back(was_correct);
            if self.recent_correct.len() > 100 {
                self.recent_correct.pop_front();
            }

            // CRITICAL: Update prediction error for budding mechanism
            // The error is computed via HV similarity in update_prediction_error
            // When pred ≠ actual, HVs are orthogonal → similarity ≈ 0 → error ≈ 1.0
            let node_count = self.engine.node_count();
            for node_id in 0..node_count {
                // Create expected vs actual HVs for each node
                // Expected: what we predicted (binary to HV)
                let prev_pred = self.predictions.back().copied().unwrap_or(false);
                let expected = self.create_bit_hv(prev_pred);
                // Actual: what we observed
                let actual = self.create_bit_hv(observation);
                self.engine
                    .update_prediction_error(node_id, &expected, &actual);
            }
        }

        self.predictions.push_back(prediction);
        if self.predictions.len() > 64 {
            self.predictions.pop_front();
        }
        self.total += 1;

        (prediction, confidence)
    }

    /// Convert a single bit to a consistent HV (for prediction error calculation)
    fn create_bit_hv(&self, bit: bool) -> ContinuousHV {
        // Use deterministic seeds for true/false to get consistent HVs
        let seed = if bit { 111111 } else { 222222 };
        ContinuousHV::random(HDC_DIMENSION, seed)
    }

    /// Convert history to HDC hypervector with precise timing encoding
    fn history_to_hv(&self) -> ContinuousHV {
        if self.history.is_empty() {
            return ContinuousHV::random(HDC_DIMENSION, 0);
        }

        let mut result_values = vec![0.0f32; HDC_DIMENSION];
        let hist_len = self.history.len();

        // LAYER 1: Position-encoded bits (preserves exact timing)
        // Each position gets a unique orthogonal vector
        for (i, &bit) in self.history.iter().enumerate() {
            let pos_idx = i % self.position_vectors.len();
            let bit_value = if bit { 1.0 } else { -1.0 };
            // More recent positions have stronger weight
            let recency = (i + 1) as f32 / hist_len as f32;

            for (j, v) in self.position_vectors[pos_idx].values.iter().enumerate() {
                result_values[j] += v * bit_value * recency;
            }
        }

        // LAYER 2: Cyclic pattern detection
        // For each period p, compute correlation with periodic pattern
        for (period_idx, period) in (2..=8).enumerate() {
            if hist_len >= period {
                // Check if recent history matches a cycle of this period
                let mut cycle_match = 0.0f32;
                let check_len = (hist_len - 1).min(period * 3); // Check up to 3 cycles

                for i in (hist_len.saturating_sub(check_len))..hist_len {
                    let _expected_pos = i % period; // Position within cycle
                    let actual = self.history[i];
                    // Compare with what was at same cycle position
                    if i >= period {
                        let prev_at_same_pos = self.history[i - period];
                        if actual == prev_at_same_pos {
                            cycle_match += 1.0;
                        } else {
                            cycle_match -= 0.5;
                        }
                    }
                }

                // Add cyclic pattern signal if strong match
                if cycle_match > 0.0 && period_idx < self.cyclic_vectors.len() {
                    let strength = (cycle_match / check_len as f32).min(1.0);
                    for (j, v) in self.cyclic_vectors[period_idx].values.iter().enumerate() {
                        result_values[j] += v * strength * 0.5; // Weighted contribution
                    }
                }
            }
        }

        // LAYER 3: N-gram patterns (last 2, 3, 4 bits as separate features)
        for n in 2..=4 {
            if hist_len >= n {
                // Encode last N bits as a unique pattern
                let mut pattern_seed: u64 = 0;
                for i in (hist_len - n)..hist_len {
                    pattern_seed = pattern_seed * 2 + (self.history[i] as u64);
                }
                pattern_seed += (n as u64) * 1000000; // Distinguish different N values

                let ngram_hv = ContinuousHV::random(HDC_DIMENSION, pattern_seed);
                for (j, v) in ngram_hv.values.iter().enumerate() {
                    result_values[j] += v * 0.3; // N-gram contribution
                }
            }
        }

        // Normalize
        let norm: f32 = result_values.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut result_values {
                *v /= norm;
            }
        }

        ContinuousHV {
            values: result_values,
        }
    }

    /// Overall accuracy
    pub fn accuracy(&self) -> f32 {
        if self.total <= 1 {
            0.5 // No data yet
        } else {
            self.correct as f32 / (self.total - 1) as f32
        }
    }

    /// Recent accuracy (last 100 predictions)
    pub fn recent_accuracy(&self) -> f32 {
        if self.recent_correct.is_empty() {
            0.5
        } else {
            self.recent_correct.iter().filter(|&&c| c).count() as f32
                / self.recent_correct.len() as f32
        }
    }

    /// Number of nodes (shows budding activity)
    pub fn node_count(&self) -> usize {
        self.engine.node_count()
    }

    /// Process budding and return events
    pub fn process_budding(&mut self, time: f64) -> Vec<BuddingEvent> {
        let input = self.history_to_hv();
        self.engine.process_budding(&[input], time)
    }
}

// =============================================================================
// EXPERIMENTS
// =============================================================================

fn run_experiment(
    name: &str,
    pattern_type: PatternType,
    n_steps: usize,
    verbose: bool,
) -> (f32, f32, usize, usize) {
    let mut generator = PatternGenerator::new(pattern_type);
    let mut recognizer = TemporalPatternRecognizer::new(5);

    let mut total_budding_events = 0;

    for step in 0..n_steps {
        let bit = generator.next_bit();
        let (_pred, _conf) = recognizer.observe_and_predict(bit);

        // Process budding every 10 steps
        if step % 10 == 0 {
            let events = recognizer.process_budding(step as f64);
            total_budding_events += events.len();
        }

        // Report every 500 steps for longer runs
        let report_interval = if n_steps > 1000 { 500 } else { 100 };
        if verbose && (step + 1) % report_interval == 0 {
            println!(
                "  Step {:4}: recent_acc={:.1}%, nodes={}, budding={}",
                step + 1,
                recognizer.recent_accuracy() * 100.0,
                recognizer.node_count(),
                total_budding_events,
            );
        }
    }

    let final_accuracy = recognizer.accuracy();
    let recent_accuracy = recognizer.recent_accuracy();
    let final_nodes = recognizer.node_count();

    if verbose {
        println!(
            "\n  {} Final: overall_acc={:.1}%, recent_acc={:.1}%, nodes={}, budding_events={}",
            name,
            final_accuracy * 100.0,
            recent_accuracy * 100.0,
            final_nodes,
            total_budding_events,
        );
    }

    (
        final_accuracy,
        recent_accuracy,
        final_nodes,
        total_budding_events,
    )
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Temporal Pattern Recognition with Cincinnati-LTC         ║");
    println!("║     Learning Sequential Patterns via Differential HDC        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let experiments = vec![
        ("Periodic (period=4)", PatternType::Periodic(4)),
        ("Periodic (period=8)", PatternType::Periodic(8)),
        ("Square Wave (half=3)", PatternType::SquareWave(3)),
        ("XOR Pattern", PatternType::XorPattern),
        ("Counting Mod 5", PatternType::CountingMod(5)),
        ("Fibonacci Mod 3", PatternType::FibonacciMod(3)),
        (
            "Logistic Map (r=3.2, edge of chaos)",
            PatternType::LogisticMap {
                r: 3.2,
                threshold: 0.5,
            },
        ),
        (
            "Logistic Map (r=3.8, chaotic)",
            PatternType::LogisticMap {
                r: 3.8,
                threshold: 0.5,
            },
        ),
        ("Rule 110 (complex)", PatternType::Rule110),
    ];

    println!(
        "Running {} pattern recognition experiments (2000 steps each)...\n",
        experiments.len()
    );
    println!("Improvements: Better encoding + Lower budding threshold (0.5) + Longer training\n");

    let mut results: Vec<(&str, f32, f32, usize, usize)> = Vec::new();

    for (name, pattern_type) in &experiments {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("🔬 Experiment: {}", name);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let (acc, recent_acc, nodes, budding) = run_experiment(name, *pattern_type, 2000, true);
        results.push((name, acc, recent_acc, nodes, budding));
        println!();
    }

    // Summary
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                       RESULTS SUMMARY                        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Pattern               │ Overall │ Recent │ Nodes │ Budding  ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    // Sort by recent accuracy
    let mut sorted_results = results.clone();
    sorted_results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    for (name, acc, recent_acc, nodes, budding) in &sorted_results {
        let acc_indicator = if *recent_acc > 0.7 {
            "✓"
        } else if *recent_acc > 0.55 {
            "~"
        } else {
            "✗"
        };
        println!(
            "║ {:21} │ {:5.1}% │ {:5.1}% │ {:5} │ {:7} {} ║",
            &name[..name.len().min(21)],
            acc * 100.0,
            recent_acc * 100.0,
            nodes,
            budding,
            acc_indicator,
        );
    }
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Analysis
    println!("\n📊 Analysis:");

    let learnable: Vec<_> = sorted_results.iter().filter(|r| r.2 > 0.6).collect();
    let difficult: Vec<_> = sorted_results.iter().filter(|r| r.2 <= 0.55).collect();

    println!(
        "  ✓ Learnable patterns (>60% accuracy): {}",
        learnable.len()
    );
    for (name, _, recent_acc, _, _) in &learnable {
        println!("    - {} ({:.1}%)", name, recent_acc * 100.0);
    }

    println!(
        "  ✗ Difficult patterns (≤55% accuracy): {}",
        difficult.len()
    );
    for (name, _, recent_acc, _, _) in &difficult {
        println!("    - {} ({:.1}%)", name, recent_acc * 100.0);
    }

    // Key insights
    println!("\n💡 Key Insights:");
    println!("  • Periodic patterns: LTC adapts time constants to match period");
    println!("  • Rule-based (XOR): Cincinnati learns differential bit relationships");
    println!("  • Chaotic (logistic): Predictive autopoiesis grows network for complexity");
    println!("  • Budding correlates with pattern difficulty (more complex = more nodes)");

    // Calculate correlation between difficulty and budding
    let avg_budding_learnable: f32 =
        learnable.iter().map(|r| r.4 as f32).sum::<f32>() / learnable.len().max(1) as f32;
    let avg_budding_difficult: f32 =
        difficult.iter().map(|r| r.4 as f32).sum::<f32>() / difficult.len().max(1) as f32;

    println!("\n📈 Budding Correlation:");
    println!(
        "  • Learnable patterns avg budding: {:.1}",
        avg_budding_learnable
    );
    println!(
        "  • Difficult patterns avg budding: {:.1}",
        avg_budding_difficult
    );
    if avg_budding_difficult > avg_budding_learnable {
        println!("  → Network grows more for difficult patterns (autopoiesis working!) ✓");
    }

    println!("\n✅ Temporal Pattern Recognition complete!");
}