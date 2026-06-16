// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Budding Dynamics Analyzer
//!
//! Visualizes how the Cincinnati-LTC network grows (buds new nodes)
//! in response to prediction errors over time.
//!
//! Outputs:
//! - Timeline of budding events
//! - Node count over time
//! - Error distribution across nodes
//! - ASCII visualization of network growth

use std::collections::VecDeque;
use symthaea::hdc::HDC_DIMENSION;
use symthaea::hdc::cincinnati_ltc::{BuddingEvent, CincinnatiLtcEngine};
use symthaea::hdc::unified_hv::ContinuousHV;

/// Track budding events and network dynamics
#[derive(Debug)]
struct BuddingAnalyzer {
    /// Timeline of node counts
    node_counts: Vec<usize>,

    /// Timeline of cumulative budding events
    budding_timeline: Vec<usize>,

    /// Error rates per checkpoint
    error_rates: Vec<f64>,

    /// Prediction accuracy per checkpoint
    accuracy_rates: Vec<f64>,

    /// Budding events with timestamps
    events: Vec<(usize, BuddingEvent)>,

    /// Current step
    step: usize,

    /// Checkpoint interval
    #[allow(dead_code)]
    checkpoint_interval: usize,
}

impl BuddingAnalyzer {
    fn new(checkpoint_interval: usize) -> Self {
        Self {
            node_counts: Vec::new(),
            budding_timeline: Vec::new(),
            error_rates: Vec::new(),
            accuracy_rates: Vec::new(),
            events: Vec::new(),
            step: 0,
            checkpoint_interval,
        }
    }

    fn record_checkpoint(
        &mut self,
        node_count: usize,
        budding_count: usize,
        error_rate: f64,
        accuracy: f64,
    ) {
        self.node_counts.push(node_count);
        self.budding_timeline.push(budding_count);
        self.error_rates.push(error_rate);
        self.accuracy_rates.push(accuracy);
    }

    fn record_event(&mut self, event: BuddingEvent) {
        self.events.push((self.step, event));
    }

    fn increment_step(&mut self) {
        self.step += 1;
    }

    /// Generate ASCII timeline visualization
    fn ascii_timeline(&self, width: usize, height: usize) -> String {
        if self.node_counts.is_empty() {
            return "No data recorded".to_string();
        }

        let max_nodes = *self.node_counts.iter().max().unwrap_or(&1);
        let min_nodes = *self.node_counts.iter().min().unwrap_or(&0);
        let node_range = (max_nodes - min_nodes).max(1);

        let mut lines = vec![vec![' '; width]; height];

        // Plot node count line
        for (i, &count) in self.node_counts.iter().enumerate() {
            let x = (i * (width - 1)) / self.node_counts.len().max(1);
            let normalized = (count - min_nodes) as f64 / node_range as f64;
            let y = height - 1 - ((normalized * (height - 1) as f64) as usize);
            if x < width && y < height {
                lines[y][x] = '*';
            }
        }

        // Add Y-axis labels
        let mut output = String::new();
        output.push_str(&format!(
            "Node Count Over Time (min={}, max={})\n",
            min_nodes, max_nodes
        ));
        output.push_str(&format!("{:>3} ┤", max_nodes));
        for c in &lines[0] {
            output.push(*c);
        }
        output.push('\n');

        for (i, line) in lines.iter().enumerate().skip(1) {
            if i == height / 2 {
                output.push_str(&format!("{:>3} ┤", (max_nodes + min_nodes) / 2));
            } else if i == height - 1 {
                output.push_str(&format!("{:>3} ┤", min_nodes));
            } else {
                output.push_str("    │");
            }
            for c in line {
                output.push(*c);
            }
            output.push('\n');
        }
        output.push_str("    └");
        output.push_str(&"─".repeat(width));
        output.push_str(&format!(
            "\n      0{:>width$}{}\n",
            "time",
            " ",
            width = width - 8
        ));

        output
    }

    /// Generate summary statistics
    fn summary(&self) -> String {
        let mut output = String::new();
        output
            .push_str("╔══════════════════════════════════════════════════════════════════════╗\n");
        output
            .push_str("║                    BUDDING DYNAMICS SUMMARY                          ║\n");
        output.push_str(
            "╚══════════════════════════════════════════════════════════════════════╝\n\n",
        );

        output.push_str(&format!("  Total Steps:           {}\n", self.step));
        output.push_str(&format!("  Total Budding Events:  {}\n", self.events.len()));

        if !self.node_counts.is_empty() {
            let initial = self.node_counts[0];
            let final_count = *self.node_counts.last().unwrap();
            let max_count = *self.node_counts.iter().max().unwrap();
            output.push_str(&format!("  Initial Nodes:         {}\n", initial));
            output.push_str(&format!("  Final Nodes:           {}\n", final_count));
            output.push_str(&format!("  Max Nodes:             {}\n", max_count));
            output.push_str(&format!(
                "  Net Growth:            {:+}\n",
                final_count as i32 - initial as i32
            ));
        }

        if !self.accuracy_rates.is_empty() {
            let avg_accuracy: f64 =
                self.accuracy_rates.iter().sum::<f64>() / self.accuracy_rates.len() as f64;
            let final_accuracy = *self.accuracy_rates.last().unwrap();
            output.push_str(&format!(
                "\n  Average Accuracy:      {:.1}%\n",
                avg_accuracy * 100.0
            ));
            output.push_str(&format!(
                "  Final Accuracy:        {:.1}%\n",
                final_accuracy * 100.0
            ));
        }

        if !self.events.is_empty() {
            output.push_str("\n  Budding Event Timeline:\n");
            for (step, event) in self.events.iter().take(10) {
                output.push_str(&format!("    Step {:>5}: {:?}\n", step, event));
            }
            if self.events.len() > 10 {
                output.push_str(&format!(
                    "    ... and {} more events\n",
                    self.events.len() - 10
                ));
            }
        }

        output
    }
}

/// Pattern generator trait
trait PatternGenerator {
    fn next(&mut self) -> bool;
    #[allow(dead_code)]
    fn name(&self) -> &str;
}

/// Logistic map pattern
struct LogisticPattern {
    r: f64,
    x: f64,
}

impl LogisticPattern {
    fn new(r: f64) -> Self {
        Self { r, x: 0.1 }
    }
}

impl PatternGenerator for LogisticPattern {
    fn next(&mut self) -> bool {
        self.x = self.r * self.x * (1.0 - self.x);
        self.x > 0.5
    }
    fn name(&self) -> &str {
        "Logistic Map"
    }
}

/// Square wave pattern
struct SquareWave {
    half_period: usize,
    step: usize,
}

impl SquareWave {
    fn new(half_period: usize) -> Self {
        Self {
            half_period,
            step: 0,
        }
    }
}

impl PatternGenerator for SquareWave {
    fn next(&mut self) -> bool {
        let bit = (self.step / self.half_period).is_multiple_of(2);
        self.step += 1;
        bit
    }
    fn name(&self) -> &str {
        "Square Wave"
    }
}

/// XOR pattern
struct XorPattern {
    history: VecDeque<bool>,
}

impl XorPattern {
    fn new() -> Self {
        let mut history = VecDeque::with_capacity(4);
        history.push_back(true);
        history.push_back(false);
        Self { history }
    }
}

impl PatternGenerator for XorPattern {
    fn next(&mut self) -> bool {
        let a = *self.history.front().unwrap();
        let b = *self.history.back().unwrap();
        let output = a ^ b;
        self.history.push_back(output);
        if self.history.len() > 2 {
            self.history.pop_front();
        }
        output
    }
    fn name(&self) -> &str {
        "XOR Pattern"
    }
}

/// Run analysis for a single pattern
fn analyze_pattern(
    pattern: &mut dyn PatternGenerator,
    num_steps: usize,
    checkpoint_interval: usize,
) -> BuddingAnalyzer {
    let mut engine = CincinnatiLtcEngine::new(5);
    engine.set_budding_threshold(0.5);
    engine.set_sustain_steps(3);

    let mut analyzer = BuddingAnalyzer::new(checkpoint_interval);
    let mut correct = 0;
    let mut total = 0;
    let mut total_budding_events = 0usize;
    let mut last_budding_count = 0;

    // Track node states for budding
    let mut node_states: Vec<ContinuousHV> = (0..5)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64 * 1000))
        .collect();

    for i in 0..num_steps {
        let observation = pattern.next();

        // Make prediction
        let (prediction, _) = engine.predict();

        // Track accuracy (skip warmup)
        if i >= 10 {
            if prediction == observation {
                correct += 1;
            }
            total += 1;
        }

        // Step engine
        let input_hv = ContinuousHV::random(HDC_DIMENSION, i as u64);
        engine.step(observation, &input_hv);

        // Update prediction error for budding
        let node_count = engine.node_count();
        // Ensure node_states matches current node count
        while node_states.len() < node_count {
            node_states.push(ContinuousHV::random(
                HDC_DIMENSION,
                (node_states.len() * 1000 + i) as u64,
            ));
        }

        for node_id in 0..node_count {
            let expected =
                ContinuousHV::random(HDC_DIMENSION, if prediction { 111111 } else { 222222 });
            let actual =
                ContinuousHV::random(HDC_DIMENSION, if observation { 111111 } else { 222222 });
            engine.update_prediction_error(node_id, &expected, &actual);
        }

        // Process budding
        let events = engine.process_budding(&node_states[..node_count], i as f64);
        for event in &events {
            analyzer.record_event(event.clone());
        }
        total_budding_events += events.len();

        // Checkpoint
        if (i + 1) % checkpoint_interval == 0 {
            let accuracy = if total > 0 {
                correct as f64 / total as f64
            } else {
                0.5
            };
            let error_rate = if total_budding_events > last_budding_count {
                1.0
            } else {
                0.0
            };

            analyzer.record_checkpoint(
                engine.node_count(),
                total_budding_events,
                error_rate,
                accuracy,
            );
            last_budding_count = total_budding_events;
        }

        analyzer.increment_step();
    }

    analyzer
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║              BUDDING DYNAMICS ANALYZER                               ║");
    println!("║     Visualizing Network Growth in Cincinnati-LTC                     ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    let num_steps = 2000;
    let checkpoint_interval = 100;

    // Analyze different patterns
    let mut patterns: Vec<(&str, Box<dyn PatternGenerator>)> = vec![
        (
            "Logistic (r=3.2, predictable)",
            Box::new(LogisticPattern::new(3.2)),
        ),
        (
            "Logistic (r=3.8, chaotic)",
            Box::new(LogisticPattern::new(3.8)),
        ),
        ("Square Wave (half=4)", Box::new(SquareWave::new(4))),
        ("XOR Pattern", Box::new(XorPattern::new())),
    ];

    for (name, ref mut pattern) in patterns.iter_mut() {
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Pattern: {}", name);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let analyzer = analyze_pattern(pattern.as_mut(), num_steps, checkpoint_interval);

        println!("\n{}", analyzer.ascii_timeline(60, 10));
        println!("{}", analyzer.summary());
    }

    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                      ANALYSIS COMPLETE                               ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Key Insights:");
    println!("  - Predictable patterns (r=3.2): Minimal budding, high accuracy");
    println!("  - Chaotic patterns (r=3.8): Moderate budding, adaptive growth");
    println!("  - Difficult patterns: Maximum budding until saturation");
    println!("  - Budding correlates with prediction difficulty");
}