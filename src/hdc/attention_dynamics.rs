//! Attention Dynamics for Consciousness
//!
//! # Attention as the Gatekeeper of Consciousness
//!
//! Attention determines what enters conscious awareness. This module implements:
//!
//! 1. **Spotlight Attention**: Focused, high-intensity awareness on one target
//! 2. **Diffuse Attention**: Broad, low-intensity awareness across many targets
//! 3. **Attention Switching**: How focus shifts between targets
//! 4. **Attentional Blink**: Temporary blindness after attention capture
//!
//! # Theoretical Foundation
//!
//! Based on:
//! - **Spotlight Model** (Posner): Attention as moveable beam
//! - **Feature Integration Theory** (Treisman): Binding through attention
//! - **Biased Competition** (Desimone): Targets compete for attention
//! - **Global Neuronal Workspace**: Attention enables global broadcast
//!
//! # Architecture
//!
//! ```text
//!   INPUT SPACE                    ATTENTION FIELD
//!   ┌─────────────────┐           ┌─────────────────┐
//!   │  o o o o o o o  │           │        *        │
//!   │  o o o o o o o  │  ──────>  │     *  @  *     │  @ = spotlight
//!   │  o o o o o o o  │  compete  │        *        │  * = penumbra
//!   │  o o o o o o o  │           │                 │
//!   └─────────────────┘           └─────────────────┘
//!           │                              │
//!           └──────────── Φ ───────────────┘
//!                    (binding strength)
//! ```

use super::real_hv::RealHV;
use super::unified_consciousness_engine::ConsciousnessUpdate;
use std::collections::VecDeque;

/// Attention mode
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AttentionMode {
    /// Narrow, high-intensity focus
    Spotlight,
    /// Broad, low-intensity awareness
    Diffuse,
    /// Balanced attention
    Distributed,
    /// In transition between targets
    Switching,
    /// Post-capture recovery (attentional blink)
    Blink,
}

impl AttentionMode {
    /// Get the characteristic width of attention
    pub fn width(&self) -> f64 {
        match self {
            AttentionMode::Spotlight => 0.2,    // Narrow
            AttentionMode::Diffuse => 0.8,      // Wide
            AttentionMode::Distributed => 0.5,   // Medium
            AttentionMode::Switching => 0.4,     // Transitional
            AttentionMode::Blink => 0.1,         // Minimal
        }
    }

    /// Get the intensity of attention
    pub fn intensity(&self) -> f64 {
        match self {
            AttentionMode::Spotlight => 1.0,
            AttentionMode::Diffuse => 0.3,
            AttentionMode::Distributed => 0.6,
            AttentionMode::Switching => 0.5,
            AttentionMode::Blink => 0.1,
        }
    }
}

/// An attention target
#[derive(Clone, Debug)]
pub struct AttentionTarget {
    /// The target representation
    pub vector: RealHV,
    /// Target salience (how attention-grabbing)
    pub salience: f64,
    /// Current attention weight (0-1)
    pub attention: f64,
    /// How long attended (steps)
    pub dwell_time: usize,
    /// Target ID
    pub id: usize,
}

/// Attention allocation result
#[derive(Clone, Debug)]
pub struct AttentionAllocation {
    /// Which targets received attention
    pub attended: Vec<(usize, f64)>,
    /// Current mode
    pub mode: AttentionMode,
    /// Focus vector (what we're attending to)
    pub focus: RealHV,
    /// Total attention deployed
    pub total_attention: f64,
    /// Attention entropy (how distributed)
    pub entropy: f64,
}

/// Attention dynamics engine
pub struct AttentionDynamics {
    /// Current attention targets
    targets: Vec<AttentionTarget>,
    /// Current focus vector
    focus: RealHV,
    /// Current mode
    mode: AttentionMode,
    /// Step counter
    step: usize,
    /// HDC dimension
    dim: usize,
    /// Attention history
    history: VecDeque<AttentionAllocation>,
    /// Blink counter (steps remaining in blink)
    blink_counter: usize,
    /// Switching threshold
    switch_threshold: f64,
    /// Target counter for IDs
    next_id: usize,
}

impl AttentionDynamics {
    /// Create new attention dynamics engine
    pub fn new(dim: usize) -> Self {
        Self {
            targets: Vec::new(),
            focus: RealHV::random(dim, 42),
            mode: AttentionMode::Diffuse,
            step: 0,
            dim,
            history: VecDeque::new(),
            blink_counter: 0,
            switch_threshold: 0.3,
            next_id: 0,
        }
    }

    /// Add a new attention target
    pub fn add_target(&mut self, vector: RealHV, salience: f64) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        self.targets.push(AttentionTarget {
            vector,
            salience,
            attention: 0.0,
            dwell_time: 0,
            id,
        });

        id
    }

    /// Remove a target
    pub fn remove_target(&mut self, id: usize) {
        self.targets.retain(|t| t.id != id);
    }

    /// Process one attention step
    pub fn step(&mut self, external_input: Option<&RealHV>) -> AttentionAllocation {
        self.step += 1;

        // Handle blink recovery
        if self.blink_counter > 0 {
            self.blink_counter -= 1;
            if self.blink_counter == 0 {
                self.mode = AttentionMode::Diffuse;
            }
        }

        // Update target saliences based on external input
        if let Some(input) = external_input {
            self.update_saliences(input);
        }

        // Compute biased competition
        let competition = self.biased_competition();

        // Update attention weights
        self.update_attention_weights(&competition);

        // Determine mode
        self.update_mode();

        // Update focus vector
        self.update_focus();

        // Check for attention capture
        self.check_attention_capture();

        // Create allocation
        let allocation = self.create_allocation();

        // Store history
        self.history.push_back(allocation.clone());
        if self.history.len() > 50 {
            self.history.pop_front();
        }

        allocation
    }

    /// Update target saliences based on similarity to input
    fn update_saliences(&mut self, input: &RealHV) {
        for target in &mut self.targets {
            let similarity = target.vector.similarity(input).max(0.0) as f64;
            // Salience increases if target matches input
            target.salience = 0.7 * target.salience + 0.3 * similarity;
        }
    }

    /// Biased competition between targets
    fn biased_competition(&self) -> Vec<f64> {
        if self.targets.is_empty() {
            return Vec::new();
        }

        let mut competition = Vec::with_capacity(self.targets.len());

        // Softmax over saliences with current attention as bias
        let bias_strength = 0.5;

        for target in &self.targets {
            // Combine salience with current attention (bias toward attended)
            let score = target.salience + bias_strength * target.attention;
            competition.push(score);
        }

        // Softmax normalization
        let max_score = competition.iter().cloned().fold(f64::MIN, f64::max);
        let exp_scores: Vec<f64> = competition.iter()
            .map(|s| (s - max_score).exp())
            .collect();
        let sum: f64 = exp_scores.iter().sum();

        if sum > 0.0 {
            exp_scores.iter().map(|e| e / sum).collect()
        } else {
            vec![1.0 / self.targets.len() as f64; self.targets.len()]
        }
    }

    /// Update attention weights based on competition
    fn update_attention_weights(&mut self, competition: &[f64]) {
        let learning_rate = 0.3;

        for (i, target) in self.targets.iter_mut().enumerate() {
            if i < competition.len() {
                // Move attention toward competition result
                let new_attention = (1.0 - learning_rate) * target.attention
                    + learning_rate * competition[i];
                target.attention = new_attention;

                // Update dwell time
                if target.attention > 0.3 {
                    target.dwell_time += 1;
                } else {
                    target.dwell_time = 0;
                }
            }
        }
    }

    /// Update attention mode based on distribution
    fn update_mode(&mut self) {
        if self.blink_counter > 0 {
            self.mode = AttentionMode::Blink;
            return;
        }

        if self.targets.is_empty() {
            self.mode = AttentionMode::Diffuse;
            return;
        }

        // Compute attention entropy
        let entropy = self.attention_entropy();

        // Check for dominant target
        let max_attention = self.targets.iter()
            .map(|t| t.attention)
            .fold(0.0_f64, f64::max);

        let was_spotlight = self.mode == AttentionMode::Spotlight;

        if max_attention > 0.7 {
            self.mode = AttentionMode::Spotlight;
        } else if entropy > 0.8 {
            self.mode = AttentionMode::Diffuse;
        } else if was_spotlight && max_attention < 0.5 {
            self.mode = AttentionMode::Switching;
        } else {
            self.mode = AttentionMode::Distributed;
        }
    }

    /// Compute attention entropy
    fn attention_entropy(&self) -> f64 {
        if self.targets.is_empty() {
            return 1.0;
        }

        let sum: f64 = self.targets.iter().map(|t| t.attention).sum();
        if sum <= 0.0 {
            return 1.0;
        }

        let mut entropy = 0.0;
        for target in &self.targets {
            let p = target.attention / sum;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }

        // Normalize to [0, 1]
        let max_entropy = (self.targets.len() as f64).ln();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            1.0
        }
    }

    /// Update focus vector based on attended targets
    fn update_focus(&mut self) {
        if self.targets.is_empty() {
            return;
        }

        // Weighted combination of targets
        let mut focus_components = Vec::new();

        for target in &self.targets {
            if target.attention > 0.1 {
                focus_components.push(target.vector.scale(target.attention as f32));
            }
        }

        if !focus_components.is_empty() {
            self.focus = RealHV::bundle(&focus_components);
        }
    }

    /// Check for sudden attention capture
    fn check_attention_capture(&mut self) {
        // Find highest salience target
        if let Some(max_salience_target) = self.targets.iter()
            .max_by(|a, b| a.salience.partial_cmp(&b.salience).unwrap())
        {
            // If a target has very high salience but low attention, capture
            if max_salience_target.salience > 0.8 && max_salience_target.attention < 0.3 {
                // Attention capture - trigger blink
                self.blink_counter = 5; // 5 step blink

                // Force attention to capturing target
                for target in &mut self.targets {
                    if target.salience > 0.8 {
                        target.attention = 1.0;
                    } else {
                        target.attention *= 0.1;
                    }
                }
            }
        }
    }

    /// Create allocation result
    fn create_allocation(&self) -> AttentionAllocation {
        let attended: Vec<(usize, f64)> = self.targets.iter()
            .filter(|t| t.attention > 0.1)
            .map(|t| (t.id, t.attention))
            .collect();

        let total_attention: f64 = self.targets.iter().map(|t| t.attention).sum();
        let entropy = self.attention_entropy();

        AttentionAllocation {
            attended,
            mode: self.mode,
            focus: self.focus.clone(),
            total_attention,
            entropy,
        }
    }

    /// Get current mode
    pub fn mode(&self) -> AttentionMode {
        self.mode
    }

    /// Get focus vector
    pub fn focus(&self) -> &RealHV {
        &self.focus
    }

    /// Get number of active targets
    pub fn num_targets(&self) -> usize {
        self.targets.len()
    }

    /// Get target by ID
    pub fn get_target(&self, id: usize) -> Option<&AttentionTarget> {
        self.targets.iter().find(|t| t.id == id)
    }

    /// Apply attention to consciousness update
    pub fn modulate_consciousness(&self, update: &ConsciousnessUpdate) -> ModulatedUpdate {
        let attention_weight = self.mode.intensity();
        let attention_width = self.mode.width();

        // Modulated Phi: attention focusing increases integration of attended
        let modulated_phi = update.phi * (0.5 + 0.5 * attention_weight);

        ModulatedUpdate {
            base_phi: update.phi,
            modulated_phi,
            attention_mode: self.mode,
            attention_intensity: attention_weight,
            attention_width,
            num_targets: self.targets.len(),
        }
    }
}

/// Consciousness update modulated by attention
#[derive(Clone, Debug)]
pub struct ModulatedUpdate {
    /// Original Phi
    pub base_phi: f64,
    /// Attention-modulated Phi
    pub modulated_phi: f64,
    /// Current attention mode
    pub attention_mode: AttentionMode,
    /// Attention intensity
    pub attention_intensity: f64,
    /// Attention width
    pub attention_width: f64,
    /// Number of targets
    pub num_targets: usize,
}

impl std::fmt::Display for ModulatedUpdate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Attention[{:?}]: Φ {:.4} -> {:.4} (intensity={:.2}, width={:.2}, targets={})",
               self.attention_mode,
               self.base_phi,
               self.modulated_phi,
               self.attention_intensity,
               self.attention_width,
               self.num_targets)
    }
}

impl std::fmt::Display for AttentionAllocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Allocation[{:?}]: {} targets, total={:.2}, entropy={:.2}",
               self.mode,
               self.attended.len(),
               self.total_attention,
               self.entropy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_basics() {
        let mut attention = AttentionDynamics::new(1024);

        // Add some targets
        let t1 = attention.add_target(RealHV::random(1024, 1), 0.8);
        let t2 = attention.add_target(RealHV::random(1024, 2), 0.3);
        let t3 = attention.add_target(RealHV::random(1024, 3), 0.5);

        println!("\nAttention Dynamics Test:");
        for i in 0..15 {
            let alloc = attention.step(None);
            if i % 3 == 0 {
                println!("Step {}: {}", i, alloc);
            }
        }

        // High salience target should win
        assert!(attention.get_target(t1).unwrap().attention > 0.5);
    }

    #[test]
    fn test_attention_capture() {
        let mut attention = AttentionDynamics::new(1024);

        // Add low salience targets
        attention.add_target(RealHV::random(1024, 1), 0.2);
        attention.add_target(RealHV::random(1024, 2), 0.3);

        // Run a few steps
        for _ in 0..5 {
            attention.step(None);
        }

        // Now add high salience target (should capture)
        let capture_target = RealHV::random(1024, 100);
        attention.add_target(capture_target.clone(), 0.95);

        println!("\nAttention Capture Test:");
        for i in 0..10 {
            let alloc = attention.step(None);
            println!("Step {}: mode={:?}, entropy={:.3}",
                    i, alloc.mode, alloc.entropy);
        }
    }

    #[test]
    fn test_mode_transitions() {
        let mut attention = AttentionDynamics::new(1024);

        // Add many equal targets -> should be diffuse
        for i in 0..5 {
            attention.add_target(RealHV::random(1024, i), 0.5);
        }

        println!("\nMode Transition Test:");

        for _ in 0..5 {
            let alloc = attention.step(None);
            println!("Diffuse phase: {:?}", alloc.mode);
        }

        // Boost one target -> should become spotlight
        attention.targets[0].salience = 0.95;

        for i in 0..10 {
            let alloc = attention.step(None);
            println!("Step {}: {:?}, entropy={:.3}", i, alloc.mode, alloc.entropy);
        }
    }
}
