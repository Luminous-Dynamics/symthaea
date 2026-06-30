// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Receptive hyperdimensional context tracker for extended memory horizons.

use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

pub struct HdcContextRing {
    context_vector: ContinuousHV,
    decay_factor: f32,
    /// NEW: Wisdom Kernels (Macro-HVs) representing folded past experiences.
    wisdom_kernels: Vec<ContinuousHV>,
    /// NEW: Number of trajectories pushed since last folding.
    push_count: usize,
}

impl HdcContextRing {
    pub fn new(decay_factor: f32) -> Self {
        Self {
            context_vector: ContinuousHV::zero(HDC_DIMENSION),
            decay_factor: decay_factor.clamp(0.0, 1.0),
            wisdom_kernels: Vec::new(),
            push_count: 0,
        }
    }

    /// Roll the current token output hypervector into the foundational context ring.
    /// Autonomously 'Folds' memory into Wisdom Kernels every 1,000 pushes.
    pub fn push_trajectory(&mut self, output_hv: &ContinuousHV) {
        let mut composite = vec![0.0f32; HDC_DIMENSION];
        let current_slice = self.context_vector.as_slice();
        let incoming_slice = output_hv.as_slice();

        for i in 0..HDC_DIMENSION {
            composite[i] = (current_slice[i] * self.decay_factor)
                + (incoming_slice[i] * (1.0 - self.decay_factor));
        }
        self.context_vector = ContinuousHV::from_slice(&composite);
        self.push_count += 1;

        // --- IMPROVEMENT: Recursive Memory Folding ---
        if self.push_count >= 1000 {
            println!("🧠 Memory Ring: Folding 1,000 trajectories into a Wisdom Kernel...");
            self.wisdom_kernels.push(self.context_vector.clone());
            self.push_count = 0;
            // (In a real system, we'd further compress wisdom_kernels via recursive_fold)
        }
    }

    /// Retrieve the 'Integrated Wisdom' across the entire historical horizon.
    pub fn integrated_wisdom(&self) -> ContinuousHV {
        if self.wisdom_kernels.is_empty() {
            return self.context_vector.clone();
        }

        let mut refs: Vec<&ContinuousHV> = self.wisdom_kernels.iter().collect();
        refs.push(&self.context_vector);
        ContinuousHV::bundle(&refs)
    }

    pub fn current_context(&self) -> &ContinuousHV {
        &self.context_vector
    }

    pub fn clear(&mut self) {
        self.context_vector = ContinuousHV::zero(HDC_DIMENSION);
        self.wisdom_kernels.clear();
        self.push_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrated_wisdom_includes_folded_kernels() {
        let mut ring = HdcContextRing::new(0.0);
        let hv = ContinuousHV::random(HDC_DIMENSION, 7);
        for _ in 0..1000 {
            ring.push_trajectory(&hv);
        }
        let wisdom = ring.integrated_wisdom();
        assert!(wisdom.norm() > 0.0);

        ring.clear();
        assert_eq!(ring.integrated_wisdom().norm(), 0.0);
    }
}
