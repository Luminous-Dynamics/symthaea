// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Per-sequence training accounting shared by CPU and GPU runners.

use symthaea_core::hdc::ContinuousHV;

/// Backend that executed a teacher-forced training sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingBackend {
    Cpu,
    Gpu,
}

/// Consolidated outcome for one teacher-forced training sequence.
///
/// The trainer keeps epoch-wide metrics for compatibility, but CPU and GPU
/// paths both report through this struct so post-sequence losses and telemetry
/// can be run without backend-specific branches.
#[derive(Debug, Clone)]
pub struct SequenceResult {
    pub backend: TrainingBackend,
    pub token_loss: f32,
    pub token_count: usize,
    pub final_output_hv: Option<ContinuousHV>,
    pub coherence_sum: f32,
    pub coherence_count: usize,
    pub grad_norm_sum: f32,
    pub clipped_steps: usize,
}

impl SequenceResult {
    pub fn new(backend: TrainingBackend) -> Self {
        Self {
            backend,
            token_loss: 0.0,
            token_count: 0,
            final_output_hv: None,
            coherence_sum: 0.0,
            coherence_count: 0,
            grad_norm_sum: 0.0,
            clipped_steps: 0,
        }
    }

    pub fn record_loss(&mut self, loss: f32) {
        self.token_loss += loss;
        self.token_count += 1;
    }

    pub fn record_coherence(&mut self, coherence: f32) {
        self.coherence_sum += coherence;
        self.coherence_count += 1;
    }

    pub fn record_gradient(&mut self, grad_norm: f32, was_clipped: bool) {
        self.grad_norm_sum += grad_norm;
        if was_clipped {
            self.clipped_steps += 1;
        }
    }

    pub fn set_final_output_hv(&mut self, output_hv: ContinuousHV) {
        self.final_output_hv = Some(output_hv);
    }

    pub fn mean_coherence(&self) -> Option<f32> {
        (self.coherence_count > 0).then(|| self.coherence_sum / self.coherence_count as f32)
    }

    pub fn mean_grad_norm(&self) -> Option<f32> {
        (self.token_count > 0).then(|| self.grad_norm_sum / self.token_count as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_result_accumulates_metrics() {
        let mut result = SequenceResult::new(TrainingBackend::Cpu);

        result.record_loss(2.0);
        result.record_loss(4.0);
        result.record_coherence(0.25);
        result.record_coherence(0.75);
        result.record_gradient(3.0, false);
        result.record_gradient(5.0, true);

        assert_eq!(result.backend, TrainingBackend::Cpu);
        assert_eq!(result.token_loss, 6.0);
        assert_eq!(result.token_count, 2);
        assert_eq!(result.clipped_steps, 1);
        assert_eq!(result.mean_coherence(), Some(0.5));
        assert_eq!(result.mean_grad_norm(), Some(4.0));
    }
}
