// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::floquet_time_crystal::TimeCrystalDetector;
use crate::hofstadter::HofstadterGenerator;
use crate::metrics::ExperimentScorecard;

/// Collects semantic vectors and runs temporal/spectral diagnostics.
pub struct SemanticDiagnosticAdapter {
    buffer: Vec<Vec<f32>>,
    max_len: usize,
}

impl SemanticDiagnosticAdapter {
    pub fn new(max_len: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(max_len),
            max_len,
        }
    }

    pub fn push(&mut self, vector: Vec<f32>) {
        if self.buffer.len() >= self.max_len {
            self.buffer.remove(0);
        }
        self.buffer.push(vector);
    }

    /// Run temporal diagnostic on a specific dimension of the semantic stream.
    pub fn temporal_diagnostic(&self, dim_idx: usize) -> ExperimentScorecard {
        let signal: Vec<f64> = self
            .buffer
            .iter()
            .map(|v| v.get(dim_idx).cloned().unwrap_or(0.0) as f64)
            .collect();

        let detector = TimeCrystalDetector;
        let score = detector.time_crystal_likeness(&signal);

        ExperimentScorecard::new(
            "Semantic Temporal Diagnostic",
            "LLM activation stream exhibits time-crystal-like subharmonic persistence.",
            score,
            &[0.0], // Null baseline
            1,
            42,
            0.5,
            "Exploratory: Semantic drift stability test.",
        )
    }
}
