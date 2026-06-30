// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Probe stream adapter
//!
//! Stream adapter bridging raw embeddings with target HDC dimensions.

use crate::backends::EmbeddingBackend;
use crate::probe::ProbeMatrix;
use crate::recorder::TrajectoryRecorder;
use std::error::Error;
use symthaea_hdc_ltc::ContinuousHV;

pub struct ProbeStreamAdapter<B: EmbeddingBackend> {
    probe: ProbeMatrix,
    backend: B,
}

impl<B: EmbeddingBackend> ProbeStreamAdapter<B> {
    pub fn new(probe: ProbeMatrix, backend: B) -> Self {
        assert_eq!(
            probe.embedding_dim(),
            backend.embedding_dim(),
            "Probe embedding dimension ({}) must match backend embedding dimension ({})",
            probe.embedding_dim(),
            backend.embedding_dim()
        );
        Self { probe, backend }
    }

    pub fn next_hv(&mut self, t: f64) -> Result<ContinuousHV, Box<dyn Error>> {
        let embedding = self.backend.fetch_embedding(t)?;
        Ok(self.probe.project(&embedding))
    }

    pub fn with_recorder(self, recorder: TrajectoryRecorder) -> RecordingAdapter<B> {
        RecordingAdapter {
            adapter: self,
            recorder,
        }
    }
}

pub struct RecordingAdapter<B: EmbeddingBackend> {
    adapter: ProbeStreamAdapter<B>,
    recorder: TrajectoryRecorder,
}

impl<B: EmbeddingBackend> RecordingAdapter<B> {
    pub fn next_hv(&mut self, t: f64) -> Result<ContinuousHV, Box<dyn Error>> {
        let hv = self.adapter.next_hv(t)?;
        self.recorder.record(t, &hv);
        Ok(hv)
    }

    pub fn finish(self) -> (ProbeStreamAdapter<B>, TrajectoryRecorder) {
        (self.adapter, self.recorder)
    }
}
