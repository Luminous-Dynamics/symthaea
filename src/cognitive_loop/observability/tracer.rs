// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Cognitive Tracer — High-Resolution Thought Observability
//!
//! Records the high-dimensional trajectory of the system's "conscious" state,
//! enabling "thought tracing" and deep alignment auditing.

use std::collections::VecDeque;
use std::time::Instant;

/// A single snapshot of the system's internal state during a cycle.
#[derive(Debug, Clone)]
pub struct ThoughtSnapshot {
    pub cycle: u64,
    pub timestamp: Instant,
    pub input_summary: String,
    pub consciousness_level: f64,
    pub prediction_error: f32,
    pub primary_neuromodulators: [f32; 4], // NE, DA, 5-HT, Ach
    pub focus_hv_checksum: u32,            // CRC32 of the top attention HV
    pub flow_state: f32,
}

pub struct CognitiveTracer {
    history: VecDeque<ThoughtSnapshot>,
    max_history: usize,
}

impl CognitiveTracer {
    pub fn new(max_history: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    pub fn record(&mut self, snapshot: ThoughtSnapshot) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(snapshot);
    }

    pub fn history(&self) -> &VecDeque<ThoughtSnapshot> {
        &self.history
    }
}
