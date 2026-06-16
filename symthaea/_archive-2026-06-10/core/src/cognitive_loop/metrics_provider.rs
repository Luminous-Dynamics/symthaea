// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! MetricsProvider trait implementation for CognitiveLoopService.

use super::CognitiveLoopService;
use crate::shell::ipc_client::MetricsSnapshot;
use crate::shell::ipc_server::MetricsProvider;

impl MetricsProvider for CognitiveLoopService {
    fn get_metrics(&self) -> MetricsSnapshot {
        let phi = self.unification_engine.psi;
        let coherence = self
            .language_comm
            .voice_coherence
            .bridge
            .smoothed_coherence() as f64;
        MetricsSnapshot {
            phi,
            coherence,
            is_conscious: phi > 0.3,
            cognitive_depth: format!("{:?}", self.cognitive_depth),
            strategy: format!("{:?}", self.fep.closed_learning_loop.current_strategy),
            in_flow: self.behavior.flow_state.in_flow,
            prediction_error: self.stats.avg_prediction_error,
            emotional_valence: self.unification_engine.emotional.state().valence as f32,
            emotional_arousal: self.unification_engine.emotional.state().arousal as f32,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            uptime_secs: self.start_time.elapsed().as_secs(),
            total_cycles: self.stats.total_cycles as u64,
            consciousness_level: (phi + coherence) / 2.0,
            latency_ms: 0,
            #[cfg(feature = "vision-manifold")]
            mental_movie: self.last_mental_movie.as_ref().map(|m| {
                crate::shell::ipc_client::MentalMovie {
                    frames: m.frames.clone(),
                    width: m.width,
                    height: m.height,
                    channels: m.channels,
                    path_length: m.path_length,
                    semantic_coherence: m.semantic_coherence,
                }
            }),
            last_observed_frame: None,
            connected_peers: Vec::new(),
        }
    }

    fn phi(&self) -> f64 {
        self.unification_engine.psi
    }

    fn coherence(&self) -> f64 {
        self.language_comm
            .voice_coherence
            .bridge
            .smoothed_coherence() as f64
    }

    fn is_conscious(&self) -> bool {
        self.unification_engine.psi > 0.3
    }

    fn cognitive_depth(&self) -> String {
        format!("{:?}", self.cognitive_depth)
    }

    fn current_strategy(&self) -> String {
        format!("{:?}", self.fep.closed_learning_loop.current_strategy)
    }

    fn in_flow(&self) -> bool {
        self.behavior.flow_state.in_flow
    }

    fn uptime_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    fn total_cycles(&self) -> u64 {
        self.stats.total_cycles as u64
    }
}
