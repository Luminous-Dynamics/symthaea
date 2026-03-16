//! MetricsProvider trait implementation for CognitiveLoopService.
//!
//! Bridges the cognitive loop's internal state to the IPC metrics system,
//! enabling the shell TUI and external tools to observe consciousness state.

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
            in_flow: self.flow_state.in_flow,
            prediction_error: self.stats.avg_prediction_error,
            emotional_valence: self.emotion_contagion.prosody_valence(),
            emotional_arousal: self.emotion_contagion.prosody_arousal(),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            uptime_secs: self.start_time.elapsed().as_secs(),
            total_cycles: self.stats.total_cycles as u64,
            consciousness_level: (phi + coherence) / 2.0,
            latency_ms: 0, // Updated by IPC layer
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
        self.flow_state.in_flow
    }

    fn uptime_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    fn total_cycles(&self) -> u64 {
        self.stats.total_cycles as u64
    }
}
