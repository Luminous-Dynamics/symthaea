//! WebSocket handler for live consciousness telemetry streaming.
//!
//! Connects a browser client to a running CognitiveLoopService, streaming
//! compact cycle metadata at configurable frequency (default 10Hz).
//!
//! ## Protocol
//!
//! - **Server → Client**: JSON `DemoCycleData` per cycle
//! - **Client → Server**: JSON `{"text": "..."}` to set next cycle input
//! - **Client → Server**: JSON `{"command": "pause"|"resume"|"reset"}` for control

use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use symthaea_core::hdc::consciousness_topology::PersistentFeature;
use tokio::sync::Mutex;

use crate::hdc::moral_topology::{ConvergenceExplanation, PersistenceDiagram};
use symthaea_types::N_HARMONIES;

use super::demo_runner::DemoRunner;

/// Compact telemetry payload streamed per cycle.
///
/// A small subset of CycleMetadata fields chosen for visualization.
#[derive(Debug, Clone, Default, Serialize)]
pub struct DemoCycleData {
    pub cycle: usize,
    pub prediction_error: f32,
    pub consciousness_level: f64,
    pub narrative_self_psi: f64,
    pub valence: f32,
    pub arousal: f32,
    pub mood_temperature: f32,
    pub thermodynamic_load: f32,
    pub moral_score: f64,
    pub coherence: f64,
    pub flow_state: bool,
    pub cycle_time_us: u64,
    pub surprise_triggered: bool,
    pub gwt_broadcast: bool,
    pub dream_insights: usize,
    pub reasoning_confidence: f32,
    pub resonance_frequency: f64,
    pub input_text: String,
    /// 32-dim projection of the thought hypervector for fractal mapping
    pub thought_vector: Vec<f32>,

    // ── Phase 6: Neuromodulator bath telemetry ──
    /// 9-dimensional neuromodulator state vector [DA, NE, 5-HT, ACh, GABA, Oxy, Glut, Aden, ECB]
    #[serde(default)]
    pub neuromod_state_vector: Vec<f32>,
    /// Bath phase space entropy
    #[serde(default)]
    pub bath_entropy: f32,
    /// Allostatic load
    #[serde(default)]
    pub allostatic_load: f32,
    /// E/I ratio (glutamate/GABA)
    #[serde(default)]
    pub ei_ratio: f32,
    /// Sleep pressure (adenosine effective)
    #[serde(default)]
    pub sleep_pressure: f32,
    /// Active injection count
    #[serde(default)]
    pub active_injection_count: u8,
    /// Whether attractor detected in bath phase space
    #[serde(default)]
    pub attractor_detected: bool,

    // ── Mesh Network Telemetry ──
    #[serde(default)]
    pub mesh_health_score: f32,
    #[serde(default)]
    pub mesh_peer_count: u32,
    #[serde(default)]
    pub mesh_bytes_sent: u64,
    #[serde(default)]
    pub mesh_bytes_received: u64,
    #[serde(default)]
    pub mesh_compression_ratio: f64,
    #[serde(default)]
    pub mesh_bandwidth_budget: u64,
    #[serde(default)]
    pub mesh_packets_throttled: u64,

    // ── Post-Phase 6: Phase tracker visualization ──
    /// Bath centroid (9D mean of recent state vectors)
    #[serde(default)]
    pub bath_centroid: Vec<f32>,
    /// Bath per-dimension variance (9D)
    #[serde(default)]
    pub bath_variance: Vec<f32>,
    /// Bath trajectory (last N state vectors)
    #[serde(default)]
    pub bath_trajectory: Vec<Vec<f32>>,
    /// 2D projection: [DA+NE mean, 5-HT+GABA mean]
    #[serde(default)]
    pub bath_projection_2d: Vec<f32>,
    /// Human-readable phase label (stressed/flow/drowsy/alert/relaxed/recovering/balanced)
    #[serde(default)]
    pub bath_phase_label: String,

    // ── Moral Topology: Conscience Radar ──
    /// 8D harmony coordinates: [RC, PSF, IW, IP, UI, SR, EP, SS].
    #[serde(default)]
    pub harmony_coordinates: [f64; N_HARMONIES],
    /// Harmony axis labels (static, included for self-describing payloads).
    #[serde(default)]
    pub harmony_labels: Vec<String>,
    /// Moral free energy (FEP surprise on harmony manifold).
    #[serde(default)]
    pub moral_free_energy: f64,
    /// KL divergence from moral prior.
    #[serde(default)]
    pub moral_kl_divergence: f64,
    /// Entropy of observed harmony distribution.
    #[serde(default)]
    pub moral_entropy: f64,
    /// Moral surprise: -log p(dominant harmony).
    #[serde(default)]
    pub moral_surprise: f64,
    /// Softmax distribution over harmonies (observed).
    #[serde(default)]
    pub moral_scenario_distribution: [f64; N_HARMONIES],
    /// Softmax distribution over harmonies (EMA prior).
    #[serde(default)]
    pub moral_prior_distribution: [f64; N_HARMONIES],
    /// Betti numbers [β₀, β₁, β₂] — components, cycles, voids.
    #[serde(default)]
    pub moral_betti: [usize; 3],
    /// Topology unity score (1.0 = fully connected).
    #[serde(default)]
    pub moral_unity: f64,
    /// Topology completeness (fraction of active harmony axes).
    #[serde(default)]
    pub moral_completeness: f64,
    /// Topology circularity (proportion of cycles in persistent features).
    #[serde(default)]
    pub moral_circularity: f64,
    /// Dominant harmony axis index (0–6).
    #[serde(default)]
    pub moral_dominant_harmony: u8,
    /// Persistence diagram: birth/death pairs of topological features.
    #[serde(default)]
    pub moral_persistent_features: Vec<PersistentFeature>,
    /// Persistence diagram summary (components, cycles, voids with statistics).
    pub moral_persistence_diagram: PersistenceDiagram,
    /// Moral trajectory: last 20 harmony coordinate snapshots.
    #[serde(default)]
    pub moral_trajectory: Vec<[f64; N_HARMONIES]>,
    /// Moral drift (L2 distance between first/second half of trajectory).
    #[serde(default)]
    pub moral_drift: f64,

    // ── Moral Anomaly Detection ──
    /// Composite anomaly score (0.0–1.0).
    #[serde(default)]
    pub moral_anomaly_score: f64,
    /// True when the dominant harmony axis flipped.
    #[serde(default)]
    pub moral_value_inversion: bool,
    /// True when free energy exceeds 2σ of rolling mean.
    #[serde(default)]
    pub moral_free_energy_spike: bool,
    /// True when moral drift exceeds configured threshold.
    #[serde(default)]
    pub moral_drift_alert: bool,
    /// True when β₀ increased since last topology evaluation.
    #[serde(default)]
    pub moral_fragmentation_increase: bool,
    /// True when anomaly response modulations were applied this cycle.
    #[serde(default)]
    pub moral_anomaly_response_applied: bool,
    /// True when trajectory convergence (compartmentalized adversarial pattern) detected.
    #[serde(default)]
    pub moral_trajectory_convergence: bool,
    /// Trajectory convergence severity [0.0, 1.0].
    #[serde(default)]
    pub moral_convergence_severity: f64,
    /// Matched hazard signature template name.
    #[serde(default)]
    pub moral_matched_hazard: Option<String>,
    /// Human-readable convergence explanation (populated when convergence detected or severity > 0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moral_convergence_explanation: Option<ConvergenceExplanation>,

    // ── Vision Manifold Telemetry ──
    /// Whether the vision manifold is active this cycle.
    #[serde(default)]
    pub vision_active: bool,
    /// Vision manifold prediction error (1 - cos_sim).
    #[serde(default)]
    pub vision_prediction_error: f32,
    /// Vision manifold coherence (state-frame alignment).
    #[serde(default)]
    pub vision_coherence: f32,
    /// Vision attention entropy (surprise distribution).
    #[serde(default)]
    pub vision_attention_entropy: f32,
    /// Number of salient (surprising) patches.
    #[serde(default)]
    pub vision_salient_patches: usize,
    /// Vision frame sequence number.
    #[serde(default)]
    pub vision_frame_sequence: u64,
    /// Multi-horizon prediction errors [next_frame, short, medium, scene_scale].
    #[serde(default)]
    pub vision_horizon_errors: Vec<f32>,
    /// Vision encode time in microseconds.
    #[serde(default)]
    pub vision_encode_us: u64,
    /// Vision evolve time in microseconds.
    #[serde(default)]
    pub vision_evolve_us: u64,
    /// Whether vision training was triggered this cycle.
    #[serde(default)]
    pub vision_training_triggered: bool,

    // ── Consciousness Engine Telemetry ──
    #[serde(default)]
    pub consciousness_weights: [f64; 4],
    #[serde(default)]
    pub consciousness_weight_variance: f64,
    #[serde(default)]
    pub weight_convergence_state: String,

    // ── Structural Phi Decomposition ──
    #[serde(default)]
    pub structural_micro_phi: f64,
    #[serde(default)]
    pub structural_meso_phi: f64,
    #[serde(default)]
    pub structural_macro_phi: f64,
    #[serde(default)]
    pub structural_emergence_ratio: f64,

    // ── Substrate ──
    #[serde(default)]
    pub substrate_feasibility: f64,

    // ── Canvas Living Topology ──
    /// SVG string from the canvas living topology pipeline (None when not generated this cycle).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canvas_svg: Option<String>,
    /// Birkhoff aesthetic score of the last generated frame (0.0–1.0).
    #[serde(default)]
    pub canvas_aesthetic_score: f32,
}

impl DemoCycleData {
    /// Replace any NaN or Infinity f32/f64 fields with 0.0 before JSON serialization.
    ///
    /// JSON has no representation for NaN/Infinity — `serde_json` will either
    /// error or produce `null`, corrupting downstream consumers. This method
    /// ensures all floating-point telemetry is wire-safe.
    pub fn sanitize_finite(&mut self) {
        fn sf64(v: &mut f64) {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        fn sf32(v: &mut f32) {
            if !v.is_finite() {
                *v = 0.0;
            }
        }

        // Scalar f32 fields
        sf32(&mut self.prediction_error);
        sf32(&mut self.valence);
        sf32(&mut self.arousal);
        sf32(&mut self.mood_temperature);
        sf32(&mut self.thermodynamic_load);
        sf32(&mut self.reasoning_confidence);
        sf32(&mut self.bath_entropy);
        sf32(&mut self.allostatic_load);
        sf32(&mut self.ei_ratio);
        sf32(&mut self.sleep_pressure);
        sf32(&mut self.mesh_health_score);
        sf32(&mut self.vision_prediction_error);
        sf32(&mut self.vision_coherence);
        sf32(&mut self.vision_attention_entropy);

        // Scalar f64 fields
        sf64(&mut self.consciousness_level);
        sf64(&mut self.narrative_self_psi);
        sf64(&mut self.moral_score);
        sf64(&mut self.coherence);
        sf64(&mut self.resonance_frequency);
        sf64(&mut self.mesh_compression_ratio);
        sf64(&mut self.moral_free_energy);
        sf64(&mut self.moral_kl_divergence);
        sf64(&mut self.moral_entropy);
        sf64(&mut self.moral_surprise);
        sf64(&mut self.moral_unity);
        sf64(&mut self.moral_completeness);
        sf64(&mut self.moral_circularity);
        sf64(&mut self.moral_drift);
        sf64(&mut self.moral_anomaly_score);
        sf64(&mut self.moral_convergence_severity);
        sf64(&mut self.consciousness_weight_variance);
        sf64(&mut self.structural_micro_phi);
        sf64(&mut self.structural_meso_phi);
        sf64(&mut self.structural_macro_phi);
        sf64(&mut self.structural_emergence_ratio);
        sf64(&mut self.substrate_feasibility);

        // Array f64 fields
        for v in &mut self.harmony_coordinates {
            sf64(v);
        }
        for v in &mut self.moral_scenario_distribution {
            sf64(v);
        }
        for v in &mut self.moral_prior_distribution {
            sf64(v);
        }
        for v in &mut self.consciousness_weights {
            sf64(v);
        }

        // Vec<f32> fields
        for v in &mut self.thought_vector {
            sf32(v);
        }
        for v in &mut self.neuromod_state_vector {
            sf32(v);
        }
        for v in &mut self.bath_centroid {
            sf32(v);
        }
        for v in &mut self.bath_variance {
            sf32(v);
        }
        for v in &mut self.bath_projection_2d {
            sf32(v);
        }
        for v in &mut self.vision_horizon_errors {
            sf32(v);
        }
        for row in &mut self.bath_trajectory {
            for v in row {
                sf32(v);
            }
        }
    }
}

/// Client message format.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ClientMessage {
    TextInput {
        text: String,
    },
    Thermodynamics {
        load: f32,
    },
    Command {
        command: String,
    },
    /// Toggle vision manifold: `{"vision": true}` / `{"vision": false}`
    Vision {
        vision: bool,
    },
}

/// WebSocket upgrade handler.
pub async fn ws_handler(ws: WebSocketUpgrade, runner: Arc<Mutex<DemoRunner>>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, runner))
}

async fn handle_socket(mut socket: WebSocket, runner: Arc<Mutex<DemoRunner>>) {
    // Send initial hello
    let hello = serde_json::json!({
        "type": "connected",
        "message": "Symthaea consciousness telemetry stream",
        "version": env!("CARGO_PKG_VERSION"),
    });
    if socket
        .send(Message::Text(hello.to_string().into()))
        .await
        .is_err()
    {
        return;
    }

    let mut interval = tokio::time::interval(std::time::Duration::from_millis(100)); // 10Hz
    let mut paused = false;

    loop {
        tokio::select! {
            _ = interval.tick() => {
                if paused {
                    continue;
                }

                let data = {
                    let mut r = runner.lock().await;
                    r.run_cycle()
                };

                let json = match serde_json::to_string(&data) {
                    Ok(j) => j,
                    Err(_) => continue,
                };

                if socket.send(Message::Text(json.into())).await.is_err() {
                    break; // Client disconnected
                }
            }
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        if let Ok(client_msg) = serde_json::from_str::<ClientMessage>(&text) {
                            match client_msg {
                                ClientMessage::TextInput { text } => {
                                    let mut r = runner.lock().await;
                                    r.set_input(&text);
                                }
                                ClientMessage::Thermodynamics { load } => {
                                    let mut r = runner.lock().await;
                                    r.update_thermodynamics(load);
                                }
                                ClientMessage::Command { command } => {
                                    match command.as_str() {
                                        "pause" => paused = true,
                                        "resume" => paused = false,
                                        "reset" => {
                                            let mut r = runner.lock().await;
                                            r.reset();
                                        }
                                        _ => {}
                                    }
                                }
                                #[cfg(feature = "vision-manifold")]
                                ClientMessage::Vision { vision } => {
                                    let mut r = runner.lock().await;
                                    if vision {
                                        r.enable_vision(64, 64);
                                    } else {
                                        r.disable_vision();
                                    }
                                }
                                #[cfg(not(feature = "vision-manifold"))]
                                ClientMessage::Vision { .. } => {
                                    // Vision not compiled in, ignore
                                }
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_finite_clears_nan_and_infinity() {
        let mut data = DemoCycleData {
            prediction_error: f32::NAN,
            consciousness_level: f64::INFINITY,
            moral_free_energy: f64::NEG_INFINITY,
            valence: f32::NAN,
            mesh_compression_ratio: f64::NAN,
            harmony_coordinates: [1.0, f64::NAN, 0.5, f64::INFINITY, 0.3, 0.2, 0.1, 0.8],
            ..Default::default()
        };
        data.sanitize_finite();

        assert_eq!(data.prediction_error, 0.0);
        assert_eq!(data.consciousness_level, 0.0);
        assert_eq!(data.moral_free_energy, 0.0);
        assert_eq!(data.valence, 0.0);
        assert_eq!(data.mesh_compression_ratio, 0.0);
        assert_eq!(data.harmony_coordinates[1], 0.0); // was NaN
        assert_eq!(data.harmony_coordinates[3], 0.0); // was Infinity
        assert_eq!(data.harmony_coordinates[0], 1.0); // preserved

        // Verify JSON serialization succeeds after sanitize
        serde_json::to_string(&data).expect("sanitized data must serialize");
    }

    #[test]
    fn test_sanitize_finite_preserves_valid_values() {
        let mut data = DemoCycleData {
            prediction_error: 0.42,
            consciousness_level: 0.75,
            moral_score: -0.5,
            coherence: 1.0,
            cycle: 100,
            ..Default::default()
        };
        data.sanitize_finite();

        assert_eq!(data.prediction_error, 0.42);
        assert_eq!(data.consciousness_level, 0.75);
        assert_eq!(data.moral_score, -0.5);
        assert_eq!(data.coherence, 1.0);
        assert_eq!(data.cycle, 100);
    }

    #[test]
    fn test_demo_cycle_data_serializes() {
        let data = DemoCycleData {
            cycle: 42,
            consciousness_level: 0.75,
            consciousness_weights: [0.35, 0.25, 0.25, 0.15],
            consciousness_weight_variance: 0.003,
            weight_convergence_state: "Converging".to_string(),
            structural_micro_phi: 0.1,
            structural_meso_phi: 0.2,
            structural_macro_phi: 0.4,
            structural_emergence_ratio: 2.0,
            substrate_feasibility: 0.71,
            ..Default::default()
        };
        let json = serde_json::to_string(&data).expect("serialization should not panic");
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        // New fields present
        assert_eq!(v["consciousness_weights"][0], 0.35);
        assert_eq!(v["weight_convergence_state"], "Converging");
        assert!(v["structural_micro_phi"].as_f64().unwrap() > 0.0);
        assert!(v["substrate_feasibility"].as_f64().unwrap() > 0.0);
        // Core fields
        assert_eq!(v["cycle"], 42);
        assert_eq!(v["consciousness_level"], 0.75);
    }
}
