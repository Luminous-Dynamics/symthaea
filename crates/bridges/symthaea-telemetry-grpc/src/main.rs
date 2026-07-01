// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use std::net::SocketAddr;
use std::time::Duration;
use symthaea_fep::{ActiveInferenceAgentConfig, EnhancedFEPBridge};
use symthaea_telemetry_grpc::{TelemetryBroadcaster, TelemetryFrame};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let addr: SocketAddr = "[::1]:50051".parse()?;
    let broadcaster = std::sync::Arc::new(TelemetryBroadcaster::new());

    // Initialize the Native Rust Covariance Integration Oracle (SpectralMIPFinder)
    let oracle_config = symthaea_phi_oracle::OracleConfig {
        window_size: 200,
        regularization: 1e-4,
        temporal_probes: vec![],
        seed: 42,
    };
    let mut oracle = symthaea_phi_oracle::IntegrationOracle::new_simple(8, oracle_config).unwrap();

    // Initialize the EnhancedFEPBridge — the real active inference loop
    // Inputs: (phi, integration, coherence, attention)
    // Outputs: free_energy, prediction_error, precision, epistemic_value, etc.
    let fep_config = ActiveInferenceAgentConfig::default();
    let mut fep = EnhancedFEPBridge::new(fep_config, 4);

    let broadcaster_clone = broadcaster.clone();
    tokio::spawn(async move {
        let mut count: u64 = 0;
        loop {
            // Run at roughly ~200Hz
            tokio::time::sleep(Duration::from_millis(5)).await;
            count += 1;
            let t = count as f64 * 0.05;

            // Generate a rich synthetic sensory-motor feedback vector
            // using complex sinusoidal interactions to simulate physical/cognitive states
            let obs = vec![
                (t * 1.3).sin() + (t * 0.7).cos(),
                (t * 2.1).sin(),
                (t * 0.9).cos() * (t * 0.3).sin(),
                (t * 1.7).sin() + (t * 1.1).cos(),
                (t * 3.1).cos(),
                (t * 0.5).sin() * (t * 2.3).cos(),
                (t * 1.5).sin(),
                (t * 2.8).cos() + (t * 0.4).sin(),
            ];

            // Ingest observation into the spectral window
            let _ = oracle.observe(&obs);

            // Periodically (or continuously) extract the invariant topology
            if let Some(report) = oracle.measure() {
                // === Ψ_IIT: SpectralMIPFinder normalized integration index ===
                let phi = report.normalized_index;

                // Betti cycle lifespans drive the concentric ring intensities
                let harmonies: Vec<f32> = report
                    .persistent_cycles
                    .iter()
                    .map(|c| c.lifespan as f32)
                    .collect();

                // === Feed SpectralMIPFinder output into the EnhancedFEPBridge ===
                // The 4 FEP inputs map directly from our oracle outputs:
                //   phi        = Psi_IIT spectral integration index
                //   integration = ratio of phi over total mutual information (broadcast ratio)
                //   coherence  = persistence structure (fraction of living cycles)
                //   attention  = temporal coherence from the observation window
                let integration = if report.total_mutual_information > 1e-9 {
                    (phi / report.total_mutual_information).min(1.0)
                } else {
                    0.0
                };

                let coherence = if report.betti_numbers[1] > 0 {
                    report
                        .persistent_cycles
                        .iter()
                        .map(|c| c.lifespan)
                        .sum::<f64>()
                        / report.betti_numbers[1] as f64
                } else {
                    0.0
                };

                // Spectral order entropy as the attention signal
                let n = report.spectral_order.len() as f64;
                let attention = if n > 0.0 {
                    let mean = report.spectral_order.iter().sum::<usize>() as f64 / n;
                    let variance = report
                        .spectral_order
                        .iter()
                        .map(|&x| (x as f64 - mean).powi(2))
                        .sum::<f64>()
                        / n;
                    (variance.sqrt() / (n + 1.0)).min(1.0)
                } else {
                    0.0
                };

                // Run the full FEP perception-action-learning cycle
                let fep_result = fep.core.process(phi, integration, coherence, attention);

                // === Map FEP outputs to 7-Theory gRPC fields ===

                // A_GWT: Global broadcast = integration (how much phi escapes local partitions)
                let gwt_broadcast = integration.clamp(0.0, 1.0);

                // M_HOT: Metacognitive confidence = FEP model_confidence (TD-learned)
                let hot_metacognitive = fep_result.model_confidence.clamp(0.0, 1.0);

                // S_AST: Attention schema = attention signal (spectral entropy of ordering)
                let ast_temporal = attention.clamp(0.0, 1.0);

                // K_W: Knowledge coherence = belief_confidence (how tight the posterior is)
                let knowledge_coherence = fep_result.belief_confidence.clamp(0.0, 1.0);

                // E_FEP: Embodiment = 1 - normalized free energy (low FE = high embodied grounding)
                // free_energy is typically in [0, ~5], clamp to [0,1] range
                let embodiment_level =
                    (1.0 - (fep_result.free_energy / 5.0).clamp(0.0, 1.0)).clamp(0.0, 1.0);

                // Φ_sync: Neural synchrony = epistemic value from action selection
                let self_awareness = fep_result.epistemic_value.clamp(0.0, 1.0);

                // Φ_master: The unified 7-theory product invariant
                let topological_unity = (phi
                    * gwt_broadcast
                    * (hot_metacognitive + 0.1)
                    * (ast_temporal + 0.1)
                    * embodiment_level
                    * (knowledge_coherence + 0.1)
                    * (self_awareness + 0.1))
                    .min(1.0);

                // Arousal ← prediction error (high surprise = high arousal)
                let arousal = fep_result.prediction_error.clamp(0.0, 1.0) as f32;
                // Uncertainty ← precision-weighted error (how confident the error estimate is)
                let uncertainty =
                    (fep_result.precision_weighted_error / 2.0).clamp(0.0, 1.0) as f32;
                // Surprise spikes when is_surprised flag fires from the FEP agent
                let surprise = if fep_result.is_surprised {
                    1.0
                } else {
                    0.0_f64
                };

                let frame = TelemetryFrame {
                    phi,
                    harmonies,
                    neuromodulators: vec![
                        fep_result.learning_rate_modulation as f32,
                        fep_result.td_error.clamp(-1.0, 1.0).abs() as f32,
                        fep_result.pragmatic_value.clamp(0.0, 1.0) as f32,
                        if fep_result.exploration_mode {
                            1.0
                        } else {
                            0.0_f32
                        },
                    ],
                    arousal,
                    uncertainty,
                    surprise,
                    timestamp: chrono::Utc::now().to_rfc3339(),
                    // === 7-Theory fields — all grounded in real FEP math ===
                    gwt_broadcast,
                    hot_metacognitive,
                    ast_temporal,
                    knowledge_coherence,
                    embodiment_level,
                    self_awareness,
                    topological_unity,
                    mental_movie: Some(symthaea_telemetry_grpc::MentalMovieFrame {
                        pixel_data: (0..128 * 128)
                            .map(|i| (i as u8).wrapping_add(count as u8))
                            .collect(),
                        width: 128,
                        height: 128,
                        channels: 1,
                        semantic_coherence: fep_result.model_confidence as f32,
                        sequence_index: count,
                    }),
                };
                broadcaster_clone.broadcast(frame);
            }
        }
    });

    broadcaster.run(addr).await?;
    Ok(())
}
