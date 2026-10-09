// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Proactive mental simulation (Imagination) module.

use super::CognitiveLoopService;
#[cfg(feature = "vision-manifold")]
use super::types::MentalMovie;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ImagineFutureError {
    #[error("Vision bridge not available or vision-manifold feature disabled")]
    NoVisionBridge,
    #[error("No geodesic path could be found for the requested horizon")]
    NoGeodesic,
    #[error("Failed to decode mental movie: {0}")]
    DecodeError(String),
    #[error("Thermodynamic load exceeded safe limit ({0:.3})")]
    ThermodynamicOverload(f32),
    #[error("Peer '{0}' not found in hive mind aggregator")]
    PeerNotFound(String),
}

impl CognitiveLoopService {
    /// Perform "Swarm Imagination": Run a mental simulation on behalf of a peer.
    ///
    /// Science: Collective Active Inference. A node with surplus thermodynamic budget
    /// can "help" a stuck peer by calculating its geodesic trajectory using a
    /// bundled "Hive Mind" state.
    #[cfg(feature = "vision-manifold")]
    pub fn collaborative_imagine_future(
        &mut self,
        peer_id: &uuid::Uuid,
        steps: usize,
    ) -> Result<MentalMovie, ImagineFutureError> {
        let bridge = self
            .sensorimotor
            .vision_sensory
            .vision_bridge
            .as_mut()
            .ok_or(ImagineFutureError::NoVisionBridge)?;

        // 1. \"Feel\" the peer: Extract their state from the swarm aggregator
        #[cfg(feature = "swarm")]
        let peer_msg = self
            .swarm_manager
            .hive_mind_aggregator
            .peer_states
            .get(peer_id)
            .ok_or_else(|| ImagineFutureError::PeerNotFound(peer_id.to_string()))?;

        #[cfg(not(feature = "swarm"))]
        return Err(ImagineFutureError::PeerNotFound(peer_id.to_string()));

        #[cfg(feature = "swarm")]
        {
            let manifold = bridge.manifold_mut();

            // 2. Auto-dilate if peer is at higher resolution (Phase 3 optimization)
            let peer_dim = peer_msg.consciousness_hv.values.len();
            if peer_dim > manifold.hdc_dim() {
                tracing::info!(
                    peer_dim,
                    local_dim = manifold.hdc_dim(),
                    "Collaborative Dreaming: Dilating to match peer resolution"
                );
                manifold.dilate(symthaea_core::hdc::HdcDimensionality::Ultra);
            }
            // 3. Co-opt the manifold: Bundle peer consciousness into local state            // This effectively projects the "Self" into the "Other's" perspective.
            let mut collaborative_start = manifold.state().clone();
            collaborative_start = symthaea_core::core::ContinuousHV::bundle(&[
                &collaborative_start,
                &peer_msg.consciousness_hv,
            ]);
            collaborative_start.normalize();

            // 3. Goal is the peer's intent
            let goal = peer_msg.intent_hv.clone();

            // 4. Run RK4 Geodesic simulation
            let path = manifold.select_best_geodesic(&collaborative_start, &goal, steps, 4);

            if path.is_empty() {
                return Err(ImagineFutureError::NoGeodesic);
            }

            // Apply thermodynamic cost (helping others costs energy!)
            let cost = manifold.telemetry().last_geodesic_cost;
            self.thermodynamic_load = (self.thermodynamic_load + cost).min(1.0);

            // 5. Decode and return the "Dream"
            let frames = manifold.decode_geodesic_to_frames_improved(&path);
            if frames.is_empty() {
                return Err(ImagineFutureError::DecodeError(
                    "Collaborative decoder failed".into(),
                ));
            }

            Ok(MentalMovie {
                frames,
                width: self.config.vision_frame_width,
                height: self.config.vision_frame_height,
                channels: bridge.manifold().last_frame_channels(),
                path_length: path.len(),
                semantic_coherence: 0.5, // Collaborative dreams are inherently uncertain
                trajectory: path,
            })
        }
    }
    /// Deliberately simulate `steps` future frames (proactive, ignores surprise).
    ///
    /// Science: Active Inference (Friston 2010). Proactive simulation minimizes
    /// Expected Free Energy by exploring high-value trajectories before execution.
    ///
    /// Returns the mental movie + applies thermodynamic cost.
    #[cfg(feature = "vision-manifold")]
    pub fn imagine_future(&mut self, steps: usize) -> Result<MentalMovie, ImagineFutureError> {
        let bridge = self
            .sensorimotor
            .vision_sensory
            .vision_bridge
            .as_mut()
            .ok_or(ImagineFutureError::NoVisionBridge)?;

        let manifold = bridge.manifold_mut();
        let current = manifold.state().clone();

        // Prefer a recognized scene only when its stored encoding is valid.
        // Otherwise, seed goal-directed refinement from the endpoint of the
        // manifold's own forward rollout instead of an arbitrary random vector.
        let remembered_goal = manifold
            .last_scene_match()
            .and_then(|match_res| manifold.get_scene_encoding(match_res.scene_id))
            .filter(|goal| {
                !goal.values.is_empty() && goal.values.iter().all(|value| value.is_finite())
            });

        // Preflight the deterministic work budget before running either rollout.
        // Keep these rates aligned with dream_ahead (0.008 per step) and
        // select_best_geodesic (0.012 per step-candidate evaluation).
        let candidate_count = 4usize;
        let Some(candidate_evaluations) = steps.checked_mul(candidate_count) else {
            return Err(ImagineFutureError::ThermodynamicOverload(f32::INFINITY));
        };
        let geodesic_cost = candidate_evaluations as f32 * 0.012;
        let rollout_cost = if remembered_goal.is_some() {
            0.0
        } else {
            steps as f32 * 0.008
        };
        let estimated_cost = geodesic_cost + rollout_cost;
        let projected_load = self.thermodynamic_load + estimated_cost;
        if !estimated_cost.is_finite() || !projected_load.is_finite() || projected_load > 0.95 {
            return Err(ImagineFutureError::ThermodynamicOverload(projected_load));
        }

        let (goal, goal_source) = if let Some(goal) = remembered_goal {
            (goal, "remembered_scene")
        } else {
            let rollout = manifold.dream_ahead(steps, 0.1);
            let Some(goal) = rollout.into_iter().last().filter(|goal| {
                goal.dim() == manifold.hdc_dim()
                    && goal.values.iter().all(|value| value.is_finite())
            }) else {
                return Err(ImagineFutureError::NoGeodesic);
            };
            (goal, "model_rollout_endpoint")
        };

        // Refine the selected target over multiple candidate paths.
        let path = manifold.select_best_geodesic(&current, &goal, steps, 4);
        tracing::debug!(goal_source, steps, "Imagination target selected");

        if path.is_empty() {
            return Err(ImagineFutureError::NoGeodesic);
        }

        // Charge only this call's deterministic work estimate. Never charge the
        // manifold's lifetime accumulated telemetry as though it were per-call cost.
        self.thermodynamic_load = (self.thermodynamic_load + estimated_cost).min(1.0);

        // Decode the path into a viewable mental movie
        let frames = manifold.decode_geodesic_to_frames_improved(&path);
        if frames.is_empty() {
            return Err(ImagineFutureError::DecodeError(
                "decoder returned zero frames".into(),
            ));
        }

        let movie = MentalMovie {
            frames,
            width: self.config.vision_frame_width,
            height: self.config.vision_frame_height,
            channels: bridge.manifold().last_frame_channels(),
            path_length: path.len(),
            semantic_coherence: 0.0, // TODO: Compute from score_path_with_fep results if needed
            trajectory: path,
        };

        Ok(movie)
    }

    /// Stub for builds without the vision manifold.
    #[cfg(not(feature = "vision-manifold"))]
    pub fn imagine_future(&mut self, steps: usize) -> Result<(), ImagineFutureError> {
        let _ = steps;
        Err(ImagineFutureError::NoVisionBridge)
    }
}
