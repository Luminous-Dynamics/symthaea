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


/// Cost model used by the three mental-imagination entry points.
/// These are deterministic admission estimates, not measurements of wall-clock work.
#[cfg(feature = "vision-manifold")]
#[derive(Debug, Clone, Copy, PartialEq)]
struct ImaginationWorkEstimate {
    rollout: f32,
    geodesic: f32,
}

#[cfg(feature = "vision-manifold")]
impl ImaginationWorkEstimate {
    fn total(self) -> Option<f32> {
        let total = self.rollout + self.geodesic;
        total.is_finite().then_some(total)
    }
}

#[cfg(feature = "vision-manifold")]
fn estimate_imagination_work(
    rollout_steps: usize,
    geodesic_steps: usize,
    candidate_count: usize,
) -> Option<ImaginationWorkEstimate> {
    let candidate_evaluations = geodesic_steps.checked_mul(candidate_count)?;
    let rollout = (rollout_steps as f64 * 0.008) as f32;
    let geodesic = (candidate_evaluations as f64 * 0.012) as f32;
    if !rollout.is_finite() || !geodesic.is_finite() {
        return None;
    }
    let estimate = ImaginationWorkEstimate { rollout, geodesic };
    estimate.total()?;
    Some(estimate)
}

#[cfg(feature = "vision-manifold")]
fn preflight_imagination_work(
    current_load: f32,
    estimate: ImaginationWorkEstimate,
) -> Result<(), f32> {
    let Some(total) = estimate.total() else {
        return Err(f32::INFINITY);
    };
    let projected_load = current_load + total;
    if !current_load.is_finite()
        || !(0.0..=0.95).contains(&current_load)
        || !projected_load.is_finite()
    {
        return Err(f32::INFINITY);
    }
    if projected_load > 0.95 {
        return Err(projected_load);
    }
    Ok(())
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
            // Admission must happen before dilation or path search. Lifetime manifold
            // telemetry is not a per-request budget and must never gate this request.
            let estimate = estimate_imagination_work(0, steps, 4)
                .ok_or(ImagineFutureError::ThermodynamicOverload(f32::INFINITY))?;
            preflight_imagination_work(self.thermodynamic_load, estimate)
                .map_err(ImagineFutureError::ThermodynamicOverload)?;

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

            // Charge this request's estimated geodesic work even when no usable path
            // is returned. The computation has already happened at this point.
            self.thermodynamic_load += estimate.geodesic;

            if path.is_empty() {
                return Err(ImagineFutureError::NoGeodesic);
            }

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

        // A remembered scene is usable only at the current manifold dimension.
        let remembered_goal = remembered_goal
            .filter(|goal| goal.dim() == manifold.hdc_dim());

        // Preflight both phases before the first rollout/search mutation.
        let rollout_steps = if remembered_goal.is_some() { 0 } else { steps };
        let estimate = estimate_imagination_work(rollout_steps, steps, 4)
            .ok_or(ImagineFutureError::ThermodynamicOverload(f32::INFINITY))?;
        preflight_imagination_work(self.thermodynamic_load, estimate)
            .map_err(ImagineFutureError::ThermodynamicOverload)?;

        let (goal, goal_source) = if let Some(goal) = remembered_goal {
            (goal, "remembered_scene")
        } else {
            let rollout = manifold.dream_ahead(steps, 0.1);
            // Rollout work has been consumed even if its output is unusable.
            self.thermodynamic_load += estimate.rollout;
            let Some(goal) = rollout.into_iter().last().filter(|goal| {
                goal.dim() == manifold.hdc_dim()
                    && !goal.values.is_empty()
                    && goal.values.iter().all(|value| value.is_finite())
            }) else {
                return Err(ImagineFutureError::NoGeodesic);
            };
            (goal, "model_rollout_endpoint")
        };

        // Refine the selected target over multiple candidate paths.
        let path = manifold.select_best_geodesic(&current, &goal, steps, 4);
        // Charge geodesic work immediately after the search, including empty results.
        self.thermodynamic_load += estimate.geodesic;
        tracing::debug!(goal_source, steps, "Imagination target selected");

        if path.is_empty() {
            return Err(ImagineFutureError::NoGeodesic);
        }

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

#[cfg(all(test, feature = "vision-manifold"))]
mod work_budget_tests {
    use super::*;

    #[test]
    fn estimator_is_deterministic_and_separates_phases() {
        let first = estimate_imagination_work(10, 10, 4).unwrap();
        let second = estimate_imagination_work(10, 10, 4).unwrap();
        assert_eq!(first, second);
        assert!((first.rollout - 0.08).abs() < 1e-6);
        assert!((first.geodesic - 0.48).abs() < 1e-6);

        let remembered_goal = estimate_imagination_work(0, 10, 4).unwrap();
        assert_eq!(remembered_goal.rollout, 0.0);
        assert!((first.total().unwrap() - remembered_goal.total().unwrap() - 0.08).abs() < 1e-6);
    }

    #[test]
    fn estimator_fails_closed_on_candidate_multiplication_overflow() {
        assert!(estimate_imagination_work(0, usize::MAX, 4).is_none());
    }

    #[test]
    fn preflight_rejects_invalid_load_and_over_budget_request() {
        let small = estimate_imagination_work(0, 1, 4).unwrap();
        assert!(preflight_imagination_work(f32::NAN, small).is_err());
        assert!(preflight_imagination_work(f32::INFINITY, small).is_err());
        assert!(preflight_imagination_work(-0.01, small).is_err());
        assert!(preflight_imagination_work(0.90, small).is_err());
    }

    #[test]
    fn preflight_does_not_mutate_load() {
        let load = 0.90;
        let estimate = estimate_imagination_work(10, 10, 4).unwrap();
        assert!(preflight_imagination_work(load, estimate).is_err());
        assert_eq!(load, 0.90);
    }
}
