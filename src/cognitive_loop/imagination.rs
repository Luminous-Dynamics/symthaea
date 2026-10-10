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
pub(super) struct ImaginationWorkEstimate {
    pub(super) rollout: f32,
    pub(super) geodesic: f32,
    /// Expected one-time cost of a required manifold dilation.
    pub(super) dilation: f32,
}

#[cfg(feature = "vision-manifold")]
impl ImaginationWorkEstimate {
    fn total(self) -> Option<f32> {
        let total = self.rollout + self.geodesic + self.dilation;
        (self.rollout.is_finite()
            && self.geodesic.is_finite()
            && self.dilation.is_finite()
            && self.rollout >= 0.0
            && self.geodesic >= 0.0
            && self.dilation >= 0.0
            && total.is_finite())
        .then_some(total)
    }
}

#[cfg(feature = "vision-manifold")]
pub(super) fn estimate_imagination_work(
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
    let estimate = ImaginationWorkEstimate {
        rollout,
        geodesic,
        dilation: 0.0,
    };
    estimate.total()?;
    Some(estimate)
}

#[cfg(feature = "vision-manifold")]
pub(super) fn preflight_imagination_work(
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
            // Inspect peer/local shape and reserve the one-time dilation budget before
            // any manifold mutation. HDC bundling requires identical dimensions.
            let peer_dim = peer_msg.consciousness_hv.dim();
            let peer_intent_dim = peer_msg.intent_hv.dim();
            let local_dim = bridge.manifold().hdc_dim();
            let local_state = bridge.manifold().state();
            let dilation_target_dim =
                symthaea_core::hdc::HdcDimensionality::Ultra.dimension();
            if peer_dim == 0
                || peer_intent_dim != peer_dim
                || peer_dim > dilation_target_dim
                || !peer_msg
                    .consciousness_hv
                    .values
                    .iter()
                    .all(|value| value.is_finite())
                || !peer_msg
                    .intent_hv
                    .values
                    .iter()
                    .all(|value| value.is_finite())
                || local_state.dim() != local_dim
                || !local_state.values.iter().all(|value| value.is_finite())
            {
                // Peer state is external input; do not resize or bundle non-finite
                // vectors into local cognitive state.
                return Err(ImagineFutureError::NoGeodesic);
            }

            let needs_dilation = peer_dim > local_dim;
            let target_dim = if needs_dilation {
                dilation_target_dim
            } else {
                local_dim
            };
            if target_dim < local_dim || peer_dim > target_dim {
                // Do not request an unsupported resolution or bundle mismatched vectors.
                return Err(ImagineFutureError::NoGeodesic);
            }

            let mut estimate = estimate_imagination_work(0, steps, 4)
                .ok_or(ImagineFutureError::ThermodynamicOverload(f32::INFINITY))?;
            if needs_dilation {
                estimate.dilation = 0.08;
            }
            preflight_imagination_work(self.thermodynamic_load, estimate)
                .map_err(ImagineFutureError::ThermodynamicOverload)?;

            let manifold = bridge.manifold_mut();

            // Dilation is performed only after budget admission. Charge its one-time
            // estimated cost after the operation, including an unsuccessful attempt.
            if needs_dilation {
                tracing::info!(
                    peer_dim,
                    local_dim,
                    "Collaborative Dreaming: Dilating to match peer resolution"
                );
                manifold.dilate(symthaea_core::hdc::HdcDimensionality::Ultra);
                self.thermodynamic_load += estimate.dilation;
                if manifold.hdc_dim() != dilation_target_dim
                    || peer_dim > manifold.hdc_dim()
                    || manifold.state().dim() != manifold.hdc_dim()
                    || !manifold.state().values.iter().all(|value| value.is_finite())
                {
                    return Err(ImagineFutureError::NoGeodesic);
                }
            }
            // 3. Co-opt the manifold: expand peer encodings to the admitted target
            // resolution before bundling. Never invoke HDC bundle on mismatched dims.
            let peer_consciousness = if peer_dim == manifold.hdc_dim() {
                peer_msg.consciousness_hv.clone()
            } else {
                peer_msg.consciousness_hv.dilate(manifold.hdc_dim())
            };
            let peer_intent = if peer_intent_dim == manifold.hdc_dim() {
                peer_msg.intent_hv.clone()
            } else {
                peer_msg.intent_hv.dilate(manifold.hdc_dim())
            };
            let mut collaborative_start = manifold.state().clone();
            collaborative_start = symthaea_core::core::ContinuousHV::bundle(&[
                &collaborative_start,
                &peer_consciousness,
            ]);
            collaborative_start.normalize();

            // 3. Goal is the peer's intent at the local admitted resolution.
            let goal = peer_intent;

            // Do not charge path work if the manifold's cumulative counter cannot
            // safely admit this search. Dilation (if any) has already been accounted.
            if !manifold.can_compute_geodesic(steps, 4) {
                return Err(ImagineFutureError::NoGeodesic);
            }

            // 4. Run RK4 Geodesic simulation.
            let path = manifold.select_best_geodesic(&collaborative_start, &goal, steps, 4);

            // Charge this request's estimated geodesic work even when no usable path
            // is returned. The computation has already been admitted and invoked.
            self.thermodynamic_load += estimate.geodesic;

            if path.is_empty() {
                return Err(ImagineFutureError::NoGeodesic);
            }

            // Report the same measured local-transition proxy used by the manifold.
            let trajectory_continuity = manifold.measure_path_coherence(&path);
            let trajectory_coherence = trajectory_continuity.unwrap_or(0.0);

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
                // Legacy field name retained; 0.0 means no proxy score was available.
                semantic_coherence: trajectory_coherence,
                trajectory_continuity,
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
            // A full-length response proves the rollout ran, even if its values are
            // invalid. An empty response can be a fail-closed manifold guard.
            if rollout.len() == steps {
                self.thermodynamic_load += estimate.rollout;
            }
            let Some(goal) = rollout.into_iter().last().filter(|goal| {
                goal.dim() == manifold.hdc_dim()
                    && !goal.values.is_empty()
                    && goal.values.iter().all(|value| value.is_finite())
            }) else {
                return Err(ImagineFutureError::NoGeodesic);
            };
            (goal, "model_rollout_endpoint")
        };

        // Refine the selected target over multiple candidate paths. The manifold's
        // own cumulative compute counter may independently fail closed; in that case
        // the rollout remains charged, but an unstarted geodesic search is not.
        if !manifold.can_compute_geodesic(steps, 4) {
            return Err(ImagineFutureError::NoGeodesic);
        }
        let path = manifold.select_best_geodesic(&current, &goal, steps, 4);
        // Charge geodesic work immediately after the search, including empty results.
        self.thermodynamic_load += estimate.geodesic;
        tracing::debug!(goal_source, steps, "Imagination target selected");

        if path.is_empty() {
            return Err(ImagineFutureError::NoGeodesic);
        }

        // Report measured local transition continuity, not a fixed semantic score.
        let trajectory_continuity = manifold.measure_path_coherence(&path);
        let trajectory_coherence = trajectory_continuity.unwrap_or(0.0);

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
            // Legacy field name retained; this is geometric continuity, not semantics.
            semantic_coherence: trajectory_coherence,
            trajectory_continuity,
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
    fn dilation_cost_is_included_in_admission() {
        let mut estimate = estimate_imagination_work(0, 2, 4).unwrap();
        assert!(preflight_imagination_work(0.80, estimate).is_ok());
        estimate.dilation = 0.08;
        assert!((estimate.total().unwrap() - 0.176).abs() < 1e-6);
        assert!(preflight_imagination_work(0.80, estimate).is_err());
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
        assert!(preflight_imagination_work(0.91, small).is_err());
    }

}
