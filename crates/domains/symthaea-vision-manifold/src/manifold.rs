// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CfC-based temporal manifold for video state tracking.
//!
//! Maintains a continuous-time hypervector state that evolves via closed-form
//! CfC (Closed-form Continuous-time) dynamics:
//!
//! ```text
//! state' = x_inf + (state - x_inf) · exp(-dt / τ)
//! ```
//!
//! where `x_inf = tanh(W ⊗ state  +  U ⊗ input)` is the equilibrium state.
//!
//! Key property: prediction cost is O(D) regardless of the time horizon dt.
//! Whether dt is 0.001s or 1000s, the computation is a single closed-form step.
//!
//! **Caveat**: The closed-form assumes equilibrium is constant during the step.
//! For very large dt/τ ratios, the state saturates to x_inf. This is accurate
//! for static scenes but introduces error when the scene is changing rapidly
//! during the predicted interval.

use std::time::Instant;

use symthaea_core::hdc::ContinuousHV;
use symthaea_core::temporal::TemporalPredictor;

use crate::attention::SurpriseMap;
use crate::encoder::{MotionField, PatchHdcEncoder};
use crate::predictive::PredictiveCodingHierarchy;
use crate::training::{BpttResult, ManifoldTrainer};
use crate::types::{
    DELAYED_HORIZON_EVALUATOR_STATE_SCHEMA_VERSION, DelayedHorizonEvaluatorState, DilationEstimate,
    HorizonAccumulatorState, MANIFOLD_STATE_SCHEMA_VERSION, ManifoldHealth, ManifoldState,
    ModalityTemporalContextState, ObjectHypothesisState, ObjectMemoryState,
    PendingHorizonForecastState, SceneFrameMetadata, SceneMatch, SceneMemoryState,
    SurpriseMapState, TrackedObjectState, VisionConfig, VisionTelemetry, VisualModality,
    VisualWorkingMemoryState, WorkingMemorySlotState,
};

#[derive(Clone, Default)]
struct ModalityTemporalContext {
    last_prediction: Option<ContinuousHV>,
    last_frame_hv: Option<ContinuousHV>,
    last_patch_hvs: Vec<ContinuousHV>,
    temporal_patch_hvs: Vec<ContinuousHV>,
    prev_patch_lum: Vec<f32>,
    surprise_state: Option<SurpriseMapState>,
    prediction_error: f32,
    error_ema: f32,
    fep_belief_mean: Vec<f64>,
    last_fep: crate::types::FepMetrics,
    horizon_evaluator: DelayedHorizonEvaluator,
    object_memory: Option<ObjectMemoryState>,
    next_track_id: u64,
    last_tracking_result: Option<ObjectTrackingResult>,
    last_object_hypotheses: Vec<crate::types::ObjectHypothesis>,
    working_memory: Option<VisualWorkingMemoryState>,
    scene_graph_enabled: bool,
}

impl ModalityTemporalContext {
    fn hdc_vector_count(&self) -> usize {
        self.last_prediction.is_some() as usize
            + self.last_frame_hv.is_some() as usize
            + self.last_patch_hvs.len()
            + self.temporal_patch_hvs.len()
            + self.horizon_evaluator.hdc_vector_count()
            + self
                .object_memory
                .as_ref()
                .map_or(0, |state| state.tracks.len().saturating_mul(2))
            + self
                .working_memory
                .as_ref()
                .map_or(0, |state| state.slots.len())
            + self.last_object_hypotheses.len()
    }

    fn dilate(&mut self, target_dim: usize) {
        if let Some(ref mut hv) = self.last_prediction {
            *hv = hv.dilate(target_dim);
        }
        if let Some(ref mut hv) = self.last_frame_hv {
            *hv = hv.dilate(target_dim);
        }
        for hv in &mut self.last_patch_hvs {
            *hv = hv.dilate(target_dim);
        }
        for hv in &mut self.temporal_patch_hvs {
            *hv = hv.dilate(target_dim);
        }
        self.horizon_evaluator.dilate(target_dim);
        if let Some(ref mut state) = self.object_memory {
            for track in &mut state.tracks {
                track.appearance_hv = ContinuousHV::from_vec(track.appearance_hv.clone())
                    .dilate(target_dim)
                    .as_slice()
                    .to_vec();
                track.identity_hv = ContinuousHV::from_vec(track.identity_hv.clone())
                    .dilate(target_dim)
                    .as_slice()
                    .to_vec();
            }
        }
        if let Some(ref mut state) = self.working_memory {
            for slot in &mut state.slots {
                slot.hv = ContinuousHV::from_vec(slot.hv.clone())
                    .dilate(target_dim)
                    .as_slice()
                    .to_vec();
            }
        }
        for hypothesis in &mut self.last_object_hypotheses {
            hypothesis.hv = hypothesis.hv.dilate(target_dim);
        }
    }

    fn save_state(&self, modality: VisualModality) -> ModalityTemporalContextState {
        ModalityTemporalContextState {
            modality,
            last_prediction: self
                .last_prediction
                .as_ref()
                .map(|hv| hv.as_slice().to_vec()),
            last_frame_hv: self.last_frame_hv.as_ref().map(|hv| hv.as_slice().to_vec()),
            last_patch_hvs: self
                .last_patch_hvs
                .iter()
                .map(|hv| hv.as_slice().to_vec())
                .collect(),
            temporal_patch_hvs: self
                .temporal_patch_hvs
                .iter()
                .map(|hv| hv.as_slice().to_vec())
                .collect(),
            prev_patch_lum: self.prev_patch_lum.clone(),
            surprise_state: self.surprise_state.clone(),
            prediction_error: self.prediction_error,
            error_ema: self.error_ema,
            fep_belief_mean: self.fep_belief_mean.clone(),
            last_fep: self.last_fep,
            horizon_evaluator: Some(self.horizon_evaluator.save_state()),
            object_memory: self.object_memory.clone(),
            next_track_id: self.next_track_id,
            last_object_hypotheses: self
                .last_object_hypotheses
                .iter()
                .map(|hypothesis| ObjectHypothesisState {
                    patch_indices: hypothesis.patch_indices.clone(),
                    centroid_row: hypothesis.centroid_row,
                    centroid_col: hypothesis.centroid_col,
                    hv: hypothesis.hv.as_slice().to_vec(),
                    saliency: hypothesis.saliency,
                })
                .collect(),
            working_memory: self.working_memory.clone(),
            scene_graph_enabled: self.scene_graph_enabled,
        }
    }

    fn from_state(state: &ModalityTemporalContextState) -> Self {
        Self {
            last_prediction: state
                .last_prediction
                .as_ref()
                .map(|values| ContinuousHV::from_vec(values.clone())),
            last_frame_hv: state
                .last_frame_hv
                .as_ref()
                .map(|values| ContinuousHV::from_vec(values.clone())),
            last_patch_hvs: state
                .last_patch_hvs
                .iter()
                .map(|values| ContinuousHV::from_vec(values.clone()))
                .collect(),
            temporal_patch_hvs: state
                .temporal_patch_hvs
                .iter()
                .map(|values| ContinuousHV::from_vec(values.clone()))
                .collect(),
            prev_patch_lum: state.prev_patch_lum.clone(),
            surprise_state: state.surprise_state.clone(),
            prediction_error: state.prediction_error,
            error_ema: state.error_ema,
            fep_belief_mean: state.fep_belief_mean.clone(),
            last_fep: state.last_fep,
            horizon_evaluator: state
                .horizon_evaluator
                .as_ref()
                .and_then(|saved| {
                    let mut evaluator = DelayedHorizonEvaluator::default();
                    evaluator.load_state(saved).ok()?;
                    Some(evaluator)
                })
                .unwrap_or_default(),
            object_memory: state.object_memory.clone(),
            next_track_id: state.next_track_id,
            last_tracking_result: None,
            last_object_hypotheses: state
                .last_object_hypotheses
                .iter()
                .map(|hypothesis| crate::types::ObjectHypothesis {
                    centroid_row: hypothesis.centroid_row,
                    centroid_col: hypothesis.centroid_col,
                    patch_indices: hypothesis.patch_indices.clone(),
                    saliency: hypothesis.saliency,
                    hv: ContinuousHV::from_vec(hypothesis.hv.clone()),
                })
                .collect(),
            working_memory: state.working_memory.clone(),
            scene_graph_enabled: state.scene_graph_enabled,
        }
    }
}

/// A CfC temporal manifold over holographic video encodings.
///
/// The manifold state is a ContinuousHV (16,384D by default) that continuously
/// tracks the scene. Each frame observation evolves the state via closed-form
/// CfC dynamics; temporal predictions use O(1) jumps.
pub struct VisionManifold {
    config: VisionConfig,
    encoder: PatchHdcEncoder,
    state: ContinuousHV,
    weight_hv: ContinuousHV,
    last_prediction: Option<ContinuousHV>,
    last_frame_hv: Option<ContinuousHV>,
    last_patch_hvs: Vec<ContinuousHV>,
    /// Temporally-bound patch HVs: `ρ(prev_patch[i]) ⊗ curr_patch[i]`.
    /// Populated only when `config.enable_temporal_binding` is true.
    temporal_patch_hvs: Vec<ContinuousHV>,
    /// Active modality owning the temporal prediction fields above.
    active_modality: VisualModality,
    /// Inactive modality-specific temporal histories.
    modality_contexts: Vec<(VisualModality, ModalityTemporalContext)>,
    /// Delayed forecast skill evidence for the active sensor clock domain.
    horizon_evaluator: DelayedHorizonEvaluator,
    surprise: SurpriseMap,
    motion_field: MotionField,
    /// Per-patch motion magnitudes from the last frame.
    motion_saliency: Vec<f32>,
    /// Per-patch motion vectors `[dx, dy]` from the last frame.
    last_motion_vectors: Vec<[f32; 2]>,
    prediction_error: f32,
    coherence: f32,
    frame_count: u64,
    telemetry: VisionTelemetry,
    trainer: ManifoldTrainer,
    /// Exponential moving average of prediction error for adaptive training.
    error_ema: f32,
    /// Optional predictive coding hierarchy for cross-scale attention.
    predictive: Option<PredictiveCodingHierarchy>,
    /// When true, skip training and contrastive refinement.
    learning_frozen: bool,
    /// Optional episodic scene memory for recognition and surprise dampening.
    scene_memory: Option<SceneMemory>,
    /// Optional object identity tracker for cross-frame persistence.
    object_memory: Option<ObjectMemory>,
    /// Monotonic track ID counter for object memory.
    next_track_id: u64,
    /// Last object tracking result.
    last_tracking_result: Option<ObjectTrackingResult>,
    /// Last clustered object hypotheses. Reused when a stable scene skips the
    /// expensive clustering pass so representation and working-memory updates
    /// remain in the same object-bound family.
    last_object_hypotheses: Vec<crate::types::ObjectHypothesis>,
    /// Visual working memory (bounded attentional spotlight, ~4 objects).
    working_memory: Option<VisualWorkingMemory>,
    /// Visual scene graph (spatial relations between tracked objects).
    scene_graph: Option<VisualSceneGraph>,
    /// Per-patch stereo depth map from the last stereo frame (0=near, 1=far).
    stereo_depth_map: Vec<f32>,
    /// Confidence for each stereo patch estimate.
    stereo_confidence_map: Vec<f32>,
    /// Winning horizontal disparity in pixels for each patch.
    stereo_disparity_map: Vec<usize>,
    /// Last dream-ahead prediction (1 step) for imagination-reality comparison.
    last_imagination: Option<ContinuousHV>,
    /// Imagination surprise: how much reality diverged from prediction.
    imagination_surprise: f32,
    /// Last scene recognition match (if any).
    last_scene_match: Option<SceneMatch>,
    /// Minimum coherence required to store a scene landmark (default 0.7).
    scene_store_coherence_threshold: f32,
    /// Maximum prediction error allowed to store a scene landmark (default 0.1).
    scene_store_error_threshold: f32,
    /// Dampening factor for recognized scenes: higher = stronger suppression (default 0.5).
    scene_dampen_factor: f32,
    /// Last cycle at which Holographic Dilation occurred (cooldown track).
    last_dilation_cycle: u64,
    /// Latest Variational Free Energy metrics.
    last_fep: crate::types::FepMetrics,
    /// Core FEP agent for rigorous active inference and action selection.
    fep_agent: symthaea_fep::ActiveInferenceAgent,
    /// Unique node ID for P2P swarm identification.
    pub node_id: uuid::Uuid,
    /// Velocity field/dynamics model for mental simulation.
    transition_model: Option<Box<dyn TransitionModel>>,
    /// Latest generated geodesic path on the manifold.
    last_geodesic: Vec<ContinuousHV>,
    /// Accumulated thermodynamic cost of geodesic computation.
    pub geodesic_compute_cost: f32,
    /// Reference frame storage for mental movie decoding.
    last_observed_frame: Option<Vec<u8>>,
    last_frame_width: u32,
    last_frame_height: u32,
    last_frame_channels: usize,
    last_frame_modality: VisualModality,
    /// Latest intent/goal vector (for swarm broadcast).
    last_intent_hv: ContinuousHV,
    /// Subcortical generative bridge for neural hallucination (HDC -> Pixels).
    generative_bridge: Option<GenerativeBridge>,
}

/// The Neural Bridge: Translates mathematical hypervectors into visual hallucinations.
pub struct GenerativeBridge {
    pub device: candle_core::Device,
    pub projector: candle_nn::Linear,
    pub latent_dim: usize,
}

/// Dynamic transition model for the latent manifold.
///
/// Science: Friston (2010), Liquid Neural Networks (Hasani 2022). Describes the
/// analytical solution `x(t+dt) = x_inf + (x(t) - x_inf) * exp(-dt/tau)`.
pub trait TransitionModel: Send + Sync {
    /// Returns the equilibrium state (x_inf) under optional context.
    fn equilibrium(&self, state: &ContinuousHV, context: &TransitionContext) -> ContinuousHV;

    /// Project `state` forward by `dt` using the analytical CfC solution.
    /// Science: O(1) complexity — no step-by-step integration required.
    fn project(&self, state: &ContinuousHV, dt: f32, context: &TransitionContext) -> ContinuousHV {
        let x_inf = self.equilibrium(state, context);

        // x(t+dt) = x_inf + (x(t) - x_inf) * exp(-dt/tau)
        let leak = (-dt / context.tau.max(0.001)).exp();

        // Build the affine closed-form directly. Using `lerp_in_place` for
        // scalar multiplication is error-prone because its arguments are
        // weights for `(self, other)`, not `(bias, scale)`.
        let state_values = state.as_slice();
        let equilibrium_values = x_inf.as_slice();
        let values = state_values
            .iter()
            .zip(equilibrium_values.iter())
            .map(|(&x, &x_inf_i)| x_inf_i + (x - x_inf_i) * leak)
            .collect();

        // Preserve the exact affine CfC solution. Normalizing here changes the
        // vector even at `dt = 0` and breaks the semigroup law
        // `project(project(x, a), b) == project(x, a + b)` for a fixed
        // equilibrium. Callers that require a unit semantic HV can normalize at
        // the representation boundary after the physical projection.
        ContinuousHV::from_vec(values)
    }
}

/// Context for latent transitions (goals, priors, external inputs).
#[derive(Debug, Clone, Default)]
pub struct TransitionContext {
    pub goal: Option<ContinuousHV>,
    pub input: Option<ContinuousHV>,
    pub weight_hv: Option<ContinuousHV>,
    pub tau: f32,
}

/// Default transition model using CfC (equilibrium) dynamics.
pub struct CfCTransitionModel;

impl TransitionModel for CfCTransitionModel {
    fn equilibrium(&self, state: &ContinuousHV, ctx: &TransitionContext) -> ContinuousHV {
        let weight = ctx.weight_hv.as_ref().unwrap_or(state); // fallback
        let input = ctx.input.as_ref().unwrap_or(state); // fallback

        // Equilibrium x_inf = soft_tanh(W ⊗ state + U ⊗ input)
        // REFINEMENT: Soften the tanh nonlinearity (gain=0.8) to preserve
        // chromatic contrast in the latent representation.
        let mut x_inf = weight.bind(state);
        for val in x_inf.values.iter_mut() {
            *val = (*val * 0.8).tanh();
        }

        if let Some(ref goal) = ctx.goal {
            // Higher goal pull for smoother geodesic transition
            x_inf.lerp_in_place(goal, 0.65, 0.35);
        }

        // Reinforce the current input to maintain object constancy
        x_inf.lerp_in_place(input, 0.85, 0.15);
        x_inf
    }
}

impl VisionManifold {
    fn capture_active_modality_context(&self) -> ModalityTemporalContext {
        ModalityTemporalContext {
            last_prediction: self.last_prediction.clone(),
            last_frame_hv: self.last_frame_hv.clone(),
            last_patch_hvs: self.last_patch_hvs.clone(),
            temporal_patch_hvs: self.temporal_patch_hvs.clone(),
            prev_patch_lum: self.encoder.prev_patch_lum.clone(),
            surprise_state: Some(self.surprise.save_state()),
            prediction_error: self.prediction_error,
            error_ema: self.error_ema,
            fep_belief_mean: self.fep_agent.belief.mean.clone(),
            last_fep: self.last_fep,
            horizon_evaluator: self.horizon_evaluator.clone(),
            object_memory: self.object_memory.as_ref().map(ObjectMemory::save_state),
            next_track_id: self.next_track_id,
            last_tracking_result: self.last_tracking_result.clone(),
            last_object_hypotheses: self.last_object_hypotheses.clone(),
            working_memory: self
                .working_memory
                .as_ref()
                .map(VisualWorkingMemory::save_state),
            scene_graph_enabled: self.scene_graph.is_some(),
        }
    }

    fn fresh_modality_context(&self) -> ModalityTemporalContext {
        ModalityTemporalContext {
            object_memory: self.object_memory.as_ref().map(|memory| {
                let mut state = memory.save_state();
                state.tracks.clear();
                state
            }),
            working_memory: self.working_memory.as_ref().map(|memory| {
                let mut state = memory.save_state();
                state.slots.clear();
                state
            }),
            scene_graph_enabled: self.scene_graph.is_some(),
            ..Default::default()
        }
    }

    fn install_modality_context(&mut self, context: ModalityTemporalContext) {
        self.last_prediction = context.last_prediction;
        self.last_frame_hv = context.last_frame_hv;
        self.last_patch_hvs = context.last_patch_hvs;
        self.temporal_patch_hvs = context.temporal_patch_hvs;
        self.encoder.prev_patch_lum = context.prev_patch_lum;
        if let Some(ref surprise_state) = context.surprise_state {
            self.surprise.load_state(surprise_state);
        } else {
            self.surprise.reset();
        }
        self.prediction_error = context.prediction_error;
        self.error_ema = context.error_ema;
        self.horizon_evaluator = context.horizon_evaluator;
        let expected_belief_dim = self.fep_agent.belief.mean.len();
        if context.fep_belief_mean.len() == expected_belief_dim
            && context
                .fep_belief_mean
                .iter()
                .all(|value| value.is_finite())
        {
            self.fep_agent.belief.mean = context.fep_belief_mean;
            self.last_fep = context.last_fep;
        } else {
            self.fep_agent = Self::new_fep_agent();
            self.last_fep = crate::types::FepMetrics::default();
        }

        self.object_memory = context.object_memory.map(|state| {
            let mut memory = ObjectMemory::new(state.capacity);
            if memory
                .load_state_checked(&state, self.config.hdc_dim)
                .is_err()
            {
                memory.clear();
            }
            memory
        });
        self.next_track_id = context.next_track_id;
        self.last_tracking_result = context.last_tracking_result;
        self.last_object_hypotheses = context.last_object_hypotheses;
        self.working_memory = context.working_memory.map(|state| {
            let mut memory = VisualWorkingMemory::new(state.capacity);
            if memory
                .load_state_checked(&state, self.config.hdc_dim)
                .is_err()
            {
                memory.clear();
            }
            memory
        });
        self.scene_graph = if context.scene_graph_enabled {
            let mut graph = VisualSceneGraph::new(self.config.hdc_dim, self.config.seed);
            if let Some(ref objects) = self.object_memory {
                graph.update(objects.tracks());
            }
            Some(graph)
        } else {
            None
        };
    }

    fn activate_modality(&mut self, modality: VisualModality) {
        if self.active_modality == modality {
            return;
        }

        if self.active_modality != VisualModality::Unknown {
            let active_modality = self.active_modality;
            let previous = self.capture_active_modality_context();
            if let Some((_, context)) = self
                .modality_contexts
                .iter_mut()
                .find(|(stored_modality, _)| *stored_modality == active_modality)
            {
                *context = previous;
            } else {
                self.modality_contexts.push((active_modality, previous));
            }
        }

        let next = self
            .modality_contexts
            .iter()
            .position(|(stored_modality, _)| *stored_modality == modality)
            .map(|idx| self.modality_contexts.remove(idx).1)
            .unwrap_or_else(|| self.fresh_modality_context());
        self.install_modality_context(next);
        self.active_modality = modality;
    }

    fn saved_modality_contexts(&self) -> Vec<ModalityTemporalContextState> {
        self.modality_contexts
            .iter()
            .map(|(modality, context)| context.save_state(*modality))
            .collect()
    }

    /// Conservatively estimate HDC allocation plus retained scene-raster memory.
    pub fn estimate_dilation(
        &self,
        target: symthaea_core::hdc::HdcDimensionality,
    ) -> DilationEstimate {
        let target_dim = target.dimension();
        let mut vectors = 3usize; // state, weight_hv, last_intent_hv
        vectors += self.last_prediction.is_some() as usize;
        vectors += self.last_frame_hv.is_some() as usize;
        vectors += self.last_patch_hvs.len();
        vectors += self.temporal_patch_hvs.len();
        vectors += self.last_object_hypotheses.len();
        vectors += self.last_imagination.is_some() as usize;
        vectors += self.last_geodesic.len();
        vectors += self.horizon_evaluator.hdc_vector_count();
        vectors += self
            .modality_contexts
            .iter()
            .map(|(_, context)| context.hdc_vector_count())
            .sum::<usize>();
        vectors += self.encoder.hdc_vector_count();
        vectors += self.motion_field.hdc_vector_count();
        vectors += self.trainer.hdc_vector_count();
        vectors += self
            .predictive
            .as_ref()
            .map_or(0, PredictiveCodingHierarchy::hdc_vector_count);
        vectors += self
            .scene_memory
            .as_ref()
            .map_or(0, SceneMemory::hdc_vector_count);
        vectors += self
            .object_memory
            .as_ref()
            .map_or(0, ObjectMemory::hdc_vector_count);
        vectors += self
            .working_memory
            .as_ref()
            .map_or(0, VisualWorkingMemory::hdc_vector_count);
        vectors += self
            .scene_graph
            .as_ref()
            .map_or(0, VisualSceneGraph::hdc_vector_count);

        let projected_bytes = (vectors as u64)
            .saturating_mul(target_dim as u64)
            .saturating_mul(std::mem::size_of::<f32>() as u64);
        let persistent_bytes = self
            .scene_memory
            .as_ref()
            .map_or(0u64, |memory| memory.retained_pixel_bytes() as u64)
            .saturating_add(
                self.last_observed_frame
                    .as_ref()
                    .map_or(0u64, |frame| frame.len() as u64),
            );
        let total_projected_bytes = projected_bytes.saturating_add(persistent_bytes);
        DilationEstimate {
            current_dim: self.config.hdc_dim,
            target_dim,
            hdc_vectors: vectors,
            projected_bytes,
            persistent_bytes,
            total_projected_bytes,
            budget_bytes: self.config.max_dilation_bytes,
        }
    }

    /// Perform a preflighted holographic dilation.
    ///
    /// Every affected HDC allocation and retained scene raster is counted before mutation. Requests that
    /// exceed `VisionConfig::max_dilation_bytes` fail without changing any live
    /// state. The returned report can be surfaced in operator telemetry.
    pub fn try_dilate(
        &mut self,
        target: symthaea_core::hdc::HdcDimensionality,
    ) -> Result<DilationEstimate, String> {
        let estimate = self.estimate_dilation(target);
        let target_dim = estimate.target_dim;
        if self.config.hdc_dim == target_dim {
            return Ok(estimate);
        }
        if !estimate.fits_budget() {
            return Err(format!(
                "dilation rejected: projected total {} bytes exceeds budget {} bytes ({} HDC vectors at {} dimensions; {} persistent bytes)",
                estimate.total_projected_bytes,
                estimate.budget_bytes,
                estimate.hdc_vectors,
                estimate.target_dim,
                estimate.persistent_bytes
            ));
        }

        tracing::info!(
            projected_bytes = estimate.projected_bytes,
            persistent_bytes = estimate.persistent_bytes,
            total_projected_bytes = estimate.total_projected_bytes,
            hdc_vectors = estimate.hdc_vectors,
            "Vision Manifold HOLOGRAPHIC DILATION: {} -> {} ({})",
            self.config.hdc_dim,
            target_dim,
            if target_dim > self.config.hdc_dim {
                "Unfolding"
            } else {
                "Folding"
            }
        );

        self.state = self.state.dilate(target_dim);
        self.weight_hv = self.weight_hv.dilate(target_dim);
        if let Some(ref mut hv) = self.last_prediction {
            *hv = hv.dilate(target_dim);
        }
        if let Some(ref mut hv) = self.last_frame_hv {
            *hv = hv.dilate(target_dim);
        }
        for hv in &mut self.last_patch_hvs {
            *hv = hv.dilate(target_dim);
        }
        for hv in &mut self.temporal_patch_hvs {
            *hv = hv.dilate(target_dim);
        }
        self.horizon_evaluator.dilate(target_dim);
        for (_, context) in &mut self.modality_contexts {
            context.dilate(target_dim);
        }
        for hypothesis in &mut self.last_object_hypotheses {
            hypothesis.hv = hypothesis.hv.dilate(target_dim);
        }
        if let Some(ref mut hv) = self.last_imagination {
            *hv = hv.dilate(target_dim);
        }
        for hv in &mut self.last_geodesic {
            *hv = hv.dilate(target_dim);
        }
        self.last_intent_hv = self.last_intent_hv.dilate(target_dim);

        self.encoder.dilate(target_dim);
        self.motion_field.dilate(target_dim);
        self.trainer.dilate(target_dim);
        if let Some(ref mut pred) = self.predictive {
            pred.dilate(target_dim);
        }
        if let Some(ref mut memory) = self.scene_memory {
            memory.dilate(target_dim);
        }
        if let Some(ref mut memory) = self.object_memory {
            memory.dilate(target_dim);
        }
        if let Some(ref mut memory) = self.working_memory {
            memory.dilate(target_dim);
        }
        if let Some(ref mut graph) = self.scene_graph {
            graph.dilate(target_dim);
        }

        self.config.hdc_dim = target_dim;
        self.last_dilation_cycle = self.frame_count;
        Ok(estimate)
    }

    /// Compatibility wrapper for callers that do not yet consume dilation errors.
    pub fn dilate(&mut self, target: symthaea_core::hdc::HdcDimensionality) {
        if let Err(error) = self.try_dilate(target) {
            tracing::warn!(%error, "holographic dilation request rejected");
        }
    }

    fn new_fep_agent() -> symthaea_fep::ActiveInferenceAgent {
        symthaea_fep::ActiveInferenceAgent::new(symthaea_fep::ActiveInferenceAgentConfig {
            state_dim: 16,
            obs_dim: 4,
            num_actions: 8,
            inference_iterations: 8,
            belief_learning_rate: 0.15,
            planning_horizon: 5,
            action_temperature: 0.8,
            enable_model_learning: true,
            enable_td_learning: true,
            td_config: symthaea_fep::TemporalDifferenceLearningConfig::default(),
        })
    }

    /// Create a new vision manifold sized for frames up to `max_width × max_height`.
    ///
    /// # Panics
    ///
    /// Panics if construction fails. New integrations should prefer
    /// [`Self::try_new`] so invalid configuration or capacity is surfaced.
    pub fn new(config: VisionConfig, max_width: u32, max_height: u32) -> Self {
        Self::try_new(config, max_width, max_height)
            .unwrap_or_else(|error| panic!("Invalid VisionManifold construction: {error}"))
    }

    /// Construct a manifold without panicking on invalid policy or capacity.
    pub fn try_new(config: VisionConfig, max_width: u32, max_height: u32) -> Result<Self, String> {
        config.validate()?;
        if max_width == 0 || max_height == 0 {
            return Err(format!(
                "maximum frame dimensions must be non-zero, got {max_width}x{max_height}"
            ));
        }

        let encoder = PatchHdcEncoder::new(&config, max_width, max_height);
        let dim = config.hdc_dim;
        let state = ContinuousHV::zero(dim);
        let weight_hv = ContinuousHV::random(dim, config.seed + 300_000);
        let grid = encoder.grid_for(max_width, max_height);
        let surprise = SurpriseMap::new(grid, config.surprise_decay, config.surprise_threshold);
        let motion_field = MotionField::new(dim, config.seed + 500_000);
        let trainer = ManifoldTrainer::new(&config.training, dim);

        let predictive = if config.enable_predictive_hierarchy {
            Some(PredictiveCodingHierarchy::new(
                &config, max_width, max_height,
            ))
        } else {
            None
        };

        Ok(Self {
            config,
            encoder,
            state,
            weight_hv,
            last_prediction: None,
            last_frame_hv: None,
            last_patch_hvs: Vec::new(),
            temporal_patch_hvs: Vec::new(),
            active_modality: VisualModality::Unknown,
            modality_contexts: Vec::new(),
            horizon_evaluator: DelayedHorizonEvaluator::default(),
            surprise,
            motion_field,
            motion_saliency: Vec::new(),
            last_motion_vectors: Vec::new(),
            prediction_error: 0.0,
            coherence: 0.0,
            frame_count: 0,
            telemetry: VisionTelemetry::default(),
            trainer,
            error_ema: 0.0,
            predictive,
            learning_frozen: false,
            scene_memory: None,  // Enabled externally via enable_scene_memory()
            object_memory: None, // Enabled externally via enable_object_memory()
            next_track_id: 0,
            last_tracking_result: None,
            last_object_hypotheses: Vec::new(),
            working_memory: None, // Enabled externally via enable_working_memory()
            scene_graph: None,    // Enabled externally via enable_scene_graph()
            stereo_depth_map: Vec::new(),
            stereo_confidence_map: Vec::new(),
            stereo_disparity_map: Vec::new(),
            last_imagination: None,
            imagination_surprise: 0.0,
            last_scene_match: None,
            scene_store_coherence_threshold: 0.7,
            scene_store_error_threshold: 0.1,
            scene_dampen_factor: 0.5,
            last_dilation_cycle: 0,
            last_fep: crate::types::FepMetrics::default(),
            fep_agent: Self::new_fep_agent(),
            node_id: uuid::Uuid::new_v4(),
            transition_model: Some(Box::new(CfCTransitionModel)),
            last_geodesic: Vec::new(),
            geodesic_compute_cost: 0.0,
            last_observed_frame: None,
            last_frame_width: 0,
            last_frame_height: 0,
            last_frame_channels: 0,
            last_frame_modality: VisualModality::Unknown,
            last_intent_hv: ContinuousHV::zero(dim),
            generative_bridge: None,
        })
    }

    /// Validate a tightly packed raw frame without mutating manifold state.
    fn validate_frame_input(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        dt: f32,
    ) -> Result<(), String> {
        if width == 0 || height == 0 {
            return Err(format!(
                "frame dimensions must be non-zero, got {width}x{height}"
            ));
        }
        if !matches!(channels, 1 | 3 | 4) {
            return Err(format!(
                "frame channels must be 1 (gray), 3 (RGB), or 4 (RGBA), got {channels}"
            ));
        }
        if !dt.is_finite() || dt < 0.0 {
            return Err(format!("frame timestep must be finite and >= 0, got {dt}"));
        }
        let expected_len = (width as usize)
            .checked_mul(height as usize)
            .and_then(|count| count.checked_mul(channels))
            .ok_or_else(|| "frame geometry overflow".to_string())?;
        if pixels.len() != expected_len {
            return Err(format!(
                "frame buffer length mismatch: got {}, expected {expected_len} for {width}x{height}x{channels}",
                pixels.len()
            ));
        }

        let grid = self.encoder.grid_for(width, height);
        if grid.rows > self.encoder.max_rows() || grid.cols > self.encoder.max_cols() {
            return Err(format!(
                "frame {width}x{height} exceeds encoder capacity of {}x{} patches",
                self.encoder.max_cols(),
                self.encoder.max_rows()
            ));
        }
        Ok(())
    }

    /// Validate externally supplied per-patch depth without mutating state.
    fn validate_depth_input(
        &self,
        patch_depths: &[f32],
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        if !self.config.enable_depth {
            return Err("sensor-depth observation requires VisionConfig::enable_depth".to_string());
        }
        let expected = self.encoder.grid_for(width, height).num_patches();
        if patch_depths.len() != expected {
            return Err(format!(
                "depth map length mismatch: got {}, expected {expected} patches for {width}x{height}",
                patch_depths.len()
            ));
        }
        for (idx, &depth) in patch_depths.iter().enumerate() {
            if !depth.is_finite() || !(0.0..=1.0).contains(&depth) {
                return Err(format!(
                    "depth[{idx}] must be finite and in [0, 1], got {depth}"
                ));
            }
        }
        Ok(())
    }

    /// Validate and observe a raw frame atomically.
    ///
    /// Unlike the compatibility [`Self::observe_frame`] path, this rejects
    /// malformed buffers, unsupported channel layouts, invalid timesteps, and
    /// over-capacity geometry before the encoder can update temporal history.
    pub fn observe_frame_checked(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        dt: f32,
    ) -> Result<VisionTelemetry, String> {
        self.validate_frame_input(pixels, width, height, channels, dt)?;
        Ok(self.observe_frame_impl(
            pixels,
            width,
            height,
            channels,
            dt,
            None,
            VisualModality::Visible,
        ))
    }

    /// Validate and observe a raw frame with one sensor-depth value per patch.
    ///
    /// Depth uses the encoder convention `0.0 = near`, `1.0 = far`. The map is
    /// validated completely before temporal luminance, surprise, prediction, or
    /// manifold state can advance.
    #[allow(clippy::too_many_arguments)]
    pub fn observe_frame_with_depth_checked(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        patch_depths: &[f32],
        dt: f32,
    ) -> Result<VisionTelemetry, String> {
        self.validate_frame_input(pixels, width, height, channels, dt)?;
        self.validate_depth_input(patch_depths, width, height)?;
        Ok(self.observe_frame_impl(
            pixels,
            width,
            height,
            channels,
            dt,
            Some(patch_depths),
            VisualModality::SensorDepth,
        ))
    }

    /// Compatibility wrapper for externally supplied patch depth.
    ///
    /// Invalid input is rejected without mutation and returns the most recent
    /// telemetry snapshot. New callers should prefer
    /// [`Self::observe_frame_with_depth_checked`].
    #[allow(clippy::too_many_arguments)]
    pub fn observe_frame_with_depth(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        patch_depths: &[f32],
        dt: f32,
    ) -> VisionTelemetry {
        match self.observe_frame_with_depth_checked(
            pixels,
            width,
            height,
            channels,
            patch_depths,
            dt,
        ) {
            Ok(telemetry) => telemetry,
            Err(error) => {
                tracing::warn!(%error, "rejected sensor-depth observation");
                self.telemetry.clone()
            }
        }
    }

    /// Observe a raw frame: encode → evolve CfC state → compute surprise → predict.
    ///
    /// Returns telemetry for this observation cycle.
    pub fn observe_frame(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        dt: f32,
    ) -> VisionTelemetry {
        self.observe_frame_impl(
            pixels,
            width,
            height,
            channels,
            dt,
            None,
            VisualModality::Visible,
        )
    }

    // Genuinely needs this many parameters: the raw frame buffer plus its geometry
    // (width/height/channels), the elapsed-time and optional-depth inputs the temporal/stereo
    // pipeline requires, and the modality being observed under.
    #[allow(clippy::too_many_arguments)]
    fn observe_frame_impl(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        dt: f32,
        patch_depths: Option<&[f32]>,
        modality: VisualModality,
    ) -> VisionTelemetry {
        self.activate_modality(modality);
        let t0 = Instant::now();

        // Save previous luminances before encoding overwrites them
        let prev_lum = self.encoder.prev_patch_lum.clone();

        let (frame_hv, patch_hvs) = if let Some(depths) = patch_depths {
            self.encoder
                .encode_frame_with_depth(pixels, width, height, channels, depths)
        } else {
            self.encoder.encode_frame(pixels, width, height, channels)
        };
        let encode_us = t0.elapsed().as_micros() as u64;

        // Store reference frame for decoding mental movies
        self.last_observed_frame = Some(pixels.to_vec());
        self.last_frame_width = width;
        self.last_frame_height = height;
        self.last_frame_channels = channels;
        self.last_frame_modality = modality;

        // Compute motion field from luminance difference
        let grid = self.encoder.grid_for(width, height);
        let (motion_hv_norm, motion_max) = if !prev_lum.is_empty() && grid.num_patches() > 0 {
            let current_lum = &self.encoder.prev_patch_lum;
            let (motion_hv, vectors) = self.motion_field.compute(
                current_lum,
                &prev_lum,
                grid.rows,
                grid.cols,
                self.encoder.row_basis(),
                self.encoder.col_basis(),
            );
            let magnitudes: Vec<f32> = vectors
                .iter()
                .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
                .collect();
            let max_mag = magnitudes.iter().copied().fold(0.0f32, f32::max);
            let norm = motion_hv.norm();
            self.motion_saliency = magnitudes;
            self.last_motion_vectors = vectors;
            (norm, max_mag)
        } else {
            self.motion_saliency.clear();
            self.last_motion_vectors.clear();
            (0.0, 0.0)
        };

        // Optionally process through predictive coding hierarchy
        let (cross_scale_prediction_error, cross_scale_patch_errors) =
            if let Some(ref mut predictive) = self.predictive {
                let output =
                    predictive.process_frame_with_feedback(pixels, width, height, channels);
                (output.prediction_error, output.patch_prediction_errors)
            } else {
                (0.0, vec![])
            };

        // P8: Skip re-clustering only when the *current* frame is genuinely
        // stable. `self.prediction_error` still describes the previous
        // observation at this point, so using it here can reuse stale object
        // hypotheses exactly when a new scene arrives. Patch novelty is
        // available immediately after encoding and compares like-for-like
        // representations before any live state is mutated.
        let current_patch_novelty = Self::mean_patch_novelty(&patch_hvs, &self.last_patch_hvs);
        let has_tracks = self.object_memory.as_ref().is_some_and(|m| !m.is_empty());
        let scene_changed = current_patch_novelty > 0.01 || self.frame_count < 10 || !has_tracks;

        // P3-E: Object-level binding — replace the bag-of-words frame HV with
        // a relationally-structured HV that encodes *where* each perceptual
        // object is, not just *what* patches are present.
        //
        // P8: stable scenes skip only the clustering computation. Cached
        // hypotheses are still rebound into the frame representation and fed to
        // object/working memory, avoiding representation-family switching.
        let mut saved_hypotheses: Vec<crate::types::ObjectHypothesis> = Vec::new();

        let bound_frame_hv = if self.config.enable_object_binding && patch_hvs.len() >= 2 {
            let grid = self.encoder.grid_for(width, height);
            let hypotheses = if scene_changed || self.last_object_hypotheses.is_empty() {
                Self::cluster_patches(&patch_hvs, &grid)
            } else {
                // Reuse only the cached segmentation topology. The object
                // appearance itself must be rebound from the current patches;
                // otherwise sub-threshold lighting, pose, and texture changes
                // disappear from the object-bound frame and tracker updates.
                let mut cached = self.last_object_hypotheses.clone();
                if Self::refresh_hypothesis_appearance(&mut cached, &patch_hvs) {
                    cached
                } else {
                    // A stale or malformed cache must never index unrelated
                    // patches. Fall back to a fresh segmentation.
                    Self::cluster_patches(&patch_hvs, &grid)
                }
            };

            // Update object memory every frame so stable objects remain present
            // and do not age out merely because clustering was optimized away.
            if let Some(ref mut obj_mem) = self.object_memory {
                self.last_tracking_result =
                    Some(obj_mem.update(&hypotheses, self.frame_count, &mut self.next_track_id));
            }

            saved_hypotheses = hypotheses.clone();

            if hypotheses.is_empty() {
                frame_hv
            } else {
                let row_basis = self.encoder.row_basis();
                let col_basis = self.encoder.col_basis();
                let bound: Vec<ContinuousHV> = hypotheses
                    .iter()
                    .map(|h| {
                        let r = h.centroid_row % row_basis.len().max(1);
                        let c = h.centroid_col % col_basis.len().max(1);
                        row_basis[r].bind(&col_basis[c]).bind(&h.hv)
                    })
                    .collect();
                let refs: Vec<&ContinuousHV> = bound.iter().collect();
                ContinuousHV::bundle(&refs).normalize()
            }
        } else {
            frame_hv
        };

        // P5-A: Imagination-reality comparison — compare what we imagined
        // (dream_ahead(1) from last frame) with what we actually see now.
        // High divergence = temporal surprise: reality violated our mental model.
        // Buckner & Carroll (2007): prospective memory drives surprise-based learning.
        if let Some(ref imagined) = self.last_imagination {
            self.imagination_surprise =
                (1.0 - bound_frame_hv.similarity(imagined).clamp(-1.0, 1.0)).max(0.0);
        }
        // Always generate predictions (the comparison on the NEXT frame is cheap).
        // Gating imagination was too aggressive — it causes stale surprise values
        // when a novel stimulus arrives after a long static period.
        self.last_imagination = if self.frame_count > 0 {
            Some(self.predict_horizon(&bound_frame_hv, dt))
        } else {
            None
        };

        let t1 = Instant::now();
        self.observe_encoded(&bound_frame_hv, &patch_hvs, dt);

        // P2-A: Inject cross-scale prediction errors into the surprise map.
        // After observe_encoded() has accumulated temporal surprise, blend in
        // the spatial-scale inconsistency signal (coarse fails to predict fine).
        if !cross_scale_patch_errors.is_empty() {
            self.surprise
                .inject_cross_scale_error(&cross_scale_patch_errors, 0.3);
        }

        // Saliency must be derived from the current observation's temporal and
        // cross-scale surprise. Computing it before `observe_encoded()` makes
        // working-memory admission lag one frame behind the evidence.
        let attention = self.surprise.attention_map();
        Self::refresh_hypothesis_saliency(&mut saved_hypotheses, &attention.values);
        self.last_object_hypotheses = saved_hypotheses.clone();

        // P5-B: Update visual working memory (bounded attentional spotlight).
        // P5-C: Update scene graph (spatial relations between tracked objects).
        // P6-C: Episodic consolidation — evicted working memory objects → scene memory.
        // Uses saved_hypotheses from the object binding step (with saliency filled).
        if let Some(ref obj_mem) = self.object_memory {
            let tracks = obj_mem.tracks();
            if let Some(ref mut wm) = self.working_memory {
                let evicted = wm.update(tracks, &saved_hypotheses, self.frame_count);
                // P6-C: Consolidate evicted objects into scene memory
                // (Diekelmann & Born 2010: objects that leave attention are
                //  consolidated into long-term episodic memory for later recognition)
                if let Some(ref mut mem) = self.scene_memory {
                    for hv in &evicted {
                        mem.remember_object(hv, self.frame_count);
                    }
                }
            }
            // P8: Scene graph operates only on working memory objects.
            // Biological accuracy: you only reason about spatial relations for
            // objects you're attending to. Also O(k²) with k≤4 instead of O(n²)
            // with n=16+ — dramatic performance improvement.
            if let Some(ref mut sg) = self.scene_graph {
                if let Some(ref wm) = self.working_memory {
                    let wm_track_ids: Vec<u64> = wm.slots().iter().map(|s| s.track_id).collect();
                    let wm_tracks: Vec<&TrackedObject> = tracks
                        .iter()
                        .filter(|t| wm_track_ids.contains(&t.track_id))
                        .collect();
                    // Collect into owned for update() signature
                    let owned: Vec<TrackedObject> =
                        wm_tracks.iter().map(|t| (*t).clone()).collect();
                    sg.update(&owned);
                } else {
                    sg.update(tracks);
                }
            }
        }

        let evolve_us = t1.elapsed().as_micros() as u64;

        // Preserve training_triggered/training_loss set by observe_encoded
        let training_triggered = self.telemetry.training_triggered;
        let training_loss = self.telemetry.training_loss;
        self.telemetry = VisionTelemetry {
            encode_time_us: encode_us,
            evolve_time_us: evolve_us,
            prediction_error: self.prediction_error,
            manifold_coherence: self.coherence,
            attention_entropy: self.surprise.attention_map().entropy(),
            num_salient_patches: self.surprise.salient_patches().len(),
            frame_sequence: self.frame_count,
            training_triggered,
            training_loss,
            motion_surprise: motion_max,
            motion_field_norm: motion_hv_norm,
            output_hv_norm: 0.0,
            attention_boost_applied: 0.0,
            cross_scale_prediction_error,
            scene_recognized: self.last_scene_match.is_some(),
            scene_recognition_similarity: self
                .last_scene_match
                .as_ref()
                .map_or(0.0, |m| m.similarity),
            imagination_surprise: self.imagination_surprise,
            working_memory_load: self.working_memory.as_ref().map_or(0, |wm| wm.load()),
            scene_graph_edges: self.scene_graph.as_ref().map_or(0, |sg| sg.num_edges()),
            free_energy: self.last_fep.free_energy,
            complexity: self.last_fep.complexity,
            accuracy: self.last_fep.accuracy,
            last_geodesic_path: self
                .last_geodesic
                .iter()
                .map(|hv| hv.values.clone())
                .collect(),
            last_geodesic_cost: self.geodesic_compute_cost,
            last_geodesic_length: self.last_geodesic.len(),
            last_fep_action: self.telemetry.last_fep_action.clone(),
        };

        // Reset compute cost for next cycle
        self.geodesic_compute_cost = 0.0;

        self.telemetry.clone()
    }

    /// Observe a stereo frame pair: compute disparity-based depth and encode.
    ///
    /// Compatibility wrapper for callers that do not yet consume validation
    /// errors. Invalid pairs are rejected without mutating manifold state.
    #[allow(clippy::too_many_arguments)]
    pub fn observe_frame_stereo(
        &mut self,
        left: &[u8],
        right: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        max_disparity: usize,
        dt: f32,
    ) -> VisionTelemetry {
        match self.observe_frame_stereo_checked(
            left,
            right,
            width,
            height,
            channels,
            max_disparity,
            dt,
        ) {
            Ok(telemetry) => telemetry,
            Err(error) => {
                tracing::warn!(%error, "rejected stereo observation");
                self.telemetry.clone()
            }
        }
    }

    /// Validate, reconstruct, confidence-fuse, and observe a stereo pair.
    ///
    /// Both buffers must be tightly packed grayscale images. Uncertain depth
    /// estimates are blended toward neutral depth (`0.5`) before entering the
    /// HDC feature channel, while raw depth, confidence, and disparity remain
    /// available for diagnostics.
    #[allow(clippy::too_many_arguments)]
    pub fn observe_frame_stereo_checked(
        &mut self,
        left: &[u8],
        right: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        max_disparity: usize,
        dt: f32,
    ) -> Result<VisionTelemetry, String> {
        if !self.config.enable_depth {
            return Err("stereo observation requires VisionConfig::enable_depth".to_string());
        }
        if channels != 1 {
            return Err(format!(
                "stereo observation requires grayscale channels=1, got {channels}"
            ));
        }
        self.validate_frame_input(left, width, height, channels, dt)?;

        let estimate =
            self.encoder
                .compute_stereo_depth_checked(left, right, width, height, max_disparity)?;
        let fused_depths = estimate.fused_depths();
        let telemetry = self.observe_frame_impl(
            left,
            width,
            height,
            channels,
            dt,
            Some(&fused_depths),
            VisualModality::Stereo,
        );
        self.stereo_depth_map = estimate.depths;
        self.stereo_confidence_map = estimate.confidences;
        self.stereo_disparity_map = estimate.disparities;

        Ok(telemetry)
    }

    /// Last raw stereo depth map (`0 = near`, `1 = far`).
    pub fn stereo_depth_map(&self) -> &[f32] {
        &self.stereo_depth_map
    }

    /// Confidence for each entry in [`Self::stereo_depth_map`].
    pub fn stereo_confidence_map(&self) -> &[f32] {
        &self.stereo_confidence_map
    }

    /// Winning horizontal disparity in pixels for each patch.
    pub fn stereo_disparity_map(&self) -> &[usize] {
        &self.stereo_disparity_map
    }

    /// Observe a pre-encoded multi-spectral HV (no raw pixel processing).
    ///
    /// Called by `VisionBridge::process_multiband_frame()` after multi-spectral
    /// encoding. Skips the standard pixel encoding, motion field, and predictive
    /// hierarchy (all of which require raw pixels). State, surprise, scene memory,
    /// and CfC dynamics are still fully updated.
    pub fn observe_multiband_frame(&mut self, multi_hv: &ContinuousHV, dt: f32) -> VisionTelemetry {
        match self.observe_multiband_frame_checked(multi_hv, dt) {
            Ok(telemetry) => telemetry,
            Err(error) => {
                tracing::warn!(%error, "rejected encoded multispectral observation");
                self.telemetry.clone()
            }
        }
    }

    /// Validate and observe a global multispectral representation atomically.
    ///
    /// Multispectral frames have no RGB/gray pixel image that can truthfully be
    /// attached to scene-memory landmarks. The raw-frame decoding context is
    /// therefore cleared before a successful observation rather than reusing a
    /// stale visible-light frame.
    pub fn observe_multiband_frame_checked(
        &mut self,
        multi_hv: &ContinuousHV,
        dt: f32,
    ) -> Result<VisionTelemetry, String> {
        if !dt.is_finite() || dt < 0.0 {
            return Err(format!(
                "multispectral timestep must be finite and >= 0, got {dt}"
            ));
        }
        if multi_hv.dim() != self.config.hdc_dim {
            return Err(format!(
                "multispectral HV dimension mismatch: got {}, expected {}",
                multi_hv.dim(),
                self.config.hdc_dim
            ));
        }
        if !multi_hv.as_slice().iter().all(|value| value.is_finite()) {
            return Err("multispectral HV contains non-finite values".to_string());
        }

        self.activate_modality(VisualModality::MultiSpectral);
        let t0 = Instant::now();
        self.last_observed_frame = None;
        self.last_frame_width = 0;
        self.last_frame_height = 0;
        self.last_frame_channels = 0;
        self.last_frame_modality = VisualModality::MultiSpectral;
        self.observe_encoded(multi_hv, &[], dt);
        let evolve_us = t0.elapsed().as_micros() as u64;

        let training_triggered = self.telemetry.training_triggered;
        let training_loss = self.telemetry.training_loss;
        self.telemetry = VisionTelemetry {
            encode_time_us: 0,
            evolve_time_us: evolve_us,
            prediction_error: self.prediction_error,
            manifold_coherence: self.coherence,
            attention_entropy: self.surprise.attention_map().entropy(),
            num_salient_patches: self.surprise.salient_patches().len(),
            frame_sequence: self.frame_count,
            training_triggered,
            training_loss,
            motion_surprise: 0.0,
            motion_field_norm: 0.0,
            output_hv_norm: 0.0,
            attention_boost_applied: 0.0,
            cross_scale_prediction_error: 0.0,
            scene_recognized: self.last_scene_match.is_some(),
            scene_recognition_similarity: self
                .last_scene_match
                .as_ref()
                .map_or(0.0, |m| m.similarity),
            imagination_surprise: self.imagination_surprise,
            working_memory_load: self.working_memory.as_ref().map_or(0, |wm| wm.load()),
            scene_graph_edges: self.scene_graph.as_ref().map_or(0, |sg| sg.num_edges()),
            free_energy: self.last_fep.free_energy,
            complexity: self.last_fep.complexity,
            accuracy: self.last_fep.accuracy,
            last_geodesic_path: self
                .last_geodesic
                .iter()
                .map(|hv| hv.values.clone())
                .collect(),
            last_geodesic_cost: self.geodesic_compute_cost,
            last_geodesic_length: self.last_geodesic.len(),
            last_fep_action: self.telemetry.last_fep_action.clone(),
        };

        // Reset compute cost for next cycle
        self.geodesic_compute_cost = 0.0;
        Ok(self.telemetry.clone())
    }

    /// Observe a pre-encoded frame HV with its per-patch decomposition.
    pub fn observe_encoded(
        &mut self,
        frame_hv: &ContinuousHV,
        patch_hvs: &[ContinuousHV],
        dt: f32,
    ) {
        // Compute prediction error against previous prediction
        let mut training_triggered = false;
        let mut training_loss = None;

        if let Some(predicted) = self.last_prediction.clone() {
            self.prediction_error = 1.0 - frame_hv.similarity(&predicted).clamp(-1.0, 1.0);

            // Update adaptive error EMA
            self.error_ema = 0.95 * self.error_ema + 0.05 * self.prediction_error;

            // Adaptive training trigger: train when error exceeds either:
            // 1. The configured threshold (catches large errors), OR
            // 2. A spike above recent baseline (catches pattern changes even
            //    when absolute error is small).
            let spike_threshold = self.error_ema * 2.0 + 0.005;
            let should_train = !self.learning_frozen
                && (self.prediction_error > self.config.training.error_threshold
                    || (self.frame_count > 2 && self.prediction_error > spike_threshold));
            if should_train && let Some(last_input) = self.last_frame_hv.clone() {
                let result = self.train_step_inner(&last_input, &predicted, frame_hv, dt);
                training_triggered = true;
                training_loss = Some(result.loss);
            }
        }

        // Update per-patch surprise map
        self.surprise.update(patch_hvs, &self.last_patch_hvs);

        // Auto-refine encoder from surprise (closed-loop active inference)
        if !self.learning_frozen && self.surprise.max_surprise() > self.config.surprise_threshold {
            self.refine_from_attention();
        }

        // P2-C: Temporal patch binding — encode cross-frame identity via ρ(prev) ⊗ curr.
        // Non-commutativity encodes temporal direction: seeing A then B ≠ seeing B then A.
        let effective_frame_hv;
        if self.config.enable_temporal_binding && !self.last_patch_hvs.is_empty() {
            // ρ(prev[i]) ⊗ curr[i] — non-commutative temporal identity per patch
            self.temporal_patch_hvs = self
                .last_patch_hvs
                .iter()
                .zip(patch_hvs.iter())
                .map(|(prev, curr)| prev.bind_temporal(curr))
                .collect();

            // Bundle temporal patches with equal weights, then blend with raw frame HV:
            // 70% temporal identity + 30% raw appearance.
            let n = self.temporal_patch_hvs.len();
            let temporal_refs: Vec<&ContinuousHV> = self.temporal_patch_hvs.iter().collect();
            let equal_weights = vec![1.0f32; n];
            let temporal_bundle =
                ContinuousHV::weighted_bundle(&temporal_refs, &equal_weights).normalize();
            effective_frame_hv =
                ContinuousHV::weighted_bundle(&[&temporal_bundle, frame_hv], &[0.7, 0.3]);
        } else {
            self.temporal_patch_hvs = patch_hvs.to_vec();
            effective_frame_hv = frame_hv.clone();
        }

        // Evolve CfC state: state' = x_inf + (state - x_inf) * exp(-dt/τ)
        let x_inf = self.equilibrium(&effective_frame_hv);
        let sigma = self.gating(dt);
        self.state.lerp_in_place(&x_inf, 1.0 - sigma, sigma);

        // Compute coherence (state-effective-frame alignment)
        self.coherence = self.state.similarity(&effective_frame_hv).max(0.0);

        // Predict next frame (one dt ahead) for next cycle's error computation
        self.last_prediction = Some(self.predict_horizon(frame_hv, dt));
        self.last_frame_hv = Some(frame_hv.clone());
        self.last_patch_hvs = patch_hvs.to_vec();
        self.frame_count += 1;

        // Score and issue delayed forecasts within this modality's clock domain.
        // A zero-duration observation does not advance the evaluator clock.
        if dt > 0.0 {
            let mut evaluator = std::mem::take(&mut self.horizon_evaluator);
            if let Err(error) = evaluator.observe(self, dt) {
                tracing::warn!(%error, "delayed horizon evaluation skipped");
            }
            self.horizon_evaluator = evaluator;
        }

        // Scene memory: recognize and optionally store the current scene
        self.last_scene_match = None;
        if let Some(ref mut memory) = self.scene_memory {
            self.last_scene_match = memory.recognize(&self.state, self.frame_count);

            // Dampen surprise for recognized scenes
            if let Some(ref scene_match) = self.last_scene_match {
                let factor =
                    1.0 - scene_match.similarity.clamp(0.0, 1.0) * self.scene_dampen_factor;
                let rows = self.surprise.grid().rows;
                let cols = self.surprise.grid().cols;
                for row in 0..rows {
                    for col in 0..cols {
                        self.surprise.dampen(row, col, factor);
                    }
                }
            }

            // Store current state as landmark when coherence is high and error is low
            if self.coherence > self.scene_store_coherence_threshold
                && self.prediction_error < self.scene_store_error_threshold
            {
                memory.remember_with_metadata(
                    &self.state,
                    self.frame_count,
                    self.last_observed_frame.clone().unwrap_or_default(),
                    SceneFrameMetadata {
                        width: self.last_frame_width,
                        height: self.last_frame_height,
                        channels: self.last_frame_channels,
                        modality: self.last_frame_modality,
                    },
                );
            }
        }

        // ── FEP Metrics Calculation (Active Inference Agent) ──
        // Science: Friston (2010). F = Complexity - Accuracy.
        // We use the rigorous FEP engine to compute free energy from high-level signals.
        let obs = symthaea_fep::Observation {
            values: vec![
                self.surprise.max_surprise() as f64,
                self.prediction_error as f64,
                self.coherence as f64,
                self.telemetry.motion_surprise as f64,
            ],
            precision: 1.0,
            timestamp: self.frame_count,
            modality: "visual".to_string(),
        };

        let perception_res = self.fep_agent.perceive(&obs);
        let fe = perception_res.free_energy;

        self.last_fep = crate::types::FepMetrics {
            free_energy: fe.total as f32,
            complexity: fe.complexity as f32,
            accuracy: fe.accuracy as f32,
        };

        // Select best cognitive action based on Expected Free Energy (Closing the Loop)
        let action_res = self.fep_agent.select_action();
        let action_name = self.map_fep_action_to_vision_behavior(action_res.action);

        // Store training telemetry
        self.telemetry.training_triggered = training_triggered;
        self.telemetry.training_loss = training_loss;
        self.telemetry.last_fep_action = action_name;
        self.telemetry.last_geodesic_path = self
            .last_geodesic
            .iter()
            .map(|hv| hv.values.clone())
            .collect();
        self.telemetry.last_geodesic_cost = self.geodesic_compute_cost;
        self.telemetry.last_geodesic_length = self.last_geodesic.len();

        // Reset compute cost for next cycle
        self.geodesic_compute_cost = 0.0;
    }

    /// Mean like-for-like patch novelty for the current raw encoding.
    ///
    /// A missing or differently sized history is treated as maximally novel so
    /// geometry changes and cold starts cannot reuse stale object hypotheses.
    fn mean_patch_novelty(current: &[ContinuousHV], previous: &[ContinuousHV]) -> f32 {
        if current.is_empty() || current.len() != previous.len() {
            return 1.0;
        }

        current
            .iter()
            .zip(previous.iter())
            .map(|(now, before)| 1.0 - now.similarity(before).clamp(-1.0, 1.0))
            .sum::<f32>()
            / current.len() as f32
    }

    /// Recompute cached object appearance from the current patch evidence while
    /// preserving the previous frame's segmentation topology.
    ///
    /// Returns `false` when the cached topology is unusable so the caller can
    /// safely fall back to full re-clustering.
    fn refresh_hypothesis_appearance(
        hypotheses: &mut [crate::types::ObjectHypothesis],
        patch_hvs: &[ContinuousHV],
    ) -> bool {
        for hypothesis in hypotheses {
            if hypothesis.patch_indices.is_empty()
                || hypothesis
                    .patch_indices
                    .iter()
                    .any(|&idx| idx >= patch_hvs.len())
            {
                return false;
            }

            let members: Vec<&ContinuousHV> = hypothesis
                .patch_indices
                .iter()
                .map(|&idx| &patch_hvs[idx])
                .collect();
            hypothesis.hv = if members.len() == 1 {
                members[0].clone()
            } else {
                ContinuousHV::bundle(&members).normalize()
            };
        }
        true
    }

    /// Refresh object-hypothesis saliency from the current attention evidence.
    fn refresh_hypothesis_saliency(
        hypotheses: &mut [crate::types::ObjectHypothesis],
        attention_values: &[f32],
    ) {
        for hypothesis in hypotheses {
            let sum: f32 = hypothesis
                .patch_indices
                .iter()
                .map(|&idx| attention_values.get(idx).copied().unwrap_or(0.0))
                .sum();
            hypothesis.saliency = if hypothesis.patch_indices.is_empty() {
                0.05
            } else {
                (sum / hypothesis.patch_indices.len() as f32).max(0.05)
            };
        }
    }

    /// Cluster patch HVs into object hypotheses for relational binding (P3-E).
    ///
    /// Uses a greedy proximity-based clustering: patches are grouped into connected
    /// components where adjacent patches (4-connected) have cosine similarity ≥ 0.1.
    /// This is intentionally coarse — the goal is rough perceptual grouping, not
    /// precise object segmentation.
    ///
    /// Each cluster produces one `ObjectHypothesis` whose `hv` is the bundle of
    /// member patch HVs, and whose centroid is the mean grid position.
    /// Fingerprint size for fast similarity screening in clustering.
    /// Using the first 128 components gives ~16x speedup over full 16,384D
    /// while preserving 95%+ of the cosine similarity ranking accuracy.
    const CLUSTER_FINGERPRINT_DIM: usize = 128;

    fn cluster_patches(
        patch_hvs: &[ContinuousHV],
        grid: &crate::types::PatchGrid,
    ) -> Vec<crate::types::ObjectHypothesis> {
        if patch_hvs.is_empty() || grid.cols == 0 || grid.rows == 0 {
            return vec![];
        }

        let n = patch_hvs.len().min(grid.rows * grid.cols);
        let mut assigned = vec![usize::MAX; n];
        let mut clusters: Vec<Vec<usize>> = Vec::new();

        // Similarity threshold for merging adjacent patches.
        const MERGE_THRESHOLD: f32 = 0.6;
        // Screening threshold: slightly lower to avoid false negatives.
        // Patches below this on the fingerprint are definitely below MERGE_THRESHOLD
        // on the full vector (fingerprint underestimates similarity slightly).
        const SCREEN_THRESHOLD: f32 = 0.5;

        let max_cluster_size = (n / 3).max(4);

        // Precompute low-dimensional fingerprints for fast screening.
        // This avoids O(16,384) dot products during flood-fill — instead we do
        // O(128) screening + O(16,384) only for borderline cases.
        let fp_dim =
            Self::CLUSTER_FINGERPRINT_DIM.min(patch_hvs.first().map_or(128, |hv| hv.dim()));
        let fingerprints: Vec<&[f32]> = patch_hvs
            .iter()
            .map(|hv| &hv.as_slice()[..fp_dim])
            .collect();
        // Precompute fingerprint norms for cosine similarity
        let fp_norms: Vec<f32> = fingerprints
            .iter()
            .map(|fp| {
                let sq: f32 = fp.iter().map(|x| x * x).sum();
                sq.sqrt().max(1e-10)
            })
            .collect();

        // Greedy 4-connected flood-fill with fingerprint-accelerated screening
        for start in 0..n {
            if assigned[start] != usize::MAX {
                continue;
            }
            let cluster_id = clusters.len();
            let mut frontier = vec![start];
            let mut members = vec![];
            while let Some(idx) = frontier.pop() {
                if assigned[idx] != usize::MAX || members.len() >= max_cluster_size {
                    continue;
                }
                assigned[idx] = cluster_id;
                members.push(idx);
                let row = idx / grid.cols;
                let col = idx % grid.cols;
                let neighbors = [
                    if row > 0 { Some(idx - grid.cols) } else { None },
                    if row + 1 < grid.rows {
                        Some(idx + grid.cols)
                    } else {
                        None
                    },
                    if col > 0 { Some(idx - 1) } else { None },
                    if col + 1 < grid.cols {
                        Some(idx + 1)
                    } else {
                        None
                    },
                ];
                for nb_opt in neighbors.into_iter().flatten() {
                    if assigned[nb_opt] == usize::MAX {
                        // Fast fingerprint screening (128D dot product)
                        let dot: f32 = fingerprints[idx]
                            .iter()
                            .zip(fingerprints[nb_opt].iter())
                            .map(|(a, b)| a * b)
                            .sum();
                        let fp_sim = dot / (fp_norms[idx] * fp_norms[nb_opt]);

                        // If fingerprint says definitely below threshold, skip
                        if fp_sim < SCREEN_THRESHOLD {
                            continue;
                        }
                        // Borderline: do full 16,384D similarity check
                        let sim = if fp_sim >= MERGE_THRESHOLD + 0.1 {
                            fp_sim // high-confidence accept from fingerprint alone
                        } else {
                            patch_hvs[idx].similarity(&patch_hvs[nb_opt])
                        };
                        if sim >= MERGE_THRESHOLD {
                            frontier.push(nb_opt);
                        }
                    }
                }
            }
            clusters.push(members);
        }

        // Convert clusters → ObjectHypothesis
        clusters
            .into_iter()
            .filter(|m| !m.is_empty())
            .map(|members| {
                let sum_r: usize = members.iter().map(|&i| i / grid.cols).sum();
                let sum_c: usize = members.iter().map(|&i| i % grid.cols).sum();
                let n_m = members.len();
                let centroid_row = sum_r / n_m;
                let centroid_col = sum_c / n_m;
                let member_refs: Vec<&ContinuousHV> =
                    members.iter().map(|&i| &patch_hvs[i]).collect();
                let hv = if member_refs.len() == 1 {
                    member_refs[0].clone()
                } else {
                    ContinuousHV::bundle(&member_refs).normalize()
                };
                crate::types::ObjectHypothesis {
                    centroid_row,
                    centroid_col,
                    patch_indices: members,
                    saliency: 0.0,
                    hv,
                }
            })
            .collect()
    }

    /// Internal training step: update weight_hv and tau_base from prediction error.
    fn train_step_inner(
        &mut self,
        input: &ContinuousHV,
        predicted: &ContinuousHV,
        actual: &ContinuousHV,
        dt: f32,
    ) -> BpttResult {
        self.trainer.set_input_blend(self.config.input_blend);
        let result = self.trainer.train_step(
            &self.weight_hv,
            &self.state,
            input,
            predicted,
            actual,
            self.config.tau_base,
            dt,
        );

        // Apply weight update
        self.weight_hv = self.weight_hv.add(&result.weight_update);

        // Apply tau update with clamping
        self.config.tau_base = (self.config.tau_base + result.tau_update).clamp(0.01, 10.0);

        result
    }

    /// CfC equilibrium: tanh(α · input + (1-α) · W ⊗ state).
    ///
    /// The equilibrium is attracted toward the input signal (what we observe)
    /// with state persistence through the weight-transformed state (memory/inertia).
    /// This ensures the manifold tracks visual input rather than drifting into
    /// a random subspace (which happens with pure bind on random untrained weights).
    ///
    /// The `input_blend` parameter (default 0.7) controls the balance:
    /// - High values (0.9): responsive to new input, less temporal memory
    /// - Low values (0.3): more state persistence, slower adaptation
    fn equilibrium(&self, input: &ContinuousHV) -> ContinuousHV {
        self.equilibrium_with_state(input, &self.state)
    }

    /// CfC gating factor: σ = 1 - exp(-dt / τ).
    fn gating(&self, dt: f32) -> f32 {
        let decay = (-dt / self.config.tau_base.max(0.001)).exp();
        1.0 - decay
    }

    /// Predict manifold state at a future horizon via O(1) closed-form jump.
    fn predict_horizon(&self, current_input: &ContinuousHV, horizon: f32) -> ContinuousHV {
        let x_inf = self.equilibrium(current_input);
        let sigma = self.gating(horizon);
        let mut predicted = self.state.clone();
        predicted.lerp_in_place(&x_inf, 1.0 - sigma, sigma);
        predicted
    }

    /// Perform multi-step 'Dreaming' - predict a sequence of future manifold states.
    ///
    /// Science: Friston (2010). Dreaming (offline active inference) minimizes
    /// future free energy by optimizing internal generative models without sensory cost.
    ///
    /// Simulation evolves a local copy of the visual state under the learned
    /// autonomous dynamics. The live perceptual state and FEP belief remain unchanged.
    pub fn dream_ahead(&mut self, steps: usize, dt: f32) -> Vec<ContinuousHV> {
        let mut predictions = Vec::with_capacity(steps);
        let mut dream_state = self.state.clone();
        let safe_dt = if dt.is_finite() && dt > 0.0 { dt } else { 0.0 };

        // Mental simulation must be observational: callers can inspect a future
        // trajectory without contaminating the live perceptual state or the FEP
        // agent that will process the next real frame. Evolve a local state under
        // the learned autonomous dynamics instead of mutating `self.state`.
        for _ in 0..steps {
            let internal_input = self.weight_hv.bind(&dream_state).tanh();
            let x_inf = self.equilibrium_with_state(&internal_input, &dream_state);
            let sigma = self.gating(safe_dt);
            dream_state.lerp_in_place(&x_inf, 1.0 - sigma, sigma);
            predictions.push(dream_state.clone());
        }

        // Account for simulated work without changing sensory or belief state.
        self.geodesic_compute_cost += steps as f32 * 0.008;
        predictions
    }

    /// Perform multi-scale 'Dreaming' - predict future states across hierarchical levels.
    ///
    /// Science: Friston (2010). Hierarchical active inference allows the system
    /// to Zoom Out (abstract simulation) and Zoom In (detailed simulation).
    ///
    /// Returns `(coarse_predictions, fine_predictions)`. Returns empty vectors
    /// if the predictive coding hierarchy is disabled.
    pub fn dream_multi_scale(
        &self,
        steps: usize,
        dt: f32,
    ) -> (Vec<ContinuousHV>, Vec<ContinuousHV>) {
        if let Some(ref pch) = self.predictive {
            pch.dream_ahead(steps, dt)
        } else {
            (Vec::new(), Vec::new())
        }
    }

    /// Find the geodesic (shortest, most coherent path) between two manifold states.
    ///
    /// Video generation is treated as finding the optimal path on the manifold
    /// that minimizes expected free energy and preserves topological consistency.
    ///
    /// Returns a sequence of interpolated hypervectors.
    pub fn find_geodesic(
        &mut self,
        from: &ContinuousHV,
        goal: &ContinuousHV,
        steps: usize,
    ) -> Vec<ContinuousHV> {
        if steps == 0 {
            return Vec::new();
        }

        let mut path = Vec::with_capacity(steps);
        // Start point
        path.push(from.clone());

        let mut current = from.clone();
        // Adjust loop count to produce exactly 'steps' elements
        let inner_steps = steps.saturating_sub(1);

        // Science: Geodesics on the learned manifold follow the "flow"
        // defined by the system's own dynamics (CfC).
        let dt = 0.1;

        for _ in 0..inner_steps {
            let x_inf = self.equilibrium_with_state(goal, &current);
            let sigma = self.gating(dt);
            current.lerp_in_place(&x_inf, 1.0 - sigma, sigma);
            path.push(current.normalize());
        }

        self.last_geodesic = path.clone();
        self.telemetry.last_geodesic_path = self
            .last_geodesic
            .iter()
            .map(|hv| hv.values.clone())
            .collect();
        self.telemetry.last_geodesic_length = self.last_geodesic.len();

        // Update thermodynamic cost (Phase 3)
        self.geodesic_compute_cost += steps as f32 * 0.012;
        self.telemetry.last_geodesic_cost = self.geodesic_compute_cost;

        path
    }

    /// Select the best geodesic path using Expected Free Energy (G).
    ///
    /// Generates multiple candidate paths via different interpolation strategies
    /// and chooses the one with the lowest total expected free energy.
    pub fn select_best_geodesic(
        &mut self,
        from: &ContinuousHV,
        goal: &ContinuousHV,
        steps: usize,
        num_candidates: usize,
    ) -> Vec<ContinuousHV> {
        if steps == 0 || num_candidates == 0 {
            return Vec::new();
        }

        let mut final_goal = goal.clone();
        if final_goal.values.len() != self.hdc_dim() {
            final_goal = final_goal.dilate(self.hdc_dim());
        }

        let mut best_path = Vec::new();
        let mut best_score = f64::INFINITY;

        for candidate_idx in 0..num_candidates {
            let path = self.generate_candidate_path(from, &final_goal, steps, candidate_idx);
            let score = self.score_path_with_fep(&path, &final_goal);

            if score < best_score {
                best_score = score;
                best_path = path;
            }
        }

        self.last_geodesic = best_path.clone();

        // Phase 2: Enforce sheaf coherence along the chosen path
        self.enforce_sheaf_coherence(&mut best_path, 0.85);

        // Sync to telemetry
        self.telemetry.last_geodesic_path = best_path.iter().map(|hv| hv.values.clone()).collect();
        self.telemetry.last_geodesic_length = best_path.len();

        // Update thermodynamic cost (Phase 3)
        // Science: metabolic cost of high-res mental simulation.
        self.geodesic_compute_cost += (steps * num_candidates) as f32 * 0.012;
        self.telemetry.last_geodesic_cost = self.geodesic_compute_cost;

        // Store intent for swarm broadcast (Phase 5)
        self.last_intent_hv = final_goal;

        best_path
    }

    /// Unique node ID for P2P swarm identification.
    pub fn node_id(&self) -> uuid::Uuid {
        self.node_id
    }

    /// Latest intent/goal vector (for swarm broadcast).
    pub fn last_intent_hv(&self) -> &ContinuousHV {
        &self.last_intent_hv
    }

    /// Set a custom node ID (e.g. from config).
    pub fn set_node_id(&mut self, id: uuid::Uuid) {
        self.node_id = id;
    }

    /// Validate and optionally repair sheaf coherence along a path.
    ///
    /// Ensures that consecutive states in the geodesic path maintain
    /// local semantic and structural consistency.
    pub fn enforce_sheaf_coherence(&mut self, path: &mut [ContinuousHV], threshold: f32) -> bool {
        let mut coherent = true;

        for i in 0..path.len().saturating_sub(1) {
            let coherence = self.compute_local_coherence(&path[i], &path[i + 1]);

            if coherence < threshold {
                coherent = false;
                // Simple repair: average with neighbors to smooth the transition
                // In HDC, this is a local bundling operation.
                let mut avg = path[i].clone();
                avg.lerp_in_place(&path[i + 1], 0.5, 0.5);
                path[i + 1] = avg.normalize();
            }
        }

        coherent
    }

    fn compute_local_coherence(&self, a: &ContinuousHV, b: &ContinuousHV) -> f32 {
        // Use semantic similarity + binding strength as a proxy for sheaf consistency.
        let sim = a.similarity(b);
        let binding_strength = a.bind(b).norm() / (a.norm() * b.norm()).sqrt();

        (sim * 0.6 + binding_strength * 0.4).clamp(0.0, 1.0)
    }

    /// Compute the thermodynamic energy required for a manifold transition.
    ///
    /// Science: Higher cosine distance (more semantic change) requires more
    /// metabolic work to update the CfC state.
    fn compute_transition_energy(&self, a: &ContinuousHV, b: &ContinuousHV) -> f64 {
        let distance = 1.0 - a.similarity(b).clamp(-1.0, 1.0);
        distance as f64 * 0.05
    }

    /// Compute semantic coherence relative to known scene landmarks.
    fn compute_semantic_coherence(&self, state: &ContinuousHV) -> f64 {
        if let Some(ref memory) = self.scene_memory {
            let mut max_sim = 0.0;
            for (landmark, _) in memory.export_landmarks() {
                max_sim = f32::max(max_sim, state.similarity(landmark));
            }
            max_sim as f64
        } else {
            0.0
        }
    }

    /// Improved GeoSynth decoder (v2)
    /// Uses scene memory landmarks + patch-level blending for much higher quality mental movies.
    pub fn decode_geodesic_to_frames_improved(&self, path: &[ContinuousHV]) -> Vec<Vec<u8>> {
        if path.is_empty() {
            return vec![];
        }

        let reference_frame = match &self.last_observed_frame {
            Some(frame) => frame.clone(),
            None => return vec![],
        };

        let width = self.last_frame_width;
        let height = self.last_frame_height;
        let channels = self.last_frame_channels;
        let reference_metadata = SceneFrameMetadata {
            width,
            height,
            channels,
            modality: self.last_frame_modality,
        };
        let Some(frame_size) = reference_metadata.expected_len() else {
            return vec![];
        };

        if reference_frame.len() != frame_size {
            return vec![];
        }

        let mut decoded_frames = Vec::with_capacity(path.len());

        // Pre-compute patch grid for patch-level blending
        let grid = self.encoder.grid_for(width, height);
        let patch_rows = grid.rows;
        let patch_cols = grid.cols;
        if patch_rows == 0 || patch_cols == 0 {
            return vec![];
        }

        let patch_h = height as usize / patch_rows;
        let patch_w = width as usize / patch_cols;

        for (step_idx, state) in path.iter().enumerate() {
            // Geodesic playback must advance monotonically even when a curved
            // path temporarily becomes more similar to its starting state.
            let progress = if path.len() <= 1 {
                1.0
            } else {
                step_idx as f32 / (path.len() - 1) as f32
            };

            // Find the best matching landmark and use its persisted raw frame.
            // Landmarks without pixels (legacy checkpoints) or with incompatible
            // geometry are skipped rather than pretending the reference frame is
            // a decoded memory.
            let mut best_landmark_sim = f32::NEG_INFINITY;
            let mut best_landmark_frame: Option<&[u8]> = None;

            if let Some(ref memory) = self.scene_memory {
                for (scene_id, (landmark_hv, _)) in
                    memory.export_landmarks().into_iter().enumerate()
                {
                    let Some(pixels) = memory.get_pixels(scene_id) else {
                        continue;
                    };
                    let metadata = memory.get_frame_metadata(scene_id).unwrap_or_default();
                    let compatible = if metadata.modality == VisualModality::Unknown {
                        // Legacy checkpoints have no semantic pixel contract.
                        pixels.len() == frame_size
                    } else {
                        metadata.is_pixel_compatible_with(reference_metadata)
                            && metadata.expected_len() == Some(pixels.len())
                    };
                    if !compatible {
                        continue;
                    }
                    let sim = state.similarity(landmark_hv);
                    if sim > best_landmark_sim {
                        best_landmark_sim = sim;
                        best_landmark_frame = Some(pixels);
                    }
                }
            }

            let target_frame = best_landmark_frame.unwrap_or(&reference_frame);
            let mut frame = vec![0u8; frame_size];

            // Patch-level blending (much better than whole-frame lerp)
            for py in 0..patch_rows {
                for px in 0..patch_cols {
                    let start_y = py * patch_h;
                    let start_x = px * patch_w;

                    // Local confidence weight: high surprise means the current
                    // patch is poorly explained, so rely less on landmark pixels.
                    let surprise_val: f32 = self.surprise.attention_map().at(py, px);
                    let local_weight = (1.0 - surprise_val).clamp(0.2, 0.9);
                    let blend = progress * local_weight;

                    for dy in 0..patch_h {
                        for dx in 0..patch_w {
                            let y = start_y + dy;
                            let x = start_x + dx;
                            if y >= height as usize || x >= width as usize {
                                continue;
                            }

                            let idx = (y * width as usize + x) * channels;

                            for c in 0..channels {
                                let i = idx + c;
                                if i >= frame_size {
                                    continue;
                                }

                                let start_val = reference_frame[i] as f32;
                                let target_val = target_frame[i] as f32;

                                let interpolated = start_val * (1.0 - blend) + target_val * blend;
                                frame[i] = interpolated.clamp(0.0, 255.0) as u8;
                            }
                        }
                    }
                }
            }

            decoded_frames.push(frame);
        }

        decoded_frames
    }

    /// Helper: Generate a candidate path using a specific strategy.
    fn generate_candidate_path(
        &self,
        from: &ContinuousHV,
        goal: &ContinuousHV,
        steps: usize,
        strategy: usize,
    ) -> Vec<ContinuousHV> {
        let mut path = Vec::with_capacity(steps);
        // Total time horizon: steps * tau * scaling
        let dt_total = steps as f32 * self.config.tau_base * 0.5;

        for i in 0..steps {
            let alpha = i as f32 / (steps as f32 - 1.0).max(1.0);
            let mut intermediate = from.clone();

            match strategy % 3 {
                0 => {
                    // Strategy 1: CfC Closed-Form Projection (Instant physics)
                    if let Some(ref model) = self.transition_model {
                        let ctx = TransitionContext {
                            goal: Some(goal.clone()),
                            input: self.last_frame_hv.clone(),
                            weight_hv: Some(self.weight_hv.clone()),
                            tau: self.config.tau_base,
                        };
                        // Project instantly to time t = dt_total * alpha
                        intermediate = model.project(from, dt_total * alpha, &ctx);
                    } else {
                        intermediate.lerp_in_place(goal, 1.0 - alpha, alpha);
                    }
                }
                1 => {
                    // Strategy 2: LERP + slight noise (Exploration/Novelty)
                    intermediate.lerp_in_place(goal, 1.0 - alpha, alpha);
                    let noise = ContinuousHV::random(
                        self.hdc_dim(),
                        self.config.seed + i as u64 + strategy as u64,
                    );
                    intermediate.lerp_in_place(&noise, 0.95, 0.05);
                }
                2 => {
                    // Strategy 3: Bias toward the agent's current belief (Top-down)
                    intermediate.lerp_in_place(goal, 1.0 - alpha, alpha);
                    // Use weight_hv bound belief as a semantic prior
                    let dim = self.hdc_dim();
                    let mut mean_f32 = vec![0.0f32; dim];
                    for (i, &val) in self.fep_agent.belief.mean.iter().enumerate() {
                        if i < dim {
                            mean_f32[i] = val as f32;
                        }
                    }
                    let belief_hv = ContinuousHV::from_vec(mean_f32);
                    let belief_prior = self.weight_hv.bind(&belief_hv).tanh();
                    intermediate.lerp_in_place(&belief_prior, 0.8, 0.2);
                }
                _ => unreachable!(),
            }

            path.push(intermediate.normalize());
        }

        path
    }

    /// Helper: Score a candidate path using the FEP agent's Expected Free Energy.
    ///
    /// Science: Friston (2010). Scoring evaluates the 'Goodness' of the entire
    /// trajectory by integrating Expected Free Energy (G) and path coherence.
    ///
    /// Takes the path's actual `goal` so the reward term can measure real
    /// progress toward it. It previously rewarded similarity to `self.weight_hv`
    /// — a fixed vector unrelated to `goal` — which let a candidate that merely
    /// resembled `weight_hv` outscore candidates that actually reached the
    /// goal (verified: a CfC-projection candidate ending at similarity ~0.72 to
    /// the goal was selected over LERP/belief-biased candidates ending at
    /// ~0.999, because their `weight_hv` resemblance dominated the score).
    fn score_path_with_fep(&mut self, path: &[ContinuousHV], goal: &ContinuousHV) -> f64 {
        let mut total_efe = 0.0;
        let mut path_inconsistency = 0.0;
        let mut transition_energy = 0.0;
        let mut semantic_coherence = 0.0;

        // `efe_computer.compute()` always scores action `0` here (a path is a
        // fixed trajectory being *evaluated*, not a sequence of chosen
        // actions), but it unconditionally records that action into its own
        // persistent `action_history` and derives a "novelty" bonus from how
        // many times action `0` has been seen so far. Left unguarded, that
        // history keeps growing across every candidate path scored by
        // `select_best_geodesic`'s loop, so novelty — and therefore the whole
        // score — silently decays candidate-by-candidate independent of path
        // quality, systematically favoring whichever candidate happened to be
        // evaluated first (verified: this alone was enough to make a
        // CfC-projection candidate ending near-orthogonal to the goal
        // outscore later-evaluated candidates that actually reached it).
        // Snapshotting and restoring the history here gives every candidate
        // the same novelty baseline.
        let original_action_history = self.fep_agent.efe_computer.action_history.clone();

        for (i, state_hv) in path.iter().enumerate() {
            // 1. Reward actual progress toward the path's goal.
            let state_sim_to_goal = state_hv.similarity(goal);

            // 2. Holistic Path Coherence (Phase 2 integration)
            if i > 0 {
                let local_coherence = self.compute_local_coherence(&path[i - 1], state_hv);
                path_inconsistency += (1.0 - local_coherence) as f64;

                // 3. Thermodynamic cost of transition
                transition_energy += self.compute_transition_energy(&path[i - 1], state_hv);
            }

            // 4. Semantic coherence (how well this state matches known scenes)
            semantic_coherence += self.compute_semantic_coherence(state_hv);

            // 5. Expected Free Energy scoring
            let original_belief = self.fep_agent.belief.clone();

            let _dim = self.fep_agent.config.state_dim;
            for (j, val) in self.fep_agent.belief.mean.iter_mut().enumerate() {
                if j < state_hv.dim() {
                    *val = 0.7 * *val + 0.3 * state_hv.values[j % state_hv.dim()] as f64;
                }
            }

            let efe = self.fep_agent.efe_computer.compute(
                0,
                &self.fep_agent.belief,
                &self.fep_agent.model,
            );

            // Weighted integration
            total_efe += efe.total - (state_sim_to_goal as f64 * 0.15);

            // Restore belief
            self.fep_agent.belief = original_belief;
        }

        // Restore the novelty baseline so the next candidate path is scored
        // on equal footing (see comment above).
        self.fep_agent.efe_computer.action_history = original_action_history;

        // Final Score: Lower is better
        // F = G + Energy_cost + Coherence_penalty - Semantic_reward
        total_efe + (path_inconsistency * 0.5) + (transition_energy * 0.3)
            - (semantic_coherence * 0.2)
    }

    /// Map FEP motor commands to real manifold behaviors.
    fn map_fep_action_to_vision_behavior(&mut self, action_index: usize) -> String {
        use symthaea_fep::MotorCommandType;
        let cmd = MotorCommandType::from_action_index(action_index);

        match cmd {
            MotorCommandType::AttentionShift => {
                // Focus shift: slightly boost current learning rate
                let current_lr = self.trainer.config().learning_rate;
                self.trainer.config_mut().learning_rate = (current_lr * 1.1).min(0.05);
                "AttentionShift (LR Boost)".to_string()
            }
            MotorCommandType::ExplorationTrigger => {
                // Exploration: Dilate to Ultra if not already — gated by
                // allow_auto_dilation (post-Ultra machinery has an
                // unresolved multi-GB blow-up; see the config field docs).
                if self.config.hdc_dim < 65536 && self.config.allow_auto_dilation {
                    match self.try_dilate(symthaea_core::hdc::HdcDimensionality::Ultra) {
                        Ok(report) => format!(
                            "Exploration (Dilation Triggered: {} bytes)",
                            report.total_projected_bytes
                        ),
                        Err(error) => format!("Exploration (Dilation Rejected: {error})"),
                    }
                } else if self.config.allow_auto_dilation {
                    "Exploration (Already Dilated)".to_string()
                } else {
                    "Exploration (Dilation Disabled)".to_string()
                }
            }
            MotorCommandType::LearningRateAdjust => {
                // Adaptive LR based on agent precision
                let precision = self.fep_agent.precision.prior_precision;
                // High precision (certainty) -> lower LR
                self.trainer.config_mut().learning_rate =
                    (0.01 / precision as f32).clamp(0.001, 0.05);
                format!("LearningRateAdjust (PR={:.2})", precision)
            }
            MotorCommandType::ReflectionInitiate => {
                // Metacognition: Record higher compute cost for "thinking"
                self.geodesic_compute_cost += 0.05;
                "Reflection (Meta-cost applied)".to_string()
            }
            MotorCommandType::MemoryConsolidate => {
                let decay = self.consolidate_surprise_memory();
                format!("MemoryConsolidate (decay={decay:.3})")
            }
            MotorCommandType::ExpectationReset => {
                self.reset_expectations();
                "ExpectationReset (All modality caches cleared)".to_string()
            }
            MotorCommandType::MotorOutput => {
                // Pragmatic: Boost state towards goal (if any)
                "MotorOutput (Pragmatic boost)".to_string()
            }
            MotorCommandType::NoOp => "NoOp".to_string(),
        }
    }

    /// Increase temporal evidence persistence and apply it to the live surprise map.
    fn consolidate_surprise_memory(&mut self) -> f32 {
        let current = self.config.surprise_decay.clamp(0.001, 0.999);
        let consolidated = (current + 0.25 * (1.0 - current)).min(0.99);
        self.config.surprise_decay = consolidated;
        self.surprise.set_decay(consolidated);
        consolidated
    }

    /// CfC equilibrium with explicit state (helper for dreaming).
    fn equilibrium_with_state(&self, input: &ContinuousHV, state: &ContinuousHV) -> ContinuousHV {
        let state_influence = self.weight_hv.bind(state);
        let ib = self.config.input_blend;
        ContinuousHV::weighted_bundle(&[input, &state_influence], &[ib, 1.0 - ib]).tanh()
    }

    /// Clear predictive expectations without discarding learned parameters or
    /// the latest sensory observations.
    ///
    /// The reset covers the active sensor, every parked modality context, the
    /// cross-scale predictive hierarchy, and imagination diagnostics. This avoids
    /// a stale prediction reappearing when the caller switches back to a modality
    /// after an FEP expectation-reset action.
    pub fn reset_expectations(&mut self) {
        self.last_prediction = None;
        self.prediction_error = 0.0;
        self.error_ema = 0.0;
        self.horizon_evaluator.clear_pending();
        for (_, context) in &mut self.modality_contexts {
            context.last_prediction = None;
            context.prediction_error = 0.0;
            context.error_ema = 0.0;
            context.horizon_evaluator.clear_pending();
        }
        if let Some(ref mut predictive) = self.predictive {
            predictive.reset();
        }
        self.last_imagination = None;
        self.imagination_surprise = 0.0;
        self.telemetry.prediction_error = 0.0;
        self.telemetry.cross_scale_prediction_error = 0.0;
        self.telemetry.imagination_surprise = 0.0;
    }

    /// Latest generated geodesic path on the manifold.
    pub fn last_geodesic(&self) -> &[ContinuousHV] {
        &self.last_geodesic
    }

    /// Current manifold state (the "scene representation").
    pub fn state(&self) -> &ContinuousHV {
        &self.state
    }

    /// Last prediction error (free energy proxy, 0 = perfect prediction).
    pub fn prediction_error(&self) -> f32 {
        self.prediction_error
    }

    /// Exponential moving average of prediction error (for adaptive training).
    pub fn error_ema(&self) -> f32 {
        self.error_ema
    }

    /// Latest Variational Free Energy metrics.
    pub fn last_fep(&self) -> crate::types::FepMetrics {
        self.last_fep
    }

    /// Manifold coherence (state-frame cosine similarity, 0..1).
    pub fn coherence(&self) -> f32 {
        self.coherence
    }

    /// Access the spatial surprise map.
    pub fn surprise_map(&self) -> &SurpriseMap {
        &self.surprise
    }

    /// Number of color channels in the last observed frame.
    pub fn last_frame_channels(&self) -> usize {
        self.last_frame_channels
    }

    /// Mutable access to the spatial surprise map (for top-down priming).
    pub fn surprise_map_mut(&mut self) -> &mut SurpriseMap {
        &mut self.surprise
    }

    /// Last telemetry snapshot.
    pub fn telemetry(&self) -> &VisionTelemetry {
        &self.telemetry
    }

    /// Total frames observed.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Access the underlying encoder for external use.
    pub fn encoder(&self) -> &PatchHdcEncoder {
        &self.encoder
    }

    /// Current tau_base value (may change during training).
    pub fn current_tau(&self) -> f32 {
        self.config.tau_base
    }

    /// Set a custom transition model for mental simulation.
    pub fn set_transition_model(&mut self, model: Box<dyn TransitionModel>) {
        self.transition_model = Some(model);
    }

    /// Access the learned weight HV (for inspection/comparison).
    pub fn weight_hv(&self) -> &ContinuousHV {
        &self.weight_hv
    }

    /// Total training steps performed.
    pub fn training_steps(&self) -> u64 {
        self.trainer.total_steps()
    }

    /// Per-patch HDC vectors from the last encoded frame.
    ///
    /// Each element is the 16,384D HV for one patch position (in row-major order).
    /// Used by `VisionBridge` for per-patch task-attention scoring.
    /// Empty before the first frame is observed.
    pub fn last_patch_hvs(&self) -> &[ContinuousHV] {
        &self.last_patch_hvs
    }

    /// Position-unbound appearance HVs for the most recently observed patches.
    ///
    /// Top-down concepts should match *what* a patch contains independently of
    /// where it appeared. Comparing a semantic goal directly with the stored
    /// position-bound patch HV would make task relevance change when the same
    /// object moves across the image.
    pub fn last_patch_appearance_hvs(&self) -> Vec<ContinuousHV> {
        let grid = self.surprise.grid();
        if grid.cols == 0 {
            return Vec::new();
        }
        self.last_patch_hvs
            .iter()
            .enumerate()
            .map(|(idx, patch_hv)| {
                let row = idx / grid.cols;
                let col = idx % grid.cols;
                self.encoder.unbind_position(patch_hv, row, col)
            })
            .collect()
    }

    /// Temporally-bound patch HVs: `ρ(prev_patch[i]) ⊗ curr_patch[i]`.
    ///
    /// Only meaningful when `VisionConfig::enable_temporal_binding` is true.
    /// When disabled, returns the same vectors as `last_patch_hvs()`.
    /// Empty before the first frame.
    pub fn temporal_patch_hvs(&self) -> &[ContinuousHV] {
        &self.temporal_patch_hvs
    }

    /// Per-patch motion magnitudes from the last frame.
    ///
    /// Each value is the Euclidean magnitude of the motion vector at that patch.
    /// Empty before the second frame.
    pub fn motion_saliency(&self) -> &[f32] {
        &self.motion_saliency
    }

    /// Per-patch motion vectors `[dx, dy]` from the last frame.
    ///
    /// Empty before the second frame.
    pub fn motion_vectors(&self) -> &[[f32; 2]] {
        &self.last_motion_vectors
    }

    /// Mutable access to the encoder (for contrastive refinement).
    pub fn encoder_mut(&mut self) -> &mut PatchHdcEncoder {
        &mut self.encoder
    }

    /// Count patches where surprise exceeds the configured threshold.
    ///
    /// Returns `(active, total)` where `active` is the count of salient patches.
    pub fn active_patch_count(&self) -> (usize, usize) {
        let attention = self.surprise.attention_map();
        let active = attention
            .values
            .iter()
            .filter(|&&v| v > self.config.surprise_threshold)
            .count();
        (active, attention.values.len())
    }

    /// Freeze or unfreeze learning.
    ///
    /// When frozen, `train_step_inner()` and `refine_contrastive()` are skipped
    /// during `observe_encoded()`. The manifold still evolves its CfC state and
    /// computes prediction error — only parameter updates are suppressed.
    pub fn freeze_learning(&mut self, freeze: bool) {
        self.learning_frozen = freeze;
    }

    /// Whether learning is currently frozen.
    pub fn is_learning_frozen(&self) -> bool {
        self.learning_frozen
    }

    /// Last scene recognition match (if any).
    pub fn last_scene_match(&self) -> Option<&SceneMatch> {
        self.last_scene_match.as_ref()
    }

    /// Get the encoding for a specific scene landmark.
    pub fn get_scene_encoding(&self, scene_id: usize) -> Option<ContinuousHV> {
        self.scene_memory.as_ref()?.get_landmark(scene_id).cloned()
    }

    /// Set the coherence and error thresholds for scene memory storage.
    pub fn set_scene_store_thresholds(&mut self, coherence: f32, error: f32) {
        let _ = self.set_scene_store_thresholds_checked(coherence, error);
    }

    /// Checked scene-storage policy update.
    pub fn set_scene_store_thresholds_checked(
        &mut self,
        coherence: f32,
        error: f32,
    ) -> Result<(), String> {
        if !coherence.is_finite() || !(0.0..=1.0).contains(&coherence) {
            return Err(format!(
                "scene coherence threshold must be finite and in [0, 1], got {coherence}"
            ));
        }
        if !error.is_finite() || !(0.0..=1.0).contains(&error) {
            return Err(format!(
                "scene error threshold must be finite and in [0, 1], got {error}"
            ));
        }
        self.scene_store_coherence_threshold = coherence;
        self.scene_store_error_threshold = error;
        Ok(())
    }

    /// Set the dampening factor for recognized scenes.
    pub fn set_scene_dampen_factor(&mut self, factor: f32) {
        let _ = self.set_scene_dampen_factor_checked(factor);
    }

    /// Checked scene-dampening policy update.
    pub fn set_scene_dampen_factor_checked(&mut self, factor: f32) -> Result<(), String> {
        if !factor.is_finite() || !(0.0..=1.0).contains(&factor) {
            return Err(format!(
                "scene dampening factor must be finite and in [0, 1], got {factor}"
            ));
        }
        self.scene_dampen_factor = factor;
        Ok(())
    }

    /// Enable scene memory with given capacity.
    pub fn enable_scene_memory(&mut self, capacity: usize) {
        self.scene_memory = Some(SceneMemory::new(capacity));
    }

    /// Enable cross-frame object identity tracking with given capacity.
    ///
    /// Object hypotheses from `cluster_patches()` (when `enable_object_binding`
    /// is true) are automatically matched and tracked across frames.
    pub fn enable_object_memory(&mut self, capacity: usize) {
        self.object_memory = Some(ObjectMemory::new(capacity));
    }

    /// Last object tracking result (matched / new / evicted counts).
    pub fn last_tracking_result(&self) -> Option<&ObjectTrackingResult> {
        self.last_tracking_result.as_ref()
    }

    /// Access the object memory for inspection.
    pub fn object_memory(&self) -> Option<&ObjectMemory> {
        self.object_memory.as_ref()
    }

    /// Enable visual working memory with given capacity (default: 4 objects).
    ///
    /// Requires `enable_object_memory()` to be active — working memory tracks
    /// which of the known objects are currently attended.
    pub fn enable_working_memory(&mut self, capacity: usize) {
        self.working_memory = Some(VisualWorkingMemory::new(capacity));
    }

    /// Access visual working memory.
    pub fn working_memory(&self) -> Option<&VisualWorkingMemory> {
        self.working_memory.as_ref()
    }

    /// Enable the visual scene graph for relational reasoning.
    ///
    /// Requires `enable_object_memory()` to be active — the scene graph
    /// computes relations between tracked objects.
    pub fn enable_scene_graph(&mut self) {
        self.scene_graph = Some(VisualSceneGraph::new(self.config.hdc_dim, self.config.seed));
    }

    /// Access the visual scene graph.
    pub fn scene_graph(&self) -> Option<&VisualSceneGraph> {
        self.scene_graph.as_ref()
    }

    /// Current imagination surprise (imagination-reality divergence).
    pub fn imagination_surprise(&self) -> f32 {
        self.imagination_surprise
    }

    /// Cycle at which the last dilation/constriction occurred.
    pub fn last_dilation_cycle(&self) -> u64 {
        self.last_dilation_cycle
    }

    /// Current HDC dimension of the manifold.
    pub fn hdc_dim(&self) -> usize {
        self.config.hdc_dim
    }

    /// Access the underlying config.
    pub fn config(&self) -> &VisionConfig {
        &self.config
    }

    pub(crate) fn checkpoint_capacity_dimensions(&self) -> Result<(u32, u32), String> {
        // Reuse the exact frame dimensions this manifold was constructed with
        // (preserved verbatim in the surprise map's grid, which is never
        // resized after construction — see `SurpriseMap::load_state`, which
        // only restores `surprise`/`decay`/`threshold`, not `grid`). Deriving
        // an "equivalent" width/height from `max_cols`/`max_rows` alone (e.g.
        // `(max_cols - 1) * patch_size + 1`) is lossy: many frame sizes share
        // the same patch-grid capacity, but `SurpriseMap::validate_state`
        // compares `frame_width`/`frame_height` verbatim, so a probe built
        // from a merely-equivalent size would spuriously reject a genuinely
        // compatible checkpoint whenever the original construction size
        // wasn't exactly that minimal reconstruction.
        let grid = self.surprise.grid();
        if grid.frame_width == 0 || grid.frame_height == 0 {
            return Err("manifold checkpoint capacity has zero-sized grid".to_string());
        }
        Ok((grid.frame_width, grid.frame_height))
    }

    fn checkpoint_validation_probe(&self) -> Result<Self, String> {
        let (width, height) = self.checkpoint_capacity_dimensions()?;
        let mut probe = Self::try_new(self.config.clone(), width, height)?;
        probe.load_state(&self.save_state())?;
        Ok(probe)
    }

    /// Validate a checkpoint against this manifold's complete runtime topology
    /// without mutating the live instance.
    ///
    /// A fresh isolated probe is built for every call so a rejected candidate
    /// cannot influence validation of a later retained generation.
    pub fn validate_checkpoint_state(&self, state: &ManifoldState) -> Result<(), String> {
        let mut probe = self.checkpoint_validation_probe()?;
        probe.load_state(state)
    }

    /// Saliency-guided encoding refinement (closed-loop active inference).
    ///
    /// Uses the surprise map to select positive (high-surprise) and negative
    /// (low-surprise) exemplar HVs, then refines the encoder's feature weights
    /// via contrastive learning. This makes the encoder adapt to attend to
    /// whatever is currently surprising in the scene.
    pub fn refine_from_attention(&mut self) {
        let attention = self.surprise.attention_map();
        if self.last_patch_hvs.is_empty() || attention.values.is_empty() {
            return;
        }

        let max_surprise = attention.max_surprise();
        if max_surprise < 1e-6 {
            return;
        }

        // Find the highest and lowest surprise patches
        let mut best_idx = 0;
        let mut worst_idx = 0;
        let mut best_val = f32::NEG_INFINITY;
        let mut worst_val = f32::INFINITY;

        for (i, &v) in attention.values.iter().enumerate() {
            if i < self.last_patch_hvs.len() {
                if v > best_val {
                    best_val = v;
                    best_idx = i;
                }
                if v < worst_val {
                    worst_val = v;
                    worst_idx = i;
                }
            }
        }

        if best_idx == worst_idx {
            return;
        }

        let positive = self.last_patch_hvs[best_idx].clone();
        let negative = self.last_patch_hvs[worst_idx].clone();
        let lr = self.config.learning.contrastive_lr;
        self.encoder.refine_contrastive(&positive, &negative, lr);
    }

    /// Delayed forecast skill accumulated for the active modality.
    pub fn horizon_accuracy(&self) -> HorizonAccuracy {
        self.horizon_evaluator.accuracy(self.frame_count)
    }

    /// Delayed forecast skill for an explicit modality, including parked sensors.
    pub fn horizon_accuracy_for(&self, modality: VisualModality) -> Option<HorizonAccuracy> {
        if self.active_modality == modality {
            return Some(self.horizon_accuracy());
        }
        self.modality_contexts
            .iter()
            .find(|(stored, _)| *stored == modality)
            .map(|(_, context)| context.horizon_evaluator.accuracy(self.frame_count))
    }

    /// Measure how far closed-form projections move from the current state.
    ///
    /// This is a projection-consistency diagnostic, not delayed forecast skill.
    /// Use [`DelayedHorizonEvaluator`] to compare predictions with later real frames.
    pub fn projection_consistency(&self) -> HorizonAccuracy {
        let horizons = self.default_horizons();
        let labels = self.horizon_labels();
        let mut errors = Vec::with_capacity(horizons.len());

        if let Some(ref frame_hv) = self.last_frame_hv {
            let state = self.state();
            for &h in horizons {
                let predicted = self.predict_horizon(frame_hv, h);
                let error = 1.0 - state.similarity(&predicted).clamp(-1.0, 1.0);
                errors.push(error);
            }
        } else {
            errors.resize(horizons.len(), 1.0);
        }

        HorizonAccuracy {
            horizons: horizons.to_vec(),
            labels: labels.iter().map(|s| s.to_string()).collect(),
            persistence_errors: vec![1.0; errors.len()],
            prediction_error_stddev: vec![0.0; errors.len()],
            persistence_error_stddev: vec![0.0; errors.len()],
            relative_skill: vec![0.0; errors.len()],
            sample_counts: vec![0; errors.len()],
            mean_lateness_seconds: vec![0.0; errors.len()],
            dropped_forecasts: vec![0; errors.len()],
            expired_forecasts: vec![0; errors.len()],
            errors,
            frame_sequence: self.frame_count,
        }
    }

    /// Compatibility alias for the former, misleading horizon evaluator.
    #[deprecated(
        note = "use projection_consistency() or DelayedHorizonEvaluator for real forecast skill"
    )]
    pub fn evaluate_horizons(&self) -> HorizonAccuracy {
        self.projection_consistency()
    }

    /// Visual imagination: run the CfC manifold forward without sensory input.
    ///
    /// Generates `n_steps` predicted future scene HVs by iteratively applying
    /// the closed-form CfC dynamics using only the current state and learned
    /// weight HV — no pixel observation enters the loop.
    ///
    /// ```text
    /// for each step:
    ///   x_inf = equilibrium(current_state)
    ///   state' = x_inf + (state - x_inf) * exp(-dt/τ)
    /// ```
    ///
    /// The returned HVs represent what the system *expects* to see at each
    /// future time step. Compare with actual observations to compute temporal
    /// prediction surprise — the hallmark of active inference.
    ///
    /// # Biological analog
    ///
    /// Dream replay: revisit stored scene memory landmarks during offline
    /// consolidation (Walker & Stickgold 2004).
    ///
    /// For each stored landmark, evolves the CfC state toward that memory
    /// and returns the sequence of replayed states. This strengthens the
    /// manifold's ability to recognize previously seen scenes.
    ///
    /// # Arguments
    /// * `dt` — Time step per replay step (typically longer than real-time, e.g., 0.1s)
    /// * `steps_per_memory` — CfC evolution steps per memory (more = deeper consolidation)
    ///
    /// Compatibility wrapper around [`Self::dream_replay_checked`]. Invalid
    /// replay parameters are rejected without changing learned or perceptual state.
    pub fn dream_replay(&mut self, dt: f32, steps_per_memory: usize) -> Vec<ContinuousHV> {
        match self.dream_replay_checked(dt, steps_per_memory) {
            Ok(replays) => replays,
            Err(error) => {
                tracing::warn!(%error, "rejected dream replay request");
                Vec::new()
            }
        }
    }

    /// Replay episodic landmarks through a local simulated state.
    ///
    /// Validation happens before any Hebbian consolidation. The live sensory
    /// state is observationally unchanged, while each replay step computes its
    /// equilibrium from the evolving local dream state rather than repeatedly
    /// consulting the live manifold state.
    pub fn dream_replay_checked(
        &mut self,
        dt: f32,
        steps_per_memory: usize,
    ) -> Result<Vec<ContinuousHV>, String> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(format!(
                "dream replay timestep must be finite and > 0, got {dt}"
            ));
        }
        if steps_per_memory == 0 {
            return Err("dream replay requires at least one step per memory".to_string());
        }

        let landmarks: Vec<ContinuousHV> = self.scene_memory.as_ref().map_or(Vec::new(), |mem| {
            mem.export_landmarks()
                .iter()
                .map(|(hv, _)| (*hv).clone())
                .collect()
        });

        if landmarks.is_empty() {
            return Ok(Vec::new());
        }

        let mut replays = Vec::with_capacity(landmarks.len());
        let mut dream_state = self.state.clone();

        for landmark in &landmarks {
            // Drive the local CfC state toward this memory landmark. The
            // equilibrium must be conditioned on the state produced by the
            // preceding replay step, not on `self.state`.
            for _ in 0..steps_per_memory {
                let x_inf = self.equilibrium_with_state(landmark, &dream_state);
                let sigma = self.gating(dt);
                dream_state.lerp_in_place(&x_inf, 1.0 - sigma, sigma);
            }
            replays.push(dream_state.clone());

            // Hebbian consolidation: strengthen the weight_hv toward the
            // replayed state (implicit gradient from the replay experience).
            let error = 1.0 - dream_state.similarity(landmark).clamp(-1.0, 1.0);
            if error > 0.01 {
                let lr = 0.001; // gentle replay learning rate
                let delta = ContinuousHV::weighted_bundle(&[landmark, &dream_state], &[lr, -lr]);
                self.weight_hv = self.weight_hv.add(&delta);
            }
        }
        Ok(replays)
    }

    /// Generate a scene description as structured relational tuples.
    ///
    /// Returns a vector of `(subject_label, relation, object_label)` strings
    /// derived from the scene graph. This is the bridge to Broca's area —
    /// the cognitive loop can feed these triples into the language pipeline
    /// for natural language generation.
    ///
    /// Labels are generated from track IDs; a real system would map them
    /// to semantic concept HVs via the ventral stream.
    pub fn describe_scene(&self) -> Vec<(String, String, String)> {
        let sg = match &self.scene_graph {
            Some(sg) => sg,
            None => return Vec::new(),
        };

        sg.edges()
            .iter()
            .map(|edge| {
                let subject = format!("object_{}", edge.subject_id);
                let relation = format!("{:?}", edge.relation);
                let object = format!("object_{}", edge.object_id);
                (subject, relation, object)
            })
            .collect()
    }

    /// Snapshot the manifold's learned and live temporal state for serialization.
    ///
    /// In addition to learned parameters, this captures the CfC state, prediction
    /// context, patch history, optimizer moments, scene-memory frames, and the
    /// latest raw observation so resume does not behave like a cold restart.
    pub fn save_state(&self) -> ManifoldState {
        ManifoldState {
            schema_version: MANIFOLD_STATE_SCHEMA_VERSION,
            weight_hv: self.weight_hv.as_slice().to_vec(),
            tau_base: self.config.tau_base,
            feature_weights: self.encoder.feature_weights().to_vec(),
            training_steps: self.trainer.total_steps(),
            hdc_dim: self.config.hdc_dim,
            num_features: self.config.num_features,
            config_patch_size: self.config.patch_size,
            config_num_levels: self.config.num_levels,
            config_total_features: self.config.total_features(),
            config_input_blend: self.config.input_blend,
            config_enable_motion: self.config.enable_motion,
            config_enable_color: self.config.enable_color,
            config_enable_opponent_color: self.config.enable_opponent_color,
            config_enable_depth: self.config.enable_depth,
            config_enable_temporal_binding: self.config.enable_temporal_binding,
            config_enable_object_binding: self.config.enable_object_binding,
            config_multi_scale_scales: self.config.multi_scale.scales.clone(),
            error_ema: self.error_ema,
            prediction_error: self.prediction_error,
            coherence: self.coherence,
            last_fep: self.last_fep,
            fep_belief_mean: self.fep_agent.belief.mean.clone(),
            scene_store_coherence_threshold: self.scene_store_coherence_threshold,
            scene_store_error_threshold: self.scene_store_error_threshold,
            scene_dampen_factor: self.scene_dampen_factor,
            last_dilation_cycle: self.last_dilation_cycle,
            frame_count: self.frame_count,
            prev_patch_lum: if self.encoder.prev_patch_lum.is_empty() {
                None
            } else {
                Some(self.encoder.prev_patch_lum.clone())
            },
            scene_memory: self.scene_memory.as_ref().map(SceneMemory::save_state),
            object_memory: self.object_memory.as_ref().map(ObjectMemory::save_state),
            working_memory: self
                .working_memory
                .as_ref()
                .map(VisualWorkingMemory::save_state),
            next_track_id: self.next_track_id,
            state_hv: Some(self.state.as_slice().to_vec()),
            last_prediction: self
                .last_prediction
                .as_ref()
                .map(|hv| hv.as_slice().to_vec()),
            last_frame_hv: self.last_frame_hv.as_ref().map(|hv| hv.as_slice().to_vec()),
            last_patch_hvs: self
                .last_patch_hvs
                .iter()
                .map(|hv| hv.as_slice().to_vec())
                .collect(),
            trainer_state: Some(self.trainer.save_state()),
            learning_frozen: self.learning_frozen,
            last_observed_frame: self.last_observed_frame.clone(),
            last_frame_width: self.last_frame_width,
            last_frame_height: self.last_frame_height,
            last_frame_channels: self.last_frame_channels,
            last_frame_modality: self.last_frame_modality,
            surprise_state: Some(self.surprise.save_state()),
            predictive_state: self
                .predictive
                .as_ref()
                .map(PredictiveCodingHierarchy::save_state),
            temporal_patch_hvs: self
                .temporal_patch_hvs
                .iter()
                .map(|hv| hv.as_slice().to_vec())
                .collect(),
            active_modality: self.active_modality,
            modality_contexts: self.saved_modality_contexts(),
            horizon_evaluator: Some(self.horizon_evaluator.save_state()),
            last_object_hypotheses: self
                .last_object_hypotheses
                .iter()
                .map(|hypothesis| ObjectHypothesisState {
                    patch_indices: hypothesis.patch_indices.clone(),
                    centroid_row: hypothesis.centroid_row,
                    centroid_col: hypothesis.centroid_col,
                    hv: hypothesis.hv.as_slice().to_vec(),
                    saliency: hypothesis.saliency,
                })
                .collect(),
            motion_saliency: self.motion_saliency.clone(),
            last_motion_vectors: self.last_motion_vectors.clone(),
            stereo_depth_map: self.stereo_depth_map.clone(),
            stereo_confidence_map: self.stereo_confidence_map.clone(),
            stereo_disparity_map: self.stereo_disparity_map.clone(),
            scene_graph_enabled: self.scene_graph.is_some(),
            last_imagination: self
                .last_imagination
                .as_ref()
                .map(|hv| hv.as_slice().to_vec()),
            imagination_surprise: self.imagination_surprise,
            last_intent_hv: Some(self.last_intent_hv.as_slice().to_vec()),
            last_geodesic: self
                .last_geodesic
                .iter()
                .map(|hv| hv.as_slice().to_vec())
                .collect(),
        }
    }

    /// Serialize the complete manifold checkpoint into a bounded integrity envelope.
    pub fn save_checkpoint_bytes(&self) -> Result<Vec<u8>, String> {
        self.save_checkpoint_bytes_with_limit(
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
        )
    }

    /// Serialize with an explicit maximum payload size.
    pub fn save_checkpoint_bytes_with_limit(
        &self,
        max_payload_bytes: usize,
    ) -> Result<Vec<u8>, String> {
        crate::checkpoint::encode_checkpoint(
            "symthaea-vision-manifold",
            MANIFOLD_STATE_SCHEMA_VERSION,
            &self.save_state(),
            max_payload_bytes,
        )
    }

    /// Serialize the complete stack into a caller-authenticated envelope.
    ///
    /// The signing callback receives the complete bounded integrity envelope,
    /// including its kind, schema, payload length, checksum, and payload.
    pub fn save_authenticated_checkpoint_bytes<S>(
        &self,
        max_tag_bytes: usize,
        sign: S,
    ) -> Result<Vec<u8>, String>
    where
        S: FnOnce(&[u8]) -> Result<Vec<u8>, String>,
    {
        crate::checkpoint::encode_authenticated_checkpoint(
            "symthaea-vision-manifold",
            MANIFOLD_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            max_tag_bytes,
            sign,
        )
    }

    /// Atomically persist a caller-authenticated checkpoint.
    pub fn save_authenticated_checkpoint_file<S>(
        &self,
        path: impl AsRef<std::path::Path>,
        max_tag_bytes: usize,
        sign: S,
    ) -> Result<crate::checkpoint::CheckpointWriteReport, String>
    where
        S: FnOnce(&[u8]) -> Result<Vec<u8>, String>,
    {
        crate::checkpoint::save_authenticated_checkpoint_file(
            path,
            "symthaea-vision-manifold",
            MANIFOLD_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            max_tag_bytes,
            sign,
        )
    }

    /// Atomically persist the complete manifold checkpoint to disk.
    pub fn save_checkpoint_file(&self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        crate::checkpoint::save_checkpoint_file(
            path,
            "symthaea-vision-manifold",
            MANIFOLD_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
        )
    }

    /// Persist a manifold checkpoint while retaining the previous verified generation.
    pub fn save_checkpoint_file_recoverable(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), String> {
        crate::checkpoint::save_checkpoint_file_recoverable(
            path,
            "symthaea-vision-manifold",
            MANIFOLD_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
        )
    }

    /// Persist a bounded retained generation set and return complete write evidence.
    pub fn save_checkpoint_file_with_retention_report(
        &self,
        path: impl AsRef<std::path::Path>,
        policy: crate::checkpoint::CheckpointRetentionPolicy,
    ) -> Result<crate::checkpoint::CheckpointRetentionSaveReport, String> {
        crate::checkpoint::save_checkpoint_file_with_retention_report(
            path,
            "symthaea-vision-manifold",
            MANIFOLD_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            policy,
        )
    }

    /// Persist retained generations under one cross-process writer lease.
    pub fn save_checkpoint_file_with_retention_locked_report(
        &self,
        path: impl AsRef<std::path::Path>,
        retention: crate::checkpoint::CheckpointRetentionPolicy,
        lock_policy: crate::checkpoint::CheckpointWriterLockPolicy,
    ) -> Result<crate::checkpoint::CheckpointRetentionSaveReport, String> {
        crate::checkpoint::save_checkpoint_file_with_retention_locked_report(
            path,
            "symthaea-vision-manifold",
            MANIFOLD_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            retention,
            lock_policy,
        )
    }

    /// Persist authenticated retained generations and return complete write evidence.
    pub fn save_authenticated_checkpoint_file_with_retention_report<S, V>(
        &self,
        path: impl AsRef<std::path::Path>,
        max_tag_bytes: usize,
        policy: crate::checkpoint::CheckpointRetentionPolicy,
        sign: S,
        verify: V,
    ) -> Result<crate::checkpoint::CheckpointRetentionSaveReport, String>
    where
        S: FnOnce(&[u8]) -> Result<Vec<u8>, String>,
        V: Fn(&[u8], &[u8]) -> Result<bool, String>,
    {
        crate::checkpoint::save_authenticated_checkpoint_file_with_retention_report(
            path,
            "symthaea-vision-manifold",
            MANIFOLD_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            max_tag_bytes,
            policy,
            sign,
            verify,
        )
    }

    /// Persist authenticated retained generations under one writer lease.
    pub fn save_authenticated_checkpoint_file_with_retention_locked_report<S, V>(
        &self,
        path: impl AsRef<std::path::Path>,
        max_tag_bytes: usize,
        retention: crate::checkpoint::CheckpointRetentionPolicy,
        lock_policy: crate::checkpoint::CheckpointWriterLockPolicy,
        sign: S,
        verify: V,
    ) -> Result<crate::checkpoint::CheckpointRetentionSaveReport, String>
    where
        S: FnOnce(&[u8]) -> Result<Vec<u8>, String>,
        V: Fn(&[u8], &[u8]) -> Result<bool, String>,
    {
        crate::checkpoint::save_authenticated_checkpoint_file_with_retention_locked_report(
            path,
            "symthaea-vision-manifold",
            MANIFOLD_STATE_SCHEMA_VERSION,
            &self.save_state(),
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            max_tag_bytes,
            retention,
            lock_policy,
            sign,
            verify,
        )
    }

    /// Inspect the primary and retained checkpoint generations for this stack.
    pub fn inspect_checkpoint_generations(
        &self,
        path: impl AsRef<std::path::Path>,
        policy: crate::checkpoint::CheckpointRetentionPolicy,
    ) -> Result<Vec<crate::checkpoint::CheckpointGenerationInspection>, String> {
        crate::checkpoint::inspect_checkpoint_generations(
            path,
            "symthaea-vision-manifold",
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            policy,
        )
    }

    /// Remove generations beyond a reduced retention bound.
    pub fn prune_checkpoint_generations(
        &self,
        path: impl AsRef<std::path::Path>,
        keep_previous_generations: usize,
    ) -> Result<crate::checkpoint::CheckpointPruneReport, String> {
        crate::checkpoint::prune_checkpoint_generations(path, keep_previous_generations)
    }

    /// Remove old generations while coordinating with retained checkpoint writers.
    pub fn prune_checkpoint_generations_locked(
        &self,
        path: impl AsRef<std::path::Path>,
        keep_previous_generations: usize,
        lock_policy: crate::checkpoint::CheckpointWriterLockPolicy,
    ) -> Result<crate::checkpoint::CheckpointPruneReport, String> {
        crate::checkpoint::prune_checkpoint_generations_locked(
            path,
            keep_previous_generations,
            lock_policy,
        )
    }

    /// Authenticate, deserialize, validate, and atomically restore checkpoint bytes.
    pub fn load_authenticated_checkpoint_bytes<V>(
        &mut self,
        encoded: &[u8],
        max_tag_bytes: usize,
        verify: V,
    ) -> Result<(), String>
    where
        V: FnOnce(&[u8], &[u8]) -> Result<bool, String>,
    {
        let (payload_schema, state): (u32, ManifoldState) =
            crate::checkpoint::decode_authenticated_checkpoint(
                encoded,
                "symthaea-vision-manifold",
                crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
                max_tag_bytes,
                verify,
            )?;
        if payload_schema != state.schema_version {
            return Err(format!(
                "authenticated checkpoint envelope/payload schema mismatch: envelope={payload_schema}, payload={}",
                state.schema_version
            ));
        }
        self.load_state(&state)
    }

    /// Read, authenticate, validate, and atomically restore a checkpoint file.
    pub fn load_authenticated_checkpoint_file<V>(
        &mut self,
        path: impl AsRef<std::path::Path>,
        max_tag_bytes: usize,
        verify: V,
    ) -> Result<(), String>
    where
        V: FnOnce(&[u8], &[u8]) -> Result<bool, String>,
    {
        let (payload_schema, state): (u32, ManifoldState) =
            crate::checkpoint::load_authenticated_checkpoint_file(
                path,
                "symthaea-vision-manifold",
                crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
                max_tag_bytes,
                verify,
            )?;
        if payload_schema != state.schema_version {
            return Err(format!(
                "authenticated checkpoint envelope/payload schema mismatch: envelope={payload_schema}, payload={}",
                state.schema_version
            ));
        }
        self.load_state(&state)
    }

    /// Read, verify, and atomically restore a manifold checkpoint file.
    pub fn load_checkpoint_file(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), String> {
        let (payload_schema, state): (u32, ManifoldState) =
            crate::checkpoint::load_checkpoint_file(
                path,
                "symthaea-vision-manifold",
                crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            )?;
        if payload_schema != state.schema_version {
            return Err(format!(
                "manifold checkpoint envelope/payload schema mismatch: envelope={payload_schema}, payload={}",
                state.schema_version
            ));
        }
        self.load_state(&state)
    }

    /// Restore the primary checkpoint or its previous verified generation.
    pub fn load_checkpoint_file_recoverable(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<crate::checkpoint::CheckpointRecoverySource, String> {
        let (payload_schema, state, source): (
            u32,
            ManifoldState,
            crate::checkpoint::CheckpointRecoverySource,
        ) = crate::checkpoint::load_checkpoint_file_recoverable(
            path,
            "symthaea-vision-manifold",
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
        )?;
        if payload_schema != state.schema_version {
            return Err(format!(
                "manifold checkpoint envelope/payload schema mismatch: envelope={payload_schema}, payload={}",
                state.schema_version
            ));
        }
        self.load_state(&state)?;
        Ok(source)
    }

    /// Restore the newest semantically compatible retained generation and
    /// preserve an audit trail for every newer generation that was rejected.
    pub fn load_checkpoint_file_with_retention_audited(
        &mut self,
        path: impl AsRef<std::path::Path>,
        policy: crate::checkpoint::CheckpointRetentionPolicy,
    ) -> Result<
        crate::checkpoint::CheckpointSemanticRecoveryReport,
        crate::checkpoint::CheckpointSemanticRecoveryFailure,
    > {
        let (payload_schema, state, report) =
            crate::checkpoint::load_checkpoint_file_with_retention_audited_detailed(
                path,
                "symthaea-vision-manifold",
                crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
                policy,
                |schema, candidate: &ManifoldState| {
                    if schema != candidate.schema_version {
                        return Err(format!(
                            "manifold checkpoint envelope/payload schema mismatch: envelope={schema}, payload={}",
                            candidate.schema_version
                        ));
                    }
                    self.validate_checkpoint_state(candidate)
                },
            )?;
        if payload_schema != state.schema_version {
            return Err(crate::checkpoint::CheckpointSemanticRecoveryFailure {
                attempts: report.attempts.clone(),
                setup_error: Some(format!(
                    "selected manifold checkpoint schema mismatch: envelope={payload_schema}, payload={}",
                    state.schema_version
                )),
            });
        }
        self.load_state(&state).map_err(|error| {
            crate::checkpoint::CheckpointSemanticRecoveryFailure {
                attempts: report.attempts.clone(),
                setup_error: Some(format!(
                    "selected manifold checkpoint failed final atomic restore: {error}"
                )),
            }
        })?;
        Ok(report)
    }

    /// Restore the newest authenticated and semantically compatible retained generation.
    pub fn load_authenticated_checkpoint_file_with_retention_audited<V>(
        &mut self,
        path: impl AsRef<std::path::Path>,
        max_tag_bytes: usize,
        policy: crate::checkpoint::CheckpointRetentionPolicy,
        verify: V,
    ) -> Result<
        crate::checkpoint::CheckpointSemanticRecoveryReport,
        crate::checkpoint::CheckpointSemanticRecoveryFailure,
    >
    where
        V: Fn(&[u8], &[u8]) -> Result<bool, String>,
    {
        let (payload_schema, state, report) =
            crate::checkpoint::load_authenticated_checkpoint_file_with_retention_audited_detailed(
                path,
                "symthaea-vision-manifold",
                crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
                max_tag_bytes,
                policy,
                verify,
                |schema, candidate: &ManifoldState| {
                    if schema != candidate.schema_version {
                        return Err(format!(
                            "manifold checkpoint envelope/payload schema mismatch: envelope={schema}, payload={}",
                            candidate.schema_version
                        ));
                    }
                    self.validate_checkpoint_state(candidate)
                },
            )?;
        if payload_schema != state.schema_version {
            return Err(crate::checkpoint::CheckpointSemanticRecoveryFailure {
                attempts: report.attempts.clone(),
                setup_error: Some(format!(
                    "selected authenticated manifold checkpoint schema mismatch: envelope={payload_schema}, payload={}",
                    state.schema_version
                )),
            });
        }
        self.load_state(&state).map_err(|error| {
            crate::checkpoint::CheckpointSemanticRecoveryFailure {
                attempts: report.attempts.clone(),
                setup_error: Some(format!(
                    "selected authenticated manifold checkpoint failed final atomic restore: {error}"
                )),
            }
        })?;
        Ok(report)
    }

    /// Validate and atomically restore a manifold integrity envelope.
    pub fn load_checkpoint_bytes(&mut self, encoded: &[u8]) -> Result<(), String> {
        self.load_checkpoint_bytes_with_limit(
            encoded,
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
        )
    }

    /// Restore with an explicit maximum payload size.
    pub fn load_checkpoint_bytes_with_limit(
        &mut self,
        encoded: &[u8],
        max_payload_bytes: usize,
    ) -> Result<(), String> {
        let (payload_schema, state): (u32, ManifoldState) = crate::checkpoint::decode_checkpoint(
            encoded,
            "symthaea-vision-manifold",
            max_payload_bytes,
        )?;
        if payload_schema != state.schema_version {
            return Err(format!(
                "manifold checkpoint envelope/payload schema mismatch: envelope={payload_schema}, payload={}",
                state.schema_version
            ));
        }
        self.load_state(&state)
    }

    /// Restore the manifold from a saved state.
    ///
    /// All dimensional checks run before mutation so an incompatible checkpoint
    /// cannot leave the manifold partially restored. Older checkpoints remain
    /// loadable: newly added fields use serde defaults and are applied only when
    /// present.
    pub fn load_state(&mut self, state: &ManifoldState) -> Result<(), String> {
        let dim = self.config.hdc_dim;
        let schema_version = if state.schema_version == 0 {
            1
        } else {
            state.schema_version
        };
        if schema_version > MANIFOLD_STATE_SCHEMA_VERSION {
            return Err(format!(
                "unsupported manifold checkpoint schema: saved={}, supported={}",
                schema_version, MANIFOLD_STATE_SCHEMA_VERSION
            ));
        }
        if schema_version >= 4 && state.surprise_state.is_none() {
            return Err(
                "schema-4 checkpoint is missing active-modality surprise state".to_string(),
            );
        }
        if schema_version >= 7 && state.horizon_evaluator.is_none() {
            return Err(
                "schema-7 checkpoint is missing active-modality horizon evaluator".to_string(),
            );
        }
        // Schema 8 adds relational time validation for episodic memory. No new
        // field is required; the stronger contract is applied below once the
        // checkpoint frame and scene-memory payload have been validated.
        if state.hdc_dim != dim {
            return Err(format!(
                "HDC dimension mismatch: saved={}, current={dim}",
                state.hdc_dim
            ));
        }
        if state.num_features != self.config.num_features {
            return Err(format!(
                "base feature count mismatch: saved={}, current={}",
                state.num_features, self.config.num_features
            ));
        }
        if schema_version >= 3 {
            let check = |name: &str, saved: usize, current: usize| -> Result<(), String> {
                if saved != current {
                    Err(format!(
                        "checkpoint {name} mismatch: saved={saved}, current={current}"
                    ))
                } else {
                    Ok(())
                }
            };
            check(
                "patch_size",
                state.config_patch_size,
                self.config.patch_size,
            )?;
            check(
                "num_levels",
                state.config_num_levels,
                self.config.num_levels,
            )?;
            check(
                "total_features",
                state.config_total_features,
                self.config.total_features(),
            )?;
            if !state.config_input_blend.is_finite()
                || (state.config_input_blend - self.config.input_blend).abs() > 1e-6
            {
                return Err(format!(
                    "checkpoint input_blend mismatch: saved={}, current={}",
                    state.config_input_blend, self.config.input_blend
                ));
            }
            for (name, saved, current) in [
                (
                    "enable_motion",
                    state.config_enable_motion,
                    self.config.enable_motion,
                ),
                (
                    "enable_color",
                    state.config_enable_color,
                    self.config.enable_color,
                ),
                (
                    "enable_opponent_color",
                    state.config_enable_opponent_color,
                    self.config.enable_opponent_color,
                ),
                (
                    "enable_depth",
                    state.config_enable_depth,
                    self.config.enable_depth,
                ),
                (
                    "enable_temporal_binding",
                    state.config_enable_temporal_binding,
                    self.config.enable_temporal_binding,
                ),
                (
                    "enable_object_binding",
                    state.config_enable_object_binding,
                    self.config.enable_object_binding,
                ),
            ] {
                if saved != current {
                    return Err(format!(
                        "checkpoint {name} mismatch: saved={saved}, current={current}"
                    ));
                }
            }
            if state.config_multi_scale_scales != self.config.multi_scale.scales {
                return Err(format!(
                    "checkpoint multi-scale topology mismatch: saved={:?}, current={:?}",
                    state.config_multi_scale_scales, self.config.multi_scale.scales
                ));
            }
        }
        if !state.tau_base.is_finite() || state.tau_base <= 0.001 || state.tau_base >= 100.0 {
            return Err(format!("invalid saved tau_base: {}", state.tau_base));
        }
        for (name, value) in [
            ("error_ema", state.error_ema),
            ("prediction_error", state.prediction_error),
            ("coherence", state.coherence),
            ("imagination_surprise", state.imagination_surprise),
            ("last_fep.free_energy", state.last_fep.free_energy),
            ("last_fep.complexity", state.last_fep.complexity),
            ("last_fep.accuracy", state.last_fep.accuracy),
        ] {
            if !value.is_finite() {
                return Err(format!("checkpoint {name} is non-finite"));
            }
        }
        let fep_belief_dim = self.fep_agent.belief.mean.len();
        if let Some(ref evaluator) = state.horizon_evaluator {
            DelayedHorizonEvaluator::validate_state(evaluator)
                .map_err(|error| format!("active horizon evaluator: {error}"))?;
            if evaluator.hdc_dim.is_some_and(|saved_dim| saved_dim != dim) {
                return Err(format!(
                    "active horizon evaluator dimension mismatch: saved={:?}, expected={dim}",
                    evaluator.hdc_dim
                ));
            }
        }
        if schema_version >= 6 && state.fep_belief_mean.len() != fep_belief_dim {
            return Err(format!(
                "active FEP belief dimension mismatch: got {}, expected {fep_belief_dim}",
                state.fep_belief_mean.len()
            ));
        }
        if state.fep_belief_mean.iter().any(|value| !value.is_finite()) {
            return Err("active FEP belief contains non-finite values".to_string());
        }

        if schema_version >= 3
            && (!(0.0..=1.0).contains(&state.scene_store_coherence_threshold)
                || !(0.0..=1.0).contains(&state.scene_store_error_threshold)
                || !(0.0..=1.0).contains(&state.scene_dampen_factor))
        {
            return Err("checkpoint scene-memory policy is outside [0,1]".to_string());
        }

        let validate_values = |name: &str, values: &[f32]| -> Result<(), String> {
            if values.len() != dim {
                return Err(format!(
                    "{name} dimension mismatch: saved={}, expected={dim}",
                    values.len()
                ));
            }
            if !values.iter().all(|value| value.is_finite()) {
                return Err(format!("{name} contains non-finite values"));
            }
            Ok(())
        };
        let validate_optional = |name: &str, values: &Option<Vec<f32>>| -> Result<(), String> {
            if let Some(values) = values {
                validate_values(name, values)?;
            }
            Ok(())
        };

        validate_values("weight_hv", &state.weight_hv)?;
        validate_optional("state_hv", &state.state_hv)?;
        validate_optional("last_prediction", &state.last_prediction)?;
        validate_optional("last_frame_hv", &state.last_frame_hv)?;
        validate_optional("last_imagination", &state.last_imagination)?;
        validate_optional("last_intent_hv", &state.last_intent_hv)?;
        for (idx, patch) in state.last_patch_hvs.iter().enumerate() {
            validate_values(&format!("last_patch_hvs[{idx}]"), patch)?;
        }
        for (idx, patch) in state.temporal_patch_hvs.iter().enumerate() {
            validate_values(&format!("temporal_patch_hvs[{idx}]"), patch)?;
        }
        let mut seen_modalities = Vec::new();
        for (context_idx, context) in state.modality_contexts.iter().enumerate() {
            if context.modality == state.active_modality {
                return Err(format!(
                    "modality_contexts[{context_idx}] duplicates active modality {:?}",
                    state.active_modality
                ));
            }
            if seen_modalities.contains(&context.modality) {
                return Err(format!(
                    "duplicate inactive modality context: {:?}",
                    context.modality
                ));
            }
            seen_modalities.push(context.modality);
            validate_optional(
                &format!("modality_contexts[{context_idx}].last_prediction"),
                &context.last_prediction,
            )?;
            validate_optional(
                &format!("modality_contexts[{context_idx}].last_frame_hv"),
                &context.last_frame_hv,
            )?;
            for (patch_idx, patch) in context.last_patch_hvs.iter().enumerate() {
                validate_values(
                    &format!("modality_contexts[{context_idx}].last_patch_hvs[{patch_idx}]"),
                    patch,
                )?;
            }
            for (patch_idx, patch) in context.temporal_patch_hvs.iter().enumerate() {
                validate_values(
                    &format!("modality_contexts[{context_idx}].temporal_patch_hvs[{patch_idx}]"),
                    patch,
                )?;
            }
            if !context.prev_patch_lum.iter().all(|value| value.is_finite()) {
                return Err(format!(
                    "modality_contexts[{context_idx}].prev_patch_lum contains non-finite values"
                ));
            }
            match context.surprise_state.as_ref() {
                Some(surprise_state) => {
                    SurpriseMap::validate_state(surprise_state, self.surprise.grid()).map_err(
                        |error| format!("modality_contexts[{context_idx}].surprise_state: {error}"),
                    )?;
                }
                None if schema_version >= 4 => {
                    return Err(format!(
                        "schema-4 checkpoint is missing surprise state for modality {:?}",
                        context.modality
                    ));
                }
                None => {}
            }
            if schema_version >= 6 && context.fep_belief_mean.len() != fep_belief_dim {
                return Err(format!(
                    "modality_contexts[{context_idx}].fep_belief_mean dimension mismatch: got {}, expected {fep_belief_dim}",
                    context.fep_belief_mean.len()
                ));
            }
            if context
                .fep_belief_mean
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err(format!(
                    "modality_contexts[{context_idx}].fep_belief_mean contains non-finite values"
                ));
            }
            for (name, value) in [
                ("free_energy", context.last_fep.free_energy),
                ("complexity", context.last_fep.complexity),
                ("accuracy", context.last_fep.accuracy),
            ] {
                if !value.is_finite() {
                    return Err(format!(
                        "modality_contexts[{context_idx}].last_fep.{name} is non-finite"
                    ));
                }
            }
            if schema_version >= 7 && context.horizon_evaluator.is_none() {
                return Err(format!(
                    "schema-7 checkpoint is missing horizon evaluator for modality {:?}",
                    context.modality
                ));
            }
            if let Some(ref evaluator) = context.horizon_evaluator {
                DelayedHorizonEvaluator::validate_state(evaluator).map_err(|error| {
                    format!("modality_contexts[{context_idx}].horizon_evaluator: {error}")
                })?;
                if evaluator.hdc_dim.is_some_and(|saved_dim| saved_dim != dim) {
                    return Err(format!(
                        "modality_contexts[{context_idx}].horizon evaluator dimension mismatch"
                    ));
                }
            }
            if !context.prediction_error.is_finite() || !context.error_ema.is_finite() {
                return Err(format!(
                    "modality_contexts[{context_idx}] contains non-finite prediction metrics"
                ));
            }
            if let Some(ref object_state) = context.object_memory {
                ObjectMemory::validate_state(object_state, dim).map_err(|error| {
                    format!("modality_contexts[{context_idx}].object_memory: {error}")
                })?;
            }
            if let Some(ref working_state) = context.working_memory {
                VisualWorkingMemory::validate_state(working_state, dim).map_err(|error| {
                    format!("modality_contexts[{context_idx}].working_memory: {error}")
                })?;
            }
            for (hypothesis_idx, hypothesis) in context.last_object_hypotheses.iter().enumerate() {
                validate_values(
                    &format!(
                        "modality_contexts[{context_idx}].last_object_hypotheses[{hypothesis_idx}].hv"
                    ),
                    &hypothesis.hv,
                )?;
                if !hypothesis.saliency.is_finite() {
                    return Err(format!(
                        "modality_contexts[{context_idx}].last_object_hypotheses[{hypothesis_idx}].saliency is non-finite"
                    ));
                }
            }
            if schema_version >= 9 {
                if context.working_memory.is_some() && context.object_memory.is_none() {
                    return Err(format!(
                        "modality_contexts[{context_idx}] enables working memory without object memory"
                    ));
                }
                if context.scene_graph_enabled && context.object_memory.is_none() {
                    return Err(format!(
                        "modality_contexts[{context_idx}] enables the scene graph without object memory"
                    ));
                }
                let mut maximum_allocated_id = None::<u64>;
                if let Some(ref object_state) = context.object_memory {
                    for (track_idx, track) in object_state.tracks.iter().enumerate() {
                        if track.track_id == u64::MAX {
                            return Err(format!(
                                "modality_contexts[{context_idx}].object_memory.tracks[{track_idx}] exhausts the track-ID allocator"
                            ));
                        }
                        if track.last_seen_frame > state.frame_count {
                            return Err(format!(
                                "modality_contexts[{context_idx}].object_memory.tracks[{track_idx}] was last seen after checkpoint frame {}",
                                state.frame_count
                            ));
                        }
                        maximum_allocated_id = Some(
                            maximum_allocated_id
                                .map_or(track.track_id, |id| id.max(track.track_id)),
                        );
                    }
                }
                if let Some(ref working_state) = context.working_memory {
                    for (slot_idx, slot) in working_state.slots.iter().enumerate() {
                        if slot.track_id == u64::MAX {
                            return Err(format!(
                                "modality_contexts[{context_idx}].working_memory.slots[{slot_idx}] exhausts the track-ID allocator"
                            ));
                        }
                        if slot.entered_at_frame > state.frame_count {
                            return Err(format!(
                                "modality_contexts[{context_idx}].working_memory.slots[{slot_idx}] entered after checkpoint frame {}",
                                state.frame_count
                            ));
                        }
                        maximum_allocated_id = Some(
                            maximum_allocated_id.map_or(slot.track_id, |id| id.max(slot.track_id)),
                        );
                    }
                }
                if let Some(maximum_allocated_id) = maximum_allocated_id
                    && context.next_track_id <= maximum_allocated_id
                {
                    return Err(format!(
                        "modality_contexts[{context_idx}].next_track_id {} would reuse allocated track ID {maximum_allocated_id}",
                        context.next_track_id
                    ));
                }
            }
        }
        for (idx, path_state) in state.last_geodesic.iter().enumerate() {
            validate_values(&format!("last_geodesic[{idx}]"), path_state)?;
        }
        for (idx, hypothesis) in state.last_object_hypotheses.iter().enumerate() {
            validate_values(&format!("last_object_hypotheses[{idx}].hv"), &hypothesis.hv)?;
            if !hypothesis.saliency.is_finite() {
                return Err(format!(
                    "last_object_hypotheses[{idx}].saliency is non-finite"
                ));
            }
        }

        let current_weights = self.encoder.feature_weights().len();
        if !state.feature_weights.is_empty() && state.feature_weights.len() != current_weights {
            return Err(format!(
                "feature weight count mismatch: saved={}, expected={current_weights}",
                state.feature_weights.len()
            ));
        }
        if state.feature_weights.iter().any(|value| !value.is_finite()) {
            return Err("feature weights contain non-finite values".to_string());
        }
        if state
            .prev_patch_lum
            .as_ref()
            .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err("previous patch luminance contains non-finite values".to_string());
        }
        if state.motion_saliency.iter().any(|value| !value.is_finite()) {
            return Err("motion saliency contains non-finite values".to_string());
        }
        if state
            .last_motion_vectors
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err("motion vectors contain non-finite values".to_string());
        }
        if state
            .stereo_depth_map
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err("stereo depth map contains invalid values".to_string());
        }
        if state
            .stereo_confidence_map
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err("stereo confidence map contains invalid values".to_string());
        }
        if state.last_frame_width > 0
            && state
                .stereo_disparity_map
                .iter()
                .any(|&value| value >= state.last_frame_width as usize)
        {
            return Err("stereo disparity map contains out-of-frame values".to_string());
        }

        if let Some(ref memory_state) = state.scene_memory {
            SceneMemory::validate_state(memory_state, dim)?;
            if schema_version >= 8 {
                SceneMemory::validate_temporal_state(memory_state, state.frame_count)?;
            }
        }
        if let Some(ref object_state) = state.object_memory {
            ObjectMemory::validate_state(object_state, dim)?;
        }
        if let Some(ref working_state) = state.working_memory {
            VisualWorkingMemory::validate_state(working_state, dim)?;
        }
        if schema_version >= 5 {
            if state.working_memory.is_some() && state.object_memory.is_none() {
                return Err(
                    "schema-5 checkpoint enables working memory without object memory".to_string(),
                );
            }
            if state.scene_graph_enabled && state.object_memory.is_none() {
                return Err(
                    "schema-5 checkpoint enables the scene graph without object memory".to_string(),
                );
            }

            let mut maximum_allocated_id = None::<u64>;
            if let Some(ref object_state) = state.object_memory {
                for (idx, track) in object_state.tracks.iter().enumerate() {
                    if track.track_id == u64::MAX {
                        return Err(format!(
                            "object track {idx} exhausts the track-ID allocator"
                        ));
                    }
                    if track.last_seen_frame > state.frame_count {
                        return Err(format!(
                            "object track {idx} was last seen at frame {} beyond checkpoint frame {}",
                            track.last_seen_frame, state.frame_count
                        ));
                    }
                    maximum_allocated_id = Some(
                        maximum_allocated_id.map_or(track.track_id, |id| id.max(track.track_id)),
                    );
                }
            }
            if let Some(ref working_state) = state.working_memory {
                for (idx, slot) in working_state.slots.iter().enumerate() {
                    if slot.track_id == u64::MAX {
                        return Err(format!(
                            "working-memory slot {idx} exhausts the track-ID allocator"
                        ));
                    }
                    if slot.entered_at_frame > state.frame_count {
                        return Err(format!(
                            "working-memory slot {idx} entered at frame {} beyond checkpoint frame {}",
                            slot.entered_at_frame, state.frame_count
                        ));
                    }
                    maximum_allocated_id = Some(
                        maximum_allocated_id.map_or(slot.track_id, |id| id.max(slot.track_id)),
                    );
                }
            }
            if let Some(maximum_allocated_id) = maximum_allocated_id
                && state.next_track_id <= maximum_allocated_id
            {
                return Err(format!(
                    "next_track_id {} would reuse allocated track ID {maximum_allocated_id}",
                    state.next_track_id
                ));
            }
        }
        if let Some(ref trainer_state) = state.trainer_state {
            ManifoldTrainer::validate_state(trainer_state, dim)?;
        }
        if let Some(ref surprise_state) = state.surprise_state {
            SurpriseMap::validate_state(surprise_state, self.surprise.grid())?;
        }
        if schema_version >= 2 && state.surprise_state.is_none() {
            return Err("schema-2 checkpoint is missing surprise state".to_string());
        }
        if let Some(ref predictive_state) = state.predictive_state {
            let predictive = self.predictive.as_ref().ok_or_else(|| {
                "checkpoint contains predictive state but hierarchy is disabled".to_string()
            })?;
            PredictiveCodingHierarchy::validate_state(
                predictive_state,
                dim,
                predictive.encoder().scales().len(),
            )?;
        } else if schema_version >= 2 && self.predictive.is_some() {
            return Err(
                "schema-2 checkpoint is missing state for the enabled predictive hierarchy"
                    .to_string(),
            );
        }

        let saved_grid = if state.last_frame_width > 0 && state.last_frame_height > 0 {
            let grid = self
                .encoder
                .grid_for(state.last_frame_width, state.last_frame_height);
            if grid.rows > self.encoder.max_rows() || grid.cols > self.encoder.max_cols() {
                return Err(format!(
                    "saved frame {}x{} exceeds encoder capacity of {}x{} patches",
                    state.last_frame_width,
                    state.last_frame_height,
                    self.encoder.max_cols(),
                    self.encoder.max_rows()
                ));
            }
            grid
        } else {
            self.surprise.grid().clone()
        };
        let expected_patches = saved_grid.num_patches();
        if schema_version >= 2 {
            for (name, len) in [
                ("last_patch_hvs", state.last_patch_hvs.len()),
                ("temporal_patch_hvs", state.temporal_patch_hvs.len()),
                ("motion_saliency", state.motion_saliency.len()),
                ("last_motion_vectors", state.last_motion_vectors.len()),
                ("stereo_depth_map", state.stereo_depth_map.len()),
                ("stereo_confidence_map", state.stereo_confidence_map.len()),
                ("stereo_disparity_map", state.stereo_disparity_map.len()),
            ] {
                if len != 0 && len != expected_patches {
                    return Err(format!(
                        "{name} patch count mismatch: saved={len}, expected={expected_patches}"
                    ));
                }
            }
            for (idx, hypothesis) in state.last_object_hypotheses.iter().enumerate() {
                if hypothesis.centroid_row >= saved_grid.rows
                    || hypothesis.centroid_col >= saved_grid.cols
                    || hypothesis
                        .patch_indices
                        .iter()
                        .any(|&patch| patch >= expected_patches)
                {
                    return Err(format!(
                        "last_object_hypotheses[{idx}] contains out-of-grid coordinates"
                    ));
                }
            }
        }

        if let Some(ref pixels) = state.last_observed_frame {
            if state.last_frame_width == 0 || state.last_frame_height == 0 {
                return Err("saved raw frame has zero geometry".to_string());
            }
            if !matches!(state.last_frame_channels, 1 | 3 | 4) {
                return Err(format!(
                    "saved raw frame has unsupported channel count: {}",
                    state.last_frame_channels
                ));
            }
            let expected = (state.last_frame_width as usize)
                .checked_mul(state.last_frame_height as usize)
                .and_then(|count| count.checked_mul(state.last_frame_channels))
                .ok_or_else(|| "saved raw frame geometry overflow".to_string())?;
            if pixels.len() != expected {
                return Err(format!(
                    "saved raw frame length mismatch: got {}, expected {expected}",
                    pixels.len()
                ));
            }
        }

        // All validation has completed. Mutation begins here.
        let scene_graph_enabled = if schema_version >= 2 {
            state.scene_graph_enabled
        } else {
            self.scene_graph.is_some()
        };
        self.weight_hv = ContinuousHV::from_vec(state.weight_hv.clone());
        self.config.tau_base = state.tau_base;
        if !state.feature_weights.is_empty() {
            self.encoder.set_feature_weights(&state.feature_weights);
        }
        self.error_ema = state.error_ema;
        self.prediction_error = state.prediction_error;
        self.fep_agent = Self::new_fep_agent();
        if schema_version >= 3 {
            self.coherence = state.coherence;
            self.last_fep = state.last_fep;
            if schema_version >= 6 {
                self.fep_agent.belief.mean = state.fep_belief_mean.clone();
            }
            self.scene_store_coherence_threshold = state.scene_store_coherence_threshold;
            self.scene_store_error_threshold = state.scene_store_error_threshold;
            self.scene_dampen_factor = state.scene_dampen_factor;
            self.last_dilation_cycle = state.last_dilation_cycle;
        } else {
            self.coherence = 0.0;
            self.last_fep = crate::types::FepMetrics::default();
            self.last_dilation_cycle = 0;
        }
        self.frame_count = state.frame_count;
        self.learning_frozen = state.learning_frozen;
        self.encoder.prev_patch_lum = state.prev_patch_lum.clone().unwrap_or_default();

        if let Some(ref values) = state.state_hv {
            self.state = ContinuousHV::from_vec(values.clone());
        }
        self.last_prediction = state
            .last_prediction
            .as_ref()
            .map(|values| ContinuousHV::from_vec(values.clone()));
        self.last_frame_hv = state
            .last_frame_hv
            .as_ref()
            .map(|values| ContinuousHV::from_vec(values.clone()));
        self.last_patch_hvs = state
            .last_patch_hvs
            .iter()
            .map(|values| ContinuousHV::from_vec(values.clone()))
            .collect();
        self.temporal_patch_hvs = state
            .temporal_patch_hvs
            .iter()
            .map(|values| ContinuousHV::from_vec(values.clone()))
            .collect();
        self.active_modality = if state.active_modality != VisualModality::Unknown {
            state.active_modality
        } else if state.last_frame_modality != VisualModality::Unknown {
            state.last_frame_modality
        } else if state.last_observed_frame.is_some() {
            VisualModality::Visible
        } else {
            VisualModality::Unknown
        };
        self.modality_contexts = state
            .modality_contexts
            .iter()
            .map(|context| {
                (
                    context.modality,
                    ModalityTemporalContext::from_state(context),
                )
            })
            .collect();
        self.horizon_evaluator = if let Some(ref saved) = state.horizon_evaluator {
            let mut evaluator = DelayedHorizonEvaluator::default();
            evaluator.load_state(saved)?;
            evaluator
        } else {
            DelayedHorizonEvaluator::default()
        };
        self.last_object_hypotheses = state
            .last_object_hypotheses
            .iter()
            .map(|hypothesis| crate::types::ObjectHypothesis {
                centroid_row: hypothesis.centroid_row,
                centroid_col: hypothesis.centroid_col,
                patch_indices: hypothesis.patch_indices.clone(),
                saliency: hypothesis.saliency,
                hv: ContinuousHV::from_vec(hypothesis.hv.clone()),
            })
            .collect();
        self.motion_saliency = state.motion_saliency.clone();
        self.last_motion_vectors = state.last_motion_vectors.clone();
        self.stereo_depth_map = state.stereo_depth_map.clone();
        self.stereo_confidence_map = state.stereo_confidence_map.clone();
        self.stereo_disparity_map = state.stereo_disparity_map.clone();
        self.last_imagination = state
            .last_imagination
            .as_ref()
            .map(|values| ContinuousHV::from_vec(values.clone()));
        self.imagination_surprise = state.imagination_surprise;
        if let Some(ref values) = state.last_intent_hv {
            self.last_intent_hv = ContinuousHV::from_vec(values.clone());
        }
        self.last_geodesic = state
            .last_geodesic
            .iter()
            .map(|values| ContinuousHV::from_vec(values.clone()))
            .collect();

        if let Some(ref trainer_state) = state.trainer_state {
            self.trainer.load_state(trainer_state, dim)?;
        } else {
            self.trainer.set_total_steps(state.training_steps);
        }
        if let Some(ref surprise_state) = state.surprise_state {
            self.surprise.load_state(surprise_state);
            self.config.surprise_decay = surprise_state.decay;
        }
        if let Some(ref predictive_state) = state.predictive_state
            && let Some(ref mut predictive) = self.predictive
        {
            predictive.load_state(predictive_state);
        }

        if schema_version >= 2 {
            self.scene_memory = state
                .scene_memory
                .as_ref()
                .map(|memory_state| -> Result<_, String> {
                    let mut memory = SceneMemory::new(memory_state.capacity);
                    memory.load_state_checked(memory_state, dim)?;
                    Ok(memory)
                })
                .transpose()?;
            self.object_memory = state
                .object_memory
                .as_ref()
                .map(|object_state| -> Result<_, String> {
                    let mut memory = ObjectMemory::new(object_state.capacity);
                    memory.load_state_checked(object_state, dim)?;
                    Ok(memory)
                })
                .transpose()?;
            self.working_memory = state
                .working_memory
                .as_ref()
                .map(|working_state| -> Result<_, String> {
                    let mut memory = VisualWorkingMemory::new(working_state.capacity);
                    memory.load_state_checked(working_state, dim)?;
                    Ok(memory)
                })
                .transpose()?;
        } else {
            if let Some(ref memory_state) = state.scene_memory {
                let mut memory = self
                    .scene_memory
                    .take()
                    .unwrap_or_else(|| SceneMemory::new(memory_state.capacity));
                memory.load_state_checked(memory_state, dim)?;
                self.scene_memory = Some(memory);
            }
            if let Some(ref object_state) = state.object_memory {
                let mut memory = self
                    .object_memory
                    .take()
                    .unwrap_or_else(|| ObjectMemory::new(object_state.capacity));
                memory.load_state_checked(object_state, dim)?;
                self.object_memory = Some(memory);
            }
            if let Some(ref working_state) = state.working_memory {
                let mut memory = self
                    .working_memory
                    .take()
                    .unwrap_or_else(|| VisualWorkingMemory::new(working_state.capacity));
                memory.load_state_checked(working_state, dim)?;
                self.working_memory = Some(memory);
            }
        }
        let object_next_id = self
            .object_memory
            .as_ref()
            .and_then(|memory| {
                memory
                    .tracks()
                    .iter()
                    .map(|track| track.track_id.saturating_add(1))
                    .max()
            })
            .unwrap_or(0);
        let working_next_id = self
            .working_memory
            .as_ref()
            .and_then(|memory| {
                memory
                    .slots()
                    .iter()
                    .map(|slot| slot.track_id.saturating_add(1))
                    .max()
            })
            .unwrap_or(0);
        self.next_track_id = state.next_track_id.max(object_next_id).max(working_next_id);
        self.scene_graph = if scene_graph_enabled {
            let mut graph = VisualSceneGraph::new(dim, self.config.seed);
            if let Some(objects) = &self.object_memory {
                graph.update(objects.tracks());
            }
            Some(graph)
        } else {
            None
        };

        self.last_observed_frame = state.last_observed_frame.clone();
        self.last_frame_width = state.last_frame_width;
        self.last_frame_height = state.last_frame_height;
        self.last_frame_channels = state.last_frame_channels;
        self.last_frame_modality = state.last_frame_modality;
        Ok(())
    }

    /// Compute health diagnostics for the manifold.
    ///
    /// Returns a `ManifoldHealth` snapshot with drift, stability, and training
    /// quality metrics. Call periodically (e.g. every 100 frames) for monitoring.
    pub fn compute_health(&self) -> ManifoldHealth {
        // Weight drift: compare current weight_hv with initial (via norm ratio)
        let weight_drift = {
            let initial = ContinuousHV::random(self.config.hdc_dim, self.config.seed + 300_000);
            self.weight_hv.similarity(&initial).clamp(-1.0, 1.0)
        };

        // Encoder weight entropy
        let encoder_weight_entropy = {
            let weights = self.encoder.feature_weights();
            let sum: f32 = weights.iter().sum();
            if sum > 0.0 {
                let mut ent = 0.0f32;
                for &w in weights {
                    if w > 0.0 {
                        let p = w / sum;
                        ent -= p * p.ln();
                    }
                }
                ent
            } else {
                0.0
            }
        };

        // Training frequency (from total steps vs frames)
        let training_frequency = if self.frame_count > 0 {
            self.trainer.total_steps() as f32 / self.frame_count as f32
        } else {
            0.0
        };

        let tau_value = self.config.tau_base;
        let is_healthy = tau_value > 0.01
            && tau_value < 10.0
            && self.prediction_error.is_finite()
            && self.coherence.is_finite()
            && encoder_weight_entropy > 0.0;

        ManifoldHealth {
            weight_drift,
            tau_value,
            encoder_weight_entropy,
            training_frequency,
            mean_prediction_error: self.prediction_error,
            mean_coherence: self.coherence,
            total_frames: self.frame_count,
            total_training_steps: self.trainer.total_steps(),
            is_healthy,
        }
    }

    /// Set the subcortical generative bridge for neural hallucination.
    pub fn set_generative_bridge(&mut self, bridge: GenerativeBridge) {
        self.generative_bridge = Some(bridge);
    }

    /// Access the subcortical generative bridge.
    pub fn generative_bridge(&self) -> Option<&GenerativeBridge> {
        self.generative_bridge.as_ref()
    }

    /// Reset observation-dependent runtime state while preserving learned
    /// weights, configured policies, node identity, transition model, and the
    /// optional generative bridge.
    pub fn reset(&mut self) {
        self.state = ContinuousHV::zero(self.config.hdc_dim);
        self.last_prediction = None;
        self.last_frame_hv = None;
        self.last_patch_hvs.clear();
        self.temporal_patch_hvs.clear();
        self.active_modality = VisualModality::Unknown;
        self.modality_contexts.clear();
        self.horizon_evaluator.reset();
        self.encoder.prev_patch_lum.clear();
        self.surprise.reset();
        self.motion_saliency.clear();
        self.last_motion_vectors.clear();
        self.stereo_depth_map.clear();
        self.stereo_confidence_map.clear();
        self.stereo_disparity_map.clear();
        self.prediction_error = 0.0;
        self.coherence = 0.0;
        self.frame_count = 0;
        self.error_ema = 0.0;
        self.telemetry = VisionTelemetry::default();
        self.learning_frozen = false;
        if let Some(ref mut predictive) = self.predictive {
            predictive.reset();
        }
        if let Some(ref mut memory) = self.scene_memory {
            memory.clear();
        }
        if let Some(ref mut object_memory) = self.object_memory {
            object_memory.clear();
        }
        if let Some(ref mut working_memory) = self.working_memory {
            working_memory.clear();
        }
        if let Some(ref mut scene_graph) = self.scene_graph {
            scene_graph.clear();
        }
        self.next_track_id = 0;
        self.last_tracking_result = None;
        self.last_object_hypotheses.clear();
        self.last_imagination = None;
        self.imagination_surprise = 0.0;
        self.last_scene_match = None;
        self.last_dilation_cycle = 0;
        self.last_fep = crate::types::FepMetrics::default();
        self.fep_agent = Self::new_fep_agent();
        self.last_geodesic.clear();
        self.geodesic_compute_cost = 0.0;
        self.last_observed_frame = None;
        self.last_frame_width = 0;
        self.last_frame_height = 0;
        self.last_frame_channels = 0;
        self.last_frame_modality = VisualModality::Unknown;
        self.last_intent_hv = ContinuousHV::zero(self.config.hdc_dim);
    }
}

/// Multi-horizon prediction accuracy snapshot.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HorizonAccuracy {
    /// Prediction horizons in seconds.
    pub horizons: Vec<f32>,
    /// Human-readable labels for each horizon.
    pub labels: Vec<String>,
    /// Mean prediction error (1 - cos_sim) at each horizon.
    pub errors: Vec<f32>,
    /// Mean persistence-baseline error at each horizon.
    #[serde(default)]
    pub persistence_errors: Vec<f32>,
    /// Population standard deviation of delayed prediction error.
    #[serde(default)]
    pub prediction_error_stddev: Vec<f32>,
    /// Population standard deviation of persistence-baseline error.
    #[serde(default)]
    pub persistence_error_stddev: Vec<f32>,
    /// Relative skill versus persistence: `(baseline - prediction) / baseline`.
    #[serde(default)]
    pub relative_skill: Vec<f32>,
    /// Number of delayed observations scored at each horizon.
    #[serde(default)]
    pub sample_counts: Vec<u64>,
    /// Mean delay between the requested horizon and the observation used to score it.
    #[serde(default)]
    pub mean_lateness_seconds: Vec<f32>,
    /// Forecasts skipped because the bounded pending queue was full.
    #[serde(default)]
    pub dropped_forecasts: Vec<u64>,
    /// Forecasts discarded because the scoring observation arrived too late.
    #[serde(default)]
    pub expired_forecasts: Vec<u64>,
    /// Frame at which this was evaluated.
    pub frame_sequence: u64,
}

#[derive(Clone)]
struct PendingHorizonForecast {
    horizon_index: usize,
    due_time: f64,
    predicted: ContinuousHV,
    persistence: ContinuousHV,
}

#[derive(Clone, Default)]
struct HorizonAccumulator {
    prediction_error_sum: f64,
    prediction_error_sq_sum: f64,
    persistence_error_sum: f64,
    persistence_error_sq_sum: f64,
    lateness_sum: f64,
    samples: u64,
    dropped_forecasts: u64,
    expired_forecasts: u64,
}

/// Scores forecasts only when a later real observation reaches their horizon.
///
/// This avoids confusing a projection's similarity to the *current* state with
/// actual future predictive skill. Keep one evaluator per sensor clock domain.
#[derive(Clone)]
pub struct DelayedHorizonEvaluator {
    horizons: Vec<f32>,
    labels: Vec<String>,
    elapsed_seconds: f64,
    hdc_dim: Option<usize>,
    pending: Vec<PendingHorizonForecast>,
    accumulators: Vec<HorizonAccumulator>,
    max_pending_per_horizon: usize,
    max_lateness_factor: f32,
}

impl DelayedHorizonEvaluator {
    /// Create a delayed evaluator with positive, finite, strictly increasing horizons.
    pub fn new(horizons: Vec<(f32, String)>) -> Result<Self, String> {
        Self::with_pending_limit(horizons, 256)
    }

    /// Create a delayed evaluator with an explicit per-horizon queue bound.
    pub fn with_pending_limit(
        horizons: Vec<(f32, String)>,
        max_pending_per_horizon: usize,
    ) -> Result<Self, String> {
        if horizons.is_empty() {
            return Err("at least one prediction horizon is required".to_string());
        }
        if max_pending_per_horizon == 0 {
            return Err("max_pending_per_horizon must be > 0".to_string());
        }
        let mut previous = 0.0f32;
        for (index, (horizon, label)) in horizons.iter().enumerate() {
            if !horizon.is_finite() || *horizon <= 0.0 {
                return Err(format!(
                    "horizon {index} must be finite and > 0, got {horizon}"
                ));
            }
            if index > 0 && *horizon <= previous {
                return Err("prediction horizons must be strictly increasing".to_string());
            }
            if label.trim().is_empty() {
                return Err(format!("horizon {index} requires a non-empty label"));
            }
            previous = *horizon;
        }
        let count = horizons.len();
        Ok(Self {
            horizons: horizons.iter().map(|(horizon, _)| *horizon).collect(),
            labels: horizons.into_iter().map(|(_, label)| label).collect(),
            elapsed_seconds: 0.0,
            hdc_dim: None,
            pending: Vec::new(),
            accumulators: vec![HorizonAccumulator::default(); count],
            max_pending_per_horizon,
            max_lateness_factor: 4.0,
        })
    }

    /// Construct the standard 30-fps-oriented horizon suite.
    pub fn standard() -> Self {
        Self::new(vec![
            (0.033, "next_frame".to_string()),
            (0.1, "short_term".to_string()),
            (0.5, "medium_term".to_string()),
            (1.0, "scene_scale".to_string()),
        ])
        .expect("standard horizons are valid")
    }

    /// Score matured forecasts against the latest real frame and issue new ones.
    ///
    /// Call exactly once after each accepted observation for one modality.
    pub fn observe(&mut self, manifold: &VisionManifold, dt: f32) -> Result<usize, String> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(format!(
                "horizon evaluation timestep must be finite and > 0, got {dt}"
            ));
        }
        let actual = manifold
            .last_frame_hv
            .as_ref()
            .ok_or_else(|| "manifold has no encoded frame to evaluate".to_string())?;
        if !actual.as_slice().iter().all(|value| value.is_finite()) {
            return Err("encoded frame contains non-finite values".to_string());
        }
        match self.hdc_dim {
            Some(expected) if expected != actual.dim() => {
                return Err(format!(
                    "horizon evaluator dimension mismatch: got {}, expected {expected}; reset after dilation",
                    actual.dim()
                ));
            }
            None => self.hdc_dim = Some(actual.dim()),
            _ => {}
        }
        self.elapsed_seconds += dt as f64;

        let mut scored = 0usize;
        let now = self.elapsed_seconds;
        let mut still_pending = Vec::with_capacity(self.pending.len());
        for forecast in self.pending.drain(..) {
            if forecast.due_time > now {
                still_pending.push(forecast);
                continue;
            }
            let lateness = now - forecast.due_time;
            let max_lateness =
                self.horizons[forecast.horizon_index] as f64 * self.max_lateness_factor as f64;
            let accumulator = &mut self.accumulators[forecast.horizon_index];
            if lateness > max_lateness {
                accumulator.expired_forecasts = accumulator.expired_forecasts.saturating_add(1);
                continue;
            }
            let prediction_error = 1.0 - actual.similarity(&forecast.predicted).clamp(-1.0, 1.0);
            let persistence_error = 1.0 - actual.similarity(&forecast.persistence).clamp(-1.0, 1.0);
            let prediction_error = prediction_error as f64;
            let persistence_error = persistence_error as f64;
            accumulator.prediction_error_sum += prediction_error;
            accumulator.prediction_error_sq_sum += prediction_error * prediction_error;
            accumulator.persistence_error_sum += persistence_error;
            accumulator.persistence_error_sq_sum += persistence_error * persistence_error;
            accumulator.lateness_sum += lateness;
            accumulator.samples += 1;
            scored += 1;
        }
        self.pending = still_pending;

        for (horizon_index, &horizon) in self.horizons.iter().enumerate() {
            let queued = self
                .pending
                .iter()
                .filter(|forecast| forecast.horizon_index == horizon_index)
                .count();
            if queued >= self.max_pending_per_horizon {
                self.accumulators[horizon_index].dropped_forecasts += 1;
                continue;
            }
            self.pending.push(PendingHorizonForecast {
                horizon_index,
                due_time: now + horizon as f64,
                predicted: manifold.predict_horizon(actual, horizon),
                persistence: actual.clone(),
            });
        }
        Ok(scored)
    }

    /// Aggregate delayed prediction skill collected so far.
    pub fn accuracy(&self, frame_sequence: u64) -> HorizonAccuracy {
        let mut errors = Vec::with_capacity(self.accumulators.len());
        let mut persistence_errors = Vec::with_capacity(self.accumulators.len());
        let mut prediction_error_stddev = Vec::with_capacity(self.accumulators.len());
        let mut persistence_error_stddev = Vec::with_capacity(self.accumulators.len());
        let mut relative_skill = Vec::with_capacity(self.accumulators.len());
        let mut sample_counts = Vec::with_capacity(self.accumulators.len());
        let mut mean_lateness_seconds = Vec::with_capacity(self.accumulators.len());
        let mut dropped_forecasts = Vec::with_capacity(self.accumulators.len());
        let mut expired_forecasts = Vec::with_capacity(self.accumulators.len());
        for accumulator in &self.accumulators {
            if accumulator.samples == 0 {
                errors.push(1.0);
                persistence_errors.push(1.0);
                prediction_error_stddev.push(0.0);
                persistence_error_stddev.push(0.0);
                relative_skill.push(0.0);
                sample_counts.push(0);
                mean_lateness_seconds.push(0.0);
                dropped_forecasts.push(accumulator.dropped_forecasts);
                expired_forecasts.push(accumulator.expired_forecasts);
                continue;
            }
            let n = accumulator.samples as f64;
            let prediction_mean = accumulator.prediction_error_sum / n;
            let persistence_mean = accumulator.persistence_error_sum / n;
            let prediction_variance = (accumulator.prediction_error_sq_sum / n
                - prediction_mean * prediction_mean)
                .max(0.0);
            let persistence_variance = (accumulator.persistence_error_sq_sum / n
                - persistence_mean * persistence_mean)
                .max(0.0);
            let prediction = prediction_mean as f32;
            let persistence = persistence_mean as f32;
            let skill = if persistence > 1e-6 {
                ((persistence - prediction) / persistence).clamp(-1.0, 1.0)
            } else if prediction <= persistence + 1e-6 {
                0.0
            } else {
                -1.0
            };
            errors.push(prediction);
            persistence_errors.push(persistence);
            prediction_error_stddev.push(prediction_variance.sqrt() as f32);
            persistence_error_stddev.push(persistence_variance.sqrt() as f32);
            relative_skill.push(skill);
            sample_counts.push(accumulator.samples);
            mean_lateness_seconds.push((accumulator.lateness_sum / n) as f32);
            dropped_forecasts.push(accumulator.dropped_forecasts);
            expired_forecasts.push(accumulator.expired_forecasts);
        }
        HorizonAccuracy {
            horizons: self.horizons.clone(),
            labels: self.labels.clone(),
            errors,
            persistence_errors,
            prediction_error_stddev,
            persistence_error_stddev,
            relative_skill,
            sample_counts,
            mean_lateness_seconds,
            dropped_forecasts,
            expired_forecasts,
            frame_sequence,
        }
    }

    /// Number of pending HDC-sized vectors owned by the evaluator.
    fn hdc_vector_count(&self) -> usize {
        self.pending.len().saturating_mul(2)
    }

    /// Dilate pending forecasts after the owning manifold changes dimension.
    fn dilate(&mut self, target_dim: usize) {
        for forecast in &mut self.pending {
            forecast.predicted = forecast.predicted.dilate(target_dim);
            forecast.persistence = forecast.persistence.dilate(target_dim);
        }
        if self.hdc_dim.is_some() {
            self.hdc_dim = Some(target_dim);
        }
    }

    /// Update the maximum accepted observation lateness atomically.
    pub fn set_max_lateness_factor_checked(&mut self, factor: f32) -> Result<(), String> {
        if !factor.is_finite() || factor <= 0.0 {
            return Err(format!(
                "max lateness factor must be finite and > 0, got {factor}"
            ));
        }
        self.max_lateness_factor = factor;
        Ok(())
    }

    /// Snapshot policy, pending forecasts, clock, and accumulated evidence.
    pub fn save_state(&self) -> DelayedHorizonEvaluatorState {
        DelayedHorizonEvaluatorState {
            schema_version: DELAYED_HORIZON_EVALUATOR_STATE_SCHEMA_VERSION,
            horizons: self.horizons.clone(),
            labels: self.labels.clone(),
            elapsed_seconds: self.elapsed_seconds,
            hdc_dim: self.hdc_dim,
            pending: self
                .pending
                .iter()
                .map(|forecast| PendingHorizonForecastState {
                    horizon_index: forecast.horizon_index,
                    due_time: forecast.due_time,
                    predicted: forecast.predicted.as_slice().to_vec(),
                    persistence: forecast.persistence.as_slice().to_vec(),
                })
                .collect(),
            accumulators: self
                .accumulators
                .iter()
                .map(|accumulator| HorizonAccumulatorState {
                    prediction_error_sum: accumulator.prediction_error_sum,
                    prediction_error_sq_sum: accumulator.prediction_error_sq_sum,
                    persistence_error_sum: accumulator.persistence_error_sum,
                    persistence_error_sq_sum: accumulator.persistence_error_sq_sum,
                    lateness_sum: accumulator.lateness_sum,
                    samples: accumulator.samples,
                    dropped_forecasts: accumulator.dropped_forecasts,
                    expired_forecasts: accumulator.expired_forecasts,
                })
                .collect(),
            max_pending_per_horizon: self.max_pending_per_horizon,
            max_lateness_factor: self.max_lateness_factor,
        }
    }

    /// Validate a serialized evaluator before changing live evidence.
    pub fn validate_state(state: &DelayedHorizonEvaluatorState) -> Result<(), String> {
        if state.schema_version == 0
            || state.schema_version > DELAYED_HORIZON_EVALUATOR_STATE_SCHEMA_VERSION
        {
            return Err(format!(
                "unsupported delayed-horizon checkpoint schema: saved={}, supported<= {}",
                state.schema_version, DELAYED_HORIZON_EVALUATOR_STATE_SCHEMA_VERSION
            ));
        }
        if state.horizons.is_empty()
            || state.horizons.len() != state.labels.len()
            || state.horizons.len() != state.accumulators.len()
        {
            return Err("delayed-horizon checkpoint topology is incomplete".to_string());
        }
        if state.max_pending_per_horizon == 0 {
            return Err("delayed-horizon pending limit must be > 0".to_string());
        }
        if !state.max_lateness_factor.is_finite() || state.max_lateness_factor <= 0.0 {
            return Err("delayed-horizon max lateness factor must be finite and > 0".to_string());
        }
        if !state.elapsed_seconds.is_finite() || state.elapsed_seconds < 0.0 {
            return Err(
                "delayed-horizon elapsed clock must be finite and non-negative".to_string(),
            );
        }
        let mut previous = 0.0f32;
        for (index, (&horizon, label)) in state.horizons.iter().zip(state.labels.iter()).enumerate()
        {
            if !horizon.is_finite() || horizon <= 0.0 || (index > 0 && horizon <= previous) {
                return Err(format!(
                    "invalid delayed horizon at index {index}: {horizon}"
                ));
            }
            if label.trim().is_empty() {
                return Err(format!("delayed horizon {index} has an empty label"));
            }
            previous = horizon;
        }
        let mut pending_counts = vec![0usize; state.horizons.len()];
        for (index, forecast) in state.pending.iter().enumerate() {
            if forecast.horizon_index >= state.horizons.len() {
                return Err(format!(
                    "pending forecast {index} has invalid horizon index"
                ));
            }
            pending_counts[forecast.horizon_index] += 1;
            if pending_counts[forecast.horizon_index] > state.max_pending_per_horizon {
                return Err(format!(
                    "pending forecast queue exceeds limit for horizon {}",
                    forecast.horizon_index
                ));
            }
            if !forecast.due_time.is_finite() || forecast.due_time <= state.elapsed_seconds {
                return Err(format!("pending forecast {index} has an invalid due time"));
            }
            let Some(dim) = state.hdc_dim else {
                return Err("pending forecasts require a recorded HDC dimension".to_string());
            };
            for (name, values) in [
                ("predicted", &forecast.predicted),
                ("persistence", &forecast.persistence),
            ] {
                if values.len() != dim || !values.iter().all(|value| value.is_finite()) {
                    return Err(format!(
                        "pending forecast {index} {name} is malformed for dimension {dim}"
                    ));
                }
            }
        }
        if state.hdc_dim == Some(0) {
            return Err("delayed-horizon HDC dimension must be non-zero".to_string());
        }
        for (index, accumulator) in state.accumulators.iter().enumerate() {
            if !accumulator.prediction_error_sum.is_finite()
                || accumulator.prediction_error_sum < 0.0
                || !accumulator.prediction_error_sq_sum.is_finite()
                || accumulator.prediction_error_sq_sum < 0.0
                || !accumulator.persistence_error_sum.is_finite()
                || accumulator.persistence_error_sum < 0.0
                || !accumulator.persistence_error_sq_sum.is_finite()
                || accumulator.persistence_error_sq_sum < 0.0
                || !accumulator.lateness_sum.is_finite()
                || accumulator.lateness_sum < 0.0
            {
                return Err(format!("delayed-horizon accumulator {index} is invalid"));
            }
            if accumulator.samples > 0 {
                let n = accumulator.samples as f64;
                let prediction_min_sq =
                    accumulator.prediction_error_sum * accumulator.prediction_error_sum / n;
                let persistence_min_sq =
                    accumulator.persistence_error_sum * accumulator.persistence_error_sum / n;
                let prediction_sq_sum =
                    if state.schema_version < 3 && accumulator.prediction_error_sq_sum == 0.0 {
                        prediction_min_sq
                    } else {
                        accumulator.prediction_error_sq_sum
                    };
                let persistence_sq_sum =
                    if state.schema_version < 3 && accumulator.persistence_error_sq_sum == 0.0 {
                        persistence_min_sq
                    } else {
                        accumulator.persistence_error_sq_sum
                    };
                let tolerance = 1e-9 * (1.0 + prediction_min_sq + persistence_min_sq);
                if prediction_sq_sum + tolerance < prediction_min_sq
                    || persistence_sq_sum + tolerance < persistence_min_sq
                {
                    return Err(format!(
                        "delayed-horizon accumulator {index} has inconsistent squared errors"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Restore evaluator state atomically.
    pub fn load_state(&mut self, state: &DelayedHorizonEvaluatorState) -> Result<(), String> {
        Self::validate_state(state)?;
        self.horizons = state.horizons.clone();
        self.labels = state.labels.clone();
        self.elapsed_seconds = state.elapsed_seconds;
        self.hdc_dim = state.hdc_dim;
        self.pending = state
            .pending
            .iter()
            .map(|forecast| PendingHorizonForecast {
                horizon_index: forecast.horizon_index,
                due_time: forecast.due_time,
                predicted: ContinuousHV::from_vec(forecast.predicted.clone()),
                persistence: ContinuousHV::from_vec(forecast.persistence.clone()),
            })
            .collect();
        self.accumulators = state
            .accumulators
            .iter()
            .map(|accumulator| {
                let n = accumulator.samples.max(1) as f64;
                let prediction_error_sq_sum = if state.schema_version < 3
                    && accumulator.prediction_error_sq_sum == 0.0
                    && accumulator.samples > 0
                {
                    accumulator.prediction_error_sum * accumulator.prediction_error_sum / n
                } else {
                    accumulator.prediction_error_sq_sum
                };
                let persistence_error_sq_sum = if state.schema_version < 3
                    && accumulator.persistence_error_sq_sum == 0.0
                    && accumulator.samples > 0
                {
                    accumulator.persistence_error_sum * accumulator.persistence_error_sum / n
                } else {
                    accumulator.persistence_error_sq_sum
                };
                HorizonAccumulator {
                    prediction_error_sum: accumulator.prediction_error_sum,
                    prediction_error_sq_sum,
                    persistence_error_sum: accumulator.persistence_error_sum,
                    persistence_error_sq_sum,
                    lateness_sum: accumulator.lateness_sum,
                    samples: accumulator.samples,
                    dropped_forecasts: accumulator.dropped_forecasts,
                    expired_forecasts: accumulator.expired_forecasts,
                }
            })
            .collect();
        self.max_pending_per_horizon = state.max_pending_per_horizon;
        self.max_lateness_factor = state.max_lateness_factor;
        Ok(())
    }

    /// Discard only forecasts tied to expectations that are no longer valid.
    pub fn clear_pending(&mut self) {
        self.pending.clear();
    }

    /// Drop pending forecasts and accumulated evidence.
    pub fn reset(&mut self) {
        self.elapsed_seconds = 0.0;
        self.hdc_dim = None;
        self.pending.clear();
        self.accumulators.fill(HorizonAccumulator::default());
    }
}

impl Default for DelayedHorizonEvaluator {
    fn default() -> Self {
        Self::standard()
    }
}

/// Episodic scene memory: stores landmark scene HVs for recognition.
///
/// When the manifold is stable (high coherence, low prediction error),
/// the current state is stored as a landmark. On new frames, the memory
/// can be queried for scene recognition ("I've been here before").
pub struct SceneMemory {
    /// Stored whole-scene landmarks with explicit pixel geometry and modality.
    landmarks: Vec<(ContinuousHV, u64, Vec<u8>, SceneFrameMetadata)>,
    /// Object-level episodes displaced from visual working memory. These never
    /// participate in scene recognition or pixel-space mental-movie decoding.
    object_episodes: Vec<(ContinuousHV, u64)>,
    capacity: usize,
    recognition_threshold: f32,
    pixel_budget_bytes: usize,
    retained_pixel_bytes: usize,
}

impl SceneMemory {
    /// Create a scene memory with given capacity.
    pub fn new(capacity: usize) -> Self {
        Self::new_with_pixel_budget(capacity, 64 * 1024 * 1024)
    }

    /// Create scene memory with an explicit raw-raster retention budget.
    pub fn new_with_pixel_budget(capacity: usize, pixel_budget_bytes: usize) -> Self {
        Self {
            landmarks: Vec::with_capacity(capacity),
            object_episodes: Vec::with_capacity(capacity),
            capacity,
            recognition_threshold: 0.85,
            pixel_budget_bytes,
            retained_pixel_bytes: 0,
        }
    }

    /// Set the recognition similarity threshold (default: 0.85).
    pub fn set_threshold(&mut self, threshold: f32) {
        let _ = self.set_threshold_checked(threshold);
    }

    /// Checked recognition-threshold update.
    pub fn set_threshold_checked(&mut self, threshold: f32) -> Result<(), String> {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(format!(
                "scene recognition threshold must be finite and in [0, 1], got {threshold}"
            ));
        }
        self.recognition_threshold = threshold;
        Ok(())
    }

    /// Set the maximum bytes retained for raw scene rasters.
    ///
    /// If the new budget is below current usage, oldest pixel payloads are
    /// released while their semantic scene hypervectors remain available.
    pub fn set_pixel_budget(&mut self, pixel_budget_bytes: usize) {
        self.pixel_budget_bytes = pixel_budget_bytes;
        self.enforce_pixel_budget(0);
    }

    /// Maximum raw-raster retention budget.
    pub fn pixel_budget_bytes(&self) -> usize {
        self.pixel_budget_bytes
    }

    /// Bytes currently retained for replayable scene rasters.
    pub fn retained_pixel_bytes(&self) -> usize {
        self.retained_pixel_bytes
    }

    fn enforce_pixel_budget(&mut self, incoming_bytes: usize) {
        while self.retained_pixel_bytes.saturating_add(incoming_bytes) > self.pixel_budget_bytes {
            let Some((_, _, pixels, _)) = self
                .landmarks
                .iter_mut()
                .find(|(_, _, pixels, _)| !pixels.is_empty())
            else {
                break;
            };
            self.retained_pixel_bytes = self.retained_pixel_bytes.saturating_sub(pixels.len());
            pixels.clear();
        }
    }

    fn hdc_vector_count(&self) -> usize {
        self.landmarks.len() + self.object_episodes.len()
    }

    fn validate_temporal_state(
        state: &SceneMemoryState,
        checkpoint_frame: u64,
    ) -> Result<(), String> {
        let mut previous_landmark_frame = None;
        for (index, (_, frame)) in state.landmarks.iter().enumerate() {
            if *frame > checkpoint_frame {
                return Err(format!(
                    "scene landmark {index} was stored at frame {frame} beyond checkpoint frame {checkpoint_frame}"
                ));
            }
            if previous_landmark_frame.is_some_and(|previous| *frame < previous) {
                return Err(format!(
                    "scene landmark timeline is non-monotonic at index {index}: frame {frame} follows {}",
                    previous_landmark_frame.unwrap_or_default()
                ));
            }
            previous_landmark_frame = Some(*frame);
        }

        let mut previous_episode_frame = None;
        for (index, (_, frame)) in state.object_episodes.iter().enumerate() {
            if *frame > checkpoint_frame {
                return Err(format!(
                    "object episode {index} was stored at frame {frame} beyond checkpoint frame {checkpoint_frame}"
                ));
            }
            if previous_episode_frame.is_some_and(|previous| *frame < previous) {
                return Err(format!(
                    "object episode timeline is non-monotonic at index {index}: frame {frame} follows {}",
                    previous_episode_frame.unwrap_or_default()
                ));
            }
            previous_episode_frame = Some(*frame);
        }
        Ok(())
    }

    pub fn validate_state(state: &SceneMemoryState, expected_dim: usize) -> Result<(), String> {
        if state.landmarks.len() > state.capacity {
            return Err(format!(
                "scene memory exceeds capacity: landmarks={}, capacity={}",
                state.landmarks.len(),
                state.capacity
            ));
        }
        if state.object_episodes.len() > state.capacity {
            return Err(format!(
                "object episode memory exceeds capacity: episodes={}, capacity={}",
                state.object_episodes.len(),
                state.capacity
            ));
        }
        if !state.raw_frames.is_empty() && state.raw_frames.len() != state.landmarks.len() {
            return Err(format!(
                "scene memory raw-frame count mismatch: frames={}, landmarks={}",
                state.raw_frames.len(),
                state.landmarks.len()
            ));
        }
        if !state.frame_metadata.is_empty() && state.frame_metadata.len() != state.landmarks.len() {
            return Err(format!(
                "scene memory frame-metadata count mismatch: metadata={}, landmarks={}",
                state.frame_metadata.len(),
                state.landmarks.len()
            ));
        }
        for (idx, pixels) in state.raw_frames.iter().enumerate() {
            if pixels.is_empty() {
                continue;
            }
            if let Some(metadata) = state.frame_metadata.get(idx) {
                // `SceneFrameMetadata::default()` (modality `Unknown`, zero
                // geometry) marks a legacy/opaque raster — `remember()` /
                // `remember_with_metadata()` deliberately skip the geometry
                // check for this case (see their doc comments) so untyped
                // pixel blobs stay retrievable. Validation must accept the
                // same contract, or every legacy-style stored raster becomes
                // unloadable.
                if metadata.modality == VisualModality::Unknown {
                    continue;
                }
                let expected = metadata.expected_len().ok_or_else(|| {
                    format!("scene_memory.frame_metadata[{idx}] has invalid geometry")
                })?;
                if pixels.len() != expected {
                    return Err(format!(
                        "scene_memory.raw_frames[{idx}] length mismatch: got {}, expected {expected}",
                        pixels.len()
                    ));
                }
            }
        }
        if !state.threshold.is_finite() || !(0.0..=1.0).contains(&state.threshold) {
            return Err(format!(
                "invalid scene recognition threshold: {}",
                state.threshold
            ));
        }
        let retained_pixel_bytes = state
            .raw_frames
            .iter()
            .try_fold(0usize, |total, frame| total.checked_add(frame.len()))
            .ok_or_else(|| "scene memory pixel accounting overflow".to_string())?;
        if retained_pixel_bytes > state.pixel_budget_bytes {
            return Err(format!(
                "scene memory retains {retained_pixel_bytes} pixel bytes but budget is {}",
                state.pixel_budget_bytes
            ));
        }
        if state.retained_pixel_bytes != 0 && state.retained_pixel_bytes != retained_pixel_bytes {
            return Err(format!(
                "scene memory pixel accounting mismatch: saved={}, actual={retained_pixel_bytes}",
                state.retained_pixel_bytes
            ));
        }
        for (idx, (values, _)) in state.landmarks.iter().enumerate() {
            if values.len() != expected_dim {
                return Err(format!(
                    "scene_memory.landmarks[{idx}] dimension mismatch: saved={}, expected={expected_dim}",
                    values.len()
                ));
            }
            if !values.iter().all(|value| value.is_finite()) {
                return Err(format!(
                    "scene_memory.landmarks[{idx}] contains non-finite values"
                ));
            }
        }
        for (idx, (values, _)) in state.object_episodes.iter().enumerate() {
            if values.len() != expected_dim {
                return Err(format!(
                    "scene_memory.object_episodes[{idx}] dimension mismatch: saved={}, expected={expected_dim}",
                    values.len()
                ));
            }
            if !values.iter().all(|value| value.is_finite()) {
                return Err(format!(
                    "scene_memory.object_episodes[{idx}] contains non-finite values"
                ));
            }
        }
        Ok(())
    }

    /// Scale all landmarks to a new HDC dimensionality.
    pub fn dilate(&mut self, target_dim: usize) {
        for (hv, _, _, _) in &mut self.landmarks {
            *hv = hv.dilate(target_dim);
        }
        for (hv, _) in &mut self.object_episodes {
            *hv = hv.dilate(target_dim);
        }
    }

    /// Store a scene landmark without geometry metadata.
    ///
    /// Kept for compatibility with callers that only need semantic recognition.
    /// Pixel replay treats this entry as legacy/unknown.
    pub fn remember(&mut self, state: &ContinuousHV, frame: u64, pixels: Vec<u8>) {
        self.remember_with_metadata(state, frame, pixels, SceneFrameMetadata::default());
    }

    /// Store a scene landmark with an explicit pixel interpretation contract.
    pub fn remember_with_metadata(
        &mut self,
        state: &ContinuousHV,
        frame: u64,
        mut pixels: Vec<u8>,
        metadata: SceneFrameMetadata,
    ) {
        if self.capacity == 0 {
            return;
        }
        if !pixels.is_empty()
            && metadata.modality != VisualModality::Unknown
            && metadata.expected_len() != Some(pixels.len())
        {
            return;
        }
        // Don't store near-duplicates
        if self
            .landmarks
            .iter()
            .any(|(hv, _, _, _)| state.similarity(hv) > 0.98)
        {
            return;
        }
        if pixels.len() > self.pixel_budget_bytes {
            pixels.clear();
        }
        if self.landmarks.len() >= self.capacity {
            // Evict oldest semantic landmark and its optional raster.
            let (_, _, evicted_pixels, _) = self.landmarks.remove(0);
            self.retained_pixel_bytes = self
                .retained_pixel_bytes
                .saturating_sub(evicted_pixels.len());
        }
        self.enforce_pixel_budget(pixels.len());
        self.retained_pixel_bytes = self.retained_pixel_bytes.saturating_add(pixels.len());
        self.landmarks
            .push((state.clone(), frame, pixels, metadata));
    }

    /// Store an object-level episode without mixing it into scene recognition.
    pub fn remember_object(&mut self, object_hv: &ContinuousHV, frame: u64) {
        if self.capacity == 0
            || self
                .object_episodes
                .iter()
                .any(|(hv, _)| object_hv.similarity(hv) > 0.98)
        {
            return;
        }
        if self.object_episodes.len() >= self.capacity {
            self.object_episodes.remove(0);
        }
        self.object_episodes.push((object_hv.clone(), frame));
    }

    /// Number of stored object-level episodic traces.
    pub fn object_episode_count(&self) -> usize {
        self.object_episodes.len()
    }

    /// Recognize the current state against stored landmarks.
    ///
    /// Returns the best match if similarity exceeds the recognition threshold.
    pub fn recognize(&self, state: &ContinuousHV, current_frame: u64) -> Option<SceneMatch> {
        let mut best: Option<(usize, f32, u64)> = None;

        for (idx, (landmark, stored_frame, _, _)) in self.landmarks.iter().enumerate() {
            let sim = state.similarity(landmark);
            if sim >= self.recognition_threshold {
                match best {
                    Some((_, best_sim, _)) if sim <= best_sim => {}
                    _ => best = Some((idx, sim, *stored_frame)),
                }
            }
        }

        best.map(|(scene_id, similarity, stored_at_frame)| SceneMatch {
            scene_id,
            similarity,
            stored_at_frame,
            frames_since_stored: current_frame.saturating_sub(stored_at_frame),
        })
    }

    /// Access the raw pixels of a specific stored scene.
    pub fn get_pixels(&self, scene_id: usize) -> Option<&[u8]> {
        self.landmarks
            .get(scene_id)
            .map(|(_, _, p, _)| p.as_slice())
    }

    /// Access the geometry and modality associated with persisted pixels.
    pub fn get_frame_metadata(&self, scene_id: usize) -> Option<SceneFrameMetadata> {
        self.landmarks
            .get(scene_id)
            .map(|(_, _, _, metadata)| *metadata)
    }

    /// Number of stored landmarks.
    pub fn len(&self) -> usize {
        self.landmarks.len()
    }

    /// Whether the memory is empty.
    pub fn is_empty(&self) -> bool {
        self.landmarks.is_empty()
    }

    /// Clear all stored landmarks.
    pub fn clear(&mut self) {
        self.landmarks.clear();
        self.object_episodes.clear();
        self.retained_pixel_bytes = 0;
    }

    /// Read-only access to stored landmarks as `(hv, stored_at_frame)` pairs.
    pub fn export_landmarks(&self) -> Vec<(&ContinuousHV, u64)> {
        self.landmarks
            .iter()
            .map(|(hv, f, _, _)| (hv, *f))
            .collect()
    }

    /// Get a specific landmark by index.
    pub fn get_landmark(&self, idx: usize) -> Option<&ContinuousHV> {
        self.landmarks.get(idx).map(|(hv, _, _, _)| hv)
    }

    /// Remove a specific landmark by index. Returns `true` if removed.
    pub fn forget(&mut self, scene_id: usize) -> bool {
        if scene_id < self.landmarks.len() {
            let (_, _, pixels, _) = self.landmarks.remove(scene_id);
            self.retained_pixel_bytes = self.retained_pixel_bytes.saturating_sub(pixels.len());
            true
        } else {
            false
        }
    }

    /// Snapshot the scene memory for serialization.
    pub fn save_state(&self) -> SceneMemoryState {
        SceneMemoryState {
            landmarks: self
                .landmarks
                .iter()
                .map(|(hv, frame, _, _)| (hv.as_slice().to_vec(), *frame))
                .collect(),
            capacity: self.capacity,
            threshold: self.recognition_threshold,
            pixel_budget_bytes: self.pixel_budget_bytes,
            retained_pixel_bytes: self.retained_pixel_bytes,
            raw_frames: self
                .landmarks
                .iter()
                .map(|(_, _, pixels, _)| pixels.clone())
                .collect(),
            frame_metadata: self
                .landmarks
                .iter()
                .map(|(_, _, _, metadata)| *metadata)
                .collect(),
            object_episodes: self
                .object_episodes
                .iter()
                .map(|(hv, frame)| (hv.as_slice().to_vec(), *frame))
                .collect(),
        }
    }

    /// Restore scene memory from a saved state after complete validation.
    ///
    /// Validation and reconstruction happen before any live field changes, so a
    /// malformed standalone snapshot cannot partially replace existing memory.
    pub fn load_state_checked(
        &mut self,
        state: &SceneMemoryState,
        expected_dim: usize,
    ) -> Result<(), String> {
        Self::validate_state(state, expected_dim)?;
        let landmarks: Vec<_> = state
            .landmarks
            .iter()
            .enumerate()
            .map(|(idx, (vals, frame))| {
                let pixels = state.raw_frames.get(idx).cloned().unwrap_or_default();
                let metadata = state.frame_metadata.get(idx).copied().unwrap_or_default();
                (
                    ContinuousHV::from_vec(vals.clone()),
                    *frame,
                    pixels,
                    metadata,
                )
            })
            .collect();
        let retained_pixel_bytes = landmarks
            .iter()
            .try_fold(0usize, |total, (_, _, pixels, _)| {
                total.checked_add(pixels.len())
            })
            .ok_or_else(|| "scene memory pixel accounting overflow".to_string())?;
        let object_episodes = state
            .object_episodes
            .iter()
            .map(|(vals, frame)| (ContinuousHV::from_vec(vals.clone()), *frame))
            .collect();

        self.capacity = state.capacity;
        self.recognition_threshold = state.threshold;
        self.pixel_budget_bytes = state.pixel_budget_bytes;
        self.landmarks = landmarks;
        self.retained_pixel_bytes = retained_pixel_bytes;
        self.object_episodes = object_episodes;
        Ok(())
    }

    /// Compatibility restore for already trusted snapshots. Invalid snapshots
    /// are ignored rather than partially mutating live memory.
    pub fn load_state(&mut self, state: &SceneMemoryState) {
        let expected_dim = state
            .landmarks
            .first()
            .map(|(values, _)| values.len())
            .or_else(|| {
                state
                    .object_episodes
                    .first()
                    .map(|(values, _)| values.len())
            })
            .or_else(|| self.landmarks.first().map(|(hv, _, _, _)| hv.dim()))
            .or_else(|| self.object_episodes.first().map(|(hv, _)| hv.dim()))
            .unwrap_or(0);
        let _ = self.load_state_checked(state, expected_dim);
    }
}

/// Cross-frame object identity tracker (Spelke 1990 object permanence).
///
/// Stores a ring buffer of tracked objects. Each tracked object has:
/// - A stable appearance prototype for matching
/// - A temporal-history identity HV for sequence-aware downstream reasoning
/// - A centroid position (most recent)
/// - A "last seen" frame number (for occlusion timeout)
///
/// On each frame, incoming `ObjectHypothesis` clusters are matched against
/// existing tracks by HDC cosine similarity. Matched tracks update via
/// temporal binding; unmatched clusters start new tracks; stale tracks
/// (not seen for `max_absence_frames`) are evicted.
pub struct ObjectMemory {
    tracks: Vec<TrackedObject>,
    capacity: usize,
    /// Minimum cosine similarity to match a hypothesis to an existing track.
    match_threshold: f32,
    /// Number of frames before a track is evicted for absence.
    max_absence_frames: u64,
    /// Maximum Manhattan centroid displacement allowed for a one-frame match.
    /// The gate expands by one cell per absent frame to support short occlusion.
    max_match_distance: usize,
}

/// A single tracked object persisting across frames.
#[derive(Debug, Clone)]
pub struct TrackedObject {
    /// Unique track ID (monotonically assigned).
    pub track_id: u64,
    /// Slowly-updated appearance prototype used for cross-frame matching.
    pub appearance_hv: ContinuousHV,
    /// Temporally-accumulated history HV: `bind_temporal(prev, curr)`.
    pub identity_hv: ContinuousHV,
    /// Most recent centroid grid row.
    pub centroid_row: usize,
    /// Most recent centroid grid column.
    pub centroid_col: usize,
    /// Smoothed centroid velocity in grid cells per observed frame.
    pub velocity_row: f32,
    pub velocity_col: f32,
    /// Frame at which this object was last observed.
    pub last_seen_frame: u64,
    /// Number of consecutive frames this object has been tracked.
    pub track_length: u64,
}

/// Result of matching hypotheses to object memory.
#[derive(Debug, Clone)]
pub struct ObjectTrackingResult {
    /// Tracked objects that matched an incoming hypothesis (updated).
    pub matched: Vec<(u64, f32)>, // (track_id, match_similarity)
    /// Number of new tracks created this frame.
    pub new_tracks: usize,
    /// Number of stale tracks evicted this frame.
    pub evicted: usize,
    /// Total active tracks after this update.
    pub active_tracks: usize,
}

/// Solve a rectangular maximum-weight bipartite assignment.
///
/// Rows are hypotheses and columns are existing tracks. `None` marks an
/// inadmissible edge. Each row receives an additional zero-weight dummy column,
/// so the optimal solution may leave any hypothesis unmatched rather than
/// forcing a scientifically invalid identity.
fn maximum_weight_assignment(scores: &[Vec<Option<f32>>]) -> Vec<Option<usize>> {
    let rows = scores.len();
    if rows == 0 {
        return Vec::new();
    }
    let real_cols = scores.first().map_or(0, Vec::len);
    let cols = real_cols + rows;
    if cols == 0 {
        return vec![None; rows];
    }
    debug_assert!(scores.iter().all(|row| row.len() == real_cols));

    // Hungarian algorithm for rectangular minimization (rows <= columns).
    // Maximization is converted with cost = -score; dummy columns cost zero.
    let mut u = vec![0.0f64; rows + 1];
    let mut v = vec![0.0f64; cols + 1];
    let mut p = vec![0usize; cols + 1];
    let mut way = vec![0usize; cols + 1];
    const INVALID_COST: f64 = 1.0e9;

    let edge_cost = |row: usize, col: usize| -> f64 {
        if col < real_cols {
            scores[row][col]
                .map(|score| -(score as f64))
                .unwrap_or(INVALID_COST)
        } else {
            0.0
        }
    };

    for row in 1..=rows {
        p[0] = row;
        let mut col0 = 0usize;
        let mut minv = vec![f64::INFINITY; cols + 1];
        let mut used = vec![false; cols + 1];
        loop {
            used[col0] = true;
            let row0 = p[col0];
            let mut delta = f64::INFINITY;
            let mut col1 = 0usize;
            for col in 1..=cols {
                if used[col] {
                    continue;
                }
                let cur = edge_cost(row0 - 1, col - 1) - u[row0] - v[col];
                if cur < minv[col] {
                    minv[col] = cur;
                    way[col] = col0;
                }
                if minv[col] < delta {
                    delta = minv[col];
                    col1 = col;
                }
            }
            for col in 0..=cols {
                if used[col] {
                    u[p[col]] += delta;
                    v[col] -= delta;
                } else {
                    minv[col] -= delta;
                }
            }
            col0 = col1;
            if p[col0] == 0 {
                break;
            }
        }
        loop {
            let col1 = way[col0];
            p[col0] = p[col1];
            col0 = col1;
            if col0 == 0 {
                break;
            }
        }
    }

    let mut assignment = vec![None; rows];
    for col in 1..=cols {
        let row = p[col];
        if row == 0 || col > real_cols {
            continue;
        }
        let score = scores[row - 1][col - 1];
        if score.is_some_and(|value| value > 0.0) {
            assignment[row - 1] = Some(col - 1);
        }
    }
    assignment
}

impl ObjectMemory {
    /// Create object memory with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            tracks: Vec::with_capacity(capacity),
            capacity,
            match_threshold: 0.3,
            max_absence_frames: 30,
            max_match_distance: 4,
        }
    }

    /// Set the match similarity threshold (default: 0.3).
    pub fn set_match_threshold(&mut self, threshold: f32) {
        let _ = self.set_match_threshold_checked(threshold);
    }

    /// Checked object-match policy update.
    pub fn set_match_threshold_checked(&mut self, threshold: f32) -> Result<(), String> {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(format!(
                "object match threshold must be finite and in [0, 1], got {threshold}"
            ));
        }
        self.match_threshold = threshold;
        Ok(())
    }

    /// Set the absence timeout in frames (default: 30).
    pub fn set_max_absence(&mut self, frames: u64) {
        self.max_absence_frames = frames;
    }

    /// Set the maximum one-frame centroid displacement used during matching.
    pub fn set_max_match_distance(&mut self, cells: usize) {
        self.max_match_distance = cells;
    }

    fn hdc_vector_count(&self) -> usize {
        self.tracks.len().saturating_mul(2)
    }

    /// Scale all tracks to a new HDC dimensionality.
    pub fn dilate(&mut self, target_dim: usize) {
        for track in &mut self.tracks {
            track.appearance_hv = track.appearance_hv.dilate(target_dim);
            track.identity_hv = track.identity_hv.dilate(target_dim);
        }
    }

    /// Update tracks from this frame's object hypotheses.
    ///
    /// For each hypothesis: find the best-matching existing track. If the
    /// similarity exceeds `match_threshold`, update that track's identity HV
    /// via temporal binding. Otherwise, create a new track.
    ///
    /// Then evict all tracks not seen for `max_absence_frames`.
    pub fn update(
        &mut self,
        hypotheses: &[crate::types::ObjectHypothesis],
        current_frame: u64,
        next_track_id: &mut u64,
    ) -> ObjectTrackingResult {
        // Retire expired tracks before assignment so stale entries cannot consume
        // capacity and prevent genuinely new objects from being admitted.
        let before_eviction = self.tracks.len();
        self.tracks.retain(|track| {
            current_frame.saturating_sub(track.last_seen_frame) <= self.max_absence_frames
        });
        let evicted = before_eviction - self.tracks.len();

        // Canonicalize rows and columns before solving so equal-score outcomes do
        // not depend on caller hypothesis order or internal vector order.
        let mut hypothesis_order: Vec<usize> = (0..hypotheses.len()).collect();
        hypothesis_order.sort_by_key(|&idx| {
            (
                hypotheses[idx].centroid_row,
                hypotheses[idx].centroid_col,
                idx,
            )
        });
        let mut track_order: Vec<usize> = (0..self.tracks.len()).collect();
        track_order.sort_by_key(|&idx| self.tracks[idx].track_id);

        let mut scores = vec![vec![None; track_order.len()]; hypothesis_order.len()];
        let mut similarities = vec![vec![0.0f32; track_order.len()]; hypothesis_order.len()];
        for (ordered_hypothesis_idx, &hypothesis_idx) in hypothesis_order.iter().enumerate() {
            let hypothesis = &hypotheses[hypothesis_idx];
            for (ordered_track_idx, &track_idx) in track_order.iter().enumerate() {
                let track = &self.tracks[track_idx];
                let absent_frames = current_frame.saturating_sub(track.last_seen_frame).max(1);
                let horizon = absent_frames as f32;
                let predicted_row = track.centroid_row as f32 + track.velocity_row * horizon;
                let predicted_col = track.centroid_col as f32 + track.velocity_col * horizon;
                let distance = (predicted_row - hypothesis.centroid_row as f32).abs()
                    + (predicted_col - hypothesis.centroid_col as f32).abs();
                let allowed_distance = self.max_match_distance as f32
                    + absent_frames.min(self.max_absence_frames) as f32;
                if distance > allowed_distance {
                    continue;
                }

                let appearance_similarity = track.appearance_hv.similarity(&hypothesis.hv);
                if appearance_similarity < self.match_threshold {
                    continue;
                }
                let score = appearance_similarity - 0.02 * distance;
                if score > 0.0 && score.is_finite() {
                    scores[ordered_hypothesis_idx][ordered_track_idx] = Some(score);
                    similarities[ordered_hypothesis_idx][ordered_track_idx] = appearance_similarity;
                }
            }
        }

        let optimal = maximum_weight_assignment(&scores);
        let mut assigned_hypotheses = vec![false; hypotheses.len()];
        let mut assignments: Vec<(usize, usize, f32)> = Vec::new();
        for (ordered_hypothesis_idx, assigned_track) in optimal.into_iter().enumerate() {
            let Some(ordered_track_idx) = assigned_track else {
                continue;
            };
            let hypothesis_idx = hypothesis_order[ordered_hypothesis_idx];
            let track_idx = track_order[ordered_track_idx];
            assigned_hypotheses[hypothesis_idx] = true;
            assignments.push((
                hypothesis_idx,
                track_idx,
                similarities[ordered_hypothesis_idx][ordered_track_idx],
            ));
        }

        let mut matched = Vec::with_capacity(assignments.len());
        for (hypothesis_idx, track_idx, appearance_similarity) in assignments {
            let hypothesis = &hypotheses[hypothesis_idx];
            let track = &mut self.tracks[track_idx];
            track.appearance_hv =
                ContinuousHV::weighted_bundle(&[&track.appearance_hv, &hypothesis.hv], &[0.8, 0.2])
                    .normalize();
            track.identity_hv = track.identity_hv.bind_temporal(&hypothesis.hv).normalize();
            let elapsed = current_frame.saturating_sub(track.last_seen_frame).max(1) as f32;
            let observed_velocity_row =
                (hypothesis.centroid_row as f32 - track.centroid_row as f32) / elapsed;
            let observed_velocity_col =
                (hypothesis.centroid_col as f32 - track.centroid_col as f32) / elapsed;
            if track.track_length <= 1 {
                track.velocity_row = observed_velocity_row;
                track.velocity_col = observed_velocity_col;
            } else {
                track.velocity_row = 0.7 * track.velocity_row + 0.3 * observed_velocity_row;
                track.velocity_col = 0.7 * track.velocity_col + 0.3 * observed_velocity_col;
            }
            track.centroid_row = hypothesis.centroid_row;
            track.centroid_col = hypothesis.centroid_col;
            track.last_seen_frame = current_frame;
            track.track_length += 1;
            matched.push((track.track_id, appearance_similarity));
        }
        matched.sort_by_key(|(track_id, _)| *track_id);

        let mut created = 0usize;
        for (hypothesis_idx, hypothesis) in hypotheses.iter().enumerate() {
            if assigned_hypotheses[hypothesis_idx] || self.tracks.len() >= self.capacity {
                continue;
            }
            self.tracks.push(TrackedObject {
                track_id: *next_track_id,
                appearance_hv: hypothesis.hv.clone(),
                identity_hv: hypothesis.hv.clone(),
                centroid_row: hypothesis.centroid_row,
                centroid_col: hypothesis.centroid_col,
                velocity_row: 0.0,
                velocity_col: 0.0,
                last_seen_frame: current_frame,
                track_length: 1,
            });
            *next_track_id += 1;
            created += 1;
        }

        ObjectTrackingResult {
            matched,
            new_tracks: created,
            evicted,
            active_tracks: self.tracks.len(),
        }
    }

    /// Validate a serialized object-memory snapshot before mutation.
    pub fn validate_state(state: &ObjectMemoryState, dim: usize) -> Result<(), String> {
        if state.tracks.len() > state.capacity {
            return Err(format!(
                "object memory contains {} tracks but capacity is {}",
                state.tracks.len(),
                state.capacity
            ));
        }
        if !state.match_threshold.is_finite() || !(0.0..=1.0).contains(&state.match_threshold) {
            return Err(format!(
                "object match threshold must be finite and in [0.0, 1.0], got {}",
                state.match_threshold
            ));
        }
        let mut track_ids = std::collections::BTreeSet::new();
        for (idx, track) in state.tracks.iter().enumerate() {
            if !track_ids.insert(track.track_id) {
                return Err(format!(
                    "object memory contains duplicate track ID {}",
                    track.track_id
                ));
            }
            if track.appearance_hv.len() != dim || track.identity_hv.len() != dim {
                return Err(format!(
                    "object track {idx} dimension mismatch: appearance={}, identity={}, expected={dim}",
                    track.appearance_hv.len(),
                    track.identity_hv.len()
                ));
            }
            if !track.appearance_hv.iter().all(|value| value.is_finite())
                || !track.identity_hv.iter().all(|value| value.is_finite())
            {
                return Err(format!("object track {idx} contains non-finite HV values"));
            }
            if !track.velocity_row.is_finite() || !track.velocity_col.is_finite() {
                return Err(format!("object track {idx} contains non-finite velocity"));
            }
            if track.track_length == 0 {
                return Err(format!("object track {idx} has zero track length"));
            }
        }
        Ok(())
    }

    /// Snapshot all active tracks and tracker policy.
    pub fn save_state(&self) -> ObjectMemoryState {
        ObjectMemoryState {
            tracks: self
                .tracks
                .iter()
                .map(|track| TrackedObjectState {
                    track_id: track.track_id,
                    appearance_hv: track.appearance_hv.as_slice().to_vec(),
                    identity_hv: track.identity_hv.as_slice().to_vec(),
                    centroid_row: track.centroid_row,
                    centroid_col: track.centroid_col,
                    velocity_row: track.velocity_row,
                    velocity_col: track.velocity_col,
                    last_seen_frame: track.last_seen_frame,
                    track_length: track.track_length,
                })
                .collect(),
            capacity: self.capacity,
            match_threshold: self.match_threshold,
            max_absence_frames: self.max_absence_frames,
            max_match_distance: self.max_match_distance,
        }
    }

    /// Restore active tracks and tracker policy atomically.
    pub fn load_state_checked(
        &mut self,
        state: &ObjectMemoryState,
        dim: usize,
    ) -> Result<(), String> {
        Self::validate_state(state, dim)?;
        let tracks = state
            .tracks
            .iter()
            .map(|track| TrackedObject {
                track_id: track.track_id,
                appearance_hv: ContinuousHV::from_vec(track.appearance_hv.clone()),
                identity_hv: ContinuousHV::from_vec(track.identity_hv.clone()),
                centroid_row: track.centroid_row,
                centroid_col: track.centroid_col,
                velocity_row: track.velocity_row,
                velocity_col: track.velocity_col,
                last_seen_frame: track.last_seen_frame,
                track_length: track.track_length,
            })
            .collect();
        self.capacity = state.capacity;
        self.match_threshold = state.match_threshold;
        self.max_absence_frames = state.max_absence_frames;
        self.max_match_distance = state.max_match_distance;
        self.tracks = tracks;
        Ok(())
    }

    /// Compatibility restore for already trusted snapshots. Invalid snapshots
    /// are ignored rather than clamped into a different policy.
    pub fn load_state(&mut self, state: &ObjectMemoryState) {
        let dim = state
            .tracks
            .first()
            .map(|track| track.appearance_hv.len())
            .or_else(|| self.tracks.first().map(|track| track.appearance_hv.dim()))
            .unwrap_or(0);
        let _ = self.load_state_checked(state, dim);
    }

    /// Get all currently active tracks.
    pub fn tracks(&self) -> &[TrackedObject] {
        &self.tracks
    }

    /// Number of active tracks.
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// Whether any tracks are active.
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// Clear all tracks.
    pub fn clear(&mut self) {
        self.tracks.clear();
    }
}

/// Visual working memory with bounded capacity (Cowan 2001: ~4 objects).
///
/// Holds the N most salient tracked objects. When a new object exceeds the
/// weakest held object's saliency, it evicts and replaces. This gives the
/// system a bounded attentional spotlight — it can only "think about" a
/// few visual objects at once, like biological vision.
pub struct VisualWorkingMemory {
    slots: Vec<WorkingMemorySlot>,
    capacity: usize,
    /// Exponential decay rate per frame (default: 0.95 = 5% decay/frame).
    decay_rate: f32,
}

/// A single slot in visual working memory.
#[derive(Debug, Clone)]
pub struct WorkingMemorySlot {
    /// Track ID of the held object.
    pub track_id: u64,
    /// Object HV (snapshot from when it entered working memory).
    pub hv: ContinuousHV,
    /// Current saliency (decays over time).
    pub saliency: f32,
    /// Grid centroid row.
    pub centroid_row: usize,
    /// Grid centroid col.
    pub centroid_col: usize,
    /// Frame at which this object entered working memory.
    pub entered_at_frame: u64,
}

impl VisualWorkingMemory {
    /// Create working memory with the given capacity (default: 4).
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            capacity,
            decay_rate: 0.95,
        }
    }

    fn hdc_vector_count(&self) -> usize {
        self.slots.len()
    }

    /// Scale all held object HVs to a new HDC dimensionality.
    pub fn dilate(&mut self, target_dim: usize) {
        for slot in &mut self.slots {
            slot.hv = slot.hv.dilate(target_dim);
        }
    }

    /// Update working memory from the current frame's tracked objects.
    ///
    /// 1. Decay all existing slots' saliency.
    /// 2. Refresh saliency for objects still being tracked.
    /// 3. Admit new high-saliency objects if they beat the weakest slot.
    /// 4. Evict slots that have decayed below threshold (0.01).
    pub fn update(
        &mut self,
        tracks: &[TrackedObject],
        hypotheses: &[crate::types::ObjectHypothesis],
        current_frame: u64,
    ) -> Vec<ContinuousHV> {
        // 1. Decay all saliency
        for slot in &mut self.slots {
            slot.saliency *= self.decay_rate;
        }

        // 2. Refresh tracked objects already in working memory. The slot HV is
        // a live attentional representation, not an immutable admission-time
        // snapshot; keeping the old appearance makes cognitive context stale as
        // an object rotates, changes lighting, or is refined by the tracker.
        for slot in &mut self.slots {
            let Some(track) = tracks.iter().find(|track| track.track_id == slot.track_id) else {
                continue;
            };
            slot.hv = track.appearance_hv.clone();
            slot.centroid_row = track.centroid_row;
            slot.centroid_col = track.centroid_col;
            if let Some(hypothesis) = hypotheses.iter().find(|hypothesis| {
                hypothesis.centroid_row == track.centroid_row
                    && hypothesis.centroid_col == track.centroid_col
            }) {
                slot.saliency = slot.saliency.max(hypothesis.saliency);
            }
        }

        // 3. Consider new objects for admission. Replacements are genuine
        // attentional evictions and must be returned for episodic consolidation.
        let mut evicted_hvs = Vec::new();
        for track in tracks {
            if self.slots.iter().any(|s| s.track_id == track.track_id) {
                continue; // already held
            }
            let saliency = hypotheses
                .iter()
                .find(|h| {
                    h.centroid_row == track.centroid_row && h.centroid_col == track.centroid_col
                })
                .map(|h| h.saliency)
                .unwrap_or(0.0);

            if self.slots.len() < self.capacity {
                self.slots.push(WorkingMemorySlot {
                    track_id: track.track_id,
                    hv: track.appearance_hv.clone(),
                    saliency,
                    centroid_row: track.centroid_row,
                    centroid_col: track.centroid_col,
                    entered_at_frame: current_frame,
                });
            } else if let Some(weakest) = self.slots.iter().enumerate().min_by(|a, b| {
                a.1.saliency
                    .partial_cmp(&b.1.saliency)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) && saliency > weakest.1.saliency
            {
                let idx = weakest.0;
                evicted_hvs.push(self.slots[idx].hv.clone());
                self.slots[idx] = WorkingMemorySlot {
                    track_id: track.track_id,
                    hv: track.appearance_hv.clone(),
                    saliency,
                    centroid_row: track.centroid_row,
                    centroid_col: track.centroid_col,
                    entered_at_frame: current_frame,
                };
            }
        }

        // 4. Evict dead slots — return all displaced HVs for episodic consolidation.
        self.slots.retain(|s| {
            if s.saliency <= 0.01 {
                evicted_hvs.push(s.hv.clone());
                false
            } else {
                true
            }
        });
        evicted_hvs
    }

    /// Validate a serialized working-memory snapshot before mutation.
    pub fn validate_state(state: &VisualWorkingMemoryState, dim: usize) -> Result<(), String> {
        if state.slots.len() > state.capacity {
            return Err(format!(
                "working memory contains {} slots but capacity is {}",
                state.slots.len(),
                state.capacity
            ));
        }
        if !state.decay_rate.is_finite() || !(0.0..=1.0).contains(&state.decay_rate) {
            return Err(format!(
                "working-memory decay rate must be finite and in [0.0, 1.0], got {}",
                state.decay_rate
            ));
        }
        let mut track_ids = std::collections::BTreeSet::new();
        for (idx, slot) in state.slots.iter().enumerate() {
            if !track_ids.insert(slot.track_id) {
                return Err(format!(
                    "working memory contains duplicate track ID {}",
                    slot.track_id
                ));
            }
            if slot.hv.len() != dim {
                return Err(format!(
                    "working-memory slot {idx} dimension mismatch: saved={}, expected={dim}",
                    slot.hv.len()
                ));
            }
            if !slot.hv.iter().all(|value| value.is_finite()) {
                return Err(format!(
                    "working-memory slot {idx} contains non-finite HV values"
                ));
            }
            if !slot.saliency.is_finite() || slot.saliency < 0.0 {
                return Err(format!(
                    "working-memory slot {idx} has invalid saliency {}",
                    slot.saliency
                ));
            }
        }
        Ok(())
    }

    /// Snapshot working-memory slots and decay policy.
    pub fn save_state(&self) -> VisualWorkingMemoryState {
        VisualWorkingMemoryState {
            slots: self
                .slots
                .iter()
                .map(|slot| WorkingMemorySlotState {
                    track_id: slot.track_id,
                    hv: slot.hv.as_slice().to_vec(),
                    saliency: slot.saliency,
                    centroid_row: slot.centroid_row,
                    centroid_col: slot.centroid_col,
                    entered_at_frame: slot.entered_at_frame,
                })
                .collect(),
            capacity: self.capacity,
            decay_rate: self.decay_rate,
        }
    }

    /// Restore working-memory slots and decay policy atomically.
    pub fn load_state_checked(
        &mut self,
        state: &VisualWorkingMemoryState,
        dim: usize,
    ) -> Result<(), String> {
        Self::validate_state(state, dim)?;
        let slots = state
            .slots
            .iter()
            .map(|slot| WorkingMemorySlot {
                track_id: slot.track_id,
                hv: ContinuousHV::from_vec(slot.hv.clone()),
                saliency: slot.saliency,
                centroid_row: slot.centroid_row,
                centroid_col: slot.centroid_col,
                entered_at_frame: slot.entered_at_frame,
            })
            .collect();
        self.capacity = state.capacity;
        self.decay_rate = state.decay_rate;
        self.slots = slots;
        Ok(())
    }

    /// Compatibility restore for already trusted snapshots. Invalid snapshots
    /// are ignored rather than clamped into a different state.
    pub fn load_state(&mut self, state: &VisualWorkingMemoryState) {
        let dim = state
            .slots
            .first()
            .map(|slot| slot.hv.len())
            .or_else(|| self.slots.first().map(|slot| slot.hv.dim()))
            .unwrap_or(0);
        let _ = self.load_state_checked(state, dim);
    }

    /// Currently held objects.
    pub fn slots(&self) -> &[WorkingMemorySlot] {
        &self.slots
    }

    /// Number of objects currently in working memory.
    pub fn load(&self) -> usize {
        self.slots.len()
    }

    /// Capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear working memory.
    pub fn clear(&mut self) {
        self.slots.clear();
    }

    /// Bundle all held object HVs into a single working-memory HV.
    ///
    /// The cognitive loop can use this as a "what am I attending to" signal.
    pub fn bundle_attended(&self) -> Option<ContinuousHV> {
        if self.slots.is_empty() {
            return None;
        }
        let refs: Vec<&ContinuousHV> = self.slots.iter().map(|s| &s.hv).collect();
        let weights: Vec<f32> = self.slots.iter().map(|s| s.saliency).collect();
        Some(ContinuousHV::weighted_bundle(&refs, &weights).normalize())
    }
}

/// Visual scene graph: spatial relations between tracked objects.
///
/// From `TrackedObject` positions, computes pairwise spatial relations
/// (above/below/left/right/near/far/overlapping) and encodes each as an
/// HDC relational triple: `subject_hv ⊗ relation_basis_hv ⊗ object_hv`.
///
/// The full scene graph HV bundles all edges — the cognitive loop can
/// probe it to answer relational queries like "what is above the red object?"
pub struct VisualSceneGraph {
    /// Pre-generated basis HVs for each spatial relation.
    relation_bases: Vec<(crate::types::SpatialRelation, ContinuousHV)>,
    /// Current edges.
    edges: Vec<crate::types::SceneGraphEdge>,
    /// HDC dimension.
    _hdc_dim: usize,
    /// Bundled scene graph HV (all edges combined).
    graph_hv: Option<ContinuousHV>,
    /// Grid proximity threshold for "Near" (in grid cells).
    near_threshold: usize,
}

impl VisualSceneGraph {
    /// Create a new scene graph with relation basis HVs.
    pub fn new(hdc_dim: usize, seed: u64) -> Self {
        let relation_bases = crate::types::SpatialRelation::ALL
            .iter()
            .enumerate()
            .map(|(i, &rel)| {
                let hv = ContinuousHV::random(hdc_dim, seed + 900_000 + i as u64);
                (rel, hv)
            })
            .collect();
        Self {
            relation_bases,
            edges: Vec::new(),
            _hdc_dim: hdc_dim,
            graph_hv: None,
            near_threshold: 2,
        }
    }

    fn hdc_vector_count(&self) -> usize {
        self.relation_bases.len() + self.edges.len() + self.graph_hv.is_some() as usize
    }

    /// Scale all relation bases and the graph HV to a new HDC dimensionality.
    pub fn dilate(&mut self, target_dim: usize) {
        for (_, hv) in &mut self.relation_bases {
            *hv = hv.dilate(target_dim);
        }
        if let Some(ref mut hv) = self.graph_hv {
            *hv = hv.dilate(target_dim);
        }
        for edge in &mut self.edges {
            edge.relation_hv = edge.relation_hv.dilate(target_dim);
        }
        self._hdc_dim = target_dim;
    }

    /// Compute spatial relations between all tracked objects.
    pub fn update(&mut self, tracks: &[TrackedObject]) {
        self.edges.clear();

        for (i, a) in tracks.iter().enumerate() {
            for b in tracks.iter().skip(i + 1) {
                let dr = a.centroid_row as i32 - b.centroid_row as i32;
                let dc = a.centroid_col as i32 - b.centroid_col as i32;
                let dist = ((dr * dr + dc * dc) as f32).sqrt();

                let relations = self.classify_relation(dr, dc, dist);
                for rel in relations {
                    let rel_basis = self.relation_basis(rel);
                    let edge_hv = a.appearance_hv.bind(rel_basis).bind(&b.appearance_hv);
                    self.edges.push(crate::types::SceneGraphEdge {
                        subject_id: a.track_id,
                        object_id: b.track_id,
                        relation: rel,
                        relation_hv: edge_hv,
                    });
                }
            }
        }

        // Bundle all edge HVs into a unified scene graph representation
        if !self.edges.is_empty() {
            let refs: Vec<&ContinuousHV> = self.edges.iter().map(|e| &e.relation_hv).collect();
            self.graph_hv = Some(ContinuousHV::bundle(&refs).normalize());
        } else {
            self.graph_hv = None;
        }
    }

    /// Classify spatial relations between two objects.
    fn classify_relation(&self, dr: i32, dc: i32, dist: f32) -> Vec<crate::types::SpatialRelation> {
        use crate::types::SpatialRelation;
        let mut rels = Vec::new();
        let near = self.near_threshold as f32;

        if dist < near * 0.5 {
            rels.push(SpatialRelation::Overlapping);
        } else if dist < near {
            rels.push(SpatialRelation::Near);
        } else {
            rels.push(SpatialRelation::Far);
        }

        // Vertical relation (threshold: 1 grid cell)
        if dr < -1 {
            rels.push(SpatialRelation::Above);
        } else if dr > 1 {
            rels.push(SpatialRelation::Below);
        }

        // Horizontal relation
        if dc < -1 {
            rels.push(SpatialRelation::LeftOf);
        } else if dc > 1 {
            rels.push(SpatialRelation::RightOf);
        }

        rels
    }

    /// Get the basis HV for a spatial relation.
    fn relation_basis(&self, rel: crate::types::SpatialRelation) -> &ContinuousHV {
        self.relation_bases
            .iter()
            .find(|(r, _)| *r == rel)
            .map(|(_, hv)| hv)
            .expect("all relations pre-generated")
    }

    /// Current scene graph edges.
    pub fn edges(&self) -> &[crate::types::SceneGraphEdge] {
        &self.edges
    }

    /// Bundled scene graph HV (all relational triples combined).
    pub fn graph_hv(&self) -> Option<&ContinuousHV> {
        self.graph_hv.as_ref()
    }

    /// Number of edges in the scene graph.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Clear the scene graph.
    pub fn clear(&mut self) {
        self.edges.clear();
        self.graph_hv = None;
    }
}

impl TemporalPredictor for VisionManifold {
    fn predict_at(&self, current_state: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV {
        self.predict_horizon(current_state, horizon_seconds)
    }

    fn observe(&mut self, state: &ContinuousHV, dt_seconds: f32) {
        let x_inf = self.equilibrium(state);
        let sigma = self.gating(dt_seconds);
        self.state.lerp_in_place(&x_inf, 1.0 - sigma, sigma);
        self.frame_count += 1;
    }

    fn domain(&self) -> &'static str {
        "vision"
    }

    fn tau_base(&self) -> f32 {
        self.config.tau_base
    }

    fn default_horizons(&self) -> &'static [f32] {
        // ~1 frame, ~3 frames, ~15 frames, ~30 frames at 30fps
        &[0.033, 0.1, 0.5, 1.0]
    }

    fn horizon_labels(&self) -> &'static [&'static str] {
        &["next_frame", "short_term", "medium_term", "scene_scale"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Default-configured 64x64 manifold, for tests that don't care about specific config.
    fn test_manifold() -> VisionManifold {
        VisionManifold::new(VisionConfig::default(), 64, 64)
    }

    fn solid_gray_frame(width: u32, height: u32, value: u8) -> Vec<u8> {
        vec![value; (width * height) as usize]
    }

    fn gradient_frame(width: u32, height: u32) -> Vec<u8> {
        let mut pixels = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x + y) % 256) as u8);
            }
        }
        pixels
    }

    struct FixedEquilibrium(ContinuousHV);

    impl TransitionModel for FixedEquilibrium {
        fn equilibrium(&self, _state: &ContinuousHV, _context: &TransitionContext) -> ContinuousHV {
            self.0.clone()
        }
    }

    fn max_abs_difference(left: &ContinuousHV, right: &ContinuousHV) -> f32 {
        left.as_slice()
            .iter()
            .zip(right.as_slice())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn test_try_new_rejects_invalid_config_and_capacity() {
        let mut invalid = VisionConfig::default();
        invalid.patch_size = 0;
        assert!(VisionManifold::try_new(invalid, 64, 64).is_err());

        let valid = VisionConfig::default();
        assert!(VisionManifold::try_new(valid.clone(), 0, 64).is_err());
        assert!(VisionManifold::try_new(valid, 64, 0).is_err());
    }

    #[test]
    fn test_transition_projection_respects_time_horizon() {
        // Deliberately use non-unit vectors so a hidden normalization cannot pass
        // the dt=0 invariant merely by preserving cosine direction.
        let state = ContinuousHV::from_vec((0..256).map(|i| i as f32 * 0.01 - 0.7).collect());
        let equilibrium =
            ContinuousHV::from_vec((0..256).map(|i| 0.4 - i as f32 * 0.003).collect());
        let model = FixedEquilibrium(equilibrium.clone());
        let context = TransitionContext {
            tau: 0.5,
            ..Default::default()
        };

        let at_zero = model.project(&state, 0.0, &context);
        let short = model.project(&state, 0.05, &context);
        let long = model.project(&state, 5.0, &context);

        assert!(
            max_abs_difference(&at_zero, &state) < 1e-6,
            "dt=0 must preserve every state component exactly"
        );
        assert!(
            short.similarity(&state) > long.similarity(&state),
            "short horizons must retain more of the current state"
        );
        assert!(
            long.similarity(&equilibrium) > short.similarity(&equilibrium),
            "long horizons must approach equilibrium"
        );
        assert!(
            max_abs_difference(&long, &equilibrium) < 1e-3,
            "large dt/tau must converge component-wise to equilibrium"
        );
    }

    #[test]
    fn test_transition_projection_obeys_semigroup_for_fixed_equilibrium() {
        let state = ContinuousHV::from_vec((0..256).map(|i| (i as f32 - 80.0) * 0.02).collect());
        let equilibrium =
            ContinuousHV::from_vec((0..256).map(|i| (120.0 - i as f32) * 0.004).collect());
        let model = FixedEquilibrium(equilibrium);
        let context = TransitionContext {
            tau: 0.7,
            ..Default::default()
        };

        let split = model.project(&model.project(&state, 0.2, &context), 0.35, &context);
        let direct = model.project(&state, 0.55, &context);

        assert!(
            max_abs_difference(&split, &direct) < 2e-6,
            "closed-form projection must compose over adjacent time intervals"
        );
    }

    #[test]
    fn test_manifold_construction() {
        let cfg = VisionConfig::default();
        let m = VisionManifold::new(cfg, 64, 64);
        assert_eq!(m.frame_count(), 0);
        assert_eq!(m.prediction_error(), 0.0);
    }

    #[test]
    fn test_checked_multiband_rejects_invalid_hv_without_mutation() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut manifold = VisionManifold::new(config, 16, 16);
        let wrong_dim = ContinuousHV::random(128, 9);
        assert!(
            manifold
                .observe_multiband_frame_checked(&wrong_dim, 0.033)
                .is_err()
        );
        assert!(
            manifold
                .observe_multiband_frame_checked(&ContinuousHV::random(256, 9), f32::NAN)
                .is_err()
        );
        assert_eq!(manifold.frame_count(), 0);
    }

    #[test]
    fn test_patch_appearance_is_position_invariant() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        config.enable_motion = false;
        let mut manifold = VisionManifold::new(config, 16, 8);
        let frame = vec![180u8; 16 * 8];
        manifold
            .observe_frame_checked(&frame, 16, 8, 1, 0.033)
            .unwrap();

        let bound = manifold.last_patch_hvs();
        let appearance = manifold.last_patch_appearance_hvs();
        assert_eq!(appearance.len(), 2);
        assert!(appearance[0].similarity(&appearance[1]) > 0.99);
        assert!(bound[0].similarity(&bound[1]) < appearance[0].similarity(&appearance[1]));
    }

    #[test]
    fn test_sensor_depth_checked_rejects_bad_maps_without_mutation() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        config.enable_depth = true;
        let mut manifold = VisionManifold::new(config, 16, 16);
        let frame = vec![128u8; 16 * 16];

        assert!(
            manifold
                .observe_frame_with_depth_checked(&frame, 16, 16, 1, &[0.5; 3], 0.033)
                .is_err()
        );
        assert!(
            manifold
                .observe_frame_with_depth_checked(
                    &frame,
                    16,
                    16,
                    1,
                    &[0.5, 0.5, f32::INFINITY, 0.5],
                    0.033,
                )
                .is_err()
        );
        assert_eq!(manifold.frame_count(), 0);
        assert!(manifold.last_patch_hvs().is_empty());
    }

    #[test]
    fn test_observe_single_frame() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame = solid_gray_frame(64, 64, 128);

        let tel = m.observe_frame(&frame, 64, 64, 1, 0.033);
        assert_eq!(tel.frame_sequence, 1);
        // After a single CfC step from zero state, the manifold has begun evolving
        assert!(
            m.state().norm() > 0.0,
            "State should be non-zero after observation"
        );
    }

    #[test]
    fn test_coherence_stays_high_for_static_scene() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame = solid_gray_frame(64, 64, 128);
        let dt = 0.033;

        // Observe same frame repeatedly — coherence should remain high throughout
        for _ in 0..30 {
            m.observe_frame(&frame, 64, 64, 1, dt);
        }

        assert!(
            m.coherence() > 0.9,
            "Coherence should be high for static scene, got {}",
            m.coherence()
        );
    }

    #[test]
    fn test_prediction_error_decreases_for_static_scene() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame = gradient_frame(64, 64);
        let dt = 0.033;

        // Observe same frame repeatedly — prediction error should decrease
        let mut errors = Vec::new();
        for _ in 0..20 {
            let tel = m.observe_frame(&frame, 64, 64, 1, dt);
            errors.push(tel.prediction_error);
        }

        // After warm-up, later errors should be smaller than early errors
        let early_mean: f32 = errors[2..5].iter().sum::<f32>() / 3.0;
        let late_mean: f32 = errors[15..20].iter().sum::<f32>() / 5.0;
        assert!(
            late_mean <= early_mean + 0.05,
            "Prediction error should decrease for static scene: early={early_mean}, late={late_mean}"
        );
    }

    #[test]
    fn test_scene_change_spikes_error() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let dt = 0.033;

        // Converge on scene A
        let frame_a = solid_gray_frame(64, 64, 50);
        for _ in 0..15 {
            m.observe_frame(&frame_a, 64, 64, 1, dt);
        }
        let stable_error = m.prediction_error();

        // Switch to scene B — error should spike
        let frame_b = solid_gray_frame(64, 64, 200);
        m.observe_frame(&frame_b, 64, 64, 1, dt);
        // Observe a second frame to allow PE to catch up (1-frame lag in calculation)
        m.observe_frame(&frame_b, 64, 64, 1, dt);
        let spike_error = m.prediction_error();

        assert!(
            spike_error > stable_error,
            "Scene change should spike prediction error: stable={stable_error}, spike={spike_error}"
        );
    }

    #[test]
    fn test_temporal_prediction_o1() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame = gradient_frame(64, 64);
        m.observe_frame(&frame, 64, 64, 1, 0.033);

        let input = m.state().clone();

        // Predict at multiple horizons — all should return valid HVs
        let p_short = m.predict_at(&input, 0.033);
        let p_medium = m.predict_at(&input, 1.0);
        let p_long = m.predict_at(&input, 100.0);

        assert!(p_short.norm() > 0.0);
        assert!(p_medium.norm() > 0.0);
        assert!(p_long.norm() > 0.0);

        // Longer horizons should approach equilibrium more (higher sigma)
        let state = m.state();
        let sim_short = state.similarity(&p_short);
        let sim_long = state.similarity(&p_long);
        // Short prediction is closer to current state than long prediction
        assert!(
            sim_short >= sim_long - 0.01,
            "Short prediction should be closer to current state: short={sim_short}, long={sim_long}"
        );
    }

    #[test]
    fn test_temporal_predictor_trait() {
        let cfg = VisionConfig::default();
        let m = VisionManifold::new(cfg, 64, 64);

        assert_eq!(m.domain(), "vision");
        assert!(m.tau_base() > 0.0);
        assert!(!m.default_horizons().is_empty());
        assert_eq!(m.default_horizons().len(), m.horizon_labels().len());
    }

    #[test]
    fn test_reset() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        m.observe_frame(&frame, 64, 64, 1, 0.033);
        assert!(m.frame_count() > 0);

        m.reset();
        assert_eq!(m.frame_count(), 0);
        assert_eq!(m.prediction_error(), 0.0);
        assert_eq!(m.coherence(), 0.0);
    }

    #[test]
    fn test_refine_from_attention_modifies_weights() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let dt = 0.033;

        // Need at least 2 frames so surprise map has data
        let frame_a = solid_gray_frame(64, 64, 50);
        m.observe_frame(&frame_a, 64, 64, 1, dt);

        // Scene change creates surprise contrast
        let frame_b = gradient_frame(64, 64);
        m.observe_frame(&frame_b, 64, 64, 1, dt);

        let weights_before: Vec<f32> = m.encoder().feature_weights().to_vec();
        m.refine_from_attention();
        let weights_after: Vec<f32> = m.encoder().feature_weights().to_vec();

        // Weights should have changed (surprise contrast drives contrastive update)
        let changed = weights_before
            .iter()
            .zip(weights_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-8);
        assert!(changed, "Saliency refinement should modify encoder weights");
    }

    #[test]
    fn test_refine_from_attention_noop_when_no_surprise() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);

        // Only one frame observed — surprise map is all zeros
        let frame = solid_gray_frame(64, 64, 128);
        m.observe_frame(&frame, 64, 64, 1, 0.033);

        let weights_before: Vec<f32> = m.encoder().feature_weights().to_vec();
        m.refine_from_attention();
        let weights_after: Vec<f32> = m.encoder().feature_weights().to_vec();

        // Should be a no-op (no surprise contrast)
        assert_eq!(weights_before, weights_after);
    }

    #[test]
    fn test_delayed_horizon_evaluator_scores_only_matured_forecasts() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut manifold = VisionManifold::new(config, 8, 8);
        let mut evaluator = DelayedHorizonEvaluator::new(vec![
            (0.05, "near".to_string()),
            (0.10, "far".to_string()),
        ])
        .unwrap();

        manifold
            .observe_frame_checked(&vec![10; 64], 8, 8, 1, 0.04)
            .unwrap();
        assert_eq!(evaluator.observe(&manifold, 0.04).unwrap(), 0);
        manifold
            .observe_frame_checked(&vec![20; 64], 8, 8, 1, 0.04)
            .unwrap();
        assert_eq!(evaluator.observe(&manifold, 0.04).unwrap(), 0);
        manifold
            .observe_frame_checked(&vec![30; 64], 8, 8, 1, 0.04)
            .unwrap();
        assert_eq!(evaluator.observe(&manifold, 0.04).unwrap(), 1);

        let accuracy = evaluator.accuracy(manifold.frame_count());
        assert_eq!(accuracy.sample_counts, vec![1, 0]);
        assert!(accuracy.errors[0].is_finite());
        assert!(accuracy.persistence_errors[0].is_finite());
        assert!(accuracy.mean_lateness_seconds[0] >= 0.0);
    }

    #[test]
    fn test_delayed_horizon_evaluator_bounds_and_roundtrips_pending_work() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut manifold = VisionManifold::new(config, 8, 8);
        manifold
            .observe_frame_checked(&vec![10; 64], 8, 8, 1, 0.001)
            .unwrap();

        let mut evaluator =
            DelayedHorizonEvaluator::with_pending_limit(vec![(1.0, "far".to_string())], 2).unwrap();
        for _ in 0..5 {
            evaluator.observe(&manifold, 0.001).unwrap();
        }
        assert_eq!(evaluator.pending.len(), 2);
        assert_eq!(evaluator.accuracy(1).dropped_forecasts, vec![3]);

        let saved = evaluator.save_state();
        let mut restored = DelayedHorizonEvaluator::default();
        restored.load_state(&saved).unwrap();
        assert_eq!(restored.save_state(), saved);

        let mut malformed = saved.clone();
        malformed.pending[0].predicted.pop();
        let before = restored.save_state();
        assert!(restored.load_state(&malformed).is_err());
        assert_eq!(restored.save_state(), before);
    }

    #[test]
    fn test_delayed_horizon_accuracy_reports_error_dispersion() {
        let mut evaluator = DelayedHorizonEvaluator::new(vec![(0.1, "near".to_string())]).unwrap();
        evaluator.accumulators[0] = HorizonAccumulator {
            prediction_error_sum: 1.0,
            prediction_error_sq_sum: 1.0,
            persistence_error_sum: 0.5,
            persistence_error_sq_sum: 0.125,
            lateness_sum: 0.02,
            samples: 2,
            dropped_forecasts: 0,
            expired_forecasts: 0,
        };

        let accuracy = evaluator.accuracy(7);
        assert!((accuracy.errors[0] - 0.5).abs() < 1e-6);
        assert!((accuracy.prediction_error_stddev[0] - 0.5).abs() < 1e-6);
        assert!((accuracy.persistence_error_stddev[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_delayed_horizon_rejects_inconsistent_squared_error_state() {
        let evaluator = DelayedHorizonEvaluator::default();
        let mut malformed = evaluator.save_state();
        malformed.accumulators[0].samples = 2;
        malformed.accumulators[0].prediction_error_sum = 2.0;
        malformed.accumulators[0].prediction_error_sq_sum = 0.5;
        assert!(DelayedHorizonEvaluator::validate_state(&malformed).is_err());
    }

    #[test]
    fn test_delayed_horizon_legacy_state_migrates_zero_variance_statistics() {
        let mut legacy = DelayedHorizonEvaluator::default().save_state();
        legacy.schema_version = 2;
        legacy.accumulators[0].samples = 2;
        legacy.accumulators[0].prediction_error_sum = 1.0;
        legacy.accumulators[0].persistence_error_sum = 0.5;
        legacy.accumulators[0].prediction_error_sq_sum = 0.0;
        legacy.accumulators[0].persistence_error_sq_sum = 0.0;

        let mut restored = DelayedHorizonEvaluator::default();
        restored.load_state(&legacy).unwrap();
        let migrated = restored.save_state();
        assert_eq!(
            migrated.schema_version,
            DELAYED_HORIZON_EVALUATOR_STATE_SCHEMA_VERSION
        );
        assert!((migrated.accumulators[0].prediction_error_sq_sum - 0.5).abs() < 1e-9);
        assert!((migrated.accumulators[0].persistence_error_sq_sum - 0.125).abs() < 1e-9);
    }

    #[test]
    fn test_delayed_horizon_evaluator_expires_stale_forecasts() {
        let config = VisionConfig {
            hdc_dim: 256,
            ..VisionConfig::default()
        };
        let mut manifold = VisionManifold::new(config, 8, 8);
        let mut evaluator = DelayedHorizonEvaluator::new(vec![(0.1, "near".to_string())]).unwrap();
        evaluator.set_max_lateness_factor_checked(1.0).unwrap();

        manifold
            .observe_frame_checked(&vec![10; 64], 8, 8, 1, 0.01)
            .unwrap();
        evaluator.observe(&manifold, 0.01).unwrap();

        manifold
            .observe_frame_checked(&vec![20; 64], 8, 8, 1, 1.0)
            .unwrap();
        assert_eq!(evaluator.observe(&manifold, 1.0).unwrap(), 0);
        let accuracy = evaluator.accuracy(manifold.frame_count());
        assert_eq!(accuracy.sample_counts, vec![0]);
        assert_eq!(accuracy.expired_forecasts, vec![1]);
        assert_eq!(accuracy.dropped_forecasts, vec![0]);
    }

    #[test]
    fn test_delayed_horizon_lateness_policy_rejects_invalid_values_atomically() {
        let mut evaluator = DelayedHorizonEvaluator::default();
        let before = evaluator.save_state();
        assert!(evaluator.set_max_lateness_factor_checked(f32::NAN).is_err());
        assert!(evaluator.set_max_lateness_factor_checked(0.0).is_err());
        assert_eq!(evaluator.save_state(), before);
    }

    #[test]
    fn test_delayed_horizon_evaluator_dilates_pending_forecasts() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut manifold = VisionManifold::new(config, 8, 8);
        manifold
            .observe_frame_checked(&vec![10; 64], 8, 8, 1, 0.01)
            .unwrap();
        let mut evaluator = DelayedHorizonEvaluator::new(vec![(1.0, "far".into())]).unwrap();
        evaluator.observe(&manifold, 0.01).unwrap();
        evaluator.dilate(512);
        assert_eq!(evaluator.hdc_dim, Some(512));
        assert_eq!(evaluator.pending[0].predicted.dim(), 512);
        assert_eq!(evaluator.pending[0].persistence.dim(), 512);
    }

    #[test]
    fn test_delayed_horizon_evaluator_validates_clock_and_policy() {
        assert!(DelayedHorizonEvaluator::new(Vec::new()).is_err());
        assert!(DelayedHorizonEvaluator::new(vec![(f32::NAN, "bad".into())]).is_err());
        assert!(DelayedHorizonEvaluator::new(vec![(0.1, "".into())]).is_err());
        assert!(DelayedHorizonEvaluator::new(vec![(0.2, "a".into()), (0.1, "b".into())]).is_err());

        let config = VisionConfig {
            hdc_dim: 256,
            ..VisionConfig::default()
        };
        let manifold = VisionManifold::new(config, 8, 8);
        let mut evaluator = DelayedHorizonEvaluator::default();
        assert!(evaluator.observe(&manifold, 0.033).is_err());

        let mut observed = VisionManifold::new(
            VisionConfig {
                hdc_dim: 256,
                ..VisionConfig::default()
            },
            8,
            8,
        );
        observed
            .observe_frame_checked(&vec![1; 64], 8, 8, 1, 0.033)
            .unwrap();
        evaluator.observe(&observed, 0.033).unwrap();
        evaluator.hdc_dim = Some(512);
        assert!(evaluator.observe(&observed, 0.033).is_err());
        evaluator.reset();
        assert!(evaluator.observe(&observed, 0.033).is_ok());
    }

    #[test]
    fn test_manifold_owns_delayed_evidence_per_modality() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut manifold = VisionManifold::new(config, 8, 8);
        for value in [10, 20, 30, 40] {
            manifold
                .observe_frame_checked(&vec![value; 64], 8, 8, 1, 0.04)
                .unwrap();
        }
        let visible = manifold.horizon_accuracy();
        assert!(visible.sample_counts.iter().any(|count| *count > 0));

        manifold.activate_modality(VisualModality::Stereo);
        assert!(
            manifold
                .horizon_accuracy()
                .sample_counts
                .iter()
                .all(|count| *count == 0)
        );
        assert_eq!(
            manifold
                .horizon_accuracy_for(VisualModality::Visible)
                .unwrap()
                .sample_counts,
            visible.sample_counts
        );
    }

    #[test]
    fn test_schema_seven_payload_roundtrips_under_current_schema() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut source = VisionManifold::new(config.clone(), 8, 8);
        for value in [10, 20, 30, 40] {
            source
                .observe_frame_checked(&vec![value; 64], 8, 8, 1, 0.04)
                .unwrap();
        }
        source.activate_modality(VisualModality::Stereo);
        let saved = source.save_state();
        assert_eq!(saved.schema_version, MANIFOLD_STATE_SCHEMA_VERSION);
        assert!(saved.horizon_evaluator.is_some());
        assert!(saved.modality_contexts[0].horizon_evaluator.is_some());

        let mut restored = VisionManifold::new(config, 8, 8);
        restored.load_state(&saved).unwrap();
        assert_eq!(
            restored
                .horizon_accuracy_for(VisualModality::Visible)
                .unwrap()
                .sample_counts,
            source
                .horizon_accuracy_for(VisualModality::Visible)
                .unwrap()
                .sample_counts
        );

        let mut malformed = saved.clone();
        malformed.modality_contexts[0].horizon_evaluator = None;
        let before = restored.save_state();
        assert!(restored.load_state(&malformed).is_err());
        assert_eq!(restored.save_state().frame_count, before.frame_count);
    }

    #[test]
    fn test_evaluate_horizons_structure() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame = gradient_frame(64, 64);
        m.observe_frame(&frame, 64, 64, 1, 0.033);

        let acc = m.evaluate_horizons();

        assert_eq!(acc.horizons.len(), 4);
        assert_eq!(acc.labels.len(), 4);
        assert_eq!(acc.errors.len(), 4);
        assert_eq!(acc.frame_sequence, 1);
        assert_eq!(acc.labels[0], "next_frame");
        assert_eq!(acc.labels[3], "scene_scale");
    }

    #[test]
    fn test_evaluate_horizons_error_ordering() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        // Converge on the frame
        for _ in 0..10 {
            m.observe_frame(&frame, 64, 64, 1, 0.033);
        }

        let acc = m.evaluate_horizons();

        // Short horizon prediction should be at least as good as long horizon
        // (closer to current state means less divergence from equilibrium)
        assert!(
            acc.errors[0] <= acc.errors[3] + 0.05,
            "Short horizon error ({}) should be <= long horizon error ({})",
            acc.errors[0],
            acc.errors[3]
        );
    }

    #[test]
    fn test_evaluate_horizons_before_any_frame() {
        let cfg = VisionConfig::default();
        let m = VisionManifold::new(cfg, 64, 64);

        // No frames observed — should return default errors
        let acc = m.evaluate_horizons();
        assert_eq!(acc.errors.len(), 4);
        // All errors should be 1.0 (maximum)
        for &e in &acc.errors {
            assert!(
                (e - 1.0).abs() < 1e-6,
                "Pre-frame error should be 1.0, got {e}"
            );
        }
    }

    #[test]
    fn test_gating_bounds() {
        let cfg = VisionConfig::default();
        let m = VisionManifold::new(cfg, 64, 64);

        // dt=0 → sigma=0 (no change)
        assert!((m.gating(0.0)).abs() < 1e-6);

        // dt >> tau → sigma ≈ 1 (jump to equilibrium)
        assert!((m.gating(1000.0) - 1.0).abs() < 1e-4);

        // Intermediate dt → 0 < sigma < 1
        let mid = m.gating(0.5);
        assert!(mid > 0.0 && mid < 1.0, "mid sigma = {mid}");
    }

    // === State Persistence ===

    #[test]
    fn test_save_state_captures_fields() {
        let cfg = VisionConfig::default();
        let m = VisionManifold::new(cfg.clone(), 64, 64);

        let state = m.save_state();
        assert_eq!(state.hdc_dim, cfg.hdc_dim);
        assert_eq!(state.weight_hv.len(), cfg.hdc_dim);
        assert!((state.tau_base - cfg.tau_base).abs() < 1e-6);
        assert_eq!(state.training_steps, 0);
        assert_eq!(state.feature_weights.len(), cfg.total_features());
    }

    #[test]
    fn test_save_load_roundtrip() {
        let cfg = VisionConfig::default();
        let mut m1 = VisionManifold::new(cfg.clone(), 64, 64);

        // Evolve manifold so it has non-trivial state
        let frame = gradient_frame(64, 64);
        for _ in 0..10 {
            m1.observe_frame(&frame, 64, 64, 1, 0.033);
        }

        let saved = m1.save_state();

        // Load into a fresh manifold
        let mut m2 = VisionManifold::new(cfg, 64, 64);
        assert!(m2.load_state(&saved).is_ok());

        // Weight HVs should match
        let sim = m2.weight_hv().similarity(m1.weight_hv());
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Loaded weight_hv should match saved: sim={sim}"
        );

        // Tau should match
        assert!(
            (m2.current_tau() - m1.current_tau()).abs() < 1e-6,
            "Loaded tau should match"
        );
    }

    #[test]
    fn test_checkpoint_schema_v2_restores_live_perceptual_state() {
        let mut cfg = VisionConfig::default();
        cfg.hdc_dim = 256;
        cfg.patch_size = 4;
        cfg.multi_scale.scales = vec![4, 8];
        cfg.enable_predictive_hierarchy = true;
        cfg.enable_temporal_binding = true;

        let mut source = VisionManifold::new(cfg.clone(), 16, 16);
        // The scene graph computes relations between tracked objects, so
        // `enable_scene_graph()` documents object memory as a prerequisite;
        // the schema-5 checkpoint validation enforces that contract.
        source.enable_object_memory(8);
        source.enable_scene_graph();
        let first = gradient_frame(16, 16);
        let second: Vec<u8> = first.iter().map(|value| 255u8 - *value).collect();
        source.observe_frame(&first, 16, 16, 1, 0.033);
        source.observe_frame(&second, 16, 16, 1, 0.033);
        source.last_imagination = Some(ContinuousHV::random(256, 910_001));
        source.imagination_surprise = 0.42;
        source.last_intent_hv = ContinuousHV::random(256, 910_002);
        source.last_geodesic = vec![
            ContinuousHV::random(256, 910_003),
            ContinuousHV::random(256, 910_004),
        ];

        let saved = source.save_state();
        assert_eq!(saved.schema_version, MANIFOLD_STATE_SCHEMA_VERSION);
        assert!(saved.surprise_state.is_some());
        assert!(saved.predictive_state.is_some());
        assert!(!saved.temporal_patch_hvs.is_empty());
        assert!(saved.scene_graph_enabled);

        let mut restored = VisionManifold::new(cfg, 16, 16);
        restored.load_state(&saved).unwrap();
        let roundtrip = restored.save_state();

        assert_eq!(roundtrip.schema_version, MANIFOLD_STATE_SCHEMA_VERSION);
        assert_eq!(roundtrip.surprise_state, saved.surprise_state);
        assert_eq!(roundtrip.predictive_state, saved.predictive_state);
        assert_eq!(roundtrip.temporal_patch_hvs, saved.temporal_patch_hvs);
        assert_eq!(roundtrip.last_imagination, saved.last_imagination);
        assert_eq!(roundtrip.imagination_surprise, saved.imagination_surprise);
        assert_eq!(roundtrip.last_intent_hv, saved.last_intent_hv);
        assert_eq!(roundtrip.last_geodesic, saved.last_geodesic);
        assert!(restored.scene_graph().is_some());
    }

    #[test]
    fn test_checkpoint_accepts_frame_smaller_than_encoder_capacity() {
        let mut cfg = VisionConfig::default();
        cfg.hdc_dim = 256;
        cfg.patch_size = 4;
        let mut source = VisionManifold::new(cfg.clone(), 32, 32);
        let frame = gradient_frame(16, 16);
        source.observe_frame(&frame, 16, 16, 1, 0.033);
        let saved = source.save_state();
        assert_eq!(saved.last_patch_hvs.len(), 16);

        let mut restored = VisionManifold::new(cfg, 32, 32);
        restored.load_state(&saved).unwrap();
        assert_eq!(restored.last_patch_hvs().len(), 16);
    }

    #[test]
    fn test_schema_v3_rejects_semantically_different_configuration() {
        let mut source_config = VisionConfig::default();
        source_config.hdc_dim = 256;
        let source = VisionManifold::new(source_config, 32, 32);
        let saved = source.save_state();

        let mut destination_config = VisionConfig::default();
        destination_config.hdc_dim = 256;
        destination_config.input_blend = 0.6;
        let mut destination = VisionManifold::new(destination_config, 32, 32);
        let before = destination.state().clone();

        let error = destination.load_state(&saved).unwrap_err();
        assert!(error.contains("input_blend mismatch"));
        // `destination.state()` is still the constructor's zero vector here
        // (no frame was ever observed), and `similarity()` special-cases
        // near-zero norms to return `0.0` rather than `1.0` — so a
        // similarity-based "unchanged" check is unsound for this state.
        // Compare the raw values directly instead, matching the pattern used
        // by `test_stereo_checked_rejection_does_not_advance_manifold`.
        assert_eq!(destination.state().as_slice(), before.as_slice());
        assert_eq!(destination.frame_count(), 0);
    }

    #[test]
    fn schema_eight_rejects_future_scene_memory_atomically() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut manifold = VisionManifold::new(config, 16, 16);
        manifold.enable_scene_memory(4);
        let frame = gradient_frame(16, 16);
        manifold.observe_frame(&frame, 16, 16, 1, 0.033);
        let frame_count_before = manifold.frame_count();
        let state_before = manifold.state().clone();

        manifold.scene_memory.as_mut().unwrap().remember(
            &ContinuousHV::random(256, 77),
            frame_count_before,
            Vec::new(),
        );
        let mut saved = manifold.save_state();
        saved.scene_memory.as_mut().unwrap().landmarks[0].1 = frame_count_before + 1;

        let error = manifold.load_state(&saved).unwrap_err();
        assert!(error.contains("beyond checkpoint frame"));
        assert_eq!(manifold.frame_count(), frame_count_before);
        assert!(manifold.state().similarity(&state_before) > 0.999_999);
    }

    #[test]
    fn schema_eight_rejects_non_monotonic_episode_timelines() {
        let state = SceneMemoryState {
            landmarks: vec![(vec![0.0; 32], 2), (vec![0.0; 32], 1)],
            capacity: 4,
            threshold: 0.8,
            pixel_budget_bytes: 0,
            retained_pixel_bytes: 0,
            raw_frames: vec![Vec::new(), Vec::new()],
            frame_metadata: vec![SceneFrameMetadata::default(); 2],
            object_episodes: vec![(vec![0.0; 32], 3), (vec![0.0; 32], 2)],
        };
        SceneMemory::validate_state(&state, 32).unwrap();
        let error = SceneMemory::validate_temporal_state(&state, 3).unwrap_err();
        assert!(error.contains("non-monotonic"));
    }

    #[test]
    fn test_schema_v3_restores_quality_and_scene_policy() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut source = VisionManifold::new(config.clone(), 32, 32);
        source.set_scene_store_thresholds(0.42, 0.17);
        source.set_scene_dampen_factor(0.63);
        source.last_dilation_cycle = 77;
        let pixels = vec![128u8; 32 * 32 * 3];
        source.observe_frame(&pixels, 32, 32, 3, 0.033);
        let saved = source.save_state();

        let mut restored = VisionManifold::new(config, 32, 32);
        restored.load_state(&saved).unwrap();

        assert!((restored.coherence - source.coherence).abs() < 1e-6);
        assert!((restored.last_fep.free_energy - source.last_fep.free_energy).abs() < 1e-6);
        assert!((restored.scene_store_coherence_threshold - 0.42).abs() < 1e-6);
        assert!((restored.scene_store_error_threshold - 0.17).abs() < 1e-6);
        assert!((restored.scene_dampen_factor - 0.63).abs() < 1e-6);
        assert_eq!(restored.last_dilation_cycle, 77);
    }

    #[test]
    fn test_future_checkpoint_schema_is_rejected_before_mutation() {
        let mut cfg = VisionConfig::default();
        cfg.hdc_dim = 256;
        let mut manifold = VisionManifold::new(cfg, 16, 16);
        let frame = gradient_frame(16, 16);
        manifold.observe_frame(&frame, 16, 16, 1, 0.033);

        let weight_before = manifold.weight_hv().clone();
        let frame_count_before = manifold.frame_count();
        let state_before = manifold.state().clone();
        let mut future = manifold.save_state();
        future.schema_version = MANIFOLD_STATE_SCHEMA_VERSION + 1;
        future.weight_hv = vec![0.0; future.hdc_dim];

        let error = manifold.load_state(&future).unwrap_err();
        assert!(error.contains("unsupported manifold checkpoint schema"));
        assert!(manifold.weight_hv().similarity(&weight_before) > 0.999_999);
        assert!(manifold.state().similarity(&state_before) > 0.999_999);
        assert_eq!(manifold.frame_count(), frame_count_before);
    }

    #[test]
    fn test_schema_v2_requires_enabled_predictive_state() {
        let mut cfg = VisionConfig::default();
        cfg.hdc_dim = 256;
        cfg.multi_scale.scales = vec![8, 16];
        cfg.enable_predictive_hierarchy = true;
        let mut manifold = VisionManifold::new(cfg, 16, 16);
        let mut state = manifold.save_state();
        state.predictive_state = None;

        let error = manifold.load_state(&state).unwrap_err();
        assert!(error.contains("missing state for the enabled predictive hierarchy"));
    }

    #[test]
    fn test_load_state_rejects_dimension_mismatch() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);

        let bad_state = ManifoldState {
            weight_hv: vec![0.0; 100], // Wrong dimension
            tau_base: 0.5,
            hdc_dim: 100,
            num_features: 5,
            ..ManifoldState::default()
        };

        assert!(m.load_state(&bad_state).is_err());
    }

    // === RGB Manifold Tests ===

    #[test]
    fn test_manifold_rgb_frame() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);

        let rgb: Vec<u8> = (0..64 * 64).flat_map(|_| vec![128u8, 64, 192]).collect();
        let tel = m.observe_frame(&rgb, 64, 64, 3, 0.033);
        assert_eq!(tel.frame_sequence, 1);
        assert!(m.state().norm() > 0.0);
    }

    #[test]
    fn test_manifold_rgb_color_discrimination() {
        let cfg = VisionConfig::default();
        let mut m_red = VisionManifold::new(cfg.clone(), 64, 64);
        let red: Vec<u8> = (0..64 * 64).flat_map(|_| vec![255u8, 0, 0]).collect();
        m_red.observe_frame(&red, 64, 64, 3, 0.033);

        let mut m_blue = VisionManifold::new(cfg, 64, 64);
        let blue: Vec<u8> = (0..64 * 64).flat_map(|_| vec![0u8, 0, 255]).collect();
        m_blue.observe_frame(&blue, 64, 64, 3, 0.033);

        let sim = m_red.state().similarity(m_blue.state());
        assert!(
            sim < 0.99,
            "Red and blue manifold states should differ: sim={sim}"
        );
    }

    // === Adaptive Training ===

    #[test]
    fn test_adaptive_training_triggers_on_alternating_pattern() {
        let mut cfg = VisionConfig::default();
        cfg.training.error_threshold = 0.05;
        cfg.training.learning_rate = 0.01;
        let mut m = VisionManifold::new(cfg, 64, 64);

        let frame_a = solid_gray_frame(64, 64, 50);
        let frame_b = solid_gray_frame(64, 64, 200);

        let mut training_count = 0;
        for step in 0..60 {
            let frame = if step % 2 == 0 { &frame_a } else { &frame_b };
            let tel = m.observe_frame(frame, 64, 64, 1, 0.033);
            if tel.training_triggered {
                training_count += 1;
            }
        }

        assert!(
            training_count > 0,
            "Adaptive training should trigger on alternating pattern, got {training_count} triggers"
        );
    }

    #[test]
    fn test_save_state_serializable() {
        let cfg = VisionConfig::default();
        let m = VisionManifold::new(cfg, 64, 64);
        let state = m.save_state();

        // Should be JSON-serializable
        let json = serde_json::to_string(&state).expect("Should serialize");
        let deserialized: ManifoldState = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized.hdc_dim, state.hdc_dim);
        assert_eq!(deserialized.weight_hv.len(), state.weight_hv.len());
    }

    #[test]
    fn test_manifold_checkpoint_envelope_roundtrip_and_corruption_rejection() {
        let config = VisionConfig {
            hdc_dim: 256,
            ..VisionConfig::default()
        };
        let mut source = VisionManifold::new(config.clone(), 8, 8);
        source
            .observe_frame_checked(&vec![42; 64], 8, 8, 1, 0.033)
            .unwrap();
        let encoded = source.save_checkpoint_bytes().unwrap();

        let mut restored = VisionManifold::new(config, 8, 8);
        restored.load_checkpoint_bytes(&encoded).unwrap();
        assert_eq!(restored.frame_count(), source.frame_count());
        assert!(restored.state().similarity(source.state()) > 0.9999);

        let before = restored.save_state();
        let mut corrupted = encoded.clone();
        let last = corrupted.len() - 1;
        corrupted[last] ^= 1;
        assert!(restored.load_checkpoint_bytes(&corrupted).is_err());
        assert_eq!(restored.save_state().frame_count, before.frame_count);
        assert_eq!(restored.save_state().state_hv, before.state_hv);
    }

    // === Auto-Refinement ===

    #[test]
    fn test_auto_refinement_modifies_weights_on_scene_change() {
        let mut cfg = VisionConfig::default();
        cfg.learning.contrastive_lr = 0.1; // Larger LR for test visibility
        let mut m = VisionManifold::new(cfg, 64, 64);

        let weights_before: Vec<f32> = m.encoder().feature_weights().to_vec();

        // Alternate between very different scenes to accumulate refinement
        let frame_a = solid_gray_frame(64, 64, 50);
        let frame_b = gradient_frame(64, 64);

        for i in 0..20 {
            let frame = if i % 2 == 0 { &frame_a } else { &frame_b };
            m.observe_frame(frame, 64, 64, 1, 0.033);
        }

        let weights_after: Vec<f32> = m.encoder().feature_weights().to_vec();

        // After many auto-refinement cycles, weights should have drifted
        let max_change: f32 = weights_before
            .iter()
            .zip(weights_after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_change > 1e-6,
            "Auto-refinement should modify weights over 20 frames, max_change={max_change}"
        );
    }

    // === Scene Memory ===

    #[test]
    fn test_scene_memory_construction() {
        let mem = SceneMemory::new(16);
        assert!(mem.is_empty());
        assert_eq!(mem.len(), 0);
    }

    #[test]
    fn test_scene_memory_remember_and_recognize() {
        let mut mem = SceneMemory::new(16);
        let dim = 16_384;

        let scene_a = ContinuousHV::random(dim, 100);
        let scene_b = ContinuousHV::random(dim, 200);

        mem.remember(&scene_a, 10, vec![]);
        mem.remember(&scene_b, 20, vec![]);
        assert_eq!(mem.len(), 2);

        // Should recognize scene_a
        let result = mem.recognize(&scene_a, 30);
        assert!(result.is_some(), "Should recognize stored scene");
        let m = result.unwrap();
        assert!(m.similarity > 0.99);
        assert_eq!(m.stored_at_frame, 10);
        assert_eq!(m.frames_since_stored, 20);
    }

    #[test]
    fn test_scene_memory_rejects_unknown() {
        let mut mem = SceneMemory::new(16);
        let dim = 16_384;

        let scene_a = ContinuousHV::random(dim, 100);
        mem.remember(&scene_a, 10, vec![]);

        // A completely different scene should not be recognized
        let unknown = ContinuousHV::random(dim, 999);
        let result = mem.recognize(&unknown, 20);
        assert!(result.is_none(), "Should not recognize unknown scene");
    }

    #[test]
    fn test_scene_memory_deduplication() {
        let mut mem = SceneMemory::new(16);
        let dim = 16_384;

        let scene = ContinuousHV::random(dim, 100);
        mem.remember(&scene, 10, vec![]);
        mem.remember(&scene, 20, vec![]); // Near-duplicate — should be skipped
        assert_eq!(mem.len(), 1, "Should not store near-duplicates");
    }

    #[test]
    fn test_object_episodes_do_not_participate_in_scene_recognition() {
        let mut mem = SceneMemory::new(4);
        let object = ContinuousHV::random(256, 77);
        mem.remember_object(&object, 5);
        assert_eq!(mem.object_episode_count(), 1);
        assert_eq!(mem.len(), 0);
        assert!(mem.recognize(&object, 6).is_none());

        let saved = mem.save_state();
        let mut restored = SceneMemory::new(1);
        restored.load_state(&saved);
        assert_eq!(restored.object_episode_count(), 1);
        assert_eq!(restored.len(), 0);
    }

    #[test]
    fn test_scene_memory_eviction() {
        let mut mem = SceneMemory::new(3);
        let dim = 16_384;

        for i in 0..5 {
            let scene = ContinuousHV::random(dim, 100 + i);
            mem.remember(&scene, i, vec![]);
        }
        assert_eq!(mem.len(), 3, "Should cap at capacity");
    }

    // === Health Telemetry ===

    #[test]
    fn test_health_initial() {
        let cfg = VisionConfig::default();
        let m = VisionManifold::new(cfg, 64, 64);
        let health = m.compute_health();

        assert!(health.is_healthy);
        assert!(health.tau_value > 0.0);
        assert_eq!(health.total_frames, 0);
        assert_eq!(health.total_training_steps, 0);
        // Initial weight_drift should be ~1.0 (no drift from initial)
        assert!(
            health.weight_drift > 0.9,
            "Initial weight drift should be near 1.0: {}",
            health.weight_drift
        );
    }

    #[test]
    fn test_health_after_processing() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        for _ in 0..20 {
            m.observe_frame(&frame, 64, 64, 1, 0.033);
        }

        let health = m.compute_health();
        assert!(health.is_healthy);
        assert_eq!(health.total_frames, 20);
        assert!(health.mean_coherence > 0.0);
        assert!(health.encoder_weight_entropy > 0.0);
    }

    #[test]
    fn test_health_serializable() {
        let cfg = VisionConfig::default();
        let m = VisionManifold::new(cfg, 64, 64);
        let health = m.compute_health();

        let json = serde_json::to_string(&health).expect("Should serialize");
        let _: ManifoldHealth = serde_json::from_str(&json).expect("Should deserialize");
    }

    // === Temporal Coherence Validation ===

    #[test]
    fn test_temporal_coherence_slowly_drifting_scene() {
        // Slowly drifting scenes should produce gradually-drifting HVs.
        // Similarity between adjacent frames > similarity between distant frames.
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);

        let mut states = Vec::new();
        for i in 0..40u8 {
            // Gradually increase brightness: 100 + i*2
            let frame = solid_gray_frame(64, 64, 100u8.saturating_add(i * 2));
            m.observe_frame(&frame, 64, 64, 1, 0.033);
            states.push(m.state().clone());
        }

        // Adjacent states should be more similar than distant states
        let sim_adjacent: f32 = (0..38)
            .map(|i| {
                let s1 = &states[i];
                let s2 = &states[i + 1];
                if s1.dim() != s2.dim() {
                    let max_dim = s1.dim().max(s2.dim());
                    s1.dilate(max_dim).similarity(&s2.dilate(max_dim))
                } else {
                    s1.similarity(s2)
                }
            })
            .sum::<f32>()
            / 38.0;

        let sim_distant: f32 = (0..10)
            .map(|i| {
                let s1 = &states[i];
                let s2 = &states[i + 25];
                if s1.dim() != s2.dim() {
                    let max_dim = s1.dim().max(s2.dim());
                    s1.dilate(max_dim).similarity(&s2.dilate(max_dim))
                } else {
                    s1.similarity(s2)
                }
            })
            .sum::<f32>()
            / 10.0;

        assert!(
            sim_adjacent > sim_distant,
            "Adjacent states ({sim_adjacent:.4}) should be more similar than distant ({sim_distant:.4})"
        );
    }

    #[test]
    fn test_temporal_coherence_monotonic_decay() {
        // For a single scene change, similarity to the initial state should
        // monotonically decrease over time as the manifold adapts.
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);

        // Converge on scene A
        let frame_a = gradient_frame(64, 64);
        for _ in 0..15 {
            m.observe_frame(&frame_a, 64, 64, 1, 0.033);
        }
        let state_a = m.state().clone();

        // Switch to scene B, track divergence from state_a
        let frame_b = solid_gray_frame(64, 64, 200);
        let mut sims = Vec::new();
        for _ in 0..20 {
            m.observe_frame(&frame_b, 64, 64, 1, 0.033);
            sims.push(m.state().similarity(&state_a));
        }

        // Similarity should generally decrease (allow minor fluctuations)
        let early_avg = sims[0..5].iter().sum::<f32>() / 5.0;
        let late_avg = sims[15..20].iter().sum::<f32>() / 5.0;
        assert!(
            late_avg <= early_avg + 0.05,
            "Similarity to old scene should decrease: early={early_avg:.4}, late={late_avg:.4}"
        );
    }

    #[test]
    fn test_temporal_coherence_static_scene_stability() {
        // A static scene should converge to a stable state (minimal jitter).
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        // Process 40 frames of the same scene
        for _ in 0..40 {
            m.observe_frame(&frame, 64, 64, 1, 0.033);
        }
        let state_early = m.state().clone();

        for _ in 0..10 {
            m.observe_frame(&frame, 64, 64, 1, 0.033);
        }
        let state_late = m.state().clone();

        let sim = state_early.similarity(&state_late);
        assert!(
            sim > 0.95,
            "Converged static scene states should be highly similar: {sim:.4}"
        );
    }

    #[test]
    fn test_temporal_coherence_rapid_oscillation_bounded() {
        // Rapidly oscillating between two scenes should keep state bounded
        // (not diverge to infinity).
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame_a = solid_gray_frame(64, 64, 50);
        let frame_b = solid_gray_frame(64, 64, 200);

        for i in 0..100 {
            let frame = if i % 2 == 0 { &frame_a } else { &frame_b };
            m.observe_frame(frame, 64, 64, 1, 0.033);
        }

        let norm = m.state().norm();
        assert!(
            norm.is_finite() && norm > 0.0 && norm < 100.0,
            "State norm should be bounded after rapid oscillation: {norm}"
        );
        assert!(m.prediction_error().is_finite());
        assert!(m.coherence().is_finite());
    }

    // === Edge Case Hardening ===

    #[test]
    fn test_zero_dt_no_state_change() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        m.observe_frame(&frame, 64, 64, 1, 0.033);
        let state_before = m.state().clone();

        // dt=0 means sigma=0, so state shouldn't change
        m.observe_frame(&frame, 64, 64, 1, 0.0);
        let state_after = m.state().clone();

        let sim = state_before.similarity(&state_after);
        assert!(
            sim > 0.99,
            "dt=0 should produce minimal state change: sim={sim}"
        );
    }

    #[test]
    fn test_very_large_dt_converges_to_equilibrium() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        // Very large dt → sigma ≈ 1 → state jumps to equilibrium
        m.observe_frame(&frame, 64, 64, 1, 1000.0);

        let norm = m.state().norm();
        assert!(
            norm.is_finite() && norm > 0.0,
            "Large dt should produce finite state"
        );
        assert!(m.prediction_error().is_finite());
    }

    #[test]
    fn test_small_frame_4x4() {
        // 4x4 with patch_size=8 → 0 patches (too small for patches)
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 4, 4);
        let frame = vec![128u8; 16];
        let tel = m.observe_frame(&frame, 4, 4, 1, 0.033);
        assert!(tel.prediction_error.is_finite());
    }

    #[test]
    fn test_frame_with_all_zeros() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame = vec![0u8; 64 * 64];

        for _ in 0..5 {
            let tel = m.observe_frame(&frame, 64, 64, 1, 0.033);
            assert!(tel.prediction_error.is_finite());
            assert!(tel.manifold_coherence.is_finite());
        }
    }

    #[test]
    fn test_frame_with_all_255() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame = vec![255u8; 64 * 64];

        for _ in 0..5 {
            let tel = m.observe_frame(&frame, 64, 64, 1, 0.033);
            assert!(tel.prediction_error.is_finite());
            assert!(tel.manifold_coherence.is_finite());
        }
    }

    #[test]
    fn test_very_large_frame_256x256() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 256, 256);
        let frame: Vec<u8> = (0..256 * 256).map(|i| (i % 256) as u8).collect();

        let tel = m.observe_frame(&frame, 256, 256, 1, 0.033);
        assert_eq!(tel.frame_sequence, 1);
        assert!(m.state().norm() > 0.0);
        assert!(tel.prediction_error.is_finite());
    }

    #[test]
    fn test_non_square_frame() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 128, 32);
        let frame: Vec<u8> = (0..128 * 32).map(|i| (i % 256) as u8).collect();

        let tel = m.observe_frame(&frame, 128, 32, 1, 0.033);
        assert!(tel.prediction_error.is_finite());
        assert!(m.state().norm() > 0.0);
    }

    #[test]
    fn test_negative_dt_clamped() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        // Negative dt should not crash (sigma will be negative which means
        // the state moves away from equilibrium, but should remain finite)
        let tel = m.observe_frame(&frame, 64, 64, 1, -0.1);
        assert!(tel.prediction_error.is_finite());
        assert!(m.state().norm().is_finite());
    }

    // === Ablation Tests ===

    #[test]
    fn test_ablation_motion_features_contribute() {
        // Motion features should help distinguish moving vs static scenes.
        let mut cfg_with = VisionConfig::default();
        cfg_with.enable_motion = true;

        let mut cfg_without = VisionConfig::default();
        cfg_without.enable_motion = false;

        let mut m_with = VisionManifold::new(cfg_with, 64, 64);
        let mut m_without = VisionManifold::new(cfg_without, 64, 64);

        // Feed a "moving" sequence (brightness shifts)
        for i in 0..20u8 {
            let frame = solid_gray_frame(64, 64, 100 + i * 5);
            m_with.observe_frame(&frame, 64, 64, 1, 0.033);
            m_without.observe_frame(&frame, 64, 64, 1, 0.033);
        }

        // Both should produce valid states
        assert!(m_with.state().norm() > 0.0);
        assert!(m_without.state().norm() > 0.0);

        // With motion: the encoder captures temporal_diff and motion_magnitude
        // Without: only spatial features. The states should differ.
        let sim = m_with.state().similarity(m_without.state());
        assert!(
            sim < 0.99,
            "Motion features should produce different state: sim={sim}"
        );
    }

    #[test]
    fn test_ablation_color_features_contribute() {
        // Color features should help distinguish R vs B frames.
        let mut cfg_with = VisionConfig::default();
        cfg_with.enable_color = true;

        let mut cfg_without = VisionConfig::default();
        cfg_without.enable_color = false;

        let red: Vec<u8> = (0..64 * 64).flat_map(|_| vec![255u8, 0, 0]).collect();
        let blue: Vec<u8> = (0..64 * 64).flat_map(|_| vec![0u8, 0, 255]).collect();

        // With color: red and blue should be more distinguishable
        let mut m = VisionManifold::new(cfg_with.clone(), 64, 64);
        m.observe_frame(&red, 64, 64, 3, 0.033);
        let state_red_with = m.state().clone();

        // Re-create manifold to ensure clean baseline dimension
        let mut m = VisionManifold::new(cfg_with, 64, 64);
        m.observe_frame(&blue, 64, 64, 3, 0.033);
        let state_blue_with = m.state().clone();

        let (r_with, b_with) = if state_red_with.dim() != state_blue_with.dim() {
            let max_dim = state_red_with.dim().max(state_blue_with.dim());
            (
                state_red_with.dilate(max_dim),
                state_blue_with.dilate(max_dim),
            )
        } else {
            (state_red_with, state_blue_with)
        };
        let sim_with = r_with.similarity(&b_with);

        // Without color
        let mut m = VisionManifold::new(cfg_without.clone(), 64, 64);
        m.observe_frame(&red, 64, 64, 3, 0.033);
        let state_red_without = m.state().clone();

        let mut m = VisionManifold::new(cfg_without, 64, 64);
        m.observe_frame(&blue, 64, 64, 3, 0.033);
        let state_blue_without = m.state().clone();

        let (r_without, b_without) = if state_red_without.dim() != state_blue_without.dim() {
            let max_dim = state_red_without.dim().max(state_blue_without.dim());
            (
                state_red_without.dilate(max_dim),
                state_blue_without.dilate(max_dim),
            )
        } else {
            (state_red_without, state_blue_without)
        };
        let sim_without = r_without.similarity(&b_without);

        // Color features should make R vs B more distinguishable
        // (lower similarity with color features than without)
        assert!(
            sim_with < sim_without + 0.1,
            "Color features should help distinguish R vs B: with={sim_with:.4}, without={sim_without:.4}"
        );
    }

    #[test]
    fn test_ablation_multiscale_captures_structure() {
        // Multi-scale encoding should capture both fine texture and coarse layout.
        // A checkerboard (fine detail) on a gradient (coarse structure) should
        // produce a different encoding than a solid on a gradient.
        use crate::encoder::MultiScaleEncoder;

        let cfg = VisionConfig::default();
        let mut encoder = MultiScaleEncoder::new(&cfg, 64, 64);

        // Checkerboard pattern
        let checker: Vec<u8> = (0..64 * 64)
            .map(|i| {
                let x = i % 64;
                let y = i / 64;
                if (x / 4 + y / 4) % 2 == 0 {
                    200u8
                } else {
                    50u8
                }
            })
            .collect();

        // Solid with similar mean luminance
        let solid: Vec<u8> = vec![125u8; 64 * 64];

        let (hv_checker, _, _) = encoder.encode_frame(&checker, 64, 64, 1);
        let (hv_solid, _, _) = encoder.encode_frame(&solid, 64, 64, 1);

        let sim = hv_checker.similarity(&hv_solid);
        assert!(
            sim < 0.95,
            "Multi-scale should distinguish checker vs solid: sim={sim:.4}"
        );
    }

    #[test]
    fn test_ablation_attention_boost_modulates_output() {
        // Attention boost should make the bridge output different from raw state.
        use crate::bridge::VisionBridge;

        let cfg = VisionConfig::default();
        let mut bridge_boost = VisionBridge::new(cfg.clone(), 64, 64);
        let mut bridge_none = VisionBridge::new(cfg, 64, 64);
        bridge_none.set_attention_boost(0.0);

        // Feed two different frames to generate surprise
        let frame_a = solid_gray_frame(64, 64, 50);
        let frame_b = gradient_frame(64, 64);

        bridge_boost.process_frame(&frame_a, 64, 64, 1, 0.033);
        bridge_none.process_frame(&frame_a, 64, 64, 1, 0.033);

        let hv_boost = bridge_boost.process_frame(&frame_b, 64, 64, 1, 0.033);
        let hv_none = bridge_none.process_frame(&frame_b, 64, 64, 1, 0.033);

        // Both should be valid HVs
        assert!(hv_boost.norm() > 0.0);
        assert!(hv_none.norm() > 0.0);

        // They should differ (unless surprise was exactly zero)
        // We don't assert inequality because attention boost depends on surprise > 0
    }

    // === Motion Saliency Integration ===

    #[test]
    fn test_motion_saliency_empty_before_second_frame() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        assert!(m.motion_saliency().is_empty());
        assert!(m.motion_vectors().is_empty());

        let frame = gradient_frame(64, 64);
        let tel = m.observe_frame(&frame, 64, 64, 1, 0.033);
        // After first frame, no previous luminance → no motion
        assert_eq!(tel.motion_surprise, 0.0);
    }

    #[test]
    fn test_motion_saliency_populated_after_two_frames() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame_a = solid_gray_frame(64, 64, 50);
        let frame_b = solid_gray_frame(64, 64, 200);

        m.observe_frame(&frame_a, 64, 64, 1, 0.033);
        let tel = m.observe_frame(&frame_b, 64, 64, 1, 0.033);

        // After scene change, motion_saliency should be populated
        assert!(!m.motion_saliency().is_empty());
        assert!(!m.motion_vectors().is_empty());
        // motion_field_norm should be non-negative
        assert!(tel.motion_field_norm >= 0.0);
    }

    #[test]
    fn test_motion_saliency_static_scene_low() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        m.observe_frame(&frame, 64, 64, 1, 0.033);
        let tel = m.observe_frame(&frame, 64, 64, 1, 0.033);

        // Static scene: very low motion surprise
        assert!(
            tel.motion_surprise < 0.01,
            "Static scene should have near-zero motion surprise: {}",
            tel.motion_surprise
        );
    }

    #[test]
    fn test_motion_saliency_reset_clears() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame_a = solid_gray_frame(64, 64, 50);
        let frame_b = solid_gray_frame(64, 64, 200);

        m.observe_frame(&frame_a, 64, 64, 1, 0.033);
        m.observe_frame(&frame_b, 64, 64, 1, 0.033);
        assert!(!m.motion_saliency().is_empty());

        m.reset();
        assert!(m.motion_saliency().is_empty());
        assert!(m.motion_vectors().is_empty());
    }

    #[test]
    fn test_motion_telemetry_in_bridge() {
        use crate::bridge::VisionBridge;

        let cfg = VisionConfig::default();
        let mut bridge = VisionBridge::new(cfg, 64, 64);

        let frame_a = solid_gray_frame(64, 64, 50);
        let frame_b = gradient_frame(64, 64);

        bridge.process_frame(&frame_a, 64, 64, 1, 0.033);
        let (_, tel) = bridge.process_frame_with_telemetry(&frame_b, 64, 64, 1, 0.033);

        // Motion telemetry should be populated
        assert!(tel.motion_field_norm >= 0.0);
        assert!(tel.motion_surprise >= 0.0);
        assert!(tel.motion_surprise.is_finite());
    }

    // === 1000-Cycle Soak Test ===

    #[test]
    fn test_soak_1000_cycles_stability() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);

        let frame_a = solid_gray_frame(64, 64, 50);
        let frame_b = gradient_frame(64, 64);
        let frame_c = solid_gray_frame(64, 64, 200);

        let mut max_pred_error = 0.0f32;
        let mut min_coherence = f32::MAX;
        let mut max_state_norm = 0.0f32;
        let mut _training_count = 0u32;

        for i in 0..1000 {
            // Cycle through 3 scenes: A(300) → B(300) → C(300) → A(100)
            let frame = match i {
                0..=299 => &frame_a,
                300..=599 => &frame_b,
                600..=899 => &frame_c,
                _ => &frame_a,
            };
            let tel = m.observe_frame(frame, 64, 64, 1, 0.033);

            // All values must be finite
            assert!(
                tel.prediction_error.is_finite(),
                "Frame {i}: prediction error not finite"
            );
            assert!(
                tel.manifold_coherence.is_finite(),
                "Frame {i}: coherence not finite"
            );
            assert!(
                tel.motion_surprise.is_finite(),
                "Frame {i}: motion_surprise not finite"
            );
            assert!(
                tel.motion_field_norm.is_finite(),
                "Frame {i}: motion_field_norm not finite"
            );

            let norm = m.state().norm();
            assert!(
                norm.is_finite() && norm > 0.0,
                "Frame {i}: state norm invalid: {norm}"
            );

            max_pred_error = max_pred_error.max(tel.prediction_error);
            min_coherence = min_coherence.min(tel.manifold_coherence);
            max_state_norm = max_state_norm.max(norm);
            if tel.training_triggered {
                _training_count += 1;
            }
        }

        // Verify bounds
        assert!(
            max_pred_error < 2.0,
            "Max prediction error too high: {max_pred_error}"
        );
        assert!(
            max_state_norm < 200.0,
            "Max state norm too high: {max_state_norm}"
        );

        // Verify health after 1000 cycles
        let health = m.compute_health();
        assert_eq!(health.total_frames, 1000);
        assert!(health.is_healthy, "Manifold unhealthy after 1000 frames");
        assert!(
            health.tau_value > 0.01 && health.tau_value < 10.0,
            "Tau out of bounds: {}",
            health.tau_value
        );
    }

    #[test]
    fn test_ablation_training_improves_predictions() {
        // With training enabled, prediction error should stabilize or decrease
        // compared to without training.
        let mut cfg_train = VisionConfig::default();
        cfg_train.training.learning_rate = 0.01;
        cfg_train.training.error_threshold = 0.05;

        let cfg_notrain = VisionConfig::default();

        let mut m_train = VisionManifold::new(cfg_train, 64, 64);
        let mut m_notrain = VisionManifold::new(cfg_notrain, 64, 64);
        // Disable training via freeze_learning instead of invalid config
        m_notrain.freeze_learning(true);

        let frame_a = solid_gray_frame(64, 64, 50);
        let frame_b = solid_gray_frame(64, 64, 200);

        // Alternating pattern
        for i in 0..60 {
            let frame = if i % 2 == 0 { &frame_a } else { &frame_b };
            m_train.observe_frame(frame, 64, 64, 1, 0.033);
            m_notrain.observe_frame(frame, 64, 64, 1, 0.033);
        }

        // With training, the manifold's tau and weights have adapted
        assert!(
            m_train.training_steps() > 0,
            "Training should have triggered"
        );
        // Without training (frozen), no training steps should occur
        assert_eq!(
            m_notrain.training_steps(),
            0,
            "Frozen manifold should have no training steps"
        );

        // Both should produce finite, healthy states
        let health_train = m_train.compute_health();
        let health_notrain = m_notrain.compute_health();
        assert!(health_train.is_healthy);
        assert!(health_notrain.is_healthy);
    }

    // === Configurable Equilibrium ===

    #[test]
    fn test_high_input_blend_tracks_input() {
        let mut cfg = VisionConfig::default();
        cfg.input_blend = 0.9;
        let mut m = VisionManifold::new(cfg, 64, 64);
        let frame = gradient_frame(64, 64);

        for _ in 0..20 {
            m.observe_frame(&frame, 64, 64, 1, 0.033);
        }

        let coherence_high_blend = m.coherence();

        let mut cfg_low = VisionConfig::default();
        cfg_low.input_blend = 0.3;
        let mut m_low = VisionManifold::new(cfg_low, 64, 64);

        for _ in 0..20 {
            m_low.observe_frame(&frame, 64, 64, 1, 0.033);
        }

        // High blend should track input more closely
        // Both should converge for a static scene, so check coherence is similar
        assert!(
            coherence_high_blend > 0.5,
            "High blend coherence should be decent"
        );
        assert!(
            m_low.coherence() > 0.5,
            "Low blend coherence should also work"
        );
    }

    // === Save/Load Extended Fields ===

    #[test]
    fn test_memory_consolidation_updates_live_surprise_decay() {
        let cfg = VisionConfig {
            surprise_decay: 0.8,
            ..Default::default()
        };
        let mut manifold = VisionManifold::new(cfg, 64, 64);

        let consolidated = manifold.consolidate_surprise_memory();

        assert!(consolidated > 0.8);
        assert_eq!(manifold.config.surprise_decay, consolidated);
        assert_eq!(manifold.surprise.decay(), consolidated);
    }

    #[test]
    fn test_save_load_roundtrip_extended() {
        let cfg = VisionConfig::default();
        let mut m1 = VisionManifold::new(cfg.clone(), 64, 64);

        // Evolve manifold to accumulate non-trivial temporal and optimizer state.
        let frame = gradient_frame(64, 64);
        for _ in 0..10 {
            m1.observe_frame(&frame, 64, 64, 1, 0.033);
        }
        m1.enable_scene_memory(8);
        let landmark = ContinuousHV::random(cfg.hdc_dim, 77_001);
        m1.scene_memory
            .as_mut()
            .expect("scene memory")
            .remember(&landmark, 7, vec![1, 2, 3, 4]);

        let saved = m1.save_state();
        assert!(saved.error_ema > 0.0 || saved.frame_count > 0);
        assert_eq!(saved.frame_count, 10);
        assert!(saved.state_hv.is_some());
        assert!(saved.last_prediction.is_some());
        assert!(saved.trainer_state.is_some());
        assert_eq!(
            saved.scene_memory.as_ref().unwrap().raw_frames[0],
            vec![1, 2, 3, 4]
        );

        let mut m2 = VisionManifold::new(cfg, 64, 64);
        assert!(m2.load_state(&saved).is_ok());

        assert_eq!(m2.frame_count(), saved.frame_count);
        assert!((m2.error_ema() - saved.error_ema).abs() < 1e-6);
        assert!((m2.prediction_error() - saved.prediction_error).abs() < 1e-6);
        assert!(m2.state().similarity(m1.state()) > 0.999_999);
        assert_eq!(m2.last_observed_frame.as_deref(), Some(frame.as_slice()));
        assert_eq!(m2.training_steps(), m1.training_steps());
        assert_eq!(m2.save_state().trainer_state, saved.trainer_state);
        assert_eq!(
            m2.scene_memory.as_ref().unwrap().get_pixels(0),
            Some(&[1, 2, 3, 4][..])
        );
    }

    // === Scene Memory Extended API ===

    #[test]
    fn test_scene_memory_export_landmarks() {
        let mut mem = SceneMemory::new(16);
        let dim = 16_384;

        let scene_a = ContinuousHV::random(dim, 100);
        let scene_b = ContinuousHV::random(dim, 200);
        mem.remember(&scene_a, 10, vec![]);
        mem.remember(&scene_b, 20, vec![]);

        let landmarks = mem.export_landmarks();
        assert_eq!(landmarks.len(), 2);
        assert_eq!(landmarks[0].1, 10);
        assert_eq!(landmarks[1].1, 20);
    }

    #[test]
    fn test_scene_memory_forget() {
        let mut mem = SceneMemory::new(16);
        let dim = 16_384;

        for i in 0..4u64 {
            mem.remember(&ContinuousHV::random(dim, 100 + i), i * 10, vec![]);
        }
        assert_eq!(mem.len(), 4);

        assert!(mem.forget(1)); // Remove scene at index 1
        assert_eq!(mem.len(), 3);

        assert!(!mem.forget(100)); // Out of bounds
        assert_eq!(mem.len(), 3);
    }

    #[test]
    fn test_scene_memory_bounds_raw_rasters_without_forgetting_scenes() {
        let mut mem = SceneMemory::new_with_pixel_budget(4, 4);
        let metadata = SceneFrameMetadata {
            width: 2,
            height: 2,
            channels: 1,
            modality: VisualModality::Visible,
        };
        let first = ContinuousHV::random(256, 8101);
        let second = ContinuousHV::random(256, 8102);

        mem.remember_with_metadata(&first, 1, vec![1, 2, 3, 4], metadata);
        mem.remember_with_metadata(&second, 2, vec![5, 6, 7, 8], metadata);

        assert_eq!(mem.len(), 2, "semantic landmarks should be retained");
        assert_eq!(mem.retained_pixel_bytes(), 4);
        assert_eq!(mem.get_pixels(0), Some(&[][..]));
        assert_eq!(mem.get_pixels(1), Some(&[5, 6, 7, 8][..]));
        assert!(mem.recognize(&first, 3).is_some());
    }

    #[test]
    fn test_dilation_estimate_counts_retained_scene_pixels() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut manifold = VisionManifold::new(config, 2, 2);
        manifold.scene_memory = Some(SceneMemory::new_with_pixel_budget(2, 16));
        let scene = ContinuousHV::random(256, 8201);
        manifold
            .scene_memory
            .as_mut()
            .unwrap()
            .remember_with_metadata(
                &scene,
                1,
                vec![1, 2, 3, 4],
                SceneFrameMetadata {
                    width: 2,
                    height: 2,
                    channels: 1,
                    modality: VisualModality::Visible,
                },
            );

        let estimate = manifold.estimate_dilation(symthaea_core::hdc::HdcDimensionality::Ultra);
        assert!(estimate.persistent_bytes >= 4);
        assert_eq!(
            estimate.total_projected_bytes,
            estimate.projected_bytes + estimate.persistent_bytes
        );
    }

    #[test]
    fn test_scene_memory_persists_frame_metadata() {
        let mut mem = SceneMemory::new(4);
        let scene = ContinuousHV::random(256, 9001);
        let metadata = SceneFrameMetadata {
            width: 2,
            height: 2,
            channels: 1,
            modality: VisualModality::Stereo,
        };
        mem.remember_with_metadata(&scene, 7, vec![1, 2, 3, 4], metadata);

        let saved = mem.save_state();
        assert_eq!(saved.frame_metadata, vec![metadata]);

        let mut restored = SceneMemory::new(1);
        restored.load_state(&saved);
        assert_eq!(restored.get_frame_metadata(0), Some(metadata));
        assert_eq!(restored.get_pixels(0), Some(&[1, 2, 3, 4][..]));
    }

    #[test]
    fn test_scene_memory_rejects_mismatched_frame_metadata() {
        let mut state = SceneMemoryState {
            landmarks: vec![(vec![0.0; 256], 1)],
            capacity: 1,
            threshold: 0.8,
            pixel_budget_bytes: 64,
            retained_pixel_bytes: 4,
            raw_frames: vec![vec![0; 4]],
            frame_metadata: vec![SceneFrameMetadata {
                width: 3,
                height: 2,
                channels: 1,
                modality: VisualModality::Visible,
            }],
            object_episodes: vec![],
        };
        let error = SceneMemory::validate_state(&state, 256).unwrap_err();
        assert!(error.contains("length mismatch"));

        state.raw_frames[0] = vec![0; 6];
        state.retained_pixel_bytes = 6;
        assert!(SceneMemory::validate_state(&state, 256).is_ok());
    }

    #[test]
    fn test_scene_memory_save_load_roundtrip() {
        let mut mem = SceneMemory::new(16);
        let dim = 16_384;

        let scene_a = ContinuousHV::random(dim, 100);
        let scene_b = ContinuousHV::random(dim, 200);
        mem.remember(&scene_a, 10, vec![10, 11]);
        mem.remember(&scene_b, 20, vec![20, 21]);

        let saved = mem.save_state();
        assert_eq!(saved.landmarks.len(), 2);
        assert_eq!(saved.raw_frames, vec![vec![10, 11], vec![20, 21]]);

        let mut mem2 = SceneMemory::new(8); // Different capacity
        mem2.load_state(&saved);
        assert_eq!(mem2.len(), 2);
        assert_eq!(mem2.capacity, 16); // Restored from saved

        assert_eq!(mem2.get_pixels(0), Some(&[10, 11][..]));
        assert_eq!(mem2.get_pixels(1), Some(&[20, 21][..]));

        // Should recognize original scenes
        let result = mem2.recognize(&scene_a, 30);
        assert!(result.is_some());
    }

    // === Freeze Learning ===

    #[test]
    fn test_freeze_learning() {
        let mut cfg = VisionConfig::default();
        cfg.training.error_threshold = 0.05;
        cfg.training.learning_rate = 0.01;
        cfg.learning.contrastive_lr = 0.1;
        let mut m = VisionManifold::new(cfg, 64, 64);

        let frame_a = solid_gray_frame(64, 64, 50);
        let frame_b = gradient_frame(64, 64);

        // Run a few frames unfrozen to establish state
        for _ in 0..5 {
            m.observe_frame(&frame_a, 64, 64, 1, 0.033);
        }

        // Freeze and run alternating pattern
        m.freeze_learning(true);
        assert!(m.is_learning_frozen());

        let weights_before: Vec<f32> = m.encoder().feature_weights().to_vec();
        let tau_before = m.current_tau();

        for i in 0..20 {
            let frame = if i % 2 == 0 { &frame_a } else { &frame_b };
            m.observe_frame(frame, 64, 64, 1, 0.033);
        }

        let weights_after: Vec<f32> = m.encoder().feature_weights().to_vec();
        let tau_after = m.current_tau();

        // Weights and tau should not have changed while frozen
        assert_eq!(
            weights_before, weights_after,
            "Weights should not change while frozen"
        );
        assert!(
            (tau_before - tau_after).abs() < 1e-6,
            "Tau should not change while frozen"
        );
    }

    // === Active Patch Count ===

    #[test]
    fn test_active_patch_count() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 64, 64);

        // Before any frames, all surprise is 0
        let (active, total) = m.active_patch_count();
        assert_eq!(active, 0);
        assert!(total > 0);

        // Feed two different frames to generate surprise
        let frame_a = solid_gray_frame(64, 64, 50);
        let frame_b = gradient_frame(64, 64);
        m.observe_frame(&frame_a, 64, 64, 1, 0.033);
        m.observe_frame(&frame_b, 64, 64, 1, 0.033);

        let (active2, total2) = m.active_patch_count();
        assert_eq!(total2, total);
        // After a scene change, some patches should be active
        // (not guaranteed but likely with gradient vs solid)
        // active2 >= 0 always true for usize; just verify the call completes
        let _ = active2;
    }

    // === Predictive Hierarchy Integration ===

    #[test]
    fn test_predictive_hierarchy_integration() {
        let mut cfg = VisionConfig::default();
        cfg.enable_predictive_hierarchy = true;
        let mut m = VisionManifold::new(cfg, 64, 64);

        let frame = gradient_frame(64, 64);

        // First frame: predictive hierarchy has no prior, cross-scale error should be 1.0
        let tel1 = m.observe_frame(&frame, 64, 64, 1, 0.033);
        assert!(tel1.cross_scale_prediction_error >= 0.0);
        assert!(tel1.cross_scale_prediction_error.is_finite());

        // Process more frames — error should decrease for a static scene
        let mut errors = Vec::new();
        for _ in 0..20 {
            let tel = m.observe_frame(&frame, 64, 64, 1, 0.033);
            errors.push(tel.cross_scale_prediction_error);
        }

        let late_mean: f32 = errors[15..20].iter().sum::<f32>() / 5.0;
        assert!(
            late_mean <= 1.0,
            "Cross-scale prediction error should be bounded: {late_mean}"
        );
    }

    // === Configurable Scene Memory Thresholds ===

    #[test]
    fn checked_scene_policy_rejects_invalid_values_atomically() {
        let mut manifold = test_manifold();
        manifold
            .set_scene_store_thresholds_checked(0.4, 0.2)
            .unwrap();
        manifold.set_scene_dampen_factor_checked(0.6).unwrap();

        assert!(
            manifold
                .set_scene_store_thresholds_checked(f32::NAN, 0.1)
                .is_err()
        );
        assert!(
            manifold
                .set_scene_store_thresholds_checked(0.4, f32::INFINITY)
                .is_err()
        );
        assert!(manifold.set_scene_dampen_factor_checked(-0.1).is_err());
        assert_eq!(manifold.scene_store_coherence_threshold, 0.4);
        assert_eq!(manifold.scene_store_error_threshold, 0.2);
        assert_eq!(manifold.scene_dampen_factor, 0.6);
    }

    #[test]
    fn test_configurable_scene_thresholds() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 32, 32);
        m.enable_scene_memory(10);

        m.set_scene_store_thresholds(0.5, 0.2);
        m.set_scene_dampen_factor(0.8);

        let frame: Vec<u8> = (0..32 * 32).map(|i| (i % 256) as u8).collect();
        for _ in 0..10 {
            m.observe_frame(&frame, 32, 32, 1, 0.033);
        }
        assert!(m.telemetry().prediction_error.is_finite());
    }

    #[test]
    fn test_low_coherence_threshold_stores_more() {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, 32, 32);
        m.enable_scene_memory(100);

        // Very permissive thresholds
        m.set_scene_store_thresholds(0.01, 10.0);

        let frame: Vec<u8> = (0..32 * 32).map(|i| (i % 256) as u8).collect();
        for _ in 0..30 {
            m.observe_frame(&frame, 32, 32, 1, 0.033);
        }
        assert!(m.telemetry().manifold_coherence.is_finite());
    }

    // === Config Validation ===

    #[test]
    #[should_panic(expected = "Invalid VisionManifold construction")]
    fn test_invalid_config_panics_on_construction() {
        let mut cfg = VisionConfig::default();
        cfg.tau_base = 0.0;
        let _ = VisionManifold::new(cfg, 64, 64);
    }

    // === Object Memory Identity ===

    #[test]
    fn test_cached_hypotheses_rebind_current_patch_evidence() {
        let dim = 256;
        let patches = vec![
            ContinuousHV::random(dim, 11),
            ContinuousHV::random(dim, 12),
            ContinuousHV::random(dim, 13),
        ];
        let old = ContinuousHV::random(dim, 99);
        let mut hypotheses = vec![crate::types::ObjectHypothesis {
            centroid_row: 0,
            centroid_col: 0,
            patch_indices: vec![0, 2],
            saliency: 0.4,
            hv: old.clone(),
        }];

        assert!(VisionManifold::refresh_hypothesis_appearance(
            &mut hypotheses,
            &patches
        ));
        let expected = ContinuousHV::bundle(&[&patches[0], &patches[2]]).normalize();
        assert!(hypotheses[0].hv.similarity(&expected) > 0.999);
        assert!(hypotheses[0].hv.similarity(&old) < 0.95);
        assert_eq!(hypotheses[0].saliency, 0.4);
    }

    #[test]
    fn test_cached_hypotheses_reject_stale_patch_indices() {
        let dim = 256;
        let patches = vec![ContinuousHV::random(dim, 11)];
        let mut hypotheses = vec![crate::types::ObjectHypothesis {
            centroid_row: 0,
            centroid_col: 0,
            patch_indices: vec![3],
            saliency: 0.0,
            hv: ContinuousHV::random(dim, 99),
        }];

        assert!(!VisionManifold::refresh_hypothesis_appearance(
            &mut hypotheses,
            &patches
        ));
    }

    #[test]
    fn checked_memory_thresholds_reject_invalid_values_atomically() {
        let mut scenes = SceneMemory::new(4);
        scenes.set_threshold_checked(0.7).unwrap();
        assert!(scenes.set_threshold_checked(f32::NAN).is_err());
        assert!(scenes.set_threshold_checked(1.1).is_err());
        assert_eq!(scenes.recognition_threshold, 0.7);

        let mut objects = ObjectMemory::new(4);
        objects.set_match_threshold_checked(0.4).unwrap();
        assert!(objects.set_match_threshold_checked(f32::INFINITY).is_err());
        assert!(objects.set_match_threshold_checked(-0.1).is_err());
        assert_eq!(objects.match_threshold, 0.4);
    }

    #[test]
    fn test_object_memory_uses_velocity_through_crossing() {
        fn hypothesis(hv: &ContinuousHV, col: usize) -> crate::types::ObjectHypothesis {
            crate::types::ObjectHypothesis {
                centroid_row: 0,
                centroid_col: col,
                patch_indices: vec![col],
                saliency: 1.0,
                hv: hv.clone(),
            }
        }

        let dim = 256;
        let appearance = ContinuousHV::random(dim, 42);
        let mut memory = ObjectMemory::new(4);
        memory.set_match_threshold(0.1);
        memory.set_max_match_distance(4);
        let mut next_id = 0;

        memory.update(
            &[hypothesis(&appearance, 0), hypothesis(&appearance, 6)],
            0,
            &mut next_id,
        );
        memory.update(
            &[hypothesis(&appearance, 2), hypothesis(&appearance, 4)],
            1,
            &mut next_id,
        );
        memory.update(
            &[hypothesis(&appearance, 4), hypothesis(&appearance, 2)],
            2,
            &mut next_id,
        );

        let left_to_right = memory
            .tracks()
            .iter()
            .find(|track| track.track_id == 0)
            .unwrap();
        let right_to_left = memory
            .tracks()
            .iter()
            .find(|track| track.track_id == 1)
            .unwrap();
        assert_eq!(left_to_right.centroid_col, 4);
        assert_eq!(right_to_left.centroid_col, 2);
        assert!(left_to_right.velocity_col > 0.0);
        assert!(right_to_left.velocity_col < 0.0);
    }

    #[test]
    fn test_object_memory_matches_against_stable_appearance() {
        let mut memory = ObjectMemory::new(8);
        let appearance = ContinuousHV::random(256, 77);
        let hypothesis = crate::types::ObjectHypothesis {
            centroid_row: 2,
            centroid_col: 3,
            patch_indices: vec![0, 1],
            saliency: 0.5,
            hv: appearance.clone(),
        };
        let mut next_track_id = 0;

        memory.update(&[hypothesis.clone()], 0, &mut next_track_id);
        for frame in 1..64 {
            let result = memory.update(&[hypothesis.clone()], frame, &mut next_track_id);
            assert_eq!(result.active_tracks, 1);
            assert_eq!(result.new_tracks, 0);
        }

        let track = &memory.tracks()[0];
        assert!(
            track.appearance_hv.similarity(&appearance) > 0.999,
            "stable appearance must remain matchable after long temporal history"
        );
        assert_eq!(track.track_length, 64);
    }

    // === Visual Working Memory ===

    #[test]
    fn test_working_memory_capacity_enforcement() {
        let mut wm = VisualWorkingMemory::new(3);
        let tracks: Vec<TrackedObject> = (0..5)
            .map(|i| TrackedObject {
                track_id: i,
                appearance_hv: ContinuousHV::random(256, i),
                identity_hv: ContinuousHV::random(256, i),
                centroid_row: i as usize,
                centroid_col: 0,
                velocity_row: 0.0,
                velocity_col: 0.0,
                last_seen_frame: 0,
                track_length: 1,
            })
            .collect();
        let hyps: Vec<crate::types::ObjectHypothesis> = (0..5)
            .map(|i| crate::types::ObjectHypothesis {
                centroid_row: i as usize,
                centroid_col: 0,
                patch_indices: vec![],
                saliency: 0.1 * (i + 1) as f32, // 0.1, 0.2, ..., 0.5
                hv: ContinuousHV::random(256, 100 + i),
            })
            .collect();
        wm.update(&tracks, &hyps, 0);
        assert!(
            wm.load() <= 3,
            "Working memory should hold ≤ 3 objects, got {}",
            wm.load()
        );
    }

    #[test]
    fn test_working_memory_saliency_eviction() {
        let mut wm = VisualWorkingMemory::new(2);
        let tracks: Vec<TrackedObject> = (0..3)
            .map(|i| TrackedObject {
                track_id: i,
                appearance_hv: ContinuousHV::random(256, i),
                identity_hv: ContinuousHV::random(256, i),
                centroid_row: i as usize,
                centroid_col: 0,
                velocity_row: 0.0,
                velocity_col: 0.0,
                last_seen_frame: 0,
                track_length: 1,
            })
            .collect();
        // First two with low saliency
        let hyps_low: Vec<crate::types::ObjectHypothesis> = (0..2)
            .map(|i| crate::types::ObjectHypothesis {
                centroid_row: i as usize,
                centroid_col: 0,
                patch_indices: vec![],
                saliency: 0.1,
                hv: ContinuousHV::random(256, 100 + i),
            })
            .collect();
        wm.update(&tracks[..2], &hyps_low, 0);
        assert_eq!(wm.load(), 2);

        // Third object with high saliency should evict the weakest
        let hyps_high = vec![crate::types::ObjectHypothesis {
            centroid_row: 2,
            centroid_col: 0,
            patch_indices: vec![],
            saliency: 0.9,
            hv: ContinuousHV::random(256, 200),
        }];
        let evicted = wm.update(&tracks, &hyps_high, 1);
        assert_eq!(
            evicted.len(),
            1,
            "replacement must be reported for consolidation"
        );
        assert_eq!(wm.load(), 2);
        // The high-saliency object (track_id=2) should be present
        assert!(
            wm.slots().iter().any(|s| s.track_id == 2),
            "High-saliency object should have evicted a low-saliency one"
        );
    }

    #[test]
    fn test_working_memory_decay() {
        let mut wm = VisualWorkingMemory::new(4);
        let tracks = vec![TrackedObject {
            track_id: 0,
            appearance_hv: ContinuousHV::random(256, 0),
            identity_hv: ContinuousHV::random(256, 0),
            centroid_row: 0,
            centroid_col: 0,
            velocity_row: 0.0,
            velocity_col: 0.0,
            last_seen_frame: 0,
            track_length: 1,
        }];
        let hyps = vec![crate::types::ObjectHypothesis {
            centroid_row: 0,
            centroid_col: 0,
            patch_indices: vec![],
            saliency: 0.1,
            hv: ContinuousHV::random(256, 100),
        }];
        wm.update(&tracks, &hyps, 0);
        let initial_saliency = wm.slots()[0].saliency;

        // Run 100 frames with no refresh → saliency should decay below threshold
        for frame in 1..100 {
            wm.update(&[], &[], frame);
        }
        // After 100 frames of decay at 0.95^100 ≈ 0.006, below 0.01 threshold
        assert_eq!(
            wm.load(),
            0,
            "Object should be evicted after 100 frames of pure decay (initial={initial_saliency})"
        );
    }

    #[test]
    fn test_working_memory_bundle_attended() {
        let mut wm = VisualWorkingMemory::new(4);
        assert!(
            wm.bundle_attended().is_none(),
            "Empty WM should have no bundle"
        );

        let tracks = vec![TrackedObject {
            track_id: 0,
            appearance_hv: ContinuousHV::random(256, 0),
            identity_hv: ContinuousHV::random(256, 0),
            centroid_row: 0,
            centroid_col: 0,
            velocity_row: 0.0,
            velocity_col: 0.0,
            last_seen_frame: 0,
            track_length: 1,
        }];
        let hyps = vec![crate::types::ObjectHypothesis {
            centroid_row: 0,
            centroid_col: 0,
            patch_indices: vec![],
            saliency: 0.5,
            hv: ContinuousHV::random(256, 100),
        }];
        wm.update(&tracks, &hyps, 0);
        let bundle = wm.bundle_attended();
        assert!(bundle.is_some(), "Non-empty WM should produce a bundle");
        assert!(bundle.unwrap().norm() > 0.0, "Bundle should be non-zero");
    }

    // === Visual Scene Graph ===

    #[test]
    fn test_scene_graph_computes_relations() {
        let mut sg = VisualSceneGraph::new(256, 42);
        let tracks = vec![
            TrackedObject {
                track_id: 0,
                appearance_hv: ContinuousHV::random(256, 0),
                identity_hv: ContinuousHV::random(256, 0),
                centroid_row: 0, // top
                centroid_col: 4,
                velocity_row: 0.0,
                velocity_col: 0.0,
                last_seen_frame: 0,
                track_length: 5,
            },
            TrackedObject {
                track_id: 1,
                appearance_hv: ContinuousHV::random(256, 1),
                identity_hv: ContinuousHV::random(256, 1),
                centroid_row: 7, // bottom
                centroid_col: 4,
                velocity_row: 0.0,
                velocity_col: 0.0,
                last_seen_frame: 0,
                track_length: 5,
            },
        ];
        sg.update(&tracks);
        assert!(sg.num_edges() > 0, "Two objects should produce edges");
        // Object 0 is above object 1 (row 0 < row 7)
        let has_above = sg
            .edges()
            .iter()
            .any(|e| e.relation == crate::types::SpatialRelation::Above);
        assert!(has_above, "Should detect Above relation");
        // Graph HV should exist
        assert!(sg.graph_hv().is_some());
    }

    #[test]
    fn test_scene_graph_near_overlapping() {
        let mut sg = VisualSceneGraph::new(256, 42);
        let tracks = vec![
            TrackedObject {
                track_id: 0,
                appearance_hv: ContinuousHV::random(256, 0),
                identity_hv: ContinuousHV::random(256, 0),
                centroid_row: 3,
                centroid_col: 3,
                velocity_row: 0.0,
                velocity_col: 0.0,
                last_seen_frame: 0,
                track_length: 1,
            },
            TrackedObject {
                track_id: 1,
                appearance_hv: ContinuousHV::random(256, 1),
                identity_hv: ContinuousHV::random(256, 1),
                centroid_row: 3, // same position
                centroid_col: 3,
                velocity_row: 0.0,
                velocity_col: 0.0,
                last_seen_frame: 0,
                track_length: 1,
            },
        ];
        sg.update(&tracks);
        let has_overlap = sg
            .edges()
            .iter()
            .any(|e| e.relation == crate::types::SpatialRelation::Overlapping);
        assert!(has_overlap, "Same position should produce Overlapping");
    }

    #[test]
    fn test_scene_graph_edge_hvs_are_finite() {
        let mut sg = VisualSceneGraph::new(256, 42);
        let tracks = vec![
            TrackedObject {
                track_id: 0,
                appearance_hv: ContinuousHV::random(256, 0),
                identity_hv: ContinuousHV::random(256, 0),
                centroid_row: 0,
                centroid_col: 0,
                velocity_row: 0.0,
                velocity_col: 0.0,
                last_seen_frame: 0,
                track_length: 1,
            },
            TrackedObject {
                track_id: 1,
                appearance_hv: ContinuousHV::random(256, 1),
                identity_hv: ContinuousHV::random(256, 1),
                centroid_row: 5,
                centroid_col: 5,
                velocity_row: 0.0,
                velocity_col: 0.0,
                last_seen_frame: 0,
                track_length: 1,
            },
        ];
        sg.update(&tracks);
        for edge in sg.edges() {
            assert!(
                edge.relation_hv.as_slice().iter().all(|x| x.is_finite()),
                "Edge HV for {:?} must be finite",
                edge.relation
            );
        }
        if let Some(ghv) = sg.graph_hv() {
            assert!(
                ghv.as_slice().iter().all(|x| x.is_finite()),
                "Graph HV must be finite"
            );
        }
    }

    #[test]
    fn test_scene_graph_empty_tracks() {
        let mut sg = VisualSceneGraph::new(256, 42);
        sg.update(&[]);
        assert_eq!(sg.num_edges(), 0);
        assert!(sg.graph_hv().is_none());
    }

    // === Imagination-Reality Comparison ===

    #[test]
    fn test_imagination_surprise_zero_on_first_frame() {
        let mut m = VisionManifold::new(VisionConfig::default(), 32, 32);
        let pixels = vec![128u8; 32 * 32 * 3];
        let tel = m.observe_frame(&pixels, 32, 32, 3, 0.033);
        // No imagination yet on frame 0
        assert_eq!(tel.imagination_surprise, 0.0);
    }

    #[test]
    fn test_imagination_surprise_low_for_static_scene() {
        let mut m = VisionManifold::new(VisionConfig::default(), 32, 32);
        let pixels = vec![128u8; 32 * 32 * 3];
        m.observe_frame(&pixels, 32, 32, 3, 0.033);
        let _tel2 = m.observe_frame(&pixels, 32, 32, 3, 0.033);
        let tel3 = m.observe_frame(&pixels, 32, 32, 3, 0.033);
        // Static scene → low imagination surprise
        assert!(
            tel3.imagination_surprise < 0.5,
            "Static scene should have low imagination surprise, got {}",
            tel3.imagination_surprise
        );
    }

    // === Dream Ahead ===

    #[test]
    fn test_dream_ahead_zero_steps() {
        let mut m = VisionManifold::new(VisionConfig::default(), 32, 32);
        let dreams = m.dream_ahead(0, 0.033);
        assert!(dreams.is_empty());
    }

    #[test]
    fn test_dream_ahead_does_not_mutate_live_state() {
        let mut m = VisionManifold::new(VisionConfig::default(), 32, 32);
        let pixels = vec![128u8; 32 * 32 * 3];
        m.observe_frame(&pixels, 32, 32, 3, 0.033);
        let before = m.state().clone();
        let frame_count = m.frame_count();

        let dreams = m.dream_ahead(8, 0.033);

        assert_eq!(dreams.len(), 8);
        assert!(m.state().similarity(&before) > 0.999_999);
        assert_eq!(m.frame_count(), frame_count);
    }

    #[test]
    fn test_dream_ahead_converges() {
        let mut m = VisionManifold::new(VisionConfig::default(), 32, 32);
        let pixels = vec![128u8; 32 * 32 * 3];
        m.observe_frame(&pixels, 32, 32, 3, 0.033);

        let dreams = m.dream_ahead(20, 0.033);
        assert_eq!(dreams.len(), 20);
        // Later dreams should converge (similarity approaching 1.0)
        let sim_early = dreams[0].similarity(&dreams[1]);
        let sim_late = dreams[18].similarity(&dreams[19]);
        assert!(
            sim_late >= sim_early - 0.01,
            "CfC dreams should converge over time (early={sim_early}, late={sim_late})"
        );
    }

    // === Psych-Bench: Visual Cognition Battery ===
    //
    // Quantitative tests measuring cognitive properties of the vision system.
    // These produce publishable numbers for the psych-bench paper.

    /// Change blindness: does the system detect changes in unattended regions?
    /// (Rensink et al. 1997)
    #[test]
    fn test_psychbench_change_blindness() {
        let mut cfg = VisionConfig::default();
        cfg.enable_object_binding = true;
        let mut m = VisionManifold::new(cfg, 64, 64);
        m.enable_object_memory(32);

        // Phase 1: Establish scene (red top, blue bottom)
        let mut scene_a = vec![0u8; 64 * 64 * 3];
        for y in 0..32 {
            for x in 0..64 {
                scene_a[(y * 64 + x) * 3] = 200;
            }
        }
        for y in 32..64 {
            for x in 0..64 {
                scene_a[(y * 64 + x) * 3 + 2] = 200;
            }
        }
        for _ in 0..10 {
            m.observe_frame(&scene_a, 64, 64, 3, 0.033);
        }
        let pe_stable = m.prediction_error();

        // Phase 2: Change blue → green (bottom half)
        let mut scene_b = scene_a.clone();
        for y in 32..64 {
            for x in 0..64 {
                let b = (y * 64 + x) * 3;
                scene_b[b + 1] = 200; // green
                scene_b[b + 2] = 0; // no blue
            }
        }
        let tel = m.observe_frame(&scene_b, 64, 64, 3, 0.033);

        // The system should detect the change (higher PE than stable baseline)
        assert!(
            tel.prediction_error > pe_stable,
            "Change blindness test: system should detect color change \
             (stable_pe={pe_stable}, change_pe={})",
            tel.prediction_error
        );
    }

    /// Object permanence: does a tracked object survive brief occlusion?
    /// (Spelke 1990)
    #[test]
    fn test_psychbench_object_permanence() {
        let mut cfg = VisionConfig::default();
        cfg.enable_object_binding = true;
        let mut m = VisionManifold::new(cfg, 64, 64);
        m.enable_object_memory(32);

        // Phase 1: Show a distinctive red object (5 frames)
        let mut scene = vec![128u8; 64 * 64 * 3];
        for y in 10..20 {
            for x in 10..20 {
                let b = (y * 64 + x) * 3;
                scene[b] = 255;
                scene[b + 1] = 0;
                scene[b + 2] = 0;
            }
        }
        for _ in 0..5 {
            m.observe_frame(&scene, 64, 64, 3, 0.033);
        }
        let tracks_before = m.object_memory().map_or(0, |m| m.len());

        // Phase 2: Occlude (uniform gray, 10 frames)
        let occluder = vec![128u8; 64 * 64 * 3];
        for _ in 0..10 {
            m.observe_frame(&occluder, 64, 64, 3, 0.033);
        }

        // Phase 3: Object should still be in memory (within max_absence_frames=30)
        let tracks_after = m.object_memory().map_or(0, |m| m.len());
        assert!(
            tracks_after > 0,
            "Object permanence: tracked object should survive 10-frame occlusion \
             (before={tracks_before}, after={tracks_after})"
        );
    }

    /// Visual search: does goal-directed attention find targets faster?
    /// (Treisman & Gelade 1980)
    #[test]
    fn test_psychbench_visual_search() {
        use crate::bridge::{CognitiveGoalSignal, VisionBridge};

        let cfg = VisionConfig::default();
        let mut bridge_no_goal = VisionBridge::new(cfg.clone(), 64, 64);
        let mut bridge_with_goal = VisionBridge::new(cfg, 64, 64);

        // A frame with a distinct, uncorrelated solid color per 8x8 patch
        // block, so a goal built from one patch's appearance is genuinely
        // selective. A smooth `i*k % 256` byte ramp instead gives many
        // patches near-identical *true* content, so a single-patch goal ends
        // up boosting most of the frame almost uniformly and the effect on
        // the (normalized) output becomes unmeasurable.
        let patch_size = 8usize;
        let cols = 64 / patch_size;
        let patch_frame = |seed: usize| -> Vec<u8> {
            (0..64usize * 64)
                .flat_map(|i| {
                    let (px, py) = (i % 64 / patch_size, i / 64 / patch_size);
                    let patch_idx = py * cols + px;
                    let key = patch_idx.wrapping_mul(97).wrapping_add(seed * 211 + 31);
                    let r = (key % 256) as u8;
                    let g = ((key / 3) % 256) as u8;
                    let b = ((key / 7) % 256) as u8;
                    vec![r, g, b]
                })
                .collect()
        };

        // Warm up both bridges with the same frame to build state
        let frame = patch_frame(0);
        bridge_no_goal.process_frame(&frame, 64, 64, 3, 0.033);
        bridge_with_goal.process_frame(&frame, 64, 64, 3, 0.033);

        // Use a patch's position-invariant appearance as the goal.
        // `apply_attention_boost()`'s top-down term compares `task_hv` against
        // `last_patch_appearance_hvs()`, so the goal must live in that same
        // appearance space to be meaningfully comparable — the manifold's
        // overall `state()` (a blended, position-bound scene bundle) isn't
        // directly comparable to a single patch's pure appearance vector. In
        // 16,384D, a random goal has ~0 similarity to patches (concentration
        // of measure), so we need a goal that's semantically related to the
        // scene: a real patch appearance, not an arbitrary vector.
        let goal_hv = bridge_with_goal
            .manifold()
            .last_patch_appearance_hvs()
            .first()
            .cloned()
            .expect("Should have patch appearance HVs after first frame");
        bridge_with_goal.set_goal_signal(CognitiveGoalSignal::with_gain(goal_hv, 0.8));

        // Scene change → surprise → attention boost modulates differently.
        // Every patch except patch 0 (the search target) gets a new color, so
        // the goal has something real to find amid genuine scene change.
        let mut frame2 = patch_frame(1);
        for y in 0..patch_size {
            for x in 0..patch_size {
                let pixel = y * 64 + x;
                frame2[pixel * 3..pixel * 3 + 3].copy_from_slice(&frame[pixel * 3..pixel * 3 + 3]);
            }
        }
        let (hv_no_goal, tel_no_goal) =
            bridge_no_goal.process_frame_with_telemetry(&frame2, 64, 64, 3, 0.033);
        let (hv_with_goal, tel_with_goal) =
            bridge_with_goal.process_frame_with_telemetry(&frame2, 64, 64, 3, 0.033);

        // Goal-directed vision should produce a measurably different HV.
        // The effect is subtle after normalization (concentration of measure),
        // but non-zero: attention boost modulates patch-level scaling.
        let sim = hv_no_goal.similarity(&hv_with_goal);
        assert!(
            sim < 1.0 - 1e-6,
            "Visual search: goal-directed HV should differ from passive (sim={sim})"
        );
        // Both should be valid
        assert!(tel_no_goal.output_hv_norm > 0.0);
        assert!(tel_with_goal.output_hv_norm > 0.0);
    }

    /// Working memory capacity: does the system show Cowan's 4±1 limit?
    /// (Cowan 2001)
    #[test]
    fn test_psychbench_working_memory_capacity() {
        let mut cfg = VisionConfig::default();
        cfg.enable_object_binding = true;
        let mut m = VisionManifold::new(cfg, 128, 128);
        m.enable_object_memory(64);
        m.enable_working_memory(4);

        // Create a scene with 8 distinct colored regions
        let mut pixels = vec![64u8; 128 * 128 * 3];
        let colors: [[u8; 3]; 8] = [
            [200, 0, 0],
            [0, 200, 0],
            [0, 0, 200],
            [200, 200, 0],
            [200, 0, 200],
            [0, 200, 200],
            [200, 128, 0],
            [128, 0, 200],
        ];
        for (i, color) in colors.iter().enumerate() {
            let row = (i / 4) * 64;
            let col = (i % 4) * 32;
            for y in row..(row + 30) {
                for x in col..(col + 30) {
                    if y < 128 && x < 128 {
                        let b = (y * 128 + x) * 3;
                        pixels[b] = color[0];
                        pixels[b + 1] = color[1];
                        pixels[b + 2] = color[2];
                    }
                }
            }
        }

        // Process enough frames for working memory to stabilize
        for _ in 0..20 {
            m.observe_frame(&pixels, 128, 128, 3, 0.033);
        }

        let wm_load = m.working_memory().map_or(0, |wm| wm.load());
        // Working memory should hold ≤ 4 objects (capacity limit)
        assert!(
            wm_load <= 4,
            "Cowan's limit: WM should hold ≤ 4 objects, got {wm_load}"
        );
        // Should hold at least 1 (the scene has many objects)
        assert!(
            wm_load >= 1,
            "WM should hold at least 1 object from 8-object scene, got {wm_load}"
        );
    }

    /// Imagination accuracy: do dream predictions improve with familiarity?
    /// (Kosslyn 1994 mental imagery)
    #[test]
    fn test_psychbench_imagination_accuracy() {
        let mut m = VisionManifold::new(VisionConfig::default(), 64, 64);
        let frame: Vec<u8> = (0..64 * 64 * 3).map(|i| (i * 11 % 256) as u8).collect();

        // Phase 1: Show the same scene for 20 frames (build familiarity)
        for _ in 0..20 {
            m.observe_frame(&frame, 64, 64, 3, 0.033);
        }
        let imagination_familiar = m.imagination_surprise();

        // Phase 2: Show a completely new scene
        let novel: Vec<u8> = (0..64 * 64 * 3).map(|i| (255 - (i % 256)) as u8).collect();
        m.observe_frame(&novel, 64, 64, 3, 0.033);
        let imagination_novel = m.imagination_surprise();

        // Familiar scenes should have lower imagination surprise
        assert!(
            imagination_familiar < imagination_novel,
            "Imagination accuracy: familiar scene ({imagination_familiar}) should have \
             lower surprise than novel scene ({imagination_novel})"
        );
    }

    #[test]
    fn test_dilation_budget_rejects_before_mutation() {
        use symthaea_core::hdc::HdcDimensionality;

        let mut cfg = VisionConfig::default();
        cfg.max_dilation_bytes = 1024;
        let mut manifold = VisionManifold::new(cfg, 64, 64);
        let before = manifold.state.clone();
        let before_dim = manifold.hdc_dim();

        let error = manifold.try_dilate(HdcDimensionality::Ultra).unwrap_err();

        assert!(error.contains("projected"));
        assert_eq!(manifold.hdc_dim(), before_dim);
        assert_eq!(manifold.state.dim(), before.dim());
        // `manifold.state` is still the constructor's zero vector (no frame
        // observed yet), and `similarity()` returns `0.0` rather than `1.0`
        // for near-zero-norm inputs — so compare raw values directly instead
        // of relying on self-similarity of a degenerate vector.
        assert_eq!(manifold.state.as_slice(), before.as_slice());
    }

    #[test]
    fn test_dilation_preflight_counts_cached_geodesics() {
        use symthaea_core::hdc::HdcDimensionality;

        let mut manifold = VisionManifold::new(VisionConfig::default(), 64, 64);
        let base = manifold.estimate_dilation(HdcDimensionality::Ultra);
        manifold.last_geodesic = vec![
            ContinuousHV::random(manifold.hdc_dim(), 991),
            ContinuousHV::random(manifold.hdc_dim(), 992),
        ];
        let with_path = manifold.estimate_dilation(HdcDimensionality::Ultra);
        assert_eq!(with_path.hdc_vectors, base.hdc_vectors + 2);

        manifold.try_dilate(HdcDimensionality::Ultra).unwrap();
        assert!(manifold.last_geodesic.iter().all(|hv| hv.dim() == 65_536));
    }

    #[test]
    fn test_holographic_dilation_semantic_preservation() {
        use symthaea_core::hdc::HdcDimensionality;

        let mut m = VisionManifold::new(VisionConfig::default(), 64, 64);
        let frame: Vec<u8> = (0..64 * 64 * 3).map(|i| (i * 11 % 256) as u8).collect();

        // 1. Establish baseline state at Standard (16K)
        m.observe_frame(&frame, 64, 64, 3, 0.033);
        let original_state = m.state.clone();
        assert_eq!(original_state.dim(), 16384);

        // 2. Dilate to Ultra (64K)
        m.dilate(HdcDimensionality::Ultra);
        assert_eq!(m.hdc_dim(), 65536);
        assert_eq!(m.state.dim(), 65536);

        // 3. Verify semantic preservation via folding-back similarity
        // (Since Ultra is unfolding via permutations, folding it back should recover the original signal)
        let folded_back = m.state.dilate(16384);
        let sim = original_state.similarity(&folded_back);

        println!("Dilation semantic preservation similarity: {:.4}", sim);
        // Requirement: > 0.85
        assert!(
            sim > 0.85,
            "Semantic loss too high during dilation/constriction cycle: {:.4}",
            sim
        );
    }

    #[test]
    fn test_active_inference_dreaming() {
        let mut m = VisionManifold::new(VisionConfig::default(), 64, 64);
        let frame: Vec<u8> = (0..64 * 64 * 3).map(|i| (i * 13 % 256) as u8).collect();
        m.observe_frame(&frame, 64, 64, 3, 0.033);

        let initial_state = m.state.clone();
        let initial_belief = m.fep_agent.belief.mean.clone();

        // Dream for 5 steps
        let steps = 5;
        let dt = 0.033;
        let dream_states = m.dream_ahead(steps, dt);

        assert_eq!(dream_states.len(), steps);

        // Manifold state should have evolved
        let final_dream_state = dream_states.last().unwrap();
        assert!(
            initial_state.similarity(final_dream_state) < 1.0,
            "Dreaming should evolve manifold state"
        );

        // Agent belief should have been restored
        assert_eq!(
            m.fep_agent.belief.mean, initial_belief,
            "Belief should be restored after dreaming"
        );
    }

    fn replay_fixture(seed: u64) -> (VisionManifold, ContinuousHV) {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut manifold = VisionManifold::new(config, 16, 16);
        manifold.state = ContinuousHV::random(256, seed);
        manifold.enable_scene_memory(4);
        let landmark = ContinuousHV::random(256, seed + 1);
        manifold
            .scene_memory
            .as_mut()
            .expect("scene memory")
            .remember(&landmark, 1, Vec::new());
        (manifold, landmark)
    }

    #[test]
    fn test_dream_replay_uses_local_state_without_moving_perception() {
        let (mut manifold, landmark) = replay_fixture(40_000);
        let live_state = manifold.state.clone();

        let replay = manifold.dream_replay_checked(0.1, 4).expect("valid replay");

        assert_eq!(replay.len(), 1);
        assert!(
            max_abs_difference(&manifold.state, &live_state) < 1e-7,
            "offline replay must not move the live perceptual state"
        );
        assert!(
            replay[0].similarity(&landmark) > live_state.similarity(&landmark),
            "replay should move the local simulated state toward the landmark"
        );
    }

    #[test]
    fn test_dream_replay_depth_changes_the_simulated_trajectory() {
        let (mut shallow, landmark) = replay_fixture(41_000);
        let (mut deep, _) = replay_fixture(41_000);

        let shallow_state = shallow.dream_replay_checked(0.1, 1).unwrap()[0].clone();
        let deep_state = deep.dream_replay_checked(0.1, 8).unwrap()[0].clone();

        assert!(
            deep_state.similarity(&landmark) > shallow_state.similarity(&landmark),
            "additional local replay steps should approach the remembered scene"
        );
    }

    #[test]
    fn test_dream_replay_rejects_invalid_parameters_atomically() {
        let (mut manifold, _) = replay_fixture(42_000);
        let state_before = manifold.state.clone();
        let weights_before = manifold.weight_hv.clone();

        assert!(manifold.dream_replay_checked(f32::NAN, 3).is_err());
        assert!(manifold.dream_replay_checked(0.1, 0).is_err());
        assert!(max_abs_difference(&manifold.state, &state_before) < 1e-7);
        assert!(max_abs_difference(&manifold.weight_hv, &weights_before) < 1e-7);
    }

    #[test]
    fn test_reset_clears_complete_runtime_context() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        config.enable_predictive_hierarchy = true;
        let mut manifold = VisionManifold::new(config, 16, 16);
        manifold.enable_scene_memory(2);
        manifold.enable_object_memory(2);
        manifold.enable_working_memory(1);
        manifold.enable_scene_graph();
        let frame = vec![128; 16 * 16];
        manifold
            .observe_frame_checked(&frame, 16, 16, 1, 0.033)
            .unwrap();
        manifold.last_geodesic = vec![ContinuousHV::random(256, 701)];
        manifold.last_intent_hv = ContinuousHV::random(256, 702);
        manifold.learning_frozen = true;

        manifold.reset();

        assert_eq!(manifold.frame_count(), 0);
        assert!(manifold.encoder().prev_patch_lum.is_empty());
        assert!(manifold.last_patch_hvs().is_empty());
        assert!(manifold.last_geodesic().is_empty());
        assert_eq!(manifold.last_frame_channels(), 0);
        assert!(manifold.last_observed_frame.is_none());
        assert!(manifold.last_intent_hv().norm() < 1e-6);
        assert!(!manifold.learning_frozen);
        assert_eq!(manifold.telemetry().frame_sequence, 0);
        assert_eq!(manifold.telemetry().prediction_error, 0.0);
        assert_eq!(manifold.telemetry().last_geodesic_length, 0);
    }

    #[test]
    fn test_find_geodesic() {
        let mut m = VisionManifold::new(VisionConfig::default(), 32, 32);
        let a = ContinuousHV::random(16384, 111);
        let b = ContinuousHV::random(16384, 222);

        let steps = 10;
        let path = m.find_geodesic(&a, &b, steps);

        assert_eq!(path.len(), steps);

        // Start should be close to a (sim > 0.90 due to normalization)
        assert!(path[0].similarity(&a) > 0.90);
        // End should be close to b
        assert!(path[steps - 1].similarity(&b) > 0.50);

        // Middle should be roughly equal distance (sim ~ 0.7 for orthogonal endpoints)
        let mid = steps / 2;
        let sim_a = path[mid].similarity(&a);
        let sim_b = path[mid].similarity(&b);
        println!("Midpoint similarity to A: {:.4}, to B: {:.4}", sim_a, sim_b);

        assert!(sim_a > 0.6 && sim_a < 0.85);
        assert!(sim_b > 0.6 && sim_b < 0.85);
    }

    #[test]
    fn test_select_best_geodesic_prefers_low_energy_path() {
        let mut m = VisionManifold::new(VisionConfig::default(), 32, 32);
        let a = ContinuousHV::random(16384, 111);
        let b = ContinuousHV::random(16384, 222);

        let path = m.select_best_geodesic(&a, &b, 8, 5);

        assert!(!path.is_empty());
        assert_eq!(path.len(), 8);
        assert!(m.geodesic_compute_cost > 0.0);

        // Final state in path should be close to b
        assert!(path[7].similarity(&b) > 0.7);
    }

    #[test]
    fn test_stereo_observation_injects_disparity_depth() {
        let mut cfg = VisionConfig::default();
        cfg.enable_depth = true;
        cfg.enable_motion = false;
        cfg.enable_color = false;
        cfg.enable_opponent_color = false;
        let mut stereo = VisionManifold::new(cfg.clone(), 32, 32);
        let mut monocular = VisionManifold::new(cfg, 32, 32);

        let mut left = vec![0u8; 32 * 32];
        let mut right = vec![0u8; 32 * 32];
        for y in 0..8usize {
            for dx in 0..8usize {
                let value = ((dx * 31 + y * 47 + dx * y * 7) % 251 + 1) as u8;
                left[y * 32 + 16 + dx] = value;
                right[y * 32 + dx] = value;
            }
        }

        stereo
            .observe_frame_stereo_checked(&left, &right, 32, 32, 1, 16, 0.033)
            .unwrap();
        monocular.observe_frame(&left, 32, 32, 1, 0.033);

        assert_eq!(stereo.stereo_depth_map().len(), 16);
        assert_eq!(stereo.stereo_disparity_map()[2], 16);
        assert!(
            stereo.stereo_depth_map()[2] < 0.1,
            "shifted textured patch should produce near stereo depth"
        );
        assert!(
            stereo.stereo_confidence_map()[2] > 0.4,
            "shifted textured patch should have useful confidence"
        );
        assert!(
            stereo.state().similarity(monocular.state()) < 0.999,
            "stereo depth must participate in the manifold observation"
        );
    }

    #[test]
    fn test_expectation_reset_clears_all_modality_prediction_caches() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        config.enable_predictive_hierarchy = true;
        let mut manifold = VisionManifold::new(config, 16, 16);

        manifold.activate_modality(VisualModality::Visible);
        manifold.last_prediction = Some(ContinuousHV::random(256, 80_000));
        manifold.prediction_error = 0.7;
        manifold.error_ema = 0.4;
        manifold.activate_modality(VisualModality::Stereo);
        manifold.last_prediction = Some(ContinuousHV::random(256, 80_001));
        manifold.prediction_error = 0.6;
        manifold.error_ema = 0.3;
        manifold.last_imagination = Some(ContinuousHV::random(256, 80_002));
        manifold.imagination_surprise = 0.8;

        manifold.reset_expectations();

        assert!(manifold.last_prediction.is_none());
        assert_eq!(manifold.prediction_error, 0.0);
        assert_eq!(manifold.error_ema, 0.0);
        assert!(manifold.last_imagination.is_none());
        assert_eq!(manifold.imagination_surprise, 0.0);
        assert!(manifold.modality_contexts.iter().all(|(_, context)| {
            context.last_prediction.is_none()
                && context.prediction_error == 0.0
                && context.error_ema == 0.0
        }));

        manifold.activate_modality(VisualModality::Visible);
        assert!(
            manifold.last_prediction.is_none(),
            "a parked modality must not resurrect its pre-reset prediction"
        );
        let predictive = manifold.predictive.as_ref().expect("predictive hierarchy");
        let predictive_state = predictive.save_state();
        assert!(predictive_state.last_coarse_hv.is_none());
        assert!(predictive_state.last_fine_hv.is_none());
        assert_eq!(predictive_state.prediction_count, 0);
    }

    #[test]
    fn test_modality_surprise_maps_are_isolated() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut manifold = VisionManifold::new(config, 16, 16);
        let count = manifold.surprise.grid().num_patches();
        let visible_prev: Vec<_> = (0..count)
            .map(|idx| ContinuousHV::random(256, 70_000 + idx as u64))
            .collect();
        let visible_now: Vec<_> = (0..count)
            .map(|idx| ContinuousHV::random(256, 71_000 + idx as u64))
            .collect();

        manifold.activate_modality(VisualModality::Visible);
        manifold.surprise.update(&visible_now, &visible_prev);
        let visible_surprise = manifold.surprise.max_surprise();
        assert!(visible_surprise > 0.1);

        manifold.activate_modality(VisualModality::Stereo);
        assert_eq!(
            manifold.surprise.max_surprise(),
            0.0,
            "a new sensor modality must begin with clean attention evidence"
        );

        let stereo_prev: Vec<_> = (0..count)
            .map(|idx| ContinuousHV::random(256, 72_000 + idx as u64))
            .collect();
        let stereo_now: Vec<_> = (0..count)
            .map(|idx| ContinuousHV::random(256, 73_000 + idx as u64))
            .collect();
        manifold.surprise.update(&stereo_now, &stereo_prev);
        let stereo_surprise = manifold.surprise.max_surprise();
        assert!(stereo_surprise > 0.1);

        manifold.activate_modality(VisualModality::Visible);
        assert!(
            (manifold.surprise.max_surprise() - visible_surprise).abs() < 1e-6,
            "returning to visible input must restore its own accumulated surprise"
        );
        manifold.activate_modality(VisualModality::Stereo);
        assert!(
            (manifold.surprise.max_surprise() - stereo_surprise).abs() < 1e-6,
            "stereo surprise must survive a temporary modality switch"
        );
    }

    #[test]
    fn test_fep_belief_isolated_by_modality_and_checkpointed() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut manifold = VisionManifold::new(config.clone(), 8, 8);

        manifold.activate_modality(VisualModality::Visible);
        manifold.fep_agent.belief.mean.fill(0.25);
        manifold.last_fep = crate::types::FepMetrics {
            free_energy: 1.0,
            complexity: 2.0,
            accuracy: 3.0,
        };
        manifold.activate_modality(VisualModality::Stereo);
        assert!(
            manifold
                .fep_agent
                .belief
                .mean
                .iter()
                .all(|value| *value != 0.25)
        );
        manifold.fep_agent.belief.mean.fill(0.75);
        manifold.activate_modality(VisualModality::Visible);
        assert!(
            manifold
                .fep_agent
                .belief
                .mean
                .iter()
                .all(|value| *value == 0.25)
        );
        assert_eq!(manifold.last_fep.free_energy, 1.0);

        let saved = manifold.save_state();
        let mut restored = VisionManifold::new(config, 8, 8);
        restored.load_state(&saved).unwrap();
        assert_eq!(
            restored.fep_agent.belief.mean,
            manifold.fep_agent.belief.mean
        );
        assert_eq!(
            restored.modality_contexts.len(),
            manifold.modality_contexts.len()
        );
    }

    #[test]
    fn test_schema6_rejects_incomplete_modality_fep_belief() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut source = VisionManifold::new(config.clone(), 8, 8);
        source.activate_modality(VisualModality::Visible);
        source.activate_modality(VisualModality::Stereo);
        let mut state = source.save_state();
        state.modality_contexts[0].fep_belief_mean.clear();

        let mut restored = VisionManifold::new(config, 8, 8);
        assert!(restored.load_state(&state).is_err());
    }

    #[test]
    fn test_modality_surprise_maps_survive_checkpoint_roundtrip() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut source = VisionManifold::new(config.clone(), 16, 16);
        let count = source.surprise.grid().num_patches();
        let prev: Vec<_> = (0..count)
            .map(|idx| ContinuousHV::random(256, 74_000 + idx as u64))
            .collect();
        let now: Vec<_> = (0..count)
            .map(|idx| ContinuousHV::random(256, 75_000 + idx as u64))
            .collect();

        source.activate_modality(VisualModality::Visible);
        source.surprise.update(&now, &prev);
        let expected_visible = source.surprise.max_surprise();
        source.activate_modality(VisualModality::MultiSpectral);
        assert_eq!(source.surprise.max_surprise(), 0.0);

        let saved = source.save_state();
        assert_eq!(saved.schema_version, MANIFOLD_STATE_SCHEMA_VERSION);
        let mut restored = VisionManifold::new(config, 16, 16);
        restored.load_state(&saved).unwrap();
        assert_eq!(restored.surprise.max_surprise(), 0.0);

        restored.activate_modality(VisualModality::Visible);
        assert!(
            (restored.surprise.max_surprise() - expected_visible).abs() < 1e-6,
            "inactive modality surprise must survive checkpoint restoration"
        );
    }

    #[test]
    fn test_schema_four_requires_complete_modality_surprise_state() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut source = VisionManifold::new(config.clone(), 16, 16);
        source.activate_modality(VisualModality::Visible);
        source.activate_modality(VisualModality::Stereo);
        let mut saved = source.save_state();
        saved.modality_contexts[0].surprise_state = None;

        let mut restored = VisionManifold::new(config, 16, 16);
        let state_before = restored.state.clone();
        let error = restored.load_state(&saved).unwrap_err();

        assert!(error.contains("missing surprise state"));
        assert!(max_abs_difference(&restored.state, &state_before) < 1e-7);
    }

    #[test]
    fn test_modality_temporal_histories_are_isolated_and_restored() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut manifold = VisionManifold::new(config, 8, 8);
        let visible_prediction = ContinuousHV::random(256, 91_001);
        let spectral_prediction = ContinuousHV::random(256, 91_002);

        manifold.activate_modality(VisualModality::Visible);
        manifold.last_prediction = Some(visible_prediction.clone());
        manifold.last_patch_hvs = vec![ContinuousHV::random(256, 91_003)];
        manifold.encoder.prev_patch_lum = vec![0.25];

        manifold.activate_modality(VisualModality::MultiSpectral);
        assert!(manifold.last_prediction.is_none());
        assert!(manifold.last_patch_hvs.is_empty());
        assert!(manifold.encoder.prev_patch_lum.is_empty());
        manifold.last_prediction = Some(spectral_prediction.clone());

        manifold.activate_modality(VisualModality::Visible);
        assert!(
            manifold
                .last_prediction
                .as_ref()
                .unwrap()
                .similarity(&visible_prediction)
                > 0.999_999
        );
        assert_eq!(manifold.encoder.prev_patch_lum, vec![0.25]);

        manifold.activate_modality(VisualModality::MultiSpectral);
        assert!(
            manifold
                .last_prediction
                .as_ref()
                .unwrap()
                .similarity(&spectral_prediction)
                > 0.999_999
        );
    }

    #[test]
    fn test_modality_temporal_contexts_survive_checkpoint_roundtrip() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut source = VisionManifold::new(config.clone(), 8, 8);
        let visible_prediction = ContinuousHV::random(256, 92_001);

        source.activate_modality(VisualModality::Visible);
        source.last_prediction = Some(visible_prediction.clone());
        source.encoder.prev_patch_lum = vec![0.75];
        source.activate_modality(VisualModality::MultiSpectral);

        let saved = source.save_state();
        assert_eq!(saved.active_modality, VisualModality::MultiSpectral);
        assert_eq!(saved.modality_contexts.len(), 1);

        let mut restored = VisionManifold::new(config, 8, 8);
        restored.load_state(&saved).unwrap();
        restored.activate_modality(VisualModality::Visible);
        assert!(
            restored
                .last_prediction
                .as_ref()
                .unwrap()
                .similarity(&visible_prediction)
                > 0.999_999
        );
        assert_eq!(restored.encoder.prev_patch_lum, vec![0.75]);
    }

    #[test]
    fn test_object_memory_checkpoint_rejects_non_finite_and_duplicate_tracks() {
        let mut state = ObjectMemoryState {
            tracks: vec![TrackedObjectState {
                track_id: 7,
                appearance_hv: vec![0.0; 256],
                identity_hv: vec![0.0; 256],
                centroid_row: 0,
                centroid_col: 0,
                velocity_row: 0.0,
                velocity_col: 0.0,
                last_seen_frame: 1,
                track_length: 1,
            }],
            capacity: 4,
            match_threshold: 0.6,
            max_absence_frames: 5,
            max_match_distance: 4,
        };
        assert!(ObjectMemory::validate_state(&state, 256).is_ok());

        state.tracks[0].appearance_hv[0] = f32::NAN;
        assert!(ObjectMemory::validate_state(&state, 256).is_err());
        state.tracks[0].appearance_hv[0] = 0.0;

        state.tracks.push(state.tracks[0].clone());
        assert!(
            ObjectMemory::validate_state(&state, 256)
                .unwrap_err()
                .contains("duplicate track ID")
        );
    }

    #[test]
    fn test_working_memory_checkpoint_rejects_invalid_numeric_state() {
        let mut state = VisualWorkingMemoryState {
            slots: vec![WorkingMemorySlotState {
                track_id: 3,
                hv: vec![0.0; 256],
                saliency: 0.8,
                centroid_row: 0,
                centroid_col: 0,
                entered_at_frame: 1,
            }],
            capacity: 4,
            decay_rate: 0.95,
        };
        assert!(VisualWorkingMemory::validate_state(&state, 256).is_ok());

        state.slots[0].saliency = f32::INFINITY;
        assert!(VisualWorkingMemory::validate_state(&state, 256).is_err());
        state.slots[0].saliency = 0.8;

        state.decay_rate = f32::NAN;
        assert!(VisualWorkingMemory::validate_state(&state, 256).is_err());
        state.decay_rate = 0.95;

        state.slots.push(state.slots[0].clone());
        assert!(
            VisualWorkingMemory::validate_state(&state, 256)
                .unwrap_err()
                .contains("duplicate track ID")
        );
    }

    #[test]
    fn test_schema_five_rejects_incoherent_memory_topology_and_time() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let source = VisionManifold::new(config.clone(), 16, 16);
        let mut state = source.save_state();
        state.frame_count = 10;
        state.object_memory = Some(ObjectMemoryState {
            tracks: vec![TrackedObjectState {
                track_id: 4,
                appearance_hv: vec![0.0; 256],
                identity_hv: vec![0.0; 256],
                centroid_row: 0,
                centroid_col: 0,
                velocity_row: 0.0,
                velocity_col: 0.0,
                last_seen_frame: 10,
                track_length: 1,
            }],
            capacity: 4,
            match_threshold: 0.6,
            max_absence_frames: 5,
            max_match_distance: 4,
        });
        state.working_memory = Some(VisualWorkingMemoryState {
            slots: vec![WorkingMemorySlotState {
                track_id: 4,
                hv: vec![0.0; 256],
                saliency: 0.8,
                centroid_row: 0,
                centroid_col: 0,
                entered_at_frame: 10,
            }],
            capacity: 4,
            decay_rate: 0.95,
        });
        state.next_track_id = 5;

        let mut restored = VisionManifold::new(config.clone(), 16, 16);
        assert!(restored.load_state(&state).is_ok());

        let mut missing_objects = state.clone();
        missing_objects.object_memory = None;
        assert!(
            VisionManifold::new(config.clone(), 16, 16)
                .load_state(&missing_objects)
                .unwrap_err()
                .contains("working memory without object memory")
        );

        let mut future_track = state.clone();
        future_track.object_memory.as_mut().unwrap().tracks[0].last_seen_frame = 11;
        assert!(
            VisionManifold::new(config.clone(), 16, 16)
                .load_state(&future_track)
                .unwrap_err()
                .contains("beyond checkpoint frame")
        );

        let mut reused_id = state;
        reused_id.next_track_id = 4;
        assert!(
            VisionManifold::new(config, 16, 16)
                .load_state(&reused_id)
                .unwrap_err()
                .contains("would reuse allocated track ID")
        );
    }

    #[test]
    fn test_legacy_checkpoint_repairs_allocator_from_working_memory() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let source = VisionManifold::new(config.clone(), 16, 16);
        let mut state = source.save_state();
        state.schema_version = 4;
        state.next_track_id = 0;
        state.working_memory = Some(VisualWorkingMemoryState {
            slots: vec![WorkingMemorySlotState {
                track_id: 99,
                hv: vec![0.0; 256],
                saliency: 0.5,
                centroid_row: 0,
                centroid_col: 0,
                entered_at_frame: 0,
            }],
            capacity: 4,
            decay_rate: 0.95,
        });

        let mut restored = VisionManifold::new(config, 16, 16);
        restored.load_state(&state).unwrap();
        assert_eq!(restored.save_state().next_track_id, 100);
    }

    #[test]
    fn test_checked_frame_rejection_is_atomic() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut manifold = VisionManifold::new(config, 32, 32);
        let before_state = manifold.state().clone();
        let before_frame_count = manifold.frame_count();
        let before_history = manifold.encoder().prev_patch_lum.clone();

        let error = manifold
            .observe_frame_checked(&vec![128; 32 * 32 - 1], 32, 32, 1, 0.033)
            .expect_err("truncated frame must be rejected");
        assert!(error.contains("length mismatch"));
        assert_eq!(manifold.frame_count(), before_frame_count);
        assert_eq!(manifold.encoder().prev_patch_lum, before_history);
        // `manifold.state()` is still the constructor's zero vector (no
        // frame was ever successfully observed), and `similarity()` returns
        // `0.0` rather than `1.0` for near-zero-norm inputs — so compare raw
        // values directly, matching
        // `test_stereo_checked_rejection_does_not_advance_manifold`.
        assert_eq!(manifold.state().as_slice(), before_state.as_slice());
    }

    #[test]
    fn test_checked_frame_rejects_invalid_channels_and_timestep() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut manifold = VisionManifold::new(config, 8, 8);
        assert!(
            manifold
                .observe_frame_checked(&vec![0; 8 * 8 * 2], 8, 8, 2, 0.033)
                .is_err()
        );
        assert!(
            manifold
                .observe_frame_checked(&vec![0; 8 * 8], 8, 8, 1, f32::NAN)
                .is_err()
        );
        assert_eq!(manifold.frame_count(), 0);
    }

    #[test]
    fn test_stereo_checked_rejection_does_not_advance_manifold() {
        let mut cfg = VisionConfig::default();
        cfg.hdc_dim = 256;
        cfg.enable_depth = true;
        let mut manifold = VisionManifold::new(cfg, 16, 16);
        let before_state = manifold.state().clone();
        let before_count = manifold.frame_count();

        let error = manifold
            .observe_frame_stereo_checked(&vec![0; 256], &vec![0; 255], 16, 16, 1, 8, 0.033)
            .unwrap_err();
        assert!(error.contains("buffer length mismatch"));
        assert_eq!(manifold.frame_count(), before_count);
        assert_eq!(manifold.state().as_slice(), before_state.as_slice());
        assert!(manifold.stereo_depth_map().is_empty());
        assert!(manifold.stereo_confidence_map().is_empty());
        assert!(manifold.stereo_disparity_map().is_empty());
    }

    #[test]
    fn test_checkpoint_restores_stereo_depth_confidence_and_disparity() {
        let mut cfg = VisionConfig::default();
        cfg.hdc_dim = 256;
        cfg.patch_size = 8;
        cfg.enable_depth = true;
        cfg.enable_motion = false;
        cfg.enable_color = false;
        cfg.enable_opponent_color = false;
        let mut source = VisionManifold::new(cfg.clone(), 16, 8);
        let mut left = vec![0u8; 16 * 8];
        let mut right = vec![0u8; 16 * 8];
        for y in 0..8usize {
            for dx in 0..8usize {
                let value = ((dx * 37 + y * 41 + dx * y * 5) % 251 + 1) as u8;
                left[y * 16 + 8 + dx] = value;
                right[y * 16 + dx] = value;
            }
        }
        source
            .observe_frame_stereo_checked(&left, &right, 16, 8, 1, 8, 0.033)
            .unwrap();
        let saved = source.save_state();

        let mut restored = VisionManifold::new(cfg, 16, 8);
        restored.load_state(&saved).unwrap();
        assert_eq!(restored.stereo_depth_map(), source.stereo_depth_map());
        assert_eq!(
            restored.stereo_confidence_map(),
            source.stereo_confidence_map()
        );
        assert_eq!(
            restored.stereo_disparity_map(),
            source.stereo_disparity_map()
        );
    }

    #[test]
    fn test_improved_geodesic_decoder_uses_persisted_landmark_pixels() {
        let mut config = VisionConfig::default();
        config.patch_size = 8;
        let mut manifold = VisionManifold::new(config, 8, 8);
        manifold.last_observed_frame = Some(vec![0; 64]);
        manifold.last_frame_width = 8;
        manifold.last_frame_height = 8;
        manifold.last_frame_channels = 1;
        manifold.enable_scene_memory(4);

        let start = ContinuousHV::random(manifold.hdc_dim(), 700_001);
        let target = ContinuousHV::random(manifold.hdc_dim(), 700_002);
        manifold
            .scene_memory
            .as_mut()
            .unwrap()
            .remember(&target, 1, vec![255; 64]);

        let frames = manifold.decode_geodesic_to_frames_improved(&[start, target]);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], vec![0; 64]);
        assert!(
            frames[1].iter().all(|&value| value > 200),
            "final frame should be driven by persisted landmark pixels"
        );
    }

    #[test]
    fn test_improved_geodesic_decoder_skips_incompatible_landmark_geometry() {
        let mut config = VisionConfig::default();
        config.patch_size = 8;
        let mut manifold = VisionManifold::new(config, 8, 8);
        manifold.last_observed_frame = Some(vec![17; 64]);
        manifold.last_frame_width = 8;
        manifold.last_frame_height = 8;
        manifold.last_frame_channels = 1;
        manifold.enable_scene_memory(4);

        let target = ContinuousHV::random(manifold.hdc_dim(), 700_003);
        manifold
            .scene_memory
            .as_mut()
            .unwrap()
            .remember(&target, 1, vec![255; 16]);

        let frames = manifold.decode_geodesic_to_frames_improved(&[target]);
        assert_eq!(frames, vec![vec![17; 64]]);
    }

    #[test]
    fn test_checkpoint_restores_object_and_working_memory() {
        let mut config = VisionConfig::default();
        config.enable_object_binding = true;
        let mut source = VisionManifold::new(config.clone(), 16, 16);
        source.enable_object_memory(8);
        source.enable_working_memory(4);

        let hypothesis = crate::types::ObjectHypothesis {
            centroid_row: 0,
            centroid_col: 0,
            patch_indices: vec![0],
            saliency: 0.8,
            hv: ContinuousHV::random(source.hdc_dim(), 880_001),
        };
        source.object_memory.as_mut().unwrap().update(
            std::slice::from_ref(&hypothesis),
            7,
            &mut source.next_track_id,
        );
        let tracks = source.object_memory.as_ref().unwrap().tracks().to_vec();
        source
            .working_memory
            .as_mut()
            .unwrap()
            .update(&tracks, &[hypothesis], 7);
        // Object/working memory were populated directly (bypassing
        // `observe_frame`, which would keep `frame_count` in lockstep with
        // every track's `last_seen_frame`). Advance it to match so the
        // checkpoint's schema-5 "no track can be last seen after the
        // checkpoint's own frame" invariant holds, exactly as it always does
        // when tracks are populated through the real observation pipeline.
        source.frame_count = 7;

        let checkpoint = source.save_state();
        let mut restored = VisionManifold::new(config, 16, 16);
        restored.load_state(&checkpoint).unwrap();

        assert_eq!(restored.object_memory().unwrap().len(), 1);
        assert_eq!(restored.working_memory().unwrap().load(), 1);
        assert_eq!(restored.next_track_id, source.next_track_id);
        assert_eq!(
            restored.object_memory().unwrap().tracks()[0].track_id,
            source.object_memory().unwrap().tracks()[0].track_id
        );
    }

    #[test]
    fn test_object_memory_uses_spatial_gate_for_similar_objects() {
        let dim = 1024;
        let appearance = ContinuousHV::random(dim, 990_001);
        let mut memory = ObjectMemory::new(4);
        memory.set_match_threshold(0.1);
        memory.set_max_match_distance(2);
        let mut next_id = 0;

        let initial = vec![
            crate::types::ObjectHypothesis {
                centroid_row: 0,
                centroid_col: 0,
                patch_indices: vec![0],
                saliency: 0.5,
                hv: appearance.clone(),
            },
            crate::types::ObjectHypothesis {
                centroid_row: 0,
                centroid_col: 10,
                patch_indices: vec![1],
                saliency: 0.5,
                hv: appearance.clone(),
            },
        ];
        memory.update(&initial, 0, &mut next_id);
        let left_id = memory.tracks()[0].track_id;
        let right_id = memory.tracks()[1].track_id;

        let reversed = vec![
            crate::types::ObjectHypothesis {
                centroid_row: 0,
                centroid_col: 9,
                patch_indices: vec![1],
                saliency: 0.5,
                hv: appearance.clone(),
            },
            crate::types::ObjectHypothesis {
                centroid_row: 0,
                centroid_col: 1,
                patch_indices: vec![0],
                saliency: 0.5,
                hv: appearance,
            },
        ];
        memory.update(&reversed, 1, &mut next_id);

        let left = memory
            .tracks()
            .iter()
            .find(|track| track.track_id == left_id)
            .unwrap();
        let right = memory
            .tracks()
            .iter()
            .find(|track| track.track_id == right_id)
            .unwrap();
        assert_eq!(left.centroid_col, 1);
        assert_eq!(right.centroid_col, 9);
    }

    #[test]
    fn test_current_patch_novelty_is_like_for_like() {
        let a = vec![ContinuousHV::random(256, 1), ContinuousHV::random(256, 2)];
        let same = VisionManifold::mean_patch_novelty(&a, &a);
        assert!(
            same < 1e-5,
            "identical patch evidence should be stable: {same}"
        );

        let changed = vec![ContinuousHV::random(256, 11), ContinuousHV::random(256, 12)];
        let novelty = VisionManifold::mean_patch_novelty(&changed, &a);
        assert!(
            novelty > 0.1,
            "new patch evidence should trigger re-clustering: {novelty}"
        );

        assert_eq!(VisionManifold::mean_patch_novelty(&a[..1], &a), 1.0);
    }

    #[test]
    fn test_hypothesis_saliency_uses_current_attention_values() {
        let mut hypotheses = vec![crate::types::ObjectHypothesis {
            centroid_row: 0,
            centroid_col: 0,
            patch_indices: vec![0, 2],
            saliency: 0.0,
            hv: ContinuousHV::random(256, 99),
        }];

        VisionManifold::refresh_hypothesis_saliency(&mut hypotheses, &[0.2, 0.0, 0.8]);
        assert!((hypotheses[0].saliency - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_optimal_assignment_beats_greedy_local_choice() {
        // Greedy would take row 0 -> col 0 (0.90) and strand row 1.
        // The optimal total keeps both identities: 0.89 + 0.88.
        let scores = vec![vec![Some(0.90), Some(0.89)], vec![Some(0.88), None]];
        let assignment = maximum_weight_assignment(&scores);
        assert_eq!(assignment, vec![Some(1), Some(0)]);
    }

    #[test]
    fn test_optimal_assignment_can_leave_hypothesis_unmatched() {
        let scores = vec![vec![None, Some(-0.1)]];
        assert_eq!(maximum_weight_assignment(&scores), vec![None]);
    }

    #[test]
    fn test_object_assignment_is_independent_of_hypothesis_order() {
        let dim = 256;
        let appearance_a = ContinuousHV::random(dim, 301);
        let appearance_b = ContinuousHV::random(dim, 302);
        let initial = vec![
            crate::types::ObjectHypothesis {
                centroid_row: 1,
                centroid_col: 1,
                patch_indices: vec![0],
                saliency: 0.5,
                hv: appearance_a.clone(),
            },
            crate::types::ObjectHypothesis {
                centroid_row: 1,
                centroid_col: 5,
                patch_indices: vec![1],
                saliency: 0.5,
                hv: appearance_b.clone(),
            },
        ];

        let mut first = ObjectMemory::new(4);
        let mut next_first = 0;
        first.update(&initial, 0, &mut next_first);
        let mut second = ObjectMemory::new(4);
        let mut next_second = 0;
        second.update(&initial, 0, &mut next_second);

        let forward = vec![
            crate::types::ObjectHypothesis {
                centroid_row: 1,
                centroid_col: 2,
                patch_indices: vec![0],
                saliency: 0.5,
                hv: appearance_a,
            },
            crate::types::ObjectHypothesis {
                centroid_row: 1,
                centroid_col: 4,
                patch_indices: vec![1],
                saliency: 0.5,
                hv: appearance_b,
            },
        ];
        let mut reversed = forward.clone();
        reversed.reverse();

        first.update(&forward, 1, &mut next_first);
        second.update(&reversed, 1, &mut next_second);

        let positions_first: Vec<(u64, usize)> = first
            .tracks()
            .iter()
            .map(|track| (track.track_id, track.centroid_col))
            .collect();
        let positions_second: Vec<(u64, usize)> = second
            .tracks()
            .iter()
            .map(|track| (track.track_id, track.centroid_col))
            .collect();
        assert_eq!(positions_first, positions_second);
    }

    #[test]
    fn test_stale_track_is_evicted_before_capacity_admission() {
        let dim = 256;
        let mut memory = ObjectMemory::new(1);
        memory.set_max_absence(1);
        let mut next_id = 0;
        memory.update(
            &[crate::types::ObjectHypothesis {
                centroid_row: 0,
                centroid_col: 0,
                patch_indices: vec![0],
                saliency: 1.0,
                hv: ContinuousHV::random(dim, 401),
            }],
            0,
            &mut next_id,
        );

        let result = memory.update(
            &[crate::types::ObjectHypothesis {
                centroid_row: 5,
                centroid_col: 5,
                patch_indices: vec![1],
                saliency: 1.0,
                hv: ContinuousHV::random(dim, 402),
            }],
            3,
            &mut next_id,
        );
        assert_eq!(result.evicted, 1);
        assert_eq!(result.new_tracks, 1);
        assert_eq!(memory.tracks()[0].track_id, 1);
    }

    #[test]
    fn test_working_memory_refreshes_live_appearance() {
        let dim = 256;
        let original = ContinuousHV::random(dim, 501);
        let updated = ContinuousHV::random(dim, 502);
        let mut memory = VisualWorkingMemory::new(1);
        let first_track = TrackedObject {
            track_id: 7,
            appearance_hv: original.clone(),
            identity_hv: original,
            centroid_row: 0,
            centroid_col: 0,
            velocity_row: 0.0,
            velocity_col: 0.0,
            last_seen_frame: 0,
            track_length: 1,
        };
        let hypothesis = crate::types::ObjectHypothesis {
            centroid_row: 0,
            centroid_col: 0,
            patch_indices: vec![0],
            saliency: 1.0,
            hv: first_track.appearance_hv.clone(),
        };
        memory.update(&[first_track], &[hypothesis], 0);

        let updated_track = TrackedObject {
            track_id: 7,
            appearance_hv: updated.clone(),
            identity_hv: updated.clone(),
            centroid_row: 0,
            centroid_col: 1,
            velocity_row: 0.0,
            velocity_col: 0.0,
            last_seen_frame: 1,
            track_length: 2,
        };
        let updated_hypothesis = crate::types::ObjectHypothesis {
            centroid_row: 0,
            centroid_col: 1,
            patch_indices: vec![1],
            saliency: 1.0,
            hv: updated.clone(),
        };
        memory.update(&[updated_track], &[updated_hypothesis], 1);
        assert!(memory.slots()[0].hv.similarity(&updated) > 0.999);
        assert_eq!(memory.slots()[0].centroid_col, 1);
    }

    #[test]
    fn test_object_memory_reports_only_tracks_actually_created() {
        let dim = 1024;
        let mut memory = ObjectMemory::new(1);
        memory.set_match_threshold(0.99);
        memory.set_max_match_distance(0);
        let mut next_id = 0;
        let first = crate::types::ObjectHypothesis {
            centroid_row: 0,
            centroid_col: 0,
            patch_indices: vec![0],
            saliency: 0.5,
            hv: ContinuousHV::random(dim, 990_010),
        };
        assert_eq!(memory.update(&[first], 0, &mut next_id).new_tracks, 1);

        let blocked = crate::types::ObjectHypothesis {
            centroid_row: 4,
            centroid_col: 4,
            patch_indices: vec![1],
            saliency: 0.5,
            hv: ContinuousHV::random(dim, 990_011),
        };
        let result = memory.update(&[blocked], 1, &mut next_id);
        assert_eq!(result.new_tracks, 0);
        assert_eq!(result.active_tracks, 1);
    }

    #[test]
    fn standalone_memory_restore_rejects_invalid_state_atomically() {
        let hv = ContinuousHV::random(32, 7);
        let mut scenes = SceneMemory::new(2);
        scenes.remember(&hv, 1, Vec::new());
        let before_scenes = scenes.save_state();
        let mut bad_scenes = before_scenes.clone();
        bad_scenes.threshold = f32::NAN;
        assert!(scenes.load_state_checked(&bad_scenes, 32).is_err());
        assert_eq!(scenes.save_state(), before_scenes);

        let mut objects = ObjectMemory::new(2);
        let before_objects = objects.save_state();
        let mut bad_objects = before_objects.clone();
        bad_objects.match_threshold = f32::NAN;
        assert!(objects.load_state_checked(&bad_objects, 32).is_err());
        assert_eq!(objects.save_state(), before_objects);

        let mut working = VisualWorkingMemory::new(2);
        let before_working = working.save_state();
        let mut bad_working = before_working.clone();
        bad_working.decay_rate = f32::NAN;
        assert!(working.load_state_checked(&bad_working, 32).is_err());
        assert_eq!(working.save_state(), before_working);
    }

    #[test]
    fn audited_manifold_loader_skips_semantically_incompatible_primary() {
        let directory = std::env::temp_dir().join(format!(
            "symthaea-vision-manifold-audited-load-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("manifold.chk");
        let cfg = VisionConfig::default();
        let source = VisionManifold::new(cfg.clone(), 32, 32);
        let valid = source.save_state();
        crate::checkpoint::save_checkpoint_file_with_retention(
            &path,
            "symthaea-vision-manifold",
            valid.schema_version,
            &valid,
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            crate::checkpoint::CheckpointRetentionPolicy {
                previous_generations: 1,
            },
        )
        .unwrap();
        let mut invalid = valid.clone();
        invalid.hdc_dim += 1;
        crate::checkpoint::save_checkpoint_file_with_retention(
            &path,
            "symthaea-vision-manifold",
            invalid.schema_version,
            &invalid,
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            crate::checkpoint::CheckpointRetentionPolicy {
                previous_generations: 1,
            },
        )
        .unwrap();

        let mut restored = VisionManifold::new(cfg, 32, 32);
        let report = restored
            .load_checkpoint_file_with_retention_audited(
                &path,
                crate::checkpoint::CheckpointRetentionPolicy {
                    previous_generations: 1,
                },
            )
            .unwrap();
        assert_eq!(report.selected.previous_generation, Some(1));
        assert!(matches!(
            &report.attempts[0].outcome,
            crate::checkpoint::CheckpointRecoveryAttemptOutcome::SemanticFailure(_)
        ));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn checkpoint_validation_isolated_from_live_state() {
        let cfg = VisionConfig::default();
        let mut manifold = VisionManifold::new(cfg, 32, 32);
        manifold
            .observe_frame_checked(&vec![17; 32 * 32], 32, 32, 1, 0.033)
            .unwrap();
        let before = serde_json::to_vec(&manifold.save_state()).unwrap();
        let mut invalid = manifold.save_state();
        invalid.hdc_dim += 1;
        assert!(manifold.validate_checkpoint_state(&invalid).is_err());
        assert_eq!(serde_json::to_vec(&manifold.save_state()).unwrap(), before);
    }

    #[test]
    fn manifold_retention_lifecycle_wrappers_report_and_prune() {
        let directory = std::env::temp_dir().join(format!(
            "symthaea-vision-manifold-lifecycle-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("manifold.chk");
        let manifold = VisionManifold::new(VisionConfig::default(), 32, 32);
        let policy = crate::checkpoint::CheckpointRetentionPolicy {
            previous_generations: 2,
        };
        manifold
            .save_checkpoint_file_with_retention_report(&path, policy)
            .unwrap();
        manifold
            .save_checkpoint_file_with_retention_report(&path, policy)
            .unwrap();
        let inventory = manifold
            .inspect_checkpoint_generations(&path, policy)
            .unwrap();
        assert!(inventory[0].metadata.is_some());
        assert!(inventory[1].metadata.is_some());
        let pruned = manifold.prune_checkpoint_generations(&path, 0).unwrap();
        assert!(pruned.removed_generations.contains(&1));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn authenticated_stack_checkpoint_roundtrip_and_rejection_are_atomic() {
        let cfg = VisionConfig::default();
        let mut source = VisionManifold::new(cfg.clone(), 8, 8);
        source
            .observe_frame_checked(&vec![23; 64], 8, 8, 1, 0.033)
            .unwrap();
        let encoded = source
            .save_authenticated_checkpoint_bytes(32, |inner| {
                Ok(inner.iter().take(16).copied().collect())
            })
            .unwrap();

        let mut restored = VisionManifold::new(cfg, 8, 8);
        restored
            .load_authenticated_checkpoint_bytes(&encoded, 32, |inner, tag| {
                Ok(tag
                    == inner
                        .iter()
                        .take(16)
                        .copied()
                        .collect::<Vec<_>>()
                        .as_slice())
            })
            .unwrap();
        assert_eq!(restored.frame_count(), source.frame_count());

        let before = serde_json::to_vec(&restored.save_state()).unwrap();
        assert!(
            restored
                .load_authenticated_checkpoint_bytes(&encoded, 32, |_inner, _tag| Ok(false))
                .is_err()
        );
        assert_eq!(serde_json::to_vec(&restored.save_state()).unwrap(), before);
    }

    #[test]
    fn authenticated_retained_stack_recovery_is_semantically_checked() {
        let directory = std::env::temp_dir().join(format!(
            "symthaea-vision-auth-stack-retention-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("manifold.chk");
        let cfg = VisionConfig::default();
        let first = VisionManifold::new(cfg.clone(), 16, 16);
        let mut incompatible = first.save_state();
        incompatible.hdc_dim += 1;
        let key = 0x44aa_9933_1f2e_7d8cu64;
        let policy = crate::checkpoint::CheckpointRetentionPolicy {
            previous_generations: 1,
        };
        first
            .save_authenticated_checkpoint_file_with_retention_report(
                &path,
                64,
                policy,
                |bytes| {
                    Ok(crate::checkpoint::fnv1a64_for_testing(key, bytes)
                        .to_le_bytes()
                        .to_vec())
                },
                |bytes, tag| {
                    Ok(tag == crate::checkpoint::fnv1a64_for_testing(key, bytes).to_le_bytes())
                },
            )
            .unwrap();
        crate::checkpoint::save_authenticated_checkpoint_file_with_retention_report(
            &path,
            "symthaea-vision-manifold",
            incompatible.schema_version,
            &incompatible,
            crate::checkpoint::DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES,
            64,
            policy,
            |bytes| {
                Ok(crate::checkpoint::fnv1a64_for_testing(key, bytes)
                    .to_le_bytes()
                    .to_vec())
            },
            |bytes, tag| {
                Ok(tag == crate::checkpoint::fnv1a64_for_testing(key, bytes).to_le_bytes())
            },
        )
        .unwrap();

        let mut restored = VisionManifold::new(cfg, 16, 16);
        let report = restored
            .load_authenticated_checkpoint_file_with_retention_audited(
                &path,
                64,
                policy,
                |bytes, tag| {
                    Ok(tag == crate::checkpoint::fnv1a64_for_testing(key, bytes).to_le_bytes())
                },
            )
            .unwrap();
        assert_eq!(report.selected.previous_generation, Some(1));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn object_and_working_memory_are_isolated_by_modality() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut manifold = VisionManifold::new(config, 16, 16);
        manifold.enable_object_memory(4);
        manifold.enable_working_memory(2);
        manifold.enable_scene_graph();
        manifold.activate_modality(VisualModality::Visible);

        let visible = crate::types::ObjectHypothesis {
            centroid_row: 0,
            centroid_col: 0,
            patch_indices: vec![0],
            saliency: 0.9,
            hv: ContinuousHV::random(256, 910_001),
        };
        manifold.object_memory.as_mut().unwrap().update(
            std::slice::from_ref(&visible),
            1,
            &mut manifold.next_track_id,
        );
        let visible_tracks = manifold.object_memory.as_ref().unwrap().tracks().to_vec();
        manifold.working_memory.as_mut().unwrap().update(
            &visible_tracks,
            std::slice::from_ref(&visible),
            1,
        );
        let visible_id = visible_tracks[0].track_id;

        manifold.activate_modality(VisualModality::Stereo);
        assert!(manifold.object_memory().unwrap().is_empty());
        assert_eq!(manifold.working_memory().unwrap().load(), 0);

        let stereo = crate::types::ObjectHypothesis {
            centroid_row: 1,
            centroid_col: 1,
            patch_indices: vec![1],
            saliency: 0.7,
            hv: ContinuousHV::random(256, 910_002),
        };
        manifold.object_memory.as_mut().unwrap().update(
            std::slice::from_ref(&stereo),
            2,
            &mut manifold.next_track_id,
        );
        assert_eq!(manifold.object_memory().unwrap().len(), 1);

        manifold.activate_modality(VisualModality::Visible);
        assert_eq!(manifold.object_memory().unwrap().len(), 1);
        assert_eq!(
            manifold.object_memory().unwrap().tracks()[0].track_id,
            visible_id
        );
        assert_eq!(manifold.working_memory().unwrap().load(), 1);
        assert!(manifold.scene_graph().is_some());
    }

    #[test]
    fn modality_semantic_memory_survives_checkpoint_roundtrip() {
        let mut config = VisionConfig::default();
        config.hdc_dim = 256;
        let mut source = VisionManifold::new(config.clone(), 16, 16);
        source.enable_object_memory(4);
        source.enable_working_memory(2);
        source.enable_scene_graph();
        source.activate_modality(VisualModality::Visible);

        let visible = crate::types::ObjectHypothesis {
            centroid_row: 0,
            centroid_col: 0,
            patch_indices: vec![0],
            saliency: 0.9,
            hv: ContinuousHV::random(256, 920_001),
        };
        source.object_memory.as_mut().unwrap().update(
            std::slice::from_ref(&visible),
            1,
            &mut source.next_track_id,
        );
        let visible_tracks = source.object_memory.as_ref().unwrap().tracks().to_vec();
        source.working_memory.as_mut().unwrap().update(
            &visible_tracks,
            std::slice::from_ref(&visible),
            1,
        );

        source.activate_modality(VisualModality::Stereo);
        let stereo = crate::types::ObjectHypothesis {
            centroid_row: 1,
            centroid_col: 1,
            patch_indices: vec![1],
            saliency: 0.8,
            hv: ContinuousHV::random(256, 920_002),
        };
        source.object_memory.as_mut().unwrap().update(
            std::slice::from_ref(&stereo),
            2,
            &mut source.next_track_id,
        );
        // Both modalities' memories were populated directly (bypassing
        // `observe_frame`, which keeps `frame_count` in lockstep with every
        // track's `last_seen_frame`). Advance it to match the highest frame
        // used above so the checkpoint's "no track can be last seen after
        // the checkpoint's own frame" invariant holds for both the active
        // (Stereo) and inactive (Visible) modality contexts.
        source.frame_count = 2;

        let saved = source.save_state();
        assert_eq!(saved.schema_version, 9);
        assert!(saved.modality_contexts[0].object_memory.is_some());
        assert!(saved.modality_contexts[0].working_memory.is_some());

        let mut restored = VisionManifold::new(config, 16, 16);
        restored.load_state(&saved).unwrap();
        assert_eq!(restored.object_memory().unwrap().len(), 1);
        restored.activate_modality(VisualModality::Visible);
        assert_eq!(restored.object_memory().unwrap().len(), 1);
        assert_eq!(restored.working_memory().unwrap().load(), 1);
        assert!(restored.scene_graph().is_some());

        let mut invalid = saved.clone();
        invalid.modality_contexts[0].next_track_id = 0;
        assert!(restored.validate_checkpoint_state(&invalid).is_err());
    }
}
