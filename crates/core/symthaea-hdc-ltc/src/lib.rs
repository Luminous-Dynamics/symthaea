// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # symthaea-hdc-ltc
//!
//! Hyperdimensional recurrent state with solver-free liquid-time evolution.
//!
//! The crate deliberately separates the research surfaces needed to test HLS
//! claims independently:
//!
//! - [`ContinuousHV`] carries arbitrary continuous distributed state.
//! - [`UnitaryRole`] carries reversible real HDC roles with components in
//!   `{ -1, +1 }`, so role binding is an isometry.
//! - [`TemporalAxis`] and [`TemporalPhasor`] expose a continuous unitary time
//!   algebra in the Fourier/phasor domain; [`TemporalInterval`] analytically
//!   encodes continuous validity regions without claiming historical recall yet.
//! - [`ValidityIntervalMemory`] uses that axis as discrete causal/version time,
//!   storing closed key/value validity spans in O(D) per span regardless of the
//!   number of checkpoints covered.
//! - [`StateTrackingValidityArchive`] bridges the standalone validity memory to
//!   the model-agnostic benchmark and evaluates historical two-hop recall without
//!   changing HLS recurrence.
//! - [`run_validity_capacity_sweep`] measures the finite-dimensional failure
//!   surface across dimension, key count, candidate count, horizon, and span
//!   length under fixed predeclared replicate seeds.
//! - [`ValidityCapacityNullModel`] freezes the pre-result idealized crosstalk
//!   prediction that the score-noise variance approaches `rho / 2` for
//!   `rho = key_count * horizon / dimension`, while making its independence
//!   assumptions explicit and testable.
//! - [`ValidityCapacityAccuracyNullModel`] adds a stricter pre-result prediction:
//!   winner-take-all cleanup accuracy under an explicitly idealized independent
//!   Gaussian score model, with deterministic quadrature convergence diagnostics.
//! - [`run_validity_capacity_controls`] separates semantic-change density from
//!   archive write segmentation while holding the null-model coordinates fixed.
//! - [`measure_validity_capacity_score_moments`] independently reconstructs the
//!   frozen synthetic capacity cases and measures target/distractor score moments
//!   plus signed true margins for direct comparison with the null model.
//! - [`measure_validity_capacity_shadow_distractors`] reconstructs the same frozen
//!   archive with a matched never-written shadow vocabulary selected on the same
//!   per-query index schedule as the real distractor probe.
//! - [`HolographicLiquidCell`] is a theorem-bearing diagonal research cell whose
//!   temporal update commutes with `UnitaryRole` binding.
//! - [`HlsParameters`] exposes the complete theorem-compatible trainable surface
//!   as one validated atomic snapshot.
//! - [`HlsEligibilityTrace`] and [`step_with_eligibility`] expose exact online
//!   forward sensitivities for the diagonal recurrence when the global norm
//!   limiter does not activate.
//! - [`associative_query`] is a fixed, parameter-free HDC decoder that supplies
//!   an exact current-state learning signal for those eligibility traces.
//! - [`train_exact_episode`] accumulates those exact query gradients while
//!   parameters remain frozen for a complete world, then applies one bounded
//!   episode-end recurrent update.
//! - [`train_current_only_episode`] is the fail-closed experiment gate for the
//!   first learned result: historical queries are rejected before the recurrent
//!   state or parameters can mutate.
//! - [`run_exact_learning_ablation`] executes the fixed-seed paired current-state
//!   learned-vs-frozen experiment without best-seed selection.
//! - [`InvariantContextMixer`] supplies O(KD) cross-dimensional magnitude
//!   context without violating the full bipolar role symmetry.
//! - [`ContextualHolographicLiquidCell`] composes invariant context with the
//!   theorem-bearing HLS transition while leaving diagonal HLS intact as an
//!   ablation baseline.
//! - [`StateTrackingBenchmark`] defines a model-agnostic irregular-time,
//!   compositional, historical state-tracking task.
//! - [`StateTrackingCodec`], [`TrackingPrototypeReadout`], and
//!   [`evaluate_frozen_reservoir`] define a shared leakage-controlled diagnostic
//!   protocol for the first frozen-recurrence ablations.
//!
//! The legacy [`HdcLtcUnifiedNeuron`] remains available so the algebraic research
//! path can be qualified without silently changing production behavior.

pub mod config;
pub mod contextual_holographic_liquid;
pub mod continuous_hv;
pub mod hls_online_trace;
pub mod holographic_liquid;
pub mod invariant_context;
pub mod network;
pub mod neuron;
pub mod state_tracking_associative;
pub mod state_tracking_benchmark;
pub mod state_tracking_codec;
pub mod state_tracking_current_only;
pub mod state_tracking_eval;
pub mod state_tracking_exact_training;
pub mod state_tracking_learning_ablation;
pub mod state_tracking_readout;
pub mod state_tracking_validity_archive;
pub mod temporal_phasor;
pub mod validity_capacity;
pub mod validity_capacity_accuracy_theory;
pub mod validity_capacity_controls;
pub mod validity_capacity_score_moments;
pub mod validity_capacity_shadow_probe;
pub mod validity_capacity_theory;
pub mod validity_interval_memory;

pub use config::{Activation, NetworkConfig, NeuronConfig};
pub use contextual_holographic_liquid::{ContextualHlsError, ContextualHolographicLiquidCell};
pub use continuous_hv::{ContinuousHV, HDC_DIMENSION, UnitaryRole};
pub use hls_online_trace::{HlsEligibilityTrace, HlsTraceError, step_with_eligibility};
pub use holographic_liquid::{
    HlsActivation, HlsConfig, HlsError, HlsParameters, HolographicLiquidCell,
};
pub use invariant_context::{ContextMixerError, InvariantContextMixer};
pub use network::{HdcLtcUnifiedNetwork, StepTimingConfig};
pub use neuron::HdcLtcUnifiedNeuron;
pub use state_tracking_associative::{
    AssociativeQueryResult, AssociativeReadoutError, associative_query,
};
pub use state_tracking_benchmark::{
    EntityId, LocationId, ObjectId, StateTrackingBenchmark, StateTrackingBenchmarkConfig,
    StateTrackingBenchmarkError, TrackingAnswer, TrackingEvent, TrackingEventKind, TrackingQuery,
    TrackingQueryKind, TrackingScore,
};
pub use state_tracking_codec::{StateTrackingCodec, TrackingCodecError};
pub use state_tracking_current_only::{
    CurrentOnlyTrainingError, evaluate_current_only_episode, train_current_only_episode,
    validate_current_only_benchmark,
};
pub use state_tracking_eval::{
    FrozenTrackingEvalConfig, FrozenTrackingEvalError, FrozenTrackingEvalResult,
    FrozenTrackingReservoir, evaluate_frozen_reservoir,
};
pub use state_tracking_exact_training::{
    AssociativeEpisodeMetrics, ExactEpisodeTrainingConfig, ExactEpisodeTrainingError,
    ExactEpisodeTrainingReport, evaluate_associative_episode, train_exact_episode,
};
pub use state_tracking_learning_ablation::{
    ExactLearningAblationError, ExactLearningAblationPlan, ExactLearningAblationResult,
    HeldOutWorldComparison, PairedEffectSummary, TrainingWorldResult,
    run_exact_learning_ablation,
};
pub use state_tracking_readout::{TrackingPrototypeReadout, TrackingReadoutError};
pub use state_tracking_validity_archive::{
    HistoricalArchiveAnswer, HistoricalArchiveEvaluation, StateTrackingValidityArchive,
    StateTrackingValidityArchiveError,
};
pub use temporal_phasor::{
    TemporalAlgebraError, TemporalAxis, TemporalInterval, TemporalPhasor,
};
pub use validity_capacity::{
    ValidityCapacityAxis, ValidityCapacityCase, ValidityCapacityError, ValidityCapacityObservation,
    ValidityCapacityPlan, ValidityCapacitySweepResult, run_validity_capacity_sweep,
};
pub use validity_capacity_accuracy_theory::ValidityCapacityAccuracyNullModel;
pub use validity_capacity_controls::{
    ValidityCapacityControlAxis, ValidityCapacityControlCase, ValidityCapacityControlError,
    ValidityCapacityControlObservation, ValidityCapacityControlPlan, ValidityCapacityControlResult,
    run_validity_capacity_controls,
};
pub use validity_capacity_score_moments::{
    ScoreMomentSummary, ValidityCapacityScoreMomentError, ValidityCapacityScoreMomentObservation,
    ValidityCapacityScoreMomentResult, measure_validity_capacity_score_moments,
};
pub use validity_capacity_shadow_probe::{
    ValidityCapacityShadowError, ValidityCapacityShadowObservation, ValidityCapacityShadowResult,
    measure_validity_capacity_shadow_distractors,
};
pub use validity_capacity_theory::{
    ValidityCapacityNullModel, ValidityCapacityTheoryError,
};
pub use validity_interval_memory::{
    ValidityCleanupResult, ValidityIntervalMemory, ValidityMemoryError,
};
