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
//! - [`HolographicLiquidCell`] is a theorem-bearing diagonal research cell whose
//!   temporal update commutes with `UnitaryRole` binding.
//! - [`HlsParameters`] exposes the complete theorem-compatible trainable surface
//!   as one validated atomic snapshot.
//! - [`HlsEligibilityTrace`] and [`step_with_eligibility`] expose exact online
//!   forward sensitivities for the diagonal recurrence when the global norm
//!   limiter does not activate.
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
pub mod state_tracking_benchmark;
pub mod state_tracking_codec;
pub mod state_tracking_eval;
pub mod state_tracking_readout;

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
pub use state_tracking_benchmark::{
    EntityId, LocationId, ObjectId, StateTrackingBenchmark, StateTrackingBenchmarkConfig,
    StateTrackingBenchmarkError, TrackingAnswer, TrackingEvent, TrackingEventKind, TrackingQuery,
    TrackingQueryKind, TrackingScore,
};
pub use state_tracking_codec::{StateTrackingCodec, TrackingCodecError};
pub use state_tracking_eval::{
    FrozenTrackingEvalConfig, FrozenTrackingEvalError, FrozenTrackingEvalResult,
    FrozenTrackingReservoir, evaluate_frozen_reservoir,
};
pub use state_tracking_readout::{TrackingPrototypeReadout, TrackingReadoutError};
