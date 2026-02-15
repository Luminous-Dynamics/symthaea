//! Temporal network backend abstraction.
//!
//! [`TemporalNetwork`] wraps either a CfC (Closed-form Continuous-time) network
//! or an HdcLtc bridge, allowing runtime selection of the temporal prediction
//! backend. All methods delegate to the active backend transparently.

use super::TemporalBackend;
use crate::dynamics::cfc::CfCNetwork;
use crate::hdc_ltc_bridge::HdcLtcBridge;
use anyhow::Result;
use ndarray::Array1;

/// Wrapper enum for temporal network backends.
///
/// This allows the CognitiveLoopService to use either CfC or HdcLtcUnified
/// as the temporal prediction backend, selected at runtime.
#[allow(dead_code)] // Some methods are provided for API completeness
pub(super) enum TemporalNetwork {
    /// CfC (Closed-form Continuous-time) network
    CfC(CfCNetwork),
    /// HdcLtcUnified network via bridge
    HdcLtc(HdcLtcBridge),
}

#[allow(dead_code)] // Methods provided for API completeness and future use
impl TemporalNetwork {
    /// Step the network forward
    pub fn step(&mut self, input: &Array1<f32>, dt: f32) -> Result<()> {
        match self {
            Self::CfC(cfc) => cfc.step(input, dt),
            Self::HdcLtc(bridge) => bridge.step(input, dt),
        }
    }

    /// Read the current state
    pub fn read_state(&self) -> Result<Array1<f32>> {
        match self {
            Self::CfC(cfc) => cfc.read_state(),
            Self::HdcLtc(bridge) => bridge.read_state(),
        }
    }

    /// Forward pass and return output
    pub fn forward(&mut self, input: &Array1<f32>, dt: f32) -> Array1<f32> {
        match self {
            Self::CfC(cfc) => cfc.forward(input, dt),
            Self::HdcLtc(bridge) => bridge.forward(input, dt),
        }
    }

    /// Train step (delegates to BPTT by default for CfC)
    pub fn train_step(
        &mut self,
        input: &Array1<f32>,
        target: &Array1<f32>,
        dt: f32,
        learning_rate: f32,
    ) -> Result<f32> {
        match self {
            Self::CfC(cfc) => cfc.train_step(input, target, dt, learning_rate),
            Self::HdcLtc(bridge) => bridge.train_step(input, target, dt, learning_rate),
        }
    }

    /// Train step using BPTT (analytical gradients).
    /// For HdcLtc this falls through to the default train_step.
    pub fn train_step_bptt(
        &mut self,
        input: &Array1<f32>,
        target: &Array1<f32>,
        dt: f32,
        learning_rate: f32,
    ) -> Result<f32> {
        match self {
            Self::CfC(cfc) => {
                cfc.train_step_bptt(&[input.clone()], &[target.clone()], &[dt], learning_rate)
            }
            Self::HdcLtc(bridge) => bridge.train_step(input, target, dt, learning_rate),
        }
    }

    /// Train step using SPSA (perturbation-based gradients).
    /// For HdcLtc this falls through to the default train_step.
    pub fn train_step_spsa(
        &mut self,
        input: &Array1<f32>,
        target: &Array1<f32>,
        dt: f32,
        learning_rate: f32,
    ) -> Result<f32> {
        match self {
            Self::CfC(cfc) => cfc.train_step_spsa(input, target, dt, learning_rate),
            Self::HdcLtc(bridge) => bridge.train_step(input, target, dt, learning_rate),
        }
    }

    /// Predict forward at a specific time horizon
    pub fn predict_forward(&mut self, input: &Array1<f32>, horizon: f32) -> Result<Array1<f32>> {
        match self {
            Self::CfC(cfc) => cfc.predict_forward(input, horizon),
            Self::HdcLtc(bridge) => bridge.predict_forward(input, horizon),
        }
    }

    /// Inject state
    pub fn inject(&mut self, state: &Array1<f32>) -> Result<()> {
        match self {
            Self::CfC(cfc) => cfc.inject(state),
            Self::HdcLtc(bridge) => bridge.inject(state),
        }
    }

    /// Reset the network
    pub fn reset(&mut self) {
        match self {
            Self::CfC(cfc) => cfc.reset(),
            Self::HdcLtc(bridge) => bridge.reset(),
        }
    }

    /// Get state diversity metric
    pub fn state_diversity(&self) -> f32 {
        match self {
            Self::CfC(cfc) => cfc.state_diversity(),
            Self::HdcLtc(bridge) => bridge.state_diversity(),
        }
    }

    /// Get all tau values for coherence tracking
    pub fn all_tau(&self) -> Vec<&Array1<f32>> {
        match self {
            Self::CfC(cfc) => cfc.all_tau(),
            Self::HdcLtc(_) => vec![], // HdcLtc returns owned, handled separately
        }
    }

    /// Get all tau values (owned version for HdcLtc compatibility)
    pub fn all_tau_owned(&self) -> Vec<Array1<f32>> {
        match self {
            Self::CfC(cfc) => cfc.all_tau().into_iter().cloned().collect(),
            Self::HdcLtc(bridge) => bridge.all_tau(),
        }
    }

    /// Get flattened tau values
    pub fn flattened_tau(&self) -> Vec<f32> {
        match self {
            Self::CfC(cfc) => cfc.flattened_tau(),
            Self::HdcLtc(bridge) => bridge.flattened_tau(),
        }
    }

    /// Adaptively resize HDC dimension based on prediction error (HdcLtc only)
    pub fn maybe_resize(&mut self, current_error: f32) {
        if let Self::HdcLtc(bridge) = self {
            bridge.maybe_resize(current_error);
        }
    }

    /// Check if using HdcLtc backend
    pub fn is_hdc_ltc(&self) -> bool {
        matches!(self, Self::HdcLtc(_))
    }

    /// Get backend type
    pub fn backend_type(&self) -> TemporalBackend {
        match self {
            Self::CfC(_) => TemporalBackend::CfC,
            Self::HdcLtc(_) => TemporalBackend::HdcLtcUnified,
        }
    }

    /// Project input directly to HDC space, bypassing CfC temporal dynamics.
    ///
    /// Returns `None` for CfC backend (no HDC projection available).
    /// Returns `Some(Vec<f32>)` for HdcLtc backend with the raw HDC vector.
    pub fn project_to_hdc_vec(&self, input: &[f32]) -> Option<Vec<f32>> {
        match self {
            Self::CfC(_) => None,
            Self::HdcLtc(bridge) => Some(bridge.project_to_hdc_vec(input)),
        }
    }

    /// Get HDC dimension (returns None for CfC backend)
    pub fn hdc_dim(&self) -> Option<usize> {
        match self {
            Self::CfC(_) => None,
            Self::HdcLtc(bridge) => Some(bridge.hdc_dim()),
        }
    }
}
