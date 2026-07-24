// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Temporal network backend abstraction.
//!
//! [`TemporalNetwork`] wraps either a CfC (Closed-form Continuous-time) network,
//! an HdcLtc bridge, or a HierarchicalCfC multi-scale hierarchy, allowing runtime
//! selection of the temporal prediction backend. All methods delegate to the active
//! backend transparently.

use super::TemporalBackend;
use crate::dynamics::cfc::CfCNetwork;
use crate::dynamics::hierarchical_cfc::HierarchicalCfC;
use crate::hdc_ltc_bridge::HdcLtcBridge;
use anyhow::Result;
use ndarray::Array1;

/// Wrapper enum for temporal network backends.
///
/// This allows the CognitiveLoopService to use CfC, HdcLtcUnified, or
/// HierarchicalCfC as the temporal prediction backend, selected at runtime.
#[allow(dead_code)] // RESERVED(future): temporal network routing
pub(super) enum TemporalNetwork {
    /// CfC (Closed-form Continuous-time) network
    CfC(CfCNetwork),
    /// HdcLtcUnified network via bridge
    HdcLtc(HdcLtcBridge),
    /// Hierarchical CfC with multi-scale temporal processing (PP-2: Butlin indicator)
    HierarchicalCfC(HierarchicalCfC),
}

/// Backend-specific evolution-state backup (see
/// [`TemporalNetwork::save_evolution_state`]).
pub(super) enum TemporalStateBackup {
    /// CfC: the small hidden-state vector, restorable via true inject
    CfC(Array1<f32>),
    /// HdcLtc: full neuron-state snapshot
    HdcLtc(symthaea_core::hdc::hdc_ltc_unified::NetworkStateSnapshot),
}

impl TemporalNetwork {
    /// Step the network forward
    pub fn step(&mut self, input: &Array1<f32>, dt: f32) -> Result<()> {
        match self {
            Self::CfC(cfc) => cfc.step(input, dt),
            Self::HdcLtc(bridge) => bridge.step(input, dt),
            Self::HierarchicalCfC(hcfc) => {
                let _ = hcfc.forward_hierarchical(input, dt);
                Ok(())
            }
        }
    }

    /// Read the current state
    pub fn read_state(&self) -> Result<Array1<f32>> {
        match self {
            Self::CfC(cfc) => cfc.read_state(),
            Self::HdcLtc(bridge) => bridge.read_state(),
            Self::HierarchicalCfC(hcfc) => {
                // Return the last layer's state (slowest temporal context)
                let states = hcfc.all_states();
                if let Some(last_layer) = states.last() {
                    if let Some(last_cell) = last_layer.last() {
                        return Ok(last_cell.clone());
                    }
                }
                Ok(Array1::zeros(hcfc.config().output_dim))
            }
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
            Self::HierarchicalCfC(hcfc) => hcfc.train_step(input, target, dt, learning_rate),
        }
    }

    /// Train step using BPTT (analytical gradients).
    /// For HdcLtc this falls through to the default train_step.
    /// For HierarchicalCfC this uses single-target multi-scale training.
    /// Train with the forward pass starting from a historical evolution-state
    /// backup instead of the live state (2026-07-17, sequence-prediction fix).
    ///
    /// For the per-cycle (enc_{t−1} → enc_t) training pair, the temporally
    /// correct starting state is the one from the END of cycle t−2 — the live
    /// state has already been stepped with enc_t by the planning phase.
    /// Only the HdcLtc backend supports historical starts; other backends
    /// fall back to their existing (state-agnostic or locally-seeded)
    /// trainers, which at least do not corrupt live state.
    pub fn train_step_from(
        &mut self,
        start: Option<&TemporalStateBackup>,
        input: &Array1<f32>,
        target: &Array1<f32>,
        dt: f32,
        learning_rate: f32,
    ) -> Result<f32> {
        match (self, start) {
            (Self::HdcLtc(bridge), Some(TemporalStateBackup::HdcLtc(snap))) => {
                bridge.train_step_from(snap, input, target, dt, learning_rate)
            }
            (me, _) => me.train_step_bptt(input, target, dt, learning_rate),
        }
    }

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
            Self::HierarchicalCfC(hcfc) => hcfc.train_step(input, target, dt, learning_rate),
        }
    }

    /// Train step using SPSA (perturbation-based gradients).
    /// For HdcLtc and HierarchicalCfC this falls through to the default train_step.
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
            Self::HierarchicalCfC(hcfc) => hcfc.train_step(input, target, dt, learning_rate),
        }
    }

    /// Predict forward at a specific time horizon.
    ///
    /// For HierarchicalCfC, this runs the full hierarchy at the given horizon as dt,
    /// returning the combined multi-scale output.
    pub fn predict_forward(&mut self, input: &Array1<f32>, horizon: f32) -> Result<Array1<f32>> {
        match self {
            Self::CfC(cfc) => cfc.predict_forward(input, horizon),
            Self::HdcLtc(bridge) => bridge.predict_forward(input, horizon),
            Self::HierarchicalCfC(hcfc) => Ok(hcfc.forward_hierarchical(input, horizon).combined),
        }
    }

    /// Whether `predict_forward` on this backend leaves live state untouched.
    ///
    /// True for HdcLtc (its predict_forward snapshots/restores internally).
    /// False for classic CfC (predict_forward advances state; callers must
    /// save/restore via read_state/inject, which is a true restore there) and
    /// for HierarchicalCfC (predict_forward advances the hierarchy AND its
    /// inject is a reset — a known unfixed footgun, tracked in
    /// docs/PHI_SIGNAL_TRACE_2026-07-15.md follow-ups).
    pub fn prediction_is_pure(&self) -> bool {
        matches!(self, Self::HdcLtc(_))
    }

    /// Inject state
    ///
    /// WARNING: only the CfC backend truly restores injected state. For HdcLtc
    /// and HierarchicalCfC this is a RESET (internal state cannot be
    /// reconstructed from the small projected vector) — never use it as the
    /// "restore" half of a save/restore pattern on those backends.
    pub fn inject(&mut self, state: &Array1<f32>) -> Result<()> {
        match self {
            Self::CfC(cfc) => cfc.inject(state),
            Self::HdcLtc(bridge) => bridge.inject(state),
            Self::HierarchicalCfC(hcfc) => {
                hcfc.reset();
                Ok(())
            }
        }
    }

    /// Best-effort snapshot of evolution state ahead of a deliberately
    /// destructive operation (e.g. consolidation replay, which resets state
    /// for clean replays). Returns None when the backend cannot snapshot
    /// (HierarchicalCfC).
    pub fn save_evolution_state(&self) -> Option<TemporalStateBackup> {
        match self {
            Self::CfC(cfc) => cfc.read_state().ok().map(TemporalStateBackup::CfC),
            Self::HdcLtc(bridge) => Some(TemporalStateBackup::HdcLtc(
                bridge.snapshot_evolution_state(),
            )),
            Self::HierarchicalCfC(_) => None,
        }
    }

    /// Restore evolution state captured by [`save_evolution_state`].
    pub fn restore_evolution_state(&mut self, backup: &TemporalStateBackup) {
        match (self, backup) {
            (Self::CfC(cfc), TemporalStateBackup::CfC(state)) => {
                let _ = cfc.inject(state);
            }
            (Self::HdcLtc(bridge), TemporalStateBackup::HdcLtc(snap)) => {
                bridge.restore_evolution_state(snap);
            }
            _ => {}
        }
    }

    /// Get state diversity metric
    pub fn state_diversity(&self) -> f32 {
        match self {
            Self::CfC(cfc) => cfc.state_diversity(),
            Self::HdcLtc(bridge) => bridge.state_diversity(),
            Self::HierarchicalCfC(hcfc) => {
                // Compute diversity across all layers as variance of layer norms
                let states = hcfc.all_states();
                let norms: Vec<f32> = states
                    .iter()
                    .filter_map(|layer| {
                        layer
                            .last()
                            .map(|s| s.iter().map(|v| v * v).sum::<f32>().sqrt())
                    })
                    .collect();
                if norms.len() < 2 {
                    return 0.5;
                }
                let mean = norms.iter().sum::<f32>() / norms.len() as f32;
                let var =
                    norms.iter().map(|n| (n - mean).powi(2)).sum::<f32>() / norms.len() as f32;
                // Normalize: high variance across layers = high diversity
                (var.sqrt() / (mean.abs() + 1e-6)).min(1.0)
            }
        }
    }

    /// Get all tau values (owned version for HdcLtc compatibility).
    /// Use [`with_tau_refs`] or [`flattened_tau`] instead when owned copies aren't needed.
    pub fn all_tau_owned(&self) -> Vec<Array1<f32>> {
        match self {
            Self::CfC(cfc) => cfc.all_tau().into_iter().cloned().collect(),
            Self::HdcLtc(bridge) => bridge.all_tau(),
            Self::HierarchicalCfC(hcfc) => {
                // Return one tau per hierarchical level
                hcfc.time_constants()
                    .iter()
                    .map(|&tau| Array1::from_vec(vec![tau]))
                    .collect()
            }
        }
    }

    /// Call `f` with tau references, avoiding owned clones on the CfC hot path.
    pub fn with_tau_refs<R>(&self, f: impl FnOnce(&[&Array1<f32>]) -> R) -> R {
        match self {
            Self::CfC(cfc) => {
                let refs = cfc.all_tau();
                f(&refs)
            }
            Self::HdcLtc(bridge) => {
                let owned = bridge.all_tau();
                let refs: Vec<&Array1<f32>> = owned.iter().collect();
                f(&refs)
            }
            Self::HierarchicalCfC(hcfc) => {
                let owned: Vec<Array1<f32>> = hcfc
                    .time_constants()
                    .iter()
                    .map(|&tau| Array1::from_vec(vec![tau]))
                    .collect();
                let refs: Vec<&Array1<f32>> = owned.iter().collect();
                f(&refs)
            }
        }
    }

    /// Flatten all tau values into a single Vec<f32>, skipping intermediate owned copies on CfC.
    pub fn flattened_tau(&self) -> Vec<f32> {
        match self {
            Self::CfC(cfc) => cfc
                .all_tau()
                .into_iter()
                .flat_map(|a| a.iter().copied())
                .collect(),
            Self::HdcLtc(bridge) => bridge
                .all_tau()
                .into_iter()
                .flat_map(|a| a.into_iter())
                .collect(),
            Self::HierarchicalCfC(hcfc) => hcfc.time_constants().to_vec(),
        }
    }

    /// Scale tau values for all layers uniformly
    pub fn scale_tau_all(&mut self, scale: f32) {
        if let Self::CfC(cfc) = self {
            cfc.scale_tau_all(scale);
        }
        // HierarchicalCfC tau is modulated by top-down feedback internally
    }

    /// Set tau values for all layers
    pub fn set_tau_all(&mut self, taus: Vec<Array1<f32>>) {
        if let Self::CfC(cfc) = self {
            for (cell, tau) in cfc.cells.iter_mut().zip(taus.into_iter()) {
                cell.tau.assign(&tau);
            }
        }
    }

    /// Adaptively resize HDC dimension based on prediction error (HdcLtc only)
    pub fn maybe_resize(&mut self, current_error: f32) {
        if let Self::HdcLtc(bridge) = self {
            bridge.maybe_resize(current_error);
        }
    }

    /// Get backend type
    pub fn backend_type(&self) -> TemporalBackend {
        match self {
            Self::CfC(_) => TemporalBackend::CfC,
            Self::HdcLtc(_) => TemporalBackend::HdcLtcUnified,
            Self::HierarchicalCfC(_) => TemporalBackend::HierarchicalCfC,
        }
    }

    /// Project input directly to HDC space, bypassing CfC temporal dynamics.
    ///
    /// Returns `None` for CfC and HierarchicalCfC backends (no HDC projection available).
    /// Returns `Some(Vec<f32>)` for HdcLtc backend with the raw HDC vector.
    pub fn project_to_hdc_vec(&self, input: &[f32]) -> Option<Vec<f32>> {
        match self {
            Self::CfC(_) | Self::HierarchicalCfC(_) => None,
            Self::HdcLtc(bridge) => Some(bridge.project_to_hdc_vec(input)),
        }
    }

    /// Get HDC dimension (returns None for CfC and HierarchicalCfC backends)
    pub fn hdc_dim(&self) -> Option<usize> {
        match self {
            Self::CfC(_) | Self::HierarchicalCfC(_) => None,
            Self::HdcLtc(bridge) => Some(bridge.hdc_dim()),
        }
    }

    /// Get per-scale outputs from the last hierarchical forward pass.
    ///
    /// Returns `Some(outputs)` only for HierarchicalCfC backend.
    /// Each element is the output at one temporal scale (fast → slow).
    #[allow(dead_code)]
    pub fn hierarchical_scale_outputs(
        &mut self,
        input: &Array1<f32>,
        dt: f32,
    ) -> Option<Vec<Array1<f32>>> {
        if let Self::HierarchicalCfC(hcfc) = self {
            Some(hcfc.forward_hierarchical(input, dt).scale_outputs)
        } else {
            None
        }
    }

    /// Get effective time constants from hierarchical backend.
    #[allow(dead_code)]
    pub fn hierarchical_effective_taus(&self) -> Option<Vec<f32>> {
        if let Self::HierarchicalCfC(hcfc) = self {
            Some(hcfc.time_constants().to_vec())
        } else {
            None
        }
    }
}
