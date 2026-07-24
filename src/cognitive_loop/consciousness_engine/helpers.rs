// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Helper methods — weight management, convergence detection, unified computation.

use super::super::types::ConsciousnessCache;
use super::ConsciousnessEngine;
use super::types::{ConsciousnessWeights, WeightConvergenceState};
use crate::consciousness::unified_consciousness_pipeline::UnifiedConsciousnessPipeline;

impl ConsciousnessEngine {
    /// Compute weight variance across recent history (100-sample window).
    pub(super) fn weight_variance(&self) -> f64 {
        let hist = &self.cache.weight_history;
        if hist.len() < 2 {
            return 0.0;
        }
        let n = hist.len() as f64;
        let means: [f64; 4] = {
            let mut m = [0.0; 4];
            for w in hist {
                for i in 0..4 {
                    m[i] += w[i];
                }
            }
            for i in 0..4 {
                m[i] /= n;
            }
            m
        };
        let mut var = 0.0;
        for w in hist {
            for i in 0..4 {
                var += (w[i] - means[i]).powi(2);
            }
        }
        var / (n - 1.0) / 4.0 // Mean per-weight variance
    }

    /// Classify the current weight convergence dynamics.
    pub(super) fn convergence_state(&self) -> WeightConvergenceState {
        let hist = &self.cache.weight_history;
        if hist.len() < 20 {
            return WeightConvergenceState::Initializing;
        }
        if self.cache.converged_streak >= 50 {
            return WeightConvergenceState::Converged;
        }
        let variance = self.weight_variance();
        if variance > 0.005 {
            return WeightConvergenceState::Oscillating;
        }
        // Compare recent half vs older half variance
        let mid = hist.len() / 2;
        let compute_half_var = |start: usize, end: usize| -> f64 {
            let slice: Vec<_> = hist.iter().skip(start).take(end - start).collect();
            if slice.len() < 2 {
                return 0.0;
            }
            let n = slice.len() as f64;
            let mut means = [0.0; 4];
            for w in &slice {
                for i in 0..4 {
                    means[i] += w[i];
                }
            }
            for m in &mut means {
                *m /= n;
            }
            let mut v = 0.0;
            for w in &slice {
                for i in 0..4 {
                    v += (w[i] - means[i]).powi(2);
                }
            }
            v / (n - 1.0) / 4.0
        };
        let older_var = compute_half_var(0, mid);
        let recent_var = compute_half_var(mid, hist.len());
        if recent_var < older_var * 0.8 {
            WeightConvergenceState::Converging
        } else {
            WeightConvergenceState::Oscillating
        }
    }

    /// Update dynamic consciousness weights based on structural Phi decomposition.
    ///
    /// EMA-smooths the emergence ratio (alpha adaptive to weight stability),
    /// then modulates weights: high emergence boosts spectral (IIT is capturing real integration),
    /// low emergence boosts equation/pipeline (local metrics more informative).
    pub(super) fn update_weights_from_emergence(&mut self, er: f64) {
        // Adaptive alpha: high variance → slow down (alpha × 0.5 at variance=0.01)
        let base_alpha = 0.3;
        let variance = self.weight_variance();
        let mut alpha = base_alpha * (1.0 / (1.0 + 50.0 * variance));

        // Track converged streak
        if variance < 0.001 {
            self.cache.converged_streak += 1;
        } else {
            self.cache.converged_streak = 0;
        }

        // Alpha floor when converged — lock to narrow band
        if self.cache.converged_streak >= 50 {
            alpha = alpha.max(0.05).min(0.1);
        }

        // EMA smooth the emergence ratio
        let smoothed = match self.cache.smoothed_emergence_ratio {
            Some(prev) => alpha * er + (1.0 - alpha) * prev,
            None => er,
        };
        self.cache.smoothed_emergence_ratio = Some(smoothed);

        // Modulation: tanh(smoothed_er - 1.0) maps to [-1, 1]
        // er=1.0 → neutral, er>1 → positive (boost spectral), er<1 → negative
        let modulation = (smoothed - 1.0).tanh();

        // Start from defaults
        let mut w = ConsciousnessWeights::default();

        // High emergence → boost spectral by up to +0.10, reduce equation/pipeline by 0.05 each
        // Low emergence → reduce spectral, boost equation/pipeline
        w.spectral += modulation * 0.10;
        w.equation -= modulation * 0.05;
        w.pipeline -= modulation * 0.05;
        // Multimodal stays constant (cross-modal binding is orthogonal to hierarchy)

        // Clamp all weights ≥ 0.05
        w.spectral = w.spectral.max(0.05);
        w.equation = w.equation.max(0.05);
        w.pipeline = w.pipeline.max(0.05);
        w.multimodal = w.multimodal.max(0.05);

        // Normalize to sum=1.0
        w.normalize();

        // Record weight snapshot
        self.cache.weight_history.push_back(w.as_array());
        if self.cache.weight_history.len() > 100 {
            self.cache.weight_history.pop_front();
        }

        self.cache.weights = w;
    }

    /// Compute weighted consensus consciousness level.
    ///
    /// Uses dynamic weights from structural Phi analysis (defaults to
    /// 0.35 spectral + 0.25 equation + 0.25 pipeline + 0.15 multimodal).
    ///
    /// Spectral Phi is mapped [0,∞) → [0,1] via sigmoid: 2/(1+exp(-phi)) - 1
    pub(super) fn compute_unified(
        &self,
        spectral_mip_phi: Option<f64>,
        multimodal_phi: Option<f64>,
        equation_v2: Option<f64>,
        pipeline: Option<f64>,
    ) -> f64 {
        // Normalize spectral phi from [0, ∞) to [0, 1] via shifted sigmoid
        let spectral_norm = spectral_mip_phi
            .filter(|phi| phi.is_finite())
            .map(|phi| 2.0 / (1.0 + (-phi).exp()) - 1.0)
            .unwrap_or(0.0);

        let w = &self.cache.weights;

        // Weighted consensus over PRESENT systems only, renormalized.
        //
        // HONESTY FIX (2026-07-15): three of the four systems are `None` at the
        // production construction site (constructor.rs). They used to enter this
        // sum as 0.0 at full weight, silently deflating unified consciousness —
        // the "consensus" was really `w.spectral × spectral_norm` mislabeled.
        // Absent systems now drop out of both numerator and denominator; the
        // value is a true consensus of whatever systems actually ran.
        // (docs/PHI_SIGNAL_TRACE_2026-07-15.md symptom 1.)
        let mut num = w.spectral * spectral_norm;
        let mut den = w.spectral;
        if let Some(v) = equation_v2 {
            num += w.equation * v;
            den += w.equation;
        }
        if let Some(v) = pipeline {
            num += w.pipeline * v;
            den += w.pipeline;
        }
        if let Some(v) = multimodal_phi {
            num += w.multimodal * v;
            den += w.multimodal;
        }
        let unified = if den > 0.0 { num / den } else { 0.0 };

        // Consciousness floor: prevent total consciousness death.
        // Even with all subsystems at zero, the temporal continuity of the
        // CfC network provides a baseline level of information processing.
        // 0.05 = minimal but nonzero consciousness (enough for Red safety level).
        const CONSCIOUSNESS_FLOOR: f64 = 0.05;
        unified.max(CONSCIOUSNESS_FLOOR).clamp(0.0, 1.0)
    }

    /// Populate a ConsciousnessCache from the engine's internal cache.
    ///
    /// Used to maintain backward compatibility with CognitiveLoopService
    /// carryover fields.
    pub fn update_cache(&self, cache: &mut ConsciousnessCache) {
        cache.last_spectral_mip_phi = self.cache.last_spectral_mip_phi;
        cache.last_spectral_mip_adapted = self.cache.last_spectral_mip_adapted;
        cache.last_spectral_mip_active_dim_count = self.cache.last_spectral_mip_active_dim_count;
        cache.last_sigma = self.cache.last_sigma;
        cache.last_multimodal_phi = self.cache.last_multimodal_phi;
        cache.last_equation_v2_consciousness = self.cache.last_equation_v2_consciousness;
        cache.last_hierarchical_mip_phi = self.cache.last_hierarchical_mip_phi;
        cache.last_structural_phi = self.cache.last_structural_phi.clone();
    }

    /// Borrow the multi-modal integrator (if present).
    pub fn multi_modal_integrator(
        &self,
    ) -> Option<&crate::consciousness::multi_modal_integration::MultiModalIntegrator> {
        self.multi_modal_integrator.as_ref()
    }

    /// Borrow the consciousness equation v2 (if present).
    pub fn consciousness_equation_v2(
        &self,
    ) -> Option<&crate::consciousness::consciousness_equation_v2::ConsciousnessEquationV2> {
        self.consciousness_equation_v2.as_ref()
    }

    /// Borrow the unified consciousness pipeline (if present).
    pub fn unified_consciousness_pipeline(&self) -> Option<&UnifiedConsciousnessPipeline> {
        self.unified_consciousness_pipeline.as_ref()
    }

    /// Get last multimodal binding coherence (for CTC wiring).
    #[cfg(feature = "ctc_wiring")]
    pub fn last_multimodal_binding_coherence(&self) -> f64 {
        self.cache.last_binding_coherence
    }

    /// Get last PAC modulation index (for CTC wiring).
    #[cfg(feature = "ctc_wiring")]
    pub fn last_pac_modulation(&self) -> f64 {
        self.cache.last_pac_modulation
    }
}
