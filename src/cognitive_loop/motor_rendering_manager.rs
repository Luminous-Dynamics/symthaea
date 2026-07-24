// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Motor Rendering Manager — motor output bridge + canvas pipeline.
//!
//! Groups motor output bridge, pending requests, results, phi gating,
//! and canvas living topology into a single data-holder manager.

/// Consolidated motor output and canvas rendering subsystem.
///
/// Holds motor bridge, pending/last results, and canvas fields
/// that were previously scattered across `CognitiveLoopService`.
pub struct MotorRenderingManager {
    /// Motor output bridge: translates FEP MotorOutput commands into real-world
    /// actions (file I/O, shell commands, tests) via SimpleExecutor.
    /// When `None`, MotorOutput commands are no-ops (default behavior).
    pub(crate) output_bridge: Option<super::motor_output_bridge::MotorOutputBridge>,

    /// Pending motor action request (string data for the next MotorOutput dispatch).
    /// Set externally before a cycle to provide path/content/args for motor commands.
    pub(crate) pending_request: Option<super::motor_output_bridge::MotorActionRequest>,

    /// Last motor output result for FEP feedback.
    pub(crate) last_result: Option<super::motor_output_bridge::MotorOutputResult>,

    /// Phi value used for motor gating in the most recent motor execution (telemetry).
    pub(crate) last_phi: f64,

    /// Canvas living topology: consciousness-driven SVG generation.
    /// When enabled via `canvas` feature, generates real-time topology SVGs
    /// from cognitive telemetry with EMA-smoothed aesthetic mapping.
    #[cfg(feature = "canvas")]
    pub(crate) canvas_manager: Option<super::canvas_bridge::CanvasManager>,

    /// Most recent canvas SVG, drained into `CycleResult.canvas_svg` each cycle.
    #[cfg(feature = "canvas")]
    pub(crate) last_canvas_svg: Option<String>,

    /// Creative pipeline manager: generative art + music + synesthesia + gallery.
    #[cfg(feature = "creative")]
    pub(crate) creative_manager: Option<super::creative_bridge::CreativeManager>,

    /// Cached topological measurement shared by the canvas/creative snapshot
    /// builders (Betti numbers from the CfC compressed state). Refreshed on
    /// an interval in cycle_phase_dynamics — the Hodge pipeline is O(n³) in
    /// the subsampled dimensions, too heavy to recompute every cycle just
    /// for art. Before 2026-07-10 both snapshot builders left topology at
    /// dormant defaults, hollowing out every topology-driven art style.
    #[cfg(any(feature = "canvas", feature = "creative"))]
    pub(crate) art_topology:
        Option<super::consciousness_engine::topological_measure::TopologicalConsciousnessResult>,

    /// Cycle number at which `art_topology` was last refreshed.
    #[cfg(any(feature = "canvas", feature = "creative"))]
    pub(crate) art_topology_cycle: u64,

    /// Active artwork-observation window (feature `art-observer`): while
    /// `Some`, the rendered artwork is injected as the vision frame each
    /// cycle and ψ is accumulated; on expiry the Δψ verdict goes back to
    /// the CreativeManager.
    #[cfg(feature = "art-observer")]
    pub(crate) art_viewing: Option<ArtViewing>,

    /// Short ψ history ring (pre-viewing baseline for observer-Δψ).
    #[cfg(feature = "art-observer")]
    pub(crate) psi_history: std::collections::VecDeque<f64>,

    /// Total viewing windows ever opened — the A/B mode alternates arms on
    /// this counter's parity (even = real artwork, odd = scrambled control).
    #[cfg(feature = "art-observer")]
    pub(crate) art_viewings_started: u64,
}

/// State of one artwork-observation window.
#[cfg(feature = "art-observer")]
pub(crate) struct ArtViewing {
    /// Channel-matched pixel frame of the rendered artwork.
    pub frame: Vec<u8>,
    /// Cycles left in the viewing window.
    pub cycles_remaining: u32,
    /// Mean ψ over the pre-viewing history window.
    pub baseline_psi: f64,
    /// Accumulated ψ during viewing.
    pub psi_sum: f64,
    /// Number of ψ samples accumulated.
    pub psi_samples: u32,
    /// Whether this window shows the pixel-scrambled CONTROL frame instead
    /// of the artwork (A/B mode; control verdicts are never rewarded).
    pub is_control: bool,
}

impl MotorRenderingManager {
    /// Create a new MotorRenderingManager with default (empty) state.
    pub fn new() -> Self {
        Self::new_with_aesthetic_path(None)
    }

    /// Create a new MotorRenderingManager, forwarding an optional aesthetic memory path
    /// to the CreativeManager. When `None`, CreativeManager uses its default path.
    pub fn new_with_aesthetic_path(aesthetic_path: Option<std::path::PathBuf>) -> Self {
        Self {
            output_bridge: None,
            pending_request: None,
            last_result: None,
            last_phi: 0.0,
            #[cfg(feature = "canvas")]
            canvas_manager: Some(super::canvas_bridge::CanvasManager::new()),
            #[cfg(feature = "canvas")]
            last_canvas_svg: None,
            #[cfg(feature = "creative")]
            creative_manager: Some(super::creative_bridge::CreativeManager::new_with_path(
                aesthetic_path,
            )),
            #[cfg(any(feature = "canvas", feature = "creative"))]
            art_topology: None,
            #[cfg(any(feature = "canvas", feature = "creative"))]
            art_topology_cycle: 0,
            #[cfg(feature = "art-observer")]
            art_viewing: None,
            #[cfg(feature = "art-observer")]
            psi_history: std::collections::VecDeque::with_capacity(8),
            #[cfg(feature = "art-observer")]
            art_viewings_started: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motor_rendering_manager_new() {
        let mgr = MotorRenderingManager::new();
        assert!(mgr.output_bridge.is_none());
        assert!(mgr.pending_request.is_none());
        assert!(mgr.last_result.is_none());
        assert_eq!(mgr.last_phi, 0.0);
    }
}
