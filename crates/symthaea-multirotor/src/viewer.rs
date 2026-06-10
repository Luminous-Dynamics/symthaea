// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Interactive viewer for MuJoCo flight simulation.
//!
//! Wraps the mujoco-rs passive viewer for real-time 3D rendering and debugging
//! of quadrotor flight trajectories. Feature-gated behind `mujoco-viewer`.
//!
//! Two modes:
//! - **Passive viewer** (`mujoco-viewer` feature): real 3D rendering via MjViewer
//! - **Headless** (default): no viewer, just simulation (for benchmarks/CI)

use crate::mujoco_sim::MuJoCoSimulator;

/// Interactive flight viewer wrapping the MuJoCo visualizer.
///
/// With `mujoco-viewer` feature: opens a window with real-time 3D rendering,
/// camera controls, and physics visualization. Without it: a no-op stub that
/// always reports "running" (for headless benchmarks/CI).
pub struct FlightViewer {
    title: String,
    frame_count: u64,
    #[cfg(feature = "mujoco-viewer")]
    viewer: Option<mujoco_rs::viewer::MjViewer<std::sync::Arc<mujoco_rs::prelude::MjModel>>>,
}

impl FlightViewer {
    /// Create a new viewer for the given simulator.
    ///
    /// With `mujoco-viewer`: opens a passive viewer window.
    /// Without: returns a headless stub.
    pub fn new(sim: &MuJoCoSimulator, title: &str) -> Self {
        #[cfg(feature = "mujoco-viewer")]
        {
            let viewer =
                match mujoco_rs::viewer::MjViewer::launch_passive(sim.model_arc().clone(), 0) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        eprintln!(
                            "[FlightViewer] Failed to launch viewer: {e:?} — running headless"
                        );
                        None
                    }
                };
            Self {
                title: title.to_string(),
                frame_count: 0,
                viewer,
            }
        }
        #[cfg(not(feature = "mujoco-viewer"))]
        {
            let _ = sim;
            Self {
                title: title.to_string(),
                frame_count: 0,
            }
        }
    }

    /// Render one frame. Syncs simulation state to the viewer and renders.
    /// Returns true if the viewer window is still open (always true for headless).
    pub fn render(&mut self, sim: &mut MuJoCoSimulator) -> bool {
        self.frame_count += 1;
        #[cfg(feature = "mujoco-viewer")]
        {
            if let Some(ref mut viewer) = self.viewer {
                viewer.sync_data(sim.data_mut());
                viewer.render();
                return viewer.running();
            }
        }
        let _ = sim;
        true
    }

    /// Check if the viewer window is still open.
    pub fn is_running(&self) -> bool {
        #[cfg(feature = "mujoco-viewer")]
        {
            if let Some(ref viewer) = self.viewer {
                return viewer.running();
            }
        }
        true
    }

    /// Get the window title.
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Get the number of frames rendered.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}
