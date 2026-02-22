//! Interactive viewer for MuJoCo flight simulation.
//!
//! Wraps the mujoco-rs passive viewer for real-time 3D rendering and debugging
//! of quadrotor flight trajectories. Feature-gated behind `mujoco-viewer`.
//!
//! Two modes:
//! - **Passive viewer**: renders in background, simulation runs in main thread
//! - **Headless**: no viewer, just simulation (for benchmarks/CI)

use crate::mujoco_sim::MuJoCoSimulator;

/// Interactive flight viewer wrapping the MuJoCo visualizer.
///
/// Provides real-time 3D rendering of the quadrotor and its trajectory,
/// with camera controls and telemetry overlay.
pub struct FlightViewer {
    title: String,
    frame_count: u64,
    // The mujoco-viewer feature is not yet wired — this is a stub.
    // When enabled, it would hold an MjViewer instance that renders
    // passively in a separate thread.
}

impl FlightViewer {
    /// Create a new viewer for the given simulator.
    ///
    /// Without the `mujoco-viewer` feature, this is a no-op stub.
    pub fn new(_sim: &MuJoCoSimulator, title: &str) -> Self {
        Self {
            title: title.to_string(),
            frame_count: 0,
        }
    }

    /// Render one frame. Returns true (viewer stub always "runs").
    pub fn render(&mut self, _sim: &mut MuJoCoSimulator) -> bool {
        self.frame_count += 1;
        true
    }

    /// Check if the viewer window is still open.
    pub fn is_running(&self) -> bool {
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
