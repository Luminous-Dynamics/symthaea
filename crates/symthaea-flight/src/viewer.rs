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
    #[cfg(feature = "mujoco-viewer")]
    viewer: mujoco_rs::MjViewer,
    title: String,
    frame_count: u64,
    #[cfg(not(feature = "mujoco-viewer"))]
    _phantom: std::marker::PhantomData<()>,
}

impl FlightViewer {
    /// Create a new viewer for the given simulator.
    ///
    /// Launches a passive viewer window. The simulation runs in the calling
    /// thread; the viewer renders asynchronously.
    #[cfg(feature = "mujoco-viewer")]
    pub fn new(sim: &MuJoCoSimulator, title: &str) -> Self {
        let viewer = mujoco_rs::MjViewer::launch_passive(
            &sim.model, 0, // window index
        );
        Self {
            viewer,
            title: title.to_string(),
            frame_count: 0,
        }
    }

    /// Create a stub viewer (no window) when mujoco-viewer feature is disabled.
    #[cfg(not(feature = "mujoco-viewer"))]
    pub fn new(_sim: &MuJoCoSimulator, title: &str) -> Self {
        Self {
            title: title.to_string(),
            frame_count: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Render one frame. Returns false when the viewer window is closed.
    ///
    /// Call this after each simulation step (or every N steps for performance).
    #[cfg(feature = "mujoco-viewer")]
    pub fn render(&mut self, sim: &mut MuJoCoSimulator) -> bool {
        self.frame_count += 1;
        self.viewer.sync(&mut sim.data);
        self.viewer.running()
    }

    /// Stub render (always returns true) when viewer is disabled.
    #[cfg(not(feature = "mujoco-viewer"))]
    pub fn render(&mut self, _sim: &mut MuJoCoSimulator) -> bool {
        self.frame_count += 1;
        true
    }

    /// Check if the viewer window is still open.
    #[cfg(feature = "mujoco-viewer")]
    pub fn is_running(&self) -> bool {
        self.viewer.running()
    }

    /// Stub: always returns true.
    #[cfg(not(feature = "mujoco-viewer"))]
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
