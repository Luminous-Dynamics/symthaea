//! Interactive viewer for MuJoCo flight simulation (behind `mujoco` feature).
//!
//! Wraps the mujoco-rs visualizer for real-time rendering and debugging
//! of quadrotor flight trajectories.

use crate::mujoco_sim::MuJoCoSimulator;

/// Interactive flight viewer wrapping the MuJoCo visualizer.
///
/// Provides real-time 3D rendering of the quadrotor and its trajectory,
/// with camera controls and telemetry overlay.
pub struct FlightViewer {
    // mujoco-rs Viewer would go here
    title: String,
    frame_count: u64,
}

impl FlightViewer {
    /// Create a new viewer for the given simulator.
    pub fn new(sim: &MuJoCoSimulator, title: &str) -> Self {
        let _ = sim; // Would initialize from sim's model
        Self {
            title: title.to_string(),
            frame_count: 0,
        }
    }

    /// Render one frame.
    ///
    /// In a real implementation, this would call the MuJoCo visualizer's
    /// render loop and handle user input (camera, pause, etc.).
    pub fn render(&mut self, sim: &MuJoCoSimulator) -> bool {
        let _ = sim;
        self.frame_count += 1;
        true // Return false when window is closed
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
