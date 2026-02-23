//! Motor bridge: cognitive loop → humanoid motor output.
//!
//! Translates the cognitive loop's HDC state vector into motor commands
//! for the humanoid controller. This is the embodiment pathway:
//!
//! ```text
//! CognitiveLoopService
//!   → current_thought (16384D ContinuousHV)
//!   → MotorBridge::translate()
//!   → HumanoidController::forward()
//!   → HumanoidCommand (21D torques)
//! ```
//!
//! The bridge also feeds back proprioceptive state from the simulator
//! as perception input to the next cognitive cycle.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

use crate::humanoid::{
    HumanoidCommand, HumanoidConfig, HumanoidController, HumanoidState,
    SimpleHumanoidSimulator, HumanoidPhysicsSimulator,
};

/// Bridge between cognitive loop and humanoid motor system.
///
/// Holds the trained controller and simulator, and provides the
/// interface between HDC vectors and physical motor commands.
pub struct MotorBridge {
    controller: HumanoidController,
    simulator: SimpleHumanoidSimulator,
    config: HumanoidConfig,
    /// Last motor command issued.
    last_command: HumanoidCommand,
    /// Timestep for physics simulation (seconds).
    dt: f64,
    /// Total steps executed.
    total_steps: usize,
}

impl MotorBridge {
    /// Create a new motor bridge with default configuration.
    pub fn new(genesis: &GenesisSeed) -> Self {
        let config = HumanoidConfig::default();
        let controller = HumanoidController::new(genesis, &config);
        let simulator = SimpleHumanoidSimulator::new();
        let dt = config.physics_dt();

        Self {
            controller,
            simulator,
            config,
            last_command: HumanoidCommand::zero(),
            dt,
            total_steps: 0,
        }
    }

    /// Create a motor bridge with a pre-trained controller (from checkpoint).
    pub fn from_controller(controller: HumanoidController, config: HumanoidConfig) -> Self {
        let simulator = SimpleHumanoidSimulator::new();
        let dt = config.physics_dt();

        Self {
            controller,
            simulator,
            config,
            last_command: HumanoidCommand::zero(),
            dt,
            total_steps: 0,
        }
    }

    /// Translate a cognitive loop HDC vector into motor output.
    ///
    /// This is the core embodiment step: the cognitive system's internal
    /// state (16,384D thought vector) drives the physical body.
    ///
    /// Returns the motor command and updated body state.
    pub fn step(&mut self, thought_hv: &ContinuousHV) -> (HumanoidCommand, &HumanoidState) {
        // 1. Controller translates thought → motor command
        let command = self.controller.forward(thought_hv, self.dt as f32);

        // 2. Physics simulation
        self.simulator.step(&command, self.dt);

        self.last_command = command.clone();
        self.total_steps += 1;

        (command, self.simulator.state())
    }

    /// Get the current body state (for encoding back into the perception loop).
    pub fn body_state(&self) -> &HumanoidState {
        self.simulator.state()
    }

    /// Get the last motor command issued.
    pub fn last_command(&self) -> &HumanoidCommand {
        &self.last_command
    }

    /// Get the underlying controller (for training/checkpoint).
    pub fn controller(&self) -> &HumanoidController {
        &self.controller
    }

    /// Get mutable controller access (for training steps).
    pub fn controller_mut(&mut self) -> &mut HumanoidController {
        &mut self.controller
    }

    /// Reset the simulator and controller for a new episode.
    pub fn reset(&mut self) {
        self.simulator.reset();
        self.controller.reset();
        self.last_command = HumanoidCommand::zero();
        self.total_steps = 0;
    }

    /// Total physics steps executed.
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }
}
