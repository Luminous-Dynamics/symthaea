// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
    HumanoidCommand, HumanoidConfig, HumanoidController, HumanoidHdcEncoder,
    HumanoidPhysicsSimulator, HumanoidState, SimpleHumanoidSimulator,
};

/// Bridge between cognitive loop and humanoid motor system.
///
/// Holds the trained controller and simulator, and provides the
/// interface between HDC vectors and physical motor commands.
///
/// The bridge is bidirectional:
/// - **Motor output**: `step(thought_hv)` translates thought → torques
/// - **Perception input**: `encode_perception()` encodes body state → HDC vector
///
/// ```text
/// Cognitive Loop ──thought_hv──→ MotorBridge ──torques──→ Simulator
///                                     ↑                       │
///                                     └── encode_perception() ←┘
/// ```
pub struct MotorBridge {
    controller: HumanoidController,
    simulator: SimpleHumanoidSimulator,
    encoder: HumanoidHdcEncoder,
    config: HumanoidConfig,
    /// Last motor command issued.
    last_command: HumanoidCommand,
    /// Last proprioceptive encoding.
    last_perception: Option<ContinuousHV>,
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
        let encoder = HumanoidHdcEncoder::new(genesis, 32);
        let dt = config.physics_dt();

        Self {
            controller,
            simulator,
            encoder,
            config,
            last_command: HumanoidCommand::zero(),
            last_perception: None,
            dt,
            total_steps: 0,
        }
    }

    /// Create a motor bridge with a pre-trained controller (from checkpoint).
    pub fn from_controller(controller: HumanoidController, config: HumanoidConfig) -> Self {
        let simulator = SimpleHumanoidSimulator::new();
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let encoder = HumanoidHdcEncoder::new(&genesis, 32);
        let dt = config.physics_dt();

        Self {
            controller,
            simulator,
            encoder,
            config,
            last_command: HumanoidCommand::zero(),
            last_perception: None,
            dt,
            total_steps: 0,
        }
    }

    /// Translate a cognitive loop HDC vector into motor output.
    ///
    /// This is the core embodiment step: the cognitive system's internal
    /// state (16,384D thought vector) drives the physical body.
    ///
    /// After physics simulation, the new body state is automatically encoded
    /// into a proprioceptive HDC vector (accessible via `last_perception()`).
    ///
    /// Returns the motor command and the proprioceptive encoding of the
    /// resulting body state.
    pub fn step(&mut self, thought_hv: &ContinuousHV) -> (HumanoidCommand, ContinuousHV) {
        // 1. Controller translates thought → motor command
        let command = self.controller.forward(thought_hv, self.dt as f32);

        // 2. Physics simulation
        self.simulator.step(&command, self.dt);

        // 3. Encode proprioceptive feedback
        let perception = self.encoder.encode(self.simulator.state());

        self.last_command = command.clone();
        self.last_perception = Some(perception.clone());
        self.total_steps += 1;

        (command, perception)
    }

    /// Encode the current body state into a proprioceptive HDC vector.
    ///
    /// This is the perception feedback pathway: the body's physical state
    /// (72 sensor channels) is encoded into the same 16,384D space that
    /// the cognitive loop operates in, enabling the loop to incorporate
    /// embodied experience.
    pub fn encode_perception(&mut self) -> ContinuousHV {
        let perception = self.encoder.encode(self.simulator.state());
        self.last_perception = Some(perception.clone());
        perception
    }

    /// Get the last proprioceptive encoding (from the most recent step or
    /// explicit `encode_perception()` call).
    pub fn last_perception(&self) -> Option<&ContinuousHV> {
        self.last_perception.as_ref()
    }

    /// Get the current body state (raw sensor values).
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

    /// Reset the simulator, controller, and encoder for a new episode.
    pub fn reset(&mut self) {
        self.simulator.reset();
        self.controller.reset();
        self.encoder.reset();
        self.last_command = HumanoidCommand::zero();
        self.last_perception = None;
        self.total_steps = 0;
    }

    /// Total physics steps executed.
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Write the last motor command to physical hardware via the HAL.
    ///
    /// Runs the command through the [`SafetyInterlock`](symthaea_hal::SafetyInterlock)
    /// and then applies it to the [`ServoOutput`](symthaea_hal::ServoOutput).
    #[cfg(feature = "hal")]
    pub fn write_to_hardware<I: symthaea_hal::I2cBus>(
        &self,
        safety: &mut symthaea_hal::SafetyInterlock,
        servo: &mut symthaea_hal::ServoOutput<I>,
    ) -> Result<(), symthaea_hal::HalError> {
        let safe_cmd = safety.filter_command(&self.last_command)?;
        servo.apply(&safe_cmd)
    }

    /// Translate sensor readings + cognitive state into a motor command.
    ///
    /// Intended for use with [`HalRuntime::run()`](symthaea_hal::HalRuntime::run).
    /// The `readings` parameter matches the sensor layout from the runtime's
    /// registered sensors. Currently, sensor readings are reserved for future
    /// proprioceptive integration; the motor command is derived from `thought_hv`.
    ///
    /// ```rust,ignore
    /// // In the runtime loop:
    /// runtime.run(Some(1000), |readings| {
    ///     bridge.step_from_readings(readings, &thought_hv)
    /// })?;
    /// ```
    #[cfg(feature = "hal")]
    pub fn step_from_readings(
        &mut self,
        _readings: &[Option<Vec<f32>>],
        thought_hv: &ContinuousHV,
    ) -> HumanoidCommand {
        let (command, _) = self.step(thought_hv);
        command
    }

    /// Create a closure suitable for [`HalRuntime::run()`](symthaea_hal::HalRuntime::run).
    ///
    /// Returns an `FnMut` that calls `step_from_readings` on each tick,
    /// translating the cognitive thought vector into motor commands.
    ///
    /// ```rust,ignore
    /// let thought_hv = cognitive_loop.current_thought();
    /// runtime.run(Some(1000), bridge.hal_callback(&thought_hv))?;
    /// ```
    #[cfg(feature = "hal")]
    pub fn hal_callback<'a>(
        &'a mut self,
        thought_hv: &'a ContinuousHV,
    ) -> impl FnMut(&[Option<Vec<f32>>]) -> HumanoidCommand + 'a {
        move |readings| self.step_from_readings(readings, thought_hv)
    }
}

#[cfg(test)]
#[cfg(feature = "hal")]
mod hal_tests {
    use super::*;

    fn make_bridge() -> MotorBridge {
        let genesis = GenesisSeed::from_phrase("test");
        MotorBridge::new(&genesis)
    }

    #[test]
    fn test_step_from_readings_basic() {
        let mut bridge = make_bridge();
        let hv = ContinuousHV::zero(16384);
        let cmd = bridge.step_from_readings(&[], &hv);
        // Should produce a valid command (all finite)
        for &t in &cmd.torques {
            assert!(t.is_finite());
        }
    }

    #[test]
    fn test_step_from_readings_increments_steps() {
        let mut bridge = make_bridge();
        let hv = ContinuousHV::zero(16384);
        assert_eq!(bridge.total_steps(), 0);
        let _ = bridge.step_from_readings(&[], &hv);
        assert_eq!(bridge.total_steps(), 1);
    }

    #[test]
    fn test_hal_callback_works() {
        let mut bridge = make_bridge();
        let hv = ContinuousHV::zero(16384);
        let mut cb = bridge.hal_callback(&hv);
        let cmd = cb(&[]);
        for &t in &cmd.torques {
            assert!(t.is_finite());
        }
    }
}
