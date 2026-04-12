// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core types for AUV control: state, command, config, chemical readings.

use serde::{Deserialize, Serialize};

/// Number of thruster actuators.
pub const NUM_ACTUATORS: usize = 8;

/// Number of kinematic state channels.
pub const NUM_KINEMATIC_CHANNELS: usize = 24;

/// Number of chemical sensor channels.
pub const NUM_CHEMICAL_CHANNELS: usize = 8;

/// Total state channels (kinematic + chemical).
pub const NUM_STATE_CHANNELS: usize = NUM_KINEMATIC_CHANNELS + NUM_CHEMICAL_CHANNELS;

// ── State ────────────────────────────────────────────────────────────────────

/// Full AUV state (32D): 24 kinematic + 8 chemical sensor readings.
///
/// Channels 0–12 are shared with quadrotor/helicopter (position, orientation, velocities).
/// Channels 13–23 are AUV-specific (depth, pressure, thruster feedback, buoyancy).
/// Channels 24–31 are chemical sensors (pH, turbidity, TDS, nitrates, arsenic, lead, chlorine, DO).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuvState {
    // ── Kinematic (24D) ──────────────────────────────────────────────
    /// World-frame position [x, y, z] in meters. z is depth (positive downward).
    pub position: [f64; 3],
    /// Orientation quaternion [w, x, y, z].
    pub quaternion: [f64; 4],
    /// Body-frame linear velocity [surge, sway, heave] in m/s.
    pub linear_velocity: [f64; 3],
    /// Body-frame angular velocity [roll_rate, pitch_rate, yaw_rate] in rad/s.
    pub angular_velocity: [f64; 3],
    /// Depth from surface (meters, positive downward).
    pub depth: f64,
    /// Hydrostatic pressure (kPa).
    pub pressure: f64,
    /// Water temperature (°C).
    pub water_temperature: f64,
    /// Current buoyancy state (positive = ascending, negative = descending).
    pub buoyancy_force: f64,
    /// Thruster RPM feedback (8 thrusters, normalized 0–1).
    pub thruster_feedback: [f64; 8],

    // ── Chemical Sensors (8D) ────────────────────────────────────────
    /// Chemical sensor readings matching water-purity zome QualityReading.
    pub chemical: ChemicalReadings,
}

/// Chemical sensor readings (8D) — maps 1:1 to water-purity QualityReading.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ChemicalReadings {
    /// pH level (0–14).
    pub ph: f64,
    /// Turbidity in NTU (0–1000).
    pub turbidity_ntu: f64,
    /// Total dissolved solids in ppm (0–5000).
    pub tds_ppm: f64,
    /// Nitrate concentration in mg/L (0–100).
    pub nitrates_mg_l: f64,
    /// Arsenic concentration in µg/L (0–100).
    pub arsenic_ug_l: f64,
    /// Lead concentration in µg/L (0–100).
    pub lead_ug_l: f64,
    /// Free chlorine in mg/L (0–5).
    pub chlorine_mg_l: f64,
    /// Dissolved oxygen in mg/L (0–20).
    pub dissolved_oxygen_mg_l: f64,
}

impl ChemicalReadings {
    /// Clean freshwater readings (all within WHO guidelines).
    pub fn clean_freshwater() -> Self {
        Self {
            ph: 7.0,
            turbidity_ntu: 1.0,
            tds_ppm: 200.0,
            nitrates_mg_l: 5.0,
            arsenic_ug_l: 2.0,
            lead_ug_l: 1.0,
            chlorine_mg_l: 0.5,
            dissolved_oxygen_mg_l: 8.0,
        }
    }

    /// Whether all readings meet WHO drinking water guidelines.
    pub fn meets_who_standards(&self) -> bool {
        self.ph >= 6.5
            && self.ph <= 8.5
            && self.turbidity_ntu <= 5.0
            && self.tds_ppm <= 1000.0
            && self.nitrates_mg_l <= 50.0
            && self.arsenic_ug_l <= 10.0
            && self.lead_ug_l <= 10.0
            && self.chlorine_mg_l <= 5.0
            && self.dissolved_oxygen_mg_l >= 4.0
    }

    /// Potability score (0.0–1.0) based on distance from WHO thresholds.
    pub fn potability_score(&self) -> f64 {
        let scores = [
            1.0 - ((self.ph - 7.0).abs() / 1.5).min(1.0), // pH
            1.0 - (self.turbidity_ntu / 5.0).min(1.0),    // turbidity
            1.0 - (self.tds_ppm / 1000.0).min(1.0),       // TDS
            1.0 - (self.nitrates_mg_l / 50.0).min(1.0),   // nitrates
            1.0 - (self.arsenic_ug_l / 10.0).min(1.0),    // arsenic
            1.0 - (self.lead_ug_l / 10.0).min(1.0),       // lead
            (self.dissolved_oxygen_mg_l / 8.0).min(1.0),  // DO (higher = better)
            1.0 - ((self.chlorine_mg_l - 0.5).abs() / 4.5).min(1.0), // chlorine (optimal ~0.5)
        ];
        scores.iter().sum::<f64>() / scores.len() as f64
    }

    /// Pack into 8-element f32 array.
    pub fn to_channels(&self) -> [f32; NUM_CHEMICAL_CHANNELS] {
        [
            self.ph as f32,
            self.turbidity_ntu as f32,
            self.tds_ppm as f32,
            self.nitrates_mg_l as f32,
            self.arsenic_ug_l as f32,
            self.lead_ug_l as f32,
            self.chlorine_mg_l as f32,
            self.dissolved_oxygen_mg_l as f32,
        ]
    }

    /// Pack into 8-element f64 array (for noise model processing).
    pub fn to_array(&self) -> [f64; NUM_CHEMICAL_CHANNELS] {
        [
            self.ph,
            self.turbidity_ntu,
            self.tds_ppm,
            self.nitrates_mg_l,
            self.arsenic_ug_l,
            self.lead_ug_l,
            self.chlorine_mg_l,
            self.dissolved_oxygen_mg_l,
        ]
    }

    /// Construct from 8-element f64 array.
    pub fn from_array(a: &[f64; NUM_CHEMICAL_CHANNELS]) -> Self {
        Self {
            ph: a[0],
            turbidity_ntu: a[1],
            tds_ppm: a[2],
            nitrates_mg_l: a[3],
            arsenic_ug_l: a[4],
            lead_ug_l: a[5],
            chlorine_mg_l: a[6],
            dissolved_oxygen_mg_l: a[7],
        }
    }
}

impl AuvState {
    /// Pack full state into 32 f32 channels for HDC encoding.
    pub fn to_channels(&self) -> [f32; NUM_STATE_CHANNELS] {
        let mut channels = [0.0f32; NUM_STATE_CHANNELS];
        // Kinematic (24D)
        channels[0] = self.position[0] as f32;
        channels[1] = self.position[1] as f32;
        channels[2] = self.position[2] as f32;
        channels[3] = self.quaternion[0] as f32;
        channels[4] = self.quaternion[1] as f32;
        channels[5] = self.quaternion[2] as f32;
        channels[6] = self.quaternion[3] as f32;
        channels[7] = self.linear_velocity[0] as f32;
        channels[8] = self.linear_velocity[1] as f32;
        channels[9] = self.linear_velocity[2] as f32;
        channels[10] = self.angular_velocity[0] as f32;
        channels[11] = self.angular_velocity[1] as f32;
        channels[12] = self.angular_velocity[2] as f32;
        channels[13] = self.depth as f32;
        channels[14] = self.pressure as f32;
        channels[15] = self.water_temperature as f32;
        channels[16] = self.buoyancy_force as f32;
        // Thruster feedback (channels 17–23, using first 7 of 8)
        for i in 0..7 {
            channels[17 + i] = self.thruster_feedback[i] as f32;
        }
        // Chemical (8D, channels 24–31)
        let chem = self.chemical.to_channels();
        for i in 0..NUM_CHEMICAL_CHANNELS {
            channels[NUM_KINEMATIC_CHANNELS + i] = chem[i];
        }
        channels
    }

    /// Depth below surface (meters, positive).
    pub fn depth_m(&self) -> f64 {
        self.depth.max(0.0)
    }

    /// Linear speed magnitude (m/s).
    pub fn speed(&self) -> f64 {
        let [u, v, w] = self.linear_velocity;
        (u * u + v * v + w * w).sqrt()
    }

    /// Whether all values are finite.
    pub fn is_finite(&self) -> bool {
        self.position.iter().all(|v| v.is_finite())
            && self.quaternion.iter().all(|v| v.is_finite())
            && self.linear_velocity.iter().all(|v| v.is_finite())
            && self.angular_velocity.iter().all(|v| v.is_finite())
            && self.depth.is_finite()
    }

    /// Neutral buoyancy at given depth with clean water.
    pub fn neutral_buoyancy(depth: f64) -> Self {
        Self {
            position: [0.0, 0.0, depth],
            quaternion: [1.0, 0.0, 0.0, 0.0],
            linear_velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            depth,
            pressure: 101.325 + depth * 9.81, // kPa (surface + hydrostatic)
            water_temperature: 15.0,
            buoyancy_force: 0.0, // Neutral
            thruster_feedback: [0.0; 8],
            chemical: ChemicalReadings::clean_freshwater(),
        }
    }

    /// Surface state.
    pub fn surface() -> Self {
        Self::neutral_buoyancy(0.0)
    }
}

// ── Command ──────────────────────────────────────────────────────────────────

/// Motor command (8D): 8 independent thruster RPMs.
///
/// Thruster layout (typical vectored configuration):
/// - 0–3: Horizontal thrusters (surge/sway/yaw)
/// - 4–5: Vertical thrusters (heave)
/// - 6–7: Vectored thrusters (pitch/roll assist)
///
/// Each value in [-1.0, 1.0] where negative = reverse thrust.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AuvCommand {
    pub thrusters: [f32; NUM_ACTUATORS],
}

impl AuvCommand {
    /// All thrusters off.
    pub fn zero() -> Self {
        Self {
            thrusters: [0.0; NUM_ACTUATORS],
        }
    }

    /// Gentle forward thrust (surge only).
    pub fn forward(thrust: f32) -> Self {
        let mut cmd = Self::zero();
        cmd.thrusters[0] = thrust;
        cmd.thrusters[1] = thrust;
        cmd
    }

    /// Descend (vertical thrusters).
    pub fn descend(thrust: f32) -> Self {
        let mut cmd = Self::zero();
        cmd.thrusters[4] = -thrust;
        cmd.thrusters[5] = -thrust;
        cmd
    }

    /// Clamp all thrusters to [-1, 1].
    pub fn clamped(mut self) -> Self {
        for t in &mut self.thrusters {
            *t = t.clamp(-1.0, 1.0);
        }
        self
    }

    /// Mean absolute thrust (control effort).
    pub fn control_effort(&self) -> f32 {
        self.thrusters.iter().map(|t| t.abs()).sum::<f32>() / NUM_ACTUATORS as f32
    }
}

// ── Config ───────────────────────────────────────────────────────────────────

/// AUV configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuvConfig {
    pub genesis_phrase: String,
    pub learning_rate: f32,
    pub network_layers: usize,
    pub neurons_per_layer: usize,
    /// Physics timestep (default: 1/100 = 10ms — slower than air due to drag dominance).
    pub physics_hz: f64,
    /// Cognitive interval in physics steps (default: 10 → 10Hz).
    pub cognitive_interval: usize,
    pub steps_per_episode: usize,
    pub num_episodes: usize,
    /// Maximum operating depth (meters).
    pub max_depth: f64,
    /// Crush depth (abort threshold).
    pub crush_depth: f64,
}

impl Default for AuvConfig {
    fn default() -> Self {
        Self {
            genesis_phrase: "symthaea-auv-commons-steward".to_string(),
            learning_rate: 0.0005,
            network_layers: 2,
            neurons_per_layer: 8,
            physics_hz: 100.0,
            cognitive_interval: 10,
            steps_per_episode: 1000,
            num_episodes: 100,
            max_depth: 200.0,
            crush_depth: 500.0,
        }
    }
}

impl AuvConfig {
    pub fn physics_dt(&self) -> f64 {
        1.0 / self.physics_hz
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_water_meets_who() {
        let chem = ChemicalReadings::clean_freshwater();
        assert!(chem.meets_who_standards());
        assert!(chem.potability_score() > 0.8);
    }

    #[test]
    fn test_contaminated_water_fails_who() {
        let chem = ChemicalReadings {
            arsenic_ug_l: 50.0, // 5× WHO limit
            lead_ug_l: 30.0,    // 3× WHO limit
            ..ChemicalReadings::clean_freshwater()
        };
        assert!(!chem.meets_who_standards());
        assert!(chem.potability_score() < 0.8);
    }

    #[test]
    fn test_state_channels_count() {
        let state = AuvState::neutral_buoyancy(10.0);
        let channels = state.to_channels();
        assert_eq!(channels.len(), NUM_STATE_CHANNELS);
    }

    #[test]
    fn test_neutral_buoyancy_state() {
        let state = AuvState::neutral_buoyancy(50.0);
        assert!(state.is_finite());
        assert_eq!(state.depth_m(), 50.0);
        assert_eq!(state.speed(), 0.0);
        assert_eq!(state.buoyancy_force, 0.0);
    }

    #[test]
    fn test_command_zero() {
        let cmd = AuvCommand::zero();
        assert_eq!(cmd.control_effort(), 0.0);
    }

    #[test]
    fn test_command_forward() {
        let cmd = AuvCommand::forward(0.5);
        assert!(cmd.control_effort() > 0.0);
        assert_eq!(cmd.thrusters[0], 0.5);
    }

    #[test]
    fn test_command_clamped() {
        let cmd = AuvCommand {
            thrusters: [2.0, -2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        let clamped = cmd.clamped();
        assert_eq!(clamped.thrusters[0], 1.0);
        assert_eq!(clamped.thrusters[1], -1.0);
        assert_eq!(clamped.thrusters[2], 0.5);
    }

    #[test]
    fn test_serde_roundtrip() {
        let state = AuvState::neutral_buoyancy(25.0);
        let json = serde_json::to_string(&state).unwrap();
        let restored: AuvState = serde_json::from_str(&json).unwrap();
        assert!((restored.depth - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_chemical_potability_score_range() {
        let clean = ChemicalReadings::clean_freshwater();
        let score = clean.potability_score();
        assert!(score >= 0.0 && score <= 1.0, "Score must be [0,1]: {score}");
    }

    #[test]
    fn test_pressure_increases_with_depth() {
        let shallow = AuvState::neutral_buoyancy(10.0);
        let deep = AuvState::neutral_buoyancy(100.0);
        assert!(deep.pressure > shallow.pressure);
    }
}
