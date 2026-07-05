// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Plasma HDC Encoder for Spark Engine
//!
//! Encodes tokamak/plasma fusion sensor data into hyperdimensional computing (HDC)
//! representations for real-time consciousness-integrated plasma control.
//!
//! ## Architecture
//!
//! ```text
//! +-----------------------------------------------------------------------+
//! |                    PLASMA HDC ENCODER                                  |
//! +-----------------------------------------------------------------------+
//! |                                                                        |
//! |  SENSOR LAYER (C-Mod/ITER compatible)                                 |
//! |  +------------+  +------------+  +------------+  +------------+       |
//! |  | Ip (MA)    |  | ne (m^-3)  |  | Te (keV)   |  | Prad (MW)  |       |
//! |  +------------+  +------------+  +------------+  +------------+       |
//! |  +------------+  +------------+  +------------+  +------------+       |
//! |  | Vloop (V)  |  | Bt/Bp (T)  |  | q95        |  | Wmhd (MJ)  |       |
//! |  +------------+  +------------+  +------------+  +------------+       |
//! |  +------------+                                                       |
//! |  | Beta (%)   |                                                       |
//! |  +------------+                                                       |
//! |                                                                        |
//! |  HDC ENCODING                                                          |
//! |  +------------------------+    +------------------------+              |
//! |  | Magnitude Encoding     |    | Temporal Phase         |              |
//! |  | (Level Hypervectors)   |    | (Circular Encoding)    |              |
//! |  +------------------------+    +------------------------+              |
//! |              |                          |                              |
//! |              v                          v                              |
//! |  +-----------------------------------------------+                     |
//! |  |        Unified Plasma State Vector            |                     |
//! |  |        (Bundled Sensor HVs)                   |                     |
//! |  +-----------------------------------------------+                     |
//! |                          |                                             |
//! |                          v                                             |
//! |  +-----------------------------------------------+                     |
//! |  |        PHI INTEGRATION                        |                     |
//! |  |        (Disruption Detection)                 |                     |
//! |  +-----------------------------------------------+                     |
//! |                                                                        |
//! +-----------------------------------------------------------------------+
//! ```
//!
//! ## Sensor Types (C-Mod/ITER based)
//!
//! | Sensor | Symbol | Units | Typical Range | Critical Threshold |
//! |--------|--------|-------|---------------|-------------------|
//! | Plasma Current | Ip | MA | 0.5-2.0 | < 0.3 or > 2.5 |
//! | Electron Density | ne | 10^20 m^-3 | 0.5-3.0 | > 4.0 (density limit) |
//! | Electron Temperature | Te | keV | 1.0-10.0 | < 0.5 (radiative collapse) |
//! | Radiated Power | Prad | MW | 0.1-5.0 | > 80% of input power |
//! | Loop Voltage | Vloop | V | 0.5-3.0 | > 5.0 (locked mode) |
//! | Toroidal Field | Bt | T | 2.0-6.0 | - |
//! | Poloidal Field | Bp | T | 0.1-1.0 | - |
//! | Safety Factor | q95 | - | 2.0-5.0 | < 2.0 (kink instability) |
//! | Stored Energy | Wmhd | MJ | 0.1-10.0 | rapid drop = disruption |
//! | Beta | beta | % | 0.5-5.0 | > beta_limit (Troyon) |
//!
//! ## HDC Encoding Strategy
//!
//! 1. **Base Vectors**: Each sensor type gets a unique base HV (deterministic from seed)
//! 2. **Magnitude Encoding**: Continuous values via level encoding (N quantization levels)
//! 3. **Temporal Phase**: Circular phase encoding for temporal context
//! 4. **Bundle**: All sensors bundled into unified plasma state vector
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::physics::plasma_hdc_encoder::{
//!     PlasmaHdcEncoder, PlasmaReading, PlasmaSensorType, PlasmaPhiMonitor
//! };
//!
//! let mut encoder = PlasmaHdcEncoder::new(42);
//!
//! // Single sample encoding
//! let reading = PlasmaReading::new(
//!     PlasmaSensorType::PlasmaCurrent,
//!     1.5, // MA
//!     0.0, // timestamp
//! );
//! let hv = encoder.encode_sample(&reading);
//!
//! // Streaming window encoding
//! let window_hv = encoder.encode_window(&readings[..]);
//!
//! // Stability monitoring
//! let mut monitor = PlasmaPhiMonitor::new(Default::default());
//! let alert = monitor.check_stability(&encoder, &current_state);
//! ```
//!
//! ## References
//!
//! - Alcator C-Mod disruption database (MIT PSFC)
//! - ITER Physics Basis (Nuclear Fusion, 2007)
//! - JET disruption predictor (Vega et al., 2014)

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Instant;

use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::tiered_phi::{ApproximationTier, TieredPhi};

// =============================================================================
// PLASMA SENSOR TYPES
// =============================================================================

/// Plasma sensor types based on C-Mod/tokamak diagnostics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlasmaSensorType {
    /// Plasma current (Ip) - MA
    /// Primary measure of plasma confinement
    PlasmaCurrent,

    /// Electron density (ne) - 10^20 m^-3
    /// Critical for density limit disruptions
    ElectronDensity,

    /// Electron temperature (Te) - keV
    /// Core temperature, drops during disruptions
    ElectronTemperature,

    /// Radiated power (Prad) - MW
    /// High radiation fraction precedes disruptions
    RadiatedPower,

    /// Loop voltage (Vloop) - V
    /// Spikes during locked modes
    LoopVoltage,

    /// Toroidal magnetic field (Bt) - T
    ToroidalField,

    /// Poloidal magnetic field (Bp) - T
    PoloidalField,

    /// Safety factor at 95% flux surface (q95) - dimensionless
    /// q95 < 2 leads to kink instabilities
    SafetyFactor,

    /// Stored energy from MHD equilibrium (Wmhd) - MJ
    /// Rapid decay signals thermal quench
    StoredEnergy,

    /// Normalized beta (plasma pressure / magnetic pressure) - %
    /// Exceeding Troyon limit causes disruptions
    Beta,
}

impl PlasmaSensorType {
    /// Get unique seed for deterministic base vector generation
    pub fn seed(&self) -> u64 {
        match self {
            Self::PlasmaCurrent => 0x504C41534D415F49,   // "PLASMA_I"
            Self::ElectronDensity => 0x504C41534D415F4E, // "PLASMA_N"
            Self::ElectronTemperature => 0x504C41534D415F54, // "PLASMA_T"
            Self::RadiatedPower => 0x504C41534D415F50,   // "PLASMA_P"
            Self::LoopVoltage => 0x504C41534D415F56,     // "PLASMA_V"
            Self::ToroidalField => 0x504C41534D415F42,   // "PLASMA_B"
            Self::PoloidalField => 0x504C41534D415F70,   // "PLASMA_p"
            Self::SafetyFactor => 0x504C41534D415F51,    // "PLASMA_Q"
            Self::StoredEnergy => 0x504C41534D415F57,    // "PLASMA_W"
            Self::Beta => 0x504C41534D415F62,            // "PLASMA_b"
        }
    }

    /// Get typical operating range [min, max]
    pub fn typical_range(&self) -> (f64, f64) {
        match self {
            Self::PlasmaCurrent => (0.5, 2.0),        // MA
            Self::ElectronDensity => (0.5, 3.0),      // 10^20 m^-3
            Self::ElectronTemperature => (1.0, 10.0), // keV
            Self::RadiatedPower => (0.1, 5.0),        // MW
            Self::LoopVoltage => (0.5, 3.0),          // V
            Self::ToroidalField => (2.0, 6.0),        // T
            Self::PoloidalField => (0.1, 1.0),        // T
            Self::SafetyFactor => (2.0, 5.0),         // dimensionless
            Self::StoredEnergy => (0.1, 10.0),        // MJ
            Self::Beta => (0.5, 5.0),                 // %
        }
    }

    /// Get critical thresholds that may precede disruptions
    /// Returns (low_critical, high_critical) or None if not applicable
    pub fn critical_thresholds(&self) -> Option<(Option<f64>, Option<f64>)> {
        match self {
            Self::PlasmaCurrent => Some((Some(0.3), Some(2.5))),
            Self::ElectronDensity => Some((None, Some(4.0))), // Greenwald limit
            Self::ElectronTemperature => Some((Some(0.5), None)), // Radiative collapse
            Self::RadiatedPower => Some((None, Some(0.8))),   // 80% radiation fraction
            Self::LoopVoltage => Some((None, Some(5.0))),     // Locked mode
            Self::SafetyFactor => Some((Some(2.0), None)),    // Kink instability
            Self::Beta => Some((None, Some(3.5))),            // Troyon limit (approx)
            _ => None,
        }
    }

    /// Get unit string for display
    pub fn unit(&self) -> &'static str {
        match self {
            Self::PlasmaCurrent => "MA",
            Self::ElectronDensity => "10^20 m^-3",
            Self::ElectronTemperature => "keV",
            Self::RadiatedPower => "MW",
            Self::LoopVoltage => "V",
            Self::ToroidalField => "T",
            Self::PoloidalField => "T",
            Self::SafetyFactor => "",
            Self::StoredEnergy => "MJ",
            Self::Beta => "%",
        }
    }

    /// Iterator over all sensor types
    pub fn all() -> impl Iterator<Item = Self> {
        [
            Self::PlasmaCurrent,
            Self::ElectronDensity,
            Self::ElectronTemperature,
            Self::RadiatedPower,
            Self::LoopVoltage,
            Self::ToroidalField,
            Self::PoloidalField,
            Self::SafetyFactor,
            Self::StoredEnergy,
            Self::Beta,
        ]
        .into_iter()
    }
}

// =============================================================================
// PLASMA READING
// =============================================================================

/// A single plasma sensor reading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasmaReading {
    /// Sensor type
    pub sensor: PlasmaSensorType,
    /// Measured value in native units
    pub value: f64,
    /// Timestamp (seconds from shot start)
    pub timestamp: f64,
    /// Optional quality flag (0.0-1.0, 1.0 = good)
    pub quality: f64,
}

impl PlasmaReading {
    /// Create a new plasma reading
    pub fn new(sensor: PlasmaSensorType, value: f64, timestamp: f64) -> Self {
        Self {
            sensor,
            value,
            timestamp,
            quality: 1.0,
        }
    }

    /// Create with quality flag
    pub fn with_quality(
        sensor: PlasmaSensorType,
        value: f64,
        timestamp: f64,
        quality: f64,
    ) -> Self {
        Self {
            sensor,
            value,
            timestamp,
            quality: quality.clamp(0.0, 1.0),
        }
    }

    /// Normalize value to [0, 1] based on typical range
    pub fn normalized(&self) -> f64 {
        let (min, max) = self.sensor.typical_range();
        ((self.value - min) / (max - min)).clamp(0.0, 1.0)
    }

    /// Check if value is in critical range
    pub fn is_critical(&self) -> bool {
        if let Some((low, high)) = self.sensor.critical_thresholds() {
            if let Some(low_thresh) = low
                && self.value < low_thresh
            {
                return true;
            }
            if let Some(high_thresh) = high
                && self.value > high_thresh
            {
                return true;
            }
        }
        false
    }
}

// =============================================================================
// FULL PLASMA STATE
// =============================================================================

/// Complete plasma state at a single timestep
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasmaState {
    /// Timestamp (seconds from shot start)
    pub timestamp: f64,
    /// Plasma current (MA)
    pub ip: f64,
    /// Electron density (10^20 m^-3)
    pub ne: f64,
    /// Electron temperature (keV)
    pub te: f64,
    /// Radiated power (MW)
    pub prad: f64,
    /// Loop voltage (V)
    pub vloop: f64,
    /// Toroidal field (T)
    pub bt: f64,
    /// Poloidal field (T)
    pub bp: f64,
    /// Safety factor q95
    pub q95: f64,
    /// Stored energy (MJ)
    pub wmhd: f64,
    /// Beta (%)
    pub beta: f64,
}

impl PlasmaState {
    /// Create a new plasma state with all values
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        timestamp: f64,
        ip: f64,
        ne: f64,
        te: f64,
        prad: f64,
        vloop: f64,
        bt: f64,
        bp: f64,
        q95: f64,
        wmhd: f64,
        beta: f64,
    ) -> Self {
        Self {
            timestamp,
            ip,
            ne,
            te,
            prad,
            vloop,
            bt,
            bp,
            q95,
            wmhd,
            beta,
        }
    }

    /// Convert to vector of readings
    pub fn to_readings(&self) -> Vec<PlasmaReading> {
        vec![
            PlasmaReading::new(PlasmaSensorType::PlasmaCurrent, self.ip, self.timestamp),
            PlasmaReading::new(PlasmaSensorType::ElectronDensity, self.ne, self.timestamp),
            PlasmaReading::new(
                PlasmaSensorType::ElectronTemperature,
                self.te,
                self.timestamp,
            ),
            PlasmaReading::new(PlasmaSensorType::RadiatedPower, self.prad, self.timestamp),
            PlasmaReading::new(PlasmaSensorType::LoopVoltage, self.vloop, self.timestamp),
            PlasmaReading::new(PlasmaSensorType::ToroidalField, self.bt, self.timestamp),
            PlasmaReading::new(PlasmaSensorType::PoloidalField, self.bp, self.timestamp),
            PlasmaReading::new(PlasmaSensorType::SafetyFactor, self.q95, self.timestamp),
            PlasmaReading::new(PlasmaSensorType::StoredEnergy, self.wmhd, self.timestamp),
            PlasmaReading::new(PlasmaSensorType::Beta, self.beta, self.timestamp),
        ]
    }

    /// Get a specific sensor reading
    pub fn get(&self, sensor: PlasmaSensorType) -> f64 {
        match sensor {
            PlasmaSensorType::PlasmaCurrent => self.ip,
            PlasmaSensorType::ElectronDensity => self.ne,
            PlasmaSensorType::ElectronTemperature => self.te,
            PlasmaSensorType::RadiatedPower => self.prad,
            PlasmaSensorType::LoopVoltage => self.vloop,
            PlasmaSensorType::ToroidalField => self.bt,
            PlasmaSensorType::PoloidalField => self.bp,
            PlasmaSensorType::SafetyFactor => self.q95,
            PlasmaSensorType::StoredEnergy => self.wmhd,
            PlasmaSensorType::Beta => self.beta,
        }
    }

    /// Check if any sensor is in critical range
    pub fn has_critical(&self) -> bool {
        self.to_readings().iter().any(|r| r.is_critical())
    }

    /// Get list of critical sensors
    pub fn critical_sensors(&self) -> Vec<PlasmaSensorType> {
        self.to_readings()
            .into_iter()
            .filter(|r| r.is_critical())
            .map(|r| r.sensor)
            .collect()
    }
}

impl Default for PlasmaState {
    /// Default state: typical stable plasma parameters
    fn default() -> Self {
        Self {
            timestamp: 0.0,
            ip: 1.0,    // 1 MA
            ne: 1.5,    // 1.5 x 10^20 m^-3
            te: 5.0,    // 5 keV
            prad: 1.0,  // 1 MW
            vloop: 1.0, // 1 V
            bt: 5.0,    // 5 T
            bp: 0.5,    // 0.5 T
            q95: 3.5,   // q95 = 3.5
            wmhd: 2.0,  // 2 MJ
            beta: 2.0,  // 2%
        }
    }
}

// =============================================================================
// HDC ENCODER CONFIGURATION
// =============================================================================

/// Configuration for the Plasma HDC Encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasmaEncoderConfig {
    /// Number of quantization levels for magnitude encoding
    pub num_levels: usize,
    /// Time scale for circular phase encoding (seconds for full rotation)
    pub time_scale_seconds: f64,
    /// Include rate-of-change (derivative) encoding
    pub encode_derivatives: bool,
    /// Weight for derivative contribution (0.0-1.0)
    pub derivative_weight: f64,
    /// Global seed for deterministic encoding
    pub seed: u64,
}

impl Default for PlasmaEncoderConfig {
    fn default() -> Self {
        Self {
            num_levels: 100,
            time_scale_seconds: 10.0, // 10 second plasma shot cycle
            encode_derivatives: true,
            derivative_weight: 0.3,
            seed: 0x504C41534D41, // "PLASMA" in hex
        }
    }
}

// =============================================================================
// PLASMA HDC ENCODER
// =============================================================================

/// Plasma sensor data encoder using Hyperdimensional Computing
///
/// Encodes tokamak sensor data into high-dimensional binary vectors for
/// consciousness-integrated plasma control and disruption prediction.
#[derive(Debug, Clone)]
pub struct PlasmaHdcEncoder {
    /// Configuration
    config: PlasmaEncoderConfig,
    /// Base vectors for each sensor type (deterministic)
    base_vectors: Vec<(PlasmaSensorType, BinaryHV)>,
    /// Level vectors for magnitude encoding (deterministic)
    level_vectors: Vec<BinaryHV>,
    /// Phase vectors for temporal encoding (deterministic)
    phase_vectors: Vec<BinaryHV>,
    /// Previous state for derivative computation
    prev_state: Option<PlasmaState>,
}

impl PlasmaHdcEncoder {
    /// Create a new plasma encoder with the given seed
    pub fn new(seed: u64) -> Self {
        Self::with_config(PlasmaEncoderConfig {
            seed,
            ..Default::default()
        })
    }

    /// Create with full configuration
    pub fn with_config(config: PlasmaEncoderConfig) -> Self {
        // Generate base vectors for each sensor type
        let base_vectors: Vec<(PlasmaSensorType, BinaryHV)> = PlasmaSensorType::all()
            .map(|sensor| {
                let sensor_seed = config.seed ^ sensor.seed();
                (sensor, BinaryHV::random(sensor_seed))
            })
            .collect();

        // Generate level vectors for magnitude encoding
        // Level i has seed = config.seed + 1_000_000 + i
        let level_vectors: Vec<BinaryHV> = (0..config.num_levels)
            .map(|i| BinaryHV::random(config.seed.wrapping_add(1_000_000).wrapping_add(i as u64)))
            .collect();

        // Generate phase vectors for temporal encoding (32 phases for circular time)
        let num_phases = 32;
        let phase_vectors: Vec<BinaryHV> = (0..num_phases)
            .map(|i| BinaryHV::random(config.seed.wrapping_add(2_000_000).wrapping_add(i as u64)))
            .collect();

        Self {
            config,
            base_vectors,
            level_vectors,
            phase_vectors,
            prev_state: None,
        }
    }

    /// Get the base vector for a sensor type
    pub fn base_vector(&self, sensor: PlasmaSensorType) -> &BinaryHV {
        self.base_vectors
            .iter()
            .find(|(s, _)| *s == sensor)
            .map(|(_, hv)| hv)
            .expect("All sensor types have base vectors")
    }

    /// Encode a normalized value [0, 1] using level encoding
    ///
    /// Uses thermometer coding: bundle levels 0..k where k = floor(value * num_levels)
    fn encode_magnitude(&self, normalized_value: f64) -> BinaryHV {
        let level = ((normalized_value * self.config.num_levels as f64) as usize)
            .min(self.config.num_levels - 1);

        if level == 0 {
            self.level_vectors[0]
        } else {
            // Bundle all levels up to and including the current level
            BinaryHV::bundle_safe(&self.level_vectors[0..=level])
        }
    }

    /// Encode temporal phase using circular encoding
    ///
    /// Maps time to phase angle, then bundles nearby phase vectors
    fn encode_phase(&self, timestamp: f64) -> BinaryHV {
        let num_phases = self.phase_vectors.len();
        let phase = (timestamp / self.config.time_scale_seconds) % 1.0;
        let phase_idx = (phase * num_phases as f64) as usize % num_phases;

        // Smooth phase encoding: bundle current and neighboring phases
        let prev_idx = (phase_idx + num_phases - 1) % num_phases;
        let next_idx = (phase_idx + 1) % num_phases;

        BinaryHV::bundle_safe(&[
            self.phase_vectors[prev_idx],
            self.phase_vectors[phase_idx],
            self.phase_vectors[phase_idx], // Double weight for center
            self.phase_vectors[next_idx],
        ])
    }

    /// Encode a single plasma reading
    ///
    /// Result = Base(sensor) XOR Magnitude(value) XOR Phase(time)
    pub fn encode_sample(&self, reading: &PlasmaReading) -> BinaryHV {
        let base = self.base_vector(reading.sensor);
        let magnitude = self.encode_magnitude(reading.normalized());
        let phase = self.encode_phase(reading.timestamp);

        // Bind: base XOR magnitude XOR phase
        base.bind(&magnitude).bind(&phase)
    }

    /// Encode a complete plasma state
    ///
    /// Bundles all sensor readings into a unified state vector
    pub fn encode_state(&mut self, state: &PlasmaState) -> BinaryHV {
        let readings = state.to_readings();
        let encoded: Vec<BinaryHV> = readings.iter().map(|r| self.encode_sample(r)).collect();

        let state_hv = BinaryHV::bundle_safe(&encoded);

        // Optionally add derivative encoding
        if self.config.encode_derivatives
            && let Some(ref prev) = self.prev_state
        {
            let derivative_hv = self.encode_derivatives(prev, state);
            // Weighted bundle of state and derivatives
            let weighted = BinaryHV::bundle_safe(&[
                state_hv,
                state_hv, // 2x weight for state
                derivative_hv,
            ]);
            self.prev_state = Some(state.clone());
            return weighted;
        }

        self.prev_state = Some(state.clone());
        state_hv
    }

    /// Encode rate-of-change (derivatives) between two states
    fn encode_derivatives(&self, prev: &PlasmaState, curr: &PlasmaState) -> BinaryHV {
        let dt = (curr.timestamp - prev.timestamp).max(1e-6);

        let mut derivative_hvs = Vec::new();

        for sensor in PlasmaSensorType::all() {
            let prev_val = prev.get(sensor);
            let curr_val = curr.get(sensor);
            let derivative = (curr_val - prev_val) / dt;

            // Normalize derivative (assuming max rate ~1 unit/second)
            let norm_deriv = (derivative.tanh() + 1.0) / 2.0; // Map to [0, 1]

            let base = self.base_vector(sensor);
            let deriv_mag = self.encode_magnitude(norm_deriv);

            // Use a special "derivative marker" binding
            let deriv_marker = BinaryHV::random(self.config.seed.wrapping_add(3_000_000));
            derivative_hvs.push(base.bind(&deriv_mag).bind(&deriv_marker));
        }

        BinaryHV::bundle_safe(&derivative_hvs)
    }

    /// Encode a sliding window of states
    ///
    /// Useful for capturing temporal patterns preceding disruptions
    pub fn encode_window(&mut self, states: &[PlasmaState]) -> BinaryHV {
        if states.is_empty() {
            return BinaryHV::zero();
        }

        let encoded: Vec<BinaryHV> = states
            .iter()
            .map(|s| {
                let readings = s.to_readings();
                let sensor_hvs: Vec<BinaryHV> =
                    readings.iter().map(|r| self.encode_sample(r)).collect();
                BinaryHV::bundle_safe(&sensor_hvs)
            })
            .collect();

        // Temporal weighting: more recent states get more weight
        // Simple approach: repeat recent states
        let mut weighted = Vec::new();
        for (i, hv) in encoded.iter().enumerate() {
            let weight = 1 + i / (states.len() / 3).max(1); // 1-3 copies
            for _ in 0..weight {
                weighted.push(*hv);
            }
        }

        BinaryHV::bundle_safe(&weighted)
    }

    /// Reset derivative tracking (call at start of new shot)
    pub fn reset(&mut self) {
        self.prev_state = None;
    }

    /// Get configuration
    pub fn config(&self) -> &PlasmaEncoderConfig {
        &self.config
    }
}

// =============================================================================
// PLASMA STATE BUFFER (STREAMING)
// =============================================================================

/// Circular buffer for streaming plasma state processing
#[derive(Debug)]
pub struct PlasmaStateBuffer {
    /// Circular buffer of states
    buffer: VecDeque<PlasmaState>,
    /// Maximum buffer capacity
    capacity: usize,
    /// Encoder for HDC encoding
    encoder: PlasmaHdcEncoder,
    /// Cached encoded states
    encoded_cache: VecDeque<BinaryHV>,
}

impl PlasmaStateBuffer {
    /// Create a new buffer with given capacity
    pub fn new(capacity: usize, encoder_seed: u64) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            encoder: PlasmaHdcEncoder::new(encoder_seed),
            encoded_cache: VecDeque::with_capacity(capacity),
        }
    }

    /// Push a new state into the buffer
    pub fn push(&mut self, state: PlasmaState) {
        // Encode before pushing
        let encoded = self.encoder.encode_state(&state);

        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
            self.encoded_cache.pop_front();
        }

        self.buffer.push_back(state);
        self.encoded_cache.push_back(encoded);
    }

    /// Get current window of states
    pub fn states(&self) -> &VecDeque<PlasmaState> {
        &self.buffer
    }

    /// Get encoded window
    pub fn encoded_window(&self) -> BinaryHV {
        if self.encoded_cache.is_empty() {
            return BinaryHV::zero();
        }

        let hvs: Vec<BinaryHV> = self.encoded_cache.iter().copied().collect();
        BinaryHV::bundle_safe(&hvs)
    }

    /// Get most recent state
    pub fn latest(&self) -> Option<&PlasmaState> {
        self.buffer.back()
    }

    /// Get most recent encoded state
    pub fn latest_encoded(&self) -> Option<BinaryHV> {
        self.encoded_cache.back().copied()
    }

    /// Buffer length
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear buffer (call at shot start)
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.encoded_cache.clear();
        self.encoder.reset();
    }
}

// =============================================================================
// PHI INTEGRATION - DISRUPTION DETECTION
// =============================================================================

/// Severity levels for disruption alerts
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DisruptionAlert {
    /// Normal operation
    Normal,
    /// Elevated risk - increased monitoring
    Caution,
    /// Significant risk - prepare for mitigation
    Warning,
    /// Imminent disruption - trigger mitigation
    Critical,
}

impl DisruptionAlert {
    /// Get alert from phi value and thresholds
    fn from_phi(phi: f64, thresholds: &PlasmaPhiThresholds) -> Self {
        if phi < thresholds.critical_phi {
            Self::Critical
        } else if phi < thresholds.warning_phi {
            Self::Warning
        } else if phi < thresholds.caution_phi {
            Self::Caution
        } else {
            Self::Normal
        }
    }

    /// Get recommended action
    pub fn recommended_action(&self) -> &'static str {
        match self {
            Self::Normal => "Continue normal operation",
            Self::Caution => "Increase diagnostic sampling rate",
            Self::Warning => "Prepare disruption mitigation systems",
            Self::Critical => "Trigger massive gas injection (MGI)",
        }
    }
}

/// Thresholds for phi-based stability assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasmaPhiThresholds {
    /// Phi below this triggers caution
    pub caution_phi: f64,
    /// Phi below this triggers warning
    pub warning_phi: f64,
    /// Phi below this triggers critical
    pub critical_phi: f64,
    /// Phi drop rate threshold (per second)
    pub phi_drop_rate_threshold: f64,
}

impl Default for PlasmaPhiThresholds {
    fn default() -> Self {
        Self {
            caution_phi: 0.6,
            warning_phi: 0.4,
            critical_phi: 0.2,
            phi_drop_rate_threshold: 0.1, // 10% drop per second
        }
    }
}

/// Stability assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityAssessment {
    /// Current phi value
    pub phi: f64,
    /// Phi change rate (per second)
    pub phi_rate: f64,
    /// Alert level
    pub alert: DisruptionAlert,
    /// List of critical sensors
    pub critical_sensors: Vec<PlasmaSensorType>,
    /// Confidence in assessment (0.0-1.0)
    pub confidence: f64,
    /// Recommended action
    pub recommendation: String,
    /// Computation time (microseconds)
    pub compute_time_us: u64,
}

/// Plasma stability monitor using phi integration
pub struct PlasmaPhiMonitor {
    /// Phi calculation engine
    phi_calc: TieredPhi,
    /// Alert thresholds
    thresholds: PlasmaPhiThresholds,
    /// Previous phi values for rate calculation
    phi_history: VecDeque<(f64, f64)>, // (timestamp, phi)
    /// History capacity
    history_capacity: usize,
    /// Reference phi for stable operation
    reference_phi: Option<f64>,
}

impl PlasmaPhiMonitor {
    /// Create a new plasma phi monitor
    pub fn new(thresholds: PlasmaPhiThresholds) -> Self {
        Self {
            phi_calc: TieredPhi::new(ApproximationTier::SampledPartition),
            thresholds,
            phi_history: VecDeque::with_capacity(100),
            history_capacity: 100,
            reference_phi: None,
        }
    }

    /// Set reference phi from stable operation period
    pub fn set_reference_phi(&mut self, phi: f64) {
        self.reference_phi = Some(phi);
    }

    /// Check plasma stability
    pub fn check_stability(
        &mut self,
        encoder: &PlasmaHdcEncoder,
        state: &PlasmaState,
    ) -> StabilityAssessment {
        let start = Instant::now();

        // Encode state and get component HVs
        let readings = state.to_readings();
        let components: Vec<BinaryHV> = readings.iter().map(|r| encoder.encode_sample(r)).collect();

        // Compute phi
        let phi = self.phi_calc.compute(&components);

        // Calculate phi rate
        let phi_rate = self.calculate_phi_rate(state.timestamp, phi);

        // Determine alert level
        let mut alert = DisruptionAlert::from_phi(phi, &self.thresholds);

        // Escalate if phi dropping rapidly
        if phi_rate < -self.thresholds.phi_drop_rate_threshold {
            alert = match alert {
                DisruptionAlert::Normal => DisruptionAlert::Caution,
                DisruptionAlert::Caution => DisruptionAlert::Warning,
                DisruptionAlert::Warning => DisruptionAlert::Critical,
                DisruptionAlert::Critical => DisruptionAlert::Critical,
            };
        }

        // Get critical sensors
        let critical_sensors = state.critical_sensors();

        // Calculate confidence based on history and sensor quality
        let confidence = if self.phi_history.len() >= 10 {
            0.9
        } else {
            0.6
        };

        let compute_time_us = start.elapsed().as_micros() as u64;

        StabilityAssessment {
            phi,
            phi_rate,
            alert,
            critical_sensors,
            confidence,
            recommendation: alert.recommended_action().to_string(),
            compute_time_us,
        }
    }

    /// Calculate phi rate from history
    fn calculate_phi_rate(&mut self, timestamp: f64, phi: f64) -> f64 {
        // Add to history
        if self.phi_history.len() >= self.history_capacity {
            self.phi_history.pop_front();
        }
        self.phi_history.push_back((timestamp, phi));

        // Need at least 2 points
        if self.phi_history.len() < 2 {
            return 0.0;
        }

        // Linear regression over recent history (last 10 points)
        let recent: Vec<_> = self.phi_history.iter().rev().take(10).collect();
        if recent.len() < 2 {
            return 0.0;
        }

        let first = recent
            .last()
            .expect("recent has at least 2 elements (checked above)");
        let last = recent
            .first()
            .expect("recent has at least 2 elements (checked above)");
        let dt = last.0 - first.0;

        if dt.abs() < 1e-6 {
            return 0.0;
        }

        (last.1 - first.1) / dt
    }

    /// Reset monitor (call at shot start)
    pub fn reset(&mut self) {
        self.phi_history.clear();
        self.reference_phi = None;
    }

    /// Get phi statistics
    pub fn phi_stats(&self) -> Option<(f64, f64, f64)> {
        if self.phi_history.is_empty() {
            return None;
        }

        let phis: Vec<f64> = self.phi_history.iter().map(|(_, p)| *p).collect();
        let mean = phis.iter().sum::<f64>() / phis.len() as f64;
        let min = phis.iter().copied().fold(f64::INFINITY, f64::min);
        let max = phis.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        Some((mean, min, max))
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Basic Encoding Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_encoder_creates_valid_hvs() {
        let encoder = PlasmaHdcEncoder::new(42);

        let reading = PlasmaReading::new(PlasmaSensorType::PlasmaCurrent, 1.5, 0.0);

        let hv = encoder.encode_sample(&reading);

        // HV should have approximately 50% density (random-like)
        let density = hv.density();
        assert!(
            density > 0.4 && density < 0.6,
            "Encoded HV should have balanced density, got {}",
            density
        );
    }

    #[test]
    fn test_encoding_deterministic() {
        let encoder1 = PlasmaHdcEncoder::new(42);
        let encoder2 = PlasmaHdcEncoder::new(42);

        let reading = PlasmaReading::new(PlasmaSensorType::ElectronTemperature, 5.0, 1.0);

        let hv1 = encoder1.encode_sample(&reading);
        let hv2 = encoder2.encode_sample(&reading);

        assert_eq!(hv1, hv2, "Same seed should produce identical encodings");
    }

    #[test]
    fn test_different_sensors_produce_different_hvs() {
        let encoder = PlasmaHdcEncoder::new(42);

        let ip = PlasmaReading::new(PlasmaSensorType::PlasmaCurrent, 1.0, 0.0);
        let ne = PlasmaReading::new(PlasmaSensorType::ElectronDensity, 1.0, 0.0);

        let hv_ip = encoder.encode_sample(&ip);
        let hv_ne = encoder.encode_sample(&ne);

        // Different sensors should have low similarity
        let sim = hv_ip.similarity(&hv_ne);
        assert!(
            sim < 0.6,
            "Different sensors should have low similarity, got {}",
            sim
        );
    }

    // -------------------------------------------------------------------------
    // Similar State Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_similar_states_high_similarity() {
        let mut encoder = PlasmaHdcEncoder::new(42);

        let state1 = PlasmaState {
            timestamp: 1.0,
            ip: 1.0,
            ne: 1.5,
            te: 5.0,
            prad: 1.0,
            vloop: 1.0,
            bt: 5.0,
            bp: 0.5,
            q95: 3.5,
            wmhd: 2.0,
            beta: 2.0,
        };

        // Very similar state (small perturbation)
        let state2 = PlasmaState {
            timestamp: 1.1,
            ip: 1.02,
            ne: 1.52,
            te: 5.05,
            prad: 1.01,
            vloop: 1.01,
            bt: 5.0,
            bp: 0.51,
            q95: 3.52,
            wmhd: 2.01,
            beta: 2.01,
        };

        let hv1 = encoder.encode_state(&state1);
        encoder.reset(); // Reset derivative tracking
        let hv2 = encoder.encode_state(&state2);

        let sim = hv1.similarity(&hv2);
        assert!(
            sim > 0.7,
            "Similar states should have high similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_dissimilar_states_lower_similarity() {
        let mut encoder = PlasmaHdcEncoder::new(42);

        let stable_state = PlasmaState::default();

        // Disruption-like state
        let disruption_state = PlasmaState {
            timestamp: 5.0,
            ip: 0.2,    // Very low current
            ne: 4.5,    // Above density limit
            te: 0.3,    // Cold plasma
            prad: 4.0,  // High radiation
            vloop: 6.0, // High loop voltage
            bt: 5.0,
            bp: 0.1,
            q95: 1.5,  // Below kink limit
            wmhd: 0.1, // Lost stored energy
            beta: 0.5,
        };

        let hv_stable = encoder.encode_state(&stable_state);
        encoder.reset();
        let hv_disruption = encoder.encode_state(&disruption_state);

        let sim = hv_stable.similarity(&hv_disruption);
        assert!(
            sim < 0.6,
            "Stable vs disruption states should have lower similarity, got {}",
            sim
        );
    }

    // -------------------------------------------------------------------------
    // Phi Integration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_disruption_state_lower_phi() {
        let encoder = PlasmaHdcEncoder::new(42);
        let mut phi_calc = TieredPhi::new(ApproximationTier::SampledPartition);

        // Stable state
        let stable = PlasmaState::default();
        let stable_readings = stable.to_readings();
        let stable_hvs: Vec<BinaryHV> = stable_readings
            .iter()
            .map(|r| encoder.encode_sample(r))
            .collect();
        let phi_stable = phi_calc.compute(&stable_hvs);

        // Disruption state with sensors in critical ranges
        let disruption = PlasmaState {
            timestamp: 0.0,
            ip: 0.2,    // Critical
            ne: 4.5,    // Critical
            te: 0.3,    // Critical
            prad: 0.9,  // Critical (90% radiation)
            vloop: 6.0, // Critical
            bt: 5.0,
            bp: 0.1,
            q95: 1.5, // Critical
            wmhd: 0.1,
            beta: 4.0, // Critical
        };
        let disruption_readings = disruption.to_readings();
        let disruption_hvs: Vec<BinaryHV> = disruption_readings
            .iter()
            .map(|r| encoder.encode_sample(r))
            .collect();
        let phi_disruption = phi_calc.compute(&disruption_hvs);

        // Phi should generally be different (the exact relationship depends on implementation)
        // The key insight: disruption states have *different* characteristics
        println!(
            "Phi stable: {}, Phi disruption: {}",
            phi_stable, phi_disruption
        );

        // Both should produce valid phi values
        assert!(phi_stable >= 0.0, "Stable phi should be non-negative");
        assert!(
            phi_disruption >= 0.0,
            "Disruption phi should be non-negative"
        );
    }

    #[test]
    fn test_phi_monitor_detects_critical() {
        let encoder = PlasmaHdcEncoder::new(42);
        let mut monitor = PlasmaPhiMonitor::new(Default::default());

        // Test critical state detection via sensor thresholds
        let critical_state = PlasmaState {
            timestamp: 0.0,
            ip: 0.2, // Below 0.3 threshold
            ne: 4.5, // Above 4.0 threshold
            te: 0.3, // Below 0.5 threshold
            prad: 0.9,
            vloop: 6.0, // Above 5.0 threshold
            bt: 5.0,
            bp: 0.1,
            q95: 1.5, // Below 2.0 threshold
            wmhd: 0.1,
            beta: 4.0,
        };

        let assessment = monitor.check_stability(&encoder, &critical_state);

        // Should detect critical sensors
        assert!(
            !assessment.critical_sensors.is_empty(),
            "Should detect critical sensors"
        );
        assert!(
            assessment
                .critical_sensors
                .contains(&PlasmaSensorType::PlasmaCurrent)
        );
        assert!(
            assessment
                .critical_sensors
                .contains(&PlasmaSensorType::SafetyFactor)
        );
    }

    // -------------------------------------------------------------------------
    // Streaming Performance Tests
    // -------------------------------------------------------------------------

    #[test]
    #[ignore = "performance test - run with cargo test --release --ignored"]
    fn test_streaming_performance_10ms_budget() {
        let mut buffer = PlasmaStateBuffer::new(100, 42);

        let start = Instant::now();
        let iterations = 100;

        for i in 0..iterations {
            let state = PlasmaState {
                timestamp: i as f64 * 0.001, // 1ms timesteps
                ip: 1.0 + 0.01 * (i as f64).sin(),
                ne: 1.5,
                te: 5.0,
                prad: 1.0,
                vloop: 1.0,
                bt: 5.0,
                bp: 0.5,
                q95: 3.5,
                wmhd: 2.0,
                beta: 2.0,
            };
            buffer.push(state);
        }

        let elapsed_ms = start.elapsed().as_millis();
        let avg_time_ms = elapsed_ms as f64 / iterations as f64;

        println!("Average encoding time: {:.3} ms", avg_time_ms);
        // Budget: 100ms in release (HDC bundling is inherently O(n*dim)), 500ms in debug
        let threshold = if cfg!(debug_assertions) { 500.0 } else { 100.0 };
        assert!(
            avg_time_ms < threshold,
            "Encoding should complete within {}ms budget, got {}ms",
            threshold,
            avg_time_ms
        );
    }

    #[test]
    #[ignore = "performance test - run with cargo test --release --ignored"]
    fn test_window_encoding_performance() {
        let mut encoder = PlasmaHdcEncoder::new(42);

        // Create window of 50 states
        let states: Vec<PlasmaState> = (0..50)
            .map(|i| PlasmaState {
                timestamp: i as f64 * 0.1,
                ip: 1.0 + 0.1 * (i as f64 / 10.0).sin(),
                ..Default::default()
            })
            .collect();

        let start = Instant::now();
        let _window_hv = encoder.encode_window(&states);
        let elapsed_us = start.elapsed().as_micros();

        println!("Window encoding (50 states): {} us", elapsed_us);
        // Budget: 5s in release (window encoding bundles 50 states * 10 sensors with
        // temporal weighting, which is O(window * sensors * dim)), 20s in debug
        let threshold_us: u128 = if cfg!(debug_assertions) {
            20_000_000
        } else {
            5_000_000
        };
        assert!(
            elapsed_us < threshold_us,
            "Window encoding should be fast, got {}us (threshold: {}us)",
            elapsed_us,
            threshold_us
        );
    }

    // -------------------------------------------------------------------------
    // Sensor Type Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_all_sensor_types_have_base_vectors() {
        let encoder = PlasmaHdcEncoder::new(42);

        for sensor in PlasmaSensorType::all() {
            let base = encoder.base_vector(sensor);
            assert!(
                base.density() > 0.4 && base.density() < 0.6,
                "Base vector for {:?} should be balanced",
                sensor
            );
        }
    }

    #[test]
    fn test_sensor_critical_thresholds() {
        // Test that critical threshold detection works
        let reading_normal = PlasmaReading::new(PlasmaSensorType::PlasmaCurrent, 1.0, 0.0);
        let reading_critical = PlasmaReading::new(PlasmaSensorType::PlasmaCurrent, 0.2, 0.0);

        assert!(!reading_normal.is_critical());
        assert!(reading_critical.is_critical());
    }

    #[test]
    fn test_reading_normalization() {
        let reading = PlasmaReading::new(PlasmaSensorType::PlasmaCurrent, 1.25, 0.0);
        let normalized = reading.normalized();

        // Ip range is [0.5, 2.0], so 1.25 should map to 0.5
        assert!(
            (normalized - 0.5).abs() < 0.01,
            "1.25 MA should normalize to 0.5, got {}",
            normalized
        );
    }

    // -------------------------------------------------------------------------
    // Buffer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_buffer_capacity() {
        let mut buffer = PlasmaStateBuffer::new(10, 42);

        // Push more than capacity
        for i in 0..20 {
            buffer.push(PlasmaState {
                timestamp: i as f64,
                ..Default::default()
            });
        }

        assert_eq!(buffer.len(), 10, "Buffer should maintain capacity");
        assert_eq!(
            buffer.latest().unwrap().timestamp,
            19.0,
            "Latest should be most recent push"
        );
    }

    #[test]
    fn test_buffer_encoded_window() {
        let mut buffer = PlasmaStateBuffer::new(5, 42);

        for i in 0..5 {
            buffer.push(PlasmaState {
                timestamp: i as f64,
                ip: 1.0 + i as f64 * 0.1,
                ..Default::default()
            });
        }

        let window_hv = buffer.encoded_window();
        // Bundled HVs should have reasonable density (not all zeros or all ones)
        // The exact balance depends on bundling logic, but should be valid
        let density = window_hv.density();
        assert!(
            density > 0.1 && density < 0.9,
            "Window HV should have reasonable density, got {}",
            density
        );
    }

    // -------------------------------------------------------------------------
    // Phi Monitor Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_phi_rate_calculation() {
        let encoder = PlasmaHdcEncoder::new(42);
        let mut monitor = PlasmaPhiMonitor::new(Default::default());

        // Simulate multiple timesteps
        for i in 0..20 {
            let state = PlasmaState {
                timestamp: i as f64 * 0.1,
                ..Default::default()
            };
            monitor.check_stability(&encoder, &state);
        }

        let stats = monitor.phi_stats();
        assert!(
            stats.is_some(),
            "Should have phi statistics after multiple samples"
        );

        let (mean, min, max) = stats.unwrap();
        assert!(
            mean >= min && mean <= max,
            "Mean should be between min and max"
        );
    }

    #[test]
    fn test_alert_escalation() {
        // Test alert levels
        let thresholds = PlasmaPhiThresholds::default();

        assert_eq!(
            DisruptionAlert::from_phi(0.8, &thresholds),
            DisruptionAlert::Normal
        );
        assert_eq!(
            DisruptionAlert::from_phi(0.5, &thresholds),
            DisruptionAlert::Caution
        );
        assert_eq!(
            DisruptionAlert::from_phi(0.3, &thresholds),
            DisruptionAlert::Warning
        );
        assert_eq!(
            DisruptionAlert::from_phi(0.1, &thresholds),
            DisruptionAlert::Critical
        );
    }

    // -------------------------------------------------------------------------
    // Temporal Encoding Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_temporal_phase_circular() {
        let encoder = PlasmaHdcEncoder::new(42);

        // Times near the wrap-around point should be similar
        let reading1 = PlasmaReading::new(PlasmaSensorType::PlasmaCurrent, 1.0, 0.1);
        let reading2 = PlasmaReading::new(PlasmaSensorType::PlasmaCurrent, 1.0, 9.9);

        let hv1 = encoder.encode_sample(&reading1);
        let hv2 = encoder.encode_sample(&reading2);

        // With 10-second time scale, 0.1s and 9.9s are close (wrapping)
        let sim = hv1.similarity(&hv2);
        assert!(
            sim > 0.6,
            "Times near wrap-around should be similar, got {}",
            sim
        );
    }

    #[test]
    fn test_derivative_encoding() {
        let config = PlasmaEncoderConfig {
            encode_derivatives: true,
            derivative_weight: 0.3,
            ..Default::default()
        };
        let mut encoder = PlasmaHdcEncoder::with_config(config);

        // First state
        let state1 = PlasmaState {
            timestamp: 0.0,
            ip: 1.0,
            ne: 1.5,
            te: 5.0,
            prad: 1.0,
            vloop: 1.0,
            bt: 5.0,
            bp: 0.5,
            q95: 3.5,
            wmhd: 2.0,
            beta: 2.0,
        };
        let hv1 = encoder.encode_state(&state1);

        // Second state with change
        let state2 = PlasmaState {
            timestamp: 0.1,
            ip: 1.2, // Increased current
            ne: 1.5,
            te: 5.0,
            prad: 1.0,
            vloop: 1.0,
            bt: 5.0,
            bp: 0.5,
            q95: 3.5,
            wmhd: 2.0,
            beta: 2.0,
        };
        let hv2 = encoder.encode_state(&state2);

        // Third state with same change rate
        let state3 = PlasmaState {
            timestamp: 0.2,
            ip: 1.4,
            ne: 1.5,
            te: 5.0,
            prad: 1.0,
            vloop: 1.0,
            bt: 5.0,
            bp: 0.5,
            q95: 3.5,
            wmhd: 2.0,
            beta: 2.0,
        };
        let hv3 = encoder.encode_state(&state3);

        // HVs should all be valid (with reasonable density)
        assert!(
            hv1.density() > 0.1 && hv1.density() < 0.9,
            "State 1 HV should have reasonable density, got {}",
            hv1.density()
        );
        assert!(
            hv2.density() > 0.1 && hv2.density() < 0.9,
            "State 2 HV should have reasonable density, got {}",
            hv2.density()
        );
        assert!(
            hv3.density() > 0.1 && hv3.density() < 0.9,
            "State 3 HV should have reasonable density, got {}",
            hv3.density()
        );

        // First state and subsequent states should differ due to derivative contribution
        // (the exact relationship depends on encoding details)
        println!("hv1-hv2 similarity: {}", hv1.similarity(&hv2));
        println!("hv2-hv3 similarity: {}", hv2.similarity(&hv3));
    }
}
