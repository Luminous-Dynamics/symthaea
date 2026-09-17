// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Matched substrate-neutral controls for physical-reservoir research.
//!
//! These controls deliberately avoid material-specific labels. They exist to make
//! exotic-substrate comparisons falsifiable: a gold, moire, photonic, or quantum
//! candidate should not receive credit for capabilities already reproduced by a
//! generic nonlinear, hysteretic, or recurrent system with a comparable budget.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::collections::BTreeMap;

use symthaea_physical_cognition::{
    BackendIdentity, ContractError, ControlFrame, EnergyBoundary, EvidenceLevel,
    ExecutionBoundary, ObservationFrame, ObservationProvenance, PhysicalBackend,
    PhysicalObservation, ResourceEnvelope,
};

/// Canonical output keys shared by every matched control.
pub const STANDARD_READOUT_KEYS: [&str; 4] = [
    "state_mean",
    "state_variance",
    "activity_fraction",
    "output",
];

/// Structural/resource budget for a control backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlBudget {
    /// Number of logical response units evaluated per frame.
    pub units: usize,
    /// Number of persistent scalar state variables retained across frames.
    pub persistent_state_scalars: usize,
    /// Number of recurrent graph edges, if any.
    pub recurrent_edges: usize,
    /// Number of trainable parameters inside the backend itself.
    pub trainable_parameters: usize,
    /// Number of scalar readouts emitted per frame.
    pub readout_scalars: usize,
}

impl ControlBudget {
    /// Validate basic budget invariants.
    pub fn validate(&self) -> Result<(), ControlError> {
        if self.units == 0 {
            return Err(ControlError::InvalidConfig("units must be > 0"));
        }
        if self.readout_scalars != STANDARD_READOUT_KEYS.len() {
            return Err(ControlError::InvalidConfig(
                "readout budget must equal the standard four-scalar interface",
            ));
        }
        Ok(())
    }
}

/// Control family used when selecting matched baselines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ControlFamily {
    /// Stateless nonlinear feature map.
    MemorylessNonlinear,
    /// Independent fading-memory hysteretic units.
    HystereticEnsemble,
    /// Generic recurrent graph dynamics.
    RecurrentGraph,
}

/// Shared metadata required from every matched control.
pub trait MatchedControl: PhysicalBackend {
    /// Control family.
    fn family(&self) -> ControlFamily;

    /// Explicit state/topology/readout budget.
    fn budget(&self) -> ControlBudget;
}

/// Configuration for the memoryless nonlinear baseline.
#[derive(Debug, Clone, PartialEq)]
pub struct MemorylessConfig {
    /// Number of independent nonlinear response units.
    pub units: usize,
    /// Input gain applied before the nonlinearity.
    pub input_gain: f64,
    /// Deterministic per-unit gain spread around `input_gain`.
    pub gain_spread: f64,
    /// Activity threshold used only for the standardized activity readout.
    pub activity_threshold: f64,
}

impl Default for MemorylessConfig {
    fn default() -> Self {
        Self {
            units: 128,
            input_gain: 1.0,
            gain_spread: 0.25,
            activity_threshold: 0.25,
        }
    }
}

impl MemorylessConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<(), ControlError> {
        validate_units(self.units)?;
        validate_nonnegative_finite("input_gain", self.input_gain)?;
        validate_nonnegative_finite("gain_spread", self.gain_spread)?;
        validate_nonnegative_finite("activity_threshold", self.activity_threshold)?;
        Ok(())
    }
}

/// Stateless nonlinear feature-map baseline.
#[derive(Debug, Clone)]
pub struct MemorylessNonlinear {
    config: MemorylessConfig,
    gains: Vec<f64>,
}

impl MemorylessNonlinear {
    /// Construct a deterministic seeded baseline.
    pub fn new(config: MemorylessConfig, seed: u64) -> Result<Self, ControlError> {
        config.validate()?;
        let mut rng = XorShift64::new(seed);
        let mut gains = Vec::with_capacity(config.units);
        for _ in 0..config.units {
            gains.push(
                (config.input_gain + symmetric_unit(&mut rng) * config.gain_spread).max(0.0),
            );
        }
        Ok(Self { config, gains })
    }

    fn evaluate_drive(&self, drive: f64) -> StandardReadout {
        let states: Vec<f64> = self.gains.iter().map(|gain| (drive * gain).tanh()).collect();
        StandardReadout::from_states(&states, self.config.activity_threshold)
    }
}

impl MatchedControl for MemorylessNonlinear {
    fn family(&self) -> ControlFamily {
        ControlFamily::MemorylessNonlinear
    }

    fn budget(&self) -> ControlBudget {
        ControlBudget {
            units: self.config.units,
            persistent_state_scalars: 0,
            recurrent_edges: 0,
            trainable_parameters: 0,
            readout_scalars: STANDARD_READOUT_KEYS.len(),
        }
    }
}

impl PhysicalBackend for MemorylessNonlinear {
    fn identity(&self) -> BackendIdentity {
        BackendIdentity::new(
            "control:memoryless-nonlinear",
            "v1",
            "matched-memoryless-bank",
        )
        .expect("static identity is canonical")
    }

    fn reset(&mut self) -> Result<(), ContractError> {
        Ok(())
    }

    fn evaluate(&mut self, input: &ControlFrame) -> Result<PhysicalObservation, ContractError> {
        let drive = required_drive(input)?;
        let readout = self.evaluate_drive(drive);
        build_observation(self.identity(), input.sequence, readout)
    }
}

/// Configuration for the fading-memory hysteretic ensemble.
#[derive(Debug, Clone, PartialEq)]
pub struct HystereticConfig {
    /// Number of persistent state units.
    pub units: usize,
    /// Relaxed baseline state.
    pub baseline: f64,
    /// Absolute switching threshold.
    pub switching_threshold: f64,
    /// Per-unit seeded threshold spread.
    pub threshold_spread: f64,
    /// State increment per unit overdrive.
    pub switching_gain: f64,
    /// Fractional relaxation toward baseline per frame.
    pub relaxation: f64,
    /// Standardized activity threshold.
    pub activity_threshold: f64,
}

impl Default for HystereticConfig {
    fn default() -> Self {
        Self {
            units: 128,
            baseline: 0.0,
            switching_threshold: 0.35,
            threshold_spread: 0.08,
            switching_gain: 0.18,
            relaxation: 0.025,
            activity_threshold: 0.1,
        }
    }
}

impl HystereticConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<(), ControlError> {
        validate_units(self.units)?;
        validate_finite("baseline", self.baseline)?;
        validate_nonnegative_finite("switching_threshold", self.switching_threshold)?;
        validate_nonnegative_finite("threshold_spread", self.threshold_spread)?;
        validate_nonnegative_finite("switching_gain", self.switching_gain)?;
        validate_unit_interval("relaxation", self.relaxation)?;
        validate_nonnegative_finite("activity_threshold", self.activity_threshold)?;
        Ok(())
    }
}

/// Generic independent hysteretic ensemble with fading memory.
#[derive(Debug, Clone)]
pub struct HystereticEnsemble {
    config: HystereticConfig,
    initial_seed: u64,
    thresholds: Vec<f64>,
    states: Vec<f64>,
}

impl HystereticEnsemble {
    /// Construct a seeded ensemble.
    pub fn new(config: HystereticConfig, seed: u64) -> Result<Self, ControlError> {
        config.validate()?;
        let mut rng = XorShift64::new(seed);
        let thresholds = (0..config.units)
            .map(|_| {
                (config.switching_threshold + symmetric_unit(&mut rng) * config.threshold_spread)
                    .max(0.0)
            })
            .collect();
        let states = vec![config.baseline; config.units];
        Ok(Self {
            config,
            initial_seed: seed,
            thresholds,
            states,
        })
    }

    /// Evolve one frame and return standardized diagnostics.
    pub fn step(&mut self, drive: f64) -> Result<StandardReadout, ControlError> {
        if !drive.is_finite() {
            return Err(ControlError::NonFiniteDrive);
        }
        let magnitude = drive.abs();
        for (state, threshold) in self.states.iter_mut().zip(&self.thresholds) {
            *state += self.config.relaxation * (self.config.baseline - *state);
            if magnitude > *threshold {
                *state += drive.signum() * self.config.switching_gain * (magnitude - *threshold);
            }
            *state = state.clamp(-1.0, 1.0);
        }
        Ok(StandardReadout::from_states(
            &self.states,
            self.config.activity_threshold,
        ))
    }
}

impl MatchedControl for HystereticEnsemble {
    fn family(&self) -> ControlFamily {
        ControlFamily::HystereticEnsemble
    }

    fn budget(&self) -> ControlBudget {
        ControlBudget {
            units: self.config.units,
            persistent_state_scalars: self.config.units,
            recurrent_edges: 0,
            trainable_parameters: 0,
            readout_scalars: STANDARD_READOUT_KEYS.len(),
        }
    }
}

impl PhysicalBackend for HystereticEnsemble {
    fn identity(&self) -> BackendIdentity {
        BackendIdentity::new(
            "control:hysteretic-ensemble",
            "v1",
            "matched-fading-hysteresis",
        )
        .expect("static identity is canonical")
    }

    fn reset(&mut self) -> Result<(), ContractError> {
        *self = Self::new(self.config.clone(), self.initial_seed).map_err(control_to_contract)?;
        Ok(())
    }

    fn evaluate(&mut self, input: &ControlFrame) -> Result<PhysicalObservation, ContractError> {
        let drive = required_drive(input)?;
        let readout = self.step(drive).map_err(control_to_contract)?;
        build_observation(self.identity(), input.sequence, readout)
    }
}

/// Configuration for the generic recurrent graph baseline.
#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentGraphConfig {
    /// Number of persistent units.
    pub units: usize,
    /// Number of seeded shortcut attempts per unit in addition to the ring.
    pub shortcuts_per_unit: usize,
    /// Input coupling strength.
    pub input_gain: f64,
    /// Recurrent-neighbor coupling strength.
    pub recurrent_gain: f64,
    /// Leaky-update factor in `[0,1]`.
    pub leak: f64,
    /// Seeded static input-weight spread.
    pub input_weight_spread: f64,
    /// Standardized activity threshold.
    pub activity_threshold: f64,
}

impl Default for RecurrentGraphConfig {
    fn default() -> Self {
        Self {
            units: 128,
            shortcuts_per_unit: 2,
            input_gain: 0.7,
            recurrent_gain: 0.45,
            leak: 0.35,
            input_weight_spread: 0.3,
            activity_threshold: 0.1,
        }
    }
}

impl RecurrentGraphConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<(), ControlError> {
        validate_units(self.units)?;
        validate_nonnegative_finite("input_gain", self.input_gain)?;
        validate_nonnegative_finite("recurrent_gain", self.recurrent_gain)?;
        validate_unit_interval("leak", self.leak)?;
        validate_nonnegative_finite("input_weight_spread", self.input_weight_spread)?;
        validate_nonnegative_finite("activity_threshold", self.activity_threshold)?;
        Ok(())
    }
}

/// Generic recurrent graph reservoir with no material-specific physics.
#[derive(Debug, Clone)]
pub struct RecurrentGraphReservoir {
    config: RecurrentGraphConfig,
    initial_seed: u64,
    adjacency: Vec<Vec<usize>>,
    input_weights: Vec<f64>,
    states: Vec<f64>,
    edge_count: usize,
    topology_fingerprint: u64,
}

impl RecurrentGraphReservoir {
    /// Construct a seeded symmetric ring-plus-shortcut graph.
    pub fn new(config: RecurrentGraphConfig, seed: u64) -> Result<Self, ControlError> {
        config.validate()?;
        let mut rng = XorShift64::new(seed);
        let mut adjacency = vec![Vec::new(); config.units];
        for i in 0..config.units {
            add_undirected_edge(&mut adjacency, i, (i + 1) % config.units);
        }
        if config.units > 2 {
            for i in 0..config.units {
                for _ in 0..config.shortcuts_per_unit {
                    let mut target = (rng.next_u64() as usize) % config.units;
                    if target == i {
                        target = (target + 1) % config.units;
                    }
                    add_undirected_edge(&mut adjacency, i, target);
                }
            }
        }
        for neighbors in &mut adjacency {
            neighbors.sort_unstable();
            neighbors.dedup();
        }
        let edge_count = adjacency.iter().map(Vec::len).sum::<usize>() / 2;
        let topology_fingerprint = fingerprint_adjacency(&adjacency);
        let input_weights = (0..config.units)
            .map(|_| 1.0 + symmetric_unit(&mut rng) * config.input_weight_spread)
            .collect();
        let states = vec![0.0; config.units];
        Ok(Self {
            config,
            initial_seed: seed,
            adjacency,
            input_weights,
            states,
            edge_count,
            topology_fingerprint,
        })
    }

    /// Stable fingerprint of the seeded graph topology.
    pub fn topology_fingerprint(&self) -> u64 {
        self.topology_fingerprint
    }

    /// Number of undirected recurrent edges.
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    /// Read-only adjacency used by topology controls.
    pub fn adjacency(&self) -> &[Vec<usize>] {
        &self.adjacency
    }

    /// Evolve one recurrent frame.
    pub fn step(&mut self, drive: f64) -> Result<StandardReadout, ControlError> {
        if !drive.is_finite() {
            return Err(ControlError::NonFiniteDrive);
        }
        let previous = self.states.clone();
        for i in 0..self.states.len() {
            let recurrent = if self.adjacency[i].is_empty() {
                0.0
            } else {
                self.adjacency[i]
                    .iter()
                    .map(|&neighbor| previous[neighbor])
                    .sum::<f64>()
                    / self.adjacency[i].len() as f64
            };
            let target = (self.config.input_gain * drive * self.input_weights[i]
                + self.config.recurrent_gain * recurrent)
                .tanh();
            self.states[i] = previous[i] + self.config.leak * (target - previous[i]);
        }
        Ok(StandardReadout::from_states(
            &self.states,
            self.config.activity_threshold,
        ))
    }
}

impl MatchedControl for RecurrentGraphReservoir {
    fn family(&self) -> ControlFamily {
        ControlFamily::RecurrentGraph
    }

    fn budget(&self) -> ControlBudget {
        ControlBudget {
            units: self.config.units,
            persistent_state_scalars: self.config.units,
            recurrent_edges: self.edge_count,
            trainable_parameters: 0,
            readout_scalars: STANDARD_READOUT_KEYS.len(),
        }
    }
}

impl PhysicalBackend for RecurrentGraphReservoir {
    fn identity(&self) -> BackendIdentity {
        BackendIdentity::new(
            "control:recurrent-graph",
            "v1",
            "matched-generic-reservoir",
        )
        .expect("static identity is canonical")
    }

    fn reset(&mut self) -> Result<(), ContractError> {
        *self = Self::new(self.config.clone(), self.initial_seed).map_err(control_to_contract)?;
        Ok(())
    }

    fn evaluate(&mut self, input: &ControlFrame) -> Result<PhysicalObservation, ContractError> {
        let drive = required_drive(input)?;
        let readout = self.step(drive).map_err(control_to_contract)?;
        build_observation(self.identity(), input.sequence, readout)
    }
}

/// Standard four-scalar readout shared across controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StandardReadout {
    /// Mean latent state.
    pub state_mean: f64,
    /// Variance of latent state.
    pub state_variance: f64,
    /// Fraction of units above the configured activity threshold in magnitude.
    pub activity_fraction: f64,
    /// Canonical scalar output. V1 uses the mean latent state.
    pub output: f64,
}

impl StandardReadout {
    fn from_states(states: &[f64], activity_threshold: f64) -> Self {
        let count = states.len() as f64;
        let mean = states.iter().copied().sum::<f64>() / count;
        let variance = states
            .iter()
            .map(|state| {
                let delta = *state - mean;
                delta * delta
            })
            .sum::<f64>()
            / count;
        let active = states
            .iter()
            .filter(|state| state.abs() > activity_threshold)
            .count() as f64
            / count;
        Self {
            state_mean: mean,
            state_variance: variance,
            activity_fraction: active,
            output: mean,
        }
    }

    fn into_map(self) -> BTreeMap<String, f64> {
        BTreeMap::from([
            ("state_mean".to_string(), self.state_mean),
            ("state_variance".to_string(), self.state_variance),
            ("activity_fraction".to_string(), self.activity_fraction),
            ("output".to_string(), self.output),
        ])
    }
}

/// Errors specific to matched-control construction/evolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlError {
    /// Structural configuration error.
    InvalidConfig(&'static str),
    /// Named floating-point parameter was NaN or infinite.
    InvalidConfigValue(&'static str),
    /// Drive input was NaN or infinite.
    NonFiniteDrive,
}

impl std::fmt::Display for ControlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid matched-control config: {message}"),
            Self::InvalidConfigValue(name) => write!(f, "non-finite matched-control value: {name}"),
            Self::NonFiniteDrive => write!(f, "drive_voltage must be finite"),
        }
    }
}

impl std::error::Error for ControlError {}

fn required_drive(input: &ControlFrame) -> Result<f64, ContractError> {
    input.validate()?;
    input
        .scalars
        .get("drive_voltage")
        .copied()
        .ok_or_else(|| ContractError::InvalidToken {
            field: "required control",
            value: "missing drive_voltage".to_string(),
        })
}

fn build_observation(
    backend: BackendIdentity,
    sequence: u64,
    readout: StandardReadout,
) -> Result<PhysicalObservation, ContractError> {
    let result = PhysicalObservation {
        observation: ObservationFrame {
            sequence,
            scalars: readout.into_map(),
            samples: Some(1),
        },
        resources: ResourceEnvelope {
            latency_ns: None,
            energy_joules: None,
            energy_boundary: EnergyBoundary::Unmeasured,
            readout_count: Some(STANDARD_READOUT_KEYS.len() as u64),
        },
        provenance: ObservationProvenance {
            backend,
            execution: ExecutionBoundary::Simulation,
            evidence: EvidenceLevel::Simulated,
            caveats: vec![
                "Substrate-neutral matched control; no physical-device claim.".to_string(),
                "No superiority claim is implied by this control result.".to_string(),
            ],
        },
    };
    result.validate()?;
    Ok(result)
}

fn validate_units(units: usize) -> Result<(), ControlError> {
    if units == 0 {
        Err(ControlError::InvalidConfig("units must be > 0"))
    } else {
        Ok(())
    }
}

fn validate_finite(name: &'static str, value: f64) -> Result<(), ControlError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ControlError::InvalidConfigValue(name))
    }
}

fn validate_nonnegative_finite(name: &'static str, value: f64) -> Result<(), ControlError> {
    validate_finite(name, value)?;
    if value < 0.0 {
        Err(ControlError::InvalidConfig("parameter must be non-negative"))
    } else {
        Ok(())
    }
}

fn validate_unit_interval(name: &'static str, value: f64) -> Result<(), ControlError> {
    validate_finite(name, value)?;
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ControlError::InvalidConfig("parameter must be in [0,1]"))
    }
}

fn control_to_contract(error: ControlError) -> ContractError {
    ContractError::InvalidToken {
        field: "matched control",
        value: error.to_string(),
    }
}

fn add_undirected_edge(adjacency: &mut [Vec<usize>], a: usize, b: usize) {
    if a == b || adjacency[a].contains(&b) {
        return;
    }
    adjacency[a].push(b);
    adjacency[b].push(a);
}

fn fingerprint_adjacency(adjacency: &[Vec<usize>]) -> u64 {
    // FNV-1a over canonical undirected edges. This is an identity convenience,
    // not a cryptographic commitment; PHYS-002 owns experiment commitments.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for (a, neighbors) in adjacency.iter().enumerate() {
        for &b in neighbors {
            if a < b {
                for byte in (a as u64)
                    .to_le_bytes()
                    .into_iter()
                    .chain((b as u64).to_le_bytes())
                {
                    hash ^= byte as u64;
                    hash = hash.wrapping_mul(0x1000_0000_01b3);
                }
            }
        }
    }
    hash
}

#[derive(Debug, Clone, Copy)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn unit_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * SCALE
    }
}

fn symmetric_unit(rng: &mut XorShift64) -> f64 {
    2.0 * rng.unit_f64() - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(sequence: u64, drive: f64) -> ControlFrame {
        let mut frame = ControlFrame::new(sequence);
        frame.scalars.insert("drive_voltage".to_string(), drive);
        frame
    }

    fn keys(observation: &PhysicalObservation) -> Vec<&str> {
        observation
            .observation
            .scalars
            .keys()
            .map(String::as_str)
            .collect()
    }

    #[test]
    fn all_controls_emit_identical_standard_interface() {
        let mut memoryless = MemorylessNonlinear::new(MemorylessConfig::default(), 7).unwrap();
        let mut hysteretic = HystereticEnsemble::new(HystereticConfig::default(), 7).unwrap();
        let mut recurrent =
            RecurrentGraphReservoir::new(RecurrentGraphConfig::default(), 7).unwrap();
        let expected = vec![
            "activity_fraction",
            "output",
            "state_mean",
            "state_variance",
        ];
        assert_eq!(keys(&memoryless.evaluate(&frame(0, 0.5)).unwrap()), expected);
        assert_eq!(keys(&hysteretic.evaluate(&frame(0, 0.5)).unwrap()), expected);
        assert_eq!(keys(&recurrent.evaluate(&frame(0, 0.5)).unwrap()), expected);
    }

    #[test]
    fn memoryless_control_has_no_history_dependence() {
        let mut control = MemorylessNonlinear::new(MemorylessConfig::default(), 11).unwrap();
        let before = control.evaluate(&frame(0, 0.2)).unwrap();
        let _ = control.evaluate(&frame(1, 0.9)).unwrap();
        let after = control.evaluate(&frame(2, 0.2)).unwrap();
        assert_eq!(before.observation.scalars, after.observation.scalars);
        assert_eq!(control.budget().persistent_state_scalars, 0);
    }

    #[test]
    fn hysteretic_control_retains_then_relaxes_state() {
        let config = HystereticConfig {
            threshold_spread: 0.0,
            ..HystereticConfig::default()
        };
        let mut control = HystereticEnsemble::new(config, 13).unwrap();
        let excited = control.step(0.9).unwrap().state_mean;
        let first_relax = control.step(0.0).unwrap().state_mean;
        let second_relax = control.step(0.0).unwrap().state_mean;
        assert!(excited.abs() > first_relax.abs());
        assert!(first_relax.abs() > second_relax.abs());
        assert!(second_relax.abs() > 0.0);
    }

    #[test]
    fn recurrent_graph_is_symmetric_without_self_edges() {
        let control = RecurrentGraphReservoir::new(RecurrentGraphConfig::default(), 17).unwrap();
        for (a, neighbors) in control.adjacency().iter().enumerate() {
            assert!(!neighbors.contains(&a));
            for &b in neighbors {
                assert!(control.adjacency()[b].contains(&a));
            }
        }
    }

    #[test]
    fn recurrent_coupling_changes_post_excitation_trajectory() {
        let coupled_config = RecurrentGraphConfig {
            input_weight_spread: 0.0,
            recurrent_gain: 0.8,
            ..RecurrentGraphConfig::default()
        };
        let uncoupled_config = RecurrentGraphConfig {
            recurrent_gain: 0.0,
            ..coupled_config.clone()
        };
        let mut coupled = RecurrentGraphReservoir::new(coupled_config, 19).unwrap();
        let mut uncoupled = RecurrentGraphReservoir::new(uncoupled_config, 19).unwrap();
        let _ = coupled.step(0.9).unwrap();
        let _ = uncoupled.step(0.9).unwrap();
        let coupled_after = coupled.step(0.0).unwrap().state_mean;
        let uncoupled_after = uncoupled.step(0.0).unwrap().state_mean;
        assert_ne!(coupled_after.to_bits(), uncoupled_after.to_bits());
    }

    #[test]
    fn seeded_graph_replays_topology_and_trajectory() {
        let config = RecurrentGraphConfig::default();
        let mut a = RecurrentGraphReservoir::new(config.clone(), 23).unwrap();
        let mut b = RecurrentGraphReservoir::new(config, 23).unwrap();
        assert_eq!(a.topology_fingerprint(), b.topology_fingerprint());
        assert_eq!(a.edge_count(), b.edge_count());
        for (sequence, drive) in [0.2, 0.8, -0.3, 0.0].into_iter().enumerate() {
            assert_eq!(
                a.evaluate(&frame(sequence as u64, drive)).unwrap(),
                b.evaluate(&frame(sequence as u64, drive)).unwrap()
            );
        }
        let first_fingerprint = a.topology_fingerprint();
        a.reset().unwrap();
        assert_eq!(first_fingerprint, a.topology_fingerprint());
    }

    #[test]
    fn control_identities_are_substrate_neutral() {
        let memoryless = MemorylessNonlinear::new(MemorylessConfig::default(), 1).unwrap();
        let hysteretic = HystereticEnsemble::new(HystereticConfig::default(), 1).unwrap();
        let recurrent = RecurrentGraphReservoir::new(RecurrentGraphConfig::default(), 1).unwrap();
        for identity in [
            memoryless.identity(),
            hysteretic.identity(),
            recurrent.identity(),
        ] {
            assert!(identity.family.starts_with("control:"));
            assert!(!identity.family.contains("gold"));
            assert!(!identity.family.contains("quantum"));
            assert!(!identity.family.contains("moire"));
        }
    }

    #[test]
    fn budgets_make_hidden_capacity_visible() {
        let memoryless = MemorylessNonlinear::new(MemorylessConfig::default(), 2).unwrap();
        let hysteretic = HystereticEnsemble::new(HystereticConfig::default(), 2).unwrap();
        let recurrent = RecurrentGraphReservoir::new(RecurrentGraphConfig::default(), 2).unwrap();
        assert_eq!(memoryless.budget().readout_scalars, 4);
        assert_eq!(hysteretic.budget().readout_scalars, 4);
        assert_eq!(recurrent.budget().readout_scalars, 4);
        assert_eq!(memoryless.budget().persistent_state_scalars, 0);
        assert_eq!(hysteretic.budget().persistent_state_scalars, 128);
        assert_eq!(recurrent.budget().persistent_state_scalars, 128);
        assert!(recurrent.budget().recurrent_edges >= 128);
    }
}
