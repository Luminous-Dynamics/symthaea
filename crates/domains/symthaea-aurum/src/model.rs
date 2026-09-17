// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Gold nanojunction physical-reservoir research models.
//!
//! This crate models a **candidate** nonlinear reservoir inspired by
//! cluster-assembled gold nanojunction networks. It is a research simulator,
//! not a calibrated device model, and it does not claim gold-specific
//! advantage, physical-hardware validation, or consciousness relevance.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::collections::{BTreeMap, BTreeSet};
use symthaea_physical_cognition::{
    BackendIdentity, ContractError, ControlFrame, EnergyBoundary, EvidenceLevel,
    ExecutionBoundary, ObservationFrame, ObservationProvenance, PhysicalBackend,
    PhysicalObservation, ResourceEnvelope,
};

/// Deterministic effective network topology used by the simulator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AurumTopology {
    /// Undirected ring lattice connecting `radius` neighbors on each side.
    Ring {
        /// Number of lattice neighbors on each side.
        radius: usize,
    },
    /// Ring lattice plus a seeded number of long-range shortcuts per node.
    SmallWorld {
        /// Number of lattice neighbors on each side.
        radius: usize,
        /// Attempted long-range shortcuts added per node before deduplication.
        shortcuts_per_node: usize,
    },
}

impl Default for AurumTopology {
    fn default() -> Self {
        Self::SmallWorld {
            radius: 2,
            shortcuts_per_node: 1,
        }
    }
}

/// Parameters for the gold nanojunction reservoir model.
#[derive(Debug, Clone, PartialEq)]
pub struct AurumConfig {
    /// Number of effective junction sites in the network model.
    pub junctions: usize,
    /// Seeded network topology.
    pub topology: AurumTopology,
    /// Baseline normalized conductance for a relaxed junction.
    pub baseline_conductance: f64,
    /// Maximum normalized conductance.
    pub max_conductance: f64,
    /// Absolute local drive threshold above which switching is promoted.
    pub switching_threshold: f64,
    /// Conductance increment per unit over-threshold local drive.
    pub switching_gain: f64,
    /// Fractional relaxation toward baseline on each frame.
    pub relaxation: f64,
    /// Seeded disorder amplitude applied to each junction threshold.
    pub threshold_disorder: f64,
    /// Seeded stochastic perturbation amplitude applied to each update.
    pub stochasticity: f64,
    /// Linear external-input coupling into the network.
    pub input_coupling: f64,
    /// Coupling from neighboring excess conductance into local drive.
    pub recurrent_coupling: f64,
}

impl Default for AurumConfig {
    fn default() -> Self {
        Self {
            junctions: 128,
            topology: AurumTopology::default(),
            baseline_conductance: 0.15,
            max_conductance: 1.0,
            switching_threshold: 0.35,
            switching_gain: 0.18,
            relaxation: 0.025,
            threshold_disorder: 0.08,
            stochasticity: 0.01,
            input_coupling: 1.0,
            recurrent_coupling: 0.25,
        }
    }
}

impl AurumConfig {
    /// Validate model parameters before construction.
    pub fn validate(&self) -> Result<(), AurumError> {
        if self.junctions < 3 {
            return Err(AurumError::InvalidConfig("junctions must be >= 3"));
        }
        match self.topology {
            AurumTopology::Ring { radius }
            | AurumTopology::SmallWorld { radius, .. } => {
                if radius == 0 || radius * 2 >= self.junctions {
                    return Err(AurumError::InvalidConfig(
                        "topology radius must satisfy 0 < 2*radius < junctions",
                    ));
                }
            }
        }
        if let AurumTopology::SmallWorld {
            radius,
            shortcuts_per_node,
        } = self.topology
        {
            if radius
                .saturating_mul(2)
                .saturating_add(shortcuts_per_node)
                >= self.junctions
            {
                return Err(AurumError::InvalidConfig(
                    "small-world shortcuts leave no eligible non-neighbor target",
                ));
            }
        }
        for (name, value) in [
            ("baseline_conductance", self.baseline_conductance),
            ("max_conductance", self.max_conductance),
            ("switching_threshold", self.switching_threshold),
            ("switching_gain", self.switching_gain),
            ("relaxation", self.relaxation),
            ("threshold_disorder", self.threshold_disorder),
            ("stochasticity", self.stochasticity),
            ("input_coupling", self.input_coupling),
            ("recurrent_coupling", self.recurrent_coupling),
        ] {
            if !value.is_finite() {
                return Err(AurumError::InvalidConfigValue(name));
            }
        }
        if !(0.0..=1.0).contains(&self.baseline_conductance)
            || !(0.0..=1.0).contains(&self.max_conductance)
            || self.max_conductance < self.baseline_conductance
        {
            return Err(AurumError::InvalidConfig(
                "conductance bounds must satisfy 0 <= baseline <= max <= 1",
            ));
        }
        if self.switching_threshold < 0.0
            || self.switching_gain < 0.0
            || !(0.0..=1.0).contains(&self.relaxation)
            || self.threshold_disorder < 0.0
            || self.stochasticity < 0.0
            || self.input_coupling < 0.0
            || self.recurrent_coupling < 0.0
        {
            return Err(AurumError::InvalidConfig(
                "threshold/gain/disorder/stochasticity/couplings must be non-negative; relaxation must be in [0,1]",
            ));
        }
        Ok(())
    }
}

/// Internal state of one effective nanojunction site.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JunctionState {
    /// Normalized conductance in `[baseline, max]`.
    pub conductance: f64,
    /// Fixed per-junction threshold offset sampled at construction.
    pub threshold_offset: f64,
}

/// Aggregate diagnostics emitted after one update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AurumDiagnostics {
    /// Mean normalized conductance across all junctions.
    pub mean_conductance: f64,
    /// Fraction of junctions that crossed their local switching threshold.
    pub switched_fraction: f64,
    /// Normalized aggregate current proxy: `external_drive * mean_conductance`.
    pub output_current: f64,
    /// Conductance variance across the reservoir.
    pub conductance_variance: f64,
    /// Mean recurrent field derived from neighboring excess conductance.
    pub mean_recurrent_field: f64,
    /// Static number of undirected network edges.
    pub edge_count: usize,
}

/// Deterministic seeded simulator for a gold nanojunction reservoir candidate.
#[derive(Debug, Clone)]
pub struct AurumReservoir {
    config: AurumConfig,
    initial_seed: u64,
    rng: XorShift64,
    junctions: Vec<JunctionState>,
    neighbors: Vec<Vec<usize>>,
}

impl AurumReservoir {
    /// Construct a simulator with deterministic seeded disorder and topology.
    pub fn new(config: AurumConfig, seed: u64) -> Result<Self, AurumError> {
        config.validate()?;
        let mut rng = XorShift64::new(seed);
        let neighbors = build_topology(&config, &mut rng)?;
        let mut junctions = Vec::with_capacity(config.junctions);
        for _ in 0..config.junctions {
            let offset = symmetric_unit(&mut rng) * config.threshold_disorder;
            junctions.push(JunctionState {
                conductance: config.baseline_conductance,
                threshold_offset: offset,
            });
        }
        Ok(Self {
            config,
            initial_seed: seed,
            rng,
            junctions,
            neighbors,
        })
    }

    /// Read-only model configuration.
    pub fn config(&self) -> &AurumConfig {
        &self.config
    }

    /// Read-only junction state for diagnostics and frozen experiment receipts.
    pub fn junction_states(&self) -> &[JunctionState] {
        &self.junctions
    }

    /// Read-only sorted adjacency list. Each edge is represented in both directions.
    pub fn neighbors(&self) -> &[Vec<usize>] {
        &self.neighbors
    }

    /// Number of undirected network edges.
    pub fn edge_count(&self) -> usize {
        self.neighbors.iter().map(Vec::len).sum::<usize>() / 2
    }

    /// Deterministic topology fingerprint useful for receipts and control matching.
    ///
    /// This is an FNV-1a-style convenience fingerprint, not a cryptographic digest.
    pub fn topology_fingerprint(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for (source, targets) in self.neighbors.iter().enumerate() {
            for &target in targets {
                if source < target {
                    for value in [source as u64, target as u64] {
                        for byte in value.to_le_bytes() {
                            hash ^= u64::from(byte);
                            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
                        }
                    }
                }
            }
        }
        hash
    }

    /// Evolve the effective junction network for one scalar drive frame.
    pub fn step(&mut self, drive_voltage: f64) -> Result<AurumDiagnostics, AurumError> {
        if !drive_voltage.is_finite() {
            return Err(AurumError::NonFiniteDrive);
        }

        let drive = drive_voltage * self.config.input_coupling;
        let external_magnitude = drive.abs();
        let previous: Vec<f64> = self
            .junctions
            .iter()
            .map(|junction| junction.conductance)
            .collect();
        let conductance_span = (self.config.max_conductance - self.config.baseline_conductance)
            .max(f64::EPSILON);
        let recurrent_fields: Vec<f64> = self
            .neighbors
            .iter()
            .map(|targets| {
                let neighbor_mean = targets.iter().map(|&index| previous[index]).sum::<f64>()
                    / targets.len() as f64;
                ((neighbor_mean - self.config.baseline_conductance) / conductance_span)
                    .clamp(0.0, 1.0)
                    * self.config.recurrent_coupling
            })
            .collect();
        let mut switched = 0usize;

        for (index, junction) in self.junctions.iter_mut().enumerate() {
            let threshold = (self.config.switching_threshold + junction.threshold_offset).max(0.0);
            let local_magnitude = external_magnitude + recurrent_fields[index];

            // Relaxation supplies fading memory even when the drive is below threshold.
            junction.conductance += self.config.relaxation
                * (self.config.baseline_conductance - junction.conductance);

            if local_magnitude > threshold {
                switched += 1;
                let overdrive = local_magnitude - threshold;
                junction.conductance += self.config.switching_gain * overdrive;
            }

            if self.config.stochasticity > 0.0 {
                junction.conductance += symmetric_unit(&mut self.rng) * self.config.stochasticity;
            }

            junction.conductance = junction
                .conductance
                .clamp(self.config.baseline_conductance, self.config.max_conductance);
        }

        let count = self.junctions.len() as f64;
        let mean = self
            .junctions
            .iter()
            .map(|junction| junction.conductance)
            .sum::<f64>()
            / count;
        let variance = self
            .junctions
            .iter()
            .map(|junction| {
                let delta = junction.conductance - mean;
                delta * delta
            })
            .sum::<f64>()
            / count;
        let mean_recurrent_field = recurrent_fields.iter().sum::<f64>() / count;

        Ok(AurumDiagnostics {
            mean_conductance: mean,
            switched_fraction: switched as f64 / count,
            output_current: drive * mean,
            conductance_variance: variance,
            mean_recurrent_field,
            edge_count: self.edge_count(),
        })
    }

    fn reset_internal(&mut self) -> Result<(), AurumError> {
        *self = Self::new(self.config.clone(), self.initial_seed)?;
        Ok(())
    }
}

impl PhysicalBackend for AurumReservoir {
    fn identity(&self) -> BackendIdentity {
        BackendIdentity::new(
            "physical:gold-nanojunction",
            "v1",
            "aurum-topological-network-sim",
        )
        .expect("static backend identity is canonical")
    }

    fn reset(&mut self) -> Result<(), ContractError> {
        self.reset_internal()
            .map_err(|_| ContractError::InvalidToken {
                field: "aurum reset",
                value: "invalid stored configuration".to_string(),
            })
    }

    fn evaluate(&mut self, input: &ControlFrame) -> Result<PhysicalObservation, ContractError> {
        input.validate()?;
        let drive = input.scalars.get("drive_voltage").copied().ok_or_else(|| {
            ContractError::InvalidToken {
                field: "required control",
                value: "missing drive_voltage".to_string(),
            }
        })?;

        let diagnostics = self.step(drive).map_err(|error| ContractError::InvalidToken {
            field: "aurum evaluation",
            value: error.to_string(),
        })?;

        let mut scalars = BTreeMap::new();
        scalars.insert("mean_conductance".to_string(), diagnostics.mean_conductance);
        scalars.insert("switched_fraction".to_string(), diagnostics.switched_fraction);
        scalars.insert("output_current".to_string(), diagnostics.output_current);
        scalars.insert(
            "conductance_variance".to_string(),
            diagnostics.conductance_variance,
        );
        scalars.insert(
            "mean_recurrent_field".to_string(),
            diagnostics.mean_recurrent_field,
        );
        scalars.insert("edge_count".to_string(), diagnostics.edge_count as f64);

        let result = PhysicalObservation {
            observation: ObservationFrame {
                sequence: input.sequence,
                scalars,
                samples: Some(1),
            },
            resources: ResourceEnvelope {
                latency_ns: None,
                energy_joules: None,
                energy_boundary: EnergyBoundary::Unmeasured,
                readout_count: Some(6),
            },
            provenance: ObservationProvenance {
                backend: self.identity(),
                execution: ExecutionBoundary::Simulation,
                evidence: EvidenceLevel::Simulated,
                caveats: vec![
                    "Effective topological nanojunction-network model; not fitted to a specific fabricated device."
                        .to_string(),
                    "Topology/coupling are research parameters, not measured Au device parameters."
                        .to_string(),
                    "No gold-specific computational advantage is claimed.".to_string(),
                ],
            },
        };
        result.validate()?;
        Ok(result)
    }
}

/// Errors specific to the AURUM simulator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AurumError {
    /// Structural configuration error.
    InvalidConfig(&'static str),
    /// A named configuration value was NaN or infinite.
    InvalidConfigValue(&'static str),
    /// Failed to construct a valid bounded network topology.
    TopologyConstruction,
    /// The applied drive was NaN or infinite.
    NonFiniteDrive,
}

impl std::fmt::Display for AurumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid AURUM config: {message}"),
            Self::InvalidConfigValue(name) => write!(f, "non-finite AURUM config value: {name}"),
            Self::TopologyConstruction => write!(f, "unable to construct requested AURUM topology"),
            Self::NonFiniteDrive => write!(f, "drive voltage must be finite"),
        }
    }
}

impl std::error::Error for AurumError {}

fn build_topology(
    config: &AurumConfig,
    rng: &mut XorShift64,
) -> Result<Vec<Vec<usize>>, AurumError> {
    let mut adjacency = vec![BTreeSet::new(); config.junctions];
    let radius = match config.topology {
        AurumTopology::Ring { radius } | AurumTopology::SmallWorld { radius, .. } => radius,
    };

    for source in 0..config.junctions {
        for offset in 1..=radius {
            let forward = (source + offset) % config.junctions;
            let backward = (source + config.junctions - offset) % config.junctions;
            add_undirected_edge(&mut adjacency, source, forward);
            add_undirected_edge(&mut adjacency, source, backward);
        }
    }

    if let AurumTopology::SmallWorld {
        shortcuts_per_node,
        ..
    } = config.topology
    {
        for source in 0..config.junctions {
            let target_degree = adjacency[source].len() + shortcuts_per_node;
            let max_attempts = config.junctions.saturating_mul(16).max(32);
            let mut attempts = 0usize;
            while adjacency[source].len() < target_degree && attempts < max_attempts {
                let target = (rng.next_u64() as usize) % config.junctions;
                attempts += 1;
                if target != source && !adjacency[source].contains(&target) {
                    add_undirected_edge(&mut adjacency, source, target);
                }
            }
            if adjacency[source].len() < target_degree {
                return Err(AurumError::TopologyConstruction);
            }
        }
    }

    Ok(adjacency
        .into_iter()
        .map(|targets| targets.into_iter().collect())
        .collect())
}

fn add_undirected_edge(adjacency: &mut [BTreeSet<usize>], a: usize, b: usize) {
    if a != b {
        adjacency[a].insert(b);
        adjacency[b].insert(a);
    }
}

#[derive(Debug, Clone, Copy)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        // xorshift cannot use an all-zero state.
        Self {
            state: if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed },
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

    fn drive_frame(sequence: u64, drive: f64) -> ControlFrame {
        let mut frame = ControlFrame::new(sequence);
        frame.scalars.insert("drive_voltage".to_string(), drive);
        frame
    }

    #[test]
    fn seeded_runs_are_deterministic_in_state_and_topology() {
        let config = AurumConfig::default();
        let mut a = AurumReservoir::new(config.clone(), 42).unwrap();
        let mut b = AurumReservoir::new(config, 42).unwrap();
        assert_eq!(a.neighbors(), b.neighbors());
        assert_eq!(a.topology_fingerprint(), b.topology_fingerprint());
        for drive in [0.1, 0.7, -0.2, 0.8, 0.0] {
            assert_eq!(a.step(drive).unwrap(), b.step(drive).unwrap());
        }
    }

    #[test]
    fn topology_is_symmetric_and_has_no_self_edges() {
        let reservoir = AurumReservoir::new(AurumConfig::default(), 5).unwrap();
        for (source, targets) in reservoir.neighbors().iter().enumerate() {
            assert!(!targets.contains(&source));
            for &target in targets {
                assert!(reservoir.neighbors()[target].contains(&source));
            }
        }
        assert!(reservoir.edge_count() > 0);
    }

    #[test]
    fn supra_threshold_drive_leaves_fading_memory() {
        let config = AurumConfig {
            stochasticity: 0.0,
            threshold_disorder: 0.0,
            recurrent_coupling: 0.0,
            ..AurumConfig::default()
        };
        let baseline = config.baseline_conductance;
        let mut reservoir = AurumReservoir::new(config, 7).unwrap();
        let excited = reservoir.step(0.9).unwrap().mean_conductance;
        let after_one_relaxation = reservoir.step(0.0).unwrap().mean_conductance;
        assert!(excited > baseline);
        assert!(after_one_relaxation > baseline);
        assert!(after_one_relaxation < excited);
    }

    #[test]
    fn recurrent_topology_changes_zero_drive_relaxation_after_excitation() {
        let base = AurumConfig {
            stochasticity: 0.0,
            threshold_disorder: 0.04,
            recurrent_coupling: 0.0,
            ..AurumConfig::default()
        };
        let mut uncoupled = AurumReservoir::new(base.clone(), 17).unwrap();
        let mut coupled = AurumReservoir::new(
            AurumConfig {
                recurrent_coupling: 0.8,
                ..base
            },
            17,
        )
        .unwrap();
        let _ = uncoupled.step(0.9).unwrap();
        let _ = coupled.step(0.9).unwrap();
        let uncoupled_after = uncoupled.step(0.0).unwrap();
        let coupled_after = coupled.step(0.0).unwrap();
        assert!(coupled_after.mean_recurrent_field > uncoupled_after.mean_recurrent_field);
        assert!(coupled_after.mean_conductance >= uncoupled_after.mean_conductance);
    }

    #[test]
    fn ring_and_small_world_have_distinct_topology_fingerprints() {
        let ring = AurumReservoir::new(
            AurumConfig {
                topology: AurumTopology::Ring { radius: 2 },
                ..AurumConfig::default()
            },
            31,
        )
        .unwrap();
        let small_world = AurumReservoir::new(AurumConfig::default(), 31).unwrap();
        assert_ne!(ring.topology_fingerprint(), small_world.topology_fingerprint());
        assert!(small_world.edge_count() > ring.edge_count());
    }

    #[test]
    fn physical_backend_is_simulation_only() {
        let mut reservoir = AurumReservoir::new(AurumConfig::default(), 9).unwrap();
        let observation = reservoir.evaluate(&drive_frame(3, 0.6)).unwrap();
        assert_eq!(observation.provenance.execution, ExecutionBoundary::Simulation);
        assert_eq!(observation.provenance.evidence, EvidenceLevel::Simulated);
        assert_eq!(observation.resources.energy_boundary, EnergyBoundary::Unmeasured);
    }

    #[test]
    fn reset_replays_seeded_topology_and_trajectory() {
        let mut reservoir = AurumReservoir::new(AurumConfig::default(), 1234).unwrap();
        let fingerprint = reservoir.topology_fingerprint();
        let first = reservoir.evaluate(&drive_frame(0, 0.77)).unwrap();
        let _ = reservoir.evaluate(&drive_frame(1, -0.41)).unwrap();
        PhysicalBackend::reset(&mut reservoir).unwrap();
        let replay = reservoir.evaluate(&drive_frame(0, 0.77)).unwrap();
        assert_eq!(fingerprint, reservoir.topology_fingerprint());
        assert_eq!(first, replay);
    }

    #[test]
    fn stronger_drive_switches_at_least_as_many_junctions_without_noise() {
        let config = AurumConfig {
            stochasticity: 0.0,
            recurrent_coupling: 0.0,
            ..AurumConfig::default()
        };
        let mut low = AurumReservoir::new(config.clone(), 99).unwrap();
        let mut high = AurumReservoir::new(config, 99).unwrap();
        let low_switched = low.step(0.36).unwrap().switched_fraction;
        let high_switched = high.step(0.8).unwrap().switched_fraction;
        assert!(high_switched >= low_switched);
    }
}
