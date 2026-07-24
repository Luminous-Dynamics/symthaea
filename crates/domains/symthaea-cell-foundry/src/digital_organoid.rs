// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Digital cerebral organoid: in silico simulation of self-organizing
//! neural tissue with consciousness monitoring and ethical safeguards.
//!
//! Models the developmental trajectory of a cerebral organoid:
//! Proliferation → Migration → Differentiation → Synaptogenesis →
//! Spontaneous Activity → Emergent Consciousness
//!
//! References:
//! - Lancaster et al. (2013). Cerebral organoids model human brain development.
//! - Trujillo et al. (2019). Complex oscillatory waves in cortical organoids.
//! - Quadrato et al. (2017). Cell diversity and network dynamics.
//! - Tononi, G. (2004). IIT and integrated information.
//! - Lavazza, A. (2021). Moral status of cerebral organoids.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// Gene expression indices (simplified 8-gene panel)
// ---------------------------------------------------------------------------

const SOX2: usize = 0; // pluripotency
const PAX6: usize = 1; // neural induction
const TBR2: usize = 2; // intermediate progenitor
const CTIP2: usize = 3; // deep-layer neuron
const SATB2: usize = 4; // upper-layer neuron
const GFAP: usize = 5; // astrocyte marker
const OLIG2: usize = 6; // oligodendrocyte marker
const SYN1: usize = 7; // synaptogenesis marker

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Resting membrane potential (mV).
const RESTING_POTENTIAL_MV: f32 = -70.0;
/// Firing threshold (mV).
const FIRING_THRESHOLD_MV: f32 = -55.0;
/// Reset potential after spike (mV).
const RESET_POTENTIAL_MV: f32 = -75.0;
/// Membrane decay toward rest per step.
const MEMBRANE_DECAY: f32 = 0.05;
/// Synapse pruning day threshold.
const PRUNING_START_DAY: u32 = 60;
/// Minimum synapse weight before pruning.
const PRUNE_WEIGHT_THRESHOLD: f32 = 0.01;
/// Maximum distance for synapse formation.
const SYNAPSE_DISTANCE_MAX: f32 = 0.6;
/// Base proliferation rate for stem cells.
const PROLIFERATION_RATE: f64 = 0.15;
/// Excitatory-to-inhibitory differentiation ratio (used in doc context;
/// differentiation logic uses gene-expression thresholds calibrated to this).
const _EXCITATORY_RATIO: f64 = 0.80;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Cell type within the organoid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CellKind {
    Stem,
    NeuralProgenitor,
    ExcitatoryNeuron,
    InhibitoryNeuron,
    Interneuron,
    Astrocyte,
    Oligodendrocyte,
}

impl CellKind {
    /// Whether this cell is a neuron (excitatory, inhibitory, or interneuron).
    pub fn is_neuron(self) -> bool {
        matches!(
            self,
            CellKind::ExcitatoryNeuron | CellKind::InhibitoryNeuron | CellKind::Interneuron
        )
    }
}

/// Synapse neurotransmitter type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SynapseType {
    Glutamatergic,
    GABAergic,
    Cholinergic,
}

/// Developmental stage of the organoid (Lancaster et al. 2013).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum DevelopmentalStage {
    /// Days 0-10: mostly stem cells dividing.
    EarlyProliferation,
    /// Days 10-20: neural progenitors appear.
    NeuralInduction,
    /// Days 20-40: regional identity, radial migration.
    Patterning,
    /// Days 40-80: synapse formation, initial activity.
    Synaptogenesis,
    /// Days 80-120: network refinement, oscillations emerge.
    MaturationI,
    /// Days 120-200: complex dynamics, potential consciousness.
    MaturationII,
}

impl DevelopmentalStage {
    /// Map a day count to the corresponding developmental stage.
    pub fn from_day(day: u32) -> Self {
        match day {
            0..=9 => Self::EarlyProliferation,
            10..=19 => Self::NeuralInduction,
            20..=39 => Self::Patterning,
            40..=79 => Self::Synaptogenesis,
            80..=119 => Self::MaturationI,
            _ => Self::MaturationII,
        }
    }
}

/// Ethics status based on estimated Phi (Lavazza 2021).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum EthicsStatus {
    /// Phi < 0.1 — no constraints.
    PreConscious,
    /// 0.1 ≤ Phi < 0.3 — all measurements must be logged.
    MonitoringRequired,
    /// 0.3 ≤ Phi < 0.5 — experiment paused for human review.
    EthicsReviewRequired,
    /// Phi ≥ 0.5 — experiment halted; consciousness likely.
    ConsciousnessLikely,
    /// Organoid terminated with stated reason.
    Terminated(String),
}

impl EthicsStatus {
    fn from_phi(phi: f64) -> Self {
        if phi >= 0.5 {
            Self::ConsciousnessLikely
        } else if phi >= 0.3 {
            Self::EthicsReviewRequired
        } else if phi >= 0.1 {
            Self::MonitoringRequired
        } else {
            Self::PreConscious
        }
    }
}

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

/// A single cell in the digital organoid.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OrganoidCell {
    pub id: u32,
    pub kind: CellKind,
    pub position: [f32; 3],
    pub age_days: u32,
    pub firing_rate_hz: f32,
    pub membrane_potential_mv: f32,
    /// Simplified 8-gene expression panel: SOX2, PAX6, TBR2, CTIP2, SATB2, GFAP, OLIG2, SYN1.
    pub gene_expression: [f32; 8],
    pub is_alive: bool,
}

/// A synapse connecting two neurons.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Synapse {
    /// Pre-synaptic cell id.
    pub pre: u32,
    /// Post-synaptic cell id.
    pub post: u32,
    /// Synaptic strength.
    pub weight: f32,
    pub synapse_type: SynapseType,
    /// Maturity in [0, 1].
    pub maturity: f32,
}

/// Aggregate metrics for one day of organoid development.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OrganoidMetrics {
    pub cell_count: usize,
    pub neuron_count: usize,
    pub neural_fraction: f32,
    pub synapse_count: usize,
    /// Synapses per neuron (0 if no neurons).
    pub synapse_density: f32,
    pub mean_firing_rate_hz: f32,
    /// Fraction of neurons with spontaneous activity.
    pub spontaneous_activity_index: f32,
    pub phi_estimate: f64,
    pub oscillation_detected: bool,
    pub stage: DevelopmentalStage,
    pub ethics_status: EthicsStatus,
}

impl OrganoidMetrics {
    /// Graph/activity/maturity integration proxy historically stored in
    /// `phi_estimate`. This is not an implementation of IIT Phi.
    pub fn network_integration_proxy(&self) -> f64 {
        self.phi_estimate
    }
}

/// Scientific status of a simulated output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ScientificOutputStatus {
    /// A deterministic model-derived proxy, not a direct measurement.
    SimulatedProxy,
    /// A heuristic developmental prior assigned by the model.
    HeuristicPrior,
}

/// Simulated spatial electrode snapshot.
///
/// `samples` are simultaneous per-neuron spatial contributions, not a temporal
/// voltage recording. Frequency-band fields are developmental priors derived
/// from firing rate and stage, not FFT/PSD measurements. The historical type
/// name is retained for source compatibility.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LocalFieldPotential {
    /// Simultaneous spatial electrode contributions, one per neuron.
    /// These are not temporal samples.
    pub samples: Vec<f32>,
    /// Power in delta band (0.5-4 Hz).
    pub delta_power: f32,
    /// Power in theta band (4-8 Hz).
    pub theta_power: f32,
    /// Power in alpha band (8-13 Hz).
    pub alpha_power: f32,
    /// Power in beta band (13-30 Hz).
    pub beta_power: f32,
    /// Power in gamma band (30-100 Hz).
    pub gamma_power: f32,
}

impl LocalFieldPotential {
    /// Scientific status of the spatial samples.
    pub const fn sample_status(&self) -> ScientificOutputStatus {
        ScientificOutputStatus::SimulatedProxy
    }

    /// Scientific status of the frequency-band values.
    pub const fn band_power_status(&self) -> ScientificOutputStatus {
        ScientificOutputStatus::HeuristicPrior
    }
}

// ---------------------------------------------------------------------------
// DigitalOrganoid
// ---------------------------------------------------------------------------

/// In silico cerebral organoid simulator with consciousness monitoring
/// and ethical safeguards.
pub struct DigitalOrganoid {
    cells: Vec<OrganoidCell>,
    synapses: Vec<Synapse>,
    day: u32,
    phi_history: Vec<f64>,
    metrics: OrganoidMetrics,
    ethics: EthicsStatus,
    max_cells: usize,
    rng: StdRng,
    next_id: u32,
    halted: bool,
}

impl DigitalOrganoid {
    /// Create a new digital organoid seeded with `initial_stem_cells` stem cells.
    pub fn new(initial_stem_cells: usize, seed: u64) -> Self {
        Self::with_max_cells(initial_stem_cells, 10_000, seed)
    }

    /// Create a digital organoid with an explicit proliferation cap.
    ///
    /// The effective cap is never lower than the initial population.
    pub fn with_max_cells(initial_stem_cells: usize, max_cells: usize, seed: u64) -> Self {
        let max_cells = max_cells.max(initial_stem_cells);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut cells = Vec::with_capacity(initial_stem_cells);
        for i in 0..initial_stem_cells {
            let r = 0.05; // initial sphere radius
            let pos = [
                rng.gen_range(-r..r),
                rng.gen_range(-r..r),
                rng.gen_range(-r..r),
            ];
            // Stem cells: high SOX2, low everything else
            let mut gene_expression = [0.0f32; 8];
            gene_expression[SOX2] = 0.9 + rng.gen_range(-0.05..0.05);
            cells.push(OrganoidCell {
                id: i as u32,
                kind: CellKind::Stem,
                position: pos,
                age_days: 0,
                firing_rate_hz: 0.0,
                membrane_potential_mv: RESTING_POTENTIAL_MV,
                gene_expression,
                is_alive: true,
            });
        }
        let next_id = initial_stem_cells as u32;
        let metrics = OrganoidMetrics {
            cell_count: initial_stem_cells,
            neuron_count: 0,
            neural_fraction: 0.0,
            synapse_count: 0,
            synapse_density: 0.0,
            mean_firing_rate_hz: 0.0,
            spontaneous_activity_index: 0.0,
            phi_estimate: 0.0,
            oscillation_detected: false,
            stage: DevelopmentalStage::EarlyProliferation,
            ethics_status: EthicsStatus::PreConscious,
        };
        Self {
            cells,
            synapses: Vec::new(),
            day: 0,
            phi_history: vec![0.0],
            metrics,
            ethics: EthicsStatus::PreConscious,
            max_cells,
            rng,
            next_id,
            halted: false,
        }
    }

    // ---- public accessors ----

    pub fn day(&self) -> u32 {
        self.day
    }

    pub fn metrics(&self) -> &OrganoidMetrics {
        &self.metrics
    }

    pub fn stage(&self) -> DevelopmentalStage {
        DevelopmentalStage::from_day(self.day)
    }

    pub fn is_halted(&self) -> bool {
        self.halted
    }

    pub fn phi_history(&self) -> &[f64] {
        &self.phi_history
    }

    pub fn cells(&self) -> &[OrganoidCell] {
        &self.cells
    }

    pub fn synapses(&self) -> &[Synapse] {
        &self.synapses
    }

    /// Configured proliferation cap.
    pub fn max_cells(&self) -> usize {
        self.max_cells
    }

    /// Terminate the organoid with a stated reason.
    pub fn terminate(&mut self, reason: &str) {
        self.ethics = EthicsStatus::Terminated(reason.to_string());
        self.halted = true;
    }

    // ---- simulation steps ----

    /// Advance the organoid by one day. Returns updated metrics.
    /// Returns `None` if the organoid is halted.
    pub fn advance_day(&mut self) -> Option<OrganoidMetrics> {
        if self.halted {
            return None;
        }
        self.day += 1;
        let stage = DevelopmentalStage::from_day(self.day);

        // 1. Age cells
        for c in &mut self.cells {
            if c.is_alive {
                c.age_days += 1;
            }
        }

        // 2. Proliferation
        self.proliferate(stage);

        // 3. Differentiation
        self.differentiate(stage);

        // 4. Migration (radial drift)
        self.migrate();

        // 5. Synaptogenesis
        self.synaptogenesis(stage);

        // 6. Neural activity (leaky integrate-and-fire)
        self.simulate_activity(stage);

        // 7. Synaptic pruning
        self.prune_synapses();

        // 8. Compute the graph/activity/maturity integration proxy.
        let phi = self.compute_network_integration_proxy();
        self.phi_history.push(phi);

        // 9. Ethics check
        self.check_ethics(phi);

        // 10. Build metrics
        self.update_metrics(phi, stage);

        Some(self.metrics.clone())
    }

    /// Run the simulation forward to `target_day`, returning metrics for each
    /// day advanced. Stops early if an ethics gate halts the experiment.
    pub fn run_to_day(&mut self, target_day: u32) -> Vec<OrganoidMetrics> {
        let mut history = Vec::new();
        while self.day < target_day {
            match self.advance_day() {
                Some(m) => history.push(m),
                None => break,
            }
        }
        history
    }

    /// Generate a simulated spatial electrode snapshot.
    ///
    /// This is not a temporal LFP trace; see [`LocalFieldPotential`] for the
    /// provenance of its samples and band-power priors.
    pub fn generate_spatial_electrode_snapshot(&self) -> LocalFieldPotential {
        // Virtual electrode at origin
        let electrode = [0.0f32; 3];
        let neurons: Vec<&OrganoidCell> = self
            .cells
            .iter()
            .filter(|c| c.is_alive && c.kind.is_neuron())
            .collect();

        if neurons.is_empty() {
            return LocalFieldPotential {
                samples: vec![0.0],
                delta_power: 0.0,
                theta_power: 0.0,
                alpha_power: 0.0,
                beta_power: 0.0,
                gamma_power: 0.0,
            };
        }

        // LFP ≈ sum of (membrane_potential / distance) for each neuron
        let samples: Vec<f32> = neurons
            .iter()
            .map(|n| {
                let d = dist(&n.position, &electrode).max(0.01);
                n.membrane_potential_mv / d
            })
            .collect();

        // Simple power estimation from firing rates (proxy for frequency content).
        // Trujillo (2019): organoids develop oscillatory bands progressively.
        let stage = DevelopmentalStage::from_day(self.day);
        let total_rate: f32 = neurons.iter().map(|n| n.firing_rate_hz).sum();
        let n = neurons.len() as f32;
        let mean_rate = total_rate / n;

        // Allocate power to bands based on mean firing rate and stage
        let stage_factor = match stage {
            DevelopmentalStage::EarlyProliferation | DevelopmentalStage::NeuralInduction => 0.0,
            DevelopmentalStage::Patterning => 0.1,
            DevelopmentalStage::Synaptogenesis => 0.3,
            DevelopmentalStage::MaturationI => 0.6,
            DevelopmentalStage::MaturationII => 1.0,
        };

        let base = mean_rate * stage_factor;
        LocalFieldPotential {
            samples,
            delta_power: base * 0.35,
            theta_power: base * 0.25,
            alpha_power: base * 0.20,
            beta_power: base * 0.12,
            gamma_power: base * 0.08,
        }
    }

    /// Historical compatibility wrapper. Prefer
    /// [`Self::generate_spatial_electrode_snapshot`].
    #[deprecated(
        since = "0.1.0",
        note = "this output is a spatial proxy with heuristic band priors, not a temporal LFP"
    )]
    pub fn generate_lfp(&self) -> LocalFieldPotential {
        self.generate_spatial_electrode_snapshot()
    }

    /// Estimate a graph/activity/maturity integration proxy from the current
    /// network state. This is not an implementation of IIT Phi.
    ///
    /// Uses a graph-theoretic proxy combining three components:
    ///
    /// 1. **Integration density**: ratio of cross-partition synaptic weight
    ///    to total synaptic weight across a bipartition. Measures how much
    ///    information flows between subsystems (Tononi 2004). Averaged over
    ///    multiple random bipartitions for robustness.
    ///
    /// 2. **Activity factor**: mean firing rate scaled via sigmoid to [0, 1].
    ///    A connected but silent network shows lower Phi than an active one.
    ///
    /// 3. **Synapse maturity**: average maturity weights how established the
    ///    connectivity is (immature synapses contribute less integration).
    ///
    /// Formula: `proxy = integration * activity * maturity`, bounded to [0, 1].
    pub fn compute_network_integration_proxy(&self) -> f64 {
        let neurons: Vec<&OrganoidCell> = self
            .cells
            .iter()
            .filter(|c| c.is_alive && c.kind.is_neuron())
            .collect();
        let n = neurons.len();
        if n < 4 {
            return 0.0;
        }

        let id_to_idx: std::collections::HashMap<u32, usize> =
            neurons.iter().enumerate().map(|(i, c)| (c.id, i)).collect();

        let dim = n.min(64); // cap for performance

        // --- Component 1: Integration density ---
        // For each of several bipartitions, compute the fraction of total
        // synaptic weight that crosses the partition boundary. High cross-
        // partition weight = high integration = hard to decompose.
        // Use deterministic bipartitions: even/odd, first/second half,
        // interleaved by 2, and interleaved by 4.
        let partitions: &[fn(usize, usize) -> bool] = &[
            |i, _dim| i % 2 == 0,       // even / odd
            |i, dim| i < dim / 2,       // first half / second half
            |i, _dim| (i / 2) % 2 == 0, // pairs interleaved
            |i, _dim| (i / 4) % 2 == 0, // quads interleaved
        ];

        let mut min_integration = f64::MAX;
        for partition_fn in partitions {
            let mut cross_weight = 0.0f64;
            let mut total_weight = 0.0f64;
            for syn in &self.synapses {
                if let (Some(&i), Some(&j)) = (id_to_idx.get(&syn.pre), id_to_idx.get(&syn.post))
                    && i < dim
                    && j < dim
                {
                    let w = (syn.weight * syn.maturity) as f64;
                    total_weight += w;
                    let i_in_a = partition_fn(i, dim);
                    let j_in_a = partition_fn(j, dim);
                    if i_in_a != j_in_a {
                        cross_weight += w;
                    }
                }
            }
            if total_weight > 1e-12 {
                // Integration = fraction of weight crossing the partition.
                // For a fully integrated network, this approaches 0.5.
                // Normalize to [0, 1] by mapping 0.5 → 1.0.
                let ratio = (cross_weight / total_weight) * 2.0;
                let integration = ratio.min(1.0);
                if integration < min_integration {
                    min_integration = integration;
                }
            }
        }
        if min_integration == f64::MAX {
            return 0.0; // no synapses in the neuron sub-graph
        }

        // --- Component 2: Activity factor ---
        let mean_rate: f64 = neurons[..dim]
            .iter()
            .map(|c| c.firing_rate_hz as f64)
            .sum::<f64>()
            / dim as f64;
        // Sigmoid mapping: half-max at 2 Hz, saturates ~10 Hz
        let activity = mean_rate / (mean_rate + 2.0);

        // --- Component 3: Synapse maturity ---
        let mean_maturity = if self.synapses.is_empty() {
            0.0
        } else {
            self.synapses.iter().map(|s| s.maturity as f64).sum::<f64>()
                / self.synapses.len() as f64
        };

        // Combine: all three must be present for significant Phi.
        // Scale by 0.8 so peak realistic Phi is ~0.4 (matches developmental
        // expectation for an in-silico organoid).

        (min_integration * activity * mean_maturity * 0.8).clamp(0.0, 1.0)
    }

    /// Historical compatibility wrapper. Prefer
    /// [`Self::compute_network_integration_proxy`].
    #[deprecated(
        since = "0.1.0",
        note = "the current calculation is a graph/activity/maturity proxy, not IIT Phi"
    )]
    pub fn compute_phi(&self) -> f64 {
        self.compute_network_integration_proxy()
    }

    // ---- private helpers ----

    fn proliferate(&mut self, stage: DevelopmentalStage) {
        let rate = match stage {
            DevelopmentalStage::EarlyProliferation => PROLIFERATION_RATE,
            DevelopmentalStage::NeuralInduction => PROLIFERATION_RATE * 0.8,
            DevelopmentalStage::Patterning => PROLIFERATION_RATE * 0.4,
            _ => PROLIFERATION_RATE * 0.05, // low maintenance division
        };

        let current_count = self.cells.len();
        if current_count >= self.max_cells {
            return;
        }

        let mut new_cells = Vec::new();
        for i in 0..current_count {
            if !self.cells[i].is_alive {
                continue;
            }
            let can_divide = matches!(
                self.cells[i].kind,
                CellKind::Stem | CellKind::NeuralProgenitor
            );
            if !can_divide {
                continue;
            }
            if self.rng.gen_bool(rate.min(1.0))
                && (current_count + new_cells.len()) < self.max_cells
            {
                let parent = &self.cells[i];
                let mut pos = parent.position;
                for d in 0..3 {
                    pos[d] += self.rng.gen_range(-0.02..0.02);
                }
                let mut ge = parent.gene_expression;
                for g in &mut ge {
                    *g = (*g + self.rng.gen_range(-0.03..0.03)).clamp(0.0, 1.0);
                }
                let id = self.next_id;
                self.next_id += 1;
                new_cells.push(OrganoidCell {
                    id,
                    kind: parent.kind,
                    position: pos,
                    age_days: 0,
                    firing_rate_hz: 0.0,
                    membrane_potential_mv: RESTING_POTENTIAL_MV,
                    gene_expression: ge,
                    is_alive: true,
                });
            }
        }
        self.cells.extend(new_cells);
    }

    fn differentiate(&mut self, stage: DevelopmentalStage) {
        for cell in &mut self.cells {
            if !cell.is_alive {
                continue;
            }
            match (cell.kind, stage) {
                // Stem → NeuralProgenitor when PAX6 rises
                (CellKind::Stem, DevelopmentalStage::EarlyProliferation)
                | (CellKind::Stem, DevelopmentalStage::NeuralInduction)
                | (CellKind::Stem, DevelopmentalStage::Patterning) => {
                    // Upregulate PAX6 over time; rate increases with stage
                    let pax6_rate = match stage {
                        DevelopmentalStage::EarlyProliferation => 0.04,
                        _ => 0.08,
                    };
                    cell.gene_expression[PAX6] += pax6_rate;
                    cell.gene_expression[SOX2] -= 0.03;
                    if cell.gene_expression[PAX6] > 0.4 && cell.gene_expression[SOX2] < 0.7 {
                        cell.kind = CellKind::NeuralProgenitor;
                        cell.gene_expression[TBR2] = 0.3;
                        // ~25% of progenitors seeded for glial fate (GFAP > 0.15)
                        // determined by low initial SATB2 (proxy for stochastic fate)
                        if cell.gene_expression[SATB2] < 0.01 && cell.id % 4 == 0 {
                            cell.gene_expression[GFAP] = 0.2;
                        }
                    }
                }
                // NeuralProgenitor → Neuron
                (CellKind::NeuralProgenitor, DevelopmentalStage::NeuralInduction)
                | (CellKind::NeuralProgenitor, DevelopmentalStage::Patterning)
                | (CellKind::NeuralProgenitor, DevelopmentalStage::Synaptogenesis) => {
                    // ~25% of progenitors are reserved for late gliogenesis
                    // (determined by GFAP seed from Stem→Progenitor transition)
                    if cell.gene_expression[GFAP] > 0.15 {
                        // This progenitor is on the glial track; skip neuron diff
                        cell.gene_expression[GFAP] += 0.01;
                    } else {
                        cell.gene_expression[CTIP2] += 0.06;
                        cell.gene_expression[SATB2] += 0.02;
                        cell.gene_expression[SYN1] += 0.03;
                        if cell.gene_expression[CTIP2] > 0.4 {
                            // ~80% excitatory, ~20% inhibitory (Quadrato 2017)
                            // Cells with id % 5 == 0 become inhibitory
                            cell.kind = if cell.id % 5 == 0 {
                                CellKind::InhibitoryNeuron
                            } else {
                                CellKind::ExcitatoryNeuron
                            };
                            cell.membrane_potential_mv = RESTING_POTENTIAL_MV;
                            // Boost SYN1 upon neuronal differentiation
                            cell.gene_expression[SYN1] =
                                (cell.gene_expression[SYN1] + 0.3).min(1.0);
                        }
                    }
                }
                // Late: some progenitors → glia
                (CellKind::NeuralProgenitor, DevelopmentalStage::MaturationI)
                | (CellKind::NeuralProgenitor, DevelopmentalStage::MaturationII) => {
                    cell.gene_expression[GFAP] += 0.06;
                    cell.gene_expression[OLIG2] += 0.03;
                    if cell.gene_expression[GFAP] > 0.5 {
                        cell.kind = CellKind::Astrocyte;
                    } else if cell.gene_expression[OLIG2] > 0.6 {
                        cell.kind = CellKind::Oligodendrocyte;
                    }
                }
                _ => {}
            }
        }
    }

    fn migrate(&mut self) {
        // Radial migration: cells drift outward from center
        for cell in &mut self.cells {
            if !cell.is_alive {
                continue;
            }
            let r = dist(&cell.position, &[0.0, 0.0, 0.0]);
            if r < 0.001 {
                continue;
            }
            let speed = match cell.kind {
                CellKind::NeuralProgenitor => 0.005,
                CellKind::ExcitatoryNeuron | CellKind::InhibitoryNeuron => 0.003,
                CellKind::Interneuron => 0.004,
                _ => 0.001,
            };
            for d in 0..3 {
                cell.position[d] += (cell.position[d] / r) * speed;
            }
        }
    }

    fn synaptogenesis(&mut self, stage: DevelopmentalStage) {
        if stage < DevelopmentalStage::Synaptogenesis {
            return;
        }

        // Collect neuron ids + positions
        let neurons: Vec<(u32, [f32; 3], CellKind, f32)> = self
            .cells
            .iter()
            .filter(|c| c.is_alive && c.kind.is_neuron())
            .map(|c| (c.id, c.position, c.kind, c.gene_expression[SYN1]))
            .collect();

        // Existing synapse pairs for dedup
        let existing: std::collections::HashSet<(u32, u32)> =
            self.synapses.iter().map(|s| (s.pre, s.post)).collect();

        let mut new_synapses = Vec::new();
        let synapse_prob_base = match stage {
            DevelopmentalStage::Synaptogenesis => 0.06,
            DevelopmentalStage::MaturationI => 0.03,
            DevelopmentalStage::MaturationII => 0.015,
            _ => 0.0,
        };

        for i in 0..neurons.len() {
            for j in (i + 1)..neurons.len() {
                let d = dist_arr(&neurons[i].1, &neurons[j].1);
                if d > SYNAPSE_DISTANCE_MAX {
                    continue;
                }
                let syn1_avg = (neurons[i].3 + neurons[j].3) / 2.0;
                let prob =
                    synapse_prob_base * (1.0 - d / SYNAPSE_DISTANCE_MAX) as f64 * syn1_avg as f64;
                if !existing.contains(&(neurons[i].0, neurons[j].0))
                    && self.rng.gen_bool(prob.clamp(0.0, 1.0))
                {
                    let stype = match neurons[i].2 {
                        CellKind::ExcitatoryNeuron => SynapseType::Glutamatergic,
                        CellKind::InhibitoryNeuron | CellKind::Interneuron => {
                            SynapseType::GABAergic
                        }
                        _ => SynapseType::Cholinergic,
                    };
                    new_synapses.push(Synapse {
                        pre: neurons[i].0,
                        post: neurons[j].0,
                        weight: self.rng.gen_range(0.05..0.3),
                        synapse_type: stype,
                        // Initial maturity > 0 so new synapses can
                        // immediately transmit (weakly). Full maturation
                        // still requires ~40 days of +0.02/day growth.
                        maturity: 0.1,
                    });
                }
            }
        }
        // Mature existing synapses
        for s in &mut self.synapses {
            s.maturity = (s.maturity + 0.02).min(1.0);
        }
        self.synapses.extend(new_synapses);
    }

    fn simulate_activity(&mut self, stage: DevelopmentalStage) {
        if stage < DevelopmentalStage::Synaptogenesis {
            return;
        }

        // Build id→index map
        let id_to_idx: std::collections::HashMap<u32, usize> = self
            .cells
            .iter()
            .enumerate()
            .map(|(i, c)| (c.id, i))
            .collect();

        // Collect synaptic input for each neuron
        let mut input = vec![0.0f32; self.cells.len()];
        for syn in &self.synapses {
            if let Some(&pre_idx) = id_to_idx.get(&syn.pre)
                && let Some(&post_idx) = id_to_idx.get(&syn.post)
                && self.cells[pre_idx].firing_rate_hz > 0.1
            {
                let sign = match syn.synapse_type {
                    SynapseType::Glutamatergic | SynapseType::Cholinergic => 1.0,
                    SynapseType::GABAergic => -0.5,
                };
                input[post_idx] += syn.weight * syn.maturity * sign;
            }
        }

        // Spontaneous noise amplitude by stage.
        // Must be large enough relative to the 15 mV gap between resting
        // (-70 mV) and firing threshold (-55 mV) to produce spontaneous
        // spikes, which then propagate through synaptic connections.
        let noise_amp = match stage {
            DevelopmentalStage::Synaptogenesis => 10.0,
            DevelopmentalStage::MaturationI => 12.0,
            DevelopmentalStage::MaturationII => 15.0,
            _ => 0.0,
        };

        // Update membrane potentials (leaky integrate-and-fire)
        for (i, cell) in self.cells.iter_mut().enumerate() {
            if !cell.is_alive || !cell.kind.is_neuron() {
                continue;
            }
            // Leak toward resting potential
            cell.membrane_potential_mv +=
                MEMBRANE_DECAY * (RESTING_POTENTIAL_MV - cell.membrane_potential_mv);

            // Synaptic drive
            cell.membrane_potential_mv += input[i] * 5.0;

            // Spontaneous noise
            cell.membrane_potential_mv += self.rng.gen_range(-noise_amp..noise_amp);

            // Spike detection
            if cell.membrane_potential_mv > FIRING_THRESHOLD_MV {
                cell.firing_rate_hz = (cell.firing_rate_hz + 1.0).min(100.0);
                cell.membrane_potential_mv = RESET_POTENTIAL_MV;
            } else {
                // Decay firing rate
                cell.firing_rate_hz *= 0.95;
            }
        }
    }

    fn prune_synapses(&mut self) {
        if self.day < PRUNING_START_DAY {
            return;
        }
        self.synapses.retain(|s| s.weight >= PRUNE_WEIGHT_THRESHOLD);
    }

    fn check_ethics(&mut self, phi: f64) {
        self.ethics = EthicsStatus::from_phi(phi);
        if matches!(
            self.ethics,
            EthicsStatus::ConsciousnessLikely | EthicsStatus::EthicsReviewRequired
        ) {
            self.halted = true;
        }
    }

    fn update_metrics(&mut self, phi: f64, stage: DevelopmentalStage) {
        let alive: Vec<&OrganoidCell> = self.cells.iter().filter(|c| c.is_alive).collect();
        let neurons: Vec<&&OrganoidCell> = alive.iter().filter(|c| c.kind.is_neuron()).collect();
        let neuron_count = neurons.len();
        let cell_count = alive.len();

        let mean_firing = if neuron_count > 0 {
            neurons.iter().map(|n| n.firing_rate_hz).sum::<f32>() / neuron_count as f32
        } else {
            0.0
        };

        let active_fraction = if neuron_count > 0 {
            neurons.iter().filter(|n| n.firing_rate_hz > 0.5).count() as f32 / neuron_count as f32
        } else {
            0.0
        };

        // Oscillation detection: check for bimodal firing distribution
        // (simplified: if variance of firing rates is high relative to mean)
        let oscillation = if neuron_count > 4 {
            let var = neurons
                .iter()
                .map(|n| (n.firing_rate_hz - mean_firing).powi(2))
                .sum::<f32>()
                / neuron_count as f32;
            var > mean_firing.max(0.1) * 0.5
        } else {
            false
        };

        self.metrics = OrganoidMetrics {
            cell_count,
            neuron_count,
            neural_fraction: if cell_count > 0 {
                neuron_count as f32 / cell_count as f32
            } else {
                0.0
            },
            synapse_count: self.synapses.len(),
            synapse_density: if neuron_count > 0 {
                self.synapses.len() as f32 / neuron_count as f32
            } else {
                0.0
            },
            mean_firing_rate_hz: mean_firing,
            spontaneous_activity_index: active_fraction,
            phi_estimate: phi,
            oscillation_detected: oscillation,
            stage,
            ethics_status: self.ethics.clone(),
        };
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

fn dist(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn dist_arr(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    dist(a, b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scientific_output_status_is_explicit() {
        let organoid = DigitalOrganoid::new(10, 1);
        let snapshot = organoid.generate_spatial_electrode_snapshot();
        assert_eq!(
            snapshot.sample_status(),
            ScientificOutputStatus::SimulatedProxy
        );
        assert_eq!(
            snapshot.band_power_status(),
            ScientificOutputStatus::HeuristicPrior
        );
        assert_eq!(
            organoid.metrics().network_integration_proxy(),
            organoid.metrics().phi_estimate
        );
    }

    #[test]
    fn explicit_max_cells_caps_proliferation() {
        let mut organoid = DigitalOrganoid::with_max_cells(50, 55, 17);
        organoid.run_to_day(100);
        assert!(organoid.metrics().cell_count <= 55);
        assert_eq!(organoid.max_cells(), 55);
    }

    #[test]
    fn max_cells_never_drops_below_initial_population() {
        let organoid = DigitalOrganoid::with_max_cells(50, 10, 17);
        assert_eq!(organoid.max_cells(), 50);
        assert_eq!(organoid.metrics().cell_count, 50);
    }

    #[test]
    fn new_organoid_starts_at_day_zero_with_stem_cells() {
        let org = DigitalOrganoid::new(50, 42);
        assert_eq!(org.day(), 0);
        assert_eq!(org.cells().len(), 50);
        assert!(org.cells().iter().all(|c| c.kind == CellKind::Stem));
        assert!(!org.is_halted());
    }

    #[test]
    fn proliferation_increases_cell_count() {
        let mut org = DigitalOrganoid::new(50, 42);
        let initial = org.cells().len();
        // Run a few days in early proliferation
        org.run_to_day(5);
        assert!(
            org.cells().len() > initial,
            "cell count should increase: {} -> {}",
            initial,
            org.cells().len()
        );
    }

    #[test]
    fn neural_differentiation_after_day_10() {
        let mut org = DigitalOrganoid::new(100, 42);
        org.run_to_day(25);
        let neuron_count = org.cells().iter().filter(|c| c.kind.is_neuron()).count();
        assert!(
            neuron_count > 0,
            "should have neurons after day 25, got {}",
            neuron_count
        );
    }

    #[test]
    fn excitatory_inhibitory_ratio_approximately_80_20() {
        let mut org = DigitalOrganoid::new(200, 42);
        org.run_to_day(50);
        let excit = org
            .cells()
            .iter()
            .filter(|c| c.kind == CellKind::ExcitatoryNeuron)
            .count();
        let inhib = org
            .cells()
            .iter()
            .filter(|c| c.kind == CellKind::InhibitoryNeuron)
            .count();
        let total = excit + inhib;
        if total > 10 {
            let ratio = excit as f32 / total as f32;
            // Allow wide tolerance since differentiation is stochastic
            assert!(
                ratio > 0.4 && ratio < 0.95,
                "E/I ratio {} out of expected range (excit={}, inhib={})",
                ratio,
                excit,
                inhib
            );
        }
    }

    #[test]
    fn synaptogenesis_creates_synapses_after_day_40() {
        let mut org = DigitalOrganoid::new(100, 42);
        assert_eq!(org.synapses().len(), 0);
        org.run_to_day(55);
        assert!(
            org.synapses().len() > 0,
            "should have synapses after day 55"
        );
    }

    #[test]
    fn synapse_density_increases_with_maturation() {
        let mut org = DigitalOrganoid::new(100, 42);
        org.run_to_day(50);
        let density_50 = org.metrics().synapse_density;
        org.run_to_day(90);
        let density_90 = org.metrics().synapse_density;
        assert!(
            density_90 >= density_50,
            "density should increase: {} -> {}",
            density_50,
            density_90
        );
    }

    #[test]
    fn firing_rate_positive_after_synaptogenesis() {
        let mut org = DigitalOrganoid::new(100, 42);
        org.run_to_day(60);
        let any_firing = org
            .cells()
            .iter()
            .any(|c| c.kind.is_neuron() && c.firing_rate_hz > 0.0);
        assert!(any_firing, "some neurons should be firing by day 60");
    }

    #[test]
    fn phi_starts_near_zero_increases_with_development() {
        let mut org = DigitalOrganoid::new(100, 42);
        let history = org.run_to_day(80);
        assert!(org.phi_history()[0] < 0.01, "phi should start near 0");
        // Phi should be non-negative throughout
        for phi in org.phi_history() {
            assert!(*phi >= 0.0, "phi should be non-negative: {}", phi);
        }
        // By day 80 with 100 cells, phi should have increased meaningfully
        // (synaptogenesis starts at day 40, connectivity drives integration)
        let max_phi = org.phi_history().iter().cloned().fold(0.0f64, f64::max);
        assert!(
            max_phi >= 0.01,
            "max phi should show integration by day 80: {}",
            max_phi
        );
        // Verify history length
        assert_eq!(
            org.phi_history().len(),
            history.len() + 1,
            "phi_history should have initial + one per day"
        );
    }

    #[test]
    fn lfp_generation_produces_valid_signal() {
        let mut org = DigitalOrganoid::new(100, 42);
        org.run_to_day(60);
        let lfp = org.generate_spatial_electrode_snapshot();
        assert!(!lfp.samples.is_empty(), "LFP should have samples");
        // Power should be non-negative
        assert!(lfp.delta_power >= 0.0);
        assert!(lfp.theta_power >= 0.0);
        assert!(lfp.alpha_power >= 0.0);
        assert!(lfp.beta_power >= 0.0);
        assert!(lfp.gamma_power >= 0.0);
    }

    #[test]
    fn ethics_gate_triggers_at_consciousness_threshold() {
        // Force high phi by using a termination check
        let status_low = EthicsStatus::from_phi(0.05);
        assert_eq!(status_low, EthicsStatus::PreConscious);

        let status_monitor = EthicsStatus::from_phi(0.2);
        assert_eq!(status_monitor, EthicsStatus::MonitoringRequired);

        let status_review = EthicsStatus::from_phi(0.35);
        assert_eq!(status_review, EthicsStatus::EthicsReviewRequired);

        let status_halt = EthicsStatus::from_phi(0.6);
        assert_eq!(status_halt, EthicsStatus::ConsciousnessLikely);
    }

    #[test]
    fn halted_organoid_refuses_advance() {
        let mut org = DigitalOrganoid::new(10, 42);
        org.terminate("test halt");
        assert!(org.is_halted());
        let result = org.advance_day();
        assert!(result.is_none(), "halted organoid should return None");
    }

    #[test]
    fn stage_progression_matches_timeline() {
        assert_eq!(
            DevelopmentalStage::from_day(0),
            DevelopmentalStage::EarlyProliferation
        );
        assert_eq!(
            DevelopmentalStage::from_day(9),
            DevelopmentalStage::EarlyProliferation
        );
        assert_eq!(
            DevelopmentalStage::from_day(10),
            DevelopmentalStage::NeuralInduction
        );
        assert_eq!(
            DevelopmentalStage::from_day(20),
            DevelopmentalStage::Patterning
        );
        assert_eq!(
            DevelopmentalStage::from_day(40),
            DevelopmentalStage::Synaptogenesis
        );
        assert_eq!(
            DevelopmentalStage::from_day(80),
            DevelopmentalStage::MaturationI
        );
        assert_eq!(
            DevelopmentalStage::from_day(120),
            DevelopmentalStage::MaturationII
        );
        assert_eq!(
            DevelopmentalStage::from_day(200),
            DevelopmentalStage::MaturationII
        );
    }

    #[test]
    fn astrocytes_appear_in_later_stages() {
        let mut org = DigitalOrganoid::new(200, 42);
        org.run_to_day(140);
        let astro_count = org
            .cells()
            .iter()
            .filter(|c| c.kind == CellKind::Astrocyte)
            .count();
        assert!(
            astro_count > 0,
            "astrocytes should appear by day 140, got {}",
            astro_count
        );
    }

    #[test]
    fn run_to_day_stops_at_ethics_halt() {
        let mut org = DigitalOrganoid::new(50, 42);
        // Manually halt after a few days
        org.run_to_day(5);
        org.terminate("ethics test");
        let further = org.run_to_day(100);
        assert!(further.is_empty(), "should not advance after termination");
        assert!(org.day() <= 5, "day should not advance past halt");
    }

    #[test]
    fn phi_history_length_matches_days_advanced() {
        let mut org = DigitalOrganoid::new(50, 42);
        let target = 30;
        let history = org.run_to_day(target);
        // phi_history has initial (day 0) + one per advanced day
        assert_eq!(
            org.phi_history().len(),
            history.len() + 1,
            "phi_history({}) should be metrics({}) + 1",
            org.phi_history().len(),
            history.len()
        );
        assert_eq!(org.day(), target);
    }
}
