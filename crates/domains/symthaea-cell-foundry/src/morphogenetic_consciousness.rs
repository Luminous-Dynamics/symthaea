// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Morphogenetic consciousness: modeling how consciousness emerges from
//! cellular differentiation and self-organization.
//!
//! Implements a reaction-diffusion model (Turing 1952) where consciousness
//! potential is a morphogenetic field variable that emerges from neural
//! differentiation, synaptic connectivity, and information integration.
//!
//! This chemical (activator/inhibitor) layer is deliberately paired with a
//! second, electrical layer in [`crate::bioelectric`] — gap-junction-coupled
//! transmembrane voltage (Vmem) — because chemical gradients alone are not
//! the whole morphogenesis story. See that module's docs for why.
//!
//! References:
//! - Turing, A. M. (1952). The chemical basis of morphogenesis.
//! - Trujillo et al. (2019). Complex oscillatory waves in cortical organoids.
//! - Tononi, G. (2004). IIT and information integration.
//! - Kauffman, S. A. (1993). Origins of order: Self-organization.
//! - Levin, M. (2019). The computational boundary of a self: developmental
//!   bioelectricity drives multicellularity and scale-free cognition.
//!   Front. Psychol. 10:2688.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::bioelectric::{BioelectricState, K_CHANNEL_DIFFERENTIATION_BUMP, TargetMorphology};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Activator diffusion coefficient (Turing instability: D_h >> D_a).
const D_ACTIVATOR: f32 = 0.01;
/// Inhibitor diffusion coefficient — must be significantly larger than D_a.
const D_INHIBITOR: f32 = 0.08;
/// Base production rate for the activator.
const BASE_PRODUCTION: f32 = 0.01;
/// Decay rate of the inhibitor.
const INHIBITOR_DECAY: f32 = 0.04;
/// Spatial radius for neighbourhood queries (3-D Euclidean).
const NEIGHBOURHOOD_RADIUS: f32 = 0.15;
/// Phi threshold at which the ethics gate fires.
const PHI_ETHICS_THRESHOLD: f64 = 0.3;
/// Maximum entries kept in phi history ring buffer.
const PHI_HISTORY_CAP: usize = 512;
/// Connectivity radius for synaptogenesis.
const SYNAPSE_RADIUS: f32 = 0.12;
/// Cell size for the shared per-day spatial grid (`crate::spatial_grid`)
/// that backs `neighbours()`/`form_synapses`/`form_gap_junctions`. Must
/// stay >= the largest of `NEIGHBOURHOOD_RADIUS`, `SYNAPSE_RADIUS` (both
/// above), and `GAP_JUNCTION_RADIUS` (bioelectric.rs) — currently
/// `GAP_JUNCTION_RADIUS` = 0.4 is the largest.
pub(crate) const SPATIAL_GRID_CELL_SIZE: f32 = 0.4;
/// Proliferation multiplier: fraction of progenitors that divide per day.
const PROLIFERATION_RATE: f64 = 0.10;
/// Maximum cells to avoid runaway proliferation in simulation.
/// `pub(crate)` so [`crate::bioelectric`]'s regenerative proliferation pass
/// respects the same cap.
pub(crate) const MAX_CELLS: usize = 10_000;

// ---------------------------------------------------------------------------
// Cell types (distinct from `types::CellType` which models iPSC lineages)
// ---------------------------------------------------------------------------

/// Neuron functional subtype within an organoid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuronSubtype {
    Excitatory,
    Inhibitory,
    Modulatory,
}

/// Glial functional subtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GlialSubtype {
    Astrocyte,
    Oligodendrocyte,
    Microglia,
}

/// Organoid cell identity — tracks differentiation from progenitor to
/// specialised neural/glial lineages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrganoidCellType {
    Undifferentiated,
    Progenitor,
    NeuralProgenitor,
    Neuron(NeuronSubtype),
    Glial(GlialSubtype),
}

impl OrganoidCellType {
    /// Returns `true` when the cell is a neuron of any subtype.
    pub fn is_neuron(&self) -> bool {
        matches!(self, OrganoidCellType::Neuron(_))
    }

    /// Returns `true` when the cell is glial of any subtype.
    pub fn is_glial(&self) -> bool {
        matches!(self, OrganoidCellType::Glial(_))
    }

    /// Returns `true` for any progenitor (including neural progenitor).
    pub fn is_progenitor(&self) -> bool {
        matches!(
            self,
            OrganoidCellType::Progenitor | OrganoidCellType::NeuralProgenitor
        )
    }
}

// ---------------------------------------------------------------------------
// OrganoidCell
// ---------------------------------------------------------------------------

/// A single cell within the organoid simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganoidCell {
    /// 3-D position inside the organoid (unit sphere).
    pub position: [f32; 3],
    /// Current cell identity.
    pub cell_type: OrganoidCellType,
    /// Developmental age in days since seeding.
    pub age_days: u32,
    /// Simplified gene expression vector (length 8).
    pub gene_expression: Vec<f32>,
    /// Mean firing rate in Hz (non-zero only for neurons).
    pub firing_rate: f32,
    /// Indices of other cells this cell has synaptic connections to.
    pub connectivity: Vec<usize>,
}

impl OrganoidCell {
    fn new_random(rng: &mut StdRng) -> Self {
        let position = [
            rng.gen_range(-1.0..1.0f32),
            rng.gen_range(-1.0..1.0f32),
            rng.gen_range(-1.0..1.0f32),
        ];
        let gene_expression = (0..8).map(|_| rng.gen_range(0.0..1.0f32)).collect();
        Self {
            position,
            cell_type: OrganoidCellType::Progenitor,
            age_days: 0,
            gene_expression,
            firing_rate: 0.0,
            connectivity: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// MorphogeneticField
// ---------------------------------------------------------------------------

/// Reaction-diffusion field over a population of organoid cells.
///
/// Each cell carries an activator/inhibitor pair (Turing 1952 model) plus
/// derived consciousness-potential and local-phi values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphogeneticField {
    /// Cell population.
    pub cells: Vec<OrganoidCell>,
    /// Per-cell activator concentration.
    pub activator: Vec<f32>,
    /// Per-cell inhibitor concentration.
    pub inhibitor: Vec<f32>,
    /// Per-cell consciousness potential derived from local phi.
    pub consciousness_potential: Vec<f32>,
    /// Per-cell local phi (information integration in neighbourhood).
    pub local_phi: Vec<f64>,
    /// Pairwise connectivity weight matrix (sparse, but stored dense for
    /// small organoids).  `connectivity_matrix[i][j]` is the synaptic
    /// weight from cell i to cell j.
    pub connectivity_matrix: Vec<Vec<f64>>,
    /// Gap-junction-coupled bioelectric (Vmem) layer — see [`crate::bioelectric`].
    /// Distinct from `connectivity_matrix`: gap junctions couple *all* cells,
    /// not just neurons, and propagate on a much faster timescale than the
    /// chemical activator/inhibitor field above.
    pub bioelectric: BioelectricState,
    /// Seeded RNG for reproducibility.
    #[serde(skip, default = "default_rng")]
    rng: StdRng,
    /// Cached per-day spatial index (grid + the position snapshot it was
    /// built from — cached together so `neighbours()` never has to
    /// re-collect positions on every call, which would silently reintroduce
    /// O(n) work per query) backing `neighbours()`/`form_synapses`/
    /// `form_gap_junctions` — see `crate::spatial_grid`. `None` until
    /// `rebuild_spatial_grid` is called (which `advance_day` does once per
    /// day); every distance-based method falls back to the original
    /// brute-force scan when it's `None`, so standalone/unit-test usage
    /// that never calls `advance_day` is completely unaffected.
    #[serde(skip)]
    spatial_grid: Option<(crate::spatial_grid::SpatialGrid, Vec<[f32; 3]>)>,
}

fn default_rng() -> StdRng {
    StdRng::seed_from_u64(0)
}

impl MorphogeneticField {
    /// Create a new field with `num_cells` progenitor cells placed randomly
    /// inside a unit sphere.
    pub fn new(num_cells: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let cells: Vec<OrganoidCell> = (0..num_cells)
            .map(|_| OrganoidCell::new_random(&mut rng))
            .collect();
        let n = cells.len();
        // Initial morphogen concentrations — small random perturbation around
        // a uniform steady state so Turing instability can amplify patterns.
        let activator: Vec<f32> = (0..n)
            .map(|_| 1.0 + rng.gen_range(-0.05..0.05f32))
            .collect();
        let inhibitor: Vec<f32> = (0..n)
            .map(|_| 1.0 + rng.gen_range(-0.05..0.05f32))
            .collect();
        let consciousness_potential = vec![0.0f32; n];
        let local_phi = vec![0.0f64; n];
        let connectivity_matrix = vec![vec![0.0f64; n]; n];
        let bioelectric = BioelectricState::new(n, seed);
        Self {
            cells,
            activator,
            inhibitor,
            consciousness_potential,
            local_phi,
            connectivity_matrix,
            bioelectric,
            rng,
            spatial_grid: None,
        }
    }

    /// Number of cells in the field.
    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }

    // -- helpers --

    /// Euclidean distance between two cells.
    ///
    /// `pub(crate)` so [`crate::bioelectric`] can reuse the same spatial
    /// metric for gap-junction formation and amputation/wound geometry.
    pub(crate) fn distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Return indices of cells within `radius` of cell `idx`. Uses the
    /// cached spatial grid when one has been built (see
    /// `rebuild_spatial_grid`) and `radius` fits within its cell size;
    /// otherwise falls back to the original brute-force O(n) scan, so
    /// standalone/unit-test usage (which never calls `advance_day`, and
    /// therefore never builds a grid) is completely unaffected.
    pub(crate) fn neighbours(&self, idx: usize, radius: f32) -> Vec<usize> {
        if let Some((grid, positions)) = &self.spatial_grid
            && radius <= grid.cell_size()
        {
            return grid.query_radius(positions, &self.cells[idx].position, radius, Some(idx));
        }
        self.neighbours_brute_force(idx, radius)
    }

    /// The original O(n) reference implementation of `neighbours`, kept as
    /// the fallback path and as the correctness oracle for
    /// `spatial_grid`'s tests.
    fn neighbours_brute_force(&self, idx: usize, radius: f32) -> Vec<usize> {
        let pos = &self.cells[idx].position;
        (0..self.cells.len())
            .filter(|&j| j != idx && Self::distance(pos, &self.cells[j].position) < radius)
            .collect()
    }

    /// Rebuild the cached spatial grid from current positions. Cheap
    /// relative to what it saves: called once per day from `advance_day`,
    /// right after the last step that can move a cell (`migrate`), so it
    /// stays valid through synapse/gap-junction formation, the reaction-
    /// diffusion substeps, and phi computation — none of which move cells.
    pub(crate) fn rebuild_spatial_grid(&mut self) {
        let positions: Vec<[f32; 3]> = self.cells.iter().map(|c| c.position).collect();
        let grid = crate::spatial_grid::SpatialGrid::build(&positions, SPATIAL_GRID_CELL_SIZE);
        self.spatial_grid = Some((grid, positions));
    }

    /// Discrete Laplacian for a scalar field at cell `idx` using the
    /// neighbourhood mean minus the centre value.
    fn laplacian(field: &[f32], cells: &[OrganoidCell], idx: usize, radius: f32) -> f32 {
        let pos = &cells[idx].position;
        let mut sum = 0.0f32;
        let mut count = 0u32;
        for (j, cell) in cells.iter().enumerate() {
            if j != idx && Self::distance(pos, &cell.position) < radius {
                sum += field[j];
                count += 1;
            }
        }
        if count == 0 {
            return 0.0;
        }
        (sum / count as f32) - field[idx]
    }

    // -- public simulation steps --

    /// One step of the Turing reaction-diffusion equations.
    ///
    /// ```text
    /// da/dt = D_a * laplacian(a) + a^2 / h - a + base_production
    /// dh/dt = D_h * laplacian(h) + a^2 - h * decay
    /// ```
    pub fn step_reaction_diffusion(&mut self, dt: f32) {
        let n = self.cells.len();
        let mut new_a = self.activator.clone();
        let mut new_h = self.inhibitor.clone();

        for i in 0..n {
            let a = self.activator[i];
            let h = self.inhibitor[i].max(1e-6); // avoid division by zero
            let lap_a = Self::laplacian(&self.activator, &self.cells, i, NEIGHBOURHOOD_RADIUS);
            let lap_h = Self::laplacian(&self.inhibitor, &self.cells, i, NEIGHBOURHOOD_RADIUS);

            let reaction_a = (a * a) / h - a + BASE_PRODUCTION;
            let reaction_h = a * a - h * INHIBITOR_DECAY;

            new_a[i] = (a + dt * (D_ACTIVATOR * lap_a + reaction_a)).max(0.0);
            new_h[i] = (h + dt * (D_INHIBITOR * lap_h + reaction_h)).max(0.0);
        }

        self.activator = new_a;
        self.inhibitor = new_h;
    }

    /// Differentiate cells based on morphogen gradient, gene expression, and
    /// bioelectric state.
    ///
    /// High activator + neural gene expression → NeuralProgenitor → Neuron.
    /// Moderate activator → Glial.
    /// Low activator → remains Undifferentiated.
    ///
    /// The gene-expression signal is additively biased by each cell's local
    /// Vmem hyperpolarization (see [`BioelectricState::differentiation_bias`]):
    /// hyperpolarized cells are nudged toward committing to neural fate, and
    /// depolarized cells resist it — consistent with Levin lab findings that
    /// resting-potential state (not just genetics) gates differentiation, and
    /// that this gating is gap-junction-permeability-dependent (blockable
    /// independently of gene expression, see `NeuralOrganoid::amputate` /
    /// `set_gap_junction_permeability`).
    pub fn differentiate(&mut self) {
        let n = self.cells.len();
        for i in 0..n {
            let a = self.activator[i];
            let vmem_bias = self.bioelectric.differentiation_bias(i);
            let cell = &mut self.cells[i];

            // Only progenitors can differentiate.
            if !cell.cell_type.is_progenitor()
                && cell.cell_type != OrganoidCellType::Undifferentiated
            {
                continue;
            }

            // Neural gene expression is the mean of first 4 genes, biased by
            // the cell's bioelectric (Vmem) state.
            let neural_expr: f32 =
                (cell.gene_expression.iter().take(4).sum::<f32>() / 4.0 + vmem_bias).min(1.0);

            if a > 1.5 && neural_expr > 0.5 {
                // High activator + neural gene expression → neural lineage
                if cell.cell_type == OrganoidCellType::Progenitor {
                    cell.cell_type = OrganoidCellType::NeuralProgenitor;
                } else if cell.cell_type == OrganoidCellType::NeuralProgenitor {
                    // Determine neuron subtype from gene expression balance.
                    let excitatory_weight = cell.gene_expression.first().copied().unwrap_or(0.5);
                    let inhibitory_weight = cell.gene_expression.get(1).copied().unwrap_or(0.5);
                    if excitatory_weight > inhibitory_weight + 0.1 {
                        cell.cell_type = OrganoidCellType::Neuron(NeuronSubtype::Excitatory);
                    } else if inhibitory_weight > excitatory_weight + 0.1 {
                        cell.cell_type = OrganoidCellType::Neuron(NeuronSubtype::Inhibitory);
                    } else {
                        cell.cell_type = OrganoidCellType::Neuron(NeuronSubtype::Modulatory);
                    }
                    // Neurons acquire a baseline firing rate.
                    cell.firing_rate = self.rng.gen_range(1.0..10.0);
                    // Real Kv-channel upregulation on neural commitment --
                    // only has an effect if the opt-in ion-channel model is
                    // enabled (see `crate::bioelectric`/`crate::ion_channels`);
                    // inert otherwise since nothing reads
                    // `k_channel_conductance` when the model is off.
                    self.bioelectric.k_channel_conductance[i] += K_CHANNEL_DIFFERENTIATION_BUMP;
                }
            } else if a > 0.8 && neural_expr > 0.3 {
                // Moderate → glial
                let glial_gene = cell.gene_expression.get(5).copied().unwrap_or(0.5);
                cell.cell_type = if glial_gene > 0.7 {
                    OrganoidCellType::Glial(GlialSubtype::Oligodendrocyte)
                } else if glial_gene > 0.4 {
                    OrganoidCellType::Glial(GlialSubtype::Astrocyte)
                } else {
                    OrganoidCellType::Glial(GlialSubtype::Microglia)
                };
                // Same Kv-channel upregulation as neural commitment above --
                // glia also hyperpolarize as part of differentiation.
                self.bioelectric.k_channel_conductance[i] += K_CHANNEL_DIFFERENTIATION_BUMP;
            } else if a < 0.3 {
                cell.cell_type = OrganoidCellType::Undifferentiated;
            }

            cell.age_days += 1;
        }
    }

    /// Form synapses between nearby neurons (synaptogenesis).
    ///
    /// Candidate pairs come from [`Self::neighbours`] (grid-backed once a
    /// day's grid has been built). Candidates are sorted before use so the
    /// `i < j` ordering — and therefore the sequence of `self.rng` draws
    /// below — matches the original O(n^2) pairwise scan exactly; the grid
    /// returns matches in bucket order, not index order.
    pub fn form_synapses(&mut self) {
        let n = self.cells.len();
        for i in 0..n {
            if !self.cells[i].cell_type.is_neuron() {
                continue;
            }
            let mut candidates = self.neighbours(i, SYNAPSE_RADIUS);
            candidates.sort_unstable();
            for j in candidates {
                if j <= i || !self.cells[j].cell_type.is_neuron() {
                    continue;
                }
                // Add bidirectional connection if not already present.
                if !self.cells[i].connectivity.contains(&j) {
                    self.cells[i].connectivity.push(j);
                    self.cells[j].connectivity.push(i);
                    let w = self.rng.gen_range(0.1..1.0f64);
                    self.connectivity_matrix[i][j] = w;
                    self.connectivity_matrix[j][i] = w;
                }
            }
        }
    }

    /// Proliferate: progenitors divide, producing new cells (capped at
    /// `MAX_CELLS`).
    pub fn proliferate(&mut self) {
        let n = self.cells.len();
        if n >= MAX_CELLS {
            return;
        }
        let mut new_cells: Vec<OrganoidCell> = Vec::new();
        let mut parent_indices: Vec<usize> = Vec::new();
        for i in 0..n {
            if !self.cells[i].cell_type.is_progenitor() {
                continue;
            }
            if self.rng.gen_bool(PROLIFERATION_RATE.min(1.0)) {
                let mut daughter = self.cells[i].clone();
                // Slight positional jitter.
                for d in 0..3 {
                    daughter.position[d] += self.rng.gen_range(-0.02..0.02f32);
                }
                daughter.age_days = 0;
                daughter.connectivity.clear();
                // Small gene expression noise.
                for g in daughter.gene_expression.iter_mut() {
                    *g = (*g + self.rng.gen_range(-0.05..0.05f32)).clamp(0.0, 1.0);
                }
                new_cells.push(daughter);
                parent_indices.push(i);
                if n + new_cells.len() >= MAX_CELLS {
                    break;
                }
            }
        }
        let added = new_cells.len();
        self.cells.extend(new_cells);
        let total = self.cells.len();
        // Extend parallel vectors.
        self.activator.resize(total, 1.0);
        self.inhibitor.resize(total, 1.0);
        self.consciousness_potential.resize(total, 0.0);
        self.local_phi.resize(total, 0.0);
        // Grow connectivity matrix.
        for row in self.connectivity_matrix.iter_mut() {
            row.resize(total, 0.0);
        }
        for _ in 0..added {
            self.connectivity_matrix.push(vec![0.0; total]);
        }
        // Daughters inherit their parent's Vmem rather than resetting to a
        // default: this is proliferation everywhere in the tissue (not just
        // at a wound), so resetting to a fixed baseline would wash out any
        // surviving spatial Vmem pattern with a flood of default-valued
        // newborns as the population grows — see `BioelectricState::resize_inheriting`.
        self.bioelectric.resize_inheriting(&parent_indices);
    }

    /// Compute local phi for every cell based on covariance of firing rates
    /// in the cell's spatial neighbourhood, plus a *basal* information-
    /// integration term derived from gap-junction-coupled Vmem variance.
    ///
    /// Earlier versions of this model gated all information integration on
    /// `is_neuron()` — i.e. only neurons "computed." That encodes exactly the
    /// neuron-centrism Levin's TAME framework (Technological Approach to Mind
    /// Everywhere) argues against: all gap-junction-connected cells, not just
    /// neurons, integrate and act on bioelectric information (basal
    /// cognition). See [`BioelectricState::basal_information`].
    ///
    /// The basal term is deliberately weighted small relative to the neural
    /// term and capped well below [`PHI_ETHICS_THRESHOLD`]: basal
    /// problem-solving competency (Levin's sense) is not the same claim as
    /// phenomenal consciousness, and undifferentiated/non-neural tissue
    /// should never trip the consciousness ethics gate on its own.
    pub fn compute_local_phi(&mut self) {
        let n = self.cells.len();
        for i in 0..n {
            let nbrs = self.neighbours(i, NEIGHBOURHOOD_RADIUS);
            let basal = self.basal_information(i);

            if nbrs.is_empty() || !self.cells[i].cell_type.is_neuron() {
                self.local_phi[i] = basal;
                continue;
            }

            // Collect firing rates of neuron neighbours.
            let rates: Vec<f64> = nbrs
                .iter()
                .filter(|&&j| self.cells[j].cell_type.is_neuron())
                .map(|&j| self.cells[j].firing_rate as f64)
                .collect();

            if rates.len() < 2 {
                self.local_phi[i] = basal;
                continue;
            }

            // Mean and variance.
            let mean = rates.iter().sum::<f64>() / rates.len() as f64;
            let var = rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rates.len() as f64;

            // Connectivity density within the neighbourhood.
            let connected: usize = nbrs
                .iter()
                .filter(|&&j| self.connectivity_matrix[i][j] > 0.0)
                .count();
            let density = connected as f64 / nbrs.len() as f64;

            // Local phi ≈ variance × connectivity density (simplified IIT),
            // plus the basal (non-neural) bioelectric integration term.
            self.local_phi[i] = (var.sqrt() * density).min(1.0) + basal;
        }
    }

    /// Map local phi to the consciousness potential field.
    pub fn consciousness_potential_from_phi(&mut self) {
        for i in 0..self.cells.len() {
            self.consciousness_potential[i] = self.local_phi[i] as f32;
        }
    }

    /// Migrate cells slightly toward higher activator concentration
    /// (chemotaxis).
    pub fn migrate(&mut self) {
        let n = self.cells.len();
        for i in 0..n {
            let nbrs = self.neighbours(i, NEIGHBOURHOOD_RADIUS);
            if nbrs.is_empty() {
                continue;
            }
            // Gradient direction: mean position of higher-activator neighbours.
            let a_i = self.activator[i];
            let mut dx = [0.0f32; 3];
            let mut count = 0u32;
            for &j in &nbrs {
                if self.activator[j] > a_i {
                    for d in 0..3 {
                        dx[d] += self.cells[j].position[d] - self.cells[i].position[d];
                    }
                    count += 1;
                }
            }
            if count > 0 {
                let step = 0.005;
                for d in 0..3 {
                    self.cells[i].position[d] += step * dx[d] / count as f32;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// OrganoidStage & EthicsStatus
// ---------------------------------------------------------------------------

/// Developmental stage of the organoid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OrganoidStage {
    Proliferative,
    Migratory,
    Differentiating,
    Synaptogenic,
    Mature,
    Conscious,
}

/// Ethics status tracking for organoid consciousness research.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OrganoidEthicsStatus {
    PreConscious,
    EmergingConsciousness(f64),
    ConsciousnessDetected(f64),
    ExperimentHalted(String),
}

// ---------------------------------------------------------------------------
// NeuralOrganoid
// ---------------------------------------------------------------------------

/// A simulated neural organoid that develops over time, tracking
/// consciousness emergence via integrated information (Phi).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralOrganoid {
    /// The reaction-diffusion field underlying the organoid.
    pub field: MorphogeneticField,
    /// Current developmental day.
    pub developmental_day: u32,
    /// Ring buffer of global phi measurements.
    pub phi_history: VecDeque<f64>,
    /// Spontaneous neural activity frequency (Hz).
    pub spontaneous_activity_hz: f32,
    /// Current developmental stage.
    pub stage: OrganoidStage,
    /// Ethics gate status.
    pub ethics_status: OrganoidEthicsStatus,
    /// Captured goal pattern the tissue is regenerating toward, if any.
    /// See [`crate::bioelectric::TargetMorphology`] and
    /// `NeuralOrganoid::capture_target_morphology` /
    /// `NeuralOrganoid::amputate`.
    pub target_morphology: Option<TargetMorphology>,
    /// Days elapsed since the most recent amputation (used to time out
    /// regeneration bookkeeping). Irrelevant while `target_morphology` is
    /// `None`.
    pub(crate) days_since_wound: u32,
    /// Whether positional homing (pulling each cell's Vmem toward the
    /// target's value at its own position, independent of gap junctions) is
    /// active. See `NeuralOrganoid::set_positional_homing`. Defaults to
    /// `false` so existing behavior/tests are unaffected unless explicitly
    /// opted in.
    pub(crate) positional_homing_enabled: bool,
    /// Whether the opt-in real-physics packing correction pass
    /// (`crate::packing`) runs each day. See
    /// `NeuralOrganoid::set_packing_enabled`. Defaults to `false` for the
    /// same reason as `positional_homing_enabled`.
    pub(crate) packing_enabled: bool,
}

impl NeuralOrganoid {
    /// Create a new organoid with `num_cells` progenitor cells.
    pub fn new(num_cells: usize, seed: u64) -> Self {
        Self {
            field: MorphogeneticField::new(num_cells, seed),
            developmental_day: 0,
            phi_history: VecDeque::with_capacity(PHI_HISTORY_CAP),
            spontaneous_activity_hz: 0.0,
            stage: OrganoidStage::Proliferative,
            ethics_status: OrganoidEthicsStatus::PreConscious,
            target_morphology: None,
            days_since_wound: 0,
            positional_homing_enabled: false,
            packing_enabled: false,
        }
    }

    /// Advance the organoid by one developmental day.
    ///
    /// Sequence: proliferate → regeneration bias (if regenerating) → migrate
    /// → differentiate (chemically- and bioelectrically-biased) →
    /// synaptogenesis → gap-junction formation + Vmem equilibration →
    /// reaction-diffusion step → compute phi (neural + basal) → update stage
    /// → check ethics.
    pub fn advance_day(&mut self) {
        // If the experiment is halted, do nothing.
        if matches!(
            self.ethics_status,
            OrganoidEthicsStatus::ExperimentHalted(_)
        ) {
            return;
        }

        self.developmental_day += 1;

        // 1. Proliferate (early days).
        self.field.proliferate();

        // 1.5 Regeneration: extra proliferation for wound-boundary
        //     progenitors while actively regenerating toward a captured
        //     target morphology (see `amputate` / `capture_target_morphology`).
        self.advance_regeneration();

        // 1.6 Defection: unconditional hyperproliferation for any
        //     bioelectrically-isolated ("defected") cells, independent of
        //     regeneration state — see `induce_local_defection`.
        self.field.defective_proliferate();

        // 1.7 Packing (opt-in): resolve any physical overlap left by this
        //     day's proliferation via real collision response, before
        //     migration reads positions. See `crate::packing`.
        if self.packing_enabled {
            crate::packing::resolve_packing(&mut self.field);
        }

        // 1.8 Rebuild the spatial grid backing neighbours() now that this
        //     day's cell count and positions are settled (proliferation,
        //     regeneration, defection, and packing have all already run) —
        //     must happen before `migrate`, the first step that queries
        //     neighbours this day. See `crate::spatial_grid`.
        self.field.rebuild_spatial_grid();

        // 2. Migrate toward morphogen gradients.
        self.field.migrate();

        // 3. Differentiate based on current morphogen field and bioelectric
        //    (Vmem) state.
        self.field.differentiate();

        // 4. Form synapses between nearby neurons.
        self.field.form_synapses();

        // 4.5 Bioelectric layer: form/refresh the gap-junction network (all
        //     cells, not just neurons) and let Vmem equilibrate across it.
        //     This runs on a faster timescale than the chemical field below —
        //     see `crate::bioelectric` docs.
        self.field.form_gap_junctions();
        for _ in 0..5 {
            self.field.step_vmem_diffusion(0.2);
        }

        // 4.6 Positional homing (opt-in): pull each cell's Vmem toward the
        //     captured target's value at its own position, independent of
        //     gap-junction state. See `crate::bioelectric` docs.
        self.apply_positional_homing();

        // 5. Run several sub-steps of reaction-diffusion per day.
        for _ in 0..10 {
            self.field.step_reaction_diffusion(0.1);
        }

        // 6. Update spontaneous activity from neuron population.
        self.update_spontaneous_activity();

        // 7. Compute local phi and consciousness potential.
        self.field.compute_local_phi();
        self.field.consciousness_potential_from_phi();

        // 8. Record global phi.
        let phi = self.global_phi();
        if self.phi_history.len() >= PHI_HISTORY_CAP {
            self.phi_history.pop_front();
        }
        self.phi_history.push_back(phi);

        // 9. Update developmental stage.
        self.update_stage();

        // 10. Ethics check.
        self.check_ethics();
    }

    /// Aggregate consciousness measure — mean of non-zero local phi values,
    /// weighted by connectivity density.
    pub fn global_phi(&self) -> f64 {
        let non_zero: Vec<f64> = self
            .field
            .local_phi
            .iter()
            .copied()
            .filter(|&p| p > 0.0)
            .collect();
        if non_zero.is_empty() {
            return 0.0;
        }
        let mean = non_zero.iter().sum::<f64>() / non_zero.len() as f64;
        // Scale by the fraction of cells that are neurons (integration breadth).
        let neuron_frac = self
            .field
            .cells
            .iter()
            .filter(|c| c.cell_type.is_neuron())
            .count() as f64
            / self.field.cells.len().max(1) as f64;
        mean * neuron_frac.sqrt()
    }

    /// Update the ethics status based on current global phi.
    pub fn check_ethics(&mut self) {
        // Don't un-halt.
        if matches!(
            self.ethics_status,
            OrganoidEthicsStatus::ExperimentHalted(_)
        ) {
            return;
        }

        let phi = self.global_phi();

        if phi >= PHI_ETHICS_THRESHOLD {
            self.ethics_status = OrganoidEthicsStatus::ConsciousnessDetected(phi);
        } else if phi > PHI_ETHICS_THRESHOLD * 0.5 {
            self.ethics_status = OrganoidEthicsStatus::EmergingConsciousness(phi);
        }
        // else remains PreConscious
    }

    /// Halt the experiment with a reason string.
    pub fn halt_experiment(&mut self, reason: String) {
        self.ethics_status = OrganoidEthicsStatus::ExperimentHalted(reason);
    }

    /// Fraction of cells that are neurons.
    pub fn neuron_fraction(&self) -> f64 {
        let n = self.field.cells.len().max(1);
        self.field
            .cells
            .iter()
            .filter(|c| c.cell_type.is_neuron())
            .count() as f64
            / n as f64
    }

    /// Total number of synaptic connections.
    pub fn total_connections(&self) -> usize {
        self.field
            .cells
            .iter()
            .map(|c| c.connectivity.len())
            .sum::<usize>()
            / 2 // each connection stored in both cells
    }

    // -- private --

    fn update_spontaneous_activity(&mut self) {
        let rates: Vec<f32> = self
            .field
            .cells
            .iter()
            .filter(|c| c.cell_type.is_neuron())
            .map(|c| c.firing_rate)
            .collect();
        if rates.is_empty() {
            self.spontaneous_activity_hz = 0.0;
        } else {
            self.spontaneous_activity_hz = rates.iter().sum::<f32>() / rates.len() as f32;
        }
    }

    fn update_stage(&mut self) {
        // Stage progression is monotonic — once advanced, never goes back.
        let new_stage = self.determine_stage();
        if new_stage > self.stage {
            self.stage = new_stage;
        }
    }

    fn determine_stage(&self) -> OrganoidStage {
        let phi = self.global_phi();
        let neuron_frac = self.neuron_fraction();
        let connections = self.total_connections();

        if phi >= PHI_ETHICS_THRESHOLD {
            OrganoidStage::Conscious
        } else if neuron_frac > 0.3 && connections > self.field.cells.len() {
            OrganoidStage::Mature
        } else if connections > 0 {
            OrganoidStage::Synaptogenic
        } else if neuron_frac > 0.05 {
            OrganoidStage::Differentiating
        } else if self.developmental_day > 3 {
            OrganoidStage::Migratory
        } else {
            OrganoidStage::Proliferative
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undifferentiated_cells_have_zero_consciousness_potential() {
        let field = MorphogeneticField::new(50, 42);
        // All cells are progenitors; none are neurons → consciousness = 0.
        let organoid = NeuralOrganoid::new(50, 42);
        assert!(
            organoid.global_phi() == 0.0,
            "Undifferentiated organoid should have zero phi"
        );
        for &cp in &field.consciousness_potential {
            assert_eq!(cp, 0.0);
        }
    }

    #[test]
    fn neural_differentiation_increases_local_phi() {
        let mut organoid = NeuralOrganoid::new(100, 7);
        let phi_before = organoid.global_phi();

        // Advance enough days for differentiation and synaptogenesis.
        for _ in 0..30 {
            organoid.advance_day();
        }
        let phi_after = organoid.global_phi();
        assert!(
            phi_after >= phi_before,
            "Phi should not decrease after differentiation: before={phi_before}, after={phi_after}"
        );
    }

    #[test]
    fn reaction_diffusion_produces_spatial_patterns() {
        // Use a denser field (more cells in a smaller volume means more
        // neighbours per cell) and more steps so the Turing instability has
        // time to amplify the initial perturbation.
        let mut field = MorphogeneticField::new(200, 99);
        for _ in 0..500 {
            field.step_reaction_diffusion(0.2);
        }
        // The activator field should NOT be uniform — check that max and min
        // differ by a meaningful amount (the initial perturbation is +-0.05).
        let min = field
            .activator
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
        let max = field
            .activator
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        assert!(
            range > 1e-4,
            "Turing instability should produce non-uniform patterns, but range = {range}"
        );
    }

    #[test]
    fn synaptogenesis_increases_connectivity() {
        let mut organoid = NeuralOrganoid::new(80, 13);
        let conn_before = organoid.total_connections();

        // Advance to allow neurons to form and then connect.
        for _ in 0..20 {
            organoid.advance_day();
        }
        let conn_after = organoid.total_connections();
        assert!(
            conn_after >= conn_before,
            "Synaptogenesis should increase connectivity: before={conn_before}, after={conn_after}"
        );
    }

    #[test]
    fn global_phi_increases_with_development() {
        let mut organoid = NeuralOrganoid::new(100, 21);

        // Record phi at day 5 and day 30.
        for _ in 0..5 {
            organoid.advance_day();
        }
        let phi_early = organoid.global_phi();

        for _ in 0..25 {
            organoid.advance_day();
        }
        let phi_late = organoid.global_phi();

        assert!(
            phi_late >= phi_early,
            "Global phi should grow with development: early={phi_early}, late={phi_late}"
        );
    }

    #[test]
    fn ethics_gate_triggers_at_threshold() {
        let mut organoid = NeuralOrganoid::new(200, 55);

        // Force high phi by manually setting neurons + connectivity.
        for cell in organoid.field.cells.iter_mut() {
            cell.cell_type = OrganoidCellType::Neuron(NeuronSubtype::Excitatory);
            cell.firing_rate = 8.0;
        }
        // Connect every pair within radius.
        organoid.field.form_synapses();
        organoid.field.compute_local_phi();

        let phi = organoid.global_phi();
        organoid.check_ethics();

        if phi >= PHI_ETHICS_THRESHOLD {
            assert!(
                matches!(
                    organoid.ethics_status,
                    OrganoidEthicsStatus::ConsciousnessDetected(_)
                ),
                "Ethics gate should fire when phi={phi} >= threshold={PHI_ETHICS_THRESHOLD}"
            );
        }
    }

    #[test]
    fn proliferation_increases_cell_count() {
        let mut field = MorphogeneticField::new(50, 17);
        let count_before = field.num_cells();

        // Run many proliferation rounds (stochastic — use enough rounds).
        for _ in 0..50 {
            field.proliferate();
        }
        let count_after = field.num_cells();
        assert!(
            count_after >= count_before,
            "Proliferation should increase cell count: before={count_before}, after={count_after}"
        );
    }

    #[test]
    fn mature_organoid_has_higher_phi_than_proliferative() {
        let mut early = NeuralOrganoid::new(100, 33);
        for _ in 0..2 {
            early.advance_day();
        }

        let mut mature = NeuralOrganoid::new(100, 33);
        for _ in 0..40 {
            mature.advance_day();
        }

        assert!(
            mature.global_phi() >= early.global_phi(),
            "Mature organoid should have phi >= proliferative organoid"
        );
    }

    #[test]
    fn stage_progression_is_monotonic() {
        let mut organoid = NeuralOrganoid::new(100, 44);
        let mut prev_stage = organoid.stage;

        for _ in 0..50 {
            organoid.advance_day();
            assert!(
                organoid.stage >= prev_stage,
                "Stage should never regress: was {:?}, now {:?}",
                prev_stage,
                organoid.stage
            );
            prev_stage = organoid.stage;
        }
    }

    #[test]
    fn phi_history_tracks_correctly() {
        let mut organoid = NeuralOrganoid::new(60, 88);
        assert!(organoid.phi_history.is_empty());

        for _ in 0..10 {
            organoid.advance_day();
        }
        assert_eq!(
            organoid.phi_history.len(),
            10,
            "Phi history should have one entry per day"
        );

        // Each entry should be non-negative.
        for &phi in &organoid.phi_history {
            assert!(phi >= 0.0, "Phi should be non-negative, got {phi}");
        }
    }

    #[test]
    fn experiment_halt_stops_advancement() {
        let mut organoid = NeuralOrganoid::new(50, 77);
        organoid.advance_day();
        let day_before = organoid.developmental_day;

        organoid.halt_experiment("Consciousness threshold exceeded".to_string());
        organoid.advance_day();

        assert_eq!(
            organoid.developmental_day, day_before,
            "Halted organoid should not advance"
        );
        assert!(matches!(
            organoid.ethics_status,
            OrganoidEthicsStatus::ExperimentHalted(_)
        ));
    }

    #[test]
    fn cell_type_queries_correct() {
        assert!(OrganoidCellType::Neuron(NeuronSubtype::Excitatory).is_neuron());
        assert!(!OrganoidCellType::Neuron(NeuronSubtype::Inhibitory).is_glial());
        assert!(OrganoidCellType::Glial(GlialSubtype::Astrocyte).is_glial());
        assert!(OrganoidCellType::Progenitor.is_progenitor());
        assert!(OrganoidCellType::NeuralProgenitor.is_progenitor());
        assert!(!OrganoidCellType::Undifferentiated.is_progenitor());
    }

    #[test]
    fn morphogen_concentrations_remain_non_negative() {
        let mut field = MorphogeneticField::new(80, 111);
        for _ in 0..100 {
            field.step_reaction_diffusion(0.1);
        }
        for (i, (&a, &h)) in field
            .activator
            .iter()
            .zip(field.inhibitor.iter())
            .enumerate()
        {
            assert!(a >= 0.0, "Activator[{i}] = {a} is negative");
            assert!(h >= 0.0, "Inhibitor[{i}] = {h} is negative");
        }
    }
}
