// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bioelectric morphogenesis: gap-junction-coupled transmembrane voltage
//! (Vmem), wound-induced depolarization, basal (non-neural) information
//! integration, and target-morphology homeostasis / regeneration.
//!
//! [`crate::morphogenetic_consciousness`] implements a classical Turing
//! (1952) chemical reaction-diffusion model. That is real pattern formation,
//! but it is not the whole story of morphogenesis — and it is specifically
//! not the part of the story most associated with Michael Levin's lab. Their
//! central, repeatedly-replicated finding is that anatomical patterning is
//! also instructed by a *distinct, faster, longer-range* bioelectric signal:
//! networks of cells electrically coupled through gap junctions (connexins)
//! propagate transmembrane voltage (Vmem) patterns that encode a kind of
//! "memory" of target anatomy — separable from both the genome and from
//! chemical morphogen gradients. Famous demonstrations: planaria regenerate
//! the *correct* head/tail regardless of fragment origin (a stored target
//! pattern, not local default rules); blocking or altering gap-junction
//! coupling can produce two-headed flatworms or eyes on a tadpole's tail
//! without touching the genome; xenobots/anthrobots show novel,
//! genome-unspecified anatomy and behavior emerging from self-organizing
//! cell collectives. This module adds that electrical layer and couples it,
//! causally, back into differentiation — so gap-junction state, not just
//! gene expression or chemical gradient, has real explanatory power here.
//!
//! It also implements a *target morphology* homeostat: a captured goal
//! pattern the tissue is regenerated toward after damage, plus wound-induced
//! depolarization as the trigger for regenerative proliferation — the
//! signature Levin-lab phenomenon (equifinality: many different starting
//! configurations converge on the same target anatomy).
//!
//! Finally, `compute_local_phi` in the sibling module no longer gates all
//! information integration on `is_neuron()`. This module's
//! [`BioelectricState::basal_information`] gives every gap-junction-coupled
//! cell — not just neurons — a (small, capped) information-integration
//! contribution, reflecting Levin's TAME (Technological Approach to Mind
//! Everywhere) claim that cognition-like problem-solving is scale-free and
//! not exclusive to neural tissue.
//!
//! References:
//! - Levin, M. (2014). Molecular bioelectricity: how endogenous voltage
//!   potentials control cell behavior and instruct pattern regulation in
//!   vivo. Mol. Biol. Cell 25(24):3835-3850.
//! - Levin, M. & Martyniuk, C. J. (2018). The bioelectric code: an
//!   ancient computational medium for dynamic control of growth and form.
//!   Biosystems 164:76-93.
//! - Levin, M. (2019). The computational boundary of a self: developmental
//!   bioelectricity drives multicellularity and scale-free cognition.
//!   Front. Psychol. 10:2688.
//! - Pezzulo, G. & Levin, M. (2016). Top-down models in biology: explanation
//!   and control of complex living systems above the molecular level.
//!   J. R. Soc. Interface 13:20160555. (active-inference framing of
//!   anatomical homeostasis toward a target morphology)
//! - Durant, F. et al. (2017). The role of early bioelectric signals in the
//!   regeneration of planarian anterior/posterior polarity.
//! - Kriegman, S., Blackiston, D., Levin, M., Bongard, J. (2020). A scalable
//!   pipeline for designing reconfigurable organisms. PNAS 117(4):1853-1859.
//!   (xenobots)

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::morphogenetic_consciousness::{
    MAX_CELLS, MorphogeneticField, NeuralOrganoid, OrganoidCellType,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Resting Vmem for depolarized / proliferative-stem-like tissue (normalized
/// units — this model does not claim literal millivolts).
pub const VMEM_DEPOLARIZED: f32 = 0.0;
/// Resting Vmem for hyperpolarized / differentiated tissue.
pub const VMEM_HYPERPOLARIZED: f32 = -1.0;
/// Transient depolarization spike applied to wound-boundary cells
/// immediately after amputation — mirrors the wound-induced Vmem change
/// documented as an early regenerative trigger (Durant et al. 2017 and
/// related planarian/Xenopus tail-regeneration work).
pub const VMEM_WOUND_SPIKE: f32 = 0.6;

/// Spatial radius for gap-junction formation (same order as the chemical
/// field's neighbourhood radius, but electrical coupling is *unrestricted by
/// cell type* — unlike synaptogenesis, which only connects neurons).
const GAP_JUNCTION_RADIUS: f32 = 0.15;
/// Electrical (gap-junction) diffusion is much faster than chemical
/// diffusion — this is deliberately larger than the Turing model's D_a/D_h.
const GAP_JUNCTION_DIFFUSION_RATE: f32 = 0.35;
/// Per-day drift rate pulling Vmem toward the fate-appropriate resting
/// potential (hyperpolarized once differentiated, depolarized otherwise).
const HYPERPOLARIZATION_DRIFT: f32 = 0.05;
/// Weight of the Vmem-hyperpolarization bias added into the neural gene
/// expression signal during differentiation.
const VMEM_DIFFERENTIATION_DRIVE: f32 = 0.20;

/// Weight applied to the basal (non-neural) information-integration term
/// before it is folded into `local_phi`. Kept small relative to the neural
/// term intentionally — see module docs.
const BASAL_PHI_WEIGHT: f64 = 1.0;
/// Hard cap on the basal contribution to `local_phi`, well below
/// `PHI_ETHICS_THRESHOLD` (0.3) so basal/non-neural tissue alone can never
/// trip the consciousness ethics gate.
const BASAL_PHI_CAP: f64 = 0.08;

/// Number of radial shells `TargetMorphology` buckets the organoid into.
const N_SHELLS: usize = 4;
/// Radius beyond which all cells are bucketed into the outermost shell.
const SHELL_MAX_RADIUS: f32 = 1.3;
/// Per-day probability a wound-boundary progenitor divides during active
/// regeneration — deliberately higher than the baseline `PROLIFERATION_RATE`
/// to model blastema-like accelerated growth at the wound site.
const REGENERATION_PROLIFERATION_BOOST: f64 = 0.25;
/// RMS shell-composition discrepancy below which regeneration is considered
/// converged (the wound is "healed").
const MORPHOLOGY_CONVERGENCE_TOLERANCE: f64 = 0.05;
/// Days after which regeneration bookkeeping times out even if the target
/// was never reached (avoids an unbounded regenerating state).
const REGENERATION_TIMEOUT_DAYS: u32 = 200;

fn default_bio_rng() -> StdRng {
    StdRng::seed_from_u64(0)
}

// ---------------------------------------------------------------------------
// BioelectricState
// ---------------------------------------------------------------------------

/// Per-cell transmembrane voltage (Vmem) and the gap-junction network that
/// couples it across the tissue.
///
/// Distinct from [`MorphogeneticField::connectivity_matrix`] (synaptic,
/// neuron-only) in two ways: gap junctions couple *any* pair of nearby
/// cells regardless of type, and `gap_junction_permeability` is a single
/// global knob that can be closed (pharmacological blocker, e.g. octanol /
/// lanthanum in Levin's experiments) or opened without touching gene
/// expression — the mechanism by which this model can demonstrate
/// bioelectrically-caused, genetically-identical developmental divergence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioelectricState {
    /// Per-cell transmembrane voltage, normalized so 0.0 = depolarized
    /// (proliferative/stem-like) and -1.0 = hyperpolarized (differentiated).
    pub vmem: Vec<f32>,
    /// Gap-junction coupling weight matrix. `gap_junction_matrix[i][j]` is
    /// the electrical coupling strength between cells i and j.
    pub gap_junction_matrix: Vec<Vec<f64>>,
    /// Global gap-junction permeability multiplier in `[0, 1]`. 1.0 = fully
    /// open (control), 0.0 = fully pharmacologically blocked. Existing
    /// gap-junction *connections* are not severed by blocking (matching real
    /// blockers, which close channels without destroying them) — only the
    /// signal they carry is attenuated.
    pub gap_junction_permeability: f32,
    /// Cells currently adjacent to an unhealed amputation wound. Cleared
    /// once regeneration converges (see [`NeuralOrganoid::advance_regeneration`]).
    pub wound_boundary: Vec<bool>,
    /// Seeded RNG, independent of [`MorphogeneticField`]'s own RNG so that
    /// adding this layer does not perturb the existing chemical/synaptic
    /// model's stochastic sequence (and therefore does not change any
    /// pre-existing test's outcome).
    #[serde(skip, default = "default_bio_rng")]
    rng: StdRng,
}

impl BioelectricState {
    /// Create a new bioelectric state for `n` cells, all initialized to the
    /// depolarized (proliferative/stem-like) resting potential.
    pub fn new(n: usize, seed: u64) -> Self {
        Self {
            vmem: vec![VMEM_DEPOLARIZED; n],
            gap_junction_matrix: vec![vec![0.0f64; n]; n],
            gap_junction_permeability: 1.0,
            wound_boundary: vec![false; n],
            rng: StdRng::seed_from_u64(seed.wrapping_add(0xB10E_1EC7)),
        }
    }

    /// Grow all per-cell arrays to `total` entries (called after
    /// proliferation adds new cells). New cells start depolarized and
    /// unconnected.
    pub(crate) fn resize(&mut self, total: usize) {
        let old = self.vmem.len();
        self.vmem.resize(total, VMEM_DEPOLARIZED);
        self.wound_boundary.resize(total, false);
        for row in self.gap_junction_matrix.iter_mut() {
            row.resize(total, 0.0);
        }
        for _ in old..total {
            self.gap_junction_matrix.push(vec![0.0; total]);
        }
    }

    /// Rebuild all per-cell arrays after amputation removes cells.
    /// `keep[i]` is whether old index `i` survived; `new_index[i]` is its
    /// position in the post-amputation arrays.
    pub(crate) fn amputate_reindex(&mut self, keep: &[bool], new_index: &[usize], next: usize) {
        let n = keep.len();
        let mut new_vmem = Vec::with_capacity(next);
        let mut new_wound = Vec::with_capacity(next);
        for i in 0..n {
            if keep[i] {
                new_vmem.push(self.vmem[i]);
                new_wound.push(self.wound_boundary[i]);
            }
        }
        let mut new_gj = vec![vec![0.0f64; next]; next];
        for i in 0..n {
            if !keep[i] {
                continue;
            }
            for j in 0..n {
                if !keep[j] {
                    continue;
                }
                new_gj[new_index[i]][new_index[j]] = self.gap_junction_matrix[i][j];
            }
        }
        self.vmem = new_vmem;
        self.wound_boundary = new_wound;
        self.gap_junction_matrix = new_gj;
    }

    /// Bias added to a cell's effective neural gene expression during
    /// differentiation, derived from how hyperpolarized its Vmem currently
    /// is. Scaled by `gap_junction_permeability`: when gap junctions are
    /// blocked, the collective Vmem pattern can't propagate/inform fate, so
    /// this bias vanishes even though the cell's own instantaneous Vmem is
    /// unchanged — this is what makes the blocker experiment (see
    /// `NeuralOrganoid::set_gap_junction_permeability`) a genuine test of
    /// bioelectric *coupling*, not just of raw voltage.
    pub(crate) fn differentiation_bias(&self, idx: usize) -> f32 {
        let v = self.vmem[idx];
        let hyperpolarization =
            ((VMEM_DEPOLARIZED - v) / (VMEM_DEPOLARIZED - VMEM_HYPERPOLARIZED)).clamp(0.0, 1.0);
        hyperpolarization * VMEM_DIFFERENTIATION_DRIVE * self.gap_junction_permeability
    }
}

// ---------------------------------------------------------------------------
// MorphogeneticField: gap junctions, Vmem diffusion, basal information,
// amputation, regenerative proliferation
// ---------------------------------------------------------------------------

impl MorphogeneticField {
    /// Form gap junctions between any two cells within [`GAP_JUNCTION_RADIUS`]
    /// (any cell type, unlike `form_synapses` which is neuron-only).
    /// Existing zero-weight pairs are (re-)attempted each call so a blocked
    /// network can "heal" once permeability is restored.
    pub fn form_gap_junctions(&mut self) {
        let n = self.cells.len();
        let permeability = self.bioelectric.gap_junction_permeability as f64;
        for i in 0..n {
            let pos_i = self.cells[i].position;
            for j in (i + 1)..n {
                if Self::distance(&pos_i, &self.cells[j].position) < GAP_JUNCTION_RADIUS
                    && self.bioelectric.gap_junction_matrix[i][j] == 0.0
                {
                    let w = self.bioelectric.rng.gen_range(0.4..1.0f64) * permeability;
                    self.bioelectric.gap_junction_matrix[i][j] = w;
                    self.bioelectric.gap_junction_matrix[j][i] = w;
                }
            }
        }
    }

    /// One step of gap-junction-mediated Vmem equilibration, plus a
    /// fate-dependent drift toward the appropriate resting potential
    /// (hyperpolarized once differentiated, depolarized otherwise).
    pub fn step_vmem_diffusion(&mut self, dt: f32) {
        let n = self.cells.len();
        let permeability = self.bioelectric.gap_junction_permeability;
        let mut new_vmem = self.bioelectric.vmem.clone();
        for i in 0..n {
            let row = &self.bioelectric.gap_junction_matrix[i];
            let mut weighted_sum = 0.0f64;
            let mut weight_total = 0.0f64;
            for (j, &w) in row.iter().enumerate() {
                if w > 0.0 {
                    weighted_sum += w * self.bioelectric.vmem[j] as f64;
                    weight_total += w;
                }
            }

            let mut v = self.bioelectric.vmem[i];
            if weight_total > 0.0 {
                let avg = (weighted_sum / weight_total) as f32;
                v += dt * GAP_JUNCTION_DIFFUSION_RATE * permeability * (avg - v);
            }

            let target =
                if self.cells[i].cell_type.is_neuron() || self.cells[i].cell_type.is_glial() {
                    VMEM_HYPERPOLARIZED
                } else {
                    VMEM_DEPOLARIZED
                };
            v += dt * HYPERPOLARIZATION_DRIFT * (target - v);

            new_vmem[i] = v;
        }
        self.bioelectric.vmem = new_vmem;
    }

    /// Basal (non-neural) information-integration measure at cell `idx`,
    /// derived from Vmem variance across *all* gap-junction neighbours
    /// (not just neurons), weighted by gap-junction connectivity density —
    /// the electrical-layer analog of `compute_local_phi`'s neural term.
    /// Capped at [`BASAL_PHI_CAP`]. See module docs for why this exists.
    pub(crate) fn basal_information(&self, idx: usize) -> f64 {
        let nbrs = self.neighbours(idx, GAP_JUNCTION_RADIUS);
        if nbrs.len() < 2 {
            return 0.0;
        }

        let vmems: Vec<f64> = nbrs
            .iter()
            .map(|&j| self.bioelectric.vmem[j] as f64)
            .collect();
        let connected = nbrs
            .iter()
            .filter(|&&j| self.bioelectric.gap_junction_matrix[idx][j] > 0.0)
            .count();
        let density = connected as f64 / nbrs.len() as f64;

        let mean = vmems.iter().sum::<f64>() / vmems.len() as f64;
        let var = vmems.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vmems.len() as f64;

        (var.sqrt() * density * BASAL_PHI_WEIGHT).min(BASAL_PHI_CAP)
    }

    /// Remove all cells whose radial position (distance from the organoid
    /// centre) falls in `[min_r, max_r)`, simulating cutting a fragment off
    /// the organoid (à la planarian transection). Reindexes every per-cell
    /// array and the connectivity/gap-junction matrices, then marks
    /// surviving cells adjacent to the cut as `wound_boundary` and applies a
    /// transient depolarizing "wound spike" to their Vmem. Returns the
    /// number of cells removed.
    pub fn amputate(&mut self, min_r: f32, max_r: f32) -> usize {
        let n = self.cells.len();
        if n == 0 {
            return 0;
        }
        let radius_of = |p: &[f32; 3]| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();

        let keep: Vec<bool> = (0..n)
            .map(|i| {
                let r = radius_of(&self.cells[i].position);
                !(r >= min_r && r < max_r)
            })
            .collect();
        let removed = keep.iter().filter(|k| !**k).count();
        if removed == 0 {
            return 0;
        }

        let removed_positions: Vec<[f32; 3]> = (0..n)
            .filter(|&i| !keep[i])
            .map(|i| self.cells[i].position)
            .collect();

        let mut new_index = vec![usize::MAX; n];
        let mut next = 0usize;
        for i in 0..n {
            if keep[i] {
                new_index[i] = next;
                next += 1;
            }
        }

        let mut new_cells = Vec::with_capacity(next);
        let mut new_activator = Vec::with_capacity(next);
        let mut new_inhibitor = Vec::with_capacity(next);
        let mut new_cp = Vec::with_capacity(next);
        let mut new_local_phi = Vec::with_capacity(next);
        for i in 0..n {
            if !keep[i] {
                continue;
            }
            let mut cell = self.cells[i].clone();
            cell.connectivity = cell
                .connectivity
                .iter()
                .filter(|&&j| keep[j])
                .map(|&j| new_index[j])
                .collect();
            new_cells.push(cell);
            new_activator.push(self.activator[i]);
            new_inhibitor.push(self.inhibitor[i]);
            new_cp.push(self.consciousness_potential[i]);
            new_local_phi.push(self.local_phi[i]);
        }

        let mut new_conn = vec![vec![0.0f64; next]; next];
        for i in 0..n {
            if !keep[i] {
                continue;
            }
            for j in 0..n {
                if !keep[j] {
                    continue;
                }
                new_conn[new_index[i]][new_index[j]] = self.connectivity_matrix[i][j];
            }
        }

        self.cells = new_cells;
        self.activator = new_activator;
        self.inhibitor = new_inhibitor;
        self.consciousness_potential = new_cp;
        self.local_phi = new_local_phi;
        self.connectivity_matrix = new_conn;
        self.bioelectric.amputate_reindex(&keep, &new_index, next);

        for idx in 0..self.cells.len() {
            let pos = self.cells[idx].position;
            let near_wound = removed_positions
                .iter()
                .any(|p| Self::distance(&pos, p) < GAP_JUNCTION_RADIUS);
            if near_wound {
                self.bioelectric.wound_boundary[idx] = true;
                self.bioelectric.vmem[idx] = VMEM_WOUND_SPIKE;
            }
        }

        removed
    }

    /// Extra proliferation pass restricted to wound-boundary progenitors,
    /// at a boosted rate — models blastema-like accelerated regenerative
    /// growth. Called from [`NeuralOrganoid::advance_regeneration`].
    pub(crate) fn regenerative_proliferate(&mut self) {
        let n = self.cells.len();
        if n >= MAX_CELLS {
            return;
        }
        let mut new_cells = Vec::new();
        for i in 0..n {
            if !self.bioelectric.wound_boundary[i] || !self.cells[i].cell_type.is_progenitor() {
                continue;
            }
            if self
                .bioelectric
                .rng
                .gen_bool(REGENERATION_PROLIFERATION_BOOST.min(1.0))
            {
                let mut daughter = self.cells[i].clone();
                for d in 0..3 {
                    daughter.position[d] += self.bioelectric.rng.gen_range(-0.02..0.02f32);
                }
                daughter.age_days = 0;
                daughter.connectivity.clear();
                new_cells.push(daughter);
                if n + new_cells.len() >= MAX_CELLS {
                    break;
                }
            }
        }
        let added = new_cells.len();
        if added == 0 {
            return;
        }
        self.cells.extend(new_cells);
        let total = self.cells.len();
        self.activator.resize(total, 1.0);
        self.inhibitor.resize(total, 1.0);
        self.consciousness_potential.resize(total, 0.0);
        self.local_phi.resize(total, 0.0);
        for row in self.connectivity_matrix.iter_mut() {
            row.resize(total, 0.0);
        }
        for _ in 0..added {
            self.connectivity_matrix.push(vec![0.0; total]);
        }
        self.bioelectric.resize(total);
        // Daughters at the wound inherit the boundary flag and stay
        // depolarized, so they keep participating in regeneration.
        for idx in (total - added)..total {
            self.bioelectric.wound_boundary[idx] = true;
            self.bioelectric.vmem[idx] = VMEM_WOUND_SPIKE;
        }
    }
}

// ---------------------------------------------------------------------------
// TargetMorphology
// ---------------------------------------------------------------------------

/// A captured "goal" anatomical pattern: the fraction of each coarse cell
/// type ([undifferentiated, progenitor, neuron, glial]) within each of
/// [`N_SHELLS`] radial shells of the organoid.
///
/// This is a deliberately coarse stand-in for what Levin's lab calls the
/// bioelectric "target morphology" a tissue's collective intelligence
/// navigates toward and maintains (equifinality / regeneration to a stored
/// anatomical setpoint, independent of which specific cells survived). A
/// full model would key this off Vmem pattern directly; this one uses
/// realized cell-type composition, which is downstream of Vmem in this
/// simulation and easier to validate against.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetMorphology {
    /// `shell_composition[shell][bucket]` — bucket 0=undifferentiated,
    /// 1=progenitor (incl. neural progenitor), 2=neuron, 3=glial.
    pub shell_composition: [[f32; 4]; N_SHELLS],
}

fn shell_index(radius: f32) -> usize {
    let r = radius.clamp(0.0, SHELL_MAX_RADIUS);
    let idx = (r / SHELL_MAX_RADIUS * N_SHELLS as f32) as usize;
    idx.min(N_SHELLS - 1)
}

fn cell_type_bucket(ct: OrganoidCellType) -> usize {
    match ct {
        OrganoidCellType::Undifferentiated => 0,
        OrganoidCellType::Progenitor | OrganoidCellType::NeuralProgenitor => 1,
        OrganoidCellType::Neuron(_) => 2,
        OrganoidCellType::Glial(_) => 3,
    }
}

impl TargetMorphology {
    /// Snapshot the current cell-type-by-shell composition of `field` as a
    /// goal pattern.
    pub fn capture(field: &MorphogeneticField) -> Self {
        let mut counts = [[0u32; 4]; N_SHELLS];
        let mut totals = [0u32; N_SHELLS];
        for cell in &field.cells {
            let r =
                (cell.position[0].powi(2) + cell.position[1].powi(2) + cell.position[2].powi(2))
                    .sqrt();
            let s = shell_index(r);
            let b = cell_type_bucket(cell.cell_type);
            counts[s][b] += 1;
            totals[s] += 1;
        }
        let mut shell_composition = [[0.0f32; 4]; N_SHELLS];
        for s in 0..N_SHELLS {
            if totals[s] == 0 {
                continue;
            }
            for b in 0..4 {
                shell_composition[s][b] = counts[s][b] as f32 / totals[s] as f32;
            }
        }
        Self { shell_composition }
    }

    /// RMS distance between this target's shell composition and `field`'s
    /// current shell composition. 0.0 = exact match.
    pub fn discrepancy(&self, field: &MorphogeneticField) -> f64 {
        let current = Self::capture(field);
        let mut sum_sq = 0.0f64;
        let mut count = 0usize;
        for s in 0..N_SHELLS {
            for b in 0..4 {
                let d = (self.shell_composition[s][b] - current.shell_composition[s][b]) as f64;
                sum_sq += d * d;
                count += 1;
            }
        }
        (sum_sq / count as f64).sqrt()
    }
}

// ---------------------------------------------------------------------------
// NeuralOrganoid: regeneration API
// ---------------------------------------------------------------------------

impl NeuralOrganoid {
    /// Capture the organoid's current cell-type-by-shell composition as its
    /// target morphology — the pattern subsequent regeneration (after
    /// `amputate`) will be driven toward.
    pub fn capture_target_morphology(&mut self) {
        self.target_morphology = Some(TargetMorphology::capture(&self.field));
    }

    /// Cut away all cells in the radial shell `[min_r, max_r)`, simulating
    /// amputation of part of the organoid. Resets the regeneration clock.
    /// Returns the number of cells removed.
    pub fn amputate(&mut self, min_r: f32, max_r: f32) -> usize {
        let removed = self.field.amputate(min_r, max_r);
        if removed > 0 {
            self.days_since_wound = 0;
        }
        removed
    }

    /// RMS discrepancy between the captured target morphology and the
    /// organoid's current composition, or `None` if no target has been
    /// captured.
    pub fn morphology_discrepancy(&self) -> Option<f64> {
        self.target_morphology
            .as_ref()
            .map(|t| t.discrepancy(&self.field))
    }

    /// Whether the organoid is actively regenerating (has a target
    /// morphology and hasn't timed out).
    pub fn is_regenerating(&self) -> bool {
        self.target_morphology.is_some() && self.days_since_wound < REGENERATION_TIMEOUT_DAYS
    }

    /// Set the global gap-junction permeability (`[0, 1]`) — 0.0 models a
    /// pharmacological gap-junction blocker (e.g. octanol / lanthanum in
    /// Levin's experiments), applied without changing any cell's gene
    /// expression. Use this to demonstrate bioelectrically-caused,
    /// genetically-identical developmental divergence: run two organoids
    /// from the same seed, block one, and compare trajectories.
    pub fn set_gap_junction_permeability(&mut self, permeability: f32) {
        self.field.bioelectric.gap_junction_permeability = permeability.clamp(0.0, 1.0);
    }

    /// Current global gap-junction permeability.
    pub fn gap_junction_permeability(&self) -> f32 {
        self.field.bioelectric.gap_junction_permeability
    }

    /// Advance the regeneration homeostat by one day: while regenerating and
    /// not yet converged, run an extra wound-boundary proliferation pass;
    /// once the shell composition is within
    /// [`MORPHOLOGY_CONVERGENCE_TOLERANCE`] of the target, clear the wound.
    /// Called automatically from `advance_day`.
    pub(crate) fn advance_regeneration(&mut self) {
        if self.target_morphology.is_none() {
            return;
        }
        if self.is_regenerating() {
            let discrepancy = self.morphology_discrepancy().unwrap_or(0.0);
            if discrepancy > MORPHOLOGY_CONVERGENCE_TOLERANCE {
                self.field.regenerative_proliferate();
            } else {
                for w in self.field.bioelectric.wound_boundary.iter_mut() {
                    *w = false;
                }
            }
        }
        self.days_since_wound = self.days_since_wound.saturating_add(1);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphogenetic_consciousness::NeuronSubtype;

    #[test]
    fn gap_junctions_form_between_nearby_cells() {
        let mut field = MorphogeneticField::new(60, 5);
        field.form_gap_junctions();
        let any_connected = field
            .bioelectric
            .gap_junction_matrix
            .iter()
            .flatten()
            .any(|&w| w > 0.0);
        assert!(
            any_connected,
            "Dense random field should form at least one gap junction"
        );
    }

    #[test]
    fn vmem_diffusion_equalizes_connected_cells() {
        let mut field = MorphogeneticField::new(2, 1);
        field.cells[0].position = [0.0, 0.0, 0.0];
        field.cells[1].position = [0.05, 0.0, 0.0];
        field.bioelectric.vmem[0] = 0.0;
        field.bioelectric.vmem[1] = -1.0;
        field.form_gap_junctions();
        assert!(
            field.bioelectric.gap_junction_matrix[0][1] > 0.0,
            "Cells 0.05 apart should be within the gap-junction radius"
        );

        let diff_before = (field.bioelectric.vmem[0] - field.bioelectric.vmem[1]).abs();
        for _ in 0..30 {
            field.step_vmem_diffusion(0.2);
        }
        let diff_after = (field.bioelectric.vmem[0] - field.bioelectric.vmem[1]).abs();
        assert!(
            diff_after < diff_before,
            "Gap-junction-coupled Vmem should equalize: before={diff_before}, after={diff_after}"
        );
    }

    #[test]
    fn gap_junction_blocker_reduces_vmem_equalization_relative_to_open() {
        // Both cells are Progenitors, so the fate-drift term in
        // `step_vmem_diffusion` pulls *both* of them toward the same
        // (depolarized) resting target regardless of gap-junction state —
        // that alone causes some convergence even when blocked. So this test
        // isolates the *incremental* contribution of gap-junction coupling
        // by comparing open vs. blocked under otherwise identical dynamics,
        // rather than asserting blocked produces literally no movement.
        let final_diff = |permeability: f32| {
            let mut field = MorphogeneticField::new(2, 1);
            field.cells[0].position = [0.0, 0.0, 0.0];
            field.cells[1].position = [0.05, 0.0, 0.0];
            field.bioelectric.vmem[0] = 0.0;
            field.bioelectric.vmem[1] = -1.0;
            field.bioelectric.gap_junction_permeability = permeability;
            field.form_gap_junctions();
            for _ in 0..30 {
                field.step_vmem_diffusion(0.2);
            }
            (field.bioelectric.vmem[0] - field.bioelectric.vmem[1]).abs()
        };

        let diff_open = final_diff(1.0);
        let diff_blocked = final_diff(0.0);
        assert!(
            diff_open < diff_blocked,
            "Open gap junctions should equalize Vmem more than blocked ones: \
             open={diff_open}, blocked={diff_blocked}"
        );
    }

    #[test]
    fn basal_information_requires_gap_junction_connectivity() {
        let field = MorphogeneticField::new(60, 9);
        // Fresh field: no gap junctions formed yet.
        for i in 0..field.num_cells() {
            assert_eq!(
                field.basal_information(i),
                0.0,
                "No gap junctions formed yet, basal information should be zero"
            );
        }
    }

    /// Build a tightly-clustered field (every pair well within
    /// `GAP_JUNCTION_RADIUS`) with alternating Vmem, so `basal_information`
    /// tests aren't at the mercy of sparse random spatial packing.
    fn clustered_heterogeneous_vmem_field(n: usize, seed: u64) -> MorphogeneticField {
        let mut field = MorphogeneticField::new(n, seed);
        for (i, cell) in field.cells.iter_mut().enumerate() {
            cell.position = [0.01 * i as f32, 0.0, 0.0];
        }
        for (i, v) in field.bioelectric.vmem.iter_mut().enumerate() {
            *v = if i % 2 == 0 {
                VMEM_DEPOLARIZED
            } else {
                VMEM_HYPERPOLARIZED
            };
        }
        field.form_gap_junctions();
        field
    }

    #[test]
    fn basal_information_nonzero_after_gap_junctions_form_with_vmem_variance() {
        let field = clustered_heterogeneous_vmem_field(6, 21);
        let any_nonzero = (0..field.num_cells()).any(|i| field.basal_information(i) > 0.0);
        assert!(
            any_nonzero,
            "Basal information should be nonzero once gap junctions form across heterogeneous Vmem"
        );
    }

    #[test]
    fn basal_information_stays_below_ethics_relevant_magnitude() {
        let field = clustered_heterogeneous_vmem_field(6, 21);
        for i in 0..field.num_cells() {
            assert!(
                field.basal_information(i) <= BASAL_PHI_CAP,
                "Basal information must stay capped well below the consciousness ethics threshold"
            );
        }
    }

    #[test]
    fn differentiation_bias_zero_at_depolarized_resting_potential() {
        let field = MorphogeneticField::new(10, 3);
        // All cells start at VMEM_DEPOLARIZED by construction.
        for i in 0..field.num_cells() {
            assert_eq!(field.bioelectric.differentiation_bias(i), 0.0);
        }
    }

    #[test]
    fn differentiation_bias_positive_when_hyperpolarized() {
        let mut field = MorphogeneticField::new(1, 4);
        field.bioelectric.vmem[0] = VMEM_HYPERPOLARIZED;
        assert!(field.bioelectric.differentiation_bias(0) > 0.0);
    }

    #[test]
    fn differentiation_bias_vanishes_when_gap_junctions_blocked() {
        let mut field = MorphogeneticField::new(1, 4);
        field.bioelectric.vmem[0] = VMEM_HYPERPOLARIZED;
        field.bioelectric.gap_junction_permeability = 0.0;
        assert_eq!(
            field.bioelectric.differentiation_bias(0),
            0.0,
            "Blocked permeability should zero out the bioelectric differentiation bias"
        );
    }

    #[test]
    fn amputate_removes_correct_cell_count_and_reindexes_safely() {
        let mut field = MorphogeneticField::new(200, 12);
        for _ in 0..5 {
            field.step_reaction_diffusion(0.1);
        }
        field.form_synapses();
        field.form_gap_junctions();

        let before = field.num_cells();
        let removed = field.amputate(0.6, 2.0);
        assert!(
            removed > 0,
            "Amputating the outer shell should remove cells"
        );
        assert_eq!(field.num_cells(), before - removed);
        assert_eq!(field.bioelectric.vmem.len(), field.num_cells());
        assert_eq!(field.connectivity_matrix.len(), field.num_cells());
        assert_eq!(
            field.bioelectric.gap_junction_matrix.len(),
            field.num_cells()
        );

        // No dangling connectivity indices.
        for cell in &field.cells {
            for &j in &cell.connectivity {
                assert!(
                    j < field.num_cells(),
                    "Connectivity index out of bounds after amputation"
                );
            }
        }
    }

    #[test]
    fn amputate_marks_wound_boundary_and_spikes_vmem() {
        let mut field = MorphogeneticField::new(300, 13);
        field.amputate(0.6, 2.0);
        let any_wound = field.bioelectric.wound_boundary.iter().any(|&w| w);
        assert!(
            any_wound,
            "Amputation should mark at least one wound-boundary cell"
        );
        let any_spiked = field
            .bioelectric
            .vmem
            .iter()
            .any(|&v| v == VMEM_WOUND_SPIKE);
        assert!(
            any_spiked,
            "Wound-boundary cells should spike to VMEM_WOUND_SPIKE"
        );
    }

    #[test]
    fn amputate_of_empty_region_removes_nothing() {
        let mut field = MorphogeneticField::new(50, 14);
        let removed = field.amputate(10.0, 20.0);
        assert_eq!(removed, 0);
        assert_eq!(field.num_cells(), 50);
    }

    #[test]
    fn target_morphology_capture_has_zero_self_discrepancy() {
        let field = MorphogeneticField::new(120, 15);
        let target = TargetMorphology::capture(&field);
        assert!(target.discrepancy(&field) < 1e-6);
    }

    #[test]
    fn target_morphology_discrepancy_positive_after_amputation() {
        let mut organoid = NeuralOrganoid::new(150, 16);
        for _ in 0..40 {
            organoid.advance_day();
        }
        organoid.capture_target_morphology();
        let removed = organoid.amputate(0.6, 2.0);
        assert!(removed > 0);
        let discrepancy = organoid.morphology_discrepancy().unwrap();
        assert!(
            discrepancy > 0.0,
            "Removing a shell of tissue should create a nonzero morphology discrepancy"
        );
    }

    #[test]
    fn regeneration_reduces_morphology_discrepancy_over_time() {
        let mut organoid = NeuralOrganoid::new(150, 17);
        for _ in 0..40 {
            organoid.advance_day();
        }
        organoid.capture_target_morphology();
        organoid.amputate(0.6, 2.0);
        let discrepancy_right_after = organoid.morphology_discrepancy().unwrap();

        for _ in 0..40 {
            organoid.advance_day();
        }
        let discrepancy_later = organoid.morphology_discrepancy().unwrap();

        assert!(
            discrepancy_later <= discrepancy_right_after,
            "Regeneration should not increase discrepancy from target: right_after={discrepancy_right_after}, later={discrepancy_later}"
        );
    }

    #[test]
    fn is_regenerating_false_without_target() {
        let organoid = NeuralOrganoid::new(20, 18);
        assert!(!organoid.is_regenerating());
    }

    #[test]
    fn set_gap_junction_permeability_clamps_to_unit_interval() {
        let mut organoid = NeuralOrganoid::new(10, 19);
        organoid.set_gap_junction_permeability(5.0);
        assert_eq!(organoid.gap_junction_permeability(), 1.0);
        organoid.set_gap_junction_permeability(-5.0);
        assert_eq!(organoid.gap_junction_permeability(), 0.0);
    }

    #[test]
    fn gap_junction_blockade_changes_differentiation_outcome() {
        // The model's analog of Levin's octanol/lanthanum gap-junction-blocker
        // experiments: identical activator level and identical gene
        // expression ("genetics"), differing only in whether the bioelectric
        // (Vmem) signal is allowed to propagate. Gene expression is tuned so
        // the neural gene-expression signal alone (0.35) sits *below* the
        // neural-fate cutoff (0.5) but *above* the glial cutoff (0.3) — only
        // the Vmem hyperpolarization bias (added when gap junctions are open)
        // can push it over 0.5 into neural fate. This is deliberately a
        // precise single-cell unit test of the causal mechanism rather than
        // an emergent multi-day simulation: with this model's cell density,
        // the Turing activator field only rarely climbs past the
        // differentiation thresholds at all (a pre-existing property of the
        // chemical layer, unrelated to this bioelectric addition), so an
        // emergent test would be at the mercy of whether *any* differentiation
        // happens to occur, not of whether blocking changes its outcome.
        let differentiate_with = |permeability: f32| {
            let mut field = MorphogeneticField::new(1, 1);
            field.activator[0] = 1.6; // clears the a > 1.5 neural threshold
            field.cells[0].cell_type = OrganoidCellType::Progenitor;
            field.cells[0].gene_expression = vec![0.35, 0.35, 0.35, 0.35, 0.5, 0.5, 0.5, 0.5];
            field.bioelectric.vmem[0] = VMEM_HYPERPOLARIZED;
            field.bioelectric.gap_junction_permeability = permeability;
            field.differentiate();
            field.cells[0].cell_type
        };

        let unblocked = differentiate_with(1.0);
        let blocked = differentiate_with(0.0);

        assert_eq!(
            unblocked,
            OrganoidCellType::NeuralProgenitor,
            "Open gap junctions: Vmem hyperpolarization bias should tip an otherwise \
             sub-threshold cell into neural fate"
        );
        assert!(
            matches!(blocked, OrganoidCellType::Glial(_)),
            "Blocked gap junctions: the same cell, same gene expression, same activator \
             level should fall back to glial fate — the causal difference is bioelectric \
             coupling, not genetics. Got {blocked:?}"
        );
        assert_ne!(unblocked, blocked);
    }

    #[test]
    fn undifferentiated_cell_type_bucket_is_zero() {
        assert_eq!(cell_type_bucket(OrganoidCellType::Undifferentiated), 0);
        assert_eq!(cell_type_bucket(OrganoidCellType::Progenitor), 1);
        assert_eq!(cell_type_bucket(OrganoidCellType::NeuralProgenitor), 1);
        assert_eq!(
            cell_type_bucket(OrganoidCellType::Neuron(NeuronSubtype::Excitatory)),
            2
        );
    }
}
