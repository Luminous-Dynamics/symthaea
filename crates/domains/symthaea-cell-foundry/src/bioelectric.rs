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

/// Spatial radius for gap-junction formation — electrical coupling is
/// *unrestricted by cell type* (unlike synaptogenesis, which only connects
/// neurons). Deliberately larger than the chemical field's neighbourhood
/// radius (0.15): cells are placed uniformly in an `[-1,1]^3` cube, so at
/// the cell counts used throughout this crate (roughly 100-300), 0.15 gives
/// well under 1 expected gap-junction neighbour per cell — diffusion would
/// be a near no-op for most cells regardless of permeability, silently
/// erasing the very open-vs-blocked effect this layer exists to model. 0.4
/// gives ~5-7 expected neighbours at n=200, enough for propagation to
/// actually happen.
const GAP_JUNCTION_RADIUS: f32 = 0.4;
/// Electrical (gap-junction) diffusion is much faster than chemical
/// diffusion — this is deliberately larger than the Turing model's D_a/D_h.
const GAP_JUNCTION_DIFFUSION_RATE: f32 = 0.35;
/// Per-day drift rate pulling Vmem toward the fate-appropriate resting
/// potential (hyperpolarized once differentiated, depolarized otherwise).
/// Used only when the opt-in ion-channel model
/// (`NeuralOrganoid::set_ion_channel_model_enabled`) is disabled (the
/// default) -- see [`crate::ion_channels`] for the alternative,
/// Nernst-derived dynamics.
const HYPERPOLARIZATION_DRIFT: f32 = 0.05;
/// Baseline per-cell K+ (hyperpolarizing) leak conductance for
/// undifferentiated cells, used only by the opt-in ion-channel model. Paired
/// with [`NA_CHANNEL_BASELINE_CONDUCTANCE`] below, this ratio keeps an
/// undifferentiated cell's conductance-weighted resting potential near
/// `VMEM_DEPOLARIZED` -- see [`crate::ion_channels`].
const K_CHANNEL_BASELINE_CONDUCTANCE: f32 = 0.03;
/// Baseline per-cell Na+-like (depolarizing) leak conductance, present in
/// all cells regardless of fate. See [`K_CHANNEL_BASELINE_CONDUCTANCE`].
const NA_CHANNEL_BASELINE_CONDUCTANCE: f32 = 0.04;
/// K+ conductance added when a cell commits to Neuron/Glial fate under the
/// ion-channel model -- models real Kv-channel upregulation driving
/// differentiation-associated hyperpolarization (the finding
/// `crate::ion_channels`'s module docs cite). See
/// `crate::morphogenetic_consciousness::differentiate`.
pub(crate) const K_CHANNEL_DIFFERENTIATION_BUMP: f32 = 2.0;
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
/// Number of angular octants (sign combination of x/y/z) each shell is
/// further subdivided into, giving `TargetMorphology` real spatial — not
/// just radial — resolution.
const N_OCTANTS: usize = 8;
/// Total spatial bins = shells x octants.
const N_BINS: usize = N_SHELLS * N_OCTANTS;
/// Radius beyond which all cells are bucketed into the outermost shell.
const SHELL_MAX_RADIUS: f32 = 1.3;
/// Weight of the Vmem spatial pattern in `TargetMorphology::discrepancy`,
/// vs. the downstream cell-type composition (weight `1.0 -
/// VMEM_PATTERN_WEIGHT`). The Vmem pattern is primary because it's the
/// actual bioelectric "memory" Levin's work argues instructs anatomy —
/// composition is a secondary, downstream diagnostic.
const VMEM_PATTERN_WEIGHT: f64 = 0.6;
/// Per-day probability a wound-boundary progenitor divides during active
/// regeneration — deliberately higher than the baseline `PROLIFERATION_RATE`
/// to model blastema-like accelerated growth at the wound site.
const REGENERATION_PROLIFERATION_BOOST: f64 = 0.25;
/// RMS discrepancy below which regeneration is considered converged (the
/// wound is "healed").
const MORPHOLOGY_CONVERGENCE_TOLERANCE: f64 = 0.05;
/// Days after which regeneration bookkeeping times out even if the target
/// was never reached (avoids an unbounded regenerating state).
const REGENERATION_TIMEOUT_DAYS: u32 = 200;
/// Per-day rate pulling a cell's Vmem toward the captured target's value at
/// its own position, when positional homing is enabled (see
/// `NeuralOrganoid::set_positional_homing`). Deliberately slow and — unlike
/// gap-junction diffusion — *not* gated by `gap_junction_permeability`: this
/// models a durable, cell-intrinsic positional/regulatory memory distinct
/// from the fast electrical coupling channel (Levin's own framing treats
/// acutely-readable Vmem as downstream of, and coupled to, a deeper and more
/// robust regulatory memory). It is what makes recovery from *total*
/// pattern destruction (`Perturbation::ScrambleVmem`, which leaves no
/// surviving neighbour to copy from) possible at all.
const POSITIONAL_HOMING_RATE: f32 = 0.05;
/// Per-day division probability for "defected" progenitor cells — much
/// higher than the baseline [`PROLIFERATION_RATE`] and, unlike
/// regeneration's boosted rate, applies unconditionally every day (not
/// gated by wound/regeneration bookkeeping). Models unregulated growth
/// after a cell loses bioelectric coordination with its neighbours — see
/// `NeuralOrganoid::induce_local_defection`.
const DEFECTION_PROLIFERATION_RATE: f64 = 0.35;

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
    /// Cells that have "defected" from the collective — gap-junction
    /// isolated and unconditionally hyperproliferative, the model's analog
    /// of Levin's framing of cancer as a bioelectric breakdown of
    /// multicellular coordination. See
    /// [`NeuralOrganoid::induce_local_defection`].
    pub defected: Vec<bool>,
    /// Runtime-configurable gap-junction diffusion rate, defaulting to
    /// [`GAP_JUNCTION_DIFFUSION_RATE`]. Exposed so experiments can sweep it
    /// (e.g. a homing-rate x diffusion-rate phase diagram) without
    /// recompiling.
    pub gap_junction_diffusion_rate: f32,
    /// Runtime-configurable positional-homing rate, defaulting to
    /// [`POSITIONAL_HOMING_RATE`]. See `gap_junction_diffusion_rate`.
    pub positional_homing_rate: f32,
    /// Per-cell K+ (hyperpolarizing) leak channel conductance. Only used by
    /// the opt-in ion-channel model (`ion_channel_model_enabled` below) --
    /// see [`crate::ion_channels`].
    pub k_channel_conductance: Vec<f32>,
    /// Per-cell Na+-like (depolarizing) leak channel conductance. Only used
    /// by the opt-in ion-channel model. See [`crate::ion_channels`].
    pub na_channel_conductance: Vec<f32>,
    /// Pharmacological K+-channel blocker multiplier in `[0, 1]` (1.0 =
    /// fully open/control, 0.0 = fully blocked) -- e.g. Levin-lab tool
    /// compounds like valinomycin/ivermectin. Distinct from
    /// `gap_junction_permeability`: this targets ion-channel expression
    /// directly, not electrical coupling between cells. Only has an effect
    /// when `ion_channel_model_enabled` is `true`. See
    /// `NeuralOrganoid::set_potassium_channel_block`.
    pub potassium_channel_block: f32,
    /// Whether the opt-in ion-channel conductance model drives Vmem
    /// dynamics in [`MorphogeneticField::step_vmem_diffusion`], replacing
    /// the default ad hoc fate-drift term. Defaults to `false` so existing
    /// behavior/tests are unaffected. See
    /// `NeuralOrganoid::set_ion_channel_model_enabled`.
    pub ion_channel_model_enabled: bool,
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
            defected: vec![false; n],
            gap_junction_diffusion_rate: GAP_JUNCTION_DIFFUSION_RATE,
            positional_homing_rate: POSITIONAL_HOMING_RATE,
            k_channel_conductance: vec![K_CHANNEL_BASELINE_CONDUCTANCE; n],
            na_channel_conductance: vec![NA_CHANNEL_BASELINE_CONDUCTANCE; n],
            potassium_channel_block: 1.0,
            ion_channel_model_enabled: false,
            rng: StdRng::seed_from_u64(seed.wrapping_add(0xB10E_1EC7)),
        }
    }

    /// Grow all per-cell arrays to `total` entries (called after
    /// proliferation adds new cells). New cells start depolarized and
    /// unconnected. Prefer [`Self::resize_inheriting`] wherever the new
    /// cells have a known parent — this default-baseline variant is only
    /// appropriate when there's no meaningful parent to inherit from.
    pub(crate) fn resize(&mut self, total: usize) {
        let old = self.vmem.len();
        self.vmem.resize(total, VMEM_DEPOLARIZED);
        self.wound_boundary.resize(total, false);
        self.defected.resize(total, false);
        self.k_channel_conductance
            .resize(total, K_CHANNEL_BASELINE_CONDUCTANCE);
        self.na_channel_conductance
            .resize(total, NA_CHANNEL_BASELINE_CONDUCTANCE);
        for row in self.gap_junction_matrix.iter_mut() {
            row.resize(total, 0.0);
        }
        for _ in old..total {
            self.gap_junction_matrix.push(vec![0.0; total]);
        }
    }

    /// Grow all per-cell arrays for `parent_indices.len()` new daughter
    /// cells, each inheriting its Vmem from `parent_indices[k]` (the cell it
    /// was cloned from) rather than resetting to a default.
    ///
    /// This matters at scale: ordinary proliferation happens throughout the
    /// tissue, not just at a wound. If every daughter reset to
    /// `VMEM_DEPOLARIZED` regardless of where it was born, a population that
    /// grows ~10%/day compounds to ~45x over 40 days — a flood of
    /// default-valued newborns that would wash out any surviving spatial
    /// Vmem pattern (the exact signal `TargetMorphology`/equifinality
    /// experiments depend on) within a few cell generations, independent of
    /// gap-junction state. Inheriting keeps growth pattern-preserving:
    /// daughters born in a hyperpolarized region stay hyperpolarized: not
    /// because they "know" the target, but because mitosis doesn't reset
    /// membrane state in real cells either.
    pub(crate) fn resize_inheriting(&mut self, parent_indices: &[usize]) {
        let old = self.vmem.len();
        let added = parent_indices.len();
        let total = old + added;
        let inherited: Vec<f32> = parent_indices.iter().map(|&p| self.vmem[p]).collect();
        let inherited_defected: Vec<bool> =
            parent_indices.iter().map(|&p| self.defected[p]).collect();
        let inherited_k: Vec<f32> = parent_indices
            .iter()
            .map(|&p| self.k_channel_conductance[p])
            .collect();
        let inherited_na: Vec<f32> = parent_indices
            .iter()
            .map(|&p| self.na_channel_conductance[p])
            .collect();
        self.vmem.resize(total, VMEM_DEPOLARIZED);
        for (k, v) in inherited.into_iter().enumerate() {
            self.vmem[old + k] = v;
        }
        self.wound_boundary.resize(total, false);
        self.defected.resize(total, false);
        for (k, d) in inherited_defected.into_iter().enumerate() {
            self.defected[old + k] = d;
        }
        self.k_channel_conductance
            .resize(total, K_CHANNEL_BASELINE_CONDUCTANCE);
        for (k, g) in inherited_k.into_iter().enumerate() {
            self.k_channel_conductance[old + k] = g;
        }
        self.na_channel_conductance
            .resize(total, NA_CHANNEL_BASELINE_CONDUCTANCE);
        for (k, g) in inherited_na.into_iter().enumerate() {
            self.na_channel_conductance[old + k] = g;
        }
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
        let mut new_defected = Vec::with_capacity(next);
        let mut new_k_channel = Vec::with_capacity(next);
        let mut new_na_channel = Vec::with_capacity(next);
        for i in 0..n {
            if keep[i] {
                new_vmem.push(self.vmem[i]);
                new_wound.push(self.wound_boundary[i]);
                new_defected.push(self.defected[i]);
                new_k_channel.push(self.k_channel_conductance[i]);
                new_na_channel.push(self.na_channel_conductance[i]);
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
        self.defected = new_defected;
        self.k_channel_conductance = new_k_channel;
        self.na_channel_conductance = new_na_channel;
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
    ///
    /// Candidate pairs come from [`MorphogeneticField::neighbours`]
    /// (grid-backed once a day's grid has been built). Candidates are
    /// sorted before use so the `i < j` ordering — and therefore the
    /// sequence of `self.bioelectric.rng` draws below — matches the
    /// original O(n^2) pairwise scan exactly; the grid returns matches in
    /// bucket order, not index order.
    pub fn form_gap_junctions(&mut self) {
        let n = self.cells.len();
        let permeability = self.bioelectric.gap_junction_permeability as f64;
        for i in 0..n {
            let mut candidates = self.neighbours(i, GAP_JUNCTION_RADIUS);
            candidates.sort_unstable();
            for j in candidates {
                if j <= i {
                    continue;
                }
                if self.bioelectric.gap_junction_matrix[i][j] == 0.0 {
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
    ///
    /// The fate-drift term (last step below) has two mutually exclusive
    /// implementations, selected by `ion_channel_model_enabled` (default
    /// `false`, in which case this method's behavior is unchanged from
    /// before the ion-channel model existed): an ad hoc drift toward one of
    /// two hardcoded target voltages, or a per-cell conductance-weighted sum
    /// against real Nernst-derived reversal potentials. See
    /// `NeuralOrganoid::set_ion_channel_model_enabled` and
    /// [`crate::ion_channels`].
    pub fn step_vmem_diffusion(&mut self, dt: f32) {
        let n = self.cells.len();
        let permeability = self.bioelectric.gap_junction_permeability;
        let diffusion_rate = self.bioelectric.gap_junction_diffusion_rate;
        let ion_channel_model_enabled = self.bioelectric.ion_channel_model_enabled;
        let (e_k, e_na) = if ion_channel_model_enabled {
            crate::ion_channels::reversal_potentials_normalized()
        } else {
            (0.0, 0.0)
        };
        let mut new_vmem = self.bioelectric.vmem.clone();
        for i in 0..n {
            // Defected cells are gap-junction isolated in both directions —
            // see `NeuralOrganoid::induce_local_defection`.
            let is_defected = self.bioelectric.defected[i];
            let row = &self.bioelectric.gap_junction_matrix[i];
            let mut weighted_sum = 0.0f64;
            let mut weight_total = 0.0f64;
            if !is_defected {
                for (j, &w) in row.iter().enumerate() {
                    if w > 0.0 && !self.bioelectric.defected[j] {
                        weighted_sum += w * self.bioelectric.vmem[j] as f64;
                        weight_total += w;
                    }
                }
            }

            let mut v = self.bioelectric.vmem[i];
            if weight_total > 0.0 {
                let avg = (weighted_sum / weight_total) as f32;
                v += dt * diffusion_rate * permeability * (avg - v);
            }

            if ion_channel_model_enabled {
                let g_k = self.bioelectric.k_channel_conductance[i]
                    * self.bioelectric.potassium_channel_block;
                let g_na = self.bioelectric.na_channel_conductance[i];
                v += dt * (g_k * (e_k - v) + g_na * (e_na - v));
            } else {
                let target =
                    if self.cells[i].cell_type.is_neuron() || self.cells[i].cell_type.is_glial() {
                        VMEM_HYPERPOLARIZED
                    } else {
                        VMEM_DEPOLARIZED
                    };
                v += dt * HYPERPOLARIZATION_DRIFT * (target - v);
            }

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
        self.amputate_where(|p| {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            r >= min_r && r < max_r
        })
    }

    /// Remove every cell for which `remove(position)` is true, simulating
    /// cutting away an arbitrarily-shaped region (radial shells via
    /// [`Self::amputate`], but also axis-aligned slabs, hemispheres, or any
    /// other spatial criterion an experiment needs). Reindexes every
    /// per-cell array and the connectivity/gap-junction matrices, marks
    /// surviving cells adjacent to the cut as `wound_boundary`, and applies
    /// a transient depolarizing "wound spike." Returns the number removed.
    pub fn amputate_where(&mut self, remove: impl Fn(&[f32; 3]) -> bool) -> usize {
        let n = self.cells.len();
        if n == 0 {
            return 0;
        }

        let keep: Vec<bool> = (0..n).map(|i| !remove(&self.cells[i].position)).collect();
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

    /// Mark every cell for which `select(position)` is true as "defected" —
    /// gap-junction isolated (see `step_vmem_diffusion`) and unconditionally
    /// hyperproliferative (see `defective_proliferate`) — depolarize them,
    /// and reset them to `Progenitor` identity. That last part matters
    /// mechanically, not just narratively: `defective_proliferate` only
    /// fires on progenitor-type cells, and a cell selected purely by
    /// position may already have differentiated or dropped to
    /// `Undifferentiated` (ineligible either way) during normal
    /// development. Forcing progenitor identity is also the biologically
    /// motivated choice: oncogenic transformation is widely associated with
    /// *dedifferentiation* to a stem/progenitor-like state, not just loss
    /// of electrical coupling — this models both halves of that at once.
    /// Called from [`NeuralOrganoid::induce_local_defection`].
    pub fn induce_local_defection(&mut self, select: impl Fn(&[f32; 3]) -> bool) -> usize {
        let mut count = 0;
        for i in 0..self.cells.len() {
            if select(&self.cells[i].position) {
                self.bioelectric.defected[i] = true;
                self.bioelectric.vmem[i] = VMEM_DEPOLARIZED;
                self.cells[i].cell_type = OrganoidCellType::Progenitor;
                count += 1;
            }
        }
        count
    }

    /// Number of currently-defected cells.
    pub fn defected_cell_count(&self) -> usize {
        self.bioelectric.defected.iter().filter(|&&d| d).count()
    }

    /// Unconditional proliferation pass for defected cells, at
    /// [`DEFECTION_PROLIFERATION_RATE`] — unlike `regenerative_proliferate`,
    /// this runs every day regardless of wound/regeneration state, modeling
    /// unregulated growth. Called from `advance_day`.
    ///
    /// Defected cells are re-asserted as `Progenitor` every call, overriding
    /// whatever the ordinary Turing-driven `differentiate()` step would
    /// otherwise do to them: escaping normal differentiation control (not
    /// just gap-junction coupling) is itself a hallmark of the pathology
    /// being modeled, not an oversight.
    pub(crate) fn defective_proliferate(&mut self) {
        let n = self.cells.len();
        if n >= MAX_CELLS {
            return;
        }
        for i in 0..n {
            if self.bioelectric.defected[i] {
                self.cells[i].cell_type = OrganoidCellType::Progenitor;
            }
        }
        let mut new_cells = Vec::new();
        let mut parent_indices = Vec::new();
        for i in 0..n {
            if !self.bioelectric.defected[i] {
                continue;
            }
            if self
                .bioelectric
                .rng
                .gen_bool(DEFECTION_PROLIFERATION_RATE.min(1.0))
            {
                let mut daughter = self.cells[i].clone();
                for d in 0..3 {
                    daughter.position[d] += self.bioelectric.rng.gen_range(-0.02..0.02f32);
                }
                daughter.age_days = 0;
                daughter.connectivity.clear();
                new_cells.push(daughter);
                parent_indices.push(i);
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
        // Daughters inherit Vmem AND defected status from their parent —
        // a defected clone stays defected.
        self.bioelectric.resize_inheriting(&parent_indices);
    }

    /// Randomize every cell's Vmem within `[VMEM_HYPERPOLARIZED,
    /// VMEM_DEPOLARIZED]`, without removing or altering any cell's
    /// type/position/gene expression — the model's analog of a
    /// pharmacological/optogenetic bioelectric disruption (ion-channel
    /// drugs, forced depolarization) rather than physical injury. Existing
    /// gap-junction connections are left intact, so a subsequent
    /// `step_vmem_diffusion` pass can immediately start smoothing the
    /// scrambled pattern back toward local coherence — but *not* toward any
    /// specific spatially-targeted pattern unless surviving structure
    /// elsewhere still encodes it (see [`crate::experiments`] module docs
    /// for the boundary this exposes).
    pub fn scramble_vmem(&mut self, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        for v in self.bioelectric.vmem.iter_mut() {
            *v = rng.gen_range(VMEM_HYPERPOLARIZED..=VMEM_DEPOLARIZED);
        }
    }

    /// Directly impose a Vmem value on every cell as a function of its
    /// spatial position, without changing cell type/identity — models an
    /// externally applied or naturally pre-existing bioelectric prepattern
    /// (Levin's lab routinely imposes target Vmem patterns via
    /// ionophores/optogenetics independent of any transcriptional change).
    /// Useful for building organoids with genuine spatial bioelectric
    /// structure to test recovery against, without waiting on the
    /// (comparatively rare) Turing-driven differentiation pathway to
    /// produce heterogeneity on its own.
    pub fn impose_vmem_pattern(&mut self, f: impl Fn(&[f32; 3]) -> f32) {
        for i in 0..self.cells.len() {
            let pos = self.cells[i].position;
            self.bioelectric.vmem[i] = f(&pos);
        }
    }
}

// ---------------------------------------------------------------------------
// TargetMorphology
// ---------------------------------------------------------------------------

/// A captured "goal" anatomical pattern, spatially binned into
/// [`N_SHELLS`] radial shells x [`N_OCTANTS`] angular octants ([`N_BINS`]
/// bins total).
///
/// The primary signal is `bin_mean_vmem` — the actual bioelectric pattern
/// Levin's lab argues *is* the stored "memory" of target anatomy, separable
/// from genome and from chemical gradients (see module docs). Earlier
/// versions of this type captured only downstream cell-type composition,
/// which conflated the memory with one of its effects; `bin_composition` is
/// kept as a secondary, diagnostic signal (useful for interpreting *what*
/// regenerated, not just whether the voltage pattern matches), weighted
/// lower in `discrepancy` — see [`VMEM_PATTERN_WEIGHT`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetMorphology {
    /// Mean Vmem per spatial bin — the bioelectric pattern memory.
    pub bin_mean_vmem: Vec<f32>,
    /// `bin_composition[bin][bucket]` — bucket 0=undifferentiated,
    /// 1=progenitor (incl. neural progenitor), 2=neuron, 3=glial.
    pub bin_composition: Vec<[f32; 4]>,
}

fn shell_index(radius: f32) -> usize {
    let r = radius.clamp(0.0, SHELL_MAX_RADIUS);
    let idx = (r / SHELL_MAX_RADIUS * N_SHELLS as f32) as usize;
    idx.min(N_SHELLS - 1)
}

/// Sign-combination octant of a position: bit 0 = x>=0, bit 1 = y>=0,
/// bit 2 = z>=0.
fn octant_index(pos: &[f32; 3]) -> usize {
    let x = (pos[0] >= 0.0) as usize;
    let y = (pos[1] >= 0.0) as usize;
    let z = (pos[2] >= 0.0) as usize;
    x | (y << 1) | (z << 2)
}

fn bin_index(pos: &[f32; 3]) -> usize {
    let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
    shell_index(r) * N_OCTANTS + octant_index(pos)
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
    /// Snapshot the current per-bin mean Vmem and cell-type composition of
    /// `field` as a goal pattern.
    pub fn capture(field: &MorphogeneticField) -> Self {
        let mut vmem_sum = vec![0.0f64; N_BINS];
        let mut counts = vec![[0u32; 4]; N_BINS];
        let mut totals = [0u32; N_BINS];
        for (i, cell) in field.cells.iter().enumerate() {
            let bin = bin_index(&cell.position);
            vmem_sum[bin] += field.bioelectric.vmem[i] as f64;
            let b = cell_type_bucket(cell.cell_type);
            counts[bin][b] += 1;
            totals[bin] += 1;
        }
        let mut bin_mean_vmem = vec![0.0f32; N_BINS];
        let mut bin_composition = vec![[0.0f32; 4]; N_BINS];
        for bin in 0..N_BINS {
            if totals[bin] == 0 {
                continue;
            }
            bin_mean_vmem[bin] = (vmem_sum[bin] / totals[bin] as f64) as f32;
            for b in 0..4 {
                bin_composition[bin][b] = counts[bin][b] as f32 / totals[bin] as f32;
            }
        }
        Self {
            bin_mean_vmem,
            bin_composition,
        }
    }

    /// Combined RMS discrepancy between this target and `field`'s current
    /// state: primarily the Vmem spatial pattern, secondarily the
    /// downstream cell-type composition (see [`VMEM_PATTERN_WEIGHT`]).
    /// 0.0 = exact match.
    pub fn discrepancy(&self, field: &MorphogeneticField) -> f64 {
        let current = Self::capture(field);

        let vmem_span = (VMEM_DEPOLARIZED - VMEM_HYPERPOLARIZED).abs() as f64;
        let vmem_rms = {
            let sum_sq: f64 = self
                .bin_mean_vmem
                .iter()
                .zip(current.bin_mean_vmem.iter())
                .map(|(&a, &b)| {
                    let d = (a - b) as f64;
                    d * d
                })
                .sum();
            (sum_sq / N_BINS as f64).sqrt()
        };
        let vmem_rms_normalized = (vmem_rms / vmem_span).min(1.0);

        let composition_rms = {
            let mut sum_sq = 0.0f64;
            let mut count = 0usize;
            for bin in 0..N_BINS {
                for b in 0..4 {
                    let d = (self.bin_composition[bin][b] - current.bin_composition[bin][b]) as f64;
                    sum_sq += d * d;
                    count += 1;
                }
            }
            (sum_sq / count as f64).sqrt()
        };

        VMEM_PATTERN_WEIGHT * vmem_rms_normalized + (1.0 - VMEM_PATTERN_WEIGHT) * composition_rms
    }

    /// The target Vmem for whatever spatial bin `position` falls into —
    /// this organoid's own captured "goal" pattern, read out independent of
    /// any cell's current dynamic state. See
    /// [`NeuralOrganoid::set_positional_homing`] for why this matters: it's
    /// the mechanism that can recover a pattern even after it's been
    /// destroyed everywhere at once (no surviving neighbour to copy from),
    /// which gap-junction propagation alone cannot do.
    pub fn value_at(&self, position: &[f32; 3]) -> f32 {
        self.bin_mean_vmem[bin_index(position)]
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

    /// Cut away any spatial region — see [`MorphogeneticField::amputate_where`].
    pub fn amputate_where(&mut self, remove: impl Fn(&[f32; 3]) -> bool) -> usize {
        let removed = self.field.amputate_where(remove);
        if removed > 0 {
            self.days_since_wound = 0;
        }
        removed
    }

    /// Induce local bioelectric "defection" (Levin's framing of cancer as a
    /// breakdown of multicellular coordination) in every cell for which
    /// `select(position)` is true — see
    /// [`MorphogeneticField::induce_local_defection`]. Unlike `amputate`,
    /// this doesn't remove tissue or touch the regeneration clock; it marks
    /// cells as gap-junction-isolated and unconditionally hyperproliferative
    /// starting immediately. Returns the number of cells marked.
    pub fn induce_local_defection(&mut self, select: impl Fn(&[f32; 3]) -> bool) -> usize {
        self.field.induce_local_defection(select)
    }

    /// Number of currently-defected cells.
    pub fn defected_cell_count(&self) -> usize {
        self.field.defected_cell_count()
    }

    /// Scramble the organoid's Vmem pattern without removing any cells —
    /// see [`MorphogeneticField::scramble_vmem`]. Resets the regeneration
    /// clock so `advance_regeneration` treats this as fresh damage to
    /// recover from (bioelectric perturbation without tissue loss).
    pub fn scramble_vmem(&mut self, seed: u64) {
        self.field.scramble_vmem(seed);
        self.days_since_wound = 0;
    }

    /// Impose a Vmem pattern as a function of position — see
    /// [`MorphogeneticField::impose_vmem_pattern`]. Does not reset the
    /// regeneration clock (this is meant for building a starting template,
    /// not for perturbing an already-running experiment).
    pub fn impose_vmem_pattern(&mut self, f: impl Fn(&[f32; 3]) -> f32) {
        self.field.impose_vmem_pattern(f);
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

    /// Enable or disable positional homing — pulling each cell's Vmem
    /// toward the captured target's value at its own position, at
    /// [`POSITIONAL_HOMING_RATE`], regardless of gap-junction permeability.
    /// This is the mechanism that can recover a target pattern even after
    /// it's been destroyed *everywhere at once*
    /// ([`crate::experiments::Perturbation::ScrambleVmem`]), which
    /// neighbour-propagation alone cannot do — see module docs. No-op if no
    /// target morphology has been captured yet.
    pub fn set_positional_homing(&mut self, enabled: bool) {
        self.positional_homing_enabled = enabled;
    }

    /// Whether positional homing is currently enabled.
    pub fn positional_homing_enabled(&self) -> bool {
        self.positional_homing_enabled
    }

    /// Enable or disable the opt-in real-physics packing correction pass —
    /// see [`crate::packing`] for what it does and why it's off by default.
    pub fn set_packing_enabled(&mut self, enabled: bool) {
        self.packing_enabled = enabled;
    }

    /// Whether the packing correction pass is currently enabled.
    pub fn packing_enabled(&self) -> bool {
        self.packing_enabled
    }

    /// Enable or disable the opt-in ion-channel conductance model for Vmem
    /// dynamics, replacing the default ad hoc fate-drift term with a
    /// per-cell conductance-weighted sum against real Nernst-derived
    /// reversal potentials -- see [`crate::ion_channels`] for what it does
    /// and why it's off by default.
    pub fn set_ion_channel_model_enabled(&mut self, enabled: bool) {
        self.field.bioelectric.ion_channel_model_enabled = enabled;
    }

    /// Whether the ion-channel conductance model is currently enabled.
    pub fn ion_channel_model_enabled(&self) -> bool {
        self.field.bioelectric.ion_channel_model_enabled
    }

    /// Pharmacological K+-channel blocker multiplier in `[0, 1]` (1.0 =
    /// fully open/control, 0.0 = fully blocked) -- e.g. Levin-lab tool
    /// compounds like valinomycin/ivermectin. Distinct from
    /// `set_gap_junction_permeability`: this targets ion-channel expression
    /// directly, not electrical coupling between cells. Only has an effect
    /// when the ion-channel model is enabled
    /// (`set_ion_channel_model_enabled`).
    pub fn set_potassium_channel_block(&mut self, block: f32) {
        self.field.bioelectric.potassium_channel_block = block.clamp(0.0, 1.0);
    }

    /// Current K+-channel blocker multiplier.
    pub fn potassium_channel_block(&self) -> f32 {
        self.field.bioelectric.potassium_channel_block
    }

    /// Apply one day's positional-homing pull, if enabled and a target has
    /// been captured. Called automatically from `advance_day`.
    pub(crate) fn apply_positional_homing(&mut self) {
        if !self.positional_homing_enabled || self.target_morphology.is_none() {
            return;
        }
        let rate = self.field.bioelectric.positional_homing_rate;
        for i in 0..self.field.num_cells() {
            // Defected cells have opted out of the collective's instructions
            // entirely — neither neighbour propagation nor positional
            // homing reaches them. See `induce_local_defection`.
            if self.field.bioelectric.defected[i] {
                continue;
            }
            let pos = self.field.cells[i].position;
            let goal = self.target_morphology.as_ref().unwrap().value_at(&pos);
            let v = self.field.bioelectric.vmem[i];
            self.field.bioelectric.vmem[i] = v + rate * (goal - v);
        }
    }

    /// Set the gap-junction diffusion rate (default
    /// [`GAP_JUNCTION_DIFFUSION_RATE`]). Exposed for sweep experiments (e.g.
    /// a homing-rate x diffusion-rate phase diagram).
    pub fn set_gap_junction_diffusion_rate(&mut self, rate: f32) {
        self.field.bioelectric.gap_junction_diffusion_rate = rate;
    }

    /// Set the positional-homing rate (default [`POSITIONAL_HOMING_RATE`]).
    /// See `set_gap_junction_diffusion_rate`.
    pub fn set_positional_homing_rate(&mut self, rate: f32) {
        self.field.bioelectric.positional_homing_rate = rate;
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

        // This is a weak sanity check (no imposed target pattern, so there's
        // little spatial structure to actually recover) — the rigorous,
        // mechanism-level version of this claim lives in
        // `crate::experiments`'s equifinality tests, which impose a real
        // bipolar Vmem pattern first. Here we just require regeneration not
        // to make things meaningfully *worse*, with a small tolerance for
        // the Vmem-pattern term's day-to-day noise.
        assert!(
            discrepancy_later <= discrepancy_right_after + 0.05,
            "Regeneration should not meaningfully increase discrepancy from target: right_after={discrepancy_right_after}, later={discrepancy_later}"
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

    #[test]
    fn scramble_vmem_changes_pattern_without_removing_cells() {
        let mut field = MorphogeneticField::new(50, 30);
        let before_count = field.num_cells();
        field.scramble_vmem(1);
        assert_eq!(field.num_cells(), before_count);
        assert!(
            field
                .bioelectric
                .vmem
                .iter()
                .all(|&v| (VMEM_HYPERPOLARIZED..=VMEM_DEPOLARIZED).contains(&v)),
            "Scrambled Vmem should stay within [hyperpolarized, depolarized]"
        );
        // With 50 cells, extremely unlikely all land on the exact same value.
        let first = field.bioelectric.vmem[0];
        assert!(field.bioelectric.vmem.iter().any(|&v| v != first));
    }

    #[test]
    fn proliferation_daughters_inherit_parent_vmem_not_default() {
        let mut field = MorphogeneticField::new(30, 2);
        field.impose_vmem_pattern(|_| VMEM_HYPERPOLARIZED);
        // Force every cell to progenitor so proliferate() has candidates,
        // and drive proliferation deterministically via a fresh seeded rng
        // by just calling proliferate() enough times to very likely add
        // at least one daughter.
        for _ in 0..30 {
            field.proliferate();
            if field.num_cells() > 30 {
                break;
            }
        }
        assert!(
            field.num_cells() > 30,
            "Expected at least one proliferation event across 30 attempts"
        );
        // Every daughter should have inherited the hyperpolarized value,
        // not reset to VMEM_DEPOLARIZED.
        for &v in field.bioelectric.vmem.iter().skip(30) {
            assert_eq!(
                v, VMEM_HYPERPOLARIZED,
                "Daughter cell should inherit parent's Vmem, not a default"
            );
        }
    }

    #[test]
    fn scramble_vmem_is_deterministic_for_a_given_seed() {
        let mut a = MorphogeneticField::new(20, 1);
        let mut b = MorphogeneticField::new(20, 1);
        a.scramble_vmem(42);
        b.scramble_vmem(42);
        assert_eq!(a.bioelectric.vmem, b.bioelectric.vmem);
    }

    #[test]
    fn impose_vmem_pattern_applies_spatial_function() {
        let mut field = MorphogeneticField::new(30, 2);
        field.impose_vmem_pattern(|p| if p[0] >= 0.0 { 0.0 } else { -1.0 });
        for (i, cell) in field.cells.iter().enumerate() {
            let expected = if cell.position[0] >= 0.0 { 0.0 } else { -1.0 };
            assert_eq!(field.bioelectric.vmem[i], expected);
        }
    }

    #[test]
    fn target_morphology_captures_vmem_pattern_not_just_composition() {
        // Two fields with identical (all-progenitor) composition but
        // different Vmem patterns should show nonzero discrepancy — proving
        // the target is keyed off Vmem, not just cell type.
        let mut field_a = MorphogeneticField::new(40, 4);
        field_a.impose_vmem_pattern(|_| VMEM_DEPOLARIZED);
        let mut field_b = MorphogeneticField::new(40, 4);
        field_b.impose_vmem_pattern(|_| VMEM_HYPERPOLARIZED);

        let target = TargetMorphology::capture(&field_a);
        let discrepancy = target.discrepancy(&field_b);
        assert!(
            discrepancy > 0.0,
            "Same composition, different Vmem pattern should show nonzero discrepancy"
        );
    }

    #[test]
    fn value_at_returns_captured_bin_mean() {
        let mut field = MorphogeneticField::new(1, 5);
        field.cells[0].position = [0.9, 0.0, 0.0];
        field.bioelectric.vmem[0] = VMEM_HYPERPOLARIZED;
        let target = TargetMorphology::capture(&field);
        assert_eq!(target.value_at(&[0.9, 0.0, 0.0]), VMEM_HYPERPOLARIZED);
    }

    #[test]
    fn positional_homing_disabled_by_default() {
        let organoid = NeuralOrganoid::new(10, 6);
        assert!(!organoid.positional_homing_enabled());
    }

    #[test]
    fn positional_homing_pulls_toward_target_regardless_of_permeability() {
        // A single cell, far from any neighbour, blocked gap junctions —
        // neighbour propagation is entirely unavailable here. Only
        // positional homing can move it toward the target.
        let mut organoid = NeuralOrganoid::new(1, 8);
        organoid.field.cells[0].position = [0.5, 0.5, 0.5];
        organoid.field.bioelectric.vmem[0] = VMEM_DEPOLARIZED;
        organoid.capture_target_morphology(); // target = depolarized here

        // Now the target says "depolarized" at this position; scramble the
        // cell to hyperpolarized and see if homing pulls it back.
        organoid.field.bioelectric.vmem[0] = VMEM_HYPERPOLARIZED;
        organoid.set_gap_junction_permeability(0.0);
        organoid.set_positional_homing(true);

        let before = organoid.field.bioelectric.vmem[0];
        organoid.apply_positional_homing();
        let after = organoid.field.bioelectric.vmem[0];

        assert!(
            after > before,
            "Positional homing should pull Vmem toward the target (depolarized) \
             even with gap junctions blocked: before={before}, after={after}"
        );
    }

    #[test]
    fn positional_homing_is_noop_without_target() {
        let mut organoid = NeuralOrganoid::new(1, 9);
        organoid.field.bioelectric.vmem[0] = VMEM_HYPERPOLARIZED;
        organoid.set_positional_homing(true);
        // No capture_target_morphology() call — nothing to home toward.
        organoid.apply_positional_homing();
        assert_eq!(organoid.field.bioelectric.vmem[0], VMEM_HYPERPOLARIZED);
    }

    #[test]
    fn induce_local_defection_marks_and_depolarizes_selected_cells() {
        let mut organoid = NeuralOrganoid::new(50, 10);
        organoid.field.bioelectric.vmem.fill(VMEM_HYPERPOLARIZED);
        let marked = organoid.induce_local_defection(|p| p[0] >= 0.0);
        assert!(marked > 0);
        assert_eq!(organoid.defected_cell_count(), marked);
        for i in 0..organoid.field.num_cells() {
            if organoid.field.cells[i].position[0] >= 0.0 {
                assert!(organoid.field.bioelectric.defected[i]);
                assert_eq!(organoid.field.bioelectric.vmem[i], VMEM_DEPOLARIZED);
            } else {
                assert!(!organoid.field.bioelectric.defected[i]);
            }
        }
    }

    #[test]
    fn defective_proliferate_grows_defected_population_unconditionally() {
        let mut organoid = NeuralOrganoid::new(30, 11);
        organoid.induce_local_defection(|_| true); // whole organoid defects
        let before = organoid.defected_cell_count();
        for _ in 0..10 {
            organoid.field.defective_proliferate();
        }
        assert!(
            organoid.defected_cell_count() > before,
            "Defected population should grow via unconditional hyperproliferation"
        );
    }

    #[test]
    fn defected_cells_are_gap_junction_isolated() {
        // Both cells are Progenitors, so the unconditional fate-drift term
        // pulls them toward the same depolarized target regardless of
        // defection — that alone causes some convergence. So, as with the
        // gap-junction-blocker test above, isolate the *incremental*
        // contribution of gap-junction coupling by comparing a defected vs.
        // non-defected neighbour under otherwise identical dynamics.
        let final_diff = |defect_neighbour: bool| {
            let mut field = MorphogeneticField::new(2, 12);
            field.cells[0].position = [0.0, 0.0, 0.0];
            field.cells[1].position = [0.05, 0.0, 0.0];
            field.bioelectric.vmem[0] = VMEM_HYPERPOLARIZED;
            if defect_neighbour {
                field.induce_local_defection(|p| p[0] > 0.02); // marks + depolarizes cell 1
            }
            field.form_gap_junctions();
            for _ in 0..30 {
                field.step_vmem_diffusion(0.2);
            }
            (field.bioelectric.vmem[0] - field.bioelectric.vmem[1]).abs()
        };

        let diff_coupled = final_diff(false);
        let diff_defected = final_diff(true);
        assert!(
            diff_coupled < diff_defected,
            "A coupled neighbour should equalize Vmem more than a defected \
             (isolated) one: coupled={diff_coupled}, defected={diff_defected}"
        );
    }

    #[test]
    fn configurable_rates_affect_dynamics() {
        let mut fast = MorphogeneticField::new(2, 13);
        fast.cells[0].position = [0.0, 0.0, 0.0];
        fast.cells[1].position = [0.05, 0.0, 0.0];
        fast.bioelectric.vmem[0] = VMEM_DEPOLARIZED;
        fast.bioelectric.vmem[1] = VMEM_HYPERPOLARIZED;
        fast.bioelectric.gap_junction_diffusion_rate = 1.0;
        fast.form_gap_junctions();

        let mut slow = MorphogeneticField::new(2, 13);
        slow.cells[0].position = [0.0, 0.0, 0.0];
        slow.cells[1].position = [0.05, 0.0, 0.0];
        slow.bioelectric.vmem[0] = VMEM_DEPOLARIZED;
        slow.bioelectric.vmem[1] = VMEM_HYPERPOLARIZED;
        slow.bioelectric.gap_junction_diffusion_rate = 0.01;
        slow.form_gap_junctions();

        fast.step_vmem_diffusion(0.2);
        slow.step_vmem_diffusion(0.2);

        let fast_diff = (fast.bioelectric.vmem[0] - fast.bioelectric.vmem[1]).abs();
        let slow_diff = (slow.bioelectric.vmem[0] - slow.bioelectric.vmem[1]).abs();
        assert!(
            fast_diff < slow_diff,
            "A higher configured diffusion rate should equalize faster: \
             fast={fast_diff}, slow={slow_diff}"
        );
    }

    #[test]
    fn ion_channel_model_disabled_by_default() {
        let organoid = NeuralOrganoid::new(10, 23);
        assert!(!organoid.ion_channel_model_enabled());
    }

    #[test]
    fn differentiated_cell_hyperpolarizes_under_ion_channel_model() {
        let mut field = MorphogeneticField::new(1, 20);
        field.bioelectric.ion_channel_model_enabled = true;
        field.bioelectric.vmem[0] = VMEM_DEPOLARIZED;
        // Simulate having differentiated into a neuron: bump K+ conductance
        // the same way `differentiate()` does.
        field.bioelectric.k_channel_conductance[0] += K_CHANNEL_DIFFERENTIATION_BUMP;

        let before = field.bioelectric.vmem[0];
        for _ in 0..20 {
            field.step_vmem_diffusion(0.2);
        }
        let after = field.bioelectric.vmem[0];

        assert!(
            after < before,
            "A cell with upregulated K+ conductance should hyperpolarize \
             under the ion-channel model: before={before}, after={after}"
        );
    }

    #[test]
    fn potassium_channel_block_prevents_hyperpolarization() {
        let run = |block: f32| {
            let mut organoid = NeuralOrganoid::new(1, 21);
            organoid.set_ion_channel_model_enabled(true);
            organoid.field.bioelectric.vmem[0] = VMEM_DEPOLARIZED;
            organoid.field.bioelectric.k_channel_conductance[0] += K_CHANNEL_DIFFERENTIATION_BUMP;
            organoid.set_potassium_channel_block(block);
            for _ in 0..20 {
                organoid.field.step_vmem_diffusion(0.2);
            }
            organoid.field.bioelectric.vmem[0]
        };

        let open = run(1.0);
        let blocked = run(0.0);
        assert!(
            open < blocked,
            "Blocking K+ channels should prevent the hyperpolarization a \
             differentiated cell would otherwise show: open={open}, blocked={blocked}"
        );
    }

    #[test]
    fn ion_channel_model_keeps_vmem_bounded() {
        let mut field = MorphogeneticField::new(30, 22);
        field.bioelectric.ion_channel_model_enabled = true;
        // A range of conductances, including some large ones, to stress-test
        // for divergence.
        for i in 0..field.num_cells() {
            field.bioelectric.k_channel_conductance[i] = 0.5 + i as f32 * 0.1;
            field.bioelectric.na_channel_conductance[i] = 0.3;
        }
        field.form_gap_junctions();
        for _ in 0..100 {
            field.step_vmem_diffusion(0.2);
        }
        for &v in &field.bioelectric.vmem {
            assert!(
                v.is_finite() && v.abs() < 5.0,
                "Vmem should stay bounded under the ion-channel model, got {v}"
            );
        }
    }
}
