// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-futures-symtropy
//!
//! The observation firewall for the Symthaea Futures Laboratory
//! (`SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`). This is the crate the plan calls "the
//! actual hard part" — not the statistics layer.
//!
//! ## Naming note
//!
//! Despite the crate name, Phase 1's concrete ground-truth adapter (see [`ecological`]) wraps
//! [`symthaea_alife::EarthForcedEnvironment`] / [`symthaea_alife::Population`] (same-workspace,
//! FEP-based, already producing a real bifurcation-collapse scenario per
//! `ALIFE_PLAN_2026-07-08.md` Phases 5a/7) — not the separate Symtropy game engine
//! (`symtropy/`, a different Cargo workspace entirely). "Symtropy" here is used the way the
//! plan's originating conversation used it: shorthand for "a controllable simulated world with
//! hidden ground truth," the category `symthaea-alife` satisfies today. A future scenario
//! family that genuinely needs the Symtropy engine would be a real cross-workspace integration,
//! not attempted in Phase 1.
//!
//! ## The firewall
//!
//! [`ObservationPolicy`] is generic over `type GroundTruth` / `type Observation` — this crate
//! doesn't hardcode one shared ground-truth/observation shape at the top level (the plan
//! commits Phase 1 to at least two structurally different scenario families, and forcing them
//! into one shape now would just mean a breaking redesign later). Each scenario submodule (e.g.
//! [`ecological`]) defines its own ground-truth type and keeps it **out of every public
//! signature of `symthaea-futures-core`, `-state`, or `-ensemble`** — those crates don't (and, by
//! construction, currently *can't*: none of them depend on this crate) import it. Only the
//! `Observation` associated type crosses that boundary.
//!
//! ## Masking primitive (from the `symthaea-futures-state` reuse spike) — a *later* pipeline stage
//!
//! `symthaea_futures_state::mask_observation` is a leakage-test-safe lerp-toward-belief
//! primitive, but `ObservationPolicy::observe` cannot call it directly — `observe`'s signature
//! has no belief-state parameter, and `mask_observation` requires one. The two stages are
//! sequential, not alternatives:
//! - **Stage A (this crate)**: `ObservationPolicy::observe` turns hidden ground truth into a
//!   small, self-evidently-safe summary (`Option<T>` fields — safe because excluded data is
//!   structurally never read, not merely blended toward a prior).
//! - **Stage B (not built yet)**: a scenario-observation → `symthaea_fep::Observation` adapter,
//!   which *would* use `mask_observation` (it needs a `BeliefState` to lerp toward), feeding the
//!   FEP-driven ensemble rung. Don't reach for `mask_observation` inside an `ObservationPolicy`
//!   impl — that's the wrong stage for it.
//!
//! Do **not** reach for `symthaea_fep::markov_blanket::MarkovBoundaryOperator::gate_observation`
//! for either stage — despite the superficial resemblance to `mask_observation` (both lerp an
//! observation toward something), its blend factor is derived from organism physiology (a
//! trust/noise model), not a declared visibility policy, and isn't guaranteed to sit at exactly
//! `0.0` for a field meant to be hidden. See `symthaea-futures-state`'s module docs for the full
//! reasoning.

pub mod ecological;
pub mod evolutionary_rescue;
pub mod predator_prey;

/// Implemented once per scenario family. `observe` is the only sanctioned path from a
/// scenario's ground-truth type to its observation type — implementations must not expose any
/// other way to read ground truth.
pub trait ObservationPolicy {
    type GroundTruth;
    type Observation;

    fn observe(&mut self, truth: &Self::GroundTruth, tick: u64) -> Self::Observation;
}
