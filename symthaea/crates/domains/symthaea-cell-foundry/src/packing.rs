// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Opt-in cell-packing correction pass, using real rigid-body physics
//! (`symtropy-physics`) instead of the point-particle model everywhere
//! else in this crate.
//!
//! Cells are otherwise bare `[f32; 3]` positions that can freely overlap
//! forever — most visibly, every daughter cell spawned by `proliferate`,
//! `regenerative_proliferate`, or `defective_proliferate` starts at its
//! parent's position plus a `±0.02` jitter, with nothing that ever
//! separates them afterward. This module adds a correction pass that
//! resolves that via real collision response, without touching how
//! position is stored or read anywhere else in the crate.
//!
//! ## Why the world is rebuilt from scratch every call
//!
//! `symtropy-physics` has no body-removal API. Cells are added and removed
//! constantly (proliferation, amputation, defection), so maintaining
//! persistent body handles across days would mean re-deriving the missing
//! removal behaviour ourselves. Instead, [`resolve_packing`] builds a
//! fresh, zero-gravity [`PhysicsWorld`] from every cell's *current*
//! position each time it's called, steps it a few times, and writes the
//! resolved positions straight back into `field.cells[..].position` — the
//! world is then dropped. This is O(n) extra work on top of the O(n log n)
//! (or O(n^2) for small n) broadphase the physics engine already does
//! internally, which is cheap next to what this crate already spends on
//! the chemical reaction-diffusion field every day. Amputated cells simply
//! aren't included next time; there is nothing to keep in sync.
//!
//! ## Why this doesn't change any existing behaviour
//!
//! This pass is only invoked when `NeuralOrganoid::packing_enabled()` is
//! `true` (default `false` — see `crate::bioelectric`), so every existing
//! test and example is byte-for-byte unaffected unless it explicitly opts
//! in. Even when enabled, [`CELL_PACKING_RADIUS`] is deliberately small
//! relative to the typical nearest-neighbour spacing at the cell densities
//! used elsewhere in this crate (roughly 0.2-0.35 for 100-1,000 cells in a
//! `[-1,1]^3` cube) — most cells are never within packing range of each
//! other, so this only acts where cells are pathologically overlapping
//! (fresh proliferation sites, wound edges, tumour cores), which is
//! exactly where it should.

use nalgebra::SVector;
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;

use crate::morphogenetic_consciousness::MorphogeneticField;

/// Physical radius given to every cell for packing/volume-exclusion
/// purposes. See module docs for why this is deliberately small.
pub const CELL_PACKING_RADIUS: f64 = 0.03;
/// Substeps run per call — enough for a fresh overlap to separate without
/// over-stepping, given the world carries no state across calls.
const PACKING_SUBSTEPS: u32 = 3;
/// Timestep per substep (arbitrary simulation-time units, not seconds —
/// this crate's whole model runs on "1 tick = 1 day," not real time).
const PACKING_DT: f64 = 0.5;

/// Rebuild a fresh physics world from every cell's current position, step
/// it a few times to resolve overlaps via real collision response, and
/// write the corrected positions back. No-op on an empty field.
pub(crate) fn resolve_packing(field: &mut MorphogeneticField) {
    let n = field.cells.len();
    if n == 0 {
        return;
    }

    let mut world = PhysicsWorld::<3>::new(SVector::zeros());
    let handles: Vec<_> = (0..n)
        .map(|i| {
            let p = field.cells[i].position;
            world.add_sphere(
                Point::new([p[0] as f64, p[1] as f64, p[2] as f64]),
                CELL_PACKING_RADIUS,
                1.0,
            )
        })
        .collect();

    for _ in 0..PACKING_SUBSTEPS {
        world.step(PACKING_DT);
    }

    for (i, handle) in handles.into_iter().enumerate() {
        if let Some(body) = world.body(handle) {
            let p = body.position();
            field.cells[i].position = [p[0] as f32, p[1] as f32, p[2] as f32];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_separates_fully_overlapping_cells() {
        let mut field = MorphogeneticField::new(2, 1);
        field.cells[0].position = [0.0, 0.0, 0.0];
        field.cells[1].position = [0.0, 0.0, 0.0];

        resolve_packing(&mut field);

        let d = MorphogeneticField::distance(&field.cells[0].position, &field.cells[1].position);
        assert!(
            d > 0.0,
            "Packing should separate two fully-overlapping cells, got distance={d}"
        );
    }

    #[test]
    fn packing_is_a_noop_for_well_spaced_cells() {
        let mut field = MorphogeneticField::new(2, 2);
        field.cells[0].position = [0.0, 0.0, 0.0];
        field.cells[1].position = [1.0, 0.0, 0.0];
        let before = field.cells[1].position;

        resolve_packing(&mut field);

        let after = field.cells[1].position;
        let drift = MorphogeneticField::distance(&before, &after);
        assert!(
            drift < 0.01,
            "Packing should not perturb cells already well beyond packing range, drift={drift}"
        );
    }

    #[test]
    fn packing_handles_empty_field() {
        let mut field = MorphogeneticField::new(0, 3);
        resolve_packing(&mut field); // must not panic
        assert_eq!(field.num_cells(), 0);
    }
}
