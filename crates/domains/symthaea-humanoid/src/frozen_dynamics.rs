// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Single-sample dynamics view for one deterministic control preparation cycle.
//!
//! Live simulation/hardware providers may advance between repeated queries. A
//! semantic/motor preparation step should therefore not derive one reference from
//! dynamics sample A and then run inverse dynamics against a newly sampled B
//! while describing both as one evidence lineage.
//!
//! This adapter captures each currently exposed dynamics contract once, then
//! replays those immutable snapshots to the existing hierarchy for the duration
//! of one preparation call. Terrain remains delegated because terrain sampling is
//! spatial/query-dependent rather than a single body-state snapshot.

use crate::contact::ContactFrame;
use crate::dynamics::{RigidBodyDynamicsProvider, RigidBodyDynamicsSnapshot};
use crate::floating_base::{FloatingBaseDynamicsProvider, FloatingBaseDynamicsSnapshot};
use crate::full_dynamics::{FullRigidBodyDynamicsProvider, FullRigidBodyDynamicsSnapshot};
use crate::terrain::{TerrainProbe, TerrainSample};
use crate::types::HumanoidState;

/// Immutable per-cycle dynamics bundle over an arbitrary terrain/environment
/// provider. This is a local execution aid, not persistent qualification evidence.
pub struct FrozenHumanoidDynamicsEnvironment<'a, T: ?Sized> {
    terrain: &'a T,
    rigid: Option<RigidBodyDynamicsSnapshot>,
    full: Option<FullRigidBodyDynamicsSnapshot>,
    floating: Option<FloatingBaseDynamicsSnapshot>,
}

impl<'a, T> FrozenHumanoidDynamicsEnvironment<'a, T>
where
    T: TerrainProbe
        + RigidBodyDynamicsProvider
        + FullRigidBodyDynamicsProvider
        + FloatingBaseDynamicsProvider
        + ?Sized,
{
    /// Capture every dynamics contract exactly once for this body/contact state.
    pub fn capture(terrain: &'a T, state: &HumanoidState, contacts: &ContactFrame) -> Self {
        Self {
            terrain,
            rigid: terrain.dynamics_snapshot(state, contacts),
            full: terrain.full_dynamics_snapshot(state, contacts),
            floating: terrain.floating_base_dynamics_snapshot(),
        }
    }

    pub fn rigid_snapshot(&self) -> Option<&RigidBodyDynamicsSnapshot> {
        self.rigid.as_ref()
    }

    pub fn full_snapshot(&self) -> Option<&FullRigidBodyDynamicsSnapshot> {
        self.full.as_ref()
    }

    pub fn floating_snapshot(&self) -> Option<&FloatingBaseDynamicsSnapshot> {
        self.floating.as_ref()
    }
}

impl<T: TerrainProbe + ?Sized> TerrainProbe for FrozenHumanoidDynamicsEnvironment<'_, T> {
    fn sample(&self, world_xy_m: [f64; 2]) -> TerrainSample {
        self.terrain.sample(world_xy_m)
    }
}

impl<T: ?Sized> RigidBodyDynamicsProvider for FrozenHumanoidDynamicsEnvironment<'_, T> {
    fn dynamics_snapshot(
        &self,
        _state: &HumanoidState,
        _contacts: &ContactFrame,
    ) -> Option<RigidBodyDynamicsSnapshot> {
        self.rigid.clone()
    }
}

impl<T: ?Sized> FullRigidBodyDynamicsProvider for FrozenHumanoidDynamicsEnvironment<'_, T> {
    fn full_dynamics_snapshot(
        &self,
        _state: &HumanoidState,
        _contacts: &ContactFrame,
    ) -> Option<FullRigidBodyDynamicsSnapshot> {
        self.full.clone()
    }
}

impl<T: ?Sized> FloatingBaseDynamicsProvider for FrozenHumanoidDynamicsEnvironment<'_, T> {
    fn floating_base_dynamics_snapshot(&self) -> Option<FloatingBaseDynamicsSnapshot> {
        self.floating.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphology::HumanoidMorphology;
    use crate::terrain::FlatTerrain;
    use crate::types::HumanoidState;

    #[test]
    fn captured_flat_environment_replays_same_full_snapshot_identity() {
        let state = HumanoidState::standing_for(HumanoidMorphology::Dmc21);
        let contacts = ContactFrame::estimated_from_state(&state, 0.05);
        let frozen = FrozenHumanoidDynamicsEnvironment::capture(&FlatTerrain, &state, &contacts);
        let first = frozen.full_dynamics_snapshot(&state, &contacts).unwrap();
        let second = frozen.full_dynamics_snapshot(&state, &contacts).unwrap();
        assert_eq!(first.model_id, second.model_id);
        assert_eq!(first.sampled_at_s.to_bits(), second.sampled_at_s.to_bits());
        assert_eq!(first.mass_matrix, second.mass_matrix);
    }
}
