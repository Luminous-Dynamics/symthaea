// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! World-genesis provenance for reproducible synthetic environments.

use serde::{Deserialize, Serialize};

use crate::{
    digest::{DigestAlgorithm, TypedDigest},
    types::{RealityLayer, WorldDescriptor, WorldOrigin, WorldRelation},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeterminismClass {
    Deterministic,
    SeededStochastic,
    HostDependent,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldGenesisManifest {
    pub schema_version: u32,
    pub world: WorldDescriptor,
    pub simulation_kernel_digest: TypedDigest,
    pub physics_profile_digest: TypedDigest,
    pub asset_manifest_digest: TypedDigest,
    pub initial_state_digest: TypedDigest,
    pub determinism: DeterminismClass,
    pub seed: Option<u64>,
    pub timebase_id: String,
}

impl WorldGenesisManifest {
    pub fn validate(&self) -> Result<(), WorldGenesisError> {
        if self.schema_version == 0 {
            return Err(WorldGenesisError::InvalidSchemaVersion);
        }
        self.world
            .validate()
            .map_err(|error| WorldGenesisError::InvalidWorld(error.to_string()))?;
        for digest in [
            &self.simulation_kernel_digest,
            &self.physics_profile_digest,
            &self.asset_manifest_digest,
            &self.initial_state_digest,
        ] {
            digest
                .validate()
                .map_err(|error| WorldGenesisError::InvalidDigest(error.to_string()))?;
        }
        if self.timebase_id.trim().is_empty() {
            return Err(WorldGenesisError::MissingTimebase);
        }
        if matches!(self.determinism, DeterminismClass::SeededStochastic) && self.seed.is_none() {
            return Err(WorldGenesisError::SeedRequired);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<TypedDigest, WorldGenesisError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        feed(&mut hasher, b"symthaea.world-genesis.v1");
        hasher.update(&self.schema_version.to_le_bytes());
        feed_world(&mut hasher, &self.world);
        for digest in [
            &self.simulation_kernel_digest,
            &self.physics_profile_digest,
            &self.asset_manifest_digest,
            &self.initial_state_digest,
        ] {
            feed_typed_digest(&mut hasher, digest);
        }
        hasher.update(&[determinism_tag(self.determinism)]);
        match self.seed {
            Some(seed) => {
                hasher.update(&[1]);
                hasher.update(&seed.to_le_bytes());
            }
            None => {
                hasher.update(&[0]);
            }
        }
        feed(&mut hasher, self.timebase_id.as_bytes());
        TypedDigest::new(
            "symthaea.world-genesis.v1",
            DigestAlgorithm::Blake3,
            hasher.finalize().to_hex().to_string(),
        )
        .map_err(|error| WorldGenesisError::InvalidDigest(error.to_string()))
    }
}

fn feed_world(hasher: &mut blake3::Hasher, world: &WorldDescriptor) {
    feed(hasher, world.world_id.0.as_bytes());
    feed(hasher, world.lineage_id.0.as_bytes());
    hasher.update(&[layer_tag(world.layer)]);
    feed(hasher, world.creator_id.as_bytes());
    hasher.update(&world.generation_depth.to_le_bytes());
    match &world.parent {
        Some(parent) => {
            hasher.update(&[1]);
            feed(hasher, parent.world_id.0.as_bytes());
            feed(hasher, parent.lineage_id.0.as_bytes());
            hasher.update(&[relation_tag(&parent.relation)]);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    match &world.origin {
        WorldOrigin::PhysicalSensorium => {
            hasher.update(&[0]);
        }
        WorldOrigin::DigitalHost { host_kind } => {
            hasher.update(&[1]);
            feed(hasher, host_kind.as_bytes());
        }
        WorldOrigin::CounterfactualBranch => {
            hasher.update(&[2]);
        }
        WorldOrigin::ReplayArtifact => {
            hasher.update(&[3]);
        }
        WorldOrigin::DreamEngine => {
            hasher.update(&[4]);
        }
        WorldOrigin::ImportedExternal { source } => {
            hasher.update(&[5]);
            feed(hasher, source.as_bytes());
        }
        WorldOrigin::Unknown => {
            hasher.update(&[6]);
        }
    }
}

fn feed_typed_digest(hasher: &mut blake3::Hasher, digest: &TypedDigest) {
    feed(hasher, digest.domain.as_bytes());
    match &digest.algorithm {
        DigestAlgorithm::Blake3 => {
            hasher.update(&[0]);
        }
        DigestAlgorithm::Sha256 => {
            hasher.update(&[1]);
        }
        DigestAlgorithm::Other(name) => {
            hasher.update(&[2]);
            feed(hasher, name.as_bytes());
        }
    }
    feed(hasher, digest.value.as_bytes());
}

fn layer_tag(layer: RealityLayer) -> u8 {
    match layer {
        RealityLayer::PhysicalGrounded => 0,
        RealityLayer::DigitalCommitted => 1,
        RealityLayer::Counterfactual => 2,
        RealityLayer::Replay => 3,
        RealityLayer::Dream => 4,
        RealityLayer::Imported => 5,
        RealityLayer::Unknown => 6,
    }
}

fn relation_tag(relation: &WorldRelation) -> u8 {
    match relation {
        WorldRelation::CounterfactualOf => 0,
        WorldRelation::ReplayOf => 1,
        WorldRelation::DreamedFrom => 2,
        WorldRelation::ImportedFrom => 3,
        WorldRelation::SpawnedFrom => 4,
    }
}

fn determinism_tag(value: DeterminismClass) -> u8 {
    match value {
        DeterminismClass::Deterministic => 0,
        DeterminismClass::SeededStochastic => 1,
        DeterminismClass::HostDependent => 2,
        DeterminismClass::Unknown => 3,
    }
}

fn feed(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WorldGenesisError {
    #[error("genesis schema version must be non-zero")]
    InvalidSchemaVersion,
    #[error("invalid world descriptor: {0}")]
    InvalidWorld(String),
    #[error("invalid typed digest: {0}")]
    InvalidDigest(String),
    #[error("timebase id may not be empty")]
    MissingTimebase,
    #[error("seeded-stochastic worlds require an explicit seed")]
    SeedRequired,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{digest::TypedDigest, types::{WorldId, WorldLineageId}};

    fn d(domain: &str) -> TypedDigest {
        TypedDigest::blake3(domain, domain.as_bytes()).unwrap()
    }

    fn world() -> WorldDescriptor {
        WorldDescriptor {
            world_id: WorldId("garden".into()),
            lineage_id: WorldLineageId("garden-lineage".into()),
            layer: RealityLayer::DigitalCommitted,
            origin: WorldOrigin::DigitalHost { host_kind: "symtropy".into() },
            parent: None,
            generation_depth: 0,
            creator_id: "symthaea".into(),
        }
    }

    #[test]
    fn seeded_world_must_record_its_seed() {
        let manifest = WorldGenesisManifest {
            schema_version: 1,
            world: world(),
            simulation_kernel_digest: d("kernel.v1"),
            physics_profile_digest: d("physics.v1"),
            asset_manifest_digest: d("assets.v1"),
            initial_state_digest: d("state.v1"),
            determinism: DeterminismClass::SeededStochastic,
            seed: None,
            timebase_id: "studio-frame".into(),
        };
        assert_eq!(manifest.validate(), Err(WorldGenesisError::SeedRequired));
    }

    #[test]
    fn changing_world_provenance_changes_genesis_digest() {
        let mut a = WorldGenesisManifest {
            schema_version: 1,
            world: world(),
            simulation_kernel_digest: d("kernel.v1"),
            physics_profile_digest: d("physics.v1"),
            asset_manifest_digest: d("assets.v1"),
            initial_state_digest: d("state.v1"),
            determinism: DeterminismClass::Deterministic,
            seed: None,
            timebase_id: "studio-frame".into(),
        };
        let before = a.digest().unwrap();
        a.world.creator_id = "different-creator".into();
        let after = a.digest().unwrap();
        assert_ne!(before, after);
    }
}
