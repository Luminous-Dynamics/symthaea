// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! World-genesis provenance for reproducible synthetic environments.

use serde::{Deserialize, Serialize};

use crate::{digest::{DigestAlgorithm, TypedDigest}, types::WorldDescriptor};

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
        feed(&mut hasher, self.world.world_id.0.as_bytes());
        feed(&mut hasher, self.world.lineage_id.0.as_bytes());
        for digest in [
            &self.simulation_kernel_digest,
            &self.physics_profile_digest,
            &self.asset_manifest_digest,
            &self.initial_state_digest,
        ] {
            feed(&mut hasher, digest.domain.as_bytes());
            feed(&mut hasher, digest.value.as_bytes());
        }
        hasher.update(&[determinism_tag(self.determinism)]);
        match self.seed {
            Some(seed) => {
                hasher.update(&[1]);
                hasher.update(&seed.to_le_bytes());
            }
            None => hasher.update(&[0]),
        };
        feed(&mut hasher, self.timebase_id.as_bytes());
        TypedDigest::new(
            "symthaea.world-genesis.v1",
            DigestAlgorithm::Blake3,
            hasher.finalize().to_hex().to_string(),
        )
        .map_err(|error| WorldGenesisError::InvalidDigest(error.to_string()))
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
    use crate::{digest::TypedDigest, types::{RealityLayer, WorldId, WorldLineageId, WorldOrigin}};

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
}
