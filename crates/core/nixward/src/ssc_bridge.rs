//! Read-only bridge from Nixward's live NixOS observation into SSC.
//!
//! This module deliberately stops at observation. It does not authorize,
//! execute, reboot, or mutate the target. The exact NixOS realization identity
//! is captured alongside running/booted/profile facts so the SSC/Nix adapter
//! can bind generation-sensitive operations to what was actually observed.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sovereign_state_compiler::{
    ContentDigest, ResourceRef, TargetId, TargetProfile, TargetSnapshot,
};
use sovereign_state_compiler_nix::{
    default_nixos_capabilities, nixos_generation_resource,
};
use thiserror::Error;

use crate::observe::generations::{GenerationInfo, GenerationObserver};

pub const NIXOS_SYSTEM_PROFILE: &str = "/nix/var/nix/profiles/system";
pub const NIXOS_CURRENT_SYSTEM: &str = "/run/current-system";
pub const NIXOS_BOOTED_SYSTEM: &str = "/run/booted-system";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixGenerationObservation {
    pub number: u64,
    pub realization: String,
    pub current: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixSystemObservation {
    pub generations: Vec<NixGenerationObservation>,
    pub system_profile_realization: String,
    pub current_system_realization: String,
    pub booted_system_realization: String,
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum SscObservationError {
    #[error("failed to observe NixOS state: {0}")]
    Io(String),
    #[error("failed to serialize NixOS observation: {0}")]
    Serialization(String),
    #[error("NixOS generation observation has no exact realization")]
    MissingRealization,
}

impl From<std::io::Error> for SscObservationError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

impl From<serde_json::Error> for SscObservationError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serialization(value.to_string())
    }
}

impl NixSystemObservation {
    pub fn observe() -> Result<Self, SscObservationError> {
        let generations = GenerationObserver::list_generations()?
            .into_iter()
            .map(Self::generation)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            generations,
            system_profile_realization: read_realization(Path::new(NIXOS_SYSTEM_PROFILE))?,
            current_system_realization: read_realization(Path::new(NIXOS_CURRENT_SYSTEM))?,
            booted_system_realization: read_realization(Path::new(NIXOS_BOOTED_SYSTEM))?,
        })
    }

    fn generation(info: GenerationInfo) -> Result<NixGenerationObservation, SscObservationError> {
        let link = generation_link(info.number);
        let realization = read_realization(&link)?;
        if realization.is_empty() {
            return Err(SscObservationError::MissingRealization);
        }

        Ok(NixGenerationObservation {
            number: info.number,
            realization,
            current: info.current,
        })
    }

    pub fn observation_digest(&self) -> Result<ContentDigest, SscObservationError> {
        let bytes = serde_json::to_vec(self)?;
        Ok(ContentDigest::blake3(&bytes))
    }

    pub fn generation_resource(&self, generation: u64) -> Option<ResourceRef> {
        self.generations
            .iter()
            .find(|entry| entry.number == generation)
            .and_then(|entry| nixos_generation_resource(entry.number, &entry.realization).ok())
    }

    pub fn target_snapshot(
        &self,
        target: impl Into<TargetId>,
        observed_at_ms: u64,
    ) -> Result<TargetSnapshot, SscObservationError> {
        let observation_digest = self.observation_digest()?;
        let resources = self
            .generations
            .iter()
            .map(|entry| {
                nixos_generation_resource(entry.number, &entry.realization)
                    .map_err(|error| {
                        SscObservationError::Io(error.to_string())
                    })
            })
            .collect::<Result<_, _>>()?;

        Ok(TargetSnapshot {
            profile: TargetProfile {
                identity: target.into(),
                platform: "nixos".into(),
                capabilities: default_nixos_capabilities(),
            },
            observed_at_ms,
            observation_digest,
            resources,
        })
    }

    pub fn current_generation(&self) -> Option<&NixGenerationObservation> {
        self.generations.iter().find(|entry| entry.current)
    }

    pub fn system_profile_generation(&self) -> Option<&NixGenerationObservation> {
        self.generations
            .iter()
            .find(|entry| entry.realization == self.system_profile_realization)
    }
}

fn generation_link(generation: u64) -> PathBuf {
    PathBuf::from(format!(
        "/nix/var/nix/profiles/system-{generation}-link"
    ))
}

fn read_realization(link: &Path) -> Result<String, SscObservationError> {
    let resolved = std::fs::canonicalize(link)?;
    let value = resolved.to_string_lossy().into_owned();
    if value.is_empty() {
        return Err(SscObservationError::MissingRealization);
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_link_is_deterministic() {
        assert_eq!(
            generation_link(42),
            PathBuf::from("/nix/var/nix/profiles/system-42-link")
        );
    }

    #[test]
    fn observation_digest_changes_when_realization_changes() {
        let mut observation = NixSystemObservation {
            generations: vec![NixGenerationObservation {
                number: 42,
                realization: "/nix/store/aaa-nixos-system-host".into(),
                current: true,
            }],
            system_profile_realization: "/nix/store/aaa-nixos-system-host".into(),
            current_system_realization: "/nix/store/aaa-nixos-system-host".into(),
            booted_system_realization: "/nix/store/aaa-nixos-system-host".into(),
        };

        let before = observation.observation_digest().expect("digest");
        observation.generations[0].realization = "/nix/store/bbb-nixos-system-host".into();
        let after = observation.observation_digest().expect("digest");

        assert_ne!(before, after);
    }

    #[test]
    fn generation_resource_tracks_exact_observed_realization() {
        let observation = NixSystemObservation {
            generations: vec![
                NixGenerationObservation {
                    number: 42,
                    realization: "/nix/store/aaa-nixos-system-host".into(),
                    current: true,
                },
                NixGenerationObservation {
                    number: 43,
                    realization: "/nix/store/bbb-nixos-system-host".into(),
                    current: false,
                },
            ],
            system_profile_realization: "/nix/store/bbb-nixos-system-host".into(),
            current_system_realization: "/nix/store/aaa-nixos-system-host".into(),
            booted_system_realization: "/nix/store/aaa-nixos-system-host".into(),
        };

        assert_eq!(
            observation.generation_resource(42).expect("resource"),
            nixos_generation_resource(42, "/nix/store/aaa-nixos-system-host")
                .expect("resource")
        );
        assert_eq!(
            observation.system_profile_generation().expect("profile generation").number,
            43
        );
        assert_eq!(
            observation.current_generation().expect("current generation").number,
            42
        );
        assert_ne!(
            observation.current_system_realization,
            observation.system_profile_realization
        );
        assert_eq!(super::NIXOS_GENERATION_RESOURCE_KIND, "nixos-generation");
    }
}
