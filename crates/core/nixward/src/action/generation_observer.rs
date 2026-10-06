// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Read-only observer for the current NixOS system profile generation.

use std::path::{Path, PathBuf};
use thiserror::Error;

const SYSTEM_PROFILE: &str = "/nix/var/nix/profiles/system";
const GENERATION_PREFIX: &str = "system-";
const GENERATION_SUFFIX: &str = "-link";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NixVerifiedNixOSGenerationV1 {
    generation: u64,
    profile_path: PathBuf,
    link_target: PathBuf,
}

impl NixVerifiedNixOSGenerationV1 {
    pub(crate) fn from_observer() -> Result<Self, NixOSGenerationObserverErrorV1> {
        let profile = Path::new(SYSTEM_PROFILE);
        let first = std::fs::read_link(profile)
            .map_err(NixOSGenerationObserverErrorV1::ReadLink)?;
        let second = std::fs::read_link(profile)
            .map_err(NixOSGenerationObserverErrorV1::ReadLink)?;
        if first != second {
            return Err(NixOSGenerationObserverErrorV1::ProfileChangedDuringObservation);
        }
        let generation = parse_generation_link(first.as_path())?;
        Ok(Self {
            generation,
            profile_path: profile.to_path_buf(),
            link_target: first,
        })
    }

    pub fn generation(&self) -> u64 { self.generation }
    pub fn profile_path(&self) -> &Path { &self.profile_path }
    pub fn link_target(&self) -> &Path { &self.link_target }

    #[cfg(test)]
    pub(crate) fn for_test(generation: u64) -> Self {
        Self {
            generation,
            profile_path: PathBuf::from(SYSTEM_PROFILE),
            link_target: PathBuf::from(format!(
                "system-{generation}-link"
            )),
        }
    }
}

fn parse_generation_link(target: &Path) -> Result<u64, NixOSGenerationObserverErrorV1> {
    validate_profile_link_target(target)?;
    let name = target
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or(NixOSGenerationObserverErrorV1::InvalidProfileTarget)?;
    let middle = name
        .strip_prefix(GENERATION_PREFIX)
        .and_then(|value| value.strip_suffix(GENERATION_SUFFIX))
        .ok_or(NixOSGenerationObserverErrorV1::InvalidProfileTarget)?;
    if middle.is_empty() || !middle.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(NixOSGenerationObserverErrorV1::InvalidGenerationLink);
    }
    let generation = middle
        .parse::<u64>()
        .map_err(|_| NixOSGenerationObserverErrorV1::GenerationOverflow)?;
    if generation == 0 {
        return Err(NixOSGenerationObserverErrorV1::InvalidGeneration);
    }
    Ok(generation)
}

fn validate_profile_link_target(
    target: &Path,
) -> Result<(), NixOSGenerationObserverErrorV1> {
    let profile_parent = Path::new(SYSTEM_PROFILE)
        .parent()
        .ok_or(NixOSGenerationObserverErrorV1::InvalidProfileTarget)?;

    if target.is_absolute() {
        if target.parent() != Some(profile_parent) {
            return Err(NixOSGenerationObserverErrorV1::InvalidProfileTarget);
        }
    } else {
        let mut components = target.components();
        match (components.next(), components.next()) {
            (Some(std::path::Component::Normal(_)), None) => {}
            _ => return Err(NixOSGenerationObserverErrorV1::InvalidProfileTarget),
        }
    }

    Ok(())
}

#[derive(Debug, Error)]
pub enum NixOSGenerationObserverErrorV1 {
    #[error("could not read NixOS system profile symlink: {0}")]
    ReadLink(#[source] std::io::Error),
    #[error("NixOS system profile changed during generation observation")]
    ProfileChangedDuringObservation,
    #[error("NixOS system profile target is not a valid generation link")]
    InvalidProfileTarget,
    #[error("NixOS generation link is malformed")]
    InvalidGenerationLink,
    #[error("NixOS generation number overflowed u64")]
    GenerationOverflow,
    #[error("NixOS generation must be non-zero")]
    InvalidGeneration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_only_exact_system_generation_link_names() {
        assert_eq!(parse_generation_link(Path::new("system-42-link")).unwrap(), 42);
        assert!(parse_generation_link(Path::new("system-link")).is_err());
        assert!(parse_generation_link(Path::new("system-0-link")).is_err());
        assert!(parse_generation_link(Path::new("system-42")).is_err());
        assert!(parse_generation_link(Path::new("other-42-link")).is_err());
    }

    #[test]
    fn generation_link_target_must_resolve_within_fixed_profile_directory() {
        assert!(parse_generation_link(Path::new("/nix/var/nix/profiles/system-42-link")).is_ok());
        assert!(parse_generation_link(Path::new("system-42-link")).is_ok());
        assert!(parse_generation_link(Path::new("/tmp/system-42-link")).is_err());
        assert!(parse_generation_link(Path::new("../profiles/system-42-link")).is_err());
        assert!(parse_generation_link(Path::new("nested/system-42-link")).is_err());
    }

    #[test]
    fn overflow_and_non_numeric_generation_links_fail_closed() {
        assert!(matches!(
            parse_generation_link(Path::new("system-18446744073709551616-link")),
            Err(NixOSGenerationObserverErrorV1::GenerationOverflow)
        ));
        assert!(matches!(
            parse_generation_link(Path::new("system-4x-link")),
            Err(NixOSGenerationObserverErrorV1::InvalidGenerationLink)
        ));
    }
}
