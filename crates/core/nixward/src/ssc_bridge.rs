//! Read-only bridge from Nixward's live NixOS observation into SSC.
//!
//! This module deliberately stops at observation. It does not authorize,
//! execute, reboot, or mutate the target. The exact NixOS realization identity
//! is captured alongside running/booted/profile facts so the SSC/Nix adapter
//! can bind generation-sensitive operations to what was actually observed.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sovereign_state_compiler::{
    AuthorizedDeploymentPlan, Capability, ContentDigest, ResourceRef, TargetId, TargetProfile,
    TargetSnapshot,
};
use sovereign_state_compiler_nix::{
    default_nixos_capabilities, nixos_generation_resource, NixOSTargetAdapter,
};
use thiserror::Error;

pub const NIXOS_SYSTEM_PROFILE: &str = "/nix/var/nix/profiles/system";
pub const NIXOS_CURRENT_SYSTEM: &str = "/run/current-system";
pub const NIXOS_BOOTED_SYSTEM: &str = "/run/booted-system";
pub const NIXOS_MACHINE_ID: &str = "/etc/machine-id";

const NIXOS_OBSERVATION_DIGEST_DOMAIN: &[u8] =
    b"LUMINOUS-DYNAMICS/SSC/NIXOS-OBSERVATION/v1\0";
const NIXOS_TARGET_ID_DOMAIN: &[u8] = b"LUMINOUS-DYNAMICS/SSC/NIXOS-TARGET-ID/v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixGenerationObservation {
    pub number: u64,
    pub realization: String,
    pub current: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NixSystemObservation {
    /// Privacy-preserving stable identity for this NixOS installation.
    ///
    /// The raw /etc/machine-id is never carried in the observation. The
    /// identity is an application-specific BLAKE3 derivation so callers cannot
    /// silently relabel a live observation as another target.
    pub target_identity: TargetId,
    pub generations: Vec<NixGenerationObservation>,
    pub system_profile_generation: u64,
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
    #[error("NixOS generation observation contains multiple current generations")]
    MultipleCurrentGenerations,
    #[error("NixOS generation observation contains a zero generation ordinal")]
    InvalidGenerationNumber,
    #[error("NixOS realization is not rooted in /nix/store")]
    InvalidRealizationPath,
    #[error("NixOS generation observation contains a duplicate generation ordinal")]
    DuplicateGenerationNumber,
    #[error("NixOS generation observations are not in canonical ordinal order")]
    NonCanonicalGenerationOrder,
    #[error("NixOS generation marked current does not match /run/current-system")]
    CurrentGenerationRealizationMismatch,
    #[error("NixOS system profile does not identify an exact generation")]
    MissingSystemProfileGeneration,
    #[error("NixOS system profile contains no generation links")]
    NoSystemGenerations,
    #[error("NixOS system profile generation does not match its exact realization")]
    SystemProfileRealizationMismatch,
    #[error("authorized deployment requires an unobserved NixOS resource")]
    AuthorizedResourceMissing(ResourceRef),
    #[error("fresh NixOS observation digest does not match the authorized pre-state")]
    AuthorizedObservationDigestMismatch,
    #[error("NixOS observation could not construct the canonical target adapter: {0}")]
    AdapterConstruction(String),
    #[error("NixOS /etc/machine-id is missing or empty")]
    MissingTargetIdentity,
    #[error("NixOS /etc/machine-id must be exactly 32 lowercase hexadecimal characters")]
    InvalidMachineId,
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
        let current_system_realization = read_realization(Path::new(NIXOS_CURRENT_SYSTEM))?;
        let generations = observe_generations(&current_system_realization)?;
        let target_identity = read_target_identity(Path::new(NIXOS_MACHINE_ID))?;

        let observation = Self {
            target_identity,
            generations,
            system_profile_generation: read_profile_generation(Path::new(NIXOS_SYSTEM_PROFILE))?,
            system_profile_realization: read_realization(Path::new(NIXOS_SYSTEM_PROFILE))?,
            current_system_realization: read_realization(Path::new(NIXOS_CURRENT_SYSTEM))?,
            booted_system_realization: read_realization(Path::new(NIXOS_BOOTED_SYSTEM))?,
        };
        observation.validate()?;
        Ok(observation)
    }

    pub fn validate(&self) -> Result<(), SscObservationError> {
        if self.target_identity.0.is_empty() || self.target_identity.0.trim().is_empty() {
            return Err(SscObservationError::MissingTargetIdentity);
        }
        if self.system_profile_generation == 0 {
            return Err(SscObservationError::InvalidGenerationNumber);
        }
        if self.system_profile_realization.is_empty()
            || self.current_system_realization.is_empty()
            || self.booted_system_realization.is_empty()
        {
            return Err(SscObservationError::MissingRealization);
        }
        validate_realization_path(&self.system_profile_realization)?;
        validate_realization_path(&self.current_system_realization)?;
        validate_realization_path(&self.booted_system_realization)?;

        let mut previous_number = None;
        for entry in &self.generations {
            if entry.number == 0 {
                return Err(SscObservationError::InvalidGenerationNumber);
            }
            if entry.realization.is_empty() {
                return Err(SscObservationError::MissingRealization);
            }
            validate_realization_path(&entry.realization)?;
            if let Some(previous) = previous_number {
                if entry.number == previous {
                    return Err(SscObservationError::DuplicateGenerationNumber);
                }
                if entry.number < previous {
                    return Err(SscObservationError::NonCanonicalGenerationOrder);
                }
            }
            previous_number = Some(entry.number);
        }

        let mut current = self.generations.iter().filter(|entry| entry.current);
        let Some(current_generation) = current.next() else {
            return Err(SscObservationError::CurrentGenerationRealizationMismatch);
        };
        if current.next().is_some() {
            return Err(SscObservationError::MultipleCurrentGenerations);
        }
        if current_generation.realization != self.current_system_realization {
            return Err(SscObservationError::CurrentGenerationRealizationMismatch);
        }

        let Some(entry) = self
            .generations
            .iter()
            .find(|entry| entry.number == self.system_profile_generation)
        else {
            return Err(SscObservationError::SystemProfileRealizationMismatch);
        };
        if entry.realization != self.system_profile_realization {
            return Err(SscObservationError::SystemProfileRealizationMismatch);
        }

        Ok(())
    }

    pub fn observation_digest(&self) -> Result<ContentDigest, SscObservationError> {
        let bytes = serde_json::to_vec(self)?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(NIXOS_OBSERVATION_DIGEST_DOMAIN);
        hasher.update(&bytes);
        Ok(ContentDigest {
            algorithm: "blake3".into(),
            value: hasher.finalize().to_hex().to_string(),
        })
    }

    pub fn generation_resource(&self, generation: u64) -> Option<ResourceRef> {
        self.generations
            .iter()
            .find(|entry| entry.number == generation)
            .and_then(|entry| nixos_generation_resource(entry.number, &entry.realization).ok())
    }

    /// Confirm that this captured observation still contains every resource
    /// required by an already-authorized plan. Callers must obtain the
    /// observation through the live observer immediately before mutation;
    /// this method validates supplied evidence but does not perform a new read.
    /// It does not authorize or execute the plan.
    pub fn validate_against_authorized_plan(
        &self,
        authorized: &AuthorizedDeploymentPlan,
    ) -> Result<(), SscObservationError> {
        self.validate()?;

        if self.observation_digest()? != authorized.plan.target_snapshot.observation_digest {
            return Err(SscObservationError::AuthorizedObservationDigestMismatch);
        }

        let observed_resources = self
            .generations
            .iter()
            .map(|entry| nixos_generation_resource(entry.number, &entry.realization))
            .collect::<Result<_, _>>()
            .map_err(|error| SscObservationError::Io(error.to_string()))?;

        for resource in &authorized.plan.intent.required_resources {
            if !observed_resources.contains(resource) {
                return Err(SscObservationError::AuthorizedResourceMissing(resource.clone()));
            }
        }

        Ok(())
    }

    /// Construct the canonical NixOS SSC adapter from this validated observation.
    /// This remains read-only: adapter creation captures the exact observation
    /// snapshot but does not authorize or execute any deployment.
    pub fn target_adapter(
        &self,
        observed_at_ms: u64,
    ) -> Result<NixOSTargetAdapter, SscObservationError> {
        let snapshot = self.target_snapshot(observed_at_ms)?;
        NixOSTargetAdapter::from_snapshot(snapshot)
            .map_err(|error| SscObservationError::AdapterConstruction(error.to_string()))
    }

    pub fn target_snapshot(
        &self,
        observed_at_ms: u64,
    ) -> Result<TargetSnapshot, SscObservationError> {
        self.validate()?;
        let observation_digest = self.observation_digest()?;
        let resources = self
            .generations
            .iter()
            .map(|entry| {
                nixos_generation_resource(entry.number, &entry.realization)
                    .map_err(|error| SscObservationError::Io(error.to_string()))
            })
            .collect::<Result<_, _>>()?;

        Ok(TargetSnapshot {
            profile: TargetProfile {
                identity: self.target_identity.clone(),
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
            .find(|entry| entry.number == self.system_profile_generation)
    }
}

fn read_target_identity(path: &Path) -> Result<TargetId, SscObservationError> {
    let value = std::fs::read_to_string(path)?;
    target_identity_from_machine_id(value.trim())
}

fn target_identity_from_machine_id(machine_id: &str) -> Result<TargetId, SscObservationError> {
    if machine_id.len() != 32
        || !machine_id
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(SscObservationError::InvalidMachineId);
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(NIXOS_TARGET_ID_DOMAIN);
    hasher.update(machine_id.as_bytes());
    Ok(TargetId(format!(
        "nixos-machine:{}",
        hasher.finalize().to_hex()
    )))
}

fn observe_generations(
    current_system_realization: &str,
) -> Result<Vec<NixGenerationObservation>, SscObservationError> {
    let mut generations = Vec::new();

    for entry in std::fs::read_dir(Path::new("/nix/var/nix/profiles"))? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(number) = parse_generation_link_name(name)? else {
            continue;
        };

        let realization = read_realization(&entry.path())?;
        generations.push(NixGenerationObservation {
            number,
            current: realization == current_system_realization,
            realization,
        });
    }

    generations.sort_by_key(|entry| entry.number);
    if generations.is_empty() {
        return Err(SscObservationError::NoSystemGenerations);
    }
    Ok(generations)
}

fn parse_generation_link_name(name: &str) -> Result<Option<u64>, SscObservationError> {
    let Some(number) = name
        .strip_prefix("system-")
        .and_then(|value| value.strip_suffix("-link"))
    else {
        return Ok(None);
    };

    let number = number
        .parse::<u64>()
        .map_err(|error| SscObservationError::Io(error.to_string()))?;
    if number == 0 {
        return Err(SscObservationError::MissingSystemProfileGeneration);
    }
    Ok(Some(number))
}

fn generation_link(generation: u64) -> PathBuf {
    PathBuf::from(format!("/nix/var/nix/profiles/system-{generation}-link"))
}

fn read_profile_generation(link: &Path) -> Result<u64, SscObservationError> {
    let target = std::fs::read_link(link)?;
    let name = target
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            SscObservationError::Io("NixOS system profile link has no UTF-8 filename".into())
        })?;

    parse_generation_link_name(name)?.ok_or(SscObservationError::MissingSystemProfileGeneration)
}

fn validate_realization_path(value: &str) -> Result<(), SscObservationError> {
    if value.starts_with("/nix/store/") && value != "/nix/store/" {
        Ok(())
    } else {
        Err(SscObservationError::InvalidRealizationPath)
    }
}

fn read_realization(link: &Path) -> Result<String, SscObservationError> {
    let resolved = std::fs::canonicalize(link)?;
    let value = resolved.to_string_lossy().into_owned();
    if value.is_empty() {
        return Err(SscObservationError::MissingRealization);
    }
    validate_realization_path(&value)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_target_id() -> TargetId {
        target_identity_from_machine_id("0123456789abcdef0123456789abcdef").expect("target id")
    }

    #[test]
    fn target_identity_derivation_is_stable_and_domain_separated() {
        let one = target_identity_from_machine_id("0123456789abcdef0123456789abcdef")
            .expect("target id");
        let two = target_identity_from_machine_id("0123456789abcdef0123456789abcdef")
            .expect("target id");
        let changed =
            target_identity_from_machine_id("fedcba9876543210fedcba9876543210").expect("target id");

        assert_eq!(one, two);
        assert_ne!(one, changed);
        assert!(one.0.starts_with("nixos-machine:"));
    }

    #[test]
    fn target_identity_derivation_rejects_noncanonical_machine_id() {
        for machine_id in [
            "",
            "0123456789abcdef0123456789abcde",
            "0123456789abcdef0123456789abcdef0",
            "0123456789ABCDEF0123456789abcdef",
            "0123456789abcdef0123456789abcdeg",
        ] {
            assert_eq!(
                target_identity_from_machine_id(machine_id).expect_err("invalid machine id"),
                SscObservationError::InvalidMachineId
            );
        }
    }

    #[test]
    fn observation_validation_requires_target_identity() {
        let mut observation = NixSystemObservation {
            target_identity: test_target_id(),
            generations: vec![NixGenerationObservation {
                number: 42,
                realization: "/nix/store/aaa-nixos-system-host".into(),
                current: true,
            }],
            system_profile_generation: 42,
            system_profile_realization: "/nix/store/aaa-nixos-system-host".into(),
            current_system_realization: "/nix/store/aaa-nixos-system-host".into(),
            booted_system_realization: "/nix/store/aaa-nixos-system-host".into(),
        };
        observation.target_identity = TargetId::from("");

        assert_eq!(
            observation.validate().expect_err("missing target identity"),
            SscObservationError::MissingTargetIdentity
        );
    }

    #[test]
    fn generation_link_parser_is_exact() {
        assert_eq!(
            parse_generation_link_name("system-42-link")
                .expect("generation")
                .expect("link"),
            42
        );
        assert_eq!(
            parse_generation_link_name("system-0042-link")
                .expect("generation")
                .expect("link"),
            42
        );
        assert_eq!(
            parse_generation_link_name("system-current-link").expect_err("non-generation"),
            SscObservationError::Io("invalid digit found in string".into())
        );
        assert_eq!(
            parse_generation_link_name("system-42").expect("non-generation"),
            None
        );
    }

    #[test]
    fn profile_generation_parser_rejects_zero() {
        assert_eq!(
            parse_generation_link_name("system-0-link").expect_err("generation zero"),
            SscObservationError::MissingSystemProfileGeneration
        );
    }

    #[test]
    fn observation_validation_rejects_missing_generation_realization() {
        let observation = NixSystemObservation {
            target_identity: test_target_id(),
            generations: vec![NixGenerationObservation {
                number: 42,
                realization: String::new(),
                current: true,
            }],
            system_profile_generation: 42,
            system_profile_realization: "/nix/store/aaa-nixos-system-host".into(),
            current_system_realization: "/nix/store/aaa-nixos-system-host".into(),
            booted_system_realization: "/nix/store/aaa-nixos-system-host".into(),
        };

        assert_eq!(
            observation.validate().expect_err("missing realization"),
            SscObservationError::MissingRealization
        );
    }

    #[test]
    fn observation_validation_rejects_non_store_realization_fields() {
        let base = NixSystemObservation {
            target_identity: test_target_id(),
            generations: vec![NixGenerationObservation {
                number: 42,
                realization: "/nix/store/aaa-nixos-system-host".into(),
                current: true,
            }],
            system_profile_generation: 42,
            system_profile_realization: "/nix/store/aaa-nixos-system-host".into(),
            current_system_realization: "/nix/store/aaa-nixos-system-host".into(),
            booted_system_realization: "/nix/store/aaa-nixos-system-host".into(),
        };

        for (name, mutate) in [
            ("system_profile_realization", |observation: &mut NixSystemObservation| {
                observation.system_profile_realization = "/etc/nixos".into();
            }),
            ("current_system_realization", |observation: &mut NixSystemObservation| {
                observation.current_system_realization = "/etc/nixos".into();
            }),
            ("booted_system_realization", |observation: &mut NixSystemObservation| {
                observation.booted_system_realization = "/etc/nixos".into();
            }),
            ("generation_realization", |observation: &mut NixSystemObservation| {
                observation.generations[0].realization = "/etc/nixos".into();
            }),
        ] {
            let mut observation = base.clone();
            mutate(&mut observation);
            assert_eq!(
                observation.validate().expect_err(name),
                SscObservationError::InvalidRealizationPath,
                "{name} must remain a concrete Nix store realization"
            );
        }
    }

    #[test]
    fn observation_validation_rejects_noncanonical_generation_order() {
        let observation = NixSystemObservation {
            target_identity: test_target_id(),
            generations: vec![
                NixGenerationObservation {
                    number: 43,
                    realization: "/nix/store/aaa-nixos-system-host".into(),
                    current: true,
                },
                NixGenerationObservation {
                    number: 42,
                    realization: "/nix/store/bbb-nixos-system-host".into(),
                    current: false,
                },
            ],
            system_profile_generation: 43,
            system_profile_realization: "/nix/store/aaa-nixos-system-host".into(),
            current_system_realization: "/nix/store/aaa-nixos-system-host".into(),
            booted_system_realization: "/nix/store/aaa-nixos-system-host".into(),
        };

        assert_eq!(
            observation.validate().expect_err("noncanonical order"),
            SscObservationError::NonCanonicalGenerationOrder
        );
    }

    #[test]
    fn observation_validation_accepts_canonical_generation_order() {
        let observation = NixSystemObservation {
            target_identity: test_target_id(),
            generations: vec![
                NixGenerationObservation {
                    number: 42,
                    realization: "/nix/store/aaa-nixos-system-host".into(),
                    current: false,
                },
                NixGenerationObservation {
                    number: 43,
                    realization: "/nix/store/bbb-nixos-system-host".into(),
                    current: true,
                },
            ],
            system_profile_generation: 43,
            system_profile_realization: "/nix/store/bbb-nixos-system-host".into(),
            current_system_realization: "/nix/store/bbb-nixos-system-host".into(),
            booted_system_realization: "/nix/store/bbb-nixos-system-host".into(),
        };

        assert!(observation.validate().is_ok());
    }

    #[test]
    fn authorized_preflight_rejects_whole_observation_drift() {
        let observation = NixSystemObservation {
            target_identity: test_target_id(),
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
            system_profile_generation: 43,
            system_profile_realization: "/nix/store/bbb-nixos-system-host".into(),
            current_system_realization: "/nix/store/aaa-nixos-system-host".into(),
            booted_system_realization: "/nix/store/ccc-nixos-system-host".into(),
        };
        let authorized = rollback_authorized_plan_for_testing();

        assert_eq!(
            observation
                .validate_against_authorized_plan(&authorized)
                .expect_err("booted-state drift"),
            SscObservationError::AuthorizedObservationDigestMismatch
        );
    }

    #[test]
    fn authorized_preflight_accepts_exact_observed_generation() {
        let observation = NixSystemObservation {
            target_identity: test_target_id(),
            generations: vec![
                NixGenerationObservation {
                    number: 42,
                    realization: "/nix/store/aaa-nixos-system-host".into(),
                    current: true,
                },
            ],
            system_profile_generation: 42,
            system_profile_realization: "/nix/store/aaa-nixos-system-host".into(),
            current_system_realization: "/nix/store/aaa-nixos-system-host".into(),
            booted_system_realization: "/nix/store/aaa-nixos-system-host".into(),
        };
        let adapter = sovereign_state_compiler_nix::NixOSTargetAdapter::from_snapshot(
            observation.target_snapshot(100).expect("snapshot"),
        )
        .expect("adapter");
        let mut intent = sovereign_state_compiler::DeploymentIntent::new("rollback-1", test_target_id());
        intent.required_resources.insert(
            nixos_generation_resource(42, "/nix/store/aaa-nixos-system-host").expect("resource"),
        );
        intent.required_capabilities.insert(Capability::Rollback);
        let plan = adapter.compile(&intent).expect("plan");
        let auth = sovereign_state_compiler::AuthorizationEvidence {
            authority_id: "authority-1".into(),
            intent_digest: plan.intent.digest().expect("intent digest"),
            target_profile_digest: plan.target_snapshot.profile.digest().expect("profile digest"),
            target_snapshot_digest: plan.target_snapshot.digest().expect("snapshot digest"),
            plan_digest: plan.digest().expect("plan digest"),
            granted_capabilities: [Capability::Rollback, Capability::ObserveState]
                .into_iter()
                .collect(),
            nonce: "nonce-1".into(),
            valid_from_ms: Some(100),
            valid_until_ms: Some(200),
        };
        let authorized = plan.authorize(auth, 100).expect("authorization");

        assert!(observation
            .validate_against_authorized_plan(&authorized)
            .is_ok());
    }

    #[test]
    fn authorized_preflight_rejects_generation_realization_drift() {
        let observation = NixSystemObservation {
            target_identity: test_target_id(),
            generations: vec![NixGenerationObservation {
                number: 42,
                realization: "/nix/store/bbb-nixos-system-host".into(),
                current: true,
            }],
            system_profile_generation: 42,
            system_profile_realization: "/nix/store/bbb-nixos-system-host".into(),
            current_system_realization: "/nix/store/bbb-nixos-system-host".into(),
            booted_system_realization: "/nix/store/bbb-nixos-system-host".into(),
        };
        let authorized = rollback_authorized_plan_for_testing();

        assert_eq!(
            observation
                .validate_against_authorized_plan(&authorized)
                .expect_err("realization drift"),
            SscObservationError::AuthorizedResourceMissing(
                nixos_generation_resource(42, "/nix/store/aaa-nixos-system-host")
                    .expect("resource")
            )
        );
    }

    #[test]
    fn authorized_preflight_rejects_generation_ordinal_drift() {
        let observation = NixSystemObservation {
            target_identity: test_target_id(),
            generations: vec![NixGenerationObservation {
                number: 43,
                realization: "/nix/store/aaa-nixos-system-host".into(),
                current: true,
            }],
            system_profile_generation: 43,
            system_profile_realization: "/nix/store/aaa-nixos-system-host".into(),
            current_system_realization: "/nix/store/aaa-nixos-system-host".into(),
            booted_system_realization: "/nix/store/aaa-nixos-system-host".into(),
        };
        let authorized = rollback_authorized_plan_for_testing();

        assert_eq!(
            observation
                .validate_against_authorized_plan(&authorized)
                .expect_err("generation drift"),
            SscObservationError::AuthorizedResourceMissing(
                nixos_generation_resource(42, "/nix/store/aaa-nixos-system-host")
                    .expect("resource")
            )
        );
    }

    fn rollback_authorized_plan_for_testing() -> AuthorizedDeploymentPlan {
        let observation = NixSystemObservation {
            target_identity: test_target_id(),
            generations: vec![NixGenerationObservation {
                number: 42,
                realization: "/nix/store/aaa-nixos-system-host".into(),
                current: true,
            }],
            system_profile_generation: 42,
            system_profile_realization: "/nix/store/aaa-nixos-system-host".into(),
            current_system_realization: "/nix/store/aaa-nixos-system-host".into(),
            booted_system_realization: "/nix/store/aaa-nixos-system-host".into(),
        };
        let adapter = sovereign_state_compiler_nix::NixOSTargetAdapter::from_snapshot(
            observation.target_snapshot(100).expect("snapshot"),
        )
        .expect("adapter");
        let mut intent = sovereign_state_compiler::DeploymentIntent::new("rollback-test", test_target_id());
        intent.required_capabilities.insert(Capability::Rollback);
        intent.required_resources.insert(
            nixos_generation_resource(42, "/nix/store/aaa-nixos-system-host").expect("resource"),
        );
        intent.desired_state.properties.insert(
            "nixos.rollback".into(),
            sovereign_state_compiler::StateValue::Bool(true),
        );
        intent.desired_state.properties.insert(
            "nixos.rollback-generation".into(),
            sovereign_state_compiler::StateValue::Integer(42),
        );
        intent.desired_state.properties.insert(
            "nixos.rollback-realization".into(),
            sovereign_state_compiler::StateValue::String(
                "/nix/store/aaa-nixos-system-host".into(),
            ),
        );
        let plan = adapter.compile(&intent).expect("plan");
        let auth = sovereign_state_compiler::AuthorizationEvidence {
            authority_id: "authority-test".into(),
            intent_digest: plan.intent.digest().expect("intent digest"),
            target_profile_digest: plan.target_snapshot.profile.digest().expect("profile digest"),
            target_snapshot_digest: plan.target_snapshot.digest().expect("snapshot digest"),
            plan_digest: plan.digest().expect("plan digest"),
            granted_capabilities: [Capability::Rollback, Capability::ObserveState]
                .into_iter()
                .collect(),
            nonce: "nonce-test".into(),
            valid_from_ms: Some(100),
            valid_until_ms: Some(200),
        };
        plan.authorize(auth, 100).expect("authorization")
    }

    #[test]
    fn realization_path_validation_rejects_non_store_paths() {
        assert_eq!(
            validate_realization_path("/etc/nixos"),
            Err(SscObservationError::InvalidRealizationPath)
        );
        assert_eq!(
            validate_realization_path("/nix/store/example-system"),
            Ok(())
        );
    }

    #[test]
    fn observation_validation_rejects_zero_generation_number() {
        let observation = NixSystemObservation {
            target_identity: test_target_id(),
            generations: vec![NixGenerationObservation {
                number: 0,
                realization: "/nix/store/aaa-nixos-system-host".into(),
                current: true,
            }],
            system_profile_generation: 42,
            system_profile_realization: "/nix/store/aaa-nixos-system-host".into(),
            current_system_realization: "/nix/store/aaa-nixos-system-host".into(),
            booted_system_realization: "/nix/store/aaa-nixos-system-host".into(),
        };

        assert_eq!(
            observation.validate().expect_err("zero generation"),
            SscObservationError::InvalidGenerationNumber
        );
    }

    #[test]
    fn observation_validation_rejects_duplicate_generation_numbers() {
        let observation = NixSystemObservation {
            target_identity: test_target_id(),
            generations: vec![
                NixGenerationObservation {
                    number: 42,
                    realization: "/nix/store/aaa-nixos-system-host".into(),
                    current: true,
                },
                NixGenerationObservation {
                    number: 42,
                    realization: "/nix/store/aaa-nixos-system-host".into(),
                    current: false,
                },
            ],
            system_profile_generation: 42,
            system_profile_realization: "/nix/store/aaa-nixos-system-host".into(),
            current_system_realization: "/nix/store/aaa-nixos-system-host".into(),
            booted_system_realization: "/nix/store/aaa-nixos-system-host".into(),
        };

        assert_eq!(
            observation.validate().expect_err("duplicate generation"),
            SscObservationError::DuplicateGenerationNumber
        );
    }

    #[test]
    fn generation_link_is_deterministic() {
        assert_eq!(
            generation_link(42),
            PathBuf::from("/nix/var/nix/profiles/system-42-link")
        );
    }

    #[test]
    fn observation_digest_is_domain_separated() {
        let observation = NixSystemObservation {
            target_identity: test_target_id(),
            generations: vec![NixGenerationObservation {
                number: 42,
                realization: "/nix/store/aaa-nixos-system-host".into(),
                current: true,
            }],
            system_profile_generation: 42,
            system_profile_realization: "/nix/store/aaa-nixos-system-host".into(),
            current_system_realization: "/nix/store/aaa-nixos-system-host".into(),
            booted_system_realization: "/nix/store/aaa-nixos-system-host".into(),
        };
        let digest = observation.observation_digest().expect("digest");
        let bytes = serde_json::to_vec(&observation).expect("serialize");
        assert_ne!(digest, ContentDigest::blake3(&bytes));
        assert_eq!(super::NIXOS_OBSERVATION_DIGEST_DOMAIN.len(), 43);
    }

    #[test]
    fn observation_digest_changes_when_realization_changes() {
        let mut observation = NixSystemObservation {
            target_identity: test_target_id(),
            generations: vec![NixGenerationObservation {
                number: 42,
                realization: "/nix/store/aaa-nixos-system-host".into(),
                current: true,
            }],
            system_profile_generation: 42,
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
    fn observation_validation_requires_current_realization_match() {
        let mut observation = NixSystemObservation {
            target_identity: test_target_id(),
            generations: vec![NixGenerationObservation {
                number: 42,
                realization: "/nix/store/aaa-nixos-system-host".into(),
                current: true,
            }],
            system_profile_generation: 42,
            system_profile_realization: "/nix/store/aaa-nixos-system-host".into(),
            current_system_realization: "/nix/store/aaa-nixos-system-host".into(),
            booted_system_realization: "/nix/store/aaa-nixos-system-host".into(),
        };

        assert!(observation.validate().is_ok());

        observation.current_system_realization = "/nix/store/bbb-nixos-system-host".into();
        assert_eq!(
            observation.validate().expect_err("mismatch"),
            SscObservationError::CurrentGenerationRealizationMismatch
        );
    }

    #[test]
    fn observation_validation_rejects_multiple_current_generations() {
        let observation = NixSystemObservation {
            target_identity: test_target_id(),
            generations: vec![
                NixGenerationObservation {
                    number: 42,
                    realization: "/nix/store/aaa-nixos-system-host".into(),
                    current: true,
                },
                NixGenerationObservation {
                    number: 43,
                    realization: "/nix/store/bbb-nixos-system-host".into(),
                    current: true,
                },
            ],
            system_profile_generation: 43,
            system_profile_realization: "/nix/store/bbb-nixos-system-host".into(),
            current_system_realization: "/nix/store/aaa-nixos-system-host".into(),
            booted_system_realization: "/nix/store/aaa-nixos-system-host".into(),
        };

        assert_eq!(
            observation.validate().expect_err("multiple current"),
            SscObservationError::MultipleCurrentGenerations
        );
    }

    #[test]
    fn observation_validation_requires_profile_realization_match() {
        let observation = NixSystemObservation {
            target_identity: test_target_id(),
            generations: vec![NixGenerationObservation {
                number: 42,
                realization: "/nix/store/aaa-nixos-system-host".into(),
                current: true,
            }],
            system_profile_generation: 42,
            system_profile_realization: "/nix/store/bbb-nixos-system-host".into(),
            current_system_realization: "/nix/store/aaa-nixos-system-host".into(),
            booted_system_realization: "/nix/store/aaa-nixos-system-host".into(),
        };

        assert_eq!(
            observation.validate().expect_err("profile mismatch"),
            SscObservationError::SystemProfileRealizationMismatch
        );
    }

    #[test]
    fn target_adapter_preserves_exact_observed_snapshot() {
        let observation = NixSystemObservation {
            target_identity: test_target_id(),
            generations: vec![NixGenerationObservation {
                number: 42,
                realization: "/nix/store/aaa-nixos-system-host".into(),
                current: true,
            }],
            system_profile_generation: 42,
            system_profile_realization: "/nix/store/aaa-nixos-system-host".into(),
            current_system_realization: "/nix/store/aaa-nixos-system-host".into(),
            booted_system_realization: "/nix/store/aaa-nixos-system-host".into(),
        };
        let adapter = observation.target_adapter(123).expect("adapter");
        let snapshot = adapter.describe_target().expect("snapshot");

        assert_eq!(snapshot.profile.platform, "nixos");
        assert_eq!(snapshot.profile.identity, test_target_id());
        assert_eq!(snapshot.observed_at_ms, 123);
        assert_eq!(snapshot.observation_digest, observation.observation_digest().expect("digest"));
        assert!(snapshot.resources.contains(
            &nixos_generation_resource(42, "/nix/store/aaa-nixos-system-host")
                .expect("generation resource")
        ));
    }

    #[test]
    fn target_snapshot_contains_exact_generation_resources() {
        let observation = NixSystemObservation {
            target_identity: test_target_id(),
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
            system_profile_generation: 43,
            system_profile_realization: "/nix/store/bbb-nixos-system-host".into(),
            current_system_realization: "/nix/store/aaa-nixos-system-host".into(),
            booted_system_realization: "/nix/store/aaa-nixos-system-host".into(),
        };

        let snapshot = observation
            .target_snapshot(123)
            .expect("snapshot");
        assert_eq!(snapshot.profile.identity, test_target_id());
        assert_eq!(snapshot.profile.platform, "nixos");
        assert!(
            snapshot
                .profile
                .capabilities
                .contains(&sovereign_state_compiler::Capability::ObserveState)
        );
        assert_eq!(snapshot.observed_at_ms, 123);
        assert_eq!(snapshot.resources.len(), 2);
        assert!(
            snapshot.resources.contains(
                &nixos_generation_resource(42, "/nix/store/aaa-nixos-system-host")
                    .expect("generation resource")
            )
        );
        assert!(
            snapshot.resources.contains(
                &nixos_generation_resource(43, "/nix/store/bbb-nixos-system-host")
                    .expect("generation resource")
            )
        );
    }

    #[test]
    fn generation_resource_tracks_exact_observed_realization() {
        let observation = NixSystemObservation {
            target_identity: test_target_id(),
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
            system_profile_generation: 43,
            system_profile_realization: "/nix/store/bbb-nixos-system-host".into(),
            current_system_realization: "/nix/store/aaa-nixos-system-host".into(),
            booted_system_realization: "/nix/store/aaa-nixos-system-host".into(),
        };

        assert_eq!(
            observation.generation_resource(42).expect("resource"),
            nixos_generation_resource(42, "/nix/store/aaa-nixos-system-host").expect("resource")
        );
        assert_eq!(
            observation
                .system_profile_generation()
                .expect("profile generation")
                .number,
            43
        );
        assert_eq!(
            observation
                .current_generation()
                .expect("current generation")
                .number,
            42
        );
        assert_ne!(
            observation.current_system_realization,
            observation.system_profile_realization
        );
        assert_eq!(super::NIXOS_GENERATION_RESOURCE_KIND, "nixos-generation");
    }
}
