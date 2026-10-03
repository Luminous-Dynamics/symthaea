//! NixOS target adapter for the Sovereign State Compiler.
//!
//! This crate owns NixOS-specific interpretation of the neutral deployment
//! contract. It deliberately stops at an abstract lifecycle plan: native
//! commands, transport, and privileged execution remain outside the adapter.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use sovereign_state_compiler::{
    Capability, DeploymentIntent, DeploymentPlan, PlanStep, PlanStepKind, PlanValidationError,
    RollbackPolicy, TargetAdapter, TargetId, TargetSnapshot, VerificationPolicy,
};
use thiserror::Error;

pub const NIXOS_PLATFORM: &str = "nixos";
pub const MAX_TARGET_SNAPSHOT_AGE_MS: u64 = 300_000;

const REBUILD_KEY: &str = "nixos.rebuild";
const HOME_MANAGER_KEY: &str = "nixos.home-manager";
const INSTALL_KEY: &str = "applications.install";
const REMOVE_KEY: &str = "applications.remove";
const REBOOT_KEY: &str = "system.reboot";
const ROLLBACK_KEY: &str = "nixos.rollback";

/// A conservative NixOS capability profile.
///
/// This is a protocol-facing description only. It does not grant authority;
/// authorization still occurs against the exact compiled plan in the neutral
/// compiler crate.
pub fn default_nixos_capabilities() -> BTreeSet<Capability> {
    [
        Capability::ObserveHardware,
        Capability::InstallApplication,
        Capability::RemoveApplication,
        Capability::ConfigureSystem,
        Capability::UpdateSystem,
        Capability::Rollback,
        Capability::Reboot,
        Capability::ModifyBootChain,
        Capability::CreateRecoveryEnvironment,
    ]
    .into_iter()
    .collect()
}

/// A target adapter that lowers a narrow, explicit NixOS state vocabulary into
/// target-neutral lifecycle steps.
///
/// Supported state properties:
///
/// * `nixos.rebuild`: "switch", "test", or "boot"
/// * `nixos.home-manager`: boolean
/// * `applications.install`: list of logical artifact/package identifiers
/// * `applications.remove`: list of logical package identifiers
/// * `system.reboot`: boolean
/// * `nixos.rollback`: boolean
///
/// Unknown properties are rejected rather than silently ignored.
#[derive(Debug, Clone)]
pub struct NixOSTargetAdapter {
    snapshot: TargetSnapshot,
}

impl NixOSTargetAdapter {
    pub fn new(target: impl Into<TargetId>, observed_at_ms: u64) -> Self {
        Self {
            snapshot: TargetSnapshot {
                profile: sovereign_state_compiler::TargetProfile {
                    identity: target.into(),
                    platform: NIXOS_PLATFORM.into(),
                    capabilities: default_nixos_capabilities(),
                },
                observed_at_ms,
                observation_digest: sovereign_state_compiler::ContentDigest::blake3(
                    b"nixos-target-observation-v0.1",
                ),
                resources: BTreeSet::new(),
            },
        }
    }

    pub fn from_snapshot(snapshot: TargetSnapshot) -> Result<Self, NixOSAdapterError> {
        if snapshot.profile.platform != NIXOS_PLATFORM {
            return Err(NixOSAdapterError::WrongPlatform(
                snapshot.profile.platform.clone(),
            ));
        }
        Ok(Self { snapshot })
    }

    fn property_bool(
        intent: &DeploymentIntent,
        key: &'static str,
    ) -> Result<Option<bool>, NixOSAdapterError> {
        match intent.desired_state.properties.get(key) {
            None => Ok(None),
            Some(sovereign_state_compiler::StateValue::Bool(value)) => Ok(Some(*value)),
            Some(_) => Err(NixOSAdapterError::PropertyType {
                key,
                expected: "boolean",
            }),
        }
    }

    fn property_string(
        intent: &DeploymentIntent,
        key: &'static str,
    ) -> Result<Option<String>, NixOSAdapterError> {
        match intent.desired_state.properties.get(key) {
            None => Ok(None),
            Some(sovereign_state_compiler::StateValue::String(value)) => Ok(Some(value.clone())),
            Some(_) => Err(NixOSAdapterError::PropertyType {
                key,
                expected: "string",
            }),
        }
    }

    fn property_string_list(
        intent: &DeploymentIntent,
        key: &'static str,
    ) -> Result<Vec<String>, NixOSAdapterError> {
        let Some(value) = intent.desired_state.properties.get(key) else {
            return Ok(Vec::new());
        };

        let sovereign_state_compiler::StateValue::List(values) = value else {
            return Err(NixOSAdapterError::PropertyType {
                key,
                expected: "list of strings",
            });
        };

        values
            .iter()
            .map(|value| match value {
                sovereign_state_compiler::StateValue::String(item) if !item.is_empty() => {
                    Ok(item.clone())
                }
                sovereign_state_compiler::StateValue::String(_) => {
                    Err(NixOSAdapterError::EmptyListItem { key })
                }
                _ => Err(NixOSAdapterError::PropertyType {
                    key,
                    expected: "list of strings",
                }),
            })
            .collect()
    }

    fn reject_unknown_properties(
        intent: &DeploymentIntent,
    ) -> Result<(), NixOSAdapterError> {
        const SUPPORTED: [&str; 6] = [
            REBUILD_KEY,
            HOME_MANAGER_KEY,
            INSTALL_KEY,
            REMOVE_KEY,
            REBOOT_KEY,
            ROLLBACK_KEY,
        ];

        if let Some(key) = intent
            .desired_state
            .properties
            .keys()
            .find(|key| !SUPPORTED.contains(&key.as_str()))
        {
            return Err(NixOSAdapterError::UnsupportedProperty(key.clone()));
        }

        Ok(())
    }

    fn push_step(
        steps: &mut Vec<PlanStep>,
        kind: PlanStepKind,
        capabilities: impl IntoIterator<Item = Capability>,
        description: impl Into<String>,
    ) {
        let sequence = steps.len() as u32;
        steps.push(PlanStep {
            sequence,
            kind,
            required_capabilities: capabilities.into_iter().collect(),
            description: description.into(),
        });
    }
}

impl TargetAdapter for NixOSTargetAdapter {
    type Error = NixOSAdapterError;

    fn describe_target(&self) -> Result<TargetSnapshot, Self::Error> {
        Ok(self.snapshot.clone())
    }

    fn compile(&self, intent: &DeploymentIntent) -> Result<DeploymentPlan, Self::Error> {
        Self::reject_unknown_properties(intent)?;

        if intent.target != self.snapshot.profile.identity {
            return Err(NixOSAdapterError::TargetMismatch);
        }

        let rebuild = Self::property_string(intent, REBUILD_KEY)?;
        let home_manager = Self::property_bool(intent, HOME_MANAGER_KEY)?.unwrap_or(false);
        let install = Self::property_string_list(intent, INSTALL_KEY)?;
        let remove = Self::property_string_list(intent, REMOVE_KEY)?;
        let reboot = Self::property_bool(intent, REBOOT_KEY)?.unwrap_or(false);
        let rollback_requested = Self::property_bool(intent, ROLLBACK_KEY)?.unwrap_or(false);

        let mut steps = Vec::new();

        Self::push_step(
            &mut steps,
            PlanStepKind::Observe,
            [Capability::ObserveHardware],
            "observe target before applying state",
        );

        if let Some(mode) = rebuild.as_deref() {
            let required = match mode {
                "switch" => [
                    Capability::ConfigureSystem,
                    Capability::UpdateSystem,
                    Capability::ModifyBootChain,
                ],
                "test" => [Capability::ConfigureSystem, Capability::UpdateSystem, Capability::ObserveHardware],
                "boot" => [
                    Capability::ConfigureSystem,
                    Capability::UpdateSystem,
                    Capability::ModifyBootChain,
                ],
                other => return Err(NixOSAdapterError::InvalidRebuildMode(other.into())),
            };

            if !intent.artifacts.is_empty() {
                Self::push_step(
                    &mut steps,
                    PlanStepKind::StageArtifacts,
                    [Capability::UpdateSystem],
                    format!("stage {} NixOS deployment artifact(s)", intent.artifacts.len()),
                );
            }

            Self::push_step(
                &mut steps,
                PlanStepKind::ApplyDesiredState,
                required,
                format!("apply NixOS rebuild mode: {mode}"),
            );
        }

        if !install.is_empty() {
            Self::push_step(
                &mut steps,
                PlanStepKind::StageArtifacts,
                [Capability::InstallApplication],
                format!("stage {} application artifact(s)", install.len()),
            );
            Self::push_step(
                &mut steps,
                PlanStepKind::ApplyDesiredState,
                [Capability::InstallApplication],
                format!("install applications: {}", install.join(", ")),
            );
        }

        if !remove.is_empty() {
            Self::push_step(
                &mut steps,
                PlanStepKind::ApplyDesiredState,
                [Capability::RemoveApplication],
                format!("remove applications: {}", remove.join(", ")),
            );
        }

        if home_manager {
            Self::push_step(
                &mut steps,
                PlanStepKind::ApplyDesiredState,
                [Capability::ConfigureSystem],
                "apply Home Manager user state",
            );
        }

        if rollback_requested {
            Self::push_step(
                &mut steps,
                PlanStepKind::Rollback,
                [Capability::Rollback],
                "roll back to the previous NixOS generation if requested",
            );
        }

        if reboot {
            Self::push_step(
                &mut steps,
                PlanStepKind::Reboot,
                [Capability::Reboot],
                "reboot target after state application",
            );
        }

        Self::push_step(
            &mut steps,
            PlanStepKind::Verify,
            [Capability::ObserveHardware],
            "verify declared target state after execution",
        );

        let verification = VerificationPolicy {
            required_properties: intent.desired_state.properties.keys().cloned().collect(),
            require_attestation: false,
        };

        let rollback_allowed = self
            .snapshot
            .profile
            .capabilities
            .contains(&Capability::Rollback);

        let plan = DeploymentPlan {
            schema_version: sovereign_state_compiler::SCHEMA_VERSION.into(),
            intent: intent.clone(),
            target_snapshot: self.snapshot.clone(),
            max_target_snapshot_age_ms: Some(MAX_TARGET_SNAPSHOT_AGE_MS),
            steps,
            verification,
            rollback: RollbackPolicy {
                allowed: rollback_allowed,
                max_attempts: u8::from(rollback_allowed && !rollback_requested),
            },
        };

        plan.validate().map_err(NixOSAdapterError::PlanValidation)?;
        Ok(plan)
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum NixOSAdapterError {
    #[error("deployment target is not the adapter target")]
    TargetMismatch,
    #[error("target platform is not NixOS: {0}")]
    WrongPlatform(String),
    #[error("unsupported NixOS state property: {0}")]
    UnsupportedProperty(String),
    #[error("state property {key} must be a {expected}")]
    PropertyType {
        key: &'static str,
        expected: &'static str,
    },
    #[error("state property {key} contains an empty list item")]
    EmptyListItem { key: &'static str },
    #[error("unsupported nixos.rebuild mode: {0}")]
    InvalidRebuildMode(String),
    #[error("compiled plan violates neutral compiler invariants: {0}")]
    PlanValidation(PlanValidationError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use sovereign_state_compiler::{
        ArtifactId, ArtifactRef, ContentDigest, StateValue,
    };

    fn adapter() -> NixOSTargetAdapter {
        NixOSTargetAdapter::new("host-01", 1_000)
    }

    #[test]
    fn compiles_a_nixos_switch_without_native_commands() {
        let adapter = adapter();
        let mut intent = DeploymentIntent::new("switch-1", "host-01");
        intent.desired_state.properties.insert(
            REBUILD_KEY.into(),
            StateValue::String("switch".into()),
        );
        intent.artifacts.push(ArtifactRef {
            id: ArtifactId::from("system-config"),
            version: Some("2026.10.03".into()),
            digest: ContentDigest::blake3(b"system-config"),
            provenance: Vec::new(),
        });

        let plan = adapter.compile(&intent).expect("compile");
        assert_eq!(plan.steps.len(), 4);
        assert_eq!(plan.steps[1].kind, PlanStepKind::StageArtifacts);
        assert_eq!(plan.steps[2].kind, PlanStepKind::ApplyDesiredState);
        assert_eq!(
            plan.steps[2].required_capabilities,
            [
                Capability::ConfigureSystem,
                Capability::ModifyBootChain,
                Capability::UpdateSystem
            ]
            .into_iter()
            .collect()
        );
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn compiles_unrelated_application_install_with_same_neutral_contract() {
        let adapter = adapter();
        let mut intent = DeploymentIntent::new("app-1", "host-01");
        intent.desired_state.properties.insert(
            INSTALL_KEY.into(),
            StateValue::List(vec![
                StateValue::String("firefox".into()),
                StateValue::String("ripgrep".into()),
            ]),
        );

        let plan = adapter.compile(&intent).expect("compile");
        assert_eq!(plan.steps[1].kind, PlanStepKind::StageArtifacts);
        assert_eq!(plan.steps[2].kind, PlanStepKind::ApplyDesiredState);
        assert!(plan.steps[2]
            .description
            .contains("firefox, ripgrep"));
    }

    #[test]
    fn rejects_unknown_state_instead_of_dropping_it() {
        let adapter = adapter();
        let mut intent = DeploymentIntent::new("bad-1", "host-01");
        intent
            .desired_state
            .properties
            .insert("arbitrary.command".into(), StateValue::String("rm".into()));

        assert_eq!(
            adapter.compile(&intent),
            Err(NixOSAdapterError::UnsupportedProperty(
                "arbitrary.command".into()
            ))
        );
    }

    #[test]
    fn rejects_invalid_rebuild_mode() {
        let adapter = adapter();
        let mut intent = DeploymentIntent::new("bad-2", "host-01");
        intent.desired_state.properties.insert(
            REBUILD_KEY.into(),
            StateValue::String("execute-shell".into()),
        );

        assert_eq!(
            adapter.compile(&intent),
            Err(NixOSAdapterError::InvalidRebuildMode(
                "execute-shell".into()
            ))
        );
    }

    #[test]
    fn target_snapshot_is_reused_without_replacing_identity() {
        let adapter = adapter();
        let snapshot = adapter.describe_target().expect("snapshot");
        assert_eq!(snapshot.profile.platform, NIXOS_PLATFORM);
        assert_eq!(snapshot.profile.identity, TargetId::from("host-01"));
    }

    #[test]
    fn plan_digest_is_deterministic() {
        let adapter = adapter();
        let mut intent = DeploymentIntent::new("digest-1", "host-01");
        intent.desired_state.properties.insert(
            REBOOT_KEY.into(),
            StateValue::Bool(true),
        );

        let a = adapter.compile(&intent).expect("compile");
        let b = adapter.compile(&intent).expect("compile");
        assert_eq!(a, b);
        assert_eq!(a.digest().expect("digest"), b.digest().expect("digest"));
    }
}
