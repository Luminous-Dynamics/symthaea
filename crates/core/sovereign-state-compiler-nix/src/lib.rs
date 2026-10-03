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
const ROLLBACK_GENERATION_KEY: &str = "nixos.rollback-generation";
pub const NIXOS_GENERATION_RESOURCE_KIND: &str = "nixos-generation";

/// Construct the resource identity used to bind a rollback to a specific
/// observed NixOS generation.
pub fn nixos_generation_resource(generation: u64) -> sovereign_state_compiler::ResourceRef {
    sovereign_state_compiler::ResourceRef {
        kind: NIXOS_GENERATION_RESOURCE_KIND.into(),
        identity: sovereign_state_compiler::ContentDigest::blake3(
            generation.to_string().as_bytes(),
        ),
    }
}

/// Concrete NixOS activation semantics carried by the target adapter.
///
/// This type is deliberately NixOS-specific and never crosses into the
/// platform-neutral SSC crate. Its discriminant is therefore part of the
/// adapter's lowering decision while the resulting generic plan digest still
/// binds the original desired state exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NixActivationMode {
    Switch,
    Test,
    Boot,
    DryActivate,
    Rollback { generation: u64 },
}

impl NixActivationMode {
    pub fn parse_rebuild(value: &str) -> Result<Self, NixOSAdapterError> {
        match value {
            "switch" => Ok(Self::Switch),
            "test" => Ok(Self::Test),
            "boot" => Ok(Self::Boot),
            "dry-activate" => Ok(Self::DryActivate),
            other => Err(NixOSAdapterError::InvalidRebuildMode(other.into())),
        }
    }
}

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
/// * `nixos.rebuild`: "switch", "test", "boot", or "dry-activate"
/// * `nixos.home-manager`: boolean
/// * `applications.install`: list of logical artifact/package identifiers
/// * `applications.remove`: list of logical package identifiers
/// * `system.reboot`: boolean
/// * `nixos.rollback`: boolean; `true` requires `nixos.rollback-generation`
/// * `nixos.rollback-generation`: positive generation number, explicitly bound in `required_resources`
///
/// Unknown properties are rejected rather than silently ignored.
#[derive(Debug, Clone)]
pub struct NixOSTargetAdapter {
    snapshot: TargetSnapshot,
}

impl NixOSTargetAdapter {
    /// Synthetic constructor for deterministic adapter tests and local contract
    /// development. Production callers should prefer `from_observation` so
    /// the capability/resource surface and observation digest come from the
    /// actual target.
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

    pub fn from_observation(
        target: impl Into<TargetId>,
        observed_at_ms: u64,
        observation_digest: sovereign_state_compiler::ContentDigest,
        capabilities: BTreeSet<Capability>,
        resources: BTreeSet<sovereign_state_compiler::ResourceRef>,
    ) -> Self {
        Self {
            snapshot: TargetSnapshot {
                profile: sovereign_state_compiler::TargetProfile {
                    identity: target.into(),
                    platform: NIXOS_PLATFORM.into(),
                    capabilities,
                },
                observed_at_ms,
                observation_digest,
                resources,
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

    fn property_u64(
        intent: &DeploymentIntent,
        key: &'static str,
    ) -> Result<Option<u64>, NixOSAdapterError> {
        match intent.desired_state.properties.get(key) {
            None => Ok(None),
            Some(sovereign_state_compiler::StateValue::Integer(value)) if *value > 0 => {
                Ok(Some(*value as u64))
            }
            Some(sovereign_state_compiler::StateValue::Integer(_)) => {
                Err(NixOSAdapterError::InvalidRollbackGeneration)
            }
            Some(_) => Err(NixOSAdapterError::PropertyType {
                key,
                expected: "non-negative integer",
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
        const SUPPORTED: [&str; 7] = [
            REBUILD_KEY,
            HOME_MANAGER_KEY,
            INSTALL_KEY,
            REMOVE_KEY,
            REBOOT_KEY,
            ROLLBACK_KEY,
            ROLLBACK_GENERATION_KEY,
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
        let rollback_generation = Self::property_u64(intent, ROLLBACK_GENERATION_KEY)?;

        if rollback_requested != rollback_generation.is_some() {
            return Err(NixOSAdapterError::RollbackGenerationRequired);
        }

        let activation_mode = match (rebuild.as_deref(), rollback_generation) {
            (Some(_), Some(_)) => return Err(NixOSAdapterError::ConflictingActivationModes),
            (Some(mode), None) => Some(NixActivationMode::parse_rebuild(mode)?),
            (None, Some(generation)) => Some(NixActivationMode::Rollback { generation }),
            (None, None) => None,
        };

        if let Some(NixActivationMode::Rollback { generation }) = activation_mode {
            let required_resource = nixos_generation_resource(generation);
            if !intent.required_resources.contains(&required_resource) {
                return Err(NixOSAdapterError::UnboundRollbackGeneration(generation));
            }
        }

        if matches!(activation_mode, Some(NixActivationMode::Rollback { .. }))
            && (!install.is_empty() || !remove.is_empty() || home_manager)
        {
            return Err(NixOSAdapterError::ConflictingActivationModes);
        }

        let mut steps = Vec::new();

        Self::push_step(
            &mut steps,
            PlanStepKind::Observe,
            [Capability::ObserveHardware],
            "observe target before applying state",
        );

        match activation_mode {
            Some(NixActivationMode::Switch)
            | Some(NixActivationMode::Test)
            | Some(NixActivationMode::Boot)
            | Some(NixActivationMode::DryActivate) => {
                let (required, description) = match activation_mode {
                    Some(NixActivationMode::Switch) => (
                        [
                            Capability::ConfigureSystem,
                            Capability::UpdateSystem,
                            Capability::ModifyBootChain,
                        ]
                        .into_iter()
                        .collect::<BTreeSet<_>>(),
                        "activate the NixOS configuration and make its generation current",
                    ),
                    Some(NixActivationMode::Test) => (
                        [
                            Capability::ConfigureSystem,
                            Capability::UpdateSystem,
                            Capability::ObserveHardware,
                        ]
                        .into_iter()
                        .collect::<BTreeSet<_>>(),
                        "temporarily activate the NixOS configuration without changing the boot default",
                    ),
                    Some(NixActivationMode::Boot) => (
                        [
                            Capability::ConfigureSystem,
                            Capability::UpdateSystem,
                            Capability::ModifyBootChain,
                        ]
                        .into_iter()
                        .collect::<BTreeSet<_>>(),
                        "build the NixOS configuration and select it for the next boot without activating now",
                    ),
                    Some(NixActivationMode::DryActivate) => (
                        [Capability::ConfigureSystem, Capability::UpdateSystem]
                            .into_iter()
                            .collect::<BTreeSet<_>>(),
                        "evaluate NixOS activation changes without activating the configuration",
                    ),
                    Some(NixActivationMode::Rollback { .. }) | None => {
                        unreachable!("matched non-rollback activation mode")
                    }
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
                    description,
                );
            }
            Some(NixActivationMode::Rollback { generation }) => {
                Self::push_step(
                    &mut steps,
                    PlanStepKind::Rollback,
                    [Capability::Rollback],
                    format!("roll back to NixOS generation {generation}"),
                );
            }
            None => {}
        }

        if !install.is_empty() {
            Self::push_step(
                &mut steps,
                PlanStepKind::StageArtifacts,
                [Capability::InstallApplication],
                format!("stage {} application package input(s)", install.len()),
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
            expected_state: intent.desired_state.clone(),
            require_attestation: false,
        };

        let has_mutation = steps.iter().any(|step| {
            match step.kind {
                PlanStepKind::StageArtifacts => {
                    !matches!(activation_mode, Some(NixActivationMode::DryActivate))
                }
                PlanStepKind::ApplyDesiredState => {
                    !matches!(activation_mode, Some(NixActivationMode::DryActivate))
                        || !install.is_empty()
                        || !remove.is_empty()
                        || home_manager
                }
                PlanStepKind::Reboot | PlanStepKind::Rollback => true,
                _ => false,
            }
        });

        let rollback_allowed = has_mutation
            && self
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
    #[error("nixos.rollback=true requires a positive nixos.rollback-generation")]
    RollbackGenerationRequired,
    #[error("nixos.rollback-generation must be greater than zero")]
    InvalidRollbackGeneration,
    #[error("rollback generation {0} is not bound to an observed NixOS generation resource")]
    UnboundRollbackGeneration(u64),
    #[error("NixOS activation modes cannot be combined in one intent")]
    ConflictingActivationModes,
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

    fn rollback_adapter(generation: u64) -> NixOSTargetAdapter {
        let mut snapshot = adapter().describe_target().expect("snapshot");
        snapshot.resources.insert(nixos_generation_resource(generation));
        NixOSTargetAdapter::from_snapshot(snapshot).expect("nixos snapshot")
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
        assert_eq!(
            plan.verification.expected_state,
            intent.desired_state
        );
    }

    #[test]
    fn deterministic_nixos_rebuild_plan_vectors() {
        let vectors = [
            (
                "switch",
                vec![
                    (PlanStepKind::Observe, vec![Capability::ObserveHardware]),
                    (
                        PlanStepKind::ApplyDesiredState,
                        vec![
                            Capability::ConfigureSystem,
                            Capability::UpdateSystem,
                            Capability::ModifyBootChain,
                        ],
                    ),
                    (PlanStepKind::Verify, vec![Capability::ObserveHardware]),
                ],
            ),
            (
                "test",
                vec![
                    (PlanStepKind::Observe, vec![Capability::ObserveHardware]),
                    (
                        PlanStepKind::ApplyDesiredState,
                        vec![
                            Capability::ObserveHardware,
                            Capability::ConfigureSystem,
                            Capability::UpdateSystem,
                        ],
                    ),
                    (PlanStepKind::Verify, vec![Capability::ObserveHardware]),
                ],
            ),
            (
                "boot",
                vec![
                    (PlanStepKind::Observe, vec![Capability::ObserveHardware]),
                    (
                        PlanStepKind::ApplyDesiredState,
                        vec![
                            Capability::ConfigureSystem,
                            Capability::UpdateSystem,
                            Capability::ModifyBootChain,
                        ],
                    ),
                    (PlanStepKind::Verify, vec![Capability::ObserveHardware]),
                ],
            ),
            (
                "dry-activate",
                vec![
                    (PlanStepKind::Observe, vec![Capability::ObserveHardware]),
                    (
                        PlanStepKind::ApplyDesiredState,
                        vec![Capability::ConfigureSystem, Capability::UpdateSystem],
                    ),
                    (PlanStepKind::Verify, vec![Capability::ObserveHardware]),
                ],
            ),
        ];

        for (mode, expected) in vectors {
            let mut intent = DeploymentIntent::new("vector-1", "host-01");
            intent.desired_state.properties.insert(
                REBUILD_KEY.into(),
                StateValue::String(mode.into()),
            );

            let plan = adapter().compile(&intent).expect("compile");
            let actual = plan
                .steps
                .iter()
                .map(|step| {
                    (
                        step.kind.clone(),
                        step.required_capabilities
                            .iter()
                            .copied()
                            .collect::<BTreeSet<_>>(),
                    )
                })
                .collect::<Vec<_>>();

            assert_eq!(actual, expected, "plan vector mismatch for {mode}");
        }
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
    fn rollback_requires_explicit_generation_identity() {
        let mut intent = DeploymentIntent::new("rollback-1", "host-01");
        intent
            .desired_state
            .properties
            .insert(ROLLBACK_KEY.into(), StateValue::Bool(true));

        assert_eq!(
            adapter().compile(&intent),
            Err(NixOSAdapterError::RollbackGenerationRequired)
        );
    }

    #[test]
    fn rollback_generation_resource_identity_is_deterministic() {
        assert_eq!(
            nixos_generation_resource(42),
            nixos_generation_resource(42)
        );
        assert_ne!(
            nixos_generation_resource(42),
            nixos_generation_resource(43)
        );
    }

    #[test]
    fn rollback_requires_observed_generation_binding() {
        let mut intent = DeploymentIntent::new("rollback-bound-1", "host-01");
        intent
            .desired_state
            .properties
            .insert(ROLLBACK_KEY.into(), StateValue::Bool(true));
        intent.desired_state.properties.insert(
            ROLLBACK_GENERATION_KEY.into(),
            StateValue::Integer(42),
        );

        assert_eq!(
            adapter().compile(&intent),
            Err(NixOSAdapterError::UnboundRollbackGeneration(42))
        );
    }

    #[test]
    fn rollback_rejects_generation_zero() {
        let mut intent = DeploymentIntent::new("rollback-zero", "host-01");
        intent
            .desired_state
            .properties
            .insert(ROLLBACK_KEY.into(), StateValue::Bool(true));
        intent.desired_state.properties.insert(
            ROLLBACK_GENERATION_KEY.into(),
            StateValue::Integer(0),
        );

        assert_eq!(
            adapter().compile(&intent),
            Err(NixOSAdapterError::InvalidRollbackGeneration)
        );
    }

    #[test]
    fn rollback_is_a_typed_activation_mode() {
        let mut intent = DeploymentIntent::new("rollback-mode-1", "host-01");
        intent
            .desired_state
            .properties
            .insert(ROLLBACK_KEY.into(), StateValue::Bool(true));
        intent.desired_state.properties.insert(
            ROLLBACK_GENERATION_KEY.into(),
            StateValue::Integer(7),
        );
        intent.required_resources.insert(nixos_generation_resource(7));

        let mode = NixActivationMode::Rollback { generation: 7 };
        assert_eq!(mode, NixActivationMode::Rollback { generation: 7 });
        let plan = rollback_adapter(7).compile(&intent).expect("compile");
        assert_eq!(plan.steps[1].kind, PlanStepKind::Rollback);
    }

    #[test]
    fn rollback_cannot_be_combined_with_rebuild() {
        let mut intent = DeploymentIntent::new("rollback-conflict-1", "host-01");
        intent.desired_state.properties.insert(
            REBUILD_KEY.into(),
            StateValue::String("switch".into()),
        );
        intent
            .desired_state
            .properties
            .insert(ROLLBACK_KEY.into(), StateValue::Bool(true));
        intent.desired_state.properties.insert(
            ROLLBACK_GENERATION_KEY.into(),
            StateValue::Integer(7),
        );
        intent.required_resources.insert(nixos_generation_resource(7));

        assert_eq!(
            adapter().compile(&intent),
            Err(NixOSAdapterError::ConflictingActivationModes)
        );
    }

    #[test]
    fn rollback_plan_binds_generation_identity() {
        let mut intent = DeploymentIntent::new("rollback-2", "host-01");
        intent
            .desired_state
            .properties
            .insert(ROLLBACK_KEY.into(), StateValue::Bool(true));
        intent.desired_state.properties.insert(
            ROLLBACK_GENERATION_KEY.into(),
            StateValue::Integer(42),
        );
        intent.required_resources.insert(nixos_generation_resource(42));

        let plan = rollback_adapter(42).compile(&intent).expect("compile");
        assert!(plan.steps.iter().any(|step| {
            step.kind == PlanStepKind::Rollback && step.description.contains("generation 42")
        }));
        assert!(plan.digest().is_ok());
    }

    #[test]
    fn dry_activate_is_not_marked_as_mutating_for_rollback_policy() {
        let mut intent = DeploymentIntent::new("dry-1", "host-01");
        intent.desired_state.properties.insert(
            REBUILD_KEY.into(),
            StateValue::String("dry-activate".into()),
        );

        let plan = adapter().compile(&intent).expect("compile");
        assert!(!plan.rollback.allowed);
        assert_eq!(plan.rollback.max_attempts, 0);
    }

    #[test]
    fn activation_mode_is_typed_before_lowering() {
        assert_eq!(
            NixActivationMode::parse_rebuild("switch").expect("switch"),
            NixActivationMode::Switch
        );
        assert_eq!(
            NixActivationMode::parse_rebuild("test").expect("test"),
            NixActivationMode::Test
        );
        assert_eq!(
            NixActivationMode::parse_rebuild("boot").expect("boot"),
            NixActivationMode::Boot
        );
        assert_eq!(
            NixActivationMode::parse_rebuild("dry-activate").expect("dry-activate"),
            NixActivationMode::DryActivate
        );
    }

    #[test]
    fn activation_modes_produce_distinct_authorized_plan_digests() {
        let modes = ["switch", "test", "boot", "dry-activate"];
        let digests = modes
            .iter()
            .map(|mode| {
                let mut intent = DeploymentIntent::new(*mode, "host-01");
                intent.desired_state.properties.insert(
                    REBUILD_KEY.into(),
                    StateValue::String((*mode).into()),
                );
                adapter()
                    .compile(&intent)
                    .expect("compile")
                    .digest()
                    .expect("digest")
            })
            .collect::<BTreeSet<_>>();

        assert_eq!(digests.len(), modes.len());
    }

    #[test]
    fn rollback_generation_changes_plan_digest() {
        let mut a = DeploymentIntent::new("rollback-a", "host-01");
        a.desired_state.properties.insert(
            ROLLBACK_KEY.into(),
            StateValue::Bool(true),
        );
        a.desired_state.properties.insert(
            ROLLBACK_GENERATION_KEY.into(),
            StateValue::Integer(42),
        );
        a.required_resources.insert(nixos_generation_resource(42));

        let mut b = a.clone();
        b.desired_state.properties.insert(
            ROLLBACK_GENERATION_KEY.into(),
            StateValue::Integer(43),
        );
        b.required_resources.clear();
        b.required_resources.insert(nixos_generation_resource(43));

        let plan_a = rollback_adapter(42).compile(&a).expect("compile a");
        let plan_b = rollback_adapter(43).compile(&b).expect("compile b");

        assert_ne!(
            plan_a.digest().expect("digest a"),
            plan_b.digest().expect("digest b")
        );
        assert!(plan_b.steps.iter().any(|step| {
            step.kind == PlanStepKind::Rollback && step.description.contains("generation 43")
        }));
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
