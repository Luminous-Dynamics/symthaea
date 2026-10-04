//! NixOS target adapter for the Sovereign State Compiler.
//!
//! This crate owns NixOS-specific interpretation of the neutral deployment
//! contract. It deliberately stops at an abstract lifecycle plan: native
//! commands, transport, and privileged execution remain outside the adapter.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use sovereign_state_compiler::{
    Capability, DeploymentDisposition, DeploymentIntent, DeploymentPlan, PlanStep, PlanStepKind,
    PlanValidationError, RollbackPolicy, TargetAdapter, TargetId, TargetSnapshot,
    VerificationPolicy,
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
const ROLLBACK_REALIZATION_KEY: &str = "nixos.rollback-realization";
pub const NIXOS_GENERATION_RESOURCE_KIND: &str = "nixos-generation";
const NIXOS_GENERATION_DIGEST_DOMAIN: &[u8] = b"LUMINOUS-DYNAMICS/SSC/NIXOS-GENERATION/v1\0";

/// Construct the resource identity used to bind a rollback to a specific
/// observed NixOS generation.
///
/// The digest uses a protocol-specific domain separator, fixed-width
/// generation number, length-prefixed realization identity, and BLAKE3. This
/// prevents a generation ordinal from being treated as a reusable authority
/// token when the underlying system realization has changed.
pub fn nixos_generation_resource(
    generation: u64,
    realization: &str,
) -> Result<sovereign_state_compiler::ResourceRef, NixOSAdapterError> {
    if generation == 0 {
        return Err(NixOSAdapterError::InvalidGenerationNumber);
    }
    if realization.is_empty() {
        return Err(NixOSAdapterError::InvalidGenerationRealization);
    }
    let store_entry = realization
        .strip_prefix("/nix/store/")
        .ok_or(NixOSAdapterError::InvalidGenerationRealizationPath)?;
    if store_entry.is_empty()
        || store_entry.contains('/')
        || store_entry == "."
        || store_entry == ".."
        || store_entry.contains("/../")
        || store_entry.contains("/./")
    {
        return Err(NixOSAdapterError::InvalidGenerationRealizationPath);
    }

    let realization_bytes = realization.as_bytes();
    let mut bytes =
        Vec::with_capacity(NIXOS_GENERATION_DIGEST_DOMAIN.len() + 8 + 8 + realization_bytes.len());
    bytes.extend_from_slice(NIXOS_GENERATION_DIGEST_DOMAIN);
    bytes.extend_from_slice(&generation.to_be_bytes());
    bytes.extend_from_slice(&(realization_bytes.len() as u64).to_be_bytes());
    bytes.extend_from_slice(realization_bytes);

    Ok(sovereign_state_compiler::ResourceRef {
        kind: NIXOS_GENERATION_RESOURCE_KIND.into(),
        identity: sovereign_state_compiler::ContentDigest::blake3(&bytes),
    })
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

    pub fn required_capabilities(self) -> BTreeSet<Capability> {
        match self {
            Self::Switch | Self::Boot => [
                Capability::ConfigureSystem,
                Capability::UpdateSystem,
                Capability::ModifyBootChain,
            ]
            .into_iter()
            .collect(),
            Self::Test => [
                Capability::ConfigureSystem,
                Capability::UpdateSystem,
                Capability::ObserveState,
            ]
            .into_iter()
            .collect(),
            Self::DryActivate => [Capability::ConfigureSystem, Capability::UpdateSystem]
                .into_iter()
                .collect(),
            Self::Rollback { .. } => [Capability::Rollback].into_iter().collect(),
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Switch => "activate the NixOS configuration and make its generation current",
            Self::Test => {
                "temporarily activate the NixOS configuration without changing the boot default"
            }
            Self::Boot => {
                "build the NixOS configuration and select it for the next boot without activating now"
            }
            Self::DryActivate => {
                "evaluate NixOS activation changes without activating the configuration"
            }
            Self::Rollback { .. } => "rollback to an explicitly identified NixOS generation",
        }
    }

    pub fn mutates_target_state(self) -> bool {
        !matches!(self, Self::DryActivate)
    }

    /// Map the native NixOS transition into the neutral SSC verification
    /// disposition. The adapter owns this mapping; the core owns only the
    /// abstract post-state contract.
    pub fn verification_disposition(self) -> DeploymentDisposition {
        match self {
            Self::Switch => DeploymentDisposition::Applied,
            Self::Test => DeploymentDisposition::TemporarilyApplied,
            Self::Boot => DeploymentDisposition::SelectedForNextActivation,
            Self::DryActivate => DeploymentDisposition::NotActivated,
            Self::Rollback { .. } => DeploymentDisposition::RollbackTarget,
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
        Capability::ObserveState,
        Capability::ConfigureSystem,
        Capability::UpdateSystem,
        Capability::Rollback,
        Capability::Reboot,
        Capability::ModifyBootChain,
    ]
    .into_iter()
    .collect()
}

/// A target adapter that lowers a narrow, explicit NixOS state vocabulary into
/// target-neutral lifecycle steps.
///
/// Recognized prototype state properties:
///
/// * `nixos.rebuild`: "switch", "test", "boot", or "dry-activate"
/// * `nixos.home-manager`: boolean
/// * `applications.install`: currently rejected until an exact user-profile resource is bound
/// * `applications.remove`: currently rejected until an exact user-profile resource is bound
/// * `system.reboot`: boolean
/// * `nixos.rollback`: boolean; `true` requires `nixos.rollback-generation`
/// * `nixos.rollback-generation`: positive generation number
/// * `nixos.rollback-realization`: exact observed system-profile/store realization identity
///
/// Rollback requires both values and binds their canonical pair into the
/// `nixos-generation` resource identity carried by `required_resources`.
///
/// Unknown properties are rejected rather than silently ignored.
#[derive(Debug, Clone)]
pub struct NixOSTargetAdapter {
    snapshot: TargetSnapshot,
}

impl NixOSTargetAdapter {
    /// Synthetic constructor reserved for deterministic unit tests and local
    /// contract development. It is intentionally unavailable to production
    /// builds so a live NixOS deployment cannot bypass the Nixward observation
    /// boundary with fabricated snapshot evidence.
    #[cfg(test)]
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
    ) -> Result<Self, NixOSAdapterError> {
        Self::from_snapshot(TargetSnapshot {
            profile: sovereign_state_compiler::TargetProfile {
                identity: target.into(),
                platform: NIXOS_PLATFORM.into(),
                capabilities,
            },
            observed_at_ms,
            observation_digest,
            resources,
        })
    }

    pub fn from_snapshot(snapshot: TargetSnapshot) -> Result<Self, NixOSAdapterError> {
        if snapshot.profile.platform != NIXOS_PLATFORM {
            return Err(NixOSAdapterError::WrongPlatform(
                snapshot.profile.platform.clone(),
            ));
        }
        if snapshot.profile.identity.0.is_empty() || snapshot.profile.identity.0.trim().is_empty() {
            return Err(NixOSAdapterError::InvalidTargetIdentity);
        }
        if snapshot.observation_digest.algorithm.is_empty()
            || snapshot.observation_digest.value.is_empty()
        {
            return Err(NixOSAdapterError::InvalidObservationDigest);
        }
        for resource in &snapshot.resources {
            if resource.kind.is_empty() {
                return Err(NixOSAdapterError::InvalidResourceIdentity);
            }
            if resource.kind != NIXOS_GENERATION_RESOURCE_KIND {
                return Err(NixOSAdapterError::UnsupportedResourceKind(resource.kind.clone()));
            }
            if resource.identity.algorithm.is_empty() || resource.identity.value.is_empty() {
                return Err(NixOSAdapterError::InvalidResourceIdentity);
            }
        }
        if let Some(capability) = snapshot
            .profile
            .capabilities
            .iter()
            .find(|capability| !default_nixos_capabilities().contains(capability))
        {
            return Err(NixOSAdapterError::UnsupportedCapability(*capability));
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
                expected: "positive integer",
            }),
        }
    }

    fn reject_unknown_properties(intent: &DeploymentIntent) -> Result<(), NixOSAdapterError> {
        const SUPPORTED: [&str; 8] = [
            REBUILD_KEY,
            HOME_MANAGER_KEY,
            INSTALL_KEY,
            REMOVE_KEY,
            REBOOT_KEY,
            ROLLBACK_KEY,
            ROLLBACK_GENERATION_KEY,
            ROLLBACK_REALIZATION_KEY,
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
        if intent.desired_state.properties.contains_key(INSTALL_KEY)
            || intent.desired_state.properties.contains_key(REMOVE_KEY)
        {
            return Err(NixOSAdapterError::ApplicationMutationScopeRequired);
        }
        let reboot = Self::property_bool(intent, REBOOT_KEY)?.unwrap_or(false);
        let rollback_requested = Self::property_bool(intent, ROLLBACK_KEY)?.unwrap_or(false);
        let rollback_generation = Self::property_u64(intent, ROLLBACK_GENERATION_KEY)?;
        let rollback_realization = Self::property_string(intent, ROLLBACK_REALIZATION_KEY)?;

        if rollback_requested && rollback_generation.is_none() {
            return Err(NixOSAdapterError::RollbackGenerationRequired);
        }
        if rollback_requested && rollback_realization.is_none() {
            return Err(NixOSAdapterError::RollbackRealizationRequired);
        }
        if !rollback_requested && (rollback_generation.is_some() || rollback_realization.is_some())
        {
            return Err(NixOSAdapterError::UnexpectedRollbackTargetIdentity);
        }

        let activation_mode = match (rebuild.as_deref(), rollback_generation) {
            (Some(_), Some(_)) => return Err(NixOSAdapterError::ConflictingActivationModes),
            (Some(mode), None) => Some(NixActivationMode::parse_rebuild(mode)?),
            (None, Some(generation)) => Some(NixActivationMode::Rollback { generation }),
            (None, None) => None,
        };

        match activation_mode {
            Some(NixActivationMode::DryActivate) if reboot || home_manager => {
                return Err(NixOSAdapterError::ConflictingTransitionSemantics);
            }
            Some(NixActivationMode::Test | NixActivationMode::Boot) if reboot => {
                return Err(NixOSAdapterError::ConflictingTransitionSemantics);
            }
            _ => {}
        }

        if let Some(NixActivationMode::Rollback { generation }) = activation_mode {
            let realization = rollback_realization
                .as_deref()
                .expect("validated rollback realization");
            let required_resource = nixos_generation_resource(generation, realization)?;
            if !intent.required_resources.contains(&required_resource) {
                return Err(NixOSAdapterError::UnboundRollbackGeneration(generation));
            }
        }

        if matches!(activation_mode, Some(NixActivationMode::Rollback { .. })) && home_manager {
            return Err(NixOSAdapterError::ConflictingActivationModes);
        }

        if !intent.artifacts.is_empty() && activation_mode.is_none() && !home_manager {
            return Err(NixOSAdapterError::ArtifactsRequireRealization);
        }

        let mut steps = Vec::new();

        Self::push_step(
            &mut steps,
            PlanStepKind::Observe,
            [Capability::ObserveState],
            "observe target before applying state",
        );

        match activation_mode {
            Some(
                mode @ (NixActivationMode::Switch
                | NixActivationMode::Test
                | NixActivationMode::Boot
                | NixActivationMode::DryActivate),
            ) => {
                if !intent.artifacts.is_empty() {
                    Self::push_step(
                        &mut steps,
                        PlanStepKind::StageArtifacts,
                        [Capability::UpdateSystem],
                        format!(
                            "stage {} NixOS deployment artifact(s)",
                            intent.artifacts.len()
                        ),
                    );
                }

                Self::push_step(
                    &mut steps,
                    PlanStepKind::ApplyDesiredState,
                    mode.required_capabilities(),
                    mode.description(),
                );
            }
            Some(NixActivationMode::Rollback { generation }) => {
                Self::push_step(
                    &mut steps,
                    PlanStepKind::Rollback,
                    [Capability::Rollback],
                    format!(
                        "{} {generation}",
                        NixActivationMode::Rollback { generation }.description()
                    ),
                );
            }
            None => {}
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
            [Capability::ObserveState],
            "verify declared target state after execution",
        );

        let disposition = match activation_mode {
            Some(mode) => mode.verification_disposition(),
            None if home_manager => DeploymentDisposition::Applied,
            None if reboot => DeploymentDisposition::Rebooted,
            None => DeploymentDisposition::Unchanged,
        };

        let verification = VerificationPolicy {
            expected_state: intent.desired_state.clone(),
            disposition,
            require_attestation: false,
        };

        let has_mutation = steps.iter().any(|step| match step.kind {
            PlanStepKind::StageArtifacts => {
                activation_mode.is_none_or(|mode| mode.mutates_target_state())
            }
            PlanStepKind::ApplyDesiredState => {
                activation_mode.is_none_or(|mode| mode.mutates_target_state()) || home_manager
            }
            PlanStepKind::Reboot | PlanStepKind::Rollback => true,
            _ => false,
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
    #[error("NixOS adapter target identity is empty")]
    InvalidTargetIdentity,
    #[error("NixOS adapter target snapshot observation digest is incomplete")]
    InvalidObservationDigest,
    #[error("NixOS adapter target resource identity is empty or incomplete")]
    InvalidResourceIdentity,
    #[error("resource kind is not supported by the canonical NixOS adapter surface: {0}")]
    UnsupportedResourceKind(String),
    #[error("capability {0:?} is not supported by the canonical NixOS adapter surface")]
    UnsupportedCapability(Capability),
    #[error("unsupported NixOS state property: {0}")]
    UnsupportedProperty(String),
    #[error("state property {key} must be a {expected}")]
    PropertyType {
        key: &'static str,
        expected: &'static str,
    },
    #[error("unsupported nixos.rebuild mode: {0}")]
    InvalidRebuildMode(String),
    #[error("nixos.rollback=true requires a positive nixos.rollback-generation")]
    RollbackGenerationRequired,
    #[error("nixos.rollback-generation must be greater than zero")]
    InvalidRollbackGeneration,
    #[error("NixOS generation number must be greater than zero")]
    InvalidGenerationNumber,
    #[error("rollback target realization identity must not be empty")]
    InvalidGenerationRealization,
    #[error("NixOS generation realization must be rooted in /nix/store")]
    InvalidGenerationRealizationPath,
    #[error("nixos.rollback=true requires an exact nixos.rollback-realization")]
    RollbackRealizationRequired,
    #[error("rollback target identity is only valid when rollback is requested")]
    UnexpectedRollbackTargetIdentity,
    #[error("rollback generation {0} is not bound to an observed NixOS generation resource")]
    UnboundRollbackGeneration(u64),
    #[error("NixOS activation modes cannot be combined in one intent")]
    ConflictingActivationModes,
    #[error("artifacts were provided without a NixOS realization operation")]
    ArtifactsRequireRealization,
    #[error("application mutation requires an explicit user-profile resource binding")]
    ApplicationMutationScopeRequired,
    #[error("requested NixOS transition has ambiguous post-state semantics")]
    ConflictingTransitionSemantics,
    #[error("compiled plan violates neutral compiler invariants: {0}")]
    PlanValidation(PlanValidationError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use sovereign_state_compiler::{ArtifactId, ArtifactRef, ContentDigest, StateValue};

    fn adapter() -> NixOSTargetAdapter {
        NixOSTargetAdapter::new("host-01", 1_000)
    }

    fn rollback_adapter(generation: u64, realization: &str) -> NixOSTargetAdapter {
        let mut snapshot = adapter().describe_target().expect("snapshot");
        snapshot.resources.insert(
            nixos_generation_resource(generation, realization).expect("generation resource"),
        );
        NixOSTargetAdapter::from_snapshot(snapshot).expect("nixos snapshot")
    }

    #[test]
    fn from_snapshot_rejects_empty_target_identity() {
        let mut snapshot = adapter().describe_target().expect("snapshot");
        snapshot.profile.identity = TargetId::from("");

        assert_eq!(
            NixOSTargetAdapter::from_snapshot(snapshot),
            Err(NixOSAdapterError::InvalidTargetIdentity)
        );
    }

    #[test]
    fn from_snapshot_rejects_incomplete_observation_digest() {
        let mut snapshot = adapter().describe_target().expect("snapshot");
        snapshot.observation_digest.value.clear();

        assert_eq!(
            NixOSTargetAdapter::from_snapshot(snapshot),
            Err(NixOSAdapterError::InvalidObservationDigest)
        );
    }

    #[test]
    fn from_snapshot_rejects_unknown_resource_kind() {
        let mut snapshot = adapter().describe_target().expect("snapshot");
        snapshot.resources.insert(sovereign_state_compiler::ResourceRef {
            kind: "arbitrary-resource".into(),
            identity: ContentDigest::blake3(b"resource"),
        });

        assert_eq!(
            NixOSTargetAdapter::from_snapshot(snapshot),
            Err(NixOSAdapterError::UnsupportedResourceKind(
                "arbitrary-resource".into()
            ))
        );
    }

    #[test]
    fn from_snapshot_rejects_incomplete_resource_identity() {
        let mut snapshot = adapter().describe_target().expect("snapshot");
        snapshot.resources.insert(sovereign_state_compiler::ResourceRef {
            kind: NIXOS_GENERATION_RESOURCE_KIND.into(),
            identity: ContentDigest {
                algorithm: String::new(),
                value: String::new(),
            },
        });

        assert_eq!(
            NixOSTargetAdapter::from_snapshot(snapshot),
            Err(NixOSAdapterError::InvalidResourceIdentity)
        );
    }

    #[test]
    fn from_snapshot_rejects_unsupported_capability_surface() {
        let mut snapshot = adapter().describe_target().expect("snapshot");
        snapshot
            .profile
            .capabilities
            .insert(Capability::CreateRecoveryEnvironment);

        assert_eq!(
            NixOSTargetAdapter::from_snapshot(snapshot),
            Err(NixOSAdapterError::UnsupportedCapability(
                Capability::CreateRecoveryEnvironment
            ))
        );
    }

    #[test]
    fn from_observation_rejects_unsupported_capability_surface() {
        let result = NixOSTargetAdapter::from_observation(
            "host-01",
            1_000,
            ContentDigest::blake3(b"observation"),
            [Capability::ReplaceOs].into_iter().collect(),
            BTreeSet::new(),
        );

        assert_eq!(
            result,
            Err(NixOSAdapterError::UnsupportedCapability(Capability::ReplaceOs))
        );
    }

    #[test]
    fn from_observation_accepts_canonical_capability_surface() {
        let result = NixOSTargetAdapter::from_observation(
            "host-01",
            1_000,
            ContentDigest::blake3(b"observation"),
            default_nixos_capabilities(),
            BTreeSet::new(),
        );

        assert!(result.is_ok());
    }

    #[test]
    fn default_capabilities_do_not_overclaim_recovery() {
        assert!(!default_nixos_capabilities().contains(&Capability::CreateRecoveryEnvironment));
        assert!(default_nixos_capabilities().contains(&Capability::ObserveState));
        assert!(!default_nixos_capabilities().contains(&Capability::ObserveHardware));
        assert!(default_nixos_capabilities().contains(&Capability::Rollback));
    }

    #[test]
    fn compiles_a_nixos_switch_without_native_commands() {
        let adapter = adapter();
        let mut intent = DeploymentIntent::new("switch-1", "host-01");
        intent
            .desired_state
            .properties
            .insert(REBUILD_KEY.into(), StateValue::String("switch".into()));
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
        assert_eq!(plan.verification.expected_state, intent.desired_state);
        assert_eq!(
            plan.verification.disposition,
            DeploymentDisposition::Applied
        );
    }

    #[test]
    fn deterministic_nixos_rebuild_plan_vectors() {
        let set =
            |capabilities: &[Capability]| capabilities.iter().copied().collect::<BTreeSet<_>>();
        let vectors = [
            (
                "switch",
                vec![
                    (PlanStepKind::Observe, set(&[Capability::ObserveState])),
                    (
                        PlanStepKind::ApplyDesiredState,
                        set(&[
                            Capability::ConfigureSystem,
                            Capability::UpdateSystem,
                            Capability::ModifyBootChain,
                        ]),
                    ),
                    (PlanStepKind::Verify, set(&[Capability::ObserveState])),
                ],
            ),
            (
                "test",
                vec![
                    (PlanStepKind::Observe, set(&[Capability::ObserveState])),
                    (
                        PlanStepKind::ApplyDesiredState,
                        set(&[
                            Capability::ObserveState,
                            Capability::ConfigureSystem,
                            Capability::UpdateSystem,
                        ]),
                    ),
                    (PlanStepKind::Verify, set(&[Capability::ObserveState])),
                ],
            ),
            (
                "boot",
                vec![
                    (PlanStepKind::Observe, set(&[Capability::ObserveState])),
                    (
                        PlanStepKind::ApplyDesiredState,
                        set(&[
                            Capability::ConfigureSystem,
                            Capability::UpdateSystem,
                            Capability::ModifyBootChain,
                        ]),
                    ),
                    (PlanStepKind::Verify, set(&[Capability::ObserveState])),
                ],
            ),
            (
                "dry-activate",
                vec![
                    (PlanStepKind::Observe, set(&[Capability::ObserveState])),
                    (
                        PlanStepKind::ApplyDesiredState,
                        set(&[Capability::ConfigureSystem, Capability::UpdateSystem]),
                    ),
                    (PlanStepKind::Verify, set(&[Capability::ObserveState])),
                ],
            ),
        ];

        for (mode, expected) in vectors {
            let mut intent = DeploymentIntent::new("vector-1", "host-01");
            intent
                .desired_state
                .properties
                .insert(REBUILD_KEY.into(), StateValue::String(mode.into()));

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
    fn application_mutation_requires_explicit_profile_scope() {
        let adapter = adapter();
        let mut intent = DeploymentIntent::new("app-1", "host-01");
        intent.desired_state.properties.insert(
            INSTALL_KEY.into(),
            StateValue::List(vec![
                StateValue::String("firefox".into()),
                StateValue::String("ripgrep".into()),
            ]),
        );

        assert_eq!(
            adapter.compile(&intent),
            Err(NixOSAdapterError::ApplicationMutationScopeRequired)
        );
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
            nixos_generation_resource(42, "/nix/store/aaa-nixos-system-host")
                .expect("generation resource"),
            nixos_generation_resource(42, "/nix/store/aaa-nixos-system-host")
                .expect("generation resource")
        );
        assert_ne!(
            nixos_generation_resource(42, "/nix/store/aaa-nixos-system-host")
                .expect("generation resource"),
            nixos_generation_resource(42, "/nix/store/bbb-nixos-system-host")
                .expect("generation resource")
        );
        assert_ne!(
            nixos_generation_resource(42, "/nix/store/aaa-nixos-system-host")
                .expect("generation resource"),
            nixos_generation_resource(43, "/nix/store/aaa-nixos-system-host")
                .expect("generation resource")
        );
    }

    #[test]
    fn rollback_generation_resource_rejects_zero_generation() {
        assert_eq!(
            nixos_generation_resource(0, "/nix/store/aaa-nixos-system-host")
                .expect_err("zero generation"),
            NixOSAdapterError::InvalidGenerationNumber
        );
    }

    #[test]
    fn rollback_generation_resource_binds_exact_realization() {
        assert_eq!(
            nixos_generation_resource(42, "/nix/store/aaa").expect("resource"),
            nixos_generation_resource(42, "/nix/store/aaa").expect("resource")
        );
        assert_ne!(
            nixos_generation_resource(42, "/nix/store/aaa").expect("resource"),
            nixos_generation_resource(42, "/nix/store/bbb").expect("resource")
        );
        assert_ne!(
            nixos_generation_resource(42, "/nix/store/aaa").expect("resource"),
            nixos_generation_resource(43, "/nix/store/aaa").expect("resource")
        );
        assert_eq!(
            nixos_generation_resource(42, "").expect_err("empty realization"),
            NixOSAdapterError::InvalidGenerationRealization
        );
        assert_eq!(
            nixos_generation_resource(42, "/etc/nixos").expect_err("non-store realization"),
            NixOSAdapterError::InvalidGenerationRealizationPath
        );
        assert_eq!(
            nixos_generation_resource(42, "/nix/store/../etc").expect_err("traversal realization"),
            NixOSAdapterError::InvalidGenerationRealizationPath
        );
        assert_eq!(
            nixos_generation_resource(42, "/nix/store/foo/bar")
                .expect_err("nested realization"),
            NixOSAdapterError::InvalidGenerationRealizationPath
        );
    }

    #[test]
    fn rollback_requires_exact_realization_identity() {
        let mut intent = DeploymentIntent::new("rollback-realization-1", "host-01");
        intent
            .desired_state
            .properties
            .insert(ROLLBACK_KEY.into(), StateValue::Bool(true));
        intent
            .desired_state
            .properties
            .insert(ROLLBACK_GENERATION_KEY.into(), StateValue::Integer(42));

        assert_eq!(
            adapter().compile(&intent),
            Err(NixOSAdapterError::RollbackRealizationRequired)
        );
    }

    #[test]
    fn rollback_rejects_unexpected_realization_without_rollback() {
        let mut intent = DeploymentIntent::new("rollback-realization-2", "host-01");
        intent.desired_state.properties.insert(
            ROLLBACK_REALIZATION_KEY.into(),
            StateValue::String("/nix/store/aaa-nixos-system-host".into()),
        );

        assert_eq!(
            adapter().compile(&intent),
            Err(NixOSAdapterError::UnexpectedRollbackTargetIdentity)
        );
    }

    #[test]
    fn rollback_requires_observed_generation_binding() {
        let mut intent = DeploymentIntent::new("rollback-bound-1", "host-01");
        intent
            .desired_state
            .properties
            .insert(ROLLBACK_KEY.into(), StateValue::Bool(true));
        intent
            .desired_state
            .properties
            .insert(ROLLBACK_GENERATION_KEY.into(), StateValue::Integer(42));
        intent.desired_state.properties.insert(
            ROLLBACK_REALIZATION_KEY.into(),
            StateValue::String("/nix/store/aaa-nixos-system-host".into()),
        );

        assert_eq!(
            adapter().compile(&intent),
            Err(NixOSAdapterError::UnboundRollbackGeneration(42))
        );
    }

    #[test]
    fn rollback_rejects_same_generation_with_different_realization() {
        let mut intent = DeploymentIntent::new("rollback-realization-drift", "host-01");
        intent
            .desired_state
            .properties
            .insert(ROLLBACK_KEY.into(), StateValue::Bool(true));
        intent
            .desired_state
            .properties
            .insert(ROLLBACK_GENERATION_KEY.into(), StateValue::Integer(42));
        intent.desired_state.properties.insert(
            ROLLBACK_REALIZATION_KEY.into(),
            StateValue::String("/nix/store/aaa-nixos-system-host".into()),
        );
        intent.required_resources.insert(
            nixos_generation_resource(42, "/nix/store/aaa-nixos-system-host")
                .expect("generation resource"),
        );

        assert_eq!(
            rollback_adapter(42, "/nix/store/bbb-nixos-system-host").compile(&intent),
            Err(NixOSAdapterError::PlanValidation(
                PlanValidationError::MissingTargetResource(
                    nixos_generation_resource(42, "/nix/store/aaa-nixos-system-host")
                        .expect("generation resource")
                )
            ))
        );
    }

    #[test]
    fn rollback_rejects_snapshot_with_different_generation_or_realization() {
        let mut intent = DeploymentIntent::new("rollback-drift", "host-01");
        intent
            .desired_state
            .properties
            .insert(ROLLBACK_KEY.into(), StateValue::Bool(true));
        intent
            .desired_state
            .properties
            .insert(ROLLBACK_GENERATION_KEY.into(), StateValue::Integer(42));
        intent.desired_state.properties.insert(
            ROLLBACK_REALIZATION_KEY.into(),
            StateValue::String("/nix/store/aaa-nixos-system-host".into()),
        );
        intent.required_resources.insert(
            nixos_generation_resource(42, "/nix/store/aaa-nixos-system-host")
                .expect("generation resource"),
        );

        assert_eq!(
            rollback_adapter(43, "/nix/store/bbb-nixos-system-host").compile(&intent),
            Err(NixOSAdapterError::PlanValidation(
                PlanValidationError::MissingTargetResource(
                    nixos_generation_resource(42, "/nix/store/aaa-nixos-system-host")
                        .expect("generation resource"),
                )
            ))
        );
    }

    #[test]
    fn rollback_rejects_generation_zero() {
        let mut intent = DeploymentIntent::new("rollback-zero", "host-01");
        intent
            .desired_state
            .properties
            .insert(ROLLBACK_KEY.into(), StateValue::Bool(true));
        intent
            .desired_state
            .properties
            .insert(ROLLBACK_GENERATION_KEY.into(), StateValue::Integer(0));

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
        intent
            .desired_state
            .properties
            .insert(ROLLBACK_GENERATION_KEY.into(), StateValue::Integer(7));
        intent.desired_state.properties.insert(
            ROLLBACK_REALIZATION_KEY.into(),
            StateValue::String("/nix/store/aaa-nixos-system-host".into()),
        );
        intent.required_resources.insert(
            nixos_generation_resource(7, "/nix/store/aaa-nixos-system-host")
                .expect("generation resource"),
        );

        let mode = NixActivationMode::Rollback { generation: 7 };
        assert_eq!(mode, NixActivationMode::Rollback { generation: 7 });
        let plan = rollback_adapter(7, "/nix/store/aaa-nixos-system-host")
            .compile(&intent)
            .expect("compile");
        assert_eq!(plan.steps[1].kind, PlanStepKind::Rollback);
    }

    #[test]
    fn rollback_cannot_be_combined_with_rebuild() {
        let mut intent = DeploymentIntent::new("rollback-conflict-1", "host-01");
        intent
            .desired_state
            .properties
            .insert(REBUILD_KEY.into(), StateValue::String("switch".into()));
        intent
            .desired_state
            .properties
            .insert(ROLLBACK_KEY.into(), StateValue::Bool(true));
        intent
            .desired_state
            .properties
            .insert(ROLLBACK_GENERATION_KEY.into(), StateValue::Integer(7));
        intent.desired_state.properties.insert(
            ROLLBACK_REALIZATION_KEY.into(),
            StateValue::String("/nix/store/aaa-nixos-system-host".into()),
        );
        intent.required_resources.insert(
            nixos_generation_resource(7, "/nix/store/aaa-nixos-system-host")
                .expect("generation resource"),
        );

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
        intent
            .desired_state
            .properties
            .insert(ROLLBACK_GENERATION_KEY.into(), StateValue::Integer(42));
        intent.desired_state.properties.insert(
            ROLLBACK_REALIZATION_KEY.into(),
            StateValue::String("/nix/store/aaa-nixos-system-host".into()),
        );
        intent.required_resources.insert(
            nixos_generation_resource(42, "/nix/store/aaa-nixos-system-host")
                .expect("generation resource"),
        );

        let plan = rollback_adapter(42, "/nix/store/aaa-nixos-system-host")
            .compile(&intent)
            .expect("compile");
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
    fn artifacts_cannot_be_silently_dropped() {
        let mut intent = DeploymentIntent::new("artifact-only", "host-01");
        intent.artifacts.push(ArtifactRef {
            id: ArtifactId::from("system-config"),
            version: Some("2026.10.03".into()),
            digest: ContentDigest::blake3(b"system-config"),
            provenance: Vec::new(),
        });

        assert_eq!(
            adapter().compile(&intent),
            Err(NixOSAdapterError::ArtifactsRequireRealization)
        );
    }

    #[test]
    fn activation_modes_reject_ambiguous_compositions() {
        let mut dry = DeploymentIntent::new("dry-install", "host-01");
        dry.desired_state.properties.insert(
            REBUILD_KEY.into(),
            StateValue::String("dry-activate".into()),
        );
        dry.desired_state.properties.insert(
            INSTALL_KEY.into(),
            StateValue::List(vec![StateValue::String("ripgrep".into())]),
        );
        assert_eq!(
            adapter().compile(&dry),
            Err(NixOSAdapterError::ApplicationMutationScopeRequired)
        );

        let mut test = DeploymentIntent::new("test-reboot", "host-01");
        test.desired_state
            .properties
            .insert(REBUILD_KEY.into(), StateValue::String("test".into()));
        test.desired_state
            .properties
            .insert(REBOOT_KEY.into(), StateValue::Bool(true));
        assert_eq!(
            adapter().compile(&test),
            Err(NixOSAdapterError::ConflictingTransitionSemantics)
        );

        let mut boot = DeploymentIntent::new("boot-reboot", "host-01");
        boot.desired_state
            .properties
            .insert(REBUILD_KEY.into(), StateValue::String("boot".into()));
        boot.desired_state
            .properties
            .insert(REBOOT_KEY.into(), StateValue::Bool(true));
        assert_eq!(
            adapter().compile(&boot),
            Err(NixOSAdapterError::ConflictingTransitionSemantics)
        );
    }

    #[test]
    fn activation_modes_map_to_distinct_verification_dispositions() {
        assert_eq!(
            NixActivationMode::Switch.verification_disposition(),
            DeploymentDisposition::Applied
        );
        assert_eq!(
            NixActivationMode::Test.verification_disposition(),
            DeploymentDisposition::TemporarilyApplied
        );
        assert_eq!(
            NixActivationMode::Boot.verification_disposition(),
            DeploymentDisposition::SelectedForNextActivation
        );
        assert_eq!(
            NixActivationMode::DryActivate.verification_disposition(),
            DeploymentDisposition::NotActivated
        );
        assert_eq!(
            NixActivationMode::Rollback { generation: 42 }.verification_disposition(),
            DeploymentDisposition::RollbackTarget
        );
    }

    #[test]
    fn activation_mode_owns_its_semantics() {
        assert_eq!(
            NixActivationMode::Switch.required_capabilities(),
            NixActivationMode::Boot.required_capabilities()
        );
        assert_ne!(
            NixActivationMode::Switch.description(),
            NixActivationMode::Boot.description()
        );
        assert!(NixActivationMode::Switch.mutates_target_state());
        assert!(!NixActivationMode::DryActivate.mutates_target_state());
        assert!(
            NixActivationMode::Rollback { generation: 42 }
                .description()
                .contains("explicitly identified")
        );
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
                intent
                    .desired_state
                    .properties
                    .insert(REBUILD_KEY.into(), StateValue::String((*mode).into()));
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
        a.desired_state
            .properties
            .insert(ROLLBACK_KEY.into(), StateValue::Bool(true));
        a.desired_state
            .properties
            .insert(ROLLBACK_GENERATION_KEY.into(), StateValue::Integer(42));
        a.required_resources.insert(
            nixos_generation_resource(42, "/nix/store/aaa-nixos-system-host")
                .expect("generation resource"),
        );

        let mut b = a.clone();
        b.desired_state
            .properties
            .insert(ROLLBACK_GENERATION_KEY.into(), StateValue::Integer(43));
        b.desired_state.properties.insert(
            ROLLBACK_REALIZATION_KEY.into(),
            StateValue::String("/nix/store/bbb-nixos-system-host".into()),
        );
        b.required_resources.clear();
        b.required_resources.insert(
            nixos_generation_resource(43, "/nix/store/bbb-nixos-system-host")
                .expect("generation resource"),
        );

        let plan_a = rollback_adapter(42, "/nix/store/aaa-nixos-system-host")
            .compile(&a)
            .expect("compile a");
        let plan_b = rollback_adapter(43, "/nix/store/bbb-nixos-system-host")
            .compile(&b)
            .expect("compile b");

        assert_ne!(
            plan_a.digest().expect("digest a"),
            plan_b.digest().expect("digest b")
        );
        assert!(plan_b.steps.iter().any(|step| {
            step.kind == PlanStepKind::Rollback && step.description.contains("generation 43")
        }));
    }

    #[test]
    fn reboot_only_uses_rebooted_disposition() {
        let mut intent = DeploymentIntent::new("reboot-only", "host-01");
        intent
            .desired_state
            .properties
            .insert(REBOOT_KEY.into(), StateValue::Bool(true));
        let plan = adapter().compile(&intent).expect("compile");

        assert_eq!(
            plan.verification.disposition,
            DeploymentDisposition::Rebooted
        );
    }

    #[test]
    fn no_state_transition_uses_unchanged_disposition() {
        let intent = DeploymentIntent::new("noop-1", "host-01");
        let plan = adapter().compile(&intent).expect("compile");

        assert_eq!(
            plan.verification.disposition,
            DeploymentDisposition::Unchanged
        );
    }

    #[test]
    fn plan_digest_is_deterministic() {
        let adapter = adapter();
        let mut intent = DeploymentIntent::new("digest-1", "host-01");
        intent
            .desired_state
            .properties
            .insert(REBOOT_KEY.into(), StateValue::Bool(true));

        let a = adapter.compile(&intent).expect("compile");
        let b = adapter.compile(&intent).expect("compile");
        assert_eq!(a, b);
        assert_eq!(a.digest().expect("digest"), b.digest().expect("digest"));
    }
}
