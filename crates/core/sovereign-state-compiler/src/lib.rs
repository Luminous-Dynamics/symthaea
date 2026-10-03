//! Platform-neutral deployment contracts.
//!
//! This crate intentionally contains no operating-system, shell, transport,
//! or privileged execution implementation. It models the common vocabulary
//! needed to lower a desired state into a target-native deployment plan.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const SCHEMA_VERSION: &str = "ssc/v0.1";

/// Stable logical identity for a deployment target.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TargetId(pub String);

impl From<&str> for TargetId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

/// Stable logical identity for a deployment artifact.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ArtifactId(pub String);

impl From<&str> for ArtifactId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

/// Capabilities are the portability boundary. An adapter exposes only the
/// operations that the concrete target can actually authorize and perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Capability {
    ObserveHardware,
    InstallApplication,
    RemoveApplication,
    ConfigureSystem,
    UpdateSystem,
    Rollback,
    Reboot,
    ReplaceOs,
    ModifyBootChain,
    ConfigureSecureBoot,
    EncryptStorage,
    CreateRecoveryEnvironment,
    RemoteExecution,
    AttestState,
}

/// A target's declared capability surface.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetProfile {
    pub identity: TargetId,
    pub platform: String,
    pub capabilities: BTreeSet<Capability>,
}

/// Desired state is intentionally structured as data, never as a shell
/// command. Higher-level domain crates may add richer typed state later.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct DesiredState {
    pub properties: BTreeMap<String, String>,
}

/// Reference to an immutable artifact or artifact set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub id: ArtifactId,
    pub version: Option<String>,
    pub digest: String,
}

/// A platform-neutral request produced by a human, an application, or a
/// planning layer such as Symthaea.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeploymentIntent {
    pub schema_version: String,
    pub intent_id: String,
    pub target: TargetId,
    pub artifacts: Vec<ArtifactRef>,
    pub desired_state: DesiredState,
    pub required_capabilities: BTreeSet<Capability>,
    pub expires_at_ms: Option<u64>,
}

impl DeploymentIntent {
    pub fn new(intent_id: impl Into<String>, target: impl Into<TargetId>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_owned(),
            intent_id: intent_id.into(),
            target: target.into(),
            artifacts: Vec::new(),
            desired_state: DesiredState::default(),
            required_capabilities: BTreeSet::new(),
            expires_at_ms: None,
        }
    }
}

/// Generic lifecycle verbs. Adapters lower these to native platform
/// operations; they do not expose native commands through this API.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanStepKind {
    Observe,
    StageArtifacts,
    ApplyDesiredState,
    Reboot,
    Verify,
    Rollback,
}

/// A single target-independent step in an authorized plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanStep {
    pub sequence: u32,
    pub kind: PlanStepKind,
    pub required_capabilities: BTreeSet<Capability>,
    pub description: String,
}

/// Policy describing what evidence must be observed after execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct VerificationPolicy {
    pub required_properties: BTreeSet<String>,
    pub require_attestation: bool,
}

/// Policy describing permitted recovery behavior.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RollbackPolicy {
    pub allowed: bool,
    pub max_attempts: u8,
}

/// Evidence authorizing a specific intent. A future version can bind a
/// signature envelope without changing the high-level concept.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorizationEvidence {
    pub authority_id: String,
    pub intent_id: String,
    pub granted_capabilities: BTreeSet<Capability>,
    pub nonce: String,
    pub valid_until_ms: Option<u64>,
}

/// A compiled deployment plan. It is still OS-neutral: the adapter is
/// responsible for lowering each lifecycle step into native mechanisms.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeploymentPlan {
    pub schema_version: String,
    pub intent: DeploymentIntent,
    pub target_profile: TargetProfile,
    pub authorization: AuthorizationEvidence,
    pub steps: Vec<PlanStep>,
    pub verification: VerificationPolicy,
    pub rollback: RollbackPolicy,
}

/// The minimal adapter contract for target-specific lowering.
///
/// No method here accepts a shell string or raw command line. A target adapter
/// receives structured intent and returns structured plan data.
pub trait TargetAdapter {
    type Error: std::error::Error + Send + Sync + 'static;

    fn describe_target(&self) -> Result<TargetProfile, Self::Error>;

    fn compile(
        &self,
        intent: &DeploymentIntent,
        authorization: &AuthorizationEvidence,
    ) -> Result<DeploymentPlan, Self::Error>;
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PlanValidationError {
    #[error("intent target does not match target profile")]
    TargetMismatch,
    #[error("authorization is for a different intent")]
    AuthorizationIntentMismatch,
    #[error("required capability is not granted")]
    MissingGrantedCapability,
    #[error("required capability is not supported by target")]
    MissingTargetCapability,
}

impl DeploymentPlan {
    /// Validate the capability and identity invariants that are independent
    /// of any particular operating system.
    pub fn validate(&self) -> Result<(), PlanValidationError> {
        if self.intent.target != self.target_profile.identity {
            return Err(PlanValidationError::TargetMismatch);
        }

        if self.authorization.intent_id != self.intent.intent_id {
            return Err(PlanValidationError::AuthorizationIntentMismatch);
        }

        for capability in &self.intent.required_capabilities {
            if !self.authorization.granted_capabilities.contains(capability) {
                return Err(PlanValidationError::MissingGrantedCapability);
            }
            if !self.target_profile.capabilities.contains(capability) {
                return Err(PlanValidationError::MissingTargetCapability);
            }
        }

        for step in &self.steps {
            for capability in &step.required_capabilities {
                if !self.authorization.granted_capabilities.contains(capability) {
                    return Err(PlanValidationError::MissingGrantedCapability);
                }
                if !self.target_profile.capabilities.contains(capability) {
                    return Err(PlanValidationError::MissingTargetCapability);
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_profile() -> TargetProfile {
        TargetProfile {
            identity: TargetId::from("host-01"),
            platform: "nixos".into(),
            capabilities: [
                Capability::ConfigureSystem,
                Capability::InstallApplication,
                Capability::Rollback,
            ]
            .into_iter()
            .collect(),
        }
    }

    #[test]
    fn rejects_target_mismatch() {
        let mut intent = DeploymentIntent::new("intent-1", "host-02");
        intent.required_capabilities.insert(Capability::ConfigureSystem);

        let auth = AuthorizationEvidence {
            authority_id: "owner".into(),
            intent_id: "intent-1".into(),
            granted_capabilities: [Capability::ConfigureSystem].into_iter().collect(),
            nonce: "nonce-1".into(),
            valid_until_ms: None,
        };

        let plan = DeploymentPlan {
            schema_version: SCHEMA_VERSION.into(),
            intent,
            target_profile: sample_profile(),
            authorization: auth,
            steps: vec![],
            verification: VerificationPolicy::default(),
            rollback: RollbackPolicy::default(),
        };

        assert_eq!(plan.validate(), Err(PlanValidationError::TargetMismatch));
    }

    #[test]
    fn accepts_intersection_of_required_granted_and_target_capabilities() {
        let mut intent = DeploymentIntent::new("intent-1", "host-01");
        intent.required_capabilities.insert(Capability::ConfigureSystem);
        intent.required_capabilities.insert(Capability::Rollback);

        let auth = AuthorizationEvidence {
            authority_id: "owner".into(),
            intent_id: "intent-1".into(),
            granted_capabilities: [
                Capability::ConfigureSystem,
                Capability::Rollback,
            ]
            .into_iter()
            .collect(),
            nonce: "nonce-1".into(),
            valid_until_ms: None,
        };

        let plan = DeploymentPlan {
            schema_version: SCHEMA_VERSION.into(),
            intent,
            target_profile: sample_profile(),
            authorization: auth,
            steps: vec![
                PlanStep {
                    sequence: 0,
                    kind: PlanStepKind::ApplyDesiredState,
                    required_capabilities: [Capability::ConfigureSystem].into_iter().collect(),
                    description: "apply target configuration".into(),
                },
                PlanStep {
                    sequence: 1,
                    kind: PlanStepKind::Rollback,
                    required_capabilities: [Capability::Rollback].into_iter().collect(),
                    description: "rollback if verification fails".into(),
                },
            ],
            verification: VerificationPolicy::default(),
            rollback: RollbackPolicy {
                allowed: true,
                max_attempts: 1,
            },
        };

        assert!(plan.validate().is_ok());
    }

    #[test]
    fn serialization_is_stable_for_btreemap_state() {
        let mut intent = DeploymentIntent::new("intent-1", "host-01");
        intent
            .desired_state
            .properties
            .insert("z".into(), "last".into());
        intent
            .desired_state
            .properties
            .insert("a".into(), "first".into());

        let bytes = serde_json::to_vec(&intent).expect("serialize");
        let text = String::from_utf8(bytes).expect("utf8");
        assert!(text.find("\"a\"").unwrap() < text.find("\"z\"").unwrap());
    }
}
