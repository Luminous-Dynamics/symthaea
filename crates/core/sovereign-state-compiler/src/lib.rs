//! Platform-neutral deployment contracts.
//!
//! This crate intentionally contains no operating-system, shell, transport,
//! or privileged execution implementation. It models the common vocabulary
//! needed to lower a desired state into a target-native deployment plan.
//!
//! Security boundary: authorization is detached from compilation and is bound
//! to the exact compiled plan digest before privileged execution is permitted.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const SCHEMA_VERSION: &str = "ssc/v0.1";
const INTENT_DIGEST_DOMAIN: &[u8] = b"LUMINOUS-DYNAMICS/SSC/INTENT-DIGEST/v1\0";
const TARGET_PROFILE_DIGEST_DOMAIN: &[u8] =
    b"LUMINOUS-DYNAMICS/SSC/TARGET-PROFILE-DIGEST/v1\0";
const TARGET_SNAPSHOT_DIGEST_DOMAIN: &[u8] =
    b"LUMINOUS-DYNAMICS/SSC/TARGET-SNAPSHOT-DIGEST/v1\0";
const PLAN_DIGEST_DOMAIN: &[u8] = b"LUMINOUS-DYNAMICS/SSC/PLAN-DIGEST/v1\0";

/// A content-addressed digest.
///
/// algorithm is explicit so external artifact references can use a standard
/// digest scheme without this crate pretending every artifact is a BLAKE3 hash.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ContentDigest {
    pub algorithm: String,
    pub value: String,
}

impl ContentDigest {
    pub fn blake3(bytes: &[u8]) -> Self {
        Self {
            algorithm: "blake3".into(),
            value: blake3::hash(bytes).to_hex().to_string(),
        }
    }
}

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

impl TargetProfile {
    pub fn digest(&self) -> Result<ContentDigest, serde_json::Error> {
        canonical_digest(self, b"LUMINOUS-DYNAMICS/SSC/TARGET-PROFILE/v1\\0")
    }
}

/// Fresh observation of a concrete target.
///
/// The observation digest is deliberately separate from the platform label:
/// a disk, boot chain, management enrollment, architecture, or other
/// target-specific fact can change while the platform name remains the same.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetSnapshot {
    pub profile: TargetProfile,
    pub observed_at_ms: u64,
    pub observation_digest: ContentDigest,
}

impl TargetSnapshot {
    pub fn digest(&self) -> Result<ContentDigest, serde_json::Error> {
        canonical_digest(self, b"LUMINOUS-DYNAMICS/SSC/TARGET-SNAPSHOT/v1\\0")
    }
}

/// Desired state is intentionally structured as data, never as a shell
/// command. Higher-level domain crates may add richer typed state later.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct DesiredState {
    pub properties: BTreeMap<String, StateValue>,
}

/// Typed desired-state values prevent the universal protocol from becoming
/// stringly-typed while remaining extensible enough for heterogeneous targets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateValue {
    Null,
    Bool(bool),
    Integer(i64),
    String(String),
    List(Vec<StateValue>),
    Object(BTreeMap<String, StateValue>),
}

/// Reference to an immutable artifact or artifact set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub id: ArtifactId,
    pub version: Option<String>,
    pub digest: ContentDigest,
    /// Optional reference to an external provenance/attestation object.
    ///
    /// The compiler does not define SLSA, in-toto, OCI, or another supply-chain
    /// format; adapters/integrators may carry those objects by reference.
    pub provenance: Vec<AttestationRef>,
}

/// Reference to externally-defined provenance or attestation evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttestationRef {
    pub media_type: String,
    pub uri: String,
    pub digest: ContentDigest,
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

    pub fn digest(&self) -> Result<ContentDigest, serde_json::Error> {
        canonical_digest(self, TARGET_PROFILE_DIGEST_DOMAIN)
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

/// Evidence authorizing a specific compiled plan.
///
/// The binding fields deliberately cover intent, target capabilities, and the
/// exact compiled plan. A detached signer can add a cryptographic signature
/// envelope above this structure without changing the core model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorizationEvidence {
    pub authority_id: String,
    pub intent_digest: ContentDigest,
    pub target_profile_digest: ContentDigest,
    pub target_snapshot_digest: ContentDigest,
    pub plan_digest: ContentDigest,
    pub granted_capabilities: BTreeSet<Capability>,
    pub nonce: String,
    pub valid_from_ms: Option<u64>,
    pub valid_until_ms: Option<u64>,
}

/// A compiled deployment plan. It is OS-neutral: the adapter is responsible
/// for lowering each lifecycle step into native mechanisms.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeploymentPlan {
    pub schema_version: String,
    pub intent: DeploymentIntent,
    pub target_snapshot: TargetSnapshot,
    /// Maximum age permitted between target observation and authorization.
    /// The executor should re-observe before execution when this window has
    /// elapsed, especially for storage or boot-chain mutations.
    pub max_target_snapshot_age_ms: Option<u64>,
    pub steps: Vec<PlanStep>,
    pub verification: VerificationPolicy,
    pub rollback: RollbackPolicy,
}

impl DeploymentPlan {
    /// Compute the digest that authorization must bind.
    ///
    /// The authorization object is intentionally excluded, avoiding a
    /// circular digest dependency.
    pub fn digest(&self) -> Result<ContentDigest, serde_json::Error> {
        canonical_digest(self, INTENT_DIGEST_DOMAIN)
    }

    /// Validate OS-independent structural, identity, and capability invariants.
    pub fn validate(&self) -> Result<(), PlanValidationError> {
        if self.intent.target != self.target_snapshot.profile.identity {
            return Err(PlanValidationError::TargetMismatch);
        }

        let mut expected_sequence = 0u32;
        for step in &self.steps {
            if step.sequence != expected_sequence {
                return Err(PlanValidationError::NonContiguousPlanSequence);
            }
            expected_sequence = expected_sequence
                .checked_add(1)
                .ok_or(PlanValidationError::SequenceOverflow)?;
            validate_capabilities(
                &step.required_capabilities,
                &self.target_snapshot.profile.capabilities,
                None,
            )?;
        }

        validate_capabilities(
            &self.intent.required_capabilities,
            &self.target_snapshot.profile.capabilities,
            None,
        )?;

        if !self.rollback.allowed && self.rollback.max_attempts != 0 {
            return Err(PlanValidationError::RollbackAttemptsWithoutPermission);
        }

        Ok(())
    }

    /// Bind detached authorization to this exact compiled plan.
    pub fn authorize(
        self,
        authorization: AuthorizationEvidence,
        now_ms: u64,
    ) -> Result<AuthorizedDeploymentPlan, PlanValidationError> {
        self.validate()?;
        authorization.validate_for(&self, now_ms)?;
        Ok(AuthorizedDeploymentPlan {
            plan: self,
            authorization,
        })
    }
}

/// A validated plan carrying detached authorization evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorizedDeploymentPlan {
    pub plan: DeploymentPlan,
    pub authorization: AuthorizationEvidence,
}

impl AuthorizedDeploymentPlan {
    pub fn validate(&self, now_ms: u64) -> Result<(), PlanValidationError> {
        self.plan.validate()?;
        self.authorization.validate_for(&self.plan, now_ms)
    }
}

impl AuthorizationEvidence {
    pub fn validate_for(
        &self,
        plan: &DeploymentPlan,
        now_ms: u64,
    ) -> Result<(), PlanValidationError> {
        if self.authority_id.is_empty() {
            return Err(PlanValidationError::EmptyAuthority);
        }
        if self.nonce.is_empty() {
            return Err(PlanValidationError::EmptyNonce);
        }

        let intent_digest = plan
            .intent
            .digest()
            .map_err(PlanValidationError::Serialization)?;
        if self.intent_digest != intent_digest {
            return Err(PlanValidationError::AuthorizationIntentDigestMismatch);
        }

        if let Some(max_age_ms) = plan.max_target_snapshot_age_ms {
            let age_ms = now_ms.saturating_sub(plan.target_snapshot.observed_at_ms);
            if age_ms > max_age_ms {
                return Err(PlanValidationError::TargetSnapshotStale {
                    age_ms,
                    max_age_ms,
                });
            }
        }

        let target_digest = plan
            .target_snapshot
            .profile
            .digest()
            .map_err(PlanValidationError::Serialization)?;
        if self.target_profile_digest != target_digest {
            return Err(PlanValidationError::AuthorizationTargetDigestMismatch);
        }

        let snapshot_digest = plan
            .target_snapshot
            .digest()
            .map_err(PlanValidationError::Serialization)?;
        if self.target_snapshot_digest != snapshot_digest {
            return Err(PlanValidationError::AuthorizationTargetSnapshotDigestMismatch);
        }

        let plan_digest = plan.digest().map_err(PlanValidationError::Serialization)?;
        if self.plan_digest != plan_digest {
            return Err(PlanValidationError::AuthorizationPlanDigestMismatch);
        }

        if let (Some(from), Some(until)) = (self.valid_from_ms, self.valid_until_ms)
            && from > until
        {
            return Err(PlanValidationError::AuthorizationWindowInvalid);
        }
        if self.valid_from_ms.is_some_and(|from| now_ms < from) {
            return Err(PlanValidationError::AuthorizationNotYetValid);
        }
        if self.valid_until_ms.is_some_and(|until| now_ms > until) {
            return Err(PlanValidationError::AuthorizationExpired);
        }

        for capability in &self.granted_capabilities {
            if !plan.target_snapshot.profile.capabilities.contains(capability) {
                return Err(PlanValidationError::GrantedCapabilityNotSupported(*capability));
            }
        }

        validate_capabilities(
            &plan.intent.required_capabilities,
            &plan.target_snapshot.profile.capabilities,
            Some(&self.granted_capabilities),
        )?;

        for step in &plan.steps {
            validate_capabilities(
                &step.required_capabilities,
                &plan.target_snapshot.profile.capabilities,
                Some(&self.granted_capabilities),
            )?;
        }

        Ok(())
    }
}

fn validate_capabilities(
    required: &BTreeSet<Capability>,
    supported: &BTreeSet<Capability>,
    granted: Option<&BTreeSet<Capability>>,
) -> Result<(), PlanValidationError> {
    for capability in required {
        if !supported.contains(capability) {
            return Err(PlanValidationError::MissingTargetCapability(*capability));
        }
        if let Some(granted) = granted
            && !granted.contains(capability)
        {
            return Err(PlanValidationError::MissingGrantedCapability(*capability));
        }
    }
    Ok(())
}

fn canonical_digest<T: Serialize>(
    value: &T,
    domain: &[u8],
) -> Result<ContentDigest, serde_json::Error> {
    let bytes = serde_json::to_vec(value)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(ContentDigest {
        algorithm: "blake3".into(),
        value: hasher.finalize().to_hex().to_string(),
    })
}

/// The minimal adapter contract for target-specific lowering.
///
/// Compilation is deliberately authorization-free. Authorization happens
/// after a concrete target adapter has produced the exact plan to execute.
///
/// No method here accepts a shell string or raw command line.
pub trait TargetAdapter {
    type Error: std::error::Error + Send + Sync + 'static;

    fn describe_target(&self) -> Result<TargetProfile, Self::Error>;

    fn compile(&self, intent: &DeploymentIntent) -> Result<DeploymentPlan, Self::Error>;
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PlanValidationError {
    #[error("intent target does not match target profile")]
    TargetMismatch,
    #[error("required capability {0:?} is not supported by target")]
    MissingTargetCapability(Capability),
    #[error("required capability {0:?} is not granted")]
    MissingGrantedCapability(Capability),
    #[error("plan step sequence is not contiguous from zero")]
    NonContiguousPlanSequence,
    #[error("plan step sequence overflowed")]
    SequenceOverflow,
    #[error("rollback attempts are configured without rollback permission")]
    RollbackAttemptsWithoutPermission,
    #[error("authorization intent digest does not match the compiled intent")]
    AuthorizationIntentDigestMismatch,
    #[error("authorization target-profile digest does not match the compiled target")]
    AuthorizationTargetDigestMismatch,
    #[error("authorization plan digest does not match the compiled plan")]
    AuthorizationPlanDigestMismatch,
    #[error("authorization target-snapshot digest does not match the observed target")]
    AuthorizationTargetSnapshotDigestMismatch,
    #[error("target snapshot is stale: age {age_ms} ms exceeds maximum {max_age_ms} ms")]
    TargetSnapshotStale { age_ms: u64, max_age_ms: u64 },
    #[error("authorization validity window is invalid")]
    AuthorizationWindowInvalid,
    #[error("authorization authority identifier is empty")]
    EmptyAuthority,
    #[error("authorization nonce is empty")]
    EmptyNonce,
    #[error("authorization grants a capability unsupported by target")]
    GrantedCapabilityNotSupported(Capability),
    #[error("authorization is not yet valid")]
    AuthorizationNotYetValid,
    #[error("authorization has expired")]
    AuthorizationExpired,
    #[error("canonical serialization failed: {0}")]
    Serialization(serde_json::Error),
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

    fn sample_plan() -> DeploymentPlan {
        let mut intent = DeploymentIntent::new("intent-1", "host-01");
        intent.required_capabilities.insert(Capability::ConfigureSystem);
        intent.required_capabilities.insert(Capability::Rollback);

        DeploymentPlan {
            schema_version: SCHEMA_VERSION.into(),
            intent,
            target_snapshot: TargetSnapshot {
                profile: sample_profile(),
                observed_at_ms: 90,
                observation_digest: ContentDigest::blake3(b"hardware-observation"),
            },
            max_target_snapshot_age_ms: Some(200),
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
        }
    }

    fn authorization_for(plan: &DeploymentPlan) -> AuthorizationEvidence {
        AuthorizationEvidence {
            authority_id: "owner".into(),
            intent_digest: plan.intent.digest().expect("intent digest"),
            target_profile_digest: plan
                .target_snapshot
                .profile
                .digest()
                .expect("target digest"),
            target_snapshot_digest: plan.target_snapshot.digest().expect("snapshot digest"),
            plan_digest: plan.digest().expect("plan digest"),
            granted_capabilities: [
                Capability::ConfigureSystem,
                Capability::Rollback,
            ]
            .into_iter()
            .collect(),
            nonce: "nonce-1".into(),
            valid_from_ms: Some(100),
            valid_until_ms: Some(200),
        }
    }

    #[test]
    fn rejects_target_mismatch() {
        let mut plan = sample_plan();
        plan.intent.target = TargetId::from("host-02");

        assert_eq!(plan.validate(), Err(PlanValidationError::TargetMismatch));
    }

    #[test]
    fn accepts_exact_authorized_plan_within_window() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);

        let authorized = plan.authorize(auth, 150).expect("authorized plan");
        assert!(authorized.validate(150).is_ok());
    }

    #[test]
    fn rejects_mutation_after_authorization() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);

        let mut mutated = plan.clone();
        mutated.steps[0].description = "tampered configuration".into();

        assert_eq!(
            mutated.authorize(auth, 150),
            Err(PlanValidationError::AuthorizationPlanDigestMismatch)
        );
    }

    #[test]
    fn rejects_target_capability_change_after_authorization() {
        let plan = sample_plan();
        let mut auth = authorization_for(&plan);
        auth.target_profile_digest.value = "tampered".into();

        assert_eq!(
            plan.authorize(auth, 150),
            Err(PlanValidationError::AuthorizationTargetSnapshotDigestMismatch)
        );
    }

    #[test]
    fn rejects_stale_target_snapshot() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);

        assert_eq!(
            plan.authorize(auth, 291),
            Err(PlanValidationError::TargetSnapshotStale {
                age_ms: 201,
                max_age_ms: 200
            })
        );
    }

    #[test]
    fn rejects_expired_authorization() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);

        assert_eq!(
            plan.authorize(auth, 201),
            Err(PlanValidationError::AuthorizationExpired)
        );
    }

    #[test]
    fn rejects_granted_capability_unsupported_by_target() {
        let plan = sample_plan();
        let mut auth = authorization_for(&plan);
        auth.granted_capabilities.insert(Capability::Reboot);

        assert_eq!(
            plan.authorize(auth, 150),
            Err(PlanValidationError::GrantedCapabilityNotSupported(
                Capability::Reboot
            ))
        );
    }

    #[test]
    fn rejects_empty_authority() {
        let plan = sample_plan();
        let mut auth = authorization_for(&plan);
        auth.authority_id.clear();

        assert_eq!(
            plan.authorize(auth, 150),
            Err(PlanValidationError::EmptyAuthority)
        );
    }

    #[test]
    fn rejects_empty_nonce() {
        let plan = sample_plan();
        let mut auth = authorization_for(&plan);
        auth.nonce.clear();

        assert_eq!(
            plan.authorize(auth, 150),
            Err(PlanValidationError::EmptyNonce)
        );
    }

    #[test]
    fn rejects_capability_added_after_authorization() {
        let mut plan = sample_plan();
        let auth = authorization_for(&plan);

        plan.intent.required_capabilities.insert(Capability::Reboot);

        assert_eq!(
            plan.authorize(auth, 150),
            Err(PlanValidationError::AuthorizationIntentDigestMismatch)
        );
    }

    #[test]
    fn rejects_non_contiguous_steps() {
        let mut plan = sample_plan();
        plan.steps[1].sequence = 3;

        assert_eq!(
            plan.validate(),
            Err(PlanValidationError::NonContiguousPlanSequence)
        );
    }

    #[test]
    fn rejects_rollback_attempts_without_permission() {
        let mut plan = sample_plan();
        plan.rollback = RollbackPolicy {
            allowed: false,
            max_attempts: 1,
        };

        assert_eq!(
            plan.validate(),
            Err(PlanValidationError::RollbackAttemptsWithoutPermission)
        );
    }

    #[test]
    fn digest_changes_when_intent_changes() {
        let plan = sample_plan();
        let before = plan.digest().expect("digest");

        let mut changed = plan;
        changed.intent.desired_state.properties.insert(
            "hostname".into(),
            StateValue::String("new-name".into()),
        );

        assert_ne!(before, changed.digest().expect("digest"));
    }

    #[test]
    fn btree_state_serializes_deterministically() {
        let mut intent = DeploymentIntent::new("intent-1", "host-01");
        intent
            .desired_state
            .properties
            .insert("z".into(), StateValue::String("last".into()));
        intent
            .desired_state
            .properties
            .insert("a".into(), StateValue::String("first".into()));

        let bytes = serde_json::to_vec(&intent).expect("serialize");
        let text = String::from_utf8(bytes).expect("utf8");
        assert!(text.find("\"a\"").unwrap() < text.find("\"z\"").unwrap());
    }
}
