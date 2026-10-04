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

pub const SCHEMA_VERSION: &str = "ssc/v0.5";
const INTENT_DIGEST_DOMAIN: &[u8] = b"LUMINOUS-DYNAMICS/SSC/INTENT-DIGEST/v1\0";
const AUTHORIZATION_DIGEST_DOMAIN: &[u8] =
    b"LUMINOUS-DYNAMICS/SSC/AUTHORIZATION-DIGEST/v1\0";
const TARGET_PROFILE_DIGEST_DOMAIN: &[u8] = b"LUMINOUS-DYNAMICS/SSC/TARGET-PROFILE-DIGEST/v1\0";
const TARGET_SNAPSHOT_DIGEST_DOMAIN: &[u8] = b"LUMINOUS-DYNAMICS/SSC/TARGET-SNAPSHOT-DIGEST/v1\0";
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
    /// Observe the target's realized state for lifecycle planning or verification.
    /// This is intentionally broader than physical-hardware observation so the
    /// neutral protocol can cover virtual, managed, and non-OS targets.
    ObserveState,
    /// Observe physical target hardware/inventory when a capability specifically
    /// requires hardware facts.
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
        canonical_digest(self, TARGET_PROFILE_DIGEST_DOMAIN)
    }
}

/// Fresh observation of a concrete target.
///
/// The observation digest is deliberately separate from the platform label:
/// a disk, boot chain, management enrollment, architecture, or other
/// target-specific fact can change while the platform name remains the same.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ResourceRef {
    pub kind: String,
    pub identity: ContentDigest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetSnapshot {
    pub profile: TargetProfile,
    pub observed_at_ms: u64,
    pub observation_digest: ContentDigest,
    pub resources: BTreeSet<ResourceRef>,
}

impl TargetSnapshot {
    pub fn digest(&self) -> Result<ContentDigest, serde_json::Error> {
        canonical_digest(self, TARGET_SNAPSHOT_DIGEST_DOMAIN)
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
    pub required_resources: BTreeSet<ResourceRef>,
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
            required_resources: BTreeSet::new(),
            expires_at_ms: None,
        }
    }

    pub fn digest(&self) -> Result<ContentDigest, serde_json::Error> {
        canonical_digest(self, INTENT_DIGEST_DOMAIN)
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

/// Platform-neutral disposition that the post-state verifier must establish.
///
/// Adapters map their native transition semantics into this vocabulary. The
/// disposition is deliberately about the resulting transition, not about the
/// native command/mechanism used to realize it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeploymentDisposition {
    /// The requested state became the resulting applied state.
    Applied,
    /// The requested state was applied only for a bounded/test transition.
    TemporarilyApplied,
    /// The requested state was selected for a future activation, but is not
    /// claimed to be active now.
    SelectedForNextActivation,
    /// The requested state was evaluated without activating it.
    NotActivated,
    /// No state transition was requested; existing target state should remain.
    Unchanged,
    /// The target completed an explicit reboot transition without another
    /// deployment disposition being primary.
    Rebooted,
    /// An explicitly identified prior state was reached as the rollback target.
    RollbackTarget,
}

impl Default for DeploymentDisposition {
    fn default() -> Self {
        Self::Applied
    }
}

/// Policy describing what evidence must be observed after execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationPolicy {
    /// Exact desired-state values that the executor must verify after mutation.
    ///
    /// Keeping values rather than property names makes verification
    /// non-ambiguous while remaining platform-neutral.
    pub expected_state: DesiredState,
    /// The transition disposition that the post-state verifier must establish.
    pub disposition: DeploymentDisposition,
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

impl VerificationPolicy {
    pub fn new(
        expected_state: DesiredState,
        disposition: DeploymentDisposition,
        require_attestation: bool,
    ) -> Self {
        Self {
            expected_state,
            disposition,
            require_attestation,
        }
    }
}

impl DeploymentPlan {
    /// Compute the digest that authorization must bind.
    ///
    /// The authorization object is intentionally excluded, avoiding a
    /// circular digest dependency.
    pub fn digest(&self) -> Result<ContentDigest, serde_json::Error> {
        canonical_digest(self, PLAN_DIGEST_DOMAIN)
    }

    /// Validate OS-independent structural, identity, and capability invariants.
    pub fn validate(&self) -> Result<(), PlanValidationError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(PlanValidationError::UnsupportedSchemaVersion(
                self.schema_version.clone(),
            ));
        }
        if self.intent.schema_version != SCHEMA_VERSION {
            return Err(PlanValidationError::UnsupportedSchemaVersion(
                self.intent.schema_version.clone(),
            ));
        }
        if self.intent.intent_id.is_empty() || self.intent.target.0.is_empty() {
            return Err(PlanValidationError::InvalidIntentIdentity);
        }
        if self.target_snapshot.profile.identity.0.is_empty()
            || self.target_snapshot.profile.platform.is_empty()
        {
            return Err(PlanValidationError::InvalidTargetIdentity);
        }
        if !has_concrete_digest(Some(&self.target_snapshot.observation_digest)) {
            return Err(PlanValidationError::InvalidObservationDigest);
        }
        for resource in &self.target_snapshot.resources {
            if resource.kind.is_empty() || !has_concrete_digest(Some(&resource.identity)) {
                return Err(PlanValidationError::InvalidResourceIdentity);
            }
        }

        if self.intent.target != self.target_snapshot.profile.identity {
            return Err(PlanValidationError::TargetMismatch);
        }

        if self.steps.is_empty()
            || !self
                .steps
                .iter()
                .any(|step| step.kind == PlanStepKind::Verify)
        {
            return Err(PlanValidationError::MissingVerificationStep);
        }
        if self.steps.last().map(|step| &step.kind) != Some(&PlanStepKind::Verify) {
            return Err(PlanValidationError::VerificationStepNotFinal);
        }

        let has_state_change = self.steps.iter().any(|step| {
            matches!(
                step.kind,
                PlanStepKind::StageArtifacts
                    | PlanStepKind::ApplyDesiredState
                    | PlanStepKind::Reboot
                    | PlanStepKind::Rollback
            )
        });
        if has_state_change && self.max_target_snapshot_age_ms.is_none() {
            return Err(PlanValidationError::TargetSnapshotFreshnessRequired);
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
            validate_step_capability_semantics(step)?;
        }

        validate_capabilities(
            &self.intent.required_capabilities,
            &self.target_snapshot.profile.capabilities,
            None,
        )?;

        validate_artifacts(&self.intent.artifacts)?;
        validate_resource_binding(self)?;

        for resource in &self.intent.required_resources {
            if !self.target_snapshot.resources.contains(resource) {
                return Err(PlanValidationError::MissingTargetResource(resource.clone()));
            }
        }

        if self.verification.require_attestation
            && !self
                .target_snapshot
                .profile
                .capabilities
                .contains(&Capability::AttestState)
        {
            return Err(PlanValidationError::MissingTargetCapability(
                Capability::AttestState,
            ));
        }

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

/// Durable record of a successful one-shot authorization consumption.
///
/// SSC does not own the persistence or transaction boundary. A target executor
/// (or authority service immediately adjacent to it) must atomically reject a
/// previously consumed `(authority_id, nonce)` pair and record the exact
/// authorization digest that was consumed before performing any mutation.
///
/// A fresh authorization for the same plan is therefore distinguishable only
/// by a fresh authorization identity/nonce under the authority's policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorizationConsumption {
    pub authorization_digest: ContentDigest,
    pub authority_id: String,
    pub nonce: String,
    pub consumed_at_ms: u64,
}

impl AuthorizationConsumption {
    /// Derive the exact consumption record from a validated authorized plan.
    /// This constructor performs validation but no persistence and consumes nothing.
    pub fn for_authorized_plan(
        authorized: &AuthorizedDeploymentPlan,
        consumed_at_ms: u64,
    ) -> Result<Self, PlanValidationError> {
        authorized.validate(consumed_at_ms)?;

        Ok(Self {
            authorization_digest: authorized
                .authorization
                .digest()
                .map_err(PlanValidationError::Serialization)?,
            authority_id: authorized.authorization.authority_id.clone(),
            nonce: authorized.authorization.nonce.clone(),
            consumed_at_ms,
        })
    }
}

/// Opaque handoff proving that a durable authorization-consumption operation
/// has successfully completed for one exact authorized plan.
///
/// The handle borrows the authorized plan that was consumed. This prevents an
/// executor from accidentally pairing a valid consumption record with a
/// different authorized plan object.
///
/// It is deliberately non-cloneable and non-serializable. It is not a
/// cryptographic credential; it is a type-level sequencing token produced only
/// by `consume_authorized_once`.
#[derive(Debug)]
pub struct ConsumedAuthorization<'a> {
    authorized: &'a AuthorizedDeploymentPlan,
    authorization_digest: ContentDigest,
    consumed_at_ms: u64,
}

impl<'a> ConsumedAuthorization<'a> {
    pub fn authorized_plan(&self) -> &'a AuthorizedDeploymentPlan {
        self.authorized
    }

    pub fn authorization_digest(&self) -> &ContentDigest {
        &self.authorization_digest
    }

    pub fn authority_id(&self) -> &str {
        &self.authorized.authorization.authority_id
    }

    pub fn nonce(&self) -> &str {
        &self.authorized.authorization.nonce
    }

    pub fn consumed_at_ms(&self) -> u64 {
        self.consumed_at_ms
    }

    /// Admit this consumed authorization for execution at `now_ms`.
    ///
    /// Revalidation here closes the gap between durable nonce consumption and
    /// mutation start: an authorization that expires after consumption is
    /// safely burned rather than executed late.
    pub fn admit_execution(
        self,
        now_ms: u64,
    ) -> Result<ExecutionAuthorization<'a>, PlanValidationError> {
        if now_ms < self.consumed_at_ms {
            return Err(PlanValidationError::AuthorizationConsumptionFromFuture {
                consumed_at_ms: self.consumed_at_ms,
                now_ms,
            });
        }
        self.authorized.validate(now_ms)?;
        Ok(ExecutionAuthorization {
            authorized: self.authorized,
            authorization_digest: self.authorization_digest,
            consumed_at_ms: self.consumed_at_ms,
            started_at_ms: now_ms,
        })
    }
}
/// Opaque execution admission for one exact, already-consumed authorization.
///
/// This handle carries the validated execution-start timestamp and exact
/// authorization digest recorded by the durable consumption boundary. It is
/// deliberately non-cloneable and non-serializable.
#[derive(Debug)]
pub struct ExecutionAuthorization<'a> {
    authorized: &'a AuthorizedDeploymentPlan,
    authorization_digest: ContentDigest,
    consumed_at_ms: u64,
    started_at_ms: u64,
}

impl<'a> ExecutionAuthorization<'a> {
    pub fn authorized_plan(&self) -> &'a AuthorizedDeploymentPlan {
        self.authorized
    }

    pub fn authorization_digest(&self) -> &ContentDigest {
        &self.authorization_digest
    }

    pub fn consumed_at_ms(&self) -> u64 {
        self.consumed_at_ms
    }

    pub fn started_at_ms(&self) -> u64 {
        self.started_at_ms
    }
}
/// Persistence/transaction boundary for one-shot authorization.
///
/// Implementations MUST make the check-and-record operation atomic with
/// respect to concurrent consumers. The uniqueness key is the authority's
/// nonce identity; the stored value must remain bound to the exact
/// authorization digest. An already-consumed nonce MUST never be silently
/// accepted, even when the presented authorization digest differs.
pub trait AuthorizationConsumptionStore {
    type Error: std::error::Error + Send + Sync + 'static;

    fn consume_once(&mut self, consumption: &AuthorizationConsumption) -> Result<(), Self::Error>;
}

#[derive(Debug, Error)]
pub enum AuthorizationConsumptionError<E: std::error::Error + Send + Sync + 'static> {
    #[error("authorization is invalid for consumption: {0}")]
    Validation(#[from] PlanValidationError),
    #[error("durable authorization consumption failed: {0}")]
    Store(E),
}

/// Validate an authorized plan, atomically consume its authorization nonce, and
/// mint the one-shot execution handoff.
///
/// The store is the durable replay boundary. SSC remains stateless: this helper
/// performs no implicit persistence beyond the store implementation supplied by
/// the caller.
pub fn consume_authorized_once<'a, S: AuthorizationConsumptionStore>(
    store: &mut S,
    authorized: &'a AuthorizedDeploymentPlan,
    consumed_at_ms: u64,
) -> Result<ConsumedAuthorization<'a>, AuthorizationConsumptionError<S::Error>> {
    authorized.validate(consumed_at_ms)?;
    let consumption =
        AuthorizationConsumption::for_authorized_plan(authorized, consumed_at_ms)?;
    store
        .consume_once(&consumption)
        .map_err(AuthorizationConsumptionError::Store)?;
    Ok(ConsumedAuthorization {
        authorized,
        authorization_digest: consumption.authorization_digest,
        consumed_at_ms,
    })
}

/// Result of applying an authorized deployment plan.
///
/// The executor/adapter may attach platform-specific evidence separately;
/// this enum intentionally describes only the protocol-level lifecycle outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionOutcome {
    Succeeded,
    Failed,
    RolledBack,
    Recovered,
}

/// Status of the required postcondition verification.
///
/// Mechanical execution and postcondition verification are intentionally
/// independent: an executor may complete without being able to prove that the
/// requested target state was established.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PostconditionOutcome {
    Satisfied,
    Violated,
    Unproven,
    NotEvaluated,
    Inapplicable,
}

/// Append-only receipt emitted by a target-specific executor.
///
/// A receipt is bound to the exact authorized plan and target snapshot. It is
/// evidence of what the executor reports, not a substitute for independent
/// verification of the underlying target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionReceipt {
    pub schema_version: String,
    pub plan_digest: ContentDigest,
    pub target_snapshot_digest: ContentDigest,
    /// Digest of the exact authorization envelope consumed for this execution.
    pub authorization_digest: ContentDigest,
    /// Digest of the target observation captured after execution.
    ///
    /// This may legitimately differ from the pre-execution snapshot digest
    /// because the deployment is expected to change state.
    pub final_target_snapshot_digest: ContentDigest,
    /// Observation time of the final target snapshot claimed by this receipt.
    /// It must fall within the execution interval so stale evidence cannot be
    /// presented as post-execution proof.
    pub final_target_snapshot_observed_at_ms: u64,
    pub started_at_ms: u64,
    pub finished_at_ms: u64,
    /// Mechanical execution result. This does not imply that the target
    /// postcondition was proven.
    pub outcome: ExecutionOutcome,
    /// Transition disposition observed after execution.
    ///
    /// This must match the disposition authorized by the compiled plan; a
    /// receipt cannot silently reinterpret an authorized transition.
    pub observed_disposition: DeploymentDisposition,
    /// Independent status of the required postcondition verification.
    pub postcondition: PostconditionOutcome,
    pub verification_digest: Option<ContentDigest>,
    pub evidence: Vec<AttestationRef>,
}

impl ExecutionReceipt {
    pub fn validate_for(
        &self,
        plan: &AuthorizedDeploymentPlan,
    ) -> Result<(), ReceiptValidationError> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(ReceiptValidationError::UnsupportedSchemaVersion(
                self.schema_version.clone(),
            ));
        }

        let plan_digest = plan
            .plan
            .digest()
            .map_err(ReceiptValidationError::Serialization)?;
        if self.plan_digest != plan_digest {
            return Err(ReceiptValidationError::PlanDigestMismatch);
        }

        if self.target_snapshot_digest != plan.authorization.target_snapshot_digest {
            return Err(ReceiptValidationError::TargetSnapshotDigestMismatch);
        }

        let authorization_digest = plan
            .authorization
            .digest()
            .map_err(ReceiptValidationError::Serialization)?;
        if self.authorization_digest != authorization_digest {
            return Err(ReceiptValidationError::AuthorizationDigestMismatch);
        }

        if self.observed_disposition != plan.plan.verification.disposition {
            return Err(ReceiptValidationError::ObservedDispositionMismatch);
        }

        if !has_concrete_digest(Some(&self.final_target_snapshot_digest)) {
            return Err(ReceiptValidationError::MissingFinalSnapshotEvidence);
        }

        if self.finished_at_ms < self.started_at_ms {
            return Err(ReceiptValidationError::TimestampOrderInvalid);
        }
        if self.final_target_snapshot_observed_at_ms < self.started_at_ms
            || self.final_target_snapshot_observed_at_ms > self.finished_at_ms
        {
            return Err(ReceiptValidationError::FinalSnapshotObservationOutsideExecution);
        }
        if self.started_at_ms < plan.plan.target_snapshot.observed_at_ms {
            return Err(ReceiptValidationError::StartedBeforeTargetSnapshot);
        }
        if let Some(max_age_ms) = plan.plan.max_target_snapshot_age_ms {
            let age_ms = self.started_at_ms - plan.plan.target_snapshot.observed_at_ms;
            if age_ms > max_age_ms {
                return Err(ReceiptValidationError::TargetSnapshotStaleAtExecution {
                    age_ms,
                    max_age_ms,
                });
            }
        }
        if self.started_at_ms
            < plan
                .authorization
                .valid_from_ms
                .unwrap_or(self.started_at_ms)
        {
            return Err(ReceiptValidationError::StartedBeforeAuthorization);
        }

        match plan
            .authorization
            .validate_for(&plan.plan, self.started_at_ms)
        {
            Ok(()) => {}
            Err(PlanValidationError::AuthorizationNotYetValid) => {
                return Err(ReceiptValidationError::StartedBeforeAuthorization);
            }
            Err(PlanValidationError::AuthorizationExpired) => {
                return Err(ReceiptValidationError::FinishedAfterAuthorizationExpiry);
            }
            Err(PlanValidationError::AuthorizationMissingExpiry) => {
                return Err(ReceiptValidationError::AuthorizationMissingExpiry);
            }
            Err(PlanValidationError::TargetSnapshotStale {
                age_ms,
                max_age_ms,
            }) => {
                return Err(ReceiptValidationError::TargetSnapshotStaleAtExecution {
                    age_ms,
                    max_age_ms,
                });
            }
            Err(error) => return Err(ReceiptValidationError::InvalidAuthorization(error)),
        }
        if let Some(valid_until_ms) = plan.authorization.valid_until_ms
            && self.finished_at_ms > valid_until_ms
        {
            return Err(ReceiptValidationError::FinishedAfterAuthorizationExpiry);
        }

        if matches!(
            self.postcondition,
            PostconditionOutcome::Satisfied | PostconditionOutcome::Violated
        ) && !has_concrete_digest(self.verification_digest.as_ref())
        {
            return Err(ReceiptValidationError::MissingVerificationEvidence);
        }

        for evidence in &self.evidence {
            if evidence.media_type.is_empty()
                || evidence.uri.is_empty()
                || !has_concrete_digest(Some(&evidence.digest))
            {
                return Err(ReceiptValidationError::InvalidAttestationEvidence);
            }
        }

        if plan.plan.verification.require_attestation && self.evidence.is_empty() {
            return Err(ReceiptValidationError::MissingAttestationEvidence);
        }

        Ok(())
    }

    /// Whether the receipt represents both mechanical success/recovery and
    /// satisfied postconditions.
    pub fn is_verified_success(&self) -> bool {
        self.outcome == ExecutionOutcome::Succeeded
            && self.postcondition == PostconditionOutcome::Satisfied
    }
}

/// Target-specific execution boundary.
///
/// The neutral crate defines the handoff and receipt contract but never
/// chooses a transport, process API, shell, operating system, or privilege
/// mechanism.
pub trait DeploymentExecutor {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Execute the exact authorized plan after durable consumption and
    /// execution-time validity admission.
    ///
    /// `ExecutionAuthorization` owns a borrow of the exact plan and can only
    /// be minted by `ConsumedAuthorization::admit_execution`, which revalidates
    /// authorization at the execution-start timestamp.
    fn execute(
        &mut self,
        execution_authorization: ExecutionAuthorization<'_>,
    ) -> Result<ExecutionReceipt, Self::Error>;
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ReceiptValidationError {
    #[error("unsupported Sovereign State Compiler schema version: {0}")]
    UnsupportedSchemaVersion(String),
    #[error("execution receipt plan digest does not match authorized plan")]
    PlanDigestMismatch,
    #[error("execution receipt target snapshot digest does not match authorization")]
    TargetSnapshotDigestMismatch,
    #[error("execution receipt authorization digest does not match the exact authorization envelope")]
    AuthorizationDigestMismatch,
    #[error("execution receipt observed disposition does not match the authorized plan")]
    ObservedDispositionMismatch,
    #[error("execution receipt carries no concrete final target snapshot digest")]
    MissingFinalSnapshotEvidence,
    #[error("execution receipt references an invalid authorized plan: {0}")]
    InvalidAuthorization(PlanValidationError),
    #[error("execution receipt references authorization without an explicit expiry")]
    AuthorizationMissingExpiry,
    #[error("execution receipt timestamps are out of order")]
    TimestampOrderInvalid,
    #[error("execution receipt final snapshot observation falls outside the execution interval")]
    FinalSnapshotObservationOutsideExecution,
    #[error("execution started before the authorized target snapshot was observed")]
    StartedBeforeTargetSnapshot,
    #[error("execution started with a target snapshot older than the plan permits")]
    TargetSnapshotStaleAtExecution { age_ms: u64, max_age_ms: u64 },
    #[error("execution started before authorization became valid")]
    StartedBeforeAuthorization,
    #[error("execution finished after authorization expired")]
    FinishedAfterAuthorizationExpiry,
    #[error(
        "postcondition claims a verified result but carries no concrete verification evidence digest"
    )]
    MissingVerificationEvidence,
    #[error("required attestation verification carries no attestation evidence")]
    MissingAttestationEvidence,
    #[error("execution receipt carries incomplete attestation evidence")]
    InvalidAttestationEvidence,
    #[error("canonical serialization failed: {0}")]
    Serialization(serde_json::Error),
}

impl AuthorizedDeploymentPlan {
    pub fn validate(&self, now_ms: u64) -> Result<(), PlanValidationError> {
        self.plan.validate()?;
        self.authorization.validate_for(&self.plan, now_ms)
    }
}

impl AuthorizationEvidence {
    pub fn digest(&self) -> Result<ContentDigest, serde_json::Error> {
        canonical_digest(self, AUTHORIZATION_DIGEST_DOMAIN)
    }

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

        if plan.target_snapshot.observed_at_ms > now_ms {
            return Err(PlanValidationError::TargetSnapshotFutureDated {
                observed_at_ms: plan.target_snapshot.observed_at_ms,
                now_ms,
            });
        }

        if let Some(max_age_ms) = plan.max_target_snapshot_age_ms {
            let age_ms = now_ms - plan.target_snapshot.observed_at_ms;
            if age_ms > max_age_ms {
                return Err(PlanValidationError::TargetSnapshotStale { age_ms, max_age_ms });
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

        if let Some(intent_expiry) = plan.intent.expires_at_ms {
            if now_ms > intent_expiry {
                return Err(PlanValidationError::IntentExpired);
            }
            if self
                .valid_until_ms
                .is_some_and(|until| until > intent_expiry)
            {
                return Err(PlanValidationError::AuthorizationExceedsIntentExpiry);
            }
        }

        if let (Some(from), Some(until)) = (self.valid_from_ms, self.valid_until_ms)
            && from > until
        {
            return Err(PlanValidationError::AuthorizationWindowInvalid);
        }
        let Some(valid_until_ms) = self.valid_until_ms else {
            return Err(PlanValidationError::AuthorizationMissingExpiry);
        };
        if now_ms > valid_until_ms {
            return Err(PlanValidationError::AuthorizationExpired);
        }
        if self.valid_from_ms.is_some_and(|from| now_ms < from) {
            return Err(PlanValidationError::AuthorizationNotYetValid);
        }

        for capability in &self.granted_capabilities {
            if !plan
                .target_snapshot
                .profile
                .capabilities
                .contains(capability)
            {
                return Err(PlanValidationError::GrantedCapabilityNotSupported(
                    *capability,
                ));
            }
        }

        validate_capabilities(
            &plan.intent.required_capabilities,
            &plan.target_snapshot.profile.capabilities,
            Some(&self.granted_capabilities),
        )?;

        for resource in &plan.intent.required_resources {
            if !plan.target_snapshot.resources.contains(resource) {
                return Err(PlanValidationError::MissingTargetResource(resource.clone()));
            }
        }

        for step in &plan.steps {
            validate_capabilities(
                &step.required_capabilities,
                &plan.target_snapshot.profile.capabilities,
                Some(&self.granted_capabilities),
            )?;
        }

        let mut required_authority = plan.intent.required_capabilities.clone();
        for step in &plan.steps {
            required_authority.extend(step.required_capabilities.iter().copied());
        }
        if plan.rollback.allowed {
            required_authority.insert(Capability::Rollback);
        }
        if plan.verification.require_attestation {
            required_authority.insert(Capability::AttestState);
        }

        for capability in &self.granted_capabilities {
            if !required_authority.contains(capability) {
                return Err(PlanValidationError::UnneededGrantedCapability(*capability));
            }
        }

        Ok(())
    }
}

fn validate_artifacts(artifacts: &[ArtifactRef]) -> Result<(), PlanValidationError> {
    for artifact in artifacts {
        if artifact.id.0.is_empty() {
            return Err(PlanValidationError::InvalidArtifactIdentity(
                artifact.id.clone(),
            ));
        }
        if artifact.version.as_deref().is_some_and(str::is_empty) {
            return Err(PlanValidationError::InvalidArtifactVersion(
                artifact.id.clone(),
            ));
        }
        if artifact.digest.algorithm.is_empty() || artifact.digest.value.is_empty() {
            return Err(PlanValidationError::InvalidArtifactDigest(
                artifact.id.clone(),
            ));
        }
        for attestation in &artifact.provenance {
            if attestation.media_type.is_empty()
                || attestation.uri.is_empty()
                || attestation.digest.algorithm.is_empty()
                || attestation.digest.value.is_empty()
            {
                return Err(PlanValidationError::InvalidAttestationReference(
                    artifact.id.clone(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_step_capability_semantics(step: &PlanStep) -> Result<(), PlanValidationError> {
    match step.kind {
        PlanStepKind::Observe | PlanStepKind::Verify => {
            if !step.required_capabilities.contains(&Capability::ObserveState) {
                return Err(PlanValidationError::StepMissingSemanticCapability {
                    kind: step.kind.clone(),
                    capability: Capability::ObserveState,
                });
            }
        }
        PlanStepKind::Reboot => {
            if !step.required_capabilities.contains(&Capability::Reboot) {
                return Err(PlanValidationError::StepMissingSemanticCapability {
                    kind: step.kind.clone(),
                    capability: Capability::Reboot,
                });
            }
        }
        PlanStepKind::Rollback => {
            if !step.required_capabilities.contains(&Capability::Rollback) {
                return Err(PlanValidationError::StepMissingSemanticCapability {
                    kind: step.kind.clone(),
                    capability: Capability::Rollback,
                });
            }
        }
        PlanStepKind::StageArtifacts | PlanStepKind::ApplyDesiredState => {
            let has_effect_capability = step.required_capabilities.iter().any(|capability| {
                !matches!(
                    capability,
                    Capability::ObserveState
                        | Capability::ObserveHardware
                        | Capability::AttestState
                )
            });
            if !has_effect_capability {
                return Err(PlanValidationError::StepMissingMutationCapability(
                    step.kind.clone(),
                ));
            }
        }
    }

    Ok(())
}

fn validate_resource_binding(plan: &DeploymentPlan) -> Result<(), PlanValidationError> {
    if plan.intent.required_resources.is_empty() {
        let mut required_capabilities = plan.intent.required_capabilities.clone();
        for step in &plan.steps {
            required_capabilities.extend(step.required_capabilities.iter().copied());
        }

        for capability in [
            Capability::InstallApplication,
            Capability::RemoveApplication,
            Capability::ReplaceOs,
            Capability::ConfigureSecureBoot,
            Capability::EncryptStorage,
        ] {
            if required_capabilities.contains(&capability) {
                return Err(PlanValidationError::ResourceBindingRequired(capability));
            }
        }
    }

    Ok(())
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

fn has_concrete_digest(digest: Option<&ContentDigest>) -> bool {
    digest.is_some_and(|value| !value.algorithm.is_empty() && !value.value.is_empty())
}

/// The minimal adapter contract for target-specific lowering.
///
/// Compilation is deliberately authorization-free. Authorization happens
/// after a concrete target adapter has produced the exact plan to execute.
///
/// No method here accepts a shell string or raw command line.
pub trait TargetAdapter {
    type Error: std::error::Error + Send + Sync + 'static;

    fn describe_target(&self) -> Result<TargetSnapshot, Self::Error>;

    fn compile(&self, intent: &DeploymentIntent) -> Result<DeploymentPlan, Self::Error>;
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PlanValidationError {
    #[error("intent target does not match target profile")]
    TargetMismatch,
    #[error("deployment intent has an empty identity")]
    InvalidIntentIdentity,
    #[error("target profile has an empty identity or platform")]
    InvalidTargetIdentity,
    #[error("target snapshot observation digest is empty or incomplete")]
    InvalidObservationDigest,
    #[error("target resource has an empty kind or incomplete identity")]
    InvalidResourceIdentity,
    #[error("required capability {0:?} is not supported by target")]
    MissingTargetCapability(Capability),
    #[error("required capability {0:?} is not granted")]
    MissingGrantedCapability(Capability),
    #[error("resource-bound capability {0:?} requires at least one explicit target resource")]
    ResourceBindingRequired(Capability),
    #[error("required target resource is absent from the observed snapshot")]
    MissingTargetResource(ResourceRef),
    #[error("plan step sequence is not contiguous from zero")]
    NonContiguousPlanSequence,
    #[error("plan must contain a verification step")]
    MissingVerificationStep,
    #[error("verification step must be the final lifecycle step")]
    VerificationStepNotFinal,
    #[error("state-changing plans must declare a maximum target-snapshot age")]
    TargetSnapshotFreshnessRequired,
    #[error("plan step {kind:?} must explicitly require semantic capability {capability:?}")]
    StepMissingSemanticCapability {
        kind: PlanStepKind,
        capability: Capability,
    },
    #[error("mutating plan step {0:?} must explicitly require at least one effect capability")]
    StepMissingMutationCapability(PlanStepKind),
    #[error("plan step sequence overflowed")]
    SequenceOverflow,
    #[error("rollback attempts are configured without rollback permission")]
    RollbackAttemptsWithoutPermission,
    #[error("artifact has an empty identity")]
    InvalidArtifactIdentity(ArtifactId),
    #[error("artifact has an empty version string")]
    InvalidArtifactVersion(ArtifactId),
    #[error("artifact has an empty or incomplete content digest")]
    InvalidArtifactDigest(ArtifactId),
    #[error("artifact attestation reference is incomplete")]
    InvalidAttestationReference(ArtifactId),
    #[error("authorization intent digest does not match the compiled intent")]
    AuthorizationIntentDigestMismatch,
    #[error("authorization target-profile digest does not match the compiled target")]
    AuthorizationTargetDigestMismatch,
    #[error("authorization plan digest does not match the compiled plan")]
    AuthorizationPlanDigestMismatch,
    #[error("authorization target-snapshot digest does not match the observed target")]
    AuthorizationTargetSnapshotDigestMismatch,
    #[error(
        "target snapshot is future-dated: observed at {observed_at_ms} ms but authorization time is {now_ms} ms"
    )]
    TargetSnapshotFutureDated { observed_at_ms: u64, now_ms: u64 },
    #[error("target snapshot is stale: age {age_ms} ms exceeds maximum {max_age_ms} ms")]
    TargetSnapshotStale { age_ms: u64, max_age_ms: u64 },
    #[error("authorization validity window is invalid")]
    AuthorizationWindowInvalid,
    #[error("deployment intent has expired")]
    IntentExpired,
    #[error("authorization outlives deployment intent expiry")]
    AuthorizationExceedsIntentExpiry,
    #[error("authorization authority identifier is empty")]
    EmptyAuthority,
    #[error("authorization nonce is empty")]
    EmptyNonce,
    #[error("authorization grants a capability unsupported by target")]
    GrantedCapabilityNotSupported(Capability),
    #[error("authorization grants a capability not required by the compiled plan")]
    UnneededGrantedCapability(Capability),
    #[error("authorization is not yet valid")]
    AuthorizationNotYetValid,
    #[error("authorization has expired")]
    AuthorizationExpired,
    #[error("authorization was consumed at {consumed_at_ms} ms but execution time is {now_ms} ms")]
    AuthorizationConsumptionFromFuture { consumed_at_ms: u64, now_ms: u64 },
    #[error("authorization must contain an explicit expiry")]
    AuthorizationMissingExpiry,
    #[error("unsupported Sovereign State Compiler schema version: {0}")]
    UnsupportedSchemaVersion(String),
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
                Capability::ObserveState,
                Capability::ObserveState,
                Capability::ObserveHardware,
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
        intent
            .required_capabilities
            .insert(Capability::ConfigureSystem);
        intent.required_capabilities.insert(Capability::Rollback);
        intent.required_resources.insert(ResourceRef {
            kind: "block-device".into(),
            identity: ContentDigest::blake3(b"disk-serial-123"),
        });

        DeploymentPlan {
            schema_version: SCHEMA_VERSION.into(),
            intent,
            target_snapshot: TargetSnapshot {
                profile: sample_profile(),
                observed_at_ms: 90,
                observation_digest: ContentDigest::blake3(b"hardware-observation"),
                resources: [ResourceRef {
                    kind: "block-device".into(),
                    identity: ContentDigest::blake3(b"disk-serial-123"),
                }]
                .into_iter()
                .collect(),
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
                PlanStep {
                    sequence: 2,
                    kind: PlanStepKind::Verify,
                    required_capabilities: [Capability::ObserveState].into_iter().collect(),
                    description: "verify target state".into(),
                },
            ],
            verification: VerificationPolicy {
                expected_state: DesiredState::default(),
                disposition: DeploymentDisposition::Applied,
                require_attestation: false,
            },
            rollback: RollbackPolicy {
                allowed: true,
                max_attempts: 1,
            },
        }
    }

    #[derive(Debug, Default)]
    struct TestConsumptionStore {
        consumed: BTreeSet<(String, String)>,
    }

    #[derive(Debug, Error, PartialEq, Eq)]
    #[error("already consumed")]
    struct TestConsumptionError;

    impl AuthorizationConsumptionStore for TestConsumptionStore {
        type Error = TestConsumptionError;

        fn consume_once(
            &mut self,
            consumption: &AuthorizationConsumption,
        ) -> Result<(), Self::Error> {
            if !self
                .consumed
                .insert((consumption.authority_id.clone(), consumption.nonce.clone()))
            {
                return Err(TestConsumptionError);
            }
            Ok(())
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
                Capability::ObserveState,
                Capability::ObserveHardware,
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
    fn rejects_empty_deployment_identity() {
        let mut plan = sample_plan();
        plan.intent.intent_id.clear();
        assert_eq!(
            plan.validate(),
            Err(PlanValidationError::InvalidIntentIdentity)
        );
    }

    #[test]
    fn rejects_empty_target_observation_identity() {
        let mut plan = sample_plan();
        plan.target_snapshot.observation_digest = ContentDigest {
            algorithm: String::new(),
            value: String::new(),
        };
        assert_eq!(
            plan.validate(),
            Err(PlanValidationError::InvalidObservationDigest)
        );
    }

    #[test]
    fn rejects_future_dated_target_snapshot() {
        let mut plan = sample_plan();
        plan.target_snapshot.observed_at_ms = 151;
        let auth = authorization_for(&plan);

        assert_eq!(
            plan.authorize(auth, 150),
            Err(PlanValidationError::TargetSnapshotFutureDated {
                observed_at_ms: 151,
                now_ms: 150,
            })
        );
    }

    #[test]
    fn rejects_observe_or_verify_steps_without_observe_state() {
        let mut observe_plan = sample_plan();
        observe_plan.steps[0].kind = PlanStepKind::Observe;
        observe_plan.steps[0].required_capabilities.clear();
        assert_eq!(
            observe_plan.validate(),
            Err(PlanValidationError::StepMissingSemanticCapability {
                kind: PlanStepKind::Observe,
                capability: Capability::ObserveState,
            })
        );

        let mut verify_plan = sample_plan();
        verify_plan.steps[2].required_capabilities.clear();
        assert_eq!(
            verify_plan.validate(),
            Err(PlanValidationError::StepMissingSemanticCapability {
                kind: PlanStepKind::Verify,
                capability: Capability::ObserveState,
            })
        );
    }

    #[test]
    fn rejects_empty_observed_resource_identity() {
        let mut plan = sample_plan();
        let mut resource = plan
            .target_snapshot
            .resources
            .iter()
            .next()
            .expect("resource")
            .clone();
        resource.identity = ContentDigest {
            algorithm: String::new(),
            value: String::new(),
        };
        plan.target_snapshot.resources.clear();
        plan.target_snapshot.resources.insert(resource);
        assert_eq!(
            plan.validate(),
            Err(PlanValidationError::InvalidResourceIdentity)
        );
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
            Err(PlanValidationError::AuthorizationTargetDigestMismatch)
        );
    }

    #[test]
    fn boot_chain_capability_can_be_target_scoped_without_resource_binding() {
        let mut plan = sample_plan();
        plan.intent.required_resources.clear();
        plan.intent.required_capabilities.remove(&Capability::Rollback);
        plan.intent
            .required_capabilities
            .insert(Capability::ModifyBootChain);
        plan.target_snapshot
            .profile
            .capabilities
            .insert(Capability::ModifyBootChain);
        plan.steps[0].required_capabilities.clear();
        plan.steps[0]
            .required_capabilities
            .insert(Capability::ObserveState);
        plan.steps[1].required_capabilities.clear();
        plan.steps[1]
            .required_capabilities
            .insert(Capability::ModifyBootChain);

        assert!(plan.validate().is_ok());
    }

    #[test]
    fn rejects_resource_bound_capability_without_resource_binding() {
        let mut plan = sample_plan();
        plan.intent.required_resources.clear();
        plan.target_snapshot.resources.clear();
        plan.target_snapshot
            .profile
            .capabilities
            .insert(Capability::EncryptStorage);
        plan.intent
            .required_capabilities
            .insert(Capability::EncryptStorage);
        plan.steps[1]
            .required_capabilities
            .insert(Capability::EncryptStorage);

        assert_eq!(
            plan.validate(),
            Err(PlanValidationError::ResourceBindingRequired(
                Capability::EncryptStorage
            ))
        );
    }

    #[test]
    fn rejects_missing_observed_resource() {
        let mut plan = sample_plan();
        plan.target_snapshot.resources.clear();

        assert_eq!(
            plan.validate(),
            Err(PlanValidationError::MissingTargetResource(
                plan.intent
                    .required_resources
                    .iter()
                    .next()
                    .expect("required resource")
                    .clone()
            ))
        );
    }

    #[test]
    fn rejects_artifact_without_identity() {
        let mut plan = sample_plan();
        plan.intent.artifacts.push(ArtifactRef {
            id: ArtifactId::from(""),
            version: Some("1.0.0".into()),
            digest: ContentDigest::blake3(b"artifact"),
            provenance: Vec::new(),
        });

        assert_eq!(
            plan.validate(),
            Err(PlanValidationError::InvalidArtifactIdentity(
                ArtifactId::from("")
            ))
        );
    }

    #[test]
    fn rejects_artifact_with_empty_version() {
        let mut plan = sample_plan();
        plan.intent.artifacts.push(ArtifactRef {
            id: ArtifactId::from("empty-version"),
            version: Some(String::new()),
            digest: ContentDigest::blake3(b"artifact"),
            provenance: Vec::new(),
        });

        assert_eq!(
            plan.validate(),
            Err(PlanValidationError::InvalidArtifactVersion(
                ArtifactId::from("empty-version")
            ))
        );
    }

    #[test]
    fn rejects_artifact_without_digest() {
        let mut plan = sample_plan();
        plan.intent.artifacts.push(ArtifactRef {
            id: ArtifactId::from("missing-digest"),
            version: Some("1.0.0".into()),
            digest: ContentDigest {
                algorithm: String::new(),
                value: String::new(),
            },
            provenance: Vec::new(),
        });

        assert_eq!(
            plan.validate(),
            Err(PlanValidationError::InvalidArtifactDigest(
                ArtifactId::from("missing-digest")
            ))
        );
    }

    #[test]
    fn rejects_incomplete_attestation_reference() {
        let mut plan = sample_plan();
        plan.intent.artifacts.push(ArtifactRef {
            id: ArtifactId::from("attested"),
            version: Some("1.0.0".into()),
            digest: ContentDigest::blake3(b"artifact"),
            provenance: vec![AttestationRef {
                media_type: "application/json".into(),
                uri: String::new(),
                digest: ContentDigest::blake3(b"attestation"),
            }],
        });

        assert_eq!(
            plan.validate(),
            Err(PlanValidationError::InvalidAttestationReference(
                ArtifactId::from("attested")
            ))
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
    fn rejects_authorization_beyond_intent_expiry() {
        let mut plan = sample_plan();
        plan.intent.expires_at_ms = Some(180);
        let mut auth = authorization_for(&plan);
        auth.valid_until_ms = Some(200);

        assert_eq!(
            plan.authorize(auth, 150),
            Err(PlanValidationError::AuthorizationExceedsIntentExpiry)
        );
    }

    #[test]
    fn rejects_expired_intent() {
        let mut plan = sample_plan();
        plan.intent.expires_at_ms = Some(149);
        let auth = authorization_for(&plan);

        assert_eq!(
            plan.authorize(auth, 150),
            Err(PlanValidationError::IntentExpired)
        );
    }

    #[test]
    fn rejects_authorization_without_expiry() {
        let plan = sample_plan();
        let mut auth = authorization_for(&plan);
        auth.valid_until_ms = None;

        assert_eq!(
            plan.authorize(auth, 150),
            Err(PlanValidationError::AuthorizationMissingExpiry)
        );
    }

    #[test]
    fn rejects_previous_schema_version_after_v0_4_change() {
        let mut plan = sample_plan();
        plan.schema_version = "ssc/v0.2".into();

        assert_eq!(
            plan.validate(),
            Err(PlanValidationError::UnsupportedSchemaVersion(
                "ssc/v0.2".into()
            ))
        );
    }

    #[test]
    fn rejects_previous_receipt_schema_version_after_v0_4_change() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth, 150).expect("authorized plan");

        let receipt = ExecutionReceipt {
            schema_version: "ssc/v0.3".into(),
            plan_digest: authorized.plan.digest().expect("plan digest"),
            target_snapshot_digest: authorized.authorization.target_snapshot_digest.clone(),
            authorization_digest: authorized.authorization.digest().expect("authorization digest"),
            final_target_snapshot_digest: ContentDigest::blake3(b"after-execution"),
            final_target_snapshot_observed_at_ms: 160,
            observed_disposition: authorized.plan.verification.disposition,
            started_at_ms: 151,
            finished_at_ms: 160,
            outcome: ExecutionOutcome::Succeeded,
            postcondition: PostconditionOutcome::Satisfied,
            verification_digest: Some(ContentDigest::blake3(b"verified-postcondition")),
            evidence: Vec::new(),
        };

        assert_eq!(
            receipt.validate_for(&authorized),
            Err(ReceiptValidationError::UnsupportedSchemaVersion(
                "ssc/v0.3".into()
            ))
        );
    }

    #[test]
    fn rejects_unsupported_plan_schema_version() {
        let mut plan = sample_plan();
        plan.schema_version = "ssc/v999".into();

        assert_eq!(
            plan.validate(),
            Err(PlanValidationError::UnsupportedSchemaVersion(
                "ssc/v999".into()
            ))
        );
    }

    #[test]
    fn rejects_unsupported_intent_schema_version() {
        let mut plan = sample_plan();
        plan.intent.schema_version = "ssc/v999".into();

        assert_eq!(
            plan.validate(),
            Err(PlanValidationError::UnsupportedSchemaVersion(
                "ssc/v999".into()
            ))
        );
    }

    #[test]
    fn rejects_authorization_before_valid_from() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);

        assert_eq!(
            plan.authorize(auth, 99),
            Err(PlanValidationError::AuthorizationNotYetValid)
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
    fn rejects_unneeded_granted_capability() {
        let plan = sample_plan();
        let mut auth = authorization_for(&plan);
        auth.granted_capabilities
            .insert(Capability::InstallApplication);

        assert_eq!(
            plan.authorize(auth, 150),
            Err(PlanValidationError::UnneededGrantedCapability(
                Capability::InstallApplication
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
    fn rejects_plan_without_verification_step() {
        let mut plan = sample_plan();
        plan.steps.retain(|step| step.kind != PlanStepKind::Verify);
        assert_eq!(
            plan.validate(),
            Err(PlanValidationError::MissingVerificationStep)
        );
    }

    #[test]
    fn rejects_non_final_verification_step() {
        let mut plan = sample_plan();
        let verify = plan.steps.pop().expect("verify step");
        plan.steps.insert(0, verify);
        for (sequence, step) in plan.steps.iter_mut().enumerate() {
            step.sequence = sequence as u32;
        }
        assert_eq!(
            plan.validate(),
            Err(PlanValidationError::VerificationStepNotFinal)
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
    fn rejects_mutating_step_without_effect_capability() {
        let mut plan = sample_plan();
        plan.steps[1].required_capabilities.clear();

        assert_eq!(
            plan.validate(),
            Err(PlanValidationError::StepMissingMutationCapability(
                PlanStepKind::ApplyDesiredState
            ))
        );
    }

    #[test]
    fn rejects_mutating_step_with_observation_only_capabilities() {
        let mut plan = sample_plan();
        plan.steps[1].required_capabilities =
            [Capability::ObserveState, Capability::AttestState].into_iter().collect();

        assert_eq!(
            plan.validate(),
            Err(PlanValidationError::StepMissingMutationCapability(
                PlanStepKind::ApplyDesiredState
            ))
        );
    }

    #[test]
    fn accepts_mutating_step_with_effect_capability() {
        let plan = sample_plan();
        assert!(plan.validate().is_ok());
    }

    #[test]
    fn rejects_state_change_without_snapshot_freshness_window() {
        let mut plan = sample_plan();
        plan.max_target_snapshot_age_ms = None;

        assert_eq!(
            plan.validate(),
            Err(PlanValidationError::TargetSnapshotFreshnessRequired)
        );
    }

    #[test]
    fn accepts_read_only_plan_without_snapshot_freshness_window() {
        let mut plan = sample_plan();
        plan.intent.required_capabilities.clear();
        plan.intent.required_resources.clear();
        plan.steps = vec![
            PlanStep {
                sequence: 0,
                kind: PlanStepKind::Observe,
                required_capabilities: [Capability::ObserveState].into_iter().collect(),
                description: "observe target".into(),
            },
            PlanStep {
                sequence: 1,
                kind: PlanStepKind::Verify,
                required_capabilities: [Capability::ObserveState].into_iter().collect(),
                description: "verify observation".into(),
            },
        ];
        plan.verification = VerificationPolicy::new(
            DesiredState::default(),
            DeploymentDisposition::Unchanged,
            false,
        );
        plan.rollback = RollbackPolicy::default();
        plan.max_target_snapshot_age_ms = None;

        assert!(plan.validate().is_ok());
    }

    #[test]
    fn reboot_also_requires_snapshot_freshness_window() {
        let mut plan = sample_plan();
        plan.steps = vec![
            PlanStep {
                sequence: 0,
                kind: PlanStepKind::Reboot,
                required_capabilities: [Capability::Reboot].into_iter().collect(),
                description: "reboot target".into(),
            },
            PlanStep {
                sequence: 1,
                kind: PlanStepKind::Verify,
                required_capabilities: [Capability::ObserveState].into_iter().collect(),
                description: "verify target".into(),
            },
        ];
        plan.intent.required_capabilities =
            [Capability::Reboot].into_iter().collect();
        plan.intent.required_resources.clear();
        plan.target_snapshot.profile.capabilities.insert(Capability::Reboot);
        plan.verification.disposition = DeploymentDisposition::Rebooted;
        plan.max_target_snapshot_age_ms = None;

        assert_eq!(
            plan.validate(),
            Err(PlanValidationError::TargetSnapshotFreshnessRequired)
        );
    }

    #[test]
    fn state_change_with_freshness_window_remains_valid() {
        let plan = sample_plan();
        assert!(plan.validate().is_ok());
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
    fn reboot_only_disposition_is_explicit() {
        let mut plan = sample_plan();
        plan.intent.required_capabilities.clear();
        plan.intent.required_resources.clear();
        plan.steps = vec![
            PlanStep {
                sequence: 0,
                kind: PlanStepKind::Reboot,
                required_capabilities: [Capability::Reboot].into_iter().collect(),
                description: "reboot target".into(),
            },
            PlanStep {
                sequence: 1,
                kind: PlanStepKind::Verify,
                required_capabilities: [Capability::ObserveState].into_iter().collect(),
                description: "verify target state".into(),
            },
        ];
        plan.target_snapshot
            .profile
            .capabilities
            .insert(Capability::Reboot);
        plan.intent.required_capabilities.insert(Capability::Reboot);
        plan.verification.disposition = DeploymentDisposition::Rebooted;

        assert_eq!(plan.validate(), Ok(()));
        assert_eq!(
            plan.verification.disposition,
            DeploymentDisposition::Rebooted
        );
    }

    #[test]
    fn verification_disposition_is_part_of_plan_digest() {
        let plan = sample_plan();
        let before = plan.digest().expect("digest");

        let mut changed = plan;
        changed.verification.disposition = DeploymentDisposition::NotActivated;

        assert_ne!(before, changed.digest().expect("digest"));
    }

    #[test]
    fn digest_changes_when_intent_changes() {
        let plan = sample_plan();
        let before = plan.digest().expect("digest");

        let mut changed = plan;
        changed
            .intent
            .desired_state
            .properties
            .insert("hostname".into(), StateValue::String("new-name".into()));

        assert_ne!(before, changed.digest().expect("digest"));
    }

    #[test]
    fn receipt_rejects_final_snapshot_observation_before_execution() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth, 150).expect("authorized plan");

        let mut receipt = ExecutionReceipt {
            schema_version: SCHEMA_VERSION.into(),
            plan_digest: authorized.plan.digest().expect("plan digest"),
            target_snapshot_digest: authorized.authorization.target_snapshot_digest.clone(),
            authorization_digest: authorized.authorization.digest().expect("authorization digest"),
            final_target_snapshot_digest: ContentDigest::blake3(b"after-execution"),
            final_target_snapshot_observed_at_ms: 150,
            observed_disposition: authorized.plan.verification.disposition,
            started_at_ms: 151,
            finished_at_ms: 200,
            outcome: ExecutionOutcome::Succeeded,
            postcondition: PostconditionOutcome::Satisfied,
            verification_digest: Some(ContentDigest::blake3(b"verified-postcondition")),
            evidence: Vec::new(),
        };

        assert_eq!(
            receipt.validate_for(&authorized).expect_err("early final observation"),
            ReceiptValidationError::FinalSnapshotObservationOutsideExecution
        );

        receipt.final_target_snapshot_observed_at_ms = receipt.finished_at_ms + 1;
        assert_eq!(
            receipt.validate_for(&authorized).expect_err("late final observation"),
            ReceiptValidationError::FinalSnapshotObservationOutsideExecution
        );
    }

    #[test]
    fn receipt_binds_exact_authorized_plan_and_snapshot() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth, 150).expect("authorized plan");

        let receipt = ExecutionReceipt {
            schema_version: SCHEMA_VERSION.into(),
            plan_digest: authorized.plan.digest().expect("plan digest"),
            target_snapshot_digest: authorized.authorization.target_snapshot_digest.clone(),
            authorization_digest: authorized.authorization.digest().expect("authorization digest"),
            final_target_snapshot_digest: ContentDigest::blake3(b"after-execution"),
            final_target_snapshot_observed_at_ms: 200,
            observed_disposition: authorized.plan.verification.disposition,
            started_at_ms: 151,
            finished_at_ms: 200,
            outcome: ExecutionOutcome::Succeeded,
            postcondition: PostconditionOutcome::Satisfied,
            verification_digest: Some(ContentDigest::blake3(b"verified-postcondition")),

            evidence: Vec::new(),
        };

        assert!(receipt.validate_for(&authorized).is_ok());
    }

    #[test]
    fn recovery_is_not_forward_verified_success() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth, 150).expect("authorized plan");

        let receipt = ExecutionReceipt {
            schema_version: SCHEMA_VERSION.into(),
            plan_digest: authorized.plan.digest().expect("plan digest"),
            target_snapshot_digest: authorized.authorization.target_snapshot_digest.clone(),
            authorization_digest: authorized.authorization.digest().expect("authorization digest"),
            final_target_snapshot_digest: ContentDigest::blake3(b"after-recovery"),
            final_target_snapshot_observed_at_ms: 160,
            observed_disposition: authorized.plan.verification.disposition,
            started_at_ms: 151,
            finished_at_ms: 160,
            outcome: ExecutionOutcome::Recovered,
            postcondition: PostconditionOutcome::Satisfied,
            verification_digest: Some(ContentDigest::blake3(b"verified-postcondition")),

            evidence: Vec::new(),
        };

        assert!(receipt.validate_for(&authorized).is_ok());
        assert!(!receipt.is_verified_success());
    }

    #[test]
    fn mechanical_success_can_remain_unproven() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth, 150).expect("authorized plan");

        let receipt = ExecutionReceipt {
            schema_version: SCHEMA_VERSION.into(),
            plan_digest: authorized.plan.digest().expect("plan digest"),
            target_snapshot_digest: authorized.authorization.target_snapshot_digest.clone(),
            authorization_digest: authorized.authorization.digest().expect("authorization digest"),
            final_target_snapshot_digest: ContentDigest::blake3(b"after-execution"),
            final_target_snapshot_observed_at_ms: 160,
            observed_disposition: authorized.plan.verification.disposition,
            started_at_ms: 151,
            finished_at_ms: 160,
            outcome: ExecutionOutcome::Succeeded,
            postcondition: PostconditionOutcome::Unproven,
            verification_digest: None,
            evidence: Vec::new(),
        };

        assert!(receipt.validate_for(&authorized).is_ok());
        assert!(!receipt.is_verified_success());
    }

    #[test]
    fn rejects_receipt_without_final_snapshot_evidence() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth, 150).expect("authorized plan");

        let receipt = ExecutionReceipt {
            schema_version: SCHEMA_VERSION.into(),
            plan_digest: authorized.plan.digest().expect("plan digest"),
            target_snapshot_digest: authorized.authorization.target_snapshot_digest.clone(),
            authorization_digest: authorized.authorization.digest().expect("authorization digest"),
            final_target_snapshot_digest: ContentDigest {
                algorithm: String::new(),
                value: String::new(),
            },
            final_target_snapshot_observed_at_ms: 160,
            observed_disposition: authorized.plan.verification.disposition,
            started_at_ms: 151,
            finished_at_ms: 160,
            outcome: ExecutionOutcome::Succeeded,
            postcondition: PostconditionOutcome::Unproven,
            verification_digest: None,
            evidence: Vec::new(),
        };

        assert_eq!(
            receipt.validate_for(&authorized),
            Err(ReceiptValidationError::MissingFinalSnapshotEvidence)
        );
    }

    #[test]
    fn receipt_rejects_forged_authorization_binding() {
        let plan = sample_plan();
        let mut auth = authorization_for(&plan);
        auth.target_snapshot_digest.value = "forged".into();
        let authorized = AuthorizedDeploymentPlan {
            plan,
            authorization: auth,
        };

        let receipt = ExecutionReceipt {
            schema_version: SCHEMA_VERSION.into(),
            plan_digest: authorized.plan.digest().expect("plan digest"),
            target_snapshot_digest: authorized.authorization.target_snapshot_digest.clone(),
            authorization_digest: authorized.authorization.digest().expect("authorization digest"),
            final_target_snapshot_digest: ContentDigest::blake3(b"after-execution"),
            final_target_snapshot_observed_at_ms: 160,
            observed_disposition: authorized.plan.verification.disposition,
            started_at_ms: 151,
            finished_at_ms: 160,
            outcome: ExecutionOutcome::Succeeded,
            postcondition: PostconditionOutcome::Unproven,
            verification_digest: None,
            evidence: Vec::new(),
        };

        assert_eq!(
            receipt.validate_for(&authorized),
            Err(ReceiptValidationError::InvalidAuthorization(
                PlanValidationError::AuthorizationTargetSnapshotDigestMismatch
            ))
        );
    }

    #[test]
    fn receipt_rejects_authorization_without_expiry() {
        let plan = sample_plan();
        let mut auth = authorization_for(&plan);
        auth.valid_until_ms = None;
        let authorized = AuthorizedDeploymentPlan {
            plan,
            authorization: auth,
        };

        let receipt = ExecutionReceipt {
            schema_version: SCHEMA_VERSION.into(),
            plan_digest: authorized.plan.digest().expect("plan digest"),
            target_snapshot_digest: authorized.plan.target_snapshot.digest().expect("snapshot"),
            final_target_snapshot_digest: ContentDigest::blake3(b"after-execution"),
            final_target_snapshot_observed_at_ms: 160,
            observed_disposition: authorized.plan.verification.disposition,
            started_at_ms: 151,
            finished_at_ms: 160,
            outcome: ExecutionOutcome::Succeeded,
            postcondition: PostconditionOutcome::Unproven,
            verification_digest: None,
            evidence: Vec::new(),
        };

        assert_eq!(
            receipt.validate_for(&authorized),
            Err(ReceiptValidationError::AuthorizationMissingExpiry)
        );
    }

    #[test]
    fn consumed_authorization_requires_fresh_execution_admission() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth, 150).expect("authorized plan");
        let mut store = TestConsumptionStore::default();
        let consumed =
            consume_authorized_once(&mut store, &authorized, 151).expect("consumed");

        let execution = consumed.admit_execution(152).expect("execution admission");
        assert!(std::ptr::eq(execution.authorized_plan(), &authorized));
        assert_eq!(execution.consumed_at_ms(), 151);
        assert_eq!(execution.started_at_ms(), 152);
    }

    #[test]
    fn consumed_authorization_cannot_start_before_consumption() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth, 150).expect("authorized plan");
        let mut store = TestConsumptionStore::default();
        let consumed =
            consume_authorized_once(&mut store, &authorized, 151).expect("consumed");

        assert_eq!(
            consumed.admit_execution(150),
            Err(PlanValidationError::AuthorizationConsumptionFromFuture {
                consumed_at_ms: 151,
                now_ms: 150,
            })
        );
    }
    #[test]
    fn consume_authorized_once_requires_the_durable_store_boundary() {
        #[derive(Default)]
        struct InMemoryStore {
            consumed: BTreeMap<(String, String), ContentDigest>,
        }

        #[derive(Debug, Error, PartialEq, Eq)]
        enum StoreError {
            #[error("already consumed")]
            AlreadyConsumed,
        }

        impl AuthorizationConsumptionStore for InMemoryStore {
            type Error = StoreError;

            fn consume_once(
                &mut self,
                consumption: &AuthorizationConsumption,
            ) -> Result<(), Self::Error> {
                let key = (consumption.authority_id.clone(), consumption.nonce.clone());
                if self.consumed.contains_key(&key) {
                    return Err(StoreError::AlreadyConsumed);
                }
                self.consumed
                    .insert(key, consumption.authorization_digest.clone());
                Ok(())
            }
        }

        let plan = sample_plan();
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth.clone(), 150).expect("authorized plan");

        let mut store = InMemoryStore::default();
        let consumed =
            consume_authorized_once(&mut store, &authorized, 151).expect("consumption");
        assert_eq!(consumed.authorization_digest(), &auth.digest().expect("digest"));
        assert_eq!(consumed.authority_id(), "owner");
        assert_eq!(consumed.nonce(), "nonce-1");
        assert_eq!(consumed.consumed_at_ms(), 151);
        assert!(std::ptr::eq(consumed.authorized_plan(), &authorized));

        assert_eq!(
            consume_authorized_once(&mut store, &authorized, 152)
                .expect_err("replay must fail")
                .to_string(),
            "durable authorization consumption failed: already consumed"
        );
    }
    #[test]
    fn authorization_consumption_binds_exact_digest_and_nonce() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth.clone(), 150).expect("authorized plan");
        let consumption =
            AuthorizationConsumption::for_authorized_plan(&authorized, 151).expect("consumption");
        assert_eq!(consumption.authority_id, auth.authority_id);
        assert_eq!(consumption.nonce, auth.nonce);
        assert_eq!(consumption.authorization_digest, auth.digest().expect("auth digest"));
        assert_eq!(consumption.consumed_at_ms, 151);
    }

    #[test]
    fn authorization_consumption_rejects_empty_identity() {
        let plan = sample_plan();
        let mut auth = authorization_for(&plan);
        auth.nonce.clear();
        assert_eq!(
            plan.clone().authorize(auth.clone(), 151),
            Err(PlanValidationError::EmptyNonce)
        );

        auth.nonce = "nonce-1".into();
        auth.authority_id.clear();
        assert_eq!(
            plan.authorize(auth, 151),
            Err(PlanValidationError::EmptyAuthority)
        );
    }

    #[test]
    fn authorization_consumption_requires_current_valid_authorization() {
        let mut plan = sample_plan();
        plan.expires_at_ms = Some(150);
        let auth = authorization_for(&plan);
        let authorized = plan
            .authorize(auth, 150)
            .expect("authorization at exact expiry");
        assert!(
            AuthorizationConsumption::for_authorized_plan(&authorized, 150).is_ok()
        );
        assert_eq!(
            AuthorizationConsumption::for_authorized_plan(&authorized, 151),
            Err(PlanValidationError::IntentExpired)
        );
    }

    #[test]
    fn authorization_consumption_store_contract_rejects_replay_and_nonce_reuse() {
        #[derive(Default)]
        struct InMemoryStore {
            consumed: BTreeMap<(String, String), ContentDigest>,
        }

        #[derive(Debug, Error, PartialEq, Eq)]
        enum StoreError {
            #[error("authorization already consumed")]
            AlreadyConsumed,
            #[error("authorization nonce was previously consumed for a different authorization")]
            NonceReused,
        }

        impl AuthorizationConsumptionStore for InMemoryStore {
            type Error = StoreError;

            fn consume_once(
                &mut self,
                consumption: &AuthorizationConsumption,
            ) -> Result<(), Self::Error> {
                let key = (consumption.authority_id.clone(), consumption.nonce.clone());
                match self.consumed.entry(key) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(consumption.authorization_digest.clone());
                        Ok(())
                    }
                    std::collections::btree_map::Entry::Occupied(entry)
                        if entry.get() == &consumption.authorization_digest =>
                    {
                        Err(StoreError::AlreadyConsumed)
                    }
                    std::collections::btree_map::Entry::Occupied(_) => {
                        Err(StoreError::NonceReused)
                    }
                }
            }
        }

        let plan = sample_plan();
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth.clone(), 150).expect("authorized plan");
        let consumption =
            AuthorizationConsumption::for_authorized_plan(&authorized, 151).expect("consumption");

        let mut store = InMemoryStore::default();
        assert!(store.consume_once(&consumption).is_ok());
        assert_eq!(
            store.consume_once(&consumption),
            Err(StoreError::AlreadyConsumed)
        );

        let mut altered = authorized.clone();
        altered.authorization.valid_from_ms = Some(149);
        let altered_consumption =
            AuthorizationConsumption::for_authorized_plan(&altered, 151).expect("altered");
        assert_ne!(
            altered_consumption.authorization_digest,
            consumption.authorization_digest
        );
        assert_eq!(
            store.consume_once(&altered_consumption),
            Err(StoreError::NonceReused)
        );
    }

    #[test]
    fn rejects_execution_with_stale_target_snapshot() {
        let mut plan = sample_plan();
        plan.max_target_snapshot_age_ms = Some(50);
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth, 150).expect("authorized plan");

        let receipt = ExecutionReceipt {
            schema_version: SCHEMA_VERSION.into(),
            plan_digest: authorized.plan.digest().expect("plan digest"),
            target_snapshot_digest: authorized.authorization.target_snapshot_digest.clone(),
            authorization_digest: authorized.authorization.digest().expect("authorization digest"),
            final_target_snapshot_digest: ContentDigest::blake3(b"after-execution"),
            final_target_snapshot_observed_at_ms: 151,
            observed_disposition: authorized.plan.verification.disposition,
            started_at_ms: 150,
            finished_at_ms: 160,
            outcome: ExecutionOutcome::Succeeded,
            postcondition: PostconditionOutcome::Unproven,
            verification_digest: None,
            evidence: Vec::new(),
        };

        assert_eq!(
            receipt.validate_for(&authorized),
            Err(ReceiptValidationError::TargetSnapshotStaleAtExecution {
                age_ms: 60,
                max_age_ms: 50,
            })
        );
    }

    #[test]
    fn accepts_execution_at_target_snapshot_age_limit() {
        let mut plan = sample_plan();
        plan.max_target_snapshot_age_ms = Some(60);
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth, 150).expect("authorized plan");

        let receipt = ExecutionReceipt {
            schema_version: SCHEMA_VERSION.into(),
            plan_digest: authorized.plan.digest().expect("plan digest"),
            target_snapshot_digest: authorized.authorization.target_snapshot_digest.clone(),
            authorization_digest: authorized.authorization.digest().expect("authorization digest"),
            final_target_snapshot_digest: ContentDigest::blake3(b"after-execution"),
            final_target_snapshot_observed_at_ms: 160,
            observed_disposition: authorized.plan.verification.disposition,
            started_at_ms: 150,
            finished_at_ms: 160,
            outcome: ExecutionOutcome::Succeeded,
            postcondition: PostconditionOutcome::Unproven,
            verification_digest: None,
            evidence: Vec::new(),
        };

        assert!(receipt.validate_for(&authorized).is_ok());
    }

    #[test]
    fn rejects_verified_postcondition_without_evidence_digest() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth, 150).expect("authorized plan");

        let receipt = ExecutionReceipt {
            schema_version: SCHEMA_VERSION.into(),
            plan_digest: authorized.plan.digest().expect("plan digest"),
            target_snapshot_digest: authorized.authorization.target_snapshot_digest.clone(),
            authorization_digest: authorized.authorization.digest().expect("authorization digest"),
            final_target_snapshot_digest: ContentDigest::blake3(b"after-execution"),
            final_target_snapshot_observed_at_ms: 160,
            observed_disposition: authorized.plan.verification.disposition,
            started_at_ms: 151,
            finished_at_ms: 160,
            outcome: ExecutionOutcome::Succeeded,
            postcondition: PostconditionOutcome::Satisfied,
            verification_digest: None,

            evidence: Vec::new(),
        };

        assert_eq!(
            receipt.validate_for(&authorized),
            Err(ReceiptValidationError::MissingVerificationEvidence)
        );
    }

    #[test]
    fn required_attestation_needs_concrete_receipt_evidence() {
        let mut plan = sample_plan();
        plan.verification.require_attestation = true;
        plan.target_snapshot
            .profile
            .capabilities
            .insert(Capability::AttestState);
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth, 150).expect("authorized plan");

        let receipt = ExecutionReceipt {
            schema_version: SCHEMA_VERSION.into(),
            plan_digest: authorized.plan.digest().expect("plan digest"),
            target_snapshot_digest: authorized.authorization.target_snapshot_digest.clone(),
            authorization_digest: authorized.authorization.digest().expect("authorization digest"),
            final_target_snapshot_digest: ContentDigest::blake3(b"after-execution"),
            final_target_snapshot_observed_at_ms: 160,
            observed_disposition: authorized.plan.verification.disposition,
            started_at_ms: 151,
            finished_at_ms: 160,
            outcome: ExecutionOutcome::Succeeded,
            postcondition: PostconditionOutcome::Satisfied,
            verification_digest: Some(ContentDigest::blake3(b"verified-postcondition")),
            evidence: Vec::new(),
        };

        assert_eq!(
            receipt.validate_for(&authorized),
            Err(ReceiptValidationError::MissingAttestationEvidence)
        );
    }

    #[test]
    fn required_attestation_accepts_concrete_receipt_evidence() {
        let mut plan = sample_plan();
        plan.verification.require_attestation = true;
        plan.target_snapshot
            .profile
            .capabilities
            .insert(Capability::AttestState);
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth, 150).expect("authorized plan");

        let receipt = ExecutionReceipt {
            schema_version: SCHEMA_VERSION.into(),
            plan_digest: authorized.plan.digest().expect("plan digest"),
            target_snapshot_digest: authorized.authorization.target_snapshot_digest.clone(),
            authorization_digest: authorized.authorization.digest().expect("authorization digest"),
            final_target_snapshot_digest: ContentDigest::blake3(b"after-execution"),
            final_target_snapshot_observed_at_ms: 160,
            observed_disposition: authorized.plan.verification.disposition,
            started_at_ms: 151,
            finished_at_ms: 160,
            outcome: ExecutionOutcome::Succeeded,
            postcondition: PostconditionOutcome::Satisfied,
            verification_digest: Some(ContentDigest::blake3(b"verified-postcondition")),
            evidence: vec![AttestationRef {
                media_type: "application/json".into(),
                uri: "urn:example:attestation:1".into(),
                digest: ContentDigest::blake3(b"attestation"),
            }],
        };

        assert!(receipt.validate_for(&authorized).is_ok());
    }

    #[test]
    fn receipt_rejects_incomplete_attestation_evidence() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth, 150).expect("authorized plan");

        let receipt = ExecutionReceipt {
            schema_version: SCHEMA_VERSION.into(),
            plan_digest: authorized.plan.digest().expect("plan digest"),
            target_snapshot_digest: authorized.authorization.target_snapshot_digest.clone(),
            authorization_digest: authorized.authorization.digest().expect("authorization digest"),
            final_target_snapshot_digest: ContentDigest::blake3(b"after-execution"),
            final_target_snapshot_observed_at_ms: 160,
            observed_disposition: authorized.plan.verification.disposition,
            started_at_ms: 151,
            finished_at_ms: 160,
            outcome: ExecutionOutcome::Succeeded,
            postcondition: PostconditionOutcome::Satisfied,
            verification_digest: Some(ContentDigest::blake3(b"verified-postcondition")),
            evidence: vec![AttestationRef {
                media_type: "application/json".into(),
                uri: String::new(),
                digest: ContentDigest::blake3(b"attestation"),
            }],
        };

        assert_eq!(
            receipt.validate_for(&authorized),
            Err(ReceiptValidationError::InvalidAttestationEvidence)
        );
    }

    #[test]
    fn unproven_postcondition_is_allowed_without_proof() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth, 150).expect("authorized plan");

        let receipt = ExecutionReceipt {
            schema_version: SCHEMA_VERSION.into(),
            plan_digest: authorized.plan.digest().expect("plan digest"),
            target_snapshot_digest: authorized.authorization.target_snapshot_digest.clone(),
            authorization_digest: authorized.authorization.digest().expect("authorization digest"),
            final_target_snapshot_digest: ContentDigest::blake3(b"after-execution"),
            final_target_snapshot_observed_at_ms: 160,
            observed_disposition: authorized.plan.verification.disposition,
            started_at_ms: 151,
            finished_at_ms: 160,
            outcome: ExecutionOutcome::Succeeded,
            postcondition: PostconditionOutcome::Unproven,
            verification_digest: None,
            evidence: Vec::new(),
        };

        assert!(receipt.validate_for(&authorized).is_ok());
        assert!(!receipt.is_verified_success());
    }

    #[test]
    fn rejects_receipt_with_wrong_observed_disposition() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth, 150).expect("authorized plan");

        let receipt = ExecutionReceipt {
            schema_version: SCHEMA_VERSION.into(),
            plan_digest: authorized.plan.digest().expect("plan digest"),
            target_snapshot_digest: authorized.authorization.target_snapshot_digest.clone(),
            authorization_digest: authorized.authorization.digest().expect("authorization digest"),
            final_target_snapshot_digest: ContentDigest::blake3(b"after-execution"),
            final_target_snapshot_observed_at_ms: 160,
            observed_disposition: DeploymentDisposition::NotActivated,
            started_at_ms: 151,
            finished_at_ms: 160,
            outcome: ExecutionOutcome::Succeeded,
            postcondition: PostconditionOutcome::Unproven,
            verification_digest: None,
            evidence: Vec::new(),
        };

        assert_eq!(
            receipt.validate_for(&authorized),
            Err(ReceiptValidationError::ObservedDispositionMismatch)
        );
    }

    #[test]
    fn rejects_receipt_for_tampered_plan() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth, 150).expect("authorized plan");

        let receipt = ExecutionReceipt {
            schema_version: SCHEMA_VERSION.into(),
            plan_digest: ContentDigest::blake3(b"wrong-plan"),
            target_snapshot_digest: authorized.authorization.target_snapshot_digest.clone(),
            authorization_digest: authorized.authorization.digest().expect("authorization digest"),
            final_target_snapshot_digest: ContentDigest::blake3(b"after-execution"),
            final_target_snapshot_observed_at_ms: 200,
            observed_disposition: authorized.plan.verification.disposition,
            started_at_ms: 151,
            finished_at_ms: 200,
            outcome: ExecutionOutcome::Succeeded,
            postcondition: PostconditionOutcome::Satisfied,
            verification_digest: Some(ContentDigest::blake3(b"verified-postcondition")),

            evidence: Vec::new(),
        };

        assert_eq!(
            receipt.validate_for(&authorized),
            Err(ReceiptValidationError::PlanDigestMismatch)
        );
    }

    #[test]
    fn rejects_receipt_with_invalid_timestamp_order() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth, 150).expect("authorized plan");

        let receipt = ExecutionReceipt {
            schema_version: SCHEMA_VERSION.into(),
            plan_digest: authorized.plan.digest().expect("plan digest"),
            target_snapshot_digest: authorized.authorization.target_snapshot_digest.clone(),
            authorization_digest: authorized.authorization.digest().expect("authorization digest"),
            final_target_snapshot_digest: ContentDigest::blake3(b"after-execution"),
            final_target_snapshot_observed_at_ms: 199,
            observed_disposition: authorized.plan.verification.disposition,
            started_at_ms: 200,
            finished_at_ms: 199,
            outcome: ExecutionOutcome::Failed,
            postcondition: PostconditionOutcome::NotEvaluated,
            verification_digest: None,
            evidence: Vec::new(),
        };

        assert_eq!(
            receipt.validate_for(&authorized),
            Err(ReceiptValidationError::TimestampOrderInvalid)
        );
    }

    #[test]
    fn rejects_receipt_started_before_authorization_window() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth, 150).expect("authorized plan");

        let receipt = ExecutionReceipt {
            schema_version: SCHEMA_VERSION.into(),
            plan_digest: authorized.plan.digest().expect("plan digest"),
            target_snapshot_digest: authorized.authorization.target_snapshot_digest.clone(),
            authorization_digest: authorized.authorization.digest().expect("authorization digest"),
            final_target_snapshot_digest: ContentDigest::blake3(b"after-execution"),
            final_target_snapshot_observed_at_ms: 150,
            observed_disposition: authorized.plan.verification.disposition,
            started_at_ms: 99,
            finished_at_ms: 150,
            outcome: ExecutionOutcome::Succeeded,
            postcondition: PostconditionOutcome::Satisfied,
            verification_digest: Some(ContentDigest::blake3(b"verified-postcondition")),

            evidence: Vec::new(),
        };

        assert_eq!(
            receipt.validate_for(&authorized),
            Err(ReceiptValidationError::StartedBeforeAuthorization)
        );
    }

    #[test]
    fn rejects_receipt_started_before_target_snapshot() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth, 150).expect("authorized plan");

        let receipt = ExecutionReceipt {
            schema_version: SCHEMA_VERSION.into(),
            plan_digest: authorized.plan.digest().expect("plan digest"),
            target_snapshot_digest: authorized.authorization.target_snapshot_digest.clone(),
            authorization_digest: authorized.authorization.digest().expect("authorization digest"),
            final_target_snapshot_digest: ContentDigest::blake3(b"after-execution"),
            final_target_snapshot_observed_at_ms: 150,
            observed_disposition: authorized.plan.verification.disposition,
            started_at_ms: 89,
            finished_at_ms: 150,
            outcome: ExecutionOutcome::Failed,
            postcondition: PostconditionOutcome::NotEvaluated,
            verification_digest: None,
            evidence: Vec::new(),
        };

        assert_eq!(
            receipt.validate_for(&authorized),
            Err(ReceiptValidationError::StartedBeforeTargetSnapshot)
        );
    }

    #[test]
    fn rejects_receipt_finished_after_authorization_expiry() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth, 150).expect("authorized plan");

        let receipt = ExecutionReceipt {
            schema_version: SCHEMA_VERSION.into(),
            plan_digest: authorized.plan.digest().expect("plan digest"),
            target_snapshot_digest: authorized.authorization.target_snapshot_digest.clone(),
            authorization_digest: authorized.authorization.digest().expect("authorization digest"),
            final_target_snapshot_digest: ContentDigest::blake3(b"after-execution"),
            final_target_snapshot_observed_at_ms: 201,
            observed_disposition: authorized.plan.verification.disposition,
            started_at_ms: 151,
            finished_at_ms: 201,
            outcome: ExecutionOutcome::Succeeded,
            postcondition: PostconditionOutcome::Satisfied,
            verification_digest: Some(ContentDigest::blake3(b"verified-postcondition")),

            evidence: Vec::new(),
        };

        assert_eq!(
            receipt.validate_for(&authorized),
            Err(ReceiptValidationError::FinishedAfterAuthorizationExpiry)
        );
    }

    #[test]
    fn receipt_accepts_distinct_post_execution_snapshot() {
        let plan = sample_plan();
        let auth = authorization_for(&plan);
        let authorized = plan.authorize(auth, 150).expect("authorized plan");

        let receipt = ExecutionReceipt {
            schema_version: SCHEMA_VERSION.into(),
            plan_digest: authorized.plan.digest().expect("plan digest"),
            target_snapshot_digest: authorized.authorization.target_snapshot_digest.clone(),
            authorization_digest: authorized.authorization.digest().expect("authorization digest"),
            final_target_snapshot_digest: ContentDigest::blake3(b"changed-state"),
            final_target_snapshot_observed_at_ms: 199,
            observed_disposition: authorized.plan.verification.disposition,
            started_at_ms: 151,
            finished_at_ms: 199,
            outcome: ExecutionOutcome::Succeeded,
            postcondition: PostconditionOutcome::Satisfied,
            verification_digest: Some(ContentDigest::blake3(b"verified-postcondition")),

            evidence: Vec::new(),
        };

        assert!(receipt.validate_for(&authorized).is_ok());
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
