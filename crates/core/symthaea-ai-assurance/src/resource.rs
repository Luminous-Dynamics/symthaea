// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Concrete-resource identity binding for strict host tool adapters.
//!
//! Logical [`crate::Scope`] values describe policy intent, but a concrete
//! executor must eventually resolve that intent onto an operating-system or
//! remote resource. Re-resolving an ambient name after authorization creates a
//! substitution/TOCTOU gap even when the authority token itself is exact.
//!
//! This module provides an adapter-neutral mechanism:
//!
//! 1. trusted resolver code creates a [`ResolvedResource`] containing an opaque
//!    handle plus a deterministic [`ResourceIdentity`];
//! 2. [`ResourceRuntime`] pins the resolver trust domain and verifies the
//!    resource before action admission;
//! 3. the resource identity digest is folded into the canonical action payload,
//!    so execution authority commits to that exact resolved identity;
//! 4. [`ResourceAction`] retains the opaque handle across typestate transitions
//!    and exposes it only to the authorized execution closure, avoiding a second
//!    ambient name lookup in the intended path.
//!
//! The mechanism does **not** make arbitrary trusted-process code unable to use
//! ambient filesystem/network APIs. Concrete adapters still need OS/WASI/Nix or
//! equivalent isolation and must construct identities that actually represent
//! the resource they retain.
//!
//! ```compile_fail
//! use symthaea_ai_assurance::{Authorized, ResourceAction, Write};
//!
//! fn bypass<H>(action: ResourceAction<Write, Authorized, H>) {
//!     // The resource-bound API intentionally exposes no direct
//!     // `record_execution` transition. Concrete execution goes through
//!     // `execute_with`, which receives the retained resource handle.
//!     let _ = action.record_execution([0; 32]);
//! }
//! ```

use crate::action::{
    ActionDescriptor, ActionId, ActionRisk, Authorized, Executed, Observation, Observed, Proposed,
    ResolutionDecision, Resolved, RiskAssessed,
};
use crate::capability::{CapabilityKind, GrantId, PrincipalId, Read, Scope};
use crate::host::{ResolutionEvidenceReceipt, ResolutionError, RuntimeAction, TrustedRuntime};
use crate::resolution::ResolutionGrant;
use crate::trusted::{
    AuthorityDomain, AuthorityDomainId, AuthorityEpoch, AuthorityVerifier, TrustError,
    TrustedBoundOneShotCapability,
};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// Stable adapter schema identity used when interpreting a concrete resource.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AdapterSchema {
    name: String,
    version: u32,
}

impl AdapterSchema {
    /// Construct an adapter schema descriptor.
    pub fn new(name: impl Into<String>, version: u32) -> Result<Self, ResourceIdentityError> {
        let name = name.into();
        if !valid_label(&name) {
            return Err(ResourceIdentityError::InvalidAdapterName(name));
        }
        Ok(Self { name, version })
    }

    /// Stable adapter family name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Adapter schema version.
    pub fn version(&self) -> u32 {
        self.version
    }
}

/// Deterministic identity of the concrete resource selected by a trusted adapter.
///
/// `stable_identity_digest` is adapter-defined: examples include a Nix store
/// path digest, repository/worktree identity, preopened-directory identity,
/// service-manager object id, or remote service/TLS identity commitment.
/// `environment_digest` commits to the environment in which that identity is
/// interpreted, such as a workspace root, mount namespace, container image, or
/// endpoint-policy snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceIdentity {
    scope: Scope,
    kind: String,
    stable_identity_digest: [u8; 32],
    environment_digest: [u8; 32],
    adapter: AdapterSchema,
    binding: [u8; 32],
}

impl ResourceIdentity {
    /// Construct a concrete resource identity.
    pub fn new(
        scope: Scope,
        kind: impl Into<String>,
        stable_identity_digest: [u8; 32],
        environment_digest: [u8; 32],
        adapter: AdapterSchema,
    ) -> Result<Self, ResourceIdentityError> {
        let kind = kind.into();
        if !valid_label(&kind) {
            return Err(ResourceIdentityError::InvalidResourceKind(kind));
        }

        let binding = compute_resource_binding(
            &scope,
            &kind,
            stable_identity_digest,
            environment_digest,
            &adapter,
        );
        Ok(Self {
            scope,
            kind,
            stable_identity_digest,
            environment_digest,
            adapter,
            binding,
        })
    }

    /// Logical policy scope that was resolved.
    pub fn scope(&self) -> &Scope {
        &self.scope
    }

    /// Adapter-defined resource-kind label.
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// Stable adapter-defined concrete identity digest.
    pub fn stable_identity_digest(&self) -> [u8; 32] {
        self.stable_identity_digest
    }

    /// Environment/namespace identity digest used during resolution.
    pub fn environment_digest(&self) -> [u8; 32] {
        self.environment_digest
    }

    /// Adapter schema that defines the meaning of the identity fields.
    pub fn adapter(&self) -> &AdapterSchema {
        &self.adapter
    }

    /// Domain-separated digest used to bind this resource into an action.
    pub fn binding(&self) -> [u8; 32] {
        self.binding
    }
}

/// Failure to construct a deterministic resource identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceIdentityError {
    /// Resource-kind label is empty or contains unsupported characters.
    InvalidResourceKind(String),
    /// Adapter name is empty or contains unsupported characters.
    InvalidAdapterName(String),
}

impl fmt::Display for ResourceIdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidResourceKind(value) => {
                write!(f, "invalid resource kind label: {value:?}")
            }
            Self::InvalidAdapterName(value) => {
                write!(f, "invalid resource adapter name: {value:?}")
            }
        }
    }
}

impl std::error::Error for ResourceIdentityError {}

/// Trusted policy object allowed to attest concrete resource resolutions.
///
/// This type belongs in trusted adapter/host code. Less-trusted components
/// should receive [`ResolvedResource`] values rather than the resolver domain.
#[derive(Debug)]
pub struct ResourceResolverDomain {
    inner: AuthorityDomain,
}

impl ResourceResolverDomain {
    /// Create an independent resource-resolution trust domain.
    pub fn new(principal: PrincipalId) -> Self {
        Self {
            inner: AuthorityDomain::new(principal),
        }
    }

    /// Resource resolver trust-domain identity.
    pub fn domain_id(&self) -> AuthorityDomainId {
        self.inner.domain_id()
    }

    /// Root resource-resolver principal.
    pub fn principal(&self) -> PrincipalId {
        self.inner.principal()
    }

    /// Create a verifier retained by [`ResourceRuntime`].
    pub fn verifier(&self) -> ResourceVerifier {
        ResourceVerifier {
            inner: self.inner.verifier(),
        }
    }

    /// Bind an opaque concrete handle to its deterministic identity.
    ///
    /// The returned value is affine when `H` is affine and carries an exact
    /// resolver-domain attestation over the resource identity binding.
    pub fn resolve<H>(
        &self,
        handle: H,
        identity: ResourceIdentity,
        expires_at: Option<SystemTime>,
    ) -> ResolvedResource<H> {
        let attestation = self.inner.issue_bound_one_shot::<Read>(
            self.inner.principal(),
            identity.scope().clone(),
            expires_at,
            identity.binding(),
        );
        ResolvedResource {
            identity,
            attestation,
            handle,
        }
    }

    /// Revoke resource resolutions from earlier resolver epochs.
    pub fn revoke_all(&self) -> Result<AuthorityEpoch, TrustError> {
        self.inner.revoke_all()
    }
}

/// Host-retained trust anchor for concrete resource resolution.
#[derive(Debug, Clone)]
pub struct ResourceVerifier {
    inner: AuthorityVerifier,
}

impl ResourceVerifier {
    /// Resource resolver trust-domain identity.
    pub fn domain_id(&self) -> AuthorityDomainId {
        self.inner.domain_id()
    }

    /// Current resource resolver revocation epoch.
    pub fn current_epoch(&self) -> AuthorityEpoch {
        self.inner.current_epoch()
    }
}

/// Opaque retained handle plus exact resource-resolution provenance.
///
/// The handle itself is deliberately not exposed through general action states.
/// [`ResourceAction::execute_with`] is the intended place where a concrete
/// adapter receives mutable access to it.
#[derive(Debug)]
pub struct ResolvedResource<H> {
    identity: ResourceIdentity,
    attestation: TrustedBoundOneShotCapability<Read>,
    handle: H,
}

impl<H> ResolvedResource<H> {
    /// Deterministic concrete resource identity.
    pub fn identity(&self) -> &ResourceIdentity {
        &self.identity
    }

    /// Resolver domain that attested this resource.
    pub fn resolver_domain(&self) -> AuthorityDomainId {
        self.attestation.domain_id()
    }

    /// Resolver epoch in which this resource was resolved.
    pub fn resolver_epoch(&self) -> AuthorityEpoch {
        self.attestation.epoch()
    }

    /// Resource-resolution attestation grant id.
    pub fn attestation_grant_id(&self) -> GrantId {
        self.attestation.metadata().grant_id()
    }

    fn validate_with(
        &self,
        verifier: &ResourceVerifier,
        now: SystemTime,
    ) -> Result<(), ResourceError> {
        self.attestation
            .validate_with(&verifier.inner, now)
            .map_err(ResourceError::Trust)?;

        if self.attestation.metadata().scope() != self.identity.scope() {
            return Err(ResourceError::ScopeMismatch {
                attested: self.attestation.metadata().scope().clone(),
                identity: self.identity.scope().clone(),
            });
        }
        if self.attestation.binding() != self.identity.binding() {
            return Err(ResourceError::BindingMismatch {
                expected: self.identity.binding(),
                actual: self.attestation.binding(),
            });
        }
        Ok(())
    }
}

/// Host wrapper that pins the concrete-resource resolver trust domain.
#[derive(Debug, Clone)]
pub struct ResourceRuntime {
    runtime: TrustedRuntime,
    resource_verifier: ResourceVerifier,
    clock: Arc<ResourceClock>,
}

impl ResourceRuntime {
    /// Construct a resource-bound runtime from an already configured strict host
    /// runtime and a resource resolver verifier selected by trusted host policy.
    pub fn new(runtime: TrustedRuntime, resource_verifier: ResourceVerifier) -> Self {
        Self {
            runtime,
            resource_verifier,
            clock: Arc::new(ResourceClock::new()),
        }
    }

    /// Pinned resource resolver trust domain.
    pub fn resource_resolver_domain(&self) -> AuthorityDomainId {
        self.resource_verifier.domain_id()
    }

    /// Admit an action only after a concrete resource has already been resolved.
    ///
    /// The logical action scope is taken from the verified resource identity,
    /// not from a second caller-provided scope. The action payload commits to
    /// both the caller's canonical operation bytes and the resource identity.
    pub fn admit_resolved<K: CapabilityKind, H>(
        &self,
        actor: PrincipalId,
        kind: impl Into<String>,
        resource: ResolvedResource<H>,
        canonical_payload: &[u8],
    ) -> Result<ResourceAction<K, Proposed, H>, ResourceError> {
        resource.validate_with(&self.resource_verifier, self.clock.now())?;
        let scope = resource.identity().scope().clone();
        let payload = bind_payload(resource.identity(), canonical_payload);
        let inner = self.runtime.admit::<K>(actor, kind, scope, &payload);
        Ok(ResourceAction {
            inner,
            resource,
            resource_verifier: self.resource_verifier.clone(),
            clock: Arc::clone(&self.clock),
        })
    }
}

/// Resource-bound strict action lifecycle.
///
/// The concrete resource handle moves with the action and cannot be recovered
/// through this API until the action is resolved. The authorized execution
/// transition exposes the retained handle to one adapter closure rather than
/// asking the adapter to resolve a name again.
pub struct ResourceAction<K: CapabilityKind, S, H> {
    inner: RuntimeAction<K, S>,
    resource: ResolvedResource<H>,
    resource_verifier: ResourceVerifier,
    clock: Arc<ResourceClock>,
}

impl<K: CapabilityKind, S, H> ResourceAction<K, S, H> {
    /// Stable action identity.
    pub fn id(&self) -> ActionId {
        self.inner.id()
    }

    /// Acting principal.
    pub fn actor(&self) -> PrincipalId {
        self.inner.actor()
    }

    /// Immutable action descriptor, whose fingerprint commits to the resource binding.
    pub fn descriptor(&self) -> &ActionDescriptor {
        self.inner.descriptor()
    }

    /// Concrete resource identity bound into this action.
    pub fn resource_identity(&self) -> &ResourceIdentity {
        self.resource.identity()
    }

    /// Resource resolver domain pinned by the host wrapper.
    pub fn resource_resolver_domain(&self) -> AuthorityDomainId {
        self.resource_verifier.domain_id()
    }
}

impl<K: CapabilityKind, H> ResourceAction<K, Proposed, H> {
    /// Attach explicit action risk.
    pub fn assess(self, risk: ActionRisk) -> ResourceAction<K, RiskAssessed, H> {
        ResourceAction {
            inner: self.inner.assess(risk),
            resource: self.resource,
            resource_verifier: self.resource_verifier,
            clock: self.clock,
        }
    }
}

impl<K: CapabilityKind, H> ResourceAction<K, RiskAssessed, H> {
    /// Action risk classification.
    pub fn risk(&self) -> ActionRisk {
        self.inner.risk()
    }

    /// Exact execution-authorization binding, including the concrete resource
    /// identity through the action fingerprint.
    pub fn authorization_binding(&self) -> [u8; 32] {
        self.inner.authorization_binding()
    }

    /// Consume exact execution authority.
    pub fn authorize(
        self,
        grant: TrustedBoundOneShotCapability<K>,
    ) -> Result<ResourceAction<K, Authorized, H>, TrustError> {
        let inner = self.inner.authorize(grant)?;
        Ok(ResourceAction {
            inner,
            resource: self.resource,
            resource_verifier: self.resource_verifier,
            clock: self.clock,
        })
    }
}

impl<K: CapabilityKind, H> ResourceAction<K, Authorized, H> {
    /// Exact authorization binding consumed by this action.
    pub fn authorization_binding(&self) -> [u8; 32] {
        self.inner.authorization_binding()
    }

    /// Execute through the already-resolved concrete resource handle.
    ///
    /// Resource resolver domain/epoch/expiry are revalidated immediately before
    /// the closure is invoked. The adapter returns a canonical output digest,
    /// which is then recorded by the strict action runtime.
    ///
    /// This prevents accidental ambient re-resolution through the resource-bound
    /// API, but trusted closure code can still ignore the handle and use ambient
    /// authority if the surrounding process grants it. Use WASI/process/OS
    /// isolation for hostile code.
    pub fn execute_with<F, E>(
        mut self,
        execute: F,
    ) -> Result<ResourceAction<K, Executed, H>, ResourceExecutionError<E>>
    where
        F: FnOnce(&mut H) -> Result<[u8; 32], E>,
    {
        self.resource
            .validate_with(&self.resource_verifier, self.clock.now())
            .map_err(ResourceExecutionError::Resource)?;
        let output_digest = execute(&mut self.resource.handle)
            .map_err(ResourceExecutionError::Adapter)?;
        let inner = self
            .inner
            .record_execution(output_digest)
            .map_err(ResourceExecutionError::Trust)?;
        Ok(ResourceAction {
            inner,
            resource: self.resource,
            resource_verifier: self.resource_verifier,
            clock: self.clock,
        })
    }
}

impl<K: CapabilityKind, H> ResourceAction<K, Executed, H> {
    /// Exact independent-observation binding.
    pub fn observation_binding(&self) -> [u8; 32] {
        self.inner.observation_binding()
    }

    /// Attach independent observation using the underlying strict host runtime.
    pub fn observe(
        self,
        observer: TrustedBoundOneShotCapability<crate::Observe>,
        observation: Observation,
    ) -> Result<ResourceAction<K, Observed, H>, TrustError> {
        let inner = self.inner.observe(observer, observation)?;
        Ok(ResourceAction {
            inner,
            resource: self.resource,
            resource_verifier: self.resource_verifier,
            clock: self.clock,
        })
    }
}

impl<K: CapabilityKind, H> ResourceAction<K, Observed, H> {
    /// Exact final-resolution binding for this observed resource/action lineage.
    pub fn resolution_binding(&self, decision: ResolutionDecision) -> [u8; 32] {
        self.inner.resolution_binding(decision)
    }

    /// Consume exact final-resolution authority and emit resource-aware evidence.
    pub fn resolve(
        self,
        grant: ResolutionGrant,
        decision: ResolutionDecision,
    ) -> Result<
        (
            ResourceAction<K, Resolved, H>,
            ResourceEvidenceReceipt,
        ),
        ResolutionError,
    > {
        let (inner, resolution_receipt) = self.inner.resolve(grant, decision)?;
        let receipt = ResourceEvidenceReceipt {
            resolution_receipt,
            resource_identity: self.resource.identity().clone(),
            resource_resolver_domain: self.resource.resolver_domain(),
            resource_resolver_epoch: self.resource.resolver_epoch(),
            resource_attestation_grant_id: self.resource.attestation_grant_id(),
        };
        Ok((
            ResourceAction {
                inner,
                resource: self.resource,
                resource_verifier: self.resource_verifier,
                clock: self.clock,
            },
            receipt,
        ))
    }
}

impl<K: CapabilityKind, H> ResourceAction<K, Resolved, H> {
    /// Final exact-resolution evidence from the underlying strict runtime.
    pub fn resolution_receipt(&self) -> &ResolutionEvidenceReceipt {
        self.inner.resolution_receipt()
    }

    /// Consume the completed action and recover the retained resource handle for
    /// trusted adapter cleanup/reuse.
    pub fn into_resource(self) -> ResolvedResource<H> {
        self.resource
    }
}

/// Final evidence binding exact execution/observation/resolution lineage to the
/// concrete resource identity retained by the adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceEvidenceReceipt {
    resolution_receipt: ResolutionEvidenceReceipt,
    resource_identity: ResourceIdentity,
    resource_resolver_domain: AuthorityDomainId,
    resource_resolver_epoch: AuthorityEpoch,
    resource_attestation_grant_id: GrantId,
}

impl ResourceEvidenceReceipt {
    /// Underlying exact execution/observation/resolution receipt.
    pub fn resolution_receipt(&self) -> &ResolutionEvidenceReceipt {
        &self.resolution_receipt
    }

    /// Concrete resource identity bound into the action fingerprint.
    pub fn resource_identity(&self) -> &ResourceIdentity {
        &self.resource_identity
    }

    /// Resource resolver trust domain.
    pub fn resource_resolver_domain(&self) -> AuthorityDomainId {
        self.resource_resolver_domain
    }

    /// Resource resolver epoch used for this concrete identity.
    pub fn resource_resolver_epoch(&self) -> AuthorityEpoch {
        self.resource_resolver_epoch
    }

    /// Resolver attestation grant associated with this concrete resource.
    pub fn resource_attestation_grant_id(&self) -> GrantId {
        self.resource_attestation_grant_id
    }
}

/// Failure while validating or admitting a concrete resource.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceError {
    /// Resource resolver trust-domain/epoch/expiry validation failed.
    Trust(TrustError),
    /// Resolver attestation scope differs from the identity scope.
    ScopeMismatch {
        /// Scope carried by the resolver attestation.
        attested: Scope,
        /// Scope committed by the resource identity.
        identity: Scope,
    },
    /// Resolver attestation was bound to another resource identity.
    BindingMismatch {
        /// Identity binding required by the resource.
        expected: [u8; 32],
        /// Binding carried by the resolver attestation.
        actual: [u8; 32],
    },
}

impl fmt::Display for ResourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Trust(error) => write!(f, "resource resolution trust failed: {error}"),
            Self::ScopeMismatch { .. } => write!(f, "resource attestation scope mismatch"),
            Self::BindingMismatch { .. } => write!(f, "resource identity binding mismatch"),
        }
    }
}

impl std::error::Error for ResourceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Trust(error) => Some(error),
            Self::ScopeMismatch { .. } | Self::BindingMismatch { .. } => None,
        }
    }
}

/// Failure of the retained-handle execution transition.
#[derive(Debug)]
pub enum ResourceExecutionError<E> {
    /// Concrete resource attestation became invalid before execution.
    Resource(ResourceError),
    /// Concrete adapter operation failed.
    Adapter(E),
    /// Strict host execution authority validation failed.
    Trust(TrustError),
}

impl<E: fmt::Display> fmt::Display for ResourceExecutionError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resource(error) => write!(f, "resource validation failed: {error}"),
            Self::Adapter(error) => write!(f, "resource adapter failed: {error}"),
            Self::Trust(error) => write!(f, "execution authority failed: {error}"),
        }
    }
}

impl<E> std::error::Error for ResourceExecutionError<E>
where
    E: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Resource(error) => Some(error),
            Self::Adapter(error) => Some(error),
            Self::Trust(error) => Some(error),
        }
    }
}

fn bind_payload(identity: &ResourceIdentity, payload: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ai-assurance/resource-action-v1\0");
    hash_field(&mut hasher, &identity.binding());
    hash_field(&mut hasher, payload);
    *hasher.finalize().as_bytes()
}

fn compute_resource_binding(
    scope: &Scope,
    kind: &str,
    stable_identity_digest: [u8; 32],
    environment_digest: [u8; 32],
    adapter: &AdapterSchema,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-ai-assurance/resource-v1\0");
    hash_field(&mut hasher, scope.namespace().as_bytes());
    for segment in scope.segments() {
        hash_field(&mut hasher, segment.as_bytes());
    }
    hash_field(&mut hasher, kind.as_bytes());
    hash_field(&mut hasher, &stable_identity_digest);
    hash_field(&mut hasher, &environment_digest);
    hash_field(&mut hasher, adapter.name().as_bytes());
    hash_field(&mut hasher, &adapter.version().to_le_bytes());
    *hasher.finalize().as_bytes()
}

fn hash_field(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn valid_label(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 96
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
}

#[derive(Debug)]
struct ResourceClock {
    last: Mutex<SystemTime>,
}

impl ResourceClock {
    fn new() -> Self {
        Self {
            last: Mutex::new(SystemTime::now()),
        }
    }

    fn now(&self) -> SystemTime {
        let observed = SystemTime::now();
        let mut last = self
            .last
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if observed > *last {
            *last = observed;
        }
        *last
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AuthorityDomain, ObservedOutcome, ResolutionAuthorityDomain, Write,
    };
    use std::sync::atomic::{AtomicBool, Ordering};

    fn scope() -> Scope {
        Scope::new("workspace", ["symthaea", "src"]).unwrap()
    }

    fn identity(stable: u8, environment: u8) -> ResourceIdentity {
        ResourceIdentity::new(
            scope(),
            "worktree-file",
            [stable; 32],
            [environment; 32],
            AdapterSchema::new("test-adapter", 1).unwrap(),
        )
        .unwrap()
    }

    fn runtime(
        resources: &ResourceResolverDomain,
    ) -> (
        AuthorityDomain,
        AuthorityDomain,
        ResolutionAuthorityDomain,
        ResourceRuntime,
    ) {
        let execution = AuthorityDomain::new(PrincipalId::new());
        let observation = AuthorityDomain::new(PrincipalId::new());
        let resolution = ResolutionAuthorityDomain::new(PrincipalId::new());
        let strict = TrustedRuntime::new(
            execution.verifier(),
            observation.verifier(),
            resolution.verifier(),
        );
        let runtime = ResourceRuntime::new(strict, resources.verifier());
        (execution, observation, resolution, runtime)
    }

    #[test]
    fn identity_changes_with_environment_or_adapter_schema() {
        let a = identity(1, 2);
        let b = identity(1, 3);
        let c = ResourceIdentity::new(
            scope(),
            "worktree-file",
            [1; 32],
            [2; 32],
            AdapterSchema::new("test-adapter", 2).unwrap(),
        )
        .unwrap();
        assert_ne!(a.binding(), b.binding());
        assert_ne!(a.binding(), c.binding());
    }

    #[test]
    fn runtime_rejects_resource_from_unrelated_resolver_domain() {
        let expected = ResourceResolverDomain::new(PrincipalId::new());
        let attacker = ResourceResolverDomain::new(PrincipalId::new());
        let (_, _, _, runtime) = runtime(&expected);
        let resource = attacker.resolve((), identity(1, 1), None);

        let result = runtime.admit_resolved::<Write, _>(
            PrincipalId::new(),
            "edit",
            resource,
            b"patch",
        );
        assert!(result.is_err());
    }

    #[test]
    fn distinct_concrete_resources_produce_distinct_action_fingerprints() {
        let resources = ResourceResolverDomain::new(PrincipalId::new());
        let (_, _, _, runtime) = runtime(&resources);
        let actor = PrincipalId::new();
        let a = runtime
            .admit_resolved::<Write, _>(
                actor,
                "edit",
                resources.resolve((), identity(1, 1), None),
                b"same-patch",
            )
            .unwrap();
        let b = runtime
            .admit_resolved::<Write, _>(
                actor,
                "edit",
                resources.resolve((), identity(2, 1), None),
                b"same-patch",
            )
            .unwrap();
        assert_ne!(a.descriptor().fingerprint(), b.descriptor().fingerprint());
    }

    #[test]
    fn grant_for_one_resource_cannot_authorize_another() {
        let resources = ResourceResolverDomain::new(PrincipalId::new());
        let (execution, _, _, runtime) = runtime(&resources);
        let actor = PrincipalId::new();
        let a = runtime
            .admit_resolved::<Write, _>(
                actor,
                "edit",
                resources.resolve((), identity(1, 1), None),
                b"same-patch",
            )
            .unwrap()
            .assess(ActionRisk::Reversible);
        let b = runtime
            .admit_resolved::<Write, _>(
                actor,
                "edit",
                resources.resolve((), identity(2, 1), None),
                b"same-patch",
            )
            .unwrap()
            .assess(ActionRisk::Reversible);
        let grant = execution.issue_bound_one_shot::<Write>(
            actor,
            scope(),
            None,
            a.authorization_binding(),
        );
        assert!(b.authorize(grant).is_err());
    }

    #[test]
    fn resolver_revocation_blocks_closure_before_side_effect() {
        let resources = ResourceResolverDomain::new(PrincipalId::new());
        let (execution, _, _, runtime) = runtime(&resources);
        let actor = PrincipalId::new();
        let action = runtime
            .admit_resolved::<Write, _>(
                actor,
                "edit",
                resources.resolve(0_u64, identity(1, 1), None),
                b"patch",
            )
            .unwrap()
            .assess(ActionRisk::Reversible);
        let grant = execution.issue_bound_one_shot::<Write>(
            actor,
            scope(),
            None,
            action.authorization_binding(),
        );
        let action = action.authorize(grant).unwrap();
        resources.revoke_all().unwrap();
        let called = AtomicBool::new(false);
        let result = action.execute_with(|handle| -> Result<[u8; 32], &'static str> {
            called.store(true, Ordering::SeqCst);
            *handle += 1;
            Ok([9; 32])
        });
        assert!(result.is_err());
        assert!(!called.load(Ordering::SeqCst));
    }

    #[test]
    fn retained_handle_flows_through_exact_evidence_path() {
        let resources = ResourceResolverDomain::new(PrincipalId::new());
        let (execution, observation, resolution, runtime) = runtime(&resources);
        let actor = PrincipalId::new();
        let observer = PrincipalId::new();
        let resolver = PrincipalId::new();
        let action = runtime
            .admit_resolved::<Write, _>(
                actor,
                "edit",
                resources.resolve(10_u64, identity(4, 5), None),
                b"patch",
            )
            .unwrap()
            .assess(ActionRisk::Reversible);
        let execution_grant = execution.issue_bound_one_shot::<Write>(
            actor,
            scope(),
            None,
            action.authorization_binding(),
        );
        let action = action
            .authorize(execution_grant)
            .unwrap()
            .execute_with(|handle| -> Result<[u8; 32], &'static str> {
                *handle += 1;
                Ok([1; 32])
            })
            .unwrap();
        let observer_grant = observation.issue_bound_one_shot::<crate::Observe>(
            observer,
            scope(),
            None,
            action.observation_binding(),
        );
        let action = action
            .observe(
                observer_grant,
                Observation::new(ObservedOutcome::Success, [2; 32]),
            )
            .unwrap();
        let decision = ResolutionDecision::Confirmed;
        let resolution_grant = resolution.issue_bound_one_shot(
            resolver,
            scope(),
            None,
            action.resolution_binding(decision),
        );
        let (resolved, receipt) = action.resolve(resolution_grant, decision).unwrap();
        assert_eq!(receipt.resource_identity().binding(), identity(4, 5).binding());
        let resource = resolved.into_resource();
        assert_eq!(resource.handle, 11);
    }
}
