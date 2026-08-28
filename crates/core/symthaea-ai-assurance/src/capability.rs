// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Capability and authority primitives.
//!
//! A trusted host owns an [`AuthorityRoot`] and issues scoped capabilities to
//! less-trusted components. Capability values are affine Rust values (neither
//! `Copy` nor `Clone`) and delegation can only preserve or narrow scope and
//! expiry. Exact-action transitions use [`BoundOneShotCapability`] so approval
//! cannot be substituted onto a different action within the same scope.

use std::fmt;
use std::marker::PhantomData;
use std::time::SystemTime;
use uuid::Uuid;

/// Stable identity of a principal that can issue or receive authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrincipalId(Uuid);

impl PrincipalId {
    /// Construct a fresh principal identity.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Return the underlying UUID.
    pub fn as_uuid(self) -> Uuid {
        self.0
    }
}

impl Default for PrincipalId {
    fn default() -> Self {
        Self::new()
    }
}

/// Stable identity of one authority grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GrantId(Uuid);

impl GrantId {
    fn fresh() -> Self {
        Self(Uuid::new_v4())
    }

    /// Return the underlying UUID.
    pub fn as_uuid(self) -> Uuid {
        self.0
    }
}

/// A deterministic hierarchical resource scope.
///
/// Scope components are logical names rather than host filesystem paths. This
/// keeps the assurance kernel free of ambient filesystem authority and makes
/// containment a purely lexical operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Scope {
    namespace: String,
    segments: Vec<String>,
}

impl Scope {
    /// Construct a scope from a namespace and path-like logical segments.
    pub fn new<I, S>(namespace: impl Into<String>, segments: I) -> Result<Self, ScopeError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let namespace = namespace.into();
        if !valid_component(&namespace) {
            return Err(ScopeError::InvalidNamespace(namespace));
        }

        let segments: Vec<String> = segments.into_iter().map(Into::into).collect();
        for segment in &segments {
            if !valid_component(segment) {
                return Err(ScopeError::InvalidSegment(segment.clone()));
            }
        }

        Ok(Self {
            namespace,
            segments,
        })
    }

    /// Construct the root of a logical namespace.
    pub fn namespace_root(namespace: impl Into<String>) -> Result<Self, ScopeError> {
        Self::new(namespace, std::iter::empty::<String>())
    }

    /// Return the namespace.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Return the logical scope segments.
    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    /// Return true when `candidate` is equal to or narrower than this scope.
    pub fn contains(&self, candidate: &Self) -> bool {
        self.namespace == candidate.namespace
            && candidate.segments.len() >= self.segments.len()
            && candidate.segments.starts_with(&self.segments)
    }
}

fn valid_component(component: &str) -> bool {
    !component.is_empty()
        && component != "."
        && component != ".."
        && !component.contains('/')
        && !component.contains('\\')
        && !component.contains('\0')
}

/// Scope construction failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScopeError {
    /// Namespace is empty or contains a forbidden path component.
    InvalidNamespace(String),
    /// A logical scope segment is invalid.
    InvalidSegment(String),
}

impl fmt::Display for ScopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidNamespace(value) => write!(f, "invalid scope namespace: {value:?}"),
            Self::InvalidSegment(value) => write!(f, "invalid scope segment: {value:?}"),
        }
    }
}

impl std::error::Error for ScopeError {}

/// Marker trait for a class of authority.
pub trait CapabilityKind: private::Sealed + 'static {
    /// Stable human-readable capability name.
    const NAME: &'static str;
}

macro_rules! capability_kinds {
    ($($name:ident => $label:literal),+ $(,)?) => {
        $(
            #[doc = concat!("Capability marker for `", $label, "` authority.")]
            #[derive(Debug)]
            pub struct $name;

            impl private::Sealed for $name {}
            impl CapabilityKind for $name {
                const NAME: &'static str = $label;
            }
        )+
    };
}

capability_kinds! {
    Read => "read",
    Write => "write",
    Execute => "execute",
    Network => "network",
    Deploy => "deploy",
    UpdateModel => "update-model",
    Observe => "observe",
}

mod private {
    pub trait Sealed {}
}

/// Immutable metadata shared by capability grants.
///
/// Cloning metadata does **not** clone authority: executable transitions accept
/// only the opaque capability wrapper types, never `GrantMetadata` itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrantMetadata {
    grant_id: GrantId,
    parent_grant_id: Option<GrantId>,
    issuer: PrincipalId,
    subject: PrincipalId,
    scope: Scope,
    expires_at: Option<SystemTime>,
    delegation_depth: u16,
}

impl GrantMetadata {
    /// Unique id of this grant.
    pub fn grant_id(&self) -> GrantId {
        self.grant_id
    }

    /// Grant from which this authority was delegated, if any.
    pub fn parent_grant_id(&self) -> Option<GrantId> {
        self.parent_grant_id
    }

    /// Principal that issued this grant.
    pub fn issuer(&self) -> PrincipalId {
        self.issuer
    }

    /// Principal that holds this grant.
    pub fn subject(&self) -> PrincipalId {
        self.subject
    }

    /// Resource scope covered by this grant.
    pub fn scope(&self) -> &Scope {
        &self.scope
    }

    /// Optional expiry instant.
    pub fn expires_at(&self) -> Option<SystemTime> {
        self.expires_at
    }

    /// Number of delegation hops from the root issuer.
    pub fn delegation_depth(&self) -> u16 {
        self.delegation_depth
    }

    /// Return true when this grant is valid at `now`.
    pub fn is_valid_at(&self, now: SystemTime) -> bool {
        self.expires_at.is_none_or(|expiry| now <= expiry)
    }
}

/// Reusable scoped authority of kind `K`.
///
/// The type intentionally implements neither `Copy` nor `Clone`.
#[derive(Debug)]
pub struct Capability<K: CapabilityKind> {
    metadata: GrantMetadata,
    _kind: PhantomData<K>,
}

impl<K: CapabilityKind> Capability<K> {
    /// Inspect immutable grant metadata.
    pub fn metadata(&self) -> &GrantMetadata {
        &self.metadata
    }

    /// Validate this grant at a particular time.
    pub fn validate_at(&self, now: SystemTime) -> Result<(), GrantError> {
        validate_metadata(&self.metadata, now)
    }

    /// Delegate equal or narrower authority to `subject`.
    ///
    /// Delegation cannot change the capability kind, widen the resource scope,
    /// or extend expiry beyond the parent grant.
    pub fn delegate(
        &self,
        subject: PrincipalId,
        scope: Scope,
        expires_at: Option<SystemTime>,
        now: SystemTime,
    ) -> Result<Self, GrantError> {
        self.validate_at(now)?;
        let metadata = delegated_metadata(&self.metadata, subject, scope, expires_at)?;
        Ok(Self {
            metadata,
            _kind: PhantomData,
        })
    }
}

/// One-shot scoped authority not bound to a particular transition digest.
///
/// This is useful for host protocols that require single-use authority but do
/// not yet have a concrete action digest. Exact action execution and observation
/// use [`BoundOneShotCapability`] instead.
#[derive(Debug)]
pub struct OneShotCapability<K: CapabilityKind> {
    metadata: GrantMetadata,
    _kind: PhantomData<K>,
}

impl<K: CapabilityKind> OneShotCapability<K> {
    /// Inspect immutable grant metadata.
    pub fn metadata(&self) -> &GrantMetadata {
        &self.metadata
    }

    /// Validate this grant at a particular time.
    pub fn validate_at(&self, now: SystemTime) -> Result<(), GrantError> {
        validate_metadata(&self.metadata, now)
    }
}

/// One-shot authority bound to an exact domain-separated transition digest.
///
/// The safe API exposes no way to alter the binding and the type intentionally
/// implements neither `Copy` nor `Clone`.
#[derive(Debug)]
pub struct BoundOneShotCapability<K: CapabilityKind> {
    metadata: GrantMetadata,
    binding: [u8; 32],
    _kind: PhantomData<K>,
}

impl<K: CapabilityKind> BoundOneShotCapability<K> {
    /// Inspect immutable grant metadata.
    pub fn metadata(&self) -> &GrantMetadata {
        &self.metadata
    }

    /// Exact transition digest this grant can authorize.
    pub fn binding(&self) -> [u8; 32] {
        self.binding
    }

    /// Validate this grant at a particular time.
    pub fn validate_at(&self, now: SystemTime) -> Result<(), GrantError> {
        validate_metadata(&self.metadata, now)
    }

    pub(crate) fn into_parts(self) -> (GrantMetadata, [u8; 32]) {
        (self.metadata, self.binding)
    }
}

/// Trusted root that can mint initial capability grants.
///
/// Holding this value is part of the host trusted computing base. Agent code
/// should receive only the capabilities it needs, never an `AuthorityRoot`.
#[derive(Debug)]
pub struct AuthorityRoot {
    principal: PrincipalId,
}

impl AuthorityRoot {
    /// Create a root for a trusted host principal.
    pub fn new(principal: PrincipalId) -> Self {
        Self { principal }
    }

    /// Root principal identity.
    pub fn principal(&self) -> PrincipalId {
        self.principal
    }

    /// Mint reusable authority.
    pub fn issue<K: CapabilityKind>(
        &self,
        subject: PrincipalId,
        scope: Scope,
        expires_at: Option<SystemTime>,
    ) -> Capability<K> {
        Capability {
            metadata: root_metadata(self.principal, subject, scope, expires_at),
            _kind: PhantomData,
        }
    }

    /// Mint unbound one-shot authority.
    pub fn issue_one_shot<K: CapabilityKind>(
        &self,
        subject: PrincipalId,
        scope: Scope,
        expires_at: Option<SystemTime>,
    ) -> OneShotCapability<K> {
        OneShotCapability {
            metadata: root_metadata(self.principal, subject, scope, expires_at),
            _kind: PhantomData,
        }
    }

    /// Mint one-shot authority for exactly one transition digest.
    pub fn issue_bound_one_shot<K: CapabilityKind>(
        &self,
        subject: PrincipalId,
        scope: Scope,
        expires_at: Option<SystemTime>,
        binding: [u8; 32],
    ) -> BoundOneShotCapability<K> {
        BoundOneShotCapability {
            metadata: root_metadata(self.principal, subject, scope, expires_at),
            binding,
            _kind: PhantomData,
        }
    }
}

fn validate_metadata(metadata: &GrantMetadata, now: SystemTime) -> Result<(), GrantError> {
    if metadata.is_valid_at(now) {
        Ok(())
    } else {
        Err(GrantError::Expired {
            grant_id: metadata.grant_id,
        })
    }
}

fn root_metadata(
    issuer: PrincipalId,
    subject: PrincipalId,
    scope: Scope,
    expires_at: Option<SystemTime>,
) -> GrantMetadata {
    GrantMetadata {
        grant_id: GrantId::fresh(),
        parent_grant_id: None,
        issuer,
        subject,
        scope,
        expires_at,
        delegation_depth: 0,
    }
}

fn delegated_metadata(
    parent: &GrantMetadata,
    subject: PrincipalId,
    scope: Scope,
    expires_at: Option<SystemTime>,
) -> Result<GrantMetadata, GrantError> {
    if !parent.scope.contains(&scope) {
        return Err(GrantError::ScopeWidening {
            parent: parent.scope.clone(),
            requested: scope,
        });
    }

    if let Some(parent_expiry) = parent.expires_at {
        match expires_at {
            Some(child_expiry) if child_expiry <= parent_expiry => {}
            _ => {
                return Err(GrantError::ExpiryWidening {
                    parent: Some(parent_expiry),
                    requested: expires_at,
                });
            }
        }
    }

    let delegation_depth = parent
        .delegation_depth
        .checked_add(1)
        .ok_or(GrantError::DelegationDepthExceeded)?;

    Ok(GrantMetadata {
        grant_id: GrantId::fresh(),
        parent_grant_id: Some(parent.grant_id),
        issuer: parent.subject,
        subject,
        scope,
        expires_at,
        delegation_depth,
    })
}

/// Grant validation or delegation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrantError {
    /// Grant expired before use.
    Expired {
        /// Expired grant identity.
        grant_id: GrantId,
    },
    /// Child scope was not contained by the parent scope.
    ScopeWidening {
        /// Parent authority scope.
        parent: Scope,
        /// Requested child scope.
        requested: Scope,
    },
    /// Child expiry was absent or later than a finite parent expiry.
    ExpiryWidening {
        /// Parent expiry bound.
        parent: Option<SystemTime>,
        /// Requested child expiry bound.
        requested: Option<SystemTime>,
    },
    /// Delegation depth overflowed the bounded representation.
    DelegationDepthExceeded,
}

impl fmt::Display for GrantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Expired { grant_id } => write!(f, "grant {:?} is expired", grant_id.as_uuid()),
            Self::ScopeWidening { .. } => write!(f, "delegation would widen authority scope"),
            Self::ExpiryWidening { .. } => write!(f, "delegation would widen authority expiry"),
            Self::DelegationDepthExceeded => write!(f, "delegation depth exceeded"),
        }
    }
}

impl std::error::Error for GrantError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn scope(parts: &[&str]) -> Scope {
        Scope::new("workspace", parts.iter().copied()).unwrap()
    }

    #[test]
    fn logical_scope_rejects_parent_traversal() {
        assert!(matches!(
            Scope::new("workspace", ["src", "..", "secret"]),
            Err(ScopeError::InvalidSegment(_))
        ));
    }

    #[test]
    fn containment_is_namespace_and_prefix_bound() {
        let parent = scope(&["symthaea"]);
        assert!(parent.contains(&scope(&["symthaea", "src"])));
        assert!(!parent.contains(&scope(&["mycelix"])));
        assert!(!parent.contains(&Scope::new("network", ["symthaea"]).unwrap()));
    }

    #[test]
    fn delegation_can_narrow_but_not_widen_scope() {
        let root = AuthorityRoot::new(PrincipalId::new());
        let holder = PrincipalId::new();
        let child = PrincipalId::new();
        let grant = root.issue::<Read>(holder, scope(&["symthaea"]), None);

        let narrowed = grant
            .delegate(child, scope(&["symthaea", "src"]), None, SystemTime::now())
            .unwrap();
        assert_eq!(
            narrowed.metadata().parent_grant_id(),
            Some(grant.metadata().grant_id())
        );
        assert_eq!(narrowed.metadata().issuer(), holder);

        let widened = grant.delegate(
            child,
            Scope::namespace_root("workspace").unwrap(),
            None,
            SystemTime::now(),
        );
        assert!(matches!(widened, Err(GrantError::ScopeWidening { .. })));
    }

    #[test]
    fn finite_parent_expiry_cannot_be_removed_or_extended() {
        let now = SystemTime::now();
        let parent_expiry = now + Duration::from_secs(60);
        let root = AuthorityRoot::new(PrincipalId::new());
        let grant = root.issue::<Write>(
            PrincipalId::new(),
            scope(&["scratch"]),
            Some(parent_expiry),
        );

        assert!(matches!(
            grant.delegate(PrincipalId::new(), scope(&["scratch"]), None, now),
            Err(GrantError::ExpiryWidening { .. })
        ));
        assert!(matches!(
            grant.delegate(
                PrincipalId::new(),
                scope(&["scratch"]),
                Some(parent_expiry + Duration::from_secs(1)),
                now,
            ),
            Err(GrantError::ExpiryWidening { .. })
        ));
        assert!(
            grant
                .delegate(
                    PrincipalId::new(),
                    scope(&["scratch", "candidate"]),
                    Some(now + Duration::from_secs(30)),
                    now,
                )
                .is_ok()
        );
    }

    #[test]
    fn expired_grant_fails_closed() {
        let now = SystemTime::now();
        let root = AuthorityRoot::new(PrincipalId::new());
        let grant = root.issue::<Execute>(
            PrincipalId::new(),
            scope(&["tests"]),
            Some(now - Duration::from_secs(1)),
        );
        assert!(matches!(
            grant.validate_at(now),
            Err(GrantError::Expired { .. })
        ));
    }

    #[test]
    fn bound_grant_binding_is_immutable_data() {
        let root = AuthorityRoot::new(PrincipalId::new());
        let binding = [42_u8; 32];
        let grant = root.issue_bound_one_shot::<Write>(
            PrincipalId::new(),
            scope(&["scratch"]),
            None,
            binding,
        );
        assert_eq!(grant.binding(), binding);
    }

    proptest::proptest! {
        #[test]
        fn adding_segments_never_widens_scope(extra in proptest::collection::vec("[a-z]{1,8}", 0..8)) {
            let parent = scope(&["root"]);
            let mut child_segments = vec!["root".to_string()];
            child_segments.extend(extra);
            let child = Scope::new("workspace", child_segments).unwrap();
            proptest::prop_assert!(parent.contains(&child));
        }
    }
}
