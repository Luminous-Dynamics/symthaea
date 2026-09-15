//! Immutable SCI-014 scientific-view authority-surface identities.
//!
//! This crate deliberately stops before live currentness.
//!
//! It provides:
//! - a closed V1 vocabulary for mutable scientific authority roles;
//! - content-addressed immutable scientific-view semantic profiles;
//! - deployment-scoped exact role-to-source bindings;
//! - anti-ABA source occurrence identities.
//!
//! It does **not** decide which profile/binding is current, capture a coherent
//! multi-root view, persist a high-water mark, select an external witness,
//! establish scientific support/truth, or grant action authority.
//!
//! Architecture: SCI-014P / #3338.

#![forbid(unsafe_code)]

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Canonical commitment algorithm for all V1 identities in this crate.
pub const COMMITMENT_ALGORITHM_V1: &str = "sha256";

const MAX_STABLE_ID_BYTES: usize = 192;

const ROLE_SEMANTIC_DOMAIN: &[u8] = b"symthaea.science.view.role-semantic.v1";
const PROFILE_DOMAIN: &[u8] = b"symthaea.science.view.semantic-profile.v1";
const SOURCE_BINDING_DOMAIN: &[u8] = b"symthaea.science.view.source-binding.v1";
const SOURCE_ROSTER_DOMAIN: &[u8] = b"symthaea.science.view.source-roster.v1";
const DEPLOYMENT_BINDING_DOMAIN: &[u8] = b"symthaea.science.view.deployment-binding.v1";
const SOURCE_OCCURRENCE_DOMAIN: &[u8] = b"symthaea.science.view.source-occurrence.v1";

/// A raw 32-byte content commitment.
///
/// Constructing one from bytes does not qualify the object those bytes purport
/// to identify. It is only a value used by the checked SCI-014 identities in
/// this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Commitment32([u8; 32]);

impl Commitment32 {
    pub const ZERO: Self = Self([0; 32]);

    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn is_zero(self) -> bool {
        self == Self::ZERO
    }
}

/// Closed V1 vocabulary for mutable scientific authority roles.
///
/// The numeric codes are protocol identity. Reordering source declarations in
/// Rust must not change their canonical meaning.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScientificAuthorityRoleV1 {
    ResearchSemanticHead = 1,
    EvidenceAdmissionHead = 2,
    SubjectSelectionHead = 3,
    VerifierPolicyHead = 4,
    ScientificViewSelectionLifecycleHead = 5,
    DiscoveryCompletenessHead = 6,
}

impl ScientificAuthorityRoleV1 {
    pub const ALL: [Self; 6] = [
        Self::ResearchSemanticHead,
        Self::EvidenceAdmissionHead,
        Self::SubjectSelectionHead,
        Self::VerifierPolicyHead,
        Self::ScientificViewSelectionLifecycleHead,
        Self::DiscoveryCompletenessHead,
    ];

    pub const fn code(self) -> u8 {
        self as u8
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScientificViewProfileError {
    InvalidStableId { field: &'static str },
    ZeroCommitment { field: &'static str },
    EmptyRoleRoster,
    DuplicateRole(ScientificAuthorityRoleV1),
    MissingRequiredRole(ScientificAuthorityRoleV1),
    ExtraRole(ScientificAuthorityRoleV1),
    ProvisioningEpochZero,
    RoleNotBound(ScientificAuthorityRoleV1),
    DeploymentBindingMismatch,
    SourceIdentityMismatch,
    ProvisioningEpochMismatch,
    LineageOverflow,
    NoOpOccurrence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleSemanticRevisionV1 {
    role: ScientificAuthorityRoleV1,
    semantic_commitment: Commitment32,
    commitment: Commitment32,
}

impl RoleSemanticRevisionV1 {
    pub fn new(
        role: ScientificAuthorityRoleV1,
        semantic_commitment: Commitment32,
    ) -> Result<Self, ScientificViewProfileError> {
        require_nonzero("role_semantic_commitment", semantic_commitment)?;

        let commitment = hash_with(ROLE_SEMANTIC_DOMAIN, |hasher| {
            put_u8(hasher, role.code());
            put_commitment(hasher, semantic_commitment);
        });

        Ok(Self {
            role,
            semantic_commitment,
            commitment,
        })
    }

    pub const fn role(&self) -> ScientificAuthorityRoleV1 {
        self.role
    }

    pub const fn semantic_commitment(&self) -> Commitment32 {
        self.semantic_commitment
    }

    pub const fn commitment(&self) -> Commitment32 {
        self.commitment
    }
}

/// Immutable, content-addressed scientific-view semantics.
///
/// This is evidence/configuration identity only. It has no `is_current` bit and
/// cannot activate itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScientificViewSemanticProfileV1 {
    view_namespace: String,
    profile_family: String,
    role_semantics: Vec<RoleSemanticRevisionV1>,
    discovery_scope_commitment: Commitment32,
    closure_policy_commitment: Commitment32,
    capture_policy_commitment: Commitment32,
    use_policy_family_commitment: Commitment32,
    commitment: Commitment32,
}

impl ScientificViewSemanticProfileV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        view_namespace: impl Into<String>,
        profile_family: impl Into<String>,
        mut role_semantics: Vec<RoleSemanticRevisionV1>,
        discovery_scope_commitment: Commitment32,
        closure_policy_commitment: Commitment32,
        capture_policy_commitment: Commitment32,
        use_policy_family_commitment: Commitment32,
    ) -> Result<Self, ScientificViewProfileError> {
        let view_namespace = view_namespace.into();
        let profile_family = profile_family.into();

        validate_stable_id("view_namespace", &view_namespace)?;
        validate_stable_id("profile_family", &profile_family)?;
        require_nonzero("discovery_scope_commitment", discovery_scope_commitment)?;
        require_nonzero("closure_policy_commitment", closure_policy_commitment)?;
        require_nonzero("capture_policy_commitment", capture_policy_commitment)?;
        require_nonzero(
            "use_policy_family_commitment",
            use_policy_family_commitment,
        )?;

        if role_semantics.is_empty() {
            return Err(ScientificViewProfileError::EmptyRoleRoster);
        }

        role_semantics.sort_by_key(|entry| entry.role.code());
        for pair in role_semantics.windows(2) {
            if pair[0].role == pair[1].role {
                return Err(ScientificViewProfileError::DuplicateRole(pair[0].role));
            }
        }

        let commitment = hash_with(PROFILE_DOMAIN, |hasher| {
            put_text(hasher, &view_namespace);
            put_text(hasher, &profile_family);
            put_u64(hasher, role_semantics.len() as u64);
            for entry in &role_semantics {
                put_u8(hasher, entry.role.code());
                put_commitment(hasher, entry.semantic_commitment);
                put_commitment(hasher, entry.commitment);
            }
            put_commitment(hasher, discovery_scope_commitment);
            put_commitment(hasher, closure_policy_commitment);
            put_commitment(hasher, capture_policy_commitment);
            put_commitment(hasher, use_policy_family_commitment);
        });

        Ok(Self {
            view_namespace,
            profile_family,
            role_semantics,
            discovery_scope_commitment,
            closure_policy_commitment,
            capture_policy_commitment,
            use_policy_family_commitment,
            commitment,
        })
    }

    pub fn view_namespace(&self) -> &str {
        &self.view_namespace
    }

    pub fn profile_family(&self) -> &str {
        &self.profile_family
    }

    pub fn role_semantics(&self) -> &[RoleSemanticRevisionV1] {
        &self.role_semantics
    }

    pub fn required_roles(
        &self,
    ) -> impl ExactSizeIterator<Item = ScientificAuthorityRoleV1> + '_ {
        self.role_semantics.iter().map(|entry| entry.role)
    }

    pub fn requires_role(&self, role: ScientificAuthorityRoleV1) -> bool {
        self.role_semantics
            .binary_search_by_key(&role.code(), |entry| entry.role.code())
            .is_ok()
    }

    pub const fn discovery_scope_commitment(&self) -> Commitment32 {
        self.discovery_scope_commitment
    }

    pub const fn closure_policy_commitment(&self) -> Commitment32 {
        self.closure_policy_commitment
    }

    pub const fn capture_policy_commitment(&self) -> Commitment32 {
        self.capture_policy_commitment
    }

    pub const fn use_policy_family_commitment(&self) -> Commitment32 {
        self.use_policy_family_commitment
    }

    pub const fn commitment(&self) -> Commitment32 {
        self.commitment
    }
}

/// Candidate binding of one semantic role to one deployment-local source.
///
/// This record is not currentness authority. `provisioning_epoch` is explicit
/// anti-ABA identity: reprovisioning the same logical source requires a new
/// epoch even when the source initially exposes byte-identical state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthoritySourceBindingV1 {
    role: ScientificAuthorityRoleV1,
    source_identity: String,
    provisioning_epoch: u64,
    qualification_profile_commitment: Commitment32,
    commitment: Commitment32,
}

impl AuthoritySourceBindingV1 {
    pub fn new(
        role: ScientificAuthorityRoleV1,
        source_identity: impl Into<String>,
        provisioning_epoch: u64,
        qualification_profile_commitment: Commitment32,
    ) -> Result<Self, ScientificViewProfileError> {
        let source_identity = source_identity.into();
        validate_stable_id("source_identity", &source_identity)?;
        if provisioning_epoch == 0 {
            return Err(ScientificViewProfileError::ProvisioningEpochZero);
        }
        require_nonzero(
            "qualification_profile_commitment",
            qualification_profile_commitment,
        )?;

        let commitment = hash_with(SOURCE_BINDING_DOMAIN, |hasher| {
            put_u8(hasher, role.code());
            put_text(hasher, &source_identity);
            put_u64(hasher, provisioning_epoch);
            put_commitment(hasher, qualification_profile_commitment);
        });

        Ok(Self {
            role,
            source_identity,
            provisioning_epoch,
            qualification_profile_commitment,
            commitment,
        })
    }

    pub const fn role(&self) -> ScientificAuthorityRoleV1 {
        self.role
    }

    pub fn source_identity(&self) -> &str {
        &self.source_identity
    }

    pub const fn provisioning_epoch(&self) -> u64 {
        self.provisioning_epoch
    }

    pub const fn qualification_profile_commitment(&self) -> Commitment32 {
        self.qualification_profile_commitment
    }

    pub const fn commitment(&self) -> Commitment32 {
        self.commitment
    }
}

/// Exact deployment-scoped binding for every role required by a semantic
/// profile.
///
/// Construction requires an exact role census: no missing role, no extra role,
/// and no duplicate role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScientificViewDeploymentBindingV1 {
    deployment_id: String,
    view_namespace: String,
    semantic_profile_commitment: Commitment32,
    source_bindings: Vec<AuthoritySourceBindingV1>,
    source_roster_commitment: Commitment32,
    commitment: Commitment32,
}

impl ScientificViewDeploymentBindingV1 {
    pub fn new(
        deployment_id: impl Into<String>,
        profile: &ScientificViewSemanticProfileV1,
        mut source_bindings: Vec<AuthoritySourceBindingV1>,
    ) -> Result<Self, ScientificViewProfileError> {
        let deployment_id = deployment_id.into();
        validate_stable_id("deployment_id", &deployment_id)?;

        source_bindings.sort_by_key(|binding| binding.role.code());

        for pair in source_bindings.windows(2) {
            if pair[0].role == pair[1].role {
                return Err(ScientificViewProfileError::DuplicateRole(pair[0].role));
            }
        }

        for binding in &source_bindings {
            if !profile.requires_role(binding.role) {
                return Err(ScientificViewProfileError::ExtraRole(binding.role));
            }
        }

        let by_role: BTreeMap<_, _> = source_bindings
            .iter()
            .map(|binding| (binding.role, binding))
            .collect();

        for required in profile.required_roles() {
            if !by_role.contains_key(&required) {
                return Err(ScientificViewProfileError::MissingRequiredRole(required));
            }
        }

        let source_roster_commitment = hash_with(SOURCE_ROSTER_DOMAIN, |hasher| {
            put_text(hasher, &deployment_id);
            put_commitment(hasher, profile.commitment);
            put_u64(hasher, source_bindings.len() as u64);
            for binding in &source_bindings {
                put_u8(hasher, binding.role.code());
                put_commitment(hasher, binding.commitment);
            }
        });

        let commitment = hash_with(DEPLOYMENT_BINDING_DOMAIN, |hasher| {
            put_text(hasher, &deployment_id);
            put_text(hasher, profile.view_namespace());
            put_commitment(hasher, profile.commitment);
            put_commitment(hasher, source_roster_commitment);
        });

        Ok(Self {
            deployment_id,
            view_namespace: profile.view_namespace.clone(),
            semantic_profile_commitment: profile.commitment,
            source_bindings,
            source_roster_commitment,
            commitment,
        })
    }

    pub fn deployment_id(&self) -> &str {
        &self.deployment_id
    }

    pub fn view_namespace(&self) -> &str {
        &self.view_namespace
    }

    pub const fn semantic_profile_commitment(&self) -> Commitment32 {
        self.semantic_profile_commitment
    }

    pub fn source_bindings(&self) -> &[AuthoritySourceBindingV1] {
        &self.source_bindings
    }

    pub fn source_for_role(
        &self,
        role: ScientificAuthorityRoleV1,
    ) -> Option<&AuthoritySourceBindingV1> {
        self.source_bindings
            .binary_search_by_key(&role.code(), |binding| binding.role.code())
            .ok()
            .map(|index| &self.source_bindings[index])
    }

    pub const fn source_roster_commitment(&self) -> Commitment32 {
        self.source_roster_commitment
    }

    pub const fn commitment(&self) -> Commitment32 {
        self.commitment
    }
}

/// One exact occurrence of a deployment-bound mutable authority source.
///
/// Occurrence identity is stronger than state bytes. A valid successor is
/// exactly adjacent, names the exact predecessor occurrence, remains under the
/// same deployment binding/source/provisioning epoch, and changes the complete
/// scientifically material state commitment.
///
/// Source replacement/reprovisioning is intentionally *not* a successor here;
/// it requires a new deployment binding plus the separate migration/control-
/// plane theorem specified by SCI-014P.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthoritySourceOccurrenceV1 {
    deployment_binding_commitment: Commitment32,
    role: ScientificAuthorityRoleV1,
    source_identity: String,
    provisioning_epoch: u64,
    lineage_position: u64,
    predecessor_occurrence: Option<Commitment32>,
    state_commitment: Commitment32,
    commitment: Commitment32,
}

impl AuthoritySourceOccurrenceV1 {
    pub fn genesis(
        binding: &ScientificViewDeploymentBindingV1,
        role: ScientificAuthorityRoleV1,
        state_commitment: Commitment32,
    ) -> Result<Self, ScientificViewProfileError> {
        require_nonzero("state_commitment", state_commitment)?;
        let source = binding
            .source_for_role(role)
            .ok_or(ScientificViewProfileError::RoleNotBound(role))?;

        Ok(Self::build(
            binding,
            source,
            1,
            None,
            state_commitment,
        ))
    }

    pub fn successor(
        &self,
        binding: &ScientificViewDeploymentBindingV1,
        next_state_commitment: Commitment32,
    ) -> Result<Self, ScientificViewProfileError> {
        require_nonzero("state_commitment", next_state_commitment)?;

        if binding.commitment != self.deployment_binding_commitment {
            return Err(ScientificViewProfileError::DeploymentBindingMismatch);
        }

        let source = binding
            .source_for_role(self.role)
            .ok_or(ScientificViewProfileError::RoleNotBound(self.role))?;

        if source.source_identity != self.source_identity {
            return Err(ScientificViewProfileError::SourceIdentityMismatch);
        }
        if source.provisioning_epoch != self.provisioning_epoch {
            return Err(ScientificViewProfileError::ProvisioningEpochMismatch);
        }
        if next_state_commitment == self.state_commitment {
            return Err(ScientificViewProfileError::NoOpOccurrence);
        }

        let next_position = self
            .lineage_position
            .checked_add(1)
            .ok_or(ScientificViewProfileError::LineageOverflow)?;

        Ok(Self::build(
            binding,
            source,
            next_position,
            Some(self.commitment),
            next_state_commitment,
        ))
    }

    fn build(
        binding: &ScientificViewDeploymentBindingV1,
        source: &AuthoritySourceBindingV1,
        lineage_position: u64,
        predecessor_occurrence: Option<Commitment32>,
        state_commitment: Commitment32,
    ) -> Self {
        let commitment = hash_with(SOURCE_OCCURRENCE_DOMAIN, |hasher| {
            put_commitment(hasher, binding.commitment);
            put_u8(hasher, source.role.code());
            put_text(hasher, &source.source_identity);
            put_u64(hasher, source.provisioning_epoch);
            put_u64(hasher, lineage_position);
            put_optional_commitment(hasher, predecessor_occurrence);
            put_commitment(hasher, state_commitment);
        });

        Self {
            deployment_binding_commitment: binding.commitment,
            role: source.role,
            source_identity: source.source_identity.clone(),
            provisioning_epoch: source.provisioning_epoch,
            lineage_position,
            predecessor_occurrence,
            state_commitment,
            commitment,
        }
    }

    pub const fn deployment_binding_commitment(&self) -> Commitment32 {
        self.deployment_binding_commitment
    }

    pub const fn role(&self) -> ScientificAuthorityRoleV1 {
        self.role
    }

    pub fn source_identity(&self) -> &str {
        &self.source_identity
    }

    pub const fn provisioning_epoch(&self) -> u64 {
        self.provisioning_epoch
    }

    pub const fn lineage_position(&self) -> u64 {
        self.lineage_position
    }

    pub const fn predecessor_occurrence(&self) -> Option<Commitment32> {
        self.predecessor_occurrence
    }

    pub const fn state_commitment(&self) -> Commitment32 {
        self.state_commitment
    }

    pub const fn commitment(&self) -> Commitment32 {
        self.commitment
    }
}

fn validate_stable_id(
    field: &'static str,
    value: &str,
) -> Result<(), ScientificViewProfileError> {
    if value.is_empty()
        || value.len() > MAX_STABLE_ID_BYTES
        || !value.is_ascii()
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'.' | b'_' | b'-' | b'/' | b':')
        })
    {
        return Err(ScientificViewProfileError::InvalidStableId { field });
    }
    Ok(())
}

fn require_nonzero(
    field: &'static str,
    commitment: Commitment32,
) -> Result<(), ScientificViewProfileError> {
    if commitment.is_zero() {
        return Err(ScientificViewProfileError::ZeroCommitment { field });
    }
    Ok(())
}

fn hash_with(domain: &[u8], write_fields: impl FnOnce(&mut Sha256)) -> Commitment32 {
    let mut hasher = Sha256::new();
    put_bytes(&mut hasher, domain);
    write_fields(&mut hasher);
    Commitment32::from_bytes(hasher.finalize().into())
}

fn put_u8(hasher: &mut Sha256, value: u8) {
    hasher.update([value]);
}

fn put_u64(hasher: &mut Sha256, value: u64) {
    hasher.update(value.to_be_bytes());
}

fn put_bytes(hasher: &mut Sha256, value: &[u8]) {
    put_u64(hasher, value.len() as u64);
    hasher.update(value);
}

fn put_text(hasher: &mut Sha256, value: &str) {
    put_bytes(hasher, value.as_bytes());
}

fn put_commitment(hasher: &mut Sha256, value: Commitment32) {
    hasher.update(value.as_bytes());
}

fn put_optional_commitment(hasher: &mut Sha256, value: Option<Commitment32>) {
    match value {
        None => put_u8(hasher, 0),
        Some(commitment) => {
            put_u8(hasher, 1);
            put_commitment(hasher, commitment);
        }
    }
}
