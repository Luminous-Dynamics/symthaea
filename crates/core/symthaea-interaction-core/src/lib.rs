// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pure semantic contracts for Symthaea's governed interaction fabric.
//!
//! This crate deliberately contains no transport, credential access, signing,
//! authority verification, persistence, retry engine, or external effect code.
//!
//! Core separation:
//!
//! ```text
//! InteractionIntent
//!   != current authority
//!   != durable reservation
//!   != DispatchPermit
//!   != external effect
//!
//! ExternalEvent
//!   != verified assertion
//!   != current authority
//!   != command
//! ```
//!
//! Canonical identities are SHA-256 commitments over language-neutral,
//! domain-separated transcripts with explicit big-endian integers and
//! length-prefixed text. Rust layout, `Debug`, JSON, Serde, and storage
//! encodings are not semantic identity.

#![deny(unsafe_code)]

use sha2::{Digest as ShaDigest, Sha256};
use std::error::Error;
use std::fmt;

/// Current language-neutral interaction transcript schema.
pub const INTERACTION_SCHEMA_VERSION: u16 = 1;

const RESOURCE_DOMAIN: &[u8] = b"symthaea.interaction.resource.v1\0";
const PRINCIPAL_DOMAIN: &[u8] = b"symthaea.interaction.principal.v1\0";
const CONNECTOR_DOMAIN: &[u8] = b"symthaea.interaction.connector.v1\0";
const OPERATION_DOMAIN: &[u8] = b"symthaea.interaction.operation.v1\0";
const INTENT_DOMAIN: &[u8] = b"symthaea.interaction.intent.v1\0";
const EVENT_DOMAIN: &[u8] = b"symthaea.interaction.event.v1\0";
const OBSERVATION_DOMAIN: &[u8] = b"symthaea.interaction.observation.v1\0";

const MAX_NAMESPACE_LEN: usize = 128;
const MAX_ATOM_LEN: usize = 256;
const MAX_COMPONENT_VALUE_LEN: usize = 4096;
const MAX_COMPONENTS: usize = 64;

/// Fixed-width semantic commitment.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Digest32([u8; 32]);

impl Digest32 {
    /// Construct from exact digest bytes.
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Return exact digest bytes.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Lowercase hexadecimal representation for logs/tests.
    ///
    /// The hexadecimal string is presentation only and never the canonical
    /// transcript representation.
    pub fn to_hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 0x0f) as usize] as char);
        }
        out
    }
}

impl fmt::Debug for Digest32 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("Digest32")
            .field(&self.to_hex())
            .finish()
    }
}

/// Fail-closed errors for canonical interaction values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InteractionCoreError {
    UnsupportedSchemaVersion(u16),
    EmptyText(&'static str),
    TextTooLong {
        field: &'static str,
        length: usize,
        maximum: usize,
    },
    NonCanonicalText {
        field: &'static str,
        byte: u8,
    },
    EmptyComponents,
    TooManyComponents(usize),
    DuplicateNamedComponent(String),
}

impl fmt::Display for InteractionCoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion(version) => {
                write!(formatter, "unsupported interaction schema version {version}")
            }
            Self::EmptyText(field) => write!(formatter, "{field} must not be empty"),
            Self::TextTooLong {
                field,
                length,
                maximum,
            } => write!(
                formatter,
                "{field} length {length} exceeds maximum {maximum}"
            ),
            Self::NonCanonicalText { field, byte } => write!(
                formatter,
                "{field} contains non-canonical byte 0x{byte:02x}; identity text must be printable ASCII"
            ),
            Self::EmptyComponents => write!(formatter, "identity requires at least one component"),
            Self::TooManyComponents(count) => {
                write!(formatter, "identity has {count} components; maximum is {MAX_COMPONENTS}")
            }
            Self::DuplicateNamedComponent(name) => {
                write!(formatter, "named-set identity repeats component name {name:?}")
            }
        }
    }
}

impl Error for InteractionCoreError {}

/// Closed schema profile. Unknown numeric versions fail closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionSchema {
    V1,
}

impl InteractionSchema {
    pub const fn as_u16(self) -> u16 {
        match self {
            Self::V1 => INTERACTION_SCHEMA_VERSION,
        }
    }
}

impl TryFrom<u16> for InteractionSchema {
    type Error = InteractionCoreError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            INTERACTION_SCHEMA_VERSION => Ok(Self::V1),
            other => Err(InteractionCoreError::UnsupportedSchemaVersion(other)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct CanonicalText(String);

impl CanonicalText {
    fn new(
        field: &'static str,
        value: impl Into<String>,
        maximum: usize,
    ) -> Result<Self, InteractionCoreError> {
        let value = value.into();
        if value.is_empty() {
            return Err(InteractionCoreError::EmptyText(field));
        }
        if value.len() > maximum {
            return Err(InteractionCoreError::TextTooLong {
                field,
                length: value.len(),
                maximum,
            });
        }
        if let Some(byte) = value
            .as_bytes()
            .iter()
            .copied()
            .find(|byte| !byte.is_ascii_graphic())
        {
            return Err(InteractionCoreError::NonCanonicalText { field, byte });
        }
        Ok(Self(value))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

/// Registered semantic namespace, e.g. `mycelix/holochain`, `mcp`, `caip`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NamespaceId(CanonicalText);

impl NamespaceId {
    pub fn new(value: impl Into<String>) -> Result<Self, InteractionCoreError> {
        CanonicalText::new("namespace", value, MAX_NAMESPACE_LEN).map(Self)
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

/// Canonical labeled identity component.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IdentityComponent {
    name: CanonicalText,
    value: CanonicalText,
}

impl IdentityComponent {
    pub fn new(name: &str, value: &str) -> Result<Self, InteractionCoreError> {
        Ok(Self {
            name: CanonicalText::new("component name", name, MAX_ATOM_LEN)?,
            value: CanonicalText::new(
                "component value",
                value,
                MAX_COMPONENT_VALUE_LEN,
            )?,
        })
    }

    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    pub fn value(&self) -> &str {
        self.value.as_str()
    }
}

/// Whether component position or component name determines identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityOrdering {
    /// Preserve the exact caller-supplied component order.
    Ordered,
    /// Sort by component name and reject duplicate names.
    NamedSet,
}

impl IdentityOrdering {
    const fn code(self) -> u8 {
        match self {
            Self::Ordered => 0,
            Self::NamedSet => 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CanonicalIdentity {
    namespace: NamespaceId,
    kind: CanonicalText,
    ordering: IdentityOrdering,
    components: Vec<IdentityComponent>,
}

impl CanonicalIdentity {
    fn new(
        namespace: NamespaceId,
        kind: &str,
        ordering: IdentityOrdering,
        mut components: Vec<IdentityComponent>,
    ) -> Result<Self, InteractionCoreError> {
        if components.is_empty() {
            return Err(InteractionCoreError::EmptyComponents);
        }
        if components.len() > MAX_COMPONENTS {
            return Err(InteractionCoreError::TooManyComponents(components.len()));
        }

        if ordering == IdentityOrdering::NamedSet {
            components.sort_by(|left, right| {
                left.name
                    .cmp(&right.name)
                    .then_with(|| left.value.cmp(&right.value))
            });
            if let Some(pair) = components
                .windows(2)
                .find(|pair| pair[0].name == pair[1].name)
            {
                return Err(InteractionCoreError::DuplicateNamedComponent(
                    pair[0].name.as_str().to_owned(),
                ));
            }
        }

        Ok(Self {
            namespace,
            kind: CanonicalText::new("resource/principal kind", kind, MAX_ATOM_LEN)?,
            ordering,
            components,
        })
    }

    fn digest(&self, domain: &[u8]) -> Digest32 {
        let mut transcript = CanonicalTranscript::new(domain);
        transcript.u16(INTERACTION_SCHEMA_VERSION);
        transcript.text(self.namespace.as_str());
        transcript.text(self.kind.as_str());
        transcript.u8(self.ordering.code());
        transcript.u32(self.components.len() as u32);
        for component in &self.components {
            transcript.text(component.name());
            transcript.text(component.value());
        }
        transcript.finish()
    }
}

/// Canonical external resource identity.
///
/// A resource is semantic identity only. Its existence does not imply a grant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceRef(CanonicalIdentity);

impl ResourceRef {
    pub fn new(
        namespace: NamespaceId,
        kind: &str,
        ordering: IdentityOrdering,
        components: Vec<IdentityComponent>,
    ) -> Result<Self, InteractionCoreError> {
        CanonicalIdentity::new(namespace, kind, ordering, components).map(Self)
    }

    pub fn namespace(&self) -> &NamespaceId {
        &self.0.namespace
    }

    pub fn kind(&self) -> &str {
        self.0.kind.as_str()
    }

    #[must_use]
    pub fn digest(&self) -> Digest32 {
        self.0.digest(RESOURCE_DOMAIN)
    }
}

/// Canonical principal identity.
///
/// Identity linkage across namespaces requires external evidence; matching text
/// does not merge principals automatically.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrincipalRef(CanonicalIdentity);

impl PrincipalRef {
    pub fn new(
        namespace: NamespaceId,
        kind: &str,
        ordering: IdentityOrdering,
        components: Vec<IdentityComponent>,
    ) -> Result<Self, InteractionCoreError> {
        CanonicalIdentity::new(namespace, kind, ordering, components).map(Self)
    }

    #[must_use]
    pub fn digest(&self) -> Digest32 {
        self.0.digest(PRINCIPAL_DOMAIN)
    }
}

/// Stable connector family identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectorId {
    namespace: NamespaceId,
    name: CanonicalText,
}

impl ConnectorId {
    pub fn new(namespace: NamespaceId, name: &str) -> Result<Self, InteractionCoreError> {
        Ok(Self {
            namespace,
            name: CanonicalText::new("connector name", name, MAX_ATOM_LEN)?,
        })
    }

    pub fn namespace(&self) -> &NamespaceId {
        &self.namespace
    }

    pub fn name(&self) -> &str {
        self.name.as_str()
    }
}

/// Exact connector profile/implementation identity.
///
/// `implementation_digest` may bind a native binary/component/descriptor, but
/// merely possessing this object grants no capability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectorIdentity {
    id: ConnectorId,
    profile: CanonicalText,
    implementation_digest: Option<Digest32>,
}

impl ConnectorIdentity {
    pub fn new(
        id: ConnectorId,
        profile: &str,
        implementation_digest: Option<Digest32>,
    ) -> Result<Self, InteractionCoreError> {
        Ok(Self {
            id,
            profile: CanonicalText::new("connector profile", profile, MAX_ATOM_LEN)?,
            implementation_digest,
        })
    }

    pub fn id(&self) -> &ConnectorId {
        &self.id
    }

    pub fn profile(&self) -> &str {
        self.profile.as_str()
    }

    #[must_use]
    pub fn digest(&self) -> Digest32 {
        let mut transcript = CanonicalTranscript::new(CONNECTOR_DOMAIN);
        transcript.u16(INTERACTION_SCHEMA_VERSION);
        transcript.text(self.id.namespace.as_str());
        transcript.text(self.id.name.as_str());
        transcript.text(self.profile.as_str());
        transcript.optional_digest(self.implementation_digest);
        transcript.finish()
    }
}

/// Canonical semantic operation identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationRef {
    namespace: NamespaceId,
    name: CanonicalText,
    profile: Option<CanonicalText>,
}

impl OperationRef {
    pub fn new(
        namespace: NamespaceId,
        name: &str,
        profile: Option<&str>,
    ) -> Result<Self, InteractionCoreError> {
        Ok(Self {
            namespace,
            name: CanonicalText::new("operation name", name, MAX_ATOM_LEN)?,
            profile: profile
                .map(|value| CanonicalText::new("operation profile", value, MAX_ATOM_LEN))
                .transpose()?,
        })
    }

    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    #[must_use]
    pub fn digest(&self) -> Digest32 {
        let mut transcript = CanonicalTranscript::new(OPERATION_DOMAIN);
        transcript.u16(INTERACTION_SCHEMA_VERSION);
        transcript.text(self.namespace.as_str());
        transcript.text(self.name.as_str());
        transcript.optional_text(self.profile.as_ref().map(CanonicalText::as_str));
        transcript.finish()
    }
}

/// Coarse semantic effect class. This is descriptive, never authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectClass {
    ObserveOnly,
    Create,
    Update,
    Delete,
    TransferValue,
    Publish,
    GrantAuthority,
    RevokeAuthority,
    ExecuteCode,
    PhysicalActuation,
}

impl EffectClass {
    const fn code(self) -> u16 {
        match self {
            Self::ObserveOnly => 0,
            Self::Create => 1,
            Self::Update => 2,
            Self::Delete => 3,
            Self::TransferValue => 4,
            Self::Publish => 5,
            Self::GrantAuthority => 6,
            Self::RevokeAuthority => 7,
            Self::ExecuteCode => 8,
            Self::PhysicalActuation => 9,
        }
    }
}

/// Retry/idempotency semantics declared for an operation.
///
/// Compensation and irreversibility are intentionally not collapsed into this
/// enum; they are orthogonal policies owned by the later reconciliation layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdempotencyClass {
    NotApplicable,
    IntrinsicallyIdempotent,
    IdempotencyKeyed,
    QueryAfterWrite,
    UnknownOutcomeUnsafeToRetry,
}

impl IdempotencyClass {
    const fn code(self) -> u16 {
        match self {
            Self::NotApplicable => 0,
            Self::IntrinsicallyIdempotent => 1,
            Self::IdempotencyKeyed => 2,
            Self::QueryAfterWrite => 3,
            Self::UnknownOutcomeUnsafeToRetry => 4,
        }
    }
}

/// Proposed semantic interaction.
///
/// This type contains no executable transport, token, key, socket, signer, or
/// authority object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InteractionIntent {
    connector: ConnectorIdentity,
    principal: Option<PrincipalRef>,
    resource: ResourceRef,
    operation: OperationRef,
    input_commitment: Digest32,
    effect_class: EffectClass,
    idempotency: IdempotencyClass,
}

impl InteractionIntent {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        connector: ConnectorIdentity,
        principal: Option<PrincipalRef>,
        resource: ResourceRef,
        operation: OperationRef,
        input_commitment: Digest32,
        effect_class: EffectClass,
        idempotency: IdempotencyClass,
    ) -> Self {
        Self {
            connector,
            principal,
            resource,
            operation,
            input_commitment,
            effect_class,
            idempotency,
        }
    }

    pub fn resource(&self) -> &ResourceRef {
        &self.resource
    }

    pub fn operation(&self) -> &OperationRef {
        &self.operation
    }

    pub const fn effect_class(&self) -> EffectClass {
        self.effect_class
    }

    pub const fn idempotency(&self) -> IdempotencyClass {
        self.idempotency
    }

    #[must_use]
    pub fn digest(&self) -> Digest32 {
        let mut transcript = CanonicalTranscript::new(INTENT_DOMAIN);
        transcript.u16(INTERACTION_SCHEMA_VERSION);
        transcript.digest(self.connector.digest());
        transcript.optional_digest(self.principal.as_ref().map(PrincipalRef::digest));
        transcript.digest(self.resource.digest());
        transcript.digest(self.operation.digest());
        transcript.digest(self.input_commitment);
        transcript.u16(self.effect_class.code());
        transcript.u16(self.idempotency.code());
        transcript.finish()
    }
}

/// Delivery guarantee reported by the source/adapter profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryGuarantee {
    Unknown,
    AtMostOnce,
    AtLeastOnce,
}

impl DeliveryGuarantee {
    const fn code(self) -> u16 {
        match self {
            Self::Unknown => 0,
            Self::AtMostOnce => 1,
            Self::AtLeastOnce => 2,
        }
    }
}

/// Ordering guarantee reported by the source/adapter profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderingGuarantee {
    Unknown,
    OrderedPerSource,
    GloballyOrdered,
    Unordered,
}

impl OrderingGuarantee {
    const fn code(self) -> u16 {
        match self {
            Self::Unknown => 0,
            Self::OrderedPerSource => 1,
            Self::GloballyOrdered => 2,
            Self::Unordered => 3,
        }
    }
}

/// Whether observations may later be explicitly superseded/reverted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RevisionSemantics {
    AppendOnly,
    Supersedable,
    ReorgCapable,
    Unknown,
}

impl RevisionSemantics {
    const fn code(self) -> u16 {
        match self {
            Self::AppendOnly => 0,
            Self::Supersedable => 1,
            Self::ReorgCapable => 2,
            Self::Unknown => 3,
        }
    }
}

/// Protocol-neutral asynchronous external event envelope.
///
/// Authentication/truth classification occurs after this source-normalization
/// step. The event itself is never a command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalEvent {
    source: ResourceRef,
    event_type: CanonicalText,
    native_event_id: Option<CanonicalText>,
    payload_commitment: Digest32,
    delivery: DeliveryGuarantee,
    ordering: OrderingGuarantee,
    revision: RevisionSemantics,
}

impl ExternalEvent {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source: ResourceRef,
        event_type: &str,
        native_event_id: Option<&str>,
        payload_commitment: Digest32,
        delivery: DeliveryGuarantee,
        ordering: OrderingGuarantee,
        revision: RevisionSemantics,
    ) -> Result<Self, InteractionCoreError> {
        Ok(Self {
            source,
            event_type: CanonicalText::new("event type", event_type, MAX_ATOM_LEN)?,
            native_event_id: native_event_id
                .map(|value| {
                    CanonicalText::new("native event id", value, MAX_COMPONENT_VALUE_LEN)
                })
                .transpose()?,
            payload_commitment,
            delivery,
            ordering,
            revision,
        })
    }

    #[must_use]
    pub fn digest(&self) -> Digest32 {
        let mut transcript = CanonicalTranscript::new(EVENT_DOMAIN);
        transcript.u16(INTERACTION_SCHEMA_VERSION);
        transcript.digest(self.source.digest());
        transcript.text(self.event_type.as_str());
        transcript.optional_text(self.native_event_id.as_ref().map(CanonicalText::as_str));
        transcript.digest(self.payload_commitment);
        transcript.u16(self.delivery.code());
        transcript.u16(self.ordering.code());
        transcript.u16(self.revision.code());
        transcript.finish()
    }
}

/// How strongly the source identity itself was authenticated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceAuthentication {
    Unverified,
    TransportAuthenticated,
    CryptographicallyAuthenticated,
}

impl SourceAuthentication {
    const fn code(self) -> u16 {
        match self {
            Self::Unverified => 0,
            Self::TransportAuthenticated => 1,
            Self::CryptographicallyAuthenticated => 2,
        }
    }
}

/// Epistemic verification state of an observation.
///
/// This remains evidence classification, not current execution authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationState {
    Unverified,
    SourceAuthenticated,
    Corroborated,
    IndependentlyVerified,
}

impl VerificationState {
    const fn code(self) -> u16 {
        match self {
            Self::Unverified => 0,
            Self::SourceAuthenticated => 1,
            Self::Corroborated => 2,
            Self::IndependentlyVerified => 3,
        }
    }
}

/// Generic remote finality vocabulary.
///
/// Adapters must preserve native evidence and explicitly map native semantics;
/// this enum does not claim one universal meaning of finality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteFinality {
    Observed,
    Acknowledged,
    AcceptedProvisional,
    Confirmed,
    Final,
    Reverted,
    Superseded,
    OutcomeUnknown,
    Failed,
}

impl RemoteFinality {
    const fn code(self) -> u16 {
        match self {
            Self::Observed => 0,
            Self::Acknowledged => 1,
            Self::AcceptedProvisional => 2,
            Self::Confirmed => 3,
            Self::Final => 4,
            Self::Reverted => 5,
            Self::Superseded => 6,
            Self::OutcomeUnknown => 7,
            Self::Failed => 8,
        }
    }
}

/// Provenance-bearing observation derived explicitly from an external event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalObservation {
    event_commitment: Digest32,
    source_authentication: SourceAuthentication,
    verification: VerificationState,
    finality: RemoteFinality,
    evidence_commitment: Option<Digest32>,
}

impl ExternalObservation {
    pub fn new(
        event: &ExternalEvent,
        source_authentication: SourceAuthentication,
        verification: VerificationState,
        finality: RemoteFinality,
        evidence_commitment: Option<Digest32>,
    ) -> Self {
        Self {
            event_commitment: event.digest(),
            source_authentication,
            verification,
            finality,
            evidence_commitment,
        }
    }

    pub const fn verification(&self) -> VerificationState {
        self.verification
    }

    pub const fn finality(&self) -> RemoteFinality {
        self.finality
    }

    #[must_use]
    pub fn digest(&self) -> Digest32 {
        let mut transcript = CanonicalTranscript::new(OBSERVATION_DOMAIN);
        transcript.u16(INTERACTION_SCHEMA_VERSION);
        transcript.digest(self.event_commitment);
        transcript.u16(self.source_authentication.code());
        transcript.u16(self.verification.code());
        transcript.u16(self.finality.code());
        transcript.optional_digest(self.evidence_commitment);
        transcript.finish()
    }
}

/// Generic reconciliation state. This is not the durable action-runtime state
/// machine and must not be used to mint dispatch authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconciliationState {
    NotRequired,
    Pending,
    OutcomeUnknown,
    Committed,
    Released,
    Failed,
    NeedsHumanResolution,
    Superseded,
}

struct CanonicalTranscript {
    hasher: Sha256,
}

impl CanonicalTranscript {
    fn new(domain: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(domain);
        Self { hasher }
    }

    fn u8(&mut self, value: u8) {
        self.hasher.update([value]);
    }

    fn u16(&mut self, value: u16) {
        self.hasher.update(value.to_be_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.hasher.update(value.to_be_bytes());
    }

    fn text(&mut self, value: &str) {
        let bytes = value.as_bytes();
        self.u32(bytes.len() as u32);
        self.hasher.update(bytes);
    }

    fn optional_text(&mut self, value: Option<&str>) {
        match value {
            Some(value) => {
                self.u8(1);
                self.text(value);
            }
            None => self.u8(0),
        }
    }

    fn digest(&mut self, value: Digest32) {
        self.hasher.update(value.as_bytes());
    }

    fn optional_digest(&mut self, value: Option<Digest32>) {
        match value {
            Some(value) => {
                self.u8(1);
                self.digest(value);
            }
            None => self.u8(0),
        }
    }

    fn finish(self) -> Digest32 {
        let digest = self.hasher.finalize();
        let mut bytes = [0_u8; 32];
        bytes.copy_from_slice(&digest);
        Digest32::new(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn component(name: &str, value: &str) -> IdentityComponent {
        IdentityComponent::new(name, value).expect("test component")
    }

    fn holochain_resource(namespace: &str) -> ResourceRef {
        ResourceRef::new(
            NamespaceId::new(namespace).expect("namespace"),
            "zome-function",
            IdentityOrdering::Ordered,
            vec![
                component("app", "pulse"),
                component("role", "messages"),
                component("zome", "messages"),
                component("fn", "send_message"),
            ],
        )
        .expect("resource")
    }

    fn golden_intent() -> InteractionIntent {
        let namespace = NamespaceId::new("mycelix/holochain").expect("namespace");
        let connector = ConnectorIdentity::new(
            ConnectorId::new(namespace.clone(), "native").expect("connector id"),
            "0.7",
            Some(Digest32::new([0x11; 32])),
        )
        .expect("connector");
        let principal = PrincipalRef::new(
            namespace.clone(),
            "agent",
            IdentityOrdering::NamedSet,
            vec![component("agent", "uhCAk-test-agent")],
        )
        .expect("principal");
        let operation =
            OperationRef::new(namespace, "call", Some("zome")).expect("operation");

        InteractionIntent::new(
            connector,
            Some(principal),
            holochain_resource("mycelix/holochain"),
            operation,
            Digest32::new([0x55; 32]),
            EffectClass::Update,
            IdempotencyClass::QueryAfterWrite,
        )
    }

    #[test]
    fn golden_resource_and_intent_vectors_are_stable() {
        assert_eq!(
            holochain_resource("mycelix/holochain").digest().to_hex(),
            "66291cec9c82f8dbccffcda46d1326d675c2523bd7f51c65fedc794d5ca29aba"
        );
        assert_eq!(
            golden_intent().digest().to_hex(),
            "754d23d3304f8609cdaf3d0c805f2e73a764fcb1d597c17b942ec57b5508d361"
        );
    }

    #[test]
    fn namespace_separation_changes_identity() {
        assert_ne!(
            holochain_resource("mycelix/holochain").digest(),
            holochain_resource("web/http").digest()
        );
    }

    #[test]
    fn component_boundaries_cannot_collide() {
        let namespace = NamespaceId::new("test").expect("namespace");
        let left = ResourceRef::new(
            namespace.clone(),
            "thing",
            IdentityOrdering::Ordered,
            vec![component("x", "a"), component("y", "bc")],
        )
        .expect("left");
        let right = ResourceRef::new(
            namespace,
            "thing",
            IdentityOrdering::Ordered,
            vec![component("x", "ab"), component("y", "c")],
        )
        .expect("right");
        assert_ne!(left.digest(), right.digest());
    }

    #[test]
    fn named_set_normalizes_insertion_order() {
        let namespace = NamespaceId::new("kubernetes").expect("namespace");
        let left = ResourceRef::new(
            namespace.clone(),
            "object",
            IdentityOrdering::NamedSet,
            vec![
                component("namespace", "prod"),
                component("kind", "Deployment"),
                component("name", "api"),
            ],
        )
        .expect("left");
        let right = ResourceRef::new(
            namespace,
            "object",
            IdentityOrdering::NamedSet,
            vec![
                component("name", "api"),
                component("namespace", "prod"),
                component("kind", "Deployment"),
            ],
        )
        .expect("right");
        assert_eq!(left.digest(), right.digest());
    }

    #[test]
    fn named_set_rejects_duplicate_component_names() {
        let error = ResourceRef::new(
            NamespaceId::new("test").expect("namespace"),
            "thing",
            IdentityOrdering::NamedSet,
            vec![component("id", "a"), component("id", "b")],
        )
        .expect_err("duplicate names must fail");
        assert!(matches!(
            error,
            InteractionCoreError::DuplicateNamedComponent(_)
        ));
    }

    #[test]
    fn semantic_intent_mutation_changes_identity() {
        let baseline = golden_intent();
        let mut changed = golden_intent();
        changed.effect_class = EffectClass::Delete;
        assert_ne!(baseline.digest(), changed.digest());
    }

    #[test]
    fn unknown_schema_versions_fail_closed() {
        assert_eq!(
            InteractionSchema::try_from(2),
            Err(InteractionCoreError::UnsupportedSchemaVersion(2))
        );
    }

    #[test]
    fn empty_and_noncanonical_identifiers_fail_closed() {
        assert!(matches!(
            NamespaceId::new(""),
            Err(InteractionCoreError::EmptyText("namespace"))
        ));
        assert!(matches!(
            NamespaceId::new("bad namespace"),
            Err(InteractionCoreError::NonCanonicalText { .. })
        ));
    }

    #[test]
    fn outcome_unknown_is_not_failure_or_release() {
        assert_ne!(
            ReconciliationState::OutcomeUnknown,
            ReconciliationState::Failed
        );
        assert_ne!(
            ReconciliationState::OutcomeUnknown,
            ReconciliationState::Released
        );
        assert_ne!(RemoteFinality::OutcomeUnknown, RemoteFinality::Failed);
    }

    #[test]
    fn final_is_distinct_from_acknowledged_and_confirmed() {
        assert_ne!(RemoteFinality::Final, RemoteFinality::Acknowledged);
        assert_ne!(RemoteFinality::Final, RemoteFinality::Confirmed);
    }

    #[test]
    fn adding_other_protocol_namespaces_does_not_change_existing_vectors() {
        let before = golden_intent().digest();
        let _mqtt = ResourceRef::new(
            NamespaceId::new("mqtt").expect("namespace"),
            "topic",
            IdentityOrdering::Ordered,
            vec![
                component("broker", "edge-1"),
                component("topic", "sensors/room1"),
            ],
        )
        .expect("mqtt resource");
        assert_eq!(before, golden_intent().digest());
    }

    #[test]
    fn event_and_observation_are_explicitly_separate() {
        let event = ExternalEvent::new(
            ResourceRef::new(
                NamespaceId::new("web/http").expect("namespace"),
                "webhook",
                IdentityOrdering::NamedSet,
                vec![
                    component("origin", "https://example.invalid"),
                    component("path", "/events"),
                ],
            )
            .expect("source"),
            "example.updated",
            Some("evt-42"),
            Digest32::new([0x77; 32]),
            DeliveryGuarantee::AtLeastOnce,
            OrderingGuarantee::Unordered,
            RevisionSemantics::AppendOnly,
        )
        .expect("event");

        let observation = ExternalObservation::new(
            &event,
            SourceAuthentication::TransportAuthenticated,
            VerificationState::SourceAuthenticated,
            RemoteFinality::Observed,
            None,
        );

        assert_ne!(event.digest(), observation.digest());
        assert_eq!(
            observation.verification(),
            VerificationState::SourceAuthenticated
        );
    }
}
