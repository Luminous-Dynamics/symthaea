// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pure connector-manifest semantics for Symthaea's interaction fabric.
//!
//! A manifest is descriptive data only:
//!
//! ```text
//! ConnectorManifest
//!     != installed connector
//!     != trusted connector
//!     != capability grant
//!     != current authority
//!     != DispatchPermit
//! ```

#![deny(unsafe_code)]

use sha2::{Digest as ShaDigest, Sha256};
use std::error::Error;
use std::fmt;
use symthaea_interaction_core::{
    ConnectorIdentity, Digest32, EffectClass, IdempotencyClass, NamespaceId, OperationRef,
    OrderingGuarantee, RevisionSemantics, SourceAuthentication,
};

pub const CONNECTOR_MANIFEST_SCHEMA_VERSION: u16 = 1;
const MANIFEST_DOMAIN: &[u8] = b"symthaea.interaction.connector-manifest.v1\0";
const RESOURCE_FAMILY_DOMAIN: &[u8] = b"symthaea.interaction.manifest.resource-family.v1\0";
const PROTOCOL_DOMAIN: &[u8] = b"symthaea.interaction.manifest.protocol.v1\0";
const OBSERVATION_DOMAIN: &[u8] = b"symthaea.interaction.manifest.observation.v1\0";
const EFFECT_DOMAIN: &[u8] = b"symthaea.interaction.manifest.effect.v1\0";
const AUTH_DOMAIN: &[u8] = b"symthaea.interaction.manifest.auth.v1\0";
const RECONCILIATION_DOMAIN: &[u8] = b"symthaea.interaction.manifest.reconciliation.v1\0";
const MAX_ATOM_LEN: usize = 256;
const MAX_CAPABILITIES: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestError {
    UnsupportedSchemaVersion(u16),
    EmptyText(&'static str),
    TextTooLong { field: &'static str, length: usize },
    NonCanonicalText { field: &'static str, byte: u8 },
    NoCapabilities,
    TooManyCapabilities { kind: &'static str, count: usize },
    DuplicateCapability(&'static str),
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion(v) => write!(f, "unsupported connector-manifest schema version {v}"),
            Self::EmptyText(field) => write!(f, "{field} must not be empty"),
            Self::TextTooLong { field, length } => write!(f, "{field} length {length} exceeds maximum {MAX_ATOM_LEN}"),
            Self::NonCanonicalText { field, byte } => write!(f, "{field} contains non-canonical byte 0x{byte:02x}"),
            Self::NoCapabilities => write!(f, "manifest must declare at least one observation or effect capability"),
            Self::TooManyCapabilities { kind, count } => write!(f, "manifest has {count} {kind} capabilities; maximum is {MAX_CAPABILITIES}"),
            Self::DuplicateCapability(kind) => write!(f, "manifest contains a duplicate {kind} capability"),
        }
    }
}
impl Error for ManifestError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestSchema { V1 }
impl TryFrom<u16> for ManifestSchema {
    type Error = ManifestError;
    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            CONNECTOR_MANIFEST_SCHEMA_VERSION => Ok(Self::V1),
            other => Err(ManifestError::UnsupportedSchemaVersion(other)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CanonicalText(String);
impl CanonicalText {
    fn new(field: &'static str, value: &str) -> Result<Self, ManifestError> {
        if value.is_empty() { return Err(ManifestError::EmptyText(field)); }
        if value.len() > MAX_ATOM_LEN { return Err(ManifestError::TextTooLong { field, length: value.len() }); }
        if let Some(byte) = value.as_bytes().iter().copied().find(|b| !b.is_ascii_graphic()) {
            return Err(ManifestError::NonCanonicalText { field, byte });
        }
        Ok(Self(value.to_owned()))
    }
    fn as_str(&self) -> &str { &self.0 }
}

struct Transcript { hasher: Sha256 }
impl Transcript {
    fn new(domain: &[u8]) -> Self { let mut hasher = Sha256::new(); hasher.update(domain); Self { hasher } }
    fn u8(&mut self, v: u8) { self.hasher.update([v]); }
    fn u16(&mut self, v: u16) { self.hasher.update(v.to_be_bytes()); }
    fn u32(&mut self, v: u32) { self.hasher.update(v.to_be_bytes()); }
    fn u64(&mut self, v: u64) { self.hasher.update(v.to_be_bytes()); }
    fn boolean(&mut self, v: bool) { self.u8(u8::from(v)); }
    fn text(&mut self, v: &str) { self.u32(v.len() as u32); self.hasher.update(v.as_bytes()); }
    fn optional_text(&mut self, v: Option<&str>) { match v { Some(v) => { self.u8(1); self.text(v); }, None => self.u8(0) } }
    fn digest(&mut self, v: Digest32) { self.hasher.update(v.as_bytes()); }
    fn finish(self) -> Digest32 { let bytes = self.hasher.finalize(); let mut out = [0_u8; 32]; out.copy_from_slice(&bytes); Digest32::new(out) }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceFamily { namespace: NamespaceId, kind: CanonicalText }
impl ResourceFamily {
    pub fn new(namespace: NamespaceId, kind: &str) -> Result<Self, ManifestError> { Ok(Self { namespace, kind: CanonicalText::new("resource-family kind", kind)? }) }
    #[must_use] pub fn digest(&self) -> Digest32 { let mut t = Transcript::new(RESOURCE_FAMILY_DOMAIN); t.u16(1); t.text(self.namespace.as_str()); t.text(self.kind.as_str()); t.finish() }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtocolBinding { family: CanonicalText, version: CanonicalText }
impl ProtocolBinding {
    pub fn new(family: &str, version: &str) -> Result<Self, ManifestError> { Ok(Self { family: CanonicalText::new("protocol family", family)?, version: CanonicalText::new("protocol version", version)? }) }
    #[must_use] pub fn digest(&self) -> Digest32 { let mut t = Transcript::new(PROTOCOL_DOMAIN); t.u16(1); t.text(self.family.as_str()); t.text(self.version.as_str()); t.finish() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationCapability { None, SourceAuthenticationOnly, CorroborationAvailable, IndependentVerificationAvailable }
impl VerificationCapability { const fn code(self) -> u16 { match self { Self::None => 0, Self::SourceAuthenticationOnly => 1, Self::CorroborationAvailable => 2, Self::IndependentVerificationAvailable => 3 } } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsistencyModel { Unknown, Strong, Sequential, Eventual, Causal }
impl ConsistencyModel { const fn code(self) -> u16 { match self { Self::Unknown => 0, Self::Strong => 1, Self::Sequential => 2, Self::Eventual => 3, Self::Causal => 4 } } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreshnessModel { Unknown, PointInTime, BoundedStaleness { max_millis: u64 }, EventDriven }
impl FreshnessModel {
    fn write(self, t: &mut Transcript) { match self { Self::Unknown => t.u16(0), Self::PointInTime => t.u16(1), Self::BoundedStaleness { max_millis } => { t.u16(2); t.u64(max_millis); }, Self::EventDriven => t.u16(3) } }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservationCapability {
    family: ResourceFamily,
    operation: OperationRef,
    source_authentication: SourceAuthentication,
    verification: VerificationCapability,
    consistency: ConsistencyModel,
    freshness: FreshnessModel,
    ordering: OrderingGuarantee,
    revision: RevisionSemantics,
    native_evidence_ids: bool,
    independent_reconciliation: bool,
}
impl ObservationCapability {
    #[allow(clippy::too_many_arguments)]
    pub fn new(family: ResourceFamily, operation: OperationRef, source_authentication: SourceAuthentication, verification: VerificationCapability, consistency: ConsistencyModel, freshness: FreshnessModel, ordering: OrderingGuarantee, revision: RevisionSemantics, native_evidence_ids: bool, independent_reconciliation: bool) -> Self {
        Self { family, operation, source_authentication, verification, consistency, freshness, ordering, revision, native_evidence_ids, independent_reconciliation }
    }
    #[must_use] pub fn digest(&self) -> Digest32 {
        let mut t = Transcript::new(OBSERVATION_DOMAIN); t.u16(1); t.digest(self.family.digest()); t.digest(self.operation.digest());
        t.u16(source_auth_code(self.source_authentication)); t.u16(self.verification.code()); t.u16(self.consistency.code()); self.freshness.write(&mut t);
        t.u16(ordering_code(self.ordering)); t.u16(revision_code(self.revision)); t.boolean(self.native_evidence_ids); t.boolean(self.independent_reconciliation); t.finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreflightSupport { None, Local, Remote, LocalAndRemote }
impl PreflightSupport { const fn code(self) -> u16 { match self { Self::None => 0, Self::Local => 1, Self::Remote => 2, Self::LocalAndRemote => 3 } } }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalityCapability { NoRemoteState, AcknowledgementOnly, ConfirmationTrackable, FinalityTrackable, ReorgTrackable }
impl FinalityCapability { const fn code(self) -> u16 { match self { Self::NoRemoteState => 0, Self::AcknowledgementOnly => 1, Self::ConfirmationTrackable => 2, Self::FinalityTrackable => 3, Self::ReorgTrackable => 4 } } }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompensationSupport { None, SeparateEffectDeclared }
impl CompensationSupport { const fn code(self) -> u16 { match self { Self::None => 0, Self::SeparateEffectDeclared => 1 } } }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectCapability {
    family: ResourceFamily, operation: OperationRef, effect_class: EffectClass, idempotency: IdempotencyClass,
    preflight: PreflightSupport, finality: FinalityCapability, compensation: CompensationSupport, preconditions_required: bool,
}
impl EffectCapability {
    #[allow(clippy::too_many_arguments)]
    pub fn new(family: ResourceFamily, operation: OperationRef, effect_class: EffectClass, idempotency: IdempotencyClass, preflight: PreflightSupport, finality: FinalityCapability, compensation: CompensationSupport, preconditions_required: bool) -> Self {
        Self { family, operation, effect_class, idempotency, preflight, finality, compensation, preconditions_required }
    }
    #[must_use] pub fn digest(&self) -> Digest32 {
        let mut t = Transcript::new(EFFECT_DOMAIN); t.u16(1); t.digest(self.family.digest()); t.digest(self.operation.digest());
        t.u16(effect_code(self.effect_class)); t.u16(idempotency_code(self.idempotency)); t.u16(self.preflight.code()); t.u16(self.finality.code()); t.u16(self.compensation.code()); t.boolean(self.preconditions_required); t.finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtocolUserRequirement { NotUsed, Optional, Required }
impl ProtocolUserRequirement { const fn code(self) -> u16 { match self { Self::NotUsed => 0, Self::Optional => 1, Self::Required => 2 } } }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthenticationRequirement { credential_class: CanonicalText, user_presence: ProtocolUserRequirement, user_verification: ProtocolUserRequirement, native_scope_profile: Option<CanonicalText> }
impl AuthenticationRequirement {
    pub fn new(credential_class: &str, user_presence: ProtocolUserRequirement, user_verification: ProtocolUserRequirement, native_scope_profile: Option<&str>) -> Result<Self, ManifestError> {
        Ok(Self { credential_class: CanonicalText::new("credential class", credential_class)?, user_presence, user_verification, native_scope_profile: native_scope_profile.map(|v| CanonicalText::new("native scope profile", v)).transpose()? })
    }
    #[must_use] pub fn digest(&self) -> Digest32 { let mut t = Transcript::new(AUTH_DOMAIN); t.u16(1); t.text(self.credential_class.as_str()); t.u16(self.user_presence.code()); t.u16(self.user_verification.code()); t.optional_text(self.native_scope_profile.as_ref().map(CanonicalText::as_str)); t.finish() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnknownOutcomeHandling { ManualOnly, QueryByNativeIdentity, EventualCallback, UnsafeToRetry }
impl UnknownOutcomeHandling { const fn code(self) -> u16 { match self { Self::ManualOnly => 0, Self::QueryByNativeIdentity => 1, Self::EventualCallback => 2, Self::UnsafeToRetry => 3 } } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReconciliationCapabilities {
    pub query_by_attempt: bool, pub query_by_native_id: bool, pub idempotency_key: bool, pub event_followup: bool,
    pub finality_observation: bool, pub reorg_tracking: bool, pub unknown_outcome: UnknownOutcomeHandling,
}
impl ReconciliationCapabilities {
    #[must_use] pub fn digest(self) -> Digest32 { let mut t = Transcript::new(RECONCILIATION_DOMAIN); t.u16(1); t.boolean(self.query_by_attempt); t.boolean(self.query_by_native_id); t.boolean(self.idempotency_key); t.boolean(self.event_followup); t.boolean(self.finality_observation); t.boolean(self.reorg_tracking); t.u16(self.unknown_outcome.code()); t.finish() }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectorManifest {
    connector: ConnectorIdentity, profile: CanonicalText, protocol: ProtocolBinding,
    observations: Vec<ObservationCapability>, effects: Vec<EffectCapability>, authentication: Vec<AuthenticationRequirement>,
    reconciliation: ReconciliationCapabilities,
}
impl ConnectorManifest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(connector: ConnectorIdentity, profile: &str, protocol: ProtocolBinding, mut observations: Vec<ObservationCapability>, mut effects: Vec<EffectCapability>, mut authentication: Vec<AuthenticationRequirement>, reconciliation: ReconciliationCapabilities) -> Result<Self, ManifestError> {
        if observations.is_empty() && effects.is_empty() { return Err(ManifestError::NoCapabilities); }
        check_count("observation", observations.len())?; check_count("effect", effects.len())?; check_count("authentication", authentication.len())?;
        normalize_unique("observation", &mut observations, ObservationCapability::digest)?;
        normalize_unique("effect", &mut effects, EffectCapability::digest)?;
        normalize_unique("authentication", &mut authentication, AuthenticationRequirement::digest)?;
        Ok(Self { connector, profile: CanonicalText::new("manifest profile", profile)?, protocol, observations, effects, authentication, reconciliation })
    }
    pub fn authentication(&self) -> &[AuthenticationRequirement] { &self.authentication }
    pub fn effects(&self) -> &[EffectCapability] { &self.effects }
    #[must_use] pub fn digest(&self) -> Digest32 {
        let mut t = Transcript::new(MANIFEST_DOMAIN); t.u16(1); t.digest(self.connector.digest()); t.text(self.profile.as_str()); t.digest(self.protocol.digest());
        t.u32(self.observations.len() as u32); for item in &self.observations { t.digest(item.digest()); }
        t.u32(self.effects.len() as u32); for item in &self.effects { t.digest(item.digest()); }
        t.u32(self.authentication.len() as u32); for item in &self.authentication { t.digest(item.digest()); }
        t.digest(self.reconciliation.digest()); t.finish()
    }
}

fn check_count(kind: &'static str, count: usize) -> Result<(), ManifestError> { if count > MAX_CAPABILITIES { Err(ManifestError::TooManyCapabilities { kind, count }) } else { Ok(()) } }
fn normalize_unique<T>(kind: &'static str, values: &mut [T], digest: fn(&T) -> Digest32) -> Result<(), ManifestError> {
    values.sort_by(|a, b| digest(a).as_bytes().cmp(digest(b).as_bytes()));
    if values.windows(2).any(|pair| digest(&pair[0]) == digest(&pair[1])) { Err(ManifestError::DuplicateCapability(kind)) } else { Ok(()) }
}
fn source_auth_code(v: SourceAuthentication) -> u16 { match v { SourceAuthentication::Unverified => 0, SourceAuthentication::TransportAuthenticated => 1, SourceAuthentication::CryptographicallyAuthenticated => 2 } }
fn ordering_code(v: OrderingGuarantee) -> u16 { match v { OrderingGuarantee::Unknown => 0, OrderingGuarantee::OrderedPerSource => 1, OrderingGuarantee::GloballyOrdered => 2, OrderingGuarantee::Unordered => 3 } }
fn revision_code(v: RevisionSemantics) -> u16 { match v { RevisionSemantics::AppendOnly => 0, RevisionSemantics::Supersedable => 1, RevisionSemantics::ReorgCapable => 2, RevisionSemantics::Unknown => 3 } }
fn effect_code(v: EffectClass) -> u16 { match v { EffectClass::ObserveOnly => 0, EffectClass::Create => 1, EffectClass::Update => 2, EffectClass::Delete => 3, EffectClass::TransferValue => 4, EffectClass::Publish => 5, EffectClass::GrantAuthority => 6, EffectClass::RevokeAuthority => 7, EffectClass::ExecuteCode => 8, EffectClass::PhysicalActuation => 9 } }
fn idempotency_code(v: IdempotencyClass) -> u16 { match v { IdempotencyClass::NotApplicable => 0, IdempotencyClass::IntrinsicallyIdempotent => 1, IdempotencyClass::IdempotencyKeyed => 2, IdempotencyClass::QueryAfterWrite => 3, IdempotencyClass::UnknownOutcomeUnsafeToRetry => 4 } }

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_interaction_core::ConnectorId;

    fn fixture() -> (ConnectorIdentity, ProtocolBinding, ObservationCapability, EffectCapability, AuthenticationRequirement, ReconciliationCapabilities) {
        let ns = NamespaceId::new("mycelix/holochain").unwrap();
        let connector = ConnectorIdentity::new(ConnectorId::new(ns.clone(), "native").unwrap(), "0.7", Some(Digest32::new([0x11; 32]))).unwrap();
        let family = ResourceFamily::new(ns.clone(), "zome-function").unwrap();
        let observation = ObservationCapability::new(family.clone(), OperationRef::new(ns.clone(), "signal", Some("holochain-signal")).unwrap(), SourceAuthentication::CryptographicallyAuthenticated, VerificationCapability::CorroborationAvailable, ConsistencyModel::Eventual, FreshnessModel::EventDriven, OrderingGuarantee::OrderedPerSource, RevisionSemantics::AppendOnly, true, true);
        let effect = EffectCapability::new(family, OperationRef::new(ns, "call", Some("zome")).unwrap(), EffectClass::Update, IdempotencyClass::QueryAfterWrite, PreflightSupport::Remote, FinalityCapability::ConfirmationTrackable, CompensationSupport::SeparateEffectDeclared, true);
        let auth = AuthenticationRequirement::new("holochain-capability", ProtocolUserRequirement::NotUsed, ProtocolUserRequirement::NotUsed, Some("cell-zome-fn")).unwrap();
        let reconciliation = ReconciliationCapabilities { query_by_attempt: false, query_by_native_id: true, idempotency_key: false, event_followup: true, finality_observation: true, reorg_tracking: false, unknown_outcome: UnknownOutcomeHandling::QueryByNativeIdentity };
        (connector, ProtocolBinding::new("holochain", "0.7.0").unwrap(), observation, effect, auth, reconciliation)
    }
    fn manifest() -> ConnectorManifest { let (c, p, o, e, a, r) = fixture(); ConnectorManifest::new(c, "native-observe-effect-v1", p, vec![o], vec![e], vec![a], r).unwrap() }

    #[test] fn golden_vector() { assert_eq!(manifest().digest().to_hex(), "71999264383969516149738867ad87fdb16947e8fdb0f0c6043b2136f69d4a0e"); }
    #[test] fn protocol_mutation_changes_identity() { let baseline = manifest(); let (c, _, o, e, a, r) = fixture(); let changed = ConnectorManifest::new(c, "native-observe-effect-v1", ProtocolBinding::new("holochain", "0.7.1").unwrap(), vec![o], vec![e], vec![a], r).unwrap(); assert_ne!(baseline.digest(), changed.digest()); }
    #[test] fn duplicate_capability_fails_closed() { let (c, p, o, e, a, r) = fixture(); assert_eq!(ConnectorManifest::new(c, "native-observe-effect-v1", p, vec![o.clone(), o], vec![e], vec![a], r), Err(ManifestError::DuplicateCapability("observation"))); }
    #[test] fn effect_does_not_imply_credentials() { let (c, p, _, e, _, r) = fixture(); let m = ConnectorManifest::new(c, "public-effect", p, vec![], vec![e], vec![], r).unwrap(); assert!(m.authentication().is_empty()); assert_eq!(m.effects().len(), 1); }
    #[test] fn unsupported_schema_fails_closed() { assert_eq!(ManifestSchema::try_from(2), Err(ManifestError::UnsupportedSchemaVersion(2))); }
    #[test] fn no_capability_fails_closed() { let (c, p, _, _, _, r) = fixture(); assert_eq!(ConnectorManifest::new(c, "empty", p, vec![], vec![], vec![], r), Err(ManifestError::NoCapabilities)); }
}
