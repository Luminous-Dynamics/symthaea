// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence references and observation/belief separation.

use std::collections::BTreeSet;
use std::num::NonZeroU64;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    ClockInstant, ReferenceFrameId, SpatialValidationError, SPATIAL_WORLD_SCHEMA_VERSION,
};

/// Stable non-zero namespace for evidence identities.
///
/// `EvidenceId` is intentionally source-local. The namespace prevents unrelated
/// evidence owners from colliding merely because they reused the same local ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EvidenceNamespaceId(NonZeroU64);

impl EvidenceNamespaceId {
    /// Construct a non-zero evidence namespace.
    pub fn new(value: u64) -> Result<Self, SpatialValidationError> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(SpatialValidationError::ZeroId {
                kind: "evidence-namespace",
            })
    }

    /// Return the numeric namespace identity.
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Stable non-zero identity local to one evidence namespace and generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EvidenceId(NonZeroU64);

impl EvidenceId {
    /// Construct a non-zero source-local evidence identity.
    pub fn new(value: u64) -> Result<Self, SpatialValidationError> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(SpatialValidationError::ZeroId { kind: "evidence" })
    }

    /// Return the numeric source-local identity.
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Supported immutable digest algorithms for exact spatial evidence content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EvidenceDigestAlgorithm {
    /// SHA-256 over the exact evidence claim/observation bytes.
    Sha256,
    /// BLAKE3 over the exact evidence claim/observation bytes.
    Blake3,
}

/// Validated immutable digest of the exact evidence claim/observation content.
///
/// The wire form is canonical lowercase `sha256:<64 hex>` or
/// `blake3:<64 hex>`. This is content identity only; it grants no observation,
/// belief, truth, or action authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EvidenceDigest {
    algorithm: EvidenceDigestAlgorithm,
    bytes: [u8; 32],
}

impl EvidenceDigest {
    /// Parse and validate a strict 256-bit evidence digest.
    pub fn parse(value: &str) -> Result<Self, SpatialValidationError> {
        let Some((algorithm, hex)) = value.split_once(':') else {
            return Err(SpatialValidationError::MalformedDigest);
        };
        let algorithm = match algorithm {
            "sha256" => EvidenceDigestAlgorithm::Sha256,
            "blake3" => EvidenceDigestAlgorithm::Blake3,
            _ => return Err(SpatialValidationError::MalformedDigest),
        };
        if hex.len() != 64 {
            return Err(SpatialValidationError::MalformedDigest);
        }

        let raw = hex.as_bytes();
        let mut bytes = [0u8; 32];
        for (index, output) in bytes.iter_mut().enumerate() {
            let hi = decode_hex_nibble(raw[index * 2]).ok_or(SpatialValidationError::MalformedDigest)?;
            let lo = decode_hex_nibble(raw[index * 2 + 1])
                .ok_or(SpatialValidationError::MalformedDigest)?;
            *output = (hi << 4) | lo;
        }
        Ok(Self { algorithm, bytes })
    }

    /// Digest algorithm.
    pub const fn algorithm(self) -> EvidenceDigestAlgorithm {
        self.algorithm
    }

    /// Raw 256-bit digest bytes.
    pub const fn bytes(self) -> [u8; 32] {
        self.bytes
    }

    fn canonical_string(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let prefix = match self.algorithm {
            EvidenceDigestAlgorithm::Sha256 => "sha256:",
            EvidenceDigestAlgorithm::Blake3 => "blake3:",
        };
        let mut output = String::with_capacity(prefix.len() + 64);
        output.push_str(prefix);
        for byte in self.bytes {
            output.push(HEX[(byte >> 4) as usize] as char);
            output.push(HEX[(byte & 0x0f) as usize] as char);
        }
        output
    }
}

impl Serialize for EvidenceDigest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.canonical_string())
    }
}

impl<'de> Deserialize<'de> for EvidenceDigest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Self::parse(&raw).map_err(D::Error::custom)
    }
}

fn decode_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Descriptive epistemic origin retained by the spatial layer.
///
/// This classification is ordinary data. It is **not** authentication,
/// authorization, scientific admission, or proof that a source actually has the
/// stated origin. The canonical evidence authority remains external to this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpatialEvidenceKind {
    /// Claimed direct measurement from a sensor boundary.
    SensorMeasurement,
    /// Claimed perceptual quantity derived from upstream observations.
    DerivedPerception,
    /// Claim reported by an external agent/source.
    ExternalReport,
    /// Internally generated prediction about a future or unobserved world state.
    InternalPrediction,
    /// Internally generated simulation/counterfactual state.
    InternalSimulation,
}

/// Non-authorizing reference to externally owned evidence.
///
/// Identity is the triple `(namespace, id, generation)`, while `claim_digest`
/// binds that identity to immutable content. The generation must change when an
/// evidence producer resets, reuses source-local IDs, or otherwise changes the
/// identity semantics of its local evidence stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRef {
    namespace: EvidenceNamespaceId,
    id: EvidenceId,
    generation: NonZeroU64,
    claim_digest: EvidenceDigest,
    kind: SpatialEvidenceKind,
}

impl EvidenceRef {
    /// Construct an ordinary source-, generation-, and content-qualified evidence reference.
    ///
    /// This does not admit the referenced evidence for any epistemic use.
    pub fn new(
        namespace: EvidenceNamespaceId,
        id: EvidenceId,
        generation: u64,
        claim_digest: EvidenceDigest,
        kind: SpatialEvidenceKind,
    ) -> Result<Self, SpatialValidationError> {
        let generation = NonZeroU64::new(generation).ok_or(SpatialValidationError::ZeroId {
            kind: "evidence-generation",
        })?;
        Ok(Self {
            namespace,
            id,
            generation,
            claim_digest,
            kind,
        })
    }

    /// Namespace that qualifies the source-local evidence ID.
    pub const fn namespace(self) -> EvidenceNamespaceId {
        self.namespace
    }

    /// Source-local evidence identity.
    pub const fn id(self) -> EvidenceId {
        self.id
    }

    /// Evidence-stream generation that qualifies the source-local ID.
    pub const fn generation(self) -> u64 {
        self.generation.get()
    }

    /// Immutable digest of the exact referenced evidence content.
    pub const fn claim_digest(self) -> EvidenceDigest {
        self.claim_digest
    }

    /// Preserved descriptive origin classification.
    pub const fn kind(self) -> SpatialEvidenceKind {
        self.kind
    }
}

/// Opaque evidence admission required to construct a local spatial observation.
///
/// V1 intentionally exposes **no public constructor** and does not implement
/// `Serialize` or `Deserialize`. A later adapter must consume the canonical
/// verified/admitted observation evidence and issue this token inside this crate's
/// reviewed trust boundary.
pub struct AdmittedObservationEvidence {
    evidence: EvidenceRef,
}

impl AdmittedObservationEvidence {
    /// Evidence reference whose observation use was admitted upstream.
    pub const fn evidence(&self) -> EvidenceRef {
        self.evidence
    }

    #[cfg(test)]
    fn for_test(evidence: EvidenceRef) -> Self {
        Self { evidence }
    }
}

/// Opaque evidence admission required to support a runtime spatial belief.
///
/// This token is distinct from observation admission: evidence may be a valid
/// observation record yet still lack authority for belief support. V1 exposes no
/// public issuer and intentionally provides no Serde implementation. A later
/// adapter should issue it only from the canonical bounded `BeliefSupport` use.
pub struct AdmittedBeliefSupportEvidence {
    evidence: EvidenceRef,
}

impl AdmittedBeliefSupportEvidence {
    /// Evidence reference admitted for spatial-belief support.
    pub const fn evidence(&self) -> EvidenceRef {
        self.evidence
    }

    #[cfg(test)]
    fn for_test(evidence: EvidenceRef) -> Self {
        Self { evidence }
    }
}

/// One admitted local spatial observation with explicit frame and clock lineage.
///
/// A raw `EvidenceRef` is structurally insufficient. Construction is crate-private
/// so external callers cannot attach admitted evidence to arbitrary frame/time/value
/// state. A future canonical-evidence adapter must bind the exact spatial statement
/// and construct this object inside the reviewed crate boundary.
///
/// This runtime type is intentionally neither serializable nor deserializable.
/// Persistence explicitly downgrades it to `SpatialObservationRecord<T>`.
#[derive(Debug)]
pub struct SpatialObservation<T> {
    evidence: EvidenceRef,
    observed_at: ClockInstant,
    frame: ReferenceFrameId,
    value: T,
}

impl<T> SpatialObservation<T> {
    /// Construct an observation inside the reviewed spatial admission boundary.
    pub(crate) fn new(
        admitted: AdmittedObservationEvidence,
        observed_at: ClockInstant,
        frame: ReferenceFrameId,
        value: T,
    ) -> Self {
        Self {
            evidence: admitted.evidence,
            observed_at,
            frame,
            value,
        }
    }

    /// Evidence that was admitted for this observation.
    pub const fn evidence(&self) -> EvidenceRef {
        self.evidence
    }

    /// Clock-domain-qualified observation time.
    pub const fn observed_at(&self) -> ClockInstant {
        self.observed_at
    }

    /// Reference frame in which this observation is expressed.
    pub const fn frame(&self) -> ReferenceFrameId {
        self.frame
    }

    /// Borrow the observed value.
    pub const fn value(&self) -> &T {
        &self.value
    }

    /// Consume the observation and downgrade it to a non-authorizing persisted record.
    pub fn into_record(self) -> SpatialObservationRecord<T> {
        SpatialObservationRecord {
            schema_version: SPATIAL_WORLD_SCHEMA_VERSION,
            evidence: self.evidence,
            observed_at: self.observed_at,
            frame: self.frame,
            value: self.value,
        }
    }

    /// Consume the observation and return only its raw value.
    ///
    /// The returned value carries no observation admission or provenance authority.
    pub fn into_value(self) -> T {
        self.value
    }
}

/// Serializable, non-authorizing record of a prior admitted observation.
///
/// This record preserves provenance, frame, time, and payload but carries no
/// `AdmittedObservationEvidence`. Deserializing it cannot recreate a runtime
/// `SpatialObservation<T>` without a fresh canonical admission step.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpatialObservationRecord<T> {
    schema_version: u16,
    evidence: EvidenceRef,
    observed_at: ClockInstant,
    frame: ReferenceFrameId,
    value: T,
}

impl<T> SpatialObservationRecord<T> {
    /// Closed-world spatial record schema version.
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    /// Non-authorizing evidence reference retained by the record.
    pub const fn evidence(&self) -> EvidenceRef {
        self.evidence
    }

    /// Clock-domain-qualified observation time retained by the record.
    pub const fn observed_at(&self) -> ClockInstant {
        self.observed_at
    }

    /// Reference frame retained by the record.
    pub const fn frame(&self) -> ReferenceFrameId {
        self.frame
    }

    /// Borrow the recorded value.
    pub const fn value(&self) -> &T {
        &self.value
    }

    /// Consume the record into ordinary, non-authorizing parts.
    pub fn into_parts(self) -> (EvidenceRef, ClockInstant, ReferenceFrameId, T) {
        (self.evidence, self.observed_at, self.frame, self.value)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpatialObservationRecordWire<T> {
    schema_version: u16,
    evidence: EvidenceRef,
    observed_at: ClockInstant,
    frame: ReferenceFrameId,
    value: T,
}

impl<'de, T> Deserialize<'de> for SpatialObservationRecord<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SpatialObservationRecordWire::<T>::deserialize(deserializer)?;
        validate_schema_version(wire.schema_version).map_err(D::Error::custom)?;
        Ok(Self {
            schema_version: wire.schema_version,
            evidence: wire.evidence,
            observed_at: wire.observed_at,
            frame: wire.frame,
            value: wire.value,
        })
    }
}

/// Runtime spatial belief backed only by explicitly admitted support evidence.
///
/// Construction is crate-private so external callers cannot attach an admitted
/// support capability to an arbitrary state. A future fusion/admission module must
/// construct the exact supported state inside this crate's reviewed boundary.
///
/// This type is intentionally neither serializable nor deserializable. Persistence
/// downgrades to `SpatialBeliefRecord<T>` and requires fresh admission after reload.
#[derive(Debug)]
pub struct SpatialBelief<T> {
    frame: ReferenceFrameId,
    updated_at: ClockInstant,
    state: T,
    support: Vec<EvidenceRef>,
}

impl<T> SpatialBelief<T> {
    /// Construct a runtime belief inside the reviewed spatial fusion boundary.
    pub(crate) fn new(
        frame: ReferenceFrameId,
        updated_at: ClockInstant,
        state: T,
        support: Vec<AdmittedBeliefSupportEvidence>,
    ) -> Result<Self, SpatialValidationError> {
        if support.is_empty() {
            return Err(SpatialValidationError::EmptyBeliefSupport);
        }
        let refs: Vec<EvidenceRef> = support.into_iter().map(|item| item.evidence).collect();
        validate_support(&refs)?;
        Ok(Self {
            frame,
            updated_at,
            state,
            support: refs,
        })
    }

    /// Reference frame of the believed state.
    pub const fn frame(&self) -> ReferenceFrameId {
        self.frame
    }

    /// Clock-domain-qualified belief update time.
    pub const fn updated_at(&self) -> ClockInstant {
        self.updated_at
    }

    /// Borrow the believed state.
    pub const fn state(&self) -> &T {
        &self.state
    }

    /// Source-, generation-, and content-qualified evidence references supporting the belief.
    pub fn support(&self) -> &[EvidenceRef] {
        &self.support
    }

    /// Consume the runtime belief and downgrade it to non-authorizing persisted data.
    pub fn into_record(self) -> SpatialBeliefRecord<T> {
        SpatialBeliefRecord {
            schema_version: SPATIAL_WORLD_SCHEMA_VERSION,
            frame: self.frame,
            updated_at: self.updated_at,
            state: self.state,
            support: self.support,
        }
    }
}

/// Serializable, non-authorizing record of a prior spatial belief.
///
/// Deserialization revalidates structural identity invariants but does **not**
/// recreate `AdmittedBeliefSupportEvidence` or a runtime `SpatialBelief<T>`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpatialBeliefRecord<T> {
    schema_version: u16,
    frame: ReferenceFrameId,
    updated_at: ClockInstant,
    state: T,
    support: Vec<EvidenceRef>,
}

impl<T> SpatialBeliefRecord<T> {
    /// Closed-world spatial record schema version.
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    /// Reference frame recorded with the prior belief.
    pub const fn frame(&self) -> ReferenceFrameId {
        self.frame
    }

    /// Clock-domain-qualified time recorded with the prior belief.
    pub const fn updated_at(&self) -> ClockInstant {
        self.updated_at
    }

    /// Borrow the recorded state.
    pub const fn state(&self) -> &T {
        &self.state
    }

    /// Non-authorizing evidence references recorded with the prior belief.
    pub fn support(&self) -> &[EvidenceRef] {
        &self.support
    }

    /// Consume the record and return its state plus non-authorizing support metadata.
    pub fn into_parts(self) -> (ReferenceFrameId, ClockInstant, T, Vec<EvidenceRef>) {
        (self.frame, self.updated_at, self.state, self.support)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpatialBeliefRecordWire<T> {
    schema_version: u16,
    frame: ReferenceFrameId,
    updated_at: ClockInstant,
    state: T,
    support: Vec<EvidenceRef>,
}

impl<'de, T> Deserialize<'de> for SpatialBeliefRecord<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SpatialBeliefRecordWire::<T>::deserialize(deserializer)?;
        validate_schema_version(wire.schema_version).map_err(D::Error::custom)?;
        if wire.support.is_empty() {
            return Err(D::Error::custom(SpatialValidationError::EmptyBeliefSupport));
        }
        validate_support(&wire.support).map_err(D::Error::custom)?;
        Ok(Self {
            schema_version: wire.schema_version,
            frame: wire.frame,
            updated_at: wire.updated_at,
            state: wire.state,
            support: wire.support,
        })
    }
}

fn validate_schema_version(found: u16) -> Result<(), SpatialValidationError> {
    if found == SPATIAL_WORLD_SCHEMA_VERSION {
        Ok(())
    } else {
        Err(SpatialValidationError::UnsupportedSchemaVersion { found })
    }
}

fn validate_support(support: &[EvidenceRef]) -> Result<(), SpatialValidationError> {
    let mut seen = BTreeSet::new();
    for evidence in support {
        let identity = (
            evidence.namespace().get(),
            evidence.id().get(),
            evidence.generation(),
        );
        if !seen.insert(identity) {
            return Err(SpatialValidationError::DuplicateEvidenceRef {
                namespace: identity.0,
                id: identity.1,
                generation: identity.2,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ClockDomainId, ClockNamespaceId, MetricPoint3, ReferenceFrameNamespaceId};

    const SHA_A: &str =
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str =
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn namespace(value: u64) -> EvidenceNamespaceId {
        EvidenceNamespaceId::new(value).unwrap()
    }

    fn digest(value: &str) -> EvidenceDigest {
        EvidenceDigest::parse(value).unwrap()
    }

    fn frame() -> ReferenceFrameId {
        ReferenceFrameId::new(ReferenceFrameNamespaceId::new(1).unwrap(), 1, 1).unwrap()
    }

    fn instant() -> ClockInstant {
        let domain = ClockDomainId::new(ClockNamespaceId::new(1).unwrap(), 1, 1).unwrap();
        ClockInstant::new(domain, 10)
    }

    fn evidence(
        namespace_id: u64,
        local_id: u64,
        generation: u64,
        claim_digest: EvidenceDigest,
        kind: SpatialEvidenceKind,
    ) -> EvidenceRef {
        EvidenceRef::new(
            namespace(namespace_id),
            EvidenceId::new(local_id).unwrap(),
            generation,
            claim_digest,
            kind,
        )
        .unwrap()
    }

    #[test]
    fn digest_contract_is_strict_and_canonical() {
        let parsed = EvidenceDigest::parse(SHA_A).unwrap();
        assert_eq!(parsed.algorithm(), EvidenceDigestAlgorithm::Sha256);
        assert_eq!(serde_json::to_string(&parsed).unwrap(), format!("\"{SHA_A}\""));

        let uppercase = format!("sha256:{}", "A".repeat(64));
        let parsed_uppercase = EvidenceDigest::parse(&uppercase).unwrap();
        assert_eq!(serde_json::to_string(&parsed_uppercase).unwrap(), format!("\"{SHA_A}\""));

        assert!(EvidenceDigest::parse("sha256:decorative").is_err());
        assert!(EvidenceDigest::parse(
            "md5:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )
        .is_err());
        assert!(EvidenceDigest::parse(
            "blake3:gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg"
        )
        .is_err());
    }

    #[test]
    fn observation_consumes_an_opaque_admission_token() {
        let raw = evidence(1, 1, 1, digest(SHA_A), SpatialEvidenceKind::SensorMeasurement);
        let admitted = AdmittedObservationEvidence::for_test(raw);
        let point = MetricPoint3::new(0.0, 0.0, 0.0).unwrap();
        let observation = SpatialObservation::new(admitted, instant(), frame(), point);
        assert_eq!(observation.evidence(), raw);
    }

    #[test]
    fn runtime_observation_downgrades_to_versioned_non_authorizing_record() {
        let raw = evidence(1, 11, 1, digest(SHA_A), SpatialEvidenceKind::SensorMeasurement);
        let admitted = AdmittedObservationEvidence::for_test(raw);
        let point = MetricPoint3::new(1.0, 2.0, 3.0).unwrap();
        let record = SpatialObservation::new(admitted, instant(), frame(), point).into_record();
        assert_eq!(record.schema_version(), SPATIAL_WORLD_SCHEMA_VERSION);
        assert_eq!(record.evidence(), raw);
        let json = serde_json::to_string(&record).unwrap();
        let restored = serde_json::from_str::<SpatialObservationRecord<MetricPoint3>>(&json).unwrap();
        assert_eq!(restored, record);
    }

    #[test]
    fn observation_record_rejects_unknown_stale_or_unsupported_schema() {
        let unknown = r#"{
            "schema_version":1,
            "evidence":{"namespace":1,"id":12,"generation":1,"claim_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","kind":"sensor_measurement"},
            "observed_at":{"domain":{"namespace":1,"local_id":1,"generation":1},"tick":10},
            "frame":{"namespace":1,"local_id":1,"generation":1},
            "value":[0.0,0.0,0.0],
            "admitted":true
        }"#;
        assert!(serde_json::from_str::<SpatialObservationRecord<MetricPoint3>>(unknown).is_err());

        let stale = r#"{
            "schema_version":1,
            "evidence":{"namespace":1,"id":12,"kind":"sensor_measurement"},
            "observed_at":{"domain":{"namespace":1,"local_id":1},"tick":10},
            "frame":{"namespace":1,"local_id":1},
            "value":[0.0,0.0,0.0]
        }"#;
        assert!(serde_json::from_str::<SpatialObservationRecord<MetricPoint3>>(stale).is_err());

        let unsupported = r#"{
            "schema_version":2,
            "evidence":{"namespace":1,"id":12,"generation":1,"claim_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","kind":"sensor_measurement"},
            "observed_at":{"domain":{"namespace":1,"local_id":1,"generation":1},"tick":10},
            "frame":{"namespace":1,"local_id":1,"generation":1},
            "value":[0.0,0.0,0.0]
        }"#;
        assert!(serde_json::from_str::<SpatialObservationRecord<MetricPoint3>>(unsupported).is_err());
    }

    #[test]
    fn belief_consumes_separate_support_admission() {
        let raw = evidence(1, 2, 1, digest(SHA_A), SpatialEvidenceKind::DerivedPerception);
        let admitted = AdmittedBeliefSupportEvidence::for_test(raw);
        let state = MetricPoint3::new(1.0, 2.0, 3.0).unwrap();
        let belief = SpatialBelief::new(frame(), instant(), state, vec![admitted]).unwrap();
        assert_eq!(belief.support(), &[raw]);
    }

    #[test]
    fn runtime_belief_downgrades_to_versioned_non_authorizing_record() {
        let raw = evidence(1, 3, 1, digest(SHA_A), SpatialEvidenceKind::DerivedPerception);
        let admitted = AdmittedBeliefSupportEvidence::for_test(raw);
        let state = MetricPoint3::new(1.0, 2.0, 3.0).unwrap();
        let belief = SpatialBelief::new(frame(), instant(), state, vec![admitted]).unwrap();
        let record = belief.into_record();
        assert_eq!(record.schema_version(), SPATIAL_WORLD_SCHEMA_VERSION);
        assert_eq!(record.support(), &[raw]);
    }

    #[test]
    fn evidence_identity_is_namespace_and_generation_qualified() {
        let a = evidence(1, 7, 1, digest(SHA_A), SpatialEvidenceKind::SensorMeasurement);
        let other_namespace =
            evidence(2, 7, 1, digest(SHA_A), SpatialEvidenceKind::SensorMeasurement);
        let other_generation =
            evidence(1, 7, 2, digest(SHA_A), SpatialEvidenceKind::SensorMeasurement);
        let state = MetricPoint3::new(0.0, 0.0, 0.0).unwrap();
        let belief = SpatialBelief::new(
            frame(),
            instant(),
            state,
            vec![
                AdmittedBeliefSupportEvidence::for_test(a),
                AdmittedBeliefSupportEvidence::for_test(other_namespace),
                AdmittedBeliefSupportEvidence::for_test(other_generation),
            ],
        )
        .unwrap();
        assert_eq!(belief.support().len(), 3);
    }

    #[test]
    fn belief_rejects_reused_identity_even_if_claim_digest_changes() {
        let a = evidence(3, 8, 1, digest(SHA_A), SpatialEvidenceKind::DerivedPerception);
        let changed = evidence(3, 8, 1, digest(SHA_B), SpatialEvidenceKind::DerivedPerception);
        let state = MetricPoint3::new(0.0, 0.0, 0.0).unwrap();
        assert!(SpatialBelief::new(
            frame(),
            instant(),
            state,
            vec![
                AdmittedBeliefSupportEvidence::for_test(a),
                AdmittedBeliefSupportEvidence::for_test(changed),
            ],
        )
        .is_err());
    }

    #[test]
    fn belief_requires_unique_explicit_support() {
        let item = evidence(3, 9, 1, digest(SHA_A), SpatialEvidenceKind::DerivedPerception);
        let state = MetricPoint3::new(0.0, 0.0, 0.0).unwrap();
        assert!(SpatialBelief::new(frame(), instant(), state, vec![]).is_err());

        let state = MetricPoint3::new(0.0, 0.0, 0.0).unwrap();
        assert!(SpatialBelief::new(
            frame(),
            instant(),
            state,
            vec![
                AdmittedBeliefSupportEvidence::for_test(item),
                AdmittedBeliefSupportEvidence::for_test(item),
            ],
        )
        .is_err());
    }

    #[test]
    fn record_deserialization_revalidates_support_but_does_not_re_admit_it() {
        let duplicate = r#"{
            "schema_version":1,
            "frame":{"namespace":1,"local_id":1,"generation":1},
            "updated_at":{"domain":{"namespace":1,"local_id":1,"generation":1},"tick":10},
            "state":[0.0,0.0,0.0],
            "support":[
                {"namespace":4,"id":9,"generation":1,"claim_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","kind":"sensor_measurement"},
                {"namespace":4,"id":9,"generation":1,"claim_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","kind":"sensor_measurement"}
            ]
        }"#;
        assert!(serde_json::from_str::<SpatialBeliefRecord<MetricPoint3>>(duplicate).is_err());

        let valid = r#"{
            "schema_version":1,
            "frame":{"namespace":1,"local_id":1,"generation":1},
            "updated_at":{"domain":{"namespace":1,"local_id":1,"generation":1},"tick":10},
            "state":[0.0,0.0,0.0],
            "support":[{"namespace":4,"id":9,"generation":1,"claim_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","kind":"internal_simulation"}]
        }"#;
        let record = serde_json::from_str::<SpatialBeliefRecord<MetricPoint3>>(valid).unwrap();
        assert_eq!(record.support()[0].kind(), SpatialEvidenceKind::InternalSimulation);
        assert_eq!(record.support()[0].claim_digest(), digest(SHA_A));
    }

    #[test]
    fn belief_record_rejects_unknown_stale_or_unsupported_schema() {
        let unknown = r#"{
            "schema_version":1,
            "frame":{"namespace":1,"local_id":1,"generation":1},
            "updated_at":{"domain":{"namespace":1,"local_id":1,"generation":1},"tick":10},
            "state":[0.0,0.0,0.0],
            "support":[{"namespace":4,"id":9,"generation":1,"claim_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","kind":"sensor_measurement"}],
            "current":true
        }"#;
        assert!(serde_json::from_str::<SpatialBeliefRecord<MetricPoint3>>(unknown).is_err());

        let stale = r#"{
            "schema_version":1,
            "frame":{"namespace":1,"local_id":1},
            "updated_at":{"domain":{"namespace":1,"local_id":1},"tick":10},
            "state":[0.0,0.0,0.0],
            "support":[{"namespace":4,"id":9,"kind":"sensor_measurement"}]
        }"#;
        assert!(serde_json::from_str::<SpatialBeliefRecord<MetricPoint3>>(stale).is_err());

        let unsupported = r#"{
            "schema_version":2,
            "frame":{"namespace":1,"local_id":1,"generation":1},
            "updated_at":{"domain":{"namespace":1,"local_id":1,"generation":1},"tick":10},
            "state":[0.0,0.0,0.0],
            "support":[{"namespace":4,"id":9,"generation":1,"claim_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","kind":"sensor_measurement"}]
        }"#;
        assert!(serde_json::from_str::<SpatialBeliefRecord<MetricPoint3>>(unsupported).is_err());
    }

    #[test]
    fn zero_evidence_namespace_id_and_generation_are_rejected() {
        assert!(EvidenceNamespaceId::new(0).is_err());
        assert!(EvidenceId::new(0).is_err());
        assert!(EvidenceRef::new(
            namespace(1),
            EvidenceId::new(1).unwrap(),
            0,
            digest(SHA_A),
            SpatialEvidenceKind::SensorMeasurement,
        )
        .is_err());
    }
}
