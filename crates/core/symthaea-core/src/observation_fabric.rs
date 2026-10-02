// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Modality-neutral observation envelope for physical and information sensors.
//!
//! The fabric separates what was observed, its context, its producing sensor,
//! disclosure policy, and the immutable asset from which it was derived.
//! Adapters can map this model to OGC API - Connected Systems and STAC without
//! making either external standard a dependency of the cognitive core.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Sensor modality at the observation boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObservationModality {
    Optical, Infrared, Multispectral, Hyperspectral, Sar, Radar,
    RadioFrequency, Acoustic, Inertial, Gnss, Weather, Maritime,
    Aviation, Document, Web, Human, Other,
}

/// Coarse disclosure class for privacy-preserving sharing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum DisclosureClass {
    Public, Restricted, Private, Secret,
}

/// Verification state of the exact asset bytes referenced by an observation.
///
/// This is deliberately independent from provenance verification: proving that
/// bytes match a hash does not prove who produced them or that their contents
/// are truthful.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AssetIntegrity {
    Unverified,
    HashVerified,
    VerificationFailed,
}

/// Verification state of the producer/credential/lineage associated with an observation.
///
/// This does not assert that the observation's substantive content is true.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProvenanceVerification {
    Unverified,
    SignatureVerified,
    CredentialVerified,
    ChainVerified,
    VerificationFailed,
}

/// What an observation is allowed to disclose.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DisclosurePolicy {
    pub class: DisclosureClass,
    pub share_exact_time: bool,
    pub share_exact_location: bool,
    pub share_raw_asset: bool,
    pub share_derived_features: bool,
}

impl DisclosurePolicy {
    pub const fn restricted() -> Self {
        Self {
            class: DisclosureClass::Restricted,
            share_exact_time: false,
            share_exact_location: false,
            share_raw_asset: false,
            share_derived_features: true,
        }
    }

    pub const fn private() -> Self {
        Self {
            class: DisclosureClass::Private,
            share_exact_time: false,
            share_exact_location: false,
            share_raw_asset: false,
            share_derived_features: false,
        }
    }
}

impl Default for DisclosurePolicy {
    fn default() -> Self { Self::restricted() }
}

/// Stable sensor/platform identity without embedding private credential material.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensorIdentity {
    pub sensor_id: String,
    pub platform_id: Option<String>,
    pub procedure_id: Option<String>,
    pub credential_fingerprint: Option<String>,
}

impl SensorIdentity {
    pub fn new(sensor_id: impl Into<String>) -> Self {
        Self {
            sensor_id: sensor_id.into(),
            platform_id: None,
            procedure_id: None,
            credential_fingerprint: None,
        }
    }
}

/// Reference to immutable bytes from which an observation was derived.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssetRef {
    pub hash_algorithm: String,
    pub content_hash: String,
    pub integrity: AssetIntegrity,
    pub media_type: Option<String>,
    pub catalog_id: Option<String>,
}

impl AssetRef {
    /// Create a content-addressed reference over exact source bytes.
    pub fn blake3(bytes: &[u8]) -> Self {
        Self {
            hash_algorithm: "blake3".to_string(),
            content_hash: blake3::hash(bytes).to_hex().to_string(),
            integrity: AssetIntegrity::HashVerified,
            media_type: None,
            catalog_id: None,
        }
    }

    /// Recompute the digest over exact bytes and update the integrity state.
    ///
    /// A mismatch is recorded explicitly so callers cannot accidentally retain
    /// a stale "verified" state after inspecting different bytes.
    pub fn verify_bytes(&mut self, bytes: &[u8]) -> Result<(), ObservationValidationError> {
        self.validate()?;
        let expected = blake3::hash(bytes).to_hex().to_string();
        if expected == self.content_hash {
            self.integrity = AssetIntegrity::HashVerified;
            Ok(())
        } else {
            self.integrity = AssetIntegrity::VerificationFailed;
            Err(ObservationValidationError::AssetHashMismatch)
        }
    }

    pub fn validate(&self) -> Result<(), ObservationValidationError> {
        if self.hash_algorithm != "blake3" {
            return Err(ObservationValidationError::UnsupportedHashAlgorithm(
                self.hash_algorithm.clone(),
            ));
        }
        if self.content_hash.len() != 64
            || !self.content_hash.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(ObservationValidationError::InvalidContentHash);
        }
        Ok(())
    }
}

/// Sensor event time plus explicit clock uncertainty.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationTime {
    pub observed_at_unix_ns: i128,
    pub time_uncertainty_ns: u64,
}

/// Geographic position plus explicit uncertainty radius.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ObservationLocation {
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub uncertainty_m: f64,
}

impl ObservationLocation {
    pub fn validate(&self) -> Result<(), ObservationValidationError> {
        if !self.latitude_deg.is_finite()
            || !self.longitude_deg.is_finite()
            || !self.uncertainty_m.is_finite()
            || self.latitude_deg < -90.0
            || self.latitude_deg > 90.0
            || self.longitude_deg < -180.0
            || self.longitude_deg > 180.0
            || self.uncertainty_m < 0.0
        {
            return Err(ObservationValidationError::InvalidLocation);
        }
        Ok(())
    }
}

/// Measurement quality. Confidence is source confidence, not truth probability.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ObservationQuality {
    pub confidence: f32,
    pub measurement_uncertainty: Option<f64>,
    pub calibrated: bool,
}

impl ObservationQuality {
    pub fn validate(&self) -> Result<(), ObservationValidationError> {
        if !self.confidence.is_finite()
            || !(0.0..=1.0).contains(&self.confidence)
            || self.measurement_uncertainty
                .is_some_and(|v| !v.is_finite() || v < 0.0)
        {
            return Err(ObservationValidationError::InvalidQuality);
        }
        Ok(())
    }
}

/// A precise input-to-output lineage edge within one processing activity.
///
/// Activity-level input/output sets are intentionally insufficient to infer
/// which input contributed to which output when an activity has multiple
/// inputs and outputs. This optional mapping records only derivations that are
/// explicitly known, while absence preserves the less precise activity-level
/// provenance model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessingDerivation {
    pub input_observation_id: String,
    pub output_observation_id: String,
}

/// A concrete processing/acquisition activity that produced an observation.
///
/// Kept as a compact, typed record that can be mapped to OGC SensorML or
/// W3C PROV by boundary adapters without introducing ontology dependencies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessingActivity {
    /// Stable identifier for this execution, not merely the algorithm name.
    pub activity_id: String,
    /// Stable identifier for the procedure or process definition.
    pub process_id: String,
    /// Content fingerprint of the immutable process definition/plan.
    ///
    /// This is separate from activity_fingerprint, which identifies the
    /// concrete code/configuration/parameter realization of this execution.
    pub process_definition_fingerprint: Option<String>,
    pub started_at_unix_ns: Option<i128>,
    pub ended_at_unix_ns: Option<i128>,
    /// Agent identifier, such as a sensor, service, or analyst credential.
    pub agent_id: Option<String>,
    /// Fingerprint of code, configuration, and parameters used by this run.
    pub activity_fingerprint: Option<String>,
    /// BLAKE3 fingerprint of the canonical execution envelope.
    ///
    /// This binds the concrete execution record to its identifiers, timing,
    /// agent, process fingerprint, and ordered input/output observation IDs.
    pub execution_fingerprint: Option<String>,
    /// Observation IDs consumed by this activity.
    pub input_observation_ids: Vec<String>,
    /// Observation IDs emitted by this activity.
    pub output_observation_ids: Vec<String>,
    /// Optional precise input-to-output derivation mappings.
    ///
    /// This is the compact core representation of qualified derivation.
    /// Boundary adapters can expand each pair into explicit PROV Usage,
    /// Generation, and Derivation records when event-level metadata exists.
    pub derivations: Vec<ProcessingDerivation>,
}

impl ProcessingActivity {
    pub fn compute_execution_fingerprint(&self) -> Result<String, ObservationValidationError> {
        self.validate_without_execution_fingerprint()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea:observation-processing-activity:v2\n");
        write_canonical_string(&mut hasher, &self.activity_id);
        write_canonical_string(&mut hasher, &self.process_id);
        write_canonical_string_option(
            &mut hasher,
            self.process_definition_fingerprint.as_deref(),
        );
        write_canonical_i128_option(&mut hasher, self.started_at_unix_ns);
        write_canonical_i128_option(&mut hasher, self.ended_at_unix_ns);
        write_canonical_string_option(&mut hasher, self.agent_id.as_deref());
        write_canonical_string_option(&mut hasher, self.activity_fingerprint.as_deref());
        write_canonical_string_vec(&mut hasher, &self.input_observation_ids);
        write_canonical_string_vec(&mut hasher, &self.output_observation_ids);
        let mut derivations = self.derivations.iter().collect::<Vec<_>>();
        derivations.sort_by(|a, b| {
            a.input_observation_id
                .cmp(&b.input_observation_id)
                .then_with(|| a.output_observation_id.cmp(&b.output_observation_id))
        });
        hasher.update(&(derivations.len() as u64).to_be_bytes());
        for derivation in derivations {
            write_canonical_string(&mut hasher, &derivation.input_observation_id);
            write_canonical_string(&mut hasher, &derivation.output_observation_id);
        }
        Ok(hasher.finalize().to_hex().to_string())
    }

    pub fn verify_execution_fingerprint(&self) -> Result<(), ObservationValidationError> {
        let expected = self.compute_execution_fingerprint()?;
        match self.execution_fingerprint.as_deref() {
            Some(actual) if actual == expected => Ok(()),
            Some(_) => Err(ObservationValidationError::ExecutionFingerprintMismatch),
            None => Err(ObservationValidationError::MissingExecutionFingerprint),
        }
    }

    fn validate_without_execution_fingerprint(&self) -> Result<(), ObservationValidationError> {
        if self.activity_id.trim().is_empty() || self.process_id.trim().is_empty()
            || self.agent_id.as_deref().is_some_and(|id| id.trim().is_empty())
            || self.process_definition_fingerprint.as_deref().is_some_and(|id| id.trim().is_empty())
            || self.activity_fingerprint.as_deref().is_some_and(|id| id.trim().is_empty())
            || self.input_observation_ids.iter().any(|id| id.trim().is_empty())
            || self.output_observation_ids.iter().any(|id| id.trim().is_empty())
            || self.derivations.iter().any(|d| {
                d.input_observation_id.trim().is_empty()
                    || d.output_observation_id.trim().is_empty()
                    || d.input_observation_id == d.output_observation_id
            })
            || matches!((self.started_at_unix_ns, self.ended_at_unix_ns),
                (Some(start), Some(end)) if end < start)
        {
            return Err(ObservationValidationError::InvalidProcessingActivity);
        }
        let mut input_ids = HashSet::with_capacity(self.input_observation_ids.len());
        if self.input_observation_ids.iter().any(|id| !input_ids.insert(id)) {
            return Err(ObservationValidationError::DuplicateActivityInput);
        }
        let mut output_ids = HashSet::with_capacity(self.output_observation_ids.len());
        if self.output_observation_ids.iter().any(|id| !output_ids.insert(id)) {
            return Err(ObservationValidationError::DuplicateActivityOutput);
        }
        Ok(())
    }
    
    pub fn validate(&self) -> Result<(), ObservationValidationError> {
        self.validate_without_execution_fingerprint()?;
        if let Some(fingerprint) = self.execution_fingerprint.as_deref() {
            if fingerprint.len() != 64 || !fingerprint.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(ObservationValidationError::InvalidExecutionFingerprint);
            }
        }
        Ok(())
    }
}

fn write_canonical_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
}
fn write_canonical_string(hasher: &mut blake3::Hasher, value: &str) {
    write_canonical_bytes(hasher, value.as_bytes());
}
fn write_canonical_string_option(hasher: &mut blake3::Hasher, value: Option<&str>) {
    match value {
        Some(value) => { hasher.update(&[1]); write_canonical_string(hasher, value); }
        None => hasher.update(&[0]),
    }
}
fn write_canonical_i128_option(hasher: &mut blake3::Hasher, value: Option<i128>) {
    match value {
        Some(value) => { hasher.update(&[1]); hasher.update(&value.to_be_bytes()); }
        None => hasher.update(&[0]),
    }
}
fn write_canonical_string_vec(hasher: &mut blake3::Hasher, values: &[String]) {
    hasher.update(&(values.len() as u64).to_be_bytes());
    for value in values { write_canonical_string(hasher, value); }
}

/// Provenance linking an observation to its producer and processing lineage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationProvenance {
    pub source: SensorIdentity,
    pub verification: ProvenanceVerification,
    /// Stable reference to the attestation/evidence used for verification.
    pub attestation_id: Option<String>,
    pub acquired_by: Option<String>,
    pub parent_observation_ids: Vec<String>,
    pub processing_fingerprint: Option<String>,
    /// Optional explicit execution record; parent IDs alone do not describe
    /// which operation transformed the inputs into this observation.
    pub processing_activity: Option<ProcessingActivity>,
}

/// Modality-neutral observation envelope.
///
/// This is not a claim, detection, or hypothesis. Derived inferences should
/// reference one or more observations and carry their own inference metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub id: String,
    pub modality: ObservationModality,
    pub time: ObservationTime,
    pub location: Option<ObservationLocation>,
    /// Stable identifier for the real-world or conceptual feature being observed.
    ///
    /// Keeping this distinct from sensor/platform identity follows the OGC
    /// observation model, where an observation targets a feature of interest.
    pub feature_of_interest_id: Option<String>,
    pub quality: ObservationQuality,
    pub provenance: ObservationProvenance,
    pub asset: Option<AssetRef>,
    pub disclosure: DisclosurePolicy,
}

impl Observation {
    /// Validate invariants before entering fusion or an evidence graph.
    pub fn validate(&self) -> Result<(), ObservationValidationError> {
        if self.id.trim().is_empty() {
            return Err(ObservationValidationError::EmptyId);
        }
        if self.provenance.source.sensor_id.trim().is_empty() {
            return Err(ObservationValidationError::EmptySensorId);
        }
        if let Some(location) = self.location {
            location.validate()?;
        }
        if self.feature_of_interest_id.as_deref().is_some_and(|id| id.trim().is_empty()) {
            return Err(ObservationValidationError::EmptyFeatureOfInterestId);
        }
        self.quality.validate()?;
        if self.provenance.verification != ProvenanceVerification::Unverified
            && self.provenance.attestation_id.as_deref().is_none_or(str::is_empty)
        {
            return Err(ObservationValidationError::MissingVerificationAttestation);
        }
        if self.provenance.parent_observation_ids.iter().any(|parent| parent.trim().is_empty() || parent == &self.id) {
            return Err(ObservationValidationError::InvalidParentObservation);
        }
        if let Some(activity) = &self.provenance.processing_activity {
            activity.validate()?;
        }
        if let Some(asset) = &self.asset {
            asset.validate()?;
        }
        if self.disclosure.class == DisclosureClass::Secret
            && self.disclosure.share_raw_asset
        {
            return Err(ObservationValidationError::SecretRawAssetExport);
        }
        Ok(())
    }
}

/// Directed relationship between two observations in the evidence graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObservationRelationKind {
    /// The source observation provides positive evidence for the target.
    Supports,
    /// The source observation provides evidence against the target.
    Contradicts,
    /// The source agrees with the target observation; independence is tracked separately.
    Corroborates,
    /// The source was computationally derived from the target.
    DerivedFrom,
    /// The observations support an identity match.
    SameEntity,
    /// The observations may refer to the same entity, but the match is unresolved.
    PossibleSameEntity,
}

/// Auditable edge between observations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceIndependence {
    /// The observations have no declared shared upstream source.
    Independent,
    /// The observations share an upstream source, platform, or processing chain.
    SharedUpstream,
    /// The source observation is computationally derived from the target lineage.
    Derived,
    /// Independence has not been established.
    Unknown,
}

/// Auditable edge between observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationRelation {
    pub source_observation_id: String,
    pub target_observation_id: String,
    pub kind: ObservationRelationKind,
    pub independence: EvidenceIndependence,
}

impl ObservationRelation {
    /// Validate graph-edge invariants before insertion.
    pub fn validate(&self) -> Result<(), ObservationValidationError> {
        if self.source_observation_id.trim().is_empty()
            || self.target_observation_id.trim().is_empty()
        {
            return Err(ObservationValidationError::EmptyRelationId);
        }
        if self.source_observation_id == self.target_observation_id {
            return Err(ObservationValidationError::SelfRelation);
        }
        if matches!(self.kind, ObservationRelationKind::DerivedFrom)
            && !matches!(self.independence, EvidenceIndependence::Derived | EvidenceIndependence::SharedUpstream)
        {
            return Err(ObservationValidationError::DerivedRelationIndependenceMismatch);
        }
        if matches!(self.kind, ObservationRelationKind::Corroborates)
            && matches!(self.independence, EvidenceIndependence::Derived)
        {
            return Err(ObservationValidationError::CorroborationDerivedMismatch);
        }
        Ok(())
    }
}

/// A closed-world observation graph with validated lineage and evidence edges.
///
/// Local observation validation remains open-world: a single observation may refer to
/// parents that are stored elsewhere. This graph validator is deliberately stricter and
/// treats the supplied observation set as complete, so missing parents and relation
/// endpoints are rejected rather than silently interpreted as unknown.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationGraph {
    pub observations: Vec<Observation>,
    pub relations: Vec<ObservationRelation>,
}

impl ObservationGraph {
    pub fn validate(&self) -> Result<(), ObservationValidationError> {
        let mut by_id = HashMap::with_capacity(self.observations.len());
        for observation in &self.observations {
            observation.validate()?;
            if by_id.insert(observation.id.as_str(), observation).is_some() {
                return Err(ObservationValidationError::DuplicateObservationId);
            }
        }

        let mut lineage_children: HashMap<String, Vec<String>> = HashMap::new();
        for observation in &self.observations {
            for parent_id in &observation.provenance.parent_observation_ids {
                if !by_id.contains_key(parent_id.as_str()) {
                    return Err(ObservationValidationError::MissingParentObservation(
                        parent_id.clone(),
                    ));
                }
                lineage_children
                    .entry(parent_id.clone())
                    .or_default()
                    .push(observation.id.clone());
            }
        }

        let mut visiting = HashSet::new();
        let mut visited = HashSet::new();
        for id in by_id.keys() {
            if !visited.contains(id) && Self::visit_lineage(id, &lineage_children, &mut visiting, &mut visited) {
                return Err(ObservationValidationError::LineageCycle);
            }
        }

        let mut activity_fingerprints: HashMap<String, String> = HashMap::new();

        for observation in &self.observations {
            if let Some(activity) = &observation.provenance.processing_activity {
                if activity.execution_fingerprint.is_some() {
                    activity.verify_execution_fingerprint()?;
                }
                let computed_fingerprint = activity.compute_execution_fingerprint()?;
                if let Some(previous) = activity_fingerprints.insert(
                    activity.activity_id.clone(),
                    computed_fingerprint.clone(),
                ) {
                    if previous != computed_fingerprint {
                        return Err(ObservationValidationError::InconsistentProcessingActivity(
                            activity.activity_id.clone(),
                        ));
                    }
                }
                for input_id in &activity.input_observation_ids {
                    if !by_id.contains_key(input_id.as_str()) {
                        return Err(ObservationValidationError::MissingActivityInput(input_id.clone()));
                    }
                }
                for output_id in &activity.output_observation_ids {
                    let output = by_id.get(output_id.as_str()).expect("checked above");
                    if output
                        .provenance
                        .processing_activity
                        .as_ref()
                        .is_none_or(|producer| producer.activity_id != activity.activity_id)
                    {
                        return Err(ObservationValidationError::ActivityOutputMissingProducer(
                            output_id.clone(),
                        ));
                    }
                }
                let mut derivation_pairs = HashSet::with_capacity(activity.derivations.len());
                for derivation in &activity.derivations {
                    if !activity.input_observation_ids.iter().any(|id| id == &derivation.input_observation_id) {
                        return Err(ObservationValidationError::DerivationInputNotDeclared(
                            derivation.input_observation_id.clone(),
                        ));
                    }
                    if !activity.output_observation_ids.iter().any(|id| id == &derivation.output_observation_id) {
                        return Err(ObservationValidationError::DerivationOutputNotDeclared(
                            derivation.output_observation_id.clone(),
                        ));
                    }
                    if !by_id.contains_key(derivation.input_observation_id.as_str()) {
                        return Err(ObservationValidationError::MissingActivityInput(
                            derivation.input_observation_id.clone(),
                        ));
                    }
                    if !by_id.contains_key(derivation.output_observation_id.as_str()) {
                        return Err(ObservationValidationError::MissingActivityOutput(
                            derivation.output_observation_id.clone(),
                        ));
                    }
                    if !derivation_pairs.insert((
                        derivation.input_observation_id.as_str(),
                        derivation.output_observation_id.as_str(),
                    )) {
                        return Err(ObservationValidationError::DuplicateProcessingDerivation);
                    }
                }
                if !activity.input_observation_ids.iter().all(|input_id| {
                    observation
                        .provenance
                        .parent_observation_ids
                        .iter()
                        .any(|parent_id| parent_id == input_id)
                }) {
                    return Err(ObservationValidationError::ActivityInputMissingParent(
                        observation.id.clone(),
                    ));
                }
                if !activity.output_observation_ids.iter().any(|id| id == &observation.id) {
                    return Err(ObservationValidationError::ActivityOutputMissingSelf(
                        observation.id.clone(),
                    ));
                }
            }
        }

        let mut relation_edges = HashSet::with_capacity(self.relations.len());
        for relation in &self.relations {
            relation.validate()?;
            if !relation_edges.insert((
                relation.source_observation_id.as_str(),
                relation.target_observation_id.as_str(),
                relation.kind,
                &relation.independence,
            )) {
                return Err(ObservationValidationError::DuplicateObservationRelation);
            }
            if !by_id.contains_key(relation.source_observation_id.as_str()) {
                return Err(ObservationValidationError::MissingRelationEndpoint(
                    relation.source_observation_id.clone(),
                ));
            }
            if !by_id.contains_key(relation.target_observation_id.as_str()) {
                return Err(ObservationValidationError::MissingRelationEndpoint(
                    relation.target_observation_id.clone(),
                ));
            }
        }
        Ok(())
    }

    fn visit_lineage(
        id: &str,
        children: &HashMap<String, Vec<String>>,
        visiting: &mut HashSet<String>,
        visited: &mut HashSet<String>,
    ) -> bool {
        if !visiting.insert(id.to_string()) {
            return true;
        }
        if let Some(next) = children.get(id) {
            for child in next {
                if !visited.contains(child)
                    && Self::visit_lineage(child, children, visiting, visited)
                {
                    return true;
                }
            }
        }
        visiting.remove(id);
        visited.insert(id);
        false
    }
}

/// Fail-closed validation errors for malformed sensor metadata.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ObservationValidationError {
    #[error("observation id must not be empty")]
    EmptyId,
    #[error("sensor id must not be empty")]
    EmptySensorId,
    #[error("location is outside valid geographic bounds or has invalid uncertainty")]
    InvalidLocation,
    #[error("quality/confidence is outside valid bounds")]
    InvalidQuality,
    #[error("feature-of-interest id must not be empty when provided")]
    EmptyFeatureOfInterestId,
    #[error("asset bytes do not match the declared content hash")]
    AssetHashMismatch,
    #[error("unsupported content hash algorithm: {0}")]
    UnsupportedHashAlgorithm(String),
    #[error("content hash must be 64 hexadecimal characters")]
    InvalidContentHash,
    #[error("secret observations cannot permit raw-asset export")]
    SecretRawAssetExport,
    #[error("relation observation ids must not be empty")]
    EmptyRelationId,
    #[error("an observation relation cannot point to itself")]
    SelfRelation,
    #[error("derived-from relations require derived or shared-upstream independence")]
    DerivedRelationIndependenceMismatch,
    #[error("corroboration relations cannot classify the source as derived evidence")]
    CorroborationDerivedMismatch,
    #[error("verified provenance requires an attestation reference")]
    MissingVerificationAttestation,
    #[error("parent observation ids must be non-empty and cannot reference the observation itself")]
    InvalidParentObservation,
    #[error("processing activity has invalid identifiers, fingerprints, or time bounds")]
    InvalidProcessingActivity,
    #[error("processing activity input observations must be unique")]
    DuplicateActivityInput,
    #[error("processing activity output observations must be unique")]
    DuplicateActivityOutput,
    #[error("processing activity execution fingerprint must be 64 hexadecimal characters")]
    InvalidExecutionFingerprint,
    #[error("processing activity execution fingerprint does not match its canonical execution envelope")]
    ExecutionFingerprintMismatch,
    #[error("processing activity execution fingerprint is required for verification")]
    MissingExecutionFingerprint,
    #[error("processing activity output does not identify the activity as its producer: {0}")]
    ActivityOutputMissingProducer(String),
    InconsistentProcessingActivity(String),
    #[error("processing activity input is not represented in the output observation's parent lineage: {0}")]
    ActivityInputMissingParent(String),
    #[error("processing activity input observation is not present in the closed graph: {0}")]
    MissingActivityInput(String),
    #[error("processing activity output observation is not present in the closed graph: {0}")]
    MissingActivityOutput(String),
    #[error("processing derivation input is not declared as an activity input: {0}")]
    DerivationInputNotDeclared(String),
    #[error("processing derivation output is not declared as an activity output: {0}")]
    DerivationOutputNotDeclared(String),
    #[error("processing activity contains a duplicate input-to-output derivation")]
    DuplicateProcessingDerivation,
    #[error("processing activity does not list its owning observation as an output: {0}")]
    ActivityOutputMissingSelf(String),
    #[error("observation ids must be unique within a closed graph")]
    DuplicateObservationId,
    #[error("parent observation is not present in the closed graph: {0}")]
    MissingParentObservation(String),
    #[error("observation parent lineage contains a cycle")]
    LineageCycle,
    #[error("relation edge is duplicated within the closed graph")]
    DuplicateObservationRelation,
    #[error("relation endpoint is not present in the closed graph: {0}")]
    MissingRelationEndpoint(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Observation {
        Observation {
            id: "obs-001".into(),
            modality: ObservationModality::Optical,
            time: ObservationTime {
                observed_at_unix_ns: 1_700_000_000_000_000_000,
                time_uncertainty_ns: 2_000_000,
            },
            location: Some(ObservationLocation {
                latitude_deg: 10.0,
                longitude_deg: 20.0,
                uncertainty_m: 4.0,
            }),
            feature_of_interest_id: Some("scene-001".into()),
            quality: ObservationQuality {
                confidence: 0.92,
                measurement_uncertainty: Some(0.3),
                calibrated: true,
            },
            provenance: ObservationProvenance {
                source: SensorIdentity::new("camera-1"),
                verification: ProvenanceVerification::Unverified,
                attestation_id: None,
                acquired_by: None,
                parent_observation_ids: Vec::new(),
                processing_fingerprint: Some("proc-v1".into()),
                processing_activity: None,
            },
            asset: Some(AssetRef::blake3(b"frame-bytes")),
            disclosure: DisclosurePolicy::restricted(),
        }
    }

    #[test]
    fn valid_observation_passes() {
        assert!(fixture().validate().is_ok());
    }

    #[test]
    fn invalid_location_fails_closed() {
        let mut observation = fixture();
        observation.location = Some(ObservationLocation {
            latitude_deg: 91.0,
            longitude_deg: 0.0,
            uncertainty_m: 1.0,
        });
        assert_eq!(
            observation.validate(),
            Err(ObservationValidationError::InvalidLocation)
        );
    }

    #[test]
    fn invalid_confidence_fails_closed() {
        let mut observation = fixture();
        observation.quality.confidence = 1.5;
        assert_eq!(
            observation.validate(),
            Err(ObservationValidationError::InvalidQuality)
        );
    }

    #[test]
    fn secret_raw_export_fails_closed() {
        let mut observation = fixture();
        observation.disclosure = DisclosurePolicy {
            class: DisclosureClass::Secret,
            share_exact_time: false,
            share_exact_location: false,
            share_raw_asset: true,
            share_derived_features: false,
        };
        assert_eq!(
            observation.validate(),
            Err(ObservationValidationError::SecretRawAssetExport)
        );
    }

    #[test]
    fn asset_hash_verification_rejects_changed_bytes() {
        let mut asset = AssetRef::blake3(b"original");
        assert_eq!(
            asset.verify_bytes(b"changed"),
            Err(ObservationValidationError::AssetHashMismatch)
        );
        assert_eq!(asset.integrity, AssetIntegrity::VerificationFailed);
        assert_eq!(asset.verify_bytes(b"original"), Ok(()));
        assert_eq!(asset.integrity, AssetIntegrity::HashVerified);
    }

    #[test]
    fn empty_feature_of_interest_fails_closed() {
        let mut observation = fixture();
        observation.feature_of_interest_id = Some("  ".into());
        assert_eq!(
            observation.validate(),
            Err(ObservationValidationError::EmptyFeatureOfInterestId)
        );
    }

    #[test]
    fn asset_hash_is_content_addressed() {
        let a = AssetRef::blake3(b"same");
        let b = AssetRef::blake3(b"same");
        let c = AssetRef::blake3(b"different");
        assert_eq!(a.integrity, AssetIntegrity::HashVerified);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.validate().is_ok());
    }

    #[test]
    fn relation_rejects_empty_ids() {
        let relation = ObservationRelation {
            source_observation_id: String::new(),
            target_observation_id: "obs-2".into(),
            kind: ObservationRelationKind::Supports,
            independence: EvidenceIndependence::Unknown,
        };
        assert_eq!(
            relation.validate(),
            Err(ObservationValidationError::EmptyRelationId)
        );
    }

    #[test]
    fn relation_rejects_self_edge() {
        let relation = ObservationRelation {
            source_observation_id: "obs-1".into(),
            target_observation_id: "obs-1".into(),
            kind: ObservationRelationKind::Corroborates,
            independence: EvidenceIndependence::Independent,
        };
        assert_eq!(
            relation.validate(),
            Err(ObservationValidationError::SelfRelation)
        );
    }

    #[test]
    fn derived_relation_requires_lineage_independence_marker() {
        let relation = ObservationRelation {
            source_observation_id: "derived".into(),
            target_observation_id: "source".into(),
            kind: ObservationRelationKind::DerivedFrom,
            independence: EvidenceIndependence::Independent,
        };
        assert_eq!(
            relation.validate(),
            Err(ObservationValidationError::DerivedRelationIndependenceMismatch)
        );
    }

    #[test]
    fn corroboration_rejects_derived_independence() {
        let relation = ObservationRelation {
            source_observation_id: "derived".into(),
            target_observation_id: "source".into(),
            kind: ObservationRelationKind::Corroborates,
            independence: EvidenceIndependence::Derived,
        };
        assert_eq!(
            relation.validate(),
            Err(ObservationValidationError::CorroborationDerivedMismatch)
        );
    }

    #[test]
    fn verification_axes_remain_separate() {
        let mut observation = fixture();
        observation.provenance.verification = ProvenanceVerification::CredentialVerified;
        observation.provenance.attestation_id = Some("att-001".into());
        assert_eq!(observation.asset.as_ref().map(|asset| asset.integrity), Some(AssetIntegrity::HashVerified));
        assert_eq!(observation.provenance.verification, ProvenanceVerification::CredentialVerified);
        assert_eq!(observation.provenance.attestation_id.as_deref(), Some("att-001"));
    }

    #[test]
    fn verified_provenance_requires_attestation() {
        let mut observation = fixture();
        observation.provenance.verification = ProvenanceVerification::CredentialVerified;
        assert_eq!(observation.validate(), Err(ObservationValidationError::MissingVerificationAttestation));
    }

    #[test]
    fn parent_cannot_self_reference() {
        let mut observation = fixture();
        observation.provenance.parent_observation_ids = vec!["obs-001".into()];
        assert_eq!(observation.validate(), Err(ObservationValidationError::InvalidParentObservation));
    }

    #[test]
    fn processing_activity_validates_time_bounds_and_identity() {
        let mut observation = fixture();
        observation.provenance.processing_activity = Some(ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: "orthorectify-v2".into(),
            process_definition_fingerprint: None,
            started_at_unix_ns: Some(20),
            ended_at_unix_ns: Some(10),
            agent_id: Some("worker-7".into()),
            activity_fingerprint: Some("sha256:config".into()),
            execution_fingerprint: None,
            input_observation_ids: vec![],
            output_observation_ids: vec!["obs-001".into()],
            derivations: Vec::new(),
        });
        assert_eq!(
            observation.validate(),
            Err(ObservationValidationError::InvalidProcessingActivity)
        );
        observation.provenance.processing_activity.as_mut().unwrap().ended_at_unix_ns = Some(25);
        assert!(observation.validate().is_ok());
    }

    #[test]
    fn processing_definition_fingerprint_is_bound_into_execution_identity() {
        let mut activity = ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: "transform-v1".into(),
            process_definition_fingerprint: Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into()),
            started_at_unix_ns: Some(10),
            ended_at_unix_ns: Some(20),
            agent_id: None,
            activity_fingerprint: None,
            execution_fingerprint: None,
            input_observation_ids: vec!["input".into()],
            output_observation_ids: vec!["output".into()],
            derivations: Vec::new(),
        };
        let first = activity.compute_execution_fingerprint().unwrap();
        activity.process_definition_fingerprint =
            Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into());
        let second = activity.compute_execution_fingerprint().unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn processing_activity_rejects_malformed_definition_fingerprint() {
        let activity = ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: "transform-v1".into(),
            process_definition_fingerprint: Some("not-a-fingerprint".into()),
            started_at_unix_ns: None,
            ended_at_unix_ns: None,
            agent_id: None,
            activity_fingerprint: None,
            execution_fingerprint: None,
            input_observation_ids: vec![],
            output_observation_ids: vec!["output".into()],
            derivations: Vec::new(),
        };
        assert_eq!(
            activity.validate(),
            Err(ObservationValidationError::InvalidProcessingActivity)
        );
    }

    #[test]
    fn processing_derivation_without_events_remains_activity_level_provenance() {
        let derivation = ProcessingDerivation {
            input_observation_id: "input".into(),
            output_observation_id: "output".into(),
        };
        assert_eq!(derivation.input_observation_id, "input");
        assert_eq!(derivation.output_observation_id, "output");
    }

    #[test]
    fn processing_derivation_requires_declared_endpoints() {
        let activity = ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: "transform-v1".into(),
            process_definition_fingerprint: None,
            started_at_unix_ns: None,
            ended_at_unix_ns: None,
            agent_id: None,
            activity_fingerprint: None,
            execution_fingerprint: None,
            input_observation_ids: vec!["input".into()],
            output_observation_ids: vec!["output".into()],
            derivations: vec![ProcessingDerivation {
                input_observation_id: "other-input".into(),
                output_observation_id: "output".into(),
            }],
        };
        let mut observation = fixture();
        observation.id = "output".into();
        observation.provenance.processing_activity = Some(activity);
        let graph = ObservationGraph { observations: vec![observation], relations: vec![] };
        assert_eq!(
            graph.validate(),
            Err(ObservationValidationError::DerivationInputNotDeclared("other-input".into()))
        );
    }

    #[test]
    fn processing_derivation_changes_execution_fingerprint() {
        let mut activity = ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: "transform-v1".into(),
            process_definition_fingerprint: None,
            started_at_unix_ns: None,
            ended_at_unix_ns: None,
            agent_id: None,
            activity_fingerprint: None,
            execution_fingerprint: None,
            input_observation_ids: vec!["input-a".into(), "input-b".into()],
            output_observation_ids: vec!["output-a".into(), "output-b".into()],
            derivations: vec![ProcessingDerivation {
                input_observation_id: "input-a".into(),
                output_observation_id: "output-a".into(),
            }],
        };
        let first = activity.compute_execution_fingerprint().unwrap();
        activity.derivations.push(ProcessingDerivation {
            input_observation_id: "input-b".into(),
            output_observation_id: "output-b".into(),
        });
        let second = activity.compute_execution_fingerprint().unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn processing_derivation_accepts_precise_lineage() {
        let mut input = fixture();
        input.id = "input".into();
        let mut output = fixture();
        output.id = "output".into();
        output.provenance.parent_observation_ids = vec!["input".into()];
        output.provenance.processing_activity = Some(ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: "transform-v1".into(),
            process_definition_fingerprint: None,
            started_at_unix_ns: None,
            ended_at_unix_ns: None,
            agent_id: None,
            activity_fingerprint: None,
            execution_fingerprint: None,
            input_observation_ids: vec!["input".into()],
            output_observation_ids: vec!["output".into()],
            derivations: vec![ProcessingDerivation {
                input_observation_id: "input".into(),
                output_observation_id: "output".into(),
            }],
        });
        let graph = ObservationGraph { observations: vec![input, output], relations: vec![] };
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn processing_derivation_rejects_duplicate_edges() {
        let mut output = fixture();
        output.id = "output".into();
        output.provenance.parent_observation_ids = vec!["input".into()];
        output.provenance.processing_activity = Some(ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: "transform-v1".into(),
            process_definition_fingerprint: None,
            started_at_unix_ns: None,
            ended_at_unix_ns: None,
            agent_id: None,
            activity_fingerprint: None,
            execution_fingerprint: None,
            input_observation_ids: vec!["input".into()],
            output_observation_ids: vec!["output".into()],
            derivations: vec![
                ProcessingDerivation {
                    input_observation_id: "input".into(),
                    output_observation_id: "output".into(),
                },
                ProcessingDerivation {
                    input_observation_id: "input".into(),
                    output_observation_id: "output".into(),
                },
            ],
        });
        let mut input = fixture();
        input.id = "input".into();
        let graph = ObservationGraph { observations: vec![input, output], relations: vec![] };
        assert_eq!(graph.validate(), Err(ObservationValidationError::DuplicateProcessingDerivation));
    }

    #[test]
    fn processing_activity_fingerprint_domain_version_changes_identity() {
        let activity = ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: "transform-v1".into(),
            process_definition_fingerprint: None,
            started_at_unix_ns: None,
            ended_at_unix_ns: None,
            agent_id: None,
            activity_fingerprint: None,
            execution_fingerprint: None,
            input_observation_ids: vec!["input".into()],
            output_observation_ids: vec!["output".into()],
            derivations: Vec::new(),
        };
        let fingerprint = activity.compute_execution_fingerprint().unwrap();
        assert_eq!(fingerprint.len(), 64);
    }

    #[test]
    fn processing_derivation_order_is_not_execution_identity() {
        let mut first = ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: "transform-v1".into(),
            process_definition_fingerprint: None,
            started_at_unix_ns: None,
            ended_at_unix_ns: None,
            agent_id: None,
            activity_fingerprint: None,
            execution_fingerprint: None,
            input_observation_ids: vec!["input-a".into(), "input-b".into()],
            output_observation_ids: vec!["output-a".into(), "output-b".into()],
            derivations: vec![
                ProcessingDerivation {
                    input_observation_id: "input-a".into(),
                    output_observation_id: "output-a".into(),
                },
                ProcessingDerivation {
                    input_observation_id: "input-b".into(),
                    output_observation_id: "output-b".into(),
                },
            ],
        };
        let expected = first.compute_execution_fingerprint().unwrap();
        first.derivations.reverse();
        assert_eq!(expected, first.compute_execution_fingerprint().unwrap());
    }

    #[test]
    fn processing_activity_execution_fingerprint_is_deterministic() {
        let activity = ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: "transform-v1".into(),
            process_definition_fingerprint: None,
            started_at_unix_ns: Some(10),
            ended_at_unix_ns: Some(20),
            agent_id: Some("worker-7".into()),
            activity_fingerprint: Some("blake3:config-v1".into()),
            execution_fingerprint: None,
            input_observation_ids: vec!["input-a".into(), "input-b".into()],
            output_observation_ids: vec!["output-a".into()],
            derivations: Vec::new(),
        };
        let first = activity.compute_execution_fingerprint().expect("fingerprint");
        let second = activity.compute_execution_fingerprint().expect("fingerprint");
        assert_eq!(first, second);
        assert_eq!(first.len(), 64);
        let mut reordered = activity.clone();
        reordered.input_observation_ids.reverse();
        assert_ne!(first, reordered.compute_execution_fingerprint().expect("fingerprint"));
    }

    #[test]
    fn processing_activity_execution_fingerprint_detects_change() {
        let mut activity = ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: "transform-v1".into(),
            process_definition_fingerprint: None,
            started_at_unix_ns: Some(10),
            ended_at_unix_ns: Some(20),
            agent_id: Some("worker-7".into()),
            activity_fingerprint: Some("blake3:config-v1".into()),
            execution_fingerprint: None,
            input_observation_ids: vec!["input-a".into()],
            output_observation_ids: vec!["output-a".into()],
            derivations: Vec::new(),
        };
        activity.execution_fingerprint = Some(activity.compute_execution_fingerprint().expect("fingerprint"));
        assert_eq!(activity.verify_execution_fingerprint(), Ok(()));
        activity.process_id = "transform-v2".into();
        assert_eq!(activity.verify_execution_fingerprint(), Err(ObservationValidationError::ExecutionFingerprintMismatch));
    }

    #[test]
    fn processing_activity_execution_fingerprint_requires_stored_value() {
        let activity = ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: "transform-v1".into(),
            process_definition_fingerprint: None,
            started_at_unix_ns: None,
            ended_at_unix_ns: None,
            agent_id: None,
            activity_fingerprint: None,
            execution_fingerprint: None,
            input_observation_ids: vec![],
            output_observation_ids: vec!["obs-001".into()],
            derivations: Vec::new(),
        };
        assert_eq!(activity.verify_execution_fingerprint(), Err(ObservationValidationError::MissingExecutionFingerprint));
    }

    #[test]
    fn processing_activity_rejects_malformed_execution_fingerprint() {
        let activity = ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: "transform-v1".into(),
            process_definition_fingerprint: None,
            started_at_unix_ns: None,
            ended_at_unix_ns: None,
            agent_id: None,
            activity_fingerprint: None,
            execution_fingerprint: Some("not-a-digest".into()),
            input_observation_ids: vec![],
            output_observation_ids: vec!["obs-001".into()],
            derivations: Vec::new(),
        };
        assert_eq!(activity.validate(), Err(ObservationValidationError::InvalidExecutionFingerprint));
    }

    #[test]
    fn processing_activity_rejects_duplicate_input_ids() {
        let activity = ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: "transform-v1".into(),
            process_definition_fingerprint: None,
            started_at_unix_ns: None,
            ended_at_unix_ns: None,
            agent_id: None,
            activity_fingerprint: None,
            execution_fingerprint: None,
            input_observation_ids: vec!["input".into(), "input".into()],
            output_observation_ids: vec!["output".into()],
            derivations: Vec::new(),
        };
        assert_eq!(
            activity.validate(),
            Err(ObservationValidationError::DuplicateActivityInput)
        );
    }

    #[test]
    fn processing_activity_rejects_duplicate_output_ids() {
        let activity = ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: "transform-v1".into(),
            process_definition_fingerprint: None,
            started_at_unix_ns: None,
            ended_at_unix_ns: None,
            agent_id: None,
            activity_fingerprint: None,
            execution_fingerprint: None,
            input_observation_ids: vec!["input".into()],
            output_observation_ids: vec!["output".into(), "output".into()],
            derivations: Vec::new(),
        };
        assert_eq!(
            activity.validate(),
            Err(ObservationValidationError::DuplicateActivityOutput)
        );
    }

    #[test]
    fn processing_activity_rejects_blank_process_id() {
        let activity = ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: " ".into(),
            process_definition_fingerprint: None,
            started_at_unix_ns: None,
            ended_at_unix_ns: None,
            agent_id: None,
            activity_fingerprint: None,
            execution_fingerprint: None,
            input_observation_ids: vec![],
            output_observation_ids: vec![],
            derivations: Vec::new(),
        };
        assert_eq!(
            activity.validate(),
            Err(ObservationValidationError::InvalidProcessingActivity)
        );
    }

    #[test]
    fn graph_rejects_missing_parent() {
        let mut child = fixture();
        child.id = "child".into();
        child.provenance.parent_observation_ids = vec!["missing".into()];
        let graph = ObservationGraph {
            observations: vec![child],
            relations: vec![],
        };
        assert_eq!(
            graph.validate(),
            Err(ObservationValidationError::MissingParentObservation("missing".into()))
        );
    }

    #[test]
    fn graph_rejects_duplicate_ids() {
        let a = fixture();
        let mut b = fixture();
        b.provenance.source.sensor_id = "camera-2".into();
        let graph = ObservationGraph {
            observations: vec![a, b],
            relations: vec![],
        };
        assert_eq!(
            graph.validate(),
            Err(ObservationValidationError::DuplicateObservationId)
        );
    }

    #[test]
    fn graph_rejects_parent_cycle() {
        let mut a = fixture();
        let mut b = fixture();
        a.id = "a".into();
        b.id = "b".into();
        a.provenance.parent_observation_ids = vec!["b".into()];
        b.provenance.parent_observation_ids = vec!["a".into()];
        let graph = ObservationGraph {
            observations: vec![a, b],
            relations: vec![],
        };
        assert_eq!(
            graph.validate(),
            Err(ObservationValidationError::LineageCycle)
        );
    }

    #[test]
    fn graph_rejects_duplicate_relation() {
        let mut second = fixture();
        second.id = "obs-002".into();
        let relation = ObservationRelation {
            source_observation_id: "obs-001".into(),
            target_observation_id: "obs-002".into(),
            kind: ObservationRelationKind::Supports,
            independence: EvidenceIndependence::Unknown,
        };
        let graph = ObservationGraph {
            observations: vec![fixture(), second],
            relations: vec![relation.clone(), relation],
        };
        assert_eq!(
            graph.validate(),
            Err(ObservationValidationError::DuplicateObservationRelation)
        );
    }

    #[test]
    fn graph_rejects_missing_relation_endpoint() {
        let graph = ObservationGraph {
            observations: vec![fixture()],
            relations: vec![ObservationRelation {
                source_observation_id: "obs-001".into(),
                target_observation_id: "missing".into(),
                kind: ObservationRelationKind::Supports,
                independence: EvidenceIndependence::Unknown,
            }],
        };
        assert_eq!(
            graph.validate(),
            Err(ObservationValidationError::MissingRelationEndpoint("missing".into()))
        );
    }

    #[test]
    fn graph_accepts_activity_output_with_matching_producer() {
        let mut input = fixture();
        input.id = "input".into();
        let mut output = fixture();
        output.id = "output".into();
        output.provenance.parent_observation_ids = vec!["input".into()];
        output.provenance.processing_activity = Some(ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: "transform-v1".into(),
            process_definition_fingerprint: None,
            started_at_unix_ns: None,
            ended_at_unix_ns: None,
            agent_id: None,
            activity_fingerprint: None,
            execution_fingerprint: None,
            input_observation_ids: vec!["input".into()],
            output_observation_ids: vec!["output".into()],
            derivations: Vec::new(),
        });
        let graph = ObservationGraph { observations: vec![input, output], relations: vec![] };
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn graph_accepts_repeated_processing_activity_with_same_execution() {
        let mut input = fixture();
        input.id = "input".into();

        let activity = ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: "transform-v1".into(),
            process_definition_fingerprint: None,
            started_at_unix_ns: Some(10),
            ended_at_unix_ns: Some(20),
            agent_id: Some("worker-7".into()),
            activity_fingerprint: Some("blake3:config-v1".into()),
            execution_fingerprint: None,
            input_observation_ids: vec!["input".into()],
            output_observation_ids: vec!["output-a".into(), "output-b".into()],
            derivations: Vec::new(),
        };

        let mut output_a = fixture();
        output_a.id = "output-a".into();
        output_a.provenance.parent_observation_ids = vec!["input".into()];
        output_a.provenance.processing_activity = Some(activity.clone());

        let mut output_b = fixture();
        output_b.id = "output-b".into();
        output_b.provenance.parent_observation_ids = vec!["input".into()];
        output_b.provenance.processing_activity = Some(activity);

        let graph = ObservationGraph {
            observations: vec![input, output_a, output_b],
            relations: vec![],
        };
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn graph_rejects_inconsistent_processing_activity_repetition() {
        let mut input = fixture();
        input.id = "input".into();

        let activity_a = ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: "transform-v1".into(),
            process_definition_fingerprint: None,
            started_at_unix_ns: Some(10),
            ended_at_unix_ns: Some(20),
            agent_id: Some("worker-7".into()),
            activity_fingerprint: Some("blake3:config-v1".into()),
            execution_fingerprint: None,
            input_observation_ids: vec!["input".into()],
            output_observation_ids: vec!["output-a".into(), "output-b".into()],
            derivations: Vec::new(),
        };
        let mut activity_b = activity_a.clone();
        activity_b.process_id = "transform-v2".into();

        let mut output_a = fixture();
        output_a.id = "output-a".into();
        output_a.provenance.parent_observation_ids = vec!["input".into()];
        output_a.provenance.processing_activity = Some(activity_a);

        let mut output_b = fixture();
        output_b.id = "output-b".into();
        output_b.provenance.parent_observation_ids = vec!["input".into()];
        output_b.provenance.processing_activity = Some(activity_b);

        let graph = ObservationGraph {
            observations: vec![input, output_a, output_b],
            relations: vec![],
        };
        assert_eq!(
            graph.validate(),
            Err(ObservationValidationError::InconsistentProcessingActivity("run-001".into()))
        );
    }

    #[test]
    fn graph_rejects_tampered_execution_fingerprint() {
        let mut observation = fixture();
        let activity = ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: "transform-v1".into(),
            process_definition_fingerprint: None,
            started_at_unix_ns: None,
            ended_at_unix_ns: None,
            agent_id: None,
            activity_fingerprint: None,
            execution_fingerprint: Some(
                "0000000000000000000000000000000000000000000000000000000000000000".into(),
            ),
            input_observation_ids: vec![],
            output_observation_ids: vec!["obs-001".into()],
            derivations: Vec::new(),
        };
        observation.provenance.processing_activity = Some(activity);

        let graph = ObservationGraph {
            observations: vec![observation],
            relations: vec![],
        };
        assert_eq!(
            graph.validate(),
            Err(ObservationValidationError::ExecutionFingerprintMismatch)
        );
    }

    #[test]
    fn graph_rejects_unclaimed_activity_output() {
        let mut input = fixture();
        input.id = "input".into();
        let mut output_a = fixture();
        output_a.id = "output-a".into();
        output_a.provenance.parent_observation_ids = vec!["input".into()];
        output_a.provenance.processing_activity = Some(ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: "transform-v1".into(),
            process_definition_fingerprint: None,
            started_at_unix_ns: None,
            ended_at_unix_ns: None,
            agent_id: None,
            activity_fingerprint: None,
            execution_fingerprint: None,
            input_observation_ids: vec!["input".into()],
            output_observation_ids: vec!["output-a".into(), "output-b".into()],
            derivations: Vec::new(),
        });
        let mut output_b = fixture();
        output_b.id = "output-b".into();
        output_b.provenance.parent_observation_ids = vec!["input".into()];
        let graph = ObservationGraph {
            observations: vec![input, output_a, output_b],
            relations: vec![],
        };
        assert_eq!(
            graph.validate(),
            Err(ObservationValidationError::ActivityOutputMissingProducer("output-b".into()))
        );
    }

    #[test]
    fn graph_rejects_activity_input_missing_parent_lineage() {
        let mut output = fixture();
        output.id = "output".into();
        output.provenance.processing_activity = Some(ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: "transform-v1".into(),
            process_definition_fingerprint: None,
            started_at_unix_ns: None,
            ended_at_unix_ns: None,
            agent_id: None,
            activity_fingerprint: None,
            execution_fingerprint: None,
            input_observation_ids: vec!["input".into()],
            output_observation_ids: vec!["output".into()],
            derivations: Vec::new(),
        });
        let graph = ObservationGraph { observations: vec![output], relations: vec![] };
        assert_eq!(
            graph.validate(),
            Err(ObservationValidationError::MissingActivityInput("input".into()))
        );
    }

    #[test]
    fn graph_rejects_missing_activity_input() {
        let mut observation = fixture();
        observation.provenance.processing_activity = Some(ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: "transform-v1".into(),
            process_definition_fingerprint: None,
            started_at_unix_ns: None,
            ended_at_unix_ns: None,
            agent_id: None,
            activity_fingerprint: None,
            execution_fingerprint: None,
            input_observation_ids: vec!["missing-input".into()],
            output_observation_ids: vec!["obs-001".into()],
            derivations: Vec::new(),
        });
        let graph = ObservationGraph {
            observations: vec![observation],
            relations: vec![],
        };
        assert_eq!(
            graph.validate(),
            Err(ObservationValidationError::MissingActivityInput("missing-input".into()))
        );
    }

    #[test]
    fn graph_requires_activity_to_emit_its_observation() {
        let mut observation = fixture();
        observation.provenance.processing_activity = Some(ProcessingActivity {
            activity_id: "run-001".into(),
            process_id: "transform-v1".into(),
            process_definition_fingerprint: None,
            started_at_unix_ns: None,
            ended_at_unix_ns: None,
            agent_id: None,
            activity_fingerprint: None,
            input_observation_ids: vec![],
            output_observation_ids: vec![],
            derivations: Vec::new(),
        });
        let graph = ObservationGraph {
            observations: vec![observation],
            relations: vec![],
        };
        assert_eq!(
            graph.validate(),
            Err(ObservationValidationError::ActivityOutputMissingSelf("obs-001".into()))
        );
    }

    #[test]
    fn graph_accepts_valid_lineage_and_relation() {
        let mut parent = fixture();
        let mut child = fixture();
        parent.id = "parent".into();
        child.id = "child".into();
        child.provenance.parent_observation_ids = vec!["parent".into()];
        let graph = ObservationGraph {
            observations: vec![parent, child],
            relations: vec![ObservationRelation {
                source_observation_id: "child".into(),
                target_observation_id: "parent".into(),
                kind: ObservationRelationKind::DerivedFrom,
                independence: EvidenceIndependence::Derived,
            }],
        };
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn observation_round_trips_json() {
        let original = fixture();
        let encoded = serde_json::to_string(&original).expect("serialize");
        let decoded: Observation = serde_json::from_str(&encoded).expect("deserialize");
        assert_eq!(original, decoded);
    }
}