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
fn write_canonical_string_bytes(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn write_canonical_string_vec_bytes(bytes: &mut Vec<u8>, values: &[String]) {
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values { write_canonical_string_bytes(bytes, value); }
}

fn write_canonical_independence_bytes(bytes: &mut Vec<u8>, value: &EvidenceIndependence) {
    bytes.push(match value {
        EvidenceIndependence::Independent => 0,
        EvidenceIndependence::VerifiedIndependent => 1,
        EvidenceIndependence::SharedUpstream => 2,
        EvidenceIndependence::Derived => 3,
        EvidenceIndependence::Unknown => 4,
    });
}

fn write_canonical_independence_basis_bytes(bytes: &mut Vec<u8>, value: &IndependenceBasis) {
    match value {
        IndependenceBasis::SharedSensor { sensor_id } => { bytes.push(0); write_canonical_string_bytes(bytes, sensor_id); }
        IndependenceBasis::SharedPlatform { platform_id } => { bytes.push(1); write_canonical_string_bytes(bytes, platform_id); }
        IndependenceBasis::SharedAncestor { observation_id } => { bytes.push(2); write_canonical_string_bytes(bytes, observation_id); }
        IndependenceBasis::SharedProcessingActivity { activity_id } => { bytes.push(3); write_canonical_string_bytes(bytes, activity_id); }
        IndependenceBasis::IdenticalAsset { hash_algorithm, content_hash } => { bytes.push(4); write_canonical_string_bytes(bytes, hash_algorithm); write_canonical_string_bytes(bytes, content_hash); }
        IndependenceBasis::InsufficientProvenance { coverage } => { bytes.push(5); write_canonical_provenance_coverage_bytes(bytes, coverage); }
        IndependenceBasis::NoSharedProvenance => bytes.push(6),
    }
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

fn write_canonical_provenance_coverage_bytes(bytes: &mut Vec<u8>, value: &ProvenanceCoverage) {
    bytes.push(match value {
        ProvenanceCoverage::Complete => 0,
        ProvenanceCoverage::Partial => 1,
        ProvenanceCoverage::Redacted => 2,
    });
}

fn write_canonical_provenance_coverage(hasher: &mut blake3::Hasher, value: &ProvenanceCoverage) {
    hasher.update(&[match value {
        ProvenanceCoverage::Complete => 0,
        ProvenanceCoverage::Partial => 1,
        ProvenanceCoverage::Redacted => 2,
    }]);
}

fn write_canonical_independence(hasher: &mut blake3::Hasher, value: &EvidenceIndependence) {
    let tag = match value {
        EvidenceIndependence::Independent => 0u8,
        EvidenceIndependence::VerifiedIndependent => 1,
        EvidenceIndependence::SharedUpstream => 2,
        EvidenceIndependence::Derived => 3,
        EvidenceIndependence::Unknown => 4,
    };
    hasher.update(&[tag]);
}

fn write_canonical_independence_basis(hasher: &mut blake3::Hasher, value: &IndependenceBasis) {
    match value {
        IndependenceBasis::SharedSensor { sensor_id } => {
            hasher.update(&[0]);
            write_canonical_string(hasher, sensor_id);
        }
        IndependenceBasis::SharedPlatform { platform_id } => {
            hasher.update(&[1]);
            write_canonical_string(hasher, platform_id);
        }
        IndependenceBasis::SharedAncestor { observation_id } => {
            hasher.update(&[2]);
            write_canonical_string(hasher, observation_id);
        }
        IndependenceBasis::SharedProcessingActivity { activity_id } => {
            hasher.update(&[3]);
            write_canonical_string(hasher, activity_id);
        }
        IndependenceBasis::IdenticalAsset { hash_algorithm, content_hash } => {
            hasher.update(&[4]);
            write_canonical_string(hasher, hash_algorithm);
            write_canonical_string(hasher, content_hash);
        }
        IndependenceBasis::InsufficientProvenance { coverage } => {
            hasher.update(&[5]);
            write_canonical_provenance_coverage(hasher, coverage);
        }
        IndependenceBasis::NoSharedProvenance => hasher.update(&[6]),
    }
}

/// Declares whether the provenance fields exposed to the observation fabric are complete enough
/// for independence verification.
///
/// This is a provenance-availability declaration, not a cryptographic trust result.
/// A value of Partial or Redacted prevents the bounded independence verifier from
/// converting absence of a discovered shared basis into VerifiedIndependent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProvenanceCoverage {
    /// The represented provenance is intended to be complete for this observation.
    Complete,
    /// Some provenance relevant to independence may be unavailable.
    Partial,
    /// Provenance was intentionally withheld or redacted.
    Redacted,
}

impl Default for ProvenanceCoverage {
    fn default() -> Self {
        Self::Complete
    }
}

/// Provenance linking an observation to its producer and processing lineage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationProvenance {
    pub source: SensorIdentity,
    pub verification: ProvenanceVerification,
    /// Explicit availability declaration for provenance relevant to independence.
    #[serde(default)]
    pub coverage: ProvenanceCoverage,
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
        let mut parent_ids = HashSet::with_capacity(self.provenance.parent_observation_ids.len());
        if self.provenance.parent_observation_ids.iter().any(|parent| !parent_ids.insert(parent)) {
            return Err(ObservationValidationError::DuplicateParentObservation);
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

/// The concrete provenance basis for an independence assessment.
///
/// This keeps an independence classification auditable instead of collapsing
/// every non-independent result into an opaque SharedUpstream label.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Predicate evaluated by the bounded provenance-independence verifier.
///
/// The verifier is intentionally narrow: these predicates describe provenance
/// overlap, not substantive truth, sensor accuracy, or semantic equivalence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndependenceVerifierPredicate {
    SensorIdentity,
    PlatformIdentity,
    AncestralLineage,
    ProcessingActivityIdentity,
    AssetIdentity,
}

/// Explicit contract for the current bounded independence verifier.
///
/// The contract is deliberately separate from a confidence score. A result of
/// VerifiedIndependent means only that every predicate in this contract found
/// no shared provenance basis in a validated closed-world graph. The verifier
/// does not evaluate relation semantics, modality, observation time/location,
/// measurement quality, credential validity, or substantive truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndependenceVerifierContract {
    pub version: &'static str,
    pub predicates: &'static [IndependenceVerifierPredicate],
    pub requires_closed_world: bool,
}

impl IndependenceVerifierContract {
    pub const CURRENT: Self = Self {
        version: "observation-fabric-independence-v2",
        predicates: &[
            IndependenceVerifierPredicate::SensorIdentity,
            IndependenceVerifierPredicate::PlatformIdentity,
            IndependenceVerifierPredicate::AncestralLineage,
            IndependenceVerifierPredicate::ProcessingActivityIdentity,
            IndependenceVerifierPredicate::AssetIdentity,
        ],
        requires_closed_world: true,
    };
}

pub enum IndependenceBasis {
    SharedSensor { sensor_id: String },
    SharedPlatform { platform_id: String },
    SharedAncestor { observation_id: String },
    SharedProcessingActivity { activity_id: String },
    IdenticalAsset { hash_algorithm: String, content_hash: String },
    /// The graph contains an explicit declaration that relevant provenance may be unavailable.
    InsufficientProvenance { coverage: ProvenanceCoverage },
    NoSharedProvenance,
}

/// Result of a bounded provenance-independence assessment.
///
/// VerifiedIndependent remains scoped to the supplied closed-world graph:
/// it means the verifier found no shared provenance basis in that graph, not
/// that the observations are substantively true or independent in reality.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndependenceAssessment {
    pub source_observation_id: String,
    pub target_observation_id: String,
    pub classification: EvidenceIndependence,
    pub basis: IndependenceBasis,
    pub examined_observation_ids: Vec<String>,
    pub verifier_version: &'static str,
    /// BLAKE3 commitment to the independence-relevant provenance projection
    /// of the observations examined by this assessment, including explicit
    /// provenance-coverage declarations.
    ///
    /// This prevents the assessment commitment from remaining unchanged when
    /// provenance fields relevant to the verifier are silently mutated.
    pub examined_scope_fingerprint: String,
    /// BLAKE3 commitment to the assessed pair, scope, classification, basis,
    /// and verifier version. This is an assessment fingerprint, not a truth claim.
    pub assessment_fingerprint: String,
}

/// Outcome of checking a receipt's integrity, verifier version, and graph binding.
///
/// These outcomes are intentionally distinct so callers can audit why a receipt
/// was not accepted. A successful result remains a bounded provenance statement,
/// not a claim that either observation is true.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReceiptVerificationOutcome {
    /// Receipt integrity and its result against the supplied graph were verified.
    VerifiedAgainstGraph,
    /// The receipt's stored assessment fingerprint does not match its fields.
    InvalidReceiptIntegrity,
    /// The receipt was produced under a verifier contract this implementation does not support.
    UnsupportedVerifierVersion,
    /// The current graph does not reproduce the receipt's recorded assessment.
    GraphMismatch,
}

/// Deterministic, attestation-ready receipt for a bounded independence assessment.
///
/// This is intentionally not a credential and contains no issuer/trust semantics.
/// External systems may bind a signature or credential to this receipt without
/// changing what the core verifier means by VerifiedIndependent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndependenceVerificationReceipt {
    pub source_observation_id: String,
    pub target_observation_id: String,
    pub classification: EvidenceIndependence,
    pub basis: IndependenceBasis,
    pub examined_observation_ids: Vec<String>,
    pub verifier_version: &'static str,
    pub examined_scope_fingerprint: String,
    pub assessment_fingerprint: String,
}

impl IndependenceVerificationReceipt {
    pub const DOMAIN_SEPARATOR: &'static [u8] = b"symthaea:observation-independence-receipt:v2\\n";

    pub fn from_assessment(assessment: &IndependenceAssessment) -> Self {
        Self {
            source_observation_id: assessment.source_observation_id.clone(),
            target_observation_id: assessment.target_observation_id.clone(),
            classification: assessment.classification.clone(),
            basis: assessment.basis.clone(),
            examined_observation_ids: assessment.examined_observation_ids.clone(),
            verifier_version: assessment.verifier_version,
            examined_scope_fingerprint: assessment.examined_scope_fingerprint.clone(),
            assessment_fingerprint: assessment.assessment_fingerprint.clone(),
        }
    }

    pub fn verify_integrity(&self) -> bool {
        let assessment = IndependenceAssessment {
            source_observation_id: self.source_observation_id.clone(),
            target_observation_id: self.target_observation_id.clone(),
            classification: self.classification.clone(),
            basis: self.basis.clone(),
            examined_observation_ids: self.examined_observation_ids.clone(),
            verifier_version: self.verifier_version,
            examined_scope_fingerprint: self.examined_scope_fingerprint.clone(),
            assessment_fingerprint: self.assessment_fingerprint.clone(),
        };
        assessment.verify_fingerprint()
    }

    /// Verify this receipt against the current closed-world observation graph.
    ///
    /// This is stronger than verify_integrity: it re-runs the bounded independence
    /// assessment and confirms that the current provenance scope still produces
    /// the same result. It does not verify an external signature or establish truth.
    pub fn verify_against_graph_detailed(
        &self,
        graph: &ObservationGraph,
    ) -> Result<ReceiptVerificationOutcome, ObservationValidationError> {
        if self.verifier_version != IndependenceAssessment::VERIFIER_VERSION {
            return Ok(ReceiptVerificationOutcome::UnsupportedVerifierVersion);
        }
        if !self.verify_integrity() {
            return Ok(ReceiptVerificationOutcome::InvalidReceiptIntegrity);
        }
        let assessment = graph.assess_independence_detailed(
            &self.source_observation_id,
            &self.target_observation_id,
        )?;
        let matches = assessment.classification == self.classification
            && assessment.basis == self.basis
            && assessment.examined_observation_ids == self.examined_observation_ids
            && assessment.examined_scope_fingerprint == self.examined_scope_fingerprint
            && assessment.verifier_version == self.verifier_version
            && assessment.assessment_fingerprint == self.assessment_fingerprint;
        Ok(if matches {
            ReceiptVerificationOutcome::VerifiedAgainstGraph
        } else {
            ReceiptVerificationOutcome::GraphMismatch
        })
    }

    /// Compatibility convenience method; use the detailed form for audit logs.
    pub fn verify_against_graph(
        &self,
        graph: &ObservationGraph,
    ) -> Result<bool, ObservationValidationError> {
        Ok(self.verify_against_graph_detailed(graph)?
            == ReceiptVerificationOutcome::VerifiedAgainstGraph)
    }

    /// Produce canonical bytes an external attestation layer can sign or hash.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(Self::DOMAIN_SEPARATOR);
        write_canonical_string_bytes(&mut bytes, &self.source_observation_id);
        write_canonical_string_bytes(&mut bytes, &self.target_observation_id);
        write_canonical_independence_bytes(&mut bytes, &self.classification);
        write_canonical_independence_basis_bytes(&mut bytes, &self.basis);
        write_canonical_string_vec_bytes(&mut bytes, &self.examined_observation_ids);
        write_canonical_string_bytes(&mut bytes, self.verifier_version);
        write_canonical_string_bytes(&mut bytes, &self.examined_scope_fingerprint);
        write_canonical_string_bytes(&mut bytes, &self.assessment_fingerprint);
        bytes
    }

    pub fn fingerprint(&self) -> String {
        blake3::hash(&self.canonical_bytes()).to_hex().to_string()
    }
}


/// Standards-aware, crypto-agnostic envelope for externally attesting an
/// independence verification receipt.
///
/// The core owns the exact receipt commitment and the semantic boundary; an
/// external adapter owns issuer/key trust, cryptographic verification, and
/// credential semantics. The envelope therefore never turns a valid proof
/// into a claim about substantive truth.
///
/// The field names intentionally mirror common proof-envelope concepts
/// (purpose, verification method, cryptographic suite, creation/expiry,
/// domain, and challenge) without claiming conformance to any particular
/// external proof format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReceiptAttestationEnvelope {
    /// Fingerprint of the exact canonical receipt being attested.
    pub receipt_fingerprint: String,
    /// Verifier contract version that produced the receipt.
    pub verifier_version: String,
    /// Provenance scope fingerprint committed by the receipt.
    pub examined_scope_fingerprint: String,
    /// Opaque identity of the attester/issuer.
    pub attester_id: String,
    /// Purpose for which the proof was created; not a truth assertion.
    pub proof_purpose: String,
    /// Attestation creation time, expressed as Unix nanoseconds.
    pub created_at_unix_ns: i128,
    /// Optional expiry boundary, also Unix nanoseconds.
    pub expires_at_unix_ns: Option<i128>,
    /// Opaque reference to the external verification method/key.
    pub verification_method: Option<String>,
    /// Opaque external cryptographic-suite identifier.
    pub cryptosuite: Option<String>,
    /// Optional replay-binding security domain.
    pub domain: Option<String>,
    /// Optional verifier-supplied replay challenge.
    pub challenge: Option<String>,
    /// Opaque proof bytes owned and interpreted by the external adapter.
    pub proof: Option<Vec<u8>>,
}

/// Time-relative validity of a detached receipt attestation envelope.\n///\n/// This is deliberately trust-neutral: it evaluates only the envelope's\n/// declared creation/expiry timestamps. It does not validate cryptographic\n/// proof material, attester identity, revocation, or the underlying receipt.\n#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]\npub enum ReceiptAttestationTemporalStatus {\n    NotYetValid,\n    Valid,\n    Expired,\n}\n\nimpl ReceiptAttestationEnvelope {
    /// Domain separator for the deterministic attestation payload.
    pub const DOMAIN_SEPARATOR: &'static [u8] =
        b"symthaea:observation-receipt-attestation:v1\\n";

    /// Build an unsigned envelope from an exact receipt.
    ///
    /// The returned value is metadata/commitment material only. It is not an
    /// authenticated attestation until an external adapter supplies and
    /// verifies a proof.
    pub fn from_receipt(
        receipt: &IndependenceVerificationReceipt,
        attester_id: impl Into<String>,
        proof_purpose: impl Into<String>,
        created_at_unix_ns: i128,
    ) -> Self {
        Self {
            receipt_fingerprint: receipt.fingerprint(),
            verifier_version: receipt.verifier_version.to_string(),
            examined_scope_fingerprint: receipt.examined_scope_fingerprint.clone(),
            attester_id: attester_id.into(),
            proof_purpose: proof_purpose.into(),
            created_at_unix_ns,
            expires_at_unix_ns: None,
            verification_method: None,
            cryptosuite: None,
            domain: None,
            challenge: None,
            proof: None,
        }
    }

    /// Verify that the envelope still names the exact receipt it claims to attest.
    ///
    /// This is a commitment check only. It does not verify the external proof,
    /// attester identity, or any substantive observation claim.
    pub fn verify_against_receipt(
        &self,
        receipt: &IndependenceVerificationReceipt,
    ) -> bool {
        self.receipt_fingerprint == receipt.fingerprint()
            && self.verifier_version == receipt.verifier_version
            && self.examined_scope_fingerprint == receipt.examined_scope_fingerprint
    }

    /// Validate the envelope's structural commitments.
    ///
    /// This does not verify the external proof, resolve the attester, or
    /// establish the truth of the underlying observations.
    pub fn validate(&self) -> Result<(), ObservationValidationError> {
        if !is_hex_fingerprint(&self.receipt_fingerprint)
            || !is_hex_fingerprint(&self.examined_scope_fingerprint)
            || self.verifier_version.trim().is_empty()
            || self.attester_id.trim().is_empty()
            || self.proof_purpose.trim().is_empty()
            || self.verification_method.as_deref().is_some_and(|v| v.trim().is_empty())
            || self.cryptosuite.as_deref().is_some_and(|v| v.trim().is_empty())
            || self.domain.as_deref().is_some_and(|v| v.trim().is_empty())
            || self.challenge.as_deref().is_some_and(|v| v.trim().is_empty())
            || matches!(self.expires_at_unix_ns, Some(expiry) if expiry <= self.created_at_unix_ns)
        {
            return Err(ObservationValidationError::InvalidReceiptAttestationEnvelope);
        }
        Ok(())
    }

    /// Produce the exact payload an external proof adapter should protect.
    ///
    /// The proof field is deliberately excluded, matching the detached-proof
    /// boundary used by common data-integrity systems. The canonical encoding
    /// is Symthaea-native and is not a claim of JSON-LD/JCS conformance.
    pub fn canonical_payload_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(Self::DOMAIN_SEPARATOR);
        write_canonical_string_bytes(&mut bytes, &self.receipt_fingerprint);
        write_canonical_string_bytes(&mut bytes, &self.verifier_version);
        write_canonical_string_bytes(&mut bytes, &self.examined_scope_fingerprint);
        write_canonical_string_bytes(&mut bytes, &self.attester_id);
        write_canonical_string_bytes(&mut bytes, &self.proof_purpose);
        bytes.extend_from_slice(&self.created_at_unix_ns.to_be_bytes());
        write_canonical_i128_option_bytes(&mut bytes, self.expires_at_unix_ns);
        write_canonical_string_option_bytes(&mut bytes, self.verification_method.as_deref());
        write_canonical_string_option_bytes(&mut bytes, self.cryptosuite.as_deref());
        write_canonical_string_option_bytes(&mut bytes, self.domain.as_deref());
        write_canonical_string_option_bytes(&mut bytes, self.challenge.as_deref());
        bytes
    }

    /// BLAKE3 commitment to the detached attestation payload.
    pub fn payload_fingerprint(&self) -> String {
        blake3::hash(&self.canonical_payload_bytes()).to_hex().to_string()
    }
}

fn is_hex_fingerprint(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn write_canonical_i128_option_bytes(bytes: &mut Vec<u8>, value: Option<i128>) {
    match value {
        Some(value) => {
            bytes.push(1);
            bytes.extend_from_slice(&value.to_be_bytes());
        }
        None => bytes.push(0),
    }
}

fn write_canonical_string_option_bytes(bytes: &mut Vec<u8>, value: Option<&str>) {
    match value {
        Some(value) => {
            bytes.push(1);
            write_canonical_string_bytes(bytes, value);
        }
        None => bytes.push(0),
    }
}


impl IndependenceAssessment {
    const VERIFIER_VERSION: &'static str = IndependenceVerifierContract::CURRENT.version;

    fn compute_fingerprint(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea:observation-independence-assessment:v2\n");
        write_canonical_string(&mut hasher, &self.source_observation_id);
        write_canonical_string(&mut hasher, &self.target_observation_id);
        write_canonical_string_vec(&mut hasher, &self.examined_observation_ids);
        write_canonical_string(&mut hasher, &self.examined_scope_fingerprint);
        write_canonical_string(&mut hasher, self.verifier_version);
        write_canonical_independence(&mut hasher, &self.classification);
        write_canonical_independence_basis(&mut hasher, &self.basis);
        hasher.finalize().to_hex().to_string()
    }


    /// Verify that the stored fingerprint still commits to this assessment.
    ///
    /// This verifies the integrity of the assessment record itself. It does
    /// not re-run graph analysis and does not assert that the assessment is true.
    pub fn verify_fingerprint(&self) -> bool {
        self.assessment_fingerprint == self.compute_fingerprint()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceIndependence {
    /// The producer declares no known shared upstream source.
    ///
    /// This is an assertion, not a verification result. Consumers must not
    /// treat it as independently established evidence without an explicit
    /// verification step.
    Independent,
    /// A verifier established that no shared upstream was found in complete
    /// provenance evidence available to it.
    ///
    /// This does not assert substantive truth or metaphysical independence;
    /// it records a bounded provenance-verification result.
    VerifiedIndependent,
    /// The observations share an upstream source, platform, or processing chain.
    SharedUpstream,
    /// The source observation is computationally derived from the target lineage.
    Derived,
    /// Independence has not been established.
    Unknown,
}

impl EvidenceIndependence {
    /// Returns true when independence is asserted or explicitly verified.
    pub const fn is_independent(&self) -> bool {
        matches!(self, Self::Independent | Self::VerifiedIndependent)
    }

    /// Returns true only when a verifier explicitly established independence
    /// from the available provenance evidence.
    pub const fn is_verified_independent(&self) -> bool {
        matches!(self, Self::VerifiedIndependent)
    }

    /// Returns true when the edge is known not to be independent.
    pub const fn is_known_non_independent(&self) -> bool {
        matches!(self, Self::SharedUpstream | Self::Derived)
    }
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
    /// Assess provenance independence and retain the concrete basis used.
    pub fn assess_independence_detailed(
        &self,
        source_observation_id: &str,
        target_observation_id: &str,
    ) -> Result<IndependenceAssessment, ObservationValidationError> {
        self.validate()?;
        if source_observation_id == target_observation_id {
            return Err(ObservationValidationError::SelfRelation);
        }

        let by_id = self
            .observations
            .iter()
            .map(|observation| (observation.id.as_str(), observation))
            .collect::<HashMap<_, _>>();

        let source = by_id
            .get(source_observation_id)
            .ok_or_else(|| ObservationValidationError::MissingRelationEndpoint(
                source_observation_id.to_string(),
            ))?;
        let target = by_id
            .get(target_observation_id)
            .ok_or_else(|| ObservationValidationError::MissingRelationEndpoint(
                target_observation_id.to_string(),
            ))?;

        let mut examined_observation_ids = self
            .observations
            .iter()
            .map(|observation| observation.id.clone())
            .collect::<Vec<_>>();
        examined_observation_ids.sort();
        let examined_scope_fingerprint = self
            .compute_independence_scope_fingerprint(&by_id, &examined_observation_ids)?;

        let assessment = |classification, basis| {

            let mut hasher = blake3::Hasher::new();
            hasher.update(b"symthaea:observation-independence-assessment:v2\n");
            write_canonical_string(&mut hasher, source_observation_id);
            write_canonical_string(&mut hasher, target_observation_id);
            write_canonical_string_vec(&mut hasher, &examined_observation_ids);
            write_canonical_string(&mut hasher, &examined_scope_fingerprint);
            write_canonical_string(&mut hasher, IndependenceAssessment::VERIFIER_VERSION);
            write_canonical_independence(&mut hasher, &classification);
            write_canonical_independence_basis(&mut hasher, &basis);
            IndependenceAssessment {
                source_observation_id: source_observation_id.to_string(),
                target_observation_id: target_observation_id.to_string(),
                classification,
                basis,
                examined_observation_ids: examined_observation_ids.clone(),
                verifier_version: IndependenceAssessment::VERIFIER_VERSION,
                examined_scope_fingerprint: examined_scope_fingerprint.clone(),
                assessment_fingerprint: hasher.finalize().to_hex().to_string(),
            }
        };

        if !matches!(source.provenance.coverage, ProvenanceCoverage::Complete)
            || !matches!(target.provenance.coverage, ProvenanceCoverage::Complete)
        {
            let coverage = if !matches!(source.provenance.coverage, ProvenanceCoverage::Complete) {
                source.provenance.coverage
            } else {
                target.provenance.coverage
            };
            return Ok(assessment(
                EvidenceIndependence::Unknown,
                IndependenceBasis::InsufficientProvenance { coverage },
            ));
        }

        if source.provenance.source.sensor_id == target.provenance.source.sensor_id {
            return Ok(assessment(
                EvidenceIndependence::SharedUpstream,
                IndependenceBasis::SharedSensor {
                    sensor_id: source.provenance.source.sensor_id.clone(),
                },
            ));
        }

        if let (Some(source_platform), Some(target_platform)) = (
            source.provenance.source.platform_id.as_ref(),
            target.provenance.source.platform_id.as_ref(),
        ) {
            if source_platform == target_platform {
                return Ok(assessment(
                    EvidenceIndependence::SharedUpstream,
                    IndependenceBasis::SharedPlatform {
                        platform_id: source_platform.clone(),
                    },
                ));
            }
        }

        let source_ancestors = Self::ancestor_ids(source_observation_id, &by_id)?;
        let target_ancestors = Self::ancestor_ids(target_observation_id, &by_id)?;
        if let Some(shared_ancestor) = source_ancestors
            .intersection(&target_ancestors)
            .min()
        {
            return Ok(assessment(
                EvidenceIndependence::SharedUpstream,
                IndependenceBasis::SharedAncestor {
                    observation_id: shared_ancestor.clone(),
                },
            ));
        }

        let source_activities = source_ancestors
            .iter()
            .filter_map(|id| by_id.get(id.as_str()))
            .chain(std::iter::once(source))
            .filter_map(|observation| {
                observation
                    .provenance
                    .processing_activity
                    .as_ref()
                    .map(|activity| activity.activity_id.as_str())
            })
            .collect::<HashSet<_>>();
        let target_activities = target_ancestors
            .iter()
            .filter_map(|id| by_id.get(id.as_str()))
            .chain(std::iter::once(target))
            .filter_map(|observation| {
                observation
                    .provenance
                    .processing_activity
                    .as_ref()
                    .map(|activity| activity.activity_id.as_str())
            })
            .collect::<HashSet<_>>();
        if let Some(shared_activity) = source_activities
            .intersection(&target_activities)
            .min()
        {
            return Ok(assessment(
                EvidenceIndependence::SharedUpstream,
                IndependenceBasis::SharedProcessingActivity {
                    activity_id: (*shared_activity).to_string(),
                },
            ));
        }

        if let Some((source_asset, target_asset)) =
            source.asset.as_ref().zip(target.asset.as_ref())
        {
            if source_asset.hash_algorithm == target_asset.hash_algorithm
                && source_asset.content_hash == target_asset.content_hash
            {
                return Ok(assessment(
                    EvidenceIndependence::SharedUpstream,
                    IndependenceBasis::IdenticalAsset {
                        hash_algorithm: source_asset.hash_algorithm.clone(),
                        content_hash: source_asset.content_hash.clone(),
                    },
                ));
            }
        }

        Ok(assessment(
            EvidenceIndependence::VerifiedIndependent,
            IndependenceBasis::NoSharedProvenance,
        ))
    }

    /// Recompute the provenance-scope commitment for an existing assessment.
    ///
    /// This is stronger than the assessment fingerprint check because it
    /// checks the current graph rather than only stored assessment fields.
    pub fn verify_independence_scope_fingerprint(
        &self,
        assessment: &IndependenceAssessment,
    ) -> Result<bool, ObservationValidationError> {
        self.validate()?;
        let mut expected_ids = self
            .observations
            .iter()
            .map(|observation| observation.id.clone())
            .collect::<Vec<_>>();
        expected_ids.sort();
        if assessment.examined_observation_ids != expected_ids {
            return Ok(false);
        }
        let by_id = self
            .observations
            .iter()
            .map(|observation| (observation.id.as_str(), observation))
            .collect::<HashMap<_, _>>();
        let expected = self.compute_independence_scope_fingerprint(
            &by_id,
            &assessment.examined_observation_ids,
        )?;
        Ok(expected == assessment.examined_scope_fingerprint)
    }

    fn compute_independence_scope_fingerprint(
        &self,
        by_id: &HashMap<&str, &Observation>,
        examined_observation_ids: &[String],
    ) -> Result<String, ObservationValidationError> {
        let mut scope_hasher = blake3::Hasher::new();
        scope_hasher.update(b"symthaea:observation-independence-scope:v1\\n");
        write_canonical_string_vec(&mut scope_hasher, examined_observation_ids);
        for observation_id in examined_observation_ids {
            let observation = by_id
                .get(observation_id.as_str())
                .ok_or_else(|| ObservationValidationError::MissingRelationEndpoint(observation_id.clone()))?;
            write_canonical_string(&mut scope_hasher, &observation.id);
            write_canonical_string(&mut scope_hasher, &observation.provenance.source.sensor_id);
            write_canonical_provenance_coverage(&mut scope_hasher, &observation.provenance.coverage);
            if let Some(platform_id) = &observation.provenance.source.platform_id {
                scope_hasher.update(&[1]);
                write_canonical_string(&mut scope_hasher, platform_id);
            } else {
                scope_hasher.update(&[0]);
            }
            let mut parent_ids = observation.provenance.parent_observation_ids.clone();
            parent_ids.sort();
            write_canonical_string_vec(&mut scope_hasher, &parent_ids);
            if let Some(activity) = &observation.provenance.processing_activity {
                scope_hasher.update(&[1]);
                write_canonical_string(&mut scope_hasher, &activity.activity_id);
                write_canonical_string(&mut scope_hasher, activity.execution_fingerprint.as_deref().unwrap_or(""));
            } else {
                scope_hasher.update(&[0]);
            }
            if let Some(asset) = &observation.asset {
                scope_hasher.update(&[1]);
                write_canonical_string(&mut scope_hasher, &asset.hash_algorithm);
                write_canonical_string(&mut scope_hasher, &asset.content_hash);
            } else {
                scope_hasher.update(&[0]);
            }
        }
        Ok(scope_hasher.finalize().to_hex().to_string())
    }

    /// Assess whether two observations have independent provenance within this
    /// closed-world graph. The returned classification is retained for callers
    /// that only need the legacy enum; use assess_independence_detailed when
    /// the audit basis should be preserved.
    pub fn assess_independence(
        &self,
        source_observation_id: &str,
        target_observation_id: &str,
    ) -> Result<EvidenceIndependence, ObservationValidationError> {
        Ok(self
            .assess_independence_detailed(source_observation_id, target_observation_id)?
            .classification)
    }

    fn ancestor_ids(
        observation_id: &str,
        by_id: &HashMap<&str, &Observation>,
    ) -> Result<HashSet<String>, ObservationValidationError> {
        let mut ancestors = HashSet::new();
        let mut pending = vec![observation_id.to_string()];
        while let Some(current) = pending.pop() {
            let observation = by_id
                .get(current.as_str())
                .ok_or_else(|| ObservationValidationError::MissingParentObservation(current.clone()))?;
            for parent_id in &observation.provenance.parent_observation_ids {
                if ancestors.insert(parent_id.clone()) {
                    pending.push(parent_id.clone());
                }
            }
        }
        Ok(ancestors)
    }

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
    #[error("parent observation ids must be unique within an observation")]
    DuplicateParentObservation,
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
    #[error("invalid receipt attestation envelope")]
    InvalidReceiptAttestationEnvelope,
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
                coverage: ProvenanceCoverage::Complete,
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
    fn receipt_attestation_envelope_is_detached_and_validatable() {
        let mut second = fixture();
        second.id = "obs-002".into();
        second.provenance.source.sensor_id = "camera-2".into();
        let graph = ObservationGraph {
            observations: vec![fixture(), second],
            relations: vec![],
        };
        let assessment = graph
            .assess_independence_detailed("obs-001", "obs-002")
            .expect("assessment");
        let receipt = IndependenceVerificationReceipt::from_assessment(&assessment);
        let mut envelope = ReceiptAttestationEnvelope::from_receipt(
            &receipt,
            "attester-1",
            "https://example.org/purpose/observation-independence",
            1_700_000_000_000_000_000,
        );
        assert_eq!(envelope.validate(), Ok(()));
        assert_eq!(envelope.receipt_fingerprint, receipt.fingerprint());
        assert!(envelope.verify_against_receipt(&receipt));
        envelope.receipt_fingerprint = "00".repeat(32);
        assert!(!envelope.verify_against_receipt(&receipt));
        envelope.receipt_fingerprint = receipt.fingerprint();
        assert_eq!(envelope.examined_scope_fingerprint, receipt.examined_scope_fingerprint);
        assert_eq!(envelope.proof, None);

        let payload = envelope.canonical_payload_bytes();
        assert!(payload.starts_with(ReceiptAttestationEnvelope::DOMAIN_SEPARATOR));
        assert_eq!(envelope.payload_fingerprint().len(), 64);

        envelope.expires_at_unix_ns = Some(1_699_999_999_000_000_000);
        assert_eq!(
            envelope.validate(),
            Err(ObservationValidationError::InvalidReceiptAttestationEnvelope)
        );

        envelope.expires_at_unix_ns = None;
        let baseline = envelope.payload_fingerprint();
        envelope.challenge = Some("challenge-1".into());
        assert_ne!(baseline, envelope.payload_fingerprint());
        envelope.challenge = None;
        envelope.proof_purpose = "https://example.org/purpose/other".into();
        assert_ne!(baseline, envelope.payload_fingerprint());
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
            independence: EvidenceIndependence::Independent,        };
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
    fn duplicate_parent_reference_fails_closed() {
        let mut observation = fixture();
        observation.provenance.parent_observation_ids = vec!["parent".into(), "parent".into()];
        assert_eq!(
            observation.validate(),
            Err(ObservationValidationError::DuplicateParentObservation)
        );
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
        );    }

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
    fn declared_independence_is_not_verified() {
        assert!(EvidenceIndependence::Independent.is_independent());
        assert!(!EvidenceIndependence::Independent.is_verified_independent());
    }

    #[test]
    fn verified_independence_is_explicit() {
        assert!(EvidenceIndependence::VerifiedIndependent.is_independent());
        assert!(EvidenceIndependence::VerifiedIndependent.is_verified_independent());
    }

    #[test]
    fn derived_relation_rejects_verified_independence() {
        let relation = ObservationRelation {
            source_observation_id: "derived".into(),
            target_observation_id: "source".into(),
            kind: ObservationRelationKind::DerivedFrom,
            independence: EvidenceIndependence::VerifiedIndependent,
        };
        assert_eq!(
            relation.validate(),
            Err(ObservationValidationError::DerivedRelationIndependenceMismatch)
        );
    }

    #[test]
    fn graph_assesses_verified_independence_without_shared_provenance() {
        let mut second = fixture();
        second.id = "obs-002".into();
        second.provenance.source.sensor_id = "camera-2".into();
        assert_eq!(
            ObservationGraph {
                observations: vec![fixture(), second],
                relations: vec![],
            }
            .assess_independence("obs-001", "obs-002"),
            Ok(EvidenceIndependence::VerifiedIndependent)
        );
    }

    #[test]
    fn graph_detailed_independence_is_stable_across_observation_order() {
        let mut second = fixture();
        second.id = "obs-002".into();
        second.provenance.source.sensor_id = "camera-2".into();
        let graph_a = ObservationGraph {
            observations: vec![fixture(), second.clone()],
            relations: vec![],
        };
        let graph_b = ObservationGraph {
            observations: vec![second, fixture()],
            relations: vec![],
        };
        let a = graph_a
            .assess_independence_detailed("obs-001", "obs-002")
            .expect("assessment");
        let b = graph_b
            .assess_independence_detailed("obs-001", "obs-002")
            .expect("assessment");
        assert_eq!(a.examined_observation_ids, b.examined_observation_ids);
        assert_eq!(a.assessment_fingerprint, b.assessment_fingerprint);
    }

    #[test]
    fn graph_detailed_independence_records_shared_sensor_basis() {
        let mut second = fixture();
        second.id = "obs-002".into();
        let assessment = ObservationGraph {
            observations: vec![fixture(), second],
            relations: vec![],
        }
        .assess_independence_detailed("obs-001", "obs-002")
        .expect("assessment");
        assert_eq!(assessment.classification, EvidenceIndependence::SharedUpstream);
        assert_eq!(
            assessment.basis,
            IndependenceBasis::SharedSensor {
                sensor_id: "camera-1".into()
            }
        );
        assert_eq!(assessment.assessment_fingerprint.len(), 64);
    }

    #[test]
    fn graph_detailed_independence_records_audit_basis() {
        let mut second = fixture();
        second.id = "obs-002".into();
        second.provenance.source.sensor_id = "camera-2".into();
        let assessment = ObservationGraph {
            observations: vec![fixture(), second],
            relations: vec![],
        }
        .assess_independence_detailed("obs-001", "obs-002")
        .expect("assessment");
        assert_eq!(assessment.source_observation_id, "obs-001");
        assert_eq!(assessment.target_observation_id, "obs-002");
        assert_eq!(assessment.classification, EvidenceIndependence::VerifiedIndependent);
        assert_eq!(assessment.basis, IndependenceBasis::NoSharedProvenance);
        assert_eq!(assessment.examined_observation_ids, vec!["obs-001", "obs-002"]);
        assert_eq!(assessment.verifier_version, "observation-fabric-independence-v2");
        assert_eq!(assessment.examined_scope_fingerprint.len(), 64);
        assert_eq!(assessment.assessment_fingerprint.len(), 64);
        assert!(assessment.assessment_fingerprint.bytes().all(|b| b.is_ascii_hexdigit()));
        assert!(assessment.verify_fingerprint());
        assert_eq!(IndependenceVerifierContract::CURRENT.version, assessment.verifier_version);
        assert!(IndependenceVerifierContract::CURRENT.requires_closed_world);
        assert_eq!(
            IndependenceVerifierContract::CURRENT.predicates,
            &[
                IndependenceVerifierPredicate::SensorIdentity,
                IndependenceVerifierPredicate::PlatformIdentity,
                IndependenceVerifierPredicate::AncestralLineage,
                IndependenceVerifierPredicate::ProcessingActivityIdentity,
                IndependenceVerifierPredicate::AssetIdentity,
            ]
        );
    }

    #[test]
    fn independence_scope_verification_detects_graph_mutation() {
        let mut second = fixture();
        second.id = "obs-002".into();
        second.provenance.source.sensor_id = "camera-2".into();
        let mut graph = ObservationGraph {
            observations: vec![fixture(), second],
            relations: vec![],
        };
        let assessment = graph
            .assess_independence_detailed("obs-001", "obs-002")
            .expect("assessment");
        assert!(graph.verify_independence_scope_fingerprint(&assessment).unwrap());
        graph.observations[0].provenance.source.platform_id = Some("platform-9".into());
        assert!(!graph.verify_independence_scope_fingerprint(&assessment).unwrap());
    }

    #[test]
    fn independence_scope_excludes_non_verifier_fields() {
        let mut second = fixture();
        second.id = "obs-002".into();
        second.provenance.source.sensor_id = "camera-2".into();
        let graph = ObservationGraph {
            observations: vec![fixture(), second],
            relations: vec![],
        };
        let baseline = graph
            .assess_independence_detailed("obs-001", "obs-002")
            .expect("assessment");
        let mut changed = graph.clone();
        changed.observations[0].time.observed_at_unix_ns += 1;
        changed.observations[0].quality.confidence = 0.1;
        changed.observations[0].location.as_mut().unwrap().latitude_deg = 11.0;
        let updated = changed
            .assess_independence_detailed("obs-001", "obs-002")
            .expect("assessment");
        assert_eq!(baseline.classification, updated.classification);
        assert_eq!(baseline.examined_scope_fingerprint, updated.examined_scope_fingerprint);
    }

    #[test]
    fn detailed_independence_fingerprint_changes_when_scope_provenance_changes() {
        let mut second = fixture();
        second.id = "obs-002".into();
        second.provenance.source.sensor_id = "camera-2".into();
        let mut assessment = ObservationGraph {
            observations: vec![fixture(), second],
            relations: vec![],
        }
        .assess_independence_detailed("obs-001", "obs-002")
        .expect("assessment");
        let original = assessment.assessment_fingerprint.clone();
        assessment.examined_scope_fingerprint = "0".repeat(64);
        assert_ne!(original, assessment.compute_fingerprint());
        assert!(!assessment.verify_fingerprint());
    }

    #[test]
    fn detailed_independence_fingerprint_changes_when_basis_changes() {
        let mut second = fixture();
        second.id = "obs-002".into();
        second.provenance.source.sensor_id = "camera-2".into();
        let mut assessment = ObservationGraph {
            observations: vec![fixture(), second],
            relations: vec![],
        }
        .assess_independence_detailed("obs-001", "obs-002")
        .expect("assessment");
        let original = assessment.assessment_fingerprint.clone();

        assessment.basis = IndependenceBasis::SharedPlatform {
            platform_id: "platform-1".into(),
        };
        assert_ne!(assessment.assessment_fingerprint, assessment.compute_fingerprint());
        assert_ne!(original, assessment.compute_fingerprint());
        assert!(!assessment.verify_fingerprint());
    }

    #[test]
    fn partial_provenance_cannot_be_verified_independent() {
        let mut second = fixture();
        second.id = "obs-002".into();
        second.provenance.source.sensor_id = "camera-2".into();
        second.provenance.coverage = ProvenanceCoverage::Partial;

        let assessment = ObservationGraph {
            observations: vec![fixture(), second],
            relations: vec![],
        }
        .assess_independence_detailed("obs-001", "obs-002")
        .expect("assessment");

        assert_eq!(assessment.classification, EvidenceIndependence::Unknown);
        assert_eq!(
            assessment.basis,
            IndependenceBasis::InsufficientProvenance {
                coverage: ProvenanceCoverage::Partial,
            }
        );
        assert!(assessment.verify_fingerprint());
    }

    #[test]
    fn redacted_provenance_cannot_be_verified_independent() {
        let mut first = fixture();
        first.provenance.coverage = ProvenanceCoverage::Redacted;
        let mut second = fixture();
        second.id = "obs-002".into();
        second.provenance.source.sensor_id = "camera-2".into();

        let assessment = ObservationGraph {
            observations: vec![first, second],
            relations: vec![],
        }
        .assess_independence_detailed("obs-001", "obs-002")
        .expect("assessment");

        assert_eq!(assessment.classification, EvidenceIndependence::Unknown);
        assert_eq!(
            assessment.basis,
            IndependenceBasis::InsufficientProvenance {
                coverage: ProvenanceCoverage::Redacted,
            }
        );
    }

    #[test]
    fn provenance_coverage_is_part_of_independence_scope() {
        let mut second = fixture();
        second.id = "obs-002".into();
        second.provenance.source.sensor_id = "camera-2".into();

        let graph = ObservationGraph {
            observations: vec![fixture(), second],
            relations: vec![],
        };
        let baseline = graph
            .assess_independence_detailed("obs-001", "obs-002")
            .expect("assessment");

        let mut changed = graph.clone();
        changed.observations[0].provenance.coverage = ProvenanceCoverage::Partial;
        assert!(!changed
            .verify_independence_scope_fingerprint(&baseline)
            .expect("scope verification"));
    }

    #[test]
    fn independence_receipt_is_attestation_ready() {
        let mut second = fixture();
        second.id = "obs-002".into();
        second.provenance.source.sensor_id = "camera-2".into();
        let graph = ObservationGraph { observations: vec![fixture(), second], relations: vec![] };
        let assessment = graph.assess_independence_detailed("obs-001", "obs-002").expect("assessment");
        let receipt = IndependenceVerificationReceipt::from_assessment(&assessment);
        assert!(receipt.verify_integrity());
        assert_eq!(receipt.verify_against_graph(&graph), Ok(true));
        assert_eq!(receipt.verify_against_graph_detailed(&graph), Ok(ReceiptVerificationOutcome::VerifiedAgainstGraph));
        let encoded = serde_json::to_string(&ReceiptVerificationOutcome::VerifiedAgainstGraph).expect("serialize outcome");
        assert_eq!(serde_json::from_str::<ReceiptVerificationOutcome>(&encoded).expect("deserialize outcome"), ReceiptVerificationOutcome::VerifiedAgainstGraph);
        assert!(!receipt.canonical_bytes().is_empty());
        assert!(receipt.canonical_bytes().starts_with(IndependenceVerificationReceipt::DOMAIN_SEPARATOR));
        assert_eq!(receipt.fingerprint().len(), 64);
        assert_eq!(receipt.fingerprint(), IndependenceVerificationReceipt::from_assessment(&assessment).fingerprint());
    }

    #[test]
    fn receipt_graph_verification_detects_provenance_mutation() {
        let mut second = fixture();
        second.id = "obs-002".into();
        second.provenance.source.sensor_id = "camera-2".into();
        let graph = ObservationGraph { observations: vec![fixture(), second], relations: vec![] };
        let assessment = graph.assess_independence_detailed("obs-001", "obs-002").expect("assessment");
        let receipt = IndependenceVerificationReceipt::from_assessment(&assessment);

        let mut changed = graph.clone();
        changed.observations[0].provenance.source.platform_id = Some("platform-9".into());
        assert_eq!(receipt.verify_against_graph(&changed), Ok(false));
        assert_eq!(receipt.verify_against_graph_detailed(&changed), Ok(ReceiptVerificationOutcome::GraphMismatch));
    }

    #[test]
    fn receipt_graph_verification_rejects_unknown_verifier_version() {
        let mut second = fixture();
        second.id = "obs-002".into();
        second.provenance.source.sensor_id = "camera-2".into();
        let graph = ObservationGraph { observations: vec![fixture(), second], relations: vec![] };
        let assessment = graph.assess_independence_detailed("obs-001", "obs-002").expect("assessment");
        let mut receipt = IndependenceVerificationReceipt::from_assessment(&assessment);
        receipt.verifier_version = "observation-fabric-independence-v0";
        assert!(!receipt.verify_integrity());
        assert_eq!(receipt.verify_against_graph(&graph), Ok(false));
        assert_eq!(receipt.verify_against_graph_detailed(&graph), Ok(ReceiptVerificationOutcome::UnsupportedVerifierVersion));
    }
    #[test]
    fn detailed_independence_fingerprint_is_not_debug_format_dependent() {
        let mut second = fixture();
        second.id = "obs-002".into();
        second.provenance.source.sensor_id = "camera-2".into();
        let assessment = ObservationGraph {
            observations: vec![fixture(), second],
            relations: vec![],
        }
        .assess_independence_detailed("obs-001", "obs-002")
        .expect("assessment");

        let mut expected = blake3::Hasher::new();
        expected.update(b"symthaea:observation-independence-assessment:v2\n");
        write_canonical_string(&mut expected, "obs-001");
        write_canonical_string(&mut expected, "obs-002");
        write_canonical_string_vec(&mut expected, &["obs-001".into(), "obs-002".into()]);
        write_canonical_string(&mut expected, &assessment.examined_scope_fingerprint);
        write_canonical_string(&mut expected, "observation-fabric-independence-v2");
        write_canonical_independence(&mut expected, &EvidenceIndependence::VerifiedIndependent);
        write_canonical_independence_basis(&mut expected, &IndependenceBasis::NoSharedProvenance);
        assert_eq!(assessment.assessment_fingerprint, expected.finalize().to_hex().to_string());
        assert!(assessment.verify_fingerprint());
    }

    #[test]
    fn detailed_independence_rejects_invalid_closed_graph() {
        let mut observation = fixture();
        observation.provenance.parent_observation_ids = vec!["missing".into()];
        let graph = ObservationGraph {
            observations: vec![observation],
            relations: vec![],
        };
        assert_eq!(
            graph.assess_independence_detailed("obs-001", "missing"),
            Err(ObservationValidationError::MissingParentObservation("missing".into()))
        );
    }


    #[test]
    fn graph_assesses_shared_ancestor_as_non_independent() {
        let mut parent = fixture();
        parent.id = "parent".into();

        let mut first = fixture();
        first.id = "obs-001".into();
        first.provenance.parent_observation_ids = vec!["parent".into()];

        let mut second = fixture();
        second.id = "obs-002".into();
        second.provenance.source.sensor_id = "camera-2".into();
        second.provenance.parent_observation_ids = vec!["parent".into()];

        assert_eq!(
            ObservationGraph {
                observations: vec![parent, first, second],
                relations: vec![],
            }
            .assess_independence("obs-001", "obs-002"),
            Ok(EvidenceIndependence::SharedUpstream)
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
