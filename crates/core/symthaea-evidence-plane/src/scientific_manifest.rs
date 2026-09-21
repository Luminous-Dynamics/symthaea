// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cryptographic export manifest for evidence-bearing Symthaea research runs.
//!
//! This is an interchange boundary, not a scientific authority boundary.
//! Building a manifest does not turn a Symthaea inference into evidence and
//! does not append anything to Mycelix. It only packages an integrity-qualified
//! run into deterministic bytes with BLAKE3 identities for later governed
//! import/admission.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{check_integrity, EvidenceCounters, Expectation, RunEvidence};

pub const SCIENTIFIC_RUN_MANIFEST_PROTOCOL: &str = "symthaea-scientific-run-manifest";
pub const SCIENTIFIC_RUN_MANIFEST_VERSION: u16 = 1;
const MANIFEST_DOMAIN: &[u8] = b"SYMTHAEA-SCIENTIFIC-RUN-MANIFEST\0";
const MAX_TEXT_BYTES: usize = 4_096;
const MAX_ARTIFACTS: usize = 4_096;

/// Stable BLAKE3 content identity used at the Symthaea -> Mycelix boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ScientificContentHash(pub [u8; 32]);

impl ScientificContentHash {
    pub fn digest(bytes: &[u8]) -> Self {
        Self(*blake3::hash(bytes).as_bytes())
    }

    pub fn to_hex(self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

impl fmt::Display for ScientificContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

/// Identity of the component that produced a run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunProducerIdentity {
    pub component: String,
    pub version: String,
    /// Exact source revision when available (e.g. git commit SHA).
    pub revision: Option<String>,
}

impl RunProducerIdentity {
    pub fn new(
        component: impl Into<String>,
        version: impl Into<String>,
        revision: Option<String>,
    ) -> Result<Self, ScientificManifestError> {
        let identity = Self {
            component: component.into(),
            version: version.into(),
            revision,
        };
        identity.validate()?;
        Ok(identity)
    }

    fn validate(&self) -> Result<(), ScientificManifestError> {
        validate_text(&self.component, "producer component")?;
        validate_text(&self.version, "producer version")?;
        if let Some(revision) = &self.revision {
            validate_text(revision, "producer revision")?;
        }
        Ok(())
    }
}

/// One content-addressed artifact emitted or consumed by an experimental run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScientificRunArtifact {
    /// Producer-local stable identifier. Mycelix assigns/uses its own canonical
    /// artifact identity at governed admission time.
    pub artifact_id: String,
    pub content_hash: ScientificContentHash,
    pub media_type: String,
    /// Non-authoritative hint for a future typed Mycelix evidence-artifact role.
    pub role_hint: Option<String>,
}

impl ScientificRunArtifact {
    pub fn from_bytes(
        artifact_id: impl Into<String>,
        media_type: impl Into<String>,
        role_hint: Option<String>,
        bytes: &[u8],
    ) -> Result<Self, ScientificManifestError> {
        let artifact = Self {
            artifact_id: artifact_id.into(),
            content_hash: ScientificContentHash::digest(bytes),
            media_type: media_type.into(),
            role_hint,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    fn validate(&self) -> Result<(), ScientificManifestError> {
        validate_text(&self.artifact_id, "artifact id")?;
        validate_text(&self.media_type, "artifact media type")?;
        if let Some(role_hint) = &self.role_hint {
            validate_text(role_hint, "artifact role hint")?;
        }
        Ok(())
    }
}

/// Exportable record of one integrity-qualified Symthaea research run.
///
/// `config_digest` hashes caller-supplied exact configuration bytes and is
/// intentionally independent from [`crate::config_hash`], whose
/// `DefaultHasher` fingerprint is explicitly non-cryptographic and unstable
/// across Rust versions.
///
/// Fields are private so callers cannot mutate a qualified manifest into an
/// invalid state. Deserialization routes through [`ScientificRunManifest::validate`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ScientificRunManifestWire")]
pub struct ScientificRunManifest {
    protocol: String,
    protocol_version: u16,
    run_id: String,
    producer: RunProducerIdentity,
    integrity_satisfied: bool,
    config_digest: ScientificContentHash,
    /// Hash of an exact environment/reproducibility manifest when available.
    environment_digest: Option<ScientificContentHash>,
    /// Hash of the exact seed-plan representation when available.
    seed_plan_digest: Option<ScientificContentHash>,
    /// Declared mechanism expectations copied from the qualified run.
    declared: BTreeMap<String, Expectation>,
    /// Measured evidence counters copied into deterministic key order.
    measured: BTreeMap<String, f64>,
    /// Sorted by `artifact_id` during construction.
    artifacts: Vec<ScientificRunArtifact>,
}

#[derive(Debug, Clone, Deserialize)]
struct ScientificRunManifestWire {
    protocol: String,
    protocol_version: u16,
    run_id: String,
    producer: RunProducerIdentity,
    integrity_satisfied: bool,
    config_digest: ScientificContentHash,
    environment_digest: Option<ScientificContentHash>,
    seed_plan_digest: Option<ScientificContentHash>,
    declared: BTreeMap<String, Expectation>,
    measured: BTreeMap<String, f64>,
    artifacts: Vec<ScientificRunArtifact>,
}

impl TryFrom<ScientificRunManifestWire> for ScientificRunManifest {
    type Error = ScientificManifestError;

    fn try_from(value: ScientificRunManifestWire) -> Result<Self, Self::Error> {
        let manifest = Self {
            protocol: value.protocol,
            protocol_version: value.protocol_version,
            run_id: value.run_id,
            producer: value.producer,
            integrity_satisfied: value.integrity_satisfied,
            config_digest: value.config_digest,
            environment_digest: value.environment_digest,
            seed_plan_digest: value.seed_plan_digest,
            declared: value.declared,
            measured: value.measured,
            artifacts: value.artifacts,
        };
        manifest.validate()?;
        Ok(manifest)
    }
}

impl ScientificRunManifest {
    #[allow(clippy::too_many_arguments)]
    pub fn from_run_evidence(
        run: &RunEvidence,
        producer: RunProducerIdentity,
        exact_config_bytes: &[u8],
        exact_environment_bytes: Option<&[u8]>,
        exact_seed_plan_bytes: Option<&[u8]>,
        mut artifacts: Vec<ScientificRunArtifact>,
    ) -> Result<Self, ScientificManifestError> {
        // Never trust the cached/serialized `satisfied` or `violations` fields
        // as an authority boundary. Recompute integrity from the actual
        // declared expectations and measured counters at export time.
        let declared_map: HashMap<String, Expectation> = run
            .declared
            .iter()
            .map(|(name, expectation)| (name.clone(), *expectation))
            .collect();
        if check_integrity(&declared_map, &run.measured).is_err() {
            return Err(ScientificManifestError::IntegrityNotSatisfied);
        }

        if exact_config_bytes.is_empty() {
            return Err(ScientificManifestError::Validation(
                "exact configuration bytes cannot be empty".to_string(),
            ));
        }
        if exact_environment_bytes.is_some_and(|bytes| bytes.is_empty()) {
            return Err(ScientificManifestError::Validation(
                "exact environment bytes cannot be empty when supplied".to_string(),
            ));
        }
        if exact_seed_plan_bytes.is_some_and(|bytes| bytes.is_empty()) {
            return Err(ScientificManifestError::Validation(
                "exact seed-plan bytes cannot be empty when supplied".to_string(),
            ));
        }

        producer.validate()?;
        validate_text(&run.run_id.0, "run id")?;
        for (name, expectation) in &run.declared {
            validate_text(name, "declared expectation name")?;
            validate_expectation(*expectation)?;
        }
        if artifacts.len() > MAX_ARTIFACTS {
            return Err(ScientificManifestError::Validation(format!(
                "scientific run manifest cannot contain more than {MAX_ARTIFACTS} artifacts"
            )));
        }

        artifacts.sort_by(|left, right| left.artifact_id.cmp(&right.artifact_id));
        let mut seen_artifacts = BTreeSet::new();
        for artifact in &artifacts {
            artifact.validate()?;
            if !seen_artifacts.insert(artifact.artifact_id.as_str()) {
                return Err(ScientificManifestError::Validation(
                    "scientific run manifest contains duplicate artifact ids".to_string(),
                ));
            }
        }

        let measured = run
            .measured
            .iter()
            .map(|(name, value)| (name.clone(), *value))
            .collect::<BTreeMap<_, _>>();
        let manifest = Self {
            protocol: SCIENTIFIC_RUN_MANIFEST_PROTOCOL.to_string(),
            protocol_version: SCIENTIFIC_RUN_MANIFEST_VERSION,
            run_id: run.run_id.0.clone(),
            producer,
            integrity_satisfied: true,
            config_digest: ScientificContentHash::digest(exact_config_bytes),
            environment_digest: exact_environment_bytes.map(ScientificContentHash::digest),
            seed_plan_digest: exact_seed_plan_bytes.map(ScientificContentHash::digest),
            declared: run.declared.clone(),
            measured,
            artifacts,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), ScientificManifestError> {
        if self.protocol != SCIENTIFIC_RUN_MANIFEST_PROTOCOL {
            return Err(ScientificManifestError::Validation(format!(
                "unsupported scientific run manifest protocol: {}",
                self.protocol
            )));
        }
        if self.protocol_version != SCIENTIFIC_RUN_MANIFEST_VERSION {
            return Err(ScientificManifestError::Validation(format!(
                "unsupported scientific run manifest version: {}",
                self.protocol_version
            )));
        }
        if !self.integrity_satisfied {
            return Err(ScientificManifestError::IntegrityNotSatisfied);
        }
        if self.config_digest == ScientificContentHash::digest(&[]) {
            return Err(ScientificManifestError::Validation(
                "exact configuration digest cannot represent empty bytes".to_string(),
            ));
        }
        if self
            .environment_digest
            .is_some_and(|digest| digest == ScientificContentHash::digest(&[]))
        {
            return Err(ScientificManifestError::Validation(
                "environment digest cannot represent empty bytes".to_string(),
            ));
        }
        if self
            .seed_plan_digest
            .is_some_and(|digest| digest == ScientificContentHash::digest(&[]))
        {
            return Err(ScientificManifestError::Validation(
                "seed-plan digest cannot represent empty bytes".to_string(),
            ));
        }

        validate_text(&self.run_id, "run id")?;
        self.producer.validate()?;
        if self.artifacts.len() > MAX_ARTIFACTS {
            return Err(ScientificManifestError::Validation(format!(
                "scientific run manifest cannot contain more than {MAX_ARTIFACTS} artifacts"
            )));
        }

        let mut previous: Option<&str> = None;
        for artifact in &self.artifacts {
            artifact.validate()?;
            if previous.is_some_and(|id| id >= artifact.artifact_id.as_str()) {
                return Err(ScientificManifestError::Validation(
                    "scientific run artifacts must be unique and sorted by artifact id".to_string(),
                ));
            }
            previous = Some(&artifact.artifact_id);
        }

        let mut declared_map = HashMap::new();
        for (name, expectation) in &self.declared {
            validate_text(name, "declared expectation name")?;
            validate_expectation(*expectation)?;
            declared_map.insert(name.clone(), *expectation);
        }

        let mut measured_counters = EvidenceCounters::new();
        for (name, value) in &self.measured {
            validate_text(name, "measured evidence name")?;
            if !value.is_finite() {
                return Err(ScientificManifestError::Validation(format!(
                    "measured evidence value for {name} must be finite"
                )));
            }
            measured_counters.record(name.clone(), *value);
        }

        if check_integrity(&declared_map, &measured_counters).is_err() {
            return Err(ScientificManifestError::IntegrityNotSatisfied);
        }
        Ok(())
    }

    pub fn protocol(&self) -> &str {
        &self.protocol
    }

    pub const fn protocol_version(&self) -> u16 {
        self.protocol_version
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn producer(&self) -> &RunProducerIdentity {
        &self.producer
    }

    pub const fn integrity_satisfied(&self) -> bool {
        self.integrity_satisfied
    }

    pub const fn config_digest(&self) -> ScientificContentHash {
        self.config_digest
    }

    pub const fn environment_digest(&self) -> Option<ScientificContentHash> {
        self.environment_digest
    }

    pub const fn seed_plan_digest(&self) -> Option<ScientificContentHash> {
        self.seed_plan_digest
    }

    pub fn declared(&self) -> &BTreeMap<String, Expectation> {
        &self.declared
    }

    pub fn measured(&self) -> &BTreeMap<String, f64> {
        &self.measured
    }

    pub fn artifacts(&self) -> &[ScientificRunArtifact] {
        &self.artifacts
    }

    /// Deterministic, explicitly framed bytes suitable for cross-repository
    /// golden vectors and cryptographic identity. This is not serde output.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ScientificManifestError> {
        self.validate()?;
        let mut encoder = ManifestEncoder::new();
        encoder.string(&self.protocol)?;
        encoder.u16(self.protocol_version);
        encoder.string(&self.run_id)?;
        encoder.producer(&self.producer)?;
        encoder.bool(self.integrity_satisfied);
        encoder.hash(self.config_digest);
        encoder.option_hash(self.environment_digest);
        encoder.option_hash(self.seed_plan_digest);

        encoder.len(self.declared.len())?;
        for (name, expectation) in &self.declared {
            encoder.string(name)?;
            encoder.expectation(*expectation);
        }

        encoder.len(self.measured.len())?;
        for (name, value) in &self.measured {
            encoder.string(name)?;
            encoder.f64(*value);
        }

        encoder.len(self.artifacts.len())?;
        for artifact in &self.artifacts {
            encoder.string(&artifact.artifact_id)?;
            encoder.hash(artifact.content_hash);
            encoder.string(&artifact.media_type)?;
            encoder.option_string(artifact.role_hint.as_deref())?;
        }
        Ok(encoder.finish())
    }

    pub fn manifest_hash(&self) -> Result<ScientificContentHash, ScientificManifestError> {
        Ok(ScientificContentHash::digest(&self.canonical_bytes()?))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScientificManifestError {
    IntegrityNotSatisfied,
    Validation(String),
    LengthOverflow,
}

impl fmt::Display for ScientificManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IntegrityNotSatisfied => {
                f.write_str("scientific run integrity was not satisfied")
            }
            Self::Validation(message) => f.write_str(message),
            Self::LengthOverflow => f.write_str("scientific manifest field exceeds u32 length"),
        }
    }
}

impl std::error::Error for ScientificManifestError {}

fn validate_text(value: &str, label: &str) -> Result<(), ScientificManifestError> {
    if value.trim().is_empty() {
        return Err(ScientificManifestError::Validation(format!(
            "{label} cannot be empty"
        )));
    }
    if value.trim() != value {
        return Err(ScientificManifestError::Validation(format!(
            "{label} cannot contain leading or trailing whitespace"
        )));
    }
    if value.len() > MAX_TEXT_BYTES {
        return Err(ScientificManifestError::Validation(format!(
            "{label} cannot exceed {MAX_TEXT_BYTES} bytes"
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(ScientificManifestError::Validation(format!(
            "{label} cannot contain control characters"
        )));
    }
    Ok(())
}

fn validate_expectation(expectation: Expectation) -> Result<(), ScientificManifestError> {
    match expectation {
        Expectation::MustExceed(value) | Expectation::MustBeBelow(value) if !value.is_finite() => {
            Err(ScientificManifestError::Validation(
                "scientific expectation threshold must be finite".to_string(),
            ))
        }
        _ => Ok(()),
    }
}

struct ManifestEncoder {
    bytes: Vec<u8>,
}

impl ManifestEncoder {
    fn new() -> Self {
        Self {
            bytes: MANIFEST_DOMAIN.to_vec(),
        }
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }

    fn bool(&mut self, value: bool) {
        self.bytes.push(u8::from(value));
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn len(&mut self, len: usize) -> Result<(), ScientificManifestError> {
        let len = u32::try_from(len).map_err(|_| ScientificManifestError::LengthOverflow)?;
        self.bytes.extend_from_slice(&len.to_be_bytes());
        Ok(())
    }

    fn string(&mut self, value: &str) -> Result<(), ScientificManifestError> {
        self.len(value.len())?;
        self.bytes.extend_from_slice(value.as_bytes());
        Ok(())
    }

    fn option_string(&mut self, value: Option<&str>) -> Result<(), ScientificManifestError> {
        match value {
            Some(value) => {
                self.u8(1);
                self.string(value)?;
            }
            None => self.u8(0),
        }
        Ok(())
    }

    fn hash(&mut self, value: ScientificContentHash) {
        self.bytes.extend_from_slice(&value.0);
    }

    fn option_hash(&mut self, value: Option<ScientificContentHash>) {
        match value {
            Some(value) => {
                self.u8(1);
                self.hash(value);
            }
            None => self.u8(0),
        }
    }

    fn f64(&mut self, value: f64) {
        self.bytes.extend_from_slice(&value.to_bits().to_be_bytes());
    }

    fn producer(&mut self, producer: &RunProducerIdentity) -> Result<(), ScientificManifestError> {
        self.string(&producer.component)?;
        self.string(&producer.version)?;
        self.option_string(producer.revision.as_deref())
    }

    fn expectation(&mut self, expectation: Expectation) {
        match expectation {
            Expectation::MustBeZero => self.u8(1),
            Expectation::MustBePositive => self.u8(2),
            Expectation::MustExceed(value) => {
                self.u8(3);
                self.f64(value);
            }
            Expectation::MustBeBelow(value) => {
                self.u8(4);
                self.f64(value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EvidenceCounters, FailedExpectation, RunId};

    fn passing_run() -> RunEvidence {
        let declared = BTreeMap::from([
            ("active".to_string(), Expectation::MustBePositive),
            ("forbidden".to_string(), Expectation::MustBeZero),
        ]);
        let mut measured = EvidenceCounters::new();
        measured.record("active", 2.0);
        RunEvidence::new(RunId::new("run-001"), &"legacy-config", declared, measured)
    }

    fn producer() -> RunProducerIdentity {
        RunProducerIdentity::new(
            "symthaea-test-runner",
            "1.0.0",
            Some("0123456789abcdef".to_string()),
        )
        .unwrap()
    }

    #[test]
    fn manifest_rejects_failed_integrity_run() {
        let declared = BTreeMap::from([("active".to_string(), Expectation::MustBePositive)]);
        let measured = EvidenceCounters::new();
        let run = RunEvidence::new(RunId::new("failed"), &"legacy-config", declared, measured);
        assert!(!run.satisfied);

        let result = ScientificRunManifest::from_run_evidence(
            &run,
            producer(),
            b"exact-config",
            None,
            None,
            Vec::new(),
        );
        assert_eq!(result.unwrap_err(), ScientificManifestError::IntegrityNotSatisfied);
    }

    #[test]
    fn forged_cached_integrity_flags_cannot_bypass_recomputation() {
        let declared = BTreeMap::from([("active".to_string(), Expectation::MustBePositive)]);
        let measured = EvidenceCounters::new();
        let mut run = RunEvidence::new(
            RunId::new("forged-failed"),
            &"legacy-config",
            declared,
            measured,
        );
        assert!(!run.satisfied);
        run.satisfied = true;
        run.violations.clear();

        let result = ScientificRunManifest::from_run_evidence(
            &run,
            producer(),
            b"exact-config",
            None,
            None,
            Vec::new(),
        );
        assert_eq!(result.unwrap_err(), ScientificManifestError::IntegrityNotSatisfied);
    }

    #[test]
    fn exact_reproducibility_inputs_cannot_be_empty() {
        let run = passing_run();
        assert!(ScientificRunManifest::from_run_evidence(
            &run,
            producer(),
            b"",
            None,
            None,
            Vec::new(),
        )
        .is_err());
        assert!(ScientificRunManifest::from_run_evidence(
            &run,
            producer(),
            b"config",
            Some(b""),
            None,
            Vec::new(),
        )
        .is_err());
        assert!(ScientificRunManifest::from_run_evidence(
            &run,
            producer(),
            b"config",
            None,
            Some(b""),
            Vec::new(),
        )
        .is_err());
    }

    #[test]
    fn non_finite_expectation_threshold_is_rejected() {
        let declared = BTreeMap::from([("active".to_string(), Expectation::MustExceed(f64::NAN))]);
        let mut measured = EvidenceCounters::new();
        measured.record("active", 2.0);
        let mut run = RunEvidence::new(
            RunId::new("nan-threshold"),
            &"legacy-config",
            declared,
            measured,
        );
        // Even if cached state is forged, threshold validation and integrity
        // recomputation keep this run out of the export boundary.
        run.satisfied = true;
        run.violations = Vec::<FailedExpectation>::new();
        assert!(ScientificRunManifest::from_run_evidence(
            &run,
            producer(),
            b"config",
            None,
            None,
            Vec::new(),
        )
        .is_err());
    }

    #[test]
    fn exact_config_bytes_receive_cryptographic_identity() {
        let run = passing_run();
        let manifest_a = ScientificRunManifest::from_run_evidence(
            &run,
            producer(),
            b"config-a",
            None,
            None,
            Vec::new(),
        )
        .unwrap();
        let manifest_b = ScientificRunManifest::from_run_evidence(
            &run,
            producer(),
            b"config-b",
            None,
            None,
            Vec::new(),
        )
        .unwrap();

        assert_ne!(manifest_a.config_digest(), manifest_b.config_digest());
        assert_ne!(manifest_a.manifest_hash().unwrap(), manifest_b.manifest_hash().unwrap());
    }

    #[test]
    fn artifact_order_does_not_change_manifest_identity() {
        let run = passing_run();
        let a = ScientificRunArtifact::from_bytes(
            "a",
            "application/octet-stream",
            Some("raw_observation".to_string()),
            b"artifact-a",
        )
        .unwrap();
        let b = ScientificRunArtifact::from_bytes(
            "b",
            "application/octet-stream",
            None,
            b"artifact-b",
        )
        .unwrap();

        let manifest_ab = ScientificRunManifest::from_run_evidence(
            &run,
            producer(),
            b"config",
            Some(b"environment"),
            Some(b"seed-plan"),
            vec![a.clone(), b.clone()],
        )
        .unwrap();
        let manifest_ba = ScientificRunManifest::from_run_evidence(
            &run,
            producer(),
            b"config",
            Some(b"environment"),
            Some(b"seed-plan"),
            vec![b, a],
        )
        .unwrap();

        assert_eq!(manifest_ab.canonical_bytes().unwrap(), manifest_ba.canonical_bytes().unwrap());
        assert_eq!(manifest_ab.manifest_hash().unwrap(), manifest_ba.manifest_hash().unwrap());
    }

    #[test]
    fn duplicate_artifact_ids_are_rejected() {
        let run = passing_run();
        let a = ScientificRunArtifact::from_bytes("same", "text/plain", None, b"a").unwrap();
        let b = ScientificRunArtifact::from_bytes("same", "text/plain", None, b"b").unwrap();
        let result = ScientificRunManifest::from_run_evidence(
            &run,
            producer(),
            b"config",
            None,
            None,
            vec![a, b],
        );
        assert!(result.is_err());
    }

    #[test]
    fn deserialization_revalidates_integrity() {
        let manifest = ScientificRunManifest::from_run_evidence(
            &passing_run(),
            producer(),
            b"config",
            None,
            None,
            Vec::new(),
        )
        .unwrap();
        let mut value = serde_json::to_value(&manifest).unwrap();
        value["measured"]["active"] = serde_json::json!(0.0);
        assert!(serde_json::from_value::<ScientificRunManifest>(value).is_err());
    }

    #[test]
    fn manifest_does_not_embed_legacy_config_hash() {
        let run = passing_run();
        let manifest = ScientificRunManifest::from_run_evidence(
            &run,
            producer(),
            b"exact-config",
            None,
            None,
            Vec::new(),
        )
        .unwrap();
        let bytes = manifest.canonical_bytes().unwrap();
        assert!(!bytes
            .windows(run.config_hash.len())
            .any(|window| window == run.config_hash.as_bytes()));
    }
}
