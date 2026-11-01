//! Replicate-scoped evidence binding for noisy multi-objective observations.
//!
//! The existing candidate-level artifact manifest is not sufficient to prove which raw bytes
//! produced one particular noisy replicate. This module binds each replicate to a canonical
//! JSON observation envelope and checks that the envelope's objective vector is exactly the
//! vector stored in the noisy history. A receipt proves byte and identity correspondence only;
//! it does not authenticate a remote store, verifier identity, evaluator correctness, or the
//! statistical independence of the sample.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fmt;

use super::evidence_artifact::{EvidenceArtifactReference, EvidenceArtifactRole};
use super::pareto_noisy_observations::{NoisyObservationPlan, NoisyReplicate};

const MANIFEST_SCHEMA: &str = "symthaea.architecture-engineering.noisy-replicate-evidence";
const MANIFEST_VERSION: &str = "0.1.0";
const ARTIFACT_SCHEMA: &str = "symthaea.architecture-engineering.noisy-replicate-observation";
const ARTIFACT_VERSION: &str = "0.1.0";
const RECEIPT_SCHEMA: &str = "symthaea.architecture-engineering.noisy-replicate-byte-receipt";
const RECEIPT_VERSION: &str = "0.1.0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoisyReplicateEvidenceError {
    pub path: String,
    pub message: String,
}

impl NoisyReplicateEvidenceError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for NoisyReplicateEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for NoisyReplicateEvidenceError {}

/// One objective coordinate in the canonical raw observation artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoisyObjectiveValue {
    pub objective_id: String,
    pub value: f64,
}

/// Canonical JSON envelope retained for one successful noisy replicate.
///
/// The objective IDs and order come from the preregistered plan. This envelope is a
/// machine-readable measurement record, not a substitute for retaining any underlying
/// instrument logs or evaluator outputs needed to justify the measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoisyReplicateObservationArtifact {
    pub schema: String,
    pub version: String,
    pub run_id: String,
    pub candidate_id: String,
    pub source_commit: String,
    pub decision_sha256: String,
    pub replicate_id: String,
    pub sample_seed: u64,
    pub plan_sha256: String,
    pub objective_values: Vec<NoisyObjectiveValue>,
}

/// Frozen reference binding one replicate to one raw observation artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoisyReplicateEvidenceManifest {
    pub schema: String,
    pub version: String,
    pub run_id: String,
    pub candidate_id: String,
    pub source_commit: String,
    pub decision_sha256: String,
    pub replicate_id: String,
    pub sample_seed: u64,
    pub plan_sha256: String,
    pub objective_vector_sha256: String,
    pub artifact: EvidenceArtifactReference,
}

#[derive(Serialize)]
struct ManifestDigestPayload<'a> {
    schema: &'a str,
    version: &'a str,
    run_id: &'a str,
    candidate_id: &'a str,
    source_commit: &'a str,
    decision_sha256: &'a str,
    replicate_id: &'a str,
    sample_seed: u64,
    plan_sha256: &'a str,
    objective_vector_sha256: &'a str,
    artifact: &'a EvidenceArtifactReference,
}

impl NoisyReplicateEvidenceManifest {
    /// Construct a replicate manifest from the same observation that will be appended to
    /// NoisyParetoStudy/AnytimeHistory. The artifact reference must name the canonical JSON
    /// envelope whose SHA-256 is also carried by NoisyReplicate.evidence_sha256.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        run_id: impl Into<String>,
        candidate_id: impl Into<String>,
        source_commit: impl Into<String>,
        decision_sha256: impl Into<String>,
        plan: &NoisyObservationPlan,
        replicate: &NoisyReplicate,
        artifact: EvidenceArtifactReference,
    ) -> Result<Self, NoisyReplicateEvidenceError> {
        plan.validate().map_err(|error| {
            NoisyReplicateEvidenceError::new("plan", error.to_string())
        })?;
        let objective_values = objective_values_for_replicate(plan, replicate)?;
        let manifest = Self {
            schema: MANIFEST_SCHEMA.to_owned(),
            version: MANIFEST_VERSION.to_owned(),
            run_id: run_id.into(),
            candidate_id: candidate_id.into(),
            source_commit: source_commit.into(),
            decision_sha256: decision_sha256.into(),
            replicate_id: replicate.replicate_id.clone(),
            sample_seed: replicate.sample_seed,
            plan_sha256: plan.plan_sha256().map_err(|error| {
                NoisyReplicateEvidenceError::new("plan", error.to_string())
            })?,
            objective_vector_sha256: digest_serializable(&objective_values)?,
            artifact,
        };
        manifest.validate()?;
        if manifest.artifact.sha256 != replicate.evidence_sha256 {
            return Err(NoisyReplicateEvidenceError::new(
                "artifact.sha256",
                "artifact digest must exactly match NoisyReplicate.evidence_sha256",
            ));
        }
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), NoisyReplicateEvidenceError> {
        if self.schema != MANIFEST_SCHEMA || self.version != MANIFEST_VERSION {
            return Err(NoisyReplicateEvidenceError::new(
                "manifest.schema",
                "unsupported noisy-replicate evidence schema/version",
            ));
        }
        for (path, value) in [
            ("run_id", self.run_id.as_str()),
            ("candidate_id", self.candidate_id.as_str()),
            ("replicate_id", self.replicate_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(NoisyReplicateEvidenceError::new(
                    format!("manifest.{path}"),
                    "value must be non-empty",
                ));
            }
        }
        validate_commit(&self.source_commit, "manifest.source_commit")?;
        validate_sha256(&self.decision_sha256, "manifest.decision_sha256")?;
        validate_sha256(&self.plan_sha256, "manifest.plan_sha256")?;
        validate_sha256(
            &self.objective_vector_sha256,
            "manifest.objective_vector_sha256",
        )?;
        if self.artifact.artifact_id.trim().is_empty() {
            return Err(NoisyReplicateEvidenceError::new(
                "manifest.artifact.artifact_id",
                "artifact ID must be non-empty",
            ));
        }
        if self.artifact.role != EvidenceArtifactRole::EvaluationOutput {
            return Err(NoisyReplicateEvidenceError::new(
                "manifest.artifact.role",
                "replicate observation artifact role must be evaluation_output",
            ));
        }
        if self.artifact.media_type != "application/json" {
            return Err(NoisyReplicateEvidenceError::new(
                "manifest.artifact.media_type",
                "canonical replicate observation artifact must use application/json",
            ));
        }
        if self.artifact.storage_locator.trim().is_empty() || self.artifact.byte_length == 0 {
            return Err(NoisyReplicateEvidenceError::new(
                "manifest.artifact",
                "storage locator and non-zero byte length are required",
            ));
        }
        validate_sha256(&self.artifact.sha256, "manifest.artifact.sha256")?;
        Ok(())
    }

    pub fn manifest_sha256(&self) -> Result<String, NoisyReplicateEvidenceError> {
        self.validate()?;
        digest_serializable(&ManifestDigestPayload {
            schema: &self.schema,
            version: &self.version,
            run_id: &self.run_id,
            candidate_id: &self.candidate_id,
            source_commit: &self.source_commit,
            decision_sha256: &self.decision_sha256,
            replicate_id: &self.replicate_id,
            sample_seed: self.sample_seed,
            plan_sha256: &self.plan_sha256,
            objective_vector_sha256: &self.objective_vector_sha256,
            artifact: &self.artifact,
        })
    }
}

/// Expected campaign and candidate identity supplied by the history/report being verified.
#[derive(Debug, Clone, Copy)]
pub struct NoisyReplicateEvidenceContext<'a> {
    pub run_id: &'a str,
    pub source_commit: &'a str,
    pub candidate_id: &'a str,
    pub decision_sha256: &'a str,
}

/// Byte-verification receipt for one exact replicate and objective vector.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoisyReplicateEvidenceReceipt {
    pub schema: String,
    pub version: String,
    pub run_id: String,
    pub candidate_id: String,
    pub source_commit: String,
    pub decision_sha256: String,
    pub replicate_id: String,
    pub sample_seed: u64,
    pub plan_sha256: String,
    pub manifest_sha256: String,
    pub artifact_sha256: String,
    pub artifact_byte_length: u64,
    pub objective_vector_sha256: String,
    pub verifier_id: String,
    pub verifier_build_sha256: String,
    pub receipt_sha256: String,
}

#[derive(Serialize)]
struct ReceiptDigestPayload<'a> {
    schema: &'a str,
    version: &'a str,
    run_id: &'a str,
    candidate_id: &'a str,
    source_commit: &'a str,
    decision_sha256: &'a str,
    replicate_id: &'a str,
    sample_seed: u64,
    plan_sha256: &'a str,
    manifest_sha256: &'a str,
    artifact_sha256: &'a str,
    artifact_byte_length: u64,
    objective_vector_sha256: &'a str,
    verifier_id: &'a str,
    verifier_build_sha256: &'a str,
}

/// Encode the canonical JSON envelope that a successful noisy replicate must retain.
pub fn encode_noisy_replicate_observation_artifact(
    context: NoisyReplicateEvidenceContext<'_>,
    plan: &NoisyObservationPlan,
    replicate: &NoisyReplicate,
) -> Result<Vec<u8>, NoisyReplicateEvidenceError> {
    plan.validate().map_err(|error| {
        NoisyReplicateEvidenceError::new("plan", error.to_string())
    })?;
    validate_context(context)?;
    let objective_values = objective_values_for_replicate(plan, replicate)?;
    let artifact = NoisyReplicateObservationArtifact {
        schema: ARTIFACT_SCHEMA.to_owned(),
        version: ARTIFACT_VERSION.to_owned(),
        run_id: context.run_id.to_owned(),
        candidate_id: context.candidate_id.to_owned(),
        source_commit: context.source_commit.to_owned(),
        decision_sha256: context.decision_sha256.to_owned(),
        replicate_id: replicate.replicate_id.clone(),
        sample_seed: replicate.sample_seed,
        plan_sha256: plan.plan_sha256().map_err(|error| {
            NoisyReplicateEvidenceError::new("plan", error.to_string())
        })?,
        objective_values,
    };
    serde_json::to_vec(&artifact).map_err(|error| {
        NoisyReplicateEvidenceError::new(
            "artifact.serialization",
            format!("cannot serialize canonical replicate observation: {error}"),
        )
    })
}

/// Verify a canonical raw observation envelope and reconstruct its byte receipt.
///
/// This verifies exact bytes, run/candidate/decision/replicate identity, the frozen sampling
/// plan, objective ID/order, and exact f64 objective values against the history sample. It
/// does not prove that the evaluator measured the physical system correctly or that the sample
/// is statistically independent.
pub fn verify_noisy_replicate_artifact_bytes(
    manifest: &NoisyReplicateEvidenceManifest,
    plan: &NoisyObservationPlan,
    context: NoisyReplicateEvidenceContext<'_>,
    replicate: &NoisyReplicate,
    bytes: &[u8],
    verifier_id: impl Into<String>,
    verifier_build_sha256: impl Into<String>,
) -> Result<NoisyReplicateEvidenceReceipt, NoisyReplicateEvidenceError> {
    manifest.validate()?;
    plan.validate().map_err(|error| {
        NoisyReplicateEvidenceError::new("plan", error.to_string())
    })?;
    validate_context(context)?;

    let plan_sha256 = plan.plan_sha256().map_err(|error| {
        NoisyReplicateEvidenceError::new("plan", error.to_string())
    })?;
    if manifest.run_id != context.run_id
        || manifest.source_commit != context.source_commit
        || manifest.candidate_id != context.candidate_id
        || manifest.decision_sha256 != context.decision_sha256
    {
        return Err(NoisyReplicateEvidenceError::new(
            "manifest.identity",
            "manifest run/source/candidate/decision identity differs from the expected history context",
        ));
    }
    if manifest.plan_sha256 != plan_sha256 {
        return Err(NoisyReplicateEvidenceError::new(
            "manifest.plan_sha256",
            "manifest is bound to a different noisy-observation plan",
        ));
    }
    if manifest.replicate_id != replicate.replicate_id
        || manifest.sample_seed != replicate.sample_seed
    {
        return Err(NoisyReplicateEvidenceError::new(
            "manifest.replicate",
            "manifest replicate ID or seed differs from the history sample",
        ));
    }
    if manifest.artifact.sha256 != replicate.evidence_sha256 {
        return Err(NoisyReplicateEvidenceError::new(
            "manifest.artifact.sha256",
            "manifested artifact digest differs from the history sample evidence digest",
        ));
    }

    let actual_length = u64::try_from(bytes.len()).map_err(|_| {
        NoisyReplicateEvidenceError::new("artifact.byte_length", "artifact exceeds u64 byte length")
    })?;
    if actual_length != manifest.artifact.byte_length {
        return Err(NoisyReplicateEvidenceError::new(
            "artifact.byte_length",
            "supplied bytes differ from the manifested byte length",
        ));
    }
    let artifact_sha256 = sha256_bytes(bytes);
    if artifact_sha256 != manifest.artifact.sha256 {
        return Err(NoisyReplicateEvidenceError::new(
            "artifact.sha256",
            "supplied bytes differ from the manifested SHA-256 digest",
        ));
    }

    let artifact: NoisyReplicateObservationArtifact =
        serde_json::from_slice(bytes).map_err(|error| {
            NoisyReplicateEvidenceError::new(
                "artifact.json",
                format!("cannot decode canonical replicate observation: {error}"),
            )
        })?;
    // Canonical encoding avoids multiple byte representations for the same logical record.
    let canonical = serde_json::to_vec(&artifact).map_err(|error| {
        NoisyReplicateEvidenceError::new(
            "artifact.serialization",
            format!("cannot re-encode replicate observation: {error}"),
        )
    })?;
    if canonical != bytes {
        return Err(NoisyReplicateEvidenceError::new(
            "artifact.json",
            "observation envelope is not in canonical serde_json encoding",
        ));
    }

    let expected_values = objective_values_for_replicate(plan, replicate)?;
    if artifact.schema != ARTIFACT_SCHEMA
        || artifact.version != ARTIFACT_VERSION
        || artifact.run_id != context.run_id
        || artifact.candidate_id != context.candidate_id
        || artifact.source_commit != context.source_commit
        || artifact.decision_sha256 != context.decision_sha256
        || artifact.replicate_id != replicate.replicate_id
        || artifact.sample_seed != replicate.sample_seed
        || artifact.plan_sha256 != plan_sha256
        || artifact.objective_values != expected_values
    {
        return Err(NoisyReplicateEvidenceError::new(
            "artifact.identity_or_objectives",
            "raw observation envelope differs from the expected history identity, plan, or objective vector",
        ));
    }
    let objective_vector_sha256 = digest_serializable(&expected_values)?;
    if objective_vector_sha256 != manifest.objective_vector_sha256 {
        return Err(NoisyReplicateEvidenceError::new(
            "manifest.objective_vector_sha256",
            "manifest objective-vector digest differs from the history sample",
        ));
    }

    let verifier_id = verifier_id.into();
    let verifier_build_sha256 = verifier_build_sha256.into();
    if verifier_id.trim().is_empty() {
        return Err(NoisyReplicateEvidenceError::new(
            "verifier_id",
            "verifier identity must be non-empty",
        ));
    }
    validate_sha256(&verifier_build_sha256, "verifier_build_sha256")?;

    let manifest_sha256 = manifest.manifest_sha256()?;
    let payload = ReceiptDigestPayload {
        schema: RECEIPT_SCHEMA,
        version: RECEIPT_VERSION,
        run_id: &manifest.run_id,
        candidate_id: &manifest.candidate_id,
        source_commit: &manifest.source_commit,
        decision_sha256: &manifest.decision_sha256,
        replicate_id: &manifest.replicate_id,
        sample_seed: manifest.sample_seed,
        plan_sha256: &manifest.plan_sha256,
        manifest_sha256: &manifest_sha256,
        artifact_sha256: &artifact_sha256,
        artifact_byte_length: actual_length,
        objective_vector_sha256: &objective_vector_sha256,
        verifier_id: &verifier_id,
        verifier_build_sha256: &verifier_build_sha256,
    };
    let receipt_sha256 = digest_serializable(&payload)?;

    Ok(NoisyReplicateEvidenceReceipt {
        schema: RECEIPT_SCHEMA.to_owned(),
        version: RECEIPT_VERSION.to_owned(),
        run_id: manifest.run_id.clone(),
        candidate_id: manifest.candidate_id.clone(),
        source_commit: manifest.source_commit.clone(),
        decision_sha256: manifest.decision_sha256.clone(),
        replicate_id: manifest.replicate_id.clone(),
        sample_seed: manifest.sample_seed,
        plan_sha256: manifest.plan_sha256.clone(),
        manifest_sha256,
        artifact_sha256,
        artifact_byte_length: actual_length,
        objective_vector_sha256,
        verifier_id,
        verifier_build_sha256,
        receipt_sha256,
    })
}

/// Recompute the byte receipt from the exact inputs; altered receipts fail closed.
pub fn verify_noisy_replicate_artifact_receipt(
    manifest: &NoisyReplicateEvidenceManifest,
    plan: &NoisyObservationPlan,
    context: NoisyReplicateEvidenceContext<'_>,
    replicate: &NoisyReplicate,
    bytes: &[u8],
    receipt: &NoisyReplicateEvidenceReceipt,
) -> Result<(), NoisyReplicateEvidenceError> {
    let reconstructed = verify_noisy_replicate_artifact_bytes(
        manifest,
        plan,
        context,
        replicate,
        bytes,
        receipt.verifier_id.clone(),
        receipt.verifier_build_sha256.clone(),
    )?;
    if &reconstructed != receipt {
        return Err(NoisyReplicateEvidenceError::new(
            "receipt",
            "replicate evidence receipt does not reconstruct from the manifest, plan, history sample, and supplied bytes",
        ));
    }
    Ok(())
}

fn objective_values_for_replicate(
    plan: &NoisyObservationPlan,
    replicate: &NoisyReplicate,
) -> Result<Vec<NoisyObjectiveValue>, NoisyReplicateEvidenceError> {
    if replicate.replicate_id.trim().is_empty() {
        return Err(NoisyReplicateEvidenceError::new(
            "replicate.replicate_id",
            "replicate ID must be non-empty",
        ));
    }
    if replicate.objective_values.len() != plan.objective_specs.len() {
        return Err(NoisyReplicateEvidenceError::new(
            "replicate.objective_values",
            "objective vector length differs from the frozen plan",
        ));
    }
    let mut objective_values = Vec::with_capacity(plan.objective_specs.len());
    for (index, (spec, value)) in plan
        .objective_specs
        .iter()
        .zip(&replicate.objective_values)
        .enumerate()
    {
        if !value.is_finite()
            || *value < spec.observation_lower_bound
            || *value > spec.observation_upper_bound
        {
            return Err(NoisyReplicateEvidenceError::new(
                format!("replicate.objective_values[{index}]"),
                "objective value is non-finite or outside its preregistered observation bounds",
            ));
        }
        objective_values.push(NoisyObjectiveValue {
            objective_id: spec.objective_id.clone(),
            value: *value,
        });
    }
    Ok(objective_values)
}

fn validate_context(
    context: NoisyReplicateEvidenceContext<'_>,
) -> Result<(), NoisyReplicateEvidenceError> {
    for (path, value) in [
        ("context.run_id", context.run_id),
        ("context.candidate_id", context.candidate_id),
    ] {
        if value.trim().is_empty() {
            return Err(NoisyReplicateEvidenceError::new(
                path,
                "value must be non-empty",
            ));
        }
    }
    validate_commit(context.source_commit, "context.source_commit")?;
    validate_sha256(context.decision_sha256, "context.decision_sha256")?;
    Ok(())
}

fn validate_sha256(value: &str, path: &str) -> Result<(), NoisyReplicateEvidenceError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(NoisyReplicateEvidenceError::new(
            path,
            "expected lowercase 64-character SHA-256 digest",
        ));
    }
    Ok(())
}

fn validate_commit(value: &str, path: &str) -> Result<(), NoisyReplicateEvidenceError> {
    if !matches!(value.len(), 40 | 64)
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(NoisyReplicateEvidenceError::new(
            path,
            "expected lowercase 40- or 64-character source commit ID",
        ));
    }
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Public digest helper for artifact producers and manifest construction.
pub fn noisy_replicate_artifact_sha256(bytes: &[u8]) -> String {
    sha256_bytes(bytes)
}

fn digest_serializable<T: Serialize>(
    value: &T,
) -> Result<String, NoisyReplicateEvidenceError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        NoisyReplicateEvidenceError::new(
            "serialization",
            format!("cannot serialize replicate evidence digest payload: {error}"),
        )
    })?;
    Ok(sha256_bytes(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::pareto_objectives::ObjectiveDirection;

    const SOURCE_COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
    const DECISION: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SAMPLING: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const NOISE: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const BOUNDS_A: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const BOUNDS_B: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const BUILD: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn plan() -> NoisyObservationPlan {
        NoisyObservationPlan::new(
            "campaign-evidence-test",
            "counter-based-independent-test",
            SAMPLING,
            "bounded-test-noise",
            NOISE,
            vec![
                super::super::pareto_noisy_observations::NoisyObjectiveSpec {
                    objective_id: "quality".to_owned(),
                    direction: ObjectiveDirection::Maximize,
                    observation_lower_bound: 0.0,
                    observation_upper_bound: 1.0,
                    bounds_certificate_sha256: BOUNDS_A.to_owned(),
                },
                super::super::pareto_noisy_observations::NoisyObjectiveSpec {
                    objective_id: "cost".to_owned(),
                    direction: ObjectiveDirection::Minimize,
                    observation_lower_bound: 0.0,
                    observation_upper_bound: 1.0,
                    bounds_certificate_sha256: BOUNDS_B.to_owned(),
                },
            ],
            0.05,
            8,
            2,
        )
        .unwrap()
    }

    fn replicate(evidence_sha256: String) -> NoisyReplicate {
        NoisyReplicate {
            replicate_id: "replicate-0001".to_owned(),
            sample_seed: 71,
            objective_values: vec![0.8, 0.25],
            evidence_sha256,
        }
    }

    fn context() -> NoisyReplicateEvidenceContext<'static> {
        NoisyReplicateEvidenceContext {
            run_id: "run-2026-11-01-01",
            source_commit: SOURCE_COMMIT,
            candidate_id: "candidate-a",
            decision_sha256: DECISION,
        }
    }

    fn manifest_and_bytes(
        sample: &NoisyReplicate,
    ) -> (NoisyReplicateEvidenceManifest, Vec<u8>) {
        let plan = plan();
        let bytes = encode_noisy_replicate_observation_artifact(context(), &plan, sample).unwrap();
        let artifact_sha256 = noisy_replicate_artifact_sha256(&bytes);
        let mut sample_with_digest = sample.clone();
        sample_with_digest.evidence_sha256 = artifact_sha256.clone();
        let artifact = EvidenceArtifactReference {
            artifact_id: "replicate-observation".to_owned(),
            role: EvidenceArtifactRole::EvaluationOutput,
            media_type: "application/json".to_owned(),
            storage_locator: "cas://campaign/run-01/candidate-a/replicate-0001".to_owned(),
            byte_length: bytes.len() as u64,
            sha256: artifact_sha256,
        };
        let manifest = NoisyReplicateEvidenceManifest::new(
            context().run_id,
            context().candidate_id,
            context().source_commit,
            context().decision_sha256,
            &plan,
            &sample_with_digest,
            artifact,
        ).unwrap();
        (manifest, bytes)
    }

    #[test]
    fn exact_raw_bytes_and_objective_vector_produce_reconstructable_receipt() {
        let plan = plan();
        let (manifest, bytes) = manifest_and_bytes(&replicate("0".repeat(64)));
        let mut sample = replicate("0".repeat(64));
        sample.evidence_sha256 = manifest.artifact.sha256.clone();

        let receipt = verify_noisy_replicate_artifact_bytes(
            &manifest, &plan, context(), &sample, &bytes, "independent-verifier-v1", BUILD,
        ).unwrap();
        assert_eq!(receipt.run_id, context().run_id);
        assert_eq!(receipt.candidate_id, context().candidate_id);
        assert_eq!(receipt.replicate_id, sample.replicate_id);
        assert_eq!(receipt.artifact_sha256, manifest.artifact.sha256);
        assert_eq!(receipt.objective_vector_sha256, manifest.objective_vector_sha256);
        verify_noisy_replicate_artifact_receipt(
            &manifest, &plan, context(), &sample, &bytes, &receipt,
        ).unwrap();
    }

    #[test]
    fn byte_mutation_is_rejected_even_when_json_still_decodes() {
        let plan = plan();
        let (manifest, mut bytes) = manifest_and_bytes(&replicate("0".repeat(64)));
        let mut sample = replicate("0".repeat(64));
        sample.evidence_sha256 = manifest.artifact.sha256.clone();
        let position = bytes
            .windows(3)
            .position(|window| window == b"0.8")
            .expect("fixture contains the first objective value");
        bytes[position + 2] = b'9';
        assert!(verify_noisy_replicate_artifact_bytes(
            &manifest, &plan, context(), &sample, &bytes, "verifier", BUILD,
        ).is_err());
    }

    #[test]
    fn changed_objective_vector_cannot_be_substituted_even_with_a_new_artifact_digest() {
        let plan = plan();
        let original = replicate("0".repeat(64));
        let mut substituted = original.clone();
        substituted.objective_values[0] = 0.2;
        let bytes = encode_noisy_replicate_observation_artifact(context(), &plan, &substituted).unwrap();
        let artifact_sha256 = noisy_replicate_artifact_sha256(&bytes);
        let mut substituted_with_digest = substituted.clone();
        substituted_with_digest.evidence_sha256 = artifact_sha256.clone();
        let artifact = EvidenceArtifactReference {
            artifact_id: "replicate-observation".to_owned(),
            role: EvidenceArtifactRole::EvaluationOutput,
            media_type: "application/json".to_owned(),
            storage_locator: "cas://campaign/run-01/candidate-a/replicate-0001".to_owned(),
            byte_length: bytes.len() as u64,
            sha256: artifact_sha256,
        };
        let manifest = NoisyReplicateEvidenceManifest::new(
            context().run_id, context().candidate_id, context().source_commit,
            context().decision_sha256, &plan, &substituted_with_digest, artifact,
        ).unwrap();

        assert!(verify_noisy_replicate_artifact_bytes(
            &manifest, &plan, context(), &original, &bytes, "verifier", BUILD,
        ).is_err());
    }

    #[test]
    fn run_source_candidate_decision_replicate_and_plan_substitution_fail_closed() {
        let plan = plan();
        let (manifest, bytes) = manifest_and_bytes(&replicate("0".repeat(64)));
        let mut sample = replicate("0".repeat(64));
        sample.evidence_sha256 = manifest.artifact.sha256.clone();

        let wrong_run = NoisyReplicateEvidenceContext { run_id: "other-run", ..context() };
        assert!(verify_noisy_replicate_artifact_bytes(
            &manifest, &plan, wrong_run, &sample, &bytes, "verifier", BUILD,
        ).is_err());

        let wrong_candidate = NoisyReplicateEvidenceContext { candidate_id: "candidate-b", ..context() };
        assert!(verify_noisy_replicate_artifact_bytes(
            &manifest, &plan, wrong_candidate, &sample, &bytes, "verifier", BUILD,
        ).is_err());

        let mut wrong_replicate = sample.clone();
        wrong_replicate.replicate_id = "replicate-0002".to_owned();
        assert!(verify_noisy_replicate_artifact_bytes(
            &manifest, &plan, context(), &wrong_replicate, &bytes, "verifier", BUILD,
        ).is_err());

        let mut wrong_seed = sample.clone();
        wrong_seed.sample_seed += 1;
        assert!(verify_noisy_replicate_artifact_bytes(
            &manifest, &plan, context(), &wrong_seed, &bytes, "verifier", BUILD,
        ).is_err());

        let mut wrong_plan = plan.clone();
        wrong_plan.noise_model_sha256 = "9999999999999999999999999999999999999999999999999999999999999999".to_owned();
        assert!(verify_noisy_replicate_artifact_bytes(
            &manifest, &wrong_plan, context(), &sample, &bytes, "verifier", BUILD,
        ).is_err());
    }

    #[test]
    fn altered_receipt_and_malformed_artifact_metadata_are_rejected() {
        let plan = plan();
        let (manifest, bytes) = manifest_and_bytes(&replicate("0".repeat(64)));
        let mut sample = replicate("0".repeat(64));
        sample.evidence_sha256 = manifest.artifact.sha256.clone();
        let mut receipt = verify_noisy_replicate_artifact_bytes(
            &manifest, &plan, context(), &sample, &bytes, "verifier", BUILD,
        ).unwrap();
        receipt.objective_vector_sha256 = "9999999999999999999999999999999999999999999999999999999999999999".to_owned();
        assert!(verify_noisy_replicate_artifact_receipt(
            &manifest, &plan, context(), &sample, &bytes, &receipt,
        ).is_err());

        let mut wrong_media = manifest.clone();
        wrong_media.artifact.media_type = "application/octet-stream".to_owned();
        assert!(wrong_media.validate().is_err());

        let mut wrong_role = manifest.clone();
        wrong_role.artifact.role = EvidenceArtifactRole::EvaluationLog;
        assert!(wrong_role.validate().is_err());
    }

    #[test]
    fn noncanonical_json_and_objective_bounds_violations_are_rejected() {
        let plan = plan();
        let sample = replicate("0".repeat(64));
        let mut out_of_bounds = sample.clone();
        out_of_bounds.objective_values[0] = 1.5;
        assert!(encode_noisy_replicate_observation_artifact(
            context(), &plan, &out_of_bounds,
        ).is_err());

        let (manifest, bytes) = manifest_and_bytes(&sample);
        let mut with_whitespace = bytes.clone();
        with_whitespace.push(b'\n');
        let mut whitespace_manifest = manifest.clone();
        whitespace_manifest.artifact.byte_length = with_whitespace.len() as u64;
        whitespace_manifest.artifact.sha256 = noisy_replicate_artifact_sha256(&with_whitespace);
        let mut sample_with_new_digest = sample;
        sample_with_new_digest.evidence_sha256 = whitespace_manifest.artifact.sha256.clone();
        assert!(verify_noisy_replicate_artifact_bytes(
            &whitespace_manifest, &plan, context(), &sample_with_new_digest,
            &with_whitespace, "verifier", BUILD,
        ).is_err());
    }
}
