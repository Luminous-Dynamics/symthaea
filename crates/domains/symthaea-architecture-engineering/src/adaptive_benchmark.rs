//! Adaptive scientific-benchmark task contracts and evidence receipts.
//!
//! This module defines versioned task concepts, recipes, public solver packets, sealed-oracle
//! commitments and benchmark-refinement receipts. It does not implement a secure oracle service.
//! Rust visibility and SHA-256 commitments are not process isolation, secrecy or authentication.
//! Oracle-bearing values must remain in an independent verifier process/service.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

const CONCEPT_SCHEMA: &str = "symthaea.adaptive-benchmark.concept";
const RECIPE_SCHEMA: &str = "symthaea.adaptive-benchmark.recipe";
const PACKET_SCHEMA: &str = "symthaea.adaptive-benchmark.public-task";
const REFINE_SCHEMA: &str = "symthaea.adaptive-benchmark.refinement";
const RECEIPT_SCHEMA: &str = "symthaea.adaptive-benchmark.verifier-receipt";
const VERSION: &str = "0.1.0";
const CONCEPT_DOMAIN: &[u8] = b"symthaea.adaptive-benchmark.concept.v0.1\0";
const RECIPE_DOMAIN: &[u8] = b"symthaea.adaptive-benchmark.recipe.v0.1\0";
const PACKET_DOMAIN: &[u8] = b"symthaea.adaptive-benchmark.packet.v0.1\0";
const ORACLE_DOMAIN: &[u8] = b"symthaea.adaptive-benchmark.oracle.v0.1\0";
const REFINE_DOMAIN: &[u8] = b"symthaea.adaptive-benchmark.refinement.v0.1\0";
const RECEIPT_DOMAIN: &[u8] = b"symthaea.adaptive-benchmark.receipt.v0.1\0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveBenchmarkError {
    pub path: String,
    pub message: String,
}

impl AdaptiveBenchmarkError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }
}

impl std::fmt::Display for AdaptiveBenchmarkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}
impl std::error::Error for AdaptiveBenchmarkError {}

/// Stable scientific construct shared across multiple recipe versions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskConcept {
    pub schema: String,
    pub version: String,
    pub concept_id: String,
    pub scientific_domain: String,
    pub data_modality: String,
    pub reasoning_capability: String,
    pub intervention_type: String,
    pub task_family: String,
}

impl TaskConcept {
    pub fn new(
        concept_id: &str,
        scientific_domain: &str,
        data_modality: &str,
        reasoning_capability: &str,
        intervention_type: &str,
        task_family: &str,
    ) -> Result<Self, AdaptiveBenchmarkError> {
        let result = Self {
            schema: CONCEPT_SCHEMA.to_owned(),
            version: VERSION.to_owned(),
            concept_id: normalize("concept_id", concept_id)?,
            scientific_domain: normalize("scientific_domain", scientific_domain)?,
            data_modality: normalize("data_modality", data_modality)?,
            reasoning_capability: normalize("reasoning_capability", reasoning_capability)?,
            intervention_type: normalize("intervention_type", intervention_type)?,
            task_family: normalize("task_family", task_family)?,
        };
        result.validate()?;
        Ok(result)
    }

    pub fn validate(&self) -> Result<(), AdaptiveBenchmarkError> {
        if self.schema != CONCEPT_SCHEMA || self.version != VERSION {
            return Err(AdaptiveBenchmarkError::new("concept.schema", "unsupported schema or version"));
        }
        for (field, value) in [
            ("concept_id", &self.concept_id),
            ("scientific_domain", &self.scientific_domain),
            ("data_modality", &self.data_modality),
            ("reasoning_capability", &self.reasoning_capability),
            ("intervention_type", &self.intervention_type),
            ("task_family", &self.task_family),
        ] {
            if normalize(field, value)? != *value {
                return Err(AdaptiveBenchmarkError::new(format!("concept.{field}"), "not canonical"));
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, AdaptiveBenchmarkError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|e| AdaptiveBenchmarkError::new("concept", e.to_string()))?;
        Ok(domain_digest(CONCEPT_DOMAIN, &bytes))
    }
}

/// Resource ceilings fixed before a solver run. These are not measured usage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceBudget {
    pub wall_time_ms: u64,
    pub cpu_time_ms: u64,
    pub memory_bytes: u64,
    pub max_tool_calls: u64,
    pub max_candidate_executions: u64,
}
impl ResourceBudget {
    pub fn validate(&self) -> Result<(), AdaptiveBenchmarkError> {
        for (field, value) in [
            ("wall_time_ms", self.wall_time_ms),
            ("cpu_time_ms", self.cpu_time_ms),
            ("memory_bytes", self.memory_bytes),
            ("max_tool_calls", self.max_tool_calls),
            ("max_candidate_executions", self.max_candidate_executions),
        ] {
            if value == 0 {
                return Err(AdaptiveBenchmarkError::new(format!("resource_budget.{field}"), "must be > 0"));
            }
        }
        Ok(())
    }
}

/// Public recipe metadata; it deliberately contains no answer key or private generation seed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRecipe {
    pub schema: String,
    pub version: String,
    pub recipe_id: String,
    pub recipe_version: u64,
    pub concept_sha256: String,
    pub generator_id: String,
    pub generator_sha256: String,
    pub public_template_sha256: String,
    pub environment_sha256: String,
    pub verifier_profile_sha256: String,
    pub resource_budget: ResourceBudget,
}
impl TaskRecipe {
    pub fn validate(&self) -> Result<(), AdaptiveBenchmarkError> {
        if self.schema != RECIPE_SCHEMA || self.version != VERSION || self.recipe_version == 0 {
            return Err(AdaptiveBenchmarkError::new("recipe.schema", "unsupported version or zero recipe revision"));
        }
        if normalize("recipe_id", &self.recipe_id)? != self.recipe_id
            || normalize("generator_id", &self.generator_id)? != self.generator_id {
            return Err(AdaptiveBenchmarkError::new("recipe", "identifiers must be canonical"));
        }
        for (field, digest) in [
            ("concept_sha256", &self.concept_sha256),
            ("generator_sha256", &self.generator_sha256),
            ("public_template_sha256", &self.public_template_sha256),
            ("environment_sha256", &self.environment_sha256),
            ("verifier_profile_sha256", &self.verifier_profile_sha256),
        ] {
            validate_sha256(field, digest)?;
        }
        self.resource_budget.validate()
    }
    pub fn digest(&self) -> Result<String, AdaptiveBenchmarkError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|e| AdaptiveBenchmarkError::new("recipe", e.to_string()))?;
        Ok(domain_digest(RECIPE_DOMAIN, &bytes))
    }
}

/// Exact solver-visible packet. It has no split label, private seed, oracle, grading rubric,
/// shortcut labels or qualification status; keep those out of this schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicTaskPacket {
    pub schema: String,
    pub version: String,
    pub task_id: String,
    pub concept_sha256: String,
    pub recipe_sha256: String,
    pub prompt: String,
    pub input_artifact_sha256: String,
    pub allowed_tools: Vec<String>,
    pub resource_budget: ResourceBudget,
}
impl PublicTaskPacket {
    pub fn new(
        task_id: &str,
        concept: &TaskConcept,
        recipe: &TaskRecipe,
        prompt: &str,
        input_artifact_sha256: &str,
        allowed_tools: Vec<String>,
    ) -> Result<Self, AdaptiveBenchmarkError> {
        concept.validate()?;
        recipe.validate()?;
        if recipe.concept_sha256 != concept.digest()? {
            return Err(AdaptiveBenchmarkError::new("packet.concept_sha256", "recipe is bound to another concept"));
        }
        let mut tools = BTreeSet::new();
        for item in allowed_tools {
            let tool = normalize("allowed_tools", &item)?;
            if !tools.insert(tool) {
                return Err(AdaptiveBenchmarkError::new("packet.allowed_tools", "duplicate canonical tool"));
            }
        }
        let packet = Self {
            schema: PACKET_SCHEMA.to_owned(),
            version: VERSION.to_owned(),
            task_id: normalize("task_id", task_id)?,
            concept_sha256: concept.digest()?,
            recipe_sha256: recipe.digest()?,
            prompt: prompt.to_owned(),
            input_artifact_sha256: input_artifact_sha256.to_owned(),
            allowed_tools: tools.into_iter().collect(),
            resource_budget: recipe.resource_budget.clone(),
        };
        packet.validate()?;
        Ok(packet)
    }
    pub fn validate(&self) -> Result<(), AdaptiveBenchmarkError> {
        if self.schema != PACKET_SCHEMA || self.version != VERSION {
            return Err(AdaptiveBenchmarkError::new("packet.schema", "unsupported schema or version"));
        }
        if self.task_id.trim().is_empty() || self.prompt.trim().is_empty() {
            return Err(AdaptiveBenchmarkError::new("packet", "task ID and prompt are required"));
        }
        for (field, digest) in [
            ("concept_sha256", &self.concept_sha256),
            ("recipe_sha256", &self.recipe_sha256),
            ("input_artifact_sha256", &self.input_artifact_sha256),
        ] {
            validate_sha256(field, digest)?;
        }
        self.resource_budget.validate()?;
        let mut seen = BTreeSet::new();
        for tool in &self.allowed_tools {
            if normalize("allowed_tools", tool)? != *tool || !seen.insert(tool) {
                return Err(AdaptiveBenchmarkError::new("packet.allowed_tools", "tools must be unique and canonical"));
            }
        }
        Ok(())
    }
    pub fn digest(&self) -> Result<String, AdaptiveBenchmarkError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|e| AdaptiveBenchmarkError::new("packet", e.to_string()))?;
        Ok(domain_digest(PACKET_DOMAIN, &bytes))
    }
}

/// Private oracle commitment; this struct must exist only inside the independent verifier service.
///
/// This is not cryptographic access control. The solver must never receive this type or nonce.
pub struct SealedOracleCommitment {
    commitment_sha256: String,
    payload_sha256: String,
    nonce_sha256: String,
}
impl SealedOracleCommitment {
    pub fn commit(payload: &[u8], nonce: &[u8]) -> Result<Self, AdaptiveBenchmarkError> {
        if payload.is_empty() || nonce.len() < 32 {
            return Err(AdaptiveBenchmarkError::new("oracle", "payload required and nonce must be at least 256 bits"));
        }
        let payload_sha256 = sha256(payload);
        let nonce_sha256 = sha256(nonce);
        let commitment_sha256 = domain_digest(
            ORACLE_DOMAIN,
            format!("{payload_sha256}:{nonce_sha256}").as_bytes(),
        );
        Ok(Self { commitment_sha256, payload_sha256, nonce_sha256 })
    }
    pub fn commitment_sha256(&self) -> &str {
        &self.commitment_sha256
    }
    pub fn verify_opening(&self, payload: &[u8], nonce: &[u8]) -> Result<(), AdaptiveBenchmarkError> {
        if sha256(payload) != self.payload_sha256 || sha256(nonce) != self.nonce_sha256 {
            return Err(AdaptiveBenchmarkError::new("oracle", "opening does not match commitment"));
        }
        let expected = domain_digest(
            ORACLE_DOMAIN,
            format!("{}:{}", self.payload_sha256, self.nonce_sha256).as_bytes(),
        );
        if expected != self.commitment_sha256 {
            return Err(AdaptiveBenchmarkError::new("oracle", "commitment reconciliation failed"));
        }
        Ok(())
    }
}

/// Adjudicated record that a hardened recipe still tests the original scientific construct.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeRefinementReceipt {
    pub schema: String,
    pub version: String,
    pub previous_recipe_sha256: String,
    pub revised_recipe_sha256: String,
    pub concept_sha256: String,
    pub shortcut_probe_sha256: String,
    pub adjudication_receipt_sha256: String,
    pub adjudicator_key_id: String,
    pub construct_validity_basis: String,
    pub construct_validity_score_tenths: u8,
    pub same_scientific_construct: bool,
}
impl RecipeRefinementReceipt {
    pub fn validate(
        &self,
        previous: &TaskRecipe,
        revised: &TaskRecipe,
        minimum_validity_score_tenths: u8,
    ) -> Result<(), AdaptiveBenchmarkError> {
        if self.schema != REFINE_SCHEMA || self.version != VERSION {
            return Err(AdaptiveBenchmarkError::new("refinement.schema", "unsupported schema or version"));
        }
        if self.previous_recipe_sha256 != previous.digest()?
            || self.revised_recipe_sha256 != revised.digest()? {
            return Err(AdaptiveBenchmarkError::new("refinement.recipe_digests", "does not bind supplied recipe versions"));
        }
        if previous.concept_sha256 != revised.concept_sha256
            || self.concept_sha256 != previous.concept_sha256
            || !self.same_scientific_construct {
            return Err(AdaptiveBenchmarkError::new("refinement.concept", "construct drift or missing adjudication"));
        }
        validate_sha256("shortcut_probe_sha256", &self.shortcut_probe_sha256)?;
        validate_sha256("adjudication_receipt_sha256", &self.adjudication_receipt_sha256)?;
        if self.adjudicator_key_id.trim().is_empty() || self.construct_validity_basis.trim().is_empty() {
            return Err(AdaptiveBenchmarkError::new("refinement.adjudication", "adjudicator and basis are required"));
        }
        if self.construct_validity_score_tenths > 100 || self.construct_validity_score_tenths < minimum_validity_score_tenths {
            return Err(AdaptiveBenchmarkError::new("refinement.construct_validity_score_tenths", "below frozen threshold"));
        }
        Ok(())
    }
    pub fn digest(&self) -> Result<String, AdaptiveBenchmarkError> {
        let bytes = serde_json::to_vec(self)
            .map_err(|e| AdaptiveBenchmarkError::new("refinement", e.to_string()))?;
        Ok(domain_digest(REFINE_DOMAIN, &bytes))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageEvidence {
    pub stage_id: String,
    pub score_tenths: u16,
    pub evidence_sha256: String,
}

/// Receipt shape validator only. Verify Ed25519 against a trusted key registry separately.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifierReceipt {
    pub schema: String,
    pub version: String,
    pub task_packet_sha256: String,
    pub solver_artifact_sha256: String,
    pub environment_sha256: String,
    pub verifier_profile_sha256: String,
    pub evidence_bundle_sha256: String,
    pub signer_key_id: String,
    pub signature_hex: String,
    pub stages: Vec<StageEvidence>,
    pub wall_time_ms: u64,
    pub cpu_time_ms: u64,
    pub peak_memory_bytes: u64,
    pub tool_calls: u64,
    pub candidate_executions: u64,
}
impl VerifierReceipt {
    pub fn validate_shape(&self, packet: &PublicTaskPacket) -> Result<(), AdaptiveBenchmarkError> {
        if self.schema != RECEIPT_SCHEMA || self.version != VERSION {
            return Err(AdaptiveBenchmarkError::new("receipt.schema", "unsupported schema or version"));
        }
        if self.task_packet_sha256 != packet.digest()? {
            return Err(AdaptiveBenchmarkError::new("receipt.task_packet_sha256", "receipt is bound to another packet"));
        }
        for (field, digest) in [
            ("solver_artifact_sha256", &self.solver_artifact_sha256),
            ("environment_sha256", &self.environment_sha256),
            ("verifier_profile_sha256", &self.verifier_profile_sha256),
            ("evidence_bundle_sha256", &self.evidence_bundle_sha256),
        ] {
            validate_sha256(field, digest)?;
        }
        if self.signer_key_id.trim().is_empty() || self.signature_hex.len() != 128
            || !self.signature_hex.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
            return Err(AdaptiveBenchmarkError::new("receipt.signature", "expected signer ID and 64-byte lowercase-hex signature"));
        }
        if self.stages.is_empty() {
            return Err(AdaptiveBenchmarkError::new("receipt.stages", "at least one stage is required"));
        }
        let mut stage_ids = BTreeSet::new();
        for stage in &self.stages {
            if stage.stage_id.trim().is_empty() || !stage_ids.insert(stage.stage_id.as_str()) || stage.score_tenths > 1000 {
                return Err(AdaptiveBenchmarkError::new("receipt.stages", "stage IDs must be unique/non-empty and scores within 0..=1000"));
            }
            validate_sha256("stage.evidence_sha256", &stage.evidence_sha256)?;
        }
        if self.wall_time_ms == 0 || self.cpu_time_ms == 0 || self.peak_memory_bytes == 0
            || self.wall_time_ms > packet.resource_budget.wall_time_ms
            || self.cpu_time_ms > packet.resource_budget.cpu_time_ms
            || self.peak_memory_bytes > packet.resource_budget.memory_bytes
            || self.tool_calls > packet.resource_budget.max_tool_calls
            || self.candidate_executions > packet.resource_budget.max_candidate_executions {
            return Err(AdaptiveBenchmarkError::new("receipt.resources", "invalid or over-budget resource measurement"));
        }
        Ok(())
    }
    /// Canonical receipt payload with the signature omitted, for signing and verification.
    pub fn signing_payload_bytes(&self) -> Result<Vec<u8>, AdaptiveBenchmarkError> {
        let payload = VerifierReceiptSigningPayload {
            schema: &self.schema,
            version: &self.version,
            task_packet_sha256: &self.task_packet_sha256,
            solver_artifact_sha256: &self.solver_artifact_sha256,
            environment_sha256: &self.environment_sha256,
            verifier_profile_sha256: &self.verifier_profile_sha256,
            evidence_bundle_sha256: &self.evidence_bundle_sha256,
            signer_key_id: &self.signer_key_id,
            stages: &self.stages,
            wall_time_ms: self.wall_time_ms,
            cpu_time_ms: self.cpu_time_ms,
            peak_memory_bytes: self.peak_memory_bytes,
            tool_calls: self.tool_calls,
            candidate_executions: self.candidate_executions,
        };
        serde_json::to_vec(&payload)
            .map_err(|e| AdaptiveBenchmarkError::new("receipt.signing_payload", e.to_string()))
    }

    /// Verify an Ed25519 signature and key validity against an independently managed registry.
    pub fn verify_signature_at(
        &self,
        packet: &PublicTaskPacket,
        registry: &TrustedVerifierKeyRegistry,
        now_unix_ms: u64,
    ) -> Result<(), AdaptiveBenchmarkError> {
        use ed25519_dalek::{Signature, Verifier, VerifyingKey};

        self.validate_shape(packet)?;
        registry.validate()?;
        let key = registry.keys.get(&self.signer_key_id).ok_or_else(|| {
            AdaptiveBenchmarkError::new("receipt.signer_key_id", "signer is not in the trusted registry")
        })?;
        if now_unix_ms < key.not_before_unix_ms
            || key.not_after_unix_ms.is_some_and(|end| now_unix_ms > end)
            || key.revoked_at_unix_ms.is_some_and(|revoked| now_unix_ms >= revoked)
        {
            return Err(AdaptiveBenchmarkError::new(
                "receipt.signer_key_id",
                "signing key is not valid at the supplied verification time",
            ));
        }

        let public_key_bytes = decode_hex::<32>("trusted_key.public_key_hex", &key.public_key_hex)?;
        let verifying_key = VerifyingKey::from_bytes(&public_key_bytes)
            .map_err(|e| AdaptiveBenchmarkError::new("trusted_key.public_key_hex", e.to_string()))?;
        let signature_bytes = decode_hex::<64>("receipt.signature_hex", &self.signature_hex)?;
        let signature = Signature::from_bytes(&signature_bytes);
        let payload = self.signing_payload_bytes()?;
        verifying_key
            .verify(&payload, &signature)
            .map_err(|_| AdaptiveBenchmarkError::new("receipt.signature", "Ed25519 verification failed"))
    }

    pub fn digest(&self) -> Result<String, AdaptiveBenchmarkError> {
        let bytes = serde_json::to_vec(self)
            .map_err(|e| AdaptiveBenchmarkError::new("receipt", e.to_string()))?;
        Ok(domain_digest(RECEIPT_DOMAIN, &bytes))
    }
}

/// Fields covered by a verifier signature; signature_hex is deliberately excluded.
#[derive(Serialize)]
struct VerifierReceiptSigningPayload<'a> {
    schema: &'a str,
    version: &'a str,
    task_packet_sha256: &'a str,
    solver_artifact_sha256: &'a str,
    environment_sha256: &'a str,
    verifier_profile_sha256: &'a str,
    evidence_bundle_sha256: &'a str,
    signer_key_id: &'a str,
    stages: &'a [StageEvidence],
    wall_time_ms: u64,
    cpu_time_ms: u64,
    peak_memory_bytes: u64,
    tool_calls: u64,
    candidate_executions: u64,
}

/// One trusted verifier public key with explicit validity and revocation times.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrustedVerifierKey {
    pub key_id: String,
    pub public_key_hex: String,
    pub not_before_unix_ms: u64,
    pub not_after_unix_ms: Option<u64>,
    pub revoked_at_unix_ms: Option<u64>,
}

/// Pinned verifier key registry. The registry itself must be distributed through a trusted channel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrustedVerifierKeyRegistry {
    pub schema: String,
    pub version: String,
    pub keys: std::collections::BTreeMap<String, TrustedVerifierKey>,
}

impl TrustedVerifierKeyRegistry {
    pub fn validate(&self) -> Result<(), AdaptiveBenchmarkError> {
        if self.schema != "symthaea.adaptive-benchmark.trusted-verifier-keys"
            || self.version != VERSION
        {
            return Err(AdaptiveBenchmarkError::new("key_registry.schema", "unsupported schema or version"));
        }
        for (map_key, key) in &self.keys {
            if key.key_id.trim().is_empty() || map_key != &key.key_id {
                return Err(AdaptiveBenchmarkError::new("key_registry.keys", "map key and key ID must match"));
            }
            decode_hex::<32>("trusted_key.public_key_hex", &key.public_key_hex)?;
            if key.not_after_unix_ms.is_some_and(|end| end < key.not_before_unix_ms) {
                return Err(AdaptiveBenchmarkError::new("trusted_key.validity", "not-after precedes not-before"));
            }
        }
        Ok(())
    }
}

fn decode_hex<const N: usize>(
    field: &str,
    value: &str,
) -> Result<[u8; N], AdaptiveBenchmarkError> {
    if value.len() != N * 2 {
        return Err(AdaptiveBenchmarkError::new(field, format!("expected {} lowercase hex characters", N * 2)));
    }
    let mut output = [0u8; N];
    for (index, byte) in output.iter_mut().enumerate() {
        let start = index * 2;
        *byte = u8::from_str_radix(&value[start..start + 2], 16)
            .map_err(|_| AdaptiveBenchmarkError::new(field, "invalid lowercase hex"))?;
    }
    if value.bytes().any(|b| b.is_ascii_uppercase()) {
        return Err(AdaptiveBenchmarkError::new(field, "uppercase hex is not canonical"));
    }
    Ok(output)
}

fn normalize(field: &str, value: &str) -> Result<String, AdaptiveBenchmarkError> {
    let result = value.split_whitespace().collect::<Vec<_>>().join(" ").to_ascii_lowercase();
    if result.is_empty() {
        Err(AdaptiveBenchmarkError::new(field, "must be non-empty"))
    } else {
        Ok(result)
    }
}
fn validate_sha256(field: &str, value: &str) -> Result<(), AdaptiveBenchmarkError> {
    if value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        Ok(())
    } else {
        Err(AdaptiveBenchmarkError::new(field, "expected lowercase 64-character SHA-256 hex"))
    }
}
fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}
fn domain_digest(domain: &[u8], bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(domain);
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    fn concept() -> TaskConcept {
        TaskConcept::new("causal-001", "Computational Biology", "tabular + time series",
            "causal intervention reasoning", "gene knockout", "invariant mechanism recovery").unwrap()
    }
    fn budget() -> ResourceBudget {
        ResourceBudget { wall_time_ms: 60_000, cpu_time_ms: 30_000, memory_bytes: 2_000_000_000,
            max_tool_calls: 100, max_candidate_executions: 20 }
    }
    fn recipe(rev: u64, template: &str) -> TaskRecipe {
        TaskRecipe { schema: RECIPE_SCHEMA.into(), version: VERSION.into(), recipe_id: "causal-recipe".into(),
            recipe_version: rev, concept_sha256: concept().digest().unwrap(), generator_id: "generator".into(),
            generator_sha256: A.into(), public_template_sha256: template.into(), environment_sha256: B.into(),
            verifier_profile_sha256: C.into(), resource_budget: budget() }
    }
    fn packet() -> PublicTaskPacket {
        PublicTaskPacket::new("task-001", &concept(), &recipe(1, A),
            "Infer the invariant mechanism after the declared intervention.", B,
            vec!["data-reader".into(), "rust-evaluator".into()]).unwrap()
    }

    #[test]
    fn concept_recipe_and_packet_digests_are_stable_and_versioned() {
        let c = concept();
        let normalized = TaskConcept::new(" CAUSAL-001 ", "computational biology", "tabular + time series",
            "causal intervention reasoning", "gene knockout", "invariant mechanism recovery").unwrap();
        assert_eq!(c, normalized);
        assert_eq!(c.digest().unwrap(), normalized.digest().unwrap());
        assert_ne!(recipe(1, A).digest().unwrap(), recipe(2, A).digest().unwrap());
        assert_ne!(packet().digest().unwrap(), PublicTaskPacket::new("task-002", &c, &recipe(1, A),
            "Infer the invariant mechanism after the declared intervention.", B, vec!["data-reader".into(), "rust-evaluator".into()]).unwrap().digest().unwrap());
    }

    #[test]
    fn public_packet_serialization_has_no_oracle_or_split_fields() {
        let value = serde_json::to_value(packet()).unwrap();
        let map = value.as_object().unwrap();
        for forbidden in ["answer", "answer_key", "oracle", "oracle_seed", "qualification_split",
            "benchmark_split", "shortcut_labels", "grading_rubric"] {
            assert!(!map.contains_key(forbidden), "public packet leaked {forbidden}");
        }
    }

    #[test]
    fn recipe_cannot_be_reused_for_another_concept() {
        let other = TaskConcept::new("other", "materials science", "spectroscopy",
            "phase classification", "temperature ramp", "phase transition").unwrap();
        let mut bad = recipe(1, A);
        bad.concept_sha256 = other.digest().unwrap();
        assert!(PublicTaskPacket::new("task", &concept(), &bad, "prompt", B, vec![]).is_err());
    }

    #[test]
    fn sealed_oracle_commitment_rejects_wrong_payload_and_nonce() {
        let seal = SealedOracleCommitment::commit(b"hidden causal graph", &[7u8; 32]).unwrap();
        assert_eq!(seal.commitment_sha256().len(), 64);
        seal.verify_opening(b"hidden causal graph", &[7u8; 32]).unwrap();
        assert!(seal.verify_opening(b"altered causal graph", &[7u8; 32]).is_err());
        assert!(seal.verify_opening(b"hidden causal graph", &[8u8; 32]).is_err());
        assert!(SealedOracleCommitment::commit(b"answer", &[1u8; 16]).is_err());
    }

    #[test]
    fn refinement_requires_same_construct_and_adjudication() {
        let before = recipe(1, A);
        let after = recipe(2, B);
        let receipt = RecipeRefinementReceipt { schema: REFINE_SCHEMA.into(), version: VERSION.into(),
            previous_recipe_sha256: before.digest().unwrap(), revised_recipe_sha256: after.digest().unwrap(),
            concept_sha256: before.concept_sha256.clone(), shortcut_probe_sha256: C.into(),
            adjudication_receipt_sha256: A.into(), adjudicator_key_id: "reviewer-key".into(),
            construct_validity_basis: "independent expert review".into(), construct_validity_score_tenths: 95,
            same_scientific_construct: true };
        receipt.validate(&before, &after, 90).unwrap();
        let mut drifted = receipt.clone();
        drifted.same_scientific_construct = false;
        assert!(drifted.validate(&before, &after, 90).is_err());
        let mut weak = receipt;
        weak.construct_validity_score_tenths = 80;
        assert!(weak.validate(&before, &after, 90).is_err());
    }

    #[test]
    fn verifier_receipt_binds_packet_and_resource_budget() {
        let packet = packet();
        let receipt = VerifierReceipt { schema: RECEIPT_SCHEMA.into(), version: VERSION.into(),
            task_packet_sha256: packet.digest().unwrap(), solver_artifact_sha256: A.into(),
            environment_sha256: B.into(), verifier_profile_sha256: C.into(), evidence_bundle_sha256: A.into(),
            signer_key_id: "verifier-key-1".into(), signature_hex: "ab".repeat(64),
            stages: vec![StageEvidence { stage_id: "execution".into(), score_tenths: 850, evidence_sha256: B.into() }],
            wall_time_ms: 1000, cpu_time_ms: 800, peak_memory_bytes: 10_000, tool_calls: 2, candidate_executions: 1 };
        receipt.validate_shape(&packet).unwrap();
        let mut over = receipt.clone();
        over.wall_time_ms = packet.resource_budget.wall_time_ms + 1;
        assert!(over.validate_shape(&packet).is_err());
        let mut replay = receipt;
        replay.task_packet_sha256 = C.into();
        assert!(replay.validate_shape(&packet).is_err());
    }

    #[test]
    fn verifier_receipt_rejects_duplicate_or_missing_evidence_stages() {
        let packet = packet();
        let mut receipt = VerifierReceipt { schema: RECEIPT_SCHEMA.into(), version: VERSION.into(),
            task_packet_sha256: packet.digest().unwrap(), solver_artifact_sha256: A.into(), environment_sha256: B.into(),
            verifier_profile_sha256: C.into(), evidence_bundle_sha256: A.into(), signer_key_id: "key".into(),
            signature_hex: "ab".repeat(64), stages: vec![
                StageEvidence { stage_id: "planning".into(), score_tenths: 800, evidence_sha256: B.into() },
                StageEvidence { stage_id: "planning".into(), score_tenths: 900, evidence_sha256: C.into() }],
            wall_time_ms: 1000, cpu_time_ms: 800, peak_memory_bytes: 10_000, tool_calls: 1, candidate_executions: 1 };
        assert!(receipt.validate_shape(&packet).is_err());
        receipt.stages.clear();
        assert!(receipt.validate_shape(&packet).is_err());
    }
}
