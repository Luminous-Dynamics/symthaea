//! Evidence manifests and scenario-run receipts for the living-city workflow.
//!
//! This module validates evidence/receipt structure, produces deterministic
//! canonical bytes, and recomputes SHA-256 digests over retained source/output
//! bytes. It is still a library contract, not by itself a separately deployed
//! independent process: deployment must ensure that this verifier consumes
//! retained artifacts rather than trusting generator-supplied digest strings.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Current schema version for evidence manifests and run receipts.
pub const EVIDENCE_SCHEMA_VERSION: u32 = 1;
pub const RECEIPT_SCHEMA_VERSION: u32 = 1;

/// Provenance class for an evidence item. "Missing" is explicit and never zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    Measured,
    Derived,
    Estimated,
    Simulated,
    Assumed,
    GovernanceDecision,
    Missing,
}

impl EvidenceKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Measured => "measured",
            Self::Derived => "derived",
            Self::Estimated => "estimated",
            Self::Simulated => "simulated",
            Self::Assumed => "assumed",
            Self::GovernanceDecision => "governance_decision",
            Self::Missing => "missing",
        }
    }
}

/// A single versioned source or an explicit record of a missing source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceItem {
    pub id: String,
    pub kind: EvidenceKind,
    pub source_uri: String,
    /// SHA-256 of the retained source artifact, not of this metadata record.
    pub content_sha256: Option<String>,
    /// License, data agreement, legal basis, or an explicit "not applicable".
    pub license_or_basis: String,
    pub units: String,
    pub spatial_scope: String,
    pub temporal_scope: String,
    pub acquired_at_utc: String,
    pub transformation_chain: Vec<String>,
    pub uncertainty_note: String,
    /// Required when kind is Missing; explains what is absent and why it matters.
    pub missing_reason: Option<String>,
}

/// Evidence manifest shared by all scenario outputs that use this input snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceManifest {
    pub schema_version: u32,
    pub project_id: String,
    pub scenario_id: String,
    pub study_boundary_id: String,
    pub coordinate_reference_system: String,
    pub items: Vec<EvidenceItem>,
}

/// A reproducible tool or solver invocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolExecution {
    pub tool_id: String,
    pub version: String,
    /// Digest of the executable/container or build artifact, when applicable.
    pub executable_sha256: Option<String>,
    pub arguments_sha256: String,
    pub status: ToolRunStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolRunStatus {
    Succeeded,
    Failed,
    Blocked,
}

impl ToolRunStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Blocked => "blocked",
        }
    }
}

/// A retained output artifact associated with the run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputArtifact {
    pub id: String,
    pub media_type: String,
    pub sha256: String,
}

/// Explicit result of an individual verification gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateStatus {
    Pass,
    Fail,
    Blocked,
    NotRun,
    NotApplicable,
}

impl GateStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::Blocked => "blocked",
            Self::NotRun => "not_run",
            Self::NotApplicable => "not_applicable",
        }
    }
}

/// A checker result tied to a verifier identity/version and supporting artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationGate {
    pub gate_id: String,
    pub required: bool,
    pub status: GateStatus,
    pub reason: Option<String>,
    /// Required when NotApplicable: an explicit human/governance decision reference.
    pub not_applicable_approval_ref: Option<String>,
    pub verifier_id: Option<String>,
    /// Digest of the retained report/log/evidence supporting this gate.
    pub evidence_sha256: Option<String>,
}

/// Promotion status for the software workflow only, not construction authorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromotionState {
    EligibleForNextWorkflowStage,
    Failed,
    Blocked,
}

impl PromotionState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EligibleForNextWorkflowStage => "eligible_for_next_workflow_stage",
            Self::Failed => "failed",
            Self::Blocked => "blocked",
        }
    }
}

/// Stable receipt for a scenario execution. Gate results are data, not a trusted
/// boolean supplied by the scenario generator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioRunReceipt {
    pub schema_version: u32,
    pub run_id: String,
    pub scenario_id: String,
    pub repository: String,
    pub commit_sha: String,
    pub dependency_lock_sha256: String,
    pub build_profile: String,
    pub toolchain: String,
    pub evidence_manifest_sha256: String,
    pub configuration_sha256: String,
    pub seed: Option<u64>,
    pub tools: Vec<ToolExecution>,
    pub outputs: Vec<OutputArtifact>,
    pub gates: Vec<ValidationGate>,
}

/// Contract validation error. Structural validity does not prove the referenced
/// content is authentic: the independent verifier must recompute all digests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceContractError {
    pub field: &'static str,
    pub reason: &'static str,
}

impl std::fmt::Display for EvidenceContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid evidence contract field {}: {}", self.field, self.reason)
    }
}

impl std::error::Error for EvidenceContractError {}

fn invalid(field: &'static str, reason: &'static str) -> EvidenceContractError {
    EvidenceContractError { field, reason }
}

fn require_non_empty(field: &'static str, value: &str) -> Result<(), EvidenceContractError> {
    if value.trim().is_empty() {
        Err(invalid(field, "must be explicitly provided"))
    } else {
        Ok(())
    }
}

fn is_lower_hex(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_sha256(field: &'static str, value: &str) -> Result<(), EvidenceContractError> {
    if value.len() != 64 || !is_lower_hex(value) {
        return Err(invalid(field, "must be a 64-character lowercase SHA-256 hex digest"));
    }
    Ok(())
}

fn validate_commit_sha(value: &str) -> Result<(), EvidenceContractError> {
    if !matches!(value.len(), 40 | 64) || !is_lower_hex(value) {
        return Err(invalid("commit_sha", "must be a lowercase 40- or 64-character Git object ID"));
    }
    Ok(())
}

fn validate_unique_ids<'a>(
    field: &'static str,
    ids: impl IntoIterator<Item = &'a str>,
) -> Result<(), EvidenceContractError> {
    let mut seen = BTreeSet::new();
    for id in ids {
        require_non_empty(field, id)?;
        if !seen.insert(id) {
            return Err(invalid(field, "IDs must be unique within this collection"));
        }
    }
    Ok(())
}

impl EvidenceManifest {
    pub fn validate(&self) -> Result<(), EvidenceContractError> {
        if self.schema_version != EVIDENCE_SCHEMA_VERSION {
            return Err(invalid("schema_version", "unsupported evidence-manifest schema version"));
        }
        require_non_empty("project_id", &self.project_id)?;
        require_non_empty("scenario_id", &self.scenario_id)?;
        require_non_empty("study_boundary_id", &self.study_boundary_id)?;
        require_non_empty("coordinate_reference_system", &self.coordinate_reference_system)?;
        if self.items.is_empty() {
            return Err(invalid("items", "must include sources or explicit missing-source records"));
        }
        validate_unique_ids("items.id", self.items.iter().map(|item| item.id.as_str()))?;

        for item in &self.items {
            require_non_empty("items.units", &item.units)?;
            require_non_empty("items.spatial_scope", &item.spatial_scope)?;
            require_non_empty("items.temporal_scope", &item.temporal_scope)?;
            require_non_empty("items.uncertainty_note", &item.uncertainty_note)?;
            if item.kind == EvidenceKind::Missing {
                require_non_empty(
                    "items.missing_reason",
                    item.missing_reason.as_deref().unwrap_or(""),
                )?;
                if item.content_sha256.is_some() {
                    return Err(invalid(
                        "items.content_sha256",
                        "a missing source cannot claim a content digest",
                    ));
                }
            } else {
                require_non_empty("items.source_uri", &item.source_uri)?;
                require_non_empty("items.license_or_basis", &item.license_or_basis)?;
                require_non_empty("items.acquired_at_utc", &item.acquired_at_utc)?;
                validate_sha256(
                    "items.content_sha256",
                    item.content_sha256.as_deref().unwrap_or(""),
                )?;
            }
            for transformation in &item.transformation_chain {
                require_non_empty("items.transformation_chain", transformation)?;
            }
        }
        Ok(())
    }

    /// Deterministic length-prefixed canonical representation. Collection order
    /// is normalized by item ID; transformation order is preserved because it
    /// represents the actual processing sequence. Hash these bytes with SHA-256
    /// in the orchestration/verifier layer.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut output = Vec::new();
        append_u64(&mut output, u64::from(self.schema_version));
        append_field(&mut output, &self.project_id);
        append_field(&mut output, &self.scenario_id);
        append_field(&mut output, &self.study_boundary_id);
        append_field(&mut output, &self.coordinate_reference_system);

        let mut items: Vec<&EvidenceItem> = self.items.iter().collect();
        items.sort_by(|a, b| a.id.cmp(&b.id));
        append_u64(&mut output, items.len() as u64);
        for item in items {
            append_field(&mut output, &item.id);
            append_field(&mut output, item.kind.as_str());
            append_field(&mut output, &item.source_uri);
            append_optional_field(&mut output, item.content_sha256.as_deref());
            append_field(&mut output, &item.license_or_basis);
            append_field(&mut output, &item.units);
            append_field(&mut output, &item.spatial_scope);
            append_field(&mut output, &item.temporal_scope);
            append_field(&mut output, &item.acquired_at_utc);
            append_u64(&mut output, item.transformation_chain.len() as u64);
            for transformation in &item.transformation_chain {
                append_field(&mut output, transformation);
            }
            append_field(&mut output, &item.uncertainty_note);
            append_optional_field(&mut output, item.missing_reason.as_deref());
        }
        output
    }
}

impl ScenarioRunReceipt {
    pub fn validate(&self) -> Result<(), EvidenceContractError> {
        if self.schema_version != RECEIPT_SCHEMA_VERSION {
            return Err(invalid("schema_version", "unsupported scenario-receipt schema version"));
        }
        require_non_empty("run_id", &self.run_id)?;
        require_non_empty("scenario_id", &self.scenario_id)?;
        require_non_empty("repository", &self.repository)?;
        validate_commit_sha(&self.commit_sha)?;
        validate_sha256("dependency_lock_sha256", &self.dependency_lock_sha256)?;
        require_non_empty("build_profile", &self.build_profile)?;
        require_non_empty("toolchain", &self.toolchain)?;
        validate_sha256("evidence_manifest_sha256", &self.evidence_manifest_sha256)?;
        validate_sha256("configuration_sha256", &self.configuration_sha256)?;
        if self.tools.is_empty() {
            return Err(invalid("tools", "at least one executed tool must be recorded"));
        }
        validate_unique_ids("tools.tool_id", self.tools.iter().map(|tool| tool.tool_id.as_str()))?;
        for tool in &self.tools {
            require_non_empty("tools.version", &tool.version)?;
            validate_sha256("tools.arguments_sha256", &tool.arguments_sha256)?;
            if let Some(digest) = tool.executable_sha256.as_deref() {
                validate_sha256("tools.executable_sha256", digest)?;
            }
        }
        if self.outputs.is_empty() {
            return Err(invalid("outputs", "at least one retained output artifact is required"));
        }
        validate_unique_ids("outputs.id", self.outputs.iter().map(|output| output.id.as_str()))?;
        for output in &self.outputs {
            require_non_empty("outputs.media_type", &output.media_type)?;
            validate_sha256("outputs.sha256", &output.sha256)?;
        }
        if self.gates.is_empty() {
            return Err(invalid("gates", "at least one verification gate must be recorded"));
        }
        if !self.gates.iter().any(|gate| gate.required) {
            return Err(invalid("gates.required", "at least one gate must be required"));
        }
        validate_unique_ids("gates.gate_id", self.gates.iter().map(|gate| gate.gate_id.as_str()))?;
        for gate in &self.gates {
            match gate.status {
                GateStatus::Pass => {
                    require_non_empty(
                        "gates.verifier_id",
                        gate.verifier_id.as_deref().unwrap_or(""),
                    )?;
                    validate_sha256(
                        "gates.evidence_sha256",
                        gate.evidence_sha256.as_deref().unwrap_or(""),
                    )?;
                }
                GateStatus::NotApplicable => {
                    require_non_empty(
                        "gates.reason",
                        gate.reason.as_deref().unwrap_or(""),
                    )?;
                    require_non_empty(
                        "gates.not_applicable_approval_ref",
                        gate.not_applicable_approval_ref.as_deref().unwrap_or(""),
                    )?;
                }
                GateStatus::Fail | GateStatus::Blocked | GateStatus::NotRun => {
                    require_non_empty(
                        "gates.reason",
                        gate.reason.as_deref().unwrap_or(""),
                    )?;
                }
            }
            if let Some(verifier) = gate.verifier_id.as_deref() {
                require_non_empty("gates.verifier_id", verifier)?;
            }
            if let Some(digest) = gate.evidence_sha256.as_deref() {
                validate_sha256("gates.evidence_sha256", digest)?;
            }
            if gate.status == GateStatus::Pass
                && !self.outputs.iter().any(|output| {
                    gate.evidence_sha256.as_deref() == Some(output.sha256.as_str())
                })
            {
                return Err(invalid(
                    "gates.evidence_sha256",
                    "passing gate evidence digest must reference a retained output artifact",
                ));
            }
        }
        Ok(())
    }

    /// Return the gate state for promotion to the next software-workflow stage.
    /// This is not a code-compliance decision, permit, or construction approval.
    pub fn promotion_state(&self) -> PromotionState {
        // Invalid receipts must never be promoted even if their declared gate
        // statuses happen to look successful.
        if self.validate().is_err() {
            return PromotionState::Blocked;
        }
        if self.tools.iter().any(|tool| tool.status == ToolRunStatus::Failed) {
            return PromotionState::Failed;
        }
        if self.tools.iter().any(|tool| tool.status == ToolRunStatus::Blocked) {
            return PromotionState::Blocked;
        }
        let required = self.gates.iter().filter(|gate| gate.required);
        let mut blocked = false;
        for gate in required {
            match gate.status {
                GateStatus::Pass | GateStatus::NotApplicable => {}
                GateStatus::Fail => return PromotionState::Failed,
                GateStatus::Blocked | GateStatus::NotRun => blocked = true,
            }
        }
        if blocked {
            PromotionState::Blocked
        } else {
            PromotionState::EligibleForNextWorkflowStage
        }
    }

    /// Deterministic length-prefixed canonical representation. Sets of tools,
    /// artifacts, and gates are normalized by ID before serialization.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut output = Vec::new();
        append_u64(&mut output, u64::from(self.schema_version));
        append_field(&mut output, &self.run_id);
        append_field(&mut output, &self.scenario_id);
        append_field(&mut output, &self.repository);
        append_field(&mut output, &self.commit_sha);
        append_field(&mut output, &self.dependency_lock_sha256);
        append_field(&mut output, &self.build_profile);
        append_field(&mut output, &self.toolchain);
        append_field(&mut output, &self.evidence_manifest_sha256);
        append_field(&mut output, &self.configuration_sha256);
        match self.seed {
            Some(seed) => {
                output.push(1);
                append_u64(&mut output, seed);
            }
            None => output.push(0),
        }

        let mut tools: Vec<&ToolExecution> = self.tools.iter().collect();
        tools.sort_by(|a, b| a.tool_id.cmp(&b.tool_id));
        append_u64(&mut output, tools.len() as u64);
        for tool in tools {
            append_field(&mut output, &tool.tool_id);
            append_field(&mut output, &tool.version);
            append_optional_field(&mut output, tool.executable_sha256.as_deref());
            append_field(&mut output, &tool.arguments_sha256);
            append_field(&mut output, tool.status.as_str());
        }

        let mut outputs: Vec<&OutputArtifact> = self.outputs.iter().collect();
        outputs.sort_by(|a, b| a.id.cmp(&b.id));
        append_u64(&mut output, outputs.len() as u64);
        for artifact in outputs {
            append_field(&mut output, &artifact.id);
            append_field(&mut output, &artifact.media_type);
            append_field(&mut output, &artifact.sha256);
        }

        let mut gates: Vec<&ValidationGate> = self.gates.iter().collect();
        gates.sort_by(|a, b| a.gate_id.cmp(&b.gate_id));
        append_u64(&mut output, gates.len() as u64);
        for gate in gates {
            append_field(&mut output, &gate.gate_id);
            output.push(u8::from(gate.required));
            append_field(&mut output, gate.status.as_str());
            append_optional_field(&mut output, gate.reason.as_deref());
            append_optional_field(&mut output, gate.not_applicable_approval_ref.as_deref());
            append_optional_field(&mut output, gate.verifier_id.as_deref());
            append_optional_field(&mut output, gate.evidence_sha256.as_deref());
        }
        output
    }
}

fn append_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn append_field(output: &mut Vec<u8>, value: &str) {
    append_u64(output, value.len() as u64);
    output.extend_from_slice(value.as_bytes());
}

fn append_optional_field(output: &mut Vec<u8>, value: Option<&str>) {
    match value {
        Some(value) => {
            output.push(1);
            append_field(output, value);
        }
        None => output.push(0),
    }
}

/// Compute a lowercase SHA-256 digest over the exact supplied bytes.
///
/// This function hashes bytes, not a path or a claim in a receipt. Callers that
/// verify a retained artifact must read the bytes independently, then compare
/// this result with the manifest/receipt digest.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

/// Stream SHA-256 over large retained artifacts without loading the whole
/// file into memory. Suitable for large BIM, GIS and simulation result files.
pub fn sha256_reader_hex<R: std::io::Read>(mut reader: R) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read as _;

    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compare independently recomputed digests for all available source items.
/// Missing-source records remain explicit and are intentionally excluded from
/// the digest map; whether a gap blocks a specific decision is controlled by its
/// required verification gates.
pub fn verify_manifest_source_digests(
    manifest: &EvidenceManifest,
    actual_sha256: &BTreeMap<String, String>,
) -> Result<(), EvidenceContractError> {
    manifest.validate()?;
    let mut expected_ids = BTreeSet::new();
    for item in &manifest.items {
        if item.kind == EvidenceKind::Missing {
            if actual_sha256.contains_key(&item.id) {
                return Err(invalid(
                    "source_digests",
                    "digest supplied for an item explicitly declared missing",
                ));
            }
            continue;
        }
        expected_ids.insert(item.id.as_str());
        let expected = item
            .content_sha256
            .as_deref()
            .ok_or_else(|| invalid("items.content_sha256", "source digest is required"))?;
        validate_sha256("items.content_sha256", expected)?;
        let actual = actual_sha256.get(&item.id).ok_or_else(|| {
            invalid("source_digests", "recomputed digest missing for a manifest source")
        })?;
        validate_sha256("source_digests", actual)?;
        if actual != expected {
            return Err(invalid(
                "items.content_sha256",
                "recomputed source digest does not match manifest",
            ));
        }
    }
    if actual_sha256.keys().any(|id| !expected_ids.contains(id.as_str())) {
        return Err(invalid("source_digests", "digest supplied for an unlisted source"));
    }
    Ok(())
}

/// Compare independently recomputed digests for every receipt output.
pub fn verify_receipt_output_digests(
    receipt: &ScenarioRunReceipt,
    actual_sha256: &BTreeMap<String, String>,
) -> Result<(), EvidenceContractError> {
    receipt.validate()?;
    let expected_ids: BTreeSet<&str> = receipt.outputs.iter().map(|item| item.id.as_str()).collect();
    for artifact in &receipt.outputs {
        let actual = actual_sha256.get(&artifact.id).ok_or_else(|| {
            invalid("output_digests", "recomputed digest missing for a receipt output")
        })?;
        validate_sha256("output_digests", actual)?;
        if actual != &artifact.sha256 {
            return Err(invalid(
                "outputs.sha256",
                "recomputed output digest does not match receipt",
            ));
        }
    }
    if actual_sha256.keys().any(|id| !expected_ids.contains(id.as_str())) {
        return Err(invalid("output_digests", "digest supplied for an unlisted output"));
    }
    Ok(())
}

/// Validate a manifest, then digest its canonical representation.
pub fn evidence_manifest_sha256(
    manifest: &EvidenceManifest,
) -> Result<String, EvidenceContractError> {
    manifest.validate()?;
    Ok(sha256_hex(&manifest.canonical_bytes()))
}

/// Independently recompute and compare a manifest's canonical digest.
pub fn verify_evidence_manifest_digest(
    manifest: &EvidenceManifest,
    expected_sha256: &str,
) -> Result<(), EvidenceContractError> {
    validate_sha256("evidence_manifest_sha256", expected_sha256)?;
    let actual = evidence_manifest_sha256(manifest)?;
    if actual != expected_sha256 {
        return Err(invalid(
            "evidence_manifest_sha256",
            "recomputed canonical manifest digest does not match the declared digest",
        ));
    }
    Ok(())
}

/// Validate a receipt, then digest its canonical representation.
pub fn scenario_run_receipt_sha256(
    receipt: &ScenarioRunReceipt,
) -> Result<String, EvidenceContractError> {
    receipt.validate()?;
    Ok(sha256_hex(&receipt.canonical_bytes()))
}

/// Independently recompute and compare a receipt's canonical digest.
pub fn verify_scenario_run_receipt_digest(
    receipt: &ScenarioRunReceipt,
    expected_sha256: &str,
) -> Result<(), EvidenceContractError> {
    validate_sha256("scenario_run_receipt_sha256", expected_sha256)?;
    let actual = scenario_run_receipt_sha256(receipt)?;
    if actual != expected_sha256 {
        return Err(invalid(
            "scenario_run_receipt_sha256",
            "recomputed canonical receipt digest does not match the declared digest",
        ));
    }
    Ok(())
}

/// Recompute the digest of retained output bytes instead of trusting a
/// generator-supplied status or hash string.
pub fn verify_output_artifact_bytes(
    artifact: &OutputArtifact,
    retained_bytes: &[u8],
) -> Result<(), EvidenceContractError> {
    validate_sha256("outputs.sha256", &artifact.sha256)?;
    let actual = sha256_hex(retained_bytes);
    if actual != artifact.sha256 {
        return Err(invalid(
            "outputs.sha256",
            "recomputed digest does not match retained artifact bytes",
        ));
    }
    Ok(())
}

/// Verify source bytes against an evidence item's digest. Missing evidence is
/// never verifiable and must remain an explicit blocked information gap.
pub fn verify_evidence_item_bytes(
    item: &EvidenceItem,
    retained_bytes: &[u8],
) -> Result<(), EvidenceContractError> {
    if item.kind == EvidenceKind::Missing {
        return Err(invalid(
            "items.content_sha256",
            "missing evidence cannot be verified as retained source bytes",
        ));
    }
    let expected = item
        .content_sha256
        .as_deref()
        .ok_or_else(|| invalid("items.content_sha256", "source digest is required"))?;
    validate_sha256("items.content_sha256", expected)?;
    if sha256_hex(retained_bytes) != expected {
        return Err(invalid(
            "items.content_sha256",
            "recomputed digest does not match retained source bytes",
        ));
    }
    Ok(())
}

/// Verify all retained source files referenced by a manifest.
///
/// Every non-missing source must be supplied by its manifest item ID and have
/// bytes that match the declared digest. Explicitly missing sources must not be
/// smuggled in under the same ID. Extra/unreferenced bytes are rejected to make
/// accidental source substitution visible.
pub fn verify_manifest_source_bytes(
    manifest: &EvidenceManifest,
    retained_sources: &BTreeMap<String, Vec<u8>>,
) -> Result<(), EvidenceContractError> {
    manifest.validate()?;
    let expected_ids: BTreeSet<&str> = manifest.items.iter().map(|item| item.id.as_str()).collect();

    for item in &manifest.items {
        match item.kind {
            EvidenceKind::Missing => {
                if retained_sources.contains_key(&item.id) {
                    return Err(invalid(
                        "retained_sources",
                        "bytes were supplied for an item explicitly declared missing",
                    ));
                }
            }
            _ => {
                let bytes = retained_sources.get(&item.id).ok_or_else(|| {
                    invalid(
                        "retained_sources",
                        "manifest source bytes are absent from the verification input",
                    )
                })?;
                verify_evidence_item_bytes(item, bytes)?;
            }
        }
    }

    if retained_sources
        .keys()
        .any(|id| !expected_ids.contains(id.as_str()))
    {
        return Err(invalid(
            "retained_sources",
            "unreferenced source bytes are not present in the manifest",
        ));
    }
    Ok(())
}

/// Verify that every receipt output digest matches the corresponding retained
/// bytes, with no missing or extra output artifacts.
pub fn verify_receipt_output_bytes(
    receipt: &ScenarioRunReceipt,
    retained_outputs: &BTreeMap<String, Vec<u8>>,
) -> Result<(), EvidenceContractError> {
    receipt.validate()?;
    let expected_ids: BTreeSet<&str> = receipt.outputs.iter().map(|item| item.id.as_str()).collect();

    for artifact in &receipt.outputs {
        let bytes = retained_outputs.get(&artifact.id).ok_or_else(|| {
            invalid(
                "retained_outputs",
                "receipt output bytes are absent from the verification input",
            )
        })?;
        verify_output_artifact_bytes(artifact, bytes)?;
    }

    if retained_outputs
        .keys()
        .any(|id| !expected_ids.contains(id.as_str()))
    {
        return Err(invalid(
            "retained_outputs",
            "unreferenced output bytes are not present in the receipt",
        ));
    }
    Ok(())
}

/// Bind a receipt to the exact evidence-manifest snapshot and scenario identity.
pub fn verify_receipt_manifest_link(
    receipt: &ScenarioRunReceipt,
    manifest: &EvidenceManifest,
) -> Result<(), EvidenceContractError> {
    receipt.validate()?;
    let actual_manifest_digest = evidence_manifest_sha256(manifest)?;
    if receipt.evidence_manifest_sha256 != actual_manifest_digest {
        return Err(invalid(
            "evidence_manifest_sha256",
            "receipt does not reference this exact canonical evidence manifest",
        ));
    }
    if receipt.scenario_id != manifest.scenario_id {
        return Err(invalid(
            "scenario_id",
            "receipt and evidence manifest refer to different scenarios",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const HASH: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const COMMIT: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn measured_item(id: &str) -> EvidenceItem {
        EvidenceItem {
            id: id.into(),
            kind: EvidenceKind::Measured,
            source_uri: format!("https://example.invalid/{id}"),
            content_sha256: Some(HASH.into()),
            license_or_basis: "synthetic fixture license".into(),
            units: "metres".into(),
            spatial_scope: "synthetic test site".into(),
            temporal_scope: "fixture instant".into(),
            acquired_at_utc: "2026-01-01T00:00:00Z".into(),
            transformation_chain: vec!["fixture-import-v1".into()],
            uncertainty_note: "synthetic fixture; not field data".into(),
            missing_reason: None,
        }
    }

    fn manifest() -> EvidenceManifest {
        EvidenceManifest {
            schema_version: EVIDENCE_SCHEMA_VERSION,
            project_id: "fixture-project".into(),
            scenario_id: "fixture-scenario".into(),
            study_boundary_id: "fixture-boundary".into(),
            coordinate_reference_system: "local synthetic CRS".into(),
            items: vec![measured_item("survey-a"), measured_item("weather-b")],
        }
    }

    fn receipt(status: GateStatus) -> ScenarioRunReceipt {
        ScenarioRunReceipt {
            schema_version: RECEIPT_SCHEMA_VERSION,
            run_id: "run-001".into(),
            scenario_id: "fixture-scenario".into(),
            repository: "Luminous-Dynamics/symthaea".into(),
            commit_sha: COMMIT.into(),
            dependency_lock_sha256: HASH.into(),
            build_profile: "test".into(),
            toolchain: "pinned-test-toolchain".into(),
            evidence_manifest_sha256: HASH.into(),
            configuration_sha256: HASH.into(),
            seed: Some(17),
            tools: vec![ToolExecution {
                tool_id: "synthetic-evaluator".into(),
                version: "0.1.0-fixture".into(),
                executable_sha256: Some(HASH.into()),
                arguments_sha256: HASH.into(),
                status: ToolRunStatus::Succeeded,
            }],
            outputs: vec![OutputArtifact {
                id: "metrics".into(),
                media_type: "application/json".into(),
                sha256: HASH.into(),
            }],
            gates: vec![ValidationGate {
                gate_id: "independent-check".into(),
                required: true,
                status,
                reason: (status != GateStatus::Pass).then(|| "synthetic test gate state".into()),
                not_applicable_approval_ref: None,
                verifier_id: (status == GateStatus::Pass).then(|| "independent-test-verifier/0.1".into()),
                evidence_sha256: (status == GateStatus::Pass).then(|| HASH.into()),
            }],
        }
    }

    #[test]
    fn manifest_canonical_bytes_ignore_source_collection_order() {
        let a = manifest();
        let mut b = a.clone();
        b.items.reverse();
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn missing_evidence_must_be_explicit_and_cannot_claim_a_digest() {
        let mut item = measured_item("missing-groundwater");
        item.kind = EvidenceKind::Missing;
        item.source_uri.clear();
        item.content_sha256 = None;
        item.license_or_basis.clear();
        item.acquired_at_utc.clear();
        item.missing_reason = Some("site investigation not yet available".into());
        let mut input = manifest();
        input.items = vec![item];
        assert!(input.validate().is_ok());

        input.items[0].content_sha256 = Some(HASH.into());
        assert_eq!(
            input.validate().unwrap_err().field,
            "items.content_sha256"
        );
    }

    #[test]
    fn duplicate_evidence_ids_are_rejected() {
        let mut input = manifest();
        input.items.push(measured_item("survey-a"));
        assert_eq!(input.validate().unwrap_err().field, "items.id");
    }

    #[test]
    fn required_not_run_gate_blocks_promotion() {
        let input = receipt(GateStatus::NotRun);
        assert!(input.validate().is_ok());
        assert_eq!(input.promotion_state(), PromotionState::Blocked);
    }

    #[test]
    fn required_failed_gate_fails_promotion() {
        let input = receipt(GateStatus::Fail);
        assert!(input.validate().is_ok());
        assert_eq!(input.promotion_state(), PromotionState::Failed);
    }

    #[test]
    fn passing_gate_requires_verifier_and_evidence_digest() {
        let mut input = receipt(GateStatus::Pass);
        input.gates[0].verifier_id = None;
        assert_eq!(input.validate().unwrap_err().field, "gates.verifier_id");
    }

    #[test]
    fn receipt_canonical_bytes_are_stable() {
        let a = receipt(GateStatus::Pass);
        let b = receipt(GateStatus::Pass);
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn passing_gate_must_reference_a_retained_output_artifact() {
        let mut input = receipt(GateStatus::Pass);
        input.gates[0].evidence_sha256 = Some("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".into());
        assert_eq!(input.validate().unwrap_err().field, "gates.evidence_sha256");
        assert_eq!(input.promotion_state(), PromotionState::Blocked);
    }

    #[test]
    fn not_applicable_gate_requires_approval_reference() {
        let mut input = receipt(GateStatus::NotApplicable);
        input.gates[0].not_applicable_approval_ref = None;
        assert_eq!(
            input.validate().unwrap_err().field,
            "gates.not_applicable_approval_ref"
        );
        input.gates[0].not_applicable_approval_ref = Some("decision-record-42".into());
        assert!(input.validate().is_ok());
        assert_eq!(
            input.promotion_state(),
            PromotionState::EligibleForNextWorkflowStage
        );
    }

    #[test]
    fn invalid_receipt_cannot_be_promoted_even_if_gate_says_pass() {
        let mut input = receipt(GateStatus::Pass);
        input.gates[0].verifier_id = None;
        assert_eq!(input.promotion_state(), PromotionState::Blocked);
    }

    #[test]
    fn receipt_with_no_required_gates_is_rejected() {
        let mut input = receipt(GateStatus::Pass);
        input.gates[0].required = false;
        assert_eq!(input.validate().unwrap_err().field, "gates.required");
        assert_eq!(input.promotion_state(), PromotionState::Blocked);
    }

    #[test]
    fn failed_solver_cannot_be_overridden_by_passing_gate() {
        let mut input = receipt(GateStatus::Pass);
        input.tools[0].status = ToolRunStatus::Failed;
        assert_eq!(input.promotion_state(), PromotionState::Failed);
    }

    #[test]
    fn blocked_solver_blocks_promotion() {
        let mut input = receipt(GateStatus::Pass);
        input.tools[0].status = ToolRunStatus::Blocked;
        assert_eq!(input.promotion_state(), PromotionState::Blocked);
    }

    #[test]
    fn malformed_hashes_fail_closed() {
        let mut input = receipt(GateStatus::Pass);
        input.configuration_sha256 = "not-a-digest".into();
        assert_eq!(input.validate().unwrap_err().field, "configuration_sha256");
    }

    #[test]
    fn sha256_matches_nist_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn manifest_digest_verification_detects_changed_metadata() {
        let original = manifest();
        let digest = evidence_manifest_sha256(&original).unwrap();
        verify_evidence_manifest_digest(&original, &digest).unwrap();

        let mut changed = original.clone();
        changed.items[0].units = "feet".into();
        assert_eq!(
            verify_evidence_manifest_digest(&changed, &digest)
                .unwrap_err()
                .field,
            "evidence_manifest_sha256"
        );
    }

    #[test]
    fn receipt_digest_verification_detects_changed_receipt() {
        let original = receipt(GateStatus::Pass);
        let digest = scenario_run_receipt_sha256(&original).unwrap();
        verify_scenario_run_receipt_digest(&original, &digest).unwrap();

        let mut changed = original.clone();
        changed.commit_sha = "cccccccccccccccccccccccccccccccccccccccc".into();
        assert_eq!(
            verify_scenario_run_receipt_digest(&changed, &digest)
                .unwrap_err()
                .field,
            "scenario_run_receipt_sha256"
        );
    }

    #[test]
    fn retained_output_bytes_are_rehashed_not_trusted() {
        let bytes = b"synthetic model output v1";
        let artifact = OutputArtifact {
            id: "metrics".into(),
            media_type: "application/json".into(),
            sha256: sha256_hex(bytes),
        };
        assert!(verify_output_artifact_bytes(&artifact, bytes).is_ok());
        assert_eq!(
            verify_output_artifact_bytes(&artifact, b"synthetic model output v2")
                .unwrap_err()
                .field,
            "outputs.sha256"
        );
    }

    #[test]
    fn retained_evidence_bytes_are_rehashed_and_missing_sources_block() {
        let bytes = b"synthetic source bytes";
        let mut item = measured_item("survey");
        item.content_sha256 = Some(sha256_hex(bytes));
        assert!(verify_evidence_item_bytes(&item, bytes).is_ok());
        assert!(verify_evidence_item_bytes(&item, b"corrupted source").is_err());

        item.kind = EvidenceKind::Missing;
        item.content_sha256 = None;
        item.missing_reason = Some("not collected".into());
        assert_eq!(
            verify_evidence_item_bytes(&item, bytes).unwrap_err().field,
            "items.content_sha256"
        );
    }

    #[test]
    fn manifest_source_set_detects_corruption_missing_and_extra_bytes() {
        let first = b"survey source bytes";
        let second = b"weather source bytes";
        let mut input = manifest();
        input.items[0].content_sha256 = Some(sha256_hex(first));
        input.items[1].content_sha256 = Some(sha256_hex(second));

        let mut retained = BTreeMap::new();
        retained.insert("survey-a".to_owned(), first.to_vec());
        retained.insert("weather-b".to_owned(), second.to_vec());
        assert!(verify_manifest_source_bytes(&input, &retained).is_ok());

        retained.insert("survey-a".to_owned(), b"tampered survey".to_vec());
        assert_eq!(
            verify_manifest_source_bytes(&input, &retained).unwrap_err().field,
            "items.content_sha256"
        );

        retained.insert("survey-a".to_owned(), first.to_vec());
        retained.remove("weather-b");
        assert_eq!(
            verify_manifest_source_bytes(&input, &retained).unwrap_err().field,
            "retained_sources"
        );

        retained.insert("weather-b".to_owned(), second.to_vec());
        retained.insert("unlisted".to_owned(), b"extra".to_vec());
        assert_eq!(
            verify_manifest_source_bytes(&input, &retained).unwrap_err().field,
            "retained_sources"
        );
    }

    #[test]
    fn receipt_output_set_detects_corruption_missing_and_extra_bytes() {
        let output = b"scenario output bytes";
        let mut input = receipt(GateStatus::Pass);
        input.outputs[0].sha256 = sha256_hex(output);
        let mut retained = BTreeMap::new();
        retained.insert("metrics".to_owned(), output.to_vec());
        assert!(verify_receipt_output_bytes(&input, &retained).is_ok());

        retained.insert("metrics".to_owned(), b"tampered output".to_vec());
        assert_eq!(
            verify_receipt_output_bytes(&input, &retained).unwrap_err().field,
            "outputs.sha256"
        );

        retained.insert("metrics".to_owned(), output.to_vec());
        retained.insert("unlisted".to_owned(), b"extra".to_vec());
        assert_eq!(
            verify_receipt_output_bytes(&input, &retained).unwrap_err().field,
            "retained_outputs"
        );

        retained.remove("metrics");
        retained.remove("unlisted");
        assert_eq!(
            verify_receipt_output_bytes(&input, &retained).unwrap_err().field,
            "retained_outputs"
        );
    }

    #[test]
    fn receipt_must_reference_exact_manifest_and_scenario() {
        let input_manifest = manifest();
        let mut input_receipt = receipt(GateStatus::Pass);
        input_receipt.evidence_manifest_sha256 = evidence_manifest_sha256(&input_manifest).unwrap();
        assert!(verify_receipt_manifest_link(&input_receipt, &input_manifest).is_ok());

        let mut changed_manifest = input_manifest.clone();
        changed_manifest.items[0].units = "feet".into();
        assert_eq!(
            verify_receipt_manifest_link(&input_receipt, &changed_manifest)
                .unwrap_err()
                .field,
            "evidence_manifest_sha256"
        );

        let mut other_scenario = input_manifest.clone();
        other_scenario.scenario_id = "different-scenario".into();
        input_receipt.evidence_manifest_sha256 = evidence_manifest_sha256(&other_scenario).unwrap();
        assert_eq!(
            verify_receipt_manifest_link(&input_receipt, &other_scenario)
                .unwrap_err()
                .field,
            "scenario_id"
        );
    }

    #[test]
    fn streaming_hash_matches_byte_hash_and_nist_vector() {
        let expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert_eq!(sha256_hex(b"abc"), expected);
        assert_eq!(
            sha256_reader_hex(std::io::Cursor::new(b"abc")).unwrap(),
            expected
        );
    }

    #[test]
    fn recomputed_source_digest_map_is_checked_exactly() {
        let first = b"survey";
        let second = b"weather";
        let mut input = manifest();
        input.items[0].content_sha256 = Some(sha256_hex(first));
        input.items[1].content_sha256 = Some(sha256_hex(second));

        let mut actual = BTreeMap::new();
        actual.insert("survey-a".to_owned(), sha256_hex(first));
        actual.insert("weather-b".to_owned(), sha256_hex(second));
        assert!(verify_manifest_source_digests(&input, &actual).is_ok());

        actual.insert("weather-b".to_owned(), sha256_hex(b"corrupted"));
        assert_eq!(
            verify_manifest_source_digests(&input, &actual).unwrap_err().field,
            "items.content_sha256"
        );

        actual.insert("weather-b".to_owned(), sha256_hex(second));
        actual.remove("survey-a");
        assert_eq!(
            verify_manifest_source_digests(&input, &actual).unwrap_err().field,
            "source_digests"
        );

        actual.insert("survey-a".to_owned(), sha256_hex(first));
        actual.insert("unknown".to_owned(), sha256_hex(b"extra"));
        assert_eq!(
            verify_manifest_source_digests(&input, &actual).unwrap_err().field,
            "source_digests"
        );
    }

    #[test]
    fn recomputed_output_digest_map_is_checked_exactly() {
        let output = b"simulated output";
        let mut input = receipt(GateStatus::Pass);
        input.outputs[0].sha256 = sha256_hex(output);
        input.gates[0].evidence_sha256 = Some(sha256_hex(output));
        let mut actual = BTreeMap::new();
        actual.insert("metrics".to_owned(), sha256_hex(output));
        assert!(verify_receipt_output_digests(&input, &actual).is_ok());

        actual.insert("metrics".to_owned(), sha256_hex(b"corrupted"));
        assert_eq!(
            verify_receipt_output_digests(&input, &actual).unwrap_err().field,
            "outputs.sha256"
        );

        actual.insert("metrics".to_owned(), sha256_hex(output));
        actual.insert("extra".to_owned(), sha256_hex(b"not referenced"));
        assert_eq!(
            verify_receipt_output_digests(&input, &actual).unwrap_err().field,
            "output_digests"
        );

        actual.remove("extra");
        actual.remove("metrics");
        assert_eq!(
            verify_receipt_output_digests(&input, &actual).unwrap_err().field,
            "output_digests"
        );
    }

    #[test]
    fn unsupported_schema_versions_are_rejected() {
        let mut input = manifest();
        input.schema_version += 1;
        assert_eq!(input.validate().unwrap_err().field, "schema_version");
    }
}
