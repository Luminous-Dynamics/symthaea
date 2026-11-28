//! Offline structural and semantic verifier for financial research benchmark v1.
//!
//! This crate composes the task/evidence/claim schemas from PR #11233. It does
//! not implement temporal parsing, revision selection, rights proof
//! authentication, a source registry, an artifact store, a financial ledger,
//! or financial authority. A caller must supply temporal dispositions from the
//! shared #11253 kernel through the typed in-memory interface below.
//!
//! V1 evidence manifests do not contain the principal-scoped availability and
//! rights proofs required by #11264. Consequently, a legacy \`known_at\` field
//! alone cannot produce a qualified temporal PASS. The expected v1 temporal
//! outcome is INDETERMINATE until a trusted #11264 adapter supplies proof-bound
//! inputs.

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt;

const MAX_DOCUMENT_BYTES: usize = 16 * 1024 * 1024;
const MAX_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;
const MAX_ARTIFACTS: usize = 4096;

const TASK_SCHEMA: &str = include_str!("../../financial-agent-v1/task.schema.json");
const EVIDENCE_SCHEMA: &str = include_str!("../../financial-agent-v1/evidence.schema.json");
const EVIDENCE_MANIFEST_SCHEMA: &str =
    include_str!("../../financial-agent-v1/evidence-manifest.schema.json");
const CLAIM_AUDIT_SCHEMA: &str =
    include_str!("../../financial-agent-v1/claim-evidence-audit-v1.schema.json");

const EVIDENCE_SCHEMA_ID: &str =
    "https://github.com/Luminous-Dynamics/symthaea/blob/main/benchmarks/financial-agent-v1/evidence.schema.json";

/// Independent disposition for one validation layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Disposition {
    Pass,
    Indeterminate,
    Fail,
}

impl Disposition {
    fn combine(self, other: Self) -> Self {
        self.max(other)
    }
}

/// Validation layer that emitted a finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Gate {
    Schema,
    ManifestIntegrity,
    PayloadIntegrity,
    Temporal,
    Semantic,
    CandidateClaims,
}

/// One deterministic, machine-readable finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub gate: Gate,
    pub disposition: Disposition,
    pub code: String,
    pub path: String,
    pub message: String,
}

/// Counts declared provenance-family IDs per claim.
///
/// These are declared family counts, **not proof of source independence**.
/// An ancestry/independence proof belongs to the source-provenance owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimSourceSummary {
    pub claim_id: String,
    pub declared_distinct_family_count: usize,
}

/// Complete vector report. The layers must not be collapsed into one
/// compensating score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationReport {
    pub schema: Disposition,
    pub manifest_integrity: Disposition,
    pub payload_integrity: Disposition,
    pub temporal: Disposition,
    pub semantic: Disposition,
    pub candidate_claims: Disposition,
    pub claim_sources: Vec<ClaimSourceSummary>,
    pub findings: Vec<Finding>,
}

impl Default for VerificationReport {
    fn default() -> Self {
        Self {
            schema: Disposition::Indeterminate,
            manifest_integrity: Disposition::Indeterminate,
            payload_integrity: Disposition::Indeterminate,
            temporal: Disposition::Indeterminate,
            semantic: Disposition::Indeterminate,
            candidate_claims: Disposition::Indeterminate,
            claim_sources: Vec::new(),
            findings: Vec::new(),
        }
    }
}

impl VerificationReport {
    /// Conservative aggregate for command exit status. The full vector above
    /// remains normative; this helper never lets one good dimension offset a
    /// failure or an unknown in another.
    pub fn overall(&self) -> Disposition {
        [
            self.schema,
            self.manifest_integrity,
            self.payload_integrity,
            self.temporal,
            self.semantic,
            self.candidate_claims,
        ]
        .into_iter()
        .fold(Disposition::Pass, Disposition::combine)
    }

    fn record(
        &mut self,
        gate: Gate,
        disposition: Disposition,
        code: impl Into<String>,
        path: impl Into<String>,
        message: impl Into<String>,
    ) {
        let slot = match gate {
            Gate::Schema => &mut self.schema,
            Gate::ManifestIntegrity => &mut self.manifest_integrity,
            Gate::PayloadIntegrity => &mut self.payload_integrity,
            Gate::Temporal => &mut self.temporal,
            Gate::Semantic => &mut self.semantic,
            Gate::CandidateClaims => &mut self.candidate_claims,
        };
        *slot = slot.combine(disposition);
        self.findings.push(Finding {
            gate,
            disposition,
            code: code.into(),
            path: path.into(),
            message: message.into(),
        });
    }

    fn sort_findings(&mut self) {
        self.findings.sort_by(|left, right| {
            (
                left.gate,
                left.path.as_str(),
                left.code.as_str(),
                left.disposition,
                left.message.as_str(),
            )
                .cmp(&(
                    right.gate,
                    right.path.as_str(),
                    right.code.as_str(),
                    right.disposition,
                    right.message.as_str(),
                ))
        });
        self.claim_sources.sort_by(|left, right| left.claim_id.cmp(&right.claim_id));
    }
}

/// Result of evaluating one exact evidence version with the shared
/// bitemporal kernel. Instances must be created by the trusted adapter, not
/// deserialized from candidate-controlled JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalDisposition {
    Pass,
    Fail,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalEvidenceResult {
    pub evidence_id: String,
    pub disposition: TemporalDisposition,
    pub reason_codes: Vec<String>,
}

/// Typed boundary for the shared #11253 time-frontier result.
///
/// The verifier checks exact ID/frontier/profile binding, but does not
/// authenticate the upstream proof itself. That remains the responsibility of
/// the #11264 schema/proof adapter and Mycelix provenance/rights verifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalEvaluation {
    pub evaluation_mode: String,
    pub decision_frontier: String,
    pub principal_profile_id: Option<String>,
    pub rights_profile_id: Option<String>,
    pub verifier_id: String,
    pub verifier_build_sha256: String,
    pub evidence: Vec<TemporalEvidenceResult>,
}

/// Bounded, caller-supplied resolver for immutable artifact bytes.
///
/// This library never dereferences URLs, filesystem paths, or CAS locators on
/// its own. A resolver must implement its own allowlist, identity binding and
/// byte limits. Returning \`None\` means unavailable and yields INDETERMINATE.
pub trait PayloadResolver: Send + Sync {
    fn resolve(
        &self,
        payload_ref: &str,
        max_bytes: usize,
    ) -> Result<Option<Vec<u8>>, String>;
}

/// Verify one financial research task, exact evidence manifest and claim audit.
///
/// The three JSON inputs are parsed with duplicate-object-key rejection and
/// checked against the committed Draft 2020-12 schemas. The evidence-manifest
/// SHA is calculated over the exact supplied UTF-8 bytes. Inline JSON payloads
/// use RFC 8785 JCS before SHA-256, as specified by PR #11233. External payload
/// bytes are checked only when an explicit bounded resolver is supplied.
///
/// Temporal dispositions must come from the shared #11253 kernel. This
/// function deliberately does not parse or reinterpret \`known_at\`,
/// \`published_at\`, \`retrieved_at\`, revision ancestry or rights.
pub fn verify_financial_research(
    task_bytes: &[u8],
    evidence_manifest_bytes: &[u8],
    claim_audit_bytes: &[u8],
    temporal: &TemporalEvaluation,
    resolver: Option<&dyn PayloadResolver>,
) -> VerificationReport {
    let mut report = VerificationReport::default();

    for (name, bytes) in [
        ("task", task_bytes),
        ("evidence_manifest", evidence_manifest_bytes),
        ("claim_audit", claim_audit_bytes),
    ] {
        if bytes.len() > MAX_DOCUMENT_BYTES {
            report.record(
                Gate::Schema,
                Disposition::Fail,
                "INPUT_SIZE_LIMIT_EXCEEDED",
                format!("$.{name}"),
                format!("document exceeds the {MAX_DOCUMENT_BYTES}-byte verifier limit"),
            );
            report.sort_findings();
            return report;
        }
    }

    let task = match parse_strict_json(task_bytes) {
        Ok(value) => value,
        Err(error) => {
            report.record(Gate::Schema, Disposition::Fail, "TASK_JSON_INVALID", "$", error);
            report.sort_findings();
            return report;
        }
    };
    let manifest = match parse_strict_json(evidence_manifest_bytes) {
        Ok(value) => value,
        Err(error) => {
            report.record(
                Gate::Schema,
                Disposition::Fail,
                "EVIDENCE_MANIFEST_JSON_INVALID",
                "$",
                error,
            );
            report.sort_findings();
            return report;
        }
    };
    let audit = match parse_strict_json(claim_audit_bytes) {
        Ok(value) => value,
        Err(error) => {
            report.record(
                Gate::Schema,
                Disposition::Fail,
                "CLAIM_AUDIT_JSON_INVALID",
                "$",
                error,
            );
            report.sort_findings();
            return report;
        }
    };

    if let Err(error) = validate_schemas(&task, &manifest, &audit) {
        report.record(
            Gate::Schema,
            Disposition::Fail,
            "JSON_SCHEMA_INVALID",
            &error.0,
            error.1,
        );
        report.sort_findings();
        return report;
    }
    report.schema = Disposition::Pass;

    verify_manifest_digest(
        &task,
        evidence_manifest_bytes,
        &mut report,
    );

    let evidence = match manifest.get("artifacts").and_then(Value::as_array) {
        Some(values) => values,
        None => {
            report.record(
                Gate::Semantic,
                Disposition::Fail,
                "EVIDENCE_ARTIFACTS_MISSING",
                "$.artifacts",
                "schema-valid manifest did not expose an artifact array",
            );
            report.sort_findings();
            return report;
        }
    };
    if evidence.len() > MAX_ARTIFACTS {
        report.record(
            Gate::ManifestIntegrity,
            Disposition::Fail,
            "ARTIFACT_COUNT_LIMIT_EXCEEDED",
            "$.artifacts",
            format!("artifact count exceeds {MAX_ARTIFACTS}"),
        );
        report.sort_findings();
        return report;
    }

    verify_payload_integrity(&manifest, evidence, resolver, &mut report);
    let temporal_by_id = verify_temporal_binding(&task, evidence, temporal, &mut report);
    verify_semantic_relations(&task, evidence, &audit, &temporal_by_id, &mut report);

    report.sort_findings();
    report
}

fn validate_schemas(
    task: &Value,
    manifest: &Value,
    audit: &Value,
) -> Result<(), (String, String)> {
    let evidence_schema: Value = serde_json::from_str(EVIDENCE_SCHEMA)
        .map_err(|error| ("$schema.evidence".into(), error.to_string()))?;
    let registry = jsonschema::Registry::new()
        .add(
            EVIDENCE_SCHEMA_ID,
            jsonschema::Resource::from_contents(evidence_schema),
        )
        .map_err(|error| ("$schema.registry".into(), error.to_string()))?
        .prepare()
        .map_err(|error| ("$schema.registry".into(), error.to_string()))?;

    validate_one_schema("task", TASK_SCHEMA, task, &registry)?;
    validate_one_schema("evidence_manifest", EVIDENCE_MANIFEST_SCHEMA, manifest, &registry)?;
    validate_one_schema("claim_audit", CLAIM_AUDIT_SCHEMA, audit, &registry)?;
    Ok(())
}

fn validate_one_schema(
    name: &str,
    schema_text: &str,
    instance: &Value,
    registry: &jsonschema::Registry,
) -> Result<(), (String, String)> {
    let schema: Value = serde_json::from_str(schema_text)
        .map_err(|error| (format!("$schema.{name}"), error.to_string()))?;
    let validator = jsonschema::draft202012::options()
        .offline()
        .with_registry(registry)
        .should_validate_formats(true)
        .build(&schema)
        .map_err(|error| (format!("$schema.{name}"), error.to_string()))?;

    let errors: Vec<String> = validator
        .iter_errors(instance)
        .map(|error| format!("{}: {error}", error.instance_path()))
        .collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err((format!("$.{name}"), errors.join("; ")))
    }
}

fn verify_manifest_digest(
    task: &Value,
    manifest_bytes: &[u8],
    report: &mut VerificationReport,
) {
    let expected = task
        .get("evidence_manifest_sha256")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let actual = sha256_hex(manifest_bytes);
    if expected == actual {
        report.manifest_integrity = Disposition::Pass;
    } else {
        report.record(
            Gate::ManifestIntegrity,
            Disposition::Fail,
            "EVIDENCE_MANIFEST_DIGEST_MISMATCH",
            "$.evidence_manifest_sha256",
            format!("task binds {expected}, exact manifest bytes hash to {actual}"),
        );
    }
}

fn verify_payload_integrity(
    manifest: &Value,
    artifacts: &[Value],
    resolver: Option<&dyn PayloadResolver>,
    report: &mut VerificationReport,
) {
    report.payload_integrity = Disposition::Pass;

    for (index, artifact) in artifacts.iter().enumerate() {
        let path = format!("$.artifacts[{index}]");
        let evidence_id = artifact
            .get("evidence_id")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");
        let payload_ref = artifact
            .get("payload_ref")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let media_type = artifact
            .get("media_type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let expected = artifact
            .get("content_sha256")
            .and_then(Value::as_str)
            .unwrap_or_default();

        let payload_bytes = if let Some(pointer) = payload_ref.strip_prefix("inline:") {
            let expected_prefix = format!("/artifacts/{index}/");
            if !pointer.starts_with(&expected_prefix) {
                report.record(
                    Gate::PayloadIntegrity,
                    Disposition::Fail,
                    "INLINE_PAYLOAD_WRONG_ARTIFACT",
                    format!("{path}.payload_ref"),
                    "inline JSON pointer is not scoped to this artifact's manifest entry",
                );
                continue;
            }
            let Some(payload) = manifest.pointer(pointer) else {
                report.record(
                    Gate::PayloadIntegrity,
                    Disposition::Fail,
                    "INLINE_PAYLOAD_POINTER_UNRESOLVED",
                    format!("{path}.payload_ref"),
                    "inline JSON pointer does not resolve in the exact evidence manifest",
                );
                continue;
            };
            if !is_json_media_type(media_type) {
                report.record(
                    Gate::PayloadIntegrity,
                    Disposition::Indeterminate,
                    "INLINE_NON_JSON_PAYLOAD_UNSUPPORTED",
                    format!("{path}.payload_ref"),
                    "inline JSON pointers are supported only for JSON media types; raw text/binary bytes are required otherwise",
                );
                continue;
            }
            match serde_json_canonicalizer::to_vec(payload) {
                Ok(bytes) => bytes,
                Err(error) => {
                    report.record(
                        Gate::PayloadIntegrity,
                        Disposition::Fail,
                        "JCS_CANONICALIZATION_FAILED",
                        format!("{path}.payload_ref"),
                        error.to_string(),
                    );
                    continue;
                }
            }
        } else {
            let Some(resolver) = resolver else {
                report.record(
                    Gate::PayloadIntegrity,
                    Disposition::Indeterminate,
                    "EXTERNAL_PAYLOAD_UNRESOLVED",
                    format!("{path}.payload_ref"),
                    format!(
                        "no bounded resolver was supplied for external payload {payload_ref:?} ({evidence_id})"
                    ),
                );
                continue;
            };
            let resolved = match resolver.resolve(payload_ref, MAX_PAYLOAD_BYTES) {
                Ok(value) => value,
                Err(error) => {
                    report.record(
                        Gate::PayloadIntegrity,
                        Disposition::Indeterminate,
                        "EXTERNAL_PAYLOAD_RESOLUTION_INDETERMINATE",
                        format!("{path}.payload_ref"),
                        error,
                    );
                    continue;
                }
            };
            let Some(bytes) = resolved else {
                report.record(
                    Gate::PayloadIntegrity,
                    Disposition::Indeterminate,
                    "EXTERNAL_PAYLOAD_UNAVAILABLE",
                    format!("{path}.payload_ref"),
                    "resolver could not provide the exact external payload bytes",
                );
                continue;
            };
            if bytes.len() > MAX_PAYLOAD_BYTES {
                report.record(
                    Gate::PayloadIntegrity,
                    Disposition::Fail,
                    "PAYLOAD_SIZE_LIMIT_EXCEEDED",
                    format!("{path}.payload_ref"),
                    format!("resolved payload exceeds {MAX_PAYLOAD_BYTES} bytes"),
                );
                continue;
            }
            if is_json_media_type(media_type) {
                let value = match parse_strict_json(&bytes) {
                    Ok(value) => value,
                    Err(error) => {
                        report.record(
                            Gate::PayloadIntegrity,
                            Disposition::Fail,
                            "EXTERNAL_JSON_PAYLOAD_INVALID",
                            format!("{path}.payload_ref"),
                            error,
                        );
                        continue;
                    }
                };
                match serde_json_canonicalizer::to_vec(&value) {
                    Ok(canonical) => canonical,
                    Err(error) => {
                        report.record(
                            Gate::PayloadIntegrity,
                            Disposition::Fail,
                            "JCS_CANONICALIZATION_FAILED",
                            format!("{path}.payload_ref"),
                            error.to_string(),
                        );
                        continue;
                    }
                }
            } else {
                bytes
            }
        };

        let actual = sha256_hex(&payload_bytes);
        if actual != expected {
            report.record(
                Gate::PayloadIntegrity,
                Disposition::Fail,
                "EVIDENCE_PAYLOAD_DIGEST_MISMATCH",
                format!("{path}.content_sha256"),
                format!("evidence {evidence_id:?} declares {expected}, resolved/canonical payload hashes to {actual}"),
            );
        }
    }
}

fn is_json_media_type(media_type: &str) -> bool {
    let essence = media_type
        .split(';')
        .next()
        .unwrap_or(media_type)
        .trim()
        .to_ascii_lowercase();
    essence == "application/json" || essence.ends_with("+json")
}

fn verify_temporal_binding(
    task: &Value,
    artifacts: &[Value],
    temporal: &TemporalEvaluation,
    report: &mut VerificationReport,
) -> BTreeMap<String, TemporalDisposition> {
    report.temporal = Disposition::Pass;

    let frontier = task
        .get("decision_at")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if temporal.decision_frontier != frontier {
        report.record(
            Gate::Temporal,
            Disposition::Fail,
            "TEMPORAL_FRONTIER_MISMATCH",
            "$.decision_at",
            "temporal result is not bound to the task's exact decision_at string",
        );
    }
    if temporal.evaluation_mode != "historical_reconstruction"
        && temporal.evaluation_mode != "prospective_streaming"
    {
        report.record(
            Gate::Temporal,
            Disposition::Fail,
            "TEMPORAL_MODE_UNKNOWN",
            "$.temporal.evaluation_mode",
            "evaluation mode must be historical_reconstruction or prospective_streaming",
        );
    }
    if temporal.verifier_id.trim().is_empty() || !is_sha256(&temporal.verifier_build_sha256) {
        report.record(
            Gate::Temporal,
            Disposition::Indeterminate,
            "TEMPORAL_VERIFIER_IDENTITY_INCOMPLETE",
            "$.temporal.verifier",
            "temporal verifier identity and a lowercase SHA-256 build digest are required",
        );
    }
    let expected_rights = task.get("rights_profile").and_then(Value::as_str);
    if temporal.principal_profile_id.as_deref().is_none_or(str::is_empty)
        || temporal.rights_profile_id.as_deref().is_none_or(str::is_empty)
    {
        report.record(
            Gate::Temporal,
            Disposition::Indeterminate,
            "TEMPORAL_PRINCIPAL_OR_RIGHTS_PROOF_MISSING",
            "$.temporal",
            "principal-scoped availability and rights profiles are required; v1 rights_profile alone is not proof",
        );
    } else if expected_rights.is_some_and(|expected| {
        temporal.rights_profile_id.as_deref() != Some(expected)
    }) {
        report.record(
            Gate::Temporal,
            Disposition::Fail,
            "TEMPORAL_RIGHTS_PROFILE_MISMATCH",
            "$.temporal.rights_profile_id",
            "temporal evaluation rights profile does not match the task profile",
        );
    }

    let expected_ids: BTreeSet<String> = artifacts
        .iter()
        .filter_map(|artifact| artifact.get("evidence_id").and_then(Value::as_str))
        .map(str::to_owned)
        .collect();
    let mut by_id = BTreeMap::new();
    for status in &temporal.evidence {
        if by_id.insert(status.evidence_id.clone(), status.disposition).is_some() {
            report.record(
                Gate::Temporal,
                Disposition::Fail,
                "DUPLICATE_TEMPORAL_EVIDENCE_RESULT",
                "$.temporal.evidence",
                format!("duplicate temporal result for {:?}", status.evidence_id),
            );
        }
    }
    for extra in by_id.keys().filter(|id| !expected_ids.contains(*id)) {
        report.record(
            Gate::Temporal,
            Disposition::Fail,
            "UNEXPECTED_TEMPORAL_EVIDENCE_RESULT",
            "$.temporal.evidence",
            format!("temporal result references evidence not present in the task manifest: {extra:?}"),
        );
    }
    for missing in expected_ids.iter().filter(|id| !by_id.contains_key(*id)) {
        report.record(
            Gate::Temporal,
            Disposition::Indeterminate,
            "TEMPORAL_EVIDENCE_RESULT_MISSING",
            "$.temporal.evidence",
            format!("no temporal-kernel result supplied for {missing:?}"),
        );
        by_id.insert(missing.clone(), TemporalDisposition::Indeterminate);
    }

    for status in &temporal.evidence {
        let disposition = match status.disposition {
            TemporalDisposition::Pass => Disposition::Pass,
            TemporalDisposition::Fail => Disposition::Fail,
            TemporalDisposition::Indeterminate => Disposition::Indeterminate,
        };
        for reason in &status.reason_codes {
            report.record(
                Gate::Temporal,
                disposition,
                if reason.trim().is_empty() {
                    "TEMPORAL_KERNEL_REASON_UNSPECIFIED"
                } else {
                    reason
                },
                format!("$.temporal.evidence[{}]", status.evidence_id),
                format!("shared temporal kernel disposition: {:?}", status.disposition),
            );
        }
        if status.reason_codes.is_empty() && status.disposition != TemporalDisposition::Pass {
            report.record(
                Gate::Temporal,
                disposition,
                "TEMPORAL_KERNEL_NONPASS_WITHOUT_REASON",
                format!("$.temporal.evidence[{}]", status.evidence_id),
                format!("shared temporal kernel returned {:?}", status.disposition),
            );
        }
        report.temporal = report.temporal.combine(disposition);
    }
    by_id
}

fn verify_semantic_relations(
    task: &Value,
    artifacts: &[Value],
    audit: &Value,
    temporal_by_id: &BTreeMap<String, TemporalDisposition>,
    report: &mut VerificationReport,
) {
    report.semantic = Disposition::Pass;
    report.candidate_claims = Disposition::Pass;

    let task_id = task.get("task_id").and_then(Value::as_str).unwrap_or_default();
    let audit_task_id = audit.get("task_id").and_then(Value::as_str).unwrap_or_default();
    if task_id != audit_task_id {
        report.record(
            Gate::Semantic,
            Disposition::Fail,
            "TASK_AUDIT_ID_MISMATCH",
            "$.task_id",
            "claim audit is not bound to the exact task ID",
        );
    }
    let decision_at = task.get("decision_at").and_then(Value::as_str).unwrap_or_default();
    let evaluation_frontier = audit
        .get("evaluation_frontier")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if decision_at != evaluation_frontier {
        report.record(
            Gate::Semantic,
            Disposition::Fail,
            "CLAIM_FRONTIER_MISMATCH",
            "$.evaluation_frontier",
            "claim audit frontier must exactly match the task decision_at representation",
        );
    }

    let mut evidence_by_id: BTreeMap<&str, &Value> = BTreeMap::new();
    for (index, artifact) in artifacts.iter().enumerate() {
        let evidence_id = artifact
            .get("evidence_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if evidence_by_id.insert(evidence_id, artifact).is_some() {
            report.record(
                Gate::Semantic,
                Disposition::Fail,
                "DUPLICATE_EVIDENCE_ID",
                format!("$.artifacts[{index}].evidence_id"),
                format!("duplicate evidence ID {evidence_id:?}"),
            );
        }
    }

    let task_ids: BTreeSet<String> = task
        .get("evidence_ids")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    let manifest_ids: BTreeSet<String> = evidence_by_id.keys().map(|id| (*id).to_owned()).collect();
    if task_ids != manifest_ids {
        report.record(
            Gate::Semantic,
            Disposition::Fail,
            "TASK_EVIDENCE_SET_MISMATCH",
            "$.evidence_ids",
            format!(
                "task evidence IDs and manifest artifact IDs differ (task {}, manifest {})",
                task_ids.len(),
                manifest_ids.len()
            ),
        );
    }

    let claims = audit
        .get("claims")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let mut claim_by_id: BTreeMap<&str, &Value> = BTreeMap::new();
    for (index, claim) in claims.iter().enumerate() {
        let claim_id = claim.get("claim_id").and_then(Value::as_str).unwrap_or_default();
        if claim_by_id.insert(claim_id, claim).is_some() {
            report.record(
                Gate::Semantic,
                Disposition::Fail,
                "DUPLICATE_CLAIM_ID",
                format!("$.claims[{index}].claim_id"),
                format!("duplicate claim ID {claim_id:?}"),
            );
        }
    }

    let mut dependency_graph: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (index, claim) in claims.iter().enumerate() {
        let claim_id = claim.get("claim_id").and_then(Value::as_str).unwrap_or_default();
        let claim_path = format!("$.claims[{index}]");
        let claim_type = claim.get("claim_type").and_then(Value::as_str).unwrap_or_default();
        let fact_assessment = claim
            .get("fact_assessment")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let trace_support = claim
            .get("trace_support")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let subject_refs: BTreeSet<&str> = claim
            .get("subject_refs")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();

        let relations = claim
            .get("evidence_relations")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let mut relation_keys = BTreeSet::new();
        let mut related_subjects = BTreeSet::new();
        let mut support_pass = false;
        let mut support_indeterminate = false;
        let mut refute_pass = false;
        let mut refute_indeterminate = false;
        let mut stale_evidence = false;
        let mut declared_families = BTreeSet::new();

        for (relation_index, relation) in relations.iter().enumerate() {
            let relation_path = format!("{claim_path}.evidence_relations[{relation_index}]");
            let evidence_id = relation
                .get("evidence_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let relation_kind = relation.get("relation").and_then(Value::as_str).unwrap_or_default();

            if !relation_keys.insert((evidence_id.to_owned(), relation_kind.to_owned())) {
                report.record(
                    Gate::Semantic,
                    Disposition::Fail,
                    "DUPLICATE_CLAIM_EVIDENCE_RELATION",
                    &relation_path,
                    "the same evidence/relation pair is repeated within one claim",
                );
            }

            let Some(source) = evidence_by_id.get(evidence_id) else {
                report.record(
                    Gate::Semantic,
                    Disposition::Fail,
                    "DANGLING_CLAIM_EVIDENCE_REFERENCE",
                    format!("{relation_path}.evidence_id"),
                    format!("claim references unknown evidence ID {evidence_id:?}"),
                );
                continue;
            };

            let expected_snapshot = source
                .get("content_sha256")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let actual_snapshot = relation
                .get("source_snapshot_sha256")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if expected_snapshot != actual_snapshot {
                report.record(
                    Gate::Semantic,
                    Disposition::Fail,
                    "CLAIM_SOURCE_SNAPSHOT_MISMATCH",
                    format!("{relation_path}.source_snapshot_sha256"),
                    "claim relation does not bind the exact digest declared by the referenced evidence version",
                );
            }

            let expected_family = source
                .get("provenance_family_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            match relation.get("source_family_id").and_then(Value::as_str) {
                Some(actual_family) if actual_family == expected_family => {
                    declared_families.insert(actual_family.to_owned());
                }
                Some(_) => report.record(
                    Gate::Semantic,
                    Disposition::Fail,
                    "SOURCE_FAMILY_IDENTITY_MISMATCH",
                    format!("{relation_path}.source_family_id"),
                    "claim relation source family differs from the referenced manifest artifact",
                ),
                None => report.record(
                    Gate::Semantic,
                    Disposition::Indeterminate,
                    "SOURCE_FAMILY_NOT_BOUND",
                    format!("{relation_path}.source_family_id"),
                    "source-family identity is absent; the document count cannot be treated as independent corroboration",
                ),
            }

            if let Some(subjects) = source.get("subject_refs").and_then(Value::as_array) {
                for subject in subjects.iter().filter_map(Value::as_str) {
                    related_subjects.insert(subject);
                }
            }

            let temporal = temporal_by_id
                .get(evidence_id)
                .copied()
                .unwrap_or(TemporalDisposition::Indeterminate);
            if let Some(claimed_admissible) = relation.get("temporally_admissible").and_then(Value::as_bool) {
                if claimed_admissible != (temporal == TemporalDisposition::Pass) {
                    report.record(
                        Gate::Semantic,
                        Disposition::Fail,
                        "UNTRUSTED_TEMPORAL_BOOLEAN_MISMATCH",
                        format!("{relation_path}.temporally_admissible"),
                        "candidate/audit boolean conflicts with the independently supplied temporal-kernel result; the boolean is never used as the admission source",
                    );
                }
            }

            match (relation_kind, temporal) {
                ("supports", TemporalDisposition::Pass) => support_pass = true,
                ("supports", TemporalDisposition::Indeterminate) => support_indeterminate = true,
                ("supports", TemporalDisposition::Fail) => stale_evidence = true,
                ("refutes" | "contradicts", TemporalDisposition::Pass) => refute_pass = true,
                ("refutes" | "contradicts", TemporalDisposition::Indeterminate) => {
                    refute_indeterminate = true
                }
                ("refutes" | "contradicts", TemporalDisposition::Fail) => stale_evidence = true,
                _ => {}
            }

            if claim_type == "observed"
                && relation_kind == "supports"
                && source.get("evidence_kind").and_then(Value::as_str) == Some("model_output")
            {
                report.record(
                    Gate::CandidateClaims,
                    Disposition::Fail,
                    "MODEL_OUTPUT_USED_AS_OBSERVED_FACT",
                    &relation_path,
                    "a model_output artifact cannot by itself establish an observed real-world fact",
                );
            }
        }

        if !relations.is_empty() {
            for subject in &subject_refs {
                if !related_subjects.contains(subject) {
                    report.record(
                        Gate::Semantic,
                        Disposition::Fail,
                        "CLAIM_SUBJECT_NOT_BOUND_BY_EVIDENCE",
                        format!("{claim_path}.subject_refs"),
                        format!("subject {subject:?} is not present in any referenced evidence artifact"),
                    );
                }
            }
        }
        report.claim_sources.push(ClaimSourceSummary {
            claim_id: claim_id.to_owned(),
            declared_distinct_family_count: declared_families.len(),
        });

        match fact_assessment {
            "supported" if !support_pass => {
                if support_indeterminate {
                    report.record(
                        Gate::CandidateClaims,
                        Disposition::Indeterminate,
                        "SUPPORTED_CLAIM_TEMPORAL_INPUT_INDETERMINATE",
                        &claim_path,
                        "claim is labelled supported but no supporting evidence has a temporal PASS",
                    );
                } else {
                    report.record(
                        Gate::CandidateClaims,
                        Disposition::Fail,
                        "SUPPORTED_CLAIM_LACKS_ADMISSIBLE_SUPPORT",
                        &claim_path,
                        "claim is labelled supported without an admissible supports relation",
                    );
                }
            }
            "contradicted" if !refute_pass => {
                let disposition = if refute_indeterminate {
                    Disposition::Indeterminate
                } else {
                    Disposition::Fail
                };
                report.record(
                    Gate::CandidateClaims,
                    disposition,
                    "CONTRADICTED_CLAIM_LACKS_ADMISSIBLE_REFUTATION",
                    &claim_path,
                    "claim is labelled contradicted without an admissible refutes/contradicts relation",
                );
            }
            "mixed_contested" if !(support_pass && refute_pass) => {
                let disposition = if support_indeterminate || refute_indeterminate {
                    Disposition::Indeterminate
                } else {
                    Disposition::Fail
                };
                report.record(
                    Gate::CandidateClaims,
                    disposition,
                    "MIXED_CLAIM_MISSING_ONE_SIDE",
                    &claim_path,
                    "mixed/contested assessment requires both admissible support and refutation",
                );
            }
            "temporally_inadmissible_stale" if !stale_evidence => {
                report.record(
                    Gate::CandidateClaims,
                    Disposition::Indeterminate,
                    "STALE_ASSESSMENT_NOT_RECONSTRUCTED",
                    &claim_path,
                    "audit labels evidence stale, but the supplied temporal result does not identify a future/stale version",
                );
            }
            _ => {}
        }

        let hard_gate_flags = claim
            .get("hard_gate_flags")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        for flag in hard_gate_flags.iter().filter_map(Value::as_str) {
            report.record(
                Gate::CandidateClaims,
                Disposition::Fail,
                "CANDIDATE_HARD_GATE_FLAG",
                format!("{claim_path}.hard_gate_flags"),
                format!("claim carries non-compensable hard-gate flag {flag:?}"),
            );
        }

        match trace_support {
            "unsupported_in_trace" | "contradicted_in_trace" => report.record(
                Gate::CandidateClaims,
                Disposition::Fail,
                "CLAIM_NOT_SUPPORTED_IN_CANDIDATE_TRACE",
                format!("{claim_path}.trace_support"),
                format!("trace support disposition is {trace_support:?}"),
            ),
            "partially_supported_in_trace" | "not_evaluable" => report.record(
                Gate::CandidateClaims,
                Disposition::Indeterminate,
                "CLAIM_TRACE_SUPPORT_INCOMPLETE",
                format!("{claim_path}.trace_support"),
                format!("trace support disposition is {trace_support:?}"),
            ),
            _ => {}
        }

        if fact_assessment == "not_evaluable" {
            report.record(
                Gate::CandidateClaims,
                Disposition::Indeterminate,
                "CLAIM_NOT_EVALUABLE",
                format!("{claim_path}.fact_assessment"),
                "not_evaluable is preserved as unknown and cannot be promoted to supported",
            );
        }
        if fact_assessment == "insufficient_evidence" {
            report.record(
                Gate::CandidateClaims,
                Disposition::Indeterminate,
                "CLAIM_EVIDENCE_INSUFFICIENT",
                format!("{claim_path}.fact_assessment"),
                "insufficient evidence remains distinct from contradicted or supported",
            );
        }

        if let Some(measurement) = claim.get("measurement").filter(|value| !value.is_null()) {
            let value_is_present = measurement.get("value").is_some_and(|value| !value.is_null());
            let unit = measurement.get("unit").and_then(Value::as_str).unwrap_or_default();
            if value_is_present && unit.trim().is_empty() {
                report.record(
                    Gate::Semantic,
                    Disposition::Fail,
                    "MEASUREMENT_UNIT_MISSING",
                    format!("{claim_path}.measurement.unit"),
                    "a material measurement value requires an explicit unit",
                );
            }
            if measurement.get("currency").and_then(Value::as_str).is_some()
                && measurement.get("value").is_some_and(Value::is_number)
            {
                report.record(
                    Gate::Semantic,
                    Disposition::Indeterminate,
                    "CURRENCY_VALUE_REQUIRES_EXACT_DECIMAL_WIRE",
                    format!("{claim_path}.measurement.value"),
                    "financial numeric values should use an exact decimal string/wire representation, not a binary floating-point JSON number",
                );
            }
            if unit == "percent"
                && measurement
                    .get("denominator")
                    .and_then(Value::as_str)
                    .is_none_or(str::is_empty)
            {
                report.record(
                    Gate::Semantic,
                    Disposition::Indeterminate,
                    "PERCENT_DENOMINATOR_UNSPECIFIED",
                    format!("{claim_path}.measurement.denominator"),
                    "percentage value has no declared denominator",
                );
            }
        }

        let mut input_ids = Vec::new();
        if let Some(derivation) = claim.get("derivation").filter(|value| !value.is_null()) {
            if let Some(ids) = derivation.get("input_claim_ids").and_then(Value::as_array) {
                for input in ids.iter().filter_map(Value::as_str) {
                    if !claim_by_id.contains_key(input) {
                        report.record(
                            Gate::Semantic,
                            Disposition::Fail,
                            "DERIVATION_INPUT_CLAIM_DANGLING",
                            format!("{claim_path}.derivation.input_claim_ids"),
                            format!("derivation references unknown claim {input:?}"),
                        );
                    }
                    input_ids.push(input.to_owned());
                }
            }
            if derivation.get("reproducible").and_then(Value::as_bool) == Some(false)
                && fact_assessment == "supported"
            {
                report.record(
                    Gate::CandidateClaims,
                    Disposition::Fail,
                    "NONREPRODUCIBLE_DERIVATION_PRESENTED_AS_SUPPORTED",
                    format!("{claim_path}.derivation.reproducible"),
                    "a derived claim with a non-reproducible calculation cannot qualify as supported",
                );
            }
        } else if claim_type == "derived" && fact_assessment == "supported" {
            report.record(
                Gate::CandidateClaims,
                Disposition::Fail,
                "DERIVED_CLAIM_MISSING_LINEAGE",
                format!("{claim_path}.derivation"),
                "a supported derived claim requires calculation inputs and derivation lineage",
            );
        }
        dependency_graph.insert(claim_id.to_owned(), input_ids);
    }

    for (claim_id, inputs) in &dependency_graph {
        if inputs.iter().any(|input| input == claim_id) {
            report.record(
                Gate::Semantic,
                Disposition::Fail,
                "DERIVATION_SELF_REFERENCE",
                format!("$.claims[{claim_id}].derivation.input_claim_ids"),
                "a claim cannot derive from itself",
            );
        }
    }
    if let Some(cycle) = find_claim_cycle(&dependency_graph) {
        report.record(
            Gate::Semantic,
            Disposition::Fail,
            "DERIVATION_CYCLE",
            "$.claims",
            format!("claim derivation graph contains a cycle: {}", cycle.join(" -> ")),
        );
    }
}

fn find_claim_cycle(graph: &BTreeMap<String, Vec<String>>) -> Option<Vec<String>> {
    fn visit(
        node: &str,
        graph: &BTreeMap<String, Vec<String>>,
        visiting: &mut Vec<String>,
        visited: &mut BTreeSet<String>,
    ) -> Option<Vec<String>> {
        if let Some(position) = visiting.iter().position(|value| value == node) {
            let mut cycle = visiting[position..].to_vec();
            cycle.push(node.to_owned());
            return Some(cycle);
        }
        if visited.contains(node) {
            return None;
        }
        visiting.push(node.to_owned());
        if let Some(children) = graph.get(node) {
            for child in children {
                if let Some(cycle) = visit(child, graph, visiting, visited) {
                    return Some(cycle);
                }
            }
        }
        visiting.pop();
        visited.insert(node.to_owned());
        None
    }

    let mut visiting = Vec::new();
    let mut visited = BTreeSet::new();
    for node in graph.keys() {
        if let Some(cycle) = visit(node, graph, &mut visiting, &mut visited) {
            return Some(cycle);
        }
    }
    None
}

/// Strict JSON parsing that rejects duplicate property names before converting
/// to serde_json::Value. serde_json's ordinary Value parser keeps the last
/// duplicate key; doing so would erase an ambiguity before a digest check.
fn parse_strict_json(bytes: &[u8]) -> Result<Value, String> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let parsed = StrictValue::deserialize(&mut deserializer).map_err(|error| error.to_string())?;
    deserializer.end().map_err(|error| error.to_string())?;
    Ok(parsed.into_value())
}

enum StrictValue {
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<StrictValue>),
    Object(Map<String, Value>),
}

impl StrictValue {
    fn into_value(self) -> Value {
        match self {
            Self::Null => Value::Null,
            Self::Bool(value) => Value::Bool(value),
            Self::Number(value) => Value::Number(value),
            Self::String(value) => Value::String(value),
            Self::Array(values) => {
                Value::Array(values.into_iter().map(StrictValue::into_value).collect())
            }
            Self::Object(values) => Value::Object(values),
        }
    }
}

struct StrictValueVisitor;

impl<'de> Visitor<'de> for StrictValueVisitor {
    type Value = StrictValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without duplicate object property names")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(StrictValue::Null)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(StrictValue::Null)
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(StrictValue::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        const MAX_SAFE: i64 = 9_007_199_254_740_991;
        if !(-MAX_SAFE..=MAX_SAFE).contains(&value) {
            return Err(E::custom("integer outside the RFC 8785 safe range; encode it as a string"));
        }
        Ok(StrictValue::Number(Number::from(value)))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        const MAX_SAFE: u64 = 9_007_199_254_740_991;
        if value > MAX_SAFE {
            return Err(E::custom("integer outside the RFC 8785 safe range; encode it as a string"));
        }
        Ok(StrictValue::Number(Number::from(value)))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Number::from_f64(value)
            .map(StrictValue::Number)
            .ok_or_else(|| E::custom("non-finite JSON number is not permitted"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(StrictValue::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(StrictValue::String(value))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<StrictValue>()? {
            values.push(value);
        }
        Ok(StrictValue::Array(values))
    }

    fn visit_map<A>(self, mut mapping: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        let mut seen = HashSet::new();
        while let Some(key) = mapping.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate JSON object property {key:?}")));
            }
            let value = mapping.next_value::<StrictValue>()?.into_value();
            values.insert(key, value);
        }
        Ok(StrictValue::Object(values))
    }
}

impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictValueVisitor)
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TASK: &[u8] = include_bytes!("../../financial-agent-v1/dev-example.task.json");
    const MANIFEST: &[u8] = include_bytes!("../../financial-agent-v1/dev-example.evidence.json");
    const AUDIT: &[u8] = include_bytes!("../../financial-agent-v1/dev-example.claim-audit.json");

    fn temporal(disposition: TemporalDisposition) -> TemporalEvaluation {
        let manifest: Value = serde_json::from_slice(MANIFEST).unwrap();
        let task: Value = serde_json::from_slice(TASK).unwrap();
        let evidence = manifest["artifacts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|artifact| TemporalEvidenceResult {
                evidence_id: artifact["evidence_id"].as_str().unwrap().to_owned(),
                disposition,
                reason_codes: if disposition == TemporalDisposition::Pass {
                    Vec::new()
                } else {
                    vec!["V1_PROOF_REFERENCE_MISSING".into()]
                },
            })
            .collect();

        TemporalEvaluation {
            evaluation_mode: "historical_reconstruction".into(),
            decision_frontier: task["decision_at"].as_str().unwrap().into(),
            principal_profile_id: Some("synthetic-eval-principal".into()),
            rights_profile_id: task["rights_profile"].as_str().map(str::to_owned),
            verifier_id: "financial-time-frontier-adapter-test".into(),
            verifier_build_sha256: "ab".repeat(32),
            evidence,
        }
    }

    fn report(audit: &[u8], temporal: &TemporalEvaluation) -> VerificationReport {
        verify_financial_research(TASK, MANIFEST, audit, temporal, None)
    }

    #[test]
    fn committed_development_example_has_valid_schemas_and_payloads_but_not_a_qualified_candidate() {
        let result = report(AUDIT, &temporal(TemporalDisposition::Indeterminate));

        assert_eq!(result.schema, Disposition::Pass);
        assert_eq!(result.manifest_integrity, Disposition::Pass);
        assert_eq!(result.payload_integrity, Disposition::Pass);
        assert_eq!(result.temporal, Disposition::Indeterminate);
        assert_eq!(result.semantic, Disposition::Fail); // The authored claim snapshot hashes are deliberately wrong.
        assert_eq!(result.candidate_claims, Disposition::Fail); // The fixture contains explicit hard-gate probes.
        assert!(result
            .findings
            .iter()
            .any(|finding| finding.code == "CLAIM_SOURCE_SNAPSHOT_MISMATCH"));
        assert!(result
            .findings
            .iter()
            .any(|finding| finding.code == "CANDIDATE_HARD_GATE_FLAG"));
    }

    #[test]
    fn supported_claims_pass_only_with_matching_source_digest_and_temporal_pass() {
        let mut audit: Value = serde_json::from_slice(AUDIT).unwrap();
        let manifest: Value = serde_json::from_slice(MANIFEST).unwrap();
        let artifacts = manifest["artifacts"].as_array().unwrap();

        let mut claims = audit["claims"].as_array().unwrap().clone();
        claims.retain(|claim| claim["claim_id"] == "C-001" || claim["claim_id"] == "C-003");
        for claim in &mut claims {
            for relation in claim["evidence_relations"].as_array_mut().unwrap() {
                let id = relation["evidence_id"].as_str().unwrap();
                let artifact = artifacts
                    .iter()
                    .find(|artifact| artifact["evidence_id"] == id)
                    .unwrap();
                relation["source_snapshot_sha256"] = artifact["content_sha256"].clone();
                relation["temporally_admissible"] = Value::Bool(true);
            }
        }
        audit["claims"] = Value::Array(claims);
        let audit_bytes = serde_json::to_vec(&audit).unwrap();

        let result = report(&audit_bytes, &temporal(TemporalDisposition::Pass));
        assert_eq!(result.schema, Disposition::Pass);
        assert_eq!(result.manifest_integrity, Disposition::Pass);
        assert_eq!(result.payload_integrity, Disposition::Pass);
        assert_eq!(result.temporal, Disposition::Pass);
        assert_eq!(result.semantic, Disposition::Pass);
        assert_eq!(result.candidate_claims, Disposition::Pass);
    }

    #[test]
    fn task_manifest_digest_is_bound_to_exact_bytes() {
        let mut task: Value = serde_json::from_slice(TASK).unwrap();
        task["evidence_manifest_sha256"] = Value::String("00".repeat(32));
        let task_bytes = serde_json::to_vec(&task).unwrap();
        let result = verify_financial_research(
            &task_bytes,
            MANIFEST,
            AUDIT,
            &temporal(TemporalDisposition::Indeterminate),
            None,
        );

        assert_eq!(result.schema, Disposition::Pass);
        assert_eq!(result.manifest_integrity, Disposition::Fail);
        assert!(result
            .findings
            .iter()
            .any(|finding| finding.code == "EVIDENCE_MANIFEST_DIGEST_MISMATCH"));
    }

    #[test]
    fn task_and_manifest_evidence_id_sets_must_match() {
        let mut task: Value = serde_json::from_slice(TASK).unwrap();
        task["evidence_ids"] = serde_json::json!(["ev.synthetic.supplier-status.0001"]);
        let task_bytes = serde_json::to_vec(&task).unwrap();
        let result = verify_financial_research(
            &task_bytes,
            MANIFEST,
            AUDIT,
            &temporal(TemporalDisposition::Indeterminate),
            None,
        );

        assert_eq!(result.semantic, Disposition::Fail);
        assert!(result
            .findings
            .iter()
            .any(|finding| finding.code == "TASK_EVIDENCE_SET_MISMATCH"));
    }

    #[test]
    fn duplicate_json_keys_are_rejected_before_schema_or_digest_processing() {
        let result = verify_financial_research(
            br#"{"schema_version":"luminous.financial-research-task.v1","task_id":"x","task_id":"y"}"#,
            MANIFEST,
            AUDIT,
            &temporal(TemporalDisposition::Indeterminate),
            None,
        );
        assert_eq!(result.schema, Disposition::Fail);
        assert!(result
            .findings
            .iter()
            .any(|finding| finding.code == "TASK_JSON_INVALID"));
    }

    #[test]
    fn temporal_boolean_in_claim_audit_is_not_an_admission_source() {
        let mut audit: Value = serde_json::from_slice(AUDIT).unwrap();
        let mut claims = audit["claims"].as_array().unwrap().clone();
        claims.truncate(1);
        audit["claims"] = Value::Array(claims);
        let bytes = serde_json::to_vec(&audit).unwrap();

        let result = report(&bytes, &temporal(TemporalDisposition::Indeterminate));
        assert!(result
            .findings
            .iter()
            .any(|finding| finding.code == "UNTRUSTED_TEMPORAL_BOOLEAN_MISMATCH"));
        assert_ne!(result.candidate_claims, Disposition::Pass);
    }

    #[test]
    fn source_family_substitution_is_rejected() {
        let mut audit: Value = serde_json::from_slice(AUDIT).unwrap();
        let mut claims = audit["claims"].as_array().unwrap().clone();
        claims.truncate(1);
        claims[0]["evidence_relations"][0]["source_family_id"] =
            Value::String("synthetic-family.forged.v1".into());
        audit["claims"] = Value::Array(claims);
        let result = report(
            &serde_json::to_vec(&audit).unwrap(),
            &temporal(TemporalDisposition::Indeterminate),
        );

        assert_eq!(result.semantic, Disposition::Fail);
        assert!(result
            .findings
            .iter()
            .any(|finding| finding.code == "SOURCE_FAMILY_IDENTITY_MISMATCH"));
    }

    #[test]
    fn derived_claim_dangling_input_is_rejected() {
        let mut audit: Value = serde_json::from_slice(AUDIT).unwrap();
        let claim = &mut audit["claims"][4];
        claim["derivation"]["input_claim_ids"] = serde_json::json!(["C-DOES-NOT-EXIST"]);
        let result = report(
            &serde_json::to_vec(&audit).unwrap(),
            &temporal(TemporalDisposition::Indeterminate),
        );

        assert_eq!(result.semantic, Disposition::Fail);
        assert!(result
            .findings
            .iter()
            .any(|finding| finding.code == "DERIVATION_INPUT_CLAIM_DANGLING"));
    }

    #[test]
    fn derived_claim_cycles_are_rejected() {
        let mut audit: Value = serde_json::from_slice(AUDIT).unwrap();
        let mut first = audit["claims"][0].clone();
        let mut second = audit["claims"][2].clone();
        first["claim_id"] = Value::String("CYCLE-A".into());
        second["claim_id"] = Value::String("CYCLE-B".into());
        first["claim_type"] = Value::String("derived".into());
        second["claim_type"] = Value::String("derived".into());
        first["derivation"] = serde_json::json!({"input_claim_ids":["CYCLE-B"]});
        second["derivation"] = serde_json::json!({"input_claim_ids":["CYCLE-A"]});
        first["hard_gate_flags"] = serde_json::json!([]);
        second["hard_gate_flags"] = serde_json::json!([]);
        first["fact_assessment"] = Value::String("insufficient_evidence".into());
        second["fact_assessment"] = Value::String("insufficient_evidence".into());
        audit["claims"] = serde_json::json!([first, second]);
        let result = report(
            &serde_json::to_vec(&audit).unwrap(),
            &temporal(TemporalDisposition::Indeterminate),
        );

        assert_eq!(result.semantic, Disposition::Fail);
        assert!(result
            .findings
            .iter()
            .any(|finding| finding.code == "DERIVATION_CYCLE"));
    }

    #[test]
    fn missing_external_bytes_remain_indeterminate_without_network_fallback() {
        let mut manifest: Value = serde_json::from_slice(MANIFEST).unwrap();
        manifest["artifacts"][0]["payload_ref"] =
            Value::String("cas://immutable/artifacts/supplier-v1".into());
        let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        let mut task: Value = serde_json::from_slice(TASK).unwrap();
        task["evidence_manifest_sha256"] = Value::String(sha256_hex(&manifest_bytes));
        let task_bytes = serde_json::to_vec(&task).unwrap();

        let result = verify_financial_research(
            &task_bytes,
            &manifest_bytes,
            AUDIT,
            &temporal(TemporalDisposition::Indeterminate),
            None,
        );
        assert_eq!(result.payload_integrity, Disposition::Indeterminate);
        assert!(result
            .findings
            .iter()
            .any(|finding| finding.code == "EXTERNAL_PAYLOAD_UNRESOLVED"));
    }

    #[test]
    fn unsupported_schema_properties_are_rejected() {
        let mut task: Value = serde_json::from_slice(TASK).unwrap();
        task["invented_authority"] = Value::Bool(true);
        let task_bytes = serde_json::to_vec(&task).unwrap();
        let result = verify_financial_research(
            &task_bytes,
            MANIFEST,
            AUDIT,
            &temporal(TemporalDisposition::Indeterminate),
            None,
        );
        assert_eq!(result.schema, Disposition::Fail);
        assert!(result
            .findings
            .iter()
            .any(|finding| finding.code == "JSON_SCHEMA_INVALID"));
    }

    #[test]
    fn source_family_count_is_not_misreported_as_independence_proof() {
        let mut audit: Value = serde_json::from_slice(AUDIT).unwrap();
        let mut claims = audit["claims"].as_array().unwrap().clone();
        claims.truncate(1);
        claims[0]["evidence_relations"][0]["source_snapshot_sha256"] = Value::String(
            "00267da342f0e0901070c16ceb5b0daeda51172cc3fd23dd7a60d99179501adc".into(),
        );
        audit["claims"] = Value::Array(claims);
        let result = report(
            &serde_json::to_vec(&audit).unwrap(),
            &temporal(TemporalDisposition::Pass),
        );
        assert_eq!(result.claim_sources[0].declared_distinct_family_count, 1);
        assert_eq!(result.semantic, Disposition::Pass);
    }
}
