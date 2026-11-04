//! Read-only MCP adapter over the native REA Evidence verifier.
//!
//! Symthaea in-process consumers should call the typed Rust API directly. This
//! crate is an interoperability adapter for external MCP clients. It accepts
//! inline JSON text only and never reads caller-selected paths or runs providers.

use rmcp::{
    Json, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    schemars::JsonSchema,
    tool, tool_handler, tool_router,
    transport::stdio,
};
use serde::{Deserialize, Serialize};
use symthaea_rea_compat::bundle::{BundleErrorKind, verify_evidence_bundle_json};
use symthaea_rea_compat::{EvidenceErrorKind, sha256_hex, verify_evidence_json};

/// Maximum inline JSON payload accepted by an MCP tool invocation.
pub const MAX_MCP_JSON_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct ClaimCeiling {
    pub producer_authenticity_verified: bool,
    pub observation_truth_verified: bool,
    pub hardware_attestation_verified: bool,
    pub runtime_integrity_verified: bool,
    pub security_compliance_verified: bool,
}

impl ClaimCeiling {
    fn unverified() -> Self {
        Self {
            producer_authenticity_verified: false,
            observation_truth_verified: false,
            hardware_attestation_verified: false,
            runtime_integrity_verified: false,
            security_compliance_verified: false,
        }
    }
}

/// Redacted result shared by both MCP tools.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct VerificationSummary {
    /// verified_semantic_consistency, verified_bundle_consistency, or rejected.
    pub status: String,
    /// Stable machine-readable category; never includes raw parser/provider text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    /// SHA-256 of the exact inline UTF-8 JSON text supplied to the tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_input_sha256: Option<String>,
    /// REA semantic Evidence identifier, only for a verified record.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence_id: Option<String>,
    /// Unprefixed semantic Evidence SHA-256, only for a verified record.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_sha256: Option<String>,
    /// Canonical bundle digest, only for a verified bundle.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_bundle_sha256: Option<String>,
    /// Number of Evidence records in a verified bundle.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub record_count: Option<usize>,
    /// Number of residual-unknown revisions in a verified bundle.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unknown_revision_count: Option<usize>,
    pub claim_ceiling: ClaimCeiling,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VerifyEvidenceInput {
    #[schemars(
        description = "Complete REA Evidence record as inline JSON text. Maximum 8 MiB UTF-8. Pass content, not a file path."
    )]
    pub evidence_json: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VerifyEvidenceBundleInput {
    #[schemars(
        description = "Complete REA EvidenceBundle as inline JSON text. Maximum 8 MiB UTF-8. Pass content, not a file path."
    )]
    pub bundle_json: String,
}

fn rejected(error_code: &str, raw_input_sha256: Option<String>) -> VerificationSummary {
    VerificationSummary {
        status: "rejected".to_owned(),
        error_code: Some(error_code.to_owned()),
        raw_input_sha256,
        evidence_id: None,
        semantic_sha256: None,
        canonical_bundle_sha256: None,
        record_count: None,
        unknown_revision_count: None,
        claim_ceiling: ClaimCeiling::unverified(),
    }
}

fn evidence_error_code(kind: EvidenceErrorKind) -> &'static str {
    match kind {
        EvidenceErrorKind::InvalidJson => "invalid_json",
        EvidenceErrorKind::SchemaViolation => "schema_violation",
        EvidenceErrorKind::Canonicalization => "canonicalization_failed",
        EvidenceErrorKind::IdentityMismatch => "identity_mismatch",
        EvidenceErrorKind::InputTooLarge => "input_too_large",
    }
}

fn bundle_error_code(kind: BundleErrorKind) -> &'static str {
    match kind {
        BundleErrorKind::InvalidJson => "invalid_json",
        BundleErrorKind::InputTooLarge => "input_too_large",
        BundleErrorKind::SchemaViolation => "schema_violation",
        BundleErrorKind::EvidenceInvalid => "embedded_evidence_invalid",
        BundleErrorKind::DuplicateEvidenceId => "duplicate_evidence_id",
        BundleErrorKind::NonCanonicalBundle => "noncanonical_bundle",
        BundleErrorKind::UnknownGraphInvalid => "unknown_graph_invalid",
        BundleErrorKind::Canonicalization => "canonicalization_failed",
    }
}

/// Verify one inline Evidence record. No raw evidence or local path is returned.
pub fn verify_evidence_summary(input: &str) -> VerificationSummary {
    let bytes = input.as_bytes();
    if bytes.len() > MAX_MCP_JSON_BYTES {
        return rejected("input_too_large", None);
    }

    let raw_input_sha256 = sha256_hex(bytes);
    match verify_evidence_json(bytes) {
        Ok(verified) => VerificationSummary {
            status: "verified_semantic_consistency".to_owned(),
            error_code: None,
            raw_input_sha256: Some(raw_input_sha256),
            evidence_id: Some(verified.evidence_id),
            semantic_sha256: Some(verified.semantic_sha256),
            canonical_bundle_sha256: None,
            record_count: None,
            unknown_revision_count: None,
            claim_ceiling: ClaimCeiling::unverified(),
        },
        Err(error) => rejected(evidence_error_code(error.kind), Some(raw_input_sha256)),
    }
}

/// Verify an inline EvidenceBundle and its residual-unknown history graph.
pub fn verify_evidence_bundle_summary(input: &str) -> VerificationSummary {
    let bytes = input.as_bytes();
    if bytes.len() > MAX_MCP_JSON_BYTES {
        return rejected("input_too_large", None);
    }

    let raw_input_sha256 = sha256_hex(bytes);
    match verify_evidence_bundle_json(bytes) {
        Ok(verified) => VerificationSummary {
            status: "verified_bundle_consistency".to_owned(),
            error_code: None,
            raw_input_sha256: Some(raw_input_sha256),
            evidence_id: None,
            semantic_sha256: None,
            canonical_bundle_sha256: Some(verified.canonical_sha256),
            record_count: Some(verified.record_count),
            unknown_revision_count: Some(verified.unknown_revision_count),
            claim_ceiling: ClaimCeiling::unverified(),
        },
        Err(error) => rejected(bundle_error_code(error.kind), Some(raw_input_sha256)),
    }
}

#[derive(Clone)]
pub struct ReaEvidenceMcpServer {
    tool_router: ToolRouter<Self>,
}

#[tool_handler(router = self.tool_router)]
impl rmcp::ServerHandler for ReaEvidenceMcpServer {}

#[tool_router(router = tool_router)]
impl ReaEvidenceMcpServer {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    /// Verify an inline REA Evidence record schema and semantic identity.
    #[tool(
        name = "rea_verify_evidence_json",
        description = "Verify an inline REA Evidence JSON record. Returns only redacted identity digests and a claim ceiling; it does not authenticate the producer or prove the observation occurred.",
        annotations(
            title = "Verify REA Evidence",
            read_only_hint = true,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    pub fn verify_evidence(
        &self,
        Parameters(input): Parameters<VerifyEvidenceInput>,
    ) -> Json<VerificationSummary> {
        Json(verify_evidence_summary(&input.evidence_json))
    }

    /// Verify an inline EvidenceBundle and residual-unknown revision graph.
    #[tool(
        name = "rea_verify_evidence_bundle_json",
        description = "Verify an inline REA EvidenceBundle, embedded Evidence IDs, canonical manifests and residual-unknown graph. Returns only redacted digests/counts and a claim ceiling.",
        annotations(
            title = "Verify REA EvidenceBundle",
            read_only_hint = true,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = false
        )
    )]
    pub fn verify_evidence_bundle(
        &self,
        Parameters(input): Parameters<VerifyEvidenceBundleInput>,
    ) -> Json<VerificationSummary> {
        Json(verify_evidence_bundle_summary(&input.bundle_json))
    }
}

/// Run the stdio-only MCP server. Protocol frames are the only stdout output.
pub async fn serve_stdio() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let service = ReaEvidenceMcpServer::new().serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn evidence_result_contains_only_redacted_semantic_identity() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../symthaea-rea-compat/tests/fixtures/rea-evidence-v1.json"
        ))
        .unwrap();
        let raw = serde_json::to_string(&fixture).unwrap();
        let summary = verify_evidence_summary(&raw);
        let value = serde_json::to_value(summary).unwrap();

        assert_eq!(value["status"], "verified_semantic_consistency");
        assert_eq!(value["evidence_id"], fixture["evidence_id"]);
        assert_eq!(
            value["claim_ceiling"]["producer_authenticity_verified"],
            false
        );
        assert_eq!(value["claim_ceiling"]["observation_truth_verified"], false);
        let rendered = value.to_string();
        assert!(!rendered.contains("/tmp/"));
        assert!(!rendered.contains("classes.dex"));
        assert!(!rendered.contains("normalized_result"));
        assert!(!rendered.contains("raw_result"));
    }

    #[test]
    fn evidence_errors_are_stable_and_do_not_echo_input() {
        let summary = verify_evidence_summary(r#"{"duplicate":1,"duplicate":2}"#);
        assert_eq!(summary.status, "rejected");
        assert_eq!(summary.error_code.as_deref(), Some("invalid_json"));
        let rendered = serde_json::to_string(&summary).unwrap();
        assert!(!rendered.contains("duplicate"));
        assert!(summary.raw_input_sha256.is_some());
    }

    #[test]
    fn mcp_limit_is_lower_than_core_limit() {
        let summary = verify_evidence_summary(&" ".repeat(MAX_MCP_JSON_BYTES + 1));
        assert_eq!(summary.error_code.as_deref(), Some("input_too_large"));
        assert_eq!(summary.raw_input_sha256, None);
    }

    #[test]
    fn bundle_errors_map_to_stable_categories() {
        let summary = verify_evidence_bundle_summary("{}");
        assert_eq!(summary.status, "rejected");
        assert!(matches!(
            summary.error_code.as_deref(),
            Some("schema_violation" | "invalid_json" | "embedded_evidence_invalid")
        ));
    }
}
