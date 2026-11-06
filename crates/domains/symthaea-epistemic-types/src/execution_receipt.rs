//! Pure, read-only interpretation of Mycelix EPI-013R tool execution receipts.
//!
//! This module interprets an exact execution observation and a caller-pinned
//! EPI-013 profile corpus. It does not execute tools, authorize collection,
//! admit evidence, establish source authenticity, or assert factual truth.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const RECEIPT_SCHEMA_VERSION: &str = "mycelix:epi-013r-tool-execution-receipt:v0.1";
pub const RECEIPT_AUTHORITY: &str = "ExecutionObservationOnly";
pub const PROFILE_CORPUS_SCHEMA: &str = "mycelix:epi-013-tool-capability-corpus:v0.1";
pub const PROFILE_CORPUS_SHA256: &str =
    "d499b680d6461dbffed9b4ea053d3758ee4bd4e7d3e26658d26334acddc35cf7";
pub const RECEIPT_CORPUS_SHA256: &str =
    "48ed00a275b8585666db8426a61e1330d44223196e50fb1017d58e0acc147d9b";
pub const INTERPRETATION_SCHEMA_VERSION: &str = "symthaea:execution-evidence-interpretation:v0.1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum ReceiptAuthority {
    ExecutionObservationOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolExecutionReceiptV1 {
    pub schema_version: String,
    pub receipt_id: String,
    pub receipt_authority: ReceiptAuthority,
    pub selected_profile: SelectedProfileV1,
    pub observed_tool: ObservedToolV1,
    pub request: RequestV1,
    pub input_scope: InputScopeV1,
    pub execution: ExecutionObservationV1,
    pub coverage: ActualCoverageV1,
    pub outputs: Vec<OutputArtifactV1>,
    pub disclosure: DisclosureObservationV1,
    pub retention: RetentionAssessmentV1,
    pub profile_drift: ProfileDriftV1,
    pub attestation_refs: Vec<String>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SelectedProfileV1 {
    pub profile_ref: String,
    pub profile_corpus_ref: String,
    pub profile_corpus_sha256: String,
    pub profile_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedToolV1 {
    pub tool_ref: String,
    pub provider_ref: String,
    pub observed_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RequestV1 {
    pub request_commitment_sha256: String,
    pub protected_request_ref: Option<String>,
    pub purpose_ref: String,
    pub policy_refs: Vec<String>,
    pub requested_bounds: RequestedBoundsV1,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RequestedBoundsV1 {
    pub max_items: Option<u64>,
    pub page_size: Option<u64>,
    pub max_pages: Option<u64>,
    pub valid_time_start: Option<String>,
    pub valid_time_end: Option<String>,
    pub corpus_commitment_sha256: Option<String>,
    pub declared_entry_count: Option<u64>,
    pub filters_commitment_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InputScopeV1 {
    pub artifact_refs: Vec<String>,
    pub corpus_commitment_sha256: Option<String>,
    pub declared_entry_count: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionObservationV1 {
    pub execution_instance_ref: String,
    pub started_at: String,
    pub ended_at: String,
    pub time_basis: TimeBasis,
    pub trusted_time_ref: Option<String>,
    pub terminal_status: TerminalStatus,
    pub items_attempted: u64,
    pub items_observed: u64,
    pub pages_attempted: u64,
    pub pages_observed: u64,
    pub retry_count: u64,
    pub errors: Vec<ExecutionError>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum TimeBasis {
    LocalClockAssertion,
    ProviderTimestamp,
    TrustedTimeRef,
    UnknownTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalStatus {
    Completed,
    Partial,
    TimedOut,
    Truncated,
    RateLimited,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionError {
    Timeout,
    TruncatedResponse,
    RateLimit,
    ProviderError,
    TransportError,
    ParseError,
    SnapshotChanged,
    PermissionDenied,
    UnknownError,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ActualCoverageV1 {
    pub state: CoverageState,
    pub scope_ref: String,
    pub evidence_refs: Vec<String>,
    pub ranked_top_k: Option<RankedTopKV1>,
    pub pagination: Option<PaginationCoverageV1>,
    pub finite_corpus: Option<FiniteCorpusCoverageV1>,
    pub account_scope: Option<AccountScopeV1>,
    pub transform: Option<TransformObservationV1>,
    pub degradation_reason: Option<DegradationReason>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum CoverageState {
    RankedTopKObserved,
    PaginationExhaustedUnderExactRunProfile,
    PartialDueToTimeout,
    PartialDueToTruncation,
    PartialDueToRateLimit,
    ExhaustiveWithinExactFiniteCorpus,
    AccountScopedObserved,
    TransformCompletedCoverageUnknown,
    UnknownCoverage,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RankedTopKV1 {
    pub requested_k: u64,
    pub observed_k: u64,
    pub ranking_scope_ref: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PaginationCoverageV1 {
    pub stable_snapshot_ref: String,
    pub declared_page_count: u64,
    pub pages_observed: u64,
    pub all_pages_accounted_for: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FiniteCorpusCoverageV1 {
    pub corpus_commitment_sha256: String,
    pub declared_entry_count: u64,
    pub entries_scanned: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AccountScopeV1 {
    pub account_context_ref: String,
    pub visibility_profile_ref: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TransformObservationV1 {
    pub input_artifact_ref: String,
    pub input_sha256: String,
    pub transform_ref: String,
    pub transform_version: String,
    pub config_commitment_sha256: Option<String>,
    pub output_sha256: String,
    pub source_native: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DegradationReason {
    Timeout,
    Truncation,
    RateLimit,
    ProviderError,
    TransportError,
    UnknownDegradation,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OutputArtifactV1 {
    pub artifact_ref: String,
    pub sha256: String,
    pub byte_length: u64,
    pub media_type: String,
    pub source_native: bool,
    pub transform_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DisclosureObservationV1 {
    pub profile_declared_surfaces: Vec<DisclosureSurface>,
    pub observed_or_necessary_surfaces: Vec<DisclosureSurface>,
    pub unexpected_surfaces: Vec<DisclosureSurface>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
pub enum DisclosureSurface {
    QueryText,
    StructuredFilter,
    AccountIdentity,
    CredentialUse,
    ClientNetworkMetadata,
    RequestTiming,
    LocationContext,
    InputArtifact,
    CorpusIdentity,
    LocalProcessMemory,
    OtherObservedSurface,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetentionAssessmentV1 {
    pub assessment: RetentionClass,
    pub evidence_refs: Vec<String>,
    pub policy_ref: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum RetentionClass {
    ObservedLocal,
    ProviderDeclared,
    Unknown,
    NotApplicable,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileDriftV1 {
    pub observed: bool,
    pub changed_fields: Vec<ProfileDriftField>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileDriftField {
    ToolIdentity,
    ProviderIdentity,
    ProviderVersion,
    AccountView,
    PaginationSemantics,
    DisclosureSurface,
    RetentionBehavior,
    TransformVersion,
    ResultCap,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct ProfileCorpusV1 {
    schema: String,
    authority: String,
    profiles: Vec<ProfileCatalogEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct ProfileCatalogEntry {
    profile_id: String,
    provider_version: String,
    auth_requirement: Option<String>,
    network_requirement: Option<String>,
    coverage_model: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum InterpretationAuthority {
    CandidateAnalysisOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutionEvidenceInterpretationV1 {
    pub schema_version: &'static str,
    pub receipt_id: String,
    pub authority_ceiling: InterpretationAuthority,
    pub selected_profile_ref: String,
    pub profile_corpus_sha256: String,
    pub tool_ref: String,
    pub provider_ref: String,
    pub observed_provider_version: String,
    pub coverage_state: CoverageState,
    pub coverage_scope_ref: String,
    pub coverage_evidence_refs: Vec<String>,
    pub degradation_reason: Option<DegradationReason>,
    pub profile_drift_observed: bool,
    pub profile_drift_fields: Vec<ProfileDriftField>,
    pub declared_disclosure_surfaces: Vec<DisclosureSurface>,
    pub observed_disclosure_surfaces: Vec<DisclosureSurface>,
    pub unexpected_disclosure_surfaces: Vec<DisclosureSurface>,
    pub retention_assessment: RetentionClass,
    pub output_artifacts: Vec<OutputArtifactV1>,
    pub limitations: Vec<String>,
    pub nonclaims: Vec<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptInterpretationError {
    pub code: &'static str,
    pub path: &'static str,
    pub message: String,
}

impl ReceiptInterpretationError {
    fn new(code: &'static str, path: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            path,
            message: message.into(),
        }
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn require_non_empty(
    value: &str,
    code: &'static str,
    path: &'static str,
    errors: &mut Vec<ReceiptInterpretationError>,
) {
    if value.trim().is_empty() {
        errors.push(ReceiptInterpretationError::new(
            code,
            path,
            "value must not be empty",
        ));
    }
}

fn profile_ref_version(profile_ref: &str) -> Option<&str> {
    profile_ref.rsplit_once(':').map(|(_, version)| version)
}

fn add_limitation(limits: &mut Vec<String>, text: &str) {
    if !limits.iter().any(|existing| existing == text) {
        limits.push(text.to_string());
    }
}

/// Interpret one exact Mycelix execution receipt using the exact profile-corpus
/// bytes pinned by both the caller and the receipt.
///
/// The caller-supplied pin must come from the trusted EPI import/frontier, not
/// from a model-generated guess. This routine validates semantics and returns
/// candidate analysis only; it does not verify a Mycelix signature or grant
/// collection, admission, memory-write, or action authority.
pub fn interpret_tool_execution_receipt(
    receipt: &ToolExecutionReceiptV1,
    profile_corpus_bytes: &[u8],
    pinned_profile_corpus_sha256: &str,
) -> Result<ExecutionEvidenceInterpretationV1, Vec<ReceiptInterpretationError>> {
    let mut errors = Vec::new();

    if receipt.schema_version != RECEIPT_SCHEMA_VERSION {
        errors.push(ReceiptInterpretationError::new(
            "unsupported_schema",
            "schema_version",
            format!(
                "expected {RECEIPT_SCHEMA_VERSION}, got {}",
                receipt.schema_version
            ),
        ));
    }
    if receipt.receipt_authority != ReceiptAuthority::ExecutionObservationOnly {
        errors.push(ReceiptInterpretationError::new(
            "authority_escalation",
            "receipt_authority",
            "only ExecutionObservationOnly receipts may be interpreted",
        ));
    }
    if receipt.selected_profile.profile_corpus_ref != PROFILE_CORPUS_SCHEMA {
        errors.push(ReceiptInterpretationError::new(
            "unknown_profile_corpus",
            "selected_profile.profile_corpus_ref",
            "receipt does not reference the expected EPI-013 profile corpus",
        ));
    }

    let actual_profile_corpus_sha256 = sha256_hex(profile_corpus_bytes);
    if actual_profile_corpus_sha256 != pinned_profile_corpus_sha256 {
        errors.push(ReceiptInterpretationError::new(
            "profile_corpus_commitment_mismatch",
            "profile_corpus",
            format!(
                "pinned profile corpus commitment {pinned_profile_corpus_sha256} does not match supplied bytes {actual_profile_corpus_sha256}"
            ),
        ));
    }
    if receipt.selected_profile.profile_corpus_sha256 != pinned_profile_corpus_sha256 {
        errors.push(ReceiptInterpretationError::new(
            "receipt_profile_commitment_mismatch",
            "selected_profile.profile_corpus_sha256",
            "receipt and trusted profile-corpus pin do not match",
        ));
    }

    let profile_corpus: Option<ProfileCorpusV1> = match serde_json::from_slice(profile_corpus_bytes)
    {
        Ok(value) => Some(value),
        Err(error) => {
            errors.push(ReceiptInterpretationError::new(
                "invalid_profile_corpus",
                "profile_corpus",
                error.to_string(),
            ));
            None
        }
    };

    let profile = profile_corpus.as_ref().and_then(|catalog| {
        if catalog.schema != PROFILE_CORPUS_SCHEMA {
            errors.push(ReceiptInterpretationError::new(
                "profile_corpus_schema_mismatch",
                "profile_corpus.schema",
                format!("expected {PROFILE_CORPUS_SCHEMA}, got {}", catalog.schema),
            ));
            None
        } else if catalog.authority != "ProfileDescriptionOnly" {
            errors.push(ReceiptInterpretationError::new(
                "profile_corpus_authority_mismatch",
                "profile_corpus.authority",
                "EPI-013 profiles must remain description-only",
            ));
            None
        } else {
            let found = catalog
                .profiles
                .iter()
                .find(|candidate| candidate.profile_id == receipt.selected_profile.profile_ref);
            if found.is_none() {
                errors.push(ReceiptInterpretationError::new(
                    "selected_profile_not_found",
                    "selected_profile.profile_ref",
                    "selected profile does not exist in the pinned EPI-013 corpus",
                ));
            }
            found
        }
    });

    if let Some(version) = profile_ref_version(&receipt.selected_profile.profile_ref) {
        if version != receipt.selected_profile.profile_version {
            errors.push(ReceiptInterpretationError::new(
                "selected_profile_version_mismatch",
                "selected_profile.profile_version",
                "selected profile version does not match the version in its profile reference",
            ));
        }
    } else {
        errors.push(ReceiptInterpretationError::new(
            "invalid_profile_reference",
            "selected_profile.profile_ref",
            "profile reference must include a version suffix",
        ));
    }

    if let Some(profile) = profile {
        let provider_version_changed =
            profile.provider_version != receipt.observed_tool.observed_version;
        let provider_version_drift_declared = receipt
            .profile_drift
            .changed_fields
            .contains(&ProfileDriftField::ProviderVersion);
        if provider_version_changed
            && (!receipt.profile_drift.observed || !provider_version_drift_declared)
        {
            errors.push(ReceiptInterpretationError::new(
                "unreported_provider_version_drift",
                "profile_drift",
                format!(
                    "profile declares provider version {}, but run observed {}; profile drift must explicitly record provider_version",
                    profile.provider_version, receipt.observed_tool.observed_version
                ),
            ));
        }
        if !provider_version_changed && provider_version_drift_declared {
            errors.push(ReceiptInterpretationError::new(
                "spurious_provider_version_drift",
                "profile_drift.changed_fields",
                "provider_version drift is declared but the observed version matches the profile",
            ));
        }

        if receipt.coverage.state == CoverageState::AccountScopedObserved
            && profile.auth_requirement.as_deref() == Some("None")
        {
            errors.push(ReceiptInterpretationError::new(
                "account_scope_without_auth_profile",
                "coverage.account_scope",
                "account-scoped observation conflicts with a profile declaring no authentication",
            ));
        }
    }

    if receipt.profile_drift.observed != !receipt.profile_drift.changed_fields.is_empty() {
        errors.push(ReceiptInterpretationError::new(
            "profile_drift_shape_mismatch",
            "profile_drift",
            "observed must be true exactly when changed_fields is non-empty",
        ));
    }
    if receipt.profile_drift.observed && receipt.profile_drift.evidence_refs.is_empty() {
        errors.push(ReceiptInterpretationError::new(
            "profile_drift_without_evidence",
            "profile_drift.evidence_refs",
            "profile drift requires at least one observation/evidence reference",
        ));
    }

    require_non_empty(
        &receipt.receipt_id,
        "empty_receipt_id",
        "receipt_id",
        &mut errors,
    );
    require_non_empty(
        &receipt.execution.execution_instance_ref,
        "empty_execution_ref",
        "execution.execution_instance_ref",
        &mut errors,
    );
    if receipt.execution.items_observed > receipt.execution.items_attempted {
        errors.push(ReceiptInterpretationError::new(
            "observed_items_exceed_attempted",
            "execution.items_observed",
            "observed item count exceeds attempted item count",
        ));
    }
    if receipt.execution.pages_observed > receipt.execution.pages_attempted {
        errors.push(ReceiptInterpretationError::new(
            "observed_pages_exceed_attempted",
            "execution.pages_observed",
            "observed page count exceeds attempted page count",
        ));
    }

    let execution = &receipt.execution;
    let coverage = &receipt.coverage;
    let has_error = |wanted: ExecutionError| execution.errors.contains(&wanted);

    match execution.terminal_status {
        TerminalStatus::TimedOut
            if coverage.state != CoverageState::PartialDueToTimeout
                || !has_error(ExecutionError::Timeout) =>
        {
            errors.push(ReceiptInterpretationError::new(
                "timeout_not_preserved",
                "coverage.state",
                "timed-out execution must preserve PartialDueToTimeout and a timeout observation",
            ));
        }
        TerminalStatus::Truncated
            if coverage.state != CoverageState::PartialDueToTruncation
                || !has_error(ExecutionError::TruncatedResponse) =>
        {
            errors.push(ReceiptInterpretationError::new(
                "truncation_not_preserved",
                "coverage.state",
                "truncated execution must preserve PartialDueToTruncation and a truncation observation",
            ));
        }
        TerminalStatus::RateLimited
            if coverage.state != CoverageState::PartialDueToRateLimit
                || !has_error(ExecutionError::RateLimit) =>
        {
            errors.push(ReceiptInterpretationError::new(
                "rate_limit_not_preserved",
                "coverage.state",
                "rate-limited execution must preserve PartialDueToRateLimit and a rate-limit observation",
            ));
        }
        _ => {}
    }

    if coverage.evidence_refs.is_empty() && coverage.state != CoverageState::UnknownCoverage {
        errors.push(ReceiptInterpretationError::new(
            "coverage_without_evidence",
            "coverage.evidence_refs",
            "non-unknown coverage requires observation evidence references",
        ));
    }

    let requested = &receipt.request.requested_bounds;
    match coverage.state {
        CoverageState::RankedTopKObserved => {
            let rank = coverage.ranked_top_k.as_ref();
            match rank {
                None => errors.push(ReceiptInterpretationError::new(
                    "missing_topk_observation",
                    "coverage.ranked_top_k",
                    "RankedTopKObserved requires an exact requested/observed K record",
                )),
                Some(rank) => {
                    if rank.observed_k != execution.items_observed
                        || rank.observed_k > rank.requested_k
                        || requested.max_items.is_some_and(|bound| rank.requested_k > bound)
                    {
                        errors.push(ReceiptInterpretationError::new(
                            "invalid_topk_observation",
                            "coverage.ranked_top_k",
                            "Top-K counts conflict with execution observations or requested bounds",
                        ));
                    }
                }
            }
            if coverage.pagination.is_some() || coverage.finite_corpus.is_some() {
                errors.push(ReceiptInterpretationError::new(
                    "topk_exhaustiveness_laundering",
                    "coverage",
                    "ranked Top-K observations cannot carry an exhaustive pagination or finite-corpus basis",
                ));
            }
        }
        CoverageState::PaginationExhaustedUnderExactRunProfile => {
            let pagination = coverage.pagination.as_ref();
            match pagination {
                None => errors.push(ReceiptInterpretationError::new(
                    "missing_pagination_observation",
                    "coverage.pagination",
                    "pagination exhaustion requires a stable snapshot and exact page accounting",
                )),
                Some(pagination) => {
                    let requested_pages_ok = requested
                        .max_pages
                        .is_some_and(|bound| bound >= pagination.declared_page_count);
                    if pagination.stable_snapshot_ref.trim().is_empty()
                        || !pagination.all_pages_accounted_for
                        || pagination.declared_page_count == 0
                        || pagination.pages_observed != pagination.declared_page_count
                        || execution.pages_observed != pagination.declared_page_count
                        || execution.pages_attempted != pagination.declared_page_count
                        || !requested_pages_ok
                        || execution.terminal_status != TerminalStatus::Completed
                        || !execution.errors.is_empty()
                    {
                        errors.push(ReceiptInterpretationError::new(
                            "pagination_exhaustion_not_proven",
                            "coverage.pagination",
                            "pagination exhaustion requires the exact requested stable snapshot, every declared page, and a clean completed execution",
                        ));
                    }
                }
            }
        }
        CoverageState::ExhaustiveWithinExactFiniteCorpus => {
            match coverage.finite_corpus.as_ref() {
                None => errors.push(ReceiptInterpretationError::new(
                    "missing_finite_corpus_observation",
                    "coverage.finite_corpus",
                    "finite-corpus exhaustion requires exact corpus identity and an all-entry scan",
                )),
                Some(finite) => {
                    let commitments_match = finite.corpus_commitment_sha256
                        == receipt.input_scope.corpus_commitment_sha256.as_deref().unwrap_or("")
                        && finite.corpus_commitment_sha256
                            == requested.corpus_commitment_sha256.as_deref().unwrap_or("");
                    let counts_match = finite.declared_entry_count == finite.entries_scanned
                        && Some(finite.declared_entry_count)
                            == receipt.input_scope.declared_entry_count
                        && Some(finite.declared_entry_count) == requested.declared_entry_count;
                    if !commitments_match
                        || !counts_match
                        || execution.items_attempted != finite.declared_entry_count
                        || execution.items_observed != finite.declared_entry_count
                        || execution.terminal_status != TerminalStatus::Completed
                        || !execution.errors.is_empty()
                    {
                        errors.push(ReceiptInterpretationError::new(
                            "finite_corpus_exhaustion_not_proven",
                            "coverage.finite_corpus",
                            "finite-corpus exhaustion requires matching exact commitments/counts and a clean all-entry scan",
                        ));
                    }
                }
            }
        }
        CoverageState::AccountScopedObserved => {
            match coverage.account_scope.as_ref() {
                Some(scope)
                    if !scope.account_context_ref.trim().is_empty()
                        && !scope.visibility_profile_ref.trim().is_empty() => {}
                _ => errors.push(ReceiptInterpretationError::new(
                    "missing_account_scope",
                    "coverage.account_scope",
                    "account-scoped results require opaque account-context and visibility-profile references",
                )),
            }
        }
        CoverageState::TransformCompletedCoverageUnknown => {
            match coverage.transform.as_ref() {
                Some(transform) => {
                    let input_is_bound = receipt
                        .input_scope
                        .artifact_refs
                        .contains(&transform.input_artifact_ref);
                    let outputs_match = !receipt.outputs.is_empty()
                        && receipt.outputs.iter().all(|output| !output.source_native)
                        && receipt
                            .outputs
                            .iter()
                            .any(|output| output.sha256 == transform.output_sha256);
                    if transform.source_native || !input_is_bound || !outputs_match {
                        errors.push(ReceiptInterpretationError::new(
                            "transform_lineage_invalid",
                            "coverage.transform",
                            "transform must bind an input artifact and derived, non-source-native output bytes",
                        ));
                    }
                }
                None => errors.push(ReceiptInterpretationError::new(
                    "missing_transform_observation",
                    "coverage.transform",
                    "transform coverage unknown requires exact input/profile/output ancestry",
                )),
            }
        }
        CoverageState::PartialDueToTimeout => {
            if !has_error(ExecutionError::Timeout)
                && execution.terminal_status != TerminalStatus::TimedOut
            {
                errors.push(ReceiptInterpretationError::new(
                    "timeout_degradation_without_observation",
                    "coverage.degradation_reason",
                    "partial timeout requires a timeout observation",
                ));
            }
        }
        CoverageState::PartialDueToTruncation => {
            if !has_error(ExecutionError::TruncatedResponse)
                && execution.terminal_status != TerminalStatus::Truncated
            {
                errors.push(ReceiptInterpretationError::new(
                    "truncation_degradation_without_observation",
                    "coverage.degradation_reason",
                    "partial truncation requires a truncation observation",
                ));
            }
        }
        CoverageState::PartialDueToRateLimit => {
            if !has_error(ExecutionError::RateLimit)
                && execution.terminal_status != TerminalStatus::RateLimited
            {
                errors.push(ReceiptInterpretationError::new(
                    "rate_limit_degradation_without_observation",
                    "coverage.degradation_reason",
                    "partial rate limit requires a rate-limit observation",
                ));
            }
        }
        CoverageState::UnknownCoverage => {}
    }

    let declared: BTreeSet<_> = receipt
        .disclosure
        .profile_declared_surfaces
        .iter()
        .copied()
        .collect();
    let observed: BTreeSet<_> = receipt
        .disclosure
        .observed_or_necessary_surfaces
        .iter()
        .copied()
        .collect();
    let unexpected: BTreeSet<_> = receipt
        .disclosure
        .unexpected_surfaces
        .iter()
        .copied()
        .collect();
    let observed_extra: BTreeSet<_> = observed.difference(&declared).copied().collect();
    if observed_extra != unexpected {
        errors.push(ReceiptInterpretationError::new(
            "disclosure_difference_mismatch",
            "disclosure.unexpected_surfaces",
            "unexpected surfaces must equal the exact observed-minus-profile-declared set",
        ));
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    let mut limitations = receipt.limitations.clone();
    match coverage.state {
        CoverageState::RankedTopKObserved => add_limitation(
            &mut limitations,
            "Ranked Top-K is not exhaustive and cannot prove world or source absence.",
        ),
        CoverageState::PaginationExhaustedUnderExactRunProfile => add_limitation(
            &mut limitations,
            "Pagination exhaustion is limited to the exact provider snapshot and selected profile.",
        ),
        CoverageState::ExhaustiveWithinExactFiniteCorpus => add_limitation(
            &mut limitations,
            "Exhaustiveness is limited to the exact committed finite corpus, not the world.",
        ),
        CoverageState::AccountScopedObserved => add_limitation(
            &mut limitations,
            "Results describe only the referenced account-scoped provider view.",
        ),
        CoverageState::TransformCompletedCoverageUnknown => add_limitation(
            &mut limitations,
            "Transform output is derived; completion does not establish semantic or source coverage.",
        ),
        CoverageState::PartialDueToTimeout
        | CoverageState::PartialDueToTruncation
        | CoverageState::PartialDueToRateLimit => add_limitation(
            &mut limitations,
            "Degraded execution cannot establish absence beyond its observed partial result set.",
        ),
        CoverageState::UnknownCoverage => add_limitation(
            &mut limitations,
            "Actual coverage is unknown; no absence conclusion is supported.",
        ),
    }

    Ok(ExecutionEvidenceInterpretationV1 {
        schema_version: INTERPRETATION_SCHEMA_VERSION,
        receipt_id: receipt.receipt_id.clone(),
        authority_ceiling: InterpretationAuthority::CandidateAnalysisOnly,
        selected_profile_ref: receipt.selected_profile.profile_ref.clone(),
        profile_corpus_sha256: actual_profile_corpus_sha256,
        tool_ref: receipt.observed_tool.tool_ref.clone(),
        provider_ref: receipt.observed_tool.provider_ref.clone(),
        observed_provider_version: receipt.observed_tool.observed_version.clone(),
        coverage_state: coverage.state,
        coverage_scope_ref: coverage.scope_ref.clone(),
        coverage_evidence_refs: coverage.evidence_refs.clone(),
        degradation_reason: coverage.degradation_reason,
        profile_drift_observed: receipt.profile_drift.observed,
        profile_drift_fields: receipt.profile_drift.changed_fields.clone(),
        declared_disclosure_surfaces: receipt.disclosure.profile_declared_surfaces.clone(),
        observed_disclosure_surfaces: receipt.disclosure.observed_or_necessary_surfaces.clone(),
        unexpected_disclosure_surfaces: receipt.disclosure.unexpected_surfaces.clone(),
        retention_assessment: receipt.retention.assessment,
        output_artifacts: receipt.outputs.clone(),
        limitations,
        nonclaims: vec![
            "execution receipt does not establish returned-content truth",
            "profile or receipt does not establish source authenticity or independent corroboration",
            "execution observation does not grant collection, credential, network, browser, EPI admission, memory-write or action authority",
            "Top-K, account-scoped and degraded results do not establish world completeness",
            "cryptographic receipt integrity does not upgrade semantic coverage or factual truth",
        ],
    })
}

pub fn interpret_tool_execution_receipt_json(
    receipt_bytes: &[u8],
    profile_corpus_bytes: &[u8],
    pinned_profile_corpus_sha256: &str,
) -> Result<ExecutionEvidenceInterpretationV1, Vec<ReceiptInterpretationError>> {
    let receipt: ToolExecutionReceiptV1 =
        serde_json::from_slice(receipt_bytes).map_err(|error| {
            vec![ReceiptInterpretationError::new(
                "invalid_receipt_shape",
                "receipt",
                error.to_string(),
            )]
        })?;
    interpret_tool_execution_receipt(&receipt, profile_corpus_bytes, pinned_profile_corpus_sha256)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECEIPT_CORPUS: &str =
        include_str!("../tests/fixtures/EPI_013R_TOOL_EXECUTION_RECEIPTS_V0_1.json");
    const PROFILE_CORPUS: &str =
        include_str!("../tests/fixtures/EPI_013_OSINT_TOOL_CAPABILITY_PROFILES_V0_1.json");

    fn fixture_values() -> (Vec<Value>, Vec<u8>) {
        let corpus: Value = serde_json::from_str(RECEIPT_CORPUS).expect("receipt fixture JSON");
        let receipts = corpus
            .get("receipts")
            .and_then(Value::as_array)
            .expect("receipt array")
            .clone();
        (receipts, PROFILE_CORPUS.as_bytes().to_vec())
    }

    fn parse_receipt(value: Value) -> ToolExecutionReceiptV1 {
        serde_json::from_value(value).expect("typed receipt fixture")
    }

    fn interpret(
        value: Value,
    ) -> Result<ExecutionEvidenceInterpretationV1, Vec<ReceiptInterpretationError>> {
        let receipt = parse_receipt(value);
        interpret_tool_execution_receipt(&receipt, PROFILE_CORPUS.as_bytes(), PROFILE_CORPUS_SHA256)
    }

    #[test]
    fn fixture_digests_match_the_mycelix_committed_bytes() {
        assert_eq!(sha256_hex(RECEIPT_CORPUS.as_bytes()), RECEIPT_CORPUS_SHA256);
        assert_eq!(sha256_hex(PROFILE_CORPUS.as_bytes()), PROFILE_CORPUS_SHA256);
    }

    #[test]
    fn every_positive_mycelix_receipt_is_interpretable() {
        let (receipts, profile_bytes) = fixture_values();
        assert_eq!(receipts.len(), 8);
        for value in receipts {
            let receipt = parse_receipt(value);
            let result =
                interpret_tool_execution_receipt(&receipt, &profile_bytes, PROFILE_CORPUS_SHA256);
            assert!(result.is_ok(), "{:?}: {result:?}", receipt.receipt_id);
            assert_eq!(
                result.unwrap().authority_ceiling,
                InterpretationAuthority::CandidateAnalysisOnly
            );
        }
    }

    #[test]
    fn topk_remains_non_exhaustive() {
        let (receipts, _) = fixture_values();
        let value = receipts
            .into_iter()
            .find(|r| r["receipt_id"] == "receipt:synthetic:ranked-topk")
            .unwrap();
        let result = interpret(value).unwrap();
        assert_eq!(result.coverage_state, CoverageState::RankedTopKObserved);
        assert!(
            result
                .limitations
                .iter()
                .any(|x| x.contains("not exhaustive"))
        );
    }

    #[test]
    fn pagination_completion_is_scoped_to_the_exact_snapshot() {
        let (receipts, _) = fixture_values();
        let value = receipts
            .into_iter()
            .find(|r| r["receipt_id"] == "receipt:synthetic:paginated-complete")
            .unwrap();
        let result = interpret(value).unwrap();
        assert_eq!(
            result.coverage_state,
            CoverageState::PaginationExhaustedUnderExactRunProfile
        );
        assert!(
            result
                .limitations
                .iter()
                .any(|x| x.contains("exact provider snapshot"))
        );
    }

    #[test]
    fn deleting_the_final_page_rejects_pagination_exhaustion() {
        let (mut receipts, _) = fixture_values();
        let mut value = receipts
            .drain(..)
            .find(|r| r["receipt_id"] == "receipt:synthetic:paginated-complete")
            .unwrap();
        value["coverage"]["pagination"]["pages_observed"] = Value::from(3);
        value["coverage"]["pagination"]["all_pages_accounted_for"] = Value::Bool(false);
        value["execution"]["pages_observed"] = Value::from(3);
        assert!(interpret(value).is_err());
    }

    #[test]
    fn timeout_is_preserved_as_partial_not_rejected_or_upgraded() {
        let (receipts, _) = fixture_values();
        let value = receipts
            .into_iter()
            .find(|r| r["receipt_id"] == "receipt:synthetic:pagination-timeout")
            .unwrap();
        let result = interpret(value).unwrap();
        assert_eq!(result.coverage_state, CoverageState::PartialDueToTimeout);
        assert_eq!(result.degradation_reason, Some(DegradationReason::Timeout));
        assert!(
            result
                .limitations
                .iter()
                .any(|x| x.contains("Degraded execution"))
        );
    }

    #[test]
    fn finite_corpus_mismatch_rejects_exhaustion() {
        let (receipts, _) = fixture_values();
        let mut value = receipts
            .into_iter()
            .find(|r| r["receipt_id"] == "receipt:synthetic:finite-corpus")
            .unwrap();
        value["input_scope"]["corpus_commitment_sha256"] = Value::String("0".repeat(64));
        assert!(interpret(value).is_err());
    }

    #[test]
    fn account_scope_is_preserved_without_world_completeness() {
        let (receipts, _) = fixture_values();
        let value = receipts
            .into_iter()
            .find(|r| r["receipt_id"] == "receipt:synthetic:authenticated-account-scope")
            .unwrap();
        let result = interpret(value).unwrap();
        assert_eq!(result.coverage_state, CoverageState::AccountScopedObserved);
        assert!(
            result
                .limitations
                .iter()
                .any(|x| x.contains("account-scoped"))
        );
        assert_eq!(result.retention_assessment, RetentionClass::Unknown);
    }

    #[test]
    fn ocr_remains_derived_and_coverage_unknown() {
        let (receipts, _) = fixture_values();
        let value = receipts
            .into_iter()
            .find(|r| r["receipt_id"] == "receipt:synthetic:ocr-transform")
            .unwrap();
        let result = interpret(value).unwrap();
        assert_eq!(
            result.coverage_state,
            CoverageState::TransformCompletedCoverageUnknown
        );
        assert!(
            result
                .output_artifacts
                .iter()
                .all(|output| !output.source_native)
        );
    }

    #[test]
    fn provider_version_drift_is_explicit_and_preserved() {
        let (receipts, _) = fixture_values();
        let value = receipts
            .into_iter()
            .find(|r| r["receipt_id"] == "receipt:synthetic:provider-version-drift")
            .unwrap();
        let result = interpret(value).unwrap();
        assert!(result.profile_drift_observed);
        assert!(
            result
                .profile_drift_fields
                .contains(&ProfileDriftField::ProviderVersion)
        );
    }

    #[test]
    fn unreported_provider_version_drift_rejects() {
        let (receipts, _) = fixture_values();
        let mut value = receipts
            .into_iter()
            .find(|r| r["receipt_id"] == "receipt:synthetic:provider-version-drift")
            .unwrap();
        value["profile_drift"]["observed"] = Value::Bool(false);
        value["profile_drift"]["changed_fields"] = Value::Array(Vec::new());
        value["profile_drift"]["evidence_refs"] = Value::Array(Vec::new());
        assert!(interpret(value).is_err());
    }

    #[test]
    fn unexpected_disclosure_is_not_erased() {
        let (receipts, _) = fixture_values();
        let value = receipts
            .into_iter()
            .find(|r| r["receipt_id"] == "receipt:synthetic:unexpected-disclosure")
            .unwrap();
        let result = interpret(value).unwrap();
        assert_eq!(
            result.unexpected_disclosure_surfaces,
            vec![DisclosureSurface::LocationContext]
        );
    }

    #[test]
    fn disclosure_omission_rejects() {
        let (receipts, _) = fixture_values();
        let mut value = receipts
            .into_iter()
            .find(|r| r["receipt_id"] == "receipt:synthetic:unexpected-disclosure")
            .unwrap();
        value["disclosure"]["unexpected_surfaces"] = Value::Array(Vec::new());
        assert!(interpret(value).is_err());
    }

    #[test]
    fn raw_secret_field_is_rejected_during_typed_deserialization() {
        let (receipts, _) = fixture_values();
        let mut value = receipts
            .into_iter()
            .find(|r| r["receipt_id"] == "receipt:synthetic:authenticated-account-scope")
            .unwrap();
        value["request"]["credential"] = Value::String("secret".to_string());
        let bytes = serde_json::to_vec(&value).unwrap();
        let result = interpret_tool_execution_receipt_json(
            &bytes,
            PROFILE_CORPUS.as_bytes(),
            PROFILE_CORPUS_SHA256,
        );
        assert!(result.is_err());
    }

    #[test]
    fn unsupported_no_retention_claim_is_rejected() {
        let (receipts, _) = fixture_values();
        let mut value = receipts
            .into_iter()
            .find(|r| r["receipt_id"] == "receipt:synthetic:authenticated-account-scope")
            .unwrap();
        value["retention"]["assessment"] = Value::String("NoRetention".to_string());
        assert!(serde_json::from_value::<ToolExecutionReceiptV1>(value).is_err());
    }

    #[test]
    fn transparency_reference_does_not_upgrade_degraded_coverage() {
        let (receipts, _) = fixture_values();
        let mut value = receipts
            .into_iter()
            .find(|r| r["receipt_id"] == "receipt:synthetic:pagination-timeout")
            .unwrap();
        value["attestation_refs"] = serde_json::json!(["transparency:receipt:synthetic"]);
        value["coverage"]["state"] =
            Value::String("PaginationExhaustedUnderExactRunProfile".to_string());
        value["coverage"]["pagination"] = serde_json::json!({
            "stable_snapshot_ref":"snapshot:bad",
            "declared_page_count":4,
            "pages_observed":2,
            "all_pages_accounted_for":false
        });
        assert!(interpret(value).is_err());
    }

    #[test]
    fn profile_corpus_digest_is_verified_before_interpretation() {
        let (receipts, _) = fixture_values();
        let receipt = parse_receipt(
            receipts
                .into_iter()
                .find(|r| r["receipt_id"] == "receipt:synthetic:ranked-topk")
                .unwrap(),
        );
        let result =
            interpret_tool_execution_receipt(&receipt, PROFILE_CORPUS.as_bytes(), &"0".repeat(64));
        assert!(result.is_err());
    }

    #[test]
    fn authority_ceiling_cannot_be_escalated() {
        let (receipts, _) = fixture_values();
        let mut value = receipts
            .into_iter()
            .find(|r| r["receipt_id"] == "receipt:synthetic:ranked-topk")
            .unwrap();
        value["receipt_authority"] = Value::String("CollectionPermit".to_string());
        assert!(serde_json::from_value::<ToolExecutionReceiptV1>(value).is_err());
    }
}
