//! Native Rust access to the REA-compatible Evidence contract.
//!
//! Enable the root crate feature rea-compat to expose the verifier and
//! semantic identity functions without routing in-process work through MCP.

#[path = "rea_compat/claims.rs"]
pub mod claims;

pub use claims::{
    CLAIM_EVALUATION_SCHEMA, CLAIM_EVALUATOR_VERSION, CLAIM_SCHEMA, ClaimDisposition, ClaimError,
    ClaimErrorKind, ClaimEvaluation, ClaimPredicate, ClaimRequest, ClaimScope, EvidenceAssessment,
    EvidenceAssessmentKind, MAX_CLAIM_JSON_BYTES, evaluate_claim, parse_claim_json, validate_claim,
};

#[allow(unused_imports)] // This adapter intentionally re-exports the complete Evidence API.
pub use symthaea_rea_compat::{
    EVIDENCE_ID_PREFIX, EvidenceError, EvidenceErrorKind, MAX_EVIDENCE_JSON_BYTES,
    VerifiedEvidence, analysis_profile_digest, canonical_jcs_bytes, compute_evidence_id,
    semantic_evidence_content, sha256_hex, validate_evidence_schema, verify_evidence_json,
    verify_evidence_value,
};
