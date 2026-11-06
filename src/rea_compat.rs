//! Native Rust access to the REA-compatible Evidence contract.
//!
//! Enable the root crate feature rea-compat to expose the verifier and
//! semantic identity functions without routing in-process work through MCP.

#[path = "rea_compat/claims.rs"]
pub mod claims;

pub use claims::{
    ClaimDisposition, ClaimError, ClaimErrorKind, ClaimEvaluation, ClaimPredicate, ClaimRequest,
    ClaimScope, EvidenceAssessment, EvidenceAssessmentKind, evaluate_claim, parse_claim_json,
    validate_claim, CLAIM_EVALUATOR_VERSION, CLAIM_SCHEMA, MAX_CLAIM_JSON_BYTES,
};

pub use symthaea_rea_compat::{
    analysis_profile_digest,
    canonical_jcs_bytes,
    compute_evidence_id,
    semantic_evidence_content,
    sha256_hex,
    validate_evidence_schema,
    verify_evidence_json,
    verify_evidence_value,
    EvidenceError,
    EvidenceErrorKind,
    VerifiedEvidence,
    EVIDENCE_ID_PREFIX,
    MAX_EVIDENCE_JSON_BYTES,
};
