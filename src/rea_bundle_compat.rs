//! Native Rust access to the REA EvidenceBundle and residual-unknown graph contract.
//!
//! Enable the root crate feature `rea-compat` to use this API in-process.

pub use symthaea_rea_compat::bundle::{
    BundleError, BundleErrorKind, VerifiedEvidenceBundle, audit_evidence_link_closure,
    canonical_evidence_bundle_bytes, evidence_bundle_for_target, verify_evidence_bundle_json,
    verify_evidence_bundle_value,
};
