use serde_json::Value;

#[path = "../../../src/rea_bundle_compat.rs"]
mod rea_bundle_compat;

#[test]
fn bundle_root_adapter_exports_public_contract() {
    let _error_kind = rea_bundle_compat::BundleErrorKind::DuplicateEvidenceId;
    let _canonicalizer: fn(&Value) -> Result<Vec<u8>, rea_bundle_compat::BundleError> =
        rea_bundle_compat::canonical_evidence_bundle_bytes;
    let _verifier: fn(
        &Value,
    ) -> Result<
        rea_bundle_compat::VerifiedEvidenceBundle,
        rea_bundle_compat::BundleError,
    > = rea_bundle_compat::verify_evidence_bundle_value;
    let _target_projection: fn(&Value, &str) -> Result<Value, rea_bundle_compat::BundleError> =
        rea_bundle_compat::evidence_bundle_for_target;
    type ClosureAudit = fn(
        &Value,
    ) -> Result<Vec<(String, String)>, rea_bundle_compat::BundleError>;
    let _closure_audit: ClosureAudit = rea_bundle_compat::audit_evidence_link_closure;
}
