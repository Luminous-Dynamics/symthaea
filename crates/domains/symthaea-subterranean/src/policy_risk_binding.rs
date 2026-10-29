// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed binding between an ordered policy set and the loss matrix used
//! to choose among those policies.
//!
//! This module verifies integrity and ordered identity. It does not authenticate
//! a manifest by itself: trusted_manifest_sha256 MUST come from a separately
//! authenticated signed release/receipt or trusted control-plane record.
//! Callers should verify the binding both before optimization and immediately
//! before action dispatch. This helper is not integrated into the production
//! controller merely by being present in the crate.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fmt;

/// Manifest schema for canonical policy-to-risk binding.
pub const POLICY_RISK_BINDING_SCHEMA: &str = "symthaea.policy-risk-binding.v1";

/// Fail-closed errors for policy/risk matrix identity mismatches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyRiskBindingError {
    EmptyPolicyIds,
    InvalidPolicyId,
    DuplicatePolicyId,
    EmptyHypothesisIds,
    InvalidHypothesisId,
    DuplicateHypothesisId,
    EmptyRiskMatrix,
    RiskMatrixShapeMismatch,
    RiskMatrixNonFinite,
    NegativeRiskValue,
    ManifestSchemaMismatch,
    ManifestSelfDigestMismatch,
    TrustedManifestDigestMismatch,
    RuntimePolicyOrderMismatch,
    RuntimeHypothesisOrderMismatch,
    ExecutionPolicyOrderMismatch,
    RiskMatrixDigestMismatch,
}

impl fmt::Display for PolicyRiskBindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Error for PolicyRiskBindingError {}

/// Ordered identities and the exact canonical loss matrix they describe.
///
/// IDs should be stable/content-addressed when possible. manifest_sha256 is
/// the digest of all other manifest fields; it is not a digital signature.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyRiskManifest {
    pub schema: String,
    pub policy_ids: Vec<String>,
    pub hypothesis_ids: Vec<String>,
    pub rows: usize,
    pub columns: usize,
    /// SHA-256 of canonical row-major IEEE-754 f64 big-endian matrix bytes.
    pub risk_matrix_sha256: [u8; 32],
    /// SHA-256 of the canonical, domain-separated manifest payload.
    pub manifest_sha256: [u8; 32],
}

fn validate_policy_ids(ids: &[String]) -> Result<(), PolicyRiskBindingError> {
    if ids.is_empty() {
        return Err(PolicyRiskBindingError::EmptyPolicyIds);
    }
    if ids.iter().any(|id| id.is_empty() || id.trim() != id) {
        return Err(PolicyRiskBindingError::InvalidPolicyId);
    }
    let mut seen = std::collections::BTreeSet::new();
    if ids.iter().any(|id| !seen.insert(id.as_str())) {
        return Err(PolicyRiskBindingError::DuplicatePolicyId);
    }
    Ok(())
}

fn validate_hypothesis_ids(ids: &[String]) -> Result<(), PolicyRiskBindingError> {
    if ids.is_empty() {
        return Err(PolicyRiskBindingError::EmptyHypothesisIds);
    }
    if ids.iter().any(|id| id.is_empty() || id.trim() != id) {
        return Err(PolicyRiskBindingError::InvalidHypothesisId);
    }
    let mut seen = std::collections::BTreeSet::new();
    if ids.iter().any(|id| !seen.insert(id.as_str())) {
        return Err(PolicyRiskBindingError::DuplicateHypothesisId);
    }
    Ok(())
}

fn checked_matrix_shape(
    rows: usize,
    columns: usize,
    values: &[f64],
) -> Result<(), PolicyRiskBindingError> {
    if rows == 0 || columns == 0 || values.is_empty() {
        return Err(PolicyRiskBindingError::EmptyRiskMatrix);
    }
    let Some(expected_len) = rows.checked_mul(columns) else {
        return Err(PolicyRiskBindingError::RiskMatrixShapeMismatch);
    };
    if expected_len != values.len() {
        return Err(PolicyRiskBindingError::RiskMatrixShapeMismatch);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PolicyRiskBindingError::RiskMatrixNonFinite);
    }
    if values.iter().any(|value| *value < 0.0) {
        return Err(PolicyRiskBindingError::NegativeRiskValue);
    }
    Ok(())
}

fn push_len_prefixed(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
}

fn matrix_digest(
    rows: usize,
    columns: usize,
    values: &[f64],
) -> Result<[u8; 32], PolicyRiskBindingError> {
    checked_matrix_shape(rows, columns, values)?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.policy-risk-matrix.v1\0");
    hasher.update((rows as u64).to_be_bytes());
    hasher.update((columns as u64).to_be_bytes());
    for value in values {
        // Canonicalize signed zero: -0.0 and +0.0 are the same numerical loss.
        let canonical = if *value == 0.0 { 0.0_f64 } else { *value };
        hasher.update(canonical.to_bits().to_be_bytes());
    }
    Ok(hasher.finalize().into())
}

fn manifest_payload_digest(manifest: &PolicyRiskManifest) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.policy-risk-manifest.v1\0");
    push_len_prefixed(&mut hasher, manifest.schema.as_bytes());
    hasher.update((manifest.policy_ids.len() as u64).to_be_bytes());
    for id in &manifest.policy_ids {
        push_len_prefixed(&mut hasher, id.as_bytes());
    }
    hasher.update((manifest.hypothesis_ids.len() as u64).to_be_bytes());
    for id in &manifest.hypothesis_ids {
        push_len_prefixed(&mut hasher, id.as_bytes());
    }
    hasher.update((manifest.rows as u64).to_be_bytes());
    hasher.update((manifest.columns as u64).to_be_bytes());
    hasher.update(manifest.risk_matrix_sha256);
    hasher.finalize().into()
}

/// Create a canonical manifest from row-major policy-loss values.
pub fn build_policy_risk_manifest(
    policy_ids: &[String],
    hypothesis_ids: &[String],
    row_major_risk_values: &[f64],
) -> Result<PolicyRiskManifest, PolicyRiskBindingError> {
    validate_policy_ids(policy_ids)?;
    validate_hypothesis_ids(hypothesis_ids)?;
    let rows = policy_ids.len();
    let columns = hypothesis_ids.len();
    let risk_matrix_sha256 = matrix_digest(rows, columns, row_major_risk_values)?;
    let mut manifest = PolicyRiskManifest {
        schema: POLICY_RISK_BINDING_SCHEMA.to_owned(),
        policy_ids: policy_ids.to_vec(),
        hypothesis_ids: hypothesis_ids.to_vec(),
        rows,
        columns,
        risk_matrix_sha256,
        manifest_sha256: [0; 32],
    };
    manifest.manifest_sha256 = manifest_payload_digest(&manifest);
    Ok(manifest)
}

/// Verify that calibration, candidate policy ordering, matrix bytes and final
/// execution ordering all match one externally trusted manifest.
///
/// trusted_manifest_sha256 must be obtained from a trusted signature/receipt
/// boundary, not copied from the same untrusted input as the proposed matrix.
pub fn verify_policy_risk_binding(
    trusted_manifest: &PolicyRiskManifest,
    trusted_manifest_sha256: &[u8; 32],
    runtime_policy_ids: &[String],
    runtime_hypothesis_ids: &[String],
    runtime_rows: usize,
    runtime_columns: usize,
    runtime_row_major_risk_values: &[f64],
    execution_policy_ids: &[String],
) -> Result<[u8; 32], PolicyRiskBindingError> {
    if trusted_manifest.schema != POLICY_RISK_BINDING_SCHEMA {
        return Err(PolicyRiskBindingError::ManifestSchemaMismatch);
    }
    validate_policy_ids(&trusted_manifest.policy_ids)?;
    validate_hypothesis_ids(&trusted_manifest.hypothesis_ids)?;
    if trusted_manifest.rows != trusted_manifest.policy_ids.len()
        || trusted_manifest.columns != trusted_manifest.hypothesis_ids.len()
    {
        return Err(PolicyRiskBindingError::RiskMatrixShapeMismatch);
    }
    if manifest_payload_digest(trusted_manifest) != trusted_manifest.manifest_sha256 {
        return Err(PolicyRiskBindingError::ManifestSelfDigestMismatch);
    }
    if &trusted_manifest.manifest_sha256 != trusted_manifest_sha256 {
        return Err(PolicyRiskBindingError::TrustedManifestDigestMismatch);
    }

    validate_policy_ids(runtime_policy_ids)?;
    validate_hypothesis_ids(runtime_hypothesis_ids)?;
    validate_policy_ids(execution_policy_ids)?;

    if runtime_policy_ids != trusted_manifest.policy_ids.as_slice() {
        return Err(PolicyRiskBindingError::RuntimePolicyOrderMismatch);
    }
    if runtime_hypothesis_ids != trusted_manifest.hypothesis_ids.as_slice() {
        return Err(PolicyRiskBindingError::RuntimeHypothesisOrderMismatch);
    }
    if execution_policy_ids != trusted_manifest.policy_ids.as_slice() {
        return Err(PolicyRiskBindingError::ExecutionPolicyOrderMismatch);
    }
    if runtime_rows != trusted_manifest.rows || runtime_columns != trusted_manifest.columns {
        return Err(PolicyRiskBindingError::RiskMatrixShapeMismatch);
    }
    let runtime_digest = matrix_digest(
        runtime_rows,
        runtime_columns,
        runtime_row_major_risk_values,
    )?;
    if runtime_digest != trusted_manifest.risk_matrix_sha256 {
        return Err(PolicyRiskBindingError::RiskMatrixDigestMismatch);
    }
    Ok(trusted_manifest.manifest_sha256)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Vec<String>, Vec<String>, Vec<f64>, PolicyRiskManifest) {
        let policies = vec![
            "policy:alpha".to_owned(),
            "policy:beta".to_owned(),
            "policy:gamma".to_owned(),
        ];
        let hypotheses = vec![
            "world:0".to_owned(),
            "world:1".to_owned(),
            "world:2".to_owned(),
            "world:3".to_owned(),
        ];
        let risk = vec![
            0.2, 0.4, 0.3, 0.1,
            0.1, 0.5, 0.2, 0.4,
            0.7, 0.2, 0.1, 0.6,
        ];
        let manifest = build_policy_risk_manifest(&policies, &hypotheses, &risk).unwrap();
        (policies, hypotheses, risk, manifest)
    }

    fn verify(
        policies: &[String],
        hypotheses: &[String],
        risk: &[f64],
        manifest: &PolicyRiskManifest,
        trusted_digest: &[u8; 32],
        execution_ids: &[String],
    ) -> Result<[u8; 32], PolicyRiskBindingError> {
        verify_policy_risk_binding(
            manifest,
            trusted_digest,
            policies,
            hypotheses,
            policies.len(),
            hypotheses.len(),
            risk,
            execution_ids,
        )
    }

    #[test]
    fn exact_binding_passes_and_manifest_roundtrips() {
        let (policies, hypotheses, risk, manifest) = fixture();
        assert_eq!(
            verify(&policies, &hypotheses, &risk, &manifest, &manifest.manifest_sha256, &policies)
                .unwrap(),
            manifest.manifest_sha256
        );
        let encoded = serde_json::to_vec(&manifest).unwrap();
        let decoded: PolicyRiskManifest = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(manifest, decoded);
    }

    #[test]
    fn row_reordering_is_rejected_even_when_shape_is_unchanged() {
        let (policies, hypotheses, risk, manifest) = fixture();
        let mut reordered = Vec::new();
        for row in [2, 0, 1] {
            reordered.extend_from_slice(&risk[row * 4..(row + 1) * 4]);
        }
        assert_eq!(
            verify(&policies, &hypotheses, &reordered, &manifest, &manifest.manifest_sha256, &policies),
            Err(PolicyRiskBindingError::RiskMatrixDigestMismatch)
        );
    }

    #[test]
    fn policy_order_is_checked_independently_of_matrix_digest() {
        let (policies, hypotheses, risk, manifest) = fixture();
        let reordered = vec![policies[2].clone(), policies[0].clone(), policies[1].clone()];
        assert_eq!(
            verify(&reordered, &hypotheses, &risk, &manifest, &manifest.manifest_sha256, &policies),
            Err(PolicyRiskBindingError::RuntimePolicyOrderMismatch)
        );
    }

    #[test]
    fn execution_order_is_bound_to_planned_order() {
        let (policies, hypotheses, risk, manifest) = fixture();
        let reordered = vec![policies[1].clone(), policies[0].clone(), policies[2].clone()];
        assert_eq!(
            verify(&policies, &hypotheses, &risk, &manifest, &manifest.manifest_sha256, &reordered),
            Err(PolicyRiskBindingError::ExecutionPolicyOrderMismatch)
        );
    }

    #[test]
    fn hypothesis_order_is_checked() {
        let (policies, hypotheses, risk, manifest) = fixture();
        let reordered = vec![hypotheses[1].clone(), hypotheses[0].clone(), hypotheses[2].clone(), hypotheses[3].clone()];
        assert_eq!(
            verify(&policies, &reordered, &risk, &manifest, &manifest.manifest_sha256, &policies),
            Err(PolicyRiskBindingError::RuntimeHypothesisOrderMismatch)
        );
    }

    #[test]
    fn external_trust_anchor_and_manifest_self_digest_are_checked() {
        let (policies, hypotheses, risk, manifest) = fixture();
        assert_eq!(
            verify(&policies, &hypotheses, &risk, &manifest, &[0; 32], &policies),
            Err(PolicyRiskBindingError::TrustedManifestDigestMismatch)
        );
        let mut tampered = manifest.clone();
        tampered.risk_matrix_sha256 = [0; 32];
        assert_eq!(
            verify(&policies, &hypotheses, &risk, &tampered, &manifest.manifest_sha256, &policies),
            Err(PolicyRiskBindingError::ManifestSelfDigestMismatch)
        );
    }

    #[test]
    fn duplicate_ids_and_bad_values_fail_closed() {
        let (policies, hypotheses, risk, _) = fixture();
        let duplicate = vec!["dup".to_owned(), "dup".to_owned()];
        assert_eq!(
            build_policy_risk_manifest(&duplicate, &hypotheses, &risk[..8]),
            Err(PolicyRiskBindingError::DuplicatePolicyId)
        );
        let non_finite = vec![f64::NAN; 12];
        assert_eq!(
            build_policy_risk_manifest(&policies, &hypotheses, &non_finite),
            Err(PolicyRiskBindingError::RiskMatrixNonFinite)
        );
        let negative = vec![-1.0; 12];
        assert_eq!(
            build_policy_risk_manifest(&policies, &hypotheses, &negative),
            Err(PolicyRiskBindingError::NegativeRiskValue)
        );
        assert_eq!(
            build_policy_risk_manifest(&policies, &hypotheses, &risk[..11]),
            Err(PolicyRiskBindingError::RiskMatrixShapeMismatch)
        );
    }

    #[test]
    fn signed_zero_is_canonicalized() {
        let policies = vec!["p0".to_owned()];
        let worlds = vec!["h0".to_owned(), "h1".to_owned()];
        let plus = build_policy_risk_manifest(&policies, &worlds, &[0.0, 1.0]).unwrap();
        let minus = build_policy_risk_manifest(&policies, &worlds, &[-0.0, 1.0]).unwrap();
        assert_eq!(plus, minus);
    }
}
