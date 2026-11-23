// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independently replayable v2 receipts for semiconductor metrology selection.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

use crate::benchmark::metrology_audit::{
    MetrologyAuditCandidateV1, MetrologyAuditPolicyV1, MetrologyAuditSelectionReportV1,
    POLICY_SCHEMA_V1, select_metrology_audit_v1,
};
use crate::benchmark::metrology_semantics::validate_binary_measurand_semantics_v1;

pub const SELECTION_RECEIPT_V2_SCHEMA: &str =
    "luminous.semi-bench.metrology-selection-receipt.v2";
pub const SELECTION_POLICY_CONFIG_V2_SCHEMA: &str =
    "luminous.semi-bench.metrology-selection-policy-config.v2";
pub const SELECTOR_ALGORITHM_ID_V1: &str = "luminous.semi-bench.exploit-srswor-selector";
pub const SELECTOR_ALGORITHM_VERSION_V1: &str = "1";
pub const CANONICALIZATION_VERSION_V1: &str = "serde-json-btreemap-v1";
pub const SEED_DERIVATION_VERSION_V1: &str = "sha256-seed-stratum-splitmix64-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UncertaintyScoreKindV2 {
    Epistemic,
    Predictive,
    HeuristicProxy,
    NotUsed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvsiScoreKindV2 {
    DecisionTheoretic,
    HeuristicProxy,
    NotUsed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetrologyScoreProvenanceV2 {
    pub risk_score_semantics_id: String,
    pub uncertainty_score_kind: UncertaintyScoreKindV2,
    pub uncertainty_score_semantics_id: String,
    pub evsi_score_kind: EvsiScoreKindV2,
    pub evsi_score_semantics_id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetrologySelectionPolicyConfigV2 {
    pub schema: String,
    /// Producer and run identifiers are recorded, but are not authenticated
    /// unless a separately configured signature/trust-root layer is added.
    pub producer_ref: String,
    pub run_id: String,
    pub policy_id: String,
    pub model_sha256: String,
    pub prediction_packet_sha256: String,
    pub binary_semantics: crate::benchmark::metrology_semantics::BinaryMeasurandSemanticsV1,
    pub total_cost_budget: u64,
    pub exploit_cost_budget: u64,
    pub audit_cost_budget: u64,
    pub risk_weight: f64,
    pub uncertainty_weight: f64,
    pub evsi_weight: f64,
    pub audit_quota_by_stratum: std::collections::BTreeMap<String, usize>,
    pub seed: u64,
    pub selector_algorithm_id: String,
    pub selector_algorithm_version: String,
    pub canonicalization_version: String,
    pub seed_derivation_version: String,
    pub score_provenance: MetrologyScoreProvenanceV2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionReceiptAttestationV2 {
    Unattested,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetrologySelectionReceiptV2 {
    pub schema: String,
    pub configuration: MetrologySelectionPolicyConfigV2,
    /// Full score inputs in canonical candidate-ID order. These are the
    /// inputs from which the selector's weighted priority is recomputed.
    pub candidate_scores: Vec<MetrologyAuditCandidateV1>,
    pub configuration_sha256: String,
    pub selection_report: MetrologyAuditSelectionReportV1,
    pub attestation_status: SelectionReceiptAttestationV2,
    /// Hash of the full receipt with this field empty. It is tamper-evident
    /// relative to an independently trusted digest, not a producer signature.
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct SelectionReceiptV2Error(pub String);

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize>(value: &T) -> Result<String, SelectionReceiptV2Error> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        SelectionReceiptV2Error(format!("canonical serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_configuration(
    config: &MetrologySelectionPolicyConfigV2,
) -> Result<(), SelectionReceiptV2Error> {
    if config.schema != SELECTION_POLICY_CONFIG_V2_SCHEMA {
        return Err(SelectionReceiptV2Error(
            "configuration.schema: unsupported".into(),
        ));
    }
    for (field, value) in [
        ("producer_ref", config.producer_ref.as_str()),
        ("run_id", config.run_id.as_str()),
        ("policy_id", config.policy_id.as_str()),
        ("selector_algorithm_id", config.selector_algorithm_id.as_str()),
        (
            "selector_algorithm_version",
            config.selector_algorithm_version.as_str(),
        ),
        ("canonicalization_version", config.canonicalization_version.as_str()),
        ("seed_derivation_version", config.seed_derivation_version.as_str()),
        (
            "risk_score_semantics_id",
            config.score_provenance.risk_score_semantics_id.as_str(),
        ),
    ] {
        if value.trim().is_empty() {
            return Err(SelectionReceiptV2Error(format!(
                "configuration.{field}: expected non-empty string"
            )));
        }
    }
    if !valid_sha256(&config.model_sha256)
        || !valid_sha256(&config.prediction_packet_sha256)
    {
        return Err(SelectionReceiptV2Error(
            "configuration requires lowercase model and prediction SHA-256 digests".into(),
        ));
    }
    if config.selector_algorithm_id != SELECTOR_ALGORITHM_ID_V1
        || config.selector_algorithm_version != SELECTOR_ALGORITHM_VERSION_V1
        || config.canonicalization_version != CANONICALIZATION_VERSION_V1
        || config.seed_derivation_version != SEED_DERIVATION_VERSION_V1
    {
        return Err(SelectionReceiptV2Error(
            "configuration selector/canonicalization/seed algorithm is unsupported".into(),
        ));
    }
    let weights = [
        config.risk_weight,
        config.uncertainty_weight,
        config.evsi_weight,
    ];
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
        || (weights.iter().sum::<f64>() - 1.0).abs() > 1.0e-9
    {
        return Err(SelectionReceiptV2Error(
            "configuration score weights must be finite, non-negative and sum to one".into(),
        ));
    }
    if config.total_cost_budget == 0
        || config.exploit_cost_budget + config.audit_cost_budget > config.total_cost_budget
        || config.exploit_cost_budget > config.total_cost_budget
        || config.audit_cost_budget > config.total_cost_budget
    {
        return Err(SelectionReceiptV2Error(
            "configuration cost budgets are inconsistent".into(),
        ));
    }
    if config.risk_weight > 0.0 && config.score_provenance.risk_score_semantics_id.trim().is_empty()
    {
        return Err(SelectionReceiptV2Error(
            "risk score semantics are required for nonzero risk weight".into(),
        ));
    }
    if config.uncertainty_weight > 0.0
        && (config.score_provenance.uncertainty_score_kind == UncertaintyScoreKindV2::NotUsed
            || config
                .score_provenance
                .uncertainty_score_semantics_id
                .trim()
                .is_empty())
    {
        return Err(SelectionReceiptV2Error(
            "uncertainty provenance is required for nonzero uncertainty weight".into(),
        ));
    }
    if config.evsi_weight > 0.0
        && (config.score_provenance.evsi_score_kind == EvsiScoreKindV2::NotUsed
            || config.score_provenance.evsi_score_semantics_id.trim().is_empty())
    {
        return Err(SelectionReceiptV2Error(
            "EVSI provenance is required for nonzero EVSI weight".into(),
        ));
    }
    validate_binary_measurand_semantics_v1(&config.binary_semantics).map_err(|error| {
        SelectionReceiptV2Error(format!("binary measurand semantics rejected: {error}"))
    })?;
    Ok(())
}

fn canonicalize_candidates(
    candidates: &[MetrologyAuditCandidateV1],
) -> Result<Vec<MetrologyAuditCandidateV1>, SelectionReceiptV2Error> {
    if candidates.is_empty() {
        return Err(SelectionReceiptV2Error(
            "candidate score packet must be non-empty".into(),
        ));
    }
    let mut sorted = candidates.to_vec();
    sorted.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    let mut identities = BTreeSet::new();
    for (index, candidate) in sorted.iter().enumerate() {
        if candidate.candidate_id.trim().is_empty()
            || candidate.subject_id.trim().is_empty()
            || candidate.stratum_id.trim().is_empty()
            || candidate.target_measurand_id
                != candidates[0].target_measurand_id
            || !identities.insert(candidate.candidate_id.as_str())
        {
            return Err(SelectionReceiptV2Error(format!(
                "candidate_scores[{index}]: invalid or duplicate candidate identity"
            )));
        }
    }
    Ok(sorted)
}

fn policy_from_receipt_inputs(
    config: &MetrologySelectionPolicyConfigV2,
    candidates: Vec<MetrologyAuditCandidateV1>,
    configuration_sha256: String,
) -> MetrologyAuditPolicyV1 {
    MetrologyAuditPolicyV1 {
        schema: POLICY_SCHEMA_V1.into(),
        policy_id: config.policy_id.clone(),
        model_sha256: config.model_sha256.clone(),
        prediction_packet_sha256: config.prediction_packet_sha256.clone(),
        policy_config_sha256: configuration_sha256,
        binary_semantics: config.binary_semantics.clone(),
        total_cost_budget: config.total_cost_budget,
        exploit_cost_budget: config.exploit_cost_budget,
        audit_cost_budget: config.audit_cost_budget,
        risk_weight: config.risk_weight,
        uncertainty_weight: config.uncertainty_weight,
        evsi_weight: config.evsi_weight,
        audit_quota_by_stratum: config.audit_quota_by_stratum.clone(),
        seed: config.seed,
        candidates,
    }
}

fn receipt_digest(
    receipt: &MetrologySelectionReceiptV2,
) -> Result<String, SelectionReceiptV2Error> {
    let mut material = receipt.clone();
    material.receipt_sha256.clear();
    json_sha256(&material)
}

/// Builds a selection receipt from the complete score inputs and a frozen
/// policy configuration. Candidate IDs are canonicalized before selection.
pub fn build_selection_receipt_v2(
    configuration: &MetrologySelectionPolicyConfigV2,
    candidates: &[MetrologyAuditCandidateV1],
) -> Result<MetrologySelectionReceiptV2, SelectionReceiptV2Error> {
    validate_configuration(configuration)?;
    let canonical_candidates = canonicalize_candidates(candidates)?;
    let candidate_digest = json_sha256(&canonical_candidates)?;
    if candidate_digest != configuration.prediction_packet_sha256 {
        return Err(SelectionReceiptV2Error(
            "candidate score packet digest does not match configuration".into(),
        ));
    }
    for (index, candidate) in canonical_candidates.iter().enumerate() {
        if candidate.target_measurand_id != configuration.binary_semantics.target_measurand_id {
            return Err(SelectionReceiptV2Error(format!(
                "candidate_scores[{index}].target_measurand_id differs from frozen semantics"
            )));
        }
    }
    let configuration_sha256 = json_sha256(configuration)?;
    let policy = policy_from_receipt_inputs(
        configuration,
        canonical_candidates.clone(),
        configuration_sha256.clone(),
    );
    let selection_report = select_metrology_audit_v1(&policy)
        .map_err(|error| SelectionReceiptV2Error(format!("selector rejected policy: {error}")))?;
    let mut receipt = MetrologySelectionReceiptV2 {
        schema: SELECTION_RECEIPT_V2_SCHEMA.into(),
        configuration: configuration.clone(),
        candidate_scores: canonical_candidates,
        configuration_sha256,
        selection_report,
        attestation_status: SelectionReceiptAttestationV2::Unattested,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = receipt_digest(&receipt)?;
    verify_selection_receipt_v2(&receipt)?;
    Ok(receipt)
}

/// Reconstructs the policy from the embedded score inputs and configuration,
/// reruns the selector, and requires exact report equality.
pub fn verify_selection_receipt_v2(
    receipt: &MetrologySelectionReceiptV2,
) -> Result<(), SelectionReceiptV2Error> {
    if receipt.schema != SELECTION_RECEIPT_V2_SCHEMA {
        return Err(SelectionReceiptV2Error("receipt.schema: unsupported".into()));
    }
    validate_configuration(&receipt.configuration)?;
    if receipt.attestation_status != SelectionReceiptAttestationV2::Unattested {
        return Err(SelectionReceiptV2Error(
            "unsupported receipt attestation state".into(),
        ));
    }
    if json_sha256(&receipt.configuration)? != receipt.configuration_sha256 {
        return Err(SelectionReceiptV2Error(
            "configuration digest does not match embedded policy configuration".into(),
        ));
    }
    let canonical_candidates = canonicalize_candidates(&receipt.candidate_scores)?;
    if canonical_candidates != receipt.candidate_scores {
        return Err(SelectionReceiptV2Error(
            "candidate score rows are not in canonical candidate-ID order".into(),
        ));
    }
    if json_sha256(&canonical_candidates)? != receipt.configuration.prediction_packet_sha256 {
        return Err(SelectionReceiptV2Error(
            "candidate score rows do not match prediction packet digest".into(),
        ));
    }
    for (index, candidate) in canonical_candidates.iter().enumerate() {
        if candidate.target_measurand_id != receipt.configuration.binary_semantics.target_measurand_id
        {
            return Err(SelectionReceiptV2Error(format!(
                "candidate_scores[{index}].target_measurand_id differs from frozen semantics"
            )));
        }
    }
    let policy = policy_from_receipt_inputs(
        &receipt.configuration,
        canonical_candidates,
        receipt.configuration_sha256.clone(),
    );
    let replayed_selection = select_metrology_audit_v1(&policy)
        .map_err(|error| SelectionReceiptV2Error(format!("selector replay rejected: {error}")))?;
    if replayed_selection != receipt.selection_report {
        return Err(SelectionReceiptV2Error(
            "embedded selection report differs from independently replayed selector output"
                .into(),
        ));
    }
    if receipt_digest(receipt)? != receipt.receipt_sha256 {
        return Err(SelectionReceiptV2Error(
            "receipt digest does not match embedded receipt payload".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::metrology_semantics::{
        BinaryMeasurandSemanticsV1, PositiveConditionV1, SEMANTICS_SCHEMA_V1,
        seal_binary_measurand_semantics_v1,
    };
    use std::collections::BTreeMap;

    fn semantics() -> BinaryMeasurandSemanticsV1 {
        let mut value = BinaryMeasurandSemanticsV1 {
            schema: SEMANTICS_SCHEMA_V1.into(),
            semantics_id: "receipt-v2-defect-rule".into(),
            revision: 1,
            target_measurand_id: "defect-score".into(),
            unit: "synthetic_unit".into(),
            positive_class_name: "defect".into(),
            negative_class_name: "pass".into(),
            positive_condition: PositiveConditionV1::GreaterThanOrEqual,
            threshold: 0.62,
            authority_ref: "synthetic-test-only".into(),
            semantics_sha256: String::new(),
        };
        seal_binary_measurand_semantics_v1(&mut value).unwrap();
        value
    }

    fn candidates() -> Vec<MetrologyAuditCandidateV1> {
        (0..40)
            .map(|index| MetrologyAuditCandidateV1 {
                candidate_id: format!("wafer-{index:03}"),
                subject_id: format!("wafer-{index:03}"),
                stratum_id: format!("tool-{:02}", index % 2),
                target_measurand_id: "defect-score".into(),
                predicted_defect_probability: (index as f64 / 50.0).clamp(0.01, 0.95),
                epistemic_uncertainty: 0.25,
                expected_value_of_information: 0.10,
                measurement_cost: 1,
            })
            .collect()
    }

    fn configuration(
        candidates: &[MetrologyAuditCandidateV1],
    ) -> MetrologySelectionPolicyConfigV2 {
        let mut quotas = BTreeMap::new();
        quotas.insert("tool-00".into(), 4);
        quotas.insert("tool-01".into(), 4);
        MetrologySelectionPolicyConfigV2 {
            schema: SELECTION_POLICY_CONFIG_V2_SCHEMA.into(),
            producer_ref: "synthetic-benchmark-producer".into(),
            run_id: "receipt-v2-run-001".into(),
            policy_id: "risk-plus-audit".into(),
            model_sha256: "a".repeat(64),
            prediction_packet_sha256: json_sha256(&candidates).unwrap(),
            binary_semantics: semantics(),
            total_cost_budget: 16,
            exploit_cost_budget: 8,
            audit_cost_budget: 8,
            risk_weight: 0.6,
            uncertainty_weight: 0.2,
            evsi_weight: 0.2,
            audit_quota_by_stratum: quotas,
            seed: 23,
            selector_algorithm_id: SELECTOR_ALGORITHM_ID_V1.into(),
            selector_algorithm_version: SELECTOR_ALGORITHM_VERSION_V1.into(),
            canonicalization_version: CANONICALIZATION_VERSION_V1.into(),
            seed_derivation_version: SEED_DERIVATION_VERSION_V1.into(),
            score_provenance: MetrologyScoreProvenanceV2 {
                risk_score_semantics_id: "calibrated-defect-risk.v1".into(),
                uncertainty_score_kind: UncertaintyScoreKindV2::Epistemic,
                uncertainty_score_semantics_id: "posterior-epistemic-uncertainty.v1".into(),
                evsi_score_kind: EvsiScoreKindV2::HeuristicProxy,
                evsi_score_semantics_id: "entropy-risk-proxy.v1".into(),
            },
        }
    }

    #[test]
    fn receipt_recomputes_scores_and_selection_from_full_inputs() {
        let candidate_scores = candidates();
        let config = configuration(&candidate_scores);
        let receipt = build_selection_receipt_v2(&config, &candidate_scores).unwrap();
        verify_selection_receipt_v2(&receipt).unwrap();
        assert_eq!(receipt.selection_report.population_count, 40);
        assert_eq!(receipt.selection_report.total_selected_cost, 16);
        assert_eq!(
            receipt.attestation_status,
            SelectionReceiptAttestationV2::Unattested
        );
    }

    #[test]
    fn changed_score_without_packet_digest_update_is_rejected() {
        let candidate_scores = candidates();
        let config = configuration(&candidate_scores);
        let mut receipt = build_selection_receipt_v2(&config, &candidate_scores).unwrap();
        receipt.candidate_scores[0].predicted_defect_probability += 0.01;
        assert!(verify_selection_receipt_v2(&receipt).is_err());
    }

    #[test]
    fn resealed_selection_tampering_is_rejected_by_selector_replay() {
        let candidate_scores = candidates();
        let config = configuration(&candidate_scores);
        let mut receipt = build_selection_receipt_v2(&config, &candidate_scores).unwrap();
        let selected = receipt
            .selection_report
            .rows
            .iter()
            .position(|row| {
                matches!(
                    &row.method,
                    crate::benchmark::metrology_audit::MetrologySelectionMethodV1::RandomizedAudit
                )
            })
            .unwrap();
        let not_selected = receipt
            .selection_report
            .rows
            .iter()
            .position(|row| {
                matches!(
                    &row.method,
                    crate::benchmark::metrology_audit::MetrologySelectionMethodV1::NotSelected
                )
            })
            .unwrap();
        receipt.selection_report.rows.swap(selected, not_selected);
        receipt.selection_report.rows[selected].selected = false;
        receipt.selection_report.rows[selected].method =
            crate::benchmark::metrology_audit::MetrologySelectionMethodV1::NotSelected;
        receipt.selection_report.rows[not_selected].selected = true;
        receipt.selection_report.rows[not_selected].method =
            crate::benchmark::metrology_audit::MetrologySelectionMethodV1::RandomizedAudit;
        receipt.selection_report.selection_sha256 =
            String::new();
        let mut selection_material = receipt.selection_report.clone();
        selection_material.selection_sha256.clear();
        receipt.selection_report.selection_sha256 =
            json_sha256(&selection_material).unwrap();
        receipt.receipt_sha256.clear();
        receipt.receipt_sha256 = receipt_digest(&receipt).unwrap();
        assert!(verify_selection_receipt_v2(&receipt).is_err());
    }

    #[test]
    fn proxy_evsi_cannot_be_mislabeled_as_unused_or_decision_theoretic() {
        let candidate_scores = candidates();
        let mut config = configuration(&candidate_scores);
        config.score_provenance.evsi_score_kind = EvsiScoreKindV2::NotUsed;
        assert!(build_selection_receipt_v2(&config, &candidate_scores).is_err());
    }

    #[test]
    fn candidate_order_is_canonicalized_and_digest_bound() {
        let mut candidate_scores = candidates();
        candidate_scores.reverse();
        let config = configuration(&candidate_scores);
        let receipt = build_selection_receipt_v2(&config, &candidate_scores).unwrap();
        assert!(receipt
            .candidate_scores
            .windows(2)
            .all(|rows| rows[0].candidate_id < rows[1].candidate_id));
        verify_selection_receipt_v2(&receipt).unwrap();
    }
}
