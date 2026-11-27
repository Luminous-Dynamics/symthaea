#![forbid(unsafe_code)]

//! Typed handoff from CMAQ-030 recoverability-margin calibration to CMAQ-031
//! anytime-valid drift monitoring.
//!
//! The bridge deliberately does not authenticate signatures or prove RNG quality.
//! It validates exact data correspondence and digest syntax after a separately
//! governed evidence-ingestion boundary has authenticated the supplied receipts.

use std::collections::{HashMap, HashSet};

use cmaq_030_recoverability_margin::{CalibrationResult, RecoveryMarginBlockSpec};
use cmaq_031_anytime_calibration_monitor::{CalibrationRankReference, CalibrationRankScore};

/// Binding produced by the external artifact-sealing/evidence-ingestion boundary.
///
/// `scope_spec` must be the exact CMAQ-030 spec covered by `scope_sha256`.
/// The upstream verifier, not this bridge, authenticates that relationship.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationBinding {
    pub calibration_profile_sha256: String,
    pub calibration_manifest_sha256: String,
    pub scope_sha256: String,
    pub scope_spec: RecoveryMarginBlockSpec,
}

/// External attestation for one independently calibrated cluster.
///
/// The aggregate-score receipt must bind the cluster ID, score, candidate block,
/// reference block manifest and reference-margin receipt. The tie-break receipt
/// must attest an independently randomized key generated under the same rule used
/// for deployment observations. This bridge checks correspondence and digest shape;
/// signature, signer authority and randomness verification remain upstream duties.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClusterRankAttestation {
    pub independent_cluster_id: String,
    pub score_units: u64,
    pub block_manifest_sha256: String,
    pub candidate_margin_block_sha256: String,
    pub reference_margin_receipt_sha256: String,
    pub aggregate_score_receipt_sha256: String,
    pub tie_break_key: u64,
    pub tie_break_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BridgeError {
    EmptyIdentity(&'static str),
    InvalidSha256(&'static str),
    CalibrationManifestMismatch,
    ScopeSpecificationMismatch,
    CalibrationProfileBindingMismatch,
    ScoreInventoryMismatch,
    MissingAttestation,
    UnexpectedAttestation,
    DuplicateAttestation,
    DuplicateTieBreakKey,
    AttestedScoreMismatch(&'static str),
}

/// Constructs the exact immutable rank reference required by CMAQ-031.
///
/// No score is recomputed here: every score is copied from CMAQ-030's calibrated
/// per-cluster result. Attestations must cover that inventory one-to-one.
pub fn build_calibration_rank_reference(
    calibration: &CalibrationResult,
    binding: &CalibrationBinding,
    attestations: &[ClusterRankAttestation],
) -> Result<CalibrationRankReference, BridgeError> {
    validate_sha256(
        &binding.calibration_profile_sha256,
        "calibration_profile_sha256",
    )?;
    validate_sha256(
        &binding.calibration_manifest_sha256,
        "calibration_manifest_sha256",
    )?;
    validate_sha256(&binding.scope_sha256, "scope_sha256")?;

    if binding.calibration_manifest_sha256 != calibration.qualified.calibration_manifest_sha256 {
        return Err(BridgeError::CalibrationManifestMismatch);
    }
    if binding.scope_spec != calibration.qualified.spec {
        return Err(BridgeError::ScopeSpecificationMismatch);
    }
    if calibration.cluster_scores.len() != calibration.qualified.calibration_cluster_count
        || attestations.len() != calibration.cluster_scores.len()
    {
        return Err(BridgeError::ScoreInventoryMismatch);
    }

    let mut by_cluster = HashMap::with_capacity(attestations.len());
    for attestation in attestations {
        if attestation.independent_cluster_id.trim().is_empty() {
            return Err(BridgeError::EmptyIdentity(
                "attestation.independent_cluster_id",
            ));
        }
        validate_sha256(
            &attestation.block_manifest_sha256,
            "attestation.block_manifest_sha256",
        )?;
        validate_sha256(
            &attestation.candidate_margin_block_sha256,
            "attestation.candidate_margin_block_sha256",
        )?;
        validate_sha256(
            &attestation.reference_margin_receipt_sha256,
            "attestation.reference_margin_receipt_sha256",
        )?;
        validate_sha256(
            &attestation.aggregate_score_receipt_sha256,
            "attestation.aggregate_score_receipt_sha256",
        )?;
        validate_sha256(
            &attestation.tie_break_receipt_sha256,
            "attestation.tie_break_receipt_sha256",
        )?;
        if by_cluster
            .insert(attestation.independent_cluster_id.as_str(), attestation)
            .is_some()
        {
            return Err(BridgeError::DuplicateAttestation);
        }
    }

    let mut seen_tie_keys = HashSet::with_capacity(attestations.len());
    let mut scores = Vec::with_capacity(calibration.cluster_scores.len());
    for score in &calibration.cluster_scores {
        if score.independent_cluster_id.trim().is_empty() {
            return Err(BridgeError::EmptyIdentity(
                "calibration.independent_cluster_id",
            ));
        }
        validate_sha256(
            &score.block_manifest_sha256,
            "calibration.block_manifest_sha256",
        )?;
        validate_sha256(
            &score.candidate_margin_block_sha256,
            "calibration.candidate_margin_block_sha256",
        )?;
        validate_sha256(
            &score.reference_margin_receipt_sha256,
            "calibration.reference_margin_receipt_sha256",
        )?;

        let attestation = by_cluster
            .remove(score.independent_cluster_id.as_str())
            .ok_or(BridgeError::MissingAttestation)?;

        if attestation.score_units != score.score_units {
            return Err(BridgeError::AttestedScoreMismatch("score_units"));
        }
        if attestation.block_manifest_sha256 != score.block_manifest_sha256 {
            return Err(BridgeError::AttestedScoreMismatch("block_manifest_sha256"));
        }
        if attestation.candidate_margin_block_sha256 != score.candidate_margin_block_sha256 {
            return Err(BridgeError::AttestedScoreMismatch(
                "candidate_margin_block_sha256",
            ));
        }
        if attestation.reference_margin_receipt_sha256 != score.reference_margin_receipt_sha256 {
            return Err(BridgeError::AttestedScoreMismatch(
                "reference_margin_receipt_sha256",
            ));
        }
        if !seen_tie_keys.insert(attestation.tie_break_key) {
            return Err(BridgeError::DuplicateTieBreakKey);
        }

        scores.push(CalibrationRankScore {
            independent_cluster_id: score.independent_cluster_id.clone(),
            score_units: score.score_units,
            tie_break_key: attestation.tie_break_key,
            block_manifest_sha256: score.block_manifest_sha256.clone(),
            aggregate_score_receipt_sha256: attestation.aggregate_score_receipt_sha256.clone(),
            tie_break_receipt_sha256: attestation.tie_break_receipt_sha256.clone(),
        });
    }

    if !by_cluster.is_empty() {
        return Err(BridgeError::UnexpectedAttestation);
    }

    Ok(CalibrationRankReference {
        calibration_profile_sha256: binding.calibration_profile_sha256.clone(),
        calibration_manifest_sha256: binding.calibration_manifest_sha256.clone(),
        scope_sha256: binding.scope_sha256.clone(),
        scores,
    })
}

fn validate_sha256(value: &str, field: &'static str) -> Result<(), BridgeError> {
    if is_sha256_hex(value) {
        Ok(())
    } else {
        Err(BridgeError::InvalidSha256(field))
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cmaq_030_recoverability_margin::{
        CalibrationClusterScore, CalibrationDiagnostics, QualifiedMarginCalibration,
    };
    use cmaq_031_anytime_calibration_monitor::{
        AnytimeCalibrationMonitor, DeploymentRankObservation, MonitorUpdate, PreviousOutcome,
    };

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn spec() -> RecoveryMarginBlockSpec {
        RecoveryMarginBlockSpec {
            contract_sha256: digest('1'),
            source_manifest_sha256: digest('2'),
            candidate_checkpoint_sha256: digest('3'),
            plant_model_sha256: digest('4'),
            context_scope_sha256: digest('5'),
            recovery_portfolio_sha256: digest('6'),
            recovery_certificate_manifest_sha256: digest('7'),
            margin_metric_sha256: digest('8'),
            selection_protocol_sha256: digest('9'),
            rollouts_per_cluster: 2,
            action_query_cell_count: 3,
            horizon_steps: 4,
            required_independent_domains: 2,
        }
    }

    fn calibration() -> CalibrationResult {
        CalibrationResult {
            qualified: QualifiedMarginCalibration {
                spec: spec(),
                calibration_manifest_sha256: digest('a'),
                optimism_radius_units: 12,
                quantile_rank_one_based: 3,
                calibration_cluster_count: 3,
            },
            diagnostics: CalibrationDiagnostics {
                calibration_family_count: 2,
                training_cluster_count: 20,
                validation_cluster_count: 10,
                held_out_evaluation_cluster_count: 10,
            },
            cluster_scores: vec![
                CalibrationClusterScore {
                    independent_cluster_id: "cluster-1".into(),
                    score_units: 3,
                    block_manifest_sha256: digest('b'),
                    candidate_margin_block_sha256: digest('c'),
                    reference_margin_receipt_sha256: digest('d'),
                },
                CalibrationClusterScore {
                    independent_cluster_id: "cluster-2".into(),
                    score_units: 7,
                    block_manifest_sha256: digest('e'),
                    candidate_margin_block_sha256: digest('f'),
                    reference_margin_receipt_sha256: digest('0'),
                },
                CalibrationClusterScore {
                    independent_cluster_id: "cluster-3".into(),
                    score_units: 11,
                    block_manifest_sha256: digest('1'),
                    candidate_margin_block_sha256: digest('2'),
                    reference_margin_receipt_sha256: digest('3'),
                },
            ],
        }
    }

    fn binding() -> CalibrationBinding {
        CalibrationBinding {
            calibration_profile_sha256: digest('4'),
            calibration_manifest_sha256: digest('a'),
            scope_sha256: digest('5'),
            scope_spec: spec(),
        }
    }

    fn attestations() -> Vec<ClusterRankAttestation> {
        calibration()
            .cluster_scores
            .iter()
            .enumerate()
            .map(|(index, score)| {
                let letter = char::from(b'g' + index as u8);
                ClusterRankAttestation {
                    independent_cluster_id: score.independent_cluster_id.clone(),
                    score_units: score.score_units,
                    block_manifest_sha256: score.block_manifest_sha256.clone(),
                    candidate_margin_block_sha256: score.candidate_margin_block_sha256.clone(),
                    reference_margin_receipt_sha256: score.reference_margin_receipt_sha256.clone(),
                    aggregate_score_receipt_sha256: digest(letter),
                    tie_break_key: 100 + index as u64,
                    tie_break_receipt_sha256: digest(char::from(b'k' + index as u8)),
                }
            })
            .collect()
    }

    #[test]
    fn bridges_exact_cluster_scores_and_attestations() {
        let calibration = calibration();
        let reference =
            build_calibration_rank_reference(&calibration, &binding(), &attestations()).unwrap();
        assert_eq!(reference.scores.len(), 3);
        for (rank_score, calibration_score) in
            reference.scores.iter().zip(&calibration.cluster_scores)
        {
            assert_eq!(
                rank_score.independent_cluster_id,
                calibration_score.independent_cluster_id
            );
            assert_eq!(rank_score.score_units, calibration_score.score_units);
            assert_eq!(
                rank_score.block_manifest_sha256,
                calibration_score.block_manifest_sha256
            );
        }
    }

    #[test]
    fn output_is_accepted_by_the_anytime_monitor() {
        let reference =
            build_calibration_rank_reference(&calibration(), &binding(), &attestations()).unwrap();
        let mut monitor = AnytimeCalibrationMonitor::new(reference, 1, 20).unwrap();
        let result = monitor
            .observe(DeploymentRankObservation {
                previous_outcome: PreviousOutcome::Reconciled,
                calibration_profile_sha256: digest('4'),
                scope_sha256: digest('5'),
                independent_cluster_id: "deployment-cluster-4".into(),
                score_units: 8,
                tie_break_key: 900,
                block_manifest_sha256: digest('6'),
                candidate_margin_block_sha256: digest('7'),
                aggregate_score_receipt_sha256: digest('8'),
                tie_break_receipt_sha256: digest('9'),
            })
            .unwrap();
        assert!(matches!(result, MonitorUpdate::Accepted { .. }));
    }

    #[test]
    fn missing_cluster_attestation_is_rejected() {
        let mut attestations = attestations();
        attestations.pop();
        assert_eq!(
            build_calibration_rank_reference(&calibration(), &binding(), &attestations),
            Err(BridgeError::ScoreInventoryMismatch)
        );
    }

    #[test]
    fn duplicate_cluster_attestation_is_rejected() {
        let mut attestations = attestations();
        attestations[1].independent_cluster_id = attestations[0].independent_cluster_id.clone();
        assert_eq!(
            build_calibration_rank_reference(&calibration(), &binding(), &attestations),
            Err(BridgeError::DuplicateAttestation)
        );
    }

    #[test]
    fn unexpected_cluster_attestation_is_rejected() {
        let mut attestations = attestations();
        attestations[2].independent_cluster_id = "unexpected-cluster".into();
        assert_eq!(
            build_calibration_rank_reference(&calibration(), &binding(), &attestations),
            Err(BridgeError::MissingAttestation)
        );
    }

    #[test]
    fn duplicate_randomized_tie_key_is_rejected() {
        let mut attestations = attestations();
        attestations[1].tie_break_key = attestations[0].tie_break_key;
        assert_eq!(
            build_calibration_rank_reference(&calibration(), &binding(), &attestations),
            Err(BridgeError::DuplicateTieBreakKey)
        );
    }

    #[test]
    fn score_receipt_must_match_exact_cluster_score() {
        let mut attestations = attestations();
        attestations[0].score_units += 1;
        assert_eq!(
            build_calibration_rank_reference(&calibration(), &binding(), &attestations),
            Err(BridgeError::AttestedScoreMismatch("score_units"))
        );
    }

    #[test]
    fn candidate_block_binding_must_match() {
        let mut attestations = attestations();
        attestations[0].candidate_margin_block_sha256 = digest('z');
        assert_eq!(
            build_calibration_rank_reference(&calibration(), &binding(), &attestations),
            Err(BridgeError::AttestedScoreMismatch(
                "candidate_margin_block_sha256"
            ))
        );
    }

    #[test]
    fn calibration_manifest_digest_must_match() {
        let mut binding = binding();
        binding.calibration_manifest_sha256 = digest('z');
        assert_eq!(
            build_calibration_rank_reference(&calibration(), &binding, &attestations()),
            Err(BridgeError::CalibrationManifestMismatch)
        );
    }

    #[test]
    fn scope_specification_must_match() {
        let mut binding = binding();
        binding.scope_spec.context_scope_sha256 = digest('z');
        assert_eq!(
            build_calibration_rank_reference(&calibration(), &binding, &attestations()),
            Err(BridgeError::ScopeSpecificationMismatch)
        );
    }

    #[test]
    fn malformed_receipt_digest_is_rejected() {
        let mut attestations = attestations();
        attestations[0].tie_break_receipt_sha256 = "not-a-digest".into();
        assert_eq!(
            build_calibration_rank_reference(&calibration(), &binding(), &attestations),
            Err(BridgeError::InvalidSha256(
                "attestation.tie_break_receipt_sha256"
            ))
        );
    }
}
