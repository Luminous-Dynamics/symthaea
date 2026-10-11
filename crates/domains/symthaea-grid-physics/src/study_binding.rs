// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-project binding for energy-study evidence.
//!
//! This adapter composes the existing frozen scenario runner and independent
//! receipt checker; it does not introduce another dispatch path. A producer
//! (for example, Mycelix after its manifest/artifact verifier has passed)
//! supplies the manifest, case, and artifact-set digests. This module validates
//! their encoding and carries them into each step envelope, but does not
//! re-verify the external manifest, authenticate its source, or prove that the
//! supplied digest came from a trusted producer.

use serde::{Deserialize, Serialize};

use crate::qualification::{
    FrozenEnergyScenario, PolicyComparisonReceipt, PolicyRunReceipt, QualificationError,
    VerifiedPolicyReceipt, compare_policy_receipts,
};
use crate::scheduling::ScenarioStepReceipt;

pub const ENERGY_STUDY_BINDING_SCHEMA_VERSION: u16 = 1;

/// External identity emitted by the shared manifest/artifact contract.
///
/// manifest_sha256 identifies the declared manifest, case_sha256 identifies
/// the model/policy-independent case, and artifact_set_sha256 identifies the
/// declared byte set. These values are opaque here; their producer must
/// establish their meaning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnergyStudyBindingV1 {
    pub schema_version: u16,
    pub manifest_sha256: String,
    pub case_sha256: String,
    pub artifact_set_sha256: String,
}

impl EnergyStudyBindingV1 {
    /// Validate schema and digest encoding, not the truth of the external claim.
    pub fn validate(&self) -> Result<(), EnergyStudyBindingError> {
        if self.schema_version != ENERGY_STUDY_BINDING_SCHEMA_VERSION {
            return Err(EnergyStudyBindingError::UnsupportedSchemaVersion);
        }
        for (field, digest) in [
            ("manifest_sha256", self.manifest_sha256.as_str()),
            ("case_sha256", self.case_sha256.as_str()),
            ("artifact_set_sha256", self.artifact_set_sha256.as_str()),
        ] {
            if digest.len() != 64
                || !digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err(EnergyStudyBindingError::InvalidDigest(field));
            }
        }
        Ok(())
    }
}

/// One scheduler step with the external identity explicitly carried beside it.
///
/// The step payload is copied from the scheduler's existing raw receipt. The
/// checker compares it exactly with the corresponding step in the full run
/// receipt, so a consumer cannot swap or edit a bound step independently.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundScenarioStepReceiptV1 {
    pub step_ordinal: usize,
    pub study_binding: EnergyStudyBindingV1,
    pub scheduler_step_receipt: ScenarioStepReceipt,
}

/// Evidence envelope binding an existing policy run to an external identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundPolicyRunReceiptV1 {
    pub schema_version: u16,
    pub study_binding: EnergyStudyBindingV1,
    pub run_receipt: PolicyRunReceipt,
    pub bound_steps: Vec<BoundScenarioStepReceiptV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundVerifiedPolicyReceiptV1 {
    pub study_binding: EnergyStudyBindingV1,
    pub verified_metrics: VerifiedPolicyReceipt,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundPolicyComparisonReceiptV1 {
    pub study_binding: EnergyStudyBindingV1,
    pub comparison: PolicyComparisonReceipt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnergyStudyBindingError {
    UnsupportedSchemaVersion,
    InvalidDigest(&'static str),
    InvalidStepSequence,
    StudyBindingMismatch,
    StepBindingMismatch,
    StepPayloadMismatch,
    UnderlyingReceipt(QualificationError),
    UnderlyingComparison(QualificationError),
    EvidenceSerializationFailed,
}

impl std::fmt::Display for EnergyStudyBindingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion => {
                f.write_str("unsupported energy-study binding schema version")
            }
            Self::InvalidDigest(field) => {
                write!(f, "energy-study binding field {field} is not lowercase SHA-256")
            }
            Self::InvalidStepSequence => {
                f.write_str("bound step receipt count or ordinal sequence is invalid")
            }
            Self::StudyBindingMismatch => {
                f.write_str("policy receipts do not share the same external study binding")
            }
            Self::StepBindingMismatch => {
                f.write_str("a step receipt does not carry the run's study binding")
            }
            Self::StepPayloadMismatch => {
                f.write_str("a bound step payload differs from the original scheduler receipt")
            }
            Self::UnderlyingReceipt(error) => {
                write!(f, "underlying scheduler receipt failed independent verification: {error}")
            }
            Self::UnderlyingComparison(error) => {
                write!(f, "underlying policy comparison failed verification: {error:?}")
            }
            Self::EvidenceSerializationFailed => {
                f.write_str("failed to serialize the bound energy-study evidence packet")
            }
        }
    }
}

impl std::error::Error for EnergyStudyBindingError {}

/// Attach a validated external identity to each step of an existing scheduler
/// run. The underlying run is independently checked before it is bound.
pub fn bind_policy_run_receipt(
    scenario: &FrozenEnergyScenario,
    run_receipt: PolicyRunReceipt,
    study_binding: EnergyStudyBindingV1,
) -> Result<BoundPolicyRunReceiptV1, EnergyStudyBindingError> {
    study_binding.validate()?;
    scenario
        .verify_receipt(&run_receipt)
        .map_err(EnergyStudyBindingError::UnderlyingReceipt)?;

    let bound_steps = run_receipt
        .trace
        .steps
        .iter()
        .cloned()
        .enumerate()
        .map(|(step_ordinal, scheduler_step_receipt)| BoundScenarioStepReceiptV1 {
            step_ordinal,
            study_binding: study_binding.clone(),
            scheduler_step_receipt,
        })
        .collect();

    Ok(BoundPolicyRunReceiptV1 {
        schema_version: ENERGY_STUDY_BINDING_SCHEMA_VERSION,
        study_binding,
        run_receipt,
        bound_steps,
    })
}

/// Re-check the original raw report and the identity carried by every step.
/// This does not authenticate the external manifest or recompute its
/// artifact-set digest.
pub fn verify_bound_policy_receipt(
    scenario: &FrozenEnergyScenario,
    bound: &BoundPolicyRunReceiptV1,
) -> Result<BoundVerifiedPolicyReceiptV1, EnergyStudyBindingError> {
    if bound.schema_version != ENERGY_STUDY_BINDING_SCHEMA_VERSION {
        return Err(EnergyStudyBindingError::UnsupportedSchemaVersion);
    }
    bound.study_binding.validate()?;

    let verified_metrics = scenario
        .verify_receipt(&bound.run_receipt)
        .map_err(EnergyStudyBindingError::UnderlyingReceipt)?;

    if bound.bound_steps.len() != bound.run_receipt.trace.steps.len() {
        return Err(EnergyStudyBindingError::InvalidStepSequence);
    }
    for (expected_ordinal, bound_step) in bound.bound_steps.iter().enumerate() {
        if bound_step.step_ordinal != expected_ordinal {
            return Err(EnergyStudyBindingError::InvalidStepSequence);
        }
        if bound_step.study_binding != bound.study_binding {
            return Err(EnergyStudyBindingError::StepBindingMismatch);
        }
        if bound_step.scheduler_step_receipt != bound.run_receipt.trace.steps[expected_ordinal] {
            return Err(EnergyStudyBindingError::StepPayloadMismatch);
        }
    }

    Ok(BoundVerifiedPolicyReceiptV1 {
        study_binding: bound.study_binding.clone(),
        verified_metrics,
    })
}

/// Compare two policies only when both carry the identical external study,
/// case, and artifact identity. Metrics remain separate; no composite score
/// is introduced.
pub fn compare_bound_policy_receipts(
    scenario: &FrozenEnergyScenario,
    baseline: &BoundPolicyRunReceiptV1,
    candidate: &BoundPolicyRunReceiptV1,
) -> Result<BoundPolicyComparisonReceiptV1, EnergyStudyBindingError> {
    if baseline.study_binding != candidate.study_binding {
        return Err(EnergyStudyBindingError::StudyBindingMismatch);
    }
    verify_bound_policy_receipt(scenario, baseline)?;
    verify_bound_policy_receipt(scenario, candidate)?;

    let comparison = compare_policy_receipts(
        scenario,
        &baseline.run_receipt,
        &candidate.run_receipt,
    )
    .map_err(EnergyStudyBindingError::UnderlyingComparison)?;

    Ok(BoundPolicyComparisonReceiptV1 {
        study_binding: baseline.study_binding.clone(),
        comparison,
    })
}

#[derive(Debug, Serialize)]
struct BoundQualificationEvidencePacket<'a> {
    schema_version: &'static str,
    scenario: &'a FrozenEnergyScenario,
    bound_policy_receipt: &'a BoundPolicyRunReceiptV1,
    verified_metrics: BoundVerifiedPolicyReceiptV1,
}

/// Emit a bound evidence packet only after both receipt checkers succeed.
pub fn serialize_bound_evidence_packet(
    scenario: &FrozenEnergyScenario,
    bound: &BoundPolicyRunReceiptV1,
) -> Result<String, EnergyStudyBindingError> {
    let verified_metrics = verify_bound_policy_receipt(scenario, bound)?;
    let packet = BoundQualificationEvidencePacket {
        schema_version: "symthaea-bound-energy-study-evidence-v1",
        scenario,
        bound_policy_receipt: bound,
        verified_metrics,
    };
    serde_json::to_string_pretty(&packet)
        .map_err(|_| EnergyStudyBindingError::EvidenceSerializationFailed)
}

#[derive(Debug, Serialize)]
struct BoundPolicyComparisonEvidencePacket<'a> {
    schema_version: &'static str,
    scenario: &'a FrozenEnergyScenario,
    baseline: &'a BoundPolicyRunReceiptV1,
    candidate: &'a BoundPolicyRunReceiptV1,
    comparison: BoundPolicyComparisonReceiptV1,
}

/// Emit a paired packet only after both bound reports and their shared identity
/// have been checked.
pub fn serialize_bound_policy_comparison_packet(
    scenario: &FrozenEnergyScenario,
    baseline: &BoundPolicyRunReceiptV1,
    candidate: &BoundPolicyRunReceiptV1,
) -> Result<String, EnergyStudyBindingError> {
    let comparison = compare_bound_policy_receipts(scenario, baseline, candidate)?;
    let packet = BoundPolicyComparisonEvidencePacket {
        schema_version: "symthaea-bound-energy-policy-comparison-v1",
        scenario,
        baseline,
        candidate,
        comparison,
    };
    serde_json::to_string_pretty(&packet)
        .map_err(|_| EnergyStudyBindingError::EvidenceSerializationFailed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qualification::deterministic_energy_corpus;
    use crate::scheduling::{ReserveAwarePolicy, naive_greedy_policy};

    const TEST_POLICY_REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

    fn test_binding() -> EnergyStudyBindingV1 {
        EnergyStudyBindingV1 {
            schema_version: ENERGY_STUDY_BINDING_SCHEMA_VERSION,
            manifest_sha256:
                "896b4579a632dac32bafd3b2347f19f5da5a74f7173c14cf818b00a8e5c484da".into(),
            // This is a format test vector, not a claim of equivalence with
            // Symthaea's distinct native scenario-input digest.
            case_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into(),
            artifact_set_sha256:
                "74a2bb57ea351ab2f0d68502443d27a971db33293fa4c11e12a45d0afa3b30e2".into(),
        }
    }

    fn run_baseline(scenario: &FrozenEnergyScenario) -> PolicyRunReceipt {
        scenario
            .run_policy("naive-greedy", TEST_POLICY_REVISION, naive_greedy_policy)
            .unwrap()
    }

    #[test]
    fn bound_steps_carry_identity_and_preserve_raw_receipts() {
        let scenario = deterministic_energy_corpus().remove(0);
        let binding = test_binding();
        let raw = run_baseline(&scenario);
        let expected_steps = raw.trace.steps.clone();
        let bound = bind_policy_run_receipt(&scenario, raw, binding.clone()).unwrap();

        assert_eq!(bound.bound_steps.len(), expected_steps.len());
        for (index, step) in bound.bound_steps.iter().enumerate() {
            assert_eq!(step.step_ordinal, index);
            assert_eq!(step.study_binding, binding);
            assert_eq!(step.scheduler_step_receipt, expected_steps[index]);
        }

        let verified = verify_bound_policy_receipt(&scenario, &bound).unwrap();
        assert_eq!(verified.study_binding, binding);
        assert_eq!(verified.verified_metrics.step_count, expected_steps.len());

        let packet = serialize_bound_evidence_packet(&scenario, &bound).unwrap();
        assert!(packet.contains("manifest_sha256"));
        assert!(packet.contains("case_sha256"));
        assert!(packet.contains("artifact_set_sha256"));
        assert!(packet.contains("scheduler_step_receipt"));
        assert!(packet.contains("step_ordinal"));
    }

    #[test]
    fn binding_rejects_malformed_digest_and_unknown_schema() {
        let mut malformed = test_binding();
        malformed.manifest_sha256.make_ascii_uppercase();
        assert_eq!(
            malformed.validate(),
            Err(EnergyStudyBindingError::InvalidDigest("manifest_sha256"))
        );

        let mut unsupported = test_binding();
        unsupported.schema_version = 2;
        assert_eq!(
            unsupported.validate(),
            Err(EnergyStudyBindingError::UnsupportedSchemaVersion)
        );
    }

    #[test]
    fn checker_rejects_tampered_step_binding_order_and_payload() {
        let scenario = deterministic_energy_corpus().remove(0);
        let raw = run_baseline(&scenario);
        let mut bound = bind_policy_run_receipt(&scenario, raw, test_binding()).unwrap();
        bound.bound_steps[0].scheduler_step_receipt.cost_delta += 1.0;
        assert_eq!(
            verify_bound_policy_receipt(&scenario, &bound),
            Err(EnergyStudyBindingError::StepPayloadMismatch)
        );

        let raw = run_baseline(&scenario);
        let mut wrong_binding = bind_policy_run_receipt(&scenario, raw, test_binding()).unwrap();
        wrong_binding.bound_steps[0].study_binding.case_sha256 =
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".into();
        assert_eq!(
            verify_bound_policy_receipt(&scenario, &wrong_binding),
            Err(EnergyStudyBindingError::StepBindingMismatch)
        );

        let raw = run_baseline(&scenario);
        let mut wrong_order = bind_policy_run_receipt(&scenario, raw, test_binding()).unwrap();
        wrong_order.bound_steps.swap(0, 1);
        assert_eq!(
            verify_bound_policy_receipt(&scenario, &wrong_order),
            Err(EnergyStudyBindingError::InvalidStepSequence)
        );
    }

    #[test]
    fn paired_policy_comparison_requires_identical_external_study_identity() {
        let scenario = deterministic_energy_corpus().remove(0);
        let baseline = bind_policy_run_receipt(
            &scenario,
            run_baseline(&scenario),
            test_binding(),
        )
        .unwrap();
        let reserve_policy = ReserveAwarePolicy {
            daytime_reserve_soc: 0.40,
            day_start_hour: 6.0,
            day_end_hour: 17.0,
        };
        let candidate_raw = scenario
            .run_policy(
                "reserve-aware-rule",
                TEST_POLICY_REVISION,
                |t, load, generation, battery| reserve_policy.decide(t, load, generation, battery),
            )
            .unwrap();
        let mut alternate_binding = test_binding();
        alternate_binding.case_sha256 =
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".into();
        let candidate =
            bind_policy_run_receipt(&scenario, candidate_raw, alternate_binding).unwrap();

        assert_eq!(
            compare_bound_policy_receipts(&scenario, &baseline, &candidate),
            Err(EnergyStudyBindingError::StudyBindingMismatch)
        );
    }

    #[test]
    fn matching_baseline_and_candidate_keep_metrics_separate() {
        let scenario = deterministic_energy_corpus().remove(0);
        let binding = test_binding();
        let baseline = bind_policy_run_receipt(
            &scenario,
            run_baseline(&scenario),
            binding.clone(),
        )
        .unwrap();
        let reserve_policy = ReserveAwarePolicy {
            daytime_reserve_soc: 0.40,
            day_start_hour: 6.0,
            day_end_hour: 17.0,
        };
        let candidate_raw = scenario
            .run_policy(
                "reserve-aware-rule",
                TEST_POLICY_REVISION,
                |t, load, generation, battery| reserve_policy.decide(t, load, generation, battery),
            )
            .unwrap();
        let candidate =
            bind_policy_run_receipt(&scenario, candidate_raw, binding.clone()).unwrap();

        let comparison = compare_bound_policy_receipts(&scenario, &baseline, &candidate).unwrap();
        assert_eq!(comparison.study_binding, binding);
        assert_eq!(
            comparison.comparison.scenario_input_digest,
            scenario.input_digest()
        );

        let packet =
            serialize_bound_policy_comparison_packet(&scenario, &baseline, &candidate).unwrap();
        assert!(packet.contains("baseline"));
        assert!(packet.contains("candidate"));
        assert!(packet.contains("study_binding"));
        assert!(packet.contains("unserved_energy_kwh"));
    }
}
