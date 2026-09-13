// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed causal qualification for the direct Butlin GWT-1 specialist theorem.
//!
//! Experiment execution remains in the root benchmark crate. This resolver
//! independently validates all four selective-lesion rows, all four matched
//! omission controls, exact rescue, non-target specificity and held-out state
//! consequences. Stored Phase-C integration is recomputed through the root
//! benchmark's narrow production-collector verification facade.
//!
//! `Qualified` is a causal-protocol outcome only. This module does not itself
//! assign a Butlin support tier.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use symthaea::benchmarks::gwt1_causal_lesion::{
    GWT1_CAUSAL_HELDOUT_STEPS_V1, GWT1_CAUSAL_LESION_SCHEMA_V1,
    Gwt1CausalArmRawV1, Gwt1CausalObservationsV1, Gwt1CausalRowRawV1,
    Gwt1CausalSignatureV1, Gwt1CausalTargetV1, Gwt1IntegratedOutputBitsV1,
    run_gwt1_causal_lesion_v1,
};
use symthaea::benchmarks::gwt1_causal_matched_sham::{
    GWT1_CAUSAL_MATCHED_SHAM_SCHEMA_V1, Gwt1MatchedShamArmRawV1,
    Gwt1MatchedShamObservationsV1, Gwt1MatchedShamRowRawV1,
    run_gwt1_causal_matched_sham_v1,
};
use symthaea::benchmarks::gwt1_causal_verifier::recompute_gwt1_collector_v1;
use symthaea::benchmarks::gwt1_specialist_qualification::{
    GWT1_SPECIALIST_IDS_V1, Gwt1SubsystemOutputBitsV1,
};

pub const GWT1_CAUSAL_QUALIFICATION_SCHEMA_V1: &str =
    "butlin-gwt1-causal-qualification-v1";
const REQUEST_CONSOLIDATION_FLAG_V1: u32 = 1 << 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gwt1CausalQualificationOutcomeV1 {
    Qualified,
    NotDemonstrated,
    Contradicted,
    Inconclusive,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Gwt1CausalQualificationFailureV1 {
    ContractDrift { field: String },
    TargetSetMismatch { context: String, observed: Vec<String> },
    TargetIdentityMismatch { target: String },
    SignatureMismatch { target: String },
    InvocationSetMismatch { target: String, arm: String, observed: Vec<String> },
    OutputCoverageMismatch { target: String, arm: String, observed: Vec<String> },
    StoredIntegrationMismatch { target: String, arm: String },
    StoredContributorSetMismatch { target: String, arm: String },
    TargetNeutral { target: String },
    TargetSignatureAbsent { target: String },
    TargetSignatureNotExclusive { target: String, leaked_to: Vec<String> },
    NonTargetLeakage { target: String, arm: String, specialist: String },
    NoOpMismatch { target: String },
    CausalConsequenceAbsent { target: String },
    RescueInjectionMismatch { target: String },
    RescueMismatch { target: String },
    HeldoutLengthMismatch { target: String, observed: usize },
    HeldoutStepMismatch { target: String, index: usize, observed: u32 },
    HeldoutCoverageMismatch { target: String, step: u32, arm: String },
    HeldoutNonTargetLeakage { target: String, step: u32, arm: String },
    MatchedShamTargetNotNeutral { target: String },
    MatchedShamOmissionMismatch { target: String },
    MatchedShamEffect { target: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1CausalQualificationResolutionV1 {
    pub schema: String,
    pub outcome: Gwt1CausalQualificationOutcomeV1,
    pub failures: Vec<Gwt1CausalQualificationFailureV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gwt1CausalEvidenceV1 {
    pub causal_raw_bytes: Vec<u8>,
    pub matched_sham_raw_bytes: Vec<u8>,
    pub resolution: Gwt1CausalQualificationResolutionV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FailureClass {
    NotDemonstrated,
    Contradicted,
    Inconclusive,
}

fn all_specialists() -> BTreeSet<String> {
    GWT1_SPECIALIST_IDS_V1
        .iter()
        .map(|id| (*id).to_string())
        .collect()
}

fn all_targets() -> BTreeSet<String> {
    Gwt1CausalTargetV1::ALL
        .iter()
        .map(|target| target.id().to_string())
        .collect()
}

fn controls_for(target: Gwt1CausalTargetV1) -> BTreeSet<String> {
    GWT1_SPECIALIST_IDS_V1
        .iter()
        .filter(|id| **id != target.id())
        .map(|id| (*id).to_string())
        .collect()
}

fn expected_signature(target: Gwt1CausalTargetV1) -> Gwt1CausalSignatureV1 {
    match target {
        Gwt1CausalTargetV1::Drive => Gwt1CausalSignatureV1::DriveValence,
        Gwt1CausalTargetV1::Memory => Gwt1CausalSignatureV1::MemoryConsolidationFlag,
        Gwt1CausalTargetV1::Learning => Gwt1CausalSignatureV1::LearningRate,
        Gwt1CausalTargetV1::Perception => Gwt1CausalSignatureV1::PerceptionConfidence,
    }
}

fn neutral(output: &Gwt1SubsystemOutputBitsV1) -> bool {
    output.confidence_delta == 0.0f64.to_bits()
        && output.lr_modulation == 1.0f64.to_bits()
        && output.exploration_delta == 0.0f64.to_bits()
        && output.arousal_delta == 0.0f32.to_bits()
        && output.valence_delta == 0.0f32.to_bits()
        && output.flags == 0
        && output.reserved == 0
}

fn signature_present(
    signature: Gwt1CausalSignatureV1,
    output: &Gwt1SubsystemOutputBitsV1,
) -> bool {
    match signature {
        Gwt1CausalSignatureV1::DriveValence => output.valence_delta != 0.0f32.to_bits(),
        Gwt1CausalSignatureV1::MemoryConsolidationFlag => {
            output.flags & REQUEST_CONSOLIDATION_FLAG_V1 != 0
        }
        Gwt1CausalSignatureV1::LearningRate => output.lr_modulation != 1.0f64.to_bits(),
        Gwt1CausalSignatureV1::PerceptionConfidence => {
            output.confidence_delta != 0.0f64.to_bits()
        }
    }
}

fn signature_changed(
    signature: Gwt1CausalSignatureV1,
    baseline: Gwt1IntegratedOutputBitsV1,
    lesion: Gwt1IntegratedOutputBitsV1,
) -> bool {
    match signature {
        Gwt1CausalSignatureV1::DriveValence => baseline.valence_delta != lesion.valence_delta,
        Gwt1CausalSignatureV1::MemoryConsolidationFlag => {
            (baseline.flags & REQUEST_CONSOLIDATION_FLAG_V1)
                != (lesion.flags & REQUEST_CONSOLIDATION_FLAG_V1)
        }
        Gwt1CausalSignatureV1::LearningRate => baseline.lr_modulation != lesion.lr_modulation,
        Gwt1CausalSignatureV1::PerceptionConfidence => {
            baseline.confidence_delta != lesion.confidence_delta
        }
    }
}

fn validate_active_arm(
    target: Gwt1CausalTargetV1,
    name: &str,
    arm: &Gwt1CausalArmRawV1,
    expected_invoked: &BTreeSet<String>,
    injected: Option<&Gwt1SubsystemOutputBitsV1>,
    failures: &mut Vec<Gwt1CausalQualificationFailureV1>,
) {
    if &arm.invoked_specialists != expected_invoked {
        failures.push(Gwt1CausalQualificationFailureV1::InvocationSetMismatch {
            target: target.id().to_string(),
            arm: name.to_string(),
            observed: arm.invoked_specialists.iter().cloned().collect(),
        });
    }
    let output_ids: BTreeSet<String> = arm.executed_outputs.keys().cloned().collect();
    if &output_ids != expected_invoked {
        failures.push(Gwt1CausalQualificationFailureV1::OutputCoverageMismatch {
            target: target.id().to_string(),
            arm: name.to_string(),
            observed: output_ids.into_iter().collect(),
        });
    }

    let recomputed = recompute_gwt1_collector_v1(
        &arm.executed_outputs,
        injected.map(|value| (target.id(), value)),
    );
    if recomputed.0 != arm.recorded_contributors {
        failures.push(Gwt1CausalQualificationFailureV1::StoredContributorSetMismatch {
            target: target.id().to_string(),
            arm: name.to_string(),
        });
    }
    if recomputed.1 != arm.integrated {
        failures.push(Gwt1CausalQualificationFailureV1::StoredIntegrationMismatch {
            target: target.id().to_string(),
            arm: name.to_string(),
        });
    }
}

fn validate_active_row(
    row: &Gwt1CausalRowRawV1,
    failures: &mut Vec<Gwt1CausalQualificationFailureV1>,
) {
    let target = row.target;
    let target_id = target.id().to_string();
    let all = all_specialists();
    let controls = controls_for(target);

    if row.target_specialist != target.id() {
        failures.push(Gwt1CausalQualificationFailureV1::TargetIdentityMismatch {
            target: target_id.clone(),
        });
    }
    let signature = expected_signature(target);
    if row.preregistered_signature != signature {
        failures.push(Gwt1CausalQualificationFailureV1::SignatureMismatch {
            target: target_id.clone(),
        });
    }

    validate_active_arm(target, "baseline", &row.baseline, &all, None, failures);
    validate_active_arm(target, "lesion", &row.lesion, &controls, None, failures);
    validate_active_arm(
        target,
        "rescue",
        &row.rescue,
        &controls,
        Some(&row.rescue_injected_target_output),
        failures,
    );
    validate_active_arm(target, "no_op", &row.sham, &all, None, failures);

    let Some(target_output) = row.baseline.executed_outputs.get(target.id()) else {
        return;
    };
    if neutral(target_output) {
        failures.push(Gwt1CausalQualificationFailureV1::TargetNeutral {
            target: target_id.clone(),
        });
    }
    if !signature_present(signature, target_output) {
        failures.push(Gwt1CausalQualificationFailureV1::TargetSignatureAbsent {
            target: target_id.clone(),
        });
    }

    let leaked_to: Vec<String> = row
        .baseline
        .executed_outputs
        .iter()
        .filter(|(id, output)| id.as_str() != target.id() && signature_present(signature, output))
        .map(|(id, _)| id.clone())
        .collect();
    if !leaked_to.is_empty() {
        failures.push(Gwt1CausalQualificationFailureV1::TargetSignatureNotExclusive {
            target: target_id.clone(),
            leaked_to,
        });
    }

    for specialist in &controls {
        let baseline = row.baseline.executed_outputs.get(specialist);
        for (arm_name, arm) in [
            ("lesion", &row.lesion),
            ("rescue", &row.rescue),
            ("no_op", &row.sham),
        ] {
            if arm.executed_outputs.get(specialist) != baseline {
                failures.push(Gwt1CausalQualificationFailureV1::NonTargetLeakage {
                    target: target_id.clone(),
                    arm: arm_name.to_string(),
                    specialist: specialist.clone(),
                });
            }
        }
    }

    if row.sham.integrated != row.baseline.integrated {
        failures.push(Gwt1CausalQualificationFailureV1::NoOpMismatch {
            target: target_id.clone(),
        });
    }
    if row.lesion.integrated == row.baseline.integrated
        || !signature_changed(signature, row.baseline.integrated, row.lesion.integrated)
    {
        failures.push(Gwt1CausalQualificationFailureV1::CausalConsequenceAbsent {
            target: target_id.clone(),
        });
    }
    if &row.rescue_injected_target_output != target_output {
        failures.push(Gwt1CausalQualificationFailureV1::RescueInjectionMismatch {
            target: target_id.clone(),
        });
    }
    if row.rescue.integrated != row.baseline.integrated
        || row.rescue.recorded_contributors != row.baseline.recorded_contributors
    {
        failures.push(Gwt1CausalQualificationFailureV1::RescueMismatch {
            target: target_id.clone(),
        });
    }

    if row.heldout_non_target_trajectory.len() != GWT1_CAUSAL_HELDOUT_STEPS_V1 as usize {
        failures.push(Gwt1CausalQualificationFailureV1::HeldoutLengthMismatch {
            target: target_id.clone(),
            observed: row.heldout_non_target_trajectory.len(),
        });
    }
    for (index, step) in row.heldout_non_target_trajectory.iter().enumerate() {
        if step.step != index as u32 {
            failures.push(Gwt1CausalQualificationFailureV1::HeldoutStepMismatch {
                target: target_id.clone(),
                index,
                observed: step.step,
            });
        }
        for (arm_name, outputs) in [
            ("baseline", &step.baseline_non_target_outputs),
            ("lesion", &step.lesion_non_target_outputs),
            ("rescue", &step.rescue_non_target_outputs),
            ("no_op", &step.sham_non_target_outputs),
        ] {
            let observed: BTreeSet<String> = outputs.keys().cloned().collect();
            if observed != controls {
                failures.push(Gwt1CausalQualificationFailureV1::HeldoutCoverageMismatch {
                    target: target_id.clone(),
                    step: step.step,
                    arm: arm_name.to_string(),
                });
            }
        }
        for (arm_name, outputs) in [
            ("lesion", &step.lesion_non_target_outputs),
            ("rescue", &step.rescue_non_target_outputs),
            ("no_op", &step.sham_non_target_outputs),
        ] {
            if outputs != &step.baseline_non_target_outputs {
                failures.push(Gwt1CausalQualificationFailureV1::HeldoutNonTargetLeakage {
                    target: target_id.clone(),
                    step: step.step,
                    arm: arm_name.to_string(),
                });
            }
        }
    }
}

fn validate_sham_arm(
    row: &Gwt1MatchedShamRowRawV1,
    name: &str,
    arm: &Gwt1MatchedShamArmRawV1,
    expected_invoked: &BTreeSet<String>,
    failures: &mut Vec<Gwt1CausalQualificationFailureV1>,
) {
    if &arm.invoked_specialists != expected_invoked {
        failures.push(Gwt1CausalQualificationFailureV1::InvocationSetMismatch {
            target: row.target.id().to_string(),
            arm: name.to_string(),
            observed: arm.invoked_specialists.iter().cloned().collect(),
        });
    }
    let output_ids: BTreeSet<String> = arm.executed_outputs.keys().cloned().collect();
    if &output_ids != expected_invoked {
        failures.push(Gwt1CausalQualificationFailureV1::OutputCoverageMismatch {
            target: row.target.id().to_string(),
            arm: name.to_string(),
            observed: output_ids.into_iter().collect(),
        });
    }
    let recomputed = recompute_gwt1_collector_v1(&arm.executed_outputs, None);
    if recomputed.0 != arm.recorded_contributors {
        failures.push(Gwt1CausalQualificationFailureV1::StoredContributorSetMismatch {
            target: row.target.id().to_string(),
            arm: name.to_string(),
        });
    }
    if recomputed.1 != arm.integrated {
        failures.push(Gwt1CausalQualificationFailureV1::StoredIntegrationMismatch {
            target: row.target.id().to_string(),
            arm: name.to_string(),
        });
    }
}

fn validate_matched_sham_row(
    row: &Gwt1MatchedShamRowRawV1,
    failures: &mut Vec<Gwt1CausalQualificationFailureV1>,
) {
    let target = row.target;
    let target_id = target.id().to_string();
    let all = all_specialists();
    let controls = controls_for(target);

    if row.target_specialist != target.id() {
        failures.push(Gwt1CausalQualificationFailureV1::TargetIdentityMismatch {
            target: target_id.clone(),
        });
    }
    validate_sham_arm(row, "matched_full", &row.full_panel, &all, failures);
    validate_sham_arm(row, "matched_omission", &row.target_omitted, &controls, failures);

    if let Some(target_output) = row.full_panel.executed_outputs.get(target.id()) {
        if !neutral(target_output) {
            failures.push(Gwt1CausalQualificationFailureV1::MatchedShamTargetNotNeutral {
                target: target_id.clone(),
            });
        }
    }

    for specialist in &controls {
        if row.target_omitted.executed_outputs.get(specialist)
            != row.full_panel.executed_outputs.get(specialist)
        {
            failures.push(Gwt1CausalQualificationFailureV1::MatchedShamOmissionMismatch {
                target: target_id.clone(),
            });
        }
    }
    if row.target_omitted.integrated != row.full_panel.integrated
        || row.target_omitted.recorded_contributors != row.full_panel.recorded_contributors
    {
        failures.push(Gwt1CausalQualificationFailureV1::MatchedShamEffect {
            target: target_id,
        });
    }
}

fn validate_target_set<T>(
    context: &str,
    rows: &[T],
    target_of: impl Fn(&T) -> Gwt1CausalTargetV1,
    failures: &mut Vec<Gwt1CausalQualificationFailureV1>,
) {
    let observed: Vec<String> = rows
        .iter()
        .map(|row| target_of(row).id().to_string())
        .collect();
    let observed_set: BTreeSet<String> = observed.iter().cloned().collect();
    if rows.len() != Gwt1CausalTargetV1::ALL.len() || observed_set != all_targets() {
        failures.push(Gwt1CausalQualificationFailureV1::TargetSetMismatch {
            context: context.to_string(),
            observed,
        });
    }
}

fn class_of(failure: &Gwt1CausalQualificationFailureV1) -> FailureClass {
    use Gwt1CausalQualificationFailureV1::*;
    match failure {
        CausalConsequenceAbsent { .. } => FailureClass::NotDemonstrated,
        TargetSignatureNotExclusive { .. }
        | NonTargetLeakage { .. }
        | NoOpMismatch { .. }
        | RescueInjectionMismatch { .. }
        | RescueMismatch { .. }
        | HeldoutNonTargetLeakage { .. } => FailureClass::Contradicted,
        ContractDrift { .. }
        | TargetSetMismatch { .. }
        | TargetIdentityMismatch { .. }
        | SignatureMismatch { .. }
        | InvocationSetMismatch { .. }
        | OutputCoverageMismatch { .. }
        | StoredIntegrationMismatch { .. }
        | StoredContributorSetMismatch { .. }
        | TargetNeutral { .. }
        | TargetSignatureAbsent { .. }
        | HeldoutLengthMismatch { .. }
        | HeldoutStepMismatch { .. }
        | HeldoutCoverageMismatch { .. }
        | MatchedShamTargetNotNeutral { .. }
        | MatchedShamOmissionMismatch { .. }
        | MatchedShamEffect { .. } => FailureClass::Inconclusive,
    }
}

pub fn resolve_gwt1_causal_v1(
    causal: &Gwt1CausalObservationsV1,
    sham: &Gwt1MatchedShamObservationsV1,
) -> Gwt1CausalQualificationResolutionV1 {
    let mut failures = Vec::new();

    if causal.schema != GWT1_CAUSAL_LESION_SCHEMA_V1 {
        failures.push(Gwt1CausalQualificationFailureV1::ContractDrift {
            field: "causal.schema".to_string(),
        });
    }
    if sham.schema != GWT1_CAUSAL_MATCHED_SHAM_SCHEMA_V1 {
        failures.push(Gwt1CausalQualificationFailureV1::ContractDrift {
            field: "matched_sham.schema".to_string(),
        });
    }

    validate_target_set("causal", &causal.rows, |row| row.target, &mut failures);
    validate_target_set("matched_sham", &sham.rows, |row| row.target, &mut failures);

    for target in Gwt1CausalTargetV1::ALL {
        if let Some(row) = causal.rows.iter().find(|row| row.target == target) {
            validate_active_row(row, &mut failures);
        }
        if let Some(row) = sham.rows.iter().find(|row| row.target == target) {
            validate_matched_sham_row(row, &mut failures);
        }
    }

    let outcome = if failures.is_empty() {
        Gwt1CausalQualificationOutcomeV1::Qualified
    } else if failures
        .iter()
        .any(|failure| class_of(failure) == FailureClass::Inconclusive)
    {
        Gwt1CausalQualificationOutcomeV1::Inconclusive
    } else if failures
        .iter()
        .any(|failure| class_of(failure) == FailureClass::Contradicted)
    {
        Gwt1CausalQualificationOutcomeV1::Contradicted
    } else {
        Gwt1CausalQualificationOutcomeV1::NotDemonstrated
    };

    Gwt1CausalQualificationResolutionV1 {
        schema: GWT1_CAUSAL_QUALIFICATION_SCHEMA_V1.to_string(),
        outcome,
        failures,
    }
}

pub fn run_gwt1_causal_evidence_v1() -> Result<Gwt1CausalEvidenceV1, String> {
    let causal = run_gwt1_causal_lesion_v1();
    let sham = run_gwt1_causal_matched_sham_v1();
    let resolution = resolve_gwt1_causal_v1(&causal, &sham);
    let causal_raw_bytes = serde_json::to_vec(&causal).map_err(|error| error.to_string())?;
    let matched_sham_raw_bytes = serde_json::to_vec(&sham).map_err(|error| error.to_string())?;
    Ok(Gwt1CausalEvidenceV1 {
        causal_raw_bytes,
        matched_sham_raw_bytes,
        resolution,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_full_protocol_resolves_qualified_at_protocol_level() {
        let causal = run_gwt1_causal_lesion_v1();
        let sham = run_gwt1_causal_matched_sham_v1();
        let resolution = resolve_gwt1_causal_v1(&causal, &sham);
        assert_eq!(resolution.schema, GWT1_CAUSAL_QUALIFICATION_SCHEMA_V1);
        assert_eq!(resolution.outcome, Gwt1CausalQualificationOutcomeV1::Qualified);
        assert!(resolution.failures.is_empty());
    }

    #[test]
    fn missing_target_is_inconclusive_not_three_of_four_success() {
        let mut causal = run_gwt1_causal_lesion_v1();
        let sham = run_gwt1_causal_matched_sham_v1();
        causal.rows.pop();
        let resolution = resolve_gwt1_causal_v1(&causal, &sham);
        assert_eq!(resolution.outcome, Gwt1CausalQualificationOutcomeV1::Inconclusive);
    }

    #[test]
    fn corrupted_stored_integration_fails_before_scientific_interpretation() {
        let mut causal = run_gwt1_causal_lesion_v1();
        let sham = run_gwt1_causal_matched_sham_v1();
        causal.rows[0].baseline.integrated.flags ^= 1;
        let resolution = resolve_gwt1_causal_v1(&causal, &sham);
        assert_eq!(resolution.outcome, Gwt1CausalQualificationOutcomeV1::Inconclusive);
        assert!(resolution.failures.iter().any(|failure| matches!(
            failure,
            Gwt1CausalQualificationFailureV1::StoredIntegrationMismatch { .. }
        )));
    }

    #[test]
    fn failed_matched_control_is_inconclusive() {
        let causal = run_gwt1_causal_lesion_v1();
        let mut sham = run_gwt1_causal_matched_sham_v1();
        sham.rows[0].target_omitted.integrated.flags ^= 1;
        let resolution = resolve_gwt1_causal_v1(&causal, &sham);
        assert_eq!(resolution.outcome, Gwt1CausalQualificationOutcomeV1::Inconclusive);
    }

    #[test]
    fn coherent_non_target_leakage_is_contradicted() {
        let mut causal = run_gwt1_causal_lesion_v1();
        let sham = run_gwt1_causal_matched_sham_v1();
        let row = &mut causal.rows[0];
        let control = row
            .lesion
            .executed_outputs
            .get_mut("memory_manager")
            .expect("memory control");
        control.confidence_delta ^= 1;
        let recomputed = recompute_gwt1_collector_v1(&row.lesion.executed_outputs, None);
        row.lesion.recorded_contributors = recomputed.0;
        row.lesion.integrated = recomputed.1;

        let resolution = resolve_gwt1_causal_v1(&causal, &sham);
        assert_eq!(resolution.outcome, Gwt1CausalQualificationOutcomeV1::Contradicted);
        assert!(resolution.failures.iter().any(|failure| matches!(
            failure,
            Gwt1CausalQualificationFailureV1::NonTargetLeakage { .. }
        )));
    }
}
