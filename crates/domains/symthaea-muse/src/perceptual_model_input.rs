// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1G0: deterministic materialization of the registered statistical
//! model inputs from the sealed P1F derived dataset.
//!
//! This layer still does not fit a model. It proves exactly which binary rows
//! a later GLMM runner may consume, preserves participant/item descriptive
//! summaries, and structurally prevents conditioning the key-secondary task on
//! primary ABX success.

use crate::evidence_digest::{
    canonical_json_sha256,
    perceptual_analysis_plan::{BinaryLinkV1, FrozenPerceptualAnalysisSpecV1},
    perceptual_study_protocol::{
        CHANCE_PROBABILITY, MEL003_FIXED_SEEDS, PerceptualTaskV1, PrimaryAnalysisModelV1,
    },
    perceptual_unblinding::{
        FrozenPerceptualDerivedDatasetV1, validate_derived_perceptual_dataset,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const PERCEPTUAL_MODEL_INPUT_BUNDLE_VERSION: &str =
    "mel003-perceptual-model-input-bundle-v1";
pub const PERCEPTUAL_ENDPOINT_MODEL_INPUT_VERSION: &str =
    "mel003-perceptual-endpoint-model-input-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegisteredEndpointRoleV1 {
    Primary,
    KeySecondary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegisteredAlternativeV1 {
    GreaterThanChance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisteredBinaryObservationV1 {
    pub participant_token: String,
    pub item_id: String,
    pub seed: u64,
    pub response: bool,
    pub source_record_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParticipantBinarySummaryV1 {
    pub participant_token: String,
    pub observations: usize,
    pub positive_responses: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ItemBinarySummaryV1 {
    pub item_id: String,
    pub seed: u64,
    pub observations: usize,
    pub positive_responses: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisteredEndpointModelInputV1 {
    pub input_version: String,
    pub role: RegisteredEndpointRoleV1,
    pub task: PerceptualTaskV1,
    pub model: PrimaryAnalysisModelV1,
    pub link: BinaryLinkV1,
    pub chance_probability: f64,
    pub alternative: RegisteredAlternativeV1,
    pub participant_random_intercept: bool,
    pub item_random_intercept: bool,
    pub observations: Vec<RegisteredBinaryObservationV1>,
    pub participant_summaries: Vec<ParticipantBinarySummaryV1>,
    pub item_summaries: Vec<ItemBinarySummaryV1>,
    pub input_sha256: String,
}

#[derive(Serialize)]
struct EndpointModelInputCommitment<'a> {
    input_version: &'a str,
    role: RegisteredEndpointRoleV1,
    task: PerceptualTaskV1,
    model: PrimaryAnalysisModelV1,
    link: BinaryLinkV1,
    chance_probability: f64,
    alternative: RegisteredAlternativeV1,
    participant_random_intercept: bool,
    item_random_intercept: bool,
    observations: &'a [RegisteredBinaryObservationV1],
    participant_summaries: &'a [ParticipantBinarySummaryV1],
    item_summaries: &'a [ItemBinarySummaryV1],
}

pub fn endpoint_model_input_commitment(
    input: &RegisteredEndpointModelInputV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&EndpointModelInputCommitment {
        input_version: &input.input_version,
        role: input.role,
        task: input.task,
        model: input.model,
        link: input.link,
        chance_probability: input.chance_probability,
        alternative: input.alternative,
        participant_random_intercept: input.participant_random_intercept,
        item_random_intercept: input.item_random_intercept,
        observations: &input.observations,
        participant_summaries: &input.participant_summaries,
        item_summaries: &input.item_summaries,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPerceptualModelInputBundleV1 {
    pub bundle_version: String,
    pub derived_dataset_sha256: String,
    pub analysis_spec_sha256: String,
    pub primary: RegisteredEndpointModelInputV1,
    pub key_secondary: RegisteredEndpointModelInputV1,
    pub bundle_sha256: String,
}

#[derive(Serialize)]
struct ModelInputBundleCommitment<'a> {
    bundle_version: &'a str,
    derived_dataset_sha256: &'a str,
    analysis_spec_sha256: &'a str,
    primary: &'a RegisteredEndpointModelInputV1,
    key_secondary: &'a RegisteredEndpointModelInputV1,
}

pub fn model_input_bundle_commitment(
    bundle: &FrozenPerceptualModelInputBundleV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ModelInputBundleCommitment {
        bundle_version: &bundle.bundle_version,
        derived_dataset_sha256: &bundle.derived_dataset_sha256,
        analysis_spec_sha256: &bundle.analysis_spec_sha256,
        primary: &bundle.primary,
        key_secondary: &bundle.key_secondary,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualModelInputIssueV1 {
    InvalidDerivedDataset,
    InvalidAnalysisSpec,
    AnalysisSpecSerializationFailed,
    AnalysisSpecDigestMismatch,
    WrongBundleVersion,
    DerivedDatasetDigestMismatch,
    BundleAnalysisSpecDigestMismatch,
    PrimaryInputMismatch,
    SecondaryInputMismatch,
    EndpointInputSerializationFailed { task: PerceptualTaskV1 },
    ParticipantPopulationMismatch,
    ItemPopulationMismatch,
    InvalidDigest { field: String },
    BundleDigestMismatch,
    SerializationFailed,
}

pub fn materialize_perceptual_model_inputs(
    dataset: &FrozenPerceptualDerivedDatasetV1,
    analysis_spec: &FrozenPerceptualAnalysisSpecV1,
) -> Result<FrozenPerceptualModelInputBundleV1, Vec<PerceptualModelInputIssueV1>> {
    let mut issues = predecessor_issues(dataset, analysis_spec);
    if !issues.is_empty() {
        return Err(issues);
    }

    let primary = build_endpoint_input(
        dataset,
        analysis_spec,
        RegisteredEndpointRoleV1::Primary,
        analysis_spec.primary_task,
    )
    .map_err(|_| {
        vec![PerceptualModelInputIssueV1::EndpointInputSerializationFailed {
            task: analysis_spec.primary_task,
        }]
    })?;
    let key_secondary = build_endpoint_input(
        dataset,
        analysis_spec,
        RegisteredEndpointRoleV1::KeySecondary,
        analysis_spec.key_secondary_task,
    )
    .map_err(|_| {
        vec![PerceptualModelInputIssueV1::EndpointInputSerializationFailed {
            task: analysis_spec.key_secondary_task,
        }]
    })?;

    let primary_participants: BTreeSet<_> = primary
        .observations
        .iter()
        .map(|row| row.participant_token.as_str())
        .collect();
    let secondary_participants: BTreeSet<_> = key_secondary
        .observations
        .iter()
        .map(|row| row.participant_token.as_str())
        .collect();
    if primary_participants != secondary_participants {
        issues.push(PerceptualModelInputIssueV1::ParticipantPopulationMismatch);
    }
    let primary_items: BTreeSet<_> = primary
        .observations
        .iter()
        .map(|row| (row.item_id.as_str(), row.seed))
        .collect();
    let secondary_items: BTreeSet<_> = key_secondary
        .observations
        .iter()
        .map(|row| (row.item_id.as_str(), row.seed))
        .collect();
    if primary_items != secondary_items {
        issues.push(PerceptualModelInputIssueV1::ItemPopulationMismatch);
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    let analysis_spec_sha256 = canonical_json_sha256(analysis_spec)
        .map_err(|_| vec![PerceptualModelInputIssueV1::AnalysisSpecSerializationFailed])?;
    let mut bundle = FrozenPerceptualModelInputBundleV1 {
        bundle_version: PERCEPTUAL_MODEL_INPUT_BUNDLE_VERSION.into(),
        derived_dataset_sha256: dataset.dataset_sha256.clone(),
        analysis_spec_sha256,
        primary,
        key_secondary,
        bundle_sha256: String::new(),
    };
    bundle.bundle_sha256 = model_input_bundle_commitment(&bundle)
        .map_err(|_| vec![PerceptualModelInputIssueV1::SerializationFailed])?;

    let validation = validate_perceptual_model_input_bundle(dataset, analysis_spec, &bundle);
    if validation.is_empty() {
        Ok(bundle)
    } else {
        Err(validation)
    }
}

pub fn validate_perceptual_model_input_bundle(
    dataset: &FrozenPerceptualDerivedDatasetV1,
    analysis_spec: &FrozenPerceptualAnalysisSpecV1,
    bundle: &FrozenPerceptualModelInputBundleV1,
) -> Vec<PerceptualModelInputIssueV1> {
    let mut issues = predecessor_issues(dataset, analysis_spec);
    if bundle.bundle_version != PERCEPTUAL_MODEL_INPUT_BUNDLE_VERSION {
        issues.push(PerceptualModelInputIssueV1::WrongBundleVersion);
    }
    if bundle.derived_dataset_sha256 != dataset.dataset_sha256 {
        issues.push(PerceptualModelInputIssueV1::DerivedDatasetDigestMismatch);
    }
    let analysis_spec_sha256 = match canonical_json_sha256(analysis_spec) {
        Ok(value) => value,
        Err(_) => {
            issues.push(PerceptualModelInputIssueV1::AnalysisSpecSerializationFailed);
            String::new()
        }
    };
    if bundle.analysis_spec_sha256 != analysis_spec_sha256 {
        issues.push(PerceptualModelInputIssueV1::BundleAnalysisSpecDigestMismatch);
    }

    let expected_primary = build_endpoint_input(
        dataset,
        analysis_spec,
        RegisteredEndpointRoleV1::Primary,
        analysis_spec.primary_task,
    );
    match expected_primary {
        Ok(expected) if expected == bundle.primary => {}
        Ok(_) => issues.push(PerceptualModelInputIssueV1::PrimaryInputMismatch),
        Err(_) => issues.push(PerceptualModelInputIssueV1::EndpointInputSerializationFailed {
            task: analysis_spec.primary_task,
        }),
    }
    let expected_secondary = build_endpoint_input(
        dataset,
        analysis_spec,
        RegisteredEndpointRoleV1::KeySecondary,
        analysis_spec.key_secondary_task,
    );
    match expected_secondary {
        Ok(expected) if expected == bundle.key_secondary => {}
        Ok(_) => issues.push(PerceptualModelInputIssueV1::SecondaryInputMismatch),
        Err(_) => issues.push(PerceptualModelInputIssueV1::EndpointInputSerializationFailed {
            task: analysis_spec.key_secondary_task,
        }),
    }

    let primary_participants: BTreeSet<_> = bundle
        .primary
        .observations
        .iter()
        .map(|row| row.participant_token.as_str())
        .collect();
    let secondary_participants: BTreeSet<_> = bundle
        .key_secondary
        .observations
        .iter()
        .map(|row| row.participant_token.as_str())
        .collect();
    if primary_participants != secondary_participants {
        issues.push(PerceptualModelInputIssueV1::ParticipantPopulationMismatch);
    }
    let primary_items: BTreeSet<_> = bundle
        .primary
        .observations
        .iter()
        .map(|row| (row.item_id.as_str(), row.seed))
        .collect();
    let secondary_items: BTreeSet<_> = bundle
        .key_secondary
        .observations
        .iter()
        .map(|row| (row.item_id.as_str(), row.seed))
        .collect();
    if primary_items != secondary_items {
        issues.push(PerceptualModelInputIssueV1::ItemPopulationMismatch);
    }

    for (field, digest) in [
        ("derived_dataset_sha256", bundle.derived_dataset_sha256.as_str()),
        ("analysis_spec_sha256", bundle.analysis_spec_sha256.as_str()),
        ("primary.input_sha256", bundle.primary.input_sha256.as_str()),
        (
            "key_secondary.input_sha256",
            bundle.key_secondary.input_sha256.as_str(),
        ),
        ("bundle_sha256", bundle.bundle_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(PerceptualModelInputIssueV1::InvalidDigest {
                field: field.into(),
            });
        }
    }
    match model_input_bundle_commitment(bundle) {
        Ok(value) if value == bundle.bundle_sha256 => {}
        Ok(_) => issues.push(PerceptualModelInputIssueV1::BundleDigestMismatch),
        Err(_) => issues.push(PerceptualModelInputIssueV1::SerializationFailed),
    }
    issues
}

fn predecessor_issues(
    dataset: &FrozenPerceptualDerivedDatasetV1,
    analysis_spec: &FrozenPerceptualAnalysisSpecV1,
) -> Vec<PerceptualModelInputIssueV1> {
    let mut issues = Vec::new();
    if !validate_derived_perceptual_dataset(dataset).is_empty() {
        issues.push(PerceptualModelInputIssueV1::InvalidDerivedDataset);
    }
    if !analysis_spec.validate().is_empty() {
        issues.push(PerceptualModelInputIssueV1::InvalidAnalysisSpec);
    }
    match canonical_json_sha256(analysis_spec) {
        Ok(value) if value == dataset.analysis_spec_sha256 => {}
        Ok(_) => issues.push(PerceptualModelInputIssueV1::AnalysisSpecDigestMismatch),
        Err(_) => issues.push(PerceptualModelInputIssueV1::AnalysisSpecSerializationFailed),
    }
    issues
}

fn build_endpoint_input(
    dataset: &FrozenPerceptualDerivedDatasetV1,
    analysis_spec: &FrozenPerceptualAnalysisSpecV1,
    role: RegisteredEndpointRoleV1,
    task: PerceptualTaskV1,
) -> Result<RegisteredEndpointModelInputV1, serde_json::Error> {
    let mut observations: Vec<_> = dataset
        .rows
        .iter()
        .filter(|row| row.task == task)
        .map(|row| RegisteredBinaryObservationV1 {
            participant_token: row.participant_token.clone(),
            item_id: row.item_id.clone(),
            seed: row.seed,
            response: row.outcome.binary_response(),
            source_record_sha256: row.source_record_sha256.clone(),
        })
        .collect();
    observations.sort_by(|left, right| {
        left.participant_token
            .cmp(&right.participant_token)
            .then_with(|| left.seed.cmp(&right.seed))
            .then_with(|| left.item_id.cmp(&right.item_id))
            .then_with(|| left.source_record_sha256.cmp(&right.source_record_sha256))
    });

    let mut participant_counts: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut item_counts: BTreeMap<(String, u64), (usize, usize)> = BTreeMap::new();
    for row in &observations {
        let participant = participant_counts
            .entry(row.participant_token.clone())
            .or_insert((0, 0));
        participant.0 += 1;
        participant.1 += usize::from(row.response);

        let item = item_counts
            .entry((row.item_id.clone(), row.seed))
            .or_insert((0, 0));
        item.0 += 1;
        item.1 += usize::from(row.response);
    }
    let participant_summaries = participant_counts
        .into_iter()
        .map(|(participant_token, (observations, positive_responses))| {
            ParticipantBinarySummaryV1 {
                participant_token,
                observations,
                positive_responses,
            }
        })
        .collect();
    let item_summaries = item_counts
        .into_iter()
        .map(|((item_id, seed), (observations, positive_responses))| ItemBinarySummaryV1 {
            item_id,
            seed,
            observations,
            positive_responses,
        })
        .collect();

    let mut input = RegisteredEndpointModelInputV1 {
        input_version: PERCEPTUAL_ENDPOINT_MODEL_INPUT_VERSION.into(),
        role,
        task,
        model: analysis_spec.primary_model,
        link: analysis_spec.link,
        chance_probability: analysis_spec.chance_probability,
        alternative: RegisteredAlternativeV1::GreaterThanChance,
        participant_random_intercept: analysis_spec.participant_random_intercept_required,
        item_random_intercept: analysis_spec.item_random_intercept_required,
        observations,
        participant_summaries,
        item_summaries,
        input_sha256: String::new(),
    };
    input.input_sha256 = endpoint_model_input_commitment(&input)?;
    Ok(input)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_digest::{
        perceptual_analysis_plan::{
            ModelFailurePolicyV1, PerceptualAnalysisExecutionIdentityV1,
            SecondaryConditioningPolicyV1, PERCEPTUAL_ANALYSIS_SPEC_VERSION,
        },
        perceptual_study_protocol::SecondaryMultiplicityPolicyV1,
        perceptual_unblinding::{
            DerivedPerceptualOutcomeV1, DerivedPerceptualTrialV1,
            PERCEPTUAL_DERIVED_DATASET_VERSION, derived_dataset_commitment,
        },
    };

    const DIGEST: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn analysis_spec() -> FrozenPerceptualAnalysisSpecV1 {
        FrozenPerceptualAnalysisSpecV1 {
            spec_version: PERCEPTUAL_ANALYSIS_SPEC_VERSION.into(),
            primary_task: PerceptualTaskV1::AbxDiscrimination,
            key_secondary_task: PerceptualTaskV1::DirectionalRearticulation2Afc,
            primary_model: PrimaryAnalysisModelV1::CrossedParticipantItemLogisticRandomIntercepts,
            link: BinaryLinkV1::Logit,
            chance_probability: CHANCE_PROBABILITY,
            alpha: 0.05,
            confidence_level: 0.95,
            participant_random_intercept_required: true,
            item_random_intercept_required: true,
            registered_item_seeds: MEL003_FIXED_SEEDS,
            primary_alternative_greater_than_chance: true,
            secondary_conditioning:
                SecondaryConditioningPolicyV1::IndependentTrialsNoConditioningOnPrimary,
            secondary_multiplicity:
                SecondaryMultiplicityPolicyV1::HolmWithinRegisteredSecondaryFamily,
            report_fixed_effect_estimate: true,
            report_odds_ratio: true,
            report_confidence_interval: true,
            report_participant_variance: true,
            report_item_variance: true,
            report_participant_summary: true,
            report_item_summary: true,
            model_failure_policy: ModelFailurePolicyV1::InconclusiveNoAutomaticSubstitution,
            execution: PerceptualAnalysisExecutionIdentityV1 {
                source_revision: "analysis-source".into(),
                analysis_program_sha256: DIGEST.into(),
                environment_sha256: DIGEST.into(),
                model_engine: "qualified-glmm".into(),
                model_engine_version: "v1".into(),
            },
        }
    }

    fn dataset(spec: &FrozenPerceptualAnalysisSpecV1) -> FrozenPerceptualDerivedDatasetV1 {
        let mut rows = Vec::new();
        for participant_index in 0..2usize {
            let participant = format!("p-{participant_index}");
            let session_digest = format!("{:064x}", 10_000 + participant_index);
            for (index, seed) in MEL003_FIXED_SEEDS.iter().copied().enumerate() {
                let record_index = participant_index * 32 + index;
                rows.push(DerivedPerceptualTrialV1 {
                    participant_token: participant.clone(),
                    source_session_sha256: session_digest.clone(),
                    source_record_sha256: format!("{record_index:064x}"),
                    sequence: index as u32,
                    trial_id: format!("{participant}-abx-{seed}"),
                    item_id: format!("sonata-seed-{seed}"),
                    seed,
                    task: PerceptualTaskV1::AbxDiscrimination,
                    outcome: DerivedPerceptualOutcomeV1::AbxCorrect {
                        correct: (participant_index + index) % 2 == 0,
                    },
                });
            }
            for (index, seed) in MEL003_FIXED_SEEDS.iter().copied().enumerate() {
                let sequence = index + MEL003_FIXED_SEEDS.len();
                let record_index = participant_index * 32 + sequence;
                rows.push(DerivedPerceptualTrialV1 {
                    participant_token: participant.clone(),
                    source_session_sha256: session_digest.clone(),
                    source_record_sha256: format!("{record_index:064x}"),
                    sequence: sequence as u32,
                    trial_id: format!("{participant}-direction-{seed}"),
                    item_id: format!("sonata-seed-{seed}"),
                    seed,
                    task: PerceptualTaskV1::DirectionalRearticulation2Afc,
                    outcome: DerivedPerceptualOutcomeV1::DirectionalInterventionSelected {
                        intervention_selected: (participant_index + index) % 3 != 0,
                    },
                });
            }
        }
        let mut dataset = FrozenPerceptualDerivedDatasetV1 {
            dataset_version: PERCEPTUAL_DERIVED_DATASET_VERSION.into(),
            protocol_sha256: DIGEST.into(),
            analysis_spec_sha256: canonical_json_sha256(spec).unwrap(),
            stimulus_pack_sha256: DIGEST.into(),
            participant_schedule_sha256: DIGEST.into(),
            private_audit_sha256: DIGEST.into(),
            raw_dataset_sha256: DIGEST.into(),
            collection_close_sha256: DIGEST.into(),
            randomization_commitment_sha256: DIGEST.into(),
            schedule_reveal_sha256: DIGEST.into(),
            cohort_id: "cohort".into(),
            completed_participant_count: 2,
            excluded_aborted_session_count: 0,
            withdrawn_and_deleted_session_count: 0,
            rows,
            dataset_sha256: String::new(),
        };
        dataset.dataset_sha256 = derived_dataset_commitment(&dataset).unwrap();
        dataset
    }

    #[test]
    fn materialization_preserves_complete_population_for_both_tasks() {
        let spec = analysis_spec();
        let dataset = dataset(&spec);
        assert!(validate_derived_perceptual_dataset(&dataset).is_empty());
        let bundle = materialize_perceptual_model_inputs(&dataset, &spec).unwrap();
        assert_eq!(bundle.primary.observations.len(), 16);
        assert_eq!(bundle.key_secondary.observations.len(), 16);
        let primary_participants: BTreeSet<_> = bundle
            .primary
            .observations
            .iter()
            .map(|row| row.participant_token.as_str())
            .collect();
        let secondary_participants: BTreeSet<_> = bundle
            .key_secondary
            .observations
            .iter()
            .map(|row| row.participant_token.as_str())
            .collect();
        assert_eq!(primary_participants, secondary_participants);
        assert_eq!(primary_participants.len(), 2);
    }

    #[test]
    fn secondary_input_is_not_conditioned_on_abx_success() {
        let spec = analysis_spec();
        let dataset = dataset(&spec);
        let bundle = materialize_perceptual_model_inputs(&dataset, &spec).unwrap();
        let primary_positive = bundle
            .primary
            .observations
            .iter()
            .filter(|row| row.response)
            .count();
        assert!(primary_positive < bundle.primary.observations.len());
        assert_eq!(
            bundle.key_secondary.observations.len(),
            bundle.primary.observations.len()
        );
    }

    #[test]
    fn validation_detects_selective_secondary_row_removal_even_after_reseal() {
        let spec = analysis_spec();
        let dataset = dataset(&spec);
        let mut bundle = materialize_perceptual_model_inputs(&dataset, &spec).unwrap();
        bundle.key_secondary.observations.pop();
        bundle.key_secondary.input_sha256 =
            endpoint_model_input_commitment(&bundle.key_secondary).unwrap();
        bundle.bundle_sha256 = model_input_bundle_commitment(&bundle).unwrap();
        assert!(
            validate_perceptual_model_input_bundle(&dataset, &spec, &bundle)
                .contains(&PerceptualModelInputIssueV1::SecondaryInputMismatch)
        );
    }
}
