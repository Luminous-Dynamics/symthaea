// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical, fully validated MEL-003 enrollment/witness qualification fixture.
//!
//! This module is test-only. It constructs a small two-slot study through the
//! same public seal/build/validate APIs used by the production contracts. The
//! structural inputs are fixed, but participant pseudonyms are produced by the
//! registered OS-CSPRNG constructor so the fixture never fabricates an entropy
//! provenance statement merely to gain deterministic test identities.

use super::{
    canonical_json_sha256, sha256_hex,
    perceptual_collection_authenticity::{
        seal_authenticity_policy, seal_withdrawal_policy, validate_authenticity_policy,
        validate_withdrawal_policy, CollectionSigningKeyV1,
        FrozenPerceptualCollectionAuthenticityPolicyV1, FrozenPerceptualWithdrawalPolicyV1,
        WithdrawalEvidenceRetentionModeV1, PERCEPTUAL_COLLECTION_AUTHENTICITY_POLICY_VERSION,
        PERCEPTUAL_COLLECTION_STUDY_DOMAIN, PERCEPTUAL_WITHDRAWAL_POLICY_VERSION,
    },
    perceptual_collection_evidence::{
        seal_collection_authority, validate_collection_authority,
        FrozenPerceptualCollectionAuthorityV1, HumanStudyReviewDispositionV1,
        PerceptualCollectionRoleV1, PerceptualCollectionRunnerIdentityV1,
        PERCEPTUAL_COLLECTION_AUTHORITY_VERSION,
    },
    perceptual_enrollment_coordinator::{
        new_confirmed_enrollment_coordinator_state, DurableEnrollmentCoordinatorStoreV1,
        DurablePerceptualEnrollmentCoordinatorStateV1,
    },
    perceptual_enrollment_lifecycle::{
        new_enrollment_allocation_ledger, seal_eligibility_gate, seal_enrollment_policy,
        validate_eligibility_gate, validate_enrollment_allocation_ledger,
        validate_enrollment_policy, EnrollmentSlotAllocationOrderV1,
        FrozenPerceptualEligibilityGateReceiptV1, FrozenPerceptualEnrollmentPolicyV1,
        PERCEPTUAL_ELIGIBILITY_GATE_VERSION, PERCEPTUAL_ENROLLMENT_POLICY_VERSION,
    },
    perceptual_enrollment_orchestrator::PerceptualEnrollmentWitnessOrchestratorV1,
    perceptual_enrollment_store::{
        DurableEnrollmentAllocationStateV1, DurableEnrollmentAllocationStoreV1,
    },
    perceptual_enrollment_store_observation::inspect_validated_current_from_confirmed_head,
    perceptual_enrollment_witness::{
        seal_enrollment_witness_policy, validate_enrollment_witness_bundle,
        validate_enrollment_witness_policy, FrozenPerceptualEnrollmentWitnessPolicyV1,
        PERCEPTUAL_ENROLLMENT_WITNESS_POLICY_VERSION,
    },
    perceptual_enrollment_witness_provider::new_external_enrollment_witness_candidate_bundle,
    perceptual_enrollment_witness_provider_fixture::{
        DurableQualificationEnrollmentWitnessProviderV1, QualificationProviderFixtureErrorV1,
    },
    perceptual_participant_identity::{
        generate_perceptual_cohort_slots_os_rng, seal_participant_identity_boundary_policy,
        validate_participant_identity_boundary_policy,
        validate_participant_token_generation_receipt, FrozenParticipantIdentityBoundaryPolicyV1,
        FrozenPerceptualParticipantTokenGenerationReceiptV1, ParticipantTokenGeneratorIdentityV1,
        PARTICIPANT_IDENTITY_BOUNDARY_POLICY_VERSION, PARTICIPANT_TOKEN_GENERATOR_VERSION,
    },
    perceptual_participant_schedule::{
        build_perceptual_participant_schedule, PerceptualCohortSlotsV1,
        PerceptualParticipantScheduleBookV1, PERCEPTUAL_SCHEDULE_BUILDER_VERSION,
    },
    perceptual_stimulus_pack::{
        C6fRenderSubjectItemV1, FrozenC6fRenderSubjectBindingV1,
        FrozenPerceptualStimulusPackV1, LoudnessMeterIdentityV1, PerceptualStimulusPairV1,
        StimulusArmV1, StimulusAudioAssetV1, StimulusRendererIdentityV1,
        StimulusTransformV1, C6F_RENDER_SUBJECT_BINDING_VERSION, GAIN_TRANSFORM_PROFILE_V1,
        LOUDNESS_MEASUREMENT_PROFILE_V1, MAX_RESIDUAL_PAIR_DELTA_LU_V1,
        PERCEPTUAL_STIMULUS_PACK_VERSION, REQUIRED_CHANNEL_COUNT, REQUIRED_SAMPLE_RATE_HZ,
    },
    perceptual_study_protocol::{
        AnalysisPolicyV1, BlindingAndRandomizationPolicyV1, ExternalPerceptualPreregistrationV1,
        ForbiddenPerceptualClaimV1, FrozenPerceptualStudyProtocolV1,
        LoudnessMatchingV1, Mel003AcousticSubjectBindingV1, MissingResponsePolicyV1,
        ParticipantPolicyV1, PerceptualEndpointRoleV1, PerceptualEndpointV1,
        PerceptualStudyItemV1, PerceptualTaskV1, PrimaryAnalysisModelV1, SampleSizePlanV1,
        SecondaryMultiplicityPolicyV1, StimulusExtentV1, StimulusPolicyV1,
        MEL003_C6F_BUNDLE_VERSION, MEL003_FIXED_SEEDS, PERCEPTUAL_STUDY_PROTOCOL_VERSION,
    },
};
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

const SCHEDULE_SECRET: [u8; 32] = [0x42; 32];
pub const QUALIFICATION_COLLECTION_SEED: [u8; 32] = [0x11; 32];
pub const QUALIFICATION_WITNESS_SEED: [u8; 32] = [0x22; 32];
const QUALIFICATION_WITHDRAWAL_SEED: [u8; 32] = [0x33; 32];
const SOURCE_COMMIT: &str = "cccccccccccccccccccccccccccccccccccccccc";
const DIGEST_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
static NEXT_FIXTURE_ID: AtomicU64 = AtomicU64::new(1);

fn digest(index: usize) -> String {
    format!("{:064x}", index.saturating_add(1))
}

fn qualification_root() -> PathBuf {
    let serial = NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or_default();
    std::env::temp_dir().join(format!(
        "symthaea-mel003-e2e-{}-{nanos}-{serial}",
        std::process::id()
    ))
}

fn protocol() -> FrozenPerceptualStudyProtocolV1 {
    FrozenPerceptualStudyProtocolV1 {
        protocol_version: PERCEPTUAL_STUDY_PROTOCOL_VERSION.into(),
        acoustic_subject: Mel003AcousticSubjectBindingV1 {
            c6f_source_commit: SOURCE_COMMIT.into(),
            c6f_bundle_sha256: DIGEST_A.into(),
            c6f_bundle_version: MEL003_C6F_BUNDLE_VERSION.into(),
        },
        external_preregistration: ExternalPerceptualPreregistrationV1 {
            registry: "qualification-fixture".into(),
            record_id: "mel003-e2e-fixture-v1".into(),
            frozen_at_utc: "2026-09-19T00:00:00Z".into(),
            record_sha256: digest(10),
        },
        analysis_spec_sha256: digest(11),
        items: MEL003_FIXED_SEEDS
            .into_iter()
            .map(|seed| PerceptualStudyItemV1 {
                item_id: format!("sonata-seed-{seed}"),
                seed,
            })
            .collect(),
        endpoints: vec![
            PerceptualEndpointV1 {
                task: PerceptualTaskV1::AbxDiscrimination,
                role: PerceptualEndpointRoleV1::Primary,
                chance_probability: 0.5,
                estimand: "qualification-fixture ABX correctness probability".into(),
            },
            PerceptualEndpointV1 {
                task: PerceptualTaskV1::DirectionalRearticulation2Afc,
                role: PerceptualEndpointRoleV1::KeySecondary,
                chance_probability: 0.5,
                estimand: "qualification-fixture directional choice probability".into(),
            },
        ],
        stimulus: StimulusPolicyV1 {
            extent: StimulusExtentV1::WholeFourBarSubject,
            synchronized_playhead_required: true,
            loudness_matching: LoudnessMatchingV1::PairwiseIntegratedLufsAttenuationOnly,
            maximum_attenuation_db: 6.0,
            preserve_pair_alignment: true,
            neutral_presentation_labels_required: true,
            disjoint_practice_material_required: true,
        },
        blinding: BlindingAndRandomizationPolicyV1 {
            randomization_commitment_sha256: sha256_hex(&SCHEDULE_SECRET),
            schedule_builder_version: PERCEPTUAL_SCHEDULE_BUILDER_VERSION.into(),
            balance_ab_label_assignment: true,
            balance_abx_hidden_identity: true,
            balance_directional_left_right_assignment: true,
            reveal_correct_answers_during_scored_collection: false,
            arm_labelled_monitoring_during_collection: false,
            investigator_can_modify_schedule_after_first_response: false,
        },
        participants: ParticipantPolicyV1 {
            minimum_age_years: 18,
            informed_consent_required: true,
            pseudonymous_participant_tokens_required: true,
            raw_names_or_contact_details_in_study_dataset_allowed: false,
            stereo_playback_check_required: true,
            task_comprehension_practice_required: true,
            practice_feedback_allowed: true,
            scored_trial_feedback_allowed: false,
        },
        sample_size: SampleSizePlanV1 {
            planning_artifact_sha256: digest(12),
            planned_completed_participants: 1,
            maximum_enrolled_participants: 2,
            outcome_adaptive_stopping_allowed: false,
        },
        analysis: AnalysisPolicyV1 {
            primary_model: PrimaryAnalysisModelV1::CrossedParticipantItemLogisticRandomIntercepts,
            participant_grouping_factor_required: true,
            item_grouping_factor_required: true,
            primary_alternative_is_greater_than_chance: true,
            alpha: 0.05,
            confidence_level: 0.95,
            secondary_multiplicity: SecondaryMultiplicityPolicyV1::HolmWithinRegisteredSecondaryFamily,
            report_item_level_outcomes: true,
            report_participant_level_outcomes: true,
            report_random_effect_variance: true,
            missing_response_policy: MissingResponsePolicyV1::RetainRawExcludeIncompleteSessionNoImputation,
        },
        forbidden_claims: vec![
            ForbiddenPerceptualClaimV1::Preference,
            ForbiddenPerceptualClaimV1::ArtisticQuality,
            ForbiddenPerceptualClaimV1::EmotionalImpact,
            ForbiddenPerceptualClaimV1::StyleIdentity,
            ForbiddenPerceptualClaimV1::CulturalAuthenticity,
            ForbiddenPerceptualClaimV1::HumanLikePerformance,
            ForbiddenPerceptualClaimV1::IndependentAcousticReplication,
            ForbiddenPerceptualClaimV1::CognitionProductAuthority,
            ForbiddenPerceptualClaimV1::GeneralizationBeyondRegisteredItemsAndEligiblePopulation,
        ],
        stimulus_pack_bound: false,
        participant_schedule_bound: false,
        collection_authorized: false,
        responses_present: false,
    }
}

fn render_binding(protocol: &FrozenPerceptualStudyProtocolV1) -> FrozenC6fRenderSubjectBindingV1 {
    FrozenC6fRenderSubjectBindingV1 {
        binding_version: C6F_RENDER_SUBJECT_BINDING_VERSION.into(),
        protocol_sha256: protocol.protocol_sha256().expect("protocol digest"),
        c6f_source_commit: protocol.acoustic_subject.c6f_source_commit.clone(),
        c6f_bundle_sha256: protocol.acoustic_subject.c6f_bundle_sha256.clone(),
        c6f_bundle_version: protocol.acoustic_subject.c6f_bundle_version.clone(),
        renderer: StimulusRendererIdentityV1 {
            source_revision: protocol.acoustic_subject.c6f_source_commit.clone(),
            renderer_version: "qualification-renderer-v1".into(),
            render_config_sha256: digest(20),
            environment_sha256: digest(21),
        },
        items: protocol
            .items
            .iter()
            .enumerate()
            .map(|(index, item)| C6fRenderSubjectItemV1 {
                item_id: item.item_id.clone(),
                seed: item.seed,
                baseline_render_sha256: digest(100 + index),
                intervention_render_sha256: digest(200 + index),
                sample_rate_hz: REQUIRED_SAMPLE_RATE_HZ,
                channel_count: REQUIRED_CHANNEL_COUNT,
                frame_count: 88_200,
            })
            .collect(),
    }
}

fn audio_asset(
    source: &str,
    output: &str,
    initial_lufs: f64,
    final_lufs: f64,
    gain_db: f64,
) -> StimulusAudioAssetV1 {
    StimulusAudioAssetV1 {
        source_sha256: source.into(),
        output_sha256: output.into(),
        sample_rate_hz: REQUIRED_SAMPLE_RATE_HZ,
        channel_count: REQUIRED_CHANNEL_COUNT,
        frame_count: 88_200,
        initial_integrated_lufs: initial_lufs,
        final_integrated_lufs: final_lufs,
        applied_gain_db: gain_db,
    }
}

fn stimulus_pack(
    protocol: &FrozenPerceptualStudyProtocolV1,
    binding: &FrozenC6fRenderSubjectBindingV1,
) -> FrozenPerceptualStimulusPackV1 {
    FrozenPerceptualStimulusPackV1 {
        pack_version: PERCEPTUAL_STIMULUS_PACK_VERSION.into(),
        protocol_sha256: protocol.protocol_sha256().expect("protocol digest"),
        render_subject_binding_sha256: binding.binding_sha256().expect("binding digest"),
        c6f_source_commit: protocol.acoustic_subject.c6f_source_commit.clone(),
        c6f_bundle_sha256: protocol.acoustic_subject.c6f_bundle_sha256.clone(),
        c6f_bundle_version: protocol.acoustic_subject.c6f_bundle_version.clone(),
        extent: StimulusExtentV1::WholeFourBarSubject,
        loudness_matching: LoudnessMatchingV1::PairwiseIntegratedLufsAttenuationOnly,
        transform: StimulusTransformV1::ConstantGainAttenuationOnly,
        gain_transform_profile: GAIN_TRANSFORM_PROFILE_V1.into(),
        max_attenuation_db: protocol.stimulus.maximum_attenuation_db,
        max_residual_pair_delta_lu: MAX_RESIDUAL_PAIR_DELTA_LU_V1,
        renderer: binding.renderer.clone(),
        loudness_meter: LoudnessMeterIdentityV1 {
            measurement_profile: LOUDNESS_MEASUREMENT_PROFILE_V1.into(),
            source_revision: SOURCE_COMMIT.into(),
            implementation_version: "qualification-loudness-meter-v1".into(),
            environment_sha256: digest(22),
        },
        items: binding
            .items
            .iter()
            .enumerate()
            .map(|(index, item)| PerceptualStimulusPairV1 {
                item_id: item.item_id.clone(),
                seed: item.seed,
                baseline: audio_asset(
                    &item.baseline_render_sha256,
                    &digest(300 + index),
                    -17.0,
                    -18.0,
                    -1.0,
                ),
                intervention: audio_asset(
                    &item.intervention_render_sha256,
                    &item.intervention_render_sha256,
                    -18.0,
                    -18.0,
                    0.0,
                ),
                attenuated_arm: Some(StimulusArmV1::Baseline),
                post_match_pair_delta_lu: 0.0,
            })
            .collect(),
        participant_labels_bound: false,
        participant_schedule_bound: false,
        responses_present: false,
    }
}

pub struct Mel003EnrollmentE2eQualificationFixtureV1 {
    pub protocol: FrozenPerceptualStudyProtocolV1,
    pub render_binding: FrozenC6fRenderSubjectBindingV1,
    pub stimulus_pack: FrozenPerceptualStimulusPackV1,
    pub cohort: PerceptualCohortSlotsV1,
    pub token_receipt: FrozenPerceptualParticipantTokenGenerationReceiptV1,
    pub identity_policy: FrozenParticipantIdentityBoundaryPolicyV1,
    pub schedule: PerceptualParticipantScheduleBookV1,
    pub enrollment_policy: FrozenPerceptualEnrollmentPolicyV1,
    pub collection_authority: FrozenPerceptualCollectionAuthorityV1,
    pub withdrawal_policy: FrozenPerceptualWithdrawalPolicyV1,
    pub authenticity_policy: FrozenPerceptualCollectionAuthenticityPolicyV1,
    pub witness_policy: FrozenPerceptualEnrollmentWitnessPolicyV1,
    pub allocation_store: DurableEnrollmentAllocationStoreV1,
    pub coordinator_store: DurableEnrollmentCoordinatorStoreV1,
    root: PathBuf,
    collection_provider_root: PathBuf,
    witness_provider_root: PathBuf,
}

impl Mel003EnrollmentE2eQualificationFixtureV1 {
    pub fn new_valid() -> Result<Self, String> {
        let root = qualification_root();
        let allocation_root = root.join("allocator");
        let coordinator_root = root.join("coordinator");
        let collection_provider_root = root.join("provider-collection");
        let witness_provider_root = root.join("provider-witness");

        let protocol = protocol();
        if !protocol.validate().is_empty() {
            return Err("qualification protocol did not validate".into());
        }
        let render_binding = render_binding(&protocol);
        if !render_binding.validate(&protocol).is_empty() {
            return Err("qualification render binding did not validate".into());
        }
        let stimulus_pack = stimulus_pack(&protocol, &render_binding);
        if !stimulus_pack.validate(&protocol, &render_binding).is_empty() {
            return Err("qualification stimulus pack did not validate".into());
        }

        let generator = ParticipantTokenGeneratorIdentityV1 {
            source_revision: "0123456789abcdef0123456789abcdef01234567".into(),
            binary_sha256: digest(30),
            environment_sha256: digest(31),
            generator_version: PARTICIPANT_TOKEN_GENERATOR_VERSION.into(),
        };
        let (cohort, token_receipt) = generate_perceptual_cohort_slots_os_rng(
            &protocol,
            "mel003-e2e-qualification-cohort-v1",
            generator,
        )
        .map_err(|issues| format!("token generation invalid: {issues:?}"))?;
        let token_issues =
            validate_participant_token_generation_receipt(&protocol, &cohort, &token_receipt);
        if !token_issues.is_empty() {
            return Err(format!("token receipt invalid: {token_issues:?}"));
        }

        let mut identity_policy = FrozenParticipantIdentityBoundaryPolicyV1 {
            policy_version: PARTICIPANT_IDENTITY_BOUNDARY_POLICY_VERSION.into(),
            token_generation_receipt_sha256: token_receipt.receipt_sha256.clone(),
            stable_participant_token_bearer_access_prohibited: true,
            full_schedule_book_client_exposure_prohibited: true,
            cohort_enumeration_prohibited: true,
            recruitment_linkage_in_study_evidence_prohibited: true,
            policy_sha256: String::new(),
        };
        seal_participant_identity_boundary_policy(&mut identity_policy)
            .map_err(|e| e.to_string())?;
        if !validate_participant_identity_boundary_policy(&token_receipt, &identity_policy).is_empty() {
            return Err("identity boundary policy did not validate".into());
        }

        let (schedule, _private_audit) = build_perceptual_participant_schedule(
            &protocol,
            &stimulus_pack,
            &render_binding,
            &cohort,
            SCHEDULE_SECRET,
        )
        .map_err(|issues| format!("schedule invalid: {issues:?}"))?;

        let mut enrollment_policy = FrozenPerceptualEnrollmentPolicyV1 {
            policy_version: PERCEPTUAL_ENROLLMENT_POLICY_VERSION.into(),
            protocol_sha256: canonical_json_sha256(&protocol).map_err(|e| e.to_string())?,
            token_generation_receipt_sha256: token_receipt.receipt_sha256.clone(),
            participant_identity_boundary_policy_sha256: identity_policy.policy_sha256.clone(),
            participant_schedule_sha256: canonical_json_sha256(&schedule).map_err(|e| e.to_string())?,
            consent_form_sha256: digest(40),
            participant_information_sha256: digest(41),
            eligibility_policy_sha256: digest(42),
            allocation_order: EnrollmentSlotAllocationOrderV1::LexicographicParticipantToken,
            screening_failure_is_pre_enrollment: true,
            pre_enrollment_withdrawal_is_pre_allocation: true,
            eligibility_required_before_allocation: true,
            operator_schedule_choice_prohibited: true,
            one_slot_per_eligibility_gate: true,
            scored_access_before_allocation_prohibited: true,
            policy_sha256: String::new(),
        };
        seal_enrollment_policy(&mut enrollment_policy).map_err(|e| e.to_string())?;
        let policy_issues = validate_enrollment_policy(
            &protocol,
            &cohort,
            &token_receipt,
            &identity_policy,
            &schedule,
            &enrollment_policy,
        );
        if !policy_issues.is_empty() {
            return Err(format!("enrollment policy invalid: {policy_issues:?}"));
        }

        let empty_ledger = new_enrollment_allocation_ledger(
            &enrollment_policy,
            &token_receipt,
            &schedule,
        )
        .map_err(|e| e.to_string())?;
        let ledger_issues = validate_enrollment_allocation_ledger(
            &protocol,
            &stimulus_pack,
            &render_binding,
            &cohort,
            &token_receipt,
            &identity_policy,
            &schedule,
            &enrollment_policy,
            &[],
            &empty_ledger,
        );
        if !ledger_issues.is_empty() {
            return Err(format!("empty enrollment ledger invalid: {ledger_issues:?}"));
        }

        let mut collection_authority = FrozenPerceptualCollectionAuthorityV1 {
            authority_version: PERCEPTUAL_COLLECTION_AUTHORITY_VERSION.into(),
            protocol_sha256: canonical_json_sha256(&protocol).map_err(|e| e.to_string())?,
            stimulus_pack_sha256: canonical_json_sha256(&stimulus_pack).map_err(|e| e.to_string())?,
            participant_schedule_sha256: canonical_json_sha256(&schedule).map_err(|e| e.to_string())?,
            external_preregistration_sha256: protocol.external_preregistration.record_sha256.clone(),
            consent_form_sha256: enrollment_policy.consent_form_sha256.clone(),
            participant_information_sha256: enrollment_policy.participant_information_sha256.clone(),
            privacy_notice_sha256: digest(43),
            recruitment_material_sha256: digest(44),
            compensation_policy_sha256: digest(45),
            human_study_review_disposition: HumanStudyReviewDispositionV1::NotRequiredByResponsibleAuthority,
            human_study_review_reference: "qualification-fixture-no-human-collection".into(),
            human_study_review_evidence_sha256: digest(46),
            planned_open_utc: "2026-09-21T00:00:00Z".into(),
            planned_close_utc: "2026-09-22T00:00:00Z".into(),
            outcome_monitoring_prohibited: true,
            private_audit_access_prohibited: true,
            randomization_key_access_prohibited: true,
            raw_identity_collection_prohibited: true,
            response_correctness_computation_prohibited: true,
            arm_label_derivation_prohibited: true,
            collection_roles: vec![
                PerceptualCollectionRoleV1::CollectionOperator,
                PerceptualCollectionRoleV1::EvidenceCustodian,
                PerceptualCollectionRoleV1::BlindedMonitor,
                PerceptualCollectionRoleV1::GovernanceOfficer,
            ],
            runner: PerceptualCollectionRunnerIdentityV1 {
                source_revision: SOURCE_COMMIT.into(),
                binary_sha256: digest(47),
                environment_sha256: digest(48),
                runner_version: "qualification-collection-runner-v1".into(),
            },
            authority_sha256: String::new(),
        };
        seal_collection_authority(&mut collection_authority).map_err(|e| e.to_string())?;
        let authority_issues = validate_collection_authority(
            &protocol,
            &stimulus_pack,
            &render_binding,
            &schedule,
            &collection_authority,
        );
        if !authority_issues.is_empty() {
            return Err(format!("collection authority invalid: {authority_issues:?}"));
        }

        let collection_key = CollectionSigningKeyV1::from_seed(
            "qualification-collection-signer",
            1,
            QUALIFICATION_COLLECTION_SEED,
        )
        .map_err(|e| format!("collection key invalid: {e:?}"))?;
        let witness_key = CollectionSigningKeyV1::from_seed(
            "qualification-independent-witness",
            1,
            QUALIFICATION_WITNESS_SEED,
        )
        .map_err(|e| format!("witness key invalid: {e:?}"))?;
        let withdrawal_key = CollectionSigningKeyV1::from_seed(
            "qualification-withdrawal-governance",
            1,
            QUALIFICATION_WITHDRAWAL_SEED,
        )
        .map_err(|e| format!("withdrawal key invalid: {e:?}"))?;

        let mut withdrawal_policy = FrozenPerceptualWithdrawalPolicyV1 {
            policy_version: PERCEPTUAL_WITHDRAWAL_POLICY_VERSION.into(),
            collection_authority_sha256: collection_authority.authority_sha256.clone(),
            consent_form_sha256: collection_authority.consent_form_sha256.clone(),
            participant_information_sha256: collection_authority.participant_information_sha256.clone(),
            privacy_notice_sha256: collection_authority.privacy_notice_sha256.clone(),
            human_study_review_evidence_sha256: collection_authority.human_study_review_evidence_sha256.clone(),
            collection_chronology_policy_sha256: digest(49),
            retention_mode: WithdrawalEvidenceRetentionModeV1::MinimalAuditCommitment,
            withdrawal_allowed_until_collection_close: true,
            raw_session_data_deleted_on_withdrawal: true,
            raw_participant_token_prohibited_in_authenticity_log: true,
            outcome_aware_withdrawal_processing_prohibited: true,
            arm_label_access_prohibited: true,
            withdrawal_authority: withdrawal_key.verifier_identity(),
            policy_sha256: String::new(),
        };
        seal_withdrawal_policy(&mut withdrawal_policy).map_err(|e| e.to_string())?;
        let withdrawal_issues = validate_withdrawal_policy(&withdrawal_policy, &collection_authority);
        if !withdrawal_issues.is_empty() {
            return Err(format!("withdrawal policy invalid: {withdrawal_issues:?}"));
        }

        let mut authenticity_policy = FrozenPerceptualCollectionAuthenticityPolicyV1 {
            policy_version: PERCEPTUAL_COLLECTION_AUTHENTICITY_POLICY_VERSION.into(),
            study_domain: PERCEPTUAL_COLLECTION_STUDY_DOMAIN.into(),
            collection_authority_sha256: collection_authority.authority_sha256.clone(),
            withdrawal_policy_sha256: withdrawal_policy.policy_sha256.clone(),
            collection_signer: collection_key.verifier_identity(),
            witness_signer: witness_key.verifier_identity(),
            witness_log_id: "mel003-qualification-collection-log-v1".into(),
            per_entry_external_witness_required: true,
            unblinding_requires_verified_close_anchor: true,
            policy_sha256: String::new(),
        };
        seal_authenticity_policy(&mut authenticity_policy).map_err(|e| e.to_string())?;
        let authenticity_issues = validate_authenticity_policy(
            &authenticity_policy,
            &withdrawal_policy,
            &collection_authority,
        );
        if !authenticity_issues.is_empty() {
            return Err(format!("authenticity policy invalid: {authenticity_issues:?}"));
        }

        let mut witness_policy = FrozenPerceptualEnrollmentWitnessPolicyV1 {
            policy_version: PERCEPTUAL_ENROLLMENT_WITNESS_POLICY_VERSION.into(),
            enrollment_policy_sha256: enrollment_policy.policy_sha256.clone(),
            collection_authenticity_policy_sha256: authenticity_policy.policy_sha256.clone(),
            witness_log_id: "mel003-qualification-enrollment-witness-log-v1".into(),
            collection_signer: authenticity_policy.collection_signer.clone(),
            witness_signer: authenticity_policy.witness_signer.clone(),
            per_allocation_external_witness_required: true,
            scored_authority_requires_verified_witness: true,
            raw_participant_token_prohibited: true,
            policy_sha256: String::new(),
        };
        seal_enrollment_witness_policy(&mut witness_policy).map_err(|e| e.to_string())?;
        let witness_policy_issues = validate_enrollment_witness_policy(
            &witness_policy,
            &enrollment_policy,
            &authenticity_policy,
        );
        if !witness_policy_issues.is_empty() {
            return Err(format!("witness policy invalid: {witness_policy_issues:?}"));
        }

        let empty_witness_bundle =
            new_external_enrollment_witness_candidate_bundle(&witness_policy)
                .map_err(|e| e.to_string())?;
        let bundle_issues = validate_enrollment_witness_bundle(
            &enrollment_policy,
            &authenticity_policy,
            &witness_policy,
            &empty_ledger,
            &empty_witness_bundle,
        );
        if !bundle_issues.is_empty() {
            return Err(format!("empty witness bundle invalid: {bundle_issues:?}"));
        }

        let allocation_store = DurableEnrollmentAllocationStoreV1::new(&allocation_root);
        let initial_allocator_state = allocation_store
            .initialize(
                &protocol,
                &stimulus_pack,
                &render_binding,
                &cohort,
                &token_receipt,
                &identity_policy,
                &schedule,
                &enrollment_policy,
                &empty_ledger,
            )
            .map_err(|e| e.to_string())?;
        let initial_coordinator = new_confirmed_enrollment_coordinator_state(
            &protocol,
            &stimulus_pack,
            &render_binding,
            &cohort,
            &token_receipt,
            &identity_policy,
            &schedule,
            &enrollment_policy,
            &witness_policy,
            &authenticity_policy,
            &initial_allocator_state,
            &empty_witness_bundle,
        )
        .map_err(|issues| format!("initial coordinator invalid: {issues:?}"))?;
        let coordinator_store = DurableEnrollmentCoordinatorStoreV1::new(&coordinator_root);
        coordinator_store
            .initialize(&initial_coordinator)
            .map_err(|e| e.to_string())?;

        Ok(Self {
            protocol,
            render_binding,
            stimulus_pack,
            cohort,
            token_receipt,
            identity_policy,
            schedule,
            enrollment_policy,
            collection_authority,
            withdrawal_policy,
            authenticity_policy,
            witness_policy,
            allocation_store,
            coordinator_store,
            root,
            collection_provider_root,
            witness_provider_root,
        })
    }

    pub fn valid_next_eligibility_gate(
        &self,
        ordinal: usize,
    ) -> Result<FrozenPerceptualEligibilityGateReceiptV1, String> {
        let mut gate = FrozenPerceptualEligibilityGateReceiptV1 {
            gate_version: PERCEPTUAL_ELIGIBILITY_GATE_VERSION.into(),
            enrollment_policy_sha256: self.enrollment_policy.policy_sha256.clone(),
            protocol_sha256: canonical_json_sha256(&self.protocol).map_err(|e| e.to_string())?,
            consent_form_sha256: self.enrollment_policy.consent_form_sha256.clone(),
            participant_information_sha256: self
                .enrollment_policy
                .participant_information_sha256
                .clone(),
            eligibility_attempt_sha256: digest(1000 + ordinal * 2),
            eligibility_chronology_event_sha256: digest(1001 + ordinal * 2),
            informed_consent_confirmed: true,
            minimum_age_eligibility_confirmed: true,
            stereo_playback_check_passed: true,
            comprehension_practice_passed: true,
            eligible_for_enrollment: true,
            gate_sha256: String::new(),
        };
        seal_eligibility_gate(&mut gate).map_err(|e| e.to_string())?;
        let issues = validate_eligibility_gate(&self.protocol, &self.enrollment_policy, &gate);
        if issues.is_empty() {
            Ok(gate)
        } else {
            Err(format!("eligibility gate invalid: {issues:?}"))
        }
    }

    pub fn orchestrator(&self) -> PerceptualEnrollmentWitnessOrchestratorV1<'_> {
        PerceptualEnrollmentWitnessOrchestratorV1 {
            protocol: &self.protocol,
            stimulus_pack: &self.stimulus_pack,
            render_binding: &self.render_binding,
            cohort: &self.cohort,
            token_receipt: &self.token_receipt,
            identity_policy: &self.identity_policy,
            schedule: &self.schedule,
            enrollment_policy: &self.enrollment_policy,
            authenticity_policy: &self.authenticity_policy,
            witness_policy: &self.witness_policy,
            allocation_store: &self.allocation_store,
            coordinator_store: &self.coordinator_store,
        }
    }

    pub fn provider(
        &self,
    ) -> Result<DurableQualificationEnrollmentWitnessProviderV1, QualificationProviderFixtureErrorV1>
    {
        DurableQualificationEnrollmentWitnessProviderV1::new(
            &self.collection_provider_root,
            &self.witness_provider_root,
            &self.witness_policy,
            QUALIFICATION_COLLECTION_SEED,
            QUALIFICATION_WITNESS_SEED,
        )
    }

    pub fn coordinator_state(&self) -> Result<DurablePerceptualEnrollmentCoordinatorStateV1, String> {
        self.coordinator_store.inspect().map_err(|e| e.to_string())
    }

    pub fn current_allocator_state(&self) -> Result<DurableEnrollmentAllocationStateV1, String> {
        let coordinator = self.coordinator_state()?;
        inspect_validated_current_from_confirmed_head(
            &self.allocation_store,
            &self.protocol,
            &self.stimulus_pack,
            &self.render_binding,
            &self.cohort,
            &self.token_receipt,
            &self.identity_policy,
            &self.schedule,
            &self.enrollment_policy,
            &coordinator.confirmed_enrollment_ledger.ledger_sha256,
        )
        .map_err(|e| e.to_string())
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

impl Drop for Mel003EnrollmentE2eQualificationFixtureV1 {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
