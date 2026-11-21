// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Offline deterministic RMI-Bench Track A fixture runner.
//!
//! This executes the synthetic oracle against explicit positive/negative
//! controls and emits a digest-bound JSON report. It does not execute
//! Melothaea, render audio, score musical relevance, or establish human
//! collaboration. Those require separate candidate adapters and studies.

use crate::rmi_benchmark::{
    evaluate_rmi_pair, evaluate_rmi_reciprocal_pair, FixedPartnerResponsePolicyV1,
    FixedResponsePolicyV1, ImitationOnlyPolicyV1, OneWayPromptPolicyV1,
    RmiAgentActionV1, RmiEvaluationEnvelopeV1, RmiHistoryActorV1, RmiHistoryEventV1,
    YokedReplayPolicyV1,
    RmiInterventionPairV1, RmiMusicalFeaturesV1, RmiPartnerObservationV1,
    RmiPartnerResponseV1, RmiReciprocalEvaluationEnvelopeV1, RmiRoleV1,
    RmiTargetFeatureV1,
    SyntheticContingentPolicyV1, SyntheticResponsivePartnerPolicyV1,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt::Write as FmtWrite;

pub const RMI_TRACK_A_RUNNER_VERSION_V1: &str = "rmi-track-a-offline-runner-v1";
pub const RMI_TRACK_A_MAX_TRIALS_PER_TARGET_V1: usize = 1_024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmiTrackAManifestV1 {
    pub schema_version: u32,
    pub runner_version: String,
    pub suite_id: String,
    pub crate_version: String,
    pub source_revision: Option<String>,
    pub rustc_version: Option<String>,
    pub target_arch: String,
    pub target_os: String,
    pub seed_schedule_id: String,
    pub seed_root: u64,
    pub trials_per_target: usize,
    pub targets: Vec<RmiTargetFeatureV1>,
    pub agent_policy_ids: Vec<String>,
    pub partner_policy_ids: Vec<String>,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmiTrackATrialV1 {
    pub trial_id: String,
    pub seed: u64,
    pub target: RmiTargetFeatureV1,
    pub intervention_pair_sha256: String,
    pub yoked_donor_trial_id: String,
    pub yoked_donor_fixture_sha256: String,
    pub fixed_agent: Result<RmiEvaluationEnvelopeV1, String>,
    pub one_way_prompt_agent: Result<RmiEvaluationEnvelopeV1, String>,
    pub yoked_replay_agent: Result<RmiEvaluationEnvelopeV1, String>,
    pub unrelated_feature_agent: Result<RmiEvaluationEnvelopeV1, String>,
    pub imitation_agent: Result<RmiEvaluationEnvelopeV1, String>,
    pub synthetic_contingent_agent: Result<RmiEvaluationEnvelopeV1, String>,
    pub fixed_agent_with_responsive_partner: Result<RmiReciprocalEvaluationEnvelopeV1, String>,
    pub synthetic_agent_with_fixed_partner: Result<RmiReciprocalEvaluationEnvelopeV1, String>,
    pub synthetic_agent_with_responsive_partner: Result<RmiReciprocalEvaluationEnvelopeV1, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RmiTrackASummaryV1 {
    pub trials: usize,
    pub evaluator_errors: usize,
    pub fixed_agent_zero_sensitivity: usize,
    pub one_way_prompt_zero_sensitivity: usize,
    pub yoked_replay_zero_sensitivity: usize,
    pub unrelated_feature_zero_target_sensitivity: usize,
    pub imitation_agent_target_sensitive: usize,
    pub positive_agent_target_sensitive: usize,
    pub positive_agent_zero_collateral: usize,
    pub fixed_agent_did_not_influence_partner: usize,
    pub fixed_partner_did_not_respond: usize,
    pub reciprocal_positive_control_both_directions: usize,
    pub reciprocal_positive_control_zero_collateral: usize,
    pub agent_order_swap_verified: usize,
    pub partner_order_swap_verified: usize,
    /// All registered synthetic positive/negative-control expectations held.
    /// This is evaluator-fixture qualification, not a Melothaea score.
    pub synthetic_fixture_expectations_satisfied: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmiTrackAReportPayloadV1 {
    pub manifest: RmiTrackAManifestV1,
    pub manifest_sha256: String,
    pub summary: RmiTrackASummaryV1,
    pub trials: Vec<RmiTrackATrialV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmiTrackAReportV1 {
    pub payload: RmiTrackAReportPayloadV1,
    pub payload_sha256: String,
}

fn targets_v1() -> Vec<RmiTargetFeatureV1> {
    vec![
        RmiTargetFeatureV1::Motif,
        RmiTargetFeatureV1::Rhythm,
        RmiTargetFeatureV1::Tempo,
        RmiTargetFeatureV1::Role,
    ]
}

fn target_name_v1(target: RmiTargetFeatureV1) -> &'static str {
    match target {
        RmiTargetFeatureV1::Motif => "motif",
        RmiTargetFeatureV1::Rhythm => "rhythm",
        RmiTargetFeatureV1::Tempo => "tempo",
        RmiTargetFeatureV1::Role => "role",
    }
}

fn unrelated_target_v1(target: RmiTargetFeatureV1) -> RmiTargetFeatureV1 {
    match target {
        RmiTargetFeatureV1::Motif => RmiTargetFeatureV1::Rhythm,
        RmiTargetFeatureV1::Rhythm => RmiTargetFeatureV1::Tempo,
        RmiTargetFeatureV1::Tempo => RmiTargetFeatureV1::Role,
        RmiTargetFeatureV1::Role => RmiTargetFeatureV1::Motif,
    }
}

fn target_index_v1(target: RmiTargetFeatureV1) -> u64 {
    match target {
        RmiTargetFeatureV1::Motif => 0,
        RmiTargetFeatureV1::Rhythm => 1,
        RmiTargetFeatureV1::Tempo => 2,
        RmiTargetFeatureV1::Role => 3,
    }
}

fn mutate_target_v1(
    features: &mut RmiMusicalFeaturesV1,
    target: RmiTargetFeatureV1,
    trial_index: usize,
) {
    match target {
        RmiTargetFeatureV1::Motif => {
            let note = trial_index % features.motif_intervals.len();
            let delta = if trial_index % 2 == 0 { 2 } else { -2 };
            features.motif_intervals[note] += delta;
        }
        RmiTargetFeatureV1::Rhythm => {
            let step = trial_index % features.rhythm_onsets.len();
            features.rhythm_onsets[step] = !features.rhythm_onsets[step];
        }
        RmiTargetFeatureV1::Tempo => {
            let delta = if trial_index % 2 == 0 { 1_500 } else { -1_500 };
            features.tempo_milli_bpm = (features.tempo_milli_bpm as i64 + delta) as u32;
        }
        RmiTargetFeatureV1::Role => {
            features.role = match features.role {
                RmiRoleV1::Lead => RmiRoleV1::Accompaniment,
                RmiRoleV1::Accompaniment => RmiRoleV1::Lead,
                RmiRoleV1::Bass => RmiRoleV1::Counterline,
                RmiRoleV1::Counterline => RmiRoleV1::Bass,
            };
        }
    }
}

/// Build one paired fixture. Both arms share exact history and event identity;
/// only the declared target feature is changed.
pub fn make_rmi_track_a_fixture_v1(
    target: RmiTargetFeatureV1,
    trial_index: usize,
    seed_root: u64,
) -> Result<RmiInterventionPairV1, String> {
    if trial_index >= RMI_TRACK_A_MAX_TRIALS_PER_TARGET_V1 {
        return Err(format!(
            "trial_index must be below {RMI_TRACK_A_MAX_TRIALS_PER_TARGET_V1}"
        ));
    }
    // A disjoint ordinal block per target guarantees unique trial seeds within
    // a suite and avoids relying on OS or process randomness.
    let target_offset = target_index_v1(target)
        .checked_mul(RMI_TRACK_A_MAX_TRIALS_PER_TARGET_V1 as u64)
        .ok_or_else(|| "target seed offset overflow".to_owned())?;
    let seed_offset = target_offset
        .checked_add(trial_index as u64)
        .ok_or_else(|| "trial seed offset overflow".to_owned())?;
    let seed = seed_root.wrapping_add(seed_offset);

    let mut control = RmiMusicalFeaturesV1 {
        motif_intervals: [2, -1, 3, 0],
        rhythm_onsets: [true, false, true, false, true, false, false, true],
        tempo_milli_bpm: 120_000,
        role: if trial_index % 2 == 0 { RmiRoleV1::Lead } else { RmiRoleV1::Bass },
    };
    // Vary the frozen context without changing the intervention definition.
    control.motif_intervals[0] += (trial_index % 4) as i16;
    control.tempo_milli_bpm += (trial_index % 3) as u32 * 250;

    let mut treatment = control.clone();
    mutate_target_v1(&mut treatment, target, trial_index);

    let trial_id = format!(
        "rmi-track-a-{}-{:06}",
        target_name_v1(target),
        trial_index
    );
    let history = vec![
        RmiHistoryEventV1 {
            event_id: format!("{trial_id}-history-partner"),
            actor: RmiHistoryActorV1::Partner,
            features: control.clone(),
        },
        crate::rmi_benchmark::RmiHistoryEventV1 {
            event_id: format!("{trial_id}-history-agent"),
            actor: RmiHistoryActorV1::Agent,
            features: control.clone(),
        },
    ];
    let pre_state_sha256 = digest_v1(&history)?;
    let pair = RmiInterventionPairV1 {
        trial_id,
        seed,
        pre_intervention_history: history,
        pre_intervention_state_sha256: pre_state_sha256,
        target,
        control_observation: RmiPartnerObservationV1 {
            source_event_id: format!("partner-event-{}-{trial_index:06}", target_name_v1(target)),
            features: control,
        },
        treatment_observation: RmiPartnerObservationV1 {
            source_event_id: format!("partner-event-{}-{trial_index:06}", target_name_v1(target)),
            features: treatment,
        },
    };
    crate::rmi_benchmark::validate_rmi_intervention_pair(&pair)
        .map_err(|issues| format!("generated invalid fixture: {issues:?}"))?;
    Ok(pair)
}

fn fixed_agent_v1() -> FixedResponsePolicyV1 {
    FixedResponsePolicyV1 {
        id: "fixed-response-agent-v1".into(),
        version: "1".into(),
        response: RmiAgentActionV1 {
            action_id: "fixed-response".into(),
            features: RmiMusicalFeaturesV1 {
                motif_intervals: [2, -1, 3, 0],
                rhythm_onsets: [true, false, true, false, true, false, false, true],
                tempo_milli_bpm: 120_000,
                role: RmiRoleV1::Lead,
            },
        },
    }
}

fn imitation_agent_v1() -> ImitationOnlyPolicyV1 {
    ImitationOnlyPolicyV1 {
        id: "imitation-only-agent-v1".into(),
        version: "1".into(),
    }
}

fn one_way_prompt_agent_v1() -> OneWayPromptPolicyV1 {
    OneWayPromptPolicyV1 {
        id: "one-way-prompt-agent-v1".into(),
        version: "1".into(),
        frozen_response: fixed_agent_v1().response,
    }
}

fn yoked_replay_agent_v1(
    target: RmiTargetFeatureV1,
    trial_index: usize,
    seed_root: u64,
) -> Result<(YokedReplayPolicyV1, String, String), String> {
    // Use a distinct, deterministic donor fixture from a matched trial.
    // It is generated before evaluation and does not receive the current
    // treatment observation; the same frozen donor response is replayed in
    // both counterfactual arms.
    let donor_index = (trial_index + 1) % RMI_TRACK_A_MAX_TRIALS_PER_TARGET_V1;
    let donor = make_rmi_track_a_fixture_v1(target, donor_index, seed_root)?;
    let donor_digest = digest_v1(&donor)?;
    let donor_trial_id = donor.trial_id.clone();
    let policy = YokedReplayPolicyV1 {
        id: "yoked-replay-agent-v1".into(),
        version: "1".into(),
        replayed_response: RmiAgentActionV1 {
            action_id: format!("yoked-from-{}", donor.trial_id),
            features: donor.control_observation.features,
        },
    };
    Ok((policy, donor_trial_id, donor_digest))
}

fn contingent_agent_v1(target: RmiTargetFeatureV1) -> SyntheticContingentPolicyV1 {
    SyntheticContingentPolicyV1 {
        id: format!("synthetic-contingent-agent-{}", target_name_v1(target)),
        version: "1".into(),
        target,
    }
}

fn fixed_partner_v1() -> FixedPartnerResponsePolicyV1 {
    FixedPartnerResponsePolicyV1 {
        id: "fixed-response-partner-v1".into(),
        version: "1".into(),
        response: RmiPartnerResponseV1 {
            response_id: "fixed-partner-response".into(),
            features: RmiMusicalFeaturesV1 {
                motif_intervals: [2, -1, 3, 0],
                rhythm_onsets: [true, false, true, false, true, false, false, true],
                tempo_milli_bpm: 120_000,
                role: RmiRoleV1::Accompaniment,
            },
        },
    }
}

fn responsive_partner_v1(target: RmiTargetFeatureV1) -> SyntheticResponsivePartnerPolicyV1 {
    SyntheticResponsivePartnerPolicyV1 {
        id: format!("synthetic-responsive-partner-{}", target_name_v1(target)),
        version: "1".into(),
        target,
    }
}

fn capture<T>(result: Result<T, impl std::fmt::Debug>) -> Result<T, String> {
    result.map_err(|error| format!("{error:?}"))
}

fn digest_v1<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    let digest = Sha256::digest(&bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    Ok(encoded)
}

pub fn run_rmi_track_a_suite_v1(
    trials_per_target: usize,
    seed_root: u64,
) -> Result<RmiTrackAReportV1, String> {
    if trials_per_target == 0 || trials_per_target > RMI_TRACK_A_MAX_TRIALS_PER_TARGET_V1 {
        return Err(format!(
            "trials_per_target must be in 1..={RMI_TRACK_A_MAX_TRIALS_PER_TARGET_V1}"
        ));
    }

    let targets = targets_v1();
    let manifest = RmiTrackAManifestV1 {
        schema_version: 1,
        runner_version: RMI_TRACK_A_RUNNER_VERSION_V1.into(),
        suite_id: "rmi-track-a-synthetic-v1".into(),
        crate_version: env!("CARGO_PKG_VERSION").to_owned(),
        source_revision: option_env!("SYMTHAEA_MUSE_GIT_REV").map(str::to_owned),
        rustc_version: option_env!("RUSTC_VERSION").map(str::to_owned),
        target_arch: std::env::consts::ARCH.to_owned(),
        target_os: std::env::consts::OS.to_owned(),
        seed_schedule_id: "target_index*max_trials_per_target+trial_index-v1".into(),
        seed_root,
        trials_per_target,
        targets: targets.clone(),
        agent_policy_ids: vec![
            "fixed-response-agent-v1".into(),
            "one-way-prompt-agent-v1".into(),
            "yoked-replay-agent-v1".into(),
            "unrelated-feature-agent-v1".into(),
            "imitation-only-agent-v1".into(),
            "synthetic-contingent-agent-motif".into(),
            "synthetic-contingent-agent-rhythm".into(),
            "synthetic-contingent-agent-tempo".into(),
            "synthetic-contingent-agent-role".into(),
        ],
        partner_policy_ids: vec![
            "fixed-response-partner-v1".into(),
            "synthetic-responsive-partner-motif".into(),
            "synthetic-responsive-partner-rhythm".into(),
            "synthetic-responsive-partner-tempo".into(),
            "synthetic-responsive-partner-role".into(),
        ],
        claim_boundary: "Synthetic evaluator qualification only; not a Melothaea performance, musical-quality, human-influence, collaboration-value, authorship, or consciousness result.".into(),
    };
    let manifest_sha256 = digest_v1(&manifest)?;
    let total_trials = trials_per_target
        .checked_mul(targets.len())
        .ok_or_else(|| "total trial count overflow".to_owned())?;

    let mut trials = Vec::with_capacity(total_trials);
    for target in targets {
        for trial_index in 0..trials_per_target {
            let pair = make_rmi_track_a_fixture_v1(target, trial_index, seed_root)?;
            let pair_sha256 = digest_v1(&pair)?;

            let fixed_agent = capture(evaluate_rmi_pair(&fixed_agent_v1(), &pair));
            let one_way_prompt_agent =
                capture(evaluate_rmi_pair(&one_way_prompt_agent_v1(), &pair));
            let (yoked_policy, yoked_donor_trial_id, yoked_donor_fixture_sha256) =
                yoked_replay_agent_v1(target, trial_index, seed_root)?;
            let yoked_replay_agent =
                capture(evaluate_rmi_pair(&yoked_policy, &pair));
            let unrelated_feature_agent = capture(evaluate_rmi_pair(
                &contingent_agent_v1(unrelated_target_v1(target)),
                &pair,
            ));
            let imitation_agent = capture(evaluate_rmi_pair(&imitation_agent_v1(), &pair));
            let synthetic_contingent_agent =
                capture(evaluate_rmi_pair(&contingent_agent_v1(target), &pair));

            let fixed_agent_with_responsive_partner = capture(evaluate_rmi_reciprocal_pair(
                &fixed_agent_v1(),
                &responsive_partner_v1(target),
                &pair,
            ));
            let synthetic_agent_with_fixed_partner = capture(evaluate_rmi_reciprocal_pair(
                &contingent_agent_v1(target),
                &fixed_partner_v1(),
                &pair,
            ));
            let synthetic_agent_with_responsive_partner = capture(evaluate_rmi_reciprocal_pair(
                &contingent_agent_v1(target),
                &responsive_partner_v1(target),
                &pair,
            ));

            trials.push(RmiTrackATrialV1 {
                trial_id: pair.trial_id.clone(),
                seed: pair.seed,
                target,
                intervention_pair_sha256: pair_sha256,
                yoked_donor_trial_id,
                yoked_donor_fixture_sha256,
                fixed_agent,
                one_way_prompt_agent,
                yoked_replay_agent,
                unrelated_feature_agent,
                imitation_agent,
                synthetic_contingent_agent,
                fixed_agent_with_responsive_partner,
                synthetic_agent_with_fixed_partner,
                synthetic_agent_with_responsive_partner,
            });
        }
    }

    let mut summary = RmiTrackASummaryV1 {
        trials: trials.len(),
        ..RmiTrackASummaryV1::default()
    };
    for trial in &trials {
        let outcomes = [
            trial.fixed_agent.is_ok(),
            trial.one_way_prompt_agent.is_ok(),
            trial.yoked_replay_agent.is_ok(),
            trial.unrelated_feature_agent.is_ok(),
            trial.imitation_agent.is_ok(),
            trial.synthetic_contingent_agent.is_ok(),
            trial.fixed_agent_with_responsive_partner.is_ok(),
            trial.synthetic_agent_with_fixed_partner.is_ok(),
            trial.synthetic_agent_with_responsive_partner.is_ok(),
        ];
        summary.evaluator_errors += outcomes.iter().filter(|ok| !**ok).count();

        if let Ok(result) = &trial.fixed_agent {
            summary.fixed_agent_zero_sensitivity +=
                usize::from(!result.payload.targeted_input_sensitivity);
            summary.agent_order_swap_verified += usize::from(result.payload.arm_order_swap_verified);
        }
        if let Ok(result) = &trial.one_way_prompt_agent {
            summary.one_way_prompt_zero_sensitivity +=
                usize::from(!result.payload.targeted_input_sensitivity);
            summary.agent_order_swap_verified += usize::from(result.payload.arm_order_swap_verified);
        }
        if let Ok(result) = &trial.yoked_replay_agent {
            summary.yoked_replay_zero_sensitivity +=
                usize::from(!result.payload.targeted_input_sensitivity);
            summary.agent_order_swap_verified += usize::from(result.payload.arm_order_swap_verified);
        }
        if let Ok(result) = &trial.unrelated_feature_agent {
            summary.unrelated_feature_zero_target_sensitivity +=
                usize::from(!result.payload.targeted_input_sensitivity);
            summary.agent_order_swap_verified += usize::from(result.payload.arm_order_swap_verified);
        }
        if let Ok(result) = &trial.imitation_agent {
            summary.imitation_agent_target_sensitive +=
                usize::from(result.payload.targeted_input_sensitivity);
            summary.agent_order_swap_verified += usize::from(result.payload.arm_order_swap_verified);
        }
        if let Ok(result) = &trial.synthetic_contingent_agent {
            summary.positive_agent_target_sensitive +=
                usize::from(result.payload.targeted_input_sensitivity);
            summary.positive_agent_zero_collateral +=
                usize::from(result.payload.collateral_changed_feature_count == 0);
            summary.agent_order_swap_verified += usize::from(result.payload.arm_order_swap_verified);
        }
        if let Ok(result) = &trial.fixed_agent_with_responsive_partner {
            summary.fixed_agent_did_not_influence_partner +=
                usize::from(!result.payload.targeted_partner_sensitivity);
            summary.partner_order_swap_verified +=
                usize::from(result.payload.partner_arm_order_swap_verified);
        }
        if let Ok(result) = &trial.synthetic_agent_with_fixed_partner {
            summary.fixed_partner_did_not_respond +=
                usize::from(!result.payload.targeted_partner_sensitivity);
            summary.partner_order_swap_verified +=
                usize::from(result.payload.partner_arm_order_swap_verified);
        }
        if let Ok(result) = &trial.synthetic_agent_with_responsive_partner {
            summary.partner_order_swap_verified +=
                usize::from(result.payload.partner_arm_order_swap_verified);
            let both = result.payload.agent_targeted_sensitivity
                && result.payload.targeted_partner_sensitivity;
            summary.reciprocal_positive_control_both_directions += usize::from(both);
            summary.reciprocal_positive_control_zero_collateral += usize::from(
                both
                    && result.payload.partner_collateral_changed_feature_count == 0
                    && result.payload.partner_arm_order_swap_verified,
            );
        }
    }

    summary.synthetic_fixture_expectations_satisfied = summary.evaluator_errors == 0
        && summary.fixed_agent_zero_sensitivity == summary.trials
        && summary.one_way_prompt_zero_sensitivity == summary.trials
        && summary.yoked_replay_zero_sensitivity == summary.trials
        && summary.unrelated_feature_zero_target_sensitivity == summary.trials
        && summary.imitation_agent_target_sensitive == summary.trials
        && summary.positive_agent_target_sensitive == summary.trials
        && summary.positive_agent_zero_collateral == summary.trials
        && summary.fixed_agent_did_not_influence_partner == summary.trials
        && summary.fixed_partner_did_not_respond == summary.trials
        && summary.reciprocal_positive_control_both_directions == summary.trials
        && summary.reciprocal_positive_control_zero_collateral == summary.trials
        && summary.agent_order_swap_verified == summary.trials * 6
        && summary.partner_order_swap_verified == summary.trials * 3;

    let payload = RmiTrackAReportPayloadV1 {
        manifest,
        manifest_sha256,
        summary,
        trials,
    };
    let payload_sha256 = digest_v1(&payload)?;
    Ok(RmiTrackAReportV1 {
        payload,
        payload_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_changes_exactly_one_registered_feature() {
        for target in targets_v1() {
            let pair = make_rmi_track_a_fixture_v1(target, 3, 42).unwrap();
            crate::rmi_benchmark::validate_rmi_intervention_pair(&pair).unwrap();
            assert_eq!(
                pair.control_observation
                    .features
                    .changed_features(&pair.treatment_observation.features),
                vec![target]
            );
        }
    }

    #[test]
    fn target_trial_seed_ordinals_are_unique_and_bounded() {
        use std::collections::BTreeSet;

        let mut seeds = BTreeSet::new();
        for target in targets_v1() {
            for trial_index in 0..32 {
                let fixture = make_rmi_track_a_fixture_v1(target, trial_index, 7).unwrap();
                assert!(seeds.insert(fixture.seed), "duplicate seed {}", fixture.seed);
            }
        }
        assert!(make_rmi_track_a_fixture_v1(
            RmiTargetFeatureV1::Motif,
            RMI_TRACK_A_MAX_TRIALS_PER_TARGET_V1,
            7
        )
        .is_err());
    }

    #[test]
    fn wrong_feature_agent_is_not_counted_as_target_sensitive() {
        for target in targets_v1() {
            let pair = make_rmi_track_a_fixture_v1(target, 3, 42).unwrap();
            let result = evaluate_rmi_pair(
                &contingent_agent_v1(unrelated_target_v1(target)),
                &pair,
            )
            .unwrap();
            assert!(!result.payload.targeted_input_sensitivity);
        }
    }

    #[test]
    fn yoked_replay_binds_a_distinct_donor_fixture() {
        let pair = make_rmi_track_a_fixture_v1(RmiTargetFeatureV1::Motif, 4, 42).unwrap();
        let (policy, donor_id, donor_digest) =
            yoked_replay_agent_v1(RmiTargetFeatureV1::Motif, 4, 42).unwrap();
        assert_ne!(donor_id, pair.trial_id);
        assert_ne!(donor_digest, digest_v1(&pair).unwrap());
        assert!(policy.replayed_response.action_id.contains(&donor_id));
        let result = evaluate_rmi_pair(&policy, &pair).unwrap();
        assert!(!result.payload.targeted_input_sensitivity);
    }

    #[test]
    fn report_is_deterministic_and_exposes_control_expectations() {
        let first = run_rmi_track_a_suite_v1(2, 42).unwrap();
        let second = run_rmi_track_a_suite_v1(2, 42).unwrap();
        assert_eq!(first, second);
        assert!(first.payload.summary.synthetic_fixture_expectations_satisfied);
        assert_eq!(first.payload.summary.trials, 8);
        assert_eq!(first.payload.summary.evaluator_errors, 0);
        assert_eq!(first.payload.summary.fixed_agent_zero_sensitivity, 8);
        assert_eq!(first.payload.summary.one_way_prompt_zero_sensitivity, 8);
        assert_eq!(first.payload.summary.yoked_replay_zero_sensitivity, 8);
        assert_eq!(first.payload.summary.unrelated_feature_zero_target_sensitivity, 8);
        assert_eq!(first.payload.summary.imitation_agent_target_sensitive, 8);
        assert_eq!(first.payload.summary.positive_agent_target_sensitive, 8);
        assert_eq!(first.payload.summary.positive_agent_zero_collateral, 8);
        assert_eq!(first.payload.summary.reciprocal_positive_control_both_directions, 8);
    }
}
