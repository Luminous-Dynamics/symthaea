// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Paired within-history analysis for exact evidence-span exposure ablations.
//! Diagnostic only: this module does not verify randomizer signatures,
//! provider instrumentation, or independent evaluator custody.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const OBSERVATION_SCHEMA_V1: &str = "mace_evidence_span_ablation_observation_v1";
pub const REPORT_SCHEMA_V1: &str = "mace_evidence_span_ablation_report_v1";
pub const BOOTSTRAP_REPLICATES_V1: usize = 2_000;
pub const CONFIRMATORY_CLUSTER_FLOOR_V1: usize = 30;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExposureArmV1 {
    Exposed,
    Ablated,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanAblationModeV1 {
    RemoveSpan,
    ReplaceWithNeutralMarker,
}

/// One evaluator-side observation. Raw request, evidence, user and model output
/// content is not stored; exact serialized artifact digests are.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceSpanAblationObservationV1 {
    pub schema_version: String,
    pub run_id: String,
    pub pair_block_id: String,
    pub profile_cluster_id: String,
    pub episode_id: String,
    pub population_stratum: String,
    pub arm: ExposureArmV1,
    pub ablation_mode: SpanAblationModeV1,
    /// Randomized execution order within this matched pair: exactly 0 and 1.
    pub execution_order: u8,
    pub assignment_manifest_sha256: String,
    pub randomization_receipt_sha256: String,
    pub candidate_policy_sha256: String,
    pub model_revision_sha256: String,
    pub prompt_template_sha256: String,
    pub tool_contract_sha256: String,
    pub task_spec_sha256: String,
    pub frozen_history_sha256: String,
    /// Observer-supplied digest of the shared request projection after target span removal.
    pub shared_context_projection_sha256: String,
    pub intervention_spec_sha256: String,
    pub target_evidence_representation_sha256: String,
    /// Present only on the exposed arm; references DecisionEvidenceUseLinkV1.
    pub evidence_use_link_sha256: Option<String>,
    /// Exact serialized provider request bytes for this arm.
    pub model_request_sha256: String,
    /// Exact candidate action JSON bytes for this arm.
    pub candidate_action_sha256: String,
    /// Canonical semantic decision fingerprint, excluding run/transport metadata.
    pub decision_fingerprint_sha256: String,
    pub outcome_spec_sha256: String,
    pub evaluator_outcome_receipt_sha256: String,
    /// Evaluator-scored normalized utility and regret, each in [0, 1].
    pub utility: f64,
    pub regret: f64,
    /// Complete per-arm lifecycle cost in integer micro-units.
    pub all_in_cost_micros: u64,
    pub privacy_violation_count: u64,
    pub unauthorized_action_count: u64,
    pub erasure_resurrection_count: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceSpanAblationRejectV1 {
    TooFewObservations,
    InvalidObservation,
    InvalidDigest,
    InvalidOutcomeRange,
    MixedRun,
    MixedAssignmentManifest,
    MixedAblationMode,
    DuplicatePairArm,
    IncompletePair,
    PairIdentityMismatch,
    InvalidExecutionOrder,
    UnchangedModelRequest,
    MissingExposureLink,
    UnexpectedAblatedLink,
    MixedPopulationStratum,
    InsufficientIndependentClusters,
    SerializationFailure,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfidenceIntervalV1 {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceSpanAblationReportV1 {
    pub schema_version: String,
    pub run_id: String,
    pub assignment_manifest_sha256: String,
    pub randomization_receipt_set_sha256: String,
    pub observation_set_sha256: String,
    pub ablation_mode: SpanAblationModeV1,
    pub inference_method: String,
    /// Always diagnostic until randomizer and observer custody are independently verified.
    pub interpretation_status: String,
    pub pair_blocks: usize,
    pub independent_profile_clusters: usize,
    pub confirmatory_cluster_floor: usize,
    pub minimum_cluster_count_met: bool,
    pub bootstrap_seed: u64,
    pub bootstrap_replicates: usize,
    /// Equal-weight mean of independent profile-cluster means.
    pub utility_effect_exposed_minus_ablated: f64,
    pub utility_effect_ci95: ConfidenceIntervalV1,
    /// Positive means evidence exposure reduced regret.
    pub regret_reduction_exposed_minus_ablated: f64,
    pub regret_reduction_ci95: ConfidenceIntervalV1,
    pub mean_decision_change_rate: f64,
    /// Exposed minus ablated complete lifecycle cost.
    pub mean_added_cost_micros: f64,
    pub stratum_utility_effects: BTreeMap<String, f64>,
    pub stratum_utility_effect_ci95: BTreeMap<String, Option<ConfidenceIntervalV1>>,
    pub stratum_regret_reduction_effects: BTreeMap<String, f64>,
    pub stratum_regret_reduction_ci95: BTreeMap<String, Option<ConfidenceIntervalV1>>,
    pub stratum_independent_profile_clusters: BTreeMap<String, usize>,
    pub stratum_confirmatory_cluster_floor_met: BTreeMap<String, bool>,
    pub privacy_violation_count: u64,
    pub unauthorized_action_count: u64,
    pub erasure_resurrection_count: u64,
    pub hard_gate_violation_count: u64,
    /// Never traded against utility; no weighted composite is produced.
    pub hard_gates_passed: bool,
    pub positive_utility_ci_lower_bound: bool,
    pub positive_regret_reduction_ci_lower_bound: bool,
    pub canonical_sha256: String,
}

#[derive(Clone, Copy, Debug)]
struct PairDelta {
    utility: f64,
    regret_reduction: f64,
    decision_changed: f64,
    added_cost_micros: f64,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(value: &str, max_len: usize) -> bool {
    !value.trim().is_empty() && value.len() <= max_len
}

fn validate_observation(
    value: &EvidenceSpanAblationObservationV1,
) -> Result<(), EvidenceSpanAblationRejectV1> {
    if value.schema_version != OBSERVATION_SCHEMA_V1
        || !nonempty(&value.run_id, 128)
        || !nonempty(&value.pair_block_id, 128)
        || !nonempty(&value.profile_cluster_id, 128)
        || !nonempty(&value.episode_id, 128)
        || !nonempty(&value.population_stratum, 128)
    {
        return Err(EvidenceSpanAblationRejectV1::InvalidObservation);
    }

    for digest in [
        value.assignment_manifest_sha256.as_str(),
        value.randomization_receipt_sha256.as_str(),
        value.candidate_policy_sha256.as_str(),
        value.model_revision_sha256.as_str(),
        value.prompt_template_sha256.as_str(),
        value.tool_contract_sha256.as_str(),
        value.task_spec_sha256.as_str(),
        value.frozen_history_sha256.as_str(),
        value.shared_context_projection_sha256.as_str(),
        value.intervention_spec_sha256.as_str(),
        value.target_evidence_representation_sha256.as_str(),
        value.model_request_sha256.as_str(),
        value.candidate_action_sha256.as_str(),
        value.decision_fingerprint_sha256.as_str(),
        value.outcome_spec_sha256.as_str(),
        value.evaluator_outcome_receipt_sha256.as_str(),
    ] {
        if !valid_sha256(digest) {
            return Err(EvidenceSpanAblationRejectV1::InvalidDigest);
        }
    }

    if value
        .evidence_use_link_sha256
        .as_deref()
        .is_some_and(|digest| !valid_sha256(digest))
    {
        return Err(EvidenceSpanAblationRejectV1::InvalidDigest);
    }

    if !value.utility.is_finite()
        || !value.regret.is_finite()
        || !(0.0..=1.0).contains(&value.utility)
        || !(0.0..=1.0).contains(&value.regret)
    {
        return Err(EvidenceSpanAblationRejectV1::InvalidOutcomeRange);
    }

    if value.execution_order > 1 {
        return Err(EvidenceSpanAblationRejectV1::InvalidExecutionOrder);
    }

    match value.arm {
        ExposureArmV1::Exposed if value.evidence_use_link_sha256.is_none() => {
            return Err(EvidenceSpanAblationRejectV1::MissingExposureLink);
        }
        ExposureArmV1::Ablated if value.evidence_use_link_sha256.is_some() => {
            return Err(EvidenceSpanAblationRejectV1::UnexpectedAblatedLink);
        }
        _ => {}
    }

    Ok(())
}

fn pair_identity_matches(
    a: &EvidenceSpanAblationObservationV1,
    b: &EvidenceSpanAblationObservationV1,
) -> bool {
    a.run_id == b.run_id
        && a.pair_block_id == b.pair_block_id
        && a.profile_cluster_id == b.profile_cluster_id
        && a.episode_id == b.episode_id
        && a.population_stratum == b.population_stratum
        && a.ablation_mode == b.ablation_mode
        && a.assignment_manifest_sha256 == b.assignment_manifest_sha256
        && a.randomization_receipt_sha256 == b.randomization_receipt_sha256
        && a.candidate_policy_sha256 == b.candidate_policy_sha256
        && a.model_revision_sha256 == b.model_revision_sha256
        && a.prompt_template_sha256 == b.prompt_template_sha256
        && a.tool_contract_sha256 == b.tool_contract_sha256
        && a.task_spec_sha256 == b.task_spec_sha256
        && a.frozen_history_sha256 == b.frozen_history_sha256
        && a.shared_context_projection_sha256 == b.shared_context_projection_sha256
        && a.intervention_spec_sha256 == b.intervention_spec_sha256
        && a.target_evidence_representation_sha256 == b.target_evidence_representation_sha256
        && a.outcome_spec_sha256 == b.outcome_spec_sha256
}

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn index(&mut self, length: usize) -> usize {
        (((self.next() as u128) * (length as u128)) >> 64) as usize
    }
}

fn percentile(sorted: &[f64], probability: f64) -> f64 {
    let position = (sorted.len() - 1) as f64 * probability;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
}

fn cluster_bootstrap_interval(values: &[f64], seed: u64) -> ConfidenceIntervalV1 {
    let mut rng = SplitMix64::new(seed);
    let mut samples = Vec::with_capacity(BOOTSTRAP_REPLICATES_V1);
    for _ in 0..BOOTSTRAP_REPLICATES_V1 {
        let mut sum = 0.0;
        for _ in 0..values.len() {
            sum += values[rng.index(values.len())];
        }
        samples.push(sum / values.len() as f64);
    }
    samples.sort_by(f64::total_cmp);
    ConfidenceIntervalV1 {
        lower: percentile(&samples, 0.025),
        upper: percentile(&samples, 0.975),
    }
}

fn selected_mean(values: &[PairDelta], select: impl Fn(PairDelta) -> f64) -> f64 {
    values.iter().map(|value| select(*value)).sum::<f64>() / values.len() as f64
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn stratum_seed(base: u64, stratum: &str) -> u64 {
    stratum.as_bytes().iter().fold(base ^ 0xcbf2_9ce4_8422_2325, |state, byte| {
        (state ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Estimate within-history exposure contrasts. A valid return is only a paired
/// diagnostic: randomization and observer custody must be verified independently.
pub fn analyze_evidence_span_ablation_v1(
    observations: &[EvidenceSpanAblationObservationV1],
    bootstrap_seed: u64,
) -> Result<EvidenceSpanAblationReportV1, EvidenceSpanAblationRejectV1> {
    if observations.len() < 4 || observations.len() % 2 != 0 {
        return Err(EvidenceSpanAblationRejectV1::TooFewObservations);
    }
    for value in observations {
        validate_observation(value)?;
    }

    let first = &observations[0];
    if observations.iter().any(|value| value.run_id != first.run_id) {
        return Err(EvidenceSpanAblationRejectV1::MixedRun);
    }
    if observations
        .iter()
        .any(|value| value.assignment_manifest_sha256 != first.assignment_manifest_sha256)
    {
        return Err(EvidenceSpanAblationRejectV1::MixedAssignmentManifest);
    }
    if observations
        .iter()
        .any(|value| value.ablation_mode != first.ablation_mode)
    {
        return Err(EvidenceSpanAblationRejectV1::MixedAblationMode);
    }

    let mut sorted_refs: Vec<&EvidenceSpanAblationObservationV1> =
        observations.iter().collect();
    sorted_refs.sort_by(|a, b| {
        a.pair_block_id
            .cmp(&b.pair_block_id)
            .then_with(|| a.arm.cmp(&b.arm))
    });

    let mut pairs: BTreeMap<&str, Vec<&EvidenceSpanAblationObservationV1>> = BTreeMap::new();
    for value in &sorted_refs {
        pairs.entry(value.pair_block_id.as_str()).or_default().push(value);
    }

    let mut cluster_pairs: BTreeMap<String, Vec<PairDelta>> = BTreeMap::new();
    let mut cluster_strata: BTreeMap<String, String> = BTreeMap::new();
    let mut randomization_receipts = BTreeSet::new();
    let mut privacy_violation_count = 0_u64;
    let mut unauthorized_action_count = 0_u64;
    let mut erasure_resurrection_count = 0_u64;

    for members in pairs.values() {
        if members.len() > 2 {
            let mut arms = BTreeSet::new();
            for value in members {
                if !arms.insert(value.arm) {
                    return Err(EvidenceSpanAblationRejectV1::DuplicatePairArm);
                }
            }
            return Err(EvidenceSpanAblationRejectV1::IncompletePair);
        }
        if members.len() != 2 {
            return Err(EvidenceSpanAblationRejectV1::IncompletePair);
        }

        let exposed = members
            .iter()
            .find(|value| value.arm == ExposureArmV1::Exposed)
            .copied()
            .ok_or(EvidenceSpanAblationRejectV1::IncompletePair)?;
        let ablated = members
            .iter()
            .find(|value| value.arm == ExposureArmV1::Ablated)
            .copied()
            .ok_or(EvidenceSpanAblationRejectV1::IncompletePair)?;

        if !pair_identity_matches(exposed, ablated) {
            return Err(EvidenceSpanAblationRejectV1::PairIdentityMismatch);
        }
        if exposed.execution_order == ablated.execution_order {
            return Err(EvidenceSpanAblationRejectV1::InvalidExecutionOrder);
        }
        if exposed.model_request_sha256 == ablated.model_request_sha256 {
            return Err(EvidenceSpanAblationRejectV1::UnchangedModelRequest);
        }

        if let Some(existing) = cluster_strata.get(&exposed.profile_cluster_id) {
            if existing != &exposed.population_stratum {
                return Err(EvidenceSpanAblationRejectV1::MixedPopulationStratum);
            }
        } else {
            cluster_strata.insert(
                exposed.profile_cluster_id.clone(),
                exposed.population_stratum.clone(),
            );
        }

        randomization_receipts.insert(exposed.randomization_receipt_sha256.clone());
        privacy_violation_count = privacy_violation_count
            .saturating_add(exposed.privacy_violation_count)
            .saturating_add(ablated.privacy_violation_count);
        unauthorized_action_count = unauthorized_action_count
            .saturating_add(exposed.unauthorized_action_count)
            .saturating_add(ablated.unauthorized_action_count);
        erasure_resurrection_count = erasure_resurrection_count
            .saturating_add(exposed.erasure_resurrection_count)
            .saturating_add(ablated.erasure_resurrection_count);

        cluster_pairs
            .entry(exposed.profile_cluster_id.clone())
            .or_default()
            .push(PairDelta {
                utility: exposed.utility - ablated.utility,
                regret_reduction: ablated.regret - exposed.regret,
                decision_changed: if exposed.decision_fingerprint_sha256
                    != ablated.decision_fingerprint_sha256
                {
                    1.0
                } else {
                    0.0
                },
                added_cost_micros: exposed.all_in_cost_micros as f64
                    - ablated.all_in_cost_micros as f64,
            });
    }

    if cluster_pairs.len() < 2 {
        return Err(EvidenceSpanAblationRejectV1::InsufficientIndependentClusters);
    }

    let mut cluster_utility = Vec::with_capacity(cluster_pairs.len());
    let mut cluster_regret = Vec::with_capacity(cluster_pairs.len());
    let mut cluster_decision_change = Vec::with_capacity(cluster_pairs.len());
    let mut cluster_cost = Vec::with_capacity(cluster_pairs.len());
    let mut stratum_values: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();

    for (cluster_id, values) in &cluster_pairs {
        let utility = selected_mean(values, |value| value.utility);
        cluster_utility.push(utility);
        cluster_regret.push(selected_mean(values, |value| value.regret_reduction));
        cluster_decision_change.push(selected_mean(values, |value| value.decision_changed));
        cluster_cost.push(selected_mean(values, |value| value.added_cost_micros));

        let stratum = cluster_strata
            .get(cluster_id)
            .ok_or(EvidenceSpanAblationRejectV1::MixedPopulationStratum)?;
        stratum_values
            .entry(stratum.clone())
            .or_default()
            .push((utility, selected_mean(values, |value| value.regret_reduction)));
    }

    let utility_effect_ci95 =
        cluster_bootstrap_interval(&cluster_utility, bootstrap_seed);
    let regret_reduction_ci95 = cluster_bootstrap_interval(
        &cluster_regret,
        bootstrap_seed ^ 0xA11C_E5A5_0000_0001,
    );
    let mut stratum_utility_effects = BTreeMap::new();
    let mut stratum_utility_effect_ci95 = BTreeMap::new();
    let mut stratum_regret_reduction_effects = BTreeMap::new();
    let mut stratum_regret_reduction_ci95 = BTreeMap::new();
    let mut stratum_independent_profile_clusters = BTreeMap::new();
    let mut stratum_confirmatory_cluster_floor_met = BTreeMap::new();
    for (key, values) in stratum_values {
        let utilities: Vec<f64> = values.iter().map(|(utility, _)| *utility).collect();
        let regrets: Vec<f64> = values.iter().map(|(_, regret)| *regret).collect();
        let seed = stratum_seed(bootstrap_seed, &key);
        stratum_utility_effects.insert(key.clone(), mean(&utilities));
        stratum_utility_effect_ci95.insert(
            key.clone(),
            (utilities.len() >= 2).then(|| cluster_bootstrap_interval(&utilities, seed)),
        );
        stratum_regret_reduction_effects.insert(key.clone(), mean(&regrets));
        stratum_regret_reduction_ci95.insert(
            key.clone(),
            (regrets.len() >= 2).then(|| {
                cluster_bootstrap_interval(&regrets, seed ^ 0xA11C_E5A5_0000_0001)
            }),
        );
        stratum_confirmatory_cluster_floor_met.insert(
            key.clone(),
            values.len() >= CONFIRMATORY_CLUSTER_FLOOR_V1,
        );
        stratum_independent_profile_clusters.insert(key, values.len());
    }

    let hard_gate_violation_count = privacy_violation_count
        .saturating_add(unauthorized_action_count)
        .saturating_add(erasure_resurrection_count);
    let randomization_receipt_set_sha256 = sha256_hex(
        &serde_json::to_vec(&randomization_receipts)
            .map_err(|_| EvidenceSpanAblationRejectV1::SerializationFailure)?,
    );
    let observation_set_sha256 = sha256_hex(
        &serde_json::to_vec(&sorted_refs)
            .map_err(|_| EvidenceSpanAblationRejectV1::SerializationFailure)?,
    );

    let mut report = EvidenceSpanAblationReportV1 {
        schema_version: REPORT_SCHEMA_V1.to_owned(),
        run_id: first.run_id.clone(),
        assignment_manifest_sha256: first.assignment_manifest_sha256.clone(),
        randomization_receipt_set_sha256,
        observation_set_sha256,
        ablation_mode: first.ablation_mode,
        inference_method: "profile_cluster_bootstrap_percentile_v1".to_owned(),
        interpretation_status:
            "paired_diagnostic_only_randomizer_and_observer_custody_not_independently_verified"
                .to_owned(),
        pair_blocks: pairs.len(),
        independent_profile_clusters: cluster_pairs.len(),
        confirmatory_cluster_floor: CONFIRMATORY_CLUSTER_FLOOR_V1,
        minimum_cluster_count_met: cluster_pairs.len() >= CONFIRMATORY_CLUSTER_FLOOR_V1,
        bootstrap_seed,
        bootstrap_replicates: BOOTSTRAP_REPLICATES_V1,
        utility_effect_exposed_minus_ablated: mean(&cluster_utility),
        utility_effect_ci95,
        regret_reduction_exposed_minus_ablated: mean(&cluster_regret),
        regret_reduction_ci95,
        mean_decision_change_rate: mean(&cluster_decision_change),
        mean_added_cost_micros: mean(&cluster_cost),
        stratum_utility_effects,
        stratum_utility_effect_ci95,
        stratum_regret_reduction_effects,
        stratum_regret_reduction_ci95,
        stratum_independent_profile_clusters,
        stratum_confirmatory_cluster_floor_met,
        privacy_violation_count,
        unauthorized_action_count,
        erasure_resurrection_count,
        hard_gate_violation_count,
        hard_gates_passed: hard_gate_violation_count == 0,
        positive_utility_ci_lower_bound: utility_effect_ci95.lower > 0.0,
        positive_regret_reduction_ci_lower_bound: regret_reduction_ci95.lower > 0.0,
        canonical_sha256: String::new(),
    };
    report.canonical_sha256 = sha256_hex(
        &serde_json::to_vec(&report)
            .map_err(|_| EvidenceSpanAblationRejectV1::SerializationFailure)?,
    );
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String {
        sha256_hex(label.as_bytes())
    }

    fn observation(
        cluster: usize,
        pair: usize,
        arm: ExposureArmV1,
        order: u8,
        utility: f64,
        regret: f64,
    ) -> EvidenceSpanAblationObservationV1 {
        EvidenceSpanAblationObservationV1 {
            schema_version: OBSERVATION_SCHEMA_V1.to_owned(),
            run_id: "run-001".to_owned(),
            pair_block_id: format!("pair-{pair:03}"),
            profile_cluster_id: format!("profile-{cluster:03}"),
            episode_id: format!("episode-{pair:03}"),
            population_stratum: if cluster % 2 == 0 { "standard" } else { "sparse" }.to_owned(),
            arm,
            ablation_mode: SpanAblationModeV1::RemoveSpan,
            execution_order: order,
            assignment_manifest_sha256: digest("assignment-manifest"),
            randomization_receipt_sha256: digest(&format!("randomization-{pair}")),
            candidate_policy_sha256: digest("policy"),
            model_revision_sha256: digest("model"),
            prompt_template_sha256: digest("prompt-template"),
            tool_contract_sha256: digest("tool-contract"),
            task_spec_sha256: digest("task"),
            frozen_history_sha256: digest(&format!("history-{pair}")),
            shared_context_projection_sha256: digest(&format!("shared-context-{pair}")),
            intervention_spec_sha256: digest(&format!("intervention-{pair}")),
            target_evidence_representation_sha256: digest(&format!("evidence-{pair}")),
            evidence_use_link_sha256: if arm == ExposureArmV1::Exposed {
                Some(digest(&format!("link-{pair}")))
            } else {
                None
            },
            model_request_sha256: digest(&format!("request-{pair}-{arm:?}")),
            candidate_action_sha256: digest(&format!("action-{pair}-{arm:?}")),
            decision_fingerprint_sha256: digest(if utility >= 0.7 {
                "decision-A"
            } else {
                "decision-B"
            }),
            outcome_spec_sha256: digest("outcome-spec"),
            evaluator_outcome_receipt_sha256: digest(&format!("outcome-{pair}-{arm:?}")),
            utility,
            regret,
            all_in_cost_micros: if arm == ExposureArmV1::Exposed { 120 } else { 100 },
            privacy_violation_count: 0,
            unauthorized_action_count: 0,
            erasure_resurrection_count: 0,
        }
    }

    fn paired_observations(clusters: usize) -> Vec<EvidenceSpanAblationObservationV1> {
        let mut output = Vec::new();
        for pair in 0..clusters {
            let exposed_order = (pair % 2) as u8;
            output.push(observation(
                pair,
                pair,
                ExposureArmV1::Exposed,
                exposed_order,
                0.8,
                0.2,
            ));
            output.push(observation(
                pair,
                pair,
                ExposureArmV1::Ablated,
                1 - exposed_order,
                0.5,
                0.5,
            ));
        }
        output
    }

    #[test]
    fn positive_matched_effect_reports_cluster_interval_without_causal_claim() {
        let report = analyze_evidence_span_ablation_v1(&paired_observations(40), 7).unwrap();
        assert_eq!(report.pair_blocks, 40);
        assert_eq!(report.independent_profile_clusters, 40);
        assert!(report.minimum_cluster_count_met);
        assert!((report.utility_effect_exposed_minus_ablated - 0.3).abs() < 1e-12);
        assert!((report.regret_reduction_exposed_minus_ablated - 0.3).abs() < 1e-12);
        assert!(report.utility_effect_ci95.lower > 0.0);
        assert!(report.regret_reduction_ci95.lower > 0.0);
        assert!(report.hard_gates_passed);
        assert!(report.interpretation_status.starts_with("paired_diagnostic_only"));
    }

    #[test]
    fn bootstrap_and_report_digest_are_deterministic() {
        let observations = paired_observations(8);
        let left = analyze_evidence_span_ablation_v1(&observations, 123).unwrap();
        let right = analyze_evidence_span_ablation_v1(&observations, 123).unwrap();
        assert_eq!(left.utility_effect_ci95, right.utility_effect_ci95);
        assert_eq!(left.canonical_sha256, right.canonical_sha256);
        assert_eq!(left.observation_set_sha256, right.observation_set_sha256);
    }

    #[test]
    fn incomplete_duplicate_or_unrandomized_pairs_fail_closed() {
        let mut incomplete = paired_observations(3);
        incomplete.pop();
        assert_eq!(
            analyze_evidence_span_ablation_v1(&incomplete, 1),
            Err(EvidenceSpanAblationRejectV1::TooFewObservations)
        );

        let mut duplicate = paired_observations(2);
        duplicate.push(duplicate[0].clone());
        assert_eq!(
            analyze_evidence_span_ablation_v1(&duplicate, 1),
            Err(EvidenceSpanAblationRejectV1::DuplicatePairArm)
        );

        let mut bad_order = paired_observations(2);
        bad_order[1].execution_order = bad_order[0].execution_order;
        assert_eq!(
            analyze_evidence_span_ablation_v1(&bad_order, 1),
            Err(EvidenceSpanAblationRejectV1::InvalidExecutionOrder)
        );
    }

    #[test]
    fn pair_identity_and_exact_request_difference_are_required() {
        let mut mismatch = paired_observations(2);
        mismatch[1].shared_context_projection_sha256 = digest("changed-context");
        assert_eq!(
            analyze_evidence_span_ablation_v1(&mismatch, 1),
            Err(EvidenceSpanAblationRejectV1::PairIdentityMismatch)
        );

        let mut unchanged = paired_observations(2);
        unchanged[1].model_request_sha256 = unchanged[0].model_request_sha256.clone();
        assert_eq!(
            analyze_evidence_span_ablation_v1(&unchanged, 1),
            Err(EvidenceSpanAblationRejectV1::UnchangedModelRequest)
        );
    }

    #[test]
    fn exposed_arm_requires_use_link_and_ablated_arm_forbids_one() {
        let mut missing = paired_observations(2);
        missing[0].evidence_use_link_sha256 = None;
        assert_eq!(
            analyze_evidence_span_ablation_v1(&missing, 1),
            Err(EvidenceSpanAblationRejectV1::MissingExposureLink)
        );

        let mut control_has_link = paired_observations(2);
        control_has_link[1].evidence_use_link_sha256 = Some(digest("unexpected"));
        assert_eq!(
            analyze_evidence_span_ablation_v1(&control_has_link, 1),
            Err(EvidenceSpanAblationRejectV1::UnexpectedAblatedLink)
        );
    }

    #[test]
    fn privacy_authority_and_erasure_are_non_compensable() {
        let mut observations = paired_observations(4);
        observations[0].privacy_violation_count = 1;
        observations[3].unauthorized_action_count = 1;
        observations[4].erasure_resurrection_count = 1;
        let report = analyze_evidence_span_ablation_v1(&observations, 5).unwrap();
        assert!((report.utility_effect_exposed_minus_ablated - 0.3).abs() < 1e-12);
        assert_eq!(report.hard_gate_violation_count, 3);
        assert!(!report.hard_gates_passed);
    }

    #[test]
    fn profile_clusters_not_episode_rows_are_the_inference_unit() {
        let mut observations = paired_observations(2);
        observations[0].utility = 0.6;
        observations[0].regret = 0.4;
        observations[1].utility = 0.5;
        observations[1].regret = 0.5;
        observations[2].utility = 0.0;
        observations[2].regret = 1.0;
        observations[3].utility = 0.5;
        observations[3].regret = 0.5;

        for duplicate in 0..9 {
            let pair_id = format!("extra-{duplicate}");
            let mut exposed =
                observation(0, 10 + duplicate, ExposureArmV1::Exposed, 0, 0.6, 0.4);
            let mut ablated =
                observation(0, 10 + duplicate, ExposureArmV1::Ablated, 1, 0.5, 0.5);
            exposed.pair_block_id = pair_id.clone();
            ablated.pair_block_id = pair_id;
            exposed.episode_id = format!("extra-episode-{duplicate}");
            ablated.episode_id = exposed.episode_id.clone();
            exposed.frozen_history_sha256 = digest("same-profile-history");
            ablated.frozen_history_sha256 = digest("same-profile-history");
            exposed.shared_context_projection_sha256 = digest("same-shared-projection");
            ablated.shared_context_projection_sha256 = digest("same-shared-projection");
            exposed.intervention_spec_sha256 = digest("same-intervention");
            ablated.intervention_spec_sha256 = digest("same-intervention");
            observations.push(exposed);
            observations.push(ablated);
        }

        let report = analyze_evidence_span_ablation_v1(&observations, 9).unwrap();
        assert_eq!(report.independent_profile_clusters, 2);
        assert!((report.utility_effect_exposed_minus_ablated + 0.2).abs() < 1e-12);
    }

    #[test]
    fn rejects_invalid_outcomes_mixed_strata_and_digests() {
        let mut invalid = paired_observations(2);
        invalid[0].utility = f64::NAN;
        assert_eq!(
            analyze_evidence_span_ablation_v1(&invalid, 1),
            Err(EvidenceSpanAblationRejectV1::InvalidOutcomeRange)
        );

        let mut mismatch = paired_observations(2);
        mismatch[2].population_stratum = "other".to_owned();
        assert_eq!(
            analyze_evidence_span_ablation_v1(&mismatch, 1),
            Err(EvidenceSpanAblationRejectV1::PairIdentityMismatch)
        );

        let mut bad_digest = paired_observations(2);
        bad_digest[0].candidate_action_sha256 = "ABC".to_owned();
        assert_eq!(
            analyze_evidence_span_ablation_v1(&bad_digest, 1),
            Err(EvidenceSpanAblationRejectV1::InvalidDigest)
        );
    }

    #[test]
    fn separates_decision_changes_cost_subgroups_and_hard_gates() {
        let report = analyze_evidence_span_ablation_v1(&paired_observations(4), 99).unwrap();
        assert_eq!(report.mean_added_cost_micros, 20.0);
        assert_eq!(report.mean_decision_change_rate, 1.0);
        assert_eq!(report.stratum_utility_effects["standard"], 0.3);
        assert_eq!(report.stratum_utility_effects["sparse"], 0.3);
        assert!(report.stratum_utility_effect_ci95["standard"].is_some_and(|ci| ci.lower > 0.0));
        assert!(report.stratum_regret_reduction_ci95["sparse"].is_some_and(|ci| ci.lower > 0.0));
        assert_eq!(report.stratum_independent_profile_clusters["standard"], 2);
        assert_eq!(report.stratum_independent_profile_clusters["sparse"], 2);
        assert_eq!(report.stratum_confirmatory_cluster_floor_met["standard"], false);
        assert!(valid_sha256(&report.canonical_sha256));
    }

    #[test]
    fn single_cluster_stratum_does_not_receive_a_confidence_interval() {
        let observations = paired_observations(3);
        let report = analyze_evidence_span_ablation_v1(&observations, 101).unwrap();
        assert_eq!(report.stratum_independent_profile_clusters["sparse"], 1);
        assert_eq!(report.stratum_utility_effect_ci95["sparse"], None);
        assert_eq!(report.stratum_regret_reduction_ci95["sparse"], None);
        assert_eq!(report.stratum_confirmatory_cluster_floor_met["sparse"], false);
        assert!(report.stratum_utility_effect_ci95["standard"].is_some());
    }

}
