//! Structural and semantic audit for query-critical omitted-hypothesis discovery.
//!
//! This tool validates cross-record integrity and reports observed-sample counts.
//! It deliberately does not estimate population recall, fit capture-recapture models,
//! authenticate evidence producers, or establish biological truth.

use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(
    name = "connectome-query-critical-discovery-audit",
    version,
    about = "Validate query-critical omitted-hypothesis discovery audit bundles"
)]
struct Args {
    /// Versioned query-critical discovery input bundle.
    #[arg(long)]
    input: PathBuf,
    /// Optional deterministic JSON report path; stdout when omitted.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema_version: u32,
    analysis_id: String,
    dataset_revision_commit_sha: String,
    query: Query,
    baseline: Baseline,
    sampling_binding: SamplingBinding,
    channels: Vec<Channel>,
    psus: Vec<Psu>,
    proposals: Vec<Proposal>,
    alternatives: Vec<Alternative>,
    adjudication_protocol_sha256: String,
    #[serde(default)]
    all_list_missed_sensitivity: Option<AllListMissedSensitivity>,
    claim_scope: String,
    #[serde(default)]
    notes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Query {
    query_id: String,
    query_spec_sha256: String,
    decision_rule_sha256: String,
    baseline_decision_met: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Baseline {
    scenario_set_sha256: String,
    joint_support_sha256: String,
    action_set_sha256: String,
    adaptive_policy_report_sha256: String,
    adaptive_policy_status: String,
    adaptive_worst_case_effort_units: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SamplingBinding {
    sampling_frame_sha256: String,
    sampling_protocol_sha256: String,
    sample_audit_report_sha256: String,
    psu_population_count: usize,
    sampled_psu_count: usize,
    design_owner: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Channel {
    channel_id: String,
    channel_kind: String,
    implementation_sha256: String,
    protocol_sha256: String,
    ancestry_group_ids: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Psu {
    psu_id: String,
    stratum_id: String,
    search_disposition: String,
    proposal_ids: Vec<String>,
    independent_search_receipt_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Proposal {
    proposal_id: String,
    channel_id: String,
    psu_id: String,
    candidate_world_sha256: String,
    source_evidence_sha256: String,
    adjudication_disposition: String,
    proposal_adjudication_sha256: String,
    #[serde(default)]
    matched_alternative_id: Option<String>,
    #[serde(default)]
    match_adjudication_sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Alternative {
    alternative_id: String,
    psu_id: String,
    graph_sha256: String,
    source_evidence_sha256: String,
    biological_disposition: String,
    query_consequence: String,
    challenge_report_sha256: String,
    challenge_report_status: String,
    before_worst_case_effort_units: Option<u64>,
    after_worst_case_effort_units: Option<u64>,
    impact_weight: f64,
    alternative_adjudication_sha256: String,
    decision_before_met: Option<bool>,
    decision_after_met: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AllListMissedSensitivity {
    owner: String,
    report_sha256: String,
    status: String,
}

#[derive(Debug, Clone, Serialize)]
struct ChannelSummary {
    channel_id: String,
    proposal_count: usize,
    confirmed_match_count: usize,
    unique_alternative_ids: Vec<String>,
    query_critical_alternative_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct SharedAncestryOverlap {
    channel_a: String,
    channel_b: String,
    shared_group_ids: Vec<String>,
}

const CRITICAL_CONSEQUENCES: [&str; 3] = [
    "QUERY_DECISION_FLIP",
    "ADAPTIVE_POLICY_INVALIDATED",
    "ADAPTIVE_COST_INCREASED",
];

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn require_sha(value: &str, length: usize, label: &str) -> Result<(), String> {
    if valid_hex(value, length) {
        Ok(())
    } else {
        Err(format!("{label} must be {length} lowercase hexadecimal characters"))
    }
}

fn validate_unique<I, S>(values: I, label: &str) -> Result<(), String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut seen = BTreeSet::new();
    for value in values {
        let value = value.into();
        if !nonempty(&value) || !seen.insert(value.clone()) {
            return Err(format!("{label} contains an empty or duplicate identity: {value}"));
        }
    }
    Ok(())
}

fn validate_alternative(alternative: &Alternative, input: &Input) -> Result<(), String> {
    if !nonempty(&alternative.alternative_id) || !nonempty(&alternative.psu_id) {
        return Err("alternative identity and PSU ID must be non-empty".into());
    }
    require_sha(&alternative.graph_sha256, 64, "alternative graph digest")?;
    require_sha(
        &alternative.source_evidence_sha256,
        64,
        "alternative source evidence digest",
    )?;
    require_sha(
        &alternative.challenge_report_sha256,
        64,
        "alternative challenge report digest",
    )?;
    require_sha(
        &alternative.alternative_adjudication_sha256,
        64,
        "alternative adjudication digest",
    )?;
    if !alternative.impact_weight.is_finite() || alternative.impact_weight <= 0.0 {
        return Err(format!(
            "alternative {} has non-finite or non-positive impact weight",
            alternative.alternative_id
        ));
    }

    let baseline_cost = input.baseline.adaptive_worst_case_effort_units;
    if alternative.before_worst_case_effort_units != baseline_cost {
        return Err(format!(
            "alternative {} before-cost does not match the frozen baseline",
            alternative.alternative_id
        ));
    }
    if alternative.decision_before_met != Some(input.query.baseline_decision_met) {
        return Err(format!(
            "alternative {} before-decision does not match the frozen query baseline",
            alternative.alternative_id
        ));
    }

    match alternative.challenge_report_status.as_str() {
        "POLICY_EXISTS" if alternative.after_worst_case_effort_units.is_none() => {
            return Err(format!(
                "alternative {} reports POLICY_EXISTS without an after-cost",
                alternative.alternative_id
            ));
        }
        "NO_POLICY" if alternative.after_worst_case_effort_units.is_some() => {
            return Err(format!(
                "alternative {} reports NO_POLICY with a non-null after-cost",
                alternative.alternative_id
            ));
        }
        "NOT_RUN" | "NOT_ESTIMABLE" => {
            if alternative.query_consequence != "NOT_ESTIMABLE" {
                return Err(format!(
                    "alternative {} has an unevaluated report but a determinate query consequence",
                    alternative.alternative_id
                ));
            }
        }
        "POLICY_EXISTS" | "NO_POLICY" => {}
        other => {
            return Err(format!(
                "alternative {} has unknown challenge report status {other}",
                alternative.alternative_id
            ));
        }
    }

    match alternative.query_consequence.as_str() {
        "QUERY_DECISION_FLIP" => {
            let before = alternative.decision_before_met.ok_or_else(|| {
                format!("alternative {} lacks its before-decision", alternative.alternative_id)
            })?;
            let after = alternative.decision_after_met.ok_or_else(|| {
                format!("alternative {} lacks its after-decision", alternative.alternative_id)
            })?;
            if before == after {
                return Err(format!(
                    "alternative {} is labelled QUERY_DECISION_FLIP but decisions are equal",
                    alternative.alternative_id
                ));
            }
        }
        "ADAPTIVE_POLICY_INVALIDATED" => {
            if input.baseline.adaptive_policy_status != "POLICY_EXISTS"
                || baseline_cost.is_none()
                || alternative.challenge_report_status != "NO_POLICY"
                || alternative.after_worst_case_effort_units.is_some()
            {
                return Err(format!(
                    "alternative {} has an inconsistent policy-invalidation disposition",
                    alternative.alternative_id
                ));
            }
        }
        "ADAPTIVE_COST_INCREASED" => {
            let before = alternative.before_worst_case_effort_units.ok_or_else(|| {
                format!("alternative {} has no baseline policy cost", alternative.alternative_id)
            })?;
            let after = alternative.after_worst_case_effort_units.ok_or_else(|| {
                format!("alternative {} has no expanded policy cost", alternative.alternative_id)
            })?;
            if alternative.challenge_report_status != "POLICY_EXISTS" || after <= before {
                return Err(format!(
                    "alternative {} is labelled ADAPTIVE_COST_INCREASED without a strict cost increase",
                    alternative.alternative_id
                ));
            }
        }
        "NO_QUERY_EFFECT" => {
            if alternative.decision_before_met != alternative.decision_after_met
                || alternative.before_worst_case_effort_units
                    != alternative.after_worst_case_effort_units
            {
                return Err(format!(
                    "alternative {} is labelled NO_QUERY_EFFECT but decision or policy cost changed",
                    alternative.alternative_id
                ));
            }
        }
        "NOT_ESTIMABLE" => {
            if alternative.challenge_report_status != "NOT_ESTIMABLE"
                && alternative.challenge_report_status != "NOT_RUN"
            {
                return Err(format!(
                    "alternative {} is NOT_ESTIMABLE but the challenge report is determinate",
                    alternative.alternative_id
                ));
            }
        }
        other => {
            return Err(format!(
                "alternative {} has unknown query consequence {other}",
                alternative.alternative_id
            ));
        }
    }

    match alternative.biological_disposition.as_str() {
        "ADMISSIBLE" | "INADMISSIBLE" | "INDETERMINATE" | "DISPUTED"
        | "NOT_ADJUDICATED" => Ok(()),
        other => Err(format!(
            "alternative {} has unknown biological disposition {other}",
            alternative.alternative_id
        )),
    }
}

fn validate_input(input: &Input, input_bytes: &[u8]) -> Result<Value, String> {
    if input.schema_version != 1 {
        return Err("schema_version must equal 1".into());
    }
    if !nonempty(&input.analysis_id) || !nonempty(&input.query.query_id) {
        return Err("analysis_id and query_id must be non-empty".into());
    }
    if !matches!(
        input.claim_scope.as_str(),
        "synthetic_known_truth" | "sampled_declared_population"
    ) {
        return Err("claim_scope is unsupported".into());
    }
    require_sha(
        &input.dataset_revision_commit_sha,
        40,
        "dataset revision commit SHA",
    )?;
    require_sha(&input.query.query_spec_sha256, 64, "query specification digest")?;
    require_sha(&input.query.decision_rule_sha256, 64, "query decision-rule digest")?;
    require_sha(
        &input.baseline.scenario_set_sha256,
        64,
        "baseline scenario-set digest",
    )?;
    require_sha(
        &input.baseline.joint_support_sha256,
        64,
        "baseline joint-support digest",
    )?;
    require_sha(
        &input.baseline.action_set_sha256,
        64,
        "baseline action-set digest",
    )?;
    require_sha(
        &input.baseline.adaptive_policy_report_sha256,
        64,
        "baseline adaptive-policy report digest",
    )?;
    require_sha(
        &input.sampling_binding.sampling_frame_sha256,
        64,
        "sampling-frame digest",
    )?;
    require_sha(
        &input.sampling_binding.sampling_protocol_sha256,
        64,
        "sampling-protocol digest",
    )?;
    require_sha(
        &input.sampling_binding.sample_audit_report_sha256,
        64,
        "sample audit report digest",
    )?;
    require_sha(
        &input.adjudication_protocol_sha256,
        64,
        "adjudication-protocol digest",
    )?;
    if input.sampling_binding.design_owner != "9204" {
        return Err("sampling design owner must remain #9204".into());
    }
    if input.sampling_binding.sampled_psu_count != input.psus.len()
        || input.sampling_binding.psu_population_count < input.sampling_binding.sampled_psu_count
        || input.psus.is_empty()
    {
        return Err("sampling population/sample counts do not reconcile with the PSU records".into());
    }
    match input.baseline.adaptive_policy_status.as_str() {
        "POLICY_EXISTS" if input.baseline.adaptive_worst_case_effort_units.is_some() => {}
        "NO_POLICY" if input.baseline.adaptive_worst_case_effort_units.is_none() => {}
        _ => return Err("baseline adaptive-policy status and effort are inconsistent".into()),
    }

    validate_unique(input.channels.iter().map(|row| row.channel_id.clone()), "channel IDs")?;
    validate_unique(input.psus.iter().map(|row| row.psu_id.clone()), "PSU IDs")?;
    validate_unique(
        input.proposals.iter().map(|row| row.proposal_id.clone()),
        "proposal IDs",
    )?;
    validate_unique(
        input.alternatives.iter().map(|row| row.alternative_id.clone()),
        "alternative IDs",
    )?;

    let mut channels = BTreeMap::new();
    for channel in &input.channels {
        if !matches!(
            channel.channel_kind.as_str(),
            "production_candidate_generator"
                | "em_specialist"
                | "morphology_mesh"
                | "skeleton_continuity"
                | "synapse_detection"
                | "blinded_expert_team"
                | "orthogonal_acquisition"
                | "other"
        ) {
            return Err(format!(
                "channel {} has unsupported kind {}",
                channel.channel_id, channel.channel_kind
            ));
        }
        require_sha(
            &channel.implementation_sha256,
            64,
            "channel implementation digest",
        )?;
        require_sha(&channel.protocol_sha256, 64, "channel protocol digest")?;
        validate_unique(
            channel.ancestry_group_ids.iter().cloned(),
            "channel ancestry groups",
        )?;
        channels.insert(channel.channel_id.as_str(), channel);
    }
    if channels.is_empty() {
        return Err("at least one discovery channel is required".into());
    }

    let mut psus = BTreeMap::new();
    let mut zero_proposal_psu_count = 0usize;
    let mut complete_search_psu_count = 0usize;
    let mut incomplete_search_psu_count = 0usize;
    let mut inaccessible_psu_count = 0usize;
    for psu in &input.psus {
        if !nonempty(&psu.stratum_id) {
            return Err(format!("PSU {} has an empty stratum ID", psu.psu_id));
        }
        require_sha(
            &psu.independent_search_receipt_sha256,
            64,
            "independent search receipt digest",
        )?;
        validate_unique(psu.proposal_ids.iter().cloned(), "PSU proposal IDs")?;
        match psu.search_disposition.as_str() {
            "COMPLETE_SEARCH" => complete_search_psu_count += 1,
            "INCOMPLETE_SEARCH" => incomplete_search_psu_count += 1,
            "SOURCE_INACCESSIBLE" => inaccessible_psu_count += 1,
            other => return Err(format!("PSU {} has unknown search disposition {other}", psu.psu_id)),
        }
        if psu.proposal_ids.is_empty() {
            zero_proposal_psu_count += 1;
        }
        psus.insert(psu.psu_id.as_str(), psu);
    }

    let mut alternatives = BTreeMap::new();
    for alternative in &input.alternatives {
        validate_alternative(alternative, input)?;
        if !psus.contains_key(alternative.psu_id.as_str()) {
            return Err(format!(
                "alternative {} references unknown PSU {}",
                alternative.alternative_id, alternative.psu_id
            ));
        }
        alternatives.insert(alternative.alternative_id.as_str(), alternative);
    }

    let mut observed_proposals_by_psu: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    let mut channel_proposal_counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut confirmed_matches_by_channel: BTreeMap<&str, usize> = BTreeMap::new();
    let mut captured_alternatives_by_channel: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    let mut captured_alternatives_union = BTreeSet::new();

    for proposal in &input.proposals {
        require_sha(
            &proposal.candidate_world_sha256,
            64,
            "candidate world digest",
        )?;
        require_sha(
            &proposal.source_evidence_sha256,
            64,
            "proposal source evidence digest",
        )?;
        require_sha(
            &proposal.proposal_adjudication_sha256,
            64,
            "proposal adjudication receipt digest",
        )?;
        if !channels.contains_key(proposal.channel_id.as_str()) {
            return Err(format!(
                "proposal {} references unknown channel {}",
                proposal.proposal_id, proposal.channel_id
            ));
        }
        if !psus.contains_key(proposal.psu_id.as_str()) {
            return Err(format!(
                "proposal {} references unknown PSU {}",
                proposal.proposal_id, proposal.psu_id
            ));
        }
        *channel_proposal_counts
            .entry(proposal.channel_id.as_str())
            .or_default() += 1;
        observed_proposals_by_psu
            .entry(proposal.psu_id.as_str())
            .or_default()
            .insert(proposal.proposal_id.as_str());

        match proposal.adjudication_disposition.as_str() {
            "CONFIRMED_MATCH" => {
                let alternative_id = proposal
                    .matched_alternative_id
                    .as_deref()
                    .ok_or_else(|| format!("confirmed proposal {} lacks alternative ID", proposal.proposal_id))?;
                let match_receipt = proposal
                    .match_adjudication_sha256
                    .as_deref()
                    .ok_or_else(|| format!("confirmed proposal {} lacks blinded match receipt", proposal.proposal_id))?;
                require_sha(match_receipt, 64, "proposal match-adjudication digest")?;
                let alternative = alternatives.get(alternative_id).ok_or_else(|| {
                    format!(
                        "proposal {} references unknown alternative {}",
                        proposal.proposal_id, alternative_id
                    )
                })?;
                if alternative.psu_id != proposal.psu_id {
                    return Err(format!(
                        "proposal {} and alternative {} belong to different PSUs",
                        proposal.proposal_id, alternative_id
                    ));
                }
                *confirmed_matches_by_channel
                    .entry(proposal.channel_id.as_str())
                    .or_default() += 1;
                captured_alternatives_by_channel
                    .entry(proposal.channel_id.as_str())
                    .or_default()
                    .insert(alternative_id);
                captured_alternatives_union.insert(alternative_id);
            }
            "FALSE_POSITIVE" | "AMBIGUOUS" | "UNADJUDICATED" => {
                if proposal.matched_alternative_id.is_some()
                    || proposal.match_adjudication_sha256.is_some()
                {
                    return Err(format!(
                        "non-confirmed proposal {} must not carry a confirmed alternative match",
                        proposal.proposal_id
                    ));
                }
            }
            other => {
                return Err(format!(
                    "proposal {} has unknown adjudication disposition {other}",
                    proposal.proposal_id
                ));
            }
        }
    }

    for (psu_id, psu) in &psus {
        let declared = psu.proposal_ids.iter().map(String::as_str).collect::<BTreeSet<_>>();
        let observed = observed_proposals_by_psu
            .get(psu_id)
            .cloned()
            .unwrap_or_default();
        if declared != observed {
            return Err(format!(
                "PSU {} proposal ledger does not exactly match proposal records",
                psu_id
            ));
        }
    }

    let mut query_critical_ids = BTreeSet::new();
    let mut query_critical_captured_ids = BTreeSet::new();
    let mut non_query_critical_admissible_count = 0usize;
    let mut admissible_alternative_count = 0usize;
    for alternative in &input.alternatives {
        if alternative.biological_disposition != "ADMISSIBLE" {
            continue;
        }
        admissible_alternative_count += 1;
        if CRITICAL_CONSEQUENCES.contains(&alternative.query_consequence.as_str()) {
            query_critical_ids.insert(alternative.alternative_id.as_str());
            if captured_alternatives_union.contains(alternative.alternative_id.as_str()) {
                query_critical_captured_ids.insert(alternative.alternative_id.as_str());
            }
        } else {
            non_query_critical_admissible_count += 1;
        }
    }
    let query_critical_missed_ids = query_critical_ids
        .difference(&query_critical_captured_ids)
        .copied()
        .collect::<Vec<_>>();

    let mut channel_summaries = Vec::new();
    for channel_id in channels.keys() {
        let alternatives_for_channel = captured_alternatives_by_channel
            .get(channel_id)
            .cloned()
            .unwrap_or_default();
        let channel_critical = alternatives_for_channel
            .iter()
            .filter(|alternative_id| query_critical_ids.contains(**alternative_id))
            .map(|value| (*value).to_string())
            .collect::<Vec<_>>();
        channel_summaries.push(ChannelSummary {
            channel_id: (*channel_id).to_string(),
            proposal_count: *channel_proposal_counts.get(channel_id).unwrap_or(&0),
            confirmed_match_count: *confirmed_matches_by_channel.get(channel_id).unwrap_or(&0),
            unique_alternative_ids: alternatives_for_channel
                .iter()
                .map(|value| (*value).to_string())
                .collect(),
            query_critical_alternative_ids: channel_critical,
        });
    }

    let ordered_channels = channels.keys().copied().collect::<Vec<_>>();
    let mut shared_ancestry_overlaps = Vec::new();
    for left_index in 0..ordered_channels.len() {
        for right_index in (left_index + 1)..ordered_channels.len() {
            let left = channels[ordered_channels[left_index]];
            let right = channels[ordered_channels[right_index]];
            let left_groups = left.ancestry_group_ids.iter().map(String::as_str).collect::<BTreeSet<_>>();
            let right_groups = right.ancestry_group_ids.iter().map(String::as_str).collect::<BTreeSet<_>>();
            let shared = left_groups
                .intersection(&right_groups)
                .map(|group| (*group).to_string())
                .collect::<Vec<_>>();
            if !shared.is_empty() {
                shared_ancestry_overlaps.push(SharedAncestryOverlap {
                    channel_a: left.channel_id.clone(),
                    channel_b: right.channel_id.clone(),
                    shared_group_ids: shared,
                });
            }
        }
    }

    let all_list_missed_sensitivity_status = if let Some(sensitivity) = &input.all_list_missed_sensitivity {
        if sensitivity.owner != "9248" {
            return Err("all-list-missed sensitivity owner must remain #9248".into());
        }
        require_sha(&sensitivity.report_sha256, 64, "all-list-missed report digest")?;
        if !matches!(
            sensitivity.status.as_str(),
            "NOT_RUN" | "ESTIMABLE_SENSITIVITY_RANGE" | "NOT_IDENTIFIABLE" | "NOT_ESTIMABLE"
        ) {
            return Err("unknown all-list-missed sensitivity status".into());
        }
        sensitivity.status.clone()
    } else {
        "NOT_SUPPLIED".into()
    };

    let status = if incomplete_search_psu_count > 0 || inaccessible_psu_count > 0 {
        "NOT_PROMOTABLE_INCOMPLETE_OR_INACCESSIBLE_SEARCH"
    } else if input.claim_scope == "synthetic_known_truth" {
        "SYNTHETIC_OBSERVED_SAMPLE_VALIDATED"
    } else {
        "SAMPLED_DECLARED_POPULATION_STRUCTURE_VALIDATED"
    };

    let input_sha256 = sha256(input_bytes);
    let report = json!({
        "schema_version": 1,
        "tool": "connectome-query-critical-discovery-audit",
        "analysis_id": input.analysis_id,
        "input_sha256": input_sha256,
        "dataset_revision_commit_sha": input.dataset_revision_commit_sha,
        "query_id": input.query.query_id,
        "query_spec_sha256": input.query.query_spec_sha256,
        "baseline_scenario_set_sha256": input.baseline.scenario_set_sha256,
        "baseline_joint_support_sha256": input.baseline.joint_support_sha256,
        "baseline_action_set_sha256": input.baseline.action_set_sha256,
        "baseline_adaptive_policy_report_sha256": input.baseline.adaptive_policy_report_sha256,
        "sampling_frame_sha256": input.sampling_binding.sampling_frame_sha256,
        "sampling_protocol_sha256": input.sampling_binding.sampling_protocol_sha256,
        "sample_audit_report_sha256": input.sampling_binding.sample_audit_report_sha256,
        "psu_population_count": input.sampling_binding.psu_population_count,
        "sampled_psu_count": input.sampling_binding.sampled_psu_count,
        "complete_search_psu_count": complete_search_psu_count,
        "incomplete_search_psu_count": incomplete_search_psu_count,
        "source_inaccessible_psu_count": inaccessible_psu_count,
        "zero_proposal_psu_count": zero_proposal_psu_count,
        "channel_count": channels.len(),
        "proposal_count": input.proposals.len(),
        "confirmed_match_count": captured_alternatives_by_channel.values().map(BTreeSet::len).sum::<usize>(),
        "alternative_count": input.alternatives.len(),
        "admissible_alternative_count": admissible_alternative_count,
        "query_critical_alternative_count": query_critical_ids.len(),
        "captured_query_critical_alternative_count": query_critical_captured_ids.len(),
        "missed_query_critical_alternative_ids": query_critical_missed_ids,
        "query_critical_alternative_ids": query_critical_ids.iter().map(|value| (*value).to_string()).collect::<Vec<_>>(),
        "non_query_critical_admissible_alternative_count": non_query_critical_admissible_count,
        "channel_summaries": channel_summaries,
        "shared_ancestry_overlaps": shared_ancestry_overlaps,
        "all_list_missed_sensitivity_owner": "9248",
        "all_list_missed_sensitivity_status": all_list_missed_sensitivity_status,
        "population_estimate_status": "NOT_COMPUTED_BY_THIS_TOOL_OWNER_9204",
        "independence_disposition": "NOT_ESTABLISHED_BY_ANCESTRY_LABELS",
        "biological_truth_disposition": "NOT_ESTABLISHED_BY_THIS_TOOL",
        "status": status,
        "scope_note": "This report validates cross-record structure and observed-sample counts only. It does not compute design-weighted population recall, confidence intervals, all-list-missed estimates, biological truth, source authenticity, channel independence, or global hypothesis-space completeness."
    });
    Ok(report)
}

fn run(input_bytes: &[u8]) -> Result<Value, String> {
    let input: Input = serde_json::from_slice(input_bytes)
        .map_err(|error| format!("input JSON/schema decoding failed: {error}"))?;
    validate_input(&input, input_bytes)
}

fn main() -> ExitCode {
    let args = Args::parse();
    let input_bytes = match fs::read(&args.input) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("failed to read {}: {error}", args.input.display());
            return ExitCode::FAILURE;
        }
    };
    let report = match run(&input_bytes) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("query-critical discovery audit rejected input: {error}");
            return ExitCode::FAILURE;
        }
    };
    let rendered = match serde_json::to_string_pretty(&report) {
        Ok(value) => value + "\n",
        Err(error) => {
            eprintln!("failed to serialize report: {error}");
            return ExitCode::FAILURE;
        }
    };
    if let Some(path) = args.output {
        if let Err(error) = fs::write(&path, rendered) {
            eprintln!("failed to write {}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    } else {
        print!("{rendered}");
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_value() -> Value {
        serde_json::from_str(include_str!(
            "../../../../docs/neuroscience/examples/CONNECTOME_QUERY_CRITICAL_DISCOVERY.example.json"
        ))
        .unwrap()
    }

    fn run_value(value: Value) -> Result<Value, String> {
        let bytes = serde_json::to_vec(&value).unwrap();
        run(&bytes)
    }

    #[test]
    fn synthetic_known_answer_fixture_is_validated() {
        let report = run_value(fixture_value()).unwrap();
        assert_eq!(report["psu_population_count"], 100);
        assert_eq!(report["sampled_psu_count"], 4);
        assert_eq!(report["zero_proposal_psu_count"], 2);
        assert_eq!(report["channel_count"], 3);
        assert_eq!(report["proposal_count"], 3);
        assert_eq!(report["alternative_count"], 3);
        assert_eq!(report["admissible_alternative_count"], 3);
        assert_eq!(report["query_critical_alternative_count"], 2);
        assert_eq!(report["captured_query_critical_alternative_count"], 1);
        assert_eq!(report["missed_query_critical_alternative_ids"][0], "ALT-OMITTED-HIGH-IMPACT");
        assert_eq!(report["non_query_critical_admissible_alternative_count"], 1);
        assert_eq!(report["population_estimate_status"], "NOT_COMPUTED_BY_THIS_TOOL_OWNER_9204");
        assert_eq!(report["independence_disposition"], "NOT_ESTABLISHED_BY_ANCESTRY_LABELS");
    }

    #[test]
    fn duplicate_channel_identity_is_rejected() {
        let mut value = fixture_value();
        value["channels"][1]["channel_id"] = value["channels"][0]["channel_id"].clone();
        assert!(run_value(value).unwrap_err().contains("duplicate identity"));
    }

    #[test]
    fn duplicate_alternative_identity_is_rejected() {
        let mut value = fixture_value();
        value["alternatives"][1]["alternative_id"] = value["alternatives"][0]["alternative_id"].clone();
        assert!(run_value(value).unwrap_err().contains("duplicate identity"));
    }

    #[test]
    fn proposal_ledger_must_match_psu_records() {
        let mut value = fixture_value();
        value["psus"][0]["proposal_ids"].as_array_mut().unwrap().pop();
        assert!(run_value(value).unwrap_err().contains("proposal ledger"));
    }

    #[test]
    fn dangling_channel_reference_is_rejected() {
        let mut value = fixture_value();
        value["proposals"][0]["channel_id"] = json!("missing-channel");
        assert!(run_value(value).unwrap_err().contains("unknown channel"));
    }

    #[test]
    fn cross_psu_match_is_rejected() {
        let mut value = fixture_value();
        value["proposals"][0]["psu_id"] = json!("PSU-002");
        assert!(run_value(value).unwrap_err().contains("different PSUs"));
    }

    #[test]
    fn confirmed_match_requires_blinded_receipt() {
        let mut value = fixture_value();
        value["proposals"][0].as_object_mut().unwrap().remove("match_adjudication_sha256");
        assert!(run_value(value).unwrap_err().contains("lacks blinded match receipt"));
    }

    #[test]
    fn query_flip_requires_different_decisions() {
        let mut value = fixture_value();
        value["alternatives"][0]["decision_after_met"] = json!(true);
        assert!(run_value(value).unwrap_err().contains("decisions are equal"));
    }

    #[test]
    fn no_query_effect_requires_unchanged_decision_and_cost() {
        let mut value = fixture_value();
        value["alternatives"][1]["after_worst_case_effort_units"] = json!(5);
        assert!(run_value(value).unwrap_err().contains("NO_QUERY_EFFECT"));
    }

    #[test]
    fn invalidated_policy_must_have_no_after_cost() {
        let mut value = fixture_value();
        value["alternatives"][2]["after_worst_case_effort_units"] = json!(5);
        assert!(run_value(value).unwrap_err().contains("NO_POLICY"));
    }

    #[test]
    fn incomplete_or_inaccessible_psu_blocks_promotion() {
        let mut value = fixture_value();
        value["psus"][3]["search_disposition"] = json!("SOURCE_INACCESSIBLE");
        let report = run_value(value).unwrap();
        assert_eq!(report["source_inaccessible_psu_count"], 1);
        assert_eq!(report["status"], "NOT_PROMOTABLE_INCOMPLETE_OR_INACCESSIBLE_SEARCH");
    }

    #[test]
    fn repeated_alternative_capture_is_counted_once_for_union() {
        let report = run_value(fixture_value()).unwrap();
        assert_eq!(report["captured_query_critical_alternative_count"], 1);
        assert_eq!(report["channel_summaries"][0]["query_critical_alternative_ids"][0], "ALT-QUERY-FLIP");
        assert_eq!(report["channel_summaries"][1]["query_critical_alternative_ids"][0], "ALT-QUERY-FLIP");
    }

    #[test]
    fn shared_ancestry_is_reported_without_independence_credit() {
        let report = run_value(fixture_value()).unwrap();
        assert!(report["shared_ancestry_overlaps"].as_array().unwrap().iter().any(
            |row| row["channel_a"] == "em-specialist"
                && row["channel_b"] == "morphology-review"
                && row["shared_group_ids"][0] == "raw-volume-001"
        ));
        assert_eq!(report["independence_disposition"], "NOT_ESTABLISHED_BY_ANCESTRY_LABELS");
    }

    #[test]
    fn all_list_missed_is_sensitivity_not_ground_truth() {
        let report = run_value(fixture_value()).unwrap();
        assert_eq!(report["all_list_missed_sensitivity_status"], "NOT_RUN");
        assert_eq!(report["biological_truth_disposition"], "NOT_ESTABLISHED_BY_THIS_TOOL");
    }

    #[test]
    fn malformed_source_digest_is_rejected() {
        let mut value = fixture_value();
        value["sampling_binding"]["sampling_frame_sha256"] = json!("not-a-digest");
        assert!(run_value(value).unwrap_err().contains("sampling-frame digest"));
    }
}
