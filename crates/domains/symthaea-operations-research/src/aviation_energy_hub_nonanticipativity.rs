// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Nonanticipativity verifier for scenario-indexed energy-hub policies.
//!
//! A scenario-specific exact optimizer sees its entire scenario and may choose
//! different early actions based on future information. This module checks the
//! nonanticipativity contract: scenarios that share the same declared information
//! set at a decision stage must choose the same action at that stage.
//!
//! Information-set labels and their semantics are declared by the caller and need
//! independent review. This verifier proves consistency with those labels; it does
//! not prove that the labels accurately represent when information is actually known.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use sha2::{Digest, Sha256};

use crate::aviation_energy_hub::{
    canonical_energy_hub_digest, verify_energy_hub_plan, EnergyHubError, EnergyHubPlan,
    EnergyHubStepAction, TimeIndexedEnergyHubScenario,
};

const MAX_POLICY_SCENARIOS: usize = 32;

/// A scenario-bound plan plus the information set available before each interval's
/// decision. The information contract reference should identify the declared
/// observation/availability policy for the scenario family.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioPolicyPlan {
    pub scenario_id: String,
    pub plan: EnergyHubPlan,
    pub information_contract_ref: String,
    pub information_set_ids_by_stage: Vec<String>,
    /// Digest of the actual observed/available information history before each action.
    /// Equal histories must share one information set; different histories cannot be merged.
    pub observed_history_digest_by_stage: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonanticipativityViolation {
    pub stage_index: usize,
    pub information_set_id: String,
    pub conflicting_scenario_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonanticipativityReport {
    pub policy_digest: String,
    pub scenarios_verified: usize,
    pub pairwise_action_constraints_checked: u64,
    pub violations: Vec<NonanticipativityViolation>,
}

impl NonanticipativityReport {
    pub fn is_nonanticipative(&self) -> bool {
        self.violations.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NonanticipativityError {
    NoScenarios,
    TooManyScenarios,
    DuplicateScenarioId,
    DuplicatePolicyPlan,
    MissingPolicyPlan { scenario_id: String },
    ExtraPolicyPlan { scenario_id: String },
    MissingIdentifier(&'static str),
    InformationSetCountMismatch { scenario_id: String },
    IncompatibleScenario(&'static str),
    ScenarioDigestMismatch { scenario_id: String },
    InvalidInformationHistory {
        scenario_id: String,
        stage_index: usize,
        information_set_id: String,
    },
    InvalidHistoryDigest {
        scenario_id: String,
        stage_index: usize,
    },
    InformationSetHistoryMismatch {
        stage_index: usize,
        information_set_id: String,
    },
    IdenticalHistorySplitAcrossInformationSets {
        stage_index: usize,
    },
    PlanVerification {
        scenario_id: String,
        source: EnergyHubError,
    },
    ArithmeticOverflow,
}

impl fmt::Display for NonanticipativityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoScenarios => write!(f, "at least one scenario is required"),
            Self::TooManyScenarios => write!(f, "policy scenario count exceeds the declared limit"),
            Self::DuplicateScenarioId => write!(f, "duplicate scenario identity"),
            Self::DuplicatePolicyPlan => write!(f, "more than one policy plan references a scenario"),
            Self::MissingPolicyPlan { scenario_id } => {
                write!(f, "policy plan is missing for scenario {scenario_id}")
            }
            Self::ExtraPolicyPlan { scenario_id } => {
                write!(f, "policy references unknown scenario {scenario_id}")
            }
            Self::MissingIdentifier(field) => write!(f, "missing identifier: {field}"),
            Self::InformationSetCountMismatch { scenario_id } => {
                write!(f, "information-set count does not match horizon for {scenario_id}")
            }
            Self::IncompatibleScenario(field) => {
                write!(f, "policy scenarios do not share comparable {field}")
            }
            Self::ScenarioDigestMismatch { scenario_id } => {
                write!(f, "scenario digest mismatch for {scenario_id}")
            }
            Self::InvalidInformationHistory {
                scenario_id,
                stage_index,
                information_set_id,
            } => write!(
                f,
                "information set {information_set_id} at stage {stage_index} has an inconsistent prior history for {scenario_id}"
            ),
            Self::InvalidHistoryDigest { scenario_id, stage_index } => write!(
                f,
                "invalid observed-history SHA-256 digest for {scenario_id} at stage {stage_index}"
            ),
            Self::InformationSetHistoryMismatch { stage_index, information_set_id } => write!(
                f,
                "information set {information_set_id} at stage {stage_index} combines different observed histories"
            ),
            Self::IdenticalHistorySplitAcrossInformationSets { stage_index } => write!(
                f,
                "identical observed histories are split into different information sets at stage {stage_index}"
            ),
            Self::PlanVerification { scenario_id, source } => {
                write!(f, "plan verification failed for {scenario_id}: {source}")
            }
            Self::ArithmeticOverflow => write!(f, "nonanticipativity accounting overflowed"),
        }
    }
}

impl std::error::Error for NonanticipativityError {}

/// Verify every scenario-bound plan and check nonanticipativity for every declared
/// information set. Scenarios must have the same service boundary and ordered time grid.
/// The returned digest binds scenario identities, plan actions and information labels.
pub fn verify_nonanticipative_policy(
    scenarios: &[TimeIndexedEnergyHubScenario],
    policy_plans: &[ScenarioPolicyPlan],
) -> Result<NonanticipativityReport, NonanticipativityError> {
    if scenarios.is_empty() {
        return Err(NonanticipativityError::NoScenarios);
    }
    if scenarios.len() > MAX_POLICY_SCENARIOS {
        return Err(NonanticipativityError::TooManyScenarios);
    }
    if scenarios.len() != policy_plans.len() {
        return Err(NonanticipativityError::MissingIdentifier("one plan per scenario"));
    }

    let mut scenario_by_id = BTreeMap::<&str, &TimeIndexedEnergyHubScenario>::new();
    for scenario in scenarios {
        if scenario.scenario_id.trim().is_empty() {
            return Err(NonanticipativityError::MissingIdentifier("scenario ID"));
        }
        if scenario_by_id.insert(scenario.scenario_id.as_str(), scenario).is_some() {
            return Err(NonanticipativityError::DuplicateScenarioId);
        }
        let digest = canonical_energy_hub_digest(scenario)
            .map_err(|_| NonanticipativityError::ScenarioDigestMismatch {
                scenario_id: scenario.scenario_id.clone(),
            })?;
        if scenario.input_digest != digest {
            return Err(NonanticipativityError::ScenarioDigestMismatch {
                scenario_id: scenario.scenario_id.clone(),
            });
        }
    }

    let reference = scenarios.first().ok_or(NonanticipativityError::NoScenarios)?;
    let reference_interval_ids = reference
        .intervals
        .iter()
        .map(|interval| interval.id.as_str())
        .collect::<Vec<_>>();
    for scenario in scenarios.iter().skip(1) {
        if scenario.currency_code != reference.currency_code {
            return Err(NonanticipativityError::IncompatibleScenario("currency"));
        }
        if scenario.quantum_kwh != reference.quantum_kwh {
            return Err(NonanticipativityError::IncompatibleScenario("energy quantum"));
        }
        if scenario.service.period_id != reference.service.period_id
            || scenario.service.lifecycle_boundary_id != reference.service.lifecycle_boundary_id
            || scenario.service.completed_passenger_km != reference.service.completed_passenger_km
            || scenario.service.available_seat_km != reference.service.available_seat_km
        {
            return Err(NonanticipativityError::IncompatibleScenario(
                "transport service/accounting boundary",
            ));
        }
        let candidate_interval_ids = scenario
            .intervals
            .iter()
            .map(|interval| interval.id.as_str())
            .collect::<Vec<_>>();
        if candidate_interval_ids != reference_interval_ids {
            return Err(NonanticipativityError::IncompatibleScenario(
                "ordered interval grid",
            ));
        }
    }

    let mut plans_by_id = BTreeMap::<&str, &ScenarioPolicyPlan>::new();
    for policy in policy_plans {
        if policy.scenario_id.trim().is_empty() {
            return Err(NonanticipativityError::MissingIdentifier("policy scenario ID"));
        }
        if policy.information_contract_ref.trim().is_empty() {
            return Err(NonanticipativityError::MissingIdentifier(
                "information contract reference",
            ));
        }
        if policy.information_set_ids_by_stage.len() != reference.intervals.len()
            || policy.observed_history_digest_by_stage.len() != reference.intervals.len()
        {
            return Err(NonanticipativityError::InformationSetCountMismatch {
                scenario_id: policy.scenario_id.clone(),
            });
        }
        if !scenario_by_id.contains_key(policy.scenario_id.as_str()) {
            return Err(NonanticipativityError::ExtraPolicyPlan {
                scenario_id: policy.scenario_id.clone(),
            });
        }
        if plans_by_id
            .insert(policy.scenario_id.as_str(), policy)
            .is_some()
        {
            return Err(NonanticipativityError::DuplicatePolicyPlan);
        }
        for (stage_index, information_set_id) in
            policy.information_set_ids_by_stage.iter().enumerate()
        {
            if information_set_id.trim().is_empty() {
                return Err(NonanticipativityError::MissingIdentifier(
                    "information-set ID",
                ));
            }
            let history_digest = &policy.observed_history_digest_by_stage[stage_index];
            if !is_valid_sha256(history_digest) {
                return Err(NonanticipativityError::InvalidHistoryDigest {
                    scenario_id: policy.scenario_id.clone(),
                    stage_index,
                });
            }
        }
    }

    for scenario in scenarios {
        if !plans_by_id.contains_key(scenario.scenario_id.as_str()) {
            return Err(NonanticipativityError::MissingPolicyPlan {
                scenario_id: scenario.scenario_id.clone(),
            });
        }
        let policy = plans_by_id[scenario.scenario_id.as_str()];
        verify_energy_hub_plan(scenario, &policy.plan).map_err(|source| {
            NonanticipativityError::PlanVerification {
                scenario_id: scenario.scenario_id.clone(),
                source,
            }
        })?;
        if policy.plan.scenario_id != policy.scenario_id {
            return Err(NonanticipativityError::PlanVerification {
                scenario_id: policy.scenario_id.clone(),
                source: EnergyHubError::PlanIdentityMismatch,
            });
        }
    }

    let mut violations = Vec::new();
    let mut pairwise_action_constraints_checked = 0_u64;

    for stage_index in 0..reference.intervals.len() {
        // The declared partition must match the supplied observable-history identity.
        // Otherwise a caller could evade nonanticipativity by splitting indistinguishable
        // histories, or by merging futures after they become observably different.
        for left_index in 0..policy_plans.len() {
            for right_index in (left_index + 1)..policy_plans.len() {
                let left = &policy_plans[left_index];
                let right = &policy_plans[right_index];
                let same_information_set = left.information_set_ids_by_stage[stage_index]
                    == right.information_set_ids_by_stage[stage_index];
                let same_observed_history = left.observed_history_digest_by_stage[stage_index]
                    == right.observed_history_digest_by_stage[stage_index];
                if same_information_set && !same_observed_history {
                    return Err(NonanticipativityError::InformationSetHistoryMismatch {
                        stage_index,
                        information_set_id: left.information_set_ids_by_stage[stage_index].clone(),
                    });
                }
                if same_observed_history && !same_information_set {
                    return Err(NonanticipativityError::IdenticalHistorySplitAcrossInformationSets {
                        stage_index,
                    });
                }
            }
        }

        let mut groups = BTreeMap::<&str, Vec<&ScenarioPolicyPlan>>::new();
        for policy in policy_plans {
            let information_set_id = &policy.information_set_ids_by_stage[stage_index];
            groups
                .entry(information_set_id.as_str())
                .or_default()
                .push(policy);
        }

        for (information_set_id, members) in groups {
            let member_count = u64::try_from(members.len())
                .map_err(|_| NonanticipativityError::ArithmeticOverflow)?;
            let pair_count = member_count
                .checked_mul(member_count.saturating_sub(1))
                .and_then(|count| count.checked_div(2))
                .ok_or(NonanticipativityError::ArithmeticOverflow)?;
            pairwise_action_constraints_checked = pairwise_action_constraints_checked
                .checked_add(pair_count)
                .ok_or(NonanticipativityError::ArithmeticOverflow)?;

            let Some(first) = members.first() else {
                continue;
            };
            let reference_action = first
                .plan
                .actions
                .get(stage_index)
                .ok_or(NonanticipativityError::InformationSetCountMismatch {
                    scenario_id: first.scenario_id.clone(),
                })?;
            let all_equal = members.iter().all(|member| {
                member.plan.actions.get(stage_index) == Some(reference_action)
            });
            if !all_equal {
                let mut scenario_ids = members
                    .iter()
                    .map(|member| member.scenario_id.clone())
                    .collect::<Vec<_>>();
                scenario_ids.sort();
                violations.push(NonanticipativityViolation {
                    stage_index,
                    information_set_id: information_set_id.to_owned(),
                    conflicting_scenario_ids: scenario_ids,
                });
            }
        }
    }

    violations.sort_by(|left, right| {
        left.stage_index
            .cmp(&right.stage_index)
            .then_with(|| left.information_set_id.cmp(&right.information_set_id))
    });

    Ok(NonanticipativityReport {
        policy_digest: canonical_policy_digest(policy_plans)?,
        scenarios_verified: scenarios.len(),
        pairwise_action_constraints_checked,
        violations,
    })
}

/// Versioned digest over the declared policy plans. The scenario digest binds the
/// model inputs; this policy digest binds the chosen actions and information partitions.
pub fn canonical_policy_digest(
    policy_plans: &[ScenarioPolicyPlan],
) -> Result<String, NonanticipativityError> {
    if policy_plans.is_empty() {
        return Err(NonanticipativityError::NoScenarios);
    }
    let mut sorted = policy_plans.iter().collect::<Vec<_>>();
    sorted.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    let mut ids = BTreeSet::new();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"symthaea.clean-airline.nonanticipative-policy.v1\0");
    put_u32(&mut bytes, u32::try_from(sorted.len()).map_err(|_| {
        NonanticipativityError::ArithmeticOverflow
    })?);
    for policy in sorted {
        if !ids.insert(policy.scenario_id.as_str()) {
            return Err(NonanticipativityError::DuplicatePolicyPlan);
        }
        put_string(&mut bytes, &policy.scenario_id);
        put_string(&mut bytes, &policy.plan.input_digest);
        put_string(&mut bytes, &policy.information_contract_ref);
        put_u32(
            &mut bytes,
            u32::try_from(policy.information_set_ids_by_stage.len())
                .map_err(|_| NonanticipativityError::ArithmeticOverflow)?,
        );
        for info_set_id in &policy.information_set_ids_by_stage {
            put_string(&mut bytes, info_set_id);
        }
        put_u32(
            &mut bytes,
            u32::try_from(policy.observed_history_digest_by_stage.len())
                .map_err(|_| NonanticipativityError::ArithmeticOverflow)?,
        );
        for history_digest in &policy.observed_history_digest_by_stage {
            put_string(&mut bytes, history_digest);
        }
        put_u32(
            &mut bytes,
            u32::try_from(policy.plan.actions.len())
                .map_err(|_| NonanticipativityError::ArithmeticOverflow)?,
        );
        for action in &policy.plan.actions {
            encode_action(&mut bytes, action);
        }
    }
    Ok(sha256_hex(&bytes))
}

fn encode_action(bytes: &mut Vec<u8>, action: &EnergyHubStepAction) {
    put_string(bytes, &action.interval_id);
    for value in [
        action.renewable_direct_quanta,
        action.grid_import_quanta,
        action.battery_discharge_output_quanta,
        action.renewable_charge_input_quanta,
        action.renewable_curtailed_quanta,
        action.soc_before_quanta,
        action.soc_after_quanta,
    ] {
        put_u32(bytes, value);
    }
}

fn is_valid_sha256(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64
        && hex.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut result = String::with_capacity(7 + digest.len() * 2);
    result.push_str("sha256:");
    for byte in digest.iter() {
        use std::fmt::Write as _;
        write!(&mut result, "{byte:02x}").expect("writing to String cannot fail");
    }
    result
}

fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_string(bytes: &mut Vec<u8>, value: &str) {
    put_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value.as_bytes());
}
