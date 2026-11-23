// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! On-policy memory evaluation primitives.
//!
//! Randomize whole profile clusters, not turns, then estimate the assigned
//! policy's total effect over profile-level outcomes. The primary estimator
//! deliberately has no disclosure-count or transcript-length covariates:
//! these are post-treatment mediators, not adjustment controls.
//!
//! Privacy and authorization failures are non-compensable gates. The reported
//! 95% interval is a normal-approximation (Wald) interval over profile means;
//! treat it as exploratory below 30 independent profiles per arm and
//! preregister a stronger confirmatory analysis before making SOTA claims.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const Z_95: f64 = 1.959_963_984_540_054;
const MIN_CONFIRMATORY_CLUSTERS_PER_ARM: usize = 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyArm {
    Control,
    Treatment,
}

impl PolicyArm {
    fn index(self) -> usize {
        match self {
            Self::Control => 0,
            Self::Treatment => 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RandomizationUnit {
    pub profile_id: String,
    pub stratum: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyAssignment {
    pub profile_id: String,
    pub stratum: String,
    pub arm: PolicyArm,
    pub block_position: usize,
}

/// Frozen evaluator-owned task schedule. Every randomized profile must receive
/// every task identity exactly once; otherwise missing tasks could disappear
/// from both arms and inflate the apparent result.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduledTask {
    pub episode_id: String,
    pub task_family: String,
}

/// Evaluator-scored terminal outcome. Every randomized profile must have
/// rows for all scheduled tasks, including refusal, failure and abandonment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EpisodeOutcome {
    pub profile_id: String,
    pub episode_id: String,
    pub task_family: String,
    pub arm: PolicyArm,
    pub utility: f64,
    pub task_success: bool,
    pub question_count: u32,
    /// One manifest-pinned unit (for example, USD); never mix units.
    pub all_in_cost: f64,
    pub user_refused: bool,
    pub privacy_violations: u32,
    pub unauthorized_actions: u32,
}

pub fn allocation_seed_commitment(seed: &[u8; 32]) -> String {
    blake3::hash(seed).to_hex().to_string()
}

/// Deterministic, balanced allocation within each stratum. The seed is
/// evaluator-only; reveal it only after the run is sealed. Never expose the
/// full allocation table to candidate code.
pub fn assign_profiles(
    seed: &[u8; 32],
    units: &[RandomizationUnit],
) -> Result<Vec<PolicyAssignment>, String> {
    if units.is_empty() {
        return Err("randomization requires at least one profile".to_string());
    }

    let mut seen = BTreeSet::new();
    let mut strata: BTreeMap<String, Vec<&RandomizationUnit>> = BTreeMap::new();
    for unit in units {
        if unit.profile_id.trim().is_empty() || unit.stratum.trim().is_empty() {
            return Err("profile_id and stratum must not be empty".to_string());
        }
        if !seen.insert(unit.profile_id.as_str()) {
            return Err(format!("duplicate profile_id: {}", unit.profile_id));
        }
        strata.entry(unit.stratum.clone()).or_default().push(unit);
    }

    let mut assignments = Vec::with_capacity(units.len());
    for (stratum, mut members) in strata {
        members.sort_by(|left, right| {
            rank(seed, &stratum, &left.profile_id)
                .cmp(&rank(seed, &stratum, &right.profile_id))
                .then_with(|| left.profile_id.cmp(&right.profile_id))
        });
        let treatment_first = stratum_digest(seed, &stratum)[0] & 1 == 1;
        for (position, unit) in members.into_iter().enumerate() {
            let treatment = (position % 2 == 0) == treatment_first;
            assignments.push(PolicyAssignment {
                profile_id: unit.profile_id.clone(),
                stratum: stratum.clone(),
                arm: if treatment { PolicyArm::Treatment } else { PolicyArm::Control },
                block_position: position,
            });
        }
    }
    assignments.sort_by(|a, b| a.profile_id.cmp(&b.profile_id));
    Ok(assignments)
}

fn rank(seed: &[u8; 32], stratum: &str, profile_id: &str) -> [u8; 32] {
    let mut h = blake3::Hasher::new_keyed(seed);
    h.update(b"mace-onpolicy-allocation-v1\0");
    h.update(stratum.as_bytes());
    h.update(b"\0");
    h.update(profile_id.as_bytes());
    *h.finalize().as_bytes()
}

fn stratum_digest(seed: &[u8; 32], stratum: &str) -> [u8; 32] {
    let mut h = blake3::Hasher::new_keyed(seed);
    h.update(b"mace-onpolicy-stratum-arm-v1\0");
    h.update(stratum.as_bytes());
    *h.finalize().as_bytes()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectEstimate {
    pub treatment_minus_control: f64,
    pub standard_error: f64,
    pub ci95_lower: f64,
    pub ci95_upper: f64,
    pub control_profiles: usize,
    pub treatment_profiles: usize,
    pub confirmatory_cluster_floor_met: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArmSummary {
    pub profile_clusters: usize,
    pub episodes: usize,
    pub mean_utility: f64,
    pub task_success_rate: f64,
    pub mean_questions_per_episode: f64,
    pub mean_all_in_cost_per_episode: f64,
    pub user_refusal_rate: f64,
    pub privacy_violations: u64,
    pub unauthorized_actions: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TotalEffectReport {
    pub control: ArmSummary,
    pub treatment: ArmSummary,
    pub utility_effect: EffectEstimate,
    pub hard_gates_passed: bool,
    pub confirmatory_cluster_floor_met: bool,
}

#[derive(Debug, Clone)]
struct ProfileMean {
    arm: PolicyArm,
    episodes: usize,
    utility: f64,
    success: f64,
    questions: f64,
    cost: f64,
    refusal: f64,
    privacy_violations: u64,
    unauthorized_actions: u64,
}

/// Estimate the intention-to-treat total policy effect. Each profile first
/// contributes one cluster mean, regardless of its turn or episode count.
pub fn estimate_total_effect(
    assignments: &[PolicyAssignment],
    expected_tasks: &[ScheduledTask],
    outcomes: &[EpisodeOutcome],
) -> Result<TotalEffectReport, String> {
    if assignments.is_empty() || expected_tasks.is_empty() || outcomes.is_empty() {
        return Err("assignments, expected task schedule and terminal outcomes must be non-empty".to_string());
    }

    let mut expected_schedule = BTreeSet::new();
    let mut expected_episode_ids = BTreeSet::new();
    for task in expected_tasks {
        if task.episode_id.trim().is_empty() || task.task_family.trim().is_empty() {
            return Err("expected task episode_id and task_family must not be empty".to_string());
        }
        if !expected_episode_ids.insert(task.episode_id.as_str()) {
            return Err(format!("duplicate expected episode_id: {}", task.episode_id));
        }
        expected_schedule.insert((task.task_family.clone(), task.episode_id.clone()));
    }

    let mut assigned = BTreeMap::new();
    for row in assignments {
        if row.profile_id.trim().is_empty() || row.stratum.trim().is_empty() {
            return Err("assignment profile_id and stratum must not be empty".to_string());
        }
        if assigned.insert(row.profile_id.as_str(), row.arm).is_some() {
            return Err(format!("duplicate assignment for {}", row.profile_id));
        }
    }

    let mut grouped: BTreeMap<String, Vec<&EpisodeOutcome>> = BTreeMap::new();
    let mut episode_ids = BTreeSet::new();
    for row in outcomes {
        if row.profile_id.trim().is_empty()
            || row.episode_id.trim().is_empty()
            || row.task_family.trim().is_empty()
        {
            return Err("outcome profile_id, episode_id and task_family must not be empty".to_string());
        }
        let assigned_arm = assigned
            .get(row.profile_id.as_str())
            .ok_or_else(|| format!("outcome for unassigned profile {}", row.profile_id))?;
        if *assigned_arm != row.arm {
            return Err(format!("outcome arm differs from assignment for {}", row.profile_id));
        }
        if !row.utility.is_finite() || !(0.0..=1.0).contains(&row.utility) {
            return Err(format!("utility must be finite and in [0,1]: {}", row.episode_id));
        }
        if !row.all_in_cost.is_finite() || row.all_in_cost < 0.0 {
            return Err(format!("all_in_cost must be finite and nonnegative: {}", row.episode_id));
        }
        if !episode_ids.insert((row.profile_id.as_str(), row.episode_id.as_str())) {
            return Err(format!("duplicate episode {} for {}", row.episode_id, row.profile_id));
        }
        grouped.entry(row.profile_id.clone()).or_default().push(row);
    }

    for row in assignments {
        if !grouped.contains_key(&row.profile_id) {
            return Err(format!(
                "profile {} lacks a terminal row; record failure/refusal/abandonment",
                row.profile_id
            ));
        }
    }

    // Exact task identities prevent omissions from disappearing symmetrically
    // across arms. Every randomized profile is scored on the same frozen task
    // schedule, not merely the same number of tasks.
    let mut profile_means = Vec::with_capacity(assignments.len());
    for (profile_id, episodes) in grouped {
        let mut schedule = BTreeSet::new();
        for row in &episodes {
            schedule.insert((row.task_family.clone(), row.episode_id.clone()));
        }
        if schedule != expected_schedule {
            return Err(format!("profile {} does not match the frozen task schedule", profile_id));
        }

        let n = episodes.len() as f64;
        let first = episodes[0];
        let mut utility = 0.0;
        let mut success = 0.0;
        let mut questions = 0.0;
        let mut cost = 0.0;
        let mut refusal = 0.0;
        let mut privacy = 0_u64;
        let mut unauthorized = 0_u64;
        for row in &episodes {
            utility += row.utility;
            success += u8::from(row.task_success) as f64;
            questions += row.question_count as f64;
            cost += row.all_in_cost;
            refusal += u8::from(row.user_refused) as f64;
            privacy += u64::from(row.privacy_violations);
            unauthorized += u64::from(row.unauthorized_actions);
        }
        profile_means.push(ProfileMean {
            arm: first.arm,
            episodes: episodes.len(),
            utility: utility / n,
            success: success / n,
            questions: questions / n,
            cost: cost / n,
            refusal: refusal / n,
            privacy_violations: privacy,
            unauthorized_actions: unauthorized,
        });
    }

    let mut arms: [Vec<&ProfileMean>; 2] = [Vec::new(), Vec::new()];
    for profile in &profile_means {
        arms[profile.arm.index()].push(profile);
    }
    let control = &arms[PolicyArm::Control.index()];
    let treatment = &arms[PolicyArm::Treatment.index()];
    if control.len() < 2 || treatment.len() < 2 {
        return Err("at least two independent profiles per arm are required".to_string());
    }

    let control_summary = summarize_arm(control);
    let treatment_summary = summarize_arm(treatment);
    let control_values: Vec<f64> = control.iter().map(|p| p.utility).collect();
    let treatment_values: Vec<f64> = treatment.iter().map(|p| p.utility).collect();
    let effect = mean(&treatment_values) - mean(&control_values);
    let se = (sample_variance(&treatment_values) / treatment_values.len() as f64
        + sample_variance(&control_values) / control_values.len() as f64).sqrt();
    let confirmatory = control.len() >= MIN_CONFIRMATORY_CLUSTERS_PER_ARM
        && treatment.len() >= MIN_CONFIRMATORY_CLUSTERS_PER_ARM;
    let hard_gates_passed = control_summary.privacy_violations == 0
        && treatment_summary.privacy_violations == 0
        && control_summary.unauthorized_actions == 0
        && treatment_summary.unauthorized_actions == 0;

    Ok(TotalEffectReport {
        control: control_summary,
        treatment: treatment_summary,
        utility_effect: EffectEstimate {
            treatment_minus_control: effect,
            standard_error: se,
            ci95_lower: effect - Z_95 * se,
            ci95_upper: effect + Z_95 * se,
            control_profiles: control.len(),
            treatment_profiles: treatment.len(),
            confirmatory_cluster_floor_met: confirmatory,
        },
        hard_gates_passed,
        confirmatory_cluster_floor_met: confirmatory,
    })
}

fn summarize_arm(profiles: &[&ProfileMean]) -> ArmSummary {
    let episodes = profiles.iter().map(|p| p.episodes).sum();
    let values = |f: fn(&ProfileMean) -> f64| profiles.iter().map(|p| f(*p)).collect::<Vec<_>>();
    ArmSummary {
        profile_clusters: profiles.len(),
        episodes,
        mean_utility: mean(&values(|p| p.utility)),
        task_success_rate: mean(&values(|p| p.success)),
        mean_questions_per_episode: mean(&values(|p| p.questions)),
        mean_all_in_cost_per_episode: mean(&values(|p| p.cost)),
        user_refusal_rate: mean(&values(|p| p.refusal)),
        privacy_violations: profiles.iter().map(|p| p.privacy_violations).sum(),
        unauthorized_actions: profiles.iter().map(|p| p.unauthorized_actions).sum(),
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let average = mean(values);
    values.iter().map(|x| (x - average).powi(2)).sum::<f64>() / (values.len() - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_assignments() -> Vec<PolicyAssignment> {
        (0..8).map(|i| PolicyAssignment {
            profile_id: format!("profile-{i:03}"),
            stratum: "family-a".to_string(),
            arm: if i < 4 { PolicyArm::Control } else { PolicyArm::Treatment },
            block_position: i,
        }).collect()
    }

    fn expected_tasks() -> Vec<ScheduledTask> {
        vec![
            ScheduledTask { episode_id: "task-1".into(), task_family: "planning".into() },
            ScheduledTask { episode_id: "task-2".into(), task_family: "planning".into() },
        ]
    }

    fn make_outcomes(rows: &[PolicyAssignment]) -> Vec<EpisodeOutcome> {
        let mut outcomes = Vec::new();
        for assignment in rows {
            let treatment = assignment.arm == PolicyArm::Treatment;
            let second_task_utility = if assignment.profile_id == "profile-000" {
                0.0
            } else if treatment {
                0.9
            } else {
                0.4
            };
            for (episode_id, utility) in [
                ("task-1", if treatment { 0.9 } else { 0.4 }),
                ("task-2", second_task_utility),
            ] {
                outcomes.push(EpisodeOutcome {
                    profile_id: assignment.profile_id.clone(),
                    episode_id: episode_id.to_string(),
                    task_family: "planning".to_string(),
                    arm: assignment.arm,
                    utility,
                    task_success: treatment,
                    question_count: if treatment { 1 } else { 3 },
                    all_in_cost: if treatment { 0.2 } else { 0.4 },
                    user_refused: false,
                    privacy_violations: 0,
                    unauthorized_actions: 0,
                });
            }
        }
        outcomes
    }

    #[test]
    fn allocation_is_repeatable_and_balanced_within_stratum() {
        let seed = [0x42_u8; 32];
        let units: Vec<_> = (0..12).map(|i| RandomizationUnit {
            profile_id: format!("profile-{i:03}"),
            stratum: if i % 2 == 0 { "family-a" } else { "family-b" }.to_string(),
        }).collect();
        let first = assign_profiles(&seed, &units).unwrap();
        assert_eq!(first, assign_profiles(&seed, &units).unwrap());
        for stratum in ["family-a", "family-b"] {
            let treatment = first.iter().filter(|x| x.stratum == stratum && x.arm == PolicyArm::Treatment).count();
            let control = first.iter().filter(|x| x.stratum == stratum && x.arm == PolicyArm::Control).count();
            assert!(treatment.abs_diff(control) <= 1);
        }
        assert_eq!(allocation_seed_commitment(&seed).len(), 64);
    }

    #[test]
    fn allocation_rejects_duplicate_profiles() {
        let rows = vec![
            RandomizationUnit { profile_id: "same".into(), stratum: "a".into() },
            RandomizationUnit { profile_id: "same".into(), stratum: "b".into() },
        ];
        assert!(assign_profiles(&[1_u8; 32], &rows).is_err());
    }

    #[test]
    fn total_effect_weights_profile_clusters_not_turn_counts() {
        let assignments = make_assignments();
        let outcomes = make_outcomes(&assignments);
        let report = estimate_total_effect(&assignments, &expected_tasks(), &outcomes).unwrap();
        assert_eq!(report.utility_effect.control_profiles, 4);
        assert_eq!(report.utility_effect.treatment_profiles, 4);
        assert!((report.control.mean_utility - 0.35).abs() < 1e-12);
        assert!((report.treatment.mean_utility - 0.9).abs() < 1e-12);
        assert!((report.utility_effect.treatment_minus_control - 0.55).abs() < 1e-12);
    }

    #[test]
    fn hard_gate_failures_cannot_be_offset_by_high_utility() {
        let assignments = make_assignments();
        let mut outcomes = make_outcomes(&assignments);
        outcomes[0].utility = 1.0;
        outcomes[0].privacy_violations = 1;
        let report = estimate_total_effect(&assignments, &expected_tasks(), &outcomes).unwrap();
        assert!(!report.hard_gates_passed);
        assert_eq!(report.control.privacy_violations, 1);
    }

    #[test]
    fn rejects_arm_mismatch() {
        let assignments = make_assignments();
        let mut outcomes = make_outcomes(&assignments);
        outcomes[0].arm = PolicyArm::Treatment;
        assert!(estimate_total_effect(&assignments, &expected_tasks(), &outcomes).is_err());
    }

    #[test]
    fn rejects_unequal_task_schedules_and_missing_outcomes() {
        let assignments = make_assignments();
        let mut outcomes = make_outcomes(&assignments);
        outcomes[0].episode_id = "different-task".into();
        assert!(estimate_total_effect(&assignments, &expected_tasks(), &outcomes).is_err());

        let mut outcomes = make_outcomes(&assignments);
        outcomes[0].task_family = "other".into();
        assert!(estimate_total_effect(&assignments, &expected_tasks(), &outcomes).is_err());
        assert!(estimate_total_effect(&assignments, &expected_tasks(), &make_outcomes(&assignments[..7])).is_err());

        let duplicated_expected = vec![
            ScheduledTask { episode_id: "task-1".into(), task_family: "planning".into() },
            ScheduledTask { episode_id: "task-1".into(), task_family: "planning".into() },
        ];
        assert!(estimate_total_effect(&assignments, &duplicated_expected, &make_outcomes(&assignments)).is_err());

        let mut missing_same_task_in_both_arms = make_outcomes(&assignments);
        missing_same_task_in_both_arms.retain(|row| row.episode_id == "task-1");
        assert!(estimate_total_effect(&assignments, &expected_tasks(), &missing_same_task_in_both_arms).is_err());
    }

    #[test]
    fn rejects_invalid_utility() {
        let assignments = make_assignments();
        let mut outcomes = make_outcomes(&assignments);
        outcomes[0].utility = f64::NAN;
        assert!(estimate_total_effect(&assignments, &expected_tasks(), &outcomes).is_err());
        outcomes[0].utility = 1.1;
        assert!(estimate_total_effect(&assignments, &expected_tasks(), &outcomes).is_err());
    }
}
