//! Matched-trial evaluation for cost-sensitive active sensing.
//!
//! This module audits evaluation structure and reports utility/resource metrics.
//! It does not establish sensor truth, simulator fidelity, or physical safety.
//! Missing energy/latency records remain missing; they are never imputed as zero.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SensingArm {
    NoExtraSensing,
    FixedRate,
    UncertaintyTriggered,
    ValueOfInformation,
    PlaceboCost,
}

impl SensingArm {
    pub const ALL: [Self; 5] = [
        Self::NoExtraSensing,
        Self::FixedRate,
        Self::UncertaintyTriggered,
        Self::ValueOfInformation,
        Self::PlaceboCost,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoExtraSensing => "no_extra_sensing",
            Self::FixedRate => "fixed_rate",
            Self::UncertaintyTriggered => "uncertainty_triggered",
            Self::ValueOfInformation => "value_of_information",
            Self::PlaceboCost => "placebo_cost",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SensingSplit {
    InDistribution,
    OutOfDistribution,
    Adversarial,
}

impl SensingSplit {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InDistribution => "test_id",
            Self::OutOfDistribution => "test_ood",
            Self::Adversarial => "test_adversarial",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SensingEvidenceClass {
    SyntheticContractFixture,
    Simulator,
    HardwareInLoop,
    PhysicalField,
    IndependentlyReviewed,
}

impl SensingEvidenceClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SyntheticContractFixture => "synthetic_contract_fixture",
            Self::Simulator => "simulator",
            Self::HardwareInLoop => "hardware_in_loop",
            Self::PhysicalField => "physical_field",
            Self::IndependentlyReviewed => "independently_reviewed",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SensingTrial {
    pub scenario_id: String,
    pub scenario_family: String,
    pub task_id: String,
    pub trial_id: String,
    pub seed: u64,
    pub split: SensingSplit,
    pub evidence_class: SensingEvidenceClass,
    pub arm: SensingArm,
    /// None means no valid rubric score was obtained; it is not a zero score.
    pub progress: Option<f64>,
    pub valid_run: bool,
    pub completed: bool,
    pub safety_violation: bool,
    pub safety_event_count: u32,
    pub sensor_requests: u32,
    pub sensor_acquisitions: u32,
    /// Sensing-only energy; None means unmeasured.
    pub sensing_energy_j: Option<f64>,
    /// Sensing-only latency; None means unmeasured.
    pub sensing_latency_ms: Option<f64>,
    /// Whole-episode energy; None means unmeasured.
    pub total_energy_j: Option<f64>,
    pub elapsed_ms: Option<f64>,
    /// Forecast emitted before the outcome; None means no forecast was recorded.
    pub predicted_success_probability: Option<f64>,
}

impl SensingTrial {
    pub fn validate(&self) -> Result<(), String> {
        for (label, value) in [
            ("scenario_id", self.scenario_id.as_str()),
            ("scenario_family", self.scenario_family.as_str()),
            ("task_id", self.task_id.as_str()),
            ("trial_id", self.trial_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(format!("{} must not be empty", label));
            }
        }
        if self.sensor_acquisitions > self.sensor_requests {
            return Err(format!(
                "{}: sensor acquisitions exceed requests",
                self.trial_id
            ));
        }
        match self.progress {
            Some(value) if !value.is_finite() || !(0.0..=1.0).contains(&value) => {
                return Err(format!("{}: progress must be finite and in [0, 1]", self.trial_id));
            }
            _ => {}
        }
        if self.valid_run != self.progress.is_some() {
            return Err(format!(
                "{}: valid_run must agree with whether progress is present",
                self.trial_id
            ));
        }
        if self.completed != self.progress.is_some_and(|value| value >= 1.0) {
            return Err(format!(
                "{}: completion must agree with a full 1.0 progress score",
                self.trial_id
            ));
        }
        for (label, value) in [
            ("sensing_energy_j", self.sensing_energy_j),
            ("sensing_latency_ms", self.sensing_latency_ms),
            ("total_energy_j", self.total_energy_j),
            ("elapsed_ms", self.elapsed_ms),
        ] {
            if value.is_some_and(|v| !v.is_finite() || v < 0.0) {
                return Err(format!("{}: {} must be finite and non-negative", self.trial_id, label));
            }
        }
        if self
            .predicted_success_probability
            .is_some_and(|p| !p.is_finite() || !(0.0..=1.0).contains(&p))
        {
            return Err(format!(
                "{}: predicted success probability must be finite and in [0, 1]",
                self.trial_id
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SensingArmSummary {
    pub split: SensingSplit,
    pub evidence_class: SensingEvidenceClass,
    pub arm: SensingArm,
    pub trial_count: usize,
    pub scenario_count: usize,
    pub scenario_family_count: usize,
    pub valid_run_rate: f64,
    pub completion_rate: f64,
    pub mean_scored_progress: Option<f64>,
    pub forecast_count: usize,
    pub brier_score: Option<f64>,
    pub mean_sensor_requests: f64,
    pub mean_sensor_acquisitions: f64,
    pub sensing_energy_measured_count: usize,
    pub sensing_energy_missing_count: usize,
    pub mean_sensing_energy_j: Option<f64>,
    pub sensing_latency_measured_count: usize,
    pub sensing_latency_missing_count: usize,
    pub mean_sensing_latency_ms: Option<f64>,
    pub total_energy_measured_count: usize,
    pub mean_total_energy_j: Option<f64>,
    pub safety_event_count: u64,
    pub safety_violation_count: usize,
    pub mean_elapsed_ms: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SensingArmComparison {
    pub split: SensingSplit,
    pub evidence_class: SensingEvidenceClass,
    pub arm: SensingArm,
    pub reference_arm: SensingArm,
    pub paired_episode_count: usize,
    pub scenario_count: usize,
    pub scenario_family_count: usize,
    /// Candidate arm minus fixed-rate baseline; invalid/missing scores are not imputed.
    pub paired_progress_delta: Option<f64>,
    /// Completion is defined for every attempt; non-completion remains in the denominator.
    pub completion_rate_delta: f64,
    pub mean_sensor_acquisition_delta: f64,
    pub sensing_energy_pair_count: usize,
    pub mean_sensing_energy_delta_j: Option<f64>,
    pub sensing_latency_pair_count: usize,
    pub mean_sensing_latency_delta_ms: Option<f64>,
    pub total_energy_pair_count: usize,
    pub mean_total_energy_delta_j: Option<f64>,
    /// Equal-weighted over scenario families after averaging repeated trials per scenario.
    pub family_macro_progress_delta: Option<f64>,
    pub bootstrap_95_ci_lower: Option<f64>,
    pub bootstrap_95_ci_upper: Option<f64>,
    pub inference_status: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SensingAudit {
    pub passed: bool,
    pub trial_count: usize,
    pub scenario_count: usize,
    pub scenario_family_count: usize,
    pub safety_violation_count: usize,
    pub arm_summaries: Vec<SensingArmSummary>,
    pub comparisons_to_fixed_rate: Vec<SensingArmComparison>,
    pub findings: Vec<String>,
    pub claim_ceiling: &'static str,
}

type CellKey = (SensingSplit, SensingEvidenceClass, String, u64);

fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let collected: Vec<f64> = values.collect();
    if collected.is_empty() {
        None
    } else {
        Some(collected.iter().sum::<f64>() / collected.len() as f64)
    }
}

fn rate(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 { 0.0 } else { numerator as f64 / denominator as f64 }
}

fn find_arm<'a, 'b>(
    cell: &'a BTreeMap<SensingArm, &'b SensingTrial>,
    arm: SensingArm,
) -> Option<&'b SensingTrial> {
    cell.get(&arm).copied()
}

fn xorshift(state: &mut u64) -> u64 {
    let mut value = *state;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    *state = value;
    value
}

fn quantile(sorted: &[f64], probability: f64) -> Option<f64> {
    if sorted.is_empty() {
        None
    } else {
        let index = ((sorted.len() - 1) as f64 * probability).floor() as usize;
        sorted.get(index).copied()
    }
}

fn family_macro_values(
    task_deltas: &BTreeMap<(String, String), (f64, usize)>,
) -> BTreeMap<String, f64> {
    let mut family_tasks: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for ((family, _scenario), (sum, count)) in task_deltas {
        if *count > 0 {
            family_tasks.entry(family.clone()).or_default().push(sum / *count as f64);
        }
    }
    family_tasks
        .into_iter()
        .filter_map(|(family, values)| {
            mean(values.into_iter()).map(|value| (family, value))
        })
        .collect()
}

/// Audit the fixed five-arm experiment. Every scenario × seed × split × evidence
/// cell must have exactly one trial from each arm. The safety violation count is
/// retained in a failed report rather than causing the evidence to disappear.
pub fn audit_sensing_trials(
    trials: &[SensingTrial],
    bootstrap_replicates: usize,
    seed: u64,
) -> Result<SensingAudit, Vec<String>> {
    if trials.is_empty() {
        return Err(vec!["no sensing trials supplied".to_string()]);
    }
    let mut errors = Vec::new();
    let mut cell_map: BTreeMap<CellKey, BTreeMap<SensingArm, &SensingTrial>> = BTreeMap::new();
    let mut metadata: BTreeMap<CellKey, (String, String, String)> = BTreeMap::new();

    for trial in trials {
        if let Err(error) = trial.validate() {
            errors.push(error);
        }
        let key = (
            trial.split,
            trial.evidence_class,
            trial.scenario_id.clone(),
            trial.seed,
        );
        let cell = cell_map.entry(key.clone()).or_default();
        if cell.insert(trial.arm, trial).is_some() {
            errors.push(format!(
                "{}: duplicate arm {} in paired scenario/seed cell",
                trial.trial_id,
                trial.arm.as_str()
            ));
        }
        let this_metadata = (
            trial.scenario_family.clone(),
            trial.task_id.clone(),
            trial.trial_id.clone(),
        );
        if let Some(previous) = metadata.insert(key.clone(), this_metadata.clone()) {
            if previous != this_metadata {
                errors.push(format!(
                    "{}: paired arms disagree on family, task, or trial ID",
                    trial.trial_id
                ));
            }
        }
    }

    for (key, cell) in &cell_map {
        for arm in SensingArm::ALL {
            if cell.get(&arm).is_none() {
                errors.push(format!(
                    "scenario {} seed {} evidence {} split {} is missing arm {}",
                    key.2,
                    key.3,
                    key.1.as_str(),
                    key.0.as_str(),
                    arm.as_str()
                ));
            }
        }
        if cell.len() != SensingArm::ALL.len() {
            errors.push(format!(
                "scenario {} seed {} has {} unique arms; expected {}",
                key.2,
                key.3,
                cell.len(),
                SensingArm::ALL.len()
            ));
        }
    }
    errors.sort();
    if !errors.is_empty() {
        return Err(errors);
    }

    let mut grouped: BTreeMap<
        (SensingSplit, SensingEvidenceClass, SensingArm),
        Vec<&SensingTrial>,
    > = BTreeMap::new();
    for trial in trials {
        grouped
            .entry((trial.split, trial.evidence_class, trial.arm))
            .or_default()
            .push(trial);
    }

    let mut arm_summaries = Vec::new();
    for ((split, evidence_class, arm), rows) in &grouped {
        let count = rows.len();
        let scenarios: BTreeSet<&str> = rows.iter().map(|row| row.scenario_id.as_str()).collect();
        let families: BTreeSet<&str> = rows.iter().map(|row| row.scenario_family.as_str()).collect();
        let scored_progress = mean(rows.iter().filter_map(|row| row.progress));
        let forecasts: Vec<f64> = rows
            .iter()
            .filter_map(|row| {
                row.predicted_success_probability.map(|p| {
                    let outcome = if row.completed { 1.0 } else { 0.0 };
                    (p - outcome).powi(2)
                })
            })
            .collect();
        let sensing_energy_values: Vec<f64> = rows.iter().filter_map(|row| row.sensing_energy_j).collect();
        let sensing_latency_values: Vec<f64> = rows.iter().filter_map(|row| row.sensing_latency_ms).collect();
        let total_energy_values: Vec<f64> = rows.iter().filter_map(|row| row.total_energy_j).collect();
        let elapsed_values: Vec<f64> = rows.iter().filter_map(|row| row.elapsed_ms).collect();

        arm_summaries.push(SensingArmSummary {
            split: *split,
            evidence_class: *evidence_class,
            arm: *arm,
            trial_count: count,
            scenario_count: scenarios.len(),
            scenario_family_count: families.len(),
            valid_run_rate: rate(rows.iter().filter(|row| row.valid_run).count(), count),
            completion_rate: rate(rows.iter().filter(|row| row.completed).count(), count),
            mean_scored_progress: scored_progress,
            forecast_count: forecasts.len(),
            brier_score: mean(forecasts.into_iter()),
            mean_sensor_requests: mean(rows.iter().map(|row| row.sensor_requests as f64)).unwrap_or(0.0),
            mean_sensor_acquisitions: mean(rows.iter().map(|row| row.sensor_acquisitions as f64)).unwrap_or(0.0),
            sensing_energy_measured_count: sensing_energy_values.len(),
            sensing_energy_missing_count: count - sensing_energy_values.len(),
            mean_sensing_energy_j: mean(sensing_energy_values.into_iter()),
            sensing_latency_measured_count: sensing_latency_values.len(),
            sensing_latency_missing_count: count - sensing_latency_values.len(),
            mean_sensing_latency_ms: mean(sensing_latency_values.into_iter()),
            total_energy_measured_count: total_energy_values.len(),
            mean_total_energy_j: mean(total_energy_values.into_iter()),
            safety_event_count: rows.iter().map(|row| row.safety_event_count as u64).sum(),
            safety_violation_count: rows.iter().filter(|row| row.safety_violation).count(),
            mean_elapsed_ms: mean(elapsed_values.into_iter()),
        });
    }
    arm_summaries.sort_by_key(|row| (row.split, row.evidence_class, row.arm));

    let mut comparisons = Vec::new();
    let strata: BTreeSet<(SensingSplit, SensingEvidenceClass)> =
        cell_map.keys().map(|(split, evidence, _, _)| (*split, *evidence)).collect();
    for (split, evidence_class) in strata {
        for arm in SensingArm::ALL {
            if arm == SensingArm::FixedRate {
                continue;
            }
            let mut task_progress_deltas: BTreeMap<(String, String), (f64, usize)> = BTreeMap::new();
            let mut task_completion_deltas: BTreeMap<(String, String), (f64, usize)> = BTreeMap::new();
            let mut episode_progress_deltas = Vec::new();
            let mut completion_deltas = Vec::new();
            let mut acquisition_deltas = Vec::new();
            let mut sensing_energy_deltas = Vec::new();
            let mut sensing_latency_deltas = Vec::new();
            let mut total_energy_deltas = Vec::new();
            let mut families = BTreeSet::new();
            let mut scenario_ids = BTreeSet::new();
            let mut paired_episode_count = 0usize;

            for (key, cell) in &cell_map {
                if key.0 != split || key.1 != evidence_class {
                    continue;
                }
                let candidate = find_arm(cell, arm).expect("pair coverage checked");
                let reference = find_arm(cell, SensingArm::FixedRate).expect("pair coverage checked");
                paired_episode_count += 1;
                families.insert(candidate.scenario_family.clone());
                scenario_ids.insert(candidate.scenario_id.clone());

                let completion_delta = (candidate.completed as u8 as f64) - (reference.completed as u8 as f64);
                completion_deltas.push(completion_delta);
                let ckey = (candidate.scenario_family.clone(), candidate.scenario_id.clone());
                let completion_entry = task_completion_deltas.entry(ckey.clone()).or_insert((0.0, 0));
                completion_entry.0 += completion_delta;
                completion_entry.1 += 1;

                acquisition_deltas.push(candidate.sensor_acquisitions as f64 - reference.sensor_acquisitions as f64);
                if let (Some(candidate_energy), Some(reference_energy)) =
                    (candidate.sensing_energy_j, reference.sensing_energy_j)
                {
                    sensing_energy_deltas.push(candidate_energy - reference_energy);
                }
                if let (Some(candidate_latency), Some(reference_latency)) =
                    (candidate.sensing_latency_ms, reference.sensing_latency_ms)
                {
                    sensing_latency_deltas.push(candidate_latency - reference_latency);
                }
                if let (Some(candidate_energy), Some(reference_energy)) =
                    (candidate.total_energy_j, reference.total_energy_j)
                {
                    total_energy_deltas.push(candidate_energy - reference_energy);
                }
                if let (Some(candidate_progress), Some(reference_progress)) =
                    (candidate.progress, reference.progress)
                {
                    let delta = candidate_progress - reference_progress;
                    episode_progress_deltas.push(delta);
                    let entry = task_progress_deltas.entry(ckey).or_insert((0.0, 0));
                    entry.0 += delta;
                    entry.1 += 1;
                }
            }

            let family_progress = family_macro_values(&task_progress_deltas);
            let family_completion = family_macro_values(&task_completion_deltas);
            let family_count = family_progress.len();
            let family_macro_progress_delta = mean(family_progress.values().copied());
            let (lower, upper, inference_status) = if family_count < 5 || bootstrap_replicates < 100 {
                (None, None, "insufficient_scenario_families_or_bootstrap_replicates".to_string())
            } else {
                let family_values: Vec<f64> = family_progress.values().copied().collect();
                let mut rng = seed
                    ^ (split.as_str().bytes().map(u64::from).sum::<u64>() << 1)
                    ^ (evidence_class.as_str().bytes().map(u64::from).sum::<u64>() << 2)
                    ^ (arm.as_str().bytes().map(u64::from).sum::<u64>() << 3);
                if rng == 0 { rng = 0x9E3779B97F4A7C15; }
                let mut samples = Vec::with_capacity(bootstrap_replicates);
                for _ in 0..bootstrap_replicates {
                    let mut total = 0.0;
                    for _ in 0..family_values.len() {
                        let idx = (xorshift(&mut rng) as usize) % family_values.len();
                        total += family_values[idx];
                    }
                    samples.push(total / family_values.len() as f64);
                }
                samples.sort_by(f64::total_cmp);
                let lo = quantile(&samples, 0.025);
                let hi = quantile(&samples, 0.975);
                let status = match (lo, hi) {
                    (Some(l), Some(h)) if l > 0.0 => "candidate_better_on_family_bootstrap",
                    (Some(l), Some(h)) if h < 0.0 => "fixed_rate_better_on_family_bootstrap",
                    _ => "inconclusive_interval_overlaps_zero",
                }.to_string();
                (lo, hi, status)
            };

            comparisons.push(SensingArmComparison {
                split,
                evidence_class,
                arm,
                reference_arm: SensingArm::FixedRate,
                paired_episode_count,
                scenario_count: scenario_ids.len(),
                scenario_family_count: families.len(),
                paired_progress_delta: mean(episode_progress_deltas.into_iter()),
                completion_rate_delta: mean(completion_deltas.into_iter()).unwrap_or(0.0),
                mean_sensor_acquisition_delta: mean(acquisition_deltas.into_iter()).unwrap_or(0.0),
                sensing_energy_pair_count: sensing_energy_deltas.len(),
                mean_sensing_energy_delta_j: mean(sensing_energy_deltas.into_iter()),
                sensing_latency_pair_count: sensing_latency_deltas.len(),
                mean_sensing_latency_delta_ms: mean(sensing_latency_deltas.into_iter()),
                total_energy_pair_count: total_energy_deltas.len(),
                mean_total_energy_delta_j: mean(total_energy_deltas.into_iter()),
                family_macro_progress_delta,
                bootstrap_95_ci_lower: lower,
                bootstrap_95_ci_upper: upper,
                inference_status,
            });
            let _family_macro_completion_delta = mean(family_completion.values().copied());
        }
    }
    comparisons.sort_by_key(|row| (row.split, row.evidence_class, row.arm));

    let safety_violation_count = trials.iter().filter(|row| row.safety_violation).count();
    let mut findings = Vec::new();
    if safety_violation_count > 0 {
        findings.push(format!(
            "{} trial(s) record a safety violation; qualification fails",
            safety_violation_count
        ));
    }
    Ok(SensingAudit {
        passed: safety_violation_count == 0,
        trial_count: trials.len(),
        scenario_count: cell_map.keys().map(|key| (key.0, key.1, key.2.clone())).collect::<BTreeSet<_>>().len(),
        scenario_family_count: metadata.values().map(|(family, _, _)| family.clone()).collect::<BTreeSet<_>>().len(),
        safety_violation_count,
        arm_summaries,
        comparisons_to_fixed_rate: comparisons,
        findings,
        claim_ceiling: "synthetic_and_recorded_trial_comparisons_do_not_establish_physical_safety_or_general_capability",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trial(scenario_index: usize, arm: SensingArm) -> SensingTrial {
        let scenario = format!("SCENARIO-{scenario_index:02}");
        let family = format!("FAMILY-{scenario_index:02}");
        let progress = match arm {
            SensingArm::ValueOfInformation => 0.95,
            SensingArm::UncertaintyTriggered => 0.90,
            SensingArm::FixedRate => 0.82,
            SensingArm::NoExtraSensing => 0.80,
            SensingArm::PlaceboCost => 0.80,
        };
        SensingTrial {
            scenario_id: scenario.clone(),
            scenario_family: family,
            task_id: format!("TASK-{scenario_index:02}"),
            trial_id: format!("TRIAL-{scenario_index:02}"),
            seed: scenario_index as u64,
            split: SensingSplit::OutOfDistribution,
            evidence_class: SensingEvidenceClass::Simulator,
            arm,
            progress: Some(progress),
            valid_run: true,
            completed: false,
            safety_violation: false,
            safety_event_count: 0,
            sensor_requests: match arm {
                SensingArm::NoExtraSensing => 0,
                SensingArm::FixedRate => 5,
                SensingArm::UncertaintyTriggered => 3,
                SensingArm::ValueOfInformation => 2,
                SensingArm::PlaceboCost => 5,
            },
            sensor_acquisitions: match arm {
                SensingArm::NoExtraSensing => 0,
                SensingArm::FixedRate => 5,
                SensingArm::UncertaintyTriggered => 3,
                SensingArm::ValueOfInformation => 2,
                SensingArm::PlaceboCost => 5,
            },
            sensing_energy_j: match arm {
                SensingArm::NoExtraSensing => Some(0.0),
                SensingArm::FixedRate => Some(50.0),
                SensingArm::UncertaintyTriggered => Some(30.0),
                SensingArm::ValueOfInformation => Some(20.0),
                SensingArm::PlaceboCost => Some(50.0),
            },
            sensing_latency_ms: match arm {
                SensingArm::NoExtraSensing => Some(0.0),
                SensingArm::FixedRate => Some(500.0),
                SensingArm::UncertaintyTriggered => Some(300.0),
                SensingArm::ValueOfInformation => Some(200.0),
                SensingArm::PlaceboCost => Some(500.0),
            },
            total_energy_j: Some(250.0),
            elapsed_ms: Some(1_000.0),
            predicted_success_probability: Some(match arm {
                SensingArm::ValueOfInformation => 0.9,
                SensingArm::UncertaintyTriggered => 0.85,
                SensingArm::FixedRate => 0.8,
                SensingArm::NoExtraSensing => 0.75,
                SensingArm::PlaceboCost => 0.75,
            }),
        }
    }

    fn matrix() -> Vec<SensingTrial> {
        let mut rows = Vec::new();
        for scenario_index in 0..6 {
            for arm in SensingArm::ALL {
                rows.push(trial(scenario_index, arm));
            }
        }
        rows
    }

    #[test]
    fn matched_matrix_reports_all_arms_and_family_cluster_intervals() {
        let audit = audit_sensing_trials(&matrix(), 1_000, 42).unwrap();
        assert!(audit.passed);
        assert_eq!(audit.trial_count, 30);
        assert_eq!(audit.scenario_count, 6);
        assert_eq!(audit.scenario_family_count, 6);
        assert_eq!(audit.arm_summaries.len(), 5);
        assert_eq!(audit.comparisons_to_fixed_rate.len(), 4);
        let voi = audit
            .comparisons_to_fixed_rate
            .iter()
            .find(|row| row.arm == SensingArm::ValueOfInformation)
            .unwrap();
        assert_eq!(voi.scenario_family_count, 6);
        assert!(voi.bootstrap_95_ci_lower.is_some());
        assert!(voi.mean_sensing_energy_delta_j.unwrap() < 0.0);
    }

    #[test]
    fn missing_arm_fails_closed_instead_of_dropping_unpaired_case() {
        let mut rows = matrix();
        rows.pop();
        let errors = audit_sensing_trials(&rows, 1_000, 42).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("missing arm")));
    }

    #[test]
    fn safety_violation_fails_qualification_but_remains_in_report() {
        let mut rows = matrix();
        rows[0].safety_violation = true;
        let audit = audit_sensing_trials(&rows, 1_000, 42).unwrap();
        assert!(!audit.passed);
        assert_eq!(audit.safety_violation_count, 1);
        assert_eq!(audit.trial_count, 30);
        assert!(!audit.findings.is_empty());
    }

    #[test]
    fn missing_sensing_energy_is_not_imputed_as_zero() {
        let mut rows = matrix();
        for row in &mut rows {
            if row.arm == SensingArm::NoExtraSensing {
                row.sensing_energy_j = None;
            }
        }
        let audit = audit_sensing_trials(&rows, 1_000, 42).unwrap();
        let row = audit.arm_summaries
            .iter()
            .find(|summary| summary.arm == SensingArm::NoExtraSensing)
            .unwrap();
        assert_eq!(row.sensing_energy_measured_count, 0);
        assert_eq!(row.sensing_energy_missing_count, 6);
        assert_eq!(row.mean_sensing_energy_j, None);
    }

    #[test]
    fn scenario_order_does_not_change_metrics_or_bootstrap_intervals() {
        let rows = matrix();
        let first = audit_sensing_trials(&rows, 1_000, 42).unwrap();
        let mut reversed = rows;
        reversed.reverse();
        let second = audit_sensing_trials(&reversed, 1_000, 42).unwrap();
        assert_eq!(first.arm_summaries, second.arm_summaries);
        assert_eq!(first.comparisons_to_fixed_rate, second.comparisons_to_fixed_rate);
    }

    #[test]
    fn rejects_inconsistent_progress_and_completion() {
        let mut rows = matrix();
        rows[0].completed = true;
        assert!(audit_sensing_trials(&rows, 1_000, 42).is_err());
    }

    #[test]
    fn rejects_acquisition_count_above_request_count() {
        let mut rows = matrix();
        rows[0].sensor_acquisitions = 1;
        rows[0].sensor_requests = 0;
        assert!(audit_sensing_trials(&rows, 1_000, 42).is_err());
    }

    #[test]
    fn missing_forecasts_remain_missing_in_brier_summary() {
        let mut rows = matrix();
        for row in &mut rows {
            if row.arm == SensingArm::PlaceboCost {
                row.predicted_success_probability = None;
            }
        }
        let audit = audit_sensing_trials(&rows, 1_000, 42).unwrap();
        let placebo = audit.arm_summaries
            .iter()
            .find(|summary| summary.arm == SensingArm::PlaceboCost)
            .unwrap();
        assert_eq!(placebo.forecast_count, 0);
        assert_eq!(placebo.brier_score, None);
    }
}
