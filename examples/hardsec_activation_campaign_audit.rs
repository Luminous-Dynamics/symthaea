//! HARDSEC-009 empirical trigger and payload activation campaign auditor.
//!
//! Reports cumulative coverage curves from a pinned JSON campaign log. Trigger
//! activation and payload observation are separate outcomes. Confidence intervals
//! are emitted only for caller-declared IID random/constrained-random campaigns
//! with complete outcomes; the tool cannot prove independence.
//!
//! Example:
//!   cargo run --example hardsec_activation_campaign_audit --features hardsec-activation-audit -- \
//!     --campaign campaign.json --receipt activation-coverage.json
//!
//! This is not a detector, simulator, formal proof engine, or security verdict.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const CLAIM_SCOPE: &str =
    "campaign_log_consistency_and_conditional_empirical_coverage_only";
const Z_95: f64 = 1.959_963_984_540_054;
const DEFAULT_CONFIDENCE: f64 = 0.95;

#[derive(Debug, Clone)]
struct Args {
    campaign: PathBuf,
    receipt: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StimulusMode {
    RandomSimulation,
    ConstrainedRandom,
    DirectedSimulation,
    RecordedReplay,
    Atpg,
    FormalCover,
}

impl StimulusMode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "random_simulation" => Ok(Self::RandomSimulation),
            "constrained_random" => Ok(Self::ConstrainedRandom),
            "directed_simulation" => Ok(Self::DirectedSimulation),
            "recorded_replay" => Ok(Self::RecordedReplay),
            "atpg" => Ok(Self::Atpg),
            "formal_cover" => Ok(Self::FormalCover),
            _ => Err(format!("unsupported stimulus_mode: {value}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::RandomSimulation => "random_simulation",
            Self::ConstrainedRandom => "constrained_random",
            Self::DirectedSimulation => "directed_simulation",
            Self::RecordedReplay => "recorded_replay",
            Self::Atpg => "atpg",
            Self::FormalCover => "formal_cover",
        }
    }

    fn supports_iid_intervals(self) -> bool {
        matches!(self, Self::RandomSimulation | Self::ConstrainedRandom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SamplingModel {
    Iid,
    Clustered,
    DeterministicEnumeration,
    Unknown,
}

impl SamplingModel {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "iid" => Ok(Self::Iid),
            "clustered" => Ok(Self::Clustered),
            "deterministic_enumeration" => Ok(Self::DeterministicEnumeration),
            "unknown" => Ok(Self::Unknown),
            _ => Err(format!("unsupported sampling_model: {value}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Iid => "iid",
            Self::Clustered => "clustered",
            Self::DeterministicEnumeration => "deterministic_enumeration",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Condition {
    Active,
    InactiveControl,
}

impl Condition {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "active" => Ok(Self::Active),
            "inactive_control" => Ok(Self::InactiveControl),
            _ => Err(format!("unsupported trial condition: {value}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::InactiveControl => "inactive_control",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Observation {
    Observed,
    NotObserved,
    Unknown,
}

impl Observation {
    fn parse(value: &str, field: &str) -> Result<Self, String> {
        match value {
            "observed" => Ok(Self::Observed),
            "not_observed" => Ok(Self::NotObserved),
            "unknown" => Ok(Self::Unknown),
            _ => Err(format!("{field} must be observed, not_observed, or unknown")),
        }
    }
}

#[derive(Debug, Clone)]
struct Trial {
    trial_id: String,
    trial_index: u64,
    condition: Condition,
    sequence_sha256: String,
    trigger: Observation,
    trigger_cycle: Option<u64>,
    trigger_witness_sha256: Option<String>,
    payload: Observation,
    payload_cycle: Option<u64>,
    payload_witness_sha256: Option<String>,
    outcome_trace_sha256: Option<String>,
    formal_result: Option<String>,
    formal_bound_cycles: Option<u64>,
}

#[derive(Debug, Clone)]
struct Campaign {
    campaign_id: String,
    dataset_id: String,
    dataset_role: String,
    design_id: String,
    netlist_sha256: String,
    reset_model_sha256: String,
    reset_description: String,
    stimulus_mode: StimulusMode,
    sampling_model: SamplingModel,
    independence_rationale: Option<String>,
    tool: String,
    tool_version: String,
    configuration_sha256: String,
    distribution_sha256: Option<String>,
    window_cycles: u64,
    budget_points: Vec<u64>,
    observation_boundary_id: String,
    observation_boundary_kind: String,
    trials: Vec<Trial>,
    iid_eligible: bool,
    unique_sequence_count: usize,
    duplicate_sequence_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct Interval {
    lower: f64,
    upper: f64,
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut campaign = None;
    let mut receipt = None;
    let mut args = env::args().skip(1);

    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            return Ok(None);
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--campaign" => campaign = Some(PathBuf::from(value)),
            "--receipt" => receipt = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }

    Ok(Some(Args {
        campaign: campaign.ok_or("--campaign is required")?,
        receipt: receipt.ok_or("--receipt is required")?,
    }))
}

fn required_string<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    let result = value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{field} must be a string"))?;
    if result.trim().is_empty() || result.trim() != result {
        return Err(format!("{field} must be non-empty and have no surrounding whitespace"));
    }
    Ok(result)
}

fn optional_string(value: &Value, field: &str) -> Result<Option<String>, String> {
    match value.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(result)) if !result.trim().is_empty() && result.trim() == result => {
            Ok(Some(result.clone()))
        }
        _ => Err(format!("{field} must be a non-empty string when present")),
    }
}

fn required_u64(value: &Value, field: &str) -> Result<u64, String> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{field} must be a positive integer"))
}

fn optional_u64(value: &Value, field: &str) -> Result<Option<u64>, String> {
    match value.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(item) => item
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("{field} must be a non-negative integer when present")),
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn required_digest(value: &Value, field: &str) -> Result<String, String> {
    let digest = required_string(value, field)?.to_ascii_lowercase();
    if !valid_sha256(&digest) {
        return Err(format!("{field} must be a 64-character SHA-256 digest"));
    }
    Ok(digest)
}

fn optional_digest(value: &Value, field: &str) -> Result<Option<String>, String> {
    let digest = optional_string(value, field)?.map(|item| item.to_ascii_lowercase());
    if digest.as_deref().is_some_and(|item| !valid_sha256(item)) {
        return Err(format!("{field} must be a SHA-256 digest when present"));
    }
    Ok(digest)
}

fn json_hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn wilson_interval(successes: u64, trials: u64) -> Option<Interval> {
    if trials == 0 || successes > trials {
        return None;
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = Z_95 * Z_95;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half = Z_95
        * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt())
        / denominator;
    Some(Interval {
        lower: (center - half).max(0.0),
        upper: (center + half).min(1.0),
    })
}

fn zero_event_upper_bound(successes: u64, trials: u64, confidence: f64) -> Option<f64> {
    if trials == 0 || successes != 0 || !confidence.is_finite() || !(0.0..1.0).contains(&confidence)
    {
        return None;
    }
    let alpha = 1.0 - confidence;
    Some(-((alpha.ln() / trials as f64).exp_m1()))
}

fn parse_campaign(value: &Value) -> Result<Campaign, String> {
    if required_string(value, "schema_version")? != "hardsec-activation-campaign-v1" {
        return Err("schema_version must be hardsec-activation-campaign-v1".to_owned());
    }

    let campaign_id = required_string(value, "campaign_id")?.to_owned();
    let dataset_id = required_string(value, "dataset_id")?.to_owned();
    let dataset_role = required_string(value, "dataset_role")?.to_owned();
    if ![
        "evaluation_corpus_member",
        "reference_fixture_only",
        "exploratory_only",
    ]
    .contains(&dataset_role.as_str())
    {
        return Err(format!("unsupported dataset_role: {dataset_role}"));
    }

    let subject = value.get("subject").ok_or("subject is required")?;
    let design_id = required_string(subject, "design_id")?.to_owned();
    let netlist_sha256 = required_digest(subject, "netlist_sha256")?;

    let reset = value.get("reset_model").ok_or("reset_model is required")?;
    let reset_model_sha256 = required_digest(reset, "sha256")?;
    let reset_description = required_string(reset, "description")?.to_owned();

    let stimulus = value.get("stimulus").ok_or("stimulus is required")?;
    let stimulus_mode = StimulusMode::parse(required_string(stimulus, "mode")?)?;
    let sampling_model = SamplingModel::parse(required_string(stimulus, "sampling_model")?)?;
    let independence_rationale = optional_string(stimulus, "independence_rationale")?;
    if sampling_model == SamplingModel::Iid {
        if !stimulus_mode.supports_iid_intervals() {
            return Err("IID sampling intervals are supported only for random_simulation or constrained_random".to_owned());
        }
        if independence_rationale.is_none() {
            return Err("IID sampling_model requires independence_rationale".to_owned());
        }
    }
    if stimulus_mode == StimulusMode::FormalCover && sampling_model == SamplingModel::Iid {
        return Err("formal_cover cannot be treated as IID sampling".to_owned());
    }

    let tool = required_string(stimulus, "tool")?.to_owned();
    let tool_version = required_string(stimulus, "tool_version")?.to_owned();
    let configuration_sha256 = required_digest(stimulus, "configuration_sha256")?;
    let distribution_sha256 = optional_digest(stimulus, "distribution_sha256")?;
    if matches!(
        stimulus_mode,
        StimulusMode::RandomSimulation | StimulusMode::ConstrainedRandom
    ) && distribution_sha256.is_none()
    {
        return Err("random/constrained-random stimulus requires distribution_sha256".to_owned());
    }

    let window_cycles = required_u64(stimulus, "window_cycles")?;
    if window_cycles == 0 {
        return Err("window_cycles must be greater than zero".to_owned());
    }

    let budget_values = value
        .get("budget_points")
        .and_then(Value::as_array)
        .ok_or("budget_points must be an array")?;
    if budget_values.is_empty() {
        return Err("budget_points must not be empty".to_owned());
    }
    let mut budget_points = Vec::with_capacity(budget_values.len());
    for item in budget_values {
        let point = item
            .as_u64()
            .ok_or("budget_points entries must be positive integers")?;
        if point == 0 {
            return Err("budget_points entries must be positive".to_owned());
        }
        if budget_points.last().is_some_and(|previous| *previous >= point) {
            return Err("budget_points must be strictly increasing".to_owned());
        }
        budget_points.push(point);
    }

    let boundary = value
        .get("observation_boundary")
        .ok_or("observation_boundary is required")?;
    let observation_boundary_id = required_string(boundary, "id")?.to_owned();
    let observation_boundary_kind = required_string(boundary, "kind")?.to_owned();
    if !["primary_output", "state_boundary", "internal_signal"].contains(&observation_boundary_kind.as_str()) {
        return Err(format!("unsupported observation boundary kind: {observation_boundary_kind}"));
    }
    let _boundary_description = required_string(boundary, "description")?;

    let trial_values = value
        .get("trials")
        .and_then(Value::as_array)
        .ok_or("trials must be an array")?;
    if trial_values.is_empty() {
        return Err("trials must not be empty".to_owned());
    }

    let mut trials = Vec::with_capacity(trial_values.len());
    let mut trial_ids = BTreeSet::new();
    let mut next_index: BTreeMap<Condition, u64> = BTreeMap::new();
    let mut sequence_digests = BTreeSet::new();

    for value in trial_values {
        let trial_id = required_string(value, "trial_id")?.to_owned();
        if !trial_ids.insert(trial_id.clone()) {
            return Err(format!("duplicate trial_id: {trial_id}"));
        }
        let trial_index = required_u64(value, "trial_index")?;
        let condition = Condition::parse(required_string(value, "condition")?)?;
        let expected = next_index.entry(condition).or_insert(1);
        if trial_index != *expected {
            return Err(format!(
                "{} trial_index must be contiguous and ordered; expected {}, got {}",
                condition.as_str(),
                *expected,
                trial_index
            ));
        }
        *expected = expected
            .checked_add(1)
            .ok_or("trial_index overflow")?;

        let sequence_sha256 = required_digest(value, "sequence_sha256")?;
        sequence_digests.insert(sequence_sha256.clone());
        let trigger = Observation::parse(
            required_string(value, "trigger_status")?,
            "trigger_status",
        )?;
        let trigger_cycle = optional_u64(value, "trigger_cycle")?;
        let trigger_witness_sha256 = optional_digest(value, "trigger_witness_sha256")?;
        let payload = Observation::parse(
            required_string(value, "payload_status")?,
            "payload_status",
        )?;
        let payload_cycle = optional_u64(value, "payload_cycle")?;
        let payload_witness_sha256 = optional_digest(value, "payload_witness_sha256")?;
        let outcome_trace_sha256 = optional_digest(value, "outcome_trace_sha256")?;
        let formal_result = optional_string(value, "formal_result")?;
        let formal_bound_cycles = optional_u64(value, "formal_bound_cycles")?;

        match trigger {
            Observation::Observed => {
                let cycle = trigger_cycle.ok_or_else(|| {
                    format!("trial {trial_id}: observed trigger requires trigger_cycle")
                })?;
                if cycle == 0 || cycle > window_cycles {
                    return Err(format!(
                        "trial {trial_id}: trigger_cycle must be within 1..={window_cycles}"
                    ));
                }
                if outcome_trace_sha256.is_none() {
                    return Err(format!(
                        "trial {trial_id}: observed trigger requires outcome_trace_sha256"
                    ));
                }
            }
            Observation::NotObserved | Observation::Unknown => {
                if trigger_witness_sha256.is_some() {
                    return Err(format!(
                        "trial {trial_id}: non-observed trigger must not include trigger_witness_sha256"
                    ));
                }
                if trigger_cycle.is_some() {
                    return Err(format!(
                        "trial {trial_id}: non-observed trigger must not include trigger_cycle"
                    ));
                }
            }
        }

        match payload {
            Observation::Observed => {
                let cycle = payload_cycle.ok_or_else(|| {
                    format!("trial {trial_id}: observed payload requires payload_cycle")
                })?;
                if trigger != Observation::Observed {
                    return Err(format!(
                        "trial {trial_id}: payload cannot be observed without trigger activation"
                    ));
                }
                if cycle == 0 || cycle > window_cycles {
                    return Err(format!(
                        "trial {trial_id}: payload_cycle must be within 1..={window_cycles}"
                    ));
                }
                if cycle < trigger_cycle.unwrap_or(u64::MAX) {
                    return Err(format!(
                        "trial {trial_id}: payload_cycle cannot precede trigger_cycle"
                    ));
                }
                if payload_witness_sha256.is_none() || outcome_trace_sha256.is_none() {
                    return Err(format!(
                        "trial {trial_id}: observed payload requires payload_witness_sha256 and outcome_trace_sha256"
                    ));
                }
            }
            Observation::NotObserved | Observation::Unknown => {
                if payload_cycle.is_some() || payload_witness_sha256.is_some() {
                    return Err(format!(
                        "trial {trial_id}: non-observed payload must not include payload_cycle or payload_witness_sha256"
                    ));
                }
            }
        }

        if stimulus_mode == StimulusMode::FormalCover {
            let status = formal_result.as_deref().ok_or_else(|| {
                format!("trial {trial_id}: formal_cover requires formal_result")
            })?;
            let bound = formal_bound_cycles.ok_or_else(|| {
                format!("trial {trial_id}: formal_cover requires formal_bound_cycles")
            })?;
            if bound == 0 || bound > window_cycles {
                return Err(format!(
                    "trial {trial_id}: formal_bound_cycles must be within 1..={window_cycles}"
                ));
            }
            match status {
                "cover_reached" => {
                    if trigger != Observation::Observed || trigger_witness_sha256.is_none() {
                        return Err(format!(
                            "trial {trial_id}: cover_reached requires observed trigger and trigger_witness_sha256"
                        ));
                    }
                }
                "bounded_no_counterexample" => {
                    if trigger != Observation::NotObserved {
                        return Err(format!(
                            "trial {trial_id}: bounded_no_counterexample requires trigger_status=not_observed"
                        ));
                    }
                }
                "timeout" | "unknown" => {
                    if trigger != Observation::Unknown {
                        return Err(format!(
                            "trial {trial_id}: {status} requires trigger_status=unknown"
                        ));
                    }
                }
                _ => {
                    return Err(format!(
                        "trial {trial_id}: unsupported formal_result {status}"
                    ));
                }
            }
        } else if formal_result.is_some() || formal_bound_cycles.is_some() {
            return Err(format!(
                "trial {trial_id}: formal fields are valid only for formal_cover"
            ));
        }

        trials.push(Trial {
            trial_id,
            trial_index,
            condition,
            sequence_sha256,
            trigger,
            trigger_cycle,
            trigger_witness_sha256,
            payload,
            payload_cycle,
            payload_witness_sha256,
            outcome_trace_sha256,
            formal_result,
            formal_bound_cycles,
        });
    }

    for point in &budget_points {
        for condition in [Condition::Active, Condition::InactiveControl] {
            let count = next_index.get(&condition).copied().unwrap_or(1) - 1;
            if count > 0 && *point > count {
                return Err(format!(
                    "budget point {point} exceeds {} trial count {count}",
                    condition.as_str()
                ));
            }
        }
    }

    let active_count = next_index.get(&Condition::Active).copied().unwrap_or(1) - 1;
    let inactive_count = next_index
        .get(&Condition::InactiveControl)
        .copied()
        .unwrap_or(1)
        - 1;
    if inactive_count > 0 && active_count != inactive_count {
        return Err("active and inactive_control conditions must have equal trial counts".to_owned());
    }
    if inactive_count > 0 && !budget_points.iter().all(|point| *point <= inactive_count) {
        return Err("budget_points must be available in both active and inactive control trials".to_owned());
    }
    if !budget_points.iter().any(|point| *point == active_count) {
        return Err("budget_points must include the full active trial count".to_owned());
    }

    let iid_eligible = sampling_model == SamplingModel::Iid && stimulus_mode.supports_iid_intervals();
    if !iid_eligible && sampling_model == SamplingModel::Iid {
        return Err("sampling_model=iid is incompatible with this stimulus mode".to_owned());
    }
    let unique_sequence_count = sequence_digests.len();
    let duplicate_sequence_count = trials.len().saturating_sub(unique_sequence_count);

    Ok(Campaign {
        campaign_id,
        dataset_id,
        dataset_role,
        design_id,
        netlist_sha256,
        reset_model_sha256,
        reset_description,
        stimulus_mode,
        sampling_model,
        independence_rationale,
        tool,
        tool_version,
        configuration_sha256,
        distribution_sha256,
        window_cycles,
        budget_points,
        observation_boundary_id,
        observation_boundary_kind,
        trials,
        iid_eligible,
        unique_sequence_count,
        duplicate_sequence_count,
    })
}

fn rate(numerator: u64, denominator: u64) -> Option<f64> {
    if denominator == 0 || numerator > denominator {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

fn interval_json(successes: u64, total: u64, enabled: bool) -> Value {
    if !enabled || total == 0 {
        return Value::Null;
    }
    match wilson_interval(successes, total) {
        Some(interval) => json!({
            "method": "wilson_95_two_sided",
            "lower": interval.lower,
            "upper": interval.upper
        }),
        None => Value::Null,
    }
}

fn count_observed(trials: &[&Trial], getter: impl Fn(&Trial) -> Observation) -> (u64, u64, u64) {
    let mut observed = 0;
    let mut known = 0;
    let mut unknown = 0;
    for trial in trials {
        match getter(trial) {
            Observation::Observed => {
                observed += 1;
                known += 1;
            }
            Observation::NotObserved => known += 1,
            Observation::Unknown => unknown += 1,
        }
    }
    (observed, known, unknown)
}

fn cycle_histogram(trials: &[&Trial], getter: impl Fn(&Trial) -> Option<u64>) -> BTreeMap<u64, u64> {
    let mut counts = BTreeMap::new();
    for trial in trials {
        if let Some(cycle) = getter(trial) {
            *counts.entry(cycle).or_insert(0) += 1;
        }
    }
    counts
}

fn summarize_condition(campaign: &Campaign, condition: Condition, budget: u64) -> Value {
    let prefix = campaign
        .trials
        .iter()
        .filter(|trial| trial.condition == condition && trial.trial_index <= budget)
        .collect::<Vec<_>>();
    let (triggered, trigger_known, trigger_unknown) =
        count_observed(&prefix, |trial| trial.trigger);
    let (payload_effects, payload_known_total, payload_unknown) =
        count_observed(&prefix, |trial| trial.payload);
    let payload_known_given_trigger = prefix
        .iter()
        .filter(|trial| trial.trigger == Observation::Observed && trial.payload != Observation::Unknown)
        .count() as u64;
    let payload_effects_given_trigger = prefix
        .iter()
        .filter(|trial| {
            trial.trigger == Observation::Observed && trial.payload == Observation::Observed
        })
        .count() as u64;
    let payload_unknown_given_trigger = prefix
        .iter()
        .filter(|trial| {
            trial.trigger == Observation::Observed && trial.payload == Observation::Unknown
        })
        .count() as u64;
    let activation_count = triggered;
    let payload_effect_count = payload_effects;
    let trigger_fraction = rate(triggered, trigger_known);
    let payload_fraction = rate(payload_effects, payload_known_total);
    let conditional_payload_fraction =
        rate(payload_effects_given_trigger, payload_known_given_trigger);
    let all_trigger_outcomes_known = trigger_unknown == 0 && trigger_known == prefix.len() as u64;
    let all_payload_outcomes_known = payload_unknown == 0 && payload_known_total == prefix.len() as u64;
    let all_conditional_effect_outcomes_known = payload_unknown_given_trigger == 0;

    let zero_activation_upper = if campaign.iid_eligible
        && all_trigger_outcomes_known
        && triggered == 0
    {
        zero_event_upper_bound(0, trigger_known, DEFAULT_CONFIDENCE)
    } else {
        None
    };

    let trigger_cycle_counts = cycle_histogram(&prefix, |trial| trial.trigger_cycle);
    let payload_cycle_counts = cycle_histogram(&prefix, |trial| trial.payload_cycle);
    let trigger_to_payload_cycles = prefix
        .iter()
        .filter_map(|trial| Some(trial.payload_cycle? - trial.trigger_cycle?))
        .collect::<Vec<_>>();
    let mean_trigger_to_payload_cycles = if trigger_to_payload_cycles.is_empty() {
        None
    } else {
        Some(
            trigger_to_payload_cycles
                .iter()
                .map(|value| *value as f64)
                .sum::<f64>()
                / trigger_to_payload_cycles.len() as f64,
        )
    };

    json!({
        "condition": condition.as_str(),
        "budget_trials": prefix.len(),
        "unique_sequence_count": prefix.iter().map(|trial| &trial.sequence_sha256).collect::<BTreeSet<_>>().len(),
        "trigger": {
            "activated": activation_count,
            "known_outcomes": trigger_known,
            "unknown_outcomes": trigger_unknown,
            "empirical_rate": trigger_fraction,
            "wilson_interval": interval_json(triggered, trigger_known, campaign.iid_eligible && all_trigger_outcomes_known),
            "zero_event_one_sided_95_upper_bound": zero_activation_upper,
            "activation_cycle_histogram": trigger_cycle_counts
        },
        "payload_effect": {
            "observed": payload_effect_count,
            "known_outcomes": payload_known_total,
            "unknown_outcomes": payload_unknown,
            "empirical_rate_all_trials": payload_fraction,
            "wilson_interval_all_trials": interval_json(payload_effects, payload_known_total, campaign.iid_eligible && all_payload_outcomes_known),
            "observed_given_trigger": payload_effects_given_trigger,
            "known_given_trigger": payload_known_given_trigger,
            "unknown_given_trigger": payload_unknown_given_trigger,
            "empirical_rate_given_trigger": conditional_payload_fraction,
            "wilson_interval_given_trigger": interval_json(payload_effects_given_trigger, payload_known_given_trigger, campaign.iid_eligible && all_conditional_effect_outcomes_known),
            "payload_cycle_histogram": payload_cycle_counts,
            "mean_trigger_to_payload_cycles": mean_trigger_to_payload_cycles
        }
    })
}

fn summarize(campaign: &Campaign, campaign_sha256: String) -> Value {
    let mut budget_curves = Vec::new();
    for budget in &campaign.budget_points {
        let mut condition_summaries = Vec::new();
        condition_summaries.push(summarize_condition(campaign, Condition::Active, *budget));
        if campaign
            .trials
            .iter()
            .any(|trial| trial.condition == Condition::InactiveControl)
        {
            condition_summaries.push(summarize_condition(
                campaign,
                Condition::InactiveControl,
                *budget,
            ));
        }
        budget_curves.push(json!({
            "cumulative_budget_per_condition": budget,
            "conditions": condition_summaries
        }));
    }

    let active = campaign
        .trials
        .iter()
        .filter(|trial| trial.condition == Condition::Active)
        .collect::<Vec<_>>();
    let inactive = campaign
        .trials
        .iter()
        .filter(|trial| trial.condition == Condition::InactiveControl)
        .collect::<Vec<_>>();
    let paired_control = if inactive.is_empty() {
        Value::Null
    } else {
        let mut trigger_discordant = 0;
        let mut payload_discordant = 0;
        let active_by_index = active
            .iter()
            .map(|trial| (trial.trial_index, *trial))
            .collect::<BTreeMap<_, _>>();
        for control in &inactive {
            if let Some(reference) = active_by_index.get(&control.trial_index) {
                if reference.trigger != control.trigger {
                    trigger_discordant += 1;
                }
                if reference.payload != control.payload {
                    payload_discordant += 1;
                }
            }
        }
        json!({
            "paired_trial_count": inactive.len(),
            "trigger_status_discordant_pairs": trigger_discordant,
            "payload_status_discordant_pairs": payload_discordant,
            "interpretation": "descriptive_paired_status_comparison_only_not_a_causal_effect_estimate"
        })
    };

    json!({
        "schema_version": "hardsec-activation-campaign-audit-receipt-v1",
        "result": "pass",
        "claim_scope": CLAIM_SCOPE,
        "campaign_sha256": campaign_sha256,
        "campaign_id": campaign.campaign_id,
        "dataset": {
            "dataset_id": campaign.dataset_id,
            "dataset_role": campaign.dataset_role
        },
        "subject": {
            "design_id": campaign.design_id,
            "netlist_sha256": campaign.netlist_sha256,
            "reset_model_sha256": campaign.reset_model_sha256,
            "reset_description": campaign.reset_description
        },
        "stimulus": {
            "mode": campaign.stimulus_mode.as_str(),
            "sampling_model": campaign.sampling_model.as_str(),
            "independence_rationale": campaign.independence_rationale,
            "independence_status": if campaign.iid_eligible { "caller_declared_not_verified" } else { "not_assumed" },
            "tool": campaign.tool,
            "tool_version": campaign.tool_version,
            "configuration_sha256": campaign.configuration_sha256,
            "distribution_sha256": campaign.distribution_sha256,
            "window_cycles": campaign.window_cycles
        },
        "observation_boundary": {
            "id": campaign.observation_boundary_id,
            "kind": campaign.observation_boundary_kind
        },
        "trial_count": campaign.trials.len(),
        "trials": campaign.trials.iter().map(|trial| json!({
            "trial_id": trial.trial_id,
            "trial_index": trial.trial_index,
            "condition": trial.condition.as_str(),
            "sequence_sha256": trial.sequence_sha256,
            "trigger_status": match trial.trigger {
                Observation::Observed => "observed",
                Observation::NotObserved => "not_observed",
                Observation::Unknown => "unknown",
            },
            "trigger_cycle": trial.trigger_cycle,
            "trigger_witness_sha256": trial.trigger_witness_sha256,
            "payload_status": match trial.payload {
                Observation::Observed => "observed",
                Observation::NotObserved => "not_observed",
                Observation::Unknown => "unknown",
            },
            "payload_cycle": trial.payload_cycle,
            "payload_witness_sha256": trial.payload_witness_sha256,
            "outcome_trace_sha256": trial.outcome_trace_sha256,
            "formal_result": trial.formal_result,
            "formal_bound_cycles": trial.formal_bound_cycles
        })).collect::<Vec<_>>(),
        "unique_sequence_count": campaign.unique_sequence_count,
        "duplicate_sequence_count": campaign.duplicate_sequence_count,
        "budget_curve": budget_curves,
        "paired_inactive_control": paired_control,
        "limitations": [
            "The audit checks internal manifest consistency, not simulator correctness or ground-truth labels.",
            "IID independence is a caller declaration and is not proven by this tool.",
            "Rates are conditional on the supplied design, reset model, stimulus distribution, window length, toolchain, and observation boundary.",
            "No finite no-activation result proves trigger unreachability.",
            "Trigger activation and payload observation are reported as separate outcomes.",
            "The receipt is unsigned and is not a detector or hardware-security verdict."
        ],
        "signature": null,
        "security_verdict": null
    })
}

fn receipt_path_outside_source(source: &Path, receipt: &Path) -> Result<PathBuf, String> {
    let source = fs::canonicalize(source)
        .map_err(|error| format!("canonicalize campaign {}: {error}", source.display()))?;
    let source_parent = source.parent().ok_or("campaign has no parent directory")?;
    let parent = receipt
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let filename = receipt
        .file_name()
        .ok_or("receipt must include a filename")?;
    let parent = fs::canonicalize(parent)
        .map_err(|error| format!("canonicalize receipt parent {}: {error}", parent.display()))?;
    let target = parent.join(filename);
    if target.starts_with(source_parent) {
        return Err("receipt must be outside the campaign directory".to_owned());
    }
    Ok(target)
}

fn run(args: Args) -> Result<(), String> {
    let metadata = fs::symlink_metadata(&args.campaign)
        .map_err(|error| format!("stat campaign {}: {error}", args.campaign.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("campaign input must be a regular non-symlink file".to_owned());
    }
    let campaign_bytes = fs::read(&args.campaign)
        .map_err(|error| format!("read campaign {}: {error}", args.campaign.display()))?;
    let before = fs::metadata(&args.campaign)
        .map_err(|error| format!("stat campaign before hashing: {error}"))?;
    let campaign_sha256 = json_hash(&campaign_bytes);
    let after = fs::metadata(&args.campaign)
        .map_err(|error| format!("stat campaign after hashing: {error}"))?;
    if before.len() != after.len() || before.modified().ok() != after.modified().ok() {
        return Err("campaign changed while being read".to_owned());
    }

    let value: Value = serde_json::from_slice(&campaign_bytes)
        .map_err(|error| format!("parse campaign JSON: {error}"))?;
    let campaign = parse_campaign(&value)?;
    let receipt_path = receipt_path_outside_source(&args.campaign, &args.receipt)?;
    let receipt = summarize(&campaign, campaign_sha256);
    let bytes = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| format!("serialize activation audit receipt: {error}"))?;

    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&receipt_path)
        .map_err(|error| format!("create new receipt {}: {error}", receipt_path.display()))?;
    let write_result = output
        .write_all(&bytes)
        .and_then(|_| output.write_all(b"\n"))
        .and_then(|_| output.sync_all());
    drop(output);
    if let Err(error) = write_result {
        let _ = fs::remove_file(&receipt_path);
        return Err(format!("write receipt {}: {error}", receipt_path.display()));
    }

    println!("HARDSEC-009 activation campaign audit: PASS");
    println!("campaign: {}", campaign.campaign_id);
    println!("trials: {}", campaign.trials.len());
    println!("budget points: {}", campaign.budget_points.len());
    println!("receipt: {}", receipt_path.display());
    println!("claim scope: {}", CLAIM_SCOPE);
    Ok(())
}

fn main() -> ExitCode {
    match parse_args() {
        Ok(None) => {
            println!("usage: hardsec_activation_campaign_audit --campaign FILE --receipt NEW_FILE");
            ExitCode::SUCCESS
        }
        Ok(Some(args)) => match run(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("HARDSEC-009 activation campaign audit failed: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT: AtomicUsize = AtomicUsize::new(0);
    const NETLIST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const RESET: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const CONFIG: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const DIST: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const SEQUENCE: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const TRACE: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    const WITNESS: &str = "1111111111111111111111111111111111111111111111111111111111111111";

    fn trial(id: &str, index: u64, condition: &str, trigger: &str, payload: &str) -> Value {
        let mut item = json!({
            "trial_id": id,
            "trial_index": index,
            "condition": condition,
            "sequence_sha256": format!(
                "{:064x}",
                if condition == "active" {
                    index
                } else {
                    index + 1000
                }
            ),
            "trigger_status": trigger,
            "payload_status": payload
        });
        if trigger == "observed" {
            item["trigger_cycle"] = json!(2);
            item["outcome_trace_sha256"] = json!(TRACE);
        }
        if payload == "observed" {
            item["payload_cycle"] = json!(4);
            item["payload_witness_sha256"] = json!(WITNESS);
        }
        item
    }

    fn fixture() -> Value {
        let mut active = vec![
            trial("a1", 1, "active", "observed", "observed"),
            trial("a2", 2, "active", "not_observed", "not_observed"),
            trial("a3", 3, "active", "observed", "not_observed"),
        ];
        active[2]["trigger_cycle"] = json!(3);
        let inactive = vec![
            trial("i1", 1, "inactive_control", "not_observed", "not_observed"),
            trial("i2", 2, "inactive_control", "not_observed", "not_observed"),
            trial("i3", 3, "inactive_control", "not_observed", "not_observed"),
        ];
        active.extend(inactive);
        json!({
            "schema_version": "hardsec-activation-campaign-v1",
            "campaign_id": "fixture-campaign",
            "dataset_id": "fixture-dataset",
            "dataset_role": "reference_fixture_only",
            "subject": {
                "design_id": "fixture-design",
                "netlist_sha256": NETLIST
            },
            "reset_model": {
                "sha256": RESET,
                "description": "known reset state"
            },
            "stimulus": {
                "mode": "random_simulation",
                "sampling_model": "iid",
                "independence_rationale": "independent RNG draws from declared distribution",
                "tool": "fixture-simulator",
                "tool_version": "1.0",
                "configuration_sha256": CONFIG,
                "distribution_sha256": DIST,
                "window_cycles": 8
            },
            "budget_points": [1, 2, 3],
            "observation_boundary": {
                "id": "primary-output-y",
                "kind": "primary_output",
                "description": "primary output y"
            },
            "trials": active
        })
    }

    #[test]
    fn reports_distinct_trigger_and_payload_rates() {
        let campaign = parse_campaign(&fixture()).unwrap();
        let receipt = summarize(&campaign, "0".repeat(64));
        let active = &receipt["budget_curve"][2]["conditions"][0];
        assert_eq!(active["trigger"]["activated"], 2);
        assert_eq!(active["trigger"]["known_outcomes"], 3);
        assert_eq!(active["payload_effect"]["observed"], 1);
        assert_eq!(active["payload_effect"]["observed_given_trigger"], 1);
        assert_eq!(active["payload_effect"]["known_given_trigger"], 2);
    }

    #[test]
    fn computes_iid_wilson_intervals_only_when_all_outcomes_are_known() {
        let campaign = parse_campaign(&fixture()).unwrap();
        let receipt = summarize(&campaign, "0".repeat(64));
        let active = &receipt["budget_curve"][2]["conditions"][0];
        assert_eq!(active["trigger"]["wilson_interval"]["method"], "wilson_95_two_sided");
        assert_eq!(active["payload_effect"]["wilson_interval_all_trials"]["method"], "wilson_95_two_sided");
    }

    #[test]
    fn computes_exact_zero_event_upper_bound_for_iid_campaign() {
        let mut value = fixture();
        for trial in value["trials"].as_array_mut().unwrap() {
            trial["trigger_status"] = json!("not_observed");
            trial.as_object_mut().unwrap().remove("trigger_cycle");
            trial.as_object_mut().unwrap().remove("outcome_trace_sha256");
        }
        let campaign = parse_campaign(&value).unwrap();
        let receipt = summarize(&campaign, "0".repeat(64));
        let active = &receipt["budget_curve"][2]["conditions"][0];
        let upper = active["trigger"]["zero_event_one_sided_95_upper_bound"].as_f64().unwrap();
        assert!((upper - 0.6316).abs() < 0.0002);
    }

    #[test]
    fn non_iid_campaign_has_no_probability_interval() {
        let mut value = fixture();
        value["stimulus"]["sampling_model"] = json!("deterministic_enumeration");
        value["stimulus"].as_object_mut().unwrap().remove("independence_rationale");
        let campaign = parse_campaign(&value).unwrap();
        let receipt = summarize(&campaign, "0".repeat(64));
        let active = &receipt["budget_curve"][2]["conditions"][0];
        assert!(active["trigger"]["wilson_interval"].is_null());
        assert!(active["trigger"]["zero_event_one_sided_95_upper_bound"].is_null());
    }

    #[test]
    fn rejects_iid_claim_for_directed_sequences() {
        let mut value = fixture();
        value["stimulus"]["mode"] = json!("directed_simulation");
        assert!(parse_campaign(&value).unwrap_err().contains("IID sampling intervals"));
    }

    #[test]
    fn rejects_iid_campaign_without_rationale() {
        let mut value = fixture();
        value["stimulus"].as_object_mut().unwrap().remove("independence_rationale");
        assert!(parse_campaign(&value).unwrap_err().contains("independence_rationale"));
    }

    #[test]
    fn rejects_payload_effect_without_trigger() {
        let mut value = fixture();
        value["trials"][1]["payload_status"] = json!("observed");
        value["trials"][1]["payload_cycle"] = json!(4);
        value["trials"][1]["payload_witness_sha256"] = json!(WITNESS);
        value["trials"][1]["outcome_trace_sha256"] = json!(TRACE);
        assert!(parse_campaign(&value).unwrap_err().contains("payload cannot be observed without trigger"));
    }

    #[test]
    fn rejects_payload_cycle_before_trigger_cycle() {
        let mut value = fixture();
        value["trials"][0]["payload_cycle"] = json!(1);
        assert!(parse_campaign(&value).unwrap_err().contains("payload_cycle cannot precede trigger_cycle"));
    }

    #[test]
    fn rejects_cycles_outside_window() {
        let mut value = fixture();
        value["trials"][0]["trigger_cycle"] = json!(9);
        assert!(parse_campaign(&value).unwrap_err().contains("trigger_cycle must be within"));
    }

    #[test]
    fn rejects_noncontiguous_trial_indices() {
        let mut value = fixture();
        value["trials"][1]["trial_index"] = json!(4);
        assert!(parse_campaign(&value).unwrap_err().contains("trial_index must be contiguous"));
    }

    #[test]
    fn rejects_duplicate_trial_ids() {
        let mut value = fixture();
        value["trials"][1]["trial_id"] = json!("a1");
        assert!(parse_campaign(&value).unwrap_err().contains("duplicate trial_id"));
    }

    #[test]
    fn rejects_missing_distribution_digest_for_random_campaign() {
        let mut value = fixture();
        value["stimulus"].as_object_mut().unwrap().remove("distribution_sha256");
        assert!(parse_campaign(&value).unwrap_err().contains("requires distribution_sha256"));
    }

    #[test]
    fn rejects_budget_beyond_trial_count() {
        let mut value = fixture();
        value["budget_points"] = json!([1, 2, 4]);
        assert!(parse_campaign(&value).unwrap_err().contains("exceeds"));
    }

    #[test]
    fn rejects_unpaired_active_and_inactive_counts() {
        let mut value = fixture();
        value["trials"].as_array_mut().unwrap().pop();
        assert!(parse_campaign(&value).unwrap_err().contains("equal trial counts"));
    }

    #[test]
    fn rejects_formal_cover_without_formal_status() {
        let mut value = fixture();
        value["stimulus"]["mode"] = json!("formal_cover");
        value["stimulus"]["sampling_model"] = json!("deterministic_enumeration");
        for trial in value["trials"].as_array_mut().unwrap() {
            trial.as_object_mut().unwrap().remove("formal_result");
            trial.as_object_mut().unwrap().remove("formal_bound_cycles");
        }
        assert!(parse_campaign(&value).unwrap_err().contains("formal_cover requires formal_result"));
    }

    #[test]
    fn rejects_bounded_no_counterexample_when_trial_claims_activation() {
        let mut value = fixture();
        value["stimulus"]["mode"] = json!("formal_cover");
        value["stimulus"]["sampling_model"] = json!("deterministic_enumeration");
        value["trials"][0]["formal_result"] = json!("bounded_no_counterexample");
        value["trials"][0]["formal_bound_cycles"] = json!(8);
        assert!(parse_campaign(&value)
            .unwrap_err()
            .contains("bounded_no_counterexample requires trigger_status=not_observed"));
    }

    #[test]
    fn rejects_cover_reached_without_trigger_witness_digest() {
        let mut value = fixture();
        value["stimulus"]["mode"] = json!("formal_cover");
        value["stimulus"]["sampling_model"] = json!("deterministic_enumeration");
        value["trials"][0]["formal_result"] = json!("cover_reached");
        value["trials"][0]["formal_bound_cycles"] = json!(8);
        assert!(parse_campaign(&value)
            .unwrap_err()
            .contains("cover_reached requires observed trigger and trigger_witness_sha256"));
    }

    #[test]
    fn iid_campaign_with_unknown_trigger_outcomes_does_not_receive_interval() {
        let mut value = fixture();
        value["trials"][1]["trigger_status"] = json!("unknown");
        let campaign = parse_campaign(&value).unwrap();
        let receipt = summarize(&campaign, "0".repeat(64));
        let active = &receipt["budget_curve"][2]["conditions"][0];
        assert_eq!(active["trigger"]["unknown_outcomes"], 1);
        assert!(active["trigger"]["wilson_interval"].is_null());
    }

    #[test]
    fn rejects_non_observed_trigger_with_witness_digest() {
        let mut value = fixture();
        value["trials"][1]["trigger_witness_sha256"] = json!(WITNESS);
        assert!(parse_campaign(&value)
            .unwrap_err()
            .contains("non-observed trigger must not include trigger_witness_sha256"));
    }

    #[test]
    fn rejects_observed_trigger_without_output_trace_digest() {
        let mut value = fixture();
        value["trials"][0].as_object_mut().unwrap().remove("outcome_trace_sha256");
        assert!(parse_campaign(&value)
            .unwrap_err()
            .contains("observed trigger requires outcome_trace_sha256"));
    }

    #[test]
    fn rejects_campaign_without_full_budget_point() {
        let mut value = fixture();
        value["budget_points"] = json!([1, 2]);
        assert!(parse_campaign(&value)
            .unwrap_err()
            .contains("include the full active trial count"));
    }

    #[test]
    fn accepts_formal_cover_witness_without_probability_intervals() {
        let mut value = fixture();
        value["stimulus"]["mode"] = json!("formal_cover");
        value["stimulus"]["sampling_model"] = json!("deterministic_enumeration");
        for trial in value["trials"].as_array_mut().unwrap() {
            trial["formal_result"] = json!("bounded_no_counterexample");
            trial["formal_bound_cycles"] = json!(8);
            if trial["trigger_status"] == "observed" {
                trial["formal_result"] = json!("cover_reached");
                trial["trigger_witness_sha256"] = json!(WITNESS);
            }
        }
        let campaign = parse_campaign(&value).unwrap();
        let receipt = summarize(&campaign, "0".repeat(64));
        let active = &receipt["budget_curve"][2]["conditions"][0];
        assert!(active["trigger"]["wilson_interval"].is_null());
    }

    #[test]
    fn zero_event_upper_bound_matches_known_answer_at_one_thousand_trials() {
        let bound = zero_event_upper_bound(0, 1000, 0.95).unwrap();
        assert!((bound - 0.002991).abs() < 0.000002);
    }

    #[test]
    fn wilson_interval_matches_known_answer_for_half_successes() {
        let interval = wilson_interval(50, 100).unwrap();
        assert!((interval.lower - 0.4038).abs() < 0.0002);
        assert!((interval.upper - 0.5962).abs() < 0.0002);
    }

    #[test]
    fn reports_duplicate_sequences_without_deduplicating_trials() {
        let mut value = fixture();
        value["trials"][1]["sequence_sha256"] = value["trials"][0]["sequence_sha256"].clone();
        let campaign = parse_campaign(&value).unwrap();
        assert_eq!(campaign.unique_sequence_count, 5);
        assert_eq!(campaign.duplicate_sequence_count, 1);
        let receipt = summarize(&campaign, "0".repeat(64));
        assert_eq!(receipt["trial_count"], 6);
    }

    #[test]
    fn rejects_unknown_status_fields_and_bad_digests() {
        let mut value = fixture();
        value["trials"][0]["sequence_sha256"] = json!("not-a-hash");
        assert!(parse_campaign(&value).unwrap_err().contains("SHA-256 digest"));
    }

    #[test]
    fn rejects_formal_fields_in_a_simulation_campaign() {
        let mut value = fixture();
        value["trials"][0]["formal_result"] = json!("cover_reached");
        value["trials"][0]["formal_bound_cycles"] = json!(8);
        assert!(parse_campaign(&value).unwrap_err().contains("formal fields are valid only"));
    }

    #[test]
    fn paired_controls_are_reported_without_causal_claims() {
        let campaign = parse_campaign(&fixture()).unwrap();
        let receipt = summarize(&campaign, "0".repeat(64));
        assert_eq!(receipt["paired_inactive_control"]["paired_trial_count"], 3);
        assert!(receipt["paired_inactive_control"]["interpretation"].as_str().unwrap().contains("not_a_causal"));
    }
}
