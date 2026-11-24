//! Group-level estimator for Mycelix DELIB-TEAM-RETENTION-001.
//!
//! This is a thin measurement extension to SympoeticBench, not a team runtime
//! or a second benchmark framework. Each row represents one randomized team
//! after preregistered task-level aggregation. Teams, not messages, turns, or
//! repeated tasks, are the independent resampling units.

use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;

const HEADER: &str = "team_id\ttask_bundle_sha256\tai_configuration_sha256\ttms_configuration_sha256\tarm\tbaseline_accuracy\tassisted_accuracy\timmediate_ai_off_accuracy\tdelayed_ai_off_accuracy\ttransfer_accuracy\tunique_fact_retention\texpertise_routing_accuracy\tagency_score\tai_off_gate_status\tai_off_receipt_sha256";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arm {
    HumanOnly,
    HumanTms,
    HumanAi,
    HumanAiTms,
}

impl Arm {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "human_only" => Ok(Self::HumanOnly),
            "human_tms" => Ok(Self::HumanTms),
            "human_ai" => Ok(Self::HumanAi),
            "human_ai_tms" => Ok(Self::HumanAiTms),
            _ => Err(format!("unknown arm {value:?}")),
        }
    }

    fn has_ai(self) -> bool {
        matches!(self, Self::HumanAi | Self::HumanAiTms)
    }

    fn has_tms(self) -> bool {
        matches!(self, Self::HumanTms | Self::HumanAiTms)
    }

    fn index(self) -> usize {
        match self {
            Self::HumanOnly => 0,
            Self::HumanTms => 1,
            Self::HumanAi => 2,
            Self::HumanAiTms => 3,
        }
    }
}

#[derive(Debug, Clone)]
struct TeamObservation {
    task_bundle_sha256: String,
    ai_configuration_sha256: Option<String>,
    tms_configuration_sha256: Option<String>,
    arm: Arm,
    baseline_accuracy: f64,
    assisted_accuracy: Option<f64>,
    immediate_ai_off_accuracy: f64,
    delayed_ai_off_accuracy: f64,
    transfer_accuracy: f64,
    unique_fact_retention: f64,
    expertise_routing_accuracy: f64,
    agency_score: f64,
}

#[derive(Debug, Clone, Copy)]
enum Metric {
    ImmediateGain,
    DelayedGain,
    TransferGain,
    UniqueFactRetention,
    ExpertiseRoutingAccuracy,
    AgencyScore,
}

impl Metric {
    const ALL: [Self; 6] = [
        Self::ImmediateGain,
        Self::DelayedGain,
        Self::TransferGain,
        Self::UniqueFactRetention,
        Self::ExpertiseRoutingAccuracy,
        Self::AgencyScore,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::ImmediateGain => "immediate_ai_off_gain",
            Self::DelayedGain => "delayed_ai_off_gain",
            Self::TransferGain => "transfer_gain",
            Self::UniqueFactRetention => "unique_fact_retention",
            Self::ExpertiseRoutingAccuracy => "expertise_routing_accuracy",
            Self::AgencyScore => "agency_score",
        }
    }

    fn value(self, row: &TeamObservation) -> f64 {
        match self {
            Self::ImmediateGain => row.immediate_ai_off_accuracy - row.baseline_accuracy,
            Self::DelayedGain => row.delayed_ai_off_accuracy - row.baseline_accuracy,
            Self::TransferGain => row.transfer_accuracy - row.baseline_accuracy,
            Self::UniqueFactRetention => row.unique_fact_retention,
            Self::ExpertiseRoutingAccuracy => row.expertise_routing_accuracy,
            Self::AgencyScore => row.agency_score,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Contrast {
    estimate: f64,
    interval: Option<(f64, f64)>,
}

#[derive(Debug, Clone, Copy)]
struct EndpointEstimate {
    arm_means: [f64; 4],
    ai_main_effect: Contrast,
    tms_main_effect: Contrast,
    interaction: Contrast,
}

#[derive(Debug)]
struct Analysis {
    teams_per_arm: [usize; 4],
    endpoints: BTreeMap<&'static str, EndpointEstimate>,
    assisted_ai_arm_mean: Option<f64>,
    task_bundle_sha256: String,
    ai_configuration_sha256: Option<String>,
    tms_configuration_sha256: Option<String>,
    draws: usize,
    seed: u64,
}

/// Validate a team-level TSV and produce a reproducible factorial summary.
///
/// The estimator is descriptive. It does not authenticate the upstream AI-off
/// attestation, adjudicate task scores, or replace a preregistered study model.
pub fn summarize_tsv(input: &str, draws: usize, seed: u64) -> Result<String, String> {
    let observations = parse_tsv(input)?;
    let analysis = analyze(&observations, draws, seed)?;
    Ok(analysis.report())
}

fn parse_tsv(input: &str) -> Result<Vec<TeamObservation>, String> {
    let mut lines = input.lines().filter(|line| !line.trim().is_empty());
    let header = lines.next().ok_or_else(|| "input is empty".to_string())?;
    if header.trim_end_matches('\r') != HEADER {
        return Err(format!(
            "unexpected header; required exact header is: {HEADER}"
        ));
    }

    let mut observations = Vec::new();
    let mut team_ids = HashSet::new();
    let mut receipt_digests = HashSet::new();
    let mut task_bundle_digest: Option<String> = None;
    let mut ai_configuration_digest: Option<String> = None;
    let mut tms_configuration_digest: Option<String> = None;

    for (line_index, line) in lines.enumerate() {
        let line_number = line_index + 2;
        let fields: Vec<&str> = line.trim_end_matches('\r').split('\t').collect();
        if fields.len() != 14 {
            return Err(format!(
                "line {line_number}: expected 14 tab-separated fields, got {}",
                fields.len()
            ));
        }

        let team_id = require_nonempty(fields[0], "team_id", line_number)?;
        if !team_ids.insert(team_id.to_string()) {
            return Err(format!(
                "line {line_number}: duplicate team_id {team_id:?}; each row must represent one independent team"
            ));
        }

        let task_digest = fields[1].to_string();
        require_sha256(&task_digest, "task_bundle_sha256", line_number)?;
        match &task_bundle_digest {
            Some(expected) if expected != &task_digest => {
                return Err(format!(
                    "line {line_number}: task bundle differs from the matched task bundle"
                ));
            }
            None => task_bundle_digest = Some(task_digest.clone()),
            _ => {}
        }

        let arm = Arm::parse(fields[4]).map_err(|error| format!("line {line_number}: {error}"))?;
        let ai_digest = if fields[2] == "NA" {
            None
        } else {
            require_sha256(fields[2], "ai_configuration_sha256", line_number)?;
            Some(fields[2].to_string())
        };
        if arm.has_ai() {
            let digest = ai_digest.as_ref().ok_or_else(|| {
                format!(
                    "line {line_number}: AI-exposure arms require a pinned AI configuration digest"
                )
            })?;
            match &ai_configuration_digest {
                Some(expected) if expected != digest => {
                    return Err(format!(
                        "line {line_number}: AI configuration digest differs across AI-exposure arms"
                    ));
                }
                None => ai_configuration_digest = Some(digest.clone()),
                _ => {}
            }
        } else if ai_digest.is_some() {
            return Err(format!(
                "line {line_number}: human-only arms must use NA for ai_configuration_sha256"
            ));
        }


        let tms_digest = if fields[3] == "NA" {
            None
        } else {
            require_sha256(fields[3], "tms_configuration_sha256", line_number)?;
            Some(fields[3].to_string())
        };
        if arm.has_tms() {
            let digest = tms_digest.as_ref().ok_or_else(|| {
                format!(
                    "line {line_number}: TMS arms require a pinned TMS configuration digest"
                )
            })?;
            match &tms_configuration_digest {
                Some(expected) if expected != digest => {
                    return Err(format!(
                        "line {line_number}: TMS configuration digest differs across TMS arms"
                    ));
                }
                None => tms_configuration_digest = Some(digest.clone()),
                _ => {}
            }
        } else if tms_digest.is_some() {
            return Err(format!(
                "line {line_number}: non-TMS arms must use NA for tms_configuration_sha256"
            ));
        }

        let baseline_accuracy = parse_score(fields[5], "baseline_accuracy", line_number)?;
        let assisted_accuracy = if fields[5] == "NA" {
            None
        } else {
            Some(parse_score(fields[6], "assisted_accuracy", line_number)?)
        };
        if arm.has_ai() != assisted_accuracy.is_some() {
            return Err(format!(
                "line {line_number}: assisted_accuracy must be numeric only for AI-exposure arms and NA for human-only arms"
            ));
        }

        let immediate_ai_off_accuracy =
            parse_score(fields[7], "immediate_ai_off_accuracy", line_number)?;
        let delayed_ai_off_accuracy =
            parse_score(fields[8], "delayed_ai_off_accuracy", line_number)?;
        let transfer_accuracy = parse_score(fields[9], "transfer_accuracy", line_number)?;
        let unique_fact_retention = parse_score(fields[10], "unique_fact_retention", line_number)?;
        let expertise_routing_accuracy =
            parse_score(fields[11], "expertise_routing_accuracy", line_number)?;
        let agency_score = parse_score(fields[12], "agency_score", line_number)?;

        if fields[13] != "VERIFIED" {
            return Err(format!(
                "line {line_number}: AI-off gate is not VERIFIED ({:?}); fail closed",
                fields[13]
            ));
        }
        require_sha256(fields[14], "ai_off_receipt_sha256", line_number)?;
        if !receipt_digests.insert(fields[14].to_string()) {
            return Err(format!(
                "line {line_number}: duplicate AI-off receipt digest; receipts must be team/run-specific"
            ));
        }

        observations.push(TeamObservation {
            task_bundle_sha256: task_digest,
            ai_configuration_sha256: ai_digest,
            tms_configuration_sha256: tms_digest,
            arm,
            baseline_accuracy,
            assisted_accuracy,
            immediate_ai_off_accuracy,
            delayed_ai_off_accuracy,
            transfer_accuracy,
            unique_fact_retention,
            expertise_routing_accuracy,
            agency_score,
        });
    }

    if observations.is_empty() {
        return Err("input contains a header but no team rows".to_string());
    }
    Ok(observations)
}

fn require_nonempty<'a>(value: &'a str, name: &str, line_number: usize) -> Result<&'a str, String> {
    if value.trim().is_empty() {
        Err(format!("line {line_number}: {name} must not be empty"))
    } else {
        Ok(value)
    }
}

fn parse_score(value: &str, name: &str, line_number: usize) -> Result<f64, String> {
    let score = value
        .parse::<f64>()
        .map_err(|_| format!("line {line_number}: {name} is not a number"))?;
    if !score.is_finite() || !(0.0..=1.0).contains(&score) {
        return Err(format!(
            "line {line_number}: {name} must be finite and in [0, 1]"
        ));
    }
    Ok(score)
}

fn require_sha256(value: &str, name: &str, line_number: usize) -> Result<(), String> {
    let valid = value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase());
    if valid {
        Ok(())
    } else {
        Err(format!(
            "line {line_number}: {name} must be 64 lowercase hexadecimal characters"
        ))
    }
}

fn analyze(observations: &[TeamObservation], draws: usize, seed: u64) -> Result<Analysis, String> {
    if draws < 100 {
        return Err("bootstrap draws must be at least 100".to_string());
    }
    if observations.is_empty() {
        return Err("at least one team observation is required".to_string());
    }

    let mut arms: [Vec<&TeamObservation>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for row in observations {
        arms[row.arm.index()].push(row);
    }
    if arms.iter().any(|arm| arm.is_empty()) {
        return Err("all four factorial arms must be represented".to_string());
    }

    let teams_per_arm = [arms[0].len(), arms[1].len(), arms[2].len(), arms[3].len()];
    let mut endpoints = BTreeMap::new();
    for (metric_index, metric) in Metric::ALL.iter().copied().enumerate() {
        let metric_seed =
            seed.wrapping_add((metric_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        endpoints.insert(
            metric.name(),
            estimate_endpoint(&arms, metric, draws, metric_seed),
        );
    }

    let assisted_scores: Vec<f64> = observations
        .iter()
        .filter(|row| row.arm.has_ai())
        .filter_map(|row| row.assisted_accuracy)
        .collect();
    let assisted_ai_arm_mean = if assisted_scores.is_empty() {
        None
    } else {
        Some(mean(&assisted_scores))
    };

    Ok(Analysis {
        teams_per_arm,
        endpoints,
        assisted_ai_arm_mean,
        task_bundle_sha256: observations[0].task_bundle_sha256.clone(),
        ai_configuration_sha256: observations.iter().find_map(|row| row.ai_configuration_sha256.clone()),
        tms_configuration_sha256: observations.iter().find_map(|row| row.tms_configuration_sha256.clone()),
        draws,
        seed,
    })
}

fn estimate_endpoint(
    arms: &[Vec<&TeamObservation>; 4],
    metric: Metric,
    draws: usize,
    seed: u64,
) -> EndpointEstimate {
    let means = [
        mean_for_arm(&arms[0], metric),
        mean_for_arm(&arms[1], metric),
        mean_for_arm(&arms[2], metric),
        mean_for_arm(&arms[3], metric),
    ];
    let point_effects = factorial_effects(means);
    let clusters = [arms[0].len(), arms[1].len(), arms[2].len(), arms[3].len()];

    if clusters.iter().any(|count| *count < 2) {
        return EndpointEstimate {
            arm_means: means,
            ai_main_effect: Contrast {
                estimate: point_effects[0],
                interval: None,
            },
            tms_main_effect: Contrast {
                estimate: point_effects[1],
                interval: None,
            },
            interaction: Contrast {
                estimate: point_effects[2],
                interval: None,
            },
        };
    }

    let mut rng = SplitMix64::new(seed);
    let mut bootstrap_effects: [Vec<f64>; 3] = [
        Vec::with_capacity(draws),
        Vec::with_capacity(draws),
        Vec::with_capacity(draws),
    ];
    for _ in 0..draws {
        let bootstrap_means = [
            bootstrap_mean(&arms[0], metric, &mut rng),
            bootstrap_mean(&arms[1], metric, &mut rng),
            bootstrap_mean(&arms[2], metric, &mut rng),
            bootstrap_mean(&arms[3], metric, &mut rng),
        ];
        let effects = factorial_effects(bootstrap_means);
        for (index, effect) in effects.iter().copied().enumerate() {
            bootstrap_effects[index].push(effect);
        }
    }

    EndpointEstimate {
        arm_means: means,
        ai_main_effect: with_interval(point_effects[0], &mut bootstrap_effects[0]),
        tms_main_effect: with_interval(point_effects[1], &mut bootstrap_effects[1]),
        interaction: with_interval(point_effects[2], &mut bootstrap_effects[2]),
    }
}

fn factorial_effects(means: [f64; 4]) -> [f64; 3] {
    // H0=human_only, H1=human_tms, A0=human_ai, A1=human_ai_tms.
    let ai_main = ((means[2] + means[3]) - (means[0] + means[1])) / 2.0;
    let tms_main = ((means[1] + means[3]) - (means[0] + means[2])) / 2.0;
    let interaction = (means[3] - means[2]) - (means[1] - means[0]);
    [ai_main, tms_main, interaction]
}

fn mean_for_arm(rows: &[&TeamObservation], metric: Metric) -> f64 {
    rows.iter().map(|row| metric.value(row)).sum::<f64>() / rows.len() as f64
}

fn bootstrap_mean(rows: &[&TeamObservation], metric: Metric, rng: &mut SplitMix64) -> f64 {
    let mut sum = 0.0;
    for _ in 0..rows.len() {
        sum += metric.value(rows[rng.below(rows.len())]);
    }
    sum / rows.len() as f64
}

fn with_interval(estimate: f64, values: &mut [f64]) -> Contrast {
    values.sort_by(f64::total_cmp);
    Contrast {
        estimate,
        interval: Some((percentile(values, 0.025), percentile(values, 0.975))),
    }
}

fn percentile(sorted: &[f64], probability: f64) -> f64 {
    let position = probability * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

impl Analysis {
    fn report(&self) -> String {
        let total: usize = self.teams_per_arm.iter().sum();
        let intervals_available = self.teams_per_arm.iter().all(|count| *count >= 2);
        let mut output = String::new();
        let _ = writeln!(output, "DELIB-TEAM-RETENTION-001 / factorial summary v0.1");
        let _ = writeln!(
            output,
            "status: {}",
            if intervals_available {
                "ESTIMATES_WITH_TEAM_BOOTSTRAP_INTERVALS"
            } else {
                "POINT_ESTIMATES_ONLY_INSUFFICIENT_TEAMS_FOR_INTERVALS"
            }
        );
        let _ = writeln!(
            output,
            "teams: {total}; per_arm: human_only={}, human_tms={}, human_ai={}, human_ai_tms={}",
            self.teams_per_arm[0],
            self.teams_per_arm[1],
            self.teams_per_arm[2],
            self.teams_per_arm[3]
        );
        let _ = writeln!(output, "task_bundle_sha256: {}", self.task_bundle_sha256);
        if let Some(digest) = &self.ai_configuration_sha256 {
            let _ = writeln!(output, "ai_configuration_sha256: {digest}");
        }
        if let Some(digest) = &self.tms_configuration_sha256 {
            let _ = writeln!(output, "tms_configuration_sha256: {digest}");
        }
        let _ = writeln!(
            output,
            "bootstrap: draws={}, seed={}, resampling_unit=team",
            self.draws, self.seed
        );
        let _ = writeln!(
            output,
            "AI-off receipt status: input asserted VERIFIED; digest format checked only; this report does not authenticate upstream isolation attestation"
        );
        if let Some(mean) = self.assisted_ai_arm_mean {
            let _ = writeln!(
                output,
                "AI-assisted mean (AI arms only; descriptive, not a randomized cross-arm contrast): {mean:.4}"
            );
        }
        let _ = writeln!(
            output,
            "\nEndpoint means and factorial contrasts (positive = higher score / gain):"
        );
        for (name, endpoint) in &self.endpoints {
            let _ = writeln!(output, "\n{name}");
            let _ = writeln!(
                output,
                "  means: H0={:.4} H1={:.4} A0={:.4} A1={:.4}",
                endpoint.arm_means[0],
                endpoint.arm_means[1],
                endpoint.arm_means[2],
                endpoint.arm_means[3]
            );
            write_contrast(&mut output, "AI main effect", endpoint.ai_main_effect);
            write_contrast(&mut output, "TMS main effect", endpoint.tms_main_effect);
            write_contrast(&mut output, "AI×TMS interaction", endpoint.interaction);
        }
        let _ = writeln!(
            output,
            "\nInterpretation boundary: synthetic or pilot estimates do not establish human benefit, causal generalization, collective intelligence, or superiority. Confirmatory claims require preregistration, adequate group-level power, independently scored held-out tasks, multiplicity control, and external replication."
        );
        output
    }
}

fn write_contrast(output: &mut String, label: &str, contrast: Contrast) {
    match contrast.interval {
        Some((lower, upper)) => {
            let _ = writeln!(
                output,
                "  {label}: {:.4} [95% percentile team-bootstrap CI {:.4}, {:.4}]",
                contrast.estimate, lower, upper
            );
        }
        None => {
            let _ = writeln!(
                output,
                "  {label}: {:.4} [CI unavailable: fewer than 2 independent teams in at least one arm]",
                contrast.estimate
            );
        }
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn below(&mut self, upper: usize) -> usize {
        let bound = upper as u64;
        let zone = u64::MAX - (u64::MAX % bound);
        loop {
            let value = self.next();
            if value < zone {
                return (value % bound) as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repeated_hash(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn unique_hash(value: &str) -> String {
        let mut state = 0xcbf2_9ce4_8422_2325u64;
        for byte in value.bytes() {
            state = (state ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
        format!(
            "{:016x}{:016x}{:016x}{:016x}",
            state,
            state ^ 0x9E37_79B9_7F4A_7C15,
            state.rotate_left(17),
            state.rotate_right(13)
        )
    }

    fn row(_team: &str, arm: Arm, immediate: f64) -> TeamObservation {
        TeamObservation {
            task_bundle_sha256: repeated_hash('a'),
            ai_configuration_sha256: arm.has_ai().then(|| repeated_hash('c')),
            tms_configuration_sha256: arm.has_tms().then(|| repeated_hash('d')),
            arm,
            baseline_accuracy: 0.5,
            assisted_accuracy: if arm.has_ai() { Some(0.8) } else { None },
            immediate_ai_off_accuracy: immediate,
            delayed_ai_off_accuracy: immediate,
            transfer_accuracy: immediate,
            unique_fact_retention: immediate,
            expertise_routing_accuracy: immediate,
            agency_score: immediate,
        }
    }

    fn all_arms() -> Vec<TeamObservation> {
        vec![
            row("h0-1", Arm::HumanOnly, 0.50),
            row("h0-2", Arm::HumanOnly, 0.50),
            row("h1-1", Arm::HumanTms, 0.60),
            row("h1-2", Arm::HumanTms, 0.60),
            row("a0-1", Arm::HumanAi, 0.40),
            row("a0-2", Arm::HumanAi, 0.40),
            row("a1-1", Arm::HumanAiTms, 0.65),
            row("a1-2", Arm::HumanAiTms, 0.65),
        ]
    }

    #[test]
    fn factorial_contrasts_match_hand_calculation() {
        let rows = all_arms();
        let result = analyze(&rows, 500, 7).unwrap();
        let endpoint = result.endpoints.get("immediate_ai_off_gain").unwrap();
        assert!((endpoint.ai_main_effect.estimate - (-0.025)).abs() < 1e-12);
        assert!((endpoint.tms_main_effect.estimate - 0.175).abs() < 1e-12);
        assert!((endpoint.interaction.estimate - 0.15).abs() < 1e-12);
    }

    #[test]
    fn bootstrap_is_deterministic_for_fixed_seed() {
        let rows = all_arms();
        let first = analyze(&rows, 1_000, 123).unwrap();
        let second = analyze(&rows, 1_000, 123).unwrap();
        let first_interval = first.endpoints["immediate_ai_off_gain"]
            .ai_main_effect
            .interval;
        let second_interval = second.endpoints["immediate_ai_off_gain"]
            .ai_main_effect
            .interval;
        assert_eq!(first_interval, second_interval);
    }

    #[test]
    fn one_team_per_arm_yields_point_estimate_without_interval() {
        let rows = vec![
            row("h0", Arm::HumanOnly, 0.50),
            row("h1", Arm::HumanTms, 0.60),
            row("a0", Arm::HumanAi, 0.40),
            row("a1", Arm::HumanAiTms, 0.65),
        ];
        let result = analyze(&rows, 500, 42).unwrap();
        assert!(
            result.endpoints["immediate_ai_off_gain"]
                .ai_main_effect
                .interval
                .is_none()
        );
    }

    #[test]
    fn parser_rejects_unverified_ai_off_and_duplicate_receipts() {
        let mut input = String::from(HEADER);
        for (index, arm) in ["human_only", "human_tms", "human_ai", "human_ai_tms"]
            .iter()
            .enumerate()
        {
            let ai_digest = if index < 2 {
                "NA".to_string()
            } else {
                repeated_hash('c')
            };
            let tms_digest = if index == 1 || index == 3 {
                repeated_hash('d')
            } else {
                "NA".to_string()
            };
            let assisted = if index < 2 { "NA" } else { "0.8" };
            let receipt = unique_hash(arm);
            let line = format!(
                "team-{index}\t{}\t{}\t{}\t{}\t0.5\t{}\t0.5\t0.4\t0.45\t0.6\t0.7\t0.8\tVERIFIED\t{}",
                repeated_hash('a'),
                ai_digest,
                tms_digest,
                arm,
                assisted,
                receipt
            );
            input.push('\n');
            input.push_str(&line);
        }
        assert!(parse_tsv(&input).is_ok());
        assert!(parse_tsv(&input.replace("VERIFIED", "UNKNOWN")).is_err());
        assert!(parse_tsv(&input.replace(&repeated_hash('d'), "NA")).is_err());
        let mismatched_tms = input.replacen(&repeated_hash('d'), &repeated_hash('e'), 1);
        assert!(
            parse_tsv(&mismatched_tms)
                .unwrap_err()
                .contains("TMS configuration digest differs")
        );
        let duplicate_receipt =
            input.replace(&unique_hash("human_ai_tms"), &unique_hash("human_only"));
        assert!(
            parse_tsv(&duplicate_receipt)
                .unwrap_err()
                .contains("duplicate AI-off receipt")
        );
    }

    #[test]
    fn checked_in_synthetic_fixture_is_valid_and_identifies_all_arms() {
        let input = include_str!(
            "../../../../docs/benchmarks/sympoetic/fixtures/team-retention-synthetic-v0.1.tsv"
        );
        let report = summarize_tsv(input, 2_000, 42).unwrap();
        assert!(
            report.contains(
                "teams: 16; per_arm: human_only=4, human_tms=4, human_ai=4, human_ai_tms=4"
            )
        );
        assert!(report.contains("resampling_unit=team"));
        assert!(report.contains("digest format checked only"));
        assert!(report.contains("AI main effect: -0.0350"));
        assert!(report.contains("TMS main effect: 0.1100"));
        assert!(report.contains("AI×TMS interaction: 0.0600"));
    }

    #[test]
    fn parser_rejects_duplicate_team_ids_and_mismatched_task_bundles() {
        let mut input = String::from(HEADER);
        for (index, arm) in ["human_only", "human_tms", "human_ai", "human_ai_tms"]
            .iter()
            .enumerate()
        {
            let ai_digest = if index < 2 {
                "NA".to_string()
            } else {
                repeated_hash('c')
            };
            let tms_digest = if index == 1 || index == 3 {
                repeated_hash('d')
            } else {
                "NA".to_string()
            };
            let assisted = if index < 2 { "NA" } else { "0.8" };
            let line = format!(
                "team-{index}\t{}\t{}\t{}\t{}\t0.5\t{}\t0.5\t0.4\t0.45\t0.6\t0.7\t0.8\tVERIFIED\t{}",
                repeated_hash('a'),
                ai_digest,
                tms_digest,
                arm,
                assisted,
                unique_hash(&format!("team-{index}"))
            );
            input.push('\n');
            input.push_str(&line);
        }
        let duplicated_team = input.replace("team-3", "team-2");
        assert!(
            parse_tsv(&duplicated_team)
                .unwrap_err()
                .contains("duplicate team_id")
        );
        let mismatched_bundle = input.replacen(&repeated_hash('a'), &repeated_hash('d'), 1);
        assert!(
            parse_tsv(&mismatched_bundle)
                .unwrap_err()
                .contains("task bundle differs")
        );
    }
}
