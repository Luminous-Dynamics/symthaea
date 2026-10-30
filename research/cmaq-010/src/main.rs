use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;

const SCHEMA: &str = "cmaq-010-evidence-integrity-v0.1";
const EPISODES: usize = 512;
const HORIZON: usize = 128;
const SHIFT_STEP: usize = 64;
const NULL_SIGNAL_RATE: f64 = 0.01;
const SHIFT_SIGNAL_RATE: f64 = 0.80;
const CUSUM_THRESHOLD: f64 = 5.0;
const ROBUST_NULL_UPPER: f64 = 0.02;
const ROBUST_SHIFT_LOWER: f64 = 0.70;
const MIN_INDEPENDENT_GROUPS: usize = 2;
const MISSING_STEPS_TO_FALLBACK: usize = 4;
const BASE_SEED: u64 = 0xC0A9_0010_2026_1030;
const SUPPORTED_UTILITY: f64 = 0.03;
const UNSUPPORTED_UTILITY: f64 = -0.20;

#[derive(Clone, Copy, Debug)]
struct SplitMix64 { state: u64 }

impl SplitMix64 {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64) }
    fn bit(&mut self, probability: f64) -> bool { self.unit() < probability }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario { CleanIndependent, CommonMode, SybilFanout, OneByzantine, ReplaySafe }

impl Scenario {
    fn all() -> [Self; 5] {
        [Self::CleanIndependent, Self::CommonMode, Self::SybilFanout, Self::OneByzantine, Self::ReplaySafe]
    }
    fn name(self) -> &'static str {
        match self {
            Self::CleanIndependent => "clean_independent",
            Self::CommonMode => "common_mode_duplicates",
            Self::SybilFanout => "sybil_fanout",
            Self::OneByzantine => "one_byzantine_source",
            Self::ReplaySafe => "stale_safe_replay",
        }
    }
    fn shifts(self) -> bool { self != Self::CleanIndependent }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Policy { DirectFallback, ClaimSum, ClaimMajority, RootDedupSum, RobustRootMajority, IntegrityGate, Oracle }

impl Policy {
    fn all() -> [Self; 7] {
        [Self::DirectFallback, Self::ClaimSum, Self::ClaimMajority, Self::RootDedupSum,
         Self::RobustRootMajority, Self::IntegrityGate, Self::Oracle]
    }
    fn name(self) -> &'static str {
        match self {
            Self::DirectFallback => "direct_fallback",
            Self::ClaimSum => "naive_claim_sum",
            Self::ClaimMajority => "naive_claim_majority",
            Self::RootDedupSum => "provenance_root_dedup_sum",
            Self::RobustRootMajority => "robust_root_majority",
            Self::IntegrityGate => "freshness_provenance_fail_closed",
            Self::Oracle => "oracle_changepoint",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Report {
    claimed_source: usize,
    provenance_group: usize,
    epoch: usize,
    context: usize,
    nonce: u64,
    positive: bool,
}

#[derive(Clone, Debug)]
struct EpisodeResult {
    utility: f64,
    active_steps: usize,
    expected_unsupported: f64,
    risk_alarm_step: Option<usize>,
    provenance_fallback_step: Option<usize>,
    rejected_stale: usize,
    rejected_duplicate: usize,
    max_evidence_updates: usize,
    max_claims_in_step: usize,
    mean_independent_groups: f64,
}

#[derive(Clone, Debug)]
struct Summary {
    scenario: &'static str,
    policy: &'static str,
    mean_utility: f64,
    utility_delta_vs_claim_sum: f64,
    model_action_coverage: f64,
    unsupported_rate_per_model_action: f64,
    false_revocation_episode_rate: f64,
    shift_detection_coverage: f64,
    p90_detection_delay_steps: f64,
    provenance_fail_closed_rate: f64,
    mean_rejected_stale: f64,
    mean_rejected_duplicate: f64,
    mean_max_evidence_updates: f64,
    max_evidence_updates: usize,
    mean_independent_groups: f64,
}

fn llr(positive: bool, p0: f64, p1: f64) -> f64 {
    if positive { (p1 / p0).ln() } else { ((1.0 - p1) / (1.0 - p0)).ln() }
}

fn update_cusum(statistic: &mut f64, positive: bool, p0: f64, p1: f64) -> bool {
    *statistic = (*statistic + llr(positive, p0, p1)).max(0.0);
    *statistic >= CUSUM_THRESHOLD
}

fn action_unsupported_rate(scenario: Scenario, step: usize) -> f64 {
    if scenario.shifts() && step >= SHIFT_STEP { 0.25 } else { 0.01 }
}

fn make_tape(scenario: Scenario, episode: usize) -> Vec<Vec<Report>> {
    let seed = BASE_SEED
        ^ ((scenario as u64) << 40)
        ^ (episode as u64).wrapping_mul(0xD134_2543_DE82_EF95);
    let mut rng = SplitMix64::new(seed);
    let mut tape = Vec::with_capacity(HORIZON);

    for step in 0..HORIZON {
        let shifted = scenario.shifts() && step >= SHIFT_STEP;
        let signal_rate = if shifted { SHIFT_SIGNAL_RATE } else { NULL_SIGNAL_RATE };
        let mut reports = Vec::new();

        match scenario {
            Scenario::CleanIndependent => {
                for source in 0..3 {
                    reports.push(Report {
                        claimed_source: source,
                        provenance_group: source,
                        epoch: step,
                        context: step,
                        nonce: (step as u64) * 16 + source as u64,
                        positive: rng.bit(signal_rate),
                    });
                }
            }
            Scenario::CommonMode => {
                let one_measurement = rng.bit(signal_rate);
                for claimed_source in 0..3 {
                    reports.push(Report {
                        claimed_source,
                        provenance_group: 0,
                        epoch: step,
                        context: step,
                        nonce: (step as u64) * 16,
                        positive: one_measurement,
                    });
                }
            }
            Scenario::SybilFanout => {
                let one_measurement = rng.bit(signal_rate);
                for claimed_source in 0..6 {
                    reports.push(Report {
                        claimed_source,
                        provenance_group: 0,
                        epoch: step,
                        context: step,
                        nonce: (step as u64) * 16,
                        positive: one_measurement,
                    });
                }
            }
            Scenario::OneByzantine => {
                for source in 0..3 {
                    let honest_measurement = rng.bit(signal_rate);
                    reports.push(Report {
                        claimed_source: source,
                        provenance_group: source,
                        epoch: step,
                        context: step,
                        nonce: (step as u64) * 16 + source as u64,
                        positive: if source == 2 { !honest_measurement } else { honest_measurement },
                    });
                }
            }
            Scenario::ReplaySafe => {
                if shifted {
                    // Replays a safe report from epoch/context zero with the same nonce.
                    for source in 0..3 {
                        reports.push(Report {
                            claimed_source: source,
                            provenance_group: source,
                            epoch: 0,
                            context: 0,
                            nonce: source as u64,
                            positive: false,
                        });
                    }
                } else {
                    for source in 0..3 {
                        reports.push(Report {
                            claimed_source: source,
                            provenance_group: source,
                            epoch: step,
                            context: step,
                            nonce: (step as u64) * 16 + source as u64,
                            positive: rng.bit(signal_rate),
                        });
                    }
                }
            }
        }
        tape.push(reports);
    }
    tape
}

fn by_claim_majority(reports: &[Report]) -> bool {
    let positive = reports.iter().filter(|r| r.positive).count();
    positive * 2 > reports.len()
}

fn first_per_provenance_group(reports: &[Report]) -> BTreeMap<usize, bool> {
    let mut groups = BTreeMap::new();
    for report in reports {
        groups.entry(report.provenance_group).or_insert(report.positive);
    }
    groups
}

fn fresh_reports(
    reports: &[Report],
    step: usize,
    seen_nonces: &mut HashSet<u64>,
) -> (Vec<Report>, usize, usize) {
    let mut fresh = Vec::new();
    let mut stale = 0;
    let mut duplicate = 0;
    for report in reports {
        if report.epoch != step || report.context != step {
            stale += 1;
            continue;
        }
        if !seen_nonces.insert(report.nonce) {
            duplicate += 1;
            continue;
        }
        fresh.push(*report);
    }
    (fresh, stale, duplicate)
}

fn run_episode(scenario: Scenario, policy: Policy, tape: &[Vec<Report>]) -> EpisodeResult {
    let mut utility = 0.0;
    let mut active_steps = 0;
    let mut expected_unsupported = 0.0;
    let mut cusum = 0.0;
    let mut risk_alarm_step = None;
    let mut provenance_fallback_step = None;
    let mut missing_run = 0;
    let mut seen_nonces = HashSet::new();
    let mut rejected_stale = 0;
    let mut rejected_duplicate = 0;
    let mut max_evidence_updates = 0;
    let mut max_claims_in_step = 0;
    let mut total_independent_groups = 0usize;

    for step in 0..HORIZON {
        let reports = &tape[step];
        max_claims_in_step = max_claims_in_step.max(reports.len());
        let active = match policy {
            Policy::DirectFallback => false,
            _ => risk_alarm_step.is_none() && provenance_fallback_step.is_none(),
        };

        if active {
            let unsupported = action_unsupported_rate(scenario, step);
            utility += SUPPORTED_UTILITY * (1.0 - unsupported) + UNSUPPORTED_UTILITY * unsupported;
            expected_unsupported += unsupported;
            active_steps += 1;
        }

        if risk_alarm_step.is_some() || provenance_fallback_step.is_some() {
            continue;
        }

        let group_map = first_per_provenance_group(reports);
        total_independent_groups += group_map.len();

        let alarm = match policy {
            Policy::DirectFallback => false,
            Policy::Oracle => {
                if scenario.shifts() && step == SHIFT_STEP {
                    true
                } else {
                    false
                }
            }
            Policy::ClaimSum => {
                max_evidence_updates = max_evidence_updates.max(reports.len());
                let mut alarm_now = false;
                for report in reports {
                    if update_cusum(&mut cusum, report.positive, NULL_SIGNAL_RATE, SHIFT_SIGNAL_RATE) {
                        alarm_now = true;
                        break;
                    }
                }
                alarm_now
            }
            Policy::ClaimMajority => {
                max_evidence_updates = max_evidence_updates.max(1);
                if reports.is_empty() { false }
                else { update_cusum(&mut cusum, by_claim_majority(reports), NULL_SIGNAL_RATE, SHIFT_SIGNAL_RATE) }
            }
            Policy::RootDedupSum => {
                max_evidence_updates = max_evidence_updates.max(group_map.len());
                let mut alarm_now = false;
                for positive in group_map.values() {
                    if update_cusum(&mut cusum, *positive, NULL_SIGNAL_RATE, SHIFT_SIGNAL_RATE) {
                        alarm_now = true;
                        break;
                    }
                }
                alarm_now
            }
            Policy::RobustRootMajority => {
                max_evidence_updates = max_evidence_updates.max(usize::from(!group_map.is_empty()));
                if group_map.is_empty() { false }
                else {
                    let positives = group_map.values().filter(|positive| **positive).count();
                    let majority = positives * 2 > group_map.len();
                    update_cusum(&mut cusum, majority, ROBUST_NULL_UPPER, ROBUST_SHIFT_LOWER)
                }
            }
            Policy::IntegrityGate => {
                let (fresh, stale, duplicate) = fresh_reports(reports, step, &mut seen_nonces);
                rejected_stale += stale;
                rejected_duplicate += duplicate;
                let fresh_groups = first_per_provenance_group(&fresh);
                max_evidence_updates = max_evidence_updates.max(usize::from(fresh_groups.len() >= MIN_INDEPENDENT_GROUPS));
                if fresh_groups.len() < MIN_INDEPENDENT_GROUPS {
                    missing_run += 1;
                    if missing_run >= MISSING_STEPS_TO_FALLBACK {
                        provenance_fallback_step = Some(step);
                    }
                    false
                } else {
                    missing_run = 0;
                    let positives = fresh_groups.values().filter(|positive| **positive).count();
                    let majority = positives * 2 > fresh_groups.len();
                    update_cusum(&mut cusum, majority, ROBUST_NULL_UPPER, ROBUST_SHIFT_LOWER)
                }
            }
        };

        if alarm {
            risk_alarm_step = Some(step);
        }
    }

    EpisodeResult {
        utility,
        active_steps,
        expected_unsupported,
        risk_alarm_step,
        provenance_fallback_step,
        rejected_stale,
        rejected_duplicate,
        max_evidence_updates,
        max_claims_in_step,
        mean_independent_groups: total_independent_groups as f64 / HORIZON as f64,
    }
}

fn percentile90(values: &mut [usize]) -> f64 {
    if values.is_empty() { return -1.0; }
    values.sort_unstable();
    let index = ((values.len() as f64 * 0.90).ceil() as usize).saturating_sub(1);
    values[index] as f64
}

fn summarize(scenario: Scenario, policy: Policy, baseline_utility: f64) -> Summary {
    let mut utility = 0.0;
    let mut active_steps = 0usize;
    let mut unsupported = 0.0;
    let mut false_revocations = 0usize;
    let mut detections = 0usize;
    let mut delays = Vec::new();
    let mut provenance_fallbacks = 0usize;
    let mut stale_rejected = 0usize;
    let mut duplicates_rejected = 0usize;
    let mut evidence_updates = 0usize;
    let mut max_updates = 0usize;
    let mut independent_groups = 0.0;

    for episode in 0..EPISODES {
        let tape = make_tape(scenario, episode);
        let result = run_episode(scenario, policy, &tape);
        utility += result.utility;
        active_steps += result.active_steps;
        unsupported += result.expected_unsupported;
        if result.provenance_fallback_step.is_some() { provenance_fallbacks += 1; }
        stale_rejected += result.rejected_stale;
        duplicates_rejected += result.rejected_duplicate;
        evidence_updates += result.max_evidence_updates;
        max_updates = max_updates.max(result.max_evidence_updates);
        independent_groups += result.mean_independent_groups;

        if let Some(alarm_step) = result.risk_alarm_step {
            if !scenario.shifts() || alarm_step < SHIFT_STEP {
                false_revocations += 1;
            } else {
                detections += 1;
                delays.push(alarm_step.saturating_sub(SHIFT_STEP));
            }
        }
    }

    let total_actions = active_steps as f64;
    let mean_utility = utility / EPISODES as f64;
    let mean_coverage = total_actions / (EPISODES * HORIZON) as f64;
    let unsupported_rate = if total_actions > 0.0 { unsupported / total_actions } else { 0.0 };

    Summary {
        scenario: scenario.name(),
        policy: policy.name(),
        mean_utility,
        utility_delta_vs_claim_sum: mean_utility - baseline_utility,
        model_action_coverage: mean_coverage,
        unsupported_rate_per_model_action: unsupported_rate,
        false_revocation_episode_rate: false_revocations as f64 / EPISODES as f64,
        shift_detection_coverage: if scenario.shifts() { detections as f64 / EPISODES as f64 } else { 0.0 },
        p90_detection_delay_steps: percentile90(&mut delays),
        provenance_fail_closed_rate: provenance_fallbacks as f64 / EPISODES as f64,
        mean_rejected_stale: stale_rejected as f64 / EPISODES as f64,
        mean_rejected_duplicate: duplicates_rejected as f64 / EPISODES as f64,
        mean_max_evidence_updates: evidence_updates as f64 / EPISODES as f64,
        max_evidence_updates: max_updates,
        mean_independent_groups: independent_groups / EPISODES as f64,
    }
}

fn number(value: f64) -> String {
    if value.is_finite() { format!("{value:.8}") } else { "null".to_string() }
}

fn report_json(rows: &[Summary]) -> String {
    let scenarios = Scenario::all().iter().map(|s| format!("\"{}\"", s.name())).collect::<Vec<_>>().join(",");
    let policies = Policy::all().iter().map(|p| format!("\"{}\"", p.name())).collect::<Vec<_>>().join(",");
    let mut output = String::new();
    writeln!(&mut output, "{{").unwrap();
    writeln!(&mut output, "  \"schema_version\": \"{SCHEMA}\",").unwrap();
    writeln!(&mut output, "  \"episode_count\": {EPISODES},").unwrap();
    writeln!(&mut output, "  \"horizon\": {HORIZON},").unwrap();
    writeln!(&mut output, "  \"shift_step\": {SHIFT_STEP},").unwrap();
    writeln!(&mut output, "  \"cusum_threshold\": {},", number(CUSUM_THRESHOLD)).unwrap();
    writeln!(&mut output, "  \"null_signal_rate\": {},", number(NULL_SIGNAL_RATE)).unwrap();
    writeln!(&mut output, "  \"shift_signal_rate\": {},", number(SHIFT_SIGNAL_RATE)).unwrap();
    writeln!(&mut output, "  \"robust_null_upper\": {},", number(ROBUST_NULL_UPPER)).unwrap();
    writeln!(&mut output, "  \"robust_shift_lower\": {},", number(ROBUST_SHIFT_LOWER)).unwrap();
    writeln!(&mut output, "  \"min_independent_groups\": {MIN_INDEPENDENT_GROUPS},").unwrap();
    writeln!(&mut output, "  \"missing_steps_to_fallback\": {MISSING_STEPS_TO_FALLBACK},").unwrap();
    writeln!(&mut output, "  \"scenarios\": [{scenarios}],").unwrap();
    writeln!(&mut output, "  \"policies\": [{policies}],").unwrap();
    writeln!(&mut output, "  \"summary\": [").unwrap();
    for (index, row) in rows.iter().enumerate() {
        let comma = if index + 1 == rows.len() { "" } else { "," };
        writeln!(&mut output,
            concat!(
              "    {{\"scenario\":\"{}\",\"policy\":\"{}\",",
              "\"mean_utility\":{},\"utility_delta_vs_claim_sum\":{},",
              "\"model_action_coverage\":{},\"unsupported_rate_per_model_action\":{},",
              "\"false_revocation_episode_rate\":{},\"shift_detection_coverage\":{},",
              "\"p90_detection_delay_steps\":{},\"provenance_fail_closed_rate\":{},",
              "\"mean_rejected_stale\":{},\"mean_rejected_duplicate\":{},",
              "\"mean_max_evidence_updates\":{},\"max_evidence_updates\":{},",
              "\"mean_independent_groups\":{}}}{}"
            ),
            row.scenario, row.policy,
            number(row.mean_utility), number(row.utility_delta_vs_claim_sum),
            number(row.model_action_coverage), number(row.unsupported_rate_per_model_action),
            number(row.false_revocation_episode_rate), number(row.shift_detection_coverage),
            number(row.p90_detection_delay_steps), number(row.provenance_fail_closed_rate),
            number(row.mean_rejected_stale), number(row.mean_rejected_duplicate),
            number(row.mean_max_evidence_updates), row.max_evidence_updates,
            number(row.mean_independent_groups), comma
        ).unwrap();
    }
    writeln!(&mut output, "  ]").unwrap();
    writeln!(&mut output, "}}").unwrap();
    output
}

fn main() {
    let mut rows = Vec::new();
    for scenario in Scenario::all() {
        let claim_sum = summarize(scenario, Policy::ClaimSum, 0.0);
        let baseline = claim_sum.mean_utility;
        rows.push(claim_sum);
        for policy in Policy::all().into_iter().skip(1) {
            rows.push(summarize(scenario, policy, baseline));
        }
    }
    print!("{}", report_json(&rows));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn robust_majority_tolerates_one_inverted_source() {
        assert!(by_claim_majority(&[
            Report { claimed_source: 0, provenance_group: 0, epoch: 0, context: 0, nonce: 0, positive: true },
            Report { claimed_source: 1, provenance_group: 1, epoch: 0, context: 0, nonce: 1, positive: true },
            Report { claimed_source: 2, provenance_group: 2, epoch: 0, context: 0, nonce: 2, positive: false },
        ]));
        assert!(!by_claim_majority(&[
            Report { claimed_source: 0, provenance_group: 0, epoch: 0, context: 0, nonce: 0, positive: false },
            Report { claimed_source: 1, provenance_group: 1, epoch: 0, context: 0, nonce: 1, positive: false },
            Report { claimed_source: 2, provenance_group: 2, epoch: 0, context: 0, nonce: 2, positive: true },
        ]));
    }

    #[test]
    fn claimed_aliases_collapse_to_one_provenance_group() {
        let reports = (0..6).map(|claim| Report {
            claimed_source: claim, provenance_group: 7, epoch: 4, context: 4,
            nonce: 64, positive: true,
        }).collect::<Vec<_>>();
        assert_eq!(first_per_provenance_group(&reports).len(), 1);
    }

    #[test]
    fn stale_epoch_or_context_is_rejected() {
        let reports = [Report {
            claimed_source: 0, provenance_group: 0, epoch: 1, context: 2,
            nonce: 9, positive: false,
        }];
        let (fresh, stale, duplicate) = fresh_reports(&reports, 2, &mut HashSet::new());
        assert!(fresh.is_empty());
        assert_eq!(stale, 1);
        assert_eq!(duplicate, 0);
    }

    #[test]
    fn replayed_nonce_is_rejected_after_first_use() {
        let reports = [
            Report { claimed_source: 0, provenance_group: 0, epoch: 0, context: 0, nonce: 9, positive: false },
            Report { claimed_source: 1, provenance_group: 1, epoch: 0, context: 0, nonce: 9, positive: false },
        ];
        let mut seen = HashSet::new();
        let (fresh, stale, duplicate) = fresh_reports(&reports, 0, &mut seen);
        assert_eq!(fresh.len(), 1);
        assert_eq!(stale, 0);
        assert_eq!(duplicate, 1);
    }

    #[test]
    fn positive_and_negative_observations_move_cusum_in_expected_directions() {
        let mut statistic = 0.0;
        assert!(!update_cusum(&mut statistic, false, 0.02, 0.70));
        let after_negative = statistic;
        assert!(!update_cusum(&mut statistic, true, 0.02, 0.70));
        assert!(statistic > after_negative);
    }

    #[test]
    fn integrity_gate_fails_closed_on_common_mode_and_sybil_lanes() {
        for scenario in [Scenario::CommonMode, Scenario::SybilFanout] {
            let tape = make_tape(scenario, 0);
            let result = run_episode(scenario, Policy::IntegrityGate, &tape);
            assert_eq!(result.provenance_fallback_step, Some(3));
        }
    }

    #[test]
    fn integrity_gate_rejects_stale_safe_replay() {
        let tape = make_tape(Scenario::ReplaySafe, 0);
        let result = run_episode(Scenario::ReplaySafe, Policy::IntegrityGate, &tape);
        assert!(result.rejected_stale > 0);
        assert_eq!(result.provenance_fallback_step, Some(SHIFT_STEP + 3));
    }

    #[test]
    fn sybil_fanout_is_not_independent_evidence() {
        let tape = make_tape(Scenario::SybilFanout, 0);
        let naive = run_episode(Scenario::SybilFanout, Policy::ClaimSum, &tape);
        let dedup = run_episode(Scenario::SybilFanout, Policy::RootDedupSum, &tape);
        assert_eq!(naive.max_evidence_updates, 6);
        assert_eq!(dedup.max_evidence_updates, 1);
    }

    #[test]
    fn deterministic_tape_repeats_for_same_episode() {
        assert_eq!(make_tape(Scenario::OneByzantine, 22), make_tape(Scenario::OneByzantine, 22));
    }
}
