use std::collections::VecDeque;
use std::fmt::Write as _;

const SCHEMA: &str = "cmaq-013-selected-action-calibration-v0.1";
const EPISODES: usize = 512;
const DECISIONS_PER_EPISODE: usize = 48;
const CALIBRATION_EPISODES: usize = 512;
const CALIBRATION_STEPS: usize = 16;
const ALPHA: f64 = 0.10;
const TARGET_COVERAGE: f64 = 1.0 - ALPHA;
const SAFE_VALUE_THRESHOLD: f64 = 0.72;
const FEEDBACK_DELAY: usize = 4;
const FEEDBACK_DROP_RATE: f64 = 0.10;
const ROLLING_WINDOW: usize = 64;
const MIN_ROLLING_SAMPLES: usize = 16;
const ONLINE_STEP_SIZE: f64 = 0.035;
const GAUSSIAN_Z90: f64 = 1.644_853_626_951_472_2;
const NOMINAL_RESIDUAL_SCALE: f64 = 0.11;
const SEARCH_WIDTHS: [usize; 3] = [8, 32, 128];
const ROLLOUT_HORIZONS: [usize; 3] = [1, 3, 8];
const BASE_SEED: u64 = 0xC0A0_0013_2026_1030;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario { Stationary, AbruptShift, GradualShift }

impl Scenario {
    fn all() -> [Self; 3] { [Self::Stationary, Self::AbruptShift, Self::GradualShift] }
    fn name(self) -> &'static str {
        match self {
            Self::Stationary => "stationary",
            Self::AbruptShift => "abrupt_shift",
            Self::GradualShift => "gradual_shift",
        }
    }
    fn shift_start(self) -> Option<usize> {
        match self {
            Self::Stationary => None,
            Self::AbruptShift => Some(DECISIONS_PER_EPISODE / 2),
            Self::GradualShift => Some(DECISIONS_PER_EPISODE / 6),
        }
    }
    fn drift_multiplier(self, step: usize) -> f64 {
        match self {
            Self::Stationary => 1.0,
            Self::AbruptShift => if step < DECISIONS_PER_EPISODE / 2 { 1.0 } else { 2.2 },
            Self::GradualShift => {
                let start = DECISIONS_PER_EPISODE / 6;
                let end = DECISIONS_PER_EPISODE / 2;
                if step < start { 1.0 }
                else if step >= end { 2.2 }
                else {
                    let fraction = (step - start) as f64 / (end - start) as f64;
                    1.0 + fraction * 1.2
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Method { FixedGaussian, RandomSplit, SelectedSplit, OnlineAciStyle, RollingQuantile, OracleSelected }

impl Method {
    fn all() -> [Self; 6] {
        [Self::FixedGaussian, Self::RandomSplit, Self::SelectedSplit,
         Self::OnlineAciStyle, Self::RollingQuantile, Self::OracleSelected]
    }
    fn name(self) -> &'static str {
        match self {
            Self::FixedGaussian => "fixed_gaussian",
            Self::RandomSplit => "random_action_split_conformal",
            Self::SelectedSplit => "selected_action_split_conformal",
            Self::OnlineAciStyle => "online_aci_style_quantile",
            Self::RollingQuantile => "rolling_residual_quantile",
            Self::OracleSelected => "oracle_selected_quantile",
        }
    }
}

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
    fn index(&mut self, upper: usize) -> usize { (self.next_u64() % upper as u64) as usize }
}

fn normal(rng: &mut SplitMix64) -> f64 {
    let u1 = rng.unit().max(1e-12);
    let u2 = rng.unit();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ActionSample {
    predicted: f64,
    actual: f64,
    nonconformity: f64,
}

fn draw_action(rng: &mut SplitMix64, drift: f64, horizon: usize) -> ActionSample {
    // Candidate complexity is coupled to latent action quality; harder actions
    // have larger model error. Search selects on predicted value, not true value.
    let quality = rng.unit();
    let actual = 0.15 + 0.65 * quality;
    let sigma = (0.02 + 0.18 * quality) * drift * (horizon as f64).sqrt();
    let bias = (drift - 1.0) * 0.035;
    let predicted = actual + sigma * normal(rng) + bias;
    ActionSample { predicted, actual, nonconformity: predicted - actual }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Observation {
    selected: ActionSample,
    random_action: ActionSample,
    feedback_available: bool,
}

type EpisodeTape = Vec<Vec<Observation>>;

fn collect_calibration(width: usize, horizon: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = SplitMix64::new(seed ^ (width as u64).wrapping_mul(0xD134_2543_DE82_EF95) ^ (horizon as u64).rotate_left(23));
    let mut random_scores = Vec::with_capacity(CALIBRATION_EPISODES * CALIBRATION_STEPS);
    let mut selected_scores = Vec::with_capacity(CALIBRATION_EPISODES * CALIBRATION_STEPS);
    for _episode in 0..CALIBRATION_EPISODES {
        for _step in 0..CALIBRATION_STEPS {
            let mut best: Option<ActionSample> = None;
            for _ in 0..width {
                let candidate = draw_action(&mut rng, 1.0, horizon);
                if best.map(|current| candidate.predicted > current.predicted).unwrap_or(true) {
                    best = Some(candidate);
                }
            }
            selected_scores.push(best.unwrap().nonconformity);
            random_scores.push(draw_action(&mut rng, 1.0, horizon).nonconformity);
        }
    }
    (random_scores, selected_scores)
}

fn generate_evaluation(scenario: Scenario, width: usize, horizon: usize) -> EpisodeTape {
    let mut episodes = Vec::with_capacity(EPISODES);
    for episode in 0..EPISODES {
        let seed = BASE_SEED
            ^ ((scenario as u64) << 48)
            ^ (width as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (horizon as u64).rotate_left(29)
            ^ (episode as u64).wrapping_mul(0xD134_2543_DE82_EF95);
        let mut rng = SplitMix64::new(seed);
        let mut tape = Vec::with_capacity(DECISIONS_PER_EPISODE);
        for step in 0..DECISIONS_PER_EPISODE {
            let drift = scenario.drift_multiplier(step);
            let mut best: Option<ActionSample> = None;
            for _ in 0..width {
                let candidate = draw_action(&mut rng, drift, horizon);
                if best.map(|current| candidate.predicted > current.predicted).unwrap_or(true) {
                    best = Some(candidate);
                }
            }
            let random_action = draw_action(&mut rng, drift, horizon);
            let feedback_available = rng.unit() >= FEEDBACK_DROP_RATE;
            tape.push(Observation { selected: best.unwrap(), random_action, feedback_available });
        }
        episodes.push(tape);
    }
    episodes
}

fn quantile(values: &[f64], level: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let rank = ((sorted.len() as f64 + 1.0) * level).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn conformal_quantile(values: &[f64], level: f64) -> f64 {
    quantile(values, level).max(0.0)
}

#[derive(Clone, Debug)]
struct MethodSummary {
    method: &'static str,
    selected_action_coverage: f64,
    pre_shift_selected_coverage: f64,
    post_shift_selected_coverage: f64,
    random_action_coverage: f64,
    false_safe_rate: f64,
    fallback_rate: f64,
    mean_realized_utility: f64,
    mean_interval_width: f64,
    feedback_updates_per_episode: f64,
    recovery_episode_rate: f64,
    p90_recovery_delay: f64,
}

fn initial_q(method: Method, random_q: f64, selected_q: f64, oracle_q: f64) -> f64 {
    match method {
        Method::FixedGaussian => GAUSSIAN_Z90 * NOMINAL_RESIDUAL_SCALE,
        Method::RandomSplit => random_q,
        Method::SelectedSplit => selected_q,
        Method::OnlineAciStyle | Method::RollingQuantile => random_q,
        Method::OracleSelected => oracle_q,
    }
}

fn evaluate_method(
    method: Method,
    scenario: Scenario,
    episodes: &EpisodeTape,
    random_q: f64,
    selected_q: f64,
    oracle_q: f64,
) -> MethodSummary {
    let total = episodes.len() * DECISIONS_PER_EPISODE;
    let mut selected_covered = 0usize;
    let mut pre_shift_covered = 0usize;
    let mut pre_shift_count = 0usize;
    let mut post_shift_covered = 0usize;
    let mut post_shift_count = 0usize;
    let mut random_covered = 0usize;
    let mut accepted = 0usize;
    let mut false_safe = 0usize;
    let mut fallback = 0usize;
    let mut utility = 0.0;
    let mut interval_width_sum = 0.0;
    let mut feedback_updates = 0usize;
    let mut recovery_delays = Vec::new();
    let mut recovery_eligible = 0usize;
    let shift_start = scenario.shift_start();

    for tape in episodes {
        let mut q = initial_q(method, random_q, selected_q, oracle_q);
        let mut pending: VecDeque<(usize, f64)> = VecDeque::new();
        let mut rolling: VecDeque<f64> = VecDeque::new();
        let mut coverage_window: VecDeque<bool> = VecDeque::new();
        let mut episode_recovery: Option<usize> = None;

        for (step, observation) in tape.iter().enumerate() {
            while pending.front().map(|(due, _)| *due <= step).unwrap_or(false) {
                let (_, score) = pending.pop_front().unwrap();
                feedback_updates += 1;
                if method == Method::OnlineAciStyle {
                    let indicator = if score <= q { 1.0 } else { 0.0 };
                    q = (q + ONLINE_STEP_SIZE * (TARGET_COVERAGE - indicator)).clamp(0.0, 2.0);
                } else if method == Method::RollingQuantile {
                    rolling.push_back(score);
                    while rolling.len() > ROLLING_WINDOW { rolling.pop_front(); }
                    if rolling.len() >= MIN_ROLLING_SAMPLES {
                        let values: Vec<f64> = rolling.iter().copied().collect();
                        q = conformal_quantile(&values, TARGET_COVERAGE);
                    }
                }
            }

            let selected = observation.selected;
            let random = observation.random_action;
            let lower_bound = selected.predicted - q;
            let covered = selected.actual >= lower_bound;
            let random_lower_bound = random.predicted - q;
            let random_is_covered = random.actual >= random_lower_bound;
            selected_covered += if covered { 1 } else { 0 };
            if let Some(start) = shift_start {
                if step < start {
                    pre_shift_count += 1;
                    pre_shift_covered += if covered { 1 } else { 0 };
                } else {
                    post_shift_count += 1;
                    post_shift_covered += if covered { 1 } else { 0 };
                }
            } else {
                pre_shift_count += 1;
                pre_shift_covered += if covered { 1 } else { 0 };
            }
            random_covered += if random_is_covered { 1 } else { 0 };
            interval_width_sum += q;

            let take_action = lower_bound >= SAFE_VALUE_THRESHOLD;
            if take_action {
                accepted += 1;
                if selected.actual < SAFE_VALUE_THRESHOLD { false_safe += 1; }
                utility += selected.actual - SAFE_VALUE_THRESHOLD;
            } else {
                fallback += 1;
            }

            coverage_window.push_back(covered);
            while coverage_window.len() > 16 { coverage_window.pop_front(); }
            if let Some(start) = shift_start {
                if step >= start && coverage_window.len() == 16 && episode_recovery.is_none() {
                    let coverage = coverage_window.iter().filter(|value| **value).count() as f64 / 16.0;
                    if coverage >= TARGET_COVERAGE { episode_recovery = Some(step - start); }
                }
            }

            if observation.feedback_available && (method == Method::OnlineAciStyle || method == Method::RollingQuantile) {
                pending.push_back((step + FEEDBACK_DELAY, selected.nonconformity));
            }
        }
        if shift_start.is_some() {
            recovery_eligible += 1;
            if let Some(delay) = episode_recovery { recovery_delays.push(delay as f64); }
        }
    }

    recovery_delays.sort_by(|a, b| a.total_cmp(b));
    let p90_recovery_delay = if recovery_delays.is_empty() {
        -1.0
    } else {
        quantile(&recovery_delays, 0.90)
    };
    MethodSummary {
        method: method.name(),
        selected_action_coverage: selected_covered as f64 / total as f64,
        pre_shift_selected_coverage: if pre_shift_count > 0 { pre_shift_covered as f64 / pre_shift_count as f64 } else { -1.0 },
        post_shift_selected_coverage: if post_shift_count > 0 { post_shift_covered as f64 / post_shift_count as f64 } else { -1.0 },
        random_action_coverage: random_covered as f64 / total as f64,
        false_safe_rate: if accepted > 0 { false_safe as f64 / accepted as f64 } else { 0.0 },
        fallback_rate: fallback as f64 / total as f64,
        mean_realized_utility: utility / total as f64,
        mean_interval_width: interval_width_sum / total as f64,
        feedback_updates_per_episode: feedback_updates as f64 / episodes.len() as f64,
        recovery_episode_rate: if recovery_eligible > 0 { recovery_delays.len() as f64 / recovery_eligible as f64 } else { 0.0 },
        p90_recovery_delay,
    }
}

#[derive(Clone, Debug)]
struct CellReport {
    scenario: &'static str,
    search_width: usize,
    rollout_horizon: usize,
    episode_count: usize,
    decisions_per_episode: usize,
    calibration_samples: usize,
    random_calibration_quantile: f64,
    selected_calibration_quantile: f64,
    oracle_selected_quantile: f64,
    methods: Vec<MethodSummary>,
}

fn run_benchmark() -> Vec<CellReport> {
    let mut cells = Vec::new();
    for width in SEARCH_WIDTHS {
        for horizon in ROLLOUT_HORIZONS {
            let (random_calibration, selected_calibration) = collect_calibration(width, horizon, BASE_SEED ^ 0xCA11_BA7E);
            let random_q = conformal_quantile(&random_calibration, TARGET_COVERAGE);
            let selected_q = conformal_quantile(&selected_calibration, TARGET_COVERAGE);
            for scenario in Scenario::all() {
                let episodes = generate_evaluation(scenario, width, horizon);
                let oracle_scores: Vec<f64> = episodes.iter().flat_map(|tape| tape.iter().map(|obs| obs.selected.nonconformity)).collect();
                let oracle_q = conformal_quantile(&oracle_scores, TARGET_COVERAGE);
                let methods = Method::all().iter().map(|method| {
                    evaluate_method(*method, scenario, &episodes, random_q, selected_q, oracle_q)
                }).collect();
                cells.push(CellReport {
                    scenario: scenario.name(), search_width: width, rollout_horizon: horizon,
                    episode_count: EPISODES, decisions_per_episode: DECISIONS_PER_EPISODE,
                    calibration_samples: random_calibration.len(),
                    random_calibration_quantile: random_q,
                    selected_calibration_quantile: selected_q,
                    oracle_selected_quantile: oracle_q,
                    methods,
                });
            }
        }
    }
    cells
}

fn number(value: f64) -> String {
    if value.is_finite() { format!("{value:.8}") } else { "null".to_string() }
}

fn render_report(cells: &[CellReport]) -> String {
    let mut out = String::new();
    writeln!(&mut out, "{{").unwrap();
    writeln!(&mut out, "  \"schema_version\": \"{SCHEMA}\",").unwrap();
    writeln!(&mut out, "  \"episode_count\": {EPISODES},").unwrap();
    writeln!(&mut out, "  \"decisions_per_episode\": {DECISIONS_PER_EPISODE},").unwrap();
    writeln!(&mut out, "  \"alpha\": {},", number(ALPHA)).unwrap();
    writeln!(&mut out, "  \"safe_value_threshold\": {},", number(SAFE_VALUE_THRESHOLD)).unwrap();
    writeln!(&mut out, "  \"feedback_delay_steps\": {FEEDBACK_DELAY},").unwrap();
    writeln!(&mut out, "  \"feedback_drop_rate\": {},", number(FEEDBACK_DROP_RATE)).unwrap();
    writeln!(&mut out, "  \"rolling_window\": {ROLLING_WINDOW},").unwrap();
    writeln!(&mut out, "  \"online_step_size\": {},", number(ONLINE_STEP_SIZE)).unwrap();
    writeln!(&mut out, "  \"search_widths\": [8,32,128],").unwrap();
    writeln!(&mut out, "  \"rollout_horizons\": [1,3,8],").unwrap();
    writeln!(&mut out, "  \"scenarios\": [\"stationary\",\"abrupt_shift\",\"gradual_shift\"],").unwrap();
    writeln!(&mut out, "  \"methods\": [\"fixed_gaussian\",\"random_action_split_conformal\",\"selected_action_split_conformal\",\"online_aci_style_quantile\",\"rolling_residual_quantile\",\"oracle_selected_quantile\"],").unwrap();
    writeln!(&mut out, "  \"cells\": [").unwrap();
    for (cell_index, cell) in cells.iter().enumerate() {
        let comma = if cell_index + 1 == cells.len() { "" } else { "," };
        writeln!(&mut out, "    {{").unwrap();
        writeln!(&mut out, "      \"scenario\": \"{}\",", cell.scenario).unwrap();
        writeln!(&mut out, "      \"search_width\": {},", cell.search_width).unwrap();
        writeln!(&mut out, "      \"rollout_horizon\": {},", cell.rollout_horizon).unwrap();
        writeln!(&mut out, "      \"episode_count\": {},", cell.episode_count).unwrap();
        writeln!(&mut out, "      \"decisions_per_episode\": {},", cell.decisions_per_episode).unwrap();
        writeln!(&mut out, "      \"calibration_samples\": {},", cell.calibration_samples).unwrap();
        writeln!(&mut out, "      \"random_calibration_quantile\": {},", number(cell.random_calibration_quantile)).unwrap();
        writeln!(&mut out, "      \"selected_calibration_quantile\": {},", number(cell.selected_calibration_quantile)).unwrap();
        writeln!(&mut out, "      \"oracle_selected_quantile\": {},", number(cell.oracle_selected_quantile)).unwrap();
        writeln!(&mut out, "      \"methods\": [").unwrap();
        for (method_index, method) in cell.methods.iter().enumerate() {
            let method_comma = if method_index + 1 == cell.methods.len() { "" } else { "," };
            writeln!(&mut out,
                concat!(
                    "        {{\"method\":\"{}\",\"selected_action_coverage\":{},",
                    "\"random_action_coverage\":{},\"pre_shift_selected_coverage\":{},\"post_shift_selected_coverage\":{},",
                    "\"false_safe_rate\":{},\"fallback_rate\":{},",
                    "\"mean_realized_utility\":{},\"mean_interval_width\":{},",
                    "\"feedback_updates_per_episode\":{},\"recovery_episode_rate\":{},",
                    "\"p90_recovery_delay\":{}}}{}"
                ),
                method.method, number(method.selected_action_coverage),
                number(method.random_action_coverage), number(method.pre_shift_selected_coverage),
                number(method.post_shift_selected_coverage), number(method.false_safe_rate),
                number(method.fallback_rate), number(method.mean_realized_utility),
                number(method.mean_interval_width), number(method.feedback_updates_per_episode),
                number(method.recovery_episode_rate), number(method.p90_recovery_delay), method_comma
            ).unwrap();
        }
        writeln!(&mut out, "      ]").unwrap();
        writeln!(&mut out, "    }}{comma}").unwrap();
    }
    writeln!(&mut out, "  ]").unwrap();
    writeln!(&mut out, "}}").unwrap();
    out
}

fn main() { print!("{}", render_report(&run_benchmark())); }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_conditioned_calibration_is_wider_than_random_at_large_search() {
        let (random, selected) = collect_calibration(128, 8, BASE_SEED ^ 0xCA11_BA7E);
        assert!(conformal_quantile(&selected, TARGET_COVERAGE) > conformal_quantile(&random, TARGET_COVERAGE));
    }

    #[test]
    fn selected_calibration_widens_with_candidate_search() {
        let (_, small) = collect_calibration(8, 1, BASE_SEED ^ 0xCA11_BA7E);
        let (_, large) = collect_calibration(128, 8, BASE_SEED ^ 0xCA11_BA7E);
        assert!(conformal_quantile(&large, TARGET_COVERAGE) > conformal_quantile(&small, TARGET_COVERAGE));
    }

    #[test]
    fn selected_residual_quantile_grows_with_rollout_horizon() {
        let (_, short) = collect_calibration(32, 1, BASE_SEED ^ 0xCA11_BA7E);
        let (_, long) = collect_calibration(32, 8, BASE_SEED ^ 0xCA11_BA7E);
        assert!(conformal_quantile(&long, TARGET_COVERAGE) > conformal_quantile(&short, TARGET_COVERAGE));
    }

    #[test]
    fn coverage_is_reported_separately_before_and_after_shift() {
        let (random, selected) = collect_calibration(32, 3, 44);
        let stationary = generate_evaluation(Scenario::Stationary, 32, 3);
        let scores: Vec<f64> = stationary.iter().flat_map(|t| t.iter().map(|o| o.selected.nonconformity)).collect();
        let oracle = conformal_quantile(&scores, TARGET_COVERAGE);
        let stationary_method = evaluate_method(Method::SelectedSplit, Scenario::Stationary, &stationary,
            conformal_quantile(&random, TARGET_COVERAGE), conformal_quantile(&selected, TARGET_COVERAGE), oracle);
        assert!(stationary_method.pre_shift_selected_coverage >= 0.0);
        assert_eq!(stationary_method.post_shift_selected_coverage, -1.0);

        let shifted = generate_evaluation(Scenario::AbruptShift, 32, 3);
        let scores: Vec<f64> = shifted.iter().flat_map(|t| t.iter().map(|o| o.selected.nonconformity)).collect();
        let oracle = conformal_quantile(&scores, TARGET_COVERAGE);
        let shifted_method = evaluate_method(Method::RandomSplit, Scenario::AbruptShift, &shifted,
            conformal_quantile(&random, TARGET_COVERAGE), conformal_quantile(&selected, TARGET_COVERAGE), oracle);
        assert!(shifted_method.pre_shift_selected_coverage >= 0.0);
        assert!(shifted_method.post_shift_selected_coverage >= 0.0);
        assert!(shifted_method.post_shift_selected_coverage <= 1.0);
    }

    #[test]
    fn finite_sample_conformal_quantile_uses_upper_order_statistic() {
        let values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(conformal_quantile(&values, 0.90), 8.0);
    }

    #[test]
    fn deterministic_tapes_repeat() {
        assert_eq!(generate_evaluation(Scenario::AbruptShift, 32, 3), generate_evaluation(Scenario::AbruptShift, 32, 3));
    }

    #[test]
    fn feedback_delay_and_drops_bound_online_updates() {
        let (random, selected) = collect_calibration(32, 3, 44);
        let episodes = generate_evaluation(Scenario::AbruptShift, 32, 3);
        let scores: Vec<f64> = episodes.iter().flat_map(|t| t.iter().map(|o| o.selected.nonconformity)).collect();
        let oracle = conformal_quantile(&scores, TARGET_COVERAGE);
        let online = evaluate_method(Method::OnlineAciStyle, Scenario::AbruptShift, &episodes,
            conformal_quantile(&random, TARGET_COVERAGE), conformal_quantile(&selected, TARGET_COVERAGE), oracle);
        assert!(online.feedback_updates_per_episode > 0.0);
        assert!(online.feedback_updates_per_episode < DECISIONS_PER_EPISODE as f64);
    }

    #[test]
    fn rolling_quantile_consumes_delayed_selected_residuals() {
        let (random, selected) = collect_calibration(8, 1, 55);
        let episodes = generate_evaluation(Scenario::GradualShift, 8, 1);
        let scores: Vec<f64> = episodes.iter().flat_map(|t| t.iter().map(|o| o.selected.nonconformity)).collect();
        let oracle = conformal_quantile(&scores, TARGET_COVERAGE);
        let rolling = evaluate_method(Method::RollingQuantile, Scenario::GradualShift, &episodes,
            conformal_quantile(&random, TARGET_COVERAGE), conformal_quantile(&selected, TARGET_COVERAGE), oracle);
        assert!(rolling.feedback_updates_per_episode > 0.0);
        assert!(rolling.mean_interval_width.is_finite());
    }

    #[test]
    fn benchmark_has_all_cells_and_calibration_controls_are_measurable() {
        let cells = run_benchmark();
        assert_eq!(cells.len(), 27);
        assert!(cells.iter().all(|cell| cell.methods.len() == 6));
        assert!(cells.iter().all(|cell| cell.episode_count == 512));

        let stationary = cells.iter().find(|c| c.scenario == "stationary" && c.search_width == 32 && c.rollout_horizon == 3).unwrap();
        let selected = stationary.methods.iter().find(|m| m.method == "selected_action_split_conformal").unwrap();
        assert!(selected.selected_action_coverage >= 0.86);
        assert!(selected.selected_action_coverage <= 0.96);

        let shifted = cells.iter().find(|c| c.scenario == "abrupt_shift" && c.search_width == 128 && c.rollout_horizon == 8).unwrap();
        let oracle = shifted.methods.iter().find(|m| m.method == "oracle_selected_quantile").unwrap();
        assert!(oracle.selected_action_coverage >= 0.88);
    }
}
