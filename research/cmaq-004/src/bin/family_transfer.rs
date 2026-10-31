//! CMAQ-004 family-level transfer panel.
//!
//! Candidate choice uses only support outcomes from contexts 00, 01 and 10.
//! The evaluator alone knows the held-out 11 outcome law. This is still a public
//! synthetic smoke generator, not a sealed benchmark or a real Symthaea adapter.

use std::process;

const REPORT_SCHEMA: &str = "cmaq-004-family-transfer-v0.3";
const FAMILY_COUNT: usize = 256;
const SUPPORT_SAMPLES: usize = 256;
const BASE_SEED: u64 = 0xC0A4_F411_1E5E_2026;
const QUERY_COSTS: [f64; 4] = [0.0, 0.02, 0.04, 0.08];
const MODEL_QUERIES_PER_DECISION: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lane {
    Additive,
    Interaction,
    HoldoutShift,
}

impl Lane {
    const ALL: [Self; 3] = [Self::Additive, Self::Interaction, Self::HoldoutShift];

    fn name(self) -> &'static str {
        match self {
            Self::Additive => "additive",
            Self::Interaction => "interaction",
            Self::HoldoutShift => "holdout_shift",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FamilySpec {
    lane: Lane,
    intercept: f64,
    x_effect: f64,
    y_effect: f64,
    interaction: f64,
    holdout_shift: f64,
}

impl FamilySpec {
    fn draw(index: usize) -> Self {
        let lane = Lane::ALL[index % Lane::ALL.len()];
        let seed = BASE_SEED ^ (index as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
        let mut rng = SplitMix64::new(seed);
        let signed = |rng: &mut SplitMix64, radius: f64| (2.0 * rng.unit() - 1.0) * radius;
        let intercept = 0.28 + 0.16 * rng.unit();
        let x_effect = signed(&mut rng, 0.18);
        let y_effect = signed(&mut rng, 0.18);
        let interaction = match lane {
            Lane::Additive => signed(&mut rng, 0.02),
            Lane::Interaction => {
                let sign = if rng.unit() < 0.5 { -1.0 } else { 1.0 };
                sign * (0.14 + 0.18 * rng.unit())
            }
            Lane::HoldoutShift => 0.0,
        };
        let holdout_shift = if lane == Lane::HoldoutShift {
            signed(&mut rng, 0.05)
        } else {
            0.0
        };
        Self {
            lane,
            intercept,
            x_effect,
            y_effect,
            interaction,
            holdout_shift,
        }
    }

    fn support_action_one_probability(self, context: usize) -> f64 {
        let (x, y) = context_factors(context);
        clamp_probability(
            self.intercept + self.x_effect * x + self.y_effect * y
                + self.interaction * x * y,
        )
    }

    fn heldout_action_one_probability(self) -> f64 {
        match self.lane {
            Lane::Additive | Lane::Interaction => clamp_probability(
                self.intercept + self.x_effect + self.y_effect + self.interaction,
            ),
            // This is a deliberately shifted held-out context. It tests transfer failure;
            // it is not claimed to be the stationary law of the support contexts.
            Lane::HoldoutShift => clamp_probability(
                self.intercept - self.x_effect - self.y_effect + self.holdout_shift,
            ),
        }
    }
}

fn context_factors(context: usize) -> (f64, f64) {
    ((context / 2) as f64, (context % 2) as f64)
}

fn clamp_probability(value: f64) -> f64 {
    value.clamp(0.01, 0.99)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SupportMeans {
    // Intentionally only three observed context rows: 00, 01 and 10.
    // There is no slot for held-out context 11.
    by_context_action: [[f64; 2]; 3],
}

impl SupportMeans {
    fn fit(spec: FamilySpec, family_index: usize) -> Self {
        let mut means = [[0.0_f64; 2]; 3];
        for context in 0..3 {
            for action in 0..2 {
                let probability = if action == 0 {
                    0.5
                } else {
                    spec.support_action_one_probability(context)
                };
                // Sampling noise is keyed by the family slot and observed cell only,
                // never by the hidden interaction/held-out-shift parameters.
                let seed = BASE_SEED
                    ^ (family_index as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
                    ^ ((context as u64 + 1).wrapping_mul(0x94D0_49BB))
                    ^ ((action as u64 + 1).wrapping_mul(0xBF58_476D));
                let mut rng = SplitMix64::new(seed);
                let successes = (0..SUPPORT_SAMPLES)
                    .filter(|_| rng.unit() < probability)
                    .count();
                means[context][action] = successes as f64 / SUPPORT_SAMPLES as f64;
            }
        }
        Self { by_context_action: means }
    }

    fn direct_nearest_context(self) -> [f64; 2] {
        // Direct non-compositional comparator: average the two observed neighboring
        // contexts (01 and 10). Both policies use the same support outcomes.
        [
            (self.by_context_action[1][0] + self.by_context_action[2][0]) / 2.0,
            (self.by_context_action[1][1] + self.by_context_action[2][1]) / 2.0,
        ]
    }

    fn additive_extrapolation(self) -> [f64; 2] {
        [
            clamp_probability(
                self.by_context_action[1][0] + self.by_context_action[2][0]
                    - self.by_context_action[0][0],
            ),
            clamp_probability(
                self.by_context_action[1][1] + self.by_context_action[2][1]
                    - self.by_context_action[0][1],
            ),
        ]
    }
}

fn choose(values: [f64; 2]) -> usize {
    if values[1] > values[0] { 1 } else { 0 }
}

fn expected_action_reward(spec: FamilySpec, action: usize) -> f64 {
    if action == 0 { 0.5 } else { spec.heldout_action_one_probability() }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Strategy {
    DirectNearest,
    AdditiveModel,
    MarginGate005,
    MarginGate020,
    MarginGate050,
    ShuffledModel,
    StaleModel,
    Oracle,
}

impl Strategy {
    const ALL: [Self; 8] = [
        Self::DirectNearest,
        Self::AdditiveModel,
        Self::MarginGate005,
        Self::MarginGate020,
        Self::MarginGate050,
        Self::ShuffledModel,
        Self::StaleModel,
        Self::Oracle,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::DirectNearest => "direct_nearest_context",
            Self::AdditiveModel => "additive_model",
            Self::MarginGate005 => "margin_gate_005",
            Self::MarginGate020 => "margin_gate_020",
            Self::MarginGate050 => "margin_gate_050",
            Self::ShuffledModel => "shuffled_model_control",
            Self::StaleModel => "stale_model_control",
            Self::Oracle => "oracle_evaluator_only",
        }
    }

    fn uses_model(self) -> bool {
        matches!(self, Self::AdditiveModel | Self::MarginGate005 | Self::MarginGate020 | Self::MarginGate050 | Self::ShuffledModel | Self::StaleModel)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FamilyOutcome {
    absolute_net_utility: f64,
    gross_lift_vs_direct: f64,
    net_lift_vs_direct: f64,
    action: usize,
    model_abs_error: f64,
    action_override: bool,
}

fn score_family(spec: FamilySpec, support: SupportMeans, strategy: Strategy, query_cost: f64) -> FamilyOutcome {
    let additive = support.additive_extrapolation();
    let direct = choose(support.direct_nearest_context());
    let proposed_additive_action = choose(additive);
    let margin_gate = |threshold: f64| {
        let estimated_lift = additive[proposed_additive_action] - additive[direct];
        if proposed_additive_action != direct && estimated_lift >= threshold {
            proposed_additive_action
        } else {
            direct
        }
    };
    let action = match strategy {
        Strategy::DirectNearest => direct,
        Strategy::AdditiveModel => proposed_additive_action,
        Strategy::MarginGate005 => margin_gate(0.005),
        Strategy::MarginGate020 => margin_gate(0.020),
        Strategy::MarginGate050 => margin_gate(0.050),
        Strategy::ShuffledModel => choose([additive[1], additive[0]]),
        Strategy::StaleModel => choose([0.5, 0.75]),
        Strategy::Oracle => {
            if spec.heldout_action_one_probability() > 0.5 { 1 } else { 0 }
        }
    };
    let cost = if strategy.uses_model() {
        query_cost * MODEL_QUERIES_PER_DECISION as f64
    } else {
        0.0
    };
    let direct_reward = expected_action_reward(spec, direct);
    let candidate_reward = expected_action_reward(spec, action);
    let gross_lift = candidate_reward - direct_reward;
    let net_utility = candidate_reward - cost;
    let model_abs_error = match strategy {
        Strategy::AdditiveModel | Strategy::MarginGate005 | Strategy::MarginGate020 | Strategy::MarginGate050 | Strategy::ShuffledModel => {
            let truth = [0.5, spec.heldout_action_one_probability()];
            ((additive[0] - truth[0]).abs() + (additive[1] - truth[1]).abs()) / 2.0
        }
        Strategy::StaleModel => {
            (0.75_f64 - spec.heldout_action_one_probability()).abs() / 2.0
        }
        Strategy::DirectNearest | Strategy::Oracle => 0.0,
    };
    FamilyOutcome {
        absolute_net_utility: net_utility,
        gross_lift_vs_direct: gross_lift,
        net_lift_vs_direct: gross_lift - cost,
        action,
        model_abs_error,
        action_override: action != direct,
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() { 0.0 } else { values.iter().sum::<f64>() / values.len() as f64 }
}

fn mean_ci95(values: &[f64]) -> (f64, f64, f64) {
    let avg = mean(values);
    if values.len() < 2 { return (avg, avg, avg); }
    let variance = values.iter().map(|value| (value - avg).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    let half = 1.96 * (variance / values.len() as f64).sqrt();
    (avg, avg - half, avg + half)
}

/// Mean of the lowest ceil(5% * n) family-level net lifts.
fn lower_cvar05(values: &[f64]) -> f64 {
    if values.is_empty() { return 0.0; }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let tail = ((sorted.len() as f64 * 0.05).ceil() as usize).max(1);
    mean(&sorted[..tail])
}

#[derive(Clone, Debug)]
struct Summary {
    lane: &'static str,
    strategy: &'static str,
    query_cost_per_query: f64,
    family_count: usize,
    mean_net_utility: f64,
    mean_gross_lift_vs_direct: f64,
    mean_break_even_query_cost: f64,
    beneficial_family_rate_before_cost: f64,
    mean_net_lift_vs_direct: f64,
    lift_ci95_low: f64,
    lift_ci95_high: f64,
    lower_cvar05_net_lift: f64,
    minimum_family_net_lift: f64,
    harmed_family_rate: f64,
    mean_additive_prediction_mae: f64,
    action_one_rate: f64,
    action_override_rate: f64,
}

fn summarize(
    lane: &'static str,
    query_cost: f64,
    strategy: Strategy,
    outcomes: &[(FamilySpec, FamilyOutcome)],
) -> Summary {
    let lifts = outcomes.iter().map(|(_, outcome)| outcome.net_lift_vs_direct).collect::<Vec<_>>();
    let gross_lifts = outcomes.iter().map(|(_, outcome)| outcome.gross_lift_vs_direct).collect::<Vec<_>>();
    let utilities = outcomes.iter().map(|(_, outcome)| outcome.absolute_net_utility).collect::<Vec<_>>();
    let errors = outcomes.iter().map(|(_, outcome)| outcome.model_abs_error).collect::<Vec<_>>();
    let actions = outcomes.iter().map(|(_, outcome)| outcome.action as f64).collect::<Vec<_>>();
    let override_values = outcomes.iter().map(|(_, outcome)| outcome.action_override as u8 as f64).collect::<Vec<_>>();
    let (avg, low, high) = mean_ci95(&lifts);
    Summary {
        lane,
        strategy: strategy.name(),
        query_cost_per_query: query_cost,
        family_count: outcomes.len(),
        mean_net_utility: mean(&utilities),
        mean_gross_lift_vs_direct: mean(&gross_lifts),
        mean_break_even_query_cost: if strategy.uses_model() { mean(&gross_lifts) / MODEL_QUERIES_PER_DECISION as f64 } else { 0.0 },
        beneficial_family_rate_before_cost: if gross_lifts.is_empty() { 0.0 } else { gross_lifts.iter().filter(|value| **value > 1e-12).count() as f64 / gross_lifts.len() as f64 },
        mean_net_lift_vs_direct: avg,
        lift_ci95_low: low,
        lift_ci95_high: high,
        lower_cvar05_net_lift: lower_cvar05(&lifts),
        minimum_family_net_lift: lifts.iter().copied().fold(f64::INFINITY, f64::min),
        harmed_family_rate: if lifts.is_empty() { 0.0 } else { lifts.iter().filter(|value| **value < -1e-12).count() as f64 / lifts.len() as f64 },
        mean_additive_prediction_mae: if matches!(strategy, Strategy::AdditiveModel | Strategy::MarginGate005 | Strategy::MarginGate020 | Strategy::MarginGate050 | Strategy::ShuffledModel) { mean(&errors) } else { 0.0 },
        action_one_rate: mean(&actions),
        action_override_rate: mean(&override_values),
    }
}

fn run() -> Vec<Summary> {
    let families = (0..FAMILY_COUNT).map(FamilySpec::draw).collect::<Vec<_>>();
    let mut rows = Vec::new();
    for cost in QUERY_COSTS {
        for strategy in Strategy::ALL {
            for lane in [None, Some(Lane::Additive), Some(Lane::Interaction), Some(Lane::HoldoutShift)] {
                let outcomes = families.iter().copied()
                    .enumerate()
                    .filter(|(_, spec)| lane.map(|selected| spec.lane == selected).unwrap_or(true))
                    .map(|(family_index, spec)| {
                        (spec, score_family(spec, SupportMeans::fit(spec, family_index), strategy, cost))
                    })
                    .collect::<Vec<_>>();
                let lane_name = lane.map(Lane::name).unwrap_or("all");
                rows.push(summarize(lane_name, cost, strategy, &outcomes));
            }
        }
    }
    rows
}

fn render(rows: &[Summary]) -> String {
    let mut out = format!(
        concat!(
            "{{\n",
            "  \"schema_version\": \"{}\",\n",
            "  \"family_count\": {},\n",
            "  \"support_samples_per_context_action\": {},\n",
            "  \"support_contexts\": [0, 1, 2],\n",
            "  \"heldout_context\": 3,\n",
            "  \"query_cost_levels\": [0.00, 0.02, 0.04, 0.08],\n",
            "  \"model_queries_per_decision\": {},\n",
            "  \"summary\": [\n"
        ),
        REPORT_SCHEMA, FAMILY_COUNT, SUPPORT_SAMPLES, MODEL_QUERIES_PER_DECISION
    );
    for (i, row) in rows.iter().enumerate() {
        out.push_str(&format!(
            concat!(
                "    {{\"lane\":\"{}\",",
                "\"strategy\":\"{}\",",
                "\"query_cost_per_query\":{:.4},",
                "\"family_count\":{},",
                "\"mean_net_utility\":{:.8},",
                "\"mean_gross_lift_vs_direct\":{:.8},",
                "\"mean_break_even_query_cost\":{:.8},",
                "\"beneficial_family_rate_before_cost\":{:.8},",
                "\"mean_net_lift_vs_direct\":{:.8},",
                "\"lift_ci95_low\":{:.8},",
                "\"lift_ci95_high\":{:.8},",
                "\"lower_cvar05_net_lift\":{:.8},",
                "\"minimum_family_net_lift\":{:.8},",
                "\"harmed_family_rate\":{:.8},",
                "\"mean_additive_prediction_mae\":{:.8},",
                "\"action_one_rate\":{:.8},",
                "\"action_override_rate\":{:.8}",
                "}}"
            ),
            row.lane, row.strategy, row.query_cost_per_query, row.family_count,
            row.mean_net_utility, row.mean_gross_lift_vs_direct, row.mean_break_even_query_cost,
            row.beneficial_family_rate_before_cost, row.mean_net_lift_vs_direct, row.lift_ci95_low,
            row.lift_ci95_high, row.lower_cvar05_net_lift, row.minimum_family_net_lift,
            row.harmed_family_rate, row.mean_additive_prediction_mae, row.action_one_rate,
            row.action_override_rate
        ));
        if i + 1 != rows.len() { out.push(','); }
        out.push('\n');
    }
    out.push_str("  ]\n}\n");
    out
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("CMAQ-004 family-level held-out transfer smoke panel. {} generated families; outputs family-level net-lift intervals and lower-tail risk.", FAMILY_COUNT);
        return;
    }
    if !args.is_empty() {
        eprintln!("unexpected argument {}; use --help", args[0]);
        process::exit(2);
    }
    print!("{}", render(&run()));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clean_support(spec: FamilySpec) -> SupportMeans {
        let action_zero = 0.5;
        let mut rows = [[action_zero; 2]; 3];
        for context in 0..3 {
            rows[context][1] = spec.support_action_one_probability(context);
        }
        SupportMeans { by_context_action: rows }
    }

    #[test]
    fn stricter_margin_gate_never_overrides_direct_more_often() {
        let mut takeovers = [0_usize; 3];
        for index in 0..FAMILY_COUNT {
            let spec = FamilySpec::draw(index);
            let support = SupportMeans::fit(spec, index);
            for (gate_index, strategy) in [
                Strategy::MarginGate005,
                Strategy::MarginGate020,
                Strategy::MarginGate050,
            ]
            .into_iter()
            .enumerate()
            {
                let outcome = score_family(spec, support, strategy, 0.0);
                takeovers[gate_index] += outcome.action_override as usize;
            }
        }
        assert!(takeovers[0] >= takeovers[1]);
        assert!(takeovers[1] >= takeovers[2]);
    }

    #[test]
    fn margin_gate_still_pays_model_inference_cost() {
        let spec = FamilySpec::draw(17);
        let support = SupportMeans::fit(spec, 17);
        for strategy in [
            Strategy::MarginGate005,
            Strategy::MarginGate020,
            Strategy::MarginGate050,
        ] {
            let free = score_family(spec, support, strategy, 0.0);
            let priced = score_family(spec, support, strategy, 0.04);
            assert!((free.gross_lift_vs_direct - priced.gross_lift_vs_direct).abs() < 1e-12);
            assert!((free.net_lift_vs_direct - priced.net_lift_vs_direct - 0.08).abs() < 1e-12);
        }
    }

    #[test]
    fn generated_family_and_support_tape_are_deterministic() {
        for i in 0..FAMILY_COUNT {
            assert_eq!(FamilySpec::draw(i), FamilySpec::draw(i));
            assert_eq!(SupportMeans::fit(FamilySpec::draw(i), i), SupportMeans::fit(FamilySpec::draw(i), i));
        }
    }

    #[test]
    fn additive_extrapolation_is_exact_when_the_heldout_law_is_additive() {
        let spec = FamilySpec {
            lane: Lane::Additive,
            intercept: 0.31,
            x_effect: 0.08,
            y_effect: -0.02,
            interaction: 0.0,
            holdout_shift: 0.0,
        };
        let predicted = clean_support(spec).additive_extrapolation();
        assert!((predicted[1] - spec.heldout_action_one_probability()).abs() < 1e-12);
    }

    #[test]
    fn support_sampling_does_not_depend_on_heldout_only_interaction_or_shift() {
        let additive = FamilySpec {
            lane: Lane::Additive,
            intercept: 0.35,
            x_effect: 0.05,
            y_effect: -0.04,
            interaction: 0.0,
            holdout_shift: 0.0,
        };
        let interaction = FamilySpec {
            lane: Lane::Interaction,
            interaction: -0.27,
            ..additive
        };
        let shifted = FamilySpec {
            lane: Lane::HoldoutShift,
            holdout_shift: 0.08,
            ..additive
        };
        assert_eq!(SupportMeans::fit(additive, 77), SupportMeans::fit(interaction, 77));
        assert_eq!(SupportMeans::fit(additive, 77), SupportMeans::fit(shifted, 77));
    }

    #[test]
    fn negative_interaction_can_make_additive_extrapolation_choose_the_wrong_action() {
        let spec = FamilySpec {
            lane: Lane::Interaction,
            intercept: 0.35,
            x_effect: 0.15,
            y_effect: 0.15,
            interaction: -0.25,
            holdout_shift: 0.0,
        };
        let support = clean_support(spec);
        assert!(support.additive_extrapolation()[1] > 0.5);
        assert!(spec.heldout_action_one_probability() < 0.5);
        let model = score_family(spec, support, Strategy::AdditiveModel, 0.0);
        let oracle = score_family(spec, support, Strategy::Oracle, 0.0);
        assert_eq!(model.action, 1);
        assert_eq!(oracle.action, 0);
        assert!(model.net_lift_vs_direct < 0.0);
    }

    #[test]
    fn gross_lift_and_query_price_frontier_are_separated() {
        let spec = FamilySpec::draw(0);
        let support = SupportMeans::fit(spec, 0);
        let free = score_family(spec, support, Strategy::AdditiveModel, 0.0);
        let priced = score_family(spec, support, Strategy::AdditiveModel, 0.04);
        assert!((free.net_lift_vs_direct - free.gross_lift_vs_direct).abs() < 1e-12);
        assert!((priced.net_lift_vs_direct - priced.gross_lift_vs_direct + 0.08).abs() < 1e-12);
    }

    #[test]
    fn model_query_cost_reduces_lift_by_two_queries_times_cost() {
        let spec = FamilySpec::draw(0);
        let support = SupportMeans::fit(spec, 0);
        let free = score_family(spec, support, Strategy::AdditiveModel, 0.0);
        let priced = score_family(spec, support, Strategy::AdditiveModel, 0.04);
        assert!((free.net_lift_vs_direct - priced.net_lift_vs_direct - 0.08).abs() < 1e-12);
    }

    #[test]
    fn report_aggregates_by_family_not_by_reward_timestep() {
        let rows = run();
        assert_eq!(rows.len(), 4 * 8 * 4);
        for row in rows {
            assert_eq!(row.family_count, match row.lane {
                "all" => 256,
                "additive" => 86,
                "interaction" | "holdout_shift" => 85,
                _ => panic!("unknown lane"),
            });
            assert!(row.mean_net_lift_vs_direct.is_finite());
            assert!(row.mean_gross_lift_vs_direct.is_finite());
            assert!(row.mean_break_even_query_cost.is_finite());
            assert!((0.0..=1.0).contains(&row.beneficial_family_rate_before_cost));
            assert!(row.lift_ci95_low.is_finite() && row.lift_ci95_high.is_finite());
            assert!(row.lift_ci95_low <= row.lift_ci95_high);
            assert!(row.minimum_family_net_lift <= row.lower_cvar05_net_lift + 1e-12);
            assert!(row.lower_cvar05_net_lift <= row.mean_net_lift_vs_direct + 1e-12);
            assert!((0.0..=1.0).contains(&row.harmed_family_rate));
            assert!((0.0..=1.0).contains(&row.action_one_rate));
        }
    }
}
