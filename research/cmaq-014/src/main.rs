use std::fmt::Write as _;

const SCHEMA: &str = "cmaq-014-dual-control-disturbance-v0.1";
const EPISODES: usize = 512;
const DAMAGE_THRESHOLD: f64 = 0.12;
const MAX_DAMAGE_PROBABILITY: f64 = 0.05;
const SEED: u64 = 0xC0A0_0014_2026_1030;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lane { Robust, Mixed, Fragile }

impl Lane {
    fn all() -> [Self; 3] { [Self::Robust, Self::Mixed, Self::Fragile] }
    fn name(self) -> &'static str {
        match self { Self::Robust => "robust_assets", Self::Mixed => "mixed_assets", Self::Fragile => "fragile_assets" }
    }
    fn fragile_probability(self) -> f64 {
        match self { Self::Robust => 0.01, Self::Mixed => 0.20, Self::Fragile => 0.60 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Probe { None, Visual, GentleContact, ForceContact, Invasive }

impl Probe {
    fn all() -> [Self; 5] { [Self::None, Self::Visual, Self::GentleContact, Self::ForceContact, Self::Invasive] }
    fn name(self) -> &'static str {
        match self {
            Self::None => "no_probe",
            Self::Visual => "visual",
            Self::GentleContact => "gentle_contact",
            Self::ForceContact => "force_contact",
            Self::Invasive => "invasive_contact",
        }
    }
    fn accuracy(self) -> f64 {
        match self {
            Self::None => 0.50,
            Self::Visual => 0.62,
            Self::GentleContact => 0.72,
            Self::ForceContact => 0.90,
            Self::Invasive => 0.98,
        }
    }
    fn disturbance(self, fragile: bool) -> f64 {
        match (self, fragile) {
            (Self::None | Self::Visual, _) => 0.0,
            (Self::GentleContact, false) => 0.002,
            (Self::GentleContact, true) => 0.040,
            (Self::ForceContact, false) => 0.020,
            (Self::ForceContact, true) => 0.200,
            (Self::Invasive, false) => 0.040,
            (Self::Invasive, true) => 0.350,
        }
    }
    fn cost(self) -> f64 {
        match self {
            Self::None => 0.0, Self::Visual => 0.010,
            Self::GentleContact => 0.020, Self::ForceContact => 0.030, Self::Invasive => 0.040,
        }
    }
    fn risk_probability(self, fragile_probability: f64) -> f64 {
        if self.disturbance(true) >= DAMAGE_THRESHOLD { fragile_probability } else { 0.0 }
    }
    fn expected_disturbance(self, fragile_probability: f64) -> f64 {
        (1.0 - fragile_probability) * self.disturbance(false)
            + fragile_probability * self.disturbance(true)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Policy {
    NoProbe,
    VisualOnly,
    InformationGreedy,
    CostOnly,
    DisturbanceAware,
    RiskConstrained,
    OracleFragilityAware,
}

impl Policy {
    fn all() -> [Self; 7] {
        [Self::NoProbe, Self::VisualOnly, Self::InformationGreedy, Self::CostOnly,
         Self::DisturbanceAware, Self::RiskConstrained, Self::OracleFragilityAware]
    }
    fn name(self) -> &'static str {
        match self {
            Self::NoProbe => "no_probe",
            Self::VisualOnly => "visual_only",
            Self::InformationGreedy => "information_greedy",
            Self::CostOnly => "cost_only",
            Self::DisturbanceAware => "disturbance_aware",
            Self::RiskConstrained => "risk_constrained",
            Self::OracleFragilityAware => "oracle_fragility_aware",
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
}

fn expected_utility(probe: Probe, fragile_probability: f64) -> f64 {
    let p_damage = probe.risk_probability(fragile_probability);
    let p_correct = probe.accuracy();
    let expected_terminal = (1.0 - p_damage) * (p_correct - (1.0 - p_correct)) + p_damage * -1.0;
    expected_terminal - probe.cost() - 0.5 * probe.expected_disturbance(fragile_probability)
}

fn choose_probe(policy: Policy, lane: Lane, hidden_fragile: Option<bool>) -> Probe {
    let p_fragile = lane.fragile_probability();
    match policy {
        Policy::NoProbe => Probe::None,
        Policy::VisualOnly => Probe::Visual,
        Policy::InformationGreedy => Probe::Invasive,
        Policy::CostOnly => Probe::all().into_iter()
            .max_by(|a, b| (a.accuracy() - a.cost()).total_cmp(&(b.accuracy() - b.cost())))
            .unwrap(),
        Policy::DisturbanceAware => [Probe::Visual, Probe::GentleContact, Probe::ForceContact, Probe::Invasive]
            .into_iter().max_by(|a,b| expected_utility(*a,p_fragile).total_cmp(&expected_utility(*b,p_fragile))).unwrap(),
        Policy::RiskConstrained => [Probe::Visual, Probe::GentleContact, Probe::ForceContact, Probe::Invasive]
            .into_iter()
            .filter(|probe| probe.risk_probability(p_fragile) <= MAX_DAMAGE_PROBABILITY)
            .max_by(|a,b| expected_utility(*a,p_fragile).total_cmp(&expected_utility(*b,p_fragile)))
            .unwrap_or(Probe::None),
        Policy::OracleFragilityAware => {
            let fragile = hidden_fragile.unwrap_or(false);
            [Probe::Visual, Probe::GentleContact, Probe::ForceContact, Probe::Invasive]
                .into_iter().max_by(|a,b| {
                    let score = |probe: Probe| {
                        let damage = probe.disturbance(fragile);
                        let task_reward = if damage >= DAMAGE_THRESHOLD { -1.0 } else { probe.accuracy() - (1.0 - probe.accuracy()) };
                        task_reward - probe.cost() - 0.5 * damage
                    };
                    score(*a).total_cmp(&score(*b))
                }).unwrap()
        }
    }
}

#[derive(Clone, Debug)]
struct Episode {
    fragile: bool,
    hidden_sign: bool,
    observation_uniform: f64,
}

fn episode_tape(lane: Lane) -> Vec<Episode> {
    let mut tape = Vec::with_capacity(EPISODES);
    let mut rng = SplitMix64::new(SEED ^ ((lane as u64) << 40));
    for _ in 0..EPISODES {
        tape.push(Episode {
            fragile: rng.unit() < lane.fragile_probability(),
            hidden_sign: rng.unit() < 0.5,
            observation_uniform: rng.unit(),
        });
    }
    tape
}

#[derive(Clone, Debug)]
struct Summary {
    policy: &'static str,
    mean_task_success: f64,
    mean_decision_accuracy: f64,
    mean_net_utility: f64,
    mean_disturbance: f64,
    mean_probe_cost: f64,
    damage_rate: f64,
    risk_budget_violation_rate: f64,
    mean_probe_accuracy: f64,
    probe_counts: [usize; 5],
}

fn run_policy(policy: Policy, lane: Lane, tape: &[Episode]) -> Summary {
    let mut successes = 0usize;
    let mut correct = 0usize;
    let mut total_utility = 0.0;
    let mut total_disturbance = 0.0;
    let mut total_cost = 0.0;
    let mut damage_count = 0usize;
    let mut risk_violations = 0usize;
    let mut probe_accuracy_sum = 0.0;
    let mut counts = [0usize; 5];

    for episode in tape {
        let probe = choose_probe(policy, lane, if policy == Policy::OracleFragilityAware { Some(episode.fragile) } else { None });
        counts[probe as usize] += 1;

        let damage = probe.disturbance(episode.fragile);
        let damaged = damage >= DAMAGE_THRESHOLD;
        let signal_correct = episode.observation_uniform < probe.accuracy();
        let decision_correct = if signal_correct { true } else { false };
        let task_success = decision_correct && !damaged;
        let reward = if task_success { 1.0 } else { -1.0 };
        let utility = reward - probe.cost() - 0.5 * damage;

        if task_success { successes += 1; }
        if decision_correct { correct += 1; }
        if damaged { damage_count += 1; }
        if probe.risk_probability(lane.fragile_probability()) > MAX_DAMAGE_PROBABILITY && policy == Policy::RiskConstrained {
            risk_violations += 1;
        }
        total_utility += utility;
        total_disturbance += damage;
        total_cost += probe.cost();
        probe_accuracy_sum += probe.accuracy();
    }

    let n = tape.len() as f64;
    Summary {
        policy: policy.name(),
        mean_task_success: successes as f64 / n,
        mean_decision_accuracy: correct as f64 / n,
        mean_net_utility: total_utility / n,
        mean_disturbance: total_disturbance / n,
        mean_probe_cost: total_cost / n,
        damage_rate: damage_count as f64 / n,
        risk_budget_violation_rate: risk_violations as f64 / n,
        mean_probe_accuracy: probe_accuracy_sum / n,
        probe_counts: counts,
    }
}

#[derive(Clone, Debug)]
struct Cell {
    lane: &'static str,
    fragile_probability: f64,
    episode_count: usize,
    summaries: Vec<Summary>,
}

fn run_benchmark() -> Vec<Cell> {
    Lane::all().into_iter().map(|lane| {
        let tape = episode_tape(lane);
        Cell {
            lane: lane.name(),
            fragile_probability: lane.fragile_probability(),
            episode_count: EPISODES,
            summaries: Policy::all().iter().map(|policy| run_policy(*policy,lane,&tape)).collect(),
        }
    }).collect()
}

fn number(value: f64) -> String {
    if value.is_finite() { format!("{value:.8}") } else { "null".to_string() }
}

fn render_report(cells: &[Cell]) -> String {
    let mut out = String::new();
    writeln!(&mut out, "{{").unwrap();
    writeln!(&mut out, "  \"schema_version\": \"{SCHEMA}\",").unwrap();
    writeln!(&mut out, "  \"episode_count\": {EPISODES},").unwrap();
    writeln!(&mut out, "  \"damage_threshold\": {},", number(DAMAGE_THRESHOLD)).unwrap();
    writeln!(&mut out, "  \"max_damage_probability\": {},", number(MAX_DAMAGE_PROBABILITY)).unwrap();
    writeln!(&mut out, "  \"lanes\": [\"robust_assets\",\"mixed_assets\",\"fragile_assets\"],").unwrap();
    writeln!(&mut out, "  \"policies\": [\"no_probe\",\"visual_only\",\"information_greedy\",\"cost_only\",\"disturbance_aware\",\"risk_constrained\",\"oracle_fragility_aware\"],").unwrap();
    writeln!(&mut out, "  \"cells\": [").unwrap();
    for (index, cell) in cells.iter().enumerate() {
        let comma = if index + 1 == cells.len() { "" } else { "," };
        writeln!(&mut out, "    {{\"lane\":\"{}\",\"fragile_probability\":{},\"episode_count\":{},\"summaries\":[",
            cell.lane, number(cell.fragile_probability), cell.episode_count).unwrap();
        for (j, row) in cell.summaries.iter().enumerate() {
            let row_comma = if j + 1 == cell.summaries.len() { "" } else { "," };
            writeln!(&mut out,
                concat!(
                    "      {{\"policy\":\"{}\",\"mean_task_success\":{},\"mean_decision_accuracy\":{},",
                    "\"mean_net_utility\":{},\"mean_disturbance\":{},\"mean_probe_cost\":{},",
                    "\"damage_rate\":{},\"risk_budget_violation_rate\":{},\"mean_probe_accuracy\":{},",
                    "\"probe_counts\":[{},{},{},{},{}]}}{}"
                ),
                row.policy, number(row.mean_task_success), number(row.mean_decision_accuracy),
                number(row.mean_net_utility), number(row.mean_disturbance), number(row.mean_probe_cost),
                number(row.damage_rate), number(row.risk_budget_violation_rate),
                number(row.mean_probe_accuracy),
                row.probe_counts[0], row.probe_counts[1], row.probe_counts[2], row.probe_counts[3], row.probe_counts[4], row_comma
            ).unwrap();
        }
        writeln!(&mut out, "    ]}}{comma}").unwrap();
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
    fn signal_quality_and_collateral_disturbance_are_distinct_axes() {
        assert!(Probe::Visual.accuracy() < Probe::GentleContact.accuracy());
        assert!(Probe::GentleContact.accuracy() < Probe::ForceContact.accuracy());
        assert!(Probe::ForceContact.accuracy() < Probe::Invasive.accuracy());
        assert_eq!(Probe::Visual.disturbance(true), 0.0);
        assert!(Probe::Invasive.disturbance(true) > Probe::GentleContact.disturbance(true));
    }

    #[test]
    fn information_and_cost_greedy_ignore_fragility() {
        for lane in Lane::all() {
            assert_eq!(choose_probe(Policy::InformationGreedy,lane,None),Probe::Invasive);
            assert_eq!(choose_probe(Policy::CostOnly,lane,None),Probe::Invasive);
        }
    }

    #[test]
    fn disturbance_aware_chooses_gentle_action_when_fragility_is_high() {
        assert_eq!(choose_probe(Policy::DisturbanceAware,Lane::Fragile,None),Probe::GentleContact);
        assert_eq!(choose_probe(Policy::DisturbanceAware,Lane::Robust,None),Probe::Invasive);
    }

    #[test]
    fn risk_constrained_policy_respects_declared_damage_probability_budget() {
        for lane in Lane::all() {
            let probe=choose_probe(Policy::RiskConstrained,lane,None);
            assert!(probe.risk_probability(lane.fragile_probability()) <= MAX_DAMAGE_PROBABILITY);
            let result=run_policy(Policy::RiskConstrained,lane,&episode_tape(lane));
            assert_eq!(result.risk_budget_violation_rate,0.0);
        }
    }

    #[test]
    fn fragile_lane_exposes_information_damage_tradeoff() {
        let lane=Lane::Fragile;
        let tape=episode_tape(lane);
        let greedy=run_policy(Policy::InformationGreedy,lane,&tape);
        let safe=run_policy(Policy::RiskConstrained,lane,&tape);
        assert!(greedy.mean_decision_accuracy > safe.mean_decision_accuracy);
        assert!(greedy.damage_rate > safe.damage_rate);
        assert!(greedy.mean_task_success < safe.mean_task_success);
    }

    #[test]
    fn visual_probe_never_disturbs_state() {
        let result=run_policy(Policy::VisualOnly,Lane::Fragile,&episode_tape(Lane::Fragile));
        assert_eq!(result.mean_disturbance,0.0);
        assert_eq!(result.damage_rate,0.0);
    }

    #[test]
    fn oracle_is_no_worse_than_prior_only_disturbance_aware_policy() {
        for lane in Lane::all() {
            let tape=episode_tape(lane);
            let oracle=run_policy(Policy::OracleFragilityAware,lane,&tape);
            let aware=run_policy(Policy::DisturbanceAware,lane,&tape);
            assert!(oracle.mean_net_utility + 1e-9 >= aware.mean_net_utility);
        }
    }

    #[test]
    fn fixed_seed_tape_is_reproducible_and_paired() {
        assert_eq!(episode_tape(Lane::Mixed),episode_tape(Lane::Mixed));
        for lane in Lane::all() {
            let tape=episode_tape(lane);
            let rows=Policy::all().iter().map(|p|run_policy(*p,lane,&tape)).collect::<Vec<_>>();
            assert_eq!(rows.len(),7);
        }
    }

    #[test]
    fn benchmark_has_one_cell_per_fragility_lane() {
        let cells=run_benchmark();
        assert_eq!(cells.len(),3);
        assert!(cells.iter().all(|cell|cell.episode_count==EPISODES && cell.summaries.len()==7));
    }
}
