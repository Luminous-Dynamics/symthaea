//! Exact finite-state evaluator for a first task-conditioned counterfactual profile.
//!
//! Unlike task_conditioned_counterfactual_profile.rs, this example generates
//! task outcomes from explicit Boolean rules and edge interventions. It exhausts
//! every exogenous input assignment and the two outcomes of a fair-coin edge cut.
//!
//! It is a tiny deterministic teaching benchmark, not a general causal simulator.
//! The "cross-cut" label comes from the declared partition. Control cuts use the
//! same signal-replacement operation and same downstream gate type where possible.
//! Intervention outputs are exact for the declared finite input/intervention law.
//!
//! Run: cargo run --example task_conditioned_finite_state_eval
//! Test: cargo test --example task_conditioned_finite_state_eval

use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    Input { index: usize },
    Copy { from: usize },
    Xor { left: usize, right: usize },
    And { left: usize, right: usize },
    Constant(bool),
}

impl Rule {
    fn sources(self) -> [Option<usize>; 2] {
        match self {
            Rule::Input { .. } | Rule::Constant(_) => [None, None],
            Rule::Copy { from } => [Some(from), None],
            Rule::Xor { left, right } | Rule::And { left, right } => {
                [Some(left), Some(right)]
            }
        }
    }

    fn canonical(self) -> String {
        match self {
            Rule::Input { index } => format!("input:{index}"),
            Rule::Copy { from } => format!("copy:{from}"),
            Rule::Xor { left, right } => format!("xor:{left}:{right}"),
            Rule::And { left, right } => format!("and:{left}:{right}"),
            Rule::Constant(value) => format!("constant:{}", if value { 1 } else { 0 }),
        }
    }
}

#[derive(Clone, Debug)]
struct Network {
    rules: Vec<Rule>,
    output_node: usize,
    input_count: usize,
    components: Vec<u8>,
}

impl Network {
    fn validate(&self) -> Result<(), String> {
        if self.rules.is_empty() || self.rules.len() > 12 {
            return Err("node count must be in 1..=12 for bounded partition enumeration".into());
        }
        if self.input_count == 0 || self.input_count > 8 {
            return Err("input count must be in 1..=8 for exact enumeration".into());
        }
        if self.output_node >= self.rules.len() {
            return Err("output node is outside the network".into());
        }
        if self.components.len() != self.rules.len() {
            return Err("component label count must equal node count".into());
        }
        if self.components.iter().any(|component| *component > 1) {
            return Err("component labels must be 0 or 1".into());
        }
        for (target, rule) in self.rules.iter().copied().enumerate() {
            if let Rule::Input { index } = rule {
                if index >= self.input_count {
                    return Err(format!("node {target} references invalid external input {index}"));
                }
            }
            for source in rule.sources().into_iter().flatten() {
                if source >= target {
                    return Err(format!(
                        "edge {source}->{target} violates topological evaluation order"
                    ));
                }
            }
        }
        Ok(())
    }

    fn canonical_rules(&self) -> String {
        self.rules
            .iter()
            .map(|rule| rule.canonical())
            .collect::<Vec<_>>()
            .join(",")
    }

    fn edge_exists(&self, source: usize, target: usize) -> bool {
        self.rules
            .get(target)
            .is_some_and(|rule| rule.sources().into_iter().flatten().any(|node| node == source))
    }

    fn edge_is_cross_partition(&self, source: usize, target: usize) -> bool {
        self.components[source] != self.components[target]
    }

    fn read_source(
        &self,
        source: usize,
        target: usize,
        values: &[bool],
        intervention: Option<&Intervention>,
        cut_coin: bool,
    ) -> bool {
        match intervention {
            Some(cut)
                if cut.source == source && cut.target == target =>
            {
                cut_coin
            }
            _ => values[source],
        }
    }

    fn evaluate_nodes(
        &self,
        inputs: &[bool],
        intervention: Option<&Intervention>,
        cut_coin: bool,
    ) -> Vec<bool> {
        let mut values = vec![false; self.rules.len()];
        for (target, rule) in self.rules.iter().copied().enumerate() {
            values[target] = match rule {
                Rule::Input { index } => inputs[index],
                Rule::Constant(value) => value,
                Rule::Copy { from } => {
                    self.read_source(from, target, &values, intervention, cut_coin)
                }
                Rule::Xor { left, right } => {
                    self.read_source(left, target, &values, intervention, cut_coin)
                        ^ self.read_source(right, target, &values, intervention, cut_coin)
                }
                Rule::And { left, right } => {
                    self.read_source(left, target, &values, intervention, cut_coin)
                        & self.read_source(right, target, &values, intervention, cut_coin)
                }
            };
        }
        values
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TaskTarget {
    XorInputs { left: usize, right: usize },
    AndInputs { left: usize, right: usize },
    CopyInput { index: usize },
    Constant(bool),
}

impl TaskTarget {
    fn evaluate(self, inputs: &[bool]) -> bool {
        match self {
            TaskTarget::XorInputs { left, right } => inputs[left] ^ inputs[right],
            TaskTarget::AndInputs { left, right } => inputs[left] & inputs[right],
            TaskTarget::CopyInput { index } => inputs[index],
            TaskTarget::Constant(value) => value,
        }
    }

    fn canonical(self) -> String {
        match self {
            TaskTarget::XorInputs { left, right } => format!("xor-inputs:{left}:{right}"),
            TaskTarget::AndInputs { left, right } => format!("and-inputs:{left}:{right}"),
            TaskTarget::CopyInput { index } => format!("copy-input:{index}"),
            TaskTarget::Constant(value) => format!("constant:{}", if value { 1 } else { 0 }),
        }
    }

    fn validate(self, input_count: usize) -> Result<(), String> {
        let indices: Vec<usize> = match self {
            TaskTarget::XorInputs { left, right } | TaskTarget::AndInputs { left, right } => {
                vec![left, right]
            }
            TaskTarget::CopyInput { index } => vec![index],
            TaskTarget::Constant(_) => vec![],
        };
        if indices.iter().any(|index| *index >= input_count) {
            return Err("task target references an invalid external input".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InterventionKind {
    CrossCut,
    MatchedControl,
}

impl InterventionKind {
    fn canonical(self) -> &'static str {
        match self {
            InterventionKind::CrossCut => "cross_cut",
            InterventionKind::MatchedControl => "matched_control",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Intervention {
    id: String,
    source: usize,
    target: usize,
    cost: u32,
}

#[derive(Clone, Debug)]
struct Fixture {
    name: String,
    network: Network,
    task_target: TaskTarget,
    interventions: Vec<Intervention>,
}

#[derive(Clone, Debug, PartialEq)]
struct InterventionResult {
    fixture: String,
    id: String,
    kind: InterventionKind,
    source: usize,
    target: usize,
    cost: u32,
    intact_utility: f64,
    utility: f64,
    utility_loss: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct ProfilePoint {
    fixture: String,
    max_cost: u32,
    intact_utility: f64,
    cross_cut_mean_loss: f64,
    matched_control_mean_loss: f64,
    tccip: f64,
    worst_cross_cut_utility: f64,
    n_cross_cut: usize,
    n_matched_control: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct PartitionProfilePoint {
    fixture: String,
    partition_mask: usize,
    partition_labels: String,
    cross_cut_mean_loss: f64,
    matched_control_mean_loss: f64,
    tccip: f64,
    n_cross_cut: usize,
    n_matched_control: usize,
}

/// Enumerate every nontrivial bipartition, canonicalizing complements by fixing
/// node 0 to component A. All partitions are reported or counted as skipped;
/// the evaluator never selects the best partition as its only reported result.
fn all_partition_profiles(
    fixture_name: &str,
    node_count: usize,
    results: &[InterventionResult],
) -> (Vec<PartitionProfilePoint>, usize, usize) {
    let total_partitions = (1usize << (node_count - 1)) - 1;
    let mut profiles = Vec::new();

    for mask in 1usize..(1usize << node_count) {
        if mask & 1 == 1 {
            continue; // complement of a canonical representative
        }
        let labels = (0..node_count)
            .map(|node| if (mask >> node) & 1 == 1 { 'B' } else { 'A' })
            .collect::<String>();
        let mut cross_losses = Vec::new();
        let mut control_losses = Vec::new();

        for result in results {
            let source_side = (mask >> result.source) & 1;
            let target_side = (mask >> result.target) & 1;
            if source_side != target_side {
                cross_losses.push(result.utility_loss);
            } else {
                control_losses.push(result.utility_loss);
            }
        }

        if cross_losses.is_empty() || control_losses.is_empty() {
            continue;
        }
        let cross_mean = cross_losses.iter().sum::<f64>() / cross_losses.len() as f64;
        let control_mean =
            control_losses.iter().sum::<f64>() / control_losses.len() as f64;
        profiles.push(PartitionProfilePoint {
            fixture: fixture_name.into(),
            partition_mask: mask,
            partition_labels: labels,
            cross_cut_mean_loss: cross_mean,
            matched_control_mean_loss: control_mean,
            tccip: cross_mean - control_mean,
            n_cross_cut: cross_losses.len(),
            n_matched_control: control_losses.len(),
        });
    }
    let skipped = total_partitions.saturating_sub(profiles.len());
    (profiles, total_partitions, skipped)
}

impl Fixture {
    fn validate(&self) -> Result<(), String> {
        self.network.validate()?;
        self.task_target.validate(self.network.input_count)?;
        if self.interventions.is_empty() {
            return Err("fixture must have at least one intervention".into());
        }
        let mut ids = HashSet::new();
        for cut in &self.interventions {
            if cut.id.trim().is_empty() || !ids.insert(cut.id.as_str()) {
                return Err("intervention IDs must be unique and non-empty".into());
            }
            if cut.cost == 0 {
                return Err(format!("intervention {} must have positive cost", cut.id));
            }
            if cut.source >= self.network.rules.len() || cut.target >= self.network.rules.len() {
                return Err(format!("intervention {} has an out-of-range node", cut.id));
            }
            if !self.network.edge_exists(cut.source, cut.target) {
                return Err(format!(
                    "intervention {} refers to missing edge {}->{}",
                    cut.id, cut.source, cut.target
                ));
            }
        }
        Ok(())
    }

    fn intact_utility(&self) -> f64 {
        self.utility(None)
    }

    /// Exact expected accuracy under uniform exogenous inputs. When an edge is
    /// cut, the replacement signal is an independent fair coin, exhaustively
    /// enumerated instead of sampled.
    fn utility(&self, intervention: Option<&Intervention>) -> f64 {
        let input_states = 1usize << self.network.input_count;
        let coin_states = if intervention.is_some() { 2 } else { 1 };
        let mut correct = 0usize;
        for state in 0..input_states {
            let inputs = (0..self.network.input_count)
                .map(|index| ((state >> index) & 1) != 0)
                .collect::<Vec<_>>();
            let expected = self.task_target.evaluate(&inputs);
            for coin_state in 0..coin_states {
                let values = self.network.evaluate_nodes(
                    &inputs,
                    intervention,
                    coin_state != 0,
                );
                if values[self.network.output_node] == expected {
                    correct += 1;
                }
            }
        }
        correct as f64 / (input_states * coin_states) as f64
    }

    fn run(&self) -> Result<(Vec<InterventionResult>, Vec<ProfilePoint>), String> {
        self.validate()?;
        let intact = self.intact_utility();
        let mut results = Vec::with_capacity(self.interventions.len());
        for cut in &self.interventions {
            let utility = self.utility(Some(cut));
            let kind = if self.network.edge_is_cross_partition(cut.source, cut.target) {
                InterventionKind::CrossCut
            } else {
                InterventionKind::MatchedControl
            };
            results.push(InterventionResult {
                fixture: self.name.clone(),
                id: cut.id.clone(),
                kind,
                source: cut.source,
                target: cut.target,
                cost: cut.cost,
                intact_utility: intact,
                utility,
                utility_loss: intact - utility,
            });
        }

        let budget = results.iter().map(|result| result.cost).max().unwrap_or(1);
        let profile = summarize(&self.name, intact, &results, &[budget])?;
        Ok((results, profile))
    }
}

fn summarize(
    fixture_name: &str,
    intact_utility: f64,
    results: &[InterventionResult],
    budgets: &[u32],
) -> Result<Vec<ProfilePoint>, String> {
    if !intact_utility.is_finite() || !(0.0..=1.0).contains(&intact_utility) {
        return Err("intact utility must be finite and in [0, 1]".into());
    }
    if budgets.is_empty() || budgets.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("budgets must be non-empty and strictly increasing".into());
    }

    budgets
        .iter()
        .copied()
        .map(|max_cost| {
            let cross = results
                .iter()
                .filter(|result| result.cost <= max_cost && result.kind == InterventionKind::CrossCut)
                .collect::<Vec<_>>();
            let controls = results
                .iter()
                .filter(|result| {
                    result.cost <= max_cost && result.kind == InterventionKind::MatchedControl
                })
                .collect::<Vec<_>>();

            if cross.is_empty() {
                return Err(format!("no cross-cut observations at budget {max_cost}"));
            }
            if controls.is_empty() {
                return Err(format!("no matched-control observations at budget {max_cost}"));
            }

            let cross_mean_loss =
                cross.iter().map(|result| result.utility_loss).sum::<f64>() / cross.len() as f64;
            let control_mean_loss =
                controls.iter().map(|result| result.utility_loss).sum::<f64>()
                    / controls.len() as f64;
            let worst_cross_cut_utility = cross
                .iter()
                .map(|result| result.utility)
                .fold(f64::INFINITY, f64::min);

            Ok(ProfilePoint {
                fixture: fixture_name.into(),
                max_cost,
                intact_utility,
                cross_cut_mean_loss: cross_mean_loss,
                matched_control_mean_loss: control_mean_loss,
                tccip: cross_mean_loss - control_mean_loss,
                worst_cross_cut_utility,
                n_cross_cut: cross.len(),
                n_matched_control: controls.len(),
            })
        })
        .collect()
}

fn intervention(id: &str, source: usize, target: usize) -> Intervention {
    Intervention {
        id: id.into(),
        source,
        target,
        cost: 1,
    }
}

fn fixtures() -> Vec<Fixture> {
    vec![
        Fixture {
            name: "useful_xor_coordination".into(),
            network: Network {
                input_count: 2,
                output_node: 4,
                rules: vec![
                    Rule::Input { index: 0 },     // 0
                    Rule::Input { index: 1 },     // 1
                    Rule::Copy { from: 0 },       // 2, component A
                    Rule::Copy { from: 1 },       // 3, component A
                    Rule::Xor { left: 2, right: 3 }, // 4, task output in component B
                    Rule::Copy { from: 0 },       // 5, control branch in B
                    Rule::Copy { from: 1 },       // 6, control branch in B
                    Rule::Xor { left: 5, right: 6 }, // 7, unused control output
                ],
                components: vec![0, 0, 0, 0, 1, 1, 1, 1],
            },
            task_target: TaskTarget::XorInputs { left: 0, right: 1 },
            interventions: vec![
                intervention("cross_left_into_task_xor", 2, 4),
                intervention("cross_right_into_task_xor", 3, 4),
                intervention("matched_cut_control_left", 5, 7),
                intervention("matched_cut_control_right", 6, 7),
            ],
        },
        Fixture {
            name: "irrelevant_cross_component_connectivity".into(),
            network: Network {
                input_count: 2,
                output_node: 2,
                rules: vec![
                    Rule::Input { index: 0 }, // 0
                    Rule::Input { index: 1 }, // 1
                    Rule::Copy { from: 0 },   // 2, task output
                    Rule::Copy { from: 2 },   // 3, cross-component side branch
                    Rule::Copy { from: 3 },   // 4, internal control branch
                ],
                components: vec![0, 0, 0, 1, 1],
            },
            task_target: TaskTarget::CopyInput { index: 0 },
            interventions: vec![
                intervention("cross_cut_irrelevant_branch", 2, 3),
                intervention("matched_internal_side_branch", 3, 4),
            ],
        },
        Fixture {
            name: "harmful_cross_component_coupling".into(),
            network: Network {
                input_count: 2,
                output_node: 6,
                rules: vec![
                    Rule::Input { index: 0 }, // 0
                    Rule::Input { index: 1 }, // 1
                    Rule::Copy { from: 0 },   // 2
                    Rule::Copy { from: 1 },   // 3
                    Rule::Xor { left: 2, right: 3 }, // 4, desired parity
                    Rule::Constant(true),     // 5, harmful bias
                    Rule::Xor { left: 4, right: 5 }, // 6, output is inverted parity
                    Rule::Copy { from: 0 },   // 7, control branch in B
                    Rule::Copy { from: 1 },   // 8, control branch in B
                    Rule::Xor { left: 7, right: 8 }, // 9, unused control output
                ],
                components: vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            },
            task_target: TaskTarget::XorInputs { left: 0, right: 1 },
            interventions: vec![
                intervention("cross_cut_correct_parity_path", 4, 6),
                intervention("cross_cut_harmful_bias_path", 5, 6),
                intervention("matched_control_left", 7, 9),
                intervention("matched_control_right", 8, 9),
            ],
        },
    ]
}

fn main() {
    println!("# schema=tccip-exact-finite-state-v1");
    let source_revision = std::env::var("SYMTHAEA_SOURCE_SHA")
        .unwrap_or_else(|_| "UNBOUND".to_string());
    println!("# source_revision={source_revision}");
    println!("# UNBOUND means exploratory output only; bind SYMTHAEA_SOURCE_SHA to the exact git HEAD for a reproducibility receipt");
    println!("# inputs are enumerated uniformly; cut source is replaced by an independent fair coin");
    println!("# component labels: 0=A, 1=B; intervention kind is derived from source/target labels");
    println!("fixture\tintervention\tkind\tsource\ttarget\tcost\tintact_utility\tutility\tutility_loss");
    let mut any_failed = false;

    for fixture in fixtures() {
        if let Err(error) = fixture.validate() {
            eprintln!("NOT QUALIFIED: fixture {}: {error}", fixture.name);
            any_failed = true;
            continue;
        }
        println!(
            "# fixture={} target={} output_node={} input_count={} components={:?} rules={}",
            fixture.name,
            fixture.task_target.canonical(),
            fixture.network.output_node,
            fixture.network.input_count,
            fixture.network.components,
            fixture.network.canonical_rules()
        );

        match fixture.run() {
            Ok((results, profile)) => {
                for result in &results {
                    println!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}",
                        result.fixture,
                        result.id,
                        result.kind.canonical(),
                        result.source,
                        result.target,
                        result.cost,
                        result.intact_utility,
                        result.utility,
                        result.utility_loss
                    );
                }
                for point in profile {
                    println!(
                        "# profile={} budget={} intact={:.9} cross_loss={:.9} control_loss={:.9} tccip={:.9} worst_cross_utility={:.9} n_cross={} n_control={}",
                        point.fixture,
                        point.max_cost,
                        point.intact_utility,
                        point.cross_cut_mean_loss,
                        point.matched_control_mean_loss,
                        point.tccip,
                        point.worst_cross_cut_utility,
                        point.n_cross_cut,
                        point.n_matched_control
                    );
                }
                let (partition_profiles, total_partitions, skipped) =
                    all_partition_profiles(&fixture.name, fixture.network.rules.len(), &results);
                println!(
                    "# partition_sweep_summary fixture={} total={} usable={} skipped={}",
                    fixture.name,
                    total_partitions,
                    partition_profiles.len(),
                    skipped
                );
                for point in partition_profiles {
                    println!(
                        "# partition_profile fixture={} mask={} labels={} cross_loss={:.9} control_loss={:.9} tccip={:.9} n_cross={} n_control={}",
                        point.fixture,
                        point.partition_mask,
                        point.partition_labels,
                        point.cross_cut_mean_loss,
                        point.matched_control_mean_loss,
                        point.tccip,
                        point.n_cross_cut,
                        point.n_matched_control
                    );
                }
            }
            Err(error) => {
                eprintln!("NOT QUALIFIED: fixture {}: {error}", fixture.name);
                any_failed = true;
            }
        }
    }
    if any_failed {
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_named(name: &str) -> (Vec<InterventionResult>, Vec<ProfilePoint>) {
        fixtures()
            .into_iter()
            .find(|fixture| fixture.name == name)
            .expect("fixture exists")
            .run()
            .expect("fixture qualifies")
    }

    #[test]
    fn xor_task_has_perfect_intact_utility_and_positive_cross_cut_contrast() {
        let (results, profile) = run_named("useful_xor_coordination");
        assert_eq!(results.len(), 4);
        assert!(results.iter().all(|result| result.intact_utility == 1.0));
        assert!(results.iter().filter(|result| result.kind == InterventionKind::CrossCut)
            .all(|result| result.utility == 0.5));
        assert!(results.iter().filter(|result| result.kind == InterventionKind::MatchedControl)
            .all(|result| result.utility == 1.0));
        assert!((profile[0].tccip - 0.5).abs() < 1e-12);
        assert_eq!(profile[0].n_cross_cut, 2);
        assert_eq!(profile[0].n_matched_control, 2);
    }

    #[test]
    fn irrelevant_cross_component_edge_does_not_gain_task_credit() {
        let (_, profile) = run_named("irrelevant_cross_component_connectivity");
        assert_eq!(profile[0].intact_utility, 1.0);
        assert_eq!(profile[0].cross_cut_mean_loss, 0.0);
        assert_eq!(profile[0].matched_control_mean_loss, 0.0);
        assert_eq!(profile[0].tccip, 0.0);
    }

    #[test]
    fn harmful_coupling_produces_negative_score_without_clamping() {
        let (_, profile) = run_named("harmful_cross_component_coupling");
        assert_eq!(profile[0].intact_utility, 0.0);
        assert_eq!(profile[0].cross_cut_mean_loss, -0.5);
        assert_eq!(profile[0].matched_control_mean_loss, 0.0);
        assert_eq!(profile[0].tccip, -0.5);
    }

    #[test]
    fn every_intervention_is_a_real_edge_and_component_classification_is_derived() {
        for fixture in fixtures() {
            fixture.validate().unwrap();
            for cut in fixture.interventions {
                assert!(fixture.network.edge_exists(cut.source, cut.target));
                assert_eq!(
                    fixture.network.edge_is_cross_partition(cut.source, cut.target),
                    fixture.network.components[cut.source] != fixture.network.components[cut.target]
                );
            }
        }
    }

    #[test]
    fn missing_cross_cut_or_control_is_not_converted_to_zero() {
        let result = InterventionResult {
            fixture: "incomplete".into(),
            id: "only_control".into(),
            kind: InterventionKind::MatchedControl,
            source: 0,
            target: 1,
            cost: 1,
            intact_utility: 1.0,
            utility: 1.0,
            utility_loss: 0.0,
        };
        assert!(summarize("incomplete", 1.0, &[result], &[1])
            .unwrap_err()
            .contains("no cross-cut"));
    }

    #[test]
    fn rejects_invalid_edges_and_non_topological_rules() {
        let mut fixture = fixtures().remove(0);
        fixture.interventions.push(intervention("bad_edge", 7, 4));
        assert!(fixture.validate().unwrap_err().contains("missing edge"));

        let bad_network = Network {
            rules: vec![Rule::Input { index: 0 }, Rule::Copy { from: 1 }],
            output_node: 1,
            input_count: 1,
            components: vec![0, 0],
        };
        assert!(bad_network.validate().unwrap_err().contains("topological"));
    }

    #[test]
    fn exact_evaluator_is_deterministic_and_within_utility_bounds() {
        for fixture in fixtures() {
            let (first, first_profile) = fixture.run().unwrap();
            let (second, second_profile) = fixture.run().unwrap();
            assert_eq!(first, second);
            assert_eq!(first_profile, second_profile);
            assert!(first.iter().all(|row| (0.0..=1.0).contains(&row.utility)));
        }
    }

    #[test]
    fn intervention_cost_must_be_positive_and_ids_unique() {
        let mut fixture = fixtures().remove(0);
        fixture.interventions.push(Intervention {
            id: "cross_left_into_task_xor".into(),
            source: 2,
            target: 4,
            cost: 1,
        });
        assert!(fixture.validate().unwrap_err().contains("unique"));

        let mut fixture = fixtures().remove(1);
        fixture.interventions[0].cost = 0;
        assert!(fixture.validate().unwrap_err().contains("positive cost"));
    }

    #[test]
    fn exhaustive_partition_sweep_contains_the_declared_partition_result() {
        for fixture in fixtures() {
            let (results, declared_profile) = fixture.run().unwrap();
            let mask = fixture.network.components.iter().enumerate().fold(
                0usize,
                |acc, (index, component)| acc | ((*component as usize) << index),
            );
            let (sweep, total, skipped) =
                all_partition_profiles(&fixture.name, fixture.network.rules.len(), &results);
            assert_eq!(total, sweep.len() + skipped);
            let alternate = sweep.iter().find(|point| point.partition_mask == mask)
                .expect("declared nontrivial partition must be reported");
            assert!((alternate.tccip - declared_profile[0].tccip).abs() < 1e-12);
        }
    }

    #[test]
    fn partition_sweep_is_stable_and_reports_all_unusable_partitions() {
        for fixture in fixtures() {
            let (results, _) = fixture.run().unwrap();
            let (first, total, skipped) =
                all_partition_profiles(&fixture.name, fixture.network.rules.len(), &results);
            let (second, total_again, skipped_again) =
                all_partition_profiles(&fixture.name, fixture.network.rules.len(), &results);
            assert_eq!(first, second);
            assert_eq!(total, total_again);
            assert_eq!(skipped, skipped_again);
            assert_eq!(total, first.len() + skipped);
            assert!(first.windows(2).all(|pair| pair[0].partition_mask < pair[1].partition_mask));
            assert!(first.iter().all(|point| (-1.0..=1.0).contains(&point.tccip)));
        }
    }

    #[test]
    fn useful_xor_partition_sweep_exposes_sign_reversal() {
        let fixture = fixtures().into_iter()
            .find(|fixture| fixture.name == "useful_xor_coordination").unwrap();
        let (results, _) = fixture.run().unwrap();
        let (profiles, total, skipped) =
            all_partition_profiles(&fixture.name, fixture.network.rules.len(), &results);
        assert_eq!(total, 127);
        assert_eq!(skipped, 15);
        let min = profiles.iter().map(|point| point.tccip).fold(f64::INFINITY, f64::min);
        let max = profiles.iter().map(|point| point.tccip).fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(min, -0.5);
        assert_eq!(max, 0.5);
    }

    #[test]
    fn harmful_coupling_partition_sweep_also_exposes_sign_reversal() {
        let fixture = fixtures().into_iter()
            .find(|fixture| fixture.name == "harmful_cross_component_coupling").unwrap();
        let (results, _) = fixture.run().unwrap();
        let (profiles, total, skipped) =
            all_partition_profiles(&fixture.name, fixture.network.rules.len(), &results);
        assert_eq!(total, 511);
        assert_eq!(skipped, 63);
        let min = profiles.iter().map(|point| point.tccip).fold(f64::INFINITY, f64::min);
        let max = profiles.iter().map(|point| point.tccip).fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(min, -0.5);
        assert_eq!(max, 0.5);
    }
}
