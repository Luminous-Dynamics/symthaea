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

    fn read_source_multi(
        &self,
        source: usize,
        target: usize,
        values: &[bool],
        interventions: &[&Intervention],
        cut_mask: usize,
    ) -> bool {
        for (index, intervention) in interventions.iter().enumerate() {
            if intervention.source == source && intervention.target == target {
                return ((cut_mask >> index) & 1) != 0;
            }
        }
        values[source]
    }

    fn evaluate_nodes_multi(
        &self,
        inputs: &[bool],
        interventions: &[&Intervention],
        cut_mask: usize,
    ) -> Vec<bool> {
        let mut values = vec![false; self.rules.len()];
        for (target, rule) in self.rules.iter().copied().enumerate() {
            values[target] = match rule {
                Rule::Input { index } => inputs[index],
                Rule::Constant(value) => value,
                Rule::Copy { from } => self.read_source_multi(
                    from, target, &values, interventions, cut_mask,
                ),
                Rule::Xor { left, right } => {
                    self.read_source_multi(left, target, &values, interventions, cut_mask)
                        ^ self.read_source_multi(right, target, &values, interventions, cut_mask)
                }
                Rule::And { left, right } => {
                    self.read_source_multi(left, target, &values, interventions, cut_mask)
                        & self.read_source_multi(right, target, &values, interventions, cut_mask)
                }
            };
        }
        values
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

fn entropy_bits(counts: &[u128]) -> f64 {
    let total = counts.iter().copied().sum::<u128>();
    if total == 0 {
        return 0.0;
    }
    counts.iter().copied().filter(|count| *count > 0).map(|count| {
        let probability = count as f64 / total as f64;
        -probability * probability.log2()
    }).sum()
}

#[derive(Clone, Debug, PartialEq)]
struct InformationBaseline {
    target_entropy_bits: f64,
    individual_input_mi_bits: Vec<f64>,
    joint_input_mi_bits: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct CounterfactualUtilityInteraction {
    fixture: String,
    first_id: String,
    second_id: String,
    intact_utility: f64,
    first_cut_utility: f64,
    second_cut_utility: f64,
    joint_cut_utility: f64,
    interaction_contrast: f64,
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
    input_state_weights: Vec<u64>,
    interventions: Vec<Intervention>,

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
    fn canonical_interventions(&self) -> String {
        self.interventions.iter()
            .map(|cut| format!("{}:{}:{}:{}", cut.id, cut.source, cut.target, cut.cost))
            .collect::<Vec<_>>()
            .join(";")
    }

    fn validate(&self) -> Result<(), String> {
        self.network.validate()?;
        self.task_target.validate(self.network.input_count)?;
        if self.input_state_weights.len() != (1usize << self.network.input_count) {
            return Err(format!("{}: input-state weight count must equal 2^input_count", self.name));
        }
        if self.input_state_weights.iter().all(|weight| *weight == 0) {
            return Err(format!("{}: at least one input-state weight must be positive", self.name));
        }
        if self.interventions.is_empty() {
            return Err("fixture must have at least one intervention".into());
        }
        let mut ids = HashSet::new();
        let mut edges = HashSet::new();
        for cut in &self.interventions {
            if cut.id.trim().is_empty() || !ids.insert(cut.id.as_str()) {
                return Err("intervention IDs must be unique and non-empty".into());
            }
            if !edges.insert((cut.source, cut.target)) {
                return Err(format!("duplicate intervention edge {}->{}", cut.source, cut.target));
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

    /// Exact expected accuracy under the declared integer-weighted input-state
    /// distribution. A cut replaces one edge signal with an independent fair
    /// coin, exhaustively enumerated instead of sampled.
    fn utility(&self, intervention: Option<&Intervention>) -> f64 {
        let input_states = 1usize << self.network.input_count;
        let coin_states = if intervention.is_some() { 2usize } else { 1usize };
        let total_input_weight = self.input_state_weights.iter()
            .map(|weight| *weight as u128)
            .sum::<u128>();
        let mut correct_weight = 0u128;
        for state in 0..input_states {
            let inputs = (0..self.network.input_count)
                .map(|index| ((state >> index) & 1) != 0)
                .collect::<Vec<_>>();
            let expected = self.task_target.evaluate(&inputs);
            let weight = self.input_state_weights[state] as u128;
            for coin_state in 0..coin_states {
                let values = self.network.evaluate_nodes(
                    &inputs,
                    intervention,
                    coin_state != 0,
                );
                if values[self.network.output_node] == expected {
                    correct_weight += weight;
                }
            }
        }
        let denominator = total_input_weight * coin_states as u128;
        correct_weight as f64 / denominator as f64
    }

    /// Exact task-utility interaction contrast for a pair of edge interventions:
    /// Δ_ij = J(∅) - J(i) - J(j) + J({i,j}).
    ///
    /// Every cut replaces that edge's input with its own independent fair coin.
    /// This is a task-utility finite difference, not PID synergy or IIT Φ.
    fn pairwise_utility_interactions(&self) -> Vec<CounterfactualUtilityInteraction> {
        let intact = self.utility(None);
        let mut output = Vec::new();
        for first_index in 0..self.interventions.len() {
            for second_index in (first_index + 1)..self.interventions.len() {
                let first = &self.interventions[first_index];
                let second = &self.interventions[second_index];
                let first_utility = self.utility(Some(first));
                let second_utility = self.utility(Some(second));
                let joint_utility = self.utility_for_cuts(&[first, second]);
                output.push(CounterfactualUtilityInteraction {
                    fixture: self.name.clone(),
                    first_id: first.id.clone(),
                    second_id: second.id.clone(),
                    intact_utility: intact,
                    first_cut_utility: first_utility,
                    second_cut_utility: second_utility,
                    joint_cut_utility: joint_utility,
                    interaction_contrast: intact - first_utility - second_utility + joint_utility,
                });
            }
        }
        output
    }

    /// Exact expected task accuracy after simultaneously cutting any specified
    /// subset of edges. Each cut edge receives an independent fair-coin input.
    fn utility_for_cuts(&self, cuts: &[&Intervention]) -> f64 {
        let input_states = 1usize << self.network.input_count;
        let coin_states = 1usize << cuts.len();
        let total_input_weight = self.input_state_weights.iter()
            .map(|weight| *weight as u128)
            .sum::<u128>();
        let mut correct_weight = 0u128;
        for state in 0..input_states {
            let inputs = (0..self.network.input_count)
                .map(|index| ((state >> index) & 1) != 0)
                .collect::<Vec<_>>();
            let expected = self.task_target.evaluate(&inputs);
            let weight = self.input_state_weights[state] as u128;
            for cut_mask in 0..coin_states {
                let values = self.network.evaluate_nodes_multi(&inputs, cuts, cut_mask);
                if values[self.network.output_node] == expected {
                    correct_weight += weight;
                }
            }
        }
        correct_weight as f64 / (total_input_weight * coin_states as u128) as f64
    }

    /// Exact information-theoretic baseline under the declared input prior.
    /// Since the task target is a deterministic function of all external inputs,
    /// I(all_inputs; target) = H(target). Individual MI values reveal whether
    /// the target's information is available from each input separately.
    fn information_baseline(&self) -> InformationBaseline {
        let state_count = 1usize << self.network.input_count;
        let total_weight = self.input_state_weights.iter().map(|w| *w as u128).sum::<u128>();
        let mut target_counts = [0u128; 2];
        let mut joint_counts = vec![[[0u128; 2]; 2]; self.network.input_count];

        for state in 0..state_count {
            let inputs = (0..self.network.input_count)
                .map(|index| ((state >> index) & 1) != 0)
                .collect::<Vec<_>>();
            let target = if self.task_target.evaluate(&inputs) { 1 } else { 0 };
            let weight = self.input_state_weights[state] as u128;
            target_counts[target] += weight;
            for index in 0..self.network.input_count {
                let input_value = if inputs[index] { 1 } else { 0 };
                joint_counts[index][input_value][target] += weight;
            }
        }

        let target_entropy = entropy_bits(&target_counts);
        let individual_input_mi_bits = joint_counts.iter().map(|counts| {
            let mut conditional_target_entropy = 0.0;
            for input_value in 0..2 {
                let input_weight = counts[input_value].iter().copied().sum::<u128>();
                if input_weight > 0 {
                    conditional_target_entropy +=
                        input_weight as f64 / total_weight as f64 * entropy_bits(&counts[input_value]);
                }
            }
            let mi = target_entropy - conditional_target_entropy;
            if mi.abs() < 1e-12 { 0.0 } else { mi }
        }).collect::<Vec<_>>();

        InformationBaseline {
            target_entropy_bits: target_entropy,
            individual_input_mi_bits,
            joint_input_mi_bits: target_entropy,
        }
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

fn parse_usize(value: &str, field: &str) -> Result<usize, String> {
    value.parse::<usize>()
        .map_err(|error| format!("invalid {field} {value:?}: {error}"))
}

fn parse_rule(spec: &str) -> Result<Rule, String> {
    let parts = spec.split(':').collect::<Vec<_>>();
    match parts.as_slice() {
        ["input", index] => Ok(Rule::Input { index: parse_usize(index, "input index")? }),
        ["copy", from] => Ok(Rule::Copy { from: parse_usize(from, "copy source")? }),
        ["xor", left, right] => Ok(Rule::Xor {
            left: parse_usize(left, "XOR left source")?,
            right: parse_usize(right, "XOR right source")?,
        }),
        ["and", left, right] => Ok(Rule::And {
            left: parse_usize(left, "AND left source")?,
            right: parse_usize(right, "AND right source")?,
        }),
        ["constant", "0"] => Ok(Rule::Constant(false)),
        ["constant", "1"] => Ok(Rule::Constant(true)),
        _ => Err(format!("unsupported rule specification {spec:?}")),
    }
}

fn parse_input_state_weights(spec: &str) -> Result<Vec<u64>, String> {
    spec.split(',')
        .map(|token| {
            token.parse::<u64>()
                .map_err(|error| format!("invalid input-state weight {token:?}: {error}"))
        })
        .collect()
}

fn parse_task_target(spec: &str) -> Result<TaskTarget, String> {
    let parts = spec.split(':').collect::<Vec<_>>();
    match parts.as_slice() {
        ["xor-inputs", left, right] => Ok(TaskTarget::XorInputs {
            left: parse_usize(left, "task XOR left input")?,
            right: parse_usize(right, "task XOR right input")?,
        }),
        ["and-inputs", left, right] => Ok(TaskTarget::AndInputs {
            left: parse_usize(left, "task AND left input")?,
            right: parse_usize(right, "task AND right input")?,
        }),
        ["copy-input", index] => Ok(TaskTarget::CopyInput {
            index: parse_usize(index, "task copied input")?,
        }),
        ["constant", "0"] => Ok(TaskTarget::Constant(false)),
        ["constant", "1"] => Ok(TaskTarget::Constant(true)),
        _ => Err(format!("unsupported task target specification {spec:?}")),
    }
}

fn parse_interventions(spec: &str) -> Result<Vec<Intervention>, String> {
    if spec.trim().is_empty() {
        return Err("intervention specification must not be empty".into());
    }
    spec.split(';')
        .map(|token| {
            let fields = token.split(':').collect::<Vec<_>>();
            if fields.len() != 4 {
                return Err(format!("intervention must be id:source:target:cost, got {token:?}"));
            }
            Ok(Intervention {
                id: fields[0].to_string(),
                source: parse_usize(fields[1], "intervention source")?,
                target: parse_usize(fields[2], "intervention target")?,
                cost: fields[3].parse::<u32>()
                    .map_err(|error| format!("invalid intervention cost {:?}: {error}", fields[3]))?,
            })
        })
        .collect()
}

/// Load executable fixture definitions from one versioned manifest.
fn fixtures() -> Result<Vec<Fixture>, String> {
    const MANIFEST: &str = include_str!("../validation/fixtures/tccip_toy_expectations.tsv");
    const SCHEMA: &str = "# schema=tccip-toy-expectations-v3";
    if !MANIFEST.lines().any(|line| line.trim() == SCHEMA) {
        return Err(format!("fixture manifest must declare {SCHEMA}"));
    }

    let mut parsed = Vec::new();
    let mut names = HashSet::new();
    for (line_number, line) in MANIFEST.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 19 {
            return Err(format!(
                "manifest line {} has {} fields; expected 19",
                line_number + 1, fields.len()
            ));
        }
        if fields[0].trim().is_empty() || !names.insert(fields[0].to_string()) {
            return Err(format!("manifest line {} has empty or duplicate fixture name", line_number + 1));
        }

        let input_state_weights = parse_input_state_weights(fields[5])?;
        let rules = fields[6].split(',').map(parse_rule)
            .collect::<Result<Vec<_>, _>>()?;
        let components = fields[7].chars().map(|label| match label {
            'A' => Ok(0),
            'B' => Ok(1),
            _ => Err(format!("invalid component label {label:?} in {}", fields[0])),
        }).collect::<Result<Vec<u8>, String>>()?;

        let fixture = Fixture {
            name: fields[0].to_string(),
            network: Network {
                rules,
                output_node: parse_usize(fields[3], "output node")?,
                input_count: parse_usize(fields[2], "input count")?,
                components,
            },
            task_target: parse_task_target(fields[4])?,
            input_state_weights,
            interventions: parse_interventions(fields[8])?,
        };
        fixture.validate()?;
        if fixture.network.rules.len() != parse_usize(fields[1], "node count")? {
            return Err(format!("node count disagrees with rules in fixture {}", fixture.name));
        }
        if fixture.input_state_weights.iter().map(|weight| weight.to_string()).collect::<Vec<_>>().join(",") != fields[5] {
            return Err(format!("noncanonical input-state weights in fixture {}", fixture.name));
        }
        if fixture.network.canonical_rules() != fields[6] {
            return Err(format!("noncanonical rule list in fixture {}", fixture.name));
        }
        let component_labels = fixture.network.components.iter()
            .map(|component| if *component == 0 { 'A' } else { 'B' })
            .collect::<String>();
        if component_labels != fields[7] {
            return Err(format!("component labels disagree with parsed network in {}", fixture.name));
        }
        if fixture.task_target.canonical() != fields[4] {
            return Err(format!("noncanonical task target in fixture {}", fixture.name));
        }
        if fixture.canonical_interventions() != fields[8] {
            return Err(format!("noncanonical interventions in fixture {}", fixture.name));
        }
        let partition_mask = fixture.network.components.iter().enumerate().fold(
            0usize,
            |acc, (index, component)| acc | ((*component as usize) << index),
        );
        if partition_mask != parse_usize(fields[9], "partition mask")? {
            return Err(format!("partition mask disagrees with labels in fixture {}", fixture.name));
        }
        for (column, field_name) in [
            (10, "intact utility"),
            (11, "cross-cut mean loss"),
            (12, "control mean loss"),
            (13, "TCCIP"),
            (17, "minimum partition TCCIP"),
            (18, "maximum partition TCCIP"),
        ] {
            let value = fields[column].parse::<f64>()
                .map_err(|error| format!("invalid {field_name} in {}: {error}", fixture.name))?;
            if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
                return Err(format!("out-of-range {field_name} in fixture {}", fixture.name));
            }
        }
        let expected_counts = [
            parse_usize(fields[14], "partition count")?,
            parse_usize(fields[15], "usable partition count")?,
            parse_usize(fields[16], "skipped partition count")?,
        ];
        let expected_values = [
            fields[10].parse::<f64>().map_err(|error| format!("invalid intact utility: {error}"))?,
            fields[11].parse::<f64>().map_err(|error| format!("invalid cross-cut mean loss: {error}"))?,
            fields[12].parse::<f64>().map_err(|error| format!("invalid control mean loss: {error}"))?,
            fields[13].parse::<f64>().map_err(|error| format!("invalid TCCIP: {error}"))?,
        ];
        let (results, profile) = fixture.run()?;
        let point = profile.first().ok_or_else(|| format!("no profile point for {}", fixture.name))?;
        let actual_values = [
            point.intact_utility,
            point.cross_cut_mean_loss,
            point.matched_control_mean_loss,
            point.tccip,
        ];
        for (index, (actual, expected)) in actual_values.iter().zip(expected_values.iter()).enumerate() {
            if !actual.is_finite() || (actual - expected).abs() > 1e-12 {
                return Err(format!(
                    "fixture {} oracle mismatch at metric {}: computed {}, expected {}",
                    fixture.name, index, actual, expected
                ));
            }
        }
        let (partition_profiles, total, skipped) =
            all_partition_profiles(&fixture.name, fixture.network.rules.len(), &results);
        let actual_counts = [total, partition_profiles.len(), skipped];
        if actual_counts != expected_counts {
            return Err(format!(
                "fixture {} partition-count oracle mismatch: computed {:?}, expected {:?}",
                fixture.name, actual_counts, expected_counts
            ));
        }
        let min_tccip = partition_profiles.iter().map(|point| point.tccip)
            .fold(f64::INFINITY, f64::min);
        let max_tccip = partition_profiles.iter().map(|point| point.tccip)
            .fold(f64::NEG_INFINITY, f64::max);
        let expected_min = fields[17].parse::<f64>()
            .map_err(|error| format!("invalid minimum partition TCCIP: {error}"))?;
        let expected_max = fields[18].parse::<f64>()
            .map_err(|error| format!("invalid maximum partition TCCIP: {error}"))?;
        if (min_tccip - expected_min).abs() > 1e-12 || (max_tccip - expected_max).abs() > 1e-12 {
            return Err(format!(
                "fixture {} partition-range oracle mismatch: computed [{}, {}], expected [{}, {}]",
                fixture.name, min_tccip, max_tccip, expected_min, expected_max
            ));
        }
        parsed.push(fixture);
    }
    if parsed.is_empty() {
        return Err("fixture manifest contains no cases".into());
    }
    Ok(parsed)
}

fn main() {
    println!("# schema=tccip-exact-finite-state-v2");
    let source_revision = std::env::var("SYMTHAEA_SOURCE_SHA")
        .unwrap_or_else(|_| "UNBOUND".to_string());
    println!("# source_revision={source_revision}");
    println!("# UNBOUND means exploratory output only; bind SYMTHAEA_SOURCE_SHA to the exact git HEAD for a reproducibility receipt");
    println!("# input-state integer weights are enumerated exactly; cut source is replaced by an independent fair coin");
    println!("# component labels: 0=A, 1=B; intervention kind is derived from source/target labels");
    println!("fixture\tintervention\tkind\tsource\ttarget\tcost\tintact_utility\tutility\tutility_loss");
    let mut any_failed = false;
    let loaded_fixtures = match fixtures() {
        Ok(fixtures) => fixtures,
        Err(error) => {
            eprintln!("NOT QUALIFIED: cannot load fixture manifest: {error}");
            std::process::exit(2);
        }
    };

    for fixture in loaded_fixtures {
        if let Err(error) = fixture.validate() {
            eprintln!("NOT QUALIFIED: fixture {}: {error}", fixture.name);
            any_failed = true;
            continue;
        }
        let information = fixture.information_baseline();
        println!(
            "# information_baseline fixture={} target_entropy_bits={:.12} individual_input_mi_bits={:?} joint_input_mi_bits={:.12}",
            fixture.name,
            information.target_entropy_bits,
            information.individual_input_mi_bits,
            information.joint_input_mi_bits
        );
        for interaction in fixture.pairwise_utility_interactions() {
            println!(
                "# utility_interaction fixture={} first={} second={} intact={:.12} first_cut={:.12} second_cut={:.12} joint_cut={:.12} contrast={:.12}",
                interaction.fixture,
                interaction.first_id,
                interaction.second_id,
                interaction.intact_utility,
                interaction.first_cut_utility,
                interaction.second_cut_utility,
                interaction.joint_cut_utility,
                interaction.interaction_contrast
            );
        }
        println!(
            "# fixture={} target={} output_node={} input_count={} input_state_weights={:?} components={:?} rules={}",
            fixture.name,
            fixture.task_target.canonical(),
            fixture.network.output_node,
            fixture.network.input_count,
            fixture.input_state_weights,
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
        fixtures().unwrap()
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
        for fixture in fixtures().unwrap() {
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
        let mut fixture = fixtures().unwrap().remove(0);
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
        for fixture in fixtures().unwrap() {
            let (first, first_profile) = fixture.run().unwrap();
            let (second, second_profile) = fixture.run().unwrap();
            assert_eq!(first, second);
            assert_eq!(first_profile, second_profile);
            assert!(first.iter().all(|row| (0.0..=1.0).contains(&row.utility)));
        }
    }

    #[test]
    fn intervention_cost_must_be_positive_and_ids_unique() {
        let mut fixture = fixtures().unwrap().remove(0);
        fixture.interventions.push(Intervention {
            id: "cross_left_into_task_xor".into(),
            source: 2,
            target: 4,
            cost: 1,
        });
        assert!(fixture.validate().unwrap_err().contains("unique"));

        let mut fixture = fixtures().unwrap().remove(1);
        fixture.interventions[0].cost = 0;
        assert!(fixture.validate().unwrap_err().contains("positive cost"));
    }

    #[test]
    fn exhaustive_partition_sweep_contains_the_declared_partition_result() {
        for fixture in fixtures().unwrap() {
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
        for fixture in fixtures().unwrap() {
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
        let fixture = fixtures().unwrap().into_iter()
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
        let fixture = fixtures().unwrap().into_iter()
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
    #[test]
    fn pairwise_utility_interaction_detects_xor_complementarity() {
        let fixture = fixtures().unwrap().into_iter()
            .find(|fixture| fixture.name == "role_matched_xor_synergy_uniform_prior")
            .unwrap();
        let interactions = fixture.pairwise_utility_interactions();
        assert_eq!(interactions.len(), 6);
        let pair = interactions.iter().find(|item| {
            item.first_id == "cross_cut_xor_left" && item.second_id == "matched_internal_xor_right"
        }).unwrap();
        assert!((pair.intact_utility - 1.0).abs() < 1e-12);
        assert!((pair.first_cut_utility - 0.5).abs() < 1e-12);
        assert!((pair.second_cut_utility - 0.5).abs() < 1e-12);
        assert!((pair.joint_cut_utility - 0.5).abs() < 1e-12);
        assert!((pair.interaction_contrast - 0.5).abs() < 1e-12);
    }

    #[test]
    fn pairwise_utility_interaction_manifest_matches_all_exact_joint_cut_values() {
        const MANIFEST: &str =
            include_str!("../validation/fixtures/tccip_pairwise_utility_interactions.tsv");
        const SCHEMA: &str = "# schema=tccip-pairwise-utility-interaction-expectations-v1";
        assert!(MANIFEST.lines().any(|line| line.trim() == SCHEMA));

        let fixtures = fixtures().unwrap();
        let mut seen = HashSet::new();
        for line in MANIFEST.lines().map(str::trim).filter(|line| {
            !line.is_empty() && !line.starts_with('#')
        }) {
            let fields = line.split('\t').collect::<Vec<_>>();
            assert_eq!(fields.len(), 8, "bad pairwise interaction row: {line}");
            let key = (fields[0].to_string(), fields[1].to_string(), fields[2].to_string());
            assert!(seen.insert(key.clone()), "duplicate pairwise interaction row: {key:?}");
            let fixture = fixtures.iter()
                .find(|fixture| fixture.name == fields[0])
                .expect("pairwise manifest fixture must exist");
            let first = fixture.interventions.iter()
                .find(|item| item.id == fields[1])
                .expect("first intervention must exist");
            let second = fixture.interventions.iter()
                .find(|item| item.id == fields[2])
                .expect("second intervention must exist");

            let actual = [
                fixture.utility(None),
                fixture.utility(Some(first)),
                fixture.utility(Some(second)),
                fixture.utility_for_cuts(&[first, second]),
            ];
            let expected = [
                fields[3].parse::<f64>().unwrap(),
                fields[4].parse::<f64>().unwrap(),
                fields[5].parse::<f64>().unwrap(),
                fields[6].parse::<f64>().unwrap(),
            ];
            for (index, (got, wanted)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - wanted).abs() < 1e-12,
                    "fixture {} pair ({}, {}) joint outcome {} expected {}, got {}",
                    fields[0], fields[1], fields[2], index, wanted, got
                );
            }
            let contrast = actual[0] - actual[1] - actual[2] + actual[3];
            let expected_contrast = fields[7].parse::<f64>().unwrap();
            assert!((contrast - expected_contrast).abs() < 1e-12);
        }
        assert_eq!(seen.len(), 16, "pairwise manifest should cover all 16 declared pairs");
    }

    #[test]
    fn duplicate_intervention_edges_are_rejected_before_pairwise_interactions() {
        let mut fixture = fixtures().unwrap().into_iter()
            .find(|fixture| fixture.name == "role_matched_xor_synergy_uniform_prior")
            .unwrap();
        fixture.interventions.push(Intervention {
            id: "duplicate_edge_other_id".into(),
            source: 2,
            target: 4,
            cost: 1,
        });
        assert!(fixture.validate().unwrap_err().contains("duplicate intervention edge"));
    }

    #[test]
    fn pairwise_utility_interaction_is_zero_for_unused_branch_controls() {
        let fixture = fixtures().unwrap().into_iter()
            .find(|fixture| fixture.name == "irrelevant_cross_component_connectivity")
            .unwrap();
        let interactions = fixture.pairwise_utility_interactions();
        assert_eq!(interactions.len(), 1);
        assert!(interactions[0].interaction_contrast.abs() < 1e-12);
    }

    #[test]
    fn xor_synergy_baseline_detects_joint_information_when_each_input_mi_is_zero() {
        let fixture = fixtures().unwrap().into_iter()
            .find(|fixture| fixture.name == "role_matched_xor_synergy_uniform_prior")
            .unwrap();
        let information = fixture.information_baseline();
        assert!((information.target_entropy_bits - 1.0).abs() < 1e-12);
        assert_eq!(information.individual_input_mi_bits.len(), 2);
        assert!(information.individual_input_mi_bits.iter().all(|mi| mi.abs() < 1e-12));
        assert!((information.joint_input_mi_bits - 1.0).abs() < 1e-12);
        let (_, profile) = fixture.run().unwrap();
        assert!((profile[0].tccip).abs() < 1e-12);
    }

    #[test]
    fn same_causal_graph_prior_change_reverses_task_conditioned_score() {
        let fixtures = fixtures().unwrap();
        let high_x1 = fixtures.iter()
            .find(|fixture| fixture.name == "task_relevant_high_x1_prior").unwrap();
        let high_x0 = fixtures.iter()
            .find(|fixture| fixture.name == "task_relevant_high_x0_prior").unwrap();

        assert_eq!(high_x1.network.canonical_rules(), high_x0.network.canonical_rules());
        assert_eq!(high_x1.network.components, high_x0.network.components);
        assert_eq!(high_x1.task_target.canonical(), high_x0.task_target.canonical());
        assert_eq!(high_x1.canonical_interventions(), high_x0.canonical_interventions());
        assert_ne!(high_x1.input_state_weights, high_x0.input_state_weights);

        let (_, high_profile) = high_x1.run().unwrap();
        let (_, low_profile) = high_x0.run().unwrap();
        assert!((high_profile[0].cross_cut_mean_loss - 0.4).abs() < 1e-12);
        assert!((high_profile[0].matched_control_mean_loss - 0.1).abs() < 1e-12);
        assert!((high_profile[0].tccip - 0.3).abs() < 1e-12);
        assert!((low_profile[0].cross_cut_mean_loss - 0.1).abs() < 1e-12);
        assert!((low_profile[0].matched_control_mean_loss - 0.4).abs() < 1e-12);
        assert!((low_profile[0].tccip + 0.3).abs() < 1e-12);
    }

    #[test]
    fn declarative_oracle_manifest_matches_fixture_rules_and_metrics() {
        const MANIFEST: &str =
            include_str!("../validation/fixtures/tccip_toy_expectations.tsv");
        let all_fixtures = fixtures().unwrap();
        let mut seen = HashSet::new();

        for line in MANIFEST.lines().map(str::trim).filter(|line| {
            !line.is_empty() && !line.starts_with('#')
        }) {
            let fields = line.split('\t').collect::<Vec<_>>();
            assert_eq!(fields.len(), 19, "bad manifest row: {line}");
            assert!(seen.insert(fields[0].to_string()), "duplicate fixture {}", fields[0]);
            let fixture = all_fixtures.iter()
                .find(|fixture| fixture.name == fields[0])
                .expect("manifest fixture must exist in the evaluator");

            assert_eq!(fixture.network.rules.len(), fields[1].parse::<usize>().unwrap());
            assert_eq!(fixture.network.input_count, fields[2].parse::<usize>().unwrap());
            assert_eq!(fixture.network.output_node, fields[3].parse::<usize>().unwrap());
            assert_eq!(fixture.task_target.canonical(), fields[4]);
            assert_eq!(fixture.input_state_weights.iter().map(|weight| weight.to_string()).collect::<Vec<_>>().join(","), fields[5]);
            assert_eq!(fixture.network.canonical_rules(), fields[6]);
            let component_labels = fixture.network.components.iter()
                .map(|component| if *component == 0 { 'A' } else { 'B' })
                .collect::<String>();
            assert_eq!(component_labels, fields[7]);
            assert_eq!(fixture.canonical_interventions(), fields[8]);
            let mask = fixture.network.components.iter().enumerate().fold(
                0usize,
                |acc, (index, component)| acc | ((*component as usize) << index),
            );
            assert_eq!(mask, fields[9].parse::<usize>().unwrap());

            let (results, declared) = fixture.run().unwrap();
            let point = &declared[0];
            let close = |actual: f64, col: usize| {
                let expected = fields[col].parse::<f64>().unwrap();
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "{} column {} expected {}, got {}",
                    fixture.name,
                    col,
                    expected,
                    actual
                );
            };
            close(point.intact_utility, 10);
            close(point.cross_cut_mean_loss, 11);
            close(point.matched_control_mean_loss, 12);
            close(point.tccip, 13);

            let (sweep, total, skipped) =
                all_partition_profiles(&fixture.name, fixture.network.rules.len(), &results);
            assert_eq!(total, fields[14].parse::<usize>().unwrap());
            assert_eq!(sweep.len(), fields[15].parse::<usize>().unwrap());
            assert_eq!(skipped, fields[16].parse::<usize>().unwrap());
            close(
                sweep.iter().map(|point| point.tccip).fold(f64::INFINITY, f64::min),
                17,
            );
            close(
                sweep.iter().map(|point| point.tccip).fold(f64::NEG_INFINITY, f64::max),
                18,
            );
        }

        assert_eq!(seen.len(), all_fixtures.len(), "manifest must cover every fixture");
    }

}
