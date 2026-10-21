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
    fn canonical_interventions(&self) -> String {
        self.interventions.iter()
            .map(|cut| format!("{}:{}:{}:{}", cut.id, cut.source, cut.target, cut.cost))
            .collect::<Vec<_>>()
            .join(";")
    }

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
    const SCHEMA: &str = "# schema=tccip-toy-expectations-v2";
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
        if fields.len() != 18 {
            return Err(format!(
                "manifest line {} has {} fields; expected 18",
                line_number + 1, fields.len()
            ));
        }
        if fields[0].trim().is_empty() || !names.insert(fields[0].to_string()) {
            return Err(format!("manifest line {} has empty or duplicate fixture name", line_number + 1));
        }

        let rules = fields[5].split(',').map(parse_rule)
            .collect::<Result<Vec<_>, _>>()?;
        let components = fields[6].chars().map(|label| match label {
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
            interventions: parse_interventions(fields[7])?,
        };
        fixture.validate()?;
        if fixture.network.rules.len() != parse_usize(fields[1], "node count")? {
            return Err(format!("node count disagrees with rules in fixture {}", fixture.name));
        }
        if fixture.network.canonical_rules() != fields[5] {
            return Err(format!("noncanonical rule list in fixture {}", fixture.name));
        }
        let component_labels = fixture.network.components.iter()
            .map(|component| if *component == 0 { 'A' } else { 'B' })
            .collect::<String>();
        if component_labels != fields[6] {
            return Err(format!("component labels disagree with parsed network in {}", fixture.name));
        }
        if fixture.task_target.canonical() != fields[4] {
            return Err(format!("noncanonical task target in fixture {}", fixture.name));
        }
        if fixture.canonical_interventions() != fields[7] {
            return Err(format!("noncanonical interventions in fixture {}", fixture.name));
        }
        let partition_mask = fixture.network.components.iter().enumerate().fold(
            0usize,
            |acc, (index, component)| acc | ((*component as usize) << index),
        );
        if partition_mask != parse_usize(fields[8], "partition mask")? {
            return Err(format!("partition mask disagrees with labels in fixture {}", fixture.name));
        }
        for (column, field_name) in [
            (9, "intact utility"),
            (10, "cross-cut mean loss"),
            (11, "control mean loss"),
            (12, "TCCIP"),
            (16, "minimum partition TCCIP"),
            (17, "maximum partition TCCIP"),
        ] {
            let value = fields[column].parse::<f64>()
                .map_err(|error| format!("invalid {field_name} in {}: {error}", fixture.name))?;
            if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
                return Err(format!("out-of-range {field_name} in fixture {}", fixture.name));
            }
        }
        for (column, field_name) in [
            (13, "partition count"),
            (14, "usable partition count"),
            (15, "skipped partition count"),
        ] {
            parse_usize(fields[column], field_name)?;
        }
        parsed.push(fixture);
    }
    if parsed.is_empty() {
        return Err("fixture manifest contains no cases".into());
    }
    Ok(parsed)
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
    fn declarative_oracle_manifest_matches_fixture_rules_and_metrics() {
        const MANIFEST: &str =
            include_str!("../validation/fixtures/tccip_toy_expectations.tsv");
        let all_fixtures = fixtures().unwrap();
        let mut seen = HashSet::new();

        for line in MANIFEST.lines().map(str::trim).filter(|line| {
            !line.is_empty() && !line.starts_with('#')
        }) {
            let fields = line.split('\t').collect::<Vec<_>>();
            assert_eq!(fields.len(), 18, "bad manifest row: {line}");
            assert!(seen.insert(fields[0].to_string()), "duplicate fixture {}", fields[0]);
            let fixture = all_fixtures.iter()
                .find(|fixture| fixture.name == fields[0])
                .expect("manifest fixture must exist in the evaluator");

            assert_eq!(fixture.network.rules.len(), fields[1].parse::<usize>().unwrap());
            assert_eq!(fixture.network.input_count, fields[2].parse::<usize>().unwrap());
            assert_eq!(fixture.network.output_node, fields[3].parse::<usize>().unwrap());
            assert_eq!(fixture.task_target.canonical(), fields[4]);
            assert_eq!(fixture.network.canonical_rules(), fields[5]);
            let component_labels = fixture.network.components.iter()
                .map(|component| if *component == 0 { 'A' } else { 'B' })
                .collect::<String>();
            assert_eq!(component_labels, fields[6]);
            assert_eq!(fixture.canonical_interventions(), fields[7]);
            let mask = fixture.network.components.iter().enumerate().fold(
                0usize,
                |acc, (index, component)| acc | ((*component as usize) << index),
            );
            assert_eq!(mask, fields[8].parse::<usize>().unwrap());

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
            close(point.intact_utility, 9);
            close(point.cross_cut_mean_loss, 10);
            close(point.matched_control_mean_loss, 11);
            close(point.tccip, 12);

            let (sweep, total, skipped) =
                all_partition_profiles(&fixture.name, fixture.network.rules.len(), &results);
            assert_eq!(total, fields[13].parse::<usize>().unwrap());
            assert_eq!(sweep.len(), fields[14].parse::<usize>().unwrap());
            assert_eq!(skipped, fields[15].parse::<usize>().unwrap());
            close(
                sweep.iter().map(|point| point.tccip).fold(f64::INFINITY, f64::min),
                16,
            );
            close(
                sweep.iter().map(|point| point.tccip).fold(f64::NEG_INFINITY, f64::max),
                17,
            );
        }

        assert_eq!(seen.len(), all_fixtures.len(), "manifest must cover every fixture");
    }

}
