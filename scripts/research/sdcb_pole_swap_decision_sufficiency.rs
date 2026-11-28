//! Decision-sufficiency adapter fixture for the frozen SDCB pole-swap pair.
//!
//! This validates tiny terminal-loss/support partitions only. It does not
//! implement a policy tree, experiment selector, likelihood engine or evidence
//! authority. Adaptive policy value remains owned by #13567.
//!
//! Two tasks:
//! - decision invariant: both worlds uniquely prefer maintain; loss 0;
//! - decision sensitive: fast/slow path is world-specific; unresolved minimax
//!   loss 1, but a genuinely disambiguating inlet/reference probe reduces it 0.
//!
//! A Bernoulli(1/2) nuisance probe has one bit outcome entropy but zero mutual
//! information with the world and zero terminal decision value.

use std::cmp::Ordering;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Q { n: i128, d: i128 }

impl Q {
    fn new(n: i128, d: i128) -> Self {
        assert_ne!(d, 0);
        let (n, d) = if d < 0 { (-n, -d) } else { (n, d) };
        let g = gcd(n.abs(), d);
        Self { n: n / g, d: d / g }
    }
    fn zero() -> Self { Self::new(0, 1) }
    fn one() -> Self { Self::new(1, 1) }
    fn add(self, rhs: Self) -> Self {
        Self::new(self.n * rhs.d + rhs.n * self.d, self.d * rhs.d)
    }
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.n * rhs.d - rhs.n * self.d, self.d * rhs.d)
    }
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.n * rhs.n, self.d * rhs.d)
    }
    fn cmp(self, rhs: Self) -> Ordering {
        (self.n * rhs.d).cmp(&(rhs.n * self.d))
    }
}
impl fmt::Display for Q {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.d == 1 { write!(f, "{}", self.n) }
        else { write!(f, "{}/{}", self.n, self.d) }
    }
}
fn gcd(mut a: i128, mut b: i128) -> i128 {
    while b != 0 { let r = a % b; a = b; b = r; }
    a.max(1)
}

const WORLD_A: u8 = 0b01;
const WORLD_B: u8 = 0b10;
const BOTH: u8 = WORLD_A | WORLD_B;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Task { DecisionInvariant, DecisionSensitive }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Probe {
    PassiveAutocovariance,
    UpstreamCascade,
    SensorInletStep,
    SensorInletFrequency,
    IndependentLatentReference,
    HighEntropyNuisance,
}

impl Probe {
    fn id(self) -> &'static str {
        match self {
            Self::PassiveAutocovariance => "passive_sensor_output_autocovariance",
            Self::UpstreamCascade => "upstream_command_first_order_relaxation_cascade",
            Self::SensorInletStep => "traceable_sensor_inlet_step",
            Self::SensorInletFrequency => "traceable_sensor_inlet_frequency_sweep",
            Self::IndependentLatentReference => "independent_reference_of_latent_process",
            Self::HighEntropyNuisance => "high_entropy_world_independent_nuisance_control",
        }
    }

    /// Each entry is the set of worlds still compatible with one observable
    /// outcome. Equal-law probes retain the full equivalence class.
    fn child_supports(self) -> &'static [u8] {
        match self {
            Self::PassiveAutocovariance | Self::UpstreamCascade => &[BOTH],
            Self::SensorInletStep | Self::SensorInletFrequency
            | Self::IndependentLatentReference => &[WORLD_A, WORLD_B],
            Self::HighEntropyNuisance => &[BOTH, BOTH],
        }
    }

    fn resolves_declared_world_pair(self) -> bool {
        matches!(
            self,
            Self::SensorInletStep | Self::SensorInletFrequency
                | Self::IndependentLatentReference
        )
    }

    fn cost(self) -> ResourceVector {
        match self {
            Self::PassiveAutocovariance => ResourceVector::zero(),
            Self::UpstreamCascade => ResourceVector { elapsed_slots: 1, reference_units: 0, actuator_energy_units: 1, compute_units: 0 },
            Self::SensorInletStep => ResourceVector { elapsed_slots: 1, reference_units: 1, actuator_energy_units: 1, compute_units: 0 },
            Self::SensorInletFrequency => ResourceVector { elapsed_slots: 2, reference_units: 1, actuator_energy_units: 2, compute_units: 0 },
            Self::IndependentLatentReference => ResourceVector { elapsed_slots: 1, reference_units: 2, actuator_energy_units: 0, compute_units: 0 },
            Self::HighEntropyNuisance => ResourceVector { elapsed_slots: 1, reference_units: 0, actuator_energy_units: 0, compute_units: 0 },
        }
    }

    fn nuisance_entropy_bits(self) -> Option<u8> {
        if self == Self::HighEntropyNuisance { Some(1) } else { None }
    }

    fn world_mutual_information_bits(self) -> Option<u8> {
        if self == Self::HighEntropyNuisance { Some(0) } else { None }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ResourceVector {
    elapsed_slots: u8,
    reference_units: u8,
    actuator_energy_units: u8,
    compute_units: u8,
}
impl ResourceVector {
    fn zero() -> Self { Self::default() }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LossTable {
    action_a: [u8; 2],
    action_b: [u8; 2],
}

fn loss_table(task: Task) -> LossTable {
    match task {
        Task::DecisionInvariant => LossTable {
            action_a: [0, 0],
            action_b: [1, 1],
        },
        Task::DecisionSensitive => LossTable {
            action_a: [0, 1],
            action_b: [1, 0],
        },
    }
}

fn action_name(task: Task, action: usize) -> &'static str {
    match (task, action) {
        (Task::DecisionInvariant, 0) => "maintain_safe_configuration",
        (Task::DecisionInvariant, 1) => "replace_sensor",
        (Task::DecisionSensitive, 0) => "select_fast_sensor_path",
        (Task::DecisionSensitive, 1) => "select_slow_sensor_path",
        _ => panic!("invalid action"),
    }
}

fn worlds_in_support(support: u8) -> impl Iterator<Item = usize> {
    (0..2).filter(move |index| support & (1 << index) != 0)
}

/// Same finite min_d max_world L(d,w) terminal semantics used by #13567.
/// This fixture code does not implement a sequential policy tree.
fn robust_terminal_decision(task: Task, support: u8) -> (usize, u8) {
    assert!(support != 0 && support & !BOTH == 0);
    let table = loss_table(task);
    let mut best_action = 0;
    let mut best_loss = u8::MAX;
    for action in 0..2 {
        let losses = if action == 0 { table.action_a } else { table.action_b };
        let worst = worlds_in_support(support).map(|world| losses[world]).max().unwrap();
        if worst < best_loss {
            best_action = action;
            best_loss = worst;
        }
    }
    (best_action, best_loss)
}

fn post_probe_worst_case_loss(task: Task, probe: Probe) -> u8 {
    probe.child_supports().iter()
        .map(|support| robust_terminal_decision(task, *support).1)
        .max().unwrap()
}

fn decision_value(task: Task, probe: Probe) -> i8 {
    let baseline = robust_terminal_decision(task, BOTH).1 as i8;
    baseline - post_probe_worst_case_loss(task, probe) as i8
}

fn mechanism_status(probe: Probe) -> &'static str {
    if probe.resolves_declared_world_pair() {
        "IDENTIFIABLE_WITHIN_FROZEN_TWO_WORLD_MODEL"
    } else {
        "NON_IDENTIFIABLE_EQUIVALENCE_CLASS"
    }
}

fn decision_status(task: Task, support: u8) -> &'static str {
    if robust_terminal_decision(task, support).1 == 0 {
        "DECISION_SUFFICIENT"
    } else {
        "DECISION_SENSITIVE_UNRESOLVED"
    }
}

fn run_fixture() {
    let probes = [
        Probe::PassiveAutocovariance,
        Probe::UpstreamCascade,
        Probe::SensorInletStep,
        Probe::SensorInletFrequency,
        Probe::IndependentLatentReference,
        Probe::HighEntropyNuisance,
    ];
    for task in [Task::DecisionInvariant, Task::DecisionSensitive] {
        let (action, loss) = robust_terminal_decision(task, BOTH);
        println!("task={task:?};baseline_action={};baseline_loss={loss};decision_status={}",
            action_name(task, action), decision_status(task, BOTH));
        for probe in probes {
            let post = post_probe_worst_case_loss(task, probe);
            println!(
                "probe={};mechanism={};post_probe_worst_loss={post};decision_value={};resources={:?}",
                probe.id(), mechanism_status(probe), decision_value(task, probe), probe.cost()
            );
        }
    }
    let nuisance = Probe::HighEntropyNuisance;
    println!(
        "nuisance_entropy_bits={};world_mutual_information_bits={}",
        nuisance.nuisance_entropy_bits().unwrap(),
        nuisance.world_mutual_information_bits().unwrap()
    );
}

fn main() { run_fixture(); }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invariant_task_has_one_unique_action_without_acquisition() {
        let (action, loss) = robust_terminal_decision(Task::DecisionInvariant, BOTH);
        assert_eq!(action, 0);
        assert_eq!(action_name(Task::DecisionInvariant, action), "maintain_safe_configuration");
        assert_eq!(loss, 0);
        assert_eq!(decision_status(Task::DecisionInvariant, BOTH), "DECISION_SUFFICIENT");
    }

    #[test]
    fn mechanism_information_has_zero_decision_value_on_invariant_task() {
        for probe in [Probe::SensorInletStep, Probe::SensorInletFrequency, Probe::IndependentLatentReference] {
            assert!(probe.resolves_declared_world_pair());
            assert_eq!(decision_value(Task::DecisionInvariant, probe), 0);
        }
    }

    #[test]
    fn sensitive_task_is_unresolved_without_disambiguating_evidence() {
        let (action, loss) = robust_terminal_decision(Task::DecisionSensitive, BOTH);
        assert_eq!(action, 0); // deterministic tie-break only; not a unique action claim
        assert_eq!(loss, 1);
        assert_eq!(decision_status(Task::DecisionSensitive, BOTH), "DECISION_SENSITIVE_UNRESOLVED");
    }

    #[test]
    fn passive_and_upstream_probes_do_not_change_compatible_world_support() {
        for probe in [Probe::PassiveAutocovariance, Probe::UpstreamCascade] {
            assert_eq!(probe.child_supports(), &[BOTH]);
            assert!(!probe.resolves_declared_world_pair());
            assert_eq!(decision_value(Task::DecisionSensitive, probe), 0);
        }
    }

    #[test]
    fn sensor_inlet_and_independent_reference_make_the_action_world_identifiable() {
        for probe in [Probe::SensorInletStep, Probe::SensorInletFrequency, Probe::IndependentLatentReference] {
            assert_eq!(probe.child_supports(), &[WORLD_A, WORLD_B]);
            assert_eq!(post_probe_worst_case_loss(Task::DecisionSensitive, probe), 0);
            assert_eq!(decision_value(Task::DecisionSensitive, probe), 1);
        }
    }

    #[test]
    fn one_bit_nuisance_probe_has_zero_world_information_and_zero_decision_value() {
        let probe = Probe::HighEntropyNuisance;
        assert_eq!(probe.nuisance_entropy_bits(), Some(1));
        assert_eq!(probe.world_mutual_information_bits(), Some(0));
        assert_eq!(probe.child_supports(), &[BOTH, BOTH]);
        assert_eq!(decision_value(Task::DecisionSensitive, probe), 0);
        assert_eq!(decision_value(Task::DecisionInvariant, probe), 0);
    }

    #[test]
    fn post_probe_decision_actions_are_correct_on_singleton_support() {
        let (action_a, loss_a) = robust_terminal_decision(Task::DecisionSensitive, WORLD_A);
        let (action_b, loss_b) = robust_terminal_decision(Task::DecisionSensitive, WORLD_B);
        assert_eq!(action_name(Task::DecisionSensitive, action_a), "select_fast_sensor_path");
        assert_eq!(action_name(Task::DecisionSensitive, action_b), "select_slow_sensor_path");
        assert_eq!(loss_a, 0);
        assert_eq!(loss_b, 0);
    }

    #[test]
    fn resource_costs_remain_a_vector_not_a_scalarized_task_loss() {
        let step = Probe::SensorInletStep.cost();
        let frequency = Probe::SensorInletFrequency.cost();
        let reference = Probe::IndependentLatentReference.cost();
        assert_eq!(step, ResourceVector { elapsed_slots: 1, reference_units: 1, actuator_energy_units: 1, compute_units: 0 });
        assert_eq!(frequency, ResourceVector { elapsed_slots: 2, reference_units: 1, actuator_energy_units: 2, compute_units: 0 });
        assert_eq!(reference, ResourceVector { elapsed_slots: 1, reference_units: 2, actuator_energy_units: 0, compute_units: 0 });
    }

    #[test]
    fn no_probe_result_authorizes_physical_execution() {
        assert_eq!(decision_value(Task::DecisionSensitive, Probe::SensorInletStep), 1);
    }
}
