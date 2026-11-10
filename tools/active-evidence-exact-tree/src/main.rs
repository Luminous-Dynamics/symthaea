//! Exact finite-horizon simulator for active evidence acquisition.
//! Research microbenchmark only; not the production Symthaea selector.

use std::{env, process::ExitCode};
type Matrix = [[f64; 3]; 4];
const PRIOR: [f64; 3] = [0.6, 0.3, 0.1];
const LOSS: [[f64; 3]; 3] = [[0., 4., 7.], [5., 0., 3.], [8., 2., 0.]];
const COST: [f64; 4] = [1., 1., 2., 1.];
const NOMINAL: Matrix = [
    [0.88, 0.34, 0.25], [0.35, 0.86, 0.31],
    [0.24, 0.38, 0.88], [0.76, 0.70, 0.18],
];
const Q0_SHIFT: Matrix = [
    [0.88, 0.76, 0.25], [0.35, 0.86, 0.31],
    [0.24, 0.38, 0.88], [0.76, 0.70, 0.18],
];
const Q1_SHIFT: Matrix = [
    [0.88, 0.34, 0.25], [0.83, 0.22, 0.31],
    [0.24, 0.38, 0.88], [0.76, 0.70, 0.18],
];
const COMPOUND: Matrix = [
    [0.88, 0.74, 0.25], [0.79, 0.22, 0.31],
    [0.24, 0.38, 0.88], [0.76, 0.83, 0.18],
];
const OUT_OF_SET: Matrix = [
    [0.15, 0.92, 0.25], [0.35, 0.86, 0.31],
    [0.24, 0.38, 0.88], [0.76, 0.70, 0.18],
];
const AMBIGUITY: [Matrix; 4] = [NOMINAL, Q0_SHIFT, Q1_SHIFT, COMPOUND];
const SCENARIOS: [(&str, Matrix); 5] = [
    ("iid", NOMINAL),
    ("q0_conditional_shift", Q0_SHIFT),
    ("q1_asymmetric_error", Q1_SHIFT),
    ("compound_shift", COMPOUND),
    ("out_of_set_flip", OUT_OF_SET),
];
const EPS: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Policy { Random, Entropy, PluginEvsi, RobustEvsi, KnownChannelEvsi }
impl Policy {
    const ALL: [Self; 5] = [Self::Random, Self::Entropy, Self::PluginEvsi, Self::RobustEvsi, Self::KnownChannelEvsi];
    fn name(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Entropy => "entropy",
            Self::PluginEvsi => "plugin_evsi",
            Self::RobustEvsi => "robust_evsi",
            Self::KnownChannelEvsi => "known_channel_evsi_reference",
        }
    }
    fn update_matrix(self, truth: Matrix) -> Matrix {
        if self == Self::KnownChannelEvsi { truth } else { NOMINAL }
    }
}
#[derive(Clone, Copy, Debug, Default)]
struct Metrics { loss: f64, regret: f64, cost: f64, queries: f64 }
#[derive(Clone, Copy)]
struct Choice { query: usize, probability: f64 }

fn p_one(m: Matrix, q: usize, h: usize) -> f64 { m[q][h] }
fn p_observe(m: Matrix, q: usize, h: usize, y: bool) -> f64 {
    let p = p_one(m, q, h);
    if y { p } else { 1.0 - p }
}
fn normalize(v: [f64; 3]) -> [f64; 3] {
    let s = v.iter().sum::<f64>();
    if s <= 0.0 { PRIOR } else { [v[0] / s, v[1] / s, v[2] / s] }
}
fn posterior(b: [f64; 3], m: Matrix, q: usize, y: bool) -> [f64; 3] {
    normalize([
        b[0] * p_observe(m, q, 0, y),
        b[1] * p_observe(m, q, 1, y),
        b[2] * p_observe(m, q, 2, y),
    ])
}
fn action_risk(b: [f64; 3], a: usize) -> f64 {
    (0..3).map(|h| b[h] * LOSS[h][a]).sum()
}
fn best_action(b: [f64; 3]) -> usize {
    let mut best = 0;
    let mut value = action_risk(b, 0);
    for a in 1..3 {
        let candidate = action_risk(b, a);
        if candidate < value - EPS { best = a; value = candidate; }
    }
    best
}
fn bayes_risk(b: [f64; 3]) -> f64 { action_risk(b, best_action(b)) }
fn entropy(b: [f64; 3]) -> f64 {
    b.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.log2()).sum()
}
fn expected_risk_after(b: [f64; 3], m: Matrix, q: usize) -> f64 {
    [false, true].into_iter().map(|y| {
        let by = posterior(b, m, q, y);
        let py = (0..3).map(|h| b[h] * p_observe(m, q, h, y)).sum::<f64>();
        py * bayes_risk(by)
    }).sum()
}
fn evsi(b: [f64; 3], m: Matrix, q: usize) -> f64 { bayes_risk(b) - expected_risk_after(b, m, q) }
fn entropy_gain(b: [f64; 3], q: usize) -> f64 {
    entropy(b) - [false, true].into_iter().map(|y| {
        let by = posterior(b, NOMINAL, q, y);
        let py = (0..3).map(|h| b[h] * p_observe(NOMINAL, q, h, y)).sum::<f64>();
        py * entropy(by)
    }).sum::<f64>()
}
/// One-step robust query value. Posterior/action updates remain nominal; this is not robust inference.
fn robust_gain(b: [f64; 3], q: usize) -> f64 {
    let a0 = best_action(posterior(b, NOMINAL, q, false));
    let a1 = best_action(posterior(b, NOMINAL, q, true));
    let mut worst = f64::NEG_INFINITY;
    for m in AMBIGUITY {
        let mut expected = 0.;
        for h in 0..3 {
            let p = p_one(m, q, h);
            expected += b[h] * ((1. - p) * LOSS[h][a0] + p * LOSS[h][a1]);
        }
        worst = worst.max(expected);
    }
    bayes_risk(b) - worst
}
fn available(mask: u8) -> Vec<usize> { (0..4).filter(|q| mask & (1 << q) != 0).collect() }
fn select(policy: Policy, b: [f64; 3], mask: u8, truth: Matrix) -> Vec<Choice> {
    let qs = available(mask);
    if qs.is_empty() { return Vec::new(); }
    if policy == Policy::Random {
        let p = 1. / qs.len() as f64;
        return qs.into_iter().map(|query| Choice { query, probability: p }).collect();
    }
    let mut best: Option<(usize, f64, f64)> = None;
    for q in qs {
        let value = match policy {
            Policy::Random => unreachable!(),
            Policy::Entropy => entropy_gain(b, q),
            Policy::PluginEvsi => evsi(b, NOMINAL, q),
            Policy::RobustEvsi => robust_gain(b, q),
            Policy::KnownChannelEvsi => evsi(b, truth, q),
        };
        let score = value / COST[q];
        if best.is_none_or(|(_, s, _)| score > s + EPS) { best = Some((q, score, value)); }
    }
    match best {
        Some((_, _, value)) if policy == Policy::RobustEvsi && value <= EPS => Vec::new(),
        Some((q, _, _)) => vec![Choice { query: q, probability: 1. }],
        None => Vec::new(),
    }
}
fn terminal(h: usize, b: [f64; 3]) -> Metrics {
    let loss = LOSS[h][best_action(b)];
    let oracle = LOSS[h].iter().copied().fold(f64::INFINITY, f64::min);
    Metrics { loss, regret: loss - oracle, cost: 0., queries: 0. }
}
fn recurse(policy: Policy, truth: Matrix, h: usize, b: [f64; 3], mask: u8, rounds: usize) -> Metrics {
    if rounds == 0 || mask == 0 { return terminal(h, b); }
    let choices = select(policy, b, mask, truth);
    if choices.is_empty() { return terminal(h, b); }
    let mut result = Metrics::default();
    let update = policy.update_matrix(truth);
    for choice in choices {
        let next_mask = mask & !(1 << choice.query);
        for y in [false, true] {
            let branch_p = choice.probability * p_observe(truth, choice.query, h, y);
            let next_b = posterior(b, update, choice.query, y);
            let child = recurse(policy, truth, h, next_b, next_mask, rounds - 1);
            result.loss += branch_p * child.loss;
            result.regret += branch_p * child.regret;
            result.cost += branch_p * (COST[choice.query] + child.cost);
            result.queries += branch_p * (1. + child.queries);
        }
    }
    result
}
fn evaluate(policy: Policy, truth: Matrix, rounds: usize) -> Metrics {
    let mut result = Metrics::default();
    for h in 0..3 {
        let m = recurse(policy, truth, h, PRIOR, 0b1111, rounds.min(4));
        result.loss += PRIOR[h] * m.loss;
        result.regret += PRIOR[h] * m.regret;
        result.cost += PRIOR[h] * m.cost;
        result.queries += PRIOR[h] * m.queries;
    }
    result
}
fn write_csv(filter: Option<&str>) -> Result<(), String> {
    if let Some(name) = filter {
        if !SCENARIOS.iter().any(|(n, _)| *n == name) { return Err(format!("unknown scenario: {name}")); }
    }
    println!("scenario,policy,query_cap,expected_loss,expected_decision_regret,expected_query_cost,expected_query_count");
    for (name, matrix) in SCENARIOS {
        if filter.is_some_and(|f| f != name) { continue; }
        for cap in 0..=3 {
            for policy in Policy::ALL {
                let m = evaluate(policy, matrix, cap);
                println!("{name},{},{cap},{:.12},{:.12},{:.12},{:.12}",
                    policy.name(), m.loss, m.regret, m.cost, m.queries);
            }
        }
    }
    Ok(())
}
fn main() -> ExitCode {
    let mut filter: Option<String> = None;
    for arg in env::args().skip(1) {
        if arg == "--help" || arg == "-h" {
            eprintln!("Usage: active-evidence-exact-tree [--scenario=NAME]");
            eprintln!("Scenarios: iid, q0_conditional_shift, q1_asymmetric_error, compound_shift, out_of_set_flip");
            return ExitCode::SUCCESS;
        } else if let Some(name) = arg.strip_prefix("--scenario=") {
            filter = Some(name.to_owned());
        } else {
            eprintln!("unknown argument: {arg}");
            return ExitCode::from(2);
        }
    }
    match write_csv(filter.as_deref()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => { eprintln!("{e}"); ExitCode::from(2) }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn near(a: f64, b: f64) -> bool { (a - b).abs() < 1.0e-9 }

    #[test]
    fn zero_query_baseline_is_exact_for_all_policies_and_scenarios() {
        for policy in Policy::ALL {
            for (_, matrix) in SCENARIOS {
                let m = evaluate(policy, matrix, 0);
                assert!(near(m.loss, 2.3));
                assert!(near(m.regret, 2.3));
                assert!(near(m.cost, 0.));
                assert!(near(m.queries, 0.));
            }
        }
    }
    #[test]
    fn robust_greedy_exposes_a_conservatism_tradeoff() {
        let m = evaluate(Policy::RobustEvsi, NOMINAL, 3);
        assert!(m.regret < 2.3);
        assert!(m.cost <= 1.0 + 1.0e-9);
        assert!(m.queries <= 1.0 + 1.0e-9);
    }
    #[test]
    fn out_of_set_shift_can_make_nominal_evsi_harmful() {
        assert!(evaluate(Policy::PluginEvsi, OUT_OF_SET, 1).regret
            > evaluate(Policy::PluginEvsi, OUT_OF_SET, 0).regret);
    }
    #[test]
    fn exact_outputs_are_finite_and_respect_query_caps() {
        for (_, matrix) in SCENARIOS {
            for cap in 0..=3 {
                for policy in Policy::ALL {
                    let m = evaluate(policy, matrix, cap);
                    assert!(m.loss.is_finite() && m.regret.is_finite());
                    assert!(m.cost.is_finite() && m.queries.is_finite());
                    assert!(m.cost >= 0. && m.queries >= 0.);
                    assert!(m.queries <= cap as f64 + 1.0e-9);
                }
            }
        }
    }
    #[test]
    fn evaluation_is_deterministic() {
        let a = evaluate(Policy::PluginEvsi, COMPOUND, 3);
        let b = evaluate(Policy::PluginEvsi, COMPOUND, 3);
        assert_eq!(a.loss, b.loss);
        assert_eq!(a.regret, b.regret);
        assert_eq!(a.cost, b.cost);
    }
    #[test]
    fn declared_ambiguity_set_does_not_include_out_of_set_control() {
        assert!(!AMBIGUITY.iter().any(|m| *m == OUT_OF_SET));
    }
}
