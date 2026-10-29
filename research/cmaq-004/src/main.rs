use std::collections::{BTreeSet, HashMap, HashSet};
use std::{env, fs, process};

const MAX_PROBES: usize = 10_000;

#[derive(Clone, Debug)]
struct Case {
    group: String,
    initial: String,
    max_probes: usize,
    budget: i64,
}
#[derive(Clone, Debug)]
struct Action { reward: i64, safe: bool }
#[derive(Clone, Debug)]
struct Probe { observation: String, cost: i64, safe: bool }
#[derive(Clone, Debug)]
struct Event { step: usize, probe: String, observed: String }
#[derive(Clone, Debug, Default)]
struct Summary {
    policy: String, count: usize, net: i128, regret: u128, cost: i128,
    probe_count: usize, static_sum: i128, unsafe_probes: usize,
    unsafe_actions: usize, forged: usize, over_count: usize,
    over_budget: usize, hard_pass: bool,
}
struct Evaluator {
    cases: HashMap<String, Case>,
    actions: HashMap<(String, u32), Action>,
    probes: HashMap<(String, String), Probe>,
    events: HashMap<(String, String), Vec<Event>>,
    decisions: HashMap<(String, String), u32>,
    policies: Vec<String>,
    static_baseline: HashMap<String, i64>,
}

fn token(s: &str, label: &str) -> Result<String, String> {
    if s.is_empty() || s.chars().any(|c| c.is_control() || c.is_whitespace()) {
        return Err(format!("{label} must be a non-empty token without whitespace"));
    }
    Ok(s.to_owned())
}
fn u32v(s: &str, label: &str) -> Result<u32, String> {
    s.parse().map_err(|_| format!("invalid {label}: {s}"))
}
fn usizev(s: &str, label: &str) -> Result<usize, String> {
    s.parse().map_err(|_| format!("invalid {label}: {s}"))
}
fn i64v(s: &str, label: &str) -> Result<i64, String> {
    s.parse().map_err(|_| format!("invalid {label}: {s}"))
}
fn boolv(s: &str, label: &str) -> Result<bool, String> {
    match s { "true" => Ok(true), "false" => Ok(false), _ => Err(format!("{label} must be true or false")) }
}
fn rows(input: &str, label: &str, expected: &[&str]) -> Result<Vec<Vec<String>>, String> {
    let mut lines = input.lines();
    let header = lines.next().ok_or_else(|| format!("{label}: empty TSV"))?;
    if header.split('\t').collect::<Vec<_>>().as_slice() != expected {
        return Err(format!("{label}: unexpected TSV header"));
    }
    let mut out = Vec::new();
    for (i, line) in lines.enumerate() {
        if line.trim().is_empty() { continue; }
        let row = line.split('\t').map(str::to_owned).collect::<Vec<_>>();
        if row.len() != expected.len() {
            return Err(format!("{label}: line {} has wrong field count", i + 2));
        }
        out.push(row);
    }
    Ok(out)
}

impl Evaluator {
    fn parse(cases_in: &str, actions_in: &str, probes_in: &str, events_in: &str, decisions_in: &str) -> Result<Self, String> {
        let case_rows = rows(cases_in, "cases", &["case_id","group_id","initial_observation","max_probes","probe_budget_milli"])?;
        let action_rows = rows(actions_in, "actions", &["case_id","action_id","reward_milli","safe"])?;
        let probe_rows = rows(probes_in, "probes", &["case_id","probe_id","observation_id","cost_milli","safe"])?;
        let event_rows = rows(events_in, "probe trace", &["case_id","policy_id","step","probe_id","observed_observation_id"])?;
        let decision_rows = rows(decisions_in, "decision trace", &["case_id","policy_id","action_id"])?;

        let mut cases = HashMap::new();
        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        for r in case_rows {
            let id = token(&r[0], "case_id")?;
            let group = token(&r[1], "group_id")?;
            let initial = token(&r[2], "initial_observation")?;
            let max_probes = usizev(&r[3], "max_probes")?;
            let budget = i64v(&r[4], "probe_budget_milli")?;
            if max_probes > MAX_PROBES || budget < 0 { return Err(format!("invalid budget for {id}")); }
            if cases.insert(id.to_string(), Case { group: group.clone(), initial, max_probes, budget }).is_some() {
                return Err(format!("duplicate case {id}"));
            }
            groups.entry(group).or_default().push(id);
        }
        if cases.is_empty() { return Err("cases cannot be empty".to_owned()); }
        for (group, ids) in &groups {
            let initial = cases[&ids[0]].initial.as_str();
            if ids.iter().any(|id| cases[id].initial.as_str() != initial) {
                return Err(format!("group {group} has different initial observations; possible hidden-label leakage"));
            }
        }

        let mut actions = HashMap::new();
        for r in action_rows {
            let id = token(&r[0], "action case_id")?;
            if !cases.contains_key(&id) { return Err(format!("unknown action case {id}")); }
            let action = u32v(&r[1], "action_id")?;
            let value = Action { reward: i64v(&r[2], "reward_milli")?, safe: boolv(&r[3], "action safe")? };
            if actions.insert((id.to_string(), action), value).is_some() { return Err(format!("duplicate action {id}/{action}")); }
        }
        for id in cases.keys() {
            if !actions.keys().any(|(cid, _)| cid == id) { return Err(format!("case {id} has no actions")); }
        }

        let mut probes = HashMap::new();
        for r in probe_rows {
            let id = token(&r[0], "probe case_id")?;
            if !cases.contains_key(&id) { return Err(format!("unknown probe case {id}")); }
            let probe_id = token(&r[1], "probe_id")?;
            let observation = token(&r[2], "observation_id")?;
            let cost = i64v(&r[3], "probe cost")?;
            if cost < 0 { return Err("probe costs cannot be negative".to_owned()); }
            let value = Probe { observation, cost, safe: boolv(&r[4], "probe safe")? };
            if probes.insert((id.to_string(), probe_id.to_string()), value).is_some() {
                return Err(format!("duplicate probe {id}/{probe_id}"));
            }
        }

        let mut events: HashMap<(String, String), Vec<Event>> = HashMap::new();
        let mut seen_steps = HashSet::new();
        for r in event_rows {
            let id = token(&r[0], "trace case_id")?;
            let policy = token(&r[1], "trace policy_id")?;
            let step = usizev(&r[2], "step")?;
            let probe = token(&r[3], "trace probe")?;
            let observed = token(&r[4], "observed signal")?;
            if !cases.contains_key(&id) { return Err(format!("unknown trace case {id}")); }
            if step == 0 || step > MAX_PROBES { return Err(format!("invalid probe step {step}")); }
            if !seen_steps.insert((id.to_string(), policy.clone(), step)) {
                return Err(format!("duplicate step {id}/{policy}/{step}"));
            }
            events.entry((id, policy)).or_default().push(Event { step, probe, observed });
        }
        for ((id, policy), es) in events.iter_mut() {
            es.sort_by_key(|e| e.step);
            for (index, event) in es.iter().enumerate() {
                if event.step != index + 1 { return Err(format!("non-contiguous probe steps for {id}/{policy}")); }
            }
        }

        let mut decisions = HashMap::new();
        let mut policy_set = BTreeSet::new();
        for r in decision_rows {
            let id = token(&r[0], "decision case_id")?;
            let policy = token(&r[1], "policy_id")?;
            let action = u32v(&r[2], "chosen action")?;
            if !cases.contains_key(&id) { return Err(format!("unknown decision case {id}")); }
            if !actions.contains_key(&(id.to_string(), action)) { return Err(format!("unavailable action {id}/{action}")); }
            if decisions.insert((id.to_string(), policy.clone()), action).is_some() { return Err(format!("duplicate decision {id}/{policy}")); }
            policy_set.insert(policy);
        }
        if policy_set.is_empty() { return Err("decision trace cannot be empty".to_owned()); }
        for ((id, policy), es) in &events {
            if !decisions.contains_key(&(id.to_string(), policy.clone())) { return Err(format!("probe trace has no decision {id}/{policy}")); }
            for e in es {
                if !probes.contains_key(&(id.to_string(), e.probe.clone())) { return Err(format!("unknown probe {}/{}/{}", id, policy, e.probe)); }
            }
        }
        for policy in &policy_set {
            for id in cases.keys() {
                if !decisions.contains_key(&(id.to_string(), policy.clone())) { return Err(format!("policy {policy} misses case {id}")); }
            }
        }

        let mut static_baseline = HashMap::new();
        for (group, ids) in &groups {
            let mut candidates = Vec::new();
            for case_id in ids {
                for (candidate_id, action_id) in actions.keys() {
                    if candidate_id == case_id { candidates.push(*action_id); }
                }
            }
            candidates.sort_unstable();
            candidates.dedup();
            let mut best: Option<(u32, i64)> = None;
            for aid in candidates {
                let mut sum = 0_i128;
                let mut safe_everywhere = true;
                for id in ids {
                    match actions.get(&(id.to_string(), aid)) {
                        Some(a) if a.safe => sum += a.reward as i128,
                        _ => { safe_everywhere = false; break; }
                    }
                }
                if safe_everywhere {
                    let mean = (sum / ids.len() as i128) as i64;
                    if mean > 0 && best.map(|(_, v)| mean > v).unwrap_or(true) { best = Some((aid, mean)); }
                }
            }
            for id in ids {
                let baseline = best.and_then(|(aid, _)| actions.get(&(id.to_string(), aid)))
                    .map(|a| if a.safe { a.reward } else { 0 }).unwrap_or(0);
                static_baseline.insert(id.to_string(), baseline);
            }
        }

        Ok(Self {
            cases, actions, probes, events, decisions,
            policies: policy_set.into_iter().collect(),
            static_baseline,
        })
    }

    fn summarize(&self, policy: &str) -> Summary {
        let mut s = Summary { policy: policy.to_owned(), count: self.cases.len(), hard_pass: true, ..Summary::default() };
        let mut ids = self.cases.keys().collect::<Vec<_>>();
        ids.sort();
        for id in ids {
            let case = &self.cases[id];
            let aid = self.decisions[&(id.to_string(), policy.to_owned())];
            let action = &self.actions[&(id.to_string(), aid)];
            let events = self.events.get(&(id.to_string(), policy.to_owned())).map(Vec::as_slice).unwrap_or(&[]);
            let mut cost = 0_i128;
            for e in events {
                let p = &self.probes[&(id.to_string(), e.probe.clone())];
                cost += p.cost as i128;
                s.probe_count += 1;
                if !p.safe { s.unsafe_probes += 1; s.hard_pass = false; }
                if e.observed != p.observation { s.forged += 1; s.hard_pass = false; }
            }
            if events.len() > case.max_probes { s.over_count += 1; s.hard_pass = false; }
            if cost > case.budget as i128 { s.over_budget += 1; s.hard_pass = false; }
            if !action.safe { s.unsafe_actions += 1; s.hard_pass = false; }
            let net = (if action.safe { action.reward as i128 } else { 0 }) - cost;
            let oracle = self.actions.iter()
                .filter_map(|((cid, _), a)| {
                    if cid.as_str() == id.as_str() && a.safe { Some(a.reward) } else { None }
                })
                .max().unwrap_or(0).max(0);
            s.net += net;
            s.regret += (oracle as i128 - net).max(0) as u128;
            s.cost += cost;
            s.static_sum += self.static_baseline[id] as i128;
        }
        s
    }

    fn report(&self) -> String {
        let mut out = String::from("policy_id\tcases\tmean_net_utility_milli\tmean_oracle_regret_milli\tmean_probe_cost_milli\tmean_probes_per_case\tvalue_vs_best_static_action_milli\tunsafe_probes\tunsafe_final_actions\tforged_observations\tover_probe_cases\tover_budget_cases\thard_gate_pass\n");
        for policy in &self.policies {
            let s = self.summarize(policy);
            let n = s.count as f64;
            out.push_str(&format!("{}\t{}\t{:.0}\t{:.0}\t{:.0}\t{:.3}\t{:.0}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                s.policy, s.count, s.net as f64 / n, s.regret as f64 / n, s.cost as f64 / n,
                s.probe_count as f64 / n, s.net as f64 / n - s.static_sum as f64 / n,
                s.unsafe_probes, s.unsafe_actions, s.forged, s.over_count, s.over_budget, s.hard_pass));
        }
        out
    }
}

fn args() -> Result<HashMap<String, String>, String> {
    let a = env::args().skip(1).collect::<Vec<_>>();
    let mut out = HashMap::new();
    let mut i = 0;
    while i < a.len() {
        if !a[i].starts_with("--") || i + 1 >= a.len() { return Err(format!("expected flag/value near {}", a[i])); }
        if out.insert(a[i].clone(), a[i+1].clone()).is_some() { return Err(format!("duplicate flag {}", a[i])); }
        i += 2;
    }
    for k in ["--cases", "--actions", "--probes", "--trace", "--decisions"] {
        if !out.contains_key(k) { return Err(format!("missing flag {k}")); }
    }
    if out.keys().any(|k| !["--cases", "--actions", "--probes", "--trace", "--decisions"].contains(&k.as_str())) {
        return Err("unknown argument".to_owned());
    }
    Ok(out)
}
fn run() -> Result<(), String> {
    let a = args()?;
    let c = fs::read_to_string(&a["--cases"]).map_err(|e| format!("cases: {e}"))?;
    let actions = fs::read_to_string(&a["--actions"]).map_err(|e| format!("actions: {e}"))?;
    let probes = fs::read_to_string(&a["--probes"]).map_err(|e| format!("probes: {e}"))?;
    let trace = fs::read_to_string(&a["--trace"]).map_err(|e| format!("trace: {e}"))?;
    let decisions = fs::read_to_string(&a["--decisions"]).map_err(|e| format!("decisions: {e}"))?;
    print!("{}", Evaluator::parse(&c, &actions, &probes, &trace, &decisions)?.report());
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("cmaq-004: {e}");
        eprintln!("usage: cmaq-004 --cases FILE --actions FILE --probes FILE --trace FILE --decisions FILE");
        process::exit(2);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Evaluator {
        Evaluator::parse(
            include_str!("../fixtures/cases.tsv"),
            include_str!("../fixtures/action-key.tsv"),
            include_str!("../fixtures/probe-key.tsv"),
            include_str!("../fixtures/probe-trace.tsv"),
            include_str!("../fixtures/decision-trace.tsv"),
        ).expect("valid fixture")
    }
    #[test]
    fn adaptive_intervention_beats_best_static_action_after_cost() {
        let s = fixture().summarize("adaptive");
        assert_eq!(s.net, 1_900);
        assert_eq!(s.static_sum, 1_100);
        assert!(s.hard_pass);
    }
    #[test]
    fn probing_low_value_cases_is_not_worth_the_cost() {
        let e = fixture();
        assert!(e.summarize("adaptive").net > e.summarize("probe_all").net);
        assert_eq!(e.summarize("probe_all").net, 1_800);
    }
    #[test]
    fn forged_observation_is_counted_and_fails_gate() {
        let s = fixture().summarize("forged");
        assert_eq!(s.forged, 1);
        assert!(!s.hard_pass);
    }
    #[test]
    fn static_policy_has_zero_value_of_information() {
        let s = fixture().summarize("static_a0");
        assert_eq!(s.net, s.static_sum);
    }
    #[test]
    fn unsafe_probe_fails_gate() {
        let probes = include_str!("../fixtures/probe-key.tsv").replace(
            "q7\tsense\tleft\t100\ttrue",
            "q7\tsense\tleft\t100\tfalse",
        );
        let e = Evaluator::parse(include_str!("../fixtures/cases.tsv"), include_str!("../fixtures/action-key.tsv"), &probes,
            include_str!("../fixtures/probe-trace.tsv"), include_str!("../fixtures/decision-trace.tsv")).expect("parse");
        let s = e.summarize("adaptive");
        assert_eq!(s.unsafe_probes, 1);
        assert!(!s.hard_pass);
    }
    #[test]
    fn exceeded_probe_budget_fails_gate() {
        let cases = include_str!("../fixtures/cases.tsv").replace(
            "group_high\thigh_context\t1\t100", "group_high\thigh_context\t1\t50");
        let e = Evaluator::parse(&cases, include_str!("../fixtures/action-key.tsv"), include_str!("../fixtures/probe-key.tsv"),
            include_str!("../fixtures/probe-trace.tsv"), include_str!("../fixtures/decision-trace.tsv")).expect("parse");
        assert_eq!(e.summarize("adaptive").over_budget, 2);
    }
    #[test]
    fn candidate_bundle_does_not_leak_hidden_labels_or_rewards() {
        let visible = format!(
            "{}{}{}",
            include_str!("../fixtures/candidate-cases.tsv"),
            include_str!("../fixtures/candidate-action-menu.tsv"),
            include_str!("../fixtures/candidate-probe-menu.tsv"),
        );
        for forbidden in ["group_id", "reward_milli", "observation_id", "left", "right"] {
            assert!(!visible.contains(forbidden), "candidate bundle leaks {forbidden}");
        }
    }

    #[test]
    fn missing_policy_case_fails_closed() {
        let d = include_str!("../fixtures/decision-trace.tsv").lines()
            .filter(|line| !line.starts_with("b9\tadaptive\t"))
            .collect::<Vec<_>>().join("\n") + "\n";
        assert!(Evaluator::parse(include_str!("../fixtures/cases.tsv"), include_str!("../fixtures/action-key.tsv"),
            include_str!("../fixtures/probe-key.tsv"), include_str!("../fixtures/probe-trace.tsv"), &d).is_err());
    }
}
