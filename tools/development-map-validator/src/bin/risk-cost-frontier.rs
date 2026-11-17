//! D6.17.2 evaluator-only integration of robust evidence risk and acquisition cost.
//!
//! Reuses the D6.17 noisy-channel evaluator for risk qualification and adds an
//! evaluator-authoritative cost overlay. Cost is the sum of per-attempt nearest-rank
//! p95 bounds; it is not claimed to be the empirical p95 of a complete trajectory.

#[allow(dead_code)]
#[path = "noisy-channel-risk.rs"]
mod noisy_channel;

use chrono::{DateTime, FixedOffset};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::{env, fs, process};

const CLAIM_CEILING: &str = "Synthetic evaluator joining D6.17 robust risk reports with evaluator-side per-attempt cost distributions. The sum of per-attempt nearest-rank p95 values is a conservative cost bound, not an empirical end-to-end p95. This does not measure real runner costs, prove globally optimal adaptive acquisition, or authorize production actions.";

#[derive(Debug, Clone)]
struct CostOption {
    available_at: DateTime<FixedOffset>,
    samples: BTreeMap<String, Vec<u64>>,
}
#[derive(Debug, Clone)]
struct AttemptPlan {
    option_id: String,
    modes: Vec<String>,
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a serde_json::Map<String, Value>, String> {
    value.as_object().ok_or_else(|| format!("{label} must be an object"))
}
fn allowed_keys(value: &Value, allowed: &[&str], label: &str) -> Result<(), String> {
    for key in object(value, label)?.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("{label} contains unknown field {key:?}"));
        }
    }
    Ok(())
}
fn required_value<'a>(value: &'a Value, key: &str, label: &str) -> Result<&'a Value, String> {
    value.get(key).ok_or_else(|| format!("{label}.{key} is required"))
}
fn required_str(value: &Value, key: &str, label: &str) -> Result<String, String> {
    required_value(value, key, label)?.as_str()
        .filter(|s| !s.trim().is_empty()).map(str::to_owned)
        .ok_or_else(|| format!("{label}.{key} must be a non-empty string"))
}
fn required_u64(value: &Value, key: &str, label: &str) -> Result<u64, String> {
    required_value(value, key, label)?.as_u64()
        .ok_or_else(|| format!("{label}.{key} must be a non-negative integer"))
}
fn parse_time(value: &Value, key: &str, label: &str) -> Result<DateTime<FixedOffset>, String> {
    let raw = required_str(value, key, label)?;
    DateTime::parse_from_rfc3339(&raw)
        .map_err(|e| format!("{label}.{key} must be RFC 3339: {e}"))
}
fn p95(samples: &[u64]) -> Result<u64, String> {
    if samples.is_empty() { return Err("cost sample list must not be empty".to_owned()); }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let rank = (sorted.len() * 95 + 99) / 100;
    Ok(sorted[rank.saturating_sub(1)])
}
fn parse_cost_samples(value: &Value, label: &str) -> Result<BTreeMap<String, Vec<u64>>, String> {
    allowed_keys(value, &["cold", "warm", "retry"], label)?;
    let mut output = BTreeMap::new();
    for mode in ["cold", "warm", "retry"] {
        let values = required_value(value, mode, label)?.as_array()
            .ok_or_else(|| format!("{label}.{mode} must be an array"))?;
        if values.is_empty() { return Err(format!("{label}.{mode} is empty; unknown cost is not zero")); }
        let mut parsed = Vec::new();
        for (index, item) in values.iter().enumerate() {
            parsed.push(item.as_u64().filter(|n| *n > 0)
                .ok_or_else(|| format!("{label}.{mode}[{index}] must be a positive integer"))?);
        }
        output.insert(mode.to_owned(), parsed);
    }
    Ok(output)
}
fn parse_cost_oracle(value: &Value) -> Result<(DateTime<FixedOffset>, u64, BTreeMap<String, CostOption>), String> {
    allowed_keys(value, &["decision_cutoff", "budget_micros", "options"], "cost_oracle")?;
    let cutoff = parse_time(value, "decision_cutoff", "cost_oracle")?;
    let budget = required_u64(value, "budget_micros", "cost_oracle")?;
    let values = required_value(value, "options", "cost_oracle")?.as_array()
        .ok_or("cost_oracle.options must be an array")?;
    if values.is_empty() || values.len() > 32 {
        return Err("cost_oracle.options must contain 1..=32 entries".to_owned());
    }
    let mut output = BTreeMap::new();
    for (index, item) in values.iter().enumerate() {
        let label = format!("cost_oracle.options[{index}]");
        allowed_keys(item, &["option_id", "available_at", "cost_samples_micros"], &label)?;
        let id = required_str(item, "option_id", &label)?;
        let available_at = parse_time(item, "available_at", &label)?;
        let samples = parse_cost_samples(required_value(item, "cost_samples_micros", &label)?, &format!("{label}.cost_samples_micros"))?;
        if output.insert(id.clone(), CostOption { available_at, samples }).is_some() {
            return Err(format!("duplicate cost option {id:?}"));
        }
    }
    Ok((cutoff, budget, output))
}
fn parse_attempt_plan(value: &Value, label: &str) -> Result<Vec<AttemptPlan>, String> {
    let values = value.as_array().ok_or_else(|| format!("{label} must be an array"))?;
    if values.len() > 32 { return Err(format!("{label} exceeds 32 acquisition plans")); }
    let mut output = Vec::new();
    let mut seen = BTreeSet::new();
    for (index, item) in values.iter().enumerate() {
        let at = format!("{label}[{index}]");
        allowed_keys(item, &["option_id", "modes"], &at)?;
        let option_id = required_str(item, "option_id", &at)?;
        if !seen.insert(option_id.clone()) {
            return Err(format!("{label} contains duplicate option {option_id:?}"));
        }
        let raw_modes = required_value(item, "modes", &at)?.as_array()
            .ok_or_else(|| format!("{at}.modes must be an array"))?;
        if raw_modes.is_empty() || raw_modes.len() > 8 {
            return Err(format!("{at}.modes must contain 1..=8 attempts"));
        }
        let mut modes = Vec::new();
        for (attempt, mode) in raw_modes.iter().enumerate() {
            let mode = mode.as_str().filter(|s| !s.trim().is_empty())
                .ok_or_else(|| format!("{at}.modes[{attempt}] must be a non-empty string"))?;
            if !["cold", "warm", "retry"].contains(&mode) {
                return Err(format!("{at}.modes[{attempt}] must be cold, warm or retry"));
            }
            if attempt == 0 && mode != "cold" {
                return Err(format!("{at}: first attempt must be cold"));
            }
            if attempt > 0 && mode == "cold" {
                return Err(format!("{at}: repeat attempts cannot be cold"));
            }
            modes.push(mode.to_owned());
        }
        output.push(AttemptPlan { option_id, modes });
    }
    Ok(output)
}
fn evaluate_cost_frontier_parts_json(
    evidence_candidate_raw: &str,
    evidence_oracle_raw: &str,
    cost_candidate_raw: &str,
    cost_oracle_raw: &str
) -> Result<Value, String> {
    let evidence_candidate: Value = serde_json::from_str(evidence_candidate_raw)
        .map_err(|e| format!("invalid D6.17 candidate JSON: {e}"))?;
    let evidence_oracle: Value = serde_json::from_str(evidence_oracle_raw)
        .map_err(|e| format!("invalid D6.17 oracle JSON: {e}"))?;
    let cost_candidate: Value = serde_json::from_str(cost_candidate_raw)
        .map_err(|e| format!("invalid cost candidate JSON: {e}"))?;
    let cost_oracle: Value = serde_json::from_str(cost_oracle_raw)
        .map_err(|e| format!("invalid cost oracle JSON: {e}"))?;

    allowed_keys(&cost_candidate, &["scenario_id", "decision_cutoff", "budget_micros", "cost_estimates", "execution_traces", "comparison_groups"], "cost_candidate")?;
    let scenario_id = required_str(&cost_candidate, "scenario_id", "cost_candidate")?;
    let cutoff = parse_time(&cost_candidate, "decision_cutoff", "cost_candidate")?;
    let budget = required_u64(&cost_candidate, "budget_micros", "cost_candidate")?;
    let evidence_scenario = required_str(&evidence_candidate, "scenario_id", "evidence_candidate")?;
    let evidence_cutoff = parse_time(&evidence_candidate, "decision_cutoff", "evidence_candidate")?;
    if scenario_id != evidence_scenario { return Err("cost/evidence scenario_id mismatch".to_owned()); }
    if cutoff != evidence_cutoff { return Err("cost/evidence candidate cutoff mismatch".to_owned()); }
    let (oracle_cutoff, oracle_budget, cost_options) = parse_cost_oracle(&cost_oracle)?;
    if cutoff != oracle_cutoff { return Err("cost candidate cutoff differs from evaluator-authoritative cutoff".to_owned()); }
    if budget != oracle_budget { return Err("candidate cost budget differs from evaluator-authoritative budget".to_owned()); }

    let visible_options = required_value(&evidence_candidate, "options", "evidence_candidate")?
        .as_array().ok_or("evidence_candidate.options must be an array")?;
    if visible_options.len() != cost_options.len() {
        return Err("cost oracle option set differs from evidence candidate option set".to_owned());
    }
    let mut visible_availability = BTreeMap::new();
    for (index, item) in visible_options.iter().enumerate() {
        let label = format!("evidence_candidate.options[{index}]");
        let id = required_str(item, "option_id", &label)?;
        let available_at = parse_time(item, "available_at", &label)?;
        visible_availability.insert(id, available_at);
    }
    for (id, option) in &cost_options {
        if visible_availability.get(id) != Some(&option.available_at) {
            return Err(format!("cost option {id:?} availability differs from evidence candidate"));
        }
    }

    let estimate_values = required_value(&cost_candidate, "cost_estimates", "cost_candidate")?
        .as_array().ok_or("cost_candidate.cost_estimates must be an array")?;
    let mut candidate_estimates = BTreeMap::new();
    for (index, item) in estimate_values.iter().enumerate() {
        let label = format!("cost_candidate.cost_estimates[{index}]");
        allowed_keys(item, &["option_id", "estimated_cost_upper_micros"], &label)?;
        let id = required_str(item, "option_id", &label)?;
        let estimate = required_u64(item, "estimated_cost_upper_micros", &label)?;
        if candidate_estimates.insert(id.clone(), estimate).is_some() {
            return Err(format!("duplicate candidate cost estimate for {id:?}"));
        }
    }
    if candidate_estimates.keys().cloned().collect::<BTreeSet<_>>()
        != cost_options.keys().cloned().collect::<BTreeSet<_>>()
    {
        return Err("candidate cost estimate IDs must match evaluator cost option IDs exactly".to_owned());
    }

    let execution_values = required_value(&cost_candidate, "execution_traces", "cost_candidate")?
        .as_array().ok_or("cost_candidate.execution_traces must be an array")?;
    let evidence_traces = required_value(&evidence_candidate, "traces", "evidence_candidate")?
        .as_array().ok_or("evidence_candidate.traces must be an array")?;
    if execution_values.len() != evidence_traces.len() {
        return Err("cost execution traces must match evidence trace cardinality exactly".to_owned());
    }
    let mut evidence_by_id = BTreeMap::new();
    let mut decision_by_id = BTreeMap::new();
    for trace in evidence_traces {
        let id = required_str(trace, "trace_id", "evidence_trace")?;
        let acquisitions = required_value(trace, "acquisitions", "evidence_trace")?.as_array()
            .ok_or("evidence_trace.acquisitions must be an array")?;
        let mut ids = BTreeSet::new();
        for step in acquisitions {
            ids.insert(required_str(step, "option_id", "evidence_acquisition")?);
        }
        let decision = required_value(trace, "decision", "evidence_trace")?;
        let kind = required_str(decision, "kind", "evidence_decision")?;
        let action_id = required_str(decision, "action_id", "evidence_decision")?;
        if evidence_by_id.insert(id.clone(), ids).is_some() {
            return Err(format!("duplicate evidence trace_id {id:?}"));
        }
        decision_by_id.insert(id, (kind, action_id));
    }

    let risk_report = noisy_channel::evaluate_noisy_channel_parts_json(evidence_candidate_raw, evidence_oracle_raw)?;
    let risk_reports = risk_report["traces"].as_array()
        .ok_or("D6.17 risk evaluator returned no traces")?;
    let mut risk_by_id = BTreeMap::new();
    for report in risk_reports {
        let id = report["trace_id"].as_str().ok_or("D6.17 report trace_id is missing")?.to_owned();
        risk_by_id.insert(id, report.clone());
    }

    let mut cost_reports = Vec::new();
    let mut execution_by_id = BTreeMap::new();
    for (index, item) in execution_values.iter().enumerate() {
        let label = format!("cost_candidate.execution_traces[{index}]");
        allowed_keys(item, &["trace_id", "acquisitions"], &label)?;
        let id = required_str(item, "trace_id", &label)?;
        if execution_by_id.contains_key(&id) { return Err(format!("duplicate cost execution trace {id:?}")); }
        let plans = parse_attempt_plan(required_value(item, "acquisitions", &label)?, &format!("{label}.acquisitions"))?;
        let expected_ids = evidence_by_id.get(&id).ok_or_else(|| format!("cost trace {id:?} has no evidence trace"))?;
        let planned_ids: BTreeSet<String> = plans.iter().map(|p| p.option_id.clone()).collect();
        if &planned_ids != expected_ids {
            return Err(format!("cost attempts for trace {id:?} do not match its evidence acquisition set exactly"));
        }

        let mut total_cost = 0u64;
        let mut cold_cost = 0u64;
        let mut warm_cost = 0u64;
        let mut retry_cost = 0u64;
        let mut attempt_count = 0usize;
        let mut step_reports = Vec::new();
        let mut estimate_undercoverage_count = 0usize;
        for plan in &plans {
            let option = cost_options.get(&plan.option_id).ok_or_else(|| format!("missing evaluator cost option {:?}", plan.option_id))?;
            let mut step_cost = 0u64;
            let mut attempts = Vec::new();
            for mode in &plan.modes {
                let samples = option.samples.get(mode)
                    .ok_or_else(|| format!("missing {mode} cost samples for {:?}", plan.option_id))?;
                let bound = p95(samples)?;
                total_cost = total_cost.checked_add(bound).ok_or("total cost overflow")?;
                step_cost = step_cost.checked_add(bound).ok_or("step cost overflow")?;
                match mode.as_str() {
                    "cold" => cold_cost = cold_cost.checked_add(bound).ok_or("cold cost overflow")?,
                    "warm" => warm_cost = warm_cost.checked_add(bound).ok_or("warm cost overflow")?,
                    "retry" => retry_cost = retry_cost.checked_add(bound).ok_or("retry cost overflow")?,
                    _ => return Err(format!("unsupported execution mode {mode:?}")),
                }
                attempt_count += 1;
                attempts.push(json!({"mode":mode,"nearest_rank_p95_cost_bound_micros":bound,"sample_count":samples.len()}));
            }
            let candidate_estimate = candidate_estimates[&plan.option_id];
            let estimate_undercovered = candidate_estimate < step_cost;
            if estimate_undercovered { estimate_undercoverage_count += 1; }
            step_reports.push(json!({
                "option_id":plan.option_id,
                "attempts":attempts,
                "cost_bound_micros":step_cost,
                "candidate_estimate_upper_micros":candidate_estimate,
                "estimate_undercovered":estimate_undercovered
            }));
        }
        let risk = risk_by_id.get(&id).ok_or_else(|| format!("D6.17 risk report missing trace {id:?}"))?;
        let (kind, action_id) = decision_by_id.get(&id).ok_or_else(|| format!("decision metadata missing for {id:?}"))?;
        let risk_ok = kind.as_str() == "act" && risk["evidence_accepted"].as_bool() == Some(true)
            && risk["robust_action_justified"].as_bool() == Some(true);
        let over_budget = total_cost > budget;
        let feasible = risk_ok && !over_budget;
        execution_by_id.insert(id.clone(), (total_cost, risk["worst_case_posterior_risk_bps"].as_u64().unwrap_or(10_000), feasible));
        cost_reports.push(json!({
            "trace_id":id,
            "action_id":action_id,
            "decision_kind":kind,
            "nominal_risk_bps":risk["nominal_posterior_risk_bps"],
            "prior_only_robust_risk_bps":risk["prior_only_robust_risk_bps"],
            "likelihood_only_robust_risk_bps":risk["likelihood_only_robust_risk_bps"],
            "worst_case_posterior_risk_bps":risk["worst_case_posterior_risk_bps"],
            "risk_envelope_bps":risk["max_residual_risk_bps"],
            "risk_qualified":risk_ok,
            "cost_bound_micros":total_cost,
            "cold_cost_bound_micros":cold_cost,
            "warm_cost_bound_micros":warm_cost,
            "retry_cost_bound_micros":retry_cost,
            "attempt_count":attempt_count,
            "cost_estimate_undercoverage_count":estimate_undercoverage_count,
            "within_budget":!over_budget,
            "budget_micros":budget,
            "feasible":feasible,
            "cost_steps":step_reports,
            "cost_interpretation":"sum_of_per_attempt_nearest_rank_p95_bounds_not_empirical_trajectory_p95"
        }));
    }
    let trace_ids: BTreeSet<String> = execution_by_id.keys().cloned().collect();
    if trace_ids != evidence_by_id.keys().cloned().collect() {
        return Err("cost execution trace IDs must equal evidence trace IDs".to_owned());
    }

    let comparison_values = required_value(&cost_candidate, "comparison_groups", "cost_candidate")?
        .as_array().ok_or("cost_candidate.comparison_groups must be an array")?;
    let mut comparison_reports = Vec::new();
    let mut comparison_ids = BTreeSet::new();
    for (index, group) in comparison_values.iter().enumerate() {
        let label = format!("cost_candidate.comparison_groups[{index}]");
        allowed_keys(group, &["comparison_id", "trace_ids"], &label)?;
        let comparison_id = required_str(group, "comparison_id", &label)?;
        if !comparison_ids.insert(comparison_id.clone()) { return Err(format!("duplicate comparison_id {comparison_id:?}")); }
        let raw_ids = required_value(group, "trace_ids", &label)?.as_array()
            .ok_or_else(|| format!("{label}.trace_ids must be an array"))?;
        if raw_ids.len() < 2 { return Err(format!("{label}.trace_ids must contain at least two plans")); }
        let mut selected_ids = BTreeSet::new();
        let mut expected_action: Option<&str> = None;
        let mut plans = Vec::new();
        for raw in raw_ids {
            let id = raw.as_str().filter(|s| !s.trim().is_empty())
                .ok_or_else(|| format!("{label}.trace_ids must contain non-empty strings"))?;
            if !selected_ids.insert(id.to_owned()) { return Err(format!("{label} contains duplicate trace {id:?}")); }
            let (kind, action_id) = decision_by_id.get(id).ok_or_else(|| format!("{label} references unknown trace {id:?}"))?;
            if kind.as_str() != "act" { return Err(format!("{label} may compare only act decisions")); }
            if expected_action.is_some_and(|expected| expected != action_id.as_str()) {
                return Err(format!("{label} must compare the same action across plans"));
            }
            expected_action = Some(action_id.as_str());
            let (cost, risk, feasible) = execution_by_id.get(id).ok_or_else(|| format!("{label} missing cost report for {id:?}"))?;
            plans.push(json!({"trace_id":id,"cost_bound_micros":cost,"worst_case_risk_bps":risk,"feasible":feasible}));
        }
        let feasible_ids: Vec<(String,u64,u64)> = plans.iter().filter_map(|p| {
            if p["feasible"] == true {
                Some((
                    p["trace_id"].as_str()?.to_owned(),
                    p["cost_bound_micros"].as_u64()?,
                    p["worst_case_risk_bps"].as_u64()?
                ))
            } else { None }
        }).collect();
        let mut frontier = Vec::new();
        for (id, cost, risk) in &feasible_ids {
            let dominated = feasible_ids.iter().any(|(other_id, other_cost, other_risk)| {
                other_id != id && *other_cost <= *cost && *other_risk <= *risk
                    && (*other_cost < *cost || *other_risk < *risk)
            });
            if !dominated { frontier.push(id.clone()); }
        }
        frontier.sort();
        comparison_reports.push(json!({
            "comparison_id":comparison_id,
            "action_id":expected_action,
            "plans":plans,
            "feasible_plan_count":feasible_ids.len(),
            "pareto_frontier_trace_ids":frontier
        }));
    }

    Ok(json!({
        "scenario_id":scenario_id,
        "adjudication_status":"synthetic_known_answer_robust_risk_cost_evaluation_only",
        "risk_report_status":risk_report["adjudication_status"],
        "trace_count":cost_reports.len(),
        "budget_micros":budget,
        "summary":{
            "robust_risk_qualified_action_count":cost_reports.iter().filter(|r|r["risk_qualified"]==true).count(),
            "within_budget_count":cost_reports.iter().filter(|r|r["within_budget"]==true).count(),
            "feasible_action_count":cost_reports.iter().filter(|r|r["feasible"]==true).count(),
            "over_budget_count":cost_reports.iter().filter(|r|r["within_budget"]==false).count(),
            "cost_estimate_undercoverage_events":cost_reports.iter()
                .map(|r|r["cost_estimate_undercoverage_count"].as_u64().unwrap_or(0)).sum::<u64>(),
            "comparison_group_count":comparison_reports.len()
        },
        "traces":cost_reports,
        "comparison_groups":comparison_reports,
        "decision_rule":"lexicographic: hard safety and robust risk first; budget admissibility second; cost/risk Pareto comparison only among feasible actions",
        "cost_interpretation":"Conservative sum of evaluator-side per-attempt nearest-rank p95 bounds. This is not an empirical p95 of full-trajectory total cost; joint runner receipts are required for that claim.",
        "claim_ceiling":CLAIM_CEILING
    }))
}

fn main() {
    let args:Vec<String>=env::args().skip(1).collect();
    if args.len()!=4 || args[0]=="--help" || args[0]=="-h" {
        eprintln!("Usage: risk-cost-frontier <evidence-candidate.json> <evidence-oracle.json> <cost-candidate.json> <cost-oracle.json>");
        eprintln!("The cost oracle must remain evaluator-only; the agent sees only candidate views.");
        process::exit(if args.len()==4 {0} else {2});
    }
    let read=|path:&str,label:&str|->String {
        match fs::read_to_string(path) {
            Ok(value)=>value,
            Err(error)=>{eprintln!("ERROR D6.17.2-{label}-IO-001 {path}: {error}");process::exit(2)}
        }
    };
    let evidence_candidate=read(&args[0],"EVIDENCE-CANDIDATE");
    let evidence_oracle=read(&args[1],"EVIDENCE-ORACLE");
    let cost_candidate=read(&args[2],"COST-CANDIDATE");
    let cost_oracle=read(&args[3],"COST-ORACLE");
    match evaluate_cost_frontier_parts_json(&evidence_candidate,&evidence_oracle,&cost_candidate,&cost_oracle) {
        Ok(report)=>match serde_json::to_string_pretty(&report) {
            Ok(output)=>println!("{output}"),
            Err(error)=>{eprintln!("ERROR D6.17.2-JSON-001: {error}");process::exit(2)}
        },
        Err(error)=>{eprintln!("ERROR D6.17.2-RISK-COST-001: {error}");process::exit(1)}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const EVIDENCE_CANDIDATE:&str=include_str!("../../../../docs/development-map/fixtures/egddm-noisy-channel-candidate-v0.1.json");
    const EVIDENCE_ORACLE:&str=include_str!("../../../../docs/development-map/fixtures/egddm-noisy-channel-oracle-v0.1.json");
    const COST_CANDIDATE:&str=include_str!("../../../../docs/development-map/fixtures/egddm-risk-cost-candidate-v0.1.json");
    const COST_ORACLE:&str=include_str!("../../../../docs/development-map/fixtures/egddm-risk-cost-oracle-v0.1.json");
    const EXPECTED:&str=include_str!("../../../../docs/development-map/fixtures/egddm-risk-cost-v0.1.expected.json");

    fn report()->Value {
        evaluate_cost_frontier_parts_json(EVIDENCE_CANDIDATE,EVIDENCE_ORACLE,COST_CANDIDATE,COST_ORACLE)
            .expect("D6.17.2 fixture evaluates")
    }

    #[test]
    fn complete_cost_frontier_matches_expected_report() {
        let expected:Value=serde_json::from_str(EXPECTED).expect("expected report JSON");
        assert_eq!(report(),expected);
    }

    #[test]
    fn nearest_rank_p95_selects_max_for_three_samples() {
        assert_eq!(p95(&[100,120,130]).unwrap(),130);
    }

    #[test]
    fn p95_rejects_unknown_cost_not_zero() {
        assert!(p95(&[]).is_err());
    }

    #[test]
    fn first_attempt_must_be_cold_and_repeats_cannot_be_cold() {
        let bad_first=json!([{"option_id":"x","modes":["warm"]}]);
        assert!(parse_attempt_plan(&bad_first,"test").unwrap_err().contains("first attempt must be cold"));
        let bad_repeat=json!([{"option_id":"x","modes":["cold","cold"]}]);
        assert!(parse_attempt_plan(&bad_repeat,"test").unwrap_err().contains("repeat attempts cannot be cold"));
    }

    #[test]
    fn risk_failure_cannot_be_traded_for_lower_cost() {
        let r=report();
        let trace=r["traces"].as_array().unwrap().iter().find(|x|x["trace_id"]=="correlated-sensors-not-independent").unwrap();
        assert_eq!(trace["risk_qualified"],false);
        assert_eq!(trace["feasible"],false);
    }

    #[test]
    fn pareto_frontier_contains_only_risk_and_budget_feasible_plans() {
        let r=report();
        let group=&r["comparison_groups"][0];
        let frontier=group["pareto_frontier_trace_ids"].as_array().unwrap();
        assert!(!frontier.is_empty());
        for id in frontier {
            let text=id.as_str().unwrap();
            let trace=r["traces"].as_array().unwrap().iter().find(|x|x["trace_id"]==text).unwrap();
            assert_eq!(trace["feasible"],true);
        }
    }
}
