//! Finite-support Wasserstein distributionally robust terminal decisions.
//!
//! For each terminal loss vector, this module solves the finite transport problem
//! over a 1-Wasserstein ball around a frozen nominal posterior. It emits both a
//! primal transport plan and the scalar dual multiplier. The result is conditional
//! on the declared support, metric and radius; it does not calibrate the ambiguity
//! set or establish that the real data-generating distribution belongs to it.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, error::Error, fmt};

use super::pareto_evsi_oracle::{ExactEvsiError, ExactEvsiProblem, EvsiDecision};

const SET_SCHEMA: &str = "symthaea.architecture-engineering.finite-wasserstein-ambiguity-set";
const SET_VERSION: &str = "0.1.0";
const REPORT_SCHEMA: &str = "symthaea.architecture-engineering.finite-wasserstein-robust-decision";
const REPORT_VERSION: &str = "0.1.0";
const MAX_SUPPORT: usize = 32;
const NUMERIC_TOLERANCE: f64 = 1e-9;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WassersteinDecisionError { pub path: String, pub message: String }
impl WassersteinDecisionError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }
}
impl fmt::Display for WassersteinDecisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}: {}", self.path, self.message) }
}
impl Error for WassersteinDecisionError {}
impl From<ExactEvsiError> for WassersteinDecisionError {
    fn from(error: ExactEvsiError) -> Self { Self { path: error.path, message: error.message } }
}

/// Finite support and metric for a 1-Wasserstein ambiguity set. Matrix rows and
/// columns must follow the exact canonical world order of the associated problem.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FiniteWassersteinAmbiguitySet {
    pub schema: String,
    pub version: String,
    pub model_id: String,
    pub model_sha256: String,
    pub world_ids: Vec<String>,
    pub ground_cost: Vec<Vec<f64>>,
    pub radius: f64,
}
impl FiniteWassersteinAmbiguitySet {
    pub fn new(
        model_id: impl Into<String>, model_sha256: impl Into<String>,
        world_ids: Vec<String>, ground_cost: Vec<Vec<f64>>, radius: f64,
    ) -> Result<Self, WassersteinDecisionError> {
        let value = Self { schema: SET_SCHEMA.into(), version: SET_VERSION.into(),
            model_id: model_id.into(), model_sha256: model_sha256.into(), world_ids,
            ground_cost, radius };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), WassersteinDecisionError> {
        if self.schema != SET_SCHEMA || self.version != SET_VERSION {
            return Err(WassersteinDecisionError::new("ambiguity_set.schema", "unsupported schema/version"));
        }
        if self.model_id.trim().is_empty() {
            return Err(WassersteinDecisionError::new("ambiguity_set.model_id", "ground-metric model ID must be non-empty"));
        }
        validate_sha256(&self.model_sha256, "ambiguity_set.model_sha256")?;
        let n = self.world_ids.len();
        if !(2..=MAX_SUPPORT).contains(&n) {
            return Err(WassersteinDecisionError::new("ambiguity_set.world_ids",
                format!("finite support must contain 2..={MAX_SUPPORT} worlds")));
        }
        let mut ids = BTreeSet::new();
        if self.world_ids.iter().any(|id| id.trim().is_empty() || !ids.insert(id.as_str())) {
            return Err(WassersteinDecisionError::new("ambiguity_set.world_ids", "world IDs must be non-empty and unique"));
        }
        if self.ground_cost.len() != n || self.ground_cost.iter().any(|row| row.len() != n) {
            return Err(WassersteinDecisionError::new("ambiguity_set.ground_cost", "ground-cost matrix must be square and match support"));
        }
        if !self.radius.is_finite() || self.radius < 0.0 {
            return Err(WassersteinDecisionError::new("ambiguity_set.radius", "radius must be finite and non-negative"));
        }
        let mut max_cost = 1.0_f64;
        for row in &self.ground_cost {
            for value in row {
                if !value.is_finite() || *value < 0.0 {
                    return Err(WassersteinDecisionError::new("ambiguity_set.ground_cost", "costs must be finite and non-negative"));
                }
                max_cost = max_cost.max(*value);
            }
        }
        let tolerance = NUMERIC_TOLERANCE * max_cost;
        for i in 0..n {
            for j in 0..n {
                let value = self.ground_cost[i][j];
                if i == j && value.abs() > tolerance {
                    return Err(WassersteinDecisionError::new(format!("ambiguity_set.ground_cost[{i}][{j}]"), "metric diagonal must be zero"));
                }
                if i != j && value <= 0.0 {
                    return Err(WassersteinDecisionError::new(format!("ambiguity_set.ground_cost[{i}][{j}]"), "distinct worlds must have positive separation"));
                }
                if (value - self.ground_cost[j][i]).abs() > tolerance {
                    return Err(WassersteinDecisionError::new(format!("ambiguity_set.ground_cost[{i}][{j}]"), "metric must be symmetric"));
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    if self.ground_cost[i][k] > self.ground_cost[i][j] + self.ground_cost[j][k] + tolerance {
                        return Err(WassersteinDecisionError::new(format!("ambiguity_set.ground_cost[{i}][{k}]"), "metric violates triangle inequality"));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn ambiguity_set_sha256(&self) -> Result<String, WassersteinDecisionError> {
        self.validate()?;
        digest(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WassersteinTransportMass {
    pub source_world_id: String,
    pub target_world_id: String,
    pub probability_mass: f64,
    pub unit_transport_cost: f64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WassersteinWorldProbability { pub world_id: String, pub probability: f64 }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WassersteinDecisionValue {
    pub decision_id: String,
    pub nominal_expected_loss: f64,
    pub worst_case_expected_loss: f64,
    pub transport_cost_used: f64,
    pub dual_radius_multiplier: Option<f64>,
    pub worst_case_distribution: Vec<WassersteinWorldProbability>,
    pub transport_plan: Vec<WassersteinTransportMass>,
    pub dual_upper_bound: f64,
    pub primal_dual_gap: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WassersteinRobustDecisionReport {
    pub schema: String,
    pub version: String,
    pub campaign_id: String,
    pub problem_sha256: String,
    pub ambiguity_set_sha256: String,
    pub nominal_optimal_decision_id: String,
    pub robust_optimal_decision_id: String,
    pub nominal_risk_of_robust_decision: f64,
    pub robust_optimal_worst_case_risk: f64,
    pub ranked_decisions: Vec<WassersteinDecisionValue>,
    pub report_sha256: String,
}
#[derive(Serialize)]
struct ReportPayload<'a> {
    schema: &'a str, version: &'a str, campaign_id: &'a str,
    problem_sha256: &'a str, ambiguity_set_sha256: &'a str,
    nominal_optimal_decision_id: &'a str, robust_optimal_decision_id: &'a str,
    nominal_risk_of_robust_decision: f64, robust_optimal_worst_case_risk: f64,
    ranked_decisions: &'a [WassersteinDecisionValue],
}

pub fn analyze_wasserstein_robust_decision(
    problem: &ExactEvsiProblem, ambiguity: &FiniteWassersteinAmbiguitySet,
) -> Result<WassersteinRobustDecisionReport, WassersteinDecisionError> {
    let problem = problem.clone().canonicalize()?;
    ambiguity.validate()?;
    let world_ids: Vec<String> = problem.worlds.iter().map(|w| w.world_id.clone()).collect();
    if ambiguity.world_ids != world_ids {
        return Err(WassersteinDecisionError::new("ambiguity_set.world_ids", "IDs/order must exactly match canonical posterior support"));
    }
    let nominal: Vec<f64> = problem.worlds.iter().map(|w| w.posterior_probability).collect();
    let nominal_best = nominal_decision(&problem.decisions, &nominal)?;
    let mut ranked = Vec::with_capacity(problem.decisions.len());

    for decision in &problem.decisions {
        let losses: Vec<f64> = decision.losses_by_world.iter().map(|row| row.loss).collect();
        let nominal_risk = compensated_dot(&nominal, &losses);
        let solution = solve_finite_transport(&nominal, &losses, &ambiguity.ground_cost, ambiguity.radius, &world_ids)?;
        let risk = compensated_dot(&solution.distribution, &losses);
        let scale = risk.abs().max(solution.dual_upper_bound.abs()).max(1.0);
        if (risk - solution.dual_upper_bound).abs() > 1e-7 * scale {
            return Err(WassersteinDecisionError::new(format!("decisions.{}.primal_dual_gap", decision.decision_id), "primal plan and dual upper bound disagree"));
        }
        if !nominal_risk.is_finite() || !risk.is_finite() || risk + NUMERIC_TOLERANCE < nominal_risk {
            return Err(WassersteinDecisionError::new(format!("decisions.{}.risk", decision.decision_id), "invalid or reduced worst-case risk"));
        }
        ranked.push(WassersteinDecisionValue {
            decision_id: decision.decision_id.clone(), nominal_expected_loss: nominal_risk,
            worst_case_expected_loss: risk, transport_cost_used: solution.transport_cost,
            dual_radius_multiplier: solution.dual_multiplier,
            worst_case_distribution: world_ids.iter().zip(&solution.distribution)
                .map(|(id,p)| WassersteinWorldProbability { world_id: id.clone(), probability: *p }).collect(),
            transport_plan: solution.transport_plan, dual_upper_bound: solution.dual_upper_bound,
            primal_dual_gap: solution.dual_upper_bound - risk,
        });
    }
    ranked.sort_by(|a,b| a.worst_case_expected_loss.total_cmp(&b.worst_case_expected_loss)
        .then_with(|| a.decision_id.cmp(&b.decision_id)));
    let best = ranked.first().ok_or_else(|| WassersteinDecisionError::new("decisions", "no terminal decisions"))?;
    let robust_id = best.decision_id.clone();
    let robust_nominal = best.nominal_expected_loss;
    let robust_risk = best.worst_case_expected_loss;
    let problem_sha = problem.problem_sha256()?;
    let ambiguity_sha = ambiguity.ambiguity_set_sha256()?;
    let payload = ReportPayload { schema: REPORT_SCHEMA, version: REPORT_VERSION,
        campaign_id: &problem.campaign_id, problem_sha256: &problem_sha,
        ambiguity_set_sha256: &ambiguity_sha, nominal_optimal_decision_id: &nominal_best,
        robust_optimal_decision_id: &robust_id, nominal_risk_of_robust_decision: robust_nominal,
        robust_optimal_worst_case_risk: robust_risk, ranked_decisions: &ranked };
    let report_sha = digest(&payload)?;
    Ok(WassersteinRobustDecisionReport { schema: REPORT_SCHEMA.into(), version: REPORT_VERSION.into(),
        campaign_id: problem.campaign_id, problem_sha256: problem_sha,
        ambiguity_set_sha256: ambiguity_sha, nominal_optimal_decision_id: nominal_best,
        robust_optimal_decision_id: robust_id, nominal_risk_of_robust_decision: robust_nominal,
        robust_optimal_worst_case_risk: robust_risk, ranked_decisions: ranked, report_sha256: report_sha })
}

pub fn verify_wasserstein_robust_decision_report(
    problem: &ExactEvsiProblem, ambiguity: &FiniteWassersteinAmbiguitySet,
    report: &WassersteinRobustDecisionReport,
) -> Result<(), WassersteinDecisionError> {
    if analyze_wasserstein_robust_decision(problem, ambiguity)? != *report {
        return Err(WassersteinDecisionError::new("report", "report differs from reconstruction"));
    }
    Ok(())
}

struct TransportSolution {
    distribution: Vec<f64>,
    transport_plan: Vec<WassersteinTransportMass>,
    transport_cost: f64,
    dual_multiplier: Option<f64>,
    dual_upper_bound: f64,
}

fn solve_finite_transport(
    nominal: &[f64], losses: &[f64], costs: &[Vec<f64>], radius: f64, ids: &[String],
) -> Result<TransportSolution, WassersteinDecisionError> {
    let n = nominal.len();
    if losses.len() != n || costs.len() != n || ids.len() != n {
        return Err(WassersteinDecisionError::new("transport", "dimension mismatch"));
    }
    if radius == 0.0 {
        let plan = (0..n).filter(|i| nominal[*i] > 0.0).map(|i| WassersteinTransportMass {
            source_world_id: ids[i].clone(), target_world_id: ids[i].clone(),
            probability_mass: nominal[i], unit_transport_cost: 0.0,
        }).collect();
        let risk = compensated_dot(nominal, losses);
        return Ok(TransportSolution { distribution: nominal.to_vec(), transport_plan: plan,
            transport_cost: 0.0, dual_multiplier: None, dual_upper_bound: risk });
    }

    // Dual: min(lambda*rho + sum_i p_i max_j(loss_j - lambda*c_ij)), lambda >= 0.
    // This is convex piecewise linear; every finite breakpoint is an intersection
    // of two destination lines for one source row.
    let mut breakpoints = vec![0.0_f64];
    for i in 0..n {
        for j in 0..n {
            for k in (j + 1)..n {
                let denominator = costs[i][j] - costs[i][k];
                if denominator.abs() > 1e-14 {
                    let lambda = (losses[j] - losses[k]) / denominator;
                    if lambda.is_finite() && lambda >= 0.0 { breakpoints.push(lambda); }
                }
            }
        }
    }
    breakpoints.sort_by(f64::total_cmp);
    breakpoints.dedup_by(|a,b| (*a-*b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0));
    let mut best_lambda = 0.0;
    let mut best_dual = f64::INFINITY;
    for lambda in breakpoints {
        let value = dual_objective(nominal, losses, costs, radius, lambda);
        if !value.is_finite() { return Err(WassersteinDecisionError::new("transport.dual", "non-finite dual objective")); }
        if !best_dual.is_finite()
            || value < best_dual - 1e-11 * best_dual.abs().max(value.abs()).max(1.0)
        {
            best_dual = value;
            best_lambda = lambda;
        }
    }

    let (plan_matrix, distribution, transport_cost) =
        build_primal_plan(nominal, losses, costs, radius, best_lambda)?;
    let mut transport_plan = Vec::new();
    for i in 0..n { for j in 0..n {
        if plan_matrix[i][j] > 1e-14 {
            transport_plan.push(WassersteinTransportMass {
                source_world_id: ids[i].clone(), target_world_id: ids[j].clone(),
                probability_mass: plan_matrix[i][j], unit_transport_cost: costs[i][j],
            });
        }
    }}
    let primal = compensated_dot(&distribution, losses);
    let dual = dual_objective(nominal, losses, costs, radius, best_lambda);
    if (primal-dual).abs() > 1e-7 * primal.abs().max(dual.abs()).max(1.0) {
        return Err(WassersteinDecisionError::new("transport.primal_dual",
            format!("primal/dual mismatch: primal={primal}, dual={dual}, lambda={best_lambda}, transport_cost={transport_cost}")));
    }
    Ok(TransportSolution { distribution, transport_plan, transport_cost,
        dual_multiplier: Some(best_lambda), dual_upper_bound: dual })
}

fn dual_objective(p: &[f64], loss: &[f64], cost: &[Vec<f64>], radius: f64, lambda: f64) -> f64 {
    let mut sum = radius * lambda;
    let mut correction = 0.0;
    for i in 0..p.len() {
        let maximum = (0..loss.len()).map(|j| loss[j] - lambda * cost[i][j])
            .fold(f64::NEG_INFINITY, f64::max);
        kahan_add(&mut sum, &mut correction, p[i] * maximum);
    }
    sum
}

fn build_primal_plan(
    p: &[f64], loss: &[f64], cost: &[Vec<f64>], radius: f64, lambda: f64,
) -> Result<(Vec<Vec<f64>>, Vec<f64>, f64), WassersteinDecisionError> {
    let n = p.len();
    let mut plan = vec![vec![0.0_f64; n]; n];
    let mut base_for_row = vec![0usize; n];
    let mut alternatives = Vec::<(f64, usize, usize)>::new();
    let mut base_cost = 0.0;
    let mut base_correction = 0.0;

    for i in 0..n {
        if p[i] <= 0.0 { continue; }
        let reduced: Vec<f64> = (0..n).map(|j| loss[j] - lambda * cost[i][j]).collect();
        let maximum = reduced.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let tolerance = 1e-8 * maximum.abs().max(1.0);
        let maximizers: Vec<usize> = (0..n).filter(|j| (reduced[*j] - maximum).abs() <= tolerance).collect();
        let base = *maximizers.iter().min_by(|a,b| cost[i][**a].total_cmp(&cost[i][**b]).then_with(|| a.cmp(b)))
            .ok_or_else(|| WassersteinDecisionError::new(format!("transport.rows[{i}]"), "no dual maximizer"))?;
        base_for_row[i] = base;
        plan[i][base] = p[i];
        kahan_add(&mut base_cost, &mut base_correction, p[i] * cost[i][base]);
        for target in maximizers {
            if target != base {
                let increment = cost[i][target] - cost[i][base];
                if increment > 1e-14 { alternatives.push((increment, i, target)); }
            }
        }
    }
    let radius_tolerance = 1e-8 * radius.abs().max(base_cost.abs()).max(1.0);
    if base_cost > radius + radius_tolerance {
        return Err(WassersteinDecisionError::new("transport.base_cost", "minimum-cost dual maximizers exceed radius"));
    }
    // At a positive optimal multiplier, dual-tied destinations have equal
    // reduced reward. To attain any required transport budget while respecting
    // each source row's mass cap, shift toward the largest-cost tied destination
    // first. Filling a cheaper alternative first can exhaust that row's mass and
    // leave the primal plan below the complementary-slackness transport budget.
    alternatives.sort_by(|a,b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)).then_with(|| a.2.cmp(&b.2)));
    let mut extra = if lambda > 1e-12 { (radius-base_cost).max(0.0) } else { 0.0 };
    let mut shifted = vec![0.0_f64; n];
    for (increment, source, target) in alternatives {
        if extra <= radius_tolerance { break; }
        let available = (p[source] - shifted[source]).max(0.0);
        if available <= 1e-14 { continue; }
        let amount = available.min(extra / increment);
        if amount <= 0.0 || !amount.is_finite() { continue; }
        plan[source][base_for_row[source]] -= amount;
        plan[source][target] += amount;
        shifted[source] += amount;
        extra = (extra - amount * increment).max(0.0);
    }

    let mut q = vec![0.0_f64; n];
    let mut actual_cost = 0.0;
    let mut cost_correction = 0.0;
    for i in 0..n {
        let row_sum: f64 = plan[i].iter().sum();
        if (row_sum-p[i]).abs() > 1e-7 {
            return Err(WassersteinDecisionError::new(format!("transport.plan[{i}]"), "row sum does not preserve nominal source mass"));
        }
        for j in 0..n {
            let mass = plan[i][j];
            if !mass.is_finite() || mass < -1e-12 {
                return Err(WassersteinDecisionError::new(format!("transport.plan[{i}][{j}]"), "negative or non-finite transport mass"));
            }
            q[j] += mass;
            kahan_add(&mut actual_cost, &mut cost_correction, mass * cost[i][j]);
        }
    }
    if actual_cost > radius + radius_tolerance || (q.iter().sum::<f64>()-1.0).abs() > 1e-7 {
        return Err(WassersteinDecisionError::new("transport.plan", "target distribution invalid or transport cost exceeds radius"));
    }
    Ok((plan, q, actual_cost))
}

fn nominal_decision(decisions: &[EvsiDecision], posterior: &[f64]) -> Result<String, WassersteinDecisionError> {
    let mut best: Option<(f64,String)> = None;
    for decision in decisions {
        let loss: Vec<f64> = decision.losses_by_world.iter().map(|r| r.loss).collect();
        let risk = compensated_dot(posterior, &loss);
        if !risk.is_finite() { return Err(WassersteinDecisionError::new("decisions", "nominal risk non-finite")); }
        match &best {
            None => best=Some((risk,decision.decision_id.clone())),
            Some((value,id)) if risk < *value || (risk == *value && decision.decision_id < *id) =>
                best=Some((risk,decision.decision_id.clone())),
            _ => {}
        }
    }
    best.map(|(_,id)| id).ok_or_else(|| WassersteinDecisionError::new("decisions","no terminal decisions"))
}
fn compensated_dot(a: &[f64], b: &[f64]) -> f64 {
    let (mut sum,mut correction)=(0.0,0.0);
    for (x,y) in a.iter().zip(b) { kahan_add(&mut sum,&mut correction,x*y); }
    sum
}
fn kahan_add(sum: &mut f64, correction: &mut f64, value: f64) {
    let adjusted=value-*correction; let next=*sum+adjusted;
    *correction=(next-*sum)-adjusted; *sum=next;
}
fn validate_sha256(value: &str, path: &str) -> Result<(), WassersteinDecisionError> {
    if value.len()!=64 || !value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err(WassersteinDecisionError::new(path,"expected lowercase 64-character SHA-256"));
    }
    Ok(())
}
fn digest<T: Serialize>(value: &T) -> Result<String,WassersteinDecisionError> {
    let bytes=serde_json::to_vec(value).map_err(|e|WassersteinDecisionError::new("serialization",e.to_string()))?;
    Ok(Sha256::digest(bytes).iter().map(|b|format!("{b:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pareto_evsi_oracle::{EvsiActionOutcome,EvsiDecision,EvsiReplicateAction,EvsiWorldLikelihood,EvsiWorldLoss,EvsiWorldProbability};
    const H1:&str="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const H2:&str="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const H3:&str="cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const H4:&str="dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn problem()->ExactEvsiProblem {
        ExactEvsiProblem {
            schema:"symthaea.architecture-engineering.exact-discrete-evsi-problem".into(),version:"0.1.0".into(),
            campaign_id:"wasserstein-robust-test".into(),posterior_model_id:"posterior-v1".into(),posterior_model_sha256:H1.into(),
            observation_model_id:"observations-v1".into(),observation_model_sha256:H2.into(),
            decision_loss_model_id:"loss-v1".into(),decision_loss_model_sha256:H3.into(),
            cost_model_id:"cost-v1".into(),cost_model_sha256:H4.into(),
            worlds:vec![
                EvsiWorldProbability{world_id:"world-a".into(),posterior_probability:0.9},
                EvsiWorldProbability{world_id:"world-b".into(),posterior_probability:0.1}],
            decisions:vec![
                EvsiDecision{decision_id:"choose-a".into(),losses_by_world:vec![
                    EvsiWorldLoss{world_id:"world-a".into(),loss:0.0},EvsiWorldLoss{world_id:"world-b".into(),loss:40.0}]},
                EvsiDecision{decision_id:"choose-b".into(),losses_by_world:vec![
                    EvsiWorldLoss{world_id:"world-a".into(),loss:10.0},EvsiWorldLoss{world_id:"world-b".into(),loss:0.0}]},
                EvsiDecision{decision_id:"abstain".into(),losses_by_world:vec![
                    EvsiWorldLoss{world_id:"world-a".into(),loss:5.0},EvsiWorldLoss{world_id:"world-b".into(),loss:5.0}]}],
            actions:vec![EvsiReplicateAction{action_id:"dummy".into(),candidate_id:"candidate".into(),objective_id:"quality".into(),
                incremental_cost_units:1,outcomes:vec![
                    EvsiActionOutcome{outcome_id:"a".into(),likelihoods_by_world:vec![
                        EvsiWorldLikelihood{world_id:"world-a".into(),probability:1.0},EvsiWorldLikelihood{world_id:"world-b".into(),probability:0.0}]},
                    EvsiActionOutcome{outcome_id:"b".into(),likelihoods_by_world:vec![
                        EvsiWorldLikelihood{world_id:"world-a".into(),probability:0.0},EvsiWorldLikelihood{world_id:"world-b".into(),probability:1.0}]}]}],
        }
    }
    fn ambiguity(distance:f64,radius:f64)->FiniteWassersteinAmbiguitySet {
        FiniteWassersteinAmbiguitySet::new("two-world-line-v1",H4.into(),
            vec!["world-a".into(),"world-b".into()],vec![vec![0.0,distance],vec![distance,0.0]],radius).unwrap()
    }

    #[test]
    fn metric_radius_and_distance_control_worst_case_mass_transfer() {
        let p=problem(); let set=ambiguity(2.0,0.2);
        let report=analyze_wasserstein_robust_decision(&p,&set).unwrap();
        let a=report.ranked_decisions.iter().find(|v|v.decision_id=="choose-a").unwrap();
        assert!((a.worst_case_distribution[0].probability-0.8).abs()<1e-9);
        assert!((a.worst_case_distribution[1].probability-0.2).abs()<1e-9);
        assert!((a.transport_cost_used-0.2).abs()<1e-9);
        assert!((a.worst_case_expected_loss-8.0).abs()<1e-8);
        assert!(a.primal_dual_gap.abs()<1e-8);
        verify_wasserstein_robust_decision_report(&p,&set,&report).unwrap();
    }

    #[test]
    fn two_world_transport_matches_closed_form_over_probability_cost_and_radius_grid() {
        for p0 in [0.0_f64, 0.1, 0.5, 0.9, 1.0] {
            for distance in [0.25_f64, 1.0, 2.0, 5.0] {
                for radius in [0.0_f64, 0.05, 0.2, 0.8, 10.0] {
                    for (loss0, loss1) in [(0.0_f64, 10.0_f64), (10.0, 0.0), (4.0, 4.0)] {
                        let p1 = 1.0 - p0;
                        let transfer = (radius / distance).min(1.0);
                        let expected_q1 = if loss1 > loss0 {
                            (p1 + transfer).min(1.0)
                        } else if loss1 < loss0 {
                            (p1 - transfer).max(0.0)
                        } else {
                            p1
                        };
                        let p = [p0, p1];
                        let losses = [loss0, loss1];
                        let costs = vec![vec![0.0, distance], vec![distance, 0.0]];
                        let ids = vec!["world-a".to_owned(), "world-b".to_owned()];
                        let solution = solve_finite_transport(&p, &losses, &costs, radius, &ids).unwrap();
                        let expected_risk = (1.0 - expected_q1) * loss0 + expected_q1 * loss1;
                        let expected_cost = (expected_q1 - p1).abs() * distance;
                        assert!((solution.distribution[1] - expected_q1).abs() < 1e-8);
                        assert!((compensated_dot(&solution.distribution, &losses) - expected_risk).abs() < 1e-7);
                        assert!((solution.transport_cost - expected_cost).abs() < 1e-7);
                        assert!(solution.transport_cost <= radius + 1e-8);
                        assert!((solution.dual_upper_bound - expected_risk).abs() < 1e-7);
                    }
                }
            }
        }
    }

    #[test]
    fn tied_destinations_fill_transport_budget_using_highest_cost_option() {
        let nominal = [1.0_f64, 0.0, 0.0];
        let losses = [0.0_f64, 5.0, 10.0];
        let costs = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 1.0],
            vec![2.0, 1.0, 0.0],
        ];
        let ids = vec!["world-a".into(), "world-b".into(), "world-c".into()];
        let solution = solve_finite_transport(&nominal, &losses, &costs, 1.5, &ids).unwrap();
        assert!((solution.dual_multiplier.unwrap() - 5.0).abs() < 1e-12);
        assert!((solution.distribution[0] - 0.25).abs() < 1e-9);
        assert!(solution.distribution[1].abs() < 1e-9);
        assert!((solution.distribution[2] - 0.75).abs() < 1e-9);
        assert!((solution.transport_cost - 1.5).abs() < 1e-9);
        assert!((compensated_dot(&solution.distribution, &losses) - 7.5).abs() < 1e-8);
        assert!((solution.dual_upper_bound - 7.5).abs() < 1e-8);
    }

    #[test]
    fn zero_radius_reproduces_nominal_bayes_decision_and_risk() {
        let p=problem(); let report=analyze_wasserstein_robust_decision(&p,&ambiguity(1.0,0.0)).unwrap();
        assert_eq!(report.nominal_optimal_decision_id,report.robust_optimal_decision_id);
        assert!((report.robust_optimal_worst_case_risk-4.0).abs()<1e-12);
        let a=report.ranked_decisions.iter().find(|v|v.decision_id=="choose-a").unwrap();
        assert_eq!(a.dual_radius_multiplier,None); assert_eq!(a.transport_plan.len(),2);
    }

    #[test]
    fn large_radius_reaches_per_decision_global_worst_case() {
        let p=problem(); let report=analyze_wasserstein_robust_decision(&p,&ambiguity(1.0,10.0)).unwrap();
        let a=report.ranked_decisions.iter().find(|v|v.decision_id=="choose-a").unwrap();
        assert!((a.worst_case_expected_loss-40.0).abs()<1e-8);
        assert!((a.worst_case_distribution[1].probability-1.0).abs()<1e-9);
        assert!(a.transport_cost_used<=1.0+1e-9);
    }

    #[test]
    fn support_order_and_metric_identity_are_enforced() {
        let p=problem();
        let wrong=FiniteWassersteinAmbiguitySet::new("wrong",H4.into(),
            vec!["world-b".into(),"world-a".into()],vec![vec![0.0,1.0],vec![1.0,0.0]],0.2).unwrap();
        assert!(analyze_wasserstein_robust_decision(&p,&wrong).is_err());
        let mut asymmetric=ambiguity(1.0,0.2); asymmetric.ground_cost[0][1]=0.5;
        assert!(analyze_wasserstein_robust_decision(&p,&asymmetric).is_err());
    }

    #[test]
    fn non_metric_and_invalid_radius_fail_closed() {
        assert!(FiniteWassersteinAmbiguitySet::new("bad",H4.into(),
            vec!["a".into(),"b".into(),"c".into()],
            vec![vec![0.0,1.0,3.0],vec![1.0,0.0,1.0],vec![3.0,1.0,0.0]],1.0).is_err());
        assert!(FiniteWassersteinAmbiguitySet::new("bad",H4.into(),
            vec!["a".into(),"b".into()],vec![vec![0.0,1.0],vec![1.0,0.0]],f64::NAN).is_err());
    }

    #[test]
    fn report_mutation_is_detected_and_world_order_is_canonical() {
        let p=problem(); let set=ambiguity(1.0,0.2);
        let baseline=analyze_wasserstein_robust_decision(&p,&set).unwrap();
        let mut reordered=p.clone(); reordered.worlds.reverse(); reordered.decisions.reverse(); reordered.actions.reverse();
        for d in &mut reordered.decisions { d.losses_by_world.reverse(); }
        for a in &mut reordered.actions { a.outcomes.reverse(); for o in &mut a.outcomes { o.likelihoods_by_world.reverse(); } }
        assert_eq!(baseline,analyze_wasserstein_robust_decision(&reordered,&set).unwrap());
        let mut changed=baseline.clone(); changed.ranked_decisions[0].transport_plan[0].probability_mass+=0.01;
        assert!(verify_wasserstein_robust_decision_report(&p,&set,&changed).is_err());
    }
}
