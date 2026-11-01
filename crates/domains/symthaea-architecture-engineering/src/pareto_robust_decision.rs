//! Finite-state robust terminal decisions over a total-variation posterior ambiguity set.
//!
//! This transparent LP reference is conditional on the declared nominal posterior,
//! losses and ambiguity radius. It does not calibrate the radius or prove that the
//! data-generating distribution belongs to the ambiguity set.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{error::Error, fmt};
use super::pareto_evsi_oracle::{ExactEvsiError, ExactEvsiProblem, EvsiDecision};

const SCHEMA: &str = "symthaea.architecture-engineering.tv-robust-decision-report";
const VERSION: &str = "0.1.0";
const SET_SCHEMA: &str = "symthaea.architecture-engineering.total-variation-ambiguity-set";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RobustDecisionError { pub path: String, pub message: String }
impl RobustDecisionError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }
}
impl fmt::Display for RobustDecisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}: {}", self.path, self.message) }
}
impl Error for RobustDecisionError {}
impl From<ExactEvsiError> for RobustDecisionError {
    fn from(e: ExactEvsiError) -> Self { Self { path: e.path, message: e.message } }
}

/// The total-variation ball TV(q,p) = 0.5 * sum |q-p| <= radius.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TotalVariationAmbiguitySet {
    pub schema: String,
    pub version: String,
    pub model_id: String,
    pub model_sha256: String,
    pub radius: f64,
}
impl TotalVariationAmbiguitySet {
    pub fn new(model_id: impl Into<String>, model_sha256: impl Into<String>, radius: f64)
        -> Result<Self, RobustDecisionError> {
        let value = Self { schema: SET_SCHEMA.into(), version: VERSION.into(),
            model_id: model_id.into(), model_sha256: model_sha256.into(), radius };
        value.validate()?;
        Ok(value)
    }
    pub fn validate(&self) -> Result<(), RobustDecisionError> {
        if self.schema != SET_SCHEMA || self.version != VERSION {
            return Err(RobustDecisionError::new("ambiguity_set.schema", "unsupported schema/version"));
        }
        if self.model_id.trim().is_empty() {
            return Err(RobustDecisionError::new("ambiguity_set.model_id", "model ID must be non-empty"));
        }
        validate_sha(&self.model_sha256, "ambiguity_set.model_sha256")?;
        if !self.radius.is_finite() || !(0.0..=1.0).contains(&self.radius) {
            return Err(RobustDecisionError::new("ambiguity_set.radius", "radius must be finite and in [0, 1]"));
        }
        Ok(())
    }
    pub fn sha256(&self) -> Result<String, RobustDecisionError> {
        self.validate()?;
        digest(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustWorldProbability { pub world_id: String, pub probability: f64 }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustDecisionValue {
    pub decision_id: String,
    pub nominal_expected_loss: f64,
    pub worst_case_expected_loss: f64,
    pub total_variation_used: f64,
    pub worst_case_posterior: Vec<RobustWorldProbability>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobustDecisionReport {
    pub schema: String,
    pub version: String,
    pub campaign_id: String,
    pub problem_sha256: String,
    pub ambiguity_set_sha256: String,
    pub nominal_optimal_decision_id: String,
    pub robust_optimal_decision_id: String,
    pub nominal_risk_of_robust_decision: f64,
    pub robust_optimal_worst_case_risk: f64,
    pub ranked_decisions: Vec<RobustDecisionValue>,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct DigestPayload<'a> {
    schema: &'a str, version: &'a str, campaign_id: &'a str,
    problem_sha256: &'a str, ambiguity_set_sha256: &'a str,
    nominal_optimal_decision_id: &'a str, robust_optimal_decision_id: &'a str,
    nominal_risk_of_robust_decision: f64, robust_optimal_worst_case_risk: f64,
    ranked_decisions: &'a [RobustDecisionValue],
}

/// For each terminal decision, maximize expected loss over a TV ball, then minimize
/// that worst-case risk. The mass-transfer solution is the exact finite LP solution
/// up to floating-point roundoff.
pub fn analyze_tv_robust_decision(
    problem: &ExactEvsiProblem,
    ambiguity: &TotalVariationAmbiguitySet,
) -> Result<RobustDecisionReport, RobustDecisionError> {
    let problem = problem.clone().canonicalize()?;
    ambiguity.validate()?;
    let nominal: Vec<f64> = problem.worlds.iter().map(|w| w.posterior_probability).collect();
    let nominal_best = nominal_decision(&problem.decisions, &nominal)?;
    let mut values = Vec::with_capacity(problem.decisions.len());

    for decision in &problem.decisions {
        let losses: Vec<f64> = decision.losses_by_world.iter().map(|r| r.loss).collect();
        let nominal_risk = dot(&nominal, &losses);
        let (worst, moved) = maximize_tv(&nominal, &losses, ambiguity.radius);
        let worst_risk = dot(&worst, &losses);
        if !nominal_risk.is_finite() || !worst_risk.is_finite() || worst_risk + 1e-9 < nominal_risk {
            return Err(RobustDecisionError::new(
                format!("decisions.{}.risk", decision.decision_id),
                "non-finite risk or worst-case risk below nominal risk",
            ));
        }
        values.push(RobustDecisionValue {
            decision_id: decision.decision_id.clone(),
            nominal_expected_loss: nominal_risk,
            worst_case_expected_loss: worst_risk,
            total_variation_used: moved,
            worst_case_posterior: problem.worlds.iter().zip(&worst)
                .map(|(w, p)| RobustWorldProbability { world_id: w.world_id.clone(), probability: *p })
                .collect(),
        });
    }
    values.sort_by(|a,b| a.worst_case_expected_loss.total_cmp(&b.worst_case_expected_loss)
        .then_with(|| a.decision_id.cmp(&b.decision_id)));
    let best = values.first().ok_or_else(|| RobustDecisionError::new("decisions", "no decisions"))?;
    let nominal_risk = best.nominal_expected_loss;
    let robust_id = best.decision_id.clone();
    let worst_risk = best.worst_case_expected_loss;
    let problem_sha = problem.problem_sha256()?;
    let ambiguity_sha = ambiguity.sha256()?;
    let payload = DigestPayload { schema: SCHEMA, version: VERSION, campaign_id: &problem.campaign_id,
        problem_sha256: &problem_sha, ambiguity_set_sha256: &ambiguity_sha,
        nominal_optimal_decision_id: &nominal_best, robust_optimal_decision_id: &robust_id,
        nominal_risk_of_robust_decision: nominal_risk,
        robust_optimal_worst_case_risk: worst_risk, ranked_decisions: &values };
    let report_sha = digest(&payload)?;
    Ok(RobustDecisionReport { schema: SCHEMA.into(), version: VERSION.into(),
        campaign_id: problem.campaign_id, problem_sha256: problem_sha,
        ambiguity_set_sha256: ambiguity_sha, nominal_optimal_decision_id: nominal_best,
        robust_optimal_decision_id: robust_id,
        nominal_risk_of_robust_decision: nominal_risk,
        robust_optimal_worst_case_risk: worst_risk, ranked_decisions: values,
        report_sha256: report_sha })
}

pub fn verify_tv_robust_decision_report(
    problem: &ExactEvsiProblem, ambiguity: &TotalVariationAmbiguitySet,
    report: &RobustDecisionReport,
) -> Result<(), RobustDecisionError> {
    if analyze_tv_robust_decision(problem, ambiguity)? != *report {
        return Err(RobustDecisionError::new("report", "report differs from independent reconstruction"));
    }
    Ok(())
}

fn nominal_decision(decisions: &[EvsiDecision], p: &[f64]) -> Result<String, RobustDecisionError> {
    let mut best: Option<(f64, String)> = None;
    for d in decisions {
        let loss: Vec<f64> = d.losses_by_world.iter().map(|r| r.loss).collect();
        let risk = dot(p, &loss);
        match &best {
            None => best = Some((risk, d.decision_id.clone())),
            Some((v,id)) if risk < *v || (risk == *v && d.decision_id < *id) =>
                best = Some((risk, d.decision_id.clone())),
            _ => {}
        }
    }
    best.map(|(_,id)| id).ok_or_else(|| RobustDecisionError::new("decisions", "no terminal decisions"))
}

/// Move probability mass from lowest-loss worlds to highest-loss worlds. Each transfer
/// consumes one unit of TV radius; a transfer stops when a donor empties, a receiver
/// fills, the radius is exhausted, or no strictly loss-improving transfer remains.
fn maximize_tv(p: &[f64], loss: &[f64], radius: f64) -> (Vec<f64>, f64) {
    let mut q = p.to_vec();
    let mut donors: Vec<usize> = (0..p.len()).collect();
    donors.sort_by(|a,b| loss[*a].total_cmp(&loss[*b]).then_with(|| a.cmp(b)));
    let mut receivers: Vec<usize> = (0..p.len()).collect();
    receivers.sort_by(|a,b| loss[*b].total_cmp(&loss[*a]).then_with(|| a.cmp(b)));
    let (mut di, mut ri, mut remaining) = (0usize, 0usize, radius);
    while remaining > 0.0 {
        while di < donors.len() && q[donors[di]] <= 0.0 { di += 1; }
        while ri < receivers.len() && q[receivers[ri]] >= 1.0 { ri += 1; }
        if di == donors.len() || ri == receivers.len() { break; }
        let (d,r) = (donors[di], receivers[ri]);
        if d == r || loss[r] <= loss[d] { break; }
        let shift = q[d].min(1.0 - q[r]).min(remaining);
        if !shift.is_finite() || shift <= 0.0 { break; }
        q[d] -= shift;
        q[r] += shift;
        remaining = (remaining - shift).max(0.0);
        if q[d] <= 0.0 { di += 1; }
        if q[r] >= 1.0 { ri += 1; }
    }
    (q, (radius - remaining).max(0.0))
}

fn dot(p: &[f64], loss: &[f64]) -> f64 {
    let (mut sum, mut correction) = (0.0, 0.0);
    for (probability, value) in p.iter().zip(loss) {
        let adjusted = probability * value - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    sum
}
fn validate_sha(value: &str, path: &str) -> Result<(), RobustDecisionError> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err(RobustDecisionError::new(path, "expected lowercase SHA-256"));
    }
    Ok(())
}
fn digest<T: Serialize>(value: &T) -> Result<String, RobustDecisionError> {
    let bytes = serde_json::to_vec(value).map_err(|e| RobustDecisionError::new("serialization", e.to_string()))?;
    Ok(Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pareto_evsi_oracle::{
        EvsiActionOutcome, EvsiDecision, EvsiReplicateAction, EvsiWorldLikelihood,
        EvsiWorldLoss, EvsiWorldProbability,
    };
    const H1: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const H2: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const H3: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const H4: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn model() -> ExactEvsiProblem {
        ExactEvsiProblem {
            schema: "symthaea.architecture-engineering.exact-discrete-evsi-problem".into(),
            version: "0.1.0".into(), campaign_id: "tv-robust-test".into(),
            posterior_model_id: "posterior".into(), posterior_model_sha256: H1.into(),
            observation_model_id: "observation".into(), observation_model_sha256: H2.into(),
            decision_loss_model_id: "loss".into(), decision_loss_model_sha256: H3.into(),
            cost_model_id: "cost".into(), cost_model_sha256: H4.into(),
            worlds: vec![
                EvsiWorldProbability { world_id: "world-a".into(), posterior_probability: 0.9 },
                EvsiWorldProbability { world_id: "world-b".into(), posterior_probability: 0.1 }],
            decisions: vec![
                EvsiDecision { decision_id: "choose-a".into(), losses_by_world: vec![
                    EvsiWorldLoss { world_id: "world-a".into(), loss: 0.0 },
                    EvsiWorldLoss { world_id: "world-b".into(), loss: 40.0 }]},
                EvsiDecision { decision_id: "choose-b".into(), losses_by_world: vec![
                    EvsiWorldLoss { world_id: "world-a".into(), loss: 10.0 },
                    EvsiWorldLoss { world_id: "world-b".into(), loss: 0.0 }]},
                EvsiDecision { decision_id: "abstain".into(), losses_by_world: vec![
                    EvsiWorldLoss { world_id: "world-a".into(), loss: 5.0 },
                    EvsiWorldLoss { world_id: "world-b".into(), loss: 5.0 }]}],
            actions: vec![EvsiReplicateAction { action_id: "measure".into(), candidate_id: "a".into(),
                objective_id: "quality".into(), incremental_cost_units: 1, outcomes: vec![
                    EvsiActionOutcome { outcome_id: "a".into(), likelihoods_by_world: vec![
                        EvsiWorldLikelihood { world_id: "world-a".into(), probability: 1.0 },
                        EvsiWorldLikelihood { world_id: "world-b".into(), probability: 0.0 }]},
                    EvsiActionOutcome { outcome_id: "b".into(), likelihoods_by_world: vec![
                        EvsiWorldLikelihood { world_id: "world-a".into(), probability: 0.0 },
                        EvsiWorldLikelihood { world_id: "world-b".into(), probability: 1.0 }]}]}],
        }
    }
    fn set(radius: f64) -> TotalVariationAmbiguitySet {
        TotalVariationAmbiguitySet::new("tv-ball-v1", H4, radius).unwrap()
    }

    #[test]
    fn robust_choice_can_differ_from_nominal_choice() {
        let p = model();
        let r = analyze_tv_robust_decision(&p, &set(0.2)).unwrap();
        assert_eq!(r.nominal_optimal_decision_id, "choose-a");
        assert_eq!(r.robust_optimal_decision_id, "abstain");
        assert!((r.robust_optimal_worst_case_risk - 5.0).abs() < 1e-12);
        let a = r.ranked_decisions.iter().find(|v| v.decision_id == "choose-a").unwrap();
        assert!((a.worst_case_expected_loss - 12.0).abs() < 1e-12);
        assert_eq!(a.worst_case_posterior[0].probability, 0.7);
        assert_eq!(a.worst_case_posterior[1].probability, 0.3);
        verify_tv_robust_decision_report(&p, &set(0.2), &r).unwrap();
    }

    #[test]
    fn zero_radius_reproduces_nominal_decision() {
        let r = analyze_tv_robust_decision(&model(), &set(0.0)).unwrap();
        assert_eq!(r.nominal_optimal_decision_id, r.robust_optimal_decision_id);
        assert!((r.robust_optimal_worst_case_risk - 4.0).abs() < 1e-12);
    }

    #[test]
    fn radius_and_ambiguity_digest_change_report_identity() {
        let p = model();
        let a = analyze_tv_robust_decision(&p, &set(0.1)).unwrap();
        let b = analyze_tv_robust_decision(&p, &set(0.2)).unwrap();
        assert_ne!(a.ambiguity_set_sha256, b.ambiguity_set_sha256);
        assert_ne!(a.report_sha256, b.report_sha256);
        let mut changed = a.clone();
        changed.robust_optimal_decision_id = "choose-a".into();
        assert!(verify_tv_robust_decision_report(&p, &set(0.1), &changed).is_err());
    }

    #[test]
    fn invalid_ambiguity_contract_fails_closed() {
        assert!(TotalVariationAmbiguitySet::new("tv", H4, f64::NAN).is_err());
        assert!(TotalVariationAmbiguitySet::new("", H4, 0.2).is_err());
        assert!(TotalVariationAmbiguitySet::new("tv", "bad", 0.2).is_err());
        assert!(TotalVariationAmbiguitySet::new("tv", H4, 1.1).is_err());
    }

    #[test]
    fn two_world_solver_matches_independent_closed_form_grid() {
        for p0 in [0.0_f64, 0.1, 0.5, 0.9, 1.0] {
            for radius in [0.0_f64, 0.05, 0.2, 0.5, 1.0] {
                for (loss0, loss1) in [(0.0_f64, 10.0_f64), (10.0, 0.0), (4.0, 4.0), (2.0, 7.0)] {
                    let p = [p0, 1.0 - p0];
                    let loss = [loss0, loss1];
                    let (q, used) = maximize_tv(&p, &loss, radius);
                    let low = (p0 - radius).max(0.0);
                    let high = (p0 + radius).min(1.0);
                    let expected_q0 = if loss0 > loss1 { high }
                        else if loss0 < loss1 { low } else { p0 };
                    let expected_risk = expected_q0 * loss0 + (1.0 - expected_q0) * loss1;
                    assert!((q[0] - expected_q0).abs() < 1e-12);
                    assert!((dot(&q, &loss) - expected_risk).abs() < 1e-10);
                    assert!((used - (expected_q0 - p0).abs()).abs() < 1e-12);
                    assert!((q[0] + q[1] - 1.0).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn canonical_input_order_reconstructs_identical_report() {
        let p = model();
        let mut reversed = p.clone();
        reversed.worlds.reverse();
        reversed.decisions.reverse();
        reversed.actions.reverse();
        for d in &mut reversed.decisions { d.losses_by_world.reverse(); }
        for a in &mut reversed.actions {
            a.outcomes.reverse();
            for o in &mut a.outcomes { o.likelihoods_by_world.reverse(); }
        }
        assert_eq!(analyze_tv_robust_decision(&p, &set(0.2)).unwrap(),
            analyze_tv_robust_decision(&reversed, &set(0.2)).unwrap());
    }
}
