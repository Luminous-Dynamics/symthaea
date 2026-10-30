//! Faithful scheduler-formula baseline for Coral Discovery (arXiv:2610.33281).
//!
//! This module reproduces the published direction-board equations as a distinct baseline.
//! It is not the full autonomous-agent protocol or independent semantic verifier. Callers must
//! provide verifier-assigned direction IDs, normalized maximization scores and matched eval events.
//! The scheduler is advisory; wanted headcounts do not force an agent to change direction.
//!
//! Formula defaults from the paper: W=32, prior_n=4, k=3, kappa=0.05. Each evaluation is one
//! equal-cost scheduler step. evaluation_cost_units are retained in the trace but do not enter
//! Coral's yield equation; heterogeneous-cost scheduling is a separate comparison.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const SCHEMA: &str = "symthaea.architecture-engineering.coral-direction-board";
const VERSION: &str = "0.1.0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoralBaselineError {
    pub path: String,
    pub message: String,
}
impl CoralBaselineError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }
}
impl std::fmt::Display for CoralBaselineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}
impl std::error::Error for CoralBaselineError {}

/// Published formula parameters. Values are fixed before the comparison campaign.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoralBaselineConfig {
    pub window_size: usize,
    pub prior_n: usize,
    pub infancy_k: usize,
    pub kappa: f64,
    pub agent_count: usize,
    pub max_agents_per_direction: usize,
    pub first_round_agents_per_direction: usize,
}
impl Default for CoralBaselineConfig {
    fn default() -> Self {
        Self {
            window_size: 32,
            prior_n: 4,
            infancy_k: 3,
            kappa: 0.05,
            agent_count: 4,
            max_agents_per_direction: 2,
            first_round_agents_per_direction: 1,
        }
    }
}
impl CoralBaselineConfig {
    pub fn validate(&self) -> Result<(), CoralBaselineError> {
        if self.window_size == 0 || self.prior_n == 0 || self.infancy_k == 0 || self.agent_count == 0 {
            return Err(CoralBaselineError::new("config", "window, priors, infancy and agent counts must be positive"));
        }
        if !self.kappa.is_finite() || self.kappa < 0.0 {
            return Err(CoralBaselineError::new("config.kappa", "must be finite and non-negative"));
        }
        if self.max_agents_per_direction == 0
            || self.first_round_agents_per_direction == 0
            || self.first_round_agents_per_direction > self.max_agents_per_direction {
            return Err(CoralBaselineError::new("config.agent_caps", "invalid per-direction claim caps"));
        }
        Ok(())
    }
}

/// One verifier-labelled evaluation. Failed evaluations have score=None and credit zero.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoralEvaluation {
    pub attempt_id: String,
    pub direction_id: String,
    /// Normalized utility; higher is always better. Negate minimization objectives before input.
    pub score: Option<f64>,
    pub parent_score: Option<f64>,
    pub inspired_by_scores: Vec<f64>,
    pub evaluation_cost_units: u64,
    /// Derived from the paper's local-record credit rule by the board, not supplied by solver.
    pub credit: f64,
    pub global_best_after: Option<f64>,
}

/// Per-direction running state. The opening global best is fixed when the verifier admits it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoralDirectionState {
    pub direction_id: String,
    pub global_best_at_open: Option<f64>,
    pub evaluations: Vec<CoralEvaluation>,
}

/// Direction-local quantities used by the published scheduler.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoralDirectionMetrics {
    pub direction_id: String,
    pub window_evaluations: usize,
    pub lifetime_evaluations: usize,
    pub window_credit: f64,
    pub shrunk_yield: f64,
    pub exploration_bonus: f64,
    pub potential: f64,
    pub infancy_evaluations: usize,
    pub infancy_gain: f64,
}

/// NEW is an option to open another research direction, not an existing direction identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "direction_id", rename_all = "snake_case")]
pub enum CoralArm {
    Direction(String),
    New,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoralAllocation {
    pub arm: CoralArm,
    pub wanted_agents: usize,
    pub estimated_yield: f64,
    pub exploration_bonus: f64,
    pub potential: f64,
}

/// Advisory direction board with windowed yield and the published NEW-arm estimator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoralDirectionBoard {
    pub schema: String,
    pub version: String,
    pub config: CoralBaselineConfig,
    pub directions: BTreeMap<String, CoralDirectionState>,
    pub evaluations: Vec<CoralEvaluation>,
    pub attempt_ids: BTreeSet<String>,
    pub global_best_score: Option<f64>,
}

impl CoralDirectionBoard {
    pub fn new(config: CoralBaselineConfig) -> Result<Self, CoralBaselineError> {
        config.validate()?;
        Ok(Self {
            schema: SCHEMA.to_owned(),
            version: VERSION.to_owned(),
            config,
            directions: BTreeMap::new(),
            evaluations: Vec::new(),
            attempt_ids: BTreeSet::new(),
            global_best_score: None,
        })
    }

    /// Register a verifier-assigned direction before evaluating a candidate in that direction.
    pub fn register_direction(&mut self, direction_id: &str) -> Result<(), CoralBaselineError> {
        self.validate_integrity()?;
        let id = canonical_id("direction_id", direction_id)?;
        if self.directions.contains_key(&id) {
            return Err(CoralBaselineError::new("direction_id", "duplicate direction identity"));
        }
        self.directions.insert(id.clone(), CoralDirectionState {
            direction_id: id,
            global_best_at_open: self.global_best_score,
            evaluations: Vec::new(),
        });
        Ok(())
    }

    /// Append one scored or failed eval. Score and parent/inspired scores use maximization utility.
    pub fn record_evaluation(
        &mut self,
        attempt_id: &str,
        direction_id: &str,
        score: Option<f64>,
        parent_score: Option<f64>,
        inspired_by_scores: Vec<f64>,
        evaluation_cost_units: u64,
    ) -> Result<f64, CoralBaselineError> {
        self.validate_integrity()?;
        let attempt_id = canonical_id("attempt_id", attempt_id)?;
        let direction_id = canonical_id("direction_id", direction_id)?;
        if self.attempt_ids.contains(&attempt_id) {
            return Err(CoralBaselineError::new("attempt_id", "duplicate attempt ID"));
        }
        if evaluation_cost_units == 0 {
            return Err(CoralBaselineError::new("evaluation_cost_units", "must be positive"));
        }
        for (field, value) in [("score", score), ("parent_score", parent_score)] {
            if value.is_some_and(|v| !v.is_finite()) {
                return Err(CoralBaselineError::new(field, "must be finite when present"));
            }
        }
        if inspired_by_scores.iter().any(|v| !v.is_finite()) {
            return Err(CoralBaselineError::new("inspired_by_scores", "all values must be finite"));
        }
        let state = self.directions.get(&direction_id)
            .ok_or_else(|| CoralBaselineError::new("direction_id", "must be registered by the verifier before evaluation"))?;
        let direction_best = state.evaluations.iter().filter_map(|event| event.score)
            .max_by(f64::total_cmp);
        let score_threshold = parent_score.into_iter()
            .chain(inspired_by_scores.iter().copied())
            .chain(direction_best)
            .max_by(f64::total_cmp);
        let global_best_after = match (self.global_best_score, score) {
            (Some(old), Some(new)) => Some(old.max(new)),
            (Some(old), None) => Some(old),
            (None, Some(new)) => Some(new),
            (None, None) => None,
        };

        let credit = match (score, score_threshold, global_best_after) {
            (Some(value), Some(threshold), Some(global_best)) if global_best.abs() > f64::EPSILON => {
                ((value - threshold).max(0.0) / global_best.abs()).max(0.0)
            }
            _ => 0.0,
        };
        if !credit.is_finite() {
            return Err(CoralBaselineError::new("credit", "derived credit is non-finite"));
        }

        let event = CoralEvaluation {
            attempt_id: attempt_id.clone(),
            direction_id: direction_id.clone(),
            score,
            parent_score,
            inspired_by_scores,
            evaluation_cost_units,
            credit,
            global_best_after,
        };
        self.directions.get_mut(&direction_id).expect("registered direction checked").evaluations.push(event.clone());
        self.evaluations.push(event);
        self.attempt_ids.insert(attempt_id);
        self.global_best_score = global_best_after;
        Ok(credit)
    }

    /// Windowed shrunk yield and lifetime-count exploration potential for each direction.
    pub fn direction_metrics(&self) -> Result<Vec<CoralDirectionMetrics>, CoralBaselineError> {
        self.validate_integrity()?;
        let window_start = self.evaluations.len().saturating_sub(self.config.window_size);
        let window = &self.evaluations[window_start..];
        let n = window.len();
        let mean_credit = if n == 0 { 0.0 } else {
            window.iter().map(|event| event.credit).sum::<f64>() / n as f64
        };
        let mut partial = Vec::with_capacity(self.directions.len());
        for state in self.directions.values() {
            let recent: Vec<&CoralEvaluation> = window.iter()
                .filter(|event| event.direction_id == state.direction_id).collect();
            let window_credit = recent.iter().map(|event| event.credit).sum::<f64>();
            let recent_count = recent.len();
            let lifetime_count = state.evaluations.len();
            let shrunk_yield = (window_credit + self.config.prior_n as f64 * mean_credit)
                / (recent_count + self.config.prior_n) as f64;
            let infancy: Vec<&CoralEvaluation> = state.evaluations.iter()
                .take(self.config.infancy_k).collect();
            let infancy_count = infancy.len();
            let infancy_gain = match (state.global_best_at_open, infancy.iter().filter_map(|event| event.score).max_by(f64::total_cmp)) {
                (Some(open_best), Some(best)) => (best - open_best).max(0.0),
                _ => 0.0,
            };
            partial.push((state.direction_id.clone(), recent_count, lifetime_count, window_credit,
                shrunk_yield, infancy_count, infancy_gain));
        }
        let y_plus = partial.iter().map(|item| item.4).fold(0.0, f64::max);
        let c = self.config.kappa * y_plus;
        let log_n = (n.max(1) as f64).ln();
        Ok(partial.into_iter().map(|(direction_id, recent_count, lifetime_count, window_credit,
            shrunk_yield, infancy_count, infancy_gain)| {
            let exploration_bonus = c * (log_n / (lifetime_count as f64 + 1.0)).sqrt();
            CoralDirectionMetrics {
                direction_id,
                window_evaluations: recent_count,
                lifetime_evaluations: lifetime_count,
                window_credit,
                shrunk_yield,
                exploration_bonus,
                potential: shrunk_yield + exploration_bonus,
                infancy_evaluations: infancy_count,
                infancy_gain,
            }
        }).collect())
    }

    /// Paper's NEW estimator: first-k improvement after opening, shrunk to k pseudo-evals at y+.
    pub fn new_direction_potential(&self) -> Result<(f64, f64, f64), CoralBaselineError> {
        self.validate_integrity()?;
        let metrics = self.direction_metrics()?;
        let y_plus = metrics.iter().map(|metric| metric.shrunk_yield).fold(0.0, f64::max);
        let mut total_gain = 0.0;
        let mut total_infancy_evals = 0usize;
        for state in self.directions.values() {
            if state.global_best_at_open.is_none() {
                continue;
            }
            let n0 = state.evaluations.len().min(self.config.infancy_k);
            total_infancy_evals += n0;
            let best_infancy = state.evaluations.iter().take(self.config.infancy_k)
                .filter_map(|event| event.score).max_by(f64::total_cmp);
            if let (Some(open_best), Some(best)) = (state.global_best_at_open, best_infancy) {
                total_gain += (best - open_best).max(0.0);
            }
        }
        let k = self.config.infancy_k;
        let estimate = (total_gain + k as f64 * y_plus) / (total_infancy_evals + k) as f64;
        let n = self.evaluations.len().min(self.config.window_size).max(1);
        let c = self.config.kappa * y_plus;
        let bonus = c * ((n as f64).ln() / (total_infancy_evals as f64 + 1.0)).sqrt();
        Ok((estimate, bonus, estimate + bonus))
    }

    /// Allocate agent_count advisory seats with largest-remainder rounding.
    pub fn recommend(&self) -> Result<Vec<CoralAllocation>, CoralBaselineError> {
        self.validate_integrity()?;
        let metrics = self.direction_metrics()?;
        let (new_yield, new_bonus, new_potential) = self.new_direction_potential()?;
        let mut arms: Vec<(CoralArm, f64, f64, f64)> = metrics.into_iter()
            .map(|metric| (CoralArm::Direction(metric.direction_id), metric.shrunk_yield,
                metric.exploration_bonus, metric.potential)).collect();
        arms.push((CoralArm::New, new_yield, new_bonus, new_potential));
        let potential_sum = arms.iter().map(|arm| arm.3).sum::<f64>();
        let mut allocations = vec![0usize; arms.len()];
        if !potential_sum.is_finite() || potential_sum < 0.0 {
            return Err(CoralBaselineError::new("potential", "sum must be finite and non-negative"));
        }
        if potential_sum <= f64::EPSILON {
            let best_direction = self.global_best_direction();
            if let Some(best_id) = best_direction {
                let index = arms.iter().position(|arm| arm.0 == CoralArm::Direction(best_id.clone()))
                    .ok_or_else(|| CoralBaselineError::new("allocation", "global-best direction is absent"))?;
                allocations[index] = self.config.agent_count;
            } else {
                let new_index = arms.len() - 1;
                allocations[new_index] = self.config.agent_count;
            }
        } else {
            let mut fractions = Vec::with_capacity(arms.len());
            let mut assigned = 0usize;
            for (index, arm) in arms.iter().enumerate() {
                let exact = self.config.agent_count as f64 * arm.3 / potential_sum;
                let whole = exact.floor() as usize;
                allocations[index] = whole;
                assigned += whole;
                fractions.push((index, exact - whole as f64));
            }
            fractions.sort_by(|(ia, fa), (ib, fb)| {
                fb.total_cmp(fa).then_with(|| arm_key(&arms[*ia].0).cmp(arm_key(&arms[*ib].0)))
            });
            for (index, _) in fractions.into_iter().take(self.config.agent_count - assigned) {
                allocations[index] += 1;
            }
        }
        Ok(arms.into_iter().zip(allocations).map(|((arm, estimated_yield, exploration_bonus, potential), wanted_agents)| {
            CoralAllocation { arm, wanted_agents, estimated_yield, exploration_bonus, potential }
        }).collect())
    }

    /// Enforce the published claim cap. The board's wanted allocation remains advisory.
    pub fn validate_claim_capacity(
        &self,
        direction_id: &str,
        currently_claimed_agents: usize,
        first_round: bool,
    ) -> Result<(), CoralBaselineError> {
        let direction_id = canonical_id("direction_id", direction_id)?;
        if !self.directions.contains_key(&direction_id) {
            return Err(CoralBaselineError::new("direction_id", "unknown verifier-assigned direction"));
        }
        let cap = if first_round {
            self.config.first_round_agents_per_direction
        } else {
            self.config.max_agents_per_direction
        };
        if currently_claimed_agents >= cap {
            return Err(CoralBaselineError::new("claims", "per-direction agent cap reached"));
        }
        Ok(())
    }

    fn global_best_direction(&self) -> Option<String> {
        self.directions.values().filter_map(|state| {
            state.evaluations.iter().filter_map(|event| event.score).max_by(f64::total_cmp)
                .map(|score| (state.direction_id.clone(), score))
        }).max_by(|(id_a, score_a), (id_b, score_b)| {
            score_a.total_cmp(score_b).then_with(|| id_b.cmp(id_a))
        }).map(|(id, _)| id)
    }

    pub fn validate_integrity(&self) -> Result<(), CoralBaselineError> {
        if self.schema != SCHEMA || self.version != VERSION {
            return Err(CoralBaselineError::new("schema", "unsupported baseline schema or version"));
        }
        self.config.validate()?;
        let mut expected_ids = BTreeSet::new();
        let mut expected_by_direction: BTreeMap<String, Vec<CoralEvaluation>> = BTreeMap::new();
        let mut reconstructed_best: Option<f64> = None;

        for (index, event) in self.evaluations.iter().enumerate() {
            if canonical_id("attempt_id", &event.attempt_id)? != event.attempt_id
                || canonical_id("direction_id", &event.direction_id)? != event.direction_id
            {
                return Err(CoralBaselineError::new(
                    format!("evaluations[{index}]"),
                    "attempt and direction IDs must be canonical",
                ));
            }
            if !expected_ids.insert(event.attempt_id.clone()) {
                return Err(CoralBaselineError::new(
                    format!("evaluations[{index}].attempt_id"),
                    "duplicate attempt ID",
                ));
            }
            if event.evaluation_cost_units == 0
                || event.score.is_some_and(|v| !v.is_finite())
                || event.parent_score.is_some_and(|v| !v.is_finite())
                || event.inspired_by_scores.iter().any(|v| !v.is_finite())
            {
                return Err(CoralBaselineError::new(
                    format!("evaluations[{index}]"),
                    "score inputs and cost are invalid",
                ));
            }
            if !self.directions.contains_key(&event.direction_id) {
                return Err(CoralBaselineError::new(
                    format!("evaluations[{index}].direction_id"),
                    "unregistered direction",
                ));
            }

            let local_best = expected_by_direction
                .get(&event.direction_id)
                .into_iter()
                .flatten()
                .filter_map(|prior| prior.score)
                .max_by(f64::total_cmp);
            let threshold = event.parent_score
                .into_iter()
                .chain(event.inspired_by_scores.iter().copied())
                .chain(local_best)
                .max_by(f64::total_cmp);
            let expected_best = match (reconstructed_best, event.score) {
                (Some(old), Some(new)) => Some(old.max(new)),
                (Some(old), None) => Some(old),
                (None, Some(new)) => Some(new),
                (None, None) => None,
            };
            let expected_credit = match (event.score, threshold, expected_best) {
                (Some(score), Some(threshold), Some(best)) if best.abs() > f64::EPSILON => {
                    (score - threshold).max(0.0) / best.abs()
                }
                _ => 0.0,
            };
            if !event.credit.is_finite()
                || (event.credit - expected_credit).abs() > 1e-12
                || !same_optional_score(event.global_best_after, expected_best)
            {
                return Err(CoralBaselineError::new(
                    format!("evaluations[{index}]"),
                    "stored credit or global-best value does not reconcile with ordered score inputs",
                ));
            }
            reconstructed_best = expected_best;
            expected_by_direction
                .entry(event.direction_id.clone())
                .or_default()
                .push(event.clone());
        }

        if expected_ids != self.attempt_ids {
            return Err(CoralBaselineError::new(
                "attempt_ids",
                "attempt ID index does not exactly match the ordered evaluation stream",
            ));
        }
        if !same_optional_score(reconstructed_best, self.global_best_score) {
            return Err(CoralBaselineError::new(
                "global_best_score",
                "does not reconcile with the ordered evaluation stream",
            ));
        }
        for (map_id, state) in &self.directions {
            if map_id != &state.direction_id || canonical_id("direction_id", map_id)? != *map_id {
                return Err(CoralBaselineError::new(
                    "directions",
                    "map key and canonical direction ID disagree",
                ));
            }
            let expected = expected_by_direction.get(map_id).cloned().unwrap_or_default();
            if state.evaluations != expected {
                return Err(CoralBaselineError::new(
                    format!("directions[{map_id}].evaluations"),
                    "direction history does not reconcile with ordered global evaluation stream",
                ));
            }
        }
        Ok(())
    }
}

fn canonical_id(field: &str, value: &str) -> Result<String, CoralBaselineError> {
    let id = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if id.is_empty() { Err(CoralBaselineError::new(field, "must be non-empty")) } else { Ok(id) }
}
fn same_optional_score(left: Option<f64>, right: Option<f64>) -> bool {
    match (left, right) {
        (Some(a), Some(b)) => (a - b).abs() <= 1e-12,
        (None, None) => true,
        _ => false,
    }
}
fn arm_key(arm: &CoralArm) -> &str {
    match arm {
        CoralArm::Direction(id) => id.as_str(),
        CoralArm::New => "~NEW",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board() -> CoralDirectionBoard {
        CoralDirectionBoard::new(CoralBaselineConfig::default()).unwrap()
    }

    #[test]
    fn defaults_match_the_published_direction_board_parameters() {
        let config = CoralBaselineConfig::default();
        assert_eq!(config.window_size, 32);
        assert_eq!(config.prior_n, 4);
        assert_eq!(config.infancy_k, 3);
        assert!((config.kappa - 0.05).abs() < f64::EPSILON);
        assert_eq!(config.agent_count, 4);
        assert_eq!(config.max_agents_per_direction, 2);
        assert_eq!(config.first_round_agents_per_direction, 1);
    }

    #[test]
    fn local_record_credit_is_normalized_by_global_best() {
        let mut b = board();
        b.register_direction("A").unwrap();
        let first = b.record_evaluation("a1", "A", Some(100.0), Some(90.0), vec![], 1).unwrap();
        assert!((first - 0.10).abs() < 1e-12);
        let second = b.record_evaluation("a2", "A", Some(95.0), Some(90.0), vec![], 1).unwrap();
        assert!((second - 0.0).abs() < 1e-12);
    }

    #[test]
    fn parent_inspired_and_direction_records_raise_credit_threshold() {
        let mut b = board();
        b.register_direction("A").unwrap();
        b.record_evaluation("a1", "A", Some(100.0), Some(90.0), vec![], 1).unwrap();
        b.register_direction("B").unwrap();
        let credit = b.record_evaluation("b1", "B", Some(110.0), Some(105.0), vec![108.0], 1).unwrap();
        assert!((credit - (2.0 / 110.0)).abs() < 1e-12);
    }

    #[test]
    fn failed_evaluations_count_in_window_and_have_zero_credit() {
        let mut b = board();
        b.register_direction("A").unwrap();
        b.record_evaluation("a1", "A", Some(100.0), Some(90.0), vec![], 1).unwrap();
        b.record_evaluation("a2", "A", None, None, vec![], 1).unwrap();
        let metrics = b.direction_metrics().unwrap();
        assert_eq!(metrics[0].window_evaluations, 2);
        assert!((metrics[0].window_credit - 0.1).abs() < 1e-12);
        assert!(metrics[0].shrunk_yield < 0.1);
    }

    #[test]
    fn sliding_window_drops_old_credit_but_lifetime_count_remains() {
        let mut config = CoralBaselineConfig::default();
        config.window_size = 2;
        config.prior_n = 1;
        let mut b = CoralDirectionBoard::new(config).unwrap();
        b.register_direction("A").unwrap();
        b.record_evaluation("a1", "A", Some(100.0), Some(90.0), vec![], 1).unwrap();
        b.record_evaluation("a2", "A", Some(95.0), Some(90.0), vec![], 1).unwrap();
        b.record_evaluation("a3", "A", Some(94.0), Some(90.0), vec![], 1).unwrap();
        let metric = &b.direction_metrics().unwrap()[0];
        assert_eq!(metric.window_evaluations, 2);
        assert_eq!(metric.lifetime_evaluations, 3);
        assert_eq!(metric.window_credit, 0.0);
    }

    #[test]
    fn new_arm_uses_first_k_infancy_improvements_after_opening() {
        let mut b = board();
        b.register_direction("A").unwrap();
        b.record_evaluation("a1", "A", Some(100.0), Some(90.0), vec![], 1).unwrap();
        b.register_direction("B").unwrap();
        b.record_evaluation("b1", "B", Some(99.0), Some(95.0), vec![], 1).unwrap();
        b.record_evaluation("b2", "B", Some(101.0), Some(95.0), vec![], 1).unwrap();
        b.record_evaluation("b3", "B", Some(102.0), Some(95.0), vec![], 1).unwrap();
        let metrics = b.direction_metrics().unwrap();
        let metric_b = metrics.iter().find(|m| m.direction_id == "B").unwrap();
        assert_eq!(metric_b.infancy_evaluations, 3);
        assert!((metric_b.infancy_gain - 2.0).abs() < 1e-12);
        let (new_yield, _bonus, _potential) = b.new_direction_potential().unwrap();
        assert!(new_yield > 0.0);
    }

    #[test]
    fn wanted_allocation_conserves_agent_count() {
        let mut b = board();
        b.register_direction("A").unwrap();
        b.register_direction("B").unwrap();
        b.register_direction("C").unwrap();
        b.record_evaluation("a1", "A", Some(100.0), Some(90.0), vec![], 1).unwrap();
        b.record_evaluation("b1", "B", Some(80.0), Some(75.0), vec![], 1).unwrap();
        b.record_evaluation("c1", "C", Some(75.0), Some(70.0), vec![], 1).unwrap();
        let wanted = b.recommend().unwrap();
        assert_eq!(wanted.iter().map(|item| item.wanted_agents).sum::<usize>(), 4);
        assert!(wanted.iter().any(|item| item.arm == CoralArm::New));
    }

    #[test]
    fn all_zero_potential_falls_back_to_global_best_direction() {
        let mut config = CoralBaselineConfig::default();
        config.kappa = 0.0;
        config.prior_n = 1;
        let mut b = CoralDirectionBoard::new(config).unwrap();
        b.register_direction("A").unwrap();
        b.register_direction("B").unwrap();
        b.record_evaluation("a1", "A", Some(100.0), Some(100.0), vec![], 1).unwrap();
        b.record_evaluation("b1", "B", Some(100.0), Some(100.0), vec![], 1).unwrap();
        let wanted = b.recommend().unwrap();
        let best = wanted.iter().find(|item| item.wanted_agents > 0).unwrap();
        assert_eq!(best.arm, CoralArm::Direction("A".to_owned()));
        assert_eq!(best.wanted_agents, 4);
    }

    #[test]
    fn published_claim_caps_are_enforced_separately_from_wanted_shares() {
        let b = board();
        let mut b = b;
        b.register_direction("A").unwrap();
        b.validate_claim_capacity("A", 0, true).unwrap();
        assert!(b.validate_claim_capacity("A", 1, true).is_err());
        b.validate_claim_capacity("A", 1, false).unwrap();
        assert!(b.validate_claim_capacity("A", 2, false).is_err());
    }

    #[test]
    fn integrity_rejects_tampered_credit_and_global_best_receipts() {
        let mut b = board();
        b.register_direction("A").unwrap();
        b.record_evaluation("a1", "A", Some(100.0), Some(90.0), vec![], 1).unwrap();
        b.validate_integrity().unwrap();

        b.evaluations[0].credit = 0.0;
        assert!(b.validate_integrity().is_err());
        b.evaluations[0].credit = 0.1;
        b.evaluations[0].global_best_after = Some(101.0);
        assert!(b.validate_integrity().is_err());
    }

    #[test]
    fn integrity_rejects_direction_history_missing_or_adding_events() {
        let mut b = board();
        b.register_direction("A").unwrap();
        b.register_direction("B").unwrap();
        b.record_evaluation("a1", "A", Some(100.0), Some(90.0), vec![], 1).unwrap();
        b.validate_integrity().unwrap();

        b.directions.get_mut("A").unwrap().evaluations.clear();
        assert!(b.validate_integrity().is_err());
    }

    #[test]
    fn cost_is_recorded_but_does_not_silently_change_published_per_eval_yield() {
        let mut b = board();
        b.register_direction("A").unwrap();
        b.record_evaluation("a1", "A", Some(100.0), Some(90.0), vec![], 10).unwrap();
        let metric = &b.direction_metrics().unwrap()[0];
        assert!((metric.window_credit - 0.1).abs() < 1e-12);
        assert_eq!(b.evaluations[0].evaluation_cost_units, 10);
    }
}
