//! Deterministic selection policies for small nanotechnology pilot studies.
//! Feature-space distance is a coverage heuristic, not a physical property, uncertainty,
//! novelty, safety assessment, or evidence of scientific discovery.

use std::cmp::Ordering;

pub const REQUEST_SCHEMA_VERSION: &str = "nano-selection-request-v1";
pub const RECEIPT_SCHEMA_VERSION: &str = "nano-selection-receipt-v1";
pub const POLICY_REVISION: &str = "nano-selection-policy-v1";
const MAX_CANDIDATES: usize = 10_000;
const MAX_FEATURES: usize = 1_024;
const MAX_DIVERSITY_WORK: usize = 100_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SelectionPolicy { Random, Cheapest, MaxMinDiversity, CostAwareDiversity }

impl SelectionPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Cheapest => "cheapest",
            Self::MaxMinDiversity => "max_min_diversity",
            Self::CostAwareDiversity => "cost_aware_diversity",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub id: String,
    /// Predeclared, label-blind features named by feature_schema_id.
    pub features: Vec<f64>,
    /// Caller-estimated cost in a documented, consistent unit.
    pub estimated_cost: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectionRequest {
    pub schema_version: String,
    pub feature_schema_id: String,
    pub candidates: Vec<Candidate>,
    /// Label-free feature vectors from previously evaluated candidates, using the same schema.
    pub reference_features: Vec<Vec<f64>>,
    pub policy: SelectionPolicy,
    pub selection_count: usize,
    pub max_total_cost: Option<f64>,
    pub seed: u64,
    /// Cost penalty exponent in [0, 4]; zero disables the penalty.
    pub cost_exponent: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateDisposition { Selected, NotSelected, OverBudget }

impl CandidateDisposition {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Selected => "selected",
            Self::NotSelected => "not_selected",
            Self::OverBudget => "over_budget",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectionRecord {
    pub candidate_id: String,
    pub rank: usize,
    pub estimated_cost: f64,
    /// Minimum normalized RMS distance to prior selections, when applicable.
    pub nearest_selected_distance: Option<f64>,
    pub selection_score: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateDecision {
    pub candidate_id: String,
    pub disposition: CandidateDisposition,
    pub selected_rank: Option<usize>,
    pub reason: Option<&'static str>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectionReceipt {
    pub schema_version: String,
    pub policy_revision: String,
    pub feature_schema_id: String,
    pub policy: SelectionPolicy,
    pub seed: u64,
    pub input_candidate_count: usize,
    pub reference_feature_count: usize,
    pub requested_selection_count: usize,
    pub selected_count: usize,
    pub max_total_cost: Option<f64>,
    pub total_estimated_cost: f64,
    pub remaining_estimated_budget: Option<f64>,
    pub selections: Vec<SelectionRecord>,
    /// Exactly one audit decision per input candidate, in input order.
    pub candidate_audit: Vec<CandidateDecision>,
    pub caveats: Vec<&'static str>,
}

#[derive(Clone, Debug)]
struct Work {
    disposition: Option<CandidateDisposition>,
    rank: Option<usize>,
    reason: Option<&'static str>,
}

impl Work {
    fn new() -> Self { Self { disposition: None, rank: None, reason: None } }
}

/// Select candidates without reading oracle outcomes or target labels.
///
/// Features are min-max normalized per dimension; distance is RMS Euclidean distance.
/// Cost-aware score = distance * exp(-exponent * ln(cost / minimum_pool_cost)).
/// The caller must independently verify feature provenance and label blinding.
pub fn select(request: SelectionRequest) -> Result<SelectionReceipt, String> {
    validate(&request)?;
    let n = request.candidates.len();
    let limit = request.selection_count.min(n);
    let (normalized, normalized_references) =
        normalize(&request.candidates, &request.reference_features);
    let center = centroid(&normalized);
    let min_cost = request.candidates.iter().map(|c| c.estimated_cost).fold(f64::INFINITY, f64::min);
    let mut work = vec![Work::new(); n];
    let mut selections = Vec::with_capacity(limit);
    let mut chosen = Vec::<usize>::with_capacity(limit);
    let mut spent = 0.0;

    match request.policy {
        SelectionPolicy::Random => {
            let mut order: Vec<usize> = (0..n).collect();
            shuffle(&mut order, request.seed);
            for i in order {
                if selections.len() >= limit { break; }
                if !affordable(&request, spent, i) {
                    work[i].disposition = Some(CandidateDisposition::OverBudget);
                    work[i].reason = Some("estimated_cost_exceeds_remaining_budget");
                } else {
                    record_selection(&request, i, None, None, &mut work, &mut selections, &mut chosen, &mut spent);
                }
            }
        }
        SelectionPolicy::Cheapest => {
            while selections.len() < limit {
                mark_over_budget(&request, spent, &mut work);
                let mut best: Option<usize> = None;
                for (i, candidate) in request.candidates.iter().enumerate() {
                    if work[i].disposition.is_some() { continue; }
                    if best.is_none_or(|j| cost_then_id(candidate, &request.candidates[j]) == Ordering::Less) {
                        best = Some(i);
                    }
                }
                let Some(i) = best else { break };
                record_selection(&request, i, None, None, &mut work, &mut selections, &mut chosen, &mut spent);
            }
        }
        SelectionPolicy::MaxMinDiversity | SelectionPolicy::CostAwareDiversity => {
            while selections.len() < limit {
                mark_over_budget(&request, spent, &mut work);
                let mut best: Option<(usize, f64, Option<f64>)> = None;
                for (i, candidate) in request.candidates.iter().enumerate() {
                    if work[i].disposition.is_some() { continue; }
                    let nearest = if chosen.is_empty() && normalized_references.is_empty() {
                        None
                    } else {
                        let prior_distance = normalized_references
                            .iter()
                            .map(|prior| distance(&normalized[i], prior))
                            .chain(chosen.iter().map(|j| distance(&normalized[i], &normalized[*j])))
                            .fold(f64::INFINITY, f64::min);
                        Some(prior_distance)
                    };
                    let raw = nearest.unwrap_or_else(|| distance(&normalized[i], &center));
                    let score = if request.policy == SelectionPolicy::CostAwareDiversity {
                        cost_adjust(raw, request.candidates[i].estimated_cost, min_cost, request.cost_exponent)
                    } else { raw };
                    let replace = match best {
                        None => true,
                        Some((j, current, _)) => score.total_cmp(&current) == Ordering::Greater
                            || (score.total_cmp(&current) == Ordering::Equal
                                && cost_then_id(&request.candidates[i], &request.candidates[j]) == Ordering::Less),
                    };
                    if replace { best = Some((i, score, nearest)); }
                }
                let Some((i, score, nearest)) = best else { break };
                record_selection(&request, i, Some(score), nearest, &mut work, &mut selections, &mut chosen, &mut spent);
            }
        }
    }

    let fallback = if selections.len() >= limit { "selection_count_reached" } else { "no_affordable_candidate_remaining" };
    let audit = request.candidates.iter().enumerate().map(|(i, c)| CandidateDecision {
        candidate_id: c.id.clone(),
        disposition: work[i].disposition.unwrap_or(CandidateDisposition::NotSelected),
        selected_rank: work[i].rank,
        reason: work[i].reason.or_else(|| if work[i].disposition.is_none() { Some(fallback) } else { None }),
    }).collect();
    Ok(SelectionReceipt {
        schema_version: RECEIPT_SCHEMA_VERSION.to_owned(),
        policy_revision: POLICY_REVISION.to_owned(),
        feature_schema_id: request.feature_schema_id,
        policy: request.policy,
        seed: request.seed,
        input_candidate_count: n,
        reference_feature_count: request.reference_features.len(),
        requested_selection_count: request.selection_count,
        selected_count: selections.len(),
        max_total_cost: request.max_total_cost,
        total_estimated_cost: spent,
        remaining_estimated_budget: request.max_total_cost.map(|b| (b - spent).max(0.0)),
        selections,
        candidate_audit: audit,
        caveats: vec![
            "Feature schema identity is caller-supplied and not independently verified.",
            "Feature provenance must establish label blindness and exclude future outcomes.",
            "Feature distance is not energy, scientific novelty, calibrated uncertainty, or safety.",
            "Estimated cost can differ from realized cost.",
            "A receipt does not establish model accuracy or scientific discovery.",
        ],
    })
}

fn validate(r: &SelectionRequest) -> Result<(), String> {
    if r.schema_version != REQUEST_SCHEMA_VERSION { return Err(format!("unsupported schema_version; expected {REQUEST_SCHEMA_VERSION}")); }
    if r.feature_schema_id.trim().is_empty() || r.feature_schema_id.len() > 256 { return Err("invalid feature_schema_id".into()); }
    if r.candidates.is_empty() { return Err("candidates must not be empty".into()); }
    if r.candidates.len() > MAX_CANDIDATES { return Err(format!("candidate count exceeds {MAX_CANDIDATES}")); }
    if r.selection_count == 0 { return Err("selection_count must be positive".into()); }
    if !r.cost_exponent.is_finite() || !(0.0..=4.0).contains(&r.cost_exponent) { return Err("cost_exponent must be finite in [0, 4]".into()); }
    if r.max_total_cost.is_some_and(|v| !v.is_finite() || v < 0.0) { return Err("max_total_cost must be finite and nonnegative".into()); }
    let dims = r.candidates[0].features.len();
    if dims == 0 || dims > MAX_FEATURES { return Err(format!("feature dimension count must be in 1..={MAX_FEATURES}")); }
    if r.candidates.len().saturating_add(r.reference_features.len()) > MAX_CANDIDATES {
        return Err(format!("candidate and reference feature count exceeds {MAX_CANDIDATES}"));
    }
    for reference in &r.reference_features {
        if reference.len() != dims {
            return Err(format!("reference feature vector has {} dimensions; expected {dims}", reference.len()));
        }
        if reference.iter().any(|v| !v.is_finite()) {
            return Err("reference feature vector contains non-finite values".into());
        }
    }
    let mut ids = std::collections::HashSet::with_capacity(r.candidates.len());
    let mut all_costs = 0.0;
    for c in &r.candidates {
        if c.id.trim().is_empty() || c.id.len() > 256 || c.id.chars().any(char::is_control) { return Err("invalid candidate ID".into()); }
        if !ids.insert(c.id.as_str()) { return Err(format!("duplicate candidate ID: {}", c.id)); }
        if c.features.len() != dims { return Err(format!("candidate {} has {} features; expected {dims}", c.id, c.features.len())); }
        if c.features.iter().any(|v| !v.is_finite()) { return Err(format!("candidate {} has non-finite feature", c.id)); }
        if !c.estimated_cost.is_finite() || c.estimated_cost <= 0.0 { return Err(format!("candidate {} estimated_cost must be finite and positive", c.id)); }
        all_costs += c.estimated_cost;
        if !all_costs.is_finite() { return Err("aggregate estimated cost overflows f64".into()); }
    }
    if matches!(r.policy, SelectionPolicy::MaxMinDiversity | SelectionPolicy::CostAwareDiversity) {
        let candidate_count = r.candidates.len();
        let reference_count = r.reference_features.len();
        let k = r.selection_count.min(candidate_count);
        // Each greedy step compares each remaining candidate with prior-reference vectors
        // and vectors selected in this batch. Include normalization work as well.
        let reference_comparisons = k.saturating_mul(reference_count);
        let within_batch_comparisons = k.saturating_mul(k.saturating_sub(1)) / 2;
        let pair_dimension_work = candidate_count
            .saturating_mul(reference_comparisons.saturating_add(within_batch_comparisons))
            .saturating_mul(dims);
        let normalization_work = candidate_count
            .saturating_add(reference_count)
            .saturating_mul(dims);
        let work = pair_dimension_work.saturating_add(normalization_work);
        if work > MAX_DIVERSITY_WORK {
            return Err(format!("diversity work estimate {work} exceeds safety limit {MAX_DIVERSITY_WORK}"));
        }
    }
    Ok(())
}

fn normalize(
    candidates: &[Candidate],
    references: &[Vec<f64>],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let dimensions = candidates[0].features.len();
    let mut low = vec![f64::INFINITY; dimensions];
    let mut high = vec![f64::NEG_INFINITY; dimensions];
    for features in candidates.iter().map(|candidate| &candidate.features).chain(references.iter()) {
        for (dimension, value) in features.iter().enumerate() {
            low[dimension] = low[dimension].min(*value);
            high[dimension] = high[dimension].max(*value);
        }
    }
    let ranges: Vec<_> = (0..dimensions).map(|dimension| {
        let scale = low[dimension].abs().max(high[dimension].abs()).max(1.0);
        let scaled_low = low[dimension] / scale;
        (scale, scaled_low, high[dimension] / scale - scaled_low)
    }).collect();
    let normalize_one = |features: &[f64]| {
        features.iter().enumerate().map(|(dimension, value)| {
            let (scale, scaled_low, span) = ranges[dimension];
            if span.total_cmp(&0.0) == Ordering::Equal {
                0.0
            } else {
                ((*value / scale - scaled_low) / span).clamp(0.0, 1.0)
            }
        }).collect()
    };
    (
        candidates.iter().map(|candidate| normalize_one(&candidate.features)).collect(),
        references.iter().map(|features| normalize_one(features)).collect(),
    )
}

fn centroid(ps: &[Vec<f64>]) -> Vec<f64> {
    let mut mean = vec![0.0; ps[0].len()];
    for p in ps { for (j, v) in p.iter().enumerate() { mean[j] += *v / ps.len() as f64; } }
    mean
}

fn distance(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
}

fn cost_adjust(distance: f64, cost: f64, min_cost: f64, exponent: f64) -> f64 {
    if exponent.total_cmp(&0.0) == Ordering::Equal { distance } else { distance * (-(exponent * (cost.ln() - min_cost.ln()).max(0.0))).exp() }
}

fn cost_then_id(a: &Candidate, b: &Candidate) -> Ordering {
    a.estimated_cost.total_cmp(&b.estimated_cost).then_with(|| a.id.cmp(&b.id))
}

fn affordable(r: &SelectionRequest, spent: f64, i: usize) -> bool {
    r.max_total_cost.is_none_or(|budget| r.candidates[i].estimated_cost <= budget - spent)
}

fn mark_over_budget(r: &SelectionRequest, spent: f64, w: &mut [Work]) {
    for (i, state) in w.iter_mut().enumerate() {
        if state.disposition.is_none() && !affordable(r, spent, i) {
            state.disposition = Some(CandidateDisposition::OverBudget);
            state.reason = Some("estimated_cost_exceeds_remaining_budget");
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn record_selection(
    r: &SelectionRequest, i: usize, score: Option<f64>, nearest: Option<f64>,
    work: &mut [Work], out: &mut Vec<SelectionRecord>, chosen: &mut Vec<usize>, spent: &mut f64,
) {
    let rank = out.len() + 1;
    let c = &r.candidates[i];
    work[i].disposition = Some(CandidateDisposition::Selected);
    work[i].rank = Some(rank);
    out.push(SelectionRecord {
        candidate_id: c.id.clone(), rank, estimated_cost: c.estimated_cost,
        nearest_selected_distance: nearest, selection_score: score,
    });
    chosen.push(i);
    *spent += c.estimated_cost;
}

fn shuffle(order: &mut [usize], seed: u64) {
    let mut rng = SplitMix64(seed);
    for end in (1..order.len()).rev() { order.swap(end, rng.bounded((end + 1) as u64) as usize); }
}

struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn bounded(&mut self, bound: u64) -> u64 {
        let threshold = bound.wrapping_neg() % bound;
        loop { let value = self.next(); if value >= threshold { return value % bound; } }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn c(id: &str, f: &[f64], cost: f64) -> Candidate {
        Candidate { id: id.into(), features: f.to_vec(), estimated_cost: cost }
    }
    fn req(policy: SelectionPolicy, candidates: Vec<Candidate>) -> SelectionRequest {
        SelectionRequest {
            schema_version: REQUEST_SCHEMA_VERSION.into(), feature_schema_id: "test-v1".into(),
            candidates, reference_features: vec![], policy, selection_count: 2,
            max_total_cost: None, seed: 7, cost_exponent: 1.0,
        }
    }

    #[test]
    fn max_min_spreads_selection() {
        let r = req(SelectionPolicy::MaxMinDiversity, vec![c("a", &[0.,0.],1.), c("b", &[0.02,0.01],1.), c("c", &[0.98,0.99],1.), c("d", &[1.,1.],1.)]);
        let receipt = select(r).unwrap();
        assert_eq!(receipt.selections.iter().map(|s| s.candidate_id.as_str()).collect::<Vec<_>>(), vec!["a","d"]);
        assert!(receipt.selections[1].nearest_selected_distance.unwrap() > 0.9);
    }

    #[test]
    fn budget_and_candidate_accounting_are_fail_closed() {
        let mut r = req(SelectionPolicy::Cheapest, vec![c("a",&[0.],2.), c("b",&[0.5],1.), c("c",&[1.],2.)]);
        r.selection_count = 3; r.max_total_cost = Some(3.);
        let receipt = select(r).unwrap();
        assert_eq!(receipt.selected_count, 2);
        assert!((receipt.total_estimated_cost - 3.0).abs() <= f64::EPSILON);
        assert_eq!(receipt.candidate_audit.len(), 3);
        assert!(receipt.candidate_audit.iter().any(|d| d.disposition == CandidateDisposition::OverBudget));
    }

    #[test]
    fn random_baseline_is_deterministic() {
        let cs = vec![c("a",&[0.,0.1],1.),c("b",&[1.,0.1],1.),c("c",&[0.3,0.9],1.),c("d",&[0.7,0.5],1.)];
        assert_eq!(select(req(SelectionPolicy::Random, cs.clone())).unwrap(), select(req(SelectionPolicy::Random, cs)).unwrap());
    }

    #[test]
    fn diversity_respects_previous_feature_only_queries() {
        let mut r = req(
            SelectionPolicy::MaxMinDiversity,
            vec![c("near", &[0.1], 1.0), c("middle", &[0.2], 1.0), c("far", &[1.0], 1.0)],
        );
        r.selection_count = 1;
        r.reference_features = vec![vec![0.0]];
        let receipt = select(r).unwrap();
        assert_eq!(receipt.selections[0].candidate_id, "far");
        assert_eq!(receipt.reference_feature_count, 1);
    }

    #[test]
    fn invalid_reference_features_are_rejected() {
        let mut r = req(
            SelectionPolicy::MaxMinDiversity,
            vec![c("candidate-a", &[0.0, 0.1], 1.0), c("candidate-b", &[0.9, 1.0], 1.0)],
        );
        r.reference_features = vec![vec![0.5]];
        assert!(select(r).unwrap_err().contains("reference feature vector has 1 dimensions"));
    }

    #[test]
    fn large_reference_history_cannot_bypass_diversity_work_cap() {
        let features = [0.0; 100];
        let candidates: Vec<_> = (0..1_000)
            .map(|index| c(&format!("candidate-{index:04}"), &features, 1.0))
            .collect();
        let references = vec![features.to_vec(); 1_000];
        let mut request = req(SelectionPolicy::MaxMinDiversity, candidates);
        request.selection_count = 1;
        request.reference_features = references;
        assert!(select(request).unwrap_err().contains("diversity work estimate"));
    }

    #[test]
    fn non_finite_reference_features_are_rejected() {
        let mut r = req(
            SelectionPolicy::MaxMinDiversity,
            vec![c("a", &[0.0], 1.0), c("b", &[1.0], 1.0)],
        );
        r.reference_features = vec![vec![f64::INFINITY]];
        assert!(select(r).unwrap_err().contains("reference feature vector contains non-finite values"));
    }

    #[test]
    fn duplicate_ids_are_rejected() {
        let r = req(SelectionPolicy::Random, vec![c("x",&[0.],1.),c("x",&[1.],1.)]);
        assert!(select(r).unwrap_err().contains("duplicate candidate ID"));
    }

    #[test]
    fn inconsistent_dimensions_are_rejected() {
        let r = req(SelectionPolicy::MaxMinDiversity, vec![c("a",&[0.],1.),c("b",&[1.,2.],1.)]);
        assert!(select(r).unwrap_err().contains("expected 1"));
    }

    #[test]
    fn non_finite_features_and_costs_are_rejected() {
        let r = req(SelectionPolicy::MaxMinDiversity, vec![c("a",&[f64::NAN],1.),c("b",&[1.],1.)]);
        assert!(select(r).unwrap_err().contains("non-finite feature"));
        let r = req(SelectionPolicy::Cheapest, vec![c("a",&[0.],f64::INFINITY)]);
        assert!(select(r).unwrap_err().contains("estimated_cost"));
    }

    #[test]
    fn zero_budget_selects_none() {
        let mut r = req(SelectionPolicy::MaxMinDiversity, vec![c("a",&[0.],1.),c("b",&[1.],2.)]);
        r.max_total_cost = Some(0.);
        let receipt = select(r).unwrap();
        assert_eq!(receipt.selected_count, 0);
        assert!(receipt.candidate_audit.iter().all(|d| d.disposition == CandidateDisposition::OverBudget));
    }

    #[test]
    fn handles_extreme_finite_features() {
        let r = req(SelectionPolicy::MaxMinDiversity, vec![c("low",&[-f64::MAX],1.),c("high",&[f64::MAX],1.)]);
        let receipt = select(r).unwrap();
        assert!(receipt.selections.iter().all(|s| s.selection_score.unwrap().is_finite()));
    }

    #[test]
    fn zero_cost_exponent_equals_max_min() {
        let cs = vec![c("a",&[0.,0.],1.),c("b",&[0.2,0.2],4.),c("c",&[1.,1.],2.)];
        let max_min = select(req(SelectionPolicy::MaxMinDiversity, cs.clone())).unwrap();
        let mut r = req(SelectionPolicy::CostAwareDiversity, cs); r.cost_exponent = 0.;
        assert_eq!(max_min.selections, select(r).unwrap().selections);
    }
}
