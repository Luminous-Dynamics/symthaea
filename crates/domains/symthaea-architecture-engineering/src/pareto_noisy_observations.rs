//! Time-uniform uncertainty intervals and confidence-aware Pareto relations.
//!
//! This module handles repeated, noisy objective observations without collapsing objectives
//! into a scalar score. Each objective must declare valid observation bounds. A union-bound
//! alpha-spending schedule provides simultaneous Hoeffding intervals across the frozen
//! candidate/objective family and every replicate prefix. Claims are conditional on independent
//! bounded observations and honest evidence/seed provenance.
//!
//! No hypervolume approximation or normality assumption is used here.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::pareto_objectives::ObjectiveDirection;

const SCHEMA: &str = "symthaea.architecture-engineering.noisy-pareto-observations";
const VERSION: &str = "0.1.0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoisyParetoError {
    pub path: String,
    pub message: String,
}

impl NoisyParetoError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { path: path.into(), message: message.into() }
    }
}

impl fmt::Display for NoisyParetoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for NoisyParetoError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoisyObjectiveSpec {
    pub objective_id: String,
    pub direction: ObjectiveDirection,
    /// Certified lower bound for every single-replicate observation, not a normalization anchor.
    pub observation_lower_bound: f64,
    /// Certified upper bound for every single-replicate observation, not a normalization anchor.
    pub observation_upper_bound: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoisyObservationPlan {
    pub schema: String,
    pub version: String,
    pub campaign_id: String,
    pub objective_specs: Vec<NoisyObjectiveSpec>,
    /// Family-wise error budget across the frozen candidate/objective family and all prefixes.
    pub familywise_alpha: f64,
    /// Frozen upper bound used for the union-bound allocation across candidates.
    pub max_candidates: usize,
    /// Minimum independent replicates required before a candidate enters confidence comparisons.
    pub minimum_replicates: usize,
}

impl NoisyObservationPlan {
    pub fn new(
        campaign_id: impl Into<String>,
        objective_specs: Vec<NoisyObjectiveSpec>,
        familywise_alpha: f64,
        max_candidates: usize,
        minimum_replicates: usize,
    ) -> Result<Self, NoisyParetoError> {
        let plan = Self {
            schema: SCHEMA.to_owned(),
            version: VERSION.to_owned(),
            campaign_id: campaign_id.into(),
            objective_specs,
            familywise_alpha,
            max_candidates,
            minimum_replicates,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<(), NoisyParetoError> {
        if self.schema != SCHEMA || self.version != VERSION {
            return Err(NoisyParetoError::new("plan.schema", "unsupported noisy-observation schema/version"));
        }
        if self.campaign_id.trim().is_empty() {
            return Err(NoisyParetoError::new("plan.campaign_id", "campaign ID must be non-empty"));
        }
        if !(2..=16).contains(&self.objective_specs.len()) {
            return Err(NoisyParetoError::new("plan.objective_specs", "v0.1 uncertainty-front analysis supports two through sixteen objectives"));
        }
        if !self.familywise_alpha.is_finite() || !(0.0..1.0).contains(&self.familywise_alpha) {
            return Err(NoisyParetoError::new("plan.familywise_alpha", "family-wise alpha must be finite and strictly between zero and one"));
        }
        if self.max_candidates == 0 || self.minimum_replicates == 0 {
            return Err(NoisyParetoError::new("plan", "candidate and minimum-replicate limits must be positive"));
        }
        let mut ids = BTreeSet::new();
        for (index, objective) in self.objective_specs.iter().enumerate() {
            if objective.objective_id.trim().is_empty() || !ids.insert(objective.objective_id.as_str()) {
                return Err(NoisyParetoError::new(format!("plan.objective_specs[{index}].objective_id"), "objective IDs must be non-empty and unique"));
            }
            if !objective.observation_lower_bound.is_finite()
                || !objective.observation_upper_bound.is_finite()
                || objective.observation_lower_bound >= objective.observation_upper_bound
                || !(objective.observation_upper_bound - objective.observation_lower_bound).is_finite()
            {
                return Err(NoisyParetoError::new(format!("plan.objective_specs[{index}].bounds"), "observation bounds must be finite and strictly increasing"));
            }
        }
        Ok(())
    }

    pub fn plan_sha256(&self) -> Result<String, NoisyParetoError> {
        self.validate()?;
        digest_serializable(self)
    }
}

/// One independently seeded noisy objective-vector observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoisyReplicate {
    pub replicate_id: String,
    /// The seed is provenance, not proof of statistical independence.
    pub sample_seed: u64,
    pub objective_values: Vec<f64>,
    /// Digest of the retained raw observation artifact.
    pub evidence_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct CandidateSamples {
    decision_sha256: String,
    samples: BTreeMap<String, NoisyReplicate>,
    seeds: BTreeSet<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectiveConfidenceInterval {
    pub objective_id: String,
    pub sample_count: u64,
    pub sample_mean: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    /// Per-coordinate, per-prefix alpha from the frozen summable spending schedule.
    pub alpha_at_prefix: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceParetoStatus {
    InsufficientReplicates,
    CertainlyDominated,
    CertainlyNondominatedAmongObserved,
    Unresolved,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoisyCandidateSummary {
    pub candidate_id: String,
    pub decision_sha256: String,
    pub sample_count: u64,
    pub confidence_intervals: Vec<ObjectiveConfidenceInterval>,
    pub certainly_dominated_by: Vec<String>,
    pub possibly_dominated_by: Vec<String>,
    pub status: ConfidenceParetoStatus,
    pub sample_set_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoisyParetoReport {
    pub schema: String,
    pub version: String,
    pub campaign_id: String,
    pub plan_sha256: String,
    pub candidate_summaries: Vec<NoisyCandidateSummary>,
    /// Conditional family-wise confidence guarantee for the candidate/objective/prefix family.
    pub nominal_familywise_confidence: f64,
    /// True only as a schema/provenance assertion; statistical independence is an experimental assumption.
    pub unique_replicate_seeds_enforced: bool,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct NoisyReportPayload<'a> {
    schema: &'a str,
    version: &'a str,
    campaign_id: &'a str,
    plan_sha256: &'a str,
    candidate_summaries: &'a [NoisyCandidateSummary],
    nominal_familywise_confidence: f64,
    unique_replicate_seeds_enforced: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoisyParetoStudy {
    pub plan: NoisyObservationPlan,
    candidates: BTreeMap<String, CandidateSamples>,
    decision_owners: BTreeMap<String, String>,
}

impl NoisyParetoStudy {
    pub fn new(plan: NoisyObservationPlan) -> Result<Self, NoisyParetoError> {
        plan.validate()?;
        Ok(Self {
            plan,
            candidates: BTreeMap::new(),
            decision_owners: BTreeMap::new(),
        })
    }

    pub fn append(
        &mut self,
        candidate_id: impl Into<String>,
        decision_sha256: impl Into<String>,
        replicate: NoisyReplicate,
    ) -> Result<(), NoisyParetoError> {
        self.validate()?;
        let candidate_id = candidate_id.into();
        let decision_sha256 = decision_sha256.into();
        if candidate_id.trim().is_empty() {
            return Err(NoisyParetoError::new("candidate_id", "candidate ID must be non-empty"));
        }
        validate_sha256(&decision_sha256, "decision_sha256")?;
        if replicate.replicate_id.trim().is_empty() {
            return Err(NoisyParetoError::new("replicate.replicate_id", "replicate ID must be non-empty"));
        }
        validate_sha256(&replicate.evidence_sha256, "replicate.evidence_sha256")?;
        if replicate.objective_values.len() != self.plan.objective_specs.len() {
            return Err(NoisyParetoError::new("replicate.objective_values", "objective vector dimension mismatch"));
        }
        for (index, (value, spec)) in replicate.objective_values.iter()
            .zip(&self.plan.objective_specs).enumerate()
        {
            if !value.is_finite()
                || *value < spec.observation_lower_bound
                || *value > spec.observation_upper_bound
            {
                return Err(NoisyParetoError::new(
                    format!("replicate.objective_values[{index}]"),
                    "observation is non-finite or outside the frozen certified bounds",
                ));
            }
        }

        if self.decision_owners.get(&decision_sha256).is_some_and(|owner| owner != &candidate_id) {
            return Err(NoisyParetoError::new("decision_sha256", "identical decision bytes must share one canonical candidate ID"));
        }
        let existing = self.candidates.get(&candidate_id);
        if existing.is_none() && self.candidates.len() >= self.plan.max_candidates {
            return Err(NoisyParetoError::new("candidates", "frozen maximum candidate count exceeded"));
        }
        if existing.is_some_and(|state| state.decision_sha256 != decision_sha256) {
            return Err(NoisyParetoError::new("decision_sha256", "candidate decision digest changed"));
        }
        if existing.is_some_and(|state| {
            state.samples.contains_key(&replicate.replicate_id)
                || state.seeds.contains(&replicate.sample_seed)
        }) {
            return Err(NoisyParetoError::new("replicate", "replicate ID and seed must both be unique within a candidate"));
        }

        // All validation is complete before state mutation.
        self.decision_owners.entry(decision_sha256.clone()).or_insert_with(|| candidate_id.clone());
        let state = self.candidates.entry(candidate_id).or_insert_with(|| CandidateSamples {
            decision_sha256,
            samples: BTreeMap::new(),
            seeds: BTreeSet::new(),
        });
        state.seeds.insert(replicate.sample_seed);
        state.samples.insert(replicate.replicate_id.clone(), replicate);
        Ok(())
    }

    pub fn validate(&self) -> Result<(), NoisyParetoError> {
        self.plan.validate()?;
        if self.candidates.len() > self.plan.max_candidates {
            return Err(NoisyParetoError::new("candidates", "candidate count exceeds frozen limit"));
        }
        let mut expected_owners = BTreeMap::<String, String>::new();
        for (candidate_id, state) in &self.candidates {
            if candidate_id.trim().is_empty() || state.samples.is_empty() {
                return Err(NoisyParetoError::new("candidates", "candidate IDs and observation sets must be non-empty"));
            }
            validate_sha256(&state.decision_sha256, "candidate.decision_sha256")?;
            if expected_owners.insert(state.decision_sha256.clone(), candidate_id.clone()).is_some() {
                return Err(NoisyParetoError::new("candidate.decision_sha256", "identical decision digest appears under multiple candidate IDs"));
            }
            if state.samples.values().any(|sample| {
                sample.replicate_id.trim().is_empty()
                    || sample.objective_values.len() != self.plan.objective_specs.len()
            }) {
                return Err(NoisyParetoError::new("candidate.samples", "invalid replicate record"));
            }
            let mut sample_ids = BTreeSet::new();
            let mut sample_seeds = BTreeSet::new();
            for sample in state.samples.values() {
                if !sample_ids.insert(sample.replicate_id.as_str()) || !sample_seeds.insert(sample.sample_seed) {
                    return Err(NoisyParetoError::new("candidate.samples", "duplicate replicate ID or seed"));
                }
                validate_sha256(&sample.evidence_sha256, "candidate.sample.evidence_sha256")?;
                for (value, spec) in sample.objective_values.iter().zip(&self.plan.objective_specs) {
                    if !value.is_finite() || *value < spec.observation_lower_bound || *value > spec.observation_upper_bound {
                        return Err(NoisyParetoError::new("candidate.samples", "replicate value outside frozen observation bounds"));
                    }
                }
            }
            if sample_seeds != state.seeds {
                return Err(NoisyParetoError::new("candidate.seeds", "cached seed index differs from the retained replicate set"));
            }
        }
        if expected_owners != self.decision_owners {
            return Err(NoisyParetoError::new("decision_owners", "decision ownership index differs from candidate records"));
        }
        Ok(())
    }

    pub fn report(&self) -> Result<NoisyParetoReport, NoisyParetoError> {
        self.validate()?;
        let plan_sha256 = self.plan.plan_sha256()?;
        let mut summaries = Vec::with_capacity(self.candidates.len());
        for (candidate_id, samples) in &self.candidates {
            summaries.push(self.summarize_candidate(candidate_id, samples)?);
        }
        let mut by_id = summaries.iter().map(|item| (item.candidate_id.clone(), item.clone()))
            .collect::<BTreeMap<_, _>>();
        let ids = by_id.keys().cloned().collect::<Vec<_>>();
        for candidate_id in &ids {
            let candidate = by_id.get(candidate_id).cloned().ok_or_else(|| NoisyParetoError::new("report", "candidate summary disappeared"))?;
            if candidate.sample_count < self.plan.minimum_replicates as u64 {
                continue;
            }
            let mut certain = Vec::new();
            let mut possible = Vec::new();
            for other_id in &ids {
                if other_id == candidate_id {
                    continue;
                }
                let other = by_id.get(other_id).ok_or_else(|| NoisyParetoError::new("report", "comparison summary disappeared"))?;
                if other.sample_count < self.plan.minimum_replicates as u64 {
                    continue;
                }
                if certainly_dominates(other, &candidate, &self.plan.objective_specs) {
                    certain.push(other_id.clone());
                }
                if possibly_dominates(other, &candidate, &self.plan.objective_specs) {
                    possible.push(other_id.clone());
                }
            }
            let status = if !certain.is_empty() {
                ConfidenceParetoStatus::CertainlyDominated
            } else if possible.is_empty() {
                ConfidenceParetoStatus::CertainlyNondominatedAmongObserved
            } else {
                ConfidenceParetoStatus::Unresolved
            };
            let entry = by_id.get_mut(candidate_id).ok_or_else(|| NoisyParetoError::new("report", "candidate summary disappeared"))?;
            entry.certainly_dominated_by = certain;
            entry.possibly_dominated_by = possible;
            entry.status = status;
        }
        summaries = by_id.into_values().collect();

        let nominal_familywise_confidence = 1.0 - self.plan.familywise_alpha;
        let payload = NoisyReportPayload {
            schema: SCHEMA,
            version: VERSION,
            campaign_id: &self.plan.campaign_id,
            plan_sha256: &plan_sha256,
            candidate_summaries: &summaries,
            nominal_familywise_confidence,
            unique_replicate_seeds_enforced: true,
        };
        let report_sha256 = digest_serializable(&payload)?;
        Ok(NoisyParetoReport {
            schema: SCHEMA.to_owned(),
            version: VERSION.to_owned(),
            campaign_id: self.plan.campaign_id.clone(),
            plan_sha256,
            candidate_summaries: summaries,
            nominal_familywise_confidence,
            unique_replicate_seeds_enforced: true,
            report_sha256,
        })
    }

    fn summarize_candidate(
        &self,
        candidate_id: &str,
        state: &CandidateSamples,
    ) -> Result<NoisyCandidateSummary, NoisyParetoError> {
        let sample_count = u64::try_from(state.samples.len())
            .map_err(|_| NoisyParetoError::new("candidate.sample_count", "sample count exceeds u64"))?;
        let mut confidence_intervals = Vec::with_capacity(self.plan.objective_specs.len());
        for (index, spec) in self.plan.objective_specs.iter().enumerate() {
            let mut sum = 0.0;
            let mut compensation = 0.0;
            for sample in state.samples.values() {
                let value = sample.objective_values[index];
                // Compensated (Kahan) summation in canonical replicate-ID order.
                let adjusted = value - compensation;
                let next = sum + adjusted;
                compensation = (next - sum) - adjusted;
                sum = next;
            }
            let mean = sum / sample_count as f64;
            let alpha_coordinate = self.plan.familywise_alpha
                / (self.plan.max_candidates as f64 * self.plan.objective_specs.len() as f64);
            let n = sample_count as f64;
            let alpha_at_prefix = alpha_coordinate / (n * (n + 1.0));
            let radius = (spec.observation_upper_bound - spec.observation_lower_bound)
                * ((2.0 / alpha_at_prefix).ln() / (2.0 * n)).sqrt();
            if !mean.is_finite() || !radius.is_finite() || !alpha_at_prefix.is_finite() || alpha_at_prefix <= 0.0 {
                return Err(NoisyParetoError::new(
                    format!("candidate[{candidate_id}].objective[{index}]"),
                    "confidence interval calculation overflowed or became non-finite",
                ));
            }
            confidence_intervals.push(ObjectiveConfidenceInterval {
                objective_id: spec.objective_id.clone(),
                sample_count,
                sample_mean: mean,
                lower_bound: (mean - radius).max(spec.observation_lower_bound),
                upper_bound: (mean + radius).min(spec.observation_upper_bound),
                alpha_at_prefix,
            });
        }
        let sample_set_sha256 = digest_serializable(&state.samples)?;
        let status = if sample_count < self.plan.minimum_replicates as u64 {
            ConfidenceParetoStatus::InsufficientReplicates
        } else {
            ConfidenceParetoStatus::Unresolved
        };
        Ok(NoisyCandidateSummary {
            candidate_id: candidate_id.to_owned(),
            decision_sha256: state.decision_sha256.clone(),
            sample_count,
            confidence_intervals,
            certainly_dominated_by: Vec::new(),
            possibly_dominated_by: Vec::new(),
            status,
            sample_set_sha256,
        })
    }
}

fn utility_interval(
    summary: &NoisyCandidateSummary,
    specs: &[NoisyObjectiveSpec],
) -> (Vec<f64>, Vec<f64>) {
    let mut lower = Vec::with_capacity(specs.len());
    let mut upper = Vec::with_capacity(specs.len());
    for (interval, spec) in summary.confidence_intervals.iter().zip(specs) {
        match spec.direction {
            ObjectiveDirection::Maximize => {
                lower.push(interval.lower_bound);
                upper.push(interval.upper_bound);
            }
            ObjectiveDirection::Minimize => {
                lower.push(-interval.upper_bound);
                upper.push(-interval.lower_bound);
            }
        }
    }
    (lower, upper)
}

fn certainly_dominates(
    left: &NoisyCandidateSummary,
    right: &NoisyCandidateSummary,
    specs: &[NoisyObjectiveSpec],
) -> bool {
    let (left_lower, _) = utility_interval(left, specs);
    let (_, right_upper) = utility_interval(right, specs);
    dominates(&left_lower, &right_upper)
}

fn possibly_dominates(
    left: &NoisyCandidateSummary,
    right: &NoisyCandidateSummary,
    specs: &[NoisyObjectiveSpec],
) -> bool {
    let (_, left_upper) = utility_interval(left, specs);
    let (right_lower, _) = utility_interval(right, specs);
    dominates(&left_upper, &right_lower)
}

fn dominates(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(a, b)| a >= b)
        && left.iter().zip(right).any(|(a, b)| a > b)
}

fn validate_sha256(value: &str, path: &str) -> Result<(), NoisyParetoError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) {
        return Err(NoisyParetoError::new(path, "expected lowercase 64-character SHA-256 digest"));
    }
    Ok(())
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String, NoisyParetoError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| NoisyParetoError::new("serialization", format!("cannot serialize noisy Pareto receipt: {error}")))?;
    let digest = Sha256::digest(bytes);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const H1: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const H2: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn plan() -> NoisyObservationPlan {
        NoisyObservationPlan::new(
            "noisy-zdt-like",
            vec![
                NoisyObjectiveSpec {
                    objective_id: "quality".to_owned(),
                    direction: ObjectiveDirection::Maximize,
                    observation_lower_bound: 0.0,
                    observation_upper_bound: 1.0,
                },
                NoisyObjectiveSpec {
                    objective_id: "cost".to_owned(),
                    direction: ObjectiveDirection::Minimize,
                    observation_lower_bound: 0.0,
                    observation_upper_bound: 1.0,
                },
            ],
            0.05,
            10,
            2,
        ).unwrap()
    }

    fn sample(replicate: usize, seed: u64, values: [f64; 2]) -> NoisyReplicate {
        NoisyReplicate {
            replicate_id: format!("replicate-{replicate:04}"),
            sample_seed: seed,
            objective_values: values.to_vec(),
            evidence_sha256: H2.to_owned(),
        }
    }

    #[test]
    fn time_uniform_confidence_intervals_shrink_with_more_replicates() {
        let mut study = NoisyParetoStudy::new(plan()).unwrap();
        for i in 0..32 {
            study.append("candidate-a", H1, sample(i, i as u64 + 1, [0.7, 0.4])).unwrap();
        }
        let short = study.report().unwrap().candidate_summaries.remove(0);
        for i in 32..512 {
            study.append("candidate-a", H1, sample(i, i as u64 + 1, [0.7, 0.4])).unwrap();
        }
        let long = study.report().unwrap().candidate_summaries.remove(0);
        let short_width = short.confidence_intervals[0].upper_bound - short.confidence_intervals[0].lower_bound;
        let long_width = long.confidence_intervals[0].upper_bound - long.confidence_intervals[0].lower_bound;
        assert!(long_width < short_width);
    }

    #[test]
    fn certain_dominance_and_unresolved_tradeoffs_are_distinguished() {
        let mut study = NoisyParetoStudy::new(plan()).unwrap();
        for i in 0..512 {
            study.append("candidate-a", H1, sample(i, i as u64 + 1, [0.9, 0.1])).unwrap();
            study.append("candidate-b", H2, sample(i, i as u64 + 10_000, [0.1, 0.9])).unwrap();
        }
        let report = study.report().unwrap();
        let a = report.candidate_summaries.iter().find(|item| item.candidate_id == "candidate-a").unwrap();
        let b = report.candidate_summaries.iter().find(|item| item.candidate_id == "candidate-b").unwrap();
        assert_eq!(a.status, ConfidenceParetoStatus::CertainlyNondominatedAmongObserved);
        assert_eq!(b.status, ConfidenceParetoStatus::CertainlyNondominatedAmongObserved);

        let mut uncertain = NoisyParetoStudy::new(plan()).unwrap();
        uncertain.append("a", H1, sample(1, 1, [0.6, 0.4])).unwrap();
        uncertain.append("a", H1, sample(2, 2, [0.4, 0.6])).unwrap();
        uncertain.append("b", H2, sample(1, 3, [0.5, 0.5])).unwrap();
        uncertain.append("b", H2, sample(2, 4, [0.5, 0.5])).unwrap();
        assert!(uncertain.report().unwrap().candidate_summaries.iter()
            .all(|item| item.status == ConfidenceParetoStatus::Unresolved));
    }

    #[test]
    fn replicate_ids_seeds_bounds_and_decision_aliases_are_enforced() {
        let mut study = NoisyParetoStudy::new(plan()).unwrap();
        study.append("a", H1, sample(1, 7, [0.5, 0.5])).unwrap();
        assert!(study.append("a", H1, sample(1, 8, [0.5, 0.5])).is_err());
        assert!(study.append("a", H1, sample(2, 7, [0.5, 0.5])).is_err());
        assert!(study.append("b", H1, sample(1, 9, [0.5, 0.5])).is_err());
        assert!(study.append("a", H1, sample(3, 10, [1.5, 0.5])).is_err());
        assert_eq!(study.candidates["a"].samples.len(), 1);
        assert!(!study.candidates.contains_key("b"));
        study.validate().unwrap();
    }

    #[test]
    fn report_digest_is_independent_of_observation_insertion_order() {
        let mut left = NoisyParetoStudy::new(plan()).unwrap();
        let mut right = NoisyParetoStudy::new(plan()).unwrap();
        for i in 0..20 {
            left.append("a", H1, sample(i, i as u64 + 1, [0.7, 0.3])).unwrap();
        }
        for i in (0..20).rev() {
            right.append("a", H1, sample(i, i as u64 + 1, [0.7, 0.3])).unwrap();
        }
        assert_eq!(left.report().unwrap(), right.report().unwrap());
    }
}
