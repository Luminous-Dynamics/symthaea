//! Cost-accounted, query-firewalled multi-fidelity tabular NAS.
//!
//! Policies can request target observations at successively higher fidelities. The final
//! target table and oracle ranking remain private. Candidates touched at an early fidelity
//! are counted separately from candidates confirmed at final fidelity.
//!
//! ResumablePrefix charges C[next] - C[current] when the higher rung resumes the same run.
//! IndependentRuns charges C[next] again when each fidelity is a fresh evaluation.
//! This is an API boundary for trusted policy code, not an operating-system sandbox.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::tabular_nas_firewall::{
    ArchitectureFeatures, MetricDirection, TabularBenchmarkManifest, TopKCapture,
};

const RECEIPT_DOMAIN: &[u8] = b"symthaea.tabular-nas.multi-fidelity-query-trace.v0.1";
const IMPORT_DOMAIN: &[u8] = b"symthaea.tabular-nas.multi-fidelity-import.v0.1";
const CANONICAL_IMPORT_SCHEMA: &str = "symthaea.tabular-nas.multi-fidelity-canonical-import";
const CANONICAL_IMPORT_VERSION: &str = "0.1.0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FidelityLevel {
    pub id: String,
    /// Cost from a clean start to this fidelity.
    pub cumulative_cost_units: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FidelityCostModel {
    ResumablePrefix,
    IndependentRuns,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultiFidelityManifest {
    pub base: TabularBenchmarkManifest,
    pub fidelity_levels: Vec<FidelityLevel>,
    pub fidelity_cost_model: FidelityCostModel,
    /// Physical or declared unit, e.g. gpu_seconds, cpu_seconds, or epoch_equivalent.
    pub resource_unit: String,
    pub max_total_cost_units: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FidelityTargetLabel {
    pub architecture_id: String,
    pub fidelity_index: usize,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FidelityQueryObservation {
    pub architecture_id: String,
    pub metric_id: String,
    pub metric_direction: MetricDirection,
    pub fidelity_id: String,
    pub fidelity_index: usize,
    pub value: f64,
    pub incremental_cost_units: u64,
    pub total_cost_units: u64,
    pub remaining_cost_units: u64,
    pub is_final_fidelity: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FidelityQueryError {
    UnknownArchitecture,
    InvalidFidelity,
    FidelityNotAdvanced,
    FinalQueryBudgetExhausted,
    ResourceBudgetExhausted,
}

impl fmt::Display for FidelityQueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownArchitecture => write!(f, "unknown architecture ID"),
            Self::InvalidFidelity => write!(f, "fidelity index is outside the declared ladder"),
            Self::FidelityNotAdvanced => write!(f, "candidate fidelity must advance monotonically"),
            Self::FinalQueryBudgetExhausted => write!(f, "final-fidelity confirmation budget exhausted"),
            Self::ResourceBudgetExhausted => write!(f, "total resource budget exhausted"),
        }
    }
}
impl Error for FidelityQueryError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FidelityQueryAttemptStatus {
    Revealed,
    UnknownArchitecture,
    InvalidFidelity,
    FidelityNotAdvanced,
    FinalQueryBudgetExhausted,
    ResourceBudgetExhausted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FidelityQueryAttempt {
    pub attempt_index: usize,
    pub architecture_id: String,
    pub fidelity_index: Option<usize>,
    pub status: FidelityQueryAttemptStatus,
    pub incremental_cost_units: u64,
    pub total_cost_units_after_attempt: u64,
    pub successful_observation_index: Option<usize>,
    pub observation_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultiFidelityQueryReceipt {
    pub benchmark_id: String,
    pub metric_id: String,
    pub metric_direction: MetricDirection,
    pub split_id: String,
    pub proxy_calibration_split_id: String,
    pub proxy_calibration_sha256: String,
    pub source_artifact_sha256: String,
    pub canonical_import_sha256: String,
    pub fidelity_levels: Vec<FidelityLevel>,
    pub fidelity_cost_model: FidelityCostModel,
    pub resource_unit: String,
    pub max_final_queries: usize,
    pub max_total_cost_units: u64,
    pub unique_architectures_touched: usize,
    pub final_fidelity_queries: usize,
    pub total_cost_units: u64,
    pub touched_architecture_ids: Vec<String>,
    pub final_fidelity_architecture_ids: Vec<String>,
    pub attempts: Vec<FidelityQueryAttempt>,
    pub query_trace_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultiFidelityEvaluation {
    pub unique_architectures_touched: usize,
    pub final_fidelity_queries: usize,
    pub final_query_budget: usize,
    pub final_query_budget_utilization: f64,
    pub total_cost_units: u64,
    pub total_cost_budget_units: u64,
    pub resource_budget_utilization: f64,
    pub resource_unit: String,
    pub intermediate_observation_count: usize,
    pub attempt_count: usize,
    pub unknown_architecture_attempts: usize,
    pub invalid_fidelity_attempts: usize,
    pub non_advanced_fidelity_attempts: usize,
    pub final_query_budget_exhausted_attempts: usize,
    pub resource_budget_exhausted_attempts: usize,
    pub proxy_feature_bank_total_seconds: f64,
    pub touched_top_1: TopKCapture,
    pub touched_top_10: TopKCapture,
    pub touched_top_100: TopKCapture,
    pub final_top_1: TopKCapture,
    pub final_top_10: TopKCapture,
    pub final_top_100: TopKCapture,
    pub best_final_metric: Option<f64>,
    pub global_best_metric: f64,
    pub final_simple_regret: Option<f64>,
    pub final_normalized_regret_percent: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultiFidelityTabularRun {
    pub receipt: MultiFidelityQueryReceipt,
    pub evaluation: MultiFidelityEvaluation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiFidelityFirewallError {
    pub field: String,
    pub message: String,
}
impl MultiFidelityFirewallError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}
impl fmt::Display for MultiFidelityFirewallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}
impl Error for MultiFidelityFirewallError {}

struct OracleParts {
    touched: Vec<String>,
    final_ids: Vec<String>,
    attempts: Vec<FidelityQueryAttempt>,
    total_cost: u64,
}

pub struct MultiFidelityQueryOracle<'a> {
    metric_id: &'a str,
    metric_direction: MetricDirection,
    levels: &'a [FidelityLevel],
    cost_model: FidelityCostModel,
    labels: &'a HashMap<(String, usize), f64>,
    allowed_ids: &'a HashSet<String>,
    max_final_queries: usize,
    max_total_cost: u64,
    current_fidelity: HashMap<String, usize>,
    touched_set: HashSet<String>,
    touched_order: Vec<String>,
    final_set: HashSet<String>,
    final_order: Vec<String>,
    attempts: Vec<FidelityQueryAttempt>,
    total_cost: u64,
}

impl MultiFidelityQueryOracle<'_> {
    pub fn query(
        &mut self,
        architecture_id: &str,
        fidelity_index: usize,
    ) -> Result<FidelityQueryObservation, FidelityQueryError> {
        let attempt_index = self.attempts.len();
        if !self.allowed_ids.contains(architecture_id) {
            self.fail(attempt_index, architecture_id, Some(fidelity_index), FidelityQueryAttemptStatus::UnknownArchitecture);
            return Err(FidelityQueryError::UnknownArchitecture);
        }
        if fidelity_index >= self.levels.len() {
            self.fail(attempt_index, architecture_id, Some(fidelity_index), FidelityQueryAttemptStatus::InvalidFidelity);
            return Err(FidelityQueryError::InvalidFidelity);
        }

        let previous = self.current_fidelity.get(architecture_id).copied();
        if previous.is_some_and(|index| fidelity_index <= index) {
            self.fail(attempt_index, architecture_id, Some(fidelity_index), FidelityQueryAttemptStatus::FidelityNotAdvanced);
            return Err(FidelityQueryError::FidelityNotAdvanced);
        }

        let final_index = self.levels.len() - 1;
        let is_final = fidelity_index == final_index;
        if is_final && self.final_order.len() >= self.max_final_queries {
            self.fail(attempt_index, architecture_id, Some(fidelity_index), FidelityQueryAttemptStatus::FinalQueryBudgetExhausted);
            return Err(FidelityQueryError::FinalQueryBudgetExhausted);
        }

        let level_cost = self.levels[fidelity_index].cumulative_cost_units;
        let incremental_cost = match (self.cost_model, previous) {
            (_, None) => level_cost,
            (FidelityCostModel::ResumablePrefix, Some(previous_index)) => {
                level_cost - self.levels[previous_index].cumulative_cost_units
            }
            (FidelityCostModel::IndependentRuns, Some(_)) => level_cost,
        };
        let Some(next_total) = self.total_cost.checked_add(incremental_cost) else {
            self.fail(attempt_index, architecture_id, Some(fidelity_index), FidelityQueryAttemptStatus::ResourceBudgetExhausted);
            return Err(FidelityQueryError::ResourceBudgetExhausted);
        };
        if next_total > self.max_total_cost {
            self.fail(attempt_index, architecture_id, Some(fidelity_index), FidelityQueryAttemptStatus::ResourceBudgetExhausted);
            return Err(FidelityQueryError::ResourceBudgetExhausted);
        }

        let Some(value) = self.labels.get(&(architecture_id.to_owned(), fidelity_index)).copied() else {
            self.fail(attempt_index, architecture_id, Some(fidelity_index), FidelityQueryAttemptStatus::InvalidFidelity);
            return Err(FidelityQueryError::InvalidFidelity);
        };
        let successful_index = self.attempts
            .iter()
            .filter(|attempt| attempt.status == FidelityQueryAttemptStatus::Revealed)
            .count();
        let observation_sha256 = digest_observation(
            self.metric_id,
            architecture_id,
            fidelity_index,
            value,
        );

        self.current_fidelity.insert(architecture_id.to_owned(), fidelity_index);
        if self.touched_set.insert(architecture_id.to_owned()) {
            self.touched_order.push(architecture_id.to_owned());
        }
        if is_final && self.final_set.insert(architecture_id.to_owned()) {
            self.final_order.push(architecture_id.to_owned());
        }
        self.total_cost = next_total;
        self.attempts.push(FidelityQueryAttempt {
            attempt_index,
            architecture_id: architecture_id.to_owned(),
            fidelity_index: Some(fidelity_index),
            status: FidelityQueryAttemptStatus::Revealed,
            incremental_cost_units: incremental_cost,
            total_cost_units_after_attempt: self.total_cost,
            successful_observation_index: Some(successful_index),
            observation_sha256: Some(observation_sha256),
        });

        Ok(FidelityQueryObservation {
            architecture_id: architecture_id.to_owned(),
            metric_id: self.metric_id.to_owned(),
            metric_direction: self.metric_direction,
            fidelity_id: self.levels[fidelity_index].id.clone(),
            fidelity_index,
            value,
            incremental_cost_units: incremental_cost,
            total_cost_units: self.total_cost,
            remaining_cost_units: self.max_total_cost - self.total_cost,
            is_final_fidelity: is_final,
        })
    }

    fn fail(
        &mut self,
        attempt_index: usize,
        architecture_id: &str,
        fidelity_index: Option<usize>,
        status: FidelityQueryAttemptStatus,
    ) {
        self.attempts.push(FidelityQueryAttempt {
            attempt_index,
            architecture_id: architecture_id.to_owned(),
            fidelity_index,
            status,
            incremental_cost_units: 0,
            total_cost_units_after_attempt: self.total_cost,
            successful_observation_index: None,
            observation_sha256: None,
        });
    }

    fn finish(self) -> OracleParts {
        OracleParts {
            touched: self.touched_order,
            final_ids: self.final_order,
            attempts: self.attempts,
            total_cost: self.total_cost,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalMultiFidelityImport {
    schema: String,
    version: String,
    manifest: MultiFidelityManifest,
    architectures: Vec<ArchitectureFeatures>,
    target_labels: Vec<FidelityTargetLabel>,
}

pub struct MultiFidelityTabularBenchmark {
    manifest: MultiFidelityManifest,
    features: Vec<ArchitectureFeatures>,
    labels: HashMap<(String, usize), f64>,
    allowed_ids: HashSet<String>,
    true_top_k_ids: Vec<String>,
    global_best_metric: f64,
    proxy_time_total_seconds: f64,
    canonical_import_sha256: String,
}

impl MultiFidelityTabularBenchmark {
    /// Strict canonical JSON reader for pinned multi-fidelity benchmark imports.
    pub fn from_canonical_json(source: &str) -> Result<Self, MultiFidelityFirewallError> {
        let imported: CanonicalMultiFidelityImport = serde_json::from_str(source)
            .map_err(|error| MultiFidelityFirewallError::new("$", format!("invalid canonical multi-fidelity import: {error}")))?;
        if imported.schema != CANONICAL_IMPORT_SCHEMA {
            return Err(MultiFidelityFirewallError::new("schema", "wrong multi-fidelity canonical import schema"));
        }
        if imported.version != CANONICAL_IMPORT_VERSION {
            return Err(MultiFidelityFirewallError::new("version", "wrong multi-fidelity canonical import version"));
        }
        Self::new(imported.manifest, imported.architectures, imported.target_labels)
    }

    pub fn new(
        manifest: MultiFidelityManifest,
        features: Vec<ArchitectureFeatures>,
        target_labels: Vec<FidelityTargetLabel>,
    ) -> Result<Self, MultiFidelityFirewallError> {
        validate_manifest(&manifest)?;
        if features.is_empty() {
            return Err(MultiFidelityFirewallError::new("features", "at least one architecture is required"));
        }
        if manifest.base.max_unique_queries > features.len() {
            return Err(MultiFidelityFirewallError::new("base.max_unique_queries", "cannot exceed architecture count"));
        }
        if manifest.base.top_k < 100.min(features.len()) || manifest.base.top_k > features.len() {
            return Err(MultiFidelityFirewallError::new("base.top_k", "must be in min(100, architecture_count)..=architecture_count"));
        }

        let proxy_keys: BTreeSet<String> = features[0].proxy_scores.keys().cloned().collect();
        if proxy_keys.is_empty() {
            return Err(MultiFidelityFirewallError::new("features[0].proxy_scores", "at least one proxy is required"));
        }
        let mut allowed_ids = HashSet::with_capacity(features.len());
        let mut proxy_time_total_seconds = 0.0;
        for (index, feature) in features.iter().enumerate() {
            let field = format!("features[{index}]");
            if feature.architecture_id.trim().is_empty() || !allowed_ids.insert(feature.architecture_id.clone()) {
                return Err(MultiFidelityFirewallError::new(format!("{field}.architecture_id"), "must be non-empty and unique"));
            }
            if feature.topology_encoding.is_empty() {
                return Err(MultiFidelityFirewallError::new(format!("{field}.topology_encoding"), "must be non-empty"));
            }
            let score_keys: BTreeSet<String> = feature.proxy_scores.keys().cloned().collect();
            let time_keys: BTreeSet<String> = feature.proxy_times_seconds.keys().cloned().collect();
            if score_keys != proxy_keys || time_keys != proxy_keys {
                return Err(MultiFidelityFirewallError::new(format!("{field}.proxy_scores"), "proxy score/runtime keys must match the shared schema"));
            }
            for (proxy_id, score) in &feature.proxy_scores {
                if proxy_id.trim().is_empty() || proxy_id == &manifest.base.metric_id || !score.is_finite() {
                    return Err(MultiFidelityFirewallError::new(format!("{field}.proxy_scores.{proxy_id}"), "proxy ID must be valid and score finite"));
                }
                let seconds = feature.proxy_times_seconds[proxy_id];
                if !seconds.is_finite() || seconds < 0.0 {
                    return Err(MultiFidelityFirewallError::new(format!("{field}.proxy_times_seconds.{proxy_id}"), "runtime must be finite and non-negative"));
                }
                proxy_time_total_seconds += seconds;
            }
        }
        if !proxy_time_total_seconds.is_finite() {
            return Err(MultiFidelityFirewallError::new("proxy_times_seconds", "total proxy runtime overflow"));
        }

        let expected = features.len().checked_mul(manifest.fidelity_levels.len())
            .ok_or_else(|| MultiFidelityFirewallError::new("target_labels", "expected label count overflow"))?;
        if target_labels.len() != expected {
            return Err(MultiFidelityFirewallError::new("target_labels", format!("expected exactly {expected} candidate/fidelity labels")));
        }
        let mut labels = HashMap::with_capacity(expected);
        for (index, label) in target_labels.into_iter().enumerate() {
            if !allowed_ids.contains(&label.architecture_id) {
                return Err(MultiFidelityFirewallError::new(format!("target_labels[{index}].architecture_id"), "unknown architecture"));
            }
            if label.fidelity_index >= manifest.fidelity_levels.len() {
                return Err(MultiFidelityFirewallError::new(format!("target_labels[{index}].fidelity_index"), "outside fidelity ladder"));
            }
            if !label.value.is_finite() {
                return Err(MultiFidelityFirewallError::new(format!("target_labels[{index}].value"), "must be finite"));
            }
            let key = (label.architecture_id, label.fidelity_index);
            if labels.insert(key, label.value).is_some() {
                return Err(MultiFidelityFirewallError::new(format!("target_labels[{index}]"), "duplicate candidate/fidelity label"));
            }
        }

        let final_index = manifest.fidelity_levels.len() - 1;
        let mut ranked: Vec<(String, f64)> = features.iter().map(|feature| {
            let key = (feature.architecture_id.clone(), final_index);
            (feature.architecture_id.clone(), labels[&key])
        }).collect();
        ranked.sort_by(|(left_id, left), (right_id, right)| {
            let order = match manifest.base.metric_direction {
                MetricDirection::Maximize => right.total_cmp(left),
                MetricDirection::Minimize => left.total_cmp(right),
            };
            order.then_with(|| left_id.cmp(right_id))
        });
        let true_top_k_ids = ranked.iter().take(manifest.base.top_k).map(|(id, _)| id.clone()).collect();
        let global_best_metric = ranked[0].1;
        let canonical_import_sha256 = digest_canonical_import(&features, &labels);

        Ok(Self {
            manifest,
            features,
            labels,
            allowed_ids,
            true_top_k_ids,
            global_best_metric,
            proxy_time_total_seconds,
            canonical_import_sha256,
        })
    }

    /// Consumes the benchmark object for one policy invocation. The callback receives only
    /// features and the narrow query capability; final labels/oracle ranking remain private.
    pub fn run_search<P>(self, policy: P) -> MultiFidelityTabularRun
    where
        P: FnOnce(&[ArchitectureFeatures], &mut MultiFidelityQueryOracle<'_>),
    {
        let mut oracle = MultiFidelityQueryOracle {
            metric_id: &self.manifest.base.metric_id,
            metric_direction: self.manifest.base.metric_direction,
            levels: &self.manifest.fidelity_levels,
            cost_model: self.manifest.fidelity_cost_model,
            labels: &self.labels,
            allowed_ids: &self.allowed_ids,
            max_final_queries: self.manifest.base.max_unique_queries,
            max_total_cost: self.manifest.max_total_cost_units,
            current_fidelity: HashMap::new(),
            touched_set: HashSet::new(),
            touched_order: Vec::new(),
            final_set: HashSet::new(),
            final_order: Vec::new(),
            attempts: Vec::new(),
            total_cost: 0,
        };
        policy(&self.features, &mut oracle);
        let parts = oracle.finish();
        let receipt = self.make_receipt(parts);
        let evaluation = self.evaluate(&receipt);
        MultiFidelityTabularRun { receipt, evaluation }
    }

    fn make_receipt(&self, parts: OracleParts) -> MultiFidelityQueryReceipt {
        let query_trace_sha256 = digest_query_trace(&self.manifest, &self.canonical_import_sha256, &parts);
        MultiFidelityQueryReceipt {
            benchmark_id: self.manifest.base.benchmark_id.clone(),
            metric_id: self.manifest.base.metric_id.clone(),
            metric_direction: self.manifest.base.metric_direction,
            split_id: self.manifest.base.split_id.clone(),
            proxy_calibration_split_id: self.manifest.base.proxy_calibration_split_id.clone(),
            proxy_calibration_sha256: self.manifest.base.proxy_calibration_sha256.clone(),
            source_artifact_sha256: self.manifest.base.artifact_sha256.clone(),
            canonical_import_sha256: self.canonical_import_sha256.clone(),
            fidelity_levels: self.manifest.fidelity_levels.clone(),
            fidelity_cost_model: self.manifest.fidelity_cost_model,
            resource_unit: self.manifest.resource_unit.clone(),
            max_final_queries: self.manifest.base.max_unique_queries,
            max_total_cost_units: self.manifest.max_total_cost_units,
            unique_architectures_touched: parts.touched.len(),
            final_fidelity_queries: parts.final_ids.len(),
            total_cost_units: parts.total_cost,
            touched_architecture_ids: parts.touched,
            final_fidelity_architecture_ids: parts.final_ids,
            attempts: parts.attempts,
            query_trace_sha256,
        }
    }

    fn evaluate(&self, receipt: &MultiFidelityQueryReceipt) -> MultiFidelityEvaluation {
        let touched: HashSet<&str> = receipt.touched_architecture_ids.iter().map(String::as_str).collect();
        let confirmed: HashSet<&str> = receipt.final_fidelity_architecture_ids.iter().map(String::as_str).collect();
        let touched_top_1 = capture_metric(1, &touched, &self.true_top_k_ids);
        let touched_top_10 = capture_metric(10, &touched, &self.true_top_k_ids);
        let touched_top_100 = capture_metric(100, &touched, &self.true_top_k_ids);
        let final_top_1 = capture_metric(1, &confirmed, &self.true_top_k_ids);
        let final_top_10 = capture_metric(10, &confirmed, &self.true_top_k_ids);
        let final_top_100 = capture_metric(100, &confirmed, &self.true_top_k_ids);
        let final_index = self.manifest.fidelity_levels.len() - 1;
        let best_final_metric = receipt.final_fidelity_architecture_ids.iter()
            .filter_map(|id| self.labels.get(&(id.clone(), final_index)).copied())
            .reduce(|best, value| match self.manifest.base.metric_direction {
                MetricDirection::Maximize => best.max(value),
                MetricDirection::Minimize => best.min(value),
            });
        let final_simple_regret = best_final_metric.map(|best| match self.manifest.base.metric_direction {
            MetricDirection::Maximize => self.global_best_metric - best,
            MetricDirection::Minimize => best - self.global_best_metric,
        });
        let final_normalized_regret_percent = final_simple_regret.and_then(|regret| {
            let denominator = self.global_best_metric.abs();
            if denominator == 0.0 { None } else { Some(regret / denominator * 100.0) }
        });
        let count = |status| receipt.attempts.iter().filter(|attempt| attempt.status == status).count();
        let final_index = self.manifest.fidelity_levels.len() - 1;
        let intermediate_observation_count = receipt.attempts.iter().filter(|attempt| {
            attempt.status == FidelityQueryAttemptStatus::Revealed
                && attempt.fidelity_index.is_some_and(|index| index < final_index)
        }).count();
        MultiFidelityEvaluation {
            unique_architectures_touched: receipt.unique_architectures_touched,
            final_fidelity_queries: receipt.final_fidelity_queries,
            final_query_budget: receipt.max_final_queries,
            final_query_budget_utilization: receipt.final_fidelity_queries as f64 / receipt.max_final_queries as f64,
            total_cost_units: receipt.total_cost_units,
            total_cost_budget_units: receipt.max_total_cost_units,
            resource_budget_utilization: receipt.total_cost_units as f64 / receipt.max_total_cost_units as f64,
            resource_unit: receipt.resource_unit.clone(),
            intermediate_observation_count,
            attempt_count: receipt.attempts.len(),
            unknown_architecture_attempts: count(FidelityQueryAttemptStatus::UnknownArchitecture),
            invalid_fidelity_attempts: count(FidelityQueryAttemptStatus::InvalidFidelity),
            non_advanced_fidelity_attempts: count(FidelityQueryAttemptStatus::FidelityNotAdvanced),
            final_query_budget_exhausted_attempts: count(FidelityQueryAttemptStatus::FinalQueryBudgetExhausted),
            resource_budget_exhausted_attempts: count(FidelityQueryAttemptStatus::ResourceBudgetExhausted),
            proxy_feature_bank_total_seconds: self.proxy_time_total_seconds,
            touched_top_1, touched_top_10, touched_top_100,
            final_top_1, final_top_10, final_top_100,
            best_final_metric,
            global_best_metric: self.global_best_metric,
            final_simple_regret,
            final_normalized_regret_percent,
        }
    }
}

fn validate_manifest(manifest: &MultiFidelityManifest) -> Result<(), MultiFidelityFirewallError> {
    let base = &manifest.base;
    if base.schema != "symthaea.tabular-nas.benchmark" || base.version != "0.1.0" {
        return Err(MultiFidelityFirewallError::new("base.schema", "unsupported benchmark schema/version"));
    }
    for (field, value) in [
        ("base.benchmark_id", base.benchmark_id.as_str()),
        ("base.metric_id", base.metric_id.as_str()),
        ("base.split_id", base.split_id.as_str()),
        ("base.proxy_calibration_split_id", base.proxy_calibration_split_id.as_str()),
        ("base.source_description", base.source_description.as_str()),
        ("resource_unit", manifest.resource_unit.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(MultiFidelityFirewallError::new(field, "must be non-empty"));
        }
    }
    if base.split_id == base.proxy_calibration_split_id {
        return Err(MultiFidelityFirewallError::new("base.proxy_calibration_split_id", "must differ from evaluation split"));
    }
    for (field, digest) in [
        ("base.artifact_sha256", base.artifact_sha256.as_str()),
        ("base.proxy_calibration_sha256", base.proxy_calibration_sha256.as_str()),
    ] {
        if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(MultiFidelityFirewallError::new(field, "must be 64 hexadecimal characters"));
        }
    }
    if base.max_unique_queries == 0 || manifest.max_total_cost_units == 0 {
        return Err(MultiFidelityFirewallError::new("budget", "final query and resource budgets must be positive"));
    }
    if manifest.fidelity_levels.len() < 2 {
        return Err(MultiFidelityFirewallError::new("fidelity_levels", "at least two levels are required"));
    }
    let mut seen = HashSet::new();
    let mut previous = 0u64;
    for (index, level) in manifest.fidelity_levels.iter().enumerate() {
        if level.id.trim().is_empty() || !seen.insert(level.id.as_str()) {
            return Err(MultiFidelityFirewallError::new(format!("fidelity_levels[{index}].id"), "must be non-empty and unique"));
        }
        if level.cumulative_cost_units == 0 || level.cumulative_cost_units <= previous {
            return Err(MultiFidelityFirewallError::new(format!("fidelity_levels[{index}].cumulative_cost_units"), "must be positive and strictly increasing"));
        }
        previous = level.cumulative_cost_units;
    }
    Ok(())
}

fn capture_metric(k: usize, queried: &HashSet<&str>, top_k_ids: &[String]) -> TopKCapture {
    let denominator = k.min(top_k_ids.len());
    let captured = top_k_ids.iter().take(denominator).filter(|id| queried.contains(id.as_str())).count();
    TopKCapture {
        k,
        true_top_k_count: denominator,
        queried_top_k_count: captured,
        capture_rate: if denominator == 0 { 0.0 } else { captured as f64 / denominator as f64 },
    }
}

fn digest_canonical_import(
    features: &[ArchitectureFeatures],
    labels: &HashMap<(String, usize), f64>,
) -> String {
    let mut h = Sha256::new();
    h.update(IMPORT_DOMAIN);
    let mut ordered_features: Vec<&ArchitectureFeatures> = features.iter().collect();
    ordered_features.sort_by(|a, b| a.architecture_id.cmp(&b.architecture_id));
    for feature in ordered_features {
        update_string(&mut h, &feature.architecture_id);
        h.update((feature.topology_encoding.len() as u64).to_be_bytes());
        h.update(&feature.topology_encoding);
        for (name, score) in &feature.proxy_scores {
            update_string(&mut h, name);
            h.update(score.to_bits().to_be_bytes());
        }
        for (name, seconds) in &feature.proxy_times_seconds {
            update_string(&mut h, name);
            h.update(seconds.to_bits().to_be_bytes());
        }
    }
    let mut ordered_labels: Vec<(&(String, usize), &f64)> = labels.iter().collect();
    ordered_labels.sort_by(|(a, _), (b, _)| a.cmp(b));
    for ((id, index), value) in ordered_labels {
        update_string(&mut h, id);
        h.update((*index as u64).to_be_bytes());
        h.update(value.to_bits().to_be_bytes());
    }
    hex_digest(h.finalize().as_slice())
}

fn digest_observation(metric: &str, id: &str, index: usize, value: f64) -> String {
    let mut h = Sha256::new();
    h.update(b"symthaea.tabular-nas.fidelity-observation.v0.1");
    update_string(&mut h, metric);
    update_string(&mut h, id);
    h.update((index as u64).to_be_bytes());
    h.update(value.to_bits().to_be_bytes());
    hex_digest(h.finalize().as_slice())
}

fn digest_query_trace(manifest: &MultiFidelityManifest, import_sha: &str, parts: &OracleParts) -> String {
    let mut h = Sha256::new();
    h.update(RECEIPT_DOMAIN);
    update_string(&mut h, &manifest.base.benchmark_id);
    update_string(&mut h, &manifest.base.metric_id);
    h.update([match manifest.base.metric_direction { MetricDirection::Maximize => 0, MetricDirection::Minimize => 1 }]);
    update_string(&mut h, &manifest.base.split_id);
    update_string(&mut h, &manifest.base.proxy_calibration_split_id);
    update_string(&mut h, &manifest.base.proxy_calibration_sha256);
    update_string(&mut h, &manifest.base.artifact_sha256);
    update_string(&mut h, import_sha);
    h.update((manifest.base.max_unique_queries as u64).to_be_bytes());
    h.update(manifest.max_total_cost_units.to_be_bytes());
    h.update([match manifest.fidelity_cost_model { FidelityCostModel::ResumablePrefix => 0, FidelityCostModel::IndependentRuns => 1 }]);
    update_string(&mut h, &manifest.resource_unit);
    for level in &manifest.fidelity_levels {
        update_string(&mut h, &level.id);
        h.update(level.cumulative_cost_units.to_be_bytes());
    }
    for id in &parts.touched { update_string(&mut h, id); }
    for id in &parts.final_ids { update_string(&mut h, id); }
    for attempt in &parts.attempts {
        h.update((attempt.attempt_index as u64).to_be_bytes());
        update_string(&mut h, &attempt.architecture_id);
        match attempt.fidelity_index {
            Some(index) => { h.update([1]); h.update((index as u64).to_be_bytes()); }
            None => h.update([0]),
        }
        h.update([status_byte(attempt.status)]);
        h.update(attempt.incremental_cost_units.to_be_bytes());
        h.update(attempt.total_cost_units_after_attempt.to_be_bytes());
        match &attempt.observation_sha256 {
            Some(value) => { h.update([1]); update_string(&mut h, value); }
            None => h.update([0]),
        }
    }
    h.update(parts.total_cost.to_be_bytes());
    hex_digest(h.finalize().as_slice())
}

fn status_byte(status: FidelityQueryAttemptStatus) -> u8 {
    match status {
        FidelityQueryAttemptStatus::Revealed => 0,
        FidelityQueryAttemptStatus::UnknownArchitecture => 1,
        FidelityQueryAttemptStatus::InvalidFidelity => 2,
        FidelityQueryAttemptStatus::FidelityNotAdvanced => 3,
        FidelityQueryAttemptStatus::FinalQueryBudgetExhausted => 4,
        FidelityQueryAttemptStatus::ResourceBudgetExhausted => 5,
    }
}

fn update_string(h: &mut Sha256, value: &str) {
    h.update((value.len() as u64).to_be_bytes());
    h.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8]) -> String {
    const DIGITS: &[u8] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes.iter().copied() {
        out.push(DIGITS[(b >> 4) as usize] as char);
        out.push(DIGITS[(b & 15) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> TabularBenchmarkManifest {
        TabularBenchmarkManifest {
            schema: "symthaea.tabular-nas.benchmark".into(),
            version: "0.1.0".into(),
            benchmark_id: "multi-fidelity-synthetic".into(),
            metric_id: "validation_accuracy".into(),
            metric_direction: MetricDirection::Maximize,
            split_id: "challenge-a".into(),
            proxy_calibration_split_id: "development-b".into(),
            proxy_calibration_sha256: "b".repeat(64),
            source_description: "synthetic test fixture".into(),
            artifact_sha256: "a".repeat(64),
            max_unique_queries: 3,
            top_k: 12,
        }
    }

    fn levels() -> Vec<FidelityLevel> {
        vec![
            FidelityLevel { id: "epoch-5".into(), cumulative_cost_units: 5 },
            FidelityLevel { id: "epoch-15".into(), cumulative_cost_units: 15 },
            FidelityLevel { id: "full-200".into(), cumulative_cost_units: 200 },
        ]
    }

    fn manifest(model: FidelityCostModel, budget: u64) -> MultiFidelityManifest {
        MultiFidelityManifest {
            base: base(),
            fidelity_levels: levels(),
            fidelity_cost_model: model,
            resource_unit: "epoch_equivalent".into(),
            max_total_cost_units: budget,
        }
    }

    fn features(count: usize) -> Vec<ArchitectureFeatures> {
        (0..count).map(|index| {
            let mut scores = BTreeMap::new();
            scores.insert("params".into(), index as f64);
            scores.insert("synflow".into(), (count - index) as f64);
            let mut times = BTreeMap::new();
            times.insert("params".into(), 0.01);
            times.insert("synflow".into(), 0.02);
            ArchitectureFeatures {
                architecture_id: format!("arch-{index:03}"),
                proxy_scores: scores,
                proxy_times_seconds: times,
                topology_encoding: vec![index as u8, 1, 2],
            }
        }).collect()
    }

    fn labels(count: usize) -> Vec<FidelityTargetLabel> {
        (0..count).flat_map(|index| {
            let id = format!("arch-{index:03}");
            [
                FidelityTargetLabel { architecture_id: id.clone(), fidelity_index: 0, value: 40.0 + index as f64 },
                FidelityTargetLabel { architecture_id: id.clone(), fidelity_index: 1, value: 60.0 + index as f64 },
                FidelityTargetLabel { architecture_id: id, fidelity_index: 2, value: 80.0 + index as f64 },
            ]
        }).collect()
    }

    fn make_benchmark(model: FidelityCostModel, budget: u64) -> MultiFidelityTabularBenchmark {
        MultiFidelityTabularBenchmark::new(manifest(model, budget), features(12), labels(12)).unwrap()
    }

    #[test]
    fn resumable_prefix_charges_only_incremental_cost() {
        let benchmark = make_benchmark(FidelityCostModel::ResumablePrefix, 220);
        let run = benchmark.run_search(|_, oracle| {
            assert_eq!(oracle.query("arch-000", 0).unwrap().incremental_cost_units, 5);
            assert_eq!(oracle.query("arch-000", 1).unwrap().incremental_cost_units, 10);
            assert_eq!(oracle.query("arch-000", 2).unwrap().incremental_cost_units, 185);
        });
        assert_eq!(run.receipt.total_cost_units, 200);
        assert_eq!(run.evaluation.final_fidelity_queries, 1);
        assert_eq!(run.evaluation.intermediate_observation_count, 2);
    }

    #[test]
    fn independent_restart_charges_full_cost_for_each_rung() {
        let benchmark = make_benchmark(FidelityCostModel::IndependentRuns, 220);
        let run = benchmark.run_search(|_, oracle| {
            oracle.query("arch-000", 0).unwrap();
            oracle.query("arch-000", 1).unwrap();
            assert_eq!(oracle.query("arch-000", 2).unwrap().incremental_cost_units, 200);
        });
        assert_eq!(run.receipt.total_cost_units, 220);
    }

    #[test]
    fn partial_queries_count_as_touched_not_final_confirmed() {
        let benchmark = make_benchmark(FidelityCostModel::ResumablePrefix, 50);
        let run = benchmark.run_search(|_, oracle| {
            oracle.query("arch-011", 0).unwrap();
            oracle.query("arch-010", 0).unwrap();
        });
        assert_eq!(run.evaluation.unique_architectures_touched, 2);
        assert_eq!(run.evaluation.final_fidelity_queries, 0);
        assert_eq!(run.evaluation.touched_top_1.queried_top_k_count, 1);
        assert_eq!(run.evaluation.final_top_1.queried_top_k_count, 0);
        assert_eq!(run.evaluation.best_final_metric, None);
    }

    #[test]
    fn final_capture_counts_only_final_fidelity_confirmations() {
        let benchmark = make_benchmark(FidelityCostModel::ResumablePrefix, 500);
        let run = benchmark.run_search(|_, oracle| {
            oracle.query("arch-011", 0).unwrap();
            oracle.query("arch-010", 0).unwrap();
            oracle.query("arch-011", 2).unwrap();
        });
        assert_eq!(run.evaluation.unique_architectures_touched, 2);
        assert_eq!(run.evaluation.final_fidelity_queries, 1);
        assert_eq!(run.evaluation.touched_top_10.queried_top_k_count, 2);
        assert_eq!(run.evaluation.final_top_10.queried_top_k_count, 1);
        assert_eq!(run.evaluation.best_final_metric, Some(91.0));
        assert_eq!(run.evaluation.final_simple_regret, Some(0.0));
    }

    #[test]
    fn final_query_cap_and_resource_cap_are_independent() {
        let mut limited = manifest(FidelityCostModel::ResumablePrefix, 1000);
        limited.base.max_unique_queries = 1;
        let bench = MultiFidelityTabularBenchmark::new(limited, features(12), labels(12)).unwrap();
        let run = bench.run_search(|_, oracle| {
            oracle.query("arch-000", 2).unwrap();
            assert_eq!(oracle.query("arch-001", 2), Err(FidelityQueryError::FinalQueryBudgetExhausted));
        });
        assert_eq!(run.receipt.final_fidelity_queries, 1);
        assert_eq!(run.evaluation.final_query_budget_exhausted_attempts, 1);

        let bench = make_benchmark(FidelityCostModel::ResumablePrefix, 5);
        let run = bench.run_search(|_, oracle| {
            oracle.query("arch-000", 0).unwrap();
            assert_eq!(oracle.query("arch-001", 0), Err(FidelityQueryError::ResourceBudgetExhausted));
        });
        assert_eq!(run.receipt.total_cost_units, 5);
        assert_eq!(run.evaluation.resource_budget_exhausted_attempts, 1);
    }

    #[test]
    fn duplicate_or_downgrade_fidelity_requests_do_not_charge_cost() {
        let benchmark = make_benchmark(FidelityCostModel::ResumablePrefix, 500);
        let run = benchmark.run_search(|_, oracle| {
            oracle.query("arch-000", 1).unwrap();
            assert_eq!(oracle.query("arch-000", 0), Err(FidelityQueryError::FidelityNotAdvanced));
            assert_eq!(oracle.query("arch-000", 1), Err(FidelityQueryError::FidelityNotAdvanced));
        });
        assert_eq!(run.receipt.total_cost_units, 15);
        assert_eq!(run.evaluation.non_advanced_fidelity_attempts, 2);
    }

    #[test]
    fn unknown_candidate_and_invalid_fidelity_reveal_nothing() {
        let benchmark = make_benchmark(FidelityCostModel::ResumablePrefix, 500);
        let run = benchmark.run_search(|_, oracle| {
            assert_eq!(oracle.query("unknown", 0), Err(FidelityQueryError::UnknownArchitecture));
            assert_eq!(oracle.query("arch-000", 20), Err(FidelityQueryError::InvalidFidelity));
        });
        assert_eq!(run.receipt.total_cost_units, 0);
        assert!(run.receipt.attempts.iter().all(|attempt| attempt.observation_sha256.is_none()));
    }

    #[test]
    fn constructor_rejects_duplicate_and_missing_candidate_fidelity_labels() {
        let mut duplicate = labels(12);
        duplicate.push(duplicate[0].clone());
        assert!(MultiFidelityTabularBenchmark::new(manifest(FidelityCostModel::ResumablePrefix, 500), features(12), duplicate).is_err());

        let mut missing = labels(12);
        missing.pop();
        assert!(MultiFidelityTabularBenchmark::new(manifest(FidelityCostModel::ResumablePrefix, 500), features(12), missing).is_err());
    }

    #[test]
    fn constructor_rejects_invalid_fidelity_ladder() {
        let mut invalid = manifest(FidelityCostModel::ResumablePrefix, 500);
        invalid.fidelity_levels[1].cumulative_cost_units = 5;
        assert!(MultiFidelityTabularBenchmark::new(invalid, features(12), labels(12)).is_err());
        let mut invalid = manifest(FidelityCostModel::ResumablePrefix, 500);
        invalid.fidelity_levels[1].id = invalid.fidelity_levels[0].id.clone();
        assert!(MultiFidelityTabularBenchmark::new(invalid, features(12), labels(12)).is_err());
    }

    #[test]
    fn canonical_json_import_is_strict_and_round_trips_the_fidelity_contract() {
        let imported = CanonicalMultiFidelityImport {
            schema: CANONICAL_IMPORT_SCHEMA.into(),
            version: CANONICAL_IMPORT_VERSION.into(),
            manifest: manifest(FidelityCostModel::ResumablePrefix, 500),
            architectures: features(12),
            target_labels: labels(12),
        };
        let json = serde_json::to_string(&imported).unwrap();
        let benchmark = MultiFidelityTabularBenchmark::from_canonical_json(&json).unwrap();
        let run = benchmark.run_search(|_, oracle| {
            oracle.query("arch-011", 2).unwrap();
        });
        assert_eq!(run.receipt.canonical_import_sha256.len(), 64);

        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["target_label_leak"] = serde_json::json!({"arch-011": 91.0});
        assert!(MultiFidelityTabularBenchmark::from_canonical_json(&value.to_string()).is_err());
    }

    #[test]
    fn receipt_binds_fidelity_cost_model_resource_unit_and_attempts() {
        let benchmark = make_benchmark(FidelityCostModel::ResumablePrefix, 500);
        let run = benchmark.run_search(|_, oracle| {
            oracle.query("arch-005", 0).unwrap();
            oracle.query("arch-005", 2).unwrap();
        });
        assert_eq!(run.receipt.resource_unit, "epoch_equivalent");
        assert_eq!(run.receipt.fidelity_cost_model, FidelityCostModel::ResumablePrefix);
        assert_eq!(run.receipt.attempts[0].incremental_cost_units, 5);
        assert_eq!(run.receipt.attempts[1].incremental_cost_units, 195);
        assert_eq!(run.receipt.query_trace_sha256.len(), 64);
    }

    #[test]
    fn minimize_metric_uses_correct_final_regret() {
        let mut m = manifest(FidelityCostModel::ResumablePrefix, 500);
        m.base.metric_direction = MetricDirection::Minimize;
        let mut target_labels = labels(12);
        for label in &mut target_labels {
            if label.fidelity_index == 2 {
                label.value = 100.0 - label.value;
            }
        }
        let benchmark = MultiFidelityTabularBenchmark::new(m, features(12), target_labels).unwrap();
        let run = benchmark.run_search(|_, oracle| {
            oracle.query("arch-008", 2).unwrap();
        });
        assert_eq!(run.evaluation.global_best_metric, 9.0);
        assert_eq!(run.evaluation.best_final_metric, Some(12.0));
        assert_eq!(run.evaluation.final_simple_regret, Some(3.0));
    }

    #[test]
    fn canonical_import_digest_is_invariant_to_input_order() {
        let forward = MultiFidelityTabularBenchmark::new(manifest(FidelityCostModel::ResumablePrefix, 500), features(12), labels(12))
            .unwrap().canonical_import_sha256;
        let mut reversed_features = features(12);
        reversed_features.reverse();
        let mut reversed_labels = labels(12);
        reversed_labels.reverse();
        let reverse = MultiFidelityTabularBenchmark::new(manifest(FidelityCostModel::ResumablePrefix, 500), reversed_features, reversed_labels)
            .unwrap().canonical_import_sha256;
        assert_eq!(forward, reverse);
    }

    #[test]
    fn predicted_but_unqueried_architectures_receive_no_capture_credit() {
        let benchmark = make_benchmark(FidelityCostModel::ResumablePrefix, 500);
        let run = benchmark.run_search(|features, _| {
            let _prediction = features.last().unwrap().architecture_id.clone();
        });
        assert_eq!(run.evaluation.unique_architectures_touched, 0);
        assert_eq!(run.evaluation.touched_top_100.capture_rate, 0.0);
        assert_eq!(run.evaluation.final_top_100.capture_rate, 0.0);
    }

    #[test]
    fn zero_query_run_has_finite_cost_metrics() {
        let benchmark = make_benchmark(FidelityCostModel::ResumablePrefix, 500);
        let run = benchmark.run_search(|_, _| {});
        assert_eq!(run.evaluation.resource_budget_utilization, 0.0);
        assert_eq!(run.evaluation.final_query_budget_utilization, 0.0);
    }
}
