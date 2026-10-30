//! Query-firewalled tabular NAS benchmark harness.
//!
//! Search policies receive architecture/proxy features and a narrow target-metric query
//! capability. The full target-label table, oracle top-K sets and scoring logic remain private
//! to TabularNasBenchmark and are never passed to the policy callback.
//!
//! This is an API/information-flow boundary for trusted Rust policy code. It is not a sandbox
//! against unsafe code, process-memory inspection, debugger access, or labels deliberately
//! copied into a policy closure by its caller. Production evaluation should run policies in a
//! separate process and send only query requests/results across the boundary.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fmt;

use sha2::{Digest, Sha256};

const BENCHMARK_SCHEMA: &str = "symthaea.tabular-nas.benchmark";
const BENCHMARK_VERSION: &str = "0.1.0";

/// Target metric orientation used for top-K ranking and regret.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricDirection {
    Maximize,
    Minimize,
}

/// Source and evaluation contract for one imported tabular benchmark split.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TabularBenchmarkManifest {
    pub schema: String,
    pub version: String,
    pub benchmark_id: String,
    pub metric_id: String,
    pub metric_direction: MetricDirection,
    /// Evaluation task/split whose target labels remain hidden until queried.
    pub split_id: String,
    /// Separate task/split used to select or calibrate the proxy subset.
    pub proxy_calibration_split_id: String,
    /// Digest of the calibration data and frozen proxy-selection decision.
    pub proxy_calibration_sha256: String,
    pub source_description: String,
    /// SHA-256 of the raw source artifact or canonical import artifact, as documented above.
    pub artifact_sha256: String,
    pub max_unique_queries: usize,
    /// Must cover at least min(100, architecture_count) candidates for P@100 reporting.
    pub top_k: usize,
}

/// Public-to-policy architecture information. It deliberately contains no target metric.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchitectureFeatures {
    pub architecture_id: String,
    /// Precomputed, permitted zero-cost proxy values. Every architecture has the same keys.
    pub proxy_scores: BTreeMap<String, f64>,
    /// Canonical topology encoding, with the encoding version documented by the importer.
    pub topology_encoding: Vec<u8>,
}

/// One target metric record supplied only to the benchmark constructor/evaluator.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetLabel {
    pub architecture_id: String,
    pub value: f64,
}

/// Candidate query failures are visible to the policy, but never reveal target values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryError {
    UnknownArchitecture,
    DuplicateQuery,
    BudgetExhausted,
}

impl fmt::Display for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownArchitecture => write!(f, "unknown architecture ID"),
            Self::DuplicateQuery => write!(f, "architecture was already queried"),
            Self::BudgetExhausted => write!(f, "unique target-query budget exhausted"),
        }
    }
}

impl Error for QueryError {}

/// A revealed observation. The value is returned only for a successful unique query.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryObservation {
    pub architecture_id: String,
    pub metric_id: String,
    pub metric_direction: MetricDirection,
    pub value: f64,
    pub query_index: usize,
    pub remaining_unique_queries: usize,
}

/// Recorded outcome of every query attempt. Raw target values are never copied into the trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryAttemptStatus {
    Revealed,
    UnknownArchitecture,
    Duplicate,
    BudgetExhausted,
}

/// One query attempt in policy order. observation_sha256 binds the returned value without
/// placing that value itself in the receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryAttempt {
    pub attempt_index: usize,
    pub architecture_id: String,
    pub status: QueryAttemptStatus,
    pub query_index: Option<usize>,
    pub observation_sha256: Option<String>,
}

/// Evaluator-owned receipt of the full request trace. It contains no raw target values and no
/// unqueried target labels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryReceipt {
    pub benchmark_id: String,
    pub metric_id: String,
    pub metric_direction: MetricDirection,
    pub split_id: String,
    pub proxy_calibration_split_id: String,
    pub proxy_calibration_sha256: String,
    pub source_artifact_sha256: String,
    pub max_unique_queries: usize,
    pub top_k: usize,
    pub unique_queries: usize,
    pub queried_architecture_ids: Vec<String>,
    pub attempts: Vec<QueryAttempt>,
    pub query_trace_sha256: String,
}

/// Coverage of the true top-k architectures among the distinct architectures actually queried.
#[derive(Debug, Clone, PartialEq)]
pub struct TopKCapture {
    pub k: usize,
    pub true_top_k_count: usize,
    pub queried_top_k_count: usize,
    pub capture_rate: f64,
}

/// Evaluator-only metrics computed after the policy callback has returned.
#[derive(Debug, Clone, PartialEq)]
pub struct TabularNasEvaluation {
    pub unique_queries: usize,
    pub query_budget: usize,
    pub budget_utilization: f64,
    pub attempt_count: usize,
    pub duplicate_attempts: usize,
    pub unknown_architecture_attempts: usize,
    pub over_budget_attempts: usize,
    pub top_1: TopKCapture,
    pub top_10: TopKCapture,
    pub top_100: TopKCapture,
    pub best_queried_metric: Option<f64>,
    pub global_best_metric: f64,
    pub simple_regret: Option<f64>,
    pub queried_target_observations_sha256: String,
}

/// One complete, policy-independent run result.
#[derive(Debug, Clone, PartialEq)]
pub struct TabularNasRun {
    pub receipt: QueryReceipt,
    pub evaluation: TabularNasEvaluation,
}

/// Construction failure. Invalid benchmark tables never produce a query capability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TabularNasError {
    pub field: String,
    pub message: String,
}

impl TabularNasError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for TabularNasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for TabularNasError {}

/// Narrow query capability passed to a search policy. The target table and candidate set are
/// private, and no method exposes them. A unique successful query consumes one unit.
pub struct QueryOracle<'a> {
    metric_id: &'a str,
    metric_direction: MetricDirection,
    labels: &'a HashMap<String, f64>,
    allowed_ids: &'a HashSet<String>,
    max_unique_queries: usize,
    queried_ids: HashSet<String>,
    queried_order: Vec<String>,
    attempts: Vec<QueryAttempt>,
}

impl QueryOracle<'_> {
    /// Query exactly one architecture's target metric. Duplicate and invalid attempts are
    /// retained in the trace but never reveal target values or consume another unique unit.
    pub fn query(&mut self, architecture_id: &str) -> Result<QueryObservation, QueryError> {
        let attempt_index = self.attempts.len();
        if !self.allowed_ids.contains(architecture_id) {
            self.attempts.push(QueryAttempt {
                attempt_index,
                architecture_id: architecture_id.to_owned(),
                status: QueryAttemptStatus::UnknownArchitecture,
                query_index: None,
                observation_sha256: None,
            });
            return Err(QueryError::UnknownArchitecture);
        }
        if self.queried_ids.contains(architecture_id) {
            self.attempts.push(QueryAttempt {
                attempt_index,
                architecture_id: architecture_id.to_owned(),
                status: QueryAttemptStatus::Duplicate,
                query_index: None,
                observation_sha256: None,
            });
            return Err(QueryError::DuplicateQuery);
        }
        if self.queried_order.len() >= self.max_unique_queries {
            self.attempts.push(QueryAttempt {
                attempt_index,
                architecture_id: architecture_id.to_owned(),
                status: QueryAttemptStatus::BudgetExhausted,
                query_index: None,
                observation_sha256: None,
            });
            return Err(QueryError::BudgetExhausted);
        }

        let value = *self.labels.get(architecture_id).ok_or(QueryError::UnknownArchitecture)?;
        let query_index = self.queried_order.len();
        let observation_sha256 = digest_observation(self.metric_id, architecture_id, value);

        self.queried_ids.insert(architecture_id.to_owned());
        self.queried_order.push(architecture_id.to_owned());
        self.attempts.push(QueryAttempt {
            attempt_index,
            architecture_id: architecture_id.to_owned(),
            status: QueryAttemptStatus::Revealed,
            query_index: Some(query_index),
            observation_sha256: Some(observation_sha256),
        });

        Ok(QueryObservation {
            architecture_id: architecture_id.to_owned(),
            metric_id: self.metric_id.to_owned(),
            metric_direction: self.metric_direction,
            value,
            query_index,
            remaining_unique_queries: self.max_unique_queries - self.queried_order.len(),
        })
    }
}

struct QueryOracleReceiptParts {
    queried_order: Vec<String>,
    attempts: Vec<QueryAttempt>,
}

impl QueryOracle<'_> {
    fn finish(self) -> QueryOracleReceiptParts {
        QueryOracleReceiptParts {
            queried_order: self.queried_order,
            attempts: self.attempts,
        }
    }
}

/// Full evaluator. Fields containing labels and the oracle ranking are private. The only
/// policy callback receives feature rows and a QueryOracle capability.
pub struct TabularNasBenchmark {
    manifest: TabularBenchmarkManifest,
    features: Vec<ArchitectureFeatures>,
    labels: HashMap<String, f64>,
    allowed_ids: HashSet<String>,
    true_top_k_ids: Vec<String>,
    global_best_metric: f64,
}

impl TabularNasBenchmark {
    /// Validate and seal a complete tabular benchmark.
    pub fn new(
        manifest: TabularBenchmarkManifest,
        features: Vec<ArchitectureFeatures>,
        target_labels: Vec<TargetLabel>,
    ) -> Result<Self, TabularNasError> {
        validate_manifest(&manifest)?;
        if features.is_empty() {
            return Err(TabularNasError::new("features", "at least one architecture is required"));
        }
        if manifest.max_unique_queries > features.len() {
            return Err(TabularNasError::new(
                "max_unique_queries",
                "cannot exceed the number of architectures",
            ));
        }
        let required_top_k = 100.min(features.len());
        if manifest.top_k < required_top_k || manifest.top_k > features.len() {
            return Err(TabularNasError::new(
                "top_k",
                "must be in min(100, architecture_count)..=architecture_count",
            ));
        }

        let expected_proxy_names: BTreeSet<String> =
            features[0].proxy_scores.keys().cloned().collect();
        if expected_proxy_names.is_empty() {
            return Err(TabularNasError::new(
                "features[0].proxy_scores",
                "at least one declared proxy feature is required",
            ));
        }

        let mut allowed_ids = HashSet::with_capacity(features.len());
        for (index, feature) in features.iter().enumerate() {
            let field = format!("features[{index}]");
            if feature.architecture_id.trim().is_empty() {
                return Err(TabularNasError::new(
                    format!("{field}.architecture_id"),
                    "must be a non-empty string",
                ));
            }
            if !allowed_ids.insert(feature.architecture_id.clone()) {
                return Err(TabularNasError::new(
                    format!("{field}.architecture_id"),
                    format!("duplicate architecture ID {:?}", feature.architecture_id),
                ));
            }
            if feature.topology_encoding.is_empty() {
                return Err(TabularNasError::new(
                    format!("{field}.topology_encoding"),
                    "must contain a versioned, non-empty topology encoding",
                ));
            }
            let proxy_names: BTreeSet<String> =
                feature.proxy_scores.keys().cloned().collect();
            if proxy_names != expected_proxy_names {
                return Err(TabularNasError::new(
                    format!("{field}.proxy_scores"),
                    "every architecture must contain the same proxy feature keys",
                ));
            }
            for (proxy_id, score) in &feature.proxy_scores {
                if proxy_id.trim().is_empty() || proxy_id == &manifest.metric_id {
                    return Err(TabularNasError::new(
                        format!("{field}.proxy_scores"),
                        "proxy IDs must be non-empty and must not reuse the target metric ID",
                    ));
                }
                if !score.is_finite() {
                    return Err(TabularNasError::new(
                        format!("{field}.proxy_scores.{proxy_id}"),
                        "proxy scores must be finite",
                    ));
                }
            }
        }

        if target_labels.len() != features.len() {
            return Err(TabularNasError::new(
                "target_labels",
                "must contain exactly one label for every architecture",
            ));
        }
        let mut labels = HashMap::with_capacity(target_labels.len());
        for (index, target) in target_labels.into_iter().enumerate() {
            let field = format!("target_labels[{index}]");
            if !allowed_ids.contains(&target.architecture_id) {
                return Err(TabularNasError::new(
                    format!("{field}.architecture_id"),
                    "target label references an unknown architecture",
                ));
            }
            if !target.value.is_finite() {
                return Err(TabularNasError::new(
                    format!("{field}.value"),
                    "target metric values must be finite",
                ));
            }
            let architecture_id = target.architecture_id;
            if labels.insert(architecture_id.clone(), target.value).is_some() {
                return Err(TabularNasError::new(
                    format!("{field}.architecture_id"),
                    format!("duplicate target label for {architecture_id:?}"),
                ));
            }
        }
        if labels.len() != features.len() {
            return Err(TabularNasError::new(
                "target_labels",
                "target table does not cover every architecture exactly once",
            ));
        }

        let mut ranked: Vec<(String, f64)> = labels
            .iter()
            .map(|(id, value)| (id.clone(), *value))
            .collect();
        ranked.sort_by(|(left_id, left_value), (right_id, right_value)| {
            let score_order = match manifest.metric_direction {
                MetricDirection::Maximize => right_value.total_cmp(left_value),
                MetricDirection::Minimize => left_value.total_cmp(right_value),
            };
            score_order.then_with(|| left_id.cmp(right_id))
        });
        let true_top_k_ids = ranked
            .iter()
            .take(manifest.top_k)
            .map(|(id, _)| id.clone())
            .collect();
        let global_best_metric = ranked[0].1;

        Ok(Self {
            manifest,
            features,
            labels,
            allowed_ids,
            true_top_k_ids,
            global_best_metric,
        })
    }

    pub fn manifest(&self) -> &TabularBenchmarkManifest {
        &self.manifest
    }

    pub fn candidate_features(&self) -> &[ArchitectureFeatures] {
        &self.features
    }

    /// Run a policy with only the public feature matrix and narrow query capability.
    ///
    /// The closure does not receive the benchmark/evaluator, target table, oracle top-K set,
    /// or evaluator methods. It must actually query architectures to receive P@K credit.
    pub fn run_search<P>(self, policy: P) -> TabularNasRun
    where
        P: FnOnce(&[ArchitectureFeatures], &mut QueryOracle<'_>),
    {
        let mut oracle = QueryOracle {
            metric_id: &self.manifest.metric_id,
            metric_direction: self.manifest.metric_direction,
            labels: &self.labels,
            allowed_ids: &self.allowed_ids,
            max_unique_queries: self.manifest.max_unique_queries,
            queried_ids: HashSet::with_capacity(self.manifest.max_unique_queries),
            queried_order: Vec::with_capacity(self.manifest.max_unique_queries),
            attempts: Vec::new(),
        };

        policy(&self.features, &mut oracle);
        let parts = oracle.finish();
        let receipt = self.build_receipt(parts);
        let evaluation = self.evaluate(&receipt);
        TabularNasRun {
            receipt,
            evaluation,
        }
    }

    fn build_receipt(&self, parts: QueryOracleReceiptParts) -> QueryReceipt {
        let query_trace_sha256 =
            digest_query_trace(&self.manifest, &parts.queried_order, &parts.attempts);
        QueryReceipt {
            benchmark_id: self.manifest.benchmark_id.clone(),
            metric_id: self.manifest.metric_id.clone(),
            metric_direction: self.manifest.metric_direction,
            split_id: self.manifest.split_id.clone(),
            proxy_calibration_split_id: self.manifest.proxy_calibration_split_id.clone(),
            proxy_calibration_sha256: self.manifest.proxy_calibration_sha256.clone(),
            source_artifact_sha256: self.manifest.artifact_sha256.clone(),
            max_unique_queries: self.manifest.max_unique_queries,
            top_k: self.manifest.top_k,
            unique_queries: parts.queried_order.len(),
            queried_architecture_ids: parts.queried_order,
            attempts: parts.attempts,
            query_trace_sha256,
        }
    }

    fn evaluate(&self, receipt: &QueryReceipt) -> TabularNasEvaluation {
        let queried: HashSet<&str> = receipt
            .queried_architecture_ids
            .iter()
            .map(String::as_str)
            .collect();

        let top_1 = capture_metric(1, &queried, &self.true_top_k_ids);
        let top_10 = capture_metric(10, &queried, &self.true_top_k_ids);
        let top_100 = capture_metric(100, &queried, &self.true_top_k_ids);

        let best_queried_metric = receipt
            .queried_architecture_ids
            .iter()
            .filter_map(|id| self.labels.get(id).copied())
            .reduce(|best, candidate| match self.manifest.metric_direction {
                MetricDirection::Maximize => best.max(candidate),
                MetricDirection::Minimize => best.min(candidate),
            });
        let simple_regret = best_queried_metric.map(|best| match self.manifest.metric_direction {
            MetricDirection::Maximize => self.global_best_metric - best,
            MetricDirection::Minimize => best - self.global_best_metric,
        });

        let duplicate_attempts = receipt
            .attempts
            .iter()
            .filter(|attempt| attempt.status == QueryAttemptStatus::Duplicate)
            .count();
        let unknown_architecture_attempts = receipt
            .attempts
            .iter()
            .filter(|attempt| attempt.status == QueryAttemptStatus::UnknownArchitecture)
            .count();
        let over_budget_attempts = receipt
            .attempts
            .iter()
            .filter(|attempt| attempt.status == QueryAttemptStatus::BudgetExhausted)
            .count();

        let queried_target_observations_sha256 =
            digest_queried_targets(&self.manifest.metric_id, &receipt.queried_architecture_ids, &self.labels);

        TabularNasEvaluation {
            unique_queries: receipt.unique_queries,
            query_budget: receipt.max_unique_queries,
            budget_utilization: receipt.unique_queries as f64
                / receipt.max_unique_queries as f64,
            attempt_count: receipt.attempts.len(),
            duplicate_attempts,
            unknown_architecture_attempts,
            over_budget_attempts,
            top_1,
            top_10,
            top_100,
            best_queried_metric,
            global_best_metric: self.global_best_metric,
            simple_regret,
            queried_target_observations_sha256,
        }
    }
}

fn validate_manifest(manifest: &TabularBenchmarkManifest) -> Result<(), TabularNasError> {
    if manifest.schema != BENCHMARK_SCHEMA {
        return Err(TabularNasError::new(
            "schema",
            format!("expected {BENCHMARK_SCHEMA:?}"),
        ));
    }
    if manifest.version != BENCHMARK_VERSION {
        return Err(TabularNasError::new(
            "version",
            format!("expected {BENCHMARK_VERSION:?}"),
        ));
    }
    for (field, value) in [
        ("benchmark_id", manifest.benchmark_id.as_str()),
        ("metric_id", manifest.metric_id.as_str()),
        ("split_id", manifest.split_id.as_str()),
        ("proxy_calibration_split_id", manifest.proxy_calibration_split_id.as_str()),
        ("source_description", manifest.source_description.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(TabularNasError::new(field, "must be non-empty"));
        }
    }
    if manifest.proxy_calibration_split_id == manifest.split_id {
        return Err(TabularNasError::new(
            "proxy_calibration_split_id",
            "strict no-oracle mode requires a calibration split distinct from the evaluation split",
        ));
    }
    for (field, digest) in [
        ("artifact_sha256", manifest.artifact_sha256.as_str()),
        ("proxy_calibration_sha256", manifest.proxy_calibration_sha256.as_str()),
    ] {
        if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(TabularNasError::new(
                field,
                "must contain exactly 64 hexadecimal characters",
            ));
        }
    }
    if manifest.max_unique_queries == 0 {
        return Err(TabularNasError::new(
            "max_unique_queries",
            "must be positive",
        ));
    }
    if manifest.top_k == 0 {
        return Err(TabularNasError::new("top_k", "must be positive"));
    }
    Ok(())
}

fn capture_metric(
    requested_k: usize,
    queried: &HashSet<&str>,
    true_top_k_ids: &[String],
) -> TopKCapture {
    let denominator = requested_k.min(true_top_k_ids.len());
    let queried_count = true_top_k_ids
        .iter()
        .take(denominator)
        .filter(|id| queried.contains(id.as_str()))
        .count();
    TopKCapture {
        k: requested_k,
        true_top_k_count: denominator,
        queried_top_k_count: queried_count,
        capture_rate: if denominator == 0 {
            0.0
        } else {
            queried_count as f64 / denominator as f64
        },
    }
}

fn digest_observation(metric_id: &str, architecture_id: &str, value: f64) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.tabular-nas.observation.v0.1");
    update_string(&mut hasher, metric_id);
    update_string(&mut hasher, architecture_id);
    hasher.update(value.to_bits().to_be_bytes());
    hex_digest(hasher.finalize().as_slice())
}

fn digest_query_trace(
    manifest: &TabularBenchmarkManifest,
    queried_order: &[String],
    attempts: &[QueryAttempt],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.tabular-nas.query-trace.v0.1");
    update_string(&mut hasher, &manifest.benchmark_id);
    update_string(&mut hasher, &manifest.metric_id);
    hasher.update([metric_direction_byte(manifest.metric_direction)]);
    hasher.update((manifest.top_k as u64).to_be_bytes());
    update_string(&mut hasher, &manifest.split_id);
    update_string(&mut hasher, &manifest.proxy_calibration_split_id);
    update_string(&mut hasher, &manifest.proxy_calibration_sha256);
    update_string(&mut hasher, &manifest.artifact_sha256);
    hasher.update((manifest.max_unique_queries as u64).to_be_bytes());
    hasher.update((queried_order.len() as u64).to_be_bytes());
    for architecture_id in queried_order {
        update_string(&mut hasher, architecture_id);
    }
    hasher.update((attempts.len() as u64).to_be_bytes());
    for attempt in attempts {
        hasher.update((attempt.attempt_index as u64).to_be_bytes());
        update_string(&mut hasher, &attempt.architecture_id);
        hasher.update([attempt_status_byte(attempt.status)]);
        match attempt.query_index {
            Some(index) => {
                hasher.update([1]);
                hasher.update((index as u64).to_be_bytes());
            }
            None => {
                hasher.update([0]);
            }
        }
        match &attempt.observation_sha256 {
            Some(digest) => {
                hasher.update([1]);
                update_string(&mut hasher, digest);
            }
            None => {
                hasher.update([0]);
            }
        }
    }
    hex_digest(hasher.finalize().as_slice())
}

fn digest_queried_targets(
    metric_id: &str,
    queried_order: &[String],
    labels: &HashMap<String, f64>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.tabular-nas.queried-targets.v0.1");
    update_string(&mut hasher, metric_id);
    for architecture_id in queried_order {
        update_string(&mut hasher, architecture_id);
        if let Some(value) = labels.get(architecture_id) {
            hasher.update(value.to_bits().to_be_bytes());
        }
    }
    hex_digest(hasher.finalize().as_slice())
}

fn metric_direction_byte(direction: MetricDirection) -> u8 {
    match direction {
        MetricDirection::Maximize => 0,
        MetricDirection::Minimize => 1,
    }
}

fn attempt_status_byte(status: QueryAttemptStatus) -> u8 {
    match status {
        QueryAttemptStatus::Revealed => 0,
        QueryAttemptStatus::UnknownArchitecture => 1,
        QueryAttemptStatus::Duplicate => 2,
        QueryAttemptStatus::BudgetExhausted => 3,
    }
}

fn update_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8]) -> String {
    let alphabet = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes.iter().copied() {
        encoded.push(alphabet[(byte >> 4) as usize] as char);
        encoded.push(alphabet[(byte & 0x0f) as usize] as char);
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> TabularBenchmarkManifest {
        TabularBenchmarkManifest {
            schema: BENCHMARK_SCHEMA.into(),
            version: BENCHMARK_VERSION.into(),
            benchmark_id: "synthetic-nas-firewall-v0".into(),
            metric_id: "validation_accuracy".into(),
            metric_direction: MetricDirection::Maximize,
            split_id: "challenge-a".into(),
            proxy_calibration_split_id: "development-b".into(),
            proxy_calibration_sha256: "b".repeat(64),
            source_description: "synthetic fixture only".into(),
            artifact_sha256: "a".repeat(64),
            max_unique_queries: 4,
            top_k: 12,
        }
    }

    fn features(count: usize) -> Vec<ArchitectureFeatures> {
        (0..count)
            .map(|index| {
                let mut proxies = BTreeMap::new();
                proxies.insert("proxy_a".into(), index as f64 / count.max(1) as f64);
                proxies.insert("proxy_b".into(), (count - index) as f64);
                ArchitectureFeatures {
                    architecture_id: format!("arch-{index:03}"),
                    proxy_scores: proxies,
                    topology_encoding: vec![index as u8, 1, 2, 3],
                }
            })
            .collect()
    }

    fn targets(count: usize) -> Vec<TargetLabel> {
        (0..count)
            .map(|index| TargetLabel {
                architecture_id: format!("arch-{index:03}"),
                value: index as f64,
            })
            .collect()
    }

    fn benchmark() -> TabularNasBenchmark {
        TabularNasBenchmark::new(manifest(), features(12), targets(12)).unwrap()
    }

    #[test]
    fn policy_receives_features_and_one_candidate_at_a_time() {
        let benchmark = benchmark();
        let run = benchmark.run_search(|features, oracle| {
            assert_eq!(features.len(), 12);
            assert!(features.iter().all(|feature| !feature.architecture_id.is_empty()));
            assert_eq!(oracle.query("arch-011").unwrap().value, 11.0);
        });
        assert_eq!(run.receipt.unique_queries, 1);
        assert_eq!(run.evaluation.top_1.queried_top_k_count, 1);
        assert_eq!(run.evaluation.best_queried_metric, Some(11.0));
    }

    #[test]
    fn actual_top_k_capture_counts_only_architectures_really_queried() {
        let benchmark = benchmark();
        let run = benchmark.run_search(|_, oracle| {
            for id in ["arch-000", "arch-009", "arch-010", "arch-011"] {
                oracle.query(id).unwrap();
            }
        });
        assert_eq!(run.evaluation.unique_queries, 4);
        assert_eq!(run.evaluation.top_1.capture_rate, 1.0);
        assert_eq!(run.evaluation.top_10.true_top_k_count, 10);
        assert_eq!(run.evaluation.top_10.queried_top_k_count, 3);
        assert_eq!(run.evaluation.top_100.true_top_k_count, 12);
        assert_eq!(run.evaluation.top_100.queried_top_k_count, 4);
    }

    #[test]
    fn query_budget_is_a_hard_cap_and_over_budget_attempts_are_traced() {
        let benchmark = benchmark();
        let run = benchmark.run_search(|_, oracle| {
            for id in ["arch-000", "arch-001", "arch-002", "arch-003"] {
                oracle.query(id).unwrap();
            }
            assert_eq!(oracle.query("arch-004"), Err(QueryError::BudgetExhausted));
        });
        assert_eq!(run.receipt.unique_queries, 4);
        assert_eq!(run.evaluation.over_budget_attempts, 1);
        assert_eq!(run.evaluation.budget_utilization, 1.0);
        assert_eq!(run.receipt.attempts.len(), 5);
    }

    #[test]
    fn duplicate_and_unknown_queries_are_visible_but_reveal_nothing() {
        let benchmark = benchmark();
        let run = benchmark.run_search(|_, oracle| {
            assert!(oracle.query("arch-001").is_ok());
            assert_eq!(oracle.query("arch-001"), Err(QueryError::DuplicateQuery));
            assert_eq!(oracle.query("not-in-catalog"), Err(QueryError::UnknownArchitecture));
        });
        assert_eq!(run.receipt.unique_queries, 1);
        assert_eq!(run.evaluation.duplicate_attempts, 1);
        assert_eq!(run.evaluation.unknown_architecture_attempts, 1);
        assert_eq!(run.receipt.attempts[1].observation_sha256, None);
        assert_eq!(run.receipt.attempts[2].observation_sha256, None);
    }

    #[test]
    fn query_receipt_is_deterministic_and_binds_revealed_values() {
        let benchmark = benchmark();
        let first = benchmark.run_search(|_, oracle| {
            oracle.query("arch-001").unwrap();
            oracle.query("arch-010").unwrap();
        });
        let second = benchmark.run_search(|_, oracle| {
            oracle.query("arch-001").unwrap();
            oracle.query("arch-010").unwrap();
        });
        assert_eq!(first.receipt, second.receipt);
        assert_eq!(first.receipt.query_trace_sha256.len(), 64);
        assert_ne!(
            first.receipt.attempts[0].observation_sha256,
            first.receipt.attempts[1].observation_sha256
        );
    }

    #[test]
    fn query_attempt_order_is_part_of_the_receipt() {
        let benchmark = benchmark();
        let first = benchmark.run_search(|_, oracle| {
            oracle.query("arch-001").unwrap();
            oracle.query("arch-002").unwrap();
        });
        let second = benchmark.run_search(|_, oracle| {
            oracle.query("arch-002").unwrap();
            oracle.query("arch-001").unwrap();
        });
        assert_ne!(first.receipt.query_trace_sha256, second.receipt.query_trace_sha256);
    }

    #[test]
    fn evaluator_does_not_credit_unqueried_recommendations() {
        let benchmark = benchmark();
        let run = benchmark.run_search(|features, oracle| {
            let _predicted_best = features.last().unwrap().architecture_id.clone();
            oracle.query("arch-000").unwrap();
        });
        assert_eq!(run.evaluation.unique_queries, 1);
        assert_eq!(run.evaluation.top_1.capture_rate, 0.0);
    }

    #[test]
    fn evaluation_reports_best_found_metric_and_simple_regret() {
        let benchmark = benchmark();
        let run = benchmark.run_search(|_, oracle| {
            oracle.query("arch-008").unwrap();
            oracle.query("arch-010").unwrap();
        });
        assert_eq!(run.evaluation.global_best_metric, 11.0);
        assert_eq!(run.evaluation.best_queried_metric, Some(10.0));
        assert_eq!(run.evaluation.simple_regret, Some(1.0));
    }

    #[test]
    fn invalid_schema_version_digest_and_query_budget_are_rejected() {
        let mut invalid = manifest();
        invalid.schema = "wrong".into();
        assert!(TabularNasBenchmark::new(invalid, features(12), targets(12)).is_err());

        let mut invalid = manifest();
        invalid.version = "9.9".into();
        assert!(TabularNasBenchmark::new(invalid, features(12), targets(12)).is_err());

        let mut invalid = manifest();
        invalid.artifact_sha256 = "xyz".into();
        assert!(TabularNasBenchmark::new(invalid, features(12), targets(12)).is_err());

        let mut invalid = manifest();
        invalid.max_unique_queries = 13;
        assert!(TabularNasBenchmark::new(invalid, features(12), targets(12)).is_err());
    }

    #[test]
    fn strict_no_oracle_manifest_requires_separate_calibration_split() {
        let mut invalid = manifest();
        invalid.proxy_calibration_split_id = invalid.split_id.clone();
        assert!(TabularNasBenchmark::new(invalid, features(12), targets(12)).is_err());
    }

    #[test]
    fn target_metric_cannot_be_smuggled_in_as_a_proxy_feature() {
        let mut rows = features(12);
        for row in &mut rows {
            row.proxy_scores.insert("validation_accuracy".into(), 999.0);
        }
        assert!(TabularNasBenchmark::new(manifest(), rows, targets(12)).is_err());
    }

    #[test]
    fn duplicate_architecture_ids_and_proxy_schema_mismatch_are_rejected() {
        let mut duplicate = features(12);
        duplicate[1].architecture_id = duplicate[0].architecture_id.clone();
        assert!(TabularNasBenchmark::new(manifest(), duplicate, targets(12)).is_err());

        let mut mismatch = features(12);
        mismatch[2].proxy_scores.remove("proxy_b");
        assert!(TabularNasBenchmark::new(manifest(), mismatch, targets(12)).is_err());
    }

    #[test]
    fn duplicate_missing_unknown_and_non_finite_target_labels_are_rejected() {
        let mut duplicate = targets(12);
        duplicate[1].architecture_id = duplicate[0].architecture_id.clone();
        assert!(TabularNasBenchmark::new(manifest(), features(12), duplicate).is_err());

        let mut missing = targets(12);
        missing.pop();
        assert!(TabularNasBenchmark::new(manifest(), features(12), missing).is_err());

        let mut unknown = targets(12);
        unknown[0].architecture_id = "unknown".into();
        assert!(TabularNasBenchmark::new(manifest(), features(12), unknown).is_err());

        let mut non_finite = targets(12);
        non_finite[0].value = f64::NAN;
        assert!(TabularNasBenchmark::new(manifest(), features(12), non_finite).is_err());
    }

    #[test]
    fn non_finite_proxy_values_and_empty_topology_are_rejected() {
        let mut proxy = features(12);
        proxy[1].proxy_scores.insert("proxy_a".into(), f64::INFINITY);
        assert!(TabularNasBenchmark::new(manifest(), proxy, targets(12)).is_err());

        let mut topology = features(12);
        topology[2].topology_encoding.clear();
        assert!(TabularNasBenchmark::new(manifest(), topology, targets(12)).is_err());
    }

    #[test]
    fn target_metric_ties_have_deterministic_architecture_id_order() {
        let mut labels = targets(12);
        for target in &mut labels {
            target.value = 0.5;
        }
        let benchmark = TabularNasBenchmark::new(manifest(), features(12), labels).unwrap();
        let run = benchmark.run_search(|_, oracle| {
            oracle.query("arch-000").unwrap();
        });
        assert_eq!(run.evaluation.top_1.queried_top_k_count, 1);
    }

    #[test]
    fn report_binds_source_metric_split_and_full_query_trace() {
        let benchmark = benchmark();
        let run = benchmark.run_search(|_, oracle| {
            oracle.query("arch-003").unwrap();
        });
        assert_eq!(run.receipt.benchmark_id, "synthetic-nas-firewall-v0");
        assert_eq!(run.receipt.metric_id, "validation_accuracy");
        assert_eq!(run.receipt.metric_direction, MetricDirection::Maximize);
        assert_eq!(run.receipt.top_k, 12);
        assert_eq!(run.receipt.split_id, "challenge-a");
        assert_eq!(run.receipt.proxy_calibration_split_id, "development-b");
        assert_eq!(run.receipt.proxy_calibration_sha256, "b".repeat(64));
        assert_eq!(run.receipt.source_artifact_sha256, "a".repeat(64));
        assert_eq!(run.receipt.attempts.len(), 1);
        assert_eq!(run.receipt.attempts[0].status, QueryAttemptStatus::Revealed);
    }

    #[test]
    fn minimize_metrics_use_correct_top_k_order_and_regret_direction() {
        let mut manifest = manifest();
        manifest.metric_id = "latency_ms".into();
        manifest.metric_direction = MetricDirection::Minimize;
        let labels = (0..12)
            .map(|index| TargetLabel {
                architecture_id: format!("arch-{index:03}"),
                value: (12 - index) as f64,
            })
            .collect();
        let benchmark = TabularNasBenchmark::new(manifest, features(12), labels).unwrap();
        let run = benchmark.run_search(|_, oracle| {
            let observed = oracle.query("arch-008").unwrap();
            assert_eq!(observed.metric_direction, MetricDirection::Minimize);
        });
        assert_eq!(run.evaluation.global_best_metric, 1.0);
        assert_eq!(run.evaluation.best_queried_metric, Some(4.0));
        assert_eq!(run.evaluation.simple_regret, Some(3.0));
        assert_eq!(run.evaluation.top_1.capture_rate, 0.0);
    }

    #[test]
    fn zero_query_runs_have_explicit_empty_metrics_without_nan() {
        let benchmark = benchmark();
        let run = benchmark.run_search(|_, _| {});
        assert_eq!(run.evaluation.unique_queries, 0);
        assert_eq!(run.evaluation.best_queried_metric, None);
        assert_eq!(run.evaluation.simple_regret, None);
        assert_eq!(run.evaluation.top_100.capture_rate, 0.0);
        assert_eq!(run.evaluation.budget_utilization, 0.0);
    }
}
