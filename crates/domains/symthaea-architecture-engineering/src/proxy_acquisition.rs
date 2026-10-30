//! Metered, queryable zero-cost-proxy acquisition.
//!
//! Candidate metadata/topology is public. Proxy values and measured runtimes are evaluator-owned
//! until the policy requests a specific (architecture, proxy) pair. Proxy acquisition has its
//! own pair-query and wall-clock budget, independent of target-training fidelity/resource budgets.
//!
//! This is a capability boundary for trusted policy code, not an OS process sandbox.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fmt;

use sha2::{Digest, Sha256};

use super::nasbench201_trajectory::ProxyDirection;
use super::tabular_nas_firewall::ArchitectureFeatures;

const CATALOG_DOMAIN: &[u8] = b"symthaea.tabular-nas.proxy-acquisition-catalog.v0.1";
const OBSERVATION_DOMAIN: &[u8] = b"symthaea.tabular-nas.proxy-acquisition-observation.v0.1";
const RECEIPT_DOMAIN: &[u8] = b"symthaea.tabular-nas.proxy-acquisition-receipt.v0.1";

#[derive(Debug, Clone, PartialEq)]
pub struct ProxyAcquisitionManifest {
    pub benchmark_id: String,
    pub target_metric_id: String,
    pub artifact_sha256: String,
    pub evaluation_split_id: String,
    pub proxy_calibration_split_id: String,
    pub proxy_calibration_sha256: String,
    pub proxy_directions: BTreeMap<String, ProxyDirection>,
    pub max_unique_proxy_queries: usize,
    pub max_total_proxy_seconds: f64,
}

/// Public candidate metadata. No proxy value or proxy runtime is exposed here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyCandidate {
    pub architecture_id: String,
    pub topology_encoding: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProxyObservation {
    pub architecture_id: String,
    pub proxy_id: String,
    pub direction: ProxyDirection,
    pub score: f64,
    pub measured_seconds: f64,
    pub cumulative_proxy_seconds: f64,
    pub remaining_proxy_seconds: f64,
    pub unique_query_index: usize,
    pub observation_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxyQueryError {
    UnknownArchitecture,
    UnknownProxy,
    DuplicatePair,
    QueryBudgetExhausted,
    ComputeBudgetExhausted,
}

impl fmt::Display for ProxyQueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownArchitecture => write!(f, "unknown architecture ID"),
            Self::UnknownProxy => write!(f, "proxy is not in the declared proxy schema"),
            Self::DuplicatePair => write!(f, "architecture/proxy pair was already acquired"),
            Self::QueryBudgetExhausted => write!(f, "unique proxy-query budget exhausted"),
            Self::ComputeBudgetExhausted => write!(f, "proxy-compute time budget exhausted"),
        }
    }
}
impl Error for ProxyQueryError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxyQueryAttemptStatus {
    Revealed,
    UnknownArchitecture,
    UnknownProxy,
    DuplicatePair,
    QueryBudgetExhausted,
    ComputeBudgetExhausted,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProxyQueryAttempt {
    pub attempt_index: usize,
    pub architecture_id: String,
    pub proxy_id: String,
    pub status: ProxyQueryAttemptStatus,
    pub measured_seconds: f64,
    pub cumulative_seconds_after_attempt: f64,
    pub unique_query_index: Option<usize>,
    pub observation_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProxyAcquisitionReceipt {
    pub benchmark_id: String,
    pub target_metric_id: String,
    pub artifact_sha256: String,
    pub evaluation_split_id: String,
    pub proxy_calibration_split_id: String,
    pub proxy_calibration_sha256: String,
    pub source_proxy_catalog_sha256: String,
    pub max_unique_proxy_queries: usize,
    pub max_total_proxy_seconds: f64,
    pub unique_proxy_queries: usize,
    pub unique_architectures_observed: usize,
    pub total_proxy_seconds: f64,
    pub attempts: Vec<ProxyQueryAttempt>,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProxyAcquisitionEvaluation {
    pub unique_proxy_queries: usize,
    pub max_unique_proxy_queries: usize,
    pub query_budget_utilization: f64,
    pub unique_architectures_observed: usize,
    pub total_proxy_seconds: f64,
    pub max_total_proxy_seconds: f64,
    pub compute_budget_utilization: f64,
    pub unknown_architecture_attempts: usize,
    pub unknown_proxy_attempts: usize,
    pub duplicate_pair_attempts: usize,
    pub query_budget_exhausted_attempts: usize,
    pub compute_budget_exhausted_attempts: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProxyAcquisitionRun {
    pub candidates: Vec<ProxyCandidate>,
    pub declared_proxy_directions: BTreeMap<String, ProxyDirection>,
    /// Contains only successfully acquired proxy observations.
    pub observations: Vec<ProxyObservation>,
    pub receipt: ProxyAcquisitionReceipt,
    pub evaluation: ProxyAcquisitionEvaluation,
}

impl ProxyAcquisitionRun {
    /// Materialize a rectangular architecture × proxy table from successfully queried evidence.
    /// Missing candidates/proxy pairs fail closed; this method never fills an unqueried value
    /// with a sentinel or copies hidden values from the evaluator's source table.
    pub fn to_complete_feature_table(
        &self,
    ) -> Result<Vec<ArchitectureFeatures>, ProxyAcquisitionError> {
        if self.declared_proxy_directions.is_empty() {
            return Err(ProxyAcquisitionError::new(
                "declared_proxy_directions",
                "at least one proxy is required",
            ));
        }
        let mut by_architecture: HashMap<String, (BTreeMap<String, f64>, BTreeMap<String, f64>)> =
            HashMap::with_capacity(self.candidates.len());
        for candidate in &self.candidates {
            by_architecture.insert(
                candidate.architecture_id.clone(),
                (BTreeMap::new(), BTreeMap::new()),
            );
        }
        for observation in &self.observations {
            let Some((scores, runtimes)) = by_architecture.get_mut(&observation.architecture_id) else {
                return Err(ProxyAcquisitionError::new(
                    "observations",
                    "observation references an architecture outside the public candidate catalog",
                ));
            };
            if !self.declared_proxy_directions.contains_key(&observation.proxy_id) {
                return Err(ProxyAcquisitionError::new(
                    "observations",
                    "observation references an undeclared proxy",
                ));
            }
            if scores.insert(observation.proxy_id.clone(), observation.score).is_some()
                || runtimes
                    .insert(observation.proxy_id.clone(), observation.measured_seconds)
                    .is_some()
            {
                return Err(ProxyAcquisitionError::new(
                    "observations",
                    "duplicate architecture/proxy observation",
                ));
            }
        }

        let mut output = Vec::with_capacity(self.candidates.len());
        for candidate in &self.candidates {
            let Some((scores, runtimes)) = by_architecture.remove(&candidate.architecture_id) else {
                return Err(ProxyAcquisitionError::new(
                    "candidates",
                    "candidate row unexpectedly missing",
                ));
            };
            let expected: BTreeSet<&str> = self
                .declared_proxy_directions
                .keys()
                .map(String::as_str)
                .collect();
            let actual_scores: BTreeSet<&str> =
                scores.keys().map(String::as_str).collect();
            let actual_runtimes: BTreeSet<&str> =
                runtimes.keys().map(String::as_str).collect();
            if actual_scores != expected || actual_runtimes != expected {
                return Err(ProxyAcquisitionError::new(
                    "observations",
                    format!(
                        "candidate {:?} does not have every declared proxy observation",
                        candidate.architecture_id
                    ),
                ));
            }
            output.push(ArchitectureFeatures {
                architecture_id: candidate.architecture_id.clone(),
                proxy_scores: scores,
                proxy_times_seconds: runtimes,
                topology_encoding: candidate.topology_encoding.clone(),
            });
        }
        Ok(output)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyAcquisitionError {
    pub field: String,
    pub message: String,
}
impl ProxyAcquisitionError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}
impl fmt::Display for ProxyAcquisitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}
impl Error for ProxyAcquisitionError {}

struct OracleParts {
    observations: Vec<ProxyObservation>,
    attempts: Vec<ProxyQueryAttempt>,
    unique_architectures: BTreeSet<String>,
    total_seconds: f64,
}

pub struct ProxyAcquisitionOracle<'a> {
    manifest: &'a ProxyAcquisitionManifest,
    scores: &'a HashMap<(String, String), f64>,
    runtimes: &'a HashMap<(String, String), f64>,
    allowed_architectures: &'a HashSet<String>,
    acquired_pairs: HashSet<(String, String)>,
    unique_architectures: BTreeSet<String>,
    observations: Vec<ProxyObservation>,
    attempts: Vec<ProxyQueryAttempt>,
    total_seconds: f64,
}

impl ProxyAcquisitionOracle<'_> {
    /// Acquire one architecture/proxy pair. Successful unique queries reveal a score and charge
    /// the proxy's measured runtime. Duplicate/unknown/over-budget attempts are traced and reveal
    /// no score or runtime.
    pub fn query(
        &mut self,
        architecture_id: &str,
        proxy_id: &str,
    ) -> Result<ProxyObservation, ProxyQueryError> {
        let attempt_index = self.attempts.len();
        if !self.allowed_architectures.contains(architecture_id) {
            self.fail(
                attempt_index,
                architecture_id,
                proxy_id,
                ProxyQueryAttemptStatus::UnknownArchitecture,
            );
            return Err(ProxyQueryError::UnknownArchitecture);
        }
        if !self.manifest.proxy_directions.contains_key(proxy_id) {
            self.fail(
                attempt_index,
                architecture_id,
                proxy_id,
                ProxyQueryAttemptStatus::UnknownProxy,
            );
            return Err(ProxyQueryError::UnknownProxy);
        }
        let pair = (architecture_id.to_owned(), proxy_id.to_owned());
        if self.acquired_pairs.contains(&pair) {
            self.fail(
                attempt_index,
                architecture_id,
                proxy_id,
                ProxyQueryAttemptStatus::DuplicatePair,
            );
            return Err(ProxyQueryError::DuplicatePair);
        }
        if self.acquired_pairs.len() >= self.manifest.max_unique_proxy_queries {
            self.fail(
                attempt_index,
                architecture_id,
                proxy_id,
                ProxyQueryAttemptStatus::QueryBudgetExhausted,
            );
            return Err(ProxyQueryError::QueryBudgetExhausted);
        }

        let seconds = self.runtimes[&pair];
        let Some(next_total) = (self.total_seconds + seconds)
            .is_finite()
            .then_some(self.total_seconds + seconds)
        else {
            self.fail(
                attempt_index,
                architecture_id,
                proxy_id,
                ProxyQueryAttemptStatus::ComputeBudgetExhausted,
            );
            return Err(ProxyQueryError::ComputeBudgetExhausted);
        };
        if next_total > self.manifest.max_total_proxy_seconds {
            self.fail(
                attempt_index,
                architecture_id,
                proxy_id,
                ProxyQueryAttemptStatus::ComputeBudgetExhausted,
            );
            return Err(ProxyQueryError::ComputeBudgetExhausted);
        }

        let score = self.scores[&pair];
        let unique_query_index = self.acquired_pairs.len();
        let observation_sha256 =
            digest_observation(architecture_id, proxy_id, score, seconds);
        self.acquired_pairs.insert(pair);
        self.unique_architectures.insert(architecture_id.to_owned());
        self.total_seconds = next_total;
        let observation = ProxyObservation {
            architecture_id: architecture_id.to_owned(),
            proxy_id: proxy_id.to_owned(),
            direction: self.manifest.proxy_directions[proxy_id],
            score,
            measured_seconds: seconds,
            cumulative_proxy_seconds: self.total_seconds,
            remaining_proxy_seconds: self.manifest.max_total_proxy_seconds
                - self.total_seconds,
            unique_query_index,
            observation_sha256: observation_sha256.clone(),
        };
        self.observations.push(observation.clone());
        self.attempts.push(ProxyQueryAttempt {
            attempt_index,
            architecture_id: architecture_id.to_owned(),
            proxy_id: proxy_id.to_owned(),
            status: ProxyQueryAttemptStatus::Revealed,
            measured_seconds: seconds,
            cumulative_seconds_after_attempt: self.total_seconds,
            unique_query_index: Some(unique_query_index),
            observation_sha256: Some(observation_sha256),
        });
        Ok(observation)
    }

    fn fail(
        &mut self,
        attempt_index: usize,
        architecture_id: &str,
        proxy_id: &str,
        status: ProxyQueryAttemptStatus,
    ) {
        self.attempts.push(ProxyQueryAttempt {
            attempt_index,
            architecture_id: architecture_id.to_owned(),
            proxy_id: proxy_id.to_owned(),
            status,
            measured_seconds: 0.0,
            cumulative_seconds_after_attempt: self.total_seconds,
            unique_query_index: None,
            observation_sha256: None,
        });
    }

    fn finish(self) -> OracleParts {
        OracleParts {
            observations: self.observations,
            attempts: self.attempts,
            unique_architectures: self.unique_architectures,
            total_seconds: self.total_seconds,
        }
    }
}

/// Evaluator that owns the complete proxy-value table. The policy receives only candidate
/// metadata and the per-pair query capability.
pub struct ProxyAcquisitionBenchmark {
    manifest: ProxyAcquisitionManifest,
    candidates: Vec<ProxyCandidate>,
    scores: HashMap<(String, String), f64>,
    runtimes: HashMap<(String, String), f64>,
    allowed_architectures: HashSet<String>,
    source_proxy_catalog_sha256: String,
}

impl ProxyAcquisitionBenchmark {
    pub fn new(
        manifest: ProxyAcquisitionManifest,
        features: Vec<ArchitectureFeatures>,
    ) -> Result<Self, ProxyAcquisitionError> {
        validate_manifest(&manifest)?;
        if features.is_empty() {
            return Err(ProxyAcquisitionError::new(
                "features",
                "at least one architecture is required",
            ));
        }
        let declared_proxy_ids: BTreeSet<&str> =
            manifest.proxy_directions.keys().map(String::as_str).collect();
        let mut candidates = Vec::with_capacity(features.len());
        let mut scores = HashMap::new();
        let mut runtimes = HashMap::new();
        let mut allowed_architectures = HashSet::with_capacity(features.len());
        let mut seen_topologies = HashSet::new();

        for (index, feature) in features.iter().enumerate() {
            let field = format!("features[{index}]");
            if feature.architecture_id.trim().is_empty()
                || !allowed_architectures.insert(feature.architecture_id.clone())
            {
                return Err(ProxyAcquisitionError::new(
                    format!("{field}.architecture_id"),
                    "must be non-empty and unique",
                ));
            }
            if feature.topology_encoding.len() != 6
                || feature.topology_encoding.iter().any(|operation| *operation >= 5)
            {
                return Err(ProxyAcquisitionError::new(
                    format!("{field}.topology_encoding"),
                    "NAS-Bench-201 topology must contain six operation IDs in 0..5",
                ));
            }
            if !seen_topologies.insert(feature.topology_encoding.as_slice()) {
                return Err(ProxyAcquisitionError::new(
                    format!("{field}.topology_encoding"),
                    "duplicate topology encoding",
                ));
            }
            let score_ids: BTreeSet<&str> =
                feature.proxy_scores.keys().map(String::as_str).collect();
            let runtime_ids: BTreeSet<&str> = feature
                .proxy_times_seconds
                .keys()
                .map(String::as_str)
                .collect();
            if score_ids != declared_proxy_ids || runtime_ids != declared_proxy_ids {
                return Err(ProxyAcquisitionError::new(
                    format!("{field}.proxy_scores"),
                    "proxy score/runtime keys must exactly match the manifest",
                ));
            }
            if feature.proxy_scores.contains_key(&manifest.target_metric_id)
                || feature.proxy_times_seconds.contains_key(&manifest.target_metric_id)
            {
                return Err(ProxyAcquisitionError::new(
                    field,
                    "target metric cannot appear in proxy scores or runtimes",
                ));
            }
            for proxy_id in &declared_proxy_ids {
                let score = feature.proxy_scores[*proxy_id];
                let seconds = feature.proxy_times_seconds[*proxy_id];
                if !score.is_finite() {
                    return Err(ProxyAcquisitionError::new(
                        format!("{field}.proxy_scores.{proxy_id}"),
                        "must be finite",
                    ));
                }
                if !seconds.is_finite() || seconds < 0.0 {
                    return Err(ProxyAcquisitionError::new(
                        format!("{field}.proxy_times_seconds.{proxy_id}"),
                        "must be finite and non-negative",
                    ));
                }
                let key = (feature.architecture_id.clone(), (*proxy_id).to_owned());
                scores.insert(key.clone(), score);
                runtimes.insert(key, seconds);
            }
            candidates.push(ProxyCandidate {
                architecture_id: feature.architecture_id.clone(),
                topology_encoding: feature.topology_encoding.clone(),
            });
        }

        let available_pairs = features
            .len()
            .checked_mul(manifest.proxy_directions.len())
            .ok_or_else(|| {
                ProxyAcquisitionError::new("features", "architecture/proxy pair count overflow")
            })?;
        if manifest.max_unique_proxy_queries > available_pairs {
            return Err(ProxyAcquisitionError::new(
                "max_unique_proxy_queries",
                "cannot exceed the number of unique architecture/proxy pairs",
            ));
        }
        let source_proxy_catalog_sha256 = digest_source_catalog(&manifest, &features);
        candidates.sort_by(|left, right| left.architecture_id.cmp(&right.architecture_id));

        Ok(Self {
            manifest,
            candidates,
            scores,
            runtimes,
            allowed_architectures,
            source_proxy_catalog_sha256,
        })
    }

    /// Consume the evaluator once. The callback is not given any unqueried proxy values.
    pub fn run_policy<P>(self, policy: P) -> ProxyAcquisitionRun
    where
        P: FnOnce(&[ProxyCandidate], &mut ProxyAcquisitionOracle<'_>),
    {
        let mut oracle = ProxyAcquisitionOracle {
            manifest: &self.manifest,
            scores: &self.scores,
            runtimes: &self.runtimes,
            allowed_architectures: &self.allowed_architectures,
            acquired_pairs: HashSet::new(),
            unique_architectures: BTreeSet::new(),
            observations: Vec::new(),
            attempts: Vec::new(),
            total_seconds: 0.0,
        };
        policy(&self.candidates, &mut oracle);
        let parts = oracle.finish();
        let receipt = self.build_receipt(&parts);
        let evaluation = evaluate_receipt(&receipt);
        ProxyAcquisitionRun {
            candidates: self.candidates,
            declared_proxy_directions: self.manifest.proxy_directions.clone(),
            observations: parts.observations,
            receipt,
            evaluation,
        }
    }

    fn build_receipt(&self, parts: &OracleParts) -> ProxyAcquisitionReceipt {
        let receipt_sha256 = digest_receipt(
            &self.manifest,
            &self.source_proxy_catalog_sha256,
            &parts.attempts,
            parts.total_seconds,
        );
        ProxyAcquisitionReceipt {
            benchmark_id: self.manifest.benchmark_id.clone(),
            target_metric_id: self.manifest.target_metric_id.clone(),
            artifact_sha256: self.manifest.artifact_sha256.clone(),
            evaluation_split_id: self.manifest.evaluation_split_id.clone(),
            proxy_calibration_split_id: self.manifest.proxy_calibration_split_id.clone(),
            proxy_calibration_sha256: self.manifest.proxy_calibration_sha256.clone(),
            source_proxy_catalog_sha256: self.source_proxy_catalog_sha256.clone(),
            max_unique_proxy_queries: self.manifest.max_unique_proxy_queries,
            max_total_proxy_seconds: self.manifest.max_total_proxy_seconds,
            unique_proxy_queries: parts
                .attempts
                .iter()
                .filter(|attempt| attempt.status == ProxyQueryAttemptStatus::Revealed)
                .count(),
            unique_architectures_observed: parts.unique_architectures.len(),
            total_proxy_seconds: parts.total_seconds,
            attempts: parts.attempts.clone(),
            receipt_sha256,
        }
    }
}

fn validate_manifest(
    manifest: &ProxyAcquisitionManifest,
) -> Result<(), ProxyAcquisitionError> {
    for (field, value) in [
        ("benchmark_id", manifest.benchmark_id.as_str()),
        ("target_metric_id", manifest.target_metric_id.as_str()),
        ("evaluation_split_id", manifest.evaluation_split_id.as_str()),
        (
            "proxy_calibration_split_id",
            manifest.proxy_calibration_split_id.as_str(),
        ),
    ] {
        if value.trim().is_empty() {
            return Err(ProxyAcquisitionError::new(field, "must be non-empty"));
        }
    }
    if manifest.evaluation_split_id == manifest.proxy_calibration_split_id {
        return Err(ProxyAcquisitionError::new(
            "proxy_calibration_split_id",
            "must differ from evaluation split",
        ));
    }
    for (field, digest) in [
        ("artifact_sha256", manifest.artifact_sha256.as_str()),
        (
            "proxy_calibration_sha256",
            manifest.proxy_calibration_sha256.as_str(),
        ),
    ] {
        if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(ProxyAcquisitionError::new(
                field,
                "must be exactly 64 hexadecimal characters",
            ));
        }
    }
    if manifest.proxy_directions.is_empty() {
        return Err(ProxyAcquisitionError::new(
            "proxy_directions",
            "at least one proxy is required",
        ));
    }
    if manifest
        .proxy_directions
        .keys()
        .any(|proxy_id| proxy_id.trim().is_empty() || proxy_id == &manifest.target_metric_id)
    {
        return Err(ProxyAcquisitionError::new(
            "proxy_directions",
            "proxy IDs must be non-empty and distinct from target metric",
        ));
    }
    if manifest.max_unique_proxy_queries == 0 {
        return Err(ProxyAcquisitionError::new(
            "max_unique_proxy_queries",
            "must be positive",
        ));
    }
    if !manifest.max_total_proxy_seconds.is_finite()
        || manifest.max_total_proxy_seconds <= 0.0
    {
        return Err(ProxyAcquisitionError::new(
            "max_total_proxy_seconds",
            "must be finite and positive",
        ));
    }
    Ok(())
}

fn evaluate_receipt(receipt: &ProxyAcquisitionReceipt) -> ProxyAcquisitionEvaluation {
    let count = |status| {
        receipt
            .attempts
            .iter()
            .filter(|attempt| attempt.status == status)
            .count()
    };
    ProxyAcquisitionEvaluation {
        unique_proxy_queries: receipt.unique_proxy_queries,
        max_unique_proxy_queries: receipt.max_unique_proxy_queries,
        query_budget_utilization: receipt.unique_proxy_queries as f64
            / receipt.max_unique_proxy_queries as f64,
        unique_architectures_observed: receipt.unique_architectures_observed,
        total_proxy_seconds: receipt.total_proxy_seconds,
        max_total_proxy_seconds: receipt.max_total_proxy_seconds,
        compute_budget_utilization: receipt.total_proxy_seconds
            / receipt.max_total_proxy_seconds,
        unknown_architecture_attempts: count(ProxyQueryAttemptStatus::UnknownArchitecture),
        unknown_proxy_attempts: count(ProxyQueryAttemptStatus::UnknownProxy),
        duplicate_pair_attempts: count(ProxyQueryAttemptStatus::DuplicatePair),
        query_budget_exhausted_attempts: count(ProxyQueryAttemptStatus::QueryBudgetExhausted),
        compute_budget_exhausted_attempts: count(
            ProxyQueryAttemptStatus::ComputeBudgetExhausted,
        ),
    }
}

fn digest_source_catalog(
    manifest: &ProxyAcquisitionManifest,
    features: &[ArchitectureFeatures],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(CATALOG_DOMAIN);
    update_string(&mut hasher, &manifest.benchmark_id);
    update_string(&mut hasher, &manifest.target_metric_id);
    update_string(&mut hasher, &manifest.artifact_sha256);
    update_string(&mut hasher, &manifest.evaluation_split_id);
    update_string(&mut hasher, &manifest.proxy_calibration_split_id);
    update_string(&mut hasher, &manifest.proxy_calibration_sha256);
    hasher.update((manifest.proxy_directions.len() as u64).to_be_bytes());
    for (proxy_id, direction) in &manifest.proxy_directions {
        update_string(&mut hasher, proxy_id);
        hasher.update([direction_byte(*direction)]);
    }
    let mut ordered: Vec<&ArchitectureFeatures> = features.iter().collect();
    ordered.sort_by(|left, right| left.architecture_id.cmp(&right.architecture_id));
    hasher.update((ordered.len() as u64).to_be_bytes());
    for feature in ordered {
        update_string(&mut hasher, &feature.architecture_id);
        hasher.update((feature.topology_encoding.len() as u64).to_be_bytes());
        hasher.update(&feature.topology_encoding);
        for proxy_id in manifest.proxy_directions.keys() {
            update_string(&mut hasher, proxy_id);
            hasher.update(feature.proxy_scores[proxy_id].to_bits().to_be_bytes());
            hasher.update(feature.proxy_times_seconds[proxy_id].to_bits().to_be_bytes());
        }
    }
    hex_digest(hasher.finalize().as_slice())
}

fn digest_observation(
    architecture_id: &str,
    proxy_id: &str,
    score: f64,
    seconds: f64,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(OBSERVATION_DOMAIN);
    update_string(&mut hasher, architecture_id);
    update_string(&mut hasher, proxy_id);
    hasher.update(score.to_bits().to_be_bytes());
    hasher.update(seconds.to_bits().to_be_bytes());
    hex_digest(hasher.finalize().as_slice())
}

fn digest_receipt(
    manifest: &ProxyAcquisitionManifest,
    catalog_sha256: &str,
    attempts: &[ProxyQueryAttempt],
    total_seconds: f64,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(RECEIPT_DOMAIN);
    update_string(&mut hasher, &manifest.benchmark_id);
    update_string(&mut hasher, &manifest.target_metric_id);
    update_string(&mut hasher, &manifest.artifact_sha256);
    update_string(&mut hasher, &manifest.evaluation_split_id);
    update_string(&mut hasher, &manifest.proxy_calibration_split_id);
    update_string(&mut hasher, &manifest.proxy_calibration_sha256);
    update_string(&mut hasher, catalog_sha256);
    hasher.update((manifest.max_unique_proxy_queries as u64).to_be_bytes());
    hasher.update(manifest.max_total_proxy_seconds.to_bits().to_be_bytes());
    hasher.update((attempts.len() as u64).to_be_bytes());
    for attempt in attempts {
        hasher.update((attempt.attempt_index as u64).to_be_bytes());
        update_string(&mut hasher, &attempt.architecture_id);
        update_string(&mut hasher, &attempt.proxy_id);
        hasher.update([status_byte(attempt.status)]);
        hasher.update(attempt.measured_seconds.to_bits().to_be_bytes());
        hasher.update(attempt.cumulative_seconds_after_attempt.to_bits().to_be_bytes());
        match attempt.unique_query_index {
            Some(index) => {
                hasher.update([1]);
                hasher.update((index as u64).to_be_bytes());
            }
            None => hasher.update([0]),
        }
        match &attempt.observation_sha256 {
            Some(digest) => {
                hasher.update([1]);
                update_string(&mut hasher, digest);
            }
            None => hasher.update([0]),
        }
    }
    hasher.update(total_seconds.to_bits().to_be_bytes());
    hex_digest(hasher.finalize().as_slice())
}

fn status_byte(status: ProxyQueryAttemptStatus) -> u8 {
    match status {
        ProxyQueryAttemptStatus::Revealed => 0,
        ProxyQueryAttemptStatus::UnknownArchitecture => 1,
        ProxyQueryAttemptStatus::UnknownProxy => 2,
        ProxyQueryAttemptStatus::DuplicatePair => 3,
        ProxyQueryAttemptStatus::QueryBudgetExhausted => 4,
        ProxyQueryAttemptStatus::ComputeBudgetExhausted => 5,
    }
}

fn direction_byte(direction: ProxyDirection) -> u8 {
    match direction {
        ProxyDirection::Maximize => 0,
        ProxyDirection::Minimize => 1,
    }
}

fn update_string(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8]) -> String {
    const DIGITS: &[u8] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes.iter().copied() {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn features() -> Vec<ArchitectureFeatures> {
        (0..6)
            .map(|index| {
                let mut proxy_scores = BTreeMap::new();
                proxy_scores.insert("proxy_a".into(), index as f64);
                proxy_scores.insert("proxy_b".into(), (5 - index) as f64);
                let mut proxy_times_seconds = BTreeMap::new();
                proxy_times_seconds.insert("proxy_a".into(), 0.01 * (index + 1) as f64);
                proxy_times_seconds.insert("proxy_b".into(), 0.02 * (index + 1) as f64);
                ArchitectureFeatures {
                    architecture_id: format!("arch-{index}"),
                    proxy_scores,
                    proxy_times_seconds,
                    topology_encoding: vec![
                        (index / 5) as u8,
                        (index % 5) as u8,
                        0,
                        0,
                        0,
                        0,
                    ],
                }
            })
            .collect()
    }

    fn manifest(query_cap: usize, seconds_cap: f64) -> ProxyAcquisitionManifest {
        let mut proxy_directions = BTreeMap::new();
        proxy_directions.insert("proxy_a".into(), ProxyDirection::Maximize);
        proxy_directions.insert("proxy_b".into(), ProxyDirection::Minimize);
        ProxyAcquisitionManifest {
            benchmark_id: "proxy-acquisition-synthetic".into(),
            target_metric_id: "val_accuracy".into(),
            artifact_sha256: "a".repeat(64),
            evaluation_split_id: "challenge-a".into(),
            proxy_calibration_split_id: "development-b".into(),
            proxy_calibration_sha256: "b".repeat(64),
            proxy_directions,
            max_unique_proxy_queries: query_cap,
            max_total_proxy_seconds: seconds_cap,
        }
    }

    fn benchmark(query_cap: usize, seconds_cap: f64) -> ProxyAcquisitionBenchmark {
        ProxyAcquisitionBenchmark::new(manifest(query_cap, seconds_cap), features()).unwrap()
    }

    #[test]
    fn policy_sees_candidate_metadata_without_proxy_columns() {
        let benchmark = benchmark(12, 10.0);
        let run = benchmark.run_policy(|candidates, _| {
            assert_eq!(candidates.len(), 6);
            assert_eq!(candidates[0].architecture_id, "arch-0");
            assert_eq!(candidates[0].topology_encoding.len(), 6);
        });
        assert!(run.observations.is_empty());
        assert_eq!(run.receipt.unique_proxy_queries, 0);
        assert_eq!(run.receipt.total_proxy_seconds, 0.0);
    }

    #[test]
    fn unique_proxy_queries_charge_measured_runtime() {
        let benchmark = benchmark(12, 10.0);
        let run = benchmark.run_policy(|_, oracle| {
            let first = oracle.query("arch-0", "proxy_a").unwrap();
            assert_eq!(first.score, 0.0);
            assert_eq!(first.measured_seconds, 0.01);
            let second = oracle.query("arch-5", "proxy_b").unwrap();
            assert_eq!(second.score, 0.0);
            assert_eq!(second.measured_seconds, 0.12);
        });
        assert_eq!(run.receipt.unique_proxy_queries, 2);
        assert_eq!(run.receipt.unique_architectures_observed, 2);
        assert!((run.receipt.total_proxy_seconds - 0.13).abs() < 1e-12);
        assert_eq!(run.observations.len(), 2);
    }

    #[test]
    fn duplicate_architecture_proxy_pair_is_rejected_without_second_charge() {
        let benchmark = benchmark(12, 10.0);
        let run = benchmark.run_policy(|_, oracle| {
            oracle.query("arch-0", "proxy_a").unwrap();
            assert_eq!(
                oracle.query("arch-0", "proxy_a"),
                Err(ProxyQueryError::DuplicatePair)
            );
        });
        assert_eq!(run.receipt.unique_proxy_queries, 1);
        assert_eq!(run.evaluation.duplicate_pair_attempts, 1);
        assert!((run.receipt.total_proxy_seconds - 0.01).abs() < 1e-12);
    }

    #[test]
    fn pair_query_cap_and_proxy_seconds_cap_are_independent() {
        let query_limited = benchmark(1, 10.0);
        let run = query_limited.run_policy(|_, oracle| {
            oracle.query("arch-0", "proxy_a").unwrap();
            assert_eq!(
                oracle.query("arch-0", "proxy_b"),
                Err(ProxyQueryError::QueryBudgetExhausted)
            );
        });
        assert_eq!(run.receipt.unique_proxy_queries, 1);
        assert_eq!(run.evaluation.query_budget_exhausted_attempts, 1);

        let compute_limited = benchmark(12, 0.02);
        let run = compute_limited.run_policy(|_, oracle| {
            oracle.query("arch-0", "proxy_a").unwrap();
            assert_eq!(
                oracle.query("arch-1", "proxy_b"),
                Err(ProxyQueryError::ComputeBudgetExhausted)
            );
        });
        assert_eq!(run.receipt.unique_proxy_queries, 1);
        assert_eq!(run.evaluation.compute_budget_exhausted_attempts, 1);
    }

    #[test]
    fn unknown_architecture_and_proxy_reveal_nothing() {
        let benchmark = benchmark(12, 10.0);
        let run = benchmark.run_policy(|_, oracle| {
            assert_eq!(
                oracle.query("unknown", "proxy_a"),
                Err(ProxyQueryError::UnknownArchitecture)
            );
            assert_eq!(
                oracle.query("arch-0", "invented"),
                Err(ProxyQueryError::UnknownProxy)
            );
        });
        assert!(run.observations.is_empty());
        assert!(run.receipt.attempts.iter().all(|attempt| attempt.observation_sha256.is_none()));
        assert_eq!(run.receipt.total_proxy_seconds, 0.0);
    }

    #[test]
    fn fully_acquired_rectangular_evidence_materializes_for_consensus() {
        let benchmark = benchmark(12, 10.0);
        let run = benchmark.run_policy(|candidates, oracle| {
            for candidate in candidates {
                oracle.query(&candidate.architecture_id, "proxy_a").unwrap();
                oracle.query(&candidate.architecture_id, "proxy_b").unwrap();
            }
        });
        let rows = run.to_complete_feature_table().unwrap();
        assert_eq!(rows.len(), 6);
        assert!(rows.iter().all(|row| row.proxy_scores.len() == 2));
        assert_eq!(rows[5].proxy_scores["proxy_a"], 5.0);
        assert_eq!(rows[5].proxy_times_seconds["proxy_b"], 0.12);
    }

    #[test]
    fn partial_acquisition_cannot_be_materialized_as_complete_evidence() {
        let benchmark = benchmark(12, 10.0);
        let run = benchmark.run_policy(|candidates, oracle| {
            oracle.query(&candidates[0].architecture_id, "proxy_a").unwrap();
        });
        assert!(run.to_complete_feature_table().is_err());
    }

    #[test]
    fn receipt_is_deterministic_and_records_failures() {
        let first = benchmark(12, 10.0).run_policy(|_, oracle| {
            oracle.query("arch-0", "proxy_a").unwrap();
            let _ = oracle.query("arch-0", "proxy_a");
        });
        let second = benchmark(12, 10.0).run_policy(|_, oracle| {
            oracle.query("arch-0", "proxy_a").unwrap();
            let _ = oracle.query("arch-0", "proxy_a");
        });
        assert_eq!(first.receipt.receipt_sha256, second.receipt.receipt_sha256);
        assert_eq!(first.receipt.attempts, second.receipt.attempts);
    }

    #[test]
    fn exact_proxy_schema_rejects_missing_and_extra_columns() {
        let mut rows = features();
        rows[0].proxy_scores.remove("proxy_b");
        assert!(ProxyAcquisitionBenchmark::new(manifest(12, 10.0), rows).is_err());

        let mut rows = features();
        rows[0].proxy_scores.insert("extra".into(), 0.0);
        rows[0].proxy_times_seconds.insert("extra".into(), 0.0);
        assert!(ProxyAcquisitionBenchmark::new(manifest(12, 10.0), rows).is_err());
    }

    #[test]
    fn target_metric_is_rejected_from_proxy_values_and_runtimes() {
        let mut rows = features();
        rows[0].proxy_scores.insert("val_accuracy".into(), 99.0);
        rows[0].proxy_times_seconds.insert("val_accuracy".into(), 0.0);
        assert!(ProxyAcquisitionBenchmark::new(manifest(12, 10.0), rows).is_err());

        let mut rows = features();
        rows[0].proxy_times_seconds.insert("val_accuracy".into(), 0.0);
        assert!(ProxyAcquisitionBenchmark::new(manifest(12, 10.0), rows).is_err());
    }

    #[test]
    fn constructor_rejects_duplicate_topologies_and_bad_provenance() {
        let mut rows = features();
        rows[1].topology_encoding = rows[0].topology_encoding.clone();
        assert!(ProxyAcquisitionBenchmark::new(manifest(12, 10.0), rows).is_err());

        let mut invalid = manifest(12, 10.0);
        invalid.evaluation_split_id = invalid.proxy_calibration_split_id.clone();
        assert!(ProxyAcquisitionBenchmark::new(invalid, features()).is_err());
    }

    #[test]
    fn max_query_budget_cannot_exceed_available_pairs() {
        assert!(ProxyAcquisitionBenchmark::new(manifest(13, 10.0), features()).is_err());
    }
}
