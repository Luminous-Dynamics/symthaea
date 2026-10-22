//! Open-world physical-law discovery reference evaluator (v0.3).
//!
//! This crate is intentionally a small, dependency-light baseline harness. It
//! does not claim to be a general discovery agent or a blind benchmark: the
//! reference corpus is source-visible. Its purpose is to freeze an executable
//! model-selection/evaluation contract before adapter work begins.
//!
//! The candidate generator and the independent scoring boundary must remain
//! separate in later versions. In v0.3, the grid-search fit is a baseline;
//! the report explicitly labels the corpus as non-blind.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fmt;

pub const REPORT_SCHEMA: &str = "open-world-physics-evaluator.v0.3";
pub const TRAIN_X_MIN: f64 = 0.75;
pub const TRAIN_X_MAX: f64 = 3.5;
pub const HOLDOUT_X_MIN: f64 = 4.0;
pub const HOLDOUT_X_MAX: f64 = 7.0;
pub const TRAIN_QUERY_BUDGET: usize = 12;
pub const HOLDOUT_POINT_COUNT: usize = 31;
const RELATIVE_NOISE: f64 = 0.005;
/// A BIC difference below this value is reported as indeterminate, not a unique recovery.
const BIC_DECISIVE_DELTA: f64 = 2.0;
/// Per-family proposal counts used to expose the candidate-selection bottleneck.
const CANDIDATES_PER_FAMILY: &[usize] = &[1, 3, 10, 30];
/// Independent paired seed replications for each diagnostic world.
const SELECTION_REPLICATION_COUNT: usize = 5;
const SELECTION_SEED_SCHEDULE: &str = "splitmix64(world_seed ^ 0x53454c4543540001 ^ (replication_index * 0x9e3779b97f4a7c15))";
const SELECTION_SEED_SALT: u64 = 0x5345_4c45_4354_0001;

const AMPLITUDES: &[f64] = &[0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
const EXPONENTS: &[f64] = &[1.0, 1.5, 2.0, 2.5, 3.0];
const SCALES: &[f64] = &[0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LawFamily {
    InversePower,
    ScreenedInversePower,
    Saturating,
}

impl LawFamily {
    fn parameter_count(self) -> usize {
        match self {
            Self::InversePower => 2,
            Self::ScreenedInversePower | Self::Saturating => 3,
        }
    }
}

/// A dimensionless scalar law used by the v0 reference suite.
///
/// `scale` is ignored by `InversePower`; it remains present so that the
/// candidate descriptor has one simple serialized shape across families.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Law {
    pub family: LawFamily,
    pub amplitude: f64,
    pub exponent: f64,
    pub scale: f64,
}

impl Law {
    pub fn predict(self, x: f64) -> Option<f64> {
        if !x.is_finite()
            || x <= 0.0
            || !self.amplitude.is_finite()
            || self.amplitude <= 0.0
            || !self.exponent.is_finite()
            || self.exponent <= 0.0
        {
            return None;
        }
        if self.family != LawFamily::InversePower && (!self.scale.is_finite() || self.scale <= 0.0)
        {
            return None;
        }

        let value = match self.family {
            LawFamily::InversePower => self.amplitude / x.powf(self.exponent),
            LawFamily::ScreenedInversePower => {
                self.amplitude * (-x / self.scale).exp() / x.powf(self.exponent)
            }
            LawFamily::Saturating => self.amplitude / (1.0 + (x / self.scale).powf(self.exponent)),
        };
        value.is_finite().then_some(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Observation {
    pub x: f64,
    pub response: f64,
    pub declared_sigma: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryPolicy {
    FixedGrid,
    SeededRandom,
}

impl QueryPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FixedGrid => "fixed_grid",
            Self::SeededRandom => "seeded_random",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionDisposition {
    DecisiveWithinReferenceGrammar,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CaseResult {
    pub world_id: String,
    pub policy: QueryPolicy,
    pub expected_family: LawFamily,
    pub selected_family: LawFamily,
    pub family_recovered: bool,
    pub selection_disposition: SelectionDisposition,
    /// BIC-like score gap between the best and second-best family; heuristic only.
    pub delta_bic_to_runner_up: f64,
    pub query_count: usize,
    pub selected_law: Law,
    /// Normalized RMSE against noisy discovery observations.
    pub discovery_nrmse: f64,
    /// Normalized RMSE against exact, noiseless outputs in the held-out range.
    pub heldout_nrmse: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PolicySummary {
    pub policy: QueryPolicy,
    pub case_count: usize,
    /// Selected family matches the hidden reference family, regardless of ambiguity label.
    pub family_recovery_rate: f64,
    /// Fraction of cases both correctly matched and decisive under the declared BIC heuristic.
    pub decisive_family_recovery_rate: f64,
    pub mean_heldout_nrmse: f64,
    pub median_heldout_nrmse: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SelectionBottleneckCase {
    pub world_id: String,
    /// Zero-based deterministic replication index; matches policies at paired seed.
    pub replication_index: usize,
    pub policy: QueryPolicy,
    pub candidates_per_family: usize,
    pub candidate_count: usize,
    pub expected_family: LawFamily,
    pub selected_family: LawFamily,
    pub family_recovered: bool,
    pub selection_disposition: SelectionDisposition,
    pub delta_bic_to_runner_up: f64,
    pub selected_training_nrmse: f64,
    pub selected_heldout_nrmse: f64,
    /// Best held-out error among the exact same candidate pool; evaluator-only oracle.
    pub oracle_pool_heldout_nrmse: f64,
    /// Nonnegative gap from the training-selected candidate to the evaluator-only pool oracle.
    pub selection_regret_nrmse: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SelectionBottleneckSummary {
    pub policy: QueryPolicy,
    pub candidates_per_family: usize,
    pub case_count: usize,
    pub family_recovery_rate: f64,
    pub mean_selected_heldout_nrmse: f64,
    pub mean_oracle_pool_heldout_nrmse: f64,
    pub mean_selection_regret_nrmse: f64,
    /// Descriptive held-out-minus-training NRMSE; the two domains/target scales differ,
    /// so this is not a distribution-matched generalization-gap estimate.
    pub mean_extrapolation_minus_training_nrmse: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReportBody {
    pub schema: String,
    /// False by design: this first reference corpus is open source and source-visible.
    pub blind_evaluation: bool,
    pub corpus_sha256: String,
    /// Digest of the full 9-world corpus used by the candidate-selection diagnostic.
    pub selection_diagnostic_corpus_sha256: String,
    /// Frozen seed protocol for repeated candidate-selection measurements.
    pub selection_replication_count: usize,
    pub selection_seed_schedule: String,
    /// Compiler identity captured by the build script.
    pub rustc_identity: String,
    /// Cargo identity captured by the build script.
    pub cargo_identity: String,
    /// Git HEAD at build time; source-tree cleanliness is recorded separately.
    pub source_commit_at_build: String,
    pub source_tree_state_at_build: String,
    /// SHA-256 over Cargo.toml, build.rs and Rust source files used by this harness.
    pub evaluator_source_sha256: String,
    /// SHA-256 over the nested Cargo.lock, or "missing" when unavailable.
    pub dependency_lock_sha256: String,
    pub training_query_budget_per_case: usize,
    pub holdout_point_count_per_case: usize,
    pub policies: Vec<PolicySummary>,
    pub cases: Vec<CaseResult>,
    /// Controlled candidate-pool-size sweep. The oracle metric is evaluator-only.
    pub selection_bottleneck_cases: Vec<SelectionBottleneckCase>,
    pub selection_bottleneck_summaries: Vec<SelectionBottleneckSummary>,
    pub claim_boundary: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BenchmarkReport {
    pub body: ReportBody,
    /// SHA-256 of the compact deterministic JSON serialization of `body`.
    pub receipt_sha256: String,
}

impl BenchmarkReport {
    pub fn verify_receipt(&self) -> bool {
        let Ok(bytes) = serde_json::to_vec(&self.body) else {
            return false;
        };
        self.receipt_sha256 == sha256_hex(&bytes)
            && self.body.corpus_sha256 == reference_corpus_sha256()
            && self.body.selection_diagnostic_corpus_sha256 == selection_diagnostic_corpus_sha256()
            && self.body.selection_replication_count == SELECTION_REPLICATION_COUNT
            && self.body.selection_seed_schedule == SELECTION_SEED_SCHEDULE
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryError {
    NonFiniteInput,
    OutsideDiscoveryDomain,
    InvalidReferenceOutput,
}

impl fmt::Display for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteInput => write!(f, "query x must be finite"),
            Self::OutsideDiscoveryDomain => {
                write!(f, "query lies outside the declared discovery domain")
            }
            Self::InvalidReferenceOutput => write!(f, "reference law returned an invalid output"),
        }
    }
}

impl std::error::Error for QueryError {}

/// The only agent-facing surface of a reference world.
///
/// The truth is private to this module and is not returned by the query API.
/// Because this is an open-source reference fixture, this encapsulation is not
/// a security boundary: a real blind evaluation needs evaluator-owned worlds
/// or a separate process/service boundary.
#[derive(Clone)]
pub struct ReferenceWorld {
    id: &'static str,
    truth: Law,
    seed: u64,
}

impl ReferenceWorld {
    pub fn id(&self) -> &'static str {
        self.id
    }

    pub fn query(&self, x: f64, observation_id: u64) -> Result<Observation, QueryError> {
        if !x.is_finite() {
            return Err(QueryError::NonFiniteInput);
        }
        if !(TRAIN_X_MIN..=TRAIN_X_MAX).contains(&x) {
            return Err(QueryError::OutsideDiscoveryDomain);
        }
        let expected = self
            .truth
            .predict(x)
            .ok_or(QueryError::InvalidReferenceOutput)?;
        let noise_seed = self.seed
            ^ observation_id.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ x.to_bits().rotate_left(17);
        let noise = standard_normal(noise_seed);
        let sigma = RELATIVE_NOISE * expected.abs().max(0.05);
        let response = expected + sigma * noise;
        if !response.is_finite() {
            return Err(QueryError::InvalidReferenceOutput);
        }
        Ok(Observation {
            x,
            response,
            declared_sigma: RELATIVE_NOISE * response.abs().max(0.05),
        })
    }
}

#[derive(Serialize)]
struct CorpusEntry {
    id: &'static str,
    truth: Law,
    seed: u64,
    train_min: f64,
    train_max: f64,
    holdout_min: f64,
    holdout_max: f64,
}

fn reference_worlds() -> Vec<ReferenceWorld> {
    vec![
        ReferenceWorld {
            id: "world-inverse-power-a",
            truth: Law {
                family: LawFamily::InversePower,
                amplitude: 2.0,
                exponent: 2.0,
                scale: 1.5,
            },
            seed: 0x5048_5953_0000_0001,
        },
        ReferenceWorld {
            id: "world-screened-power-b",
            truth: Law {
                family: LawFamily::ScreenedInversePower,
                amplitude: 2.5,
                exponent: 2.0,
                scale: 2.5,
            },
            seed: 0x5048_5953_0000_0002,
        },
        ReferenceWorld {
            id: "world-saturating-c",
            truth: Law {
                family: LawFamily::Saturating,
                amplitude: 1.5,
                exponent: 2.0,
                scale: 1.5,
            },
            seed: 0x5048_5953_0000_0003,
        },
    ]
}

/// Additional source-visible worlds whose true parameters lie between the
/// grid-search values. These holdout-transfer fixtures test whether the
/// selection curve is robust to parameter values the grid cannot represent
/// exactly; they do not constitute a blind benchmark.
fn selection_diagnostic_worlds() -> Vec<ReferenceWorld> {
    let mut worlds = reference_worlds();
    worlds.extend([
        ReferenceWorld {
            id: "transfer-inverse-power-off-grid-a",
            truth: Law {
                family: LawFamily::InversePower,
                amplitude: 1.35,
                exponent: 1.8,
                scale: 1.5,
            },
            seed: 0x5048_5953_0000_0004,
        },
        ReferenceWorld {
            id: "transfer-inverse-power-off-grid-b",
            truth: Law {
                family: LawFamily::InversePower,
                amplitude: 2.35,
                exponent: 2.7,
                scale: 2.0,
            },
            seed: 0x5048_5953_0000_0005,
        },
        ReferenceWorld {
            id: "transfer-screened-power-off-grid-a",
            truth: Law {
                family: LawFamily::ScreenedInversePower,
                amplitude: 2.2,
                exponent: 1.8,
                scale: 2.25,
            },
            seed: 0x5048_5953_0000_0006,
        },
        ReferenceWorld {
            id: "transfer-screened-power-off-grid-b",
            truth: Law {
                family: LawFamily::ScreenedInversePower,
                amplitude: 1.65,
                exponent: 2.7,
                scale: 1.25,
            },
            seed: 0x5048_5953_0000_0007,
        },
        ReferenceWorld {
            id: "transfer-saturating-off-grid-a",
            truth: Law {
                family: LawFamily::Saturating,
                amplitude: 1.35,
                exponent: 1.8,
                scale: 2.25,
            },
            seed: 0x5048_5953_0000_0008,
        },
        ReferenceWorld {
            id: "transfer-saturating-off-grid-b",
            truth: Law {
                family: LawFamily::Saturating,
                amplitude: 2.65,
                exponent: 2.7,
                scale: 1.25,
            },
            seed: 0x5048_5953_0000_0009,
        },
    ]);
    worlds
}

/// Return the source-visible reference worlds through their query-only API.
///
/// This is useful for smoke tests and local agent development. It is not a
/// blind evaluation interface because the corpus implementation is open source.
pub fn open_reference_worlds() -> Vec<ReferenceWorld> {
    reference_worlds()
}

fn reference_corpus_manifest() -> Vec<CorpusEntry> {
    reference_worlds()
        .into_iter()
        .map(|world| CorpusEntry {
            id: world.id,
            truth: world.truth,
            seed: world.seed,
            train_min: TRAIN_X_MIN,
            train_max: TRAIN_X_MAX,
            holdout_min: HOLDOUT_X_MIN,
            holdout_max: HOLDOUT_X_MAX,
        })
        .collect()
}

pub fn reference_corpus_sha256() -> String {
    let bytes = serde_json::to_vec(&reference_corpus_manifest())
        .expect("the fixed reference corpus must serialize deterministically");
    sha256_hex(&bytes)
}

fn selection_diagnostic_corpus_manifest() -> Vec<CorpusEntry> {
    selection_diagnostic_worlds()
        .into_iter()
        .map(|world| CorpusEntry {
            id: world.id,
            truth: world.truth,
            seed: world.seed,
            train_min: TRAIN_X_MIN,
            train_max: TRAIN_X_MAX,
            holdout_min: HOLDOUT_X_MIN,
            holdout_max: HOLDOUT_X_MAX,
        })
        .collect()
}

pub fn selection_diagnostic_corpus_sha256() -> String {
    let bytes = serde_json::to_vec(&selection_diagnostic_corpus_manifest())
        .expect("the selection diagnostic corpus must serialize deterministically");
    sha256_hex(&bytes)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn uniform01(state: &mut u64) -> f64 {
    let bits = splitmix64(state) >> 11;
    (bits as f64) / ((1_u64 << 53) as f64)
}

fn standard_normal(seed: u64) -> f64 {
    let mut state = seed;
    let u1 = uniform01(&mut state).max(1.0e-12);
    let u2 = uniform01(&mut state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn query_points(policy: QueryPolicy, seed: u64) -> Vec<f64> {
    let n = TRAIN_QUERY_BUDGET;
    let mut points = match policy {
        QueryPolicy::FixedGrid => (0..n)
            .map(|i| {
                TRAIN_X_MIN + (i as f64 + 1.0) * (TRAIN_X_MAX - TRAIN_X_MIN) / (n as f64 + 1.0)
            })
            .collect::<Vec<_>>(),
        QueryPolicy::SeededRandom => {
            let mut state = seed ^ 0x4558_5045_5249_4d45;
            (0..n)
                .map(|_| TRAIN_X_MIN + uniform01(&mut state) * (TRAIN_X_MAX - TRAIN_X_MIN))
                .collect::<Vec<_>>()
        }
    };
    points.sort_by(f64::total_cmp);
    points
}

fn candidate_grid(family: LawFamily) -> Vec<Law> {
    let scales: &[f64] = match family {
        LawFamily::InversePower => &[1.0],
        LawFamily::ScreenedInversePower | LawFamily::Saturating => SCALES,
    };
    let mut candidates = Vec::new();
    for &amplitude in AMPLITUDES {
        for &exponent in EXPONENTS {
            for &scale in scales {
                candidates.push(Law {
                    family,
                    amplitude,
                    exponent,
                    scale,
                });
            }
        }
    }
    candidates
}

fn normalized_rmse(predicted: &[f64], observed: &[f64]) -> Option<f64> {
    if predicted.is_empty() || predicted.len() != observed.len() {
        return None;
    }
    if predicted
        .iter()
        .chain(observed.iter())
        .any(|value| !value.is_finite())
    {
        return None;
    }
    let mse = predicted
        .iter()
        .zip(observed)
        .map(|(p, y)| (p - y).powi(2))
        .sum::<f64>()
        / predicted.len() as f64;
    let scale = (observed.iter().map(|y| y * y).sum::<f64>() / observed.len() as f64).sqrt();
    (mse.is_finite() && scale.is_finite() && scale > 0.0).then_some(mse.sqrt() / scale.max(1.0e-12))
}

fn bic_for(law: Law, observations: &[Observation]) -> Option<f64> {
    if observations.is_empty() {
        return None;
    }
    // Gaussian -2 log-likelihood up to the candidate-independent constant
    // ln(2*pi), followed by the BIC parameter-count penalty.
    let mut negative_two_log_likelihood = 0.0;
    for observation in observations {
        if !observation.declared_sigma.is_finite() || observation.declared_sigma <= 0.0 {
            return None;
        }
        let predicted = law.predict(observation.x)?;
        let residual = (predicted - observation.response) / observation.declared_sigma;
        negative_two_log_likelihood += residual * residual + 2.0 * observation.declared_sigma.ln();
    }
    let n = observations.len() as f64;
    let score = negative_two_log_likelihood + law.family.parameter_count() as f64 * n.ln();
    score.is_finite().then_some(score)
}

#[derive(Debug, Clone, Copy)]
struct FitOutcome {
    selected: Law,
    delta_bic_to_runner_up: f64,
    disposition: SelectionDisposition,
}

/// Exhaustive, deterministic grid-search baseline using BIC-like complexity
/// penalization. The best candidate is found separately per family so the
/// ambiguity margin is between structural families, not neighbouring
/// parameter-grid points. This remains a bounded baseline, not Ramanujan or
/// open-ended theory generation.
fn disposition_from_delta_bic(delta: f64) -> SelectionDisposition {
    if delta.is_finite() && delta >= BIC_DECISIVE_DELTA {
        SelectionDisposition::DecisiveWithinReferenceGrammar
    } else {
        SelectionDisposition::Indeterminate
    }
}

fn fit_grid_baseline(observations: &[Observation]) -> Option<FitOutcome> {
    if observations.len() < 3 {
        return None;
    }
    let mut family_best: Vec<(f64, Law)> = Vec::new();
    for family in [
        LawFamily::InversePower,
        LawFamily::ScreenedInversePower,
        LawFamily::Saturating,
    ] {
        let mut best_for_family: Option<(f64, Law)> = None;
        for law in candidate_grid(family) {
            let Some(score) = bic_for(law, observations) else {
                continue;
            };
            if best_for_family
                .as_ref()
                .is_none_or(|(best_score, _)| score < *best_score)
            {
                best_for_family = Some((score, law));
            }
        }
        family_best.push(best_for_family?);
    }
    family_best.sort_by(|a, b| a.0.total_cmp(&b.0));
    let delta = (family_best[1].0 - family_best[0].0).max(0.0);
    let disposition = disposition_from_delta_bic(delta);
    Some(FitOutcome {
        selected: family_best[0].1,
        delta_bic_to_runner_up: delta,
        disposition,
    })
}

fn evaluate_case(world: &ReferenceWorld, policy: QueryPolicy) -> Option<CaseResult> {
    let points = query_points(policy, world.seed);
    let observations = points
        .iter()
        .enumerate()
        .map(|(i, x)| world.query(*x, i as u64))
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    let fit = fit_grid_baseline(&observations)?;
    let selected = fit.selected;

    let discovery_prediction = observations
        .iter()
        .map(|observation| selected.predict(observation.x))
        .collect::<Option<Vec<_>>>()?;
    let discovery_response = observations
        .iter()
        .map(|observation| observation.response)
        .collect::<Vec<_>>();
    let discovery_nrmse = normalized_rmse(&discovery_prediction, &discovery_response)?;

    let mut heldout_prediction = Vec::with_capacity(HOLDOUT_POINT_COUNT);
    let mut heldout_truth = Vec::with_capacity(HOLDOUT_POINT_COUNT);
    for i in 0..HOLDOUT_POINT_COUNT {
        let x = HOLDOUT_X_MIN
            + i as f64 * (HOLDOUT_X_MAX - HOLDOUT_X_MIN)
                / (HOLDOUT_POINT_COUNT.saturating_sub(1) as f64);
        heldout_prediction.push(selected.predict(x)?);
        heldout_truth.push(world.truth.predict(x)?);
    }
    let heldout_nrmse = normalized_rmse(&heldout_prediction, &heldout_truth)?;

    Some(CaseResult {
        world_id: world.id.to_owned(),
        policy,
        expected_family: world.truth.family,
        selected_family: selected.family,
        family_recovered: selected.family == world.truth.family,
        selection_disposition: fit.disposition,
        delta_bic_to_runner_up: fit.delta_bic_to_runner_up,
        query_count: observations.len(),
        selected_law: selected,
        discovery_nrmse,
        heldout_nrmse,
    })
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

fn summarize(policy: QueryPolicy, rows: &[CaseResult]) -> Option<PolicySummary> {
    let selected = rows
        .iter()
        .filter(|row| row.policy == policy)
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return None;
    }
    let recovered = selected.iter().filter(|row| row.family_recovered).count();
    let decisive_recovered = selected
        .iter()
        .filter(|row| {
            row.family_recovered
                && row.selection_disposition == SelectionDisposition::DecisiveWithinReferenceGrammar
        })
        .count();
    let mut errors = selected
        .iter()
        .map(|row| row.heldout_nrmse)
        .collect::<Vec<_>>();
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let median = median(&mut errors)?;
    Some(PolicySummary {
        policy,
        case_count: selected.len(),
        family_recovery_rate: recovered as f64 / selected.len() as f64,
        decisive_family_recovery_rate: decisive_recovered as f64 / selected.len() as f64,
        mean_heldout_nrmse: mean,
        median_heldout_nrmse: median,
    })
}

fn candidate_pool(family: LawFamily, count: usize, seed: u64) -> Vec<Law> {
    let mut candidates = candidate_grid(family);
    let family_salt = match family {
        LawFamily::InversePower => 0x4341_4e44_0000_0001,
        LawFamily::ScreenedInversePower => 0x4341_4e44_0000_0002,
        LawFamily::Saturating => 0x4341_4e44_0000_0003,
    };
    // Shuffle once per world/family, then take prefixes. Larger pools are
    // nested, so their evaluator-side oracle must never get worse.
    let mut state = seed ^ family_salt;
    for index in (1..candidates.len()).rev() {
        let other = (splitmix64(&mut state) % (index as u64 + 1)) as usize;
        candidates.swap(index, other);
    }
    candidates.truncate(count.min(candidates.len()));
    candidates
}

fn heldout_nrmse_for_law(world: &ReferenceWorld, law: Law) -> Option<f64> {
    let mut predictions = Vec::with_capacity(HOLDOUT_POINT_COUNT);
    let mut truth = Vec::with_capacity(HOLDOUT_POINT_COUNT);
    for i in 0..HOLDOUT_POINT_COUNT {
        let x = HOLDOUT_X_MIN
            + i as f64 * (HOLDOUT_X_MAX - HOLDOUT_X_MIN)
                / (HOLDOUT_POINT_COUNT.saturating_sub(1) as f64);
        predictions.push(law.predict(x)?);
        truth.push(world.truth.predict(x)?);
    }
    normalized_rmse(&predictions, &truth)
}

fn training_nrmse_for_law(law: Law, observations: &[Observation]) -> Option<f64> {
    let predicted = observations
        .iter()
        .map(|observation| law.predict(observation.x))
        .collect::<Option<Vec<_>>>()?;
    let observed = observations
        .iter()
        .map(|observation| observation.response)
        .collect::<Vec<_>>();
    normalized_rmse(&predicted, &observed)
}

fn selection_replica(world: &ReferenceWorld, replication_index: usize) -> ReferenceWorld {
    let mut state = world.seed
        ^ SELECTION_SEED_SALT
        ^ (replication_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    ReferenceWorld {
        id: world.id,
        truth: world.truth,
        seed: splitmix64(&mut state),
    }
}

fn evaluate_selection_bottleneck(
    world: &ReferenceWorld,
    policy: QueryPolicy,
    candidates_per_family: usize,
    replication_index: usize,
) -> Option<SelectionBottleneckCase> {
    let points = query_points(policy, world.seed);
    let observations = points
        .iter()
        .enumerate()
        .map(|(i, x)| world.query(*x, i as u64))
        .collect::<Result<Vec<_>, _>>()
        .ok()?;

    // Keep proposal coverage and ranking quality separate: every structural
    // family receives the same candidate budget, but the parameters are a
    // deterministic sampled subset of that family's declared grid.
    let mut family_best: Vec<(f64, Law, f64, f64)> = Vec::new();
    let mut candidate_count = 0usize;
    let mut oracle_pool_heldout_nrmse = f64::INFINITY;
    for family in [
        LawFamily::InversePower,
        LawFamily::ScreenedInversePower,
        LawFamily::Saturating,
    ] {
        let pool = candidate_pool(family, candidates_per_family, world.seed);
        candidate_count += pool.len();
        let mut best_for_family: Option<(f64, Law, f64, f64)> = None;
        for law in pool {
            let Some(score) = bic_for(law, &observations) else {
                continue;
            };
            let Some(training_error) = training_nrmse_for_law(law, &observations) else {
                continue;
            };
            let Some(heldout_error) = heldout_nrmse_for_law(world, law) else {
                continue;
            };
            oracle_pool_heldout_nrmse = oracle_pool_heldout_nrmse.min(heldout_error);
            if best_for_family
                .as_ref()
                .is_none_or(|(best_score, _, _, _)| score < *best_score)
            {
                best_for_family = Some((score, law, training_error, heldout_error));
            }
        }
        family_best.push(best_for_family?);
    }
    family_best.sort_by(|left, right| left.0.total_cmp(&right.0));
    let delta_bic_to_runner_up = (family_best[1].0 - family_best[0].0).max(0.0);
    let (_, selected_law, selected_training_nrmse, selected_heldout_nrmse) = family_best[0];
    let selection_regret_nrmse = (selected_heldout_nrmse - oracle_pool_heldout_nrmse).max(0.0);

    Some(SelectionBottleneckCase {
        world_id: world.id.to_owned(),
        replication_index,
        policy,
        candidates_per_family,
        candidate_count,
        expected_family: world.truth.family,
        selected_family: selected_law.family,
        family_recovered: selected_law.family == world.truth.family,
        selection_disposition: disposition_from_delta_bic(delta_bic_to_runner_up),
        delta_bic_to_runner_up,
        selected_training_nrmse,
        selected_heldout_nrmse,
        oracle_pool_heldout_nrmse,
        selection_regret_nrmse,
    })
}

fn summarize_selection_bottleneck(
    policy: QueryPolicy,
    candidates_per_family: usize,
    rows: &[SelectionBottleneckCase],
) -> Option<SelectionBottleneckSummary> {
    let selected = rows
        .iter()
        .filter(|row| row.policy == policy && row.candidates_per_family == candidates_per_family)
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return None;
    }
    let count = selected.len() as f64;
    let recovered = selected.iter().filter(|row| row.family_recovered).count();
    Some(SelectionBottleneckSummary {
        policy,
        candidates_per_family,
        case_count: selected.len(),
        family_recovery_rate: recovered as f64 / count,
        mean_selected_heldout_nrmse: selected
            .iter()
            .map(|row| row.selected_heldout_nrmse)
            .sum::<f64>()
            / count,
        mean_oracle_pool_heldout_nrmse: selected
            .iter()
            .map(|row| row.oracle_pool_heldout_nrmse)
            .sum::<f64>()
            / count,
        mean_selection_regret_nrmse: selected
            .iter()
            .map(|row| row.selection_regret_nrmse)
            .sum::<f64>()
            / count,
        mean_extrapolation_minus_training_nrmse: selected
            .iter()
            .map(|row| row.selected_heldout_nrmse - row.selected_training_nrmse)
            .sum::<f64>()
            / count,
    })
}

pub fn run_reference_benchmark() -> BenchmarkReport {
    let worlds = reference_worlds();
    let mut rows = Vec::new();
    for policy in [QueryPolicy::FixedGrid, QueryPolicy::SeededRandom] {
        for world in &worlds {
            rows.push(evaluate_case(world, policy).expect("fixed reference case must evaluate"));
        }
    }
    let policies = [QueryPolicy::FixedGrid, QueryPolicy::SeededRandom]
        .into_iter()
        .map(|policy| summarize(policy, &rows).expect("each policy has reference cases"))
        .collect();

    let selection_worlds = selection_diagnostic_worlds();
    let mut selection_bottleneck_cases = Vec::new();
    for replication_index in 0..SELECTION_REPLICATION_COUNT {
        for &candidates_per_family in CANDIDATES_PER_FAMILY {
            for policy in [QueryPolicy::FixedGrid, QueryPolicy::SeededRandom] {
                for world in &selection_worlds {
                    let replica = selection_replica(world, replication_index);
                    selection_bottleneck_cases.push(
                        evaluate_selection_bottleneck(
                            &replica,
                            policy,
                            candidates_per_family,
                            replication_index,
                        )
                        .expect("fixed replicated candidate-selection diagnostic must evaluate"),
                    );
                }
            }
        }
    }
    let mut selection_bottleneck_summaries = Vec::new();
    for &candidates_per_family in CANDIDATES_PER_FAMILY {
        for policy in [QueryPolicy::FixedGrid, QueryPolicy::SeededRandom] {
            selection_bottleneck_summaries.push(
                summarize_selection_bottleneck(
                    policy,
                    candidates_per_family,
                    &selection_bottleneck_cases,
                )
                .expect("each candidate pool size has reference cases"),
            );
        }
    }

    let body = ReportBody {
        schema: REPORT_SCHEMA.to_owned(),
        blind_evaluation: false,
        corpus_sha256: reference_corpus_sha256(),
        selection_diagnostic_corpus_sha256: selection_diagnostic_corpus_sha256(),
        selection_replication_count: SELECTION_REPLICATION_COUNT,
        selection_seed_schedule: SELECTION_SEED_SCHEDULE.to_owned(),
        rustc_identity: option_env!("OPEN_WORLD_RUSTC_ID")
            .unwrap_or("unavailable")
            .to_owned(),
        cargo_identity: option_env!("OPEN_WORLD_CARGO_ID")
            .unwrap_or("unavailable")
            .to_owned(),
        source_commit_at_build: option_env!("OPEN_WORLD_SOURCE_COMMIT")
            .unwrap_or("unavailable")
            .to_owned(),
        source_tree_state_at_build: option_env!("OPEN_WORLD_SOURCE_TREE_STATE")
            .unwrap_or("unavailable")
            .to_owned(),
        evaluator_source_sha256: option_env!("OPEN_WORLD_SOURCE_SHA256")
            .unwrap_or("unavailable")
            .to_owned(),
        dependency_lock_sha256: option_env!("OPEN_WORLD_LOCK_SHA256")
            .unwrap_or("missing")
            .to_owned(),
        training_query_budget_per_case: TRAIN_QUERY_BUDGET,
        holdout_point_count_per_case: HOLDOUT_POINT_COUNT,
        policies,
        cases: rows,
        selection_bottleneck_cases,
        selection_bottleneck_summaries,
        claim_boundary: "Open, source-visible synthetic reference suite and bounded grid-search baseline only; not a blind SOTA result, not evidence of new real-world physics, and not production solver qualification.".to_owned(),
    };
    let bytes = serde_json::to_vec(&body).expect("report body must serialize deterministically");
    BenchmarkReport {
        body,
        receipt_sha256: sha256_hex(&bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytical_values_match_hand_calculation() {
        let inverse = Law {
            family: LawFamily::InversePower,
            amplitude: 2.0,
            exponent: 2.0,
            scale: 1.0,
        };
        assert!((inverse.predict(2.0).unwrap() - 0.5).abs() < 1.0e-12);

        let screened = Law {
            family: LawFamily::ScreenedInversePower,
            amplitude: 2.0,
            exponent: 2.0,
            scale: 2.0,
        };
        assert!(
            (screened.predict(2.0).unwrap() - std::f64::consts::E.recip() / 2.0).abs() < 1.0e-12
        );

        let saturating = Law {
            family: LawFamily::Saturating,
            amplitude: 2.0,
            exponent: 2.0,
            scale: 2.0,
        };
        assert!((saturating.predict(2.0).unwrap() - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        let law = Law {
            family: LawFamily::InversePower,
            amplitude: 1.0,
            exponent: 2.0,
            scale: 1.0,
        };
        assert!(law.predict(0.0).is_none());
        assert!(law.predict(f64::NAN).is_none());
        assert!(law.predict(f64::INFINITY).is_none());
        assert!(
            Law {
                amplitude: f64::NAN,
                ..law
            }
            .predict(1.0)
            .is_none()
        );
        assert!(
            Law {
                exponent: 0.0,
                ..law
            }
            .predict(1.0)
            .is_none()
        );
        assert!(
            Law {
                scale: f64::NAN,
                ..law
            }
            .predict(1.0)
            .is_some()
        );
        assert!(
            Law {
                family: LawFamily::ScreenedInversePower,
                scale: f64::NAN,
                ..law
            }
            .predict(1.0)
            .is_none()
        );
    }

    #[test]
    fn query_api_rejects_nonfinite_and_holdout_domain_inputs() {
        let world = open_reference_worlds().remove(0);
        assert_eq!(world.query(f64::NAN, 0), Err(QueryError::NonFiniteInput));
        assert_eq!(
            world.query(HOLDOUT_X_MIN, 1),
            Err(QueryError::OutsideDiscoveryDomain)
        );
        assert_eq!(
            world.query(TRAIN_X_MIN - 0.1, 2),
            Err(QueryError::OutsideDiscoveryDomain)
        );

        let first = world.query(1.5, 99).unwrap();
        let repeated = world.query(1.5, 99).unwrap();
        assert_eq!(first, repeated);
        assert!(first.response.is_finite());
        assert!(first.declared_sigma.is_finite() && first.declared_sigma > 0.0);
    }

    #[test]
    fn noiseless_grid_search_recovers_each_reference_family() {
        for world in reference_worlds() {
            let points = query_points(QueryPolicy::FixedGrid, world.seed);
            let observations = points
                .into_iter()
                .map(|x| Observation {
                    x,
                    response: world.truth.predict(x).unwrap(),
                    declared_sigma: RELATIVE_NOISE
                        * world.truth.predict(x).unwrap().abs().max(0.05),
                })
                .collect::<Vec<_>>();
            let fitted = fit_grid_baseline(&observations).unwrap();
            assert_eq!(
                fitted.selected.family, world.truth.family,
                "failed to recover reference family for {}",
                world.id
            );
        }
    }

    #[test]
    fn ambiguity_gate_fails_closed_at_nonfinite_or_small_score_gaps() {
        assert_eq!(
            disposition_from_delta_bic(BIC_DECISIVE_DELTA - f64::EPSILON),
            SelectionDisposition::Indeterminate
        );
        assert_eq!(
            disposition_from_delta_bic(BIC_DECISIVE_DELTA),
            SelectionDisposition::DecisiveWithinReferenceGrammar
        );
        assert_eq!(
            disposition_from_delta_bic(f64::NAN),
            SelectionDisposition::Indeterminate
        );
        assert_eq!(
            disposition_from_delta_bic(f64::INFINITY),
            SelectionDisposition::Indeterminate
        );
    }

    #[test]
    fn model_selection_rejects_malformed_observation_contracts() {
        let valid = Observation {
            x: 1.0,
            response: 1.0,
            declared_sigma: 0.1,
        };
        let valid_observations = vec![valid; 12];
        assert!(fit_grid_baseline(&valid_observations).is_some());

        let mut zero_sigma = valid_observations.clone();
        zero_sigma[0].declared_sigma = 0.0;
        assert!(fit_grid_baseline(&zero_sigma).is_none());

        let mut nonfinite_response = valid_observations.clone();
        nonfinite_response[0].response = f64::NAN;
        assert!(fit_grid_baseline(&nonfinite_response).is_none());

        let mut invalid_domain = valid_observations;
        invalid_domain[0].x = 0.0;
        assert!(fit_grid_baseline(&invalid_domain).is_none());
    }

    #[test]
    fn noisy_reference_suite_recovers_all_identifiable_families_decisively() {
        let report = run_reference_benchmark();
        assert_eq!(report.body.cases.len(), 6);
        for row in &report.body.cases {
            assert!(
                row.family_recovered,
                "wrong family for {} under {:?}: selected {:?}, expected {:?}",
                row.world_id, row.policy, row.selected_family, row.expected_family
            );
            assert_eq!(
                row.selection_disposition,
                SelectionDisposition::DecisiveWithinReferenceGrammar,
                "ambiguous model selection for {} under {:?}",
                row.world_id,
                row.policy
            );
            assert!(
                row.heldout_nrmse < 1.0e-12,
                "held-out fit drift for {} under {:?}: {}",
                row.world_id,
                row.policy,
                row.heldout_nrmse
            );
        }
    }

    #[test]
    fn report_is_deterministic_and_receipt_verifies() {
        let first = run_reference_benchmark();
        let second = run_reference_benchmark();
        assert_eq!(first, second);
        assert!(first.verify_receipt());
        assert!(!first.body.blind_evaluation);
        assert_eq!(first.body.cases.len(), 6);
        assert!(!first.body.rustc_identity.is_empty());
        assert!(!first.body.cargo_identity.is_empty());
        assert!(!first.body.source_commit_at_build.is_empty());
        assert!(!first.body.source_tree_state_at_build.is_empty());
        assert!(!first.body.evaluator_source_sha256.is_empty());
        assert!(!first.body.dependency_lock_sha256.is_empty());
    }

    #[test]
    fn report_preserves_per_case_metrics_and_fixed_budgets() {
        let report = run_reference_benchmark();
        assert_eq!(
            report.body.training_query_budget_per_case,
            TRAIN_QUERY_BUDGET
        );
        assert_eq!(
            report.body.holdout_point_count_per_case,
            HOLDOUT_POINT_COUNT
        );
        assert_eq!(report.body.policies.len(), 2);
        for row in &report.body.cases {
            assert_eq!(row.query_count, TRAIN_QUERY_BUDGET);
            assert!(row.discovery_nrmse.is_finite());
            assert!(row.heldout_nrmse.is_finite());
            assert!(row.discovery_nrmse >= 0.0);
            assert!(row.heldout_nrmse >= 0.0);
            assert!(row.delta_bic_to_runner_up.is_finite());
            assert!(row.delta_bic_to_runner_up >= 0.0);
            assert_eq!(
                row.selection_disposition == SelectionDisposition::DecisiveWithinReferenceGrammar,
                row.delta_bic_to_runner_up >= BIC_DECISIVE_DELTA
            );
        }
    }

    #[test]
    fn candidate_selection_curve_separates_ranking_regret_from_pool_quality() {
        let first = run_reference_benchmark();
        let second = run_reference_benchmark();
        assert_eq!(first, second);
        assert!(first.verify_receipt());
        assert_eq!(first.body.selection_bottleneck_cases.len(), 72 * SELECTION_REPLICATION_COUNT);
        assert_eq!(first.body.selection_replication_count, SELECTION_REPLICATION_COUNT);
        assert_eq!(first.body.selection_seed_schedule, SELECTION_SEED_SCHEDULE);
        assert_eq!(first.body.selection_bottleneck_summaries.len(), 8);
        assert_eq!(
            first.body.selection_diagnostic_corpus_sha256,
            selection_diagnostic_corpus_sha256()
        );

        for row in &first.body.selection_bottleneck_cases {
            assert_eq!(row.candidate_count, row.candidates_per_family * 3);
            assert!(row.selected_training_nrmse.is_finite());
            assert!(row.selected_heldout_nrmse.is_finite());
            assert!(row.oracle_pool_heldout_nrmse.is_finite());
            assert!(row.selection_regret_nrmse.is_finite());
            assert!(row.selection_regret_nrmse >= 0.0);
            assert!(
                row.oracle_pool_heldout_nrmse <= row.selected_heldout_nrmse + 1.0e-12,
                "pool oracle must be no worse than the selected candidate"
            );
            assert!(
                (row.selection_regret_nrmse
                    - (row.selected_heldout_nrmse - row.oracle_pool_heldout_nrmse).max(0.0))
                .abs()
                    <= 1.0e-12
            );
        }

        // The pool is built from deterministic shuffled prefixes. More candidate
        // proposals may improve the evaluator-only oracle, but may not worsen it.
        for world in selection_diagnostic_worlds() {
            for replication_index in 0..SELECTION_REPLICATION_COUNT {
                for policy in [QueryPolicy::FixedGrid, QueryPolicy::SeededRandom] {
                    let mut previous_oracle = f64::INFINITY;
                    for &pool_size in CANDIDATES_PER_FAMILY {
                        let row = first
                            .body
                            .selection_bottleneck_cases
                            .iter()
                            .find(|row| {
                                row.world_id == world.id
                                    && row.replication_index == replication_index
                                    && row.policy == policy
                                    && row.candidates_per_family == pool_size
                            })
                            .expect("candidate-selection row exists");
                        assert!(
                            row.oracle_pool_heldout_nrmse <= previous_oracle + 1.0e-12,
                            "oracle error worsened as nested candidate pool grew: {} replication {}",
                            world.id,
                            replication_index
                        );
                        previous_oracle = row.oracle_pool_heldout_nrmse;
                    }
                }
            }
        }

        for summary in &first.body.selection_bottleneck_summaries {
            assert_eq!(summary.case_count, 9 * SELECTION_REPLICATION_COUNT);
            assert!(summary.family_recovery_rate.is_finite());
            assert!((0.0..=1.0).contains(&summary.family_recovery_rate));
            assert!(summary.mean_selected_heldout_nrmse.is_finite());
            assert!(summary.mean_oracle_pool_heldout_nrmse.is_finite());
            assert!(summary.mean_selection_regret_nrmse.is_finite());
            assert!(summary.mean_extrapolation_minus_training_nrmse.is_finite());
        }
    }

    #[test]
    fn receipt_detects_report_tampering() {
        let mut report = run_reference_benchmark();
        report.body.cases[0].heldout_nrmse += 0.01;
        assert!(!report.verify_receipt());
    }

    #[test]
    fn receipt_rejects_a_changed_selection_corpus_even_if_resealed() {
        let mut report = run_reference_benchmark();
        report.body.selection_diagnostic_corpus_sha256 = "00".to_owned();
        let body = serde_json::to_vec(&report.body).unwrap();
        report.receipt_sha256 = sha256_hex(&body);
        assert!(!report.verify_receipt());
    }

    #[test]
    fn receipt_rejects_a_changed_replication_schedule_even_if_resealed() {
        let mut report = run_reference_benchmark();
        report.body.selection_replication_count += 1;
        let body = serde_json::to_vec(&report.body).unwrap();
        report.receipt_sha256 = sha256_hex(&body);
        assert!(!report.verify_receipt());

        let mut report = run_reference_benchmark();
        report.body.selection_seed_schedule = "untracked-seed-schedule".to_owned();
        let body = serde_json::to_vec(&report.body).unwrap();
        report.receipt_sha256 = sha256_hex(&body);
        assert!(!report.verify_receipt());
    }
}
