// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reusable experimental-core primitives for architecture research.
//!
//! This module is deliberately architecture-agnostic. It provides provenance,
//! task-program identity, continual-learning metrics, paired uncertainty, and a
//! non-scalar claim ledger. It does not decide whether Symthaea (or any other
//! system) "wins" an experiment.

use crate::harness::analysis::bootstrap_ci_bca;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const EXPERIMENT_MANIFEST_SCHEMA_V1: &str = "symthaea.experiment-manifest/v1";
pub const TASK_PROGRAM_SCHEMA_V1: &str = "symthaea.task-program/v1";
pub const CLAIM_LEDGER_SCHEMA_V1: &str = "symthaea.claim-ledger/v1";

const MANIFEST_HASH_DOMAIN: &[u8] = b"symthaea.experiment-manifest.hash/v1";
const SEED_HASH_DOMAIN: &[u8] = b"symthaea.experiment-seeds.hash/v1";
const TASK_HASH_DOMAIN: &[u8] = b"symthaea.task-program.hash/v1";

fn canonical_hash<T: Serialize>(domain: &[u8], value: &T) -> Result<String, serde_json::Error> {
    let bytes = serde_json::to_vec(value)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&[0]);
    hasher.update(&bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

fn looks_like_blake3_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn unique(values: &[u64]) -> bool {
    let set: BTreeSet<u64> = values.iter().copied().collect();
    set.len() == values.len()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamNamespace {
    Dev,
    Confirm,
    Repl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TuningStatus {
    Exploratory,
    ConfirmatoryFirstUse,
    Replication,
    PostHoc,
}

/// Independent sources of randomness are tracked separately so repeated RNG
/// seeds over one hand-authored problem cannot be mistaken for task-family
/// generalization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeedManifest {
    pub environment_seeds: Vec<u64>,
    pub representation_seeds: Vec<u64>,
    pub learner_seeds: Vec<u64>,
    pub stream_seeds: Vec<u64>,
}

impl SeedManifest {
    pub fn validate(&self) -> Result<(), String> {
        if self.environment_seeds.is_empty() {
            return Err("at least one environment seed is required".into());
        }
        for (name, values) in [
            ("environment", &self.environment_seeds),
            ("representation", &self.representation_seeds),
            ("learner", &self.learner_seeds),
            ("stream", &self.stream_seeds),
        ] {
            if !unique(values) {
                return Err(format!("duplicate {name} seed"));
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, serde_json::Error> {
        canonical_hash(SEED_HASH_DOMAIN, self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentManifest {
    pub schema: String,
    pub experiment_id: String,
    pub experiment_version: String,
    pub code_revision: String,
    pub preregistration_hash: String,
    pub generator_hash: String,
    pub stream_namespace: StreamNamespace,
    pub tuning_status: TuningStatus,
    /// Whether behavioral metrics from this same stream namespace/version were
    /// already visible before this run began.
    pub prior_results_observed: bool,
    pub seed_manifest: SeedManifest,
    pub primary_hypothesis: String,
    pub primary_comparator: String,
    /// Smallest effect worth the added architectural complexity, in the natural
    /// units of the primary metric.
    pub sesoi: f64,
}

impl ExperimentManifest {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != EXPERIMENT_MANIFEST_SCHEMA_V1 {
            return Err(format!("unsupported manifest schema: {}", self.schema));
        }
        if self.experiment_id.trim().is_empty() || self.experiment_version.trim().is_empty() {
            return Err("experiment id/version must be non-empty".into());
        }
        if self.code_revision.trim().is_empty() {
            return Err("code revision must be recorded".into());
        }
        if !looks_like_blake3_hex(&self.preregistration_hash)
            || !looks_like_blake3_hex(&self.generator_hash)
        {
            return Err("preregistration/generator hashes must be 32-byte hex digests".into());
        }
        if !self.sesoi.is_finite() || self.sesoi < 0.0 {
            return Err("SESOI must be finite and non-negative".into());
        }
        if self.primary_hypothesis.trim().is_empty() || self.primary_comparator.trim().is_empty() {
            return Err("primary hypothesis/comparator must be frozen".into());
        }
        self.seed_manifest.validate()?;

        match self.tuning_status {
            TuningStatus::Exploratory if self.stream_namespace != StreamNamespace::Dev => {
                return Err("exploratory runs must use the DEV namespace".into());
            }
            TuningStatus::ConfirmatoryFirstUse
                if self.stream_namespace != StreamNamespace::Confirm
                    || self.prior_results_observed =>
            {
                return Err(
                    "confirmatory-first-use requires an untouched CONFIRM namespace".into(),
                );
            }
            TuningStatus::Replication
                if self.stream_namespace != StreamNamespace::Repl
                    || self.prior_results_observed =>
            {
                return Err("replication requires an untouched REPL namespace".into());
            }
            TuningStatus::PostHoc if !self.prior_results_observed => {
                return Err("post-hoc status requires previously observed results".into());
            }
            _ => {}
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, serde_json::Error> {
        canonical_hash(MANIFEST_HASH_DOMAIN, self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextVisibility {
    Explicit,
    Latent,
    TaskFree,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimingRegime {
    Uniform,
    Jittered,
    Bursty,
    MixedRate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum RuleExpr {
    Eq {
        left: String,
        right: String,
    },
    Ne {
        left: String,
        right: String,
    },
    Parity {
        factor: String,
        modulus: u32,
        remainder: u32,
    },
    Not {
        inner: Box<RuleExpr>,
    },
    And {
        terms: Vec<RuleExpr>,
    },
    Or {
        terms: Vec<RuleExpr>,
    },
    Xor {
        left: Box<RuleExpr>,
        right: Box<RuleExpr>,
    },
}

impl RuleExpr {
    pub fn depth(&self) -> usize {
        match self {
            Self::Eq { .. } | Self::Ne { .. } | Self::Parity { .. } => 1,
            Self::Not { inner } => 1 + inner.depth(),
            Self::And { terms } | Self::Or { terms } => {
                1 + terms.iter().map(Self::depth).max().unwrap_or(0)
            }
            Self::Xor { left, right } => 1 + left.depth().max(right.depth()),
        }
    }

    fn validate(&self) -> Result<(), String> {
        match self {
            Self::Eq { left, right } | Self::Ne { left, right } => {
                if left.trim().is_empty() || right.trim().is_empty() {
                    return Err("relation operands must be non-empty".into());
                }
            }
            Self::Parity {
                factor,
                modulus,
                remainder,
            } => {
                if factor.trim().is_empty() || *modulus == 0 || *remainder >= *modulus {
                    return Err("invalid parity rule".into());
                }
            }
            Self::Not { inner } => inner.validate()?,
            Self::And { terms } | Self::Or { terms } => {
                if terms.len() < 2 {
                    return Err("and/or rules require at least two terms".into());
                }
                for term in terms {
                    term.validate()?;
                }
            }
            Self::Xor { left, right } => {
                left.validate()?;
                right.validate()?;
            }
        }
        Ok(())
    }
}

/// Typed, hashable description of a generated experimental world. Execution is
/// intentionally separate: the program is the auditable identity/ground truth
/// contract that a generator/runtime must implement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskProgram {
    pub schema: String,
    pub program_id: String,
    pub family: String,
    pub rule: RuleExpr,
    pub context_visibility: ContextVisibility,
    pub timing_regime: TimingRegime,
    pub positive_examples: usize,
    pub negative_examples: usize,
    pub train_support: Vec<String>,
    pub eval_support: Vec<String>,
    pub oracle_digest: String,
}

impl TaskProgram {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != TASK_PROGRAM_SCHEMA_V1 {
            return Err(format!("unsupported task-program schema: {}", self.schema));
        }
        if self.program_id.trim().is_empty() || self.family.trim().is_empty() {
            return Err("program id/family must be non-empty".into());
        }
        self.rule.validate()?;
        if self.positive_examples == 0 || self.negative_examples == 0 {
            return Err("both classes must be represented".into());
        }
        if self.train_support.is_empty() || self.eval_support.is_empty() {
            return Err("train and evaluation support must both be explicit".into());
        }
        if !looks_like_blake3_hex(&self.oracle_digest) {
            return Err("oracle digest must be a 32-byte hex digest".into());
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, serde_json::Error> {
        canonical_hash(TASK_HASH_DOMAIN, self)
    }
}

/// Complete continual-learning performance matrix.
///
/// For `T` tasks the matrix has `T + 1` rows and `T` columns. Row 0 is the
/// pre-training baseline. Row `i + 1` is evaluation after training task `i`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceMatrix {
    rows: Vec<Vec<f64>>,
}

impl PerformanceMatrix {
    pub fn new(rows: Vec<Vec<f64>>) -> Result<Self, String> {
        if rows.is_empty() || rows[0].is_empty() {
            return Err("performance matrix must be non-empty".into());
        }
        let tasks = rows[0].len();
        if rows.len() != tasks + 1 {
            return Err("expected T+1 rows for T evaluation tasks".into());
        }
        if rows.iter().any(|row| row.len() != tasks) {
            return Err("performance matrix must be rectangular".into());
        }
        if rows
            .iter()
            .flatten()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err("performance entries must be finite probabilities".into());
        }
        Ok(Self { rows })
    }

    pub fn rows(&self) -> &[Vec<f64>] {
        &self.rows
    }

    pub fn tasks(&self) -> usize {
        self.rows[0].len()
    }

    pub fn final_accuracy(&self) -> f64 {
        let final_row = self.rows.last().expect("validated non-empty matrix");
        final_row.iter().sum::<f64>() / final_row.len() as f64
    }

    /// Mean accuracy over learned tasks after each training stage.
    pub fn average_incremental_accuracy(&self) -> f64 {
        let tasks = self.tasks();
        let mut total = 0.0;
        for stage in 1..=tasks {
            total += self.rows[stage][..stage].iter().sum::<f64>() / stage as f64;
        }
        total / tasks as f64
    }

    /// Final performance on prior tasks minus performance immediately after each
    /// task was learned. Negative values indicate backward interference.
    pub fn backward_transfer(&self) -> f64 {
        let tasks = self.tasks();
        if tasks <= 1 {
            return 0.0;
        }
        let final_row = &self.rows[tasks];
        (0..tasks - 1)
            .map(|task| final_row[task] - self.rows[task + 1][task])
            .sum::<f64>()
            / (tasks - 1) as f64
    }

    /// Performance on a future task after learning earlier tasks, relative to the
    /// untouched pre-training baseline for that future task.
    pub fn forward_transfer(&self) -> f64 {
        let tasks = self.tasks();
        if tasks <= 1 {
            return 0.0;
        }
        (1..tasks)
            .map(|task| self.rows[task][task] - self.rows[0][task])
            .sum::<f64>()
            / (tasks - 1) as f64
    }

    /// Peak post-acquisition performance minus final performance, averaged over
    /// tasks. Lower is better.
    pub fn mean_forgetting(&self) -> f64 {
        let tasks = self.tasks();
        let final_row = &self.rows[tasks];
        (0..tasks)
            .map(|task| {
                let peak = (task + 1..=tasks)
                    .map(|stage| self.rows[stage][task])
                    .fold(f64::NEG_INFINITY, f64::max);
                (peak - final_row[task]).max(0.0)
            })
            .sum::<f64>()
            / tasks as f64
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedEstimate {
    pub n_pairs: usize,
    pub mean_delta: f64,
    pub ci95_low: f64,
    pub ci95_high: f64,
}

/// Paired candidate-minus-control estimate using the psych-bench BCa bootstrap.
/// Pairing is preserved before resampling; callers should pass one aggregate
/// value per independent environment for confirmatory claims.
pub fn paired_delta_bca(
    candidate: &[f64],
    control: &[f64],
    n_resamples: usize,
    seed: u64,
) -> Result<PairedEstimate, String> {
    if candidate.is_empty() || candidate.len() != control.len() {
        return Err("paired samples must be non-empty and equal length".into());
    }
    if candidate
        .iter()
        .chain(control)
        .any(|value| !value.is_finite())
    {
        return Err("paired samples must be finite".into());
    }
    if n_resamples < 100 {
        return Err("at least 100 bootstrap resamples are required".into());
    }

    let deltas: Vec<f64> = candidate
        .iter()
        .zip(control)
        .map(|(candidate, control)| candidate - control)
        .collect();
    let mean_delta = deltas.iter().sum::<f64>() / deltas.len() as f64;
    let (ci95_low, ci95_high) = bootstrap_ci_bca(&deltas, n_resamples, 0.05, seed);

    Ok(PairedEstimate {
        n_pairs: deltas.len(),
        mean_delta,
        ci95_low,
        ci95_high,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceOutcome {
    Supported,
    Equivalent,
    NotDemonstrated,
    Contradicted,
    Inconclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SupportKind {
    ArchitecturalOnly,
    Observed,
    AblationCausal,
    InterventionCausal,
    FunctionallySupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeneralizationLevel {
    DevOnly,
    IidConfirm,
    OodComposition,
    OodFamily,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplicationState {
    Unreplicated,
    Replicated,
    FailedReplication,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceStatus {
    NotAssessed,
    WithinBudget,
    Regression,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceStatus {
    Valid,
    Invalid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimWordingCeiling {
    Exploratory,
    Observational,
    Causal,
    ReplicatedCausal,
}

/// Claim dimensions remain separate by design. Do not add a first-party scalar
/// that averages evidence kind, generalization, replication, and resource status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimLedgerEntry {
    pub schema: String,
    pub claim_id: String,
    pub statement: String,
    pub outcome: EvidenceOutcome,
    pub support_kind: SupportKind,
    pub generalization: GeneralizationLevel,
    pub replication: ReplicationState,
    pub resources: ResourceStatus,
    pub provenance: ProvenanceStatus,
    pub qualifiers: Vec<String>,
}

impl ClaimLedgerEntry {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != CLAIM_LEDGER_SCHEMA_V1 {
            return Err(format!("unsupported claim-ledger schema: {}", self.schema));
        }
        if self.claim_id.trim().is_empty() || self.statement.trim().is_empty() {
            return Err("claim id/statement must be non-empty".into());
        }
        if self.replication == ReplicationState::Replicated
            && self.generalization == GeneralizationLevel::DevOnly
        {
            return Err("DEV-only evidence cannot be labeled replicated".into());
        }
        Ok(())
    }

    pub fn wording_ceiling(&self) -> ClaimWordingCeiling {
        if self.provenance != ProvenanceStatus::Valid
            || self.outcome != EvidenceOutcome::Supported
            || self.replication == ReplicationState::FailedReplication
        {
            return ClaimWordingCeiling::Exploratory;
        }

        let causal = matches!(
            self.support_kind,
            SupportKind::AblationCausal
                | SupportKind::InterventionCausal
                | SupportKind::FunctionallySupported
        );
        if !causal {
            return match self.support_kind {
                SupportKind::Observed => ClaimWordingCeiling::Observational,
                _ => ClaimWordingCeiling::Exploratory,
            };
        }

        if self.replication == ReplicationState::Replicated
            && self.generalization >= GeneralizationLevel::OodComposition
            && self.resources != ResourceStatus::Regression
        {
            ClaimWordingCeiling::ReplicatedCausal
        } else {
            ClaimWordingCeiling::Causal
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest() -> String {
        "ab".repeat(32)
    }

    fn seed_manifest() -> SeedManifest {
        SeedManifest {
            environment_seeds: vec![1, 2, 3],
            representation_seeds: vec![11, 12],
            learner_seeds: vec![21, 22],
            stream_seeds: vec![31, 32],
        }
    }

    #[test]
    fn manifest_hash_is_deterministic_and_seed_namespaces_are_explicit() {
        let manifest = ExperimentManifest {
            schema: EXPERIMENT_MANIFEST_SCHEMA_V1.into(),
            experiment_id: "SYM-ARCH-002A".into(),
            experiment_version: "v1".into(),
            code_revision: "deadbeef".into(),
            preregistration_hash: digest(),
            generator_hash: digest(),
            stream_namespace: StreamNamespace::Dev,
            tuning_status: TuningStatus::Exploratory,
            prior_results_observed: false,
            seed_manifest: seed_manifest(),
            primary_hypothesis: "measurement infrastructure is deterministic".into(),
            primary_comparator: "reference implementation".into(),
            sesoi: 0.05,
        };
        manifest.validate().unwrap();
        assert_eq!(manifest.digest().unwrap(), manifest.digest().unwrap());
        assert_eq!(manifest.seed_manifest.digest().unwrap().len(), 64);
    }

    #[test]
    fn confirmatory_manifest_fails_closed_after_results_are_observed() {
        let manifest = ExperimentManifest {
            schema: EXPERIMENT_MANIFEST_SCHEMA_V1.into(),
            experiment_id: "SYM-ARCH-002".into(),
            experiment_version: "v1".into(),
            code_revision: "deadbeef".into(),
            preregistration_hash: digest(),
            generator_hash: digest(),
            stream_namespace: StreamNamespace::Confirm,
            tuning_status: TuningStatus::ConfirmatoryFirstUse,
            prior_results_observed: true,
            seed_manifest: seed_manifest(),
            primary_hypothesis: "candidate improves retention".into(),
            primary_comparator: "random-features RLS".into(),
            sesoi: 0.05,
        };
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn task_program_is_typed_hashable_and_non_degenerate() {
        let program = TaskProgram {
            schema: TASK_PROGRAM_SCHEMA_V1.into(),
            program_id: "parity-composition-001".into(),
            family: "relational".into(),
            rule: RuleExpr::And {
                terms: vec![
                    RuleExpr::Parity {
                        factor: "x".into(),
                        modulus: 2,
                        remainder: 0,
                    },
                    RuleExpr::Ne {
                        left: "x".into(),
                        right: "y".into(),
                    },
                ],
            },
            context_visibility: ContextVisibility::TaskFree,
            timing_regime: TimingRegime::Jittered,
            positive_examples: 32,
            negative_examples: 32,
            train_support: vec!["depth<=2".into()],
            eval_support: vec!["depth=3".into()],
            oracle_digest: digest(),
        };
        program.validate().unwrap();
        assert_eq!(program.rule.depth(), 2);
        assert_eq!(program.digest().unwrap().len(), 64);
    }

    #[test]
    fn continual_matrix_metrics_match_reference_values() {
        let matrix = PerformanceMatrix::new(vec![
            vec![0.50, 0.50, 0.50],
            vec![0.80, 0.55, 0.50],
            vec![0.75, 0.80, 0.55],
            vec![0.70, 0.78, 0.85],
        ])
        .unwrap();

        assert!((matrix.final_accuracy() - 0.7766666667).abs() < 1e-9);
        assert!((matrix.average_incremental_accuracy() - 0.7838888889).abs() < 1e-9);
        assert!((matrix.backward_transfer() + 0.06).abs() < 1e-9);
        assert!((matrix.forward_transfer() - 0.05).abs() < 1e-9);
        assert!((matrix.mean_forgetting() - 0.04).abs() < 1e-9);
    }

    #[test]
    fn paired_delta_uses_existing_bca_infrastructure() {
        let candidate = [0.70, 0.80, 0.75, 0.90, 0.85, 0.78];
        let control = [0.60, 0.70, 0.70, 0.80, 0.80, 0.73];
        let estimate = paired_delta_bca(&candidate, &control, 500, 42).unwrap();
        assert_eq!(estimate.n_pairs, candidate.len());
        assert!(estimate.mean_delta > 0.0);
        assert!(estimate.ci95_low.is_finite());
        assert!(estimate.ci95_high.is_finite());
        assert!(estimate.ci95_low <= estimate.ci95_high);
    }

    #[test]
    fn claim_ledger_preserves_replication_and_provenance_ceiling() {
        let mut claim = ClaimLedgerEntry {
            schema: CLAIM_LEDGER_SCHEMA_V1.into(),
            claim_id: "semantic-timescale-retention".into(),
            statement: "group-specific timescales reduce cross-timescale interference".into(),
            outcome: EvidenceOutcome::Supported,
            support_kind: SupportKind::InterventionCausal,
            generalization: GeneralizationLevel::OodComposition,
            replication: ReplicationState::Unreplicated,
            resources: ResourceStatus::WithinBudget,
            provenance: ProvenanceStatus::Valid,
            qualifiers: Vec::new(),
        };
        claim.validate().unwrap();
        assert_eq!(claim.wording_ceiling(), ClaimWordingCeiling::Causal);

        claim.replication = ReplicationState::Replicated;
        assert_eq!(
            claim.wording_ceiling(),
            ClaimWordingCeiling::ReplicatedCausal
        );

        claim.replication = ReplicationState::FailedReplication;
        assert_eq!(claim.wording_ceiling(), ClaimWordingCeiling::Exploratory);

        claim.replication = ReplicationState::Unreplicated;
        claim.provenance = ProvenanceStatus::Invalid;
        assert_eq!(claim.wording_ceiling(), ClaimWordingCeiling::Exploratory);
    }
}
