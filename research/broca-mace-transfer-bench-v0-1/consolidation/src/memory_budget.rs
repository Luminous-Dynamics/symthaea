// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Per-target-model memory dose-response calibration.
//!
//! The selected context budget is fitted on paired calibration episodes and
//! qualified on a held-out split. One model's memory curve is never reused
//! for another target model.

use std::collections::{BTreeMap, BTreeSet};

use broca_mace_policy::{Scope, SourceRootDigest};
use sha2::{Digest, Sha256};

use crate::salience::{
    select_for_context, MemorySurfacingCandidateV1, MemorySurfacingModeV1,
    MemorySurfacingPlanV1, SalienceReject,
};

pub const MIN_MEMORY_BUDGET_CLUSTERS: usize = 32;
pub const MIN_MEMORY_BUDGET_FAMILIES: usize = 4;
pub const MAX_MEMORY_BUDGETS: usize = 16;
pub const MAX_MEMORY_BUDGET_TOKENS: u32 = 16_384;
pub const MEMORY_BUDGET_BOOTSTRAP_REPLICATES: usize = 2_000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryDoseTrialV1 {
    pub event_id: u32,
    pub cluster_id: u64,
    pub task_family_id: u32,
    pub source_root: SourceRootDigest,
    /// Exact task/episode identity shared by all budget arms in a cluster.
    pub episode_sha256: String,
    pub environment_sha256: String,
    pub evaluation_protocol_sha256: String,
    /// Digest of the target agent model whose performance is measured.
    pub target_model_sha256: String,
    /// Digest of the frozen memory/consolidation pipeline under test.
    pub memory_pipeline_sha256: String,
    pub context_budget_tokens: u32,
    /// Realized episode utility after declared token, latency and execution costs.
    pub net_utility_micros: i64,
}

impl MemoryDoseTrialV1 {
    pub fn canonical_digest(&self) -> String {
        let fields = [
            self.event_id.to_string(),
            self.cluster_id.to_string(),
            self.task_family_id.to_string(),
            sha256_hex(self.source_root.as_bytes()),
            self.episode_sha256.clone(),
            self.environment_sha256.clone(),
            self.evaluation_protocol_sha256.clone(),
            self.target_model_sha256.clone(),
            self.memory_pipeline_sha256.clone(),
            self.context_budget_tokens.to_string(),
            self.net_utility_micros.to_string(),
        ];
        digest_fields(b"mace:memory-dose-trial:v1\\0", &fields)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryDoseTrialReceiptV1 {
    pub trial_sha256: String,
    pub episode_sha256: String,
    pub environment_sha256: String,
    pub evaluation_protocol_sha256: String,
    pub target_model_sha256: String,
    pub memory_pipeline_sha256: String,
    pub verified_at_ms: i64,
    pub valid_until_ms: i64,
    pub verifier_id: String,
    pub receipt_sha256: String,
}

impl MemoryDoseTrialReceiptV1 {
    pub fn canonical_digest(&self) -> String {
        let fields = [
            self.trial_sha256.clone(),
            self.episode_sha256.clone(),
            self.environment_sha256.clone(),
            self.evaluation_protocol_sha256.clone(),
            self.target_model_sha256.clone(),
            self.memory_pipeline_sha256.clone(),
            self.verified_at_ms.to_string(),
            self.valid_until_ms.to_string(),
            self.verifier_id.clone(),
        ];
        digest_fields(b"mycelix:mace:memory-dose-trial-receipt:v1\\0", &fields)
    }
}

/// Production implementations authenticate an independent run receipt and
/// independently validate the measured outcome and utility-cost calculation.
pub trait MemoryDoseTrialVerifierV1 {
    fn verify_trial(
        &self,
        trial: &MemoryDoseTrialV1,
        as_known_at_ms: i64,
    ) -> Option<MemoryDoseTrialReceiptV1>;
}

/// Opaque record admitted only after the injected verifier accepts its receipt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedMemoryDoseTrialV1 {
    trial: MemoryDoseTrialV1,
    receipt: MemoryDoseTrialReceiptV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryBudgetMeanV1 {
    pub context_budget_tokens: u32,
    pub mean_net_utility_micros: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryBudgetCalibrationV1 {
    pub target_model_sha256: String,
    pub memory_pipeline_sha256: String,
    pub candidate_budgets_tokens: Vec<u32>,
    pub selected_context_budget_tokens: u32,
    pub calibration_dataset_sha256: String,
    pub calibration_task_family_split_sha256: String,
    pub calibration_source_root_split_sha256: String,
    pub environment_sha256: String,
    pub evaluation_protocol_sha256: String,
    pub calibration_cluster_count: u32,
    pub budget_means: Vec<MemoryBudgetMeanV1>,
    pub calibration_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MemoryBudgetReject {
    Empty,
    InvalidDigest,
    MixedTargetModel,
    MixedMemoryPipeline,
    MixedEnvironment,
    MixedEvaluationProtocol,
    MissingTrialReceipt,
    InvalidTrialReceipt,
    TrialReceiptNotValidAtSnapshot,
    InvalidBudgetSet,
    UnknownSourceRoot,
    DuplicateEvent,
    DuplicateClusterBudget,
    ClusterIdentityMismatch,
    MissingBudgetArm,
    TooFewClusters,
    TooFewFamilies,
    TooFewRoots,
    SplitOverlap,
    CalibrationProfileMismatch,
    HeldoutLiftNotEstablished,
    InvalidUtilityRange,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Identity {
    clusters: BTreeSet<u64>,
    families: BTreeSet<u32>,
    roots: BTreeSet<SourceRootDigest>,
    events: BTreeSet<u32>,
    episodes: BTreeSet<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ClusterCurve {
    task_family_id: u32,
    source_root: SourceRootDigest,
    episode_sha256: String,
    utilities: BTreeMap<u32, i64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ValidatedDoseResponse {
    identity: Identity,
    cluster_curves: BTreeMap<u64, ClusterCurve>,
    target_model_sha256: String,
    memory_pipeline_sha256: String,
    environment_sha256: String,
    evaluation_protocol_sha256: String,
    dataset_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QualifiedModelMemoryBudgetV1 {
    target_model_sha256: String,
    memory_pipeline_sha256: String,
    selected_context_budget_tokens: u32,
    calibration_sha256: String,
    heldout_dataset_sha256: String,
    heldout_cluster_count: u32,
    heldout_mean_lift_micros: i64,
    heldout_ci95_lower_micros: i64,
    heldout_ci95_upper_micros: i64,
    qualification_sha256: String,
}

impl QualifiedModelMemoryBudgetV1 {
    pub fn target_model_sha256(&self) -> &str {
        &self.target_model_sha256
    }

    pub fn memory_pipeline_sha256(&self) -> &str {
        &self.memory_pipeline_sha256
    }

    pub fn context_budget_tokens(&self) -> u32 {
        self.selected_context_budget_tokens
    }

    pub fn calibration_sha256(&self) -> &str {
        &self.calibration_sha256
    }

    pub fn heldout_dataset_sha256(&self) -> &str {
        &self.heldout_dataset_sha256
    }

    pub fn heldout_cluster_count(&self) -> u32 {
        self.heldout_cluster_count
    }

    pub fn heldout_mean_lift_micros(&self) -> i64 {
        self.heldout_mean_lift_micros
    }

    pub fn heldout_ci95_lower_micros(&self) -> i64 {
        self.heldout_ci95_lower_micros
    }

    pub fn heldout_ci95_upper_micros(&self) -> i64 {
        self.heldout_ci95_upper_micros
    }

    pub fn qualification_sha256(&self) -> &str {
        &self.qualification_sha256
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelBudgetSurfacingReject {
    TargetModelMismatch,
    InvalidTargetModelDigest,
    SalienceTargetModelMismatch,
    Selection(SalienceReject),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelBudgetSurfacingPlanV1 {
    pub plan: MemorySurfacingPlanV1,
    pub target_model_sha256: String,
    pub memory_pipeline_sha256: String,
    pub calibration_sha256: String,
    pub heldout_dataset_sha256: String,
    pub budget_qualification_sha256: String,
    pub plan_sha256: String,
}

impl ModelBudgetSurfacingPlanV1 {
    pub fn canonical_digest(&self) -> String {
        let fields = [
            self.plan.canonical_digest(),
            self.target_model_sha256.clone(),
            self.memory_pipeline_sha256.clone(),
            self.calibration_sha256.clone(),
            self.heldout_dataset_sha256.clone(),
            self.budget_qualification_sha256.clone(),
        ];
        digest_fields(b"mace:model-budget-surfacing-plan:v1\0", &fields)
    }
}

/// Admit raw dose-response rows only after a verifier attests each exact
/// task/outcome record at the requested knowledge time.
pub fn verify_memory_dose_trials(
    trials: &[MemoryDoseTrialV1],
    as_known_at_ms: i64,
    verifier: &impl MemoryDoseTrialVerifierV1,
) -> Result<Vec<VerifiedMemoryDoseTrialV1>, MemoryBudgetReject> {
    if as_known_at_ms < 0 || trials.is_empty() {
        return Err(MemoryBudgetReject::Empty);
    }
    let mut verified = Vec::with_capacity(trials.len());
    for trial in trials {
        let digests = [
            &trial.episode_sha256,
            &trial.environment_sha256,
            &trial.evaluation_protocol_sha256,
            &trial.target_model_sha256,
            &trial.memory_pipeline_sha256,
        ];
        if digests.iter().any(|value| !is_lower_sha256(value)) {
            return Err(MemoryBudgetReject::InvalidDigest);
        }
        let receipt = verifier
            .verify_trial(trial, as_known_at_ms)
            .ok_or(MemoryBudgetReject::MissingTrialReceipt)?;
        if receipt.trial_sha256 != trial.canonical_digest()
            || receipt.episode_sha256 != trial.episode_sha256
            || receipt.environment_sha256 != trial.environment_sha256
            || receipt.evaluation_protocol_sha256 != trial.evaluation_protocol_sha256
            || receipt.target_model_sha256 != trial.target_model_sha256
            || receipt.memory_pipeline_sha256 != trial.memory_pipeline_sha256
            || !is_lower_sha256(&receipt.trial_sha256)
            || !is_lower_sha256(&receipt.receipt_sha256)
            || receipt.verifier_id.trim().is_empty()
            || receipt.receipt_sha256 != receipt.canonical_digest()
        {
            return Err(MemoryBudgetReject::InvalidTrialReceipt);
        }
        if receipt.verified_at_ms > as_known_at_ms
            || receipt.valid_until_ms <= as_known_at_ms
        {
            return Err(MemoryBudgetReject::TrialReceiptNotValidAtSnapshot);
        }
        verified.push(VerifiedMemoryDoseTrialV1 {
            trial: trial.clone(),
            receipt,
        });
    }
    Ok(verified)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn field(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}

fn digest_fields(domain: &[u8], fields: &[String]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for value in fields {
        field(&mut hasher, value.as_bytes());
    }
    sha256_hex(&hasher.finalize())
}

fn validate_budget_set(budgets: &[u32]) -> Result<(), MemoryBudgetReject> {
    if budgets.len() < 2
        || budgets.len() > MAX_MEMORY_BUDGETS
        || budgets.first() != Some(&0)
        || budgets.iter().any(|budget| *budget > MAX_MEMORY_BUDGET_TOKENS)
        || budgets.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(MemoryBudgetReject::InvalidBudgetSet);
    }
    Ok(())
}

fn dataset_digest(rows: &[VerifiedMemoryDoseTrialV1]) -> String {
    let mut ordered: Vec<_> = rows.iter().collect();
    ordered.sort_by_key(|row| row.trial.event_id);
    let mut hasher = Sha256::new();
    hasher.update(b"mace:model-memory-dose-response:v2\0");
    for verified in ordered {
        field(
            &mut hasher,
            verified.trial.canonical_digest().as_bytes(),
        );
        field(&mut hasher, verified.receipt.receipt_sha256.as_bytes());
    }
    sha256_hex(&hasher.finalize())
}

fn identity_digest(identity: &Identity, domain: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for cluster in &identity.clusters {
        hasher.update(cluster.to_be_bytes());
    }
    for family in &identity.families {
        hasher.update(family.to_be_bytes());
    }
    for root in &identity.roots {
        hasher.update(root.as_bytes());
    }
    for event in &identity.events {
        hasher.update(event.to_be_bytes());
    }
    for episode in &identity.episodes {
        field(&mut hasher, episode.as_bytes());
    }
    sha256_hex(&hasher.finalize())
}

fn validate_trials(
    rows: &[VerifiedMemoryDoseTrialV1],
    budgets: &[u32],
) -> Result<ValidatedDoseResponse, MemoryBudgetReject> {
    validate_budget_set(budgets)?;
    if rows.is_empty() {
        return Err(MemoryBudgetReject::Empty);
    }

    let first_trial = &rows[0].trial;
    let target_model = first_trial.target_model_sha256.clone();
    let pipeline = first_trial.memory_pipeline_sha256.clone();
    let environment = first_trial.environment_sha256.clone();
    let protocol = first_trial.evaluation_protocol_sha256.clone();
    if !is_lower_sha256(&target_model)
        || !is_lower_sha256(&pipeline)
        || !is_lower_sha256(&environment)
        || !is_lower_sha256(&protocol)
    {
        return Err(MemoryBudgetReject::InvalidDigest);
    }

    let mut events = BTreeSet::new();
    let mut clusters = BTreeSet::new();
    let mut families = BTreeSet::new();
    let mut roots = BTreeSet::new();
    let mut episodes = BTreeSet::new();
    let mut curves = BTreeMap::<u64, ClusterCurve>::new();

    for verified in rows {
        let row = &verified.trial;
        if row.target_model_sha256 != target_model {
            return Err(MemoryBudgetReject::MixedTargetModel);
        }
        if row.memory_pipeline_sha256 != pipeline {
            return Err(MemoryBudgetReject::MixedMemoryPipeline);
        }
        if row.environment_sha256 != environment {
            return Err(MemoryBudgetReject::MixedEnvironment);
        }
        if row.evaluation_protocol_sha256 != protocol {
            return Err(MemoryBudgetReject::MixedEvaluationProtocol);
        }
        if row.event_id == 0 || row.cluster_id == 0 || row.task_family_id == 0 {
            return Err(MemoryBudgetReject::InvalidDigest);
        }
        if !row.source_root.is_known() {
            return Err(MemoryBudgetReject::UnknownSourceRoot);
        }
        if !is_lower_sha256(&row.episode_sha256) {
            return Err(MemoryBudgetReject::InvalidDigest);
        }
        if !budgets.contains(&row.context_budget_tokens) {
            return Err(MemoryBudgetReject::InvalidBudgetSet);
        }
        if !(-2_000_000..=2_000_000).contains(&row.net_utility_micros) {
            return Err(MemoryBudgetReject::InvalidUtilityRange);
        }
        if !events.insert(row.event_id) {
            return Err(MemoryBudgetReject::DuplicateEvent);
        }

        clusters.insert(row.cluster_id);
        families.insert(row.task_family_id);
        roots.insert(row.source_root);
        episodes.insert(row.episode_sha256.clone());

        let curve = curves.entry(row.cluster_id).or_insert_with(|| ClusterCurve {
            task_family_id: row.task_family_id,
            source_root: row.source_root,
            episode_sha256: row.episode_sha256.clone(),
            utilities: BTreeMap::new(),
        });
        if curve.task_family_id != row.task_family_id
            || curve.source_root != row.source_root
            || curve.episode_sha256 != row.episode_sha256
        {
            return Err(MemoryBudgetReject::ClusterIdentityMismatch);
        }
        if curve
            .utilities
            .insert(row.context_budget_tokens, row.net_utility_micros)
            .is_some()
        {
            return Err(MemoryBudgetReject::DuplicateClusterBudget);
        }
    }

    if clusters.len() < MIN_MEMORY_BUDGET_CLUSTERS {
        return Err(MemoryBudgetReject::TooFewClusters);
    }
    if families.len() < MIN_MEMORY_BUDGET_FAMILIES {
        return Err(MemoryBudgetReject::TooFewFamilies);
    }
    if roots.len() < MIN_MEMORY_BUDGET_CLUSTERS / 2 {
        return Err(MemoryBudgetReject::TooFewRoots);
    }
    for curve in curves.values() {
        if curve.utilities.len() != budgets.len()
            || budgets
                .iter()
                .any(|budget| !curve.utilities.contains_key(budget))
        {
            return Err(MemoryBudgetReject::MissingBudgetArm);
        }
    }

    let identity = Identity {
        clusters,
        families,
        roots,
        events,
        episodes,
    };
    Ok(ValidatedDoseResponse {
        identity,
        cluster_curves: curves,
        target_model_sha256: target_model,
        memory_pipeline_sha256: pipeline,
        environment_sha256: environment,
        evaluation_protocol_sha256: protocol,
        dataset_sha256: dataset_digest(rows),
    })
}

fn ensure_disjoint(left: &Identity, right: &Identity) -> Result<(), MemoryBudgetReject> {
    if !left.clusters.is_disjoint(&right.clusters)
        || !left.families.is_disjoint(&right.families)
        || !left.roots.is_disjoint(&right.roots)
        || !left.events.is_disjoint(&right.events)
        || !left.episodes.is_disjoint(&right.episodes)
    {
        return Err(MemoryBudgetReject::SplitOverlap);
    }
    Ok(())
}

fn mean_utility(curves: &BTreeMap<u64, ClusterCurve>, budget: u32) -> i64 {
    let sum: i128 = curves
        .values()
        .map(|curve| i128::from(curve.utilities[&budget]))
        .sum();
    (sum / curves.len() as i128) as i64
}

fn calibration_digest(profile: &MemoryBudgetCalibrationV1) -> String {
    let mut fields = vec![
        profile.target_model_sha256.clone(),
        profile.memory_pipeline_sha256.clone(),
        profile.selected_context_budget_tokens.to_string(),
        profile.calibration_dataset_sha256.clone(),
        profile.calibration_task_family_split_sha256.clone(),
        profile.calibration_source_root_split_sha256.clone(),
        profile.environment_sha256.clone(),
        profile.evaluation_protocol_sha256.clone(),
        profile.calibration_cluster_count.to_string(),
    ];
    fields.extend(
        profile
            .candidate_budgets_tokens
            .iter()
            .map(u32::to_string),
    );
    for mean in &profile.budget_means {
        fields.push(mean.context_budget_tokens.to_string());
        fields.push(mean.mean_net_utility_micros.to_string());
    }
    digest_fields(b"mace:model-memory-budget-calibration:v1\0", &fields)
}

/// Choose a context budget using calibration episodes only. The candidate
/// budget set is frozen by the caller and must include the zero-memory arm.
/// Ties select the smaller budget.
pub fn fit_model_memory_budget(
    calibration_rows: &[MemoryDoseTrialV1],
    candidate_budgets_tokens: &[u32],
    as_known_at_ms: i64,
    verifier: &impl MemoryDoseTrialVerifierV1,
) -> Result<MemoryBudgetCalibrationV1, MemoryBudgetReject> {
    let verified_rows =
        verify_memory_dose_trials(calibration_rows, as_known_at_ms, verifier)?;
    let validated = validate_trials(&verified_rows, candidate_budgets_tokens)?;
    let budget_means: Vec<_> = candidate_budgets_tokens
        .iter()
        .map(|budget| MemoryBudgetMeanV1 {
            context_budget_tokens: *budget,
            mean_net_utility_micros: mean_utility(
                &validated.cluster_curves,
                *budget,
            ),
        })
        .collect();

    let mut selected = candidate_budgets_tokens[0];
    let mut selected_mean = budget_means[0].mean_net_utility_micros;
    for mean in budget_means.iter().skip(1) {
        if mean.mean_net_utility_micros > selected_mean {
            selected = mean.context_budget_tokens;
            selected_mean = mean.mean_net_utility_micros;
        }
    }

    let mut profile = MemoryBudgetCalibrationV1 {
        target_model_sha256: validated.target_model_sha256,
        memory_pipeline_sha256: validated.memory_pipeline_sha256,
        candidate_budgets_tokens: candidate_budgets_tokens.to_vec(),
        selected_context_budget_tokens: selected,
        calibration_dataset_sha256: validated.dataset_sha256,
        calibration_task_family_split_sha256: identity_digest(
            &validated.identity,
            b"mace:memory-budget-task-families:v1\0",
        ),
        calibration_source_root_split_sha256: identity_digest(
            &validated.identity,
            b"mace:memory-budget-source-roots:v1\0",
        ),
        environment_sha256: validated.environment_sha256,
        evaluation_protocol_sha256: validated.evaluation_protocol_sha256,
        calibration_cluster_count: validated.identity.clusters.len() as u32,
        budget_means,
        calibration_sha256: String::new(),
    };
    profile.calibration_sha256 = calibration_digest(&profile);
    Ok(profile)
}

fn paired_differences(
    curves: &BTreeMap<u64, ClusterCurve>,
    selected_budget: u32,
) -> Vec<i64> {
    curves
        .values()
        .map(|curve| {
            curve.utilities[&selected_budget]
                .saturating_sub(curve.utilities[&0])
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct BootstrapRng {
    state: u64,
}

impl BootstrapRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1),
        }
    }

    fn next(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.state = value;
        value.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next() % upper as u64) as usize
    }
}

fn bootstrap_ci95(differences: &[i64], seed_text: &str) -> (i64, i64) {
    if differences.is_empty() {
        return (0, 0);
    }
    let seed = u64::from_str_radix(&seed_text[..16], 16).unwrap_or(1);
    let mut rng = BootstrapRng::new(seed);
    let mut means = Vec::with_capacity(MEMORY_BUDGET_BOOTSTRAP_REPLICATES);
    for _ in 0..MEMORY_BUDGET_BOOTSTRAP_REPLICATES {
        let sum: i128 = (0..differences.len())
            .map(|_| i128::from(differences[rng.index(differences.len())]))
            .sum();
        means.push(sum as f64 / differences.len() as f64);
    }
    means.sort_by(f64::total_cmp);
    let lower_index = (MEMORY_BUDGET_BOOTSTRAP_REPLICATES - 1) * 25 / 1_000;
    let upper_index = (MEMORY_BUDGET_BOOTSTRAP_REPLICATES - 1) * 975 / 1_000;
    (
        means[lower_index].round() as i64,
        means[upper_index].round() as i64,
    )
}

/// Qualify the calibration-selected budget on a fully disjoint held-out set.
/// A nonzero budget is returned only when its paired cluster-bootstrap 95%
/// lower confidence bound beats the zero-memory arm.
pub fn qualify_model_memory_budget(
    calibration: &MemoryBudgetCalibrationV1,
    calibration_rows: &[MemoryDoseTrialV1],
    heldout_rows: &[MemoryDoseTrialV1],
    as_known_at_ms: i64,
    verifier: &impl MemoryDoseTrialVerifierV1,
) -> Result<QualifiedModelMemoryBudgetV1, MemoryBudgetReject> {
    if calibration.calibration_sha256 != calibration_digest(calibration) {
        return Err(MemoryBudgetReject::CalibrationProfileMismatch);
    }
    let reproduced = fit_model_memory_budget(
        calibration_rows,
        &calibration.candidate_budgets_tokens,
        as_known_at_ms,
        verifier,
    )?;
    if &reproduced != calibration {
        return Err(MemoryBudgetReject::CalibrationProfileMismatch);
    }

    let verified_heldout =
        verify_memory_dose_trials(heldout_rows, as_known_at_ms, verifier)?;
    let heldout = validate_trials(
        &verified_heldout,
        &calibration.candidate_budgets_tokens,
    )?;
    if heldout.target_model_sha256 != calibration.target_model_sha256
        || heldout.memory_pipeline_sha256 != calibration.memory_pipeline_sha256
        || heldout.environment_sha256 != calibration.environment_sha256
        || heldout.evaluation_protocol_sha256 != calibration.evaluation_protocol_sha256
    {
        return Err(MemoryBudgetReject::CalibrationProfileMismatch);
    }
    let verified_calibration =
        verify_memory_dose_trials(calibration_rows, as_known_at_ms, verifier)?;
    let calibration_identity = validate_trials(
        &verified_calibration,
        &calibration.candidate_budgets_tokens,
    )?
    .identity;
    ensure_disjoint(&calibration_identity, &heldout.identity)?;

    let differences = paired_differences(
        &heldout.cluster_curves,
        calibration.selected_context_budget_tokens,
    );
    let mean_lift = (differences.iter().map(|value| i128::from(*value)).sum::<i128>()
        / differences.len() as i128) as i64;
    let (lower, upper) = if calibration.selected_context_budget_tokens == 0 {
        (0, 0)
    } else {
        bootstrap_ci95(&differences, &heldout.dataset_sha256)
    };
    if calibration.selected_context_budget_tokens != 0 && lower <= 0 {
        return Err(MemoryBudgetReject::HeldoutLiftNotEstablished);
    }

    let fields = vec![
        calibration.target_model_sha256.clone(),
        calibration.memory_pipeline_sha256.clone(),
        calibration.selected_context_budget_tokens.to_string(),
        calibration.calibration_sha256.clone(),
        heldout.dataset_sha256.clone(),
        heldout.identity.clusters.len().to_string(),
        mean_lift.to_string(),
        lower.to_string(),
        upper.to_string(),
    ];
    let qualification_sha256 =
        digest_fields(b"mace:qualified-model-memory-budget:v1\0", &fields);

    Ok(QualifiedModelMemoryBudgetV1 {
        target_model_sha256: calibration.target_model_sha256.clone(),
        memory_pipeline_sha256: calibration.memory_pipeline_sha256.clone(),
        selected_context_budget_tokens: calibration.selected_context_budget_tokens,
        calibration_sha256: calibration.calibration_sha256.clone(),
        heldout_dataset_sha256: heldout.dataset_sha256,
        heldout_cluster_count: heldout.identity.clusters.len() as u32,
        heldout_mean_lift_micros: mean_lift,
        heldout_ci95_lower_micros: lower,
        heldout_ci95_upper_micros: upper,
        qualification_sha256,
    })
}

/// Compose the qualified per-model budget with the existing evidence-bound
/// knapsack selector. The wrapper digest binds the target model, frozen memory
/// pipeline, calibration, held-out qualification and final surfacing plan.
pub fn select_for_qualified_model_budget(
    candidates: &[MemorySurfacingCandidateV1],
    mode: MemorySurfacingModeV1,
    target_scope: Scope,
    as_known: broca_mace_policy::AsKnownQuery,
    runtime_target_model_sha256: &str,
    budget: &QualifiedModelMemoryBudgetV1,
) -> Result<ModelBudgetSurfacingPlanV1, ModelBudgetSurfacingReject> {
    if !is_lower_sha256(runtime_target_model_sha256) {
        return Err(ModelBudgetSurfacingReject::InvalidTargetModelDigest);
    }
    if runtime_target_model_sha256 != budget.target_model_sha256 {
        return Err(ModelBudgetSurfacingReject::TargetModelMismatch);
    }
    if budget.selected_context_budget_tokens > 0
        && candidates.iter().any(|candidate| {
            candidate
                .target_model_sha256()
                .is_some_and(|model| model != runtime_target_model_sha256)
        })
    {
        return Err(ModelBudgetSurfacingReject::SalienceTargetModelMismatch);
    }
    let plan = if budget.selected_context_budget_tokens == 0 {
        let mut empty_plan = MemorySurfacingPlanV1 {
            schema_version: 1,
            mode,
            target_scope,
            as_known,
            context_budget_tokens: 0,
            context_tokens_used: 0,
            predicted_net_value_micros: 0,
            surfaced: Vec::new(),
            plan_sha256: String::new(),
        };
        empty_plan.plan_sha256 = empty_plan.canonical_digest();
        empty_plan
    } else {
        select_for_context(
            candidates,
            mode,
            target_scope,
            as_known,
            budget.selected_context_budget_tokens,
        )
        .map_err(ModelBudgetSurfacingReject::Selection)?
    };
    let fields = [
        plan.canonical_digest(),
        budget.target_model_sha256.clone(),
        budget.memory_pipeline_sha256.clone(),
        budget.calibration_sha256.clone(),
        budget.heldout_dataset_sha256.clone(),
        budget.qualification_sha256.clone(),
    ];
    let plan_sha256 =
        digest_fields(b"mace:model-budget-surfacing-plan:v1\0", &fields);
    Ok(ModelBudgetSurfacingPlanV1 {
        plan,
        target_model_sha256: budget.target_model_sha256.clone(),
        memory_pipeline_sha256: budget.memory_pipeline_sha256.clone(),
        calibration_sha256: budget.calibration_sha256.clone(),
        heldout_dataset_sha256: budget.heldout_dataset_sha256.clone(),
        budget_qualification_sha256: budget.qualification_sha256.clone(),
        plan_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const BUDGETS: [u32; 4] = [0, 128, 512, 1_024];

    struct FixtureMemoryDoseTrialVerifier;

    impl MemoryDoseTrialVerifierV1 for FixtureMemoryDoseTrialVerifier {
        fn verify_trial(
            &self,
            trial: &MemoryDoseTrialV1,
            as_known_at_ms: i64,
        ) -> Option<MemoryDoseTrialReceiptV1> {
            let mut receipt = MemoryDoseTrialReceiptV1 {
                trial_sha256: trial.canonical_digest(),
                episode_sha256: trial.episode_sha256.clone(),
                environment_sha256: trial.environment_sha256.clone(),
                evaluation_protocol_sha256: trial.evaluation_protocol_sha256.clone(),
                target_model_sha256: trial.target_model_sha256.clone(),
                memory_pipeline_sha256: trial.memory_pipeline_sha256.clone(),
                verified_at_ms: as_known_at_ms.saturating_sub(1),
                valid_until_ms: as_known_at_ms.saturating_add(10_000),
                verifier_id: "fixture-dose-trial-verifier".into(),
                receipt_sha256: String::new(),
            };
            receipt.receipt_sha256 = receipt.canonical_digest();
            Some(receipt)
        }
    }

    const AS_KNOWN_AT_MS: i64 = 100;

    fn digest(label: &str) -> String {
        sha256_hex(label.as_bytes())
    }


    fn fit(
        rows: &[MemoryDoseTrialV1],
        budgets: &[u32],
    ) -> Result<MemoryBudgetCalibrationV1, MemoryBudgetReject> {
        fit_model_memory_budget(
            rows,
            budgets,
            AS_KNOWN_AT_MS,
            &FixtureMemoryDoseTrialVerifier,
        )
    }

    fn qualify(
        profile: &MemoryBudgetCalibrationV1,
        calibration: &[MemoryDoseTrialV1],
        heldout: &[MemoryDoseTrialV1],
    ) -> Result<QualifiedModelMemoryBudgetV1, MemoryBudgetReject> {
        qualify_model_memory_budget(
            profile,
            calibration,
            heldout,
            AS_KNOWN_AT_MS,
            &FixtureMemoryDoseTrialVerifier,
        )
    }

    fn rows(
        model: &str,
        pipeline: &str,
        cluster_start: u64,
        event_start: u32,
        family_start: u32,
        root_start: u32,
        uplifts: [i64; 4],
    ) -> Vec<MemoryDoseTrialV1> {
        let mut result = Vec::new();
        for cluster_offset in 0..48_u64 {
            let cluster = cluster_start + cluster_offset + 1;
            let family = family_start + (cluster_offset as u32 % 8);
            let mut root_bytes = [0_u8; 32];
            root_bytes[..4]
                .copy_from_slice(&(root_start + cluster_offset as u32 + 1).to_be_bytes());
            let root = SourceRootDigest::from_bytes(root_bytes);
            let baseline = 500_000 + i64::from((cluster_offset % 5) as u32) * 500;
            for (index, budget) in BUDGETS.iter().copied().enumerate() {
                let event_id =
                    event_start + cluster_offset as u32 * BUDGETS.len() as u32 + index as u32 + 1;
                result.push(MemoryDoseTrialV1 {
                    event_id,
                    cluster_id: cluster,
                    task_family_id: family,
                    source_root: root,
                    episode_sha256: digest(&format!("episode-{cluster}")),
                    environment_sha256: digest("fixed-appworld-environment-v1"),
                    evaluation_protocol_sha256: digest("memory-dose-protocol-v1"),
                    target_model_sha256: model.to_owned(),
                    memory_pipeline_sha256: pipeline.to_owned(),
                    context_budget_tokens: budget,
                    net_utility_micros: baseline + uplifts[index],
                });
            }
        }
        result
    }

    #[test]
    fn forged_or_snapshot_expired_trial_receipts_are_rejected() {
        struct WrongDigestVerifier;
        impl MemoryDoseTrialVerifierV1 for WrongDigestVerifier {
            fn verify_trial(
                &self,
                trial: &MemoryDoseTrialV1,
                as_known_at_ms: i64,
            ) -> Option<MemoryDoseTrialReceiptV1> {
                let fixture = FixtureMemoryDoseTrialVerifier;
                let mut receipt = fixture.verify_trial(trial, as_known_at_ms)?;
                receipt.trial_sha256 = digest("not-the-trial-digest");
                receipt.receipt_sha256 = receipt.canonical_digest();
                Some(receipt)
            }
        }
        struct ExpiredVerifier;
        impl MemoryDoseTrialVerifierV1 for ExpiredVerifier {
            fn verify_trial(
                &self,
                trial: &MemoryDoseTrialV1,
                as_known_at_ms: i64,
            ) -> Option<MemoryDoseTrialReceiptV1> {
                let fixture = FixtureMemoryDoseTrialVerifier;
                let mut receipt = fixture.verify_trial(trial, as_known_at_ms)?;
                receipt.valid_until_ms = as_known_at_ms;
                receipt.receipt_sha256 = receipt.canonical_digest();
                Some(receipt)
            }
        }

        let model = digest("receipt-test-model");
        let pipeline = digest("receipt-test-pipeline");
        let trials = rows(
            &model,
            &pipeline,
            1,
            1,
            10,
            100,
            [0, 10_000, 20_000, 30_000],
        );
        assert_eq!(
            verify_memory_dose_trials(
                &trials,
                AS_KNOWN_AT_MS,
                &WrongDigestVerifier,
            ),
            Err(MemoryBudgetReject::InvalidTrialReceipt)
        );
        assert_eq!(
            verify_memory_dose_trials(
                &trials,
                AS_KNOWN_AT_MS,
                &ExpiredVerifier,
            ),
            Err(MemoryBudgetReject::TrialReceiptNotValidAtSnapshot)
        );
    }

    #[test]
    fn every_budget_arm_in_a_cluster_must_measure_the_same_episode() {
        let model = digest("paired-episode-model");
        let pipeline = digest("paired-episode-pipeline");
        let mut trials = rows(
            &model,
            &pipeline,
            1,
            1,
            10,
            100,
            [0, 10_000, 20_000, 30_000],
        );
        trials[0].episode_sha256 = digest("substituted-episode");
        assert_eq!(
            fit(&trials, &BUDGETS),
            Err(MemoryBudgetReject::ClusterIdentityMismatch)
        );
    }

    #[test]
    fn missing_receipt_and_mixed_protocol_are_rejected() {
        struct MissingVerifier;
        impl MemoryDoseTrialVerifierV1 for MissingVerifier {
            fn verify_trial(
                &self,
                _trial: &MemoryDoseTrialV1,
                _as_known_at_ms: i64,
            ) -> Option<MemoryDoseTrialReceiptV1> {
                None
            }
        }

        let model = digest("protocol-test-model");
        let pipeline = digest("protocol-test-pipeline");
        let mut trials = rows(
            &model,
            &pipeline,
            1,
            1,
            10,
            100,
            [0, 10_000, 20_000, 30_000],
        );
        assert_eq!(
            verify_memory_dose_trials(
                &trials,
                AS_KNOWN_AT_MS,
                &MissingVerifier,
            ),
            Err(MemoryBudgetReject::MissingTrialReceipt)
        );

        trials[0].evaluation_protocol_sha256 = digest("different-protocol");
        assert_eq!(
            fit(&trials, &BUDGETS),
            Err(MemoryBudgetReject::MixedEvaluationProtocol)
        );

        trials[0].evaluation_protocol_sha256 = digest("memory-dose-protocol-v1");
        trials[0].environment_sha256 = digest("different-environment");
        assert_eq!(
            fit(&trials, &BUDGETS),
            Err(MemoryBudgetReject::MixedEnvironment)
        );
    }

    #[test]
    fn each_target_model_gets_its_own_heldout_qualified_memory_budget() {
        let pipeline = digest("stable-memory-pipeline-v1");
        let weak_model = digest("small-agent-model");
        let strong_model = digest("large-agent-model");
        let weak_calibration = rows(
            &weak_model,
            &pipeline,
            1,
            1,
            10,
            100,
            [0, 50_000, 24_000, -12_000],
        );
        let weak_heldout = rows(
            &weak_model,
            &pipeline,
            1_000,
            10_000,
            100,
            10_000,
            [0, 48_000, 22_000, -14_000],
        );
        let strong_calibration = rows(
            &strong_model,
            &pipeline,
            2_000,
            20_000,
            200,
            20_000,
            [0, 16_000, 48_000, 70_000],
        );
        let strong_heldout = rows(
            &strong_model,
            &pipeline,
            3_000,
            30_000,
            300,
            30_000,
            [0, 14_000, 46_000, 68_000],
        );

        let weak_profile =
            fit(&weak_calibration, &BUDGETS).unwrap();
        let strong_profile =
            fit(&strong_calibration, &BUDGETS).unwrap();
        assert_eq!(weak_profile.selected_context_budget_tokens, 128);
        assert_eq!(strong_profile.selected_context_budget_tokens, 1_024);

        let weak = qualify(
            &weak_profile,
            &weak_calibration,
            &weak_heldout,
        )
        .unwrap();
        let strong = qualify(
            &strong_profile,
            &strong_calibration,
            &strong_heldout,
        )
        .unwrap();
        assert_eq!(weak.context_budget_tokens(), 128);
        assert_eq!(strong.context_budget_tokens(), 1_024);
        assert!(weak.heldout_ci95_lower_micros() > 0);
        assert!(strong.heldout_ci95_lower_micros() > 0);
        assert_ne!(weak.target_model_sha256(), strong.target_model_sha256());
    }

    #[test]
    fn target_model_and_memory_pipeline_cannot_be_mixed() {
        let model = digest("target-model");
        let pipeline = digest("memory-pipeline");
        let mut calibration =
            rows(&model, &pipeline, 1, 1, 10, 100, [0, 10_000, 20_000, 30_000]);
        calibration[0].target_model_sha256 = digest("other-model");
        assert_eq!(
            fit(&calibration, &BUDGETS),
            Err(MemoryBudgetReject::MixedTargetModel)
        );
        let mut wrong_pipeline =
            rows(&model, &pipeline, 1, 1, 10, 100, [0, 10_000, 20_000, 30_000]);
        wrong_pipeline[0].memory_pipeline_sha256 = digest("other-pipeline");
        assert_eq!(
            fit(&wrong_pipeline, &BUDGETS),
            Err(MemoryBudgetReject::MixedMemoryPipeline)
        );
    }

    #[test]
    fn heldout_must_be_disjoint_in_clusters_task_families_roots_and_events() {
        let model = digest("target-model");
        let pipeline = digest("memory-pipeline");
        let calibration =
            rows(&model, &pipeline, 1, 1, 10, 100, [0, 10_000, 20_000, 30_000]);
        let profile = fit(&calibration, &BUDGETS).unwrap();
        let mut heldout =
            rows(&model, &pipeline, 1_000, 10_000, 100, 10_000, [0, 10_000, 20_000, 30_000]);
        heldout[0].source_root = calibration[0].source_root;
        assert_eq!(
            qualify(&profile, &calibration, &heldout),
            Err(MemoryBudgetReject::SplitOverlap)
        );
    }

    #[test]
    fn calibration_selected_budget_must_demonstrate_positive_heldout_lift() {
        let model = digest("target-model");
        let pipeline = digest("memory-pipeline");
        let calibration =
            rows(&model, &pipeline, 1, 1, 10, 100, [0, 15_000, 40_000, 70_000]);
        let profile = fit(&calibration, &BUDGETS).unwrap();
        let heldout =
            rows(&model, &pipeline, 1_000, 10_000, 100, 10_000, [0, -15_000, -40_000, -70_000]);
        assert_eq!(
            qualify(&profile, &calibration, &heldout),
            Err(MemoryBudgetReject::HeldoutLiftNotEstablished)
        );
    }

    #[test]
    fn zero_memory_remains_a_valid_qualified_choice_when_memory_does_not_help() {
        let model = digest("target-model");
        let pipeline = digest("memory-pipeline");
        let calibration =
            rows(&model, &pipeline, 1, 1, 10, 100, [20_000, 10_000, -10_000, -20_000]);
        let heldout =
            rows(&model, &pipeline, 1_000, 10_000, 100, 10_000, [18_000, 8_000, -12_000, -22_000]);
        let profile = fit(&calibration, &BUDGETS).unwrap();
        assert_eq!(profile.selected_context_budget_tokens, 0);
        let qualified =
            qualify(&profile, &calibration, &heldout).unwrap();
        assert_eq!(qualified.context_budget_tokens(), 0);
        assert_eq!(qualified.heldout_mean_lift_micros(), 0);
    }

    #[test]
    fn qualified_budget_cannot_be_reused_for_a_different_runtime_model() {
        let model = digest("target-model");
        let pipeline = digest("memory-pipeline");
        let calibration =
            rows(&model, &pipeline, 1, 1, 10, 100, [0, 10_000, 20_000, 30_000]);
        let heldout =
            rows(&model, &pipeline, 1_000, 10_000, 100, 10_000, [0, 10_000, 20_000, 30_000]);
        let profile = fit(&calibration, &BUDGETS).unwrap();
        let qualified =
            qualify(&profile, &calibration, &heldout).unwrap();
        let wrong_runtime_model = digest("different-runtime-model");
        let result = select_for_qualified_model_budget(
            &[],
            MemorySurfacingModeV1::RecencyFirst,
            Scope {
                regime: broca_mace_policy::Regime::Baseline,
                objective: broca_mace_policy::Objective::Quality,
            },
            broca_mace_policy::AsKnownQuery {
                valid_at_ms: 100,
                known_at_ms: 100,
            },
            &wrong_runtime_model,
            &qualified,
        );
        assert_eq!(
            result,
            Err(ModelBudgetSurfacingReject::TargetModelMismatch)
        );
    }

    #[test]
    fn qualified_zero_memory_budget_produces_a_valid_empty_surfacing_plan() {
        let model = digest("zero-memory-model");
        let pipeline = digest("memory-pipeline");
        let calibration =
            rows(&model, &pipeline, 1, 1, 10, 100, [20_000, 10_000, -10_000, -20_000]);
        let heldout =
            rows(&model, &pipeline, 1_000, 10_000, 100, 10_000, [18_000, 8_000, -12_000, -22_000]);
        let profile = fit(&calibration, &BUDGETS).unwrap();
        let qualified =
            qualify(&profile, &calibration, &heldout).unwrap();
        assert_eq!(qualified.context_budget_tokens(), 0);

        let scope = Scope {
            regime: broca_mace_policy::Regime::Baseline,
            objective: broca_mace_policy::Objective::Quality,
        };
        let result = select_for_qualified_model_budget(
            &[],
            MemorySurfacingModeV1::PredictedFutureValue,
            scope,
            broca_mace_policy::AsKnownQuery {
                valid_at_ms: 100,
                known_at_ms: 100,
            },
            &model,
            &qualified,
        )
        .unwrap();

        assert_eq!(result.plan.context_budget_tokens, 0);
        assert_eq!(result.plan.context_tokens_used, 0);
        assert!(result.plan.surfaced.is_empty());
        assert_eq!(
            result.plan.plan_sha256,
            result.plan.canonical_digest()
        );
        assert_eq!(result.heldout_dataset_sha256, qualified.heldout_dataset_sha256());
        assert_eq!(result.plan_sha256, result.canonical_digest());
    }

    #[test]
    fn calibration_profile_cannot_be_mutated_after_fitting() {
        let model = digest("target-model");
        let pipeline = digest("memory-pipeline");
        let calibration =
            rows(&model, &pipeline, 1, 1, 10, 100, [0, 10_000, 20_000, 30_000]);
        let mut profile = fit(&calibration, &BUDGETS).unwrap();
        profile.selected_context_budget_tokens = 0;
        let heldout =
            rows(&model, &pipeline, 1_000, 10_000, 100, 10_000, [0, 10_000, 20_000, 30_000]);
        assert_eq!(
            qualify(&profile, &calibration, &heldout),
            Err(MemoryBudgetReject::CalibrationProfileMismatch)
        );
    }
}
