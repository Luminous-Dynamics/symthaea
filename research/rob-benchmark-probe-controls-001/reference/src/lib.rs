//! Shortcut-solvability and privileged-input probe audit.
//!
//! This crate evaluates caller-supplied, frozen probe outcomes. It does not
//! train models, prove declared model capacity, authenticate feature provenance,
//! certify a benchmark, or establish physical robot safety.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

/// Parts-per-million scale used for deterministic rates and gap thresholds.
pub const PPM_SCALE: u32 = 1_000_000;
/// Defensive bound on the number of independent clusters in one audit.
pub const MAX_INDEPENDENT_CLUSTERS: usize = 10_000;

/// Input families declared as available to a policy or probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum InputFamily {
    /// Task identifiers, object names, fixture metadata and task descriptors.
    TaskMetadata,
    /// Natural-language task instruction available to the deployed policy.
    LanguageInstruction,
    /// Camera, depth, tactile or other declared sensory observations.
    VisualObservation,
    /// Joint state, force, torque, pose and proprioceptive state.
    RobotState,
    /// Previously issued actions and policy-visible interaction history.
    ActionHistory,
    /// Non-privileged environment descriptors available at deployment.
    EnvironmentMetadata,
    /// Stable episode identity, which can accidentally encode seed or label.
    EpisodeIdentity,
    /// Simulator or evaluation random seed.
    EnvironmentSeed,
    /// Evaluator-internal state unavailable to the deployed policy.
    EvaluatorState,
    /// Ground-truth task outcome or hidden answer label.
    GroundTruthLabel,
}

impl InputFamily {
    /// Whether this input family is prohibited for the headline candidate.
    #[must_use]
    pub const fn is_privileged(self) -> bool {
        matches!(
            self,
            Self::EpisodeIdentity
                | Self::EnvironmentSeed
                | Self::EvaluatorState
                | Self::GroundTruthLabel
        )
    }
}

/// Model family declared by the benchmark owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    /// Headline policy being audited.
    FullCandidatePolicy,
    /// Constant or majority-class probe.
    ConstantMajority,
    /// A one-feature categorical lookup probe.
    SingleFeatureLookup,
    /// A deliberately capacity-bounded small linear probe.
    SmallLinear,
    /// A deliberately shallow decision-tree probe.
    ShallowDecisionTree,
}

/// Independence evidence state for one experimental cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceDisposition {
    /// The experiment owner has qualified the cluster definition.
    Qualified,
    /// Independence or provenance is not established.
    Unknown,
    /// The supplied independence evidence is invalid.
    Rejected,
}

/// Frozen evaluation scope shared by the candidate and every probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeAuditScope {
    /// Stable benchmark identifier.
    pub benchmark_id: String,
    /// Frozen benchmark protocol/version.
    pub benchmark_version: String,
    /// Frozen task-manifest digest.
    pub task_manifest_digest: String,
    /// Frozen scenario and environment-manifest digest.
    pub scenario_manifest_digest: String,
    /// Exact robot article and configuration digest.
    pub robot_configuration_digest: String,
    /// Exact outcome evaluator and safety-specification digest.
    pub evaluator_digest: String,
    /// Shared time, compute, observation, energy and resource-budget digest.
    pub resource_budget_digest: String,
    /// Immutable hidden confirmatory evaluation-split digest.
    pub hidden_evaluation_split_digest: String,
    /// Expected number of independent clusters.
    pub expected_independent_clusters: u16,
    /// Minimum qualified independent clusters required for a diagnostic.
    pub minimum_qualified_clusters: u16,
    /// Maximum median candidate-minus-probe safe-success gap considered near.
    pub near_candidate_gap_ppm: u32,
}

/// Identity and declared input/training boundary for one evaluated arm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArmManifest {
    /// Stable arm identifier.
    pub arm_id: String,
    /// Content digest of the policy/model artifact.
    pub model_artifact_digest: String,
    /// Independent architecture/capacity audit receipt digest.
    pub architecture_audit_digest: String,
    /// Policy digest.
    pub policy_digest: String,
    /// Controller or execution-harness digest.
    pub harness_digest: String,
    /// Task-manifest digest.
    pub task_manifest_digest: String,
    /// Scenario-manifest digest.
    pub scenario_manifest_digest: String,
    /// Robot configuration digest.
    pub robot_configuration_digest: String,
    /// Evaluator and safety-specification digest.
    pub evaluator_digest: String,
    /// Resource-budget digest.
    pub resource_budget_digest: String,
    /// Training split manifest digest.
    pub training_split_digest: String,
    /// Calibration / model-selection split manifest digest.
    pub calibration_split_digest: String,
    /// Evaluation split manifest digest; must equal the frozen hidden split.
    pub evaluation_split_digest: String,
    /// Declared model family.
    pub model_family: ModelFamily,
    /// Declared feature/input families available to this arm.
    pub input_families: Vec<InputFamily>,
}

/// Mutually exclusive per-episode outcome categories for one arm/cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpisodeTally {
    /// Every scheduled episode, including failure, timeout, Unknown and censoring.
    pub episodes: u32,
    /// Episodes completing the native task, including unsafe completions.
    pub task_successes: u32,
    /// Episodes completing the task and satisfying all applicable safety specs.
    pub safe_successes: u32,
    /// Known task failures.
    pub task_failures: u32,
    /// Episodes whose outcome could not be established.
    pub unknown: u32,
    /// Episodes missing a declared task or mission deadline.
    pub timed_out: u32,
    /// Episodes censored under the frozen protocol.
    pub censored: u32,
    /// Episodes with at least one critical safety violation.
    pub critical_safety_violations: u32,
    /// Raw control ticks; descriptive only and never an independent sample count.
    pub control_ticks: u64,
}

impl EpisodeTally {
    fn validate(self, arm_id: &str, cluster_id: &str) -> Result<(), ProbeAuditError> {
        if self.episodes == 0 {
            return Err(ProbeAuditError::InvalidEpisodeTally {
                arm_id: arm_id.to_owned(),
                cluster_id: cluster_id.to_owned(),
            });
        }
        let classified = self
            .task_successes
            .checked_add(self.task_failures)
            .and_then(|v| v.checked_add(self.unknown))
            .and_then(|v| v.checked_add(self.timed_out))
            .and_then(|v| v.checked_add(self.censored))
            .ok_or_else(|| ProbeAuditError::InvalidEpisodeTally {
                arm_id: arm_id.to_owned(),
                cluster_id: cluster_id.to_owned(),
            })?;
        if classified != self.episodes
            || self.safe_successes > self.task_successes
            || self.critical_safety_violations > self.episodes
        {
            return Err(ProbeAuditError::InvalidEpisodeTally {
                arm_id: arm_id.to_owned(),
                cluster_id: cluster_id.to_owned(),
            });
        }
        Ok(())
    }
}

/// One cluster's outcomes for a candidate or probe arm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterOutcome {
    /// Stable independent-cluster identity, e.g. site × robot article × session block.
    pub cluster_id: String,
    /// Exact paired-episode manifest shared by the candidate and all probes.
    pub paired_episode_manifest_digest: String,
    /// Evidence receipt supporting the cluster's independence definition.
    pub independence_receipt_digest: String,
    /// Disposition supplied by the experiment-design/adjudication owner.
    pub independence_disposition: EvidenceDisposition,
    /// Outcome tally for this arm over the paired episodes.
    pub tally: EpisodeTally,
}

/// One low-capacity or input-ablation probe and its outcomes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeRun {
    /// Arm identity, frozen split and declared feature profile.
    pub manifest: ArmManifest,
    /// One paired outcome record per independent cluster.
    pub clusters: Vec<ClusterOutcome>,
}

/// Complete shortcut/input-ablation audit input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeAuditInput {
    /// Frozen comparison scope.
    pub scope: ProbeAuditScope,
    /// Headline candidate manifest and outcomes.
    pub candidate_manifest: ArmManifest,
    /// Candidate outcomes, one record per independent cluster.
    pub candidate_clusters: Vec<ClusterOutcome>,
    /// Low-capacity and input-ablation probes.
    pub probes: Vec<ProbeRun>,
}

/// Descriptive signal for one probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeSignal {
    /// Low-capacity, non-privileged probe is near the candidate.
    ShortcutSolvabilitySignal,
    /// Probe using evaluator-only or identity/seed leakage is near the candidate.
    PrivilegedInputLeakageSignal,
    /// No declared near-candidate signal was observed.
    NoSignal,
    /// Evidence is insufficient for this diagnostic.
    Inconclusive,
}

/// Overall diagnostic disposition. This is not a benchmark-validity certificate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeAuditDisposition {
    /// At least one permitted low-capacity probe approached the candidate.
    ShortcutSolvabilitySignalDetected,
    /// A privileged-input probe approached the candidate.
    PrivilegedInputLeakageSignalDetected,
    /// All required probe arms were evaluable and none approached the candidate.
    NoKnownShortcutSignal,
    /// Split, independence or outcome completeness prevented a diagnostic.
    Inconclusive,
}

/// Per-cluster paired safe-success comparison, in parts per million.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterGap {
    /// Independent-cluster identity.
    pub cluster_id: String,
    /// Exact paired-episode manifest digest.
    pub paired_episode_manifest_digest: String,
    /// Candidate safe-success rate over all episodes in this cluster.
    pub candidate_safe_success_ppm: u32,
    /// Probe safe-success rate over all episodes in this cluster.
    pub probe_safe_success_ppm: u32,
    /// Candidate minus probe safe-success rate; positive means candidate is better.
    pub candidate_minus_probe_gap_ppm: i64,
    /// Independence evidence state.
    pub independence_disposition: EvidenceDisposition,
    /// Candidate Unknown outcomes retained in the denominator.
    pub candidate_unknown: u32,
    /// Probe Unknown outcomes retained in the denominator.
    pub probe_unknown: u32,
    /// Candidate censored outcomes retained in the denominator.
    pub candidate_censored: u32,
    /// Probe censored outcomes retained in the denominator.
    pub probe_censored: u32,
}

/// One probe's deterministic descriptive report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeArmReport {
    /// Stable probe arm identifier.
    pub probe_id: String,
    /// Declared low-capacity model family.
    pub model_family: ModelFamily,
    /// Canonically sorted input families declared by the probe.
    pub input_families: Vec<InputFamily>,
    /// Whether at least one declared input family is privileged.
    pub uses_privileged_inputs: bool,
    /// Number of independent clusters with Qualified independence evidence.
    pub qualified_independent_cluster_count: u16,
    /// Total scheduled episodes across clusters; descriptive denominator.
    pub total_episodes: u64,
    /// Raw control ticks across candidate and probe; descriptive only.
    pub total_control_ticks: u64,
    /// Candidate safe-success rate over all scheduled episodes.
    pub candidate_safe_success_ppm: u32,
    /// Probe safe-success rate over all scheduled episodes.
    pub probe_safe_success_ppm: u32,
    /// Lower middle order statistic of paired cluster gaps (not a confidence bound).
    pub median_gap_lower_ppm: Option<i64>,
    /// Upper middle order statistic of paired cluster gaps (not a confidence bound).
    pub median_gap_upper_ppm: Option<i64>,
    /// Candidate critical-violation episode count.
    pub candidate_critical_safety_violations: u64,
    /// Probe critical-violation episode count.
    pub probe_critical_safety_violations: u64,
    /// Whether unknown/censored outcomes or unqualified clusters prevented confirmation.
    pub evidence_complete: bool,
    /// Descriptive diagnostic result.
    pub signal: ProbeSignal,
    /// Canonically sorted cluster comparisons.
    pub cluster_gaps: Vec<ClusterGap>,
}

/// Full probe-control report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeAuditReport {
    /// Overall diagnostic signal.
    pub disposition: ProbeAuditDisposition,
    /// Frozen benchmark identifier and version.
    pub benchmark_id: String,
    /// Hidden evaluation split digest.
    pub hidden_evaluation_split_digest: String,
    /// Candidate arm identifier.
    pub candidate_arm_id: String,
    /// Per-probe reports in canonical probe-ID order.
    pub probes: Vec<ProbeArmReport>,
    /// This evaluator has no physical action authority.
    pub actuation_authorized: bool,
}

/// Structural or evidence-contract failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeAuditError {
    /// Required identity or digest is empty.
    EmptyIdentity(&'static str),
    /// Scope values are internally inconsistent.
    InvalidScope(&'static str),
    /// A declared digest or scope does not match the frozen benchmark.
    ScopeMismatch { arm_id: String, field: &'static str },
    /// Candidate is not the full policy arm.
    CandidateModelFamilyInvalid,
    /// A probe is not one of the declared low-capacity families.
    ProbeModelFamilyInvalid(String),
    /// Candidate declared privileged input.
    CandidatePrivilegedInput(InputFamily),
    /// Input family is duplicated within one arm.
    DuplicateInputFamily { arm_id: String, family: InputFamily },
    /// Training, calibration and hidden evaluation split digests overlap.
    SplitOverlap(String),
    /// No probe arms were supplied.
    NoProbeArms,
    /// Duplicate candidate/probe arm identifier.
    DuplicateArmId(String),
    /// Candidate/probe cluster count differs from the frozen manifest.
    ClusterCountMismatch {
        arm_id: String,
        expected: usize,
        observed: usize,
    },
    /// Duplicate or empty independent-cluster identifier.
    DuplicateOrEmptyClusterId(String),
    /// Distinct cluster IDs reused one episode-set or independence receipt.
    DuplicateClusterReceipt(String),
    /// A non-privileged probe used an input family absent from the candidate.
    ProbeInputOutsideCandidate { arm_id: String, family: InputFamily },
    /// Missing pairing or independence receipt identity.
    MissingClusterReceipt(String),
    /// One arm omitted a cluster from the frozen manifest.
    MissingCluster(String),
    /// Paired episode or independence-receipt identities disagree.
    PairingMismatch(String),
    /// Candidate and probe do not have the same episode count in a cluster.
    PairedEpisodeCountMismatch(String),
    /// An independence receipt is explicitly rejected.
    RejectedIndependenceEvidence(String),
    /// Supplied episode categories or counts are inconsistent.
    InvalidEpisodeTally { arm_id: String, cluster_id: String },
    /// A rate or aggregate count overflowed its bounded representation.
    ArithmeticOverflow,
}

fn require_nonempty(name: &'static str, value: &str) -> Result<(), ProbeAuditError> {
    if value.trim().is_empty() {
        return Err(ProbeAuditError::EmptyIdentity(name));
    }
    Ok(())
}

fn rate_ppm(successes: u64, episodes: u64) -> Result<u32, ProbeAuditError> {
    if episodes == 0 {
        return Err(ProbeAuditError::ArithmeticOverflow);
    }
    let value = (u128::from(successes) * u128::from(PPM_SCALE)) / u128::from(episodes);
    u32::try_from(value).map_err(|_| ProbeAuditError::ArithmeticOverflow)
}

fn total_add(current: &mut u64, value: u64) -> Result<(), ProbeAuditError> {
    *current = current
        .checked_add(value)
        .ok_or(ProbeAuditError::ArithmeticOverflow)?;
    Ok(())
}

fn validate_scope(scope: &ProbeAuditScope) -> Result<(), ProbeAuditError> {
    require_nonempty("benchmark_id", &scope.benchmark_id)?;
    require_nonempty("benchmark_version", &scope.benchmark_version)?;
    require_nonempty("task_manifest_digest", &scope.task_manifest_digest)?;
    require_nonempty("scenario_manifest_digest", &scope.scenario_manifest_digest)?;
    require_nonempty(
        "robot_configuration_digest",
        &scope.robot_configuration_digest,
    )?;
    require_nonempty("evaluator_digest", &scope.evaluator_digest)?;
    require_nonempty("resource_budget_digest", &scope.resource_budget_digest)?;
    require_nonempty(
        "hidden_evaluation_split_digest",
        &scope.hidden_evaluation_split_digest,
    )?;
    let expected = usize::from(scope.expected_independent_clusters);
    let minimum = usize::from(scope.minimum_qualified_clusters);
    if expected == 0 || expected > MAX_INDEPENDENT_CLUSTERS || minimum == 0 || minimum > expected {
        return Err(ProbeAuditError::InvalidScope(
            "independent-cluster count or minimum is outside the supported range",
        ));
    }
    if scope.near_candidate_gap_ppm > PPM_SCALE {
        return Err(ProbeAuditError::InvalidScope(
            "near-candidate gap must be between zero and one million PPM",
        ));
    }
    Ok(())
}

fn validate_arm_manifest(
    scope: &ProbeAuditScope,
    manifest: &ArmManifest,
    is_candidate: bool,
) -> Result<(), ProbeAuditError> {
    require_nonempty("arm_id", &manifest.arm_id)?;
    require_nonempty("model_artifact_digest", &manifest.model_artifact_digest)?;
    require_nonempty(
        "architecture_audit_digest",
        &manifest.architecture_audit_digest,
    )?;
    require_nonempty("policy_digest", &manifest.policy_digest)?;
    require_nonempty("harness_digest", &manifest.harness_digest)?;
    require_nonempty("training_split_digest", &manifest.training_split_digest)?;
    require_nonempty(
        "calibration_split_digest",
        &manifest.calibration_split_digest,
    )?;
    require_nonempty("evaluation_split_digest", &manifest.evaluation_split_digest)?;

    for (field, observed, expected) in [
        (
            "task_manifest_digest",
            manifest.task_manifest_digest.as_str(),
            scope.task_manifest_digest.as_str(),
        ),
        (
            "scenario_manifest_digest",
            manifest.scenario_manifest_digest.as_str(),
            scope.scenario_manifest_digest.as_str(),
        ),
        (
            "robot_configuration_digest",
            manifest.robot_configuration_digest.as_str(),
            scope.robot_configuration_digest.as_str(),
        ),
        (
            "evaluator_digest",
            manifest.evaluator_digest.as_str(),
            scope.evaluator_digest.as_str(),
        ),
        (
            "resource_budget_digest",
            manifest.resource_budget_digest.as_str(),
            scope.resource_budget_digest.as_str(),
        ),
        (
            "evaluation_split_digest",
            manifest.evaluation_split_digest.as_str(),
            scope.hidden_evaluation_split_digest.as_str(),
        ),
    ] {
        if observed != expected {
            return Err(ProbeAuditError::ScopeMismatch {
                arm_id: manifest.arm_id.clone(),
                field,
            });
        }
    }

    if manifest.training_split_digest == manifest.calibration_split_digest
        || manifest.training_split_digest == manifest.evaluation_split_digest
        || manifest.calibration_split_digest == manifest.evaluation_split_digest
    {
        return Err(ProbeAuditError::SplitOverlap(manifest.arm_id.clone()));
    }

    if manifest.input_families.is_empty() {
        return Err(ProbeAuditError::InvalidScope(
            "each arm must declare at least one input family",
        ));
    }
    let mut families = BTreeSet::new();
    for family in &manifest.input_families {
        if !families.insert(*family) {
            return Err(ProbeAuditError::DuplicateInputFamily {
                arm_id: manifest.arm_id.clone(),
                family: *family,
            });
        }
        if is_candidate && family.is_privileged() {
            return Err(ProbeAuditError::CandidatePrivilegedInput(*family));
        }
    }

    if is_candidate {
        if manifest.model_family != ModelFamily::FullCandidatePolicy {
            return Err(ProbeAuditError::CandidateModelFamilyInvalid);
        }
    } else if manifest.model_family == ModelFamily::FullCandidatePolicy {
        return Err(ProbeAuditError::ProbeModelFamilyInvalid(
            manifest.arm_id.clone(),
        ));
    }
    Ok(())
}

fn index_clusters<'a>(
    arm_id: &str,
    clusters: &'a [ClusterOutcome],
    expected_count: usize,
) -> Result<BTreeMap<String, &'a ClusterOutcome>, ProbeAuditError> {
    if clusters.len() != expected_count {
        return Err(ProbeAuditError::ClusterCountMismatch {
            arm_id: arm_id.to_owned(),
            expected: expected_count,
            observed: clusters.len(),
        });
    }
    let mut indexed = BTreeMap::new();
    let mut paired_manifests = BTreeSet::new();
    let mut independence_receipts = BTreeSet::new();
    for cluster in clusters {
        require_nonempty("cluster_id", &cluster.cluster_id)?;
        require_nonempty(
            "paired_episode_manifest_digest",
            &cluster.paired_episode_manifest_digest,
        )
        .map_err(|_| ProbeAuditError::MissingClusterReceipt(cluster.cluster_id.clone()))?;
        require_nonempty(
            "independence_receipt_digest",
            &cluster.independence_receipt_digest,
        )
        .map_err(|_| ProbeAuditError::MissingClusterReceipt(cluster.cluster_id.clone()))?;
        if !paired_manifests.insert(cluster.paired_episode_manifest_digest.clone())
            || !independence_receipts.insert(cluster.independence_receipt_digest.clone())
        {
            return Err(ProbeAuditError::DuplicateClusterReceipt(
                cluster.cluster_id.clone(),
            ));
        }
        if cluster.independence_disposition == EvidenceDisposition::Rejected {
            return Err(ProbeAuditError::RejectedIndependenceEvidence(
                cluster.cluster_id.clone(),
            ));
        }
        cluster.tally.validate(arm_id, &cluster.cluster_id)?;
        if indexed
            .insert(cluster.cluster_id.clone(), cluster)
            .is_some()
        {
            return Err(ProbeAuditError::DuplicateOrEmptyClusterId(
                cluster.cluster_id.clone(),
            ));
        }
    }
    Ok(indexed)
}

fn paired_cluster_gaps(
    candidate: &BTreeMap<String, &ClusterOutcome>,
    probe: &BTreeMap<String, &ClusterOutcome>,
) -> Result<Vec<ClusterGap>, ProbeAuditError> {
    if candidate.len() != probe.len() {
        return Err(ProbeAuditError::PairingMismatch(
            "candidate and probe cluster sets differ".to_owned(),
        ));
    }
    let mut gaps = Vec::with_capacity(candidate.len());
    for (cluster_id, candidate_cluster) in candidate {
        let probe_cluster = probe
            .get(cluster_id)
            .ok_or_else(|| ProbeAuditError::MissingCluster(cluster_id.clone()))?;
        if candidate_cluster.paired_episode_manifest_digest
            != probe_cluster.paired_episode_manifest_digest
            || candidate_cluster.independence_receipt_digest
                != probe_cluster.independence_receipt_digest
            || candidate_cluster.independence_disposition != probe_cluster.independence_disposition
        {
            return Err(ProbeAuditError::PairingMismatch(cluster_id.clone()));
        }
        if candidate_cluster.tally.episodes != probe_cluster.tally.episodes {
            return Err(ProbeAuditError::PairedEpisodeCountMismatch(
                cluster_id.clone(),
            ));
        }

        let candidate_rate = rate_ppm(
            u64::from(candidate_cluster.tally.safe_successes),
            u64::from(candidate_cluster.tally.episodes),
        )?;
        let probe_rate = rate_ppm(
            u64::from(probe_cluster.tally.safe_successes),
            u64::from(probe_cluster.tally.episodes),
        )?;
        gaps.push(ClusterGap {
            cluster_id: cluster_id.clone(),
            paired_episode_manifest_digest: candidate_cluster
                .paired_episode_manifest_digest
                .clone(),
            candidate_safe_success_ppm: candidate_rate,
            probe_safe_success_ppm: probe_rate,
            candidate_minus_probe_gap_ppm: i64::from(candidate_rate) - i64::from(probe_rate),
            independence_disposition: candidate_cluster.independence_disposition,
            candidate_unknown: candidate_cluster.tally.unknown,
            probe_unknown: probe_cluster.tally.unknown,
            candidate_censored: candidate_cluster.tally.censored,
            probe_censored: probe_cluster.tally.censored,
        });
    }
    Ok(gaps)
}

fn median_range(values: &[i64]) -> Option<(i64, i64)> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let lower = sorted[(sorted.len() - 1) / 2];
    let upper = sorted[sorted.len() / 2];
    Some((lower, upper))
}

fn summarize_probe(
    scope: &ProbeAuditScope,
    candidate_manifest: &ArmManifest,
    candidate_clusters: &BTreeMap<String, &ClusterOutcome>,
    probe: &ProbeRun,
) -> Result<ProbeArmReport, ProbeAuditError> {
    validate_arm_manifest(scope, &probe.manifest, false)?;
    let candidate_families: BTreeSet<_> =
        candidate_manifest.input_families.iter().copied().collect();
    for family in &probe.manifest.input_families {
        if !family.is_privileged() && !candidate_families.contains(family) {
            return Err(ProbeAuditError::ProbeInputOutsideCandidate {
                arm_id: probe.manifest.arm_id.clone(),
                family: *family,
            });
        }
    }
    let expected = usize::from(scope.expected_independent_clusters);
    let probe_clusters = index_clusters(&probe.manifest.arm_id, &probe.clusters, expected)?;
    let mut cluster_gaps = paired_cluster_gaps(candidate_clusters, &probe_clusters)?;

    let mut total_episodes = 0_u64;
    let mut candidate_successes = 0_u64;
    let mut probe_successes = 0_u64;
    let mut candidate_critical = 0_u64;
    let mut probe_critical = 0_u64;
    let mut total_ticks = 0_u64;
    let mut qualified_count = 0_u16;
    let mut complete = true;
    let mut qualified_gaps = Vec::new();

    for gap in &cluster_gaps {
        let candidate = candidate_clusters
            .get(&gap.cluster_id)
            .ok_or_else(|| ProbeAuditError::MissingCluster(gap.cluster_id.clone()))?;
        let probe_cluster = probe_clusters
            .get(&gap.cluster_id)
            .ok_or_else(|| ProbeAuditError::MissingCluster(gap.cluster_id.clone()))?;

        total_add(&mut total_episodes, u64::from(candidate.tally.episodes))?;
        total_add(
            &mut candidate_successes,
            u64::from(candidate.tally.safe_successes),
        )?;
        total_add(
            &mut probe_successes,
            u64::from(probe_cluster.tally.safe_successes),
        )?;
        total_add(
            &mut candidate_critical,
            u64::from(candidate.tally.critical_safety_violations),
        )?;
        total_add(
            &mut probe_critical,
            u64::from(probe_cluster.tally.critical_safety_violations),
        )?;
        total_add(&mut total_ticks, candidate.tally.control_ticks)?;
        total_add(&mut total_ticks, probe_cluster.tally.control_ticks)?;

        if gap.independence_disposition == EvidenceDisposition::Qualified {
            qualified_count = qualified_count
                .checked_add(1)
                .ok_or(ProbeAuditError::ArithmeticOverflow)?;
            qualified_gaps.push(gap.candidate_minus_probe_gap_ppm);
        } else {
            complete = false;
        }

        if candidate.tally.unknown > 0
            || probe_cluster.tally.unknown > 0
            || candidate.tally.censored > 0
            || probe_cluster.tally.censored > 0
        {
            complete = false;
        }
    }

    cluster_gaps.sort_by(|a, b| a.cluster_id.cmp(&b.cluster_id));
    let median = median_range(&qualified_gaps);
    let enough_clusters =
        usize::from(qualified_count) >= usize::from(scope.minimum_qualified_clusters);
    let evidence_complete = complete && enough_clusters && usize::from(qualified_count) == expected;

    let uses_privileged_inputs = probe
        .manifest
        .input_families
        .iter()
        .any(|family| family.is_privileged());
    let near_candidate = evidence_complete
        && median
            .map(|(_, upper)| upper <= i64::from(scope.near_candidate_gap_ppm))
            .unwrap_or(false);

    let signal = if !evidence_complete {
        ProbeSignal::Inconclusive
    } else if near_candidate && uses_privileged_inputs {
        ProbeSignal::PrivilegedInputLeakageSignal
    } else if near_candidate {
        ProbeSignal::ShortcutSolvabilitySignal
    } else {
        ProbeSignal::NoSignal
    };

    let mut input_families = probe.manifest.input_families.clone();
    input_families.sort_unstable();

    Ok(ProbeArmReport {
        probe_id: probe.manifest.arm_id.clone(),
        model_family: probe.manifest.model_family,
        input_families,
        uses_privileged_inputs,
        qualified_independent_cluster_count: qualified_count,
        total_episodes,
        total_control_ticks: total_ticks,
        candidate_safe_success_ppm: rate_ppm(candidate_successes, total_episodes)?,
        probe_safe_success_ppm: rate_ppm(probe_successes, total_episodes)?,
        median_gap_lower_ppm: median.map(|(lower, _)| lower),
        median_gap_upper_ppm: median.map(|(_, upper)| upper),
        candidate_critical_safety_violations: candidate_critical,
        probe_critical_safety_violations: probe_critical,
        evidence_complete,
        signal,
        cluster_gaps,
    })
}

/// Audit low-capacity and input-ablation probes against one frozen candidate.
///
/// The reported median is descriptive over independent-cluster paired gaps.
/// This function intentionally does not calculate p-values, confidence
/// intervals, statistical power, or benchmark-wide validity dispositions.
pub fn audit_probe_controls(input: &ProbeAuditInput) -> Result<ProbeAuditReport, ProbeAuditError> {
    validate_scope(&input.scope)?;
    validate_arm_manifest(&input.scope, &input.candidate_manifest, true)?;
    if input.probes.is_empty() {
        return Err(ProbeAuditError::NoProbeArms);
    }

    let expected = usize::from(input.scope.expected_independent_clusters);
    let candidate_clusters = index_clusters(
        &input.candidate_manifest.arm_id,
        &input.candidate_clusters,
        expected,
    )?;

    let mut arm_ids = BTreeSet::new();
    arm_ids.insert(input.candidate_manifest.arm_id.clone());
    let mut reports = Vec::with_capacity(input.probes.len());
    for probe in &input.probes {
        if !arm_ids.insert(probe.manifest.arm_id.clone()) {
            return Err(ProbeAuditError::DuplicateArmId(
                probe.manifest.arm_id.clone(),
            ));
        }
        if !matches!(
            probe.manifest.model_family,
            ModelFamily::ConstantMajority
                | ModelFamily::SingleFeatureLookup
                | ModelFamily::SmallLinear
                | ModelFamily::ShallowDecisionTree
        ) {
            return Err(ProbeAuditError::ProbeModelFamilyInvalid(
                probe.manifest.arm_id.clone(),
            ));
        }
        reports.push(summarize_probe(
            &input.scope,
            &input.candidate_manifest,
            &candidate_clusters,
            probe,
        )?);
    }
    reports.sort_by(|a, b| a.probe_id.cmp(&b.probe_id));

    let disposition = if reports
        .iter()
        .any(|report| report.signal == ProbeSignal::PrivilegedInputLeakageSignal)
    {
        ProbeAuditDisposition::PrivilegedInputLeakageSignalDetected
    } else if reports
        .iter()
        .any(|report| report.signal == ProbeSignal::ShortcutSolvabilitySignal)
    {
        ProbeAuditDisposition::ShortcutSolvabilitySignalDetected
    } else if reports
        .iter()
        .any(|report| report.signal == ProbeSignal::Inconclusive)
    {
        ProbeAuditDisposition::Inconclusive
    } else {
        ProbeAuditDisposition::NoKnownShortcutSignal
    };

    Ok(ProbeAuditReport {
        disposition,
        benchmark_id: input.scope.benchmark_id.clone(),
        hidden_evaluation_split_digest: input.scope.hidden_evaluation_split_digest.clone(),
        candidate_arm_id: input.candidate_manifest.arm_id.clone(),
        probes: reports,
        actuation_authorized: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope() -> ProbeAuditScope {
        ProbeAuditScope {
            benchmark_id: "robot-manipulation-v1".into(),
            benchmark_version: "1.0.0".into(),
            task_manifest_digest: "tasks-sha256".into(),
            scenario_manifest_digest: "scenarios-sha256".into(),
            robot_configuration_digest: "robot-sha256".into(),
            evaluator_digest: "evaluator-sha256".into(),
            resource_budget_digest: "budget-sha256".into(),
            hidden_evaluation_split_digest: "hidden-confirmatory-sha256".into(),
            expected_independent_clusters: 4,
            minimum_qualified_clusters: 3,
            near_candidate_gap_ppm: 150_000,
        }
    }

    fn manifest(
        arm_id: &str,
        model_family: ModelFamily,
        input_families: Vec<InputFamily>,
    ) -> ArmManifest {
        ArmManifest {
            arm_id: arm_id.into(),
            model_artifact_digest: format!("{arm_id}-model-sha256"),
            architecture_audit_digest: format!("{arm_id}-architecture-audit-sha256"),
            policy_digest: format!("{arm_id}-policy-sha256"),
            harness_digest: format!("{arm_id}-harness-sha256"),
            task_manifest_digest: "tasks-sha256".into(),
            scenario_manifest_digest: "scenarios-sha256".into(),
            robot_configuration_digest: "robot-sha256".into(),
            evaluator_digest: "evaluator-sha256".into(),
            resource_budget_digest: "budget-sha256".into(),
            training_split_digest: format!("{arm_id}-train-sha256"),
            calibration_split_digest: format!("{arm_id}-calibration-sha256"),
            evaluation_split_digest: "hidden-confirmatory-sha256".into(),
            model_family,
            input_families,
        }
    }

    fn tally(safe_successes: u32) -> EpisodeTally {
        EpisodeTally {
            episodes: 10,
            task_successes: safe_successes,
            safe_successes,
            task_failures: 10 - safe_successes,
            unknown: 0,
            timed_out: 0,
            censored: 0,
            critical_safety_violations: 0,
            control_ticks: 10_000,
        }
    }

    fn clusters(safe_successes: u32, arm_id: &str) -> Vec<ClusterOutcome> {
        (0..4)
            .map(|index| {
                let cluster_id = format!("cluster-{index:02}");
                ClusterOutcome {
                    paired_episode_manifest_digest: format!("paired-episodes-{index:02}"),
                    independence_receipt_digest: format!("independence-{index:02}"),
                    independence_disposition: EvidenceDisposition::Qualified,
                    cluster_id,
                    tally: tally(safe_successes),
                }
            })
            .map(|mut cluster| {
                cluster.tally.control_ticks +=
                    u64::try_from(arm_id.len()).expect("test arm ID length fits u64");
                cluster
            })
            .collect()
    }

    fn input(probe_successes: u32, family: InputFamily) -> ProbeAuditInput {
        ProbeAuditInput {
            scope: scope(),
            candidate_manifest: manifest(
                "candidate",
                ModelFamily::FullCandidatePolicy,
                vec![
                    InputFamily::TaskMetadata,
                    InputFamily::LanguageInstruction,
                    InputFamily::VisualObservation,
                    InputFamily::RobotState,
                ],
            ),
            candidate_clusters: clusters(9, "candidate"),
            probes: vec![ProbeRun {
                manifest: manifest("probe", ModelFamily::SingleFeatureLookup, vec![family]),
                clusters: clusters(probe_successes, "probe"),
            }],
        }
    }

    #[test]
    fn low_capacity_probe_near_candidate_emits_shortcut_signal() {
        let report = audit_probe_controls(&input(8, InputFamily::TaskMetadata)).unwrap();
        assert_eq!(
            report.disposition,
            ProbeAuditDisposition::ShortcutSolvabilitySignalDetected
        );
        assert_eq!(report.probes[0].probe_safe_success_ppm, 800_000);
        assert_eq!(report.probes[0].median_gap_lower_ppm, Some(100_000));
        assert_eq!(
            report.probes[0].signal,
            ProbeSignal::ShortcutSolvabilitySignal
        );
        assert!(!report.actuation_authorized);
    }

    #[test]
    fn privileged_label_probe_near_candidate_emits_leakage_signal() {
        let report = audit_probe_controls(&input(9, InputFamily::GroundTruthLabel)).unwrap();
        assert_eq!(
            report.disposition,
            ProbeAuditDisposition::PrivilegedInputLeakageSignalDetected
        );
        assert!(report.probes[0].uses_privileged_inputs);
        assert_eq!(
            report.probes[0].signal,
            ProbeSignal::PrivilegedInputLeakageSignal
        );
    }

    #[test]
    fn environment_seed_is_treated_as_privileged_input() {
        let report = audit_probe_controls(&input(9, InputFamily::EnvironmentSeed)).unwrap();
        assert_eq!(
            report.disposition,
            ProbeAuditDisposition::PrivilegedInputLeakageSignalDetected
        );
    }

    #[test]
    fn low_performing_probe_does_not_emit_shortcut_signal() {
        let report = audit_probe_controls(&input(4, InputFamily::TaskMetadata)).unwrap();
        assert_eq!(
            report.disposition,
            ProbeAuditDisposition::NoKnownShortcutSignal
        );
        assert_eq!(report.probes[0].signal, ProbeSignal::NoSignal);
        assert_eq!(report.probes[0].median_gap_upper_ppm, Some(500_000));
    }

    #[test]
    fn candidate_cannot_declare_privileged_evaluator_input() {
        let mut audit = input(8, InputFamily::TaskMetadata);
        audit
            .candidate_manifest
            .input_families
            .push(InputFamily::EvaluatorState);
        assert_eq!(
            audit_probe_controls(&audit),
            Err(ProbeAuditError::CandidatePrivilegedInput(
                InputFamily::EvaluatorState
            ))
        );
    }

    #[test]
    fn train_calibration_and_hidden_splits_must_be_disjoint() {
        let mut audit = input(8, InputFamily::TaskMetadata);
        audit.probes[0].manifest.training_split_digest =
            audit.probes[0].manifest.evaluation_split_digest.clone();
        assert_eq!(
            audit_probe_controls(&audit),
            Err(ProbeAuditError::SplitOverlap("probe".into()))
        );
    }

    #[test]
    fn wrong_evaluator_scope_is_rejected() {
        let mut audit = input(8, InputFamily::TaskMetadata);
        audit.probes[0].manifest.evaluator_digest = "different-evaluator".into();
        assert_eq!(
            audit_probe_controls(&audit),
            Err(ProbeAuditError::ScopeMismatch {
                arm_id: "probe".into(),
                field: "evaluator_digest",
            })
        );
    }

    #[test]
    fn wrong_hidden_split_is_rejected() {
        let mut audit = input(8, InputFamily::TaskMetadata);
        audit.probes[0].manifest.evaluation_split_digest = "development-split".into();
        assert!(matches!(
            audit_probe_controls(&audit),
            Err(ProbeAuditError::ScopeMismatch {
                field: "evaluation_split_digest",
                ..
            })
        ));
    }

    #[test]
    fn mismatched_paired_episode_manifest_is_rejected() {
        let mut audit = input(8, InputFamily::TaskMetadata);
        audit.probes[0].clusters[1].paired_episode_manifest_digest = "different-pair".into();
        assert!(matches!(
            audit_probe_controls(&audit),
            Err(ProbeAuditError::PairingMismatch(cluster))
                if cluster == "cluster-01"
        ));
    }

    #[test]
    fn missing_cluster_is_rejected() {
        let mut audit = input(8, InputFamily::TaskMetadata);
        audit.probes[0].clusters.pop();
        assert!(matches!(
            audit_probe_controls(&audit),
            Err(ProbeAuditError::ClusterCountMismatch {
                arm_id,
                expected: 4,
                observed: 3
            }) if arm_id == "probe"
        ));
    }

    #[test]
    fn unknown_independence_evidence_makes_audit_inconclusive() {
        let mut audit = input(8, InputFamily::TaskMetadata);
        audit.probes[0].clusters[0].independence_disposition = EvidenceDisposition::Unknown;
        audit.candidate_clusters[0].independence_disposition = EvidenceDisposition::Unknown;
        let report = audit_probe_controls(&audit).unwrap();
        assert_eq!(report.disposition, ProbeAuditDisposition::Inconclusive);
        assert_eq!(report.probes[0].qualified_independent_cluster_count, 3);
    }

    #[test]
    fn rejected_independence_evidence_rejects_input() {
        let mut audit = input(8, InputFamily::TaskMetadata);
        audit.candidate_clusters[0].independence_disposition = EvidenceDisposition::Rejected;
        assert!(matches!(
            audit_probe_controls(&audit),
            Err(ProbeAuditError::RejectedIndependenceEvidence(cluster))
                if cluster == "cluster-00"
        ));
    }

    #[test]
    fn unknown_or_censored_outcomes_prevent_shortcut_confirmation() {
        let mut audit = input(8, InputFamily::TaskMetadata);
        audit.candidate_clusters[0].tally.unknown = 1;
        audit.candidate_clusters[0].tally.task_failures -= 1;
        let report = audit_probe_controls(&audit).unwrap();
        assert_eq!(report.disposition, ProbeAuditDisposition::Inconclusive);
        assert!(!report.probes[0].evidence_complete);
    }

    #[test]
    fn raw_control_ticks_do_not_inflate_independent_cluster_count() {
        let mut audit = input(8, InputFamily::TaskMetadata);
        for cluster in &mut audit.candidate_clusters {
            cluster.tally.control_ticks = 10_000_000_000;
        }
        for cluster in &mut audit.probes[0].clusters {
            cluster.tally.control_ticks = 10_000_000_000;
        }
        let report = audit_probe_controls(&audit).unwrap();
        assert_eq!(report.probes[0].qualified_independent_cluster_count, 4);
        assert_eq!(report.probes[0].total_episodes, 40);
        assert_eq!(report.probes[0].total_control_ticks, 80_000_000_000);
    }

    #[test]
    fn duplicate_input_family_is_rejected() {
        let mut audit = input(8, InputFamily::TaskMetadata);
        audit.probes[0]
            .manifest
            .input_families
            .push(InputFamily::TaskMetadata);
        assert!(matches!(
            audit_probe_controls(&audit),
            Err(ProbeAuditError::DuplicateInputFamily {
                arm_id,
                family: InputFamily::TaskMetadata
            }) if arm_id == "probe"
        ));
    }

    #[test]
    fn duplicate_cluster_identity_is_rejected() {
        let mut audit = input(8, InputFamily::TaskMetadata);
        audit.probes[0].clusters[1].cluster_id = audit.probes[0].clusters[0].cluster_id.clone();
        assert!(matches!(
            audit_probe_controls(&audit),
            Err(ProbeAuditError::DuplicateOrEmptyClusterId(cluster))
                if cluster == "cluster-00"
        ));
    }

    #[test]
    fn invalid_episode_tally_is_rejected() {
        let mut audit = input(8, InputFamily::TaskMetadata);
        audit.probes[0].clusters[0].tally.task_failures = 0;
        assert!(matches!(
            audit_probe_controls(&audit),
            Err(ProbeAuditError::InvalidEpisodeTally { arm_id, .. })
                if arm_id == "probe"
        ));
    }

    #[test]
    fn candidate_must_be_full_policy_and_probe_must_be_low_capacity() {
        let mut audit = input(8, InputFamily::TaskMetadata);
        audit.candidate_manifest.model_family = ModelFamily::SmallLinear;
        assert_eq!(
            audit_probe_controls(&audit),
            Err(ProbeAuditError::CandidateModelFamilyInvalid)
        );

        let mut audit = input(8, InputFamily::TaskMetadata);
        audit.probes[0].manifest.model_family = ModelFamily::FullCandidatePolicy;
        assert!(matches!(
            audit_probe_controls(&audit),
            Err(ProbeAuditError::ProbeModelFamilyInvalid(arm_id))
                if arm_id == "probe"
        ));
    }

    #[test]
    fn no_probe_arms_is_not_silently_treated_as_a_clean_benchmark() {
        let mut audit = input(8, InputFamily::TaskMetadata);
        audit.probes.clear();
        assert_eq!(
            audit_probe_controls(&audit),
            Err(ProbeAuditError::NoProbeArms)
        );
    }

    #[test]
    fn probe_cannot_use_non_privileged_input_unavailable_to_candidate() {
        let mut audit = input(8, InputFamily::ActionHistory);
        assert!(matches!(
            audit_probe_controls(&audit),
            Err(ProbeAuditError::ProbeInputOutsideCandidate {
                arm_id,
                family: InputFamily::ActionHistory
            }) if arm_id == "probe"
        ));
        audit
            .candidate_manifest
            .input_families
            .push(InputFamily::ActionHistory);
        assert!(audit_probe_controls(&audit).is_ok());
    }

    #[test]
    fn one_cluster_receipt_cannot_be_reused_for_another_cluster() {
        let mut audit = input(8, InputFamily::TaskMetadata);
        audit.candidate_clusters[1].paired_episode_manifest_digest = audit.candidate_clusters[0]
            .paired_episode_manifest_digest
            .clone();
        assert!(matches!(
            audit_probe_controls(&audit),
            Err(ProbeAuditError::DuplicateClusterReceipt(cluster))
                if cluster == "cluster-01"
        ));
    }

    #[test]
    fn record_order_does_not_change_canonical_report() {
        let mut audit_a = input(8, InputFamily::TaskMetadata);
        let mut audit_b = audit_a.clone();
        audit_b.candidate_clusters.reverse();
        audit_b.probes[0].clusters.reverse();

        let report_a = audit_probe_controls(&audit_a).unwrap();
        let report_b = audit_probe_controls(&audit_b).unwrap();
        assert_eq!(report_a, report_b);

        audit_a.probes.reverse();
        assert_eq!(audit_probe_controls(&audit_a).unwrap(), report_a);
    }
}
