//! Deterministic, offline reference checks for Symthaea RSI causal attribution.
//!
//! This crate validates declared intervention-manifest structure and synthetic
//! 2x2 fixture arithmetic. It does not execute candidate code, authenticate
//! provenance, establish causal identification, or authorize promotion.

use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Factorial2x2 {
    pub y00: f64,
    pub y10: f64,
    pub y01: f64,
    pub y11: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FactorialContrasts {
    pub a_marginal: f64,
    pub b_marginal: f64,
    pub ab_interaction: f64,
    pub a_when_b_zero: f64,
    pub a_when_b_one: f64,
    pub b_when_a_zero: f64,
    pub b_when_a_one: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContrastError {
    EmptyBlocks,
    EmptyBlockId,
    EmptyClusterId,
    EmptyContextDigest,
    InvalidContextDigest,
    DuplicateBlockId,
    EvaluationContextMismatch,
    NonFiniteInput,
    NonFiniteContrast,
    NonFiniteAggregate,
}

impl fmt::Display for ContrastError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ContrastError {}

impl Factorial2x2 {
    /// Computes marginal effects, conditional effects, and the A×B interaction.
    ///
    /// The outcome scale and direction must be frozen before the experiment.
    /// This function assumes all four outcomes came from matched interventions.
    pub fn contrasts(self) -> Result<FactorialContrasts, ContrastError> {
        let outcomes = [self.y00, self.y10, self.y01, self.y11];
        if outcomes.iter().any(|value| !value.is_finite()) {
            return Err(ContrastError::NonFiniteInput);
        }

        // Conditional effects avoid summing the raw outcomes first, which can
        // overflow even when the contrasts themselves would be representable.
        let a_when_b_zero = self.y10 - self.y00;
        let a_when_b_one = self.y11 - self.y01;
        let b_when_a_zero = self.y01 - self.y00;
        let b_when_a_one = self.y11 - self.y10;

        let values = [
            a_when_b_zero,
            a_when_b_one,
            b_when_a_zero,
            b_when_a_one,
        ];
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ContrastError::NonFiniteContrast);
        }

        let a_marginal = stable_mean(&[a_when_b_zero, a_when_b_one]);
        let b_marginal = stable_mean(&[b_when_a_zero, b_when_a_one]);
        let ab_interaction = a_when_b_one - a_when_b_zero;

        let all_values = [
            a_marginal,
            b_marginal,
            ab_interaction,
            a_when_b_zero,
            a_when_b_one,
            b_when_a_zero,
            b_when_a_one,
        ];
        if all_values.iter().any(|value| !value.is_finite()) {
            return Err(ContrastError::NonFiniteContrast);
        }

        Ok(FactorialContrasts {
            a_marginal,
            b_marginal,
            ab_interaction,
            a_when_b_zero,
            a_when_b_one,
            b_when_a_zero,
            b_when_a_one,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FactorialBlock {
    pub block_id: String,
    pub cluster_id: String,
    pub evaluation_context_digest: String,
    pub outcomes: Factorial2x2,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlockContrast {
    pub block_id: String,
    pub cluster_id: String,
    pub contrasts: FactorialContrasts,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClusterContrast {
    pub cluster_id: String,
    pub blocks: usize,
    pub contrasts: FactorialContrasts,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClusteredContrastSummary {
    /// Per-block effects are retained; they must not be treated as IID rows.
    pub block_contrasts: Vec<BlockContrast>,
    /// Within-cluster means, with each independent cluster retained separately.
    pub cluster_contrasts: Vec<ClusterContrast>,
    /// Equal-weight mean across clusters, not across raw task rows.
    pub equal_cluster_mean: FactorialContrasts,
}

/// Aggregates matched 2×2 blocks without treating within-cluster rows as IID.
///
/// All blocks must share one frozen evaluation-context digest. The cluster ID
/// must come from the externally qualified task/family sampling manifest; this
/// function validates only that it is present and consistent within each block.
/// It computes point estimates only: no confidence interval, p-value, power,
/// independence proof, or promotion decision is provided.
pub fn summarize_equal_cluster_blocks(
    blocks: &[FactorialBlock],
) -> Result<ClusteredContrastSummary, ContrastError> {
    if blocks.is_empty() {
        return Err(ContrastError::EmptyBlocks);
    }

    let mut block_ids = BTreeSet::new();
    let mut expected_context: Option<&str> = None;
    let mut per_cluster: BTreeMap<String, Vec<FactorialContrasts>> = BTreeMap::new();
    let mut block_contrasts = Vec::with_capacity(blocks.len());

    for block in blocks {
        if block.block_id.trim().is_empty() {
            return Err(ContrastError::EmptyBlockId);
        }
        if block.cluster_id.trim().is_empty() {
            return Err(ContrastError::EmptyClusterId);
        }
        if block.evaluation_context_digest.trim().is_empty() {
            return Err(ContrastError::EmptyContextDigest);
        }
        if !is_canonical_sha256(&block.evaluation_context_digest) {
            return Err(ContrastError::InvalidContextDigest);
        }
        if !block_ids.insert(block.block_id.as_str()) {
            return Err(ContrastError::DuplicateBlockId);
        }

        match expected_context {
            Some(context) if context != block.evaluation_context_digest.as_str() => {
                return Err(ContrastError::EvaluationContextMismatch);
            }
            None => expected_context = Some(&block.evaluation_context_digest),
            _ => {}
        }

        let contrasts = block.outcomes.contrasts()?;
        per_cluster
            .entry(block.cluster_id.clone())
            .or_default()
            .push(contrasts);
        block_contrasts.push(BlockContrast {
            block_id: block.block_id.clone(),
            cluster_id: block.cluster_id.clone(),
            contrasts,
        });
    }

    let mut cluster_contrasts = Vec::with_capacity(per_cluster.len());
    for (cluster_id, values) in per_cluster {
        cluster_contrasts.push(ClusterContrast {
            cluster_id,
            blocks: values.len(),
            contrasts: mean_contrasts(&values)?,
        });
    }

    let cluster_values: Vec<_> = cluster_contrasts
        .iter()
        .map(|cluster| cluster.contrasts)
        .collect();
    let equal_cluster_mean = mean_contrasts(&cluster_values)?;

    Ok(ClusteredContrastSummary {
        block_contrasts,
        cluster_contrasts,
        equal_cluster_mean,
    })
}

fn mean_contrasts(
    values: &[FactorialContrasts],
) -> Result<FactorialContrasts, ContrastError> {
    if values.is_empty() {
        return Err(ContrastError::EmptyBlocks);
    }

    let pick = |project: fn(FactorialContrasts) -> f64| {
        let selected: Vec<_> = values.iter().copied().map(project).collect();
        stable_mean(&selected)
    };
    let mean = FactorialContrasts {
        a_marginal: pick(|value| value.a_marginal),
        b_marginal: pick(|value| value.b_marginal),
        ab_interaction: pick(|value| value.ab_interaction),
        a_when_b_zero: pick(|value| value.a_when_b_zero),
        a_when_b_one: pick(|value| value.a_when_b_one),
        b_when_a_zero: pick(|value| value.b_when_a_zero),
        b_when_a_one: pick(|value| value.b_when_a_one),
    };

    let fields = [
        mean.a_marginal,
        mean.b_marginal,
        mean.ab_interaction,
        mean.a_when_b_zero,
        mean.a_when_b_one,
        mean.b_when_a_zero,
        mean.b_when_a_one,
    ];
    if fields.iter().any(|value| !value.is_finite()) {
        return Err(ContrastError::NonFiniteAggregate);
    }
    Ok(mean)
}

/// Scaled Neumaier compensated mean. Input values are expected to be finite.
fn stable_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }

    let scale = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return 0.0;
    }

    let mut sum = 0.0;
    let mut correction = 0.0;
    for raw_value in values {
        let value = *raw_value / scale;
        let next = sum + value;
        if sum.abs() >= value.abs() {
            correction += (sum - next) + value;
        } else {
            correction += (value - next) + sum;
        }
        sum = next;
    }

    let normalized_mean = ((sum + correction) / values.len() as f64).clamp(-1.0, 1.0);
    normalized_mean * scale
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChangeClass {
    Data,
    Harness,
    Algorithm,
    MetaProcedure,
    Configuration,
    EvaluatorOrTaskContract,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChangeUnit {
    pub id: String,
    pub origin_parent_id: Option<String>,
    pub class: ChangeClass,
    #[serde(default)]
    pub requires: Vec<String>,
    #[serde(default)]
    pub conflicts: Vec<String>,
    #[serde(default)]
    pub coupled_group: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InterventionManifest {
    pub schema_version: String,
    pub experiment_id: String,
    pub base_candidate_id: String,
    pub parent_candidate_ids: Vec<String>,
    pub source_tree_digest: String,
    pub executable_digest: String,
    pub source_build_evidence_ref: String,
    pub evaluation_context_digest: String,
    pub task_population_digest: String,
    pub evaluator_digest: String,
    pub seed_plan_digest: String,
    pub budget_digest: String,
    pub analysis_plan_digest: String,
    pub change_units: Vec<ChangeUnit>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestIssue {
    EmptyField(&'static str),
    UnsupportedSchemaVersion(String),
    InvalidSha256(&'static str),
    DuplicateParent(String),
    DuplicateChangeUnit(String),
    DuplicateRelation {
        unit: String,
        relation: &'static str,
        target: String,
    },
    UnknownOriginParent {
        unit: String,
        parent: String,
    },
    UnknownRequiredUnit {
        unit: String,
        required: String,
    },
    UnknownConflictingUnit {
        unit: String,
        conflicting: String,
    },
    SelfRequirement(String),
    SelfConflict(String),
    DependencyCycle,
    AsymmetricConflict {
        unit: String,
        other: String,
    },
    ConflictingRequiredUnit {
        unit: String,
        other: String,
    },
    EmptyCoupledGroup {
        unit: String,
    },
    SingletonCoupledGroup(String),
    ForbiddenChangeClass {
        unit: String,
    },
}

/// Validates manifest structure, not truth or authority of the referenced evidence.
///
/// Digests are checked for canonical syntax only. This function does not verify
/// signatures, source snapshots, executable provenance, task independence,
/// evaluation isolation, or whether a proposed change unit is causally atomic.
pub fn validate_manifest(
    manifest: &InterventionManifest,
) -> Result<(), Vec<ManifestIssue>> {
    let mut issues = Vec::new();

    if manifest.schema_version != "sym-rsi-intervention-manifest-v0" {
        issues.push(ManifestIssue::UnsupportedSchemaVersion(
            manifest.schema_version.clone(),
        ));
    }

    for (field, value) in [
        ("schema_version", manifest.schema_version.as_str()),
        ("experiment_id", manifest.experiment_id.as_str()),
        ("base_candidate_id", manifest.base_candidate_id.as_str()),
        ("source_build_evidence_ref", manifest.source_build_evidence_ref.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(ManifestIssue::EmptyField(field));
        }
    }

    for (field, digest) in [
        ("source_tree_digest", manifest.source_tree_digest.as_str()),
        ("executable_digest", manifest.executable_digest.as_str()),
        ("evaluation_context_digest", manifest.evaluation_context_digest.as_str()),
        ("task_population_digest", manifest.task_population_digest.as_str()),
        ("evaluator_digest", manifest.evaluator_digest.as_str()),
        ("seed_plan_digest", manifest.seed_plan_digest.as_str()),
        ("budget_digest", manifest.budget_digest.as_str()),
        ("analysis_plan_digest", manifest.analysis_plan_digest.as_str()),
    ] {
        if !is_canonical_sha256(digest) {
            issues.push(ManifestIssue::InvalidSha256(field));
        }
    }

    let mut parents: BTreeSet<String> = BTreeSet::new();
    for parent in &manifest.parent_candidate_ids {
        if parent.trim().is_empty() {
            issues.push(ManifestIssue::EmptyField("parent_candidate_ids"));
        } else if !parents.insert(parent.clone()) {
            issues.push(ManifestIssue::DuplicateParent(parent.clone()));
        }
    }

    if manifest.change_units.is_empty() {
        issues.push(ManifestIssue::EmptyField("change_units"));
    }

    let mut units: BTreeMap<String, &ChangeUnit> = BTreeMap::new();
    for unit in &manifest.change_units {
        if unit.id.trim().is_empty() {
            issues.push(ManifestIssue::EmptyField("change_unit.id"));
            continue;
        }
        if units.insert(unit.id.clone(), unit).is_some() {
            issues.push(ManifestIssue::DuplicateChangeUnit(unit.id.clone()));
        }
    }

    let unit_ids: BTreeSet<String> = units.keys().cloned().collect();
    let mut coupled_groups: BTreeMap<String, usize> = BTreeMap::new();

    for (id, unit) in &units {
        if matches!(unit.class, ChangeClass::EvaluatorOrTaskContract) {
            issues.push(ManifestIssue::ForbiddenChangeClass { unit: id.clone() });
        }

        if let Some(parent) = unit.origin_parent_id.as_deref() {
            if !parents.contains(parent) {
                issues.push(ManifestIssue::UnknownOriginParent {
                    unit: id.clone(),
                    parent: parent.to_owned(),
                });
            }
        }

        let mut required = BTreeSet::new();
        for dependency in &unit.requires {
            if dependency == id {
                issues.push(ManifestIssue::SelfRequirement(id.clone()));
            }
            if !required.insert(dependency.as_str()) {
                issues.push(ManifestIssue::DuplicateRelation {
                    unit: id.clone(),
                    relation: "requires",
                    target: dependency.clone(),
                });
            }
            if !unit_ids.contains(dependency) {
                issues.push(ManifestIssue::UnknownRequiredUnit {
                    unit: id.clone(),
                    required: dependency.clone(),
                });
            }
        }

        let mut conflicts = BTreeSet::new();
        for conflicting in &unit.conflicts {
            if conflicting == id {
                issues.push(ManifestIssue::SelfConflict(id.clone()));
            }
            if !conflicts.insert(conflicting.as_str()) {
                issues.push(ManifestIssue::DuplicateRelation {
                    unit: id.clone(),
                    relation: "conflicts",
                    target: conflicting.clone(),
                });
            }
            if !unit_ids.contains(conflicting) {
                issues.push(ManifestIssue::UnknownConflictingUnit {
                    unit: id.clone(),
                    conflicting: conflicting.clone(),
                });
            }
            if required.contains(conflicting.as_str()) {
                issues.push(ManifestIssue::ConflictingRequiredUnit {
                    unit: id.clone(),
                    other: conflicting.clone(),
                });
            }
        }

        if let Some(group) = unit.coupled_group.as_deref() {
            if group.trim().is_empty() {
                issues.push(ManifestIssue::EmptyCoupledGroup { unit: id.clone() });
            } else {
                *coupled_groups.entry(group.to_owned()).or_default() += 1;
            }
        }
    }

    for (id, unit) in &units {
        for other in &unit.conflicts {
            if let Some(other_unit) = units.get(other) {
                if !other_unit.conflicts.iter().any(|candidate| candidate == id) {
                    issues.push(ManifestIssue::AsymmetricConflict {
                        unit: id.clone(),
                        other: other.clone(),
                    });
                }
            }
        }
    }

    for (group, count) in coupled_groups {
        if count < 2 {
            issues.push(ManifestIssue::SingletonCoupledGroup(group));
        }
    }

    if has_dependency_cycle(&units) {
        issues.push(ManifestIssue::DependencyCycle);
    }

    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

/// Parses and structurally validates one intervention manifest JSON document.
pub fn validate_manifest_json(input: &str) -> Result<(), String> {
    let manifest: InterventionManifest =
        serde_json::from_str(input).map_err(|error| format!("invalid manifest JSON: {error}"))?;
    validate_manifest(&manifest).map_err(|issues| format!("manifest rejected: {issues:#?}"))
}

fn has_dependency_cycle(units: &BTreeMap<String, &ChangeUnit>) -> bool {
    let mut indegree: BTreeMap<String, usize> =
        units.keys().cloned().map(|id| (id, 0)).collect();
    let mut dependents: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for (id, unit) in units {
        for required in &unit.requires {
            if units.contains_key(required) {
                *indegree.entry(id.clone()).or_default() += 1;
                dependents
                    .entry(required.clone())
                    .or_default()
                    .push(id.clone());
            }
        }
    }

    let mut ready: BTreeSet<String> = indegree
        .iter()
        .filter_map(|(id, degree)| (*degree == 0).then_some(id.clone()))
        .collect();
    let mut visited = 0;

    while let Some(id) = ready.pop_first() {
        visited += 1;
        if let Some(children) = dependents.get(&id) {
            for child in children {
                if let Some(degree) = indegree.get_mut(child) {
                    *degree -= 1;
                    if *degree == 0 {
                        ready.insert(child.clone());
                    }
                }
            }
        }
    }

    visited != units.len()
}

fn is_canonical_sha256(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64
        && hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureFile {
    fixture_version: String,
    purpose: String,
    claim_boundary: String,
    outcome: OutcomeMetadata,
    contrasts: ContrastMetadata,
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OutcomeMetadata {
    name: String,
    scale: String,
    unit: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContrastMetadata {
    a_marginal: String,
    b_marginal: String,
    ab_interaction: String,
    a_when_b_zero: String,
    a_when_b_one: String,
    b_when_a_zero: String,
    b_when_a_one: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    id: String,
    description: String,
    observed: Option<Observed>,
    expected: Expected,
    variant_context_digests: Option<BTreeMap<String, String>>,
    provenance: Option<Provenance>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Observed {
    y00: Option<f64>,
    y10: Option<f64>,
    y01: Option<f64>,
    y11: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Expected {
    status: Option<String>,
    reason: Option<String>,
    a_marginal: Option<f64>,
    b_marginal: Option<f64>,
    ab_interaction: Option<f64>,
    a_when_b_zero: Option<f64>,
    a_when_b_one: Option<f64>,
    b_when_a_zero: Option<f64>,
    b_when_a_one: Option<f64>,
    causal_contrasts: Option<serde_json::Value>,
    outcome_contrast_status: Option<String>,
    mechanism_claim_status: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Provenance {
    declared_source_tree_digest: String,
    verified_executable_source_tree_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixtureValidationReport {
    pub fixture_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixtureValidationError(pub String);

impl fmt::Display for FixtureValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for FixtureValidationError {}

/// Parses and validates the inert JSON reference corpus.
///
/// The corpus is a test oracle for arithmetic and claim-gating behavior only.
pub fn validate_fixture_file(
    input: &str,
) -> Result<FixtureValidationReport, FixtureValidationError> {
    let file: FixtureFile = serde_json::from_str(input)
        .map_err(|error| FixtureValidationError(format!("invalid fixture JSON: {error}")))?;

    if file.fixture_version != "sym-rsi-causal-lineage-2x2-v0" {
        return Err(FixtureValidationError(format!(
            "unexpected fixture_version: {}",
            file.fixture_version
        )));
    }
    for (field, value) in [
        ("purpose", file.purpose.as_str()),
        ("claim_boundary", file.claim_boundary.as_str()),
        ("outcome.name", file.outcome.name.as_str()),
        ("outcome.scale", file.outcome.scale.as_str()),
        ("outcome.unit", file.outcome.unit.as_str()),
        ("contrasts.a_marginal", file.contrasts.a_marginal.as_str()),
        ("contrasts.b_marginal", file.contrasts.b_marginal.as_str()),
        ("contrasts.ab_interaction", file.contrasts.ab_interaction.as_str()),
        ("contrasts.a_when_b_zero", file.contrasts.a_when_b_zero.as_str()),
        ("contrasts.a_when_b_one", file.contrasts.a_when_b_one.as_str()),
        ("contrasts.b_when_a_zero", file.contrasts.b_when_a_zero.as_str()),
        ("contrasts.b_when_a_one", file.contrasts.b_when_a_one.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(FixtureValidationError(format!("empty fixture metadata: {field}")));
        }
    }

    let mut ids = BTreeSet::new();
    let mut fixture_ids = Vec::with_capacity(file.fixtures.len());
    for fixture in &file.fixtures {
        if fixture.id.trim().is_empty() {
            return Err(FixtureValidationError("fixture id is empty".to_owned()));
        }
        if !ids.insert(fixture.id.as_str()) {
            return Err(FixtureValidationError(format!(
                "duplicate fixture id: {}",
                fixture.id
            )));
        }
        if fixture.description.trim().is_empty() {
            return Err(FixtureValidationError(format!(
                "{}: fixture description is empty",
                fixture.id
            )));
        }

        validate_fixture(fixture)?;
        fixture_ids.push(fixture.id.clone());
    }

    if fixture_ids.is_empty() {
        return Err(FixtureValidationError("fixture corpus is empty".to_owned()));
    }

    Ok(FixtureValidationReport { fixture_ids })
}

fn validate_fixture(fixture: &Fixture) -> Result<(), FixtureValidationError> {
    let fail = |message: &str| {
        FixtureValidationError(format!("{}: {message}", fixture.id))
    };

    match fixture.expected.status.as_deref() {
        Some("identified") => {
            let observed = fixture.observed.as_ref().ok_or_else(|| fail("missing observed outcomes"))?;
            let values = [observed.y00, observed.y10, observed.y01, observed.y11];
            if values.iter().any(|value| value.is_none()) {
                return Err(fail("identified fixture has a missing counterfactual"));
            }
            let outcomes = Factorial2x2 {
                y00: observed.y00.expect("checked above"),
                y10: observed.y10.expect("checked above"),
                y01: observed.y01.expect("checked above"),
                y11: observed.y11.expect("checked above"),
            };
            let actual = outcomes
                .contrasts()
                .map_err(|error| fail(&format!("contrast computation failed: {error}")))?;

            compare_expected("a_marginal", actual.a_marginal, fixture.expected.a_marginal, &fail)?;
            compare_expected("b_marginal", actual.b_marginal, fixture.expected.b_marginal, &fail)?;
            compare_expected("ab_interaction", actual.ab_interaction, fixture.expected.ab_interaction, &fail)?;
            compare_expected("a_when_b_zero", actual.a_when_b_zero, fixture.expected.a_when_b_zero, &fail)?;
            compare_expected("a_when_b_one", actual.a_when_b_one, fixture.expected.a_when_b_one, &fail)?;
            compare_expected("b_when_a_zero", actual.b_when_a_zero, fixture.expected.b_when_a_zero, &fail)?;
            compare_expected("b_when_a_one", actual.b_when_a_one, fixture.expected.b_when_a_one, &fail)?;
        }
        Some("not_identified") => {
            let observed = fixture.observed.as_ref().ok_or_else(|| fail("missing observed record"))?;
            let values = [observed.y00, observed.y10, observed.y01, observed.y11];
            if values.iter().all(|value| value.is_some()) {
                return Err(fail("not_identified fixture contains all four outcomes"));
            }
            if fixture.expected.reason.as_deref() != Some("missing_required_counterfactual")
                || fixture.expected.a_marginal.is_some()
                || fixture.expected.b_marginal.is_some()
                || fixture.expected.ab_interaction.is_some()
            {
                return Err(fail("missing-counterfactual claim gate is inconsistent"));
            }
        }
        Some("reject") => {
            let contexts = fixture
                .variant_context_digests
                .as_ref()
                .ok_or_else(|| fail("missing variant context digests"))?;
            let required_keys = ["y00", "y10", "y01", "y11"];
            if contexts.len() != required_keys.len()
                || required_keys.iter().any(|key| !contexts.contains_key(*key))
            {
                return Err(fail("context map must contain exactly y00/y10/y01/y11"));
            }
            let distinct: BTreeSet<&str> = contexts.values().map(String::as_str).collect();
            if distinct.len() < 2 || fixture.expected.causal_contrasts.is_some() {
                return Err(fail("context mismatch must reject causal contrasts"));
            }
        }
        None if fixture.expected.mechanism_claim_status.as_deref() == Some("blocked") => {
            let provenance = fixture
                .provenance
                .as_ref()
                .ok_or_else(|| fail("missing source/executable provenance fields"))?;
            if provenance.declared_source_tree_digest
                == provenance.verified_executable_source_tree_digest
            {
                return Err(fail("mechanism claim blocked without a source/executable mismatch"));
            }
            if fixture.expected.outcome_contrast_status.as_deref()
                != Some("reported_with_provenance_warning")
            {
                return Err(fail("outcome contrast must remain explicitly provenance-qualified"));
            }
        }
        other => {
            return Err(fail(&format!("unsupported expected status: {other:?}")));
        }
    }

    Ok(())
}

fn compare_expected(
    name: &str,
    actual: f64,
    expected: Option<f64>,
    fail: &impl Fn(&str) -> FixtureValidationError,
) -> Result<(), FixtureValidationError> {
    let Some(expected) = expected else {
        return Err(fail(&format!("missing expected value for {name}")));
    };
    if !expected.is_finite() {
        return Err(fail(&format!("non-finite expected value for {name}")));
    }

    let tolerance = 1e-12 * actual.abs().max(expected.abs()).max(1.0);
    if (actual - expected).abs() > tolerance {
        return Err(fail(&format!(
            "{name} mismatch: computed {actual}, expected {expected}, tolerance {tolerance}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        format!("sha256:{}", byte.to_string().repeat(64))
    }

    fn valid_manifest() -> InterventionManifest {
        InterventionManifest {
            schema_version: "sym-rsi-intervention-manifest-v0".to_owned(),
            experiment_id: "synthetic-two-parent-0001".to_owned(),
            base_candidate_id: "base-0001".to_owned(),
            parent_candidate_ids: vec!["lineage-a".to_owned(), "lineage-b".to_owned()],
            source_tree_digest: digest('a'),
            executable_digest: digest('b'),
            source_build_evidence_ref: "synthetic-unverified-build-ref".to_owned(),
            evaluation_context_digest: digest('c'),
            task_population_digest: digest('d'),
            evaluator_digest: digest('e'),
            seed_plan_digest: digest('f'),
            budget_digest: digest('1'),
            analysis_plan_digest: digest('2'),
            change_units: vec![
                ChangeUnit {
                    id: "change-a".to_owned(),
                    origin_parent_id: Some("lineage-a".to_owned()),
                    class: ChangeClass::Algorithm,
                    requires: vec![],
                    conflicts: vec![],
                    coupled_group: None,
                },
                ChangeUnit {
                    id: "change-b".to_owned(),
                    origin_parent_id: Some("lineage-b".to_owned()),
                    class: ChangeClass::Algorithm,
                    requires: vec![],
                    conflicts: vec![],
                    coupled_group: None,
                },
            ],
        }
    }

    #[test]
    fn synthetic_fixture_corpus_matches_expected_arithmetic_and_gates() {
        let input = include_str!(
            "../../../docs/research/fixtures/SYM_RSI_CAUSAL_LINEAGE_2X2_FIXTURES_V0_2026-11-06.json"
        );
        let report = validate_fixture_file(input).expect("reference fixture corpus should validate");
        assert_eq!(report.fixture_ids.len(), 8);
    }

    #[test]
    fn inert_manifest_example_passes_structural_validation() {
        let input = include_str!(
            "../../../docs/research/fixtures/SYM_RSI_INTERVENTION_MANIFEST_V0_EXAMPLE_2026-11-06.json"
        );
        assert_eq!(validate_manifest_json(input), Ok(()));
    }

    #[test]
    fn factorial_contrasts_recover_pure_synergy() {
        let result = Factorial2x2 {
            y00: 0.0,
            y10: 0.0,
            y01: 0.0,
            y11: 4.0,
        }
        .contrasts()
        .expect("finite outcomes");
        assert_eq!(result.a_marginal, 2.0);
        assert_eq!(result.b_marginal, 2.0);
        assert_eq!(result.ab_interaction, 4.0);
        assert_eq!(result.a_when_b_zero, 0.0);
        assert_eq!(result.a_when_b_one, 4.0);
    }

    #[test]
    fn factorial_contrasts_reject_non_finite_inputs() {
        let result = Factorial2x2 {
            y00: 0.0,
            y10: f64::NAN,
            y01: 0.0,
            y11: 1.0,
        }
        .contrasts();
        assert_eq!(result, Err(ContrastError::NonFiniteInput));
    }

    #[test]
    fn equal_cluster_summary_does_not_weight_large_clusters_more() {
        let blocks = vec![
            FactorialBlock {
                block_id: "small-1".to_owned(),
                cluster_id: "small".to_owned(),
                evaluation_context_digest: digest('c'),
                outcomes: Factorial2x2 { y00: 0.0, y10: 1.0, y01: 0.0, y11: 1.0 },
            },
            FactorialBlock {
                block_id: "large-1".to_owned(),
                cluster_id: "large".to_owned(),
                evaluation_context_digest: "context".to_owned(),
                outcomes: Factorial2x2 { y00: 0.0, y10: 3.0, y01: 0.0, y11: 3.0 },
            },
            FactorialBlock {
                block_id: "large-2".to_owned(),
                cluster_id: "large".to_owned(),
                evaluation_context_digest: "context".to_owned(),
                outcomes: Factorial2x2 { y00: 0.0, y10: 3.0, y01: 0.0, y11: 3.0 },
            },
            FactorialBlock {
                block_id: "large-3".to_owned(),
                cluster_id: "large".to_owned(),
                evaluation_context_digest: "context".to_owned(),
                outcomes: Factorial2x2 { y00: 0.0, y10: 3.0, y01: 0.0, y11: 3.0 },
            },
        ];
        let result = summarize_equal_cluster_blocks(&blocks).expect("matched blocks");
        assert_eq!(result.cluster_contrasts.len(), 2);
        assert_eq!(result.equal_cluster_mean.a_marginal, 2.0);
        assert_eq!(result.block_contrasts.len(), 4);
    }

    #[test]
    fn equal_cluster_summary_rejects_non_digest_context() {
        let blocks = vec![FactorialBlock {
            block_id: "block-a".to_owned(),
            cluster_id: "family-a".to_owned(),
            evaluation_context_digest: "not-a-digest".to_owned(),
            outcomes: Factorial2x2 { y00: 0.0, y10: 1.0, y01: 0.0, y11: 1.0 },
        }];
        assert_eq!(
            summarize_equal_cluster_blocks(&blocks),
            Err(ContrastError::InvalidContextDigest)
        );
    }

    #[test]
    fn scaled_mean_handles_same_sign_near_maximum_values() {
        let value = stable_mean(&[f64::MAX * 0.9, f64::MAX * 0.9]);
        assert!(value.is_finite());
        assert!((value / f64::MAX - 0.9).abs() < 1e-15);
    }

    #[test]
    fn equal_cluster_summary_rejects_context_mismatch() {
        let blocks = vec![
            FactorialBlock {
                block_id: "block-a".to_owned(),
                cluster_id: "family-a".to_owned(),
                evaluation_context_digest: digest('a'),
                outcomes: Factorial2x2 { y00: 0.0, y10: 1.0, y01: 0.0, y11: 1.0 },
            },
            FactorialBlock {
                block_id: "block-b".to_owned(),
                cluster_id: "family-b".to_owned(),
                evaluation_context_digest: digest('b'),
                outcomes: Factorial2x2 { y00: 0.0, y10: 1.0, y01: 0.0, y11: 1.0 },
            },
        ];
        assert_eq!(
            summarize_equal_cluster_blocks(&blocks),
            Err(ContrastError::EvaluationContextMismatch)
        );
    }

    #[test]
    fn valid_manifest_passes_structure_checks() {
        assert_eq!(validate_manifest(&valid_manifest()), Ok(()));
    }

    #[test]
    fn manifest_rejects_unknown_schema_versions() {
        let mut manifest = valid_manifest();
        manifest.schema_version = "sym-rsi-intervention-manifest-v9".to_owned();
        let result = validate_manifest(&manifest).expect_err("unknown schema must fail closed");
        assert!(result.contains(&ManifestIssue::UnsupportedSchemaVersion(
            "sym-rsi-intervention-manifest-v9".to_owned()
        )));
    }

    #[test]
    fn manifest_rejects_dependency_cycles() {
        let mut manifest = valid_manifest();
        manifest.change_units[0].requires = vec!["change-b".to_owned()];
        manifest.change_units[1].requires = vec!["change-a".to_owned()];
        let result = validate_manifest(&manifest).expect_err("cycle must be rejected");
        assert!(result.contains(&ManifestIssue::DependencyCycle));
    }

    #[test]
    fn manifest_rejects_asymmetric_conflicts() {
        let mut manifest = valid_manifest();
        manifest.change_units[0].conflicts = vec!["change-b".to_owned()];
        let result = validate_manifest(&manifest).expect_err("one-sided conflict must be rejected");
        assert!(result.iter().any(|issue| matches!(
            issue,
            ManifestIssue::AsymmetricConflict { unit, other }
                if unit == "change-a" && other == "change-b"
        )));
    }

    #[test]
    fn manifest_rejects_evaluator_or_task_contract_edits() {
        let mut manifest = valid_manifest();
        manifest.change_units[0].class = ChangeClass::EvaluatorOrTaskContract;
        let result = validate_manifest(&manifest).expect_err("candidate cannot edit its judge");
        assert!(result.contains(&ManifestIssue::ForbiddenChangeClass {
            unit: "change-a".to_owned()
        }));
    }

    #[test]
    fn manifest_rejects_malformed_digest_syntax() {
        let mut manifest = valid_manifest();
        manifest.evaluator_digest = "not-a-digest".to_owned();
        let result = validate_manifest(&manifest).expect_err("malformed digest must fail");
        assert!(result.contains(&ManifestIssue::InvalidSha256("evaluator_digest")));
    }

    #[test]
    fn manifest_rejects_dependency_conflict() {
        let mut manifest = valid_manifest();
        manifest.change_units[0].requires = vec!["change-b".to_owned()];
        manifest.change_units[0].conflicts = vec!["change-b".to_owned()];
        manifest.change_units[1].conflicts = vec!["change-a".to_owned()];
        let result = validate_manifest(&manifest).expect_err("required unit cannot conflict");
        assert!(result.contains(&ManifestIssue::ConflictingRequiredUnit {
            unit: "change-a".to_owned(),
            other: "change-b".to_owned()
        }));
    }

    #[test]
    fn manifest_rejects_singleton_coupled_group() {
        let mut manifest = valid_manifest();
        manifest.change_units[0].coupled_group = Some("only-one".to_owned());
        let result = validate_manifest(&manifest).expect_err("one unit is not a coupled group");
        assert!(result.contains(&ManifestIssue::SingletonCoupledGroup(
            "only-one".to_owned()
        )));
    }
}
