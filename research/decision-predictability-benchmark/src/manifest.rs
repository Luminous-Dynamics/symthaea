//! Strict, versioned validation for held-out decision-transfer split manifests.
//!
//! This module validates structural integrity and computes a canonical
//! SHA-256 digest. It does not prove that generator or solver claims are true.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const SCHEMA_V1: &str = "luminous.decision-transfer.split-manifest.v1";
pub const MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_SCENARIOS: usize = 100_000;
pub const MAX_GROUPS_PER_AXIS: usize = 100_000;
const DIGEST_PREFIX: &str = "sha256:";
const OPAQUE_ID_HEX_LENGTH: usize = 32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ManifestError {
    EmptyInput,
    InputTooLarge {
        actual: usize,
        maximum: usize,
    },
    InvalidJson(String),
    UnsupportedSchema(String),
    InvalidField(&'static str),
    InvalidIdentifier(&'static str),
    InvalidDigest(&'static str),
    InvalidCommit(&'static str),
    EmptyCollection(&'static str),
    DuplicateValue(&'static str),
    UnknownAxis(String),
    GroupRoleOverlap {
        axis: String,
        group: String,
    },
    GroupRoleMismatch {
        scenario: String,
        axis: String,
        role: String,
    },
    MissingAxisMembership {
        scenario: String,
        axis: String,
    },
    MissingDeclaredGroup {
        axis: String,
        group: String,
    },
    UnreferencedDeclaredGroup {
        axis: String,
        group: String,
    },
    NestingRelationshipMismatch {
        child_axis: String,
        child_group: String,
        parent_axis: String,
    },
    InsufficientIndependentUnits {
        role: String,
        actual: usize,
        minimum: usize,
    },
    CanonicalizationFailure,
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "manifest input is empty"),
            Self::InputTooLarge { actual, maximum } => {
                write!(f, "manifest is {actual} bytes; maximum is {maximum}")
            }
            Self::InvalidJson(reason) => write!(f, "invalid manifest JSON: {reason}"),
            Self::UnsupportedSchema(schema) => write!(f, "unsupported manifest schema: {schema}"),
            Self::InvalidField(field) => write!(f, "invalid manifest field: {field}"),
            Self::InvalidIdentifier(field) => write!(f, "invalid opaque identifier: {field}"),
            Self::InvalidDigest(field) => write!(f, "invalid SHA-256 digest: {field}"),
            Self::InvalidCommit(field) => write!(f, "invalid full commit SHA: {field}"),
            Self::EmptyCollection(field) => write!(f, "empty manifest collection: {field}"),
            Self::DuplicateValue(field) => write!(f, "duplicate manifest value: {field}"),
            Self::UnknownAxis(axis) => write!(f, "unknown split axis: {axis}"),
            Self::GroupRoleOverlap { axis, group } => {
                write!(f, "group {group} overlaps split roles on {axis}")
            }
            Self::GroupRoleMismatch {
                scenario,
                axis,
                role,
            } => write!(
                f,
                "scenario {scenario} assigns {axis} to a role other than {role}"
            ),
            Self::MissingAxisMembership { scenario, axis } => {
                write!(f, "scenario {scenario} has no membership for {axis}")
            }
            Self::MissingDeclaredGroup { axis, group } => write!(
                f,
                "scenario group {group} on {axis} is not declared for its role"
            ),
            Self::UnreferencedDeclaredGroup { axis, group } => {
                write!(f, "declared group {group} on {axis} has no scenario")
            }
            Self::NestingRelationshipMismatch {
                child_axis,
                child_group,
                parent_axis,
            } => write!(
                f,
                "nested group {child_group} on {child_axis} maps to more than one parent {parent_axis}"
            ),
            Self::InsufficientIndependentUnits {
                role,
                actual,
                minimum,
            } => write!(
                f,
                "{role} has {actual} independent units; minimum is {minimum}"
            ),
            Self::CanonicalizationFailure => write!(f, "canonical JSON serialization failed"),
        }
    }
}

impl Error for ManifestError {}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum SplitAxisKind {
    Trajectory,
    InitialConditionFamily,
    ParameterCell,
    ForcingRegime,
    CouplingRegime,
    EquationFamily,
    ObservationProcess,
    InterventionMechanism,
}

impl SplitAxisKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Trajectory => "trajectory",
            Self::InitialConditionFamily => "initial_condition_family",
            Self::ParameterCell => "parameter_cell",
            Self::ForcingRegime => "forcing_regime",
            Self::CouplingRegime => "coupling_regime",
            Self::EquationFamily => "equation_family",
            Self::ObservationProcess => "observation_process",
            Self::InterventionMechanism => "intervention_mechanism",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum SplitRole {
    Train,
    Validation,
    ConfirmatoryTest,
    NegativeControl,
    PositiveControl,
}

impl SplitRole {
    fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
            Self::ConfirmatoryTest => "confirmatory_test",
            Self::NegativeControl => "negative_control",
            Self::PositiveControl => "positive_control",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SplitPolicy {
    GroupDisjoint,
    HeldOutRegion,
    HeldOutFamily,
    HeldOutMechanism,
    HeldOutObservationProcess,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GeneratorIdentity {
    pub repository: String,
    pub commit: String,
    pub generator_id: String,
    pub solver_id: String,
    pub solver_configuration_digest: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaskIdentity {
    pub task_id: String,
    pub task_contract_digest: String,
    pub loss_contract_digest: String,
    pub observation_operator_digest: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SamplingDesign {
    pub primary_independent_unit: SplitAxisKind,
    pub nesting: Vec<SplitAxisKind>,
    pub minimum_independent_units_per_role: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SplitAxis {
    pub axis: SplitAxisKind,
    pub policy: SplitPolicy,
    pub train_group_ids: Vec<String>,
    pub validation_group_ids: Vec<String>,
    pub confirmatory_test_group_ids: Vec<String>,
    #[serde(default)]
    pub negative_control_group_ids: Vec<String>,
    #[serde(default)]
    pub positive_control_group_ids: Vec<String>,
}

impl SplitAxis {
    fn groups_for(&self, role: SplitRole) -> &[String] {
        match role {
            SplitRole::Train => &self.train_group_ids,
            SplitRole::Validation => &self.validation_group_ids,
            SplitRole::ConfirmatoryTest => &self.confirmatory_test_group_ids,
            SplitRole::NegativeControl => &self.negative_control_group_ids,
            SplitRole::PositiveControl => &self.positive_control_group_ids,
        }
    }

    fn all_groups(&self) -> impl Iterator<Item = (&str, SplitRole)> {
        self.train_group_ids
            .iter()
            .map(|id| (id.as_str(), SplitRole::Train))
            .chain(
                self.validation_group_ids
                    .iter()
                    .map(|id| (id.as_str(), SplitRole::Validation)),
            )
            .chain(
                self.confirmatory_test_group_ids
                    .iter()
                    .map(|id| (id.as_str(), SplitRole::ConfirmatoryTest)),
            )
            .chain(
                self.negative_control_group_ids
                    .iter()
                    .map(|id| (id.as_str(), SplitRole::NegativeControl)),
            )
            .chain(
                self.positive_control_group_ids
                    .iter()
                    .map(|id| (id.as_str(), SplitRole::PositiveControl)),
            )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ScenarioGroup {
    pub axis: SplitAxisKind,
    pub group_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Scenario {
    pub scenario_id: String,
    pub split_role: SplitRole,
    pub groups: Vec<ScenarioGroup>,
    pub seed_commitment: String,
    pub reference_data_digest: String,
    pub solver_qualification_digest: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Randomization {
    pub seed_commitment_algorithm: String,
    pub permutation_suite_digest: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FreezeRecord {
    pub protocol_frozen_at: String,
    pub analysis_plan_digest: String,
}

/// The manifest digest is intentionally external: including it inside this
/// object would create an impossible self-reference. Store it in a detached
/// evidence receipt after canonicalization.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SplitManifestV1 {
    pub schema: String,
    pub protocol_revision: String,
    pub generator: GeneratorIdentity,
    pub task: TaskIdentity,
    pub sampling: SamplingDesign,
    pub split_axes: Vec<SplitAxis>,
    pub scenarios: Vec<Scenario>,
    pub randomization: Randomization,
    pub freeze: FreezeRecord,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedManifest {
    manifest: SplitManifestV1,
    canonical_sha256: String,
}

impl ValidatedManifest {
    pub fn manifest(&self) -> &SplitManifestV1 {
        &self.manifest
    }

    /// Integrity identifier, not an authenticity signature or truth proof.
    pub fn canonical_sha256(&self) -> &str {
        &self.canonical_sha256
    }
}

pub fn parse_and_validate(input: &[u8]) -> Result<ValidatedManifest, ManifestError> {
    if input.is_empty() {
        return Err(ManifestError::EmptyInput);
    }
    if input.len() > MAX_MANIFEST_BYTES {
        return Err(ManifestError::InputTooLarge {
            actual: input.len(),
            maximum: MAX_MANIFEST_BYTES,
        });
    }

    // Derived struct deserialization rejects duplicate known fields; every
    // struct uses deny_unknown_fields to reject unreviewed schema extensions.
    let manifest: SplitManifestV1 = serde_json::from_slice(input)
        .map_err(|error| ManifestError::InvalidJson(error.to_string()))?;
    validate_manifest(&manifest)?;

    let value =
        serde_json::to_value(&manifest).map_err(|_| ManifestError::CanonicalizationFailure)?;
    let mut canonical = Vec::new();
    write_canonical_json(&value, &mut canonical)?;
    let digest = sha256_hex(&canonical);

    Ok(ValidatedManifest {
        manifest,
        canonical_sha256: format!("{DIGEST_PREFIX}{digest}"),
    })
}

pub fn validate_manifest(manifest: &SplitManifestV1) -> Result<(), ManifestError> {
    if manifest.schema != SCHEMA_V1 {
        return Err(ManifestError::UnsupportedSchema(manifest.schema.clone()));
    }
    validate_sha256(&manifest.protocol_revision, "protocol_revision")?;
    validate_repository(&manifest.generator.repository)?;
    validate_commit(&manifest.generator.commit, "generator.commit")?;
    validate_nonempty(&manifest.generator.generator_id, "generator.generator_id")?;
    validate_nonempty(&manifest.generator.solver_id, "generator.solver_id")?;
    validate_sha256(
        &manifest.generator.solver_configuration_digest,
        "generator.solver_configuration_digest",
    )?;

    validate_nonempty(&manifest.task.task_id, "task.task_id")?;
    validate_sha256(
        &manifest.task.task_contract_digest,
        "task.task_contract_digest",
    )?;
    validate_sha256(
        &manifest.task.loss_contract_digest,
        "task.loss_contract_digest",
    )?;
    validate_sha256(
        &manifest.task.observation_operator_digest,
        "task.observation_operator_digest",
    )?;

    validate_nesting(&manifest.sampling)?;
    validate_sha256(
        &manifest.randomization.permutation_suite_digest,
        "randomization.permutation_suite_digest",
    )?;
    if manifest.randomization.seed_commitment_algorithm != "sha256" {
        return Err(ManifestError::InvalidField(
            "randomization.seed_commitment_algorithm",
        ));
    }
    validate_timestamp(&manifest.freeze.protocol_frozen_at)?;
    validate_sha256(
        &manifest.freeze.analysis_plan_digest,
        "freeze.analysis_plan_digest",
    )?;

    if manifest.scenarios.is_empty() {
        return Err(ManifestError::EmptyCollection("scenarios"));
    }
    if manifest.scenarios.len() > MAX_SCENARIOS {
        return Err(ManifestError::InvalidField(
            "scenarios exceeds maximum count",
        ));
    }
    if manifest.split_axes.is_empty() {
        return Err(ManifestError::EmptyCollection("split_axes"));
    }

    let mut axes = BTreeMap::new();
    for split_axis in &manifest.split_axes {
        if axes.insert(split_axis.axis, split_axis).is_some() {
            return Err(ManifestError::DuplicateValue("split_axes.axis"));
        }
        validate_axis_groups(split_axis)?;
    }

    if !axes.contains_key(&manifest.sampling.primary_independent_unit) {
        return Err(ManifestError::InvalidField(
            "sampling.primary_independent_unit is not a split axis",
        ));
    }
    for axis in axes.keys() {
        if !manifest.sampling.nesting.contains(axis) {
            return Err(ManifestError::InvalidField(
                "split axis is absent from sampling.nesting",
            ));
        }
    }

    let mut scenario_ids = BTreeSet::new();
    let mut referenced_groups: BTreeMap<SplitAxisKind, BTreeSet<String>> = BTreeMap::new();
    let mut parent_by_child: BTreeMap<(SplitAxisKind, String), (SplitAxisKind, String)> =
        BTreeMap::new();

    for scenario in &manifest.scenarios {
        validate_opaque_id(&scenario.scenario_id, "scenario.scenario_id")?;
        if !scenario_ids.insert(scenario.scenario_id.as_str()) {
            return Err(ManifestError::DuplicateValue("scenario_id"));
        }
        validate_sha256(&scenario.seed_commitment, "scenario.seed_commitment")?;
        validate_sha256(
            &scenario.reference_data_digest,
            "scenario.reference_data_digest",
        )?;
        validate_sha256(
            &scenario.solver_qualification_digest,
            "scenario.solver_qualification_digest",
        )?;

        let mut scenario_axes = BTreeSet::new();
        let mut scenario_groups = BTreeMap::new();
        for membership in &scenario.groups {
            if !scenario_axes.insert(membership.axis) {
                return Err(ManifestError::DuplicateValue("scenario.groups.axis"));
            }
            validate_opaque_id(&membership.group_id, "scenario.groups.group_id")?;
            scenario_groups.insert(membership.axis, membership.group_id.as_str());
            // Only axes listed in split_axes impose a cross-role holdout.
            // Other declared group labels are contextual metadata and may be
            // shared across train/validation/test (e.g. parameter transfer
            // within one equation family).
            if let Some(split_axis) = axes.get(&membership.axis) {
                if !split_axis
                    .groups_for(scenario.split_role)
                    .iter()
                    .any(|id| id == &membership.group_id)
                {
                    let any_role_has_group = split_axis
                        .all_groups()
                        .any(|(id, _)| id == membership.group_id);
                    if any_role_has_group {
                        return Err(ManifestError::GroupRoleMismatch {
                            scenario: scenario.scenario_id.clone(),
                            axis: membership.axis.as_str().to_owned(),
                            role: scenario.split_role.as_str().to_owned(),
                        });
                    }
                    return Err(ManifestError::MissingDeclaredGroup {
                        axis: membership.axis.as_str().to_owned(),
                        group: membership.group_id.clone(),
                    });
                }
                referenced_groups
                    .entry(membership.axis)
                    .or_default()
                    .insert(membership.group_id.clone());
            } else if !manifest.sampling.nesting.contains(&membership.axis) {
                return Err(ManifestError::InvalidField(
                    "scenario context axis is absent from sampling.nesting",
                ));
            }
        }

        // Every axis in the declared outer-to-inner nesting path must be
        // present on every scenario, including axes used only as context.
        for axis in &manifest.sampling.nesting {
            if !scenario_axes.contains(axis) {
                return Err(ManifestError::MissingAxisMembership {
                    scenario: scenario.scenario_id.clone(),
                    axis: axis.as_str().to_owned(),
                });
            }
        }

        // A child group represents one nested sampling unit and therefore
        // cannot be attached to different parent groups in different
        // scenarios. This catches structurally impossible manifests such as
        // one parameter-cell ID reused across unrelated equation families.
        for pair in manifest.sampling.nesting.windows(2) {
            let parent_axis = pair[0];
            let child_axis = pair[1];
            let parent_group = scenario_groups.get(&parent_axis).ok_or_else(|| {
                ManifestError::MissingAxisMembership {
                    scenario: scenario.scenario_id.clone(),
                    axis: parent_axis.as_str().to_owned(),
                }
            })?;
            let child_group = scenario_groups.get(&child_axis).ok_or_else(|| {
                ManifestError::MissingAxisMembership {
                    scenario: scenario.scenario_id.clone(),
                    axis: child_axis.as_str().to_owned(),
                }
            })?;
            let key = (child_axis, (*child_group).to_owned());
            let parent = (parent_axis, (*parent_group).to_owned());
            if let Some(previous_parent) = parent_by_child.insert(key.clone(), parent.clone())
                && previous_parent != parent
            {
                return Err(ManifestError::NestingRelationshipMismatch {
                    child_axis: child_axis.as_str().to_owned(),
                    child_group: key.1,
                    parent_axis: parent_axis.as_str().to_owned(),
                });
            }
        }
    }

    for split_axis in &manifest.split_axes {
        for (group, _) in split_axis.all_groups() {
            if !referenced_groups
                .get(&split_axis.axis)
                .is_some_and(|set| set.contains(group))
            {
                return Err(ManifestError::UnreferencedDeclaredGroup {
                    axis: split_axis.axis.as_str().to_owned(),
                    group: group.to_owned(),
                });
            }
        }
    }

    let primary_axis = axes[&manifest.sampling.primary_independent_unit];
    let minimum = manifest.sampling.minimum_independent_units_per_role;
    for role in [
        SplitRole::Train,
        SplitRole::Validation,
        SplitRole::ConfirmatoryTest,
    ] {
        let independent_units = primary_axis.groups_for(role).len();
        if independent_units < minimum {
            return Err(ManifestError::InsufficientIndependentUnits {
                role: role.as_str().to_owned(),
                actual: independent_units,
                minimum,
            });
        }
    }
    Ok(())
}

fn validate_nesting(sampling: &SamplingDesign) -> Result<(), ManifestError> {
    if sampling.nesting.is_empty() {
        return Err(ManifestError::EmptyCollection("sampling.nesting"));
    }
    let mut seen = BTreeSet::new();
    for axis in &sampling.nesting {
        if !seen.insert(*axis) {
            return Err(ManifestError::DuplicateValue("sampling.nesting"));
        }
    }
    if !seen.contains(&sampling.primary_independent_unit) {
        return Err(ManifestError::InvalidField(
            "primary independent unit is absent from nesting",
        ));
    }
    if sampling.minimum_independent_units_per_role == 0 {
        return Err(ManifestError::InvalidField(
            "sampling.minimum_independent_units_per_role",
        ));
    }
    Ok(())
}

fn validate_axis_groups(axis: &SplitAxis) -> Result<(), ManifestError> {
    let policy_matches = match axis.policy {
        SplitPolicy::GroupDisjoint => true,
        SplitPolicy::HeldOutRegion => axis.axis == SplitAxisKind::ParameterCell,
        SplitPolicy::HeldOutFamily => matches!(
            axis.axis,
            SplitAxisKind::InitialConditionFamily
                | SplitAxisKind::ForcingRegime
                | SplitAxisKind::CouplingRegime
                | SplitAxisKind::EquationFamily
        ),
        SplitPolicy::HeldOutMechanism => axis.axis == SplitAxisKind::InterventionMechanism,
        SplitPolicy::HeldOutObservationProcess => axis.axis == SplitAxisKind::ObservationProcess,
    };
    if !policy_matches {
        return Err(ManifestError::InvalidField(
            "split policy is incompatible with its axis",
        ));
    }
    let lists = [
        (&axis.train_group_ids, SplitRole::Train),
        (&axis.validation_group_ids, SplitRole::Validation),
        (
            &axis.confirmatory_test_group_ids,
            SplitRole::ConfirmatoryTest,
        ),
        (&axis.negative_control_group_ids, SplitRole::NegativeControl),
        (&axis.positive_control_group_ids, SplitRole::PositiveControl),
    ];
    let mut seen = BTreeMap::new();
    let mut total = 0_usize;
    for (ids, role) in lists {
        let mut local = BTreeSet::new();
        for id in ids {
            validate_opaque_id(id, "split_axes.group_id")?;
            if !local.insert(id.as_str()) {
                return Err(ManifestError::DuplicateValue(
                    "split_axes group ID within role",
                ));
            }
            if seen.insert(id.as_str(), role).is_some() {
                return Err(ManifestError::GroupRoleOverlap {
                    axis: axis.axis.as_str().to_owned(),
                    group: id.clone(),
                });
            }
            total += 1;
        }
    }
    if axis.train_group_ids.is_empty()
        || axis.validation_group_ids.is_empty()
        || axis.confirmatory_test_group_ids.is_empty()
    {
        return Err(ManifestError::EmptyCollection(
            "train/validation/confirmatory groups",
        ));
    }
    if total > MAX_GROUPS_PER_AXIS {
        return Err(ManifestError::InvalidField(
            "split axis exceeds maximum group count",
        ));
    }
    Ok(())
}

fn validate_repository(value: &str) -> Result<(), ManifestError> {
    let mut parts = value.split('/');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(owner), Some(name), None)
            if !owner.is_empty()
                && !name.is_empty()
                && owner.chars().all(is_repo_char)
                && name.chars().all(is_repo_char) =>
        {
            Ok(())
        }
        _ => Err(ManifestError::InvalidField(
            "generator.repository must be owner/name",
        )),
    }
}

fn is_repo_char(character: char) -> bool {
    character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.')
}

fn validate_nonempty(value: &str, field: &'static str) -> Result<(), ManifestError> {
    if value.trim().is_empty() || value.len() > 512 || value != value.trim() {
        return Err(ManifestError::InvalidField(field));
    }
    Ok(())
}

fn validate_commit(value: &str, field: &'static str) -> Result<(), ManifestError> {
    if value.len() != 40 || !value.bytes().all(is_lower_hex) {
        return Err(ManifestError::InvalidCommit(field));
    }
    Ok(())
}

fn validate_opaque_id(value: &str, field: &'static str) -> Result<(), ManifestError> {
    if value.len() != OPAQUE_ID_HEX_LENGTH || !value.bytes().all(is_lower_hex) {
        return Err(ManifestError::InvalidIdentifier(field));
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &'static str) -> Result<(), ManifestError> {
    let Some(hex) = value.strip_prefix(DIGEST_PREFIX) else {
        return Err(ManifestError::InvalidDigest(field));
    };
    if hex.len() != 64 || !hex.bytes().all(is_lower_hex) {
        return Err(ManifestError::InvalidDigest(field));
    }
    Ok(())
}

fn is_lower_hex(value: u8) -> bool {
    value.is_ascii_digit() || (b'a'..=b'f').contains(&value)
}

fn validate_timestamp(value: &str) -> Result<(), ManifestError> {
    // Accept the UTC RFC3339 subset YYYY-MM-DDTHH:MM:SS[.fraction]Z.
    // Validate calendar and clock ranges rather than checking punctuation only.
    let invalid = || {
        ManifestError::InvalidField(
            "freeze.protocol_frozen_at must be a valid UTC RFC3339 timestamp",
        )
    };
    let bytes = value.as_bytes();
    if bytes.len() < 20
        || bytes.get(4) != Some(&b'-')
        || bytes.get(7) != Some(&b'-')
        || bytes.get(10) != Some(&b'T')
        || bytes.get(13) != Some(&b':')
        || bytes.get(16) != Some(&b':')
        || bytes.last() != Some(&b'Z')
    {
        return Err(invalid());
    }
    let parse = |start: usize, end: usize| -> Result<u32, ManifestError> {
        let slice = bytes.get(start..end).ok_or_else(invalid)?;
        if slice.is_empty() || !slice.iter().all(u8::is_ascii_digit) {
            return Err(invalid());
        }
        let text = std::str::from_utf8(slice).map_err(|_| invalid())?;
        text.parse::<u32>().map_err(|_| invalid())
    };

    let year = parse(0, 4)?;
    let month = parse(5, 7)?;
    let day = parse(8, 10)?;
    let hour = parse(11, 13)?;
    let minute = parse(14, 16)?;
    let second = parse(17, 19)?;
    if year == 0 || !(1..=12).contains(&month) || hour > 23 || minute > 59 || second > 59 {
        return Err(invalid());
    }

    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return Err(invalid()),
    };
    if day == 0 || day > days_in_month {
        return Err(invalid());
    }

    match bytes.get(19) {
        Some(b'Z') if bytes.len() == 20 => Ok(()),
        Some(b'.') if bytes.len() > 21 && bytes.last() == Some(&b'Z') => {
            let fraction = &bytes[20..bytes.len() - 1];
            if fraction.is_empty() || !fraction.iter().all(u8::is_ascii_digit) {
                return Err(invalid());
            }
            Ok(())
        }
        _ => Err(invalid()),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn write_canonical_json(
    value: &serde_json::Value,
    output: &mut Vec<u8>,
) -> Result<(), ManifestError> {
    match value {
        serde_json::Value::Null => output.extend_from_slice(b"null"),
        serde_json::Value::Bool(value) => {
            output.extend_from_slice(if *value { &b"true"[..] } else { &b"false"[..] })
        }
        serde_json::Value::Number(number) => {
            output.extend_from_slice(number.to_string().as_bytes())
        }
        serde_json::Value::String(value) => {
            let encoded =
                serde_json::to_string(value).map_err(|_| ManifestError::CanonicalizationFailure)?;
            output.extend_from_slice(encoded.as_bytes());
        }
        serde_json::Value::Array(values) => {
            output.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                write_canonical_json(value, output)?;
            }
            output.push(b']');
        }
        serde_json::Value::Object(values) => {
            output.push(b'{');
            for (index, (key, value)) in values.iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                let encoded = serde_json::to_string(key)
                    .map_err(|_| ManifestError::CanonicalizationFailure)?;
                output.extend_from_slice(encoded.as_bytes());
                output.push(b':');
                write_canonical_json(value, output)?;
            }
            output.push(b'}');
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = r#"{
      "schema":"luminous.decision-transfer.split-manifest.v1",
      "protocol_revision":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "generator":{"repository":"Luminous-Dynamics/symthaea","commit":"0123456789abcdef0123456789abcdef01234567","generator_id":"chaos-reference-suite-v1","solver_id":"rk4-reference-v1","solver_configuration_digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
      "task":{"task_id":"lorenz-transfer-decision-v1","task_contract_digest":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","loss_contract_digest":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","observation_operator_digest":"sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"},
      "sampling":{"primary_independent_unit":"parameter_cell","nesting":["equation_family","parameter_cell","trajectory"],"minimum_independent_units_per_role":1},
      "split_axes":[
        {"axis":"equation_family","policy":"held_out_family","train_group_ids":["11111111111111111111111111111111"],"validation_group_ids":["22222222222222222222222222222222"],"confirmatory_test_group_ids":["33333333333333333333333333333333"]},
        {"axis":"parameter_cell","policy":"held_out_region","train_group_ids":["44444444444444444444444444444444"],"validation_group_ids":["55555555555555555555555555555555"],"confirmatory_test_group_ids":["66666666666666666666666666666666"]},
        {"axis":"trajectory","policy":"group_disjoint","train_group_ids":["77777777777777777777777777777777"],"validation_group_ids":["88888888888888888888888888888888"],"confirmatory_test_group_ids":["99999999999999999999999999999999"]}
      ],
      "scenarios":[
        {"scenario_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1","split_role":"train","groups":[{"axis":"equation_family","group_id":"11111111111111111111111111111111"},{"axis":"parameter_cell","group_id":"44444444444444444444444444444444"},{"axis":"trajectory","group_id":"77777777777777777777777777777777"}],"seed_commitment":"sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","reference_data_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","solver_qualification_digest":"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
        {"scenario_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa2","split_role":"validation","groups":[{"axis":"equation_family","group_id":"22222222222222222222222222222222"},{"axis":"parameter_cell","group_id":"55555555555555555555555555555555"},{"axis":"trajectory","group_id":"88888888888888888888888888888888"}],"seed_commitment":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","reference_data_digest":"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","solver_qualification_digest":"sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"},
        {"scenario_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa3","split_role":"confirmatory_test","groups":[{"axis":"equation_family","group_id":"33333333333333333333333333333333"},{"axis":"parameter_cell","group_id":"66666666666666666666666666666666"},{"axis":"trajectory","group_id":"99999999999999999999999999999999"}],"seed_commitment":"sha256:1111111111111111111111111111111111111111111111111111111111111111","reference_data_digest":"sha256:2222222222222222222222222222222222222222222222222222222222222222","solver_qualification_digest":"sha256:3333333333333333333333333333333333333333333333333333333333333333"}
      ],
      "randomization":{"seed_commitment_algorithm":"sha256","permutation_suite_digest":"sha256:4444444444444444444444444444444444444444444444444444444444444444"},
      "freeze":{"protocol_frozen_at":"2026-11-17T12:00:00Z","analysis_plan_digest":"sha256:5555555555555555555555555555555555555555555555555555555555555555"}
    }"#;

    #[test]
    fn accepts_valid_manifest_and_produces_sha256_digest() {
        let parsed = parse_and_validate(VALID.as_bytes()).expect("valid manifest");
        assert_eq!(parsed.manifest().schema, SCHEMA_V1);
        assert_eq!(parsed.manifest().scenarios.len(), 3);
        assert_eq!(parsed.canonical_sha256().len(), 71);
        assert!(parsed.canonical_sha256().starts_with("sha256:"));
    }

    #[test]
    fn rejects_duplicate_json_fields_and_unknown_fields() {
        let duplicate = VALID.replacen(
            r#""schema":"luminous.decision-transfer.split-manifest.v1","#,
            r#""schema":"luminous.decision-transfer.split-manifest.v1","schema":"luminous.decision-transfer.split-manifest.v1","#,
            1,
        );
        assert!(matches!(
            parse_and_validate(duplicate.as_bytes()),
            Err(ManifestError::InvalidJson(_))
        ));

        let unknown = VALID.replacen(
            r#""schema":"luminous.decision-transfer.split-manifest.v1","#,
            r#""schema":"luminous.decision-transfer.split-manifest.v1","unreviewed":true,"#,
            1,
        );
        assert!(matches!(
            parse_and_validate(unknown.as_bytes()),
            Err(ManifestError::InvalidJson(_))
        ));
    }

    #[test]
    fn rejects_group_overlap_between_split_roles() {
        let overlapping = VALID.replace(
            "22222222222222222222222222222222",
            "11111111111111111111111111111111",
        );
        assert!(matches!(
            parse_and_validate(overlapping.as_bytes()),
            Err(ManifestError::GroupRoleOverlap { .. })
        ));
    }

    #[test]
    fn rejects_scenario_assigned_to_wrong_role() {
        let mismatched = VALID.replace(r#""split_role":"validation""#, r#""split_role":"train""#);
        assert!(matches!(
            parse_and_validate(mismatched.as_bytes()),
            Err(ManifestError::GroupRoleMismatch { .. })
        ));
    }

    #[test]
    fn rejects_missing_axis_membership() {
        let needle = r#",{"axis":"trajectory","group_id":"77777777777777777777777777777777"}"#;
        let missing = VALID.replace(needle, "");
        assert!(matches!(
            parse_and_validate(missing.as_bytes()),
            Err(ManifestError::MissingAxisMembership { .. })
        ));
    }

    #[test]
    fn rejects_scenario_group_not_declared_for_role() {
        let needle = r#"{"axis":"parameter_cell","group_id":"44444444444444444444444444444444"}"#;
        let replacement =
            r#"{"axis":"parameter_cell","group_id":"abababababababababababababababab"}"#;
        let missing = VALID.replacen(needle, replacement, 1);
        assert!(matches!(
            parse_and_validate(missing.as_bytes()),
            Err(ManifestError::MissingDeclaredGroup { .. })
        ));
    }

    #[test]
    fn rejects_unreferenced_declared_groups() {
        let extra = VALID.replace(
            r#""train_group_ids":["11111111111111111111111111111111"]"#,
            r#""train_group_ids":["11111111111111111111111111111111","abababababababababababababababab"]"#,
        );
        assert!(matches!(
            parse_and_validate(extra.as_bytes()),
            Err(ManifestError::UnreferencedDeclaredGroup { .. })
        ));
    }

    #[test]
    fn rejects_insufficient_independent_units() {
        let too_few = VALID.replace(
            r#""minimum_independent_units_per_role":1"#,
            r#""minimum_independent_units_per_role":2"#,
        );
        assert!(matches!(
            parse_and_validate(too_few.as_bytes()),
            Err(ManifestError::InsufficientIndependentUnits { .. })
        ));
    }

    #[test]
    fn rejects_bad_opaque_ids_and_digests() {
        let bad_id = VALID.replace("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1", "scenario-1");
        assert!(matches!(
            parse_and_validate(bad_id.as_bytes()),
            Err(ManifestError::InvalidIdentifier(_))
        ));

        let bad_digest = VALID.replace(
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "sha256:abc",
        );
        assert!(matches!(
            parse_and_validate(bad_digest.as_bytes()),
            Err(ManifestError::InvalidDigest(_))
        ));
    }

    #[test]
    fn rejects_unsupported_schema_and_empty_input() {
        assert_eq!(parse_and_validate(b""), Err(ManifestError::EmptyInput));
        let unsupported = VALID.replace(SCHEMA_V1, "luminous.decision-transfer.split-manifest.v2");
        assert!(matches!(
            parse_and_validate(unsupported.as_bytes()),
            Err(ManifestError::UnsupportedSchema(_))
        ));
    }

    #[test]
    fn canonical_digest_ignores_object_key_order() {
        let schema_line = concat!(
            "      ",
            r#""schema":"luminous.decision-transfer.split-manifest.v1","#,
            "\n",
        );
        let without_schema = VALID.replacen(schema_line, "", 1);
        let task_line = concat!("      ", r#""task":{"#);
        let reordered = without_schema.replacen(task_line, &format!("{schema_line}{task_line}"), 1);
        let left = parse_and_validate(VALID.as_bytes()).unwrap();
        let right = parse_and_validate(reordered.as_bytes()).unwrap();
        assert_eq!(left.canonical_sha256(), right.canonical_sha256());
    }

    #[test]
    fn allows_parameter_holdout_within_a_shared_equation_family() {
        let mut value: serde_json::Value = serde_json::from_str(VALID).unwrap();
        let axes = value["split_axes"].as_array_mut().unwrap();
        axes.retain(|axis| axis["axis"] != "equation_family");

        let scenarios = value["scenarios"].as_array_mut().unwrap();
        for scenario in scenarios {
            let groups = scenario["groups"].as_array_mut().unwrap();
            groups.retain(|group| group["axis"] != "equation_family");
            groups.push(serde_json::json!({
                "axis": "equation_family",
                "group_id": "abababababababababababababababab"
            }));
        }

        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(parse_and_validate(&bytes).is_ok());
    }

    #[test]
    fn rejects_calendar_invalid_or_non_utc_timestamps() {
        for timestamp in [
            "2026-13-17T12:00:00Z",
            "2026-02-30T12:00:00Z",
            "2026-11-17T25:00:00Z",
            "2026-11-17T12:60:00Z",
            "2026-11-17T12:00:60Z",
            "2026-11-17T12:00:00+02:00",
            "not-a-time",
        ] {
            let changed = VALID.replace("2026-11-17T12:00:00Z", timestamp);
            assert!(
                matches!(
                    parse_and_validate(changed.as_bytes()),
                    Err(ManifestError::InvalidField(_))
                ),
                "expected invalid timestamp to be rejected: {timestamp}"
            );
        }

        let fractional = VALID.replace("2026-11-17T12:00:00Z", "2026-11-17T12:00:00.123456Z");
        assert!(parse_and_validate(fractional.as_bytes()).is_ok());
    }

    #[test]
    fn rejects_duplicate_scenario_identifiers() {
        let mut value: serde_json::Value = serde_json::from_str(VALID).unwrap();
        let scenarios = value["scenarios"].as_array_mut().unwrap();
        let first_id = scenarios[0]["scenario_id"].clone();
        scenarios[1]["scenario_id"] = first_id;
        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(matches!(
            parse_and_validate(&bytes),
            Err(ManifestError::DuplicateValue("scenario_id"))
        ));
    }

    #[test]
    fn rejects_incompatible_split_policy_and_axis() {
        let mut value: serde_json::Value = serde_json::from_str(VALID).unwrap();
        let axes = value["split_axes"].as_array_mut().unwrap();
        axes[0]["policy"] = serde_json::json!("held_out_region");
        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(matches!(
            parse_and_validate(&bytes),
            Err(ManifestError::InvalidField(
                "split policy is incompatible with its axis"
            ))
        ));
    }

    #[test]
    fn rejects_nested_group_reused_under_different_parent_groups() {
        let mut value: serde_json::Value = serde_json::from_str(VALID).unwrap();
        let axes = value["split_axes"].as_array_mut().unwrap();
        axes.retain(|axis| axis["axis"] == "trajectory");
        value["sampling"]["primary_independent_unit"] = serde_json::json!("trajectory");

        let scenarios = value["scenarios"].as_array_mut().unwrap();
        let train_groups = scenarios[0]["groups"].as_array().unwrap();
        let train_parameter = train_groups
            .iter()
            .find(|group| group["axis"] == "parameter_cell")
            .unwrap()["group_id"]
            .clone();
        let validation_groups = scenarios[1]["groups"].as_array_mut().unwrap();
        let validation_parameter = validation_groups
            .iter_mut()
            .find(|group| group["axis"] == "parameter_cell")
            .unwrap();
        validation_parameter["group_id"] = train_parameter;

        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(matches!(
            parse_and_validate(&bytes),
            Err(ManifestError::NestingRelationshipMismatch { .. })
        ));
    }

    #[test]
    fn requires_contextual_nesting_axis_membership_on_every_scenario() {
        let mut value: serde_json::Value = serde_json::from_str(VALID).unwrap();
        let scenarios = value["scenarios"].as_array_mut().unwrap();
        let groups = scenarios[1]["groups"].as_array_mut().unwrap();
        groups.retain(|group| group["axis"] != "equation_family");
        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(matches!(
            parse_and_validate(&bytes),
            Err(ManifestError::MissingAxisMembership { axis, .. })
                if axis == "equation_family"
        ));
    }

    #[test]
    fn rejects_manifest_size_limit_before_deserialization() {
        let oversized = vec![b' '; MAX_MANIFEST_BYTES + 1];
        assert!(matches!(
            parse_and_validate(&oversized),
            Err(ManifestError::InputTooLarge { .. })
        ));
    }
}
