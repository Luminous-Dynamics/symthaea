// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Preregistered matched-arm equivalence sealing for GEOM-003D0/D1.
//!
//! Individual reproducibility is not enough for a causal comparison: every arm
//! can be internally sealed while an undeclared second variable differs between
//! them. This crate validates the complete four-arm primary GWT protocol before
//! scientific execution and commits the relationship among arms.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;
use symthaea::cognitive_loop::CognitiveLoopConfig;
use symthaea_geometric_environment_preflight::{
    FORBIDDEN_ENVIRONMENT_VARIABLES, GeomEnvironmentPreflightReport,
};
use symthaea_geometric_run_seal::{
    RunEnvironmentSnapshot, SealedRunEnvironment, SourceIdentity, RuntimeIdentity,
    canonical_json_commitment, seal_run_environment,
};

pub const MATCHED_ARM_SCHEMA: &str = "symthaea.geom.matched-arm-set.v1";
const MATCHED_SET_DOMAIN: &str = "symthaea:geom:matched-arm-set:v1";
const SHARED_ENVIRONMENT_DOMAIN: &str = "symthaea:geom:matched-shared-environment:v1";
const NORMALIZED_CONFIG_DOMAIN: &str = "symthaea:geom:matched-normalized-config:v1";

// Frozen domains from GEOM-003D0A3. D0A4 uses them only to independently
// cross-bind the supplied config/preflight objects to each D0A3 snapshot.
const D0A3_CONFIG_DOMAIN: &str = "symthaea:geom:cognitive-loop-config-json:v1";
const D0A3_PREFLIGHT_DOMAIN: &str = "symthaea:geom:environment-preflight-json:v1";

const PERSISTENCE_ROOT_TOKEN: &str = "<GEOM_ARM_PERSISTENCE_ROOT>";

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
pub enum ArmRole {
    Intact,
    ShamIntact,
    BroadGwtOff,
    HandlerDeliveryBlocked,
}

impl ArmRole {
    pub const ORDERED: [Self; 4] = [
        Self::Intact,
        Self::ShamIntact,
        Self::BroadGwtOff,
        Self::HandlerDeliveryBlocked,
    ];

    pub const fn arm_id(self) -> &'static str {
        match self {
            Self::Intact => "A-intact",
            Self::ShamIntact => "A-prime-sham-intact",
            Self::BroadGwtOff => "B-broad-gwt-off",
            Self::HandlerDeliveryBlocked => "C-handler-delivery-blocked",
        }
    }

    pub const fn expected_gate_plan(self) -> GwtGatePlan {
        match self {
            Self::Intact | Self::ShamIntact => GwtGatePlan::InstalledEnabled,
            Self::BroadGwtOff => GwtGatePlan::AbsentBecauseGwtDisabled,
            Self::HandlerDeliveryBlocked => GwtGatePlan::InstalledBlocked,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GwtGatePlan {
    InstalledEnabled,
    AbsentBecauseGwtDisabled,
    InstalledBlocked,
}

#[derive(Debug, Clone)]
pub struct PrimaryArmInput {
    pub role: ArmRole,
    pub config: CognitiveLoopConfig,
    pub preflight: GeomEnvironmentPreflightReport,
    pub run_environment: SealedRunEnvironment,
    pub gate_plan: GwtGatePlan,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchedArmRecord {
    pub role: ArmRole,
    pub arm_id: String,
    pub gate_plan: GwtGatePlan,
    pub run_environment_commitment: String,
    pub config_commitment: String,
    pub preflight_commitment: String,
    pub normalized_config_commitment: String,
    pub canonical_persistence_root: String,
    pub aesthetic_relative_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchedArmSetSeal {
    pub schema: String,
    pub shared_environment_commitment: String,
    pub arms: Vec<MatchedArmRecord>,
    pub commitment: String,
}

#[derive(Debug)]
pub enum MatchedArmError {
    WrongArmCount { observed: usize },
    MissingRole { role: ArmRole },
    DuplicateRole { role: ArmRole },
    WrongArmId {
        role: ArmRole,
        expected: &'static str,
        observed: String,
    },
    WrongGatePlan {
        role: ArmRole,
        expected: GwtGatePlan,
        observed: GwtGatePlan,
    },
    InvalidRunSeal { role: ArmRole, reason: String },
    ConfigCrossBindingMismatch { role: ArmRole },
    PreflightCrossBindingMismatch { role: ArmRole },
    InvalidPreflight { role: ArmRole, reason: &'static str },
    PersistenceRootMismatch { role: ArmRole },
    DuplicatePersistenceRoot,
    PersistentDatabaseForbidden { role: ArmRole, field: &'static str },
    MissingAestheticPath { role: ArmRole },
    AestheticPathMismatchPreflight { role: ArmRole },
    AestheticPathEscapesRoot { role: ArmRole },
    NonUtf8AestheticRelativePath { role: ArmRole },
    AestheticRelativeLayoutMismatch { role: ArmRole },
    SharedEnvironmentMismatch { role: ArmRole },
    ConfigSerialization { role: ArmRole, reason: String },
    ConfigNotObject { role: ArmRole },
    ConfigFieldMissing { role: ArmRole, field: &'static str },
    ConfigFieldWrongType { role: ArmRole, field: &'static str },
    UnexpectedConfigDifferences { role: ArmRole, differences: Vec<String> },
    BroadGwtDirectionInvalid,
    CommitmentFailure(String),
}

impl fmt::Display for MatchedArmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongArmCount { observed } => {
                write!(f, "primary matched set requires exactly four arms, observed {observed}")
            }
            Self::MissingRole { role } => write!(f, "missing required arm role: {role:?}"),
            Self::DuplicateRole { role } => write!(f, "duplicate arm role: {role:?}"),
            Self::WrongArmId { role, expected, observed } => write!(
                f,
                "wrong arm id for {role:?}: expected {expected}, observed {observed}"
            ),
            Self::WrongGatePlan { role, expected, observed } => write!(
                f,
                "wrong GWT gate plan for {role:?}: expected {expected:?}, observed {observed:?}"
            ),
            Self::InvalidRunSeal { role, reason } => {
                write!(f, "invalid D0A3 run seal for {role:?}: {reason}")
            }
            Self::ConfigCrossBindingMismatch { role } => {
                write!(f, "config does not match D0A3 config commitment for {role:?}")
            }
            Self::PreflightCrossBindingMismatch { role } => {
                write!(f, "preflight does not match D0A3 preflight commitment for {role:?}")
            }
            Self::InvalidPreflight { role, reason } => {
                write!(f, "invalid primary preflight for {role:?}: {reason}")
            }
            Self::PersistenceRootMismatch { role } => write!(
                f,
                "D0A2 and D0A3 persistence roots disagree for {role:?}"
            ),
            Self::DuplicatePersistenceRoot => {
                write!(f, "matched arms must use pairwise-distinct persistence roots")
            }
            Self::PersistentDatabaseForbidden { role, field } => write!(
                f,
                "{field} must be None for primary matched arm {role:?}"
            ),
            Self::MissingAestheticPath { role } => {
                write!(f, "missing aesthetic_memory_path for {role:?}")
            }
            Self::AestheticPathMismatchPreflight { role } => write!(
                f,
                "config aesthetic path does not match accepted preflight for {role:?}"
            ),
            Self::AestheticPathEscapesRoot { role } => write!(
                f,
                "aesthetic path is not strictly inside persistence root for {role:?}"
            ),
            Self::NonUtf8AestheticRelativePath { role } => write!(
                f,
                "aesthetic relative path is not UTF-8 for {role:?}"
            ),
            Self::AestheticRelativeLayoutMismatch { role } => write!(
                f,
                "aesthetic relative layout differs from intact arm for {role:?}"
            ),
            Self::SharedEnvironmentMismatch { role } => write!(
                f,
                "shared scientific environment differs from intact arm for {role:?}"
            ),
            Self::ConfigSerialization { role, reason } => {
                write!(f, "config serialization failed for {role:?}: {reason}")
            }
            Self::ConfigNotObject { role } => {
                write!(f, "serialized CognitiveLoopConfig is not an object for {role:?}")
            }
            Self::ConfigFieldMissing { role, field } => {
                write!(f, "config field {field} missing for {role:?}")
            }
            Self::ConfigFieldWrongType { role, field } => {
                write!(f, "config field {field} has wrong type for {role:?}")
            }
            Self::UnexpectedConfigDifferences { role, differences } => write!(
                f,
                "unexpected config differences for {role:?}: {}",
                differences.join(", ")
            ),
            Self::BroadGwtDirectionInvalid => write!(
                f,
                "broad GWT contrast must be intact enable_gwt=true versus B enable_gwt=false"
            ),
            Self::CommitmentFailure(reason) => write!(f, "matched-set commitment failed: {reason}"),
        }
    }
}

impl std::error::Error for MatchedArmError {}

#[derive(Debug, Clone)]
struct PreparedArm {
    role: ArmRole,
    normalized_config: Value,
    shared_environment: SharedEnvironmentIdentity,
    record: MatchedArmRecord,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct NormalizedSourceIdentity {
    subject_sha: String,
    tree_sha: String,
    clean_tree: bool,
    cargo_lock_blake3: String,
    flake_lock_blake3: Option<String>,
    rust_toolchain_blake3: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct NormalizedRuntimeIdentity {
    rustc_version: String,
    cargo_version: String,
    host_triple: String,
    target_triple: String,
    os: String,
    arch: String,
    hardware_identity: String,
    thread_policy: String,
    cargo_features: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct SharedCampaignIdentity {
    campaign_id: String,
    analysis_authority_revision: String,
    input_schedule_commitment: String,
    arm_order_commitment: String,
    fixed_utc_hour: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct SharedEnvironmentIdentity {
    source: NormalizedSourceIdentity,
    runtime: NormalizedRuntimeIdentity,
    campaign: SharedCampaignIdentity,
    process_environment_commitment: String,
}

#[derive(Serialize)]
struct MatchedSetCommitmentView<'a> {
    schema: &'a str,
    shared_environment_commitment: &'a str,
    arms: &'a [MatchedArmRecord],
}

/// Validate and seal the fixed four-arm primary GEOM GWT protocol.
///
/// This function is strictly pre-run protocol authority. It does not construct
/// a cognitive service, install a gate, execute cycles, or inspect outcomes.
pub fn seal_primary_gwt_matched_set(
    inputs: Vec<PrimaryArmInput>,
) -> Result<MatchedArmSetSeal, MatchedArmError> {
    if inputs.len() != ArmRole::ORDERED.len() {
        return Err(MatchedArmError::WrongArmCount {
            observed: inputs.len(),
        });
    }

    let mut by_role = BTreeMap::new();
    for input in inputs {
        let role = input.role;
        if by_role.contains_key(&role) {
            return Err(MatchedArmError::DuplicateRole { role });
        }
        by_role.insert(role, prepare_arm(input)?);
    }

    for role in ArmRole::ORDERED {
        if !by_role.contains_key(&role) {
            return Err(MatchedArmError::MissingRole { role });
        }
    }

    let intact = arm(&by_role, ArmRole::Intact)?;
    let sham = arm(&by_role, ArmRole::ShamIntact)?;
    let broad = arm(&by_role, ArmRole::BroadGwtOff)?;
    let blocked = arm(&by_role, ArmRole::HandlerDeliveryBlocked)?;

    // Shared scientific environment must be identical across arms after
    // excluding only the separately governed arm-specific fields.
    for candidate in [sham, broad, blocked] {
        if candidate.shared_environment != intact.shared_environment {
            return Err(MatchedArmError::SharedEnvironmentMismatch {
                role: candidate.role,
            });
        }
    }

    // Independently constructed replicas may never share mutable persistence.
    let mut roots = BTreeSet::new();
    for role in ArmRole::ORDERED {
        let prepared = arm(&by_role, role)?;
        if !roots.insert(prepared.record.canonical_persistence_root.clone()) {
            return Err(MatchedArmError::DuplicatePersistenceRoot);
        }
    }

    // Persistence normalization is allowed only if all arms use the same
    // relative aesthetic layout beneath their distinct isolated roots.
    for candidate in [sham, broad, blocked] {
        if candidate.record.aesthetic_relative_path != intact.record.aesthetic_relative_path {
            return Err(MatchedArmError::AestheticRelativeLayoutMismatch {
                role: candidate.role,
            });
        }
    }

    require_no_config_differences(intact, sham)?;
    require_no_config_differences(intact, blocked)?;
    require_broad_gwt_only_difference(intact, broad)?;

    let shared_environment_commitment = canonical_json_commitment(
        SHARED_ENVIRONMENT_DOMAIN,
        &intact.shared_environment,
    )
    .map_err(|err| MatchedArmError::CommitmentFailure(err.to_string()))?;

    let mut records = Vec::with_capacity(4);
    for role in ArmRole::ORDERED {
        records.push(arm(&by_role, role)?.record.clone());
    }

    let view = MatchedSetCommitmentView {
        schema: MATCHED_ARM_SCHEMA,
        shared_environment_commitment: &shared_environment_commitment,
        arms: &records,
    };
    let commitment = canonical_json_commitment(MATCHED_SET_DOMAIN, &view)
        .map_err(|err| MatchedArmError::CommitmentFailure(err.to_string()))?;

    Ok(MatchedArmSetSeal {
        schema: MATCHED_ARM_SCHEMA.to_string(),
        shared_environment_commitment,
        arms: records,
        commitment,
    })
}

fn arm(
    arms: &BTreeMap<ArmRole, PreparedArm>,
    role: ArmRole,
) -> Result<&PreparedArm, MatchedArmError> {
    arms.get(&role).ok_or(MatchedArmError::MissingRole { role })
}

fn prepare_arm(input: PrimaryArmInput) -> Result<PreparedArm, MatchedArmError> {
    let role = input.role;
    let expected_arm_id = role.arm_id();
    let observed_arm_id = input.run_environment.snapshot.campaign.arm_id.clone();
    if observed_arm_id != expected_arm_id {
        return Err(MatchedArmError::WrongArmId {
            role,
            expected: expected_arm_id,
            observed: observed_arm_id,
        });
    }

    let expected_plan = role.expected_gate_plan();
    if input.gate_plan != expected_plan {
        return Err(MatchedArmError::WrongGatePlan {
            role,
            expected: expected_plan,
            observed: input.gate_plan,
        });
    }

    // Recompute D0A3's authoritative run seal so a stale/tampered serialized
    // commitment cannot be smuggled into the matched-set authority.
    let resealed = seal_run_environment(input.run_environment.snapshot.clone(), None).map_err(|err| {
        MatchedArmError::InvalidRunSeal {
            role,
            reason: err.to_string(),
        }
    })?;
    if resealed.commitment != input.run_environment.commitment {
        return Err(MatchedArmError::InvalidRunSeal {
            role,
            reason: "serialized commitment does not match snapshot".to_string(),
        });
    }

    validate_primary_preflight(role, &input.preflight)?;

    if input.run_environment.snapshot.canonical_persistence_root
        != input.preflight.canonical_persistence_root
    {
        return Err(MatchedArmError::PersistenceRootMismatch { role });
    }

    let config_commitment = canonical_json_commitment(D0A3_CONFIG_DOMAIN, &input.config)
        .map_err(|err| MatchedArmError::ConfigSerialization {
            role,
            reason: err.to_string(),
        })?;
    if config_commitment != input.run_environment.snapshot.config_commitment {
        return Err(MatchedArmError::ConfigCrossBindingMismatch { role });
    }

    let preflight_commitment = canonical_json_commitment(D0A3_PREFLIGHT_DOMAIN, &input.preflight)
        .map_err(|err| MatchedArmError::CommitmentFailure(err.to_string()))?;
    if preflight_commitment != input.run_environment.snapshot.preflight_commitment {
        return Err(MatchedArmError::PreflightCrossBindingMismatch { role });
    }

    let run_hour = normalize_hour(input.run_environment.snapshot.campaign.fixed_utc_hour)
        .ok_or(MatchedArmError::InvalidPreflight {
            role,
            reason: "run fixed UTC hour invalid",
        })?;
    let preflight_hour = normalize_hour(input.preflight.fixed_utc_hour).ok_or(
        MatchedArmError::InvalidPreflight {
            role,
            reason: "preflight fixed UTC hour invalid",
        },
    )?;
    if run_hour.to_bits() != preflight_hour.to_bits() {
        return Err(MatchedArmError::InvalidPreflight {
            role,
            reason: "preflight/run fixed UTC hour mismatch",
        });
    }

    let (normalized_config, aesthetic_relative_path) =
        normalized_config(role, &input.config, &input.preflight)?;
    let normalized_config_commitment = canonical_json_commitment(
        NORMALIZED_CONFIG_DOMAIN,
        &normalized_config,
    )
    .map_err(|err| MatchedArmError::CommitmentFailure(err.to_string()))?;

    let shared_environment = shared_environment_identity(&input.run_environment.snapshot);

    Ok(PreparedArm {
        role,
        normalized_config,
        shared_environment,
        record: MatchedArmRecord {
            role,
            arm_id: expected_arm_id.to_string(),
            gate_plan: input.gate_plan,
            run_environment_commitment: input.run_environment.commitment,
            config_commitment,
            preflight_commitment,
            normalized_config_commitment,
            canonical_persistence_root: input.preflight.canonical_persistence_root,
            aesthetic_relative_path,
        },
    })
}

fn validate_primary_preflight(
    role: ArmRole,
    preflight: &GeomEnvironmentPreflightReport,
) -> Result<(), MatchedArmError> {
    if !preflight.genesis_phrase_present {
        return Err(MatchedArmError::InvalidPreflight {
            role,
            reason: "genesis phrase not present",
        });
    }
    if preflight.async_training || preflight.online_learning {
        return Err(MatchedArmError::InvalidPreflight {
            role,
            reason: "adaptive training enabled",
        });
    }
    if !preflight.persistence_root_empty {
        return Err(MatchedArmError::InvalidPreflight {
            role,
            reason: "persistence root was not empty",
        });
    }
    if preflight.memory_db_path.is_some() {
        return Err(MatchedArmError::PersistentDatabaseForbidden {
            role,
            field: "memory_db_path",
        });
    }
    if preflight.epistemic_auditor_db_path.is_some() {
        return Err(MatchedArmError::PersistentDatabaseForbidden {
            role,
            field: "epistemic_auditor_db_path",
        });
    }

    if preflight.environment.variables.len() != FORBIDDEN_ENVIRONMENT_VARIABLES.len() {
        return Err(MatchedArmError::InvalidPreflight {
            role,
            reason: "forbidden environment contract has wrong cardinality",
        });
    }
    for expected in FORBIDDEN_ENVIRONMENT_VARIABLES {
        let mut matching = preflight
            .environment
            .variables
            .iter()
            .filter(|entry| entry.name == expected);
        let Some(entry) = matching.next() else {
            return Err(MatchedArmError::InvalidPreflight {
                role,
                reason: "forbidden environment channel missing",
            });
        };
        if matching.next().is_some() || entry.present {
            return Err(MatchedArmError::InvalidPreflight {
                role,
                reason: "forbidden environment channel duplicated or present",
            });
        }
    }

    Ok(())
}

fn normalized_config(
    role: ArmRole,
    config: &CognitiveLoopConfig,
    preflight: &GeomEnvironmentPreflightReport,
) -> Result<(Value, String), MatchedArmError> {
    if config.memory_db_path.is_some() {
        return Err(MatchedArmError::PersistentDatabaseForbidden {
            role,
            field: "memory_db_path",
        });
    }
    if config.epistemic_auditor_db_path.is_some() {
        return Err(MatchedArmError::PersistentDatabaseForbidden {
            role,
            field: "epistemic_auditor_db_path",
        });
    }

    let aesthetic = config
        .aesthetic_memory_path
        .as_deref()
        .ok_or(MatchedArmError::MissingAestheticPath { role })?;
    if aesthetic != preflight.aesthetic_memory_path {
        return Err(MatchedArmError::AestheticPathMismatchPreflight { role });
    }

    let root = Path::new(&preflight.canonical_persistence_root);
    let aesthetic_path = Path::new(aesthetic);
    let relative = aesthetic_path
        .strip_prefix(root)
        .map_err(|_| MatchedArmError::AestheticPathEscapesRoot { role })?;
    if relative.as_os_str().is_empty() {
        return Err(MatchedArmError::AestheticPathEscapesRoot { role });
    }
    let relative = relative
        .to_str()
        .ok_or(MatchedArmError::NonUtf8AestheticRelativePath { role })?
        .replace('\\', "/");

    let mut value = serde_json::to_value(config).map_err(|err| {
        MatchedArmError::ConfigSerialization {
            role,
            reason: err.to_string(),
        }
    })?;
    let object = value
        .as_object_mut()
        .ok_or(MatchedArmError::ConfigNotObject { role })?;
    if !object.contains_key("aesthetic_memory_path") {
        return Err(MatchedArmError::ConfigFieldMissing {
            role,
            field: "aesthetic_memory_path",
        });
    }
    object.insert(
        "aesthetic_memory_path".to_string(),
        Value::String(format!("{PERSISTENCE_ROOT_TOKEN}/{relative}")),
    );

    Ok((value, relative))
}

fn shared_environment_identity(snapshot: &RunEnvironmentSnapshot) -> SharedEnvironmentIdentity {
    SharedEnvironmentIdentity {
        source: normalized_source(&snapshot.source),
        runtime: normalized_runtime(&snapshot.runtime),
        campaign: SharedCampaignIdentity {
            campaign_id: snapshot.campaign.campaign_id.clone(),
            analysis_authority_revision: snapshot.campaign.analysis_authority_revision.clone(),
            input_schedule_commitment: snapshot
                .campaign
                .input_schedule_commitment
                .to_ascii_lowercase(),
            arm_order_commitment: snapshot.campaign.arm_order_commitment.to_ascii_lowercase(),
            fixed_utc_hour: normalize_hour(snapshot.campaign.fixed_utc_hour)
                .unwrap_or(snapshot.campaign.fixed_utc_hour),
        },
        process_environment_commitment: snapshot
            .process_environment_commitment
            .to_ascii_lowercase(),
    }
}

fn normalized_source(source: &SourceIdentity) -> NormalizedSourceIdentity {
    NormalizedSourceIdentity {
        subject_sha: source.subject_sha.to_ascii_lowercase(),
        tree_sha: source.tree_sha.to_ascii_lowercase(),
        clean_tree: source.clean_tree,
        cargo_lock_blake3: source.cargo_lock_blake3.to_ascii_lowercase(),
        flake_lock_blake3: source.flake_lock_blake3.as_ref().map(|value| value.to_ascii_lowercase()),
        rust_toolchain_blake3: source
            .rust_toolchain_blake3
            .as_ref()
            .map(|value| value.to_ascii_lowercase()),
    }
}

fn normalized_runtime(runtime: &RuntimeIdentity) -> NormalizedRuntimeIdentity {
    let mut features: Vec<String> = runtime
        .cargo_features
        .iter()
        .map(|feature| feature.trim().to_string())
        .filter(|feature| !feature.is_empty())
        .collect();
    features.sort();
    features.dedup();
    NormalizedRuntimeIdentity {
        rustc_version: runtime.rustc_version.clone(),
        cargo_version: runtime.cargo_version.clone(),
        host_triple: runtime.host_triple.clone(),
        target_triple: runtime.target_triple.clone(),
        os: runtime.os.clone(),
        arch: runtime.arch.clone(),
        hardware_identity: runtime.hardware_identity.clone(),
        thread_policy: runtime.thread_policy.clone(),
        cargo_features: features,
    }
}

fn normalize_hour(hour: f64) -> Option<f64> {
    if !hour.is_finite() || !(0.0..24.0).contains(&hour) {
        return None;
    }
    Some(if hour == 0.0 { 0.0 } else { hour })
}

fn require_no_config_differences(
    intact: &PreparedArm,
    candidate: &PreparedArm,
) -> Result<(), MatchedArmError> {
    let differences = config_differences(&intact.normalized_config, &candidate.normalized_config);
    if differences.is_empty() {
        Ok(())
    } else {
        Err(MatchedArmError::UnexpectedConfigDifferences {
            role: candidate.role,
            differences,
        })
    }
}

fn require_broad_gwt_only_difference(
    intact: &PreparedArm,
    broad: &PreparedArm,
) -> Result<(), MatchedArmError> {
    let differences = config_differences(&intact.normalized_config, &broad.normalized_config);
    if differences != ["/enable_gwt".to_string()] {
        return Err(MatchedArmError::UnexpectedConfigDifferences {
            role: broad.role,
            differences,
        });
    }

    if config_bool(&intact.normalized_config, intact.role, "enable_gwt")? != true
        || config_bool(&broad.normalized_config, broad.role, "enable_gwt")? != false
    {
        return Err(MatchedArmError::BroadGwtDirectionInvalid);
    }
    Ok(())
}

fn config_bool(
    config: &Value,
    role: ArmRole,
    field: &'static str,
) -> Result<bool, MatchedArmError> {
    let object = config
        .as_object()
        .ok_or(MatchedArmError::ConfigNotObject { role })?;
    let value = object
        .get(field)
        .ok_or(MatchedArmError::ConfigFieldMissing { role, field })?;
    value
        .as_bool()
        .ok_or(MatchedArmError::ConfigFieldWrongType { role, field })
}

fn config_differences(left: &Value, right: &Value) -> Vec<String> {
    let mut differences = Vec::new();
    collect_differences(left, right, "", &mut differences);
    differences.sort();
    differences
}

fn collect_differences(left: &Value, right: &Value, path: &str, out: &mut Vec<String>) {
    match (left, right) {
        (Value::Object(left), Value::Object(right)) => {
            let keys: BTreeSet<&str> = left
                .keys()
                .map(String::as_str)
                .chain(right.keys().map(String::as_str))
                .collect();
            for key in keys {
                let child = format!("{path}/{}", escape_pointer_token(key));
                match (left.get(key), right.get(key)) {
                    (Some(left), Some(right)) => collect_differences(left, right, &child, out),
                    _ => out.push(child),
                }
            }
        }
        (Value::Array(left), Value::Array(right)) => {
            if left.len() != right.len() {
                out.push(nonempty_pointer(path));
                return;
            }
            for (index, (left, right)) in left.iter().zip(right.iter()).enumerate() {
                let child = format!("{path}/{index}");
                collect_differences(left, right, &child, out);
            }
        }
        _ if left != right => out.push(nonempty_pointer(path)),
        _ => {}
    }
}

fn nonempty_pointer(path: &str) -> String {
    if path.is_empty() {
        "/".to_string()
    } else {
        path.to_string()
    }
}

fn escape_pointer_token(token: &str) -> String {
    token.replace('~', "~0").replace('/', "~1")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use symthaea_geometric_environment_preflight::{
        EnvironmentSnapshot, preflight_primary_campaign_with_environment,
    };
    use symthaea_geometric_run_seal::{
        CampaignIdentity, SourceIdentity, RuntimeIdentity,
        build_run_environment_snapshot, seal_run_environment,
    };

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    fn temp_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "symthaea_geom_matched_{label}_{}_{}",
            std::process::id(),
            NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        root.canonicalize().unwrap()
    }

    fn hex64(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn source() -> SourceIdentity {
        SourceIdentity {
            subject_sha: "a".repeat(40),
            tree_sha: "b".repeat(40),
            clean_tree: true,
            cargo_lock_blake3: hex64('c'),
            flake_lock_blake3: Some(hex64('d')),
            rust_toolchain_blake3: Some(hex64('e')),
        }
    }

    fn runtime(hardware: &str) -> RuntimeIdentity {
        RuntimeIdentity {
            rustc_version: "rustc 1.96.0".to_string(),
            cargo_version: "cargo 1.96.0".to_string(),
            host_triple: "x86_64-unknown-linux-gnu".to_string(),
            target_triple: "x86_64-unknown-linux-gnu".to_string(),
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            hardware_identity: hardware.to_string(),
            thread_policy: "single-thread".to_string(),
            cargo_features: vec!["scientific_method".to_string(), "mathematics".to_string()],
        }
    }

    fn config(root: &Path, enable_gwt: bool, relative: &str) -> CognitiveLoopConfig {
        let mut config = CognitiveLoopConfig::default();
        config.genesis_phrase = Some("geom-matched-seed".to_string());
        config.async_training = false;
        config.enable_online_learning = false;
        config.enable_gwt = enable_gwt;
        config.aesthetic_memory_path = Some(root.join(relative).to_string_lossy().into_owned());
        config.memory_db_path = None;
        config.epistemic_auditor_db_path = None;
        config
    }

    fn arm_input(
        role: ArmRole,
        root: &Path,
        enable_gwt: bool,
        relative: &str,
        hardware: &str,
    ) -> PrimaryArmInput {
        let config = config(root, enable_gwt, relative);
        let preflight = preflight_primary_campaign_with_environment(
            &config,
            12.5,
            root,
            EnvironmentSnapshot::from_present_names(&[]),
        )
        .unwrap();
        let campaign = CampaignIdentity {
            campaign_id: "GEOM-003D1-HOLDOUT".to_string(),
            arm_id: role.arm_id().to_string(),
            analysis_authority_revision: "GEOM-003D0B-v1".to_string(),
            input_schedule_commitment: hex64('2'),
            arm_order_commitment: hex64('3'),
            fixed_utc_hour: 12.5,
        };
        let snapshot = build_run_environment_snapshot(
            source(),
            runtime(hardware),
            campaign,
            &config,
            &preflight,
            hex64('f'),
        )
        .unwrap();
        let run_environment = seal_run_environment(snapshot, None).unwrap();
        PrimaryArmInput {
            role,
            config,
            preflight,
            run_environment,
            gate_plan: role.expected_gate_plan(),
        }
    }

    fn valid_inputs() -> (Vec<PrimaryArmInput>, Vec<PathBuf>) {
        let a = temp_root("a");
        let ap = temp_root("ap");
        let b = temp_root("b");
        let c = temp_root("c");
        let inputs = vec![
            arm_input(ArmRole::Intact, &a, true, "aesthetic.json", "cpu-1"),
            arm_input(
                ArmRole::ShamIntact,
                &ap,
                true,
                "aesthetic.json",
                "cpu-1",
            ),
            arm_input(
                ArmRole::BroadGwtOff,
                &b,
                false,
                "aesthetic.json",
                "cpu-1",
            ),
            arm_input(
                ArmRole::HandlerDeliveryBlocked,
                &c,
                true,
                "aesthetic.json",
                "cpu-1",
            ),
        ];
        (inputs, vec![a, ap, b, c])
    }

    fn cleanup(roots: Vec<PathBuf>) {
        for root in roots {
            std::fs::remove_dir_all(root).ok();
        }
    }

    #[test]
    fn valid_primary_matched_set_passes_and_input_order_does_not_matter() {
        let (inputs, roots) = valid_inputs();
        let seal = seal_primary_gwt_matched_set(inputs.clone()).unwrap();
        assert_eq!(seal.arms.len(), 4);
        assert_eq!(seal.arms[0].role, ArmRole::Intact);
        assert_eq!(seal.arms[2].role, ArmRole::BroadGwtOff);

        let reversed: Vec<_> = inputs.into_iter().rev().collect();
        let reversed_seal = seal_primary_gwt_matched_set(reversed).unwrap();
        assert_eq!(seal.commitment, reversed_seal.commitment);
        cleanup(roots);
    }

    #[test]
    fn undeclared_second_config_difference_is_rejected() {
        let (mut inputs, roots) = valid_inputs();
        let sham = inputs
            .iter_mut()
            .find(|arm| arm.role == ArmRole::ShamIntact)
            .unwrap();
        sham.config.learning_threshold += 0.01;
        assert!(matches!(
            seal_primary_gwt_matched_set(inputs),
            Err(MatchedArmError::ConfigCrossBindingMismatch {
                role: ArmRole::ShamIntact
            })
        ));
        cleanup(roots);
    }

    #[test]
    fn broad_arm_with_second_sealed_difference_is_rejected_by_recursive_diff() {
        let a = temp_root("diff_a");
        let ap = temp_root("diff_ap");
        let b = temp_root("diff_b");
        let c = temp_root("diff_c");
        let intact = arm_input(ArmRole::Intact, &a, true, "aesthetic.json", "cpu-1");
        let sham = arm_input(
            ArmRole::ShamIntact,
            &ap,
            true,
            "aesthetic.json",
            "cpu-1",
        );
        let mut broad_config = config(&b, false, "aesthetic.json");
        broad_config.learning_threshold += 0.01;
        let broad_preflight = preflight_primary_campaign_with_environment(
            &broad_config,
            12.5,
            &b,
            EnvironmentSnapshot::from_present_names(&[]),
        )
        .unwrap();
        let broad_snapshot = build_run_environment_snapshot(
            source(),
            runtime("cpu-1"),
            CampaignIdentity {
                campaign_id: "GEOM-003D1-HOLDOUT".to_string(),
                arm_id: ArmRole::BroadGwtOff.arm_id().to_string(),
                analysis_authority_revision: "GEOM-003D0B-v1".to_string(),
                input_schedule_commitment: hex64('2'),
                arm_order_commitment: hex64('3'),
                fixed_utc_hour: 12.5,
            },
            &broad_config,
            &broad_preflight,
            hex64('f'),
        )
        .unwrap();
        let broad = PrimaryArmInput {
            role: ArmRole::BroadGwtOff,
            config: broad_config,
            preflight: broad_preflight,
            run_environment: seal_run_environment(broad_snapshot, None).unwrap(),
            gate_plan: GwtGatePlan::AbsentBecauseGwtDisabled,
        };
        let blocked = arm_input(
            ArmRole::HandlerDeliveryBlocked,
            &c,
            true,
            "aesthetic.json",
            "cpu-1",
        );
        let result = seal_primary_gwt_matched_set(vec![intact, sham, broad, blocked]);
        assert!(matches!(
            result,
            Err(MatchedArmError::UnexpectedConfigDifferences {
                role: ArmRole::BroadGwtOff,
                ..
            })
        ));
        cleanup(vec![a, ap, b, c]);
    }

    #[test]
    fn wrong_gate_plan_is_rejected_before_execution() {
        let (mut inputs, roots) = valid_inputs();
        let blocked = inputs
            .iter_mut()
            .find(|arm| arm.role == ArmRole::HandlerDeliveryBlocked)
            .unwrap();
        blocked.gate_plan = GwtGatePlan::InstalledEnabled;
        assert!(matches!(
            seal_primary_gwt_matched_set(inputs),
            Err(MatchedArmError::WrongGatePlan {
                role: ArmRole::HandlerDeliveryBlocked,
                ..
            })
        ));
        cleanup(roots);
    }

    #[test]
    fn relative_persistence_layout_mismatch_is_rejected() {
        let a = temp_root("layout_a");
        let ap = temp_root("layout_ap");
        let b = temp_root("layout_b");
        let c = temp_root("layout_c");
        let inputs = vec![
            arm_input(ArmRole::Intact, &a, true, "aesthetic.json", "cpu-1"),
            arm_input(
                ArmRole::ShamIntact,
                &ap,
                true,
                "nested/aesthetic.json",
                "cpu-1",
            ),
            arm_input(
                ArmRole::BroadGwtOff,
                &b,
                false,
                "aesthetic.json",
                "cpu-1",
            ),
            arm_input(
                ArmRole::HandlerDeliveryBlocked,
                &c,
                true,
                "aesthetic.json",
                "cpu-1",
            ),
        ];
        assert!(matches!(
            seal_primary_gwt_matched_set(inputs),
            Err(MatchedArmError::AestheticRelativeLayoutMismatch {
                role: ArmRole::ShamIntact
            })
        ));
        cleanup(vec![a, ap, b, c]);
    }

    #[test]
    fn shared_environment_mismatch_is_rejected() {
        let a = temp_root("shared_a");
        let ap = temp_root("shared_ap");
        let b = temp_root("shared_b");
        let c = temp_root("shared_c");
        let inputs = vec![
            arm_input(ArmRole::Intact, &a, true, "aesthetic.json", "cpu-1"),
            arm_input(
                ArmRole::ShamIntact,
                &ap,
                true,
                "aesthetic.json",
                "cpu-2",
            ),
            arm_input(
                ArmRole::BroadGwtOff,
                &b,
                false,
                "aesthetic.json",
                "cpu-1",
            ),
            arm_input(
                ArmRole::HandlerDeliveryBlocked,
                &c,
                true,
                "aesthetic.json",
                "cpu-1",
            ),
        ];
        assert!(matches!(
            seal_primary_gwt_matched_set(inputs),
            Err(MatchedArmError::SharedEnvironmentMismatch {
                role: ArmRole::ShamIntact
            })
        ));
        cleanup(vec![a, ap, b, c]);
    }

    #[test]
    fn duplicate_role_is_rejected() {
        let (mut inputs, roots) = valid_inputs();
        inputs[1].role = ArmRole::Intact;
        assert!(matches!(
            seal_primary_gwt_matched_set(inputs),
            Err(MatchedArmError::DuplicateRole {
                role: ArmRole::Intact
            })
        ));
        cleanup(roots);
    }

    #[test]
    fn duplicate_persistence_root_is_rejected() {
        let a = temp_root("dup_a");
        let b = temp_root("dup_b");
        let c = temp_root("dup_c");
        let inputs = vec![
            arm_input(ArmRole::Intact, &a, true, "aesthetic.json", "cpu-1"),
            arm_input(
                ArmRole::ShamIntact,
                &a,
                true,
                "aesthetic.json",
                "cpu-1",
            ),
            arm_input(
                ArmRole::BroadGwtOff,
                &b,
                false,
                "aesthetic.json",
                "cpu-1",
            ),
            arm_input(
                ArmRole::HandlerDeliveryBlocked,
                &c,
                true,
                "aesthetic.json",
                "cpu-1",
            ),
        ];
        assert!(matches!(
            seal_primary_gwt_matched_set(inputs),
            Err(MatchedArmError::DuplicatePersistenceRoot)
        ));
        cleanup(vec![a, b, c]);
    }

    #[test]
    fn recursive_diff_uses_json_pointer_paths() {
        let left = serde_json::json!({"a": {"b/c": 1}, "x": [1, 2]});
        let right = serde_json::json!({"a": {"b/c": 2}, "x": [1, 3]});
        assert_eq!(
            config_differences(&left, &right),
            vec!["/a/b~1c".to_string(), "/x/1".to_string()]
        );
    }
}
