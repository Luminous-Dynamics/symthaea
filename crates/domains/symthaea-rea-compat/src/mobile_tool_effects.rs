//! Typed, fail-closed mobile tool effects and capability evaluation.
//!
//! This is a Rust-native port of the pinned REA mobile tool-effect contract.
//! It distinguishes local process/network/filesystem effects from changes to
//! the attached device or execution of foreign code. MCP annotations are
//! derived metadata; they are never the authorization authority.

use crate::{canonical_jcs_bytes, sha256_hex};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fmt;

pub const UPSTREAM_REA_COMMIT: &str = "8250bf2320ea57aab4e664db4b4d44b8abd2004e";
pub const MOBILE_DEVICE_READ_ONLY_POLICY_ID: &str = "mobile-device-read-only-v1";

/// Closed set of mobile-facing public REA operations audited by this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MobileTool {
    InspectAndroidJniBindings,
    TraceAndroidReferences,
    InspectAndroidPackage,
    SearchAndroidClasses,
    InspectAndroidClass,
    InspectAndroidMethod,
    InspectAdbClient,
    ListAdbDevices,
    InspectAdbDevice,
    ListAdbPackages,
    PullAdbPackage,
    ReadAdbLogcat,
    InspectAdbPackage,
    PullAdbFile,
    PushAdbFile,
    CaptureAdbScreen,
    ListAdbProcesses,
    ListAdbDirectory,
    ListAdbFeatures,
    ListAdbServices,
    InspectAdbDisplay,
    InspectAdbWindow,
    ReadAdbSetting,
    CollectAdbBugreport,
    ResolveAdbPackages,
    InstallAdbPackage,
    UninstallAdbPackage,
    StartAdbApp,
    StartAdbActivity,
    StopAdbApp,
}

impl MobileTool {
    pub const ALL: [Self; 30] = [
        Self::InspectAndroidJniBindings,
        Self::TraceAndroidReferences,
        Self::InspectAndroidPackage,
        Self::SearchAndroidClasses,
        Self::InspectAndroidClass,
        Self::InspectAndroidMethod,
        Self::InspectAdbClient,
        Self::ListAdbDevices,
        Self::InspectAdbDevice,
        Self::ListAdbPackages,
        Self::PullAdbPackage,
        Self::ReadAdbLogcat,
        Self::InspectAdbPackage,
        Self::PullAdbFile,
        Self::PushAdbFile,
        Self::CaptureAdbScreen,
        Self::ListAdbProcesses,
        Self::ListAdbDirectory,
        Self::ListAdbFeatures,
        Self::ListAdbServices,
        Self::InspectAdbDisplay,
        Self::InspectAdbWindow,
        Self::ReadAdbSetting,
        Self::CollectAdbBugreport,
        Self::ResolveAdbPackages,
        Self::InstallAdbPackage,
        Self::UninstallAdbPackage,
        Self::StartAdbApp,
        Self::StartAdbActivity,
        Self::StopAdbApp,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InspectAndroidJniBindings => "inspect_android_jni_bindings",
            Self::TraceAndroidReferences => "trace_android_references",
            Self::InspectAndroidPackage => "inspect_android_package",
            Self::SearchAndroidClasses => "search_android_classes",
            Self::InspectAndroidClass => "inspect_android_class",
            Self::InspectAndroidMethod => "inspect_android_method",
            Self::InspectAdbClient => "inspect_adb_client",
            Self::ListAdbDevices => "list_adb_devices",
            Self::InspectAdbDevice => "inspect_adb_device",
            Self::ListAdbPackages => "list_adb_packages",
            Self::PullAdbPackage => "pull_adb_package",
            Self::ReadAdbLogcat => "read_adb_logcat",
            Self::InspectAdbPackage => "inspect_adb_package",
            Self::PullAdbFile => "pull_adb_file",
            Self::PushAdbFile => "push_adb_file",
            Self::CaptureAdbScreen => "capture_adb_screen",
            Self::ListAdbProcesses => "list_adb_processes",
            Self::ListAdbDirectory => "list_adb_directory",
            Self::ListAdbFeatures => "list_adb_features",
            Self::ListAdbServices => "list_adb_services",
            Self::InspectAdbDisplay => "inspect_adb_display",
            Self::InspectAdbWindow => "inspect_adb_window",
            Self::ReadAdbSetting => "read_adb_setting",
            Self::CollectAdbBugreport => "collect_adb_bugreport",
            Self::ResolveAdbPackages => "resolve_adb_packages",
            Self::InstallAdbPackage => "install_adb_package",
            Self::UninstallAdbPackage => "uninstall_adb_package",
            Self::StartAdbApp => "start_adb_app",
            Self::StartAdbActivity => "start_adb_activity",
            Self::StopAdbApp => "stop_adb_app",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|tool| tool.as_str() == name)
    }
}

/// Canonical upstream ToolEffects shape. Every field is explicit; missing
/// catalog entries never receive an implicit all-false default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ToolEffects {
    pub mutates_target: bool,
    pub mutates_session: bool,
    pub writes_filesystem: bool,
    pub launches_process: bool,
    pub accesses_network: bool,
    pub changes_ui_state: bool,
    pub may_discard_data: bool,
    pub executes_foreign_code: bool,
    pub mutates_device: bool,
    pub idempotent: bool,
}

impl ToolEffects {
    pub const NONE: Self = Self {
        mutates_target: false,
        mutates_session: false,
        writes_filesystem: false,
        launches_process: false,
        accesses_network: false,
        changes_ui_state: false,
        may_discard_data: false,
        executes_foreign_code: false,
        mutates_device: false,
        idempotent: true,
    };

    pub const fn is_read_only_hint(self) -> bool {
        !(self.mutates_target
            || self.mutates_session
            || self.writes_filesystem
            || self.launches_process
            || self.accesses_network
            || self.changes_ui_state
            || self.may_discard_data
            || self.executes_foreign_code
            || self.mutates_device)
    }

    pub const fn is_destructive_hint(self) -> bool {
        self.may_discard_data || self.executes_foreign_code || self.mutates_device
    }

    pub const fn is_open_world_hint(self) -> bool {
        self.launches_process || self.accesses_network
    }
}

/// Equivalent of upstream's derived MCP hints. These values are presentation
/// and routing metadata only; policy is evaluated against ToolEffects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DerivedToolAnnotations {
    pub read_only_hint: bool,
    pub destructive_hint: bool,
    pub idempotent_hint: bool,
    pub open_world_hint: bool,
}

pub const fn annotations_from_effects(effects: ToolEffects) -> DerivedToolAnnotations {
    DerivedToolAnnotations {
        read_only_hint: effects.is_read_only_hint(),
        destructive_hint: effects.is_destructive_hint(),
        idempotent_hint: effects.idempotent,
        open_world_hint: effects.is_open_world_hint(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MobileToolEffectEntry {
    pub tool: MobileTool,
    pub effects: ToolEffects,
    /// All selected mobile tools can return provider/device/target-originated
    /// text on success or failure; callers must mark those results untrusted.
    pub returns_target_content: bool,
}

const fn observed_with_process_and_session() -> ToolEffects {
    ToolEffects {
        mutates_session: true,
        launches_process: true,
        ..ToolEffects::NONE
    }
}

const fn observed_with_adb_transport() -> ToolEffects {
    ToolEffects {
        mutates_session: true,
        launches_process: true,
        accesses_network: true,
        ..ToolEffects::NONE
    }
}

const fn static_android_analysis() -> ToolEffects {
    ToolEffects {
        mutates_session: true,
        launches_process: true,
        writes_filesystem: true,
        ..ToolEffects::NONE
    }
}

const fn adb_host_capture() -> ToolEffects {
    ToolEffects {
        mutates_session: true,
        launches_process: true,
        writes_filesystem: true,
        accesses_network: true,
        idempotent: false,
        ..ToolEffects::NONE
    }
}

const fn adb_device_mutation() -> ToolEffects {
    ToolEffects {
        mutates_target: true,
        mutates_session: true,
        launches_process: true,
        accesses_network: true,
        mutates_device: true,
        idempotent: false,
        ..ToolEffects::NONE
    }
}

/// Pinned mobile subset from REA's TOOL_EFFECTS, not inferred from names or
/// MCP descriptions. Review the source commit when this table changes.
pub const MOBILE_TOOL_EFFECTS: [MobileToolEffectEntry; 30] = [
    MobileToolEffectEntry {
        tool: MobileTool::InspectAndroidJniBindings,
        effects: static_android_analysis(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::TraceAndroidReferences,
        effects: static_android_analysis(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::InspectAndroidPackage,
        effects: static_android_analysis(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::SearchAndroidClasses,
        effects: static_android_analysis(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::InspectAndroidClass,
        effects: static_android_analysis(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::InspectAndroidMethod,
        effects: static_android_analysis(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::InspectAdbClient,
        effects: observed_with_process_and_session(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::ListAdbDevices,
        effects: observed_with_adb_transport(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::InspectAdbDevice,
        effects: observed_with_adb_transport(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::ListAdbPackages,
        effects: observed_with_adb_transport(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::PullAdbPackage,
        effects: adb_host_capture(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::ReadAdbLogcat,
        effects: observed_with_adb_transport(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::InspectAdbPackage,
        effects: observed_with_adb_transport(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::PullAdbFile,
        effects: adb_host_capture(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::PushAdbFile,
        effects: ToolEffects {
            writes_filesystem: true,
            ..adb_device_mutation()
        },
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::CaptureAdbScreen,
        effects: adb_host_capture(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::ListAdbProcesses,
        effects: observed_with_adb_transport(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::ListAdbDirectory,
        effects: observed_with_adb_transport(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::ListAdbFeatures,
        effects: observed_with_adb_transport(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::ListAdbServices,
        effects: observed_with_adb_transport(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::InspectAdbDisplay,
        effects: observed_with_adb_transport(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::InspectAdbWindow,
        effects: observed_with_adb_transport(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::ReadAdbSetting,
        effects: observed_with_adb_transport(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::CollectAdbBugreport,
        effects: adb_host_capture(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::ResolveAdbPackages,
        effects: observed_with_adb_transport(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::InstallAdbPackage,
        effects: adb_device_mutation(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::UninstallAdbPackage,
        effects: adb_device_mutation(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::StartAdbApp,
        effects: adb_device_mutation(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::StartAdbActivity,
        effects: adb_device_mutation(),
        returns_target_content: true,
    },
    MobileToolEffectEntry {
        tool: MobileTool::StopAdbApp,
        effects: adb_device_mutation(),
        returns_target_content: true,
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectCeiling {
    pub allow_mutates_target: bool,
    pub allow_mutates_session: bool,
    pub allow_writes_filesystem: bool,
    pub allow_launches_process: bool,
    pub allow_accesses_network: bool,
    pub allow_changes_ui_state: bool,
    pub allow_may_discard_data: bool,
    pub allow_executes_foreign_code: bool,
    pub allow_mutates_device: bool,
}

impl EffectCeiling {
    /// Explicit observation lane: it may invoke the chosen host adb/JADX
    /// providers, use the selected connection and append REA evidence/session
    /// state. Host filesystem writes remain denied until a separate caller-bound
    /// output-sink capability is implemented; device mutation, UI driving, data
    /// discard and foreign-code execution are also denied.
    pub const MOBILE_DEVICE_READ_ONLY: Self = Self {
        allow_mutates_target: false,
        allow_mutates_session: true,
        allow_writes_filesystem: false,
        allow_launches_process: true,
        allow_accesses_network: true,
        allow_changes_ui_state: false,
        allow_may_discard_data: false,
        allow_executes_foreign_code: false,
        allow_mutates_device: false,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectFlag {
    MutatesTarget,
    MutatesSession,
    WritesFilesystem,
    LaunchesProcess,
    AccessesNetwork,
    ChangesUiState,
    MayDiscardData,
    ExecutesForeignCode,
    MutatesDevice,
}

impl EffectFlag {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MutatesTarget => "mutates_target",
            Self::MutatesSession => "mutates_session",
            Self::WritesFilesystem => "writes_filesystem",
            Self::LaunchesProcess => "launches_process",
            Self::AccessesNetwork => "accesses_network",
            Self::ChangesUiState => "changes_ui_state",
            Self::MayDiscardData => "may_discard_data",
            Self::ExecutesForeignCode => "executes_foreign_code",
            Self::MutatesDevice => "mutates_device",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolAuthorizationError {
    UnknownTool(String),
    MissingCatalogEntry(MobileTool),
    EffectExceedsCeiling {
        tool: MobileTool,
        effect: EffectFlag,
    },
    Canonicalization(String),
    InvalidProviderCapability(String),
    ProviderOperationUnavailable {
        tool: MobileTool,
        reason: String,
    },
}

impl fmt::Display for ToolAuthorizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTool(name) => write!(f, "unknown mobile tool '{name}'; denied by default"),
            Self::MissingCatalogEntry(tool) => write!(
                f,
                "mobile tool {} has no effect entry; denied by default",
                tool.as_str()
            ),
            Self::EffectExceedsCeiling { tool, effect } => write!(
                f,
                "mobile tool {} exceeds policy ceiling for {}; denied",
                tool.as_str(),
                effect.as_str()
            ),
            Self::Canonicalization(message) => write!(
                f,
                "cannot bind tool authorization to effect/policy digest: {message}"
            ),
            Self::InvalidProviderCapability(message) => {
                write!(f, "provider capability report is invalid: {message}")
            }
            Self::ProviderOperationUnavailable { tool, reason } => write!(
                f,
                "provider reports {} unavailable: {reason}",
                tool.as_str()
            ),
        }
    }
}

impl std::error::Error for ToolAuthorizationError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolAuthorization {
    pub tool: MobileTool,
    pub policy_id: String,
    pub policy_sha256: String,
    pub effect_sha256: String,
    pub effects: ToolEffects,
    pub annotations: DerivedToolAnnotations,
    pub returns_target_content: bool,
}

/// Availability is a property of the provider/operation at a particular
/// provider revision. It is deliberately separate from the policy decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderAvailability {
    Available,
    PartiallyAvailable,
    Unavailable,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderOperationCapability {
    pub tool: MobileTool,
    pub available: bool,
    pub reason: Option<String>,
    pub effects: ToolEffects,
    pub returns_target_content: bool,
    /// Order is preserved exactly; limitations are not silently sorted.
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MobileProviderCapabilityReport {
    pub provider_id: String,
    pub provider_version: Option<String>,
    pub provider_revision: String,
    /// Optional digest of the selected provider executable. This is supplied
    /// metadata until an independent acquisition receipt authenticates it.
    pub provider_executable_sha256: Option<String>,
    pub upstream_rea_commit: String,
    pub effect_catalog_sha256: String,
    pub availability: ProviderAvailability,
    pub reason: Option<String>,
    pub operations: Vec<ProviderOperationCapability>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderBoundToolAuthorization {
    pub provider_id: String,
    pub provider_version: Option<String>,
    pub provider_revision: String,
    pub provider_executable_sha256: Option<String>,
    pub capability_report_sha256: String,
    pub authorization: ToolAuthorization,
}

/// Validate provider availability and prove that its advertised mobile
/// operation effects exactly match the pinned Rust effect catalog.
///
/// This is not producer authentication: the report and optional executable
/// digest are self-reported until checked by an independent acquisition lane.
impl MobileProviderCapabilityReport {
    pub fn validate(&self) -> Result<(), ToolAuthorizationError> {
        if self.provider_id.trim().is_empty() {
            return Err(ToolAuthorizationError::InvalidProviderCapability(
                "provider_id must not be empty".to_owned(),
            ));
        }
        if self.provider_revision.trim().is_empty() {
            return Err(ToolAuthorizationError::InvalidProviderCapability(
                "provider_revision must not be empty".to_owned(),
            ));
        }
        if self.upstream_rea_commit != UPSTREAM_REA_COMMIT {
            return Err(ToolAuthorizationError::InvalidProviderCapability(
                "upstream REA commit does not match the pinned effect catalog".to_owned(),
            ));
        }
        if self.effect_catalog_sha256 != mobile_tool_catalog_sha256()? {
            return Err(ToolAuthorizationError::InvalidProviderCapability(
                "effect catalog digest does not match the pinned Rust table".to_owned(),
            ));
        }
        if let Some(digest) = &self.provider_executable_sha256 {
            validate_lowercase_sha256(digest)
                .map_err(|message| ToolAuthorizationError::InvalidProviderCapability(message))?;
        }
        if let Some(reason) = &self.reason {
            if reason.trim().is_empty() {
                return Err(ToolAuthorizationError::InvalidProviderCapability(
                    "provider availability reason must not be empty when present".to_owned(),
                ));
            }
        }

        let mut seen = std::collections::HashSet::new();
        let mut available_count = 0usize;
        let mut unavailable_count = 0usize;
        for operation in &self.operations {
            if !seen.insert(operation.tool) {
                return Err(ToolAuthorizationError::InvalidProviderCapability(format!(
                    "duplicate operation {}",
                    operation.tool.as_str()
                )));
            }
            let expected = mobile_tool_effect_entry(operation.tool)
                .ok_or(ToolAuthorizationError::MissingCatalogEntry(operation.tool))?;
            if operation.effects != expected.effects
                || operation.returns_target_content != expected.returns_target_content
            {
                return Err(ToolAuthorizationError::InvalidProviderCapability(format!(
                    "provider effect/target-content declaration differs from the canonical row for {}",
                    operation.tool.as_str()
                )));
            }
            if operation.available {
                available_count += 1;
            } else {
                unavailable_count += 1;
                if operation
                    .reason
                    .as_deref()
                    .is_none_or(|reason| reason.trim().is_empty())
                {
                    return Err(ToolAuthorizationError::InvalidProviderCapability(format!(
                        "unavailable operation {} requires a reason",
                        operation.tool.as_str()
                    )));
                }
            }
        }

        match self.availability {
            ProviderAvailability::Available => {
                if self.operations.is_empty() || unavailable_count != 0 {
                    return Err(ToolAuthorizationError::InvalidProviderCapability(
                        "available provider must advertise at least one available operation and no unavailable operation".to_owned(),
                    ));
                }
            }
            ProviderAvailability::PartiallyAvailable => {
                if available_count == 0 || unavailable_count == 0 {
                    return Err(ToolAuthorizationError::InvalidProviderCapability(
                        "partially available provider must contain both available and unavailable operations".to_owned(),
                    ));
                }
            }
            ProviderAvailability::Unavailable => {
                if available_count != 0
                    || self
                        .reason
                        .as_deref()
                        .is_none_or(|reason| reason.trim().is_empty())
                {
                    return Err(ToolAuthorizationError::InvalidProviderCapability(
                        "unavailable provider requires a reason and cannot advertise an available operation".to_owned(),
                    ));
                }
            }
            ProviderAvailability::Unknown => {
                if self
                    .reason
                    .as_deref()
                    .is_none_or(|reason| reason.trim().is_empty())
                {
                    return Err(ToolAuthorizationError::InvalidProviderCapability(
                        "unknown provider availability requires an explicit reason".to_owned(),
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn canonical_sha256(&self) -> Result<String, ToolAuthorizationError> {
        let value = serde_json::to_value(self)
            .map_err(|error| ToolAuthorizationError::Canonicalization(error.to_string()))?;
        let bytes = canonical_jcs_bytes(&value)
            .map_err(|error| ToolAuthorizationError::Canonicalization(error.to_string()))?;
        Ok(sha256_hex(&bytes))
    }
}

/// Authorize only when the exact provider revision reports the selected
/// operation available, the effect row matches the pinned catalog, and the
/// independent policy ceiling allows all declared effects.
pub fn authorize_reported_mobile_tool(
    name: &str,
    report: &MobileProviderCapabilityReport,
    policy_id: &str,
    ceiling: EffectCeiling,
) -> Result<ProviderBoundToolAuthorization, ToolAuthorizationError> {
    report.validate()?;
    let tool = MobileTool::parse(name)
        .ok_or_else(|| ToolAuthorizationError::UnknownTool(name.to_owned()))?;
    if report.availability == ProviderAvailability::Unknown {
        return Err(ToolAuthorizationError::ProviderOperationUnavailable {
            tool,
            reason: report.reason.clone().unwrap_or_else(|| {
                "provider availability is unknown; denied by default".to_owned()
            }),
        });
    }
    let operation = report
        .operations
        .iter()
        .find(|operation| operation.tool == tool)
        .ok_or_else(|| {
            ToolAuthorizationError::InvalidProviderCapability(format!(
                "provider did not advertise operation {}",
                tool.as_str()
            ))
        })?;
    if !operation.available {
        return Err(ToolAuthorizationError::ProviderOperationUnavailable {
            tool,
            reason: operation
                .reason
                .clone()
                .unwrap_or_else(|| "not available".to_owned()),
        });
    }

    let authorization = authorize_mobile_tool(name, policy_id, ceiling)?;
    Ok(ProviderBoundToolAuthorization {
        provider_id: report.provider_id.clone(),
        provider_version: report.provider_version.clone(),
        provider_revision: report.provider_revision.clone(),
        provider_executable_sha256: report.provider_executable_sha256.clone(),
        capability_report_sha256: report.canonical_sha256()?,
        authorization,
    })
}

fn validate_lowercase_sha256(value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(
            "provider executable digest must be 64 lowercase hexadecimal characters".to_owned(),
        );
    }
    Ok(())
}

/// Stable metadata summary used by the CLI, MCP adapter and test harness.
pub fn mobile_tool_effect_entry(tool: MobileTool) -> Option<&'static MobileToolEffectEntry> {
    MOBILE_TOOL_EFFECTS.iter().find(|entry| entry.tool == tool)
}

/// Unknown names and catalog omissions are both denied. This function is a
/// policy evaluator; adapters must invoke it at the execution boundary.
pub fn authorize_mobile_tool(
    name: &str,
    policy_id: &str,
    ceiling: EffectCeiling,
) -> Result<ToolAuthorization, ToolAuthorizationError> {
    let tool = MobileTool::parse(name)
        .ok_or_else(|| ToolAuthorizationError::UnknownTool(name.to_owned()))?;
    let entry =
        mobile_tool_effect_entry(tool).ok_or(ToolAuthorizationError::MissingCatalogEntry(tool))?;
    let effects = entry.effects;

    for (active, allowed, flag) in [
        (
            effects.mutates_target,
            ceiling.allow_mutates_target,
            EffectFlag::MutatesTarget,
        ),
        (
            effects.mutates_session,
            ceiling.allow_mutates_session,
            EffectFlag::MutatesSession,
        ),
        (
            effects.writes_filesystem,
            ceiling.allow_writes_filesystem,
            EffectFlag::WritesFilesystem,
        ),
        (
            effects.launches_process,
            ceiling.allow_launches_process,
            EffectFlag::LaunchesProcess,
        ),
        (
            effects.accesses_network,
            ceiling.allow_accesses_network,
            EffectFlag::AccessesNetwork,
        ),
        (
            effects.changes_ui_state,
            ceiling.allow_changes_ui_state,
            EffectFlag::ChangesUiState,
        ),
        (
            effects.may_discard_data,
            ceiling.allow_may_discard_data,
            EffectFlag::MayDiscardData,
        ),
        (
            effects.executes_foreign_code,
            ceiling.allow_executes_foreign_code,
            EffectFlag::ExecutesForeignCode,
        ),
        (
            effects.mutates_device,
            ceiling.allow_mutates_device,
            EffectFlag::MutatesDevice,
        ),
    ] {
        if active && !allowed {
            return Err(ToolAuthorizationError::EffectExceedsCeiling { tool, effect: flag });
        }
    }

    let effect_value = serde_json::to_value(effects)
        .map_err(|error| ToolAuthorizationError::Canonicalization(error.to_string()))?;
    let ceiling_value = serde_json::to_value(ceiling)
        .map_err(|error| ToolAuthorizationError::Canonicalization(error.to_string()))?;
    let policy_value = json!({
        "policy_id": policy_id,
        "ceiling": ceiling_value,
        "upstream_rea_commit": UPSTREAM_REA_COMMIT
    });
    let effect_bytes = canonical_jcs_bytes(&effect_value)
        .map_err(|error| ToolAuthorizationError::Canonicalization(error.to_string()))?;
    let policy_bytes = canonical_jcs_bytes(&policy_value)
        .map_err(|error| ToolAuthorizationError::Canonicalization(error.to_string()))?;

    Ok(ToolAuthorization {
        tool,
        policy_id: policy_id.to_owned(),
        policy_sha256: sha256_hex(&policy_bytes),
        effect_sha256: sha256_hex(&effect_bytes),
        effects,
        annotations: annotations_from_effects(effects),
        returns_target_content: entry.returns_target_content,
    })
}

/// Evaluate a tool under the explicit read-only attached-device observation
/// profile. This grants local collection/transport behavior but not device
/// mutation, UI control or foreign-code execution.
pub fn authorize_mobile_device_observation(
    name: &str,
) -> Result<ToolAuthorization, ToolAuthorizationError> {
    authorize_mobile_tool(
        name,
        MOBILE_DEVICE_READ_ONLY_POLICY_ID,
        EffectCeiling::MOBILE_DEVICE_READ_ONLY,
    )
}

/// Stable JCS digest of the audited mobile catalog. Updating upstream REA or
/// this Rust table changes the digest and requires a reviewed contract update.
pub fn mobile_tool_catalog_sha256() -> Result<String, ToolAuthorizationError> {
    let value = serde_json::to_value(MOBILE_TOOL_EFFECTS)
        .map_err(|error| ToolAuthorizationError::Canonicalization(error.to_string()))?;
    let bytes = canonical_jcs_bytes(&value)
        .map_err(|error| ToolAuthorizationError::Canonicalization(error.to_string()))?;
    Ok(sha256_hex(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_covers_every_declared_mobile_tool_exactly_once() {
        assert_eq!(MOBILE_TOOL_EFFECTS.len(), MobileTool::ALL.len());
        for (index, tool) in MobileTool::ALL.iter().enumerate() {
            assert_eq!(
                MOBILE_TOOL_EFFECTS
                    .iter()
                    .filter(|entry| entry.tool == *tool)
                    .count(),
                1,
                "mobile catalog index {index} must contain {} once",
                tool.as_str()
            );
        }
    }

    #[test]
    fn read_only_observation_allows_transport_but_denies_device_mutation() {
        for tool in [
            "list_adb_devices",
            "inspect_adb_device",
            "list_adb_packages",
            "read_adb_logcat",
        ] {
            let grant = authorize_mobile_device_observation(tool).unwrap();
            assert_eq!(grant.tool.as_str(), tool);
            assert!(!grant.effects.mutates_device);
            assert!(!grant.effects.executes_foreign_code);
        }

        for tool in [
            "push_adb_file",
            "install_adb_package",
            "uninstall_adb_package",
            "start_adb_app",
            "start_adb_activity",
            "stop_adb_app",
        ] {
            let error = authorize_mobile_device_observation(tool).unwrap_err();
            assert!(matches!(
                error,
                ToolAuthorizationError::EffectExceedsCeiling {
                    effect: EffectFlag::MutatesTarget | EffectFlag::MutatesDevice,
                    ..
                }
            ));
        }
    }

    #[test]
    fn local_artifact_writes_are_denied_until_a_caller_bound_sink_exists() {
        for tool in [
            "inspect_android_jni_bindings",
            "trace_android_references",
            "inspect_android_package",
            "search_android_classes",
            "inspect_android_class",
            "inspect_android_method",
            "pull_adb_package",
            "pull_adb_file",
            "capture_adb_screen",
            "collect_adb_bugreport",
        ] {
            assert!(
                matches!(
                    authorize_mobile_device_observation(tool),
                    Err(ToolAuthorizationError::EffectExceedsCeiling {
                        effect: EffectFlag::WritesFilesystem,
                        ..
                    })
                ),
                "{tool} must not write outside a bound output sink"
            );
        }
    }

    #[test]
    fn effect_json_uses_upstream_camel_case_field_names() {
        let entry = mobile_tool_effect_entry(MobileTool::InspectAdbDevice).unwrap();
        let value = serde_json::to_value(entry.effects).unwrap();
        assert_eq!(value["mutatesSession"], true);
        assert_eq!(value["launchesProcess"], true);
        assert_eq!(value["accessesNetwork"], true);
        assert_eq!(value["mutates_session"], Value::Null);
    }

    #[test]
    fn unknown_or_unaudited_tool_names_fail_closed() {
        assert!(matches!(
            authorize_mobile_device_observation("shell"),
            Err(ToolAuthorizationError::UnknownTool(_))
        ));
        assert!(matches!(
            authorize_mobile_device_observation("push_adb_file "),
            Err(ToolAuthorizationError::UnknownTool(_))
        ));
    }

    #[test]
    fn annotations_are_derived_from_effects_not_used_as_authority() {
        let entry = mobile_tool_effect_entry(MobileTool::PushAdbFile).unwrap();
        let annotations = annotations_from_effects(entry.effects);
        assert!(!annotations.read_only_hint);
        assert!(annotations.destructive_hint);
        assert!(!annotations.idempotent_hint);
        assert!(annotations.open_world_hint);

        let read = mobile_tool_effect_entry(MobileTool::ListAdbDevices).unwrap();
        let annotations = annotations_from_effects(read.effects);
        assert!(!annotations.read_only_hint);
        assert!(!annotations.destructive_hint);
        assert!(annotations.idempotent_hint);
        assert!(annotations.open_world_hint);
    }

    #[test]
    fn effect_and_policy_receipts_bind_exact_catalog_semantics() {
        let before = mobile_tool_catalog_sha256().unwrap();
        let read = authorize_mobile_device_observation("inspect_adb_device").unwrap();
        let second = authorize_mobile_device_observation("inspect_adb_device").unwrap();
        assert_eq!(before, mobile_tool_catalog_sha256().unwrap());
        assert_eq!(read, second);
        assert_eq!(read.policy_id, MOBILE_DEVICE_READ_ONLY_POLICY_ID);
        assert!(!read.effect_sha256.is_empty());
        assert!(!read.policy_sha256.is_empty());
    }

    fn report(
        available_tools: &[MobileTool],
        unavailable_tools: &[MobileTool],
    ) -> MobileProviderCapabilityReport {
        let operations = available_tools
            .iter()
            .map(|tool| {
                let entry = mobile_tool_effect_entry(*tool).unwrap();
                ProviderOperationCapability {
                    tool: *tool,
                    available: true,
                    reason: None,
                    effects: entry.effects,
                    returns_target_content: entry.returns_target_content,
                    limitations: vec!["bounded provider inspection".to_owned()],
                }
            })
            .chain(unavailable_tools.iter().map(|tool| {
                let entry = mobile_tool_effect_entry(*tool).unwrap();
                ProviderOperationCapability {
                    tool: *tool,
                    available: false,
                    reason: Some("required platform capability is absent".to_owned()),
                    effects: entry.effects,
                    returns_target_content: entry.returns_target_content,
                    limitations: vec!["operation unavailable for this provider build".to_owned()],
                }
            }))
            .collect();

        MobileProviderCapabilityReport {
            provider_id: "rea.adb".to_owned(),
            provider_version: Some("1.0.0".to_owned()),
            provider_revision: "adb-adapter-2026-11-02".to_owned(),
            provider_executable_sha256: Some("a".repeat(64)),
            upstream_rea_commit: UPSTREAM_REA_COMMIT.to_owned(),
            effect_catalog_sha256: mobile_tool_catalog_sha256().unwrap(),
            availability: if available_tools.is_empty() {
                ProviderAvailability::Unavailable
            } else if unavailable_tools.is_empty() {
                ProviderAvailability::Available
            } else {
                ProviderAvailability::PartiallyAvailable
            },
            reason: if available_tools.is_empty() {
                Some("adb provider not attached".to_owned())
            } else {
                None
            },
            operations,
            limitations: vec![
                "USB and network device transports are caller-managed".to_owned(),
                "hardware-backed claims require separate evidence".to_owned(),
            ],
        }
    }

    #[test]
    fn provider_availability_is_separate_from_policy_authority() {
        let capability = report(&[MobileTool::InspectAdbDevice], &[MobileTool::PushAdbFile]);
        let grant = authorize_reported_mobile_tool(
            "inspect_adb_device",
            &capability,
            MOBILE_DEVICE_READ_ONLY_POLICY_ID,
            EffectCeiling::MOBILE_DEVICE_READ_ONLY,
        )
        .unwrap();
        assert_eq!(grant.provider_id, "rea.adb");
        assert_eq!(grant.authorization.tool, MobileTool::InspectAdbDevice);
        assert!(!grant.capability_report_sha256.is_empty());

        let denied = authorize_reported_mobile_tool(
            "push_adb_file",
            &capability,
            MOBILE_DEVICE_READ_ONLY_POLICY_ID,
            EffectCeiling::MOBILE_DEVICE_READ_ONLY,
        )
        .unwrap_err();
        assert!(matches!(
            denied,
            ToolAuthorizationError::ProviderOperationUnavailable { .. }
        ));

        // Even an available tool remains subject to the independent effect ceiling.
        let mut permissive_capability = report(&[MobileTool::PushAdbFile], &[]);
        permissive_capability.availability = ProviderAvailability::Available;
        let denied = authorize_reported_mobile_tool(
            "push_adb_file",
            &permissive_capability,
            MOBILE_DEVICE_READ_ONLY_POLICY_ID,
            EffectCeiling::MOBILE_DEVICE_READ_ONLY,
        )
        .unwrap_err();
        assert!(matches!(
            denied,
            ToolAuthorizationError::EffectExceedsCeiling { .. }
        ));
    }

    #[test]
    fn unknown_provider_availability_never_grants_a_tool() {
        let mut capability = report(&[MobileTool::InspectAdbDevice], &[]);
        capability.availability = ProviderAvailability::Unknown;
        capability.reason = Some("capability discovery did not complete".to_owned());
        let error = authorize_reported_mobile_tool(
            "inspect_adb_device",
            &capability,
            MOBILE_DEVICE_READ_ONLY_POLICY_ID,
            EffectCeiling::MOBILE_DEVICE_READ_ONLY,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ToolAuthorizationError::ProviderOperationUnavailable { .. }
        ));
    }

    #[test]
    fn provider_report_rejects_effect_downgrades_and_wrong_catalog_digest() {
        let mut capability = report(&[MobileTool::InspectAdbDevice], &[]);
        capability.operations[0].effects.mutates_device = true;
        assert!(matches!(
            capability.validate(),
            Err(ToolAuthorizationError::InvalidProviderCapability(_))
        ));

        let mut capability = report(&[MobileTool::InspectAdbDevice], &[]);
        capability.effect_catalog_sha256 = "0".repeat(64);
        assert!(matches!(
            capability.validate(),
            Err(ToolAuthorizationError::InvalidProviderCapability(_))
        ));
    }

    #[test]
    fn provider_report_requires_unavailable_reasons_and_preserves_limitations_order() {
        let mut capability = report(&[MobileTool::InspectAdbDevice], &[MobileTool::PushAdbFile]);
        capability.operations[1].reason = None;
        assert!(matches!(
            capability.validate(),
            Err(ToolAuthorizationError::InvalidProviderCapability(_))
        ));

        let capability = report(&[MobileTool::InspectAdbDevice], &[MobileTool::PushAdbFile]);
        assert_eq!(
            capability.limitations[0],
            "USB and network device transports are caller-managed"
        );
        assert_eq!(
            capability.limitations[1],
            "hardware-backed claims require separate evidence"
        );
    }

    #[test]
    fn stricter_host_effect_policy_can_reject_local_process_and_filesystem_effects() {
        let strict = EffectCeiling {
            allow_mutates_target: false,
            allow_mutates_session: true,
            allow_writes_filesystem: false,
            allow_launches_process: false,
            allow_accesses_network: false,
            allow_changes_ui_state: false,
            allow_may_discard_data: false,
            allow_executes_foreign_code: false,
            allow_mutates_device: false,
        };
        assert!(matches!(
            authorize_mobile_tool("inspect_adb_device", "offline-no-process-v1", strict),
            Err(ToolAuthorizationError::EffectExceedsCeiling {
                effect: EffectFlag::LaunchesProcess,
                ..
            })
        ));
    }
}
