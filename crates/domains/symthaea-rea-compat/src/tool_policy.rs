//! Fail-closed tool-effect policy for the REA Android read-only profile.
//!
//! This is a bounded mobile catalog slice derived from the pinned upstream
//! REA effect audit. It distinguishes target/device mutation from harmless
//! session Evidence mutation and from local filesystem/process/network effects.
//! Unknown tools are denied; MCP annotations are never used as the authority.

use serde::{Deserialize, Serialize};

/// Upstream REA toolEffects.ts source revision used by this audited slice.
pub const UPSTREAM_REA_EFFECTS_COMMIT: &str =
    "8250bf2320ea57aab4e664db4b4d44b8abd2004e";

/// Explicit effects audited for a public REA tool.
///
/// JSON field names intentionally match upstream REA's camelCase contract.
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
    const fn read_observation() -> Self {
        Self {
            mutates_target: false,
            mutates_session: true,
            writes_filesystem: false,
            launches_process: true,
            accesses_network: true,
            changes_ui_state: false,
            may_discard_data: false,
            executes_foreign_code: false,
            mutates_device: false,
            idempotent: true,
        }
    }

    const fn local_write() -> Self {
        Self {
            writes_filesystem: true,
            idempotent: false,
            ..Self::read_observation()
        }
    }

    const fn target_mutation() -> Self {
        Self {
            mutates_target: true,
            mutates_device: true,
            idempotent: false,
            ..Self::read_observation()
        }
    }
}

/// Target scoping is part of the tool contract, not inferred from its name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetScope {
    /// The operation queries the local ADB client and does not inspect a device.
    None,
    /// The caller must select one exact device before the operation is allowed.
    SelectedDeviceRequired,
    /// Device discovery is permitted only when the caller explicitly requests it.
    ExplicitDiscoveryRequired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuditedTool {
    pub name: &'static str,
    pub effects: ToolEffects,
    pub target_scope: TargetScope,
}

const READ: ToolEffects = ToolEffects::read_observation();
const LOCAL_WRITE: ToolEffects = ToolEffects::local_write();
const MUTATE: ToolEffects = ToolEffects::target_mutation();

/// Positive and negative mobile ADB effect audit, pinned to UPSTREAM_REA_EFFECTS_COMMIT.
///
/// This is intentionally not the complete REA tool catalog. Unlisted tools
/// remain unknown and are denied by the mobile policy.
pub const MOBILE_ADB_TOOL_AUDIT: &[AuditedTool] = &[
    AuditedTool { name: "inspect_adb_client", effects: READ, target_scope: TargetScope::None },
    AuditedTool { name: "list_adb_devices", effects: READ, target_scope: TargetScope::ExplicitDiscoveryRequired },
    AuditedTool { name: "inspect_adb_device", effects: READ, target_scope: TargetScope::SelectedDeviceRequired },
    AuditedTool { name: "list_adb_packages", effects: READ, target_scope: TargetScope::SelectedDeviceRequired },
    AuditedTool { name: "read_adb_logcat", effects: READ, target_scope: TargetScope::SelectedDeviceRequired },
    AuditedTool { name: "inspect_adb_package", effects: READ, target_scope: TargetScope::SelectedDeviceRequired },
    AuditedTool { name: "list_adb_processes", effects: READ, target_scope: TargetScope::SelectedDeviceRequired },
    AuditedTool { name: "list_adb_directory", effects: READ, target_scope: TargetScope::SelectedDeviceRequired },
    AuditedTool { name: "list_adb_features", effects: READ, target_scope: TargetScope::SelectedDeviceRequired },
    AuditedTool { name: "list_adb_services", effects: READ, target_scope: TargetScope::SelectedDeviceRequired },
    AuditedTool { name: "inspect_adb_display", effects: READ, target_scope: TargetScope::SelectedDeviceRequired },
    AuditedTool { name: "inspect_adb_window", effects: READ, target_scope: TargetScope::SelectedDeviceRequired },
    AuditedTool { name: "read_adb_setting", effects: READ, target_scope: TargetScope::SelectedDeviceRequired },
    AuditedTool { name: "resolve_adb_packages", effects: READ, target_scope: TargetScope::SelectedDeviceRequired },
    // Device-read operations that write local artifacts are denied by the strict profile.
    AuditedTool { name: "pull_adb_package", effects: LOCAL_WRITE, target_scope: TargetScope::SelectedDeviceRequired },
    AuditedTool { name: "pull_adb_file", effects: LOCAL_WRITE, target_scope: TargetScope::SelectedDeviceRequired },
    AuditedTool { name: "capture_adb_screen", effects: LOCAL_WRITE, target_scope: TargetScope::SelectedDeviceRequired },
    AuditedTool { name: "collect_adb_bugreport", effects: LOCAL_WRITE, target_scope: TargetScope::SelectedDeviceRequired },
    // Explicit device mutators are represented so policy tests prove they are denied.
    AuditedTool {
        name: "push_adb_file",
        effects: ToolEffects { writes_filesystem: true, ..MUTATE },
        target_scope: TargetScope::SelectedDeviceRequired,
    },
    AuditedTool { name: "install_adb_package", effects: MUTATE, target_scope: TargetScope::SelectedDeviceRequired },
    AuditedTool { name: "uninstall_adb_package", effects: MUTATE, target_scope: TargetScope::SelectedDeviceRequired },
    AuditedTool { name: "start_adb_app", effects: MUTATE, target_scope: TargetScope::SelectedDeviceRequired },
    AuditedTool { name: "start_adb_activity", effects: MUTATE, target_scope: TargetScope::SelectedDeviceRequired },
    AuditedTool { name: "stop_adb_app", effects: MUTATE, target_scope: TargetScope::SelectedDeviceRequired },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MobileToolProfile {
    /// No target writes, local filesystem writes, UI changes, data discard
    /// or foreign-code execution. ADB subprocess and transport use are allowed.
    StrictReadOnlyObservation,
    /// Same target/device write restrictions, but local evidence-file writes are
    /// allowed for separately governed artifact acquisition.
    ReadOnlyDeviceWithLocalArtifacts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvocationScope<'a> {
    pub selected_target: Option<&'a str>,
    pub discovery_explicitly_requested: bool,
}

impl InvocationScope<'_> {
    pub const fn unbound() -> Self {
        Self {
            selected_target: None,
            discovery_explicitly_requested: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenialReason {
    UnknownToolOrMissingEffectAudit,
    ExplicitDiscoveryRequired,
    SelectedTargetRequired,
    MutatesTarget,
    MutatesDevice,
    WritesFilesystem,
    ChangesUiState,
    MayDiscardData,
    ExecutesForeignCode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccessDecision {
    pub allowed: bool,
    pub denial: Option<DenialReason>,
    pub tool: Option<&'static AuditedTool>,
}

impl AccessDecision {
    const fn allow(tool: &'static AuditedTool) -> Self {
        Self { allowed: true, denial: None, tool: Some(tool) }
    }

    const fn deny(reason: DenialReason, tool: Option<&'static AuditedTool>) -> Self {
        Self { allowed: false, denial: Some(reason), tool }
    }
}

pub fn audited_mobile_tool(name: &str) -> Option<&'static AuditedTool> {
    MOBILE_ADB_TOOL_AUDIT.iter().find(|tool| tool.name == name)
}

/// Decide whether one operation may run under a mobile read-only profile.
///
/// Unknown operations are denied before inspecting invocation scope. Device
/// operations require an explicit selected target; discovery must be separately
/// requested. This policy does not authenticate the selected target or verify
/// ADB transport integrity.
pub fn authorize_mobile_tool(
    name: &str,
    profile: MobileToolProfile,
    scope: InvocationScope<'_>,
) -> AccessDecision {
    let Some(tool) = audited_mobile_tool(name) else {
        return AccessDecision::deny(DenialReason::UnknownToolOrMissingEffectAudit, None);
    };

    match tool.target_scope {
        TargetScope::None => {}
        TargetScope::SelectedDeviceRequired => {
            if !scope.selected_target.is_some_and(|target| !target.trim().is_empty()) {
                return AccessDecision::deny(DenialReason::SelectedTargetRequired, Some(tool));
            }
        }
        TargetScope::ExplicitDiscoveryRequired => {
            if !scope.discovery_explicitly_requested {
                return AccessDecision::deny(DenialReason::ExplicitDiscoveryRequired, Some(tool));
            }
        }
    }

    let effects = tool.effects;
    if effects.mutates_target {
        return AccessDecision::deny(DenialReason::MutatesTarget, Some(tool));
    }
    if effects.mutates_device {
        return AccessDecision::deny(DenialReason::MutatesDevice, Some(tool));
    }
    if effects.may_discard_data {
        return AccessDecision::deny(DenialReason::MayDiscardData, Some(tool));
    }
    if effects.executes_foreign_code {
        return AccessDecision::deny(DenialReason::ExecutesForeignCode, Some(tool));
    }
    if effects.changes_ui_state {
        return AccessDecision::deny(DenialReason::ChangesUiState, Some(tool));
    }
    if profile == MobileToolProfile::StrictReadOnlyObservation && effects.writes_filesystem {
        return AccessDecision::deny(DenialReason::WritesFilesystem, Some(tool));
    }

    AccessDecision::allow(tool)
}

/// Return audited effect metadata for bindings and deterministic fixture generation.
pub fn mobile_tool_effects(name: &str) -> Option<ToolEffects> {
    audited_mobile_tool(name).map(|tool| tool.effects)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target() -> InvocationScope<'static> {
        InvocationScope {
            selected_target: Some("adb:serial-placeholder"),
            discovery_explicitly_requested: false,
        }
    }

    #[test]
    fn permits_only_explicit_read_observations_for_a_selected_device() {
        let allowed = [
            "inspect_adb_device", "list_adb_packages", "read_adb_logcat",
            "inspect_adb_package", "list_adb_processes", "list_adb_directory",
            "list_adb_features", "list_adb_services", "inspect_adb_display",
            "inspect_adb_window", "read_adb_setting", "resolve_adb_packages",
        ];
        for name in allowed {
            let result = authorize_mobile_tool(
                name, MobileToolProfile::StrictReadOnlyObservation, target()
            );
            assert!(result.allowed, "{name} should be allowed: {:?}", result.denial);
        }
    }

    #[test]
    fn discovery_and_client_inspection_have_distinct_scope_requirements() {
        assert!(authorize_mobile_tool(
            "inspect_adb_client",
            MobileToolProfile::StrictReadOnlyObservation,
            InvocationScope::unbound(),
        ).allowed);
        assert_eq!(
            authorize_mobile_tool(
                "list_adb_devices",
                MobileToolProfile::StrictReadOnlyObservation,
                InvocationScope::unbound(),
            ).denial,
            Some(DenialReason::ExplicitDiscoveryRequired)
        );
        assert!(authorize_mobile_tool(
            "list_adb_devices",
            MobileToolProfile::StrictReadOnlyObservation,
            InvocationScope { selected_target: None, discovery_explicitly_requested: true },
        ).allowed);
    }

    #[test]
    fn denies_device_observation_without_a_selected_target() {
        assert_eq!(
            authorize_mobile_tool(
                "inspect_adb_device",
                MobileToolProfile::StrictReadOnlyObservation,
                InvocationScope::unbound(),
            ).denial,
            Some(DenialReason::SelectedTargetRequired)
        );
        assert_eq!(
            authorize_mobile_tool(
                "read_adb_logcat",
                MobileToolProfile::StrictReadOnlyObservation,
                InvocationScope { selected_target: Some("  "), discovery_explicitly_requested: false },
            ).denial,
            Some(DenialReason::SelectedTargetRequired)
        );
    }

    #[test]
    fn denies_host_file_writes_in_strict_profile_and_allows_them_only_in_artifact_profile() {
        for name in ["pull_adb_package", "pull_adb_file", "capture_adb_screen", "collect_adb_bugreport"] {
            assert_eq!(
                authorize_mobile_tool(
                    name, MobileToolProfile::StrictReadOnlyObservation, target()
                ).denial,
                Some(DenialReason::WritesFilesystem)
            );
            assert!(authorize_mobile_tool(
                name, MobileToolProfile::ReadOnlyDeviceWithLocalArtifacts, target()
            ).allowed);
        }
    }

    #[test]
    fn denies_every_audited_adb_device_mutator() {
        for name in [
            "push_adb_file", "install_adb_package", "uninstall_adb_package",
            "start_adb_app", "start_adb_activity", "stop_adb_app",
        ] {
            let result = authorize_mobile_tool(
                name, MobileToolProfile::StrictReadOnlyObservation, target()
            );
            assert!(!result.allowed, "{name} was accidentally allowed");
            assert!(matches!(
                result.denial,
                Some(DenialReason::MutatesTarget | DenialReason::MutatesDevice)
            ));
        }
    }

    #[test]
    fn denies_unknown_tools_even_if_the_name_sounds_read_only() {
        for name in ["reboot_adb_device", "shell_adb_command", "unknown_read_only_probe"] {
            assert_eq!(
                authorize_mobile_tool(
                    name, MobileToolProfile::StrictReadOnlyObservation, target()
                ).denial,
                Some(DenialReason::UnknownToolOrMissingEffectAudit)
            );
        }
    }

    #[test]
    fn process_and_transport_effects_are_visible_but_not_mistaken_for_device_mutation() {
        let tool = audited_mobile_tool("inspect_adb_device").unwrap();
        assert!(tool.effects.launches_process);
        assert!(tool.effects.accesses_network);
        assert!(tool.effects.mutates_session);
        assert!(!tool.effects.mutates_target);
        assert!(!tool.effects.mutates_device);
        assert!(authorize_mobile_tool(
            tool.name, MobileToolProfile::StrictReadOnlyObservation, target()
        ).allowed);
    }

    #[test]
    fn effects_serialize_with_exact_upstream_field_names() {
        let json = serde_json::to_value(audited_mobile_tool("push_adb_file").unwrap().effects).unwrap();
        assert_eq!(json["mutatesTarget"], true);
        assert_eq!(json["mutatesDevice"], true);
        assert_eq!(json["writesFilesystem"], true);
        assert_eq!(json["executesForeignCode"], false);
        assert_eq!(json["idempotent"], false);
        assert!(json.get("mutates_target").is_none());
    }

    #[test]
    fn the_audit_records_the_exact_upstream_reference() {
        assert_eq!(UPSTREAM_REA_EFFECTS_COMMIT, "8250bf2320ea57aab4e664db4b4d44b8abd2004e");
        assert!(MOBILE_ADB_TOOL_AUDIT.iter().any(|tool| tool.name == "install_adb_package"));
        assert!(MOBILE_ADB_TOOL_AUDIT.iter().any(|tool| tool.name == "push_adb_file"));
    }
}
