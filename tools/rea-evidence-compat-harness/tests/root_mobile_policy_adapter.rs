use serde_json::Value;

#[path = "../../../src/mobile_tool_policy.rs"]
mod mobile_tool_policy;

#[test]
fn mobile_policy_root_adapter_exports_canonical_contract() {
    let _tool = mobile_tool_policy::MobileTool::InspectAdbDevice;
    assert_eq!(
        mobile_tool_policy::MOBILE_DEVICE_READ_ONLY_POLICY_ID,
        "mobile-device-read-only-v1"
    );
    assert_eq!(mobile_tool_policy::MobileTool::ALL.len(), 30);

    let _authorization: fn(
        &str,
        &str,
        mobile_tool_policy::EffectCeiling,
    ) -> Result<
        mobile_tool_policy::ToolAuthorization,
        mobile_tool_policy::ToolAuthorizationError,
    > = mobile_tool_policy::authorize_mobile_tool;

    let _reported_authorization: fn(
        &str,
        &mobile_tool_policy::MobileProviderCapabilityReport,
        &str,
        mobile_tool_policy::EffectCeiling,
    ) -> Result<
        mobile_tool_policy::ProviderBoundToolAuthorization,
        mobile_tool_policy::ToolAuthorizationError,
    > = mobile_tool_policy::authorize_reported_mobile_tool;

    let _catalog_digest: fn() -> Result<String, mobile_tool_policy::ToolAuthorizationError> =
        mobile_tool_policy::mobile_tool_catalog_sha256;

    let _entry: fn(
        mobile_tool_policy::MobileTool,
    ) -> Option<&'static mobile_tool_policy::MobileToolEffectEntry> =
        mobile_tool_policy::mobile_tool_effect_entry;

    let _ = Value::Null;
}
