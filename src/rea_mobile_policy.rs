//! Direct Rust facade for the fail-closed REA mobile tool-effect policy.

pub use symthaea_rea_compat::tool_policy::{
    authorize_mobile_tool,
    audited_mobile_tool,
    mobile_tool_effects,
    AccessDecision,
    AuditedTool,
    DenialReason,
    InvocationScope,
    MobileToolProfile,
    TargetScope,
    ToolEffects,
    MOBILE_ADB_TOOL_AUDIT,
    UPSTREAM_REA_EFFECTS_COMMIT,
};
