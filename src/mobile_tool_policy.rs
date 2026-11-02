//! Root facade for the Rust-native mobile tool-effect contract.

pub use symthaea_rea_compat::mobile_tool_effects::{
    DerivedToolAnnotations, EffectCeiling, EffectFlag, MOBILE_DEVICE_READ_ONLY_POLICY_ID,
    MOBILE_TOOL_EFFECTS, MobileProviderCapabilityReport, MobileTool, MobileToolEffectEntry,
    ProviderAvailability, ProviderBoundToolAuthorization, ProviderOperationCapability,
    ToolAuthorization, ToolAuthorizationError, ToolEffects, UPSTREAM_REA_COMMIT,
    annotations_from_effects, authorize_mobile_device_observation, authorize_mobile_tool,
    authorize_reported_mobile_tool, mobile_tool_catalog_sha256, mobile_tool_effect_entry,
};
