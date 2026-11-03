//! Direct Rust facade for caller/session-bound mobile invocation authorization.

pub use symthaea_rea_compat::mobile_invocation::{
    AuthorizedMobileInvocation, MobileConsentGrant, MobileDeviceSessionSnapshot,
    MobileInvocationError, MobileInvocationReceipt, MobileInvocationReplayGuard,
    MobileInvocationRequest, MOBILE_INVOCATION_MAX_CLOCK_SKEW_MS,
    MOBILE_INVOCATION_MAX_TTL_MS, MOBILE_INVOCATION_MAX_LOCAL_REPLAY_ENTRIES,
    mobile_invocation_receipt_sha256,
    prepare_mobile_invocation, verify_mobile_invocation_receipt,
};
