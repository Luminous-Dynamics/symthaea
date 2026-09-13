//! Research-result contracts with an immutable V1 manifest and stricter V2 evidence boundaries.

pub mod result_v1;
pub use result_v1::*;

pub mod confirmatory_v2;
pub use confirmatory_v2::{
    ConfirmatoryClaimBinding, ConfirmatoryResearchResultV2, ConfirmatoryResultError,
};

pub mod complete_endpoint_v2;
pub use complete_endpoint_v2::{
    CompleteEndpointResearchResultV2, CompleteEndpointResultError,
};
