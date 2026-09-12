//! Research protocol contracts with an immutable V1 core and typed V2 confirmatory envelope.

pub mod protocol_v1;
pub use protocol_v1::*;

pub mod typed_endpoints;
pub use typed_endpoints::{
    ConfirmatoryDecisionRule, ConfirmatoryEndpointSpec, ConfirmatoryProtocolError,
    ConfirmatoryRunBinding, FrozenConfirmatoryProtocol, MetricSchemaBinding, MetricValueSchema,
};

pub mod canonical_endpoints;
pub use canonical_endpoints::{
    CanonicalConfirmatoryProtocol, CanonicalConfirmatoryRunBinding,
};
