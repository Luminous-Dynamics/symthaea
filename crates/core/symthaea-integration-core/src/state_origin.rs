// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed provenance for desired-state assertions.
//!
//! v0.1 keeps the `StateAssertion` wire shape stable while promoting desired
//! origin out of adapter-private convention. The canonical wire representation
//! lives under one core-owned reserved attribute and is exposed only through
//! typed helpers. Missing legacy metadata is surfaced as `Unspecified`; unknown
//! values and desired-origin metadata on observed assertions fail closed.
//!
//! Origin describes how a target value came to exist. It does not confer
//! operational authority and must not be used as an authorization signal.

use crate::{StateAssertion, StateRole, StateSnapshot};
use serde::{Deserialize, Serialize};

pub const DESIRED_STATE_ORIGIN_ATTRIBUTE: &str = "symthaea.desired_origin";

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum DesiredStateOrigin {
    /// Legacy or source data that does not yet identify how the target arose.
    Unspecified,
    /// Explicit declarative/operator intent such as a Kubernetes spec, Git,
    /// Nix, Terraform/OpenTofu, Helm, or another authored configuration source.
    Declared,
    /// A default supplied by an API/schema/runtime rather than explicitly set
    /// by the operator.
    Defaulted,
    /// A value mechanically derived by the control plane from declared state,
    /// such as Kubernetes metadata generation.
    SystemDerived,
    /// A controller-computed target, such as DaemonSet desiredNumberScheduled.
    ControllerDerived,
    /// A target imposed by an explicit organizational/SLO/governance policy.
    PolicyDerived,
    /// A model-generated recommendation or inferred target. This never confers
    /// authority by itself.
    Inferred,
}

impl DesiredStateOrigin {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unspecified => "unspecified",
            Self::Declared => "declared",
            Self::Defaulted => "defaulted",
            Self::SystemDerived => "system_derived",
            Self::ControllerDerived => "controller_derived",
            Self::PolicyDerived => "policy_derived",
            Self::Inferred => "inferred",
        }
    }

    pub fn parse(value: &str) -> Result<Self, StateOriginError> {
        match value {
            "unspecified" => Ok(Self::Unspecified),
            "declared" => Ok(Self::Declared),
            "defaulted" => Ok(Self::Defaulted),
            "system_derived" => Ok(Self::SystemDerived),
            "controller_derived" => Ok(Self::ControllerDerived),
            "policy_derived" => Ok(Self::PolicyDerived),
            "inferred" => Ok(Self::Inferred),
            other => Err(StateOriginError::UnknownDesiredOrigin(other.to_string())),
        }
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct DesiredStateOriginEvidence {
    pub origin: DesiredStateOrigin,
    /// False only when legacy/missing metadata was conservatively mapped to
    /// `Unspecified`; true when the producer explicitly supplied the origin.
    pub explicit: bool,
}

impl StateAssertion {
    /// Read desired-state provenance using the core-owned typed vocabulary.
    ///
    /// Desired assertions without v0.1 provenance remain usable but are
    /// surfaced as `Unspecified { explicit: false }`. Observed assertions must
    /// not carry desired-state origin metadata.
    pub fn desired_state_origin(
        &self,
    ) -> Result<Option<DesiredStateOriginEvidence>, StateOriginError> {
        let encoded = self.attributes.get(DESIRED_STATE_ORIGIN_ATTRIBUTE);
        match self.role {
            StateRole::Desired => match encoded {
                Some(value) => Ok(Some(DesiredStateOriginEvidence {
                    origin: DesiredStateOrigin::parse(value)?,
                    explicit: true,
                })),
                None => Ok(Some(DesiredStateOriginEvidence {
                    origin: DesiredStateOrigin::Unspecified,
                    explicit: false,
                })),
            },
            StateRole::Observed => match encoded {
                Some(value) => Err(StateOriginError::DesiredOriginOnObservedAssertion {
                    assertion_id: self.assertion_id.clone(),
                    encoded_origin: value.clone(),
                }),
                None => Ok(None),
            },
        }
    }

    /// Attach a typed desired-state origin. The reserved wire key is owned by
    /// integration-core; adapters should not write arbitrary strings there.
    pub fn set_desired_state_origin(
        &mut self,
        origin: DesiredStateOrigin,
    ) -> Result<(), StateOriginError> {
        if self.role != StateRole::Desired {
            return Err(StateOriginError::CannotSetDesiredOriginOnObserved {
                assertion_id: self.assertion_id.clone(),
            });
        }
        self.attributes.insert(
            DESIRED_STATE_ORIGIN_ATTRIBUTE.to_string(),
            origin.as_str().to_string(),
        );
        Ok(())
    }

    pub fn with_desired_state_origin(
        mut self,
        origin: DesiredStateOrigin,
    ) -> Result<Self, StateOriginError> {
        self.set_desired_state_origin(origin)?;
        Ok(self)
    }

    pub fn validate_state_origin(&self) -> Result<(), StateOriginError> {
        self.desired_state_origin().map(|_| ())
    }
}

pub fn validate_state_snapshot_origins(snapshot: &StateSnapshot) -> Result<(), StateOriginError> {
    for assertion in &snapshot.assertions {
        assertion.validate_state_origin()?;
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StateOriginError {
    #[error("unknown desired-state origin `{0}`")]
    UnknownDesiredOrigin(String),
    #[error(
        "observed assertion `{assertion_id}` carries desired-state origin `{encoded_origin}`"
    )]
    DesiredOriginOnObservedAssertion {
        assertion_id: String,
        encoded_origin: String,
    },
    #[error("cannot set desired-state origin on observed assertion `{assertion_id}`")]
    CannotSetDesiredOriginOnObserved { assertion_id: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EntityRef, StateAssertionSource, StateValue};
    use std::collections::BTreeMap;

    fn assertion(role: StateRole) -> StateAssertion {
        StateAssertion {
            assertion_id: "a-1".into(),
            subject: EntityRef::new("k8s:fixture", "k8s_deployment", "dep-1"),
            dimension: "workload.replicas".into(),
            role,
            value: StateValue::Unsigned(3),
            source_confidence: 1.0,
            source: StateAssertionSource {
                integration_id: "fixture".into(),
                collector_id: None,
                tenant: None,
            },
            observed_at_unix_ms: 10,
            valid_from_unix_ms: None,
            valid_until_unix_ms: None,
            evidence_observation_ids: vec![],
            attributes: BTreeMap::new(),
        }
    }

    #[test]
    fn legacy_desired_origin_is_explicitly_unspecified() {
        let evidence = assertion(StateRole::Desired)
            .desired_state_origin()
            .unwrap()
            .unwrap();
        assert_eq!(evidence.origin, DesiredStateOrigin::Unspecified);
        assert!(!evidence.explicit);
    }

    #[test]
    fn controller_derived_origin_round_trips_through_typed_api() {
        let mut assertion = assertion(StateRole::Desired);
        assertion
            .set_desired_state_origin(DesiredStateOrigin::ControllerDerived)
            .unwrap();
        let evidence = assertion.desired_state_origin().unwrap().unwrap();
        assert_eq!(evidence.origin, DesiredStateOrigin::ControllerDerived);
        assert!(evidence.explicit);
        assert_eq!(
            assertion.attributes.get(DESIRED_STATE_ORIGIN_ATTRIBUTE),
            Some(&"controller_derived".to_string())
        );
    }

    #[test]
    fn observed_assertion_rejects_desired_origin_metadata() {
        let mut assertion = assertion(StateRole::Observed);
        assertion.attributes.insert(
            DESIRED_STATE_ORIGIN_ATTRIBUTE.into(),
            "declared".into(),
        );
        assert!(matches!(
            assertion.validate_state_origin(),
            Err(StateOriginError::DesiredOriginOnObservedAssertion { .. })
        ));
    }

    #[test]
    fn unknown_origin_fails_closed() {
        let mut assertion = assertion(StateRole::Desired);
        assertion.attributes.insert(
            DESIRED_STATE_ORIGIN_ATTRIBUTE.into(),
            "probably-from-git".into(),
        );
        assert!(matches!(
            assertion.validate_state_origin(),
            Err(StateOriginError::UnknownDesiredOrigin(_))
        ));
    }

    #[test]
    fn inferred_origin_is_not_an_authority_type() {
        let mut assertion = assertion(StateRole::Desired);
        assertion
            .set_desired_state_origin(DesiredStateOrigin::Inferred)
            .unwrap();
        assert_eq!(
            assertion.desired_state_origin().unwrap().unwrap().origin,
            DesiredStateOrigin::Inferred
        );
        // State provenance has deliberately no authorization/capability field.
    }
}
