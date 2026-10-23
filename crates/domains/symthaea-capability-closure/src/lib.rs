// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pure, deterministic contracts for productive-capability closure analysis.
//!
//! This crate is intentionally authority-free. It defines identifiers, routes,
//! closure statuses, canonical problem/report containers, and validation only.
//! It does not execute manufacturing, allocate resources, or yet implement the
//! fixed-point reachability theorem tracked by CIV-BOOT-002B.

#![deny(unsafe_code)]

use serde::{de, Deserialize, Deserializer, Serialize};
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ValidationError {
    InvalidId { kind: &'static str, reason: &'static str },
    DuplicateCapability { role: &'static str, id: CapabilityId },
    EmptyRouteRequirements { route: RouteId },
    DuplicateRouteRequirement { route: RouteId, capability: CapabilityId },
    DuplicateRouteId { id: RouteId },
    DuplicateTargetReport { target: CapabilityId },
    DuplicateImportLeverage { import: CapabilityId },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidId { kind, reason } => write!(f, "invalid {kind}: {reason}"),
            Self::DuplicateCapability { role, id } => {
                write!(f, "duplicate capability in {role}: {id}")
            }
            Self::EmptyRouteRequirements { route } => {
                write!(f, "route {route} must declare at least one requirement")
            }
            Self::DuplicateRouteRequirement { route, capability } => {
                write!(f, "route {route} repeats required capability {capability}")
            }
            Self::DuplicateRouteId { id } => write!(f, "duplicate route id: {id}"),
            Self::DuplicateTargetReport { target } => {
                write!(f, "duplicate target report: {target}")
            }
            Self::DuplicateImportLeverage { import } => {
                write!(f, "duplicate import leverage entry: {import}")
            }
        }
    }
}

impl std::error::Error for ValidationError {}

fn validate_id(kind: &'static str, value: &str) -> Result<(), ValidationError> {
    if value.is_empty() {
        return Err(ValidationError::InvalidId {
            kind,
            reason: "must not be empty",
        });
    }
    if value.trim() != value {
        return Err(ValidationError::InvalidId {
            kind,
            reason: "leading or trailing whitespace is forbidden",
        });
    }
    if value.chars().any(char::is_control) {
        return Err(ValidationError::InvalidId {
            kind,
            reason: "control characters are forbidden",
        });
    }
    Ok(())
}

macro_rules! typed_id {
    ($name:ident, $kind:literal) => {
        #[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, ValidationError> {
                let value = value.into();
                validate_id($kind, &value)?;
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }

        impl TryFrom<String> for $name {
            type Error = ValidationError;

            fn try_from(value: String) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        impl From<$name> for String {
            fn from(value: $name) -> Self {
                value.0
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Self::new(value).map_err(de::Error::custom)
            }
        }
    };
}

typed_id!(CapabilityId, "capability id");
typed_id!(RouteId, "route id");

/// One AND-set route for producing or providing a capability.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RouteV1 {
    pub id: RouteId,
    pub output: CapabilityId,
    pub required_capabilities: Vec<CapabilityId>,
}

impl RouteV1 {
    pub fn new(
        id: RouteId,
        output: CapabilityId,
        mut required_capabilities: Vec<CapabilityId>,
    ) -> Result<Self, ValidationError> {
        if required_capabilities.is_empty() {
            return Err(ValidationError::EmptyRouteRequirements { route: id });
        }
        required_capabilities.sort();
        for pair in required_capabilities.windows(2) {
            if pair[0] == pair[1] {
                return Err(ValidationError::DuplicateRouteRequirement {
                    route: id,
                    capability: pair[0].clone(),
                });
            }
        }
        Ok(Self {
            id,
            output,
            required_capabilities,
        })
    }
}

#[derive(Deserialize)]
struct RouteV1Wire {
    id: RouteId,
    output: CapabilityId,
    required_capabilities: Vec<CapabilityId>,
}

impl<'de> Deserialize<'de> for RouteV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RouteV1Wire::deserialize(deserializer)?;
        Self::new(wire.id, wire.output, wire.required_capabilities).map_err(de::Error::custom)
    }
}

/// Canonical structural-closure problem input.
///
/// Targets and imports are roles over `CapabilityId`, not separate identity
/// namespaces. This avoids inventing multiple IDs for the same capability.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ClosureProblemV1 {
    pub local_primitives: Vec<CapabilityId>,
    pub imports: Vec<CapabilityId>,
    pub targets: Vec<CapabilityId>,
    pub routes: Vec<RouteV1>,
}

impl ClosureProblemV1 {
    pub fn new(
        mut local_primitives: Vec<CapabilityId>,
        mut imports: Vec<CapabilityId>,
        mut targets: Vec<CapabilityId>,
        mut routes: Vec<RouteV1>,
    ) -> Result<Self, ValidationError> {
        canonicalize_capability_role(&mut local_primitives, "local_primitives")?;
        canonicalize_capability_role(&mut imports, "imports")?;
        canonicalize_capability_role(&mut targets, "targets")?;

        routes.sort_by(|a, b| a.id.cmp(&b.id));
        for pair in routes.windows(2) {
            if pair[0].id == pair[1].id {
                return Err(ValidationError::DuplicateRouteId {
                    id: pair[0].id.clone(),
                });
            }
        }

        Ok(Self {
            local_primitives,
            imports,
            targets,
            routes,
        })
    }
}

#[derive(Deserialize)]
struct ClosureProblemV1Wire {
    local_primitives: Vec<CapabilityId>,
    imports: Vec<CapabilityId>,
    targets: Vec<CapabilityId>,
    routes: Vec<RouteV1>,
}

impl<'de> Deserialize<'de> for ClosureProblemV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ClosureProblemV1Wire::deserialize(deserializer)?;
        Self::new(wire.local_primitives, wire.imports, wire.targets, wire.routes)
            .map_err(de::Error::custom)
    }
}

fn canonicalize_capability_role(
    values: &mut Vec<CapabilityId>,
    role: &'static str,
) -> Result<(), ValidationError> {
    values.sort();
    for pair in values.windows(2) {
        if pair[0] == pair[1] {
            return Err(ValidationError::DuplicateCapability {
                role,
                id: pair[0].clone(),
            });
        }
    }
    Ok(())
}

/// Structural status only. No quantity, throughput, safety, or authority claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum ClosureStatus {
    LocallyClosed,
    ImportDependent,
    Unavailable,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TargetReportV1 {
    pub target: CapabilityId,
    pub status: ClosureStatus,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ImportLeverageV1 {
    pub import: CapabilityId,
    pub targets_lost_if_removed: Vec<CapabilityId>,
    pub capability_count_lost: usize,
}

impl ImportLeverageV1 {
    pub fn new(
        import: CapabilityId,
        mut targets_lost_if_removed: Vec<CapabilityId>,
        capability_count_lost: usize,
    ) -> Result<Self, ValidationError> {
        canonicalize_capability_role(&mut targets_lost_if_removed, "targets_lost_if_removed")?;
        Ok(Self {
            import,
            targets_lost_if_removed,
            capability_count_lost,
        })
    }
}

#[derive(Deserialize)]
struct ImportLeverageV1Wire {
    import: CapabilityId,
    targets_lost_if_removed: Vec<CapabilityId>,
    capability_count_lost: usize,
}

impl<'de> Deserialize<'de> for ImportLeverageV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ImportLeverageV1Wire::deserialize(deserializer)?;
        Self::new(
            wire.import,
            wire.targets_lost_if_removed,
            wire.capability_count_lost,
        )
        .map_err(de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ClosureReportV1 {
    pub locally_reproducible: Vec<CapabilityId>,
    pub operationally_reachable: Vec<CapabilityId>,
    pub targets: Vec<TargetReportV1>,
    pub import_leverage: Vec<ImportLeverageV1>,
}

impl ClosureReportV1 {
    pub fn new(
        mut locally_reproducible: Vec<CapabilityId>,
        mut operationally_reachable: Vec<CapabilityId>,
        mut targets: Vec<TargetReportV1>,
        mut import_leverage: Vec<ImportLeverageV1>,
    ) -> Result<Self, ValidationError> {
        canonicalize_capability_role(&mut locally_reproducible, "locally_reproducible")?;
        canonicalize_capability_role(&mut operationally_reachable, "operationally_reachable")?;

        targets.sort_by(|a, b| a.target.cmp(&b.target));
        for pair in targets.windows(2) {
            if pair[0].target == pair[1].target {
                return Err(ValidationError::DuplicateTargetReport {
                    target: pair[0].target.clone(),
                });
            }
        }

        import_leverage.sort_by(|a, b| a.import.cmp(&b.import));
        for pair in import_leverage.windows(2) {
            if pair[0].import == pair[1].import {
                return Err(ValidationError::DuplicateImportLeverage {
                    import: pair[0].import.clone(),
                });
            }
        }

        Ok(Self {
            locally_reproducible,
            operationally_reachable,
            targets,
            import_leverage,
        })
    }
}

#[derive(Deserialize)]
struct ClosureReportV1Wire {
    locally_reproducible: Vec<CapabilityId>,
    operationally_reachable: Vec<CapabilityId>,
    targets: Vec<TargetReportV1>,
    import_leverage: Vec<ImportLeverageV1>,
}

impl<'de> Deserialize<'de> for ClosureReportV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ClosureReportV1Wire::deserialize(deserializer)?;
        Self::new(
            wire.locally_reproducible,
            wire.operationally_reachable,
            wire.targets,
            wire.import_leverage,
        )
        .map_err(de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(value: &str) -> CapabilityId {
        CapabilityId::new(value).unwrap()
    }

    fn route(value: &str) -> RouteId {
        RouteId::new(value).unwrap()
    }

    #[test]
    fn identifiers_fail_closed() {
        assert!(CapabilityId::new("").is_err());
        assert!(CapabilityId::new(" bearing").is_err());
        assert!(CapabilityId::new("bearing ").is_err());
        assert!(CapabilityId::new("bear\ning").is_err());
        assert_eq!(cap("bearing").as_str(), "bearing");
    }

    #[test]
    fn route_requirements_are_an_unordered_and_set() {
        let r = RouteV1::new(
            route("motor-route"),
            cap("motor"),
            vec![cap("copper"), cap("bearing")],
        )
        .unwrap();
        assert_eq!(r.required_capabilities, vec![cap("bearing"), cap("copper")]);

        assert!(RouteV1::new(route("empty"), cap("x"), vec![]).is_err());
        assert!(RouteV1::new(
            route("dup"),
            cap("x"),
            vec![cap("bearing"), cap("bearing")]
        )
        .is_err());
    }

    #[test]
    fn closure_problem_is_canonical_independent_of_input_order() {
        let a = ClosureProblemV1::new(
            vec![cap("metal"), cap("copper")],
            vec![cap("controller"), cap("bearing")],
            vec![cap("machine"), cap("motor")],
            vec![
                RouteV1::new(
                    route("motor-route"),
                    cap("motor"),
                    vec![cap("copper"), cap("bearing")],
                )
                .unwrap(),
                RouteV1::new(
                    route("machine-route"),
                    cap("machine"),
                    vec![cap("motor"), cap("metal")],
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let b = ClosureProblemV1::new(
            vec![cap("copper"), cap("metal")],
            vec![cap("bearing"), cap("controller")],
            vec![cap("motor"), cap("machine")],
            vec![
                RouteV1::new(
                    route("machine-route"),
                    cap("machine"),
                    vec![cap("metal"), cap("motor")],
                )
                .unwrap(),
                RouteV1::new(
                    route("motor-route"),
                    cap("motor"),
                    vec![cap("bearing"), cap("copper")],
                )
                .unwrap(),
            ],
        )
        .unwrap();
        assert_eq!(a, b);
        assert_eq!(serde_json::to_string(&a).unwrap(), serde_json::to_string(&b).unwrap());
    }

    #[test]
    fn duplicates_fail_closed_instead_of_being_silently_deduplicated() {
        assert!(ClosureProblemV1::new(
            vec![cap("metal"), cap("metal")],
            vec![],
            vec![],
            vec![]
        )
        .is_err());

        let r1 = RouteV1::new(route("same"), cap("a"), vec![cap("x")]).unwrap();
        let r2 = RouteV1::new(route("same"), cap("b"), vec![cap("y")]).unwrap();
        assert!(ClosureProblemV1::new(vec![], vec![], vec![], vec![r1, r2]).is_err());
    }

    #[test]
    fn deserialization_cannot_bypass_validation_or_canonicalization() {
        let bad_id = r#"{"local_primitives":[" bad"],"imports":[],"targets":[],"routes":[]}"#;
        assert!(serde_json::from_str::<ClosureProblemV1>(bad_id).is_err());

        let duplicate_requirement = r#"{"id":"r","output":"x","required_capabilities":["a","a"]}"#;
        assert!(serde_json::from_str::<RouteV1>(duplicate_requirement).is_err());

        let unordered = r#"{"local_primitives":["z","a"],"imports":[],"targets":[],"routes":[]}"#;
        let decoded: ClosureProblemV1 = serde_json::from_str(unordered).unwrap();
        assert_eq!(decoded.local_primitives, vec![cap("a"), cap("z")]);
    }

    #[test]
    fn report_order_is_canonical() {
        let report = ClosureReportV1::new(
            vec![cap("frame"), cap("copper")],
            vec![cap("motor"), cap("frame"), cap("copper")],
            vec![
                TargetReportV1 {
                    target: cap("motor"),
                    status: ClosureStatus::ImportDependent,
                },
                TargetReportV1 {
                    target: cap("frame"),
                    status: ClosureStatus::LocallyClosed,
                },
            ],
            vec![ImportLeverageV1::new(
                cap("bearing"),
                vec![cap("motor"), cap("machine")],
                2,
            )
            .unwrap()],
        )
        .unwrap();

        assert_eq!(report.locally_reproducible, vec![cap("copper"), cap("frame")]);
        assert_eq!(report.targets[0].target, cap("frame"));
        assert_eq!(
            report.import_leverage[0].targets_lost_if_removed,
            vec![cap("machine"), cap("motor")]
        );
    }
}
