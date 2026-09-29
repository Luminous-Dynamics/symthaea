// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed recursive relations between canonical scientific/engineering objects.
//!
//! Relations are semantic statements, not authority grants. The Mycelix DKG may
//! contain recursive or cyclic knowledge; qualification projections must select
//! only the relation families and paths they explicitly admit.

use crate::engineering_identity::EngineeringObjectId;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

pub const ENGINEERING_RELATION_SCHEMA: &str = "symthaea.engineering-relation.v1";

/// Semantic family prevents epistemic knowledge relations from silently becoming
/// deterministic engineering dependencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationFamily {
    Composition,
    EngineeringDependency,
    Scientific,
    Epistemic,
}

/// Typed relation vocabulary shared by recursive engineering/scientific graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EngineeringRelationKind {
    PartOf,
    Contains,
    Requires,
    Implements,
    Parameterizes,
    ExecutesWith,
    CompiledBy,
    RunsOn,
    Deploys,
    Executes,
    Observes,
    DependsOn,
    Constrains,
    Formalizes,
    Instantiates,
    Produces,
    Quantifies,
    HasProperty,
    Supports,
    Contradicts,
    DerivedFrom,
    Equivalent,
    Generalizes,
    Specializes,
}

impl EngineeringRelationKind {
    /// Stable wire name used for canonical relation identity.
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::PartOf => "part_of", Self::Contains => "contains",
            Self::Requires => "requires", Self::Implements => "implements",
            Self::Parameterizes => "parameterizes", Self::ExecutesWith => "executes_with",
            Self::CompiledBy => "compiled_by", Self::RunsOn => "runs_on",
            Self::Deploys => "deploys", Self::Executes => "executes",
            Self::Observes => "observes", Self::DependsOn => "depends_on",
            Self::Constrains => "constrains", Self::Formalizes => "formalizes",
            Self::Instantiates => "instantiates", Self::Produces => "produces",
            Self::Quantifies => "quantifies", Self::HasProperty => "has_property",
            Self::Supports => "supports", Self::Contradicts => "contradicts",
            Self::DerivedFrom => "derived_from", Self::Equivalent => "equivalent",
            Self::Generalizes => "generalizes", Self::Specializes => "specializes",
        }
    }

    /// Explicit endpoint ontology. None means validation is deferred to a
    /// higher-level adapter because the relation is intentionally polymorphic.
    pub const fn endpoint_pairs(self) -> Option<&'static [(&'static str, &'static str)]> {
        match self {
            Self::Requires => Some(&[("requirement", "representation")]),
            Self::Implements => Some(&[("representation", "model")]),
            Self::Parameterizes => Some(&[("model", "model_parameters")]),
            Self::ExecutesWith => Some(&[("model", "runtime")]),
            Self::CompiledBy => Some(&[("runtime", "toolchain")]),
            Self::RunsOn => Some(&[("runtime", "accelerator")]),
            Self::Deploys => Some(&[("accelerator", "deployment_artifact")]),
            Self::Executes => Some(&[("deployment_artifact", "execution_context")]),
            Self::Observes => Some(&[("execution_context", "observation"), ("experiment", "observation")]),
            Self::Constrains => Some(&[("equation", "model"), ("theorem", "physical_model"), ("definition", "model")]),
            Self::Formalizes => Some(&[("model", "physical_system"), ("equation", "model")]),
            Self::Instantiates => Some(&[("simulation", "model"), ("experiment", "physical_system")]),
            Self::Produces => Some(&[("simulation", "prediction"), ("process", "material"), ("experiment", "observation")]),
            Self::Quantifies => Some(&[("observation", "uncertainty"), ("observation", "statistics")]),
            Self::HasProperty => Some(&[("material", "property"), ("material", "measured_property"), ("material", "predicted_property")]),
            Self::PartOf | Self::Contains | Self::DependsOn | Self::Supports | Self::Contradicts |
            Self::DerivedFrom | Self::Equivalent | Self::Generalizes | Self::Specializes => None,
        }
    }

    pub const fn family(self) -> RelationFamily {
        match self {
            Self::PartOf | Self::Contains => RelationFamily::Composition,
            Self::Requires
            | Self::Implements
            | Self::Parameterizes
            | Self::ExecutesWith
            | Self::CompiledBy
            | Self::RunsOn
            | Self::Deploys
            | Self::Executes => RelationFamily::EngineeringDependency,
            Self::Observes
            | Self::DependsOn
            | Self::Constrains
            | Self::Formalizes
            | Self::Instantiates
            | Self::Produces
            | Self::Quantifies
            | Self::HasProperty => RelationFamily::Scientific,
            Self::Supports
            | Self::Contradicts
            | Self::DerivedFrom
            | Self::Equivalent
            | Self::Generalizes
            | Self::Specializes => RelationFamily::Epistemic,
        }
    }

    /// Whether this edge may participate in a deterministic qualification DAG.
    ///
    /// Epistemic edges deliberately return false: their recursive/cyclic
    /// semantics belong to the DKG and must not silently become invalidation edges.
    pub const fn admissible_for_qualification(self) -> bool {
        matches!(
            self.family(),
            RelationFamily::Composition | RelationFamily::EngineeringDependency | RelationFamily::Scientific
        )
    }
}

/// A deterministic semantic edge between two canonical objects.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EngineeringRelation {
    pub source: EngineeringObjectId,
    pub target: EngineeringObjectId,
    pub kind: EngineeringRelationKind,
}

impl EngineeringRelation {
    pub fn new(
        source: EngineeringObjectId,
        target: EngineeringObjectId,
        kind: EngineeringRelationKind,
    ) -> Result<Self, RelationError> {
        source.validate().map_err(RelationError::InvalidSource)?;
        target.validate().map_err(RelationError::InvalidTarget)?;

        if source == target {
            return Err(RelationError::SelfRelation);
        }

        if let Some(pairs) = kind.endpoint_pairs() {
            if !pairs.iter().any(|(s, t)| *s == source.object_kind && *t == target.object_kind) {
                return Err(RelationError::InvalidEndpoints {
                    kind: kind.wire_name(),
                    source_kind: source.object_kind.clone(),
                    target_kind: target.object_kind.clone(),
                });
            }
        }

        Ok(Self { source, target, kind })
    }

    pub const fn family(&self) -> RelationFamily {
        self.kind.family()
    }

    pub const fn admissible_for_qualification(&self) -> bool {
        self.kind.admissible_for_qualification()
    }

    /// Length-delimited canonical bytes; derived digests are excluded from identity.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        fn push_field(out: &mut Vec<u8>, value: &[u8]) {
            out.extend_from_slice(value.len().to_string().as_bytes());
            out.push(b':');
            out.extend_from_slice(value);
        }

        let mut out = Vec::new();
        push_field(&mut out, ENGINEERING_RELATION_SCHEMA.as_bytes());
        push_field(&mut out, self.source.identity_digest().as_bytes());
        push_field(&mut out, self.target.identity_digest().as_bytes());
        push_field(&mut out, self.kind.wire_name().as_bytes());
        out
    }

    pub fn relation_digest(&self) -> String {
        let digest = Sha256::digest(self.canonical_bytes());
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelationError {
    InvalidSource(crate::engineering_identity::IdentityError),
    InvalidTarget(crate::engineering_identity::IdentityError),
    SelfRelation,
    InvalidEndpoints { kind: &'static str, source_kind: String, target_kind: String },
}

impl fmt::Display for RelationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSource(err) => write!(f, "invalid relation source: {err}"),
            Self::InvalidTarget(err) => write!(f, "invalid relation target: {err}"),
            Self::SelfRelation => write!(f, "relation source and target must differ"),
            Self::InvalidEndpoints { kind, source_kind, target_kind } =>
                write!(f, "invalid endpoints for {kind}: {source_kind} -> {target_kind}"),
        }
    }
}

impl std::error::Error for RelationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EngineeringObjectId;

    const A: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn object(kind: &str, id: &str, digest: &str) -> EngineeringObjectId {
        EngineeringObjectId::new("symthaea", kind, id, "1", digest).unwrap()
    }

    fn relation(kind: EngineeringRelationKind) -> EngineeringRelation {
        EngineeringRelation::new(
            object("simulation", "model", A),
            object("prediction", "candidate", B),
            kind,
        )
        .unwrap()
    }

    #[test]
    fn relation_digest_is_deterministic() {
        assert_eq!(
            relation(EngineeringRelationKind::Produces).relation_digest(),
            relation(EngineeringRelationKind::Produces).relation_digest()
        );
    }

    #[test]
    fn endpoint_mutation_changes_relation_digest() {
        let base = relation(EngineeringRelationKind::Produces);
        let changed = EngineeringRelation::new(
            object("physical-model", "other-model", A),
            object("material", "candidate", B),
            EngineeringRelationKind::Produces,
        )
        .unwrap();
        assert_ne!(base.relation_digest(), changed.relation_digest());
    }

    #[test]
    fn relation_kind_mutation_changes_relation_digest() {
        let produces = relation(EngineeringRelationKind::Produces);
        let supports = relation(EngineeringRelationKind::Supports);
        assert_ne!(produces.relation_digest(), supports.relation_digest());
    }

    #[test]
    fn epistemic_edges_are_not_qualification_edges() {
        assert!(!EngineeringRelationKind::Supports.admissible_for_qualification());
        assert!(!EngineeringRelationKind::Contradicts.admissible_for_qualification());
        assert!(EngineeringRelationKind::Produces.admissible_for_qualification());
        assert!(EngineeringRelationKind::Requires.admissible_for_qualification());
    }

    #[test]
    fn self_relations_are_rejected() {
        let object = object("model", "same", A);
        let err = EngineeringRelation::new(
            object.clone(),
            object,
            EngineeringRelationKind::DependsOn,
        )
        .unwrap_err();
        assert_eq!(err, RelationError::SelfRelation);
    }

    #[test]
    fn invalid_endpoints_are_rejected() {
        let err = EngineeringRelation::new(
            object("material", "m", A),
            object("runtime", "r", B),
            EngineeringRelationKind::Formalizes,
        ).unwrap_err();
        assert!(matches!(err, RelationError::InvalidEndpoints { .. }));
    }

    #[test]
    fn wire_names_are_stable_and_distinct() {
        assert_eq!(EngineeringRelationKind::Produces.wire_name(), "produces");
        assert_eq!(EngineeringRelationKind::Supports.wire_name(), "supports");
        assert_ne!(
            EngineeringRelationKind::Produces.wire_name(),
            EngineeringRelationKind::Supports.wire_name()
        );
    }

    #[test]
    fn serde_round_trip_preserves_relation() {
        let relation = relation(EngineeringRelationKind::Constrains);
        let json = serde_json::to_string(&relation).unwrap();
        let restored: EngineeringRelation = serde_json::from_str(&json).unwrap();
        assert_eq!(relation, restored);
    }
}
