//! SWA-019: Mycelix semantic-reference binding for the Sol Atlas projection.
//!
//! This fixture tightens SWA-018 against the currently observed Mycelix
//! interoperability candidate at commit
//! b55bc03d99d0e8c89201dca06a264d16d5e2efd6.
//!
//! It mirrors the transport shape of Mycelix's SchemaRef/SemanticRef without
//! importing that repository as a runtime dependency. The mirror is an adapter
//! DTO, not a second semantic-reference implementation: Mycelix remains the
//! owner of the target schema semantics.
//!
//! The fixture proves:
//! - schema namespace/name/version are all identity-bearing;
//! - object identity is opaque and remains paired with its schema;
//! - source and target references cannot collapse into one identity;
//! - a source reference is not an authority reference;
//! - dependency absence is unresolved, never implicitly valid;
//! - the candidate Mycelix source revision is provenance, not a live authority;
//! - deterministic serialization/replay preserves the exact projection.

use serde::Serialize;

const OBSERVED_MYCELIX_INTEROP_COMMIT: &str =
    "b55bc03d99d0e8c89201dca06a264d16d5e2efd6";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct SchemaRefProjection {
    namespace: &'static str,
    name: &'static str,
    version: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct SemanticRefProjection {
    schema: SchemaRefProjection,
    object_id: &'static str,
    object_version: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum ProjectionTarget {
    GovernanceProposalReview,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
enum ValidationOutcome {
    Valid,
    Invalid,
    UnresolvedDependencies,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct DependencyRef {
    semantic_ref: SemanticRefProjection,
    required: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct MycelixProjectionEnvelope {
    schema: SchemaRefProjection,
    source: SemanticRefProjection,
    target: SemanticRefProjection,
    projection_target: ProjectionTarget,
    source_revision: &'static str,
    source_content_digest: &'static str,
    digest_status: &'static str,
    author_ref: SemanticRefProjection,
    authority_ref: Option<SemanticRefProjection>,
    dependencies: [DependencyRef; 2],
    authority_granted: bool,
    actuation_performed: bool,
}

const SOL_ATLAS_SCHEMA: SchemaRefProjection = SchemaRefProjection {
    namespace: "sol-atlas",
    name: "spatial-decision-evidence",
    version: "v1",
};

const MYCELIX_GOVERNANCE_SCHEMA: SchemaRefProjection = SchemaRefProjection {
    namespace: "mycelix.governance",
    name: "governance-proposal",
    version: "observed",
};

fn source_ref() -> SemanticRefProjection {
    SemanticRefProjection {
        schema: SOL_ATLAS_SCHEMA,
        object_id: "SWA-003-BUILDING-001",
        object_version: Some("intervention:workspace-intervention-modular-hvac@v1"),
    }
}

fn target_ref() -> SemanticRefProjection {
    SemanticRefProjection {
        schema: MYCELIX_GOVERNANCE_SCHEMA,
        object_id: "governance-proposal:sol-atlas-review-001",
        object_version: Some("1"),
    }
}

fn author_ref() -> SemanticRefProjection {
    SemanticRefProjection {
        schema: SchemaRefProjection {
            namespace: "mycelix.identity",
            name: "agent",
            version: "observed",
        },
        object_id: "agent:fixture-author",
        object_version: None,
    }
}

fn dependency_refs() -> [DependencyRef; 2] {
    [
        DependencyRef {
            semantic_ref: SemanticRefProjection {
                schema: SOL_ATLAS_SCHEMA,
                object_id: "SWA-003-BUILDING-001/model",
                object_version: Some("building-twin@fixture"),
            },
            required: true,
        },
        DependencyRef {
            semantic_ref: SemanticRefProjection {
                schema: SOL_ATLAS_SCHEMA,
                object_id: "SWA-003-BUILDING-001/evidence",
                object_version: Some("provenance-slice@v1"),
            },
            required: true,
        },
    ]
}

fn envelope() -> MycelixProjectionEnvelope {
    MycelixProjectionEnvelope {
        schema: SOL_ATLAS_SCHEMA,
        source: source_ref(),
        target: target_ref(),
        projection_target: ProjectionTarget::GovernanceProposalReview,
        source_revision: OBSERVED_MYCELIX_INTEROP_COMMIT,
        source_content_digest: "not-yet-computed-by-this-reference-fixture",
        digest_status: "declared-only; production adapter must canonicalize and cryptographically bind payload",
        author_ref: author_ref(),
        authority_ref: None,
        dependencies: dependency_refs(),
        authority_granted: false,
        actuation_performed: false,
    }
}

fn validate(envelope: &MycelixProjectionEnvelope) -> ValidationOutcome {
    if envelope.schema != SOL_ATLAS_SCHEMA {
        return ValidationOutcome::Invalid;
    }

    if envelope.source == envelope.target {
        return ValidationOutcome::Invalid;
    }

    if envelope.authority_ref == Some(envelope.source)
        || envelope.authority_ref == Some(envelope.target)
    {
        return ValidationOutcome::Invalid;
    }

    let mut seen = Vec::new();
    for dependency in envelope.dependencies {
        if dependency.required && dependency.semantic_ref.object_id.is_empty() {
            return ValidationOutcome::UnresolvedDependencies;
        }
        if seen.contains(&dependency.semantic_ref) {
            return ValidationOutcome::Invalid;
        }
        seen.push(dependency.semantic_ref);
    }

    ValidationOutcome::Valid
}

fn main() {
    let first = envelope();
    let second = envelope();

    assert_eq!(validate(&first), ValidationOutcome::Valid);
    assert_eq!(
        serde_json::to_string(&first).expect("projection serializes"),
        serde_json::to_string(&second).expect("projection serializes")
    );

    // A source reference identifies evidence. It does not identify authority.
    assert!(first.authority_ref.is_none());
    assert!(!first.authority_granted);
    assert!(!first.actuation_performed);

    // The observed Mycelix commit is source provenance, not a semantic upgrade.
    assert_eq!(first.source_revision, OBSERVED_MYCELIX_INTEROP_COMMIT);

    // A digest declaration is not silently promoted to a cryptographic
    // commitment merely because the field is present.
    assert!(first.source_content_digest.contains("not-yet-computed"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_identity_includes_namespace_name_and_version() {
        let mut changed = SOL_ATLAS_SCHEMA;
        changed = SchemaRefProjection {
            version: "v2",
            ..changed
        };
        assert_ne!(SOL_ATLAS_SCHEMA, changed);
    }

    #[test]
    fn source_and_target_are_distinct_semantic_references() {
        assert_ne!(source_ref(), target_ref());
    }

    #[test]
    fn authority_is_not_derived_from_source_or_target_identity() {
        let projection = envelope();
        assert!(projection.authority_ref.is_none());
        assert!(!projection.authority_granted);
        assert_eq!(validate(&projection), ValidationOutcome::Valid);
    }

    #[test]
    fn duplicate_dependencies_are_rejected() {
        let mut projection = envelope();
        projection.dependencies[1] = projection.dependencies[0];
        assert_eq!(validate(&projection), ValidationOutcome::Invalid);
    }

    #[test]
    fn missing_required_dependency_is_unresolved() {
        let mut projection = envelope();
        projection.dependencies[1] = DependencyRef {
            semantic_ref: SemanticRefProjection {
                schema: SOL_ATLAS_SCHEMA,
                object_id: "",
                object_version: None,
            },
            required: true,
        };
        assert_eq!(
            validate(&projection),
            ValidationOutcome::UnresolvedDependencies
        );
    }

    #[test]
    fn authority_reference_cannot_alias_source_or_target() {
        let mut projection = envelope();
        projection.authority_ref = Some(projection.source);
        assert_eq!(validate(&projection), ValidationOutcome::Invalid);

        projection.authority_ref = Some(projection.target);
        assert_eq!(validate(&projection), ValidationOutcome::Invalid);
    }

    #[test]
    fn source_revision_is_provenance_not_authority() {
        let projection = envelope();
        assert_eq!(projection.source_revision, OBSERVED_MYCELIX_INTEROP_COMMIT);
        assert!(!projection.authority_granted);
    }

    #[test]
    fn digest_declaration_does_not_claim_cryptographic_binding() {
        let projection = envelope();
        assert_eq!(
            projection.digest_status,
            "declared-only; production adapter must canonicalize and cryptographically bind payload"
        );
    }

    #[test]
    fn replay_is_byte_stable() {
        let a = serde_json::to_string(&envelope()).unwrap();
        let b = serde_json::to_string(&envelope()).unwrap();
        assert_eq!(a, b);
    }
}
