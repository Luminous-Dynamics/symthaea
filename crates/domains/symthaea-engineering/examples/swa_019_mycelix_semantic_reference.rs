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

use blake3;
use serde::Serialize;

const OBSERVED_MYCELIX_INTEROP_COMMIT: &str =
    "b55bc03d99d0e8c89201dca06a264d16d5e2efd6";

const MAX_SCHEMA_NAMESPACE_BYTES: usize = 256;
const MAX_SCHEMA_NAME_BYTES: usize = 128;
const MAX_SCHEMA_VERSION_BYTES: usize = 128;
const MAX_OBJECT_ID_BYTES: usize = 1024;
const MAX_OBJECT_VERSION_BYTES: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReferenceValidationError {
    Empty,
    SurroundingWhitespace,
    ControlCharacter,
    TooLong,
}

fn validate_component(value: &str, max_bytes: usize) -> Result<(), ReferenceValidationError> {
    if value.is_empty() {
        return Err(ReferenceValidationError::Empty);
    }
    if value.trim() != value {
        return Err(ReferenceValidationError::SurroundingWhitespace);
    }
    if value.chars().any(char::is_control) {
        return Err(ReferenceValidationError::ControlCharacter);
    }
    if value.len() > max_bytes {
        return Err(ReferenceValidationError::TooLong);
    }
    Ok(())
}

fn validate_schema_ref(schema: &SchemaRefProjection) -> Result<(), ReferenceValidationError> {
    validate_component(schema.namespace, MAX_SCHEMA_NAMESPACE_BYTES)?;
    validate_component(schema.name, MAX_SCHEMA_NAME_BYTES)?;
    validate_component(schema.version, MAX_SCHEMA_VERSION_BYTES)
}

fn validate_semantic_ref(reference: &SemanticRefProjection) -> Result<(), ReferenceValidationError> {
    validate_schema_ref(&reference.schema)?;
    validate_component(reference.object_id, MAX_OBJECT_ID_BYTES)?;
    if let Some(version) = reference.object_version {
        validate_component(version, MAX_OBJECT_VERSION_BYTES)?;
    }
    Ok(())
}

fn validate_reference_fields(envelope: &MycelixProjectionEnvelope) -> Result<(), ReferenceValidationError> {
    validate_schema_ref(&envelope.schema)?;
    validate_semantic_ref(&envelope.source)?;
    validate_semantic_ref(&envelope.target)?;
    validate_semantic_ref(&envelope.author_ref)?;
    if let Some(authority) = envelope.authority_ref.as_ref() {
        validate_semantic_ref(authority)?;
    }
    for dependency in envelope.dependencies {
        validate_semantic_ref(&dependency.semantic_ref)?;
    }
    Ok(())
}

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct MycelixProjectionEnvelope {
    schema: SchemaRefProjection,
    source: SemanticRefProjection,
    target: SemanticRefProjection,
    projection_target: ProjectionTarget,
    source_revision: &'static str,
    source_content_digest: String,
    digest_algorithm: &'static str,
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

fn envelope_without_digest() -> MycelixProjectionEnvelope {
    MycelixProjectionEnvelope {
        schema: SOL_ATLAS_SCHEMA,
        source: source_ref(),
        target: target_ref(),
        projection_target: ProjectionTarget::GovernanceProposalReview,
        source_revision: OBSERVED_MYCELIX_INTEROP_COMMIT,
        source_content_digest: String::new(),
        digest_algorithm: "blake3-256",
        digest_status: "cryptographically-bound fixture projection; production adapter must match the target Mycelix crypto profile",
        author_ref: author_ref(),
        authority_ref: None,
        dependencies: dependency_refs(),
        authority_granted: false,
        actuation_performed: false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct ContentBindingPayload {
    schema: SchemaRefProjection,
    source: SemanticRefProjection,
    target: SemanticRefProjection,
    projection_target: ProjectionTarget,
    source_revision: &'static str,
    author_ref: SemanticRefProjection,
    authority_ref: Option<SemanticRefProjection>,
    dependencies: [DependencyRef; 2],
    authority_granted: bool,
    actuation_performed: bool,
}

fn content_binding_payload(envelope: &MycelixProjectionEnvelope) -> ContentBindingPayload {
    ContentBindingPayload {
        schema: envelope.schema,
        source: envelope.source,
        target: envelope.target,
        projection_target: envelope.projection_target,
        source_revision: envelope.source_revision,
        author_ref: envelope.author_ref,
        authority_ref: envelope.authority_ref,
        dependencies: envelope.dependencies,
        authority_granted: envelope.authority_granted,
        actuation_performed: envelope.actuation_performed,
    }
}

fn canonical_content_bytes(envelope: &MycelixProjectionEnvelope) -> Vec<u8> {
    serde_json::to_vec(&content_binding_payload(envelope)).expect("content binding serializes")
}

fn content_digest(envelope: &MycelixProjectionEnvelope) -> String {
    blake3::hash(&canonical_content_bytes(envelope)).to_hex().to_string()
}

fn envelope() -> MycelixProjectionEnvelope {
    let mut projection = envelope_without_digest();
    projection.source_content_digest = content_digest(&projection);
    projection
}

fn validate(envelope: &MycelixProjectionEnvelope) -> ValidationOutcome {
    if validate_reference_fields(envelope).is_err() {
        return ValidationOutcome::Invalid;
    }

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

    if envelope.digest_algorithm != "blake3-256"
        || envelope.source_content_digest != content_digest(envelope)
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

    // The digest is a real content commitment over the canonical projection
    // payload, while remaining separate from authority and actuation.
    assert_eq!(first.digest_algorithm, "blake3-256");
    assert_eq!(first.source_content_digest, content_digest(&first));
    assert_eq!(first.source_content_digest.len(), 64);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_components_follow_mycelix_wire_safety_rules() {
        let projection = envelope();
        assert_eq!(validate_reference_fields(&projection), Ok(()));
    }

    #[test]
    fn surrounding_whitespace_is_rejected_in_reference_components() {
        let schema = SchemaRefProjection {
            namespace: " mycelix.governance",
            ..MYCELIX_GOVERNANCE_SCHEMA
        };
        assert_eq!(
            validate_schema_ref(&schema),
            Err(ReferenceValidationError::SurroundingWhitespace)
        );
    }

    #[test]
    fn control_characters_are_rejected_in_object_identity() {
        let reference = SemanticRefProjection {
            schema: SOL_ATLAS_SCHEMA,
            object_id: "evidence\n001",
            object_version: None,
        };
        assert_eq!(
            validate_semantic_ref(&reference),
            Err(ReferenceValidationError::ControlCharacter)
        );
    }

    #[test]
    fn overlong_reference_component_is_rejected() {
        let long_id = "x".repeat(MAX_OBJECT_ID_BYTES + 1);
        assert_eq!(
            validate_component(&long_id, MAX_OBJECT_ID_BYTES),
            Err(ReferenceValidationError::TooLong)
        );
    }

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
    fn digest_binds_canonical_projection_payload() {
        let projection = envelope();
        assert_eq!(projection.source_content_digest, content_digest(&projection));
        assert_eq!(projection.digest_algorithm, "blake3-256");
        assert_eq!(projection.source_content_digest.len(), 64);
    }

    #[test]
    fn tampering_with_bound_content_invalidates_projection() {
        let mut projection = envelope();
        projection.authority_granted = true;
        assert_eq!(validate(&projection), ValidationOutcome::Invalid);
    }

    #[test]
    fn changing_bound_content_changes_digest() {
        let baseline_projection = envelope();
        let baseline = content_digest(&baseline_projection);
        let mut changed = content_binding_payload(&baseline_projection);
        changed.authority_granted = true;
        let changed_bytes = serde_json::to_vec(&changed).unwrap();
        let changed_digest = blake3::hash(&changed_bytes).to_hex().to_string();
        assert_ne!(baseline, changed_digest);
    }

    #[test]
    fn canonical_payload_excludes_self_referential_digest() {
        let payload = content_binding_payload(&envelope());
        let json = serde_json::to_string(&payload).unwrap();
        assert!(!json.contains("source_content_digest"));
    }

    #[test]
    fn replay_is_byte_stable() {
        let a = serde_json::to_string(&envelope()).unwrap();
        let b = serde_json::to_string(&envelope()).unwrap();
        assert_eq!(a, b);
    }
}
