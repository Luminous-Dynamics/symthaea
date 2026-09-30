//! SWA-018: Mycelix/Holochain receipt boundary contract.
//!
//! Maps Sol Atlas lifecycle facts to existing Mycelix domain records without
//! pretending that this fixture writes DHT data. It deliberately keeps
//! attribution, governance authorization, and physical outcomes distinct.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum ProjectionTarget {
    /// Only for a voluntary declaration of dependency usage.
    AttributionUsageReceipt,
    /// A request for collective consideration; not a decision.
    GovernanceProposal,
    /// A recorded cooperative resolution; not a physical execution proof.
    HousingResolution,
    /// A maintenance report/workflow record; not intervention authorization.
    MaintenanceRequest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum ValidationOutcome {
    Valid,
    Invalid,
    UnresolvedDependencies,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Disclosure {
    Public,
    CommonsScoped,
    PrivateReferenceOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SourceRef {
    namespace: String,
    object_id: String,
    revision: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ReceiptProjection {
    projection_id: String,
    target: ProjectionTarget,
    source: SourceRef,
    author_agent_ref: String,
    subject_ref: String,
    content_digest_algorithm: String,
    content_digest: String,
    disclosure: Disclosure,
    required_dependencies: Vec<String>,
    authority_record_ref: Option<String>,
    physical_outcome_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ContractError {
    EmptyIdentity,
    MissingDependency,
    DuplicateDependency,
    InvalidDigestDeclaration,
    DomainMismatch,
    AuthorityConflatedWithEvidence,
}

impl ReceiptProjection {
    fn validate_contract(&self) -> Result<(), ContractError> {
        if self.projection_id.is_empty()
            || self.source.namespace.is_empty()
            || self.source.object_id.is_empty()
            || self.author_agent_ref.is_empty()
            || self.subject_ref.is_empty()
        {
            return Err(ContractError::EmptyIdentity);
        }
        if self.content_digest_algorithm.is_empty() || self.content_digest.is_empty() {
            return Err(ContractError::InvalidDigestDeclaration);
        }
        if self.required_dependencies.iter().any(String::is_empty) {
            return Err(ContractError::MissingDependency);
        }
        let mut dependencies = self.required_dependencies.clone();
        dependencies.sort();
        dependencies.dedup();
        if dependencies.len() != self.required_dependencies.len() {
            return Err(ContractError::DuplicateDependency);
        }
        if self.authority_record_ref.as_deref() == Some(self.projection_id.as_str()) {
            return Err(ContractError::AuthorityConflatedWithEvidence);
        }
        // A prediction/evidence source is not itself a usage declaration,
        // governance proposal, housing resolution, or maintenance request.
        if self.source.namespace == "sol-atlas/evidence"
            && self.target == ProjectionTarget::AttributionUsageReceipt
        {
            return Err(ContractError::DomainMismatch);
        }
        Ok(())
    }

    /// Missing DHT dependencies remain unresolved; never coerce absence to pass.
    fn validation_outcome(&self, resolved_dependencies: &[String]) -> ValidationOutcome {
        if self.validate_contract().is_err() {
            return ValidationOutcome::Invalid;
        }
        if self.required_dependencies.iter().any(|dependency|
            !resolved_dependencies.contains(dependency))
        {
            return ValidationOutcome::UnresolvedDependencies;
        }
        ValidationOutcome::Valid
    }
}

fn fixture() -> ReceiptProjection {
    ReceiptProjection {
        projection_id: "sol-atlas-projection-001".into(),
        target: ProjectionTarget::GovernanceProposal,
        source: SourceRef {
            namespace: "sol-atlas/intervention-review".into(),
            object_id: "heat-pump-retrofit".into(),
            revision: 2,
        },
        author_agent_ref: "agent-pubkey-ref:author-001".into(),
        subject_ref: "building-ref:building-001".into(),
        content_digest_algorithm: "sha256".into(),
        content_digest: "fixture-digest-not-a-computed-commitment".into(),
        disclosure: Disclosure::CommonsScoped,
        required_dependencies: vec!["evidence:v2-comfort".into(), "evidence:v2-energy".into()],
        authority_record_ref: None,
        physical_outcome_ref: None,
    }
}

fn main() {
    let projection = fixture();
    assert_eq!(projection.validate_contract(), Ok(()));
    assert_eq!(projection.validation_outcome(&[]), ValidationOutcome::UnresolvedDependencies);
    println!("target={:?}; outcome={:?}; physical_outcome={:?}",
        projection.target, projection.validation_outcome(&projection.required_dependencies),
        projection.physical_outcome_ref);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn governance_proposal_is_a_review_request_not_authorization() {
        let p = fixture();
        assert_eq!(p.target, ProjectionTarget::GovernanceProposal);
        assert!(p.authority_record_ref.is_none());
        assert!(p.physical_outcome_ref.is_none());
    }

    #[test]
    fn missing_dependency_is_unresolved_not_valid() {
        assert_eq!(fixture().validation_outcome(&[]), ValidationOutcome::UnresolvedDependencies);
    }

    #[test]
    fn complete_declared_dependencies_allow_contract_validity_only() {
        let p = fixture();
        assert_eq!(p.validation_outcome(&p.required_dependencies), ValidationOutcome::Valid);
        assert!(p.authority_record_ref.is_none());
    }

    #[test]
    fn evidence_cannot_be_misrepresented_as_usage_receipt() {
        let mut p = fixture();
        p.target = ProjectionTarget::AttributionUsageReceipt;
        p.source.namespace = "sol-atlas/evidence".into();
        assert_eq!(p.validate_contract(), Err(ContractError::DomainMismatch));
    }

    #[test]
    fn duplicate_dependency_is_rejected() {
        let mut p = fixture();
        p.required_dependencies.push(p.required_dependencies[0].clone());
        assert_eq!(p.validate_contract(), Err(ContractError::DuplicateDependency));
    }

    #[test]
    fn evidence_and_authority_references_cannot_alias() {
        let mut p = fixture();
        p.authority_record_ref = Some(p.projection_id.clone());
        assert_eq!(p.validate_contract(), Err(ContractError::AuthorityConflatedWithEvidence));
    }

    #[test]
    fn identity_and_digest_are_required() {
        let mut p = fixture();
        p.author_agent_ref.clear();
        assert_eq!(p.validate_contract(), Err(ContractError::EmptyIdentity));
        let mut p = fixture();
        p.content_digest.clear();
        assert_eq!(p.validate_contract(), Err(ContractError::InvalidDigestDeclaration));
    }

    #[test]
    fn outcome_is_not_physical_execution() {
        let p = fixture();
        assert_eq!(p.validation_outcome(&p.required_dependencies), ValidationOutcome::Valid);
        assert!(p.physical_outcome_ref.is_none());
    }
}
